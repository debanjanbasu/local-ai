#!/usr/bin/env python3
"""Run real coding clients (Codex CLI, Oh My Pi) against `local-ai serve`.

Every client request passes through a recording reverse proxy, so a failing
scenario carries the exact request body, the exact response (SSE included)
and the slice of the server's stderr written while it ran. Scenarios run in
disposable scratch repositories under the output directory; nothing touches
the user's real client configuration: each client gets a scratch `HOME`
(and `CODEX_HOME`), and credential-looking environment variables are
removed from the clients' environment.

The model is loaded by `serve`, so wrap a run in the shared GPU lock and keep
each run bounded:

    lockf -k /tmp/local-ai-gpu.lock tools/client_compat.py \\
        --codex-bin .amp/in/codex/node_modules/.bin/codex
    tools/client_compat.py --list
    tools/client_compat.py --scenarios http-custom-tool,omp-fix-bug

Scenario kinds:

- `omp-*` / `codex-*`: the client itself runs a task (read, edit, tool result,
  final answer; tool-error recovery; a resumed second user turn; SIGINT
  cancellation mid-stream). `task_ok` checks the repository or the answer;
  `protocol_ok` requires that no proxied exchange failed.
- `http-*`: direct protocol probes of request shapes those clients send
  (hosted and custom tools, reasoning summaries, compaction, encrypted
  reasoning replay, disconnect and background cancellation).

Each result names the workstream that owns a failure, so a rerun after a
merge shows which gaps closed. Output: `results.json`, `summary.md`,
`exchanges.jsonl`, `server.log` and per-scenario client logs.

The proxy strips `Accept-Encoding` so recorded bodies are plain text; the
server's zstd response path is therefore not exercised through it.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import http.server
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "ternary-bonsai-2-27b"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
REDACTED_HEADERS = {"authorization", "api-key", "x-api-key", "cookie", "proxy-authorization"}
SECRET_ENV = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
BODY_LIMIT = 400_000


# ---------------------------------------------------------------- recording


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if k.lower() in REDACTED_HEADERS else v) for k, v in headers.items()}


def parse_body(raw: bytes) -> object:
    """JSON when it parses, else (truncated) text."""
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text[:BODY_LIMIT]


def sse_events(text: str) -> list[dict]:
    """The JSON `data:` payloads of an SSE body (non-JSON data is skipped)."""
    events = []
    for block in re.split(r"\r?\n\r?\n", text):
        data = "\n".join(
            line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except ValueError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def exchange_failure(exchange: dict) -> str | None:
    """Why an exchange failed, or None. Covers HTTP errors and SSE failures."""
    status = exchange.get("status")
    body = exchange.get("response_body")
    if status is None:
        return f"no response: {exchange.get('error')}"
    if status >= 400:
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            return f"HTTP {status}: {body['error'].get('message')}"
        return f"HTTP {status}: {str(body)[:500]}"
    if isinstance(body, str):
        for event in sse_events(body):
            kind = event.get("type")
            if kind == "response.failed":
                error = (event.get("response") or {}).get("error")
                return f"response.failed: {error}"
            if kind == "error":
                return f"error event: {event.get('message') or event.get('error') or event}"
            if "error" in event and kind is None:
                return f"error chunk: {event['error']}"
    return None


def response_object(exchange: dict) -> dict | None:
    """The final Responses object of an exchange (JSON or SSE terminal event)."""
    body = exchange.get("response_body")
    if isinstance(body, dict) and body.get("object") == "response":
        return body
    if isinstance(body, str):
        for event in reversed(sse_events(body)):
            if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                return event.get("response")
    return None


def output_text(response: dict | None) -> str:
    if not response:
        return ""
    texts = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    return "".join(texts)


def function_outputs(request_body: object) -> list[dict]:
    if not isinstance(request_body, dict) or not isinstance(request_body.get("input"), list):
        return []
    return [i for i in request_body["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]


class Recorder:
    """A streaming reverse proxy that records every exchange it forwards."""

    def __init__(self, upstream_port: int, log_path: Path, timeout: float = 7200.0):
        self.upstream_port = upstream_port
        self.log_path = log_path
        self.timeout = timeout
        self.lock = threading.Lock()
        self.exchanges: list[dict] = []
        self.scenario = "-"
        self.first_chunk = threading.Event()
        self.server: http.server.ThreadingHTTPServer | None = None

    def begin(self, scenario: str) -> int:
        with self.lock:
            self.scenario = scenario
            self.first_chunk.clear()
            return len(self.exchanges)

    def since(self, index: int) -> list[dict]:
        with self.lock:
            return list(self.exchanges[index:])

    def _record(self, exchange: dict) -> None:
        with self.lock, self.log_path.open("a", encoding="utf-8") as log:
            exchange["seq"] = len(self.exchanges)
            self.exchanges.append(exchange)
            log.write(json.dumps(exchange) + "\n")

    def start(self) -> int:
        recorder = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence default stderr logging
                pass

            def _body(self) -> bytes:
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    chunks = []
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        chunks.append(self.rfile.read(size))
                        self.rfile.readline()
                    return b"".join(chunks)
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _forward(self) -> None:
                body = self._body()
                exchange = {
                    "scenario": recorder.scenario,
                    "method": self.command,
                    "path": self.path,
                    "request_headers": redact_headers(dict(self.headers.items())),
                    "request_body": parse_body(body),
                    "started": time.time(),
                }
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in HOP_BY_HOP | {"host", "content-length", "accept-encoding"}
                }
                received = bytearray()
                recorded = []

                def finish() -> None:
                    if not recorded:
                        recorded.append(True)
                        exchange["duration_seconds"] = time.time() - exchange["started"]
                        exchange["response_body"] = parse_body(bytes(received))
                        recorder._record(exchange)

                upstream = http.client.HTTPConnection("127.0.0.1", recorder.upstream_port, timeout=recorder.timeout)
                try:
                    upstream.request(self.command, self.path, body or None, headers)
                    response = upstream.getresponse()
                    exchange["status"] = response.status
                    exchange["response_headers"] = dict(response.getheaders())
                    exchange["first_byte_seconds"] = time.time() - exchange["started"]
                    self.send_response_only(response.status, response.reason)
                    for key, value in response.getheaders():
                        if key.lower() not in HOP_BY_HOP | {"content-length"}:
                            self.send_header(key, value)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        received.extend(chunk)
                        try:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                        except OSError:
                            exchange["client_disconnected"] = True
                            break
                        recorder.first_chunk.set()
                    # Record before the terminating chunk so a client that has
                    # read the whole response always finds its exchange logged.
                    finish()
                    if not exchange.get("client_disconnected"):
                        try:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                        except OSError:
                            exchange["client_disconnected"] = True
                except OSError as error:
                    exchange["error"] = repr(error)
                    try:
                        self.send_error(502, "upstream failed")
                    except OSError:
                        pass
                finally:
                    upstream.close()
                    finish()

            do_GET = do_POST = do_DELETE = _forward

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                if not isinstance(sys.exc_info()[1], ConnectionError):
                    super().handle_error(request, client_address)

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server.server_address[1]

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()


# ---------------------------------------------------------------- server


class LocalServer:
    """`local-ai serve`, ready when it prints its listening line on stderr."""

    def __init__(self, binary: Path, port: int, log_path: Path, args: list[str]):
        self.binary = binary
        self.port = port
        self.log_path = log_path
        self.args = args
        self.process: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.lock = threading.Lock()

    def start(self, timeout: float = 900.0) -> None:
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} is already in use")
        command = [str(self.binary), "serve", "--port", str(self.port), *self.args]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as log:
            log.write(f"$ {' '.join(command)}\n")
        self.process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        threading.Thread(target=self._pump, daemon=True).start()
        # The listening line is the server's only readiness signal; waiting on
        # it is event-driven (no /health polling).
        if not self.ready.wait(timeout) or self.process.poll() is not None:
            self.stop()
            raise RuntimeError(f"server did not start; see {self.log_path}")

    def _pump(self) -> None:
        assert self.process and self.process.stderr
        for line in self.process.stderr:
            with self.lock, self.log_path.open("a") as log:
                log.write(line)
            if "server listening on http://" in line:
                self.ready.set()
        self.ready.set()

    def offset(self) -> int:
        with self.lock:
            return self.log_path.stat().st_size if self.log_path.exists() else 0

    def since(self, offset: int) -> str:
        with self.lock, self.log_path.open("rb") as log:
            log.seek(offset)
            return log.read().decode("utf-8", errors="replace")[-20000:]

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------- clients


def client_env(home: Path, **extra: str) -> dict[str, str]:
    """Environment for a client: scratch HOME, no credential-looking vars."""
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OMP_PROFILE", "PI_CODING_AGENT_DIR", "CODEX_HOME"):
        env.pop(key, None)
    env["HOME"] = str(home)
    env.update(extra)
    return env


OMP_TOOLS = "read,bash,edit,write,grep,glob"
CODEX_COMPAT_FEATURES = ["multi_agent", "goals", "view_image", "skill_search", "apps", "plugins"]


def codex_config(proxy_port: int, effort: str, compat: bool) -> str:
    """config.toml for a scratch CODEX_HOME. `compat` keeps only function tools."""
    lines = [
        f'model = "{MODEL_ID}"',
        'model_provider = "localai"',
        f'model_reasoning_effort = "{effort}"',
        'model_reasoning_summary = "none"',
        "model_context_window = 65536",
        'approval_policy = "never"',
        'sandbox_mode = "workspace-write"',
    ]
    tail = [
        "",
        "[sandbox_workspace_write]",
        "network_access = false",
        "exclude_tmpdir_env_var = true",
        "exclude_slash_tmp = true",
    ]
    if compat:
        lines.append('web_search = "disabled"')
        lines.append("")
        lines.append("[features]")
        lines.extend(f"{name} = false" for name in CODEX_COMPAT_FEATURES)
    lines += tail
    lines += [
        "",
        "[model_providers.localai]",
        'name = "local-ai"',
        f'base_url = "http://127.0.0.1:{proxy_port}/v1"',
        'wire_api = "responses"',
        "",
    ]
    return "\n".join(lines)


def omp_models_yml(proxy_port: int, context_window: int = 65536) -> str:
    """The engine README's Oh My Pi provider, pointed at the proxy."""
    return f"""providers:
  local-ai:
    baseUrl: http://127.0.0.1:{proxy_port}/v1
    auth: none
    api: openai-responses
    compat:
      supportsStrictMode: false
      supportsReasoningSummary: false
      includeEncryptedReasoning: false
      reasoningDisableMode: none-effort
      supportsForcedToolChoice: false
      reasoningEffortMap: {{minimal: xhigh, low: xhigh, medium: xhigh, high: xhigh, max: xhigh}}
    models:
      - id: {MODEL_ID}
        name: Ternary Bonsai 2 27B (local-ai)
        reasoning: true
        thinking: {{mode: effort, efforts: [xhigh], defaultLevel: xhigh, requiresEffort: false}}
        input: [text]
        contextWindow: {context_window}
        maxTokens: 8192
        cost: {{input: 0, output: 0, cacheRead: 0, cacheWrite: 0}}
"""


@dataclass
class Context:
    out: Path
    recorder: Recorder
    proxy_port: int
    server_port: int
    server: LocalServer | None
    codex_bin: str | None
    omp_bin: str | None
    timeout: float


@dataclass
class Result:
    name: str
    client: str
    effort: str
    owner: str
    passed: bool = False
    protocol_ok: bool = False
    task_ok: bool = False
    detail: str = ""
    seconds: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failing_exchange: dict | None = None
    server_log: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- fixtures


def git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=eval", "-c", "user.email=eval@localhost", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )


def write_files(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


BUG_FILES = {
    "calc.py": (
        "def average(values):\n"
        '    """Return the arithmetic mean of a non-empty list of numbers."""\n'
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total / (len(values) + 1)\n"
    ),
    "test_calc.py": (
        "import unittest\n\nfrom calc import average\n\n\n"
        "class AverageTest(unittest.TestCase):\n"
        "    def test_three(self):\n"
        "        self.assertEqual(average([2, 4, 6]), 4)\n\n"
        "    def test_one(self):\n"
        "        self.assertEqual(average([5]), 5)\n\n\n"
        'if __name__ == "__main__":\n    unittest.main()\n'
    ),
}
BUG_PROMPT = (
    "The unit tests in this repository fail. Run `python3 -m unittest -q` to see the failure, "
    "fix the bug in calc.py (do not edit test_calc.py), then run the tests again to confirm "
    "they pass. Finish with a one-sentence summary."
)
READ_FILES = {
    "settings/limits.py": "MAX_RETRIES = 7\nDEFAULT_TIMEOUT_SECONDS = 42\nBACKOFF_FACTOR = 1.5\n",
    "legacy/limits_old.py": "MAX_RETRIES = 3\nDEFAULT_TIMEOUT_SECONDS = 10\n",
}
READ_PROMPT = (
    "Read settings/limits.py and tell me the value it assigns to DEFAULT_TIMEOUT_SECONDS. "
    "Reply with just the number."
)
FOLLOWUP_PROMPT = "And what value does that same file assign to MAX_RETRIES? Reply with just the number."
RECOVERY_FILES = {"docs/notes.md": "# Notes\n\nThe code word is PERIWINKLE-42.\n"}
RECOVERY_PROMPT = (
    "Read the file docs/missing.md first. It may not exist; if reading it fails, read "
    "docs/notes.md instead. Reply with only the code word written in whichever file you could read."
)
CANCEL_PROMPT = (
    "Without using any tools, write a detailed 1500-word explanation of how TCP congestion "
    "control works, with numbered sections."
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- client runs


def prepare_client(ctx: Context, client: str, name: str, effort: str, compat: bool = True) -> tuple[Path, Path, dict]:
    base = ctx.out / "scenarios" / name
    if base.exists():
        shutil.rmtree(base)
    repo, home = base / "repo", base / "home"
    repo.mkdir(parents=True)
    home.mkdir()
    (base / "tmp").mkdir()
    scratch_tmp = str((base / "tmp").resolve())
    if client == "codex":
        codex_home = home / ".codex"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(codex_config(ctx.proxy_port, effort, compat))
        env = client_env(home, CODEX_HOME=str(codex_home), TMPDIR=scratch_tmp)
    else:
        agent = home / ".omp" / "agent"
        agent.mkdir(parents=True)
        (agent / "models.yml").write_text(omp_models_yml(ctx.proxy_port))
        env = client_env(home, TMPDIR=scratch_tmp)
    return repo, base, env


def write_jail(base: Path) -> list[str]:
    """`sandbox-exec` prefix that denies file writes outside `base` (and /dev).

    Oh My Pi has no sandbox of its own and runs with `--auto-approve`, so the
    harness confines it with macOS Seatbelt. Codex is not wrapped: Seatbelt
    cannot nest, and its own `workspace-write` sandbox is configured instead.
    """
    root = str(base.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    profile = f'(version 1)(allow default)(deny file-write*)(allow file-write* (subpath "{root}") (subpath "/dev"))'
    return ["/usr/bin/sandbox-exec", "-p", profile]


def client_command(ctx: Context, client: str, prompt: str, effort: str, repo: Path, resume: bool = False) -> list[str]:
    if client == "codex":
        assert ctx.codex_bin
        head = [ctx.codex_bin, "exec", "--json", "--skip-git-repo-check"]
        if resume:
            return [*head, "resume", "--last", prompt]
        return [*head, "-C", str(repo), prompt]
    assert ctx.omp_bin
    command = [
        *write_jail(repo.parent),
        ctx.omp_bin,
        "-p",
        "--mode",
        "json",
        "--model",
        f"local-ai/{MODEL_ID}",
        "--thinking",
        "off" if effort == "none" else "xhigh",
        "--auto-approve",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-lsp",
        "--no-title",
        # No sub-agent (`task`), `eval` or web tools: bounded process count, no network.
        "--tools",
        OMP_TOOLS,
        "--max-time",
        str(int(ctx.timeout)),
    ]
    if resume:
        command.append("--continue")
    return [*command, prompt]


def run_client(ctx: Context, command: list[str], repo: Path, env: dict, log: Path) -> tuple[int, float]:
    started = time.monotonic()
    with log.open("ab") as out:
        out.write(f"$ {' '.join(command)}\n".encode())
        out.flush()
        try:
            completed = subprocess.run(  # noqa: PLW1510 -- the exit code is the result
                command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                timeout=ctx.timeout, start_new_session=True,
            )
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = -999
    return code, time.monotonic() - started


def summarize_exchanges(result: Result, exchanges: list[dict]) -> None:
    result.requests = len(exchanges)
    for exchange in exchanges:
        response = response_object(exchange)
        usage = (response or {}).get("usage") or {}
        result.input_tokens += int(usage.get("input_tokens") or 0)
        result.output_tokens += int(usage.get("output_tokens") or 0)
        if result.failing_exchange is None and exchange_failure(exchange):
            result.failing_exchange = exchange
    result.protocol_ok = result.failing_exchange is None and bool(exchanges)


def final_answer(exchanges: list[dict]) -> str:
    for exchange in reversed(exchanges):
        response = response_object(exchange)
        if response is not None:
            return output_text(response)
    return ""


def client_scenario(kind: str, client: str, effort: str, compat: bool = True) -> Callable[[Context, Result], None]:
    def run(ctx: Context, result: Result) -> None:
        repo, base, env = prepare_client(ctx, client, result.name, effort, compat)
        files = {"fix": BUG_FILES, "read": READ_FILES, "recover": RECOVERY_FILES, "cancel": {"README.md": "scratch\n"}}[kind]
        write_files(repo, files)
        git_init(repo)
        log = base / "client.log"
        start = ctx.recorder.begin(result.name)
        if kind == "cancel":
            cancel_run(ctx, result, client, effort, repo, env, log)
            return
        prompt = {"fix": BUG_PROMPT, "read": READ_PROMPT, "recover": RECOVERY_PROMPT}[kind]
        code, seconds = run_client(ctx, client_command(ctx, client, prompt, effort, repo), repo, env, log)
        answers = [final_answer(ctx.recorder.since(start))]
        if kind == "read" and code == 0:
            mark = len(ctx.recorder.since(0))
            code, more = run_client(
                ctx, client_command(ctx, client, FOLLOWUP_PROMPT, effort, repo, resume=True), repo, env, log
            )
            seconds += more
            answers.append(final_answer(ctx.recorder.since(mark)))
        exchanges = ctx.recorder.since(start)
        summarize_exchanges(result, exchanges)
        result.seconds = seconds
        result.extra["exit_code"] = code
        result.extra["answers"] = answers
        tool_results = max((len(function_outputs(e.get("request_body"))) for e in exchanges), default=0)
        result.extra["max_tool_results_in_request"] = tool_results
        if kind == "fix":
            tests = subprocess.run(  # noqa: PLW1510 -- the exit code is the result
                [sys.executable, "-m", "unittest", "-q"], cwd=repo, capture_output=True, text=True, timeout=60
            )
            untouched = file_hash(repo / "test_calc.py") == hashlib.sha256(BUG_FILES["test_calc.py"].encode()).hexdigest()
            result.task_ok = tests.returncode == 0 and untouched and tool_results >= 2
            result.detail = f"tests rc={tests.returncode}, tests untouched={untouched}, tool results={tool_results}"
            result.extra["diff"] = subprocess.run(["git", "diff"], check=False, cwd=repo, capture_output=True, text=True).stdout
        elif kind == "read":
            ok = len(answers) == 2 and "42" in answers[0] and "7" in answers[1] and tool_results >= 1
            result.task_ok = ok
            result.detail = f"answers={answers!r}, tool results={tool_results}"
        else:
            result.task_ok = "PERIWINKLE" in answers[0] and tool_results >= 2
            result.detail = f"answer={answers[0]!r}, tool results={tool_results}"
        result.passed = result.protocol_ok and result.task_ok and code == 0
        if code != 0:
            result.detail += f"; client exit {code}"

    return run


def cancel_run(ctx: Context, result: Result, client: str, effort: str, repo: Path, env: dict, log: Path) -> None:
    command = client_command(ctx, client, CANCEL_PROMPT, effort, repo)
    start = ctx.recorder.begin(result.name)
    began = time.monotonic()
    with log.open("ab") as out:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True
        )
        streamed = ctx.recorder.first_chunk.wait(ctx.timeout)
        time.sleep(3)
        interrupted = time.monotonic()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(20)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    exit_seconds = time.monotonic() - interrupted
    # Give the proxy a moment to see the closed socket and record the exchange.
    deadline = time.monotonic() + 30
    exchanges = ctx.recorder.since(start)
    while time.monotonic() < deadline and not any(e.get("client_disconnected") for e in exchanges):
        time.sleep(0.5)  # the proxy records asynchronously; bounded wait
        exchanges = ctx.recorder.since(start)
    disconnected = any(e.get("client_disconnected") for e in exchanges)
    probe = probe_request(ctx.server_port)
    result.seconds = time.monotonic() - began
    result.requests = len(exchanges)
    result.extra.update(
        streamed=streamed, client_exited=exited, exit_seconds=exit_seconds,
        proxy_saw_disconnect=disconnected, followup=probe,
    )
    result.protocol_ok = probe["ok"]
    result.task_ok = streamed and exited and disconnected
    result.passed = result.protocol_ok and result.task_ok
    result.detail = (
        f"streamed={streamed}, client exited after SIGINT={exited} ({exit_seconds:.1f}s), "
        f"upstream closed on disconnect={disconnected}, follow-up {probe['status']} in {probe['seconds']:.1f}s"
    )


# ---------------------------------------------------------------- http probes


def post(port: int, path: str, body: dict | None, timeout: float = 1800.0, method: str = "POST") -> tuple[int, object, float]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, parse_body(response.read()), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, parse_body(error.read()), time.monotonic() - started


def probe_request(port: int) -> dict:
    status, body, seconds = post(
        port, "/v1/responses",
        {"input": "Reply with the single word OK.", "reasoning": {"effort": "none"}, "max_output_tokens": 8, "store": False},
        timeout=300,
    )
    return {"ok": status == 200, "status": status, "seconds": seconds, "text": output_text(body if isinstance(body, dict) else None)}


SIMPLE = {"input": "Reply with the single word OK.", "reasoning": {"effort": "none"}, "max_output_tokens": 16, "store": False}
APPLY_PATCH_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply a patch in the apply_patch format.",
    "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "*** Begin Patch" /(.|\\n)*/ "*** End Patch"'},
}
EXEC_TOOL = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
}
SHAPE_PROBES: dict[str, tuple[str, dict, str]] = {
    # name: (owner if refused, body, path)
    "http-reasoning-summary-auto": (
        "reasoning-summaries", {**SIMPLE, "reasoning": {"effort": "xhigh", "summary": "auto"}, "max_output_tokens": 64}, "/v1/responses"),
    "http-custom-tool-apply-patch": ("hosted-custom-tools", {**SIMPLE, "tools": [APPLY_PATCH_TOOL]}, "/v1/responses"),
    "http-local-shell-tool": ("hosted-custom-tools", {**SIMPLE, "tools": [{"type": "local_shell"}]}, "/v1/responses"),
    "http-web-search-tool": (
        "hosted-custom-tools", {**SIMPLE, "tools": [{"type": "web_search", "external_web_access": False}]}, "/v1/responses"),
    "http-namespace-tool": (
        "hosted-custom-tools",
        {**SIMPLE, "tools": [{"type": "namespace", "name": "multi_agent_v1", "description": "sub-agents", "tools": [EXEC_TOOL]}]},
        "/v1/responses"),
    "http-compact": (
        "compaction",
        {"model": MODEL_ID, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                                      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}],
         "instructions": "You are a coding agent."},
        "/v1/responses/compact"),
}


def shape_probe(owner: str, body: dict, path: str) -> Callable[[Context, Result], None]:
    def run(ctx: Context, result: Result) -> None:
        start = ctx.recorder.begin(result.name)
        status, response, seconds = post(ctx.proxy_port, path, body)
        exchanges = ctx.recorder.since(start)
        summarize_exchanges(result, exchanges)
        result.seconds = seconds
        result.task_ok = status == 200
        result.passed = result.protocol_ok and result.task_ok
        result.owner = owner
        result.detail = f"HTTP {status}: {json.dumps(response)[:300]}"

    return run


def encrypted_replay(ctx: Context, result: Result) -> None:
    """xhigh turn with encrypted reasoning, then replay its items statelessly."""
    start = ctx.recorder.begin(result.name)
    tools = [EXEC_TOOL]
    first = {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Use exec_command to run `cat VERSION`."}]}],
        "tools": tools, "reasoning": {"effort": "xhigh"}, "include": ["reasoning.encrypted_content"],
        "store": False, "max_output_tokens": 1024, "temperature": 0,
    }
    status1, body1, s1 = post(ctx.proxy_port, "/v1/responses", first)
    items = (body1 or {}).get("output", []) if isinstance(body1, dict) else []
    calls = [i for i in items if i.get("type") == "function_call"]
    encrypted = [i for i in items if i.get("type") == "reasoning" and i.get("encrypted_content")]
    status2, body2, s2 = (None, None, 0.0)
    if status1 == 200 and calls:
        replay = [i for i in items if i.get("type") in ("reasoning", "function_call")]
        for item in replay:
            if item.get("type") == "reasoning":
                item.pop("content", None)  # replay must work from the envelope alone
        second = {**first, "input": first["input"] + replay + [
            {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "2.7.1\n"}]}
        status2, body2, s2 = post(ctx.proxy_port, "/v1/responses", second)
    summarize_exchanges(result, ctx.recorder.since(start))
    answer = output_text(body2 if isinstance(body2, dict) else None)
    result.seconds = s1 + s2
    result.task_ok = status2 == 200 and bool(encrypted) and "2.7.1" in answer
    result.passed = result.protocol_ok and result.task_ok
    result.detail = f"turn1 {status1} calls={len(calls)} encrypted={len(encrypted)}; turn2 {status2} answer={answer[:120]!r}"


def stream_disconnect(ctx: Context, result: Result) -> None:
    """Close a streamed Responses request after its first delta; server must recover."""
    start = ctx.recorder.begin(result.name)
    body = json.dumps({"input": CANCEL_PROMPT, "reasoning": {"effort": "none"}, "stream": True, "max_output_tokens": 4096, "store": False})
    connection = http.client.HTTPConnection("127.0.0.1", ctx.proxy_port, timeout=600)
    connection.request("POST", "/v1/responses", body, {"Content-Type": "application/json"})
    response = connection.getresponse()
    seen = b""
    while b"response.output_text.delta" not in seen:
        chunk = response.read1(4096)
        if not chunk:
            break
        seen += chunk
    connection.sock.shutdown(socket.SHUT_RDWR)
    connection.close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and len(ctx.recorder.since(start)) < 1:
        time.sleep(0.5)  # bounded wait for the proxy thread to record the closed exchange
    exchanges = ctx.recorder.since(start)
    probe = probe_request(ctx.server_port)
    result.requests = len(exchanges)
    disconnected = any(e.get("client_disconnected") for e in exchanges)
    result.extra.update(followup=probe, proxy_saw_disconnect=disconnected)
    result.protocol_ok = probe["ok"]
    result.task_ok = b"output_text.delta" in seen and disconnected
    result.passed = result.protocol_ok and result.task_ok
    result.detail = f"delta seen={b'output_text.delta' in seen}, disconnect={disconnected}, follow-up {probe['status']} {probe['seconds']:.1f}s"


def background_cancel(ctx: Context, result: Result) -> None:
    start = ctx.recorder.begin(result.name)
    status, created, _ = post(
        ctx.proxy_port, "/v1/responses",
        {"input": CANCEL_PROMPT, "reasoning": {"effort": "none"}, "background": True, "max_output_tokens": 4096, "store": False},
    )
    rid = created.get("id") if isinstance(created, dict) else None
    time.sleep(5)
    cstatus, _, _ = post(ctx.proxy_port, f"/v1/responses/{rid}/cancel", None) if rid else (None, None, 0)
    _, fetched, _ = post(ctx.proxy_port, f"/v1/responses/{rid}", None, method="GET") if rid else (None, None, 0)
    summarize_exchanges(result, ctx.recorder.since(start))
    state = fetched.get("status") if isinstance(fetched, dict) else None
    result.task_ok = status == 200 and cstatus == 200 and state == "cancelled"
    result.passed = result.protocol_ok and result.task_ok
    result.detail = f"create {status}, cancel {cstatus}, final status {state!r}"


def invalid_then_valid(ctx: Context, result: Result) -> None:
    start = ctx.recorder.begin(result.name)
    bad, body, _ = post(ctx.proxy_port, "/v1/responses", {**SIMPLE, "reasoning": {"effort": "medium"}})
    good, _, seconds = post(ctx.proxy_port, "/v1/responses", SIMPLE)
    exchanges = ctx.recorder.since(start)
    result.requests = len(exchanges)
    result.seconds = seconds
    result.protocol_ok = good == 200
    result.task_ok = bad == 400
    result.passed = result.protocol_ok and result.task_ok
    result.detail = f"invalid effort -> {bad} {json.dumps(body)[:200]}; valid -> {good}"


# ---------------------------------------------------------------- registry


@dataclass
class Scenario:
    name: str
    client: str
    effort: str
    owner: str
    run: Callable[[Context, Result], None]
    needs: str = ""


def scenarios() -> list[Scenario]:
    items: list[Scenario] = []
    for client in ("omp", "codex"):
        for kind, owner in (("fix", "protocol"), ("read", "protocol"), ("recover", "protocol")):
            items.append(Scenario(f"{client}-{kind}", client, "none", owner, client_scenario(kind, client, "none"), client))
        items.append(Scenario(f"{client}-fix-xhigh", client, "xhigh", "protocol", client_scenario("fix", client, "xhigh"), client))
        items.append(Scenario(f"{client}-cancel", client, "none", "protocol", client_scenario("cancel", client, "none"), client))
    items.append(Scenario(
        "codex-default-config", "codex", "none", "hosted-custom-tools",
        client_scenario("read", "codex", "none", compat=False), "codex"))
    for name, (owner, body, path) in SHAPE_PROBES.items():
        items.append(Scenario(name, "http", "-", owner, shape_probe(owner, body, path)))
    items.append(Scenario("http-encrypted-reasoning-replay", "http", "xhigh", "protocol", encrypted_replay))
    items.append(Scenario("http-stream-disconnect", "http", "none", "protocol", stream_disconnect))
    items.append(Scenario("http-background-cancel", "http", "none", "background", background_cancel))
    items.append(Scenario("http-invalid-then-valid", "http", "none", "protocol", invalid_then_valid))
    return items


def summary_markdown(results: list[Result]) -> str:
    lines = ["| Scenario | Client | Effort | Pass | Protocol | Task | Requests | Seconds | Owner if failing | Detail |",
             "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |"]
    for r in results:
        detail = r.detail.replace("|", "\\|").replace("\n", " ")[:220]
        lines.append(
            f"| {r.name} | {r.client} | {r.effort} | {'PASS' if r.passed else 'FAIL'} | "
            f"{'ok' if r.protocol_ok else 'fail'} | {'ok' if r.task_ok else 'fail'} | {r.requests} | "
            f"{r.seconds:.0f} | {'' if r.passed else r.owner} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/local-ai")
    parser.add_argument("--port", type=int, default=8107)
    parser.add_argument("--no-serve", action="store_true", help="use a server already on --port")
    parser.add_argument("--codex-bin", default=shutil.which("codex"))
    parser.add_argument("--omp-bin", default=shutil.which("omp"))
    parser.add_argument("--scenarios", default="", help="comma-separated names or prefixes")
    parser.add_argument("--timeout", type=float, default=900.0, help="per client run, seconds")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    chosen = scenarios()
    if args.scenarios:
        wanted = [w for w in args.scenarios.split(",") if w]
        chosen = [s for s in chosen if any(s.name == w or s.name.startswith(w) for w in wanted)]
    if args.list:
        for s in chosen:
            print(f"{s.name:36} {s.client:6} {s.effort:6} {s.owner}")
        return 0
    out = (args.out or ROOT / ".amp/in/artifacts/client-compat" / time.strftime("run-%Y%m%dT%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    versions = {"omp": tool_version(args.omp_bin), "codex": tool_version(args.codex_bin)}
    server = None
    if not args.no_serve:
        key = out / "reasoning.key"
        server = LocalServer(args.server_bin, args.port, out / "server.log", ["--reasoning-key", str(key)])
        server.start()
    recorder = Recorder(args.port, out / "exchanges.jsonl")
    results: list[Result] = []
    try:
        proxy_port = recorder.start()
        ctx = Context(out, recorder, proxy_port, args.port, server, args.codex_bin, args.omp_bin, args.timeout)
        for scenario in chosen:
            result = Result(scenario.name, scenario.client, scenario.effort, scenario.owner)
            binary = {"codex": args.codex_bin, "omp": args.omp_bin}.get(scenario.needs, "-")
            if not binary:
                result.detail = f"skipped: {scenario.needs} not installed"
                results.append(result)
                continue
            offset = server.offset() if server else 0
            print(f"== {scenario.name}", flush=True)
            try:
                scenario.run(ctx, result)
            except Exception as error:  # noqa: BLE001 -- record and continue with the next scenario
                result.detail = f"harness error: {error!r}"
            if server and not result.passed:
                result.server_log = server.since(offset)
            print(f"   {'PASS' if result.passed else 'FAIL'} {result.detail[:200]}", flush=True)
            results.append(result)
            write_report(out, results, versions)
    finally:
        recorder.stop()
        if server:
            server.stop()
    write_report(out, results, versions)
    print(f"report: {out / 'summary.md'}")
    return 0


def tool_version(binary: str | None) -> str | None:
    if not binary:
        return None
    try:
        return subprocess.run([binary, "--version"], check=False, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def write_report(out: Path, results: list[Result], versions: dict) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=False, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    payload = {"commit": commit, "versions": versions, "results": [r.__dict__ for r in results]}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / "summary.md").write_text(
        f"commit {commit}; clients {versions}\n\n" + summary_markdown(results))


if __name__ == "__main__":
    sys.exit(main())
