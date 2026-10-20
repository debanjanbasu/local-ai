#!/usr/bin/env python3
"""Targeted tool-call reliability runs: real Codex and opencode sessions
against `local-ai serve`, recorded through client_compat's proxy, then
audited by audit.py. Scratch tooling until the agent benchmark lands.

    lockf -k /tmp/local-ai-gpu.lock .scratch/venv/bin/python tools/tool_calls/suite.py \
        --out .scratch/tool-calls/runs/baseline --repeats 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import client_compat as cc  # noqa: E402

TASKS: dict[str, dict] = {
    "fix": {"files": cc.BUG_FILES, "prompt": cc.BUG_PROMPT},
    "rename": {
        "files": {
            "utils/__init__.py": "",
            "utils/strings.py": (
                "import re\n\n\n"
                "def slugify_text(text):\n"
                '    """Lower-case `text` and join its words with hyphens."""\n'
                "    words = re.findall(r\"[a-z0-9]+\", text.lower())\n"
                "    return \"-\".join(words)\n"
            ),
            "app/__init__.py": "",
            "app/models.py": (
                "from utils.strings import slugify_text\n\n\n"
                "class Article:\n"
                "    def __init__(self, title):\n"
                "        self.title = title\n"
                "        self.slug = slugify_text(title)\n"
            ),
            "app/views.py": (
                "from utils import strings\n\n\n"
                "def article_url(title):\n"
                "    return \"/articles/\" + strings.slugify_text(title) + \"/\"\n"
            ),
            "test_app.py": (
                "import unittest\n\n"
                "from app.models import Article\n"
                "from app.views import article_url\n"
                "from utils.strings import make_slug\n\n\n"
                "class AppTest(unittest.TestCase):\n"
                "    def test_slug(self):\n"
                "        self.assertEqual(make_slug(\"Hello, World!\"), \"hello-world\")\n\n"
                "    def test_article(self):\n"
                "        self.assertEqual(Article(\"A B\").slug, \"a-b\")\n\n"
                "    def test_url(self):\n"
                "        self.assertEqual(article_url(\"X y\"), \"/articles/x-y/\")\n\n\n"
                "if __name__ == \"__main__\":\n    unittest.main()\n"
            ),
        },
        "prompt": (
            "Rename the function slugify_text to make_slug everywhere in this repository: its "
            "definition and every call site. Do not edit test_app.py. Then run "
            "`python3 -m unittest -q` to confirm the tests pass. Finish with a one-sentence summary."
        ),
    },
    "config": {
        "files": {
            "README.md": "# service\n\nHTTP client defaults live in the settings package.\n",
            "service/__init__.py": "",
            "service/settings/__init__.py": "",
            "service/settings/network.py": (
                "# Network defaults. Values are in seconds unless stated otherwise.\n"
                "CONNECT_TIMEOUT = 5\n"
                "REQUEST_TIMEOUT = 30\n"
                "RETRY_LIMIT = 3\n"
            ),
            "service/settings/cache.py": "CACHE_TTL = 30\nCACHE_SIZE = 1024\n",
            "service/client.py": (
                "from service.settings.network import CONNECT_TIMEOUT, REQUEST_TIMEOUT\n\n\n"
                "def timeouts():\n"
                "    # The request timeout was 30 seconds when this was written.\n"
                "    return (CONNECT_TIMEOUT, REQUEST_TIMEOUT)\n"
            ),
        },
        "prompt": (
            "Find where the default HTTP request timeout is defined in this repository and change "
            "it from 30 to 45 seconds. Change nothing else. Finish with a one-sentence summary."
        ),
    },
    "implement": {
        "files": {
            "textstats.py": (
                "def word_frequencies(text):\n"
                '    """Return a dict mapping each lower-cased word of `text` to its count.\n\n'
                "    Words are runs of letters and apostrophes; everything else separates them.\n"
                '    """\n'
                "    raise NotImplementedError\n"
            ),
            "test_textstats.py": (
                "import unittest\n\nfrom textstats import word_frequencies\n\n\n"
                "class FrequencyTest(unittest.TestCase):\n"
                "    def test_counts(self):\n"
                "        self.assertEqual(word_frequencies(\"The cat, the HAT.\"), {\"the\": 2, \"cat\": 1, \"hat\": 1})\n\n"
                "    def test_apostrophe(self):\n"
                "        self.assertEqual(word_frequencies(\"don't stop; don't\"), {\"don't\": 2, \"stop\": 1})\n\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(word_frequencies(\"  ...  \"), {})\n\n\n"
                "if __name__ == \"__main__\":\n    unittest.main()\n"
            ),
        },
        "prompt": (
            "Implement word_frequencies in textstats.py so that the tests in test_textstats.py pass "
            "(do not edit the tests), then run `python3 -m unittest -q` to confirm. Finish with a "
            "one-sentence summary."
        ),
    },
}


def tests_pass(repo: Path) -> bool:
    return subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=repo, capture_output=True,
                          timeout=60, check=False).returncode == 0


def check(task: str, repo: Path) -> tuple[bool, str]:
    def same(name: str) -> bool:
        return (repo / name).read_text() == TASKS[task]["files"][name]
    if task == "fix":
        ok = tests_pass(repo) and same("test_calc.py")
    elif task == "rename":
        leftover = subprocess.run(["grep", "-rn", "slugify_text", "--include=*.py", "."], cwd=repo,
                                  capture_output=True, text=True, check=False).stdout
        ok = tests_pass(repo) and same("test_app.py") and not leftover
    elif task == "config":
        network = (repo / "service/settings/network.py").read_text()
        others = all(same(n) for n in ("service/settings/cache.py", "service/client.py"))
        ok = "REQUEST_TIMEOUT = 45" in network and "CONNECT_TIMEOUT = 5" in network and others
    else:
        ok = tests_pass(repo) and same("test_textstats.py")
    diff = subprocess.run(["git", "diff", "--stat"], cwd=repo, capture_output=True, text=True, check=False).stdout
    return ok, diff.strip().replace("\n", "; ")


def run_client(ctx, command, repo, env, log):
    """cc.run_client, but a timeout stops the client's whole process group:
    Codex's npm launcher dies alone and leaves the native agent running."""
    started = time.monotonic()
    with log.open("ab") as out:
        out.write(f"$ {' '.join(command)}\n".encode())
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=out, start_new_session=True)
        try:
            code = process.wait(ctx.timeout)
        except subprocess.TimeoutExpired:
            cc.signal_group(process.pid, cc.signal.SIGKILL)
            process.wait()
            code = -999
    return code, time.monotonic() - started


CLIENTS = {
    "codex": ("codex", ""),
    "opencode": ("opencode", ""),
    "opencode-responses": ("opencode", "responses"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/local-ai")
    parser.add_argument("--server-args", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--codex-bin", default=str(ROOT / ".scratch/codex/node_modules/.bin/codex"))
    parser.add_argument("--opencode-bin", default="/opt/homebrew/bin/opencode")
    parser.add_argument("--clients", default="codex,opencode,opencode-responses")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--effort", default="none")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=450.0)
    parser.add_argument("--inject", default="", help="JSON members added to generation requests that lack them")
    args = parser.parse_args()
    if args.inject:
        extra = json.loads(args.inject)
        base_connection = cc.http.client.HTTPConnection

        class Injecting(base_connection):
            def request(self, method, url, body=None, headers=None, **kw):
                if body and url.split("?")[0] in ("/v1/responses", "/v1/chat/completions"):
                    value = json.loads(body)
                    if value.get("tools"):
                        for key, member in extra.items():
                            value.setdefault(key, member)
                        body = json.dumps(value).encode()
                return super().request(method, url, body, headers or {}, **kw)

        cc.http.client.HTTPConnection = Injecting

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    server = cc.LocalServer(args.server_bin, args.port, out / "server.log",
                            ["--reasoning-key", str(out / "reasoning.key"), *args.server_args.split()])
    server.start()
    recorder = cc.Recorder(args.port, out / "exchanges.jsonl")
    results = []
    try:
        proxy = recorder.start()
        ctx = cc.Context(out, recorder, proxy, args.port, server, args.codex_bin, None,
                         args.opencode_bin, args.timeout)
        for repeat in range(args.repeats):
            for task in args.tasks.split(","):
                for label in args.clients.split(","):
                    client, variant = CLIENTS[label]
                    name = f"{label}-{task}-{repeat}"
                    repo, base, env = cc.prepare_client(ctx, client, name, args.effort, True, variant)
                    if client == "codex":
                        # Codex's default workspace-write sandbox keeps the
                        # temp dirs writable; without them /usr/bin/python3
                        # (xcrun) and heredocs fail, which is harness noise.
                        config = base / "home/.codex/config.toml"
                        config.write_text(config.read_text()
                                          .replace("exclude_tmpdir_env_var = true", "exclude_tmpdir_env_var = false")
                                          .replace("exclude_slash_tmp = true", "exclude_slash_tmp = false"))
                    cc.write_files(repo, TASKS[task]["files"])
                    cc.git_init(repo)
                    start = recorder.begin(name)
                    command = cc.client_command(ctx, client, TASKS[task]["prompt"], args.effort, repo)
                    code, seconds = run_client(ctx, command, repo, env, base / "client.log")
                    ok, diff = check(task, repo)
                    record = {"name": name, "client": label, "task": task, "repeat": repeat,
                              "task_ok": ok, "exit": code, "seconds": round(seconds, 1),
                              "requests": len(recorder.since(start)), "diff": diff}
                    results.append(record)
                    print(json.dumps(record), flush=True)
                    (out / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        recorder.stop()
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
