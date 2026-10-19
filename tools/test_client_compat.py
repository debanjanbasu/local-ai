import http.client
import http.server
import json
import tempfile
import threading
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import client_compat as cc

COMPLETED = {
    "type": "response.completed",
    "response": {
        "object": "response",
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hm"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "4"}, {"type": "output_text", "text": "2"}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    },
}


def sse(*events):
    return "".join(f"event: {e.get('type')}\ndata: {json.dumps(e)}\n\n" for e in events)


class ParsingTest(unittest.TestCase):
    def test_sse_events_skip_done_and_non_json(self):
        text = sse({"type": "a"}) + "data: [DONE]\n\ndata: not json\n\n" + sse({"type": "b"})
        self.assertEqual([e["type"] for e in cc.sse_events(text)], ["a", "b"])

    def test_failures(self):
        self.assertIsNone(cc.exchange_failure({"status": 200, "response_body": sse(COMPLETED)}))
        bad = {"status": 400, "response_body": {"error": {"message": "tool type \"custom\" is not supported"}}}
        self.assertIn("custom", cc.exchange_failure(bad))
        failed = {"status": 200, "response_body": sse({"type": "response.failed", "response": {"error": {"code": "x"}}})}
        self.assertIn("response.failed", cc.exchange_failure(failed))
        self.assertIn("no response", cc.exchange_failure({"status": None, "error": "refused"}))

    def test_final_text_from_sse_and_json(self):
        self.assertEqual(cc.output_text(cc.response_object({"response_body": sse(COMPLETED)})), "42")
        self.assertEqual(cc.output_text(cc.response_object({"response_body": COMPLETED["response"]})), "42")
        self.assertEqual(cc.final_answer([{"response_body": sse(COMPLETED)}, {"response_body": None}]), "42")

    def test_summarize_counts_tokens_and_first_failure(self):
        result = cc.Result("x", "omp", "none", "protocol")
        bad = {"status": 400, "response_body": {"error": {"message": "nope"}}}
        cc.summarize_exchanges(result, [{"status": 200, "response_body": sse(COMPLETED)}, bad])
        self.assertEqual((result.requests, result.input_tokens, result.output_tokens), (2, 10, 3))
        self.assertIs(result.failing_exchange, bad)
        self.assertFalse(result.protocol_ok)

    def test_function_outputs(self):
        body = {"input": [{"type": "message"}, {"type": "function_call_output", "call_id": "c", "output": "x"}]}
        self.assertEqual(len(cc.function_outputs(body)), 1)
        self.assertEqual(cc.function_outputs("text"), [])
        chat = {"messages": [{"role": "user", "content": "q"}, {"role": "tool", "content": "ENOENT"}]}
        self.assertEqual(cc.function_outputs(chat), [{"output": "ENOENT"}])

    def test_chat_answers_and_side_requests(self):
        chunks = [{"choices": [{"delta": {"content": "4"}}]}, {"choices": [{"delta": {"content": "2"}}]},
                  {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}]
        streamed = {"path": "/v1/chat/completions", "request_body": {"tools": [{}]},
                    "response_body": "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"}
        title = {"path": "/v1/chat/completions", "request_body": {"messages": []},
                 "response_body": {"choices": [{"message": {"content": "Title"}}]}}
        # The tool-less title request finishing last is not the agent's answer.
        self.assertEqual(cc.final_answer([streamed, title]), "42")
        result = cc.Result("x", "opencode", "none", "protocol")
        cc.summarize_exchanges(result, [{**streamed, "status": 200}])
        self.assertEqual((result.input_tokens, result.output_tokens), (9, 2))


class ConfigTest(unittest.TestCase):
    def test_redaction_and_environment(self):
        self.assertEqual(cc.redact_headers({"Authorization": "Bearer s", "A": "b"}), {"Authorization": "<redacted>", "A": "b"})
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "s", "GH_TOKEN": "t", "PATH": "/bin", "CODEX_HOME": "/x"}, clear=True):
            env = cc.client_env(Path("/scratch"), CODEX_HOME="/scratch/.codex")
        self.assertEqual(env, {"PATH": "/bin", "HOME": "/scratch", "CODEX_HOME": "/scratch/.codex"})

    def test_codex_config(self):
        compat = tomllib.loads(cc.codex_config(9999, "none", compat=True))
        self.assertEqual(compat["model_providers"]["localai"]["base_url"], "http://127.0.0.1:9999/v1")
        self.assertEqual(compat["model_providers"]["localai"]["wire_api"], "responses")
        self.assertEqual(compat["web_search"], "disabled")
        self.assertFalse(any(compat["features"].values()))
        self.assertTrue(compat["sandbox_workspace_write"]["exclude_slash_tmp"])
        self.assertFalse(compat["sandbox_workspace_write"]["network_access"])
        # Codex's default developer messages are kept.
        self.assertNotIn("include_permissions_instructions", compat)
        self.assertNotIn("skills", compat)
        default = tomllib.loads(cc.codex_config(9999, "xhigh", compat=False))
        self.assertNotIn("features", default)
        self.assertEqual(default["model_reasoning_effort"], "xhigh")

    def test_default_config_flag_drops_the_compat_config(self):
        with tempfile.TemporaryDirectory() as scratch:
            ctx = cc.Context(Path(scratch), None, 1, 2, None, "codex", None, None, 60, codex_default=True)
            _, base, env = cc.prepare_client(ctx, "codex", "codex-read", "none")
            config = tomllib.loads((Path(env["CODEX_HOME"]) / "config.toml").read_text())
            self.assertNotIn("features", config)
            self.assertNotIn("web_search", config)
            self.assertTrue(str(base).startswith(scratch))

    def test_omp_runs_in_write_jail(self):
        ctx = cc.Context(Path("/s"), None, 1, 2, None, None, "omp", "opencode", 60)
        command = cc.client_command(ctx, "omp", "hi", "none", Path("/scratch/base/repo"))
        self.assertEqual(command[0], "/usr/bin/sandbox-exec")
        self.assertIn('(deny file-write*)', command[2])
        self.assertIn('(subpath "/scratch/base")', command[2])
        self.assertEqual(command[3:5], ["omp", "-p"])

    def test_opencode_config_and_jail(self):
        chat = json.loads(cc.opencode_config(4321, responses=False, effort="none"))
        provider = chat["provider"]["local-ai"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"]["baseURL"], "http://127.0.0.1:4321/v1")
        self.assertEqual(provider["models"][cc.MODEL_ID]["options"]["reasoningEffort"], "none")
        responses = json.loads(cc.opencode_config(4321, responses=True, effort="xhigh"))
        self.assertEqual(responses["provider"]["local-ai"]["npm"], "@ai-sdk/openai")
        ctx = cc.Context(Path("/s"), None, 1, 2, None, None, None, "opencode", 60)
        command = cc.client_command(ctx, "opencode", "hi", "none", Path("/scratch/base/repo"), resume=True)
        self.assertEqual(command[0], "/usr/bin/sandbox-exec")
        self.assertEqual(command[3:5], ["opencode", "run"])
        self.assertIn("--continue", command)

    def test_codex_catalog_variant(self):
        bundled = {"models": [{"slug": "gpt-x", "apply_patch_tool_type": "freeform", "base_instructions": "b"}]}
        done = mock.Mock(stdout=json.dumps(bundled))
        with tempfile.TemporaryDirectory() as scratch, mock.patch.object(cc.subprocess, "run", return_value=done):
            ctx = cc.Context(Path(scratch), None, 1, 2, None, "codex", None, None, 60)
            _, _, env = cc.prepare_client(ctx, "codex", "codex-apply-patch", "none", variant="apply-patch")
            config = tomllib.loads((Path(env["CODEX_HOME"]) / "config.toml").read_text())
            catalog = json.loads(Path(config["model_catalog_json"]).read_text())
            self.assertEqual(catalog["models"][0]["slug"], cc.MODEL_ID)
            self.assertEqual(catalog["models"][0]["apply_patch_tool_type"], "freeform")
            self.assertEqual(catalog["models"][0]["base_instructions"], "b")

    def test_omp_models(self):
        text = cc.omp_models_yml(1234)
        self.assertIn("baseUrl: http://127.0.0.1:1234/v1", text)
        self.assertIn(f"- id: {cc.MODEL_ID}", text)

    def test_scenario_names_unique(self):
        names = [s.name for s in cc.scenarios()]
        self.assertEqual(len(names), len(set(names)))

    def test_summary_escapes_pipes(self):
        result = cc.Result("x", "http", "-", "compaction", detail="a|b")
        self.assertIn("a\\|b", cc.summary_markdown([result]))


class Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert "zstd" not in self.headers.get("Accept-Encoding", "")
        if body.get("bad"):
            payload = b'{"error":{"message":"refused"}}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in ({"type": "response.created"}, COMPLETED):
            data = sse(event).encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.write(b"0\r\n\r\n")


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        self.recorder = cc.Recorder(self.upstream.server_address[1], Path(self.tmp.name) / "x.jsonl", timeout=10)
        self.port = self.recorder.start()

    def tearDown(self):
        self.recorder.stop()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.tmp.cleanup()

    def request(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("POST", "/v1/responses", json.dumps(body),
                           {"Content-Type": "application/json", "Authorization": "Bearer secret", "Accept-Encoding": "zstd"})
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    def test_streams_and_records(self):
        start = self.recorder.begin("s1")
        status, data = self.request({"input": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(cc.output_text(cc.response_object({"response_body": data.decode()})), "42")
        status, _ = self.request({"bad": True})
        self.assertEqual(status, 400)
        exchanges = self.recorder.since(start)
        self.assertEqual([e["status"] for e in exchanges], [200, 400])
        self.assertEqual(exchanges[0]["scenario"], "s1")
        self.assertEqual(exchanges[0]["request_headers"]["Authorization"], "<redacted>")
        self.assertIsNone(cc.exchange_failure(exchanges[0]))
        self.assertEqual(cc.exchange_failure(exchanges[1]), "HTTP 400: refused")
        self.assertTrue(self.recorder.first_chunk.is_set())
        logged = [json.loads(line) for line in (Path(self.tmp.name) / "x.jsonl").read_text().splitlines()]
        self.assertEqual(len(logged), 2)


if __name__ == "__main__":
    unittest.main()
