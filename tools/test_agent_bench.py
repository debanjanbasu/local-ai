import json
import tempfile
import unittest
from pathlib import Path

import agent_bench as ab

SLUGIFY_SOLUTION = '''import re


def slugify(text, max_length=None):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if max_length is not None:
        slug = slug[:max_length].rstrip("-")
    return slug or "n-a"
'''


def edit(name, old, new):
    def apply(repo):
        path = repo / name
        text = path.read_text()
        assert old in text, (name, old)
        path.write_text(text.replace(old, new))
    return apply


def rename_everywhere(repo):
    for path in repo.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            path.write_text(path.read_text().replace("calc_total", "compute_total"))


def add_ignore_blank(repo):
    edit("linecount.py", "def count(text):", "def count(text, ignore_blank=False):")(repo)
    edit("linecount.py", "    return len(lines),",
         "    counted = [line for line in lines if line.strip()] if ignore_blank else lines\n"
         "    return len(counted),")(repo)
    edit("linecount.py", '    args = parser.parse_args(argv)',
         '    parser.add_argument("--ignore-blank", action="store_true")\n    args = parser.parse_args(argv)')(repo)
    edit("linecount.py", "count(handle.read())", "count(handle.read(), args.ignore_blank)")(repo)
    edit("README.md", "- `-h`", "- `--ignore-blank`: do not count blank lines.\n- `-h`")(repo)


# Per task: (edits that solve it, final answer). A reference solution must pass
# and the untouched fixture must fail, or the task measures nothing.
SOLUTIONS = {
    "fix-pagination": ([edit("shop/util.py", "return numerator // denominator",
                             "return -(-numerator // denominator)")], ""),
    "implement-slugify": ([lambda repo: (repo / "textkit/slug.py").write_text(SLUGIFY_SOLUTION)], ""),
    "rename-function": ([rename_everywhere], ""),
    "locate-answer": ([], "status=409 limit=1000"),
    "add-cli-flag": ([add_ignore_blank], ""),
    "fix-large-file": ([edit("units/convert.py", "POUNDS_TO_KILOGRAMS = 0.4535237",
                             "POUNDS_TO_KILOGRAMS = 0.45359237")], ""),
    "make-check-pass": ([edit("app/geometry.py", "def rectangle_perimeter(width, height):\n",
                              'def rectangle_perimeter(width, height):\n    """Perimeter."""\n'),
                         edit("app/text.py", "return text[:1].upper() + text[1:].lower()",
                              'return " ".join(word.capitalize() for word in text.split(" "))')], ""),
}


class TaskCheckTest(unittest.TestCase):
    def repo(self, task):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        repo = Path(directory.name)
        ab.cc.write_files(repo, task.files)
        ab.cc.git_init(repo)
        return repo

    def test_every_task_has_a_reference_solution(self):
        self.assertEqual(sorted(SOLUTIONS), sorted(t.name for t in ab.TASKS))

    def test_fixture_fails_and_reference_solution_passes(self):
        for task in ab.TASKS:
            with self.subTest(task=task.name):
                passed, failures = ab.check_task(task, self.repo(task), "")
                self.assertFalse(passed)
                repo = self.repo(task)
                edits, answer = SOLUTIONS[task.name]
                for apply in edits:
                    apply(repo)
                passed, failures = ab.check_task(task, repo, answer)
                self.assertTrue(passed, failures)
                self.assertFalse((repo / ab.HIDDEN_TEST).exists())

    def test_symptom_fixes_and_shortcuts_fail(self):
        cases = [
            # Patching page_count instead of the helper's rounding.
            (ab.PAGINATION, [edit("shop/pagination.py", "max(1, ceil_div(total, per_page))",
                                  "max(1, -(-total // per_page))")], ""),
            # Editing the test to match the bug.
            (ab.LARGE_FILE, [edit("tests/test_convert.py", "4.5359237", "4.535237")], ""),
            # An alias under the old name.
            (ab.RENAME, [rename_everywhere, edit("billing/calc.py", "def compute_total",
                                                 "def calc_total(*a):\n    return compute_total(*a)\n\n\ndef compute_total")], ""),
            # The decoy's answer, or the right answer after modifying the repository.
            (ab.LOCATE, [], "status=400 limit=500"),
            (ab.LOCATE, [edit("orders/status.py", "CONFLICT = 409", "CONFLICT = 409  # note")], "status=409 limit=1000"),
            # The flag works but is not documented.
            (ab.LINECOUNT, [add_ignore_blank, edit("README.md", "--ignore-blank", "--skip-empty")], ""),
        ]
        for task, edits, answer in cases:
            with self.subTest(task=task.name, answer=answer):
                repo = self.repo(task)
                for apply in edits:
                    apply(repo)
                self.assertFalse(ab.check_task(task, repo, answer)[0])

    def test_large_file_is_large_and_stable(self):
        self.assertGreater(len(ab.LARGE_FILE.files["units/convert.py"].splitlines()), 500)
        # Regenerated identically, so prompts do not drift between runs.
        self.assertEqual(ab._units_module()[0], ab.LARGE_FILE.files["units/convert.py"])


def sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


EXEC_TOOL = {"type": "function", "name": "exec_command", "parameters": {
    "type": "object", "properties": {"cmd": {"type": "string"}, "yield_time_ms": {"type": "number"}},
    "required": ["cmd"], "additionalProperties": False}}


def responses_turn(calls, tools=(EXEC_TOOL,), inputs=(), text="", ttft=1.5):
    output = [{"type": "function_call", "call_id": cid, "name": name, "arguments": args} for cid, name, args in calls]
    if text:
        output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    completed = {"type": "response.completed", "response": {"object": "response", "output": output, "usage": {
        "input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600}, "output_tokens": 20}}}
    return {"path": "/v1/responses", "status": 200, "first_token_seconds": ttft, "duration_seconds": 3.0,
            "request_body": {"tools": list(tools), "input": list(inputs)}, "response_body": sse(completed)}


def output(cid, text):
    return {"type": "function_call_output", "call_id": cid, "output": text}


class AnalyzeTest(unittest.TestCase):
    def test_categories_turns_and_timing(self):
        ok_args = json.dumps({"cmd": "python3 -m unittest -q"})
        exchanges = [
            responses_turn([("a", "apply_patch", "{}"), ("b", "exec_command", '{"cmd": "ls"')]),
            responses_turn([("c", "exec_command", json.dumps({"command": "ls"})),
                            ("d", "exec_command", json.dumps({"cmd": 5}))],
                           inputs=[output("a", "unsupported call: apply_patch"), output("b", "x")], ttft=4.0),
            # The same call three times: the third is a loop; formatting does not matter.
            responses_turn([("e", "exec_command", ok_args)]),
            responses_turn([("f", "exec_command", ok_args)]),
            responses_turn([("g", "exec_command", json.dumps({"cmd": "python3 -m unittest -q"}, indent=1))]),
            responses_turn([("h", "exec_command", json.dumps({"cmd": "cat x"}))], ttft=None),
            responses_turn([], inputs=[output("h", "failed to parse function arguments: missing field `cmd`")],
                           text='<tool_call>{"name": "exec_command"}</tool_call>'),
            # A title request (no tools) is not a turn; a compaction turn is counted apart.
            {"path": "/v1/responses", "status": 200, "request_body": {"input": "title"}, "response_body": sse()},
            {"path": "/v1/responses", "status": 200, "request_body": {"tools": [EXEC_TOOL],
                                                                     "input": [{"type": "compaction_trigger"}]},
             "response_body": sse()},
        ]
        metrics = ab.analyze(exchanges)
        categories = [c["category"] for c in metrics["calls"]]
        self.assertEqual(categories, ["wrong_tool", "bad_args", "bad_args", "bad_args", None, None, "loop",
                                      "bad_args", "bad_args"])
        self.assertIn("malformed JSON", metrics["calls"][1]["evidence"])
        self.assertIn("schema", metrics["calls"][2]["evidence"])
        self.assertIn("client rejected", metrics["calls"][7]["evidence"])
        self.assertEqual(metrics["tool_failures"], {"wrong_tool": 1, "bad_args": 5, "loop": 1})
        self.assertEqual((metrics["turns"], metrics["compactions"], metrics["tool_calls"]), (7, 1, 8))
        self.assertEqual((metrics["ttft_first"], metrics["ttft_max"]), (1.5, 4.0))
        self.assertEqual((metrics["prompt_tokens"], metrics["uncached_prompt_tokens"]), (7000, 2800))

    def test_streamed_chat_tool_calls(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"filePath":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "a.py"}'}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 10},
                                      "completion_tokens": 5}},
        ]
        tool = {"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}
        turn = {"path": "/v1/chat/completions", "status": 200, "request_body": {"tools": [tool], "messages": []},
                "response_body": sse(*chunks) + "data: [DONE]\n\n"}
        rejected = {"path": "/v1/chat/completions", "status": 200, "request_body": {"tools": [tool], "messages": [
            {"role": "tool", "tool_call_id": "call_1", "content": "The read tool was called with invalid arguments: x"}]},
            "response_body": sse({"choices": [{"delta": {"content": "done"}}]})}
        calls, _ = ab.model_calls(turn)
        self.assertEqual(calls, [{"call_id": "call_1", "name": "read", "arguments": '{"filePath": "a.py"}'}])
        metrics = ab.analyze([turn, rejected])
        self.assertEqual(metrics["tool_failures"], {"wrong_tool": 0, "bad_args": 1, "loop": 0})
        self.assertEqual((metrics["prompt_tokens"], metrics["uncached_prompt_tokens"]), (50, 40))

    def test_shell_output_does_not_count_as_a_rejection(self):
        # A command's own error text deep in its output is not a client rejection.
        result = "Process exited with code 2\nOutput:\n" + "x\n" * 200 + "error: invalid arguments"
        self.assertEqual(ab.classify_call({"name": "exec_command", "arguments": '{"cmd": "x"}'},
                                          ab.declared_tools({"tools": [EXEC_TOOL]}), result), (None, ""))


class ReportTest(unittest.TestCase):
    def test_compare_lines_up_runs(self):
        run = {"task": "t", "client": "codex", "seed": 1, "success": True, "turns": 3, "tool_calls": 2,
               "tool_failures": {"wrong_tool": 0, "bad_args": 0, "loop": 0}, "protocol_errors": [], "timed_out": False,
               "wall_seconds": 30.0, "ttft_first": 9.0, "per_turn": [{"ttft": 9.0}], "prompt_tokens": 1,
               "uncached_prompt_tokens": 1, "check_failures": [], "calls": [], "exit_code": 0, "max_prompt_tokens": 1,
               "ttft_median": 9.0}
        worse = {**run, "success": False, "turns": 5, "tool_failures": {"wrong_tool": 1, "bad_args": 0, "loop": 2}}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for label, r in (("a", run), ("b", worse)):
                out = Path(directory) / label
                out.mkdir()
                ab.write_report(out, {"label": label, "server": {"commit": label * 9}, "versions": {}, "effort": "none",
                                      "seeds": "1", "runs": [r]})
                paths.append(out / "results.json")
            text = ab.compare(*paths)
        self.assertIn("| codex | 1/1 → 0/1 | 3.0 → 5.0 | 0 → 1 | 0 → 0 | 0 → 2 |", text)
        self.assertIn("| t | codex | 1 | PASS → FAIL | 3 → 5 | 0 → 3 |", text)


if __name__ == "__main__":
    unittest.main()
