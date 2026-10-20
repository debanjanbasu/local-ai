import json
import shutil
import unittest

import mtp_code_bench as bench

# A correct and a plausible wrong answer per checker: a check that cannot tell
# them apart would report task success the model did not earn.
CODE = {
    "py-merge-intervals": (
        ("def merge_intervals(intervals):\n    out = []\n"
        "    for s, e in sorted(intervals):\n"
        "        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n            out.append([s, e])\n    return out\n"),
        # Touching intervals stay apart.
        ("def merge_intervals(intervals):\n    out = []\n"
        "    for s, e in sorted(intervals):\n"
        "        if out and s < out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n            out.append([s, e])\n    return out\n"),
    ),
    "py-fix-binary-search": (
        ("def binary_search(items, target):\n    lo, hi = 0, len(items)\n    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
        "        if items[mid] < target:\n            lo = mid + 1\n        else:\n"
        "            hi = mid\n    return -1\n"),
        ("def binary_search(items, target):\n    lo, hi = 0, len(items)\n    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
        "        if items[mid] < target:\n            lo = mid + 1\n        else:\n"
        "            hi = mid - 1\n    return -1\n"),
    ),
    "rs-top-k-frequent": (
        ("use std::collections::HashMap;\n"
        "pub fn top_k_frequent(words: &[&str], k: usize) -> Vec<String> {\n"
        "    let mut c: HashMap<&str, usize> = HashMap::new();\n"
        "    for &w in words { *c.entry(w).or_insert(0) += 1; }\n"
        "    let mut v: Vec<_> = c.into_iter().collect();\n"
        "    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));\n"
        "    v.into_iter().take(k).map(|(w, _)| w.to_string()).collect()\n}\n"),
        ("pub fn top_k_frequent(words: &[&str], k: usize) -> Vec<String> {\n"
        "    words.iter().take(k).map(|w| w.to_string()).collect()\n}\n"),
    ),
    "js-parse-csv-line": (
        ("function parseCsvLine(line) {\n  const out = []; let f = ''; let q = false;\n"
        "  for (let i = 0; i < line.length; i++) {\n    const c = line[i];\n"
        "    if (q) { if (c === '\"') { if (line[i + 1] === '\"') { f += '\"'; i++; } else q = false; }"
        " else f += c; }\n"
        "    else if (c === '\"') q = true; else if (c === ',') { out.push(f); f = ''; } else f += c;\n"
        "  }\n  out.push(f); return out;\n}\nmodule.exports = { parseCsvLine };\n"),
        "function parseCsvLine(line) { return line.split(','); }\nmodule.exports = { parseCsvLine };\n",
    ),
    "sql-top-earner": (
        ("SELECT dept, name FROM (SELECT dept, name, ROW_NUMBER() OVER (PARTITION BY dept "
        "ORDER BY salary DESC, name) AS r FROM employees) WHERE r = 1 ORDER BY dept;"),
        "SELECT dept, name FROM employees GROUP BY dept HAVING salary = MAX(salary) ORDER BY dept;",
    ),
}


def fence(lang, code):
    return f"Here it is:\n\n```{lang}\n{code}```\n"


def reply(*calls, content=""):
    return {"content": content, "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)]}


class CodeCheckTests(unittest.TestCase):
    def test_code_checks_separate_right_from_wrong(self):
        tasks = {task["id"]: task for task in bench.all_tasks()}
        for task_id, (right, wrong) in CODE.items():
            task = tasks[task_id]
            if task["lang"] == "rust" and not shutil.which("rustc"):
                continue
            if task["lang"] == "javascript" and not shutil.which("node"):
                continue
            with self.subTest(task_id):
                self.assertTrue(bench.check_code(task, fence(task["lang"], right))[0])
                self.assertFalse(bench.check_code(task, fence(task["lang"], wrong))[0])

    def test_missing_block_fails(self):
        task = next(t for t in bench.all_tasks() if t["id"] == "py-merge-intervals")
        self.assertEqual(bench.check_code(task, "no code"), (False, "no code block"))


class ToolCheckTests(unittest.TestCase):
    def check(self, task_id, message):
        task = next(t for t in bench.all_tasks() if t["id"] == task_id)
        return bench.check_tool(task, message)[0]

    def test_python_edit_must_apply_and_pass(self):
        good = {"path": "app/config.py", "old_string": "settings[key] = raw[key]",
                "new_string": "settings[key] = raw.get(key, DEFAULTS[key])"}
        self.assertTrue(self.check("tool-edit-python", reply(("edit_file", good))))
        absent = dict(good, old_string="settings[key] = raw [key]")
        self.assertFalse(self.check("tool-edit-python", reply(("edit_file", absent))))
        wrong = dict(good, new_string="settings[key] = raw.get(key)")
        self.assertFalse(self.check("tool-edit-python", reply(("edit_file", wrong))))

    @unittest.skipUnless(shutil.which("rustc"), "rustc")
    def test_rust_edit_must_break_ties(self):
        old = "pairs.sort_by(|a, b| b.1.cmp(&a.1));"
        good = {"path": "src/rank.rs", "old_string": old,
                "new_string": "pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));"}
        self.assertTrue(self.check("tool-edit-rust", reply(("edit_file", good))))
        stable = dict(good, new_string="pairs.sort_by_key(|p| std::cmp::Reverse(p.1));")
        self.assertFalse(self.check("tool-edit-rust", reply(("edit_file", stable))))

    def test_argument_checks(self):
        self.assertTrue(self.check("tool-parallel-reads", reply(
            ("read_file", {"path": "pyproject.toml"}), ("read_file", {"path": "README.md"}))))
        self.assertFalse(self.check("tool-parallel-reads", reply(("read_file", {"path": "README.md"}))))
        issue = {"title": "Leak", "body": "b", "labels": ["bug", "performance"], "priority": 1}
        self.assertTrue(self.check("tool-create-issue", reply(("create_issue", issue))))
        self.assertFalse(self.check("tool-create-issue", reply(("create_issue", dict(issue, priority=4)))))
        self.assertFalse(self.check("tool-create-issue", reply(("create_issue", dict(issue, labels=["urgent"])))))
        self.assertFalse(self.check("tool-final-summary", reply(("run_command", {"command": "ls"}))))
        malformed = {"content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "grep", "arguments": "{"}}]}
        self.assertFalse(self.check("tool-find-definition", malformed))


if __name__ == "__main__":
    unittest.main()
