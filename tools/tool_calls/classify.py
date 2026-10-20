#!/usr/bin/env python3
"""Tally known-bad call patterns in replay records, per point and config."""
import json, re, sys
from collections import defaultdict
PATTERNS = {
    "gnu_sed": re.compile(r"sed -i (?!''|\"\")['\"]"),
    "cat_A": re.compile(r"cat -A"),
    "bare_python": re.compile(r"(?:^|[\s;&|(\"'])python(?:\s|$)"),
    "rm_f": re.compile(r"\brm -f"),
}
DECLARED = {"exec_command", "write_stdin", "request_user_input", "bash", "edit", "glob", "grep", "read", "skill", "todowrite", "write"}
GREP = {"pattern", "path", "include"}
for path in sys.argv[1:]:
    table = defaultdict(lambda: defaultdict(int))
    for r in json.load(open(path)):
        cell = table[(r["point"], r["config"])]
        cell["n"] += 1
        cell["stop"] += not r["calls"]
        bad = set()
        for c in r["calls"]:
            args = c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])
            if c["name"] not in DECLARED:
                bad.add("unknown_tool")
            try:
                value = json.loads(args)
            except ValueError:
                bad.add("not_json"); continue
            if c["name"] == "grep" and set(value) - GREP:
                bad.add("unknown_param")
            text = value.get("cmd") or value.get("command") or ""
            for name, pattern in PATTERNS.items():
                if isinstance(text, str) and pattern.search(text):
                    bad.add(name)
        for b in bad:
            cell[b] += 1
        cell["any_bad"] += bool(bad)
    print("==", path)
    for (point, config), cell in table.items():
        print(f"{point:34} {config:11} " + " ".join(f"{k}={v}" for k, v in cell.items()))
