#!/usr/bin/env python3
"""Audit recorded client traffic (client_compat exchanges.jsonl) for
tool-call failures: wrong tool, bad arguments, loops, and server-side call
failures. Usage: audit.py RUN_DIR [--calls]"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import client_compat as cc

LEAK = re.compile(r"<tool_call>|</tool_call>|<function=|<parameter=|</function>")
EXIT = re.compile(r"Process exited with code (\d+)")
# A misused command whose exit code a pipe hides, or one the client refused.
MISUSE = re.compile(
    r"unrecognized (?:flag|option)|illegal option|invalid option|unknown option|^usage: |\nusage: |"
    r"command not found|exec_command failed|rejected:|apply_patch verification failed|unsupported call|"
    r"No such file or directory|was called with invalid arguments|Invalid input|oldString not found|"
    r"Could not find oldString|must be different|You must read|^sed: \d|\nsed: \d", re.IGNORECASE)


def declared(body: dict) -> dict[str, dict | None]:
    tools = {}
    for tool in body.get("tools") or []:
        if tool.get("type") == "function" and "function" in tool:
            tools[tool["function"]["name"]] = tool["function"].get("parameters")
        elif tool.get("name"):
            tools[tool["name"]] = tool.get("parameters") if tool.get("type") == "function" else None
    return tools


def chat_calls(exchange: dict) -> tuple[list[dict], str, str | None]:
    body = exchange.get("response_body")
    calls: dict[int, dict] = {}
    text, finish = [], None
    if isinstance(body, dict):
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        return ([{"name": c["function"]["name"], "arguments": c["function"]["arguments"], "id": c.get("id")}
                 for c in message.get("tool_calls") or []], message.get("content") or "", finish)
    for event in cc.sse_events(body) if isinstance(body, str) else []:
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or "")
            finish = choice.get("finish_reason") or finish
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": "", "id": None})
                fn = call.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
                slot["id"] = call.get("id") or slot["id"]
    return list(calls.values()), "".join(text), finish


def responses_calls(exchange: dict) -> tuple[list[dict], str, str | None]:
    response = cc.response_object(exchange) or {}
    calls = []
    for item in response.get("output") or []:
        if item.get("type") == "function_call":
            calls.append({"name": item["name"], "arguments": item.get("arguments", ""), "id": item.get("call_id")})
        elif item.get("type") == "custom_tool_call":
            calls.append({"name": item["name"], "arguments": json.dumps({"input": item.get("input")}),
                          "id": item.get("call_id"), "custom": True})
    status = response.get("status")
    reason = (response.get("incomplete_details") or {}).get("reason")
    return calls, cc.output_text(response), reason or status


def results_by_id(exchanges: list[dict]) -> dict[str, str]:
    out = {}
    for e in exchanges:
        body = e.get("request_body")
        if not isinstance(body, dict):
            continue
        for m in body.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "tool":
                content = m.get("content")
                out[m.get("tool_call_id")] = content if isinstance(content, str) else json.dumps(content)
        for i in body.get("input") or [] if isinstance(body.get("input"), list) else []:
            if isinstance(i, dict) and i.get("type") in ("function_call_output", "custom_tool_call_output"):
                o = i.get("output")
                out[i.get("call_id")] = o if isinstance(o, str) else json.dumps(o)
    return out


def canonical(arguments: str) -> str:
    try:
        return json.dumps(json.loads(arguments), sort_keys=True)
    except ValueError:
        return arguments


def audit(run: Path, show_calls: bool) -> dict:
    exchanges = [json.loads(line) for line in (run / "exchanges.jsonl").open()]
    by_scenario = defaultdict(list)
    for e in exchanges:
        by_scenario[e["scenario"]].append(e)
    raw = json.loads((run / "results.json").read_text()) if (run / "results.json").exists() else []
    if isinstance(raw, dict):  # client_compat's layout
        raw = raw["results"]
    results = {r["name"]: r for r in raw}
    totals = Counter()
    rows = []
    for scenario, items in by_scenario.items():
        tool_results = results_by_id(items)
        counts = Counter()
        seen = Counter()
        previous = None
        problems = []
        for e in items:
            body = e.get("request_body")
            if not isinstance(body, dict) or not body.get("tools"):
                continue
            counts["turns"] += 1
            failure = cc.exchange_failure(e)
            if failure:
                counts["server_failure"] += 1
                problems.append(f"SERVER {failure[:300]}")
                continue
            tools = declared(body)
            chat = str(e.get("path", "")).startswith("/v1/chat")
            calls, text, finish = chat_calls(e) if chat else responses_calls(e)
            if LEAK.search(text or ""):
                counts["leaked_markup"] += 1
                problems.append(f"LEAK {text[-300:]!r}")
            if finish in ("length", "max_output_tokens"):
                counts["truncated"] += 1
                problems.append(f"TRUNCATED finish={finish}")
            for call in calls:
                counts["calls"] += 1
                name, args = call["name"], call["arguments"]
                key = (name, canonical(args))
                seen[key] += 1
                tags = []
                if name not in tools:
                    tags.append("unknown_tool")
                elif not call.get("custom"):
                    try:
                        value = json.loads(args)
                        schema = tools[name]
                        if schema:
                            jsonschema.validate(value, schema)
                            extra = set(value) - set(schema.get("properties") or {})
                            if extra:
                                tags.append(f"unknown_param:{sorted(extra)}")
                    except ValueError:
                        tags.append("args_not_json")
                    except jsonschema.ValidationError as error:
                        tags.append(f"schema:{error.message[:120]}")
                if seen[key] > 2:
                    tags.append(f"loop#{seen[key]}")
                elif seen[key] > 1:
                    tags.append("repeat")
                if previous == key:
                    tags.append("consecutive")
                previous = key
                result = tool_results.get(call.get("id"), "")
                exit_code = EXIT.search(result or "")
                if exit_code and exit_code.group(1) != "0":
                    tags.append(f"exit{exit_code.group(1)}")
                if result and MISUSE.search(result[:2000]):
                    tags.append("misuse")
                for tag in tags:
                    counts[tag.split(":")[0].split("#")[0]] += 1
                if tags or show_calls:
                    problems.append(f"{name} {args[:240]} -> {' '.join(tags) or 'ok'} | {(result or '')[:200]!r}")
        r = results.get(scenario, {})
        rows.append((scenario, r.get("task_ok"), r.get("seconds"), r.get("exit"), counts, problems))
        totals.update(counts)
        totals["scenarios"] += 1
        totals["task_ok"] += bool(r.get("task_ok"))
        totals["timeouts"] += r.get("exit") == -999
    for scenario, ok, seconds, code, counts, problems in rows:
        print(f"== {scenario} task_ok={ok} exit={code} seconds={seconds} {dict(counts)}")
        for p in problems:
            print("   ", p.replace("\n", "\\n"))
    print("TOTAL", dict(totals))
    return totals


if __name__ == "__main__":
    audit(Path(sys.argv[1]), "--calls" in sys.argv)
