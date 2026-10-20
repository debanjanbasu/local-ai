#!/usr/bin/env python3
"""Turns (scenario:index among agent requests) whose response made a call the
audit flags: unknown tool, unknown parameter, schema error, or misuse."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit

run = Path(sys.argv[1])
exchanges = [json.loads(l) for l in (run / "exchanges.jsonl").open()]
points = []
for scenario in dict.fromkeys(e["scenario"] for e in exchanges):
    items = [e for e in exchanges if e["scenario"] == scenario and isinstance(e.get("request_body"), dict) and e["request_body"].get("tools")]
    results = audit.results_by_id(items)
    for index, e in enumerate(items):
        tools = audit.declared(e["request_body"])
        chat = e["path"].startswith("/v1/chat")
        calls, _, _ = audit.chat_calls(e) if chat else audit.responses_calls(e)
        for call in calls:
            bad = call["name"] not in tools
            try:
                value = json.loads(call["arguments"])
                schema = tools.get(call["name"]) or {}
                bad |= bool(set(value) - set(schema.get("properties") or value))
            except ValueError:
                bad = True
            result = results.get(call.get("id"), "")
            bad |= bool(result and audit.MISUSE.search(result[:2000]))
            if bad:
                points.append(f"{scenario}:{index}")
                break
print(",".join(points))
