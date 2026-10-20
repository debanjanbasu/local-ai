#!/usr/bin/env python3
"""Decision-point replays: resend recorded client requests to a fresh
`local-ai serve` under several seeds and sampling settings, and tally what
the model does next (final message, or which call). Cheap, targeted
reproductions of the failure modes seen in whole sessions.

    lockf -k /tmp/local-ai-gpu.lock .amp/in/venv/bin/python tools/tool_calls/replay.py \
        --exchanges RUN/exchanges.jsonl --server-bin BIN --out OUT.json \
        --points codex-implement-0:4,codex-config-0:0 --seeds 8 \
        --configs '{"default":{},"t07":{"temperature":0.7,"top_p":0.8}}'

A point is `scenario:turn` (the turn-th agent request of the scenario).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import client_compat as cc


def agent_requests(exchanges: list[dict], scenario: str) -> list[dict]:
    return [e for e in exchanges if e["scenario"] == scenario and isinstance(e.get("request_body"), dict)
            and e["request_body"].get("tools")]


SUCCESS = ("Ran 2 tests", "Ran 3 tests")


def auto_points(exchanges: list[dict], prefix: str) -> list[str]:
    """Per scenario: its first turn (which tool, which arguments), and the
    first turn whose newest tool result shows the task verified (the agent
    should answer and stop there)."""
    points = []
    for scenario in dict.fromkeys(e["scenario"] for e in exchanges if e["scenario"].startswith(prefix)):
        requests = agent_requests(exchanges, scenario)
        if not requests:
            continue
        points.append(f"{scenario}:0")
        for index, exchange in enumerate(requests):
            body = exchange["request_body"]
            if isinstance(body.get("messages"), list):
                results = [m.get("content") for m in body["messages"] if m.get("role") == "tool"]
            else:
                results = [i.get("output") for i in body.get("input") or []
                           if isinstance(i, dict) and i.get("type") == "function_call_output"]
            newest = json.dumps(results[-1]) if results else ""
            tested = any(s in newest for s in SUCCESS) and "OK" in newest and "FAIL" not in newest
            if tested or "REQUEST_TIMEOUT = 45" in newest:
                points.append(f"{scenario}:{index}")
                break
    return points


def summarize(path: str, body: object) -> dict:
    if path.startswith("/v1/chat"):
        message = ((body or {}).get("choices") or [{}])[0].get("message") or {}
        calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                 for c in message.get("tool_calls") or []]
        return {"text": message.get("content") or "", "calls": calls}
    output = (body or {}).get("output") or []
    calls = [{"name": i["name"], "arguments": i.get("arguments") or i.get("input")}
             for i in output if i.get("type") in ("function_call", "custom_tool_call")]
    return {"text": cc.output_text(body), "calls": calls, "status": (body or {}).get("status")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchanges", type=Path, required=True)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--reasoning-key", type=Path)
    parser.add_argument("--points", required=True, help="scenario:turn,...")
    parser.add_argument("--prefix", default="", help="auto points only for scenarios with this prefix")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--configs", default='{"default":{}}')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    exchanges = [json.loads(line) for line in args.exchanges.open()]
    configs = json.loads(args.configs)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = ["--reasoning-key", str(args.reasoning_key)] if args.reasoning_key else []
    server = cc.LocalServer(args.server_bin, args.port, out.with_suffix(".server.log"), extra)
    server.start()
    records = []
    try:
        points = auto_points(exchanges, args.prefix) if args.points == "auto" else args.points.split(",")
        print("points:", points, flush=True)
        for point in points:
            scenario, turn = point.rsplit(":", 1)
            exchange = agent_requests(exchanges, scenario)[int(turn)]
            path = exchange["path"].split("?")[0]
            for label, overrides in configs.items():
                tally = Counter()
                for seed in range(args.seeds):
                    body = json.loads(json.dumps(exchange["request_body"]))
                    overrides = dict(overrides)
                    note = overrides.pop("_note", None)
                    if note:
                        # Appended to the leading instruction message, as a
                        # client's developer instructions would be.
                        first = (body.get("messages") or body.get("input"))[0]
                        if isinstance(first.get("content"), str):
                            first["content"] += "\n\n" + note
                        else:
                            first["content"][-1]["text"] += "\n\n" + note
                    body.update(overrides)
                    body["stream"] = False
                    body.pop("stream_options", None)
                    body["seed"] = seed
                    started = time.monotonic()
                    status, reply, _ = cc.post(args.port, path, body)
                    result = summarize(path, reply) if status == 200 else {"error": str(reply)[:300], "calls": []}
                    kind = ("error" if status != 200 else
                            "message" if not result["calls"] else
                            "call:" + ",".join(c["name"] for c in result["calls"]))
                    tally[kind] += 1
                    records.append({"point": point, "config": label, "seed": seed, "kind": kind,
                                    "seconds": round(time.monotonic() - started, 1), **result})
                    out.write_text(json.dumps(records, indent=1))
                print(point, label, dict(tally), flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
