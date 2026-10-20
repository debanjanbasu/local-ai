#!/bin/sh
# One GPU-lock hold (lockf -k): baseline sessions, then decision-point replays of them
# under several sampling settings and on the key-order build.
cd "$(dirname "$0")/../.." || exit 1
PY=.amp/in/venv/bin/python
R=.amp/in/tool-calls/runs
$PY tools/tool_calls/suite.py --server-bin .amp/in/bin/local-ai-base --out $R/baseline
$PY tools/tool_calls/replay.py --exchanges $R/baseline/exchanges.jsonl --reasoning-key $R/baseline/reasoning.key \
  --server-bin .amp/in/bin/local-ai-base --out $R/replay-base.json --points auto --seeds 6 \
  --configs '{"default":{},"t07p08":{"temperature":0.7,"top_p":0.8},"greedy":{"temperature":0}}'
$PY tools/tool_calls/replay.py --exchanges $R/baseline/exchanges.jsonl --reasoning-key $R/baseline/reasoning.key \
  --server-bin .amp/in/bin/local-ai-order --out $R/replay-order.json --points auto --seeds 6 \
  --configs '{"default":{}}'
echo ROUND1-DONE
