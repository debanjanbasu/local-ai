#!/bin/sh
# One GPU-lock hold (lockf -k): replay the baseline turns that produced a bad call.
cd "$(dirname "$0")/../.." || exit 1
PY=.amp/in/venv/bin/python
R=.amp/in/tool-calls/runs
$PY tools/tool_calls/replay.py --exchanges $R/baseline/exchanges.jsonl --reasoning-key $R/baseline/reasoning.key \
  --server-bin .amp/in/bin/local-ai-base --out $R/replay-errors-base.json --points codex-fix-0:1,codex-rename-0:2,codex-config-0:2,codex-config-0:8,opencode-config-0:0,opencode-responses-config-0:0,codex-implement-0:1,opencode-implement-0:1 --seeds 8 \
  --configs '{"default":{},"t07p08":{"temperature":0.7,"top_p":0.8},"t06":{"temperature":0.6},"macos-note":{"_note":"This machine runs macOS, so shell commands use the BSD userland (for example `sed -i \"\"`, and `cat` has no `-A`) and Python is `python3`."}}'
$PY tools/tool_calls/replay.py --exchanges $R/baseline/exchanges.jsonl --reasoning-key $R/baseline/reasoning.key \
  --server-bin .amp/in/bin/local-ai-order --out $R/replay-errors-order.json --points codex-fix-0:1,codex-rename-0:2,codex-config-0:2,codex-config-0:8,opencode-config-0:0,opencode-responses-config-0:0,codex-implement-0:1,opencode-implement-0:1 --seeds 8 \
  --configs '{"default":{}}'
$PY tools/tool_calls/replay.py --exchanges $R/baseline/exchanges.jsonl --reasoning-key $R/baseline/reasoning.key \
  --server-bin .amp/in/bin/local-ai-base --out $R/replay-errors-medium.json --points codex-fix-0:1,codex-rename-0:2,codex-config-0:2,codex-config-0:8,opencode-config-0:0,opencode-responses-config-0:0,codex-implement-0:1,opencode-implement-0:1,codex-rename-0:5 --seeds 6 \
  --configs '{"medium":{"reasoning":{"effort":"medium"},"reasoning_effort":"medium"}}'
echo ROUND2-DONE
