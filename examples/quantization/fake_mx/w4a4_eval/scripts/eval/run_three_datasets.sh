#!/usr/bin/env bash
# Serial evaluation driver: smoke -> MATH-500 -> MMLU-Pro -> LiveCodeBench.
#
# Runs against an already-started OpenAI-compatible server. The per-dataset
# python scripts live next to this file and share the same interface:
#   run_<dataset>.py <port> <work_dir>
#
# Usage:
#   SCENARIO=<name> PORT=<port> WORK_DIR=<dir> bash run_three_datasets.sh
#
# Environment:
#   SCENARIO         label for this evaluation run (output subdirectory)
#   PORT             port of the target server
#   WORK_DIR         output root; results land in $WORK_DIR/$SCENARIO/
#   EVAL_PY          optional; python interpreter with evalscope installed
#                    (default: python3 from PATH)
#   MODELSCOPE_CACHE optional; dataset cache location, passed through to the
#                    eval scripts unchanged
set -euo pipefail

: "${SCENARIO:?}" "${PORT:?}" "${WORK_DIR:?}"
EVAL_PY="${EVAL_PY:-python3}"

base="http://127.0.0.1:$PORT"

# Wait for the server to come up (up to 5 minutes).
for _ in $(seq 1 60); do
  curl -fsS "$base/v1/models" >/dev/null && break
  sleep 5
done
curl -fsS "$base/v1/models" >/dev/null

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$WORK_DIR/$SCENARIO"
mkdir -p "$run_dir"

# Smoke: one greedy generation before the long runs.
curl -fsS "$base/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"1+1="}],"temperature":0,"max_tokens":16}' \
  >"$run_dir/smoke.json"

"$EVAL_PY" "$script_dir/run_math500.py" "$PORT" "$run_dir/math500"
"$EVAL_PY" "$script_dir/run_mmlu_pro.py" "$PORT" "$run_dir/mmlu_pro"
"$EVAL_PY" "$script_dir/run_livecodebench.py" "$PORT" "$run_dir/livecodebench"

echo "Evaluation complete: $run_dir"
