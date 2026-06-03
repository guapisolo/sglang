#!/usr/bin/env bash
# Crash-safe STEADY-STATE decode profile:
#   - clear any dangling profiler session first (the bug that crashed the server)
#   - start sustained bench_serving load
#   - ABORT if the load dies (never start a profiler with no decode happening)
#   - ONE profiler session, num_steps auto-stops & dumps
set -u
TAG="${1:?usage: profile_steady.sh <tag> [num_steps] [concurrency]}"
NUM_STEPS="${2:-40}"
CONC="${3:-16}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
OUT="/cluster-storage/bench_runs/trace_${TAG}_$(date -u +%H%M%S)"
mkdir -p "$OUT"

curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER NOT UP on $PORT"; exit 1; }
# clear any dangling profiler from a previous failed capture
curl -s -m 5 -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null 2>&1 || true
sleep 2

# FIXED batch = pure decode: num-prompts == concurrency, long output, no admission churn
# -> profiled steps are clean steady-state decode (no chunked-prefill interleaving)
echo "[load] bench_serving FIXED batch=$CONC output=3072 (pure decode) in background"
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" \
  --dataset-name random --random-input-len 256 --random-output-len 3072 --random-range-ratio 1.0 \
  --num-prompts "$CONC" --max-concurrency "$CONC" \
  > "/tmp/solo-logs/sglang/load_${TAG}.out" 2>&1 &
LOADPID=$!

echo "[load] pid=$LOADPID; waiting 18s for steady-state decode"
sleep 18
if ! kill -0 "$LOADPID" 2>/dev/null; then
  echo "[ABORT] load died early; not starting profiler. tail:"; tail -15 "/tmp/solo-logs/sglang/load_${TAG}.out"; exit 2
fi
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || {
  echo "[ABORT] server unhealthy before profiling"; kill "$LOADPID" 2>/dev/null; exit 3; }

echo "[profile] $NUM_STEPS steps -> $OUT"
python -m sglang.profiler --url "http://127.0.0.1:$PORT" --num-steps "$NUM_STEPS" \
  --output-dir "$OUT" 2>&1 | tail -4

for i in $(seq 1 40); do
  ls "$OUT"/*/*TP-0*.trace.json.gz >/dev/null 2>&1 && break
  sleep 2
done
kill "$LOADPID" 2>/dev/null; wait "$LOADPID" 2>/dev/null
echo "TRACE_DIR=$OUT"
find "$OUT" -name "*TP-0*.trace.json.gz" 2>/dev/null
