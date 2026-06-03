#!/usr/bin/env bash
# nemotron_spec_overlap.sh — boot a Nemotron MoE server with DP-attention + MTP(NEXTN)
# spec-v2, run a gsm8k sanity check, and optionally capture a profiling trace, to verify
# whether spec-v2 breaks the overlap scheduler.
#
# Model selector (MODEL_NAME):
#   super  -> /models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16                  (default)
#   ultra  -> /models/ea_nvidia_nemotron_3_ultra_550b_a55b_bf16_rl_050826_vv0.1
# Or override MODEL=/abs/path directly.
#
# Experiment knobs (env), so we can build the matrix in the plan:
#   SPEC=1|0        MTP(NEXTN) spec decoding on/off                                  (default 1)
#   OVERLAP=1|0     overlap scheduler; 0 -> --disable-overlap-schedule (spec v1 path) (default 1)
#   CUDA_GRAPH=1|0  1 -> cuda graph on (default; lets us see literal cudaGraphLaunch);
#                   0 -> --disable-cuda-graph (eager)
#   PROFILE=0|1     after boot, capture a torch-profiler decode trace via send_one    (default 0)
#
# Usage:
#   bash script.sh                          # super, tp8 ep8, spec v2, cuda graph, gsm8k
#   PROFILE=1 bash script.sh                # + capture a decode trace (spec v2, cuda graph)
#   OVERLAP=0 PROFILE=1 bash script.sh      # spec v1 (no overlap) reference trace
#   CUDA_GRAPH=0 PROFILE=1 bash script.sh   # eager (no cudaGraphLaunch to observe)
#   SPEC=0 PROFILE=1 bash script.sh         # no-spec baseline overlap trace
#   SKIP_BOOT=1 PROFILE=1 bash script.sh    # server already on $PORT, just profile/gsm8k
set -u

# --- model selection ---
: "${MODEL_NAME:=super}"
case "$MODEL_NAME" in
  super) : "${MODEL:=/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}" ;;
  ultra) : "${MODEL:=/models/ea_nvidia_nemotron_3_ultra_550b_a55b_bf16_rl_050826_vv0.1}" ;;
  *)     : "${MODEL:?MODEL_NAME must be 'super' or 'ultra' (or set MODEL=/abs/path)}" ;;
esac

# --- config (override via env) ---
: "${PORT:=30000}"
: "${TP:=8}"
: "${EP:=8}"             # expert parallel size (ep8); 1 disables EP
: "${DP_SIZE:=2}"        # dp-attention groups
: "${MTP_STEPS:=1}"
: "${MTP_TOPK:=1}"       # spec v2 requires topk == 1
: "${MTP_DRAFT:=2}"
# 0.6 is conservative: this is a shared node (~55 GiB/GPU already held by other
# containers), so 0.88*183GiB would not fit in the ~128 GiB free per B200. Bump on a clean node.
: "${MEMFRAC:=0.6}"
: "${CTXLEN:=8192}"
: "${MAX_RUNNING:=48}"
: "${REASONING_PARSER:=nemotron_3}"
: "${N:=200}"            # gsm8k questions
: "${PARALLEL:=32}"      # concurrent requests
: "${PROFILE_STEPS:=12}" # forward steps captured when PROFILE=1
: "${SPEC:=1}"
: "${OVERLAP:=1}"
: "${CUDA_GRAPH:=1}"
: "${PROFILE:=0}"
: "${SKIP_BOOT:=}"
RUN_TAG="${MODEL_NAME}_spec${SPEC}_ovlp${OVERLAP}_cg${CUDA_GRAPH}"
: "${LOG:=/cluster-storage/bench_runs/nemo_${RUN_TAG}_$(date -u +%Y%m%d_%H%M%S).log}"
: "${PROFILE_DIR:=/cluster-storage/bench_runs/prof_${RUN_TAG}_$(date -u +%Y%m%d_%H%M%S)}"

# --- assemble the serve command ---
# DP attention runs attention/Mamba on attn-TP subgroups while MoE stays expert-parallel (ep$EP);
# NEXTN draft-verifies MTP_DRAFT tokens/step via the extra_buffer mamba scheduler.
EXTRA=()

if [[ "$SPEC" == "1" ]]; then
  EXTRA+=(--speculative-algorithm NEXTN
          --speculative-num-steps "$MTP_STEPS"
          --speculative-eagle-topk "$MTP_TOPK"
          --speculative-num-draft-tokens "$MTP_DRAFT")
fi

if [[ "$OVERLAP" == "1" ]]; then
  export SGLANG_ENABLE_SPEC_V2=1     # NEXTN/MTP + DP-attention spec-v2 (overlap) path
  # spec v2 + radix cache requires the extra_buffer mamba scheduler
  [[ "$SPEC" == "1" ]] && EXTRA+=(--mamba-scheduler-strategy extra_buffer)
else
  # Force the legacy serialized (spec v1) path: the "broken overlap" reference.
  # extra_buffer needs radix cache; the no-overlap path uses no_buffer + no radix.
  export SGLANG_ENABLE_SPEC_V2=0
  EXTRA+=(--disable-overlap-schedule)
  [[ "$SPEC" == "1" ]] && EXTRA+=(--mamba-scheduler-strategy no_buffer --disable-radix-cache)
fi

if [[ "$CUDA_GRAPH" == "0" ]]; then
  EXTRA+=(--disable-cuda-graph)
else
  # Keep the captured graph set small so it fits alongside the weights on a shared node.
  EXTRA+=(--cuda-graph-max-bs "$MAX_RUNNING")
fi

SERVE_CMD="SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-1} \
SGLANG_TORCH_PROFILER_DIR='$PROFILE_DIR' \
sglang serve \
  --model-path $MODEL --trust-remote-code --tp-size $TP --ep-size $EP \
  --host 0.0.0.0 --port $PORT --mem-fraction-static $MEMFRAC \
  --context-length $CTXLEN --reasoning-parser $REASONING_PARSER \
  --max-running-requests $MAX_RUNNING --log-level info \
  --enable-dp-attention --dp-size $DP_SIZE --enable-dp-lm-head \
  --enable-layerwise-nvtx-marker \
  ${EXTRA[*]} \
  --watchdog-timeout 600"

if [[ -z "$SKIP_BOOT" ]]; then
  echo "[boot] $RUN_TAG (TP=$TP EP=$EP DP=$DP_SIZE spec=$SPEC overlap=$OVERLAP cuda_graph=$CUDA_GRAPH)"
  echo "[boot] model: $MODEL"
  echo "[boot] cmd: $SERVE_CMD"
  echo "[boot] log: $LOG"
  mkdir -p "$(dirname "$LOG")" "$PROFILE_DIR"
  pkill -9 -f "[s]glang.launch_server" 2>/dev/null
  pkill -9 -f "[s]glang serve" 2>/dev/null
  pkill -9 -f "sglang::" 2>/dev/null
  tmux kill-session -t nemo_srv 2>/dev/null
  sleep 3
  tmux new-session -d -s nemo_srv "$SERVE_CMD >>'$LOG' 2>&1"
  echo -n "[boot] waiting for ready"
  for i in $(seq 1 90); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo " READY"; break; }
    pgrep -f "[s]glang serve" >/dev/null || pgrep -f "[s]glang.launch_server" >/dev/null || {
      echo " DIED — see $LOG"; tail -30 "$LOG"; exit 1; }
    echo -n "."; sleep 10
  done
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
    echo "[boot] not ready after ~900s — see $LOG"; exit 1; }
fi

cd /sgl-workspace/sglang

echo "[gsm8k] N=$N parallel=$PARALLEL port=$PORT"
python -m sglang.test.few_shot_gsm8k --num-questions "$N" --parallel "$PARALLEL" --port "$PORT" 2>&1 \
  | grep -iE "accuracy|invalid|latency"

if [[ "$PROFILE" == "1" ]]; then
  echo "[profile] capturing $PROFILE_STEPS decode steps -> $PROFILE_DIR"
  python -m sglang.test.send_one --profile --profile-steps "$PROFILE_STEPS" --port "$PORT" 2>&1 | tail -5
  echo "[profile] traces under: $PROFILE_DIR"
  ls -lh "$PROFILE_DIR" 2>/dev/null | tail -20
fi

echo "[done] server still in tmux 'nemo_srv' (tmux kill-session -t nemo_srv to stop); log: $LOG"
