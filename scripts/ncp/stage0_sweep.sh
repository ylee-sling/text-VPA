#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage0_sweep.sh -- a diagnostic after Stage 0, outside the protocol: the trained realizer at more steps than
# the depth budget allows, on MBPP's validation split (90 problems; the test sets stay untouched).
#
#   bash scripts/ncp/stage0_sweep.sh          start in a detached tmux session ("<run>-sweep") and return
#   bash scripts/ncp/stage0_sweep.sh run      run in the foreground
#
# It answers what kind of failure H2 is. If Pass@1 rises steeply with K, the denoiser is sound and few-step
# realization is the problem (the plan's job, Prop. 1, or distillation's). If even K = 128 stays poor, the
# adaptation itself is too weak. Caveat: the realizer was trained at t = k/K and h = 1/K for K in {2, 4, 8}
# only, so K > 8 also feeds it (t, h) values it has never seen.
#
#   realizer  K = 8, 16, 32, 64, 128, exact jump kernel and confidence order, 5 samples per problem
#   baseline  the released model left to right, the same 5 samples per problem
#   results   RUN/sweep/report.md (Pass@1 table) and RUN/sweep/analysis.md (how the samples failed)
#
# Variables: RUN [runs/stage0], DATA [data2], MODEL [google/gemma-4-E2B], K_STEPS ["8 16 32 64 128"],
# NUM_SAMPLES [5], PY [.venv/bin/python]. Rerunning skips finished configurations. Never shuts the server down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN="${RUN:-runs/stage0}"
DATA="${DATA:-data2}"
MODEL="${MODEL:-google/gemma-4-E2B}"
K_STEPS="${K_STEPS:-8 16 32 64 128}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
PY="${PY:-$ROOT/.venv/bin/python}"
SESSION="${SESSION:-$(basename "$RUN")-sweep}"
OUT="$RUN/sweep"
LOG="$OUT/sweep.log"

die() { echo "error: $*" >&2; exit 1; }

run() {
  [[ -x "$PY" ]] || die "$PY is missing: this script uses the repository's .venv"
  [[ -f "$RUN/last.pt" ]] || die "$RUN/last.pt is missing: run the Stage 0 training first"
  if command -v tmux >/dev/null && tmux has-session -t "=$(basename "$RUN")" 2>/dev/null; then
    die "the Stage 0 run '$(basename "$RUN")' is still running; start the sweep after it has finished"
  fi
  local model_dir
  model_dir="$DATA/models/$(basename "$MODEL")"
  [[ -f "$model_dir/.complete" ]] || die "$model_dir is missing: the Stage 0 run downloads it"
  local workers=$(( $(nproc) < 8 ? $(nproc) : 8 ))
  read -r -a ks <<< "$K_STEPS"
  local common=(--model "$model_dir" --eval-dir "$DATA/eval" --out "$OUT" --benchmarks mbpp
                --mbpp-split validation --num-samples "$NUM_SAMPLES" --exec-workers "$workers")
  echo "[$(date '+%F %T')] sweep: K = ${ks[*]}, MBPP validation, $NUM_SAMPLES samples per problem"
  "$PY" -m text_vpa.stage0.evaluate --decoder ar --canvas-len 256 "${common[@]}"
  "$PY" -m text_vpa.stage0.evaluate --decoder realizer --checkpoint "$RUN/last.pt" --k-steps "${ks[@]}" \
    --orders random confidence "${common[@]}"
  "$PY" -m text_vpa.stage0.report --eval "$OUT" --no-verdict >/dev/null
  "$PY" -m text_vpa.stage0.analyze --eval "$OUT" >/dev/null
  echo "[$(date '+%F %T')] sweep finished: $OUT/report.md, $OUT/analysis.md"
}

launch() {
  mkdir -p "$OUT"
  command -v tmux >/dev/null || die "tmux is not installed; run: bash scripts/ncp/stage0_sweep.sh run"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then die "tmux session '$SESSION' already exists"; fi
  local env_args=() name
  for name in RUN DATA MODEL K_STEPS NUM_SAMPLES PY; do
    env_args+=(-e "$name=${!name}")
  done
  tmux new-session -d -s "$SESSION" -c "$ROOT" "${env_args[@]}" \
    "bash scripts/ncp/stage0_sweep.sh run 2>&1 | tee -a '$LOG'"
  echo "started in tmux session '$SESSION'"
  echo "  log:     tail -f $LOG"
  echo "  results: $OUT/report.md and $OUT/analysis.md"
}

case "${1:-launch}" in
  launch) launch ;;
  run)    mkdir -p "$OUT"; run ;;
  *)      die "usage: stage0_sweep.sh [launch|run]" ;;
esac
