#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage0_sweep.sh -- a diagnostic after Stage 0, outside the protocol: the trained realizer with three samplers
# and more steps than the depth budget allows, on MBPP's validation split (90 problems; the test sets stay
# untouched).
#
#   bash scripts/ncp/stage0_sweep.sh          start in a detached tmux session ("<run>-sweep") and return
#   bash scripts/ncp/stage0_sweep.sh run      run in the foreground
#
# It answers what kind of failure H2 is.
#   1. left_to_right at K = 128 reveals one token per step for the first 52 tokens: close to the adapted model
#      decoding autoregressively. Near the baseline: the implementation and the adaptation are sound, and the
#      failure is parallel realization. Far below it: the adaptation (or the code) is at fault; look there first.
#   2. left_to_right at K = 8 is a semi-autoregressive sampler inside the depth budget (blocks of 5, 15, 24, 32
#      tokens from the left).
#   3. random (the exact jump kernel) and confidence at K = 8, 32, 128: how fast quality rises with the steps.
# Caveat: the realizer was trained at t = k/K and h = 1/K for K in {2, 4, 8} only, so K > 8 also feeds it
# (t, h) values it has never seen.
#
# Results: RUN/sweep/report.md (Pass@1 table) and RUN/sweep/analysis.md (how the samples failed), both rewritten
# after each stage, so partial results can be read while the sweep runs. Rerunning skips finished configurations.
#
# Variables: RUN [runs/stage0], DATA [data2], MODEL [google/gemma-4-E2B], K_STEPS ["8 32 128"],
# NUM_SAMPLES [5], PY [.venv/bin/python]. Never shuts the server down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN="${RUN:-runs/stage0}"
DATA="${DATA:-data2}"
MODEL="${MODEL:-google/gemma-4-E2B}"
K_STEPS="${K_STEPS:-8 32 128}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
PY="${PY:-$ROOT/.venv/bin/python}"
SESSION="${SESSION:-$(basename "$RUN")-sweep}"
OUT="$RUN/sweep"
LOG="$OUT/sweep.log"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

summarize() {
  "$PY" -m text_vpa.stage0.report --eval "$OUT" --no-verdict >/dev/null
  "$PY" -m text_vpa.stage0.analyze --eval "$OUT" >/dev/null
  log "tables updated: $OUT/report.md, $OUT/analysis.md"
}

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
  local realizer=(--decoder realizer --checkpoint "$RUN/last.pt")
  log "sweep: MBPP validation, $NUM_SAMPLES samples per problem, K = ${ks[*]}"

  log "1/4 baseline, left to right"
  "$PY" -m text_vpa.stage0.evaluate --decoder ar --canvas-len 256 "${common[@]}"
  log "2/4 realizer, left_to_right at K = 128 (about one token per step) and K = 8"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 128 8 --orders left_to_right "${common[@]}"
  summarize
  log "3/4 realizer, exact jump kernel and confidence order at K = ${ks[*]}"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders random confidence "${common[@]}"
  summarize
  log "4/4 realizer, left_to_right at K = ${ks[*]}"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders left_to_right "${common[@]}"
  summarize
  log "sweep finished: $OUT/report.md, $OUT/analysis.md"
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
  echo "  results: $OUT/report.md and $OUT/analysis.md (rewritten after each stage)"
}

case "${1:-launch}" in
  launch) launch ;;
  run)    mkdir -p "$OUT"; run ;;
  *)      die "usage: stage0_sweep.sh [launch|run]" ;;
esac
