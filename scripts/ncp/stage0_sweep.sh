#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage0_sweep.sh -- a diagnostic after Stage 0, outside the protocol, on MBPP's validation split (90 problems;
# the test sets stay untouched). It splits the realizer's gap to the baseline into three parts:
#
#   1. code check            the realizer's code path with the adapters off, causal masks and left_to_right at
#                            K = 256 (exactly one token per step) is the backbone's own decoding: it must match
#                            the baseline (on the tiny Gemma 4 it reproduces greedy `generate` token for token).
#                            If it does not, there is a bug and every Stage 0 number has to be redone.
#   2. cost of the adaptation  the trained realizer, left_to_right at K = 256, against the code check. Training
#                            used uniformly random masks and never a revealed prefix with a masked suffix, so part
#                            of this cost can be that mismatch.
#   3. cost of parallel decisions  the trained realizer at fewer steps: left_to_right at K = 128 (one token per
#                            step for the first 42 positions, then 2 to 3) and K = 8 (blocks of 5, 15, 24, 32), and
#                            the exact jump kernel, the confidence order and left_to_right at K = 8, 32, 128.
#
# The realizer was trained at t = k/K and h = 1/K for K in {2, 4, 8} only, so K > 8 also feeds it (t, h) values
# it has never seen. A left-to-right row stops at its <eos> or once the truncation rule has found its stop in the
# revealed prefix; neither changes the program.
#
#   bash scripts/ncp/stage0_sweep.sh          start in a detached tmux session ("<run>-sweep") and return
#   bash scripts/ncp/stage0_sweep.sh run      run in the foreground
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

  log "1/6 baseline, the released model's own generate"
  "$PY" -m text_vpa.stage0.evaluate --decoder ar --canvas-len 256 "${common[@]}"
  log "2/6 code check: the backbone through the realizer's code path (adapters off, causal, one token per step)"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right --canvas-mode causal \
    --no-adapters "${common[@]}"
  summarize
  log "3/6 cost of the adaptation: the trained realizer, one token per step"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right "${common[@]}"
  summarize
  log "4/6 the trained realizer, left_to_right at K = 128 and K = 8"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 128 8 --orders left_to_right "${common[@]}"
  summarize
  log "5/6 the trained realizer, exact jump kernel and confidence order at K = ${ks[*]}"
  "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders random confidence "${common[@]}"
  summarize
  log "6/6 the trained realizer, left_to_right at K = ${ks[*]}"
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
