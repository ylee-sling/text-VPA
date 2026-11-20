#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage0_sweep.sh -- a diagnostic after Stage 0, outside the protocol, on MBPP's validation split (90 problems;
# the test sets stay untouched). It splits the realizer's gap to the baseline into three parts. Stages (STAGES
# runs them in the order given):
#
#   baseline     the released model's own generate.
#   code         code check: the realizer's code path with the adapters off, causal masks and left_to_right at
#                K = 256 (exactly one token per step) is the backbone's own decoding: it must match the baseline
#                (on the tiny Gemma 4 it reproduces greedy `generate` token for token). If it does not, there is
#                a bug and every Stage 0 number has to be redone.
#   adaptation   cost of the adaptation: the trained realizer, left_to_right at K = 256, against the code check.
#                With both evaluated, report.md decides direction A's one-token-per-step gate (report.py).
#   matched      the same with (t, h) as seen in training (h = 1/8, t matched to the fraction already revealed),
#                and the exact jump kernel at K = 32 likewise: separates the step-size input from the adaptation.
#   l2r          cost of parallel decisions, contiguous: left_to_right at K_STEPS (K = 128: one token per step
#                for the first 42 positions, then 2 to 3; K = 8: blocks of 5, 15, 24, 32).
#   parallel     cost of parallel decisions: the exact jump kernel and the confidence order at K_STEPS.
#
# The first run's realizer was trained at t = k/K and h = 1/K for K in {2, 4, 8} only, so K > 8 also feeds it
# (t, h) values it has never seen. A left-to-right row stops at its <eos> or once the truncation rule has found
# its stop in the revealed prefix; neither changes the program.
#
#   bash scripts/ncp/stage0_sweep.sh          start in a detached tmux session ("<run>-sweep") and return
#   bash scripts/ncp/stage0_sweep.sh run      run in the foreground (stage0.sh does this with VALIDATION_ONLY=1)
#
# Results: RUN/sweep/report.md (Pass@1 table, the gate) and RUN/sweep/analysis.md (how the samples failed, the
# def-line check), both rewritten after each stage, so partial results can be read while the sweep runs.
# Rerunning skips finished configurations.
#
# Variables: RUN [runs/stage0], DATA [data2], MODEL [google/gemma-4-E2B], K_STEPS ["8 32 128"],
# NUM_SAMPLES [5], STAGES ["baseline code adaptation matched l2r parallel"], LIMIT [] (first problems only, for
# a smoke run), PY [.venv/bin/python]. Never shuts the server down.
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
STAGES="${STAGES:-baseline code adaptation matched l2r parallel}"
LIMIT="${LIMIT:-}"
PY="${PY:-$ROOT/.venv/bin/python}"
SESSION="${SESSION:-$(basename "$RUN")-sweep}"
OUT="$RUN/sweep"
LOG="$OUT/sweep.log"
KNOWN_STAGES="baseline code adaptation matched l2r parallel"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

summarize() {
  "$PY" -m text_vpa.stage0.report --eval "$OUT" --no-verdict >/dev/null
  "$PY" -m text_vpa.stage0.analyze --eval "$OUT" --mbpp "$DATA/eval/mbpp.jsonl" >/dev/null
  log "tables updated: $OUT/report.md, $OUT/analysis.md"
}

run() {
  [[ -x "$PY" ]] || die "$PY is missing: this script uses the repository's .venv"
  [[ -f "$RUN/last.pt" ]] || die "$RUN/last.pt is missing: run the Stage 0 training first"
  local run_session own=""
  run_session="$(basename "$RUN")"
  if [[ -n "${TMUX:-}" ]]; then own="$(tmux display-message -p '#S' 2>/dev/null || true)"; fi
  # Inside the run's own session (stage0.sh with VALIDATION_ONLY=1) the training has finished by now.
  if [[ "$own" != "$run_session" ]] && command -v tmux >/dev/null && tmux has-session -t "=$run_session" 2>/dev/null
  then
    die "the Stage 0 run '$run_session' is still running; start the sweep after it has finished"
  fi
  local model_dir
  model_dir="$DATA/models/$(basename "$MODEL")"
  [[ -f "$model_dir/.complete" ]] || die "$model_dir is missing: the Stage 0 run downloads it"
  local workers=$(( $(nproc) < 8 ? $(nproc) : 8 ))
  local ks stages stage
  read -r -a ks <<< "$K_STEPS"
  read -r -a stages <<< "$STAGES"
  for stage in "${stages[@]}"; do
    [[ " $KNOWN_STAGES " == *" $stage "* ]] || die "unknown stage '$stage' (known: $KNOWN_STAGES)"
  done
  local common=(--model "$model_dir" --eval-dir "$DATA/eval" --out "$OUT" --benchmarks mbpp
                --mbpp-split validation --num-samples "$NUM_SAMPLES" --exec-workers "$workers")
  if [[ -n "$LIMIT" ]]; then common+=(--limit "$LIMIT"); fi
  local realizer=(--decoder realizer --checkpoint "$RUN/last.pt")
  log "sweep of $RUN: MBPP validation, $NUM_SAMPLES samples per problem, K = ${ks[*]}, stages: ${stages[*]}"

  local i=0
  for stage in "${stages[@]}"; do
    i=$((i + 1))
    case "$stage" in
      baseline)
        log "$i/${#stages[@]} baseline: the released model's own generate"
        "$PY" -m text_vpa.stage0.evaluate --decoder ar --canvas-len 256 "${common[@]}" ;;
      code)
        log "$i/${#stages[@]} code check: the backbone through the realizer's code path (adapters off, causal, one token per step)"
        "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right --canvas-mode causal \
          --no-adapters "${common[@]}" ;;
      adaptation)
        log "$i/${#stages[@]} cost of the adaptation: the trained realizer, one token per step"
        "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right "${common[@]}" ;;
      matched)
        log "$i/${#stages[@]} (t, h) as seen in training: one token per step, and the exact jump kernel at K = 32"
        "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right --time-input matched \
          "${common[@]}"
        "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps 32 --orders random --time-input matched \
          "${common[@]}" ;;
      l2r)
        log "$i/${#stages[@]} the trained realizer, left_to_right at K = ${ks[*]}"
        "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders left_to_right "${common[@]}" ;;
      parallel)
        log "$i/${#stages[@]} the trained realizer, exact jump kernel and confidence order at K = ${ks[*]}"
        "$PY" -m text_vpa.stage0.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders random confidence \
          "${common[@]}" ;;
    esac
    summarize
  done
  log "sweep finished: $OUT/report.md, $OUT/analysis.md"
}

launch() {
  mkdir -p "$OUT"
  command -v tmux >/dev/null || die "tmux is not installed; run: bash scripts/ncp/stage0_sweep.sh run"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then die "tmux session '$SESSION' already exists"; fi
  local env_args=() name
  for name in RUN DATA MODEL K_STEPS NUM_SAMPLES STAGES LIMIT PY; do
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
