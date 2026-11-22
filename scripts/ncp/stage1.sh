#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage1.sh -- the oracles of the planner stage (docs/stage1.md) on the NCP server (one NVIDIA L4), resumable at
# every step. PLAN=oracle-structure (step 1, the default): the realizer is told the reference program's line
# structure. PLAN=oracle-latent (step 2): it is told the frozen backbone's states of the reference's lines. Both are
# diagnostics ("oracle: not the method"). Nothing of them is a result of the method, and nothing touches a test set.
#
#   PLAN=oracle-latent bash scripts/ncp/stage1.sh            step 2 into runs/stage1-latent
#   PLAN=oracle-latent TRAIN_ADAPTERS=0 bash scripts/ncp/stage1.sh   the same with the realizer's adapters frozen
#
#   checks -> preflight -> oracle training from the Stage 0 checkpoint -> evaluation on MBPP validation -> report
#
#   bash scripts/ncp/stage1.sh            start in a detached tmux session named after the run and return
#   bash scripts/ncp/stage1.sh run        run in the foreground (what the tmux session executes)
#   bash scripts/ncp/stage1.sh status     where the run is: finished steps, last metrics, report, GPU
#
#   SMOKE=1 bash scripts/ncp/stage1.sh    the whole pipeline in miniature (30 steps, 8 problems, 2 samples, its
#                                         own run directory and its own baseline): a check of the server, of the
#                                         memory setting and of the speed before the long run
#
# Re-running the same command continues: finished steps are skipped (markers in RUN/done/), training resumes from
# RUN/last.pt, and every evaluated configuration is skipped once its summary.json exists.
#
# Evaluation stages (STAGES runs them in the order given; report.md is rewritten after each):
#   gate        the oracle plan, exact jump kernel, at K_STEPS. K = 8 is the row of the gate after step 1.
#   delivery    the oracle plan, left to right, one token per step (K = N): do the samples follow their plans?
#               A ratio under 0.5 at the gate is read as a statement about structure only if they do.
#   confidence  the oracle plan, confidence order, at K_STEPS.
#   null        the same weights with the null plan: exact kernel at K_STEPS, and one token per step.
#   shuffled    (not run by default) the plan of another problem's reference, exact kernel at K_STEPS.
# The baseline (the released model's own generate) and the realizer without a plan come from REFERENCE, the
# Stage 0 sweep of the checkpoint the oracle starts from, if it holds the same problems and samples; otherwise
# the baseline is evaluated into this run's own sweep.
#
# This script never shuts the server down and does not read SHUTDOWN_ON_SUCCESS / SHUTDOWN_ON_FAILURE.
# It uses the repository's .venv only (no activation, no other interpreter), clears PYTHONPATH and PYTHONHOME
# (ROS 2 Humble's Python 3.10 paths must not reach Python 3.12), installs nothing, and touches nothing outside
# the repository. Never `git pull` while it runs: bash reads a running script from disk.
#
# Generated programs are executed by the Pass@k harness in resource-limited processes. That is isolation, not a
# security sandbox: they run as this user.
#
# Variables (defaults in brackets):
#   PLAN         [oracle-structure]     oracle-structure (step 1) or oracle-latent (step 2)
#   RUN          [runs/stage1-oracle; latent: runs/stage1-latent; smoke: ...-smoke]   checkpoints, metrics, results
#   TRAIN_ADAPTERS [1]                  0: the realizer's own parts stay at INIT (only the bridge and oracle train)
#   STRUCTURE_RUN [latent: runs/stage1-oracle; smoke: none]   the structure oracle's run, read by step 2's gate
#   INIT         [runs/stage0-a/last.pt] the Stage 0 checkpoint the oracle starts from (its recipe is kept)
#   REFERENCE    [runs/stage0-a/sweep]  that checkpoint's sweep: baseline `ar` and the realizer without a plan
#   DATA         [data2]                DATA/corpus, DATA/eval, DATA/models
#   MODEL        [google/gemma-4-E2B]   the backbone (must already be under DATA/models: Stage 0 downloaded it)
#   STEPS        [5000]                 optimizer steps (32 examples each); a finished run continues with a larger
#                                       STEPS at the floor learning rate
#   TRAIN_SET    []                     extra Stage1Config overrides for a new run, e.g. "plan_dropout=0.0"
#   STAGES       [gate delivery confidence null]   evaluation stages, in order
#   K_STEPS      [8 32 128]             step counts of the parallel stages
#   NUM_SAMPLES  [5; smoke: 2]          samples per problem
#   LIMIT        [; smoke: 8]           first problems only
#   CODEBOOK_SAMPLE [200000; smoke: 5000] corpus rows of the preflight's skeleton codebook
#   CANVAS_LEN   [256]                  N, checked against the checkpoint by the preflight
#   ALLOW_BUSY_GPU [0]                  1: start although another process holds GPU memory
#   SKIP_TESTS   [0]                    1: skip the unit tests
#   REQUIRE_GPU  [1]                    0: run on the CPU (dry runs of this script with a tiny checkpoint)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ROS 2 and any other environment's Python paths stay out, whatever the calling shell or tmux server holds.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SMOKE="${SMOKE:-0}"
PLAN="${PLAN:-oracle-structure}"
case "$PLAN" in
  oracle-structure) RUN_NAME="stage1-oracle"; STRUCTURE_DEFAULT="" ;;
  oracle-latent)    RUN_NAME="stage1-latent"; STRUCTURE_DEFAULT="runs/stage1-oracle" ;;
  *) echo "error: PLAN must be oracle-structure or oracle-latent" >&2; exit 1 ;;
esac
TRAIN_ADAPTERS="${TRAIN_ADAPTERS:-1}"
[[ "$TRAIN_ADAPTERS" == 0 || "$TRAIN_ADAPTERS" == 1 ]] || { echo "error: TRAIN_ADAPTERS must be 0 or 1" >&2; exit 1; }
if [[ "$SMOKE" == 1 ]]; then
  STRUCTURE_DEFAULT=""                             # a smoke run's held-out set is not the structure run's
  RUN="${RUN:-runs/$RUN_NAME-smoke}"; STEPS="${STEPS:-30}"; NUM_SAMPLES="${NUM_SAMPLES:-2}"; LIMIT="${LIMIT:-8}"
  K_STEPS="${K_STEPS:-8 32}"; CODEBOOK_SAMPLE="${CODEBOOK_SAMPLE:-5000}"
  REFERENCE="${REFERENCE:-}"                       # the smoke run scores its own baseline on its 8 problems
  SMOKE_SET="save_every=15 eval_every=15 log_every=5 warmup_steps=5 val_examples=32 probe_samples=1"
else
  RUN="${RUN:-runs/$RUN_NAME}"; STEPS="${STEPS:-5000}"; NUM_SAMPLES="${NUM_SAMPLES:-5}"; LIMIT="${LIMIT:-}"
  K_STEPS="${K_STEPS:-8 32 128}"; CODEBOOK_SAMPLE="${CODEBOOK_SAMPLE:-200000}"
  REFERENCE="${REFERENCE-runs/stage0-a/sweep}"
  SMOKE_SET=""
fi
STRUCTURE_RUN="${STRUCTURE_RUN-$STRUCTURE_DEFAULT}"
INIT="${INIT:-runs/stage0-a/last.pt}"
DATA="${DATA:-data2}"
MODEL="${MODEL:-google/gemma-4-E2B}"
TRAIN_SET="${TRAIN_SET:-}"
STAGES="${STAGES:-gate delivery confidence null}"
CANVAS_LEN="${CANVAS_LEN:-256}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
export REQUIRE_GPU="${REQUIRE_GPU:-1}"
PY="${PY:-$ROOT/.venv/bin/python}"
SESSION="${SESSION:-$(basename "$RUN")}"
LOG="$RUN/stage1.log"
KNOWN_STAGES="gate delivery confidence null shuffled"
LABEL="oracle: not the method"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "error: $*" >&2; exit 1; }
done_marker() { [[ -f "$RUN/done/$1" ]]; }
mark() { mkdir -p "$RUN/done" && date '+%F %T' > "$RUN/done/$1"; }

status() {
  echo "run: $RUN   ($LABEL)"
  echo "finished steps: $(ls "$RUN/done" 2>/dev/null | tr '\n' ' ')"
  if [[ -f "$RUN/memory.json" ]]; then echo "memory check: $(tr -d '\n' < "$RUN/memory.json" | tr -s ' ')"; fi
  if [[ -f "$RUN/preflight.json" && -x "$PY" ]]; then
    "$PY" - "$RUN/preflight.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
o, m, k, lines = p.get("offsets", {}), p.get("mbpp_validation", {}), p.get("skeletons", {}), p.get("lines", {})
print(f"preflight: token offsets give the same slots as decoded prefixes for {o.get('corpus_same_slots', 0):.1%} "
      f"of {o.get('corpus_programs')} corpus canvases and {o.get('mbpp_same_slots', 0):.1%} of MBPP's references")
print(f"  MBPP validation references: {m.get('normalised')}/{m.get('problems')} normalise, pass "
      f"{m.get('pass_as_written')} as written and {m.get('pass_normalised')} normalised; indented in fours "
      f"{m.get('indented_in_fours_as_written', 0):.0%} -> {m.get('indented_in_fours_normalised', 0):.0%}; "
      f"within {p.get('max_slots')} slots {m.get('within_slots', 0):.0%}")
print("  corpus within slots: " + ", ".join(f"{name.split('_', 1)[1]} {v.get('within_slots', 0):.1%} "
                                             f"(indented in fours {v.get('indented_in_fours', 0):.1%})"
                                             for name, v in lines.items()))
for key in ("with_depth", "without_depth"):
    v = k.get(key, {})
    print(f"  skeletons {key.replace('_', ' ')}: codebook from {v.get('programs')} programs ({v.get('distinct')} "
          f"distinct); covers held-out corpus {v.get('held_out', 0):.1%}, MBPP validation "
          f"{v.get('mbpp_validation', 0):.1%}")
print(f"  {k.get('rule', '')} (for step 3; read it on the full run's codebook, not a smoke run's)")
if p.get("problems"):
    print("  PROBLEMS: " + "; ".join(p["problems"]))
EOF
  fi
  if [[ -f "$RUN/metrics.jsonl" && -x "$PY" ]]; then
    "$PY" - "$RUN/metrics.jsonl" "$(dirname "$INIT")" "$RUN/config.json" <<'EOF'
import json, os, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
# The gates start closed, so the evaluation at step 0 must repeat the last evaluation of the Stage 0 run the
# oracle started from (same held-out functions, same draws, same probe seeds), if both evaluate the same things.
start = next((r for r in rows if r["step"] == 0 and "eval/val_loss" in r), None)
init_metrics, init_config = os.path.join(sys.argv[2], "metrics.jsonl"), os.path.join(sys.argv[2], "config.json")
if start and os.path.exists(init_metrics):
    before = [r for r in map(json.loads, open(init_metrics, encoding="utf-8")) if "eval/val_loss" in r]
    keys = ("val_examples", "probe_samples", "probe_steps")
    mine = json.load(open(sys.argv[3])) if os.path.exists(sys.argv[3]) else {}
    theirs = json.load(open(init_config)) if os.path.exists(init_config) else {}
    differ = [f"{key} {mine.get(key)} against {theirs.get(key)}" for key in keys if mine.get(key) != theirs.get(key)]
    if before and differ:
        print("step 0 against the Stage 0 run: not comparable, this run evaluates other things ("
              + ", ".join(differ) + "; a smoke run)")
    elif before:
        pairs = [(key, start[key], before[-1][key]) for key in ("eval/val_loss", "eval/val_loss_next",
                                                                "eval/probe_pass@1") if key in start and key in before[-1]]
        same = all(abs(a - b) <= 2e-3 * max(abs(b), 1.0) for _, a, b in pairs)
        print(f"step 0 against the Stage 0 run's last evaluation (step {before[-1]['step']}): "
              + ", ".join(f"{key.split('/')[1]} {a:.4f} / {b:.4f}" for key, a, b in pairs)
              + ("   -> the same" if same else "   -> DIFFERENT: a bug, stop the run (docs/stage1.md)"))
train = [r for r in rows if "train/loss" in r]
if train:
    r = train[-1]
    print(f"training: step {r['step']}, loss {r['train/loss']:.3f}, {r['train/steps_per_s']:.3f} steps/s, "
          f"{r['train/eta_h']:.1f} h to go, mean gate {r.get('train/gate_mean', 0):.3f} (max "
          f"{r.get('train/gate_max', 0):.3f})")
evals = {r["step"]: r for r in rows if "eval/plan_gain_k0" in r}
if evals:
    print("held-out loss with everything masked, nats: step: oracle plan / null plan / another function's plan")
    for step, r in sorted(evals.items()):
        print(f"  {step}: {r['eval/val_loss_k0']:.3f} / {r['eval/val_loss_null_k0']:.3f} / "
              f"{r.get('eval/val_loss_shuffled_k0', float('nan')):.3f}")
probes = {r["step"]: r for r in rows if "eval/probe_pass@1" in r}
if probes:
    print("probe (MBPP validation, K = 8, oracle plan): Pass@1, share of lines with the planned kind"
          + (", share reproducing the reference" if any("eval/probe_reproduced" in r for r in probes.values()) else ""))
    print("  " + ", ".join(f"{s}: {100 * r['eval/probe_pass@1']:.1f}% / {r.get('eval/probe_kind_match', 0):.2f}"
                           + (f" / {100 * r['eval/probe_reproduced']:.1f}%" if "eval/probe_reproduced" in r else "")
                           for s, r in sorted(probes.items())))
EOF
  fi
  if [[ -f "$RUN/sweep/report.md" ]]; then echo; cat "$RUN/sweep/report.md"; fi
  if command -v tmux >/dev/null && tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is running"
  else
    echo "no tmux session '$SESSION'"
  fi
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
  fi
  if [[ -f "$LOG" ]]; then echo "--- last log lines ($LOG)"; tail -n 5 "$LOG"; fi
}

launch() {
  mkdir -p "$RUN"
  [[ -x "$PY" ]] || die "$PY is missing: this script uses the repository's .venv"
  if command -v tmux >/dev/null; then
    # "=name" matches the session name exactly: a bare -t matches by prefix.
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      die "tmux session '$SESSION' already exists (tmux attach -t '=$SESSION')"
    fi
    # The variables are passed explicitly (-e, tmux >= 3.2): a running tmux server would otherwise hand its
    # own, older environment to the new session. REFERENCE may be set and empty (no reference), so it is
    # passed whenever it is set.
    local env_args=() name
    for name in SMOKE PLAN TRAIN_ADAPTERS RUN INIT DATA MODEL STEPS TRAIN_SET STAGES K_STEPS NUM_SAMPLES LIMIT \
                CODEBOOK_SAMPLE \
                CANVAS_LEN ALLOW_BUSY_GPU SKIP_TESTS REQUIRE_GPU PY HF_TOKEN; do
      if [[ -n "${!name:-}" ]]; then env_args+=(-e "$name=${!name}"); fi
    done
    env_args+=(-e "REFERENCE=$REFERENCE" -e "STRUCTURE_RUN=$STRUCTURE_RUN")
    tmux new-session -d -s "$SESSION" -c "$ROOT" "${env_args[@]}" \
      "bash scripts/ncp/stage1.sh run 2>&1 | tee -a '$LOG'"
    echo "started in tmux session '$SESSION'   ($LABEL)"
    echo "  watch:   tmux attach -t '=$SESSION'      (detach: Ctrl-b d)"
  else
    nohup setsid bash scripts/ncp/stage1.sh run >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (tmux is not installed), pid $!"
  fi
  echo "  log:     tail -f $LOG        (Ctrl-C ends the tail, not the run)"
  echo "  status:  RUN=$RUN bash scripts/ncp/stage1.sh status"
  echo "  do not 'git pull' until the run has finished"
}

summarize() {
  # shellcheck disable=SC2086
  "$PY" -m text_vpa.stage1.report --eval "$RUN/sweep" --run "$RUN" $REFERENCE_ARGS $STRUCTURE_ARGS >/dev/null
  "$PY" -m text_vpa.stage0.analyze --eval "$RUN/sweep" --mbpp "$DATA/eval/mbpp.jsonl" >/dev/null
  log "tables updated: $RUN/sweep/report.md, $RUN/sweep/analysis.md"
}

run() {
  mkdir -p "$RUN/done"
  trap 'log "stage 1 stopped with an error (line $LINENO); rerun the same command to continue"' ERR
  log "Stage 1, $PLAN ($LABEL): RUN=$RUN INIT=$INIT REFERENCE='$REFERENCE' STRUCTURE_RUN='$STRUCTURE_RUN'" \
    "DATA=$DATA STEPS=$STEPS samples=$NUM_SAMPLES K='$K_STEPS' stages='$STAGES' smoke=$SMOKE" \
    "train_adapters=$TRAIN_ADAPTERS train_set='$TRAIN_SET'"
  STRUCTURE_ARGS=""
  if [[ "$PLAN" == oracle-latent && -n "$STRUCTURE_RUN" ]]; then
    [[ -f "$STRUCTURE_RUN/metrics.jsonl" ]] || die "$STRUCTURE_RUN/metrics.jsonl is missing: step 2's gate reads the structure oracle's run"
    STRUCTURE_ARGS="--structure-run $STRUCTURE_RUN"
  fi
  [[ -x "$PY" ]] || die "$PY is missing: this script uses the repository's .venv"
  local stage
  read -r -a STAGE_LIST <<< "$STAGES"
  for stage in "${STAGE_LIST[@]}"; do
    [[ " $KNOWN_STAGES " == *" $stage "* ]] || die "unknown stage '$stage' (known: $KNOWN_STAGES)"
  done
  read -r -a KS <<< "$K_STEPS"
  LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
  EXEC_WORKERS="$(( $(nproc) < 8 ? $(nproc) : 8 ))"
  MODEL_DIR="$DATA/models/$(basename "$MODEL")"

  # ---- 1. environment ---------------------------------------------------------------------------
  "$PY" - <<'EOF'
import sys
assert sys.version_info[:2] >= (3, 12), f"the .venv runs Python {sys.version.split()[0]}; this stage expects 3.12"
stray = [p for p in sys.path if "/opt/ros" in p or "python3.10" in p]
assert not stray, f"foreign paths on sys.path: {stray}"
import os, torch, transformers
assert hasattr(transformers, "Gemma4ForConditionalGeneration"), "transformers has no Gemma 4: run Stage 0's script"
if os.environ.get("REQUIRE_GPU", "1") != "1":
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, no GPU required (REQUIRE_GPU=0)")
    sys.exit(0)
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).float().sum().item()
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
      f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
EOF
  if [[ "$ALLOW_BUSY_GPU" != 1 && "$REQUIRE_GPU" == 1 ]]; then
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    [[ "${USED:-0}" -lt 1024 ]] || die "another process holds ${USED} MiB of GPU memory (nvidia-smi); stop it or set ALLOW_BUSY_GPU=1"
  fi

  # ---- 2. what must already be on the server ----------------------------------------------------
  [[ -f "$INIT" ]] || die "$INIT is missing: the oracle starts from the Stage 0 decoder's checkpoint"
  [[ -f "$MODEL_DIR/.complete" ]] || die "$MODEL_DIR is missing: the Stage 0 run downloads it"
  ls "$DATA"/corpus/train-*.jsonl.gz >/dev/null 2>&1 || die "$DATA/corpus has no train-*.jsonl.gz shards"
  [[ -s "$DATA/eval/mbpp.jsonl" ]] || die "$DATA/eval/mbpp.jsonl is missing"

  # ---- 3. tests ---------------------------------------------------------------------------------
  if [[ "$SKIP_TESTS" != 1 ]] && ! done_marker tests; then
    log "unit tests (stage 1, then the whole fast suite)"
    "$PY" -m pytest tests/test_stage1.py tests/test_stage1_cli.py -q -p no:cacheprovider
    "$PY" -m pytest -q -p no:cacheprovider
    mark tests
  fi

  # ---- 4. preflight -----------------------------------------------------------------------------
  if ! done_marker preflight; then
    log "preflight: token offsets, lines, MBPP's references as the oracle reads them, skeleton coverage"
    "$PY" -m text_vpa.stage1.preflight --model "$MODEL_DIR" --corpus "$DATA/corpus" --eval-dir "$DATA/eval" \
      --init "$INIT" --canvas-len "$CANVAS_LEN" --codebook-sample "$CODEBOOK_SAMPLE" \
      --exec-workers "$EXEC_WORKERS" --out "$RUN/preflight.json"
    mark preflight
  fi

  # ---- 5. training ------------------------------------------------------------------------------
  if ! done_marker "train-$STEPS"; then
    log "training the $PLAN to step $STEPS"
    if [[ -f "$RUN/last.pt" ]]; then
      "$PY" -m text_vpa.stage1.train --corpus "$DATA/corpus" --eval-dir "$DATA/eval" --run "$RUN" --steps "$STEPS"
    else
      # shellcheck disable=SC2086
      "$PY" -m text_vpa.stage1.train --init "$INIT" --corpus "$DATA/corpus" --eval-dir "$DATA/eval" --run "$RUN" \
        --set "model=$MODEL_DIR" "steps=$STEPS" "plan=$PLAN" "train_adapters=$([[ $TRAIN_ADAPTERS == 1 ]] && echo true || echo false)" \
        $SMOKE_SET $TRAIN_SET
    fi
    mark "train-$STEPS"
  fi

  # ---- 6. the baseline and the realizer without a plan ------------------------------------------
  # REFERENCE is used only if its baseline was scored on the problems and samples of this evaluation.
  REFERENCE_ARGS=""
  if [[ -n "$REFERENCE" ]] && "$PY" - "$REFERENCE/mbpp/ar/summary.json" "$DATA/eval" "$NUM_SAMPLES" "$LIMIT" <<'EOF'
import json, sys
from pathlib import Path
from text_vpa.stage0.evaluate import load_tasks
path, eval_dir, samples, limit = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
if not path.exists():
    sys.exit(1)
summary = json.loads(path.read_text())
problems = len(load_tasks("mbpp", eval_dir, "validation", int(limit) if limit else None))
ok = (summary.get("split") == "validation" and summary.get("problems") == problems
      and summary.get("samples_per_problem") == samples)
sys.exit(0 if ok else 1)
EOF
  then
    REFERENCE_ARGS="--reference $REFERENCE"
    log "baseline and no-plan rows: $REFERENCE"
  else
    [[ -z "$REFERENCE" ]] || log "REFERENCE=$REFERENCE holds no baseline on these problems and samples: not used"
    log "baseline: the released model's own generate, into $RUN/sweep"
    "$PY" -m text_vpa.stage0.evaluate --decoder ar --model "$MODEL_DIR" --eval-dir "$DATA/eval" --out "$RUN/sweep" \
      --benchmarks mbpp --mbpp-split validation --num-samples "$NUM_SAMPLES" --k 1 --canvas-len "$CANVAS_LEN" \
      --exec-workers "$EXEC_WORKERS" "${LIMIT_ARGS[@]}"
  fi

  # ---- 7. evaluation on MBPP's validation split -------------------------------------------------
  ONE="$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['canvas_len'])" "$RUN/config.json")"
  EVAL=("$PY" -m text_vpa.stage1.evaluate --checkpoint "$RUN/last.pt" --model "$MODEL_DIR" --eval-dir "$DATA/eval"
        --out "$RUN/sweep" --mbpp-split validation --num-samples "$NUM_SAMPLES" --exec-workers "$EXEC_WORKERS"
        "${LIMIT_ARGS[@]}")
  local i=0
  for stage in "${STAGE_LIST[@]}"; do
    i=$((i + 1))
    case "$stage" in
      gate)
        log "$i/${#STAGE_LIST[@]} gate: oracle plan, exact jump kernel, K = ${KS[*]}"
        "${EVAL[@]}" --plans oracle --orders random --k-steps "${KS[@]}" ;;
      delivery)
        log "$i/${#STAGE_LIST[@]} delivery: oracle plan, left to right, one token per step (K = $ONE)"
        "${EVAL[@]}" --plans oracle --orders left_to_right --k-steps "$ONE" ;;
      confidence)
        log "$i/${#STAGE_LIST[@]} oracle plan, confidence order, K = ${KS[*]}"
        "${EVAL[@]}" --plans oracle --orders confidence --k-steps "${KS[@]}" ;;
      null)
        log "$i/${#STAGE_LIST[@]} null plan (the same weights without a plan): exact kernel at K = ${KS[*]}, one token per step"
        "${EVAL[@]}" --plans null --orders random --k-steps "${KS[@]}"
        "${EVAL[@]}" --plans null --orders left_to_right --k-steps "$ONE" ;;
      shuffled)
        log "$i/${#STAGE_LIST[@]} another problem's plan, exact jump kernel, K = ${KS[*]}"
        "${EVAL[@]}" --plans shuffled --orders random --k-steps "${KS[@]}" ;;
    esac
    summarize
  done
  # shellcheck disable=SC2086
  "$PY" -m text_vpa.stage1.report --eval "$RUN/sweep" --run "$RUN" $REFERENCE_ARGS $STRUCTURE_ARGS
  log "stage 1, $PLAN ($LABEL) finished: $RUN/sweep/report.md, $RUN/sweep/analysis.md"
}

case "${1:-launch}" in
  launch) launch ;;
  run)    run ;;
  status) status ;;
  *)      die "usage: stage1.sh [launch|run|status]" ;;
esac
