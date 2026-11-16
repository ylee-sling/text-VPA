#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage0.sh -- Stage 0 of the Text-VPA protocol on the NCP server (one NVIDIA L4), resumable at every step:
#
#   checks -> model download -> preflight -> harness check -> AR baseline -> training -> realizer at K = 2, 4, 8
#          -> report (H2) -> instruction-tuned reference row
#
#   bash scripts/ncp/stage0.sh            start in a detached tmux session named after the run and return
#   bash scripts/ncp/stage0.sh run        run in the foreground (what the tmux session executes)
#   bash scripts/ncp/stage0.sh status     where the run is: finished steps, last metrics, GPU
#
#   SMOKE=1 bash scripts/ncp/stage0.sh    the whole pipeline in miniature (30 steps, 8 problems, 2 samples) in its
#                                         own run directory: a check of the server before the long run
#
# Re-running the same command continues: finished steps are skipped (markers in RUN/done/), training resumes
# from RUN/last.pt, and every evaluated configuration is skipped once its summary.json exists.
#
# This script never shuts the server down and does not read SHUTDOWN_ON_SUCCESS / SHUTDOWN_ON_FAILURE.
# It uses the repository's .venv only (no activation, no other interpreter), clears PYTHONPATH and PYTHONHOME
# (ROS 2 Humble's Python 3.10 paths must not reach Python 3.12) and touches nothing outside the repository
# except the Hugging Face download cache under DATA/models.
#
# Generated programs are executed by the Pass@k harness in resource-limited processes. That is isolation, not a
# security sandbox: they run as this user.
#
# Variables (defaults in brackets):
#   RUN          [runs/stage0]          checkpoints, metrics, evaluation results, report
#   DATA         [data2]                DATA/corpus (train-*.jsonl.gz, val.jsonl.gz), DATA/eval, DATA/models
#   MODEL        [google/gemma-4-E2B]   the pre-trained backbone (H2's baseline and the realizer's initialisation)
#   MODEL_IT     [google/gemma-4-E2B-it] the instruction-tuned checkpoint of the reference row (AR_IT=0 skips it)
#   STEPS        [25000]                optimizer steps (32 examples each); a finished run continues with a larger
#                                       STEPS at the floor learning rate (the cosine horizon stays where it was)
#   CANVAS_LEN   [256]                  N: canvas positions of the realizer and token budget of the AR baseline
#   NUM_SAMPLES  [10]                   samples per problem
#   ORDERS       [random confidence]    realizer samplers: the exact jump kernel (H2) and the confidence order
#   TRAIN_SET    []                     extra Stage0Config overrides for a new run, e.g. "lora_rank=32 lr=1e-4"
#   AR_IT        [1]                    also evaluate MODEL_IT through its chat template (after the report)
#   HF_TOKEN     [unset]                only needed if the Hub asks for it
#   ALLOW_BUSY_GPU [0]                  1: start although another process holds GPU memory
#   SKIP_TESTS   [0]                    1: skip the unit tests
#   REQUIRE_GPU  [1]                    0: run on the CPU (dry runs of this script with a tiny checkpoint)
#   PREFLIGHT_ARGS []                   extra arguments of text_vpa.stage0.preflight, e.g. --allow-mismatch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ROS 2 and any other environment's Python paths stay out, whatever the calling shell or tmux server holds.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == 1 ]]; then
  RUN="${RUN:-runs/stage0-smoke}"; STEPS="${STEPS:-30}"; NUM_SAMPLES="${NUM_SAMPLES:-2}"; LIMIT="${LIMIT:-8}"
  SMOKE_SET="save_every=15 eval_every=15 log_every=5 warmup_steps=5 val_examples=32 probe_samples=1"
else
  RUN="${RUN:-runs/stage0}"; STEPS="${STEPS:-25000}"; NUM_SAMPLES="${NUM_SAMPLES:-10}"; LIMIT="${LIMIT:-}"
  SMOKE_SET=""
fi
DATA="${DATA:-data2}"
MODEL="${MODEL:-google/gemma-4-E2B}"
MODEL_IT="${MODEL_IT:-google/gemma-4-E2B-it}"
CANVAS_LEN="${CANVAS_LEN:-256}"
ORDERS="${ORDERS:-random confidence}"
TRAIN_SET="${TRAIN_SET:-}"
AR_IT="${AR_IT:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
export REQUIRE_GPU="${REQUIRE_GPU:-1}"
PREFLIGHT_ARGS="${PREFLIGHT_ARGS:-}"
PY="${PY:-$ROOT/.venv/bin/python}"
SESSION="${SESSION:-$(basename "$RUN")}"
LOG="$RUN/stage0.log"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "error: $*" >&2; exit 1; }
done_marker() { [[ -f "$RUN/done/$1" ]]; }
mark() { mkdir -p "$RUN/done" && date '+%F %T' > "$RUN/done/$1"; }
model_dir() { echo "$DATA/models/$(basename "$1")"; }

status() {
  echo "run: $RUN"
  echo "finished steps: $(ls "$RUN/done" 2>/dev/null | tr '\n' ' ')"
  if [[ -f "$RUN/metrics.jsonl" && -x "$PY" ]]; then
    "$PY" - "$RUN/metrics.jsonl" <<'EOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
train = [r for r in rows if "train/loss" in r]
if train:
    r = train[-1]
    print(f"training: step {r['step']}, loss {r['train/loss']:.3f}, masked-token accuracy {r['train/accuracy']:.3f}, "
          f"{r['train/steps_per_s']:.3f} steps/s, {r['train/eta_h']:.1f} h to go")
probes = {r["step"]: r["eval/probe_pass@1"] for r in rows if "eval/probe_pass@1" in r}
if probes:
    print("probe (Pass@1 on MBPP validation, K = 8): "
          + ", ".join(f"{s}: {100 * v:.1f}%" for s, v in sorted(probes.items())))
EOF
  fi
  if [[ -f "$RUN/eval/report.md" ]]; then echo; cat "$RUN/eval/report.md"; fi
  if command -v tmux >/dev/null && tmux has-session -t "$SESSION" 2>/dev/null; then
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
    tmux has-session -t "$SESSION" 2>/dev/null && die "tmux session '$SESSION' already exists (tmux attach -t $SESSION)"
    # The variables are passed explicitly (-e, tmux >= 3.2): a running tmux server would otherwise hand its
    # own, older environment to the new session.
    local env_args=() name
    for name in SMOKE RUN DATA MODEL MODEL_IT STEPS CANVAS_LEN NUM_SAMPLES LIMIT ORDERS TRAIN_SET AR_IT \
                ALLOW_BUSY_GPU SKIP_TESTS REQUIRE_GPU PREFLIGHT_ARGS PY HF_TOKEN; do
      if [[ -n "${!name:-}" ]]; then env_args+=(-e "$name=${!name}"); fi
    done
    tmux new-session -d -s "$SESSION" -c "$ROOT" "${env_args[@]}" \
      "bash scripts/ncp/stage0.sh run 2>&1 | tee -a '$LOG'"
    echo "started in tmux session '$SESSION'"
    echo "  watch:   tmux attach -t $SESSION      (detach: Ctrl-b d)"
  else
    nohup setsid bash scripts/ncp/stage0.sh run >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (tmux is not installed), pid $!"
  fi
  echo "  log:     tail -f $LOG"
  echo "  status:  RUN=$RUN bash scripts/ncp/stage0.sh status"
}

download() {   # download <hub id> -> DATA/models/<name>, the revision recorded next to it
  local repo="$1" dest
  dest="$(model_dir "$repo")"
  if [[ -f "$dest/.complete" ]]; then log "model present: $dest ($(cat "$dest/.complete"))"; return 0; fi
  log "downloading $repo to $dest"
  "$PY" - "$repo" "$dest" <<'EOF'
import os, sys
from huggingface_hub import HfApi, snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or None
try:
    sha = HfApi().model_info(repo, token=token).sha
    snapshot_download(repo, revision=sha, local_dir=dest, token=token,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.md"])
except Exception as exc:  # gated or private repositories answer 401/403
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403) or type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
        sys.exit(f"the Hub refused {repo} ({type(exc).__name__}): open https://huggingface.co/{repo}, accept its "
                 "terms if it shows any, create a read token, and rerun with HF_TOKEN=hf_... set")
    raise
open(os.path.join(dest, ".complete"), "w").write(f"{repo}@{sha}\n")
print(f"{repo}@{sha}")
EOF
}

run() {
  mkdir -p "$RUN/done"
  trap 'log "Stage 0 stopped with an error (line $LINENO); rerun the same command to continue"' ERR
  log "Stage 0: RUN=$RUN DATA=$DATA MODEL=$MODEL STEPS=$STEPS N=$CANVAS_LEN samples=$NUM_SAMPLES smoke=$SMOKE"
  [[ -x "$PY" ]] || die "$PY is missing: this script uses the repository's .venv"
  LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
  read -r -a ORDER_ARGS <<< "$ORDERS"
  EXEC_WORKERS="$(( $(nproc) < 8 ? $(nproc) : 8 ))"

  # ---- 1. environment ---------------------------------------------------------------------------
  "$PY" - <<'EOF'
import sys
assert sys.version_info[:2] >= (3, 12), f"the .venv runs Python {sys.version.split()[0]}; Stage 0 expects 3.12"
stray = [p for p in sys.path if "/opt/ros" in p or "python3.10" in p]
assert not stray, f"foreign paths on sys.path: {stray}"
import os, torch
if os.environ.get("REQUIRE_GPU", "1") != "1":
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, no GPU required (REQUIRE_GPU=0)")
    sys.exit(0)
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
name = torch.cuda.get_device_name(0)
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).float().sum().item()
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, {name}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
EOF
  if [[ "$ALLOW_BUSY_GPU" != 1 && "$REQUIRE_GPU" == 1 ]]; then
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    [[ "${USED:-0}" -lt 1024 ]] || die "another process holds ${USED} MiB of GPU memory (nvidia-smi); stop it or set ALLOW_BUSY_GPU=1"
  fi

  # ---- 2. packages ------------------------------------------------------------------------------
  if ! "$PY" -c "import transformers, huggingface_hub, safetensors, tokenizers, numpy, pytest; transformers.Gemma4ForConditionalGeneration" 2>/dev/null; then
    log "installing the repository's pinned requirements into .venv (transformers with Gemma 4 is missing)"
    UV="$(command -v uv || true)"
    if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"; fi
    freeze() { "$PY" -m pip freeze 2>/dev/null || "$UV" pip freeze --python "$PY"; }
    freeze > "$RUN/packages-before.txt" || true
    if "$PY" -m pip --version >/dev/null 2>&1; then "$PY" -m pip install -r requirements-dev.txt
    elif [[ -n "$UV" ]]; then "$UV" pip install --python "$PY" -r requirements-dev.txt
    else die "the .venv has no pip and uv is not on PATH; install requirements-dev.txt into .venv by hand"; fi
    freeze > "$RUN/packages-after.txt"
    comm -13 <(sort "$RUN/packages-before.txt") <(sort "$RUN/packages-after.txt") > "$RUN/packages-added.txt" || true
    log "packages added or changed (also in $RUN/packages-added.txt):"; cat "$RUN/packages-added.txt"
  else
    log "packages: nothing to install ($("$PY" -c 'import transformers; print("transformers", transformers.__version__)'))"
  fi

  # ---- 3. data ----------------------------------------------------------------------------------
  ls "$DATA"/corpus/train-*.jsonl.gz >/dev/null 2>&1 || die "$DATA/corpus has no train-*.jsonl.gz shards"
  mkdir -p "$DATA/eval"
  [[ -s "$DATA/eval/HumanEval.jsonl.gz" ]] || curl -fsSL -o "$DATA/eval/HumanEval.jsonl.gz" \
    https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
  [[ -s "$DATA/eval/mbpp.jsonl" ]] || curl -fsSL -o "$DATA/eval/mbpp.jsonl" \
    https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl

  # ---- 4. tests ---------------------------------------------------------------------------------
  if [[ "$SKIP_TESTS" != 1 ]] && ! done_marker tests; then
    log "unit tests (Stage 0, then the whole fast suite)"
    "$PY" -m pytest tests/test_stage0.py tests/test_stage0_cli.py -q -p no:cacheprovider
    "$PY" -m pytest -q -p no:cacheprovider
    mark tests
  fi

  # ---- 5. model and preflight -------------------------------------------------------------------
  download "$MODEL"
  MODEL_DIR="$(model_dir "$MODEL")"
  if ! done_marker preflight; then
    # shellcheck disable=SC2086
    "$PY" -m text_vpa.stage0.preflight --model "$MODEL_DIR" --corpus "$DATA/corpus" --eval-dir "$DATA/eval" \
      --canvas-len "$CANVAS_LEN" --out "$RUN/preflight.json" $PREFLIGHT_ARGS
    mark preflight
  fi

  # ---- 6. harness check: canonical solutions through Stage 0's truncation -----------------------
  if ! done_marker harness; then
    "$PY" -m text_vpa.stage0.evaluate --decoder reference --eval-dir "$DATA/eval" --out "$RUN/harness" \
      --exec-workers "$EXEC_WORKERS" >/dev/null
    "$PY" - "$RUN/harness" <<'EOF'
import json, sys
he = json.load(open(f"{sys.argv[1]}/humaneval/reference/summary.json"))
mb = json.load(open(f"{sys.argv[1]}/mbpp/reference/summary.json"))
print(f"harness check: HumanEval {he['solved_problems']}/{he['problems']}, MBPP {mb['solved_problems']}/{mb['problems']}")
assert he["solved_problems"] == he["problems"] == 164, "HumanEval's canonical solutions must all pass"
assert mb["problems"] == 500 and mb["solved_problems"] >= 499, "MBPP's canonical solutions must pass (367 excepted)"
EOF
    mark harness
  fi

  EVAL_COMMON=(--model "$MODEL_DIR" --eval-dir "$DATA/eval" --out "$RUN/eval" --num-samples "$NUM_SAMPLES"
               --canvas-len "$CANVAS_LEN" --exec-workers "$EXEC_WORKERS" "${LIMIT_ARGS[@]}")

  # ---- 7. the autoregressive baseline -----------------------------------------------------------
  if ! done_marker ar; then
    log "autoregressive baseline: the model's own generate, $NUM_SAMPLES samples per problem"
    "$PY" -m text_vpa.stage0.evaluate --decoder ar "${EVAL_COMMON[@]}"
    mark ar
  fi

  # ---- 8. training ------------------------------------------------------------------------------
  if ! done_marker "train-$STEPS"; then
    log "training the realizer to step $STEPS"
    if [[ -f "$RUN/last.pt" ]]; then
      "$PY" -m text_vpa.stage0.train --corpus "$DATA/corpus" --eval-dir "$DATA/eval" --run "$RUN" --steps "$STEPS"
    else
      # shellcheck disable=SC2086
      "$PY" -m text_vpa.stage0.train --corpus "$DATA/corpus" --eval-dir "$DATA/eval" --run "$RUN" \
        --set "model=$MODEL_DIR" "canvas_len=$CANVAS_LEN" "steps=$STEPS" $SMOKE_SET $TRAIN_SET
    fi
    mark "train-$STEPS"
  fi

  # ---- 9. the realizer at K = 2, 4, 8 -----------------------------------------------------------
  if ! done_marker "realizer-$STEPS"; then
    log "realizer at K = 2, 4, 8 ($ORDERS)"
    "$PY" -m text_vpa.stage0.evaluate --decoder realizer --checkpoint "$RUN/last.pt" --k-steps 2 4 8 \
      --orders "${ORDER_ARGS[@]}" "${EVAL_COMMON[@]}"
    mark "realizer-$STEPS"
  fi

  # ---- 10. report -------------------------------------------------------------------------------
  "$PY" -m text_vpa.stage0.report --eval "$RUN/eval" --run "$RUN"

  # ---- 11. reference row: the instruction-tuned checkpoint through its chat template ------------
  if [[ "$AR_IT" == 1 ]] && ! done_marker ar-chat; then
    log "reference row: $MODEL_IT through its chat template (not part of H2)"
    download "$MODEL_IT"
    "$PY" -m text_vpa.stage0.evaluate --decoder ar --chat "${EVAL_COMMON[@]}" --model "$(model_dir "$MODEL_IT")"
    mark ar-chat
    "$PY" -m text_vpa.stage0.report --eval "$RUN/eval" --run "$RUN" >/dev/null
  fi
  log "Stage 0 finished: $RUN/eval/report.md"
}

case "${1:-launch}" in
  launch) launch ;;
  run)    run ;;
  status) status ;;
  *)      die "usage: stage0.sh [launch|run|status]" ;;
esac
