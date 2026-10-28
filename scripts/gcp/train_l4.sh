#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# train_l4.sh -- the whole PR-JEPA pipeline on one NVIDIA L4 VM, resumable at every stage:
#
#   corpus (download + build, decontaminated) -> memory check -> Stage 1 -> Stage 2 -> Pass@k (HumanEval, MBPP)
#
# Run on the VM after scripts/gcp/setup_vm.sh, inside tmux so that it survives a closed SSH session:
#
#   tmux new -s train 'bash scripts/gcp/train_l4.sh 2>&1 | tee -a train_l4.log'
#   GCS=gs://my-bucket/text-vpa bash scripts/gcp/train_l4.sh     # also copy runs/ to Cloud Storage
#
# Re-running the same command after a stop (e.g. a Spot preemption) skips finished stages and resumes the
# interrupted one from its last checkpoint (written every training.save_every = 1000 steps). Keep STAGE1_STEPS
# unchanged when resuming Stage 1: the EMA momentum schedule (Eq. 4) spans it, and a change is refused.
#
# Variables (defaults in brackets):
#   RUN            [runs/l4]   checkpoints, metrics, memory check, evaluation results
#   DATA           [data]      raw Parquet files, corpus, dataset cache; data/eval from setup_vm.sh
#   HF_DATASET     [codeparrot/github-code-clean]   HF_CONFIG [Python-all]   HF_FILES [10] Parquet files
#   CORPUS_SOURCE  [unset]     instead of the download, a source spec of text_vpa.data.corpus (dir:/jsonl:/parquet:)
#   CORPUS_LICENSE [unset]     license of a CORPUS_SOURCE that has none (e.g. mit for your own code)
#   PRESET         [planner=base realizer=l4 data=code training=l4]   Hydra overrides of Stage 1
#   STAGE2_PRESET  [realizer=l4 training=l4]   Stage 2 (planner, data and VICReg come from Stage 1)
#   STAGE1_STEPS   [50000]   STAGE2_STEPS [100000]   EVAL_EVERY [2000]
#   NUM_SAMPLES    [10]  samples per problem for Pass@k     EVAL_LIMIT [unset] first N problems only
#   DEVICE         [auto]    GCS [unset]   SYNC_MINUTES [30]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN="${RUN:-runs/l4}"
DATA="${DATA:-data}"
HF_DATASET="${HF_DATASET:-codeparrot/github-code-clean}"
HF_CONFIG="${HF_CONFIG:-Python-all}"
HF_FILES="${HF_FILES:-10}"
PRESET="${PRESET:-planner=base realizer=l4 data=code training=l4}"
STAGE2_PRESET="${STAGE2_PRESET:-realizer=l4 training=l4}"
STAGE1_STEPS="${STAGE1_STEPS:-50000}"
STAGE2_STEPS="${STAGE2_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
DEVICE="${DEVICE:-auto}"
SYNC_MINUTES="${SYNC_MINUTES:-30}"
WORKERS="$(nproc)"
read -r -a PRESET_ARGS <<< "$PRESET"
read -r -a STAGE2_ARGS <<< "$STAGE2_PRESET"
log() { echo "[$(date '+%F %T')] $*"; }

for f in "$DATA/eval/HumanEval.jsonl.gz" "$DATA/eval/mbpp.jsonl"; do
  [[ -s "$f" ]] || { echo "error: $f is missing; run scripts/gcp/setup_vm.sh first" >&2; exit 1; }
done
mkdir -p "$RUN"

# Optional copy of $RUN to Cloud Storage: every SYNC_MINUTES and when the script exits.
sync_gcs() {
  [[ -n "${GCS:-}" ]] || return 0
  gcloud storage rsync --recursive "$RUN" "${GCS%/}/$(basename "$RUN")" >/dev/null 2>&1 \
    || log "warning: copy to $GCS failed"
}
if [[ -n "${GCS:-}" ]]; then
  ( while sleep "$((SYNC_MINUTES * 60))"; do sync_gcs; done ) &
  SYNC_PID=$!
  trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_gcs' EXIT
fi

# ---------------------------------------------------------------------------------------------
# 1. Corpus
# ---------------------------------------------------------------------------------------------
if [[ ! -s "$DATA/corpus/stats.json" ]]; then
  if [[ -n "${CORPUS_SOURCE:-}" ]]; then
    SOURCE="$CORPUS_SOURCE"
  else
    log "1/5 download $HF_FILES Parquet file(s) of $HF_DATASET ($HF_CONFIG)"
    python -m text_vpa.data.corpus download --dataset "$HF_DATASET" --config "$HF_CONFIG" --out "$DATA/raw" \
      --max-files "$HF_FILES"
    SOURCE="parquet:$DATA/raw/*.parquet"
  fi
  log "1/5 build the corpus from $SOURCE"
  LICENSE_ARGS=()
  [[ -n "${CORPUS_LICENSE:-}" ]] && LICENSE_ARGS=(--assume-license "$CORPUS_LICENSE")
  python -m text_vpa.data.corpus build --source "$SOURCE" "${LICENSE_ARGS[@]}" \
    --humaneval "$DATA/eval/HumanEval.jsonl.gz" --mbpp "$DATA/eval/mbpp.jsonl" --out "$DATA/corpus" \
    --workers "$WORKERS"
else
  log "1/5 corpus present ($DATA/corpus)"
fi
TRAIN_DATA="jsonl:$DATA/corpus/train-*.jsonl.gz"
VAL_DATA="jsonl:$DATA/corpus/val.jsonl.gz"
COMMON=(--device "$DEVICE" --dataset-cache "$DATA/cache" --build-workers "$WORKERS")

# ---------------------------------------------------------------------------------------------
# 2. Memory check (once): worst-case batches must fit the GPU
# ---------------------------------------------------------------------------------------------
if [[ ! -s "$RUN/memory_probe.json" ]]; then
  log "2/5 memory check"
  if ! python -m text_vpa.utils.memory_probe --device "$DEVICE" "${PRESET_ARGS[@]}" > "$RUN/memory_probe.tmp"; then
    cat "$RUN/memory_probe.tmp"
    echo "error: the preset does not fit; lower training.batch_size and raise training.<stage>.grad_accum_steps" >&2
    exit 1
  fi
  mv "$RUN/memory_probe.tmp" "$RUN/memory_probe.json"
fi
cat "$RUN/memory_probe.json"

# ---------------------------------------------------------------------------------------------
# 3. Stage 1: planner
# ---------------------------------------------------------------------------------------------
S1="$RUN/stage1"
if [[ ! -s "$S1/stage1_final.pt" ]]; then
  RESUME=()
  [[ -s "$S1/stage1_last.pt" ]] && RESUME=(--resume "$S1/stage1_last.pt")
  log "3/5 Stage 1 (${RESUME[*]:-fresh start})"
  set +e
  python train.py --stage 1 --data "$TRAIN_DATA" --output-dir "$S1" --max-steps "$STAGE1_STEPS" \
    "${COMMON[@]}" "${RESUME[@]}" "${PRESET_ARGS[@]}"
  rc=$?
  set -e
  if [[ $rc -eq 2 ]]; then
    log "Stage 1 was rejected by the stable-rank guard (srank < d/2); see $S1/metrics.jsonl"
    exit 2
  fi
  [[ $rc -eq 0 ]] || exit "$rc"
else
  log "3/5 Stage 1 done ($S1/stage1_final.pt)"
fi

# ---------------------------------------------------------------------------------------------
# 4. Stage 2: realizer + plan readout on the frozen planner
# ---------------------------------------------------------------------------------------------
S2="$RUN/stage2"
if [[ ! -s "$S2/stage2_final.pt" ]]; then
  RESUME=()
  [[ -s "$S2/stage2_last.pt" ]] && RESUME=(--resume "$S2/stage2_last.pt")
  log "4/5 Stage 2 (${RESUME[*]:-fresh start})"
  python train.py --stage 2 --planner-checkpoint "$S1/stage1_final.pt" --data "$TRAIN_DATA" \
    --eval-data "$VAL_DATA" --eval-every "$EVAL_EVERY" --output-dir "$S2" --max-steps "$STAGE2_STEPS" \
    "${COMMON[@]}" "${RESUME[@]}" "${STAGE2_ARGS[@]}"
else
  log "4/5 Stage 2 done ($S2/stage2_final.pt)"
fi

# ---------------------------------------------------------------------------------------------
# 5. Pass@k
# ---------------------------------------------------------------------------------------------
LIMIT_ARGS=()
[[ -n "${EVAL_LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$EVAL_LIMIT")
for bench in humaneval mbpp; do
  out="$RUN/eval/$bench"
  [[ -s "$out/summary.json" ]] && { log "5/5 $bench evaluated ($out)"; continue; }
  file="$DATA/eval/HumanEval.jsonl.gz"
  [[ $bench == mbpp ]] && file="$DATA/eval/mbpp.jsonl"
  log "5/5 Pass@k on $bench"
  python evaluate.py --benchmark "$bench" --data "$file" --checkpoint "$S2/best_stage2.pt" \
    --num-samples "$NUM_SAMPLES" --k 1 10 --device "$DEVICE" --out "$out" "${LIMIT_ARGS[@]}" >/dev/null
done
python - "$RUN" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
for bench in ("humaneval", "mbpp"):
    s = json.loads((run / "eval" / bench / "summary.json").read_text())
    scores = ", ".join(f"{k} {v:.3f}" for k, v in s.items() if k.startswith("pass@"))
    print(f"{bench}: {scores} ({s['problems']} problems, n = {s['samples_per_problem']}, "
          f"K = {s['k_steps']}, depth {s['sequential_depth']}, {s['seconds_per_problem']} s/problem)")
EOF
log "done: checkpoints in $S1 and $S2, results in $RUN/eval"
