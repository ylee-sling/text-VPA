#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# train_l4.sh -- the whole PR-JEPA pipeline on one NVIDIA L4 VM, resumable at every stage:
#
#   corpus (download + build, decontaminated) -> memory check -> Stage 1 -> Stage 2 -> Pass@k (HumanEval, MBPP)
#
# Run on the VM after scripts/gcp/setup_vm.sh, inside tmux so that it survives a closed SSH session (or let
# scripts/gcp/run_all.sh do all of it):
#
#   tmux new -s train 'bash scripts/gcp/train_l4.sh 2>&1 | tee -a train_l4.log'
#   GCS=gs://my-bucket/text-vpa bash scripts/gcp/train_l4.sh     # also copy runs/ to Cloud Storage
#
# Re-running the same command after a stop (e.g. a Spot preemption) skips finished stages and resumes the
# interrupted one from its last checkpoint (written every training.save_every = 1000 steps). Keep STAGE1_STEPS
# unchanged when resuming Stage 1: the EMA momentum schedule (Eq. 4) spans it, and a change is refused.
#
# Corpus (CORPUS):
#   shards  every shard of github-code-clean's main branch (880 Parquet files, 314 GB, all languages), kept to
#           permissively licensed Python while downloading (DATA/shards, ~5 GB per 100 shards; the raw shard is
#           deleted); functions, methods of classes and undocumented functions; documented ones first, at most
#           MAX_FUNCTIONS. Hours of download for all 880 shards; SHARDS=0-99 takes a tenth.
#   export  Run 1's corpus: HF_FILES files of the Parquet export (HF_CONFIG), documented top-level functions.
# A corpus directory remembers its recipe (DATA/corpus/recipe) and a run directory its corpus (RUN/corpus):
# a different CORPUS for an existing DATA, or a different corpus for a run that has checkpoints, is refused.
#
# Variables (defaults in brackets):
#   RUN            [runs/l4]   checkpoints, metrics, memory check, evaluation results
#   DATA           [data]      shards or raw Parquet files, corpus, dataset cache; DATA/eval from setup_vm.sh
#   CORPUS         [shards]    shards | export (see above)
#   MAX_FUNCTIONS  [2000000]   SHARDS [all, e.g. 0-99]   DOWNLOAD_WORKERS [8]   (CORPUS=shards)
#   HF_DATASET     [codeparrot/github-code-clean]   HF_CONFIG [Python-all]   HF_FILES [10]   (CORPUS=export)
#   CORPUS_SOURCE  [unset]     instead of a download, a source spec of text_vpa.data.corpus (dir:/jsonl:/parquet:)
#   CORPUS_LICENSE [unset]     license of a CORPUS_SOURCE that has none (e.g. mit for your own code)
#   PRESET         [planner=base realizer=l4 data=code training=l4]   Hydra overrides of Stage 1
#   STAGE2_PRESET  [realizer=l4 training=l4]   Stage 2 (planner, data and VICReg come from Stage 1)
#   STAGE1_STEPS   [50000]   STAGE2_STEPS [60000]   EVAL_EVERY [2000]
#   COLLAPSE_GRACE [STAGE1_STEPS + 1]  first step at which the stable-rank guard may reject Stage 1; by default
#                  it only logs srank (Run 1 rose from 12 to 125 of d = 256 over 50K steps, below d/2 throughout)
#   NUM_SAMPLES    [10]  samples per problem for Pass@k     EVAL_LIMIT [unset] first N problems only
#   EVAL_VARIANTS  [energy no-energy]  Pass@k as configured, and without the constraint energies
#   DEVICE         [auto]    GCS [unset]   SYNC_MINUTES [30]   HF_TOKEN [unset, optional]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN="${RUN:-runs/l4}"
DATA="${DATA:-data}"
CORPUS="${CORPUS:-shards}"
MAX_FUNCTIONS="${MAX_FUNCTIONS:-2000000}"
SHARDS="${SHARDS:-}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
HF_DATASET="${HF_DATASET:-codeparrot/github-code-clean}"
HF_CONFIG="${HF_CONFIG:-Python-all}"
HF_FILES="${HF_FILES:-10}"
PRESET="${PRESET:-planner=base realizer=l4 data=code training=l4}"
STAGE2_PRESET="${STAGE2_PRESET:-realizer=l4 training=l4}"
STAGE1_STEPS="${STAGE1_STEPS:-50000}"
STAGE2_STEPS="${STAGE2_STEPS:-60000}"
COLLAPSE_GRACE="${COLLAPSE_GRACE:-$((STAGE1_STEPS + 1))}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
EVAL_VARIANTS="${EVAL_VARIANTS:-energy no-energy}"
DEVICE="${DEVICE:-auto}"
SYNC_MINUTES="${SYNC_MINUTES:-30}"
WORKERS="$(nproc)"
read -r -a PRESET_ARGS <<< "$PRESET"
read -r -a STAGE2_ARGS <<< "$STAGE2_PRESET"
read -r -a VARIANTS <<< "$EVAL_VARIANTS"
log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "error: $*" >&2; exit 1; }

case "$CORPUS" in shards|export) ;; *) die "CORPUS must be shards or export, not '$CORPUS'" ;; esac
for v in "${VARIANTS[@]}"; do
  case "$v" in energy|no-energy) ;; *) die "EVAL_VARIANTS takes energy and no-energy, not '$v'" ;; esac
done
if [[ ! -s "$DATA/eval/HumanEval.jsonl.gz" && -s data/eval/HumanEval.jsonl.gz && -s data/eval/mbpp.jsonl ]]; then
  log "copying the evaluation sets of data/eval to $DATA/eval"
  mkdir -p "$DATA/eval" && cp data/eval/HumanEval.jsonl.gz data/eval/mbpp.jsonl "$DATA/eval/"
fi
for f in "$DATA/eval/HumanEval.jsonl.gz" "$DATA/eval/mbpp.jsonl"; do
  [[ -s "$f" ]] || die "$f is missing; run scripts/gcp/setup_vm.sh first (or scripts/gcp/run_all.sh)"
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
if [[ -n "${CORPUS_SOURCE:-}" ]]; then
  RECIPE="source $CORPUS_SOURCE"
elif [[ "$CORPUS" == shards ]]; then
  RECIPE="shards $HF_DATASET shards=${SHARDS:-all} methods undocumented documented-first max=$MAX_FUNCTIONS"
else
  RECIPE="export $HF_DATASET $HF_CONFIG files=$HF_FILES"
fi
if [[ -s "$DATA/corpus/stats.json" ]]; then
  BUILT="$(cat "$DATA/corpus/recipe" 2>/dev/null || echo "export (a corpus built before recipes were recorded)")"
  if [[ "$BUILT" != "$RECIPE" && ! ( "$CORPUS" == export && "$BUILT" == export* && -z "${CORPUS_SOURCE:-}" ) ]]; then
    die "$DATA/corpus was built by another recipe:
    built:     $BUILT
    requested: $RECIPE
  use a new DATA directory for this corpus (e.g. DATA=${DATA%/}-$CORPUS), or delete $DATA/corpus to rebuild it there"
  fi
  log "1/5 corpus present ($DATA/corpus: $BUILT)"
else
  BUILD_ARGS=()
  if [[ -n "${CORPUS_SOURCE:-}" ]]; then
    SOURCE="$CORPUS_SOURCE"
    [[ -n "${CORPUS_LICENSE:-}" ]] && BUILD_ARGS+=(--assume-license "$CORPUS_LICENSE")
  elif [[ "$CORPUS" == shards ]]; then
    log "1/5 download the shards of $HF_DATASET (${SHARDS:-all}), keeping permissively licensed Python"
    SHARD_ARGS=()
    [[ -n "$SHARDS" ]] && SHARD_ARGS=(--shards "$SHARDS")
    python -m text_vpa.data.corpus download-shards --dataset "$HF_DATASET" --out "$DATA/shards" \
      --workers "$DOWNLOAD_WORKERS" "${SHARD_ARGS[@]}"
    SOURCE="parquet:$DATA/shards/*.parquet"
    BUILD_ARGS+=(--include-methods --allow-undocumented --prefer-documented --max-functions "$MAX_FUNCTIONS")
  else
    log "1/5 download $HF_FILES Parquet file(s) of $HF_DATASET ($HF_CONFIG)"
    python -m text_vpa.data.corpus download --dataset "$HF_DATASET" --config "$HF_CONFIG" --out "$DATA/raw" \
      --max-files "$HF_FILES"
    SOURCE="parquet:$DATA/raw/*.parquet"
  fi
  log "1/5 build the corpus from $SOURCE"
  python -m text_vpa.data.corpus build --source "$SOURCE" "${BUILD_ARGS[@]}" \
    --humaneval "$DATA/eval/HumanEval.jsonl.gz" --mbpp "$DATA/eval/mbpp.jsonl" --out "$DATA/corpus" \
    --workers "$WORKERS"
  echo "$RECIPE" > "$DATA/corpus/recipe"
fi
python - "$DATA/corpus/stats.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
kept = s["train"] + s["val"]
extra = (f"; {s['documented']} documented, {s['methods']} methods" if "documented" in s else "")
print(f"corpus: {kept} functions ({s['train']} train, {s['val']} validation{extra}) from {s['files_used']} "
      f"permissive files; {s['duplicates']} duplicates and {s['contaminated']} contaminated dropped")
EOF

# The run remembers its corpus: checkpoints trained on one corpus are not continued on another.
IDENTITY="$(cd "$DATA/corpus" && pwd) $(sha256sum "$DATA/corpus/stats.json" | cut -c1-16)"
if [[ -s "$RUN/corpus" ]]; then
  [[ "$(cat "$RUN/corpus")" == "$IDENTITY" ]] || die "$RUN was trained on another corpus:
    $RUN: $(cat "$RUN/corpus")
    now:  $IDENTITY
  use a new RUN directory (e.g. RUN=runs/run2)"
else
  [[ -e "$RUN/stage1" ]] && log "warning: $RUN has checkpoints but no record of its corpus; assuming $DATA/corpus"
  echo "$IDENTITY" > "$RUN/corpus"
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
    die "the preset does not fit; lower training.batch_size and raise training.<stage>.grad_accum_steps"
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
  log "3/5 Stage 1 (${RESUME[*]:-fresh start}; the stable-rank guard can reject from step $COLLAPSE_GRACE)"
  set +e
  python train.py --stage 1 --data "$TRAIN_DATA" --output-dir "$S1" --max-steps "$STAGE1_STEPS" \
    "${COMMON[@]}" "${RESUME[@]}" "${PRESET_ARGS[@]}" "training.collapse_grace_steps=$COLLAPSE_GRACE"
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
python - "$S1/metrics.jsonl" <<'EOF' || true
import json, sys
last = {}
for line in open(sys.argv[1]):
    row = json.loads(line)
    last.update({k.split("/")[1]: v for k, v in row.items() if k.endswith("/stable_rank")})
if last:
    print("Stage 1 stable rank at the last check: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(last.items()))
          + " (the paper's guard asks for d/2)")
EOF

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
# 5. Pass@k (best_stage2.pt), as configured and without the constraint energies
# ---------------------------------------------------------------------------------------------
LIMIT_ARGS=()
[[ -n "${EVAL_LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$EVAL_LIMIT")
for variant in "${VARIANTS[@]}"; do
  for bench in humaneval mbpp; do
    out="$RUN/eval/$bench"
    VARIANT_ARGS=()
    if [[ $variant == no-energy ]]; then
      out="$RUN/eval/$bench-no-energy"
      VARIANT_ARGS=(--no-energy)
    fi
    [[ -s "$out/summary.json" ]] && { log "5/5 $bench ($variant) evaluated ($out)"; continue; }
    file="$DATA/eval/HumanEval.jsonl.gz"
    [[ $bench == mbpp ]] && file="$DATA/eval/mbpp.jsonl"
    log "5/5 Pass@k on $bench ($variant)"
    python evaluate.py --benchmark "$bench" --data "$file" --checkpoint "$S2/best_stage2.pt" \
      --num-samples "$NUM_SAMPLES" --k 1 10 --device "$DEVICE" --out "$out" "${LIMIT_ARGS[@]}" \
      "${VARIANT_ARGS[@]}" >/dev/null
  done
done
python - "$RUN" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
for name in ("humaneval", "mbpp", "humaneval-no-energy", "mbpp-no-energy"):
    path = run / "eval" / name / "summary.json"
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    scores = ", ".join(f"{k} {v:.3f}" for k, v in s.items() if k.startswith("pass@"))
    print(f"{name}: {scores} ({s['problems']} problems, n = {s['samples_per_problem']}, K = {s['k_steps']}, "
          f"depth {s['sequential_depth']}, {s['seconds_per_problem']} s/problem)")
EOF
log "done: checkpoints in $S1 and $S2, results in $RUN/eval"
