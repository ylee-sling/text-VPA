#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# overfit_check.sh -- can the realizer reproduce programs it was trained on? Trains Stage 2 on the frozen
# Stage-1 planner of a finished run, on a few hundred training functions, for long enough to know them by
# heart (STEPS x 64 examples = hundreds of epochs), then runs text_vpa.evaluation.diagnose on those functions.
# Run on the VM, in tmux (about 1 hour on an L4):
#
#   tmux new -d -s overfit 'bash ~/text-VPA/scripts/gcp/overfit_check.sh 2>&1 | tee ~/text-VPA/runs/overfit.log'
#
# Reading the report (part 3, "training contexts" = the memorized functions):
#   "training conditions" (no energies, reference length and mode) parses and matches them  -> the recipe
#       and the sampler work at this scale; the full run failed for lack of data, so collect more data;
#   it does not  -> the realizer cannot fit even a few hundred programs: fix the recipe before more data.
#
# Variables (defaults in brackets):
#   RUN [runs/l4]  finished run (stage1/stage1_final.pt, batch_overrides)   DATA [data] its corpus
#   OUT [runs/overfit]   FUNCTIONS [512]   STEPS [2000]   EVAL_EVERY [250]   DEVICE [auto]
#   STAGE2_PRESET [realizer=l4 training=l4]   SHUTDOWN_WHEN_DONE [1]: shut the VM down after the report
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD   # as run_all.sh: no other project's settings
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN="${RUN:-runs/l4}"
DATA="${DATA:-data}"
OUT="${OUT:-runs/overfit}"
FUNCTIONS="${FUNCTIONS:-512}"
STEPS="${STEPS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
DEVICE="${DEVICE:-auto}"
STAGE2_PRESET="${STAGE2_PRESET:-realizer=l4 training=l4}"
read -r -a PRESET <<< "$STAGE2_PRESET"
BATCH=()
[[ -s "$RUN/batch_overrides" ]] && read -r -a BATCH < "$RUN/batch_overrides"
log() { echo "[$(date '+%F %T')] $*"; }

[[ -s "$RUN/stage1/stage1_final.pt" ]] || { echo "error: $RUN/stage1/stage1_final.pt is missing" >&2; exit 1; }
SHARD="$(find "$DATA/corpus" -maxdepth 1 -name 'train-*.jsonl.gz' | sort | head -1)"
[[ -n "$SHARD" ]] || { echo "error: no $DATA/corpus/train-*.jsonl.gz" >&2; exit 1; }

BATCH_SIZE="$(grep -o 'training.batch_size=[0-9]*' <<< "${BATCH[*]:-} $STAGE2_PRESET" | tail -1 | cut -d= -f2)"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Batches are formed per canvas-length bucket and a bucket's remainder is dropped, so a function in a bucket
# with fewer members than the batch size would never be trained. Whole buckets are taken instead, each a
# multiple of the batch size, so every chosen function is trained in every epoch.
log "1/3 about $FUNCTIONS training functions of $SHARD, in whole canvas buckets -> $OUT/data/corpus"
mkdir -p "$OUT/data/corpus"
python - "$SHARD" "$OUT/data/corpus/train-00000.jsonl.gz" "$FUNCTIONS" "$RUN/stage1/stage1_final.pt" \
  "$BATCH_SIZE" <<'PYEOF'
import collections, gzip, json, sys

import torch

from text_vpa.data.tokenization import tokenizer_from_spec

shard, dest, wanted, checkpoint, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
tok = tokenizer_from_spec(state["tokenizer"])
bucket, max_out = state["config"]["data"]["bucket_size"], state["config"]["data"]["max_output_tokens"]
buckets = collections.defaultdict(list)
with gzip.open(shard, "rt", encoding="utf-8") as handle:
    for i, line in enumerate(handle):
        if i == 20 * wanted:
            break
        length = len(tok.encode(json.loads(line)["target"])) + 1        # target + the closing [PAD]
        if length <= max_out:
            buckets[-(-length // bucket) * bucket].append(line)
chosen, used = [], []
for canvas, lines in sorted(buckets.items(), key=lambda item: -len(item[1])):
    take = min(len(lines), wanted - len(chosen)) // batch * batch
    if take:
        chosen += lines[:take]
        used.append(f"{canvas}: {take}")
if not chosen:
    raise SystemExit(f"no canvas bucket holds {batch} functions; raise FUNCTIONS")
with gzip.open(dest, "wt", encoding="utf-8") as out:
    out.writelines(chosen)
print(f"{len(chosen)} functions in canvas buckets (tokens: functions) {', '.join(used)}")
PYEOF
[[ -s "$DATA/corpus/val.jsonl.gz" ]] && cp "$DATA/corpus/val.jsonl.gz" "$OUT/data/corpus/val.jsonl.gz"
SUBSET="jsonl:$OUT/data/corpus/train-00000.jsonl.gz"

if [[ ! -s "$OUT/stage2/stage2_final.pt" ]]; then
  log "2/3 Stage 2 on those functions only: $STEPS steps (validation = the same functions)"
  python train.py --stage 2 --planner-checkpoint "$RUN/stage1/stage1_final.pt" --data "$SUBSET" \
    --eval-data "$SUBSET" --eval-every "$EVAL_EVERY" --output-dir "$OUT/stage2" --max-steps "$STEPS" \
    --device "$DEVICE" --dataset-cache "$DATA/cache" "${PRESET[@]}" "${BATCH[@]}" \
    training.realizer.warmup_steps=100
else
  log "2/3 Stage 2 done earlier ($OUT/stage2/stage2_final.pt)"
fi

log "3/3 diagnose: can it reproduce the functions it trained on?"
python -m text_vpa.evaluation.diagnose --run "$OUT" --data "$OUT/data" --examples 16 --device "$DEVICE"
log "done; the report is above (and in runs/overfit.log when started with the tmux command)"

if [[ "${SHUTDOWN_WHEN_DONE:-1}" == 1 ]]; then
  log "shutting down in 1 minute (SHUTDOWN_WHEN_DONE=0 keeps the VM up); cancel with: sudo shutdown -c"
  sudo -n shutdown -h +1 "text-VPA overfit check finished" || log "warning: shutdown failed"
fi
