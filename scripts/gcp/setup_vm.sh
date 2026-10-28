#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# setup_vm.sh -- prepare a fresh GPU VM (GCP Deep Learning VM, Ubuntu 24.04) for text-VPA. Run on the VM from
# the repository root, once:
#
#   bash scripts/gcp/setup_vm.sh                # venv, pinned dependencies, GPU check, evaluation sets, tests
#   bash scripts/gcp/setup_vm.sh --skip-tests   # the same without the test suite
#
# Steps: the NVIDIA driver answers; Python >= 3.12 (the pinned numpy needs it); .venv with requirements-dev.txt
# and the package; PyTorch sees the GPU and runs a bfloat16 matmul; HumanEval and MBPP downloaded to
# data/eval (decontamination and Pass@k) and the evaluation harness checked on their reference solutions;
# the fast test suite.
set -euo pipefail

SKIP_TESTS=0
case "${1:-}" in
  --skip-tests) SKIP_TESTS=1 ;;
  -h|--help) sed -n '5,15p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
log() { echo "[$(date '+%F %T')] $*"; }

log "1/6 GPU driver"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi fails; create the VM from a Deep Learning VM image (scripts/gcp/create_vm.sh)" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "2/6 Python"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "error: Python >= 3.12 is required (found $("$PYTHON" -V 2>&1)); use the Ubuntu 24.04 image" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  if ! "$PYTHON" -m venv .venv 2>/dev/null; then          # Ubuntu ships ensurepip separately
    rm -rf .venv
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip
    "$PYTHON" -m venv .venv
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "3/6 dependencies (pinned)"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-dev.txt
python -m pip install -q --no-deps -e .

log "4/6 PyTorch on the GPU"
python - <<'EOF'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see the GPU: install the torch build matching the driver "
                     "(https://pytorch.org/get-started/locally/), then re-run this script")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
float((x @ x).float().sum())
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB, bf16 {torch.cuda.is_bf16_supported()}")
EOF

log "5/6 evaluation sets"
mkdir -p data/eval
[[ -s data/eval/HumanEval.jsonl.gz ]] || curl -fsSL -o data/eval/HumanEval.jsonl.gz \
  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
[[ -s data/eval/mbpp.jsonl ]] || curl -fsSL -o data/eval/mbpp.jsonl \
  https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
for bench in humaneval mbpp; do
  file=data/eval/HumanEval.jsonl.gz; [[ $bench == mbpp ]] && file=data/eval/mbpp.jsonl
  python evaluate.py --benchmark "$bench" --data "$file" --reference --out "runs/harness-check/$bench" >/dev/null
done
python - <<'EOF'
import json
he = json.load(open("runs/harness-check/humaneval/summary.json"))
mb = json.load(open("runs/harness-check/mbpp/summary.json"))
print(f"harness check: HumanEval {he['solved_problems']}/{he['problems']}, MBPP {mb['solved_problems']}/{mb['problems']} "
      "reference solutions pass (expected 164/164 and 499/500)")
if he["solved_problems"] != he["problems"] or mb["solved_problems"] < mb["problems"] - 1:
    raise SystemExit("the execution harness does not reproduce the reference results")
EOF

if [[ $SKIP_TESTS -eq 0 ]]; then
  log "6/6 test suite"
  python -m pytest -q -x
else
  log "6/6 test suite skipped"
fi
log "ready. Start training with:"
echo "  tmux new -s train 'bash scripts/gcp/train_l4.sh 2>&1 | tee -a train_l4.log'"
