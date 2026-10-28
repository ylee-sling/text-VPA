#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# create_vm.sh -- create a GCP VM for text-VPA training: g2-standard-16 = 1 x NVIDIA L4 (24 GB), 16 vCPU, 64 GB.
# Run on your own machine (gcloud must be installed and logged in).
#
#   ZONE=asia-northeast3-b scripts/gcp/create_vm.sh             # create
#   ZONE=asia-northeast3-b scripts/gcp/create_vm.sh --dry-run   # print the gcloud command only
#
# Variables (defaults in brackets):
#   ZONE           required; zones with L4 GPUs: gcloud compute accelerator-types list --filter="name=nvidia-l4"
#   PROJECT        [gcloud config get-value project]
#   NAME           [text-vpa-l4]
#   MACHINE        [g2-standard-16]
#   DISK_GB        [300]   boot disk (corpus Parquet files, dataset cache, checkpoints)
#   IMAGE_FAMILY   [common-cu129-ubuntu-2404-nvidia-580]  Deep Learning VM: NVIDIA driver 580, Python 3.12
#   PROVISIONING   [STANDARD]  or SPOT (much cheaper; may be stopped at any time -- train_l4.sh resumes)
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '5,20p' "$0"; exit 0; }

ZONE="${ZONE:?set ZONE to a zone with L4 GPUs; list them with: gcloud compute accelerator-types list --filter=name=nvidia-l4}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
[[ -n "$PROJECT" ]] || { echo "error: set PROJECT or run: gcloud config set project YOUR_PROJECT" >&2; exit 1; }
NAME="${NAME:-text-vpa-l4}"
MACHINE="${MACHINE:-g2-standard-16}"
DISK_GB="${DISK_GB:-300}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
PROVISIONING="${PROVISIONING:-STANDARD}"

cmd=(gcloud compute instances create "$NAME"
  --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE"
  --image-family "$IMAGE_FAMILY" --image-project deeplearning-platform-release
  --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced
  --maintenance-policy TERMINATE --provisioning-model "$PROVISIONING"
  --scopes https://www.googleapis.com/auth/cloud-platform
  --metadata enable-oslogin=TRUE)
if [[ "$PROVISIONING" == "SPOT" ]]; then
  cmd+=(--instance-termination-action STOP)
fi

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
"${cmd[@]}"
cat <<EOF

VM $NAME created in $ZONE. Next:
  gcloud compute ssh $NAME --zone $ZONE --project $PROJECT
  git clone https://github.com/ylee-sling/text-VPA.git && cd text-VPA
  bash scripts/gcp/setup_vm.sh
  tmux new -s train 'bash scripts/gcp/train_l4.sh 2>&1 | tee -a train_l4.log'
EOF
