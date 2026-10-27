#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# init_repo.sh -- initialise the text-VPA git repository (idempotent; safe to re-run).
#
#   scripts/init_repo.sh                      # git init on branch main, .gitignore, remote origin
#   scripts/init_repo.sh --remote URL         # another remote URL
#   TEXT_VPA_REMOTE=URL scripts/init_repo.sh  # same, from the environment
#
# It never commits or pushes; scripts/release.sh does that after the tests pass.
set -euo pipefail

REMOTE_URL="${TEXT_VPA_REMOTE:-https://github.com/ylee-sling/text-VPA.git}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE_URL="$2"; shift 2 ;;
    -h|--help) sed -n '5,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f pyproject.toml && -d text_vpa ]] || { echo "error: run from the text-VPA repository" >&2; exit 1; }
command -v git >/dev/null || { echo "error: git is not installed" >&2; exit 1; }

# 1. Repository on branch main
if [[ ! -d .git ]]; then
  git init --quiet
  echo "initialised an empty git repository in $ROOT"
fi
if [[ -z "$(git rev-parse --verify --quiet HEAD 2>/dev/null)" ]]; then
  git symbolic-ref HEAD refs/heads/main            # unborn branch: just rename it
else
  git branch -M main
fi

# 2. .gitignore: checkpoints, datasets, caches and run outputs are never committed
REQUIRED_IGNORES=(
  "__pycache__/" "*.pyc" "*.egg-info/" "build/" "dist/"
  ".venv/" "venv/" ".pytest_cache/" ".ruff_cache/" ".mypy_cache/"
  "checkpoints/" "*.pt" "*.pth" "*.ckpt" "*.safetensors"
  "/data/" "datasets/" "*.jsonl.gz"
  "runs/" "outputs/" "multirun/" "wandb/" "logs/"
  ".DS_Store" ".idea/" ".vscode/"
)
touch .gitignore
added=0
for pattern in "${REQUIRED_IGNORES[@]}"; do
  if ! grep -qxF -- "$pattern" .gitignore; then
    if [[ $added -eq 0 ]]; then printf '\n# Added by scripts/init_repo.sh\n' >> .gitignore; fi
    printf '%s\n' "$pattern" >> .gitignore
    added=$((added + 1))
  fi
done
echo ".gitignore: $added pattern(s) added"

# 3. Remote origin
if git remote get-url origin >/dev/null 2>&1; then
  current="$(git remote get-url origin)"
  if [[ "$current" != "$REMOTE_URL" ]]; then
    git remote set-url origin "$REMOTE_URL"
    echo "origin: $current -> $REMOTE_URL"
  else
    echo "origin: $REMOTE_URL (unchanged)"
  fi
else
  git remote add origin "$REMOTE_URL"
  echo "origin: $REMOTE_URL"
fi

echo "branch: $(git symbolic-ref --short HEAD)"
echo "next:   scripts/release.sh   (tests, lint, commit, tag v0.1.0, push)"
