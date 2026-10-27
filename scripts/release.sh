#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# release.sh -- verify, commit, tag and push text-VPA.
#
#   scripts/release.sh                 # tests + lint + Appendix B checks, commit, tag v0.1.0, push main and tag
#   scripts/release.sh --dry-run       # run the checks and print the git commands without changing anything
#   scripts/release.sh --tag v0.1.1    # another tag
#   scripts/release.sh --no-push       # commit and tag locally only
#   scripts/release.sh --slow          # also run the slow convergence test (pytest -m slow)
#   TEXT_VPA_REMOTE=URL scripts/release.sh   # push somewhere else (e.g. a fork or a local bare repository)
#
# Nothing is committed if a check fails. An existing tag is never moved.
set -euo pipefail

REMOTE_URL="${TEXT_VPA_REMOTE:-https://github.com/ylee-sling/text-VPA.git}"
TAG="v0.1.0"
DRY_RUN=0
PUSH=1
SLOW=0
DOI="10.5281/zenodo.23498811"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-push) PUSH=0; shift ;;
    --slow) SLOW=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) sed -n '5,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -d .git ]] || { echo "error: not a git repository; run scripts/init_repo.sh first" >&2; exit 1; }
PYTHON="${PYTHON:-python}"

run() {  # run a git command, or print it in --dry-run mode
  if [[ $DRY_RUN -eq 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

echo "== 1/4 checks"
"$PYTHON" -m pytest -q
if [[ $SLOW -eq 1 ]]; then "$PYTHON" -m pytest -q -m slow; fi
"$PYTHON" -m ruff check .
"$PYTHON" -m text_vpa.utils.verification > /dev/null       # Appendix B numerical checks (Table 2)
"$PYTHON" -m text_vpa.utils.profiler --roofline > /dev/null # Table 3 roofline model
echo "checks passed"

echo "== 2/4 remote and branch"
if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "$REMOTE_URL" ]] || run git remote set-url origin "$REMOTE_URL"
else
  run git remote add origin "$REMOTE_URL"
fi
if [[ -n "$(git rev-parse --verify --quiet HEAD 2>/dev/null)" ]]; then
  run git branch -M main
else
  run git symbolic-ref HEAD refs/heads/main
fi

echo "== 3/4 commit and tag"
run git add -A
if [[ $DRY_RUN -eq 0 ]] && git diff --cached --quiet; then
  echo "nothing to commit"
else
  run git commit -F - <<EOF
feat: complete implementation of PR-JEPA for text-VPA (DOI: $DOI)

Reference implementation of "PR-JEPA: Decoupling Latent Planning from Few-Step
Discrete Realization for Non-Autoregressive Generation of Code, Legal Documents,
and Control" (Yeonseok Lee, SLING AI Inc., https://doi.org/$DOI):

- JEPA planner: context encoder, mode selector, ensemble predictor (Eqs. 3-8),
  EMA target encoder (Eq. 4), VICReg objective and stable-rank guard (Eqs. 5-7)
- cross-attention bridge with spectrally normalised, cached keys and values
- conditional FS-DFM realizer with the exact energy-tilted jump kernel
  (Eqs. 11-17) and the Logit-KL realizer (Eqs. 18-21)
- Algorithm 1 sampler with a runtime K + 3 depth audit and a Prop. 4(d) check
- tree-sitter structural views, staged training (Stages 1-3, DDP, bf16)
- numerical checks of Sec. 4.6 (Table 2) and the roofline model (Table 3)
EOF
fi
if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  echo "error: tag $TAG already exists; choose another with --tag" >&2
  exit 1
fi
run git tag -a "$TAG" -m "text-VPA $TAG: PR-JEPA reference implementation (DOI: $DOI)"

echo "== 4/4 push"
if [[ $PUSH -eq 1 ]]; then
  run git push -u origin main
  run git push origin "$TAG"
else
  echo "skipped (--no-push)"
fi
echo "done: $TAG"
