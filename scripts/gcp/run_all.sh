#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# run_all.sh -- one command, unattended: set up a fresh GPU VM, train text-VPA end to end, evaluate Pass@k, and
# shut the VM down. On the VM (e.g. the Console's "SSH" window):
#
#   git clone https://github.com/ylee-sling/text-VPA.git ~/text-VPA && bash ~/text-VPA/scripts/gcp/run_all.sh
#
# It returns at once; the work runs in a detached tmux session ("textvpa") and is logged to
# runs/logs/run-YYYYmmdd-HHMMSS.log (runs/logs/latest.log links to the newest):
#   tail -f ~/text-VPA/runs/logs/latest.log        or   tmux attach -t textvpa   (detach: Ctrl-b d)
#
# Steps in the tmux session:
#   1. GPU and driver (nvidia-smi; waits up to 10 minutes for a driver that is still initialising)
#   2. system packages (apt-get -y: git tmux curl cron python3-venv) and the pinned Python environment (.venv)
#   3. PyTorch on the GPU (bf16 matmul); HumanEval and MBPP; Pass@k harness checked on the reference solutions
#   4. test suite (informational: failures are reported, not fatal)
#   5. memory check on worst-case batches; if the L4 preset does not fit, smaller micro-batches with more
#      accumulation (same effective batch, 64) are tried
#   6. scripts/gcp/train_l4.sh: corpus -> Stage 1 -> Stage 2 -> Pass@k on HumanEval and MBPP (resumable;
#      retried up to TRAIN_ATTEMPTS times; a Stage-1 collapse rejection is not retried). The default corpus is
#      all of github-code-clean, filtered to permissive Python while downloading (up to 2M functions)
#   7. shutdown -h +1 (cancel within the minute: sudo shutdown -c) after success -- and also after a failure,
#      unless SHUTDOWN_ON_FAILURE=0. Logs, checkpoints and results stay on the boot disk.
# If the VM restarts unexpectedly (e.g. host maintenance with automatic restart), an @reboot cron entry starts
# the run again and it resumes from the last checkpoint; the entry is removed when the run ends.
#
#   bash ~/text-VPA/scripts/gcp/run_all.sh --status   # running or not, outcome markers, last log lines
#   bash ~/text-VPA/scripts/gcp/run_all.sh --cancel   # stop the run without shutting down; no restart on boot
#
# Each run directory (RUN, default runs/l4) records its own outcome (RUN/run_all.done, .failed, ...): a finished
# run is not started again, and another run takes a new RUN (and a new DATA for another corpus), e.g.
#   RUN=runs/run2 DATA=data2 bash ~/text-VPA/scripts/gcp/run_all.sh
#
# A VM reused from another project: check it first with scripts/gcp/preflight.sh (read-only). The run itself
# - uses its own .venv (recreated if it is broken or older than Python 3.12); without a system Python >= 3.12
#   (e.g. Ubuntu 22.04) it installs a standalone Python 3.12 with uv into .python/ -- nothing outside the
#   repository changes;
# - clears settings another project may pass on, also through a tmux server started earlier (PYTHONPATH,
#   PYTHONHOME, LD_LIBRARY_PATH, LD_PRELOAD, conda and venv variables, PIP_* and pip.conf files); the
#   launching shell's CUDA_VISIBLE_DEVICES (set or unset) is the one the run gets;
# - installs PyTorch's CUDA 13.0 build (PyPI) for NVIDIA driver 580 or newer, its CUDA 12.6 build for 525-579.
#
# Settings (environment variables, saved for restarts): everything train_l4.sh reads (CORPUS [shards],
# MAX_FUNCTIONS [2000000], STAGE1_STEPS [50000], STAGE2_STEPS [60000], NUM_SAMPLES [10], GCS, ...), and
# SHUTDOWN_ON_SUCCESS [1], SHUTDOWN_ON_FAILURE [1], SHUTDOWN_DELAY_MIN [1], RUN_TESTS [1], TRAIN_ATTEMPTS [3],
# PYTHON [auto: the interpreter for a new .venv, a path or "uv" for the standalone one], TORCH_INDEX_URL
# [by driver], KEEP_ENV [0], UV_VERSION [0.13.2].
set -Eeuo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"
SESSION="${SESSION:-textvpa}"
STATE="$ROOT/runs"
LOG_DIR="$STATE/logs"
ENV_FILE="$STATE/run_all.env"
CRON_TAG="# text-vpa-run-all"
SAVED_VARS=(RUN DATA CORPUS MAX_FUNCTIONS SHARDS DOWNLOAD_WORKERS HF_DATASET HF_CONFIG HF_FILES HF_TOKEN HF_ENDPOINT
  CORPUS_SOURCE CORPUS_LICENSE PRESET STAGE2_PRESET STAGE1_STEPS STAGE2_STEPS COLLAPSE_GRACE EVAL_EVERY NUM_SAMPLES
  EVAL_LIMIT EVAL_VARIANTS DEVICE GCS SYNC_MINUTES SHUTDOWN_ON_SUCCESS SHUTDOWN_ON_FAILURE SHUTDOWN_DELAY_MIN
  NO_SHUTDOWN RUN_TESTS TRAIN_ATTEMPTS SKIP_GPU_CHECK SKIP_APT SESSION MIN_FREE_GB PYTHON TORCH_INDEX_URL KEEP_ENV
  UV_VERSION)

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "error: $*" >&2; exit 1; }

# The run directory of this launch: RUN, else the one saved for restarts, else runs/l4.
run_dir() {
  local run="${RUN:-}"
  if [[ -z "$run" && -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC2016
    run="$(bash -c 'source "$1" >/dev/null 2>&1; echo "${RUN:-}"' _ "$ENV_FILE")"
  fi
  run="${run:-runs/l4}"
  if [[ "$run" == /* ]]; then echo "$run"; else echo "$ROOT/$run"; fi
}

# Outcome markers used to live in runs/ for the only run directory of the time, runs/l4.
migrate_markers() {
  local m
  for m in "done" failed interrupted; do
    if [[ -f "$STATE/run_all.$m" && -d "$STATE/l4" && ! -e "$STATE/l4/run_all.$m" ]]; then
      mv "$STATE/run_all.$m" "$STATE/l4/run_all.$m"
    fi
  done
}

# ---------------------------------------------------------------------------------------------
# @reboot entry: relaunch after an unexpected restart (removed whenever the run ends)
# ---------------------------------------------------------------------------------------------
cron_install() {
  command -v crontab >/dev/null 2>&1 || { log "warning: no crontab; an unexpected restart will not resume"; return 0; }
  local line="@reboot sleep 90 && /bin/bash $SELF --from-reboot >> $LOG_DIR/reboot.log 2>&1 $CRON_TAG"
  { crontab -l 2>/dev/null | grep -vF "$CRON_TAG" || true; echo "$line"; } | crontab -
}

cron_remove() {
  command -v crontab >/dev/null 2>&1 || return 0
  if crontab -l 2>/dev/null | grep -qF "$CRON_TAG"; then
    { crontab -l 2>/dev/null | grep -vF "$CRON_TAG" || true; } | crontab -
  fi
}

# ---------------------------------------------------------------------------------------------
# Launcher (returns immediately)
# ---------------------------------------------------------------------------------------------
launch() {
  local from_reboot=$1 run_abs
  mkdir -p "$LOG_DIR"
  migrate_markers
  if [[ $from_reboot -eq 1 ]]; then
    [[ -f "$ENV_FILE" ]] || { cron_remove; return 0; }
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  fi
  run_abs="$(run_dir)"
  if [[ -f "$run_abs/run_all.done" ]]; then
    log "this run already finished ($(cat "$run_abs/run_all.done")); results in $run_abs/eval"
    log "another run takes a new directory, e.g. RUN=runs/run2 DATA=data2 bash $SELF; to redo it: rm -rf $run_abs"
    cron_remove
    return 0
  fi
  if [[ $from_reboot -eq 1 ]]; then
    if [[ -f "$run_abs/run_all.failed" || -f "$run_abs/run_all.interrupted" ]]; then
      cron_remove                                           # the run has ended: never relaunch it
      return 0
    fi
    log "restart detected: resuming the run"
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "already running: tmux attach -t $SESSION   (log: $LOG_DIR/latest.log)"
    return 0
  fi
  # Checks that must pass while you are still watching.
  if [[ "${SKIP_APT:-0}" != 1 || "${NO_SHUTDOWN:-0}" != 1 ]]; then
    sudo -n true 2>/dev/null || die "passwordless sudo is required (apt-get and shutdown); log in as a VM admin"
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=1200 install -y -q tmux >/dev/null
  fi
  local avail
  avail="$(df --output=avail -BG "$ROOT" | tail -1 | tr -dc '0-9')"
  [[ "$avail" -ge "${MIN_FREE_GB:-60}" ]] || die "only ${avail} GB free on the boot disk; 150 GB or more is recommended"
  if [[ $from_reboot -eq 0 ]]; then
    rm -f "$run_abs/run_all.failed" "$run_abs/run_all.interrupted" "$run_abs/run_all.cancel"
    : > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    local name
    for name in "${SAVED_VARS[@]}"; do
      if [[ -n "${!name+x}" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$ENV_FILE"; fi
    done
    # The run gets this shell's GPU selection, not a stale one from the environment of a tmux server.
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
      printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES" >> "$ENV_FILE"
    else
      echo "unset CUDA_VISIBLE_DEVICES" >> "$ENV_FILE"
    fi
  fi
  local stamp logfile
  stamp="$(date +%Y%m%d-%H%M%S)"
  logfile="$LOG_DIR/run-$stamp.log"
  ln -sfn "$(basename "$logfile")" "$LOG_DIR/latest.log"
  tmux new-session -d -s "$SESSION" "/bin/bash $(printf '%q' "$SELF") --worker 2>&1 | tee -i -a $(printf '%q' "$logfile")"
  cron_install
  log "started in tmux session '$SESSION' at $(hostname): run directory $run_abs"
  echo "  log:     $logfile"
  echo "  watch:   tail -f $LOG_DIR/latest.log      (or: tmux attach -t $SESSION, then Ctrl-b d to detach)"
  echo "  status:  bash $SELF --status"
  echo "  cancel:  bash $SELF --cancel"
  if [[ "${NO_SHUTDOWN:-0}" == 1 ]]; then
    echo "NO_SHUTDOWN=1: the shutdown at the end is only logged. You can close this window now."
  elif [[ "${SHUTDOWN_ON_SUCCESS:-1}" == 1 || "${SHUTDOWN_ON_FAILURE:-1}" == 1 ]]; then
    echo "The VM shuts itself down when the run ends (SHUTDOWN_ON_SUCCESS=${SHUTDOWN_ON_SUCCESS:-1}," \
         "SHUTDOWN_ON_FAILURE=${SHUTDOWN_ON_FAILURE:-1}). You can close this window now."
  else
    echo "The VM keeps running when the run ends. You can close this window now."
  fi
}

# ---------------------------------------------------------------------------------------------
# Worker (inside tmux)
# ---------------------------------------------------------------------------------------------
APT=(sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=1200
     -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold -y -q)

maybe_shutdown() {
  local enabled=$1 reason=$2
  if [[ "$enabled" != 1 ]]; then
    log "not shutting down ($reason); the VM keeps running"
    return 0
  fi
  local delay="${SHUTDOWN_DELAY_MIN:-1}"
  if [[ "${NO_SHUTDOWN:-0}" == 1 ]]; then
    log "NO_SHUTDOWN=1: would run 'sudo shutdown -h +$delay' ($reason)"
    return 0
  fi
  sync
  log "shutting down in $delay minute(s) ($reason); cancel with: sudo shutdown -c"
  sudo -n shutdown -h "+$delay" "text-VPA run finished ($reason)" || log "warning: shutdown failed"
}

finish() {
  local rc=$1
  trap - EXIT INT HUP TERM
  set +e
  if [[ $rc -ne 0 && "$(systemctl is-system-running 2>/dev/null)" == "stopping" ]]; then
    log "the system is shutting down (exit $rc): the run resumes after the next boot"
    exit "$rc"
  fi
  cron_remove
  mkdir -p "$RUN_ABS"
  if [[ -f "$RUN_ABS/run_all.cancel" ]]; then
    rm -f "$RUN_ABS/run_all.cancel"
    echo "cancelled $(date '+%F %T')" > "$RUN_ABS/run_all.interrupted"
    log "cancelled with --cancel; not shutting down"
  elif [[ $rc -eq 0 ]]; then
    echo "$(date '+%F %T') after $(( (SECONDS + 59) / 60 )) min" > "$RUN_ABS/run_all.done"
    log "SUCCESS: the whole pipeline finished; results in $RUN_ABS/eval"
    maybe_shutdown "${SHUTDOWN_ON_SUCCESS:-1}" "success"
  elif [[ $rc -eq 129 || $rc -eq 130 ]]; then
    echo "interrupted (exit $rc) $(date '+%F %T')" > "$RUN_ABS/run_all.interrupted"
    log "interrupted (exit $rc); not shutting down"
  else
    echo "failed (exit $rc) $(date '+%F %T')" > "$RUN_ABS/run_all.failed"
    log "FAILED (exit $rc); see this log. Checkpoints are kept: re-run the same command to resume."
    maybe_shutdown "${SHUTDOWN_ON_FAILURE:-1}" "failure"
  fi
  exit "$rc"
}

step_gpu() {
  log "1/7 GPU and driver"
  if [[ "${SKIP_GPU_CHECK:-0}" == 1 ]]; then log "skipped (SKIP_GPU_CHECK=1)"; return 0; fi
  local i
  for i in $(seq 1 20); do
    nvidia-smi >/dev/null 2>&1 && break
    if [[ $i -eq 1 && -x /opt/deeplearning/install-driver.sh ]]; then
      log "installing the NVIDIA driver with the Deep Learning VM installer"
      sudo -n /opt/deeplearning/install-driver.sh || true
    fi
    log "nvidia-smi is not ready yet ($i/20); waiting 30 s"
    sleep 30
  done
  nvidia-smi >/dev/null 2>&1 || die "no working NVIDIA driver: create the VM from a Deep Learning VM image"
  nvidia-smi
  local name driver cuda
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  cuda="$(nvidia-smi | grep -o 'CUDA Version: *[0-9.]*' | head -1 || true)"
  log "GPU $name, driver $driver, ${cuda:-CUDA version unknown}"
  [[ "$name" == *L4* ]] || log "warning: expected an NVIDIA L4, found '$name' (the l4 presets may not fit)"
  local major="${driver%%.*}"
  if [[ "$major" =~ ^[0-9]+$ ]]; then
    (( major >= 525 )) || die "NVIDIA driver $driver is too old for PyTorch 2.14 (its CUDA 12.6 build needs 525" \
      "or newer, the CUDA 13.0 build 580 or newer); update the driver, e.g. sudo /opt/deeplearning/install-driver.sh"
    (( major >= 580 )) || log "driver $driver is older than 580: PyTorch's CUDA 12.6 build will be installed"
  fi
}

# ---------------------------------------------------------------------------------------------
# Environment, Python and PyTorch
# ---------------------------------------------------------------------------------------------
clean_environment() {   # a reused VM may pass another project's settings on, also through a tmux server
  if [[ "${KEEP_ENV:-0}" == 1 ]]; then log "KEEP_ENV=1: the inherited environment is kept"; return 0; fi
  local name removed=()
  # pip's network settings (PIP_CERT, PIP_CLIENT_CERT, PIP_PROXY, PIP_TRUSTED_HOST) are kept; the others
  # (PIP_INDEX_URL, PIP_USER, PIP_CONSTRAINT, ...) would change what gets installed.
  for name in PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV \
      CONDA_SHLVL LD_LIBRARY_PATH LD_PRELOAD \
      $(compgen -e | grep -E '^PIP_' | grep -vxE 'PIP_(CERT|CLIENT_CERT|PROXY|TRUSTED_HOST)' || true); do
    if [[ -n "${!name+x}" ]]; then
      removed+=("$name")
      unset "$name"
    fi
  done
  export PIP_CONFIG_FILE=/dev/null     # pip.conf files (user = true, another index, constraints) do not apply
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin${PATH:+:$PATH}"
  if [[ ${#removed[@]} -gt 0 ]]; then log "cleared inherited settings for this run: ${removed[*]}"; fi
}

pick_python() {   # PYTHON; else the system Python first (some images put an older conda Python first on PATH)
  local candidate candidates=(/usr/bin/python3.12 /usr/bin/python3 python3.12 python3)
  [[ -n "${PYTHON:-}" && "$PYTHON" != uv ]] && candidates=("$PYTHON")
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

provision_python() {   # prints the path of a standalone CPython 3.12 under $ROOT/.python (installed with uv)
  local base boot="$ROOT/.python/bootstrap" version="${UV_VERSION:-0.13.2}"
  base="$(command -v /usr/bin/python3 || command -v python3 || true)"
  [[ -n "$base" ]] || { log "error: no python3 to install uv with" >&2; return 1; }
  local why
  why="no Python >= 3.12 here ($("$base" -V 2>&1))"
  [[ "${PYTHON:-}" == uv ]] && why="PYTHON=uv"
  if [[ ! -x "$boot/bin/uv" ]]; then
    log "$why: installing a standalone Python 3.12 into $ROOT/.python with uv $version" \
      "(nothing outside the repository changes)" >&2
    rm -rf "$boot"
    "$base" -m venv "$boot" >&2 || { log "error: $base -m venv failed (install python3-venv)" >&2; return 1; }
    "$boot/bin/python" -m pip install -q --no-cache-dir --retries 10 --timeout 60 "uv==$version" >&2 \
      || { log "error: could not install uv from PyPI" >&2; return 1; }
  fi
  # uv without another project's uv settings (UV_* variables but its TLS ones, uv.toml files); env takes its
  # -u options before the assignments
  local name uv_env=(env)
  for name in $(compgen -e | grep -E '^UV_' | grep -vxE 'UV_(NATIVE_TLS|SYSTEM_CERTS|INSECURE_HOST|HTTP_TIMEOUT)' || true); do
    uv_env+=(-u "$name")
  done
  uv_env+=(UV_NO_CONFIG=1 UV_PYTHON_INSTALL_DIR="$ROOT/.python" UV_CACHE_DIR="$ROOT/.python/cache")
  "${uv_env[@]}" "$boot/bin/uv" python install --no-bin 3.12 >&2 \
    || { log "error: uv could not download Python 3.12 (from GitHub)" >&2; return 1; }
  "${uv_env[@]}" "$boot/bin/uv" python find --managed-python 3.12
}

torch_index_url() {   # PyTorch 2.14 builds: CUDA 13.0 on PyPI (driver >= 580), CUDA 12.6 (driver >= 525)
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then echo "$TORCH_INDEX_URL"; return 0; fi
  [[ "${SKIP_GPU_CHECK:-0}" == 1 ]] && return 0
  local driver
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  driver="${driver%%.*}"
  if [[ "$driver" =~ ^[0-9]+$ ]] && (( driver < 580 )); then echo "https://download.pytorch.org/whl/cu126"; fi
  return 0
}

install_torch() {   # the torch pin of requirements.txt, from INDEX (a PyTorch wheel index) if given
  local index="${1:-}" pin current
  [[ -n "$index" ]] || return 0             # the PyPI build comes with requirements.txt
  pin="$(grep -E '^torch==' "$ROOT/requirements.txt" | head -1)"
  current="$(python -m pip show torch 2>/dev/null | sed -n 's/^Version: //p')"
  [[ "$current" == "${pin#torch==}+${index##*/}" ]] && return 0        # e.g. 2.14.1+cu126 is installed
  log "installing PyTorch ${pin#torch==} from $index"
  if [[ -n "$current" ]]; then python -m pip uninstall -y -q torch; fi
  python -m pip install -q --no-cache-dir --retries 10 --timeout 60 "$pin" --index-url "$index" \
    --extra-index-url https://pypi.org/simple
}

step_system() {
  log "2/7 system packages and Python environment"
  if [[ "${SKIP_APT:-0}" != 1 ]]; then
    "${APT[@]}" update >/dev/null
    "${APT[@]}" install git tmux curl cron ca-certificates python3-venv python3-pip >/dev/null
    sudo -n systemctl enable --now cron >/dev/null 2>&1 || true
  fi
  cron_install
  if [[ -e "$ROOT/.venv" ]] \
      && ! "$ROOT/.venv/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    log "the existing .venv is broken or older than Python 3.12: recreating it"
    rm -rf "$ROOT/.venv"
  fi
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    log "reusing .venv (Python $("$ROOT/.venv/bin/python" -V 2>&1 | cut -d' ' -f2))"
  else
    local python_bin=""
    if [[ "${PYTHON:-}" != uv ]]; then python_bin="$(pick_python)" || python_bin=""; fi
    if [[ -z "$python_bin" ]]; then
      [[ -z "${PYTHON:-}" || "$PYTHON" == uv ]] || die "PYTHON=$PYTHON is not a working Python >= 3.12"
      python_bin="$(provision_python)" || die "no Python 3.12 (see above); install one and set PYTHON=/path/to/python3.12"
    fi
    if [[ "$python_bin" == /usr/bin/* && "${SKIP_APT:-0}" != 1 ]] && ! "$python_bin" -c 'import ensurepip' 2>/dev/null; then
      local venv_pkg   # e.g. python3.12-venv for deadsnakes' python3.12 on Ubuntu 22.04
      venv_pkg="python$("$python_bin" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv"
      "${APT[@]}" install "$venv_pkg" >/dev/null || log "warning: could not install $venv_pkg"
    fi
    log "creating .venv with $python_bin ($("$python_bin" -V 2>&1))"
    "$python_bin" -m venv "$ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  python -m pip install -q --no-cache-dir --upgrade pip
  install_torch "$(torch_index_url)"
  python -m pip install -q --no-cache-dir --retries 10 --timeout 60 -r "$ROOT/requirements-dev.txt"
  python -m pip install -q --no-cache-dir --no-deps -e "$ROOT"
  log "Python $(python -V 2>&1 | cut -d' ' -f2), torch $(python -c 'import torch; print(torch.__version__)')"
}

step_torch() {
  log "3/7 PyTorch on the GPU, evaluation sets, Pass@k harness"
  if [[ "${SKIP_GPU_CHECK:-0}" != 1 ]]; then
    if ! python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
        && [[ "$(python -c 'import torch; print(torch.version.cuda or "")')" != 12.6 ]]; then
      log "PyTorch $(python -c 'import torch; print(torch.__version__)') does not see the GPU; trying its CUDA 12.6 build"
      install_torch https://download.pytorch.org/whl/cu126 || true
    fi
    python - <<'EOF'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see the GPU")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
float((x @ x).float().sum())
props = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}): {props.name}, {props.total_memory / 2**30:.1f} GiB, "
      f"bf16 {torch.cuda.is_bf16_supported()}")
EOF
  fi
  mkdir -p "$DATA_ABS/eval"
  [[ -s "$DATA_ABS/eval/HumanEval.jsonl.gz" ]] || curl -fsSL --retry 5 --retry-delay 10 \
    -o "$DATA_ABS/eval/HumanEval.jsonl.gz" https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
  [[ -s "$DATA_ABS/eval/mbpp.jsonl" ]] || curl -fsSL --retry 5 --retry-delay 10 \
    -o "$DATA_ABS/eval/mbpp.jsonl" https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
  python "$ROOT/evaluate.py" --benchmark humaneval --data "$DATA_ABS/eval/HumanEval.jsonl.gz" --reference \
    --out "$STATE/harness-check/humaneval" >/dev/null
  python "$ROOT/evaluate.py" --benchmark mbpp --data "$DATA_ABS/eval/mbpp.jsonl" --reference \
    --out "$STATE/harness-check/mbpp" >/dev/null
  python - "$STATE/harness-check" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
he = json.loads((root / "humaneval" / "summary.json").read_text())
mb = json.loads((root / "mbpp" / "summary.json").read_text())
print(f"harness check: HumanEval {he['solved_problems']}/{he['problems']}, MBPP {mb['solved_problems']}/{mb['problems']} "
      "reference solutions pass (expected 164/164 and 499/500)")
if he["solved_problems"] != he["problems"] or mb["solved_problems"] < mb["problems"] - 1:
    raise SystemExit("the execution harness does not reproduce the reference results")
EOF
}

step_tests() {
  log "4/7 test suite (informational)"
  if [[ "${RUN_TESTS:-1}" != 1 ]]; then log "skipped (RUN_TESTS=0)"; return 0; fi
  if (cd "$ROOT" && python -m pytest -q -p no:cacheprovider 2>&1 | tail -n 15); then
    log "tests passed"
  else
    log "WARNING: some tests failed (see above); continuing with the run"
  fi
}

step_memory() {
  log "5/7 memory check (worst-case batches)"
  local json="$RUN_ABS/memory_probe.json" choice="$RUN_ABS/batch_overrides"
  mkdir -p "$RUN_ABS"
  if [[ -s "$json" && -f "$choice" ]]; then
    log "done earlier: $(cat "$choice")"
  else
    local candidates=(
      "training.batch_size=16 training.planner.grad_accum_steps=4 training.realizer.grad_accum_steps=4"
      "training.batch_size=8 training.planner.grad_accum_steps=8 training.realizer.grad_accum_steps=8"
      "training.batch_size=4 training.planner.grad_accum_steps=16 training.realizer.grad_accum_steps=16"
    )
    local extra preset_args extra_args ok=0
    read -r -a preset_args <<< "$PRESET"
    for extra in "${candidates[@]}"; do
      read -r -a extra_args <<< "$extra"
      if python -m text_vpa.utils.memory_probe --device "${DEVICE:-auto}" "${preset_args[@]}" "${extra_args[@]}" \
          > "$json.tmp"; then
        mv "$json.tmp" "$json"
        echo "$extra" > "$choice"
        ok=1
        break
      fi
      log "does not fit with: $extra"
    done
    [[ $ok -eq 1 ]] || die "no micro-batch size fits the GPU; see the memory check output above"
  fi
  cat "$json"
  python - "$json" "${STAGE1_STEPS:-50000}" "${STAGE2_STEPS:-60000}" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
steps = {1: int(sys.argv[2]), 2: int(sys.argv[3])}
for s in report["stages"]:
    peak = "n/a" if s["peak_GiB"] is None else f"{s['peak_GiB']} of {s['device_GiB']} GiB"
    hours = s["seconds_per_step"] * steps[s["stage"]] / 3600
    print(f"Stage {s['stage']}: peak {peak}, {s['seconds_per_step']} s per step on worst-case batches, "
          f"at most ~{hours:.1f} h for {steps[s['stage']]} steps (real batches are shorter)")
EOF
  local overrides
  overrides="$(cat "$choice")"
  export PRESET="$PRESET $overrides"
  export STAGE2_PRESET="$STAGE2_PRESET $overrides"
}

step_train() {
  local attempts="${TRAIN_ATTEMPTS:-3}" attempt rc=0
  for attempt in $(seq 1 "$attempts"); do
    log "6/7 corpus -> Stage 1 -> Stage 2 -> Pass@k (attempt $attempt/$attempts)"
    if bash "$ROOT/scripts/gcp/train_l4.sh"; then
      return 0
    else
      rc=$?
    fi
    if [[ $rc -eq 2 ]]; then
      log "Stage 1 was rejected by the paper's stable-rank guard (srank < d/2): not retried"
      return 2
    fi
    if [[ $attempt -lt $attempts ]]; then
      log "train_l4.sh failed (exit $rc); retrying in 60 s from the last checkpoint"
      sleep 60
    fi
  done
  return "$rc"
}

worker() {
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  fi
  clean_environment
  export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
  export RUN="${RUN:-runs/l4}" DATA="${DATA:-data}"
  export PRESET="${PRESET:-planner=base realizer=l4 data=code training=l4}"
  export STAGE2_PRESET="${STAGE2_PRESET:-realizer=l4 training=l4}"
  RUN_ABS="$ROOT/$RUN"; [[ "$RUN" == /* ]] && RUN_ABS="$RUN"
  DATA_ABS="$ROOT/$DATA"; [[ "$DATA" == /* ]] && DATA_ABS="$DATA"
  trap 'finish $?' EXIT
  trap 'finish 130' INT
  trap 'finish 129' HUP
  trap 'finish 143' TERM
  log "text-VPA unattended run on $(hostname): commit $(git -C "$ROOT" log -1 --format='%h %s' 2>/dev/null || echo unknown)"
  log "settings: RUN=$RUN DATA=$DATA CORPUS=${CORPUS:-shards} MAX_FUNCTIONS=${MAX_FUNCTIONS:-2000000}" \
    "STAGE1_STEPS=${STAGE1_STEPS:-50000} STAGE2_STEPS=${STAGE2_STEPS:-60000} NUM_SAMPLES=${NUM_SAMPLES:-10}" \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
  step_gpu
  step_system
  step_torch
  step_tests
  step_memory
  step_train
  log "7/7 done"
}

# ---------------------------------------------------------------------------------------------
# Status and cancel
# ---------------------------------------------------------------------------------------------
status() {
  migrate_markers
  local m run_abs f
  run_abs="$(run_dir)"
  if tmux has-session -t "$SESSION" 2>/dev/null; then echo "running: tmux attach -t $SESSION"; else echo "not running"; fi
  echo "run directory: $run_abs"
  for m in "done" failed interrupted; do
    [[ -f "$run_abs/run_all.$m" ]] && echo "$m: $(cat "$run_abs/run_all.$m")"
  done
  for f in "$STATE"/*/run_all.done "$STATE"/*/run_all.failed "$STATE"/*/run_all.interrupted; do
    [[ -f "$f" && "$(dirname "$f")" != "$run_abs" ]] && echo "other run $(dirname "$f"): ${f##*.} ($(cat "$f"))"
  done
  if crontab -l 2>/dev/null | grep -qF "$CRON_TAG"; then echo "resume after an unexpected restart: on"; fi
  if [[ -e "$LOG_DIR/latest.log" ]]; then
    echo "--- $(readlink -f "$LOG_DIR/latest.log")"
    tail -n 15 "$LOG_DIR/latest.log"
  fi
}

cancel() {
  local run_abs
  run_abs="$(run_dir)"
  mkdir -p "$run_abs"
  touch "$run_abs/run_all.cancel"
  cron_remove
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux send-keys -t "$SESSION" C-c
    sleep 15
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "cancelled; the VM keeps running. Re-run the start command to resume from the last checkpoint."
  else
    rm -f "$run_abs/run_all.cancel"
    echo "no run in progress"
  fi
}

case "${1:-}" in
  "") launch 0 ;;
  --from-reboot) launch 1 ;;
  --worker) worker ;;
  --status) status ;;
  --cancel) cancel ;;
  -h|--help) sed -n '5,49p' "$SELF" ;;
  *) echo "unknown argument: $1 (use --status, --cancel or no argument)" >&2; exit 2 ;;
esac
