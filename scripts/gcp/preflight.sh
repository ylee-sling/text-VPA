#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# preflight.sh -- read-only checks of a VM before scripts/gcp/run_all.sh: a fresh VM or one reused from another
# project. Nothing is installed, started or changed.
#
#   bash ~/text-VPA/scripts/gcp/preflight.sh     # exit 0: ready (read any WARN lines); 1: fix the FAIL lines
#
# Checks:
#   system      OS, vCPUs, memory; passwordless sudo; free disk space (MIN_FREE_GB [60])
#   GPU         nvidia-smi, GPU model and memory, NVIDIA driver -> PyTorch build (CUDA 13.0 from driver 580,
#               CUDA 12.6 from 525)
#   other work  processes on the GPU (a failure unless ALLOW_BUSY_GPU=1), tmux and screen sessions, containers
#               and CPU load -- the end-of-run shutdown (SHUTDOWN_ON_SUCCESS, SHUTDOWN_ON_FAILURE) stops them
#   Python      a Python >= 3.12 for .venv (otherwise run_all.sh installs a standalone one); an existing .venv
#   settings    what another project can pass on to the run: PYTHONPATH, LD_LIBRARY_PATH, conda, pip variables
#               and configuration files, the environment of a running tmux server, CUDA_VISIBLE_DEVICES
#   text-VPA    the clone (branch, local changes, behind GitHub); an earlier run, corpus and checkpoints
#   network     GitHub, PyPI, the PyTorch wheel index (drivers older than 580), HumanEval and MBPP, and the
#               corpus files on Hugging Face (CORPUS, HF_DATASET, HF_CONFIG, HF_TOKEN as in train_l4.sh)
# The other variables run_all.sh reads (PYTHON, TORCH_INDEX_URL, KEEP_ENV, CORPUS_SOURCE, DATA, RUN, SESSION)
# are taken into account.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
HF_DATASET="${HF_DATASET:-codeparrot/github-code-clean}"
HF_CONFIG="${HF_CONFIG:-Python-all}"
HF_FILES="${HF_FILES:-10}"
CORPUS="${CORPUS:-shards}"
SESSION="${SESSION:-textvpa}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-1}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-1}"
DATA_DIR="${DATA:-data}"; [[ "$DATA_DIR" == /* ]] || DATA_DIR="$ROOT/$DATA_DIR"
RUN_DIR="${RUN:-runs/l4}"; [[ "$RUN_DIR" == /* ]] || RUN_DIR="$ROOT/$RUN_DIR"
WAIT=25                                   # seconds per network request
LEAKY='PYTHONPATH|PYTHONHOME|PYTHONSTARTUP|PYTHONUSERBASE|VIRTUAL_ENV|CONDA_PREFIX|CONDA_DEFAULT_ENV|LD_LIBRARY_PATH|LD_PRELOAD'
# pip variables that run_all.sh keeps (network settings) or that cannot change what gets installed
PIP_HARMLESS='PIP_(CERT|CLIENT_CERT|PROXY|TRUSTED_HOST|DISABLE_PIP_VERSION_CHECK|NO_INPUT|ROOT_USER_ACTION|PROGRESS_BAR|NO_COLOR|TIMEOUT|RETRIES|CACHE_DIR|NO_CACHE_DIR)'

if [[ -t 1 ]]; then
  GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; BOLD=$'\e[1m'; PLAIN=$'\e[0m'
else
  GREEN=""; YELLOW=""; RED=""; BOLD=""; PLAIN=""
fi
passes=0; warnings=0; failures=0
pass() { passes=$((passes + 1)); echo "  ${GREEN}PASS${PLAIN}  $*"; }
warn() { warnings=$((warnings + 1)); echo "  ${YELLOW}WARN${PLAIN}  $*"; }
fail() { failures=$((failures + 1)); echo "  ${RED}FAIL${PLAIN}  $*"; }
info() { echo "  info  $*"; }
section() { echo; echo "${BOLD}== $*${PLAIN}"; }
have() { command -v "$1" >/dev/null 2>&1; }
short() { local s="$*"; (( ${#s} > 70 )) && s="${s:0:67}..."; echo "$s"; }

shutdown_note=""
if [[ "$SHUTDOWN_ON_SUCCESS" == 1 || "$SHUTDOWN_ON_FAILURE" == 1 ]]; then
  shutdown_note="run_all.sh shuts the VM down when the run ends (SHUTDOWN_ON_SUCCESS=$SHUTDOWN_ON_SUCCESS,"
  shutdown_note+=" SHUTDOWN_ON_FAILURE=$SHUTDOWN_ON_FAILURE), which stops it; set both to 0 to keep the VM up"
fi
JSON_PY="$(command -v /usr/bin/python3 || command -v python3 || true)"   # any Python 3 reads JSON

echo "${BOLD}text-VPA pre-flight check${PLAIN} -- $(hostname), $(date -u '+%F %T') UTC, repository $ROOT"

# ---------------------------------------------------------------------------------------------
section "System"
os_name="unknown OS"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  os_name="$(. /etc/os-release && echo "${PRETTY_NAME:-unknown OS}")"
fi
cpus="$(nproc 2>/dev/null || echo 1)"
mem_gib="$(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo 2>/dev/null)"
info "$os_name, kernel $(uname -r), $cpus vCPUs"
if awk -v m="${mem_gib:-0}" 'BEGIN {exit !(m >= 30)}'; then
  pass "memory: $mem_gib GiB (the l4 presets were sized for 64 GB)"
else
  warn "memory: ${mem_gib:-?} GiB; the l4 presets were sized for 64 GB (g2-standard-16): the corpus and dataset" \
    "builds may run out of memory"
fi
if sudo -n true 2>/dev/null; then
  pass "passwordless sudo (run_all.sh uses it for apt-get and the final shutdown)"
else
  fail "no passwordless sudo: run_all.sh needs it for apt-get and the final shutdown; log in as a VM admin" \
    "(OS Login: roles/compute.osAdminLogin)"
fi
avail="$(df --output=avail -BG "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
mount_point="$(df --output=target "$ROOT" 2>/dev/null | tail -1)"
if [[ -z "$avail" ]]; then
  fail "cannot read the free space of $ROOT"
else
  uses="about 30 GB"; [[ "$CORPUS" == shards ]] && uses="about 60 GB with CORPUS=shards"
  if (( avail >= MIN_FREE_GB )); then
    pass "disk: $avail GB free on $mount_point (run_all.sh needs $MIN_FREE_GB; the run uses $uses)"
  else
    fail "disk: $avail GB free on $mount_point; run_all.sh needs $MIN_FREE_GB GB (the run uses $uses):" \
      "free space or enlarge the boot disk"
  fi
fi

# ---------------------------------------------------------------------------------------------
section "GPU and NVIDIA driver"
driver_major=""; gpu_used=0; gpu_procs=""
if ! have nvidia-smi; then
  fail "nvidia-smi not found: no NVIDIA driver (on a Deep Learning VM image: sudo /opt/deeplearning/install-driver.sh)"
elif ! smi_out="$(nvidia-smi -L 2>&1)"; then
  fail "nvidia-smi fails ($(short "$(head -1 <<< "$smi_out")")): after a kernel update the driver often needs a" \
    "reboot or a reinstall (sudo /opt/deeplearning/install-driver.sh)"
else
  IFS=',' read -r gpu_name gpu_total gpu_used driver < <(nvidia-smi \
    --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits | head -1)
  gpu_name="$(xargs <<< "$gpu_name")"; gpu_total="$(xargs <<< "$gpu_total")"
  gpu_used="$(xargs <<< "$gpu_used")"; driver="$(xargs <<< "$driver")"
  gpus="$(grep -c '^GPU ' <<< "$smi_out")"
  gpu_gib="$(awk -v m="$gpu_total" 'BEGIN {printf "%.1f", m / 1024}')"
  cuda="$(nvidia-smi | grep -o 'CUDA Version: *[0-9.]*' | head -1 | awk '{print $3}')"
  if [[ "$gpu_name" == *L4* ]]; then
    pass "GPU: $gpu_name, $gpu_gib GiB ($gpus GPU(s))"
  else
    warn "GPU: $gpu_name, $gpu_gib GiB: the l4 presets were sized for an NVIDIA L4 (24 GB); the memory" \
      "check lowers the micro-batch if they do not fit"
  fi
  driver_major="${driver%%.*}"
  if ! [[ "$driver_major" =~ ^[0-9]+$ ]]; then
    warn "cannot read the driver version ('$driver')"; driver_major=""
  elif (( driver_major >= 580 )); then
    pass "driver $driver (supports CUDA ${cuda:-?}): PyTorch's default CUDA 13.0 build"
  elif (( driver_major >= 525 )); then
    pass "driver $driver (supports CUDA ${cuda:-?}): run_all.sh installs PyTorch's CUDA 12.6 build (CUDA 13.0 needs 580+)"
  else
    fail "driver $driver is too old for PyTorch 2.14 (CUDA 12.6 build: 525 or newer; CUDA 13.0: 580 or newer)"
  fi
  [[ -n "${TORCH_INDEX_URL:-}" ]] && info "TORCH_INDEX_URL=$TORCH_INDEX_URL: PyTorch is installed from there"
  if have nvcc; then
    info "CUDA toolkit $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}') is installed:" \
      "not used (PyTorch brings its own CUDA libraries)"
  fi
  gpu_procs="$(nvidia-smi -q -d PIDS 2>/dev/null | awk '
    function value(line) { sub(/^[^:]*:[ \t]*/, "", line); return line }
    /Process ID/      { pid = value($0) }
    /^[ \t]*Type/     { type = value($0) }
    /^[ \t]*Name/     { name = value($0) }
    /Used GPU Memory/ { printf "%s (pid %s, %s, type %s); ", name, pid, value($0), type }')"
fi

# ---------------------------------------------------------------------------------------------
section "Other work on this VM"
others=0
if [[ -n "$gpu_procs" ]]; then
  others=1
  if [[ "${ALLOW_BUSY_GPU:-0}" == 1 ]]; then
    warn "the GPU is in use: ${gpu_procs%; }. Training shares the L4 (slower; it can run out of GPU memory)." \
      "${shutdown_note:+$shutdown_note.}"
  else
    fail "the GPU is in use: ${gpu_procs%; }. Stop it first (training would share the L4 and can run out of GPU" \
      "memory${shutdown_note:+; $shutdown_note}), or accept sharing with ALLOW_BUSY_GPU=1"
  fi
elif [[ "$gpu_used" =~ ^[0-9]+$ ]] && (( gpu_used > 1024 )); then
  others=1
  warn "$gpu_used MiB of GPU memory are in use by processes not visible here (containers?)"
fi
if have tmux && tmux list-sessions >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    info "tmux session '$SESSION' exists: a text-VPA run is in progress (bash $ROOT/scripts/gcp/run_all.sh --status)"
  fi
  sessions="$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -vxF "$SESSION" | paste -sd ' ' -)"
  if [[ -n "$sessions" ]]; then
    others=1
    if [[ -n "$shutdown_note" ]]; then warn "other tmux sessions are open: $sessions. $shutdown_note"
    else info "other tmux sessions are open: $sessions (the VM keeps running after the run)"; fi
  fi
fi
if have screen; then
  screens="$(screen -ls 2>/dev/null | awk '/^\t[0-9]+\./ {print $1}' | paste -sd ' ' -)"
  if [[ -n "$screens" ]]; then
    others=1
    if [[ -n "$shutdown_note" ]]; then warn "screen sessions are open: $screens. $shutdown_note"
    else info "screen sessions are open: $screens"; fi
  fi
fi
if have docker; then
  containers="$( (docker ps --format '{{.Names}}' 2>/dev/null || sudo -n docker ps --format '{{.Names}}' 2>/dev/null) \
    | paste -sd ' ' -)"
  if [[ -n "$containers" ]]; then
    others=1
    if [[ -n "$shutdown_note" ]]; then warn "containers are running: $containers. $shutdown_note"
    else info "containers are running: $containers"; fi
  fi
fi
load="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
if awk -v l="$load" -v c="$cpus" 'BEGIN {exit !(l > c / 2)}'; then
  others=1
  top="$(ps -eo pid=,pcpu=,comm= --sort=-pcpu 2>/dev/null | head -3 \
    | awk '{printf "%s%s (pid %s, %s%% CPU)", (NR > 1 ? ", " : ""), $3, $1, $2}')"
  warn "the CPUs are busy (load $load on $cpus vCPUs): $top${shutdown_note:+. $shutdown_note}"
fi
if (( others == 0 )); then pass "nothing else running that the run would compete with or that a shutdown would stop"; fi
if [[ -n "$shutdown_note" ]]; then
  info "at the end, run_all.sh shuts the VM down (SHUTDOWN_ON_SUCCESS=$SHUTDOWN_ON_SUCCESS," \
    "SHUTDOWN_ON_FAILURE=$SHUTDOWN_ON_FAILURE); logs, checkpoints and results stay on the disk"
else
  info "at the end, the VM keeps running (SHUTDOWN_ON_SUCCESS=0, SHUTDOWN_ON_FAILURE=0)"
fi

# ---------------------------------------------------------------------------------------------
section "Python"
py=""; need_uv=0
if [[ "${PYTHON:-}" == uv ]]; then
  need_uv=1
  info "PYTHON=uv: run_all.sh installs a standalone Python 3.12 into $ROOT/.python"
else
  candidates=(/usr/bin/python3.12 /usr/bin/python3 python3.12 python3)
  [[ -n "${PYTHON:-}" ]] && candidates=("$PYTHON")
  for candidate in "${candidates[@]}"; do
    if have "$candidate" && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
    then py="$(command -v "$candidate")"; break; fi
  done
  if [[ -n "$py" ]]; then
    pass "Python $("$py" -V 2>&1 | cut -d' ' -f2) at $py for .venv"
    if ! "$py" -c 'import ensurepip' 2>/dev/null; then
      info "$py lacks ensurepip: run_all.sh installs python$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv"
    fi
  elif [[ -n "${PYTHON:-}" ]]; then
    fail "PYTHON=$PYTHON is not a working Python >= 3.12"
  else
    need_uv=1
    system_py="$(command -v /usr/bin/python3 || command -v python3 || true)"
    if [[ -z "$system_py" ]]; then
      fail "no Python at all: sudo apt-get install -y python3 python3-venv"
    else
      warn "no Python >= 3.12 (python3 is $("$system_py" -V 2>&1 | cut -d' ' -f2)): run_all.sh installs a standalone" \
        "Python 3.12 with uv into $ROOT/.python (from PyPI and GitHub; nothing outside the repository changes)"
    fi
  fi
fi
if [[ -n "${CONDA_PREFIX:-}" ]] || have conda; then
  info "conda is installed${CONDA_PREFIX:+ (active: $CONDA_PREFIX)}: the run uses its own .venv, not conda"
fi
if [[ -e "$ROOT/.venv" ]]; then
  if venv_version="$("$ROOT/.venv/bin/python" -c 'import sys
assert sys.version_info >= (3, 12)
print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"; then
    info "the existing .venv (Python $venv_version) is reused; pip brings it to the pinned versions"
  else
    warn "the existing .venv is broken or older than Python 3.12: run_all.sh recreates it"
  fi
fi

# ---------------------------------------------------------------------------------------------
section "Settings another project can pass on"
found="$(compgen -e | grep -xE "$LEAKY|PIP_[A-Z_]+" | grep -vxE "$PIP_HARMLESS" | paste -sd ' ' -)"
if [[ -n "$found" ]]; then
  if [[ "${KEEP_ENV:-0}" == 1 ]]; then warn "this shell sets $found; KEEP_ENV=1 passes them to the run"
  else info "this shell sets $found: run_all.sh clears them for the run"; fi
fi
tmux_env=""
if have tmux && tmux list-sessions >/dev/null 2>&1; then
  # New tmux sessions get the environment the tmux server started with, not this shell's.
  tmux_env="$(tmux show-environment -g 2>/dev/null | grep -E "^($LEAKY|PIP_[A-Z_]+)=" | cut -d= -f1 \
    | grep -vxE "$PIP_HARMLESS" | paste -sd ' ' -)"
  tmux_cuda="$(tmux show-environment -g CUDA_VISIBLE_DEVICES 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' || true)"
  tmux_env="$tmux_env${tmux_cuda:+ $tmux_cuda}"; tmux_env="${tmux_env# }"
  if [[ -n "$tmux_env" ]]; then
    if [[ "${KEEP_ENV:-0}" == 1 ]]; then
      warn "the running tmux server passes $tmux_env to new sessions, and KEEP_ENV=1 keeps them"
    else
      info "the running tmux server passes $tmux_env to new sessions: run_all.sh clears them and uses this" \
        "shell's CUDA_VISIBLE_DEVICES"
    fi
  fi
fi
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  if [[ "$CUDA_VISIBLE_DEVICES" == 0 || "$CUDA_VISIBLE_DEVICES" == GPU-* ]]; then
    info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is passed to the run"
  else
    warn "CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' is passed to the run and can hide the GPU:" \
      "unset CUDA_VISIBLE_DEVICES before launching"
  fi
fi
pip_files=()
for f in /etc/pip.conf /etc/xdg/pip/pip.conf "$HOME/.pip/pip.conf" "${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf"; do
  [[ -f "$f" ]] && pip_files+=("$f")
done
if (( ${#pip_files[@]} )); then
  if grep -qiE '^[[:space:]]*(cert|client-cert|proxy|trusted-host)[[:space:]]*=' "${pip_files[@]}" 2>/dev/null; then
    warn "pip configuration files ${pip_files[*]} set network options, but the run ignores pip.conf files:" \
      "export PIP_CERT / PIP_PROXY / PIP_TRUSTED_HOST instead"
  else
    info "pip configuration files ${pip_files[*]}: ignored by the run"
  fi
fi
if [[ -z "$found" && -z "$tmux_env" ]] && (( ${#pip_files[@]} == 0 )); then
  pass "no Python, pip, conda or library-path settings to leak into the run"
fi

# ---------------------------------------------------------------------------------------------
section "text-VPA"
corpus_built=0
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  branch="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  pass "clone at $ROOT: $branch, $(git -C "$ROOT" log -1 --format='%h %s' | cut -c1-70)"
  changed="$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
  if (( changed > 0 )); then warn "$changed locally modified file(s): git pull --ff-only can refuse (git -C $ROOT status)"; fi
  remote_head="$(timeout "$WAIT" git -C "$ROOT" ls-remote origin "refs/heads/$branch" 2>/dev/null | cut -f1)"
  if [[ -n "$remote_head" && "$remote_head" != "$(git -C "$ROOT" rev-parse HEAD)" ]]; then
    warn "GitHub's $branch is at ${remote_head:0:7}, this clone at $(git -C "$ROOT" rev-parse --short HEAD):" \
      "git -C $ROOT pull --ff-only"
  fi
else
  warn "$ROOT is not a git clone: it cannot be updated with git pull"
fi
done_marker="$RUN_DIR/run_all.done"
[[ ! -f "$done_marker" && "$RUN_DIR" == "$ROOT/runs/l4" && -f "$ROOT/runs/run_all.done" ]] && done_marker="$ROOT/runs/run_all.done"
if [[ -f "$done_marker" ]]; then
  warn "the run in $RUN_DIR finished earlier ($(cat "$done_marker")): run_all.sh does not start it again. Another" \
    "run takes a new RUN (and DATA for another corpus), e.g. RUN=runs/run2 DATA=data2"
fi
for marker in failed interrupted; do
  for f in "$RUN_DIR/run_all.$marker" "$ROOT/runs/run_all.$marker"; do
    [[ "$f" == "$ROOT/runs/run_all.$marker" && "$RUN_DIR" != "$ROOT/runs/l4" ]] && continue
    if [[ -f "$f" ]]; then
      info "the previous run $marker ($(cat "$f")): run_all.sh resumes from the last checkpoint"
      break
    fi
  done
done
if [[ -s "$DATA_DIR/corpus/stats.json" ]]; then
  corpus_built=1
  recipe="$(cat "$DATA_DIR/corpus/recipe" 2>/dev/null || echo "export (no recipe recorded)")"
  if [[ -z "${CORPUS_SOURCE:-}" && "$recipe" != "$CORPUS"* ]]; then
    fail "$DATA_DIR/corpus was built by another recipe ($recipe) than CORPUS=$CORPUS: train_l4.sh refuses it;" \
      "use a new DATA directory (e.g. DATA=data2)"
  else
    info "the corpus is built ($DATA_DIR/corpus: $recipe): no download"
  fi
fi
[[ -s "$RUN_DIR/stage1/stage1_final.pt" ]] && info "Stage 1 finished earlier ($RUN_DIR/stage1/stage1_final.pt): it is not rerun"
[[ -s "$RUN_DIR/stage2/stage2_final.pt" ]] && info "Stage 2 finished earlier ($RUN_DIR/stage2/stage2_final.pt): it is not rerun"

# ---------------------------------------------------------------------------------------------
section "Network"
get_ok() { curl -fsS -o /dev/null --max-time "$WAIT" --retry 2 "$@" 2>/dev/null; }
answers() {   # any HTTP answer (even an error status) proves the host is reachable
  local code; code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$WAIT" "$1" 2>/dev/null)"
  [[ -n "$code" && "$code" != 000 ]]
}
if ! have curl; then
  fail "curl is missing (sudo apt-get install -y curl); the network checks are skipped"
else
  if timeout "$WAIT" git ls-remote --heads https://github.com/ylee-sling/text-VPA.git main >/dev/null 2>&1; then
    pass "GitHub: the text-VPA repository"
  else
    fail "cannot reach github.com"
  fi
  if get_ok https://pypi.org/simple/torch/ && answers https://files.pythonhosted.org/; then
    pass "PyPI (pypi.org, files.pythonhosted.org)"
  else
    fail "cannot reach PyPI (pypi.org / files.pythonhosted.org): the Python environment cannot be installed"
  fi
  torch_index="${TORCH_INDEX_URL:-}"
  if [[ -z "$torch_index" && -n "$driver_major" ]] && (( driver_major < 580 )); then
    torch_index="https://download.pytorch.org/whl/cu126"
  fi
  if [[ -n "$torch_index" ]]; then
    if get_ok "${torch_index%/}/torch/"; then pass "PyTorch wheel index ($torch_index)"
    else fail "cannot reach the PyTorch wheel index $torch_index"; fi
  fi
  if (( need_uv )); then
    if answers https://github.com/astral-sh/python-build-standalone/releases/latest; then
      pass "GitHub releases (the standalone Python 3.12)"
    else
      fail "cannot reach GitHub releases: the standalone Python 3.12 cannot be downloaded"
    fi
  fi
  eval_urls=()
  [[ -s "$DATA_DIR/eval/HumanEval.jsonl.gz" ]] \
    || eval_urls+=(https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz)
  [[ -s "$DATA_DIR/eval/mbpp.jsonl" ]] \
    || eval_urls+=(https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl)
  if (( ${#eval_urls[@]} )); then
    eval_ok=1
    for u in "${eval_urls[@]}"; do get_ok -r 0-0 "$u" || eval_ok=0; done
    if (( eval_ok )); then pass "HumanEval and MBPP (raw.githubusercontent.com)"
    else fail "cannot download HumanEval / MBPP from raw.githubusercontent.com"; fi
  fi

  if (( corpus_built )) || [[ -n "${CORPUS_SOURCE:-}" ]]; then
    info "Hugging Face is not needed (${CORPUS_SOURCE:+CORPUS_SOURCE=$CORPUS_SOURCE}${CORPUS_SOURCE:-corpus built})"
  elif [[ -z "$JSON_PY" ]]; then
    warn "no python3 to read Hugging Face's file listing; the corpus download is not checked"
  elif [[ "$CORPUS" == shards ]]; then
    # download-shards: the Parquet files under data/ of the repository's main branch, then one ranged read
    auth=()
    [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    tree="https://huggingface.co/api/datasets/$HF_DATASET/tree/main/data?recursive=true"
    response="$(curl -sS --max-time "$WAIT" --retry 2 "${auth[@]}" -w $'\n%{http_code}' "$tree" 2>/dev/null)"
    status="${response##*$'\n'}"
    read -r count gigabytes first < <(printf '%s' "${response%$'\n'*}" | "$JSON_PY" -I -c '
import json, sys, urllib.parse
try:
    entries = json.load(sys.stdin)
except ValueError:
    entries = []
files = sorted((e["path"], e.get("size") or 0) for e in (entries if isinstance(entries, list) else [])
               if isinstance(e, dict) and e.get("type") == "file" and str(e.get("path", "")).endswith(".parquet"))
url = (f"https://huggingface.co/datasets/{sys.argv[1]}/resolve/main/{urllib.parse.quote(files[0][0])}"
       if files else "-")
print(len(files), round(sum(s for _, s in files) / 1e9), url)' "$HF_DATASET")
    if [[ "${count:-0}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
      magic="$(curl -sSL --max-time "$WAIT" -r 0-3 "${auth[@]}" "$first" 2>/dev/null | head -c 4)"
      if [[ "$magic" == PAR1 ]]; then
        pass "Hugging Face: $count shard(s) of $HF_DATASET listed under data/ (about $gigabytes GB, the first" \
          "page of the listing); the first one downloads"
      else
        fail "Hugging Face lists $count shard(s) of $HF_DATASET, but the first one does not download: $first"
      fi
    elif [[ -z "$status" || "$status" == 000 ]]; then
      fail "cannot reach huggingface.co (the corpus download): check the VM's external IP or Cloud NAT and firewall"
    else
      fail "Hugging Face: no shards of $HF_DATASET under data/: $tree answered HTTP $status" \
        "${HF_TOKEN:+(HF_TOKEN was sent)}"
    fi
  else
    auth=()
    [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    listing="https://huggingface.co/api/datasets/$HF_DATASET/parquet/$HF_CONFIG/train"
    response="$(curl -sS --max-time "$WAIT" --retry 2 "${auth[@]}" -w $'\n%{http_code}' "$listing" 2>/dev/null)"
    status="${response##*$'\n'}"
    read -r count first < <(printf '%s' "${response%$'\n'*}" | "$JSON_PY" -I -c '
import json, sys
try:
    urls = json.load(sys.stdin)
except ValueError:
    urls = None
urls = urls if isinstance(urls, list) else []
print(len(urls), urls[0] if urls else "-")')
    where="listed by the Hub"
    if [[ "${count:-0}" == 0 ]]; then
      # The downloader's fallback: the files of the Parquet export (refs/convert/parquet) of this split.
      tree="https://huggingface.co/api/datasets/$HF_DATASET/tree/refs%2Fconvert%2Fparquet/$HF_CONFIG?recursive=true"
      read -r count first < <(curl -sS --max-time "$WAIT" --retry 2 "${auth[@]}" "$tree" 2>/dev/null \
        | "$JSON_PY" -I -c '
import json, sys, urllib.parse
dataset, config = sys.argv[1], sys.argv[2]
try:
    entries = json.load(sys.stdin)
except ValueError:
    entries = []
folders = {f"{config}/train", f"{config}/partial-train", f"{config}/partial/train"}
paths = sorted(e["path"] for e in entries if isinstance(e, dict) and e.get("type") == "file"
               and str(e.get("path", "")).endswith(".parquet") and e["path"].rsplit("/", 1)[0] in folders)
url = (f"https://huggingface.co/datasets/{dataset}/resolve/refs%2Fconvert%2Fparquet/{urllib.parse.quote(paths[0])}"
       if paths else "-")
print(len(paths), url)' "$HF_DATASET" "$HF_CONFIG")
      where="in the dataset's Parquet export (the downloader falls back to it)"
    fi
    if [[ "${count:-0}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
      magic="$(curl -sSL --max-time "$WAIT" -r 0-3 "${auth[@]}" "$first" 2>/dev/null | head -c 4)"
      if [[ "$magic" == PAR1 ]]; then
        pass "Hugging Face: $count Parquet file(s) of $HF_DATASET ($HF_CONFIG) $where; the first one downloads" \
          "(the run takes $HF_FILES)"
      else
        fail "Hugging Face lists $count Parquet file(s) of $HF_DATASET, but the first one does not download: $first"
      fi
    elif [[ -z "$status" || "$status" == 000 ]]; then
      fail "cannot reach huggingface.co (the corpus download): check the VM's external IP or Cloud NAT and firewall"
    else
      fail "Hugging Face: no Parquet files of $HF_DATASET ($HF_CONFIG): $listing answered HTTP $status and the" \
        "export tree lists none${HF_TOKEN:+ (HF_TOKEN was sent)}"
    fi
  fi
fi

# ---------------------------------------------------------------------------------------------
echo
echo "${BOLD}== $passes passed, $warnings warning(s), $failures failure(s)${PLAIN}"
if (( failures > 0 )); then
  echo "Not ready: fix the FAIL lines above, then run this check again."
  exit 1
fi
if (( warnings > 0 )); then
  echo "Ready, once you have read the WARN lines above. Start the run with: bash $ROOT/scripts/gcp/run_all.sh"
else
  echo "Ready. Start the run with: bash $ROOT/scripts/gcp/run_all.sh"
fi
