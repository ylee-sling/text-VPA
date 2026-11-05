# text-VPA

**PR-JEPA for non-autoregressive code and text generation**: a JEPA planner decides the structure of a
program once, and a few-step discrete flow realizer writes all of its tokens in parallel, in K + 3
sequential network evaluations whatever the output length.

[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23498811.svg)](https://doi.org/10.5281/zenodo.23498811)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)

This repository is the reference implementation of the preprint
*PR-JEPA: Decoupling Latent Planning from Few-Step Discrete Realization for Non-Autoregressive Generation
of Code, Legal Documents, and Control* (Yeonseok Lee, SLING AI Inc.,
[doi:10.5281/zenodo.23498811](https://doi.org/10.5281/zenodo.23498811); copy in
[`docs/pr_jepa_preprint.pdf`](docs/pr_jepa_preprint.pdf)). It extends the
[Vision-Predictor-Action](https://github.com/ylee-sling/Vision-Predictor-Action) framework from continuous
actions to discrete tokens.

## Status

> [!IMPORTANT]
> **A research prototype with equation-level verification. No released checkpoints and no competitive
> benchmark results yet.** The paper's evaluation protocol (Sec. 5) has not been run at the paper's scale. A
> first L4 run on a 61K-function corpus scored 0% Pass@k: its realizer reproduces programs it trained on but
> did not generalize, and the hard plan-vocabulary energy blocked valid code. The second run uses a
> 2M-function corpus and a soft, calibrated energy (see "Training on a GCP L4 VM"). What is verified:
> - 1,150 unit and integration tests check every equation against independent float64 references, the
>   K + 3 depth, Proposition 4(d), and the Stage-2 anti-leakage invariant;
> - the Pass@k harness passes every HumanEval reference solution (164/164) and 499/500 of MBPP's test split;
> - `scripts/gcp/train_l4.sh` (corpus → Stage 1 → Stage 2 → Pass@k) runs end to end on an L4 (first run)
>   and resumes after an interruption; the full-corpus pipeline is checked on CPU with tiny settings against
>   a local stand-in for Hugging Face;
> - `scripts/gcp/run_all.sh` also runs end to end on a simulated reused VM (checked on CPU): a tmux server that
>   passes another project's `PYTHONPATH`, pip index, `PIP_USER`, uv settings and `CUDA_VISIBLE_DEVICES` on,
>   no system Python 3.12, and a broken `.venv`;
> - the numerical checks of Sec. 4.6 reproduce Table 2 exactly (`python -m text_vpa.utils.verification`);
> - the roofline model reproduces every cell of Table 3 (`python -m text_vpa.utils.profiler --roofline`);
> - the full pipeline (Stage 1 → Stage 2 → Algorithm 1) runs end to end on CPU and with two-process DDP, and
>   a small model trained on two programs of the same canvas length writes, for each context, its own
>   program as valid Python (`pytest -m slow`).

## How it works

```mermaid
flowchart LR
    C["context c<br/>(spec, signature)"] --> F["① f_θ<br/>context encoder"]
    F --> SX["s_x"]
    SX --> P["② π_φ^h<br/>mode selector"]
    P --> U["u ~ π(·|s_x)"]
    SX --> G["③ g_φ<br/>JEPA predictor, n_ens heads"]
    U --> G
    G --> Z["Z_target, σ̄"]
    Z --> B["bridge K/V of [Z; s_x; e(u)]/γ<br/>+ plan readout<br/>(once, fused into pass 4)"]
    B --> R["④ … ⑪ F_ψ × K<br/>all N positions in parallel"]
    R --> W["w ∈ V^N"]
    E["constraint energies E(j | w, Z)<br/>tilt + clip (Eq. 17)"] -.-> R
```

1. **Plan (passes 1–3, once).** The context encoder f_θ embeds the specification. The mode selector π_φ^h
   samples a discrete design mode u (control flow × library family), which resolves multimodality
   (Prop. 1). The JEPA predictor g_φ predicts the plan Z_target ∈ R^{M×d} over the program's structural slots
   (imports, signatures, class skeletons, declared identifiers, block headers), with an ensemble spread σ̄.
   It is trained against an EMA target encoder with VICReg (Eqs. 4–8).
2. **Bridge (once).** Keys and values of Z̃ = [Z_target; s_x; e(u)]/γ are computed once, with spectrally
   normalized projections, and reused at every realizer pass. Their memory is independent of N (Eq. 10).
3. **Realize (passes 4 … K + 3).** The FS-DFM realizer refines all N positions in parallel with the exact
   few-step jump kernel, tilted by constraint energies and clipped to remain valid (Eqs. 11–17, Algorithm 1).
   The Logit-KL realizer instead integrates straight lines in logit space (Eqs. 18–21).

`PRJEPASampler` audits at run time that a generation costs exactly K + 3 sequential network evaluations,
and that every parallel update keeps plan-admissible positions admissible (Prop. 4(d)).

## Installation

Python ≥ 3.10 (the exact pins of `requirements*.txt` need Python ≥ 3.12, because numpy 2.5 does). Install
PyTorch for your backend first ([pytorch.org](https://pytorch.org/get-started/locally/)): CUDA, ROCm, Apple MPS
and CPU are supported. Then:

```bash
git clone https://github.com/ylee-sling/text-VPA.git && cd text-VPA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # exact verified versions (torch 2.14.1, transformers 5.21, tree-sitter 0.26)
pip install -e .                         # the text_vpa package (setup.py states minimum versions: torch >= 2.2)
python -m pytest                         # ~1 minute on CPU
```

## Quick start (CPU, synthetic data, about two minutes)

```bash
# Stage 1: planner pretraining (L_plan, EMA target encoder, stable-rank guard)
python train.py --stage 1 --data synthetic:512 --output-dir runs/s1 --max-steps 300 \
    planner=tiny realizer=tiny data=tiny data.tokenizer=bpe data.bpe_vocab_size=700 \
    training.batch_size=16 training.collapse_grace_steps=100000

# Stage 2: realizer + plan readout on the frozen planner's predicted plan
python train.py --stage 2 --planner-checkpoint runs/s1/stage1_final.pt --data synthetic:512 \
    --eval-data synthetic:64 --output-dir runs/s2 --max-steps 1000 realizer=tiny training.batch_size=16

# Algorithm 1 with K = 4
python generate.py --prompt "Write solve(a, b, c) that returns a * b + c." --k_steps 4 --realizer fs_dfm \
    --checkpoint runs/s2/best_stage2.pt
```

`generate.py` prints the generated text, the sampled mode u, the plan uncertainty σ̄, the audited depth
(K + 3) and a per-pass latency breakdown. The tiny synthetic run demonstrates the pipeline; it is not a model.

## Data

Records pair a context c (specification, docstring, signature) with a target program w₁.
`text_vpa.data.dataset` derives everything else from the target with tree-sitter (Python and C++/HLS-C):
- the structural view y: `[SLOT_<KIND>]` + slot text per slot, at most M slots;
- the design mode u;
- the plan-readout targets.

| Data spec | Source (local files, no network needed) |
|---|---|
| `synthetic:N` | built-in generator of specification → program pairs over 8 templates (tests, smoke runs) |
| `humaneval:PATH` | `HumanEval.jsonl(.gz)`: context = prompt, target = prompt + canonical solution |
| `mbpp:PATH` | MBPP JSONL: context = text + first test, target = code |
| `jsonl:PATH` | one `{"context": ..., "target": ..., "language": "python" \| "cpp"}` object per line; PATH may be a quoted glob (`'jsonl:data/corpus/train-*.jsonl.gz'`) |

Several specs can be combined: `--data humaneval:data/HumanEval.jsonl.gz jsonl:data/corpus.jsonl`.
- **Decontamination:** every training corpus must be deduplicated against the evaluation sets (Sec. 5.1);
  the corpus builder below does it.
- **Tokenizers:** `data.tokenizer=bpe` trains a byte-level BPE on the training corpus offline.
  `data.tokenizer=hf data.tokenizer_name=Qwen/Qwen2.5-Coder-1.5B` uses a pretrained one.
  `data.tokenizer=byte` uses raw bytes.

### Building a training corpus

`python -m text_vpa.data.corpus` turns permissively licensed Python into specification → function pairs:

```bash
python -m text_vpa.data.corpus download --dataset codeparrot/github-code-clean --config Python-all \
    --out data/raw --max-files 10                     # the Hub's partial Parquet conversion; resumable
python -m text_vpa.data.corpus build --source 'parquet:data/raw/*.parquet' \
    --humaneval data/eval/HumanEval.jsonl.gz --mbpp data/eval/mbpp.jsonl --out data/corpus

# all of github-code-clean (880 shards, 314 GB): kept to permissive Python while downloading, ~20 GB
python -m text_vpa.data.corpus download-shards --dataset codeparrot/github-code-clean --out data2/shards
python -m text_vpa.data.corpus build --source 'parquet:data2/shards/*.parquet' --include-methods \
    --allow-undocumented --prefer-documented --max-functions 2000000 \
    --humaneval data/eval/HumanEval.jsonl.gz --mbpp data/eval/mbpp.jsonl --out data2/corpus
```

`download-shards` fetches 8 shards at a time (`--workers`), checks each against its size and re-fetches a
damaged one. A rerun skips finished shards. `--shards 0-99` takes a subset, and `HF_TOKEN` is sent to the Hub
but not to the CDN its downloads redirect to.

- **Sources:** `parquet:GLOB`, `jsonl:GLOB` (a `content`/`code` field and an optional license field) or
  `dir:PATH` (`*.py`, with `--assume-license`).
- **License:** only files whose license is permissive (MIT, Apache-2.0, BSD, ISC, ...) are used.
- **Pairs:** every top-level function with a docstring, a real body and no reference to anything except its
  parameters, locals, builtins and module-level imports. Context = the imports it uses + signature +
  docstring (HumanEval's prompt format); target = the imports + the whole function.
  - `--include-methods` adds methods of top-level classes other than test classes (`Class.method`, moved to
    column 0).
  - `--allow-undocumented` adds functions without a docstring that have at least 2 statements. Their context
    is the imports + signature (+ comment lines before the first statement).
  - `--prefer-documented` with `--max-functions` holds undocumented functions back until the documented
    ones are counted.
  - On site-packages, these options multiply the yield by 3.0 (methods), 1.8 (undocumented) and 6.9
    (both).
- **Deduplication:** exact, on the function's AST without its docstring.
- **Decontamination:** a function sharing any 13-word n-gram (identifiers and numbers; formatting ignored) with
  a HumanEval or MBPP problem is dropped.
- **Output:** `train-*.jsonl.gz` shards, `val.jsonl.gz` (0.5%, by hash) and `stats.json` with the count each
  filter removed. On the Python standard library it keeps about 8,000 functions from 18,000 files.

## Training (Sec. 5.1)

| Stage | Trains | Command |
|---|---|---|
| 1 | f_θ, π_φ^h, g_φ with L_plan; f_θ̄ by EMA only | `train.py --stage 1 --data … --output-dir runs/s1` |
| 2 | F_ψ (+ bridge) and the plan readout; **planner frozen**, conditioned on the predicted plan only | `train.py --stage 2 --planner-checkpoint runs/s1/stage1_final.pt …` |
| 3 (optional) | everything; planner at 0.1 × lr, VICReg on | `train.py --stage 3 --init-checkpoint runs/s2/best_stage2.pt --planner-checkpoint runs/s1/stage1_final.pt …` |

- **Collapse guard (Stage 1).** v(Z), c(Z), the stable rank and the eigenvalue range of the context and plan
  latents are logged every `training.collapse_check_every` steps. If srank(C) < d/2 after
  `training.collapse_grace_steps`, training stops and the checkpoint is saved as rejected; Stage 2 refuses it.
- **Anti-leakage (Stage 2).**
  - No target encoder exists in Stage 2.
  - The realizer accepts only a `PredictedPlan`.
  - The planner does not require gradients, is not in the optimizer, stays in eval mode, and its weights
    are re-checked by SHA-256.
- **Step budgets and plan dropout.** K is drawn from `realizer.step_budgets` (h = 1/K conditions the network)
  and t from the grid {k/K}. 10% plan dropout enables predictor-free guidance (`realizer.guidance_scale`).
- **Distributed training.** Data parallel with `torchrun --nproc_per_node=8 train.py …`: NCCL on CUDA/ROCm,
  Gloo otherwise.
- **Mixed precision and logging.** bf16 with `training.precision=bf16`. Weights & Biases with
  `--wandb-project NAME`; metrics are also written to `<output-dir>/metrics.jsonl`.
- **Checkpoints and resumption.** Checkpoints are self-contained (config, tokenizer, weights) and atomic.
  `--resume` continues the exact data stream (with `data.num_workers=0`); Stage 1 reuses the checkpoint's
  tokenizer.
- **One memory-limited GPU.** `training.<stage>.grad_accum_steps=k` makes one optimizer step from k
  micro-batches of `training.batch_size`; `training.activation_checkpointing=true` recomputes realizer blocks
  in the backward pass. Both leave the gradients unchanged (tested).
- **Large corpora.** `--build-workers N` tokenizes and parses in N processes. `--dataset-cache DIR` stores the
  built examples (int32 arrays, a few KB each), so Stage 2 and resumed runs load them instead of rebuilding.
- **Memory check.** `python -m text_vpa.utils.memory_probe planner=base realizer=l4 data=code training=l4` runs
  optimizer steps on worst-case batches and reports the peak GPU memory, the time per step and the projected
  wall time.

Configuration is a typed Pydantic schema composed by Hydra (`text_vpa/configs/conf`). Pass overrides as
trailing arguments, e.g. `planner=tiny realizer=logit_kl realizer.num_steps=4 training.batch_size=32`.
Unknown keys, wrong types and violated constraints fail at load time. Examples of the constraints:
1 ≤ K ≤ 8 with K among the trained budgets, 0 < ε < γ², c_max ∈ (0, 1], n_ens ≥ 2, m₀ ≤ m_end.
Defaults follow the paper:
- d = 256, M = 256, n_ens = 5;
- VICReg weights 25 : 25 : 1, γ = 1, ε = 10⁻⁴, m₀ = 0.996 → 1;
- d_r = 2048, L_r = 24, K ∈ {2, 4, 8};
- c_max = 1, T = 1, B = 64.

## Training on a GCP L4 VM

The `l4` presets fit one NVIDIA L4 (24 GB) with 64 GB of host memory (GCP `g2-standard-16`):
- the planner of the paper (42M parameters);
- a 325M-parameter realizer (d_r = 1024, 16 layers, N ≤ 512);
- a 32K BPE vocabulary;
- bf16, activation checkpointing, and micro-batches of 16 × 4 accumulation steps (B = 64).

### 1. Create the VM

Use the Cloud Console (**Compute Engine → VM instances → Create instance**), or `scripts/gcp/create_vm.sh`.

| Setting | Value |
|---|---|
| Region / zone | a region where you have L4 quota (e.g. `asia-northeast3`, Seoul); a zone that offers L4 |
| Machine configuration | **GPUs** → GPU type **NVIDIA L4**, number **1** → machine type **`g2-standard-16`** (16 vCPU, 64 GB) |
| VM provisioning model | **Standard** (Spot is cheaper but can be stopped at any time) |
| On host maintenance / automatic restart | **Terminate VM instance** (required with a GPU) / **On** |
| Boot disk image | **Deep Learning on Linux**, the Ubuntu 24.04 version with CUDA 12.9 and NVIDIA driver 580 (image family `common-cu129-ubuntu-2404-nvidia-580`). Ubuntu 22.04 has Python 3.10, too old for the pinned requirements. |
| Boot disk type / size | **Balanced persistent disk**, **200 GB** (image, Python environment ~10 GB, filtered shards ~20 GB, corpus and dataset cache ~15 GB, checkpoints ~15 GB) |
| Identity and API access | Compute Engine default service account, **Allow default access**. Choose Storage **Read Write** only if you set `GCS` to back up `runs/` to a bucket. |
| Networking | an external IPv4 address (ephemeral is fine): the run downloads from Hugging Face, PyPI and GitHub |
| Shielded VM | leave **Secure Boot** off (the default) |

New projects usually need an L4 GPU quota increase in the region first. If a zone has no free L4 capacity, try
another zone of the region.

### 2. Run everything unattended (one command)

In the VM's **SSH** window (or `gcloud compute ssh VM --zone ZONE --command '...'` from your machine):

```bash
[ -d ~/text-VPA ] || git clone https://github.com/ylee-sling/text-VPA.git ~/text-VPA; bash ~/text-VPA/scripts/gcp/run_all.sh
```

`scripts/gcp/run_all.sh` returns at once. The work runs in the detached tmux session `textvpa`, so closing the SSH
window does not stop it, and is logged to `~/text-VPA/runs/logs/run-<timestamp>.log`. Watch it with
`tail -f ~/text-VPA/runs/logs/latest.log` or `tmux attach -t textvpa` (detach: Ctrl-b d).

The session runs these steps in order:
1. checks the GPU and driver with `nvidia-smi`;
2. installs the system packages (`apt-get -y`) and the pinned Python environment (`.venv`);
3. checks that PyTorch runs on the GPU, downloads HumanEval and MBPP, and checks the Pass@k harness on their reference solutions;
4. runs the test suite (failures are reported, not fatal);
5. runs the memory check, falling back to smaller micro-batches with more accumulation if needed;
6. runs `scripts/gcp/train_l4.sh`: corpus, Stage 1 (50K steps), Stage 2 (60K steps), Pass@k on HumanEval and MBPP.

When the run ends, the VM shuts itself down (`shutdown -h +1`), after a failure too. Use `SHUTDOWN_ON_FAILURE=0`
to keep it up. Logs, checkpoints and results stay on the boot disk. If the VM restarts unexpectedly (host
maintenance), an `@reboot` entry restarts the run, and it resumes from the last checkpoint. `run_all.sh
--status` shows the state; `run_all.sh --cancel` stops the run without shutting down. Times in the log are the
VM's clock (UTC). Settings are environment variables, such as `STAGE2_STEPS=30000 bash .../run_all.sh`.

Each run directory records its outcome (`runs/l4/run_all.done`, `.failed`, ...), so a finished run is not
started again. A second run takes a new run directory, and a new data directory if its corpus differs:

```bash
RUN=runs/run2 DATA=data2 bash ~/text-VPA/scripts/gcp/run_all.sh
```

Step by step instead: `bash scripts/gcp/setup_vm.sh`, then
`tmux new -s train 'bash scripts/gcp/train_l4.sh 2>&1 | tee -a train_l4.log'`.

### Reusing a VM from another project

A VM that already hosts other software, environments or running work can break the run in ways a fresh one
cannot. Check it first: `scripts/gcp/preflight.sh` is read-only (it installs, starts and changes nothing) and
exits with status 1 if something would stop the run.

```bash
cd ~ && if [ -d text-VPA/.git ]; then git -C text-VPA pull --ff-only; else git clone https://github.com/ylee-sling/text-VPA.git; fi && bash text-VPA/scripts/gcp/preflight.sh
```

| Area | Checked | FAIL when |
|---|---|---|
| System | OS, vCPUs, memory; passwordless sudo; free disk | no sudo; less than `MIN_FREE_GB` (60) GB free |
| GPU | `nvidia-smi`, GPU model and memory, driver → PyTorch build | no working driver; a driver older than 525 |
| Other work | processes on the GPU; tmux and screen sessions, containers and CPU load that the final shutdown would stop | the GPU is in use (`ALLOW_BUSY_GPU=1` makes it a warning) |
| Python | a Python ≥ 3.12 for `.venv`; an existing `.venv` | `PYTHON` names a Python older than 3.12 |
| Settings | `PYTHONPATH`, `LD_LIBRARY_PATH`, conda, `PIP_*` and pip.conf files, a running tmux server's environment, `CUDA_VISIBLE_DEVICES` | — (reported) |
| text-VPA | the clone (local changes, behind GitHub); an earlier run, corpus and checkpoints | — (reported) |
| Network | GitHub, PyPI, the PyTorch wheel index (drivers < 580), HumanEval and MBPP, the corpus files on Hugging Face | a host the run needs is unreachable |

Then start the run with the check in front, so that it starts only if the check passes:

```bash
cd ~ && if [ -d text-VPA/.git ]; then git -C text-VPA pull --ff-only; else git clone https://github.com/ylee-sling/text-VPA.git; fi && bash text-VPA/scripts/gcp/preflight.sh && bash text-VPA/scripts/gcp/run_all.sh
```

What `run_all.sh` does differently on such a VM:
- **Python.** It always builds its own `.venv`, and recreates one that is broken or older than Python 3.12.
  Without a system Python ≥ 3.12 (e.g. Ubuntu 22.04), it installs a standalone Python 3.12 with
  [uv](https://docs.astral.sh/uv/) into `.python/`; nothing outside the repository changes. `PYTHON=/path/to/python3.12`
  picks an interpreter, `PYTHON=uv` forces the standalone one.
- **Inherited settings.** It clears `PYTHONPATH`, `PYTHONHOME`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, conda and venv
  variables, and the `PIP_*` variables except pip's network ones (`PIP_CERT`, `PIP_CLIENT_CERT`, `PIP_PROXY`,
  `PIP_TRUSTED_HOST`), and ignores pip.conf files. A tmux server started earlier passes its own environment to new
  sessions, so this matters even when your shell is clean. The run gets the launching shell's
  `CUDA_VISIBLE_DEVICES`. `KEEP_ENV=1` turns the cleanup off.
- **PyTorch.** PyTorch 2.14 is built for CUDA 13.0 (the PyPI wheel; NVIDIA driver 580 or newer) and CUDA 12.6
  (driver 525 or newer). With a driver from 525 to 579 the run installs the CUDA 12.6 build;
  `TORCH_INDEX_URL` overrides the choice.
- **Shutdown.** The VM still shuts down when the run ends, which stops anything else running on it. To keep other
  work alive, start with `SHUTDOWN_ON_SUCCESS=0 SHUTDOWN_ON_FAILURE=0 bash text-VPA/scripts/gcp/run_all.sh`.

`train_l4.sh` runs these steps in order:
1. download and build the corpus (below);
2. check memory on worst-case batches;
3. Stage 1 (50K steps; the stable rank is logged, see below);
4. Stage 2 (60K steps, validation loss and the inventory calibration every 2,000 steps);
5. Pass@k on HumanEval and MBPP, with the constraint energies and without them (`RUN/eval/<benchmark>` and
   `RUN/eval/<benchmark>-no-energy`).

Re-running the same command after a stop skips finished steps and resumes the interrupted stage from its last
checkpoint (every 1,000 steps). Set `GCS=gs://BUCKET/PATH` to copy `runs/` to Cloud Storage every 30 minutes.
Every setting is a variable at the top of the script.

**Corpus.** `CORPUS=shards` (the default) downloads every shard of
[github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean)'s main branch (880 Parquet
files, 314 GB, all languages) with `python -m text_vpa.data.corpus download-shards`. Each shard is filtered to
permissively licensed Python while it downloads and then deleted, so the filtered files take ~20 GB. The
build keeps self-contained functions, methods of top-level classes (moved to column 0) and undocumented
functions (their context is the signature). Documented functions come first and fill `MAX_FUNCTIONS`
(2,000,000); undocumented ones fill what is left. `SHARDS=0-99` takes a tenth of the shards. `CORPUS=export`
is Run 1's smaller corpus: 10 files of the dataset's partial Parquet export, documented top-level functions
only (61K functions). A corpus directory records its recipe (`DATA/corpus/recipe`), and a run directory its
corpus (`RUN/corpus`). A different recipe for an existing `DATA`, or another corpus for a run with
checkpoints, is refused.

**Stable-rank guard.** The paper rejects a Stage-1 planner whose stable rank is below d/2. In Run 1 it rose
from 12 to 125 (d = 256) over 50K steps, so the guard would have rejected a planner that was still improving.
`train_l4.sh` sets `training.collapse_grace_steps` to `STAGE1_STEPS + 1`: the stable rank is logged every 1,000
steps and printed after Stage 1, but it does not stop the run. `COLLAPSE_GRACE=N` lets the guard reject from
step N.

## Evaluation (Pass@k)

```bash
python evaluate.py --benchmark humaneval --data data/eval/HumanEval.jsonl.gz \
    --checkpoint runs/l4/stage2/best_stage2.pt --num-samples 10 --k 1 10 --out runs/eval/humaneval
python evaluate.py --benchmark mbpp --data data/eval/mbpp.jsonl --reference --out runs/eval/mbpp-reference
```

`--no-energy` turns the constraint energies off. The plan-vocabulary energy licenses the readout's predicted
token inventory, keywords, builtins and the context's words. It is soft: an unlicensed identifier token keeps
e^{-2/T} of its probability (`energy.identifier_weight = 2.0`). A hard energy blocked valid code in Run 1,
because the readout licensed a third of the tokens a program needs. The readout's threshold is calibrated at
every Stage-2 evaluation to license 95% of the validation programs' identifier tokens, and the checkpoint
stores it. These flags override a checkpoint's settings: `--[no-]license-context`, `--identifier-weight W`
(`inf` = hard), and `--inventory-threshold P`. Checkpoints trained before these defaults keep their own
settings (a hard energy without the context's words, threshold 0.5).

After a run, `python -m text_vpa.evaluation.diagnose` reports which checkpoint was scored (every Stage-2
validation loss and the step in `best_stage2.pt`), prints HumanEval completions with their errors, and
generates from training and validation contexts with and without the energies and with the reference
canvas length. That separates a model that memorized its corpus from a sampling problem.
`scripts/gcp/overfit_check.sh` trains Stage 2 on a few hundred training functions only (about an hour on an
L4) and runs the diagnostic on them: a realizer that cannot memorize them cannot be fixed with more data.

- **Sampling:** every problem gets n samples from Algorithm 1, with the context the training loaders build.
- **Programs run:**
  - HumanEval: prompt + sample + tests + `check(entry_point)`;
  - MBPP: setup + sample + asserts (test split 11–510 by default).
- **Execution:** each program runs in its own resource-limited process (CPU, memory, file size, timeout). This
  isolates crashes and hangs but is not a security sandbox, so run it on a disposable VM.
- **Score:** pass@k is the unbiased estimator of Chen et al. (2021).
- **Harness check:** `--reference` runs the canonical solutions. HumanEval passes 164/164 and MBPP 499/500;
  MBPP task 367's tests use an undefined class.

## Inference and profiling

```bash
python generate.py --prompt "def fibonacci(n: int) -> int:" --k_steps 4 --realizer fs_dfm \
    --checkpoint ./checkpoints/best_stage2.pt
python -m text_vpa.utils.profiler --roofline                                  # Table 3
python -m text_vpa.utils.profiler --checkpoint runs/s2/best_stage2.pt --lengths 64 256 --steps 1 2 4 8
```

The profiler times batch-1 generation for each K and compares it with a simulated autoregressive decoder:
same width and depth, with a KV cache, on the same device. Each run asserts the K + 3 depth.

## Equation-to-code map

| Paper | Code |
|---|---|
| Eq. 3, Eq. 8 (plan, σ̄) | `models/predictor.py`: `JEPAPredictor`, `ensemble_mean`, `ensemble_uncertainty` |
| Eq. 4 (EMA) | `models/ema.py`: `EMATargetEncoder`, `ema_momentum` |
| Eqs. 5–7 (L_plan, v(Z), c(Z)) | `engine/losses.py`: `PlanLoss`, `VICRegLoss`, `variance_hinge`, `covariance_penalty` |
| Thm. 1, Props. 2–3 | `engine/losses.py`: `theorem1_lambda_min_bound`, `stable_rank`, `calculate_stable_rank`; `tests/test_theory_verification.py` |
| Prop. 1 (modes) | `models/selector.py`, `data/ast_parser.py`: `ModeExtractor` |
| Eq. 10 (bridge) | `models/bridge.py`: `CrossAttentionBridge.precompute_kv`, `cross_attention` |
| Eqs. 11–12 (mixture path, L_real) | `models/realizers/paths.py`: `mixture_path_sample`; `fs_dfm.py`: `loss` |
| Eqs. 15–17, Prop. 4 (jump kernel) | `models/realizers/kernels.py`; `inference/energy_terms.py`: `check_parallel_update` |
| Eqs. 18–21, Lemma 2 (Logit-KL) | `models/realizers/logit_kl.py`, `paths.py` |
| Algorithm 1 | `inference/sampler.py`: `PRJEPASampler.sample`, `DepthAudit` |
| Sec. 3.3 energies | `inference/energy_terms.py` |
| Sec. 4.6, Table 2 | `utils/verification.py` |
| Sec. 5.4, Table 3, Eqs. 33–34 | `utils/profiler.py` |
| Sec. 5.1 (staged training) | `engine/trainer_stage1.py`, `engine/trainer_stage2.py`, `train.py` |

## Decisions where the paper is silent

These are recorded in [`CLAUDE.md`](CLAUDE.md). The most consequential:

- **Invariance normalization.** The prediction term is normalized per element (VICReg's MSE), so the
  published 25 : 25 : 1 weights keep VICReg's balance at every d. Eq. 5 sums the error over the d features;
  with weight 25 that over-weights it d-fold, and the plan's variance collapsed in Stage-1 runs.
  `vicreg.invariance_normalization=frobenius vicreg.lambda_invar=1` is Eq. 5 verbatim.
- **Plan readout.** The paper's "slot decoder" is `PlanReadout`, trained in Stage 2 on the predicted plan. It
  decodes slot kinds, signatures, a token inventory and the output length from Z_target, which is what the
  energies need at inference.
- **Energy terms.** Only identifier licensing is plan-mediated, so only it carries Prop. 4(d)'s guarantee
  under parallel updates. It licenses the readout's token inventory, keywords, builtins and the context's
  words. It is soft by default (`energy.identifier_weight = 2.0`; `.inf` = hard), and the readout's threshold is
  calibrated to 95% recall at every Stage-2 evaluation. Grammar (hard), argument-count and type-annotation
  (soft) terms read the draft prefix. A network-free left-to-right repair enforces the hard ones at the end.
- **Repair of stalls.** A draft-dependent term evaluated on a noisy prefix can reject the token the posterior
  wants (for example [PAD] after a noise token that opened a bracket). The rejected mass becomes probability of
  staying (Prop. 4(b)), so after the last step such a position can still hold its source token. The repair
  treats these positions like a leftover [MASK]: it refills them with argmax p·e^(−E/T), where p is the last
  posterior and E is computed on the repaired prefix. `SampleResult.unrealized` counts them.
- **Canvas length.** Training canvases are the target plus [PAD], rounded up to `data.bucket_size`. The
  sampler takes N from the readout's length head by default, so the realizer may use N as a cue; a length
  that no training example had is out of distribution.
- **Stable-rank guard grace period.** The guard does not abort during a grace period, because random
  initializations are low-rank. The paper checks after each epoch.
- **Gradient accumulation.** Each micro-batch loss is scaled by 1/k. Two effects differ from one batch of k
  times the size:
  - VICReg's batch statistics are computed per micro-batch;
  - the spectral-norm power iteration on W_K, W_V advances once per micro-batch.
- **Decontamination.** A training function is dropped if it shares any 13-word n-gram with an evaluation
  problem, where words are identifiers and numbers (GPT-3's 13-gram rule). Punctuation is not counted, because
  a shared signature alone such as `(numbers: List[int]) -> List[int]` is not contamination.

## Repository layout

```
text-VPA/
├── text_vpa/
│   ├── configs/            Pydantic schema + Hydra YAML groups (planner, bridge, realizer, vicreg, training, data, readout, energy, sampler)
│   ├── models/             encoders, ema, selector, predictor, bridge, readout, plan, text_vpa_model, layers
│   │   └── realizers/      backbone (AdaLN), fs_dfm, logit_kl, kernels, paths
│   ├── data/               tokenization, ast_parser (tree-sitter), dataset, corpus (builder), synthetic
│   ├── engine/             losses, trainer_stage1, trainer_stage2, runtime
│   ├── inference/          sampler (Algorithm 1), energy_terms, loading
│   ├── evaluation/         pass@k, sandboxed execution, HumanEval / MBPP tasks
│   └── utils/              verification (App. B), profiler (Sec. 5.4), memory_probe, device
├── tests/                  equation-level tests, theory verification, end-to-end
├── train.py  generate.py  evaluate.py   CLIs
├── scripts/                init_repo.sh, release.sh; gcp/ (create_vm, setup_vm, train_l4, run_all, preflight)
└── docs/pr_jepa_preprint.pdf
```

## Citation

```bibtex
@misc{lee2026prjepa,
  author    = {Lee, Yeonseok},
  title     = {{PR-JEPA}: Decoupling Latent Planning from Few-Step Discrete Realization for
               Non-Autoregressive Generation of Code, Legal Documents, and Control},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23498811},
  url       = {https://doi.org/10.5281/zenodo.23498811},
  note      = {Preprint}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff). The VPA framework this builds on:
[doi:10.5281/zenodo.22960919](https://doi.org/10.5281/zenodo.22960919).

## License

GPL-3.0-only, as the VPA reference implementation. Copyright © 2026 SLING AI Inc.
