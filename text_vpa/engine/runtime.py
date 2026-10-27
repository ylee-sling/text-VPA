# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
runtime.py -- infrastructure shared by the stage trainers: distributed setup, mixed precision,
optimisation, checkpoints and metric logging.

Distributed data parallel: launch with ``torchrun --nproc_per_node=G train.py ...``; ``DistributedContext``
reads RANK / WORLD_SIZE / LOCAL_RANK, uses NCCL on CUDA (and ROCm) and Gloo otherwise.
Mixed precision: ``autocast`` runs the networks in bfloat16 when ``training.precision = bf16``; losses and
VICReg statistics are computed in float32 by the trainers.
Checkpoints are written atomically (temporary file + rename) and load with ``weights_only=True``.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from text_vpa.utils.device import resolve_device

__all__ = [
    "CHECKPOINT_FORMAT",
    "DistributedContext",
    "MetricLogger",
    "autocast",
    "build_optimizer",
    "load_checkpoint",
    "rng_state",
    "save_checkpoint",
    "set_rng_state",
    "set_seed",
    "warmup_cosine",
]

CHECKPOINT_FORMAT = "text-vpa/1"
_NO_DECAY = ("bias", "norm", "embedding", "position", "slot_queries", "null_token", "table", "text_positions")


# =============================================================================================
# Distributed
# =============================================================================================
@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, device: str = "auto") -> DistributedContext:
        """Single process, or torchrun's environment (RANK, WORLD_SIZE, LOCAL_RANK)."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            return cls(device=resolve_device(device))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available() and device in ("auto", "cuda"):
            torch.cuda.set_device(local_rank)
            target, backend = torch.device("cuda", local_rank), "nccl"
        else:
            target, backend = resolve_device("cpu" if device == "auto" else device), "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return cls(rank, world_size, local_rank, target)

    def any_true(self, flag: bool) -> bool:
        """True if ``flag`` holds on any rank (all ranks must call this together)."""
        if not self.enabled:
            return flag
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def mean(self, value: float) -> float:
        if not self.enabled:
            return value
        t = torch.tensor([value], device=self.device, dtype=torch.float64 if self.device.type != "mps" else torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item()) / self.world_size

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def shutdown(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def autocast(device: torch.device, precision: str) -> AbstractContextManager:
    """bfloat16 autocast for ``precision == "bf16"``, otherwise a no-op."""
    if precision != "bf16":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state(), "python": list(random.getstate()[1]),
                             "python_pos": random.getstate()[2]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate((3, tuple(state["python"]), state["python_pos"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# =============================================================================================
# Optimisation
# =============================================================================================
_NO_DECAY_MODULES = (nn.Embedding, nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)


def _param_groups(module: nn.Module, weight_decay: float, lr: float, tag: str) -> list[dict]:
    exempt = {id(p) for m in module.modules() if isinstance(m, _NO_DECAY_MODULES) for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        lowered = name.lower()
        exempt_param = id(p) in exempt or p.ndim < 2 or any(k in lowered for k in _NO_DECAY)
        (no_decay if exempt_param else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "lr": lr, "base_lr": lr, "name": f"{tag}/decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr, "base_lr": lr, "name": f"{tag}/no_decay"})
    return groups


def build_optimizer(parts: list[tuple[str, nn.Module, float]], weight_decay: float) -> torch.optim.AdamW:
    """AdamW over (tag, module, lr) parts; no weight decay on biases, normalisation layers, embeddings (by module
    type and by name: positions, slot queries, null token, AdaLN tables) and other 1-D tensors."""
    groups: list[dict] = []
    for tag, module, lr in parts:
        groups += _param_groups(module, weight_decay, lr, tag)
    if not groups:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int,
                  final_fraction: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to each group's lr, then cosine decay to ``final_fraction`` of it at ``max_steps``."""
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# =============================================================================================
# Checkpoints
# =============================================================================================
def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomic ``torch.save`` of a payload of tensors and plain Python containers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": CHECKPOINT_FORMAT, **payload}, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a text-VPA checkpoint (format {payload.get('format')!r})")
    return payload


# =============================================================================================
# Logging
# =============================================================================================
class MetricLogger:
    """JSON-lines metrics (always), console lines, and Weights & Biases when a project is given (rank 0)."""

    def __init__(self, output_dir: str | Path, enabled: bool = True, wandb_project: str | None = None,
                 run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        self.enabled = enabled
        self.path = Path(output_dir) / "metrics.jsonl"
        self._wandb = None
        self._start = time.time()
        if not enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if wandb_project:
            try:
                import wandb  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError("--wandb-project needs the wandb package (pip install wandb)") from exc
            self._wandb = wandb.init(project=wandb_project, name=run_name, config=config)

    def log(self, step: int, metrics: dict[str, float], prefix: str = "") -> None:
        if not self.enabled:
            return
        row: dict[str, Any] = {"step": step, "time_s": round(time.time() - self._start, 3)}
        for key, value in metrics.items():
            if isinstance(value, int | float | torch.Tensor):
                value = float(value)
                if not math.isfinite(value):                # JSON has no NaN / Infinity literals
                    value = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
            row[f"{prefix}{key}"] = value
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                         if k not in ("time_s",))
        print(shown, file=sys.stdout, flush=True)
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
