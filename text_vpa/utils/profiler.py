# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
profiler.py -- latency of PR-JEPA generation: the analytical roofline model (Sec. 5.4, Table 3; Appendix A.2),
measured batch-1 latency of the sampler, and a simulated autoregressive (AR) baseline.

Roofline (predictions, not measurements). With P parameters, L layers, h_kv KV heads of width d_h, width d,
b = 2 bytes, effective bandwidth BW_e = eta_bw BW and compute F_e = eta_F F, and a 512-token prompt:

    KV(n)  = 2 L h_kv d_h b n                                                     KV-cache bytes for n tokens
    T_AR   = 2P 512 / F_e + sum_{i=0}^{N-1} max( (2P + KV(512 + i)) / BW_e, (2P + 4L(512 + i) d) / F_e )   (Eq. 33)
    T_NAR  = K max( 2P / BW_e, (2P T + 4 L T^2 d) / F_e ),  T = 512 + N                                  (Eq. 34)

Planner and bridge passes are small and not modelled. ``table3()`` reproduces Table 3 exactly.

Measurement. ``measure_sampler_latency`` times ``PRJEPASampler.sample`` at batch 1 for K in {1, 2, 4, 8}
(median and 95th percentile after warm-up, device-synchronised) and checks the runtime depth audit
(K + 3 sequential network evaluations) on every run. ``SimulatedARDecoder`` decodes N tokens one by one
with a KV cache through a causal transformer of the realizer's width and depth -- the AR cost of a
backbone of the same size, measured on the same device.

CLI::

    python -m text_vpa.utils.profiler --roofline
    python -m text_vpa.utils.profiler --checkpoint runs/stage2/best_stage2.pt --lengths 64 256 --steps 1 2 4 8
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "FSDFM_1P3B",
    "H100_SXM",
    "QWEN25_CODER_7B",
    "Backbone",
    "Hardware",
    "LatencyRecord",
    "SimulatedARDecoder",
    "ar_compute_bound_batch",
    "ar_latency",
    "kv_bytes_per_token",
    "kv_limited_batch",
    "measure_ar_latency",
    "measure_sampler_latency",
    "nar_latency",
    "saturated_throughput_ratio",
    "table3",
]


# =============================================================================================
# Roofline model (Appendix A.2)
# =============================================================================================
@dataclass(frozen=True)
class Hardware:
    """Vendor peak figures and assumed achieved fractions."""

    name: str
    bandwidth: float                 # bytes/s
    flops: float                     # dense bf16 FLOP/s
    eta_bw: float = 0.7
    eta_flops: float = 0.5
    bytes_per_element: int = 2
    memory_bytes: float = 80e9

    @property
    def effective_bandwidth(self) -> float:
        return self.eta_bw * self.bandwidth

    @property
    def effective_flops(self) -> float:
        return self.eta_flops * self.flops


@dataclass(frozen=True)
class Backbone:
    name: str
    params: float
    layers: int
    kv_heads: int
    head_dim: int
    width: int


H100_SXM = Hardware("H100 SXM", bandwidth=3.35e12, flops=989e12)
QWEN25_CODER_7B = Backbone("7.6B, GQA", 7.6e9, 28, 4, 128, 3584)      # public Qwen2.5-Coder-7B configuration
FSDFM_1P3B = Backbone("1.3B, MHA", 1.3e9, 24, 16, 128, 2048)          # assumed multi-head configuration


def kv_bytes_per_token(m: Backbone, hw: Hardware = H100_SXM) -> float:
    """2 L h_kv d_h b (Eq. 2)."""
    return 2 * m.layers * m.kv_heads * m.head_dim * hw.bytes_per_element


def ar_latency(m: Backbone, n_tokens: int, prompt: int = 512, hw: Hardware = H100_SXM) -> float:
    """Eq. 33, seconds: compute-bound prefill, then per token the larger of memory and compute time."""
    weights = m.params * hw.bytes_per_element
    t = 2 * m.params * prompt / hw.effective_flops
    kv = kv_bytes_per_token(m, hw)
    for i in range(n_tokens):
        ctx = prompt + i
        t += max((weights + kv * ctx) / hw.effective_bandwidth,
                 (2 * m.params + 4 * m.layers * ctx * m.width) / hw.effective_flops)
    return t


def nar_latency(m: Backbone, n_tokens: int, num_steps: int, prompt: int = 512, hw: Hardware = H100_SXM,
                planner_ms: float = 0.0) -> float:
    """Eq. 34, seconds: K full passes over T = prompt + N positions (+ an optional planner time)."""
    t_len = prompt + n_tokens
    flops = 2 * m.params * t_len + 4 * m.layers * t_len * t_len * m.width
    step = max(m.params * hw.bytes_per_element / hw.effective_bandwidth, flops / hw.effective_flops)
    return num_steps * step + planner_ms / 1e3


def saturated_throughput_ratio(n_tokens: int, num_steps: int, prompt: int = 512) -> float:
    """NAR / AR throughput when both are compute-bound: N / (K (prompt + N))."""
    return n_tokens / (num_steps * (prompt + n_tokens))


def ar_compute_bound_batch(m: Backbone, hw: Hardware = H100_SXM) -> float:
    """Batch at which AR decoding becomes compute-bound, (2P/BW_e)/(2P/F_e) as in the paper's script."""
    return (m.params * hw.bytes_per_element / hw.effective_bandwidth) / (2 * m.params / hw.effective_flops)


def kv_limited_batch(m: Backbone, n_tokens: int, prompt: int = 512, hw: Hardware = H100_SXM) -> float:
    """Largest batch whose KV cache fits beside the weights: (80 GB - 2P) / KV(prompt + N)."""
    return (hw.memory_bytes - m.params * hw.bytes_per_element) / (kv_bytes_per_token(m, hw) * (prompt + n_tokens))


def table3(hw: Hardware = H100_SXM, prompt: int = 512) -> list[dict[str, float | str | int]]:
    """Rows of Table 3: latency in ms for AR and NAR at K = 4, 8, the ratio at K = 8 and the AR KV cache (MiB)."""
    rows = []
    for m in (QWEN25_CODER_7B, FSDFM_1P3B):
        for n in (4096, 1024, 256):
            ar = ar_latency(m, n, prompt, hw)
            k4, k8 = nar_latency(m, n, 4, prompt, hw), nar_latency(m, n, 8, prompt, hw)
            rows.append({"backbone": m.name, "N": n, "ar_ms": ar * 1e3, "nar_k4_ms": k4 * 1e3,
                         "nar_k8_ms": k8 * 1e3, "ratio_k8": ar / k8,
                         "kv_mib": kv_bytes_per_token(m, hw) * (prompt + n) / 2 ** 20})
    return rows


# =============================================================================================
# Measured latency
# =============================================================================================
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class LatencyRecord:
    method: str
    num_steps: int | None
    length: int
    median_ms: float
    p95_ms: float
    depth: int | None                   # sequential network evaluations (K + 3 for PR-JEPA)
    stages_ms: dict[str, float]         # median per stage (PR-JEPA)


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


@torch.no_grad()
def measure_sampler_latency(sampler, context: Tensor, lengths: Sequence[int], steps: Sequence[int] = (1, 2, 4, 8),
                            warmup: int = 3, repeats: int = 20, seed: int = 0) -> list[LatencyRecord]:
    """Batch-1 latency of ``PRJEPASampler.sample``; the K + 3 depth audit must pass on every run."""
    if context.shape[0] != 1:
        raise ValueError("single-request latency is measured at batch 1")
    device = context.device
    records = []
    for n in lengths:
        for k in steps:
            times: list[float] = []
            stage_runs: list[dict[str, float]] = []
            depth = None
            for run in range(warmup + repeats):
                gen = torch.Generator(device=device).manual_seed(seed + run)
                _sync(device)
                start = time.perf_counter()
                result = sampler.sample(context, length=n, num_steps=k, generator=gen, profile=True)
                _sync(device)
                elapsed = (time.perf_counter() - start) * 1e3
                if result.depth != k + 3:
                    raise AssertionError(f"depth {result.depth} != K + 3 = {k + 3}")
                depth = result.depth
                if run >= warmup:
                    times.append(elapsed)
                    stage_runs.append(result.timings_ms)
            stages = {name: statistics.median(r.get(name, 0.0) for r in stage_runs) for name in stage_runs[0]}
            records.append(LatencyRecord("pr_jepa", k, n, statistics.median(times), _p95(times), depth, stages))
    return records


class SimulatedARDecoder(nn.Module):
    """Causal transformer with a KV cache: the AR cost of a backbone of a given size (random weights).

    Args:
        vocab_size, dim, num_layers, num_heads, mlp_ratio: backbone size (use the realizer's).
        max_len: maximum prompt + generated length.
    """

    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int, mlp_ratio: float = 4.0,
                 max_len: int = 4096) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads, self.head_dim, self.max_len = num_heads, dim // num_heads, max_len
        hidden = int(dim * mlp_ratio)
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList(
            nn.ModuleDict({"norm1": nn.LayerNorm(dim), "qkv": nn.Linear(dim, 3 * dim), "out": nn.Linear(dim, dim),
                           "norm2": nn.LayerNorm(dim), "fc1": nn.Linear(dim, hidden), "fc2": nn.Linear(hidden, dim)})
            for _ in range(num_layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def _step(self, x: Tensor, cache: list[tuple[Tensor, Tensor] | None], start: int) -> Tensor:
        b, t, d = x.shape
        x = x + self.pos(torch.arange(start, start + t, device=x.device))
        for i, layer in enumerate(self.layers):
            q, k, v = layer["qkv"](layer["norm1"](x)).view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], dim=2)
                v = torch.cat([cache[i][1], v], dim=2)
            cache[i] = (k, v)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1)
            x = x + layer["out"](attn.transpose(1, 2).reshape(b, t, d))
            x = x + layer["fc2"](F.gelu(layer["fc1"](layer["norm2"](x))))
        return self.head(self.norm(x[:, -1]))

    @torch.no_grad()
    def generate(self, prompt: Tensor, num_tokens: int) -> Tensor:
        """Greedy decoding of ``num_tokens`` tokens: one prefill pass, then one pass per token."""
        if prompt.shape[1] + num_tokens > self.max_len:
            raise ValueError("prompt + num_tokens exceeds max_len")
        cache: list[tuple[Tensor, Tensor] | None] = [None] * len(self.layers)
        logits = self._step(self.embed(prompt), cache, 0)
        out = []
        pos = prompt.shape[1]
        for _ in range(num_tokens):
            token = logits.argmax(dim=-1, keepdim=True)
            out.append(token)
            logits = self._step(self.embed(token), cache, pos)
            pos += 1
        return torch.cat(out, dim=1)


@torch.no_grad()
def measure_ar_latency(decoder: SimulatedARDecoder, prompt: Tensor, lengths: Sequence[int], warmup: int = 2,
                       repeats: int = 10) -> list[LatencyRecord]:
    """Batch-1 latency of N-token greedy AR decoding (N + 1 sequential passes)."""
    device = prompt.device
    records = []
    for n in lengths:
        times = []
        for run in range(warmup + repeats):
            _sync(device)
            start = time.perf_counter()
            decoder.generate(prompt, n)
            _sync(device)
            if run >= warmup:
                times.append((time.perf_counter() - start) * 1e3)
        records.append(LatencyRecord("ar_simulated", None, n, statistics.median(times), _p95(times), n + 1, {}))
    return records


# =============================================================================================
# CLI
# =============================================================================================
def _print_table3() -> None:
    print("Table 3 (roofline prediction, H100 SXM, eta_bw = 0.7, eta_F = 0.5, 512-token prompt)")
    print(f"{'backbone':<10} {'N':>5} {'AR (ms)':>9} {'K=4 (ms)':>9} {'K=8 (ms)':>9} {'AR/NAR K=8':>11} {'KV (MiB)':>9}")
    for r in table3():
        print(f"{r['backbone']:<10} {r['N']:>5} {r['ar_ms']:>9,.0f} {r['nar_k4_ms']:>9,.0f} {r['nar_k8_ms']:>9,.0f} "
              f"{r['ratio_k8']:>10.1f}x {r['kv_mib']:>9,.0f}")
    for m in (QWEN25_CODER_7B, FSDFM_1P3B):
        print(f"{m.name}: AR compute-bound batch ~ {ar_compute_bound_batch(m):.0f}; KV-limited batch "
              f"{kv_limited_batch(m, 1024):.0f} (N = 1,024), {kv_limited_batch(m, 8192):.0f} (N = 8,192)")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PR-JEPA latency: roofline model and measurements")
    p.add_argument("--roofline", action="store_true", help="print Table 3")
    p.add_argument("--checkpoint", help="Stage-2/3 checkpoint to measure")
    p.add_argument("--prompt", default="def fibonacci(n: int) -> int:")
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 256])
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-ar", action="store_true", help="skip the simulated AR baseline")
    args = p.parse_args(argv)
    if args.roofline or not args.checkpoint:
        _print_table3()
    if not args.checkpoint:
        return
    from text_vpa.inference.loading import build_sampler, load_model
    from text_vpa.utils.device import resolve_device

    device = resolve_device(args.device)
    loaded = load_model(args.checkpoint, device)
    sampler = build_sampler(loaded)
    context = torch.tensor([loaded.tokenizer.encode(args.prompt)], device=device)
    print(f"\nMeasured on {device} (batch 1, median / p95 over {args.repeats} runs)")
    for r in measure_sampler_latency(sampler, context, args.lengths, args.steps, repeats=args.repeats):
        print(f"PR-JEPA K={r.num_steps} N={r.length}: {r.median_ms:8.2f} / {r.p95_ms:8.2f} ms  depth {r.depth}")
    if not args.no_ar:
        rc = loaded.cfg.realizer
        decoder = SimulatedARDecoder(rc.vocab_size, rc.hidden_dim, rc.num_layers, rc.num_heads, rc.mlp_ratio,
                                     max_len=context.shape[1] + max(args.lengths) + 1).to(device).eval()
        for r in measure_ar_latency(decoder, context, args.lengths, repeats=max(3, args.repeats // 2)):
            print(f"AR (simulated, same width/depth) N={r.length}: {r.median_ms:8.2f} / {r.p95_ms:8.2f} ms  "
                  f"depth {r.depth}")


if __name__ == "__main__":
    main()
