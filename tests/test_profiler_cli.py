# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Independent tests of the latency model and the command-line tools:

    profiler.py   the roofline model of Sec. 5.4 / Appendix A.2 (Eqs. 2, 33, 34; Table 3 and the batch figures
                  quoted in Sec. 5.4), the simulated AR baseline with a KV cache, and the measurement helpers
    generate.py   few-step generation from a Stage-2 checkpoint (Algorithm 1; depth K + 3)

The roofline reference below is written from Appendix A.2 (vectorised over the decode steps), and Table 3 is
compared with the numbers as printed in the paper. The AR reference recomputes the whole sequence at every
step with an explicit causal mask, in float64, from the decoder's weights.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import generate as generate_cli
from text_vpa.configs import load_config
from text_vpa.data.dataset import MixtureCorruption, PRJEPACollator, PRJEPADataset
from text_vpa.data.synthetic import synthetic_corpus
from text_vpa.data.tokenization import train_bpe_tokenizer
from text_vpa.engine.runtime import DistributedContext, MetricLogger
from text_vpa.engine.trainer_stage2 import Stage2Trainer
from text_vpa.inference.loading import build_sampler, load_model
from text_vpa.models.text_vpa_model import TextVPAModel
from text_vpa.utils import profiler
from text_vpa.utils.profiler import (
    FSDFM_1P3B,
    H100_SXM,
    QWEN25_CODER_7B,
    Backbone,
    Hardware,
    SimulatedARDecoder,
    ar_compute_bound_batch,
    ar_latency,
    kv_bytes_per_token,
    kv_limited_batch,
    measure_ar_latency,
    measure_sampler_latency,
    nar_latency,
    saturated_throughput_ratio,
    table3,
)

# =============================================================================================
# Roofline reference (Appendix A.2)
# =============================================================================================
#: Table 3 as printed: backbone, N, AR (ms), NAR K=4 (ms), NAR K=8 (ms), AR / NAR at K=8, AR KV cache (MiB).
PAPER_TABLE3 = [
    ("7.6B, GQA", 4096, 26822, 636, 1271, "21.1", 252),
    ("7.6B, GQA", 1024, 6679, 197, 393, "17.0", 84),
    ("7.6B, GQA", 256, 1679, 96, 193, "8.7", 42),
    ("1.3B, MHA", 4096, 5423, 131, 261, "20.7", 864),
    ("1.3B, MHA", 1024, 1226, 36, 72, "17.0", 288),
    ("1.3B, MHA", 256, 300, 17, 34, "8.8", 144),
]
#: H100 SXM peak figures and achieved fractions; 2-byte weights and cache entries; 512-token prompt.
REF_HW = {"bw": 3.35e12 * 0.7, "flops": 989e12 * 0.5, "bytes": 2, "memory": 80e9}
REF_MODELS = {
    "7.6B, GQA": {"P": 7.6e9, "L": 28, "hkv": 4, "dh": 128, "d": 3584},     # Qwen2.5-Coder-7B
    "1.3B, MHA": {"P": 1.3e9, "L": 24, "hkv": 16, "dh": 128, "d": 2048},    # assumed multi-head 1.3B
}


def ref_kv(m: dict, tokens, hw: dict = REF_HW):
    """Eq. 2: M_KV(T) = 2 L h_kv d_h b T bytes."""
    return 2 * m["L"] * m["hkv"] * m["dh"] * hw["bytes"] * tokens


def ref_ar(m: dict, n: int, prompt: int = 512, hw: dict = REF_HW) -> float:
    """Eq. 33: compute-bound prefill, then per decode step i the larger of memory and compute time."""
    ctx = prompt + np.arange(n, dtype=np.float64)
    weights = hw["bytes"] * m["P"]
    memory = (weights + ref_kv(m, ctx, hw)) / hw["bw"]
    compute = (2 * m["P"] + 4 * m["L"] * ctx * m["d"]) / hw["flops"]
    return 2 * m["P"] * prompt / hw["flops"] + float(np.maximum(memory, compute).sum())


def ref_nar(m: dict, n: int, k: int, prompt: int = 512, hw: dict = REF_HW) -> float:
    """Eq. 34: K full passes over T = prompt + N positions."""
    t = prompt + n
    return k * max(hw["bytes"] * m["P"] / hw["bw"], (2 * m["P"] * t + 4 * m["L"] * t * t * m["d"]) / hw["flops"])


def test_reference_roofline_reproduces_the_paper_table() -> None:
    """Validates the reference itself against the printed Table 3 before it is used on the library."""
    for name, n, ar_ms, k4_ms, k8_ms, ratio, kv_mib in PAPER_TABLE3:
        m = REF_MODELS[name]
        ar, k4, k8 = ref_ar(m, n), ref_nar(m, n, 4), ref_nar(m, n, 8)
        assert (round(ar * 1e3), round(k4 * 1e3), round(k8 * 1e3)) == (ar_ms, k4_ms, k8_ms)
        assert f"{ar / k8:.1f}" == ratio and round(ref_kv(m, 512 + n) / 2 ** 20) == kv_mib


def test_table3_reproduces_every_cell_of_the_paper() -> None:
    rows = table3()
    assert len(rows) == len(PAPER_TABLE3)
    for row, (name, n, ar_ms, k4_ms, k8_ms, ratio, kv_mib) in zip(rows, PAPER_TABLE3, strict=True):
        assert row["backbone"] == name and row["N"] == n
        assert round(row["ar_ms"]) == ar_ms
        assert round(row["nar_k4_ms"]) == k4_ms
        assert round(row["nar_k8_ms"]) == k8_ms
        assert f"{row['ratio_k8']:.1f}" == ratio
        assert round(row["kv_mib"]) == kv_mib
        m = REF_MODELS[name]                                            # and at full precision
        assert row["ar_ms"] == pytest.approx(ref_ar(m, n) * 1e3, rel=1e-12)
        assert row["nar_k8_ms"] == pytest.approx(ref_nar(m, n, 8) * 1e3, rel=1e-12)
        assert row["ratio_k8"] == pytest.approx(row["ar_ms"] / row["nar_k8_ms"], rel=1e-12)


def test_kv_cache_per_token_eq2() -> None:
    """Sec. 1.1: 56 KiB per token for Qwen2.5-Coder-7B, 192 KiB for the multi-head 1.3B model."""
    assert kv_bytes_per_token(QWEN25_CODER_7B) == 56 * 1024 == ref_kv(REF_MODELS["7.6B, GQA"], 1)
    assert kv_bytes_per_token(FSDFM_1P3B) == 192 * 1024 == ref_kv(REF_MODELS["1.3B, MHA"], 1)


def test_compute_bound_and_kv_limited_batches() -> None:
    """Sec. 5.4: AR becomes compute-bound at a batch of about 211 for both sizes; on 80 GB the KV cache admits
    736 / 130 sequences (7.6B GQA, N = 1,024 / 8,192) and 256 / 45 (1.3B MHA)."""
    b_star = REF_HW["flops"] / REF_HW["bw"]                              # (2P / BW_e) / (2P / F_e)
    for m in (QWEN25_CODER_7B, FSDFM_1P3B):
        assert ar_compute_bound_batch(m) == pytest.approx(b_star, rel=1e-12)
        assert round(ar_compute_bound_batch(m)) == 211
    for m, name, expected in ((QWEN25_CODER_7B, "7.6B, GQA", (736, 130)), (FSDFM_1P3B, "1.3B, MHA", (256, 45))):
        ref = REF_MODELS[name]
        for n, want in zip((1024, 8192), expected, strict=True):
            value = kv_limited_batch(m, n)
            assert round(value) == want
            assert value == pytest.approx((80e9 - 2 * ref["P"]) / ref_kv(ref, 512 + n), rel=1e-12)


@pytest.mark.parametrize("hw_kind", ["h100", "memory_rich", "compute_rich"])
def test_eq33_eq34_against_an_independent_implementation(hw_kind) -> None:
    hardware = {
        "h100": H100_SXM,
        "memory_rich": Hardware("toy-bw", bandwidth=8e12, flops=5e13, eta_bw=0.8, eta_flops=0.6),
        "compute_rich": Hardware("toy-f", bandwidth=5e11, flops=2e15, eta_bw=0.9, eta_flops=0.4),
    }[hw_kind]
    hw = {"bw": hardware.bandwidth * hardware.eta_bw, "flops": hardware.flops * hardware.eta_flops, "bytes": 2,
          "memory": 80e9}
    model = Backbone("toy", 3.0e8, 12, 6, 64, 768)
    ref = {"P": 3.0e8, "L": 12, "hkv": 6, "dh": 64, "d": 768}
    for prompt in (0, 16, 512):
        for n in (1, 7, 300, 2048):
            assert ar_latency(model, n, prompt, hardware) == pytest.approx(ref_ar(ref, n, prompt, hw), rel=1e-12)
            for k in (1, 2, 4, 8):
                assert nar_latency(model, n, k, prompt, hardware) == pytest.approx(ref_nar(ref, n, k, prompt, hw),
                                                                                    rel=1e-12)
    assert nar_latency(model, 64, 4, 16, hardware, planner_ms=3.0) == pytest.approx(
        ref_nar(ref, 64, 4, 16, hw) + 3e-3, rel=1e-12)


def test_saturated_throughput_ratio() -> None:
    """NAR / AR throughput when both are compute-bound: N / (K (512 + N)), about 1/K for long outputs."""
    for n, k in ((256, 4), (1024, 8), (4096, 8), (1, 1)):
        assert saturated_throughput_ratio(n, k) == pytest.approx(n / (k * (512 + n)), rel=1e-15)
    assert saturated_throughput_ratio(1024, 8) == pytest.approx(1 / 12)
    assert saturated_throughput_ratio(10 ** 9, 8) == pytest.approx(1 / 8, rel=1e-6)
    assert saturated_throughput_ratio(100, 2, prompt=100) == pytest.approx(0.25)


# =============================================================================================
# Simulated AR baseline: the KV cache must not change the greedy decode
# =============================================================================================
def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def recompute_last_logits(decoder: SimulatedARDecoder, tokens: torch.Tensor) -> torch.Tensor:
    """Pre-LN causal transformer over the whole sequence (no cache), float64, the decoder's weights."""
    w = {k: v.detach().double() for k, v in decoder.state_dict().items()}
    b, t = tokens.shape
    d = w["embed.weight"].shape[1]
    heads = decoder.num_heads
    hd = d // heads
    x = w["embed.weight"][tokens] + w["pos.weight"][:t]
    causal = torch.ones(t, t, dtype=torch.bool).tril()

    def split(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(b, t, heads, hd).transpose(1, 2)

    for i in range(len(decoder.layers)):
        p = f"layers.{i}."
        qkv = layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"]) @ w[p + "qkv.weight"].T + w[p + "qkv.bias"]
        q, k, v = split(qkv[..., :d]), split(qkv[..., d:2 * d]), split(qkv[..., 2 * d:])
        scores = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~causal, float("-inf"))
        attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + attn @ w[p + "out.weight"].T + w[p + "out.bias"]
        h = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
        x = x + gelu(h @ w[p + "fc1.weight"].T + w[p + "fc1.bias"]) @ w[p + "fc2.weight"].T + w[p + "fc2.bias"]
    return layer_norm(x[:, -1], w["norm.weight"], w["norm.bias"]) @ w["head.weight"].T + w["head.bias"]


def test_simulated_ar_decoder_kv_cache_equals_recomputation() -> None:
    torch.manual_seed(0)
    decoder = SimulatedARDecoder(vocab_size=37, dim=32, num_layers=2, num_heads=4, mlp_ratio=2.0, max_len=40).eval()
    prompt = torch.randint(0, 37, (3, 5), generator=torch.Generator().manual_seed(1))
    steps = 14
    cached = decoder.generate(prompt, steps)
    assert cached.shape == (3, steps) and cached.dtype == torch.int64
    sequence, expected = prompt.clone(), []
    for _ in range(steps):
        logits = recompute_last_logits(decoder, sequence)
        top2 = logits.topk(2, dim=-1).values
        assert float((top2[:, 0] - top2[:, 1]).min()) > 1e-6         # no near-tie: the argmax is well defined
        token = logits.argmax(-1, keepdim=True)
        expected.append(token)
        sequence = torch.cat([sequence, token], dim=1)
    assert torch.equal(cached, torch.cat(expected, dim=1))
    with pytest.raises(ValueError):
        decoder.generate(prompt, 40 - 5 + 1)                              # prompt + N > max_len


def test_measure_ar_latency_records() -> None:
    decoder = SimulatedARDecoder(vocab_size=19, dim=16, num_layers=1, num_heads=2, max_len=32).eval()
    records = measure_ar_latency(decoder, torch.zeros(1, 4, dtype=torch.long), [3, 5], warmup=1, repeats=3)
    assert [(r.method, r.num_steps, r.length, r.depth) for r in records] == [("ar_simulated", None, 3, 4),
                                                                              ("ar_simulated", None, 5, 6)]
    assert all(0 < r.median_ms <= r.p95_ms for r in records)


# =============================================================================================
# Measured sampler latency
# =============================================================================================
class _ScriptedSampler:
    """Stand-in for PRJEPASampler: reports a fixed depth and per-stage timings, records its calls."""

    def __init__(self, depth_offset: int = 3) -> None:
        self.depth_offset = depth_offset
        self.calls: list[tuple[int, int]] = []

    def sample(self, context, length=None, num_steps=None, generator=None, profile=False):
        self.calls.append((length, num_steps))
        stage = float(len(self.calls))
        return SimpleNamespace(depth=num_steps + self.depth_offset, timings_ms={"planner": stage, "realizer": 2 * stage})


def test_measure_sampler_latency_statistics_with_a_scripted_clock(monkeypatch) -> None:
    warmup, repeats = 2, 20
    durations_ms = [1000.0, 900.0] + [float(v) for v in random_permutation(20, seed=4)]   # warm-ups are excluded
    ticks: list[float] = []
    now = 0.0
    for duration in durations_ms:
        ticks += [now, now + duration / 1e3]
        now += 1.0
    clock = iter(ticks)
    monkeypatch.setattr(profiler, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    sampler = _ScriptedSampler()
    (record,) = measure_sampler_latency(sampler, torch.zeros(1, 3, dtype=torch.long), [16], [4], warmup=warmup,
                                        repeats=repeats)
    measured = sorted(durations_ms[warmup:])
    assert record.method == "pr_jepa" and record.num_steps == 4 and record.length == 16 and record.depth == 7
    assert record.median_ms == pytest.approx(10.5)                       # median of 1..20
    below = sum(v <= record.p95_ms for v in measured)
    above = sum(v >= record.p95_ms for v in measured)
    assert below >= 0.95 * repeats and above >= 0.05 * repeats            # a 95th percentile of the measured runs
    assert record.stages_ms == pytest.approx({"planner": 12.5, "realizer": 25.0})   # medians over runs 3..22
    assert sampler.calls == [(16, 4)] * (warmup + repeats)


def random_permutation(n: int, seed: int) -> list[int]:
    return (torch.randperm(n, generator=torch.Generator().manual_seed(seed)) + 1).tolist()


def test_measure_sampler_latency_enforces_the_depth_audit() -> None:
    with pytest.raises(AssertionError):
        measure_sampler_latency(_ScriptedSampler(depth_offset=2), torch.zeros(1, 3, dtype=torch.long), [8], [2],
                                warmup=0, repeats=1)
    with pytest.raises(ValueError):
        measure_sampler_latency(_ScriptedSampler(), torch.zeros(2, 3, dtype=torch.long), [8], [2])


# =============================================================================================
# A tiny Stage-2 checkpoint for the real sampler and the CLIs
# =============================================================================================
PROMPT = "Write solve(a, b, c) that returns a * b + c."
CONTROL_LABELS = ("unparseable", "straight_line", "for_loop", "while_loop", "recursion", "comprehension")


@pytest.fixture(scope="module")
def stage2_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("stage2")
    records = synthetic_corpus(16, seed=0)
    tokenizer = train_bpe_tokenizer([r.target for r in records] + [r.context for r in records], vocab_size=400)
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=tiny", "data.tokenizer=bpe", "training.batch_size=4",
                       "training.realizer.warmup_steps=1", "training.realizer.max_steps=10"]
                      ).with_vocab_size(tokenizer.vocab_size)
    dataset = PRJEPADataset(records, tokenizer, cfg.planner.num_slots, cfg.data.max_context_tokens,
                            cfg.data.max_output_tokens, cfg.data.max_structure_tokens, cfg.readout.max_text_tokens,
                            cfg.data.mode_scheme)
    mixture = MixtureCorruption(tokenizer.vocab_size, cfg.realizer.step_budgets, cfg.realizer.schedule,
                                cfg.realizer.source, tokenizer.mask_id)
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens, mixture)
    batches = [collator([dataset[i] for i in range(j * 4, j * 4 + 4)]) for j in range(2)]
    model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
    trainer = Stage2Trainer(cfg, model, tokenizer, batches, out, DistributedContext(device=torch.device("cpu")),
                            None, MetricLogger(out, enabled=False))
    result = trainer.train(max_steps=2)
    return result.best_checkpoint


def test_measure_sampler_latency_on_a_tiny_model(stage2_checkpoint) -> None:
    loaded = load_model(stage2_checkpoint)
    sampler = build_sampler(loaded)
    context = torch.tensor([loaded.tokenizer.encode(PROMPT)])
    records = measure_sampler_latency(sampler, context, lengths=[8, 12], steps=[1, 2], warmup=1, repeats=3)
    assert [(r.length, r.num_steps) for r in records] == [(8, 1), (8, 2), (12, 1), (12, 2)]
    for r in records:
        assert r.method == "pr_jepa" and r.depth == r.num_steps + 3
        assert 0 < r.median_ms <= r.p95_ms
        assert "pass1-3_planner" in r.stages_ms
        assert {f"pass{k + 4}_realizer" for k in range(r.num_steps)} <= set(r.stages_ms)


def parse_generation(out: str) -> tuple[int, str, float, int, int]:
    sample = re.search(r"--- sample 0: mode u = (\d+) \(([^)]*)\), sigma_bar = (\d+\.\d{4})", out)
    depth = re.search(r"sequential network evaluations: (\d+) \(K \+ 3 = (\d+), audited\)", out)
    assert sample and depth, out
    return int(sample.group(1)), sample.group(2), float(sample.group(3)), int(depth.group(1)), int(depth.group(2))


def eq8_sigma_bar(loaded, prompt: str, mode: int) -> float:
    """Eq. 8 from the ensemble heads of the loaded planner for the printed mode (float64)."""
    ids = loaded.tokenizer.encode(prompt)[: loaded.cfg.data.max_context_tokens]
    with torch.no_grad():
        plan = loaded.model.planner.plan(torch.tensor([ids]), mode=torch.tensor([mode]))
    heads = plan.head_predictions.double()[:, 0]                          # [n_ens, M, d]
    z = heads.mean(dim=0)
    n_ens, m, d = heads.shape
    return math.sqrt(float((heads - z).pow(2).sum()) / (n_ens * m * d))


@pytest.mark.parametrize("k_steps", [2, None])
def test_generate_cli_prints_mode_sigma_bar_and_depth(stage2_checkpoint, capsys, k_steps) -> None:
    args = ["--prompt", PROMPT, "--checkpoint", str(stage2_checkpoint), "--length", "20", "--device", "cpu",
            "--seed", "3"]
    if k_steps is not None:
        args += ["--k_steps", str(k_steps)]
    assert generate_cli.main(args) == 0
    out = capsys.readouterr().out
    loaded = load_model(stage2_checkpoint)
    k = k_steps if k_steps is not None else loaded.cfg.realizer.num_steps
    mode, label, sigma_bar, depth, k_plus_3 = parse_generation(out)
    assert depth == k_plus_3 == k + 3
    assert f"K = {k}, N = 20" in out
    assert 0 <= mode < loaded.cfg.planner.num_modes
    assert label == (CONTROL_LABELS[mode] if mode < len(CONTROL_LABELS) else "unassigned mode")
    assert abs(sigma_bar - eq8_sigma_bar(loaded, PROMPT, mode)) <= 5e-5 + 1e-7    # printed with 4 decimals
    assert "latency breakdown (ms)" in out and "pass1-3_planner" in out


def test_generate_cli_rejects_a_realizer_mismatch(stage2_checkpoint, capsys) -> None:
    rc = generate_cli.main(["--prompt", PROMPT, "--checkpoint", str(stage2_checkpoint), "--realizer", "logit_kl",
                            "--device", "cpu"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "fs_dfm" in captured.err and "sequential network evaluations" not in captured.out
    assert generate_cli.main(["--prompt", PROMPT, "--checkpoint", str(stage2_checkpoint), "--realizer", "fs_dfm",
                              "--k_steps", "2", "--length", "8", "--device", "cpu", "--no-energy"]) == 0
    assert "hard-constraint violations" not in capsys.readouterr().out


# =============================================================================================
# Profiler CLI
# =============================================================================================
def test_profiler_cli_prints_table3_and_the_batch_figures(capsys) -> None:
    profiler.main(["--roofline"])
    out = capsys.readouterr().out
    pattern = re.compile(r"^(\S+, \S+)\s+(\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)x\s+([\d,]+)\s*$")
    rows = [m.groups() for m in map(pattern.match, out.splitlines()) if m]
    assert len(rows) == len(PAPER_TABLE3)
    for row, (name, n, ar_ms, k4_ms, k8_ms, ratio, kv_mib) in zip(rows, PAPER_TABLE3, strict=True):
        assert row[0] == name and int(row[1]) == n
        assert [int(v.replace(",", "")) for v in (row[2], row[3], row[4], row[6])] == [ar_ms, k4_ms, k8_ms, kv_mib]
        assert row[5] == ratio
    assert re.search(r"7\.6B, GQA: AR compute-bound batch ~ 211; KV-limited batch 736 \(N = 1,024\), 130", out)
    assert re.search(r"1\.3B, MHA: AR compute-bound batch ~ 211; KV-limited batch 256 \(N = 1,024\), 45", out)


def test_profiler_cli_measures_a_checkpoint(stage2_checkpoint, capsys) -> None:
    profiler.main(["--checkpoint", str(stage2_checkpoint), "--lengths", "8", "--steps", "1", "2", "--repeats", "2",
                   "--device", "cpu"])
    out = capsys.readouterr().out
    for k in (1, 2):
        assert re.search(rf"PR-JEPA K={k} N=8: .* depth {k + 3}$", out, re.MULTILINE), out
    assert re.search(r"AR \(simulated, same width/depth\) N=8: .* depth 9$", out, re.MULTILINE), out
