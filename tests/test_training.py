# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Independent tests of the staged training protocol (PR-JEPA Sec. 5.1) and its runtime:

    runtime.py          warm-up + cosine schedule, AdamW groups, atomic weights-only checkpoints, JSONL metrics
    trainer_stage1.py   L_plan steps, the EMA target encoder (Eq. 4), the stable-rank collapse guard, resume
    trainer_stage2.py   Stage 2 (planner frozen, anti-leakage invariant, predicted plan only) and Stage 3
    loading.py          load_model refuses Stage-1 and rejected checkpoints
    train.py            the CLI of the three stages

References are written from the paper and the docstrings (Eq. 4 recursion in float64, the schedule
formula, hand-built expectations), never with the function under test. Tiny configs, synthetic data and
checkpoints under tmp_path only.
"""

from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import train as train_cli
from tests.conftest import ATOL, RTOL
from text_vpa.configs import load_config
from text_vpa.data.dataset import MixtureCorruption, PRJEPACollator, PRJEPADataset, build_dataloader
from text_vpa.data.synthetic import TEMPLATES, synthetic_corpus
from text_vpa.data.tokenization import train_bpe_tokenizer
from text_vpa.engine.runtime import (
    DistributedContext,
    MetricLogger,
    autocast,
    build_optimizer,
    load_checkpoint,
    rng_state,
    save_checkpoint,
    set_rng_state,
    warmup_cosine,
)
from text_vpa.engine.trainer_stage1 import Stage1Trainer
from text_vpa.engine.trainer_stage2 import LeakageError, Stage2Trainer, Stage3Trainer
from text_vpa.inference.loading import load_model
from text_vpa.models.encoders import TargetEncoder
from text_vpa.models.plan import PredictedPlan
from text_vpa.models.text_vpa_model import TextVPAModel, TextVPAPlanner

M0 = 0.996                                    # Eq. 4
CPU = DistributedContext(device=torch.device("cpu"))
BASE = [
    "planner=tiny", "data=tiny", "realizer=tiny", "data.tokenizer=bpe", "training.batch_size=4",
    "training.planner.warmup_steps=2", "training.planner.max_steps=40",
    "training.realizer.warmup_steps=1", "training.realizer.max_steps=40",
    "training.joint.warmup_steps=1", "training.joint.max_steps=20",
    "training.log_every=1000", "training.save_every=100000",
]


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus(48, seed=0)


@pytest.fixture(scope="module")
def tokenizer(corpus):
    return train_bpe_tokenizer([r.target for r in corpus] + [r.context for r in corpus], vocab_size=400)


def make_cfg(tokenizer, *extra: str):
    return load_config([*BASE, *extra]).with_vocab_size(tokenizer.vocab_size)


def make_dataset(cfg, tokenizer, records) -> PRJEPADataset:
    return PRJEPADataset(records, tokenizer, cfg.planner.num_slots, cfg.data.max_context_tokens,
                         cfg.data.max_output_tokens, cfg.data.max_structure_tokens, cfg.readout.max_text_tokens,
                         cfg.data.mode_scheme)


def make_batches(cfg, tokenizer, records, n_batches: int, batch_size: int, corruption: bool = False):
    """A fixed list of collated batches (deterministic data order for every trainer)."""
    dataset = make_dataset(cfg, tokenizer, records)
    assert len(dataset) >= n_batches * batch_size
    mixture = (MixtureCorruption(tokenizer.vocab_size, cfg.realizer.step_budgets, cfg.realizer.schedule,
                                 cfg.realizer.source, tokenizer.mask_id) if corruption else None)
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens, mixture)
    torch.manual_seed(1234)
    return [collator([dataset[i] for i in range(j * batch_size, (j + 1) * batch_size)]) for j in range(n_batches)]


def new_planner(cfg, seed: int = 0) -> TextVPAPlanner:
    torch.manual_seed(seed)
    return TextVPAPlanner.from_config(cfg.planner)


def new_model(cfg, tokenizer, seed: int = 0) -> TextVPAModel:
    torch.manual_seed(seed)
    return TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)


def quiet(tmp_path) -> MetricLogger:
    return MetricLogger(tmp_path, enabled=False)


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def changed(before: dict[str, torch.Tensor], module: nn.Module) -> list[str]:
    after = module.state_dict()
    return [k for k, v in before.items() if not torch.equal(v, after[k])]


# =============================================================================================
# runtime.py
# =============================================================================================
def reference_factor(step: int, warmup: int, total: int, final: float = 0.1) -> float:
    """Docstring: linear warm-up to the group's lr (the k-th of the first W updates runs at k/W of it), then a
    cosine from 1 at the end of the warm-up down to ``final`` at ``total``, constant afterwards."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / (total - warmup))
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def test_warmup_cosine_factors_follow_the_formula() -> None:
    warmup, total = 4, 12
    params = [nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))]
    optimizer = torch.optim.SGD([{"params": [params[0]], "lr": 1.0}, {"params": [params[1]], "lr": 0.5}], lr=1.0)
    scheduler = warmup_cosine(optimizer, warmup, total)
    seen = []
    for _ in range(total + 4):
        seen.append([g["lr"] for g in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    for step, (lr_a, lr_b) in enumerate(seen):
        f = reference_factor(step, warmup, total)
        assert lr_a == pytest.approx(f, abs=1e-12), step
        assert lr_b == pytest.approx(0.5 * f, abs=1e-12), step         # each group's own lr
    assert [round(seen[s][0], 12) for s in (0, 1, 3, 4, 8, 12, 15)] == [0.25, 0.5, 1.0, 1.0, 0.55, 0.1, 0.1]
    assert all(seen[s + 1][0] <= seen[s][0] + 1e-12 for s in range(warmup, total + 3))   # no rise after warm-up


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(11, 4)
        self.norm = nn.LayerNorm(4)
        self.proj = nn.Linear(4, 3)
        self.frozen = nn.Linear(3, 3).requires_grad_(False)


def group_sets(optimizer) -> tuple[set[int], set[int]]:
    decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    no_decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0 for p in g["params"]}
    return decay, no_decay


def test_build_optimizer_groups_on_a_toy_module() -> None:
    toy = _Toy()
    optimizer = build_optimizer([("toy", toy, 0.01)], weight_decay=0.05)
    assert isinstance(optimizer, torch.optim.AdamW)
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert set(groups) == {"toy/decay", "toy/no_decay"}
    assert groups["toy/decay"]["weight_decay"] == 0.05 and groups["toy/no_decay"]["weight_decay"] == 0.0
    assert all(g["lr"] == 0.01 for g in groups.values())
    assert {id(p) for p in groups["toy/decay"]["params"]} == {id(toy.proj.weight)}
    assert {id(p) for p in groups["toy/no_decay"]["params"]} == {
        id(toy.token_embedding.weight), id(toy.norm.weight), id(toy.norm.bias), id(toy.proj.bias)}
    with pytest.raises(ValueError):
        build_optimizer([("frozen", toy.frozen, 0.01)], weight_decay=0.05)


def test_build_optimizer_groups_on_the_text_vpa_model(tokenizer) -> None:
    """Every embedding table and norm parameter is undecayed, every Linear weight decayed, each exactly once."""
    model = new_model(make_cfg(tokenizer), tokenizer)
    optimizer = build_optimizer([("model", model, 1e-3)], weight_decay=0.1)
    decay, no_decay = group_sets(optimizer)
    assert not decay & no_decay
    assert decay | no_decay == {id(p) for p in model.parameters() if p.requires_grad}
    counts = {"embedding": 0, "norm": 0, "linear": 0}
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            assert id(module.weight) in no_decay
            counts["embedding"] += 1
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            assert {id(module.weight), id(module.bias)} <= no_decay
            counts["norm"] += 1
        elif isinstance(module, nn.Linear):
            weight = (module.parametrizations.weight.original if parametrize.is_parametrized(module, "weight")
                      else module.weight)
            assert id(weight) in decay
            if module.bias is not None:
                assert id(module.bias) in no_decay
            counts["linear"] += 1
    assert all(counts.values())
    for p in model.parameters():                                       # every 1-D tensor is undecayed
        if p.ndim < 2:
            assert id(p) in no_decay
    named = dict(model.named_parameters())                             # learned query / position embeddings
    assert id(named["planner.predictor.slot_queries"]) in no_decay     # the slot queries Delta (Eq. 3)
    assert id(named["readout.text_positions"]) in no_decay


def test_build_optimizer_does_not_decay_an_embedding_with_a_generic_name() -> None:
    module = nn.Module()
    module.lookup = nn.Embedding(10, 4)
    module.proj = nn.Linear(4, 4)
    decay, no_decay = group_sets(build_optimizer([("m", module, 1e-3)], weight_decay=0.1))
    assert id(module.proj.weight) in decay
    assert id(module.lookup.weight) in no_decay


class _NotAllowed:
    """A pickled object that a weights-only load must refuse."""


def test_checkpoint_round_trip_is_atomic_and_weights_only(tmp_path) -> None:
    payload = {"stage": 2, "step": 7, "weights": {"a": torch.randn(3, 4), "b": torch.arange(5)},
               "config": {"x": [1, 2.5, "s", None, True]}, "rejected": False}
    path = save_checkpoint(tmp_path / "nested" / "ckpt.pt", payload)
    assert path == tmp_path / "nested" / "ckpt.pt" and path.exists()
    assert [p.name for p in path.parent.iterdir()] == ["ckpt.pt"]       # no temporary file left behind
    loaded = load_checkpoint(path)
    assert loaded["format"] == "text-vpa/1"
    assert loaded["stage"] == 2 and loaded["step"] == 7 and loaded["rejected"] is False
    assert loaded["config"] == payload["config"]
    assert torch.equal(loaded["weights"]["a"], payload["weights"]["a"])
    assert torch.equal(loaded["weights"]["b"], payload["weights"]["b"])
    assert torch.load(path, weights_only=True)["step"] == 7            # loadable without arbitrary pickles
    unsafe = save_checkpoint(tmp_path / "unsafe.pt", {"obj": _NotAllowed()})
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(unsafe)


def test_checkpoint_with_a_wrong_format_tag_is_refused(tmp_path) -> None:
    torch.save({"format": "other/1", "stage": 1}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "wrong.pt")
    torch.save({"stage": 1}, tmp_path / "untagged.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "untagged.pt")


def strict_json(line: str):
    def refuse(name: str):
        raise ValueError(f"non-standard JSON constant {name}")
    return json.loads(line, parse_constant=refuse)


def test_metric_logger_writes_valid_jsonl(tmp_path, capsys) -> None:
    logger = MetricLogger(tmp_path / "run")
    logger.log(1, {"loss": 2.5, "count": 3, "lr": torch.tensor(1e-3)}, prefix="stage1/")
    logger.log(2, {"loss": 1.25, "note": "ok"})
    logger.close()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    rows = [strict_json(line) for line in lines]
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["stage1/loss"] == 2.5 and rows[0]["stage1/count"] == 3.0
    assert rows[0]["stage1/lr"] == pytest.approx(1e-3) and rows[1]["note"] == "ok" and rows[1]["loss"] == 1.25
    assert all(isinstance(r["time_s"], float) for r in rows)
    assert "loss=2.5" in capsys.readouterr().out
    MetricLogger(tmp_path / "off", enabled=False).log(1, {"loss": 1.0})
    assert not (tmp_path / "off" / "metrics.jsonl").exists()


def test_metric_logger_keeps_jsonl_valid_for_non_finite_metrics(tmp_path) -> None:
    logger = MetricLogger(tmp_path)
    logger.log(1, {"loss": float("nan"), "grad_norm": float("inf")})
    for line in (tmp_path / "metrics.jsonl").read_text().splitlines():
        strict_json(line)


def test_single_process_context_and_autocast(monkeypatch) -> None:
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    ctx = DistributedContext.initialize("cpu")
    assert (ctx.rank, ctx.world_size, ctx.local_rank, ctx.device) == (0, 1, 0, torch.device("cpu"))
    assert not ctx.enabled and ctx.is_main
    assert ctx.any_true(True) is True and ctx.any_true(False) is False and ctx.mean(2.5) == 2.5
    ctx.barrier()
    ctx.shutdown()
    a = torch.randn(4, 4)
    with autocast(torch.device("cpu"), "bf16"):
        assert (a @ a).dtype == torch.bfloat16
    with autocast(torch.device("cpu"), "fp32"):
        assert (a @ a).dtype == torch.float32


def test_rng_state_round_trip() -> None:
    state = rng_state()
    first = (torch.rand(4), random.random())
    set_rng_state(state)
    second = (torch.rand(4), random.random())
    assert torch.equal(first[0], second[0]) and first[1] == second[1]


# =============================================================================================
# Stage 1
# =============================================================================================
def test_stage1_ema_follows_eq4_and_the_hand_recursion(tmp_path, tokenizer, corpus) -> None:
    total = 40                                                          # T_train = training.planner.max_steps
    cfg = make_cfg(tokenizer, f"training.planner.max_steps={total}")
    batches = make_batches(cfg, tokenizer, corpus, 5, 4)
    trainer = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path, CPU, quiet(tmp_path))
    online = dict(trainer.planner.context_encoder.named_parameters())
    target = dict(trainer.target_encoder.named_parameters())
    assert online.keys() == target.keys()
    for name in online:                                                 # theta_bar <- theta at initialisation
        assert torch.equal(target[name], online[name])
    start = {n: p.detach().clone() for n, p in online.items()}
    hand = {n: p.detach().double().clone() for n, p in online.items()}
    for k, batch in enumerate(batches):
        metrics = trainer.train_step(batch)
        m_k = M0 + (1.0 - M0) * k / total                               # Eq. 4, k = updates so far
        assert metrics["ema_momentum"] == pytest.approx(m_k, abs=1e-12)
        for name, p in online.items():                                  # theta_bar <- m_k theta_bar + (1 - m_k) theta
            hand[name] = m_k * hand[name] + (1.0 - m_k) * p.detach().double()
    for name, p in target.items():
        torch.testing.assert_close(p.detach().double(), hand[name], rtol=RTOL, atol=ATOL, msg=name)
        assert not p.requires_grad
    assert max(float((online[n].detach() - start[n]).abs().max()) for n in online) > 1e-4   # the online encoder trained
    assert max(float((online[n].detach() - target[n]).abs().max()) for n in online) > 1e-6  # the target lags behind
    in_optimizer = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert not in_optimizer & {id(p) for p in trainer.target_encoder.parameters()}
    assert trainer.step == 5


def test_stage1_feeds_the_reference_mode_to_g_phi_and_l_ce(tmp_path, tokenizer, corpus) -> None:
    """Sec. 5.1: modes u are extracted from the reference; g_phi is fed u and L_CE scores pi_phi^h against it."""
    cfg = make_cfg(tokenizer)
    batch = make_batches(cfg, tokenizer, corpus, 1, 4)[0]
    trainer = Stage1Trainer(cfg, new_planner(cfg), tokenizer, [batch], tmp_path, CPU, quiet(tmp_path))
    seen: dict[str, torch.Tensor] = {}
    hooks = [
        trainer.planner.predictor.register_forward_hook(lambda _m, args, _o: seen.update(u=args[1].clone())),
        trainer.planner.selector.register_forward_hook(lambda _m, _a, out: seen.update(logits=out.detach().clone())),
    ]
    metrics = trainer.train_step(batch)
    for hook in hooks:
        hook.remove()
    assert torch.equal(seen["u"], batch.modes)
    log_p = seen["logits"].double().log_softmax(dim=-1)
    cross_entropy = -log_p[torch.arange(batch.batch_size), batch.modes].mean()
    assert metrics["mode_ce"] == pytest.approx(float(cross_entropy), rel=1e-5)


def real_slot_mask(batch, num_slots: int) -> torch.Tensor:
    """[B, M]: slots that own at least one real token of the structural view y."""
    mask = torch.zeros(batch.batch_size, num_slots, dtype=torch.bool)
    for b, (index, keep) in enumerate(zip(batch.slot_index.tolist(), batch.structure_mask.tolist(), strict=True)):
        for slot, real in zip(index, keep, strict=True):
            if real and slot >= 0:
                mask[b, slot] = True
    return mask


def slot_rows(batch) -> int:
    """Real slots of a batch (rows of the plan stream)."""
    return int(real_slot_mask(batch, int(batch.slot_index.max()) + 1).sum())


@pytest.fixture(scope="module")
def many_slot_records():
    """Programs with five slots each (import, signature, two variables, a for block)."""
    template = [t for t in TEMPLATES if t.__name__ == "_for_sum"]
    return synthetic_corpus(24, seed=1, templates=tuple(template))


@pytest.mark.parametrize("grace", [0, 2, 3])
def test_stage1_collapse_rejection_after_the_grace_period(tmp_path, tokenizer, many_slot_records, grace) -> None:
    cfg = make_cfg(tokenizer, "training.batch_size=8", "vicreg.srank_abort_fraction=1.0",
                   "training.collapse_check_every=1", f"training.collapse_grace_steps={grace}")
    batches = make_batches(cfg, tokenizer, many_slot_records, 3, 8)
    d = cfg.planner.latent_dim
    for batch in batches:                                               # every step fills a whole window
        assert slot_rows(batch) > d and int(batch.context_mask.sum()) > d
    trainer = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path, CPU, quiet(tmp_path))
    result = trainer.train(max_steps=6)
    # srank(C) <= rank(C) <= d with equality only for an isotropic C, so a threshold of 1.0 d always fires;
    # it may abort only from step max(grace, 1) on.
    assert result.rejected and result.steps == max(grace, 1)
    assert result.checkpoint == tmp_path / "stage1_rejected.pt" and result.checkpoint.exists()
    assert not (tmp_path / "stage1_final.pt").exists()
    payload = load_checkpoint(result.checkpoint)
    assert payload["rejected"] is True and payload["stage"] == 1 and payload["step"] == max(grace, 1)
    assert "stable rank" in payload["rejection_reason"]
    assert min(v for k, v in result.collapse_report.items() if k.endswith("/stable_rank")) < d


def test_stage1_no_abort_inside_the_grace_period(tmp_path, tokenizer, many_slot_records) -> None:
    cfg = make_cfg(tokenizer, "training.batch_size=8", "vicreg.srank_abort_fraction=1.0",
                   "training.collapse_check_every=1", "training.collapse_grace_steps=1000")
    batches = make_batches(cfg, tokenizer, many_slot_records, 3, 8)
    trainer = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path, CPU, quiet(tmp_path))
    result = trainer.train(max_steps=3)
    assert not result.rejected and result.steps == 3
    assert result.checkpoint == tmp_path / "stage1_final.pt"
    assert not (tmp_path / "stage1_rejected.pt").exists()
    payload = load_checkpoint(result.checkpoint)
    assert payload["rejected"] is False and payload["step"] == 3
    # the abort condition held at every check; only the grace period prevented the abort
    assert min(v for k, v in trainer.last_report.items() if k.endswith("/stable_rank")) < cfg.planner.latent_dim


def test_stage1_stable_rank_window_matches_an_independent_covariance(tmp_path, tokenizer, many_slot_records) -> None:
    """The guard's window statistics equal Eqs. 6-7 and srank(C) = (tr C)^2 / ||C||_F^2 of all rows of the window
    (context latents at real tokens, plan rows at real slots), computed here in float64 from captured latents."""
    cfg = make_cfg(tokenizer, "training.batch_size=8", "training.collapse_check_every=2",
                   "training.collapse_grace_steps=1000")
    batches = make_batches(cfg, tokenizer, many_slot_records, 2, 8)
    trainer = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path, CPU, MetricLogger(tmp_path))
    captured: dict[str, list[torch.Tensor]] = {"context": [], "plan": []}
    hooks = [
        trainer.planner.context_encoder.register_forward_hook(
            lambda _m, _a, out: captured["context"].append(out.detach().clone())),
        trainer.planner.predictor.register_forward_hook(
            lambda _m, _a, out: captured["plan"].append(out.z_target.detach().clone())),
    ]
    result = trainer.train(max_steps=2)
    for hook in hooks:
        hook.remove()
    assert not result.rejected
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    checks = [r for r in rows if "collapse/context/stable_rank" in r]
    assert [r["step"] for r in checks] == [2]                           # one window of collapse_check_every steps
    d, gamma, eps = cfg.planner.latent_dim, 1.0, 1e-4
    stacked = {
        "context": torch.cat([s[b.context_mask] for s, b in zip(captured["context"], batches, strict=True)]),
        "plan": torch.cat([z[real_slot_mask(b, cfg.planner.num_slots)]
                           for z, b in zip(captured["plan"], batches, strict=True)]),
    }
    for stream, z in stacked.items():
        z = z.double()
        centred = z - z.mean(dim=0)
        cov = centred.T @ centred / (z.shape[0] - 1)                    # Eq. 7, unbiased
        diag = torch.diagonal(cov)
        off = cov - torch.diag(diag)
        eig = torch.linalg.eigvalsh(cov)
        logged = checks[0]
        prefix = f"collapse/{stream}/"
        srank = float(diag.sum() ** 2 / cov.pow(2).sum())
        assert logged[prefix + "stable_rank"] == pytest.approx(srank, rel=1e-4)
        assert logged[prefix + "srank_fraction"] == pytest.approx(srank / d, rel=1e-4)
        assert logged[prefix + "variance_hinge"] == pytest.approx(
            float(torch.clamp(gamma - torch.sqrt(diag + eps), min=0).mean()), rel=1e-4, abs=1e-7)       # Eq. 6
        assert logged[prefix + "covariance_penalty"] == pytest.approx(float(off.pow(2).sum() / d), rel=1e-3)
        assert logged[prefix + "lambda_max"] == pytest.approx(float(eig[-1]), rel=1e-4)
        assert logged[prefix + "lambda_min"] == pytest.approx(float(eig[0]), abs=1e-5 * float(eig[-1]))


def assert_same_state(a: Stage1Trainer, b: Stage1Trainer) -> None:
    for (name, p), q in zip(a.planner.state_dict().items(), b.planner.state_dict().values(), strict=True):
        assert torch.equal(p, q), name
    for (name, p), q in zip(a.target_encoder.state_dict().items(), b.target_encoder.state_dict().values(),
                            strict=True):
        assert torch.equal(p, q), name
    for state_a, state_b in zip(a.optimizer.state.values(), b.optimizer.state.values(), strict=True):
        assert torch.equal(state_a["exp_avg"], state_b["exp_avg"])
        assert torch.equal(state_a["exp_avg_sq"], state_b["exp_avg_sq"])
    assert a.step == b.step and a.ema.step == b.ema.step
    assert [g["lr"] for g in a.optimizer.param_groups] == [g["lr"] for g in b.optimizer.param_groups]


def test_stage1_resume_continues_like_uninterrupted_training(tmp_path, tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer, "training.save_every=2")
    batches = make_batches(cfg, tokenizer, corpus, 4, 4)
    full = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path / "full", CPU, quiet(tmp_path))
    full.train(max_steps=4)
    first = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path / "first", CPU, quiet(tmp_path))
    first.train(max_steps=2)
    checkpoint = tmp_path / "first" / "stage1_last.pt"                  # written at step 2 (save_every = 2)
    saved = load_checkpoint(checkpoint)
    assert saved["stage"] == 1 and saved["step"] == 2 and saved["ema"]["step"] == 2
    torch.rand(100)                                                     # disturb the generator before resuming
    resumed = Stage1Trainer(cfg, new_planner(cfg, seed=99), tokenizer, batches[2:], tmp_path / "resumed", CPU,
                            quiet(tmp_path))
    resumed.resume(checkpoint)
    assert resumed.step == 2 and resumed.ema.step == 2
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
    resumed.train(max_steps=4)                                          # steps 3 and 4 on batches 3 and 4
    assert_same_state(resumed, full)


def test_stage1_resume_with_the_bucketed_loader_continues_like_uninterrupted(tmp_path, tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer, "training.save_every=2")
    dataset = make_dataset(cfg, tokenizer, corpus)
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens)

    def loader():
        return build_dataloader(dataset, collator, 4, cfg.data.bucket_size, shuffle=True, seed=0, drop_last=True)

    assert len(loader()) > 4                                            # 4 steps stay inside epoch 0
    full = Stage1Trainer(cfg, new_planner(cfg), tokenizer, loader(), tmp_path / "full", CPU, quiet(tmp_path))
    full.train(max_steps=4)
    first = Stage1Trainer(cfg, new_planner(cfg), tokenizer, loader(), tmp_path / "first", CPU, quiet(tmp_path))
    first.train(max_steps=2)
    resumed = Stage1Trainer(cfg, new_planner(cfg, seed=99), tokenizer, loader(), tmp_path / "resumed", CPU,
                            quiet(tmp_path))
    resumed.resume(tmp_path / "first" / "stage1_last.pt")
    resumed.train(max_steps=4)
    assert_same_state(resumed, full)


# =============================================================================================
# Stage 2
# =============================================================================================
@pytest.fixture
def stage2_setup(tmp_path, tokenizer, corpus):
    cfg = make_cfg(tokenizer)
    batches = make_batches(cfg, tokenizer, corpus, 3, 4, corruption=True)
    model = new_model(cfg, tokenizer)
    trainer = Stage2Trainer(cfg, model, tokenizer, batches, tmp_path, CPU, None, quiet(tmp_path))
    return cfg, model, trainer, batches


def test_stage2_planner_frozen_realizer_trained_on_the_predicted_plan(tmp_path, tokenizer, corpus,
                                                                      monkeypatch) -> None:
    cfg = make_cfg(tokenizer)
    batches = make_batches(cfg, tokenizer, corpus, 3, 4, corruption=True)
    model = new_model(cfg, tokenizer)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the target encoder f_theta_bar must not run in Stage 2")

    monkeypatch.setattr(TargetEncoder, "forward", forbidden)
    monkeypatch.setattr(TargetEncoder, "from_online", classmethod(forbidden))
    seen: list[PredictedPlan] = []
    prepare = model.realizer.prepare

    def recording_prepare(plan, *args, **kwargs):
        seen.append(plan)
        return prepare(plan, *args, **kwargs)

    monkeypatch.setattr(model.realizer, "prepare", recording_prepare)
    planner_before, realizer_before = snapshot(model.planner), snapshot(model.realizer)
    readout_before = snapshot(model.readout)
    trainer = Stage2Trainer(cfg, model, tokenizer, batches, tmp_path, CPU, None, quiet(tmp_path))
    result = trainer.train(max_steps=3)

    assert result.steps == 3
    assert changed(planner_before, model.planner) == []                 # bit-identical
    assert not any(p.requires_grad for p in model.planner.parameters()) and not model.planner.training
    assert changed(realizer_before, model.realizer) and changed(readout_before, model.readout)
    # the realizer was conditioned, at every step, on the plan the frozen planner predicts from the context
    assert len(seen) == 3 and all(isinstance(p, PredictedPlan) for p in seen)
    with torch.no_grad():
        for plan, batch in zip(seen, batches, strict=True):
            reference = model.planner.plan(batch.context, batch.context_mask, mode=batch.modes)
            assert torch.equal(plan.z_target, reference.z_target)
            assert torch.equal(plan.mode, batch.modes)
    assert result.best_checkpoint == tmp_path / "best_stage2.pt" and result.best_checkpoint.exists()
    assert result.checkpoint == tmp_path / "stage2_final.pt" and result.checkpoint.exists()
    payload = load_checkpoint(result.best_checkpoint)
    assert payload["stage"] == 2 and payload["rejected"] is False and "realizer" in payload
    assert all(torch.equal(v, planner_before[k]) for k, v in payload["planner"].items())


def test_stage2_best_checkpoint_follows_the_evaluation(tmp_path, tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer)
    batches = make_batches(cfg, tokenizer, corpus, 3, 4, corruption=True)
    model = new_model(cfg, tokenizer)
    trainer = Stage2Trainer(cfg, model, tokenizer, batches[:2], tmp_path, CPU, batches[2:], quiet(tmp_path))
    result = trainer.train(max_steps=2, eval_every=1)
    assert result.best_checkpoint == tmp_path / "best_stage2.pt" and result.best_checkpoint.exists()
    payload = load_checkpoint(result.best_checkpoint)
    assert payload["best_eval_loss"] == pytest.approx(result.best_eval_loss) and math.isfinite(result.best_eval_loss)


def test_stage2_plan_depends_on_the_context_only(stage2_setup) -> None:
    _, _, trainer, batches = stage2_setup
    batch = batches[0]
    generator = torch.Generator().manual_seed(0)
    scrambled = replace(
        batch,
        structure=torch.randint(0, 400, batch.structure.shape, generator=generator),
        slot_index=torch.zeros_like(batch.slot_index), structure_mask=torch.ones_like(batch.structure_mask),
        target=torch.randint(0, 400, batch.target.shape, generator=generator),
        w_t=torch.randint(0, 400, batch.target.shape, generator=generator),
        slot_kinds=torch.zeros_like(batch.slot_kinds), slot_texts=torch.zeros_like(batch.slot_texts),
        inventory=torch.ones_like(batch.inventory),
    )
    a, b = trainer.plan(batch), trainer.plan(scrambled)
    assert torch.equal(a.z_target, b.z_target) and torch.equal(a.s_x, b.s_x) and torch.equal(a.sigma_bar, b.sigma_bar)
    other = trainer.plan(replace(batch, context=batch.context.roll(1, dims=1)))
    assert not torch.equal(a.z_target, other.z_target)                  # not vacuous: the context matters


def violate(trainer: Stage2Trainer, kind: str) -> None:
    planner = trainer.model.planner
    weight = planner.predictor.blocks[0].self_attn.q_proj.weight
    if kind == "requires_grad":
        weight.requires_grad_(True)
    elif kind == "train_mode":
        planner.train()
    elif kind == "in_optimizer":
        trainer.optimizer.add_param_group({"params": [weight], "lr": 1e-3})
    elif kind == "weights_changed":
        with torch.no_grad():
            weight.add_(1e-3)
    elif kind == "sign_flip":
        with torch.no_grad():
            weight[0].neg_()                                           # a change that keeps every norm


@pytest.mark.parametrize("kind", ["requires_grad", "train_mode", "in_optimizer"])
def test_stage2_leakage_error_on_an_unfrozen_planner(stage2_setup, kind) -> None:
    _, _, trainer, _ = stage2_setup
    trainer.assert_planner_frozen()
    trainer.verify_planner_unchanged()
    violate(trainer, kind)
    with pytest.raises(LeakageError):
        trainer.assert_planner_frozen()
    with pytest.raises(LeakageError):
        trainer.verify_planner_unchanged()


def test_stage2_training_loop_enforces_the_invariant(stage2_setup) -> None:
    _, _, trainer, _ = stage2_setup
    violate(trainer, "requires_grad")
    with pytest.raises(LeakageError):
        trainer.train(max_steps=1)


@pytest.mark.parametrize("kind", [
    "weights_changed",
    pytest.param("sign_flip"),
])
def test_stage2_leakage_error_on_changed_planner_weights(stage2_setup, kind) -> None:
    _, _, trainer, _ = stage2_setup
    violate(trainer, kind)
    trainer.assert_planner_frozen()                                    # still frozen; only the weights moved
    with pytest.raises(LeakageError):
        trainer.verify_planner_unchanged()


# =============================================================================================
# Stage 3
# =============================================================================================
def test_stage3_trains_the_planner_at_the_scaled_learning_rate(tmp_path, tokenizer, corpus) -> None:
    joint_steps = 20
    cfg = make_cfg(tokenizer, "training.joint.learning_rate=1e-3", "training.joint.planner_lr_scale=0.25",
                   f"training.joint.max_steps={joint_steps}", "training.joint.warmup_steps=1")
    batches = make_batches(cfg, tokenizer, corpus, 3, 4, corruption=True)
    model = new_model(cfg, tokenizer)
    stage1_target = TargetEncoder.from_online(model.planner.context_encoder)
    with torch.no_grad():
        for p in stage1_target.parameters():
            p.add_(0.01)                                               # an EMA state distinct from f_theta
    trainer = Stage3Trainer(cfg, model, tokenizer, batches, tmp_path, CPU, None, quiet(tmp_path),
                            target_state=stage1_target.state_dict())
    assert all(torch.equal(v, stage1_target.state_dict()[k]) for k, v in trainer.target_encoder.state_dict().items())
    planner_ids = {id(p) for p in model.planner.parameters()}
    assert all(p.requires_grad for p in model.planner.parameters())
    groups = trainer.optimizer.param_groups
    planner_groups = [g for g in groups if g["name"].startswith("planner/")]
    realizer_groups = [g for g in groups if g["name"].startswith("realizer/")]
    assert planner_groups and realizer_groups
    assert {id(p) for g in planner_groups for p in g["params"]} == planner_ids
    planner_before, realizer_before = snapshot(model.planner), snapshot(model.realizer)
    target_before = snapshot(trainer.target_encoder)
    momenta, terms = [], None
    for step, batch in enumerate(batches):
        metrics = trainer.train_step(batch)
        momenta.append(metrics["ema_momentum"])
        terms = metrics
        factor = reference_factor(step + 1, 1, joint_steps)            # lr after step + 1 scheduler steps
        for g in realizer_groups:
            assert g["lr"] == pytest.approx(1e-3 * factor, rel=1e-9)
        for g in planner_groups:                                        # joint.planner_lr_scale x lr
            assert g["lr"] == pytest.approx(0.25 * realizer_groups[0]["lr"], rel=1e-12)
    assert momenta == pytest.approx([M0 + (1 - M0) * k / joint_steps for k in range(3)], abs=1e-12)
    assert changed(planner_before, model.planner) and changed(realizer_before, model.realizer)
    assert changed(target_before, trainer.target_encoder)               # EMA of the fine-tuned f_theta
    for name in ("plan_invariance", "plan_variance_context", "plan_variance_plan", "plan_covariance_context",
                 "plan_covariance_plan", "plan_mode_ce", "loss_real"):
        assert name in terms and math.isfinite(terms[name])             # VICReg kept on (Sec. 5.1, Stage 3)
    assert load_config([]).training.joint.planner_lr_scale == pytest.approx(0.1)   # "one tenth" by default


# =============================================================================================
# load_model
# =============================================================================================
def test_load_model_refuses_stage1_and_rejected_checkpoints(tmp_path, tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer)
    batches = make_batches(cfg, tokenizer, corpus, 2, 4, corruption=True)
    stage1 = Stage1Trainer(cfg, new_planner(cfg), tokenizer, batches, tmp_path / "s1", CPU, quiet(tmp_path))
    stage1.train_step(batches[0])
    stage1_path = stage1.save("stage1.pt")
    with pytest.raises(ValueError, match="realizer"):
        load_model(stage1_path)
    model = new_model(cfg, tokenizer)
    stage2 = Stage2Trainer(cfg, model, tokenizer, batches, tmp_path / "s2", CPU, None, quiet(tmp_path))
    result = stage2.train(max_steps=2)
    loaded = load_model(result.best_checkpoint)                         # positive control
    assert loaded.stage == 2 and loaded.step == 2 and not loaded.model.training
    for (name, p), q in zip(model.state_dict().items(), loaded.model.state_dict().values(), strict=True):
        assert torch.equal(p, q), name
    assert loaded.tokenizer.encode(corpus[0].target) == tokenizer.encode(corpus[0].target)
    payload = load_checkpoint(result.best_checkpoint)
    payload.update(rejected=True, rejection_reason="stable rank below 16")
    rejected = save_checkpoint(tmp_path / "rejected.pt", {k: v for k, v in payload.items() if k != "format"})
    with pytest.raises(ValueError, match="rejected"):
        load_model(rejected)
    payload1 = stage1.payload(rejected=True, reason="collapse")
    rejected1 = save_checkpoint(tmp_path / "stage1_rejected.pt", payload1)
    with pytest.raises(ValueError):
        load_model(rejected1)


# =============================================================================================
# train.py
# =============================================================================================
CLI_TINY = ["planner=tiny", "realizer=tiny", "data=tiny", "data.tokenizer=bpe", "data.bpe_vocab_size=400",
            "training.batch_size=4", "training.planner.warmup_steps=1", "training.planner.max_steps=10",
            "training.realizer.warmup_steps=1", "training.realizer.max_steps=10", "training.joint.warmup_steps=1",
            "training.joint.max_steps=10", "training.save_every=100000"]


def cli(stage: int, out, *extra: str, overrides=CLI_TINY, data: str = "synthetic:48") -> int:
    return train_cli.main(["--stage", str(stage), "--data", data, "--output-dir", str(out), "--device", "cpu",
                           *extra, *overrides])


@pytest.fixture(scope="module")
def cli_runs(tmp_path_factory):
    """Stage 1 (with non-default planner and VICReg settings) and Stage 2 through the CLI."""
    root = tmp_path_factory.mktemp("cli")
    rc1 = cli(1, root / "s1", "--max-steps", "3", overrides=[*CLI_TINY, "vicreg.lambda_u=0.5", "planner.num_slots=12"])
    rc2 = cli(2, root / "s2", "--max-steps", "2", "--planner-checkpoint", str(root / "s1" / "stage1_final.pt"))
    return root, rc1, rc2


def test_train_cli_stage1_returns_0_and_writes_the_final_checkpoint(cli_runs) -> None:
    root, rc1, _ = cli_runs
    assert rc1 == 0
    payload = load_checkpoint(root / "s1" / "stage1_final.pt")
    assert payload["stage"] == 1 and payload["step"] == 3 and payload["rejected"] is False
    assert payload["tokenizer"]["type"] == "hf" and "json" in payload["tokenizer"]          # the trained BPE
    assert payload["config"]["planner"]["vocab_size"] == payload["config"]["realizer"]["vocab_size"]
    rows = [json.loads(line) for line in (root / "s1" / "metrics.jsonl").read_text().splitlines()]
    assert rows and rows[0]["step"] == 1


def test_train_cli_stage2_takes_planner_data_vicreg_and_tokenizer_from_stage1(cli_runs) -> None:
    root, _, rc2 = cli_runs
    assert rc2 == 0
    stage1 = load_checkpoint(root / "s1" / "stage1_final.pt")
    stage2 = load_checkpoint(root / "s2" / "best_stage2.pt")
    assert (root / "s2" / "stage2_final.pt").exists()
    for section in ("planner", "data", "vicreg"):
        assert stage2["config"][section] == stage1["config"][section], section
    assert stage2["config"]["planner"]["num_slots"] == 12 and stage2["config"]["vicreg"]["lambda_u"] == 0.5
    assert stage2["tokenizer"] == stage1["tokenizer"]
    assert all(torch.equal(v, stage1["planner"][k]) for k, v in stage2["planner"].items())   # frozen planner
    assert stage2["source_checkpoint"] == str(root / "s1" / "stage1_final.pt")


def test_train_cli_stage2_and_stage3_need_their_checkpoints(cli_runs) -> None:
    root, _, _ = cli_runs
    with pytest.raises(SystemExit):
        cli(2, root / "missing2", "--max-steps", "1")
    with pytest.raises(SystemExit):
        cli(3, root / "missing3", "--max-steps", "1", "--planner-checkpoint", str(root / "s1" / "stage1_final.pt"))
    with pytest.raises(SystemExit):                                      # a Stage-2 file is not a Stage-1 checkpoint
        cli(2, root / "wrong2", "--max-steps", "1", "--planner-checkpoint", str(root / "s2" / "best_stage2.pt"))


def test_train_cli_stage3_fine_tunes_from_stage2(cli_runs) -> None:
    root, _, _ = cli_runs
    rc = cli(3, root / "s3", "--max-steps", "2", "--planner-checkpoint", str(root / "s1" / "stage1_final.pt"),
             "--init-checkpoint", str(root / "s2" / "best_stage2.pt"))
    assert rc == 0
    payload = load_checkpoint(root / "s3" / "best_stage3.pt")
    assert payload["stage"] == 3 and payload["step"] == 2 and "target_encoder" in payload and "ema" in payload
    stage2 = load_checkpoint(root / "s2" / "best_stage2.pt")
    assert any(not torch.equal(v, stage2["planner"][k]) for k, v in payload["planner"].items())   # planner trained


def test_train_cli_stage2_refuses_a_rejected_stage1_checkpoint_unless_allowed(tmp_path) -> None:
    collapse = [*CLI_TINY, "training.batch_size=8", "vicreg.srank_abort_fraction=1.0",
                "training.collapse_check_every=1", "training.collapse_grace_steps=0"]
    # >= 8 plan rows per step, so a window of more than d = 32 rows is checked (and aborts) by step 5 at the latest
    rc = cli(1, tmp_path / "s1", "--max-steps", "8", overrides=collapse)
    rejected = tmp_path / "s1" / "stage1_rejected.pt"
    assert rc == 2 and rejected.exists() and not (tmp_path / "s1" / "stage1_final.pt").exists()
    assert load_checkpoint(rejected)["rejected"] is True
    with pytest.raises(SystemExit, match="rejected"):
        cli(2, tmp_path / "s2", "--max-steps", "1", "--planner-checkpoint", str(rejected))
    assert not (tmp_path / "s2" / "stage2_final.pt").exists()
    rc = cli(2, tmp_path / "s2", "--max-steps", "1", "--planner-checkpoint", str(rejected), "--allow-rejected")
    assert rc == 0 and (tmp_path / "s2" / "best_stage2.pt").exists()


def test_train_cli_documented_stage2_command(cli_runs, tmp_path) -> None:
    root, _, _ = cli_runs
    rc = train_cli.main(["--stage", "2", "--planner-checkpoint", str(root / "s1" / "stage1_final.pt"),
                         "--data", "synthetic:48", "--output-dir", str(tmp_path / "s2"), "--max-steps", "1",
                         "--device", "cpu", "realizer=tiny", "training.batch_size=4"])
    assert rc == 0


def test_train_cli_max_steps_overrides_the_stage_max_steps(tmp_path) -> None:
    rc = cli(1, tmp_path / "s1", "--max-steps", "3", overrides=[*CLI_TINY, "training.planner.max_steps=50"])
    assert rc == 0
    payload = load_checkpoint(tmp_path / "s1" / "stage1_final.pt")
    assert payload["step"] == 3
    assert payload["ema"]["total_steps"] == 3
    assert payload["config"]["training"]["planner"]["max_steps"] == 3
