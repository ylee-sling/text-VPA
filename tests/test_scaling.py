# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
Tests of the features for training on one memory-limited GPU (no change to the paper's mathematics):

    gradient accumulation   one optimizer step over k micro-batches. Each micro-batch loss is scaled by 1/k,
                            so for losses that are means over equally sized micro-batches (Stage 2: Eq. 12
                            averaged over B N, and the readout terms) the accumulated gradient equals the
                            gradient of one batch made of all of them.
    activation checkpoints  recomputing realizer blocks in the backward pass leaves the loss and every
                            gradient unchanged.
    compact examples        int32 arrays instead of Python lists; the list views and the collated batches are
                            unchanged; a parallel build yields the same examples in the same order; a cached
                            build is reloaded instead of rebuilt.

References: the same computation without the feature (one big batch, no checkpointing, a single-process
build), never the feature itself.
"""

from __future__ import annotations

import pickle
from dataclasses import fields

import pytest
import torch

from tests.conftest import ATOL, RTOL
from text_vpa.configs import load_config
from text_vpa.data.dataset import Batch, PRJEPACollator, PRJEPADataset
from text_vpa.data.synthetic import synthetic_corpus
from text_vpa.data.tokenization import train_bpe_tokenizer
from text_vpa.engine.runtime import (
    DistributedContext,
    MetricLogger,
    grouped,
    maybe_no_sync,
    set_activation_checkpointing,
)
from text_vpa.engine.trainer_stage1 import Stage1Trainer
from text_vpa.engine.trainer_stage2 import Stage2Trainer, Stage3Trainer
from text_vpa.models.text_vpa_model import TextVPAModel, TextVPAPlanner

CPU = DistributedContext(device=torch.device("cpu"))
BASE = [
    "planner=tiny", "data=tiny", "realizer=tiny", "data.tokenizer=bpe", "training.batch_size=4",
    "training.planner.warmup_steps=1", "training.planner.max_steps=40",
    "training.realizer.warmup_steps=1", "training.realizer.max_steps=40",
    "training.joint.warmup_steps=1", "training.joint.max_steps=20",
    "training.log_every=1000", "training.save_every=100000", "bridge.plan_dropout=0.0",
]


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus(64, seed=5)


@pytest.fixture(scope="module")
def tokenizer(corpus):
    return train_bpe_tokenizer([r.target for r in corpus] + [r.context for r in corpus], vocab_size=400)


def make_cfg(tokenizer, *extra: str):
    return load_config([*BASE, *extra]).with_vocab_size(tokenizer.vocab_size)


def make_dataset(cfg, tokenizer, records, **kwargs) -> PRJEPADataset:
    return PRJEPADataset(records, tokenizer, cfg.planner.num_slots, cfg.data.max_context_tokens,
                         cfg.data.max_output_tokens, cfg.data.max_structure_tokens, cfg.readout.max_text_tokens,
                         cfg.data.mode_scheme, **kwargs)


def big_batch(cfg, tokenizer, records, size: int, seed: int = 0) -> Batch:
    """One collated batch with the FS-DFM state drawn once (so every split sees the same w_t, t, h)."""
    from text_vpa.data.dataset import MixtureCorruption

    dataset = make_dataset(cfg, tokenizer, records)
    mixture = MixtureCorruption(tokenizer.vocab_size, cfg.realizer.step_budgets, cfg.realizer.schedule,
                                cfg.realizer.source, tokenizer.mask_id)
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens, mixture)
    torch.manual_seed(seed)
    return collator([dataset[i] for i in range(size)])


def split(batch: Batch, parts: int) -> list[Batch]:
    """Equal row slices of every tensor field (same padded shapes as the big batch)."""
    b = batch.batch_size
    assert b % parts == 0
    step = b // parts
    out = []
    for k in range(parts):
        values = {}
        for f in fields(Batch):
            v = getattr(batch, f.name)
            values[f.name] = v[k * step:(k + 1) * step] if isinstance(v, torch.Tensor) else v
        out.append(Batch(**values))
    return out


def grads(module) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}


def assert_same_grads(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys() and a
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=RTOL, atol=ATOL, msg=name)


# =============================================================================================
# Helpers
# =============================================================================================
def test_grouped_yields_full_groups_then_the_remainder() -> None:
    assert list(grouped(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(grouped(range(4), 1)) == [[0], [1], [2], [3]]
    assert list(grouped([], 2)) == []
    with pytest.raises(ValueError):
        list(grouped(range(3), 0))


def test_maybe_no_sync_is_a_no_op_without_ddp() -> None:
    module = torch.nn.Linear(2, 2)
    with maybe_no_sync(module, skip_sync=True):
        module(torch.ones(1, 2)).sum().backward()
    assert module.weight.grad is not None


def test_accumulation_settings_are_validated() -> None:
    cfg = load_config(["training.realizer.grad_accum_steps=4", "training.activation_checkpointing=true"])
    assert cfg.training.realizer.grad_accum_steps == 4 and cfg.training.planner.grad_accum_steps == 1
    assert cfg.training.activation_checkpointing
    with pytest.raises(ValueError):
        load_config(["training.realizer.grad_accum_steps=0"])


# =============================================================================================
# Gradient accumulation
# =============================================================================================
@pytest.mark.parametrize("parts", [2, 4])
def test_stage2_accumulated_gradient_equals_the_big_batch_gradient(tmp_path, tokenizer, corpus, parts) -> None:
    """Eq. 12 is a mean over B N, so k equal micro-batches scaled by 1/k give the big batch's realizer gradient.

    Two documented differences are switched off: clipping (the readout's text term is a mean over valid text
    tokens, whose count differs between micro-batches, so through the global norm it would rescale every
    gradient) and the spectral-norm power iteration on W_K, W_V, which advances once per training forward,
    i.e. k times per optimizer step with k micro-batches."""
    cfg = make_cfg(tokenizer, "training.batch_size=8", "training.grad_clip_norm=1e9")
    batch = big_batch(cfg, tokenizer, corpus, 8)
    reference: dict[str, dict[str, torch.Tensor]] = {}
    params: dict[str, dict[str, torch.Tensor]] = {}
    for name, step_input in (("big", batch), ("accumulated", split(batch, parts))):
        torch.manual_seed(0)
        model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
        frozen = [m for m in model.modules() if hasattr(m, "n_power_iterations")]
        assert len(frozen) == 2 * cfg.realizer.num_layers               # W_K^l and W_V^l of the bridge
        for m in frozen:
            m.n_power_iterations = 0
        trainer = Stage2Trainer(cfg, model, tokenizer, [batch], tmp_path / name, CPU, None,
                                MetricLogger(tmp_path, enabled=False))
        metrics = trainer.train_step(step_input)
        reference[name] = grads(model.realizer)
        params[name] = {n: p.detach().clone() for n, p in model.realizer.named_parameters()}
        assert trainer.step == 1 and metrics["loss_real"] > 0
    assert_same_grads(reference["big"], reference["accumulated"])
    for n in params["big"]:
        torch.testing.assert_close(params["big"][n], params["accumulated"][n], rtol=1e-4, atol=1e-5, msg=n)


def test_stage1_accumulation_consumes_k_micro_batches_per_optimizer_step(tmp_path, tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer, "training.planner.grad_accum_steps=3")
    dataset = make_dataset(cfg, tokenizer, corpus)
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens)
    batches = [collator([dataset[i] for i in range(j * 4, (j + 1) * 4)]) for j in range(7)]
    torch.manual_seed(0)
    planner = TextVPAPlanner.from_config(cfg.planner)
    trainer = Stage1Trainer(cfg, planner, tokenizer, batches, tmp_path, CPU, MetricLogger(tmp_path, enabled=False))
    calls: list[int] = []
    original = trainer.train_step

    def counting(group):
        calls.append(len(group))
        return original(group)

    trainer.train_step = counting  # type: ignore[method-assign]
    result = trainer.train(max_steps=4)
    # 7 batches per epoch: groups 3, 3, 1 (epoch end), then 3 from the next epoch
    assert calls == [3, 3, 1, 3] and result.steps == 4
    assert trainer.epoch == 1 and trainer.batch_in_epoch == 3


def test_stage3_accumulation_runs_and_counts_optimizer_steps(tmp_path, tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer, "training.joint.grad_accum_steps=2")
    batch = big_batch(cfg, tokenizer, corpus, 8)
    torch.manual_seed(0)
    model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
    trainer = Stage3Trainer(cfg, model, tokenizer, split(batch, 2) * 2, tmp_path, CPU, None,
                            MetricLogger(tmp_path, enabled=False), target_state=None)
    result = trainer.train(max_steps=2)
    assert result.steps == 2 and trainer.batch_in_epoch == 4 and trainer.epoch == 0   # 2 x 2 micro-batches


# =============================================================================================
# Activation checkpointing
# =============================================================================================
@pytest.mark.parametrize("kind", ["fs_dfm", "logit_kl"])
def test_activation_checkpointing_leaves_loss_and_gradients_unchanged(tmp_path, tokenizer, corpus, kind) -> None:
    cfg = make_cfg(tokenizer, f"realizer.kind={kind}", "training.batch_size=8")
    batch = big_batch(cfg, tokenizer, corpus, 8)
    out = {}
    for flag in (False, True):
        torch.manual_seed(0)
        model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
        assert set_activation_checkpointing(model.realizer, flag) == 1
        trainer = Stage2Trainer(cfg.model_copy(update={"training": cfg.training.model_copy(
            update={"activation_checkpointing": flag})}), model, tokenizer, [batch], tmp_path / str(flag), CPU,
            None, MetricLogger(tmp_path, enabled=False))
        torch.manual_seed(1)                                     # same Logit-KL training state draw
        metrics = trainer.train_step(batch)
        assert model.realizer.backbone.activation_checkpointing is flag
        out[flag] = (metrics["loss_real"], grads(model.realizer))
    assert out[True][0] == pytest.approx(out[False][0], rel=1e-6)
    assert_same_grads(out[False][1], out[True][1])


def test_activation_checkpointing_is_inactive_in_eval_mode(tokenizer) -> None:
    cfg = make_cfg(tokenizer)
    model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id).eval()
    set_activation_checkpointing(model.realizer, True)
    calls = {"n": 0}
    import text_vpa.models.realizers.backbone as backbone_module

    original = backbone_module.checkpoint

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    backbone_module.checkpoint = counting
    try:
        context = torch.randint(0, 200, (2, 10))
        plan = model.planner.plan(context, None)
        cache = model.realizer.prepare(plan)
        w = torch.randint(0, tokenizer.vocab_size, (2, 16))
        model.realizer(w, torch.zeros(2), torch.full((2,), 0.25), cache)
    finally:
        backbone_module.checkpoint = original
    assert calls["n"] == 0


# =============================================================================================
# Compact examples, parallel build, cache
# =============================================================================================
def test_compact_examples_keep_the_list_views_and_are_small(tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer, "planner.num_slots=64", "readout.max_text_tokens=48")
    dataset = make_dataset(cfg, tokenizer, corpus[:8])
    for e in dataset.examples:
        dense = e.slot_texts
        assert len(dense) == 64 and all(len(row) == 48 for row in dense)
        for m, row in enumerate(dense):
            if m not in e.text_slots.tolist():
                assert row == [tokenizer.pad_id] * 48
        assert e.target == e.target_ids.tolist() and len(e.slot_index) == len(e.structure)
        # A dense Python layout would hold 64 x 48 slot-text entries alone (8 bytes per reference).
        assert e.nbytes < 64 * 48 * 8 / 4
    restored = pickle.loads(pickle.dumps(dataset.examples))
    assert [r.slot_texts for r in restored] == [e.slot_texts for e in dataset.examples]


def test_collated_batch_matches_the_dense_layout(tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer)
    dataset = make_dataset(cfg, tokenizer, corpus[:4])
    collator = PRJEPACollator(tokenizer, cfg.planner.num_slots, cfg.readout.max_text_tokens, cfg.data.bucket_size,
                              cfg.data.max_output_tokens)
    batch = collator(dataset.examples)
    for i, e in enumerate(dataset.examples):
        assert batch.slot_texts[i].tolist() == e.slot_texts
        assert batch.slot_kinds[i].tolist() == e.slot_kinds
        assert batch.context[i, : len(e.context)].tolist() == e.context
        assert batch.target[i, : len(e.target)].tolist() == e.target
        assert batch.structure[i, : len(e.structure)].tolist() == e.structure
        assert batch.slot_index[i, : len(e.slot_index)].tolist() == e.slot_index
        assert sorted(torch.nonzero(batch.inventory[i]).flatten().tolist()) == e.inventory


def same_examples(a: PRJEPADataset, b: PRJEPADataset) -> bool:
    keys = ("context", "target", "structure", "slot_index", "mode", "slot_kinds", "slot_texts", "inventory",
            "num_slots", "task_id")
    return len(a) == len(b) and all(getattr(x, k) == getattr(y, k) for x, y in zip(a.examples, b.examples,
                                                                                    strict=True) for k in keys)


def test_parallel_build_gives_the_same_examples_in_the_same_order(tokenizer, corpus) -> None:
    cfg = make_cfg(tokenizer)
    serial = make_dataset(cfg, tokenizer, corpus)
    parallel = make_dataset(cfg, tokenizer, corpus, num_proc=3)
    assert same_examples(serial, parallel) and serial.skipped == parallel.skipped


def test_cached_build_is_reloaded_not_rebuilt(tmp_path, tokenizer, corpus, monkeypatch) -> None:
    cfg = make_cfg(tokenizer)
    args = (tokenizer, cfg.planner.num_slots, cfg.data.max_context_tokens, cfg.data.max_output_tokens,
            cfg.data.max_structure_tokens, cfg.readout.max_text_tokens, cfg.data.mode_scheme)
    first = PRJEPADataset.cached(tmp_path, corpus, *args)
    assert len(list(tmp_path.glob("dataset-*.pkl"))) == 1

    def forbidden(*_a, **_k):
        raise AssertionError("rebuilt instead of loading the cache")

    monkeypatch.setattr(PRJEPADataset, "__init__", forbidden)
    second = PRJEPADataset.cached(tmp_path, corpus, *args)
    assert same_examples(first, second) and second.canvas_lengths(16) == first.canvas_lengths(16)
    monkeypatch.undo()
    other = PRJEPADataset.cached(tmp_path, corpus[:10], *args)              # different records: a new entry
    assert len(other) <= 10 and len(list(tmp_path.glob("dataset-*.pkl"))) == 2


# =============================================================================================
# Memory probe
# =============================================================================================
def test_worst_case_batch_has_maximal_lengths_and_bounded_slot_texts() -> None:
    from text_vpa.utils.memory_probe import worst_case_batch

    cfg = load_config(["planner=tiny", "realizer=tiny", "data=tiny"])
    batch = worst_case_batch(cfg, 3, torch.Generator().manual_seed(0))
    assert batch.context.shape == (3, cfg.data.max_context_tokens)
    assert batch.target.shape == (3, cfg.data.max_output_tokens) and batch.w_t.shape == batch.target.shape
    assert batch.structure.shape == batch.slot_index.shape == (3, cfg.data.max_structure_tokens)
    assert int(batch.slot_index.max()) == cfg.planner.num_slots - 1
    text_slots = int(batch.slot_text_mask[0].sum())
    assert text_slots * cfg.readout.max_text_tokens <= max(cfg.data.max_output_tokens, cfg.readout.max_text_tokens)
    assert int(batch.target.max()) < cfg.realizer.vocab_size and int(batch.modes.max()) < cfg.planner.num_modes


def test_memory_probe_runs_both_stages_with_accumulation(capsys) -> None:
    from text_vpa.utils import memory_probe

    rc = memory_probe.main(["--device", "cpu", "--steps", "1", "--warmup", "0", "planner=tiny", "realizer=tiny",
                            "data=tiny", "training.batch_size=2", "training.realizer.grad_accum_steps=2",
                            "training.activation_checkpointing=true"])
    report = __import__("json").loads(capsys.readouterr().out)
    assert rc == 0 and [s["stage"] for s in report["stages"]] == [1, 2]
    stage2 = report["stages"][1]
    assert stage2["target_tokens_per_step"] == 2 * 2 * 64 and stage2["seconds_per_step"] > 0
    assert stage2["trainable_M"] < stage2["parameters_M"]                     # the planner is frozen
