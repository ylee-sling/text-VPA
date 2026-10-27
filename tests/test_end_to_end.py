# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
End-to-end smoke test of the staged pipeline (Sec. 5.1) and Algorithm 1:

    1. synthetic dataset (spec -> program pairs, BPE tokenizer trained on them)
    2. 10 steps of Stage 1 (L_plan, EMA target encoder, collapse monitor)
    3. frozen planner, 10 steps of Stage 2 (realizer + readout on the predicted plan)
    4. PRJEPASampler with K = 4
    5. non-collapsed output, and syntax the sampler guarantees

Ten steps cannot teach a model to write programs, so "valid syntax" is asserted as what the constraint
energies guarantee regardless of training: no output closes a bracket it did not open (checked by an
independent scanner below, not by the library's lexer), no control token appears, and -- under an oracle
plan -- every identifier-bearing token is licensed by the plan. The ``slow`` test (run with
``pytest -m slow``) trains longer on two programs and checks real ``ast.parse`` validity and that each
context's plan selects its own program.
"""

from __future__ import annotations

import ast

import pytest
import torch

import train as train_script
from text_vpa.configs import load_config
from text_vpa.data.ast_parser import TreeSitterASTExtractor
from text_vpa.data.synthetic import synthetic_corpus
from text_vpa.data.tokenization import SPECIAL_TOKENS, train_bpe_tokenizer
from text_vpa.engine.losses import calculate_stable_rank
from text_vpa.engine.runtime import DistributedContext, MetricLogger
from text_vpa.engine.trainer_stage1 import Stage1Trainer
from text_vpa.engine.trainer_stage2 import Stage2Trainer
from text_vpa.inference.energy_terms import PlanConstraints, base_licensed_words
from text_vpa.inference.loading import LoadedModel, build_sampler
from text_vpa.models.encoders import TargetEncoder
from text_vpa.models.text_vpa_model import TextVPAModel, TextVPAPlanner

OVERRIDES = [
    "planner=tiny", "realizer=tiny", "data=tiny", "data.tokenizer=bpe", "training.batch_size=8",
    "training.collapse_check_every=5", "training.collapse_grace_steps=1000", "training.log_every=5",
    "training.save_every=100000", "training.planner.warmup_steps=2", "training.realizer.warmup_steps=2",
]


def bracket_violations(text: str) -> int:
    """Closers outside strings and comments that do not match the innermost open bracket (independent scanner)."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    bad = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch in "'\"":
            quote = text[i:i + 3] if text[i:i + 3] in ("'''", '"""') else ch
            j = i + len(quote)
            while j < n and not text.startswith(quote, j):
                if text[j] == "\\":
                    j += 1
                elif text[j] == "\n" and len(quote) == 1:
                    break                                # unterminated single-line string ends at the newline
                j += 1
            i = j + (len(quote) if text.startswith(quote, j) else 0)
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
            else:
                bad += 1
        i += 1
    return bad


def test_bracket_scanner_itself() -> None:
    assert bracket_violations("f(a[1], {'x': ')'})  # ) in a comment") == 0
    assert bracket_violations('s = """ ) ] """\nx = (1]') == 1
    assert bracket_violations(")(") == 1


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict:
    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("e2e")
    records = synthetic_corpus(96, seed=0)                                          # 1. synthetic dataset
    tokenizer = train_bpe_tokenizer([r.target for r in records] + [r.context for r in records], vocab_size=600)
    cfg = load_config(OVERRIDES).with_vocab_size(tokenizer.vocab_size)
    ctx = DistributedContext(device=torch.device("cpu"))
    logger = MetricLogger(out, enabled=True)

    loader1 = train_script.build_loader(cfg, records, tokenizer, ctx, corruption=False, shuffle=True, drop_last=True)
    planner = TextVPAPlanner.from_config(cfg.planner)
    stage1 = Stage1Trainer(cfg, planner, tokenizer, loader1, out / "s1", ctx, logger)
    result1 = stage1.train(max_steps=10)                                            # 2. Stage 1

    model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
    model.planner.load_state_dict(planner.state_dict())
    planner_before = {k: v.clone() for k, v in model.planner.state_dict().items()}
    realizer_before = {k: v.clone() for k, v in model.realizer.state_dict().items()}
    loader2 = train_script.build_loader(cfg, records, tokenizer, ctx, corruption=True, shuffle=True, drop_last=True)
    stage2 = Stage2Trainer(cfg, model, tokenizer, loader2, out / "s2", ctx, None, logger)

    target_calls = {"n": 0}
    original_forward = TargetEncoder.forward

    def counting_forward(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        target_calls["n"] += 1
        return original_forward(self, *args, **kwargs)

    TargetEncoder.forward = counting_forward  # type: ignore[method-assign]
    try:
        result2 = stage2.train(max_steps=10)                                        # 3. frozen planner, Stage 2
    finally:
        TargetEncoder.forward = original_forward  # type: ignore[method-assign]

    loaded = LoadedModel(cfg, tokenizer, model.eval(), 2, result2.steps)
    sampler = build_sampler(loaded)
    contexts = [tokenizer.encode(r.context) for r in records[:6]]
    width = max(map(len, contexts))
    context = torch.full((len(contexts), width), tokenizer.pad_id, dtype=torch.long)
    mask = torch.zeros(len(contexts), width, dtype=torch.bool)
    for i, ids in enumerate(contexts):
        context[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = True
    sample = sampler.sample(context, mask, length=48, num_steps=4,                 # 4. Algorithm 1, K = 4
                            generator=torch.Generator().manual_seed(0))
    extractor = TreeSitterASTExtractor("python")
    oracle = [PlanConstraints.from_reference(r.target, tokenizer, extractor.parse(r.target), sampler.energy.analysis)
              for r in records[:6]]
    oracle_sample = sampler.sample(context, mask, length=48, num_steps=4, plans=oracle,
                                   generator=torch.Generator().manual_seed(1))
    return dict(cfg=cfg, tokenizer=tokenizer, result1=result1, stage1=stage1, result2=result2, model=model,
                planner_before=planner_before, realizer_before=realizer_before, target_calls=target_calls,
                sample=sample, oracle=oracle, oracle_sample=oracle_sample, sampler=sampler, context=context,
                mask=mask, out=out)


# ---------------------------------------------------------------------------------------------
def test_stage1_runs_ten_steps_without_collapse_abort(pipeline: dict) -> None:
    r = pipeline["result1"]
    assert r.steps == 10 and not r.rejected and r.checkpoint is not None and r.checkpoint.exists()
    report = pipeline["stage1"].last_report
    for stream in ("context", "plan"):
        srank = report[f"{stream}/stable_rank"]
        assert torch.isfinite(torch.tensor(srank)) and srank > 1.0      # no rank-one collapse (Theorem 1)
    assert 0.996 <= pipeline["stage1"].ema.momentum() <= 1.0


def test_stage2_keeps_the_planner_frozen_and_never_uses_the_target_encoder(pipeline: dict) -> None:
    model = pipeline["model"]
    for name, value in model.planner.state_dict().items():
        assert torch.equal(value, pipeline["planner_before"][name]), name
    assert pipeline["target_calls"]["n"] == 0
    changed = any(not torch.equal(v, pipeline["realizer_before"][k]) for k, v in model.realizer.state_dict().items())
    assert changed
    assert pipeline["result2"].steps == 10 and pipeline["result2"].checkpoint.exists()


def test_sampler_depth_is_k_plus_3(pipeline: dict) -> None:
    for sample in (pipeline["sample"], pipeline["oracle_sample"]):
        assert sample.depth == 7
        counted = [e for e in sample.events if e in ("f_theta", "pi_phi", "g_phi", "F_psi")]
        assert counted == ["f_theta", "pi_phi", "g_phi"] + ["F_psi"] * 4
        assert sample.events.index("bridge") < sample.events.index("F_psi")


def test_output_is_not_collapsed(pipeline: dict) -> None:
    sample = pipeline["sample"]
    assert sample.tokens.shape == (6, 48)
    assert torch.isfinite(sample.sigma_bar).all() and (sample.sigma_bar >= 0).all()
    assert sample.tokens.unique().numel() > 1
    with torch.no_grad():
        plan = pipeline["model"].planner.plan(pipeline["context"], pipeline["mask"],
                                              mode=torch.zeros(6, dtype=torch.long))
    rows = plan.z_target.reshape(-1, plan.z_target.shape[-1])
    assert float(calculate_stable_rank(rows)) > 1.0                    # the plan is neither a point nor a line


def test_outputs_satisfy_the_guaranteed_syntax(pipeline: dict) -> None:
    for sample in (pipeline["sample"], pipeline["oracle_sample"]):
        assert sample.hard_violations == 0
        for text in sample.texts:
            assert bracket_violations(text) == 0, text
            assert not any(t in text for t in SPECIAL_TOKENS if t != "[PAD]")


def test_oracle_plan_licenses_every_identifier_token(pipeline: dict) -> None:
    tokenizer = pipeline["tokenizer"]
    analysis = pipeline["sampler"].energy.analysis
    base = analysis.licensed_ids_for_words(tokenizer, base_licensed_words("python"))
    sample = pipeline["oracle_sample"]
    for row, plan in zip(sample.tokens.tolist(), pipeline["oracle"], strict=True):
        for token in row:
            if token == tokenizer.pad_id:
                break
            if analysis.word_bearing[token]:
                assert token in plan.licensed_tokens or token in base


# ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_memorised_programs_are_realized_from_their_own_plans(tmp_path) -> None:
    """Longer training on two programs, both on the same canvas length (``data.bucket_size`` = the canvas), so
    the realizer can tell them apart only through its conditioning (Z_target, s_x, e(u)). Every sample of each
    context must be valid Python, most must reproduce that context's program exactly, and none may reproduce
    the other program.

    With length buckets the canvas length alone identifies each of the two programs, and a canvas length that
    no training example had breaks that shortcut; the sampler's default takes N from the readout's length head.
    """
    torch.manual_seed(0)
    base = synthetic_corpus(8, seed=3)
    programs = [base[1], base[4]]                                   # gcd (while loop) and z * x + y
    records = programs * 8
    tokenizer = train_bpe_tokenizer([r.target for r in records] * 4 + [r.context for r in records] * 4, vocab_size=700)
    canvas = 64
    cfg = load_config(OVERRIDES + ["realizer.hidden_dim=128", "realizer.num_layers=4", "training.log_every=1000",
                                   "training.planner.learning_rate=1e-3", "training.realizer.learning_rate=1e-3",
                                   "training.planner.warmup_steps=20", "training.realizer.warmup_steps=50",
                                   "training.planner.max_steps=100", "training.realizer.max_steps=1200",
                                   f"data.max_output_tokens={canvas}", f"data.bucket_size={canvas}"]
                      ).with_vocab_size(tokenizer.vocab_size)
    ctx = DistributedContext(device=torch.device("cpu"))
    logger = MetricLogger(tmp_path, enabled=False)
    planner = TextVPAPlanner.from_config(cfg.planner)
    Stage1Trainer(cfg, planner, tokenizer, train_script.build_loader(cfg, records, tokenizer, ctx, False, True, True),
                  tmp_path, ctx, logger).train(100)
    model = TextVPAModel.from_config(cfg, mask_id=tokenizer.mask_id)
    model.planner.load_state_dict(planner.state_dict())
    loader = train_script.build_loader(cfg, records, tokenizer, ctx, True, True, True)
    assert {int(b.target.shape[1]) for b in loader} == {canvas}         # one canvas length for both programs
    Stage2Trainer(cfg, model, tokenizer, loader, tmp_path, ctx, None, logger).train(1200)
    sampler = build_sampler(LoadedModel(cfg, tokenizer, model.eval(), 2, 1200))
    parsed, exact = 0, [0, 0]
    for i, record in enumerate(records):
        own, other = programs.index(record), 1 - programs.index(record)
        text = sampler.sample(torch.tensor([tokenizer.encode(record.context)]), length=canvas, num_steps=8,
                              generator=torch.Generator().manual_seed(i)).texts[0]
        try:
            ast.parse(text)
            parsed += 1
        except SyntaxError:
            pass
        exact[own] += text.strip() == programs[own].target.strip()
        assert text.strip() != programs[other].target.strip(), (i, text)
    assert parsed >= len(records) - 2, parsed
    assert min(exact) >= 6, exact
