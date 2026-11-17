# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Stage 0 (plan-free Gemma 4 realizer): the backbone wrapper against transformers' own Gemma 4 implementation,
the attention masks against explicit loops, the sampler against the paper's jump kernel, data, truncation,
training and the H2 report. All on a tiny random Gemma 4 text model; nothing is downloaded."""

from __future__ import annotations

import copy
import gzip
import json
import math
import random

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForCausalLM"):                    # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from text_vpa.evaluation.tasks import EvalTask  # noqa: E402
from text_vpa.models.realizers.kernels import kernel_distribution, tilted_jump_kernel  # noqa: E402
from text_vpa.models.realizers.paths import KappaSchedule  # noqa: E402
from text_vpa.stage0 import data as s0data  # noqa: E402
from text_vpa.stage0.autoregressive import cut_at_eos, generate_autoregressive  # noqa: E402
from text_vpa.stage0.backbone import (  # noqa: E402
    RealizerSpec,
    Stage0Realizer,
    canvas_masks,
    left_pad,
    prefill_masks,
)
from text_vpa.stage0.evaluate import realizer_samples, reference_text, score  # noqa: E402
from text_vpa.stage0.postprocess import (  # noqa: E402
    build_sample,
    extract_fenced,
    is_complete,
    parses,
    truncate_complete,
    truncate_write,
)
from text_vpa.stage0.report import bootstrap_ratio, h2_verdict  # noqa: E402
from text_vpa.stage0.sampler import SamplerConfig, draw_tokens, realize, reveal_counts  # noqa: E402
from text_vpa.stage0.train import Stage0Config, Stage0Trainer, learning_rate  # noqa: E402

VOCAB, WINDOW, MASK = 320, 6, 4
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
               "sliding_attention", "full_attention", "full_attention"]


def tiny_backbone(seed: int = 0):
    """A 7-block Gemma 4 text model with every feature of E2B that matters here: local and global blocks with
    different head sizes, a 6-token window, per-layer embeddings, KV sharing in the upper 3 blocks (which are
    double-wide), and the final logit soft cap."""
    torch.manual_seed(seed)
    cfg = transformers.Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=7, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=WINDOW, layer_types=list(LAYER_TYPES),
        final_logit_softcapping=30.0, vocab_size_per_layer_input=VOCAB, hidden_size_per_layer_input=8,
        num_kv_shared_layers=3, use_double_wide_mlp=True, max_position_embeddings=512)
    return transformers.Gemma4ForCausalLM(cfg).eval()


def tiny_realizer(seed: int = 0, **spec):
    model = tiny_backbone(seed)
    reference = copy.deepcopy(model)
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec(mask_id=MASK, lora_rank=4, lora_alpha=4.0,
                                                                        time_dim=32, **spec))
    return realizer, reference


def randomize_adapters(realizer: Stage0Realizer, scale: float = 0.05) -> None:
    with torch.no_grad():
        for name, p in realizer.named_parameters():
            if p.requires_grad and ("lora_b" in name or "mod_" in name):
                p.normal_(0.0, scale)


class CharTokenizer:
    """Characters as tokens: ids 0-4 are <pad>, <eos>, <bos>, <unk>, <mask>; a printable character c is
    ord(c) - 32 + 6 and the newline is 5."""

    bos_id, eos_id, pad_id, mask_id, vocab_size = 2, 1, 0, MASK, VOCAB

    def encode_batch(self, texts):
        return [[5 if c == "\n" else min(ord(c), 126) - 32 + 6 for c in text] for text in texts]

    def decode(self, ids):
        return "".join("\n" if i == 5 else chr(i - 6 + 32) for i in ids if i >= 5)


# =============================================================================================
# The backbone wrapper
# =============================================================================================
def test_causal_canvas_reproduces_the_backbone():
    """Prefill + one canvas pass with the backbone's masks = transformers' own forward pass (adapters at their
    initial values), for prefixes of different lengths in one left-padded batch."""
    realizer, reference = tiny_realizer()
    ids = torch.randint(5, VOCAB, (2, 24))
    starts, prefix_len = [0, 4], 11                                   # row 1 drops its first 4 tokens
    rows = [ids[b, starts[b]:prefix_len].tolist() for b in range(2)]
    canvas = ids[:, prefix_len:]
    with torch.no_grad():
        cache = realizer.prefill(*left_pad(rows, 0))
        hidden = realizer.canvas_hidden(cache, canvas, torch.zeros_like(canvas, dtype=torch.bool),
                                        torch.zeros(2), torch.ones(2), mode="causal")
        got = realizer.logits(hidden)
        for b in range(2):
            want = reference(ids[b:b + 1, starts[b]:]).logits[0, prefix_len - starts[b]:]
            torch.testing.assert_close(got[b], want, rtol=1e-4, atol=1e-5)
    assert realizer.passes == {"prefill": 1, "canvas": 1}


def _loop_masks(prefix_len: int, canvas_len: int, window: int, mode: str) -> dict[str, torch.Tensor]:
    """The realizer's attention pattern over [prefix; canvas], written out position by position."""
    total = prefix_len + canvas_len
    full = torch.zeros(total, total, dtype=torch.bool)
    sliding = torch.zeros(total, total, dtype=torch.bool)
    for q in range(total):
        for k in range(total):
            if q < prefix_len:                                        # a prompt row: the backbone, causal
                see = k <= q
            elif k < prefix_len:                                      # canvas row, prompt key
                see = True
            else:                                                     # canvas row, canvas key
                see = True if mode == "bidirectional" else k <= q
            full[q, k] = see
            sliding[q, k] = see and abs(q - k) < window
    return {"full_attention": full[None, None], "sliding_attention": sliding[None, None]}


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_canvas_pass_matches_a_single_masked_forward(mode):
    """The two-pass computation (cached prompt + canvas) equals one forward pass of the unmodified model over
    [prompt; canvas] under the explicit mask; and under that mask the prompt rows are the causal backbone's."""
    realizer, reference = tiny_realizer()
    prefix_len, canvas_len = 10, 14                                   # both longer than the 6-token window
    ids = torch.randint(5, VOCAB, (1, prefix_len + canvas_len))
    masks = _loop_masks(prefix_len, canvas_len, WINDOW, mode)
    with torch.no_grad():
        want = reference(ids, attention_mask=masks).logits
        cache = realizer.prefill(*left_pad([ids[0, :prefix_len].tolist()], 0))
        canvas = ids[:, prefix_len:]
        hidden = realizer.canvas_hidden(cache, canvas, torch.zeros_like(canvas, dtype=torch.bool),
                                        torch.zeros(1), torch.ones(1), mode=mode)
        torch.testing.assert_close(realizer.logits(hidden), want[:, prefix_len:], rtol=1e-4, atol=1e-5)
        causal = reference(ids).logits
    torch.testing.assert_close(want[:, :prefix_len], causal[:, :prefix_len], rtol=1e-4, atol=1e-5)
    if mode == "bidirectional":
        assert not torch.allclose(want[:, prefix_len:-1], causal[:, prefix_len:-1], atol=1e-3)


def test_mask_builders_against_loops():
    prefix_len, canvas_len = 9, 11
    pos, valid = torch.arange(prefix_len)[None], torch.ones(1, prefix_len, dtype=torch.bool)
    canvas_pos = prefix_len + torch.arange(canvas_len)[None]
    for mode in ("bidirectional", "causal"):
        want = _loop_masks(prefix_len, canvas_len, WINDOW, mode)
        got_prefix = prefill_masks(pos, valid, WINDOW)
        got_canvas = canvas_masks(pos, valid, canvas_pos, WINDOW, mode)
        for kind in ("full_attention", "sliding_attention"):
            assert torch.equal(got_prefix[kind], want[kind][:, :, :prefix_len, :prefix_len])
            assert torch.equal(got_canvas[kind], want[kind][:, :, prefix_len:, :])
    # Left padding: padded keys are invisible, a padded query sees only itself, and distances use positions.
    ids, valid, pos = left_pad([[7, 8, 9], [7]], 0)
    masks = prefill_masks(pos, valid, window=2)
    assert masks["full_attention"][1, 0].tolist() == [[True, False, False], [False, True, False],
                                                      [False, False, True]]
    assert masks["sliding_attention"][0, 0].tolist() == [[True, False, False], [True, True, False],
                                                         [False, True, True]]
    assert ids.tolist() == [[7, 8, 9], [0, 0, 7]] and pos.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_adapters_start_as_the_identity_and_never_touch_the_prompt():
    realizer, _ = tiny_realizer()
    ids = torch.randint(5, VOCAB, (2, 20))
    canvas, nothing = ids[:, 8:], torch.zeros(2, 12, dtype=torch.bool)
    t, h = torch.tensor([0.25, 0.5]), torch.tensor([0.125, 0.5])
    with torch.no_grad():
        cache = realizer.prefill(*left_pad([r.tolist() for r in ids[:, :8]], 0))
        on = realizer.canvas_hidden(cache, canvas, nothing, t, h, adapters=True)
        off = realizer.canvas_hidden(cache, canvas, nothing, t, h, adapters=False)
        torch.testing.assert_close(on, off, rtol=0, atol=0)          # LoRA B = 0, modulation = 0
        assert float(realizer.modulation(t, h).abs().max()) == 0.0
        randomize_adapters(realizer)
        changed = realizer.canvas_hidden(cache, canvas, nothing, t, h)
        assert float((changed - off).abs().max()) > 1e-3
        again = realizer.prefill(*left_pad([r.tolist() for r in ids[:, :8]], 0))
        for layer, keys in cache.keys.items():                       # one prompt cache for every role
            torch.testing.assert_close(again.keys[layer], keys, rtol=0, atol=0)
            torch.testing.assert_close(again.values[layer], cache.values[layer], rtol=0, atol=0)
        other_h = realizer.canvas_hidden(cache, canvas, nothing, t, torch.tensor([0.5, 0.125]))
        assert float((other_h - changed).abs().max()) > 1e-4         # the step size reaches the network
    assert sorted(cache.keys) == [0, 1, 2, 3]                        # only the KV-storing blocks ran
    frozen = [n for n, p in realizer.named_parameters() if not p.requires_grad]
    assert frozen and all("lora_" not in n for n in frozen)
    trainable = {n for n, p in realizer.named_parameters() if p.requires_grad}
    assert all(("lora_" in n) or n.startswith(("mask_embedding", "t_mlp", "h_mlp", "mod_")) for n in trainable)


def test_mask_positions_use_the_mask_embedding_and_ignore_their_ids():
    realizer, _ = tiny_realizer()
    torch.testing.assert_close(realizer.mask_embedding.detach(),
                               realizer.text.embed_tokens(torch.tensor([MASK]))[0].detach())
    ids = torch.randint(5, VOCAB, (1, 16))
    canvas = ids[:, 6:].clone()
    is_mask = torch.zeros(1, 10, dtype=torch.bool)
    is_mask[0, [2, 3, 7]] = True
    t, h = torch.zeros(1), torch.ones(1)
    with torch.no_grad():
        cache = realizer.prefill(*left_pad([ids[0, :6].tolist()], 0))
        first = realizer.canvas_hidden(cache, canvas, is_mask, t, h)
        canvas[0, [2, 3, 7]] = torch.tensor([9, 10, 11])
        second = realizer.canvas_hidden(cache, canvas, is_mask, t, h)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        explicit = canvas.clone()
        explicit[0, [2, 3, 7]] = MASK                                 # the <mask> token itself: same at init
        third = realizer.canvas_hidden(cache, explicit, torch.zeros_like(is_mask), t, h)
        torch.testing.assert_close(first, third, rtol=1e-5, atol=1e-6)
        with pytest.raises(ValueError, match="anchor"):
            realizer.canvas_hidden(cache, canvas, torch.ones_like(is_mask), t, h)


def test_activation_checkpointing_gives_the_same_gradients():
    realizer, _ = tiny_realizer()
    randomize_adapters(realizer)
    ids = torch.randint(5, VOCAB, (2, 18))
    canvas = ids[:, 7:]
    is_mask = torch.zeros(2, 11, dtype=torch.bool)
    is_mask[:, 3:9] = True
    t, h = torch.tensor([0.0, 0.5]), torch.tensor([0.25, 0.5])
    cache = realizer.prefill(*left_pad([r.tolist() for r in ids[:, :7]], 0))
    targets = torch.randint(5, VOCAB, (2, 11))
    grads = []
    for checkpointing in (False, True):
        realizer.zero_grad()
        hidden = realizer.canvas_hidden(cache, canvas, is_mask, t, h, checkpointing=checkpointing)
        loss, _ = realizer.weighted_nll(hidden[is_mask], targets[is_mask], torch.ones(int(is_mask.sum())), chunk=5)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in realizer.named_parameters() if p.requires_grad})
    assert set(grads[0]) == set(grads[1]) and len(grads[0]) > 40
    for name, grad in grads[0].items():
        torch.testing.assert_close(grads[1][name], grad, rtol=1e-4, atol=1e-6, msg=name)
    stored = [n for n in grads[0] if ".3.self_attn.k_proj.lora_b" in n or ".2.self_attn.v_proj.lora_b" in n]
    assert len(stored) == 2 and all(float(grads[0][n].abs().sum()) > 0 for n in stored)   # shared K/V carry grads


def test_weighted_nll_matches_a_direct_computation():
    realizer, _ = tiny_realizer()
    hidden = torch.randn(13, 64, requires_grad=True)
    targets = torch.randint(0, VOCAB, (13,))
    weights = torch.rand(13)
    total, hits = realizer.weighted_nll(hidden, targets, weights, chunk=4)
    logits = realizer.lm_head(hidden.detach().double().float()).double()
    logits = 30.0 * torch.tanh(logits / 30.0)                        # the final soft cap
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    want = -(weights.double() * logp[torch.arange(13), targets]).sum()
    assert math.isclose(float(total.detach()), float(want), rel_tol=1e-5)
    assert int(hits) == int((logits.argmax(-1) == targets).sum())
    total.backward()
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0


def test_checkpoint_state_round_trip():
    realizer, _ = tiny_realizer()
    randomize_adapters(realizer)
    state = realizer.trainable_state()
    other, _ = tiny_realizer()
    other.load_trainable_state(state)
    for (name, a), (_, b) in zip(sorted(realizer.trainable_state().items()), sorted(other.trainable_state().items()),
                                 strict=True):
        torch.testing.assert_close(a, b, msg=name)
    assert all(".base." not in name and "embed_tokens" not in name for name in state)
    with pytest.raises(ValueError, match="disagree"):
        other.load_trainable_state({k: v for k, v in state.items() if "mask_embedding" not in k})


# =============================================================================================
# The sampler
# =============================================================================================
def test_draw_tokens_is_temperature_top_k_then_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]]))
    # Reference: T = 1; top-4 keeps {0, 1, 2, 3} -> 0.5/.95 ...; nucleus 0.8 keeps the smallest prefix of mass
    # >= 0.8: {0, 1, 2} (0.526 + 0.211 + 0.158 = 0.895), renormalised.
    kept = np.array([0.5, 0.2, 0.15]) / 0.85
    generator = torch.Generator().manual_seed(0)
    draws, probs = draw_tokens(logits.expand(40_000, -1), 1.0, 0.8, 4, generator=generator)
    freq = np.bincount(draws.numpy(), minlength=5) / 40_000
    assert freq[3] == 0 and freq[4] == 0
    assert np.all(np.abs(freq[:3] - kept) < 5 * np.sqrt(kept * (1 - kept) / 40_000))
    posterior = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])           # the reported probability is the posterior's
    torch.testing.assert_close(probs, posterior[draws], rtol=1e-5, atol=1e-6)
    greedy, confidence = draw_tokens(logits, 0.0, 0.8, 4, forbidden=[0])
    assert greedy.tolist() == [1] and math.isclose(float(confidence), 0.2 / 0.5, rel_tol=1e-5)
    sharp, confidence = draw_tokens(logits.expand(64, -1), 0.01, 1.0, 5, generator=generator)
    assert sharp.tolist() == [0] * 64 and bool((confidence - 0.5).abs().max() < 1e-5)


def test_reveal_counts():
    masked = torch.tensor([10, 3, 0, 1])
    assert reveal_counts(masked, 0.25, last=False).tolist() == [2, 1, 0, 1]
    assert reveal_counts(masked, 0.25, last=True).tolist() == [10, 3, 0, 1]


def test_one_step_is_the_exact_jump_kernel_on_the_mask_source():
    """With the mask source and no energies, Eq. 17's kernel is: stay at [MASK] with probability 1 - pi, else
    draw from the posterior restricted to V minus [MASK]. Checked by Monte Carlo against ``kernel_distribution``
    for the first step of K = 4 (cosine): 5 standard errors."""
    realizer, _ = tiny_realizer()
    randomize_adapters(realizer)
    cfg = SamplerConfig(num_steps=4, schedule="cosine", temperature=1.0, top_p=1.0, top_k=VOCAB)
    prefix = [2, 40, 41, 42, 43]
    rows, canvas_len = 4000, 3
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        cache = realizer.prefill(*left_pad([prefix], 0))
        anchor = torch.full((1,), 44)
        tokens = torch.full((1, canvas_len), MASK)
        hidden = realizer.canvas_hidden(cache, torch.cat([anchor[:, None], tokens], 1),
                                        torch.tensor([[False, True, True, True]]), torch.zeros(1),
                                        torch.full((1,), 0.25))[:, :canvas_len]
        logits = realizer.logits(hidden)[0].double()
        logits[:, MASK] = -math.inf
        posterior = torch.softmax(logits, dim=-1)                                    # [N, S]
        pi = KappaSchedule("cosine").exact_jump_mass(0.0, 0.25)
        kernel = tilted_jump_kernel(posterior, torch.full((canvas_len,), MASK), pi)
        want = kernel_distribution(kernel, torch.full((canvas_len,), MASK)).numpy()  # [N, S]
        # One step of the sampler, many times: run K = 4 but look only at what the first step produced.
        seen = np.zeros_like(want)
        original = realizer.canvas_hidden
        states: list[torch.Tensor] = []

        def spy(cache_, canvas_ids, is_mask, t, h, **kw):
            states.append(torch.where(is_mask, torch.full_like(canvas_ids, MASK), canvas_ids)[:, 1:].clone())
            return original(cache_, canvas_ids, is_mask, t, h, **kw)

        realizer.canvas_hidden = spy                                                 # type: ignore[method-assign]
        result = realize(realizer, cache.expand(rows), anchor.expand(rows), canvas_len, cfg, generator=generator)
        realizer.canvas_hidden = original                                            # type: ignore[method-assign]
    after_first = states[1].numpy()                                                  # the state step 2 starts from
    for j in range(canvas_len):
        seen[j] = np.bincount(after_first[:, j], minlength=VOCAB) / rows
    assert math.isclose(float(want[0, MASK]), 1.0 - pi, rel_tol=1e-9)
    error = 5 * np.sqrt(want * (1 - want) / rows) + 1e-9
    assert np.all(np.abs(seen - want) <= error)
    assert result.canvas_passes == 4 and result.sequential_depth == 5
    assert not bool((result.tokens == MASK).any())
    assert sum(result.revealed) == rows * canvas_len


@pytest.mark.parametrize("order", ["random", "confidence"])
def test_realize_reveals_once_and_keeps_revealed_tokens(order):
    realizer, _ = tiny_realizer()
    randomize_adapters(realizer)
    cfg = SamplerConfig(num_steps=8, schedule="cosine", order=order, temperature=0.7)
    rows, canvas_len = 6, 40
    states: list[torch.Tensor] = []
    original = realizer.canvas_hidden

    def spy(cache_, canvas_ids, is_mask, t, h, **kw):
        assert not bool(is_mask[:, 0].any()) and bool((canvas_ids[:, 0] == 50).all())     # the anchor is clamped
        states.append(torch.where(is_mask, torch.full_like(canvas_ids, MASK), canvas_ids)[:, 1:].clone())
        return original(cache_, canvas_ids, is_mask, t, h, **kw)

    realizer.canvas_hidden = spy                                                     # type: ignore[method-assign]
    with torch.no_grad():
        cache = realizer.prefill(*left_pad([[2, 30, 31]], 0))
        result = realize(realizer, cache.expand(rows), torch.full((rows,), 50), canvas_len, cfg, forbidden=[0, 2],
                         generator=torch.Generator().manual_seed(1))
    assert len(states) == 8 and bool((states[0] == MASK).all())
    states.append(result.tokens)
    for before, after in zip(states[:-1], states[1:], strict=True):
        kept = before != MASK
        assert torch.equal(after[kept], before[kept])                 # revealed positions never change
    assert not bool(torch.isin(result.tokens, torch.tensor([MASK, 0, 2])).any())
    if order == "confidence":                                         # round(pi * masked) per row, cosine kappa
        kappa = [1 - math.cos(math.pi * k / 16) for k in range(9)]
        masked = canvas_len
        for k in range(8):
            count = masked if k == 7 else max(1, round(masked * (kappa[k + 1] - kappa[k]) / (1 - kappa[k])))
            assert result.revealed[k] == rows * count
            masked -= count


# =============================================================================================
# Data
# =============================================================================================
ROW = {
    "context": 'import math as m\n\n\ndef area(r: float, *, scale=1) -> float:\n    """Return the area of a circle.\n\n'
               '    Uses pi.\n    """\n',
    "target": 'import math as m\n\n\ndef area(r: float, *, scale=1) -> float:\n    """Return the area of a circle.\n\n'
              '    Uses pi.\n    """\n    # comment\n    return m.pi * r ** 2 * scale\n',
    "documented": True, "method": False,
}


def test_example_formats():
    prompt, body = s0data.complete_example(ROW)
    assert prompt == ROW["context"] and prompt + body == ROW["target"]
    prompt, code = s0data.write_example(ROW)
    assert prompt == '"""\nReturn the area of a circle.\n\nUses pi.\nassert area(r, scale) == ...\n"""\n\n'
    assert code == ("import math as m\n\n\ndef area(r: float, *, scale=1) -> float:\n    # comment\n"
                    "    return m.pi * r ** 2 * scale\n")
    compile(code, "<write>", "exec")
    assert s0data.write_example({**ROW, "method": True}) is None
    import warnings

    escaped = {**ROW, "target": ROW["target"].replace("return m.pi", 'x = "\\d"\n    return m.pi')}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # a SyntaxWarning would raise here
        assert s0data.write_example(escaped) is not None
    assert s0data.write_example({**ROW, "documented": False}) is None
    one_line = 'def f(x):\n    """Doc."""; return x\n'
    assert s0data.write_example({"context": "", "target": one_line, "documented": True}) is None
    assert s0data.complete_example({"context": "def f():\n", "target": "def g():\n    pass\n"}) is None
    assert s0data.mbpp_prompt("Add two numbers.\nassert add(1, 2) == 3") == \
        '"""\nAdd two numbers.\nassert add(1, 2) == 3\n"""\n\n'      # the blank line: see data.py


def test_tokenize_collate_and_training_state():
    tok = CharTokenizer()
    pairs = [("def f(x):\n", "    return x\n", "complete"), ("p", "y" * 40, "complete"), ("q" * 30, "z", "complete")]
    examples, dropped = s0data.tokenize_examples(pairs, tok, canvas_len=20, max_prompt_tokens=16)
    assert len(examples) == 1 and dropped == {"canvas_too_long": 1, "prompt_too_long": 1, "special_token": 0}
    ex = examples[0]
    assert ex.prompt_ids[0] == tok.bos_id and ex.canvas_ids[-1] == tok.eos_id and len(ex.canvas_ids) == 14
    batch = s0data.collate([ex, ex], 20, tok.eos_id)
    assert batch.prefix_rows[0] == ex.prompt_ids[:-1] and batch.anchor.tolist() == [5, 5]     # "\n" is the anchor
    assert batch.length.tolist() == [14, 14] and bool((batch.target[:, 13:] == tok.eos_id).all())
    schedule = KappaSchedule("cosine")
    state = s0data.training_state(batch, schedule, [2, 4, 8], MASK, VOCAB, 0.1, torch.Generator().manual_seed(0),
                                  budget=8, step_index=4)
    assert state.t.tolist() == [0.5, 0.5] and state.h.tolist() == [0.125, 0.125]
    assert state.canvas_ids.shape == (2, 21) and not bool(state.is_mask[:, 0].any())
    masked = state.is_mask[:, 1:]
    assert torch.equal(state.canvas_ids[:, 1:][~masked], batch.target[~masked])               # revealed = clean
    want = torch.where(torch.arange(20)[None] < 14, 1.0, 0.1) * masked
    torch.testing.assert_close(state.weights, want.float())
    zero = s0data.training_state(batch, schedule, [2, 4, 8], MASK, VOCAB, 0.1, torch.Generator().manual_seed(0),
                                 budget=2, step_index=0)
    assert bool(zero.is_mask[:, 1:].all())                           # t = 0: the source, everything masked
    # Drawn states: K from the budgets, t on K's grid, and the masked fraction follows 1 - kappa_t.
    big = s0data.collate([ex] * 512, 20, tok.eos_id)
    drawn = s0data.training_state(big, schedule, [2, 4, 8], MASK, VOCAB, 0.1, torch.Generator().manual_seed(3))
    budgets = (1.0 / drawn.h).round()
    assert set(budgets.tolist()) == {2.0, 4.0, 8.0}
    torch.testing.assert_close(drawn.t * budgets, (drawn.t * budgets).round(), atol=1e-5, rtol=0)
    expected = (1.0 - schedule.kappa(drawn.t)).mean()
    observed = drawn.is_mask[:, 1:].float().mean()
    assert abs(float(observed - expected)) < 5 * 0.5 / math.sqrt(512 * 20)


def _write_corpus(root, functions: int = 60, shards: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(functions):
        context = f'def f{i}(a, b):\n    """Add {i} to the sum of a and b, twice over."""\n'
        rows.append({"context": context, "target": context + f"    c = a + b + {i}\n    return c\n",
                     "documented": True, "method": False})
    for s in range(shards):
        with gzip.open(root / f"train-{s:05d}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for row in rows[s::shards]:
                handle.write(json.dumps(row) + "\n")
    with gzip.open(root / "val.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows[:8]:
            handle.write(json.dumps(row) + "\n")


def test_corpus_stream_is_deterministic_and_resumes_exactly(tmp_path):
    _write_corpus(tmp_path)
    tok = CharTokenizer()

    def stream(state=None):
        return s0data.CorpusStream(tmp_path, tok, 96, 128, write_fraction=0.5, seed=3, chunk_rows=7, state=state)

    first = stream()
    taken = first.take(150)                                           # more than two epochs of 60
    assert [e.canvas_ids for e in stream().take(150)] == [e.canvas_ids for e in taken]
    assert {e.kind for e in taken} == {"complete", "write"}
    epoch = [tuple(e.prompt_ids) for e in taken[:60]]
    assert len({tuple(e.prompt_ids + e.canvas_ids) for e in taken[:60]}) == 60 and epoch != sorted(epoch)
    other = s0data.CorpusStream(tmp_path, tok, 96, 128, write_fraction=0.5, seed=4, chunk_rows=7)
    assert [e.canvas_ids for e in other.take(60)] != [e.canvas_ids for e in taken[:60]]
    for cut in (0, 5, 7, 23, 60, 97):
        head = stream()
        head.take(cut)
        resumed = stream(json.loads(json.dumps(head.state())))
        assert [e.canvas_ids for e in resumed.take(40)] == [e.canvas_ids for e in taken[cut:cut + 40]], cut
    assert len(s0data.load_validation(tmp_path, tok, 96, 128, limit=6)) == 6


# =============================================================================================
# Truncation and scoring
# =============================================================================================
def test_truncation_rules():
    body = "    x = 1\n\n    return x\n\n\ndef other():\n    pass\n"
    assert truncate_complete(body) == "    x = 1\n\n    return x\n"
    assert is_complete(body, "complete") and not is_complete("    x = 1\n    return x\n", "complete")
    assert truncate_complete("    return 1\nprint(f(2))\n") == "    return 1\n"
    program = "import re\n\nclass H:\n    pass\n\ndef f(x):\n    return x\n\nassert f(1) == 1\nprint(f(2))\n"
    assert truncate_write(program) == "import re\n\nclass H:\n    pass\n\ndef f(x):\n    return x\n"
    assert truncate_write('def f(x):\n    return x\n"""\nNext task\n"""\n') == "def f(x):\n    return x\n"
    assert not is_complete("def f(x):\n    return x\n", "write")
    assert extract_fenced("Here:\n```python\ndef f():\n    return 1\n```\nDone.") == "def f():\n    return 1\n"
    assert extract_fenced("def f():\n    return 1") == "def f():\n    return 1\n"
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # a warning would raise here
        assert parses('import re\nx = re.compile("\\d+")\n') and not parses("def f(:\n")


def _tasks() -> tuple[EvalTask, EvalTask]:
    prompt = 'def add(a, b):\n    """Add two numbers."""\n'
    humaneval = EvalTask("HE/0", prompt, prompt.rstrip() + "\n\n\n", "\n\nassert add(1, 2) == 3\n",
                         prompt + "    return a + b\n")
    mbpp = EvalTask("MBPP/1", "Add two numbers.\nassert add(1, 2) == 3", "", "\n\nassert add(1, 2) == 3\n",
                    "def add(a, b):\n    return a + b\n")
    return humaneval, mbpp


def test_score_uses_the_same_truncation_for_every_decoder():
    humaneval, mbpp = _tasks()
    assert reference_text(humaneval, "humaneval") == "    return a + b\n"
    texts = [["    return a + b\n\n\ndef junk(:\n", "    return a - b\n"]]
    summary, rows = score([humaneval], "humaneval", texts, [1, 2], workers=1)
    assert [r["passed"] for r in rows] == [True, False] and summary["pass@1"] == 0.5 and summary["pass@2"] == 1.0
    assert summary["parse_rate"] == 1.0 and rows[0]["completion"].endswith("    return a + b\n")
    summary, rows = score([mbpp], "mbpp", [["def add(a, b):\n    return a + b\nassert add(1, 1) == 3\n",
                                           "def add(a, b)\n"]], [1], workers=1)
    assert [r["passed"] for r in rows] == [True, False] and [r["parses"] for r in rows] == [True, False]
    assert build_sample(mbpp, "mbpp", "```python\ndef add(a, b):\n    return a + b\n```", chat=True) == \
        "def add(a, b):\n    return a + b\n"


def test_realizer_samples_shapes_and_depth():
    realizer, _ = tiny_realizer()
    humaneval, mbpp = _tasks()
    cfg = SamplerConfig(num_steps=4, schedule="cosine")
    texts, info = realizer_samples(realizer, CharTokenizer(), [humaneval], "humaneval", cfg, canvas_len=12,
                                   num_samples=5, batch_size=2, seed=0, device=torch.device("cpu"))
    assert len(texts) == 1 and len(texts[0]) == 5 and all(isinstance(t, str) for t in texts[0])
    assert info["sequential_depth"] == 5 and math.isclose(sum(info["mean_revealed_per_step"]), 12, abs_tol=0.05)
    again, _ = realizer_samples(realizer, CharTokenizer(), [humaneval], "humaneval", cfg, canvas_len=12,
                                num_samples=5, batch_size=2, seed=0, device=torch.device("cpu"))
    assert again == texts                                             # seeded


# =============================================================================================
# The autoregressive baseline
# =============================================================================================
def test_cut_at_eos():
    assert cut_at_eos([7, 8, 1, 9], [1]) == [7, 8] and cut_at_eos([7, 8], [1, 106]) == [7, 8]


def test_greedy_generate_equals_the_backbone_run_through_the_realizer_blocks():
    """transformers' ``generate`` (the baseline) and the realizer's blocks with causal masks and no adapters are
    the same network: the greedy continuation is identical token by token."""
    realizer, reference = tiny_realizer(seed=1)
    prompt = [2, 60, 61, 62, 63, 64, 65, 66, 67]

    class Tok:
        eos_token_id, pad_token_id = 1, 0

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    reference.generation_config.eos_token_id = 1
    texts, lengths = generate_autoregressive(reference, Tok(), prompt, 1, 10, 0.0, 1.0, 50, seed=0)
    generated = [int(x) for x in texts[0].split()]
    assert lengths == [len(generated)]
    with torch.no_grad():
        cache = realizer.prefill(*left_pad([prompt[:-1]], 0))
        canvas = [prompt[-1]]
        for _ in range(len(generated)):
            ids = torch.tensor([canvas])
            hidden = realizer.canvas_hidden(cache, ids, torch.zeros_like(ids, dtype=torch.bool), torch.zeros(1),
                                            torch.ones(1), mode="causal", adapters=False)
            canvas.append(int(realizer.logits(hidden[0, -1]).argmax()))
    assert canvas[1:] == generated


# =============================================================================================
# Training
# =============================================================================================
def test_learning_rate_schedule():
    cfg = Stage0Config(lr=2e-4, warmup_steps=10, schedule_steps=110, min_lr_fraction=0.1)
    assert math.isclose(learning_rate(0, cfg), 2e-5) and math.isclose(learning_rate(9, cfg), 2e-4)
    half = 2e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert math.isclose(learning_rate(60, cfg), half)
    assert math.isclose(learning_rate(110, cfg), 2e-5) and math.isclose(learning_rate(5000, cfg), 2e-5)


def _trainer(tmp_path, run: str, steps: int, **overrides):
    model = tiny_backbone()
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec(mask_id=MASK, lora_rank=16, lora_alpha=16.0,
                                                                        time_dim=32))
    tok = CharTokenizer()
    cfg = Stage0Config(canvas_len=48, max_prompt_tokens=96, lora_rank=16, lora_alpha=16.0, lr=1e-2, warmup_steps=5,
                       steps=steps, schedule_steps=steps, micro_batch=4, grad_accum=2, log_every=1000,
                       save_every=1000, eval_every=1000, val_examples=8, write_fraction=0.0, **overrides)
    stream = s0data.CorpusStream(tmp_path / "corpus", tok, cfg.canvas_len, cfg.max_prompt_tokens, cfg.write_fraction,
                                 cfg.seed, chunk_rows=16)
    validation = s0data.load_validation(tmp_path / "corpus", tok, cfg.canvas_len, cfg.max_prompt_tokens, 8)
    return Stage0Trainer(realizer, tok, cfg, stream, validation, tmp_path / run, torch.device("cpu"))


def test_training_reduces_the_loss_and_resumes_exactly(tmp_path):
    _write_corpus(tmp_path / "corpus", functions=16, shards=2)
    trainer = _trainer(tmp_path, "a", steps=45)
    report = trainer.fit_memory()
    assert report["micro_batch"] == 4 and report["grad_checkpoint"] is False
    backbone_before = {n: p.clone() for n, p in trainer.realizer.named_parameters() if not p.requires_grad}
    before = trainer.validate()["val_loss"]
    trainer.realizer.train()
    for _ in range(12):
        trainer.train_step()
    trainer.save()
    for _ in range(33):
        last = trainer.train_step()
    after = trainer.validate()
    # The tiny model's tied embeddings are small (std 0.02), so its logits stay flat and the loss moves slowly;
    # the argmax is what shows that the canvas tokens are being learned (chance: 1/320).
    assert after["val_loss"] < before - 0.3 and after["val_accuracy"] > 0.4 and math.isfinite(last["loss"])
    for name, p in trainer.realizer.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, backbone_before[name]), name       # the released weights never change

    resumed = _trainer(tmp_path, "b", steps=45)
    payload = torch.load(tmp_path / "a" / "last.pt", weights_only=True)
    resumed.stream = s0data.CorpusStream(tmp_path / "corpus", CharTokenizer(), 48, 96, 0.0, 0, chunk_rows=16,
                                         state=payload["stream"])
    resumed.restore(payload)
    assert resumed.step == 12 and resumed.examples_seen == 12 * 8
    resumed.realizer.train()
    for _ in range(33):
        resumed.train_step()
    for (name, a), (_, b) in zip(sorted(trainer.realizer.trainable_state().items()),
                                 sorted(resumed.realizer.trainable_state().items()), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7, msg=name)


def test_memory_fallback_switches_on_checkpointing_then_halves_the_micro_batch(tmp_path, monkeypatch):
    _write_corpus(tmp_path / "corpus", functions=16, shards=2)
    trainer = _trainer(tmp_path, "m", steps=1)
    calls: list[tuple[int, bool]] = []
    real = trainer.loss_on

    def tight(batch, **kw):
        calls.append((batch.target.shape[0], trainer.cfg.grad_checkpoint))
        if batch.target.shape[0] > 2:
            raise torch.OutOfMemoryError("simulated")
        return real(batch, **kw)

    monkeypatch.setattr(trainer, "loss_on", tight)
    report = trainer.fit_memory()
    assert calls == [(4, False), (4, True), (2, True)]
    assert (report["micro_batch"], report["grad_accum"], report["grad_checkpoint"]) == (2, 4, True)
    assert len(report["tried"]) == 2


# =============================================================================================
# The report
# =============================================================================================
def test_bootstrap_ratio_and_verdict():
    rng = np.random.default_rng(0)
    baseline = (rng.random(400) < 0.4).astype(float)
    realizer = baseline * (rng.random(400) < 0.75)                    # solves 3/4 of what the baseline solves
    rho, low, high = bootstrap_ratio(realizer, baseline)
    assert math.isclose(rho, realizer.mean() / baseline.mean()) and low < rho < high
    assert 0.6 < low and high < 0.9
    # An independent reference for the interval: a plain loop with another generator.
    loop_rng = random.Random(1)
    ratios = []
    for _ in range(2000):
        index = [loop_rng.randrange(400) for _ in range(400)]
        ratios.append(realizer[index].mean() / baseline[index].mean())
    assert abs(low - np.percentile(ratios, 2.5)) < 0.02 and abs(high - np.percentile(ratios, 97.5)) < 0.02
    assert math.isnan(bootstrap_ratio(np.zeros(5), np.zeros(5))[0])
    assert h2_verdict({"humaneval": (0.7, 0.6, 0.8), "mbpp": (0.6, 0.52, 0.7)}) == "clearly holds"
    assert h2_verdict({"humaneval": (0.7, 0.45, 0.8), "mbpp": (0.6, 0.52, 0.7)}) == "holds"
    assert h2_verdict({"humaneval": (0.2, 0.1, 0.3), "mbpp": (0.1, 0.05, 0.2)}) == "clearly fails"
    assert h2_verdict({"humaneval": (0.4, 0.3, 0.55), "mbpp": (0.1, 0.05, 0.2)}) == "fails"
    assert h2_verdict({"humaneval": (0.6, 0.5, 0.7), "mbpp": (0.3, 0.2, 0.4)}) == "mixed"
    assert h2_verdict({}).startswith("undecided")
