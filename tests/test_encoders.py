# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of ``text_vpa.models.encoders`` against PR-JEPA Sec. 3.1 (pass 1 and the training-only target branch).

Specification (Sec. 3.1, Fig. 1, Eq. 4) as checked here:

    s_x = f_theta(c) in R^{M_c x d}
        x   = E_tok[c_t] + E_pos[t]                                (learned absolute positions)
        x  <- x + MHA(LN(x));  x <- x + W_2 GELU(W_1 LN(x) + b_1) + b_2   (pre-norm, bidirectional)
        s_x = W_h LN(x) + b_h;  exactly 0 at padded tokens, which are never attended to
    s_y = sg(f_theta_bar(y)) in R^{M x d}
        s_y[b, m] = mean of f_theta_bar(y)[b, t] over the REAL tokens t with slot_index[b, t] = m
        slot_index = -1: token belongs to no slot; an empty slot is a zero vector with slot_mask False
    f_theta_bar is the EMA copy of f_theta: same parameter names, frozen, forward under no_grad
    (the stop-gradient of Eq. 5), always in eval mode.

References are float64 computations written from these formulas with the module's weights cast to
float64 (the ``ref_*`` helpers); they never call the code under test.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tests.conftest import ATOL, RTOL, accelerator
from text_vpa.configs import load_config
from text_vpa.models.encoders import ContextEncoder, TargetEncoder, TargetOutput

VOCAB, LATENT, WIDTH, HEADS, MAX_POSITIONS = 37, 12, 16, 4, 16
NUM_SLOTS = 5
# Slots that receive no real token in ``make_slot_index`` (two of them receive only padded tokens).
EMPTY_SLOTS = {(0, 3), (0, 4), (1, 3), (1, 4), (2, 2), (2, 3)}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
def randomise_layer_norms(module: nn.Module, seed: int = 7) -> None:
    """Give every LayerNorm a non-trivial gain and shift, so a reference that ignores them fails."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for norm in (m for m in module.modules() if isinstance(m, nn.LayerNorm)):
            norm.weight.copy_(1.0 + 0.1 * torch.randn(norm.weight.shape, generator=generator))
            norm.bias.copy_(0.1 * torch.randn(norm.bias.shape, generator=generator))


def make_encoder(num_layers: int = 1, dropout: float = 0.0) -> ContextEncoder:
    """A small f_theta in eval mode."""
    encoder = ContextEncoder(VOCAB, LATENT, dim=WIDTH, num_layers=num_layers, num_heads=HEADS, mlp_ratio=2.0,
                             dropout=dropout, max_positions=MAX_POSITIONS)
    randomise_layer_norms(encoder)
    return encoder.eval()


def make_batch() -> tuple[Tensor, Tensor]:
    """Three rows of 9 tokens: no padding, trailing padding, and left padding with interior holes."""
    tokens = torch.randint(0, VOCAB, (3, 9), generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([
        [True] * 9,
        [True] * 6 + [False] * 3,
        [False, False, True, True, False, True, True, True, False],
    ])
    return tokens, mask


def make_slot_index() -> Tensor:
    """Slots of ``make_batch``'s tokens, interleaved, with unassigned (-1) real tokens and padded tokens
    that carry a valid slot index (row 1: positions 6-8; row 2: positions 0, 1, 4, 8)."""
    return torch.tensor([
        [0, 0, 1, -1, 2, 1, -1, 0, 2],
        [2, 2, -1, 0, 0, 1, 2, 3, 3],
        [4, 0, 4, -1, 1, 1, 0, 4, 2],
    ])


# ---------------------------------------------------------------------------------------------
# float64 references written from the specification
# ---------------------------------------------------------------------------------------------
def f64(t: Tensor) -> Tensor:
    return t.detach().to(device="cpu", dtype=torch.float64)


def ref_layer_norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """(x - mean) / sqrt(biased variance + eps) * gain + shift over the last axis."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + norm.eps) * f64(norm.weight) + f64(norm.bias)


def ref_linear(x: Tensor, linear: nn.Linear) -> Tensor:
    return x @ f64(linear.weight).T + f64(linear.bias)


def ref_gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def ref_self_attention(x: Tensor, attention: nn.Module, num_heads: int, key_mask: Tensor | None) -> Tensor:
    """softmax((X W_Q)(X W_K)^T / sqrt(d_h)) (X W_V) per head over the real keys, heads concatenated, then W_O."""
    batch, length, width = x.shape
    head_dim = width // num_heads

    def split(t: Tensor) -> Tensor:
        return t.reshape(batch, length, num_heads, head_dim).transpose(1, 2)

    q = split(ref_linear(x, attention.q_proj))
    k = split(ref_linear(x, attention.k_proj))
    v = split(ref_linear(x, attention.v_proj))
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :], -math.inf)
    weights = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    heads = (weights @ v).transpose(1, 2).reshape(batch, length, width)
    return ref_linear(heads, attention.out_proj)


def ref_encoder(encoder: ContextEncoder, tokens: Tensor, mask: Tensor | None, num_heads: int) -> Tensor:
    """float64 f_theta: embeddings, pre-norm blocks, final LayerNorm, head; exact zeros at padded tokens."""
    x = f64(encoder.token_embedding.weight)[tokens] + f64(encoder.position_embedding.weight)[: tokens.shape[1]]
    for block in encoder.blocks:
        x = x + ref_self_attention(ref_layer_norm(x, block.norm_self), block.self_attn, num_heads, mask)
        fc1, fc2 = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
        x = x + ref_linear(ref_gelu(ref_linear(ref_layer_norm(x, block.norm_mlp), fc1)), fc2)
    out = ref_linear(ref_layer_norm(x, encoder.final_norm), encoder.head)
    if mask is not None:
        out = torch.where(mask[..., None], out, torch.zeros((), dtype=out.dtype))
    return out


def ref_slot_pool(latents: Tensor, slot_index: Tensor, num_slots: int, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Loop over (b, m): mean of latents[b, t] over real t with slot_index[b, t] == m; zero and False if empty."""
    batch, length, width = latents.shape
    s_y = torch.zeros(batch, num_slots, width, dtype=torch.float64)
    slot_mask = torch.zeros(batch, num_slots, dtype=torch.bool)
    for b in range(batch):
        for m in range(num_slots):
            members = [t for t in range(length)
                       if int(slot_index[b, t]) == m and (mask is None or bool(mask[b, t]))]
            if members:
                s_y[b, m] = sum(latents[b, t] for t in members) / len(members)
                slot_mask[b, m] = True
    return s_y, slot_mask


# ---------------------------------------------------------------------------------------------
# f_theta: shapes, padding, reference forward
# ---------------------------------------------------------------------------------------------
def test_context_encoder_output_shape_and_dtype() -> None:
    """f_theta maps [B, L] token ids to [B, L, d] float latents."""
    tokens, mask = make_batch()
    s_x = make_encoder()(tokens, mask)
    assert s_x.shape == (3, 9, LATENT)
    assert s_x.dtype == torch.float32


def test_padded_positions_are_exact_zeros() -> None:
    """Padded positions are returned as exact zeros; real positions are not zeroed."""
    tokens, mask = make_batch()
    s_x = make_encoder()(tokens, mask)
    assert torch.equal(s_x[~mask], torch.zeros(int((~mask).sum()), LATENT))
    assert bool((s_x[mask].abs().sum(dim=-1) > 0).all())


def test_real_outputs_do_not_depend_on_padded_token_ids() -> None:
    """Changing the ids at padded positions leaves every real output bit-identical (padded keys are masked)."""
    encoder = make_encoder()
    tokens, mask = make_batch()
    other = tokens.clone()
    other[~mask] = (tokens[~mask] + 1) % VOCAB
    torch.testing.assert_close(encoder(other, mask), encoder(tokens, mask), rtol=0.0, atol=0.0)


def test_real_outputs_do_not_depend_on_amount_of_trailing_padding() -> None:
    """Appending padded tokens (as a longer batch would) does not change the real latents."""
    encoder = make_encoder()
    tokens, mask = make_batch()
    longer_tokens = torch.cat([tokens, torch.randint(0, VOCAB, (3, 4))], dim=1)
    longer_mask = torch.cat([mask, torch.zeros(3, 4, dtype=torch.bool)], dim=1)
    longer = encoder(longer_tokens, longer_mask)
    torch.testing.assert_close(longer[:, :9], encoder(tokens, mask), rtol=RTOL, atol=ATOL)
    assert torch.equal(longer[:, 9:], torch.zeros(3, 4, LATENT))


def test_encoder_is_bidirectional() -> None:
    """The first position's latent depends on the last real token (no causal mask)."""
    encoder = make_encoder()
    tokens, mask = make_batch()
    other = tokens.clone()
    other[0, 8] = (tokens[0, 8] + 1) % VOCAB
    assert (encoder(other, mask)[0, 0] - encoder(tokens, mask)[0, 0]).abs().max() > 1e-4


def test_one_layer_forward_matches_float64_reference() -> None:
    """f_theta run in float64 equals the hand-written float64 forward of the specification."""
    encoder = make_encoder(num_layers=1)
    tokens, mask = make_batch()
    expected = ref_encoder(encoder, tokens, mask, HEADS)
    actual = copy.deepcopy(encoder).double()(tokens, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_one_layer_float32_forward_matches_float64_reference() -> None:
    """The float32 forward agrees with the float64 reference within the float32 tolerance."""
    encoder = make_encoder(num_layers=1)
    tokens, mask = make_batch()
    expected = ref_encoder(encoder, tokens, mask, HEADS)
    torch.testing.assert_close(encoder(tokens, mask).double(), expected, rtol=RTOL, atol=ATOL)


def make_tiny_config_encoder(planner) -> tuple[ContextEncoder, Tensor, Tensor]:
    encoder = ContextEncoder.from_config(planner)
    randomise_layer_norms(encoder)
    encoder.eval()
    tokens = torch.randint(0, planner.vocab_size, (3, 11), generator=torch.Generator().manual_seed(2))
    mask = torch.ones(3, 11, dtype=torch.bool)
    mask[1, 7:] = False
    mask[2, :3] = False
    return encoder, tokens, mask


def test_tiny_config_forward_matches_float64_reference(tiny_config) -> None:
    """The two-layer tiny-config f_theta in float64 equals the reference (this also checks the head count)."""
    planner = tiny_config.planner
    encoder, tokens, mask = make_tiny_config_encoder(planner)
    expected = ref_encoder(encoder, tokens, mask, planner.encoder.num_heads)
    actual = copy.deepcopy(encoder).double()(tokens, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_tiny_config_float32_forward_matches_float64_reference(tiny_config) -> None:
    """The float32 two-layer forward agrees with the float64 reference.

    Tolerance: the forward chains ~15 float32 reductions (per block Q, K, V, scores, weighted sum, W_O and
    two MLP layers; then the head) of length n <= 96, interleaved with five LayerNorms whose 1/std rescales
    absolute errors. Each reduction may contribute up to n u relative error (u = 2^-24 ~ 6e-8, n u <= 6e-6),
    so the accumulated worst case reaches the default rtol=1e-5 with no margin; rtol=1e-4, atol=1e-5 is used.
    The formulas themselves are checked at 1e-10 by the float64 test above.
    """
    planner = tiny_config.planner
    encoder, tokens, mask = make_tiny_config_encoder(planner)
    expected = ref_encoder(encoder, tokens, mask, planner.encoder.num_heads)
    torch.testing.assert_close(encoder(tokens, mask).double(), expected, rtol=1e-4, atol=1e-5)


def test_missing_attention_mask_treats_every_token_as_real() -> None:
    """attention_mask=None means no padding: matches the unmasked reference and an all-True mask."""
    encoder = make_encoder()
    tokens, _ = make_batch()
    expected = ref_encoder(encoder, tokens, None, HEADS)
    torch.testing.assert_close(encoder(tokens).double(), expected, rtol=RTOL, atol=ATOL)
    all_real = torch.ones(tokens.shape, dtype=torch.bool)
    torch.testing.assert_close(encoder(tokens), encoder(tokens, all_real), rtol=RTOL, atol=ATOL)


def test_accepts_sequences_of_exactly_max_positions_tokens() -> None:
    """The position table covers lengths up to and including max_positions."""
    tokens = torch.randint(0, VOCAB, (2, MAX_POSITIONS))
    assert make_encoder()(tokens).shape == (2, MAX_POSITIONS, LATENT)


def _context_inputs(case: str) -> tuple[Tensor, Tensor | None]:
    tokens = torch.randint(0, VOCAB, (2, 5))
    cases: dict[str, Callable[[], tuple[Tensor, Tensor | None]]] = {
        "float tokens": lambda: (tokens.float(), None),
        "bool tokens": lambda: (tokens > 3, None),
        "1-D tokens": lambda: (tokens[0], None),
        "longer than max_positions": lambda: (torch.randint(0, VOCAB, (2, MAX_POSITIONS + 1)), None),
        "integer mask": lambda: (tokens, torch.ones(2, 5, dtype=torch.int64)),
        "mask shape mismatch": lambda: (tokens, torch.ones(2, 4, dtype=torch.bool)),
        "all-padding row": lambda: (tokens, torch.tensor([[True] * 5, [False] * 5])),
    }
    return cases[case]()


@pytest.mark.parametrize("case", ["float tokens", "bool tokens", "1-D tokens", "longer than max_positions",
                                  "integer mask", "mask shape mismatch", "all-padding row"])
def test_context_encoder_rejects_invalid_inputs(case: str) -> None:
    """Invalid token tensors and attention masks raise ValueError."""
    tokens, mask = _context_inputs(case)
    with pytest.raises(ValueError):
        make_encoder()(tokens, mask)


def test_from_config_uses_planner_config_widths(tiny_config) -> None:
    """ContextEncoder.from_config takes vocab, d_enc, depth, MLP width and d from PlannerConfig."""
    planner = tiny_config.planner
    encoder = ContextEncoder.from_config(planner)
    assert encoder.token_embedding.weight.shape == (planner.vocab_size, planner.encoder.dim)
    assert len(encoder.blocks) == planner.encoder.num_layers
    fc1 = next(m for m in encoder.blocks[0].mlp.modules() if isinstance(m, nn.Linear))
    assert (fc1.in_features, fc1.out_features) == (planner.encoder.dim,
                                                   round(planner.encoder.dim * planner.encoder.mlp_ratio))
    assert (encoder.head.in_features, encoder.head.out_features) == (planner.encoder.dim, planner.latent_dim)
    tokens = torch.randint(0, planner.vocab_size, (2, 7))
    assert encoder(tokens).shape == (2, 7, planner.latent_dim)


def test_from_config_position_table_covers_context_and_structure_lengths() -> None:
    """f_theta and f_theta_bar share one position table of size max(max_context_len, max_structure_len)."""
    cfg = load_config(["planner=tiny", "data=tiny", "realizer=tiny", "planner.max_context_len=40", "planner.max_structure_len=56",
                       "data.max_context_tokens=40", "data.max_structure_tokens=56"])
    encoder = ContextEncoder.from_config(cfg.planner)
    assert encoder.position_embedding.weight.shape == (56, cfg.planner.encoder.dim)
    assert encoder(torch.zeros(1, 56, dtype=torch.long)).shape == (1, 56, cfg.planner.latent_dim)
    with pytest.raises(ValueError):
        encoder(torch.zeros(1, 57, dtype=torch.long))


# ---------------------------------------------------------------------------------------------
# f_theta_bar: slot pooling
# ---------------------------------------------------------------------------------------------
def test_target_forward_returns_slot_latents_and_mask() -> None:
    """forward returns TargetOutput(s_y [B, M, d] float, slot_mask [B, M] bool)."""
    tokens, mask = make_batch()
    out = TargetEncoder.from_online(make_encoder())(tokens, make_slot_index(), NUM_SLOTS, mask)
    assert isinstance(out, TargetOutput)
    assert out.s_y.shape == (3, NUM_SLOTS, LATENT) and out.s_y.dtype == torch.float32
    assert out.slot_mask.shape == (3, NUM_SLOTS) and out.slot_mask.dtype == torch.bool


def test_slot_pooling_matches_float64_loop_reference() -> None:
    """In float64, s_y and slot_mask equal the loop reference over the hand-written forward."""
    online = make_encoder()
    tokens, mask = make_batch()
    slot_index = make_slot_index()
    expected_s_y, expected_mask = ref_slot_pool(ref_encoder(online, tokens, mask, HEADS), slot_index, NUM_SLOTS, mask)
    out = TargetEncoder.from_online(copy.deepcopy(online).double())(tokens, slot_index, NUM_SLOTS, mask)
    torch.testing.assert_close(out.s_y, expected_s_y, rtol=1e-10, atol=1e-10)
    assert torch.equal(out.slot_mask, expected_mask)


def test_slot_pooling_float32_matches_float64_reference() -> None:
    """The float32 target agrees with the float64 loop reference within the float32 tolerance."""
    online = make_encoder()
    tokens, mask = make_batch()
    slot_index = make_slot_index()
    expected_s_y, expected_mask = ref_slot_pool(ref_encoder(online, tokens, mask, HEADS), slot_index, NUM_SLOTS, mask)
    out = TargetEncoder.from_online(online)(tokens, slot_index, NUM_SLOTS, mask)
    torch.testing.assert_close(out.s_y.double(), expected_s_y, rtol=RTOL, atol=ATOL)
    assert torch.equal(out.slot_mask, expected_mask)


def test_slot_pooling_without_mask_uses_every_assigned_token() -> None:
    """With attention_mask=None every token is real; only slot_index = -1 tokens are left out."""
    online = make_encoder()
    tokens, _ = make_batch()
    slot_index = make_slot_index()
    expected_s_y, expected_mask = ref_slot_pool(ref_encoder(online, tokens, None, HEADS), slot_index, NUM_SLOTS, None)
    out = TargetEncoder.from_online(online)(tokens, slot_index, NUM_SLOTS)
    torch.testing.assert_close(out.s_y.double(), expected_s_y, rtol=RTOL, atol=ATOL)
    assert torch.equal(out.slot_mask, expected_mask)


def test_tokens_outside_every_slot_do_not_contribute() -> None:
    """Row 0 is all real and tokens 3, 6 have slot -1: slot 0 is the mean of tokens 0, 1, 7 only."""
    online = make_encoder()
    tokens, mask = make_batch()
    latents = ref_encoder(online, tokens, mask, HEADS)
    s_y = TargetEncoder.from_online(online)(tokens, make_slot_index(), NUM_SLOTS, mask).s_y
    expected = (latents[0, 0] + latents[0, 1] + latents[0, 7]) / 3
    torch.testing.assert_close(s_y[0, 0].double(), expected, rtol=RTOL, atol=ATOL)
    including_unassigned = latents[0, [0, 1, 3, 6, 7]].mean(dim=0)
    assert (including_unassigned - expected).abs().max() > 1e-3  # the check can tell the two apart


def test_padded_tokens_with_a_valid_slot_index_are_excluded() -> None:
    """Padded tokens never enter a slot mean, whatever their slot index or token id."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    tokens, mask = make_batch()
    slot_index = make_slot_index()
    latents = ref_encoder(online, tokens, mask, HEADS)
    s_y = target(tokens, slot_index, NUM_SLOTS, mask).s_y
    torch.testing.assert_close(s_y[1, 2].double(), (latents[1, 0] + latents[1, 1]) / 2, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s_y[2, 4].double(), (latents[2, 2] + latents[2, 7]) / 2, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s_y[2, 0].double(), latents[2, 6], rtol=RTOL, atol=ATOL)
    other = tokens.clone()
    other[~mask] = (tokens[~mask] + 1) % VOCAB
    assert torch.equal(target(other, slot_index, NUM_SLOTS, mask).s_y, s_y)


def test_empty_slots_are_zero_and_masked_out() -> None:
    """A slot with no real token (none at all, or only padded ones) is a zero vector with slot_mask False."""
    tokens, mask = make_batch()
    out = TargetEncoder.from_online(make_encoder())(tokens, make_slot_index(), NUM_SLOTS, mask)
    for b in range(3):
        for m in range(NUM_SLOTS):
            if (b, m) in EMPTY_SLOTS:
                assert not bool(out.slot_mask[b, m]), (b, m)
                assert torch.equal(out.s_y[b, m], torch.zeros(LATENT)), (b, m)
            else:
                assert bool(out.slot_mask[b, m]), (b, m)


def test_tokens_assigned_to_no_slot_give_all_empty_slots() -> None:
    """If every slot_index is -1, every slot is empty: s_y is all zeros and slot_mask all False."""
    tokens, mask = make_batch()
    out = TargetEncoder.from_online(make_encoder())(tokens, torch.full_like(tokens, -1), NUM_SLOTS, mask)
    assert torch.equal(out.s_y, torch.zeros(3, NUM_SLOTS, LATENT))
    assert not bool(out.slot_mask.any())


def _target_inputs(case: str) -> tuple[Tensor, Tensor, int, Tensor | None]:
    tokens, mask = make_batch()
    slot_index = make_slot_index()
    cases: dict[str, Callable[[], tuple[Tensor, Tensor, int, Tensor | None]]] = {
        "slot_index >= num_slots": lambda: (tokens, slot_index, 4, mask),
        "slot_index < -1": lambda: (tokens, slot_index.masked_fill(slot_index == 1, -2), NUM_SLOTS, mask),
        "float slot_index": lambda: (tokens, slot_index.float(), NUM_SLOTS, mask),
        "bool slot_index": lambda: (tokens, slot_index > 0, NUM_SLOTS, mask),
        "slot_index shape mismatch": lambda: (tokens, slot_index[:, :8], NUM_SLOTS, mask),
        "num_slots < 1": lambda: (tokens, torch.full_like(slot_index, -1), 0, mask),
        "float tokens": lambda: (tokens.float(), slot_index, NUM_SLOTS, mask),
        "longer than max_positions": lambda: (torch.randint(0, VOCAB, (3, MAX_POSITIONS + 1)),
                                              torch.zeros(3, MAX_POSITIONS + 1, dtype=torch.long), NUM_SLOTS, None),
        "all-padding row": lambda: (tokens, slot_index, NUM_SLOTS, mask & torch.tensor([[True], [True], [False]])),
    }
    return cases[case]()


@pytest.mark.parametrize("case", [
    "slot_index >= num_slots",
    "slot_index < -1",
    "float slot_index",
    "bool slot_index",
    "slot_index shape mismatch",
    "num_slots < 1",
    "float tokens",
    "longer than max_positions",
    "all-padding row",
])
def test_target_encoder_rejects_invalid_inputs(case: str) -> None:
    """Invalid slot assignments, slot counts and token tensors raise ValueError."""
    tokens, slot_index, num_slots, mask = _target_inputs(case)
    target = TargetEncoder.from_online(make_encoder())
    with pytest.raises(ValueError):
        target(tokens, slot_index, num_slots, mask)


# ---------------------------------------------------------------------------------------------
# f_theta_bar: no gradient, frozen, eval mode
# ---------------------------------------------------------------------------------------------
def test_target_parameters_are_frozen_at_construction() -> None:
    """Every parameter of f_theta_bar has requires_grad False, while f_theta's stay trainable."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    assert all(p.requires_grad for p in online.parameters())
    assert not any(p.requires_grad for p in target.parameters())
    target.assert_frozen()


def test_target_outputs_never_require_grad() -> None:
    """Even with grad enabled, trainable online weights and tokens produced inside a graph, s_y has no grad."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    _, mask = make_batch()
    scores = torch.randn(3, 9, VOCAB, requires_grad=True)
    tokens = (scores * 2.0).argmax(dim=-1)
    with torch.enable_grad():
        online_latents = online(tokens, mask)
        out = target(tokens, make_slot_index(), NUM_SLOTS, mask)
    assert online_latents.requires_grad  # the online branch of the same step is differentiable
    assert not out.s_y.requires_grad and out.s_y.grad_fn is None
    assert not out.slot_mask.requires_grad


def test_target_forward_runs_with_autograd_disabled() -> None:
    """Every submodule of f_theta_bar is evaluated with grad mode off, even when the caller has it on."""
    target = TargetEncoder.from_online(make_encoder())
    grad_modes: list[bool] = []
    handles = [m.register_forward_hook(lambda *_: grad_modes.append(torch.is_grad_enabled()))
               for m in (target.token_embedding, target.blocks[0], target.head)]
    tokens, mask = make_batch()
    with torch.enable_grad():
        target(tokens, make_slot_index(), NUM_SLOTS, mask)
    for handle in handles:
        handle.remove()
    assert grad_modes == [False, False, False]


def test_target_is_a_stop_gradient_in_a_regression_loss() -> None:
    """||pred - s_y||^2 backpropagates 2 (pred - s_y) into pred and nothing into theta_bar (sg of Eq. 5)."""
    target = TargetEncoder.from_online(make_encoder())
    tokens, mask = make_batch()
    s_y = target(tokens, make_slot_index(), NUM_SLOTS, mask).s_y
    prediction = torch.randn(s_y.shape, requires_grad=True)
    ((prediction - s_y) ** 2).sum().backward()
    assert all(p.grad is None for p in target.parameters())
    torch.testing.assert_close(prediction.grad, 2.0 * (prediction.detach() - s_y), rtol=RTOL, atol=ATOL)


def test_target_stays_in_eval_mode_after_train() -> None:
    """train(), train(True) and a parent's train() all leave f_theta_bar and its submodules in eval mode."""
    target = TargetEncoder.from_online(make_encoder(dropout=0.5))
    assert target.train() is target
    target.train(True)
    parent = nn.ModuleDict({"target": target, "other": nn.Linear(2, 2)})
    parent.train()
    assert parent["other"].training
    assert not any(module.training for module in target.modules())


def test_target_output_is_deterministic_although_dropout_is_configured() -> None:
    """With dropout 0.5 the online encoder is stochastic in train mode; the target never is."""
    online = make_encoder(dropout=0.5)
    target = TargetEncoder.from_online(online)
    target.train()
    online.train()
    tokens, mask = make_batch()
    slot_index = make_slot_index()
    assert not torch.equal(online(tokens, mask), online(tokens, mask))  # dropout is active online
    first = target(tokens, slot_index, NUM_SLOTS, mask).s_y
    assert torch.equal(target(tokens, slot_index, NUM_SLOTS, mask).s_y, first)


def test_requires_grad_true_raises_runtime_error() -> None:
    """requires_grad_(True) raises; requires_grad_(False) is allowed and keeps the target frozen."""
    target = TargetEncoder.from_online(make_encoder())
    with pytest.raises(RuntimeError):
        target.requires_grad_(True)
    with pytest.raises(RuntimeError):
        target.requires_grad_()
    assert target.requires_grad_(False) is target
    assert not any(p.requires_grad for p in target.parameters())


def test_assert_frozen_raises_after_manual_unfreeze() -> None:
    """Setting one parameter's requires_grad directly is caught by assert_frozen and by forward."""
    target = TargetEncoder.from_online(make_encoder())
    target.head.weight.requires_grad = True
    with pytest.raises(RuntimeError):
        target.assert_frozen()
    tokens, mask = make_batch()
    with pytest.raises(RuntimeError):
        target(tokens, make_slot_index(), NUM_SLOTS, mask)


def test_unfreezing_through_a_parent_module_is_caught_at_the_next_forward() -> None:
    """A parent's requires_grad_(True) (e.g. unfreezing a whole planner) bypasses the override; forward refuses."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    nn.ModuleDict({"online": online, "target": target}).requires_grad_(True)
    tokens, mask = make_batch()
    with pytest.raises(RuntimeError):
        target(tokens, make_slot_index(), NUM_SLOTS, mask)


# ---------------------------------------------------------------------------------------------
# f_theta_bar: construction from f_theta
# ---------------------------------------------------------------------------------------------
def test_from_online_copies_weights_exactly_with_the_same_names() -> None:
    """theta_bar <- theta at initialisation: same parameter names in the same order, bit-identical values."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    online_params = list(online.named_parameters())
    target_params = list(target.named_parameters())
    assert [name for name, _ in target_params] == [name for name, _ in online_params]
    for (name, theta_bar), (_, theta) in zip(target_params, online_params, strict=True):
        assert torch.equal(theta_bar, theta), name
    assert target.state_dict().keys() == online.state_dict().keys()


def test_from_online_makes_an_independent_copy() -> None:
    """The target owns its storage: a later change to theta leaves theta_bar unchanged."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    before = {name: p.clone() for name, p in target.named_parameters()}
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    for name, p in target.named_parameters():
        assert torch.equal(p, before[name]), name


def test_from_online_target_reproduces_online_token_latents() -> None:
    """Right after from_online, f_theta_bar's token latents equal f_theta's exactly."""
    online = make_encoder()
    target = TargetEncoder.from_online(online)
    tokens, mask = make_batch()
    assert torch.equal(target.encode(tokens, mask), online.encode(tokens, mask))


def test_from_online_preserves_dtype_and_hyperparameters() -> None:
    """A float64 online encoder gives a float64 target with the same constructor arguments."""
    online = copy.deepcopy(make_encoder()).double()
    target = TargetEncoder.from_online(online)
    assert all(p.dtype == torch.float64 for p in target.parameters())
    assert target.hyperparameters() == online.hyperparameters()


# ---------------------------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------------------------
def test_encoders_on_accelerator_match_cpu() -> None:
    """On CUDA/ROCm/MPS, f_theta and f_theta_bar run on the device and agree with the CPU results.

    Tolerance: accelerator kernels (fused attention, other reduction orders) are not bitwise equal to the
    CPU path; their float32 differences are O(1e-6) relative, so rtol=1e-4, atol=1e-5 is used.
    """
    device = accelerator()
    if device is None:
        pytest.skip("no CUDA/ROCm/MPS device available")
    online = make_encoder()
    tokens, mask = make_batch()
    slot_index = make_slot_index()
    expected_s_x = online(tokens, mask)
    expected_s_y = TargetEncoder.from_online(online)(tokens, slot_index, NUM_SLOTS, mask)
    moved = copy.deepcopy(online).to(device)
    s_x = moved(tokens.to(device), mask.to(device))
    assert s_x.device.type == device.type
    torch.testing.assert_close(s_x.cpu(), expected_s_x, rtol=1e-4, atol=1e-5)
    assert torch.equal(s_x.cpu()[~mask], torch.zeros(int((~mask).sum()), LATENT))
    target = TargetEncoder.from_online(moved)
    assert all(p.device.type == device.type and not p.requires_grad for p in target.parameters())
    out = target(tokens.to(device), slot_index.to(device), NUM_SLOTS, mask.to(device))
    torch.testing.assert_close(out.s_y.cpu(), expected_s_y.s_y, rtol=1e-4, atol=1e-5)
    assert torch.equal(out.slot_mask.cpu(), expected_s_y.slot_mask)
