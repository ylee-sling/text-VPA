# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
backbone.py -- Gemma 4 E2B as the Text-VPA realizer without a plan (Text-VPA Sec. 3, Stage 0 of Sec. 6).

One pretrained decoder is used in two ways.

    frozen backbone B_0   reads the prompt once, causally, exactly as released: ``prefill`` stores the keys and
                          values of its KV-storing blocks (the lower 15 of Gemma 4 E2B's 35; the upper 20 share
                          them) in a ``PrefixCache``. No adapter acts here, so the cache is the one any other role
                          of B_0 (planner encoder, verifier, autoregressive decoding) computes.
    realizer B_psi        refines a canvas of N positions in parallel: ``canvas_hidden`` runs the same blocks on
                          the canvas only, attending to the cached prompt and, bidirectionally, to the canvas.

What the realizer adds to the backbone, all of it on canvas positions only:

    LoRA             W x + (alpha / r) B A x on the attention and MLP projections; B = 0 at initialisation
    mask embedding   one trainable vector in place of the token embedding at [MASK] positions (the tied
                     embedding matrix stays frozen); initialised to the tokenizer's <mask> embedding
    (t, h) modulation  y (1 + scale) + shift on the outputs of the two pre-norms of every block (before
                     self-attention and before the MLP); "adaLN-single" as in ``models/realizers/backbone.py``:
                     one zero-initialised projection of c = MLP_t(sin t) + MLP_h(sin h) plus zero per-block tables
    attention mask   canvas rows see the valid prompt positions and every canvas position; in local
                     ("sliding_attention") blocks only positions at distance < W on either side (W = 512), which
                     on the left is the backbone's own causal window

so at initialisation B_psi is B_0 with a bidirectional canvas, and with ``mode="causal"`` it is B_0 exactly
(tested against ``Gemma4ForCausalLM``).

The AR shift. The backbone's output at position i predicts token i + 1, and the realizer keeps that alignment:
the canvas holds [a, w^1, ..., w^N] where the anchor a is the last prompt token, always clamped, and the output
at canvas index j - 1 is the posterior of w^j. The prefix cache therefore covers the prompt without its last
token.

Layout of one sample (absolute positions; the prefix is left-padded in a batch):

    prefix  c_1 .. c_{P-1}        positions 0 .. P-2      frozen, cached
    canvas  a = c_P, w^1 .. w^N   positions P-1 .. P+N-1  realizer

Shapes: B batch, P padded prefix length, C = N + 1 canvas length, d hidden size, S vocabulary size.
"""

from __future__ import annotations

from collections import UserDict
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from text_vpa.models.realizers.backbone import timestep_embedding

__all__ = [
    "AdapterState",
    "LoRALinear",
    "ModulatedNorm",
    "PrefixCache",
    "RealizerSpec",
    "Stage0Realizer",
    "canvas_masks",
    "left_pad",
    "load_gemma4",
    "prefill_masks",
]

LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class RealizerSpec:
    """Hyperparameters of the realizer's additions (stored in every checkpoint)."""

    mask_id: int
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_targets: tuple[str, ...] = LORA_TARGETS
    time_dim: int = 256            # width of the sinusoidal features and of c
    modulation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RealizerSpec:
        return cls(**{**data, "lora_targets": tuple(data.get("lora_targets", LORA_TARGETS))})


class AdapterState:
    """Switch shared by every adapter of one realizer: off while the frozen backbone runs (prefill), on for the
    canvas. ``modulation`` is ``[B, L, 4, d]`` (shift and scale for the two pre-norms of each block) or None.

    ``plan`` and ``residual`` belong to the planner stage (``text_vpa.stage1.bridge``): the cached keys and
    values of a plan while a canvas block runs with one, and that block's input. Stage 0 leaves both at None.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.modulation: Tensor | None = None
        self.plan: Any = None
        self.residual: Tensor | None = None


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a low-rank update that acts only while ``state.active``.

    Shapes: ``[..., in]`` -> ``[..., out]``. ``lora_a [r, in]`` (Kaiming-uniform), ``lora_b [out, r]`` (zero).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, state: AdapterState) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("the LoRA rank must be >= 1")
        self.base = base
        self.scale = alpha / rank
        self._state = [state]                                    # in a list: not a submodule, not in state_dict
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if not self._state[0].active:
            return y
        update = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return y + self.scale * update.to(y.dtype)


class ModulatedNorm(nn.Module):
    """A frozen pre-norm whose output is modulated by (t, h) on the canvas: y (1 + scale) + shift.

    ``slot`` 0 is the norm before self-attention, 1 the norm before the MLP. Shapes: ``[B, T, d]`` -> same.
    """

    def __init__(self, norm: nn.Module, state: AdapterState, layer_idx: int, slot: int) -> None:
        super().__init__()
        self.norm = norm
        self._state = [state]
        self.layer_idx = layer_idx
        self.slot = slot

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        state = self._state[0]
        if not state.active or state.modulation is None:
            return y
        m = state.modulation[:, self.layer_idx, 2 * self.slot: 2 * self.slot + 2].to(y.dtype)    # [B, 2, d]
        return y * (1.0 + m[:, 1:2]) + m[:, 0:1]


# ---------------------------------------------------------------------------------------------
# Attention masks (bool, True = may attend), shape [B, 1, Q, KV]
# ---------------------------------------------------------------------------------------------
def left_pad(rows: list[list[int]], pad_id: int, device: torch.device | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Left-pad id lists to one length. Returns ids ``[B, P]``, valid ``[B, P]`` (bool) and positions ``[B, P]``
    (0 .. len-1 on the real tokens, 0 on the padding)."""
    if not rows or any(len(r) == 0 for r in rows):
        raise ValueError("every prefix needs at least one token")
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    pos = torch.zeros((len(rows), width), dtype=torch.long)
    for b, row in enumerate(rows):
        n = len(row)
        ids[b, width - n:] = torch.tensor(row, dtype=torch.long)
        valid[b, width - n:] = True
        pos[b, width - n:] = torch.arange(n)
    return ids.to(device), valid.to(device), pos.to(device)


def prefill_masks(pos: Tensor, valid: Tensor, window: int) -> dict[str, Tensor]:
    """The backbone's own causal masks on a left-padded prefix.

    full_attention: key valid and not after the query. sliding_attention: additionally pos_q - pos_k < W
    (transformers' ``sliding_window_overlay``). A padding query attends to itself only (its output is unused; a
    row without any key would be NaN).
    """
    p = pos.shape[1]
    idx = torch.arange(p, device=pos.device)
    causal = (idx[None, :] <= idx[:, None])[None] & valid[:, None, :]                        # [B, Q, K]
    near = (pos[:, :, None] - pos[:, None, :]) < window
    eye = torch.eye(p, dtype=torch.bool, device=pos.device)[None]
    pad_query = ~valid[:, :, None]
    full = torch.where(pad_query, eye, causal)
    sliding = torch.where(pad_query, eye, causal & near)
    return {"full_attention": full[:, None], "sliding_attention": sliding[:, None]}


def canvas_masks(prefix_pos: Tensor, prefix_valid: Tensor, canvas_pos: Tensor, window: int,
                 mode: str = "bidirectional") -> dict[str, Tensor]:
    """Masks of the canvas rows over the keys [prefix; canvas].

    ``bidirectional``: every canvas position; local blocks keep |pos_q - pos_k| < W on both sides.
    ``causal``: the backbone's own masks (canvas keys not after the query), for tests and ablations.
    Shapes: prefix ``[B, P]``, canvas ``[B, C]`` -> ``[B, 1, C, P + C]``.
    """
    if mode not in ("bidirectional", "causal"):
        raise ValueError("mode must be 'bidirectional' or 'causal'")
    b, c = canvas_pos.shape
    to_prefix = prefix_valid[:, None, :].expand(b, c, -1)                                    # [B, C, P]
    prefix_near = (canvas_pos[:, :, None] - prefix_pos[:, None, :]) < window
    delta = canvas_pos[:, :, None] - canvas_pos[:, None, :]                                  # [B, C, C] q - k
    to_canvas = torch.ones_like(delta, dtype=torch.bool) if mode == "bidirectional" else delta >= 0
    canvas_near = delta.abs() < window
    full = torch.cat([to_prefix, to_canvas], dim=2)
    sliding = torch.cat([to_prefix & prefix_near, to_canvas & canvas_near], dim=2)
    return {"full_attention": full[:, None], "sliding_attention": sliding[:, None]}


def _for_implementation(masks: dict[str, Tensor], implementation: str, dtype: torch.dtype) -> dict[str, Tensor]:
    """Boolean masks for sdpa; additive masks (0 / most negative) for the eager implementation."""
    if implementation != "eager":
        return masks
    low = torch.finfo(dtype).min
    return {k: torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, low) for k, m in masks.items()}


# ---------------------------------------------------------------------------------------------
# The prompt cache
# ---------------------------------------------------------------------------------------------
class _Recorder:
    """Cache protocol of ``Gemma4TextAttention`` for the prefill: remember each block's keys and values."""

    def __init__(self) -> None:
        self.keys: dict[int, Tensor] = {}
        self.values: dict[int, Tensor] = {}

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        self.keys[layer_idx], self.values[layer_idx] = key_states, value_states
        return key_states, value_states


class PrefixCache:
    """Keys and values of the frozen backbone on the prompt prefix, read-only.

    ``update`` (the cache protocol of ``Gemma4TextAttention``) returns [prefix; new] and stores nothing, so one
    cache serves every realizer step, a replay under activation checkpointing, and any other reader.

    keys[l], values[l]: ``[B, kv_heads, P, head_dim_l]`` for each KV-storing block l; pos, valid: ``[B, P]``.
    """

    def __init__(self, keys: dict[int, Tensor], values: dict[int, Tensor], pos: Tensor, valid: Tensor) -> None:
        self.keys, self.values, self.pos, self.valid = keys, values, pos, valid

    @property
    def lengths(self) -> Tensor:
        """Real prefix length of each row, ``[B]``."""
        return self.valid.sum(dim=1)

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        k, v = self.keys[layer_idx], self.values[layer_idx]
        return torch.cat([k.to(key_states.dtype), key_states], dim=2), torch.cat([v.to(value_states.dtype),
                                                                                    value_states], dim=2)

    def expand(self, rows: int) -> PrefixCache:
        """A cache of one prompt repeated ``rows`` times (views, no copy)."""
        if self.pos.shape[0] != 1:
            raise ValueError("expand needs a cache of one prompt")
        return PrefixCache({i: k.expand(rows, -1, -1, -1) for i, k in self.keys.items()},
                           {i: v.expand(rows, -1, -1, -1) for i, v in self.values.items()},
                           self.pos.expand(rows, -1), self.valid.expand(rows, -1))


# ---------------------------------------------------------------------------------------------
# The realizer
# ---------------------------------------------------------------------------------------------
class Stage0Realizer(nn.Module):
    """Gemma 4 text decoder + canvas-only adapters. The backbone's own parameters are frozen.

    Args:
        text_model: a ``Gemma4TextModel`` (its blocks are modified in place: LoRA and modulated pre-norms, both
            inert unless the adapters are switched on by ``canvas_hidden``).
        lm_head: the (tied) output projection ``d -> S``.
        spec: ``RealizerSpec``.
    """

    def __init__(self, text_model: nn.Module, lm_head: nn.Linear, spec: RealizerSpec) -> None:
        super().__init__()
        cfg = text_model.config
        self.spec = spec
        self.text = text_model
        self.lm_head = lm_head
        self.hidden_size: int = cfg.hidden_size
        self.vocab_size: int = cfg.vocab_size
        self.window: int = cfg.sliding_window
        self.layer_types: list[str] = list(cfg.layer_types)
        self.num_layers: int = cfg.num_hidden_layers
        self.softcap: float | None = cfg.final_logit_softcapping
        self.pad_id: int = cfg.pad_token_id or 0
        # Blocks below this index compute and store keys and values; the blocks above reuse them.
        self.num_kv_layers: int = self.num_layers - int(getattr(cfg, "num_kv_shared_layers", 0) or 0)
        if not 0 <= spec.mask_id < self.vocab_size:
            raise ValueError(f"mask_id {spec.mask_id} is outside the vocabulary")
        for p in self.text.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        self.state = AdapterState()
        self.num_lora: int = 0
        for i, layer in enumerate(self.text.layers):
            for owner_name in ("self_attn", "mlp"):
                owner = getattr(layer, owner_name)
                for target in spec.lora_targets:
                    base = getattr(owner, target, None)
                    if isinstance(base, nn.Linear):
                        wrapped = LoRALinear(base, spec.lora_rank, spec.lora_alpha, self.state)
                        setattr(owner, target, wrapped)
                        self.num_lora += 1
            if spec.modulation:
                layer.input_layernorm = ModulatedNorm(layer.input_layernorm, self.state, i, 0)
                layer.pre_feedforward_layernorm = ModulatedNorm(layer.pre_feedforward_layernorm, self.state, i, 1)

        with torch.no_grad():
            init = self.text.embed_tokens(torch.tensor([spec.mask_id], device=self.text.embed_tokens.weight.device))
        self.mask_embedding = nn.Parameter(init[0].float().clone())                      # [d], already scaled
        if spec.modulation:
            dim = spec.time_dim
            self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.h_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.mod_proj = nn.Linear(dim, 4 * self.hidden_size)
            nn.init.zeros_(self.mod_proj.weight)
            nn.init.zeros_(self.mod_proj.bias)
            self.mod_tables = nn.Parameter(torch.zeros(self.num_layers, 4, self.hidden_size))
        self.passes: dict[str, int] = {"prefill": 0, "canvas": 0}

    # -- construction and persistence ----------------------------------------------------------
    @classmethod
    def from_pretrained_model(cls, model: nn.Module, spec: RealizerSpec) -> Stage0Realizer:
        """Build on a loaded ``Gemma4ForConditionalGeneration`` or ``Gemma4ForCausalLM`` (shares its weights)."""
        inner = model.model
        return cls(getattr(inner, "language_model", inner), model.lm_head, spec)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state(self) -> dict[str, Tensor]:
        """Only what Stage 0 trains (LoRA, mask embedding, modulation): the checkpoint's model part."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: p.detach().cpu() for n, p in self.named_parameters() if n in names}

    def load_trainable_state(self, state: dict[str, Tensor]) -> None:
        own = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(own) != set(state):
            missing, extra = sorted(set(own) - set(state))[:3], sorted(set(state) - set(own))[:3]
            raise ValueError(f"checkpoint and realizer disagree (missing {missing}, unexpected {extra})")
        with torch.no_grad():
            for name, p in own.items():
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))

    def train(self, mode: bool = True) -> Stage0Realizer:
        super().train(mode)
        self.text.eval()                                         # the backbone has no train-time behaviour to use
        return self

    # -- pieces --------------------------------------------------------------------------------
    @property
    def _implementation(self) -> str:
        return getattr(self.text.config, "_attn_implementation", None) or "sdpa"

    def _embed(self, ids: Tensor, is_mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Token embeddings (the mask vector at [MASK] positions) and the per-layer inputs ``[B, T, L, d_ple]``."""
        if is_mask is not None:
            ids = torch.where(is_mask, torch.full_like(ids, self.spec.mask_id), ids)
        embeds = self.text.embed_tokens(ids)
        if is_mask is not None:
            embeds = torch.where(is_mask[..., None], self.mask_embedding.to(embeds.dtype), embeds)
        per_layer = None
        if getattr(self.text, "hidden_size_per_layer_input", 0):
            per_layer = self.text.project_per_layer_inputs(embeds, self.text.get_per_layer_inputs(ids, None))
        return embeds, per_layer

    def _block(self, hidden: Tensor, layer_input: Tensor | None, *, index: int, active: bool,
               modulation: Tensor | None, plan: Any = None, **kwargs: Any) -> Tensor:
        """Block ``index`` with the adapter switch set for the duration of the call. The switch is an argument,
        not ambient state, because activation checkpointing calls this again during the backward pass. The
        same holds for ``plan`` (the planner stage's cached keys and values; None in Stage 0)."""
        previous = (self.state.active, self.state.modulation, self.state.plan)
        self.state.active, self.state.modulation, self.state.plan = active, modulation, plan
        try:
            return self.text.layers[index](hidden, layer_input, **kwargs)
        finally:
            self.state.active, self.state.modulation, self.state.plan = previous
            self.state.residual = None

    def _run(self, hidden: Tensor, per_layer: Tensor | None, position_ids: Tensor, masks: dict[str, Tensor],
             cache: Any, num_layers: int, checkpointing: bool, active: bool = False,
             modulation: Tensor | None = None, plan: Any = None) -> Tensor:
        """Blocks 0 .. num_layers-1 of the backbone (the loop of ``Gemma4TextModel.forward``)."""
        masks = _for_implementation(masks, self._implementation, hidden.dtype)
        rotary = {kind: self.text.rotary_emb(hidden, position_ids, kind) for kind in set(self.layer_types)}
        shared: UserDict = UserDict()
        for i in range(num_layers):
            kind = self.layer_types[i]
            block = partial(self._block, index=i, active=active, modulation=modulation, plan=plan,
                            shared_kv_states=shared, position_embeddings=rotary[kind],
                            attention_mask=masks[kind], position_ids=position_ids, past_key_values=cache)
            layer_input = per_layer[:, :, i, :] if per_layer is not None else None
            if checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, layer_input, use_reentrant=False)
            else:
                hidden = block(hidden, layer_input)
        return hidden

    def modulation(self, t: Tensor, h: Tensor) -> Tensor | None:
        """``[B]``, ``[B]`` -> ``[B, L, 4, d]``: (shift, scale) for the two pre-norms of each block; zero at init."""
        if not self.spec.modulation:
            return None
        dim = self.spec.time_dim
        c = self.t_mlp(timestep_embedding(t, dim)) + self.h_mlp(timestep_embedding(h, dim))
        shared = self.mod_proj(F.silu(c)).view(-1, 1, 4, self.hidden_size)
        return shared + self.mod_tables[None]

    # -- the two passes ------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: Tensor, valid: Tensor, pos: Tensor) -> PrefixCache:
        """The frozen backbone on the (left-padded) prompt prefix: one pass, adapters off.

        Only the KV-storing blocks run: the blocks above them reuse their keys and values, and the realizer
        never reads the prefix's hidden states. Shapes: ``[B, P]`` each.
        """
        embeds, per_layer = self._embed(ids, None)
        recorder = _Recorder()
        self._run(embeds, per_layer, pos, prefill_masks(pos, valid, self.window), recorder, self.num_kv_layers,
                  checkpointing=False)
        self.passes["prefill"] += 1
        return PrefixCache(recorder.keys, recorder.values, pos, valid)

    def canvas_hidden(self, cache: PrefixCache, canvas_ids: Tensor, is_mask: Tensor, t: Tensor, h: Tensor,
                      mode: str = "bidirectional", adapters: bool = True, checkpointing: bool = False,
                      plan: Any = None) -> Tensor:
        """One realizer pass over the canvas.

        Args:
            cache: the prompt prefix's keys and values (``prefill``).
            canvas_ids: ``[B, C]`` int64, [anchor, w_t^1 .. w_t^N]; ids at masked positions are ignored.
            is_mask: ``[B, C]`` bool, True where the position holds [MASK] (never the anchor).
            t, h: ``[B]`` flow time and step size.
            mode: ``bidirectional`` (the realizer) or ``causal`` (the backbone's masks).
            adapters: False runs the frozen backbone on the canvas (tests, ablations).
            plan: the planner stage's ``PlanCache`` (``text_vpa.stage1.bridge``; the bridge must be attached):
                every block then adds its gated cross-attention to the plan on the canvas. None: Stage 0.

        Returns:
            ``[B, C, d]`` final hidden states (after the last norm); index j - 1 predicts canvas token j.
        """
        if bool(is_mask[:, 0].any()):
            raise ValueError("the anchor (canvas index 0) is clamped and can never be [MASK]")
        if plan is not None and not adapters:
            raise ValueError("a plan acts through the realizer's adapters; adapters=False is the frozen backbone")
        canvas_pos = cache.lengths[:, None] + torch.arange(canvas_ids.shape[1], device=canvas_ids.device)[None]
        masks = canvas_masks(cache.pos, cache.valid, canvas_pos, self.window, mode)
        modulation = self.modulation(t, h) if adapters else None
        embeds, per_layer = self._embed(canvas_ids, is_mask)
        hidden = self._run(embeds, per_layer, canvas_pos, masks, cache, self.num_layers, checkpointing,
                           active=adapters, modulation=modulation, plan=plan)
        hidden = self.text.norm(hidden)
        self.passes["canvas"] += 1
        return hidden

    @torch.no_grad()
    def frozen_states(self, cache: PrefixCache, canvas_ids: Tensor, num_layers: int) -> Tensor:
        """The frozen backbone reading a canvas causally after its prompt: hidden states after the first
        ``num_layers`` blocks, adapters off, no final norm (the planner stage's latent oracle).

        Args:
            cache: the prompt prefix's keys and values (``prefill``).
            canvas_ids: ``[B, C]`` int64, [anchor, y^1 .. y^N]: a whole program, no [MASK].
            num_layers: blocks to run, 1 .. ``num_kv_layers`` (the blocks above reuse keys and values that a
                partial pass does not compute for the canvas).

        Returns:
            ``[B, C, d]``; index j has read canvas_ids[:, :j + 1], so index k + 1 is the state after token y^(k+1).
        """
        if not 1 <= num_layers <= self.num_kv_layers:
            raise ValueError(f"num_layers must be in 1 .. {self.num_kv_layers}")
        canvas_pos = cache.lengths[:, None] + torch.arange(canvas_ids.shape[1], device=canvas_ids.device)[None]
        masks = canvas_masks(cache.pos, cache.valid, canvas_pos, self.window, "causal")
        embeds, per_layer = self._embed(canvas_ids, None)
        hidden = self._run(embeds, per_layer, canvas_pos, masks, cache, num_layers, checkpointing=False)
        self.passes["frozen"] = self.passes.get("frozen", 0) + 1     # the planner stage's count only
        return hidden

    def logits(self, hidden: Tensor) -> Tensor:
        """``[..., d]`` -> ``[..., S]`` float32 logits with the backbone's final soft cap (30 tanh(x / 30))."""
        logits = self.lm_head(hidden).float()
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def _chunk_nll(self, hidden: Tensor, targets: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.logits(hidden)
        nll = F.cross_entropy(logits, targets, reduction="none")
        return (nll * weights).sum(), (logits.argmax(dim=-1) == targets).sum().detach()

    def weighted_nll(self, hidden: Tensor, targets: Tensor, weights: Tensor, chunk: int = 512
                     ) -> tuple[Tensor, Tensor]:
        """sum_m weights_m * (-log p(targets_m | hidden_m)) and the number of argmax hits.

        The S = 262,144 logits of a position are 1 MB in float32, so positions are processed ``chunk`` at a
        time and each chunk is checkpointed: its logits are freed after the forward pass and recomputed in the
        backward pass. Shapes: hidden ``[M, d]``, targets ``[M]`` int64, weights ``[M]``.
        """
        total = hidden.new_zeros((), dtype=torch.float32)
        hits = torch.zeros((), dtype=torch.long, device=hidden.device)
        for start in range(0, hidden.shape[0], chunk):
            part = (hidden[start:start + chunk], targets[start:start + chunk], weights[start:start + chunk])
            if hidden.requires_grad:
                nll, hit = checkpoint(self._chunk_nll, *part, use_reentrant=False)
            else:
                nll, hit = self._chunk_nll(*part)
            total = total + nll
            hits = hits + hit
        return total, hits


def load_gemma4(model_id: str, dtype: torch.dtype = torch.bfloat16, revision: str | None = None,
                token: str | None = None) -> tuple[nn.Module, Any]:
    """The released checkpoint as ``Gemma4ForConditionalGeneration`` (on the CPU, eval mode) and its tokenizer.

    The whole released model is loaded with its own class, so the autoregressive baseline is the model's own
    ``generate``; the realizer shares the text decoder's weights (``Stage0Realizer.from_pretrained_model``).
    """
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=dtype, revision=revision, token=token)
    return model.eval(), tokenizer
