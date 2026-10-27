# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
encoders.py -- pass 1 of PR-JEPA: the context encoder f_theta and its EMA copy f_theta_bar (Sec. 3.1).

    s_x = f_theta(c)       in R^{M_c x d}   one latent per context token (specification, term sheet)
    s_y = f_theta_bar(y)   in R^{M x d}     one latent per structural slot of the target view y

f_theta: token embedding + learned absolute position embedding (width d_enc), L pre-norm
bidirectional transformer blocks, a final LayerNorm, and a linear head to the latent width d.

f_theta_bar is the same network with EMA weights (Eq. 4, updated by ``EMATargetEncoder``), applied
to the tokens of the structural view y (a serialised AST or clause outline). The data pipeline
assigns every token of y to a slot (``slot_index``; -1 for tokens outside every slot), and slot m's
latent is the mean of its tokens' outputs:

    s_y[b, m] = (1 / |T_bm|) sum_{t in T_bm} f_theta_bar(y)[b, t],
    T_bm = {t : slot_index[b, t] = m and token t is real}.

Mean pooling has no parameters, so f_theta_bar has exactly the parameters of f_theta, as the EMA
update theta_bar <- m theta_bar + (1 - m) theta requires (the two share parameter names, so the
update pairs them by name). f_theta_bar never receives gradients: its parameters are frozen, its
forward runs under ``torch.no_grad`` (the stop-gradient sg of Eq. 5), it stays in eval mode, and
``requires_grad_(True)`` raises. It is used in training only (Fig. 1, dashed branch).

Padded positions are returned as exact zeros, so downstream statistics cannot pick them up by mistake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from text_vpa.models.layers import TransformerBlock

if TYPE_CHECKING:
    from text_vpa.configs.schema import PlannerConfig

__all__ = ["ContextEncoder", "TargetEncoder", "TargetOutput"]


def _check_tokens(tokens: Tensor, mask: Tensor | None, max_positions: int, what: str) -> None:
    if tokens.dim() != 2 or tokens.dtype.is_floating_point or tokens.dtype == torch.bool:
        raise ValueError(f"expected integer {what} of shape [B, L], got {tuple(tokens.shape)} {tokens.dtype}")
    if tokens.shape[1] > max_positions:
        raise ValueError(f"{what} length {tokens.shape[1]} exceeds max_positions={max_positions}")
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != tokens.shape:
            raise ValueError(f"expected a bool attention_mask of shape {tuple(tokens.shape)}, got "
                             f"{tuple(mask.shape)} {mask.dtype}")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every row of attention_mask needs at least one real token")


class ContextEncoder(nn.Module):
    """f_theta: bidirectional transformer from token ids to latents.

    Args:
        vocab_size: size of the tokenizer of c (and y).
        latent_dim: d, width of the output latents.
        dim: internal transformer width d_enc.
        num_layers, num_heads, mlp_ratio, dropout: transformer hyperparameters.
        max_positions: size of the learned position table (max sequence length).

    Shapes:
        ``forward(tokens [B, L] int64, attention_mask [B, L] bool or None)`` -> ``s_x [B, L, d]``
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_positions: int = 2048,
    ) -> None:
        super().__init__()
        self._hparams: dict[str, Any] = dict(
            vocab_size=vocab_size, latent_dim=latent_dim, dim=dim, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, max_positions=max_positions,
        )
        self.vocab_size: int = vocab_size
        self.latent_dim: int = latent_dim
        self.dim: int = dim
        self.max_positions: int = max_positions
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_positions, dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers))
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, latent_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    @classmethod
    def from_config(cls, cfg: PlannerConfig) -> ContextEncoder:
        """Build f_theta from ``PlannerConfig``."""
        return cls(
            vocab_size=cfg.vocab_size, latent_dim=cfg.latent_dim, dim=cfg.encoder.dim,
            num_layers=cfg.encoder.num_layers, num_heads=cfg.encoder.num_heads, mlp_ratio=cfg.encoder.mlp_ratio,
            dropout=cfg.encoder.dropout, max_positions=cfg.max_positions,
        )

    def hyperparameters(self) -> dict[str, Any]:
        """Constructor arguments, used to build f_theta_bar with the same architecture."""
        return dict(self._hparams)

    def encode(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Token-level latents ``[B, L] -> [B, L, d]``, exact zeros at padded tokens."""
        _check_tokens(tokens, attention_mask, self.max_positions, "tokens")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed_dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x, attention_mask)
        latents = self.head(self.final_norm(x))                                   # [B, L, d]
        if attention_mask is not None:
            latents = latents.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        return latents

    def forward(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Pass 1: s_x = f_theta(c). ``[B, M_c]`` -> ``[B, M_c, d]``."""
        return self.encode(tokens, attention_mask)


class TargetOutput(NamedTuple):
    """Output of ``TargetEncoder.forward``."""

    s_y: Tensor        # [B, M, d]   slot latents, zeros at empty slots
    slot_mask: Tensor  # [B, M] bool True where the slot received at least one real token


class TargetEncoder(ContextEncoder):
    """f_theta_bar: frozen EMA copy of f_theta, pooled to one latent per structural slot.

    Build it with ``TargetEncoder.from_online(context_encoder)``; update it with
    ``EMATargetEncoder``. Training only.

    Shapes:
        ``forward(tokens [B, T], slot_index [B, T] int64 in {-1, 0, ..., M - 1}, num_slots = M,
        attention_mask [B, T] bool or None)`` -> ``TargetOutput(s_y [B, M, d], slot_mask [B, M])``
    """

    def __init__(self, **hparams: Any) -> None:
        super().__init__(**hparams)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        nn.Module.train(self, False)

    @classmethod
    def from_online(cls, online: ContextEncoder) -> TargetEncoder:
        """theta_bar <- theta at initialisation (the EMA starts from the online weights)."""
        reference = next(online.parameters())
        target = cls(**online.hyperparameters()).to(device=reference.device, dtype=reference.dtype)
        target.load_state_dict(online.state_dict())
        return target

    def requires_grad_(self, requires_grad: bool = True) -> TargetEncoder:
        if requires_grad:
            raise RuntimeError("f_theta_bar receives no gradients; it changes only through the EMA update (Eq. 4)")
        return super().requires_grad_(False)

    def train(self, mode: bool = True) -> TargetEncoder:
        """Stays in eval mode: targets are deterministic (no dropout) whatever the trainer's mode."""
        return super().train(False)

    def assert_frozen(self) -> None:
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(f"f_theta_bar must be frozen, but these parameters require grad: {trainable[:5]}")

    @torch.no_grad()
    def forward(  # type: ignore[override]
        self,
        tokens: Tensor,
        slot_index: Tensor,
        num_slots: int,
        attention_mask: Tensor | None = None,
    ) -> TargetOutput:
        """s_y = sg(f_theta_bar(y)), mean-pooled per slot.

        Args:
            tokens: ``[B, T]`` tokens of the structural view y.
            slot_index: ``[B, T]`` int64 slot of each token (-1: belongs to no slot).
            num_slots: M, the number of target slots (must match the predictor's M).
            attention_mask: ``[B, T]`` bool, True at real tokens.
        """
        self.assert_frozen()
        if slot_index.shape != tokens.shape or slot_index.dtype.is_floating_point or slot_index.dtype == torch.bool \
                or slot_index.is_complex():
            raise ValueError(f"expected integer slot_index of shape {tuple(tokens.shape)}, got "
                             f"{tuple(slot_index.shape)} {slot_index.dtype}")
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if bool((slot_index >= num_slots).any()) or bool((slot_index < -1).any()):
            raise ValueError(f"slot_index entries must lie in {{-1, 0, ..., {num_slots - 1}}}")
        h = self.encode(tokens, attention_mask)                                  # [B, T, d]
        b, _, d = h.shape
        member = slot_index >= 0
        if attention_mask is not None:
            member = member & attention_mask
        weight = member.to(h.dtype)                                              # [B, T]
        index = slot_index.clamp_min(0)
        sums = h.new_zeros(b, num_slots, d).scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, d), h * weight.unsqueeze(-1))
        counts = h.new_zeros(b, num_slots).scatter_add_(1, index, weight)       # |T_bm|
        s_y = sums / counts.clamp_min(1.0).unsqueeze(-1)
        return TargetOutput(s_y, counts > 0)
