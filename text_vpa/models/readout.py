# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
readout.py -- the plan's slot decoder: symbolic content read out of Z_target (PR-JEPA Sec. 3.3).

Plan-mediated energies need symbols, not latents: "the plan's slot decoder predicts a width b_v for each
variable v" (HLS-C), "only identifiers the plan declares may be referenced" (code), defined terms and
section numbers (contracts). ``PlanReadout`` decodes from the predicted plan Z_target [B, M, d]:

    kind      per slot, the structural kind (``SlotKind``: EMPTY, IMPORT, FUNCTION, CLASS, VARIABLE, BLOCK,
              DIRECTIVE); EMPTY marks padding slots at inference, when the slot count is unknown
    text      per slot, the slot text in L_text parallel token positions (one classifier per position,
              no autoregression), e.g. "def total(xs: List[int]) -> int"; signatures are parsed from it
    inventory per sample, which identifier-bearing tokens the program uses (multi-label over V); these
              are the tokens ``PlanVocabularyEnergy`` licenses
    length    per sample, the output length bucket, so the canvas length N can be set from the plan

Training (Stage 2, Sec. 5.1): the readout learns on the *predicted* plan of the frozen planner against
targets extracted from the reference -- supervision, never conditioning. The realizer still sees only
the plan. At inference the readout reads Z_target once, in parallel with the bridge projections, so it
adds no sequential evaluation to K + 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from text_vpa.data.ast_parser import SlotKind

if TYPE_CHECKING:
    from text_vpa.data.ast_parser import TreeSitterASTExtractor
    from text_vpa.data.tokenization import TextVPATokenizer
    from text_vpa.inference.energy_terms import PlanConstraints, VocabularyAnalysis

__all__ = ["PlanReadout", "ReadoutOutput", "ReadoutTargets", "TEXT_KINDS"]

NUM_KINDS = len(SlotKind)
#: Slot kinds whose text is decoded (signatures feed the arity and type energies).
TEXT_KINDS: tuple[int, ...] = (int(SlotKind.FUNCTION), int(SlotKind.IMPORT), int(SlotKind.CLASS))


class ReadoutOutput(NamedTuple):
    kind_logits: Tensor       # [B, M, NUM_KINDS]
    inventory_logits: Tensor  # [B, S]
    length_logits: Tensor     # [B, num_length_buckets]
    slot_features: Tensor     # [B, M, hidden]


@dataclass(frozen=True)
class ReadoutTargets:
    """Targets extracted from the reference program (see ``text_vpa.data.dataset``)."""

    kinds: Tensor          # [B, M] int64 SlotKind (EMPTY for padding slots)
    texts: Tensor          # [B, M, L_text] int64 slot text tokens, [PAD]-padded
    text_mask: Tensor      # [B, M] bool: slot has a text target (kind in TEXT_KINDS)
    inventory: Tensor      # [B, S] float multi-hot of identifier-bearing tokens of the program
    length_bucket: Tensor  # [B] int64


class PlanReadout(nn.Module):
    """Slot decoder heads on Z_target.

    Args:
        latent_dim: d.
        vocab_size: S.
        hidden_dim: width of the per-slot text decoder.
        max_text_tokens: L_text, decoded positions per slot text.
        num_length_buckets: classes of the length head; bucket k covers lengths ((k) B_s, (k + 1) B_s].
        length_bucket_size: B_s.
    """

    def __init__(self, latent_dim: int, vocab_size: int, hidden_dim: int = 256, max_text_tokens: int = 48,
                 num_length_buckets: int = 32, length_bucket_size: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.num_length_buckets = num_length_buckets
        self.length_bucket_size = length_bucket_size
        self.norm = nn.LayerNorm(latent_dim)
        self.kind_head = nn.Linear(latent_dim, NUM_KINDS)
        self.slot_proj = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.text_positions = nn.Parameter(torch.zeros(max_text_tokens, hidden_dim))
        self.text_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, vocab_size))
        self.inventory_head = nn.Linear(2 * latent_dim, vocab_size)
        self.length_head = nn.Linear(2 * latent_dim, num_length_buckets)
        nn.init.normal_(self.text_positions, std=0.02)

    def forward(self, z_target: Tensor) -> ReadoutOutput:
        """``z_target [B, M, d]`` -> ``ReadoutOutput``."""
        if z_target.dim() != 3 or z_target.shape[-1] != self.latent_dim:
            raise ValueError(f"expected z_target [B, M, {self.latent_dim}], got {tuple(z_target.shape)}")
        z = self.norm(z_target)
        pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=-1)                 # [B, 2d]
        return ReadoutOutput(self.kind_head(z), self.inventory_head(pooled), self.length_head(pooled),
                             self.slot_proj(z))

    def text_logits(self, slot_features: Tensor) -> Tensor:
        """``[n, hidden]`` selected slots -> ``[n, L_text, S]`` (all positions in parallel)."""
        return self.text_out(slot_features[:, None, :] + self.text_positions[None, :, :])

    def length_bucket(self, lengths: Tensor) -> Tensor:
        """Bucket index of output lengths ``[B]`` (clamped to the last bucket)."""
        return ((lengths.clamp_min(1) - 1) // self.length_bucket_size).clamp_max(self.num_length_buckets - 1)

    def loss(self, out: ReadoutOutput, targets: ReadoutTargets, pad_id: int, weights: Sequence[float] = (1, 1, 1, 0.1)
             ) -> tuple[Tensor, dict[str, Tensor]]:
        """Kind CE + slot-text CE (non-[PAD] positions of text slots) + inventory BCE + length CE."""
        w_kind, w_text, w_inv, w_len = weights
        kind = F.cross_entropy(out.kind_logits.reshape(-1, NUM_KINDS), targets.kinds.reshape(-1))
        if bool(targets.text_mask.any()):
            feats = out.slot_features[targets.text_mask]                            # [n, hidden]
            gold = targets.texts[targets.text_mask]                                 # [n, L]
            logits = self.text_logits(feats)
            # Texts are left-aligned: the tokens, then [PAD]s. The first [PAD] ends the text and is
            # predicted; later [PAD]s are ignored.
            length = (gold != pad_id).sum(dim=1, keepdim=True)                      # [n, 1]
            keep = torch.arange(gold.shape[1], device=gold.device)[None, :] <= length
            text = F.cross_entropy(logits[keep], gold[keep])
        else:
            text = out.kind_logits.sum() * 0.0
        inventory = F.binary_cross_entropy_with_logits(out.inventory_logits, targets.inventory)
        length = F.cross_entropy(out.length_logits, targets.length_bucket)
        total = w_kind * kind + w_text * text + w_inv * inventory + w_len * length
        return total, {"readout_kind": kind.detach(), "readout_text": text.detach(),
                       "readout_inventory": inventory.detach(), "readout_length": length.detach()}

    # -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, out: ReadoutOutput, tokenizer: TextVPATokenizer, extractor: TreeSitterASTExtractor,
               analysis: VocabularyAnalysis, threshold: float = 0.5) -> tuple[list[PlanConstraints], Tensor]:
        """Symbolic plans and canvas lengths [B] from the readout heads (inference)."""
        from text_vpa.inference.energy_terms import PlanConstraints

        kinds = out.kind_logits.argmax(dim=-1)                                      # [B, M]
        inventory = torch.sigmoid(out.inventory_logits.float()) > threshold        # [B, S]
        lengths = (out.length_logits.argmax(dim=-1) + 1) * self.length_bucket_size
        plans: list[PlanConstraints] = []
        for b in range(kinds.shape[0]):
            signatures = {}
            declared: set[str] = set()
            fn_slots = (kinds[b] == int(SlotKind.FUNCTION)).nonzero().flatten()
            if fn_slots.numel():
                ids = self.text_logits(out.slot_features[b, fn_slots]).argmax(dim=-1)   # [n, L]
                for row in ids.tolist():
                    text = tokenizer.decode(row)
                    sig = _parse_signature(text, extractor)
                    if sig is not None:
                        signatures[sig.name] = sig
                        declared.add(sig.name)
            licensed = frozenset(int(i) for i in inventory[b].nonzero().flatten().tolist()
                                 if analysis.word_bearing[int(i)])
            plans.append(PlanConstraints(licensed_tokens=licensed, signatures=signatures,
                                         declared_identifiers=frozenset(declared), source="readout"))
        return plans, lengths


def _parse_signature(text: str, extractor: TreeSitterASTExtractor):
    """Parse a decoded FUNCTION slot text such as "def f(x: int) -> int" into a ``FunctionSignature``."""
    text = text.strip()
    if not text:
        return None
    source = text + (":\n    pass\n" if extractor.language == "python" else " {}\n")
    try:
        structure = extractor.parse(source)
    except ValueError:
        return None
    if structure.has_errors or len(structure.functions) != 1:
        return None
    return structure.functions[0]
