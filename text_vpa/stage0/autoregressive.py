# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
autoregressive.py -- the baseline of H2: the released model's own left-to-right decoding.

The model is ``Gemma4ForConditionalGeneration`` as loaded from the Hub and the decoder is its ``generate`` with
the KV cache; nothing of the realizer is involved. Prompt, token budget (the canvas length N), token draw
(temperature, top-k, top-p) and truncation (``postprocess``) are the realizer's.

Generation stops early when every row has reached the truncation rule's stop (checked every few tokens on the
decoded text); this changes no program, only the time spent. Depth: one prefill + one pass per generated token.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

__all__ = ["cut_at_eos", "generate_autoregressive"]


def cut_at_eos(ids: Sequence[int], eos_ids: Sequence[int]) -> list[int]:
    """The tokens before the first end-of-sequence id."""
    stops = set(eos_ids)
    out: list[int] = []
    for token in ids:
        if token in stops:
            break
        out.append(int(token))
    return out


def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
    configured = getattr(model.generation_config, "eos_token_id", None)
    ids = [configured] if isinstance(configured, int) else list(configured or [])
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(int(i) for i in ids))


@torch.no_grad()
def generate_autoregressive(model: Any, tokenizer: Any, prompt_ids: Sequence[int], num_samples: int,
                            max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int,
                            done: Callable[[str], bool] | None = None, check_every: int = 8
                            ) -> tuple[list[str], list[int]]:
    """``num_samples`` continuations of one prompt.

    Args:
        model: the released causal model (``generate``), on its device.
        tokenizer: its Hugging Face tokenizer.
        prompt_ids: the prompt's ids, <bos> included.
        done: text -> True once the continuation needs no more tokens (``postprocess.is_complete``).

    Returns:
        the decoded continuations (cut at the first <eos>) and the number of tokens generated for each.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    device = next(model.parameters()).device
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    stops = [*eos_ids, pad_id]                                   # rows that stopped early are padded
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([list(prompt_ids)] * num_samples, dtype=torch.long, device=device)

    def decode(row: Tensor) -> str:
        return tokenizer.decode(cut_at_eos(row[prompt_len:].tolist(), stops), skip_special_tokens=True)

    class _AllComplete(StoppingCriteria):
        def __init__(self) -> None:
            self.finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

        def __call__(self, ids: Tensor, scores: Tensor, **kwargs: Any) -> Tensor:
            new = ids.shape[1] - prompt_len
            if done is not None and new % check_every == 0:
                for row in range(ids.shape[0]):
                    if not bool(self.finished[row]) and done(decode(ids[row])):
                        self.finished[row] = True
            return self.finished.clone()

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sampling: dict[str, Any] = {"do_sample": False}
    if temperature > 0:
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    output = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids, pad_token_id=pad_id,
        stopping_criteria=StoppingCriteriaList([_AllComplete()]), **sampling,
    )
    texts = [decode(row) for row in output]
    lengths = [len(cut_at_eos(row[prompt_len:].tolist(), stops)) for row in output]
    return texts, lengths
