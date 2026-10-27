# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
tokenization.py -- discrete tokenizers and the special tokens of text-VPA.

One vocabulary V (|V| = S) serves the context c, the structural view y and the output w. Special tokens
are ordinary ids of V (``ByteTokenizer`` appends them after the 256 bytes; a BPE trained here puts them
first; a pretrained tokenizer gets them appended). Text that merely spells a special token, such as the
string "[PAD]" inside a program, is always encoded as text:

    [PAD]          canvas filler; the output ends at the first [PAD] (the realizer learns to emit it)
    [MASK]         absorbing source token of the mask-source probability path (Sec. 3.3)
    [NULL_PLAN]    the structural view of a program without any slot, so f_theta_bar never sees an
                   empty sequence (the bridge's plan-dropout token is a learned latent, not this token)
    [SLOT_<KIND>]  slot delimiters of the serialised structural view y, one per ``SlotKind``
                   (IMPORT, FUNCTION, CLASS, VARIABLE, BLOCK, DIRECTIVE)

Backends:
    ``ByteTokenizer``     256 byte tokens + specials; offline, deterministic (tests, smoke runs)
    ``HFTokenizer``       any Hugging Face fast tokenizer, e.g. byte-level BPE of Qwen2.5-Coder
                          (``HFTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")``) or CodeLlama
    ``train_bpe_tokenizer``  trains a byte-level BPE offline with the ``tokenizers`` library

``spec()`` returns a JSON-serialisable description stored in checkpoints; ``tokenizer_from_spec``
rebuilds the same tokenizer (a trained BPE embeds its JSON, so it needs no file or network).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Any

__all__ = [
    "SLOT_KIND_NAMES",
    "SPECIAL_TOKENS",
    "ByteTokenizer",
    "HFTokenizer",
    "TextVPATokenizer",
    "tokenizer_from_spec",
    "train_bpe_tokenizer",
]

#: Slot kinds with a delimiter token, in ``SlotKind`` order (1..6; EMPTY has none).
SLOT_KIND_NAMES: tuple[str, ...] = ("IMPORT", "FUNCTION", "CLASS", "VARIABLE", "BLOCK", "DIRECTIVE")
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[MASK]", "[NULL_PLAN]") + tuple(f"[SLOT_{k}]" for k in SLOT_KIND_NAMES)


class TextVPATokenizer(ABC):
    """Common interface. Ids are dense in [0, vocab_size); the specials are ordinary ids of V."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """S = |V|, specials included."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` without any special token."""

    @abstractmethod
    def _decode_ids(self, ids: Sequence[int]) -> str:
        """Text of non-special ids."""

    @abstractmethod
    def special_token_id(self, token: str) -> int:
        """Id of one of ``SPECIAL_TOKENS``."""

    @abstractmethod
    def spec(self) -> dict[str, Any]:
        """JSON-serialisable description for ``tokenizer_from_spec``."""

    @abstractmethod
    def _token_string(self, token_id: int) -> str:
        """Surface text of a single non-special token (invalid UTF-8 replaced by U+FFFD)."""

    # -----------------------------------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.special_token_id("[PAD]")

    @property
    def mask_id(self) -> int:
        return self.special_token_id("[MASK]")

    @property
    def null_plan_id(self) -> int:
        return self.special_token_id("[NULL_PLAN]")

    def slot_token_id(self, kind: int) -> int:
        """Delimiter of slot kind ``kind`` (``SlotKind`` value 1..6)."""
        if not 1 <= int(kind) <= len(SLOT_KIND_NAMES):
            raise ValueError(f"slot kind must be in 1..{len(SLOT_KIND_NAMES)}, got {kind}")
        return self.special_token_id(f"[SLOT_{SLOT_KIND_NAMES[int(kind) - 1]}]")

    @cached_property
    def special_ids(self) -> frozenset[int]:
        return frozenset(self.special_token_id(t) for t in SPECIAL_TOKENS)

    def is_special(self, token_id: int) -> bool:
        return int(token_id) in self.special_ids

    def decode(self, ids: Iterable[int], stop_at_pad: bool = True, skip_special: bool = True) -> str:
        """Text of ``ids``; stops at the first [PAD] (end of the output) unless ``stop_at_pad`` is False.

        With ``skip_special`` the special tokens are removed and the remaining ids are decoded together (as
        Hugging Face's ``skip_special_tokens``), so a byte sequence split by a special token still decodes.
        """
        out: list[int] = []
        parts: list[str] = []
        for token in ids:
            token = int(token)
            if stop_at_pad and token == self.pad_id:
                break
            if token in self.special_ids:
                if skip_special:
                    continue
                if out:
                    parts.append(self._decode_ids(out))
                    out = []
                parts.append(self._special_name(token))
                continue
            out.append(token)
        if out:
            parts.append(self._decode_ids(out))
        return "".join(parts)

    def _special_name(self, token_id: int) -> str:
        for name in SPECIAL_TOKENS:
            if self.special_token_id(name) == token_id:
                return name
        raise KeyError(token_id)

    @cached_property
    def token_strings(self) -> tuple[str, ...]:
        """Surface text of every id (special tokens give their bracketed name). Used by the energy terms."""
        return tuple(
            self._special_name(i) if i in self.special_ids else self._token_string(i) for i in range(self.vocab_size)
        )

    def pad_to(self, ids: Sequence[int], length: int) -> list[int]:
        """Truncate or right-pad with [PAD] to exactly ``length`` tokens."""
        ids = list(ids)[:length]
        return ids + [self.pad_id] * (length - len(ids))


# ---------------------------------------------------------------------------------------------
# Byte tokenizer
# ---------------------------------------------------------------------------------------------
class ByteTokenizer(TextVPATokenizer):
    """UTF-8 bytes 0..255, then the special tokens. No training, no files, no network."""

    BYTE_VOCAB = 256

    @property
    def vocab_size(self) -> int:
        return self.BYTE_VOCAB + len(SPECIAL_TOKENS)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _decode_ids(self, ids: Sequence[int]) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def _token_string(self, token_id: int) -> str:
        return bytes([token_id]).decode("utf-8", errors="replace")

    def special_token_id(self, token: str) -> int:
        try:
            return self.BYTE_VOCAB + SPECIAL_TOKENS.index(token)
        except ValueError as exc:
            raise KeyError(f"unknown special token {token!r}") from exc

    def spec(self) -> dict[str, Any]:
        return {"type": "byte"}


# ---------------------------------------------------------------------------------------------
# Hugging Face fast tokenizers (byte-level BPE: Qwen2.5-Coder, GPT-2; SentencePiece: CodeLlama)
# ---------------------------------------------------------------------------------------------
class HFTokenizer(TextVPATokenizer):
    """Wraps a ``transformers.PreTrainedTokenizerFast`` and registers the text-VPA special tokens.

    Args:
        tokenizer: the fast tokenizer (special tokens are added to it if missing).
        source: ``{"name_or_path": ...}`` or ``{"json": ...}``, recorded in ``spec()``.
    """

    def __init__(self, tokenizer: Any, source: dict[str, Any]) -> None:
        missing = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
        if missing:
            tokenizer.add_special_tokens({"additional_special_tokens": missing})
        self._tok = tokenizer
        self._source = dict(source)
        vocab = tokenizer.get_vocab()
        self._special = {t: int(vocab[t]) for t in SPECIAL_TOKENS}
        size = len(tokenizer)
        if any(not 0 <= i < size for i in self._special.values()):
            raise ValueError("special token ids must lie inside the vocabulary")

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> HFTokenizer:
        """Load e.g. ``"Qwen/Qwen2.5-Coder-1.5B"`` (needs the files locally or network access)."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, **kwargs)
        return cls(tok, {"name_or_path": name_or_path})

    @classmethod
    def from_json(cls, tokenizer_json: str) -> HFTokenizer:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(tokenizer_json))
        return cls(tok, {"json": tokenizer_json})

    @property
    def backend(self) -> Any:
        return self._tok

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def encode(self, text: str) -> list[int]:
        # split_special_tokens: the text "[PAD]" inside a program is ordinary text, never the [PAD] id.
        return list(self._tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"])

    def _decode_ids(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _token_string(self, token_id: int) -> str:
        return self._decode_ids([token_id])

    def special_token_id(self, token: str) -> int:
        try:
            return self._special[token]
        except KeyError as exc:
            raise KeyError(f"unknown special token {token!r}") from exc

    def spec(self) -> dict[str, Any]:
        return {"type": "hf", **self._source}


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int = 1024, min_frequency: int = 2) -> HFTokenizer:
    """Train a byte-level BPE (GPT-2 style pre-tokenisation) offline; the specials are added on top.

    The resulting tokenizer is fully described by its JSON, which ``spec()`` embeds.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must be at least {256 + len(SPECIAL_TOKENS)} (bytes plus specials)")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return HFTokenizer.from_json(tok.to_str())


def tokenizer_from_spec(spec: dict[str, Any]) -> TextVPATokenizer:
    """Rebuild a tokenizer from ``TextVPATokenizer.spec()``."""
    kind = spec.get("type")
    if kind == "byte":
        return ByteTokenizer()
    if kind == "hf":
        if "json" in spec:
            return HFTokenizer.from_json(spec["json"])
        if "name_or_path" in spec:
            return HFTokenizer.from_pretrained(spec["name_or_path"])
    raise ValueError(f"unknown tokenizer spec: {json.dumps(spec)[:200]}")
