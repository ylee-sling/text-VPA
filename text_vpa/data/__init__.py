# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Data pipeline (PR-JEPA Table 1, Sec. 5.1).

    tokenization.py  tokenizers and the special tokens ([PAD], [MASK], [NULL_PLAN], slot delimiters)
    ast_parser.py    tree-sitter extraction of the structural view y and of the design mode u
    synthetic.py     a small generator of specification -> program pairs (tests, smoke runs)
    dataset.py       examples, mixture-path corruption, bucketed collation, loaders (HumanEval, MBPP, JSONL)
"""

from text_vpa.data.ast_parser import (
    FunctionSignature,
    ModeExtractor,
    ProgramStructure,
    SlotKind,
    StructuralSlot,
    TreeSitterASTExtractor,
)
from text_vpa.data.synthetic import CodeRecord, synthetic_corpus
from text_vpa.data.tokenization import (
    SPECIAL_TOKENS,
    ByteTokenizer,
    HFTokenizer,
    TextVPATokenizer,
    tokenizer_from_spec,
    train_bpe_tokenizer,
)

__all__ = [
    "SPECIAL_TOKENS",
    "ByteTokenizer",
    "CodeRecord",
    "FunctionSignature",
    "HFTokenizer",
    "ModeExtractor",
    "ProgramStructure",
    "SlotKind",
    "StructuralSlot",
    "TextVPATokenizer",
    "TreeSitterASTExtractor",
    "synthetic_corpus",
    "tokenizer_from_spec",
    "train_bpe_tokenizer",
]
