# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Packaging for text-VPA. Metadata lives in pyproject.toml; the dependencies are declared here
(pyproject lists them as dynamic), so ``pip install .`` and ``pip install -e .[dev]`` both work."""

from setuptools import setup

INSTALL_REQUIRES = [
    "torch>=2.2",
    "numpy>=1.26",
    "einops>=0.7",
    "pydantic>=2.5",
    "hydra-core>=1.3,<1.4",
    "omegaconf>=2.3,<2.4",
    "PyYAML>=6.0",
    "transformers>=4.40",
    "tokenizers>=0.15",
    "tree-sitter>=0.22",
    "tree-sitter-python>=0.21",
    "tree-sitter-cpp>=0.22",
]

EXTRAS_REQUIRE = {
    "dev": ["pytest>=8", "ruff>=0.6"],
    "corpus": ["pyarrow>=14"],          # Parquet sources of text_vpa.data.corpus
    "wandb": ["wandb>=0.16"],
}

setup(install_requires=INSTALL_REQUIRES, extras_require=EXTRAS_REQUIRE)
