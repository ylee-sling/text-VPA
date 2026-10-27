# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
loader.py -- compose YAML with Hydra and validate it against the Pydantic schema.

The YAML groups ship inside the package (``text_vpa/configs/conf``), so ``load_config`` works from
a source checkout and from an installed wheel::

    from text_vpa.configs import load_config
    cfg = load_config(overrides=["planner=tiny", "realizer=logit_kl", "realizer.num_steps=4"])
    cfg.realizer.sequential_depth   # 7 == K + 3

Hydra performs composition and command-line overrides; Pydantic performs all type and constraint
checks, so a typo such as ``realizer.num_step=4`` fails instead of being silently ignored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from text_vpa.configs.schema import TextVPAConfig

__all__ = ["compose_raw", "config_dir", "load_config", "validate_config"]


def config_dir() -> Path:
    """Directory of the packaged Hydra config groups."""
    return Path(__file__).resolve().parent / "conf"


def validate_config(raw: DictConfig | Mapping[str, Any]) -> TextVPAConfig:
    """Validate a raw (Hydra or plain) config tree.

    Raises:
        pydantic.ValidationError: unknown key, wrong type, or violated constraint.
    """
    data = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
    return TextVPAConfig.model_validate(data)


def compose_raw(overrides: Sequence[str] = (), config_name: str = "config",
                directory: str | Path | None = None) -> dict[str, Any]:
    """Compose the YAML groups with Hydra and return a plain dict, *without* validation.

    For callers that replace whole sections (e.g. the planner of a Stage-1 checkpoint) before validating,
    so constraints between sections are checked on the final combination only.
    """
    from hydra import compose, initialize_config_dir

    root = Path(directory) if directory is not None else config_dir()
    with initialize_config_dir(version_base="1.3", config_dir=str(root.resolve())):
        raw = compose(config_name=config_name, overrides=list(overrides))
    data = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(data, dict)
    return data  # type: ignore[return-value]


def load_config(
    overrides: Sequence[str] = (),
    config_name: str = "config",
    directory: str | Path | None = None,
) -> TextVPAConfig:
    """Compose ``<directory>/<config_name>.yaml``, apply Hydra ``overrides`` and validate.

    Args:
        overrides: Hydra override strings, e.g. ``["realizer=logit_kl", "training.batch_size=32"]``.
        config_name: primary config file without ``.yaml``.
        directory: config directory; defaults to the packaged ``conf/``.
    """
    return validate_config(compose_raw(overrides, config_name, directory))
