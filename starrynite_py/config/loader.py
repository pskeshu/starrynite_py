"""Load pipeline configuration from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import PipelineConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated PipelineConfig instance.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
