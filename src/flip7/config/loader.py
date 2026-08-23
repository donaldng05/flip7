"""Load framework-neutral YAML experiment configuration."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml


def load_config(path: Path) -> Mapping[str, object]:
    """Load a YAML configuration whose top-level value is a mapping."""
    with path.open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)

    if not isinstance(value, dict):
        raise ValueError("configuration must contain a top-level mapping")

    return cast(Mapping[str, object], value)
