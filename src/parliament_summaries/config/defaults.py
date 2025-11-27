from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .models import AppConfig

DEFAULTS_PATH = Path(__file__).with_suffix(".yaml")


def load_defaults() -> AppConfig:
    """Load configuration defaults from the bundled YAML file."""
    with DEFAULTS_PATH.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh)
    return AppConfig.model_validate(data)
