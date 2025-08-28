#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT / "pyproject.toml"


@lru_cache(maxsize=1)
def load_playground_config() -> dict[str, object]:
    payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    tool = payload.get("tool", {})
    config = tool.get("playground")
    if not isinstance(config, dict):
        raise ValueError("pyproject.toml is missing [tool.playground] config.")
    return config
