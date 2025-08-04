#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MANIFEST_PATH = Path(__file__).with_name("service_manifest.json")


@lru_cache(maxsize=1)
def load_service_manifest() -> dict[str, dict[str, object]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{MANIFEST_PATH} must contain a top-level object.")
    return payload


def service_names() -> list[str]:
    return sorted(load_service_manifest())
