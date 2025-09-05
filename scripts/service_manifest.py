#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MANIFEST_PATH = Path(__file__).with_name("service_manifest.json")
TYPE_REQUIREMENTS = {
    "static browser app": ("main.py", "README.md", "index.html"),
    "python web service": (
        "main.py",
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "tests",
    ),
    "worker service": (
        "main.py",
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "tests",
    ),
}
TYPE_DEFAULTS: dict[str, dict[str, object]] = {
    "static browser app": {
        "run": "make run {service} 8080",
        "health_endpoint": "/health",
        "local_run": {
            "readme_command": "python main.py",
            "mode": "smoke",
        },
    },
    "python web service": {
        "run": "make run {service} 8080",
        "health_endpoint": "/health",
        "local_run": {
            "readme_command": "python main.py",
            "mode": "smoke",
        },
    },
    "worker service": {
        "run": "make run {service}",
        "health_endpoint": None,
        "local_run": {},
    },
}


def normalize_service_spec(
    service_name: str, payload: dict[str, object]
) -> dict[str, object]:
    service_type = str(payload["type"])
    if service_type not in TYPE_DEFAULTS:
        raise ValueError(
            f"{MANIFEST_PATH} has unknown service type {service_type!r} for {service_name!r}."
        )

    defaults = TYPE_DEFAULTS[service_type]
    record: dict[str, object] = {
        "type": service_type,
        "run": str(payload.get("run", defaults["run"])).format(service=service_name),
        "health_endpoint": payload.get("health_endpoint", defaults["health_endpoint"]),
    }

    local_run_defaults = dict(defaults.get("local_run", {}))
    local_run_payload = payload.get("local_run", {})
    if local_run_payload:
        if not isinstance(local_run_payload, dict):
            raise ValueError(
                f"{MANIFEST_PATH} local_run for {service_name!r} must be an object."
            )
        local_run_defaults.update(local_run_payload)
    record["local_run"] = local_run_defaults
    return record


@lru_cache(maxsize=1)
def load_service_manifest() -> dict[str, dict[str, object]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{MANIFEST_PATH} must contain a top-level object.")
    return {
        service_name: normalize_service_spec(service_name, dict(spec))
        for service_name, spec in payload.items()
    }


def service_names() -> list[str]:
    return sorted(load_service_manifest())


def iter_service_dirs() -> list[Path]:
    services: list[Path] = []
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        services.append(path)
    return services
