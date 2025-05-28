#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

SERVICE_TYPES = {
    "bert": "static_browser",
    "counterpoint": "static_browser",
    "ctscan": "python_web",
    "debate": "static_browser",
    "disasters": "python_web",
    "facemesh": "static_browser",
    "manipulation": "static_browser",
    "memorypalace": "static_browser",
    "realitycheck": "static_browser",
    "realitymix": "static_browser",
    "test": "worker_service",
    "vibedj": "static_browser",
    "voiceforge": "python_web",
}

TYPE_REQUIREMENTS = {
    "static_browser": ("main.py", "README.md", "index.html"),
    "python_web": ("main.py", "README.md", "requirements.txt", "Dockerfile", "tests"),
    "worker_service": ("main.py", "README.md", "requirements.txt", "Dockerfile", "tests"),
}


def iter_services() -> list[Path]:
    services: list[Path] = []
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        services.append(path)
    return services


def missing_requirements(service_dir: Path) -> list[str]:
    service_type = SERVICE_TYPES.get(service_dir.name)
    if service_type is None:
        return ["<service type mapping>"]

    missing: list[str] = []
    for requirement in TYPE_REQUIREMENTS[service_type]:
        if not (service_dir / requirement).exists():
            missing.append(requirement)
    return missing


def main() -> int:
    failures: list[tuple[str, str, list[str]]] = []
    unexpected = sorted(set(SERVICE_TYPES) - {path.name for path in iter_services()})
    if unexpected:
        print(
            "Service type file check failed: stale service type mappings for "
            + ", ".join(unexpected),
            file=sys.stderr,
        )
        return 1

    for service_dir in iter_services():
        service_type = SERVICE_TYPES.get(service_dir.name, "<missing>")
        missing = missing_requirements(service_dir)
        if missing:
            failures.append((service_dir.name, service_type, missing))

    if failures:
        print("Service type file check failed:", file=sys.stderr)
        for service, service_type, missing in failures:
            print(
                f"- {service} ({service_type}): missing {', '.join(missing)}",
                file=sys.stderr,
            )
        return 1

    type_counts: dict[str, int] = {}
    for service_type in SERVICE_TYPES.values():
        type_counts[service_type] = type_counts.get(service_type, 0) + 1
    counts = ", ".join(f"{service_type}={count}" for service_type, count in sorted(type_counts.items()))
    print(f"Service type file check passed for {len(SERVICE_TYPES)} services ({counts}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
