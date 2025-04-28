#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
HTTP_MARKERS = (
    "FastAPI(",
    "uvicorn.run(",
    "ThreadingHTTPServer(",
    "SimpleHTTPRequestHandler",
    "gr.mount_gradio_app(",
)
HEALTH_ROUTE_RE = re.compile(r"['\"]/health['\"]")


def iter_docker_services() -> list[Path]:
    services: list[Path] = []
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        if (path / "Dockerfile").exists():
            services.append(path)
    return services


def main() -> int:
    checked: list[str] = []
    skipped: list[str] = []
    failures: list[str] = []

    for service_dir in iter_docker_services():
        main_path = service_dir / "main.py"
        if not main_path.exists():
            failures.append(f"{service_dir.name}: missing main.py")
            continue

        source = main_path.read_text(encoding="utf-8", errors="replace")
        if not any(marker in source for marker in HTTP_MARKERS):
            skipped.append(service_dir.name)
            continue

        if HEALTH_ROUTE_RE.search(source):
            checked.append(service_dir.name)
            continue

        failures.append(f"{service_dir.name}: Dockerized HTTP service is missing /health")

    if failures:
        print("Docker health endpoint check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        if checked:
            print(
                "Checked services with /health: " + ", ".join(checked),
                file=sys.stderr,
            )
        if skipped:
            print(
                "Skipped non-HTTP Docker services: " + ", ".join(skipped),
                file=sys.stderr,
            )
        return 1

    print(
        "Checked Dockerized HTTP services with /health: "
        + (", ".join(checked) if checked else "(none)")
    )
    if skipped:
        print("Skipped non-HTTP Docker services: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
