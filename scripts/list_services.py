#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
KEY_PATHS = (
    "main.py",
    "README.md",
    "requirements.txt",
    "Dockerfile",
    "tests",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Playground services with generated command-reference details."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the default table.",
    )
    return parser.parse_args()


def iter_services() -> list[Path]:
    services = []
    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name == "__pycache__":
            continue
        services.append(path)
    return services


def collect_service(path: Path) -> dict[str, object]:
    rel_path = path.relative_to(ROOT).as_posix()
    readme_path = path / "README.md"
    tests_path = path / "tests"
    docker_path = path / "Dockerfile"
    record: dict[str, object] = {
        "service": path.name,
        "path": rel_path,
        "type": detect_service_type(path),
        "run": detect_run_command(path),
        "tests_path": tests_path.relative_to(ROOT).as_posix() if tests_path.exists() else "-",
        "docker_path": docker_path.relative_to(ROOT).as_posix() if docker_path.exists() else "-",
        "health_endpoint": detect_health_endpoint(path),
        "readme_path": readme_path.relative_to(ROOT).as_posix(),
    }
    for key_path in KEY_PATHS:
        target = path / key_path
        record[key_path] = target.exists()
    return record


def detect_service_type(path: Path) -> str:
    if path.name == "test":
        return "worker service"
    if (path / "index.html").exists():
        return "static browser app"
    if (path / "Dockerfile").exists():
        return "python web service"
    return "python service"


def detect_run_command(path: Path) -> str:
    if path.name == "test":
        return "make run test"
    return f"make run {path.name} 8080"


def detect_health_endpoint(path: Path) -> str:
    if path.name == "test":
        return "-"
    return "/health"


def render_table(records: list[dict[str, object]]) -> str:
    headers = [
        "service",
        "type",
        "run",
        "tests_path",
        "docker_path",
        "health_endpoint",
        "readme_path",
    ]
    rendered_rows: list[list[str]] = []
    for record in records:
        row = []
        for header in headers:
            value = record[header]
            if isinstance(value, bool):
                row.append("yes" if value else "no")
            else:
                row.append(str(value))
        rendered_rows.append(row)

    widths = []
    for index, header in enumerate(headers):
        width = len(header)
        for row in rendered_rows:
            width = max(width, len(row[index]))
        widths.append(width)

    def render_row(values: list[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    lines = [render_row(headers), render_row(["-" * width for width in widths])]
    lines.extend(render_row(row) for row in rendered_rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = [collect_service(path) for path in iter_services()]
    if args.json:
        print(json.dumps(records, indent=2))
        return
    print(render_table(records))


if __name__ == "__main__":
    main()
