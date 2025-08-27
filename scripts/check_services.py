#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .service_manifest import (
        ROOT,
        TYPE_REQUIREMENTS,
        iter_service_dirs,
        load_service_manifest,
        service_names,
    )
except ImportError:
    from service_manifest import (
        ROOT,
        TYPE_REQUIREMENTS,
        iter_service_dirs,
        load_service_manifest,
        service_names,
    )

DEFAULT_PORT = 18080
DEFAULT_TIMEOUT = 30.0
SERVICE_SPECS = load_service_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Playground services from the shared service manifest."
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=("static", "local-run"),
        help="Validation group to run. Default is static.",
    )
    parser.add_argument(
        "--service",
        choices=["all", *service_names()],
        default="all",
        help="Service to validate for local-run checks. Default runs all services.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Base port for smoke-checked local runs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for each local-run check.",
    )
    return parser.parse_args()


def expected_paths_for_service(service_name: str) -> tuple[str, ...]:
    service_type = str(SERVICE_SPECS[service_name]["type"])
    try:
        return TYPE_REQUIREMENTS[service_type]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown service type '{service_type}' for service '{service_name}'."
        ) from exc


def missing_required_paths(service_dir: Path) -> list[str]:
    return [
        name
        for name in expected_paths_for_service(service_dir.name)
        if not (service_dir / name).exists()
    ]


def collect_static_failures() -> list[str]:
    failures: list[str] = []
    repo_services = {path.name for path in iter_service_dirs()}
    manifest_services = set(SERVICE_SPECS)

    for service_name in sorted(repo_services - manifest_services):
        failures.append(f"{service_name}: missing service manifest entry")
    for service_name in sorted(manifest_services - repo_services):
        failures.append(f"{service_name}: stale service manifest entry")

    for service_dir in iter_service_dirs():
        if service_dir.name not in SERVICE_SPECS:
            continue

        missing = missing_required_paths(service_dir)
        if missing:
            failures.append(f"{service_dir.name}: missing {', '.join(missing)}")
            continue

        health_endpoint = SERVICE_SPECS[service_dir.name].get("health_endpoint")
        if not (service_dir / "Dockerfile").exists() or not health_endpoint:
            continue

        source = (service_dir / "main.py").read_text(encoding="utf-8", errors="replace")
        endpoint = str(health_endpoint)
        if f'"{endpoint}"' not in source and f"'{endpoint}'" not in source:
            failures.append(
                f"{service_dir.name}: Dockerized HTTP service is missing {endpoint}"
            )

    return failures


def run_static_checks() -> None:
    failures = collect_static_failures()
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"Service validation failed:\n{details}")

    type_counts: dict[str, int] = {}
    for spec in SERVICE_SPECS.values():
        service_type = str(spec["type"])
        type_counts[service_type] = type_counts.get(service_type, 0) + 1
    counts = ", ".join(
        f"{service_type}={count}" for service_type, count in sorted(type_counts.items())
    )
    print(
        f"Static service validation passed for {len(SERVICE_SPECS)} services ({counts})."
    )


def service_dir(name: str) -> Path:
    for path in iter_service_dirs():
        if path.name == name:
            return path
    raise SystemExit(f"Unknown service '{name}'.")


def readme_path(name: str) -> Path:
    return service_dir(name) / "README.md"


def readme_contains_local_run(path: Path, command: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return "## Local Run" in text and command in text


def service_env(name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in SERVICE_SPECS[name]["local_run"].get("env", {}).items():
        env.setdefault(str(key), str(value))
    return env


def run_smoke_service(name: str, port: int, timeout: float) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/smoke_service.py"),
        name,
        "--port",
        str(port),
        "--timeout",
        str(timeout),
    ]
    print("Running " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=service_env(name), check=True)


def run_oneshot_service(name: str, timeout: float) -> None:
    path = service_dir(name)
    argv = [str(part) for part in SERVICE_SPECS[name]["local_run"].get("argv", [])]
    command = [sys.executable, "main.py", *argv]
    print("Running " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(
        command,
        cwd=path,
        env=service_env(name),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise SystemExit(
            "One-shot Local Run check failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"Stdout:\n{completed.stdout or '(no stdout)'}\n"
            f"Stderr:\n{completed.stderr or '(no stderr)'}"
        )
    if "--config-json" in argv:
        try:
            payload = json.loads(completed.stdout or "")
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "One-shot Local Run check did not produce valid JSON config output.\n"
                f"Stdout:\n{completed.stdout or '(no stdout)'}"
            ) from exc
        if payload.get("service") != f"{name}-service":
            raise SystemExit(
                "One-shot Local Run check produced unexpected service metadata.\n"
                f"Stdout:\n{completed.stdout or '(no stdout)'}"
            )
        if payload.get("dry_run") is not True:
            raise SystemExit(
                "One-shot Local Run check produced config JSON, but dry_run was not true.\n"
                f"Stdout:\n{completed.stdout or '(no stdout)'}"
            )
    if "Dry run only; skipping Redis connection and write loop" not in output:
        raise SystemExit(
            "One-shot Local Run check passed but did not emit the expected dry-run marker.\n"
            f"Output:\n{output or '(no output)'}"
        )
    print(f"One-shot local run check passed for '{name}'.")


def run_local_service_check(name: str, port: int, timeout: float) -> None:
    local_run = SERVICE_SPECS[name]["local_run"]
    readme_command = str(local_run["readme_command"])
    if not readme_contains_local_run(readme_path(name), readme_command):
        raise SystemExit(
            f"{readme_path(name).relative_to(ROOT)} is missing the expected Local Run command: {readme_command}"
        )

    mode = str(local_run["mode"])
    if mode == "smoke":
        run_smoke_service(name, port, timeout)
        return
    if mode == "oneshot":
        run_oneshot_service(name, timeout)
        return
    raise SystemExit(f"Unsupported local run mode '{mode}' for service '{name}'.")


def run_local_checks(service_name: str, port: int, timeout: float) -> None:
    names = service_names() if service_name == "all" else [service_name]
    for offset, name in enumerate(names):
        run_local_service_check(name, port + offset, timeout)
    print(f"Local run validation passed for {len(names)} services.")


def main() -> int:
    args = parse_args()
    checks = args.check or ["static"]
    for check_name in checks:
        if check_name == "static":
            run_static_checks()
        elif check_name == "local-run":
            run_local_checks(args.service, args.port, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
