#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

try:
    from .service_manifest import ROOT, SRC_DIR, load_service_manifest, service_names
except ImportError:
    from service_manifest import ROOT, SRC_DIR, load_service_manifest, service_names

DEFAULT_PORT = 18080

SERVICE_SPECS = load_service_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify each service README Local Run command still starts the service."
    )
    parser.add_argument(
        "--service",
        choices=["all", *service_names()],
        default="all",
        help="Service to check. Default runs all services.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Base port for smoke-checked services. Each additional service increments from here.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each service check.",
    )
    return parser.parse_args()


def service_dir(name: str) -> Path:
    path = SRC_DIR / name
    if not path.is_dir():
        raise SystemExit(f"Unknown service '{name}'. Expected {path}.")
    return path


def readme_path(name: str) -> Path:
    return service_dir(name) / "README.md"


def readme_contains_local_run(path: Path, command: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return "## Local Run" in text and command in text


def service_env(name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    extra_env = SERVICE_SPECS[name]["local_run"].get("env", {})
    for key, value in extra_env.items():
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


def tail_log(log_path: Path, lines: int = 20) -> str:
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_process_only_service(name: str, timeout: float) -> None:
    path = service_dir(name)
    log_path = Path(tempfile.gettempdir()) / f"playground-local-run-{name}.log"
    command = [sys.executable, "main.py"]
    print("Running " + " ".join(shlex.quote(part) for part in command), flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=path,
            env=service_env(name),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                details = tail_log(log_path)
                raise SystemExit(
                    "Service exited before the Local Run check completed.\n"
                    f"Log tail:\n{details or '(no log output)'}"
                )
            time.sleep(0.25)
            if "Starting test-service on redis://" in tail_log(log_path):
                print(f"Process-only local run check passed for '{name}'.")
                return
        raise SystemExit(
            "Service stayed alive but never emitted the expected startup log.\n"
            f"Log tail:\n{tail_log(log_path) or '(no log output)'}"
        )
    finally:
        terminate_process(process)


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


def run_service_check(name: str, port: int, timeout: float) -> None:
    readme_command = str(SERVICE_SPECS[name]["local_run"]["readme_command"])
    if not readme_contains_local_run(readme_path(name), readme_command):
        raise SystemExit(
            f"{readme_path(name).relative_to(ROOT)} is missing the expected Local Run command: {readme_command}"
        )

    mode = str(SERVICE_SPECS[name]["local_run"]["mode"])
    if mode == "process":
        run_process_only_service(name, timeout)
        return
    if mode == "oneshot":
        run_oneshot_service(name, timeout)
        return
    run_smoke_service(name, port, timeout)


def main() -> int:
    args = parse_args()
    names = service_names() if args.service == "all" else [args.service]
    for offset, service_name in enumerate(names):
        port = args.port + offset
        run_service_check(service_name, port, args.timeout)
    print(f"README Local Run checks passed for {len(names)} services.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
