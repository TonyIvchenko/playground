#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable

try:
    from .poll_http_health import poll_url
except ImportError:
    from poll_http_health import poll_url

try:
    from .service_manifest import SRC_DIR, load_service_manifest
except ImportError:
    from service_manifest import SRC_DIR, load_service_manifest

DEFAULT_PORT = 8080
SERVICE_SPECS = load_service_manifest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke a Playground service or probe an already-running HTTP endpoint."
    )
    parser.add_argument(
        "service",
        nargs="?",
        help="Service name under src/<service> when starting a local smoke run.",
    )
    parser.add_argument(
        "--url",
        help="Probe an already-running HTTP endpoint instead of starting a local service.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind for the smoke run."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="How long to wait for the smoke endpoint to respond.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay between probe attempts while the service is starting.",
    )
    parser.add_argument(
        "--path",
        help="Optional path override for the smoke probe.",
    )
    parser.add_argument(
        "--expect-content-type",
        help="Optional content-type override for the smoke probe.",
    )
    parser.add_argument(
        "--expect-body-fragment",
        help="Optional body-fragment override for the smoke probe.",
    )
    parser.add_argument(
        "--write-body",
        help="Optional file path to write the response body to after a successful smoke probe.",
    )
    return parser.parse_args()


def service_dir(name: str) -> Path:
    path = SRC_DIR / name
    if not path.is_dir():
        raise SystemExit(f"Unknown service '{name}'. Expected a directory at {path}.")
    if not (path / "main.py").exists():
        raise SystemExit(f"Service '{name}' is missing {path / 'main.py'}.")
    if not SERVICE_SPECS[name]["health_endpoint"]:
        raise SystemExit(
            f"Service '{name}' does not expose an HTTP smoke path. Use its documented local run path instead."
        )
    return path


def smoke_path(name: str) -> str:
    return str(SERVICE_SPECS[name]["health_endpoint"])


def expected_content_type(name: str) -> str:
    return "application/json"


def expected_body_fragment(name: str) -> str | None:
    return None


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


def process_abort_message(
    process: subprocess.Popen[bytes], log_path: Path
) -> str | None:
    if process.poll() is None:
        return None
    details = tail_log(log_path)
    return (
        "Service exited before the smoke probe succeeded.\n"
        f"Log tail:\n{details or '(no log output)'}"
    )


def probe_url(
    *,
    url: str,
    timeout: float,
    interval: float,
    content_type: str | None,
    body_fragment: str | None,
    write_body: str | None,
    abort_message: Callable[[], str | None] | None = None,
) -> bytes:
    body = poll_url(
        url,
        timeout=timeout,
        interval=interval,
        expect_content_type=content_type,
        body_fragment=body_fragment.encode("utf-8") if body_fragment else None,
        abort_message=abort_message,
    )
    if write_body:
        Path(write_body).write_bytes(body)
    return body


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.service) == bool(args.url):
        raise SystemExit("Provide exactly one of: <service> or --url.")


def smoke_running_url(args: argparse.Namespace) -> None:
    probe_url(
        url=str(args.url),
        timeout=args.timeout,
        interval=args.interval,
        content_type=args.expect_content_type,
        body_fragment=args.expect_body_fragment,
        write_body=args.write_body,
    )
    print(f"Smoke check passed via {args.url}")


def smoke_local_service(args: argparse.Namespace) -> None:
    path = service_dir(str(args.service))
    probe_path = args.path or smoke_path(str(args.service))
    url = f"http://127.0.0.1:{args.port}{probe_path}"
    content_type = args.expect_content_type or expected_content_type(str(args.service))
    body_fragment = args.expect_body_fragment or expected_body_fragment(
        str(args.service)
    )

    log_path = Path(tempfile.gettempdir()) / f"playground-smoke-{args.service}.log"
    env = os.environ.copy()
    env["PORT"] = str(args.port)
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=path,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    try:
        probe_url(
            url=url,
            timeout=args.timeout,
            interval=args.interval,
            content_type=content_type,
            body_fragment=body_fragment,
            write_body=args.write_body,
            abort_message=lambda: process_abort_message(process, log_path),
        )
    finally:
        terminate_process(process)

    print(f"Smoke check passed for '{args.service}' via {url}")
    print(f"Log: {log_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.url:
        smoke_running_url(args)
        return
    smoke_local_service(args)


if __name__ == "__main__":
    main()
