from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC_SERVICES = (
    "bert",
    "counterpoint",
    "debate",
    "facemesh",
    "manipulation",
    "memorypalace",
    "realitycheck",
    "realitymix",
    "vibedj",
)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("service", STATIC_SERVICES)
def test_static_app_serves_index_html(service: str) -> None:
    port = pick_free_port()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_service.py",
            service,
            "--port",
            str(port),
            "--timeout",
            "20",
            "--interval",
            "0.5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Static smoke failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert f"http://127.0.0.1:{port}/" in result.stdout
