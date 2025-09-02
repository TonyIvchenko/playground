from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading


ROOT = Path(__file__).resolve().parents[1]


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_smoke_service_can_probe_existing_url() -> None:
    port = pick_free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    output_path = Path(tempfile.gettempdir()) / "playground-smoke-health.json"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/smoke_service.py",
                "--url",
                f"http://127.0.0.1:{port}/health",
                "--timeout",
                "5",
                "--interval",
                "0.2",
                "--expect-content-type",
                "application/json",
                "--write-body",
                str(output_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, (
        f"URL smoke failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Smoke check passed via" in result.stdout
    assert output_path.read_text(encoding="utf-8") == '{"status":"ok"}'
