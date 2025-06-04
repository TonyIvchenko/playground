#!/usr/bin/env python3
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def read_static_bind_address() -> tuple[str, int]:
    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    return host, port


def display_static_host(host: str) -> str:
    return "127.0.0.1" if host in {"", "0.0.0.0"} else host


def serve_static_health(handler: SimpleHTTPRequestHandler, service_name: str) -> bool:
    if handler.path != "/health":
        return False

    body = json.dumps(
        {
            "status": "ok",
            "service": service_name,
            "app_type": "static_browser",
        }
    ).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)
    return True


def serve_shared_asset(handler: SimpleHTTPRequestHandler) -> bool:
    if not handler.path.startswith("/shared/"):
        return False

    shared_name = handler.path.removeprefix("/shared/")
    shared_path = (SHARED_DIR / shared_name).resolve()
    if not shared_path.is_file() or SHARED_DIR not in shared_path.parents:
        handler.send_error(404, "Shared browser asset not found.")
        return True

    body = shared_path.read_bytes()
    content_type = (
        mimetypes.guess_type(shared_path.name)[0] or "application/octet-stream"
    )
    handler.send_response(200)
    if content_type.startswith("text/"):
        content_type = f"{content_type}; charset=utf-8"
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(body)
    return True


def build_static_handler(
    root: Path, service_name: str
) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self) -> None:
            if serve_static_health(self, service_name):
                return
            if serve_shared_asset(self):
                return
            super().do_GET()

        def do_HEAD(self) -> None:
            if serve_static_health(self, service_name):
                return
            if serve_shared_asset(self):
                return
            super().do_HEAD()

    return Handler


def serve_static_app(service_name: str, root: Path) -> None:
    host, port = read_static_bind_address()
    server = ThreadingHTTPServer((host, port), build_static_handler(root, service_name))
    print(f"Serving {service_name} on http://{display_static_host(host)}:{port}")
    server.serve_forever()
