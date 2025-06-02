#!/usr/bin/env python3
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mimetypes
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"


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


def build_static_handler(root: Path) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self) -> None:
            if serve_shared_asset(self):
                return
            super().do_GET()

        def do_HEAD(self) -> None:
            if serve_shared_asset(self):
                return
            super().do_HEAD()

    return Handler


def serve_static_app(service_name: str, root: Path) -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), build_static_handler(root))
    print(f"Serving {service_name} on http://127.0.0.1:{port}")
    server.serve_forever()
