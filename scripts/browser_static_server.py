#!/usr/bin/env python3
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
SHARED_TOKEN_PATH = SHARED_DIR / "browser-tokens.css"


def serve_shared_asset(handler: SimpleHTTPRequestHandler) -> bool:
    if handler.path != "/shared/browser-tokens.css":
        return False

    if not SHARED_TOKEN_PATH.exists():
        handler.send_error(404, "Shared browser token stylesheet not found.")
        return True

    body = SHARED_TOKEN_PATH.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/css; charset=utf-8")
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
