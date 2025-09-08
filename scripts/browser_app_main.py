#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def run_browser_app(app_file: str) -> None:
    app_root = Path(app_file).resolve().parent
    repo_root = app_root.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.browser_static_server import serve_static_app

    serve_static_app(app_root.name, app_root)


def main() -> None:
    app_file = globals().get("APP_FILE")
    if not app_file and len(sys.argv) == 2:
        app_file = sys.argv[1]
    if not app_file:
        raise SystemExit(
            "browser_app_main.py expects an app file path via APP_FILE or argv."
        )
    run_browser_app(str(app_file))


if __name__ == "__main__":
    main()
