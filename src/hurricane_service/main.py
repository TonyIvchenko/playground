"""Entry point for running hurricane service locally."""

from __future__ import annotations

import uvicorn

try:
    from .api import app
    from .settings import load_settings
except ImportError:  # pragma: no cover - supports direct script execution
    from api import app  # type: ignore
    from settings import load_settings  # type: ignore


if __name__ == "__main__":
    settings = load_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
