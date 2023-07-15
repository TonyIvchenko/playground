"""Runtime shim so existing launch workflows can call main.py."""

from __future__ import annotations

from app import demo, settings


if __name__ == "__main__":
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )
