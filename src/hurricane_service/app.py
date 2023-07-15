"""Gradio app entrypoint (Hugging Face Spaces friendly)."""

from __future__ import annotations

from .gradio_app import create_demo
from .settings import load_settings


settings = load_settings()
demo = create_demo(settings)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )
