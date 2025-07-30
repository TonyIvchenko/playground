from pathlib import Path
import os
import sys

from fastapi import FastAPI
import gradio as gr
import uvicorn

try:
    from inference import (
        format_initial_status,
        run_inference as run_voiceforge_inference,
    )
    from model.speecht5 import DEFAULT_MODEL_DIR
    from ui import build_app as build_voiceforge_app
except ImportError:
    from src.voiceforge.inference import (
        format_initial_status,
        run_inference as run_voiceforge_inference,
    )
    from src.voiceforge.model.speecht5 import DEFAULT_MODEL_DIR
    from src.voiceforge.ui import build_app as build_voiceforge_app


PORT = int(os.getenv("PORT", "8080"))
HOST = "0.0.0.0"
MODEL_DIR = Path(os.getenv("VOICEFORGE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def print_http_startup(service_name: str, host: str, port: int) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.service_startup import print_http_service_startup

    print_http_service_startup(service_name, host, port)


def run_inference(reference_audio: str | None, text: str) -> tuple[str | None, str]:
    return run_voiceforge_inference(reference_audio, text, model_dir=MODEL_DIR)


def build_app():
    return build_voiceforge_app(
        run_inference_fn=run_inference,
        initial_status=format_initial_status(MODEL_DIR),
    )


demo = build_app()
app = FastAPI(title="VoiceForge")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "service": "VoiceForge",
        "status": "ok",
        "model_dir": str(MODEL_DIR),
        "model_ready": (MODEL_DIR / "config.json").exists(),
    }


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    print_http_startup("VoiceForge", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
