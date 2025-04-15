from pathlib import Path
import os

import gradio as gr

try:
    from model.speecht5 import DEFAULT_MODEL_DIR, load_speecht5_bundle, synthesize_to_temp_wav
except ImportError:
    from src.voiceforge.model.speecht5 import DEFAULT_MODEL_DIR, load_speecht5_bundle, synthesize_to_temp_wav


PORT = int(os.getenv("PORT", "8080"))
MODEL_DIR = Path(os.getenv("VOICEFORGE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def run_inference(reference_audio: str | None, text: str) -> tuple[str | None, str]:
    text = (text or "").strip()
    if not reference_audio:
        return None, "Upload a reference clip first."
    if not text:
        return None, "Type text to synthesize first."

    try:
        bundle = load_speecht5_bundle(model_dir=str(MODEL_DIR))
        output_path, status = synthesize_to_temp_wav(text=text, reference_audio_path=reference_audio, bundle=bundle)
        return output_path, status
    except Exception as exc:  # noqa: BLE001
        return None, f"Voice synthesis failed: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VoiceForge") as demo:
        gr.Markdown(
            """
            # VoiceForge
            Upload a short clean reference voice clip, type text, and synthesize speech in that voice.

            If `models/speecht5-finetuned` exists, the app uses the fine-tuned checkpoint.
            Otherwise it falls back to the base pretrained SpeechT5 model.
            """
        )
        with gr.Row():
            reference_audio = gr.Audio(label="Reference Voice", type="filepath")
            text_input = gr.Textbox(
                label="Text",
                lines=8,
                value="I am ready for the fine-tuned voice cloning demo. This sentence should be spoken in the uploaded reference voice.",
            )
        generate_button = gr.Button("Generate Voice")
        output_audio = gr.Audio(label="Synthesized Audio")
        status = gr.Textbox(label="Status", value=f"Looking for model in {MODEL_DIR}")

        generate_button.click(run_inference, inputs=[reference_audio, text_input], outputs=[output_audio, status])

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=PORT)
