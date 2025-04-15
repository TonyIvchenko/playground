from pathlib import Path
import os

import gradio as gr


PORT = int(os.getenv("PORT", "8080"))
SERVICE_DIR = Path(__file__).resolve().parent


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VoiceForge") as demo:
        gr.Markdown(
            """
            # VoiceForge
            Upload a reference voice clip, type text, and generate cloned speech.

            The data prep, training, and model loading pipeline are wired up in this service.
            This initial shell gets the app in place while the training path lands.
            """
        )
        reference_audio = gr.Audio(label="Reference Voice", type="filepath")
        text_input = gr.Textbox(label="Text", lines=5, placeholder="Type what the cloned voice should say...")
        generate_button = gr.Button("Generate")
        output_audio = gr.Audio(label="Synthesized Audio")
        status = gr.Textbox(label="Status", value="VoiceForge scaffold is ready. Training and inference wiring are being added.")

        def _not_ready(_reference_audio: str | None, _text: str) -> tuple[None, str]:
            return None, "Inference wiring is not committed yet. Next commits will add data prep, fine-tuning, and synthesis."

        generate_button.click(_not_ready, inputs=[reference_audio, text_input], outputs=[output_audio, status])

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=PORT)
