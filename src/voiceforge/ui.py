from __future__ import annotations

from collections.abc import Callable

import gradio as gr


VOICEFORGE_INTRO_MARKDOWN = """
# VoiceForge
Upload a short clean reference voice clip, type text, and synthesize speech in that voice.
The reference upload uses a plain file picker so the app does not depend on system `ffprobe`.

If `models/speecht5-finetuned` exists, the app uses the fine-tuned checkpoint.
Otherwise it falls back to the base pretrained SpeechT5 model.
""".strip()

DEFAULT_TEXT = (
    "I am ready for the fine-tuned voice cloning demo. "
    "This sentence should be spoken in the uploaded reference voice."
)

REFERENCE_FILE_TYPES = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]


def build_app(
    *,
    run_inference_fn: Callable[[str | None, str], tuple[str | None, str]],
    initial_status: str,
) -> gr.Blocks:
    with gr.Blocks(title="VoiceForge") as demo:
        gr.Markdown(VOICEFORGE_INTRO_MARKDOWN)
        with gr.Row():
            reference_audio = gr.File(
                label="Reference Voice File",
                type="filepath",
                file_types=REFERENCE_FILE_TYPES,
            )
            text_input = gr.Textbox(
                label="Text",
                lines=8,
                value=DEFAULT_TEXT,
            )
        generate_button = gr.Button("Generate Voice")
        output_audio = gr.Audio(label="Synthesized Audio")
        status = gr.Textbox(label="Status", value=initial_status)

        # Gradio's API schema generation is broken for file-backed components in the
        # pinned version we ship, so keep the interactive UI but hide the auto API docs.
        generate_button.click(
            run_inference_fn,
            inputs=[reference_audio, text_input],
            outputs=[output_audio, status],
            show_api=False,
        )

    return demo
