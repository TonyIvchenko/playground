# VoiceForge

Voice-cloning text-to-speech service built around transfer learning instead of training a voice model from zero.

## Plan

- Download open multi-speaker speech corpora.
- Prepare a speaker-conditioned training manifest.
- Fine-tune a pretrained `SpeechT5` text-to-speech model.
- Run a local Gradio demo that speaks user text in the voice of a reference clip.

## Layout

- `model/speecht5.py`
- `scripts/download_data.py`
- `scripts/prepare_dataset.py`
- `scripts/train_model.py`
- `main.py`

## Local Run

From `src/voiceforge`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Notes

- The first implementation targets English speech.
- Training is designed for Apple Silicon MPS or CUDA.
- The service can fall back to the base pretrained model before a fine-tuned checkpoint exists.
