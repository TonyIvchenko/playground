# VoiceForge

Voice-cloning text-to-speech service built around transfer learning instead of training a voice model from zero.

## Model Stack

- Base text-to-speech model: `microsoft/speecht5_tts`
- Vocoder: `microsoft/speecht5_hifigan`
- Speaker encoder: `speechbrain/spkrec-ecapa-voxceleb`
- Fine-tuning target: speaker-conditioned SpeechT5 on open English speech corpora

## Data

The service is wired for these open datasets:

- `LibriTTS` (`train-clean-100`, `dev-clean` by default)
- `VCTK 0.92`

The downloader uses `torchaudio` dataset loaders so it pulls the official archives instead of a private mirror.

## Layout

- `model/speecht5.py`
- `scripts/download_data.py`
- `scripts/prepare_dataset.py`
- `scripts/train_model.py`
- `notebooks/voiceforge.ipynb`
- `main.py`

## Local Run

From `src/voiceforge`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Docker

From repo root:

```bash
docker build --pull -t voiceforge -f src/voiceforge/Dockerfile .
docker run --rm --name voiceforge -p 8080:8080 -e PORT=8080 voiceforge
curl --fail --silent --show-error http://127.0.0.1:8080/health
```


## End-To-End Flow

Download the raw corpora:

```bash
python scripts/download_data.py
```

Build train/eval manifests:

```bash
python scripts/prepare_dataset.py --max-per-speaker 200
```

Run a smoke fine-tune on current hardware:

```bash
python scripts/train_model.py --epochs 1 --max-train-samples 64 --max-eval-samples 16 --preview-samples 2
```

Run a larger local training pass on MPS:

```bash
python scripts/train_model.py --device mps --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 --resume-from-checkpoint models/speecht5-finetuned/checkpoint-100
```

After training, the Gradio app will automatically use `models/speecht5-finetuned` if the checkpoint exists.

## References

- SpeechT5 model card: https://huggingface.co/microsoft/speecht5_tts
- SpeechT5 vocoder: https://huggingface.co/microsoft/speecht5_hifigan
- SpeechBrain ECAPA speaker embeddings: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- LibriTTS dataset: https://www.openslr.org/60/
- VCTK 0.92 dataset: https://datashare.ed.ac.uk/handle/10283/3443

## Notes

- The first implementation targets English speech.
- Training is designed for Apple Silicon MPS or CUDA.
- If no fine-tuned checkpoint exists yet, the demo falls back to the base pretrained model.
- Short clean reference clips work better than noisy or heavily reverberant audio.

## Training Notes

- `--resume-from-checkpoint` continues from a saved trainer checkpoint.
- `--save-total-limit` prunes older checkpoints so the run does not grow forever.
- After each training run, VoiceForge generates preview `.wav` files under `models/speecht5-finetuned/previews`.
