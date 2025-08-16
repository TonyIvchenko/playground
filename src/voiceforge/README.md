# VoiceForge

Voice-cloning text-to-speech service built around SpeechT5 fine-tuning.

## What It Is

- base TTS model: `microsoft/speecht5_tts`
- vocoder: `microsoft/speecht5_hifigan`
- speaker encoder: `speechbrain/spkrec-ecapa-voxceleb`
- open-data training path wired for `LibriTTS` and `VCTK 0.92`
- app uses `models/speecht5-finetuned` if a local checkpoint exists; otherwise
  it falls back to the base pretrained model

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

Reference clip guidance:

- safest local formats: `wav`, `flac`
- also accepted: `mp3`, `m4a`, `ogg`
- short, clean, single-speaker clips work best

## Data And Training

Download raw corpora:

```bash
python scripts/download_data.py
```

Build train/eval manifests:

```bash
python scripts/prepare_dataset.py --max-per-speaker 200
```

Quickest local smoke-train:

```bash
python scripts/run_tiny_smoke_train.py
```

Practical longer local MPS run:

```bash
python scripts/train_model.py --base-model models/speecht5-finetuned --device mps --epochs 2 --batch-size 2 --gradient-accumulation-steps 4 --max-audio-seconds 10 --max-text-chars 160 --group-by-target-length --mps-empty-cache-steps 25 --save-steps 200 --eval-steps 200 --preview-samples 4
```

Prune older checkpoints later:

```bash
python scripts/prune_checkpoints.py --keep 2
```

## Docker

```bash
docker build --pull -t voiceforge -f src/voiceforge/Dockerfile .
docker run --rm --name voiceforge -p 8080:8080 -e PORT=8080 voiceforge
curl --fail --silent --show-error http://127.0.0.1:8080/health
```

For a local app smoke check without Docker:

```bash
PORT=8091 python main.py
curl --fail --silent --show-error http://127.0.0.1:8091/health
```

## Tests

```bash
python -m pytest -q src/voiceforge/tests
```

## Key Caveats

- English speech only
- training is meant for Apple Silicon MPS or CUDA; CPU is mainly for smoke
  paths
- on Apple Silicon, preview generation and demo inference may split work across
  MPS and CPU
- if no fine-tuned checkpoint exists yet, the app falls back to the base model
