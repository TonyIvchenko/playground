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

- `inference.py`
- `model/speecht5.py`
- `ui.py`
- `scripts/download_data.py`
- `scripts/prepare_dataset.py`
- `scripts/prune_checkpoints.py`
- `scripts/run_tiny_smoke_train.py`
- `scripts/train_model.py`
- `notebooks/voiceforge.ipynb`
- `main.py`

## Local Run

From `src/voiceforge`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Notebook

Open `notebooks/voiceforge.ipynb` to inspect:

- manifest balance across LibriTTS and VCTK
- the latest `artifact.json` training summary
- generated preview clips under `models/speecht5-finetuned/previews`

## Docker

From repo root:

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

Run the named tiny smoke-train preset:

```bash
python scripts/run_tiny_smoke_train.py
```

Forward a few overrides to the underlying trainer when needed:

```bash
python scripts/run_tiny_smoke_train.py --device mps --output-dir models/speecht5-smoke
```

Run a bounded local continuation pass on MPS:

```bash
python scripts/train_model.py --base-model models/speecht5-finetuned --device mps --epochs 1 --max-train-samples 128 --max-eval-samples 32 --preview-samples 4 --save-steps 32 --eval-steps 32
```

Resume a larger run later:

```bash
python scripts/train_model.py --base-model models/speecht5-finetuned --device mps --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 --resume-from-checkpoint models/speecht5-finetuned/checkpoint-8
```

Use this Apple Silicon tuned command for a practical long local run:

```bash
python scripts/train_model.py --base-model models/speecht5-finetuned --device mps --epochs 2 --batch-size 2 --gradient-accumulation-steps 4 --max-audio-seconds 10 --max-text-chars 160 --group-by-target-length --mps-empty-cache-steps 25 --save-steps 200 --eval-steps 200 --preview-samples 4
```

Prune old local checkpoints later without touching the newest two:

```bash
python scripts/prune_checkpoints.py --keep 2
```

After training, the Gradio app will automatically use `models/speecht5-finetuned` if the checkpoint exists.

## Verified local status

On this machine we verified:

- full manifest refresh from LibriTTS + VCTK: `11178` train rows, `787` eval rows, `396` speakers
- MPS smoke fine-tune completed successfully
- local app health check returned `model_ready: true`
- direct app inference returned a real `.wav` from the fine-tuned checkpoint

Current local artifact reports:

- `base_model`: `models/speecht5-finetuned`
- `device`: `mps`
- `train_rows`: `128`
- `eval_rows`: `32`

When running on Apple Silicon, VoiceForge now generates mel spectrograms on MPS and runs the vocoder on CPU so preview generation and demo inference do not hit the unsupported MPS convolution path.

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
- `scripts/run_tiny_smoke_train.py` is the quickest way to verify the local fine-tune path without copying a long `train_model.py` command by hand.
- `scripts/prune_checkpoints.py --dry-run` shows which old `checkpoint-*` folders would be removed from a local run directory.
- After each training run, VoiceForge generates preview `.wav` files under `models/speecht5-finetuned/previews`.
- `--max-audio-seconds` and `--max-text-chars` trim the worst outlier utterances, which matters a lot on Apple Silicon.
- `--group-by-target-length` reduces padding waste for SpeechT5 batches.
- `device_info.json` and `artifact.json` now record the requested device, resolved device, and filtered row counts for each run.
