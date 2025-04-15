from __future__ import annotations

import argparse
import json
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = SERVICE_DIR / "data" / "voiceforge" / "raw"
DEFAULT_LIBRITTS_SPLITS = ("train-clean-100", "dev-clean")


def _require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit("torchaudio is required for dataset download. Install src/voiceforge/requirements.txt first.") from exc
    return torchaudio


def ensure_libritts(root: Path, split: str) -> dict[str, str]:
    torchaudio = _require_torchaudio()
    root.mkdir(parents=True, exist_ok=True)
    torchaudio.datasets.LIBRITTS(root=str(root), url=split, download=True)
    return {
        "dataset": "libritts",
        "split": split,
        "root": str(root),
        "expected_subdir": str(root / "LibriTTS" / split),
    }


def ensure_vctk(root: Path, mic_id: str) -> dict[str, str]:
    torchaudio = _require_torchaudio()
    root.mkdir(parents=True, exist_ok=True)
    try:
        torchaudio.datasets.VCTK_092(root=str(root), mic_id=mic_id, download=True)
    except TypeError:
        torchaudio.datasets.VCTK_092(root=str(root), download=True)
    return {
        "dataset": "vctk_092",
        "mic_id": mic_id,
        "root": str(root),
        "expected_subdir": str(root / "VCTK-Corpus-0.92"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download open speech datasets for VoiceForge.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--libritts-splits",
        default=",".join(DEFAULT_LIBRITTS_SPLITS),
        help="Comma-separated LibriTTS splits to download.",
    )
    parser.add_argument("--skip-libritts", action="store_true")
    parser.add_argument("--skip-vctk", action="store_true")
    parser.add_argument("--vctk-mic-id", default="mic1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[dict[str, str]] = []
    if not args.skip_libritts:
        for split in [item.strip() for item in args.libritts_splits.split(",") if item.strip()]:
            downloads.append(ensure_libritts(raw_dir / "libritts", split))
    if not args.skip_vctk:
        downloads.append(ensure_vctk(raw_dir / "vctk", args.vctk_mic_id))

    manifest = {
        "raw_dir": str(raw_dir),
        "downloads": downloads,
    }
    manifest_path = raw_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote download manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
