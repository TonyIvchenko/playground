from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable


SERVICE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = SERVICE_DIR / "data" / "voiceforge" / "raw"
DEFAULT_OUTPUT_DIR = SERVICE_DIR / "data" / "voiceforge" / "processed"
DEFAULT_LIBRITTS_SPLITS = ("train-clean-100", "dev-clean")


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def collect_libritts_records(raw_dir: Path, splits: Iterable[str]) -> list[dict[str, str]]:
    base = raw_dir / "libritts" / "LibriTTS"
    records: list[dict[str, str]] = []
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for transcript_path in split_dir.rglob("*.normalized.txt"):
            stem = transcript_path.stem.replace(".normalized", "")
            audio_path = transcript_path.with_name(f"{stem}.wav")
            if not audio_path.exists():
                continue
            text = clean_text(transcript_path.read_text(encoding="utf-8"))
            if not text:
                continue
            speaker_id = transcript_path.parts[-3]
            records.append(
                {
                    "source": "libritts",
                    "source_split": split,
                    "speaker_id": f"libritts_{speaker_id}",
                    "utterance_id": f"libritts_{stem}",
                    "audio_path": str(audio_path.resolve()),
                    "text": text,
                    "language": "en",
                }
            )
    return records


def collect_vctk_records(raw_dir: Path, mic_id: str = "mic1") -> list[dict[str, str]]:
    base = raw_dir / "vctk" / "VCTK-Corpus-0.92"
    txt_root = base / "txt"
    wav_root = base / "wav48_silence_trimmed"
    records: list[dict[str, str]] = []
    if not txt_root.exists() or not wav_root.exists():
        return records

    suffix = f"_{mic_id}.flac"
    for transcript_path in txt_root.rglob("*.txt"):
        speaker = transcript_path.parent.name
        utterance = transcript_path.stem
        audio_path = wav_root / speaker / f"{utterance}{suffix}"
        if not audio_path.exists():
            fallback = wav_root / speaker / f"{utterance}.flac"
            if fallback.exists():
                audio_path = fallback
            else:
                continue
        text = clean_text(transcript_path.read_text(encoding="utf-8"))
        if not text:
            continue
        records.append(
            {
                "source": "vctk",
                "source_split": "all",
                "speaker_id": f"vctk_{speaker}",
                "utterance_id": f"vctk_{utterance}",
                "audio_path": str(audio_path.resolve()),
                "text": text,
                "language": "en",
            }
        )
    return records


def trim_records(records: list[dict[str, str]], max_per_speaker: int | None) -> list[dict[str, str]]:
    if max_per_speaker is None:
        return records
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["speaker_id"]].append(record)
    trimmed: list[dict[str, str]] = []
    for speaker_id in sorted(grouped):
        items = sorted(grouped[speaker_id], key=lambda item: item["utterance_id"])
        trimmed.extend(items[:max_per_speaker])
    return trimmed


def speaker_balanced_split(records: list[dict[str, str]], eval_items_per_speaker: int = 2) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["speaker_id"]].append(record)

    train: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for speaker_id in sorted(grouped):
        items = sorted(grouped[speaker_id], key=lambda item: item["utterance_id"])
        eval_count = min(eval_items_per_speaker, max(len(items) - 1, 0))
        if eval_count:
            eval_rows.extend(items[-eval_count:])
            train.extend(items[:-eval_count])
        else:
            train.extend(items)
    return train, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_summary(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> dict[str, object]:
    speakers = sorted({row["speaker_id"] for row in train_rows + eval_rows})
    return {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "speaker_count": len(speakers),
        "sources": sorted({row["source"] for row in train_rows + eval_rows}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SpeechT5 manifests for VoiceForge.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--libritts-splits",
        default=",".join(DEFAULT_LIBRITTS_SPLITS),
        help="Comma-separated LibriTTS splits to scan.",
    )
    parser.add_argument("--skip-vctk", action="store_true")
    parser.add_argument("--vctk-mic-id", default="mic1")
    parser.add_argument("--max-per-speaker", type=int, default=None)
    parser.add_argument("--eval-items-per-speaker", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    libritts_splits = [item.strip() for item in args.libritts_splits.split(",") if item.strip()]
    rows = collect_libritts_records(raw_dir, libritts_splits)
    if not args.skip_vctk:
        rows.extend(collect_vctk_records(raw_dir, args.vctk_mic_id))

    rows = trim_records(rows, args.max_per_speaker)
    train_rows, eval_rows = speaker_balanced_split(rows, args.eval_items_per_speaker)

    train_path = output_dir / "train_manifest.jsonl"
    eval_path = output_dir / "eval_manifest.jsonl"
    summary_path = output_dir / "summary.json"
    references_path = output_dir / "reference_manifest.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    reference_rows: list[dict[str, str]] = []
    seen_speakers: set[str] = set()
    for row in train_rows:
        speaker_id = row["speaker_id"]
        if speaker_id in seen_speakers:
            continue
        seen_speakers.add(speaker_id)
        reference_rows.append(row)
    write_jsonl(references_path, reference_rows)

    summary = build_summary(train_rows, eval_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote train manifest: {train_path}")
    print(f"Wrote eval manifest: {eval_path}")
    print(f"Wrote reference manifest: {references_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
