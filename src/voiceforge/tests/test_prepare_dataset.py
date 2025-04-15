from pathlib import Path

from src.voiceforge.scripts.prepare_dataset import (
    clean_text,
    collect_libritts_records,
    collect_vctk_records,
    speaker_balanced_split,
    trim_records,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello\n   there   ") == "hello there"


def test_collect_libritts_records(tmp_path: Path):
    clip_dir = tmp_path / "libritts" / "LibriTTS" / "train-clean-100" / "19" / "198"
    clip_dir.mkdir(parents=True)
    (clip_dir / "19_198_000001.wav").write_bytes(b"wav")
    (clip_dir / "19_198_000001.normalized.txt").write_text("Hello world\n", encoding="utf-8")

    rows = collect_libritts_records(tmp_path, ["train-clean-100"])
    assert len(rows) == 1
    assert rows[0]["speaker_id"] == "libritts_19"
    assert rows[0]["text"] == "Hello world"


def test_collect_vctk_records(tmp_path: Path):
    base = tmp_path / "vctk" / "VCTK-Corpus-0.92"
    txt_dir = base / "txt" / "p225"
    wav_dir = base / "wav48_silence_trimmed" / "p225"
    txt_dir.mkdir(parents=True)
    wav_dir.mkdir(parents=True)
    (txt_dir / "p225_001.txt").write_text("Testing one two three", encoding="utf-8")
    (wav_dir / "p225_001_mic1.flac").write_bytes(b"flac")

    rows = collect_vctk_records(tmp_path, "mic1")
    assert len(rows) == 1
    assert rows[0]["speaker_id"] == "vctk_p225"
    assert rows[0]["utterance_id"] == "vctk_p225_001"


def test_trim_and_split_records():
    rows = []
    for speaker in ["a", "b"]:
        for idx in range(4):
            rows.append(
                {
                    "source": "libritts",
                    "source_split": "train-clean-100",
                    "speaker_id": speaker,
                    "utterance_id": f"{speaker}_{idx}",
                    "audio_path": f"/{speaker}_{idx}.wav",
                    "text": f"sample {idx}",
                    "language": "en",
                }
            )

    trimmed = trim_records(rows, 3)
    assert len(trimmed) == 6

    train_rows, eval_rows = speaker_balanced_split(trimmed, eval_items_per_speaker=1)
    assert len(eval_rows) == 2
    assert len(train_rows) == 4
    assert {row["speaker_id"] for row in eval_rows} == {"a", "b"}
