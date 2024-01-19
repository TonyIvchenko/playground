from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.ctscan.scripts.segmentation.build_lndb_manifest import (
    build_rows,
    discover_pairs,
    load_pairs_from_csv,
    merge_rows,
)


def test_load_pairs_from_csv_resolves_relative_paths(tmp_path: Path):
    root = tmp_path / "lndb"
    root.mkdir(parents=True, exist_ok=True)

    image_path = root / "scan_a.npy"
    mask_path = root / "scan_a_label.npy"
    np.save(image_path, np.zeros((4, 8, 8), dtype=np.float32))
    np.save(mask_path, np.ones((4, 8, 8), dtype=np.uint8))

    pairs_csv = root / "pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "image_path", "mask_path", "spacing_z", "spacing_y", "spacing_x"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "lndb_case_a",
                "image_path": "scan_a.npy",
                "mask_path": "scan_a_label.npy",
                "spacing_z": "1.2",
                "spacing_y": "0.7",
                "spacing_x": "0.7",
            }
        )

    pairs = load_pairs_from_csv(pairs_csv, root)
    assert len(pairs) == 1
    assert pairs[0].spacing_zyx == (1.2, 0.7, 0.7)


def test_build_rows_from_explicit_pair_and_merge_replace(tmp_path: Path):
    root = tmp_path / "lndb"
    root.mkdir(parents=True, exist_ok=True)

    image_path = root / "scan.npy"
    mask_path = root / "scan_mask.npy"
    np.save(image_path, np.random.randn(4, 8, 8).astype(np.float32))
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[:, 2:5, 2:5] = 2
    np.save(mask_path, mask)

    pairs_csv = root / "pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerow({"case_id": "scan_case", "image_path": "scan.npy", "mask_path": "scan_mask.npy"})

    pairs = load_pairs_from_csv(pairs_csv, root)
    rows = build_rows(
        pairs=pairs,
        output_dir=root / "cases",
        label_map={},
        target_class=5,
        overwrite=True,
        include_empty_masks=False,
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "lndb"

    existing = [{"case_id": "old", "source": "lndb"}, {"case_id": "other", "source": "lidc_idri"}]
    merged = merge_rows(existing, rows, replace_lndb_rows=True)
    ids = {row["case_id"] for row in merged}
    assert "old" not in ids
    assert "other" in ids
    assert "scan_case" in ids


def test_discover_pairs_skips_cases_folder_and_preserves_rad_case_id(tmp_path: Path):
    root = tmp_path / "lndb"
    root.mkdir(parents=True, exist_ok=True)

    image_path = root / "LNDb-0001.npy"
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / "LNDb-0001_rad2.npy"
    np.save(image_path, np.zeros((2, 4, 4), dtype=np.float32))
    np.save(mask_path, np.ones((2, 4, 4), dtype=np.uint8))

    # This should be ignored by discovery.
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cases_dir / "junk_case.npz", image=np.zeros((2, 4, 4), dtype=np.float32), mask=np.zeros((2, 4, 4), dtype=np.uint8))

    pairs = discover_pairs(root, max_cases=0)
    assert len(pairs) == 1
    assert pairs[0].case_id == "LNDb-0001_rad2"


def test_build_rows_include_empty_masks(tmp_path: Path):
    root = tmp_path / "lndb"
    root.mkdir(parents=True, exist_ok=True)

    image_path = root / "scan.npy"
    mask_path = root / "scan_mask.npy"
    np.save(image_path, np.random.randn(2, 4, 4).astype(np.float32))
    np.save(mask_path, np.zeros((2, 4, 4), dtype=np.uint8))

    pairs_csv = root / "pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerow({"case_id": "scan_case", "image_path": "scan.npy", "mask_path": "scan_mask.npy"})

    pairs = load_pairs_from_csv(pairs_csv, root)
    rows_skipping = build_rows(
        pairs=pairs,
        output_dir=root / "cases_skip",
        label_map={},
        target_class=5,
        overwrite=True,
        include_empty_masks=False,
    )
    rows_including = build_rows(
        pairs=pairs,
        output_dir=root / "cases_keep",
        label_map={},
        target_class=5,
        overwrite=True,
        include_empty_masks=True,
    )
    assert len(rows_skipping) == 0
    assert len(rows_including) == 1
