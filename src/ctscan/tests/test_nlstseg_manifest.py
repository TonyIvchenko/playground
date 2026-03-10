from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ctscan.scripts.segmentation.build_nlstseg_manifest import (
    build_rows,
    discover_pairs,
    merge_rows,
    strip_mask_suffix,
)


def test_strip_mask_suffix_removes_known_suffixes():
    assert strip_mask_suffix("case_001_mask") == "case_001"
    assert strip_mask_suffix("case-001-segmentation") == "case-001"


def test_discover_pairs_and_build_rows(tmp_path: Path):
    root = tmp_path / "nlstseg"
    root.mkdir(parents=True, exist_ok=True)

    image_path = root / "patient01.npy"
    mask_path = root / "patient01_mask.npy"
    np.save(image_path, np.random.randn(8, 16, 16).astype(np.float32))
    mask = np.zeros((8, 16, 16), dtype=np.uint8)
    mask[2:4, 6:10, 6:10] = 1
    np.save(mask_path, mask)

    pairs = discover_pairs(root, max_cases=0)
    assert len(pairs) == 1

    rows = build_rows(
        pairs=pairs,
        output_dir=root / "cases",
        label_map={},
        target_class=5,
        overwrite=True,
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "nlstseg"


def test_merge_rows_can_replace_nlstseg_source():
    existing = [
        {"case_id": "a", "source": "lidc_idri"},
        {"case_id": "b", "source": "nlstseg"},
    ]
    incoming = [{"case_id": "c", "source": "nlstseg"}]
    merged = merge_rows(existing, incoming, replace_nlstseg_rows=True)
    ids = {row["case_id"] for row in merged}
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids
