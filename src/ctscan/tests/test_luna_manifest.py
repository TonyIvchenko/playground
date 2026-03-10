from __future__ import annotations

import numpy as np

from src.ctscan.scripts.segmentation.build_luna_manifest import (
    draw_ellipsoid_mask,
    merge_rows,
    world_to_voxel_zyx,
)


def test_world_to_voxel_zyx_conversion():
    coord_xyz = np.asarray([12.0, 24.0, 36.0], dtype=np.float32)
    origin_xyz = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
    spacing_xyz = np.asarray([2.0, 4.0, 6.0], dtype=np.float32)
    voxel_zyx = world_to_voxel_zyx(coord_xyz, origin_xyz, spacing_xyz)
    assert np.allclose(voxel_zyx, np.asarray([5.0, 5.0, 5.0], dtype=np.float32))


def test_draw_ellipsoid_mask_marks_positive_region():
    mask = np.zeros((16, 32, 32), dtype=np.uint8)
    draw_ellipsoid_mask(
        mask_zyx=mask,
        center_zyx=np.asarray([8.0, 16.0, 16.0], dtype=np.float32),
        radius_mm=4.0,
        spacing_zyx=np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
    )
    assert int((mask > 0).sum()) > 0
    assert int(mask[8, 16, 16]) == 5


def test_luna_merge_rows_can_replace_luna_source():
    existing = [
        {"case_id": "a", "source": "lidc_idri"},
        {"case_id": "b", "source": "luna16"},
    ]
    incoming = [{"case_id": "c", "source": "luna16"}]
    merged = merge_rows(existing, incoming, replace_luna_rows=True)
    ids = {row["case_id"] for row in merged}
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids
