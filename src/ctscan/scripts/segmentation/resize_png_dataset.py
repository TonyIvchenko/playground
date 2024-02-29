from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from PIL import Image


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = CTSCAN_ROOT / "data" / "legacy_compatible_png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize legacy CT PNG dataset in place.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _target_size(size: int) -> tuple[int, int]:
    value = max(int(size), 1)
    return value, value


def _folder_config(name: str) -> tuple[str, int]:
    if name == "images":
        return "L", Image.Resampling.BILINEAR
    if name == "masks":
        return "L", Image.Resampling.NEAREST
    raise ValueError(f"unsupported folder: {name}")


def resize_folder(folder: Path, size: int, dry_run: bool) -> dict[str, object]:
    mode, resample = _folder_config(folder.name)
    target = _target_size(size)
    resized = 0
    unchanged = 0
    counts: Counter[tuple[int, int]] = Counter()

    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as image:
            original_size = image.size
            counts[(int(original_size[0]), int(original_size[1]))] += 1
            if original_size == target:
                unchanged += 1
                continue
            resized_image = image.convert(mode).resize(target, resample=resample)
        resized += 1
        if dry_run:
            continue
        temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        resized_image.save(temp_path)
        temp_path.replace(path)

    return {
        "folder": str(folder),
        "total": resized + unchanged,
        "resized": resized,
        "unchanged": unchanged,
        "distinct_sizes": {f"{width}x{height}": count for (width, height), count in sorted(counts.items())},
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    size = max(int(args.size), 1)
    dry_run = bool(args.dry_run)

    summaries: list[dict[str, object]] = []
    for name in ("images", "masks"):
        folder = root / name
        if not folder.exists():
            print(f"missing folder: {folder}", file=sys.stderr)
            return 1
        summaries.append(resize_folder(folder, size=size, dry_run=dry_run))

    print(f"root={root}")
    print(f"target_size={size}x{size}")
    print(f"dry_run={str(dry_run).lower()}")
    for summary in summaries:
        print(
            f"{Path(str(summary['folder'])).name}: "
            f"total={summary['total']} resized={summary['resized']} unchanged={summary['unchanged']}"
        )
        for dims, count in dict(summary["distinct_sizes"]).items():
            print(f"  {dims}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
