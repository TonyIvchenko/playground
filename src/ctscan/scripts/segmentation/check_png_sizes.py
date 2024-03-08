from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from PIL import Image, UnidentifiedImageError


CTSCAN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = CTSCAN_ROOT / "data" / "legacy_compatible_png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report distinct PNG sizes in a legacy CT PNG dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=20, help="Max number of pair mismatches to print.")
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def collect_sizes(folder: Path) -> tuple[Counter[tuple[int, int]], dict[str, tuple[int, int]], list[str]]:
    counts: Counter[tuple[int, int]] = Counter()
    by_stem: dict[str, tuple[int, int]] = {}
    invalid: list[str] = []
    for path in sorted(folder.glob("*.png")):
        try:
            size = image_size(path)
        except (UnidentifiedImageError, OSError) as exc:
            invalid.append(f"{path.name}: {type(exc).__name__}")
            continue
        counts[size] += 1
        by_stem[path.stem] = size
    return counts, by_stem, invalid


def print_counts(label: str, counts: Counter[tuple[int, int]]) -> None:
    total = sum(counts.values())
    print(f"{label}: total={total}")
    for size, count in sorted(counts.items()):
        print(f"  {size[0]}x{size[1]}: {count}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"missing dataset dirs under {root}", file=sys.stderr)
        return 1

    image_counts, image_sizes, image_invalid = collect_sizes(images_dir)
    mask_counts, mask_sizes, mask_invalid = collect_sizes(masks_dir)

    print_counts("images", image_counts)
    print_counts("masks", mask_counts)
    print(f"invalid_images={len(image_invalid)}")
    for line in image_invalid[: max(int(args.limit), 0)]:
        print(f"  {line}")
    print(f"invalid_masks={len(mask_invalid)}")
    for line in mask_invalid[: max(int(args.limit), 0)]:
        print(f"  {line}")

    all_stems = sorted(set(image_sizes) | set(mask_sizes))
    mismatches: list[str] = []
    for stem in all_stems:
        image_size_value = image_sizes.get(stem)
        mask_size_value = mask_sizes.get(stem)
        if image_size_value is None or mask_size_value is None:
            mismatches.append(
                f"{stem}: image={image_size_value or 'missing'} mask={mask_size_value or 'missing'}"
            )
            continue
        if image_size_value != mask_size_value:
            mismatches.append(f"{stem}: image={image_size_value} mask={mask_size_value}")

    print(f"pair_mismatches={len(mismatches)}")
    for line in mismatches[: max(int(args.limit), 0)]:
        print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
