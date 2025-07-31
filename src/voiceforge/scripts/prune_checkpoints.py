from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

SERVICE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = SERVICE_DIR / "models" / "speecht5-finetuned"


def parse_checkpoint_step(checkpoint_dir: Path) -> int:
    name = checkpoint_dir.name
    if not name.startswith("checkpoint-"):
        raise ValueError(f"Not a checkpoint directory: {checkpoint_dir}")
    try:
        return int(name.split("-", 1)[1])
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint directory has a non-numeric step: {checkpoint_dir}"
        ) from exc


def list_checkpoint_dirs(output_dir: Path) -> list[Path]:
    checkpoints = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    return sorted(
        checkpoints, key=lambda path: (parse_checkpoint_step(path), path.name)
    )


def prune_checkpoints(
    output_dir: Path,
    *,
    keep: int = 2,
    dry_run: bool = False,
) -> dict[str, object]:
    if keep < 0:
        raise ValueError("keep must be >= 0")

    output_dir = output_dir.resolve()
    checkpoints = list_checkpoint_dirs(output_dir) if output_dir.exists() else []
    keep_paths = checkpoints[-keep:] if keep > 0 else []
    remove_paths = checkpoints[:-keep] if keep > 0 else checkpoints

    if not dry_run:
        for path in remove_paths:
            shutil.rmtree(path)

    return {
        "output_dir": str(output_dir),
        "keep": keep,
        "dry_run": dry_run,
        "kept_checkpoints": [str(path) for path in keep_paths],
        "removed_checkpoints": [str(path) for path in remove_paths],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune old VoiceForge trainer checkpoints."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--keep",
        type=int,
        default=2,
        help="How many newest checkpoint-* directories to keep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary instead of plain text.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prune_checkpoints(args.output_dir, keep=args.keep, dry_run=args.dry_run)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    action = "Would remove" if args.dry_run else "Removed"
    kept = summary["kept_checkpoints"]
    removed = summary["removed_checkpoints"]
    print(f"Checkpoint directory: {summary['output_dir']}")
    print(f"Keeping newest {args.keep} checkpoint(s).")
    print(f"Kept {len(kept)} checkpoint(s).")
    for path in kept:
        print(f"  keep: {path}")
    print(f"{action} {len(removed)} checkpoint(s).")
    for path in removed:
        print(f"  prune: {path}")


if __name__ == "__main__":
    main()
