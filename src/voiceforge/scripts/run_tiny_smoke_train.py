from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


SERVICE_DIR = Path(__file__).resolve().parents[1]
TRAIN_MODEL_SCRIPT = Path(__file__).resolve().with_name("train_model.py")
SMOKE_TRAIN_PRESET_ARGS = [
    "--epochs",
    "1",
    "--batch-size",
    "1",
    "--eval-batch-size",
    "1",
    "--gradient-accumulation-steps",
    "1",
    "--max-train-samples",
    "32",
    "--max-eval-samples",
    "8",
    "--preview-samples",
    "1",
    "--logging-steps",
    "4",
    "--save-steps",
    "16",
    "--eval-steps",
    "16",
    "--save-total-limit",
    "1",
]


def normalize_passthrough_args(extra_args: list[str]) -> list[str]:
    if extra_args[:1] == ["--"]:
        return extra_args[1:]
    return extra_args


def build_smoke_train_command(
    *,
    python_executable: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    return [
        python_executable,
        str(TRAIN_MODEL_SCRIPT),
        *SMOKE_TRAIN_PRESET_ARGS,
        *normalize_passthrough_args(extra_args or []),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tiny VoiceForge smoke-train preset."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without starting training.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the preset summary as JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    passthrough_args = normalize_passthrough_args(extra_args)
    command = build_smoke_train_command(
        python_executable=args.python,
        extra_args=passthrough_args,
    )
    summary = {
        "service_dir": str(SERVICE_DIR),
        "train_script": str(TRAIN_MODEL_SCRIPT),
        "preset_args": SMOKE_TRAIN_PRESET_ARGS,
        "passthrough_args": passthrough_args,
        "command": command,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        if args.dry_run:
            return
    elif args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return
    else:
        print(
            "Running tiny smoke-train preset:\n"
            + " ".join(shlex.quote(part) for part in command),
            flush=True,
        )

    raise SystemExit(subprocess.run(command, cwd=SERVICE_DIR).returncode)


if __name__ == "__main__":
    main()
