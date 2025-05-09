#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRACKED_PATTERNS = (
    ".github/**/*.yml",
    ".github/**/*.yaml",
    "*.yml",
    "*.yaml",
    ".vscode/**/*.json",
    "src/disasters/tiles/*/overlay.json",
)


class DuplicateKeyError(ValueError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def construct_unique_mapping(
    loader: UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DuplicateKeyError(f"duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
)


def tracked_config_paths() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    paths: list[Path] = []
    for rel_path in tracked:
        pure_path = PurePosixPath(rel_path)
        if any(pure_path.match(pattern) for pattern in TRACKED_PATTERNS):
            paths.append(ROOT / rel_path)
    return sorted(paths)


def load_json_with_unique_keys(path: Path) -> None:
    def unique_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        json.load(handle, object_pairs_hook=unique_object_pairs)


def load_yaml_with_unique_keys(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        yaml.load(handle, Loader=UniqueKeyLoader)


def validate_path(path: Path) -> str | None:
    rel_path = path.relative_to(ROOT)
    try:
        if path.suffix == ".json":
            load_json_with_unique_keys(path)
        elif path.suffix in {".yml", ".yaml"}:
            load_yaml_with_unique_keys(path)
        else:
            return f"{rel_path}: unsupported config extension"
    except (json.JSONDecodeError, yaml.YAMLError, DuplicateKeyError) as exc:
        return f"{rel_path}: {exc}"
    return None


def main() -> int:
    paths = tracked_config_paths()
    failures = [error for path in paths if (error := validate_path(path)) is not None]
    if failures:
        print("JSON/YAML config lint failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"JSON/YAML config lint passed for {len(paths)} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
