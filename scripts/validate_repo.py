#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import re
import subprocess
import sys
from typing import Any

import yaml

try:
    from .project_config import ROOT, load_playground_config
except ImportError:
    from project_config import ROOT, load_playground_config

REPO_CONFIG = load_playground_config()
DOCS_CONFIG = REPO_CONFIG["docs"]
TRACKED_CONFIG_PATTERNS = tuple(REPO_CONFIG["config"]["tracked_patterns"])
AVAILABLE_CHECKS = ("docs", "config", "hygiene")

HEADING_RE = re.compile(r"^(#{1,6}) (.+\S)\s*$")
FENCE_RE = re.compile(r"^```")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
URL_RE = re.compile(r"https?://\S+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']+\b")
CODE_BLOCK_PATTERN = re.compile(r"(?ms)^```(?P<info>[^\n`]*)\n(?P<body>.*?)^```$")
PATH_PATTERNS = tuple(
    re.compile(pattern) for pattern in DOCS_CONFIG["readme_code_path_patterns"]
)
TYPO_FIXES = {
    "acheive": "achieve",
    "arguement": "argument",
    "definately": "definitely",
    "enviroment": "environment",
    "huricaines": "hurricanes",
    "mispelled": "misspelled",
    "occurence": "occurrence",
    "recieve": "receive",
    "seperate": "separate",
    "teh": "the",
}
TRACKED_JUNK_NAMES = {".DS_Store"}
TRACKED_JUNK_PARTS = {"__pycache__"}


class ValidationError(RuntimeError):
    pass


class DuplicateKeyError(ValueError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    pass


@dataclass(frozen=True)
class MissingPath:
    readme: Path
    line: int
    referenced_path: str
    resolved_path: Path


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


def expand_globs(patterns: list[str]) -> list[Path]:
    return sorted({path for pattern in patterns for path in ROOT.glob(pattern)})


README_PATHS = expand_globs(list(DOCS_CONFIG["readme_globs"]))
DOC_PATHS = expand_globs(list(DOCS_CONFIG["doc_globs"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repo docs, config, and hygiene validation from one command."
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        choices=AVAILABLE_CHECKS,
        help=f"Run only a named validation group. Available: {', '.join(AVAILABLE_CHECKS)}.",
    )
    return parser.parse_args()


def lint_readme_markdown(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    first_nonempty_line: tuple[int, str] | None = None
    h1_count = 0
    previous_heading_level = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_fence and stripped and first_nonempty_line is None:
            first_nonempty_line = (line_number, line)

        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue

        if in_fence or not line.startswith("#"):
            continue

        heading_match = HEADING_RE.match(line)
        if not heading_match:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} uses an invalid heading format"
            )
            continue

        level = len(heading_match.group(1))
        if level == 1:
            h1_count += 1

        if previous_heading_level and level > previous_heading_level + 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} jumps from heading level "
                f"{previous_heading_level} to {level}"
            )

        previous_heading_level = level

    if in_fence:
        errors.append(f"{path.relative_to(ROOT)} has an unclosed fenced code block")

    if first_nonempty_line is None:
        errors.append(f"{path.relative_to(ROOT)} is empty")
    else:
        line_number, line = first_nonempty_line
        if not line.startswith("# "):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} should start with a single H1 heading"
            )

    if h1_count != 1:
        errors.append(
            f"{path.relative_to(ROOT)} should contain exactly one H1 heading (found {h1_count})"
        )

    return errors


def prose_lines(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    out: list[tuple[int, str]] = []

    for line_number, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        line = URL_RE.sub(" ", line)
        line = MARKDOWN_LINK_RE.sub(r"\1", line)
        line = INLINE_CODE_RE.sub(" ", line)
        out.append((line_number, line))

    return out


def lint_doc_spelling(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in prose_lines(path):
        for match in WORD_RE.finditer(line):
            word = match.group(0)
            typo = word.lower()
            if typo not in TYPO_FIXES:
                continue
            suggestion = TYPO_FIXES[typo]
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} uses '{word}' "
                f"(prefer '{suggestion}')"
            )
    return errors


def extract_code_path_candidates(readme_path: Path) -> list[tuple[int, str]]:
    text = readme_path.read_text(encoding="utf-8")
    seen: set[tuple[int, str]] = set()
    candidates: list[tuple[int, str]] = []

    for block_match in CODE_BLOCK_PATTERN.finditer(text):
        body = block_match.group("body")
        block_offset = block_match.start("body")
        for pattern in PATH_PATTERNS:
            for path_match in pattern.finditer(body):
                candidate = path_match.group(1)
                candidate_offset = block_offset + path_match.start(1)
                line = text.count("\n", 0, candidate_offset) + 1
                key = (line, candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((line, candidate))

    return candidates


def resolve_code_path_candidate(readme_path: Path, candidate: str) -> Path:
    if candidate.startswith("src/") or candidate == "environment.yml":
        return ROOT / candidate
    return readme_path.parent / candidate


def find_missing_code_paths() -> tuple[list[MissingPath], int]:
    missing: list[MissingPath] = []
    checked = 0

    for readme_path in README_PATHS:
        for line, candidate in extract_code_path_candidates(readme_path):
            resolved = resolve_code_path_candidate(readme_path, candidate)
            checked += 1
            if resolved.exists():
                continue
            missing.append(
                MissingPath(
                    readme=readme_path.relative_to(ROOT),
                    line=line,
                    referenced_path=candidate,
                    resolved_path=resolved.relative_to(ROOT),
                )
            )
    return missing, checked


def run_docs_checks() -> dict[str, Any]:
    results: dict[str, Any] = {}

    markdown_errors: list[str] = []
    for path in README_PATHS:
        markdown_errors.extend(lint_readme_markdown(path))
    results["readme-markdown"] = {
        "errors": markdown_errors,
        "readme_count": len(README_PATHS),
    }

    spelling_errors: list[str] = []
    for path in DOC_PATHS:
        spelling_errors.extend(lint_doc_spelling(path))
    results["docs-spelling"] = {
        "errors": spelling_errors,
        "doc_count": len(DOC_PATHS),
    }

    missing, checked = find_missing_code_paths()
    results["readme-code-paths"] = {
        "errors": [
            f"{item.readme}:{item.line} references '{item.referenced_path}' "
            f"but {item.resolved_path} does not exist"
            for item in missing
        ],
        "readme_count": len(README_PATHS),
        "checked_paths": checked,
    }

    return results


def tracked_config_paths() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    paths: list[Path] = []
    for rel_path in tracked:
        pure_path = PurePosixPath(rel_path)
        if any(pure_path.match(pattern) for pattern in TRACKED_CONFIG_PATTERNS):
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


def validate_config_path(path: Path) -> str | None:
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


def run_config_checks() -> dict[str, Any]:
    paths = tracked_config_paths()
    return {
        "errors": [
            error for path in paths if (error := validate_config_path(path)) is not None
        ],
        "path_count": len(paths),
    }


def tracked_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    entries = completed.stdout.decode("utf-8").split("\0")
    return [entry for entry in entries if entry]


def junk_paths() -> list[str]:
    matches: list[str] = []
    for entry in tracked_paths():
        path = PurePosixPath(entry)
        if path.name in TRACKED_JUNK_NAMES or any(
            part in TRACKED_JUNK_PARTS for part in path.parts
        ):
            matches.append(entry)
    return sorted(matches)


def run_hygiene_checks() -> dict[str, Any]:
    matches = junk_paths()
    return {
        "errors": matches,
        "match_count": len(matches),
    }


def run_checks(selected_checks: list[str] | None = None) -> dict[str, Any]:
    selected = selected_checks or list(AVAILABLE_CHECKS)
    unknown = sorted(set(selected) - set(AVAILABLE_CHECKS))
    if unknown:
        raise ValidationError("Unknown repo checks: " + ", ".join(unknown))

    results: dict[str, Any] = {}
    if "docs" in selected:
        results["docs"] = run_docs_checks()
    if "config" in selected:
        results["config"] = run_config_checks()
    if "hygiene" in selected:
        results["hygiene"] = run_hygiene_checks()
    return results


def docs_summary(result: dict[str, Any]) -> str:
    return ", ".join(
        [
            f"readme-markdown={result['readme-markdown']['readme_count']} README files",
            f"docs-spelling={result['docs-spelling']['doc_count']} markdown files",
            "readme-code-paths="
            f"{result['readme-code-paths']['readme_count']} README files/"
            f"{result['readme-code-paths']['checked_paths']} paths",
        ]
    )


def main() -> int:
    args = parse_args()
    try:
        results = run_checks(args.check)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    failed = False
    if "docs" in results:
        docs_failures = {
            name: data["errors"]
            for name, data in results["docs"].items()
            if data["errors"]
        }
        if docs_failures:
            failed = True
            print("Repo validation failed:", file=sys.stderr)
            print("[docs]", file=sys.stderr)
            for check_name, errors in docs_failures.items():
                print(f"  [{check_name}]", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)

    if "config" in results and results["config"]["errors"]:
        if not failed:
            print("Repo validation failed:", file=sys.stderr)
        failed = True
        print("[config]", file=sys.stderr)
        for error in results["config"]["errors"]:
            print(f"- {error}", file=sys.stderr)

    if "hygiene" in results and results["hygiene"]["errors"]:
        if not failed:
            print("Repo validation failed:", file=sys.stderr)
        failed = True
        print("[hygiene]", file=sys.stderr)
        for path in results["hygiene"]["errors"]:
            print(f"- {path}", file=sys.stderr)

    if failed:
        return 1

    summaries: list[str] = []
    if "docs" in results:
        summaries.append(f"docs={docs_summary(results['docs'])}")
    if "config" in results:
        summaries.append(f"config={results['config']['path_count']} tracked files")
    if "hygiene" in results:
        summaries.append("hygiene=no tracked .DS_Store or __pycache__ paths")

    print("Repo validation passed for " + "; ".join(summaries) + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
