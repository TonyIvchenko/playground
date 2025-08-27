#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "scripts" / "browser_app_contracts.json"
BUTTON_TAG_RE = re.compile(r"<button(?P<attrs>[^>]*)>", re.MULTILINE)
CLASS_ATTR_RE = re.compile(r'\bclass=["\'](?P<classes>[^"\']+)["\']')


class ValidationError(RuntimeError):
    pass


def load_contracts(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate shared browser-app UI contracts from one config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the browser-app contracts JSON file.",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="Run only a named contract check. Can be passed multiple times.",
    )
    return parser.parse_args()


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def service_path(service: str) -> str:
    return f"src/{service}/index.html"


def iter_services(check: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in (
        "services",
        "required_fragments_by_service",
        "banned_fragments_by_service",
        "required_element_classes_by_service",
        "required_any_of_fragments_by_service",
    ):
        value = check.get(key)
        if isinstance(value, list):
            names.extend(str(item) for item in value)
        elif isinstance(value, dict):
            names.extend(str(item) for item in value.keys())
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def has_class(text: str, class_name: str) -> bool:
    for match in re.finditer(r'class="[^"]*"', text):
        class_names = match.group(0)[7:-1].split()
        if class_name in class_names:
            return True
    return False


def element_has_classes(
    text: str, element_id: str, required_classes: list[str] | tuple[str, ...]
) -> bool:
    pattern = re.compile(
        rf"<[a-zA-Z0-9]+(?P<attrs>[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*)>",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return False
    attrs = match.group("attrs")
    class_match = CLASS_ATTR_RE.search(attrs)
    if not class_match:
        return False
    classes = set(class_match.group("classes").split())
    return all(class_name in classes for class_name in required_classes)


def button_has_class(attrs: str, class_name: str) -> bool:
    match = CLASS_ATTR_RE.search(attrs)
    if not match:
        return False
    classes = set(match.group("classes").split())
    return class_name in classes


def srgb_to_linear(channel: int) -> float:
    value = channel / 255
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return (
        0.2126 * srgb_to_linear(red)
        + 0.7152 * srgb_to_linear(green)
        + 0.0722 * srgb_to_linear(blue)
    )


def contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def normalize_banned_entries(entries: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"text": entry})
        else:
            normalized.append(
                {"text": str(entry["text"]), "guidance": str(entry.get("guidance", ""))}
            )
    return normalized


def merge_rule_values(check: dict[str, Any], key: str, service: str) -> list[Any]:
    values = list(check.get(key, []))
    by_service = check.get(f"{key}_by_service", {})
    if isinstance(by_service, dict):
        values.extend(by_service.get(service, []))
    return values


def validate_text_rules(
    relative_path: str,
    text: str,
    rule: dict[str, Any],
) -> list[str]:
    failures: list[str] = []

    for fragment in rule.get("required_fragments", []):
        if fragment not in text:
            failures.append(f"{relative_path} is missing fragment {fragment!r}.")

    for group in rule.get("required_any_of_fragments", []):
        if not any(fragment in text for fragment in group):
            choices = " or ".join(repr(fragment) for fragment in group)
            failures.append(f"{relative_path} is missing one of: {choices}.")

    for entry in normalize_banned_entries(rule.get("banned_fragments", [])):
        fragment = entry["text"]
        if fragment in text:
            guidance = entry.get("guidance", "")
            suffix = f" {guidance}" if guidance else ""
            failures.append(
                f"{relative_path} still contains banned fragment {fragment!r}.{suffix}"
            )

    for class_name in rule.get("required_classes_anywhere", []):
        if not has_class(text, str(class_name)):
            failures.append(
                f"{relative_path} is missing required class {class_name!r}."
            )

    for element_id, required_classes in rule.get(
        "required_element_classes", {}
    ).items():
        if not element_has_classes(text, str(element_id), list(required_classes)):
            failures.append(
                f"{relative_path} is missing {', '.join(required_classes)} on #{element_id}."
            )

    for pattern in rule.get("banned_regexes", []):
        if re.search(str(pattern), text, re.MULTILINE):
            failures.append(
                f"{relative_path} matches banned regex pattern {pattern!r}."
            )

    button_class = rule.get("button_requires_class")
    if button_class:
        for index, match in enumerate(BUTTON_TAG_RE.finditer(text), start=1):
            if not button_has_class(match.group("attrs"), str(button_class)):
                failures.append(
                    f"{relative_path} button #{index} is missing the shared {button_class} class."
                )

    return failures


def validate_check(check: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    for file_rule in check.get("file_rules", []):
        relative_path = str(file_rule["path"])
        failures.extend(
            validate_text_rules(relative_path, read_text(relative_path), file_rule)
        )

    for service in iter_services(check):
        relative_path = service_path(service)
        text = read_text(relative_path)
        service_rule = {
            "required_fragments": merge_rule_values(
                check, "required_fragments", service
            ),
            "required_any_of_fragments": merge_rule_values(
                check, "required_any_of_fragments", service
            ),
            "banned_fragments": merge_rule_values(check, "banned_fragments", service),
            "required_classes_anywhere": merge_rule_values(
                check, "required_classes_anywhere", service
            ),
            "banned_regexes": merge_rule_values(check, "banned_regexes", service),
            "button_requires_class": check.get("button_requires_class"),
            "required_element_classes": check.get(
                "required_element_classes_by_service", {}
            ).get(service, {}),
        }
        failures.extend(validate_text_rules(relative_path, text, service_rule))

    for case in check.get("contrast_cases", []):
        ratio = contrast_ratio(case["foreground"], case["background"])
        minimum = float(case["minimum"])
        if ratio < minimum:
            failures.append(
                f"{case['label']} contrast is {ratio:.2f}, below the required {minimum:.1f}."
            )

    return failures


def run_checks(
    contracts: dict[str, Any], selected_checks: list[str] | None = None
) -> tuple[dict[str, list[str]], int]:
    check_specs = contracts.get("checks", [])
    names = [str(check["name"]) for check in check_specs]
    if selected_checks:
        unknown = sorted(set(selected_checks) - set(names))
        if unknown:
            raise ValidationError("Unknown browser-app checks: " + ", ".join(unknown))
        check_specs = [
            check for check in check_specs if check["name"] in selected_checks
        ]

    failures_by_check: dict[str, list[str]] = {}
    for check in check_specs:
        failures = validate_check(check)
        if failures:
            failures_by_check[str(check["name"])] = failures
    return failures_by_check, len(check_specs)


def main() -> int:
    args = parse_args()
    try:
        failures_by_check, check_count = run_checks(
            load_contracts(args.config),
            selected_checks=args.check,
        )
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if failures_by_check:
        print("Browser app contract check failed:", file=sys.stderr)
        for check_name, failures in failures_by_check.items():
            print(f"[{check_name}]", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Browser app contract check passed for {check_count} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
