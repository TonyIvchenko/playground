#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

try:
    from .service_manifest import (
        ROOT,
        TYPE_REQUIREMENTS,
        iter_service_dirs,
        load_service_manifest,
        service_names,
    )
except ImportError:
    from service_manifest import (
        ROOT,
        TYPE_REQUIREMENTS,
        iter_service_dirs,
        load_service_manifest,
        service_names,
    )

DEFAULT_PORT = 18080
DEFAULT_TIMEOUT = 30.0
SERVICE_SPECS = load_service_manifest()
BROWSER_CONTRACTS_PATH = ROOT / "scripts" / "browser_app_contracts.json"
BUTTON_TAG_RE = re.compile(r"<button(?P<attrs>[^>]*)>", re.MULTILINE)
CLASS_ATTR_RE = re.compile(r'\bclass=["\'](?P<classes>[^"\']+)["\']')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Playground services from the shared service manifest."
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=("static", "browser-apps", "local-run"),
        help="Validation group to run. Default is static.",
    )
    parser.add_argument(
        "--service",
        choices=["all", *service_names()],
        default="all",
        help="Service to validate for local-run checks. Default runs all services.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Base port for smoke-checked local runs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for each local-run check.",
    )
    return parser.parse_args()


def expected_paths_for_service(service_name: str) -> tuple[str, ...]:
    service_type = str(SERVICE_SPECS[service_name]["type"])
    try:
        return TYPE_REQUIREMENTS[service_type]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown service type '{service_type}' for service '{service_name}'."
        ) from exc


def missing_required_paths(service_dir: Path) -> list[str]:
    return [
        name
        for name in expected_paths_for_service(service_dir.name)
        if not (service_dir / name).exists()
    ]


def collect_static_failures() -> list[str]:
    failures: list[str] = []
    repo_services = {path.name for path in iter_service_dirs()}
    manifest_services = set(SERVICE_SPECS)

    for service_name in sorted(repo_services - manifest_services):
        failures.append(f"{service_name}: missing service manifest entry")
    for service_name in sorted(manifest_services - repo_services):
        failures.append(f"{service_name}: stale service manifest entry")

    for service_dir in iter_service_dirs():
        if service_dir.name not in SERVICE_SPECS:
            continue

        missing = missing_required_paths(service_dir)
        if missing:
            failures.append(f"{service_dir.name}: missing {', '.join(missing)}")
            continue

        health_endpoint = SERVICE_SPECS[service_dir.name].get("health_endpoint")
        if not (service_dir / "Dockerfile").exists() or not health_endpoint:
            continue

        source = (service_dir / "main.py").read_text(encoding="utf-8", errors="replace")
        endpoint = str(health_endpoint)
        if f'"{endpoint}"' not in source and f"'{endpoint}'" not in source:
            failures.append(
                f"{service_dir.name}: Dockerized HTTP service is missing {endpoint}"
            )

    return failures


def load_browser_contracts() -> dict[str, Any]:
    return json.loads(BROWSER_CONTRACTS_PATH.read_text(encoding="utf-8"))


def browser_service_path(service: str) -> str:
    return f"src/{service}/index.html"


def read_repo_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def iter_browser_services(check: dict[str, Any]) -> list[str]:
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


def validate_browser_text_rules(
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


def validate_browser_contract_check(check: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    for file_rule in check.get("file_rules", []):
        relative_path = str(file_rule["path"])
        failures.extend(
            validate_browser_text_rules(
                relative_path,
                read_repo_text(relative_path),
                file_rule,
            )
        )

    for service in iter_browser_services(check):
        relative_path = browser_service_path(service)
        text = read_repo_text(relative_path)
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
        failures.extend(validate_browser_text_rules(relative_path, text, service_rule))

    for case in check.get("contrast_cases", []):
        ratio = contrast_ratio(case["foreground"], case["background"])
        minimum = float(case["minimum"])
        if ratio < minimum:
            failures.append(
                f"{case['label']} contrast is {ratio:.2f}, below the required {minimum:.1f}."
            )

    return failures


def collect_browser_contract_failures() -> tuple[dict[str, list[str]], int]:
    check_specs = load_browser_contracts().get("checks", [])
    failures_by_check: dict[str, list[str]] = {}
    for check in check_specs:
        failures = validate_browser_contract_check(check)
        if failures:
            failures_by_check[str(check["name"])] = failures
    return failures_by_check, len(check_specs)


def run_browser_contract_checks() -> None:
    failures_by_check, check_count = collect_browser_contract_failures()
    if failures_by_check:
        details: list[str] = []
        for check_name, failures in failures_by_check.items():
            details.append(f"[{check_name}]")
            details.extend(f"- {failure}" for failure in failures)
        raise SystemExit("Browser app contract check failed:\n" + "\n".join(details))
    print(f"Browser app contract check passed for {check_count} checks.")


def run_static_checks() -> None:
    failures = collect_static_failures()
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"Service validation failed:\n{details}")

    type_counts: dict[str, int] = {}
    for spec in SERVICE_SPECS.values():
        service_type = str(spec["type"])
        type_counts[service_type] = type_counts.get(service_type, 0) + 1
    counts = ", ".join(
        f"{service_type}={count}" for service_type, count in sorted(type_counts.items())
    )
    print(
        f"Static service validation passed for {len(SERVICE_SPECS)} services ({counts})."
    )


def service_dir(name: str) -> Path:
    for path in iter_service_dirs():
        if path.name == name:
            return path
    raise SystemExit(f"Unknown service '{name}'.")


def readme_path(name: str) -> Path:
    return service_dir(name) / "README.md"


def readme_contains_local_run(path: Path, command: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return "## Local Run" in text and command in text


def service_env(name: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in SERVICE_SPECS[name]["local_run"].get("env", {}).items():
        env.setdefault(str(key), str(value))
    return env


def run_smoke_service(name: str, port: int, timeout: float) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/smoke_service.py"),
        name,
        "--port",
        str(port),
        "--timeout",
        str(timeout),
    ]
    print("Running " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=service_env(name), check=True)


def run_oneshot_service(name: str, timeout: float) -> None:
    path = service_dir(name)
    argv = [str(part) for part in SERVICE_SPECS[name]["local_run"].get("argv", [])]
    command = [sys.executable, "main.py", *argv]
    print("Running " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(
        command,
        cwd=path,
        env=service_env(name),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise SystemExit(
            "One-shot Local Run check failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"Stdout:\n{completed.stdout or '(no stdout)'}\n"
            f"Stderr:\n{completed.stderr or '(no stderr)'}"
        )
    if "--config-json" in argv:
        try:
            payload = json.loads(completed.stdout or "")
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "One-shot Local Run check did not produce valid JSON config output.\n"
                f"Stdout:\n{completed.stdout or '(no stdout)'}"
            ) from exc
        if payload.get("service") != f"{name}-service":
            raise SystemExit(
                "One-shot Local Run check produced unexpected service metadata.\n"
                f"Stdout:\n{completed.stdout or '(no stdout)'}"
            )
        if payload.get("dry_run") is not True:
            raise SystemExit(
                "One-shot Local Run check produced config JSON, but dry_run was not true.\n"
                f"Stdout:\n{completed.stdout or '(no stdout)'}"
            )
    if "Dry run only; skipping Redis connection and write loop" not in output:
        raise SystemExit(
            "One-shot Local Run check passed but did not emit the expected dry-run marker.\n"
            f"Output:\n{output or '(no output)'}"
        )
    print(f"One-shot local run check passed for '{name}'.")


def run_local_service_check(name: str, port: int, timeout: float) -> None:
    local_run = SERVICE_SPECS[name]["local_run"]
    readme_command = str(local_run["readme_command"])
    if not readme_contains_local_run(readme_path(name), readme_command):
        raise SystemExit(
            f"{readme_path(name).relative_to(ROOT)} is missing the expected Local Run command: {readme_command}"
        )

    mode = str(local_run["mode"])
    if mode == "smoke":
        run_smoke_service(name, port, timeout)
        return
    if mode == "oneshot":
        run_oneshot_service(name, timeout)
        return
    raise SystemExit(f"Unsupported local run mode '{mode}' for service '{name}'.")


def run_local_checks(service_name: str, port: int, timeout: float) -> None:
    names = service_names() if service_name == "all" else [service_name]
    for offset, name in enumerate(names):
        run_local_service_check(name, port + offset, timeout)
    print(f"Local run validation passed for {len(names)} services.")


def main() -> int:
    args = parse_args()
    checks = args.check or ["static"]
    for check_name in checks:
        if check_name == "static":
            run_static_checks()
        elif check_name == "browser-apps":
            run_browser_contract_checks()
        elif check_name == "local-run":
            run_local_checks(args.service, args.port, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
