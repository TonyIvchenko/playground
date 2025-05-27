#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SERVICE_SNIPPETS: dict[str, dict[str, object]] = {
    "ctscan": {
        "readme": ROOT / "src/ctscan/README.md",
        "snippets": [
            "## Docker",
            "From repo root:",
            "docker build --pull -t ctscan -f src/ctscan/Dockerfile .",
            "docker run --rm --name ctscan -p 8080:8080 -e PORT=8080 ctscan",
            "curl --fail --silent --show-error http://127.0.0.1:8080/health",
        ],
    },
    "disasters": {
        "readme": ROOT / "src/disasters/README.md",
        "snippets": [
            "## Docker",
            "From repo root:",
            "docker build --pull -t disasters -f src/disasters/Dockerfile .",
            "docker run --rm --name disasters -p 8080:8080 -e PORT=8080 disasters",
            "curl --fail --silent --show-error http://127.0.0.1:8080/health",
        ],
    },
    "test": {
        "readme": ROOT / "src/test/README.md",
        "snippets": [
            "## Docker",
            "From repo root:",
            "docker build --pull -t test -f src/test/Dockerfile .",
            "# macOS / Windows Docker Desktop",
            "docker run --rm --name test \\",
            "-e REDIS_HOST=host.docker.internal \\",
            "-e REDIS_KEY=smoke \\",
            "-e REDIS_VALUE=healthy \\",
            "# Linux",
            "docker run --rm --name test --network host \\",
            "-e REDIS_HOST=127.0.0.1 \\",
            "redis-cli -h 127.0.0.1 -p 6379 GET smoke",
        ],
    },
    "voiceforge": {
        "readme": ROOT / "src/voiceforge/README.md",
        "snippets": [
            "## Docker",
            "From repo root:",
            "docker build --pull -t voiceforge -f src/voiceforge/Dockerfile .",
            "docker run --rm --name voiceforge -p 8080:8080 -e PORT=8080 voiceforge",
            "curl --fail --silent --show-error http://127.0.0.1:8080/health",
        ],
    },
}


def main() -> int:
    failures: list[tuple[str, str, list[str]]] = []

    for service, spec in SERVICE_SNIPPETS.items():
        readme_path = spec["readme"]
        snippets = spec["snippets"]
        text = Path(readme_path).read_text(encoding="utf-8")
        missing = [snippet for snippet in snippets if snippet not in text]
        if missing:
            failures.append(
                (
                    service,
                    Path(readme_path).relative_to(ROOT).as_posix(),
                    missing,
                )
            )

    if failures:
        print("Docker smoke documentation check failed:")
        for service, readme_path, missing in failures:
            print(f"- {service} ({readme_path})")
            for snippet in missing:
                print(f"  missing: {snippet}")
        return 1

    print(
        "Docker smoke documentation check passed for "
        f"{len(SERVICE_SNIPPETS)} Dockerized services."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
