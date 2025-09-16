from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import os
import socket
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVICE_MANIFEST = json.loads(
    (ROOT / "scripts" / "service_manifest.json").read_text(encoding="utf-8")
)
STATIC_SERVICES = tuple(
    sorted(
        name
        for name, metadata in SERVICE_MANIFEST.items()
        if metadata["type"] == "static browser app"
    )
)
STATIC_SMOKE_CASES = (
    pytest.param(
        "index html",
        (
            "--path",
            "/",
            "--expect-content-type",
            "text/html",
            "--expect-body-fragment",
            "<html",
        ),
        "/",
        id="index-html",
    ),
    pytest.param(
        "health",
        (
            "--path",
            "/health",
            "--expect-content-type",
            "application/json",
        ),
        "/health",
        id="health",
    ),
    pytest.param(
        "shared browser tokens",
        (
            "--path",
            "/shared/browser-tokens.css",
            "--expect-content-type",
            "text/css",
            "--expect-body-fragment",
            "color-accent",
        ),
        "/shared/browser-tokens.css",
        id="browser-tokens",
    ),
    pytest.param(
        "shared browser starter",
        (
            "--path",
            "/shared/browser-starter.css",
            "--expect-content-type",
            "text/css",
            "--expect-body-fragment",
            "app-starter-shell",
        ),
        "/shared/browser-starter.css",
        id="browser-starter",
    ),
    pytest.param(
        "shared browser status",
        (
            "--path",
            "/shared/browser-status.js",
            "--expect-content-type",
            "text/javascript",
            "--expect-body-fragment",
            "setStatusState",
        ),
        "/shared/browser-status.js",
        id="browser-status",
    ),
    pytest.param(
        "shared browser actions",
        (
            "--path",
            "/shared/browser-actions.js",
            "--expect-content-type",
            "text/javascript",
            "--expect-body-fragment",
            "downloadTextFile",
        ),
        "/shared/browser-actions.js",
        id="browser-actions",
    ),
    pytest.param(
        "shared browser guidance",
        (
            "--path",
            "/shared/browser-actions.js",
            "--expect-content-type",
            "text/javascript",
            "--expect-body-fragment",
            "createSoftLimitGuidanceUpdater",
        ),
        "/shared/browser-actions.js",
        id="browser-guidance",
    ),
)
HOST_ENV_SERVICES = ("bert", "realitycheck")
STATUS_HELPER_SERVICES = (
    "bert",
    "counterpoint",
    "debate",
    "facemesh",
    "manipulation",
    "realitycheck",
    "realitymix",
    "vibedj",
)
ACTION_HELPER_SERVICES = (
    "bert",
    "counterpoint",
    "debate",
    "manipulation",
    "memorypalace",
    "realitycheck",
    "realitymix",
    "vibedj",
)
STATIC_PAGE_CASES = (
    pytest.param(
        "bert",
        (
            "Load toxic sample",
            "Load non-toxic sample",
            "Try an example:",
        ),
        id="bert-samples",
    ),
    pytest.param(
        "bert",
        (
            'id="truncation-warning"',
            "Tokenizer truncation warning:",
        ),
        id="bert-truncation-warning",
    ),
    pytest.param(
        "bert",
        (
            'id="score-note"',
            "Interpretation: the percentage is the model's confidence",
        ),
        id="bert-score-note",
    ),
    pytest.param(
        "counterpoint",
        (
            'id="word-count"',
            'id="soft-limit-note"',
            "Soft limit: around 140 words keeps the counter-side sharper.",
        ),
        id="counterpoint-word-count-guidance",
    ),
    pytest.param(
        "counterpoint",
        (
            'id="copy-button"',
            "Copy Notes",
        ),
        id="counterpoint-copy-notes",
    ),
    pytest.param(
        "counterpoint",
        (
            'id="output-mode-select"',
            "Strongest steelman only",
            "Full prep breakdown",
        ),
        id="counterpoint-output-mode",
    ),
    pytest.param(
        "debate",
        (
            'id="char-count-a"',
            'id="char-count-b"',
            "Soft limit: around 900 chars keeps the sparring turns sharper.",
        ),
        id="debate-character-counters",
    ),
    pytest.param(
        "debate",
        (
            'id="export-button"',
            "Export Markdown",
            "# Debate Sparring Notes",
        ),
        id="debate-markdown-export",
    ),
    pytest.param(
        "debate",
        (
            'id="fallback-pill"',
            "Fallback mode active",
            "Debate is using fallback mode for this run.",
        ),
        id="debate-fallback-banner",
    ),
    pytest.param(
        "manipulation",
        (
            "signal-tooltip",
            "Looks for countdowns, last-chance deadlines",
            "Catches shame, obligation, and loyalty-test language",
        ),
        id="manipulation-signal-tooltips",
    ),
    pytest.param(
        "manipulation",
        (
            'id="view-mode-select"',
            'id="simple-summary-card"',
            "Detailed signals",
            "Simple summary",
        ),
        id="manipulation-summary-toggle",
    ),
    pytest.param(
        "memorypalace",
        (
            'id="save-palace-button"',
            'id="load-palace-button"',
            'id="saved-palace-status"',
            "Save Palace",
            "Load Saved",
            "playground.memorypalace.saved-palace.v1",
        ),
        id="memorypalace-local-save-load",
    ),
    pytest.param(
        "realitycheck",
        (
            "The source took too long to respond. Retry, use a faster page, or paste the text directly.",
            "That URL did not return a readable page.",
            "That ${label} URL looks blocked by the source site.",
            "Use a direct ${label} URL or upload the ${label} instead.",
        ),
        id="realitycheck-url-fetch-errors",
    ),
    pytest.param(
        "realitycheck",
        (
            'id="url-source-metadata"',
            'id="source-title-value"',
            'id="source-final-url-value"',
            'id="source-content-type-value"',
            "Fetched Source",
            "Final URL",
        ),
        id="realitycheck-source-metadata",
    ),
    pytest.param(
        "realitycheck",
        (
            'id="image-size-guidance"',
            'id="video-size-guidance"',
            "Upload tip: images under 12 MB decode faster in the browser.",
            "Upload tip: videos under 80 MB work best here. Larger clips take longer to load and sample.",
        ),
        id="realitycheck-upload-size-guidance",
    ),
    pytest.param(
        "realitymix",
        (
            'data-style-sample="sunset-weave"',
            'data-style-sample="blueprint-bloom"',
            'data-style-sample="poster-pulse"',
            "Sunset Weave",
            "Blueprint Bloom",
            "Poster Pulse",
            "Loading sample style image…",
        ),
        id="realitymix-style-samples",
    ),
    pytest.param(
        "realitymix",
        (
            'id="output-fps-text"',
            'id="inference-text"',
            "Output FPS:",
            "Inference:",
            "measuredOutputFps",
            "averageStylizeMs",
        ),
        id="realitymix-performance-readout",
    ),
    pytest.param(
        "realitymix",
        (
            'id="camera-troubleshooting-card"',
            'id="camera-troubleshooting-summary"',
            'id="camera-troubleshooting-list"',
            "Camera Troubleshooting",
            "Camera permission was denied or blocked for this page.",
            "Camera access needs a secure context before the browser will prompt for webcam permission.",
        ),
        id="realitymix-camera-troubleshooting",
    ),
    pytest.param(
        "realitymix",
        (
            'id="mirror-toggle-button"',
            'aria-pressed="true"',
            "Mirror: On",
            "Mirror: Off",
            "mirrorEnabled",
        ),
        id="realitymix-mirror-toggle",
    ),
    pytest.param(
        "vibedj",
        (
            'id="global-stop-button"',
            "Stop All Audio",
            "syncGlobalStopButton",
            'setStatus("All playback stopped.")',
        ),
        id="vibedj-global-stop",
    ),
    pytest.param(
        "vibedj",
        (
            'id="export-md-button"',
            'id="export-json-button"',
            "Export Markdown",
            "Export JSON",
            "buildRecommendationMarkdown",
            "buildRecommendationJson",
        ),
        id="vibedj-exports",
    ),
    pytest.param(
        "vibedj",
        (
            'id="audio-unlock-hint"',
            "Audio starts only after your first click.",
            "syncAudioUnlockHint",
        ),
        id="vibedj-audio-unlock-hint",
    ),
)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_local_smoke(
    service: str,
    *,
    port: int,
    extra_args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/smoke_service.py",
            service,
            "--port",
            str(port),
            "--timeout",
            "20",
            "--interval",
            "0.5",
            *extra_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def assert_smoke_passed(
    result: subprocess.CompletedProcess[str],
    *,
    service: str,
    label: str,
    expected_url: str,
) -> None:
    assert result.returncode == 0, (
        f"{label} failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert expected_url in result.stdout


@lru_cache(maxsize=None)
def service_page_text(service: str) -> str:
    return (ROOT / "src" / service / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("service", STATIC_SERVICES)
@pytest.mark.parametrize(("label", "extra_args", "expected_path"), STATIC_SMOKE_CASES)
def test_static_app_smoke_contracts(
    service: str,
    label: str,
    extra_args: tuple[str, ...],
    expected_path: str,
) -> None:
    port = pick_free_port()
    result = run_local_smoke(service, port=port, extra_args=extra_args)

    assert_smoke_passed(
        result,
        service=service,
        label=label,
        expected_url=f"http://127.0.0.1:{port}{expected_path}",
    )


@pytest.mark.parametrize("service", HOST_ENV_SERVICES)
def test_static_app_supports_host_env_var(service: str) -> None:
    port = pick_free_port()
    env = os.environ.copy()
    env["HOST"] = "127.0.0.1"
    result = run_local_smoke(service, port=port, env=env)

    assert_smoke_passed(
        result,
        service=service,
        label="HOST smoke",
        expected_url=f"http://127.0.0.1:{port}/health",
    )


@pytest.mark.parametrize(("service", "fragments"), STATIC_PAGE_CASES)
def test_static_page_contracts(service: str, fragments: tuple[str, ...]) -> None:
    text = service_page_text(service)
    for fragment in fragments:
        assert fragment in text


@pytest.mark.parametrize("service", STATUS_HELPER_SERVICES)
def test_model_driven_pages_load_shared_status_helper(service: str) -> None:
    assert "/shared/browser-status.js" in service_page_text(service)


@pytest.mark.parametrize("service", ACTION_HELPER_SERVICES)
def test_pages_load_shared_browser_actions_helper(service: str) -> None:
    assert "/shared/browser-actions.js" in service_page_text(service)
