from __future__ import annotations

from pathlib import Path
import os
import socket
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC_SERVICES = (
    "bert",
    "counterpoint",
    "debate",
    "facemesh",
    "manipulation",
    "memorypalace",
    "realitycheck",
    "realitymix",
    "vibedj",
)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("service", STATIC_SERVICES)
def test_static_app_serves_index_html(service: str) -> None:
    port = pick_free_port()
    result = subprocess.run(
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
            "--path",
            "/",
            "--expect-content-type",
            "text/html",
            "--expect-body-fragment",
            "<html",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Static smoke failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert f"http://127.0.0.1:{port}/" in result.stdout


@pytest.mark.parametrize("service", STATIC_SERVICES)
def test_static_app_serves_health(service: str) -> None:
    port = pick_free_port()
    result = subprocess.run(
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
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Static health smoke failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert f"http://127.0.0.1:{port}/health" in result.stdout


@pytest.mark.parametrize("service", STATIC_SERVICES)
def test_static_app_serves_shared_browser_tokens(service: str) -> None:
    port = pick_free_port()
    result = subprocess.run(
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
            "--path",
            "/shared/browser-tokens.css",
            "--expect-content-type",
            "text/css",
            "--expect-body-fragment",
            "color-accent",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Shared token smoke failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert f"http://127.0.0.1:{port}/shared/browser-tokens.css" in result.stdout


@pytest.mark.parametrize("service", STATIC_SERVICES)
def test_static_app_serves_shared_browser_starter(service: str) -> None:
    port = pick_free_port()
    result = subprocess.run(
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
            "--path",
            "/shared/browser-starter.css",
            "--expect-content-type",
            "text/css",
            "--expect-body-fragment",
            "app-starter-shell",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Shared starter smoke failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert f"http://127.0.0.1:{port}/shared/browser-starter.css" in result.stdout


@pytest.mark.parametrize("service", ("bert", "realitycheck"))
def test_static_app_supports_host_env_var(service: str) -> None:
    port = pick_free_port()
    env = os.environ.copy()
    env["HOST"] = "127.0.0.1"
    result = subprocess.run(
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
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"HOST smoke failed for {service}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert f"Smoke check passed for '{service}'" in result.stdout
    assert f"http://127.0.0.1:{port}/" in result.stdout


def test_bert_page_includes_example_samples() -> None:
    text = (ROOT / "src" / "bert" / "index.html").read_text(encoding="utf-8")

    assert "Load toxic sample" in text
    assert "Load non-toxic sample" in text
    assert "Try an example:" in text


def test_bert_page_includes_truncation_warning() -> None:
    text = (ROOT / "src" / "bert" / "index.html").read_text(encoding="utf-8")

    assert 'id="truncation-warning"' in text
    assert "Tokenizer truncation warning:" in text


def test_bert_page_includes_score_interpretation_note() -> None:
    text = (ROOT / "src" / "bert" / "index.html").read_text(encoding="utf-8")

    assert 'id="score-note"' in text
    assert "Interpretation: the percentage is the model's confidence" in text


def test_counterpoint_page_includes_word_count_guidance() -> None:
    text = (ROOT / "src" / "counterpoint" / "index.html").read_text(encoding="utf-8")

    assert 'id="word-count"' in text
    assert 'id="soft-limit-note"' in text
    assert "Soft limit: around 140 words keeps the counter-side sharper." in text


def test_counterpoint_page_includes_copy_notes_button() -> None:
    text = (ROOT / "src" / "counterpoint" / "index.html").read_text(encoding="utf-8")

    assert 'id="copy-button"' in text
    assert "Copy Notes" in text


def test_counterpoint_page_includes_output_mode_toggle() -> None:
    text = (ROOT / "src" / "counterpoint" / "index.html").read_text(encoding="utf-8")

    assert 'id="output-mode-select"' in text
    assert "Strongest steelman only" in text
    assert "Full prep breakdown" in text


def test_debate_page_includes_per_document_character_counters() -> None:
    text = (ROOT / "src" / "debate" / "index.html").read_text(encoding="utf-8")

    assert 'id="char-count-a"' in text
    assert 'id="char-count-b"' in text
    assert "Soft limit: around 900 chars keeps the sparring turns sharper." in text


def test_debate_page_includes_markdown_export_button() -> None:
    text = (ROOT / "src" / "debate" / "index.html").read_text(encoding="utf-8")

    assert 'id="export-button"' in text
    assert "Export Markdown" in text
    assert "# Debate Sparring Notes" in text


def test_debate_page_includes_heuristic_fallback_banner() -> None:
    text = (ROOT / "src" / "debate" / "index.html").read_text(encoding="utf-8")

    assert 'id="fallback-pill"' in text
    assert "Fallback mode active" in text
    assert "Debate is using fallback mode for this run." in text


def test_manipulation_page_includes_signal_tooltips() -> None:
    text = (ROOT / "src" / "manipulation" / "index.html").read_text(encoding="utf-8")

    assert "signal-tooltip" in text
    assert "Looks for countdowns, last-chance deadlines" in text
    assert "Catches shame, obligation, and loyalty-test language" in text


def test_manipulation_page_includes_simple_summary_view_toggle() -> None:
    text = (ROOT / "src" / "manipulation" / "index.html").read_text(encoding="utf-8")

    assert 'id="view-mode-select"' in text
    assert 'id="simple-summary-card"' in text
    assert "Detailed signals" in text
    assert "Simple summary" in text


def test_memorypalace_page_includes_local_save_load_controls() -> None:
    text = (ROOT / "src" / "memorypalace" / "index.html").read_text(encoding="utf-8")

    assert 'id="save-palace-button"' in text
    assert 'id="load-palace-button"' in text
    assert 'id="saved-palace-status"' in text
    assert "Save Palace" in text
    assert "Load Saved" in text
    assert "playground.memorypalace.saved-palace.v1" in text
