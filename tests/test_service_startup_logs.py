from __future__ import annotations

from scripts.service_startup import (
    format_http_service_startup,
    format_service_startup,
)


def test_format_http_service_startup_normalizes_wildcard_host() -> None:
    assert (
        format_http_service_startup("bert", "0.0.0.0", 8080)
        == "Starting bert on http://127.0.0.1:8080"
    )


def test_format_http_service_startup_preserves_specific_host() -> None:
    assert (
        format_http_service_startup("CT Scan", "127.0.0.1", 8090)
        == "Starting CT Scan on http://127.0.0.1:8090"
    )


def test_format_service_startup_supports_non_http_services() -> None:
    assert (
        format_service_startup("test-service", "redis://localhost:6379")
        == "Starting test-service on redis://localhost:6379"
    )
