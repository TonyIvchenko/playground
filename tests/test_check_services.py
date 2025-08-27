from __future__ import annotations

from scripts import check_services


def test_collect_static_failures_passes_for_repo() -> None:
    assert check_services.collect_static_failures() == []
