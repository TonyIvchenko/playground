from __future__ import annotations

from scripts import check_services


def test_collect_static_failures_passes_for_repo() -> None:
    assert check_services.collect_static_failures() == []


def test_browser_app_contracts_pass_current_repo() -> None:
    failures_by_check, check_count = check_services.collect_browser_contract_failures()

    assert failures_by_check == {}
    assert check_count >= 10
