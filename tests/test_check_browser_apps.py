from __future__ import annotations

from scripts.check_browser_apps import load_contracts, run_checks


def test_browser_app_contracts_pass_current_repo() -> None:
    failures_by_check, check_count = run_checks(load_contracts())
    assert failures_by_check == {}
    assert check_count >= 10
