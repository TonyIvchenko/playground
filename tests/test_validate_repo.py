from __future__ import annotations

from scripts.validate_repo import run_checks


def test_repo_checks_pass_current_repo() -> None:
    results = run_checks()

    assert results["docs"]["readme-markdown"]["errors"] == []
    assert results["docs"]["docs-spelling"]["errors"] == []
    assert results["docs"]["readme-code-paths"]["errors"] == []
    assert results["config"]["errors"] == []
    assert results["hygiene"]["errors"] == []
