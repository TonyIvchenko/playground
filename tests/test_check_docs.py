from __future__ import annotations

from scripts.check_docs import run_checks


def test_docs_checks_pass_current_repo() -> None:
    results = run_checks()
    assert results["readme-markdown"]["errors"] == []
    assert results["docs-spelling"]["errors"] == []
    assert results["readme-code-paths"]["errors"] == []
