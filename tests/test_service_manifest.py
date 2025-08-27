from __future__ import annotations

import pytest

from scripts import check_services, smoke_service
from scripts.list_services import collect_service
from scripts.service_manifest import ROOT, load_service_manifest


def test_service_manifest_matches_repo_services() -> None:
    manifest = load_service_manifest()
    repo_services = sorted(
        path.name
        for path in (ROOT / "src").iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name != "__pycache__"
    )

    assert sorted(manifest) == repo_services


def test_list_services_uses_manifest_backed_metadata() -> None:
    record = collect_service(ROOT / "src" / "test")

    assert record["type"] == "worker service"
    assert record["run"] == "make run test"
    assert record["health_endpoint"] == "-"
    assert record["readme_path"] == "src/test/README.md"


def test_check_services_uses_shared_manifest() -> None:
    manifest = load_service_manifest()

    assert sorted(check_services.SERVICE_SPECS) == sorted(manifest)
    assert (
        check_services.SERVICE_SPECS["disasters"]["local_run"]["env"]["GMAPS_API_KEY"]
        == "ci-local-run-key"
    )


def test_smoke_service_rejects_non_http_worker_service() -> None:
    with pytest.raises(SystemExit, match="does not expose an HTTP smoke path"):
        smoke_service.service_dir("test")
