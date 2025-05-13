#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys


def classify_job(name: str) -> str:
    if name.startswith("tests-"):
        return "suite"
    if name.startswith("smoke-"):
        return "smoke"
    return "other"


def format_bullets(jobs: list[dict[str, str]]) -> list[str]:
    if not jobs:
        return ["- none"]
    return [f"- `{job['name']}`: `{job['conclusion']}`" for job in jobs]


def render_summary(payload: dict) -> str:
    jobs = [
        {
            "name": job["name"],
            "status": job.get("status", "unknown"),
            "conclusion": job.get("conclusion") or job.get("status", "unknown"),
        }
        for job in payload.get("jobs", [])
        if job.get("name") != "workflow-summary"
    ]

    suites = [job for job in jobs if classify_job(job["name"]) == "suite"]
    smokes = [job for job in jobs if classify_job(job["name"]) == "smoke"]
    skipped = [job for job in jobs if job["conclusion"] == "skipped"]
    failures = [
        job
        for job in jobs
        if job["conclusion"] in {"failure", "timed_out", "cancelled", "action_required"}
    ]
    passed_checks = [
        job
        for job in jobs
        if job["conclusion"] == "success" and classify_job(job["name"]) == "other"
    ]

    lines = [
        "# CI Workflow Summary",
        "",
        "## Passed Suites",
        *format_bullets([job for job in suites if job["conclusion"] == "success"]),
        "",
        "## Smoke Results",
        *format_bullets(smokes),
        "",
        "## Skipped Jobs",
        *format_bullets(skipped),
        "",
        "## Other Passed Checks",
        *format_bullets(passed_checks),
        "",
        "## Failing Jobs",
        *format_bullets(failures),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: render_ci_summary.py <jobs.json> <output.md>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.write_text(render_summary(payload), encoding="utf-8")
    print(f"Wrote CI summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
