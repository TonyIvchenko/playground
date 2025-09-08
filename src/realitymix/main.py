"""Serve the RealityMix browser app through the shared launcher."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "browser_app_main.py"),
    run_name="__main__",
    init_globals={"APP_FILE": __file__},
)
