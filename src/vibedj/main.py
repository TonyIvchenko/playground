"""Serve the VibeDJ browser app through the shared launcher."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "browser_static_server.py"),
    run_name="__main__",
    init_globals={"APP_FILE": __file__},
)
