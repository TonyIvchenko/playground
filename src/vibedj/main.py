"""Serve the VibeDJ browser app through the shared static launcher."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]


def main() -> None:
    """Start the local static server for the VibeDJ app."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.browser_static_server import serve_static_app

    serve_static_app("vibedj", ROOT)


if __name__ == "__main__":
    main()
