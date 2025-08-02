import argparse
from pathlib import Path
import logging
import signal
import sys
import threading

try:
    from .service import build_client, run_forever
    from .settings import load_settings
except ImportError:
    from service import build_client, run_forever
    from settings import load_settings

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def format_worker_startup(url: str) -> str:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.service_startup import format_service_startup

    return format_service_startup("test-service", url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal Redis write-loop service used for runtime and smoke checks."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load settings, print startup context, and exit without connecting to Redis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    settings = load_settings()
    stop_event = threading.Event()
    startup_url = f"redis://{settings.redis_host}:{settings.redis_port}"

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s, stopping service", signum)
        stop_event.set()

    logger.info(
        "%s key=%s sleep=%s connect_timeout=%s socket_timeout=%s backoff_initial=%s backoff_max=%s backoff_multiplier=%s",
        format_worker_startup(startup_url),
        settings.redis_key,
        settings.sleep_seconds,
        settings.redis_socket_connect_timeout,
        settings.redis_socket_timeout,
        settings.redis_backoff_initial_seconds,
        settings.redis_backoff_max_seconds,
        settings.redis_backoff_multiplier,
    )
    if args.dry_run:
        logger.info("Dry run only; skipping Redis connection and write loop")
        return 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cache = build_client(
        host=settings.redis_host,
        port=settings.redis_port,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        socket_timeout=settings.redis_socket_timeout,
    )
    try:
        run_forever(
            cache,
            sleep_seconds=settings.sleep_seconds,
            key=settings.redis_key,
            value=settings.redis_value,
            should_stop=stop_event.is_set,
            backoff_initial_seconds=settings.redis_backoff_initial_seconds,
            backoff_max_seconds=settings.redis_backoff_max_seconds,
            backoff_multiplier=settings.redis_backoff_multiplier,
        )
    finally:
        close_method = getattr(cache, "close", None)
        if callable(close_method):
            close_method()
        logger.info("Service stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
