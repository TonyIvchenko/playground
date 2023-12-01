import logging
import signal
import threading

from .service import build_client, run_forever
from .settings import load_settings

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = load_settings()
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("Received signal %s, stopping service", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Starting test-service host=%s port=%s key=%s sleep=%s",
        settings.redis_host,
        settings.redis_port,
        settings.redis_key,
        settings.sleep_seconds,
    )
    cache = build_client(host=settings.redis_host, port=settings.redis_port)
    try:
        run_forever(
            cache,
            sleep_seconds=settings.sleep_seconds,
            key=settings.redis_key,
            value=settings.redis_value,
            should_stop=stop_event.is_set,
        )
    finally:
        close_method = getattr(cache, "close", None)
        if callable(close_method):
            close_method()
        logger.info("Service stopped")
