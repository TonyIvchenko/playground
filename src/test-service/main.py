import signal
import threading

from service import build_client, run_forever
from settings import load_settings

mount_path = "/data"

if __name__ == "__main__":
    print("starting test-service")
    settings = load_settings()
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        print(f"received signal {signum}, stopping service")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

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
        print("service stopped")

        
            
