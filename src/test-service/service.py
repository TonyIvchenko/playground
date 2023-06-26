import logging
import time

import redis


logger = logging.getLogger(__name__)

DEFAULT_REDIS_HOST = "redis-service"
DEFAULT_REDIS_PORT = 6379
DEFAULT_KEY = "key"
DEFAULT_VALUE = "value"
DEFAULT_SLEEP_SECONDS = 60
DEFAULT_BACKOFF_INITIAL_SECONDS = 1.0
DEFAULT_BACKOFF_MAX_SECONDS = 60.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


def build_client(host=DEFAULT_REDIS_HOST, port=DEFAULT_REDIS_PORT):
    return redis.Redis(host=host, port=port)


def run_once(cache, key=DEFAULT_KEY, value=DEFAULT_VALUE):
    cache.set(key, value)


def run_forever(
    cache,
    sleep_seconds=DEFAULT_SLEEP_SECONDS,
    key=DEFAULT_KEY,
    value=DEFAULT_VALUE,
    should_stop=None,
    backoff_initial_seconds=DEFAULT_BACKOFF_INITIAL_SECONDS,
    backoff_max_seconds=DEFAULT_BACKOFF_MAX_SECONDS,
    backoff_multiplier=DEFAULT_BACKOFF_MULTIPLIER,
):
    stop_check = should_stop or (lambda: False)
    retry_sleep = backoff_initial_seconds

    while not stop_check():
        try:
            run_once(cache, key=key, value=value)
            retry_sleep = backoff_initial_seconds
            time.sleep(sleep_seconds)
        except Exception as exc:
            logger.warning("Redis write failed; retrying in %.1f seconds", retry_sleep)
            time.sleep(retry_sleep)
            retry_sleep = min(backoff_max_seconds, retry_sleep * backoff_multiplier)
