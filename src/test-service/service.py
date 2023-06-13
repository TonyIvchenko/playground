import time

import redis


DEFAULT_REDIS_HOST = "redis-service"
DEFAULT_REDIS_PORT = 6379
DEFAULT_KEY = "key"
DEFAULT_VALUE = "value"
DEFAULT_SLEEP_SECONDS = 60


def build_client(host=DEFAULT_REDIS_HOST, port=DEFAULT_REDIS_PORT):
    return redis.Redis(host=host, port=port)


def run_once(cache, key=DEFAULT_KEY, value=DEFAULT_VALUE):
    cache.set(key, value)


def run_forever(
    cache,
    sleep_seconds=DEFAULT_SLEEP_SECONDS,
    key=DEFAULT_KEY,
    value=DEFAULT_VALUE,
):
    while True:
        try:
            run_once(cache, key=key, value=value)
        except Exception as exc:
            print(f"Error: {exc}")
        time.sleep(sleep_seconds)
