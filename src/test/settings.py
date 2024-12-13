from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ServiceSettings:
    redis_host: str = "redis-service"
    redis_port: int = 6379
    redis_key: str = "key"
    redis_value: str = "value"
    sleep_seconds: float = 60.0
    redis_socket_connect_timeout: float = 5.0
    redis_socket_timeout: float = 5.0
    redis_backoff_initial_seconds: float = 1.0
    redis_backoff_max_seconds: float = 60.0
    redis_backoff_multiplier: float = 2.0


def _read_int(env, key, default):
    raw_value = env.get(key)
    if raw_value in (None, ""):
        return default
    return int(raw_value)


def _read_non_empty_string(env, key, default):
    raw_value = env.get(key)
    if raw_value is None:
        return default
    if raw_value.strip() == "":
        raise ValueError(f"{key} must not be empty")
    return raw_value


def _read_float(env, key, default):
    raw_value = env.get(key)
    if raw_value in (None, ""):
        return default
    return float(raw_value)


def _read_positive_int(env, key, default):
    value = _read_int(env, key, default)
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _read_non_negative_float(env, key, default):
    value = _read_float(env, key, default)
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _read_positive_float(env, key, default):
    value = _read_float(env, key, default)
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def load_settings(env=None):
    values = os.environ if env is None else env
    loaded = ServiceSettings(
        redis_host=_read_non_empty_string(values, "REDIS_HOST", ServiceSettings.redis_host),
        redis_port=_read_positive_int(values, "REDIS_PORT", ServiceSettings.redis_port),
        redis_key=_read_non_empty_string(values, "REDIS_KEY", ServiceSettings.redis_key),
        redis_value=_read_non_empty_string(values, "REDIS_VALUE", ServiceSettings.redis_value),
        sleep_seconds=_read_non_negative_float(values, "SLEEP_SECONDS", ServiceSettings.sleep_seconds),
        redis_socket_connect_timeout=_read_positive_float(
            values,
            "REDIS_SOCKET_CONNECT_TIMEOUT",
            ServiceSettings.redis_socket_connect_timeout,
        ),
        redis_socket_timeout=_read_positive_float(
            values,
            "REDIS_SOCKET_TIMEOUT",
            ServiceSettings.redis_socket_timeout,
        ),
        redis_backoff_initial_seconds=_read_positive_float(
            values,
            "REDIS_BACKOFF_INITIAL_SECONDS",
            ServiceSettings.redis_backoff_initial_seconds,
        ),
        redis_backoff_max_seconds=_read_positive_float(
            values,
            "REDIS_BACKOFF_MAX_SECONDS",
            ServiceSettings.redis_backoff_max_seconds,
        ),
        redis_backoff_multiplier=_read_positive_float(
            values,
            "REDIS_BACKOFF_MULTIPLIER",
            ServiceSettings.redis_backoff_multiplier,
        ),
    )
    if loaded.redis_backoff_multiplier < 1.0:
        raise ValueError("REDIS_BACKOFF_MULTIPLIER must be >= 1")
    if loaded.redis_backoff_max_seconds < loaded.redis_backoff_initial_seconds:
        raise ValueError("REDIS_BACKOFF_MAX_SECONDS must be >= REDIS_BACKOFF_INITIAL_SECONDS")
    return loaded
