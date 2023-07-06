from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ServiceSettings:
    redis_host: str = "redis-service"
    redis_port: int = 6379
    redis_key: str = "key"
    redis_value: str = "value"
    sleep_seconds: float = 60.0


def _read_int(env, key, default):
    raw_value = env.get(key)
    if raw_value in (None, ""):
        return default
    return int(raw_value)


def _read_float(env, key, default):
    raw_value = env.get(key)
    if raw_value in (None, ""):
        return default
    return float(raw_value)


def load_settings(env=None):
    values = os.environ if env is None else env
    return ServiceSettings(
        redis_host=values.get("REDIS_HOST", ServiceSettings.redis_host),
        redis_port=_read_int(values, "REDIS_PORT", ServiceSettings.redis_port),
        redis_key=values.get("REDIS_KEY", ServiceSettings.redis_key),
        redis_value=values.get("REDIS_VALUE", ServiceSettings.redis_value),
        sleep_seconds=_read_float(values, "SLEEP_SECONDS", ServiceSettings.sleep_seconds),
    )
