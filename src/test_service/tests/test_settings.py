from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "settings.py"
SPEC = spec_from_file_location("test_service_settings", MODULE_PATH)
settings = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(settings)


def test_load_settings_uses_defaults_when_env_is_empty():
    loaded = settings.load_settings({})
    assert loaded.redis_host == "redis-service"
    assert loaded.redis_port == 6379
    assert loaded.redis_key == "key"
    assert loaded.redis_value == "value"
    assert loaded.sleep_seconds == 60.0


def test_load_settings_reads_values_from_env_mapping():
    loaded = settings.load_settings(
        {
            "REDIS_HOST": "cache.local",
            "REDIS_PORT": "6380",
            "REDIS_KEY": "alpha",
            "REDIS_VALUE": "beta",
            "SLEEP_SECONDS": "2.5",
        }
    )
    assert loaded.redis_host == "cache.local"
    assert loaded.redis_port == 6380
    assert loaded.redis_key == "alpha"
    assert loaded.redis_value == "beta"
    assert loaded.sleep_seconds == 2.5


def test_load_settings_raises_for_invalid_numeric_values():
    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_PORT": "not-a-number"})

    with pytest.raises(ValueError):
        settings.load_settings({"SLEEP_SECONDS": "nope"})
