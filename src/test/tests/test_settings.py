from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "settings.py"
SPEC = spec_from_file_location("test_settings_runtime", MODULE_PATH)
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
    assert loaded.redis_socket_connect_timeout == 5.0
    assert loaded.redis_socket_timeout == 5.0
    assert loaded.redis_backoff_initial_seconds == 1.0
    assert loaded.redis_backoff_max_seconds == 60.0
    assert loaded.redis_backoff_multiplier == 2.0


def test_load_settings_reads_values_from_env_mapping():
    loaded = settings.load_settings(
        {
            "REDIS_HOST": "cache.local",
            "REDIS_PORT": "6380",
            "REDIS_KEY": "alpha",
            "REDIS_VALUE": "beta",
            "SLEEP_SECONDS": "2.5",
            "REDIS_SOCKET_CONNECT_TIMEOUT": "1.5",
            "REDIS_SOCKET_TIMEOUT": "4.5",
            "REDIS_BACKOFF_INITIAL_SECONDS": "0.5",
            "REDIS_BACKOFF_MAX_SECONDS": "8.0",
            "REDIS_BACKOFF_MULTIPLIER": "1.75",
        }
    )
    assert loaded.redis_host == "cache.local"
    assert loaded.redis_port == 6380
    assert loaded.redis_key == "alpha"
    assert loaded.redis_value == "beta"
    assert loaded.sleep_seconds == 2.5
    assert loaded.redis_socket_connect_timeout == 1.5
    assert loaded.redis_socket_timeout == 4.5
    assert loaded.redis_backoff_initial_seconds == 0.5
    assert loaded.redis_backoff_max_seconds == 8.0
    assert loaded.redis_backoff_multiplier == 1.75


def test_load_settings_raises_for_invalid_numeric_values():
    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_PORT": "not-a-number"})

    with pytest.raises(ValueError):
        settings.load_settings({"SLEEP_SECONDS": "nope"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_SOCKET_CONNECT_TIMEOUT": "bad"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_SOCKET_TIMEOUT": "bad"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_INITIAL_SECONDS": "bad"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_MAX_SECONDS": "bad"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_MULTIPLIER": "bad"})


def test_load_settings_rejects_invalid_numeric_ranges():
    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_PORT": "0"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_PORT": "-1"})

    with pytest.raises(ValueError):
        settings.load_settings({"SLEEP_SECONDS": "-0.1"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_SOCKET_CONNECT_TIMEOUT": "0"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_SOCKET_TIMEOUT": "-1"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_INITIAL_SECONDS": "0"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_MAX_SECONDS": "-1"})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_MULTIPLIER": "0"})


def test_load_settings_rejects_empty_string_values():
    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_HOST": ""})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_HOST": "   "})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_KEY": ""})

    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_VALUE": ""})


def test_load_settings_rejects_invalid_backoff_policy():
    with pytest.raises(ValueError):
        settings.load_settings({"REDIS_BACKOFF_MULTIPLIER": "0.5"})

    with pytest.raises(ValueError):
        settings.load_settings(
            {
                "REDIS_BACKOFF_INITIAL_SECONDS": "5",
                "REDIS_BACKOFF_MAX_SECONDS": "4",
            }
        )
