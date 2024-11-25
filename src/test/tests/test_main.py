from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


MODULE_PATH = Path(__file__).resolve().parents[1] / "main.py"


@dataclass(frozen=True)
class FakeSettings:
    redis_host: str = "cache.local"
    redis_port: int = 6380
    redis_key: str = "alpha"
    redis_value: str = "beta"
    sleep_seconds: float = 2.5
    redis_socket_connect_timeout: float = 1.5
    redis_socket_timeout: float = 4.5


class FakeCache:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_main_wires_settings_into_runtime(monkeypatch):
    fake_cache = FakeCache()
    calls = {"signal": [], "run_forever": None, "build_client": None, "basicConfig": None}

    service_module = types.ModuleType("service")
    settings_module = types.ModuleType("settings")

    def fake_build_client(**kwargs):
        calls["build_client"] = kwargs
        return fake_cache

    def fake_run_forever(cache, **kwargs):
        calls["run_forever"] = {"cache": cache, **kwargs}

    def fake_load_settings():
        return FakeSettings()

    service_module.build_client = fake_build_client
    service_module.run_forever = fake_run_forever
    settings_module.load_settings = fake_load_settings
    sys.modules["service"] = service_module
    sys.modules["settings"] = settings_module

    spec = spec_from_file_location("test_main_runtime", MODULE_PATH)
    main_module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(main_module)

    monkeypatch.setattr(main_module.logging, "basicConfig", lambda **kwargs: calls.__setitem__("basicConfig", kwargs))
    monkeypatch.setattr(main_module.signal, "signal", lambda signum, handler: calls["signal"].append(signum))

    main_module.main()

    assert calls["basicConfig"] == {
        "level": main_module.logging.INFO,
        "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
    }
    assert calls["signal"] == [main_module.signal.SIGINT, main_module.signal.SIGTERM]
    assert calls["build_client"] == {
        "host": "cache.local",
        "port": 6380,
        "socket_connect_timeout": 1.5,
        "socket_timeout": 4.5,
    }
    assert calls["run_forever"]["cache"] is fake_cache
    assert calls["run_forever"]["sleep_seconds"] == 2.5
    assert calls["run_forever"]["key"] == "alpha"
    assert calls["run_forever"]["value"] == "beta"
    assert callable(calls["run_forever"]["should_stop"])
    assert fake_cache.closed is True
