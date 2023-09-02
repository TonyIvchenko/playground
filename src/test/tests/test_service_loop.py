from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


MODULE_PATH = Path(__file__).resolve().parents[1] / "service.py"
SPEC = spec_from_file_location("test_runtime", MODULE_PATH)
service = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["redis"] = types.SimpleNamespace(Redis=object)
SPEC.loader.exec_module(service)


class FakeCache:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = []

    def set(self, key, value):
        self.calls.append((key, value))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("forced error")


def test_run_once_writes_expected_value():
    cache = FakeCache()
    service.run_once(cache, key="hello", value="world")
    assert cache.calls == [("hello", "world")]


def test_run_forever_success_path_uses_regular_sleep(monkeypatch):
    cache = FakeCache()
    sleep_calls = []
    should_stop = {"value": False}

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 2:
            should_stop["value"] = True

    monkeypatch.setattr(service.time, "sleep", fake_sleep)
    service.run_forever(
        cache,
        sleep_seconds=7,
        key="a",
        value="b",
        should_stop=lambda: should_stop["value"],
    )

    assert cache.calls == [("a", "b"), ("a", "b")]
    assert sleep_calls == [7, 7]


def test_run_forever_failure_path_uses_backoff(monkeypatch):
    cache = FakeCache(fail_times=2)
    sleep_calls = []
    should_stop = {"value": False}

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            should_stop["value"] = True

    monkeypatch.setattr(service.time, "sleep", fake_sleep)
    service.run_forever(
        cache,
        sleep_seconds=7,
        key="x",
        value="y",
        should_stop=lambda: should_stop["value"],
        backoff_initial_seconds=1,
        backoff_max_seconds=3,
        backoff_multiplier=2,
    )

    assert cache.calls == [("x", "y"), ("x", "y"), ("x", "y")]
    assert sleep_calls == [1, 2, 7]
