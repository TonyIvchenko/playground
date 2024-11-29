# Test Service

Minimal Redis write-loop service used for runtime/container smoke checks.

## Local Run

From `src/test`:

```bash
REDIS_HOST=localhost REDIS_PORT=6379 python main.py
```

Environment variables:

- `REDIS_HOST` (default `redis-service`; for a local Redis on your machine, set `REDIS_HOST=localhost`)
- `REDIS_PORT` (default `6379`)
- `REDIS_KEY` (default `key`)
- `REDIS_VALUE` (default `value`)
- `SLEEP_SECONDS` (default `60`)
- `REDIS_SOCKET_CONNECT_TIMEOUT` (default `5`)
- `REDIS_SOCKET_TIMEOUT` (default `5`)
- `REDIS_BACKOFF_INITIAL_SECONDS` (default `1`)
- `REDIS_BACKOFF_MAX_SECONDS` (default `60`)
- `REDIS_BACKOFF_MULTIPLIER` (default `2`)

## Docker

From repo root:

```bash
docker build --pull -t test -f src/test/Dockerfile .

# macOS / Windows Docker Desktop
docker run --rm --name test -e REDIS_HOST=host.docker.internal -e REDIS_PORT=6379 -e REDIS_KEY=smoke -e REDIS_VALUE=healthy -e SLEEP_SECONDS=1 test

# Linux
docker run --rm --name test --network host -e REDIS_HOST=127.0.0.1 -e REDIS_PORT=6379 -e REDIS_KEY=smoke -e REDIS_VALUE=healthy -e SLEEP_SECONDS=1 test

redis-cli -h 127.0.0.1 -p 6379 GET smoke
```

Use `host.docker.internal` on Docker Desktop. Use `--network host` only on Linux.
The container healthcheck uses the same `REDIS_SOCKET_CONNECT_TIMEOUT` and `REDIS_SOCKET_TIMEOUT` env vars as the service runtime.

The CI workflow uses the same Redis-backed smoke pattern for this service.

## Tests

```bash
python -m pytest -q src/test/tests
```
