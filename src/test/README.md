# Test Service

Minimal Redis write-loop service used for runtime/container smoke checks.

## What It Is

- narrow Redis-backed worker for runtime and container validation
- supports dry-run and config JSON output for one-shot local smoke checks
- not a user-facing app

## Local Run

From `src/test`:

```bash
REDIS_HOST=localhost REDIS_PORT=6379 python main.py --dry-run --config-json
```

That is the quickest local smoke path because it loads and validates settings without needing a live Redis instance, and it prints the resolved runtime config as JSON on stdout.

Run the real write loop against a local Redis:

```bash
REDIS_HOST=localhost REDIS_PORT=6379 python main.py
```

If you want the same machine-readable config echo before the real loop starts:

```bash
REDIS_HOST=localhost REDIS_PORT=6379 python main.py --config-json
```

Key environment variables:

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

`REDIS_HOST`, `REDIS_KEY`, and `REDIS_VALUE` are trimmed at startup, and blank values are rejected.

## Docker

From repo root:

```bash
docker build --pull -t test -f src/test/Dockerfile .

# macOS / Windows Docker Desktop
docker run --rm --name test \
  -e REDIS_HOST=host.docker.internal \
  -e REDIS_PORT=6379 \
  -e REDIS_KEY=smoke \
  -e REDIS_VALUE=healthy \
  -e SLEEP_SECONDS=1 \
  -e REDIS_SOCKET_CONNECT_TIMEOUT=5 \
  -e REDIS_SOCKET_TIMEOUT=5 \
  -e REDIS_BACKOFF_INITIAL_SECONDS=1 \
  -e REDIS_BACKOFF_MAX_SECONDS=30 \
  -e REDIS_BACKOFF_MULTIPLIER=2 \
  test

# Linux
docker run --rm --name test --network host \
  -e REDIS_HOST=127.0.0.1 \
  -e REDIS_PORT=6379 \
  -e REDIS_KEY=smoke \
  -e REDIS_VALUE=healthy \
  -e SLEEP_SECONDS=1 \
  -e REDIS_SOCKET_CONNECT_TIMEOUT=5 \
  -e REDIS_SOCKET_TIMEOUT=5 \
  -e REDIS_BACKOFF_INITIAL_SECONDS=1 \
  -e REDIS_BACKOFF_MAX_SECONDS=30 \
  -e REDIS_BACKOFF_MULTIPLIER=2 \
  test

redis-cli -h 127.0.0.1 -p 6379 GET smoke
```

Use `host.docker.internal` on Docker Desktop. Use `--network host` only on Linux.
The container healthcheck uses the same `REDIS_HOST`, `REDIS_PORT`, `REDIS_SOCKET_CONNECT_TIMEOUT`, and `REDIS_SOCKET_TIMEOUT` env vars as the service runtime.

## Tests

```bash
python -m pytest -q src/test/tests
```

## Key Caveats

- dry-run is the quickest local smoke path because it does not need a live Redis instance
- the real write loop needs reachable Redis settings
