# Test Service

Minimal Redis write-loop service used for runtime/container smoke checks.

## Local Run

From repo root:

```bash
python3 src/test/main.py
```

Environment variables:

- `REDIS_HOST` (default `redis-service`)
- `REDIS_PORT` (default `6379`)
- `REDIS_KEY` (default `key`)
- `REDIS_VALUE` (default `value`)
- `SLEEP_SECONDS` (default `60`)

## Docker Run

From repo root:

```bash
docker build -t test -f src/test/Dockerfile .
docker run --rm --name test -e REDIS_HOST=host.docker.internal -e REDIS_PORT=6379 test
```

## Tests

```bash
pytest -q src/test/tests
```
