# Playground

Minimal multi-service playground.

For the current generated service reference, run:

```bash
python scripts/list_services.py
```

For machine-readable output, run:

```bash
python scripts/list_services.py --json
```

For repo-wide conventions and service-shape reasoning, see
[CONTRIBUTING.md](/Users/toxa/git/playground/CONTRIBUTING.md) and
[ARCHITECTURE.md](/Users/toxa/git/playground/ARCHITECTURE.md).
For rolling repo notes, see
[RECENT_CHANGES.md](/Users/toxa/git/playground/RECENT_CHANGES.md).

## Services

The repo currently has three broad service shapes:

- Browser apps:
  - `bert`
  - `counterpoint`
  - `debate`
  - `facemesh`
  - `manipulation`
  - `memorypalace`
  - `realitycheck`
  - `realitymix`
  - `vibedj`
- Python web services:
  - `ctscan`
  - `disasters`
  - `voiceforge`
- Worker service:
  - `test`

Use `python scripts/list_services.py` for the current run command, tests,
Docker path, health support, and per-service README path.

Service-specific env vars, data requirements, Docker commands, and caveats live
in `src/<service>/README.md`.

## Quick Start

```bash
make update
make run <service> [port]
make smoke <service> [port]
make test <service>
make lint
make format
```

Notes:

- Most services use `make run <service> [port]`.
- `test` uses `make run test`.
- `test` is a Redis-backed worker, so its narrow local smoke path is a one-shot
  dry run instead of an HTTP probe.
- `ctscan`, `disasters`, and `voiceforge` are the heavy data or ML services;
  use their own READMEs for setup and runtime details.

## Repo Layout

- `src/`: services
- `shared/`: browser-app shared assets
- `scripts/`: repo tooling
- `tests/`: root smoke and shared behavior tests

## More Detail

If you need more than the basics in this file, the next place to look should
usually be:

- `python scripts/list_services.py`
- `src/<service>/README.md`
- [CONTRIBUTING.md](/Users/toxa/git/playground/CONTRIBUTING.md)
- [ARCHITECTURE.md](/Users/toxa/git/playground/ARCHITECTURE.md)
- [RECENT_CHANGES.md](/Users/toxa/git/playground/RECENT_CHANGES.md)
