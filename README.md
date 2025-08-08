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

That generated view is backed by `scripts/service_manifest.json`.

Use the generated service reference and each service's own README instead of
repeating service details here.

## Common Commands

```bash
make update
make run <service> [port]
make smoke <service> [port]
make test <service>
make lint
make format
```

## More Detail

If you need more than the basics in this file, the next place to look should
usually be:

- `python scripts/list_services.py`
- `src/<service>/README.md`
- [ARCHITECTURE.md](/Users/toxa/git/playground/ARCHITECTURE.md)
- `git log --oneline -n 20`
