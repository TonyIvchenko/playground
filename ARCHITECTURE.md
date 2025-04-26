# Architecture Notes

This repo is intentionally mixed-shape. It is a playground, not a single product, so the architecture optimizes for fast iteration, low ceremony, and keeping each app in the smallest shape that still fits the problem.

## Why Some Apps Are Static

Many services in `src/` are really browser apps with a thin Python launcher:

- `bert`
- `counterpoint`
- `debate`
- `facemesh`
- `manipulation`
- `memorypalace`
- `realitycheck`
- `realitymix`
- `vibedj`

These stay mostly static because:

- most of the product logic lives in the browser
- the local run story stays simple
- there is less backend code to maintain
- demos remain easy to move, rewrite, or delete

For these apps, `main.py` is usually just a static file server plus a little routing when needed.

## Why Some Apps Are Python Services

Some services need server-side runtime behavior, not just static assets:

- `ctscan`
- `disasters`
- `voiceforge`

These use Python services because they need one or more of:

- model loading and inference
- larger local assets or model checkpoints
- upload handling
- health endpoints for local smoke checks and Docker
- integration with Gradio or FastAPI

The goal is still to keep the service entrypoint small and push reusable logic into service modules.

## Why `test` Is Different

`src/test` is not a browser app and not a user-facing ML app. It exists as runtime tooling for Redis-backed smoke checks and container validation.

That service is intentionally narrow:

- it exercises runtime wiring
- it gives CI and Docker something simple to validate
- it keeps infrastructure checks separate from the heavier ML apps

## Why Training Code Lives Next To Apps

Some services carry training, notebooks, or data-prep code alongside the app:

- `ctscan`
- `disasters`
- `voiceforge`

That is a pragmatic choice:

- the training and inference code share assets and assumptions
- the service README can document the full local workflow in one place
- experimentation stays close to the app it supports

This does make some service folders heavier. The tradeoff is accepted here because the repo favors local iteration over strict package boundaries.

## Practical Rule Of Thumb

When adding or reshaping a service:

1. keep it static if the browser can do the real work
2. use a Python service when you need server-side inference, uploads, or health-checked runtime behavior
3. keep `main.py` thin even when the service is server-backed
4. keep service-specific docs, tests, and dependencies local to that service

The repo should feel lightweight by default. Heavier architecture is justified only when the app actually needs it.
