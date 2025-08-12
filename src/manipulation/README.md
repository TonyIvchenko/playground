# Manipulation Detector

Browser-based manipulation-pattern analyzer for newsletters, community posts, and group chats.

## What It Is

- paste a message, newsletter, or discussion excerpt
- score urgency, guilt, exclusivity, authority pressure, fear, and us-vs-them framing
- use browser-side toxicity and sentiment models as supporting signals
- show matched cues and a pattern breakdown instead of a black-box verdict

## Local Run

From `src/manipulation`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## Key Caveats

- this is a pattern detector, not a lie detector or intent reader
- results are signals for review, not proof of manipulation
