# Manipulation Detector

Browser-based manipulation-pattern analyzer for newsletters, community posts, and group chats.

## Local Run

From `src/manipulation`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## What It Does

- Lets you paste a message, newsletter, or discussion excerpt.
- Scores manipulation-related patterns such as urgency, guilt, exclusivity, authority pressure, fear, and us-vs-them framing.
- Uses browser-side text classifiers for toxicity and sentiment as supporting signals.
- Shows matched cues and a pattern breakdown instead of a black-box verdict.

## Caveat

This is a pattern detector, not a lie detector or intent reader. Results are signals for review, not proof of manipulation.
