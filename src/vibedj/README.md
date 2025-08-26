# VibeDJ

Browser-based creative direction generator for writing.

## What It Is

- takes pasted writing
- reads tone, energy, tension, warmth, and dreaminess
- generates a browser synth loop tied to the detected vibe
- builds a color palette, typography recommendation, and motion direction

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Key Caveats

- first run may download a small browser sentiment model
- playback uses `Tone.js`, so audio starts only after a user click
- if the model fails, the app falls back to heuristic analysis
- meant for creative direction, not objective literary analysis
