# VibeDJ

Browser-based creative direction generator for writing.

## Local Run

From `src/vibedj`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## What It Does

- Takes pasted writing.
- Reads tone, energy, tension, warmth, and dreaminess.
- Generates a browser synth loop tied to the detected vibe.
- Shows reactive visuals while the loop plays.
- Builds a color palette.
- Recommends typography.
- Proposes motion treatment for a page, teaser, or visual identity.

## Notes

- First run may download a small browser sentiment model.
- Playback uses `Tone.js` in the browser, so audio starts only after a user click.
- If the model fails, the app falls back to heuristic analysis.
- It is meant for creative direction, not objective literary analysis.
