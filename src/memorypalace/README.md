# MemoryPalace

Browser-based 3D memory palace builder.

## Local Run

From `src/memorypalace`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## What It Does

- Takes pasted notes.
- Turns each note into a distinct room in a connected 3D palace.
- Lets you walk the palace in the browser.
- Highlights the current room and its mnemonic cues.
- Generates quick room titles, anchors, and keywords automatically.

## Controls

- `W A S D` to move
- Arrow keys or drag on the scene to look around
- Click a room chip to jump straight there

## Notes

- This is a browser-only app built with `three.js`.
- It is for memorization and recall practice, not long-form document storage.
