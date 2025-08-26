# MemoryPalace

Browser-based 3D memory palace builder.

## What It Is

- turn pasted notes into a connected 3D palace
- generate quick room titles, anchors, and keywords automatically
- walk the palace locally in the browser
- save and reload the last built palace from local browser storage

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Controls

- `W A S D` to move
- Arrow keys or drag on the scene to look around
- Click a room chip to jump straight there

## Key Caveats

- browser-only app built with `three.js`
- intended for memorization and recall practice, not long-form document storage
