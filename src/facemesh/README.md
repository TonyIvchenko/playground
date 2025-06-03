# Facemesh

Minimal MediaPipe Face Mesh browser demo.

## Local Run

From `src/facemesh`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080` and allow camera access. The page captures fresh video frames in the browser and updates the mesh continuously on the live camera stream.
