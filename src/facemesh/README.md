# Facemesh

Minimal MediaPipe Face Mesh browser demo.

## What It Is

- browser-side live face-mesh demo
- captures webcam frames locally and overlays the mesh in real time
- no server-side inference path

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080` and allow camera access. The page captures fresh video frames in the browser and updates the mesh continuously on the live camera stream.

## Key Caveats

- camera access is required
- behavior depends on browser camera and WebGL support
