# RealityCheck

Browser-based AI-likelihood analyzer.

## What It Is

- `Text`: browser-side text AI-likelihood scoring with an ONNX classifier
- `Image`: browser-side image AI-likelihood scoring with an ONNX classifier
- `Video`: sampled-frame scoring that reuses the browser-side image classifier
- `URL`: fetch visible page text through the local server and run text analysis

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Inputs

- `Text`: paste raw text
- `Image`: upload a local image or load a direct image URL
- `Video`: upload a local video or load a direct video URL
- `URL`: fetch visible page text through the local proxy and run text analysis

## Key Caveats

- this is an AI-likelihood tool, not a source-of-truth detector
- results are signals, not proof
- if the browser cannot load the image model, image and video runs fall back to heuristics for that session
