# RealityCheck

Browser-based AI-likelihood analyzer.

## Local Run

From `src/realitycheck`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## What It Does

- `Text`: browser-side text AI-likelihood scoring with an ONNX classifier.
- `Image`: browser-side heuristic scoring from uploaded images.
- `Video`: browser-side heuristic scoring from sampled uploaded video frames.
- `URL`: fetches a page through the local server, extracts visible text, and runs text analysis.

## Caveat

This is an AI-likelihood tool, not a source-of-truth detector. Results should be treated as signals, not proof.
