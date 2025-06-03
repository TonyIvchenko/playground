# BERT

Minimal browser-side toxicity analyzer.

## Local Run

From `src/bert`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## Notes

- The page loads the model directly in the browser.
- The first run downloads model assets, so internet access is required.
- It uses `Xenova/toxic-bert`, which ships ONNX weights compatible with `@xenova/transformers`.
