# BERT

Minimal browser-side toxicity analyzer.

## What It Is

- browser-side toxicity scoring with `Xenova/toxic-bert`
- built-in toxic and non-toxic sample buttons for quick checks
- no server-side inference path

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Key Caveats

- first run downloads model assets, so internet access is required
- the page loads the model directly in the browser
