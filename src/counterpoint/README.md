# Counterpoint

Browser-based argument ghostwriter.

## Local Run

From `src/counterpoint`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## What It Does

- Takes one pasted point of view.
- Builds the strongest opposing case in the browser.
- Generates likely attack lines, trap questions, weak spots, and phrases to avoid.
- Produces prep notes so you can stress-test your side before a real conversation.

## Notes

- First run downloads a small browser text model.
- If the model cannot load, the app falls back to heuristic counterargument templates.
- Short, clear positions work better than dumping long reports.
