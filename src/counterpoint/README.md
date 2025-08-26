# Counterpoint

Browser-based argument ghostwriter.

## What It Is

- takes one pasted point of view
- builds the strongest opposing case in the browser
- generates likely attacks, trap questions, weak spots, and phrases to avoid
- produces prep notes for stress-testing your side before a real conversation

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## Key Caveats

- first run downloads a small browser text model
- if the model cannot load, the app falls back to heuristic templates
- short, clear positions work better than dumping long reports
