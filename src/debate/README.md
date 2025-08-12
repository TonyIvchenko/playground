# Debate

Browser-based argument sparring simulator.

## What It Is

- takes two pasted documents or two uploaded text files
- extracts the strongest points from each side
- uses a browser text-generation model to stage a back-and-forth argument
- produces prep notes for likely hits, rebuttals, and counteroffense

## Local Run

From `src/debate`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## Key Caveats

- first run downloads the browser model
- shorter inputs work better than dumping full reports
- this is a sparring tool, not a truth machine
