# Debate

Browser-based argument sparring simulator.

## Local Run

From `src/debate`:

```bash
python main.py
```

Optional: set `HOST` and `PORT` before running, for example `HOST=127.0.0.1 PORT=8090 python main.py`.

Then open `http://127.0.0.1:8080`.

## What It Does

- Takes two pasted documents or two uploaded text files.
- Extracts the strongest points from each side.
- Uses a browser text-generation model to stage a back-and-forth argument.
- Produces prep notes for what lands, what the other side will likely hit back with, and what counteroffense to prepare.

## Notes

- First run downloads the browser model.
- Shorter inputs work better than dumping full reports.
- This is a sparring tool, not a truth machine.
