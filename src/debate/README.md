# Debate

Browser-based courtroom simulator.

## Local Run

From `src/debate`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080`.

## What It Does

- Takes two pasted documents or two uploaded text files.
- Extracts the strongest claims from each side.
- Uses a browser text-generation model to stage a prosecutor vs defense exchange.
- Plays the transcript turn by turn, then ends with a judge-style synthesis.

## Notes

- First run downloads the browser model.
- Shorter inputs work better than dumping full reports.
- This is a rhetorical simulator, not legal advice.
