# RealityMix

Browser-based live neural style transfer.

## Local Run

From `src/realitymix`:

```bash
python main.py
```

Then open `http://127.0.0.1:8080` and allow camera access.

## What It Does

- Uses the webcam as the live content source.
- Lets you upload a style reference image.
- Runs arbitrary neural style transfer in the browser.
- Uses reduced internal resolution and throttled inference to keep live updates usable.

## Notes

- First run needs to download the browser model.
- Lower internal resolution is faster.
- Higher update rate is smoother but heavier.
