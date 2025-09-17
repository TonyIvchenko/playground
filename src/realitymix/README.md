# RealityMix

Browser-based live neural style transfer.

## What It Is

- uses the webcam as the live content source
- lets you upload or generate a style reference image
- runs neural style transfer in the browser
- exposes live performance controls for resolution, update rate, and mirroring

## Local Run

```bash
python main.py
```

Then open `http://127.0.0.1:8080` and allow camera access.

## Key Caveats

- first run needs to download the browser model
- lower internal resolution is faster
- higher update rate is smoother but heavier
- camera access is required
- built-in style samples are intentionally inline SVG data URLs in `index.html` to keep this app self-contained with fewer moving files
