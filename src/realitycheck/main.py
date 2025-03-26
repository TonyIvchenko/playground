from __future__ import annotations

from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import json
import os
import re


ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8080"))
USER_AGENT = "RealityCheck/1.0 (+http://127.0.0.1)"


def infer_media_content_type(url: str, declared_type: str) -> str:
    if declared_type and declared_type != "application/octet-stream":
        return declared_type

    guessed_type, _ = guess_type(url)
    return (guessed_type or declared_type or "").lower()


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self._title_parts.append(cleaned)
        else:
            self._text_parts.append(cleaned)

    def result(self) -> dict[str, str]:
        title = " ".join(self._title_parts).strip()
        text = " ".join(self._text_parts).strip()
        text = re.sub(r"\s+", " ", text)
        return {
            "title": title,
            "text": text[:20000],
        }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/fetch":
            self.handle_fetch(parsed.query)
            return
        if parsed.path == "/api/media":
            self.handle_media(parsed.query)
            return
        super().do_GET()

    def handle_fetch(self, query: str) -> None:
        params = parse_qs(query)
        target = (params.get("url") or [""])[0].strip()
        if not target:
            self.send_json({"error": "Missing url parameter."}, status=400)
            return

        parsed = urlparse(target)
        if parsed.scheme not in {"http", "https"}:
            self.send_json({"error": "Only http and https URLs are supported."}, status=400)
            return

        try:
            request = Request(target, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=12) as response:
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                body = response.read()
        except Exception as exc:
            self.send_json({"error": f"Failed to fetch URL: {exc}"}, status=502)
            return

        if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
            self.send_json(
                {
                    "error": f"Unsupported content type: {content_type or 'unknown'}",
                    "content_type": content_type,
                },
                status=415,
            )
            return

        try:
            html = body.decode("utf-8", errors="replace")
        except Exception as exc:
            self.send_json({"error": f"Failed to decode response: {exc}"}, status=500)
            return

        parser = VisibleTextParser()
        parser.feed(html)
        parsed_content = parser.result()

        self.send_json(
            {
                "url": target,
                "content_type": content_type,
                "title": parsed_content["title"],
                "text": parsed_content["text"],
            }
        )

    def handle_media(self, query: str) -> None:
        params = parse_qs(query)
        target = (params.get("url") or [""])[0].strip()
        if not target:
            self.send_json({"error": "Missing url parameter."}, status=400)
            return

        parsed = urlparse(target)
        if parsed.scheme not in {"http", "https"}:
            self.send_json({"error": "Only http and https URLs are supported."}, status=400)
            return

        try:
            request = Request(target, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=20) as response:
                declared_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                content_type = infer_media_content_type(target, declared_type)
                body = response.read()
        except Exception as exc:
            self.send_json({"error": f"Failed to fetch media URL: {exc}"}, status=502)
            return

        if not (content_type.startswith("image/") or content_type.startswith("video/")):
            self.send_json(
                {
                    "error": f"Unsupported media content type: {content_type or 'unknown'}",
                    "content_type": content_type,
                },
                status=415,
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving realitycheck on http://127.0.0.1:{PORT}")
    server.serve_forever()
