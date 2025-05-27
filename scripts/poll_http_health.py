#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen


def poll_url(
    url: str,
    *,
    timeout: float,
    interval: float,
    expect_status: int = 200,
    expect_content_type: str | None = None,
    body_fragment: bytes | None = None,
    abort_message: Callable[[], str | None] | None = None,
) -> bytes:
    deadline = time.monotonic() + timeout
    last_error = "service did not answer yet"

    while time.monotonic() < deadline:
        if abort_message is not None and (message := abort_message()) is not None:
            raise RuntimeError(message)

        try:
            with urlopen(url, timeout=3) as response:
                if response.status != expect_status:
                    last_error = f"unexpected status {response.status}"
                    time.sleep(interval)
                    continue

                actual_content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                if (
                    expect_content_type is not None
                    and actual_content_type != expect_content_type
                ):
                    last_error = (
                        "unexpected content type "
                        f"{actual_content_type or 'unknown'} "
                        f"(expected {expect_content_type})"
                    )
                    time.sleep(interval)
                    continue

                body = response.read()
                if body_fragment is not None and body_fragment.lower() not in body.lower():
                    last_error = (
                        f"response body did not include expected fragment {body_fragment!r}"
                    )
                    time.sleep(interval)
                    continue

                return body
        except URLError as exc:
            last_error = str(exc.reason)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(interval)

    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll an HTTP endpoint until it returns the expected health response."
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--expect-status", type=int, default=200)
    parser.add_argument("--expect-content-type")
    parser.add_argument("--body-fragment")
    parser.add_argument("--write-body")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    body = poll_url(
        args.url,
        timeout=args.timeout,
        interval=args.interval,
        expect_status=args.expect_status,
        expect_content_type=args.expect_content_type,
        body_fragment=args.body_fragment.encode("utf-8")
        if args.body_fragment is not None
        else None,
    )
    if args.write_body:
        Path(args.write_body).write_bytes(body)
    else:
        print(body.decode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
