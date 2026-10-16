"""HTTP gateway on the laptop: the phone POSTs detections, the browser shows them live.

Run with ``uv run door-gateway`` and open http://127.0.0.1:8000. Over USB the phone reaches it
after ``adb reverse tcp:8000 tcp:8000``; over Wi-Fi start it with ``--host 0.0.0.0`` and point
the phone at the laptop's LAN address.

Endpoints:
    POST /frames       a batch from the phone (see messages.py)
    GET  /             the live page
    GET  /api/latest   the current state as JSON, polled by the page
    GET  /preview.jpg  the newest preview picture
"""

import argparse
import json
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from door_counter.gateway.live_state import LiveSnapshot, LiveState
from door_counter.gateway.messages import MessageError, parse_batch

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 8 * 1024 * 1024


def snapshot_to_json(*, snapshot: LiveSnapshot, now_wall_time_s: float) -> dict[str, object]:
    """Turn a snapshot into the JSON the live page reads.

    Args:
        snapshot: the current state.
        now_wall_time_s: laptop clock now, seconds since 1970.

    Returns:
        A JSON-ready dict. The preview picture itself is left out (it is served at /preview.jpg).
    """
    preview = snapshot.preview
    last = snapshot.last_batch_wall_time_s
    return {
        "session_id": snapshot.session_id,
        "latest_frame": None if snapshot.latest_frame is None else asdict(snapshot.latest_frame),
        "preview": None if preview is None else {"frame_id": preview.frame_id, "rotation_deg": preview.rotation_deg},
        "preview_boxes": [asdict(box) for box in snapshot.preview_boxes],
        "health": None if snapshot.health is None else asdict(snapshot.health),
        "queued_frames": snapshot.queued_frames,
        "dropped_frames": snapshot.dropped_frames,
        "seconds_since_last_batch": None if last is None else now_wall_time_s - last,
    }


def make_handler(*, state: LiveState, page: bytes) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to one shared state.

    Args:
        state: the store shared by all requests.
        page: the live page HTML.

    Returns:
        A handler class for ThreadingHTTPServer.
    """

    class GatewayHandler(BaseHTTPRequestHandler):
        """Handles one HTTP request from the phone or the browser."""

        def do_POST(self) -> None:
            """Take a batch from the phone."""
            if self.path != "/frames":
                self.send_error(HTTPStatus.NOT_FOUND)
                return None
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length > 0 else HTTPStatus.LENGTH_REQUIRED)
                return None
            try:
                batch = parse_batch(payload=json.loads(self.rfile.read(length)))
            except (json.JSONDecodeError, UnicodeDecodeError, MessageError) as e:
                self.send_error(HTTPStatus.BAD_REQUEST, explain=str(e))
                return None
            state.add_batch(batch=batch, received_wall_time_s=time.time())
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return None

        def do_GET(self) -> None:
            """Serve the page, the state as JSON, or the preview picture."""
            path = self.path.split("?", maxsplit=1)[0]
            if path == "/":
                self._send(body=page, content_type="text/html; charset=utf-8")
            elif path == "/api/latest":
                data = snapshot_to_json(snapshot=state.snapshot(), now_wall_time_s=time.time())
                self._send(body=json.dumps(data).encode(), content_type="application/json")
            elif path == "/preview.jpg":
                preview = state.snapshot().preview
                if preview is None:
                    self.send_error(HTTPStatus.NOT_FOUND, explain="no preview yet (is preview switched on?)")
                else:
                    self._send(body=preview.jpeg, content_type="image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return None

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (stdlib signature)
            """Stay quiet: the phone posts several times per second."""
            return None

        def _send(self, *, body: bytes, content_type: str) -> None:
            """Send a 200 response without caching.

            Args:
                body: the response body.
                content_type: its media type.
            """
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return None

    return GatewayHandler


def main() -> None:
    """Parse the command line and serve until Ctrl+C."""
    parser = argparse.ArgumentParser(description="Door counter gateway: receives the phone's detections.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to listen on (0.0.0.0 for Wi-Fi)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    page = resources.files("door_counter.gateway").joinpath("live.html").read_bytes()
    handler = make_handler(state=LiveState(), page=page)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"gateway on http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopped", flush=True)
    return None


if __name__ == "__main__":
    main()
