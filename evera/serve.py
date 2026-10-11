#!/usr/bin/env python3
"""Serve the evera app and pass /api/* and /stream/* through to the robot dashboard.

    python3 evera/serve.py [dashboard_url] [port]
    python3 evera/serve.py http://169.233.72.19:8080 8765

Then open http://<this computer's IP>:8765/ on the iPad (same Wi-Fi). Because the app, the
camera streams and the arm API all come from this one address, nothing needs to change on the
GMKtec. For voice (the iPad mic needs HTTPS), put a tunnel in front of this port:
    cloudflared tunnel --url http://localhost:8765
"""
import os
import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

DASHBOARD = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EVERA_DASHBOARD", "http://169.233.72.19:8080")).rstrip("/")
PORT = int(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PORT", 8765))
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/api/status", "/stream/")):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def proxied(self):
        return self.path.startswith(("/api/", "/stream/"))

    def proxy(self, body=None):
        req = urllib.request.Request(DASHBOARD + self.path, data=body, method=self.command)
        if body is not None:
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            up = urllib.request.urlopen(req, timeout=8)
        except urllib.error.HTTPError as e:
            up = e
        except Exception as e:  # dashboard unreachable
            msg = f"robot dashboard unreachable at {DASHBOARD}: {e}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(up.status if hasattr(up, "status") else up.code)
        ctype = up.headers.get("Content-Type", "application/octet-stream")
        self.send_header("Content-Type", ctype)
        streaming = ctype.startswith("multipart/")
        if not streaming and up.headers.get("Content-Length"):
            self.send_header("Content-Length", up.headers["Content-Length"])
        self.end_headers()
        try:
            while True:  # MJPEG camera streams never end: relay chunks as they arrive
                chunk = up.read1(65536) if streaming else up.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                if streaming:
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            up.close()

    def do_GET(self):
        if self.proxied():
            return self.proxy()
        return super().do_GET()

    def do_POST(self):
        if not self.proxied():
            return self.send_error(404)
        n = int(self.headers.get("Content-Length") or 0)
        return self.proxy(self.rfile.read(n) if n else b"{}")


if __name__ == "__main__":
    print(f"evera on http://0.0.0.0:{PORT}  ->  robot dashboard {DASHBOARD}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
