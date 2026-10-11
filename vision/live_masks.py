"""Limelight live view with grid, raw-frame capture, and optional CNN masks."""

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from segmentation_dataset import CLASS_NAMES
from segmentation_model import draw_masks, load_model, predict_frame


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Evera Limelight view</title>
<style>body{font:16px system-ui,sans-serif;background:#12151b;color:#f5f7fa;margin:0;padding:20px}
main{max-width:900px;margin:auto}h1{font-size:24px}p{color:#c6ceda}
img{display:block;width:100%;background:#000;border:1px solid #424b59}
button{font:inherit;padding:9px 14px;margin:8px 8px 0 0;border:0;border-radius:7px;background:#2878ee;color:white;cursor:pointer}
#status{min-height:24px;margin-top:12px}</style></head><body><main>
<h1>Limelight 3A · live mask view</h1><p id="mode">Checking camera…</p><p id="focus">Focus score: measuring…</p>
<img src="/stream" alt="Live Limelight feed with alignment grid and instance masks">
<p>Saved photos are raw camera frames without the grid or colored masks.</p>
<button onclick="capture('raspberry_pi')">Capture Pi</button><button onclick="capture('arduino')">Capture Arduino</button>
<button onclick="capture('pinecil')">Capture Pinecil</button><button onclick="capture('wire')">Capture wires</button>
<button onclick="capture('empty')">Capture empty scene</button><div id="status" role="status"></div>
<script>async function capture(label){try{const r=await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label})});const d=await r.json();document.getElementById('status').textContent=r.ok?'Saved '+d.file:'Capture failed: '+d.error}catch(e){document.getElementById('status').textContent='Capture failed: '+e}}
async function status(){try{const d=await(await fetch('/health')).json();document.getElementById('mode').textContent=d.camera_ready?(d.model_ready?'CNN mask overlay active':'Camera live; CNN checkpoint not loaded yet'):'Waiting for camera frames…';document.getElementById('focus').textContent=d.focus_score===null?'Focus score: waiting for frame':'Focus score: '+d.focus_score+' (higher is sharper)'}catch(e){document.getElementById('mode').textContent='Preview server unavailable'}}status();setInterval(status,1000)</script>
</main></body></html>"""


class Camera:
    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_at = 0.0
        self.running = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while self.running:
            backend = cv2.CAP_FFMPEG if isinstance(self.source, str) and self.source.startswith(("http://", "https://", "rtsp://")) else cv2.CAP_ANY
            device = cv2.VideoCapture(self.source, backend)
            if not device.isOpened():
                device.release()
                time.sleep(1)
                continue
            while self.running:
                ok, frame = device.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
                    self.last_frame_at = time.monotonic()
            device.release()
            with self.lock:
                self.frame = None
            time.sleep(0.5)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def grid(frame):
    height, width = frame.shape[:2]
    for fraction in (1 / 3, 2 / 3):
        cv2.line(frame, (round(width * fraction), 0), (round(width * fraction), height), (0, 205, 255), 1)
        cv2.line(frame, (0, round(height * fraction)), (width, round(height * fraction)), (0, 205, 255), 1)
    return frame


def focus_score(frame):
    height, width = frame.shape[:2]
    roi = frame[round(height * 0.3):round(height * 0.9), round(width * 0.25):round(width * 0.85)]
    roi = cv2.resize(roi, (800, 600), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var()), 1)


def serve(camera, captures: Path, port: int, checkpoint: Path | None, device: str, confidence: float):
    model = load_model(checkpoint, device) if checkpoint else None
    state = {"frame": None, "model_error": None}
    state_lock = threading.Lock()

    if model is not None:
        def infer():
            while camera.running:
                raw = camera.latest()
                if raw is None:
                    with state_lock:
                        state["frame"] = None
                    time.sleep(0.1)
                    continue
                try:
                    results = predict_frame(model, raw, device, confidence)
                    painted = draw_masks(raw, results)
                    with state_lock:
                        state["frame"] = painted
                        state["model_error"] = None
                except Exception as error:
                    with state_lock:
                        state["model_error"] = str(error)
                    time.sleep(1)
        threading.Thread(target=infer, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_bytes(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/health":
                with state_lock:
                    error = state["model_error"]
                frame = camera.latest()
                self.send_json(HTTPStatus.OK, {"camera_ready": frame is not None,
                                               "model_ready": model is not None and error is None,
                                               "focus_score": focus_score(frame) if frame is not None else None,
                                               "model_error": error})
            elif self.path == "/stream":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        with state_lock:
                            frame = None if state["frame"] is None else state["frame"].copy()
                        if camera.latest() is None:
                            frame = None
                        if frame is None:
                            frame = camera.latest()
                        if frame is None:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "Camera reconnecting...", (70, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        elif frame.shape[1] > 1280:
                            height = round(frame.shape[0] * 1280 / frame.shape[1])
                            frame = cv2.resize(frame, (1280, height), interpolation=cv2.INTER_AREA)
                        ok, encoded = cv2.imencode(".jpg", grid(frame), [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if ok:
                            payload = encoded.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                            self.wfile.write(payload + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/capture":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("Request too large")
                label = json.loads(self.rfile.read(length))["label"]
                if label not in CLASS_NAMES[1:] + ("empty",):
                    raise ValueError("Unknown label")
                frame = camera.latest()
                if frame is None:
                    raise ValueError("Camera frame is not ready")
                folder = captures / label
                folder.mkdir(parents=True, exist_ok=True)
                name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ.jpg")
                file = folder / name
                if not cv2.imwrite(str(file), frame):
                    raise ValueError("Could not save frame")
                self.send_json(HTTPStatus.OK, {"file": str(file.relative_to(captures))})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def send_bytes(self, status, payload, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status, value):
            self.send_bytes(status, json.dumps(value).encode(), "application/json")

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Limelight mask view listening on remote localhost:{port}", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="http://172.29.0.1:5802/")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--captures", type=Path, default=Path("captures/limelight-candidates"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    serve(Camera(source), args.captures, args.port, args.checkpoint, args.device, args.confidence)


if __name__ == "__main__":
    main()
