"""Private overhead-camera preview with a grid and a raw-frame capture button."""

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

from board_refine import refine_board_classes


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Overhead camera grid</title>
<style>
body{margin:0;background:#13151a;color:#f5f5f5;font:16px system-ui,sans-serif;padding:20px}
main{max-width:960px;margin:auto}h1{font-size:24px;margin:0 0 8px}
p{color:#b8bdc8}img{display:block;width:100%;max-width:800px;max-height:65vh;aspect-ratio:4/3;object-fit:contain;background:#000;border:1px solid #454954}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}button{font:inherit;padding:10px 16px;background:#2979ff;color:white;border:0;border-radius:8px;cursor:pointer}
#status{min-height:24px;margin-top:12px}
</style></head><body><main><h1>Overhead camera · 3×3 grid</h1>
<p>Keep the entire object in frame. The grid is only for positioning; captured training images are saved without it.</p>
<div class="actions"><button onclick="capture('raspberry_pi')">Capture Raspberry Pi</button>
<button onclick="capture('arduino')">Capture Arduino</button>
<button onclick="capture('soldering_iron')">Capture soldering iron</button>
<button onclick="capture('empty')">Capture empty scene</button></div>
<div id="status" role="status"></div><p><a style="color:#9ac4ff" href="/labeler">Draw bounding boxes on captured images</a></p>
<img src="/stream" alt="Live overhead camera with alignment grid">
<script>async function capture(label){const r=await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label})});const d=await r.json();document.getElementById('status').textContent=r.ok?'Saved '+d.file:'Capture failed: '+d.error}</script>
</main></body></html>"""

LABELER = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Draw bounding boxes</title><style>
body{margin:0;background:#13151a;color:#f5f5f5;font:16px system-ui,sans-serif;padding:20px}main{max-width:900px;margin:auto}
a{color:#9ac4ff}canvas{display:block;max-width:100%;height:auto;touch-action:none;border:1px solid #777;background:#000;cursor:crosshair}
select,button{font:inherit;margin:8px 8px 8px 0;padding:8px}button{background:#2979ff;color:#fff;border:0;border-radius:6px;cursor:pointer}
</style></head><body><main><h1>Draw bounding boxes</h1>
<p><a href="/">Back to live grid</a> · Select an image, choose a class, then drag a tight rectangle around each visible target. Save when finished. Empty scenes should have no boxes.</p>
<select id="files" aria-label="Captured image"></select><select id="class" aria-label="Box class"><option value="0">Raspberry Pi</option><option value="1">Arduino</option><option value="2">Soldering iron</option></select>
<button id="undo">Undo last box</button><button id="save">Save labels</button>
<canvas id="image" width="640" height="480"></canvas><div id="status" role="status"></div>
<script>
const canvas=document.getElementById('image'),ctx=canvas.getContext('2d'),list=document.getElementById('files');
let photo=new Image(),boxes=[],drag=null,current='';
function point(e){const r=canvas.getBoundingClientRect();return [Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))]}
const classNames=['Raspberry Pi','Arduino','Soldering iron'],classColors=['#00ff79','#47b9ff','#ffb000'];
function draw(){if(!photo.complete)return;ctx.drawImage(photo,0,0,canvas.width,canvas.height);for(const b of [...boxes,...(drag?[drag]:[])]){ctx.strokeStyle=classColors[b.cls];ctx.lineWidth=3;ctx.strokeRect(b.x1*canvas.width,b.y1*canvas.height,(b.x2-b.x1)*canvas.width,(b.y2-b.y1)*canvas.height);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(classNames[b.cls],b.x1*canvas.width,Math.max(14,b.y1*canvas.height-4))}}
async function load(name){current=name;photo=new Image();photo.src='/image?file='+encodeURIComponent(name);await photo.decode();canvas.width=photo.naturalWidth;canvas.height=photo.naturalHeight;boxes=await (await fetch('/annotations?file='+encodeURIComponent(name))).json();drag=null;draw();document.getElementById('status').textContent='Loaded '+name}
canvas.addEventListener('pointerdown',e=>{const [x,y]=point(e);drag={cls:Number(document.getElementById('class').value),x1:x,y1:y,x2:x,y2:y};canvas.setPointerCapture(e.pointerId);draw()});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const [x,y]=point(e);drag.x2=x;drag.y2=y;draw()});
canvas.addEventListener('pointerup',e=>{if(!drag)return;const [x,y]=point(e);const b={cls:drag.cls,x1:Math.min(drag.x1,x),y1:Math.min(drag.y1,y),x2:Math.max(drag.x1,x),y2:Math.max(drag.y1,y)};if(b.x2-b.x1>.005&&b.y2-b.y1>.005)boxes.push(b);drag=null;draw()});
document.getElementById('undo').onclick=()=>{boxes.pop();draw()};
document.getElementById('save').onclick=async()=>{const r=await fetch('/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file:current,boxes})});const d=await r.json();document.getElementById('status').textContent=r.ok?'Saved '+boxes.length+' boxes to '+d.file:'Save failed: '+d.error};
list.onchange=()=>load(list.value);
(async()=>{const files=await (await fetch('/files')).json();for(const f of files){const o=document.createElement('option');o.value=o.textContent=f;list.appendChild(o)}const q=new URLSearchParams(location.search).get('file');if(files.length){list.value=files.includes(q)?q:files[files.length-1];await load(list.value)}})();
</script></main></body></html>"""


def is_stream_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "rtsp://"))


class Camera:
    def __init__(self, source: int | str):
        self.source = source
        self.device = None
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        while self.running:
            if self.device is None:
                if isinstance(self.source, str) and not is_stream_url(self.source) and not Path(self.source).exists():
                    time.sleep(1)
                    continue
                backend = cv2.CAP_FFMPEG if isinstance(self.source, str) and is_stream_url(self.source) else cv2.CAP_ANY if isinstance(self.source, int) else cv2.CAP_V4L2
                self.device = cv2.VideoCapture(self.source, backend)
                if not self.device.isOpened():
                    self.device.release()
                    self.device = None
                    time.sleep(1)
                    continue
            ok, frame = self.device.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                with self.lock:
                    self.frame = None
                self.device.release()
                self.device = None
                time.sleep(0.5)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.running = False
        if self.device is not None:
            self.device.release()


def grid(frame):
    height, width = frame.shape[:2]
    color = (0, 210, 255)
    for fraction in (1 / 3, 2 / 3):
        cv2.line(frame, (round(width * fraction), 0), (round(width * fraction), height), color, 1)
        cv2.line(frame, (0, round(height * fraction)), (width, round(height * fraction)), color, 1)
    cv2.drawMarker(frame, (width // 2, height // 2), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return frame


def serve(camera: Camera, captures: Path, port: int, detector=None, confidence: float = 0.35, device: str | None = None, board_color_refine: bool = False) -> None:
    captures = captures.resolve()
    labels = captures.parent / "training_labels"

    def image_path(query: str) -> tuple[Path, Path]:
        name = parse_qs(query).get("file", [""])[0]
        path = (captures / name).resolve()
        if not path.is_relative_to(captures) or path.suffix.lower() != ".jpg" or not path.is_file():
            raise ValueError("Unknown captured image")
        label_path = labels / path.relative_to(captures).with_suffix(".txt")
        return path, label_path

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                page = PAGE.replace("Overhead camera · 3×3 grid", "Overhead camera · boxes + 3×3 grid") if detector else PAGE
                body = page.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/labeler":
                body = LABELER.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/files":
                files = sorted(str(p.relative_to(captures)) for p in captures.rglob("*.jpg")) if captures.exists() else []
                self.send_json(HTTPStatus.OK, files)
            elif path in {"/image", "/annotations"}:
                try:
                    image, label = image_path(parsed.query)
                except ValueError as error:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if path == "/image":
                    body = image.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    boxes = []
                    if label.exists():
                        for line in label.read_text().splitlines():
                            cls, xc, yc, w, h = map(float, line.split())
                            boxes.append({"cls": int(cls), "x1": xc - w / 2, "y1": yc - h / 2, "x2": xc + w / 2, "y2": yc + h / 2})
                    self.send_json(HTTPStatus.OK, boxes)
            elif path == "/stream":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        frame = camera.latest()
                        if frame is None:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "Camera reconnecting...", (75, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        else:
                            if detector is not None:
                                options = {"conf": confidence, "iou": 0.5, "agnostic_nms": True, "verbose": False}
                                if device is not None:
                                    options["device"] = device
                                result = detector.predict(frame, **options)[0]
                                if board_color_refine:
                                    refine_board_classes(frame, result)
                                frame = result.plot()
                        ok, encoded = cv2.imencode(".jpg", grid(frame), [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            image = encoded.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(image)}\r\n\r\n".encode())
                            self.wfile.write(image + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in {"/capture", "/annotations"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("Request is too large")
                data = json.loads(self.rfile.read(length))
                if path == "/capture":
                    label = data["label"]
                    if label not in {"raspberry_pi", "arduino", "soldering_iron", "empty"}:
                        raise ValueError("Unknown label")
                    frame = camera.latest()
                    if frame is None:
                        raise ValueError("Camera frame is not ready")
                    folder = captures / label
                    folder.mkdir(parents=True, exist_ok=True)
                    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".jpg"
                    saved = folder / name
                    if not cv2.imwrite(str(saved), frame):
                        raise ValueError("Could not save image")
                    payload = {"file": str(saved.relative_to(captures))}
                else:
                    _, label_file = image_path("file=" + quote(data["file"], safe=""))
                    boxes = data["boxes"]
                    lines = []
                    for box in boxes:
                        cls = box["cls"]
                        x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
                        if cls not in (0, 1, 2) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                            raise ValueError("Invalid box")
                        lines.append(f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
                    label_file.parent.mkdir(parents=True, exist_ok=True)
                    label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
                    payload = {"file": str(label_file)}
                status = HTTPStatus.OK
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                status, payload = HTTPStatus.BAD_REQUEST, {"error": str(error)}
            self.send_json(status, payload)

        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Grid feed listening on remote localhost:{port}", flush=True)
        try:
            server.serve_forever()
        finally:
            camera.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="Camera index, /dev/v4l/by-id path, or HTTP/RTSP stream URL")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--captures", type=Path, default=Path("captures/training_candidates"))
    parser.add_argument("--model", type=Path, help="Trained detection weights for live boxes")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", help="Inference device, e.g. 0 or cpu")
    parser.add_argument("--board-color-refine", action="store_true", help="Apply the workshop overhead-camera green Pi / blue Arduino correction")
    args = parser.parse_args()
    detector = None
    if args.model:
        if not args.model.is_file():
            parser.error(f"Model weights do not exist: {args.model}")
        from ultralytics import YOLO

        detector = YOLO(str(args.model))
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    serve(Camera(source), args.captures.resolve(), args.port, detector, args.confidence, args.device, args.board_color_refine)


if __name__ == "__main__":
    main()
