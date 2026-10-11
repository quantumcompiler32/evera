"""Save distinct raw frames from Evera's LeRobot-compatible ZMQ camera hub.

The camera hub publishes JSON with base64 JPEG images. Its JPEG channel order is
RGB by convention, even though OpenCV decodes it into a BGR-shaped array.
"""

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import zmq


def decode_camera_frame(message: str, camera_name: str) -> np.ndarray:
    payload = json.loads(message)
    encoded = payload["images"][camera_name]
    pixels = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    decoded = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Camera hub sent an invalid JPEG")
    return cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)


def signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="tcp://127.0.0.1:5557")
    parser.add_argument("--camera-name", default="camera1")
    parser.add_argument("--output", type=Path, required=True,
                        help="One folder per recording session; keep sessions separate for data splits")
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--min-change", type=float, default=5,
                        help="Minimum mean pixel difference since the last saved photo")
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    args = parser.parse_args()
    if args.seconds <= 0 or args.min_change < 0:
        parser.error("seconds must be positive and min-change cannot be negative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output session already contains files: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.connect(args.address)
    previous = None
    last_saved = None
    stable = 0
    saved = []
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            try:
                frame = decode_camera_frame(socket.recv_string(), args.camera_name)
            except zmq.Again as error:
                raise RuntimeError(f"No frames from {args.address} for 3 seconds") from error
            if frame.shape[:2] != (args.expected_height, args.expected_width):
                raise ValueError(f"Unexpected camera size {frame.shape[:2]}")
            small = signature(frame)
            motion = float(np.mean(np.abs(small - previous))) if previous is not None else 100.0
            previous = small
            stable = stable + 1 if motion < 12 else 0
            change = float(np.mean(np.abs(small - last_saved))) if last_saved is not None else 100.0
            if stable < 2 or change < args.min_change:
                continue
            filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ.jpg")
            if not cv2.imwrite(str(args.output / filename), frame):
                raise RuntimeError(f"Could not save {filename}")
            saved.append({"file": filename, "change": round(change, 1)})
            print(f"saved {filename} change={change:.1f}", flush=True)
            last_saved = small
            stable = 0
    finally:
        socket.close(linger=0)
        context.term()
        (args.output / "capture_manifest.json").write_text(json.dumps({
            "camera": args.camera_name, "address": args.address,
            "width": args.expected_width, "height": args.expected_height,
            "frames": saved,
        }, indent=2) + "\n")
    print(f"Saved {len(saved)} distinct stable frames", flush=True)


if __name__ == "__main__":
    main()
