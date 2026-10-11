"""Capture varied, stable Limelight frames during one short arrangement session."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def signature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="http://172.29.0.1:5802/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=50)
    parser.add_argument("--min-change", type=float, default=5.0,
                        help="Mean grayscale difference from last saved scene")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_FFMPEG)
    if not camera.isOpened():
        raise RuntimeError("Limelight stream did not open")
    deadline = time.monotonic() + args.seconds
    previous = None
    last_saved = None
    stable = 0
    saved = []
    try:
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Limelight stream stopped")
            small = signature(frame)
            motion = float(np.mean(np.abs(small - previous))) if previous is not None else 100.0
            previous = small
            stable = stable + 1 if motion < 12.0 else 0
            changed = float(np.mean(np.abs(small - last_saved))) if last_saved is not None else 100.0
            if stable < 2 or changed < args.min_change:
                continue
            sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if sharpness < 50:
                continue
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ.jpg")
            file = args.output / name
            if cv2.imwrite(str(file), frame):
                saved.append({"file": name, "sharpness": round(sharpness, 1), "change": round(changed, 1)})
                last_saved = small
                stable = 0
                print(f"saved {name} change={changed:.1f} sharpness={sharpness:.1f}", flush=True)
    finally:
        camera.release()
        (args.output / "capture_manifest.json").write_text(json.dumps(saved, indent=2) + "\n")
    print(f"Saved {len(saved)} distinct stable frames", flush=True)


if __name__ == "__main__":
    main()
