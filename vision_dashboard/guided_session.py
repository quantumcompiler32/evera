"""Capture one moving object, propose masks, and train a separate provisional model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision_training.sample_frames import decode_camera_frame  # noqa: E402
from vision_dashboard.server import LABELS, OUTPUTS, ROLES_FILE, build_dataset, classes, read_json, write_json  # noqa: E402


def foreground_box(frame: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    difference = cv2.absdiff(frame, background)
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    changed = (difference > 23).astype(np.uint8) * 255
    changed[:, :280] = 0  # fixed Evera camera: moving object belongs on the right-side mat
    changed[:120, :] = 0
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(changed, 8)
    candidates = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
                  if 90 <= stats[i, cv2.CC_STAT_AREA] <= 18000]
    if not candidates:
        return None
    _, index = max(candidates)
    x, y, width, height, _ = stats[index]
    if width < 10 or height < 10:
        return None
    box = np.array([max(0, x - 10), max(0, y - 10),
                    min(frame.shape[1] - 1, x + width + 10),
                    min(frame.shape[0] - 1, y + height + 10)], dtype=np.float32)
    return centers[index].astype(np.float32), box


def capture(session: Path, seconds: int) -> tuple[np.ndarray, list[Path]]:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.connect("tcp://127.0.0.1:5557")
    session.mkdir(parents=True)
    baseline_frames = []
    selected = []
    current_cycle = -1
    best: tuple[float, np.ndarray] | None = None
    started = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= seconds:
                break
            frame = decode_camera_frame(socket.recv_string(), "camera1")
            if frame.shape[:2] != (480, 640):
                raise ValueError(f"camera1 frame must be 640×480, got {frame.shape}")
            if elapsed < 5:
                if elapsed >= 2:
                    baseline_frames.append(frame)
                continue
            cycle = int((elapsed - 5) // 8)
            phase = (elapsed - 5) % 8
            if cycle != current_cycle:
                if best is not None:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    path = session / f"{stamp}.jpg"
                    cv2.imwrite(str(path), best[1])
                    selected.append(path)
                    print(f"Saved placement {current_cycle + 1}: {path.name}", flush=True)
                current_cycle = cycle
                best = None
            if phase < 4.5:
                continue  # hand-moving window and first part of the hold
            sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if best is None or sharpness > best[0]:
                best = (sharpness, frame.copy())
        if best is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = session / f"{stamp}.jpg"
            cv2.imwrite(str(path), best[1])
            selected.append(path)
            print(f"Saved placement {current_cycle + 1}: {path.name}", flush=True)
    finally:
        socket.close(linger=0)
        context.term()
    if len(baseline_frames) < 3:
        raise RuntimeError("No clear background frames; check the camera feed and retry")
    background = np.median(np.stack(baseline_frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(session / "background.jpg"), background)
    write_json(session / "capture_manifest.json", {"camera": "camera1", "guided": True,
                                                  "seconds": seconds, "frames": [p.name for p in selected]})
    return background, selected


def propose(session: Path, target: str, background: np.ndarray, images: list[Path]) -> int:
    from segment_anything import SamPredictor, sam_model_registry

    checkpoint = ROOT / "models" / "sam_vit_b_01ec64.pth"
    if not checkpoint.exists():
        raise RuntimeError("SAM mask proposal weights are missing")
    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to("cuda"))
    accepted = 0
    for image in images:
        frame = cv2.imread(str(image))
        seed = foreground_box(frame, background)
        if seed is None:
            print(f"No clear object in {image.name}; skipped", flush=True)
            continue
        point, box = seed
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(point_coords=point.reshape(1, 2),
                                             point_labels=np.array([1]), box=box,
                                             multimask_output=True)
        mask = masks[int(np.argmax(scores))].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if not 150 <= cv2.contourArea(contour) <= 19000 or len(contour) < 3:
            print(f"Uncertain mask in {image.name}; skipped", flush=True)
            continue
        height, width = mask.shape
        points = [[round(float(x / width), 5), round(float(y / height), 5)]
                  for x, y in contour[:, 0]]
        write_json(LABELS / session.name / f"{image.stem}.json",
                   {"reviewed": False, "auto_accepted": True,
                    "source": "guided_background_sam_proposal",
                    "polygons": [{"class": target, "points": points}]})
        accepted += 1
        print(f"Auto-outlined {image.name}", flush=True)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.target not in classes():
        parser.error("Unknown target class")
    if not 21 <= args.seconds <= 120:
        parser.error("Guided capture must be 21–120 seconds")
    if args.session.exists():
        parser.error("Session already exists")
    print("Clear the right-side mat and keep hands out for five seconds.", flush=True)
    background, images = capture(args.session, args.seconds)
    print(f"Captured {len(images)} object placements; proposing outlines", flush=True)
    accepted = propose(args.session, args.target, background, images)
    print(f"Auto-outlined {accepted} placements", flush=True)
    if accepted < 3:
        raise RuntimeError("Too few clear object outlines for training. Keep hands out longer and retry.")
    roles = read_json(ROLES_FILE, {})
    roles[args.session.name] = "train"
    write_json(ROLES_FILE, roles)
    run_name = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    data, counts = build_dataset(run_name)
    print(f"Starting provisional training: {json.dumps(counts)}", flush=True)
    output = OUTPUTS / run_name / "model"
    weights = ROOT / "outputs" / "yolo-seg" / "workshop-provisional" / "weights" / "best.pt"
    subprocess.run([sys.executable, str(ROOT / "vision_dashboard" / "train_job.py"),
                    "--data", str(data), "--weights", str(weights),
                    "--output", str(output), "--epochs", str(args.epochs)], check=True)
    print(f"Finished. New model is in {output.parent}; live model unchanged.", flush=True)


if __name__ == "__main__":
    main()
