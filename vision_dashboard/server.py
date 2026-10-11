"""Local-only Evera dataset and training dashboard (binds to loopback)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
LABELS = ROOT / "datasets" / "dashboard-labels"
OUTPUTS = ROOT / "outputs" / "dashboard"
PROPOSALS = ROOT / "datasets" / "segmentation"
CLASSES_FILE = HERE / "classes.json"
ROLES_FILE = HERE / "roles.json"
FRAME_RE = re.compile(r"\d{8}T\d{12}Z\.jpg")
NAME_RE = re.compile(r"[a-z][a-z0-9_-]{1,80}")
lock = Lock()
job: dict = {"process": None, "kind": None, "log": None, "output": None}


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def classes() -> list[str]:
    return read_json(CLASSES_FILE, ["raspberry_pi", "arduino", "pinecil", "wire", "tape_measure", "multimeter"])


def session_path(name: str) -> Path:
    if not NAME_RE.fullmatch(name):
        raise ValueError("Invalid session name")
    path = CAPTURES / name
    if not path.is_dir():
        raise ValueError("Session does not exist")
    return path


def frame_path(session: str, filename: str) -> Path:
    if not FRAME_RE.fullmatch(filename):
        raise ValueError("Invalid frame name")
    path = session_path(session) / filename
    if not path.is_file():
        raise ValueError("Frame does not exist")
    return path


def annotation_path(session: str, filename: str) -> Path:
    frame_path(session, filename)
    return LABELS / session / (Path(filename).stem + ".json")


def session_summary(name: str, roles: dict) -> dict:
    frames = [p for p in session_path(name).iterdir() if FRAME_RE.fullmatch(p.name)]
    reviewed = sum(bool(read_json(LABELS / name / (p.stem + ".json"), {}).get("reviewed")) for p in frames)
    labeled = sum((LABELS / name / (p.stem + ".json")).exists() for p in frames)
    return {"name": name, "frames": len(frames), "labeled": labeled,
            "reviewed": reviewed, "role": roles.get(name, "ignore")}


def job_state() -> dict:
    with lock:
        process = job["process"]
        log = job["log"]
        result = {"kind": job["kind"], "running": process is not None and process.poll() is None,
                  "exit_code": None if process is None else process.poll(), "output": job["output"]}
    if log and Path(log).exists():
        content = Path(log).read_text(errors="replace")
        result["log"] = "\n".join(content.splitlines()[-25:])
        if result["running"]:
            if "Starting provisional training" in content or kind_is_train(result):
                progress = re.findall(r"(?:^|\s)(\d+)/(\d+)\s+\d+(?:\.\d+)?G", content)
                result["summary"] = (f"Training the new model · round {progress[-1][0]} of {progress[-1][1]}"
                                     if progress else "Training the new model…")
            elif "proposing outlines" in content:
                result["summary"] = "Finding the object in the captured pictures…"
            else:
                result["summary"] = "Capturing pictures · follow the move and hands-out cues"
        elif result["exit_code"] == 0:
            result["summary"] = "New provisional model ready. The live model was not changed."
        else:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            result["summary"] = "Session stopped: " + (lines[-1][:180] if lines else "see details")
    else:
        latest = sorted(OUTPUTS.glob("run-*/model_paths.txt")) if OUTPUTS.exists() else []
        result["summary"] = ("Last provisional model: " + latest[-1].read_text().splitlines()[0]
                             if latest else "Ready to start a guided training session.")
    return result


def kind_is_train(current: dict) -> bool:
    return current["kind"] == "train"


def start_job(kind: str, command: list[str], output: Path) -> dict:
    with lock:
        if job["process"] is not None and job["process"].poll() is None:
            raise ValueError("A capture or training job is already running")
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = OUTPUTS / f"{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        job.update(process=process, kind=kind, log=str(log_path), output=str(output))
    return job_state()


def import_proposals(name: str) -> dict:
    import cv2

    if not NAME_RE.fullmatch(name):
        raise ValueError("Invalid proposal name")
    folder = PROPOSALS / name
    manifest = read_json(folder / "manifest.json", None)
    if not manifest:
        raise ValueError("Proposal manifest is missing")
    session = manifest.get("session")
    session_path(session)
    imported = 0
    for item in manifest["images"]:
        filename = Path(item["image"]).name
        target = annotation_path(session, filename)
        if target.exists():
            continue
        polygons = []
        for instance in item["instances"]:
            if instance["class"] not in classes():
                continue
            mask = cv2.imread(str(folder / instance["mask"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            contours, _ = cv2.findContours((mask > 127).astype("uint8"), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            height, width = mask.shape
            points = [[round(float(x / width), 5), round(float(y / height), 5)]
                      for x, y in contour[:, 0]]
            polygons.append({"class": instance["class"], "points": points})
        write_json(target, {"reviewed": False, "polygons": polygons,
                            "source": f"proposal:{name}"})
        imported += 1
    return {"session": session, "imported": imported}


def build_dataset(run_name: str) -> tuple[Path, dict]:
    base = ROOT / "datasets" / "yolo-seg" / "workshop-provisional-clean"
    if not (base / "train.txt").exists() or not (base / "val.txt").exists():
        raise ValueError("The existing four-class training dataset is missing")
    output = OUTPUTS / run_name / "dataset"
    if output.exists():
        raise ValueError("Training output already exists")
    names = classes()
    if names[:4] != ["raspberry_pi", "arduino", "pinecil", "wire"]:
        raise ValueError("The original four class IDs must stay unchanged")
    roles = read_json(ROLES_FILE, {})
    paths = {split: (base / f"{split}.txt").read_text().splitlines() if split != "test" else []
             for split in ("train", "val", "test")}
    counts = {name: 0 for name in names}
    for session, split in roles.items():
        if split not in paths:
            continue
        for frame in sorted(session_path(session).iterdir()):
            if not FRAME_RE.fullmatch(frame.name):
                continue
            annotation = read_json(LABELS / session / (frame.stem + ".json"), None)
            if not annotation or not (annotation.get("reviewed") or annotation.get("auto_accepted")):
                continue
            stem = f"{session}-{frame.stem}"
            image = output / "images" / split / f"{stem}.jpg"
            label = output / "labels" / split / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(frame.read_bytes())
            lines = []
            for polygon in annotation["polygons"]:
                name = polygon["class"]
                if name not in names:
                    raise ValueError(f"Unknown class in annotation: {name}")
                points = polygon["points"]
                if len(points) < 3:
                    raise ValueError(f"Polygon needs three points: {frame.name}")
                coords = [coordinate for pair in points for coordinate in pair]
                if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in coords):
                    raise ValueError(f"Polygon is outside image: {frame.name}")
                lines.append(" ".join([str(names.index(name))] + [f"{v:.6f}" for v in coords]))
                counts[name] += 1
            label.write_text("\n".join(lines) + ("\n" if lines else ""))
            paths[split].append(str(image))
    if not any(counts[name] for name in names[4:]):
        raise ValueError("Review and assign training frames for a new object first")
    for split, entries in paths.items():
        (output / f"{split}.txt").write_text("\n".join(entries) + ("\n" if entries else ""))
    yaml = [f"path: {json.dumps(str(output))}", "train: train.txt", "val: val.txt", "test: test.txt", "names:"]
    yaml.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(yaml) + "\n")
    write_json(output / "dataset_report.json", {"new_instances": counts,
                                                  "frames": {key: len(value) for key, value in paths.items()},
                                                  "provisional": True})
    return output / "data.yaml", counts


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, payload, content_type: str = "application/json") -> None:
        body = (json.dumps(payload).encode() if content_type == "application/json" else payload)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def request_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 1_000_000:
            raise ValueError("Invalid request size")
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        try:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self.respond(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                roles = read_json(ROLES_FILE, {})
                sessions = [session_summary(path.name, roles) for path in CAPTURES.iterdir()
                            if path.is_dir() and NAME_RE.fullmatch(path.name)
                            and any(FRAME_RE.fullmatch(p.name) for p in path.iterdir())]
                proposals = [p.name for p in PROPOSALS.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
                self.respond(200, {"classes": classes(), "sessions": sorted(sessions, key=lambda s: s["name"], reverse=True),
                                   "proposals": sorted(proposals), "job": job_state()})
            elif url.path == "/api/session":
                name = query["name"][0]
                frames = sorted(p.name for p in session_path(name).iterdir() if FRAME_RE.fullmatch(p.name))
                self.respond(200, {"name": name, "frames": frames})
            elif url.path == "/api/annotation":
                self.respond(200, read_json(annotation_path(query["session"][0], query["file"][0]),
                                            {"reviewed": False, "polygons": []}))
            elif url.path == "/api/image":
                data = frame_path(query["session"][0], query["file"][0]).read_bytes()
                self.respond(200, data, "image/jpeg")
            elif url.path == "/api/job":
                self.respond(200, job_state())
            else:
                self.respond(404, {"error": "Not found"})
        except (ValueError, KeyError, IndexError, OSError) as error:
            self.respond(400, {"error": str(error)})

    def do_POST(self) -> None:
        try:
            body = self.request_body()
            path = urlsplit(self.path).path
            if path == "/api/class":
                name = body["name"].strip().lower().replace(" ", "_")
                if not re.fullmatch(r"[a-z][a-z0-9_]{1,39}", name):
                    raise ValueError("Use letters, numbers, and underscores for the class name")
                names = classes()
                if name in names:
                    raise ValueError("That class already exists")
                names.append(name)
                write_json(CLASSES_FILE, names)
                result = {"classes": names}
            elif path == "/api/role":
                name = body["session"]
                session_path(name)
                role = body["role"]
                if role not in ("train", "val", "test", "ignore"):
                    raise ValueError("Invalid role")
                roles = read_json(ROLES_FILE, {})
                roles[name] = role
                write_json(ROLES_FILE, roles)
                result = {"session": name, "role": role}
            elif path == "/api/annotation":
                session, filename = body["session"], body["file"]
                polygons = body["polygons"]
                if not isinstance(polygons, list) or len(polygons) > 100:
                    raise ValueError("Invalid polygons")
                for polygon in polygons:
                    if polygon["class"] not in classes() or len(polygon["points"]) < 3:
                        raise ValueError("Invalid class or polygon")
                    if any(len(point) != 2 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1
                                                     for v in point) for point in polygon["points"]):
                        raise ValueError("Polygon points must be within the image")
                write_json(annotation_path(session, filename),
                           {"reviewed": bool(body.get("reviewed")), "polygons": polygons,
                            "source": "dashboard"})
                result = {"saved": True}
            elif path == "/api/import-proposals":
                result = import_proposals(body["name"])
            elif path == "/api/capture":
                target = body["class"]
                if target not in classes():
                    raise ValueError("Unknown class")
                seconds = int(body.get("seconds", 45))
                if not 5 <= seconds <= 120:
                    raise ValueError("Capture must be 5–120 seconds")
                name = f"camera1-{target}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                output = CAPTURES / name
                result = start_job("capture", [sys.executable, str(ROOT / "vision_training" / "sample_frames.py"),
                                               "--output", str(output), "--seconds", str(seconds)], output)
            elif path == "/api/guided":
                target = body["class"]
                if target not in classes():
                    raise ValueError("Unknown class")
                seconds = int(body.get("seconds", 45))
                if not 21 <= seconds <= 120:
                    raise ValueError("Guided capture must be 21–120 seconds")
                name = f"camera1-{target}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                output = CAPTURES / name
                result = start_job("guided", [sys.executable, str(HERE / "guided_session.py"),
                                              "--target", target, "--session", str(output),
                                              "--seconds", str(seconds)], output)
            elif path == "/api/select":
                source = session_path(body["session"])
                output = CAPTURES / f"{source.name}-selected"
                if output.exists():
                    raise ValueError("Selection already exists for this session")
                result = start_job("select", [sys.executable, str(ROOT / "vision_training" / "select_frames.py"),
                                              "--session", str(source), "--output", str(output)], output)
            elif path == "/api/train":
                epochs = int(body.get("epochs", 15))
                if not 1 <= epochs <= 100:
                    raise ValueError("Epochs must be 1–100")
                run_name = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                data, counts = build_dataset(run_name)
                output = OUTPUTS / run_name / "model"
                weights = ROOT / "outputs" / "yolo-seg" / "workshop-provisional" / "weights" / "best.pt"
                if not weights.exists():
                    raise ValueError("Base model weights are missing")
                result = start_job("train", [sys.executable, str(HERE / "train_job.py"),
                                             "--data", str(data), "--weights", str(weights),
                                             "--output", str(output), "--epochs", str(epochs)], output)
                result["new_instances"] = counts
            else:
                self.respond(404, {"error": "Not found"})
                return
            self.respond(200, result)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            self.respond(400, {"error": str(error)})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    CAPTURES.mkdir(exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    print(f"Evera training dashboard: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
