"""Choose one sharp representative per time window from a camera recording."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between selected frames")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output already contains files: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.session.glob("*.jpg"))
    if not paths:
        parser.error(f"No JPEG frames in {args.session}")

    first = datetime.strptime(paths[0].stem, "%Y%m%dT%H%M%S%fZ")
    groups = {}
    for path in paths:
        timestamp = datetime.strptime(path.stem, "%Y%m%dT%H%M%S%fZ")
        bucket = int((timestamp - first).total_seconds() / args.interval)
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable frame: {path}")
        score = sharpness(frame)
        entry = groups.get(bucket)
        if entry is None or score > entry[0]:
            groups[bucket] = (score, path)

    selected = []
    tiles = []
    for _, (score, path) in sorted(groups.items()):
        target = args.output / path.name
        shutil.copy2(path, target)
        selected.append({"file": path.name, "sharpness": round(score, 1)})
        frame = cv2.imread(str(path))
        tile = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (320, 25), (0, 0, 0), -1)
        cv2.putText(tile, path.stem[9:15] + f"  S={score:.0f}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    (args.output / "selection.json").write_text(json.dumps({
        "source_session": str(args.session), "interval_seconds": args.interval,
        "frames": selected,
    }, indent=2) + "\n")
    columns = 5
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * 240, columns * 320, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * 240:(row + 1) * 240, column * 320:(column + 1) * 320] = tile
    cv2.imwrite(str(args.output / "contact-sheet.jpg"), sheet)
    print(f"Selected {len(selected)} of {len(paths)} frames; contact sheet: {args.output / 'contact-sheet.jpg'}")


if __name__ == "__main__":
    main()
