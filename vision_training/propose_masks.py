"""Create reviewable three-class mask proposals from saved camera1 frames.

These are model predictions, not ground truth. Correct every instance and add
missed objects before using a session for evaluation or final training.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.onnx_segmenter import Segmenter  # noqa: E402


CLASSES = ("raspberry_pi", "arduino", "pinecil")
COLORS = {"raspberry_pi": (40, 230, 60), "arduino": (250, 170, 40),
          "pinecil": (50, 120, 250)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True,
                        help="Assign an entire recording session to one split")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output already contains files: {args.output}")
    paths = sorted(path for path in args.frames.glob("*.jpg")
                   if re.fullmatch(r"\d{8}T\d{12}Z\.jpg", path.name))
    if not paths:
        parser.error(f"No JPEG frames in {args.frames}")
    for folder in ("images", "masks", "previews"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)

    model = Segmenter(str(args.model), conf_threshold=args.confidence)
    manifest = {"classes": list(CLASSES), "session": args.frames.name,
                "source": "onnx_proposal", "images": []}
    counts = {name: 0 for name in CLASSES}
    for image_path in paths:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Cannot read {image_path}")
        results = model.segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        preview = bgr.copy()
        instances = []
        for index, result in enumerate(results):
            if result.cls not in CLASSES:
                continue
            mask_path = f"masks/{image_path.stem}-{index:02d}.png"
            cv2.imwrite(str(args.output / mask_path), result.mask.astype(np.uint8) * 255)
            preview[result.mask] = COLORS[result.cls]
            x1, y1, _, _ = result.box
            cv2.putText(preview, f"{result.cls} {result.conf:.2f}",
                        (int(x1 * bgr.shape[1]), max(18, int(y1 * bgr.shape[0]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[result.cls], 2)
            instances.append({"class": result.cls, "mask": mask_path,
                              "confidence": round(result.conf, 4), "source": "onnx_proposal"})
            counts[result.cls] += 1
        (args.output / "images" / image_path.name).write_bytes(image_path.read_bytes())
        cv2.imwrite(str(args.output / "previews" / image_path.name),
                    cv2.addWeighted(bgr, 0.5, preview, 0.5, 0))
        manifest["images"].append({"image": f"images/{image_path.name}",
                                   "split": args.split, "instances": instances,
                                   "reviewed": False})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(paths)} frames; unreviewed proposals: {counts}")


if __name__ == "__main__":
    main()
