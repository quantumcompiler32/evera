"""Propose multimeter-body masks from hand-drawn boxes on camera1 frames.

Boxes must enclose the meter housing only. Loose probes and leads are excluded
from the target mask. Every SAM result remains unreviewed until corrected.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.class_map import TARGET_CLASSES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True,
                        help="JSON mapping frame filename to body boxes [[x1,y1,x2,y2], ...]; [] for absent")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output already contains files: {args.output}")
    paths = sorted(path for path in args.frames.glob("*.jpg")
                   if re.fullmatch(r"\d{8}T\d{12}Z\.jpg", path.name))
    if not paths:
        parser.error("No timestamped camera frames found")
    boxes_by_frame = json.loads(args.boxes.read_text())
    if set(boxes_by_frame) != {path.name for path in paths}:
        parser.error("Boxes JSON must include every selected frame, including [] for meter-absent frames")

    from segment_anything import SamPredictor, sam_model_registry

    predictor = SamPredictor(sam_model_registry["vit_b"](
        checkpoint=str(args.checkpoint)).to(args.device))
    for folder in ("images", "masks", "previews"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = {"classes": list(TARGET_CLASSES), "session": args.frames.name,
                "source": "box_sam_proposal", "images": []}
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (480, 640):
            raise ValueError(f"Expected readable 640x480 frame: {path}")
        boxes = boxes_by_frame[path.name]
        if not isinstance(boxes, list):
            raise ValueError(f"Expected a box list for {path.name}")
        instances = []
        preview = frame.copy()
        if boxes:
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for index, box in enumerate(boxes):
            if (not isinstance(box, list) or len(box) != 4 or
                    not all(isinstance(v, (int, float)) for v in box)):
                raise ValueError(f"Invalid box in {path.name}: {box}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
                raise ValueError(f"Out-of-bounds box in {path.name}: {box}")
            masks, scores, _ = predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            mask_path = f"masks/{path.stem}-{index:02d}.png"
            if not cv2.imwrite(str(args.output / mask_path), mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Could not write {mask_path}")
            preview[mask] = (0, 0, 255)
            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 0, 255), 1)
            instances.append({"class": "multimeter", "mask": mask_path,
                              "source": "box_sam_proposal"})
        (args.output / "images" / path.name).write_bytes(path.read_bytes())
        cv2.imwrite(str(args.output / "previews" / path.name),
                    cv2.addWeighted(frame, 0.5, preview, 0.5, 0))
        manifest["images"].append({"image": f"images/{path.name}",
                                   "split": args.split, "instances": instances,
                                   "reviewed": False})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(paths)} frames; all masks are unreviewed proposals")


if __name__ == "__main__":
    main()
