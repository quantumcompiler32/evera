"""Propose housing-only tape-measure masks from camera1 frames.

Bright-yellow pixels locate the case; SAM refines that location to a mask.
Every result is an unreviewed proposal, not ground truth.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.class_map import TARGET_CLASSES  # noqa: E402


def yellow_seed(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 105, 105), (42, 255, 255))
    # The fixed camera view places the working mat to the right of the robot.
    # Exclude the brightly colored hardware beside the left edge.
    yellow[:, :280] = 0
    yellow[:125, :] = 0
    count, components, stats, centroids = cv2.connectedComponentsWithStats(yellow, 8)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if 45 <= area <= 3500 and width >= 8 and height >= 7:
            candidates.append((area, index))
    if not candidates:
        return None
    _, index = max(candidates)
    x, y, width, height, _ = stats[index]
    center = centroids[index].astype(np.float32)
    margin = 17
    box = np.array([max(0, x - margin), max(0, y - margin),
                    min(frame.shape[1] - 1, x + width + margin),
                    min(frame.shape[0] - 1, y + height + margin)], dtype=np.float32)
    return center, box


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts", type=Path,
                        help="Reviewed per-frame point and box prompts; frames absent here are skipped")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output already contains files: {args.output}")

    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(args.device)
    predictor = SamPredictor(model)
    prompts = json.loads(args.prompts.read_text()) if args.prompts else None
    for folder in ("images", "masks", "previews"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = {"classes": list(TARGET_CLASSES),
                "session": args.frames.name, "source": "yellow_seed_sam_proposal", "images": []}
    for image_path in sorted(args.frames.glob("*.jpg")):
        if not re.fullmatch(r"\d{8}T\d{12}Z\.jpg", image_path.name):
            continue
        if prompts is not None and image_path.name not in prompts:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unreadable frame: {image_path}")
        if prompts is not None:
            prompt = prompts[image_path.name]
            seed = (np.asarray(prompt["point"], dtype=np.float32),
                    np.asarray(prompt["box"], dtype=np.float32))
        else:
            seed = yellow_seed(frame)
        instances = []
        preview = frame.copy()
        if seed is not None:
            center, box = seed
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(
                point_coords=center.reshape(1, 2), point_labels=np.array([1]),
                box=box, multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            if mask.sum() >= 40:
                mask_path = f"masks/{image_path.stem}-tape.png"
                cv2.imwrite(str(args.output / mask_path), mask.astype(np.uint8) * 255)
                preview[mask] = (0, 0, 255)
                instances.append({"class": "tape_measure", "mask": mask_path,
                                  "source": "yellow_seed_sam_proposal"})
            cv2.rectangle(preview, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)),
                          (255, 0, 255), 1)
            cv2.circle(preview, tuple(center.astype(int)), 3, (0, 255, 0), -1)
        (args.output / "images" / image_path.name).write_bytes(image_path.read_bytes())
        cv2.imwrite(str(args.output / "previews" / image_path.name),
                    cv2.addWeighted(frame, 0.5, preview, 0.5, 0))
        manifest["images"].append({"image": f"images/{image_path.name}", "split": "train",
                                   "instances": instances, "reviewed": False})
        print(f"{image_path.name}: {len(instances)} proposal(s)", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
