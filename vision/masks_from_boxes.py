"""Create reviewable instance-mask candidates from existing YOLO boxes using SAM.

The generated masks are proposals, not ground truth. Inspect and correct them
before using them for training or reporting evaluation results.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from segmentation_dataset import CLASS_NAMES


BOX_CLASSES = ("raspberry_pi", "arduino", "pinecil")


def yolo_box(line: str, width: int, height: int) -> tuple[str, np.ndarray]:
    class_id, cx, cy, w, h = map(float, line.split())
    if not class_id.is_integer() or int(class_id) not in range(len(BOX_CLASSES)):
        raise ValueError(f"Unknown YOLO class in {line!r}")
    box = np.array([(cx - w / 2) * width, (cy - h / 2) * height,
                    (cx + w / 2) * width, (cy + h / 2) * height], dtype=np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    return BOX_CLASSES[int(class_id)], box


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Folder with images/ and labels/ YOLO pairs")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official SAM ViT-B checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(args.device)
    predictor = SamPredictor(sam)
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    masks_dir = args.output / "masks"
    previews_dir = args.output / "previews"
    for folder in (images_dir, masks_dir, previews_dir):
        folder.mkdir(exist_ok=True)
    manifest = {"classes": list(CLASS_NAMES[1:]), "images": []}
    colors = {"raspberry_pi": (60, 240, 70), "arduino": (250, 170, 30), "pinecil": (20, 130, 255)}
    for image_path in sorted((args.source / "images").glob("*.jpg")):
        label_path = args.source / "labels" / (image_path.stem + ".txt")
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Cannot read {image_path}")
        height, width = bgr.shape[:2]
        predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        instances = []
        preview = bgr.copy()
        for index, line in enumerate(label_path.read_text().splitlines()):
            if not line.strip():
                continue
            name, box = yolo_box(line, width, height)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
            mask = masks[int(np.argmax(scores))]
            if mask.sum() < 4:
                continue
            relative_mask = f"masks/{image_path.stem}-{index:02d}.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(args.output / relative_mask)
            instances.append({"class": name, "mask": relative_mask, "source": "sam_box_proposal"})
            preview[mask] = np.array(colors[name], dtype=np.uint8)
            cv2.putText(preview, name, (int(box[0]), max(16, int(box[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)
        (images_dir / image_path.name).write_bytes(image_path.read_bytes())
        cv2.imwrite(str(previews_dir / image_path.name), cv2.addWeighted(bgr, 0.5, preview, 0.5, 0))
        manifest["images"].append({"image": f"images/{image_path.name}",
                                   "split": "val" if image_path.stem == "scene07" else "train",
                                   "instances": instances, "reviewed": False})
        print(f"{image_path.name}: {len(instances)} proposed masks", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All masks are unreviewed proposals; do not train or score them as ground truth.")


if __name__ == "__main__":
    main()
