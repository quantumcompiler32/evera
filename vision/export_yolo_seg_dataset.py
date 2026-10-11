"""Convert reviewed binary instance masks to Ultralytics segmentation polygons.

Unreviewed SAM proposals are excluded unless --allow-unreviewed is provided.
That flag is for a provisional visual baseline, never a quality evaluation.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from class_map import SUPPORTED_CLASS_ORDERS


def mask_polygon(mask: np.ndarray) -> tuple[list[float] | None, float]:
    count, components, areas, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count < 2:
        return None, 0.0
    best = 1 + int(np.argmax(areas[1:, cv2.CC_STAT_AREA]))
    largest = (components == best).astype(np.uint8)
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None, 0.0
    height, width = mask.shape
    points = contour[:, 0].astype(np.float32)
    points[:, 0] /= width
    points[:, 1] /= height
    coverage = float(largest.sum() / max(mask.sum(), 1))
    return points.clip(0, 1).flatten().tolist(), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--camera-repeat", type=int, default=1,
                        help="Repeat camera training images to balance public wire examples")
    parser.add_argument("--min-coverage", type=float, default=0.9,
                        help="Skip images if a disconnected instance cannot fit one YOLO polygon")
    args = parser.parse_args()
    if args.camera_repeat < 1:
        parser.error("camera-repeat must be positive")
    if not 0 < args.min_coverage <= 1:
        parser.error("min-coverage must be in (0, 1]")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output already contains files: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text())
    classes = tuple(manifest["classes"])
    if classes not in SUPPORTED_CLASS_ORDERS:
        parser.error("Dataset classes do not match a supported append-only class order")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    paths = {"train": [], "val": [], "test": []}
    counts = Counter()
    warnings = []
    unreviewed_validation = False
    for index, item in enumerate(manifest["images"]):
        split = item["split"]
        if split not in paths:
            parser.error(f"Unknown split: {split}")
        reviewed = item.get("reviewed", True)
        if not reviewed and not args.allow_unreviewed:
            continue
        if not reviewed and split != "train":
            unreviewed_validation = True
        image = dataset / item["image"]
        source = cv2.imread(str(image))
        if source is None:
            raise ValueError(f"Cannot read image: {image}")
        height, width = source.shape[:2]
        stem = f"{index:05d}"
        dest_image = output / "images" / split / f"{stem}{image.suffix.lower()}"
        labels = []
        item_counts = Counter()
        skip_image = False
        for number, instance in enumerate(item["instances"]):
            if instance["class"] not in classes:
                raise ValueError(f"Unknown class: {instance['class']}")
            mask = cv2.imread(str(dataset / instance["mask"]), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"Mask missing or wrong size: {instance['mask']}")
            polygon, coverage = mask_polygon(mask > 127)
            if polygon is None:
                warnings.append(f"{item['image']} instance {number}: no usable polygon")
                skip_image = True
                break
            if coverage < args.min_coverage:
                warnings.append(f"{item['image']} instance {number}: skipped image; polygon covers {coverage:.1%} of mask")
                skip_image = True
                break
            class_id = classes.index(instance["class"])
            labels.append(" ".join([str(class_id)] + [f"{coordinate:.6f}" for coordinate in polygon]))
            item_counts[f"{split}:{instance['class']}"] += 1
        if skip_image:
            continue
        shutil.copy2(image, dest_image)
        counts.update(item_counts)
        (output / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        repeats = args.camera_repeat if split == "train" and item["image"].startswith("camera/") else 1
        paths[split].extend([str(dest_image)] * repeats)

    if not paths["train"] or not paths["val"]:
        raise ValueError("Need train and val images; review masks or use --allow-unreviewed for a provisional baseline")
    for split, names in paths.items():
        (output / f"{split}.txt").write_text("\n".join(names) + ("\n" if names else ""))
    lines = [f"path: {json.dumps(str(output))}", "train: train.txt", "val: val.txt", "names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(classes)]
    (output / "data.yaml").write_text("\n".join(lines) + "\n")
    (output / "conversion_report.json").write_text(json.dumps({
        "instances": dict(counts), "warnings": warnings,
        "unreviewed_validation": unreviewed_validation,
        "provisional": bool(args.allow_unreviewed),
    }, indent=2) + "\n")
    print(f"Converted {sum(len(v) for v in paths.values())} training/evaluation entries")
    print(f"Polygon warnings: {len(warnings)}; unreviewed validation: {unreviewed_validation}")


if __name__ == "__main__":
    main()
