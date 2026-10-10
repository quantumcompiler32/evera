"""Validate captured image boxes and build a YOLO detection dataset."""

import argparse
import json
import random
import shutil
from pathlib import Path


CLASS_NAMES = ("raspberry_pi", "arduino", "soldering_iron")


def check_labels(path: Path) -> set[int]:
    classes = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: expected class and four coordinates")
        cls = int(parts[0])
        xc, yc, width, height = map(float, parts[1:])
        if cls not in range(len(CLASS_NAMES)) or not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{number}: invalid class or box size")
        if not (0 <= xc - width / 2 < xc + width / 2 <= 1):
            raise ValueError(f"{path}:{number}: box exceeds image width")
        if not (0 <= yc - height / 2 < yc + height / 2 <= 1):
            raise ValueError(f"{path}:{number}: box exceeds image height")
        classes.add(cls)
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("captures/training_candidates"))
    parser.add_argument("--labels", type=Path, default=Path("captures/training_labels"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists; choose a new version: {args.output}")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")

    samples = []
    missing = []
    for image in sorted(args.captures.rglob("*.jpg")):
        relative = image.relative_to(args.captures)
        label = args.labels / relative.with_suffix(".txt")
        if not label.is_file():
            missing.append(str(relative))
            continue
        classes = check_labels(label)
        samples.append((image, label, relative, classes))
    if missing:
        parser.error("Unlabeled images: " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    if len(samples) < 6:
        parser.error("Capture and label at least six distinct images before training")
    seen = set().union(*(sample[3] for sample in samples))
    if seen != set(range(len(CLASS_NAMES))):
        parser.error("Raspberry Pi, Arduino, and soldering iron must each have at least one box")

    random.Random(42).shuffle(samples)
    val_count = max(2, round(len(samples) * args.validation_fraction))
    groups = {"val": samples[:val_count], "train": samples[val_count:]}
    for split, group in groups.items():
        for image, label, relative, _ in group:
            name = relative.parent.name + "__" + relative.name
            image_out = args.output / "images" / split / name
            label_out = args.output / "labels" / split / Path(name).with_suffix(".txt")
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_out)
            shutil.copy2(label, label_out)

    yaml_path = args.output / "dataset.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(str(args.output.resolve()))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    for split, group in groups.items():
        counts = {name: sum(i in sample[3] for sample in group) for i, name in enumerate(CLASS_NAMES)}
        print(f"{split}: {len(group)} images, class presence {counts}")
    print(f"Dataset: {yaml_path}")


if __name__ == "__main__":
    main()
