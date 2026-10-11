"""Import a small, attributed, per-instance wire subset from WireSeg-36K.

Uses the `wire_on_plane` shard for tabletop scenes. Each RLE is one cable
instance, so crossing wires remain separate training targets.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation_dataset import CLASS_NAMES


DATASET = "DeformX/WireSeg-36K"
SHARD = "hf://datasets/DeformX/WireSeg-36K@main/data/train-00030-of-00045.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    from datasets import load_dataset
    from pycocotools import mask as coco_mask

    root = args.output
    for part in ("images", "masks"):
        (root / part).mkdir(parents=True, exist_ok=True)
    manifest = {"classes": list(CLASS_NAMES[1:]), "images": [], "sources": [{
        "name": DATASET, "url": "https://huggingface.co/datasets/DeformX/WireSeg-36K",
        "license": "CC-BY-4.0", "shard": SHARD,
    }]}
    stream = load_dataset("parquet", data_files={"train": SHARD}, split="train", streaming=True)
    for index, row in enumerate(stream):
        if len(manifest["images"]) >= args.limit:
            break
        if row["scene_family"] != "wire_on_plane":
            continue
        stem = f"wireseg-{index:05d}"
        image_file = f"images/{stem}.png"
        row["image"].convert("RGB").save(root / image_file)
        instances = []
        for wire_index, rle in enumerate(json.loads(row["masks_rle"])):
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
            mask = coco_mask.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.sum() < 4:
                continue
            mask_file = f"masks/{stem}-{wire_index:02d}.png"
            Image.fromarray((mask.astype(np.uint8) * 255)).save(root / mask_file)
            instances.append({"class": "wire", "mask": mask_file})
        if instances:
            manifest["images"].append({"image": image_file, "split": "train", "instances": instances})
        if len(manifest["images"]) % 25 == 0:
            print(f"Imported {len(manifest['images'])} images", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Imported {len(manifest['images'])} images with individual wire masks", flush=True)
    os._exit(0)  # Work around pyarrow/HF streaming shutdown crash on the ROCm host.


if __name__ == "__main__":
    main()
