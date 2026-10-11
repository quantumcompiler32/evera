"""Combine reviewed camera mask proposals and licensed public wire masks.

Source manifests and original files are left intact. Camera proposal masks
remain explicitly tagged as proposals until individually corrected.
"""

import argparse
import json
import shutil
from pathlib import Path

from segmentation_dataset import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--wires", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = {"classes": list(CLASS_NAMES[1:]), "images": [], "sources": []}
    for name, root in (("camera", args.camera), ("wires", args.wires)):
        source = json.loads((root / "manifest.json").read_text())
        if source["classes"] != combined["classes"]:
            raise ValueError(f"Class mismatch: {root}")
        combined["sources"].extend(source.get("sources", []))
        for item in source["images"]:
            updated = dict(item)
            updated["image"] = f"{name}/{item['image']}"
            paths = [item["image"]] + [instance["mask"] for instance in item["instances"]]
            for rel in paths:
                target = args.output / name / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, target)
            updated["instances"] = [{**instance, "mask": f"{name}/{instance['mask']}"}
                                    for instance in item["instances"]]
            combined["images"].append(updated)
    (args.output / "manifest.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(f"Combined {len(combined['images'])} images in {args.output}")


if __name__ == "__main__":
    main()
