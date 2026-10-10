"""Train a workshop object detector on the machine that holds the labeled images."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Detection dataset YAML")
    parser.add_argument("--model", required=True, help="Pretrained detection weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu or 0")
    parser.add_argument("--project", type=Path, default=Path("outputs/detection"))
    parser.add_argument("--name", default="workshop")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset YAML does not exist: {args.data}")

    model = YOLO(args.model)
    train_args = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": str(args.project.resolve()),
        "name": args.name,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished without expected weights: {best}")

    trained = YOLO(str(best))
    metadata = {
        "class_names": trained.names,
        "image_size": args.imgsz,
        "training_weights": best.name,
    }
    metadata_path = best.parent / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Best weights: {best}")
    print(f"Model metadata: {metadata_path}")


if __name__ == "__main__":
    main()
