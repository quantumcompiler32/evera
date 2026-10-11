"""Fine-tune YOLO11n-seg on reviewed masks or a clearly provisional proposal set."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO segmentation data.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs and batch must be positive; workers cannot be negative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output already contains files: {args.output}")
    report = args.data.parent / "conversion_report.json"
    if report.exists():
        status = json.loads(report.read_text())
        if status.get("unreviewed_validation"):
            print("PROVISIONAL: validation masks are unreviewed proposals; metrics do not measure real accuracy", flush=True)

    from ultralytics import YOLO

    model = YOLO("yolo11n-seg.pt")
    model.train(data=str(args.data.resolve()), imgsz=640, epochs=args.epochs,
                batch=args.batch, device=args.device, workers=args.workers,
                project=str(args.output.parent.resolve()), name=args.output.name,
                exist_ok=True)


if __name__ == "__main__":
    main()
