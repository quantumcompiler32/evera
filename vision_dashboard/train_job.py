"""Fine-tune a dashboard dataset without replacing the live checkpoint."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Training from {args.weights} on {args.data}", flush=True)
    model = YOLO(str(args.weights))
    model.train(data=str(args.data), imgsz=640, epochs=args.epochs, batch=8,
                device=0, workers=2, project=str(args.output.parent), name=args.output.name,
                exist_ok=False)
    best = Path(model.trainer.best)
    if not best.exists():
        raise RuntimeError(f"Training did not produce {best}")
    exported = YOLO(str(best)).export(format="onnx", imgsz=640, opset=17,
                                     nms=False, simplify=False)
    print(f"Provisional PyTorch model: {best}", flush=True)
    print(f"Provisional ONNX model: {exported}", flush=True)
    (args.output.parent / "model_paths.txt").write_text(f"{best}\n{exported}\n")


if __name__ == "__main__":
    main()
