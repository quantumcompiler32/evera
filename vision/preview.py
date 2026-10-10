"""Preview text-prompted bounding boxes before a custom detector is trained."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image path or camera index")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--classes", nargs="+", required=True, help="Object descriptions")
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--model", default="yolov8s-world.pt")
    args = parser.parse_args()

    image_path = Path(args.source)
    if image_path.is_file():
        frame = cv2.imread(str(image_path))
    elif args.source.isdecimal():
        camera = cv2.VideoCapture(int(args.source))
        try:
            for _ in range(3):
                camera.read()
            ok, frame = camera.read()
            if not ok:
                parser.error(f"Could not read camera: {args.source}")
        finally:
            camera.release()
    else:
        parser.error("--source must be an existing image or a camera index")
    if frame is None:
        parser.error(f"Could not read image: {args.source}")

    model = YOLOE(args.model) if Path(args.model).name.startswith("yoloe-") else YOLO(args.model)
    model.set_classes(args.classes)
    result = model.predict(frame, conf=args.confidence, verbose=False)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result.plot(masks=False)):
        raise RuntimeError(f"Could not save annotated image: {args.output}")
    print(f"Saved {len(result.boxes)} provisional boxes to {args.output}")
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        print(f"{name}: {box.conf.item():.3f} {box.xyxy[0].tolist()}")


if __name__ == "__main__":
    main()
