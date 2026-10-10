"""Draw labeled bounding boxes on a still image or live camera feed."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from board_refine import refine_board_classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trained best.pt")
    parser.add_argument("--source", required=True, help="Image path, camera index, or HTTP/RTSP stream URL")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--output", type=Path, help="Save an annotated image; for a camera, capture one frame")
    parser.add_argument("--board-color-refine", action="store_true", help="Apply the workshop overhead-camera green Pi / blue Arduino correction")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Model weights do not exist: {args.model}")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")

    model = YOLO(str(args.model))
    image_path = Path(args.source)
    if image_path.is_file():
        image = cv2.imread(str(image_path))
        if image is None:
            parser.error(f"Could not read image: {image_path}")
        result = model.predict(image, conf=args.confidence, iou=0.5, agnostic_nms=True, verbose=False)[0]
        if args.board_color_refine:
            refine_board_classes(image, result)
        output = args.output or image_path.with_name(f"{image_path.stem}_boxes.jpg")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result.plot()):
            raise RuntimeError(f"Could not save annotated image: {output}")
        print(f"Annotated image: {output}")
        return

    is_url = args.source.startswith(("http://", "https://", "rtsp://"))
    if not args.source.isdecimal() and not is_url:
        parser.error("--source must be an existing image, camera index, or HTTP/RTSP stream URL")
    camera = cv2.VideoCapture(args.source if is_url else int(args.source), cv2.CAP_FFMPEG if is_url else cv2.CAP_ANY)
    if not camera.isOpened():
        parser.error(f"Could not open camera: {args.source}")
    try:
        if args.output is not None:
            for _ in range(3):
                camera.read()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f"Could not read camera: {args.source}")
            result = model.predict(frame, conf=args.confidence, iou=0.5, agnostic_nms=True, verbose=False)[0]
            if args.board_color_refine:
                refine_board_classes(frame, result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), result.plot()):
                raise RuntimeError(f"Could not save annotated image: {args.output}")
            print(f"Annotated camera frame: {args.output}")
            return
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            result = model.predict(frame, conf=args.confidence, iou=0.5, agnostic_nms=True, verbose=False)[0]
            if args.board_color_refine:
                refine_board_classes(frame, result)
            cv2.imshow("Workshop object detection (q to quit)", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
