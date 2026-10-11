"""Smoke-check ONNX mask output on five saved camera frames (not an accuracy test)."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from onnx_segmenter import CLASS_NAMES, InstanceMask, Segmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    args = parser.parse_args()
    paths = [path for path in sorted(args.frames.glob("*.jpg")) if path.name != "contact-sheet.jpg"][:5]
    if len(paths) != 5:
        parser.error("Need at least five saved camera frames")
    segmenter = Segmenter(str(args.model))
    checked = 0
    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Cannot read {path}")
        if index == 1:
            bgr = cv2.resize(bgr, (320, 240))
        elif index == 2:
            bgr = cv2.resize(bgr, (1280, 960))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for item in segmenter.segment(rgb):
            assert isinstance(item, InstanceMask)
            assert item.cls in CLASS_NAMES
            assert 0 <= item.conf <= 1
            x1, y1, x2, y2 = item.box
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
            assert item.mask.shape == rgb.shape[:2] and item.mask.dtype == np.bool_
            assert item.mask.any()
            checked += 1
    assert checked > 0
    print(f"Validated instance-mask types and bounds on five frames ({checked} predictions)")


if __name__ == "__main__":
    main()
