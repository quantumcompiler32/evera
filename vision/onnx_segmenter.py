"""CPU-only YOLO11 instance masks from ONNX; no camera or training dependencies."""

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
try:
    from .class_map import LEGACY_CLASSES, SUPPORTED_CLASS_ORDERS
except ImportError:  # direct script execution from vision/
    from class_map import LEGACY_CLASSES, SUPPORTED_CLASS_ORDERS


CLASS_NAMES = LEGACY_CLASSES
INPUT_SIZE = 640


@dataclass(frozen=True)
class InstanceMask:
    cls: str
    conf: float
    box: tuple[float, float, float, float]
    mask: np.ndarray  # HxW bool array in the input frame's coordinates


class Segmenter:
    def __init__(self, model_path: str, conf_threshold: float = 0.4,
                 iou_threshold: float = 0.5, threads: int = 2):
        if not 0 < conf_threshold < 1 or not 0 < iou_threshold < 1:
            raise ValueError("Thresholds must be between 0 and 1")
        if threads < 1:
            raise ValueError("threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(Path(model_path)), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
            raise ValueError("Expected a static 640x640 YOLO11 segmentation ONNX input")
        if len(outputs) != 2 or outputs[1].shape != [1, 32, 160, 160]:
            raise ValueError("ONNX outputs do not match YOLO11n-seg mask prototypes")
        metadata = self.session.get_modelmeta().custom_metadata_map
        names = literal_eval(metadata.get("names", "{}"))
        class_names = tuple(names.get(index) for index in range(len(names)))
        if class_names not in SUPPORTED_CLASS_ORDERS:
            raise ValueError("ONNX class names or order do not match Evera's mask classes")
        if outputs[0].shape != [1, 36 + len(class_names), 8400]:
            raise ValueError("ONNX prediction channels do not match the class map")
        self.class_names = class_names
        self.input_name = inputs[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def segment(self, frame_rgb: np.ndarray) -> list[InstanceMask]:
        if (not isinstance(frame_rgb, np.ndarray) or frame_rgb.dtype != np.uint8
                or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3):
            raise ValueError("frame_rgb must be an HxWx3 uint8 RGB array")
        height, width = frame_rgb.shape[:2]
        gain = min(INPUT_SIZE / width, INPUT_SIZE / height)
        new_width = round(width * gain)
        new_height = round(height * gain)
        pad_x = (INPUT_SIZE - new_width) // 2
        pad_y = (INPUT_SIZE - new_height) // 2
        resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        letterboxed[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        tensor = np.ascontiguousarray(letterboxed.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        prediction, prototype = self.session.run(None, {self.input_name: tensor})
        prediction = prediction[0].T
        mask_offset = 4 + len(self.class_names)
        scores = prediction[:, 4:mask_offset]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        candidates = np.flatnonzero(confidences >= self.conf_threshold)
        if not len(candidates):
            return []

        boxes = prediction[candidates, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        keep = []
        for class_id in range(len(self.class_names)):
            group = np.flatnonzero(class_ids[candidates] == class_id)
            if not len(group):
                continue
            xywh = np.column_stack((boxes[group, :2], boxes[group, 2:] - boxes[group, :2]))
            selected = cv2.dnn.NMSBoxes(xywh.tolist(), confidences[candidates[group]].tolist(),
                                        self.conf_threshold, self.iou_threshold)
            keep.extend(group[np.asarray(selected).reshape(-1)].tolist())
        keep.sort(key=lambda index: float(confidences[candidates[index]]), reverse=True)
        # The two boards can produce near-identical, overlapping predictions.
        # Keep the higher-scoring board when both classes claim the same region.
        distinct = []
        for kept in keep:
            current = candidates[kept]
            current_class = int(class_ids[current])
            if current_class in (0, 1):
                duplicate_board = False
                for accepted in distinct:
                    other = candidates[accepted]
                    if int(class_ids[other]) not in (0, 1):
                        continue
                    a, b = boxes[kept], boxes[accepted]
                    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                        0.0, min(a[3], b[3]) - max(a[1], b[1]))
                    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    if overlap / max(area_a + area_b - overlap, 1e-6) >= 0.8:
                        duplicate_board = True
                        break
                if duplicate_board:
                    continue
            distinct.append(kept)
        proto = prototype[0].reshape(32, -1)
        results = []
        for kept in distinct[:30]:
            index = candidates[kept]
            box = boxes[kept].copy()
            box[[0, 2]] = (box[[0, 2]] - pad_x) / gain
            box[[1, 3]] = (box[[1, 3]] - pad_y) / gain
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            logits = (prediction[index, mask_offset:] @ proto).reshape(160, 160)
            logits = cv2.resize(logits, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            logits = logits[pad_y:pad_y + new_height, pad_x:pad_x + new_width]
            logits = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR)
            binary = logits > 0
            bounds = np.zeros((height, width), dtype=np.uint8)
            bounds[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 1
            binary &= bounds.astype(bool)
            if binary.sum() < 4:
                continue
            results.append(InstanceMask(
                self.class_names[int(class_ids[index])], float(confidences[index]),
                (float(x1 / width), float(y1 / height), float(x2 / width), float(y2 / height)),
                binary,
            ))
        return results
