"""Pretrained convolutional Mask R-CNN and live mask visualization."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from segmentation_dataset import CLASS_NAMES


COLORS = ((0, 0, 0), (30, 230, 90), (255, 175, 40), (40, 150, 255), (235, 70, 220))


def make_model(pretrained: bool = False):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=640,
        max_size=960,
        box_detections_per_img=30,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, len(CLASS_NAMES))
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, len(CLASS_NAMES))
    return model


def load_model(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if tuple(payload["classes"]) != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match this code")
    model = make_model(pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_frame(model, frame: np.ndarray, device: str, confidence: float = 0.5):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).to(device)
    output = model([image])[0]
    results = []
    for label, score, mask in zip(output["labels"], output["scores"], output["masks"]):
        if float(score) < confidence:
            continue
        class_id = int(label)
        if class_id <= 0 or class_id >= len(CLASS_NAMES):
            continue
        binary = mask[0].detach().cpu().numpy() >= 0.5
        if binary.sum() < 4:
            continue
        results.append((class_id, float(score), binary))
    return results


def draw_masks(frame: np.ndarray, results):
    overlay = frame.copy()
    for class_id, _, mask in results:
        overlay[mask] = COLORS[class_id]
    painted = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
    for class_id, score, mask in results:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(painted, contours, -1, COLORS[class_id], 2)
        ys, xs = np.where(mask)
        text = f"{CLASS_NAMES[class_id]} {score:.2f}"
        cv2.putText(painted, text, (int(xs.min()), max(20, int(ys.min()) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[class_id], 2)
    return painted
