"""Dataset format and loading for instance masks of workshop objects."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_tensor


CLASS_NAMES = ("__background__", "raspberry_pi", "arduino", "pinecil", "wire")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES) if index}


class InstanceMaskDataset(Dataset):
    """Read `manifest.json` with image paths and one binary PNG per object."""

    def __init__(self, root: Path, split: str, augment: bool = False, max_side: int = 640):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        if tuple(manifest["classes"]) != CLASS_NAMES[1:]:
            raise ValueError("Dataset classes do not match the model class order")
        self.items = [item for item in manifest["images"] if item["split"] == split]
        if not self.items:
            raise ValueError(f"No images in {split} split: {self.root}")
        self.augment = augment
        self.max_side = max_side
        self.jitter = ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(self.root / item["image"]).convert("RGB")
        width, height = image.size
        masks = []
        labels = []
        boxes = []
        for instance in item["instances"]:
            label = CLASS_IDS[instance["class"]]
            mask = np.asarray(Image.open(self.root / instance["mask"]).convert("L")) > 127
            if mask.shape != (height, width):
                raise ValueError(f"Mask size mismatch: {instance['mask']}")
            ys, xs = np.where(mask)
            if len(xs) < 4:
                raise ValueError(f"Empty or tiny mask: {instance['mask']}")
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            masks.append(mask)
            labels.append(label)

        if max(width, height) > self.max_side:
            scale = self.max_side / max(width, height)
            new_size = (round(width * scale), round(height * scale))
            image = image.resize(new_size, Image.Resampling.BILINEAR)
            masks = [np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(new_size, Image.Resampling.NEAREST)) > 127
                     for mask in masks]
            width, height = new_size
            boxes = []
            for mask in masks:
                ys, xs = np.where(mask)
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])

        pixels = to_tensor(image)
        mask_tensor = torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, height, width), dtype=torch.bool)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment:
            pixels = self.jitter(pixels)
            if random.random() < 0.5:
                pixels = pixels.flip(-1)
                mask_tensor = mask_tensor.flip(-1)
                if len(box_tensor):
                    old = box_tensor[:, [0, 2]].clone()
                    box_tensor[:, 0], box_tensor[:, 2] = width - old[:, 1], width - old[:, 0]
            if random.random() < 0.5:
                pixels = pixels.flip(-2)
                mask_tensor = mask_tensor.flip(-2)
                if len(box_tensor):
                    old = box_tensor[:, [1, 3]].clone()
                    box_tensor[:, 1], box_tensor[:, 3] = height - old[:, 1], height - old[:, 0]

        target = {
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": mask_tensor.to(torch.uint8),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": mask_tensor.flatten(1).sum(1).to(torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return pixels, target


def collate(batch):
    return tuple(zip(*batch))
