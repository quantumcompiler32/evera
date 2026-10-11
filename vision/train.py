"""Fine-tune a convolutional Mask R-CNN for workshop instance segmentation."""

import argparse
import json
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from segmentation_dataset import CLASS_NAMES, InstanceMaskDataset, collate
from segmentation_model import make_model


def mask_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    intersection = (first & second).sum().item()
    union = (first | second).sum().item()
    return intersection / union if union else 0.0


@torch.inference_mode()
def evaluate(model, loader, device: str, threshold: float = 0.5):
    model.eval()
    truth = {class_id: 0 for class_id in range(1, len(CLASS_NAMES))}
    guesses = {class_id: [] for class_id in truth}
    image_index = 0
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        for prediction, target in zip(predictions, targets):
            gt_masks = target["masks"].bool()
            gt_labels = target["labels"]
            for class_id in truth:
                matches = (gt_labels == class_id).nonzero(as_tuple=True)[0].tolist()
                truth[class_id] += len(matches)
                for index in (prediction["labels"] == class_id).nonzero(as_tuple=True)[0].tolist():
                    score = float(prediction["scores"][index])
                    if score < 0.05:
                        continue
                    mask = prediction["masks"][index, 0].cpu() >= threshold
                    overlaps = [(gt_index, mask_iou(mask, gt_masks[gt_index])) for gt_index in matches]
                    best_gt, best_iou = max(overlaps, key=lambda pair: pair[1]) if overlaps else (-1, 0.0)
                    guesses[class_id].append((score, image_index, best_gt, best_iou))
            image_index += 1

    metrics = {}
    for class_id, name in enumerate(CLASS_NAMES[1:], 1):
        found = set()
        tp, fp = [], []
        for score, image_id, gt_id, iou in sorted(guesses[class_id], reverse=True):
            key = (image_id, gt_id)
            correct = iou >= 0.5 and key not in found
            tp.append(int(correct))
            fp.append(int(not correct))
            if correct:
                found.add(key)
        if truth[class_id]:
            recall = np.cumsum(tp) / truth[class_id]
            precision = np.cumsum(tp) / np.maximum(1, np.cumsum(tp) + np.cumsum(fp))
            ap50 = float(np.mean([max(precision[recall >= point], default=0.0) for point in np.linspace(0, 1, 101)]))
        else:
            ap50 = None
        metrics[name] = {"mask_ap50": ap50, "ground_truth_instances": truth[class_id], "predictions": len(tp)}
    valid = [item["mask_ap50"] for item in metrics.values() if item["mask_ap50"] is not None]
    metrics["mean_mask_ap50"] = float(np.mean(valid)) if valid else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder containing manifest.json and binary instance masks")
    parser.add_argument("--output", type=Path, default=Path("outputs/segmentation/workshop"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Limit batches for a quick provisional model")
    parser.add_argument("--head-only", action="store_true",
                        help="Freeze the pretrained backbone and proposal network")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or (args.steps_per_epoch is not None and args.steps_per_epoch < 1):
        parser.error("epochs, batch size, and steps per epoch must be positive; workers cannot be negative")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    train_set = InstanceMaskDataset(args.dataset, "train", augment=True)
    val_set = InstanceMaskDataset(args.dataset, "val")
    # Public wire images greatly outnumber the camera scenes. Sample both
    # groups evenly so a wire-only majority does not erase board learning.
    has_wire = [any(instance["class"] == "wire" for instance in item["instances"])
                for item in train_set.items]
    groups = [sum(has_wire), len(has_wire) - sum(has_wire)]
    weights = [1 / groups[0 if wire else 1] if groups[0] and groups[1] else 1
               for wire in has_wire]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    observed = {instance["class"] for item in train_set.items for instance in item["instances"]}
    unknown = observed - set(CLASS_NAMES[1:])
    if unknown:
        parser.error("Training annotations contain unknown classes: " + ", ".join(sorted(unknown)))
    missing = set(CLASS_NAMES[1:]) - observed
    if missing:
        parser.error("Training split is missing classes: " + ", ".join(sorted(missing)))

    model = make_model(pretrained=True).to(args.device)
    if args.head_only:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in model.rpn.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.0001)
    args.output.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        batches = islice(train_loader, args.steps_per_epoch) if args.steps_per_epoch else train_loader
        for step, (images, targets) in enumerate(batches, 1):
            images = [image.to(args.device) for image in images]
            targets = [{key: value.to(args.device) for key, value in target.items()} for target in targets]
            components = model(images, targets)
            loss = sum(components.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 5 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": float(np.mean(losses[-5:]))}), flush=True)
        metrics = evaluate(model, val_loader, args.device)
        metrics["epoch"] = epoch
        metrics["training_loss"] = float(np.mean(losses))
        metrics["validation_masks_reviewed"] = all(item.get("reviewed", True) for item in val_set.items)
        print(json.dumps(metrics), flush=True)
        if metrics["mean_mask_ap50"] is not None and metrics["mean_mask_ap50"] > best_ap:
            best_ap = metrics["mean_mask_ap50"]
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "epoch": epoch, "validation": metrics}, args.output / "best.pt")
            (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Best checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
