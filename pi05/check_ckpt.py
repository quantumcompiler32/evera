#!/usr/bin/env python
"""Offline sanity check of a pi0.5 checkpoint against recorded demos, before it touches the arm.

Loads the checkpoint the same way lerobot-rollout does (policy + saved pre/post processors), feeds it
recorded frames, and compares the predicted action chunk with what the teleoperator actually did.

    python check_ckpt.py ~/pi05/outputs/<run>/checkpoints/last/pretrained_model --dataset so101_pick_place
"""

import argparse
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--dataset", default="test_run")
    ap.add_argument("--samples", type=int, default=8, help="frames to test, spread across the dataset")
    args = ap.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = "cuda"
    policy = PI05Policy.from_pretrained(args.ckpt, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.ckpt))

    horizon = cfg.n_action_steps
    ds = LeRobotDataset(
        f"Neevdharod/{args.dataset}",
        root=Path.home() / "hackathon/data" / args.dataset,
        delta_timestamps={"action": [i / 30 for i in range(horizon)]},
    )
    idxs = torch.linspace(0, len(ds) - 1, args.samples).long().tolist()

    errs = []
    for i in idxs:
        frame = ds[i]
        obs = {k: v for k, v in frame.items() if k.startswith("observation.")}
        obs["task"] = frame["task"]
        with torch.inference_mode():
            pred = post(policy.predict_action_chunk(pre(obs))).squeeze(0).float().cpu()
        true = frame["action"][: pred.shape[0]]
        # action_is_pad marks steps past the episode end; don't score those
        valid = ~frame["action_is_pad"][: pred.shape[0]]
        errs.append((pred - true)[valid].abs().mean(0))
        print(f"frame {i:5d} | first predicted step {[round(x, 1) for x in pred[0].tolist()]}")

    mae = torch.stack(errs).mean(0)
    print(f"\nmean |pred - teleop| over {horizon}-step chunks (joint units as recorded, degrees / gripper %):")
    for name, e in zip(JOINTS, mae.tolist(), strict=True):
        print(f"  {name:14s} {e:6.2f}")


if __name__ == "__main__":
    main()
