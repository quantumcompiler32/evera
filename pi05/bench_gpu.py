#!/usr/bin/env python
"""Synthetic pi0.5 benchmark for the GMKtec EVO-X2 (Ryzen AI Max+ 395 / Radeon 8060S, gfx1151).

Loads lerobot/pi05_base with our SO-101 feature layout (scene + wrist cams at 320x240, 6-D
state/action) and feeds random inputs, so it needs neither the gated PaliGemma tokenizer nor a
recorded dataset. Use it to pick a training config that fits in the iGPU's memory and to check
whether action-chunk latency is low enough for synchronous rollout.

    python bench_gpu.py infer
    python bench_gpu.py train --mode expert_only --batch-sizes 8 16 32
"""

import argparse
import statistics
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

PALIGEMMA_VOCAB = 257_152
CAMERAS = ("scene", "wrist")
IMG_H, IMG_W = 240, 320
STATE_DIM = ACTION_DIM = 6  # SO-101: 5 joints + gripper
FPS = 30

# Which parts of the 3.3B-param model receive gradients.
MODES = {
    "full": dict(freeze_vision_encoder=False, train_expert_only=False),
    "freeze_vision": dict(freeze_vision_encoder=True, train_expert_only=False),
    "expert_only": dict(freeze_vision_encoder=True, train_expert_only=True),
}


def make_policy(mode: str, grad_ckpt: bool, chunk_size: int, tokenizer_max_length: int) -> PI05Policy:
    cfg = PI05Config(
        dtype=torch.bfloat16,
        device="cuda",
        gradient_checkpointing=grad_ckpt,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        tokenizer_max_length=tokenizer_max_length,
        **MODES[mode],
    )
    cfg.input_features = {
        **{f"observation.images.{c}": PolicyFeature(FeatureType.VISUAL, (3, IMG_H, IMG_W)) for c in CAMERAS},
        "observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    return PI05Policy.from_pretrained("lerobot/pi05_base", config=cfg)


def make_batch(bsize: int, cfg: PI05Config, with_action: bool) -> dict[str, torch.Tensor]:
    dev = "cuda"
    n_tok = cfg.tokenizer_max_length
    batch = {f"observation.images.{c}": torch.rand(bsize, 3, IMG_H, IMG_W, device=dev) for c in CAMERAS}
    batch["observation.state"] = torch.randn(bsize, STATE_DIM, device=dev)
    batch["observation.language.tokens"] = torch.randint(0, PALIGEMMA_VOCAB, (bsize, n_tok), device=dev)
    batch["observation.language.attention_mask"] = torch.ones(bsize, n_tok, dtype=torch.bool, device=dev)
    if with_action:
        batch["action"] = torch.randn(bsize, cfg.chunk_size, ACTION_DIM, device=dev)
    return batch


def gib(n_bytes: int) -> float:
    return n_bytes / 2**30


def bench_train(policy: PI05Policy, batch_sizes: list[int], steps: int) -> None:
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"trainable params: {n_train / 1e9:.2f}B / {n_total / 1e9:.2f}B")
    opt = torch.optim.AdamW(params, lr=2.5e-5, betas=(0.9, 0.95), weight_decay=0.01)
    policy.train()

    print(f"{'batch':>5} | {'peak alloc':>10} | {'reserved':>9} | {'s/step':>7} | {'samples/s':>9}")
    for bsize in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            for i in range(steps + 2):  # first 2 steps are warmup (kernel selection, optimizer state alloc)
                batch = make_batch(bsize, policy.config, with_action=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss, _ = policy(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.perf_counter() - t0)
            dt = statistics.median(times)
            print(
                f"{bsize:>5} | {gib(torch.cuda.max_memory_allocated()):>8.1f}Gi | "
                f"{gib(torch.cuda.max_memory_reserved()):>7.1f}Gi | {dt:>7.2f} | {bsize / dt:>9.1f}"
            )
        except torch.OutOfMemoryError:
            print(f"{bsize:>5} | OOM")
            opt.zero_grad(set_to_none=True)
            break


@torch.inference_mode()
def bench_infer(policy: PI05Policy, iters: int) -> None:
    policy.eval()
    batch = make_batch(1, policy.config, with_action=False)
    for _ in range(3):
        policy.predict_action_chunk(batch)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ms = sorted(t * 1000 for t in times)
    chunk_s = policy.config.chunk_size / FPS
    print(f"action chunk ({policy.config.chunk_size} steps, {policy.config.num_inference_steps} flow steps):")
    print(f"  median {statistics.median(ms):.0f} ms | p90 {ms[int(0.9 * len(ms))]:.0f} ms | min {ms[0]:.0f} ms")
    print(f"  chunk covers {chunk_s:.2f}s of motion at {FPS} fps")
    print(f"  peak alloc {gib(torch.cuda.max_memory_allocated()):.1f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "infer"])
    ap.add_argument("--mode", choices=MODES, default="full")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    # Our task prompt + discretized state is ~42 PaliGemma tokens; pi05 pads to 200 by default.
    ap.add_argument("--tokenizer-max-length", type=int, default=200)
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__} | hip {torch.version.hip} | {props.name} ({props.gcnArchName})")
    print(f"device memory visible to torch: {gib(props.total_memory):.1f} GiB")

    t0 = time.perf_counter()
    policy = make_policy(args.mode, not args.no_grad_ckpt, args.chunk_size, args.tokenizer_max_length)
    print(
        f"loaded pi05_base in {time.perf_counter() - t0:.0f}s | mode={args.mode} | "
        f"tokenizer_max_length={args.tokenizer_max_length}"
    )

    if args.what == "train":
        bench_train(policy, args.batch_sizes, args.steps)
    else:
        bench_infer(policy, iters=20)


if __name__ == "__main__":
    main()
