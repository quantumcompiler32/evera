"""Find what makes ACT training slow: GPU compute vs. video data loading.

    python bench.py gpu        # ACT forward+backward on fake data
    python bench.py data 16    # dataloader alone, with N workers
"""
import sys
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else "mps"
REPO = "lerobot/svla_so101_pickplace"


def sync():
    torch.cuda.synchronize() if DEV == "cuda" else torch.mps.synchronize()


def bench_gpu(batch=64, steps=10):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType

    meta = LeRobotDatasetMetadata(REPO)
    feats = dataset_to_policy_features(meta.features)
    cfg = ACTConfig(
        input_features={k: f for k, f in feats.items() if f.type is not FeatureType.ACTION},
        output_features={k: f for k, f in feats.items() if f.type is FeatureType.ACTION},
        device=DEV,
    )
    policy = ACTPolicy(cfg).to(DEV).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    b = {}
    for k, f in cfg.input_features.items():
        b[k] = torch.rand(batch, *f.shape, device=DEV)
    b["action"] = torch.rand(batch, cfg.chunk_size, cfg.output_features["action"].shape[0], device=DEV)
    b["action_is_pad"] = torch.zeros(batch, cfg.chunk_size, dtype=torch.bool, device=DEV)

    for i in range(steps):
        sync()
        t = time.time()
        loss, _ = policy.forward(b)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync()
        print(f"gpu step {i}: {time.time() - t:.2f}s", flush=True)


def bench_data(workers=16, batches=10, batch=64):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO, delta_timestamps={"action": [i / 30 for i in range(100)]})
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers)
    t = time.time()
    for i, _ in enumerate(dl):
        print(f"data batch {i}: {time.time() - t:.2f}s", flush=True)
        t = time.time()
        if i + 1 >= batches:
            break


if __name__ == "__main__":
    print("torch", torch.__version__, "| gpu:", torch.cuda.get_device_name(0) if DEV == "cuda" else DEV, flush=True)
    if sys.argv[1] == "gpu":
        bench_gpu()
    else:
        bench_data(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
