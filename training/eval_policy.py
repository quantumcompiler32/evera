"""Offline check of a trained ACT checkpoint: is it accurate, and does it actually use the camera?

    python training/eval_policy.py <checkpoint_dir> <dataset_root> --train 15-29 --heldout 30,31

1. Vision test: at frame 0 every episode starts from the same rest pose; only the Pinecil's position
   differs. If the model ignores the camera it predicts the same reach every time ("averaging").
   We compare its predicted reach (joint targets ~3 s ahead) against what the human actually did,
   and against the average-of-demos baseline. An image swap test feeds episode A's state with
   episode B's images: a vision-driven model follows B.
2. Held-out accuracy: along the held-out episodes, predict 1 s of actions every 0.5 s and compare
   with the recorded actions (degrees), vs. the average-demo baseline.
"""
import argparse

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def parse_eps(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("root")
p.add_argument("--train", default="15-29")
p.add_argument("--heldout", default="30,31")
p.add_argument("--repo", default="Neevdharod/so101_pick_place")
args = p.parse_args()
train_eps, held_eps = parse_eps(args.train), parse_eps(args.heldout)

policy = ACTPolicy.from_pretrained(args.ckpt).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})
ds = LeRobotDataset(args.repo, root=args.root, episodes=train_eps + held_eps)
H = policy.config.chunk_size
img_keys = list(policy.config.image_features)

ep_index = np.asarray(ds.hf_dataset.with_format("numpy")["episode_index"]).reshape(-1)
rows = {e: np.where(ep_index == e)[0] for e in train_eps + held_eps}
actions = np.stack(ds.hf_dataset.with_format("numpy")["action"])


def obs(i, img_from=None):
    item = ds[int(i)]
    src = ds[int(img_from)] if img_from is not None else item
    o = {"observation.state": item["observation.state"]}
    for k in img_keys:
        o[k] = src[k]
    return o


@torch.no_grad()
def chunk(o):
    batch = pre(o)
    a = policy.predict_action_chunk(batch)
    return post(a).squeeze(0).cpu().numpy()  # (H, 6) degrees


# ---- 1. vision test from the rest pose
K = H - 1  # compare the predicted target at the end of the chunk (~3.3 s ahead)
print(f"== Vision test: predicted joint targets {K / ds.fps:.1f} s after the start, per episode ==")
print(f"{'ep':>4} {'set':>7}  {'pred pan':>8} {'real pan':>8}  {'pred lift':>9} {'real lift':>9}  {'model err':>9} {'avg err':>7}")
gt_k = {e: actions[rows[e][min(K, len(rows[e]) - 1)]] for e in rows}
mean_train = np.mean([gt_k[e] for e in train_eps], axis=0)
preds, gts, m_err, b_err = [], [], [], []
for e in train_eps + held_eps:
    pr = chunk(obs(rows[e][0]))[K]
    gt = gt_k[e]
    me, be = np.abs(pr - gt)[:5].mean(), np.abs(mean_train - gt)[:5].mean()
    preds.append(pr); gts.append(gt); m_err.append(me); b_err.append(be)
    print(f"{e:>4} {'heldout' if e in held_eps else 'train':>7}  {pr[0]:8.1f} {gt[0]:8.1f}  {pr[1]:9.1f} {gt[1]:9.1f}  {me:9.1f} {be:7.1f}")
preds, gts = np.array(preds), np.array(gts)
spread_pred, spread_real = preds[:, 0].std(), gts[:, 0].std()
corr = np.corrcoef(preds[:, 0], gts[:, 0])[0, 1] if spread_pred > 1e-3 else 0.0
print(f"pan spread across episodes: predicted {spread_pred:.1f} deg vs real {spread_real:.1f} deg | "
      f"pred-vs-real pan correlation {corr:.2f}")
print(f"mean error: model {np.mean(m_err):.1f} deg vs 'average of demos' {np.mean(b_err):.1f} deg")

# image swap: state of A, images of B (both at frame 0)
print("\n== Image swap (state from A, camera from B): does the reach follow the camera? ==")
pairs = [(held_eps[0], held_eps[-1]), (held_eps[-1], held_eps[0])] if len(held_eps) > 1 else []
pairs += [(train_eps[0], train_eps[-1]), (train_eps[-1], train_eps[0])]
for a_ep, b_ep in pairs:
    pr_own = chunk(obs(rows[a_ep][0]))[K]
    pr_swap = chunk(obs(rows[a_ep][0], img_from=rows[b_ep][0]))[K]
    print(f"  A={a_ep} B={b_ep}: pan with A's image {pr_own[0]:6.1f}, with B's image {pr_swap[0]:6.1f} | "
          f"real pan A {gt_k[a_ep][0]:6.1f}, B {gt_k[b_ep][0]:6.1f}")

# ---- 2. held-out accuracy along the trajectory
print("\n== Held-out episodes: 1 s ahead predictions every 0.5 s (mean abs error, degrees) ==")
step, horizon = ds.fps // 2, ds.fps
train_trajs = [actions[rows[e]] for e in train_eps]
for e in held_eps:
    idx = rows[e]
    errs, base = [], []
    for t in range(0, len(idx) - horizon, step):
        pr = chunk(obs(idx[t]))[:horizon]
        gt = actions[idx[t:t + horizon]]
        errs.append(np.abs(pr - gt).mean(axis=0))
        avg = np.mean([tr[t:t + horizon] for tr in train_trajs if len(tr) >= t + horizon], axis=0)
        base.append(np.abs(avg - gt).mean(axis=0))
    errs, base = np.array(errs).mean(0), np.array(base).mean(0)
    print(f"  ep {e}: model  " + " ".join(f"{j}={v:4.1f}" for j, v in zip(JOINTS, errs)) + f"  | mean {errs[:5].mean():.1f}")
    print(f"        average " + " ".join(f"{j}={v:4.1f}" for j, v in zip(JOINTS, base)) + f"  | mean {base[:5].mean():.1f}")
