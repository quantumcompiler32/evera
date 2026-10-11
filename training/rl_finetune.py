"""Offline RL fine-tune of an ACT checkpoint: reward-weighted regression on a range of recorded episodes.

    python training/rl_finetune.py <init_ckpt> <dataset_root> --episodes 50-80 --heldout 60,70,80 \
        --out outputs/rl_act [--steps 4000] [--beta 0.1] [--uniform] [--dry-run]

We can't let the arm explore on its own (no reset, no safety cage), so this is the offline form of RL:
1. Reward: each demo gets a scalar return from its joint log. Failures (no grasp, never let go) get 0
   and are dropped. Successes start at 1 and lose points for dropping the object away from the usual
   spot, not returning to rest, a slow reach, mid-task stalls, and jerky motion.
2. Advantage weighting (AWR, Peng et al. 2019): A = R - mean(R), w = exp(A / beta), clipped at 20.
   Training frames are sampled in proportion to their episode's weight, so the policy imitates the
   best demos most and the sloppy ones barely. Weighted sampling = weighted L1 loss in expectation.
3. Fine-tune from <init_ckpt> with lerobot-train, only the sampler is swapped.
--uniform trains the same episodes with equal weights (plain behaviour cloning), as the control.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
SQUEEZE, MIN_HOLD_S, MIN_OPEN = 1.5, 1.0, 4  # same grasp test as robot/review_episodes.py


def parse_eps(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def longest_run(mask):
    best, start = (0, 0), None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def episode_rewards(repo, root, eps):
    ds = LeRobotDataset(repo, root=root, episodes=eps)
    fps = ds.fps
    cols = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(cols["episode_index"]).reshape(-1)
    state, action = np.stack(cols["observation.state"]), np.stack(cols["action"])

    M = {}
    for e in eps:
        m = np.where(ep_idx == e)[0]
        s, a = state[m], action[m]
        squeeze = s[:, 5] - a[:, 5]
        k0, k1 = longest_run(squeeze > SQUEEZE)
        grasp = s[:, 5].max() - np.percentile(s[:, 5], 5) > MIN_OPEN and (k1 - k0) / fps >= MIN_HOLD_S
        released = squeeze[-max(1, fps // 2):].mean() < 1.0
        vel = np.abs(np.diff(s[:, :5], axis=0)).max(axis=1) * fps
        moving = np.where(vel > 8)[0]
        t_move = moving[0] / fps if len(moving) else len(s) / fps
        M[e] = dict(
            success=bool(grasp and released),
            start=s[0, :5], end=s[-1, :5],
            release=s[k1 - 1, :4] if grasp else None,
            reach_s=max(0.0, k0 / fps - t_move),  # first motion -> jaws close on the object
            pauses=int((np.diff(moving) / fps > 1.0).sum()) if len(moving) > 1 else 0,
            jerk=float(np.abs(np.diff(s[:, :5], n=2, axis=0)).mean() * fps * fps),
        )

    ok = [e for e in eps if M[e]["success"]]
    if not ok:
        sys.exit("no successful episodes in that range")
    rest = np.median(np.stack([M[e]["start"] for e in eps]), axis=0)
    place = np.median(np.stack([M[e]["release"] for e in ok]), axis=0)
    jerk_lo, jerk_hi = np.percentile([M[e]["jerk"] for e in ok], [10, 90])
    for e in eps:
        r = M[e]
        if not r["success"]:
            r.update(R=0.0, place_off=float("nan"), end_off=float("nan"))
            continue
        r["place_off"] = float(np.abs(r["release"] - place).max())
        r["end_off"] = float(np.abs(r["end"] - rest).max())
        r["R"] = float(1.0
                       - 0.40 * min(r["place_off"] / 30, 1)
                       - 0.20 * min(r["end_off"] / 30, 1)
                       - 0.20 * min(r["reach_s"] / 10, 1)
                       - 0.10 * min(r["pauses"] / 3, 1)
                       - 0.10 * float(np.clip((r["jerk"] - jerk_lo) / max(jerk_hi - jerk_lo, 1e-6), 0, 1)))
    return M, ok


class WeightedEpisodeSampler(EpisodeAwareSampler):
    """EpisodeAwareSampler that draws frames with probability proportional to their episode's weight."""

    weights = {}  # episode index -> weight, set before training starts

    def __init__(self, dataset_from_indices, dataset_to_indices, episode_indices_to_use=None, **kw):
        super().__init__(dataset_from_indices, dataset_to_indices, episode_indices_to_use, **kw)
        used = sorted(set(episode_indices_to_use)) if episode_indices_to_use is not None \
            else range(len(dataset_from_indices))
        lengths = np.diff(np.concatenate([[0], self._cum_lengths]))
        self._p = None
        if self.shuffle and self.weights:  # train loader only; eval stays uniform
            w = np.concatenate([np.full(n, self.weights.get(e, 0.0)) for e, n in zip(used, lengths)])
            self._p = torch.as_tensor(w / w.sum(), dtype=torch.double)

    def _iter_epoch(self, epoch, start):
        if self._p is None:
            yield from super()._iter_epoch(epoch, start)
            return
        order = torch.multinomial(self._p, self._num_frames, replacement=True,
                                  generator=self._epoch_generator(epoch))
        for k in range(start, self._num_frames):
            yield self._frame_index(int(order[k]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="pretrained_model dir to start from")
    p.add_argument("root", help="local dataset root (use a snapshot copy)")
    p.add_argument("--repo", default="Neevdharod/so101_pick_place")
    p.add_argument("--episodes", default="50-80")
    p.add_argument("--heldout", default="60,70,80", help="kept out of training, for eval_policy.py")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1, help="AWR temperature: lower = favour the best demos harder")
    p.add_argument("--uniform", action="store_true", help="equal weights (plain BC control)")
    p.add_argument("--dry-run", action="store_true", help="print rewards and weights, don't train")
    args = p.parse_args()

    eps, held = parse_eps(args.episodes), set(parse_eps(args.heldout))
    M, ok = episode_rewards(args.repo, args.root, eps)
    train_eps = [e for e in ok if e not in held]
    R = np.array([M[e]["R"] for e in train_eps])
    adv = R - R.mean()
    w = np.ones_like(R) if args.uniform else np.minimum(np.exp(adv / args.beta), 20.0)
    w = w / w.mean()
    weights = dict(zip(train_eps, w.tolist()))

    print(f"{'ep':>3} {'ok':>3} {'place_off':>9} {'end_off':>7} {'reach_s':>7} {'pauses':>6} {'jerk':>6} {'R':>5} {'weight':>6}")
    for e in eps:
        r = M[e]
        tag = "held" if e in held else f"{weights.get(e, 0.0):6.2f}"
        print(f"{e:>3} {'Y' if r['success'] else 'N':>3} {r['place_off']:9.0f} {r['end_off']:7.0f} {r['reach_s']:7.1f} "
              f"{r['pauses']:6d} {r['jerk']:6.0f} {r['R']:5.2f} {tag:>6}")
    dropped = [e for e in eps if not M[e]["success"]]
    top = sorted(weights, key=weights.get, reverse=True)
    print(f"\ntrain on {len(train_eps)} eps, dropped failures {dropped}, held out {sorted(held)}")
    print(f"effective sample size {w.sum() ** 2 / (w ** 2).sum():.1f} of {len(w)}; top 5: {top[:5]}")

    out = Path(args.out)
    if args.dry_run:
        return
    if out.exists():
        sys.exit(f"{out} exists: pick another --out")

    from lerobot.scripts import lerobot_train
    WeightedEpisodeSampler.weights = weights
    lerobot_train.EpisodeAwareSampler = WeightedEpisodeSampler
    sys.argv = [
        "lerobot-train",
        f"--policy.path={args.ckpt}", "--policy.device=cuda", "--policy.push_to_hub=false",
        f"--dataset.repo_id={args.repo}", f"--dataset.root={args.root}",
        f"--dataset.episodes={json.dumps(train_eps)}",
        f"--batch_size={args.batch}", f"--steps={args.steps}", f"--save_freq={args.steps // 2}",
        "--log_freq=100", "--num_workers=8", "--wandb.enable=false",
        f"--output_dir={out}", f"--job_name={out.name}",
    ]
    lerobot_train.main()
    (out / "rl_rewards.json").write_text(json.dumps(
        {"beta": args.beta, "uniform": args.uniform, "held_out": sorted(held), "dropped": dropped,
         "episodes": {e: {"R": M[e]["R"], "weight": weights.get(e, 0.0)} for e in eps}}, indent=1))


if __name__ == "__main__":
    main()
