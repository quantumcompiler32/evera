"""Ghost replay: overlay each model's planned joint trajectory on what the human did, for held-out episodes.

    python training/replay_plot.py replay.npz <dataset_root> --heldout 60,70,80 \
        --model "Fine-tune 50-80=outputs/bc_act_50_80/checkpoints/last/pretrained_model" \
        --model "Old trial_act=outputs/trial_act/checkpoints/last/pretrained_model"
    python3 training/replay_plot.py replay.png --from replay.npz     # plot (needs matplotlib, e.g. on a laptop)

The model re-plans the way it does on the arm: every n_action_steps frames it looks at the cameras + joints and
emits the next chunk. Here the observation at each re-plan is the recorded one, so this shows "would it make the
same moves as the human from the same situation", not closed-loop success (errors can compound on the real arm).
"""
import argparse
import json

import numpy as np

JOINTS = {0: "Pan (deg)", 1: "Lift (deg)", 2: "Elbow (deg)", 5: "Gripper (0-100)"}
HUMAN, COLORS = "#52514e", ["#2a78d6", "#eb6834"]  # reference palette: secondary ink, slots 1-2

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("root", nargs="?")
p.add_argument("--from", dest="src", help="plot a saved .npz instead of running models")
p.add_argument("--repo", default="Neevdharod/so101_pick_place")
p.add_argument("--heldout", default="60,70,80")
p.add_argument("--model", action="append", default=[], help="label=checkpoint_dir")
args = p.parse_args()


def run_models():
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    eps = [int(e) for e in args.heldout.split(",")]
    ds = LeRobotDataset(args.repo, root=args.root, episodes=eps)
    fps = ds.fps
    ep_index = np.asarray(ds.hf_dataset.with_format("numpy")["episode_index"]).reshape(-1)
    actions = np.stack(ds.hf_dataset.with_format("numpy")["action"])
    rows = {e: np.where(ep_index == e)[0] for e in eps}


    @torch.no_grad()
    def replay(ckpt):
        policy = ACTPolicy.from_pretrained(ckpt).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                             preprocessor_overrides={"device_processor": {"device": "cuda"}})
        keys = ["observation.state", *policy.config.image_features]
        n = policy.config.n_action_steps
        out = {}
        for e in eps:
            idx, traj = rows[e], []
            for t in range(0, len(idx), n):
                item = ds[int(idx[t])]
                chunk = post(policy.predict_action_chunk(pre({k: item[k] for k in keys}))).squeeze(0).cpu().numpy()
                traj.append(chunk[:n])
            out[e] = np.concatenate(traj)[:len(idx)]
        return out, n


    models = []
    for spec in args.model:
        label, _, ckpt = spec.partition("=")
        pred, n = replay(ckpt)
        err = np.mean([np.abs(pred[e][:, :5] - actions[rows[e]][:, :5]).mean() for e in eps])
        models.append((label, pred, err))
        print(f"{label}: mean arm-joint error over held-out eps {err:.1f} deg")
    data = {"fps": fps, "n": n, "eps": eps, "labels": [m[0] for m in models]}
    arrays = {f"human_{e}": actions[rows[e]] for e in eps}
    arrays.update({f"pred{i}_{e}": m[1][e] for i, m in enumerate(models) for e in eps})
    np.savez(args.out, meta=json.dumps(data), **arrays)
    print("saved", args.out)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.src)
    meta = json.loads(str(z["meta"]))
    fps, n, eps = meta["fps"], meta["n"], meta["eps"]
    human = {e: z[f"human_{e}"] for e in eps}
    models = []
    for i, label in enumerate(meta["labels"]):
        pred = {e: z[f"pred{i}_{e}"] for e in eps}
        err = np.mean([np.abs(pred[e][:, :5] - human[e][:, :5]).mean() for e in eps])
        models.append((label, pred, err))


    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d8d7d0", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b"})
    fig, axes = plt.subplots(len(eps), len(JOINTS), figsize=(14, 3.1 * len(eps)), sharex="row")
    fig.patch.set_facecolor("#fcfcfb")
    for r, e in enumerate(eps):
        t = np.arange(len(human[e])) / fps
        for c, (j, name) in enumerate(JOINTS.items()):
            ax = axes[r, c]
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, color="#ecebe6", linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for k in np.arange(0, len(t), n) / fps:  # re-plan moments
                ax.axvline(k, color="#d8d7d0", linewidth=0.8, linestyle=(0, (2, 3)))
            ax.plot(t, human[e][:, j], color=HUMAN, linewidth=2.6, label="Human demo", solid_capstyle="round")
            for (label, pred, _), col in zip(models, COLORS):
                ax.plot(t, pred[e][:, j], color=col, linewidth=1.6, label=label, solid_capstyle="round")
            if r == 0:
                ax.set_title(name, fontsize=11, loc="left")
            if c == 0:
                ax.set_ylabel(f"Episode {e}", fontsize=11, color="#0b0b0b")
            if r == len(eps) - 1:
                ax.set_xlabel("Seconds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [l if l == "Human demo" else f"{l} (avg error {m[2]:.1f} deg)"
              for l, m in zip(labels, [None] + models)]
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 1.0), fontsize=11)
    fig.suptitle("Held-out episodes the models never trained on. Dotted lines: the model looks again and re-plans.",
                 x=0.01, y=0.955, ha="left", fontsize=10, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", args.out)


plot() if args.src else run_models()
