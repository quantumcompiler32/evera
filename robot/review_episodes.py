"""Quality report for recorded episodes (read-only; run on a snapshot copy while recording continues).

    nice -n 19 python robot/review_episodes.py <dataset_root> <first_ep> [out_dir]
    (or just: bash robot/record.sh review)

Prints one line per episode (grasp, release, drop-spot consistency, rest pose, idle time, jerkiness,
tracking) and writes contact sheets (top camera at start / grasp / end, wrist at grasp) to out_dir.

Grasp detection: the SO-101 gripper only moves ~8-20 units (of 100) when grabbing a thin object, so
the gripper position alone is a weak signal. What is unambiguous is the *squeeze*: while the jaws are
closed on an object, the leader commands the gripper further shut than the follower can go, so
state - action stays positive. A grasp = the gripper opened, then squeezed for >= 1 s.
"""
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root, first = Path(sys.argv[1]), int(sys.argv[2])
out = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/review")
out.mkdir(parents=True, exist_ok=True)
# repo id is only a label when the data is local; strip snapshot/clean suffixes from the folder name
repo = os.environ.get("REPO") or "Neevdharod/" + re.sub(r"_(snap\d*|clean)$", "", root.name)

SQUEEZE = 1.5        # gripper state - action (units of 0-100) that means "jaws blocked by the object"
MIN_HOLD_S = 1.0     # squeeze this long to count as a grasp
MIN_OPEN = 4         # gripper must open this far above its closed position first
PLACE_TOL = 15       # deg: release pose further than this from the typical one = different drop spot

ds = LeRobotDataset(repo, root=root)
fps = ds.fps
names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
cols = ds.hf_dataset.with_format("numpy")
ep_idx = np.asarray(cols["episode_index"]).reshape(-1)
state = np.stack(cols["observation.state"])
action = np.stack(cols["action"])
eps = [e for e in sorted(set(ep_idx.tolist())) if e >= first]


def longest_run(mask):
    """(start, end) of the longest stretch of True, end exclusive; (0, 0) if none."""
    best, start = (0, 0), None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


# ---- pass 1: per-episode metrics
M = {}
for e in eps:
    m = np.where(ep_idx == e)[0]
    s, a = state[m], action[m]
    g = s[:, 5]
    squeeze = g - a[:, 5]
    open_amt = g.max() - np.percentile(g, 5)
    k0, k1 = longest_run(squeeze > SQUEEZE)
    hold_s = (k1 - k0) / fps
    grasp = open_amt > MIN_OPEN and hold_s >= MIN_HOLD_S
    vel = np.abs(np.diff(s[:, :5], axis=0)).max(axis=1) * fps  # deg/s, arm joints
    moving = vel > 8
    mv = np.where(moving)[0]
    M[e] = dict(
        m=m, s=s, a=a, grasp=grasp, open_amt=open_amt, hold_s=hold_s,
        released=squeeze[-max(1, fps // 2):].mean() < 1.0,  # not squeezing in the last 0.5 s
        k_grasp=min(k0 + fps // 2, k1 - 1) if grasp else int(np.argmax(g)),
        release_pose=s[k1 - 1, :4] if grasp else None,  # arm pose when the object was let go
        vel=vel,
        idle_start=(np.argmax(moving) if moving.any() else len(vel)) / fps,
        idle_end=(np.argmax(moving[::-1]) if moving.any() else len(vel)) / fps,
        pauses=int((np.diff(mv) / fps > 1.0).sum()) if len(mv) > 1 else 0,  # >1 s stalls mid-task
        track=np.abs(a - s)[:, :5].mean(),
    )

rest = np.median(np.stack([M[e]["s"][0] for e in eps]), axis=0)  # typical rest pose
releases = [M[e]["release_pose"] for e in eps if M[e]["grasp"]]
place = np.median(np.stack(releases), axis=0) if releases else None  # typical drop-spot pose

# ---- pass 2: flags, report, contact sheets
rows, report = [], []
print(f"{'ep':>3} {'sec':>5} {'open':>5} {'hold_s':>6} {'grasp':>6} {'release':>7} {'place_off':>9} {'start_off':>9} "
      f"{'end_off':>7} {'idle_start':>10} {'idle_end':>8} {'pauses':>6} {'max_vel':>7} {'track':>5}  flags")
for e in eps:
    r = M[e]
    s, m = r["s"], r["m"]
    start_off = np.abs(s[0, :5] - rest[:5]).max()
    end_off = np.abs(s[-1, :5] - rest[:5]).max()
    place_off = np.abs(r["release_pose"] - place).max() if r["grasp"] else float("nan")
    flags = []
    if not r["grasp"]: flags.append("NO_GRASP")
    if r["grasp"] and not r["released"]: flags.append("NOT_RELEASED")  # still squeezing at the end: overload risk
    if r["grasp"] and place_off > PLACE_TOL: flags.append("PLACE_OFF")
    if start_off > 15: flags.append("START_NOT_REST")
    if end_off > 20: flags.append("END_NOT_REST")
    if r["idle_start"] > 4: flags.append("SLOW_START")
    if r["pauses"] >= 2: flags.append("HESITANT")
    if r["vel"].max() > 400: flags.append("JERK")
    if r["track"] > 6: flags.append("POOR_TRACKING")
    report.append((e, flags))
    print(f"{e:>3} {len(m)/fps:5.1f} {r['open_amt']:5.1f} {r['hold_s']:6.1f} {str(r['grasp']):>6} {str(r['released']):>7} "
          f"{place_off:9.0f} {start_off:9.0f} {end_off:7.0f} {r['idle_start']:10.1f} {r['idle_end']:8.1f} "
          f"{r['pauses']:6d} {r['vel'].max():7.0f} {r['track']:5.1f}  {' '.join(flags)}")

    # contact-sheet row: top camera at start, grasp moment, end; wrist at grasp
    frames = []
    k_grasp = r["k_grasp"]
    for k, key in ((0, "observation.images.scene"), (k_grasp, "observation.images.scene"),
                   (len(m) - 1, "observation.images.scene"), (k_grasp, "observation.images.wrist")):
        img = ds[int(m[k])][key]  # CHW float [0,1]
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(cv2.cvtColor(cv2.resize(img, (240, 180)), cv2.COLOR_RGB2BGR))
    row = np.hstack(frames)
    cv2.putText(row, f"ep {e} {' '.join(flags) or 'OK'}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if flags else (0, 200, 0), 2)
    rows.append(row)

for i in range(0, len(rows), 8):
    cv2.imwrite(str(out / f"sheet_{i // 8}.jpg"), np.vstack(rows[i:i + 8]), [cv2.IMWRITE_JPEG_QUALITY, 80])

bad = [e for e, f in report if f]
print(f"\nrest pose (median start): {dict(zip(names, rest.round(0).tolist()))}")
if releases:
    spread = np.stack(releases).std(axis=0)
    print(f"drop-spot spread (std of release pose, deg): {dict(zip(names[:4], spread.round(1).tolist()))}"
          f"  <- aim for < 5 on pan: one fixed bin")
    opens = [M[e]["open_amt"] for e in eps if M[e]["grasp"]]
    print(f"gripper opening before grasp: median {np.median(opens):.0f} (of 100). Under ~10 leaves little margin; open wider.")
print(f"{len(eps)} episodes checked, {len(eps) - len(bad)} clean, flagged: {bad}")
print(f"contact sheets: {sorted(p.name for p in out.glob('sheet_*.jpg'))}  (columns: top@start, top@grasp, top@end, wrist@grasp)")
