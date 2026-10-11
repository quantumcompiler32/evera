#!/usr/bin/env python
"""Cut a recorded dataset down to a range of episodes, minus the ones marked bad, and optionally push it.

Uses lerobot's delete_episodes (the same call record.sh's upload uses to split off bad episodes), so the
kept episodes are renumbered from 0 and the normalization stats are recomputed for just that subset.

    python make_subset.py so101_pick_place 50 80                 # -> ~/hackathon/data/so101_pick_place_50_80
    python make_subset.py so101_pick_place 50 80 --push          # also push as private Neevdharod/<name>
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA_DIR = Path.home() / "hackathon/data"
HF_USER = "Neevdharod"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int, help="inclusive")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    full = LeRobotDataset(f"{HF_USER}/{args.dataset}", root=DATA_DIR / args.dataset)
    bad_file = DATA_DIR / f"{args.dataset}.bad"  # written by record.sh / the dashboard's "mark bad"
    bad = {int(x) for x in bad_file.read_text().split()} if bad_file.exists() else set()
    keep = [e for e in range(args.first, args.last + 1) if e not in bad and e < full.num_episodes]
    print(f"keeping original episodes {keep} ({len(keep)}), skipping bad {sorted(bad & set(range(args.first, args.last + 1)))}")

    name = f"{args.dataset}_{args.first}_{args.last}"
    drop = [e for e in range(full.num_episodes) if e not in keep]
    ds = delete_episodes(full, drop, output_dir=DATA_DIR / name, repo_id=f"{HF_USER}/{name}")
    task = ds[len(ds) - 1]["task"]
    print(f"{name}: {ds.num_episodes} episodes, {len(ds) / ds.fps / 60:.1f} min | task: {task!r}")
    if args.push:
        ds.push_to_hub(private=True)
        print(f"pushed private dataset {HF_USER}/{name}")


if __name__ == "__main__":
    main()
