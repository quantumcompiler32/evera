#!/usr/bin/env python3
"""Sanity-check a LeRobot dataset on the Hugging Face Hub before spending GPU time on it.

Needs only the standard library, so it runs anywhere (Mac, GMKtec, cloud):
    python3 training/check_dataset.py YOUR_HF_USER/so101_pick_place

For a private dataset, set HF_TOKEN in the environment first.
"""
import json
import os
import sys
import urllib.error
import urllib.request

REQUIRED_VERSION = "v3.0"  # LeRobot v0.4.1 refuses datasets without this tag
MIN_EPISODES = 30


def get_json(url):
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main(repo_id):
    problems, warnings = [], []
    base = f"https://huggingface.co/api/datasets/{repo_id}"

    try:
        refs = get_json(f"{base}/refs")
    except urllib.error.HTTPError as e:
        sys.exit(f"Can't reach {repo_id} (HTTP {e.code}). Wrong name, or private without HF_TOKEN?")
    tags = [t["name"] for t in refs.get("tags", [])]
    if REQUIRED_VERSION not in tags:
        problems.append(
            f"no '{REQUIRED_VERSION}' tag (found {tags or 'none'}). lerobot-record adds it automatically; "
            f"for a manual upload, tag it with HfApi().create_tag(repo_id, tag='{REQUIRED_VERSION}', repo_type='dataset')"
        )
        info = get_json(f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json")
    else:
        info = get_json(f"https://huggingface.co/datasets/{repo_id}/resolve/{REQUIRED_VERSION}/meta/info.json")

    features = info.get("features", {})
    cameras = [k.removeprefix("observation.images.") for k in features if k.startswith("observation.images.")]
    episodes = info.get("total_episodes", 0)
    frames = info.get("total_frames", 0)
    fps = info.get("fps", 0)

    print(f"dataset   : {repo_id}")
    print(f"version   : {info.get('codebase_version')}   tags: {tags}")
    print(f"robot     : {info.get('robot_type')}")
    print(f"episodes  : {episodes}   frames: {frames}   fps: {fps}")
    if episodes and fps:
        print(f"avg ep len: {frames / episodes / fps:.1f} s")
    print(f"cameras   : {cameras}")
    for key in ("action", "observation.state"):
        if key in features:
            print(f"{key:<18}: {features[key].get('names')}")

    for key in ("action", "observation.state"):
        if key not in features:
            problems.append(f"missing '{key}'")
    if not cameras:
        problems.append("no camera streams, so ACT has nothing to see")
    elif len(cameras) < 2:
        warnings.append("only one camera. A second angle (top + side) usually helps a lot")
    if episodes < MIN_EPISODES:
        warnings.append(f"only {episodes} episodes. Aim for 50+ (about 10 per object position)")

    print()
    for w in warnings:
        print(f"WARN : {w}")
    for p in problems:
        print(f"FAIL : {p}")
    if problems:
        sys.exit(1)
    print("OK   : ready to train")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
