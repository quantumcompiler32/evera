#!/usr/bin/env python
"""Copy the finished episodes of a dataset that lerobot-record is still writing, so training can start
while recording continues.

lerobot-record keeps the current session's parquet/mp4 files open until the session ends (the parquet
has no footer yet), so a plain copy of the folder is unloadable. Episode metadata is only written for
closed files, so this copies exactly the episodes listed there, plus the files they point to, and
rewrites info.json's totals to match. The live dataset is only read, never modified.

    python snapshot_dataset.py so101_pick_place        # -> ~/hackathon/data/so101_pick_place_snap<N>
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATA_DIR = Path.home() / "hackathon/data"


def readable(path: Path) -> bool:
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    args = ap.parse_args()
    src = DATA_DIR / args.dataset
    info = json.loads((src / "meta/info.json").read_text())
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]

    ep_files = sorted(p for p in (src / "meta/episodes").glob("chunk-*/file-*.parquet") if readable(p))
    episodes = [row for p in ep_files for row in pq.read_table(p).to_pylist()]
    episodes.sort(key=lambda r: r["episode_index"])
    n = len(episodes)
    assert [r["episode_index"] for r in episodes] == list(range(n)), "finished episodes are not contiguous"

    needed = {src / "meta/tasks.parquet", src / "meta/stats.json", *ep_files}
    for r in episodes:
        needed.add(src / info["data_path"].format(chunk_index=r["data/chunk_index"], file_index=r["data/file_index"]))
        for vk in video_keys:
            needed.add(src / info["video_path"].format(
                video_key=vk,
                chunk_index=r[f"videos/{vk}/chunk_index"],
                file_index=r[f"videos/{vk}/file_index"],
            ))
    unreadable = [p for p in needed if p.suffix == ".parquet" and not readable(p)]
    assert not unreadable, f"finished episodes point at files still being written: {unreadable}"

    dst = DATA_DIR / f"{args.dataset}_snap{n}"
    if dst.exists():
        shutil.rmtree(dst)
    for p in needed:
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

    # stats.json may already include the live session's episodes; quantiles only widen slightly, which
    # is harmless for normalization, so it's copied as-is rather than recomputed.
    info["total_episodes"] = n
    info["total_frames"] = episodes[-1]["dataset_to_index"]
    info["splits"] = {"train": f"0:{n}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    # Prove it loads and that the last frame of the last episode decodes from every camera.
    ds = LeRobotDataset(f"local/{dst.name}", root=dst)
    last = ds[len(ds) - 1]
    shapes = {k: tuple(last[k].shape) for k in last if k.startswith("observation.images")}
    print(f"{dst}: {ds.num_episodes} episodes, {len(ds)} frames ({len(ds) / info['fps'] / 60:.1f} min) | {shapes}")


if __name__ == "__main__":
    main()
