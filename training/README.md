# ACT training on AMD Radeon Cloud

The dataset comes from `../robot/` (Hugging Face: `Neevdharod/so101_pick_place`). This folder trains an ACT
policy on a Radeon Cloud notebook instance (radeon-global.anruicloud.com, 1 GPU gfx1100, ROCm).

## Steps
1. Launch a notebook instance. **Its disk is wiped on every launch**, so redo steps 2–3 each time and push checkpoints to the Hub.
2. Upload this folder to `/workspace/training/`, plus `lerobot_src.tgz` to `/workspace/`. That tarball is the exact
   LeRobot 0.6.2 source used on the robot (not committed here); the trained policy must match the robot's LeRobot version.
3. In a Jupyter terminal:
   ```bash
   bash /workspace/training/cloud_setup.sh
   ```
   This takes about a minute and reuses the image's torch 2.9.1+rocm7.2. Then log in: `hf auth login`, `wandb login`.
4. Check the data, then train (runs in the background and survives closing the browser):
   ```bash
   python3 /workspace/training/check_dataset.py Neevdharod/so101_pick_place
   ```
   ```bash
   bash /workspace/training/train.sh real so101_pick_place
   ```
   ```bash
   tail -f /workspace/train.log
   ```
5. Share an early checkpoint with the robot: `bash /workspace/training/push_checkpoint.sh act_so101_pick_place 004000`.
6. If the instance stops: relaunch, redo 2–3, then `bash /workspace/training/train.sh resume act_so101_pick_place`
   (only if `/workspace/outputs` survived; otherwise restart from the last checkpoint on the Hub).

## Platform quirks handled by `cloud_setup.sh`
- `/tmp` is a 4 GB tmpfs, so pip temp files go to `/workspace/tmp`.
- Egress goes through a slow proxy: pip uses `--timeout 300`. github.com is blocked, so nothing is installed from git.
- LeRobot sets `cudnn.benchmark=True`; on ROCm that triggers MIOpen's exhaustive search and stalls for many minutes
  before step 1, so it's patched off and `MIOPEN_FIND_MODE=FAST` is set.
- Open issue: the image's torchcodec 0.11 doesn't load with its custom torch build. Try `torchcodec==0.9.*`, otherwise
  pass `--dataset.video_backend=pyav`.

## Measured speed
ACT with 2 cameras at 640x480, batch 64: about 3.3 s/step on this GPU. The robot records at 320x240, which is much faster.
`bench.py` separates GPU compute from data loading if training is slow.

## Model, for Q&A
ACT: a ResNet18 CNN (ImageNet-pretrained) encodes each camera image. A transformer with a CVAE then predicts a chunk of
100 future joint positions at once. It's trained by imitation (L1 on actions plus KL) from about 50 teleop demos.
ACT ignores the task text, so choosing *which* object to pick needs detection (see `../vision/`) or a VLA.
