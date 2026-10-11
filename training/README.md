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

## Offline RL fine-tune (runs on the GMKtec GPU)
`rl_finetune.py` fine-tunes an existing ACT checkpoint on a range of episodes with reward-weighted regression (AWR):
each demo gets a reward from its joint log (failures dropped; points off for an off-target drop, not returning to
rest, a slow reach, stalls, jerky motion) and frames are sampled in proportion to `exp(advantage / beta)`.
`--uniform` trains the same episodes unweighted, as the control. `--dry-run` prints the reward table only.
```bash
python training/rl_finetune.py outputs/trial_act/checkpoints/last/pretrained_model ~/rl_work/snap \
  --episodes 50-80 --heldout 60,70,80 --beta 0.2 --out outputs/rl_act
```
Compare checkpoints offline with `eval_policy.py <ckpt> <root> --train <eps> --heldout 60,70,80`.

Results (Oct 10; fine-tuned 4000 steps from `trial_act`, which saw eps 15-29; held out eps 60, 70, 80; lower is better):

| Model | Reach error from rest, held out (deg) | 1 s-ahead error along held-out eps (deg) | Pan correlation, all eps |
|---|---|---|---|
| `trial_act` (eps 15-29) | 13.3 | 9.4 | 0.11 |
| AWR, beta 0.2 (`Neevdharod/act_so101_pick_place_awr`) | 8.9 | 6.8 | 0.53 |
| Plain fine-tune, eps 50-80 (`Neevdharod/act_so101_pick_place_ep50_80`) | **5.7** | **6.8** | **0.80** |

Training on the newer, cleaner episodes is what helped (about 30% lower error). Reward weighting did not beat
the plain fine-tune: it matched it along trajectories and on typical start spots (eps 70, 80), but focusing on
~14 effective demos cost coverage of the unusual one (ep 60). Use the plain fine-tune on the arm. These are offline
numbers; success rate on the real arm is still untested.

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
