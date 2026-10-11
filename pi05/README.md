# pi0.5 on the SO-101

Fine-tune and run LeRobot's π₀.₅ (`pi05`) policy on the SO-101 arm, entirely on the AMD NucBox
(GMKtec EVO-X2, Radeon 8060S iGPU, ROCm). Training data is the teleop dataset recorded with
`~/hackathon/robot/record.sh` (`Neevdharod/so101_pick_place`, cameras `scene` + `wrist`).

Copy the scripts to the NucBox from a laptop checkout with `bash pi05/sync.sh` (lands in `~/pi05`).

| Script          | What it does                                                                        |
| --------------- | ----------------------------------------------------------------------------------- |
| `train.sh`      | Fine-tunes `lerobot/pi05_base` on the recorded demos; checkpoints every 500 steps   |
| `make_subset.py` | Cuts a recorded dataset to an episode range minus bad episodes, optionally pushes it |
| `snapshot_dataset.py` | Copies the finished episodes of a dataset that is still being recorded |
| `cloud_train.sh` | Runs the fine-tune on the Radeon Cloud Jupyter instance (upload it; GitHub is blocked there) |
| `check_ckpt.py` | Compares a checkpoint's predicted actions with recorded teleop, without the arm     |
| `rollout.sh`    | Runs a checkpoint on the follower arm (RTC inference, so the arm doesn't stall)     |
| `bench_gpu.py`  | Synthetic memory / speed benchmark with random inputs                               |
| `env.sh`        | Shared conda env, ROCm setting, ports and camera config                             |

## Workflow on the NucBox

```bash
cd ~/pi05 && nohup bash train.sh > train.log 2>&1 &     # ~5.6 s/step at batch 8
python check_ckpt.py outputs/<run>/checkpoints/last/pretrained_model --dataset so101_pick_place
bash rollout.sh                                          # moves the arm: stand next to it
```

## Prerequisites

- Conda env `lerobot` with ROCm torch, plus `scipy` and `sentencepiece` (lerobot's `[pi]` extra).
- The Hugging Face account logged in on the box must have accepted the gated
  [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) license (pi0.5's tokenizer).
- Camera bridges on ports 5555/5556 (`rollout.sh` starts them if needed).

## Measured on the Radeon 8060S (gfx1151)

- bf16 matmul ~16–22 TFLOPS; fp32 only ~2 TFLOPS, so training always uses `--policy.dtype=bfloat16`.
- Expert-only fine-tune (VLM frozen, 0.69B of 4.1B params trainable), batch 8, no gradient checkpointing:
  5.6 s/step, 28 GiB peak. Gradient checkpointing made it 1.65× slower for no needed memory saving.
- Inference: ~545 ms per 50-step action chunk (1.67 s of motion at 30 fps), 9 GiB.
