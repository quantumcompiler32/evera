#!/usr/bin/env bash
# ACT training on AMD Radeon Cloud. Run `bash training/cloud_setup.sh` once per instance first.
#
#   bash training/train.sh smoke                        # 200 steps on a public SO-101 dataset: proves the pipeline
#   bash training/train.sh real so101_screwdriver_bin        # full run on our dataset (name under your HF user)
#   bash training/train.sh real so101_screwdriver_bin 20000  # custom step count
#   bash training/train.sh resume act_so101_screwdriver_bin  # continue after the instance restarts
#   tail -f /workspace/train.log                        # watch progress (training survives closing the browser)
#
# Measured on this cloud's GPU (gfx1100), 640x480 x 2 cameras: ~3.3 s/step at batch 64 (~19 samples/s).
# Recording at 320x240 cuts that a lot. Instances seem to stop after ~1 h, so checkpoints go to
# /workspace every SAVE steps and `resume` picks up from the last one.
set -euo pipefail
cd /workspace
export TMPDIR=/workspace/tmp HF_HOME=/workspace/.hf MIOPEN_FIND_MODE=FAST PYTHONWARNINGS=ignore
. /opt/venv/bin/activate

MODE="${1:-}"
hf_user() { echo "${HF_USER:-$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")}"; }
BATCH="${BATCH:-16}"
SAVE="${SAVE:-2000}"
WANDB="${WANDB:-true}"
LOG=/workspace/train.log

case "$MODE" in
  smoke)
    nohup lerobot-train \
      --dataset.repo_id=lerobot/svla_so101_pickplace \
      --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false \
      --batch_size="$BATCH" --steps=200 --save_freq=200 --log_freq=20 --num_workers=16 \
      --output_dir="outputs/train/smoke_$(date +%H%M%S)" --job_name=smoke \
      --wandb.enable="$WANDB" > "$LOG" 2>&1 &
    ;;
  real)
    DATASET="${2:?dataset name, e.g. so101_screwdriver_bin}"
    STEPS="${3:-15000}"
    JOB="act_${DATASET}"
    HF_USER="$(hf_user)"
    python3 /workspace/training/check_dataset.py "${HF_USER}/${DATASET}"
    nohup lerobot-train \
      --dataset.repo_id="${HF_USER}/${DATASET}" \
      --policy.type=act --policy.device=cuda \
      --policy.repo_id="${HF_USER}/${JOB}" --policy.push_to_hub=true \
      --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE" --log_freq=100 --num_workers=16 \
      --output_dir="outputs/train/${JOB}" --job_name="${JOB}" \
      --wandb.enable="$WANDB" > "$LOG" 2>&1 &
    ;;
  resume)
    JOB="${2:?job name, e.g. act_so101_screwdriver_bin}"
    nohup lerobot-train \
      --config_path="outputs/train/${JOB}/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true >> "$LOG" 2>&1 &
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1
    ;;
esac
echo "Started in the background. Watch: tail -f $LOG   (Ctrl-C stops watching, not training)"
