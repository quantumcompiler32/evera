#!/usr/bin/env bash
# Fine-tune pi0.5 (lerobot/pi05_base) on our SO-101 demos on the GMKtec's Radeon 8060S.
#   bash ~/pi05/train.sh                                  # full 50-episode set, Neevdharod/so101_pick_place
#   DATASET=test_run STEPS=20 bash ~/pi05/train.sh        # smoke test on the 2-episode recording
# Runs in the foreground; wrap in tmux/nohup for long runs. Checkpoints land in ~/pi05/outputs/<run>/.
set -euo pipefail
# gradient_checkpointing=false: with the VLM frozen, checkpointing re-runs its forward in backward;
# turning it off measured 1.65x faster (1.5 vs 0.9 samples/s). tokenizer_max_length=64: our prompt
# (task + 6 discretized joints) is ~42 PaliGemma tokens, so the default 200 is mostly padding.
source "$(dirname "$0")/env.sh"

DATASET="${DATASET:-so101_pick_place}"
STEPS="${STEPS:-3000}"  # pi05's cosine LR schedule auto-shrinks its 1k warmup / 30k decay to fit
RUN="pi05_${DATASET}_$(date +%m%d_%H%M)"

# Expert-only: freeze the 2.9B PaliGemma VLM and train just the ~0.3B action expert + projections.
# The iGPU is compute-bound (~16-22 bf16 TFLOPS), and skipping the VLM backward is ~3-4x faster per
# step, which matters more on a hackathon clock than the modest success-rate gain of a full fine-tune.
FREEZE_VISION="${FREEZE_VISION:-true}"
EXPERT_ONLY="${EXPERT_ONLY:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"  # 28 GiB peak; 16 needs 46 GiB for the same samples/s

lerobot-train \
  --dataset.repo_id="$HF_USER/$DATASET" --dataset.root="$DATA_DIR/$DATASET" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=false \
  --policy.tokenizer_max_length=64 \
  --policy.freeze_vision_encoder="$FREEZE_VISION" \
  --policy.train_expert_only="$EXPERT_ONLY" \
  --policy.push_to_hub=false \
  --batch_size="$BATCH_SIZE" \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_freq=500 \
  --log_freq=10 \
  --output_dir="$OUT_DIR/$RUN" --job_name="$RUN"
