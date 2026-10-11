#!/usr/bin/env bash
# Run a fine-tuned pi0.5 checkpoint on the SO-101 follower (leader arm not needed).
#   bash ~/pi05/rollout.sh                                   # newest run's last checkpoint
#   CKPT=~/pi05/outputs/<run>/checkpoints/003000/pretrained_model bash ~/pi05/rollout.sh
#   INFERENCE=sync bash ~/pi05/rollout.sh                    # robot pauses while each chunk is computed
set -euo pipefail
source "$(dirname "$0")/env.sh"

LATEST_RUN="$(ls -td "$OUT_DIR"/*/ 2>/dev/null | head -1)"
CKPT="${CKPT:-${LATEST_RUN}checkpoints/last/pretrained_model}"
[ -f "$CKPT/config.json" ] || { echo "no checkpoint at '$CKPT'; train first or set CKPT=..."; exit 1; }

for n in top wrist; do
  if ! { [ -f ~/cam_$n.pid ] && kill -0 "$(cat ~/cam_$n.pid)" 2>/dev/null; }; then
    echo "camera bridges not running, starting them..."
    bash ~/hackathon/robot/start_limelight.sh
    sleep 8
    break
  fi
done

# RTC computes the next chunk in a background thread while the current one executes, hiding the
# iGPU's inference latency; sync is simpler but the arm stalls between chunks.
echo "rollout: $CKPT (inference=${INFERENCE:-rtc})"
lerobot-rollout \
  --strategy.type=base \
  --policy.path="$CKPT" \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
  --robot.cameras="$CAMS" \
  --inference.type="${INFERENCE:-rtc}" \
  --task="$TASK" \
  --fps=30 \
  --duration="${DURATION:-60}" \
  --device=cuda
