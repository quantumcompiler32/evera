#!/usr/bin/env bash
# pi0.5 expert-only fine-tune on the Radeon Cloud instance (gfx1100), in the background.
# Reuses the teammate's /opt/venv (ROCm torch + lerobot 0.6.2 from training/cloud_setup.sh) and needs two
# uploads in /workspace, built on the Mac from the NucBox:
#   pi05_upload.tar            finished-episode dataset snapshot (snapshot_dataset.py)
#   paligemma_tokenizer.tgz    gated PaliGemma tokenizer as an HF cache entry, so no HF login is needed
#
#   bash /workspace/cloud_train.sh
#   tail -f /workspace/pi05_train.log
# Instances stop after ~1 h: rerun the same command after a restart and it resumes from the last checkpoint.
set -euo pipefail
cd /workspace
export TMPDIR=/workspace/tmp HF_HOME=/workspace/.hf MIOPEN_FIND_MODE=FAST PYTHONWARNINGS=ignore
mkdir -p "$TMPDIR" "$HF_HOME/hub"
. /opt/venv/bin/activate

DATASET="${DATASET:-so101_pick_place_snap32}"
STEPS="${STEPS:-3000}"
DATA_ROOT=/workspace/pi05_upload
OUT="/workspace/pi05_outputs/$DATASET"
LOG=/workspace/pi05_train.log

if pgrep -f "lerobot-train.*pi05_outputs" >/dev/null; then
  echo "pi0.5 training is already running: tail -f $LOG"; exit 0
fi

if ! python -c "import scipy, sentencepiece, transformers" 2>/dev/null; then
  echo "== installing pi0.5 deps (pinned to the existing ROCm torch)"
  pip install -q --timeout 300 --retries 10 -c /workspace/torch_pin.txt \
    "transformers>=5.4.0,<5.6.0" "scipy>=1.14.0,<2.0.0" "sentencepiece>=0.2.0,<0.3.0"
fi
if [ ! -f "$DATA_ROOT/$DATASET/meta/info.json" ]; then
  echo "== unpacking dataset"
  mkdir -p "$DATA_ROOT" && tar xf pi05_upload.tar -C "$DATA_ROOT" --warning=no-unknown-keyword
fi
if [ ! -d "$HF_HOME/hub/models--google--paligemma-3b-pt-224/snapshots" ]; then
  echo "== unpacking PaliGemma tokenizer"
  tar xzf paligemma_tokenizer.tgz -C "$HF_HOME/hub" --warning=no-unknown-keyword
fi
echo "== lerobot/pi05_base (14.5 GB on first run)"
hf download lerobot/pi05_base > /dev/null
# Everything is cached now; offline mode keeps the gated tokenizer from being re-checked on the Hub.
export HF_HUB_OFFLINE=1

read -r GPU VRAM < <(python -c "import torch; p = torch.cuda.get_device_properties(0); print(p.gcnArchName.split(':')[0], round(p.total_memory / 2**30))")
# Expert-only without gradient checkpointing measured ~11 GiB + ~2.2 GiB per sample on the NucBox.
if [ "$VRAM" -ge 40 ]; then BATCH="${BATCH:-8}"; else BATCH="${BATCH:-4}"; fi
echo "== $GPU with ${VRAM} GiB -> batch $BATCH, $STEPS steps"

if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
  echo "== resuming from $(readlink -f "$OUT/checkpoints/last")"
  ARGS=(--config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true)
else
  rm -rf "$OUT"  # a run that died before its first checkpoint
  ARGS=(
    --dataset.repo_id="Neevdharod/$DATASET" --dataset.root="$DATA_ROOT/$DATASET"
    --policy.type=pi05
    --policy.pretrained_path=lerobot/pi05_base
    --policy.dtype=bfloat16
    --policy.device=cuda
    --policy.gradient_checkpointing=false
    --policy.freeze_vision_encoder=true
    --policy.train_expert_only=true
    --policy.push_to_hub=false
    --batch_size="$BATCH"
    --num_workers=8
    --steps="$STEPS"
    --save_freq=250
    --log_freq=10
    --output_dir="$OUT" --job_name="pi05_$DATASET"
  )
fi
nohup lerobot-train "${ARGS[@]}" > "$LOG" 2>&1 &
echo "== training in the background (pid $!). Watch it with: tail -f $LOG"
