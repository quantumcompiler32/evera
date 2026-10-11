#!/usr/bin/env bash
# pi0.5 expert-only fine-tune on the Radeon Cloud instance (gfx1100), in the background.
# Trains on the clean dataset the recording dashboard uploads (bad episodes already split off).
#
# Needs, in /workspace: the teammate's ACT setup done (`bash training/cloud_setup.sh`, gives /opt/venv
# with ROCm torch + lerobot 0.6.2), an HF login with access to the private dataset (`hf auth login`),
# and paligemma_tokenizer.tgz uploaded in case that login hasn't accepted the gated PaliGemma license.
# GitHub is unreachable from this cloud, so upload this file through Jupyter rather than curl-ing it.
#
#   bash /workspace/cloud_train.sh                     # Neevdharod/so101_screwdriver_bin
#   DATASET=Neevdharod/other STEPS=2000 bash /workspace/cloud_train.sh
#   tail -f /workspace/pi05_train.log
# Rerun the same command after an instance restart: it resumes from the last checkpoint if the disk kept it.
set -euo pipefail
cd /workspace
export TMPDIR=/workspace/tmp HF_HOME=/workspace/.hf MIOPEN_FIND_MODE=FAST PYTHONWARNINGS=ignore
mkdir -p "$TMPDIR" "$HF_HOME/hub"

DATASET="${DATASET:-Neevdharod/so101_screwdriver_bin}"
STEPS="${STEPS:-3000}"
NAME="${DATASET#*/}"
DATA_ROOT=/workspace/pi05_data
OUT="/workspace/pi05_outputs/$NAME"
LOG=/workspace/pi05_train.log

[ -f /opt/venv/bin/activate ] && . /opt/venv/bin/activate
python -c "import lerobot.scripts.lerobot_train" 2>/dev/null \
  || { echo "lerobot isn't installed in /opt/venv yet: run 'bash training/cloud_setup.sh' first"; exit 1; }
hf auth whoami >/dev/null 2>&1 \
  || { echo "not logged in to Hugging Face: run 'hf auth login' (needs access to $DATASET)"; exit 1; }
if pgrep -f "lerobot-train.*pi05_outputs" >/dev/null; then
  echo "pi0.5 training is already running: tail -f $LOG"; exit 0
fi

if ! python -c "import scipy, sentencepiece, transformers" 2>/dev/null; then
  echo "== installing pi0.5 deps (pinned to the existing ROCm torch)"
  pip install -q --timeout 300 --retries 10 -c /workspace/torch_pin.txt \
    "transformers>=5.4.0,<5.6.0" "scipy>=1.14.0,<2.0.0" "sentencepiece>=0.2.0,<0.3.0"
fi

echo "== $DATASET"
hf download "$DATASET" --repo-type dataset --local-dir "$DATA_ROOT/$NAME" > /dev/null
python -c "import json; i = json.load(open('$DATA_ROOT/$NAME/meta/info.json')); print(f\"   {i['total_episodes']} episodes, {i['total_frames'] / i['fps'] / 60:.1f} min\")"

echo "== PaliGemma tokenizer"
if ! python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')" >/dev/null 2>&1; then
  echo "   this HF login can't fetch it; using the uploaded paligemma_tokenizer.tgz"
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
    --dataset.repo_id="$DATASET" --dataset.root="$DATA_ROOT/$NAME"
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
    --output_dir="$OUT" --job_name="pi05_$NAME"
  )
fi
nohup lerobot-train "${ARGS[@]}" > "$LOG" 2>&1 &
echo "== training in the background (pid $!). Watch it with: tail -f $LOG"
