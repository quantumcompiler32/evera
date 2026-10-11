#!/usr/bin/env bash
# Upload an intermediate checkpoint so the arm team can test it while training continues.
#   bash push_checkpoint.sh act_so101_pick_place 010000
set -euo pipefail
cd /workspace && . /opt/venv/bin/activate
JOB="${1:?job name}"
CKPT="${2:?checkpoint step, e.g. 010000 (ls outputs/train/$JOB/checkpoints)}"
HF_USER="${HF_USER:-$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")}"
hf upload "${HF_USER}/${JOB}_${CKPT}" "outputs/train/${JOB}/checkpoints/${CKPT}/pretrained_model"
echo "Arm team: hf download ${HF_USER}/${JOB}_${CKPT} --local-dir models/${JOB}_${CKPT}"
