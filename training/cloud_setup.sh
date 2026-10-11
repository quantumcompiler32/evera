#!/usr/bin/env bash
# One-shot setup for an AMD Radeon Cloud notebook instance (radeon-global.anruicloud.com).
# Needs /workspace/lerobot_src.tgz: the exact LeRobot 0.6.2 source from the GMKtec (git main, b9cb121c),
# so the trained policy loads on the robot. Upload it first, then:
#   bash /workspace/training/cloud_setup.sh
# Safe to re-run. Everything lives on /workspace.
#
# Platform quirks this handles:
#   - /tmp is a 4 GB tmpfs: pip temp files go to /workspace/tmp
#   - egress is via a slow proxy: pip --timeout 300; github.com is blocked, so no git installs
#   - the image's torch 2.9.1+rocm7.2 satisfies lerobot 0.6.2 (torch<2.12), so lerobot goes into the
#     image's /opt/venv with torch pinned, instead of downloading another ~4 GB wheel
#   - lerobot-train turns on cudnn.benchmark, which on ROCm runs MIOpen's exhaustive search and
#     stalls for many minutes before step 1: patched off, plus MIOPEN_FIND_MODE=FAST
set -euo pipefail
cd /workspace

ENV_LINE='export TMPDIR=/workspace/tmp PIP_CACHE_DIR=/workspace/.pipcache HF_HOME=/workspace/.hf MIOPEN_FIND_MODE=FAST PYTHONWARNINGS=ignore'
grep -qF "$ENV_LINE" ~/.bashrc || echo "$ENV_LINE" >> ~/.bashrc
eval "$ENV_LINE"
mkdir -p "$TMPDIR"

if ! command -v ffmpeg >/dev/null; then
  echo "== ffmpeg"
  apt-get update -qq || true   # one AMD apt mirror 503s; the Ubuntu ones still work
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg < /dev/null > /workspace/ffmpeg.log 2>&1
fi

# Install straight into the image's own env (/opt/venv) so its ROCm torch is used as-is.
. /opt/venv/bin/activate
PIP="pip install -q --timeout 300 --retries 10"

if ! python -c "import lerobot" 2>/dev/null; then
  echo "== lerobot 0.6.2 from /workspace/lerobot_src.tgz"
  [ -f lerobot_src.tgz ] || { echo "MISSING /workspace/lerobot_src.tgz"; exit 1; }
  rm -rf lerobot && tar -xzf lerobot_src.tgz
  TORCH_PIN="torch==$(python -c 'import torch; print(torch.__version__)')"
  # Keep the image's ROCm torch: forbid pip from swapping it for a CUDA/CPU build.
  echo "$TORCH_PIN" > /workspace/torch_pin.txt
  python -c "import torchvision; print('torchvision==' + torchvision.__version__)" >> /workspace/torch_pin.txt
  $PIP -c /workspace/torch_pin.txt -e "./lerobot[dataset,training]" wandb
fi

TRAIN_PY=lerobot/src/lerobot/scripts/lerobot_train.py
sed -i 's/torch.backends.cudnn.benchmark = True/torch.backends.cudnn.benchmark = False  # ROCm: avoid MIOpen exhaustive search/' "$TRAIN_PY"

echo "== checks"
python -c "import torch, lerobot; print('torch', torch.__version__, '| gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torchcodec; print('torchcodec', torchcodec.__version__)" 2>&1 | tail -1
grep -c "cudnn.benchmark = False" "$TRAIN_PY" | sed 's/^/benchmark-off lines: /'
ffmpeg -version | head -1
python -c "from huggingface_hub import whoami; print('HF login:', whoami()['name'])" 2>/dev/null || echo "HF login: NOT LOGGED IN (run: hf auth login)"
echo "SETUP OK"
