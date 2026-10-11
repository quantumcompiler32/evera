#!/usr/bin/env bash
# Edge setup for the GMKtec (Ryzen AI Max+) — drives the SO-101 arms and runs inference.
# Based on AMD's hackathon QuickStart: Ubuntu 24.04 + ROCm 6.3 + PyTorch 2.7.1 + LeRobot v0.4.1.
# Reference only: the NucBox actually runs Ubuntu 26.04 + ROCm 7.2 + LeRobot 0.6.2 (see docs/SETUP.md).
#
# Run in two stages (stage 1 needs a reboot):
#   bash edge_setup_ubuntu.sh rocm     # installs ROCm userspace, then reboot
#   bash edge_setup_ubuntu.sh lerobot  # conda env, torch, lerobot, checks
set -euo pipefail

STAGE="${1:-}"
ENV_NAME=lerobot
LEROBOT_TAG=v0.4.1

if [[ "$STAGE" == "rocm" ]]; then
  sudo apt update
  sudo apt install -y "linux-headers-$(uname -r)" python3-setuptools python3-wheel git
  sudo usermod -a -G render,video,dialout "$LOGNAME"
  cd /tmp
  wget -nc https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
  sudo apt install -y ./amdgpu-install_6.3.60304-1_all.deb
  amdgpu-install -y --usecase=rocm --no-dkms   # --no-dkms: keep the built-in kernel driver
  echo "ROCm installed. Reboot now, then run: bash $0 lerobot"
  exit 0
fi

if [[ "$STAGE" != "lerobot" ]]; then
  echo "usage: $0 rocm|lerobot"; exit 1
fi

# Ryzen AI Max+ iGPU is gfx1151, which ROCm 6.3 doesn't officially list.
# The override makes it present as gfx1100. If torch still can't see the GPU,
# that's fine for us: ACT inference runs OK on CPU, and training happens on the MI300X.
grep -q HSA_OVERRIDE_GFX_VERSION ~/.bashrc || echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" >> ~/.bashrc
export HSA_OVERRIDE_GFX_VERSION=11.0.0

if ! command -v conda >/dev/null; then
  cd /tmp
  wget -nc https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
  "$HOME/miniforge3/bin/conda" init bash
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3
conda install -y ffmpeg=7.1.1 -c conda-forge

if [[ ! -d "$HOME/lerobot" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$HOME/lerobot"
fi
cd "$HOME/lerobot"
git checkout -B "$LEROBOT_TAG" "$LEROBOT_TAG"
pip install -e ".[feetech]"
pip install wandb pynput==1.7.7

echo "=== checks ==="
python -c "import torch; print('torch', torch.__version__, '| gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(CPU only)')"
pip list 2>/dev/null | grep -E "^(lerobot|torch) "
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q svtav1 && echo "ffmpeg: libsvtav1 OK" || echo "ffmpeg: libsvtav1 MISSING"
echo "Done. Log out/in once so the dialout/render groups apply, then see docs/SETUP.md."
