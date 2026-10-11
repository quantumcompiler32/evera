# Shared setup for the pi0.5 scripts on the GMKtec EVO-X2 (Radeon 8060S iGPU, gfx1151, torch ROCm 7.2 wheels).
# Sourced by train.sh and rollout.sh; not meant to be run directly.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# rocBLAS outran hipBLASLt on gfx1151 bf16 GEMMs in our microbenchmark (~22 vs ~16 TFLOPS on Gemma MLP shapes).
export TORCH_BLAS_PREFER_HIPBLASLT=0

HF_USER=Neevdharod
DATA_DIR=~/hackathon/data
OUT_DIR=~/pi05/outputs
TASK="${TASK:-pick up the screwdriver and place it in the bin}"

# Same stable port, robot id and camera keys as hackathon/robot/record.sh, so rollout observations
# have exactly the names and resolution the policy was trained on.
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125836-if00}"
CAMS="{scene: {type: zmq, server_address: 127.0.0.1, port: 5555, camera_name: top, width: 320, height: 240, fps: 30, warmup_s: 3}, wrist: {type: zmq, server_address: 127.0.0.1, port: 5556, camera_name: wrist, width: 320, height: 240, fps: 30, warmup_s: 3}}"
