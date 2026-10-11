# Setup

One-time setup for the NucBox, the arms, the cameras, and the training machines. For day-to-day operation
and the current state of things, see [HANDOFF.md](HANDOFF.md).

## 1. NucBox software

The NucBox runs Ubuntu 26.04 with the BIOS GPU carve-out set to 64 GiB, so torch sees ~64 GiB of VRAM.
ROCm comes from the PyTorch ROCm 7.2 wheels. Everything lives in one conda env, `lerobot`.

```bash
# Miniforge, then the env
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n lerobot python=3.12 ffmpeg -c conda-forge
conda activate lerobot

# LeRobot 0.6.2 from source (the NucBox is on commit b9cb121c), editable
git clone https://github.com/huggingface/lerobot ~/lerobot && cd ~/lerobot
pip install -e ".[feetech,smolvla,pi]"

# Replace the default torch with the ROCm 7.2 build
pip install --force-reinstall --no-deps "torch==2.11.0" "torchvision>=0.26,<0.27" \
  --index-url https://download.pytorch.org/whl/rocm7.2
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

- The `pi` extra pulls in `scipy` and `sentencepiece` for π₀.₅.
- `scripts/edge_setup_ubuntu.sh` is AMD's reference setup (Ubuntu 24.04, ROCm 6.3, LeRobot 0.4.1). The
  NucBox doesn't use it. Policies and datasets must match the NucBox's LeRobot version, which is 0.6.2.
- On the 8060S, bf16 is ~10× faster than fp32. Train with `--policy.dtype=bfloat16`.

Log in to Hugging Face with the account that owns the datasets (`Neevdharod`). For π₀.₅, that account must
also have accepted the license for the gated [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) model.

```bash
hf auth login
```

## 2. Deploy this repo's code to the NucBox

The scripts expect fixed paths on the NucBox. From a laptop checkout:

```bash
GMK=aup@<GMKTEC_IP>
rsync -av robot training evera "$GMK:hackathon/"
GMK=$GMK bash pi05/sync.sh
rsync -av vision vision_training "$GMK:evera-vision/"
```

| Code | Path on the NucBox |
|---|---|
| `robot/`, `training/`, `evera/` | `~/hackathon/` |
| `pi05/` | `~/pi05/` |
| `vision/`, `vision_training/` | `~/evera-vision/` (its own `.venv`, see [vision/README.md](../vision/README.md)) |

## 3. Arms

Seeed Pro kits run the follower at **12 V** and the leader at **5 V**. A wrong supply can damage the servos.
Both arms on our bench read ~12.3 V, so check with the hardware desk before swapping supplies.

The `/dev/ttyACM0/1` names swap whenever a cable is replugged, so everything uses the stable by-id paths.
Keep them in `~/arms.env`:

```bash
FOLLOWER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125836-if00
LEADER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128274-if00
```

Calibrate once per arm. The calibration file is stored under the id, so use the same ids everywhere:

```bash
source ~/arms.env
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=my_leader
```

Calibrations are saved under `~/.cache/huggingface/lerobot/calibration/`. The repo keeps a backup in
`robot/calibration/`. If LeRobot ever says "Mismatch between calibration values… Press ENTER", **don't press
Enter blindly**: it writes that file's calibration to whatever arm is on the port.

Check both arms (read-only) with `python ~/hackathon/robot/check_arms.py`, or `bash ~/hackathon/robot/record.sh check`.

## 4. Cameras

Mount the Limelight 3A rigidly above the bench and the USB camera on the follower's wrist. Then start the
bridges:

```bash
bash ~/hackathon/robot/start_limelight.sh      # top -> ZMQ 5555, wrist -> ZMQ 5556
tail ~/cam_top.log ~/cam_wrist.log
python ~/hackathon/robot/test_camera.py        # grabs a frame through LeRobot's camera classes
```

The script applies the Limelight and wrist camera profiles every time it runs (see
[ARCHITECTURE.md](ARCHITECTURE.md#cameras)). Use `robot/limelight_ws.py` and `robot/v4l2_ctl.py` to inspect
or change settings. **Don't move a camera** after recording starts: a trained policy depends on its exact placement.

## 5. Dashboard

```bash
cd ~/hackathon && conda activate lerobot
setsid nohup python robot/dashboard.py > ~/dashboard.log 2>&1 < /dev/null & echo $! > ~/dashboard.pid
```

Open `http://<GMKTEC_IP>:8080` for recording and `http://<GMKTEC_IP>:8080/evera/` for the iPad app. To restart
it, `kill $(cat ~/dashboard.pid)` and run the same command. Don't restart it while a session is recording.
For HTTPS (the iPad microphone needs it), see [evera/README.md](../evera/README.md#https-for-voice-needed-on-the-ipad).

## 6. Training machines

- **Radeon Cloud, ACT:** [training/README.md](../training/README.md). `training/cloud_setup.sh` installs LeRobot
  from `lerobot_src.tgz`, an archive of the NucBox's `~/lerobot` that isn't committed. Make it with
  `tar czf lerobot_src.tgz --exclude=.git -C ~ lerobot` (it must unpack to `lerobot/`).
- **Radeon Cloud, π₀.₅:** upload `pi05/cloud_train.sh` and run it. See [pi05/README.md](../pi05/README.md).
- **NucBox, π₀.₅ and offline RL:** they use the `lerobot` env directly. Check `pgrep -af lerobot-record` first:
  the GPU and the box are shared with live recording.

## 7. Laptop

Only needed to edit and deploy code, look at datasets, or run the iPad app against the simulator:

```bash
python3 -m http.server 8765 --directory evera                 # http://localhost:8765/?sim=1
python3 training/check_dataset.py Neevdharod/so101_pick_place # stdlib only
```

`scripts/calibrate_so101.sh [follower|leader] [id]` calibrates an arm plugged into a laptop. It needs a repo
`.venv` with `pip install 'lerobot[core_scripts,training,feetech]'`.

## Known errors

| Error | Fix |
|---|---|
| `Could not connect on port` | Check the by-id path exists (`ls /dev/serial/by-id/`), then `sudo chmod 666` it |
| `Overload` or the gripper doesn't answer | The gripper locked from squeezing at the end of an episode. Unplug the follower's power barrel for 5 s. |
| `Lock` error | Unplug and replug the arm's power |
| `Magnitude 30841 exceeds 2047` | Run the Seeed middle-position tool ([Seeed_RoboController](https://github.com/Seeed-Projects/Seeed_RoboController)), then recalibrate |
| `Could not push packet to decoder` | A corrupted video file. Find it with `ffmpeg -v error -i file.mp4 -f null -`. Never Ctrl-C while LeRobot writes video. |
| Camera bridge log stops updating | It restarts itself. If the Limelight came back grayscale with its LED on, rerun `start_limelight.sh`. |
| `RevisionNotFoundError` / "must be tagged with a codebase version" | The dataset lacks its version tag. `lerobot-record` adds it; for a manual upload, see `training/check_dataset.py`. |
| ACT training stalls for minutes before step 1 on ROCm | MIOpen exhaustive search. `training/cloud_setup.sh` turns `cudnn.benchmark` off and sets `MIOPEN_FIND_MODE=FAST`. |
| torchcodec fails to load or aborts with `bad_alloc` | Decode with PyAV: `--dataset.video_backend=pyav` |
