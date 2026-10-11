# AMD Track Runbook: LeRobot + SO-101

Source guides: [AMD hackathon repo](https://github.com/npurusho/AMD_Hackathon/tree/main/robotics/robotics_2026) and the [Seeed SO-ARM wiki](https://wiki.seeedstudio.com/lerobot_so100m_new).

Pinned stack (from AMD): **Ubuntu 24.04, ROCm 6.3, PyTorch 2.7.1, LeRobot v0.4.1, Python 3.10.**
LeRobot needs `torch < 2.8`, so don't install newer ROCm PyTorch wheels.

## Who does what

| Machine | Job |
|---|---|
| **GMKtec (Ryzen AI Max+)** on Ubuntu 24.04 | Plugs into the arms and cameras. Records datasets, runs inference. |
| **AMD MI300X cloud (Jupyter)** | Trains the ACT policy. The link is handed out at the event. |
| Laptops | Code, the Devpost write-up, the GitHub repo |

## 0. Accounts (do it now)
- A Hugging Face account and a **write** token
- A Weights & Biases account and API key
- A GitHub repo. Winners must publish their code, datasets, models and demo video. Template: [AMD project template](https://github.com/AlexHe99/AMD_Robotics_Hackathon_2025_ProjectTemplate)

## 1. Edge setup (GMKtec)
```bash
bash setup/edge_setup_ubuntu.sh rocm      # then reboot
bash setup/edge_setup_ubuntu.sh lerobot
```
If `torch.cuda.is_available()` prints False, **don't sink time into it**. ACT inference runs on CPU (`--policy.device=cpu`), and training happens on the MI300X.

## 2. Ports and cameras
Plug in the **leader first** (`/dev/ttyACM0`), then the **follower** (`/dev/ttyACM1`).
```bash
lerobot-find-port
lerobot-find-cameras opencv
sudo chmod 666 /dev/ttyACM*
```
Put the cameras on **different USB ports or hubs**, otherwise video stutters.

## 3. Calibrate (skip motor ID setup: the kits come pre-configured)
Seeed **Pro** kits: the follower runs at **12V** and the leader at **5V**. Using the wrong supply can damage the servos.
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM0 --teleop.id=leader
```
Use the **same ids** everywhere after this: the calibration file is stored under the id.

## 4. Teleop check
```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader \
  --display_data=true
```

## 5. Mission 1: record pick-and-place
```bash
hf auth login
HF_USER=$(hf auth whoami | cut -c 16-)
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/so101_pick_place \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="pick up the lego brick and place it in the bin"
```
Keys: → next episode, ← redo, ESC stop and upload. Arrow keys broken? Log in with **"Ubuntu on Xorg"** (pynput 1.7.7 is already installed).
Never Ctrl-C while it's writing video: that corrupts the dataset. Use `--resume=true` to add episodes.

Data tips: fixed cameras, about 10 episodes per object location, the same grasp style each time, and the object always in view.
Rule of thumb: you should be able to do the task yourself by looking only at the camera feeds.

## 6. Train on MI300X (Jupyter terminal)
Run the cells in AMD's `training-models-on-rocm.ipynb`: ffmpeg 7, then LeRobot v0.4.1, wandb login, HF login. Then:
```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_pick_place \
  --policy.type=act --policy.device=cuda \
  --batch_size=64 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/train/act_pick_place --job_name=act_pick_place \
  --policy.repo_id=${HF_USER}/act_pick_place --policy.push_to_hub=true \
  --wandb.enable=true
```
Start with a 1k-step run to confirm the pipeline works before the long run.

## 7. Run the policy on the arm (GMKtec)
```bash
hf download ${HF_USER}/act_pick_place --local-dir models/act_pick_place
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="pick up the lego brick and place it in the bin" \
  --dataset.repo_id=${HF_USER}/eval_act_pick_place \
  --dataset.episode_time_s=20 --dataset.num_episodes=1 --dataset.push_to_hub=false \
  --policy.path=models/act_pick_place
```
Add `--policy.device=cpu` if the iGPU isn't detected.

## Known errors
| Error | Fix |
|---|---|
| `Could not connect on port` | `sudo chmod 666 /dev/ttyACM*`, check with `lerobot-find-port` |
| `Lock` error | Unplug and replug the arm's power |
| `Magnitude 30841 exceeds 2047` | Run the Seeed middle-position tool ([Seeed_RoboController](https://github.com/Seeed-Projects/Seeed_RoboController)), then recalibrate |
| `Could not push packet to decoder` | A corrupted video file. Find it with `ffmpeg -v error -i file.mp4 -f null -` |
| `FileExistsError` on train | Change `--output_dir` or delete the old one |
| `stack(): argument 'tensors'` | **Don't** follow Seeed's `datasets==2.19` tip: LeRobot v0.4.1 needs `datasets>=4.0,<4.2`. Reinstall with `pip install -e ".[feetech]"` inside `lerobot/` |
| `RevisionNotFoundError` / "must be tagged with a codebase version" | The dataset is missing its `v3.0` tag. `lerobot-record` adds it; for a manual upload, see `training/check_dataset.py` |
