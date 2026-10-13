# Architecture

evera has four parts: recording demonstrations, training policies on them, running a policy on the arm
from the iPad, and segmenting tools in the overhead camera. Everything that touches hardware runs on the
AMD NucBox. Training runs either on the NucBox's iGPU or on AMD Radeon Cloud.

## Machines

| Machine | Role | Software |
|---|---|---|
| **NucBox** (GMKtec EVO-X2: Ryzen AI Max+, Radeon 8060S gfx1151, 64 GiB VRAM carve-out) | Arms, cameras, dashboard, recording, policy runs, π₀.₅ training, vision inference | Ubuntu 26.04, ROCm 7.2, conda env `lerobot` (LeRobot 0.6.2 editable at `~/lerobot`, torch 2.11+rocm7.2, Python 3.12) |
| **Radeon Cloud** (Jupyter, 1× gfx1100) | ACT and π₀.₅ training | The image's ROCm torch plus LeRobot 0.6.2. The disk is wiped on every launch, and GitHub is unreachable from it. |
| **iPad** | evera front end | Safari, added to the home screen |
| **Laptop** | Development, deploys to the NucBox | Any. `pi05/sync.sh` and `scp` copy code across. |

## Cameras

```text
Limelight 3A (USB network, 172.29.0.1)
  ├─ :5800 MJPEG  ─▶ limelight_zmq.py --name top    ─▶ ZMQ :5555, 320x240 ─▶ dataset camera "scene"
  └─ :5802 clean  ─▶ limelight_zmq.py --name camera1 ─▶ ZMQ :5557, 640x480 ─▶ vision
                  └─▶ vision/live_masks.py, capture_sequence.py (read :5802 directly)
icSpring USB cam (/dev/v4l/by-id/...) ─▶ limelight_zmq.py --name wrist ─▶ ZMQ :5556, 320x240 ─▶ dataset camera "wrist"
```

LeRobot can't open `http://` camera URLs, so every camera goes through a small ZMQ bridge
(`robot/limelight_zmq.py`). The bridge keeps only the newest frame and exits when frames stall, and
`robot/start_limelight.sh` restarts it in a loop. Because the dashboard, `lerobot-record` and the policy runner
all subscribe to the same bridges, none of them fights the others for a camera device.

`start_limelight.sh` also reapplies a fixed camera profile on every start: a 640x480 viewfinder pipeline,
60 Hz flicker correction, overlays hidden, and the wrist camera's exposure and white balance locked. Policies
are sensitive to the image domain, so recording and inference must use the same profile and the same camera
placement.

## Recording

```text
leader arm ──teleop──▶ follower arm
                          │ joint positions (6)
                          ▼
dashboard.py ──pty──▶ record.sh ──▶ lerobot-record ──▶ ~/hackathon/data/<dataset> ──upload──▶ HF Neevdharod/<dataset>
   ▲  buttons send LeRobot's n / r / q keys                                             (bad episodes ─▶ <dataset>_rejects)
   └─ live previews from the ZMQ bridges
```

- Arms are addressed by `/dev/serial/by-id/` because `/dev/ttyACM0/1` swap on replug. Calibration ids are
  `my_follower` and `my_leader`.
- `record.sh` runs a 12-servo preflight before every session and relaxes the follower afterwards.
- Episodes are marked bad in `<dataset>.bad`. `review_episodes.py` scores grasp, release, drop spot and
  rest pose from the joint log, and `upload` splits good and bad episodes into separate Hub datasets.
- Datasets record at 30 fps with two 320x240 cameras (`scene`, `wrist`) and the 6 follower joints.

## Training: two policy families

| | ACT (`training/`) | π₀.₅ (`pi05/`) |
|---|---|---|
| What it is | ResNet18 image encoder plus a transformer CVAE that predicts 100-step action chunks | A 3B PaliGemma vision-language model plus an action expert (flow matching) |
| Reads the task text? | **No.** It imitates whatever motion the demos show. | **Yes.** The task text must match the dataset's. |
| Where it trains | Radeon Cloud (`train.sh`). Offline AWR fine-tune on the NucBox (`rl_finetune.py`). | The NucBox iGPU (`train.sh`) or Radeon Cloud (`cloud_train.sh`) |
| Cost | Minutes to hours | ~5.6 s/step at batch 8 on the 8060S, bf16 only |
| Inference on the NucBox | `lerobot-rollout` through `record.sh policy` | `lerobot-rollout` with RTC (`pi05/rollout.sh`), ~545 ms per 50-step chunk |

ACT is the baseline implementation for a single fixed task. π₀.₅ is the route to one model that follows several
spoken instructions, which is what evera's tool modes need eventually. Each mode in `evera/actions.js` carries
both a task text and a checkpoint, so a mode can point at either kind of policy.

## Running a policy from the iPad

```text
iPad evera/ ──POST /api/policy {ckpt, task, secs}──▶ dashboard.py ──▶ record.sh policy <ckpt> <secs>
       │                                                                 └─▶ lerobot-rollout (follower only, max 8°/step)
       ├──GET /api/status (polling)──▶ run state, camera frames
       └──POST /api/key {k:"q"}──▶ STOP: ends the run, and the arm returns to its start pose
```

- The dashboard serves the app as static files at `/evera/` and sends CORS headers, so the app also works
  from a dev server or a tunnel.
- The app has no build step. three.js is vendored, the URDF and STL meshes ship in `evera/assets/so101/`,
  and OpenAI and ElevenLabs keys are typed in on the iPad and never committed.
- Voice: OpenAI transcription, then a GPT tool call picks the tool and mode, then ElevenLabs speaks the reply. The 3D arm uses scripted animation, not measured joint telemetry.
  "Stop", "halt" and "freeze" are matched on the iPad itself, without waiting for GPT.

## Vision: tool segmentation

Goal: a class, a confidence and a pixel mask for each visible Raspberry Pi, Arduino and Pinecil in the
overhead view. The code doesn't drive the arm. Masks alone don't give grasp points or 3D positions; using
them for arm control also needs a camera-to-robot calibration, a known work plane, and separate safety
testing.

```text
Limelight :5802 clean feed ─▶ YOLO11n-seg ONNX on the NucBox CPU (vision/onnx_segmenter.py)
                           ─▶ per-instance masks, classes, scores ─▶ browser overlay (vision/live_masks.py)

Captured sessions ─▶ SAM box-prompted mask proposals ─▶ per-mask review ─▶ grouped train/val/test splits
                  ─▶ fine-tune on an AMD GPU ─▶ ONNX export ─▶ held-out Limelight evaluation
```

- The live model is provisional. The current ONNX file still has a fourth `wire` class, which the live view
  filters out. The next supported class order appends `tape_measure` and `multimeter`; trained six-class
  weights and held-out results are not included in this repository.
- The Torchvision Mask R-CNN code is an untrained comparison baseline. SAM only proposes training masks.
- Gates before anyone relies on it: per-class precision, recall and mask AP; false positives on an empty bench;
  live CPU latency on the NucBox; all measured on a capture session that wasn't used for training.

Details and commands are in [vision/README.md](../vision/README.md). Model options are in
[research/evera-segmentation-options.md](research/evera-segmentation-options.md).
