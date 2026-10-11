# Handoff: current state (Sat Oct 10, ~6:40 PM)

Read this first if you're picking up on a new computer (or a new Claude session).
Background on the event/plan: `TEAM_CONTEXT.md`. Arm setup reference: `ARM_RUNBOOK.md`.

## Where things run
| Machine | Address | What's on it |
|---|---|---|
| **GMKtec NucBox EVO-X2** (Ubuntu 26.04, Ryzen AI Max+, Radeon 8060S, LeRobot **0.6.2** in conda env `lerobot`) | `ssh aup@<GMKTEC_IP>` | Arms, cameras, recording, **dashboard**, all scripts in `~/hackathon/` |
| AMD Radeon Cloud (Jupyter) | radeon-global.anruicloud.com | Training only. **Disk is wiped on every launch.** Stop the instance when idle (1 credit/h) |
| Any laptop | browser | Dashboard: **http://<GMKTEC_IP>:8080** |

## evera iPad app (Mission 2 demo UI)
- `evera/`: a JARVIS-style iPad web app with a 3D SO-101, tool orbs, voice (OpenAI + ElevenLabs) and a STOP button. Each tool mode runs through `/api/policy`.
- Served by `robot/dashboard.py` at `/evera/`. That file also sends CORS headers.
- Deploy, tunnel (HTTPS for the mic), iPad setup, and how to point modes at new checkpoints: **`evera/README.md`**. Original build spec: `docs/EVERA_PROMPT.md`.

## Recording dashboard (the main tool)
- http://<GMKTEC_IP>:8080 shows live top + wrist cameras and START SESSION / DONE-NEXT (Space) / REDO (R) / STOP (Q) / Check motors / Mark bad / Upload.
- Restart it: `ssh aup@<GMKTEC_IP> 'cd ~/hackathon && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && kill $(cat ~/dashboard.pid); setsid nohup python robot/dashboard.py > ~/dashboard.log 2>&1 < /dev/null & echo $! > ~/dashboard.pid'`
- Camera bridges (must be running): `bash ~/hackathon/robot/start_limelight.sh` (logs `~/cam_top.log`, `~/cam_wrist.log`; they auto-restart on stalls)
- CLI equivalent: `bash ~/hackathon/robot/record.sh one|session N|bad|status|upload|check|practice`

## Hardware facts (learned the hard way)
- **Follower (gripper arm)** = USB serial `5C4C125836`, **leader (handle arm)** = `5C4C128274`. Scripts use `/dev/serial/by-id/...` because `/dev/ttyACM0/1` swap on replug.
- Calibration ids: `my_follower`, `my_leader`. If LeRobot ever says "Mismatch between calibration values... Press ENTER", **don't press Enter blindly**: it writes that file's calibration to whatever arm is on that port.
- Cameras: **top = Limelight 3A** (USB network device at 172.29.0.1, MJPEG on :5800) via `robot/limelight_zmq.py` -> ZMQ port 5555; **wrist = icSpring USB cam** (`/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0`) -> ZMQ 5556. Both 320x240. In datasets the cameras are named `scene` and `wrist`.
- **Camera profile (tuned Oct 10 evening, reapplied by `start_limelight.sh` on every start):**
  - Limelight: viewfinder pipeline at 640x480 @ 90 fps processing (was 1280x960 @ 22), 60 Hz flicker correction, crosshair and fps overlay hidden. Its MJPEG stream still tops out at ~24 fps.
  - Wrist cam: 60 Hz anti-flicker (was 50 Hz), fixed exposure 110 (11 ms) and white balance 3000 K (both were auto). 0% clipped pixels, down from 4.5%.
  - The wrist cam only delivers **20 fps** in every mode (hardware limit).
  - Tools: `robot/limelight_ws.py get|set|decorations|net` and `robot/v4l2_ctl.py <dev> [name=value]`.
  - **Don't restart the Limelight's vision server**: it once came back on an AprilTag pipeline (grayscale, LED on). If that happens, rerun `start_limelight.sh`.
- Both arms read **~12.3 V**. Seeed says the leader should be 5 V on Pro kits: confirm with the AMD desk.
- **The gripper overloads if it squeezes at the end of an episode**, which locks motor #6. Always release (open) before the episode ends. Fix: unplug the follower's power barrel 5 s.
- Whole USB hub has dropped out a few times: keep arms plugged directly into the GMKtec if possible; don't touch cables while recording.

## Status right now
- ⛔ **BLOCKER: follower gripper motor #6 not answering on any ID** (after an overload). Next: power-cycle follower → reseat the 3-pin cable between motor #5 (wrist roll) and #6 (gripper) → if still dead, get a spare STS3215 or a spare follower arm from the AMD hardware desk (new servo needs ID 6 + gripper recalibration).
- Old dataset `Neevdharod/so101_pick_place` (local `~/hackathon/data/so101_pick_place`, 32+ eps): **don't train on it.** Review of eps 0–31: grasps are fine in 30/32, but the screwdriver is dropped somewhere different every time (release pan std 22°, no bin), eps 1/6/7/8/12 end still squeezing the gripper, and the task text says "lego brick".
- **New dataset `Neevdharod/so101_screwdriver_bin`** (record.sh + dashboard default): task "pick up the screwdriver and place it in the bin", 25 s cap. Target 50 good.

## Recording protocol (what makes the data good)
1. **One taped-down bin**, the same spot every episode. A consistent drop spot matters most: ACT averages demos that disagree.
2. **4–5 taped start spots** for the screwdriver, ~10 demos each, handle pointing the same way.
3. Same rest pose at start and end. **Open the gripper wide** (the old data opened only ~10/100) and fully release before DONE.
4. One smooth reach → grab → lift → drop → return. REDO anything fumbled.
5. Clear table, no people or hands in view, same lighting. Wrist cam exposure/WB are now locked by the bridge; after a restart, copy the `Pin them:` line from `~/cam_wrist.log` so every session uses the same values.
6. Every ~10 episodes: `bash ~/hackathon/robot/record.sh review`. It flags NO_GRASP / NOT_RELEASED / PLACE_OFF / START_NOT_REST etc. and writes contact sheets to `~/hackathon/review/`. Aim for drop-spot pan spread < 5°.
7. Optional: turn off the Limelight crosshair in its web UI (http://172.29.0.1:5801 from the GMKtec), **without moving the camera**, before starting the new dataset.
- `Neevdharod/test_run` (2 eps) on the Hub, private: just a pipeline test.
- Hugging Face on the GMKtec is logged in as **Neevdharod**.

## Training plan (after ~50 good episodes)
- **ACT** (guaranteed baseline) on Radeon Cloud: upload `lerobot_src.tgz` (GMKtec's exact LeRobot 0.6.2 source) + `training/` → `bash training/cloud_setup.sh` (~1 min; uses the image's torch 2.9.1+rocm) → `bash training/train.sh real so101_screwdriver_bin`. Open issue: torchcodec 0.11 fails to load on the cloud image's torch; fall back to PyAV (`--dataset.video_backend=pyav`) if pinning `torchcodec==0.9.*` doesn't work.
  - LeRobot's train script sets `cudnn.benchmark=True`, which on ROCm stalls for minutes (MIOpen exhaustive search). cloud_setup.sh patches it off.
  - Measured: ~3.3 s/step at batch 64 with 640x480; 320x240 should be much faster.
  - Cloud disk is wiped between launches: push checkpoints to the Hub.
- **Stretch:** pi0.5 fine-tune overnight on the GMKtec (69 GB GPU memory), or try **FLUX 3 Action SO-101** zero-shot (LeRobot 0.6.2 supports `flux3`; ~39 GB download; expects `scene` + `wrist` cameras).

## Files
- `robot/`: GMKtec-side: `dashboard.py`, `record.sh`, `limelight_zmq.py`, `start_limelight.sh`, `check_arms.py`, `test_camera.py`
- `training/`: cloud-side: `cloud_setup.sh`, `train.sh`, `check_dataset.py`, `push_checkpoint.sh`, `bench.py`
- `evera/`: iPad control app (served by `dashboard.py` at `/evera/`)
- `robot/calibration/`: backup of the `my_follower` / `my_leader` calibration files from the GMKtec
- `docs/TEAM_CONTEXT.md` (team context, detection plan), `docs/ARM_RUNBOOK.md` (setup steps), this file.
