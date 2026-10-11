# Team Context: UCSC AI Hackathon, AMD Physical AI Track

Read this first. It covers what we're building, what's already done, and what the **object detection** work needs to plug into.

## The event
- **UCSC AI Hackathon @ SF Tech Week**, Oct 10–11, 2026, at UCSC Silicon Valley (Santa Clara).
- **Hard deadlines:** the venue closes at **10 PM Saturday** (no overnight). **Devpost is due at 1:30 PM Sunday**, with our table number. Judging starts at **2:00 PM**: a 3-minute pitch plus 3 minutes of Q&A.
- **Judging:** technical execution, creativity, impact, demo/presentation. **Every team member must be able to explain how the whole thing works**, or we don't place.
- **To win:** AMD requires winners to publish their **code, datasets, trained models and a demo video** on GitHub.

## AMD track: the two missions
1. **Mission 1, pick and place:** set up LeRobot end to end on AMD hardware, and have the arm pick a small object (LEGO brick, pen, game piece) and put it in a bin. **This is the current focus.**
2. **Mission 2, freestyle:** our own real-world robot application on the same stack. **Object detection mostly matters here.**

## Hardware
| Thing | Role |
|---|---|
| **SO-101 arms** (leader + follower) | Follower = the robot with the gripper. Leader = the one a human moves to teleoperate it. **Clamped to the table, not mobile.** |
| **GMKtec mini PC** (Ryzen AI Max+, Radeon iGPU, **running Windows**) | Plugs into the arms and cameras. Records data, runs the trained policy. A teammate drives it over SSH. |
| **AMD Radeon Cloud** (radeon-global.anruicloud.com) | GPU machine with Jupyter for training. 1 credit/hour, so stop it when it's idle. |
| Cameras | Two USB cameras: **top** (bird's-eye) and **side**. Each must be on its own USB port. |
| Optional | ESP32s, LEDs, breadboard parts, a possible down-facing projector (see the Mission 2 ideas) |

## Software stack (pinned, don't change versions)
- **LeRobot v0.4.1**, **Python 3.10**, **PyTorch 2.7.1** (LeRobot needs torch < 2.8), **ffmpeg 7.1.1** (required: training crashes without it).
- Policy: **ACT** = a ResNet18 CNN image encoder plus a transformer that predicts chunks of 100 future joint positions. It's trained by imitation from about 50 human teleop demos.
- The **Hugging Face** account holding datasets and models is **`Neevdharod`** (capital N). Dataset name: `Neevdharod/so101_pick_place`.
- On Windows, arm ports are `COM3`/`COM4`, not `/dev/ttyACM*`, and inference runs with `--policy.device=cpu`.

## Who does what
| Person | Owns |
|---|---|
| Teleop teammate | GMKtec setup, arm calibration, teleop, **recording datasets**, running models on the arm |
| Neev | **Training** on Radeon Cloud: dataset in, ACT model out |
| **You** | **Object detection**: finding named objects in the camera image and giving their position on the table |

## Status
- ✅ Setup guides read: AMD's repo ([npurusho/AMD_Hackathon](https://github.com/npurusho/AMD_Hackathon/tree/main/robotics/robotics_2026)) and the [Seeed SO-ARM wiki](https://wiki.seeedstudio.com/lerobot_so100m_new).
- ✅ Training scripts written and **tested**: a 20-step ACT run on a public SO-101 dataset worked on a Mac (loss went 57 → 10, checkpoint saved).
- ✅ Dataset checker works (`training/check_dataset.py`).
- ⏳ Arm teammate connected to the GMKtec and is working on teleop and recording.
- ⏳ Radeon Cloud logged in; the instance isn't launched yet.
- ⏳ No dataset uploaded yet.

## Repo layout
```
CONTEXT.md                  ← this file
RUNBOOK.md                  ← full arm setup: calibrate, teleop, record, train, run (+ error fixes)
setup/edge_setup_ubuntu.sh  ← Linux setup (we're on Windows now; the RUNBOOK has notes)
training/README.md          ← training checklist
training/train.sh           ← smoke | real <dataset> | resume <job>
training/check_dataset.py   ← validates a HF dataset before training (stdlib only)
training/push_checkpoint.sh ← share an in-progress checkpoint with the arm team
```

---

## Object detection: what you need to know

### Why it matters
**ACT doesn't understand language or object names.** The `single_task` text we record is ignored by ACT. It just imitates whatever motion it was trained on, given the camera images. So for anything like "pick up *the red one*" or "bring me *the pen*," something else has to decide **which object** and **where it is**. That's your part.

The detector's output can be used two ways in Mission 2:
1. **Scripted picks (most reliable):** detection gives a table position, then inverse kinematics moves the arm there for a fixed grasp.
2. **Policy selection:** detection decides which trained ACT skill to run, or confirms the task succeeded (the object is now in the bin).

### Suggested approach
1. **Prototype on a laptop with a webcam.** Use an **open-vocabulary detector** so you can name objects by text without training: **YOLO-World** through `ultralytics`:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8s-worldv2.pt")
   model.set_classes(["lego brick", "pen", "bin"])
   results = model.predict(source=0, show=True)  # webcam
   ```
   If it misses our specific objects, fall back to a small custom-trained YOLOv8 (about 50–100 labeled photos is enough for 3–4 objects).
2. **Pixel to table coordinates.** Put **4 ArUco markers** at known spots on the table. OpenCV's `cv2.aruco` finds them, and `cv2.findHomography` maps pixels to table millimeters. This only works for points on the flat table surface, which is fine for objects lying on it. Measure where the **arm base** sits relative to the markers, so positions can be given in the arm's frame.
3. **Output format.** Agree on this with the arm side early:
   ```json
   {"t": 1728590000.12, "objects": [
     {"label": "pen", "conf": 0.87, "bbox_px": [x1, y1, x2, y2],
      "center_px": [u, v], "table_mm": [x, y]}
   ]}
   ```
   Publish it as a small local HTTP/JSON endpoint or a UDP message, whichever is easiest for the arm code to read.
4. **AMD edge angle (stretch, good for judging).** Export to ONNX and run on the **Ryzen AI NPU** with AMD's Ryzen AI Software (ONNX Runtime + Vitis AI EP). AMD has a Jan 2026 guide for exactly YOLO-World: [Deploying object detection on AMD AI PC](https://www.amd.com/en/developer/resources/technical-articles/2026/deploying-object-detection-model-on-amd-ai-pc.html), with examples in [RyzenAI-SW](https://github.com/amd/RyzenAI-SW). Notes: NMS runs on the CPU, quantization (AMD Quark) needs about 100–1000 calibration images from our setup, and **check that the Ryzen AI Max+ NPU and our Windows version are supported** before investing time. Plain ONNX Runtime on CPU is the fallback.

### Gotchas
- **Only one program can open a camera at a time.** If LeRobot is recording or running a policy with the top camera, your detector can't also open it. Options: use a **separate third camera** for detection (simplest), or grab frames only when the policy isn't running.
- **Don't move the cameras** once the arm team starts recording. The trained policy depends on the exact camera placement. If you need a different view, add a camera rather than moving theirs.
- **Lighting changes** at the venue (day vs. evening) affect detection confidence. Test at both times, and keep detection thresholds adjustable.
- Fix your camera mount and markers with tape, because one bump breaks the calibration. Add a one-key "recalibrate" in your script.

## Mission 2 ideas (not locked yet)
The arm is fixed to the table. The leading idea is **"Reach Desk," an assistive fetch station**:
- A voice or big-button command ("bring me the pen") goes to **object detection (yours)**, then the arm picks the item and hands it over.
- **ESP32 stations** with load cells and LEDs under each item slot confirm the item was actually taken, which closes the loop with a real sensor.
- **Stretch:** a **down-facing projector** draws labels on detected objects and previews the arm's planned path on the table ("Tony Stark bench"). This reuses your detection and homography directly: the same pixel-to-table mapping plus a table-to-projector mapping.
- **Backup demo:** the arm plays a simple board game, with detection reading the board.

Impact story: a low-cost assistive robot for people with limited mobility or speech.

## Quick links
- AMD hackathon repo: https://github.com/npurusho/AMD_Hackathon/tree/main/robotics/robotics_2026
- Seeed SO-101 wiki: https://wiki.seeedstudio.com/lerobot_so100m_new
- LeRobot docs: https://huggingface.co/docs/lerobot/index (ask questions at https://deepwiki.com/huggingface/lerobot)
- AMD YOLO-World on the NPU: https://www.amd.com/en/developer/resources/technical-articles/2026/deploying-object-detection-model-on-amd-ai-pc.html
- Radeon Cloud (training): https://radeon-global.anruicloud.com/
