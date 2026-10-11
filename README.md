# evera

evera is a voice- and touch-controlled bench assistant built on an SO-101 robot arm and AMD hardware.
Tap a tool on the iPad, or say "evera, pass me the soldering iron", and the arm runs a learned policy
for that action while a live 3D model of the arm mirrors what it's doing.

Built at the UCSC AI Hackathon (SF Tech Week, Oct 10–11, 2026) for the AMD Physical AI track. See
[docs/HACKATHON.md](docs/HACKATHON.md) for the event and the two missions.

![evera on the iPad](docs/images/evera_preview.png)

## How it fits together

```text
 iPad (evera/)  ──HTTP──▶  dashboard.py on the AMD NucBox  ──▶  lerobot-record / policy run  ──▶  SO-101 follower
                              ▲            │                                                        ▲
     Limelight 3A (top) ─ZMQ──┤            └─▶ datasets (Hugging Face) ─▶ ACT on Radeon Cloud ───────┤
     wrist USB cam     ─ZMQ──┘                                          ─▶ π₀.₅ on the NucBox ───────┘
```

- **Record:** a person teleoperates the follower arm with the leader arm. [`robot/dashboard.py`](robot/README.md)
  drives `lerobot-record` and shows both camera feeds.
- **Train:** [`training/`](training/README.md) trains ACT on AMD Radeon Cloud (plus an offline reward-weighted
  fine-tune on the NucBox). [`pi05/`](pi05/README.md) fine-tunes the π₀.₅ vision-language-action model on the
  NucBox's Radeon 8060S.
- **Run:** the dashboard's `/api/policy` runs a checkpoint on the arm. [`evera/`](evera/README.md) is the iPad
  front end for it, served by the dashboard at `/evera/`.
- **See:** [`vision/`](vision/README.md) segments Raspberry Pi boards, Arduinos and Pinecil soldering irons in
  the overhead camera view, on the NucBox CPU through ONNX Runtime.

The full picture, including the cameras and why there are two policy types, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository map

| Path | What's there | Runs on |
|---|---|---|
| [`robot/`](robot/README.md) | Recording dashboard, `record.sh`, camera bridges, arm checks, episode review | NucBox (`~/hackathon/robot`) |
| [`training/`](training/README.md) | ACT training, dataset checks, offline RL fine-tune and evaluation | Radeon Cloud and the NucBox |
| [`pi05/`](pi05/README.md) | π₀.₅ fine-tune, checkpoint check, rollout, dataset subsets | NucBox (`~/pi05`) and Radeon Cloud |
| [`evera/`](evera/README.md) | iPad web app: 3D arm, tool picker, voice, STOP | iPad Safari, served by the dashboard |
| [`vision/`](vision/README.md), `vision_training/` | Instance segmentation: capture, labeling, training, live masks | NucBox (`~/evera-vision`) |
| [`scripts/`](scripts) | Ubuntu edge setup and SO-101 calibration helpers | NucBox or a laptop |
| [`docs/`](docs) | Architecture, setup, current state, hackathon context, research notes | — |

## Hardware

- **SO-101 arms**, a leader and a follower (Seeed kit, Feetech STS3215 servos), clamped to the bench.
- **AMD NucBox (GMKtec EVO-X2):** Ryzen AI Max+, Radeon 8060S iGPU (gfx1151), Ubuntu 26.04, ROCm 7.2.
  Drives the arms and cameras, records, runs policies and the dashboard, and trains π₀.₅.
- **AMD Radeon Cloud:** a Jupyter GPU instance (gfx1100) for ACT training.
- **Cameras:** a Limelight 3A above the bench and a USB camera on the wrist.

## Getting started

1. Set up the NucBox, arms and cameras: [docs/SETUP.md](docs/SETUP.md).
2. Check what state things are in right now: [docs/HANDOFF.md](docs/HANDOFF.md).
3. Record demonstrations with the dashboard: [robot/README.md](robot/README.md).
4. Train a policy: [training/README.md](training/README.md) (ACT) or [pi05/README.md](pi05/README.md) (π₀.₅).
5. Put the iPad app in front of it: [evera/README.md](evera/README.md).

## Status

<!-- TODO(human): one short paragraph on what the arm demonstrably does today on the real hardware -->

Offline numbers for the trained ACT checkpoints are in [training/README.md](training/README.md).
Open problems and next steps are in [docs/HANDOFF.md](docs/HANDOFF.md).

## Credits

- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) for recording, ACT and π₀.₅.
- SO-101 URDF and meshes from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
- [three.js](https://threejs.org) r160 (MIT), vendored in `evera/vendor/`.
- Vision models from [Ultralytics](https://www.ultralytics.com/license) (AGPL-3.0).
