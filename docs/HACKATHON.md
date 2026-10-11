# Hackathon context

## The event

- **UCSC AI Hackathon @ SF Tech Week**, Oct 10–11, 2026, at UCSC Silicon Valley (Santa Clara). AMD Physical AI track.
- **Deadlines:** the venue closes at 10 PM Saturday (no overnight). **Devpost is due at 1:30 PM Sunday**, with
  our table number. Judging starts at 2:00 PM: a 3-minute pitch plus 3 minutes of Q&A.
- **Judging:** technical execution, creativity, impact, demo and presentation. **Every team member must be
  able to explain how the whole system works.**
- **To win,** AMD requires publishing the **code, datasets, trained models and a demo video**. Template:
  [AMD project template](https://github.com/AlexHe99/AMD_Robotics_Hackathon_2025_ProjectTemplate).

## The two missions

1. **Mission 1, pick and place:** run LeRobot end to end on AMD hardware, and have the arm pick up a small
   object and put it in a bin. Our version: a screwdriver into a taped-down bin, with ACT trained on our own
   teleop demos ([training/](../training/README.md)).
2. **Mission 2, freestyle:** our own real-world application on the same stack. Ours is **evera**, a bench
   assistant for people working at an electronics bench. You ask for a tool by voice or touch, and the arm
   fetches it ([evera/](../evera/README.md)). Tool segmentation in [vision/](../vision/README.md) and the
   language-conditioned π₀.₅ policy in [pi05/](../pi05/README.md) are the steps toward choosing *which* tool
   from a spoken request.

Impact story: a low-cost assistant that hands over tools for people with limited reach or mobility, and an
extra pair of hands for everyone else at the bench.

## What runs on AMD

- **Ryzen AI Max+ / Radeon 8060S (NucBox):** recording, policy inference, π₀.₅ fine-tuning in bf16 on the iGPU
  (~64 GiB of unified VRAM), and vision inference on the CPU through ONNX Runtime.
- **Radeon Cloud (gfx1100):** ACT training with ROCm.

Measured numbers are in [pi05/README.md](../pi05/README.md#measured-on-the-radeon-8060s-gfx1151) and
[training/README.md](../training/README.md#measured-speed).

## Q&A cheat sheet

- **ACT:** a ResNet18 encodes each camera image, and a transformer CVAE predicts the next 100 joint positions
  at once. Chunking gives smooth motion and fewer compounding errors. It learns by imitation from ~20–50 demos.
  **It ignores the task text**, so on its own it can't choose between objects.
- **π₀.₅:** a PaliGemma vision-language model plus an action expert. It does read the instruction, which is
  why it's the route to "pass me the *multimeter*". We fine-tune only the action expert (0.69B of 4.1B params).
- **Why the offline RL experiment:** reward-weighting demos by quality (AWR) sounded like a free win. Offline,
  it didn't beat a plain fine-tune on cleaner episodes. Data quality mattered more than the training objective.
- **Why ZMQ camera bridges:** LeRobot can't read the Limelight's HTTP stream, and only one program can open a
  USB camera. One bridge per camera lets the dashboard, the recorder and the policy all share the frames.
- **Safety:** policy runs cap each joint at 8° per step, STOP is on the iPad and the dashboard, and "stop" is
  matched on the iPad without a round trip to GPT.

## Links

- AMD hackathon repo: https://github.com/npurusho/AMD_Hackathon/tree/main/robotics/robotics_2026
- Seeed SO-101 wiki: https://wiki.seeedstudio.com/lerobot_so100m_new
- LeRobot docs: https://huggingface.co/docs/lerobot/index
- Radeon Cloud: https://radeon-global.anruicloud.com/
