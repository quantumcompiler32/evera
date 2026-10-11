# Evera workshop vision

This repository contains a Limelight 3A live view, a CNN instance segmentation
pipeline, and an SO-101 setup script. The vision model is intended to produce
separate masks for Raspberry Pi boards, Arduino boards, Pinecil soldering irons,
and individual wires. Training and inference run on an AMD NucBox; the camera
feeds images to that computer over USB networking.

See [the vision guide](vision/README.md) for the current dataset, camera setup,
training commands, and accuracy limits. Images and weights are kept out of
Git. A separate held-out Limelight test set is still needed before any reliable
accuracy or autonomous arm claim.

See [the architecture](docs/ARCHITECTURE.md) and [the SO-101 setup guide](docs/SETUP.md).

See [pi05/](pi05/README.md) for fine-tuning and running the π₀.₅ VLA policy on the SO-101.
