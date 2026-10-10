# Evera workshop vision

This repository contains source for a workshop object detector and an SO-101
setup script. The current vision model detects Raspberry Pi, Arduino, and
soldering iron in camera images and draws bounding boxes in a local browser
preview. See [the vision guide](vision/README.md) for camera, labeling, training,
and Limelight 3A setup.

The detector is fine-tuned from pretrained YOLO26n weights on the AMD NucBox.
Captured images, labels, and trained weights are local artifacts and are not
included in this repository. The current training set is too small for a
reliable accuracy claim or autonomous arm control.

See [the architecture](docs/ARCHITECTURE.md) for the vision pipeline and
[the setup guide](docs/SETUP.md) for the separate LeRobot environment.
