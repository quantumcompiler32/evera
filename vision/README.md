# Workshop object detector

The current model is a fine-tuned Ultralytics YOLO26n detector. Its three classes
are `raspberry_pi`, `arduino`, and `soldering_iron`. `train.py` loads pretrained
weights and trains on 640-pixel images. The AMD computer runs both training and
live inference; the browser displays the MJPEG feed and offers capture and
labeling controls.

The code is here, but captured photos, labels, and model weights stay out of
Git. The first workshop run used only eight distinct labeled scenes, so its
validation result is too small to establish reliable accuracy. Expect board
confusion and false positives until more varied images and negative scenes are
labeled and evaluated on the intended camera.

## Live camera

Run from the project root on the AMD computer:

```bash
.venv/bin/python vision/live_grid.py \
  --camera /dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0 \
  --port 8765 --captures captures/training_candidates \
  --model outputs/detection/workshop/weights/best.pt --device 0 \
  --confidence 0.35 --board-color-refine
```

The color refinement is specific to the green Pi and blue Arduino under the
original overhead camera. Omit `--board-color-refine` for another camera until
its color distribution is checked.

The server listens on remote localhost. From a trusted Mac terminal, forward
`127.0.0.1:8765` to the remote port over SSH, then open
`http://127.0.0.1:8765/`. The live page draws the 3-by-3 grid and detections;
saved training images do not include the overlay. The labeler writes YOLO
bounding boxes as normalized `class x_center y_center width height` lines.

## Limelight 3A camera source

The Limelight 3A is a USB-C network camera with an MJPEG stream. Once it is
connected to the AMD computer and its stream URL is reachable, `live_grid.py`
can read it using `--camera http://<reachable-address>:5800`. A USB connection
may expose `http://172.29.0.1:5800`; confirm the actual address and stream in
the Limelight web interface before starting the server. The interface is
typically on port 5801. This integration has not yet been tested with the
physical Limelight.

Using a new camera changes viewpoint, optics, exposure, and color. Capture and
label fresh Limelight images, including empty scenes and similar boards, then
evaluate and retrain before relying on the live boxes. The existing overhead
camera's color refinement should stay disabled for the Limelight until measured.

## Build and train a dataset

Collect distinct arrangements and draw a box around every visible target. Avoid
burst frames of the same arrangement. Keep the photos and labels outside Git.
`build_dataset.py` expects matching files beneath `captures/training_candidates`
and `captures/training_labels`; it rejects unlabeled captures. If there are
unlabeled experiments in those folders, first copy only curated pairs into
separate capture and label directories and pass `--captures` and `--labels`.

```bash
.venv/bin/python vision/build_dataset.py \
  --captures captures/curated_images --labels captures/curated_labels \
  --output datasets/workshop-v3
.venv/bin/python vision/train.py \
  --data datasets/workshop-v3/dataset.yaml --model yolo26n.pt \
  --device 0 --name workshop-v3
```

The builder makes a seeded train/validation split. For trustworthy metrics,
collect enough scenes to reserve a separate capture session for final testing;
its simple split alone does not prevent related frames landing in both sets.
Compare new weights on held-out Limelight scenes and the three individual
classes before replacing the current live model.

Official references: [Limelight 3A quick-start](https://docs.limelightvision.io/docs/docs-limelight/getting-started/limelight-3a)
and [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26).
