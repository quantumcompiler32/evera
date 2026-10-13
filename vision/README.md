# Workshop instance segmentation

## Next checkpoint: tape measure and multimeter

The append-only YOLO11n-seg class order for the next checkpoint is
`0 raspberry_pi`, `1 arduino`, `2 pinecil`, `3 wire`, `4 tape_measure`,
`5 multimeter`. This supersedes the earlier three-class plan below. Coordinate
tape-measure labels with this exact order before training. The existing
four-class ONNX file remains a separate provisional checkpoint; no six-class
weights have been trained by this change.

Use the existing NucBox `camera1` 640×480 relay and save raw recordings and
weights outside Git. Record a multimeter session with varied positions and
rotations, screen up, partial occlusion, adjacent existing objects, and a few
frames without the meter. Select sharp representative frames with
`vision_training/select_frames.py`. Reserve another whole recording session
for testing; do not split neighboring frames across train and test.

For `vision_training/propose_multimeter.py`, supply JSON mapping **every**
selected timestamped JPG filename to body boxes `[[x1, y1, x2, y2]]`, or `[]`
when the meter is absent. Enclose the multimeter housing only; loose probes
and leads are outside the target. Inspect and correct every proposal, including
any lead mistakenly captured by SAM, and add missed objects before marking a
frame reviewed. Report reliability only from reviewed labels in the separate
test session.

The current **provisional** live model is a YOLO11n-seg convolutional network
trained on proposals and exported to ONNX. `onnx_segmenter.py` runs masks on the
NucBox CPU using only NumPy, ONNX Runtime, and OpenCV. Its existing ONNX file
still embeds four classes: `raspberry_pi`, `arduino`, `pinecil`, and `wire`.
An earlier experiment narrowed proposals to the first **three** classes; the
current append-only six-class target is described above.
The live page filters out wire predictions and no longer offers wire capture;
the ONNX file itself has not yet been retrained or validated for three classes.
The earlier Mask R-CNN code remains as an experimental baseline; it has not
produced a checkpoint.

## Current data and evidence

- Eight earlier camera scenes have 21 box labels. `masks_from_boxes.py` uses
  Meta's SAM with those boxes to propose instance masks. The proposal contact
  sheet has been visually inspected, but the masks are not pixel-level reviewed
  ground truth.
- `import_wireseg.py` imported 200 licensed synthetic tabletop images from
  [WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K), with 391
  separate wire masks. These differ visibly from real workbench wires.
- One held-out earlier camera scene with proposed masks cannot establish
  real-world accuracy. A later Limelight capture session must be labeled and
  reserved for final evaluation. Current scores are not a basis for arm control.
- A new 45-second `camera1` session yielded 210 captures; `select_frames.py`
  reduced it to 30 representative frames with rotations and partial occlusion.
  These new frames are **not reviewed or fully labeled yet**. A three-class
  proposal pass on the 30 raw frames made 23 Pi, 23 Arduino, and 23 Pinecil
  proposals at 0.35 confidence. Missed objects require manual additions.
  An earlier four-class pass made only 1 wire prediction at 0.4 confidence.
  This is a failure check, not recall or accuracy, because no reviewed truth
  exists for that session.
- Two empty-mat `camera1` frames produced no detections at 0.4 confidence.
  Six new clear/hand-covered frames show all three detections when unobstructed,
  with some boards lost during hand coverage. Lowering confidence to 0.2 did
  not recover those misses. A can example shown in a screenshot was not present
  in the raw captures, and the user confirmed a can is not part of the target
  setting. Realistic robot-arm occlusion remains a data need.

All images, masks, and weights are stored outside Git.

## Limelight live view

The Limelight is reachable from the NucBox at `172.29.0.1` over USB. Pipeline
1, `evera-color-viewfinder`, sends clean color frames on MJPEG port 5802.
Pipeline 0 remains the original AprilTag setup. A backup of both pipeline
settings is at `~/evera-vision/limelight-backup/` on the NucBox. The viewfinder
was balanced with white paper; its saved values are red balance 1300, blue
balance 1850, exposure 2000 (0.01 ms units), and sensor gain 12. The clean
source is now 1280×960 with its pipeline LEDs off. The camera's factory
intrinsics are installed. Keep its physical position and focus fixed.

The separate `camera1` ZMQ relay on the NucBox publishes this clean view at
640×480 on port 5557. It uses the team's existing
`/home/aup/hackathon/robot/limelight_zmq.py` with `--name camera1 --port 5557
--url http://172.29.0.1:5802/ --width 640 --height 480`; the existing `top`
and `wrist` relays were not changed. `vision_training/sample_frames.py` saves
distinct raw frames from this relay, one folder per recording session.
For example, on the NucBox:

```bash
.venv/bin/python vision_training/sample_frames.py \
  --output captures/camera1-session-001 --seconds 45
```

Run it only while the objects are being rearranged. Repeat with a new folder
for another session, and reserve one whole reviewed session for final testing.

The preview currently uses the provisional CPU ONNX model. Run on the NucBox
from `~/evera-vision`:

```bash
.venv/bin/python vision/live_masks.py --camera http://172.29.0.1:5802/ \
  --port 8765 --captures captures/limelight-candidates \
  --checkpoint outputs/yolo-seg/workshop-provisional/weights/best.onnx
```

The page colors each predicted mask, displays a 3-by-3 alignment grid, and
saves raw frames. Its provisional warning is deliberate. `GET /health` reports
camera/model status, focus score, and CPU mask processing time.
Board predictions with at least 0.8 box IoU are now de-duplicated across Pi
and Arduino classes, retaining the higher-confidence label. This removes a
double label seen on one clear frame; it does not resolve genuine board
misclassification.
Forward remote port 8765 through SSH and open `http://127.0.0.1:8765/`.

## Data preparation and training

```bash
.venv/bin/python vision/masks_from_boxes.py \
  --source captures/workshop-v1-source \
  --checkpoint models/sam_vit_b_01ec64.pth \
  --output datasets/segmentation/camera-proposals --device cuda
.venv/bin/python vision/import_wireseg.py \
  --output datasets/segmentation/wireseg-plane --limit 200
.venv/bin/python vision/combine_segmentation.py \
  --camera datasets/segmentation/camera-proposals \
  --wires datasets/segmentation/wireseg-plane \
  --output datasets/segmentation/workshop-mixed
.venv/bin/python vision/train.py \
  --dataset datasets/segmentation/workshop-mixed \
  --output outputs/segmentation/workshop-provisional \
  --epochs 5 --batch-size 2 --device cuda
```

`train.py` balances public wire images against the far smaller camera set. It
reports classwise mask AP50 and flags unreviewed validation masks. A provisional
checkpoint is a development artifact. Train a final model only after correcting
mask boundaries and making a separate, manually labeled Limelight test split
with all four classes, empty scenes, and confusing background objects.

For a short data session, `capture_sequence.py` saves only distinct, stable
frames while objects are moved under the fixed Limelight. It does not auto-label
them. The captured frames require individual object and wire masks.

The first full Mask R-CNN run was stopped before producing a checkpoint. The
[research comparison](../docs/research/evera-segmentation-options.md) led to a
small YOLO11n-seg baseline. Its conversion excluded 77 WireSeg images whose
disconnected wires could not be accurately represented by one polygon,
retaining 237 wire instances. Seven camera training scenes were each repeated
20 times to balance the synthetic wire images; repetition adds no independent
camera evidence. The one validation image is an unreviewed SAM proposal with no
wire, so its near-perfect reported mAP is **not a valid quality result**.

```bash
.venv/bin/python vision/export_yolo_seg_dataset.py \
  --dataset datasets/segmentation/workshop-mixed \
  --output datasets/yolo-seg/workshop-provisional-clean \
  --allow-unreviewed --camera-repeat 20
.venv/bin/python vision/train_yolo_seg.py \
  --data datasets/yolo-seg/workshop-provisional-clean/data.yaml \
  --output outputs/yolo-seg/workshop-provisional --epochs 15 --batch 8 --device 0
.venv/bin/python -c "from ultralytics import YOLO; YOLO('outputs/yolo-seg/workshop-provisional/weights/best.pt').export(format='onnx', imgsz=640, opset=17, nms=False, simplify=False)"
.venv/bin/python vision/test_segmenter.py \
  --model outputs/yolo-seg/workshop-provisional/weights/best.onnx \
  --frames captures/camera1-session-20261010-selected
```

`test_segmenter.py` checks output types, boxes, and mask dimensions on five
saved frames; it does not test recognition accuracy. The custom ONNX mask
decoder matched reference PyTorch masks at IoU 0.978–0.998 on five frames.
The live CPU mask pass had a 26.6 ms median and 30.3 ms sampled p95 in 39
status snapshots while one preview server ran. This excludes camera arrival
and browser delivery latency. Held-out accuracy is still needed. Ultralytics code
and trained models use AGPL-3.0 by default; check that license before
distributing a derivative.

## Historical three-class proposal dataset

`vision_training/propose_masks.py` saves raw `camera1` frames, individual
prediction masks, previews, and a session manifest. It excludes contact sheets
from training input. The 30-frame session is prepared at
`datasets/segmentation/camera1-session-20261010-proposals-v2` on the NucBox;
all 69 masks are explicitly **unreviewed proposals**, not labels suitable for
quality claims. Review each frame, correct masks, and add the missed instances.
Keep the whole recording session in one data split. A separate whole session
with reviewed masks is still required for an honest test result, along with
more camera-specific instances of each class.

## References

- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
- [Meta Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Limelight 3A](https://docs.limelightvision.io/docs/docs-limelight/getting-started/limelight-3a)
- [Limelight REST API](https://docs.limelightvision.io/docs/docs-limelight/apis/rest-http-api)
