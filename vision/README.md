# Workshop instance segmentation

The current experimental pipeline is a convolutional Mask R-CNN with a ResNet-50/FPN backbone. It
predicts a separate pixel mask and class score for each `raspberry_pi`,
`arduino`, `pinecil`, and `wire` instance. Each loose wire gets its own mask,
including at crossings. Training and inference run on the AMD NucBox's ROCm
PyTorch GPU. The Limelight 3A supplies video over USB networking; the CNN runs
on the NucBox, not inside the Limelight.

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

Run on the NucBox from `~/evera-vision`:

```bash
.venv/bin/python vision/live_masks.py --camera http://172.29.0.1:5802/ \
  --port 8765 --captures captures/limelight-candidates
```

After a validated checkpoint exists, add
`--checkpoint outputs/segmentation/workshop/best.pt --device cuda`. The live
page colors each predicted mask, displays a 3-by-3 alignment grid, and saves
raw frames. The `GET /health` endpoint reports camera and model status.
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
[research comparison](../docs/research/evera-segmentation-options.md) recommends
trying YOLO11n-seg next for a smaller CPU model, once enough real camera masks
have been reviewed. That is a recommendation, not a trained model or a measured
accuracy/speed result. Ultralytics code and trained models use AGPL-3.0 by
default; check that license before distributing a derivative.

## References

- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
- [Meta Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Limelight 3A](https://docs.limelightvision.io/docs/docs-limelight/getting-started/limelight-3a)
- [Limelight REST API](https://docs.limelightvision.io/docs/docs-limelight/apis/rest-http-api)
