# Workshop vision architecture

## Goal and boundary

Segment each visible Raspberry Pi, Arduino, Pinecil soldering iron, and loose
wire in the Limelight 3A view. The output is a class, confidence, and binary
pixel mask for each instance. The robot arm is not controlled by this code.
Masks alone do not provide safe grasp points or 3D coordinates; arm integration
also needs camera-to-robot geometry, depth or a known work plane, motion safety,
and evaluation of failure cases.

## Data flow

```text
Limelight 3A clean MJPEG frame (USB network, port 5802)
    -> AMD NucBox camera reader and separate 640x480 camera1 ZMQ relay
    -> YOLO11n-seg ONNX on the NucBox CPU (provisional)
    -> per-instance masks, classes, scores
    -> browser overlay or downstream robot perception adapter

Existing photos + licensed public wire masks + new Limelight scenes
    -> individual mask review -> grouped train/validation/test splits
    -> AMD GPU fine-tuning offline -> CPU ONNX export
    -> held-out Limelight evaluation
    -> versioned checkpoint with class map and preprocessing
```

The provisional live model is YOLO11n-seg exported to ONNX, with a small
NumPy/OpenCV/ONNX Runtime CPU adapter. The Torchvision Mask R-CNN code is an
untrained comparison baseline. Meta SAM only proposes training masks from
legacy boxes; it is not the deployed CNN. Public WireSeg masks are synthetic
and do not replace real Limelight wire annotations. The source dataset stores
one binary PNG per object, including separate masks for crossing wires.

The earlier nine-class bounding-box `Detector` contract is incompatible with
the user's later choice of four-class instance masks. No robot integration
should import this provisional segmenter as if it satisfied that contract;
the consuming interface must be agreed with the other workstream.

## Camera configuration

The Limelight's original AprilTag pipeline remains in slot 0. Slot 1 is an
Evera color viewfinder. Port 5802 supplies a clean feed without the Limelight
crosshair. The browser server adds its grid after reading the raw frame and
saves training captures without overlays. Limelight factory lens intrinsics
are present. The white-paper color adjustment is a photometric setup step; it
does not determine camera-to-arm geometry.

Pipeline 1 now produces clean 1280×960 video with LEDs off. A separate relay
downsamples it to `camera1` 640×480 ZMQ on port 5557 for training and robot
integration. The existing `top` and `wrist` relays remain separate.

Mount the camera rigidly above the work area, keep focus and lighting stable,
and include the whole object and wire in frame. Changing camera position,
lighting, or pipeline settings after labeling creates a new image domain and
calls for new evaluation images.

## Validation gates

1. Review every proposal mask and add missing objects, including each wire.
2. Keep related frames and augmentations in the same split. Reserve a separate
   Limelight capture session for final testing.
3. Report per-class precision, recall, mask AP, false positives on empty scenes,
   and wire instance errors at crossings. Do not infer real-world accuracy from
   the one legacy validation scene.
4. Measure live CPU frame latency on the AMD PC. Test the exact
   checkpoint and class map used in the browser.
5. Before arm control, calibrate the camera-to-robot transform and test grasp
   planning and safety separately.

## References

- [Torchvision Mask R-CNN model](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
- [Torchvision instance segmentation fine tuning](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Limelight camera calibration](https://docs.limelightvision.io/docs/docs-limelight/getting-started/performing-charuco-camera-calibration)
