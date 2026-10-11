# Evera: existing tools for four-class instance segmentation

> **Scope changed after this note:** the next model is three-class (`raspberry_pi`, `arduino`, `pinecil`).
> The `wire` class was dropped. The model advice below still applies. Current status is in
> [vision/README.md](../../vision/README.md).

Research snapshot: 2026-10-10. Scope: `raspberry_pi`, `arduino`, Pinecil soldering iron, and each individual loose `wire` in the overhead Limelight view. This is a research recommendation, not a trained or validated detector.

## Recommended route

1. **Fix the final camera view first.** Keep the Limelight mounted and focused. Use its unannotated viewfinder stream, LEDs off, and one fixed resolution/exposure/color balance for both data collection and inference. If the integration contract's 640×480 `camera1` hub feed is authoritative, collect from that feed. Captures from the current 5 MP direct stream are useful for inspection but are a different input domain. Limelight documents a low-latency [Viewfinder pipeline](https://docs.limelightvision.io/docs/docs-limelight/software-change-log), a [Focus pipeline with a score for hand-turning the lens](https://docs.limelightvision.io/docs/docs-limelight/software-change-log), and a [REST API for pipeline settings](https://docs.limelightvision.io/docs/docs-limelight/apis/rest-http-api).
2. **Label final-camera images with assistance, then review every mask.** Seed Pi, Arduino, and Pinecil masks with the existing box-prompted SAM proposals. Mark each visible wire separately. Use [SAM 2 image/video prompting](https://github.com/facebookresearch/sam2) or [CVAT's SAM 2 tracker](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/) to carry corrected masks across a recording, then inspect and fix each selected frame. SAM produces boundaries; it does not establish the object's class. Keep distinct recording sessions for train, validation, and test to avoid adjacent-frame leakage.
3. **Use public data only as a supplement.** [IoTKITs](https://data.mendeley.com/datasets/x5thzmkxhy/1) supplies CC BY 4.0 polygon annotations for Arduino and Raspberry Pi boards, but mixes web images and original photos; its viewpoint and provenance may differ from this overhead camera. [WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K) supplies CC BY 4.0 cable/wire instance RLE masks; synthetic and data-center scenes differ from small loose wires on Evera's mat. No verified public Pinecil **instance-mask** dataset was found. Product photos from [PINE64's Pinecil page](https://wiki.pine64.org/wiki/Pinecil) show the appearance, but their reuse license and segmentation annotations were not established here. Avoid treating these sources as held-out evidence of camera accuracy.
4. **Train a small pretrained instance segmenter and benchmark the whole path.** [YOLO11n-seg](https://docs.ultralytics.com/models/yolo11) is an existing trainable, exportable mask model; its outputs include per-instance masks, boxes, class IDs, and confidence. [Ultralytics documents ONNX export and mask metrics](https://docs.ultralytics.com/tasks/segment). It is the likely faster CPU candidate than the current Torchvision Mask R-CNN, but that is an **inference from model size**, not an Evera benchmark. Test CPU latency including JPEG decode, preprocessing, ONNX inference, NMS, and mask reconstruction at 640×480 before switching. If quality is short, try `yolo11s-seg` only after establishing the `n` baseline. [Ultralytics says its code and trained models use AGPL-3.0 by default](https://www.ultralytics.com/license); document that in any deployment. If the team's fixed interface still requires nine-class bounding-box `Detector` output, this four-class mask model needs an explicit integration contract revision rather than an unnoticed substitution.
5. **Gate accuracy on new, reviewed camera data.** Report mask AP per class, false positives on empty mat, board confusion, wire recall, and measured CPU frame time on a whole held-out session. The current camera masks are SAM proposals and the 200 imported WireSeg examples are supplemental; neither establishes the requested accuracy. Do not describe an unreviewed pseudo-label validation result as ground truth.

## Model choice

| Candidate | Useful now | Limitation for Evera |
| --- | --- | --- |
| [YOLO11n-seg](https://docs.ultralytics.com/models/yolo11) | Pretrained instance-mask architecture; training, validation and ONNX export are documented. Smallest YOLO11 mask variant, so the first CPU benchmark candidate. | Four target classes need fine-tuning. The official benchmark is on other hardware/data, so a 40 ms NucBox target is unproven. AGPL-3.0 applies. |
| [Torchvision Mask R-CNN ResNet50 FPN v2](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) | Existing training code and COCO weights. Good baseline to compare mask quality. | Published model is about 46.4 million parameters and 333.6 GFLOPs; likely too expensive for the desired CPU live loop. Actual NucBox time still must be measured. |
| [SAM 2](https://github.com/facebookresearch/sam2) | Excellent candidate for assisted annotation and propagation through video, with Apache-2.0 model/code. | Promptable masks are not an autonomous four-class classifier; use it to make labels rather than as the production classifier. |
| [YOLOE](https://docs.ultralytics.com/models/yoloe) | Open-vocabulary detection/segmentation can propose boxes or masks before manual review. | Text prompts may confuse board variants and thin wires; check every proposal. It does not replace a camera-specific held-out evaluation. |

## Immediate decision

The highest-value next step is to confirm which final input reaches the robot app: direct Limelight stream or the team's 640×480 camera-hub `camera1` feed. Then collect and review new masks in that exact view, train the small segmenter, and measure its accuracy and CPU latency. Continuing to add internet photos or epochs before fixing the input/evaluation split can make training metrics look better without improving the live system.

## Implementation update

The clean Limelight viewfinder now feeds a separate `camera1` 640×480 ZMQ relay;
the existing robot `top` and `wrist` relays were left alone. A provisional
YOLO11n-seg ONNX model runs on the NucBox CPU. Thirty newly recorded camera1
frames expose its largest gap: only one `wire` prediction at 0.4 confidence.
Those frames still need reviewed labels, and the nine-class box interface still
needs an explicit revision before robot integration.
