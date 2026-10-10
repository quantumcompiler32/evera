# Workshop object detection architecture

## Goal

Detect and draw bounding boxes around `raspberry_pi`, `arduino`, and
`soldering_iron` in a camera image. The current detector is a fine-tuned
Ultralytics YOLO26n model. The AMD NucBox reached over SSH currently holds the
training images, model weights, camera service, and inference runtime.

The first deliverable is a live camera view with a class label, confidence, and
box for each detected object. A box is a 2D image location, not a safe robot
grasp point or a 3D pose. Robot motion is a later integration that needs camera
calibration, coordinate transforms, and its own safety checks.

## Data flow

```text
camera frames ──> detector ──> boxes, classes, scores ──> display / saved demo
                     ^
                     │ versioned model + class map
                     │
labeled images ──> train and validate on another machine ──> export ──> copy to inference PC
```

The detector should return image pixel coordinates in a small, runtime-independent
record: `{class_name, confidence, x_min, y_min, x_max, y_max}`. Keep the camera,
model runtime, and renderer separate so a model trained elsewhere can be swapped
without rewriting the camera loop. Store the class map with the exported model;
an output index by itself does not identify an object.

## Build sequence

1. **Collect:** Use the intended camera and workbench to capture images of each
   object across angles, distances, lighting, backgrounds, and occlusions. Include
   images with no target object and visually similar distractors. A soldering iron
   should be unpowered while collecting close-up training images.
2. **Label:** Draw a tight box around every visible target instance. Start with
   exact class names `raspberry_pi`, `arduino`, and `soldering_iron`; record which Raspberry
   Pi boards and iron styles are included. Review labels for missing instances and
   ambiguous objects before training.
3. **Split:** Put related frames from the same capture session in one split. Keep
   separate train, validation, and test sessions so near-duplicate video frames
   cannot make evaluation look better than deployment.
4. **Train:** Fine-tune a pretrained object detection model on the training
   machine. A small YOLO detector is a practical baseline. Use the validation
   split for tuning; keep the test split untouched until a model candidate is
   chosen. Training hardware and inference hardware may differ.
5. **Export and transfer:** Package the model weights, class map, image size,
   preprocessing settings, confidence threshold, framework versions, and a
   checksum in a versioned model folder. Export to ONNX if the chosen runtime
   supports it. Copy this folder to the inference PC and run a fixed-image
   comparison before opening a live camera.
6. **Run and evaluate:** Draw boxes over live frames. Check precision and recall
   per class on held-out images, then measure latency and failure cases on the
   actual inference device and camera. Save example successes and failures for
   the demo and the next data collection pass.

## Device boundaries

| Device | Work |
| --- | --- |
| Camera / SO-101 station | Capture images and, if needed later, record robot demonstrations. |
| AMD NucBox reached over SSH | Image storage, label review, model training, live inference, and the browser preview server. ROCm PyTorch has detected its Radeon 8060S GPU. |
| Mac browser | View the live feed through an SSH port forward and control capture and labeling. |
| Limelight 3A, when connected | Supply MJPEG frames to the AMD detector. Its own CPU-only neural pipelines are an alternative runtime, not the one currently deployed. |

The LeRobot environment already in this repo is for the SO-101 robot workflow.
Object detection needs its own image/box dataset and training run. If robot
manipulation is added, the detector can provide candidate object locations to
the robot system, but LeRobot policy training and object detector training are
distinct pipelines.

## Remote training handoff

The SSH session is the training side of the boundary. Once images are labeled,
transfer the **same** dataset split and class names to that machine. Train and
validate there, then copy only the versioned exported model package and results
back to the inference PC. Do not copy an untracked `best.pt` alone: the class map
and preprocessing settings must travel with it.

Before choosing install commands or a GPU device flag, check the remote OS,
Python version, GPU, free storage, and whether PyTorch can see an accelerator.
The camera PC needs an inference runtime, but it does not need the full training
dataset or training dependencies. A fixed set of sample images should be run
through both machines to confirm that transfer and export preserved predictions.

## Adding a new device class

Define one stable class name, collect and box examples of the new physical item,
include old classes in the training dataset, then train and evaluate a new model
version. Check regression on the existing Raspberry Pi and soldering iron test
images before replacing the deployed model.

## Acceptance check for the first demo

- Live video shows the correct class and a tight box for both target objects.
- A held-out capture session reports per-class precision and recall, with
  representative false positives and misses inspected by a person.
- The exported package produces comparable detections on the training machine
  and the AMD inference PC from the same sample images.
- The measured live frame rate and latency are recorded on the actual AMD PC.

## References

- [Ultralytics object detection task and training](https://docs.ultralytics.com/tasks/detect/)
- [Ultralytics detection dataset layout](https://docs.ultralytics.com/datasets/detect/)
- [ONNX Runtime MIGraphX execution provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
- [LeRobot dataset documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
