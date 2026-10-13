# Capture and mask proposal tools

These scripts prepare data for the separate [vision pipeline](../vision/README.md).
Run them from the repository root; use `python <script> --help` for arguments.

| Script | Purpose |
|---|---|
| `sample_frames.py` | Save distinct raw frames from the `camera1` ZMQ relay |
| `select_frames.py` | Choose sharp representatives from a saved session |
| `propose_masks.py` | Generate three-class proposals with the provisional ONNX model |
| `propose_tape_measure.py` | Generate housing-only SAM proposals for tape measures |
| `propose_multimeter.py` | Generate body-only SAM proposals from supplied boxes |

Capture needs NumPy, OpenCV and pyzmq. ONNX proposals additionally need ONNX Runtime;
SAM proposal tools need a separate training environment, PyTorch, Segment Anything,
and external weights. Do not add GPU training packages to the CPU inference environment.

Save captures, annotations, and weights outside Git. Review every proposal and add
missed instances. Reserve a separate whole recording session for testing; selected
frames and their source session must never be split across training and evaluation.
Class IDs are defined in `vision/class_map.py`; preserve their order.
