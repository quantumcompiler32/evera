# Evera local object trainer

Run on the NucBox, then forward port 8766 through SSH:

```bash
cd ~/evera-vision
.venv/bin/python vision_dashboard/server.py --port 8766
```

The server binds to `127.0.0.1` on the NucBox. On the Mac, use
`ssh -L 8766:127.0.0.1:8766 aup@169.233.72.19` and open
`http://127.0.0.1:8766/`.

## Workflow

1. Add or choose an object class. New classes append to the list, preserving
   the existing class IDs.
2. Capture from the `camera1` ZMQ feed and select representative sharp frames.
3. Assign whole sessions to train, validation, or test. Do not split nearby
   frames from one recording across roles.
4. Draw all visible target object outlines on each frame, or import mask
   proposals and correct them. Save empty frames as reviewed when no target
   object is present.
5. Train a new model. The dashboard combines reviewed session labels with the
   older four-class dataset and fine-tunes from the existing checkpoint.

The output is saved under `outputs/dashboard/run-*/model/weights/`. The
dashboard never replaces the live preview checkpoint. The older validation
labels and imported proposals are provisional; a trained run must be checked
against a separately recorded, manually reviewed test session before making
reliability claims or using detections for robot actions.

The dashboard is intended for one local operator and has no login. Keep it
bound to loopback and access it through SSH forwarding.
