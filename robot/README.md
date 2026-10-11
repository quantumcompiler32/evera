# SO-101 demo recording (GMKtec side)

Tools for recording LeRobot teleoperation datasets on the AMD NucBox (GMKtec EVO-X2, Ubuntu,
LeRobot 0.6.2 in the `lerobot` conda env). The scripts expect this folder at `~/hackathon/robot/`.

| File | What it does |
|---|---|
| `dashboard.py` | Web dashboard on port 8080: live top + wrist previews, START SESSION, DONE/NEXT (Space), REDO (R), STOP (Q), motor check, mark-bad, upload. Drives one `lerobot-record` session through a pseudo-terminal using LeRobot's `n`/`r`/`q` keys. |
| `record.sh` | Same actions from the CLI: `session N`, `one`, `bad`, `status`, `upload`, `check`, `practice`, `test`. Runs a 12-servo preflight first and relaxes the follower afterwards. |
| `limelight_zmq.py` | Camera bridge: reads the Limelight MJPEG stream (or a USB camera by `/dev/v4l/by-id` path), keeps only the newest frame, downsizes to 320x240 and publishes it as a LeRobot `zmq` camera. Exits if frames stall so the wrapper restarts it. |
| `start_limelight.sh` | Starts both bridges under restart loops: `top` (Limelight, port 5555) and `wrist` (USB cam, port 5556). |
| `check_arms.py` | Live per-servo check of both arms (read-only), for debugging power and cables. |
| `test_camera.py` | Reads frames through LeRobot's camera classes and saves a snapshot. |

## Typical use
```bash
bash ~/hackathon/robot/start_limelight.sh                      # camera bridges
python ~/hackathon/robot/dashboard.py                          # then open http://<GMKTEC_IP>:8080
```

## Things that bit us
- `/dev/ttyACM0/1` swap whenever a cable is replugged, so the scripts address arms by `/dev/serial/by-id/` (follower `...5C4C125836`, leader `...5C4C128274`).
- If LeRobot asks "Press ENTER to use provided calibration file", pressing Enter writes that file's calibration to whatever arm is on the port. Check which arm it is first.
- The gripper servo overload-locks if it is still squeezing when an episode ends. Release before stopping; power-cycle the follower to clear it.
- LeRobot can't open `http://` camera URLs (`index_or_path` becomes a `Path`), which is why the Limelight goes through the ZMQ bridge.
- The dataset recorded so far uses the Limelight's port 5800 stream (with crosshair). Port 5802 is a clean feed (see `docs/ARCHITECTURE.md`). Use it for new datasets, but don't mix the two within one dataset.

Training on AMD Radeon Cloud: see `../training/README.md`. Current status: `../docs/HANDOFF.md`.
