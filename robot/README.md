# SO-101 demo recording (GMKtec side)

Tools for recording LeRobot teleoperation datasets on the AMD NucBox (GMKtec EVO-X2, Ubuntu,
LeRobot 0.6.2 in the `lerobot` conda env). The scripts expect this folder at `~/hackathon/robot/`.

| File | What it does |
|---|---|
| `dashboard.py` | Web dashboard on port 8080: live top + wrist previews, START SESSION, DONE/NEXT (Space), REDO (R), STOP (Q), PRACTICE, RUN POLICY, motor check, mark-bad, upload. Drives one `lerobot-record` session through a pseudo-terminal using LeRobot's `n`/`r`/`q` keys. Also serves the iPad app (`../evera/`) at `/evera/` and the JSON API it uses (`/api/status`, `/api/policy`, `/api/key`). |
| `record.sh` | Same actions from the CLI: `session N`, `one`, `more N`, `full`, `bad`, `status`, `review`, `upload`, `check`, `practice`, `policy <ckpt> [secs]`, `test`. Runs a 12-servo preflight first and relaxes the follower afterwards. The dataset comes from `DATASET` (default `so101_screwdriver_bin`). |
| `limelight_zmq.py` | Camera bridge: reads the Limelight MJPEG stream (or a USB camera by `/dev/v4l/by-id` path), keeps only the newest frame, downsizes to 320x240 and publishes it as a LeRobot `zmq` camera. Exits if frames stall so the wrapper restarts it. |
| `start_limelight.sh` | Starts both bridges under restart loops: `top` (Limelight, port 5555) and `wrist` (USB cam, port 5556). |
| `check_arms.py` | Live per-servo check of both arms (read-only), for debugging power and cables. |
| `test_camera.py` | Reads frames through LeRobot's camera classes and saves a snapshot. |
| `review_episodes.py` | Quality report per episode (grasp, release, drop spot, rest pose, stalls) plus contact sheets. `record.sh review [N]`. |
| `v4l2_ctl.py` | List/set USB camera V4L2 controls (no `v4l2-ctl` on the GMKtec); used to lock wrist exposure/WB. |
| `calibration/` | Backup of the `my_follower` / `my_leader` calibration files from the NucBox |
| `limelight_ws.py` | Read/change Limelight 3A pipeline settings over its web UI WebSocket (crosshair off, stream rate). |

## Typical use
```bash
bash ~/hackathon/robot/start_limelight.sh                      # camera bridges
python ~/hackathon/robot/dashboard.py                          # then open http://<GMKTEC_IP>:8080
DATASET=so101_pick_place bash ~/hackathon/robot/record.sh status # CLI, against an existing dataset
```

## Things that bit us
- `/dev/ttyACM0/1` swap whenever a cable is replugged, so the scripts address arms by `/dev/serial/by-id/` (follower `...5C4C125836`, leader `...5C4C128274`).
- If LeRobot asks "Press ENTER to use provided calibration file", pressing Enter writes that file's calibration to whatever arm is on the port. Check which arm it is first.
- The gripper servo overload-locks if it is still squeezing when an episode ends. Release before stopping; power-cycle the follower to clear it.
- LeRobot can't open `http://` camera URLs (`index_or_path` becomes a `Path`), which is why the Limelight goes through the ZMQ bridge.
- Episodes recorded before the evening of Oct 10 have the Limelight crosshair burned into the `scene` camera. `start_limelight.sh` now hides it on port 5800, so newer episodes are clean. Don't mix the two looks within one dataset you train on.

Setup: [docs/SETUP.md](../docs/SETUP.md). Training: [training/README.md](../training/README.md). Current datasets and models: [docs/HANDOFF.md](../docs/HANDOFF.md).
