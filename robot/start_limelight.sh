#!/usr/bin/env bash
# (Re)start both camera bridges in the background, each serving 320x240 frames to LeRobot over ZMQ:
#   top   = Limelight 3A stream  -> port 5555
#   wrist = USB wrist cam        -> port 5556
#   bash ~/hackathon/robot/start_limelight.sh      then check: tail ~/cam_top.log ~/cam_wrist.log
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/hackathon
start() {  # name port source [extra bridge args]
  local pidf=~/cam_$1.pid
  # pid file holds the process-group leader (the restart loop); kill the whole group
  if [ -f "$pidf" ]; then kill -- -"$(cat "$pidf")" 2>/dev/null || kill "$(cat "$pidf")" 2>/dev/null; sleep 1; fi
  # restart loop: the bridge exits on its own if its camera stalls, and comes straight back
  setsid nohup bash -c "while true; do python robot/limelight_zmq.py --name '$1' --port '$2' --url '$3' ${4:-}; echo 'bridge exited, restarting'; sleep 1; done" > ~/cam_$1.log 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$1 bridge started (pid $!, port $2, source $3)"
}
[ -f ~/limelight_zmq.pid ] && kill "$(cat ~/limelight_zmq.pid)" 2>/dev/null && rm ~/limelight_zmq.pid  # old single bridge

# Limelight camera profile (tuned Oct 10). Re-applied on every start: a vision-server restart once dropped
# the camera back to an AprilTag pipeline (grayscale, LED on). pipeline_res=1 is 640x480 @ 90 fps (was
# 1280x960 @ 22), flicker=2 is 60 Hz mains. "decorations off" hides the crosshair + fps text on :5800 and
# restarts the Limelight's MJPEG server, so the top bridge's first connect retries for a few seconds.
timeout 20 python robot/limelight_ws.py set pipeline_type=pipe_viewfinder pipeline_res=1 flicker=2 exposure=2000 \
  lcgain=12 red_balance=1300 blue_balance=1850 black_level=0 pipeline_led_enabled=0 || echo "WARN: Limelight settings not applied"
timeout 10 python robot/limelight_ws.py decorations off || echo "WARN: Limelight crosshair not hidden"
start top 5555 http://172.29.0.1:5800

# Wrist camera profile (tuned Oct 10): 60 Hz anti-flicker (was 50 Hz), fixed 11 ms exposure (was auto ~31 ms:
# blur, 4.5% blown out), fixed 3000 K white balance (was auto: drifting orange cast). Override with WRIST_CTRLS.
WRIST_CTRLS="${WRIST_CTRLS:-power_line_frequency=2 auto_exposure=1 exposure_time_absolute=110 white_balance_automatic=0 white_balance_temperature=3000}"
start wrist 5556 "${WRIST_CAM:-/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0}" \
  "$(for c in $WRIST_CTRLS; do printf -- '--ctrl %s ' "$c"; done)"
