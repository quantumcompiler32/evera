#!/usr/bin/env bash
# (Re)start both camera bridges in the background, each serving 320x240 frames to LeRobot over ZMQ:
#   top   = Limelight 3A stream  -> port 5555
#   wrist = USB wrist cam        -> port 5556
#   bash ~/hackathon/robot/start_limelight.sh      then check: tail ~/cam_top.log ~/cam_wrist.log
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/hackathon
start() {  # name port source
  local pidf=~/cam_$1.pid
  # pid file holds the process-group leader (the restart loop); kill the whole group
  if [ -f "$pidf" ]; then kill -- -"$(cat "$pidf")" 2>/dev/null || kill "$(cat "$pidf")" 2>/dev/null; sleep 1; fi
  # restart loop: the bridge exits on its own if its camera stalls, and comes straight back
  setsid nohup bash -c "while true; do python robot/limelight_zmq.py --name '$1' --port '$2' --url '$3'; echo 'bridge exited, restarting'; sleep 1; done" > ~/cam_$1.log 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$1 bridge started (pid $!, port $2, source $3)"
}
[ -f ~/limelight_zmq.pid ] && kill "$(cat ~/limelight_zmq.pid)" 2>/dev/null && rm ~/limelight_zmq.pid  # old single bridge
start top 5555 http://172.29.0.1:5800
start wrist 5556 "${WRIST_CAM:-/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0}"
