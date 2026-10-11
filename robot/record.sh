#!/usr/bin/env bash
# Record teleop demos on the GMKtec (Ubuntu, LeRobot 0.6.2). Run inside your SSH window:
#   bash ~/hackathon/robot/record.sh test        # 2 episodes  -> Neevdharod/test_run
#   bash ~/hackathon/robot/record.sh full        # 50 episodes -> Neevdharod/so101_pick_place
#   bash ~/hackathon/robot/record.sh more 10     # add 10 more episodes to the full dataset
#   bash ~/hackathon/robot/record.sh practice    # just teleoperate, nothing recorded (Ctrl-C to stop)
#
# One episode per command (recommended):
#   bash ~/hackathon/robot/record.sh one         # record ONE episode, added to Neevdharod/so101_pick_place
#   bash ~/hackathon/robot/record.sh bad         # mark the episode you just recorded as bad (dropped at upload)
#   bash ~/hackathon/robot/record.sh status      # how many episodes so far, which are marked bad
#   bash ~/hackathon/robot/record.sh upload      # drop bad episodes, upload the rest to Hugging Face
# Each episode: EP_TIME s to do the task, then RESET_TIME s to put the brick back. Never Ctrl-C mid-episode.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# Stable USB-serial paths: /dev/ttyACM0/1 swap whenever a cable is replugged.
# Follower (gripper arm) = ...5836, leader (handle arm) = ...8274 (identified Oct 10 from motor calibration).
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125836-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128274-if00}"
EP_TIME="${EP_TIME:-30}"
RESET_TIME="${RESET_TIME:-10}"
TASK="${TASK:-pick up the lego brick and place it in the bin}"
CAMS="{scene: {type: zmq, server_address: 127.0.0.1, port: 5555, camera_name: top, width: 320, height: 240, fps: 30, warmup_s: 3}, wrist: {type: zmq, server_address: 127.0.0.1, port: 5556, camera_name: wrist, width: 320, height: 240, fps: 30, warmup_s: 3}}"

# Camera bridges must be running (they serve the Limelight + wrist cam at 320x240)
for n in top wrist; do
  if ! { [ -f ~/cam_$n.pid ] && kill -0 "$(cat ~/cam_$n.pid)" 2>/dev/null; }; then
    echo "camera bridges not running, starting them..."
    bash ~/hackathon/robot/start_limelight.sh
    sleep 8
    break
  fi
done

MAIN=Neevdharod/so101_pick_place
MAIN_ROOT=~/hackathon/data/so101_pick_place
BAD_FILE=$MAIN_ROOT.bad
count_eps() { python -c "import json,sys; print(json.load(open(sys.argv[1]+'/meta/info.json'))['total_episodes'])" "$MAIN_ROOT" 2>/dev/null || echo 0; }

case "${1:-}" in
  status)
    echo "episodes recorded: $(count_eps)   marked bad: $(tr '\n' ' ' 2>/dev/null < "$BAD_FILE" || echo none)"
    exit 0 ;;
  bad)
    N=$(count_eps)
    [ "$N" -gt 0 ] || { echo "nothing recorded yet"; exit 1; }
    LAST=$((N - 1))
    grep -qx "$LAST" "$BAD_FILE" 2>/dev/null || echo "$LAST" >> "$BAD_FILE"
    echo "episode $LAST marked bad (will be dropped at upload). Bad so far: $(tr '\n' ' ' < "$BAD_FILE")"
    exit 0 ;;
  upload)
    python - "$MAIN" "$MAIN_ROOT" "$BAD_FILE" <<'PY'
import shutil, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
repo, root, bad_file = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
ds = LeRobotDataset(repo, root=root)
bad = sorted({int(x) for x in bad_file.read_text().split()}) if bad_file.exists() else []
if bad:
    clean_root = root.with_name(root.name + "_clean")
    shutil.rmtree(clean_root, ignore_errors=True)
    print(f"dropping bad episodes {bad} ...")
    ds = delete_episodes(ds, bad, output_dir=clean_root, repo_id=repo)
ds.push_to_hub(private=True)
print(f"UPLOADED: {repo} ({ds.num_episodes} good episodes, {ds.num_frames} frames)")
PY
    exit 0 ;;
  practice)
    echo "Practice mode: move the leader arm, the follower copies it. Nothing is recorded. Ctrl-C to stop."
    exec lerobot-teleoperate \
      --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
      --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=my_leader ;;
  check) REPO=$MAIN; N=0; RESUME=false; CHECK_ONLY=1 ;;
  session)  # driven by the dashboard: N episodes in one connection, keys n/r/q control it
        REPO=$MAIN; N="${2:-50}"; RESET_TIME="${RESET_TIME:-300}"
        if [ "$(count_eps)" -gt 0 ]; then RESUME=true; else RESUME=false; rm -rf "$MAIN_ROOT" "$BAD_FILE"; fi
        SKIP_PUSH=1 ;;
  one)  REPO=$MAIN; N=1; RESET_TIME=0
        if [ "$(count_eps)" -gt 0 ]; then RESUME=true; else RESUME=false; rm -rf "$MAIN_ROOT" "$BAD_FILE"; fi
        SKIP_PUSH=1
        echo ">>> Episode $(count_eps). Leader arm at rest; start moving when you see 'Recording episode'." ;;
  test) REPO=Neevdharod/test_run;         N=2;            RESUME=false; rm -rf ~/hackathon/data/test_run ;;
  full) REPO=$MAIN; N=50; RESUME=false
        [ "$(count_eps)" -gt 0 ] && { echo "$MAIN already has $(count_eps) episodes. Use 'one' or 'more N' to add, or delete $MAIN_ROOT to start over."; exit 1; }
        rm -rf "$MAIN_ROOT" ;;
  more) REPO=Neevdharod/so101_pick_place; N="${2:?how many more episodes}"; RESUME=true ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
ROOT=~/hackathon/data/${REPO#*/}
SKIP_PUSH=${SKIP_PUSH:-0}
EPS_BEFORE=$(count_eps)

# Preflight: ping all 12 servos first so a dead/locked motor fails fast with a clear fix.
python - "$FOLLOWER_PORT" "$LEADER_PORT" <<'PY' || exit 1
import os, sys
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
motors = {n: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(names, 1)}
ok_all = True
for label, port in (("FOLLOWER (gripper arm)", sys.argv[1]), ("LEADER (handle arm)", sys.argv[2])):
    if not os.path.exists(port):
        print(f"PREFLIGHT FAIL: {label} USB not connected. Plug its USB cable in."); ok_all = False; continue
    bus = FeetechMotorsBus(port=port, motors=motors)
    try:
        bus.connect(handshake=False)
        missing = [i for i in range(1, 7) if not bus.ping(i, num_retry=1)]
    finally:
        bus.port_handler.closePort()
    if len(missing) == 6:
        print(f"PREFLIGHT FAIL: {label} has NO motor power. Check its 12V brick and barrel plug."); ok_all = False
    elif missing:
        hint = " (gripper overload lock: unplug this arm's POWER brick 5 s, plug back in)" if missing == [6] else " (loose servo cable)"
        print(f"PREFLIGHT FAIL: {label} motor(s) {missing} not answering{hint}"); ok_all = False
if ok_all:
    print("preflight: all 12 motors OK")
sys.exit(0 if ok_all else 1)
PY
[ "${CHECK_ONLY:-0}" = 1 ] && exit 0

echo "Recording $N episode(s) -> $REPO  (${EP_TIME}s task, ${RESET_TIME}s reset)"
echo "Watch for 'Recording episode N' and 'Reset the environment' below."
set +e   # a gripper "Overload" error at disconnect shouldn't stop us: the episode is already saved by then
lerobot-record \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
  --robot.cameras="$CAMS" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=my_leader \
  --display_data=false --play_sounds=false \
  --dataset.repo_id="$REPO" --dataset.root="$ROOT" \
  --dataset.num_episodes="$N" \
  --dataset.episode_time_s="$EP_TIME" --dataset.reset_time_s="$RESET_TIME" \
  --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --resume="$RESUME"
REC_STATUS=$?
set -e
# Always relax the follower afterwards. If the gripper was still squeezing, its overload protection
# can reject the torque-off command at disconnect and leave it straining; retry here, ignoring errors.
python - "$FOLLOWER_PORT" <<'PY' 2>/dev/null || true
import sys, time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
bus = FeetechMotorsBus(port=sys.argv[1], motors={n: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(names, 1)})
bus.connect(handshake=False)
for _ in range(5):
    for n in names:
        try:
            bus.write("Torque_Enable", n, 0, num_retry=2)
        except Exception:
            pass
    try:
        if not any(bus.sync_read("Torque_Enable", normalize=False).values()):
            print("follower relaxed (torque off)")
            break
    except Exception:
        pass
    time.sleep(0.5)
bus.port_handler.closePort()
PY
# "one" mode: don't upload after every episode; run `record.sh upload` at the end.
if [ "$(count_eps)" -le "$EPS_BEFORE" ] && [ "$REPO" = "$MAIN" ]; then
  echo "NOT SAVED: no new episode was recorded (see the error above). Nothing to mark bad."
  exit 1
fi
if [ "$SKIP_PUSH" = 1 ]; then
  echo "SAVED episode $(( $(count_eps) - 1 )) (total $(count_eps)). Bad one? -> bash ~/hackathon/robot/record.sh bad"
  exit 0
fi
# lerobot-record's own push didn't land for the first test run, so always push here (safe to repeat).
python - "$REPO" "$ROOT" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo, root = sys.argv[1], sys.argv[2]
ds = LeRobotDataset(repo, root=root)
ds.push_to_hub(private=True)
print(f"UPLOADED: {repo} ({ds.num_episodes} episodes, {ds.num_frames} frames)")
PY
echo "DONE: $REPO"
