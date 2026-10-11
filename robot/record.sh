#!/usr/bin/env bash
# Record teleop demos on the GMKtec (Ubuntu, LeRobot 0.6.2). Run inside your SSH window:
#   bash ~/hackathon/robot/record.sh test        # 2 episodes  -> Neevdharod/test_run
#   bash ~/hackathon/robot/record.sh full        # 50 episodes -> Neevdharod/$DATASET
#   bash ~/hackathon/robot/record.sh more 10     # add 10 more episodes to the full dataset
#   bash ~/hackathon/robot/record.sh practice    # just teleoperate, nothing recorded (Ctrl-C to stop)
#
# One episode per command (recommended):
#   bash ~/hackathon/robot/record.sh one         # record ONE episode, added to Neevdharod/$DATASET
#   bash ~/hackathon/robot/record.sh bad         # mark the episode you just recorded as bad (kept, but split off at upload)
#   bash ~/hackathon/robot/record.sh status      # how many episodes so far, which are marked bad
#   bash ~/hackathon/robot/record.sh upload      # good -> $DATASET, bad -> ${DATASET}_rejects on Hugging Face
#   bash ~/hackathon/robot/record.sh review [N]  # quality report + contact sheets for episodes N.. (safe while recording)
#   bash ~/hackathon/robot/record.sh replay E    # play episode E of $REPLAY_DATASET back on the follower (open loop)
# Each episode: EP_TIME s to do the task, then RESET_TIME s to put the object back. Never Ctrl-C mid-episode.
# DATASET defaults to so101_screwdriver_bin (fixed bin + taped start spots). The older so101_pick_place
# episodes dropped the screwdriver in a different place every time, so they stay out of it.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# Stable USB-serial paths: /dev/ttyACM0/1 swap whenever a cable is replugged.
# Follower (gripper arm) = ...5836, leader (handle arm) = ...8274 (identified Oct 10 from motor calibration).
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125836-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128274-if00}"
EP_TIME="${EP_TIME:-25}"  # a cap: press DONE as soon as the arm is back at rest
RESET_TIME="${RESET_TIME:-10}"
TASK="${TASK:-pick up the screwdriver and place it in the bin}"
DATASET="${DATASET:-so101_screwdriver_bin}"
REPLAY_DATASET="${REPLAY_DATASET:-so101_pick_place_90_100}"  # all 11 episodes pass review_episodes.py
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

MAIN=Neevdharod/$DATASET
MAIN_ROOT=~/hackathon/data/$DATASET
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
    echo "episode $LAST marked bad (will go to ${DATASET}_rejects at upload). Bad so far: $(tr '\n' ' ' < "$BAD_FILE")"
    exit 0 ;;
  upload)
    python - "$MAIN" "$MAIN_ROOT" "$BAD_FILE" <<'PY'
import shutil, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
repo, root, bad_file = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
full = LeRobotDataset(repo, root=root)
bad = sorted({int(x) for x in bad_file.read_text().split()} & set(range(full.num_episodes))) if bad_file.exists() else []
ds = full
if bad:
    # Bad episodes are kept, not thrown away: they go to a separate <repo>_rejects dataset, and the
    # raw local folder ($MAIN_ROOT) still has everything. Only the good split is trained on.
    good = [e for e in range(full.num_episodes) if e not in bad]
    clean_root, rej_root = (root.with_name(root.name + s) for s in ("_clean", "_rejects"))
    shutil.rmtree(clean_root, ignore_errors=True)
    shutil.rmtree(rej_root, ignore_errors=True)
    print(f"splitting off bad episodes {bad} ...")
    ds = delete_episodes(full, bad, output_dir=clean_root, repo_id=repo)
    if good:
        rej = delete_episodes(full, good, output_dir=rej_root, repo_id=repo + "_rejects")
        rej.push_to_hub(private=True)
        print(f"UPLOADED: {repo}_rejects ({rej.num_episodes} bad episodes: original numbers {bad})")
ds.push_to_hub(private=True)
print(f"UPLOADED: {repo} ({ds.num_episodes} good episodes, {ds.num_frames} frames)")
PY
    exit 0 ;;
  review)  # snapshot the dataset (recording can keep going) and run the quality report on it
    [ "$(count_eps)" -gt 0 ] || { echo "nothing recorded yet"; exit 1; }
    SNAP=${MAIN_ROOT}_snap; OUT=~/hackathon/review
    rm -rf "$SNAP" "$OUT"; cp -r "$MAIN_ROOT" "$SNAP"
    cd ~/hackathon && REPO=$MAIN nice -n 19 python robot/review_episodes.py "$SNAP" "${2:-0}" "$OUT" 2>&1 | grep -v Warning
    echo "flagged episodes worth dropping: record.sh bad marks only the LAST one; for others: echo <ep> >> $BAD_FILE"
    exit 0 ;;
  practice)
    echo "Practice mode: move the leader arm, the follower copies it. Nothing is recorded. Ctrl-C to stop."
    exec lerobot-teleoperate \
      --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
      --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=my_leader ;;
  check) REPO=$MAIN; N=0; RESUME=false; CHECK_ONLY=1 ;;
  policy)  # run a trained policy on the follower (no leader needed): record.sh policy <ckpt_dir> [seconds]
        REPO=$MAIN; N=0; RESUME=false; POLICY_CKPT="${2:?checkpoint dir (.../pretrained_model)}"; POLICY_SECS="${3:-30}" ;;
  replay)  # send a recorded episode's joint targets to the follower: record.sh replay <episode>
        REPO=Neevdharod/$REPLAY_DATASET; N=0; RESUME=false; REPLAY_EP="${2:?episode index}" ;;
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
  more) REPO=$MAIN; N="${2:?how many more episodes}"; RESUME=true ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
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

# Relax the follower after a run. If the gripper was still squeezing, its overload protection
# can reject the torque-off command at disconnect and leave it straining; retry here, ignoring errors.
relax_follower() {
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
}

if [ -n "${POLICY_CKPT:-}" ]; then
  echo "Running policy $POLICY_CKPT for ${POLICY_SECS}s (max 8 deg per joint per step). Keep a hand near the follower's power plug."
  set +e
  lerobot-rollout --strategy.type=base \
    --policy.path="$POLICY_CKPT" --device=cuda \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
    --robot.cameras="$CAMS" --robot.max_relative_target=8 \
    --task="$TASK" --duration="$POLICY_SECS" --play_sounds=false
  set -e
  exit 0
fi

if [ -n "${REPLAY_EP:-}" ]; then
  # Open loop: the object and bin must sit where they were in this episode. Each episode starts and
  # ends at the rest pose; max_relative_target eases the arm in if it starts somewhere else.
  # Shoulder lift and elbow carry the arm's weight and trail the leader by up to ~15 deg while
  # recording (which has no cap). An 8 deg cap shrinks the error their servos push against, so
  # they sag and drag the arm down; give them 30.
  echo "Replaying episode $REPLAY_EP of $REPO (max 8 deg per step, 30 for shoulder lift and elbow). Keep a hand near the follower's power plug."
  set +e
  # STOP sends Ctrl-C. Without a trap bash dies with lerobot-replay and skips relax_follower, which
  # matters most when STOP lands mid-grasp with the gripper squeezing.
  trap 'echo "STOP: ending the replay"' INT
  lerobot-replay \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
    --robot.max_relative_target="{shoulder_pan: 8, shoulder_lift: 30, elbow_flex: 30, wrist_flex: 8, wrist_roll: 8, gripper: 8}" \
    --dataset.repo_id="$REPO" --dataset.root="$ROOT" --dataset.episode="$REPLAY_EP" \
    --play_sounds=false
  trap - INT
  set -e
  echo "Replay finished"
  relax_follower
  exit 0
fi

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
relax_follower
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
