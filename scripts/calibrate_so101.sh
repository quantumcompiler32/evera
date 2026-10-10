#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
calibrate="$repo_dir/.venv/bin/lerobot-calibrate"

if [[ ! -x "$calibrate" ]]; then
  echo "LeRobot is not installed. Run: $repo_dir/.venv/bin/python -m pip install 'lerobot[core_scripts,training,feetech]'" >&2
  exit 1
fi

arm_kind="${1:-follower}"
arm_id="${2:-so101_${arm_kind}_arm}"

case "$arm_kind" in
  follower) robot_flag="--robot.type=so101_follower"; port_flag="--robot.port"; id_flag="--robot.id" ;;
  leader) robot_flag="--teleop.type=so101_leader"; port_flag="--teleop.port"; id_flag="--teleop.id" ;;
  *) echo "Usage: $0 [follower|leader] [unique-arm-id]" >&2; exit 2 ;;
esac

ports=()
while IFS= read -r port; do
  ports+=("$port")
done < <(find /dev -maxdepth 1 -type c \( -name 'tty.usb*' -o -name 'tty.SLAB*' -o -name 'tty.usbserial*' \) -print 2>/dev/null | sort)

if [[ "${#ports[@]}" -ne 1 ]]; then
  echo "Expected exactly one SO-101 USB serial port; found ${#ports[@]}." >&2
  if [[ "${#ports[@]}" -eq 0 ]]; then
    echo "Connect and power the $arm_kind arm, then run this command again." >&2
  else
    printf 'Detected ports:\n  %s\n' "${ports[@]}" >&2
    echo "Disconnect other USB serial devices, or run the official port finder: .venv/bin/lerobot-find-port" >&2
  fi
  exit 1
fi

port="${ports[0]}"
echo "About to calibrate $arm_kind arm '$arm_id' on $port."
echo "The arm will move during calibration. Ensure it has clear space and is powered."
read -r -p "Press Enter to continue, or Ctrl-C to cancel: " _

exec "$calibrate" "$robot_flag" "$port_flag=$port" "$id_flag=$arm_id"
