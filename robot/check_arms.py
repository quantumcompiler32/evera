"""Live arm checker: pings every servo on both arms once a second. Read-only, nothing moves.

    python ~/hackathon/robot/check_arms.py        (Ctrl-C to stop)

Use it while fixing power/USB: FOLLOWER must show 6/6 before recording.
"""
import os
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

ARMS = {
    "FOLLOWER (gripper arm)": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C125836-if00",
    "LEADER   (handle arm) ": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128274-if00",
}
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(NAMES, 1)}


def check(port):
    if not os.path.exists(port):
        return "USB NOT CONNECTED (board unplugged or hub dropped out)"
    bus = FeetechMotorsBus(port=port, motors=MOTORS)
    try:
        bus.connect(handshake=False)
        ok = [i for i in range(1, 7) if bus.ping(i, num_retry=1)]
        if not ok:
            return "0/6 motors -> USB ok, but NO SERVO POWER (check 12V brick + barrel plug + board-to-servo cable)"
        volts = bus.read("Present_Voltage", NAMES[ok[0] - 1], normalize=False) / 10
        missing = [i for i in range(1, 7) if i not in ok]
        status = "OK" if not missing else f"MISSING motor ids {missing} (loose servo-to-servo cable)"
        return f"{len(ok)}/6 motors, {volts:.1f} V  {status}"
    except Exception as e:
        return f"error: {e!r}"[:120]
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass


while True:
    stamp = time.strftime("%H:%M:%S")
    for name, port in ARMS.items():
        print(f"{stamp}  {name}: {check(port)}", flush=True)
    print(flush=True)
    time.sleep(1)
