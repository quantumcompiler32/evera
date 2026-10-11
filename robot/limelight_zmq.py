#!/usr/bin/env python3
"""Serve the Limelight 3A's MJPEG stream to LeRobot as a ZMQ camera (no sudo, no v4l2loopback).

LeRobot's OpenCV camera can't open http:// URLs, but its `zmq` camera type reads frames from a
ZMQ publisher. This script is that publisher, speaking the same JSON format as
lerobot/cameras/zmq/image_server.py: {"timestamps": {name: t}, "images": {name: base64-jpeg}}.

    nohup python robot/limelight_zmq.py > ~/limelight_zmq.log 2>&1 &

Then in lerobot-record:
    --robot.cameras="{top: {type: zmq, server_address: 127.0.0.1, port: 5555, camera_name: top, width: 320, height: 240, fps: 30}}"
"""
import argparse
import base64
import json
import os
import threading
import time

import cv2
import zmq

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://172.29.0.1:5800")
p.add_argument("--port", type=int, default=5555)
p.add_argument("--name", default="top")
p.add_argument("--width", type=int, default=320)
p.add_argument("--height", type=int, default=240)
args = p.parse_args()

sock = zmq.Context().socket(zmq.PUB)
sock.setsockopt(zmq.SNDHWM, 2)
sock.bind(f"tcp://*:{args.port}")


def open_stream():
    while True:
        if args.url.startswith("/dev/"):  # local USB camera: capture at its native 640x480, downscale below
            # Resolve /dev/v4l/by-id symlinks on every (re)open: the /dev/videoN number changes on replug.
            cap = cv2.VideoCapture(os.path.realpath(args.url), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"connected to {args.url}", flush=True)
            return cap
        print(f"can't open {args.url}, retrying in 1s", flush=True)
        time.sleep(1)


# Watchdog: a replugged USB camera can leave cap.read() blocked forever. If no frame arrives for
# 5 s, exit; start_limelight.sh restarts the bridge, which reopens the camera cleanly.
last_frame = time.time()


def watchdog():
    while True:
        time.sleep(1)
        if time.time() - last_frame > 5:
            print("no frames for 5s, restarting bridge", flush=True)
            os._exit(1)


cap = open_stream()
last_frame = time.time()
threading.Thread(target=watchdog, daemon=True).start()

# Capture thread reads flat out and keeps only the newest frame, so publishing never adds a backlog.
newest = {"frame": None, "seq": 0}
cond = threading.Condition()


def capture():
    global cap, last_frame
    while True:
        ok, frame = cap.read()  # BGR
        if not ok:
            print("stream dropped, reconnecting", flush=True)
            cap.release()
            cap = open_stream()
            continue
        last_frame = time.time()
        with cond:
            newest["frame"], newest["seq"] = frame, newest["seq"] + 1
            cond.notify()


threading.Thread(target=capture, daemon=True).start()
sent, t_report, seen = 0, time.time(), 0
while True:
    with cond:
        cond.wait_for(lambda: newest["seq"] != seen, timeout=1.0)
        if newest["seq"] == seen:
            continue
        frame, seen = newest["frame"], newest["seq"]
    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
    # The LeRobot client returns the decoded JPEG as-is and treats it as RGB, so encode RGB order.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, jpg = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    msg = {"timestamps": {args.name: time.time()}, "images": {args.name: base64.b64encode(jpg).decode()}}
    try:
        sock.send_string(json.dumps(msg), zmq.NOBLOCK)
    except zmq.Again:
        pass
    sent += 1
    if time.time() - t_report >= 10:
        print(f"publishing {sent / (time.time() - t_report):.1f} fps", flush=True)
        sent, t_report = 0, time.time()
