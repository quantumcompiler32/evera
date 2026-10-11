"""Read frames through LeRobot's own camera classes, report fps, and save a snapshot.

    python robot/test_camera.py zmq            # Limelight via the ZMQ bridge
    python robot/test_camera.py /dev/video3    # a regular USB camera
"""
import sys
import time

import cv2

src = sys.argv[1]
if src == "zmq":
    from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

    cam = ZMQCamera(ZMQCameraConfig(server_address="127.0.0.1", port=5555, camera_name="top", width=320, height=240, fps=30))
    out = "/tmp/cam_top.jpg"
else:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=src, width=320, height=240, fps=30))
    out = "/tmp/cam_" + src.strip("/").replace("/", "_") + ".jpg"

cam.connect()
for _ in range(30):  # let auto-exposure settle
    cam.read()
t = time.time()
for _ in range(60):
    f = cam.read()
fps = 60 / (time.time() - t)
cv2.imwrite(out, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
print(f"{src}: shape={f.shape} read_fps={fps:.1f} mean_rgb={f.reshape(-1, 3).mean(0).round(1)} saved={out}")
cam.disconnect()
