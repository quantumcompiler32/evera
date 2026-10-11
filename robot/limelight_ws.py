#!/usr/bin/env python3
"""Read or change Limelight 3A pipeline settings from the GMKtec (the web UI's WebSocket, no extra deps).

    python robot/limelight_ws.py get                         # print the current pipeline settings
    python robot/limelight_ws.py set pipeline_res=1 flicker=2  # change settings (saved on the Limelight)
    python robot/limelight_ws.py decorations off             # hide the crosshair + fps text on :5800
    python robot/limelight_ws.py net                         # networking info (includes stream rate/res)
    python robot/limelight_ws.py streamrate 30               # MJPEG stream fps

Message format copied from the Limelight web UI (main.js): "<cmd> <payload_len>*<args joined by :>#<payload>".
Don't change settings while recording: the scene camera must look the same in every episode.
"""
import base64
import json
import os
import socket
import struct
import sys
import time

HOST, PORT = os.environ.get("LIMELIGHT", "172.29.0.1"), 5805


class WS:
    def __init__(self):
        self.s = socket.create_connection((HOST, PORT), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET / HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                        f"Sec-WebSocket-Protocol: soap, xmpp\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.s.recv(4096)
        if b" 101 " not in resp.split(b"\r\n")[0]:
            sys.exit(f"websocket handshake failed: {resp[:200]!r}")
        self.buf = resp.split(b"\r\n\r\n", 1)[1]

    def send(self, cmd, args=(), payload=""):
        text = f"{cmd} {len(payload)}*{':'.join(args)}#{payload}".encode()
        n = len(text)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n)
                                 if n < 65536 else bytes([0x80 | 127]) + struct.pack(">Q", n))
        mask = os.urandom(4)
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(text)))

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        b0, b1 = self._read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._read(8))[0]
        data = self._read(n)
        return b0 & 0x0F, data

    def wait_for(self, prefix, timeout=5):
        end = time.time() + timeout
        self.s.settimeout(timeout)
        while time.time() < end:
            op, data = self.recv()
            if op == 1 and data.startswith(prefix.encode()):
                return data.decode(errors="replace")
        return None


def get_pipeline(ws):
    ws.send("request_current_pipeline")
    msg = ws.wait_for("request_current_pipeline_response")
    if not msg:
        sys.exit("no pipeline response from the Limelight")
    return json.loads(msg.split(" ", 1)[1])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "get"
    ws = WS()
    if cmd == "get":
        print(json.dumps(get_pipeline(ws), indent=1, sort_keys=True))
    elif cmd == "set":
        changes = {}
        for kv in sys.argv[2:]:
            k, v = kv.split("=", 1)
            changes[k] = float(v) if "." in v else int(v) if v.lstrip("-").isdigit() else v
        before = get_pipeline(ws)
        ws.send("set_setting", payload=json.dumps(changes))
        time.sleep(0.3)
        ws.send("flush_settings")  # the web UI does this 250 ms after a change: persists it
        time.sleep(0.5)
        after = get_pipeline(ws)
        for k in changes:
            print(f"{k}: {before.get(k)} -> {after.get(k)}")
    elif cmd == "net":
        ws.send("get_all_networking_info")
        print(ws.wait_for("get_all_networking_info_response"))
    elif cmd == "streamrate":  # MJPEG stream fps; the UI offers 22 ("Normal") and 15 ("Low")
        ws.send("set_stream_rate", args=[sys.argv[2]])
        print(ws.wait_for("set_stream_rate_response") or "no response")
    elif cmd == "decorations":
        on = sys.argv[2] == "on"
        # the UI sends suppress_image_decorations with the boolean string as its args
        ws.send("suppress_image_decorations", args=list(str(not on).lower()))
        print("crosshair/text overlay", "shown" if on else "hidden")
    else:
        sys.exit(__doc__)
