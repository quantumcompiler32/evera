#!/usr/bin/env python3
"""Recording dashboard for the SO-101 on the GMKtec: live cameras + buttons to drive lerobot-record.

    python ~/hackathon/robot/dashboard.py      then open http://<gmktec-ip>:8080

One lerobot-record session runs under a pseudo-terminal; the buttons send it the keys LeRobot 0.6.2
listens for on a TTY: n = done/next, r = redo episode, q = stop. Camera views come from the ZMQ
bridges (start_limelight.sh), so the dashboard never competes with recording for the cameras.
"""
import base64
import json
import os
import pty
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import zmq

HOME = Path.home()
RECORD = HOME / "hackathon/robot/record.sh"
DATA_ROOT = HOME / "hackathon/data/so101_pick_place"
BAD_FILE = HOME / "hackathon/data/so101_pick_place.bad"
TARGET = 50
CAMS = {"top": 5555, "wrist": 5556}
PREVIEW_FPS = 8  # keep Wi-Fi load low; recording itself is unaffected
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# ---------------------------------------------------------------- cameras
latest_jpeg = {name: None for name in CAMS}


def camera_reader(name, port):
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://127.0.0.1:{port}")
    while True:
        try:
            msg = json.loads(sock.recv_string())
            b64 = next(iter(msg["images"].values()))
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            # bridges encode RGB order (what LeRobot expects); flip back so browsers show true colors
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 55])
            if ok:
                latest_jpeg[name] = jpg.tobytes()
        except Exception:
            time.sleep(0.2)


for _name, _port in CAMS.items():
    threading.Thread(target=camera_reader, args=(_name, _port), daemon=True).start()

# ---------------------------------------------------------------- recording session
lock = threading.Lock()
S = {
    "state": "idle",  # idle | starting | recording | reset | saving | error
    "episode": None,
    "t_state": time.time(),
    "ep_time": 30,
    "running": False,
    "log": deque(maxlen=200),
    "message": "",
    "busy": "",  # name of a one-shot job (upload/check) while it runs
}
proc = None
master_fd = None


def set_state(state, message=None):
    S["state"], S["t_state"] = state, time.time()
    if message is not None:
        S["message"] = message


def parse_line(line):
    S["log"].append(line)
    m = re.search(r"Recording episode (\d+)", line)
    if m:
        S["episode"] = int(m.group(1))
        set_state("recording", f"Recording episode {m.group(1)}: do the task, then press DONE")
    elif "Reset the environment" in line:
        set_state("reset", "Reset: put the object at the next spot, arm to rest, then press NEXT")
    elif "Re-record episode" in line:
        S["message"] = "Episode discarded, it will be recorded again"
    elif "Stop recording" in line:
        set_state("saving", "Stopping and saving...")
    elif "PREFLIGHT FAIL" in line:
        set_state("error", line.strip())
    elif "preflight: all 12 motors OK" in line:
        S["message"] = "All 12 motors OK, connecting arms + cameras..."
    elif "SAVED episode" in line or "NOT SAVED" in line:
        S["message"] = line.strip()
    elif "Traceback" in line or "RuntimeError" in line or "Error:" in line:
        S["message"] = "Error: see log below"


def reader():
    buf = ""
    while True:
        try:
            chunk = os.read(master_fd, 4096).decode(errors="replace")
        except OSError:
            break
        if not chunk:
            break
        buf += ANSI.sub("", chunk)
        *lines, buf = re.split(r"[\r\n]+", buf)
        with lock:
            for line in lines:
                if line.strip():
                    parse_line(line)
    proc.wait()
    with lock:
        S["running"] = False
        if S["state"] != "error":
            set_state("idle", f"Session ended (exit code {proc.returncode}). Good episodes: {counts()['good']}")


def start_session(n, ep_time, task):
    global proc, master_fd
    if S["running"]:
        return "a session is already running"
    master_fd, slave_fd = pty.openpty()
    env = dict(os.environ, EP_TIME=str(ep_time), TASK=task, PYTHONUNBUFFERED="1")
    for k in ("DISPLAY", "WAYLAND_DISPLAY"):  # force LeRobot's terminal (TTY) key listener
        env.pop(k, None)
    proc = subprocess.Popen(["bash", str(RECORD), "session", str(n)], stdin=slave_fd, stdout=slave_fd,
                            stderr=slave_fd, env=env, start_new_session=True, close_fds=True)
    os.close(slave_fd)
    S.update(running=True, ep_time=ep_time, episode=None)
    S["log"].clear()
    set_state("starting", "Checking motors, connecting arms + cameras (about 10 s)...")
    threading.Thread(target=reader, daemon=True).start()
    return "started"


def send_key(k):
    if not S["running"] or master_fd is None:
        return "no session running"
    os.write(master_fd, k.encode())
    # Update the banner immediately instead of waiting for LeRobot's log line.
    if k == "n" and S["state"] == "recording":
        set_state("reset", "Reset: put the object at the next spot, arm to rest, then press NEXT")
    elif k == "n" and S["state"] == "reset":
        set_state("starting", "Starting next episode...")
    elif k == "r":
        set_state("starting", "Discarding this episode, it will be recorded again...")
    elif k == "q":
        set_state("saving", "Stopping and saving...")
    return "sent"


def run_job(name, args):
    def job():
        with lock:
            S["busy"] = name
            S["message"] = f"{name}..."
        out = subprocess.run(["bash", str(RECORD), *args], capture_output=True, text=True)
        text = ANSI.sub("", out.stdout + out.stderr)
        with lock:
            for line in text.splitlines()[-25:]:
                if line.strip() and "Warn" not in line:
                    S["log"].append(line)
            last = [l for l in text.splitlines() if l.strip()]
            S["message"] = (last[-1] if last else f"{name} done")[:200]
            S["busy"] = ""

    if S["running"] or S["busy"]:
        return "busy"
    threading.Thread(target=job, daemon=True).start()
    return "started"


def counts():
    try:
        total = json.loads((DATA_ROOT / "meta/info.json").read_text())["total_episodes"]
    except Exception:
        total = 0
    bad = sorted({int(x) for x in BAD_FILE.read_text().split()}) if BAD_FILE.exists() else []
    return {"total": total, "bad": bad, "good": total - len([b for b in bad if b < total])}


def mark_bad_last():
    c = counts()
    if c["total"] == 0:
        return "nothing saved yet"
    last = c["total"] - 1
    if last not in c["bad"]:
        with open(BAD_FILE, "a") as f:
            f.write(f"{last}\n")
    return f"episode {last} marked bad"


# ---------------------------------------------------------------- http
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 Recorder</title><style>
:root{--bg:#0f1115;--panel:#181b22;--fg:#e8eaf0;--mut:#8b92a5;--red:#e5484d;--blue:#3e7bfa;--green:#30a46c;--yellow:#f5a524}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:16px}
#banner{border-radius:10px;padding:14px 18px;font-size:22px;font-weight:700;display:flex;justify-content:space-between;align-items:center;background:#2a2f3a}
#banner.recording{background:var(--red)}#banner.reset{background:var(--blue)}#banner.starting,#banner.saving{background:var(--yellow);color:#111}
#banner.error{background:#7a1d20}#timer{font-variant-numeric:tabular-nums;font-size:30px}
#msg{color:var(--mut);margin:8px 2px 12px;min-height:20px}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cam{background:var(--panel);border-radius:10px;padding:8px}
.cam img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;border-radius:6px;display:block}.cam div{color:var(--mut);font-size:13px;margin-top:4px}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
button{border:0;border-radius:10px;padding:16px 22px;font-size:18px;font-weight:700;cursor:pointer;color:#fff;background:#2f3542}
button:disabled{opacity:.35;cursor:default}#done{background:var(--green);flex:2}#redo{background:var(--yellow);color:#111}#stop{background:#5b2a2c}
#start{background:var(--blue);flex:2}.small{font-size:14px;padding:10px 14px;font-weight:600}
.stats{display:flex;gap:18px;align-items:center;background:var(--panel);border-radius:10px;padding:12px 16px;flex-wrap:wrap}
.big{font-size:28px;font-weight:800}.bar{flex:1;min-width:160px;height:10px;background:#2a2f3a;border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--green)}
label{color:var(--mut);font-size:13px}input{background:#11141a;color:var(--fg);border:1px solid #2f3542;border-radius:6px;padding:8px;font-size:14px}
pre{background:#0a0c10;border-radius:8px;padding:10px;height:180px;overflow:auto;font-size:12px;color:#aab}
kbd{background:#2f3542;border-radius:4px;padding:1px 6px;font-size:12px}
@media(max-width:700px){.cams{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div id="banner" class="idle"><span id="state">IDLE</span><span id="timer"></span></div>
<div id="msg"></div>
<div class="cams"><div class="cam"><img id="imgtop" src="/stream/top"><div>Top (Limelight): scene</div></div>
<div class="cam"><img id="imgwrist" src="/stream/wrist"><div>Wrist camera</div></div></div>
<div class="row"><button id="done" onclick="key('n')">✓ DONE / NEXT <kbd>Space</kbd></button>
<button id="redo" onclick="key('r')">↺ REDO <kbd>R</kbd></button><button id="stop" onclick="key('q')">■ STOP <kbd>Q</kbd></button></div>
<div class="stats"><div><div class="big" id="good">0</div><label>good / 50</label></div><div class="bar"><i id="prog" style="width:0"></i></div>
<div><div id="bad">bad: none</div><label>total saved: <span id="total">0</span></label></div></div>
<div class="row" style="align-items:end">
<div><label>episodes this session</label><br><input id="n" type="number" value="50" style="width:90px"></div>
<div><label>max seconds per episode</label><br><input id="ept" type="number" value="30" style="width:90px"></div>
<div style="flex:1"><label>task text</label><br><input id="task" value="pick up the lego brick and place it in the bin" style="width:100%"></div>
<button id="start" onclick="start()">▶ START SESSION</button></div>
<div class="row"><button class="small" id="markbad" onclick="post('/api/bad')">✗ Mark last saved episode bad</button>
<button class="small" id="check" onclick="post('/api/check')">Check motors</button>
<button class="small" id="upload" onclick="post('/api/upload')">⬆ Upload to Hugging Face</button>
<button class="small" id="camtoggle" onclick="toggleCams()">Cameras: ON</button></div>
<pre id="log"></pre>
<label>Flow: START → episode records (red) → do the task → DONE → reset (blue): put object back, arm to rest → NEXT → … Messed up? REDO discards the episode. Keys: <kbd>Space</kbd> done/next, <kbd>R</kbd> redo, <kbd>Q</kbd> stop.</label>
</div><script>
let st={},camsOn=true;
function toggleCams(){camsOn=!camsOn;for(const c of ['top','wrist']){const i=document.getElementById('img'+c);i.src=camsOn?'/stream/'+c+'?'+Date.now():'';}
 document.getElementById('camtoggle').textContent='Cameras: '+(camsOn?'ON':'OFF')}
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json()}
function key(k){post('/api/key',{k})}
function start(){post('/api/start',{n:+n.value,ep_time:+ept.value,task:task.value})}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 if(e.code==='Space'){e.preventDefault();key('n')}else if(e.key==='r'||e.key==='R')key('r');else if(e.key==='q'||e.key==='Q')key('q')});
async function tick(){try{st=await (await fetch('/api/status')).json()}catch(e){setTimeout(tick,1000);return}
 const b=document.getElementById('banner');b.className=st.state;
 const names={idle:'IDLE',starting:'CONNECTING…',recording:'● RECORDING ep '+st.episode,reset:'RESET: get ready',saving:'SAVING…',error:'PROBLEM'};
 document.getElementById('state').textContent=names[st.state]||st.state;
 const el=(Date.now()/1000-st.t_state);document.getElementById('timer').textContent=
  st.state==='recording'?Math.max(0,Math.ceil(st.ep_time-el))+'s':st.state==='reset'?Math.floor(el)+'s':'';
 document.getElementById('msg').textContent=st.busy?st.busy+'…':st.message;
 good.textContent=st.counts.good;total.textContent=st.counts.total;prog.style.width=Math.min(100,st.counts.good/50*100)+'%';
 bad.textContent='bad: '+(st.counts.bad.length?st.counts.bad.join(', '):'none');
 const live=st.running;for(const id of ['done','redo','stop'])document.getElementById(id).disabled=!live;
 document.getElementById('done').textContent=st.state==='reset'?'▶ NEXT EPISODE  (Space)':'✓ DONE  (Space)';
 for(const id of ['start','upload','check'])document.getElementById(id).disabled=live||!!st.busy;
 const lg=document.getElementById('log');const atBottom=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-5;
 lg.textContent=st.log.join('\n');if(atBottom)lg.scrollTop=lg.scrollHeight;setTimeout(tick,250)}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/stream/"):
            name = self.path.split("?")[0].split("/")[-1]
            if name not in CAMS:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    jpg = latest_jpeg[name]
                    if jpg and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(1 / PREVIEW_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/api/status":
            with lock:
                out = {k: (list(v)[-40:] if k == "log" else v) for k, v in S.items()}
            out["counts"] = counts()
            self._json(out)
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/start":
            with lock:
                r = start_session(max(1, int(body.get("n", TARGET))), max(5, int(body.get("ep_time", 30))),
                                  body.get("task") or "pick up the lego brick and place it in the bin")
        elif self.path == "/api/key":
            r = send_key(body.get("k", "")[:1]) if body.get("k") in ("n", "r", "q") else "bad key"
        elif self.path == "/api/bad":
            with lock:
                r = mark_bad_last()
                S["message"] = r
        elif self.path == "/api/upload":
            r = run_job("Uploading to Hugging Face", ["upload"])
        elif self.path == "/api/check":
            r = run_job("Checking motors", ["check"])
        else:
            return self.send_error(404)
        self._json({"result": r})


if __name__ == "__main__":
    print("dashboard on http://0.0.0.0:8080", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
