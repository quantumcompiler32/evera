# evera: iPad control app for the SO-101

evera is a JARVIS-style bench assistant. Tap a tool (or say "evera, pass me the multimeter"), and the arm runs that action through the dashboard's `/api/policy`. A live 3D model of the real SO-101 (from its URDF) animates with the run state. evera answers out loud through ElevenLabs.

It's a plain web app with no build step. `robot/dashboard.py` serves it at **`/evera/`**.

## Quickest: run it from a laptop (no changes on the GMKtec)

```bash
python3 evera/serve.py http://169.233.72.19:8080 8765
```

It serves the app and passes `/api/*` and `/stream/*` through to the robot dashboard, so the two camera feeds work even on the old dashboard. On the iPad (same Wi-Fi), open `http://<laptop IP>:8765/` and Add to Home Screen. Over plain http the iPad blocks the mic, so voice needs the HTTPS tunnel below (point it at port 8765). In `keys.local.js`, `link: 'sim'` keeps arm actions simulated while the cameras stay live. Set `'auto'` to drive the real arm.

## Deploy to the GMKtec (from a laptop in this repo)

```bash
scp -r evera aup@<GMKTEC_IP>:~/hackathon/
scp robot/dashboard.py aup@<GMKTEC_IP>:~/hackathon/robot/dashboard.py
```

Then restart the dashboard (same command as in [docs/HANDOFF.md](../docs/HANDOFF.md)):

```bash
ssh aup@<GMKTEC_IP> 'cd ~/hackathon && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && kill $(cat ~/dashboard.pid); setsid nohup python robot/dashboard.py > ~/dashboard.log 2>&1 < /dev/null & echo $! > ~/dashboard.pid'
```

Quick check from a laptop on the same Wi-Fi: open **http://<GMKTEC_IP>:8080/evera/**. Touch control works over plain http.

## HTTPS for voice (needed on the iPad)

iPad Safari only gives a page the microphone over HTTPS. A Cloudflare quick tunnel gives you an https URL in one command. It needs venue internet, which the voice features need anyway.

On the GMKtec, the first time only:

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared && chmod +x ~/cloudflared
```

Every time:

```bash
~/cloudflared tunnel --url http://localhost:8080
```

It prints a URL like `https://something.trycloudflare.com`. On the iPad, open that URL with **`/evera/`** on the end. The URL changes every time the tunnel restarts.

**Warning:** this link is public. Anyone who has it can move the arm. Don't post it anywhere, and stop the tunnel (Ctrl-C) when you're not demoing.

## Set up the iPad

1. Open the tunnel URL in Safari, tap **Share → Add to Home Screen**, then launch **evera** from the home screen. It runs fullscreen.
2. Tap the gear and paste in the **OpenAI** key and the **ElevenLabs** key. They're saved on that iPad only and never in the code.
   - Defaults: chat `gpt-4.1-mini`, transcription `gpt-4o-mini-transcribe` (falls back to `whisper-1`), voice `JBFqnCBsd6RMkjVDRZzb` (a British male voice), model `eleven_flash_v2_5`.
3. Tap **Test voice**. Tap the screen once after launching too: iOS only allows audio and the mic after a tap.
4. **Hold the mic button** to talk, or turn on **WAKE** and say "evera, …". Hold-to-talk is the reliable one in a noisy room.

## Using it

- **Tap a tool:** the arm (folded at rest) dissolves and a 3D model of that tool materializes in the center, spinning on the pedestal. Its mode chips appear at the top. Tap a chip (e.g. *Hand over*, *Heat-set insert*) to run it: the arm comes back to perform the action, then the tool model returns. Tap the empty stage to go back to the arm.
- **Voice / brain:** evera is a general conversational AI (default: Groq `openai/gpt-oss-120b`, set in `keys.local.js`). It chats, answers questions, and for project or how-to questions ("I want to build a PCB") runs a live **web search** first (~7 s). It runs arm actions right away.
- **Teleop:** the **Teleop** button in the header (or say "start teleop") starts the dashboard's practice mode: move the leader arm and the follower copies it. **End teleop** or STOP ends it.
- **Got it / drop it:** during a hand-over, say it (or tap **Got it · release**) and the gripper opens and the arm returns to start. "Stop", "halt" or "freeze" is caught on the iPad itself and stops instantly, without waiting for GPT.
- **STOP** (header, plus the big red button while running), or **Esc** on a keyboard: sends `POST /api/key {k:"q"}`. The arm returns to its start pose.
- **Drag the 3D arm** to orbit it. Double-tap to reset the view. Tap a camera feed to enlarge it.
- **Without keys** it still works: the offline brain matches keywords and the voice falls back to the browser's.

## Make actions real: `actions.js`

Every mode is `{task, ckpt, secs, motion}` and gets sent as `POST /api/policy {ckpt, task, secs}`.

- **`task`** is the language prompt. pi0.5 reads it; ACT ignores it.
- **`ckpt`** is a checkpoint folder on the GMKtec. Empty means the dashboard's default.
- **`secs`** is how long the policy runs.
- **`motion`** is only how the 3D arm animates: `handover | press | hold | dock`.

Right now only *Soldering iron → Hand over* points at a real model (`ACT_PICK`): `rl_act`, the pick-and-place ACT trained on episodes 50–80. ACT ignores `task`, so it does the same pick whatever it's asked. Point other modes at new checkpoints as they get trained. Which checkpoint is best is tracked in [docs/HANDOFF.md](../docs/HANDOFF.md#trained-models).

## Simulator

Add `?sim=1` to the URL, or pick Settings → Robot link → Simulator, to demo everything without the arm. In **Auto** mode the app only simulates if the arm has never connected. If a connection drops later, it shows **Offline** instead, so commands never silently go to the simulator.

## Local development

```bash
python3 -m http.server 8765 --directory evera
```

Then open http://localhost:8765/?sim=1. Debug hashes for testing:
- `#fast` skips the boot screen
- `#sel=iron` selects a tool
- `#run=iron/heat` starts a run
- `#say=pass me the pliers` sends a voice command
- `#pose={"shoulder_lift":0,...}` holds a joint pose

## Files

| File | What it does |
|---|---|
| `index.html`, `app.css` | Layout and styling |
| `app.js` | Main controller: tool orbs, chips, run flow, conversation, settings, boot screen, animations |
| `arm3d.js` | three.js renderer: parses the SO-101 URDF, loads the STLs, pose timelines, floor rings, heat glow, arm ↔ tool swap |
| `tools3d.js` | Procedural 3D models of the 12 tools, built from three.js shapes (no downloaded assets) |
| `api.js` | Dashboard client plus the simulator |
| `brain.js` | OpenAI tool calling plus the offline keyword fallback |
| `voice.js` | Hold-to-talk (MediaRecorder → OpenAI transcription), wake word, ElevenLabs speech, waveform |
| `actions.js` | Tool and mode config (edit this one) |
| `serve.py` | Laptop server: the app plus a pass-through to the dashboard's API and camera streams |
| `assets/so101/` | URDF + STL meshes from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0) |
| `vendor/three/` | three.js r160 (MIT), so it runs without a CDN |
