# Build EVERA: a JARVIS-style iPad control app for a robot arm

> **Historical build spec.** This is the brief the iPad app was originally built from. The shipped app
> differs in places: three.js is vendored instead of loaded from jsdelivr, and each mode is a `{task, ckpt,
> secs, motion}` entry. For current behavior see [evera/README.md](../evera/README.md).

You are building **EVERA**, a futuristic "Tony Stark workshop" assistant that runs on an iPad propped next to a desktop robot arm (SO-101, 6-DOF, LeRobot). The user taps a tool, or says "EVERA, grab the soldering iron", and the arm runs that action. EVERA replies out loud in a witty British-butler voice. This is a hackathon demo that judges will see live, so it has to **look incredible and never crash**.

Read everything below before writing code.

---

## 1. Platform and delivery

- **A web app, not native.** It's opened in iPad Safari and installed with *Add to Home Screen*, so it runs fullscreen like an app.
- **No build step.** Use plain HTML, CSS and ES modules. Load libraries through an `<script type="importmap">` from jsdelivr with **exact pinned versions** (e.g. `three@0.160.0`). The whole app must deploy by copying a folder.
- **Target devices:** base iPad (1180×820 landscape) and iPad mini (1133×744 landscape). Design for **landscape only**. Show a "rotate your iPad" screen in portrait.
- **PWA polish:**
  - `manifest.json`, `apple-touch-icon` (an arc-reactor style icon), `apple-mobile-web-app-capable`, black-translucent status bar.
  - `viewport-fit=cover` with safe-area padding, no pinch zoom.
  - Disable text selection, long-press callouts and pull-to-refresh / overscroll.
- **File layout:**
  ```
  evera/
    index.html
    app.css
    app.js            ← state, polling, UI wiring
    arm3d.js          ← the 3D hologram arm (three.js)
    voice.js          ← wake word, push-to-talk, ElevenLabs TTS
    brain.js          ← Claude API + tool calling
    actions.js        ← THE CONFIG: tools, modes, checkpoints, prompts (editable by humans)
    manifest.json, icon-180.png, icon-512.png
    assets/so101/     ← URDF + STL meshes (see §4)
  ```

## 2. The backend you talk to (already exists, do not redesign it)

A Python stdlib HTTP server (`dashboard.py`) runs on the robot's mini PC (Ubuntu, `http://<GMKTEC_IP>:8080`). The app is served **from that same server** at `/evera/`, so every call is same-origin (no CORS).

| Call | Purpose | Details |
|---|---|---|
| `GET /api/status` | Poll every 500 ms | JSON: `{state, mode, running, message, episode, ep_time, busy, log:[...last 40 lines], counts, cams:{top:<seconds since last frame>, wrist:...}}`.<br>`state` ∈ `idle, starting, robot, saving, error, recording, reset, practice`.<br>`robot` = the policy is driving the arm. `saving` (in policy mode) = the arm is returning to rest. `starting` = the motor check and connect, about 10 s. |
| `POST /api/policy` | Run an action | Body `{"ckpt": "<path or empty>", "task": "<language prompt>", "secs": 30}`. Returns a JSON string: `"started"`, or an error such as `"a session is already running"` or `"camera down"`. Empty `ckpt` means the server default. |
| `POST /api/key` | Stop / E-STOP | Body `{"k":"q"}`. Sends SIGINT to the running policy, and the arm returns to its start pose. |
| `GET /stream/top`, `GET /stream/wrist` | Live cameras | MJPEG (`multipart/x-mixed-replace`), about 8 fps, 320×240. Just use `<img src>`. If a camera's age in `/api/status` is over 3 s, show a "SIGNAL LOST" overlay. |

**Backend change you must also write** (small, in `dashboard.py`): add a branch in `do_GET` that serves static files from `~/hackathon/evera/` for paths starting with `/evera/`. Requirements:
- correct MIME types (html, js as `text/javascript`, css, json, png, stl as `model/stl`, urdf as `application/xml`)
- `Cache-Control: no-cache`
- reject `..` in paths
- `/evera` redirects to `/evera/`

Give me that as a clean diff.

**Mock mode (required):** if `/api/status` fails, or `?mock=1` is in the URL, simulate the backend in JS so the whole UI can be developed and demoed on a laptop with no robot. The simulated run should go `starting` 3 s → `robot` for `secs` → `saving` 3 s → `idle`. Show a small "SIM" chip in the header when mocked.

## 3. Actions are config, not code: `actions.js`

The robot's skills are being trained right now (ACT and pi0.5 VLA policies), so **every action is just data**. The humans will edit this file during the event to point actions at new checkpoints. **All actions are always clickable.** Don't add readiness badges or disabled states.

```js
export const TOOLS = [
  { id:'iron', name:'Soldering Iron', sub:'Pinecil V2', bay:1, hot:true, modes:[
     { id:'pick', label:'Pick up & hand over', task:'pick up the soldering iron and hand it to me', ckpt:'', secs:30 },
     { id:'heat', label:'Heat-set insert',     task:'press the heat-set insert into the part with the soldering iron', ckpt:'', secs:45, motion:'press' } ]},
  { id:'drv',  name:'Screwdriver',    sub:'Precision',   bay:2,  modes:[pick('screwdriver')] },
  { id:'dmm',  name:'Multimeter',     sub:'Digital',     bay:3,  modes:[pick('multimeter')] },
  { id:'tape', name:'Measuring Tape', sub:'5 m',         bay:4,  modes:[pick('measuring tape')] },
  { id:'hex',  name:'Allen Key',      sub:'Low-grip',    bay:5,  modes:[pick('allen key')] },
  { id:'glue', name:'Hot Glue Gun',   sub:'Mini',        bay:6,  hot:true, modes:[pick('hot glue gun')] },
  { id:'plr',  name:'Pliers',         sub:'Needle-nose', bay:7,  modes:[pick('pliers')] },
  { id:'twz',  name:'Tweezers',       sub:'ESD',         bay:8,  modes:[pick('tweezers')] },
  { id:'cut',  name:'Flush Cutters',  sub:'Side-cut',    bay:9,  modes:[pick('flush cutters')] },
  { id:'cal',  name:'Calipers',       sub:'Digital',     bay:10, modes:[pick('calipers')] },
  { id:'hand', name:'Helping Hand',   sub:'Hold steady', bay:11, modes:[{ id:'hold', label:'Hold part steady', task:'hold the part steady', ckpt:'', secs:60, motion:'hold' }] },
  { id:'dock', name:'Return All',     sub:'Clean bench', bay:12, modes:[{ id:'dock', label:'Return all to dock', task:'return all tools to the dock', ckpt:'', secs:60, motion:'dock' }] },
];
// pick(n) => { id:'pick', label:'Pick up & hand over', task:`pick up the ${n} and hand it to me`, ckpt:'', secs:30, motion:'handover' }
```

**Running an action:**
1. If something is already running, POST stop first and wait for `idle`.
2. POST `/api/policy` with that mode's `{ckpt, task, secs}`.
3. Show any non-`"started"` reply in the transcript, in EVERA's voice ("The camera seems to be down, sir. Check the USB cable.").

**Hand-over flow:** after a `handover` action finishes, EVERA says "Here you go. Say 'got it' when you have it." The user's "got it" (voice or a button) currently just acknowledges. Leave a clearly marked `onGotIt()` hook for a future "release gripper" action.

## 4. THE CENTERPIECE: a real 3D holographic arm (three.js)

The center of the screen is a **live 3D hologram of the SO-101 arm**, not a 2D drawing. It's what judges will remember, so spend the most effort here.

**Model:**
- Use the official SO-101 URDF and STL meshes from the open-source repo `TheRobotStudio/SO-ARM100` (`Simulation/SO101/`, file `so101_new_calib.urdf` plus its `assets/*.stl`). Copy them into `evera/assets/so101/` so the app works offline.
- Load them with `urdf-loader` (gkjohnson, pinned version, via importmap).
- The joint names should be `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. **Verify the names and limits by reading the URDF**, and don't guess.
- **Fallback:** if loading fails, build a procedural low-poly SO-101 (boxes and cylinders with the same kinematic chain and proportions) so the panel is never empty.

**Hologram look:**
- Each mesh gets a custom additive **fresnel rim** ShaderMaterial: cyan `#5CE1FF`, a transparent core, `depthWrite:false`.
- Overlay `EdgesGeometry` lines on every mesh, plus a faint horizontal **scanline sweep** that moves up the model.
- **Bloom** with `UnrealBloomPass` at half resolution.
- **Floor:** a polar grid disc with two slowly counter-rotating dashed rings (like an arc reactor), and 12 small bay markers around the base, numbered 01–12.
- The **selected tool's bay** gets a pulsing reticle on the floor, plus a dashed glowing **bezier trajectory** from the gripper to that bay. Both update when the selection changes.
- **Hot-tool state:** while a `hot:true` tool is running, lerp the whole hologram (and the app accent color) from cyan to **amber `#FFA630`**. Add a heat shimmer on the gripper tip.
- **Error state:** a red flicker.
- The arm subtly flickers and materializes (a dissolve-in) when the app loads.

**Motion is honest: it comes from run state, never fake numbers.** There is no joint-angle feed yet, so the arm is driven by keyframed poses selected by `/api/status`:
- `idle`: rest pose with a gentle "breathing" idle and a slow auto-rotate of the camera
- `starting`: a boot scan, where a bright band sweeps the model base-to-tip while the joints twitch to their limits and back
- `robot`: play the selected mode's `motion` sequence over `secs`:
  - `handover`: rest → pan to the bay angle → reach down → close gripper → lift → swing to the hand-over pose toward the user → hold
  - `press`: grab → move over the work spot → a repeating down/up press loop on `wrist_flex` / `shoulder_lift`
  - `hold`: reach the center → hold steady with tiny stabilizing motion
  - `dock`: sweep through several bays
- `saving`: ease back to rest
- E-STOP: freeze the pose instantly and flash red

Use smooth interpolation between keyframes (slerp/ease, about 1–2 s per segment). Expose `arm.setJointTargets({shoulder_pan:..., ...})` so a future real-joints endpoint can drive it directly.

**Interaction and performance:**
- One-finger drag orbits (OrbitControls with damping and a limited polar angle). Pinch zooms within limits. Double-tap resets the view.
- Base iPads are not powerful:
  - `setPixelRatio(Math.min(devicePixelRatio, 2))`
  - pause rendering when `document.hidden`
  - auto-disable bloom if the average FPS drops below 40
  - keep the draw-call count low

## 5. Voice and brain

- **Persona:** EVERA is a calm, dry-witted British butler AI for a maker's bench. Replies are short and spoken aloud (1–2 sentences). She calls the user "sir" sparingly.
- **Brain, Claude:**
  - Call the Messages API directly from the browser: `POST https://api.anthropic.com/v1/messages` with headers `x-api-key`, `anthropic-version: 2023-06-01`, `anthropic-dangerous-direct-browser-access: true`.
  - Default model `claude-haiku-5-5` for speed, selectable in Settings (`claude-sonnet-5-5`, `claude-opus-5-5`).
  - Give Claude **tools**:
    - `run_action {tool_id, mode_id}`, with enums generated from `actions.js`
    - `stop_arm {}`
    - `got_it {}`
  - The system prompt includes the tool list and the current arm state. Claude can also just chat or answer bench questions ("what temp for M3 heat-set inserts in PLA?") without moving the arm.
  - **No confirmation step.** When the intent is clear, act immediately and say so ("Fetching the multimeter.").
  - Keep the last ~10 turns of history.
- **Voice out, ElevenLabs TTS:** use the low-latency model and a configurable `voice_id` (default: a British male voice). Check the current ElevenLabs API docs for exact endpoints and model names.
  - **iOS gotcha:** audio must be unlocked by a user gesture. On the first tap anywhere, create and resume an `AudioContext` and play a silent buffer.
  - While EVERA speaks, the arc-reactor button and a waveform animate.
- **Voice in:**
  - **Hold-to-talk:** a big glowing arc-reactor button. Hold to record, release to send. Prefer ElevenLabs Speech-to-Text (record with `MediaRecorder`, which is more robust in a noisy venue), and fall back to `webkitSpeechRecognition`.
  - **Wake word "EVERA"** (toggle in the UI, default ON): continuous `webkitSpeechRecognition`. Fuzzy-match variants like "evera / ever a / a vera / evra", and auto-restart on `end` and `error`.
    - On wake: a chime, the reactor brightens, and it captures the rest of the utterance (or the next one).
    - iOS Safari recognition is flaky, so the hold-to-talk button must always work even if the wake word dies. Show a small status for the wake listener.
- **Mic access requires HTTPS.** The app will be reached through a Cloudflare quick tunnel (`cloudflared tunnel --url http://localhost:8080` → `https://<random>.trycloudflare.com/evera/`). Over plain `http://`, detect that `getUserMedia` / `SpeechRecognition` is unavailable, show a clear "voice needs HTTPS" notice, and keep everything else working by touch.
- **Keys:**
  - A **Settings panel** (gear icon) where the user types the Anthropic API key, ElevenLabs API key, ElevenLabs voice ID and Claude model. Store them in `localStorage` (wrap in try/catch).
  - **Never hardcode keys**, and never commit them. The repo is public.
  - If the keys are missing, voice and brain show "configure in Settings", and touch control still works.

## 6. Layout and visual design (landscape, 1180×820)

The style is the classic JARVIS workshop HUD: deep black-navy background, glowing ice-cyan linework, thin technical type, and HUD corner brackets on panels.

- **Colors:** background `#02060C`, panel `rgba(9,28,44,.62)` with a `#17445C` border, accent `#5CE1FF`, text `#D6F6FF`, dim labels `#86AFC4`, OK green `#4CFFB0`, hot amber `#FFA630`, danger `#FF3B4E`.
- **Fonts (Google Fonts):** *Chakra Petch* for display and UI, *JetBrains Mono* for labels and data. Don't use Inter, Roboto or Arial, and no emoji anywhere. Icons are inline stroke SVGs.
- **Panels:** subtle 1 px borders with 16 px glowing corner brackets (top-left and bottom-right).

Layout, as a three-column grid under a 52 px header:

- **Header:**
  - Left: the EVERA wordmark (wide letter-spacing) with a small arc-reactor glyph and "BENCH INTELLIGENCE · SO-101".
  - Right: status chips, each with a live dot: ARM LINK, CAMS 2/2 (from `cams` ages), CLAUDE, VOICE, plus a SIM chip in mock mode and a Settings gear.
- **Left (≈290 px), TOOL BAY:**
  - A 2-column grid of 12 tool cards: icon, name, bay number, "pick up" or "2 modes".
  - The selected card glows (accent border, inner glow). Tapping selects it, and the 3D reticle moves to that bay.
- **Center:**
  - **Top (most of the height):** the **3D hologram arm** panel, with overlays: "HOLO-VIEW · FOLLOWER ARM" top-left, "STATE · STANDBY / EXECUTING / RETURNING / ERROR" top-right, and "TARGET LOCK · BAY 01 · SOLDERING IRON" bottom-left.
  - **Bottom (≈236 px), action panel:**
    - the selected tool's name
    - segmented mode buttons
    - a dashed mono line showing the exact task prompt and duration (`› task: "pick up the soldering iron and hand it to me" · 30 s`; judges love seeing this)
    - a big **ENGAGE** button. While running it becomes a blinking "EXECUTING 00:17" with a progress bar driven by elapsed time vs `secs`.
- **Right (≈300 px):**
  - Two camera panels (scene + wrist) with HUD brackets, a moving scanline and a LIVE dot
  - The EVERA transcript (chat bubbles: user right, EVERA left, plus system errors in amber; auto-scroll)
  - At the bottom: the **arc-reactor hold-to-talk** button (about 110 px), a wake-word toggle, and a big striped red **E-STOP** that is always visible and always works, even in mock mode or mid-speech.
- Every touch target is at least 44 px. Use subtle motion everywhere (rotating rings, scanlines, pulse on the active state), but keep it at 60 fps.

## 7. Quality bar / acceptance checklist

- [ ] Opens with `?mock=1` on a laptop, and every flow works (select, engage, run timeline, E-STOP, transcript, Settings).
- [ ] Served from `dashboard.py` at `/evera/`, polls real `/api/status`, and runs and stops real actions.
- [ ] The 3D arm loads the real SO-101 URDF (or the procedural fallback), looks like a hologram with bloom, animates per state and per motion type, turns amber for hot tools, and stays smooth on a base iPad.
- [ ] Touch-only works with no API keys. With keys over HTTPS, "EVERA, grab the multimeter" → Claude → `run_action` → the arm runs → EVERA speaks a reply.
- [ ] E-STOP is never blocked by any modal, loading state or audio.
- [ ] No keys in the source. All CDN versions are pinned. No console errors.
- [ ] A short `README` section: how to deploy (copy `evera/` to `~/hackathon/evera/`, restart the dashboard), start the tunnel, add to the Home Screen, enter keys, and edit `actions.js`.

Build it file by file, starting with `actions.js`, mock mode and the layout shell, then the 3D arm, then voice and brain. Show me each file in full.
