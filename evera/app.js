// EVERA main controller: layout + motion, run flow, conversation, voice, settings.
import { TOOLS, LEFT, RIGHT, toolById } from './actions.js';
import { icon } from './icons.js';
import { Arm3D } from './arm3d.js';
import { Backend } from './api.js';
import { Brain, RELEASE } from './brain.js';
import { Voice } from './voice.js';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const easeOutBack = (t) => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2);
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

// ------------------------------------------------------------------ settings
const DEFAULTS = {
  llmProvider: 'openai', openaiKey: '', openaiModel: '', sttModel: 'gpt-4o-mini-transcribe',
  elevenKey: '', elevenVoice: 'JBFqnCBsd6RMkjVDRZzb', elevenModel: 'eleven_flash_v2_5',
  link: 'auto', backend: '', wake: true,
};
let settings = { ...DEFAULTS };
try { settings = { ...DEFAULTS, ...JSON.parse(localStorage.getItem('evera.settings') || '{}') }; } catch { /* private mode */ }
// keys.local.js (gitignored) pre-fills the keys so the app works with no Settings step.
try {
  const k = (await import('./keys.local.js')).default || {};
  const map = { llmProvider: k.llmProvider, openaiKey: k.llmKey, openaiModel: k.llmModel, elevenKey: k.elevenKey, elevenVoice: k.elevenVoice, link: k.link, backend: k.dashboard };
  for (const [f, v] of Object.entries(map)) if (v) settings[f] = v;
} catch { /* no local keys file */ }
const saveSettings = () => { try { localStorage.setItem('evera.settings', JSON.stringify(settings)); } catch { /* ignore */ } };
const getSettings = () => settings;

// ------------------------------------------------------------------ state
const S = { sel: null, run: null, status: null, voice: 'idle', awaitGotIt: null, booted: false, toolPan: 0, lastErr: '' };
const stage = $('#stage');
const toolsEl = $('#tools');
const modesEl = $('#modes');
const tools = {};
let pointer = { x: -1e4, y: -1e4 };

// ------------------------------------------------------------------ 3D arm
const arm = new Arm3D($('#arm'));
window.evera = { arm, S, say: (t) => handleUtterance(t) };
const armReady = arm.load('assets/so101/so101.urdf', (p) => {
  $('#armpct').textContent = `Loading arm ${Math.round(p * 100)}%`;
  bootLine('arm', `SO-101 model ${Math.round(p * 100)}%`, p >= 1);
}).then(() => $('#armload').classList.add('done')).catch((e) => {
  console.error(e);
  $('#armpct').textContent = 'Arm model failed to load';
  toast('Could not load the 3D arm model', 'err');
});

// ------------------------------------------------------------------ backend, brain, voice
const backend = new Backend(getSettings, onStatus);
const brain = new Brain(getSettings);
const voice = new Voice(getSettings, {
  state: setVoiceState,
  interim: (t) => setVText(t),
  heard: (t) => showHeard(t),
  command: (t) => handleUtterance(t),
  error: (m) => toast(m, 'warn'),
});

// ------------------------------------------------------------------ tools
function buildTools() {
  [...LEFT, ...RIGHT].forEach((id, i) => {
    const t = toolById(id);
    const el = document.createElement('button');
    el.className = 'tool' + (t.hot ? ' hot-tag' : '');
    el.setAttribute('aria-label', t.name);
    el.innerHTML = `<span class="orb">${icon(t.icon, 40).replace('<svg', '<svg class="ic"')}<svg class="prog" viewBox="0 0 120 120"><circle cx="60" cy="60" r="57" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100"/></svg></span><span class="label">${t.name}</span>`;
    el.addEventListener('pointerdown', (e) => ripple(e, el.querySelector('.orb')));
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      voice.unlock();
      selectTool(id, { toggle: true });
    });
    toolsEl.append(el);
    tools[id] = { id, el, side: LEFT.includes(id) ? -1 : 1, idx: i, x: 0, y: 0, tx: 0, ty: 0, phase: Math.random() * Math.PI * 2, born: 0 };
  });
}

function layoutTools() {
  const w = stage.clientWidth, h = stage.clientHeight;
  const cx = w / 2, cy = h * 0.5;
  const rx = Math.min(w * 0.4, h * 0.6), ry = h * 0.43;
  const a0 = 30, a1 = 150;
  const place = (ids, side) => ids.forEach((id, i) => {
    const th = ((a0 + ((a1 - a0) * i) / (ids.length - 1)) * Math.PI) / 180;
    tools[id].tx = cx + side * rx * Math.sin(th);
    tools[id].ty = cy - ry * Math.cos(th);
  });
  place(LEFT, -1);
  place(RIGHT, 1);
  S.center = { x: cx, y: cy };
}

function ripple(e, orb) {
  const r = orb.getBoundingClientRect();
  const s = document.createElement('span');
  const d = Math.max(r.width, r.height) * 2.2;
  s.className = 'ripple';
  Object.assign(s.style, { left: `${e.clientX - r.left}px`, top: `${e.clientY - r.top}px`, width: `${d}px`, height: `${d}px` });
  orb.append(s);
  setTimeout(() => s.remove(), 700);
}

function selectTool(id, { toggle = false } = {}) {
  if (toggle && S.sel === id && !S.run) return deselect();
  S.sel = id;
  for (const t of Object.values(tools)) t.el.classList.toggle('sel', t.id === id);
  toolsEl.classList.add('has-sel');
  stage.classList.add('has-sel');
  if (!S.run && !S.teleop) arm.setDisplay(id); // swap the centre to this tool's 3D model
  buildModes();
}

// After a run, bring the selected tool's model back once the arm has folded away.
function showSelectedSoon(ms = 2400) {
  clearTimeout(S.showT);
  S.showT = setTimeout(() => { if (S.sel && !S.run && !S.teleop) arm.setDisplay(S.sel); }, ms);
}

function deselect() {
  S.sel = null;
  for (const t of Object.values(tools)) t.el.classList.remove('sel');
  toolsEl.classList.remove('has-sel');
  stage.classList.remove('has-sel');
  modesEl.classList.remove('show');
  arm.setDisplay(null);
  if (!S.run) arm.idle();
}

function buildModes() {
  const tool = toolById(S.sel);
  modesEl.classList.remove('show');
  modesEl.innerHTML = '';
  tool.modes.forEach((m, i) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.style.setProperty('--i', i);
    b.innerHTML = `<span class="pl">${icon('play', 16, 2.2)}</span><span><b>${m.label}</b><small>${m.secs} s · “${m.task}”</small></span>`;
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      runMode(tool, m);
    });
    modesEl.append(b);
  });
  requestAnimationFrame(() => requestAnimationFrame(() => updateModesVisibility()));
}

function updateModesVisibility() {
  const show = S.sel && !S.teleop && !S.teleopPending && !(S.run && S.run.tool.id === S.sel);
  modesEl.classList.toggle('show', !!show);
}

// Tap empty stage = deselect; double tap = reset the camera.
{
  let down = null, lastTap = 0;
  stage.addEventListener('pointerdown', (e) => (down = { x: e.clientX, y: e.clientY, t: performance.now() }));
  stage.addEventListener('pointerup', (e) => {
    if (!down || e.target !== $('#arm')) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    if (moved < 6 && performance.now() - down.t < 350) {
      const now = performance.now();
      if (now - lastTap < 320) arm.resetView();
      else if (S.sel && !S.run) deselect();
      lastTap = now;
    }
    down = null;
  });
  stage.addEventListener('pointermove', (e) => {
    const r = stage.getBoundingClientRect();
    pointer = { x: e.clientX - r.left, y: e.clientY - r.top };
  });
  stage.addEventListener('pointerleave', () => (pointer = { x: -1e4, y: -1e4 }));
}

// ------------------------------------------------------------------ running actions
function defaultSay(tool, mode) {
  if (mode.motion === 'handover') return `Fetching the ${tool.name.toLowerCase()}.`;
  if (mode.motion === 'press') return 'Heat-set mode. Fingers clear, please.';
  if (mode.motion === 'hold') return 'Holding it steady. Work away.';
  if (mode.motion === 'dock') return 'Tidying up the bench.';
  return `${mode.label}. On it.`;
}

async function runMode(tool, mode, { say } = {}) {
  voice.unlock();
  selectTool(tool.id);
  S.toolPan = arm.solveAim(tools[tool.id].tx, tools[tool.id].ty).pan;
  clearTimeout(S.showT);
  arm.setDisplay(null); // the arm comes back to perform the action
  if (backend.status.running) {
    if (S.run) S.run.stopped = true;
    await backend.stop().catch(() => {});
    arm.home();
    await backend.waitIdle();
  }
  S.run = { tool, mode, phase: 'launch', t0: null, launched: performance.now(), stopped: false };
  S.awaitGotIt = null;
  updateModesVisibility();
  renderRun();
  let res;
  try { res = await backend.run(mode); } catch (e) { res = e.message || 'network error'; }
  if (res !== 'started') {
    S.run = null;
    renderRun();
    updateModesVisibility();
    arm.setDisplay(tool.id);
    sysMsg(`Couldn't start: ${res}`);
    evera(res === 'camera down' ? 'A camera seems to be down. Check its USB cable.' : `I couldn't start that: ${res}.`);
    return;
  }
  arm.scan();
  if (say !== false) evera(say || defaultSay(tool, mode));
}

async function estop() {
  voice.stopSpeaking();
  arm.freeze();
  document.body.classList.remove('flash');
  void document.body.offsetWidth;
  document.body.classList.add('flash');
  setTimeout(() => document.body.classList.remove('flash'), 1300);
  if (S.run) S.run.stopped = true;
  S.awaitGotIt = null;
  try { await backend.stop(); } catch { /* offline */ }
  evera('All motion halted.');
  setTimeout(() => { if (arm.mode === 'frozen') arm.home(); }, 900);
}

// Teleop: the dashboard's practice mode (the leader arm drives the follower; nothing recorded).
async function startTeleop({ say = true } = {}) {
  voice.unlock();
  if (S.teleop) return say && evera('Teleop is already on. Move the leader arm.');
  // Never interrupt something evera didn't start (e.g. a teammate recording on the dashboard).
  if (backend.real?.running && !S.run) {
    sysMsg(`Dashboard busy: ${backend.real.message || backend.real.state}`);
    return evera('The arm is busy on the dashboard right now. Stop that session there first.');
  }
  if (S.run) {
    S.run.stopped = true;
    await backend.stop().catch(() => {});
    arm.home();
    await backend.waitIdle();
  }
  let res;
  try { res = await backend.teleop(); } catch (e) { res = e.message || 'network error'; }
  if (res !== 'started') {
    sysMsg(`Teleop didn't start: ${res}`);
    return evera(res === 'camera down' ? "A camera seems to be down, so the dashboard won't start teleop." : `Teleop didn't start: ${res}.`);
  }
  S.teleopPending = true;
  document.body.classList.add('teleop');
  updateModesVisibility();
  if (say) evera('Teleop engaged. Move the leader arm and the follower will copy you.');
}

async function stopTeleop({ say = true } = {}) {
  if (!S.teleop && !S.teleopPending) return say && evera('Teleop is not running.');
  try { await backend.stop(); } catch { /* offline */ }
  if (say) evera('Teleop ended. Arm returning to rest.');
}

$('#teleop').addEventListener('click', () => (S.teleop || S.teleopPending ? stopTeleop() : startTeleop()));

// "Got it" / "drop it": open the gripper and send the arm back to its start position.
// (Stopping the policy makes LeRobot return the arm to its start pose.)
async function release({ say = true } = {}) {
  S.awaitGotIt = null;
  renderRun();
  const r = S.run;
  if (r && r.phase !== 'saving') {
    r.stopped = true;
    r.released = true;
    arm.release();
    if (say) evera('Releasing it now.');
    try { await backend.stop(); } catch { /* offline */ }
  } else if (say) evera(r ? 'Already on its way back.' : 'Nothing in hand at the moment.');
}

function onStatus(s) {
  S.status = s;
  renderConn(s);
  renderCams(s);
  const tele = s.state === 'practice';
  if (tele && !S.teleop) {
    S.teleop = { t0: performance.now() };
    S.teleopPending = false;
    arm.setDisplay(null);
    arm.teleop();
  } else if (!tele && S.teleop && s.state !== 'offline') {
    S.teleop = null;
    arm.home();
    if (S.sel) showSelectedSoon();
  }
  if (S.teleopPending && !['starting', 'practice'].includes(s.state) && !s.running) S.teleopPending = false;
  document.body.classList.toggle('teleop', !!S.teleop || !!S.teleopPending);
  const r = S.run;
  if (s.state === 'starting') {
    if (r && r.phase === 'launch') r.phase = 'starting';
  } else if (s.state === 'robot') {
    if (!r) S.run = { tool: toolById(S.sel) || TOOLS[0], mode: (toolById(S.sel) || TOOLS[0]).modes[0], phase: 'launch', external: true, launched: performance.now() };
    const run = S.run;
    if (run.phase !== 'robot') {
      run.phase = 'robot';
      run.t0 = performance.now();
      arm.play(run.mode.motion, { toolPan: S.toolPan, userPan: arm.userPan() });
      arm.setHot(!!run.tool.hot);
      document.body.classList.toggle('hot', !!run.tool.hot);
    }
  } else if (s.state === 'saving') {
    if (r && r.phase !== 'saving' && r.phase !== 'launch') {
      r.phase = 'saving';
      if (arm.mode !== 'home') arm.home();
      arm.setHot(false);
      document.body.classList.remove('hot');
    }
  } else if (r && s.state !== 'offline' && (r.phase !== 'launch' || performance.now() - r.launched > 15000)) {
    finishRun(r);
  }
  if (s.state === 'error' && s.message && s.message !== S.lastErr) {
    S.lastErr = s.message;
    toast(s.message, 'err');
    sysMsg(s.message);
  }
  renderRun();
  updateModesVisibility();
}

function finishRun(r) {
  const reachedRobot = !!r.t0;
  S.run = null;
  arm.setHot(false);
  document.body.classList.remove('hot');
  if (arm.mode === 'run' || arm.mode === 'frozen') arm.home();
  else if (arm.mode !== 'home') arm.idle();
  if (S.sel) showSelectedSoon();
  S.awaitGotIt = null;
  if (!r.stopped && reachedRobot) evera(r.mode.motion === 'press' ? 'Insert seated.' : r.offered ? 'Taking it back. Arm is at rest.' : 'Done.');
}

function renderRun() {
  const r = S.run;
  document.body.classList.toggle('running', !!r || !!S.teleop);
  document.body.classList.toggle('offering', !!(r && S.awaitGotIt));
  for (const t of Object.values(tools)) t.el.classList.toggle('runningtool', !!r && t.id === r.tool.id);
}

// ------------------------------------------------------------------ header / cams
function renderConn(s) {
  const c = $('#conn');
  c.classList.toggle('ok', s.connected || !!s.robot);
  c.classList.toggle('sim', s.sim && !s.robot);
  c.classList.toggle('off', !s.connected && !s.sim);
  const txt = s.robot && s.sim ? 'Connected · sim actions' : s.connected ? 'Connected' : s.sim ? 'Simulator' : 'Offline';
  if ($('#conntext').dataset.v !== txt) scramble($('#conntext'), txt);
}

const camState = {};
function renderCams(s) {
  for (const name of ['top', 'wrist']) {
    const feed = $(`.feed[data-cam="${name}"]`);
    const img = feed.querySelector('img');
    const url = backend.camUrl(name);
    const st = (camState[name] ||= { url: null, ok: false });
    if (url !== st.url) {
      st.url = url;
      st.ok = false;
      if (url) {
        img.onerror = () => { st.ok = false; setTimeout(() => { if (st.url === url) img.src = `${url}?t=${Date.now()}`; }, 3000); };
        img.src = url;
      } else img.removeAttribute('src');
    }
    // MJPEG <img> streams don't reliably fire "load": a decoded frame (naturalWidth) means it's live.
    if (url && img.naturalWidth > 0) st.ok = true;
    const age = s.connected ? s.cams?.[name] : null;
    const live = !!url && st.ok && (age == null || age < 3);
    feed.classList.toggle('ok', live);
    $(`#live-${name}`).classList.toggle('off', !live);
    $(`#live-${name}`).textContent = live ? 'LIVE' : 'OFFLINE';
    feed.querySelector('.nosig').textContent = url && age > 3 ? 'Signal lost' : 'Connecting to camera…';
  }
}

// ------------------------------------------------------------------ conversation
function addMsg(who, text) {
  const el = document.createElement('div');
  el.className = `msg ${who}`;
  el.innerHTML = `<span class="who">${who === 'you' ? 'You' : who === 'sys' ? 'Arm' : 'evera'}</span><div class="bub"></div>`;
  const bub = el.querySelector('.bub');
  bub.textContent = text;
  const convo = $('#convo');
  convo.append(el);
  while (convo.children.length > 30) convo.firstChild.remove();
  convo.scrollTo({ top: convo.scrollHeight, behavior: 'smooth' });
  return bub;
}

function typeInto(bub, text, cps = 38) {
  bub.textContent = '';
  const caret = document.createElement('span');
  caret.className = 'caret';
  let i = 0;
  const step = () => {
    i = Math.min(text.length, i + Math.max(1, Math.round(cps / 30)));
    bub.textContent = text.slice(0, i);
    if (i < text.length) {
      bub.append(caret);
      setTimeout(step, 1000 / 30);
    }
    $('#convo').scrollTop = $('#convo').scrollHeight;
  };
  step();
}

function evera(text, { speak = true } = {}) {
  const bub = addMsg('evera', '');
  typeInto(bub, text);
  if (S.voice === 'idle' || S.voice === 'thinking') setVText(text, false);
  if (speak) voice.speak(text);
}
const sysMsg = (t) => addMsg('sys', t);

function armContext() {
  const r = S.run;
  let s = S.teleop ? 'in teleop (the user is driving it with the leader arm)' : r ? `running "${r.mode.label}" with the ${r.tool.name} (${r.phase})` : 'idle';
  if (!r && !S.teleop && S.sel) s += `, ${toolById(S.sel).name} selected on screen`;
  if (S.awaitGotIt) s += `, waiting for the user to take the ${toolById(S.awaitGotIt).name}`;
  if (S.status && !S.status.connected) s += S.status.sim ? ' (simulator mode)' : ' (robot offline)';
  return s;
}

async function handleUtterance(text) {
  text = (text || '').trim();
  if (!text) return setVoiceState('idle');
  addMsg('you', text);
  setVText(text, false);
  if (/\b(stop|halt|freeze|abort|emergency)\b/i.test(text)) return estop();
  if (RELEASE.test(text.toLowerCase())) return release();
  setVoiceState('thinking');
  let out;
  try {
    out = await brain.think(text, armContext());
    if (S.brainErr) { S.brainErr = null; $('#brainbadge').classList.remove('on'); }
  } catch (e) {
    if (e.message !== S.brainErr) toast(`Brain offline, using keywords: ${e.message}`, 'warn');
    S.brainErr = e.message;
    $('#brainbadge').classList.add('on');
    $('#brainbadge').title = e.message;
    out = brain.fallback(text);
  }
  let research = null;
  for (const c of out.calls) {
    if (c.name === 'research') research = c.args;
    else if (c.name === 'start_teleop') startTeleop({ say: !out.say });
    else if (c.name === 'stop_teleop') stopTeleop({ say: !out.say });
    else if (c.name === 'run_action') {
      const tool = toolById(c.args.tool_id);
      if (!tool) continue;
      const mode = tool.modes.find((m) => m.id === c.args.mode_id) || tool.modes[0];
      runMode(tool, mode, { say: false });
    } else if (c.name === 'stop_arm') {
      return estop();
    } else if (c.name === 'got_it') {
      release({ say: !out.say });
    }
  }
  if (research) return doResearch(research.query || text, out.say || research.say);
  if (out.say) evera(out.say);
  else setVoiceState('idle');
}

// Web research: say a short line, search, then answer (full text in the chat, spoken too).
async function doResearch(query, opener) {
  evera(opener || 'Let me look into that.');
  S.researching = true;
  setVoiceState('researching');
  try {
    const answer = await brain.research(query);
    S.researching = false;
    evera(answer);
  } catch (e) {
    S.researching = false;
    toast(`Research failed: ${e.message}`, 'warn');
    evera("I couldn't reach the web just now. Try me again in a moment.");
  }
}

// ------------------------------------------------------------------ voice card
const VSTATE = {
  idle: ['Ready', () => (S.wakeOn ? 'Say “evera…” or hold the mic' : 'Hold the mic to talk'), 'Hold to talk'],
  listening: ['Listening', null, 'Release to send'],
  thinking: ['Thinking', null, 'One moment'],
  researching: ['Researching', null, 'Searching the web…'],
  speaking: ['evera', null, 'Tap the mic to interrupt'],
};
function setVoiceState(st) {
  if (st === 'idle' && S.researching) st = 'researching';
  S.voice = st;
  const vc = $('#vc');
  vc.className = `vc ${st}`;
  document.body.classList.remove('v-idle', 'v-listening', 'v-thinking', 'v-speaking');
  document.body.classList.add(`v-${st}`);
  const [label, ph, hint] = VSTATE[st];
  $('#vstate').textContent = label;
  $('#vhint').textContent = st === 'listening' && !$('#mic').classList.contains('held') ? 'Go ahead' : hint;
  if (st === 'listening') setVText('…', false);
  if (ph) setVText(ph(), true);
}
// Show what the wake-word listener hears, so you can tell it's working.
function showHeard(t) {
  if (S.voice !== 'idle' || !t) return;
  const h = $('#vhint');
  h.textContent = `Heard: “${t.slice(-48)}”`;
  clearTimeout(S.heardT);
  S.heardT = setTimeout(() => { if (S.voice === 'idle') h.textContent = VSTATE.idle[2]; }, 2500);
}

function setVText(t, placeholder = false) {
  const el = $('#vtext');
  el.textContent = t;
  el.classList.toggle('ph', placeholder);
}

{
  const mic = $('#mic');
  mic.innerHTML = icon('mic', 26, 1.8);
  let holding = false;
  const down = async (e) => {
    e.preventDefault();
    voice.unlock();
    if (S.voice === 'speaking') voice.stopSpeaking();
    holding = true;
    mic.classList.add('held');
    if (e.pointerType) try { mic.setPointerCapture(e.pointerId); } catch { /* not a real pointer */ }
    const ok = await voice.startHold();
    if (!ok) { holding = false; mic.classList.remove('held'); }
    else if (!holding) up();
  };
  const up = async () => {
    if (!mic.classList.contains('held')) return;
    holding = false;
    mic.classList.remove('held');
    if (!voice.recording) return;
    const text = await voice.stopHold();
    if (text) handleUtterance(text);
  };
  mic.addEventListener('pointerdown', down);
  mic.addEventListener('pointerup', () => { holding = false; up(); });
  mic.addEventListener('pointercancel', () => { holding = false; up(); });
  mic.addEventListener('contextmenu', (e) => e.preventDefault());
  // Desktop: hold Space to talk, Esc = stop.
  window.addEventListener('keydown', (e) => {
    if (e.target.closest('input,select')) return;
    if (e.code === 'Space') {
      e.preventDefault(); // don't "click" whatever button has focus
      if (!e.repeat && !holding) down({ preventDefault() {} });
    }
    if (e.code === 'Escape') estop();
  });
  window.addEventListener('keyup', (e) => {
    if (e.code !== 'Space' || e.target.closest('input,select')) return;
    e.preventDefault();
    holding = false;
    up();
  });
}

function setWake(on) {
  S.wakeOn = on && voice.wakeSupported;
  settings.wake = on;
  saveSettings();
  const b = $('#wakebtn');
  b.textContent = S.wakeOn ? 'WAKE ON' : 'WAKE OFF';
  b.classList.toggle('on', S.wakeOn);
  if (S.gesture) voice.setWake(S.wakeOn); // iOS needs a tap before it will open the mic
  if (S.voice === 'idle') setVoiceState('idle');
}
$('#wakebtn').addEventListener('click', () => {
  voice.unlock();
  if (!voice.wakeSupported) return toast(voice.secure ? 'This browser has no speech recognition.' : 'Wake word needs HTTPS (open EVERA via the tunnel link).', 'warn');
  setWake(!S.wakeOn);
});

// Waveform: real audio when listening/speaking, a calm idle line otherwise.
const wave = $('#wave'), wctx = wave.getContext('2d');
let waveSmooth = new Float32Array(44);
function drawWave(t) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const W = wave.clientWidth, H = wave.clientHeight;
  if (wave.width !== W * dpr) { wave.width = W * dpr; wave.height = H * dpr; }
  wctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  wctx.clearRect(0, 0, W, H);
  const n = waveSmooth.length, gap = W / n, lv = voice.levels();
  const grad = wctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#3fb4ea');
  grad.addColorStop(0.5, '#a5e4ff');
  grad.addColorStop(1, '#3fb4ea');
  wctx.fillStyle = grad;
  for (let i = 0; i < n; i++) {
    const mid = Math.abs(i - n / 2) / (n / 2);
    let v;
    if (lv) {
      const bin = Math.floor(2 + Math.pow(1 - mid, 1.4) * 60);
      v = Math.pow(lv[bin] / 255, 1.3) * (1 - mid * 0.35);
    } else if (S.voice === 'thinking') v = 0.18 + 0.5 * Math.max(0, Math.sin(t * 6 - i * 0.45)) * (1 - mid * 0.5);
    else if (S.voice === 'listening') v = 0.1 + 0.08 * Math.sin(t * 4 + i);
    else v = 0.06 + 0.05 * Math.sin(t * 1.6 + i * 0.5) * (1 - mid);
    waveSmooth[i] += (v - waveSmooth[i]) * 0.35;
    const bh = Math.max(3, waveSmooth[i] * H);
    const x = i * gap + gap * 0.3, bw = Math.max(2, gap * 0.42);
    wctx.globalAlpha = 0.45 + 0.55 * (1 - mid);
    roundRect(wctx, x, (H - bh) / 2, bw, bh, bw / 2);
  }
  wctx.globalAlpha = 1;
}
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.fill();
}

// ------------------------------------------------------------------ connector line
const linkPath = $('#link-path'), linkGlow = $('#link-glow'), linkDots = $('#link-dots'), target = $('#target');
const lgrad = $('#lg-link');
for (let i = 0; i < 4; i++) {
  const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  c.setAttribute('r', i === 0 ? 3 : 2);
  linkDots.append(c);
}
function drawLink(t) {
  if (!S.sel || !arm.loaded) return;
  const tl = tools[S.sel];
  const tip = arm.tipScreen();
  if (!tip) return;
  const sx = tl.x, sy = tl.y;
  const dx = tip.x - sx, dy = tip.y - sy;
  const len = Math.hypot(dx, dy) || 1;
  const r0 = tl.el.classList.contains('sel') ? 64 : 48;
  const ax = sx + (dx / len) * r0, ay = sy + (dy / len) * r0;
  const c1x = ax + dx * 0.35, c1y = ay - 60;
  const c2x = tip.x - dx * 0.25, c2y = tip.y - 40;
  const d = `M${ax.toFixed(1)},${ay.toFixed(1)} C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${tip.x.toFixed(1)},${tip.y.toFixed(1)}`;
  linkPath.setAttribute('d', d);
  linkGlow.setAttribute('d', d);
  lgrad.setAttribute('x1', ax); lgrad.setAttribute('y1', ay); lgrad.setAttribute('x2', tip.x); lgrad.setAttribute('y2', tip.y);
  linkPath.style.strokeDashoffset = String(-t * (S.run ? 40 : 18));
  target.setAttribute('transform', `translate(${tip.x.toFixed(1)},${tip.y.toFixed(1)}) scale(${1 + 0.15 * Math.sin(t * 4)})`);
  const L = linkPath.getTotalLength();
  [...linkDots.children].forEach((c, i) => {
    const f = (t * (S.run ? 0.55 : 0.28) + i / 4) % 1;
    const p = linkPath.getPointAtLength(f * L);
    c.setAttribute('cx', p.x); c.setAttribute('cy', p.y);
    c.setAttribute('opacity', String(Math.sin(f * Math.PI)));
  });
}

// ------------------------------------------------------------------ per-frame motion
let lastAz = null, parallax = 0, t0 = performance.now();
arm.onFrame = () => {
  const now = performance.now();
  const t = (now - t0) / 1000;

  // Tool orbs: spring in from the centre, then float; drift with the 3D camera for depth.
  const az = arm.controls.getAzimuthalAngle();
  if (lastAz != null) parallax = clamp(parallax + Math.atan2(Math.sin(az - lastAz), Math.cos(az - lastAz)) * 60, -18, 18);
  lastAz = az;
  parallax *= 0.96;
  for (const tl of Object.values(tools)) {
    let p = 1;
    if (S.booted) {
      if (!tl.born) tl.born = now + tl.idx * 55;
      p = clamp((now - tl.born) / 700, 0, 1);
    } else p = 0;
    const e = p > 0 ? easeOutBack(p) : 0;
    const cx = S.center?.x ?? tl.tx, cy = S.center?.y ?? tl.ty;
    let x = cx + (tl.tx - cx) * e + Math.cos(t * 0.6 + tl.phase) * 2.5 + parallax * (tl.ty / stage.clientHeight - 0.5) * -1.2;
    let y = cy + (tl.ty - cy) * e + Math.sin(t * 0.8 + tl.phase) * 4;
    const md = Math.hypot(pointer.x - x, pointer.y - y);
    if (md < 150) {
      const k = (1 - md / 150) * 0.14;
      x += (pointer.x - x) * k;
      y += (pointer.y - y) * k;
    }
    tl.x = x; tl.y = y;
    tl.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) scale(${(0.4 + 0.6 * e).toFixed(3)})`;
    tl.el.style.opacity = String(clamp(p * 1.6, 0, 1));
  }

  // Mode chips follow the selected orb.
  if (S.sel) {
    // Mode chips: centred at the top of the stage, above the tool's 3D model.
    const h = modesEl.offsetHeight, w = modesEl.offsetWidth;
    const mx = S.center.x - w / 2;
    const my = clamp(stage.clientHeight * 0.1, 12, 90);
    modesEl.style.transform = `translate3d(${mx.toFixed(1)}px, ${my.toFixed(1)}px, 0)`;
  }

  // Run progress + labels.
  const r = S.run;
  const label = $('#armlabel');
  let lab = S.sel && !S.run ? toolById(S.sel).name : 'SO-101 · Follower', pill = '';
  if (S.teleop) {
    const el = (now - S.teleop.t0) / 1000;
    lab = `Teleop · ${mmss(el)}`;
    pill = `Teleop · leader arm · ${mmss(el)}`;
  } else if (r) {
    const name = `${r.tool.name} · ${r.mode.label}`;
    if (r.phase === 'robot') {
      const el = (now - r.t0) / 1000;
      lab = `${r.mode.label} · ${mmss(el)}`;
      pill = `${name} · ${mmss(el)}`;
      if (r.mode.motion === 'handover' && !r.offered && !r.stopped && el > 5.8) {
        r.offered = true;
        S.awaitGotIt = r.tool.id;
        renderRun();
        evera('Here you go. Say “got it” or “drop it” when you have it.');
      }
      const circ = tools[r.tool.id]?.el.querySelector('.prog circle');
      if (circ) circ.style.strokeDashoffset = String(100 - clamp((el / r.mode.secs) * 100, 0, 100));
    } else if (r.phase === 'saving') { lab = 'Returning to rest'; pill = `${name} · returning`; }
    else { lab = 'Connecting to arm…'; pill = `${name} · starting`; }
  }
  if (label.dataset.v !== lab) { label.dataset.v = lab; label.textContent = lab; }
  $('#runtext').textContent = pill;
  const base = arm.baseScreen();
  label.style.transform = `translate3d(${base.x.toFixed(1)}px, ${(base.y + 22).toFixed(1)}px, 0) translateX(-50%)`;

  drawLink(t);
  drawWave(t);
  drawBg(t);
  drawNoise(t);
};

// ------------------------------------------------------------------ background particles
const bg = $('#bg'), bctx = bg.getContext('2d');
const parts = Array.from({ length: 64 }, () => ({ x: Math.random(), y: Math.random(), vx: (Math.random() - 0.5) * 0.00012, vy: (Math.random() - 0.5) * 0.00012, r: Math.random() * 1.4 + 0.4 }));
let bgFrame = 0;
function drawBg() {
  if (++bgFrame % 2) return; // 30 fps is plenty
  const dpr = Math.min(devicePixelRatio || 1, 1.5);
  const W = innerWidth, H = innerHeight;
  if (bg.width !== Math.round(W * dpr)) { bg.width = Math.round(W * dpr); bg.height = Math.round(H * dpr); }
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, W, H);
  for (const p of parts) {
    p.x = (p.x + p.vx * 2 + 1) % 1;
    p.y = (p.y + p.vy * 2 + 1) % 1;
  }
  bctx.lineWidth = 0.6;
  for (let i = 0; i < parts.length; i++) {
    const a = parts[i];
    for (let j = i + 1; j < parts.length; j++) {
      const b = parts[j];
      const d = Math.hypot((a.x - b.x) * W, (a.y - b.y) * H);
      if (d < 120) {
        bctx.strokeStyle = `rgba(127,211,247,${0.07 * (1 - d / 120)})`;
        bctx.beginPath();
        bctx.moveTo(a.x * W, a.y * H);
        bctx.lineTo(b.x * W, b.y * H);
        bctx.stroke();
      }
    }
    bctx.fillStyle = 'rgba(127,211,247,0.28)';
    bctx.beginPath();
    bctx.arc(a.x * W, a.y * H, a.r, 0, Math.PI * 2);
    bctx.fill();
  }
}

// Static noise in camera panels without a signal.
let noiseFrame = 0;
function drawNoise() {
  if (++noiseFrame % 4) return;
  for (const c of $$('.feed:not(.ok) .noise')) {
    const w = 96, h = 54;
    if (c.width !== w) { c.width = w; c.height = h; }
    const g = c.getContext('2d');
    const img = g.createImageData(w, h);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = Math.random() * 60;
      img.data[i] = v * 0.7; img.data[i + 1] = v * 0.9; img.data[i + 2] = v * 1.1; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }
}

// ------------------------------------------------------------------ text scramble
function scramble(el, text) {
  el.dataset.v = text;
  const glyphs = '▚▞▙▟░▒#%&@ABCDEFGHJKLMNPQRSTUVWXYZ0123456789';
  const from = el.textContent;
  const len = Math.max(from.length, text.length);
  let f = 0;
  const id = (el._sid = Math.random());
  const tick = () => {
    if (el._sid !== id) return;
    f++;
    let out = '';
    for (let i = 0; i < len; i++) {
      if (f > i * 1.2 + 6) out += text[i] || '';
      else out += f > i * 1.2 ? glyphs[Math.floor(Math.random() * glyphs.length)] : (from[i] || '');
    }
    el.textContent = out;
    if (f < len * 1.2 + 7) requestAnimationFrame(tick);
    else el.textContent = text;
  };
  tick();
}

// ------------------------------------------------------------------ toasts, lightbox
function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $('#toasts').append(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 320); }, 3800);
}

for (const f of $$('.feed')) {
  f.addEventListener('click', () => {
    const url = backend.camUrl(f.dataset.cam);
    if (!url || !f.classList.contains('ok')) return toast('No camera signal yet', 'warn');
    const lb = $('#lightbox');
    lb.querySelector('img').src = url;
    lb.querySelector('.lb-title').textContent = f.dataset.cam === 'top' ? 'Bench' : 'Wrist';
    lb.classList.add('open');
  });
}
$('#lightbox').addEventListener('click', () => {
  const lb = $('#lightbox');
  lb.classList.remove('open');
  setTimeout(() => lb.querySelector('img').removeAttribute('src'), 300);
});

// ------------------------------------------------------------------ settings
const FIELDS = ['llmProvider', 'openaiKey', 'openaiModel', 'sttModel', 'elevenKey', 'elevenVoice', 'elevenModel', 'link', 'backend'];
function openSettings() {
  for (const f of FIELDS) $(`#s-${f}`).value = settings[f] ?? '';
  $('#s-wake').checked = !!settings.wake;
  renderDiag();
  $('#sheet').classList.add('open');
  $('#sheet').setAttribute('aria-hidden', 'false');
}
function closeSettings() {
  $('#sheet').classList.remove('open');
  $('#sheet').setAttribute('aria-hidden', 'true');
}
for (const f of FIELDS) $(`#s-${f}`).addEventListener('input', (e) => { settings[f] = e.target.value.trim(); saveSettings(); renderDiag(); });
$('#s-wake').addEventListener('change', (e) => setWake(e.target.checked));
$('#gear').innerHTML = icon('gear', 26, 1.6);
$('#teleop').insertAdjacentHTML('afterbegin', icon('gamepad', 22, 1.8));
$('#sh-close').innerHTML = icon('close', 22, 1.8);
$('#gear').addEventListener('click', () => { voice.unlock(); openSettings(); });
$('#sh-close').addEventListener('click', closeSettings);
$('#sheet').addEventListener('click', (e) => { if (e.target.id === 'sheet') closeSettings(); });
$('#s-test').addEventListener('click', () => { voice.unlock(); evera('Good evening. All systems are online and the bench is ready.'); });
$('#s-view').addEventListener('click', () => arm.resetView());
function renderDiag() {
  const ok = (b) => (b ? '<b>yes</b>' : 'no');
  $('#s-diag').innerHTML = [
    `HTTPS (needed for the mic): ${ok(window.isSecureContext)}`,
    `Wake word supported: ${ok(voice.wakeSupported)}`,
    `${{ groq: 'Groq', xai: 'xAI' }[settings.llmProvider] || 'OpenAI'} key: ${ok(!!settings.openaiKey)} · ElevenLabs key: ${ok(!!settings.elevenKey)}`,
    `Robot: <b>${S.status ? (S.status.connected ? 'connected' : S.status.sim ? 'simulator' : 'offline') : '…'}</b>`,
  ].join('<br>');
}

// ------------------------------------------------------------------ stop buttons
$('#hstop').addEventListener('click', () => estop());
$('#bigstop').addEventListener('click', () => estop());
$('#release').addEventListener('click', () => release());

// ------------------------------------------------------------------ boot sequence
const bootLines = {};
function bootLine(key, text, ok = false) {
  let el = bootLines[key];
  if (!el) {
    el = bootLines[key] = document.createElement('div');
    $('#bootlines').append(el);
  }
  el.textContent = (ok ? '✓ ' : '') + text;
  el.classList.toggle('ok', ok);
}
function boot() {
  $('#bootword').innerHTML = [...'evera'].map((c, i) => `<span style="animation-delay:${0.35 + i * 0.08}s">${c}</span>`).join('');
  bootLine('robot', 'Linking robot…');
  bootLine('voice', 'Voice systems…');
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    $('#boot').classList.add('gone');
    setTimeout(() => $('#boot').remove(), 900);
    S.booted = true;
    setTimeout(() => {
      const hasKeys = settings.openaiKey || settings.elevenKey;
      evera(hasKeys ? 'Good evening. Bench is online. What do you need?' : 'Bench online. Add your API keys in settings to talk to me.', { speak: false });
    }, 700);
  };
  $('#boot').addEventListener('click', () => { voice.unlock(); finish(); });
  if (DBG.has('fast')) {
    for (const tl of Object.values(tools)) tl.born = 1;
    return finish();
  }
  const min = new Promise((r) => setTimeout(r, 2200));
  const robot = new Promise((r) => {
    const chk = setInterval(() => {
      if (S.status) {
        clearInterval(chk);
        bootLine('robot', S.status.connected ? 'Robot linked' : S.status.sim ? 'Simulator engaged' : 'Robot offline', true);
        r();
      }
    }, 150);
  });
  bootLine('voice', `${settings.elevenKey ? 'ElevenLabs voice' : 'Browser voice'} · ${settings.openaiKey ? ({ groq: 'Groq brain', xai: 'Grok brain' }[settings.llmProvider] || 'GPT brain') : 'offline brain'}`, true);
  Promise.all([min, Promise.race([Promise.all([armReady, robot]), new Promise((r) => setTimeout(r, 7000))])]).then(() => setTimeout(finish, 350));
}

// ------------------------------------------------------------------ debug hooks (for tuning)
// #fast skips the boot, #pose={"shoulder_lift":-90,...} holds a pose, #sel=iron selects, #run=iron/heat runs.
const DBG = new URLSearchParams(location.hash.slice(1));
armReady.then(() => {
  if (DBG.has('pose')) {
    const pose = JSON.parse(DBG.get('pose'));
    arm.opacity = 0.999;
    arm.mode = 'frozen';
    arm.target = { ...arm.cur, ...pose };
    Object.assign(arm.cur, pose);
  }
  if (DBG.has('sel')) setTimeout(() => { selectTool(DBG.get('sel')); if (DBG.has('snap')) arm.snap(); }, 300);
  if (DBG.has('say')) setTimeout(() => handleUtterance(DBG.get('say')), 500);
  if (DBG.has('then')) setTimeout(() => handleUtterance(DBG.get('then')), +(DBG.get('thenAt') || 10000));
  if (DBG.has('stopAt')) setTimeout(() => handleUtterance('stop'), +DBG.get('stopAt'));
  if (DBG.has('run')) {
    const [tid, mid] = DBG.get('run').split('/');
    const tool = toolById(tid);
    setTimeout(() => runMode(tool, tool.modes.find((m) => m.id === mid) || tool.modes[0]), 400);
  }
});

// ------------------------------------------------------------------ start
buildTools();
layoutTools();
new ResizeObserver(() => layoutTools()).observe(stage);
setVoiceState('idle');
setWake(settings.wake !== false);
if (!settings.openaiKey) $('#brainbadge').classList.add('on');
backend.start();
boot();
window.addEventListener('pointerdown', () => {
  voice.unlock();
  S.gesture = true;
  voice.setWake(S.wakeOn);
}, { once: true });
