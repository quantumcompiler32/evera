// Talks to the GMKtec dashboard (robot/dashboard.py). Falls back to a simulator so the whole
// UI can be demoed without the arm (?sim=1, or Settings → Robot link → Simulator).
//
//   GET  /api/status           {state, mode, running, message, cams:{top, wrist}, ...}
//   POST /api/policy           {ckpt, task, secs} → "started" | error string
//   POST /api/key {k:"q"}      stop (SIGINT); the arm returns to its start pose
//   GET  /stream/top|wrist     MJPEG

export class Backend {
  constructor(getSettings, onStatus) {
    this.getSettings = getSettings;
    this.onStatus = onStatus;
    this.fails = 0;
    this.everConnected = false;
    this.sim = new Sim();
    this.status = { connected: false, sim: false, state: 'idle', running: false, message: '', cams: {} };
  }

  get base() {
    return (this.getSettings().backend || '').replace(/\/$/, '');
  }

  get useSim() {
    const link = this.getSettings().link || 'auto';
    if (new URLSearchParams(location.search).has('sim') || link === 'sim') return true;
    if (link === 'real') return false;
    // auto: simulate only if the real arm has never answered. Once it has, a dropout shows
    // "Offline" instead, so commands never silently go to the simulator mid-demo.
    return !this.everConnected && this.fails >= 2;
  }

  start() {
    const tick = async () => {
      await this.poll();
      setTimeout(tick, 600);
    };
    tick();
  }

  async poll() {
    let real = null;
    if (!(new URLSearchParams(location.search).has('sim') || this.getSettings().link === 'sim')) {
      try {
        const ctl = new AbortController();
        const to = setTimeout(() => ctl.abort(), 5000);
        const r = await fetch(this.base + '/api/status', { cache: 'no-store', signal: ctl.signal });
        clearTimeout(to);
        if (!r.ok) throw new Error(r.status);
        real = await r.json();
        this.fails = 0;
        this.everConnected = true;
      } catch {
        this.fails++;
      }
    }
    const s = real
      ? { connected: true, sim: false, state: real.state, mode: real.mode, running: real.running, message: real.message, cams: real.cams || {} }
      : this.useSim
        ? { connected: false, sim: true, ...this.sim.status() }
        : { connected: false, sim: false, state: 'offline', running: false, message: 'Arm offline', cams: {} };
    this.status = s;
    this.onStatus(s);
  }

  async post(path, body) {
    if (this.useSim) return this.sim.post(path, body);
    const r = await fetch(this.base + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    return typeof j === 'string' ? j : j.result; // dashboard replies {"result": "started"}
  }

  run(mode) {
    return this.post('/api/policy', { ckpt: mode.ckpt || '', task: mode.task, secs: mode.secs });
  }

  stop() {
    return this.post('/api/key', { k: 'q' });
  }

  camUrl(name) {
    return this.useSim ? null : `${this.base}/stream/${name}`;
  }

  // Resolves once the arm is idle again (or after `ms`).
  async waitIdle(ms = 12000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (!this.status.running && ['idle', 'error', 'offline'].includes(this.status.state)) return true;
      await new Promise((r) => setTimeout(r, 300));
    }
    return false;
  }
}

class Sim {
  constructor() {
    this.state = 'idle';
    this.running = false;
    this.timers = [];
    this.message = 'Simulator';
  }

  status() {
    return { state: this.state, mode: 'policy', running: this.running, message: this.message, cams: { top: 0, wrist: 0 } };
  }

  clear() {
    this.timers.forEach(clearTimeout);
    this.timers = [];
  }

  at(ms, fn) {
    this.timers.push(setTimeout(fn, ms));
  }

  post(path, body) {
    if (path === '/api/policy') {
      if (this.running) return 'a session is already running';
      this.clear();
      this.running = true;
      this.state = 'starting';
      this.message = 'Checking motors, connecting arms + cameras...';
      this.at(2200, () => { this.state = 'robot'; this.message = 'Robot is running the policy by itself.'; });
      this.at(2200 + body.secs * 1000, () => { this.state = 'saving'; this.message = 'Policy run finished, returning arm to its start pose...'; });
      this.at(2200 + body.secs * 1000 + 3200, () => { this.state = 'idle'; this.running = false; this.message = 'Session ended.'; });
      return 'started';
    }
    if (path === '/api/key' && body.k === 'q') {
      if (!this.running) return 'no session running';
      this.clear();
      this.state = 'saving';
      this.message = 'Stopping the policy, arm returning to its start pose...';
      this.at(3000, () => { this.state = 'idle'; this.running = false; });
      return 'sent';
    }
    return 'unknown';
  }
}
