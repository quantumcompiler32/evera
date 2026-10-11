// Talks to the GMKtec dashboard (robot/dashboard.py). Tool actions can run on a simulator
// (link: 'sim') while teleop, demo replays, STOP and the cameras use the real arm.
//
//   GET  /api/status           {state, mode, running, message, cams:{top, wrist}, ...}
//   POST /api/policy           {ckpt, task, secs} → "started" | error string
//   POST /api/replay           {episode?} replay a recorded demo episode → "started" | error string
//   POST /api/practice         teleop: the leader arm drives the follower (nothing recorded)
//   POST /api/key {k:"q"}      stop (SIGINT); the arm returns to its start pose
//   GET  /stream/top|wrist     MJPEG

export class Backend {
  constructor(getSettings, onStatus) {
    this.getSettings = getSettings;
    this.onStatus = onStatus;
    this.fails = 0;
    this.everConnected = false;
    this.sim = new Sim();
    this.real = null; // last status from the real dashboard (null = unreachable)
    this.realUntil = 0; // show the real dashboard's state until then (after we started something there)
    this.status = { connected: false, sim: false, state: 'idle', running: false, message: '', cams: {} };
  }

  get base() {
    const b = (this.getSettings().backend || '').replace(/\/$/, '');
    // An https page can't load http resources: then the server in front (serve.py / tunnel) proxies.
    if (location.protocol === 'https:' && b.startsWith('http:')) return '';
    return b;
  }

  get forceSim() {
    return new URLSearchParams(location.search).has('sim');
  }

  // Tool actions (/api/policy) go to the simulator when link is 'sim'. Teleop and STOP always
  // go to the real arm when it's reachable.
  get useSim() {
    const link = this.getSettings().link || 'auto';
    if (this.forceSim || link === 'sim') return true;
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
    if (!this.forceSim) {
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
    this.real = real;
    if (real?.running) this.realUntil = Math.max(this.realUntil, Date.now() + 8000);
    const fromReal = (x) => ({ connected: true, sim: false, robot: true, state: x.state, mode: x.mode, running: x.running, message: x.message, cams: x.cams || {} });
    let s;
    if (real && (!this.useSim || Date.now() < this.realUntil)) s = fromReal(real); // real arm busy (e.g. teleop)
    else if (this.useSim) s = { connected: false, sim: true, robot: !!real, ...this.sim.status(), cams: real?.cams || {} };
    else s = { connected: false, sim: false, robot: false, state: 'offline', running: false, message: 'Arm offline', cams: {} };
    this.status = s;
    this.onStatus(s);
  }

  async postReal(path, body) {
    const r = await fetch(this.base + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) return `dashboard error ${r.status}`;
    const j = await r.json();
    return typeof j === 'string' ? j : j.result; // dashboard replies {"result": "started"}
  }

  async post(path, body) {
    if (this.useSim) return this.sim.post(path, body);
    return this.postReal(path, body);
  }

  async run(mode) {
    if (!mode.replay) return this.post('/api/policy', { ckpt: mode.ckpt || '', task: mode.task, secs: mode.secs });
    // A replay is a known-good real-arm action: like teleop, it goes to the real arm whenever that
    // answers, even when tool actions are set to the simulator (link: 'sim').
    if (this.forceSim || !this.real) return this.post('/api/replay', { secs: mode.secs });
    const res = await this.postReal('/api/replay', { secs: mode.secs });
    if (res === 'started') this.realUntil = Date.now() + 8000;
    return res;
  }

  // Teleop = the dashboard's practice mode: the leader arm drives the real follower.
  async teleop() {
    if (this.forceSim) return this.sim.post('/api/practice', {});
    if (!this.real) return 'robot dashboard unreachable';
    const res = await this.postReal('/api/practice', {});
    if (res === 'started') this.realUntil = Date.now() + 8000;
    return res;
  }

  // STOP reaches everything that might be moving: the simulator and the real arm.
  async stop() {
    const out = [];
    if (this.sim.running) out.push(this.sim.post('/api/key', { k: 'q' }));
    if (this.real?.running || !this.useSim) out.push(this.postReal('/api/key', { k: 'q' }).catch((e) => e.message));
    const res = await Promise.all(out);
    return res.includes('sent') ? 'sent' : res[0] || 'no session running';
  }

  // Cameras come from the dashboard even while actions are simulated.
  camUrl(name) {
    return `${this.base}/stream/${name}`;
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
    if (path === '/api/policy' || path === '/api/replay') {
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
    if (path === '/api/practice') {
      if (this.running) return 'a session is already running';
      this.clear();
      this.running = true;
      this.state = 'starting';
      this.message = 'Connecting arms...';
      this.at(1800, () => { this.state = 'practice'; this.message = 'Practice: move the leader arm, the follower copies it.'; });
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
