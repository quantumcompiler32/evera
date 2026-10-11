// The 3D SO-101 in the middle of the screen: real URDF + STL meshes, rendered with three.js.
// Motion is driven by the run state (never by fake joint numbers): idle breathing, aiming at
// the selected tool, and a keyframed timeline per action type while the policy runs.
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildTool } from './tools3d.js';

const R = THREE.MathUtils.degToRad;
const JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// Poses in degrees. Pan is filled in per target (tool direction / toward the user).
const P = {
  rest: { shoulder_lift: -95, elbow_flex: 90, wrist_flex: 55, wrist_roll: 0, gripper: 0 },
  stand: { shoulder_lift: 0, elbow_flex: -35, wrist_flex: 35, wrist_roll: 0, gripper: 8 },
  ready: { shoulder_lift: 5, elbow_flex: -10, wrist_flex: 50, wrist_roll: 0, gripper: 25 },
  over: { shoulder_lift: 15, elbow_flex: 12, wrist_flex: 62, wrist_roll: 0, gripper: 60 },
  down: { shoulder_lift: 25, elbow_flex: 25, wrist_flex: 50, wrist_roll: 0, gripper: 60 },
  grasp: { shoulder_lift: 25, elbow_flex: 25, wrist_flex: 50, wrist_roll: 0, gripper: 4 },
  lift: { shoulder_lift: 0, elbow_flex: 0, wrist_flex: 50, wrist_roll: 0, gripper: 4 },
  offer: { shoulder_lift: 15, elbow_flex: -25, wrist_flex: 35, wrist_roll: 0, gripper: 4 },
  pressUp: { shoulder_lift: 10, elbow_flex: 15, wrist_flex: 65, wrist_roll: 0, gripper: 4 },
  pressDown: { shoulder_lift: 20, elbow_flex: 22, wrist_flex: 58, wrist_roll: 0, gripper: 4 },
  hold: { shoulder_lift: 10, elbow_flex: -5, wrist_flex: 55, wrist_roll: 0, gripper: 6 },
};

export class Arm3D {
  constructor(canvas) {
    this.canvas = canvas;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.setClearColor(0x000000, 0);
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = (this.scene = new THREE.Scene());
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.camera = new THREE.PerspectiveCamera(26, 1, 0.01, 30);
    this.camera.position.set(-0.55, 0.42, 0.95);
    const c = (this.controls = new OrbitControls(this.camera, canvas));
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.enablePan = false;
    c.minPolarAngle = R(18);
    c.maxPolarAngle = R(88);
    c.rotateSpeed = 0.7;
    this.lastTouch = -1e9;
    c.addEventListener('start', () => (this.lastTouch = performance.now()));
    c.addEventListener('end', () => {
      this.lastTouch = performance.now();
      this.onViewEnd?.();
    });

    const key = new THREE.DirectionalLight(0xfff1e0, 2.4);
    key.position.set(0.5, 1.3, 0.9);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, { left: -0.45, right: 0.45, top: 0.45, bottom: -0.45, near: 0.1, far: 4 });
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.002;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x5ec8f2, 2.6);
    rim.position.set(-0.9, 0.7, -1.0);
    scene.add(rim);
    this.rim = rim;
    scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x0b0f17, 0.45));

    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2; // URDF is Z-up, three.js is Y-up
    this.holder = new THREE.Group();
    this.holder.add(this.root);
    scene.add(this.holder);
    this.buildFloor();

    this.joints = {};
    this.cur = {};
    this.target = {};
    for (const j of JOINTS) this.cur[j] = this.target[j] = 0;
    Object.assign(this.target, P.rest);
    Object.assign(this.cur, P.rest);
    this.mode = 'idle';
    this.aimPan = 0;
    this.timeline = null;
    this.hot = false;
    this.flash = 0;
    this.opacity = 0;
    this.time = 0;
    this.clock = new THREE.Clock();
    this.onFrame = null;
    this.display = null; // tool id shown instead of the arm
    this.armVis = 1;
    this.toolObjs = {};
    this.running = true;

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    document.addEventListener('visibilitychange', () => {
      this.running = !document.hidden;
      if (this.running) { this.clock.getDelta(); this.loop(); }
    });
    this.loop();
  }

  buildFloor() {
    const s = this.scene;
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.ShadowMaterial({ opacity: 0.42 }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.receiveShadow = true;
    s.add(shadow);

    const glowTex = radialTexture([[0, 'rgba(94,200,242,0.55)'], [0.35, 'rgba(94,200,242,0.16)'], [1, 'rgba(94,200,242,0)']]);
    const glow = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.9), new THREE.MeshBasicMaterial({ map: glowTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = 0.0005;
    s.add(glow);
    this.glow = glow;

    const ringMat = (o) => new THREE.LineBasicMaterial({ color: 0x5ec8f2, transparent: true, opacity: o, depthWrite: false, blending: THREE.AdditiveBlending });
    const dashed = (radius, n, fill, o) => {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const a0 = (i / n) * Math.PI * 2, a1 = ((i + fill) / n) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a0) * radius, 0, Math.sin(a0) * radius), new THREE.Vector3(Math.cos(a1) * radius, 0, Math.sin(a1) * radius));
      }
      const m = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), ringMat(o));
      m.position.y = 0.001;
      s.add(m);
      return m;
    };
    this.rings = [dashed(0.2, 72, 0.5, 0.55), dashed(0.27, 24, 0.7, 0.3), dashed(0.34, 120, 0.15, 0.35)];

    // Scan band used by the boot / "starting" effect.
    const band = new THREE.Mesh(
      new THREE.CylinderGeometry(0.11, 0.11, 0.006, 64, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x7fd8ff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending })
    );
    s.add(band);
    this.band = band;
    this.scanT = -1;

    // Heat glow at the gripper tip for hot tools.
    const hotTex = radialTexture([[0, 'rgba(255,190,90,1)'], [0.25, 'rgba(255,140,40,0.6)'], [1, 'rgba(255,120,20,0)']]);
    this.hotSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: hotTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.hotSprite.scale.setScalar(0.09);
    this.hotSprite.visible = false;
    this.hotLight = new THREE.PointLight(0xffa630, 0, 0.6, 1.5);
  }

  async load(url, onProgress = () => {}) {
    const text = await (await fetch(url)).text();
    const xml = new DOMParser().parseFromString(text, 'application/xml');
    const robot = xml.querySelector('robot');
    const kids = [...robot.children];
    const linkEls = kids.filter((e) => e.tagName === 'link');
    const jointEls = kids.filter((e) => e.tagName === 'joint');
    const base = url.slice(0, url.lastIndexOf('/') + 1);

    const files = [...new Set(linkEls.flatMap((l) => [...l.querySelectorAll('visual mesh')].map((m) => m.getAttribute('filename'))))];
    const loader = new STLLoader();
    const geos = {};
    let done = 0;
    await Promise.all(files.map(async (f) => {
      const buf = await (await fetch(base + f)).arrayBuffer();
      let g = loader.parse(buf);
      try { g = toCreasedNormals(g, R(35)); } catch { /* keep flat normals */ }
      geos[f] = g;
      onProgress(++done / files.length);
    }));

    this.mats = {
      '3d_printed': new THREE.MeshPhysicalMaterial({ color: 0xff7a1a, roughness: 0.42, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.45, envMapIntensity: 0.85, transparent: true, opacity: 0 }),
      sts3215: new THREE.MeshStandardMaterial({ color: 0x16181c, roughness: 0.36, metalness: 0.3, envMapIntensity: 0.9, transparent: true, opacity: 0 }),
    };

    const links = {};
    for (const l of linkEls) {
      const g = new THREE.Group();
      g.name = l.getAttribute('name');
      for (const v of l.querySelectorAll(':scope > visual')) {
        const mesh = v.querySelector('mesh');
        if (!mesh) continue;
        const matName = v.querySelector('material')?.getAttribute('name') || '3d_printed';
        const m = new THREE.Mesh(geos[mesh.getAttribute('filename')], this.mats[matName] || this.mats['3d_printed']);
        applyOrigin(m, v.querySelector('origin'));
        m.castShadow = true;
        m.receiveShadow = true;
        g.add(m);
      }
      links[g.name] = g;
    }
    const children = new Set();
    for (const j of jointEls) {
      const frame = new THREE.Group();
      applyOrigin(frame, j.querySelector('origin'));
      const pivot = new THREE.Group();
      frame.add(pivot);
      const parent = j.querySelector('parent').getAttribute('link');
      const child = j.querySelector('child').getAttribute('link');
      links[parent].add(frame);
      pivot.add(links[child]);
      children.add(child);
      if (j.getAttribute('type') === 'revolute') {
        const ax = (j.querySelector('axis')?.getAttribute('xyz') || '0 0 1').split(/\s+/).map(Number);
        const lim = j.querySelector('limit');
        this.joints[j.getAttribute('name')] = {
          pivot,
          axis: new THREE.Vector3(...ax).normalize(),
          lower: +lim.getAttribute('lower'),
          upper: +lim.getAttribute('upper'),
        };
      }
    }
    const rootLink = Object.keys(links).find((n) => !children.has(n));
    this.root.add(links[rootLink]);
    this.links = links;
    this.tip = links.gripper_frame_link || links.gripper_link;
    this.tip.add(this.hotSprite);
    this.tip.add(this.hotLight);

    // Sit the base on the floor, centred on the origin.
    this.applyJoints(this.cur);
    this.holder.updateMatrixWorld(true);
    const bb = new THREE.Box3();
    for (const m of links[rootLink].children) if (m.isMesh) bb.expandByObject(m, true);
    const ctr = bb.getCenter(new THREE.Vector3());
    this.holder.position.set(-ctr.x, -bb.min.y, -ctr.z);
    this.holder.updateMatrixWorld(true);
    this.basePos = new THREE.Vector3(0, 0, 0);
    this.calibratePan();
    this.frame();
    this.loaded = true;
    this.scan();
  }

  // Which world direction does shoulder_pan = 0 point to, and which way is positive?
  calibratePan() {
    const ang = (pan) => {
      this.applyJoints({ ...this.cur, ...P.over, shoulder_pan: pan });
      this.holder.updateMatrixWorld(true);
      const p = this.tip.getWorldPosition(new THREE.Vector3());
      return Math.atan2(p.z, p.x);
    };
    this.fwd0 = ang(0);
    this.panSign = Math.sign(wrap(ang(R(30)) - this.fwd0)) || 1;
    this.applyJoints(this.cur);
  }

  panForWorldAngle(a) {
    const deg = THREE.MathUtils.radToDeg(this.panSign * wrap(a - this.fwd0));
    return THREE.MathUtils.clamp(deg, -105, 105);
  }

  frame() {
    const c = this.controls;
    c.target.set(0, 0.15, 0);
    this.camera.position.set(-0.78, 0.6, 1.38);
    c.minDistance = 0.8;
    c.maxDistance = 2.6;
    c.update();
    this.home0 = this.camera.position.clone();
  }

  resetView() {
    this.camera.position.copy(this.home0);
    this.controls.target.set(0, 0.15, 0);
  }

  resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.w = w;
    this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Keep the arm the same size on screen whatever the aspect ratio.
    this.camera.fov = w / h < 1.1 ? 32 : 26;
    this.camera.updateProjectionMatrix();
  }

  applyJoints(vals) {
    for (const name of JOINTS) {
      const j = this.joints[name];
      if (!j) continue;
      const rad = THREE.MathUtils.clamp(R(vals[name] ?? 0), j.lower, j.upper);
      j.pivot.quaternion.setFromAxisAngle(j.axis, rad);
    }
  }

  // ---------------------------------------------------------------- behaviours
  // Idle = folded in at rest (where the real arm sits between runs).
  idle() {
    this.mode = 'idle';
    this.startTimeline([{ pose: { ...P.rest, shoulder_pan: 0 }, dur: 1.6 }]);
  }

  // Arm stretched out toward elevation e (deg above horizontal), rotated by pan.
  aimPose(pan, e) {
    return { shoulder_pan: pan, shoulder_lift: 78 - e, elbow_flex: -72, wrist_flex: -6, wrist_roll: 0, gripper: 30 };
  }

  // Find the pan/elevation whose gripper lines up with a point on screen (stage px), as seen
  // from the current camera, so the arm visibly points at the tool orb. Pure: doesn't move the arm.
  solveAim(sx, sy) {
    if (!this.loaded) return { pan: 0, e: 30 };
    this.camera.updateMatrixWorld();
    const v = new THREE.Vector3();
    const scr = (obj) => {
      obj.getWorldPosition(v).project(this.camera);
      return { x: ((v.x + 1) / 2) * this.w, y: ((1 - v.y) / 2) * this.h };
    };
    const o = scr(this.links.shoulder_link);
    const want = Math.atan2(sy - o.y, sx - o.x);
    let best = null;
    for (let e = 0; e <= 70; e += 7) {
      for (let pan = -105; pan <= 105; pan += 3) {
        this.applyJoints(this.aimPose(pan, e));
        this.holder.updateMatrixWorld(true);
        const t = scr(this.tip);
        const err = Math.abs(wrap(Math.atan2(t.y - o.y, t.x - o.x) - want));
        const len = Math.hypot(t.x - o.x, t.y - o.y);
        const score = err - len * 0.0006;
        if (!best || score < best.score) best = { score, pan, e };
      }
    }
    this.applyJoints(this.cur);
    this.holder.updateMatrixWorld(true);
    return best;
  }

  // Swing straight from wherever the arm is to point at the orb. Returns the pan used.
  aim(sx, sy) {
    if (!this.loaded || this.mode === 'run' || this.mode === 'home') return null;
    const b = this.solveAim(sx, sy);
    this.mode = 'aim';
    this.aimPan = b.pan;
    this.startTimeline([{ pose: this.aimPose(b.pan, b.e), dur: 1.25 }]);
    return b.pan;
  }

  play(motion, { toolPan = 0, userPan = 0 } = {}) {
    this.mode = 'run';
    const at = (pose, pan, dur) => ({ pose: { ...P[pose], shoulder_pan: pan }, dur });
    let keys, loopFrom = null;
    if (motion === 'press') {
      keys = [at('over', toolPan, 1.2), at('down', toolPan, 1.0), at('grasp', toolPan, 0.6), at('lift', toolPan, 1.1), at('pressUp', 0, 1.5), at('pressDown', 0, 0.8), at('pressUp', 0, 0.8)];
      loopFrom = 5;
    } else if (motion === 'hold') {
      keys = [at('hold', 0, 1.6)];
    } else if (motion === 'dock') {
      keys = [];
      for (const p of [-70, -35, 0, 35, 70]) keys.push(at('over', p, 1.3), at('down', p, 0.9), at('over', p, 0.8));
      keys.push(at('rest', 0, 1.6));
    } else {
      keys = [at('over', toolPan, 1.2), at('down', toolPan, 1.0), at('grasp', toolPan, 0.6), at('lift', toolPan, 1.1), at('offer', userPan, 1.6)];
    }
    this.startTimeline(keys, loopFrom);
  }

  // Teleop: the human drives the real arm. No joint feed yet, so stand ready and breathe.
  teleop() {
    this.mode = 'teleop';
    this.startTimeline([{ pose: { ...P.stand, shoulder_pan: 0 }, dur: 1.6 }]);
  }

  // Let go: open the gripper where it is, then fold back to rest.
  release() {
    this.mode = 'home';
    this.startTimeline([{ pose: { ...this.cur, gripper: 60 }, dur: 0.6 }, { pose: { ...P.rest, shoulder_pan: 0 }, dur: 2.2 }], null, () => { this.mode = 'idle'; });
  }

  // Fold back in to rest (the real arm returns to its start pose after every run).
  home() {
    this.mode = 'home';
    this.startTimeline([{ pose: { ...P.rest, shoulder_pan: 0 }, dur: 2.2 }], null, () => { this.mode = 'idle'; });
  }

  // Show a tool's 3D model in place of the arm (null = show the arm).
  setDisplay(id) {
    if (id === this.display) return;
    this.display = id;
    if (id && !this.toolObjs[id]) {
      const o = buildTool(id, 0.26);
      o.position.set(0, 0.17, 0);
      o.visible = false;
      this.scene.add(o);
      this.toolObjs[id] = o;
    }
    if (id) this.toolObjs[id].rotation.y = -0.6;
    this.scan();
  }

  freeze() {
    this.mode = 'frozen';
    this.timeline = null;
    Object.assign(this.target, this.cur);
    this.flash = 1;
  }

  setHot(on) {
    this.hot = on;
    this.hotSprite.visible = on;
  }

  scan() {
    this.scanT = 0;
    this.scanStart = performance.now();
  }

  // Jump to the end of the current timeline (used by debug screenshots).
  snap() {
    if (!this.timeline) return;
    this.timeline.start = -1e9;
    Object.assign(this.cur, this.evalTimeline());
    this.applyJoints(this.cur);
  }

  startTimeline(keys, loopFrom, onDone) {
    this.timeline = { keys, loopFrom, start: performance.now(), from: { ...this.cur }, onDone };
  }

  // Timelines run on wall-clock time, so motion speed doesn't depend on frame rate.
  evalTimeline() {
    const tl = this.timeline;
    let t = (performance.now() - tl.start) / 1000, from = tl.from;
    const total = tl.keys.reduce((s, k) => s + k.dur, 0);
    if (t >= total && tl.loopFrom != null) {
      const loopKeys = tl.keys.slice(tl.loopFrom);
      const loopLen = loopKeys.reduce((s, k) => s + k.dur, 0);
      let lt = (t - total) % loopLen;
      let prev = tl.keys[tl.keys.length - 1].pose;
      for (const k of loopKeys) {
        if (lt <= k.dur) return lerpPose(prev, k.pose, easeInOut(lt / k.dur));
        lt -= k.dur;
        prev = k.pose;
      }
      return prev;
    }
    for (const k of tl.keys) {
      if (t <= k.dur) return lerpPose({ ...from }, k.pose, easeInOut(t / k.dur));
      t -= k.dur;
      from = { ...from, ...k.pose };
    }
    if (tl.onDone) {
      const cb = tl.onDone;
      tl.onDone = null;
      setTimeout(cb, 0);
    }
    return from;
  }

  userPan() {
    const c = this.camera.position;
    return this.panForWorldAngle(Math.atan2(c.z, c.x));
  }

  tipScreen() {
    if (!this.loaded) return null;
    const tool = this.display && this.armVis < 0.5 && this.toolObjs[this.display];
    const p = (tool ? tool.getWorldPosition(new THREE.Vector3()) : this.tip.getWorldPosition(new THREE.Vector3())).project(this.camera);
    return { x: ((p.x + 1) / 2) * this.w, y: ((1 - p.y) / 2) * this.h };
  }

  baseScreen() {
    const p = new THREE.Vector3(0, 0, 0).project(this.camera);
    return { x: ((p.x + 1) / 2) * this.w, y: ((1 - p.y) / 2) * this.h };
  }

  // ---------------------------------------------------------------- frame loop
  loop() {
    if (!this.running) return;
    requestAnimationFrame(() => this.loop());
    const dt = Math.min(this.clock.getDelta(), 0.05);
    this.time += dt;
    const t = this.time;

    if (this.loaded) {
      if (this.opacity < 1) {
        this.loadT ??= performance.now();
        this.opacity = Math.max(this.opacity, Math.min(1, (performance.now() - this.loadT) / 1100));
      }
      this.armVis += ((this.display ? 0 : 1) - this.armVis) * (1 - Math.exp(-dt * 7));
      if (Math.abs(this.armVis - Math.round(this.armVis)) < 0.002) this.armVis = Math.round(this.armVis);
      const op = this.opacity * this.armVis;
      if (op !== this.lastOp) {
        this.lastOp = op;
        for (const m of Object.values(this.mats)) {
          m.opacity = op;
          m.transparent = op < 1;
          m.depthWrite = op > 0.6;
        }
        this.holder.visible = op > 0.004;
        const sc = 0.82 + 0.18 * this.armVis;
        this.holder.scale.setScalar(sc);
      }
      let goal;
      if (this.timeline) goal = this.evalTimeline();
      else if (this.mode === 'frozen') goal = this.target;
      else goal = { ...P.rest, shoulder_pan: 0 };
      const g = { ...goal };
      if (this.mode === 'idle' || this.mode === 'aim' || this.mode === 'teleop') {
        const amp = this.mode === 'idle' ? 0.6 : 1;
        g.shoulder_lift += Math.sin(t * 1.1) * 1.4 * amp;
        g.elbow_flex -= Math.sin(t * 1.1 + 0.6) * 1.4 * amp;
        g.wrist_flex += Math.sin(t * 0.9 + 1.2) * 2 * amp;
        g.wrist_roll += Math.sin(t * 0.45) * 5 * amp;
      } else if (this.mode === 'run' && this.timeline && this.timeline.keys.length <= 2) {
        g.wrist_roll += Math.sin(t * 2.3) * 1.2;
      }
      const k = 1 - Math.exp(-dt * (this.timeline ? 14 : 3.2));
      for (const j of JOINTS) this.cur[j] += ((g[j] ?? this.cur[j]) - this.cur[j]) * k;
      this.applyJoints(this.cur);
    }

    // Tool models: rise in, spin slowly, fade out when deselected.
    for (const [id, o] of Object.entries(this.toolObjs)) {
      const want = id === this.display ? 1 : 0;
      let v = o.userData.vis;
      v += (want - v) * (1 - Math.exp(-dt * (want ? 5 : 9)));
      if (Math.abs(v - want) < 0.003) v = want;
      o.userData.vis = v;
      o.visible = v > 0.003;
      if (!o.visible) continue;
      const e = 1 - Math.pow(1 - v, 3);
      o.scale.setScalar(0.55 + 0.45 * e);
      o.position.y = 0.17 + Math.sin(t * 1.3) * 0.007 - (1 - e) * 0.06;
      o.rotation.y += dt * 0.55;
      for (const m of o.userData.mats) {
        m.opacity = m.userData.base * v;
        m.transparent = v < 1 || m.userData.base < 1;
        m.depthWrite = v > 0.5 && m.userData.base >= 1;
      }
    }

    // Floor rings, scan band, heat glow, E-stop flash.
    this.rings[0].rotation.y = t * 0.12;
    this.rings[1].rotation.y = -t * 0.2;
    this.rings[2].rotation.y = t * 0.05;
    this.glow.material.opacity = 0.8 + Math.sin(t * 1.6) * 0.15;
    if (this.scanT >= 0) {
      const s = (this.scanT = (performance.now() - this.scanStart) / 1600);
      this.band.position.y = 0.02 + s * 0.42;
      this.band.material.opacity = Math.sin(Math.min(s, 1) * Math.PI) * 0.55;
      if (s >= 1) this.scanT = -1, this.band.material.opacity = 0;
    }
    if (this.hot) {
      const pulse = 0.75 + Math.sin(t * 5) * 0.25;
      this.hotSprite.material.opacity = pulse;
      this.hotSprite.scale.setScalar(0.07 + pulse * 0.03);
      this.hotLight.intensity = 0.6 * pulse;
    } else this.hotLight.intensity = 0;
    if (this.mats) {
      const e = this.flash > 0 ? (this.flash -= dt * 1.2, Math.max(0, this.flash) * (0.5 + 0.5 * Math.sin(t * 30))) : 0;
      for (const m of Object.values(this.mats)) m.emissive.setRGB(e * 0.9, 0, 0);
    }
    this.rim.color.lerp(new THREE.Color(this.hot ? 0xffa630 : 0x5ec8f2), 0.05);

    const c = this.controls;
    c.autoRotate = performance.now() - this.lastTouch > 7000 && this.mode === 'idle';
    c.autoRotateSpeed = Math.sin(t * 0.22) * 0.55;
    c.update();
    this.renderer.render(this.scene, this.camera);
    if (this.onFrame) this.onFrame();
  }
}

function applyOrigin(obj, el) {
  if (!el) return;
  const xyz = (el.getAttribute('xyz') || '0 0 0').trim().split(/\s+/).map(Number);
  const rpy = (el.getAttribute('rpy') || '0 0 0').trim().split(/\s+/).map(Number);
  obj.position.set(...xyz);
  obj.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX'); // URDF fixed-axis roll/pitch/yaw
}

function lerpPose(a, b, t) {
  const o = { ...a };
  for (const k of Object.keys(b)) o[k] = (a[k] ?? b[k]) + (b[k] - (a[k] ?? b[k])) * t;
  return o;
}

function radialTexture(stops) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 256;
  const g = cv.getContext('2d');
  const grd = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  for (const [o, c] of stops) grd.addColorStop(o, c);
  g.fillStyle = grd;
  g.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
