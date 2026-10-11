// Procedural 3D models of the 12 bench tools (built from three.js primitives, no downloads).
// buildTool(id) returns a Group normalised to ~0.3 m, centred on the origin.
import * as THREE from 'three';

const std = (color, roughness = 0.5, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
const phys = (color, roughness = 0.42) => new THREE.MeshPhysicalMaterial({ color, roughness, clearcoat: 0.4, clearcoatRoughness: 0.4 });

function mats() {
  return {
    orange: phys(0xff7a1a), black: std(0x16181c, 0.42, 0.1), rubber: std(0x25282d, 0.85), metal: std(0xd6dae0, 0.24, 1),
    steel: std(0x8f959d, 0.3, 1), oxide: std(0x2b2e34, 0.32, 0.9), iron: std(0x2b2f36, 0.5, 0.6),
    red: phys(0xd62f2f, 0.5), blue: phys(0x2f7bff, 0.45), yellow: phys(0xffc21a, 0.4), white: phys(0xf1f3f5, 0.35),
    glass: Object.assign(new THREE.MeshPhysicalMaterial({ color: 0xcfefff, roughness: 0.05, transparent: true, opacity: 0.22 }), { userData: { base: 0.22 } }),
    stick: Object.assign(new THREE.MeshPhysicalMaterial({ color: 0xf4f1e8, roughness: 0.3, transparent: true, opacity: 0.8 }), { userData: { base: 0.8 } }),
  };
}

// ------------------------------------------------------------------ helpers
function add(g, geo, mat, p = [0, 0, 0], r = [0, 0, 0]) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(...p);
  m.rotation.set(...r);
  m.castShadow = m.receiveShadow = true;
  g.add(m);
  return m;
}
function rrect(w, h, r) {
  const s = new THREE.Shape(), x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  return s;
}
function poly(pts) {
  const s = new THREE.Shape();
  pts.forEach(([x, y], i) => (i ? s.lineTo(x, y) : s.moveTo(x, y)));
  s.closePath();
  return s;
}
function slab(shape, depth, bevel = 0.25) {
  const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: bevel > 0, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 3, curveSegments: 18 });
  g.translate(0, 0, -depth / 2);
  return g;
}
const lathe = (pts, seg = 40) => new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), seg);
const tube = (pts, r, seg = 48) => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map((p) => new THREE.Vector3(...p))), seg, r, 14, false);
const cyl = (r, h, seg = 32) => new THREE.CylinderGeometry(r, r, h, seg);
const Z = Math.PI / 2;

function screen(text, { fg = '#7fd3f7', bg = '#07131b', w = 256, h = 128, font = 72 } = {}) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.fillStyle = bg; g.fillRect(0, 0, w, h);
  g.fillStyle = fg; g.shadowColor = fg; g.shadowBlur = 12;
  g.font = `600 ${font}px "SF Mono", Menlo, monospace`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, w / 2, h / 2 + 4);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshBasicMaterial({ map: t, toneMapped: false });
}

// ------------------------------------------------------------------ tools (units ≈ cm)
function iron(M) {
  const g = new THREE.Group();
  add(g, lathe([[0, 0], [0.82, 0], [0.9, 0.6], [0.9, 3.5], [0.8, 9], [0.62, 11.5], [0.42, 12.4], [0, 12.4]]), M.black);
  add(g, cyl(0.94, 3.6), M.blue, [0, 5.6, 0]);
  for (let i = 0; i < 6; i++) add(g, new THREE.TorusGeometry(0.94, 0.07, 8, 40), M.rubber, [0, 4 + i * 0.7, 0], [Z, 0, 0]);
  add(g, new THREE.PlaneGeometry(0.9, 1.7), screen('320°', { fg: '#ffb24d', w: 128, h: 256, font: 46 }), [0, 1.7, 0.91]);
  add(g, cyl(0.34, 4.4), M.metal, [0, 14.6, 0]);
  add(g, new THREE.CylinderGeometry(0.05, 0.34, 2.6, 32), M.steel, [0, 18.1, 0]);
  add(g, tube([[0, 0.2, 0], [0, -2, 0], [0.7, -4, 0], [2.4, -5.2, 0], [4.8, -5.5, 0]], 0.28), M.black);
  g.rotation.z = -1.05;
  return g;
}

function driver(M) {
  const g = new THREE.Group();
  add(g, lathe([[0, 0], [1.1, 0], [1.35, 0.6], [1.42, 3], [1.28, 6.5], [1.0, 8], [0.7, 8.6], [0.42, 9], [0, 9]]), M.orange);
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    add(g, new THREE.BoxGeometry(0.4, 5.2, 0.4), M.black, [Math.cos(a) * 1.33, 3.6, Math.sin(a) * 1.33], [0, -a, 0]);
  }
  add(g, cyl(0.3, 11), M.metal, [0, 14.4, 0]);
  add(g, new THREE.BoxGeometry(0.66, 1.4, 0.14), M.steel, [0, 20.4, 0]);
  g.rotation.z = -1.0;
  return g;
}

function dmm(M, { probes = true } = {}) {
  const g = new THREE.Group();
  add(g, slab(rrect(8.4, 15.4, 1.6), 3, 0.5), M.yellow);
  add(g, slab(rrect(7, 13.6, 1), 0.3, 0.1), M.black, [0, 0, 1.8]);
  add(g, new THREE.PlaneGeometry(5.6, 2.8), screen('12.34', { font: 84 }), [0, 4.4, 2.08]);
  const dial = new THREE.Group();
  dial.position.set(0, -1.4, 2.0);
  add(dial, cyl(2.3, 0.3), M.rubber, [0, 0, 0], [Z, 0, 0]);
  add(dial, cyl(1.9, 0.7), M.black, [0, 0, 0.3], [Z, 0, 0]);
  add(dial, new THREE.BoxGeometry(0.4, 1.9, 0.25), M.white, [0, 0.55, 0.7]);
  dial.rotation.z = -0.6;
  g.add(dial);
  for (let i = 0; i < 4; i++) add(g, new THREE.BoxGeometry(1.1, 0.55, 0.3), M.rubber, [-2.4 + i * 1.6, 1.85, 2.05]);
  [[-2.2, M.red], [0, M.black], [2.2, M.black]].forEach(([x, m]) => {
    add(g, cyl(0.45, 0.4), m, [x, -5.5, 2.05], [Z, 0, 0]);
    add(g, cyl(0.2, 0.42), M.iron, [x, -5.5, 2.1], [Z, 0, 0]);
  });
  if (probes) {
    add(g, tube([[-2.2, -5.5, 2.3], [-2.6, -8, 3], [-5, -9.6, 2], [-7.4, -9, 0]], 0.2), M.red);
    add(g, tube([[0, -5.5, 2.3], [0.6, -8.4, 3.2], [3.4, -10.2, 2], [6.6, -9.6, 0]], 0.2), M.black);
  }
  g.rotation.x = -0.25;
  return g;
}

function tape(M) {
  const g = new THREE.Group();
  add(g, slab(rrect(7.6, 7.6, 2.3), 3.4, 0.4), M.yellow);
  for (const z of [1.95, -1.95]) add(g, slab(rrect(6.2, 6.2, 1.9), 0.4, 0.12), M.black, [0, 0, z]);
  add(g, cyl(0.6, 0.3), M.metal, [0, 0, 2.3], [Z, 0, 0]);
  const c = document.createElement('canvas');
  c.width = 512; c.height = 64;
  const x = c.getContext('2d');
  x.fillStyle = '#ffd23a'; x.fillRect(0, 0, 512, 64); x.fillStyle = '#111';
  for (let i = 0; i < 64; i++) x.fillRect(i * 8, 0, 2, i % 5 ? 14 : 30);
  const blade = new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(c), roughness: 0.4, metalness: 0.2 });
  blade.map.colorSpace = THREE.SRGBColorSpace;
  add(g, new THREE.BoxGeometry(9, 0.18, 2.4), blade, [8.1, -3.55, 0]);
  add(g, new THREE.BoxGeometry(0.25, 1.3, 2.4), M.steel, [12.6, -3.0, 0]);
  add(g, new THREE.BoxGeometry(1.8, 0.8, 1.4), M.black, [1.2, 3.95, 0]);
  g.rotation.y = -0.55;
  return g;
}

function allen(M) {
  const g = new THREE.Group();
  const r = 0.55;
  add(g, cyl(r, 11, 6), M.metal, [0, 6.5, 0]);
  add(g, new THREE.TorusGeometry(1, r, 6, 16, Z), M.metal, [1, 1, 0], [0, 0, Math.PI]);
  add(g, cyl(r, 4, 6), M.metal, [3, 0, 0], [0, 0, Z]);
  g.rotation.set(0.3, 0.5, 0.45);
  return g;
}

function glue(M) {
  const g = new THREE.Group();
  add(g, slab(rrect(12, 4.4, 2), 3.2, 0.35), M.blue, [0, 4.2, 0]);
  const handle = new THREE.Shape();
  handle.moveTo(-0.8, 2.6); handle.lineTo(-2.3, -5); handle.quadraticCurveTo(-3.2, -6, -4.5, -5.6); handle.lineTo(-4.6, 2.6); handle.closePath();
  add(g, slab(handle, 2.6, 0.35), M.blue);
  const trig = new THREE.Shape();
  trig.moveTo(0.4, 2); trig.lineTo(1.1, -0.4); trig.quadraticCurveTo(0.8, -1.4, -0.1, -1); trig.lineTo(-0.5, 2); trig.closePath();
  add(g, slab(trig, 1.1, 0.15), M.black, [0.4, 0, 0]);
  add(g, cyl(1.8, 1.2), M.black, [-6.5, 4.2, 0], [0, 0, Z]);
  add(g, cyl(0.58, 6.5), M.stick, [-10.2, 4.2, 0], [0, 0, Z]);
  add(g, cyl(1.15, 1), M.black, [6.4, 4.2, 0], [0, 0, Z]);
  add(g, new THREE.CylinderGeometry(0.16, 0.9, 3, 32), M.metal, [8.3, 4.2, 0], [0, 0, -Z]);
  g.rotation.y = 0.45;
  return g;
}

function plierLike(M, cutter) {
  const g = new THREE.Group();
  for (const s of [1, -1]) {
    add(g, tube([[0, 0, 0], [-2, 0.6 * s, 0], [-6, 1.6 * s, 0], [-11, 2.4 * s, 0]], 0.38), M.steel);
    add(g, tube([[-4, 1.25 * s, 0], [-6, 1.6 * s, 0], [-8.5, 2.05 * s, 0], [-11.4, 2.45 * s, 0]], 0.66), cutter ? M.blue : M.red);
    const jaw = cutter
      ? poly([[0, 0.05 * s], [0, 1.4 * s], [1.6, 1.2 * s], [2.8, 0.55 * s], [3.2, 0.05 * s]])
      : poly([[0, 0.05 * s], [0, 1.3 * s], [6.6, 0.28 * s], [6.6, 0.05 * s]]);
    add(g, slab(jaw, 0.8, 0.12), M.steel);
  }
  add(g, cyl(0.95, 1.2), M.steel, [0, 0, 0], [Z, 0, 0]);
  add(g, cyl(0.45, 1.35), M.metal, [0, 0, 0], [Z, 0, 0]);
  g.rotation.set(0.35, 0, 0.9);
  return g;
}

function tweezers(M) {
  const g = new THREE.Group();
  for (const s of [1, -1]) {
    const geo = new THREE.BoxGeometry(12, 0.14, 0.8, 12, 1, 1);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) p.setZ(i, p.getZ(i) * (1 - 0.7 * ((p.getX(i) + 6) / 12)));
    geo.computeVertexNormals();
    const m = add(g, geo, M.metal, [0, 0.45 * s, 0], [0, 0, -0.06 * s]);
    m.geometry.translate(6, 0, 0);
    m.position.x = -6;
  }
  add(g, new THREE.BoxGeometry(1.4, 1.1, 0.9), M.metal, [-6.4, 0, 0]);
  g.rotation.set(0.5, 0, 0.75);
  return g;
}

function calipers(M) {
  const g = new THREE.Group();
  add(g, new THREE.BoxGeometry(16, 1.6, 0.35), M.metal, [2, 0, 0]);
  add(g, slab(poly([[-6, 0.8], [-6, -4.6], [-5.3, -5.1], [-4.4, -1], [-4.4, 0.8]]), 0.4, 0.05), M.metal);
  add(g, slab(poly([[-6, 0.8], [-6, 2.9], [-5.4, 2.9], [-4.8, 0.8]]), 0.4, 0.05), M.metal);
  const sl = new THREE.Group();
  sl.position.x = 2.5;
  add(sl, slab(poly([[-4.2, 0.8], [-4.2, -1], [-3.4, -5.1], [-2.7, -4.6], [-2.4, 0.8]]), 0.4, 0.05), M.metal);
  add(sl, slab(poly([[-4.2, 0.8], [-4.7, 2.9], [-4.1, 2.9], [-3.6, 0.8]]), 0.4, 0.05), M.metal);
  add(sl, slab(rrect(5.2, 3.2, 0.5), 1, 0.15), M.black, [-1.2, 0.3, 0.5]);
  add(sl, new THREE.PlaneGeometry(3.6, 1.4), screen('25.40', { font: 80 }), [-1.2, 0.75, 1.17]);
  add(sl, cyl(0.5, 0.35), M.steel, [-1.6, -1.8, 0], [Z, 0, 0]);
  g.add(sl);
  g.rotation.set(0.1, -0.35, 0.12);
  return g;
}

function hand(M) {
  const g = new THREE.Group();
  add(g, cyl(4.5, 1, 56), M.iron, [0, 0.5, 0]);
  add(g, cyl(0.35, 9), M.metal, [0, 5.5, 0]);
  add(g, new THREE.SphereGeometry(0.7, 24, 16), M.steel, [0, 10, 0]);
  add(g, cyl(0.3, 10), M.metal, [0, 10, 0], [0, 0, Z]);
  for (const s of [1, -1]) {
    add(g, new THREE.SphereGeometry(0.55, 24, 16), M.steel, [5 * s, 10, 0]);
    add(g, tube([[5 * s, 10, 0], [5.6 * s, 8.6, 1.2], [6.4 * s, 7, 2.6]], 0.24), M.metal);
    const clip = new THREE.Group();
    clip.position.set(6.4 * s, 7, 2.6);
    add(clip, new THREE.BoxGeometry(0.75, 2.6, 0.35), M.red, [0, -1.2, 0.25], [0.25, 0, 0]);
    add(clip, new THREE.BoxGeometry(0.75, 2.6, 0.35), M.red, [0, -1.2, -0.25], [-0.25, 0, 0]);
    g.add(clip);
  }
  add(g, tube([[0, 10, 0], [0, 12, 1.6], [0, 13.2, 3.6]], 0.24), M.metal);
  add(g, new THREE.TorusGeometry(3, 0.3, 14, 56), M.black, [0, 15.2, 4.4], [-0.35, 0, 0]);
  add(g, cyl(2.9, 0.12, 56), M.glass, [0, 15.2, 4.4], [Z - 0.35, 0, 0]);
  return g;
}

function dock(M) {
  const g = new THREE.Group();
  add(g, new THREE.BoxGeometry(16, 0.6, 9), M.orange, [0, 0, 0]);
  for (const z of [4.2, -4.2]) add(g, new THREE.BoxGeometry(16, 3, 0.6), M.orange, [0, 1.6, z]);
  for (const x of [7.7, -7.7]) add(g, new THREE.BoxGeometry(0.6, 3, 9), M.orange, [x, 1.6, 0]);
  for (const x of [-2.6, 2.6]) add(g, new THREE.BoxGeometry(0.4, 2.4, 7.8), M.black, [x, 1.3, 0]);
  const d = driver(M);
  d.rotation.set(0, 0, -Z);
  d.scale.setScalar(0.3);
  d.position.set(-7, 1.2, -1.6);
  g.add(d);
  const p = plierLike(M, false);
  p.rotation.set(-Z, 0, 0.2);
  p.scale.setScalar(0.42);
  p.position.set(1.5, 1.4, 0);
  g.add(p);
  const m = dmm(M, { probes: false });
  m.rotation.set(-0.2, 0, 0);
  m.scale.setScalar(0.28);
  m.position.set(5.2, 2.6, 0);
  g.add(m);
  g.rotation.set(0.45, 0.5, 0);
  return g;
}

const BUILD = {
  iron, driver, dmm, tape, allen, glue, tweezers, calipers, hand, dock,
  pliers: (M) => plierLike(M, false),
  cutters: (M) => plierLike(M, true),
};

export function buildTool(id, size = 0.3) {
  const M = mats();
  const inner = (BUILD[id] || BUILD.dock)(M);
  const wrap = new THREE.Group();
  wrap.add(inner);
  wrap.updateMatrixWorld(true);
  const bb = new THREE.Box3().setFromObject(inner);
  const c = bb.getCenter(new THREE.Vector3());
  const sz = bb.getSize(new THREE.Vector3());
  inner.position.sub(c);
  wrap.scale.setScalar(size / Math.max(sz.x, sz.y, sz.z));
  const holder = new THREE.Group();
  holder.add(wrap);
  const list = new Set();
  holder.traverse((o) => { if (o.isMesh) list.add(o.material); });
  for (const m of list) m.userData.base ??= 1;
  holder.userData.mats = [...list];
  holder.userData.vis = 0;
  return holder;
}
