// Geometry helpers: cable route clean-up, lane separation, filleted tubes,
// port shapes. All coordinates here are SCENE coordinates (mm, z = front->rear)
// except where noted; conversion to three.js happens in toThree().
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export const toThree = (p) => new THREE.Vector3(p[0], p[1], -p[2]);
export const dirToThree = (n) => new THREE.Vector3(n[0], n[1], -n[2]);

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (v, s) => [v[0] * s, v[1] * s, v[2] * s];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const norm = (v) => {
  const l = len(v);
  return l > 1e-9 ? scale(v, 1 / l) : [0, 0, 0];
};

// remove duplicate points and straight-through middle points
export function cleanRoute(pts) {
  const out = [];
  for (const p of pts) {
    if (!out.length || len(sub(p, out[out.length - 1])) > 0.05) out.push(p.slice());
  }
  let changed = true;
  while (changed && out.length > 2) {
    changed = false;
    for (let i = 1; i < out.length - 1; i++) {
      const d1 = norm(sub(out[i], out[i - 1]));
      const d2 = norm(sub(out[i + 1], out[i]));
      if (dot(d1, d2) > 0.99999) {
        out.splice(i, 1);
        changed = true;
        break;
      }
    }
  }
  return out;
}

function axisOf(d) {
  const nz = [0, 1, 2].filter((k) => Math.abs(d[k]) > 0.01);
  return nz.length === 1 ? nz[0] : -1;
}

// Separate collinear, overlapping segments of different cables into side-by-side
// lanes so that every cable stays visible (display only; ends stay on the ports).
// routes: Map id -> pts (scene coords, cleaned); meta: Map id -> {r, key}
export function separateLanes(routes, meta) {
  const groups = new Map();
  for (const [id, pts] of routes) {
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      const ax = axisOf(sub(b, a));
      if (ax < 0) continue;
      const o = [0, 1, 2].filter((k) => k !== ax);
      const key = `${ax}:${Math.round(a[o[0]] * 2)}:${Math.round(a[o[1]] * 2)}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ id, i, ax, lo: Math.min(a[ax], b[ax]), hi: Math.max(a[ax], b[ax]) });
    }
  }
  const segOff = new Map(); // id -> Map(segIndex -> offset vec)
  for (const segs of groups.values()) {
    if (new Set(segs.map((s) => s.id)).size < 2) continue;
    segs.sort((p, q) => (meta.get(p.id).key < meta.get(q.id).key ? -1 : meta.get(p.id).key > meta.get(q.id).key ? 1 : p.lo - q.lo));
    const placed = [];
    let maxR = 0;
    for (const s of segs) maxR = Math.max(maxR, meta.get(s.id).r);
    const spacing = 2 * maxR + 1.2;
    for (const s of segs) {
      const used = new Set();
      for (const q of placed) {
        if (q.id === s.id) continue;
        if (Math.min(q.hi, s.hi) - Math.max(q.lo, s.lo) > 2) used.add(q.lane);
      }
      let lane = 0;
      while (used.has(lane)) lane++;
      s.lane = lane;
      placed.push(s);
      if (lane > 0) {
        // offset direction: x-runs and z-runs move up (+y), y-runs move right (+x)
        const v = [0, 0, 0];
        const perp = s.ax === 1 ? 0 : 1;
        // alternate sides to keep the bundle centred on the nominal route
        const k = Math.ceil(lane / 2) * (lane % 2 ? 1 : -1);
        v[perp] = k * spacing;
        if (!segOff.has(s.id)) segOff.set(s.id, new Map());
        segOff.get(s.id).set(s.i, v);
      }
    }
  }
  const out = new Map();
  for (const [id, pts] of routes) {
    const offs = segOff.get(id);
    if (!offs) {
      out.set(id, pts);
      continue;
    }
    const res = pts.map((p) => p.slice());
    for (let vi = 1; vi < pts.length - 1; vi++) {
      const o1 = offs.get(vi - 1) || [0, 0, 0];
      const o2 = offs.get(vi) || [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const v = Math.abs(o1[k]) >= Math.abs(o2[k]) ? o1[k] : o2[k];
        res[vi][k] += v;
      }
    }
    out.set(id, res);
  }
  return out;
}

// Fillet every corner of a polyline (quadratic Bezier, tangent length t).
export function filletRoute(pts, bendR) {
  if (pts.length < 3) return pts.map((p) => p.slice());
  const out = [pts[0].slice()];
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1];
    const v = pts[i];
    const b = pts[i + 1];
    const l1 = len(sub(v, a));
    const l2 = len(sub(b, v));
    const d1 = norm(sub(v, a));
    const d2 = norm(sub(b, v));
    const c = dot(d1, d2);
    if (c < -0.995 || c > 0.9999 || l1 < 1e-3 || l2 < 1e-3) {
      out.push(v.slice());
      continue;
    }
    const phi = Math.acos(Math.max(-1, Math.min(1, c)));
    let t = bendR * Math.tan(phi / 2);
    t = Math.min(t, 0.45 * l1, 0.45 * l2);
    const ps = sub(v, scale(d1, t));
    const pe = add(v, scale(d2, t));
    const n = Math.max(4, Math.round((phi / (Math.PI / 2)) * 7));
    for (let k = 0; k <= n; k++) {
      const s = k / n;
      const w0 = (1 - s) * (1 - s);
      const w1 = 2 * (1 - s) * s;
      const w2 = s * s;
      out.push([
        w0 * ps[0] + w1 * v[0] + w2 * pe[0],
        w0 * ps[1] + w1 * v[1] + w2 * pe[1],
        w0 * ps[2] + w1 * v[2] + w2 * pe[2],
      ]);
    }
  }
  out.push(pts[pts.length - 1].slice());
  // drop near duplicates produced by tiny fillets
  const res = [out[0]];
  for (let i = 1; i < out.length; i++) if (len(sub(out[i], res[res.length - 1])) > 0.02) res.push(out[i]);
  return res;
}

export function polylineLength(pts) {
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += len(sub(pts[i], pts[i - 1]));
  return s;
}

export function pointAtFraction(pts, f) {
  const total = polylineLength(pts);
  let target = total * f;
  for (let i = 1; i < pts.length; i++) {
    const l = len(sub(pts[i], pts[i - 1]));
    if (target <= l || i === pts.length - 1) {
      const s = l > 0 ? Math.min(1, target / l) : 0;
      return add(pts[i - 1], scale(sub(pts[i], pts[i - 1]), s));
    }
    target -= l;
  }
  return pts[pts.length - 1];
}

// Tube along a sampled path (THREE.Vector3[]), parallel-transport frames.
// uv.x = arc length / dashPeriod so a striped alphaMap draws dashes.
export function tubeGeometry(points, radius, radial = 8, dashPeriod = 24) {
  const n = points.length;
  const T = [];
  for (let i = 0; i < n; i++) {
    const a = points[Math.max(0, i - 1)];
    const b = points[Math.min(n - 1, i + 1)];
    let t = new THREE.Vector3().subVectors(b, a);
    if (t.lengthSq() < 1e-10 && i > 0) t = T[i - 1].clone();
    T.push(t.normalize());
  }
  const N = [];
  let n0 = new THREE.Vector3(0, 1, 0);
  if (Math.abs(T[0].dot(n0)) > 0.9) n0 = new THREE.Vector3(1, 0, 0);
  n0.sub(T[0].clone().multiplyScalar(n0.dot(T[0]))).normalize();
  N.push(n0);
  for (let i = 1; i < n; i++) {
    const prev = N[i - 1].clone();
    prev.sub(T[i].clone().multiplyScalar(prev.dot(T[i])));
    if (prev.lengthSq() < 1e-8) {
      prev.set(0, 1, 0);
      if (Math.abs(T[i].y) > 0.9) prev.set(1, 0, 0);
      prev.sub(T[i].clone().multiplyScalar(prev.dot(T[i])));
    }
    N.push(prev.normalize());
  }
  const pos = new Float32Array(n * (radial + 1) * 3);
  const nor = new Float32Array(n * (radial + 1) * 3);
  const uv = new Float32Array(n * (radial + 1) * 2);
  let arc = 0;
  for (let i = 0; i < n; i++) {
    if (i > 0) arc += points[i].distanceTo(points[i - 1]);
    const B = new THREE.Vector3().crossVectors(T[i], N[i]).normalize();
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      const cx = Math.cos(a);
      const sy = Math.sin(a);
      const nx = cx * N[i].x + sy * B.x;
      const ny = cx * N[i].y + sy * B.y;
      const nz = cx * N[i].z + sy * B.z;
      const k = i * (radial + 1) + j;
      pos[k * 3] = points[i].x + radius * nx;
      pos[k * 3 + 1] = points[i].y + radius * ny;
      pos[k * 3 + 2] = points[i].z + radius * nz;
      nor[k * 3] = nx;
      nor[k * 3 + 1] = ny;
      nor[k * 3 + 2] = nz;
      uv[k * 2] = arc / dashPeriod;
      uv[k * 2 + 1] = j / radial;
    }
  }
  const idx = [];
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * (radial + 1) + j;
      const b = (i + 1) * (radial + 1) + j;
      const c = (i + 1) * (radial + 1) + j + 1;
      const d = i * (radial + 1) + j + 1;
      idx.push(a, b, d, b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

// ---- port shapes (modelled along local +Z, base on the face at z = 0) -----

function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2;
  const y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h);
  s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r);
  s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

function extrudeRR(w, h, r, depth) {
  const g = new THREE.ExtrudeGeometry(roundedRect(w, h, r), { depth, bevelEnabled: false, curveSegments: 4 });
  return g;
}

function cylZ(r, l, seg, z0 = 0) {
  const g = new THREE.CylinderGeometry(r, r, l, seg);
  g.rotateX(Math.PI / 2);
  g.translate(0, 0, z0 + l / 2);
  return g;
}

function boxZ(w, h, d, z0 = 0) {
  const g = new THREE.BoxGeometry(w, h, d);
  g.translate(0, 0, z0 + d / 2);
  return g;
}

function merged(parts) {
  // parts: [geometry, ...] -> one geometry with one group per part
  const gs = parts.map((g) => (g.index ? g.toNonIndexed() : g));
  for (const g of gs) {
    for (const k of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(k)) g.deleteAttribute(k);
  }
  return mergeGeometries(gs, true);
}

// returns {geom, mats:[{color, metal, rough}], size:[w,h]} per port type
export function portShapes() {
  const gold = { color: '#d9b44a', metal: 0.75, rough: 0.3 };
  const pin = { color: '#f4efe1', metal: 0.2, rough: 0.5 };
  const dark = { color: '#1c1f24', metal: 0.1, rough: 0.7 };
  const silver = { color: '#b9c0c8', metal: 0.7, rough: 0.35 };
  const black = { color: '#121417', metal: 0.0, rough: 0.8 };
  return {
    SMA: { geom: merged([cylZ(3.6, 5.5, 6), cylZ(1.7, 7.2, 12)]), mats: [gold, pin], size: [8, 8] },
    RJ45: { geom: merged([boxZ(11.5, 9.5, 1.6), boxZ(7.5, 4.5, 0.4, 1.6)]), mats: [dark, silver], size: [12, 10] },
    SFP: { geom: merged([boxZ(14, 9, 2.4), boxZ(11, 5, 0.5, 2.4)]), mats: [silver, black], size: [14, 9] },
    C13: { geom: merged([extrudeRR(22, 15, 4, 2.5), boxZ(2, 5, 0.4, 2.5)]), mats: [black, silver], size: [22, 15] },
    C14: { geom: merged([extrudeRR(24, 16.5, 4, 2.5), boxZ(16, 3, 0.5, 2.5)]), mats: [dark, silver], size: [24, 16.5] },
    C5: { geom: merged([extrudeRR(14, 9, 3.5, 2.5), boxZ(2, 2, 0.4, 2.5)]), mats: [black, silver], size: [14, 9] },
    'C6?': { geom: merged([extrudeRR(14, 9, 3.5, 2.5), boxZ(2, 2, 0.4, 2.5)]), mats: [black, silver], size: [14, 9] },
    DC: { geom: merged([boxZ(9, 9, 3.5), cylZ(1.8, 1.0, 10, 3.5)]), mats: [{ color: '#e8862a', metal: 0.1, rough: 0.6 }, silver], size: [9, 9] },
    WG: { geom: merged([boxZ(28, 28, 4), boxZ(15, 7.5, 0.5, 4)]), mats: [{ color: '#c9a64a', metal: 0.7, rough: 0.35 }, black], size: [28, 28] },
    IEC60309: { geom: merged([cylZ(21, 26, 28), cylZ(13, 0.6, 24, 26)]), mats: [{ color: '#2563c9', metal: 0.05, rough: 0.6 }, black], size: [42, 42] },
    N: { geom: merged([cylZ(7, 9, 16), cylZ(2.5, 9.6, 12)]), mats: [silver, pin], size: [14, 14] },
    default: { geom: merged([boxZ(8, 8, 3), boxZ(3, 3, 0.4, 3)]), mats: [dark, silver], size: [8, 8] },
  };
}

// Orientation that turns local +Z into the port normal (three coords).
export function quatForNormal(n3) {
  const q = new THREE.Quaternion();
  const z = new THREE.Vector3(0, 0, 1);
  if (n3.lengthSq() < 1e-9) return q;
  q.setFromUnitVectors(z, n3.clone().normalize());
  return q;
}

export { sub, len, dot, add, scale, norm };
