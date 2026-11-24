// Canvas textures for device faces: ids, titles, ears, module plates, port
// captions. Units in mm; S = pixels per mm.
import * as THREE from 'three';

export const FONT = 'Inter, "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif';
export const MONO = '"JetBrains Mono", "DejaVu Sans Mono", Consolas, Menlo, monospace';

let maxAniso = 4;
export function setMaxAnisotropy(v) {
  maxAniso = v;
}

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const v = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
export function rgbToHex([r, g, b]) {
  return '#' + [r, g, b].map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0')).join('');
}
export function luminance(hex) {
  const [r, g, b] = hexToRgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
export function mix(hex, target, t) {
  const a = hexToRgb(hex);
  const b = hexToRgb(target);
  return rgbToHex(a.map((v, i) => v + (b[i] - v) * t));
}
export const lighten = (hex, t) => mix(hex, '#ffffff', t);
export const darken = (hex, t) => mix(hex, '#000000', t);
export const textColorFor = (bg) => (luminance(bg) > 0.33 ? '#15181d' : '#f4f6f8');

// Largest canvas side; drawing code must use fitScale() so a long face (e.g. a
// 42U rack post) is drawn at a scale that fits instead of being clipped and then
// stretched onto the face (which shifted the post U numbers by up to 2 U).
const MAX_CANVAS = 4096;
const fitScale = (wmm, hmm, S) => Math.min(S, MAX_CANVAS / Math.max(1, wmm), MAX_CANVAS / Math.max(1, hmm));

function makeCanvas(wmm, hmm, S) {
  const c = document.createElement('canvas');
  c.width = Math.max(4, Math.min(MAX_CANVAS, Math.round(wmm * S)));
  c.height = Math.max(4, Math.min(MAX_CANVAS, Math.round(hmm * S)));
  return c;
}

export function canvasTexture(canvas) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = maxAniso;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

function rr(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function wrap(ctx, text, maxW) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = '';
  for (const w of words) {
    const t = cur ? cur + ' ' + w : w;
    if (ctx.measureText(t).width <= maxW || !cur) cur = t;
    else {
      lines.push(cur);
      cur = w;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

// Fit a block of lines [{text, weight, size(mm), family, color}] into a box.
// Shrinks sizes uniformly; wraps 'wrap' lines; drops optional lines that do not fit.
function drawBlock(ctx, S, box, lines, align = 'left') {
  const [bx, by, bw, bh] = box; // mm, canvas coordinates (y down)
  let k = 1;
  for (let iter = 0; iter < 14; iter++) {
    const laid = [];
    let total = 0;
    let ok = true;
    for (const ln of lines) {
      const px = ln.size * k * S;
      ctx.font = `${ln.weight || 400} ${px}px ${ln.family || FONT}`;
      let parts = ln.wrap ? wrap(ctx, ln.text, bw * S) : [ln.text];
      if (ln.wrap && parts.length > (ln.maxLines || 2)) {
        parts = parts.slice(0, ln.maxLines || 2);
        parts[parts.length - 1] += '…';
      }
      for (const p of parts) {
        const w = ctx.measureText(p).width / S;
        if (w > bw + 0.01) ok = false;
        laid.push({ ...ln, text: p, px, h: ln.size * k * 1.22 });
        total += ln.size * k * 1.22;
      }
    }
    if (total > bh) ok = false;
    if (ok || iter === 13) {
      if (!ok) {
        // drop optional lines and retry once at this size
        const req = lines.filter((l) => !l.optional);
        if (req.length < lines.length) return drawBlock(ctx, S, box, req, align);
      }
      let y = by + (bh - total) / 2;
      for (const ln of laid) {
        ctx.font = `${ln.weight || 400} ${ln.px}px ${ln.family || FONT}`;
        ctx.fillStyle = ln.color;
        ctx.textBaseline = 'middle';
        ctx.textAlign = align;
        const x = align === 'center' ? bx + bw / 2 : bx;
        ctx.fillText(ln.text, x * S, (y + ln.h / 2) * S);
        y += ln.h;
      }
      return true;
    }
    k *= 0.88;
  }
  return false;
}

function freeIntervals(lo, hi, blocks) {
  // blocks: [[a,b]] -> free [[a,b]] inside [lo,hi]
  const bs = blocks.map(([a, b]) => [Math.max(lo, a), Math.min(hi, b)]).filter(([a, b]) => b > a).sort((p, q) => p[0] - q[0]);
  const out = [];
  let cur = lo;
  for (const [a, b] of bs) {
    if (a > cur) out.push([cur, a]);
    cur = Math.max(cur, b);
  }
  if (hi > cur) out.push([cur, hi]);
  return out;
}

function portCaption(dev, p) {
  const n = p.name;
  switch (dev.kind) {
    case 'spider': {
      const m = /\.(T\d+)$/.exec(n);
      if (m) return m[1];
      return n.replace(/^[A-Z0-9]+\./, '');
    }
    case 'ztdat':
    case 'mwatt': {
      const m = /^ch\d+\.(in|out)$/i.exec(n);
      if (m) return m[1].toUpperCase();
      return n.toUpperCase();
    }
    case 'sw_data':
    case 'sw_oob':
      return n.replace(/^P/, '');
    case 'pdu':
    case 'strip':
      if (/^O\d+$/.test(n)) return n.slice(1);
      if (n === 'inlet') return 'INLET';
      return n.toUpperCase();
    case 'gnss':
      return n.replace('out', '');
    default:
      return (p.label && p.label.length <= 8 ? p.label : n).toUpperCase();
  }
}

// ports: [{p, u, v, w, h}] with u,v = port centre in face mm coords (u from left, v from bottom)
export function drawFace({ dev, face, W, H, S, bg, ports, ears, uN, shortId }) {
  S = fitScale(W, H, S);
  const canvas = makeCanvas(W, H, S);
  const ctx = canvas.getContext('2d');
  const fg = textColorFor(bg);
  const sub = mix(bg, fg, 0.72);
  const faint = mix(bg, fg, 0.35);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // subtle panel gradient
  const g = ctx.createLinearGradient(0, 0, 0, canvas.height);
  g.addColorStop(0, 'rgba(255,255,255,0.07)');
  g.addColorStop(1, 'rgba(0,0,0,0.07)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const cy = (v) => (H - v) * S; // face v (from bottom) -> canvas px
  const cx = (u) => u * S;

  const earW = ears ? 16 : 0;
  if (ears) {
    for (const x0 of [0, W - earW]) {
      ctx.fillStyle = mix(bg, '#000000', 0.12);
      ctx.fillRect(cx(x0), 0, earW * S, canvas.height);
      ctx.fillStyle = mix(bg, '#ffffff', 0.08);
      ctx.fillRect(cx(x0 === 0 ? earW - 0.6 : x0), 0, 0.6 * S, canvas.height);
      const n = Math.max(1, uN || Math.round(H / 44.45));
      for (let k = 0; k < n; k++) {
        for (const hv of [6.35, 38.1]) {
          const v = k * 44.45 + hv;
          if (v > H - 2) continue;
          ctx.fillStyle = '#0d0f12';
          rr(ctx, cx(x0 + earW / 2 - 3.2), cy(v) - 1.9 * S, 6.4 * S, 3.8 * S, 1.8 * S);
          ctx.fill();
          ctx.fillStyle = '#9aa3ad';
          ctx.beginPath();
          ctx.arc(cx(x0 + earW / 2), cy(v), 1.5 * S, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  const blocks = [];
  // spider module plates (front)
  if (dev.kind === 'spider' && face === 'front' && dev.modules) {
    for (const m of dev.modules) {
      const x0 = m.x0;
      const x1 = m.x1;
      ctx.fillStyle = mix(bg, '#ffffff', 0.13);
      rr(ctx, cx(x0), cy(H - 3), (x1 - x0) * S, (H - 6) * S, 1.5 * S);
      ctx.fill();
      ctx.strokeStyle = mix(bg, '#ffffff', 0.4);
      ctx.lineWidth = Math.max(1, 0.4 * S);
      ctx.stroke();
      ctx.fillStyle = fg;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.font = `700 ${5.2 * S}px ${FONT}`;
      ctx.fillText(m.m, cx(x0 + 2.5), cy(H - 8.2));
      const mw = ctx.measureText(m.m).width / S;
      ctx.font = `600 ${5.2 * S}px ${FONT}`;
      ctx.fillStyle = luminance(bg) > 0.3 ? '#8a4500' : '#ffd479';
      let sec = m.sector || '';
      while (sec && ctx.measureText(sec).width / S > x1 - x0 - mw - 6) sec = sec.slice(0, -1);
      ctx.fillText(sec, cx(x0 + 2.5 + mw + 2.2), cy(H - 8.2));
      blocks.push([x0 - 2, x1 + 2]);
    }
  }

  // port fields (switch cages, PDU outlet banks, breaker rows)
  const fieldTypes = new Set(['RJ45', 'SFP', 'C13', 'DC']);
  const fieldPorts = ports.filter((q) => fieldTypes.has(q.p.type));
  if (fieldPorts.length >= 6) {
    const xs0 = Math.min(...fieldPorts.map((q) => q.u - q.w / 2)) - 4;
    const xs1 = Math.max(...fieldPorts.map((q) => q.u + q.w / 2)) + 4;
    const ys0 = Math.max(1.5, Math.min(...fieldPorts.map((q) => q.v - q.h / 2)) - 3.5);
    const ys1 = Math.min(H - 1.5, Math.max(...fieldPorts.map((q) => q.v + q.h / 2)) + 3.5);
    ctx.fillStyle = mix(bg, '#000000', 0.35);
    rr(ctx, cx(xs0), cy(ys1), (xs1 - xs0) * S, (ys1 - ys0) * S, 1.5 * S);
    ctx.fill();
  }

  // port captions
  const placed = [];
  const portRects = ports.map((q) => [q.u - q.w / 2 - 0.5, q.v - q.h / 2 - 0.5, q.u + q.w / 2 + 0.5, q.v + q.h / 2 + 0.5]);
  const hit = (r, list) => list.some((o) => r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1]);
  const capSize = H < 50 ? 3.6 : 4.6;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (const q of ports) {
    const t = portCaption(dev, q.p);
    if (!t) continue;
    ctx.font = `600 ${capSize * S}px ${FONT}`;
    const tw = ctx.measureText(t).width / S;
    const tries = [q.v - q.h / 2 - 0.8 - capSize / 2, q.v + q.h / 2 + 0.8 + capSize / 2];
    if (dev.kind === 'ztdat' || dev.kind === 'mwatt') tries.reverse(); // IN/OUT above the jack
    for (const vv of tries) {
      const r = [q.u - tw / 2, vv - capSize / 2, q.u + tw / 2, vv + capSize / 2];
      if (r[1] < 0.6 || r[3] > H - 0.6 || r[0] < earW || r[2] > W - earW) continue;
      if (hit(r, placed) || hit(r, portRects)) continue;
      placed.push(r);
      ctx.fillStyle = sub;
      ctx.fillText(t, cx(q.u), cy(vv));
      break;
    }
    blocks.push([q.u - q.w / 2 - 2, q.u + q.w / 2 + 2]);
  }
  // channel captions for attenuators / MWATT: "CH1" under each in/out pair
  if (dev.kind === 'ztdat' || dev.kind === 'mwatt') {
    const ch = new Map();
    for (const q of ports) {
      const m = /^ch(\d+)\./i.exec(q.p.name);
      if (m) {
        if (!ch.has(m[1])) ch.set(m[1], []);
        ch.get(m[1]).push(q);
      }
    }
    for (const [n, qs] of ch) {
      const u = qs.reduce((s, q) => s + q.u, 0) / qs.length;
      const v = Math.min(...qs.map((q) => q.v - q.h / 2)) - 1 - capSize * 0.62;
      if (v - capSize / 2 < 0.4) continue;
      ctx.font = `700 ${capSize * 1.1 * S}px ${FONT}`;
      ctx.fillStyle = fg;
      ctx.fillText(`CH${n}`, cx(u), cy(v));
      const u0 = Math.min(...qs.map((q) => q.u)) - 4;
      const u1 = Math.max(...qs.map((q) => q.u)) + 4;
      ctx.strokeStyle = faint;
      ctx.lineWidth = Math.max(1, 0.35 * S);
      rr(ctx, cx(u0 - 3), cy(Math.max(...qs.map((q) => q.v + q.h / 2)) + capSize + 2), (u1 - u0 + 6) * S, (capSize * 2 + qs[0].h + 4.5) * S, 1.2 * S);
      ctx.stroke();
    }
  }

  // cable manager fingers
  if (dev.kind === 'cm' && face === 'front') {
    ctx.fillStyle = mix(bg, '#000000', 0.45);
    for (let u = earW + 18; u < W - earW - 12; u += 22) {
      rr(ctx, cx(u), cy(H - 6), 9 * S, (H - 12) * S, 2 * S);
      ctx.fill();
    }
    blocks.push([earW + 14, W - earW - 8]);
  }

  // id + title block
  const lo = earW + 3;
  const hi = W - earW - 3;
  const free = freeIntervals(lo, hi, blocks).filter(([a, b]) => b - a > 8);
  if (free.length) {
    let pick = free.find(([a, b]) => b - a >= 40) || free.reduce((p, q) => (q[1] - q[0] > p[1] - p[0] ? q : p));
    if (face === 'rear') pick = free.reduce((p, q) => (q[1] - q[0] > p[1] - p[0] ? q : p));
    const bw = pick[1] - pick[0];
    const idSize = Math.min(H * (face === 'top' ? 0.18 : 0.36), face === 'top' ? 34 : H > 60 ? 22 : 13, bw / Math.max(3, (shortId || dev.id).length * 0.62));
    const lines = [{ text: shortId || dev.id, weight: 800, size: idSize, color: fg, family: FONT }];
    let title = dev.title || '';
    if (title.startsWith(dev.id)) title = title.slice(dev.id.length).replace(/^\s*[·:,-]\s*/, '');
    if (face === 'rear') title = title ? `REAR · ${title}` : 'REAR';
    if (title) lines.push({ text: title, weight: 500, size: Math.max(3.4, idSize * 0.44), color: sub, wrap: true, maxLines: H > 60 ? 3 : 2, optional: true });
    if (dev.sub && face !== 'rear' && dev.sub !== title && H > 30) lines.push({ text: dev.sub, weight: 400, size: Math.max(3.2, idSize * 0.36), color: faint, wrap: true, maxLines: 2, optional: true });
    if (dev.u && (face === 'front' || face === 'rear') && bw > 60 && H < 100) {
      // U tag at the far end of the block
    }
    drawBlock(ctx, S, [pick[0] + 1, 1.2, bw - 2, H - 2.4], lines, face === 'top' ? 'center' : 'left');
  }
  return canvasTexture(canvas);
}

export function drawPlain({ W, H, S, bg, ears, uN, caption }) {
  S = fitScale(W, H, S);
  const canvas = makeCanvas(W, H, S);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const earW = ears ? 16 : 0;
  if (ears) {
    for (const x0 of [0, W - earW]) {
      ctx.fillStyle = mix(bg, '#ffffff', 0.06);
      ctx.fillRect(x0 * S, 0, earW * S, canvas.height);
      const n = Math.max(1, uN || 1);
      for (let k = 0; k < n; k++) {
        for (const hv of [6.35, 38.1]) {
          const v = k * 44.45 + hv;
          ctx.fillStyle = '#0d0f12';
          ctx.beginPath();
          ctx.arc((x0 + earW / 2) * S, (H - v) * S, 1.7 * S, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    // vent slots
    ctx.fillStyle = mix(bg, '#000000', 0.35);
    for (let u = earW + 30; u < W - earW - 30; u += 14) {
      rr(ctx, u * S, (H / 2 - 9) * S, 5 * S, 18 * S, 2 * S);
      ctx.fill();
    }
  }
  if (caption) {
    ctx.fillStyle = mix(bg, '#ffffff', 0.45);
    ctx.font = `600 ${Math.min(9, H * 0.25) * S}px ${FONT}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(caption, (earW + 4) * S, (H / 2) * S);
  }
  return canvasTexture(canvas);
}

// U numbers + square mounting holes on the front face of a rack post.
export function drawRailFace({ W, H, S, u1, units, side }) {
  S = fitScale(W, H, S);
  const canvas = makeCanvas(W, H, S);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#3a4048';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const g = ctx.createLinearGradient(0, 0, canvas.width, 0);
  g.addColorStop(0, 'rgba(255,255,255,0.10)');
  g.addColorStop(1, 'rgba(0,0,0,0.12)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const cy = (y) => (H - y) * S;
  // hole column near the inner edge (where the ears land)
  const holeU = side === 'left' ? W - 8 : 8;
  for (let k = 0; k < units; k++) {
    const y0 = u1 + k * 44.45;
    for (const hv of [6.35, 22.225, 38.1]) {
      ctx.fillStyle = '#0c0e11';
      ctx.fillRect((holeU - 4.75) * S, cy(y0 + hv) - 4.75 * S, 9.5 * S, 9.5 * S);
    }
    // U boundary tick
    ctx.fillStyle = '#d7dde5';
    ctx.fillRect(0, cy(y0) - 0.3 * S, (W - 18) * S, 0.6 * S);
    if (side === 'left') {
      ctx.fillStyle = '#ffffff';
      ctx.font = `700 ${11 * S}px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(k + 1), ((W - 16) / 2) * S, cy(y0 + 22.225));
    }
  }
  ctx.fillStyle = '#d7dde5';
  ctx.fillRect(0, cy(u1 + units * 44.45) - 0.3 * S, (W - 18) * S, 0.6 * S);
  return canvasTexture(canvas);
}

export function stripeTexture() {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 4;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 64, 4);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, 36, 4);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  return t;
}

// Light / dark bands along a cable (multiplies the class colour): DC, GNSS and
// WG carry them so they stay apart from POE, FIB and AC on paper.
export function bandTexture() {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 4;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 64, 4);
  ctx.fillStyle = '#4a4a4a';
  ctx.fillRect(40, 0, 24, 4);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  return t;
}
