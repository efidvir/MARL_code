// Render the installation-guide pictures from the digital-twin viewer.
//
//   node twin/render.mjs [poc1] [poc2] [--only=<regex>]
//
// Opens out/twin/<poc>_twin.html headless (1600 x 1000 CSS px unless a render asks
// for another size, deviceScaleFactor 2), drives window.twin and writes into
// out/twin/<poc>_renders/:
//   overview_iso / _front / _rear / _top .png       whole-bench views
//   overview_front_rack.png, overview_rear_rack.png  rack elevations: portrait, no cables,
//                                                    every item labelled, left / right alternating
//   overview_table.png       bench-table top from the operator side (~50 deg), cropped to the table
//   overview_table_plan.png  bench table straight from above
//   step_NN.png              state after step NN (cumulative), its new items highlighted.
//                            Rear steps show a straight-on band of the rack (narrow field of
//                            view, every U labelled); a band taller than 22U is split into
//                            step_NN.png (upper) and step_NN_lower.png; step_NN_bench.png adds
//                            the whole bench when the step's cables also run to the table.
//                            Table steps are cropped to the table top (the operator-station
//                            step to the PC, monitor and keyboard).
//   sb_SBn.png, sb_rear_SBn.png   Spider box front / rear panel, straight on, one 1U, no cables
//   panel_<ID>.png           the connector face of a switch, DC unit, PDU, GNSS splitter or
//                            the MWATT (if the bench has one), straight on, no cables (the guide
//                            tags every port)
//   link_<NAME>.png          IF link / coupling arm, its cables traced and highlighted
//   link_<NAME>_tiles.png    the Spider-box tiles of that link, straight on (no cables)
//   link_<NAME>_pads.png     (fixed-pad benches) the link's pad stacks close up: on the splitter
//                            tray seen from above (the units above it left out), or on the tiles
//                            seen from the front right
//   tray_<ID>.png            (fixed-pad benches) the splitter tray from above with the pad stacks
//                            on its splitters, the units above and below left out, no cables
//   power.png, data_mgmt.png       AC + DC + POE only / FIB + ETH only
//   ip50_chains.png, ip50_chains_top.png    IP-50 conducted chains
//   ip50_col_1/2.png         (with an MWATT) one close-up per radio column beside it
//   ip50_hop_<HOP>.png       (fixed pads, no MWATT) one close-up per hop: radio, transition,
//                            pad stack, SMA cable, pad stack, transition, radio
//   event_t0.png             (PoC with a t0 event) the bench after t0
// Every picture carries an orientation badge (FRONT VIEW / REAR VIEW / ...) and the
// TENTATIVE mark; clipped pictures carry both in a band above the drawing.
// manifest.json: per render its title, view, side, clip, the trace paths and the
// anchors (image fractions of the saved PNG) that twin/guide.py uses for its tags,
// plus the after-t0 'dark' set the viewer derives.
// Names are stable: they depend only on step numbers, link names and device ids.
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync, existsSync, readdirSync, unlinkSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..'); // testbed_design/
const outDir = join(root, 'out', 'twin');

const args = process.argv.slice(2);
const only = (args.find((a) => a.startsWith('--only=')) || '').slice(7);
const onlyRe = only ? new RegExp(only) : null;
const pocs = args.filter((a) => !a.startsWith('--'));
if (!pocs.length) pocs.push('poc1', 'poc2');

const VIEWPORT = { width: 1600, height: 1000 };
const U = 44.45;
const SCALE = 2;
const pad2 = (n) => String(n).padStart(2, '0');
const RACK_X = [-60, 545]; // rack width incl. posts, scene mm
const LABEL_CSS = `
.tlabel { font-size: 15px !important; padding: 2px 7px !important; }
.tlabel.part { font-size: 13.5px !important; padding: 1px 5px !important; }
.tlabel.cab { font-size: 14.5px !important; padding: 1px 6px !important; }
.tlabel.cab i { width: 10px !important; height: 10px !important; }
.tlabel.rackcall { font-size: 14px !important; line-height: 19px !important; padding: 0 6px !important; }
.tlabel.off { font-size: 14px !important; }
.watermark { font-size: 14px !important; padding: 4px 10px !important; }
`;

// ----------------------------------------------------------------- in-page
// Runs inside the page (serialised by page.evaluate): reset, apply one spec,
// render, and return the anchors / clip in CSS pixels.
function pageApply(spec) {
  const t = window.twin;
  const app = t.app;
  const V = app.viewer;
  const STRIP = spec.strip || 32;
  if (!document.getElementById('render-css')) {
    const css = document.createElement('style');
    css.id = 'render-css';
    css.textContent = `body.hide-tl .tlabel{display:none!important} body.hide-wm .watermark{display:none!important}
.rbadge{position:fixed;z-index:21;font:800 13.5px Inter,Arial,sans-serif;letter-spacing:.05em;color:#fff;background:#1f2933;border-radius:4px;padding:3px 9px;pointer-events:none;white-space:nowrap}
.rbadge i{font-style:normal;font-weight:600;letter-spacing:0;color:#d5dbe3;margin-left:7px}
.rstrip{position:fixed;z-index:20;background:#f6f7f9;border-bottom:1px solid #d5dae1;pointer-events:none}`;
    document.head.appendChild(css);
    const strip = document.createElement('div');
    strip.className = 'rstrip';
    const badge = document.createElement('div');
    badge.className = 'rbadge';
    document.body.append(strip, badge);
  }
  document.body.classList.toggle('hide-tl', !!spec.hideLabels);
  // the viewport may have changed for this render
  V.resize();
  // The app methods are used directly (window.twin.* renders after every
  // call, which at 3200 x 2000 in software GL costs seconds); one render at the end.
  // ---- reset to a known state
  if (app.uiMode !== 'none') app.setUI('none');
  V.camera.fov = spec.fov || 30;
  V.camera.updateProjectionMatrix();
  V.state.rackCallAlt = !!spec.rackAlt;
  V.state.rackCallAll = !!spec.rackCallAll;
  V.state.cableLabelMax = spec.cableLabelMax || 16;
  V.state.partLabels = spec.partLabels !== false;
  app.setScenario('before');
  app.setLabels(true);
  app.setXray(false);
  app.showTable(true);
  app.showBlanks(true);
  app.setDims(false);
  app.setConfTint(false);
  app.setColorMode('class');
  app.setClasses(null);
  app.setStep(null);
  app.clearSelection();
  V.currentView = null; // set again only by a named view below
  // ---- apply
  const o = spec.opts || {};
  if (o.classes) app.setClasses(o.classes);
  if (o.xray != null) app.setXray(o.xray);
  if (o.table != null) app.showTable(o.table);
  if (o.blanks != null) app.showBlanks(o.blanks);
  if (o.labels != null) app.setLabels(o.labels);
  if (o.scenario) app.setScenario(o.scenario);
  const traces = {};
  const boxFramed = !!(spec.view || spec.box || spec.boxIds || spec.subject);
  if (spec.step != null) app.setStep(spec.step, { frame: !boxFramed });
  if (spec.view) app.setView(spec.view);
  if (spec.trace) {
    // union of the traces of the listed cables
    const dev = new Set();
    const prt = new Set();
    for (const cid of spec.trace) {
      traces[cid] = app.doTrace(cid, false);
      const tr = app.trace;
      if (!tr) continue;
      tr.devices.forEach((x) => dev.add(x));
      tr.ports.forEach((x) => prt.add(x));
    }
    app.highlight({ devices: [...dev], cables: spec.trace, ports: [...prt] });
  } else if (spec.hl) {
    app.highlight(spec.hl);
  }
  if (o.classes && (spec.trace || spec.hl)) app.setClasses(o.classes);
  // ---- camera
  const unionBox = (items) => {
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    for (const [kind, id] of items) {
      const b = V.boxOf(kind, id);
      if (!b) continue;
      for (let k = 0; k < 3; k++) {
        mn[k] = Math.min(mn[k], b.min[k]);
        mx[k] = Math.max(mx[k], b.max[k]);
      }
    }
    return { min: mn, max: mx };
  };
  const insets = spec.insets ? { top: 0, right: 0, bottom: 0, left: 0, ...spec.insets } : { top: 0, right: 0, bottom: 0, left: 0 };
  let frame = null;
  if (spec.subject) {
    frame = V.subjectBox(spec.subject);
    V.frameBox(frame.min, frame.max, spec.dir, { margin: spec.margin || 1.04, fov: spec.fov || 30, insets });
  } else if (spec.box || spec.boxIds) {
    const b = spec.box ? { min: spec.box[0].slice(), max: spec.box[1].slice() } : unionBox(spec.boxIds);
    if (spec.boxPad) for (let k = 0; k < 3; k++) { b.min[k] -= spec.boxPad[k]; b.max[k] += spec.boxPad[k]; }
    if (spec.boxClampX) {
      b.min[0] = Math.min(b.min[0], spec.boxClampX[0]);
      b.max[0] = Math.max(b.max[0], spec.boxClampX[1]);
    }
    frame = b;
    V.frameBox(b.min, b.max, spec.dir, { margin: spec.margin || 1.08, fov: spec.fov || 30, insets });
  }
  // devices left out of this picture (e.g. the lab wall in front of a rear panel)
  V._checkWallFade();
  for (const id of spec.hideDevices || []) {
    const e = V.dev.get(id);
    if (!e) continue;
    e.grp.visible = false;
    for (const pid of e.d.ports || []) {
      const pe = V.portObj.get(pid);
      if (pe) pe.mesh.visible = false;
    }
  }
  V.renderNow();
  // ---- projections (CSS px of the page)
  const cam = V.camera;
  const Vec = cam.position.constructor;
  const rect = V.renderer.domElement.getBoundingClientRect();
  const proj = (p) => {
    const v = new Vec(p[0], p[1], -p[2]).project(cam);
    return [rect.left + ((v.x + 1) / 2) * rect.width, rect.top + ((1 - v.y) / 2) * rect.height, v.z > -1 && v.z < 1];
  };
  let clip = null;
  if (spec.clip) {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    const [mn, mx] = spec.clip.box === 'rack' ? [V.rackBox.min, V.rackBox.max] : spec.clip.box === 'frame' ? [frame.min, frame.max] : spec.clip.box;
    for (const x of [mn[0], mx[0]]) for (const y of [mn[1], mx[1]]) for (const z of [mn[2], mx[2]]) {
      const [sx, sy] = proj([x, y, z]);
      x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
    }
    if (spec.clip.labels) {
      // rack callouts ("U23 SB1") sit beside the rack: keep them in the clip
      for (const el of document.querySelectorAll('.tlabel.rackcall')) {
        const r = el.getBoundingClientRect();
        if (!r.width || getComputedStyle(el).display === 'none' || getComputedStyle(el).visibility === 'hidden' || +getComputedStyle(el).opacity === 0) continue;
        if (r.bottom < y0 - 4 || r.top > y1 + 4) continue;
        x0 = Math.min(x0, r.left); x1 = Math.max(x1, r.right); y0 = Math.min(y0, r.top); y1 = Math.max(y1, r.bottom);
      }
      // labels of items just beside the rack (earth bar, arrestor, cable bridge) that the clip would cut
      const [cx0, cx1] = [x0, x1];
      for (const el of document.querySelectorAll('.tlabel:not(.rackcall)')) {
        const r = el.getBoundingClientRect();
        if (!r.width || getComputedStyle(el).display === 'none' || getComputedStyle(el).visibility === 'hidden' || +getComputedStyle(el).opacity === 0) continue;
        if (r.bottom < y0 || r.top > y1) continue;
        const cut = (r.left < cx0 && r.right > cx0) || (r.left < cx1 && r.right > cx1);
        if (!cut || r.left < cx0 - 160 || r.right > cx1 + 160) continue;
        x0 = Math.min(x0, r.left); x1 = Math.max(x1, r.right);
      }
    }
    const pad = spec.clip.pad ?? 12;
    x0 = Math.max(0, Math.floor(x0 - pad));
    y0 = Math.max(0, Math.floor(y0 - pad));
    x1 = Math.min(window.innerWidth, Math.ceil(x1 + pad));
    y1 = Math.min(window.innerHeight, Math.ceil(y1 + pad));
    clip = { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
  }
  // ---- orientation badge (FRONT / REAR / ...) from the camera direction
  const tg = V.controls.target;
  let d = [cam.position.x - tg.x, cam.position.y - tg.y, -(cam.position.z - tg.z)];
  const L = Math.hypot(d[0], d[1], d[2]) || 1;
  d = d.map((v) => v / L);
  const el = (Math.atan2(d[1], Math.hypot(d[0], d[2])) * 180) / Math.PI;
  const az = (Math.atan2(d[0], -d[2]) * 180) / Math.PI; // 0 = from the front, +90 = from the right (table side), 180 = from the rear
  let side;
  let text;
  let sub;
  if (el > 66) {
    side = 'top';
    text = 'VIEW FROM ABOVE';
    sub = d[2] <= 0 ? 'operator side at the bottom' : 'rear at the bottom';
  } else if (Math.abs(az) <= 25) {
    side = 'front'; text = 'FRONT VIEW'; sub = 'operator side: the rack front faces you';
  } else if (Math.abs(az) >= 155) {
    side = 'rear'; text = 'REAR VIEW'; sub = 'from behind the rack: its left and right are swapped';
  } else if (az > 25 && az < 70) {
    side = 'front-right'; text = 'FRONT RIGHT'; sub = 'from the operator side, bench-table side';
  } else if (az < -25 && az > -70) {
    side = 'front-left'; text = 'FRONT LEFT'; sub = 'from the operator side, left';
  } else if (az >= 70 && az <= 110) {
    side = 'right'; text = 'RIGHT SIDE'; sub = 'from the bench-table side';
  } else if (az <= -70 && az >= -110) {
    side = 'left'; text = 'LEFT SIDE'; sub = 'from the left of the rack';
  } else if (az > 110) {
    side = 'rear-right'; text = 'REAR VIEW'; sub = 'from behind, bench-table side: left and right are swapped';
  } else {
    side = 'rear-left'; text = 'REAR VIEW'; sub = 'from behind, left: left and right are swapped';
  }
  if (side !== 'top' && el > 28) sub += ', from above';
  if (spec.badge) [text, sub] = spec.badge;
  const badge = document.querySelector('.rbadge');
  const strip = document.querySelector('.rstrip');
  const wm = document.querySelector('.watermark');
  badge.innerHTML = `${text}<i>${sub}</i>`;
  badge.style.display = '';
  strip.style.display = 'none';
  wm.style.cssText = '';
  wm.style.display = '';
  if (clip) {
    // a band above the clip carries the badge and the TENTATIVE mark, outside the drawing
    const top = Math.max(0, clip.y - STRIP);
    clip.height += clip.y - top;
    clip.y = top;
    Object.assign(strip.style, { display: 'block', left: `${clip.x}px`, top: `${clip.y}px`, width: `${clip.width}px`, height: `${STRIP - 2}px` });
    const small = clip.width < 520;
    badge.style.fontSize = small ? '11.5px' : '';
    badge.querySelector('i').style.display = clip.width < 700 ? 'none' : '';
    Object.assign(badge.style, { left: `${clip.x + 6}px`, top: `${clip.y + 4}px` });
    const br = badge.getBoundingClientRect();
    wm.style.position = 'fixed';
    wm.style.right = 'auto';
    wm.style.bottom = 'auto';
    wm.style.zIndex = '21';
    if (small) wm.style.fontSize = '10.5px';
    if (small) wm.textContent = 'TENTATIVE';
    else wm.textContent = 'TENTATIVE · geometry partly estimated · not for construction';
    const r = wm.getBoundingClientRect();
    if (r.width + br.width + 24 > clip.width) wm.textContent = 'TENTATIVE';
    const r2 = wm.getBoundingClientRect();
    wm.style.left = `${clip.x + clip.width - r2.width - 6}px`;
    wm.style.top = `${clip.y + 4}px`;
  } else {
    wm.textContent = 'TENTATIVE · geometry partly estimated · not for construction';
    badge.style.fontSize = '';
    badge.querySelector('i').style.display = '';
    Object.assign(badge.style, { left: '12px', top: '10px' });
  }
  const cx = clip ? clip.x : 0;
  const cy = clip ? clip.y : 0;
  const cw = clip ? clip.width : window.innerWidth;
  const ch = clip ? clip.height : window.innerHeight;
  const anchors = {};
  for (const a of spec.anchors || []) {
    let p = null;
    if (a.kind === 'device') {
      const dv = app.model.devices.get(a.id);
      if (!dv) continue;
      const fz = a.face === 'rear' ? dv.pos[2] + dv.size[2] : a.face === 'front' ? dv.pos[2] : dv.pos[2] + dv.size[2] / 2;
      const fy = a.face === 'top' ? dv.pos[1] + dv.size[1] : dv.pos[1] + dv.size[1] / 2;
      const fx = a.face === 'right' ? dv.pos[0] + dv.size[0] : a.face === 'left' ? dv.pos[0] : dv.pos[0] + dv.size[0] / 2;
      p = [fx, fy, fz];
    } else if (a.kind === 'port') {
      const pp = app.model.ports.get(a.id);
      if (pp) p = pp.pos;
    } else if (a.kind === 'cable') {
      const e = V.cab.get(a.id);
      if (!e) continue;
      const pts = e.pts;
      const seg = [];
      let len = 0;
      for (let i = 1; i < pts.length; i++) {
        const dd = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
        seg.push(dd);
        len += dd;
      }
      const target = len * (a.at == null ? 0.5 : a.at);
      let acc = 0;
      p = pts[0];
      for (let i = 1; i < pts.length; i++) {
        if (acc + seg[i - 1] >= target) {
          const f = seg[i - 1] ? (target - acc) / seg[i - 1] : 0;
          p = [0, 1, 2].map((k) => pts[i - 1][k] + f * (pts[i][k] - pts[i - 1][k]));
          break;
        }
        acc += seg[i - 1];
      }
    } else if (a.kind === 'point') {
      p = a.p;
    }
    if (!p) continue;
    const [sx, sy, ok] = proj(p);
    const x = (sx - cx) / cw;
    const y = (sy - cy) / ch;
    anchors[a.key || a.id] = { x: +x.toFixed(4), y: +y.toFixed(4), in: ok && x >= 0 && x <= 1 && y >= 0 && y <= 1 };
  }
  const st = t.getState();
  return {
    anchors, traces, clip, view: st.view, side, badge: `${text} · ${sub}`,
    size: { w: cw, h: ch },
    strip: clip ? STRIP : 0,
    camera: { pos: cam.position.toArray().map((v) => +v.toFixed(1)), target: V.controls.target.toArray().map((v) => +v.toFixed(1)), fov: cam.fov },
  };
}

// ----------------------------------------------------------------- render list
function buildSpecs(scene) {
  const specs = [];
  const devs = new Map(scene.devices.map((d) => [d.id, d]));
  const ports = new Map(scene.ports.map((p) => [p.id, p]));
  const cabs = new Map(scene.cables.map((c) => [c.id, c]));
  const portCable = new Map();
  for (const c of scene.cables) {
    portCable.set(c.a, c);
    portCable.set(c.b, c);
  }
  const R = scene.rack;
  const T = scene.table;
  const W = R.panel_w;
  const rd = R.rail_depth;
  const rackTopY = R.u1_y + R.units * U;
  const INS = { top: 40 }; // room above a clipped subject for the badge band
  const idsOf = (cls) => scene.cables.filter((c) => cls.includes(c.cls)).map((c) => c.id);
  const devAnchors = (ids, face) => ids.filter((id) => devs.has(id)).map((id) => ({ kind: 'device', id, face }));
  const rackDevs = scene.devices.filter((d) => d.group === 'rack' && !d.parent && d.kind !== 'blank').map((d) => d.id);
  const tableDevs = scene.devices.filter((d) => d.group === 'table').map((d) => d.id);
  const isTile = (pid) => /^SB\d+\.M\d+\.T\d+$/.test(pid);
  const spiders = scene.devices.filter((d) => d.kind === 'spider').sort((a, b) => b.pos[1] - a.pos[1]);
  const overlapXZ = (a, b) => Math.min(
    Math.min(a.pos[0] + a.size[0], b.pos[0] + b.size[0]) - Math.max(a.pos[0], b.pos[0]),
    Math.min(a.pos[2] + a.size[2], b.pos[2] + b.size[2]) - Math.max(a.pos[2], b.pos[2]),
  );
  // something stands on d (small gap allowed): its top is hidden, tag its front instead
  const covered = (d) => scene.devices.some((o) => o !== d && o.group === d.group && !o.parent
    && o.pos[1] - (d.pos[1] + d.size[1]) >= -1 && o.pos[1] - (d.pos[1] + d.size[1]) <= 15 && overlapXZ(o, d) > 10);
  const tableAnchors = (ids) => ids.filter((id) => devs.has(id)).map((id) => {
    const d = devs.get(id);
    return { kind: 'device', id, face: d.parent ? 'center' : covered(d) ? 'front' : 'top' };
  });
  // tile port + a point just under it (where the guide puts the tile number)
  const tileAnchors = (pids) => pids.flatMap((id) => {
    const p = ports.get(id);
    return p ? [{ kind: 'port', id }, { kind: 'point', key: `${id}#below`, p: [p.pos[0], p.pos[1] - 15, p.pos[2]] }] : [];
  });
  const spiderFront = (sbIds, padY) => {
    const ds = sbIds.map((id) => devs.get(id)).filter(Boolean);
    const y0 = Math.min(...ds.map((d) => d.pos[1])) - padY;
    const y1 = Math.max(...ds.map((d) => d.pos[1] + d.size[1])) + padY;
    return [[-36, y0, -12], [W + 36, y1, 0]];
  };
  // a horizontal band of the rack, straight on, with the U callouts beside it
  const LBL = 300;
  const band = (y0, y1, side) => {
    const zf = side === 'rear' ? rd : 0;
    return {
      box: [[-LBL, y0, 0], [W + LBL, y1, rd]],
      dir: side === 'rear' ? [0, 0.05, 1] : [0, 0.05, -1],
      fov: 14, margin: 1.03, insets: INS, rackCallAll: true,
      clip: { box: [[-60, y0, zf], [W + 62, y1, zf]], labels: true, pad: 10 },
    };
  };
  // one face of a unit, straight on, cables hidden (ports tagged by the guide)
  const panel = (id, face) => {
    const d = devs.get(id);
    const zf = face === 'rear' ? d.pos[2] + d.size[2] : d.pos[2];
    const inset = d.group === 'rack' && face === 'rear' && d.size[0] > 470 ? (482.6 - 440) / 2 : 0;
    const x0 = d.pos[0] + inset;
    const x1 = d.pos[0] + d.size[0] - inset;
    const y0 = d.pos[1];
    const y1 = d.pos[1] + d.size[1];
    const facePorts = (d.ports || []).filter((pid) => {
      const p = ports.get(pid);
      return p && (face === 'rear' ? p.normal[2] > 0.5 : p.normal[2] < -0.5) && Math.abs(p.pos[2] - zf) < 15;
    });
    const m = 4;
    return {
      device: id, face, ports: facePorts,
      opts: { classes: [] }, hideLabels: true, hideDevices: ['WALL'],
      box: [[x0 - m, y0 - m, zf - 2], [x1 + m, y1 + m, zf + 2]],
      dir: face === 'rear' ? [0, 0, 1] : [0, 0, -1], fov: 8, margin: 1.0, insets: INS,
      clip: { box: [[x0 - m, y0 - m, zf], [x1 + m, y1 + m, zf]], pad: 2 },
      anchors: [...facePorts.map((pid) => ({ kind: 'port', id: pid })),
        { kind: 'point', key: '#left', p: [x0, (y0 + y1) / 2, zf] }, { kind: 'point', key: '#right', p: [x1, (y0 + y1) / 2, zf] }],
    };
  };
  const tableTop = () => ({
    box: [[T.x0, T.top, T.z0], [T.x1, T.top + 330, T.z1]],
    dir: [0, 1.2, -1], fov: 24, margin: 1.02, insets: INS,
    clip: { box: [[T.x0 - 20, T.top - 30, T.z0 - 20], [T.x1 + 20, T.top + 330, T.z1 + 20]], pad: 8 },
  });
  const rackRange = (st) => {
    let y0 = Infinity;
    let y1 = -Infinity;
    const add = (d) => {
      if (!d || (d.group !== 'rack' && d.group !== 'room')) return;   // room: earth bars, arrestor by the rack
      const top = d.parent && d.group === 'rack' ? devs.get(d.parent) : d;
      y0 = Math.min(y0, top.pos[1]);
      y1 = Math.max(y1, top.pos[1] + top.size[1]);
    };
    for (const id of st.devices) add(devs.get(id));
    for (const cid of st.cables) {
      const c = cabs.get(cid);
      for (const pid of [c.a, c.b]) if (ports.get(pid)) add(devs.get(ports.get(pid).device));
    }
    return [y0, y1];
  };

  // ---------------- overview
  specs.push({ name: 'overview_iso', title: 'Whole bench, iso view', view: 'iso' });
  specs.push({ name: 'overview_front', title: 'Rack front, whole bench', view: 'front' });
  specs.push({ name: 'overview_rear', title: 'Rack rear, whole bench', view: 'rear' });
  for (const side of ['front', 'rear']) {
    // elevation drawing: portrait, no cables, every item labelled, left / right alternating
    specs.push({
      name: `overview_${side}_rack`, title: `Rack ${side} elevation`, viewport: { width: 1150, height: 1700 },
      opts: { classes: [], table: false }, rackAlt: true, rackCallAll: true, partLabels: false, hideDevices: ['WALL'],
      box: [[-240, R.u1_y - 90, 0], [W + 240, rackTopY + 70, rd]], dir: side === 'front' ? [0, 0.03, -1] : [0, 0.03, 1], fov: 12, margin: 1.02, insets: INS,
      clip: { box: 'rack', labels: true, pad: 10 },
      anchors: devAnchors(rackDevs, side),
    });
  }
  // the same elevations at twice the scale: upper and lower half of the rack
  const half = R.u1_y + Math.ceil(R.units / 2) * U;
  for (const side of ['front', 'rear']) {
    for (const [part, y0, y1] of [['upper', half - 8, rackTopY + 45], ['lower', R.u1_y - 75, half + 8]]) {
      specs.push({
        name: `overview_${side}_rack_${part}`, title: `Rack ${side} elevation, ${part} half`,
        ...band(y0, y1, side), opts: { classes: [], table: false }, partLabels: false, hideDevices: ['WALL'], fov: 12,
        rackCallAll: true,   // every U keeps its tag; room items beside the rack (earth bar, arrestor) give way
        anchors: devAnchors(rackDevs, side),
      });
    }
  }
  specs.push({ name: 'overview_table', title: 'Bench table, from the operator side above', hideLabels: true, ...tableTop(), anchors: tableAnchors(tableDevs.filter((id) => !['transition', 'padstack'].includes(devs.get(id).kind))) });
  specs.push({ name: 'overview_top', title: 'Plan view from above', view: 'top' });
  specs.push({
    name: 'overview_table_plan', title: 'Bench table seen from above',
    box: [[T.x0 - 30, T.top - 5, T.z0 - 30], [T.x1 + 30, T.top + 120, T.z1 + 30]], dir: [0, 1, -0.0015], margin: 1.02, insets: INS,
    clip: { box: [[T.x0 - 40, T.top, T.z0 - 60], [T.x1 + 40, T.top, T.z1 + 60]], pad: 16 },
    anchors: devAnchors(tableDevs, 'top'),
  });

  // ---------------- steps
  for (const st of scene.steps) {
    const face = st.view === 'rear' ? 'rear' : st.view === 'table' ? 'top' : 'front';
    const anchors = [...devAnchors(st.devices, face), ...st.cables.map((id) => ({ kind: 'cable', id }))];
    const base = { title: `Step ${st.n}: ${st.title}`, step: st.n, stepView: st.view, cableLabelMax: 40 };
    const xs = st.cables.flatMap((id) => (cabs.get(id).route || []).map((p) => p[0]));
    const toTable = xs.length && Math.max(...xs) > 650;
    const rr = rackRange(st);
    if (st.view === 'rear' && Number.isFinite(rr[0]) && (st.devices.length || st.cables.length)) {
      let [y0, y1] = rr;
      y0 = Math.max(R.u1_y - 70, y0 - 0.7 * U);
      y1 = Math.min(rackTopY + 50, y1 + 0.7 * U);
      const parts = y1 - y0 > 22 * U ? [[(y0 + y1) / 2 - 10, y1], [y0, (y0 + y1) / 2 + 10]] : [[y0, y1]];
      parts.forEach(([a, b], i) => specs.push({
        name: `step_${pad2(st.n)}${i ? '_lower' : ''}`, ...base, ...band(a, b, 'rear'), anchors,
        part: parts.length > 1 ? (i ? 'lower' : 'upper') : null,
      }));
      if (toTable) {
        specs.push({
          name: `step_${pad2(st.n)}_bench`, ...base, title: `${base.title} (whole bench)`,
          boxIds: st.cables.map((id) => ['cable', id]), boxClampX: RACK_X, dir: [0.35, 0.75, 1], margin: 1.04, anchors,
        });
      }
    } else if (st.view === 'table') {
      const aux = st.cables.length && st.cables.every((id) => cabs.get(id).cls === 'AUX');
      if (aux) {
        // operator station: zoom on the PC, monitor and keyboard and their cables
        const ends = [...new Set(st.cables.flatMap((id) => [cabs.get(id).a, cabs.get(id).b]).map((pid) => (ports.get(pid) || {}).device).filter(Boolean))];
        specs.push({
          name: `step_${pad2(st.n)}`, ...base, cableLabelMax: 16,
          boxIds: [...ends.map((id) => ['device', id]), ...st.cables.map((id) => ['cable', id])], boxPad: [60, 20, 60],
          dir: [0.35, 0.9, -1], fov: 26, margin: 1.1, anchors,
        });
      } else {
        // the bench-table chain (tableDevs: table group only, so the IP-50 pad stacks, never the IF ones)
        const chainKinds = ['ip50', 'mwatt', 'psu', 'padstack'];
        const wgStep = st.cables.length && st.cables.every((id) => cabs.get(id).cls === 'WG');
        const ids = !st.devices.length ? tableDevs.filter((id) => chainKinds.includes(devs.get(id).kind))
          : wgStep ? [...tableDevs.filter((id) => chainKinds.includes(devs.get(id).kind)), ...st.devices] : st.devices;
        specs.push({
          name: `step_${pad2(st.n)}`, ...base, hideLabels: true, ...tableTop(),
          anchors: [...tableAnchors(ids.filter((id) => devs.get(id).kind !== 'transition')), ...st.cables.map((id) => ({ kind: 'cable', id }))],
        });
      }
    } else {
      specs.push({ name: `step_${pad2(st.n)}`, ...base, fov: st.view === 'front' ? 20 : 30, anchors });
    }
  }

  // ---------------- Spider boxes, straight on, one 1U, no cables
  for (const d of spiders) {
    const tiles = (d.ports || []).filter(isTile);
    const box = spiderFront([d.id], 30);
    specs.push({
      name: `sb_${d.id}`, title: `${d.id} front panel (${d.u})`, device: d.id,
      opts: { classes: [] }, hideLabels: true,
      // as fitted in the Spider-box step: the pad stacks (IF-link steps) would hide the tiles end on
      hideDevices: scene.devices.filter((x) => x.kind === 'padstack' && x.parent === d.id).map((x) => x.id),
      box, dir: [0, 0, -1], fov: 8, margin: 1.0, insets: INS,
      clip: { box: [[box[0][0], d.pos[1] - 1, 0], [box[1][0], d.pos[1] + d.size[1] + 1, 0]], pad: 2 },
      anchors: [
        ...tileAnchors(tiles),
        ...(d.modules || []).map((m) => ({ kind: 'point', key: `${d.id}.${m.m}`, p: [(m.x0 + m.x1) / 2, d.pos[1] + d.size[1], 0] })),
      ],
      tiles,
    });
    const rp = panel(d.id, 'rear');
    specs.push({ name: `sb_rear_${d.id}`, title: `${d.id} rear panel (${d.u}), seen from the rear`, ...rp });
  }

  // ---------------- splitter tray from above, with any pad stacks on its splitters (fixed-pad benches: the tray step's
  // picture, its joints made in the hand)
  for (const tr of scene.devices.filter((d) => d.kind === 'tray')) {
    const spl = scene.devices.filter((d) => d.parent === tr.id && d.kind === 'splitter');
    const pads = scene.devices.filter((d) => d.kind === 'padstack' && spl.some((x) => x.id === d.parent));
    if (!spl.length || !(pads.length || (scene.pad_schedule || []).length)) continue;
    const top = tr.pos[1] + tr.size[1];
    const away = scene.devices.filter((d) => d.group === 'rack' && !d.parent && d.id !== tr.id
      && (d.pos[1] >= top - 1 || d.pos[1] + d.size[1] <= tr.pos[1] + 1)).map((d) => d.id);
    const kids = scene.devices.filter((d) => d.parent && away.includes(d.parent)).map((d) => d.id);
    const zs = [...spl, ...pads].flatMap((d) => [d.pos[2], d.pos[2] + d.size[2]]);
    const ys = [...spl, ...pads].flatMap((d) => [d.pos[1], d.pos[1] + d.size[1]]);
    specs.push({
      name: `tray_${tr.id}`, title: `${tr.id} from above: the splitters` + (pads.length ? ' and the pad stacks on their ports' : ''), device: tr.id,
      opts: { classes: [] }, hideLabels: true, hideDevices: [...away, ...kids, 'WALL'],
      box: [[tr.pos[0] - 5, Math.min(...ys) - 5, Math.min(...zs) - 20], [tr.pos[0] + tr.size[0] + 5, Math.max(...ys) + 10, Math.max(...zs) + 15]],
      dir: [0.06, 2.6, -1], fov: 18, margin: 1.04, insets: INS, clip: { box: 'frame', pad: 6 },
      badge: ['VIEW FROM ABOVE', 'front edge at the bottom; the units above and below the tray left out'],
      anchors: [...spl.map((d) => ({ kind: 'device', id: d.id, face: 'top' })), ...pads.map((d) => ({ kind: 'device', id: d.id, face: 'center' })),
        ...spl.flatMap((d) => (d.ports || []).map((pid) => ({ kind: 'port', id: pid })))],
      pads: pads.map((d) => d.id),
    });
  }

  // ---------------- unit faces the guide tags port by port
  const faceOf = (id) => {
    const d = devs.get(id);
    const ps = (d.ports || []).map((pid) => ports.get(pid)).filter(Boolean);
    const fr = ps.filter((p) => p.normal[2] < -0.5).length;
    const re = ps.filter((p) => p.normal[2] > 0.5).length;
    return fr >= re ? 'front' : 'rear';
  };
  for (const d of scene.devices) {
    if (!['sw_data', 'sw_oob', 'dcdist', 'fuse', 'pdu', 'gnss', 'mwatt'].includes(d.kind)) continue;
    const f = faceOf(d.id);
    specs.push({ name: `panel_${d.id}`, title: `${d.id} ${f} panel, straight on`, ...panel(d.id, f) });
  }

  // ---------------- IF links / coupling arms
  const linkTiles = (rf) => {
    // tile ports of a link: direct, behind a splitter / coupler, or under a pad stack
    // (the cable starts on the stack's OUT; its IN is screwed onto a tile or a splitter port)
    const out = [];
    const add = (pid) => { if (isTile(pid) && !out.includes(pid)) out.push(pid); };
    for (const id of rf) {
      const c = cabs.get(id);
      for (const end of [c.a, c.b]) {
        let pid = end;
        const pe = ports.get(pid);
        const pd = pe && devs.get(pe.device);
        if (pd && pd.kind === 'padstack') {
          const pin = ports.get(`${pd.id}.in`);
          if (!pin || !pin.mate) continue;
          pid = pin.mate;
        }
        add(pid);
        const p = ports.get(pid);
        const d = p && devs.get(p.device);
        if (!d || !['splitter', 'coupler'].includes(d.kind)) continue;
        for (const q of d.ports || []) {
          const cc = portCable.get(q);
          if (cc) [cc.a, cc.b].forEach(add);
        }
      }
    }
    return out;
  };
  for (const st of scene.steps) {
    const m = /^(?:IF link|Coupling arm)\s+(\S+)/.exec(st.title);
    if (!m) continue;
    const name = m[1];
    const rf = st.cables.filter((id) => (cabs.get(id) || {}).cls === 'RF');
    const endDevs = new Set();
    for (const id of rf) {
      const c = cabs.get(id);
      for (const pid of [c.a, c.b]) if (ports.get(pid)) endDevs.add(ports.get(pid).device);
    }
    specs.push({
      name: `link_${name}`, title: `${st.title}: traced cables`, link: name, stepN: st.n,
      trace: rf,
      boxIds: rf.map((id) => ['cable', id]), boxPad: [20, 30, 0], boxClampX: RACK_X,
      dir: [0.22, 0.1, -1], margin: 1.08,
      anchors: [...rf.map((id) => ({ kind: 'cable', id })), ...devAnchors([...endDevs], 'front')],
    });
    // the link's IF pad stacks, fitted in its own step (on the A tiles, or on the B tiles after a hub splitter)
    const padIds = scene.devices.filter((d) => d.kind === 'padstack' && d.chain === 'if' && d.link === name).map((d) => d.id);
    if (padIds.length) {
      // the link's pad stacks close up, with their host ports and the first part of each cable
      const pads = padIds.map((id) => devs.get(id));
      const hosts = pads.map((d) => ports.get(d.host)).filter(Boolean);
      const onTray = pads.every((d) => (devs.get(d.parent) || {}).kind === 'splitter');
      const mn = [Infinity, Infinity, Infinity];
      const mx = [-Infinity, -Infinity, -Infinity];
      const grow = (p) => { for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); } };
      for (const d of pads) { grow(d.pos); grow([0, 1, 2].map((k) => d.pos[k] + d.size[k])); }
      for (const h of hosts) grow(h.pos);
      const parents = [...new Set(pads.map((d) => d.parent))].map((id) => devs.get(id)).filter(Boolean);
      let spec;
      if (onTray) {
        // the splitters and their stacks seen from above: every rack unit over the tray left out
        for (const d of parents) { grow(d.pos); grow([0, 1, 2].map((k) => d.pos[k] + d.size[k])); }
        const tray = devs.get(parents[0].parent);
        const top = tray ? tray.pos[1] + tray.size[1] : mx[1] + 20;
        const bottom = tray ? tray.pos[1] : mn[1] - 20;
        // every rack unit above and below the tray left out: the tray seen from above, on its own
        const away = scene.devices.filter((d) => d.group === 'rack' && !d.parent && d.id !== (tray && tray.id)
          && (d.pos[1] >= top - 1 || d.pos[1] + d.size[1] <= bottom + 1)).map((d) => d.id);
        const kids = scene.devices.filter((d) => d.parent && away.includes(d.parent)).map((d) => d.id);
        // the whole tray width: the link's column and the other one (pale) side by side
        const tx0 = tray ? Math.min(mn[0] - 60, tray.pos[0] - 5) : mn[0] - 60;
        const tx1 = tray ? Math.max(mx[0] + 40, tray.pos[0] + tray.size[0] + 5) : mx[0] + 40;
        spec = {
          box: [[tx0, mn[1] - 5, mn[2] - 20], [tx1, mx[1] + 10, mx[2] + 15]], dir: [0.06, 2.6, -1], fov: 18,
          hideDevices: [...away, ...kids, 'WALL'], opts: { classes: ['RF'] },
          badge: ['VIEW FROM ABOVE', 'splitter tray, front edge at the bottom; the units above and below it left out'],
        };
      } else {
        spec = {
          box: [[mn[0] - 170, mn[1] - 45, mn[2] - 30], [mx[0] + 150, mx[1] + 40, mx[2] + 10]], dir: [0.7, 0.42, -1], fov: 18,
          opts: { classes: ['RF'] },
        };
      }
      specs.push({
        name: `link_${name}_pads`, title: `${st.title}: pad stacks close up`, link: name, stepN: st.n,
        step: st.n, stepView: null, hideLabels: true, cableLabelMax: 0, margin: 1.06, insets: INS, clip: { box: 'frame', pad: 6 },
        ...spec,
        anchors: [...pads.map((d) => ({ kind: 'device', id: d.id, face: 'center' })),
          ...pads.map((d) => ({ kind: 'port', id: `${d.id}.in` })), ...pads.map((d) => ({ kind: 'port', id: `${d.id}.out` })),
          ...hosts.map((h) => ({ kind: 'port', id: h.id })), ...rf.map((id) => ({ kind: 'cable', id, at: 0.12 }))],
        pads: padIds,
      });
    }
    const tiles = linkTiles(rf);
    const sbIds = [...new Set(tiles.map((p) => p.split('.')[0]))];
    if (sbIds.length) {
      // the Spider-box rows of the link only, straight on, tight to the boxes
      const box = spiderFront(spiders.map((d) => d.id), 20);
      const band2 = spiderFront(sbIds, 1);
      specs.push({
        name: `link_${name}_tiles`, title: `${st.title}: Spider-box tiles`, link: name, stepN: st.n,
        opts: { classes: [] }, hideLabels: true,
        hl: { devices: spiders.map((d) => d.id), ports: tiles },
        box, dir: [0, 0, -1], fov: 8, margin: 1.0, insets: INS,
        clip: { box: [[band2[0][0], band2[0][1], 0], [band2[1][0], band2[1][1], 0]], pad: 2 },
        anchors: tileAnchors(tiles),
        tiles,
      });
    }
  }

  // ---------------- power, data / management, IP-50
  const rearIso = [-0.55, 0.5, 1];
  specs.push({ name: 'power', title: 'Power: AC mains, DC and PoE only (from the rear)', opts: { classes: ['AC', 'DC', 'POE'] }, subject: 'all', dir: rearIso });
  specs.push({ name: 'data_mgmt', title: 'Data (SW-DATA) and management (SW-OOB) only (from the rear)', opts: { classes: ['FIB', 'ETH'] }, subject: 'all', dir: rearIso });
  const wg = idsOf(['WG']);
  if (wg.length) {
    const ip = scene.devices.filter((d) => d.group === 'table' && ['ip50', 'transition', 'padstack', 'mwatt', 'psu'].includes(d.kind)).map((d) => d.id);
    const box = { boxIds: [...ip.map((id) => ['device', id]), ...wg.map((id) => ['cable', id])], boxPad: [60, 20, 60] };
    const anchors = [...wg.map((id) => ({ kind: 'cable', id })), ...devAnchors(ip, 'top')];
    specs.push({ name: 'ip50_chains', title: 'IP-50 conducted chains on the bench table', opts: { classes: ['WG'] }, hl: { devices: ip, cables: wg }, ...box, dir: [0, 1.05, -1], margin: 1.05, anchors });
    specs.push({ name: 'ip50_chains_top', title: 'IP-50 conducted chains, from above', opts: { classes: ['WG'] }, hl: { devices: ip, cables: wg }, ...box, dir: [0, 1, -0.0015], margin: 1.04, anchors });
    // one close-up per radio column (radios on either side of the MWATT)
    const mw = scene.devices.find((d) => d.kind === 'mwatt');
    const radios = scene.devices.filter((d) => d.group === 'table' && d.kind === 'ip50');
    if (mw && radios.length) {
      const cols = [radios.filter((d) => d.pos[0] + d.size[0] / 2 < mw.pos[0]), radios.filter((d) => d.pos[0] + d.size[0] / 2 >= mw.pos[0])];
      cols.forEach((col, i) => {
        if (!col.length) return;
        const sx = i ? -1 : 1; // column 1: transitions on its right (+x); column 2: on its left
        const ids = col.map((d) => d.id);
        const tr = scene.devices.filter((d) => d.kind === 'transition' && ids.includes(d.parent)).map((d) => d.id);
        const mine = new Set([...ids, ...tr]);
        const wgc = scene.cables.filter((c) => c.cls === 'WG' && [c.a, c.b].some((pid) => mine.has((ports.get(pid) || {}).device))).map((c) => c.id);
        specs.push({
          name: `ip50_col_${i + 1}`, title: `IP-50 radio column ${i + 1} (${i ? 'right of' : 'left of'} the MWATT), seen from the MWATT side`,
          badge: [`SEEN FROM THE ${i ? 'LEFT' : 'RIGHT'}, ABOVE`, `operator side on the ${i ? 'right' : 'left'}, wall side on the ${i ? 'left' : 'right'}`],
          opts: { classes: ['WG'] }, hideLabels: true, hl: { devices: [...ids, ...tr], cables: wgc, ports: [] },
          boxIds: [...ids, ...tr].map((id) => ['device', id]), boxPad: [40, 10, 40],
          dir: [sx, 1.15, 0], fov: 20, margin: 1.04, insets: INS, clip: { box: 'frame', pad: 6 },
          anchors: [...ids.map((id) => ({ kind: 'device', id, face: covered(devs.get(id)) ? (i ? 'left' : 'right') : 'top' })), ...tr.map((id) => ({ kind: 'device', id, face: 'center' })),
            ...wgc.map((id) => ({ kind: 'cable', id, at: 0.5 })),
            ...tr.flatMap((id) => (devs.get(id).ports || []).map((pid) => ({ kind: 'port', id: pid })))],
          radios: ids, transitions: tr, cables: wgc,
        });
      });
    }
    const hops = [...new Set(radios.filter((d) => d.hop).map((d) => d.hop))];
    if (!mw && hops.length) {
      // fixed pads: the two radios of a hop sit side by side, flanges facing (A on the left);
      // one close-up per hop from the operator side, above
      for (const hop of hops) {
        const rs = radios.filter((d) => d.hop === hop).sort((a, b) => a.pos[0] - b.pos[0]);
        const ids = rs.map((d) => d.id);
        const tr = scene.devices.filter((d) => d.kind === 'transition' && ids.includes(d.parent)).map((d) => d.id);
        const pads = scene.devices.filter((d) => d.kind === 'padstack' && tr.includes(d.parent)).map((d) => d.id);
        const mine = new Set([...ids, ...tr, ...pads]);
        const wgc = scene.cables.filter((c) => c.cls === 'WG' && [c.a, c.b].some((pid) => mine.has((ports.get(pid) || {}).device))).map((c) => c.id);
        specs.push({
          name: `ip50_hop_${hop}`, title: `IP-50 hop ${hop}: radio, transition, pads, SMA cable, pads, transition, radio`,
          opts: { classes: ['WG'] }, hideLabels: true, hl: { devices: [...mine], cables: wgc, ports: [] },
          boxIds: [...[...mine].map((id) => ['device', id]), ...wgc.map((id) => ['cable', id])], boxPad: [25, 10, 25],
          dir: [0, 1.0, -1], fov: 20, margin: 1.04, insets: INS, clip: { box: 'frame', pad: 6 },
          anchors: [...ids.map((id) => ({ kind: 'device', id, face: 'top' })), ...tr.map((id) => ({ kind: 'device', id, face: 'center' })),
            ...pads.map((id) => ({ kind: 'device', id, face: 'center' })), ...wgc.map((id) => ({ kind: 'cable', id, at: 0.5 })),
            ...[...tr, ...pads].flatMap((id) => (devs.get(id).ports || []).map((pid) => ({ kind: 'port', id: pid })))],
          radios: ids, transitions: tr, pads, cables: wgc, hop,
        });
      }
    }
  }
  if ((scene.meta.switched || []).length) {
    // the rack plus every switched device (some stand on the bench table)
    const sw = scene.meta.switched.map((id) => devs.get(id)).filter(Boolean);
    const mn = [RACK_X[0], 0, -100];
    const mx = [RACK_X[1], rackTopY + 60, rd];
    for (const d of sw) for (let k = 0; k < 3; k++) {
      mn[k] = Math.min(mn[k], d.pos[k]);
      mx[k] = Math.max(mx[k], d.pos[k] + d.size[k]);
    }
    specs.push({
      name: 'event_t0', title: 'After t0: switched-off devices dark (red), what they fed dark (brown), dead cables grey', opts: { scenario: 'after' },
      box: [mn, mx], boxPad: [80, 0, 80], dir: [0.62, 0.5, -1], margin: 1.04, anchors: devAnchors(scene.meta.switched, 'front'),
    });
  }
  return specs;
}

// ----------------------------------------------------------------- main
async function renderPoc(browser, poc) {
  const html = join(outDir, `${poc}_twin.html`);
  if (!existsSync(html)) throw new Error(`missing ${html} (run: node twin/viewer/build.mjs)`);
  const dir = join(outDir, `${poc}_renders`);
  mkdirSync(dir, { recursive: true });
  const ctx = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: SCALE });
  // larger ID tags than on screen so they stay legible on an A4 page; injected
  // before the viewer boots so its label layout measures the final sizes
  await ctx.addInitScript((css) => {
    document.addEventListener('DOMContentLoaded', () => {
      const st = document.createElement('style');
      st.id = 'print-labels';
      st.textContent = css;
      document.head.appendChild(st);
    });
  }, LABEL_CSS);
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(pathToFileURL(html).href + '?ui=none');
  await page.waitForFunction(() => window.twin && window.twin.ready, null, { timeout: 120000 });
  await page.evaluate(() => window.twin.ready);
  const scene = await page.evaluate(() => window.twin.scene);
  const specs = buildSpecs(scene);
  const names = specs.map((s) => s.name);
  const dup = names.filter((n, i) => names.indexOf(n) !== i);
  if (dup.length) throw new Error(`${poc}: duplicate render names ${dup.join(', ')}`);
  const manifest = {
    poc,
    name: scene.meta.name,
    viewport: VIEWPORT,
    scale: SCALE,
    n_cables: scene.cables.length,
    n_devices: scene.devices.length,
    n_steps: scene.steps.length,
    cable_ids: scene.cables.map((c) => c.id),
    renders: [],
  };
  const dark = await page.evaluate(() => {
    const d = window.twin.app.model.dark;
    return {
      devices: [...d.devices], ports: [...d.ports], cables: [...d.cables], links: [...d.links], nodes: d.nodes,
      why: Object.fromEntries([...d.why.entries()]),
    };
  });
  manifest.dark = dark;
  const t0 = Date.now();
  let vp = VIEWPORT;
  for (const spec of specs) {
    if (onlyRe && !onlyRe.test(spec.name)) continue;
    const tr0 = Date.now();
    const want = spec.viewport || VIEWPORT;
    if (want.width !== vp.width || want.height !== vp.height) {
      await page.setViewportSize(want);
      vp = want;
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r())));
    }
    const res = await page.evaluate(pageApply, spec);
    // let the label overlay settle for one more frame
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    const file = `${spec.name}.png`;
    const shot = { path: join(dir, file), type: 'png' };
    if (res.clip) shot.clip = res.clip;
    await page.screenshot(shot);
    manifest.renders.push({
      name: spec.name, file, title: spec.title, step: spec.step ?? spec.stepN ?? null, link: spec.link || null, device: spec.device || null,
      // the step's own view for step pictures, else the named view used (if any); 'side' is what the camera sees
      view: spec.stepView || spec.view || res.view || null, side: res.side, badge: res.badge,
      px: { w: Math.round(res.size.w * SCALE), h: Math.round(res.size.h * SCALE) }, clip: res.clip, strip_px: res.strip * SCALE,
      anchors: res.anchors, traces: res.traces, tiles: spec.tiles || null, camera: res.camera,
      face: spec.face || null, ports: spec.ports || null, part: spec.part || null,
      radios: spec.radios || null, transitions: spec.transitions || null, cables: spec.cables || null,
      ...(spec.pads ? { pads: spec.pads } : {}), ...(spec.hop ? { hop: spec.hop } : {}),
    });
    process.stdout.write(`  ${poc} ${file} (${((Date.now() - tr0) / 1000).toFixed(1)} s)\n`);
  }
  if (!onlyRe) {
    // drop renders an older scene produced (e.g. a step that no longer exists)
    const keep = new Set(manifest.renders.map((r) => r.file));
    for (const f of readdirSync(dir)) if ((f.endsWith('.png') && !keep.has(f)) || f === 'manifest.partial.json') unlinkSync(join(dir, f));
    writeFileSync(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 1));
  } else {
    writeFileSync(join(dir, 'manifest.partial.json'), JSON.stringify(manifest, null, 1));
  }
  console.log(`${poc}: ${manifest.renders.length} renders in ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${dir}`);
  await ctx.close();
  if (errors.length) throw new Error(`${poc}: page errors:\n  ${errors.join('\n  ')}`);
}

const browser = await chromium.launch();
try {
  // one PoC after the other: the software GL process is shared, so parallel
  // pages only compete for it
  for (const poc of pocs) {
    try {
      await renderPoc(browser, poc);
    } catch (e) {
      console.error(String(e && e.stack ? e.stack : e));
      process.exitCode = 1;
    }
  }
} finally {
  await browser.close();
}
