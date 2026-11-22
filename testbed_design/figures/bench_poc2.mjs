// The PoC 2 conducted bench as an Anatomy-style isometric figure
// (kit vendored from github.com/wheresryan22/anatomy, MIT).
//
// PoC 2 is a fixed point-to-point Terragraph ring on the two GTS003 Spider boxes, with two
// cabled co-channel couplings for DPIR to route around.  One 19-inch rack, top to bottom:
//   U19 switch, U18 shelf of four traffic hosts, U17 cable manager,
//   U16 GTS003 #1 (W1, E2: all sectors even), U15-U14 empty (PoC 1's hub and S/N boxes),
//   U13 GTS003 #2 (E1, W2: all odd), U12 the coupler tray (one 10 dB coupler per coupled sector:
//   W1.E, E2.W, E1.W, W2.E; in, coupled and through ports), U11-U8 the four link attenuators
//   ATT-1..4 (LN, LE, LS, LW), U7-U6 the coupling-arm attenuators ATT-5 (KW) and ATT-6 (KE),
//   U5-U3 blank (three PoC 1 attenuators spare), U2 the DC drawer, U1 the PDU.
// Front cables are drawn one per sector side (each link really uses four tile ports through
// four attenuator channels); the two coupling arms are the lit accent.  On the table beside the
// rack: the DPIR server and the TG E2E controller on the lower shelf, the spectrum analyser and
// the GNSS/PTP time server on top.
//
// World units: 1 unit = 2 mm; x across the rack front, y toward the front, z up.
//
// Run:  node figures/bench_poc2.mjs [--dark]   -> out/bench_poc2.html (+ out/bench_poc2.svg)

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../vendor/anatomy/kit/iso-kit.mjs";
import { tubePieces, tubeSvg } from "../vendor/anatomy/kit/tube.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "out");
const theme = process.argv.includes("--dark") ? "dark" : "light";

const WIDTH = 1200;
const HEIGHT = 720;

// ---- the rack ---------------------------------------------------------------------
const U = 22;                       // one rack unit, 44.45 mm
const FEET_H = 6;
const PLINTH_H = 14;
const Z0 = FEET_H + PLINTH_H;       // floor of the first unit
const UNITS = 20;
const ZTOP = Z0 + UNITS * U;        // top of the rails
const RACK = { x: 0, y: 0, w: 260, d: 200, r: 2 };
const POST = 12;
const FACE_Y = 198;                 // back of the front panels
const FACE_T = 3;                   // panel thickness
const FY = FACE_Y + FACE_T;         // front face plane
const PANEL = { x: 4, w: 252 };
const BODY = { x: 20, y: 30, w: 220, d: FACE_Y - 30 };
const zU = (u) => Z0 + (u - 1) * U;  // bottom of unit u (1-based)

// ---- the table --------------------------------------------------------------------
const TABLE = { x: 300, y: -250, w: 330, d: 200, r: 4 };
const TABLE_Z = 300;                // top of the table (600 mm)
const TABLE_T = 8;

const P = k.fitProjection(
  [...k.boxCorners(RACK, 0, ZTOP + 10), [128, FY + 70, Z0], ...k.boxCorners(TABLE, 0, TABLE_Z + 80)],
  WIDTH, HEIGHT, { pad: 30, azimuth: 56 },
);

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const fpt = (x, z, y = FY) => at([x, y, z]);
const faceRect = (x0, z0, w, h, y = FY) => k.pathOf([fpt(x0, z0, y), fpt(x0 + w, z0, y), fpt(x0 + w, z0 + h, y), fpt(x0, z0 + h, y)], true);
const sideRect = (x, y0, z0, d, h) => k.pathOf([at([x, y0, z0]), at([x, y0 + d, z0]), at([x, y0 + d, z0 + h]), at([x, y0, z0 + h])], true);
const box = (plan, z, h, style = {}, steps = 2, bevel = 0) => S(k.slabOf(plan, z, h, P, steps, bevel), style);

// ---- small parts on a front panel ------------------------------------------------
function sma(x, z, { lit = false } = {}) {
  // hex nut on the panel, the threaded body, the bore and its centre pin
  return [
    box({ x: x - 2.3, y: FY, w: 4.6, d: 1.4, r: 0.4 }, z - 2.3, 4.6, { tone: "lo" }),
    box({ x: x - 1.6, y: FY + 1.4, w: 3.2, d: 3.4, r: 0.9 }, z - 1.6, 3.2, { tone: lit ? "hi" : "mid", lit }),
    L(k.sideRing([x, FY + 4.8, z], 0.9, "y", P, 14), { tone: "lo" }),
    D([at([x, FY + 4.8, z])], { size: 0.35, tone: "hi" }),
  ].join("");
}

function screws(z0, h) {
  const out = [];
  for (const x of [8.5, PANEL.x + PANEL.w - 4.5]) {
    for (const z of [z0 + h * 0.28, z0 + h * 0.72]) {
      out.push(L(k.sideRing([x, FY, z], 1.5, "y", P, 14), { tone: "lo" }));
      out.push(D([fpt(x, z)], { size: 0.45, tone: "mid" }));
    }
  }
  return out.join("");
}

function labelPlate(x, z, w, h) {
  // a small engraved label plate: outline and three rule lines standing for text
  const out = [L(faceRect(x, z, w, h), { tone: "lo" })];
  [0.72, 0.5, 0.28].forEach((f, i) => {
    out.push(L(k.pathOf([fpt(x + 2, z + h * f), fpt(x + w * (i === 0 ? 0.8 : 0.55), z + h * f)]), { tone: "faint", free: true }));
  });
  return out.join("");
}

function vents(x0, z0, count, gap, h) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(faceRect(x0 + i * gap, z0, 1.6, h));
  return L(out.join(""), { tone: "faint" });
}

function leds(points) {
  return D(points.map(([x, z]) => fpt(x, z)), { size: 0.55, tone: "mid" });
}

function panel(u, units = 1) {
  const z0 = zU(u) + 0.4;
  const h = units * U - 0.8;
  return [
    box({ x: PANEL.x, y: FACE_Y, w: PANEL.w, d: FACE_T, r: 0.6 }, z0, h, { tone: "mid" }, 2, 0.4),
    screws(z0, h),
  ].join("");
}

// ---- devices ------------------------------------------------------------------------
// Spider boxes: four FATJ0498 sector modules of four SMA (tile) ports each, right of the vents.
const SPIDER_X0 = 104;
const MOD_W = 36;
const smaX = (m, i) => SPIDER_X0 + m * MOD_W + 6 + i * 8.2;
const SPIDERS = [16, 13];            // GTS003 #1 (W1 = M1, M2; E2 = M3, M4) and GTS003 #2 (E1; W2)

function spider(u, litPorts = []) {
  const z = zU(u);
  const zc = z + U / 2;
  const out = [panel(u)];
  out.push(vents(16, z + 4, 14, 4.2, U - 8));
  out.push(labelPlate(76, z + 5, 24, U - 10));
  for (let m = 0; m < 4; m++) {
    const x0 = SPIDER_X0 + m * MOD_W;
    out.push(L(faceRect(x0, z + 2.6, MOD_W - 2, U - 5.2), { tone: "lo" }));
    out.push(D([fpt(x0 + 1.6, z + 4), fpt(x0 + MOD_W - 3.6, z + 4)], { size: 0.4, tone: "mid" }));
    for (let i = 0; i < 4; i++) out.push(sma(smaX(m, i), zc, { lit: litPorts.some(([mm, ii]) => mm === m && ii === i) }));
  }
  return out.join("");
}

const ATT_X = (c, side) => 116 + c * 32 + (side === "out" ? 14 : 0);
const ATTS = [
  { u: 11, link: "LN" }, { u: 10, link: "LE" }, { u: 9, link: "LS" }, { u: 8, link: "LW" },
  { u: 7, link: "KW", lit: true }, { u: 6, link: "KE", lit: true },
];

function attenuator(u, lit = false) {
  const z = zU(u);
  const zc = z + U / 2;
  const out = [panel(u)];
  out.push(labelPlate(16, z + 5, 22, U - 10));
  // display window with its recess line, and a row of status LEDs
  out.push(L(faceRect(44, z + 5, 34, U - 10), { tone: "lo" }));
  out.push(L(faceRect(45.6, z + 6.6, 30.8, U - 13.2), { tone: "faint" }));
  out.push(leds([[84, zc + 3], [84, zc - 3], [90, zc + 3], [90, zc - 3]]));
  for (let c = 0; c < 4; c++) {
    out.push(L(faceRect(ATT_X(c, "in") - 5, z + 3, 24, U - 6), { tone: "faint" }));
    out.push(sma(ATT_X(c, "in"), zc, { lit: lit && c === 0 }), sma(ATT_X(c, "out"), zc, { lit: lit && c === 0 }));
  }
  return out.join("");
}

// the coupler tray: one directional coupler per coupled sector, ports in / coupled / through
const CPL_X0 = (j) => 30 + j * 56;
const CPL_PORT = { in: 9, cpl: 23, thru: 37 };
const COUPLERS = ["W1.E", "E2.W", "E1.W", "W2.E"];

function couplerTray(u) {
  const z = zU(u);
  const zc = z + U / 2;
  const out = [panel(u)];
  for (let j = 0; j < 4; j++) {
    const x0 = CPL_X0(j);
    out.push(box({ x: x0, y: FY, w: 46, d: 5, r: 1 }, z + 4, U - 8, { tone: "mid" }, 2, 0.4));
    for (const [name, dx] of Object.entries(CPL_PORT)) {
      const x = x0 + dx;
      out.push(box({ x: x - 1.6, y: FY + 5, w: 3.2, d: 3, r: 0.9 }, zc - 1.6, 3.2, { tone: name === "cpl" ? "hi" : "mid", lit: name === "cpl" }));
      out.push(L(k.sideRing([x, FY + 8, zc], 0.9, "y", P, 14), { tone: "lo" }), D([at([x, FY + 8, zc])], { size: 0.35, tone: "hi" }));
    }
  }
  return out.join("");
}

function blank(u, units = 1) {
  return panel(u, units);
}

function cableManager(u) {
  const z = zU(u);
  const out = [panel(u)];
  for (let i = 0; i < 6; i++) {
    const x = 26 + i * 41;
    out.push(box({ x, y: FY, w: 7, d: 12, r: 1.5 }, z + 3, U - 6, { tone: "mid" }, 3, 0.3));
    out.push(box({ x: x - 2, y: FY + 10, w: 11, d: 3, r: 1 }, z + 3, U - 6, { tone: "mid" }, 3, 0.3));
  }
  return out.join("");
}

function switchUnit(u) {
  const z = zU(u);
  const out = [panel(u)];
  out.push(labelPlate(14, z + 5, 26, U - 10));
  out.push(leds([[46, z + 15], [46, z + 7]]));
  const ports = [];
  for (let i = 0; i < 12; i++) for (const row of [0, 1]) {
    const x = 56 + i * 11.2 + Math.floor(i / 6) * 4;
    const zz = z + 3 + row * 8.2;
    ports.push(faceRect(x, zz, 8, 6.8));
    ports.push(faceRect(x + 2.6, zz + (row ? 0 : 5.2), 2.8, 1.6));
  }
  out.push(L(ports.join(""), { tone: "lo" }));
  for (let i = 0; i < 4; i++) {
    const x = 204 + i * 11;
    out.push(L(faceRect(x, z + 4, 9, U - 8), { tone: "lo" }), L(faceRect(x + 1.2, z + 5.6, 6.6, U - 11.2), { tone: "faint" }));
  }
  out.push(leds(Array.from({ length: 12 }, (_, i) => [58 + i * 11.2 + Math.floor(i / 6) * 4, z + 19.6])));
  return out.join("");
}

function shelfFront(u) {
  // the shelf's front lip on the panel line
  const z = zU(u);
  return box({ x: PANEL.x, y: FACE_Y, w: PANEL.w, d: FACE_T, r: 0.6 }, z + 0.4, 5, { tone: "mid" }, 2, 0.4) + screws(z + 0.4, 5);
}

function hostShelf(u) {
  // the tray and the four traffic hosts, drawn with the chassis so the switch above covers them
  const z = zU(u);
  const out = [];
  out.push(box({ x: PANEL.x + 6, y: 40, w: PANEL.w - 12, d: FACE_Y - 40, r: 1 }, z + 0.5, 2.2, { tone: "lo" }, 2));
  for (let i = 0; i < 4; i++) {
    const plan = { x: 22 + i * 56, y: 118, w: 44, d: 72, r: 2.4 };
    out.push(box(plan, z + 2.7, 16, { tone: "hi", crease: "lo" }, 4, 0.6));
    out.push(L(k.planOutline(k.insetPlan(plan, 3), z + 18.7, P, 4), { tone: "faint" }));
    out.push(L(k.sideRing([plan.x + 8, plan.y + plan.d, z + 10.7], 1.6, "y", P, 14), { tone: "lo" }));
    out.push(D([at([plan.x + 8, plan.y + plan.d, z + 10.7]), at([plan.x + 15, plan.y + plan.d, z + 10.7])], { size: 0.5, tone: "mid" }));
    out.push(L(k.pathOf([at([plan.x + 26, plan.y + plan.d, z + 8]), at([plan.x + 38, plan.y + plan.d, z + 8]), at([plan.x + 38, plan.y + plan.d, z + 13.4]), at([plan.x + 26, plan.y + plan.d, z + 13.4])], true), { tone: "faint" }));
  }
  return out.join("");
}

function dcDrawer(u) {
  const z = zU(u);
  const out = [panel(u)];
  out.push(labelPlate(16, z + 5, 22, U - 10));
  for (let i = 0; i < 8; i++) {
    const x = 60 + i * 14;
    out.push(L(faceRect(x - 4, z + 3, 10, U - 6), { tone: "faint" }));
    out.push(box({ x: x - 1.4, y: FY, w: 2.8, d: 3.2, r: 0.6 }, z + 8, 7, { tone: "mid" }, 2));
  }
  out.push(L(faceRect(186, z + 4, 54, U - 8), { tone: "lo" }));
  for (let i = 0; i < 6; i++) out.push(L(k.sideRing([192 + i * 8.6, FY, z + U / 2], 2, "y", P, 14), { tone: "lo" }));
  out.push(leds([[178, z + 15]]));
  return out.join("");
}

function pdu(u) {
  const z = zU(u);
  const out = [panel(u)];
  out.push(L(faceRect(16, z + 5, 30, U - 10), { tone: "lo" }), L(faceRect(18, z + 7, 26, U - 14), { tone: "faint" }));
  for (let i = 0; i < 8; i++) {
    const x = 56 + i * 20;
    out.push(L(faceRect(x, z + 4, 14, U - 8), { tone: "lo" }));
    out.push(L(k.pathOf([fpt(x + 3, z + 9), fpt(x + 5, z + 9), fpt(x + 5, z + 12), fpt(x + 3, z + 12)], true) + k.pathOf([fpt(x + 9, z + 9), fpt(x + 11, z + 9), fpt(x + 11, z + 12), fpt(x + 9, z + 12)], true), { tone: "faint" }));
    out.push(D([fpt(x + 7, z + 15.5)], { size: 0.45, tone: "mid" }));
  }
  out.push(box({ x: 226, y: FY, w: 16, d: 4, r: 1 }, z + 5, U - 10, { tone: "mid" }, 2, 0.3));
  return out.join("");
}

// ---- cables --------------------------------------------------------------------------
const spiderPort = (u, m) => [smaX(m, 0), FY + 4.8, zU(u) + U / 2];
const attPort = (u, side) => [ATT_X(0, side), FY + 4.8, zU(u) + U / 2];
const cplPort = (j, name) => [CPL_X0(j) + CPL_PORT[name], FY + 8, zU(12) + U / 2];
const B1 = (m) => spiderPort(16, m);            // W1: M1 east (LN), M2 south (LW); E2: M3 west (LS), M4 north (LE)
const B2 = (m) => spiderPort(13, m);            // E1: M1 west (LN), M2 south (LE); W2: M3 east (LS), M4 north (LW)
const C = { W1E: 0, E2W: 1, E1W: 2, W2E: 3 };
const CABLES = [
  // LN: W1.E -> coupler -> ATT-1 -> coupler -> E1.W
  [B1(0), cplPort(C.W1E, "in")], [cplPort(C.W1E, "thru"), attPort(11, "in")],
  [attPort(11, "out"), cplPort(C.E1W, "thru")], [cplPort(C.E1W, "in"), B2(0)],
  // LE: E1.S -> ATT-2 -> E2.N
  [B2(1), attPort(10, "in")], [attPort(10, "out"), B1(3)],
  // LS: E2.W -> coupler -> ATT-3 -> coupler -> W2.E
  [B1(2), cplPort(C.E2W, "in")], [cplPort(C.E2W, "thru"), attPort(9, "in")],
  [attPort(9, "out"), cplPort(C.W2E, "thru")], [cplPort(C.W2E, "in"), B2(2)],
  // LW: W2.N -> ATT-4 -> W1.S
  [B2(3), attPort(8, "in")], [attPort(8, "out"), B1(1)],
  // KW: W1.E coupled -> ATT-5 -> W2.E coupled;  KE: E1.W coupled -> ATT-6 -> E2.W coupled
  [cplPort(C.W1E, "cpl"), attPort(7, "in"), true], [attPort(7, "out"), cplPort(C.W2E, "cpl"), true],
  [cplPort(C.E1W, "cpl"), attPort(6, "in"), true], [attPort(6, "out"), cplPort(C.E2W, "cpl"), true],
];

function bezier(p0, p1, p2, p3, n = 28) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, s = 1 - t;
    out.push([0, 1, 2].map((j) => s * s * s * p0[j] + 3 * s * s * t * p1[j] + 3 * s * t * t * p2[j] + t * t * t * p3[j]));
  }
  return out;
}

function cables() {
  const items = [];
  CABLES.forEach(([a, b, lit], i) => {
    const span = Math.abs(a[2] - b[2]) + Math.abs(a[0] - b[0]) * 0.4;
    const reach = 10 + span * 0.22 + (i % 4) * 2.2;
    const route = bezier(a, [a[0], a[1] + reach, a[2]], [b[0], b[1] + reach, b[2]], b);
    for (const piece of tubePieces(route, 1.25, P, { maxLength: 14 })) {
      items.push({ key: k.depthOf(piece.mid, P), svg: tubeSvg(piece, { tone: lit ? "lit" : "mid" }) });
    }
  });
  items.sort((p, q) => p.key - q.key);
  return items.map((it) => it.svg).join("");
}

// ---- rack structure ------------------------------------------------------------------
function rackBack() {
  const out = [`<path class="iso-halo" d="${k.haloOf(RACK, 0, 4, P)}"/>`];
  for (const [x, y] of k.corners(RACK, 18)) {
    out.push(S(k.cylinder(x, y, 7, 0, 2, P, 24), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 4.2, 2, FEET_H - 2, P, 20), { tone: "mid" }));
  }
  out.push(box(RACK, FEET_H, PLINTH_H, { tone: "lo" }, 3, 1));
  out.push(L(k.planOutline(k.insetPlan(RACK, 6), Z0, P, 3), { tone: "faint" }));
  // rear posts and the left side bars
  for (const x of [0, RACK.w - POST]) out.push(box({ x, y: 0, w: POST, d: POST, r: 1 }, Z0, ZTOP - Z0, { tone: "mid" }, 2, 0.4));
  out.push(box({ x: 0, y: POST, w: POST, d: RACK.d - 2 * POST, r: 1 }, ZTOP - 8, 8, { tone: "mid" }, 2, 0.4));
  return out.join("");
}

function bodies() {
  // device chassis behind the panels, bottom to top, so a higher one hides the top below it
  const out = [];
  const units = [1, 2, ...ATTS.map((a) => a.u).sort((a, b) => a - b), 12, 13, 16, 17, 19];
  for (const u of units) {
    const deep = u === 12 ? 30 : u === 17 ? 40 : BODY.d;
    const plan = { x: BODY.x, y: FACE_Y - deep, w: BODY.w, d: deep, r: 1 };
    out.push(box(plan, zU(u) + 0.6, U - 1.2, { tone: u === 13 || u === 16 ? "hi" : "mid", crease: "lo" }, 2, 0.4));
    if (u === 17) out.push(hostShelf(18));
    if (u === 13 || u === 16) {
      const side = [];
      for (let i = 0; i < 9; i++) side.push(sideRect(BODY.x + BODY.w, 50 + i * 14, zU(u) + 5, 9, U - 10));
      out.push(L(side.join(""), { tone: "faint" }));
    }
  }
  return out.join("");
}

function rackFront() {
  const out = [];
  // right side bars, then the two front posts with their rail holes
  out.push(box({ x: RACK.w - POST, y: POST, w: POST, d: RACK.d - 2 * POST, r: 1 }, ZTOP - 8, 8, { tone: "mid" }, 2, 0.4));
  out.push(box({ x: RACK.w - POST, y: POST, w: POST, d: RACK.d - 2 * POST, r: 1 }, Z0, 8, { tone: "mid" }, 2, 0.4));
  for (const x of [0, RACK.w - POST]) {
    out.push(box({ x, y: RACK.d - POST, w: POST, d: POST, r: 1 }, Z0, ZTOP - Z0, { tone: "mid" }, 2, 0.4));
    const holes = [];
    for (let u = 1; u <= UNITS; u++) for (const f of [0.18, 0.5, 0.82]) holes.push(at([x + POST / 2, RACK.d, zU(u) + f * U]));
    out.push(D(holes, { size: 0.42, tone: "lo" }));
  }
  return out.join("");
}

function rackTop() {
  const plan = { x: -2, y: -2, w: RACK.w + 4, d: RACK.d + 4, r: 2 };
  return [
    box(plan, ZTOP, 10, { tone: "mid" }, 3, 1),
    L(k.planOutline(k.insetPlan(plan, 8), ZTOP + 10, P, 3), { tone: "faint" }),
    D(k.dotGrid(60, 60, 12, 6, 12, ZTOP + 10, P), { size: 0.45, tone: "lo" }),
  ].join("");
}

// ---- the table: controllers below, analyser and time server on top ----------------------
function server(plan, z) {
  const out = [`<path class="iso-halo" d="${k.haloOf(plan, z, 1, P)}"/>`];
  out.push(box(plan, z, 90, { tone: "mid", crease: "lo" }, 4, 0.8));
  out.push(L(k.planOutline(k.insetPlan(plan, 4), z + 90, P, 4), { tone: "faint" }));
  const fy = plan.y + plan.d;
  const x = plan.x;
  const slots = [];
  for (let j = 0; j < 7; j++) slots.push(k.pathOf([at([x + 8 + j * 8, fy, z + 55]), at([x + 12 + j * 8, fy, z + 55]), at([x + 12 + j * 8, fy, z + 81]), at([x + 8 + j * 8, fy, z + 81])], true));
  out.push(L(slots.join(""), { tone: "faint" }));
  out.push(L(k.sideRing([x + 14, fy, z + 25], 3, "y", P, 18), { tone: "lo" }), D([at([x + 14, fy, z + 25]), at([x + 26, fy, z + 25])], { size: 0.55, tone: "mid" }));
  out.push(L(k.pathOf([at([x + 36, fy, z + 15]), at([x + 62, fy, z + 15]), at([x + 62, fy, z + 35]), at([x + 36, fy, z + 35])], true), { tone: "lo" }));
  return out.join("");
}

function analyser(plan, z) {
  // bench spectrum analyser: body, screen on the left of the face, keypad, knob and the RF input
  const out = [`<path class="iso-halo" d="${k.haloOf(plan, z, 1, P)}"/>`];
  out.push(box(plan, z, 64, { tone: "hi", crease: "lo" }, 5, 1));
  out.push(L(k.planOutline(k.insetPlan(plan, 4), z + 64, P, 5), { tone: "faint" }));
  const fy = plan.y + plan.d;
  const fr = (x0, z0, w, h) => k.pathOf([at([x0, fy, z0]), at([x0 + w, fy, z0]), at([x0 + w, fy, z0 + h]), at([x0, fy, z0 + h])], true);
  out.push(L(fr(plan.x + 8, z + 14, 76, 42), { tone: "lo" }), L(fr(plan.x + 11, z + 17, 70, 36), { tone: "faint" }));
  // a trace on the screen, rooted on the screen frame at both ends
  const tr = [];
  for (let i = 0; i <= 14; i++) {
    const x = plan.x + 11 + i * 5;
    const h = i === 7 ? 30 : i === 6 || i === 8 ? 16 : 6 + ((i * 7) % 5);
    tr.push(at([x, fy, z + 17 + h]));
  }
  out.push(L(k.pathOf([at([plan.x + 11, fy, z + 23]), ...tr.slice(1, -1), at([plan.x + 81, fy, z + 23])]), { tone: "mid" }));
  const keys = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) keys.push(fr(plan.x + 94 + c * 9, z + 36 + r * 8, 6, 5));
  out.push(L(keys.join(""), { tone: "faint" }));
  out.push(L(k.sideRing([plan.x + 130, fy, z + 44], 7, "y", P, 24), { tone: "lo" }), L(k.sideRing([plan.x + 130, fy, z + 44], 2.6, "y", P, 14), { tone: "faint" }));
  out.push(L(k.sideRing([plan.x + 130, fy, z + 16], 3, "y", P, 18), { tone: "lo" }), D([at([plan.x + 130, fy, z + 16])], { size: 0.5, tone: "hi" }));
  return out.join("");
}

function timeServer(plan, z) {
  const out = [`<path class="iso-halo" d="${k.haloOf(plan, z, 1, P)}"/>`];
  out.push(box(plan, z, 16, { tone: "mid", crease: "lo" }, 4, 0.6));
  const fy = plan.y + plan.d;
  out.push(D([at([plan.x + 10, fy, z + 8]), at([plan.x + 16, fy, z + 8]), at([plan.x + 22, fy, z + 8])], { size: 0.5, tone: "mid" }));
  out.push(L(k.sideRing([plan.x + plan.w - 12, fy, z + 8], 2.4, "y", P, 16), { tone: "lo" }));
  // the GNSS antenna puck on top
  const ax = plan.x + plan.w - 20, ay = plan.y + 20;
  out.push(S(k.cylinder(ax, ay, 9, z + 16, 6, P, 24), { tone: "hi" }));
  return out.join("");
}

function table() {
  const out = [`<path class="iso-halo" d="${k.haloOf(TABLE, 0, 4, P)}"/>`];
  for (const [x, y] of k.corners(TABLE, 14)) out.push(box({ x: x - 5, y: y - 5, w: 10, d: 10, r: 1.5 }, 0, TABLE_Z - TABLE_T, { tone: "mid" }, 2, 0.4));
  out.push(box(k.insetPlan(TABLE, 10), 60, 5, { tone: "lo" }, 3, 0.6));
  // lower shelf: the TG E2E controller and the DPIR controller + predictor (GPU)
  out.push(server({ x: TABLE.x + 40, y: TABLE.y + 40, w: 70, d: 140, r: 3 }, 65));
  out.push(server({ x: TABLE.x + 130, y: TABLE.y + 40, w: 70, d: 140, r: 3 }, 65));
  out.push(box(TABLE, TABLE_Z - TABLE_T, TABLE_T, { tone: "mid" }, 4, 1.2));
  out.push(L(k.planOutline(k.insetPlan(TABLE, 6), TABLE_Z, P, 4), { tone: "faint" }));
  out.push(timeServer({ x: TABLE.x + 30, y: TABLE.y + 24, w: 110, d: 60, r: 3 }, TABLE_Z));
  out.push(analyser({ x: TABLE.x + 150, y: TABLE.y + 70, w: 150, d: 110, r: 4 }, TABLE_Z));
  return out.join("");
}

// ---- assemble ---------------------------------------------------------------------------
function front() {
  const out = [];
  out.push(pdu(1), dcDrawer(2), blank(3, 3));
  for (const a of ATTS.slice().sort((p, q) => p.u - q.u)) out.push(attenuator(a.u, !!a.lit));
  out.push(couplerTray(12));
  out.push(spider(13), blank(14, 2), spider(16));
  out.push(cableManager(17), shelfFront(18), switchUnit(19));
  return out.join("");
}

const BODY_PARTS = () => [rackBack(), bodies(), rackFront(), front(), rackTop(), cables(), table()];
const LABEL = "The PoC 2 conducted bench. A 19-inch rack holds, top to bottom, a switch, a shelf of four traffic hosts, a cable manager, the two GTS003 Spider boxes with the two slots between them empty, a tray of four directional couplers, six 4-channel IF attenuators (four links and the two coupling arms, highlighted), blank panels, a DC drawer and a PDU. Front cables join each sector module to its link attenuator, through a coupler on the four coupled sectors. On the table beside the rack: two controller servers below, a GNSS time server and a spectrum analyser on top.";

const svg = k.figureSvg({ width: WIDTH, height: HEIGHT, label: LABEL, body: BODY_PARTS() });

const plate = k.plateHtml({
  fig: "Fig 2",
  title: "PoC 2 conducted bench",
  hint: "rack front: TG ring over IF cables · table: controllers and test gear",
  readout: "8 sector modules · 4 PtP links · 2 couplings",
  keys: [{ mark: "lit", label: "coupling arms KW, KE" }, { mark: "raised", label: "Spider boxes" }, { mark: "flat", label: "cables" }],
  caption: "One cable is drawn per sector side; each link uses all four tile ports of its sectors, each through its own attenuator channel, and the couplers sit on all four. U14-U15 held PoC 1's hub and S/N boxes.",
  body: `<div class="iso-stage">${svg}</div>`,
});

mkdirSync(OUT, { recursive: true });
const name = theme === "light" ? "bench_poc2" : "bench_poc2-dark";
writeFileSync(join(OUT, `${name}.html`), k.pageHtml({ title: "PoC 2 conducted bench", theme, body: plate, width: 1240 }));
writeFileSync(join(OUT, `${name}.svg`), k.figureSvg({ width: WIDTH, height: HEIGHT, label: "PoC 2 conducted bench", body: BODY_PARTS(), standalone: theme }));
console.log(join(OUT, `${name}.html`));
