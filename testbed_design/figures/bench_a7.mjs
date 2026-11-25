// The A7 conducted bench as an Anatomy-style isometric figure
// (kit vendored from github.com/wheresryan22/anatomy, MIT).
//
// One 19-inch rack, front to the viewer, top to bottom:
//   U14-U11 blanking panels, U10 Netgear switch, U9 agent-PC shelf, U8 cable manager,
//   U7-U4 the four Siklu Spider boxes (16 SMA ports each: 4 sector modules x 4),
//   U3 the tray with the hub's two splitters, U2 the DC drawer, U1 the switched PDU.
// PoC 1 has no programmable attenuators (user, 2026-11-25): each link's front cables run
// tile to tile through a fixed pad stack screwed onto a Spider tile (the hub's W and E links:
// splitter P1 / P2 -> cable -> stack on the B-end tile, so the tray carries only splitters;
// the others: stack on the A-end tile -> cable -> B tile); one cable drawn per sector side;
// the E2-W2 bridge is the lit accent.  On the bench beside the rack, the four IP-50 hops in
// a 2 x 2 grid, the two radios of a hop facing: radio, transition, 30 dB pad, 10 dB pad,
// one SMA cable, 30 dB pad, transition, radio.
//
// World units: 1 unit = 2 mm; x across the rack front, y toward the front, z up.
//
// Run:  node figures/bench_a7.mjs [--dark]   -> out/bench_a7.html (+ out/bench_a7.svg)

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
const UNITS = 14;
const PDU_U = 1, DC_U = 2, TRAY_U = 3, CM_U = 8, SHELF_U = 9, SW_U = 10;
const SB = { E: 4, SN: 5, HUB: 6, W: 7 };           // Spider boxes, bottom to top
const BLANKS = [11, 12, 13, 14];
const ZTOP = Z0 + UNITS * U;        // top of the rails
const RACK = { x: 0, y: 0, w: 260, d: 200, r: 2 };
const POST = 12;
const FACE_Y = 198;                 // back of the front panels
const FACE_T = 3;                   // panel thickness
const FY = FACE_Y + FACE_T;         // front face plane
const PANEL = { x: 4, w: 252 };
const BODY = { x: 20, y: 30, w: 220, d: FACE_Y - 30 };
const zU = (u) => Z0 + (u - 1) * U;  // bottom of unit u (1-based)

// ---- the bench --------------------------------------------------------------------
const TABLE = { x: 300, y: -330, w: 420, d: 240, r: 4 };
const TABLE_Z = 300;                // top of the table (600 mm)
const TABLE_T = 8;

const P = k.fitProjection(
  [...k.boxCorners(RACK, 0, ZTOP + 10), [128, FY + 70, Z0], ...k.boxCorners(TABLE, 0, TABLE_Z + 60)],
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
// Spider boxes: modules[m] = [label, role]; the sector modules sit right of the vents.
const SPIDER_X0 = 104;
const MOD_W = 36;
const smaX = (m, i) => SPIDER_X0 + m * MOD_W + 6 + i * 8.2;
const SPIDERS = [
  { u: SB.W, code: "GTS003 · W1, W2" },
  { u: SB.HUB, code: "GTST002B · HUB" },
  { u: SB.SN, code: "GTST003 · S, N" },
  { u: SB.E, code: "GTS003 · E1, E2" },
];

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

// a fixed pad stack (two SMA pads screwed together) standing out of a port toward the viewer:
// from y0 forward, two segments, the second one shorter (the smaller pad)
const PAD_A = 9, PAD_B = 7;
function padStack(x, y0, zc, { lit = false } = {}) {
  return [
    box({ x: x - 1.9, y: y0, w: 3.8, d: PAD_A, r: 1.7 }, zc - 1.9, 3.8, { tone: lit ? "hi" : "mid", lit }, 3, 0.3),
    L(k.sideRing([x, y0 + PAD_A, zc], 1.9, "y", P, 14), { tone: "lo" }),
    box({ x: x - 1.6, y: y0 + PAD_A, w: 3.2, d: PAD_B, r: 1.4 }, zc - 1.6, 3.2, { tone: lit ? "hi" : "mid", lit }, 3, 0.3),
    L(k.sideRing([x, y0 + PAD_A + PAD_B, zc], 1.6, "y", P, 14), { tone: "lo" }),
  ].join("");
}

function splitters(u) {
  const z = zU(u);
  const zc = z + U / 2;
  const out = [panel(u)];
  out.push(labelPlate(16, z + 5, 22, U - 10));
  for (const x0 of [60, 150]) {
    out.push(box({ x: x0, y: FY, w: 50, d: 6, r: 1 }, z + 3, U - 6, { tone: "mid" }, 2, 0.4));
    for (const dx of [8, 25, 42]) {
      out.push(L(k.sideRing([x0 + dx, FY + 6, zc], 2.1, "y", P, 14), { tone: "faint" }));
    }
  }
  return out.join("");
}
const SPLIT_PORT = (which, port) => [(which === "W" ? 60 : 150) + [8, 25, 42][port], FY + 6];

function splitterPorts(u) {
  // S, P1 and P2 of each splitter group: plain SMA ports (the hub links' pad stacks sit on their B tiles)
  const zc = zU(u) + U / 2;
  const out = [];
  for (const which of ["W", "E"]) for (let p = 0; p < 3; p++) {
    const [x, y] = SPLIT_PORT(which, p);
    out.push(box({ x: x - 1.6, y, w: 3.2, d: 3, r: 0.9 }, zc - 1.6, 3.2, { tone: "mid" }));
    out.push(L(k.sideRing([x, y + 3, zc], 0.9, "y", P, 14), { tone: "lo" }), D([at([x, y + 3, zc])], { size: 0.35, tone: "hi" }));
  }
  return out.join("");
}

function tilePads(u, mods, litMods = []) {
  // pad stacks on the first tile of the modules that carry them (one drawn per sector side)
  const zc = zU(u) + U / 2;
  return mods.map((m) => padStack(smaX(m, 0), FY + 4.8, zc, { lit: litMods.includes(m) })).join("");
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

function agentShelf(u) {
  // the tray and the six agent PCs, drawn with the chassis so the switch above covers them
  const z = zU(u);
  const out = [];
  out.push(box({ x: PANEL.x + 6, y: 40, w: PANEL.w - 12, d: FACE_Y - 40, r: 1 }, z + 0.5, 2.2, { tone: "lo" }, 2));
  for (let i = 0; i < 6; i++) {
    const plan = { x: 18 + i * 38, y: 118, w: 32, d: 72, r: 2.4 };
    out.push(box(plan, z + 2.7, 16, { tone: "hi", crease: "lo" }, 4, 0.6));
    out.push(L(k.planOutline(k.insetPlan(plan, 3), z + 18.7, P, 4), { tone: "faint" }));
    out.push(L(k.sideRing([plan.x + 8, plan.y + plan.d, z + 10.7], 1.6, "y", P, 14), { tone: "lo" }));
    out.push(D([at([plan.x + 8, plan.y + plan.d, z + 10.7]), at([plan.x + 15, plan.y + plan.d, z + 10.7])], { size: 0.5, tone: "mid" }));
    out.push(L(k.pathOf([at([plan.x + 20, plan.y + plan.d, z + 8]), at([plan.x + 28, plan.y + plan.d, z + 8]), at([plan.x + 28, plan.y + plan.d, z + 13.4]), at([plan.x + 20, plan.y + plan.d, z + 13.4])], true), { tone: "faint" }));
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
const padEnd = ([x, y, z]) => [x, y + PAD_A + PAD_B, z];            // the OUT end of a stack on a port
const splitPort = (which, p) => { const [x, y] = SPLIT_PORT(which, p); return [x, y + 3, zU(TRAY_U) + U / 2]; };
const W1 = (m) => spiderPort(SB.W, m);          // W1 modules 0,1 ; W2 modules 2,3
const HUB = (m) => spiderPort(SB.HUB, m);       // W, E, S, N
const SN = (m) => spiderPort(SB.SN, m);         // S modules 0,1 ; N modules 2,3
const E12 = (m) => spiderPort(SB.E, m);         // E1 modules 0,1 ; E2 modules 2,3
// one cable per tile path: splitter P1 / P2 to the pad stack on the B tile (hub W / E links), else from the
// pad stack on the A tile to the B tile
const CABLES = [
  [HUB(0), splitPort("W", 0)], [HUB(1), splitPort("E", 0)],          // hub W / E tiles -> splitter S
  [splitPort("W", 1), padEnd(W1(0))], [splitPort("W", 2), padEnd(W1(2))],   // HUB-W1, HUB-W2
  [splitPort("E", 1), padEnd(E12(0))], [splitPort("E", 2), padEnd(E12(2))], // HUB-E1, HUB-E2
  [padEnd(HUB(2)), SN(0)], [padEnd(HUB(3)), SN(2)],                         // HUB-S, HUB-N
  [padEnd(E12(3)), W1(3), true],                                            // E2-W2 (the lit bridge)
  [padEnd(E12(1)), SN(3)], [padEnd(W1(1)), SN(1)],                          // E1-N, W1-S
];
// the modules that carry a pad stack, per Spider box (A ends, and the hub W / E links' B ends)
const TILE_PADS = { [SB.HUB]: [2, 3], [SB.E]: [0, 1, 2, 3], [SB.W]: [0, 1, 2] };

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
  const units = [PDU_U, DC_U, TRAY_U, SB.E, SB.SN, SB.HUB, SB.W, CM_U, SW_U];
  const isSb = (u) => u >= SB.E && u <= SB.W;
  for (const u of units) {
    const deep = u === TRAY_U || u === CM_U ? 40 : BODY.d;
    const plan = { x: BODY.x, y: FACE_Y - deep, w: BODY.w, d: deep, r: 1 };
    out.push(box(plan, zU(u) + 0.6, U - 1.2, { tone: isSb(u) ? "hi" : "mid", crease: "lo" }, 2, 0.4));
    if (u === CM_U) out.push(agentShelf(SHELF_U));
    if (isSb(u)) {
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

// ---- the IP-50 bench ------------------------------------------------------------------
const HOPS = [
  { y: 4, name: "W1–CORE" }, { y: 60, name: "CORE–E1" }, { y: 116, name: "W1–W2" }, { y: 172, name: "E1–E2" },
];

const RADIO_W = 46;
function ip50(x, y, z, face) {
  // an all-outdoor radio lying on its back: body, cooling fins, the RFU flange facing the chain
  const out = [];
  const plan = { x, y, w: RADIO_W, d: 44, r: 4 };
  out.push(`<path class="iso-halo" d="${k.haloOf(plan, z, 1, P)}"/>`);
  out.push(box(plan, z, 22, { tone: "hi", crease: "lo" }, 5, 1));
  const fins = [];
  for (let i = 0; i < 6; i++) fins.push(k.lineOnTop([x + 8 + i * 6, y + 3], [x + 8 + i * 6, y + plan.d - 3], z + 22, P));
  out.push(L(fins.join(""), { tone: "faint" }));
  out.push(L(k.planOutline(k.insetPlan(plan, 3), z + 22, P, 5), { tone: "faint" }));
  const fx = face > 0 ? x + plan.w : x - 6;
  out.push(box({ x: fx, y: y + 12, w: 6, d: 20, r: 1 }, z + 2, 18, { tone: "mid" }, 2, 0.4));
  out.push(L(k.sideRing([x + 10, y + plan.d, z + 11], 2.6, "y", P, 16), { tone: "lo" }));
  out.push(D([at([x + 10, y + plan.d, z + 11]), at([x + 20, y + plan.d, z + 11])], { size: 0.5, tone: "mid" }));
  return { svg: out.join(""), port: [fx + (face > 0 ? 6 : 0), y + 22, z + 11] };
}

function bench() {
  const out = [`<path class="iso-halo" d="${k.haloOf(TABLE, 0, 4, P)}"/>`];
  const legs = k.corners(TABLE, 14);
  for (const [x, y] of legs) out.push(box({ x: x - 5, y: y - 5, w: 10, d: 10, r: 1.5 }, 0, TABLE_Z - TABLE_T, { tone: "mid" }, 2, 0.4));
  out.push(box(k.insetPlan(TABLE, 10), 60, 5, { tone: "lo" }, 3, 0.6));
  // on the lower shelf: the TG controller and the DPIR server (both lost at t0), and the power meter
  for (const [i, x] of [[0, TABLE.x + 40], [1, TABLE.x + 130]]) {
    const plan = { x, y: TABLE.y + 60, w: 70, d: 150, r: 3 };
    out.push(`<path class="iso-halo" d="${k.haloOf(plan, 65, 1, P)}"/>`);
    out.push(box(plan, 65, 90, { tone: "mid", crease: "lo" }, 4, 0.8));
    out.push(L(k.planOutline(k.insetPlan(plan, 4), 155, P, 4), { tone: "faint" }));
    const fy = plan.y + plan.d;
    const slots = [];
    for (let j = 0; j < 7; j++) slots.push(k.pathOf([at([x + 8 + j * 8, fy, 120]), at([x + 12 + j * 8, fy, 120]), at([x + 12 + j * 8, fy, 146]), at([x + 8 + j * 8, fy, 146])], true));
    out.push(L(slots.join(""), { tone: "faint" }));
    out.push(L(k.sideRing([x + 14, fy, 90], 3, "y", P, 18), { tone: "lo" }), D([at([x + 14, fy, 90]), at([x + 26, fy, 90])], { size: 0.55, tone: "mid" }));
    out.push(L(k.pathOf([at([x + 36, fy, 80]), at([x + 62, fy, 80]), at([x + 62, fy, 100]), at([x + 36, fy, 100])], true), { tone: "lo" }));
    if (i === 0) out.push(D([at([x + 40, fy, 74]), at([x + 46, fy, 74])], { size: 0.5, tone: "mid" }));
  }
  {
    const plan = { x: TABLE.x + 240, y: TABLE.y + 120, w: 90, d: 70, r: 4 };
    out.push(`<path class="iso-halo" d="${k.haloOf(plan, 65, 1, P)}"/>`);
    out.push(box(plan, 65, 34, { tone: "mid", crease: "lo" }, 4, 0.8));
    const fy = plan.y + plan.d;
    out.push(L(k.pathOf([at([plan.x + 8, fy, 74]), at([plan.x + 48, fy, 74]), at([plan.x + 48, fy, 92]), at([plan.x + 8, fy, 92])], true), { tone: "lo" }));
    out.push(L(k.sideRing([plan.x + 66, fy, 83], 6, "y", P, 24), { tone: "lo" }), L(k.sideRing([plan.x + 66, fy, 83], 2.2, "y", P, 14), { tone: "faint" }));
    out.push(D([at([plan.x + 80, fy, 76]), at([plan.x + 80, fy, 90])], { size: 0.5, tone: "mid" }));
  }
  out.push(box(TABLE, TABLE_Z - TABLE_T, TABLE_T, { tone: "mid" }, 4, 1.2));
  out.push(L(k.planOutline(k.insetPlan(TABLE, 6), TABLE_Z, P, 4), { tone: "faint" }));
  const z = TABLE_Z;
  // hop j: row j % 2 (0 = front), column j // 2; A radio on the left, flanges facing across the gap that takes the
  // transitions, the pad stacks and the hop's one SMA cable; back row first so the front row covers it
  const grid = HOPS.map((hop, j) => ({ hop, row: j % 2, col: Math.floor(j / 2) }));
  for (const { row, col } of grid.slice().sort((p, q) => q.row - p.row || p.col - q.col)) {
    const x0 = TABLE.x + 12 + col * 204;
    const y0 = TABLE.y + (row === 0 ? 150 : 40);
    const a = ip50(x0, y0, z, +1);
    const yc = y0 + 22;
    const pad30a = { x: a.port[0] + 4, y: yc - 4, w: 18, d: 8, r: 3.6 };
    const pad10 = { x: pad30a.x + pad30a.w, y: yc - 3, w: 12, d: 6, r: 2.8 };
    const gap = 24;                                   // the one SMA cable (0.30 m, drawn shorter)
    const pad30b = { x: pad10.x + pad10.w + gap, y: yc - 4, w: 18, d: 8, r: 3.6 };
    const b = ip50(pad30b.x + pad30b.w + 4 + 6, y0, z, -1);
    const pieces = [];
    pieces.push(box(pad30a, z, 8, { tone: "mid" }, 4, 0.4), box(pad10, z + 1, 6, { tone: "mid" }, 4, 0.4));
    pieces.push(box(pad30b, z, 8, { tone: "mid" }, 4, 0.4));
    const wire = (p, q) => k.wireSvg(at(p), at(q), { tone: "lo", size: 1.1 });
    pieces.push(wire(a.port, [pad30a.x, yc, z + 4]), wire([pad10.x + pad10.w, yc, z + 4], [pad30b.x, yc, z + 4]));
    pieces.push(wire([pad30b.x + pad30b.w, yc, z + 4], b.port));
    out.push(a.svg, pieces.join(""), b.svg);
  }
  return out.join("");
}

// ---- assemble ---------------------------------------------------------------------------
function front() {
  const out = [];
  out.push(pdu(PDU_U), dcDrawer(DC_U));
  out.push(splitters(TRAY_U), splitterPorts(TRAY_U));
  const lit = { [SB.W]: [[3, 0]], [SB.E]: [[3, 0]] };
  for (const s of SPIDERS.slice().sort((p, q) => p.u - q.u)) {
    out.push(spider(s.u, lit[s.u] || []), tilePads(s.u, TILE_PADS[s.u] || [], s.u === SB.E ? [3] : []));
  }
  out.push(cableManager(CM_U), shelfFront(SHELF_U), switchUnit(SW_U));
  for (const u of BLANKS) out.push(panel(u));
  return out.join("");
}

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label: "The A7 conducted bench. A 19-inch rack holds, top to bottom, blanking panels, a Netgear switch, a shelf of six agent PCs, a cable manager, the four Siklu Spider boxes with sixteen SMA ports each, a tray with the hub's splitters, a DC drawer and a PDU. Each link's front cables run tile to tile through fixed pad stacks screwed onto Spider tiles (on the B-end tiles for the hub's W and E links, after the splitter; on the A-end tiles for the others); the E2 to W2 bridge is highlighted. On the bench beside the rack, the four IP-50 hops, each radio to radio through a 30 dB pad, a 10 dB pad, one SMA cable and a 30 dB pad.",
  body: [rackBack(), bodies(), rackFront(), front(), rackTop(), cables(), bench()],
});

const plate = k.plateHtml({
  fig: "Fig 1",
  title: "A7 conducted bench",
  hint: "rack front: TG over IF cables through fixed pads · bench: IP-50 hops",
  readout: "16 sector modules · 9 TG links · 36 pad stacks · 4 IP-50 hops",
  keys: [{ mark: "lit", label: "E2⇄W2 bridge" }, { mark: "raised", label: "Spider boxes" }, { mark: "flat", label: "cables, pads and chains" }],
  caption: "One cable is drawn per sector side; a link uses all four ports of a sector, each tile path through its own pad stack. PoC 1 has no programmable attenuators. The IP-50 radios and the hop cables are drawn smaller than scale.",
  body: `<div class="iso-stage">${svg}</div>`,
});

mkdirSync(OUT, { recursive: true });
const name = theme === "light" ? "bench_a7" : "bench_a7-dark";
writeFileSync(join(OUT, `${name}.html`), k.pageHtml({ title: "A7 conducted bench", theme, body: plate, width: 1240 }));
writeFileSync(join(OUT, `${name}.svg`), k.figureSvg({ width: WIDTH, height: HEIGHT, label: "A7 conducted bench", body: [rackBack(), bodies(), rackFront(), front(), rackTop(), cables(), bench()], standalone: theme }));
console.log(join(OUT, `${name}.html`));
