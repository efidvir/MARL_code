"""Installation blueprints for the PoC 1 (A7) and PoC 2 benches.

Devices are drawn from their real faces (Spider box modules and SMA tile ports, attenuator channel
pairs, splitter and coupler bodies, switch cages, PDU outlets, rectifier, breaker and fuse drawers,
DC/PoE adaptors, IP-50 radios and their waveguide chains), spread out so that every cable can be
drawn from the port it leaves to the port it reaches, with its ID at both ends.  Seven A3 sheets
per bench:

  1 rack elevation and off-rack equipment
  2 front panels and IF cabling          (every SMA cable, tile by tile)
  3 power: AC, DC and PoE                (wall -> PDU -> loads; DC drawers -> adaptors -> node ETH1)
  4 data and management                  (site VLAN clusters between SW-DATA and SW-OOB)
  5 bench table                          (IP-50 conducted hops, compute, GNSS)
  6 RF ladder and attenuator schedule    (one row per path, settings, one link tile by tile)
  7 exploded parts list

Data: bench_spec.py.  Run:  python3 blueprint.py
  -> out/poc1_blueprint.pdf, out/poc2_blueprint.pdf (+ one PNG per sheet), cable schedules, parts lists
"""
import math
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle, RegularPolygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402

import bench_spec as BS  # noqa: E402

OUT = BS.OUT
PAPER = "#123a6b"
GRID = "#1c4a82"
FACE = "#1b4c88"
PLATE = "#163f73"
INK = "#eaf2ff"
DIM = "#9fb8dc"
OFF = "#ff6b6b"
CLS = {
    "AC": ("#ffb3a1", 1.2, "-", "AC mains"),
    "DC": ("#ffd479", 1.1, "-", "DC 48 V / -48 V"),
    "POE": ("#ffe9a8", 1.0, (0, (4, 1.6)), "Cat6A, adaptor -> node ETH1 (power only)"),
    "FIB": ("#7fe0ff", 0.9, "-", "data (colour = site VLAN)"),
    "ETH": ("#bfe6ff", 0.8, (0, (2, 1.2)), "1G management, VLAN 100"),
    "RF": ("#d2b8ff", 0.7, "-", "IF coax SMA 18 GHz"),
    "GNSS": ("#9cf3b8", 0.9, (0, (1, 1.2)), "GNSS coax (if the boxes have an input)"),
    "WG": ("#ffc0e6", 1.3, "-", "IP-50 chain"),
}
MEDIUM = {"FZ2ERLNLNSNM002": ("-", "OM4 fibre, SR optics"), "AXC763-10000S": ((0, (5, 1.6)), "SFP+ DAC 3 m"),
          "STP6X2MBU": ((0, (1.2, 1.1)), "Cat6A + AGM734"), "STP6X3MBU": ((0, (1.2, 1.1)), "Cat6A + AGM734")}
PALETTE = ["#ff8a50", "#ffe14d", "#7dff6b", "#3ee6ff", "#ff5ccd", "#b58cff", "#49ffc0", "#ffb3e6", "#8fb8ff"]
ARM_COLORS = ["#ff4d4d", "#c9ff4d", "#ffffff"]
TAG_FS = 4.3
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 7, "svg.fonttype": "none"})
MM = 1 / 25.4


# ------------------------------------------------------------------------------------------
# sheet frame
# ------------------------------------------------------------------------------------------
class Sheet:
    def __init__(self, bench, number, total, title, subtitle):
        self.fig = plt.figure(figsize=(420 * MM, 297 * MM))
        self.fig.patch.set_facecolor(PAPER)
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 420)
        ax.set_ylim(0, 297)
        ax.axis("off")
        self.ax = ax
        for x in range(10, 411, 10):
            ax.plot([x, x], [10, 287], color=GRID, lw=0.25, zorder=0)
        for y in range(10, 288, 10):
            ax.plot([10, 410], [y, y], color=GRID, lw=0.25, zorder=0)
        ax.add_patch(Rectangle((6, 6), 408, 285, fill=False, ec=INK, lw=1.4, zorder=1))
        ax.add_patch(Rectangle((10, 10), 400, 277, fill=False, ec=INK, lw=0.6, zorder=1))
        for i, x in enumerate(range(10, 410, 50)):
            for y in (8, 289):
                ax.text(x + 25, y, str(i + 1), ha="center", va="center", color=DIM, fontsize=6)
        for i, y in enumerate(range(287, 20, -46)):
            for x in (8, 412):
                ax.text(x, y - 23, "ABCDEF"[i], ha="center", va="center", color=DIM, fontsize=6)
        ax.add_patch(Rectangle((290, 10), 120, 34, fc=PAPER, ec=INK, lw=0.9, zorder=20))
        for y in (22, 32):
            ax.plot([290, 410], [y, y], color=INK, lw=0.5, zorder=21)
        ax.plot([360, 360], [10, 22], color=INK, lw=0.5, zorder=21)
        ax.plot([385, 385], [10, 22], color=INK, lw=0.5, zorder=21)
        d = bench.d
        ax.text(293, 41, f"UNITY-6G · {d['poc']} conducted bench", color=INK, fontsize=8.5, fontweight="bold",
                va="center", zorder=22)
        ax.text(293, 35.5, title, color=INK, fontsize=8, va="center", zorder=22)
        ax.text(293, 27, subtitle, color=DIM, fontsize=5.6, va="center", zorder=22)
        ax.text(293, 18.5, f"DRAWING  {d['drawing']}", color=DIM, fontsize=5.5, va="center", zorder=22)
        ax.text(293, 13.5, "testbed_design/blueprint.py  ·  not to scale", color=DIM, fontsize=5, va="center", zorder=22)
        ax.text(372.5, 18.5, "SHEET", color=DIM, fontsize=5.5, ha="center", va="center", zorder=22)
        ax.text(372.5, 13.5, f"{number} / {total}", color=INK, fontsize=8, ha="center", va="center", zorder=22)
        ax.text(397.5, 18.5, "REV", color=DIM, fontsize=5.5, ha="center", va="center", zorder=22)
        ax.text(397.5, 13.5, "C", color=INK, fontsize=8, ha="center", va="center", zorder=22)
        ax.text(16, 281, title.upper(), color=INK, fontsize=12, fontweight="bold", va="center", zorder=22)
        self.b = bench

    def text(self, x, y, s, fs=6, color=INK, **kw):
        kw.setdefault("zorder", 12)
        self.ax.text(x, y, s, fontsize=fs, color=color, **kw)

    def label(self, x, y, s, fs=4.4, color=INK, fc=FACE, **kw):
        """Text on a small plate, readable over cables."""
        kw.setdefault("zorder", 13)
        self.ax.text(x, y, s, fontsize=fs, color=color, bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec="none"), **kw)

    def rect(self, x, y, w, h, fc=FACE, ec=INK, lw=0.7, z=3, ls="-", r=0):
        if r:
            self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                             lw=lw, zorder=z, ls=ls))
        else:
            self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))

    def line(self, pts, color=DIM, lw=0.4, z=4, ls="-"):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="round")

    def table(self, x, y, cols, rows, widths, fs=5.0, rh=3.9, wrap=None):
        W = sum(widths)
        n = len(rows) + 1
        self.rect(x, y - n * rh, W, n * rh, fc=PAPER, lw=0.6, z=5)
        cy = y
        for i, r in enumerate([cols] + rows):
            cx = x
            for c, wd in zip(r, widths):
                self.text(cx + 1, cy - rh / 2, str(c), fs=fs, va="center", fontweight="bold" if i == 0 else "normal")
                cx += wd
            cy -= rh
            self.line([(x, cy), (x + W, cy)], color=INK if i == 0 else GRID, lw=0.5 if i == 0 else 0.3, z=6)
        cx = x
        for wd in widths[:-1]:
            cx += wd
            self.line([(cx, y), (cx, y - n * rh)], color=GRID, lw=0.3, z=6)
        return y - n * rh

    def legend(self, x, y, items, title="LEGEND"):
        self.text(x, y, title, fs=6, fontweight="bold")
        for i, (col, lw, dash, lab) in enumerate(items):
            yy = y - 4.4 - i * 3.9
            self.line([(x, yy), (x + 10, yy)], color=col, lw=lw + 0.4, ls=dash, z=12)
            self.text(x + 12, yy, lab, fs=5, va="center")
        return y - 4.4 - len(items) * 3.9

    def notes(self, x, y, lines, title="NOTES", fs=5.0, step=3.4, width=None):
        self.text(x, y, title, fs=6, fontweight="bold")
        yy = y - 4
        for i, s in enumerate(lines):
            parts = textwrap.wrap(f"{i + 1}. {s}", width) if width else [f"{i + 1}. {s}"]
            for j, p in enumerate(parts):
                self.text(x + (3 if j else 0), yy, p, fs=fs, va="top")
                yy -= step
        return yy

    def save(self, pdf, path):
        pdf.savefig(self.fig, facecolor=PAPER)
        self.fig.savefig(path, dpi=200, facecolor=PAPER)
        plt.close(self.fig)

    # cables ----------------------------------------------------------------------------------------
    def cable(self, p0, p1, color, lw=0.8, d0=(0, -1), d1=(0, 1), k=None, ls="-", z=9, alpha=0.95):
        """Smooth cable: leaves p0 in direction d0 and arrives at p1 from direction d1."""
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if k is None:
            if d0[0] == 0 and d1[0] == 0 and d0[1] == -d1[1] != 0:
                k = max(3.0, 0.45 * abs(p1[1] - p0[1]))     # vertical S-bend: never let the handles cross
            else:
                k = max(6.0, 0.42 * L)
        c0 = (p0[0] + d0[0] * k, p0[1] + d0[1] * k)
        c1 = (p1[0] + d1[0] * k, p1[1] + d1[1] * k)
        path = Path([p0, c0, c1, p1], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        self.ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                                    capstyle="round"))

    def tag(self, x, y, s, color, rot=90, fs=TAG_FS, va="center", ha="center"):
        self.ax.text(x, y, s, fontsize=fs, color=PAPER, rotation=rot, ha=ha, va=va, zorder=14,
                     bbox=dict(boxstyle="round,pad=0.12", fc=color, ec="none"))

    def badge(self, x, y, s, fs=3.8, color=OFF, rot=0, va="center", ha="center"):
        self.ax.text(x, y, s, fontsize=fs, color=color, rotation=rot, ha=ha, va=va, zorder=15, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec=color, lw=0.5))


# ------------------------------------------------------------------------------------------
# device faces
# ------------------------------------------------------------------------------------------
def sma(s, x, y, r=1.25, used=True):
    s.ax.add_patch(RegularPolygon((x, y), 6, radius=r * 1.18, orientation=0, fc=PLATE, ec=INK if used else DIM,
                                  lw=0.45, zorder=8))
    s.ax.add_patch(Circle((x, y), r * 0.58, fc=PAPER, ec=INK if used else DIM, lw=0.35, zorder=8))
    s.ax.add_patch(Circle((x, y), r * 0.16, fc=INK if used else DIM, ec="none", zorder=8))


def rj45(s, x, y, w=3.0, h=2.4, used=True):
    s.rect(x - w / 2, y - h / 2, w, h, fc=PAPER, ec=INK if used else DIM, lw=0.4, z=8)
    s.rect(x - w * 0.22, y + h / 2 - 0.55, w * 0.44, 0.55, fc=INK if used else DIM, ec="none", lw=0, z=8)


def sfp(s, x, y, w=3.4, h=1.7, used=True):
    s.rect(x - w / 2, y - h / 2, w, h, fc=PAPER, ec=INK if used else DIM, lw=0.4, z=8)
    s.line([(x - w / 2 + 0.5, y), (x + w / 2 - 0.5, y)], color=INK if used else DIM, lw=0.3, z=8)


def c13(s, x, y, used=True):
    s.rect(x - 2.0, y - 1.5, 4.0, 3.0, fc=PAPER, ec=INK if used else DIM, lw=0.4, z=8, r=0.5)
    for dx in (-0.9, 0.9):
        s.rect(x + dx - 0.22, y - 0.2, 0.45, 0.9, fc=INK if used else DIM, ec="none", lw=0, z=8)
    s.rect(x - 0.22, y - 1.1, 0.45, 0.7, fc=INK if used else DIM, ec="none", lw=0, z=8)


def c14(s, x, y):
    s.rect(x - 2.0, y - 1.5, 4.0, 3.0, fc=PAPER, ec=INK, lw=0.4, z=8)
    for dx in (-0.9, 0.9):
        s.line([(x + dx, y - 0.1), (x + dx, y + 0.8)], color=INK, lw=0.6, z=8)
    s.line([(x, y - 1.0), (x, y - 0.3)], color=INK, lw=0.6, z=8)


def vents(s, x, y, w, h, n):
    for i in range(n):
        xx = x + (i + 0.5) * w / n
        s.line([(xx, y + 1), (xx, y + h - 1)], color=DIM, lw=0.35, z=4)


def rack_face(s, x, y, w, h, label, sub=None, u=None, fs=5.2, bottom=False):
    """1U/2U device face with rack ears; returns the inner x range."""
    s.rect(x, y, w, h, fc=FACE, lw=0.8, z=3)
    for ex in (x, x + w - 3):
        s.rect(ex, y, 3, h, fc=PLATE, lw=0.5, z=4)
        for ey in (y + h * 0.28, y + h * 0.72):
            s.ax.add_patch(Circle((ex + 1.5, ey), 0.55, fc=PAPER, ec=DIM, lw=0.3, zorder=5))
    ly, sy = (y + 5.6, y + 2.6) if bottom else (y + h - 2.6, y + h - 5.4)
    s.text(x + 4.2, ly, label, fs=fs, fontweight="bold", va="center")
    if sub:
        s.text(x + 4.2 + (len(label) * fs * 0.22 + 1.5 if bottom else 0), sy if not bottom else ly, sub, fs=4.0, color=DIM,
               va="center")
    if u:
        s.text(x - 1.2, y + h / 2, u, fs=4, color=DIM, ha="right", va="center")
    return x + 3, x + w - 3


def spider_front(s, b, box, x, y, w=94, h=26):
    """Front of a Spider box: vents, label, four sector-module plates of four SMA tile ports (T labels above)."""
    d = b.d
    rack_face(s, x, y, w, h, box["id"], box["code"])
    vents(s, x + 3.5, y + 1, 7, h - 10, 6)
    s.text(x + 4.2, y + 2.4, ", ".join(box["nodes"]), fs=4.4, color=INK, va="center", fontweight="bold")
    used = {m: sec for sec, (bx, m) in d["module"].items() if bx == box["id"]}
    ports = {}
    x0, pw = x + 13, (w - 16) / 4
    for k in range(4):
        m = f"M{k + 1}"
        px = x0 + k * pw
        s.rect(px + 0.6, y + 1.6, pw - 1.2, h - 3.4, fc=PLATE, lw=0.5, z=5)
        sec = used.get(m)
        s.text(px + 1.6, y + h - 4.2, m, fs=4.8, fontweight="bold", va="center")
        s.text(px + 7.2, y + h - 4.2, f"{sec[0]}.{sec[1]}" if sec else "spare", fs=4.4 if sec else 4.0,
               color=INK if sec else DIM, va="center")
        for t in range(4):
            tx = px + pw * (t + 0.5) / 4
            ty = y + 6.0
            sma(s, tx, ty, used=bool(sec) and t < b.tiles)
            s.text(tx, ty + 3.4, f"T{t + 1}", fs=3.4, color=DIM, ha="center", va="center")
            ports[(m, t + 1)] = (tx, ty)
    return ports


def att_front(s, unit, carries, x, y, w=40, h=22):
    """Attenuator face: four channel plates, IN and OUT SMA each."""
    rack_face(s, x, y, w, h, unit, carries, fs=5.0, bottom=True)
    ports = {}
    gw = (w - 7) / 4
    for c in range(4):
        gx = x + 3.5 + c * gw
        s.rect(gx + 0.25, y + 8.6, gw - 0.5, h - 9.4, fc=PLATE, lw=0.4, z=5)
        s.text(gx + gw / 2, y + 10.2, f"CH{c + 1}", fs=3.5, color=DIM, ha="center", va="center")
        for j, side in enumerate(("in", "out")):
            px = gx + gw * (0.27 + 0.46 * j)
            sma(s, px, y + h - 4.2, r=1.05)
            s.text(px, y + h - 7.4, "I" if side == "in" else "O", fs=3.3, color=DIM, ha="center", va="center")
            ports[(c + 1, side)] = (px, y + h - 4.2)
    return ports


def splitter_body(s, x, y):
    """2-way splitter, S on top, ports 1 and 2 below."""
    s.rect(x - 2.2, y - 2.0, 4.4, 4.0, fc=PLATE, lw=0.5, z=10, r=0.5)
    S = (x, y + 2.0)
    P = {1: (x - 1.2, y - 2.0), 2: (x + 1.2, y - 2.0)}
    for p in [S] + list(P.values()):
        s.ax.add_patch(Circle(p, 0.55, fc=PAPER, ec=INK, lw=0.4, zorder=11))
    return S, P


def coupler_body(s, x, y):
    """Directional coupler, IN on top (module side), THRU below (attenuator side), CPL on the right."""
    s.rect(x - 1.9, y - 2.0, 3.8, 4.0, fc=PLATE, lw=0.5, z=10, r=0.4)
    IN, THRU, CPL = (x, y + 2.0), (x, y - 2.0), (x + 1.9, y - 0.6)
    for p in (IN, THRU, CPL):
        s.ax.add_patch(Circle(p, 0.55, fc=PAPER, ec=INK, lw=0.4, zorder=11))
    return IN, THRU, CPL


def body_key(s, x, y, kind):
    """Enlarged splitter / coupler glyph with its port names (a key for the tray bodies)."""
    k = 3.2
    if kind == "spl":
        s.rect(x - 2.2 * k, y - 2.0 * k, 4.4 * k, 4.0 * k, fc=PLATE, lw=0.6, z=10, r=1.2)
        for (px, py), nm, dy in (((x, y + 2 * k), "S  (from the module tile)", 2.4),
                                 ((x - 1.2 * k, y - 2 * k), "P1", -2.6), ((x + 1.2 * k, y - 2 * k), "P2", -2.6)):
            s.ax.add_patch(Circle((px, py), 1.4, fc=PAPER, ec=INK, lw=0.5, zorder=11))
            s.text(px, py + dy, nm, fs=4.4, ha="center", va="center")
        s.text(x + 9, y, "ZN2PD-02183-S+\nP1 / P2 to the two links'\nattenuators (P1 = first link)", fs=4.2,
               va="center", color=DIM)
    else:
        s.rect(x - 1.9 * k, y - 2.0 * k, 3.8 * k, 4.0 * k, fc=PLATE, lw=0.6, z=10, r=1.0)
        for (px, py), nm, dx, dy in (((x, y + 2 * k), "IN (module)", 0, 2.4), ((x, y - 2 * k), "THRU (link att.)", 0, -2.6),
                                     ((x + 1.9 * k, y - 0.6 * k), "CPL (arm)", 7.5, 0)):
            s.ax.add_patch(Circle((px, py), 1.4, fc=PAPER, ec=INK, lw=0.5, zorder=11))
            s.text(px + dx, py + dy, nm, fs=4.4, ha="center", va="center")
        s.text(x + 17, y + 3.5, "KRYTAR 1821 10 dB", fs=4.2, va="center", color=DIM)


def spider_rear(s, b, box, x, y, w=94, h=22, mode="power"):
    """Rear of a Spider box (layout assumed): per node ETH1 (power in), ETH2, SFP+; a box GNSS input (TBC)."""
    d = b.d
    rack_face(s, x, y, w, h, box["id"] + " rear", box["code"] + " · layout TBC", fs=4.8)
    s.ax.add_patch(Circle((x + 8, y + 9.5), 3.0, fc=PLATE, ec=DIM, lw=0.4, zorder=5))
    for r in (1.0, 1.9, 2.7):
        s.ax.add_patch(Circle((x + 8, y + 9.5), r, fc="none", ec=DIM, lw=0.3, zorder=6))
    ports = {}
    nn = len(box["nodes"])
    cw = (w - 22) / nn
    for i, nd in enumerate(box["nodes"]):
        cx = x + 18 + i * cw
        s.rect(cx + 0.5, y + 1.4, cw - 1, h - 8.4, fc=PLATE, lw=0.5, z=5)
        s.text(cx + 1.6, y + h - 8.4, f"node {nd}", fs=4.5, fontweight="bold", va="center")
        for j, nm in enumerate(("ETH1", "ETH2", "SFP+")):
            px = cx + 5 + j * (cw - 10) / 2
            used = (mode == "power" and nm == "ETH1") or (mode == "data" and nm == "SFP+" and nd not in d.get("no_data", []))
            (sfp if nm == "SFP+" else rj45)(s, px, y + 7.0, used=used)
            s.text(px, y + 3.6, nm, fs=3.5, color=INK if used else DIM, ha="center", va="center")
            ports[(nd, nm)] = (px, y + 7.0)
        if mode == "data" and nd in d.get("no_data", []):
            s.text(cx + cw / 2, y + 10.3, "no wired data (radio only)", fs=3.5, color=OFF, ha="center", va="center")
    gx = x + 14.6
    sma(s, gx, y + 3.4, r=0.85, used=False)
    s.text(gx - 2.0, y + 3.4, "GNSS", fs=3.0, color=DIM, ha="right", va="center")
    ports[("box", "GNSS")] = (gx, y + 4.6)
    return ports


def switch_row(s, label, sub, x, y, w, kinds, h=20, first=None):
    """Switch face drawn with its cages in one row (port numbers left of each cage); returns {port: (x, y)}."""
    rack_face(s, x, y, w, h, label, sub)
    ports = {}
    n = len(kinds)
    x0 = x + (first or 34)
    step = (x + w - 8 - x0) / max(1, n - 1)
    for i, kind in enumerate(kinds):
        p = i + 1
        px = x0 + i * step
        py = y + 7.5
        (sfp if kind == "sfp" else rj45)(s, px, py)
        s.text(px - 2.6, py + 3.4, str(p), fs=4.0, color=INK, ha="center", va="center", fontweight="bold")
        ports[p] = (px, py)
    return ports, step


def pdu_face(s, label, sub, x, y, w=101, h=22):
    rack_face(s, x, y, w, h, label, sub, fs=5.0)
    ports = {}
    for o in range(1, 17):
        g, i = (o - 1) // 8, (o - 1) % 8
        px = x + 7.5 + g * 44.5 + i * 5.3
        py = y + 6.4
        c13(s, px, py)
        s.text(px, py + 3.2, f"{o}", fs=3.4, color=DIM, ha="center", va="center")
        ports[o] = (px, py - 1.5)
    s.rect(x + w - 15.5, y + 12.5, 11, 8.5, fc=PLATE, lw=0.4, z=6)
    s.text(x + w - 10, y + 16.7, "IEC 60309\n32 A in", fs=3.0, ha="center", va="center", color=DIM)
    ports["inlet"] = (x + w - 10, y + 21)
    rj45(s, x + w - 21, y + 16.5)
    s.text(x + w - 21, y + 13.5, "net", fs=3.0, color=DIM, ha="center", va="center")
    ports["net"] = (x + w - 21, y + 16.5)
    return ports


def dist_face(s, x, y, label, sub, n=10, kind="breaker", w=150, h=16, used=()):
    """DC distribution: n breakers (or GMT fuses), output terminal on top of each; bus terminal on the right."""
    rack_face(s, x, y, w, h, label, sub)
    ports = {}
    for i in range(n):
        px = x + 50 + i * 9
        if kind == "breaker":
            s.rect(px - 1.6, y + 2.6, 3.2, 6.4, fc=PLATE, lw=0.4, z=6)
            s.line([(px, y + 5.0), (px, y + 8.0)], color=INK if (i + 1) in used else DIM, lw=0.8, z=7)
        else:
            s.rect(px - 1.4, y + 2.6, 2.8, 6.4, fc=PLATE, lw=0.4, z=6, r=0.6)
            s.rect(px - 0.6, y + 3.6, 1.2, 4.4, fc=INK if (i + 1) in used else DIM, ec="none", lw=0, z=7)
        s.rect(px - 1.0, y + 9.4, 2.0, 1.4, fc=PAPER, ec=INK if (i + 1) in used else DIM, lw=0.4, z=7)
        s.text(px, y + 1.4, f"{'CB' if kind == 'breaker' else 'F'}{i + 1}", fs=3.3, color=DIM, ha="center", va="center")
        ports[i + 1] = (px, y + 10.8)
    s.rect(x + w - 13, y + 3, 8, 6, fc=PLATE, lw=0.4, z=6)
    s.text(x + w - 9, y + 10.6, "bus", fs=3.2, color=DIM, ha="center", va="center")
    ports["bus"] = (x + w - 9, y + 3)
    return ports


def rect_face(s, label, sub, x, y, w=150, h=16, n=3):
    rack_face(s, x, y, w, h, label, sub)
    for i in range(n):
        bx = x + 50 + i * 26
        s.rect(bx, y + 2.0, 23, h - 8.6, fc=PLATE, lw=0.4, z=5)
        vents(s, bx + 2, y + 2.0, 14, h - 8.6, 6)
        s.ax.add_patch(Circle((bx + 19.5, y + 4.6), 0.7, fc=INK if i < 2 else DIM, ec="none", zorder=7))
    c14(s, x + w - 21, y + 4.5)
    s.text(x + w - 21, y + 8.0, "AC", fs=3.2, color=DIM, ha="center", va="center")
    s.rect(x + w - 13, y + h - 6, 8, 4.4, fc=PLATE, lw=0.4, z=6)
    s.text(x + w - 9, y + h - 7.6, "DC out", fs=3.0, color=DIM, ha="center", va="center")
    return {"ac": (x + w - 21, y + 3.0), "out": (x + w - 9, y + h - 1.6)}


def adaptor(s, x, y, name, used=True):
    """Passive DC/PoE adaptor: terminal block (DC in, bottom) and RJ-45 (out, top)."""
    s.rect(x - 4, y - 5, 8, 10, fc=PLATE, ec=INK if used else DIM, lw=0.5, z=6, r=0.8)
    s.text(x, y + 0.4, name, fs=3.4, ha="center", va="center", color=INK if used else DIM)
    rj45(s, x, y + 3.2, w=2.6, h=2.0, used=used)
    for dx in (-1.2, 1.2):
        s.rect(x + dx - 0.9, y - 4.6, 1.8, 1.8, fc=PAPER, ec=INK if used else DIM, lw=0.3, z=8)
        s.ax.add_patch(Circle((x + dx, y - 3.7), 0.35, fc=INK if used else DIM, ec="none", zorder=9))
    return {"rj45": (x, y + 4.2), "dc": (x, y - 4.6)}


def injector(s, x, y, name):
    """AC PoE injector (powers a node that is switched off at t0): AC in at the bottom, PoE out on top."""
    s.rect(x - 6, y - 5, 12, 10, fc=PLATE, ec=OFF, lw=0.6, z=6, r=0.8)
    s.text(x, y + 0.2, name, fs=3.4, ha="center", va="center")
    rj45(s, x, y + 3.2, w=2.6, h=2.0)
    s.text(x, y - 3.2, "AC PoE", fs=2.8, color=DIM, ha="center", va="center")
    return {"poe": (x, y + 4.2), "ac": (x, y - 5)}


def ip50_radio(s, x, y, name, face=+1):
    s.rect(x, y, 22, 14, fc=FACE, lw=0.7, z=5, r=1.6)
    for i in range(6):
        s.line([(x + 3 + i * 3, y + 2), (x + 3 + i * 3, y + 12)], color=DIM, lw=0.3, z=6)
    fx = x + 22 if face > 0 else x - 4
    s.rect(fx, y + 4, 4, 6, fc=PLATE, lw=0.5, z=6)
    s.text(x + 11, y + 16.0, name, fs=4.2, ha="center", va="center", fontweight="bold")
    for px, nm in ((x + 5, "data"), (x + 11, "mgmt"), (x + 17, "DC")):
        s.text(px, y + 1.6, nm, fs=3.0, color=DIM, ha="center", va="center")
    return {"wg": (fx + (4 if face > 0 else 0), y + 7), "data": (x + 5, y), "mgmt": (x + 11, y), "dc": (x + 17, y)}


def transition(s, x, y, face=+1):
    """Waveguide-to-coax transition: flange block toward the radio (face=+1: radio on the left)."""
    if face > 0:
        s.rect(x, y - 2.5, 5, 5, fc=PLATE, lw=0.5, z=6)
        s.rect(x + 5, y - 1.2, 3, 2.4, fc=PLATE, lw=0.5, z=6)
        return {"wg": (x, y), "coax": (x + 8, y)}
    s.rect(x + 3, y - 2.5, 5, 5, fc=PLATE, lw=0.5, z=6)
    s.rect(x, y - 1.2, 3, 2.4, fc=PLATE, lw=0.5, z=6)
    return {"wg": (x + 8, y), "coax": (x, y)}


def pad(s, x, y, label="30 dB"):
    s.rect(x, y - 1.5, 9, 3, fc=PLATE, lw=0.5, z=6, r=1.2)
    s.text(x + 4.5, y + 3.0, label, fs=3.4, color=DIM, ha="center", va="center")
    return {"a": (x, y), "b": (x + 9, y)}


def tile(s, x, y, w, h, label, sub=None, ec=INK, fs=4.6):
    s.rect(x, y, w, h, fc=FACE, ec=ec, lw=0.6, z=4, r=0.8)
    s.text(x + 1.2, y + h - 2.5, label, fs=fs, fontweight="bold", va="center")
    if sub:
        s.text(x + 1.2, y + h - 5.4, sub, fs=3.7, color=DIM, va="center")


def vtile(s, x, y, w, h, label, sub=None, ec=INK):
    """Narrow upright tile with its name rotated: the load at the end of a straight-down lead."""
    s.rect(x - w / 2, y, w, h, fc=FACE, ec=ec, lw=0.6, z=4, r=0.6)
    s.text(x - (0.9 if sub else 0), y + h / 2, label, fs=4.2, rotation=90, ha="center", va="center", fontweight="bold")
    if sub:
        s.text(x + 1.4, y + h / 2, sub, fs=3.2, rotation=90, ha="center", va="center", color=DIM)


# ------------------------------------------------------------------------------------------
# sheet 1: rack elevation
# ------------------------------------------------------------------------------------------
def sheet_rack(b, pdf, n, N, prefix):
    s = Sheet(b, n, N, "Rack elevation and off-rack equipment", b.d["name"][:95])
    size, rows, total = b.size, b.rows, b.total
    h = min(8.6, 230.0 / size)
    x0, y_top = 34, 266
    s.rect(x0 - 6, y_top - size * h - 4, 82, size * h + 8, fc="none", lw=1.0, z=2)
    for u in range(size):
        s.text(x0 - 3, y_top - (u + 0.5) * h, str(size - u), fs=3.8, color=DIM, ha="right", va="center")
    for u in range(total + 1, size + 1):
        y = y_top - (size - u + 1) * h
        s.rect(x0, y + 0.3, 70, h - 0.6, fc=PAPER, ec=GRID, lw=0.4, z=3)
    s.text(x0 + 35, y_top - (size - total) * h / 2, f"U{total + 1}-U{size}: blanking panels", fs=4.4, color=DIM,
           ha="center", va="center")
    for r in rows:
        y = y_top - (size - r["top"] + r["n"]) * h
        hh = r["n"] * h
        sp = r["id"].startswith("SB")
        s.rect(x0, y + 0.3, 70, hh - 0.6, fc="#245a9b" if sp else FACE, lw=0.8 if sp else 0.5, z=3)
        s.text(x0 + 1.5, y + hh / 2, r["label"][:46], fs=4.3, va="center", fontweight="bold" if sp else "normal")
    s.text(x0 - 6, y_top + 5.5, f"FRONT ELEVATION  ·  {size}U rack  ·  {total}U used, U1 up", fs=6, fontweight="bold")
    cols = ["U", "Unit", "Part / notes"]
    trs = [[f"{r['top']}" + (f"-{r['top'] - r['n'] + 1}" if r["n"] > 1 else ""), r["label"], r["sub"]] for r in rows]
    s.text(128, 276, "RACK SCHEDULE", fs=6.5, fontweight="bold")
    yb = s.table(128, 272, cols, trs, [12, 82, 74], fs=4.4, rh=3.7)
    off = [[sv[0], sv[1], f"site {sv[2]} VLAN" if sv[2] else "bench table, mgmt only"] for sv in b.d["servers"]]
    off += [[f"HOST-{h}", "traffic host (iperf3 / TRex)", "sites " + ", ".join(v)] for h, v in b.d["hosts"].items()]
    off += [[k, f"IP-50 radio, hop {hop}", f"site {site}"] for k, (hop, site) in b.ip50.items()]
    off += [[v, "48 V supply, switched (t0)", f"next to {k}"] for k, v in b.psu.items()]
    if b.ip50:
        off.append(["MWATT", "Keysight J7204B, 4 ch", "IP-50 hop attenuator"])
    off += [["TABLE-STRIP", "bench mains strip, C13", f"{len(b.table_loads)} loads"],
            ["SPECAN", "spectrum analyser / power sensor", "bench table"], ["GNSS-ANT", "GNSS antenna (TBC)", "roof"]]
    s.text(302, 276, "OFF-RACK", fs=6.5, fontweight="bold")
    s.table(302, 272, ["ID", "Item", "Where"], off, [24, 50, 30], fs=4.2, rh=3.6)
    s.notes(128, min(yb - 6, 100), [
        "Spider boxes shaded; their modules and ports are drawn on sheet 2 (front) and sheets 3-4 (rear).",
        "Power at the bottom (PDUs, rectifiers, distribution, adaptors), as in figures/bench_a7.mjs.",
        "IF cable lengths follow the U span between each box or tray and its attenuator (sheet 6).",
        f"Keep AC, DC, Ethernet/fibre and SMA bundles apart. Blank the {size - total} empty U (AR8136BLK).",
        "Rack depth, rail type and the Spider boxes' rear panels to confirm before ordering shelves.",
    ] + ([f"Event: {b.d['event']}"] if b.d.get("event") else []), width=78)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 2: front panels and IF cabling
# ------------------------------------------------------------------------------------------
def link_colours(d):
    col = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(d["link_ends"])}
    col.update({name: ARM_COLORS[i % len(ARM_COLORS)] for i, name in enumerate(d["couplings"])})
    return col


def sheet_front(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Front panels and IF cabling", "every SMA cable, tile by tile, ID tagged at both ends")
    sp_ports = {}
    for i, box in enumerate(d["boxes"]):
        for (m, t), xy in spider_front(s, b, box, 16 + i * 98, 240).items():
            sp_ports[(box["id"], m, t)] = xy
    col = link_colours(d)
    T = range(1, b.tiles + 1)
    # tray row (splitters or couplers), each body under its module tile
    tray_y, tray_h = 199, 24
    split_ports, cpl_ports = {}, {}
    if b.split or b.coupled:
        s.rect(16, tray_y, 390, tray_h, fc=FACE, lw=0.7, z=3)
        for ex in (16, 403):
            s.rect(ex, tray_y, 3, tray_h, fc=PLATE, lw=0.5, z=4)
        s.text(22, tray_y + tray_h / 2, "SPL-TRAY" if b.split else "CPL-TRAY", fs=4.6, rotation=90, ha="center",
               va="center", fontweight="bold")
        for sec, sid in list(b.split.items()) + list(b.coupled.items()):
            bx, m = d["module"][sec]
            for t in T:
                mx, _ = sp_ports[(bx, m, t)]
                yb = tray_y + (13.0 if t % 2 else 7.4)
                if sec in b.split:
                    split_ports[(sec, t)] = splitter_body(s, mx, yb)
                else:
                    cpl_ports[(sec, t)] = coupler_body(s, mx, yb)
            x1 = sp_ports[(bx, m, 1)][0]
            s.label(x1 - 2.4, tray_y + 21.0, sid, fs=3.8, va="center")
    # attenuators, one row
    units = list(b.atts.items())
    na = len(units)
    wA = (390 - (na - 1) * 2.5) / na
    att_ports = {}
    ay, ah = 112, 22
    for i, (a, (kind, name)) in enumerate(units):
        x = 16 + i * (wA + 2.5)
        for k, xy in att_front(s, a, f"{'arm ' if kind == 'arm' else ''}{name}", x, ay, w=wA, h=ah).items():
            att_ports[(a, k[0], k[1])] = xy
    top_tag = ay + ah + 0.8

    def at_att(att, t, side, cid, c):
        tgt = att_ports[(att, t, side)]
        s.tag(tgt[0], top_tag, cid, c, va="bottom")
        return (tgt[0], tgt[1] + 1.1)

    box_tag = 238.8
    for name, (sa, sb) in d["link_ends"].items():
        att = b.att_of[name]
        c = col[name]
        for side, sec in (("in", sa), ("out", sb)):
            bx, m = d["module"][sec]
            for t in T:
                cab = b.cable(cls="RF", b=f"{att}.ch{t}.{side}")
                if sec in b.split:
                    p0 = split_ports[(sec, t)][1][d["splitters"][sec].index(name) + 1]
                    s.tag(p0[0], p0[1] - 0.7, cab["id"], c, va="top", fs=3.6)
                elif sec in b.coupled:
                    p0 = cpl_ports[(sec, t)][1]
                    s.tag(p0[0], p0[1] - 0.7, cab["id"], c, va="top", fs=3.6)
                else:
                    p0 = sp_ports[(bx, m, t)]
                    s.tag(p0[0], box_tag, cab["id"], c, va="top")
                s.cable(p0, at_att(att, t, side, cab["id"], c), c, lw=0.6)
    for sec, sid in list(b.split.items()) + list(b.coupled.items()):
        bx, m = d["module"][sec]
        for t in T:
            p = sp_ports[(bx, m, t)]
            if sec in b.split:
                S = split_ports[(sec, t)][0]
                cab = b.cable(cls="RF", b=f"{sid}.t{t}.S")
            else:
                S = cpl_ports[(sec, t)][0]
                cab = b.cable(cls="RF", b=f"{sid}.t{t}.IN")
            s.cable(p, S, INK, lw=0.6)
            s.tag(p[0], box_tag, cab["id"], INK, va="top")
            s.tag(S[0], S[1] + 0.7, cab["id"], INK, va="bottom", fs=3.4)
    for name, (x, y) in d["couplings"].items():
        att = b.att_of[name]
        c = col[name]
        for side, sec in (("in", x), ("out", y)):
            for t in T:
                CPL = cpl_ports[(sec, t)][2]
                stub = (CPL[0] + 1.0, CPL[1])
                cab = b.cable(cls="RF", b=f"{att}.ch{t}.in") if side == "in" else b.cable(cls="RF", a=f"{att}.ch{t}.out")
                s.line([CPL, stub], color=c, lw=0.6, z=9)
                s.cable(stub, at_att(att, t, side, cab["id"], c), c, lw=0.6, d0=(0, -1), ls=(0, (3, 1.2)))
                s.tag(stub[0] + 0.4, stub[1] + 0.8, cab["id"], c, va="bottom", fs=3.4)
    # legend of links and keys
    s.text(16, 100, "LINK COLOURS", fs=6, fontweight="bold")
    for i, (name, c) in enumerate(col.items()):
        x = 16 + (i % 3) * 62
        y = 94 - (i // 3) * 4.6
        s.line([(x, y), (x + 8, y)], color=c, lw=1.8, z=12, ls="-" if name in d["link_ends"] else (0, (3, 1.2)))
        if name in d["link_ends"]:
            (a, sa), (bb, sb) = d["link_ends"][name]
            txt = f"{name}  {a}.{sa} - {bb}.{sb}  {b.att_of[name]}"
        else:
            (a, sa), (bb, sb) = d["couplings"][name]
            txt = f"arm {name}  {a}.{sa} -> {bb}.{sb}  {b.att_of[name]}"
        s.text(x + 10, y, txt, fs=4.6, va="center")
    yj = 94 - (len(col) + 2) // 3 * 4.6
    s.line([(16, yj), (24, yj)], color=INK, lw=1.8, z=12)
    s.text(26, yj, "module tile -> tray (splitter S / coupler IN)", fs=4.6, va="center")
    ky = yj - 20
    if b.split:
        body_key(s, 30, ky, "spl")
    if b.coupled:
        body_key(s, 30 if not b.split else 110, ky, "cpl")
    s.notes(200, 100, [
        "Tile i of a module goes to channel i of its attenuator at both ends of a path; never swap tiles.",
        "Tag both ends of every cable with the ID shown; torque SMA to 8 in-lb with a backing wrench.",
        "Before cabling, measure DC, LO and IF level on one tile port per module; DC blocks if DC is present.",
        "Cable lengths and their U spans: sheet 6 and the cable schedule.",
    ] + (["Splitters: S to the hub module tile; P1 to the first link's attenuator, P2 to the second "
          "(one splitter per tile)."] if b.split else [])
      + (["Couplers: IN on the module, THRU to the link attenuator, CPL to the arm; DC block and pad only at the "
          "arm's input coupler; arms phase-matched."] if b.coupled else [])
      + (["Bridges use each node's second sector module (figures/bench_a7.mjs); only the hub's W and E modules "
          "are split, so all 16 modules are in use."] if b.split else []), width=74, fs=4.9)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 3: power
# ------------------------------------------------------------------------------------------
def sheet_power(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Power: AC, DC and PoE", "wall -> PDU -> loads · DC drawers -> DC/PoE adaptors -> node ETH1")
    rear = {}
    for i, box in enumerate(d["boxes"]):
        for k, v in spider_rear(s, b, box, 16 + i * 98, 244, mode="power").items():
            rear[(box["id"],) + k] = v
    # adaptor shelf: each adaptor under the node it feeds, spares at the right end, unwired
    sy = 200
    s.rect(16, sy, 390, 22, fc=FACE, lw=0.7, z=2)
    s.text(19.5, sy + 19.4, "PA-SHELF", fs=5.2, fontweight="bold", va="center")
    s.text(42, sy + 19.4, "Siklu EH-PoE-DC-Adaptor-10" + (" + AC PoE injector (red: switched off at t0)" if b.inj else "")
           + " on a 1U shelf", fs=4.0, color=DIM, va="center")
    nodes = list(d["nodes"])
    dcn = b.dc_nodes
    ad = {}
    for nd in dcn:
        a = b.adaptor[nd]
        ad[a] = adaptor(s, rear[(d["nodes"][nd]["box"], nd, "ETH1")][0], sy + 8, a)
    for nd, inj in b.inj.items():
        ad[inj] = injector(s, rear[(d["nodes"][nd]["box"], nd, "ETH1")][0], sy + 8, inj)
        ad[inj]["rj45"] = ad[inj]["poe"]
    nsp = b.n_adaptors - len(dcn)
    taken = [v["rj45"][0] for v in ad.values()]
    xs, x = [], 399.0
    while len(xs) < nsp:                                  # spares from the right end, clear of the wired ones
        if all(abs(x - t) > 10 for t in taken):
            xs.append(x)
        x -= 9.5
    for j, x in enumerate(sorted(xs)):
        a = f"PA-{len(dcn) + j + 1:02d}"
        ad[a] = adaptor(s, x, sy + 8, a, used=False)
    if nsp:
        s.text(min(xs) - 4, sy + 15.6, "spare, unwired", fs=3.6, color=DIM, va="center")
    # PoE cables, tagged at both ends
    for nd in nodes:
        a = b.inj.get(nd) or b.adaptor[nd]
        p0 = ad[a]["rj45"]
        p1 = rear[(d["nodes"][nd]["box"], nd, "ETH1")]
        cab = b.cable(cls="POE", a=f"{a}.poe" if nd in b.inj else f"{a}.rj45")
        s.line([p0, (p1[0], p1[1] - 1.2)], color=CLS["POE"][0], lw=1.0, z=9, ls=CLS["POE"][2])
        s.tag(p0[0], sy + 23.2, cab["id"], CLS["POE"][0], va="bottom")
        s.tag(p1[0], 242.5, cab["id"], CLS["POE"][0], va="top")
    # DC drawers, left column
    used = [i + 1 for i in range(len(dcn))]
    dist = dist_face(s, 16, 166, "DC-DIST", "Eaton DCD1M · 48 V · 10 breakers", used=used)
    rect = rect_face(s, "DC-RECT", "Mean Well RCP-1UI + 2 x RCP-1000-48 · 48 V", 16, 145)
    for i, nd in enumerate(dcn):
        a = b.adaptor[nd]
        p0, p1 = dist[i + 1], ad[a]["dc"]
        cab = b.cable(cls="DC", b=f"{a}.dc")
        s.cable(p0, p1, CLS["DC"][0], lw=0.9, d0=(0, 1), d1=(0, -1))
        s.tag(p0[0], p0[1] + 0.8, cab["id"], CLS["DC"][0], va="bottom", fs=3.8)
        s.tag(p1[0], p1[1] - 0.8, cab["id"], CLS["DC"][0], va="top", fs=3.8)
    for nd, inj in b.inj.items():
        cab = b.cable(cls="AC", b=f"{inj}.ac")
        p = ad[inj]["ac"]
        s.badge(p[0], p[1] - 3.6, f"AC {cab['id']} from {cab['a'].replace('.', ' ')}\nswitched off at t0", fs=3.4)
    for i in b.spare_breakers:
        s.text(dist[i][0], dist[i][1] + 1.6, "OFF", fs=3.2, color=DIM, ha="center", va="bottom")
    bus = b.cable(cls="DC", a="DC-RECT.bus")
    s.line([rect["out"], dist["bus"]], color=CLS["DC"][0], lw=1.6, z=9)
    s.tag(rect["out"][0] + 3.4, (rect["out"][1] + dist["bus"][1]) / 2, bus["id"], CLS["DC"][0], rot=0, fs=3.8)
    faces = {"DC-RECT": rect}
    if b.ip50:
        fused = list(range(1, len(b.fuse) + 1))
        dist2 = dist_face(s, 16, 112, "DC-DIST2", "Westell N250110-N-0803 · -48 V · GMT fuses", kind="fuse", used=fused)
        rect2 = rect_face(s, "DC-RECT2", "RCP-1UI + RCP-1000-48 · -48 V (+ grounded)", 16, 91, n=1)
        faces["DC-RECT2"] = rect2
        bus2 = b.cable(cls="DC", a="DC-RECT2.out")
        s.line([rect2["out"], dist2["bus"]], color=CLS["DC"][0], lw=1.6, z=9)
        s.tag(rect2["out"][0] + 3.4, (rect2["out"][1] + dist2["bus"][1]) / 2, bus2["id"], CLS["DC"][0], rot=0, fs=3.8)
        for k, i in b.fuse.items():
            p0 = dist2[i]
            cab = b.cable(cls="DC", a=f"DC-DIST2.F{i}")
            s.line([p0, (p0[0], p0[1] + 3.5)], color=CLS["DC"][0], lw=0.9, z=9)
            s.tag(p0[0], p0[1] + 3.8, f"{cab['id']} {k.replace('IP50-', '')}", CLS["DC"][0], va="bottom", fs=3.3)
        s.text(16, 108.5, "-> IP-50 radios on the bench table (sheet 5)" +
               (f"; {', '.join(b.psu)} have their own switched supplies" if b.psu else ""), fs=4.0, color=CLS["DC"][0])
    # PDUs side by side, loads straight under their outlets
    pdu_ports = {}
    for j, p in enumerate(b.pdus):
        pdu_ports[p] = pdu_face(s, p, "APC AP7922B · 32 A · switched", 196 + j * 106, 150)
    vy, vh = 104, 30
    for ld, (pdu, o) in b.outlet.items():
        p0 = pdu_ports[pdu][o]
        cab = b.cable(cls="AC", b=f"{ld}.ac")
        if ld in faces:
            p1 = faces[ld]["ac"]
            p1 = (p1[0] + 2.0, p1[1] + 1.5)
            s.cable(p0, p1, CLS["AC"][0], lw=0.9, d0=(0, -1), d1=(1, 0))
            s.tag(p1[0] + 5.5, p1[1] + 0.6, cab["id"], CLS["AC"][0], rot=0, va="bottom", fs=3.8)
        else:
            off = ld in b.switched
            sub = ("on PA-SHELF" if ld in b.inj.values() else "3 m to table") if off else \
                ("C5 brick" if "C5" in cab["part"] else "C14")
            vtile(s, p0[0], vy, 4.8, vh, ld, sub, ec=OFF if off else INK)
            s.line([p0, (p0[0], vy + vh)], color=CLS["AC"][0], lw=0.9, z=9)
            if off:
                s.badge(p0[0], vy - 2.6, "OFF t0", fs=3.2)
        s.tag(p0[0], p0[1] - 0.6, cab["id"], CLS["AC"][0], va="top", fs=3.8)
    for p in b.pdus:
        cab = b.cable(cls="AC", b=f"{p}.inlet")
        pi = pdu_ports[p]["inlet"]
        s.line([pi, (pi[0], pi[1] + 5)], color=CLS["AC"][0], lw=1.6, z=9)
        s.tag(pi[0], pi[1] + 5.3, cab["id"] + " to wall 32 A", CLS["AC"][0], rot=0, va="bottom", fs=3.8)
    s.legend(300, 96, [CLS[k] for k in ("AC", "DC", "POE")])
    s.notes(16, 84, [
        "N367 power is ETH1 only: 36-57 V DC through the passive adaptor; ETH1 then carries no data.",
        f"{len(dcn)} nodes x 48 W (PoE-Out off) = {48 * len(dcn)} W on the 48 V drawer; breakers 3-5 A. "
        + (f"PA-{len(dcn) + 1:02d}..PA-{b.n_adaptors:02d} spare and unwired; CB{b.spare_breakers[0]}-{b.spare_breakers[-1]} OFF."
           if b.spare_breakers else ""),
    ] + ([f"Devices that die at t0 lose power at switched PDU outlets (red): {', '.join(b.switched)}. "
          + (f"{', '.join(b.inj)} is powered by an AC PoE injector instead of a DC adaptor so its outlet can cut it. "
             if b.inj else "")
          + "Links between surviving devices are dropped with attenuators instead."] if b.switched else []) + [
        "Check adaptor terminal polarity with a DMM before power-up; keep node PoE-Out (ETH2) disabled.",
    ] + (["IP-50 radios: -48 V (positive grounded) from DC-RECT2 through one GMT fuse each on DC-DIST2."] if b.ip50 else [])
      + ["Spider rear layout (ETH1/ETH2/SFP+ access, GNSS input, any box inlet) is assumed: confirm on the units.",
         "Servers, hosts, the analyser and MWATT take mains from the bench-table strip (sheet 5)"
         + (", except the switched event loads." if b.switched else ".")]
      + ([f"Event: {d['event']}"] if d.get("event") else []), width=118, fs=4.8, step=3.3)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 4: data and management
# ------------------------------------------------------------------------------------------
def site_colours(d):
    return {site: PALETTE[i % len(PALETTE)] for i, site in enumerate(d["sites"])}


def sheet_data(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Data and management", "site VLAN clusters between SW-DATA (above) and SW-OOB (below)")
    sc = site_colours(d)
    sc["trunk"] = INK
    rear = {}
    for i, box in enumerate(d["boxes"]):
        for k, v in spider_rear(s, b, box, 16 + i * 98, 244, mode="data").items():
            rear[(box["id"],) + k] = v
    # SW-DATA in one row
    kinds = ["sfp"] * BS.DATA_PORTS
    sw, step = switch_row(s, "SW-DATA", "M4350-24F4V", 16, 206, 390, kinds)
    for p, (px, py) in sw.items():
        cage = "1/10G" if p <= BS.DATA_1G else ("10G" if p <= 24 else "SFP28")
        if p in (1, BS.DATA_1G + 1, 25):
            s.text(px - 2.6, 206 + 1.6, cage, fs=3.2, color=DIM, va="center")
    for src, (p, v, site, speed) in b.sw_port.items():
        px, py = sw[p]
        s.rect(px + 2.0, py - 1.6, 5.0, 3.2, fc=sc.get(site, INK), ec="none", lw=0, z=9, r=0.4)
        s.text(px + 4.5, py, v.replace("trunk ", "T"), fs=3.3 if not v.startswith("trunk") else 2.6, color=PAPER,
               ha="center", va="center", zorder=10)
    # clusters
    groups = []
    for site in d["sites"] + ["trunk", "table"]:
        items = []
        for k, (hop, st) in b.ip50.items():
            if st == site:
                items.append(dict(name="IP-50", sub=hop, data=f"{k}.data", mgmt=None))
        if site in d.get("agents", []):
            items.append(dict(name=f"AGENT-{site}", sub="MARL agent", data=f"AGENT-{site}.data", mgmt=f"AGENT-{site}.mgmt"))
        for h, sts in d["hosts"].items():
            if (sts == [site]) or (site == "trunk" and len(sts) > 1):
                items.append(dict(name=f"HOST-{h}", sub="sites " + ",".join(sts) if len(sts) > 1 else "traffic host",
                                  data=f"HOST-{h}.nic10g", mgmt=f"HOST-{h}.mgmt"))
        for sid, role, st in d["servers"]:
            if st == site or (site == "table" and st is None):
                items.append(dict(name=sid, sub=role.split(" (")[0][:20], data=f"{sid}.data" if st else None,
                                  mgmt=f"{sid}.mgmt"))
        if items:
            groups.append((site, items))
    ntile = sum(len(g[1]) for g in groups)
    gap = 4.0
    tw = min(19.0, (390 - gap * (len(groups) - 1) - 1.5 * ntile) / ntile)
    x = 16
    ep_top, ep_bot = {}, {}
    cy, ch = 152, 24
    for site, items in groups:
        w = len(items) * (tw + 1.5) - 1.5
        c = sc.get(site, DIM)
        s.rect(x, cy + ch - 5, w, 5, fc=c if site in sc else PLATE, ec="none", lw=0, z=5, r=0.6)
        head = {"trunk": "TRUNKS", "table": "MGMT ONLY"}[site] if site in ("trunk", "table") else \
            f"{site} · {b.vlan(site)}"
        s.text(x + 1.2, cy + ch - 2.5, head, fs=4.4, color=PAPER if site in sc else INK, va="center", fontweight="bold",
               zorder=6)
        for j, it in enumerate(items):
            tx = x + j * (tw + 1.5)
            tile(s, tx, cy, tw, ch - 6, it["name"], it["sub"], ec=c if site in sc else INK, fs=4.2)
            if it["data"]:
                cab = b.cable(cls="FIB", a=it["data"])
                (sfp if cab["speed"] == "10G" else rj45)(s, tx + tw * 0.74, cy + ch - 6, w=3.0, h=1.8)
                ep_top[it["data"]] = (tx + tw * 0.74, cy + ch - 5.1)
            else:
                s.text(tx + 1.2, cy + 7.0, "no data NIC", fs=3.4, color=DIM)
            if it["mgmt"]:
                rj45(s, tx + tw * 0.5, cy, w=3.0, h=1.8)
                ep_bot[it["mgmt"]] = (tx + tw * 0.5, cy - 0.9)
            elif it["name"] == "IP-50":
                s.text(tx + 1.2, cy + 3.2, "mgmt in-band", fs=3.4, color=DIM)
        x += w + gap
    # SW-OOB in one row (RJ-45 then SFP+ cages)
    okinds = ["rj45"] * BS.OOB_RJ45 + ["sfp"] * BS.OOB_SFP
    so, ostep = switch_row(s, "SW-OOB", "GSM4230PX · VLAN 100", 16, 112, 390, okinds)
    s.text(so[BS.OOB_RJ45 + 1][0] - 2.6, 113.6, "SFP+ (AGM734)", fs=3.2, color=DIM, va="center")
    # lab equipment row under SW-OOB
    lab = [(a, "attenuator", f"{a}.lan") for a in b.atts]
    if b.ip50:
        lab.append(("MWATT", "J7204B", "MWATT.lan"))
    lab += [(p, "PDU", f"{p}.net") for p in b.pdus] + [("SPECAN", "analyser", "SPECAN.lan")]
    lw_ = ostep - 1.2
    ly = 84
    for nm, sub, key in lab:
        px = so[b.oob_port[key][0]][0]
        tx = px - lw_ / 2
        s.rect(tx, ly, lw_, 14, fc=FACE, ec=INK, lw=0.6, z=4, r=0.8)
        s.text(px, ly + 6.2, nm, fs=4.0, rotation=0 if len(nm) <= 6 else 0, ha="center", va="center",
               fontweight="bold")
        s.text(px, ly + 2.6, sub, fs=3.2, color=DIM, ha="center", va="center")
        rj45(s, px, ly + 14, w=3.0, h=1.8)
        ep_top[key] = (px, ly + 14.9)
    # cables
    for c in b.cables:
        if c["cls"] == "FIB":
            p = int(c["b"].split(".P")[1])
            px, py = sw[p]
            colr = sc.get(c["site"], INK)
            ls = MEDIUM[c["pn"]][0]
            if c["a"] in ep_top:
                p0 = ep_top[c["a"]]
                s.cable(p0, (px, py - 0.9), colr, lw=0.8, d0=(0, 1), d1=(0, -1), ls=ls)
                s.tag(p0[0], cy + ch + 0.7, c["id"], colr, va="bottom")
                s.tag(px, py - 1.8, c["id"], colr, va="top")
            else:
                box, nd, port = c["a"].split(".")
                p0 = rear[(box, nd, port)]
                s.cable((p0[0], p0[1] - 0.9), (px, py + 0.9), colr, lw=0.8, d0=(0, -1), d1=(0, 1), ls=ls)
                s.tag(p0[0], 242.6, c["id"], colr, va="top")
                s.tag(px, py + 5.4, c["id"], colr, va="bottom")
        elif c["cls"] == "ETH":
            q = int(c["b"].split(".P")[1])
            px, py = so[q]
            colr = CLS["ETH"][0]
            if c["a"] in ep_bot:
                p0 = ep_bot[c["a"]]
                s.cable(p0, (px, py + 1.2), colr, lw=0.7, d0=(0, -1), d1=(0, 1), ls=CLS["ETH"][2])
                s.tag(p0[0], p0[1] - 0.6, c["id"], colr, va="top")
                s.tag(px, py + 5.4, c["id"], colr, va="bottom")
            else:
                p0 = ep_top[c["a"]]
                s.cable(p0, (px, py - 1.2), colr, lw=0.7, d0=(0, 1), d1=(0, -1), ls=CLS["ETH"][2])
                s.tag(p0[0], p0[1] + 0.6, c["id"], colr, va="bottom")
                s.tag(px, py - 2.0, c["id"], colr, va="top")
    # VLAN table
    def ranges(ps):
        ps = sorted(ps)
        out, st = [], ps[0]
        for a, bb in zip(ps, ps[1:] + [None]):
            if bb != a + 1:
                out.append(f"P{st}" + (f"-{a}" if a != st else ""))
                st = bb
        return ", ".join(out)
    vr = []
    for site in d["sites"]:
        ps = [p for src, (p, v, st, sp) in b.sw_port.items() if st == site]
        mem = [src.split(".")[0] if not src.endswith("SFP+") else f"node {site}" for src, (p, v, st, sp) in b.sw_port.items()
               if st == site]
        tr = [f"HOST-{h}" for h, v in d["hosts"].items() if len(v) > 1 and site in v]
        vr.append([str(b.vlan(site)), site, ranges(ps) if ps else "-", ", ".join(mem + [t + " (tagged)" for t in tr]) or
                   ("hub: radio only" if site in d.get("no_data", []) else "-")])
    trunks = [p for src, (p, v, st, sp) in b.sw_port.items() if st == "trunk"]
    if trunks:
        vr.append(["tagged", "trunks", ranges(trunks), "802.1Q trunk ports of the multi-site traffic hosts"])
    vr.append(["100", "mgmt", f"SW-OOB P1-{len(b.oob_port)}", "lab equipment and PC second NICs; not uplinked"])
    s.text(16, 74, "VLAN PLAN (port-based on SW-DATA; no inter-VLAN routing)", fs=6, fontweight="bold")
    yt = s.table(16, 70, ["VLAN", "Site", "SW-DATA ports", "Members"], vr, [12, 12, 30, 134], fs=4.4, rh=3.7)
    items = [(INK, 0.9, MEDIUM[k][0], MEDIUM[k][1]) for k in ("FZ2ERLNLNSNM002", "AXC763-10000S", "STP6X2MBU")] + \
        [CLS["ETH"]]
    for i, (col_, lw0, dash, lab_) in enumerate(items):
        lx = 16 + i * 47
        s.line([(lx, yt - 5), (lx + 9, yt - 5)], color=col_, lw=lw0 + 0.4, ls=dash, z=12)
        s.text(lx + 11, yt - 5, lab_, fs=4.6, va="center")
    s.text(16, yt - 9.5, "Data cable colour = site VLAN (white = trunk). Every cable is tagged at both ends.", fs=4.6,
           color=DIM, va="center")
    s.notes(212, 74, [
        "Nodes are managed in-band: the E2E controller reaches them over the radio network from its data NIC; "
        "node ETH2 is not cabled" + (" and the hub has no wired port." if d.get("no_data") else "."),
        "IP-50: in-band management on its data port; cable its MGMT port to SW-OOB only once it is shown "
        "not to bridge to the data port.",
        "SW-DATA: 1G only on P1-16 (AGM734 copper SFPs); P25-28 are SFP28, run as 4 x 10G.",
        "Node SFP+: 3.3 V Class-1 SR optic; a DAC in a Siklu cage is not documented.",
    ] + ([f"All {BS.DATA_PORTS} SW-DATA ports are used: no spare."] if len(b.sw_port) == BS.DATA_PORTS else []),
        width=82, fs=4.6, step=3.2)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 5: bench table
# ------------------------------------------------------------------------------------------
def sheet_table(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Bench table: IP-50 hops, compute, GNSS", "conducted microwave chains and the off-rack equipment")
    y = 246
    if b.ip50:
        s.text(16, 270, "IP-50 CONDUCTED HOPS  (the J7204B channels drawn inline; MWATT is one 4-channel unit)",
               fs=6.2, fontweight="bold")
        hops = list(d["mw_links"].items())
        top, bot = y + 18, y - (len(hops) - 1) * 28 - 4
        s.rect(150, bot, 46, top - bot, fc=FACE, lw=0.8, z=3)
        s.text(152, top - 2.6, "MWATT", fs=5.2, fontweight="bold", va="center")
        s.text(152, top - 5.4, "Keysight J7204B opt 002", fs=3.8, color=DIM, va="center")
        for j, (k, (a, bb)) in enumerate(hops):
            yy = y - j * 28
            ra = ip50_radio(s, 16, yy, f"IP50-{k}@{a}", +1)
            ta = transition(s, 44, yy + 7, +1)
            pa = pad(s, 53, yy + 7)
            rb = ip50_radio(s, 310, yy, f"IP50-{k}@{bb}", -1)
            tb = transition(s, 298, yy + 7, -1)
            pb = pad(s, 289, yy + 7)
            s.rect(156, yy + 2.5, 34, 9, fc=PLATE, lw=0.4, z=5)
            s.text(173, yy + 9.6, f"CH{j + 1}", fs=3.6, color=DIM, ha="center", va="center")
            s.text(160, yy + 4.0, "in", fs=3.2, color=DIM, ha="center", va="center")
            s.text(186, yy + 4.0, "out", fs=3.2, color=DIM, ha="center", va="center")
            sma(s, 160, yy + 7, r=1.0)
            sma(s, 186, yy + 7, r=1.0)
            wg = CLS["WG"][0]
            s.line([ra["wg"], ta["wg"]], color=wg, lw=1.8, z=9)
            s.line([ta["coax"], pa["a"]], color=wg, lw=1.0, z=9)
            s.line([tb["wg"], rb["wg"]], color=wg, lw=1.8, z=9)
            s.line([pb["b"], tb["coax"]], color=wg, lw=1.0, z=9)
            s.cable(pa["b"], (158.8, yy + 7), wg, lw=1.0, d0=(1, 0), d1=(-1, 0), k=20)
            s.cable((187.2, yy + 7), pb["a"], wg, lw=1.0, d0=(1, 0), d1=(-1, 0), k=20)
            w1 = b.cable(cls="WG", a=f"IP50-{k}@{a}.wg")
            w2 = b.cable(cls="WG", a=f"WGT-{k}-A.coax")
            w3 = b.cable(cls="WG", b=f"WGT-{k}-B.coax")
            w4 = b.cable(cls="WG", b=f"IP50-{k}@{bb}.wg")
            s.tag(66, yy + 7, w2["id"], wg, rot=0, ha="left")
            s.tag(155, yy + 7, w2["id"], wg, rot=0, ha="right")
            s.tag(191, yy + 7, w3["id"], wg, rot=0, ha="left")
            s.tag(286, yy + 7, w3["id"], wg, rot=0, ha="right")
            s.tag(41, yy + 11.5, w1["id"], wg, rot=0)
            s.tag(305, yy + 11.5, w4["id"], wg, rot=0)
            s.text(66, yy + 14.4, f"hop {k}:  {a}  <->  {bb}", fs=4.8, fontweight="bold")
            if d.get("mw_set", {}).get(k):
                rest, t0 = d["mw_set"][k]
                s.text(173, yy - 1.4, f"rest {rest} dB · t0: {t0}", fs=3.6,
                       color=OFF if ("max" in t0 or "off" in t0) else DIM, ha="center", va="center")
            for r, nm, x0 in ((ra, f"IP50-{k}@{a}", 16), (rb, f"IP50-{k}@{bb}", 310)):
                dc = b.cable(cls="DC", b=f"{nm}.dc")
                da = b.cable(cls="FIB", a=f"{nm}.data")
                s.text(x0, yy - 3.6, f"DC {dc['id']} from {dc['a']} · data {da['id']} to {da['b']} (VLAN {da['vlan']})",
                       fs=3.6, color=DIM)
                if nm in b.psu:
                    ac_ = b.cable(cls="AC", b=f"{b.psu[nm]}.ac")
                    s.badge(x0, yy + 21.2, f"OFF at t0: {b.psu[nm]} on {ac_['a'].replace('.', ' ')} ({ac_['id']})",
                            fs=3.4, ha="left")
        ac = b.cable(cls="AC", b="MWATT.ac")
        lan = b.cable(cls="ETH", a="MWATT.lan")
        bot2 = bot - 14
        s.rect(150, bot2, 46, 14.4, fc=FACE, lw=0.8, z=3)
        c14(s, 160, bot2 + 4.5)
        rj45(s, 176, bot2 + 4.5)
        s.text(160, bot2 + 8.6, "AC in", fs=3.4, color=DIM, ha="center", va="center")
        s.text(176, bot2 + 8.6, "LAN", fs=3.4, color=DIM, ha="center", va="center")
        s.line([(176, bot2 + 3.3), (176, bot2 - 4)], color=CLS["ETH"][0], lw=0.8, z=9, ls=CLS["ETH"][2])
        s.tag(176, bot2 - 4.3, f"{lan['id']} to {lan['b']}", CLS["ETH"][0], rot=0, va="top", fs=3.8)
        y = bot2 - 8
    # compute, test and timing, with the bench mains strip
    y0 = min(y, 150)
    s.text(16, y0, "COMPUTE AND TEST", fs=6.2, fontweight="bold")
    items = [(sv[0], sv[1], "server") for sv in d["servers"]] + \
            [(f"HOST-{h}", "traffic host · " + ",".join(v), "host") for h, v in d["hosts"].items()] + \
            [("SPECAN", "analyser / NRP18S + NF-SF50+", "test")]
    order = [nm for nm, _, _ in items if nm in b.table_loads] + [x for x in b.table_loads if x == "MWATT"]
    strip_y = y0 - 14
    s.rect(16, strip_y, 30 + 9 * len(order), 8, fc=FACE, lw=0.7, z=3, r=1)
    s.text(18, strip_y + 5.6, "TABLE-STRIP", fs=4.6, fontweight="bold", va="center")
    s.text(18, strip_y + 2.4, "lab mains, C13", fs=3.4, color=DIM, va="center")
    outlets = {}
    for i, nm in enumerate(order):
        ox = 46 + i * 9
        c13(s, ox, strip_y + 4)
        outlets[nm] = (ox, strip_y + 2.5)
    if "MWATT" in outlets:
        p0 = (outlets["MWATT"][0], strip_y + 5.5)
        s.cable(p0, (160, bot2 + 3.0), CLS["AC"][0], lw=0.8, d0=(0, 1), d1=(0, -1))
        s.tag(160, bot2 + 2.4, ac["id"], CLS["AC"][0], va="top", fs=3.8)
        s.tag(p0[0], p0[1] + 0.6, ac["id"], CLS["AC"][0], va="bottom", fs=3.8)
    yb = strip_y - 18
    for i, (nm, sub, kind) in enumerate(items):
        x = 16 + i * 26
        if kind == "server":
            s.rect(x + 2, yb - 22, 13, 22, fc=FACE, lw=0.6, z=4, r=0.8)
            for k2 in range(5):
                s.line([(x + 4, yb - 4 - k2 * 3), (x + 13, yb - 4 - k2 * 3)], color=DIM, lw=0.3, z=5)
            inlet = (x + 8.5, yb)
        elif kind == "host":
            s.rect(x + 2, yb - 8, 18, 8, fc=FACE, lw=0.6, z=4, r=0.8)
            s.ax.add_patch(Circle((x + 17, yb - 4), 0.6, fc=INK, ec="none", zorder=6))
            inlet = (x + 11, yb)
        else:
            s.rect(x + 2, yb - 12, 20, 12, fc=FACE, lw=0.6, z=4, r=0.8)
            s.rect(x + 4, yb - 10, 10, 8, fc=PAPER, lw=0.4, z=5)
            inlet = (x + 15, yb)
        s.text(x + 2, yb - 25.5, nm, fs=4.6, fontweight="bold")
        ln = textwrap.wrap(sub, 15)[:2]
        ids = [c["id"] + (" data" if c["cls"] == "FIB" else " mgmt") for c in b.cables
               if c["cls"] in ("FIB", "ETH") and c["a"].startswith(nm + ".")]
        for j, t in enumerate(ln + ids):
            s.text(x + 2, yb - 28.8 - j * 3.0, t, fs=3.6, color=DIM if j < len(ln) else INK)
        cab = b.cable(cls="AC", b=f"{nm}.ac")
        if nm in outlets:
            s.cable(outlets[nm], inlet, CLS["AC"][0], lw=0.8, d0=(0, -1), d1=(0, 1))
            s.tag(inlet[0], inlet[1] + 0.6, cab["id"], CLS["AC"][0], va="bottom", fs=3.8)
        else:
            s.line([inlet, (inlet[0], inlet[1] + 4)], color=CLS["AC"][0], lw=0.9, z=9)
            s.badge(inlet[0] + 2, inlet[1] + 7.5, f"{cab['id']} from {cab['a']}\nswitched, off at t0", fs=3.4)
    # GNSS
    gx, gy = 262, y0
    s.text(gx, gy, "GNSS (only if the Spider boxes take an external input)", fs=5.0, fontweight="bold")
    s.rect(gx, gy - 22, 6, 14, fc=FACE, lw=0.6, z=4, r=2.6)
    s.text(gx + 3, gy - 24.5, "ANT 101898-00 (roof)", fs=3.6, color=DIM, ha="center", va="top")
    g = b.cable(cls="GNSS", a="GNSS-ANT")
    s.rect(gx + 26, gy - 18, 8, 6, fc=PLATE, lw=0.5, z=5)
    s.text(gx + 30, gy - 10.6, "arrestor", fs=3.4, color=DIM, ha="center")
    s.rect(gx + 50, gy - 22, 30, 14, fc=FACE, lw=0.6, z=4)
    s.text(gx + 51.5, gy - 10.4, "GNSS-SPL 1x8", fs=4.2, fontweight="bold", va="center")
    s.line([(gx + 3, gy - 8), (gx + 3, gy - 4), (gx + 30, gy - 4), (gx + 30, gy - 12)], color=CLS["GNSS"][0], lw=1.0, z=9)
    s.line([(gx + 34, gy - 15), (gx + 50, gy - 15)], color=CLS["GNSS"][0], lw=1.0, z=9)
    s.tag(gx + 42, gy - 15, g["id"], CLS["GNSS"][0], rot=0)
    for i, box in enumerate(d["boxes"]):
        ox = gx + 54 + i * 6.5
        sma(s, ox, gy - 19, r=0.9)
        cab = b.cable(cls="GNSS", b=f"{box['id']}.GNSS")
        s.line([(ox, gy - 20), (ox, gy - 30)], color=CLS["GNSS"][0], lw=0.8, z=9, ls=CLS["GNSS"][2])
        s.tag(ox, gy - 30.4, f"{cab['id']} {box['id']}", CLS["GNSS"][0], va="top", fs=3.6)
    s.text(gx + 84, gy - 15, "-> Spider box rear\nGNSS input (sheets\n3-4); N367 GPS\nis internal", fs=3.8,
           color=DIM, va="center")
    s.notes(gx, gy - 48, [
        "IP-50 band, flange (WR-62 / WR-42 / WR-28) and Tx power TBC; >= 110 dB in the chain takes a hop down.",
        "30 dB 10 W pads screw onto both transitions; a 1 m SMA cable joins each pad to its MWATT port.",
        "MWATT is commanded over its LAN from SW-OOB; set each channel before the radios are powered.",
    ] + (["Event: CORE's hops go to maximum attenuation at t0; W1-W2 and E1-E2 stay."] if d.get("event") else []),
        fs=4.6, step=3.1, width=66)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 6: RF ladder and attenuator schedule
# ------------------------------------------------------------------------------------------
def rng(cs):
    return f"{cs[0]['id']}-{cs[-1]['id'][1:]}" if len(cs) > 1 else cs[0]["id"]


def sheet_ladder(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "RF ladder and attenuator schedule",
              "one row per path; each line = 4 tile cables (ID range); settings from the radio model")
    y0, rh = 264, 10.5
    xm1, xs1, xa, xs2, xm2, MW = 16, 83, 112, 189, 206, 40
    RF = CLS["RF"][0]
    rows = list(d["link_ends"].items())
    ys = {name: y0 - i * rh for i, (name, _) in enumerate(rows)}
    drawn = set()

    def module_box(x0, yc, sec, hgt=6.4):
        bx, m = d["module"][sec]
        s.rect(x0, yc - hgt / 2, MW, hgt, fc="#245a9b", lw=0.6, z=4)
        s.text(x0 + 1.2, yc + 1.2, f"{bx}.{m}", fs=4.8, fontweight="bold", va="center")
        s.text(x0 + 1.2, yc - 1.6, f"{sec[0]}.{sec[1]}", fs=4.0, color=DIM, va="center")

    for name, (sa, sb) in rows:
        y = ys[name]
        att = b.att_of[name]
        s.rect(xa, y - 3.6, 50, 7.2, fc=FACE, lw=0.7, z=4)
        s.text(xa + 25, y + 1.3, f"{att} · {name}", fs=4.9, fontweight="bold", ha="center", va="center")
        s.text(xa + 25, y - 1.9, "in ch1-4  ·  out ch1-4", fs=3.7, color=DIM, ha="center", va="center")
        for side, sec in (("in", sa), ("out", sb)):
            x0 = xm1 if side == "in" else xm2
            mid = xs1 if side == "in" else xs2
            xl, xr = (xm1 + MW, xa) if side == "in" else (xa + 50, xm2)
            bx, m = d["module"][sec]
            if sec in b.split:
                sib = d["splitters"][sec]
                ysib = [ys[x] for x in sib]
                yc = sum(ysib) / 2
                sid = b.split[sec]
                k = sib.index(name) + 1
                if sec not in drawn:
                    drawn.add(sec)
                    module_box(x0, yc, sec, hgt=6.4)
                    s.rect(mid - 7, yc - 3.4, 14, 6.8, fc=PLATE, lw=0.6, z=5)
                    s.text(mid, yc + 1.2, sid, fs=3.7, ha="center", va="center")
                    s.text(mid, yc - 1.6, "4 splitters", fs=3.2, color=DIM, ha="center", va="center")
                    c1 = b.bundle(f"{bx}.{m}.T", f"{sid}.")
                    a0, a1 = (xl, mid - 7) if side == "in" else (mid + 7, xr)
                    s.line([(a0, yc), (a1, yc)], color=INK, lw=1.4, z=6)
                    s.text((a0 + a1) / 2, yc + 2.4, rng(c1) + f" · {c1[0]['length']}", fs=3.6, ha="center",
                           color=INK)
                c2 = b.bundle(f"{sid}.", f"{att}.ch")
                ax0, ax1 = (mid + 7, xr) if side == "in" else (xl, mid - 7)
                pts = [(ax0, yc + (0.8 if k == 1 else -0.8)), (ax0 + 4, y), (ax1, y)] if side == "in" else \
                    [(ax0, y), (ax1 - 4, y), (ax1, yc + (0.8 if k == 1 else -0.8))]
                s.line(pts, color=RF, lw=1.5, z=6)
                s.text((ax0 + ax1) / 2 + 2, y + 1.9, f"P{k} " + rng(c2) + f" · {c2[0]['length']}", fs=3.6,
                       ha="center", color=RF)
            elif sec in b.coupled:
                module_box(x0, y, sec)
                cid = b.coupled[sec]
                s.rect(mid - 7, y - 2.6, 14, 5.2, fc=PLATE, lw=0.6, z=5)
                s.text(mid, y, cid.replace("CPL-", "C:"), fs=3.6, ha="center", va="center")
                c1 = b.bundle(f"{bx}.{m}.T", f"{cid}.")
                c2 = b.bundle(f"{cid}.", f"{att}.ch")
                a1, a2 = ((xl, mid - 7), (mid + 7, xr)) if side == "in" else ((mid + 7, xr), (xl, mid - 7))
                s.line([(a1[0], y), (a1[1], y)], color=INK, lw=1.4, z=6)
                s.line([(a2[0], y), (a2[1], y)], color=RF, lw=1.5, z=6)
                s.text(sum(a1) / 2, y + 1.9, rng(c1) + f" · {c1[0]['length']}", fs=3.4, ha="center", color=INK)
                s.text(sum(a2) / 2, y + 1.9, rng(c2) + f" · {c2[0]['length']}", fs=3.4, ha="center", color=RF)
            else:
                module_box(x0, y, sec)
                c1 = b.bundle(f"{bx}.{m}.T", f"{att}.ch")
                s.line([(xl, y), (xr, y)], color=RF, lw=1.5, z=6)
                s.text((xl + xr) / 2, y + 1.9, rng(c1) + f" · {c1[0]['length']}", fs=3.6, ha="center", color=RF)
    y = y0 - len(rows) * rh
    for name, (x, z) in d["couplings"].items():
        att = b.att_of[name]
        s.rect(xa, y - 3.6, 50, 7.2, fc=FACE, lw=0.7, z=4)
        s.text(xa + 25, y + 1.3, f"{att} · arm {name}", fs=4.9, fontweight="bold", ha="center", va="center")
        s.text(xa + 25, y - 1.9, "in ch1-4  ·  out ch1-4", fs=3.7, color=DIM, ha="center", va="center")
        for x0, sec, extra in ((xm1, x, " + DC block + pad"), (xm2, z, "")):
            s.rect(x0, y - 3.2, MW, 6.4, fc=PAPER, lw=0.6, z=4, ls=(0, (2, 1)))
            s.text(x0 + 1.2, y + 1.2, f"{b.coupled[sec]} CPL{extra}", fs=3.9, fontweight="bold", va="center")
            s.text(x0 + 1.2, y - 1.6, f"{sec[0]}.{sec[1]}", fs=4.0, color=DIM, va="center")
        c1 = b.bundle(f"{b.coupled[x]}.", f"{att}.ch")
        c2 = b.bundle(f"{att}.ch", f"{b.coupled[z]}.")
        s.line([(xm1 + MW, y), (xa, y)], color=RF, lw=1.4, z=6, ls=(0, (3, 1.2)))
        s.line([(xa + 50, y), (xm2, y)], color=RF, lw=1.4, z=6, ls=(0, (3, 1.2)))
        s.text((xm1 + MW + xa) / 2, y + 1.9, rng(c1) + f" · {c1[0]['length']}", fs=3.6, ha="center", color=RF)
        s.text((xa + 50 + xm2) / 2, y + 1.9, rng(c2) + f" · {c2[0]['length']}", fs=3.6, ha="center", color=RF)
        y -= rh
    ladder_bottom = y
    # schedule with settings
    st = d.get("att_set", {})
    rows_t = []
    for a, (kind, name) in b.atts.items():
        if kind == "link":
            sa, sb = d["link_ends"][name]
            fa = "{}.{}".format(*d["module"][sa]) + (f" via {b.split[sa]}" if sa in b.split else "") + \
                (f" via {b.coupled[sa]}" if sa in b.coupled else "")
            fb = "{}.{}".format(*d["module"][sb]) + (f" via {b.coupled[sb]}" if sb in b.coupled else "")
            car = f"{name}"
        else:
            x, z = d["couplings"][name]
            fa, fb, car = f"{b.coupled[x]} CPL", f"{b.coupled[z]} CPL", f"arm {name}"
        job, bef, t0, aft, mar = st.get(name, ("TBD", "-", "-", "-", "-"))
        rows_t.append([a, car, fa, fb, job, bef, t0, aft, mar])
    s.text(252, 276, "ATTENUATOR SCHEDULE (dB)", fs=6.5, fontweight="bold")
    yt = s.table(252, 272, ["Unit", "Path", "in ch1-4 from", "out to", "Job", "Before", "t0", "After", "Margin"],
                 rows_t, [12, 14, 34, 17, 15, 11, 9, 10, 32], fs=3.9, rh=3.8)
    nts = [
        "Tile i to channel i at both ends of a path; never swap tiles.",
        f"{b.count('RF')} IF cables, length from the U span + port offset + {BS.BEND_MM} mm: " +
        ", ".join(f"{b.count_pn(pn)} x {pn}" for pn, _ in BS.RF_STOCK if b.count_pn(pn)) + ". The team list has 10.",
    ]
    if st:
        nts += ["Settings are the A7 radio model's (out/option_A7_sectors.md): calibrate each path in margin terms on "
                "the real radios (find the drop-out value, back off by the margin, go 14 dB past it to take a path down).",
                "That model assumed a 20 dB pad on each sector port; this IF chain has none: measure the IF levels "
                "and add BW-S pads only if a port needs them."]
    else:
        nts += ["Settings: to come from poc2.py (the final PoC 2 design)."]
    nts += ["Single-tile operation, if the links allow it, divides the IF cabling and attenuator channels by four."]
    nts += ([f"Event: {d['event']}"] if d.get("event") else [])
    s.notes(252, yt - 5, nts, fs=4.6, step=3.1, width=66)
    # detail: one path, tile by tile
    detail(s, b, min(ladder_bottom - 6, 150))
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


def detail(s, b, ytop):
    """DETAIL A: the first link drawn tile by tile, module -> (splitter / coupler) -> attenuator -> module."""
    d = b.d
    name, (sa, sb) = next(iter(d["link_ends"].items()))
    att = b.att_of[name]
    RF = CLS["RF"][0]
    s.text(16, ytop, f"DETAIL A  ·  {name} tile by tile ({sa[0]}.{sa[1]} -> {att} -> {sb[0]}.{sb[1]})", fs=6,
           fontweight="bold")
    yc = ytop - 30

    def module(x, sec, ylab):
        bx, m = d["module"][sec]
        s.rect(x, yc - 18, 22, 36, fc="#245a9b", lw=0.6, z=4)
        s.text(x + 11, yc + ylab, f"{bx}.{m}", fs=4.6, fontweight="bold", ha="center", va="center")
        s.text(x + 11, yc + ylab - 3, f"{sec[0]}.{sec[1]}", fs=3.8, color=DIM, ha="center", va="center")
        out = {}
        for t in range(1, b.tiles + 1):
            py = yc + 10 - (t - 1) * 7
            sma(s, x + (22 if x < 100 else 0), py, r=1.2)
            s.text(x + (17 if x < 100 else 5), py, f"T{t}", fs=3.8, color=DIM, ha="center", va="center")
            out[t] = (x + (22 if x < 100 else 0), py)
        return out
    left = module(16, sa, 15)
    right = module(206, sb, 15)
    s.rect(110, yc - 18, 40, 36, fc=FACE, lw=0.7, z=4)
    s.text(130, yc + 15, att, fs=4.8, fontweight="bold", ha="center", va="center")
    ins, outs = {}, {}
    for t in range(1, b.tiles + 1):
        py = yc + 10 - (t - 1) * 7
        s.text(130, py, f"CH{t}", fs=3.8, color=DIM, ha="center", va="center")
        sma(s, 112, py, r=1.1)
        sma(s, 148, py, r=1.1)
        ins[t], outs[t] = (112, py), (148, py)
    for side, sec, ports, aport in (("in", sa, left, ins), ("out", sb, right, outs)):
        for t in range(1, b.tiles + 1):
            cab = b.cable(cls="RF", b=f"{att}.ch{t}.{side}")
            p_mod, p_att = ports[t], aport[t]
            if sec in b.split or sec in b.coupled:
                xm = 70 if side == "in" else 180
                body = (xm, p_mod[1])
                s.rect(xm - 3, p_mod[1] - 2.2, 6, 4.4, fc=PLATE, lw=0.5, z=10, r=0.5)
                s.text(xm, p_mod[1], "S" if sec in b.split else "C", fs=3.4, ha="center", va="center", zorder=11)
                jump = b.cable(cls="RF", a=f"{d['module'][sec][0]}.{d['module'][sec][1]}.T{t}")
                a0, a1 = (p_mod, (xm - 3, body[1])) if side == "in" else ((xm + 3, body[1]), p_mod)
                s.line([a0, a1], color=INK, lw=0.8, z=9)
                s.tag((a0[0] + a1[0]) / 2, body[1] + 1.0, f"{jump['id']} {jump['length']}", INK, rot=0, va="bottom", fs=3.6)
                b0, b1 = ((xm + 3, body[1]), p_att) if side == "in" else (p_att, (xm - 3, body[1]))
                s.line([b0, b1], color=RF, lw=0.8, z=9)
                s.tag((b0[0] + b1[0]) / 2, body[1] + 1.0, f"{cab['id']} {cab['length']}", RF, rot=0, va="bottom", fs=3.6)
            else:
                a0, a1 = (p_mod, p_att) if side == "in" else (p_att, p_mod)
                s.line([a0, a1], color=RF, lw=0.8, z=9)
                s.tag((a0[0] + a1[0]) / 2, a0[1] + 1.0, f"{cab['id']} {cab['length']}", RF, rot=0, va="bottom", fs=3.6)
    if sa in b.split:
        other = [x for x in d["splitters"][sa] if x != name][0]
        s.text(70, yc - 22, f"splitter P2 of each tile -> {b.att_of[other]} ({other})", fs=4.0, color=DIM, ha="center")


def sheet_parts(b, P, pdf, n, N, prefix):
    s = Sheet(b, n, N, "Exploded parts list", "the team kit list expanded to part numbers and quantities")
    rows = [[p["line"], p["team"], p["group"], p["item"][:74], p["pn"][:48], p["mfr"][:24], str(p["qty"])[:36],
             p["use"][:66], p["conf"][:42]] for p in P]
    rh = min(4.6, 226 / (len(rows) + 1))
    s.table(16, 272, ["#", "Team", "Group", "Item", "Part number", "Manufacturer", "Qty", "Use", "Status"], rows,
            [6, 8, 16, 80, 62, 30, 40, 94, 54], fs=min(4.4, rh * 0.98), rh=rh)
    s.notes(16, 40, [
        "Team = item on the lab's kit list (1-11); '-' = needed but not on the list.",
        "Status: on hand; confirmed (part number checked); likely; to confirm with the lab or vendor; to buy.",
        "Quantities are counted from the cable schedule and rack layout, which bench_spec.py generates.",
    ], title="KEY", fs=4.8)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


def build(design, prefix):
    b = BS.Bench(design)
    P = BS.parts(b)
    N = 7
    with PdfPages(os.path.join(OUT, f"{prefix}_blueprint.pdf")) as pdf:
        sheet_rack(b, pdf, 1, N, prefix)
        sheet_front(b, pdf, 2, N, prefix)
        sheet_power(b, pdf, 3, N, prefix)
        sheet_data(b, pdf, 4, N, prefix)
        sheet_table(b, pdf, 5, N, prefix)
        sheet_ladder(b, pdf, 6, N, prefix)
        sheet_parts(b, P, pdf, 7, N, prefix)
    BS.write_tables(b, P, prefix)
    return b, P


def main():
    for design, prefix in ((BS.poc1_design(), "poc1"), (BS.bowtie_design(), "poc2")):
        b, P = build(design, prefix)
        print(f"{prefix}: {len(b.cables)} cables, {len(P)} part lines -> out/{prefix}_blueprint.pdf (+ 7 PNG)")


if __name__ == "__main__":
    main()
