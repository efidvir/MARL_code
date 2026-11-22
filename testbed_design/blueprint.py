"""Installation blueprints for the PoC 1 (A7) and PoC 2 benches.

Devices are drawn from their real faces (Spider box modules and SMA tile ports, attenuator channel
pairs, splitter and coupler bodies, switch cages, PDU outlets, rectifier and breaker drawers,
DC/PoE adaptors, IP-50 radios and their waveguide chains), spread out so that every cable can be
drawn from the port it leaves to the port it reaches, with its ID at both ends.  Seven A3 sheets
per bench:

  1 rack elevation and off-rack equipment
  2 front panels and IF cabling          (every SMA cable, tile by tile)
  3 power: AC, DC and PoE                (wall -> PDU -> loads; DC drawer -> adaptors -> node ETH1)
  4 data and management                  (node SFP+ / ETH2, switches, VLANs)
  5 bench table                          (IP-50 conducted hops, servers, hosts, GNSS)
  6 RF ladder and attenuator schedule    (one row per link / coupling arm)
  7 exploded parts list

Data: bench_spec.py.  Run:  python3 blueprint.py
  -> out/poc1_blueprint.pdf, out/poc2_blueprint.pdf (+ one PNG per sheet), cable schedules, parts lists
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle, RegularPolygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402

import bench_spec as BS  # noqa: E402

OUT = BS.OUT
PAPER = "#123a6b"
GRID = "#1c4a82"
FACE = "#1b4c88"
PLATE = "#163f73"
INK = "#eaf2ff"
DIM = "#9fb8dc"
CLS = {
    "AC": ("#ffb3a1", 1.2, "-", "AC mains"),
    "DC": ("#ffd479", 1.1, "-", "DC 48 V / -48 V"),
    "POE": ("#ffe9a8", 1.0, (0, (4, 1.6)), "Cat6A, adaptor -> node ETH1 (power only)"),
    "FIB": ("#7fe0ff", 0.9, "-", "10G fibre (OM4 LC-LC) / SFP+ DAC"),
    "ETH": ("#bfe6ff", 0.8, (0, (2, 1.2)), "1G Ethernet (management)"),
    "RF": ("#d2b8ff", 0.7, "-", "IF coax SMA 18 GHz"),
    "GNSS": ("#9cf3b8", 0.9, (0, (1, 1.2)), "GNSS coax (if the boxes have an input)"),
    "WG": ("#ffc0e6", 1.3, "-", "IP-50 chain"),
}
LINK_COLORS = ["#ff9e80", "#ffd180", "#f4ff81", "#b9f6ca", "#84ffff", "#80d8ff", "#b388ff", "#ff80ab", "#ea80fc",
               "#ccff90", "#a7ffeb", "#ffe57f"]
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
        tb = (290, 10, 120, 34)
        ax.add_patch(Rectangle(tb[:2], tb[2], tb[3], fc=PAPER, ec=INK, lw=0.9, zorder=20))
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
        ax.text(397.5, 13.5, "B", color=INK, fontsize=8, ha="center", va="center", zorder=22)
        ax.text(16, 281, title.upper(), color=INK, fontsize=12, fontweight="bold", va="center", zorder=22)
        self.b = bench

    def text(self, x, y, s, fs=6, color=INK, **kw):
        kw.setdefault("zorder", 12)
        self.ax.text(x, y, s, fontsize=fs, color=color, **kw)

    def rect(self, x, y, w, h, fc=FACE, ec=INK, lw=0.7, z=3, ls="-", r=0):
        if r:
            self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                             lw=lw, zorder=z, ls=ls))
        else:
            self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))

    def line(self, pts, color=DIM, lw=0.4, z=4, ls="-"):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls)

    def table(self, x, y, cols, rows, widths, fs=5.0, rh=3.9):
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

    def legend(self, x, y, items):
        self.text(x, y, "LEGEND", fs=6, fontweight="bold")
        for i, (col, lw, dash, lab) in enumerate(items):
            yy = y - 4.4 - i * 3.8
            self.line([(x, yy), (x + 10, yy)], color=col, lw=lw + 0.4, ls=dash, z=12)
            self.text(x + 12, yy, lab, fs=5, va="center")

    def notes(self, x, y, lines, title="NOTES", fs=4.9, step=3.3):
        self.text(x, y, title, fs=6, fontweight="bold")
        for i, s in enumerate(lines):
            self.text(x, y - 4 - i * step, f"{i + 1}. {s}", fs=fs, va="top")

    def save(self, pdf, path):
        pdf.savefig(self.fig, facecolor=PAPER)
        self.fig.savefig(path, dpi=200, facecolor=PAPER)
        plt.close(self.fig)

    # cables ----------------------------------------------------------------------------------------
    def cable(self, p0, p1, color, lw=0.8, d0=(0, -1), d1=(0, 1), k=None, ls="-", z=9, alpha=0.95):
        """Smooth cable: leaves p0 in direction d0 and arrives at p1 from direction d1."""
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        k = k or max(6.0, 0.42 * L)
        c0 = (p0[0] + d0[0] * k, p0[1] + d0[1] * k)
        c1 = (p1[0] + d1[0] * k, p1[1] + d1[1] * k)
        path = Path([p0, c0, c1, p1], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        self.ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                                    capstyle="round"))

    def tag(self, x, y, s, color, rot=90, fs=3.0, va="center"):
        self.ax.text(x, y, s, fontsize=fs, color=PAPER, rotation=rot, ha="center", va=va, zorder=14,
                     bbox=dict(boxstyle="round,pad=0.12", fc=color, ec="none"))


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
    s.rect(x - 2.2, y - 1.6, 4.4, 3.2, fc=PAPER, ec=INK if used else DIM, lw=0.4, z=8, r=0.6)
    for dx in (-1.0, 1.0):
        s.rect(x + dx - 0.25, y - 0.2, 0.5, 1.0, fc=INK if used else DIM, ec="none", lw=0, z=8)
    s.rect(x - 0.25, y - 1.2, 0.5, 0.8, fc=INK if used else DIM, ec="none", lw=0, z=8)


def c14(s, x, y):
    s.rect(x - 2.0, y - 1.5, 4.0, 3.0, fc=PAPER, ec=INK, lw=0.4, z=8)
    for dx in (-0.9, 0.9):
        s.line([(x + dx, y - 0.1), (x + dx, y + 0.8)], color=INK, lw=0.6, z=8)
    s.line([(x, y - 1.0), (x, y - 0.3)], color=INK, lw=0.6, z=8)


def vents(s, x, y, w, h, n):
    for i in range(n):
        xx = x + (i + 0.5) * w / n
        s.line([(xx, y + 1), (xx, y + h - 1)], color=DIM, lw=0.35, z=4)


def rack_face(s, x, y, w, h, label, sub=None, u=None):
    """1U/2U device face with rack ears; returns the inner x range."""
    s.rect(x, y, w, h, fc=FACE, lw=0.8, z=3)
    for ex in (x, x + w - 3):
        s.rect(ex, y, 3, h, fc=PLATE, lw=0.5, z=4)
        for ey in (y + h * 0.28, y + h * 0.72):
            s.ax.add_patch(Circle((ex + 1.5, ey), 0.55, fc=PAPER, ec=DIM, lw=0.3, zorder=5))
    s.text(x + 4.2, y + h - 2.4, label, fs=5.2, fontweight="bold", va="center")
    if sub:
        s.text(x + 4.2, y + h - 5.2, sub, fs=3.9, color=DIM, va="center")
    if u:
        s.text(x - 1.2, y + h / 2, u, fs=4, color=DIM, ha="right", va="center")
    return x + 3, x + w - 3


def spider_front(s, b, box, x, y, w=94, h=26):
    """Front of a Spider box: vents, label, four sector-module plates of four SMA tile ports."""
    d = b.d
    rack_face(s, x, y, w, h, box["id"], box["code"])
    vents(s, x + 3.5, y + 1, 7, h - 9, 6)
    s.text(x + 4.2, y + 2.2, ", ".join(box["nodes"]), fs=4.2, color=INK, va="center", fontweight="bold")
    mods = sorted([(m, sec) for sec, (bx, m) in d["module"].items() if bx == box["id"]])
    used = {m: sec for m, sec in mods}
    ports, centres = {}, {}
    x0, pw = x + 13, (w - 16) / 4
    for k in range(4):
        m = f"M{k + 1}"
        px = x0 + k * pw
        s.rect(px + 0.6, y + 2, pw - 1.2, h - 4, fc=PLATE, lw=0.5, z=5)
        sec = used.get(m)
        s.text(px + 1.6, y + h - 4.4, m, fs=4.6, fontweight="bold", va="center")
        if sec:
            s.text(px + 6.6, y + h - 4.4, f"{sec[0]}.{sec[1]}", fs=4.2, va="center")
        else:
            s.text(px + 6.6, y + h - 4.4, "spare", fs=4.0, color=DIM, va="center")
        for t in range(4):
            tx = px + pw * (t + 0.5) / 4
            ty = y + 7
            sma(s, tx, ty, used=bool(sec) and t < b.tiles)
            s.text(tx, ty - 3.2, f"T{t + 1}", fs=3.0, color=DIM, ha="center", va="center")
            ports[(m, t + 1)] = (tx, ty)
        centres[m] = (px + pw / 2, y + h)
    return ports, centres


def att_front(s, unit, carries, x, y, w=62, h=19):
    rack_face(s, x, y, w, h, unit, carries)
    s.rect(x + 4.2, y + 3, 9, 5.5, fc=PAPER, lw=0.4, z=6)
    s.text(x + 8.7, y + 5.7, "dB", fs=3.4, color=DIM, ha="center", va="center")
    ports = {}
    gw = (w - 20) / 4
    for c in range(4):
        gx = x + 16 + c * gw
        s.rect(gx + 0.4, y + 1.5, gw - 0.8, h - 6.5, fc=PLATE, lw=0.4, z=5)
        s.text(gx + gw / 2, y + h - 6.0, f"CH{c + 1}", fs=3.4, color=DIM, ha="center", va="center")
        for j, side in enumerate(("in", "out")):
            px = gx + gw * (0.28 + 0.44 * j)
            sma(s, px, y + 6.2, r=1.1)
            s.text(px, y + 3.2, side.upper(), fs=2.7, color=DIM, ha="center", va="center")
            ports[(c + 1, side)] = (px, y + 6.2)
    return ports


def tray_front(s, label, sub, x, y, w, h=15):
    rack_face(s, x, y, w, h, label, sub)


def splitter_body(s, x, y, name):
    """2-way splitter, S on top, ports 1 and 2 below."""
    s.rect(x - 2.2, y - 2.0, 4.4, 4.0, fc=PLATE, lw=0.5, z=6, r=0.5)
    s.text(x, y + 0.1, name, fs=2.4, ha="center", va="center", color=DIM)
    S = (x, y + 2.0)
    P = {1: (x - 1.2, y - 2.0), 2: (x + 1.2, y - 2.0)}
    for p in [S] + list(P.values()):
        s.ax.add_patch(Circle(p, 0.55, fc=PAPER, ec=INK, lw=0.4, zorder=8))
    return S, P


def coupler_body(s, x, y, name):
    """Directional coupler, IN on top (module side), THRU below (attenuator side), CPL on the right."""
    s.rect(x - 1.9, y - 2.0, 3.8, 4.0, fc=PLATE, lw=0.5, z=6, r=0.4)
    s.text(x, y, name, fs=2.3, ha="center", va="center", color=DIM)
    IN, THRU, CPL = (x, y + 2.0), (x, y - 2.0), (x + 1.9, y - 1.0)
    for p in (IN, THRU, CPL):
        s.ax.add_patch(Circle(p, 0.55, fc=PAPER, ec=INK, lw=0.4, zorder=8))
    return IN, THRU, CPL


def spider_rear(s, b, box, x, y, w=94, h=26):
    """Rear of a Spider box (layout assumed): per node ETH1 (power in), ETH2, SFP+, GNSS."""
    d = b.d
    rack_face(s, x, y, w, h, box["id"] + " rear", box["code"] + " · layout to confirm")
    for i in range(3):
        s.ax.add_patch(Circle((x + 9 + i * 0.01, y + 9), 4.2, fc=PLATE, ec=DIM, lw=0.4, zorder=5))
        break
    for r in (1.4, 2.6, 3.6):
        s.ax.add_patch(Circle((x + 9, y + 9), r, fc="none", ec=DIM, lw=0.3, zorder=6))
    ports = {}
    nn = len(box["nodes"])
    cw = (w - 22) / nn
    for i, nd in enumerate(box["nodes"]):
        cx = x + 19 + i * cw
        s.rect(cx + 0.5, y + 2, cw - 1, h - 9, fc=PLATE, lw=0.5, z=5)
        s.text(cx + 1.6, y + h - 9.2, f"node {nd}", fs=4.4, fontweight="bold", va="center")
        items = [("ETH1", rj45), ("ETH2", rj45), ("SFP+", sfp)]
        for j, (nm, fn) in enumerate(items):
            px = cx + 4 + j * (cw - 8) / 2.6
            used = not (nm == "SFP+" and nd in d.get("no_data", []))
            fn(s, px, y + 7.5, used=used)
            s.text(px, y + 4.2, nm, fs=2.8, color=DIM, ha="center", va="center")
            ports[(nd, nm)] = (px, y + 7.5)
    gx = x + w - 5.5
    sma(s, gx, y + h - 5.2, r=1.0, used=False)
    s.text(gx, y + h - 8.4, "GNSS?", fs=2.7, color=DIM, ha="center", va="center")
    ports[("box", "GNSS")] = (gx, y + h - 5.2)
    return ports


def switch_front(s, label, sub, x, y, n_ports, kind, w=200, h=20, extra=0):
    """Switch face: n_ports cages (SFP+ or RJ-45) in two rows; returns {port: (x, y)}."""
    rack_face(s, x, y, w, h, label, sub)
    ports = {}
    cols = math.ceil(n_ports / 2)
    x0 = x + 30
    step = (w - 40) / cols
    for p in range(1, n_ports + 1):
        c = (p - 1) // 2
        r = (p - 1) % 2
        px = x0 + c * step + (2.2 if c >= cols / 2 else 0)
        py = y + h - 8.5 - r * 5.6
        (sfp if kind == "sfp" else rj45)(s, px, py)
        s.text(px, py + (2.2 if r == 0 else -2.2), str(p), fs=2.4, color=DIM, ha="center", va="center")
        ports[p] = (px, py)
    for i in range(extra):
        sfp(s, x + w - 8, y + h - 6 - i * 3.2, used=False)
    return ports


def pdu_face(s, label, sub, x, y, w=190, h=22):
    rack_face(s, x, y, w, h, label, sub)
    ports = {}
    for o in range(1, 17):
        g, i = (o - 1) // 8, (o - 1) % 8
        px = x + 30 + g * 82 + i * 9.6
        py = y + 9
        c13(s, px, py)
        s.text(px, py - 3.4, f"O{o}", fs=2.8, color=DIM, ha="center", va="center")
        ports[o] = (px, py)
    s.rect(x + w - 16, y + 4, 9, 10, fc=PLATE, lw=0.4, z=6)
    s.text(x + w - 11.5, y + 9, "IEC\n60309", fs=2.6, ha="center", va="center", color=DIM)
    ports["inlet"] = (x + w - 11.5, y + 4)
    rj45(s, x + w - 22, y + 9)
    ports["net"] = (x + w - 22, y + 9)
    return ports


def dcdist_face(s, x, y, w=150, h=16):
    rack_face(s, x, y, w, h, "DC-DIST", "Eaton DCD1M · 48 V · 10 breakers")
    ports = {}
    for i in range(10):
        px = x + 42 + i * 10
        s.rect(px - 1.6, y + 3.6, 3.2, 7, fc=PLATE, lw=0.4, z=6)
        s.line([(px, y + 6.6), (px, y + 9.6)], color=INK, lw=0.8, z=7)
        s.text(px, y + 1.8, f"CB{i + 1}", fs=2.6, color=DIM, ha="center", va="center")
        ports[i + 1] = (px, y + 3.6)
    ports["bus"] = (x + 30, y + 8)
    s.rect(x + 27, y + 5, 6, 6, fc=PLATE, lw=0.4, z=6)
    return ports


def rect_face(s, label, sub, x, y, w=150, h=16, n=3):
    rack_face(s, x, y, w, h, label, sub)
    for i in range(n):
        bx = x + 42 + i * 30
        s.rect(bx, y + 2.4, 26, h - 4.8, fc=PLATE, lw=0.4, z=5)
        vents(s, bx + 2, y + 2.4, 16, h - 4.8, 6)
        s.ax.add_patch(Circle((bx + 22, y + h / 2), 0.7, fc=INK if i < 2 else DIM, ec="none", zorder=7))
    c14(s, x + w - 12, y + h / 2)
    return {"ac": (x + w - 12, y + h / 2), "out": (x + 34, y + h / 2)}


def adaptor(s, x, y, name):
    """Passive DC/PoE adaptor: terminal block (DC in, bottom) and RJ-45 (out, top)."""
    s.rect(x - 4, y - 5, 8, 10, fc=PLATE, lw=0.5, z=6, r=0.8)
    s.text(x, y + 0.4, name, fs=2.8, ha="center", va="center")
    rj45(s, x, y + 3.2, w=2.6, h=2.0)
    for dx in (-1.2, 1.2):
        s.rect(x + dx - 0.9, y - 4.6, 1.8, 1.8, fc=PAPER, ec=INK, lw=0.3, z=8)
        s.ax.add_patch(Circle((x + dx, y - 3.7), 0.35, fc=INK, ec="none", zorder=9))
    return {"rj45": (x, y + 4.2), "dc": (x, y - 4.6)}


def ip50_radio(s, x, y, name, face=+1):
    s.rect(x, y, 22, 14, fc=FACE, lw=0.7, z=5, r=1.6)
    for i in range(6):
        s.line([(x + 3 + i * 3, y + 2), (x + 3 + i * 3, y + 12)], color=DIM, lw=0.3, z=6)
    fx = x + 22 if face > 0 else x - 4
    s.rect(fx, y + 4, 4, 6, fc=PLATE, lw=0.5, z=6)
    s.text(x + 11, y - 2.2, name, fs=3.6, ha="center", va="center")
    return {"wg": (fx + (4 if face > 0 else 0), y + 7), "data": (x + 6, y), "mgmt": (x + 11, y), "dc": (x + 16, y)}


def transition(s, x, y):
    s.rect(x, y - 2.5, 5, 5, fc=PLATE, lw=0.5, z=6)
    s.rect(x + 5, y - 1.2, 3, 2.4, fc=PLATE, lw=0.5, z=6)
    return {"wg": (x, y), "coax": (x + 8, y)}


def pad(s, x, y, label="30 dB"):
    s.rect(x, y - 1.5, 9, 3, fc=PLATE, lw=0.5, z=6, r=1.2)
    s.text(x + 4.5, y + 2.8, label, fs=2.8, color=DIM, ha="center", va="center")
    return {"a": (x, y), "b": (x + 9, y)}


def tile(s, x, y, w, h, label, sub=None):
    s.rect(x, y, w, h, fc=FACE, lw=0.6, z=4, r=0.8)
    s.text(x + 1.4, y + h - 2.3, label, fs=4.4, fontweight="bold", va="center")
    if sub:
        s.text(x + 1.4, y + h - 4.9, sub, fs=3.4, color=DIM, va="center")


# ------------------------------------------------------------------------------------------
# sheet 1: rack elevation
# ------------------------------------------------------------------------------------------
def sheet_rack(b, pdf, n, N, prefix):
    s = Sheet(b, n, N, "Rack elevation and off-rack equipment", b.d["name"][:95])
    size, rows, total = BS.rack_layout(b)
    h = min(8.6, 230.0 / size)
    x0, y_top = 34, 266
    s.rect(x0 - 6, y_top - size * h - 4, 82, size * h + 8, fc="none", lw=1.0, z=2)
    for u in range(size):
        s.text(x0 - 3, y_top - (u + 0.5) * h, str(size - u), fs=3.8, color=DIM, ha="right", va="center")
    for r in rows:
        y = y_top - (size - r["top"] + r["n"]) * h
        hh = r["n"] * h
        sp = r["id"].startswith("SB")
        s.rect(x0, y + 0.3, 70, hh - 0.6, fc="#245a9b" if sp else FACE, lw=0.8 if sp else 0.5, z=3)
        s.text(x0 + 1.5, y + hh / 2 + (1.1 if hh > 6 else 0), r["label"][:46], fs=4.3, va="center",
               fontweight="bold" if sp else "normal")
        if hh > 6:
            s.text(x0 + 1.5, y + hh / 2 - 1.9, r["sub"][:52], fs=3.5, color=DIM, va="center")
    s.text(x0 - 6, y_top + 5.5, f"FRONT ELEVATION  ·  {size}U rack  ·  {total}U used", fs=6, fontweight="bold")
    cols = ["U", "Unit", "Part / notes"]
    trs = [[f"{r['top']}" + (f"-{r['top'] - r['n'] + 1}" if r["n"] > 1 else ""), r["label"], r["sub"]] for r in rows]
    s.text(128, 276, "RACK SCHEDULE", fs=6.5, fontweight="bold")
    yb = s.table(128, 272, cols, trs, [12, 82, 72], fs=4.3, rh=3.55)
    off = [[sv[0], sv[1], f"site {sv[2]}" if sv[2] else "bench table"] for sv in b.d["servers"]]
    off += [[f"HOST-{h}", "traffic host (iperf3 / TRex)", "sites " + ", ".join(v)] for h, v in b.d["hosts"].items()]
    off += [[k, f"IP-50 radio, hop {hop}", f"site {site}"] for k, (hop, site) in b.ip50.items()]
    if b.ip50:
        off.append(["MWATT", "Keysight J7204B, 4 ch", "IP-50 hop attenuator"])
    off += [["SPECAN", "spectrum analyser / power sensor", "bench table"], ["GNSS-ANT", "GNSS antenna", "roof"]]
    s.text(300, 276, "OFF-RACK", fs=6.5, fontweight="bold")
    s.table(300, 272, ["ID", "Item", "Where"], off, [26, 52, 28], fs=4.2, rh=3.55)
    s.notes(128, min(yb - 6, 100), [
        "Spider boxes shaded; their modules and ports are drawn on sheet 2 (front) and sheets 3-4 (rear).",
        "Keep AC, DC, Ethernet/fibre and SMA bundles apart; SMA bundles run down the front to the attenuators.",
        "Blank every empty U (APC AR8136BLK); label both ends of every cable with its ID (sheets 2-5).",
        "Rack depth, rail type and the Spider boxes' rear panels to confirm before ordering shelves.",
    ] + ([f"Event: {b.d['event']}"] if b.d.get("event") else []))
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 2: front panels and IF cabling
# ------------------------------------------------------------------------------------------
def sheet_front(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Front panels and IF cabling", "every SMA cable, tile by tile; IDs at both ends")
    # spider boxes, top row
    sp_ports = {}
    for i, box in enumerate(d["boxes"]):
        ports, _ = spider_front(s, b, box, 16 + i * 98, 238)
        for (m, t), xy in ports.items():
            sp_ports[(box["id"], m, t)] = xy
    # link colours
    col = {name: LINK_COLORS[i % len(LINK_COLORS)] for i, name in enumerate(list(d["link_ends"]) + list(d["couplings"]))}
    # tray row (splitters or couplers), placed under their modules
    tray_y = 204
    split_ports, cpl_ports = {}, {}
    if b.split or b.coupled:
        tray_front(s, "SPL-TRAY" if b.split else "CPL-TRAY",
                   "2-way splitters ZN2PD-02183-S+" if b.split else "10 dB couplers KRYTAR 1821 · IN up, THRU down",
                   16, tray_y, 390, 18)
        for sec, sid in list(b.split.items()) + list(b.coupled.items()):
            bx, m = d["module"][sec]
            for t in range(1, b.tiles + 1):
                mx, _ = sp_ports[(bx, m, t)]
                if sec in b.split:
                    S, P = splitter_body(s, mx, tray_y + (10.5 if t % 2 else 5.5), f"{t}")
                    split_ports[(sec, t)] = (S, P)
                else:
                    IN, THRU, CPL = coupler_body(s, mx, tray_y + (10.5 if t % 2 else 5.0), f"{t}")
                    cpl_ports[(sec, t)] = (IN, THRU, CPL)
            s.text(sp_ports[(bx, m, 1)][0] - 3, tray_y + 14.6, sid, fs=3.4, color=INK, va="center")
    # attenuators, two rows
    units = list(b.atts.items())
    per_row = math.ceil(len(units) / 2)
    att_ports = {}
    wA = min(62, (390 - (per_row - 1) * 4) / per_row)
    for i, (a, (kind, name)) in enumerate(units):
        r, c = divmod(i, per_row)
        x = 16 + c * (wA + 4)
        y = 160 - r * 52
        ports = att_front(s, a, f"{kind} {name}", x, y, w=wA)
        for k, xy in ports.items():
            att_ports[(a, k[0], k[1])] = xy
    # cables
    T = range(1, b.tiles + 1)
    for name, (sa, sb) in d["link_ends"].items():
        att = b.att_of[name]
        c = col[name]
        for side, sec in (("in", sa), ("out", sb)):
            bx, m = d["module"][sec]
            for t in T:
                tgt = att_ports[(att, t, side)]
                cab = [x for x in b.cables if x["cls"] == "RF" and x["b"] == f"{att}.ch{t}.{side}"][0]
                if sec in b.split:
                    k = d["splitters"][sec].index(name) + 1
                    p0 = split_ports[(sec, t)][1][k]
                elif sec in b.coupled:
                    p0 = cpl_ports[(sec, t)][1]
                else:
                    p0 = sp_ports[(bx, m, t)]
                s.cable(p0, (tgt[0], tgt[1] + 1.1), c, lw=0.55)
                s.tag(tgt[0], tgt[1] + 4.8, cab["id"], c)
                if sec not in b.split and sec not in b.coupled:
                    s.tag(p0[0], p0[1] - 5.4, cab["id"], c)
                else:
                    s.tag(p0[0], p0[1] - 3.2, cab["id"], c, fs=2.6)
    for sec, sid in b.split.items():
        bx, m = d["module"][sec]
        for t in T:
            S = split_ports[(sec, t)][0]
            p = sp_ports[(bx, m, t)]
            cab = [x for x in b.cables if x["cls"] == "RF" and x["b"] == f"{sid}.t{t}.S"][0]
            s.cable(p, S, INK, lw=0.55)
            s.tag(p[0], p[1] - 5.4, cab["id"], INK)
    for sec, cid in b.coupled.items():
        bx, m = d["module"][sec]
        for t in T:
            IN = cpl_ports[(sec, t)][0]
            p = sp_ports[(bx, m, t)]
            cab = [x for x in b.cables if x["cls"] == "RF" and x["b"] == f"{cid}.t{t}.IN"][0]
            s.cable(p, IN, INK, lw=0.55)
            s.tag(p[0], p[1] - 5.4, cab["id"], INK)
    for name, (x, y) in d["couplings"].items():
        att = b.att_of[name]
        c = col[name]
        for side, sec in (("in", x), ("out", y)):
            for t in T:
                CPL = cpl_ports[(sec, t)][2]
                tgt = att_ports[(att, t, side)]
                s.cable(CPL, (tgt[0], tgt[1] + 1.1), c, lw=0.55, d0=(1, -1), ls=(0, (3, 1.2)))
                cab = [z for z in b.cables if z["cls"] == "RF" and (z["b"] == f"{att}.ch{t}.{side}" or z["a"] == f"{att}.ch{t}.{side}")][0]
                s.tag(tgt[0], tgt[1] + 4.8, cab["id"], c)
    # legend of links
    s.text(16, 84, "LINK COLOURS", fs=6, fontweight="bold")
    for i, (name, c) in enumerate(col.items()):
        x = 16 + (i % 4) * 44
        y = 78 - (i // 4) * 4.2
        s.line([(x, y), (x + 8, y)], color=c, lw=1.6, z=12, ls="-" if name in d["link_ends"] else (0, (3, 1.2)))
        if name in d["link_ends"]:
            (a, sa), (bb, sb) = d["link_ends"][name]
            txt = f"{name}  {a}.{sa}-{bb}.{sb}  {b.att_of[name]}"
        else:
            txt = f"arm {name}  {b.att_of[name]}"
        s.text(x + 10, y, txt, fs=3.9, va="center")
    s.notes(200, 84, [
        "Tile i of a module goes to channel i of its attenuator at both ends; never swap tiles.",
        "Tag both ends of every cable with the ID shown; torque SMA to 8 in-lb with a backing wrench.",
        "Before cabling, measure DC, LO and IF level on one tile port per module; DC blocks if DC is present.",
    ] + (["Splitters: S to the hub module tile, port 1 and 2 to the two links' attenuators (one splitter per tile)."]
         if b.split else []) + (["Couplers: IN on the module, THRU to the link attenuator, CPL to the arm (BLK-18-S+ and pad "
                                  "at the coupler); arms phase-matched."] if b.coupled else []), fs=4.6, step=3.1)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 3: power
# ------------------------------------------------------------------------------------------
def sheet_power(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Power: AC, DC and PoE", "wall -> PDU -> loads · DC drawer -> DC/PoE adaptors -> node ETH1")
    rear = {}
    for i, box in enumerate(d["boxes"]):
        rp = spider_rear(s, b, box, 16 + i * 98, 238)
        for k, v in rp.items():
            rear[(box["id"],) + k] = v
    # adaptor shelf, each adaptor under the node it feeds
    s.rect(16, 196, 390, 22, fc=FACE, lw=0.7, z=2)
    s.text(19.5, 215.6, "PA-SHELF", fs=5.2, fontweight="bold", va="center")
    s.text(19.5, 212.8, "Siklu EH-PoE-DC-Adaptor x 10 on a 1U shelf", fs=3.8, color=DIM, va="center")
    ad = {}
    nodes = list(d["nodes"])
    for i in range(b.n_adaptors):
        name = f"PA-{i + 1:02d}"
        if i < len(nodes):
            nd = nodes[i]
            x = rear[(d["nodes"][nd]["box"], nd, "ETH1")][0]
        else:
            xs = sorted([20] + [rear[(d["nodes"][n]["box"], n, "ETH1")][0] for n in nodes] + [402])
            gap = max(zip(xs, xs[1:]), key=lambda g: g[1] - g[0])
            nsp = b.n_adaptors - len(nodes)
            x = gap[0] + (gap[1] - gap[0]) * (i - len(nodes) + 1) / (nsp + 1)
        ad[name] = adaptor(s, x, 204, name)
    # DC drawer
    dist = dcdist_face(s, 16, 164)
    rect = rect_face(s, "DC-RECT", "Mean Well RCP-1UI + 2 x RCP-1000-48 · 48 V", 16, 142)
    rect2 = rect_face(s, "DC-RECT2", "RCP-1UI + RCP-1000-48 · -48 V for IP-50", 16, 120, n=1) if b.ip50 else None
    # PoE cables
    for nd in nodes:
        a = b.adaptor[nd]
        p0 = ad[a]["rj45"]
        p1 = rear[(d["nodes"][nd]["box"], nd, "ETH1")]
        cab = [c for c in b.cables if c["cls"] == "POE" and c["a"] == f"{a}.rj45"][0]
        s.cable(p0, (p1[0], p1[1] - 1.2), CLS["POE"][0], lw=1.0, d0=(0, 1), d1=(0, -1), ls=CLS["POE"][2], k=8)
        s.tag(p0[0] + 2.6, p0[1] + 4, cab["id"], CLS["POE"][0], rot=0, fs=3.0)
    # DC cables breaker -> adaptor
    for i in range(b.n_adaptors):
        a = f"PA-{i + 1:02d}"
        p0 = dist[i + 1]
        p1 = ad[a]["dc"]
        cab = [c for c in b.cables if c["cls"] == "DC" and c["b"] == f"{a}.dc"][0]
        s.cable(p0, p1, CLS["DC"][0], lw=0.9, d0=(0, -1), d1=(0, -1), k=14)
        s.tag(p0[0], p0[1] - 4.5, cab["id"], CLS["DC"][0], fs=2.8)
    bus = [c for c in b.cables if c["cls"] == "DC" and c["a"] == "DC-RECT.bus"][0]
    s.cable(rect["out"], dist["bus"], CLS["DC"][0], lw=1.4, d0=(0, 1), d1=(0, -1), k=6)
    s.tag(rect["out"][0] - 4, rect["out"][1] + 6, bus["id"], CLS["DC"][0], rot=0, fs=3)
    if rect2:
        k48 = [c for c in b.cables if c["cls"] == "DC" and c["a"] == "DC-RECT2.out"]
        s.text(rect2["out"][0] - 4, 112, "-> IP-50 radios (sheet 5): " + ", ".join(c["id"] for c in k48), fs=3.8,
               color=CLS["DC"][0])
        s.line([rect2["out"], (rect2["out"][0], 114)], color=CLS["DC"][0], lw=1.0, z=9)
    # PDUs and AC loads
    pdu_ports = {}
    for j, p in enumerate(b.pdus):
        pdu_ports[p] = pdu_face(s, p, "APC AP7922B · 32 A · 16 x C13" + (" · switched" if d.get("event") else ""),
                                200, 150 - j * 30)
    # load inlets
    loads = list(b.outlet.items())
    inlet_xy = {}
    cols = 8
    for i, (ld, (pdu, o)) in enumerate(loads):
        r, c = divmod(i, cols)
        x = 202 + c * 25
        y = 100 - r * 12 - (len(b.pdus) - 1) * 30
        tile(s, x, y - 4, 22, 8.5, ld)
        c14(s, x + 18, y - 1.4)
        inlet_xy[ld] = (x + 18, y + 0.1)
    for ld, (pdu, o) in loads:
        p0 = pdu_ports[pdu][o]
        p1 = inlet_xy[ld]
        cab = [c for c in b.cables if c["cls"] == "AC" and c["a"] == f"{pdu}.O{o}"][0]
        s.cable((p0[0], p0[1] - 1.6), p1, CLS["AC"][0], lw=0.8, d0=(0, -1), d1=(0, 1))
        s.tag(p0[0], p0[1] - 5.6, cab["id"], CLS["AC"][0], fs=2.8)
    # wall feed
    for j, p in enumerate(b.pdus):
        cab = [c for c in b.cables if c["cls"] == "AC" and c["b"] == f"{p}.inlet"][0]
        pi = pdu_ports[p]["inlet"]
        s.line([pi, (pi[0], pi[1] - 4), (404, pi[1] - 4)], color=CLS["AC"][0], lw=1.4, z=9)
        s.tag(398, pi[1] - 4, cab["id"] + " wall", CLS["AC"][0], rot=0, fs=3)
    # GNSS
    gx, gy = 300, 220
    s.rect(gx - 2, gy - 3, 48, 7, fc=FACE, lw=0.6, z=3)
    s.text(gx, gy + 2.2, "GNSS-SPL (TBC)", fs=4, fontweight="bold", va="center")
    for i, box in enumerate(d["boxes"]):
        ox = gx + 20 + i * 6
        sma(s, ox, gy - 0.8, r=0.9)
        p1 = rear[(box["id"], "box", "GNSS")]
        cab = [c for c in b.cables if c["cls"] == "GNSS" and c["b"] == f"{box['id']}.GNSS"][0]
        s.cable((ox, gy + 0.2), (p1[0], p1[1] - 1), CLS["GNSS"][0], lw=0.7, d0=(0, 1), d1=(0, -1), ls=CLS["GNSS"][2])
        s.tag(ox, gy + 6, cab["id"], CLS["GNSS"][0], fs=2.6)
    s.legend(16, 104 if not rect2 else 100, [CLS[k] for k in ("AC", "DC", "POE", "GNSS")])
    s.notes(16, 78, [
        "N367 power is ETH1 only: 36-57 V DC through the passive adaptor; ETH1 then carries no data.",
        f"{len(nodes)} nodes x 48 W (PoE-Out off) = {48 * len(nodes)} W; breakers 3-5 A; spare adaptors unwired.",
        "Check adaptor terminal polarity with a DMM before power-up; keep node PoE-Out (ETH2) disabled.",
        "IP-50 radios take -48 V (positive grounded) from DC-RECT2, isolated from the 48 V node bus.",
        "Spider rear layout (ETH1/ETH2/SFP+ access, GNSS input, any box inlet) is assumed: confirm on the units.",
        "Servers, hosts and the analyser take mains from the bench-table strip (sheet 5).",
    ] + ([f"Event (t0): CORE servers and IP-50s and the hub's breaker are switched off; see sheet 6."] if d.get("event") else []),
        fs=4.6, step=3.1)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 4: data and management
# ------------------------------------------------------------------------------------------
def sheet_data(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Data and management", "node SFP+ to SW-DATA (one VLAN per site) · management on SW-OOB")
    rear = {}
    for i, box in enumerate(d["boxes"]):
        rp = spider_rear(s, b, box, 16 + i * 98, 238)
        for k, v in rp.items():
            rear[(box["id"],) + k] = v
    sw = switch_front(s, "SW-DATA", "Netgear M4350-24F4V · 24 SFP+ + 4 SFP28", 16, 190, 28, "sfp", w=200, h=20)
    oob_n = b.oob_model[1]
    so = switch_front(s, "SW-OOB", "Netgear " + b.oob_model[0], 16, 158, oob_n, "rj45", w=390 if oob_n > 26 else 200,
                      h=20)
    # endpoints row(s)
    eps = []
    for k, (hop, site) in b.ip50.items():
        eps.append((k, f"hop {hop} @ {site}", [("data", f"{k}.data"), ("mgmt", f"{k}.mgmt")]))
    for sv in d["servers"]:
        lst = [("mgmt", f"{sv[0]}.mgmt")] + ([("data", f"{sv[0]}.data")] if sv[2] else [])
        eps.append((sv[0], sv[1][:22], lst))
    for h, sites in d["hosts"].items():
        eps.append((f"HOST-{h}", "sites " + ",".join(sites), [("10G", f"HOST-{h}.nic10g"), ("mgmt", f"HOST-{h}.mgmt")]))
    for a in d.get("agents", []):
        eps.append((f"AGENT-{a}", "MARL agent", [("data", f"AGENT-{a}.data"), ("mgmt", f"AGENT-{a}.mgmt")]))
    for a in b.atts:
        eps.append((a, "LAN", [("lan", f"{a}.lan")]))
    for p in b.pdus:
        eps.append((p, "PDU", [("net", f"{p}.net")]))
    eps.append(("SPECAN", "analyser", [("lan", "SPECAN.lan")]))
    per_row = 13
    ep_xy = {}
    for i, (nm, sub, ports) in enumerate(eps):
        r, c = divmod(i, per_row)
        x = 16 + c * 30
        y = 118 - r * 20
        tile(s, x, y, 27, 10, nm, sub)
        for j, (pn, key) in enumerate(ports):
            px = x + 5 + j * 9
            (sfp if pn in ("10G",) else rj45)(s, px, y + 1.6, w=3.0, h=1.8)
            s.text(px, y - 1.2, pn, fs=2.6, color=DIM, ha="center", va="center")
            ep_xy[key] = (px, y + 2.5)
    for c in b.cables:
        if c["cls"] not in ("FIB", "ETH"):
            continue
        dst = c["b"]
        if dst.startswith("SW-DATA"):
            p1 = sw[int(dst.split(".P")[1])]
        else:
            p1 = so[int(dst.split(".P")[1])]
        key = c["a"]
        if key in ep_xy:
            p0 = ep_xy[key]
            s.cable(p0, (p1[0], p1[1] - 1), CLS[c["cls"]][0], lw=0.6, d0=(0, 1), d1=(0, -1), ls=CLS[c["cls"]][2])
            s.tag(p0[0], p0[1] + 3.6, c["id"], CLS[c["cls"]][0], fs=2.4)
        else:
            parts = key.split(".")
            p0 = rear[(parts[0], parts[1], parts[2])]
            s.cable((p0[0], p0[1] - 1), (p1[0], p1[1] + 1), CLS[c["cls"]][0], lw=0.7, d0=(0, -1), d1=(0, 1),
                    ls=CLS[c["cls"]][2])
            s.tag(p0[0], p0[1] - 5, c["id"], CLS[c["cls"]][0], fs=2.6)
    # VLAN table
    vr = []
    for site in d["sites"]:
        mem = [f"{site} SFP+"] if site in d["nodes"] and site not in d.get("no_data", []) else []
        mem += [k for k, (hop, st) in b.ip50.items() if st == site]
        mem += [f"HOST-{h}" for h, v in d["hosts"].items() if site in v]
        mem += [f"AGENT-{site}"] if site in d.get("agents", []) else []
        mem += [sv[0] for sv in d["servers"] if sv[2] == site]
        vr.append([str(b.vlan(site)), site, ", ".join(mem)])
    vr.append(["100", "mgmt", "everything on SW-OOB"])
    nrows_ep = math.ceil(len(eps) / per_row)
    ytab = min(118 - nrows_ep * 20 + 14, 78)
    s.text(222, ytab + 3, "VLAN PLAN (SW-DATA)", fs=6, fontweight="bold")
    s.table(222, ytab, ["VLAN", "Site", "Members"], vr, [11, 11, 160], fs=3.8, rh=3.0)
    s.legend(16, 52 if len(eps) > per_row * 3 else 74, [CLS["FIB"], CLS["ETH"]])
    s.notes(70, 74 if len(eps) <= per_row * 3 else 52, [
        "One VLAN per site; no inter-VLAN routing on SW-DATA: sites meet only through the radio links.",
        "SW-OOB is not uplinked to SW-DATA; controllers reach the nodes through ETH2.",
        "Node SFP+: 3.3 V Class-1 SR optic; DAC in a Siklu cage is not documented.",
    ] + (["The hub has no wired data port (its links are all radio)."] if d.get("no_data") else []), fs=4.6, step=3.1)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 5: bench table
# ------------------------------------------------------------------------------------------
def sheet_table(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "Bench table: IP-50 hops, compute, GNSS", "conducted microwave chains and the off-rack equipment")
    y = 250
    if b.ip50:
        s.text(16, 266, "IP-50 CONDUCTED HOPS  (the J7204B channels drawn inline; the unit itself is one 4-channel box)",
               fs=6.2, fontweight="bold")
        hops = list(d["mw_links"].items())
        top, bot = y + 14, y - (len(hops) - 1) * 26 - 2
        s.rect(150, bot, 46, top - bot, fc=FACE, lw=0.8, z=3)
        s.text(152, top - 2.5, "MWATT", fs=5.2, fontweight="bold", va="center")
        s.text(152, top - 5.2, "Keysight J7204B (4 ch)", fs=3.6, color=DIM, va="center")
        for j, (k, (a, bb)) in enumerate(hops):
            yy = y - j * 26
            ra = ip50_radio(s, 16, yy, f"IP50-{k}@{a}", +1)
            ta = transition(s, 44, yy + 7)
            pa = pad(s, 64, yy + 7)
            rb = ip50_radio(s, 310, yy, f"IP50-{k}@{bb}", -1)
            tb = transition(s, 296, yy + 7)
            pb = pad(s, 272, yy + 7)
            s.rect(156, yy + 2.5, 34, 9, fc=PLATE, lw=0.4, z=5)
            s.text(173, yy + 9.4, f"CH{j + 1}", fs=3.4, color=DIM, ha="center", va="center")
            sma(s, 160, yy + 7, r=1.0)
            sma(s, 186, yy + 7, r=1.0)
            s.line([ra["wg"], ta["wg"]], color=CLS["WG"][0], lw=1.6, z=9)
            s.line([ta["coax"], pa["a"]], color=CLS["WG"][0], lw=1.0, z=9)
            s.line([tb["wg"], rb["wg"]], color=CLS["WG"][0], lw=1.6, z=9)
            s.line([pb["b"], tb["coax"]], color=CLS["WG"][0], lw=1.0, z=9)
            s.line([pa["b"], (158.8, yy + 7)], color=CLS["WG"][0], lw=1.0, z=9)
            s.line([(187.2, yy + 7), pb["a"]], color=CLS["WG"][0], lw=1.0, z=9)
            cabs = [c for c in b.cables if c["cls"] == "WG" and (f"-{k}-" in c["a"] + c["b"] or f"-{k}@" in c["a"] + c["b"])]
            s.tag(110, yy + 9.4, cabs[1]["id"], CLS["WG"][0], rot=0, fs=3.2)
            s.tag(232, yy + 9.4, cabs[2]["id"], CLS["WG"][0], rot=0, fs=3.2)
            s.tag(41, yy + 10.6, cabs[0]["id"], CLS["WG"][0], rot=0, fs=2.8)
            s.tag(303, yy + 10.6, cabs[3]["id"], CLS["WG"][0], rot=0, fs=2.8)
            s.text(82, yy + 14.2, f"hop {k}:  {a}  <->  {bb}", fs=4.6, fontweight="bold")
            for r, nm, x0 in ((ra, f"IP50-{k}@{a}", 16), (rb, f"IP50-{k}@{bb}", 310)):
                ids = [c["id"] for c in b.cables if c["cls"] in ("DC", "FIB", "ETH") and (c["a"].startswith(nm + ".")
                                                                                          or c["b"].startswith(nm + "."))]
                s.text(x0, yy - 5.2, "power / data / mgmt: " + " ".join(ids), fs=3.0, color=DIM)
        y = bot - 12
    # compute, test, timing, with the bench mains strip
    y0 = min(y, 150)
    s.text(16, y0, "COMPUTE, TEST AND TIMING", fs=6.2, fontweight="bold")
    strip_y = y0 - 14
    s.rect(16, strip_y, 120, 8, fc=FACE, lw=0.7, z=3, r=1)
    s.text(18, strip_y + 5.6, "TABLE-STRIP", fs=4.6, fontweight="bold", va="center")
    s.text(18, strip_y + 2.4, "lab mains, C13", fs=3.4, color=DIM, va="center")
    items = [(sv[0], sv[1], "server") for sv in d["servers"]] + \
            [(f"HOST-{h}", "traffic host · " + ",".join(v), "host") for h, v in d["hosts"].items()] + \
            [("SPECAN", "analyser / NRP18S", "test")]
    outlets = {}
    for i, (nm, sub, kind) in enumerate(items):
        ox = 48 + i * (86 / max(1, len(items) - 1)) if len(items) > 1 else 60
        c13(s, ox, strip_y + 4)
        outlets[nm] = (ox, strip_y)
    yy = strip_y - 26
    for i, (nm, sub, kind) in enumerate(items):
        r, c = divmod(i, 9)
        x = 16 + c * 43
        yb = yy - r * 40
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
            s.rect(x + 2, yb - 12, 26, 12, fc=FACE, lw=0.6, z=4, r=0.8)
            s.rect(x + 4, yb - 10, 12, 8, fc=PAPER, lw=0.4, z=5)
            inlet = (x + 15, yb)
        s.text(x + 2, yb - 26, nm, fs=4.4, fontweight="bold")
        s.text(x + 2, yb - 29, sub[:32], fs=3.3, color=DIM)
        ids = [c["id"] for c in b.cables if c["cls"] in ("FIB", "ETH") and c["a"].startswith(nm + ".")]
        s.text(x + 22, yb - 2, "\n".join(ids), fs=3.0, color=DIM, va="top")
        cab = [c for c in b.cables if c["cls"] == "AC" and c["b"] == f"{nm}.ac"][0]
        s.cable(outlets[nm], inlet, CLS["AC"][0], lw=0.8, d0=(0, -1), d1=(0, 1), k=6)
        s.tag(inlet[0] + 3.2, inlet[1] + 3.5, cab["id"], CLS["AC"][0], rot=0, fs=2.8)
    # GNSS antenna
    gx = 260
    s.rect(gx, y0 - 26, 6, 14, fc=FACE, lw=0.6, z=4, r=2.6)
    s.text(gx + 9, y0 - 16, "GNSS-ANT  101898-00 (roof)", fs=4.4, fontweight="bold")
    g = [c for c in b.cables if c["cls"] == "GNSS" and c["a"] == "GNSS-ANT"][0]
    s.text(gx + 9, y0 - 19.5, f"{g['id']}: LMR-400 + surge arrestor -> GNSS-SPL in the rack (sheet 3)", fs=3.6, color=DIM)
    s.notes(16, 40, [
        "IP-50 band, flange (WR-62 / WR-42 / WR-28) and Tx power TBC; >= 110 dB in the chain takes a hop down.",
        "30 dB 10 W pads sit at both radio ports; one J7204B channel per hop sets the fade.",
    ] + (["Event: CORE's hops go to maximum attenuation at t0; W1-W2 and E1-E2 stay."] if d.get("event") else []),
        fs=4.6, step=3.1)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


# ------------------------------------------------------------------------------------------
# sheet 6: RF ladder and attenuator schedule
# ------------------------------------------------------------------------------------------
def rng(cs):
    return f"{cs[0]['id']}-{cs[-1]['id'][1:]}" if len(cs) > 1 else cs[0]["id"]


def sheet_ladder(b, pdf, n, N, prefix):
    d = b.d
    s = Sheet(b, n, N, "RF ladder and attenuator schedule", "one row per link and coupling arm; each line = 4 tile cables")
    y, rh = 262, 10
    xm1, xs1, xa, xs2, xm2 = 16, 74, 104, 166, 196
    for name, (sa, sb) in d["link_ends"].items():
        att = b.att_of[name]
        for x0, sec in ((xm1, sa), (xm2, sb)):
            bx, m = d["module"][sec]
            s.rect(x0, y - 3, 46, 6.4, fc="#245a9b", lw=0.6, z=4)
            s.text(x0 + 1.2, y + 1, f"{bx}.{m}", fs=4.6, fontweight="bold", va="center")
            s.text(x0 + 1.2, y - 1.6, f"{sec[0]}.{sec[1]}", fs=3.8, color=DIM, va="center")
        s.rect(xa, y - 3.6, 50, 7.2, fc=FACE, lw=0.7, z=4)
        s.text(xa + 25, y + 1.2, f"{att} · {name}", fs=4.8, fontweight="bold", ha="center", va="center")
        s.text(xa + 25, y - 1.9, "in ch1-4      out ch1-4", fs=3.6, color=DIM, ha="center", va="center")
        for side, sec, xl, xr in (("in", sa, xm1 + 46, xa), ("out", sb, xa + 50, xm2)):
            bx, m = d["module"][sec]
            mid = (xs1 if side == "in" else xs2)
            if sec in b.split:
                sid = b.split[sec]
                k = d["splitters"][sec].index(name) + 1
                s.rect(mid - 6, y - 2.4, 12, 4.8, fc=PLATE, lw=0.6, z=5)
                s.text(mid, y, f"{sid} P{k}", fs=3.0, ha="center", va="center")
                c1 = b.bundle(f"{bx}.{m}.T", f"{sid}.")
                c2 = b.bundle(f"{sid}.", f"{att}.ch")
                s.line([(xl, y), (mid - 6, y)] if side == "in" else [(mid + 6, y), (xr, y)], color=INK, lw=1.4, z=6)
                s.line([(mid + 6, y), (xr, y)] if side == "in" else [(xl, y), (mid - 6, y)], color=CLS["RF"][0], lw=1.6,
                       z=6)
                s.text((xl + mid - 6) / 2 if side == "in" else (mid + 6 + xr) / 2, y + 1.6, rng(c1), fs=3.0, ha="center",
                       color=DIM)
                s.text((mid + 6 + xr) / 2 if side == "in" else (xl + mid - 6) / 2, y + 1.6, rng(c2), fs=3.0,
                       ha="center", color=CLS["RF"][0])
            elif sec in b.coupled:
                cid = b.coupled[sec]
                s.rect(mid - 6, y - 2.4, 12, 4.8, fc=PLATE, lw=0.6, z=5)
                s.text(mid, y, cid.replace("CPL-", "C:"), fs=3.2, ha="center", va="center")
                c1 = b.bundle(f"{bx}.{m}.T", f"{cid}.")
                c2 = b.bundle(f"{cid}.", f"{att}.ch")
                a1, a2 = ((xl, mid - 6), (mid + 6, xr)) if side == "in" else ((mid + 6, xr), (xl, mid - 6))
                s.line([(a1[0], y), (a1[1], y)], color=CLS["RF"][0], lw=1.6, z=6)
                s.line([(a2[0], y), (a2[1], y)], color=CLS["RF"][0], lw=1.6, z=6)
                s.text(sum(a1) / 2, y + 1.6, rng(c1), fs=3.0, ha="center", color=CLS["RF"][0])
                s.text(sum(a2) / 2, y + 1.6, rng(c2), fs=3.0, ha="center", color=CLS["RF"][0])
            else:
                c1 = b.bundle(f"{bx}.{m}.T", f"{att}.ch")
                s.line([(xl, y), (xr, y)], color=CLS["RF"][0], lw=1.6, z=6)
                s.text((xl + xr) / 2, y + 1.6, rng(c1), fs=3.2, ha="center", color=CLS["RF"][0])
        y -= rh
    for name, (x, z) in d["couplings"].items():
        att = b.att_of[name]
        s.rect(xa, y - 3.6, 50, 7.2, fc=FACE, lw=0.7, z=4)
        s.text(xa + 25, y + 1.2, f"{att} · arm {name}", fs=4.8, fontweight="bold", ha="center", va="center")
        s.text(xa + 25, y - 1.9, "in ch1-4      out ch1-4", fs=3.6, color=DIM, ha="center", va="center")
        for x0, sec in ((xm1, x), (xm2, z)):
            s.rect(x0, y - 3, 46, 6.4, fc=PAPER, lw=0.6, z=4, ls=(0, (2, 1)))
            s.text(x0 + 1.2, y + 1, f"{b.coupled[sec]} CPL", fs=4.4, fontweight="bold", va="center")
            s.text(x0 + 1.2, y - 1.6, f"{sec[0]}.{sec[1]} (DC block + pad)", fs=3.6, color=DIM, va="center")
        c1 = b.bundle(f"{b.coupled[x]}.", f"{att}.ch")
        c2 = b.bundle(f"{att}.ch", f"{b.coupled[z]}.")
        s.line([(xm1 + 46, y), (xa, y)], color=CLS["RF"][0], lw=1.4, z=6, ls=(0, (3, 1.2)))
        s.line([(xa + 50, y), (xm2, y)], color=CLS["RF"][0], lw=1.4, z=6, ls=(0, (3, 1.2)))
        s.text((xm1 + 46 + xa) / 2, y + 1.6, rng(c1), fs=3.2, ha="center", color=CLS["RF"][0])
        s.text((xa + 50 + xm2) / 2, y + 1.6, rng(c2), fs=3.2, ha="center", color=CLS["RF"][0])
        y -= rh
    # schedule
    rows = []
    for a, (kind, name) in b.atts.items():
        if kind == "link":
            sa, sb = d["link_ends"][name]
            fa = "{}.{} {}.{}".format(*d["module"][sa], *sa) + (f" via {b.split[sa]}" if sa in b.split else "") + \
                (f" via {b.coupled[sa]}" if sa in b.coupled else "")
            fb = "{}.{} {}.{}".format(*d["module"][sb], *sb) + (f" via {b.coupled[sb]}" if sb in b.coupled else "")
            rows.append([a, f"link {name}", fa, fb])
        else:
            x, z = d["couplings"][name]
            rows.append([a, f"arm {name}", f"{b.coupled[x]} CPL", f"{b.coupled[z]} CPL"])
    s.text(250, 276, "ATTENUATOR SCHEDULE", fs=6.5, fontweight="bold")
    yt = s.table(250, 272, ["Unit", "Carries", "in ch1-4 from", "out ch1-4 to"], rows, [15, 22, 63, 56], fs=4.0, rh=3.6)
    s.notes(250, yt - 6, [
        "Tile i to channel i at both ends of a path; never swap tiles.",
        f"{b.count('RF')} IF cables: {b.count('RF', 'CBL-2FT')} x CBL-2FT-SMSM+ (0.6 m), {b.count('RF', '086-4SM')} x "
        "086-4SM+ jumpers. The team list has 10.",
        "Single-tile operation, if the links allow it, divides the IF cabling and attenuator channels by four.",
    ] + ([f"Event: {d['event']}"] if d.get("event") else []), fs=4.4, step=3.0)
    s.save(pdf, os.path.join(OUT, f"{prefix}_blueprint_{n}.png"))


def sheet_parts(b, P, pdf, n, N, prefix):
    s = Sheet(b, n, N, "Exploded parts list", "the team kit list expanded to part numbers and quantities")
    rows = [[p["line"], p["team"], p["group"], p["item"], p["pn"], p["mfr"], p["qty"], p["conf"]] for p in P]
    s.table(16, 272, ["#", "Team", "Group", "Item", "Part number", "Manufacturer", "Qty", "Status"], rows,
            [7, 9, 17, 108, 92, 28, 30, 24], fs=4.3, rh=min(4.6, 210 / (len(rows) + 1)))
    s.notes(16, 40, [
        "Team = item on the lab's kit list (1-11); '-' = needed but not on the list.",
        "Status: on hand; confirmed (part number checked); likely; to confirm with the lab or vendor.",
    ], title="KEY")
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
