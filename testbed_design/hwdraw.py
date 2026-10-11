"""Isometric line-art drawings of the testbed hardware.

A small painter's-algorithm scene (far items first) with primitives for
boxes, features on box faces (panels, ports, displays, vents, flanges),
cylinders and cable curves, and one drawing function per device.  The
devices are drawn as recognisable technical illustrations, not to scale.

Face names: "R" = the front face (y = y0, facing the viewer's right),
"L" = the side face (x = x0), "T" = the top face.
"""
import math

from matplotlib.patches import Polygon

C30, S30 = math.cos(math.radians(30)), 0.5
EDGE = "#2a313c"
INK = "#1d2430"


def P(x, y, z=0.0):
    return ((x - y) * C30, (x + y) * S30 + z)


def shade(base, k):
    """Lighten (k > 0) or darken (k < 0) a hex colour."""
    base = base.lstrip("#")
    r, g, b = (int(base[i:i + 2], 16) for i in (0, 2, 4))
    if k >= 0:
        r, g, b = (int(c + (255 - c) * k) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + k)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


class Scene:
    def __init__(self, ax):
        self.ax = ax
        self.items = []
        self.anchors = {}
        self._eps = 0.0

    # -- bookkeeping ----------------------------------------------------------------
    def anchor(self, num, pt):
        self.anchors.setdefault(num, []).append(pt)

    @staticmethod
    def depth(x, y, z=0.0):
        return -(x + y) + 0.002 * z

    def _add(self, d, kind, payload):
        self._eps += 1e-7
        self.items.append((d + self._eps, kind, payload))

    # -- primitives -----------------------------------------------------------------
    def poly2d(self, pts, fc, ec=EDGE, lw=0.7, d=0.0, alpha=1.0, hatch=None):
        self._add(d, "poly", (pts, fc, ec, lw, alpha, hatch))

    def line2d(self, pts, color=EDGE, lw=0.7, d=0.0, ls="-", alpha=1.0):
        self._add(d, "line", (pts, color, lw, ls, alpha))

    def box(self, x, y, z, dx, dy, dz, base, lw=0.8, alpha=1.0, d=None):
        """Shaded box: top light, side mid, front darker.  Returns its geometry."""
        x1, y1, z1 = x + dx, y + dy, z + dz
        if d is None:
            d = self.depth(x + dx / 2, y + dy / 2, z)
        left = [P(x, y, z), P(x, y1, z), P(x, y1, z1), P(x, y, z1)]
        right = [P(x, y, z), P(x1, y, z), P(x1, y, z1), P(x, y, z1)]
        top = [P(x, y, z1), P(x1, y, z1), P(x1, y1, z1), P(x, y1, z1)]
        self.poly2d(left, shade(base, -0.12), lw=lw, d=d, alpha=alpha)
        self.poly2d(right, shade(base, 0.0), lw=lw, d=d, alpha=alpha)
        self.poly2d(top, shade(base, 0.35), lw=lw, d=d, alpha=alpha)
        return dict(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, d=d)

    def fpt(self, b, face, u, v):
        if face == "R":
            return P(b["x"] + u, b["y"], b["z"] + v)
        if face == "L":
            return P(b["x"], b["y"] + u, b["z"] + v)
        return P(b["x"] + u, b["y"] + v, b["z"] + b["dz"])

    def frect(self, b, face, u0, v0, u1, v1, fc, ec=EDGE, lw=0.5, alpha=1.0):
        pts = [self.fpt(b, face, u0, v0), self.fpt(b, face, u1, v0),
               self.fpt(b, face, u1, v1), self.fpt(b, face, u0, v1)]
        self.poly2d(pts, fc, ec=ec, lw=lw, d=b["d"], alpha=alpha)

    def fcircle(self, b, face, uc, vc, r, fc, ec=EDGE, lw=0.5, n=24):
        pts = [self.fpt(b, face, uc + r * math.cos(2 * math.pi * k / n), vc + r * math.sin(2 * math.pi * k / n))
               for k in range(n)]
        self.poly2d(pts, fc, ec=ec, lw=lw, d=b["d"])
        return self.fpt(b, face, uc, vc)

    def fline(self, b, face, uv, color=EDGE, lw=0.5):
        self.line2d([self.fpt(b, face, u, v) for u, v in uv], color=color, lw=lw, d=b["d"])

    def cylinder(self, axis, c, r, length, base, lw=0.6, d=None, n=28):
        """Cylinder along axis 'x', 'y' or 'z', starting at centre point c."""
        cx, cy, cz = c

        def ring(t):
            pts = []
            for k in range(n):
                a = 2 * math.pi * k / n
                if axis == "x":
                    pts.append((cx + t, cy + r * math.cos(a), cz + r * math.sin(a)))
                elif axis == "y":
                    pts.append((cx + r * math.cos(a), cy + t, cz + r * math.sin(a)))
                else:
                    pts.append((cx + r * math.cos(a), cy + r * math.sin(a), cz + t))
            return pts
        a_end, b_end = ring(0), ring(length)
        if d is None:
            mid = {"x": (cx + length / 2, cy, cz), "y": (cx, cy + length / 2, cz),
                   "z": (cx, cy, cz + length / 2)}[axis]
            d = self.depth(*mid)
        body = hull([P(*p) for p in a_end + b_end])
        self.poly2d(body, shade(base, -0.05), lw=lw, d=d)
        cap = b_end if axis == "z" else a_end   # the end that faces the viewer
        self.poly2d([P(*p) for p in cap], shade(base, 0.3), lw=lw, d=d)
        return P(cx, cy, cz + (length if axis == "z" else 0))

    def cable(self, pts3, color, lw=1.8, d=None, outline="#1f2329"):
        """Smooth cable through 3-D control points (Catmull-Rom)."""
        pts = []
        cps = [pts3[0]] + list(pts3) + [pts3[-1]]
        for i in range(1, len(cps) - 2):
            p0, p1, p2, p3 = cps[i - 1], cps[i], cps[i + 1], cps[i + 2]
            for k in range(12):
                t = k / 12
                pts.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                                        + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                                        + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t ** 3)
                                 for j in range(3)))
        pts.append(pts3[-1])
        scr = [P(*p) for p in pts]
        if d is None:
            d = min(self.depth(*p) for p in pts3) - 0.05
        self.line2d(scr, color=outline, lw=lw + 1.0, d=d)
        self.line2d(scr, color=color, lw=lw, d=d)
        return scr[len(scr) // 2]

    def flush(self, z0=10):
        for i, (_, kind, p) in enumerate(sorted(self.items, key=lambda t: t[0])):
            z = z0 + i * 0.002
            if kind == "poly":
                pts, fc, ec, lw, alpha, hatch = p
                self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z,
                                          joinstyle="round", hatch=hatch))
            else:
                pts, color, lw, ls, alpha = p
                self.ax.plot([q[0] for q in pts], [q[1] for q in pts], color=color, lw=lw, ls=ls,
                             alpha=alpha, zorder=z, solid_capstyle="round", solid_joinstyle="round")


# ---------------------------------------------------------------------------------
# devices; each returns a callout anchor point (screen coordinates)
# ---------------------------------------------------------------------------------
def rj45_row(sc, b, face, u0, v0, n, w=0.045, h=0.032, gap=0.012):
    for i in range(n):
        u = u0 + i * (w + gap)
        sc.frect(b, face, u, v0, u + w, v0 + h, "#2b2f36", lw=0.3)
        sc.frect(b, face, u + w * 0.25, v0 + h * 0.55, u + w * 0.75, v0 + h * 0.9, "#6b7280", ec="none", lw=0)


def led_row(sc, b, face, u0, v, n, colors=("#22c55e", "#f59e0b")):
    for i in range(n):
        sc.fcircle(b, face, u0 + i * 0.03, v, 0.008, colors[i % len(colors)], ec="none", n=10)


def lab_cart(sc, x, y, w=1.25, dd=0.85, h=0.72):
    """Two-shelf lab cart with posts, casters and a handle."""
    base = "#d8d4ca"
    post = "#9aa0a8"
    # far posts first, then shelves, near posts
    for (px, py) in ((x + w - 0.05, y + dd - 0.05), (x, y + dd - 0.05)):
        sc.box(px, py, 0.08, 0.05, 0.05, h - 0.08, post, lw=0.5)
    sc.box(x, y, 0.16, w, dd, 0.035, base)
    sc.box(x + w - 0.05, y, 0.08, 0.05, 0.05, h - 0.08, post, lw=0.5)
    top = sc.box(x, y, h - 0.045, w, dd, 0.045, base)
    sc.box(x, y, 0.08, 0.05, 0.05, h - 0.08, post, lw=0.5)
    for (px, py) in ((x + 0.025, y + 0.025), (x + w - 0.025, y + 0.025), (x + 0.025, y + dd - 0.025),
                     (x + w - 0.025, y + dd - 0.025)):
        sc.cylinder("x", (px - 0.02, py, 0.04), 0.04, 0.04, "#3f444c", lw=0.4)
    # handle on the side face
    sc.line2d([P(x - 0.06, y + 0.15, h - 0.02), P(x - 0.06, y + dd - 0.15, h - 0.02)], color="#6b7280", lw=2.2,
              d=sc.depth(x, y + dd / 2, h) + 0.01)
    for yy in (y + 0.15, y + dd - 0.15):
        sc.line2d([P(x, yy, h - 0.02), P(x - 0.06, yy, h - 0.02)], color="#6b7280", lw=1.6,
                  d=sc.depth(x, yy, h) + 0.01)
    return top


def router(sc, x, y, z, w=0.95, dd=0.55, h=0.09):
    b = sc.box(x, y, z, w, dd, h, "#e7e9ed")
    rj45_row(sc, b, "R", 0.05, 0.025, 8)
    for i in range(2):  # SFP+ cages
        sc.frect(b, "R", 0.56 + i * 0.09, 0.025, 0.64 + i * 0.09, 0.06, "#9aa3b2", lw=0.35)
        sc.frect(b, "R", 0.575 + i * 0.09, 0.033, 0.625 + i * 0.09, 0.052, "#2b2f36", ec="none", lw=0)
    led_row(sc, b, "R", 0.78, 0.045, 5)
    for k in range(6):  # vents on the side
        sc.fline(b, "L", [(0.08 + k * 0.07, 0.02), (0.08 + k * 0.07, 0.07)], color="#8a919c", lw=0.4)
    return sc.fpt(b, "R", w * 0.3, h * 0.5)


def pdu(sc, x, y, z, w=0.14, dd=0.85, h=0.08):
    b = sc.box(x, y, z, w, dd, h, "#4b5563")
    for i in range(6):  # outlets along the long side
        u = 0.08 + i * 0.12
        sc.frect(b, "L", u, 0.015, u + 0.07, 0.065, "#1f2329", lw=0.3)
        sc.fcircle(b, "L", u + 0.02, 0.04, 0.006, "#9aa3b2", ec="none", n=8)
        sc.fcircle(b, "L", u + 0.05, 0.04, 0.006, "#9aa3b2", ec="none", n=8)
    sc.frect(b, "R", 0.02, 0.02, 0.12, 0.06, "#0f172a", lw=0.3)   # meter display
    sc.frect(b, "R", 0.035, 0.03, 0.1, 0.05, "#22c55e", ec="none", lw=0, alpha=0.8)
    return sc.fpt(b, "L", dd * 0.5, h * 0.5)


def mini_pc(sc, x, y, z, w=0.36, dd=0.36, h=0.13, base="#c9ced6", accent="#2563eb"):
    b = sc.box(x, y, z, w, dd, h, base)
    sc.fcircle(b, "R", 0.05, h * 0.55, 0.018, "#eef0f3", lw=0.4)
    sc.fcircle(b, "R", 0.05, h * 0.55, 0.007, accent, ec="none", n=8)
    for i in range(2):
        sc.frect(b, "R", 0.12 + i * 0.06, h * 0.4, 0.16 + i * 0.06, h * 0.62, "#2b2f36", lw=0.3)
    sc.frect(b, "R", 0.26, h * 0.42, 0.33, h * 0.6, "#2b2f36", lw=0.3)
    sc.frect(b, "T", 0.05, 0.05, w - 0.05, dd - 0.05, shade(base, 0.25), ec=shade(base, -0.2), lw=0.4)
    return sc.fpt(b, "R", w * 0.5, h * 0.5)


def poe_injector(sc, x, y, z, w=0.26, dd=0.18, h=0.09):
    b = sc.box(x, y, z, w, dd, h, "#e7e9ed")
    rj45_row(sc, b, "R", 0.06, 0.025, 2, w=0.05, h=0.036, gap=0.04)
    led_row(sc, b, "R", 0.2, 0.065, 1)
    sc.cable([(x + w, y + dd * 0.5, z + h * 0.3), (x + w + 0.12, y + dd * 0.5, z + 0.02),
              (x + w + 0.16, y + dd * 0.2, z - 0.05)], "#30353d", lw=1.0)
    return sc.fpt(b, "R", w * 0.5, h * 0.5)


def ip50(sc, x, y, z, w=0.36, dd=0.3, h=0.38):
    """Ceragon IP-50C-style all-outdoor unit: finned body, round antenna
    interface on the front, waveguide-coax transition fitted for the bench."""
    b = sc.box(x, y, z, w, dd, h, "#d9dde3")
    # cooling fins on the side face and the top
    for k in range(7):
        u = 0.03 + k * (dd - 0.06) / 6
        sc.fline(b, "L", [(u, 0.04), (u, h - 0.04)], color="#8a919c", lw=0.6)
    for k in range(5):
        u = 0.04 + k * (w - 0.08) / 4
        sc.fline(b, "T", [(u, 0.03), (u, dd - 0.03)], color="#a3aab5", lw=0.5)
    # antenna interface: round flange with bolts
    cu, cv = w * 0.5, h * 0.55
    sc.fcircle(b, "R", cu, cv, 0.12, "#b9c0ca", lw=0.6)
    sc.fcircle(b, "R", cu, cv, 0.075, "#1f4e9c", lw=0.5)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        sc.fcircle(b, "R", cu + 0.1 * math.cos(a), cv + 0.1 * math.sin(a), 0.01, "#4b5563", ec="none", n=8)
    # Ceragon-blue label band and connector at the bottom
    sc.frect(b, "R", 0.03, 0.03, w - 0.03, 0.07, "#1f4e9c", ec="none", lw=0)
    sc.cylinder("z", (x + w * 0.8, y + dd * 0.5, z - 0.05), 0.022, 0.05, "#9aa3b2", lw=0.4)
    # waveguide-coax transition on the flange: block + N connector
    tb = sc.box(x + cu - 0.06, y - 0.08, z + cv - 0.06, 0.12, 0.08, 0.12, "#b08a22", lw=0.5,
                d=b["d"] + 1e-4)
    sc.cylinder("y", (x + cu, y - 0.08, z + cv), 0.025, -0.06, "#c0c6cf", lw=0.4, d=b["d"] + 2e-4)
    return sc.fpt(b, "R", w * 0.25, h * 0.85), sc.fpt(tb, "R", 0.06, 0.06)


def tg_node(sc, x, y, z, ports=2, w=0.42, dd=0.42, h=0.34, terminal=False):
    """Siklu MultiHaul TG-style unit: radome sector panels (one face per 90 deg
    sector), pole bracket; RF-port build with ports at the bottom."""
    if terminal:
        w, dd, h = 0.3, 0.12, 0.28
    b = sc.box(x, y, z, w, dd, h, "#f4f1ec")
    pad = 0.035
    sc.frect(b, "R", pad, pad, w - pad, h - pad, "#fbfaf7", ec="#c86d12", lw=0.7)
    if not terminal:
        sc.frect(b, "L", pad, pad, dd - pad, h - pad, "#f6efe6", ec="#c86d12", lw=0.7)
    sc.frect(b, "R", w * 0.4, h * 0.12, w * 0.6, h * 0.18, "#e07a10", ec="none", lw=0)   # logo bar
    sc.frect(b, "T", 0.04, 0.04, w - 0.04, dd - 0.04, "#efe9e0", ec="#c9c2b6", lw=0.4)
    # pole bracket behind
    sc.box(x + w * 0.35, y + dd, z + h * 0.2, w * 0.3, 0.05, h * 0.6, "#9aa0a8", lw=0.5,
           d=b["d"] - 0.01)
    # RF test ports (one per used sector) under the unit
    for k in range(max(1, ports)):
        sc.cylinder("z", (x + 0.07 + k * 0.08, y + dd * 0.35, z - 0.06), 0.02, 0.06, "#b08a22", lw=0.4)
    return sc.fpt(b, "R", w * 0.5, h * 0.75)


def prog_attenuator(sc, x, y, z, w=0.64, dd=0.44, h=0.2):
    """Bench programmable attenuator: display, knob, two RF connectors, feet."""
    for (fx, fy) in ((x + 0.04, y + 0.04), (x + w - 0.08, y + 0.04)):
        sc.box(fx, fy, z, 0.04, 0.04, 0.02, "#2b2f36", lw=0.3)
    b = sc.box(x, y, z + 0.02, w, dd, h, "#3a3f47")
    sc.frect(b, "R", 0.04, 0.06, 0.3, 0.16, "#0f172a", lw=0.4)
    sc.frect(b, "R", 0.06, 0.09, 0.26, 0.13, "#38bdf8", ec="none", lw=0, alpha=0.9)
    sc.fcircle(b, "R", 0.38, 0.11, 0.035, "#9aa3b2", lw=0.4)
    for u in (0.48, 0.58):
        sc.fcircle(b, "R", u, 0.1, 0.022, "#c0c6cf", lw=0.4)
        sc.fcircle(b, "R", u, 0.1, 0.008, "#b08a22", ec="none", n=8)
    return sc.fpt(b, "R", 0.17, 0.11)


def fixed_pad(sc, x, y, z, axis="x"):
    """Inline coaxial attenuator: knurled body between two connectors."""
    L = 0.2
    d = sc.depth(x, y, z)
    sc.cylinder(axis, (x, y, z), 0.03, L * 0.2, "#c0c6cf", d=d)
    off = (L * 0.2, 0, 0) if axis == "x" else (0, L * 0.2, 0)
    sc.cylinder(axis, (x + off[0], y + off[1], z), 0.045, L * 0.6, "#d4a93a", d=d + 1e-4)
    off2 = (L * 0.8, 0, 0) if axis == "x" else (0, L * 0.8, 0)
    sc.cylinder(axis, (x + off2[0], y + off2[1], z), 0.03, L * 0.2, "#c0c6cf", d=d + 2e-4)
    return P(x + off[0] + 0.05, y + off[1], z + 0.045)


def rack_frame(sc, x, y, w=0.9, dd=0.8, h=1.7, color="#6b7280"):
    """Open-frame 19-inch rack: four posts, top and bottom frames."""
    d0 = sc.depth(x + w / 2, y + dd / 2, 0)
    posts = [(x + w - 0.05, y + dd - 0.05), (x, y + dd - 0.05), (x + w - 0.05, y), (x, y)]
    for i, (px, py) in enumerate(posts):
        sc.box(px, py, 0, 0.05, 0.05, h, color, lw=0.4, d=d0 + (0.3 if i >= 2 else -0.3))
    for zz in (0.0, h - 0.05):
        sc.box(x, y, zz, w, 0.05, 0.05, color, lw=0.4, d=d0 + 0.31)
        sc.box(x, y + 0.05, zz, 0.05, dd - 0.1, 0.05, color, lw=0.4, d=d0 + 0.31)
    return d0


def rack_unit(sc, x, y, z, w, dd, h, base, kind="ports", d=None):
    b = sc.box(x, y, z, w, dd, h, base, d=d)
    if kind == "ports":
        rj45_row(sc, b, "R", 0.06, h * 0.3, min(12, int((w - 0.2) / 0.06)), w=0.04, h=min(0.04, h * 0.45))
    elif kind == "server":
        for r in range(2):
            for k in range(6):
                u = 0.05 + k * 0.1
                sc.frect(b, "R", u, 0.02 + r * h * 0.45, u + 0.085, 0.02 + r * h * 0.45 + h * 0.38,
                         "#e5e7eb", lw=0.3)
        for k in range(4):
            sc.fline(b, "R", [(0.7 + k * 0.04, 0.03), (0.7 + k * 0.04, h - 0.03)], color="#6b7280", lw=0.5)
    elif kind == "rf":
        for k in range(8):
            u = 0.08 + k * 0.1
            sc.fcircle(b, "R", u, h * 0.5, 0.022, "#c0c6cf", lw=0.35)
            sc.fcircle(b, "R", u, h * 0.5, 0.008, "#b08a22", ec="none", n=8)
    elif kind == "display":
        sc.frect(b, "R", 0.05, h * 0.25, 0.35, h * 0.75, "#0f172a", lw=0.3)
        sc.frect(b, "R", 0.07, h * 0.35, 0.33, h * 0.65, "#22c55e", ec="none", lw=0, alpha=0.8)
        led_row(sc, b, "R", 0.45, h * 0.5, 4)
    return b


def splitter(sc, x, y, z, w=0.5, dd=0.3, h=0.1, ports=4):
    b = sc.box(x, y, z, w, dd, h, "#c2410c")
    for k in range(ports):
        sc.fcircle(b, "R", 0.07 + k * (w - 0.14) / max(1, ports - 1), h * 0.5, 0.02, "#e5e7eb", lw=0.35)
    sc.fcircle(b, "L", dd * 0.5, h * 0.5, 0.02, "#e5e7eb", lw=0.35)
    return sc.fpt(b, "R", w * 0.5, h * 0.8)


def gnss_clock(sc, x, y, z):
    b = rack_unit(sc, x, y, z, 0.6, 0.45, 0.12, "#e7e9ed", kind="display")
    sc.cylinder("z", (x + 0.5, y + 0.3, z + 0.12), 0.012, 0.75, "#6b7280")
    sc.cylinder("z", (x + 0.5, y + 0.3, z + 0.87), 0.07, 0.05, "#f4f5f7")
    return sc.fpt(b, "R", 0.2, 0.06)


def dc_supply(sc, x, y, z):
    b = sc.box(x, y, z, 0.6, 0.45, 0.3, "#4b5563")
    sc.frect(b, "R", 0.05, 0.16, 0.28, 0.25, "#0f172a", lw=0.3)
    sc.frect(b, "R", 0.07, 0.18, 0.26, 0.23, "#f59e0b", ec="none", lw=0, alpha=0.85)
    for k in range(3):
        sc.fcircle(b, "R", 0.36 + k * 0.08, 0.1, 0.022, "#dc2626" if k == 0 else "#1f2329", lw=0.4)
    for k in range(5):
        sc.fline(b, "L", [(0.06 + k * 0.08, 0.05), (0.06 + k * 0.08, 0.25)], color="#9aa3b2", lw=0.5)
    return sc.fpt(b, "R", 0.3, 0.2)


def power_meter(sc, x, y, z):
    b = sc.box(x, y, z, 0.34, 0.24, 0.12, "#f2c94c")
    sc.frect(b, "R", 0.03, 0.03, 0.2, 0.1, "#0f172a", lw=0.3)
    sc.frect(b, "R", 0.045, 0.045, 0.185, 0.085, "#22c55e", ec="none", lw=0, alpha=0.8)
    sc.fcircle(b, "R", 0.27, 0.06, 0.02, "#4b5563", lw=0.3)
    sc.cable([(x + 0.34, y + 0.12, z + 0.06), (x + 0.5, y + 0.1, z + 0.02), (x + 0.6, y - 0.05, z)],
             "#30353d", lw=1.0)
    sc.cylinder("x", (x + 0.6, y - 0.05, z + 0.03), 0.03, 0.12, "#9aa3b2", lw=0.4)
    return sc.fpt(b, "R", 0.12, 0.06)
