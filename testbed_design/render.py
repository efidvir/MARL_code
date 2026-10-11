"""Draw the four testbed configurations (one figure each + an overview).

Run:  python3 render.py        -> out/config_*.png|svg, out/overview.png|svg
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (FancyArrowPatch, Polygon, Rectangle,  # noqa: E402
                                RegularPolygon, Circle, Wedge)

import networkx as nx  # noqa: E402

from topologies import CONFIGS, ROLE, ZONE_X, ZONE_Y  # noqa: E402
from validate import analyze, bearing, P_PLAN  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

for f in font_manager.findSystemFonts():
    if "Inter" in os.path.basename(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10,
    "svg.fonttype": "none",
})

INK = "#1d2430"
MUTED = "#6b7280"
MW = "#1f4e9c"
TG = "#e07a10"
BRIDGE = "#12945a"
LOST = "#d0342c"
CAND = "#8a94a6"
ZONE = "#e5484d"
ISLAND = ["#2563eb", "#9333ea", "#c2410c", "#be185d", "#475569"]
FLOW = {"F1": "#7c3aed", "F2": "#0891b2", "F3": "#db2777", "F4": "#b45309"}
SVC = "#64748b"

# label placement per site: (dx, dy, ha, va)
LABEL = {"W1": "left", "W2": "left", "CORE": "upper-right", "HUB": "right",
         "E1": "right", "E2": "right", "S": "right", "N": "right"}


# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------
def unit(a, b):
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def dish(ax, xy, toward, color=MW, size=34, alpha=1.0, z=6):
    """IP-50: dish profile (half disc, aperture facing the peer) with its feed arm."""
    b = bearing(xy, toward)
    ux, uy = math.cos(math.radians(b)), math.sin(math.radians(b))
    cx, cy = xy[0] + ux * 100, xy[1] + uy * 100
    ax.add_patch(Wedge((cx, cy), size, b + 90, b + 270, fc=color, ec="white", lw=0.9,
                       alpha=alpha, zorder=z))
    ax.plot([cx, cx + ux * size * 0.95], [cy, cy + uy * size * 0.95], color=color, lw=1.8,
            alpha=alpha, zorder=z, solid_capstyle="round")
    ax.add_patch(Circle((cx + ux * size, cy + uy * size), size * 0.16, fc=color, ec="none",
                        alpha=alpha, zorder=z))


def beam(ax, xy, toward, color=TG, length=230, half=8, alpha=0.30, ls="-", fill=True, z=3, lw=1.0):
    b = bearing(xy, toward)
    w = Wedge(xy, length, b - half, b + half, fc=color if fill else "none",
              ec=color, alpha=alpha if fill else 0.9, lw=lw, ls=ls, zorder=z)
    ax.add_patch(w)


def tg_icon(ax, xy, kind, color=TG, alpha=1.0, z=7):
    if kind == "DN":
        ax.add_patch(Circle(xy, 25, fc=color, ec="white", lw=1.2, alpha=alpha, zorder=z))
        for k in range(4):  # four 90-degree sector panels (east, north, west, south)
            a = math.radians(90 * k)
            c = (xy[0] + 27 * math.cos(a), xy[1] + 27 * math.sin(a))
            w, h = (10, 26) if k % 2 == 0 else (26, 10)
            ax.add_patch(Rectangle((c[0] - w / 2, c[1] - h / 2), w, h, fc=color, ec="white", lw=0.9,
                                   alpha=alpha, zorder=z + 0.1))
    else:
        ax.add_patch(Rectangle((xy[0] - 22, xy[1] - 22), 44, 44, fc=color, ec="white", lw=1.2,
                               alpha=alpha, zorder=z))
        ax.add_patch(Circle(xy, 8, fc="white", ec="none", alpha=alpha, zorder=z + 0.1))


def site_base(ax, xy, dead=False, has_tg=False):
    ax.add_patch(Circle(xy, 46, fc="#f3f4f6" if not dead else "#e5e7eb",
                        ec=INK if not dead else "#9ca3af", lw=1.3, zorder=5))
    if not has_tg:
        ax.add_patch(Circle(xy, 11, fc=INK if not dead else "#9ca3af", ec="none", zorder=6))


def cross(ax, xy, size=58):
    for s in (1, -1):
        ax.plot([xy[0] - size, xy[0] + size], [xy[1] - s * size, xy[1] + s * size],
                color=LOST, lw=3.4, zorder=12, solid_capstyle="round")


def rounded_hull(points, pad):
    pts = []
    for x, y in points:
        for k in range(48):
            a = 2 * math.pi * k / 48
            pts.append((x + pad * math.cos(a), y + pad * math.sin(a)))
    from matplotlib.path import Path  # noqa: F401
    # monotone chain convex hull
    pts = sorted(set(pts))

    def crossz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and crossz(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and crossz(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def link_line(ax, a, b, color, lw, ls="-", alpha=1.0, z=2):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z,
            solid_capstyle="round", dash_capstyle="round")


def offset_poly(pts, off):
    """Offset a polyline sideways by `off` (metres) for parallel path drawing."""
    out = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            u = unit(pts[0], pts[1])
        elif i == n - 1:
            u = unit(pts[-2], pts[-1])
        else:
            u1, u2 = unit(pts[i - 1], pts[i]), unit(pts[i], pts[i + 1])
            u = (u1[0] + u2[0], u1[1] + u2[1])
            m = math.hypot(*u) or 1
            u = (u[0] / m, u[1] / m)
        out.append((pts[i][0] - u[1] * off, pts[i][1] + u[0] * off))
    return out


# ----------------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------------
def frame(ax, r, pad=230):
    xs = [p[0] for p in r["pos"].values()]
    ys = [p[1] for p in r["pos"].values()]
    ax.set_xlim(min(xs) - 470, max(xs) + 470)
    ax.set_ylim(min(ys) - pad - 40, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def zone(ax, r, mode):
    ys = [p[1] for p in r["pos"].values()]
    y0, y1 = max(ZONE_Y[0], min(ys) - 60), min(ZONE_Y[1], max(ys) + 60)
    w = ZONE_X[1] - ZONE_X[0]
    if mode == "planned":
        ax.add_patch(Rectangle((ZONE_X[0], y0), w, y1 - y0, fc="none", ec=ZONE, lw=1.2,
                               ls=(0, (4, 3)), alpha=0.7, zorder=1))
        ax.text(ZONE_X[0] + 22, (y0 + y1) / 2 + 120, "planned disaster zone", ha="left", va="center",
                rotation=90, fontsize=8.5, color=ZONE, zorder=1)
    else:
        ax.add_patch(Rectangle((ZONE_X[0], y0), w, y1 - y0, fc=ZONE, ec="none", alpha=0.07, zorder=0))
        ax.add_patch(Rectangle((ZONE_X[0], y0), w, y1 - y0, fc="none", ec=ZONE, lw=0, hatch="////",
                               alpha=0.18, zorder=0))
        ax.text(ZONE_X[0] + 22, (y0 + y1) / 2 + 120, "disaster zone", ha="left", va="center",
                rotation=90, fontsize=8.5, color=ZONE, fontweight="bold", zorder=1)


def inventory(r, s):
    cfg = r["cfg"]
    parts = []
    if r["ip_units"][s]:
        parts.append(f"{r['ip_units'][s]}× IP-50")
    if s in cfg["tg_units"]:
        parts.append("TG node" if cfg["tg_units"][s] == "DN" else "TG terminal")
    return " + ".join(parts)


def labels(ax, r, dead=(), small=False):
    for s, xy in r["pos"].items():
        side = LABEL[s]
        col = MUTED if s in dead else INK
        role = ROLE[s]
        if s == "CORE" and s in r["cfg"]["tg_units"]:
            role = "Core + MW hub + TG\nsevered by the event"
        role = role if not small else role.split("\n")[0]
        sub = role if small else f"{role}\n{inventory(r, s)}"
        if s in dead:
            sub = "destroyed" if small else sub + "\n(lost in the event)"
        fs_n, fs_s = (10.5, 7.4) if not small else (9.0, 6.9)
        nl = sub.count("\n") + 1
        lh = fs_s * 1.15 * 1.2
        kw = dict(textcoords="offset points", zorder=20, annotation_clip=False,
                  path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        if side in ("left", "right"):
            sg = -1 if side == "left" else 1
            ha = "right" if side == "left" else "left"
            ax.annotate(s, xy, xytext=(sg * 22, 1), ha=ha, va="bottom", fontsize=fs_n,
                        fontweight="bold", color=col, **kw)
            ax.annotate(sub, xy, xytext=(sg * 22, -1), ha=ha, va="top", fontsize=fs_s,
                        color=MUTED, linespacing=1.15, **kw)
        elif side == "upper-right":
            ax.annotate(sub, xy, xytext=(20, 6), ha="left", va="bottom", fontsize=fs_s,
                        color=MUTED, linespacing=1.15, **kw)
            ax.annotate(s, xy, xytext=(20, 7 + nl * lh), ha="left", va="bottom", fontsize=fs_n,
                        fontweight="bold", color=col, **kw)
        elif side == "above":
            ax.annotate(sub, xy, xytext=(0, 21), ha="center", va="bottom", fontsize=fs_s,
                        color=MUTED, linespacing=1.15, **kw)
            ax.annotate(s, xy, xytext=(0, 22 + nl * lh), ha="center", va="bottom", fontsize=fs_n,
                        fontweight="bold", color=col, **kw)
        else:
            ax.annotate(s, xy, xytext=(0, -21), ha="center", va="top", fontsize=fs_n,
                        fontweight="bold", color=col, **kw)
            ax.annotate(sub, xy, xytext=(0, -22 - fs_n * 1.3), ha="center", va="top", fontsize=fs_s,
                        color=MUTED, linespacing=1.15, **kw)


def draw_sites(ax, r, dead=(), devices=True, tg_alpha=1.0):
    cfg = r["cfg"]
    pos = r["pos"]
    for s, xy in pos.items():
        d = s in dead
        site_base(ax, xy, dead=d, has_tg=s in cfg["tg_units"])
        if s in cfg["tg_units"]:
            tg_icon(ax, xy, cfg["tg_units"][s], color=TG if not d else "#c4c8cf", alpha=tg_alpha)
        if devices:
            for a, b in cfg["mw"]:
                if s in (a, b):
                    peer = b if s == a else a
                    dish(ax, xy, pos[peer], color=MW if not d else "#b6bcc6")
        if d:
            cross(ax, xy)


def fixed_tg(ax, a, b, dead=False):
    """TG link that never steers, carried on a spare sector of installed nodes."""
    link_line(ax, a, b, TG if not dead else LOST, 2.6, ls=(0, (5, 2.5)))
    beam(ax, a, b, length=150, half=6, alpha=0.22)
    beam(ax, b, a, length=150, half=6, alpha=0.22)


def panel_before(ax, r):
    cfg, pos = r["cfg"], r["pos"]
    frame(ax, r)
    zone(ax, r, "planned")
    for a, b in cfg["mw"]:
        link_line(ax, pos[a], pos[b], MW, 3.4)
    for a, b in cfg["tg_links"]:
        link_line(ax, pos[a], pos[b], TG, 2.6)
        beam(ax, pos[a], pos[b])
        beam(ax, pos[b], pos[a])
    for a, b in cfg.get("tg_fixed", []):
        fixed_tg(ax, pos[a], pos[b])
    draw_sites(ax, r)
    labels(ax, r)
    # flow tags at sources / destinations
    tags = {}
    for fid, s, t in cfg["flows"]:
        tags.setdefault(s, []).append((fid, "src"))
        tags.setdefault(t, []).append((fid, "dst"))
    for s, t in cfg.get("service", []):
        tags.setdefault(s, []).append(("svc", "src"))
        tags.setdefault(t, []).append(("svc", "dst"))
    for s, lst in tags.items():
        xy = pos[s]
        n = len(lst)
        for i, (fid, kind) in enumerate(lst):
            ax.annotate(f"{fid} {'out' if kind == 'src' else 'in'}", xy,
                        xytext=((1 if LABEL[s] != "right" else -1) * (8 + 36 * i), -21),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=7.2, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.22", fc=FLOW.get(fid, SVC), ec="none"), zorder=21)
    # TG sectors whose airtime two links share: where DPIR's flows contend
    for (site, sec), lst in r["shared"].items():
        c = sec * 90
        ax.add_patch(Wedge(pos[site], 330, c - 45, c + 45, fc=TG, alpha=0.08, ec=TG, lw=1.0,
                           ls=(0, (3, 2)), zorder=1.2))
        peers = " + ".join(y for _, y in lst)
        ang = math.radians(c + (25 if sec in (0, 2) else 0))
        ax.text(pos[site][0] + 360 * math.cos(ang), pos[site][1] + 360 * math.sin(ang),
                f"shared sector\n{peers}", ha="center", va="center", fontsize=6.8, color="#b45309",
                zorder=19, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def island_name(isl):
    if isl == ["S"]:
        return "shelter"
    if isl == ["N"]:
        return "hospital"
    if any(s.startswith("W") for s in isl):
        return "west"
    return "incident"


def panel_event(ax, r):
    cfg, pos = r["cfg"], r["pos"]
    dead = set(cfg["destroyed"])
    frame(ax, r)
    zone(ax, r, "event")
    for k, isl in enumerate(r["islands"]):
        hull = rounded_hull([pos[s] for s in isl], 150)
        col = ISLAND[k % len(ISLAND)]
        ax.add_patch(Polygon(hull, closed=True, fc=col, ec=col, alpha=0.10, lw=0, zorder=0.5))
        ax.add_patch(Polygon(hull, closed=True, fc="none", ec=col, alpha=0.9, lw=1.6, ls=(0, (5, 3)),
                             zorder=0.6))
        top = max(hull, key=lambda p: p[1] if "S" not in isl else -p[1])
        cx = sum(pos[s][0] for s in isl) / len(isl)
        name = f"Island {k + 1} · {{{', '.join(isl)}}}"
        y = max(p[1] for p in hull) + 14
        va = "bottom"
        if isl == ["S"]:
            y = min(p[1] for p in hull) - 14
            va = "top"
        ax.text(cx, y, name, ha="center", va=va, fontsize=10, fontweight="bold", color=col, zorder=22)
    for a, b in cfg["mw"] + cfg["tg_links"]:
        tech = MW if (a, b) in cfg["mw"] else TG
        if a in dead or b in dead:
            link_line(ax, pos[a], pos[b], LOST, 1.5, ls=(0, (3, 3)), alpha=0.75)
        else:
            link_line(ax, pos[a], pos[b], tech, 3.4 if tech == MW else 2.6)
    for a, b in cfg.get("tg_fixed", []):
        fixed_tg(ax, pos[a], pos[b])
    # surviving TG beams: the ones that served a destroyed peer now point at nothing
    for a, b in cfg["tg_links"]:
        for x, y in ((a, b), (b, a)):
            if x in dead:
                continue
            if y in dead:
                beam(ax, pos[x], pos[y], color="#9ca3af", fill=False, ls=(0, (3, 2)), length=200)
            else:
                beam(ax, pos[x], pos[y])
    draw_sites(ax, r, dead=dead)
    labels(ax, r, dead=dead, small=True)


def bridge_label_point(a, b):
    def inz(q):
        return ZONE_X[0] <= q[0] <= ZONE_X[1] and ZONE_Y[0] <= q[1] <= ZONE_Y[1]
    ts = [0.22 + 0.56 * k / 56 for k in range(57)]
    pts = [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in ts]
    if inz(a) is False and inz(b) is False and all(inz(q) for q in pts[20:37]):
        return pts[28]  # crosses the zone: label in the (empty) middle
    runs, cur = [], []
    for q in pts:
        if not inz(q):
            cur.append(q)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return pts[28]
    best = max(runs, key=len)
    return best[len(best) // 2]


def panel_recovery(ax, r, table=True):
    cfg, pos = r["cfg"], r["pos"]
    dead = set(cfg["destroyed"])
    frame(ax, r)
    zone(ax, r, "event")
    surv = [s for s in pos if s not in dead]
    hull = rounded_hull([pos[s] for s in surv], 150)
    ax.add_patch(Polygon(hull, closed=True, fc=BRIDGE, ec=BRIDGE, alpha=0.06, lw=0, zorder=0.4))
    ax.add_patch(Polygon(hull, closed=True, fc="none", ec=BRIDGE, alpha=0.8, lw=1.5, ls=(0, (5, 3)),
                         zorder=0.5))
    for k, isl in enumerate(r["islands"]):
        h = rounded_hull([pos[s] for s in isl], 115)
        col = ISLAND[k % len(ISLAND)]
        ax.add_patch(Polygon(h, closed=True, fc="none", ec=col, alpha=0.45, lw=1.0, ls=(0, (2, 3)),
                             zorder=0.6))
    for a, b in cfg["mw"] + cfg["tg_links"]:
        if a in dead or b in dead:
            link_line(ax, pos[a], pos[b], LOST, 1.0, ls=(0, (2, 4)), alpha=0.35)
    # for each bridge, the best alternative MARL could have used between the same two islands
    comp = r["comp"]
    for h, p in cfg["bridges"]:
        pair = {comp[h], comp[p]}
        alts = [c for c in r["cands"] if c["feasible"] and {comp[c["a"]], comp[c["b"]]} == pair
                and {c["a"], c["b"]} != {h, p}]
        if alts:
            c = max(alts, key=lambda c: c["budget"]["snr"])
            link_line(ax, pos[c["a"]], pos[c["b"]], CAND, 1.5, ls=(0, (1.5, 3)), alpha=0.95, z=1.5)
    # surviving links
    g = r["g_post"]
    for a, b, d in g.edges(data=True):
        if d["tech"] == "BR":
            continue
        tech = d["tech"]
        if d.get("fixed"):
            fixed_tg(ax, pos[a], pos[b])
            continue
        link_line(ax, pos[a], pos[b], MW if tech == "MW" else TG, 3.4 if tech == "MW" else 2.6)
        if tech == "TG":
            beam(ax, pos[a], pos[b])
            beam(ax, pos[b], pos[a])
    # bridges
    rows = []
    for i, (h, p) in enumerate(cfg["bridges"], 1):
        link_line(ax, pos[h], pos[p], BRIDGE, 4.6, z=2.5)
        c = [c for c in r["cands"] if {c["a"], c["b"]} == {h, p}][0]
        bud = c["budget"]
        m = bridge_label_point(pos[h], pos[p])
        ax.text(m[0], m[1], str(i), ha="center", va="center", fontsize=9, color="white", zorder=23,
                fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc=BRIDGE, ec="white", lw=1.2))
        rows.append(f"{i}  {h} ⇄ {p}   {c['d']:.0f} m · SNR {bud['snr']:.0f} dB → {bud['mcs']}"
                    f"   ({h} ignites)")
    # re-steered beams
    for st in r["steer"]:
        x = pos[st["unit"]]
        tgt = pos[st["new"]]
        beam(ax, x, tgt, color=BRIDGE, alpha=0.45, length=250, half=7, z=4)
        if st["old"] is not None:
            old = pos[st["old"]]
            beam(ax, x, old, color="#9ca3af", fill=False, ls=(0, (3, 2)), length=200, z=4)
            a0, a1 = st["b_old"], st["b_new"]
            rad = 150
            # short arc arrow from old to new azimuth
            d = (a1 - a0 + 180) % 360 - 180
            p0 = (x[0] + rad * math.cos(math.radians(a0)), x[1] + rad * math.sin(math.radians(a0)))
            p1 = (x[0] + rad * math.cos(math.radians(a0 + d)), x[1] + rad * math.sin(math.radians(a0 + d)))
            ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={0.35 if d > 0 else -0.35}",
                                         arrowstyle="-|>,head_length=5,head_width=3", color=BRIDGE,
                                         lw=1.6, zorder=13))
            am = math.radians(a0 + d / 2)
            ax.text(x[0] + (rad + 40) * math.cos(am), x[1] + (rad + 40) * math.sin(am),
                    f"{st['delta']:.0f}°", ha="center", va="center", fontsize=8, color=BRIDGE,
                    fontweight="bold", zorder=23,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    draw_sites(ax, r, dead=dead)
    labels(ax, r, dead=dead, small=True)
    if not table:
        return
    ax.text(0.01, -0.01, "\n".join(rows) + f"\nSNR: MARL-paper link model at {P_PLAN} dBm; MCS = paper Table II class",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color=BRIDGE,
            linespacing=1.4, zorder=30, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BRIDGE, lw=0.8, alpha=0.95))


def panel_flow(ax, r, fid, paths, graph_edges, title, bridges=(), post=False):
    cfg, pos = r["cfg"], r["pos"]
    frame(ax, r, pad=150)
    dead = set(cfg["destroyed"]) if post else set()
    for a, b, tech in graph_edges:
        col = {"MW": "#b9c2d3", "TG": "#f2cfa6", "BR": "#a8dcc1"}[tech]
        link_line(ax, pos[a], pos[b], col, 3.0 if tech != "TG" else 2.4)
    for s, xy in pos.items():
        if s in dead:
            ax.add_patch(Circle(xy, 62, fc="#eef0f3", ec="#d1d5db", lw=0.8, zorder=5))
            cross(ax, xy, size=48)
        else:
            ax.add_patch(Circle(xy, 62, fc="white", ec="#9aa3b2", lw=0.9, zorder=5))
        ax.text(xy[0], xy[1], s, ha="center", va="center", fontsize=7.2, color=INK if s not in dead
                else "#9ca3af", zorder=6, fontweight="bold")
    styles = ["-", (0, (6, 3)), (0, (1.5, 2.5))]
    offs = [0, 46, -46]
    if not post:
        col = FLOW[fid]
        for i, p in enumerate(paths):
            pts = offset_poly([pos[n] for n in p], offs[i])
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, lw=3.4 if i == 0 else 2.6,
                    ls=styles[i], zorder=4, solid_capstyle="round", alpha=0.95)
        s, t = paths[0][0], paths[0][-1]
        ax.add_patch(Circle(pos[s], 80, fc="none", ec=col, lw=2.6, zorder=7))
        ax.add_patch(Circle(pos[t], 80, fc="none", ec=col, lw=2.6, ls=(0, (2, 1.5)), zorder=7))
    else:
        for i, (f, p) in enumerate(paths.items()):
            pts = offset_poly([pos[n] for n in p], (i - (len(paths) - 1) / 2) * 26)
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=FLOW.get(f, SVC), lw=3.0,
                    zorder=4, solid_capstyle="round", alpha=0.95,
                    ls="-" if f in FLOW else (0, (2, 1.5)))
    ax.set_title(title, fontsize=10, fontweight="bold", color=INK, loc="left", pad=2)


def tech_str(t):
    out, prev, n = [], None, 0
    for x in t + [None]:
        if x == prev:
            n += 1
        else:
            if prev is not None:
                out.append(f"{'MW' if prev == 'MW' else ('TG' if prev == 'TG' else 'bridge')}"
                           + (f"×{n}" if n > 1 else ""))
            prev, n = x, 1
    return " + ".join(out)


def legend(ax):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    y = 9.6
    ax.text(0, y, "Legend", fontsize=11, fontweight="bold", color=INK, va="top")
    y -= 0.85

    def row(draw, text):
        nonlocal y
        draw(y)
        ax.text(1.45, y, text, va="center", fontsize=8.2, color=INK)
        y -= 0.66

    sc = 0.0042  # metre -> legend unit

    def dish_l(yy):
        ax.add_patch(Wedge((0.62, yy), 0.32, 90, 270, fc=MW, ec="white", lw=0.6))
        ax.plot([0.62, 0.92], [yy, yy], color=MW, lw=1.6)
        ax.add_patch(Circle((0.95, yy), 0.06, fc=MW, ec="none"))
    row(dish_l, "IP-50 microwave unit (fixed dish, cannot re-aim)")
    def dn_l(yy):
        ax.add_patch(Circle((0.6, yy), 0.2, fc=TG, ec="white"))
        for k in range(4):
            a = math.radians(90 * k)
            cx, cy = 0.6 + 0.24 * math.cos(a), yy + 0.24 * math.sin(a)
            w, h = (0.09, 0.22) if k % 2 == 0 else (0.22, 0.09)
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=TG, ec="white", lw=0.5))
    row(dn_l, "TG node-class unit (DN, 4×90° sectors = 360°)")
    row(lambda yy: ax.add_patch(Rectangle((0.38, yy - 0.17), 0.34, 0.34, fc=TG, ec="white")),
        "TG terminal-class unit (CN, steers ±45°)")
    row(lambda yy: ax.add_patch(Wedge((0.1, yy), 1.0, -9, 9, fc=TG, alpha=0.35, ec=TG, lw=0.8)),
        "60 GHz beam (directivity)")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=MW, lw=3.2), "IP-50 microwave link (2 units)")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=TG, lw=2.4), "TG 60 GHz link to the hub")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=TG, lw=2.4, ls=(0, (5, 2.5))),
        "fixed TG link on a spare sector (no extra radio)")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=LOST, lw=1.5, ls=(0, (3, 3))),
        "link lost with a destroyed site")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=BRIDGE, lw=4.2),
        "MARL bridge, numbered (new TG link after re-steer)")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=CAND, lw=1.4, ls=(0, (1.5, 3))),
        "best alternative bridge (not chosen)")
    row(lambda yy: ax.add_patch(Rectangle((0.1, yy - 0.25), 1.0, 0.5, fc=ZONE, alpha=0.15, ec=ZONE,
                                          hatch="////", lw=0.5)), "disaster zone (sites destroyed)")
    row(lambda yy: (ax.plot([0.1, 1.1], [yy, yy], color=FLOW["F1"], lw=3),
                    ax.plot([0.1, 1.1], [yy - 0.18, yy - 0.18], color=FLOW["F1"], lw=2.2, ls=(0, (6, 3)))),
        "DPIR candidate a (solid) / b (dashed)")
    row(lambda yy: ax.plot([0.1, 1.1], [yy, yy], color=SVC, lw=2.2, ls=(0, (2, 1.5))),
        "pocket service flow (MARL only)")
    _ = sc


def config_figure(key):
    r = analyze(key)
    cfg = r["cfg"]
    nfl = len(cfg["flows"])
    fig = plt.figure(figsize=(22, 15.5), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.25], hspace=0.17, left=0.015, right=0.985,
                          top=0.865, bottom=0.075)
    top = gs[0].subgridspec(1, 3, wspace=0.03)
    bot = gs[1].subgridspec(1, nfl + 2, wspace=0.06, width_ratios=[1] * (nfl + 1) + [1.3])

    n_isl = len(r["islands"])
    lost = ", ".join(cfg["destroyed"])
    fig.text(0.015, 0.975, f"Configuration {cfg['name'].replace(' - ', ' — ')}",
             fontsize=20, fontweight="bold", color=INK, va="top")
    sub = (f"{len(r['pos'])} sites · {len(cfg['mw'])} microwave links ({cfg['ip50']} IP-50 units) · "
           f"{len(cfg['tg_links'])} TG links"
           + (f" + {len(cfg['tg_fixed'])} on spare sectors" if cfg.get("tg_fixed") else "")
           + f" ({cfg['tg']} TG units: "
           f"{sum(1 for v in cfg['tg_units'].values() if v == 'DN')} node, "
           f"{sum(1 for v in cfg['tg_units'].values() if v == 'CN')} terminal) · "
           f"{nfl} DPIR flows · event destroys {lost} ({r['lost_ip']} IP-50 + {r['lost_tg']} TG) → "
           f"{n_isl} islands · MARL forms {len(cfg['bridges'])} bridge{'s' if len(cfg['bridges']) > 1 else ''}")
    fig.text(0.015, 0.945, sub, fontsize=11.5, color=MUTED, va="top")

    a1, a2, a3 = (fig.add_subplot(top[0, i]) for i in range(3))
    panel_before(a1, r)
    panel_event(a2, r)
    panel_recovery(a3, r)
    n_feas = sum(c["feasible"] for c in r["cands"])
    for ax, n, t1, t2 in (
        (a1, "1", "Before the event",
         "DPIR: each flow can go via the core (IP-50) or the TG hub, whose sectors the flows share"),
        (a2, "2", f"Event: core severed, {lost} destroyed → {n_isl} islands",
         "IP-50 dishes are fixed; TG beams that pointed at the hub lose their peer; no route crosses"),
        (a3, "3", f"MARL recovery: {len(cfg['bridges'])} bridge{'s' if len(cfg['bridges']) > 1 else ''}",
         "policy: whether each site relays · planner: the peer · DN end ignites (core gone: local agents)"),
    ):
        ax.text(0.0, 1.075, n, transform=ax.transAxes, fontsize=13, fontweight="bold", color="white",
                ha="left", va="center", bbox=dict(boxstyle="circle,pad=0.3", fc=INK, ec="none"))
        ax.text(0.06, 1.075, t1, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=INK,
                va="center")
        ax.text(0.06, 1.035, t2, transform=ax.transAxes, fontsize=9.4, color=MUTED, va="center")

    edges0 = ([(a, b, "MW") for a, b in cfg["mw"]] + [(a, b, "TG") for a, b in cfg["tg_links"]]
              + [(a, b, "TG") for a, b in cfg.get("tg_fixed", [])])
    for i, (fid, s, t) in enumerate(cfg["flows"]):
        ax = fig.add_subplot(bot[0, i])
        ps = r["dpir"][fid]
        panel_flow(ax, r, fid, ps, edges0, f"DPIR {fid}: {s} → {t}")
        lines = []
        for j, p in enumerate(ps):
            lines.append(f"{'abc'[j]}  {'–'.join(p)}  ({tech_str(r['path_tech'](p))})")
        secs = sorted({r["end_sector"](x, y) for p in ps for a, b in zip(p, p[1:])
                       if r["g0"].edges[a, b]["tech"] == "TG"
                       for x, y in ((a, b), (b, a)) if r["end_sector"](x, y) in r["shared"]})
        if secs:
            lines.append("contends at " + ", ".join(
                f"{site} {['east', 'north', 'west', 'south'][sec]} sector" for site, sec in secs))
        ax.text(0.0, -0.01, "\n".join(lines), transform=ax.transAxes, fontsize=7.6, color=FLOW[fid],
                va="top", linespacing=1.35)
    axp = fig.add_subplot(bot[0, nfl])
    g = r["g_post"]
    edges_post = [(a, b, d["tech"]) for a, b, d in g.edges(data=True)]
    panel_flow(axp, r, None, r["post"], edges_post, "Routes after recovery (a tree: one route per flow)", post=True)
    lines = [f"{f if f in FLOW else 'svc'}  {'–'.join(p)}" for f, p in r["post"].items()]
    axp.text(0.0, -0.01, "\n".join(lines), transform=axp.transAxes,
             fontsize=7.6, color=INK, va="top", linespacing=1.35)
    axl = fig.add_subplot(bot[0, nfl + 1])
    legend(axl)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"config_{key}.{ext}"), dpi=150 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)
    return r


def overview():
    keys = list(CONFIGS)
    H = 6.0 * len(keys) + 1.2 + 3.4
    fig = plt.figure(figsize=(19, H), facecolor="white")
    gs = fig.add_gridspec(len(keys), 3, wspace=0.02, hspace=0.12, left=0.11, right=0.995,
                          top=1 - 1.0 / H, bottom=3.5 / H)
    fig.text(0.01, 1 - 0.25 / H, "Joint DPIR + MARL testbed — four equipment levels",
             fontsize=20, fontweight="bold", color=INK, va="top")
    legend(fig.add_axes([0.11, 0.2 / H, 0.42, 3.2 / H]))
    cols = ["1 · Before the event", "2 · Event → islands", "3 · MARL recovery"]
    for i, k in enumerate(keys):
        r = analyze(k)
        cfg = r["cfg"]
        axs = [fig.add_subplot(gs[i, j]) for j in range(3)]
        panel_before(axs[0], r)
        panel_event(axs[1], r)
        panel_recovery(axs[2], r, table=False)
        for j, ax in enumerate(axs):
            if i == 0:
                ax.set_title(cols[j], fontsize=13, fontweight="bold", color=INK, pad=6)
        y = axs[0].get_position()
        fig.text(0.01, y.y1 - 0.01, cfg["name"].split(" - ")[0], fontsize=30, fontweight="bold",
                 color=INK, va="top")
        fig.text(0.01, y.y1 - 0.045,
                 f"{cfg['ip50']} × IP-50\n{cfg['tg']} × TG\n\n{len(r['pos'])} sites\n"
                 f"{len(cfg['flows'])} DPIR flows\n{len(r['islands'])} islands\n"
                 f"{len(cfg['bridges'])} MARL bridge{'s' if len(cfg['bridges']) > 1 else ''}",
                 fontsize=11.5, color=MUTED, va="top", linespacing=1.35)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"overview.{ext}"), dpi=110 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for k in CONFIGS:
        config_figure(k)
    overview()
    print("written to", OUT)
