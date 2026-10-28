"""Exploded isometric technical illustration of each testbed option.

The bench is fully conducted (galvanic): microwave hops run through
attenuator chains on the floor, and the 60 GHz radios are cabled into a
V-band RF mesh (splitters + one programmable attenuator per possible link).
Devices are line-art drawings (hwdraw.py).  Every item drawn and counted
comes from the topology spec, the validator and bom.py, so the drawing
matches the bill of materials.

Run:  python3 iso_render.py    -> out/iso_{A,B,C,D}.png, out/iso_A_final.png (final plan only)
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import hwdraw as hw  # noqa: E402
from hwdraw import P, Scene  # noqa: E402
from bom import ITEMS, counts, rf_mesh  # noqa: E402
from topologies import CONFIGS, ROLE, ZONE_X, ZONE_Y  # noqa: E402
from validate import analyze  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
for f in font_manager.findSystemFonts():
    if "Inter" in os.path.basename(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10})

BG = "#fbfaf7"
INK = "#1d2430"
MUTED = "#5b6472"
RED = "#b42318"
COAX = "#3a3f47"
VCAB = "#c2410c"
FIBRE = "#0e9488"
CAT6 = "#8a94a6"
HALO = [pe.withStroke(linewidth=3, foreground=BG)]
CART = dict(w=1.25, d=0.85, h=0.72)

SHORT = {
    1: "IP-50 radio unit", 2: "Waveguide–coax transition", 3: "Fixed pad ~30 dB (MW)",
    4: "Programmable attenuator (MW)", 5: "Coax jumper (MW)", 6: "TG node, RF-port build",
    7: "TG terminal, RF-port build", 8: "V-band programmable attenuator", 9: "V-band splitter / combiner",
    10: "V-band fixed pad", 11: "V-band jumpers", 12: "PoE injector (TG)", 13: "Switched, metered PDU",
    14: "Site router / L3 switch", 15: "Edge agent PC (MARL actor)", 16: "Traffic host (iperf3 / TRex)",
    17: "Servers: DPIR GPU, TG controller", 18: "OOB log switch", 19: "PTP / GNSS clock",
    20: "−48 VDC supply (IP-50)", 21: "SFP+ + LC fibre", 22: "Cat6A cable, approx.",
    23: "MW + V-band power meters"}


def floor_xy(tx, ty):
    """Topology drawing units -> lab floor (north compressed so the bench reads as a plan)."""
    return (tx / 100.0 * 0.8, ty / 100.0 * 0.56)


def balloon(ax, anchor, at, n, z=950):
    ax.plot([anchor[0], at[0]], [anchor[1], at[1]], color=INK, lw=0.75, zorder=z, alpha=0.9)
    ax.add_patch(Circle(anchor, 0.04, fc=INK, ec="white", lw=0.5, zorder=z + 1))
    ax.text(at[0], at[1], str(n), ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=z + 2, bbox=dict(boxstyle="circle,pad=0.28", fc=INK, ec="white", lw=1.0))


def assembly(sc, x, y, z_from, z_to):
    sc.line2d([P(x, y, z_from), P(x, y, z_to)], color=MUTED, lw=0.7, d=sc.depth(x, y, z_to) - 0.02,
              ls=(0, (2, 2)), alpha=0.8)


def station(sc, cx, cy, kit, explode=0.34):
    """A lab cart with the site's kit exploded upward, tier by tier."""
    x0, y0 = cx - CART["w"] / 2, cy - CART["d"] / 2
    hw.lab_cart(sc, x0, y0, CART["w"], CART["d"], CART["h"])
    top = CART["h"]
    z = top + explode
    # tier 1: router + PDU
    w1 = 0.95 + 0.08 + 0.14
    px = cx - w1 / 2
    if kit["switch"]:
        sc.anchor(14, hw.router(sc, px, cy - 0.275, z))
        assembly(sc, px + 0.47, cy, z, top)
    sc.anchor(13, hw.pdu(sc, px + 1.03, cy - 0.425, z))
    assembly(sc, px + 1.1, cy, z, top)
    z += 0.09 + explode * 0.8
    # tier 2: agent PC, traffic host, PoE injector
    t2 = []
    if kit["agent"]:
        t2.append(("agent", 0.36))
    if kit["traffic"]:
        t2.append(("traffic", 0.36))
    if kit["tg"]:
        t2.append(("poe", 0.26))
    if t2:
        w2 = sum(w for _, w in t2) + 0.08 * (len(t2) - 1)
        px = cx - w2 / 2
        for name, w in t2:
            if name == "agent":
                sc.anchor(15, hw.mini_pc(sc, px, cy - 0.18, z))
            elif name == "traffic":
                sc.anchor(16, hw.mini_pc(sc, px, cy - 0.18, z, base="#cfe0d4", accent="#16a34a"))
            else:
                sc.anchor(12, hw.poe_injector(sc, px, cy - 0.09, z))
            assembly(sc, px + w / 2, cy, z, top)
            px += w + 0.08
        z += 0.13 + explode * 0.8
    # tier 3: radios
    t3 = [("ip50", 0.36)] * kit["ip50"]
    if kit["tg"]:
        t3.append(("tg", 0.42 if kit["tg"] == "DN" else 0.3))
    if t3:
        w3 = sum(w for _, w in t3) + 0.12 * (len(t3) - 1)
        px = cx - w3 / 2
        for name, w in t3:
            if name == "ip50":
                a1, a2 = hw.ip50(sc, px, cy - 0.15, z + 0.06)
                sc.anchor(1, a1)
                sc.anchor(2, a2)
            else:
                term = kit["tg"] == "CN"
                sc.anchor(7 if term else 6, hw.tg_node(sc, px, cy - 0.21 if not term else cy - 0.06, z + 0.08,
                                                       ports=kit["ports"], terminal=term))
            assembly(sc, px + w / 2, cy, z, top)
            px += w + 0.12
    return x0, y0


def site_kit(r, s, ports):
    cfg = r["cfg"]
    endpoints = {x for _, a, b in cfg["flows"] for x in (a, b)} | \
        {x for a, b in cfg.get("service", []) for x in (a, b)}
    agent = s in cfg["tg_units"] and s not in cfg["destroyed"]
    return dict(agent=agent, traffic=s in endpoints, tg=cfg["tg_units"].get(s), ip50=r["ip_units"][s],
                ports=ports, switch=bool(r["ip_units"][s] or agent or s in endpoints))


def render(key, final_only=False):
    r = analyze(key)
    cfg = r["cfg"]
    c = counts(cfg, r, final_only)
    mesh = rf_mesh(r, final_only)
    ports_per_site = {}
    for (site, _sec) in mesh["ports"]:
        ports_per_site[site] = ports_per_site.get(site, 0) + 1
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 0.565, 0.88])
    ax.set_aspect("equal")
    ax.axis("off")
    sc = Scene(ax)
    site_xy = {s: floor_xy(*r["pos"][s]) for s in cfg["sites"]}
    dead = set(cfg["destroyed"])

    # ---- floor and disaster zone ----------------------------------------------------
    xs = [p[0] for p in site_xy.values()]
    ys = [p[1] for p in site_xy.values()]
    fx0, fx1, fy0, fy1 = min(xs) - 2.4, max(xs) + 2.6, min(ys) - 2.4, max(ys) + 1.6
    floor = [(fx0, fy0, 0), (fx1, fy0, 0), (fx1, fy1, 0), (fx0, fy1, 0)]
    sc.poly2d([P(*p) for p in floor], "#f1efe9", ec="#d6d2c8", lw=1.0, d=-1e6)
    for k in range(math.floor(fx0), math.ceil(fx1) + 1):
        if fx0 <= k <= fx1:
            sc.line2d([P(k, fy0), P(k, fy1)], color="#e4e0d6", lw=0.6, d=-1e6 + 1)
    for k in range(math.floor(fy0), math.ceil(fy1) + 1):
        if fy0 <= k <= fy1:
            sc.line2d([P(fx0, k), P(fx1, k)], color="#e4e0d6", lw=0.6, d=-1e6 + 1)
    zc = [floor_xy(ZONE_X[0], ZONE_Y[0]), floor_xy(ZONE_X[1], ZONE_Y[0]),
          floor_xy(ZONE_X[1], ZONE_Y[1]), floor_xy(ZONE_X[0], ZONE_Y[1])]
    sc.poly2d([P(p[0], p[1], 0.005) for p in zc], "#e5484d", ec="#e5484d", lw=1.2, d=-1e6 + 2,
              alpha=0.13, hatch="////")

    rack = (max(xs) + 0.7, min(ys) - 1.35)   # V-band RF mesh, front-right of the sites
    corner = (fx0 + 1.1, fy0 + 1.0)          # control & measurement, front-left

    # ---- microwave hops: conducted runs with pads and a programmable attenuator --------
    for (a, b) in cfg["mw"]:
        (ax_, ay_), (bx_, by_) = site_xy[a], site_xy[b]
        L = math.hypot(bx_ - ax_, by_ - ay_)
        ux, uy = (bx_ - ax_) / L, (by_ - ay_) / L
        pa = (ax_ + ux * 0.9, ay_ + uy * 0.9)
        pb = (bx_ - ux * 0.9, by_ - uy * 0.9)
        mid = ((ax_ + bx_) / 2, (ay_ + by_) / 2)
        for (sx, sy), (ex, ey) in ((site_xy[a], pa), (site_xy[b], pb)):
            sc.cable([(sx, sy, 0.7), ((sx + ex) / 2, (sy + ey) / 2, 0.45), (ex, ey, 0.06)], COAX, lw=1.4)
        sc.anchor(5, sc.cable([(pa[0], pa[1], 0.06), (mid[0] - ux * 0.36, mid[1] - uy * 0.36, 0.06)],
                              COAX, lw=1.4))
        sc.cable([(mid[0] + ux * 0.36, mid[1] + uy * 0.36, 0.06), (pb[0], pb[1], 0.06)], COAX, lw=1.4)
        axis = "x" if abs(ux) > abs(uy) else "y"
        for (px, py) in (pa, pb):
            sc.anchor(3, hw.fixed_pad(sc, px - (0.1 if axis == "x" else 0), py - (0.1 if axis == "y" else 0),
                                      0.06, axis=axis))
        sc.anchor(4, hw.prog_attenuator(sc, mid[0] - 0.32, mid[1] - 0.22, 0.0))

    # ---- V-band jumpers from every TG cart to the mesh rack ---------------------------------
    for s in cfg["sites"]:
        if s not in cfg["tg_units"]:
            continue
        cx, cy = site_xy[s]
        sx, sy = cx + CART["w"] / 2, cy - CART["d"] / 2 + 0.25
        end = (rack[0] + 0.45, rack[1] + 0.8)
        sc.anchor(11, sc.cable([(sx, sy, 0.7), (sx + 0.2, sy, 0.1), ((sx + end[0]) / 2, sy, 0.04),
                                (end[0], (sy + end[1]) / 2, 0.04), (end[0], end[1], 0.3)],
                               VCAB, lw=0.7 + 0.18 * ports_per_site.get(s, 1), outline="#7c2d12"))

    # ---- site carts ------------------------------------------------------------------------
    for s in sorted(cfg["sites"], key=lambda s: -(site_xy[s][0] + site_xy[s][1])):
        cx, cy = site_xy[s]
        x0, y0 = station(sc, cx, cy, site_kit(r, s, ports_per_site.get(s, 0)))
        ax.annotate(s, P(cx, y0 - 0.05, 0.0), xytext=(0, -10), textcoords="offset points", ha="center",
                    va="top", fontsize=12, fontweight="bold", color=INK if s not in dead else RED,
                    zorder=900, path_effects=HALO)
        ax.annotate(ROLE[s].split("\n")[0] + (" · off at t0" if s in dead else ""), P(cx, y0 - 0.05, 0.0),
                    xytext=(0, -25), textcoords="offset points", ha="center", va="top", fontsize=8.5,
                    color=MUTED if s not in dead else RED, zorder=900, path_effects=HALO)

    # ---- the V-band RF mesh rack -------------------------------------------------------------
    rx_, ry_ = rack
    d0 = hw.rack_frame(sc, rx_, ry_, 0.9, 0.8, 1.6)
    n_mod = min(5, max(2, math.ceil(c["tg_paths"] / 4)))
    for i in range(n_mod):  # attenuator modules, exploded forward out of the rack
        b = hw.rack_unit(sc, rx_ + 0.02, ry_ - 0.55, 0.15 + i * 0.27, 0.86, 0.6, 0.14, "#f0a35e", kind="rf",
                         d=d0 + 0.5 + i * 0.001)
        sc.anchor(8, sc.fpt(b, "R", 0.4, 0.07))
        sc.line2d([P(rx_ + 0.45, ry_ - 0.25, 0.22 + i * 0.27), P(rx_ + 0.45, ry_ + 0.1, 0.22 + i * 0.27)],
                  color=MUTED, lw=0.6, ls=(0, (2, 2)), d=d0 + 0.4)
    sc.anchor(9, hw.splitter(sc, rx_ + 0.2, ry_ + 0.25, 1.75))
    sc.anchor(10, hw.fixed_pad(sc, rx_ + 0.3, ry_ + 0.35, 2.02, axis="x"))
    sc.line2d([P(rx_ + 0.45, ry_ + 0.4, 1.6), P(rx_ + 0.45, ry_ + 0.4, 1.75)], color=MUTED, lw=0.6,
              ls=(0, (2, 2)), d=d0)
    ax.annotate(f"V-band RF mesh · {c['tg_paths']} paths", P(rx_ + 0.45, ry_ - 0.7, 0), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top", fontsize=9.5, fontweight="bold",
                color=VCAB, zorder=900, path_effects=HALO)

    # ---- control & measurement corner ---------------------------------------------------------
    kx, ky = corner
    d1 = hw.rack_frame(sc, kx, ky, 0.9, 0.8, 1.6)
    for i in range(2):
        b = hw.rack_unit(sc, kx + 0.02, ky - 0.5, 0.3 + i * 0.42, 0.86, 0.7, 0.2, "#d7dbe2", kind="server",
                         d=d1 + 0.5 + i * 0.001)
        sc.anchor(17, sc.fpt(b, "R", 0.3, 0.1))
    b = hw.rack_unit(sc, kx + 0.02, ky + 0.05, 1.9, 0.86, 0.6, 0.07, "#4b5563", kind="ports", d=d1 + 0.6)
    sc.anchor(18, sc.fpt(b, "R", 0.4, 0.035))
    sc.line2d([P(kx + 0.45, ky + 0.35, 1.6), P(kx + 0.45, ky + 0.35, 1.9)], color=MUTED, lw=0.6,
              ls=(0, (2, 2)), d=d1)
    sc.anchor(19, hw.gnss_clock(sc, kx + 1.25, ky + 0.05, 0.0))
    sc.anchor(20, hw.dc_supply(sc, kx + 0.15, ky + 1.25, 0.0))
    sc.anchor(23, hw.power_meter(sc, kx + 1.2, ky + 1.1, 0.0))
    ax.annotate("Control & measurement", P(kx + 0.45, ky + 0.4, 1.95), xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK, zorder=900,
                path_effects=HALO)

    sc.flush()
    corners = [P(*p) for p in floor] + [P(fx0, fy1, 2.9), P(fx1, fy1, 2.9)]
    X = [q[0] for q in corners]
    Y = [q[1] for q in corners]
    ax.set_xlim(min(X) - 0.9, max(X) + 0.9)
    ax.set_ylim(min(Y) - 1.0, max(Y) + 0.2)

    # ---- scene callouts: system-level parts ----------------------------------------------------
    scene_nums = [17, 18, 19, 20, 23, 8, 9, 10, 11, 4, 3, 5, 1, 6, 7]
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    chosen = []
    for n in scene_nums:
        pts = sc.anchors.get(n, [])
        if pts:
            chosen.append((n, max(pts, key=lambda p: min([math.dist(p, q) for _, q in chosen] or [99]))))
    cx_ = sum(p[0] for _, p in chosen) / len(chosen)
    cy_ = sum(p[1] for _, p in chosen) / len(chosen)
    spots = []
    for n, a in chosen:
        vx, vy = a[0] - cx_, a[1] - cy_
        m = math.hypot(vx, vy) or 1
        spots.append([n, a, a[0] + vx / m * 1.1, a[1] + vy / m * 1.1 + 0.25])
    for _ in range(120):
        moved = False
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                dx = spots[i][2] - spots[j][2]
                dy = spots[i][3] - spots[j][3]
                dd = math.hypot(dx, dy)
                if dd < 0.8:
                    k = (0.8 - dd) / 2 / (dd or 1)
                    spots[i][2] += dx * k + 0.01
                    spots[i][3] += dy * k
                    spots[j][2] -= dx * k
                    spots[j][3] -= dy * k
                    moved = True
        if not moved:
            break
    for n, a, bx, by in spots:
        balloon(ax, a, (min(max(bx, xl + 0.3), xr - 0.3), min(max(by, yb + 0.3), yt - 0.3)), n)

    # ---- inset: a fully equipped site cart, exploded ----------------------------------------------
    ins = fig.add_axes([0.567, 0.36, 0.175, 0.515])
    ins.set_aspect("equal")
    ins.axis("off")
    ins.add_patch(Rectangle((0, 0), 1, 1, transform=ins.transAxes, fc="#ffffff", ec="#d6d2c8", lw=1.0,
                            zorder=0))
    s2 = Scene(ins)
    x0, y0 = station(s2, 0, 0, dict(agent=True, traffic=True, tg="DN", ip50=2, ports=3, switch=True), explode=0.42)
    s2.anchor(3, hw.fixed_pad(s2, x0 + 1.45, y0 + 0.15, 0.06, axis="x"))
    s2.cable([(x0 + CART["w"], y0 + 0.15, 0.68), (x0 + 1.4, y0 + 0.15, 0.3), (x0 + 1.45, y0 + 0.15, 0.06)],
             COAX, lw=1.4)
    s2.anchor(5, s2.cable([(x0 + 1.65, y0 + 0.15, 0.06), (x0 + 2.3, y0 + 0.15, 0.04)], COAX, lw=1.4))
    s2.anchor(11, s2.cable([(x0 + CART["w"], y0 + 0.6, 0.68), (x0 + 1.5, y0 + 0.6, 0.2),
                            (x0 + 2.3, y0 + 0.6, 0.04)], VCAB, lw=2.0, outline="#7c2d12"))
    s2.anchor(22, s2.cable([(x0, y0 + 0.75, 0.68), (x0 - 0.3, y0 + 0.75, 0.3), (x0 - 0.75, y0 + 0.75, 0.04)],
                           CAT6, lw=1.4, outline="#5b6472"))
    s2.anchor(21, s2.cable([(x0, y0 + 0.45, 0.6), (x0 - 0.3, y0 + 0.45, 0.25), (x0 - 0.75, y0 + 0.45, 0.04)],
                           FIBRE, lw=1.2, outline="#065f56"))
    s2.flush()
    ins.set_xlim(-2.6, 2.9)
    ins.set_ylim(-1.0, 4.1)
    ins.text(0.05, 0.975, "Fully equipped site cart, exploded", transform=ins.transAxes, fontsize=12,
             fontweight="bold", color=INK, va="top")
    ins.text(0.05, 0.915, "each site carries what its role needs", transform=ins.transAxes,
             fontsize=9, color=MUTED, va="top")
    pts = sorted((lst[0][1], n, lst[0]) for n, lst in s2.anchors.items())
    left = {22, 21, 13, 14, 15}
    yl, yr = -0.8, -0.8
    for y, n, a in pts:
        if n in left:
            yl = max(yl + 0.42, y)
            balloon(ins, a, (-2.25, yl), n)
        else:
            yr = max(yr + 0.42, y)
            balloon(ins, a, (2.55, yr), n)

    # ---- line key -------------------------------------------------------------------------------
    key_ax = fig.add_axes([0.567, 0.03, 0.175, 0.31])
    key_ax.axis("off")
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.add_patch(Rectangle((0, 0), 1, 1, fc="#ffffff", ec="#d6d2c8", lw=1.0))
    key_ax.text(0.06, 0.93, "Key", fontsize=12.5, fontweight="bold", color=INK, va="top")
    rows = [(COAX, 2.4, "-", "microwave coax (conducted)"),
            (VCAB, 3.0, "-", "V-band jumpers to the RF mesh"),
            (CAT6, 2.2, "-", "Cat6A (PoE + data)"),
            (FIBRE, 2.0, "-", "LC fibre (TG uplink)"),
            (MUTED, 1.0, (0, (2, 2)), "assembly line (exploded part)"),
            (None, None, None, "disaster zone (off at t0)")]
    yy = 0.78
    for col, lw, ls, txt in rows:
        if col:
            key_ax.plot([0.06, 0.24], [yy, yy], color=col, lw=lw, ls=ls, solid_capstyle="round")
        else:
            key_ax.add_patch(Rectangle((0.06, yy - 0.04), 0.18, 0.08, fc="#e5484d", alpha=0.18, ec="#e5484d",
                                       hatch="////", lw=0.6))
        key_ax.text(0.29, yy, txt, fontsize=9.5, color=INK, va="center")
        yy -= 0.13

    # ---- title and parts list -----------------------------------------------------------------
    fig.text(0.02, 0.965, f"Option {key} — {cfg['ip50']} IP-50 + {cfg['tg']} TG · conducted bench",
             fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.918, "Exploded isometric · not to scale · every radio is cabled (galvanic): "
             "microwave through attenuator chains, 60 GHz through the V-band RF mesh",
             fontsize=11.5, color=MUTED, va="top")
    fig.text(0.02, 0.891, "Red floor = disaster zone: at t0 the switched PDUs cut CORE and HUB and the "
             "attenuators to them go to maximum", fontsize=11.5, color=RED, va="top")
    px = fig.add_axes([0.75, 0.03, 0.24, 0.875])
    px.axis("off")
    px.set_xlim(0, 1)
    px.set_ylim(0, 1)
    px.add_patch(Rectangle((0, 0), 1, 1, fc="#ffffff", ec="#d6d2c8", lw=1.0))
    px.text(0.05, 0.975, "Parts list", fontsize=14, fontweight="bold", color=INK, va="top")
    px.text(0.95, 0.975, "Qty", fontsize=11, fontweight="bold", color=INK, va="top", ha="right")
    y = 0.925
    step = 0.875 / len(ITEMS)
    for n, _g, _item, rule, _note in ITEMS:
        q = rule(c)
        absent = q in (0, "0")
        col = INK if not absent else "#a3a9b3"
        px.text(0.06, y, f"{n}", fontsize=9, fontweight="bold", color="white" if not absent else "#a3a9b3",
                va="center", ha="center",
                bbox=dict(boxstyle="circle,pad=0.25", fc=INK if not absent else "#eef0f3", ec="none"))
        px.text(0.13, y, SHORT[n], fontsize=10, color=col, va="center")
        px.text(0.95, y, str(q).replace(" / ", " + "), fontsize=10.5, fontweight="bold", color=col,
                va="center", ha="right")
        y -= step
    px.text(0.05, 0.012, "Carts, power cords and PoE leads not counted.\nQuantities from bom.py.",
            fontsize=8.5, color=MUTED, va="bottom")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"iso_{key}{'_final' if final_only else ''}.png")
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


if __name__ == "__main__":
    for k in CONFIGS:
        print(render(k))
    print(render("A", final_only=True))  # selected option, final plan cabled only
