"""Animated topology per option: normal -> broken -> reconnected (GIF).

Uses the same analysis and drawing primitives as render.py, so the
animation shows exactly the network, islands, beam turns and bridges of the
static figures.

Run:  python3 animate.py      -> out/anim_{A,B,C,D}.gif
"""
import io
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge  # noqa: E402
from PIL import Image  # noqa: E402

import render as R  # noqa: E402
from topologies import CONFIGS, ZONE_X, ZONE_Y  # noqa: E402
from validate import analyze, bearing  # noqa: E402

OUT = R.OUT
FPS = 8
# timeline in seconds
T_NORMAL, T_EVENT, T_BROKEN, T_STEER, T_RECON, T_HOLD = 5.0, 2.5, 4.0, 7.5, 5.0, 1.0
TURN = 0.7   # share of T_STEER in which the beams turn; the bridges form in the rest
T_TOTAL = T_NORMAL + T_EVENT + T_BROKEN + T_STEER + T_RECON + T_HOLD


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def lerp_color(c1, c2, k):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(a[i] + (b[i] - a[i]) * k):02x}" for i in range(3))


def polyline_point(pts, s):
    """Point at arc length s along a polyline."""
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if s <= L:
            k = s / L if L else 0
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        s -= L
    return pts[-1]


def poly_len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def traffic(ax, r, routes, t, alpha=1.0):
    """Packets moving along each route, coloured by flow."""
    pos = r["pos"]
    n = len(routes)
    for i, (fid, path) in enumerate(routes):
        pts = R.offset_poly([pos[s] for s in path], (i - (n - 1) / 2) * 30)
        L = poly_len(pts)
        col = R.FLOW.get(fid, R.SVC)
        spacing = 230
        for k in range(int(L // spacing) + 1):
            s = (t * 220 + k * spacing) % L
            p = polyline_point(pts, s)
            ax.add_patch(Circle(p, 22, fc=col, ec="white", lw=1.2, alpha=alpha, zorder=15))
        # a faint guide of the route under the packets
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, lw=2.0, alpha=0.28 * alpha, zorder=1.8,
                solid_capstyle="round")


def draw(ax, r, t):
    cfg, pos = r["cfg"], r["pos"]
    dead_sites = set(cfg["destroyed"])
    t1 = T_NORMAL
    t2 = t1 + T_EVENT
    t3 = t2 + T_BROKEN
    t4 = t3 + T_STEER
    ev = ease((t - t1) / T_EVENT)              # 0 -> 1 during the event
    st = ease((t - t3) / (TURN * T_STEER))     # 0 -> 1 while the beams turn
    isl = ease((t - t2) / 1.2)                 # islands fade in
    rec = ease((t - t4) / 1.0)                 # one-network hull fades in
    xs = [p[0] for p in pos.values()]
    ys_ = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 330, max(xs) + 360)
    ax.set_ylim(min(ys_) - 260, max(ys_) + 200)
    ax.set_aspect("equal")
    ax.axis("off")

    # disaster zone: planned outline, then the red band
    ys = [p[1] for p in pos.values()]
    y0, y1 = max(ZONE_Y[0], min(ys) - 60), min(ZONE_Y[1], max(ys) + 60)
    w = ZONE_X[1] - ZONE_X[0]
    ax.add_patch(Rectangle((ZONE_X[0], y0), w, y1 - y0, fc="none", ec=R.ZONE, lw=1.2, ls=(0, (4, 3)),
                           alpha=0.6 * (1 - ev), zorder=1))
    if ev > 0:
        ax.add_patch(Rectangle((ZONE_X[0], y0), w, y1 - y0, fc=R.ZONE, ec="none", alpha=0.09 * ev, zorder=0))
        ax.add_patch(Rectangle((ZONE_X[0], y0), w, y1 - y0, fc="none", ec=R.ZONE, lw=0, hatch="////",
                               alpha=0.2 * ev, zorder=0))

    # islands, then the one-network hull
    if isl > 0:
        fade = 1 - 0.65 * rec
        for k, isl_sites in enumerate(r["islands"]):
            hull = R.rounded_hull([pos[s] for s in isl_sites], 150)
            col = R.ISLAND[k % len(R.ISLAND)]
            ax.add_patch(Polygon(hull, closed=True, fc=col, ec="none", alpha=0.10 * isl * fade, zorder=0.5))
            ax.add_patch(Polygon(hull, closed=True, fc="none", ec=col, lw=1.6, ls=(0, (5, 3)),
                                 alpha=0.9 * isl * fade, zorder=0.6))
            if rec < 0.5:
                top = max(p[1] for p in hull)
                yy, va = (top + 14, "bottom") if isl_sites != ["S"] else (min(p[1] for p in hull) - 14, "top")
                ax.text(sum(pos[s][0] for s in isl_sites) / len(isl_sites), yy,
                        f"Island {k + 1}", ha="center", va=va, fontsize=12, fontweight="bold", color=col,
                        alpha=isl * (1 - 2 * rec), zorder=22)
    if rec > 0:
        surv = [s for s in pos if s not in dead_sites]
        hull = R.rounded_hull([pos[s] for s in surv], 170)
        ax.add_patch(Polygon(hull, closed=True, fc=R.BRIDGE, ec="none", alpha=0.06 * rec, zorder=0.4))
        ax.add_patch(Polygon(hull, closed=True, fc="none", ec=R.BRIDGE, lw=1.6, ls=(0, (5, 3)),
                             alpha=0.85 * rec, zorder=0.45))

    # links
    tg_all = list(cfg["tg_links"]) + list(cfg.get("tg_fixed", []))
    for a, b in list(cfg["mw"]) + tg_all:
        dead = a in dead_sites or b in dead_sites
        tech = "MW" if (a, b) in cfg["mw"] else "TG"
        base = R.MW if tech == "MW" else R.TG
        lw = 3.4 if tech == "MW" else 2.6
        ls = "-" if (a, b) not in cfg.get("tg_fixed", []) else (0, (5, 2.5))
        if dead and ev > 0:
            col = lerp_color(base, R.LOST, ev)
            R.link_line(ax, pos[a], pos[b], col, lw - (lw - 1.5) * ev, ls=(0, (3, 3)) if ev > 0.5 else ls,
                        alpha=1 - 0.45 * ev - 0.3 * rec)
        else:
            R.link_line(ax, pos[a], pos[b], base, lw, ls=ls)

    # beams: before the event every TG end points at its peer
    for a, b in tg_all:
        for x, y in ((a, b), (b, a)):
            if x in dead_sites and ev >= 1:
                continue
            if y in dead_sites and ev > 0:
                # beam toward a lost peer: greys out; later it is the one that re-steers
                st_entry = [s for s in r["steer"] if s["unit"] == x and s["old"] == y]
                if st_entry and st > 0:
                    continue
                R.beam(ax, pos[x], pos[y], color=lerp_color(R.TG, "#9ca3af", ev), fill=ev < 0.6,
                       ls=(0, (3, 2)) if ev >= 0.6 else "-", alpha=0.30, length=210)
            else:
                R.beam(ax, pos[x], pos[y])

    # MARL: each surviving beam turns, slowly, from where it pointed to its new peer
    steering = t3 <= t < t4 + 0.5
    for s_ in r["steer"]:
        x = pos[s_["unit"]]
        if t < t3:
            continue
        a1 = s_["b_new"]
        # a beam that served a lost peer starts there; an idle sector starts at its boresight
        a0 = s_["b_old"] if s_["old"] is not None else s_["sector"] * 90.0
        d = (a1 - a0 + 180) % 360 - 180
        ang = a0 + d * st
        if steering:   # pulse ring on the steering site
            pr = 78 + 10 * math.sin((t - t3) * 4)
            ax.add_patch(Circle(x, pr, fc="none", ec=R.BRIDGE, lw=2.0, alpha=0.6 * (1 - 0.6 * ease(t - t4)),
                                zorder=8))
        for j, back in enumerate((0.0, 0.08, 0.16, 0.24)):  # motion trail
            stb = ease((t - t3) / (TURN * T_STEER) - back)
            if j and (st >= 1 or stb <= 0):
                continue
            aj = a0 + d * stb
            col = lerp_color("#f59e0b", R.BRIDGE, st)
            ax.add_patch(Wedge(x, 300, aj - 8, aj + 8, fc=col, ec="none", alpha=(0.6 if j == 0 else 0.18 / j),
                               zorder=4))
        if s_["old"] is not None and st < 1:
            ax.add_patch(Wedge(x, 210, a0 - 8, a0 + 8, fc="none", ec="#9ca3af", lw=1.0, ls=(0, (3, 2)),
                               alpha=0.8 * (1 - st), zorder=4))
        if 0.02 < st and t < t4:
            rad = 165
            p0 = (x[0] + rad * math.cos(math.radians(a0)), x[1] + rad * math.sin(math.radians(a0)))
            p1 = (x[0] + rad * math.cos(math.radians(ang)), x[1] + rad * math.sin(math.radians(ang)))
            if abs(d * st) > 3:
                ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={0.3 if d > 0 else -0.3}",
                                             arrowstyle="-|>,head_length=6,head_width=3.5", color=R.BRIDGE,
                                             lw=1.8, zorder=13))
            am = math.radians(a0 + d * st / 2)
            ax.text(x[0] + (rad + 55) * math.cos(am), x[1] + (rad + 55) * math.sin(am), f"{abs(d) * st:.0f}°",
                    ha="center", va="center", fontsize=10.5, fontweight="bold", color=R.BRIDGE, zorder=23,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    grow = ease((t - t3 - TURN * T_STEER) / ((1 - TURN) * T_STEER))
    for i, (h, p) in enumerate(cfg["bridges"], 1):
        if grow <= 0:
            continue
        a, b = pos[h], pos[p]
        # grows from both ends toward the middle: two beams meeting
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        for e in (a, b):
            q = (e[0] + (m[0] - e[0]) * grow, e[1] + (m[1] - e[1]) * grow)
            ax.plot([e[0], q[0]], [e[1], q[1]], color=R.BRIDGE, lw=4.6, zorder=2.5, solid_capstyle="round")
        if grow >= 1:
            lp = R.bridge_label_point(a, b)
            ax.text(lp[0], lp[1], str(i), ha="center", va="center", fontsize=11, color="white", fontweight="bold",
                    zorder=23, bbox=dict(boxstyle="circle,pad=0.25", fc=R.BRIDGE, ec="white", lw=1.2))

    # sites (crossed out once the event has hit)
    R.draw_sites(ax, r, dead=dead_sites if ev >= 0.5 else ())
    R.labels(ax, r, dead=dead_sites if ev >= 0.5 else (), small=True)

    # traffic
    if t < t1 + 0.6:
        routes = [(fid, ps[0]) for fid, ps in r["dpir"].items()] + \
            [("svc", p) for p in r["service"].values()]
        traffic(ax, r, routes, t, alpha=1 - ease((t - t1) / 0.6))
    if t > t4:
        routes = [(f if f in R.FLOW else "svc", p) for f, p in r["post"].items()]
        traffic(ax, r, routes, t, alpha=ease((t - t4) / 0.8))


STEPS = [("1", "Normal", "DPIR routes the flows via the core and the TG hub"),
         ("2", "Broken", "core and TG hub lost: islands, no route crosses"),
         ("3", "Reconnected", "TG beams re-steer; MARL bridges rejoin the islands")]


def step_of(t):
    if t < T_NORMAL:
        return 0
    if t < T_NORMAL + T_EVENT + T_BROKEN:
        return 1
    return 2


def side_panel(fig, r, k, t):
    from matplotlib.lines import Line2D  # noqa: F401
    cfg = r["cfg"]
    px = fig.add_axes([0.60, 0.05, 0.38, 0.76])
    px.axis("off")
    px.set_xlim(0, 1)
    px.set_ylim(0, 1)
    px.add_patch(Rectangle((0, 0), 1, 1, fc="#fbfaf7", ec="#e3e0d8", lw=1.0))
    y = 0.93
    col = [R.INK, R.LOST, R.BRIDGE][k]
    head = ["Before the event", "After the event", "After MARL recovery"][k]
    px.text(0.05, y, head, fontsize=14, fontweight="bold", color=col, va="top")
    y -= 0.09
    lines = []
    if k == 0:
        for fid, s_, t_ in cfg["flows"]:
            ps = r["dpir"][fid]
            lines.append((R.FLOW[fid], f"{fid}  {s_} → {t_}"))
            for j, p in enumerate(ps):
                via = "core, IP-50" if "CORE" in p else "TG hub"
                lines.append((R.MUTED, f"      {'ab'[j]}: {'–'.join(p)}  ({via})"))
        for (s_, t_), p in r["service"].items():
            lines.append((R.SVC, f"svc  {s_} → {t_}  (pocket, MARL only)"))
        lines.append((R.MUTED, "The flows share the hub's sector airtime."))
    elif k == 1:
        lines.append((R.LOST, "Lost: CORE (core + MW hub), HUB (TG hub)"))
        for i, isl in enumerate(r["islands"]):
            lines.append((R.ISLAND[i % len(R.ISLAND)], f"Island {i + 1}:  {', '.join(isl)}"))
        lines.append((R.MUTED, "Microwave dishes are fixed; no route crosses."))
        lines.append((R.MUTED, "TG beams that served the hub lose their peer."))
    elif t < T_NORMAL + T_EVENT + T_BROKEN + T_STEER:
        head = "MARL: beams re-steer"
        px.texts[-1].set_text(head)
        for s_ in r["steer"]:
            a0 = s_["b_old"] if s_["old"] is not None else s_["sector"] * 90.0
            dlt = abs((s_["b_new"] - a0 + 180) % 360 - 180)
            frm = f"from {s_['old']}" if s_["old"] else "from an idle sector"
            lines.append((R.BRIDGE, f"{s_['unit']}: beam turns {dlt:.0f}° {frm} → {s_['new']}"))
        lines.append((R.MUTED, "The policy decides whether each site relays;"))
        lines.append((R.MUTED, "the planner picks the peer; the node ignites."))
    else:
        for i, (h, p) in enumerate(cfg["bridges"], 1):
            c = [c for c in r["cands"] if {c["a"], c["b"]} == {h, p}][0]
            lines.append((R.BRIDGE, f"Bridge {i}:  {h} ⇄ {p}   {c['d']:.0f} m · {c['budget']['mcs']}"))
        for f, p in r["post"].items():
            lines.append((R.FLOW.get(f, R.SVC), f"{f if f in R.FLOW else 'svc'}  {'–'.join(p)}"))
        lines.append((R.MUTED, "Policy: whether a site relays · planner: the peer"))
    for c_, txt in lines:
        px.text(0.05, y, txt, fontsize=11, color=c_, va="top",
                fontweight="bold" if c_ not in (R.MUTED,) and not txt.startswith(" ") else "normal")
        y -= 0.058
    # legend
    y = 0.27
    px.text(0.05, y, "Key", fontsize=11, fontweight="bold", color=R.INK, va="center")
    items = [(R.MW, 3.2, "-", "IP-50 microwave link (fixed)"), (R.TG, 2.4, "-", "TG 60 GHz link"),
             (R.TG, 2.4, (0, (5, 2.5)), "TG link on a spare sector"), (R.LOST, 1.5, (0, (3, 3)), "lost link"),
             (R.BRIDGE, 4.0, "-", "MARL bridge")]
    for i, (c_, lw, ls, txt) in enumerate(items):
        yy = y - 0.05 * (i + 1)
        px.plot([0.05, 0.15], [yy, yy], color=c_, lw=lw, ls=ls)
        px.text(0.18, yy, txt, fontsize=10, color=R.INK, va="center")


def render_frame(r, t, key):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 0.6, 0.84])
    draw(ax, r, t)
    cfg = r["cfg"]
    k = step_of(t)
    fig.text(0.025, 0.965, f"Option {key} — {cfg['ip50']} IP-50 + {cfg['tg']} TG", fontsize=20,
             fontweight="bold", color=R.INK, va="top")
    x = 0.025
    for i, (num, name, _) in enumerate(STEPS):
        on = i == k
        col = [R.INK, R.LOST, R.BRIDGE][i]
        fig.text(x, 0.885, f" {num} ", fontsize=13, fontweight="bold", color="white", va="center",
                 bbox=dict(boxstyle="circle,pad=0.25", fc=col if on else "#d1d5db", ec="none"))
        fig.text(x + 0.028, 0.885, name, fontsize=15, fontweight="bold" if on else "normal",
                 color=col if on else "#9ca3af", va="center")
        x += 0.028 + 0.012 * len(name) + 0.035
    fig.text(0.60, 0.885, STEPS[k][2], fontsize=12.5, color=R.MUTED, va="center")
    side_panel(fig, r, k, t)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_gif(key):
    r = analyze(key)
    n = int(T_TOTAL * FPS)
    frames = [render_frame(r, i / FPS, key) for i in range(n)]
    pal = frames[min(len(frames) - 1, int((T_TOTAL - 2) * FPS))].quantize(colors=128, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    path = os.path.join(OUT, f"anim_{key}.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=True,
              disposal=1)
    return path, os.path.getsize(path)


if __name__ == "__main__":
    for k in CONFIGS:
        print(*make_gif(k))
