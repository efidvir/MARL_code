"""What changed from option A to A7 (option A on the delivered N367 kit), drawn.

Two maps side by side, each node drawn with the 90-degree sectors it has:
filled where a sector carries a link, outlined where it is a spare (or, on a
4-sector node, unused).  Bridges are drawn three ways: the plan chosen,
others MARL could still pick, and (A7 only) those the 2-sector nodes can no
longer reach.  Numbered markers on the A7 map match the list on the right.

Run:  python3 kit_change.py    -> out/kit_change.png
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle, Wedge  # noqa: E402

from render import (BRIDGE, CAND, INK, LOST, MUTED, MW, OUT, TG, ZONE, dish, link_line,  # noqa: E402
                    site_base, tg_icon)
from topologies import ZONE_X, ZONE_Y  # noqa: E402
from validate import analyze  # noqa: E402

R_SECTOR = 150
HALO = [pe.withStroke(linewidth=3, foreground="white")]
SHORT = {"W1": "W1", "W2": "W2", "E1": "E1", "E2": "E2", "CORE": "CORE", "HUB": "HUB",
         "S": "S shelter", "N": "N hospital"}
NAME_AT = {"W1": (-30, 0, "right"), "W2": (-30, 0, "right"), "E1": (30, 0, "left"), "E2": (30, 0, "left"),
           "CORE": (28, 14, "left"), "HUB": (30, -16, "left"), "S": (30, -10, "left"), "N": (30, 10, "left")}


def used_sectors(r):
    u = {s: set() for s in r["cfg"]["tg_units"]}
    for a, b in list(r["tg_all"]) + list(r["cfg"]["bridges"]):
        for x, y in ((a, b), (b, a)):
            u[x].add(r["end_sector"](x, y)[1])
    return u


def sector_wedges(ax, r, s, used, show_unused=True):
    xy = r["pos"][s]
    kind = r["cfg"]["tg_units"][s]
    if kind == "CN":  # a terminal steers +/-45 deg around its mounting azimuth
        m = r["cn_mount"][s]
        ax.add_patch(Wedge(xy, R_SECTOR, m - 45, m + 45, fc=TG, ec=TG, alpha=0.22, lw=1.0, zorder=3))
        return
    for k in r["installed"][s]:
        th = 90 * k
        if k in used[s]:
            ax.add_patch(Wedge(xy, R_SECTOR, th - 45, th + 45, fc=TG, ec=TG, alpha=0.24, lw=1.0, zorder=3))
        elif show_unused:
            ax.add_patch(Wedge(xy, R_SECTOR, th - 45, th + 45, fc="none", ec="#9aa3b2", lw=1.1,
                               ls=(0, (3, 2.5)), zorder=3))


def panel(ax, key, title, sub, marks=False):
    r = analyze(key)
    cfg, pos = r["cfg"], r["pos"]
    used = used_sectors(r)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 330, max(xs) + 420)
    ax.set_ylim(min(ys) - 200, max(ys) + 190)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=11.5, color=MUTED, va="bottom")
    # disaster zone
    y0, y1 = max(ZONE_Y[0], min(ys) - 60), min(ZONE_Y[1], max(ys) + 60)
    ax.add_patch(Rectangle((ZONE_X[0], y0), ZONE_X[1] - ZONE_X[0], y1 - y0, fc=ZONE, ec="none", alpha=0.05,
                           zorder=0))
    ax.add_patch(Rectangle((ZONE_X[0], y0), ZONE_X[1] - ZONE_X[0], y1 - y0, fc="none", ec=ZONE, lw=1.0,
                           ls=(0, (4, 3)), alpha=0.55, zorder=0))
    ax.text(ZONE_X[0] + 16, y0 + 30, "destroyed at t0", fontsize=9, color=ZONE, rotation=90, va="bottom",
            ha="left", alpha=0.9)

    # bridges: chosen, still possible, out of reach
    chosen = {frozenset(b) for b in cfg["bridges"]}
    reach = {frozenset((c["a"], c["b"])) for c in r["mount_reach"]}
    for c in r["cands"]:
        if not c["feasible"]:
            continue
        e = frozenset((c["a"], c["b"]))
        a, b = pos[c["a"]], pos[c["b"]]
        if e in chosen:
            continue
        if e in reach:
            link_line(ax, a, b, CAND, 1.6, ls=(0, (1.5, 3.0)), z=2)
        else:
            link_line(ax, a, b, LOST, 1.6, ls=(0, (1.5, 3.0)), alpha=0.85, z=2)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ax.text(mx, my, "×", fontsize=17, color=LOST, ha="center", va="center", fontweight="bold",
                    zorder=9, path_effects=HALO)
    for a, b in cfg["bridges"]:
        link_line(ax, pos[a], pos[b], BRIDGE, 4.2, z=4)
    # fixed links
    for a, b in cfg["mw"]:
        link_line(ax, pos[a], pos[b], MW, 3.6, z=5)
    for a, b in r["tg_all"]:
        link_line(ax, pos[a], pos[b], TG, 3.0, z=5)
    # sectors, sites, devices
    for s in cfg["tg_units"]:
        sector_wedges(ax, r, s, used)
    for s, xy in pos.items():
        site_base(ax, xy, has_tg=s in cfg["tg_units"])
        if s in cfg["tg_units"]:
            tg_icon(ax, xy, cfg["tg_units"][s], sectors=r["installed"][s])
        for a, b in cfg["mw"]:
            if s in (a, b):
                dish(ax, xy, pos[b if s == a else a])
        dx, dy, ha = NAME_AT[s]
        ax.annotate(SHORT[s], xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=12.5, fontweight="bold", color=INK, zorder=20, path_effects=HALO)
    return r


def marker(ax, xy, n, dx=0, dy=0):
    ax.annotate(str(n), xy, xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=30,
                bbox=dict(boxstyle="circle,pad=0.32", fc=INK, ec="white", lw=1.2))


def legend(fig):
    ax = fig.add_axes([0.02, 0.015, 0.66, 0.075])
    ax.set_xlim(0, 66)
    ax.set_ylim(0, 2)
    ax.axis("off")
    items = [
        ("wedge_used", "sector in use"), ("wedge_spare", "sector with no link (spare)"),
        ("tg", "TG link before the event"), ("mw", "IP-50 link"),
        ("bridge", "bridge in the drawn plan"), ("cand", "other bridge MARL can pick"),
        ("lost", "bridge out of reach"),
    ]
    for i, (kind, text) in enumerate(items):
        x, y = 0.6 + 16.5 * (i % 4), (1.45 if i < 4 else 0.45)
        if kind == "wedge_used":
            ax.add_patch(Wedge((x + 0.4, y - 0.35), 1.0, 45, 135, fc=TG, ec=TG, alpha=0.3))
        elif kind == "wedge_spare":
            ax.add_patch(Wedge((x + 0.4, y - 0.35), 1.0, 45, 135, fc="none", ec="#9aa3b2", ls=(0, (3, 2.5))))
        else:
            col, lw, ls = {"tg": (TG, 3.0, "-"), "mw": (MW, 3.6, "-"), "bridge": (BRIDGE, 4.2, "-"),
                           "cand": (CAND, 1.8, (0, (1.5, 3.0))), "lost": (LOST, 1.8, (0, (1.5, 3.0)))}[kind]
            ax.plot([x - 0.4, x + 1.4], [y, y], color=col, lw=lw, ls=ls, solid_capstyle="round")
            if kind == "lost":
                ax.text(x + 0.5, y, "×", color=LOST, fontsize=13, ha="center", va="center", fontweight="bold")
        ax.text(x + 2.0, y, text, fontsize=11.5, color=INK, va="center")


def main():
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    fig.text(0.02, 0.965, "What changed from option A to the delivered kit", fontsize=24, fontweight="bold",
             color=INK, va="top")
    axa = fig.add_axes([0.01, 0.1, 0.33, 0.77])
    panel(axa, "A", "Option A, as planned", "6 four-sector nodes + 2 terminals (8 TG units)")
    axb = fig.add_axes([0.35, 0.1, 0.33, 0.77])
    r = panel(axb, "A7", "A7, on the delivered kit", "1 four-sector + 6 two-sector N367 (7 TG units)",
              marks=True)
    pos = r["pos"]
    # numbered markers on the A7 map
    marker(axb, pos["CORE"], 1, dx=-30, dy=26)
    hub_core = (pos["HUB"][0], pos["HUB"][1] + 0.75 * (pos["CORE"][1] - pos["HUB"][1]))  # on the hub-hospital link
    marker(axb, hub_core, 2, dx=24)
    marker(axb, pos["S"], 3, dx=-34, dy=-22)
    marker(axb, pos["N"], 3, dx=-34, dy=20)
    marker(axb, pos["W1"], 4, dx=-36, dy=28)
    marker(axb, pos["E1"], 4, dx=36, dy=28)
    lost = [c for c in r["cands"] if c["feasible"] and c not in r["mount_reach"]]
    if lost:  # mark the out-of-reach bridge lowest on the map, clear of the other markers
        c = min(lost, key=lambda c: pos[c["a"]][1] + pos[c["b"]][1])
        a, b = pos[c["a"]], pos[c["b"]]
        marker(axb, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), 5, dx=24, dy=0)

    # the list of changes
    ax = fig.add_axes([0.70, 0.1, 0.29, 0.77])
    ax.axis("off")
    n_plans_a = analyze("A")["n_plans"]
    rows = [
        ("1", "CORE loses its TG node", "It keeps its IP-50 hub, the TG controller and the DPIR server."),
        ("2", "The hospital hangs off the hub", "Fed from the hub's 4th (north) sector, 606 m, instead of "
         "from CORE, 270 m. The hub now uses all four sectors."),
        ("3", "S and N are nodes, not terminals", "The N367 runs only as a node. Each uses one sector "
         "and keeps one spare."),
        ("4", "W1 and E1 use both their sectors", "Hub on one, the new bridge on the other: they must "
         "be side by side."),
        ("5", "4 bridges are out of reach", f"A 2-sector node cannot point outside its two sectors: "
         f"MARL keeps {len(r['mount_plans'])} of {n_plans_a} recovery plans."),
        ("=", "Unchanged", "The sites, the IP-50 links, the event, the 4 islands and the 3 bridges "
         "E2⇄W2, E1⇄N, W1⇄S."),
    ]
    y = 0.98
    for n, head, body in rows:
        ax.text(0.0, y, n, transform=ax.transAxes, fontsize=13, fontweight="bold", color="white",
                ha="center", va="top", bbox=dict(boxstyle="circle,pad=0.35",
                                                 fc=BRIDGE if n == "=" else INK, ec="none"))
        ax.text(0.07, y, head, transform=ax.transAxes, fontsize=14.5, fontweight="bold", color=INK, va="top")
        import textwrap
        wrapped = "\n".join(textwrap.wrap(body, 46))
        ax.text(0.07, y - 0.045, wrapped, transform=ax.transAxes, fontsize=12, color=MUTED, va="top",
                linespacing=1.35)
        y -= 0.07 + 0.042 * (wrapped.count("\n") + 1)
    legend(fig)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "kit_change.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
