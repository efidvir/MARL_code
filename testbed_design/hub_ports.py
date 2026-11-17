"""How the A7 hub serves six links from four RF ports, drawn.

A sector is point-to-multipoint: the hub's west sector serves W1 and W2, its
east sector E1 and E2, its north and south sectors one node each.  On the
cabled bench a 2-way splitter divides a port that serves two nodes into two
paths, each with its own programmable attenuator.

Run:  python3 hub_ports.py    -> out/hub_ports.png
"""
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402

from option_a_sectors import plan  # noqa: E402
from render import INK, MUTED, OUT, TG  # noqa: E402

SEC = ["E", "N", "W", "S"]
PORT_Y = {1: 47.0, 0: 35.0, 2: 19.0, 3: 7.0}       # N, E, W, S ports, top to bottom
PEER_DY = 4.6                                        # vertical spread of two peers on one port


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.8, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def main(key="A7"):
    r, nodes, paths, junctions, ports, rf_cables, mw, sw = plan(key)
    hub_paths = {}
    for p in paths:
        for x, px, y in ((p["a"], p["pa"], p["b"]), (p["b"], p["pb"], p["a"])):
            if x == "HUB":
                other = p["pb"] if px == p["pa"] else p["pa"]
                hub_paths.setdefault(px[1], []).append((y, other[1], p["before"]))

    fig = plt.figure(figsize=(19.2, 10.8), facecolor="white")
    fig.text(0.03, 0.95, "Four RF ports, six links: how the hub’s sectors are shared", fontsize=26,
             fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.895, "A7 hub: the one 4-sector node (Spider box GTST002B). Each sector is one module and can serve "
             "more than one node.", fontsize=14, color=MUTED, va="top")
    ax = fig.add_axes([0.02, 0.06, 0.6, 0.8])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    # the hub unit and its four ports
    box(ax, 2, 2, 20, 50, "#FDF1E4", TG, lw=2.2, r=1.6)
    ax.text(8, 28.5, "HUB", ha="center", va="bottom", fontsize=20, fontweight="bold", color=INK, zorder=5)
    ax.text(8, 27.5, "GTST002B\n4 sectors", ha="center", va="top", fontsize=12.5, color=MUTED, zorder=5,
            linespacing=1.3)
    for k, y in PORT_Y.items():
        box(ax, 14.5, y - 1.8, 9, 3.6, TG, TG, r=0.6, z=4)
        ax.text(19, y, f"{SEC[k]} port", ha="center", va="center", fontsize=13, fontweight="bold",
                color="white", zorder=6)

    # paths: port -> (splitter) -> attenuator -> peer node
    for k, y in PORT_Y.items():
        peers = sorted(hub_paths.get(k, []), key=lambda t: t[0])
        if len(peers) == 1:
            ys = [y]
            src = (23.5, y)
        else:
            ys = [y + PEER_DY, y - PEER_DY][:len(peers)]
            ax.plot([23.5, 33], [y, y], color=TG, lw=3.2, zorder=2, solid_capstyle="round")
            sx = 0.6 * 19.2 / 100   # inches per x unit
            sy = 0.8 * 10.8 / 54    # inches per y unit
            ax.add_patch(Ellipse((35, y), 4.0, 4.0 * sx / sy, fc="white", ec=INK, lw=1.6, zorder=4))
            ax.text(35, y, "1:2", ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=5)
            ax.text(35, y - 2.4, "splitter", ha="center", va="top", fontsize=10.5, color=MUTED, zorder=5)
            src = (37, y)
        for (peer, peer_sec, att), yy in zip(peers, ys):
            ax.plot([src[0], 47, 50], [src[1], yy, yy], color=TG, lw=3.2, zorder=2, solid_capstyle="round",
                    solid_joinstyle="round")
            box(ax, 50, yy - 1.7, 12, 3.4, INK, INK, r=0.5, z=4)
            ax.text(56, yy, f"{att:.1f} dB", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white", zorder=5)
            ax.plot([62, 72], [yy, yy], color=TG, lw=3.2, zorder=2, solid_capstyle="round")
            box(ax, 72, yy - 1.9, 26, 3.8, "#FFFFFF", "#C9CED8", r=0.6, z=4)
            ax.text(74, yy, peer, ha="left", va="center", fontsize=15, fontweight="bold", color=INK, zorder=5)
            ax.text(80.5, yy, "module 1", ha="left", va="center", fontsize=12, color=MUTED, zorder=5)
    ax.text(56, 52.5, "programmable attenuator (dB before the event)", ha="center", va="bottom", fontsize=11,
            color=MUTED)

    # explanation
    tx = fig.add_axes([0.64, 0.06, 0.34, 0.8])
    tx.axis("off")
    pts = [
        ("One sector, one RF port, several nodes",
         "A Terragraph sector is point-to-multipoint. The hub’s W port serves W1 and W2, its E port E1 and "
         "E2, its N and S ports one node each: 6 links on 4 ports."),
        ("Over the air they share airtime",
         "The sector’s beam visits each node in its own time slots, so W1 and W2 split the W sector’s "
         "capacity. That shared airtime is the coupling DPIR plans around."),
        ("On the bench: two ports or a splitter",
         "If a module’s four SMA ports are taps of one sector, two of them reach W1 and W2 directly; "
         "otherwise a 2-way splitter divides the port. Each path keeps its own attenuator."),
        ("Two per sector is the limit",
         "Terragraph allows at most 2 node-to-node links per sector: the W and E ports are full."),
        ("Elsewhere, a second module does it",
         "W2, E2, S and N put their bridge on their second module, so only the hub’s W and E modules "
         "serve two nodes."),
    ]
    y = 0.99
    for head, body in pts:
        tx.text(0.0, y, head, fontsize=15.5, fontweight="bold", color=INK, va="top", transform=tx.transAxes)
        wrapped = "\n".join(textwrap.wrap(body, 52))
        tx.text(0.0, y - 0.05, wrapped, fontsize=13, color=MUTED, va="top", linespacing=1.4,
                transform=tx.transAxes)
        y -= 0.075 + 0.04 * (wrapped.count("\n") + 1)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "hub_ports.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
