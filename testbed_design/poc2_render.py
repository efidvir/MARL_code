"""Draw the PoC 2 figures from out/poc2_results.json (run poc2.py first).

Run:  python3 poc2_render.py   -> out/poc2_topology.png|svg, out/poc2_sweep.png, out/poc2_dynamic.png
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from poc2 import COUPLINGS, FLOWS, LINKS, NODES, POS, end_sector, mcs_of  # noqa: E402
from render import CAND, FLOW, INK, LOST, MUTED, TG, link_line, offset_poly, tg_icon  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
# chart series (dataviz reference palette, first three slots: validated all-pairs, light mode)
S_DPIR, S_OPENR, S_HOP = "#2a78d6", "#eb6834", "#1baf7a"
SEC_IDX = {"E": 0, "N": 1, "W": 2, "S": 3}


def load():
    with open(os.path.join(OUT, "poc2_results.json")) as fh:
        return json.load(fh)["results"]


# ----------------------------------------------------------------------------
# topology: hop-count routing against DPIR
# ----------------------------------------------------------------------------
def coupling_arc(ax, x, y, active, label):
    """A cabled coupling between sector x and sector y (drawn inside the ring, zig-zag)."""
    (sx, kx), (sy, ky) = x, y
    a, b = POS[sx], POS[sy]
    inset = 120 if a[0] < 300 else -120
    p0, p1 = (a[0] + inset, a[1] - 70), (b[0] + inset, b[1] + 70)
    n = 14
    pts = []
    for i in range(n + 1):
        t = i / n
        px = p0[0] + (p1[0] - p0[0]) * t
        py = p0[1] + (p1[1] - p0[1]) * t
        off = 0 if i in (0, n) else (16 if i % 2 else -16)
        pts.append((px + off, py))
    col = LOST if active else "#d9b7b4"
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.6 if active else 1.2,
            zorder=3, solid_joinstyle="round")
    for p, q in ((p0, (a[0] + (34 if inset > 0 else -34), a[1])), (p1, (b[0] + (34 if inset > 0 else -34), b[1]))):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=1.0, zorder=3)
    mx = (p0[0] + p1[0]) / 2 + (40 if inset > 0 else -40)
    ax.text(mx, 110 if inset > 0 else -110, f"{label}\n{'interfering' if active else 'silent'}",
            ha="left" if inset > 0 else "right",
            va="center", fontsize=11, color=LOST if active else "#b08884", zorder=9, linespacing=1.25)


def flow_path(ax, nodes, color, off):
    pts = offset_poly([POS[n] for n in nodes], off)
    # pull the ends in to the site edge
    def pull(p, q, d=70):
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        return (p[0] + (q[0] - p[0]) * d / L, p[1] + (q[1] - p[1]) * d / L)
    pts[0] = pull(pts[0], pts[1])
    pts[-1] = pull(pts[-1], pts[-2])
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=2.4, zorder=8, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14, color=color,
                                 lw=2.4, zorder=8, shrinkA=0, shrinkB=0))


def panel(ax, res, which):
    st = res["static"]
    row = st["hopcount"] if which == "hop" else st["dpir"]
    paths = [p.split("-") for p in row["paths"]]
    used = {frozenset(e) for p in paths for e in zip(p, p[1:])}
    idle = {k for k, (a, b, *_r) in LINKS.items() if frozenset((a, b)) not in used}
    for k, (a, b, ch, gol, snr, att, role) in LINKS.items():
        on = k not in idle
        link_line(ax, POS[a], POS[b], TG if on else CAND, 5.0 if snr > 20 else 3.0, alpha=1.0 if on else 0.45, z=2)
        mx, my = (POS[a][0] + POS[b][0]) / 2, (POS[a][1] + POS[b][1]) / 2
        m, r = mcs_of(snr)
        horiz = abs(POS[a][1] - POS[b][1]) < 1
        dx, dy = (0, 64 if my > 0 else -64) if horiz else (-64 if mx < 300 else 64, 0)
        txt = f"{k} · MCS{m}" + ("" if on else " · idle")
        ax.text(mx + dx, my + dy, txt, ha="center" if horiz else ("right" if mx < 300 else "left"),
                va="center", fontsize=11, color=INK if on else MUTED, zorder=9)
    # couplings: active when both coupled spans carry traffic in the hit direction
    act = not ({"LN", "LS"} & idle)
    for kname, (x, y, inr, att) in COUPLINGS.items():
        coupling_arc(ax, x, y, act, kname)
    for n, (box, slot, node, ns, pol) in NODES.items():
        secs = [SEC_IDX[end_sector(n, nb)] for nb in ("W1", "E1", "E2", "W2") if nb != n
                and frozenset((n, nb)) in {frozenset((a, b)) for a, b, *_r in LINKS.values()}]
        tg_icon(ax, POS[n], "DN", sectors=secs)
        top = POS[n][1] > 0
        left = POS[n][0] < 300
        ax.text(POS[n][0] + (-62 if left else 62), POS[n][1] + (40 if top else -40), n, ha="right" if left else "left",
                va="center", fontsize=16, fontweight="bold", color=INK, zorder=9)
        ax.text(POS[n][0] + (-62 if left else 62), POS[n][1] + (-14 if top else -98), f"{box} · {pol}",
                ha="right" if left else "left", va="center", fontsize=10, color=MUTED, zorder=9)
    # direct routes run just inside the ring, detours just outside it
    for (fid, s, t, dem), p in zip(FLOWS, paths):
        flow_path(ax, p, FLOW[fid], -26 if len(p) == 2 else -40)
    tot = row["total"]
    title = "Hop-count routing (OSPF / Open/R, flat metrics)" if which == "hop" else "DPIR"
    ax.text(300, 700, title, ha="center", va="bottom", fontsize=16, fontweight="bold", color=INK)
    sub = " · ".join(f"{f[0]} {t:.0f}/{f[3]:.0f}" for f, t in zip(FLOWS, row["thr"]))
    ax.text(300, 630, f"{tot:.0f} Mb/s delivered   ({sub} Mb/s)", ha="center", va="bottom", fontsize=12.5,
            color=INK)
    ax.set_xlim(-430, 1030)
    ax.set_ylim(-640, 760)
    ax.set_aspect("equal")
    ax.axis("off")


def topology(res):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0))
    fig.patch.set_facecolor("white")
    panel(axes[0], res, "hop")
    panel(axes[1], res, "dpir")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TG, lw=5, label="TG span (MCS12)"),
               Line2D([], [], color=TG, lw=3, label="TG side hop (MCS9)"),
               Line2D([], [], color=CAND, lw=3, alpha=0.6, label="idle link"),
               Line2D([], [], color=LOST, lw=1.6, marker=(2, 0, 45), ms=7, label="cabled coupling (emulated co-channel)"),
               Line2D([], [], color=FLOW["F1"], lw=2.4, label="F1 W1→E1 1600 Mb/s"),
               Line2D([], [], color=FLOW["F2"], lw=2.4, label="F2 E2→W2 900 Mb/s")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11.5, handlelength=2.6,
               columnspacing=2.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.12, wspace=0.02)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"poc2_topology.{ext}"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b2")
    ax.tick_params(colors="#52514e", length=0, labelsize=10.5)
    ax.grid(axis="y", color=GRID, lw=1, ls="-")
    ax.set_axisbelow(True)


def sweep(res):
    sw = res["static"]["sweep"]
    x = [s["inr"] for s in sw]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    nom = 18
    ax.axvline(nom, color="#b9b8b2", lw=1, zorder=1)
    ax.text(nom, 2720, "nominal\nINR 18 dB", ha="center", va="bottom", fontsize=8.5, color="#52514e")
    sw_inr = res["static"]["switch_inr"]
    ax.axvspan(sw_inr - 1, x[-1] + 2, color="#2a78d6", alpha=0.05, lw=0, zorder=0)
    ax.text(sw_inr - 0.6, 150, "DPIR sends F2 the long way", ha="left", va="bottom", fontsize=8.5,
            color="#52514e")
    series = [("DPIR", S_DPIR, [s["dpir"] for s in sw], 0), ("Open/R MCS metric", S_OPENR, [s["openr"] for s in sw], 6),
              ("Hop-count", S_HOP, [s["hopcount"] for s in sw], -6)]
    for name, col, ys, dy in series:
        ax.plot(x, ys, color=col, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=4, label=name)
        ax.plot(x, ys, "o", ms=5.5, color=col, mec=SURFACE, mew=2, zorder=5)
        yn = ys[x.index(nom)]
        ax.annotate(f"{name} {yn:.0f}" + (" (avg)" if name.startswith("Open") else ""), (nom, yn), xytext=(10, 10 if name == "DPIR" else 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.set_ylim(0, 3000)
    ax.set_xticks(x)
    ax.set_xlabel("Coupling strength at the victim: INR (dB), set by ATT-5 / ATT-6", fontsize=10, color="#52514e")
    ax.set_ylabel("Delivered, both flows (Mb/s)", fontsize=10, color="#52514e")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, ncol=3, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "poc2_sweep.png"), dpi=200)
    plt.close(fig)


def dynamic(res):
    A, B = res["dynamic_silent"], res["dynamic_qosnull"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0], A, "Idle links silent: routing alone", [
            ("DPIR with predictor", S_DPIR, "dpir_predictive"), ("Open/R MCS metric", S_OPENR, "openr_mcs"),
            ("Hop-count", S_HOP, "hopcount")]),
        (axes[1], B, "Worst case (QoS-Null): DPIR also takes links down", [
            ("DPIR + links, predictive", S_DPIR, "dpir_links_predictive"),
            ("DPIR + links, one step late", S_OPENR, "dpir_links_lagged"), ("Hop-count", S_HOP, "hopcount")]),
    ]
    for ax, run, title, series in panels:
        style(ax)
        t = list(range(len(run["offered"])))
        ax.fill_between(t, run["offered"], step="post", color="#e9e8e4", lw=0, zorder=1, label="offered")
        for name, col, key in series:
            ax.step(t, run[key]["trace"], where="post", color=col, lw=2, zorder=4, label=f"{name} ({run[key]['delivered_pct']:.0f} %)")
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=50)
        ax.legend(loc="lower left", frameon=False, fontsize=10.5, ncol=2, bbox_to_anchor=(0, 0.995),
                  handlelength=1.6, columnspacing=1.2)
        ax.set_xlim(0, len(t))
        ax.set_ylim(0, 2900)
        ax.set_xlabel("time (s), 1 s routing step", fontsize=9.5, color="#52514e")
    axes[0].set_ylabel("Delivered (Mb/s)", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "poc2_dynamic.png"), dpi=200)
    plt.close(fig)


def main():
    res = load()
    topology(res)
    sweep(res)
    dynamic(res)
    print("wrote out/poc2_topology.png|svg, out/poc2_sweep.png, out/poc2_dynamic.png")


if __name__ == "__main__":
    main()
