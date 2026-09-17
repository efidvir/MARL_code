# -*- coding: utf-8 -*-
"""Three-panel process figure for the manuscript (replaces fig_severance_map).

  (a) the intact network before the event
  (b) the same network right after the severance: fragments, cut links
  (c) the end state under the deployed learned policy: the 60 GHz bridges it
      formed and the resulting single component

Panels (a) and (b) are re-derived from the scenario builders through
plot_severance_maps.derive_instance (pure function of the seed); panel (c)
reads the end-state dump produced on the server by dump_endstate.py, which
ran the real harness with the deployed checkpoint.  Nothing is drawn that is
not in one of those two sources.

Usage: python plot_process_figure.py --seed 58 --dump auth_results/endstate/ce_s1234_seed58.json
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from plot_severance_maps import derive_instance

FRAG = ['#0072B2', '#D55E00', '#CC79A7', '#E69F00', '#56B4E9', '#8C564B']
GREY_LINK = '#B8BEC4'
CUT = '#D62728'
BRIDGE = '#006D2C'
CORE = '#8A8A8A'
UE = '#C9CDD1'

# The story needs three kinds of thing, not seven: a site, a site that can
# re-point a 60 GHz head (the only hardware that can change the graph), and a
# core node that the event removes.  Node function is in Table I, not here.
SITE = dict(marker='o', size=17)
STEER = dict(marker='*', size=88)
COREM = dict(marker='X', size=34)


def hull(points):
    """Monotone-chain convex hull; returns list of (x, y)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def pad_hull(points, pad):
    """Expand a hull outward by `pad` metres (rounded look via extra points)."""
    if len(points) < 3:
        # capsule around 1-2 points
        out = []
        for (x, y) in points:
            for k in range(24):
                a = 2 * math.pi * k / 24
                out.append((x + pad * math.cos(a), y + pad * math.sin(a)))
        return hull(out)
    cx = sum(p[0] for p in points) / len(points); cy = sum(p[1] for p in points) / len(points)
    out = []
    for (x, y) in points:
        for k in range(12):
            a = 2 * math.pi * k / 12
            out.append((x + pad * math.cos(a), y + pad * math.sin(a)))
    return hull(out)


def draw_panel(ax, inst, mode, dump=None, title='', stat=''):
    pos, ntype = inst['pos'], inst['ntype']
    core, infra = inst['core'], set(inst['infra_surv'])
    frag_of = dict(inst['frag_of'])
    # (c): recolour by the relay-inclusive component the harness reports
    if mode == 'end' and dump is not None:
        end = [s for s in dump['snapshots'] if s['tag'] == 'end'][0]
        lab = {nid: v['label'] for nid, v in end['nodes'].items() if v['label'] is not None}
        order = {}
        for nid in inst['infra_surv']:
            l = lab.get(str(nid))
            if l is not None and l not in order:
                order[l] = len(order)
        frag_of = {nid: order[lab[str(nid)]] for nid in inst['infra_surv'] if str(nid) in lab}
        n_comp = end['components']
    else:
        n_comp = inst['n_fragments']

    # ---- fragment shading (b, c) ----
    if mode in ('cut', 'end'):
        groups = {}
        for nid, f in frag_of.items():
            groups.setdefault(f, []).append(pos[nid])
        for f, pts in groups.items():
            poly = pad_hull(pts, 260 if len(pts) > 1 else 330)
            if len(poly) >= 3:
                ax.add_patch(Polygon(poly, closed=True, facecolor=FRAG[f % len(FRAG)], alpha=0.075,
                                     edgecolor=FRAG[f % len(FRAG)], linewidth=0.7, zorder=0))

    # ---- UE dots (faint) ----
    for ue in inst['ue_ids']:
        x, y = pos[ue]
        ax.plot(x, y, marker='.', ms=1.9, color=UE, zorder=2, linestyle='none')

    # ---- links ----
    for u, v, kind in inst['edges']:
        if kind == 'uu':
            continue
        (x1, y1), (x2, y2) = pos[u], pos[v]
        if mode == 'pre':
            ax.plot([x1, x2], [y1, y2], color=GREY_LINK, lw=0.6, alpha=0.85, zorder=3)
        else:
            if kind == 'core':
                continue                                  # gone with the core
            if kind == 'cut':
                ax.plot([x1, x2], [y1, y2], color=CUT, lw=0.6, ls=(0, (1.8, 2.4)), alpha=0.5, zorder=2)
            else:
                f = frag_of.get(u, frag_of.get(v, 0))
                ax.plot([x1, x2], [y1, y2], color=FRAG[f % len(FRAG)], lw=0.8, alpha=0.8, zorder=3)

    # ---- bridges (c) ----
    n_bridges = 0
    if mode == 'end' and dump is not None:
        end = [s for s in dump['snapshots'] if s['tag'] == 'end'][0]
        tree = set(end['bridge_link_ids'])
        for rl in end['relay_links']:
            if not rl['is_up']:
                continue
            a, b = rl['endpoints']
            if a not in pos or b not in pos:
                continue
            (x1, y1), (x2, y2) = pos[a], pos[b]
            is_tree = rl['id'] in tree
            ax.plot([x1, x2], [y1, y2], color=BRIDGE, lw=2.6 if is_tree else 1.4,
                    ls='-' if is_tree else (0, (3, 2)), alpha=1.0 if is_tree else 0.7, zorder=6,
                    solid_capstyle='round')
            n_bridges += int(is_tree)

    # ---- nodes: a site, a steerable site, a core node.  Nothing else. ----
    for nid in inst['infra_surv']:
        x, y = pos[nid]
        col = FRAG[frag_of.get(nid, 0) % len(FRAG)] if mode != 'pre' else '#5A5A5A'
        if nid in inst['multihaul']:
            ax.scatter([x], [y], marker=STEER['marker'], s=STEER['size'], facecolor=col,
                       edgecolor='white', linewidths=0.45, zorder=8)
        else:
            ax.scatter([x], [y], marker=SITE['marker'], s=SITE['size'], facecolor='white',
                       edgecolor=col, linewidths=0.9, zorder=7)
    for nid in core:
        x, y = pos[nid]
        ax.scatter([x], [y], marker=COREM['marker'], s=COREM['size'],
                   facecolor=('#5A5A5A' if mode == 'pre' else 'none'),
                   edgecolor=CORE, linewidths=0.9, zorder=6, alpha=1.0 if mode == 'pre' else 0.55)

    ax.set_aspect('equal'); ax.set_xlim(-250, 8250); ax.set_ylim(-600, 8250)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#C4C4C4'); sp.set_linewidth(0.6)
    ax.set_title(title, fontsize=8.2, pad=3.5)
    if stat:
        # below the panel, never over the map
        ax.text(0.0, -0.035, stat, transform=ax.transAxes, fontsize=6.7, va='top', ha='left',
                linespacing=1.45, color='#222222')
    # scale bar, inside the reserved band at the foot of the map
    ax.plot([6500, 7500], [-330, -330], color='black', lw=1.1, zorder=9)
    ax.text(7000, -240, '1 km', ha='center', va='bottom', fontsize=6.2)
    return n_comp, n_bridges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', default='paper/figs/fig_process_map')
    a = ap.parse_args()
    dump = json.load(open(a.dump))
    assert dump['seed'] == a.seed
    inst = derive_instance(a.seed, 'recovery')
    if inst['mismatch']:
        raise SystemExit('builder mismatch: %s' % inst['mismatch'])
    # actionable steerable sites = MultiHaul flag AND an agent (the radio
    # hardware); this is the 22-site set the paper counts, not the 26 flags
    import run_timeline_comparison as R
    from sixg_sim.simulation import Simulator
    _t = R.build_comparison_topology(a.seed); topo = _t[0] if isinstance(_t, tuple) else _t
    scen = R.build_comparison_scenario('recovery', a.seed, topology=topo)
    sim = Simulator(topo, scen, R.SimulationConfig())
    actionable = {n for n in sim.agents if getattr(topo.nodes[n], 'has_multihaul', False)}
    inst['multihaul'] = actionable & set(inst['infra_surv'])
    inst['n_multihaul_surviving'] = len(inst['multihaul'])
    print('actionable steerable sites: %d' % inst['n_multihaul_surviving'])
    cut_snap = [s for s in dump['snapshots'] if s['tag'] == 'cut'][0]
    end_snap = [s for s in dump['snapshots'] if s['tag'] == 'end'][0]
    assert cut_snap['components_raw'] == inst['n_fragments'], (cut_snap['components_raw'], inst['n_fragments'])

    optimum = [i for i in json.load(open('benchmark_manifest.json'))['instances']
               if i['scenario_kind'] == 'recovery' and i['topology_seed'] == a.seed][0]['attainable_optimum']
    n_steer = inst['n_multihaul_surviving']
    n_site = len(inst['infra_surv'])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.62))
    plt.subplots_adjust(left=0.008, right=0.992, top=0.945, bottom=0.315, wspace=0.035)
    draw_panel(axes[0], inst, 'pre', title='(a) Intact network',
               stat='%d sites, %d with a steerable head\nno flow stranded' % (n_site, n_steer))
    draw_panel(axes[1], inst, 'cut',
               title='(b) $t=%d$: the severance' % inst['severance_tick'],
               stat='%d links cut, core lost\n%d fragments: %s sites\n%d of %d flows stranded (%.0f\\%%)'
                    .replace('\\%', '%')
                    % (inst['n_cut_links'], inst['n_fragments'],
                       '/'.join(str(x) for x in inst['comp_sizes']),
                       inst['cross_flows'], inst['total_flows'], 100 * inst['cross_share']))
    draw_panel(axes[2], inst, 'end', dump=dump,
               title='(c) $t=%d$: under MARL-RIC' % end_snap['tick'],
               stat='%d bridges formed\n%d component%s remain%s (%d attainable)\n%.0f%% of demand delivered' % (
                   end_snap['distinct_bridges'], end_snap['components'],
                   '' if end_snap['components'] == 1 else 's', 's' if end_snap['components'] == 1 else '',
                   optimum, dump['steady_achievability']))
    handles = [
        Line2D([], [], marker='o', ms=4, mfc='white', mec='#5A5A5A', mew=0.9, ls='none',
               label='node without radio (RIC, O-CU, edge UPF)'),
        Line2D([], [], marker='*', ms=9, mfc='#5A5A5A', mec='white', mew=0.4, ls='none',
               label='site with a steerable \\SI{60}{GHz} head'.replace('\\SI{60}{GHz}', '60 GHz')),
        Line2D([], [], marker='X', ms=5.5, mfc='none', mec=CORE, ls='none', label='core node (lost at $t_0$)'),
        Line2D([], [], marker='.', ms=5, color=UE, ls='none', label='subscriber'),
        Line2D([], [], color=GREY_LINK, lw=1.1, label='transport link'),
        Line2D([], [], color=CUT, lw=1.1, ls=(0, (1.8, 2.4)), label='link severed by the event'),
        Patch(facecolor=FRAG[0], alpha=0.14, edgecolor=FRAG[0], label='fragment (one colour each)'),
        Line2D([], [], color=BRIDGE, lw=2.4, label='60 GHz link that rejoins two fragments'),
        Line2D([], [], color=BRIDGE, lw=1.3, ls=(0, (3, 2)), alpha=0.7,
               label='60 GHz link that does not'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 0.004), handlelength=1.9, columnspacing=1.8, handletextpad=0.55)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + '.pdf'); fig.savefig(a.out + '.png', dpi=300)
    tree = set(end_snap['bridge_link_ids'])
    engaged = set()
    for rl in end_snap['relay_links']:
        if rl['is_up'] and rl['id'] in tree:
            engaged.update(rl['endpoints'])
    stats = {'seed': a.seed, 'end_engaged_sites': len(engaged),
             'severance_tick': inst['severance_tick'], 'n_cut_links': inst['n_cut_links'],
             'n_core_links': inst['n_core_links'], 'n_fragments': inst['n_fragments'],
             'comp_sizes': inst['comp_sizes'], 'cross_flows': inst['cross_flows'], 'total_flows': inst['total_flows'],
             'cross_share': inst['cross_share'], 'n_multihaul_surviving': inst['n_multihaul_surviving'],
             'n_infra_surviving': len(inst['infra_surv']), 'end_tick': end_snap['tick'],
             'end_components': end_snap['components'], 'attainable_optimum': optimum, 'end_bridges': end_snap['distinct_bridges'],
             'steady_components': dump['steady_components'], 'steady_achievability': dump['steady_achievability']}
    json.dump(stats, open(a.out + '_stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1))
    print('wrote', a.out + '.pdf/.png')


if __name__ == '__main__':
    main()
