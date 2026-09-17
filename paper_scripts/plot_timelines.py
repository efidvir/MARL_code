# -*- coding: utf-8 -*-
"""Timeline figure: achievability, latency and energy over the hour, for both
scenarios, from the per-bin summaries written by extract_timelines.py.

Palette validated with the dataviz validator (light, white surface, all pairs):
MARL-RIC #006D2C, XL-DET #1F5AA0, SDN #CC7A00 pass lightness, chroma, CVD
(worst protan dE 11.7), normal-vision (20.8) and 3:1 contrast.  The paper's
red (#C0392B) failed against the green under protanopia (dE 4.3), so SDN is
amber here.  Routing protocols and the random control are neutral context
series.  Every series also has its own line style (secondary encoding).

Usage: python plot_timelines.py --data auth_results/timelines_dev.json
                                --out paper/figs/fig_timelines --label "development set"
"""
import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INK = '#1f1f1f'
MUTED = '#5a5a5a'
GRID = '#e6e6e6'
SERIES = [  # fixed order; colour follows the entity
    ('marl_freeze', 'MARL-RIC', '#006D2C', '-', 1.7),
    ('xldet', 'XL-DET', '#1F5AA0', '-', 1.2),
    ('sdn', 'SDN/OpenFlow', '#CC7A00', (0, (5, 2)), 1.2),
    ('routing', 'routing protocols (mean of four)', '#6E6E6E', (0, (6, 2, 1.5, 2)), 1.1),
    ('random', 'uniform random', '#A3A3A3', (0, (1.2, 1.8)), 1.1),
]
BANDED = ('marl_freeze', 'xldet')
ROUTING = ('ospf', 'olsr', 'batman', 'aodv')
ROWS = [('conn', 'Achievability (%)'), ('latency', 'Latency (ms)'), ('energy', 'Energy (J per tick)')]
SHADE = '#8a8a8a'
PRE_SURGE = (900, 1400)   # Scenario B pre-surge window (ticks = s), as in the metrics
COLS = [('A', 'Scenario A: fragmenting core severance', 200, 'severance'),
        ('B', 'Scenario B: responder surge on a severed network', 1400, 'responder surge')]


def arr(v):
    return [float('nan') if x is None else x for x in v]


def routing_mean(arms, key):
    cols = [arr(arms[r][key]['mean']) for r in ROUTING]
    out, lo, hi = [], [], []
    for vals in zip(*cols):
        v = [x for x in vals if not math.isnan(x)]
        out.append(sum(v) / len(v) if v else float('nan'))
        lo.append(min(v) if v else float('nan'))
        hi.append(max(v) if v else float('nan'))
    return out, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--marl-arm', default='marl_static')
    a = ap.parse_args()
    global SERIES, BANDED
    SERIES = [(a.marl_arm,) + SERIES[0][1:]] + SERIES[1:]
    BANDED = (a.marl_arm, 'xldet')
    D = json.load(open(a.data))
    bin_s = D['bin_ticks']

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 7.6, 'axes.titlesize': 8.2,
        'axes.labelsize': 7.8, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0,
        'axes.edgecolor': '#9a9a9a', 'axes.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.labelcolor': INK,
    })
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 5.55), sharex='col', sharey='row')
    plt.subplots_adjust(left=0.075, right=0.982, top=0.875, bottom=0.075, hspace=0.12, wspace=0.10)

    for c, (k, title, t_event, ev_label) in enumerate(COLS):
        S = D['scenarios'][k]
        arms = S['arms']
        nb = S['n_bins']
        t = [(i + 0.5) * bin_s / 60.0 for i in range(nb)]
        for r, (key, ylab) in enumerate(ROWS):
            ax = axes[r][c]
            ax.grid(axis='y', color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            # event markers
            ax.axvline(t_event / 60.0, color='#8a8a8a', lw=0.8, ls=(0, (3, 2)), zorder=1)
            if k == 'B':
                ax.axvline(0.0, color='#8a8a8a', lw=0.8, ls=(0, (3, 2)), zorder=1)
            # series, drawn back to front so MARL-RIC sits on top
            for arm, name, col, ls, lw in reversed(SERIES):
                if arm == 'routing':
                    m, lo, hi = routing_mean(arms, key)
                    if key != 'conn':
                        ax.fill_between(t, lo, hi, color=col, alpha=0.12, lw=0, zorder=2)
                    ax.plot(t, m, color=col, ls=ls, lw=lw, zorder=3)
                    continue
                s = arms[arm][key]
                if arm in BANDED:
                    ax.fill_between(t, arr(s['q25']), arr(s['q75']), color=col, alpha=0.16, lw=0, zorder=2)
                ax.plot(t, arr(s['mean']), color=col, ls=ls, lw=lw, zorder=5 if arm == SERIES[0][0] else 4,
                        solid_capstyle='round')
            if c == 0:
                ax.set_ylabel(ylab)
            # shaded context: Scenario A's intact period (all arms run the same
            # fixed configuration until island mode), Scenario B's pre-surge
            # measurement window [900, 1400) s
            if k == 'A':
                ax.axvspan(0, t_event / 60.0, color=SHADE, alpha=0.10, lw=0, zorder=0)
            else:
                ax.axvspan(PRE_SURGE[0] / 60.0, PRE_SURGE[1] / 60.0, color=SHADE, alpha=0.10, lw=0, zorder=0)
            if r == 0:
                ax.set_title(title, loc='left', pad=12)
                ax.annotate(ev_label, xy=(t_event / 60.0, 1.0), xycoords=('data', 'axes fraction'),
                            xytext=(3, 1), textcoords='offset points', fontsize=6.8, color=MUTED,
                            va='bottom', ha='left')
                if k == 'A':
                    ax.annotate('intact network: all arms act identically\n(achievability curves coincide until the severance)',
                                xy=(t_event / 120.0, 41.5), xytext=(6.0, 60.5), textcoords='data',
                                fontsize=6.4, color=MUTED, va='center', ha='left',
                                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
                else:
                    ax.annotate('core severed at t = 0', xy=(0.0, 1.0), xycoords=('data', 'axes fraction'),
                                xytext=(3, 1), textcoords='offset points', fontsize=6.8, color=MUTED,
                                va='bottom', ha='left')
                    ax.text(sum(PRE_SURGE) / 120.0, 8.0, 'pre-surge\nwindow', fontsize=6.4, color=MUTED,
                            va='center', ha='center')
            if r == 2:
                ax.set_xlabel('Time (min)')
            ax.set_xlim(0, nb * bin_s / 60.0)
            ax.set_xticks([0, 10, 20, 30, 40, 50, 60])

    # y ranges chosen from the data so no series is clipped
    def rng(key, pad):
        vals = []
        for k, *_ in COLS:
            arms = D['scenarios'][k]['arms']
            for arm in (SERIES[0][0], 'xldet', 'sdn', 'random') + ROUTING:
                for stat in ('mean', 'q25', 'q75') if arm in BANDED else ('mean',):
                    vals += [x for x in arms[arm][key][stat] if x is not None]
        lo, hi = min(vals), max(vals)
        return lo - pad * (hi - lo), hi + pad * (hi - lo)
    axes[0][0].set_ylim(0, max(70, rng('conn', 0.05)[1]))
    axes[1][0].set_ylim(*rng('latency', 0.06))
    axes[2][0].set_ylim(*rng('energy', 0.06))

    handles = [Line2D([], [], color=col, ls=ls, lw=lw + 0.2, label=name) for _, name, col, ls, lw in SERIES]
    handles += [Patch(facecolor=SERIES[0][2], alpha=0.30, lw=0, label='interquartile range, MARL-RIC'),
                Patch(facecolor=SERIES[1][2], alpha=0.30, lw=0, label='interquartile range, XL-DET'),
                Patch(facecolor='#6E6E6E', alpha=0.22, lw=0, label='range of the four routing protocols')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.3, columnspacing=1.1, handletextpad=0.5)
    if a.label:
        fig.text(0.995, 0.005, a.label, ha='right', va='bottom', fontsize=6.2, color=MUTED)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=250)
    print('wrote', a.out + '.pdf/.png')
    # numbers the caption may quote
    for k, *_ in COLS:
        arms = D['scenarios'][k]['arms']
        for key, _ in ROWS:
            row = []
            for arm, name, *_ in SERIES:
                if arm == 'routing':
                    m, _, _ = routing_mean(arms, key)
                else:
                    m = arr(arms[arm][key]['mean'])
                tail = [x for x in m[-50:] if not math.isnan(x)]
                row.append('%s %.2f' % (name.split(' ')[0], sum(tail) / len(tail)))
            print('  %s %-8s last-500-tick mean: %s' % (k, key, ' | '.join(row)))


if __name__ == '__main__':
    main()
