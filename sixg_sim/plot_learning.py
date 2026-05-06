"""
plot_learning.py — MARL Training Convergence Visualisation

Loads learning_kpis.json produced by LearningTracker and generates a
9-panel figure showing real training signals.

Panel layout (3x3):
  Row 0: Episode Reward | Policy Entropy (exploration decay) | Transport relay Count
  Row 1: Policy Loss    | Value Loss                          | Reward vs Entropy (scatter)
  Row 2: Per-tick Reward ep1 vs epLast | Per-tick Transport relay ep1 vs epLast | Episode Summary bars

All panels use data that is ALWAYS non-empty regardless of island_mode.

Usage:
    python -m sixg_sim.plot_learning [kpi_json] [output_png]
    from sixg_sim.plot_learning import plot_convergence
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(values: Sequence, w: int = 3) -> list:
    """Centred running-mean; NaN values are skipped."""
    out = []
    for i, v in enumerate(values):
        chunk = [x for x in values[max(0, i - w + 1):i + 1]
                 if x is not None and not math.isnan(float(x))]
        out.append(sum(chunk) / len(chunk) if chunk else float('nan'))
    return out


def _valid(xs: list, ys: list):
    """Return x, y pairs where y is finite."""
    pairs = [(x, float(y)) for x, y in zip(xs, ys)
             if y is not None and not math.isnan(float(y))]
    if not pairs:
        return [], []
    return zip(*pairs)


def _trend(ax, xs, ys, color, alpha=0.35):
    """Dashed linear trend + slope annotation."""
    import numpy as np
    xv, yv = _valid(xs, ys)
    xv, yv = list(xv), list(yv)
    if len(xv) < 2:
        return
    z = np.polyfit(xv, yv, 1)
    p = np.poly1d(z)
    ax.plot(xv, [p(x) for x in xv], '--', color=color, linewidth=1.2, alpha=alpha)
    slope = z[0]
    sign  = '+' if slope > 0 else ''
    col   = '#34c759' if slope > 0 else '#ff6b35'
    ax.text(0.97, 0.06, f'slope {sign}{slope:.4f}/ep',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color=col, style='italic')


def _fill_plot(ax, xs, ys, color, label=None, smooth_w=3, alpha_fill=0.18):
    """Fill-under + line with smoothed overlay."""
    sm = _smooth(ys, w=smooth_w)
    ax.fill_between(xs, ys, alpha=alpha_fill, color=color)
    ax.plot(xs, ys, 'o--', color=color, alpha=0.45, markersize=3, linewidth=0.8)
    ax.plot(xs, sm, '-', color=color, linewidth=2.4, label=label)


# ── palette & style ───────────────────────────────────────────────────────────

DARK = '#0d1117'
MID  = '#161b22'
GRID = '#30363d'
FG   = '#e6edf3'
C1   = '#3d8ef8'    # blue
C2   = '#34c759'    # green
C3   = '#e3b341'    # amber
C4   = '#bf5af2'    # purple
C5   = '#ff6b35'    # orange
C6   = '#30d5c8'    # teal
CSEV = '#ff3b30'    # red


def _setup_rcparams():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': DARK, 'axes.facecolor': MID,
        'axes.edgecolor': GRID, 'axes.labelcolor': FG,
        'xtick.color': FG, 'ytick.color': FG, 'text.color': FG,
        'grid.color': GRID, 'grid.alpha': 0.45, 'grid.linestyle': '--',
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'legend.framealpha': 0.3, 'legend.facecolor': MID,
    })
    return plt


def _ax(fig, gs, row, col, title, xlabel='Episode', ylabel=''):
    ax = fig.add_subplot(gs[row, col])
    ax.set_title(title, color=FG, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, color=FG, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=8)
    ax.grid(True)
    return ax


# ── main entry point ──────────────────────────────────────────────────────────

def plot_convergence(kpi_json: str, out_png: str, dpi: int = 180) -> None:
    plt = _setup_rcparams()
    import matplotlib.gridspec as gridspec
    import numpy as np

    # ── Load ──────────────────────────────────────────────────────────────────
    with open(kpi_json) as f:
        data = json.load(f)

    ticks    = data.get('ticks', [])
    episodes = data.get('episodes', [])

    if not episodes:
        print("[plot_learning] No episode data — nothing to plot.")
        return

    n_ep    = len(episodes)
    ep_nums = [e['episode'] for e in episodes]

    # Per-episode scalars
    ep_reward   = [e.get('ep_reward_mean', float('nan'))    for e in episodes]
    pol_loss_ep = [e.get('final_policy_loss', float('nan')) for e in episodes]
    val_loss_ep = [e.get('final_value_loss',  float('nan')) for e in episodes]
    entropy_ep  = [e.get('final_entropy',     float('nan')) for e in episodes]
    # Transport relay: use max of post_sev/peak, fallback to peak
    iab_ep      = [max(e.get('post_sev_transport_relay', 0),
                        e.get('peak_transport_relay', 0))          for e in episodes]
    sc_type     = [e.get('scenario_type', 'full_core')       for e in episodes]

    # Per-tick: average per-episode across all ticks
    def per_ep_mean(key):
        from collections import defaultdict
        buckets = defaultdict(list)
        for t in ticks:
            v = t.get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                buckets[t['episode']].append(float(v))
        return [
            (sum(buckets[ep]) / len(buckets[ep])) if buckets[ep] else float('nan')
            for ep in ep_nums
        ]

    iab_per_ep_mean = per_ep_mean('transport_relay_count')

    # Per-tick within ep1 and last ep
    def ticks_for_ep(ep_num):
        return [t for t in ticks if t['episode'] == ep_num]

    def series(recs, key):
        return [t.get(key, 0) for t in recs]

    ep1_ticks  = ticks_for_ep(1)
    epL_ticks  = ticks_for_ep(n_ep)

    # Per-tick losses (taken from ticks that have them)
    def tick_series_by_ep(key):
        """Return {ep: list of (tick_in_ep, value)} for valid ticks."""
        from collections import defaultdict
        res = defaultdict(list)
        for t in ticks:
            v = t.get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                res[t['episode']].append((t['tick'], float(v)))
        return res

    pl_by_ep  = tick_series_by_ep('policy_loss')
    vl_by_ep  = tick_series_by_ep('value_loss')
    ent_by_ep = tick_series_by_ep('entropy')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=DARK)
    fig.suptitle(
        '6G MARL Training Convergence — Policy, Relay & Reward Signals\n'
        f'({n_ep} Episodes | {len(ticks):,} ticks | '
        f'Diverse scenario types: {", ".join(sorted(set(sc_type)))})',
        color=FG, fontsize=13, fontweight='bold', y=0.985
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.96, top=0.93, bottom=0.06)

    # ── [0,0]  Episode Reward ──────────────────────────────────────────────
    ax = _ax(fig, gs, 0, 0, 'Episode Mean Reward', ylabel='Mean reward / tick')
    _fill_plot(ax, ep_nums, ep_reward, C1, label='Reward', smooth_w=3)
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=':')
    _trend(ax, ep_nums, ep_reward, C1)
    ax.legend(fontsize=7)
    ax.set_xlabel('Episode', color=FG, fontsize=8)

    # ── [0,1]  Policy Entropy (exploration decay) ──────────────────────────
    ax = _ax(fig, gs, 0, 1, 'Policy Entropy\n(Exploration → Exploitation decay)',
             ylabel='Entropy (nats)')
    valid_e = [(ep, v) for ep, v in zip(ep_nums, entropy_ep)
               if v is not None and not math.isnan(v)]
    if valid_e:
        ex, vy = zip(*valid_e)
        _fill_plot(ax, list(ex), list(vy), C3, smooth_w=3)
        _trend(ax, list(ex), list(vy), C3)
    else:
        ax.text(0.5, 0.5, 'No entropy data', transform=ax.transAxes,
                ha='center', color=FG, fontsize=9)

    # ── [0,2]  Transport relay Count per Episode ────────────────────────────────
    ax = _ax(fig, gs, 0, 2, 'Transport relay Adoption\n(Mean relay nodes per episode)',
             ylabel='Mean nodes in relay mode')
    _fill_plot(ax, ep_nums, iab_per_ep_mean, C2, label='Mean Transport relays', smooth_w=3)
    ax.plot(ep_nums, iab_ep, 's--', color=C5, linewidth=1.2, markersize=3,
            alpha=0.6, label='Peak Transport relays (ep)')
    _trend(ax, ep_nums, iab_per_ep_mean, C2)
    ax.legend(fontsize=7)

    # ── [1,0]  Policy Loss across ticks ───────────────────────────────────
    ax = _ax(fig, gs, 1, 0, 'MAPPO Policy Loss\n(Decreasing = policy improving)',
             ylabel='Policy loss')
    # Plot per-tick losses grouped by episode (scatter + episode-final line)
    valid_pol = [(ep, v) for ep, v in zip(ep_nums, pol_loss_ep)
                 if v is not None and not math.isnan(v)]
    if valid_pol:
        ex, vy = zip(*valid_pol)
        _fill_plot(ax, list(ex), list(vy), C4, smooth_w=3)
        _trend(ax, list(ex), list(vy), C4)
    else:
        ax.text(0.5, 0.5, 'No policy loss data', transform=ax.transAxes,
                ha='center', color=FG, fontsize=9)

    # ── [1,1]  Value Loss ─────────────────────────────────────────────────
    ax = _ax(fig, gs, 1, 1, 'MAPPO Value (Critic) Loss\n(Critic accuracy over training)',
             ylabel='Value loss')
    valid_vl = [(ep, v) for ep, v in zip(ep_nums, val_loss_ep)
                if v is not None and not math.isnan(v)]
    if valid_vl:
        ex, vy = zip(*valid_vl)
        _fill_plot(ax, list(ex), list(vy), C6, smooth_w=3)
        _trend(ax, list(ex), list(vy), C6)
    else:
        ax.text(0.5, 0.5, 'No value loss data', transform=ax.transAxes,
                ha='center', color=FG, fontsize=9)

    # ── [1,2]  Reward vs Entropy scatter ──────────────────────────────────
    ax = _ax(fig, gs, 1, 2, 'Reward vs Entropy\n(Low entropy = decisive policy)',
             xlabel='Entropy', ylabel='Mean reward / tick')
    type_colors = {
        'full_core':    C1,
        'partial_core': C2,
        'zone_loss':    C3,
        'cascading':    C5,
    }
    for ep_i, (ep_n, sc, rew, ent) in enumerate(
            zip(ep_nums, sc_type, ep_reward, entropy_ep)):
        if ent is None or math.isnan(float(ent if ent is not None else float('nan'))):
            continue
        c = type_colors.get(sc, FG)
        ax.scatter(ent, rew, color=c, s=55, zorder=3,
                   label=sc if ep_i == sc_type.index(sc) else '')
        ax.annotate(f'Ep{ep_n}', (ent, rew), fontsize=6,
                    color=FG, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
    # Add a legend showing scenario type → color
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=col, label=sc, markersize=7)
        for sc, col in type_colors.items()
        if sc in sc_type
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # ── [2,0]  Per-tick Reward: Ep1 vs EpLast ─────────────────────────────
    ax = _ax(fig, gs, 2, 0, f'Per-Tick Reward: Ep1 vs Ep{n_ep}',
             xlabel='Tick within episode', ylabel='Reward')
    if ep1_ticks:
        t1 = series(ep1_ticks, 'tick')
        r1 = _smooth(series(ep1_ticks, 'reward'), w=7)
        ax.plot(t1, r1, '-', color=C1, linewidth=1.4, alpha=0.85, label='Ep 1')
    if epL_ticks:
        tL = series(epL_ticks, 'tick')
        rL = _smooth(series(epL_ticks, 'reward'), w=7)
        ax.plot(tL, rL, '-', color=C2, linewidth=2.0, alpha=0.9, label=f'Ep {n_ep}')
    ax.axhline(0, color=GRID, linewidth=0.7, linestyle=':')
    ax.legend(fontsize=7)

    # ── [2,1]  Per-tick Transport relay: Ep1 vs EpLast ─────────────────────────
    ax = _ax(fig, gs, 2, 1, f'Per-Tick Transport relay: Ep1 vs Ep{n_ep}',
             xlabel='Tick within episode', ylabel='Nodes in Transport relay mode')
    if ep1_ticks:
        t1 = series(ep1_ticks, 'tick')
        i1 = _smooth(series(ep1_ticks, 'transport_relay_count'), w=7)
        ax.plot(t1, i1, '-', color=C1, linewidth=1.4, alpha=0.85, label='Ep 1')
    if epL_ticks:
        tL = series(epL_ticks, 'tick')
        iL = _smooth(series(epL_ticks, 'transport_relay_count'), w=7)
        ax.plot(tL, iL, '-', color=C3, linewidth=2.0, alpha=0.9, label=f'Ep {n_ep}')
    ax.legend(fontsize=7)

    # ── [2,2]  Per-episode bar summary ────────────────────────────────────
    ax = _ax(fig, gs, 2, 2, 'Key Metrics: First vs Last Episode\n(Absolute values)',
             xlabel='Metric', ylabel='Value')
    ep1_data = episodes[0]
    epL_data = episodes[-1]
    labels   = ['Policy\nLoss', 'Entropy', 'Transport
Relays', 'Reward']
    def _safe(v):
        return float(v) if v is not None and not math.isnan(float(v if v is not None else float('nan'))) else 0.0

    vals1 = [
        _safe(ep1_data.get('final_policy_loss')),
        _safe(ep1_data.get('final_entropy')),
        _safe(iab_per_ep_mean[0] if iab_per_ep_mean else 0),
        _safe(ep1_data.get('ep_reward_mean')),
    ]
    valsL = [
        _safe(epL_data.get('final_policy_loss')),
        _safe(epL_data.get('final_entropy')),
        _safe(iab_per_ep_mean[-1] if iab_per_ep_mean else 0),
        _safe(epL_data.get('ep_reward_mean')),
    ]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, vals1, w, color=C1,  alpha=0.85, label='Episode 1')
    ax.bar(x + w/2, valsL, w, color=C2,  alpha=0.85, label=f'Episode {n_ep}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color=GRID, linewidth=0.6, linestyle=':')
    ax.legend(fontsize=7)
    # Delta annotations
    for i, (v1, v2) in enumerate(zip(vals1, valsL)):
        delta = v2 - v1
        if abs(delta) > 1e-6:
            sign = ('+' if delta >= 0 else '')
            col  = C2 if delta >= 0 else CSEV
            y_pos = max(v1, v2, 0) + abs(max(v1, v2, 0)) * 0.04 + 0.01
            ax.text(i + w/2, y_pos, f'{sign}{delta:.2f}',
                    ha='center', va='bottom', color=col, fontsize=7, fontweight='bold')

    # ── Footer ────────────────────────────────────────────────────────────────
    note = (
        f"Training: {n_ep} episodes x {len(ticks)//max(1,n_ep)} ticks  |  "
        f"Entropy: {_safe(entropy_ep[0]):.2f} -> {_safe(entropy_ep[-1]):.2f}  |  "
        f"Policy loss: {_safe(pol_loss_ep[0]):.4f} -> {_safe(pol_loss_ep[-1]):.4f}  |  "
        f"Transport relays ep1: {iab_per_ep_mean[0]:.1f}  ep{n_ep}: {iab_per_ep_mean[-1]:.1f}"
    )
    fig.text(0.5, 0.008, note, ha='center', color='#8b949e', fontsize=7.5)

    plt.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"[plot_learning] Saved -> {out_png}")


if __name__ == '__main__':
    kpi_json = sys.argv[1] if len(sys.argv) > 1 else 'output/learning_kpis.json'
    out_png  = sys.argv[2] if len(sys.argv) > 2 else 'output/marl_convergence.png'
    plot_convergence(kpi_json, out_png)
