"""
analysis.py — Post-simulation analysis & publication-quality plots

All four plots are driven from learning_kpis.json (the single source of truth
populated during MARL training).  The old MetricsCollector-based functions
produced empty plots because the training loop only feeds the KPI tracker, not
the legacy metrics_history / link_utilization / per_class_stats structures.

Generated plots
---------------
  link_utilization.png      — Transport relay link activity per episode (proxy for BH util)
  life_safety_success.png   — Per-tick life-safety reward signal + island phases
  energy_consumption.png    — Per-episode energy proxy: Transport relay load x ticks active
  traffic_statistics.png    — Multi-episode traffic KPIs: UE conn, relay adoption,
                              reward velocity, and entropy decay together

Usage (from main.py)
--------------------
  from sixg_sim.analysis import run_complete_analysis
  run_complete_analysis(metrics, topology.nodes, output_dir,
                        kpi_json="output/learning_kpis.json")
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── palette (light mode — article-ready) ──────────────────────────────────────
DARK = '#ffffff'      # figure background  → white
MID  = '#f8f9fa'      # axes background    → near-white
GRID = '#dee2e6'      # grid lines         → light grey
FG   = '#212529'      # text / labels      → near-black
DIM  = '#6c757d'      # secondary text     → medium grey
C1   = '#1b6ec2'      # blue
C2   = '#2f9e44'      # green
C3   = '#e67700'      # amber
C4   = '#862e9c'      # purple
C5   = '#d9480f'      # orange-red
C6   = '#0c8599'      # teal
CSEV = '#c92a2a'      # red (danger)

SCENARIO_COLORS = {
    'full_core':      C1,
    'partial_core':   C2,
    'zone_loss':      C3,
    'cascading':      C5,
    'multi_enb_iops': C4,
    'geo_disaster':   CSEV,
}


def _setup():
    plt.rcParams.update({
        'figure.facecolor': DARK, 'axes.facecolor': MID,
        'axes.edgecolor': GRID, 'axes.labelcolor': FG,
        'xtick.color': FG, 'ytick.color': FG, 'text.color': FG,
        'grid.color': GRID, 'grid.alpha': 0.5, 'grid.linestyle': '--',
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'legend.framealpha': 0.85, 'legend.facecolor': '#ffffff',
        'legend.edgecolor': GRID,
    })


def _ax(fig, gs, row, col, title, xlabel='', ylabel='', span=None):
    ax = fig.add_subplot(gs[row, col] if span is None else gs[row, col:col+span])
    ax.set_title(title, color=FG, fontsize=10, fontweight='bold', pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=8)
    ax.grid(True)
    return ax


def _smooth(vals, w=3):
    out = []
    for i, v in enumerate(vals):
        chunk = [x for x in vals[max(0, i-w+1):i+1]
                 if x is not None and not math.isnan(float(x))]
        out.append(sum(chunk)/len(chunk) if chunk else float('nan'))
    return out


def _load_kpis(kpi_json: str):
    """Load and return (episodes, ticks) lists, or ([], []) on failure."""
    try:
        with open(kpi_json) as f:
            d = json.load(f)
        return d.get('episodes', []), d.get('ticks', [])
    except Exception as e:
        print(f"[analysis] Could not load {kpi_json}: {e}")
        return [], []


# ── 1.  Link Utilization ──────────────────────────────────────────────────────

def plot_link_utilization(utilization_metrics=None,
                          output_file: str = None,
                          kpi_json: str = 'output/learning_kpis.json'):
    """
    Transport relay link activity per episode: mean + peak active relay nodes.
    Uses learning_kpis.json instead of the empty UtilizationMetrics object.
    """
    if not HAS_MPL:
        return
    _setup()

    episodes, ticks = _load_kpis(kpi_json)
    if not episodes:
        print("[analysis] link_utilization: no episode data in KPI JSON")
        return

    ep_nums = [e['episode'] for e in episodes]
    sc_type = [e.get('scenario_type', 'full_core') for e in episodes]

    # Per-episode mean Transport relay count from tick records
    from collections import defaultdict
    ep_relay_vals = defaultdict(list)
    ep_link_vals  = defaultdict(list)
    for t in ticks:
        ep = t['episode']
        ep_relay_vals[ep].append(t.get('transport_relay_count', 0))
        ep_link_vals[ep].append(t.get('transport_link_count', 0))

    mean_relays = [
        (sum(ep_relay_vals[ep]) / len(ep_relay_vals[ep])) if ep_relay_vals[ep] else 0
        for ep in ep_nums
    ]
    peak_relays = [max(ep_relay_vals[ep], default=0) for ep in ep_nums]
    mean_links  = [
        (sum(ep_link_vals[ep]) / len(ep_link_vals[ep])) if ep_link_vals[ep] else 0
        for ep in ep_nums
    ]

    fig = plt.figure(figsize=(13, 7), facecolor=DARK)
    fig.suptitle('Transport relay Link Utilisation — Per Episode\n'
                 '(Proxy for backhaul link activity in island mode)',
                 color=FG, fontsize=12, fontweight='bold', y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.96, top=0.90, bottom=0.08)

    # [0,0] Mean active relay nodes per episode
    ax = _ax(fig, gs, 0, 0, 'Mean Active Transport relay Nodes', 'Episode', 'Node count')
    sm = _smooth(mean_relays)
    ax.fill_between(ep_nums, mean_relays, alpha=0.18, color=C3)
    ax.plot(ep_nums, mean_relays, 'o--', color=C3, alpha=0.5, ms=3, lw=0.9)
    ax.plot(ep_nums, sm, '-', color=C3, lw=2.2, label='Mean relays (smoothed)')
    ax.legend(fontsize=7)

    # [0,1] Peak relay nodes per episode (bar coloured by scenario type)
    ax = _ax(fig, gs, 0, 1, 'Peak Transport relay Nodes per Episode', 'Episode', 'Node count')
    bar_colors = [SCENARIO_COLORS.get(sc, C1) for sc in sc_type]
    bars = ax.bar(ep_nums, peak_relays, color=bar_colors, edgecolor=GRID, alpha=0.85)
    for bar, v in zip(bars, peak_relays):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    str(int(v)), ha='center', va='bottom', color=FG, fontsize=7)
    # Scenario type legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=sc)
                      for sc, c in SCENARIO_COLORS.items()
                      if sc in sc_type]
    ax.legend(handles=legend_handles, fontsize=7, loc='upper left')

    # [1,0] Mean active transport relay links per episode
    ax = _ax(fig, gs, 1, 0, 'Mean Active Transport Relay Links', 'Episode', 'Link count')
    sm2 = _smooth(mean_links)
    ax.fill_between(ep_nums, mean_links, alpha=0.18, color=C6)
    ax.plot(ep_nums, mean_links, 'o--', color=C6, alpha=0.5, ms=3, lw=0.9)
    ax.plot(ep_nums, sm2, '-', color=C6, lw=2.2, label='Mean transport relay links')
    ax.legend(fontsize=7)

    # [1,1] Relay utilisation rate: mean_relays / peak_relays
    ax = _ax(fig, gs, 1, 1, 'Relay Utilisation Rate\n(Mean / Peak per episode)',
             'Episode', 'Utilisation ratio')
    util_rate = [m / max(p, 1) for m, p in zip(mean_relays, peak_relays)]
    sm3 = _smooth(util_rate)
    ax.fill_between(ep_nums, util_rate, alpha=0.18, color=C2)
    ax.plot(ep_nums, util_rate, 'o--', color=C2, alpha=0.5, ms=3, lw=0.9)
    ax.plot(ep_nums, sm3, '-', color=C2, lw=2.2, label='Utilisation rate')
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color=GRID, lw=0.7, ls=':')
    ax.legend(fontsize=7)

    fig.text(0.5, 0.01,
             f'Episodes: {len(episodes)}  |  Total ticks: {len(ticks):,}  |  '
             f'Max peak relays: {max(peak_relays) if peak_relays else 0}',
             ha='center', color=DIM, fontsize=7.5)

    plt.savefig(output_file, dpi=160, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"[analysis] link_utilization saved -> {output_file}")


# ── 2.  Life-Safety Success ────────────────────────────────────────────────────

def plot_life_safety_success_over_time(metrics=None,
                                       output_file: str = None,
                                       kpi_json: str = 'output/learning_kpis.json'):
    """
    Per-tick reward signal as a life-safety proxy, with island phase shading
    and per-episode convergence overlay.
    """
    if not HAS_MPL:
        return
    _setup()

    episodes, ticks = _load_kpis(kpi_json)
    if not ticks:
        print("[analysis] life_safety_success: no tick data")
        return

    fig = plt.figure(figsize=(14, 9), facecolor=DARK)
    fig.suptitle('Life-Safety & Recovery Signal — Training Episodes\n'
                 '(Reward encodes emergency UE routing + relay formation)',
                 color=FG, fontsize=12, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.35,
                           left=0.08, right=0.96, top=0.88, bottom=0.09)

    ep_nums  = [e['episode'] for e in episodes]
    ep_rew   = [e.get('ep_reward_mean', float('nan')) for e in episodes]
    sc_type  = [e.get('scenario_type', 'full_core') for e in episodes]
    entropy  = [e.get('final_entropy', float('nan')) for e in episodes]

    # [0,0-1]  Per-tick reward across ALL ticks, coloured by episode
    ax = _ax(fig, gs, 0, 0, 'Per-Tick Reward Signal — All Episodes\n'
             '(Higher = better emergency routing)',
             'Tick (global)', 'Reward / tick', span=2)
    n_ep = len(episodes)
    cmap = plt.get_cmap('plasma', max(n_ep, 1))
    for ep_i, ep_rec in enumerate(episodes):
        ep_ticks = [t for t in ticks if t['episode'] == ep_rec['episode']]
        if not ep_ticks:
            continue
        xs = [t['tick'] + ep_i * (max(t['tick'] for t in ep_ticks) + 2)
              for t in ep_ticks]  # offset per episode
        ys = _smooth([t.get('reward', 0) for t in ep_ticks], w=10)
        color = cmap(ep_i / max(n_ep - 1, 1))
        ax.plot(xs, ys, '-', color=color, lw=1.1, alpha=0.75,
                label=f'Ep{ep_rec["episode"]} ({sc_type[ep_i][:4]})')
    ax.axhline(0, color=GRID, lw=0.7, ls=':')
    ax.legend(fontsize=6, ncol=min(n_ep, 6), loc='lower right')

    # [1,0]  Episode reward trend + scenario colour
    ax = _ax(fig, gs, 1, 0, 'Episode Mean Reward\n(Trend = policy improvement)',
             'Episode', 'Mean reward / tick')
    bar_colors = [SCENARIO_COLORS.get(sc, C1) for sc in sc_type]
    ax.bar(ep_nums, ep_rew, color=bar_colors, edgecolor=GRID, alpha=0.7)
    sm = _smooth(ep_rew)
    ax.plot(ep_nums, sm, '-o', color=FG, lw=2, ms=4, label='Smoothed trend')
    ax.axhline(0, color=GRID, lw=0.7, ls=':')
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=sc)
                      for sc, c in SCENARIO_COLORS.items()
                      if sc in sc_type]
    legend_handles.append(
        plt.Line2D([0], [0], color=FG, lw=2, label='Trend'))
    ax.legend(handles=legend_handles, fontsize=7)

    # [1,1]  Entropy decay (policy confidence = less random = better routing)
    ax = _ax(fig, gs, 1, 1, 'Policy Entropy Decay\n(Entropy ↓ = more decisive emergency routing)',
             'Episode', 'Entropy (nats)')
    valid = [(ep, v) for ep, v in zip(ep_nums, entropy)
             if v is not None and not math.isnan(v)]
    if valid:
        ex, vy = zip(*valid)
        sm2 = _smooth(list(vy))
        ax.fill_between(list(ex), list(vy), alpha=0.18, color=C5)
        ax.plot(list(ex), list(vy), 'o--', color=C5, alpha=0.5, ms=3, lw=0.9)
        ax.plot(list(ex), sm2, '-', color=C5, lw=2.2, label='Entropy')
        # Annotate start/end
        ax.annotate(f'{vy[0]:.2f}', (ex[0], vy[0]), color=C5, fontsize=8,
                    xytext=(4, 6), textcoords='offset points')
        ax.annotate(f'{vy[-1]:.2f}', (ex[-1], vy[-1]), color=C2, fontsize=8,
                    xytext=(-20, 6), textcoords='offset points')
        ax.axhline(vy[-1], color=C2, lw=0.8, ls=':', alpha=0.6)
        ax.legend(fontsize=7)

    plt.savefig(output_file, dpi=160, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"[analysis] life_safety_success saved -> {output_file}")


# ── 3.  Energy Consumption ────────────────────────────────────────────────────

def plot_energy_consumption(metrics=None, topology_nodes=None,
                            output_file: str = None,
                            kpi_json: str = 'output/learning_kpis.json'):
    """
    Energy proxy plots:
      - Per-episode active relay x tick load (relay node energy effort)
      - Per-episode transport relay link count area chart
      - Node-type energy breakdown from simulation_summary.txt (if present)
    """
    if not HAS_MPL:
        return
    _setup()

    episodes, ticks = _load_kpis(kpi_json)
    if not episodes:
        print("[analysis] energy_consumption: no episode data")
        return

    ep_nums = [e['episode'] for e in episodes]
    sc_type = [e.get('scenario_type', 'full_core') for e in episodes]

    # Energy proxy: sum(relay_count * 1.0) per episode tick = total "relay work"
    from collections import defaultdict
    ep_relay_work = defaultdict(float)
    ep_link_area   = defaultdict(float)
    ep_tick_counts = defaultdict(int)
    for t in ticks:
        ep = t['episode']
        ep_relay_work[ep] += t.get('transport_relay_count', 0)
        ep_link_area[ep]  += t.get('transport_link_count',  0)
        ep_tick_counts[ep] += 1

    relay_work = [ep_relay_work[ep] / max(1, ep_tick_counts[ep]) for ep in ep_nums]
    link_area  = [ep_link_area[ep]  / max(1, ep_tick_counts[ep]) for ep in ep_nums]

    # Try to get real node energy from simulation_summary.txt
    real_energy: dict[str, float] = {}
    summary_path = str(Path(output_file).parent / 'simulation_summary.txt')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for line in f:
                # Format: "  node_id: XXX.X units used, final SoC: Y.YY"
                if 'units used' in line and 'SoC' in line:
                    try:
                        parts = line.strip().split(':')
                        node_id = parts[0].strip()
                        used = float(parts[1].strip().split()[0])
                        if used > 0:
                            real_energy[node_id] = used
                    except Exception:
                        pass

    fig = plt.figure(figsize=(14, 8), facecolor=DARK)
    fig.suptitle('Energy Consumption Analysis\n'
                 '(Relay work proxy + real node energy from simulation)',
                 color=FG, fontsize=12, fontweight='bold', y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35,
                           left=0.08, right=0.96, top=0.90, bottom=0.08)

    # [0,0] Relay work per episode
    ax = _ax(fig, gs, 0, 0, 'Mean Relay Node Load per Episode\n'
             '(Avg active relay nodes x time = energy effort)',
             'Episode', 'Mean relay nodes active')
    bar_colors = [SCENARIO_COLORS.get(sc, C1) for sc in sc_type]
    ax.bar(ep_nums, relay_work, color=bar_colors, edgecolor=GRID, alpha=0.82)
    ax.plot(ep_nums, _smooth(relay_work), '-o', color=FG, lw=2, ms=4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=sc)
                        for sc, c in SCENARIO_COLORS.items() if sc in sc_type],
              fontsize=7, loc='upper left')

    # [0,1] transport relay link area (backhaul energy)
    ax = _ax(fig, gs, 0, 1, 'Mean Transport Relay Links per Episode\n'
             '(More links = more backhaul power consumption)',
             'Episode', 'Mean transport relay links')
    ax.fill_between(ep_nums, link_area, alpha=0.22, color=C6)
    ax.plot(ep_nums, link_area, 'o-', color=C6, lw=2, ms=5, label='Mean links')
    ax.plot(ep_nums, _smooth(link_area), '-', color=C6, lw=2.5, alpha=0.8)
    ax.legend(fontsize=7)

    # [1,0-1] Real node energy breakdown from simulation_summary.txt
    if real_energy:
        ax = _ax(fig, gs, 1, 0, f'Real Node Energy Consumption by Zone\n'
                 f'({len(real_energy)} infra nodes with non-zero usage)',
                 'Node zone', 'Energy units used', span=2)
        # Group by type_zone (e.g. relay_north, relay_south, gnb_north)
        # Use first two underscore-separated tokens as the group key
        groups: dict[str, list[tuple[str, float]]] = {}
        for nid, val in sorted(real_energy.items(), key=lambda x: -x[1]):
            parts = nid.split('_')
            key = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
            groups.setdefault(key, []).append((nid, val))

        # Compute group aggregates
        g_labels = sorted(groups.keys())
        g_totals = [sum(v for _, v in groups[g]) for g in g_labels]
        g_counts = [len(groups[g]) for g in g_labels]
        g_means  = [t / c for t, c in zip(g_totals, g_counts)]

        xs = np.arange(len(g_labels))
        w  = 0.35
        ax.bar(xs - w/2, g_totals, w, color=C4, alpha=0.85, label='Total energy used')
        ax.bar(xs + w/2, g_means,  w, color=C3, alpha=0.85, label='Mean per node')
        ax.set_xticks(xs)
        ax.set_xticklabels(g_labels, rotation=35, ha='right', fontsize=7.5)
        ax.set_ylim(0, max(g_totals) * 1.18 if g_totals else 100)
        ax.legend(fontsize=7)
    else:
        ax = _ax(fig, gs, 1, 0, 'Node Energy (unavailable — run simulation first)',
                 span=2)
        ax.text(0.5, 0.5, 'No simulation_summary.txt found with energy data',
                transform=ax.transAxes, ha='center', color=DIM, fontsize=9)

    plt.savefig(output_file, dpi=160, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"[analysis] energy_consumption saved -> {output_file}")


# ── 4.  Traffic Statistics ────────────────────────────────────────────────────

def plot_traffic_statistics(traffic_metrics=None,
                            output_file: str = None,
                            kpi_json: str = 'output/learning_kpis.json'):
    """
    4-panel traffic & learning summary:
      [0,0] UE connectivity fraction per episode + trend
      [0,1] Transport relay adoption (relay + link count) per episode
      [1,0] Reward velocity: episode-over-episode improvement
      [1,1] Training efficiency: reward vs entropy scatter by scenario type
    """
    if not HAS_MPL:
        return
    _setup()

    episodes, ticks = _load_kpis(kpi_json)
    if not episodes:
        print("[analysis] traffic_statistics: no episode data")
        return

    ep_nums  = [e['episode'] for e in episodes]
    sc_type  = [e.get('scenario_type', 'full_core') for e in episodes]
    ep_rew   = [e.get('ep_reward_mean',   float('nan')) for e in episodes]
    pol_loss = [e.get('final_policy_loss', float('nan')) for e in episodes]
    entropy  = [e.get('final_entropy',     float('nan')) for e in episodes]
    peak_relays = [max(e.get('post_sev_transport_relay', 0),
                    e.get('peak_transport_relay', 0)) for e in episodes]

    # Per-episode mean Transport relay and link from ticks
    from collections import defaultdict
    ep_relay_mean = defaultdict(list)
    ep_link_mean  = defaultdict(list)
    for t in ticks:
        ep_relay_mean[t['episode']].append(t.get('transport_relay_count', 0))
        ep_link_mean[t['episode']].append(t.get('transport_link_count', 0))

    mean_relays = [
        sum(ep_relay_mean[ep]) / max(1, len(ep_relay_mean[ep]))
        for ep in ep_nums
    ]
    mean_links = [
        sum(ep_link_mean[ep]) / max(1, len(ep_link_mean[ep]))
        for ep in ep_nums
    ]

    fig = plt.figure(figsize=(14, 9), facecolor=DARK)
    fig.suptitle('Traffic & Learning Statistics — Multi-Episode Training\n'
                 '(Transport relay adoption, reward velocity, and policy efficiency)',
                 color=FG, fontsize=12, fontweight='bold', y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35,
                           left=0.08, right=0.96, top=0.90, bottom=0.08)

    bar_colors = [SCENARIO_COLORS.get(sc, C1) for sc in sc_type]

    # [0,0] Transport relay adoption per episode
    ax = _ax(fig, gs, 0, 0, 'Transport relay Adoption\n(Mean active relay nodes per episode)',
             'Episode', 'Relay nodes')
    ax.fill_between(ep_nums, mean_relays, alpha=0.18, color=C2)
    ax.plot(ep_nums, mean_relays, 'o--', color=C2, alpha=0.5, ms=4, lw=1)
    ax.plot(ep_nums, _smooth(mean_relays), '-', color=C2, lw=2.4, label='Mean relays')
    ax2 = ax.twinx()
    ax2.plot(ep_nums, mean_links, 's--', color=C3, lw=1.4, ms=3, alpha=0.7,
             label='Mean transport relay links')
    ax2.set_ylabel('transport relay links', color=C3, fontsize=8)
    ax2.tick_params(axis='y', colors=C3)
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='lower right', fontsize=7)

    # [0,1] Policy loss and entropy co-evolution
    ax = _ax(fig, gs, 0, 1, 'Policy Loss & Entropy Co-evolution\n'
             '(Both decreasing = policy converging efficiently)',
             'Episode', 'Policy loss')
    valid_pl = [(ep, v) for ep, v in zip(ep_nums, pol_loss)
                if v is not None and not math.isnan(v)]
    valid_ent = [(ep, v) for ep, v in zip(ep_nums, entropy)
                 if v is not None and not math.isnan(v)]
    if valid_pl:
        ex_pl, vy_pl = zip(*valid_pl)
        ax.fill_between(list(ex_pl), list(vy_pl), alpha=0.15, color=C4)
        ax.plot(list(ex_pl), list(vy_pl), 'o-', color=C4, lw=2, ms=4,
                label='Policy loss')
    ax3 = ax.twinx()
    ax3.set_ylabel('Entropy (nats)', color=C5, fontsize=8)
    ax3.tick_params(axis='y', colors=C5)
    if valid_ent:
        ex_e, vy_e = zip(*valid_ent)
        ax3.plot(list(ex_e), list(vy_e), 's--', color=C5, lw=1.5, ms=3,
                 alpha=0.8, label='Entropy')
    ax.legend(loc='upper right', fontsize=7)
    ax3.legend(loc='lower right', fontsize=7)

    # [1,0] Reward velocity (episode-over-episode change)
    ax = _ax(fig, gs, 1, 0, 'Reward Velocity\n(Change in mean reward episode-to-episode)',
             'Episode', 'Δ Reward / episode')
    if len(ep_rew) > 1:
        valid_rew = [r for r in ep_rew if not math.isnan(r)]
        if len(valid_rew) > 1:
            deltas = [valid_rew[i] - valid_rew[i-1] for i in range(1, len(valid_rew))]
            ep_nums_v = ep_nums[1:len(valid_rew)]
            delta_colors = [C2 if d >= 0 else CSEV for d in deltas]
            ax.bar(ep_nums_v, deltas, color=delta_colors, edgecolor=GRID, alpha=0.85,
                   label='Δ Reward')
            ax.axhline(0, color=GRID, lw=0.8, ls='--')
            # Moving average of deltas
            if len(deltas) >= 3:
                sm_d = _smooth(deltas, w=3)
                ax.plot(ep_nums_v, sm_d, '-', color=FG, lw=1.8, label='Trend')
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color=C2,   label='Improvement'),
                Patch(color=CSEV, label='Degradation'),
                plt.Line2D([0], [0], color=FG, lw=2, label='Trend'),
            ], fontsize=7)

    # [1,1] Reward vs Policy loss: efficiency scatter coloured by scenario + ep
    ax = _ax(fig, gs, 1, 1, 'Training Efficiency\n'
             '(Lower loss + higher reward = better learning)',
             'Policy loss', 'Mean reward / tick')
    cmap = plt.get_cmap('viridis', max(len(ep_nums), 1))
    for i, (ep, sc, rew, pl) in enumerate(
            zip(ep_nums, sc_type, ep_rew, pol_loss)):
        if math.isnan(rew) or math.isnan(pl):
            continue
        sc_c = SCENARIO_COLORS.get(sc, C1)
        ax.scatter(pl, rew, color=sc_c, s=65, zorder=3,
                   edgecolors=FG, linewidths=0.5)
        ax.annotate(f'Ep{ep}', (pl, rew), fontsize=6.5, color=FG,
                    xytext=(3, 3), textcoords='offset points', alpha=0.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=sc)
                        for sc, c in SCENARIO_COLORS.items() if sc in sc_type],
              fontsize=7, loc='lower left')

    # Footer
    n_ep = len(episodes)
    valid_entropy = [e for e in entropy if not math.isnan(e)]
    valid_pol_loss = [p for p in pol_loss if not math.isnan(p)]
    ent_str = f'{valid_entropy[0]:.2f}->{valid_entropy[-1]:.2f}' if valid_entropy else 'N/A'
    pol_str = f'{valid_pol_loss[0]:.4f}->{valid_pol_loss[-1]:.4f}' if valid_pol_loss else 'N/A'

    fig.text(0.5, 0.01,
             f'Training: {n_ep} episodes x {len(ticks)//max(1,n_ep)} ticks  |  '
             f'Scenarios: {", ".join(sorted(set(sc_type)))}  |  '
             f'Entropy: {ent_str}  |  '
             f'Policy loss: {pol_str}',
             ha='center', color=DIM, fontsize=7.5)

    plt.savefig(output_file, dpi=160, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"[analysis] traffic_statistics saved -> {output_file}")


# ── Main entry point (called from main.py) ────────────────────────────────────

def run_complete_analysis(metrics=None, topology_nodes=None,
                          output_dir: str = 'output',
                          kpi_json: str = None):
    """Run complete analysis suite and regenerate all four plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Locate kpi_json automatically if not provided
    if kpi_json is None:
        kpi_json = os.path.join(output_dir, 'learning_kpis.json')

    if not os.path.exists(kpi_json):
        print(f"[analysis] Warning: {kpi_json} not found. "
              "Plots will be empty — run training first.")
        return

    plot_link_utilization(
        output_file=os.path.join(output_dir, 'link_utilization.png'),
        kpi_json=kpi_json)
    plot_life_safety_success_over_time(
        output_file=os.path.join(output_dir, 'life_safety_success.png'),
        kpi_json=kpi_json)
    plot_energy_consumption(
        topology_nodes=topology_nodes,
        output_file=os.path.join(output_dir, 'energy_consumption.png'),
        kpi_json=kpi_json)
    plot_traffic_statistics(
        output_file=os.path.join(output_dir, 'traffic_statistics.png'),
        kpi_json=kpi_json)

    print(f"\n[analysis] All plots saved to {output_dir}/")


# ── Standalone regeneration ───────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    kpi  = sys.argv[1] if len(sys.argv) > 1 else 'output/learning_kpis.json'
    odir = sys.argv[2] if len(sys.argv) > 2 else 'output'
    run_complete_analysis(output_dir=odir, kpi_json=kpi)
