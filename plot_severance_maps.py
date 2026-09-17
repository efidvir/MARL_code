"""
plot_severance_maps.py — geographic maps of every severance scenario used in
the UNITY-6G MARL disaster-recovery study.

WHY THIS SCRIPT EXISTS
----------------------
The published results rest on a *fragmenting* severance: the core is cut AND a
geographically-correlated damage corridor partitions the surviving transport
mesh into disconnected islands.  Curves alone do not show what the disaster
actually did to the network.  This script draws it: real node geometry, real
link cuts, real post-severance connected components.

NOTHING HERE IS SIMULATED AND NOTHING IS HARDCODED.  Every number printed on
every map is re-derived from the SAME builders the evaluation and training
drivers call:

    run_timeline_comparison.build_comparison_topology(seed)
    run_timeline_comparison._build_fragmenting_severance(topo, tick, seed)
    run_timeline_comparison._replicate_ue_flows(topo)

Because that construction is a pure deterministic function of the seed, the
partition drawn here is byte-for-byte the partition every arm experienced.
All arms (MARL, OSPF, SDN, OLSR, BATMAN, AODV) receive the identical event
list — the maps therefore describe the shared environment, not any one arm.

WHAT IS DRAWN
-------------
  * Infrastructure nodes at their true (x_pos, y_pos), distinct marker per
    node type; MultiHaul relay-capable sites (the bridging hardware) are
    drawn as large stars so they stand out.
  * Infrastructure links: surviving links neutral grey; partition cuts in
    red dashed; links killed by `sever_core` in red dotted.
  * Post-severance fragments: infra nodes coloured by connected component of
    the INFRASTRUCTURE graph only.  UEs are excluded from the graph — a UE
    dual-homed onto two islands must never be allowed to merge them.
  * UEs: small faded dots, tinted by the fragment of their serving anchor
    (grey when their anchors straddle fragments).  UE-to-UE flows are NOT
    drawn as lines (222 of them would be unreadable) — the cross-fragment
    flow count and share appear in the panel annotation instead.

USAGE
-----
    python plot_severance_maps.py                      # everything (default)
    python plot_severance_maps.py --seeds 50,51 --kind recovery
    python plot_severance_maps.py --training-samples 6
    python plot_severance_maps.py --outdir output/maps_v2

Outputs (under --outdir's parent for the contact sheets):
    output/maps/map_scenA_seed50.png ...      (one PNG per scenario instance)
    output/FINAL_severance_maps_eval.png      (10 evaluation maps, 2x5 grid)
    output/FINAL_severance_maps_training.png  (training sample grid)
    output/severance_scenarios.csv            (machine-readable summary)

This script is standalone: it never modifies run_timeline_comparison.py,
train_on_comparison.py or anything under sixg_sim/.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import csv
import contextlib
import io
import math
import re
import sys
import time

try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS MIRRORED FROM THE ENGINE  (single source of truth documented)
# ══════════════════════════════════════════════════════════════════════
# Exactly Simulator._identify_core_nodes() (sixg_sim/simulation.py):
# EdgeUPFs are deliberately NOT core — they survive severance and steer
# local DN traffic (3GPP TS 23.501 §6.3.3).
CORE_TYPES = {"Core", "SMO", "Non-RT-RIC", "AMF", "UPF"}

# Held-out evaluation seeds (train_on_comparison.HELD_OUT_EVAL_SEEDS).
EVAL_SEEDS = [50, 51, 52, 53, 54]

# Verified expected values for Scenario A on the evaluation seeds.  These are
# the published numbers.  The script CHECKS against them and reports any
# disagreement loudly — it never adjusts anything to make them match.
EXPECTED_SCEN_A = {
    #  seed: (n_fragments, cross_fragment_share, n_cut_links)
    50: (4, 0.378, 33),
    51: (4, 0.554, 22),
    52: (4, 0.378, 36),
    53: (4, 0.392, 32),
    54: (4, 0.464, 33),
}

CAPTION = (
    "All arms (MARL-RIC, OSPF, SDN, OLSR, B.A.T.M.A.N., AODV) experience the "
    "IDENTICAL severance: the event list is a pure deterministic function of "
    "the seed, built once and replayed per arm. Fragment counts, cut-link "
    "counts and cross-fragment flow shares shown here are MEASURED from the "
    "scenario builder (run_timeline_comparison._build_fragmenting_severance), "
    "not from any simulation run and not from literals. Fragments are "
    "connected components of the INFRASTRUCTURE graph only; UEs are excluded "
    "so that a dual-homed UE cannot spuriously merge two islands."
)


# ══════════════════════════════════════════════════════════════════════
#  VISUAL STYLE
# ══════════════════════════════════════════════════════════════════════
# marker, size, z-order, label  — keyed by NodeType.value
NODE_STYLE = {
    'O-RU':        dict(marker='^', size=52,  z=5, label='O-RU (radio unit)'),
    'O-DU':        dict(marker='s', size=52,  z=5, label='O-DU'),
    'O-CU-CP':     dict(marker='D', size=46,  z=5, label='O-CU-CP'),
    'O-CU-UP':     dict(marker='d', size=52,  z=5, label='O-CU-UP'),
    'EdgeUPF':     dict(marker='P', size=66,  z=5, label='Edge UPF (survives)'),
    'Relay':       dict(marker='o', size=44,  z=5, label='Relay site'),
    'Near-RT-RIC': dict(marker='v', size=52,  z=5, label='Near-RT-RIC'),
    'SMO':         dict(marker='X', size=60,  z=4, label='SMO (core, down)'),
    'AMF':         dict(marker='X', size=60,  z=4, label='AMF (core, down)'),
    'UPF':         dict(marker='X', size=60,  z=4, label='UPF (core, down)'),
}
MULTIHAUL_STYLE = dict(marker='*', size=230, z=7)

# Colour-blind-safe qualitative palette for fragments (Okabe-Ito derived).
FRAG_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
               '#E69F00', '#56B4E9', '#8C564B', '#7F7F7F']
CORE_COLOR = '#9E9E9E'
SURVIVING_LINK_COLOR = '#B0B7BE'
CUT_LINK_COLOR = '#D62728'
GEO_SIZE = 8000.0        # build_comparison_topology lays out an 8 km x 8 km grid


def frag_color(i):
    return FRAG_COLORS[i % len(FRAG_COLORS)]


# ══════════════════════════════════════════════════════════════════════
#  BUILDER IMPORT  (with the retry the brief asks for)
# ══════════════════════════════════════════════════════════════════════

_BUILDERS = {}


def load_builders(retry_wait_s=180.0):
    """Import the scenario builders from run_timeline_comparison.

    Importing that module is safe: everything executable is behind
    `if __name__ == "__main__":` (verified) — the import only defines
    functions/classes and sets module constants.  If the module is mid-edit
    by another agent and fails to import, wait and retry once, exactly as
    instructed, before giving up loudly.
    """
    if _BUILDERS:
        return _BUILDERS
    for attempt in (1, 2):
        try:
            t0 = time.time()
            import run_timeline_comparison as R
            _BUILDERS.update(
                build_topology=R.build_comparison_topology,
                build_frag=R._build_fragmenting_severance,
                replicate_flows=R._replicate_ue_flows,
                SEVERANCE_TICK=R.SEVERANCE_TICK,
                RESCUE_ARRIVAL_TICK=R.RESCUE_ARRIVAL_TICK,
                TOTAL_TICKS=R.TOTAL_TICKS,
            )
            print(f"[MAPS] builders imported from run_timeline_comparison "
                  f"({time.time() - t0:.1f}s)")
            return _BUILDERS
        except Exception as exc:
            if attempt == 1:
                print(f"[MAPS] WARNING: importing run_timeline_comparison "
                      f"failed ({exc!r}). Another agent may be mid-edit — "
                      f"waiting {retry_wait_s:.0f}s and retrying once.")
                time.sleep(retry_wait_s)
            else:
                raise SystemExit(
                    f"[MAPS] FATAL: could not import the scenario builders "
                    f"after a retry ({exc!r}). No maps were produced. The "
                    f"builders are the only source of truth for the "
                    f"partition — nothing is invented here.")


def load_training_plan():
    """Import the training episode plan from train_on_comparison.

    Also guarded by `if __name__ == '__main__':`, so the import is inert.
    Returns (randomized_event_ticks, scenario_kinds, held_out_seeds,
             default_seed_pool, default_episode_ticks, default_base_seed) or
    None when unavailable (the training sample then falls back to an
    explicitly-described local plan).
    """
    try:
        import train_on_comparison as T
        return dict(
            randomized_ticks=T._randomized_event_ticks,
            kinds=T.SCENARIO_KINDS,
            held_out=T.HELD_OUT_EVAL_SEEDS,
            seed_pool=list(range(42, 50)) + list(range(100, 140)),
            episode_ticks=800,     # train_on_comparison --episode-ticks default
            base_seed=42,          # train_on_comparison --base-seed default
        )
    except Exception as exc:
        print(f"[MAPS] WARNING: could not import train_on_comparison "
              f"({exc!r}); training-sample timing will be omitted.")
        return None


# ══════════════════════════════════════════════════════════════════════
#  DERIVATION — one scenario instance -> everything needed to draw it
# ══════════════════════════════════════════════════════════════════════

_BUILDER_LINE = re.compile(
    r"fragments=(\d+) cross_fragment_flows=(\d+)/(\d+).*?"
    r"cut_pairs=(\d+), cut_links=(\d+)")


def derive_instance(seed, kind, severance_tick=None, rescue_tick=None,
                    scenario_seed=None, group='eval', label=None,
                    rescue_zone='zone_0', rescue_num_ues=20):
    """Re-derive one scenario instance from the builders.

    Returns a plain dict of plot data + measured statistics.  The heavy
    Topology object is NOT retained.
    """
    B = load_builders()
    if severance_tick is None:
        severance_tick = 0 if kind == 'rescue_ops' else B['SEVERANCE_TICK']
    if kind == 'rescue_ops' and rescue_tick is None:
        rescue_tick = B['RESCUE_ARRIVAL_TICK']
    if scenario_seed is None:
        scenario_seed = seed

    # ── topology (identical call the physics pass makes) ──
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        topology, topo_info = B['build_topology'](seed)
    ntype = {nid: n.node_type.value for nid, n in topology.nodes.items()}
    pos = {nid: (n.x_pos, n.y_pos) for nid, n in topology.nodes.items()}
    multihaul = {nid for nid, n in topology.nodes.items()
                 if getattr(n, 'has_multihaul', False)}
    coverage = {nid: getattr(n, 'coverage_area', None)
                for nid, n in topology.nodes.items()}

    core = {nid for nid in topology.nodes if ntype[nid] in CORE_TYPES}
    ue_ids = [nid for nid in topology.nodes if ntype[nid] == 'UE']
    infra_surv = [nid for nid, n in topology.nodes.items()
                  if nid not in core and ntype[nid] != 'UE' and n.is_survivor]
    infra_set = set(infra_surv)

    # ── the severance event list, straight from the builder ──
    builder_stats = None
    cut_lids = set()
    if kind == 'recovery':
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            events = B['build_frag'](topology, severance_tick, scenario_seed)
        cut_lids = {e.parameters['link_id'] for e in events}
        for line in buf2.getvalue().splitlines():
            if '[SCENARIO]' in line:
                m = _BUILDER_LINE.search(line)
                if m:
                    g = [int(x) for x in m.groups()]
                    builder_stats = dict(fragments=g[0], cross=g[1],
                                         total=g[2], cut_pairs=g[3],
                                         cut_links=g[4])
                break
    # kind == 'rescue_ops': scenario events are sever_core@0 +
    # rescue_force_arrival only — no fail_link cuts (see
    # build_comparison_scenario).  The fragmentation is whatever sever_core
    # alone leaves behind, which is the honest picture for Scenario B.

    # ── link classification ──
    # 'core' : killed by sever_core (touches a core node)
    # 'cut'  : killed by the fragmenting fail_link events
    # 'up'   : survives
    # 'uu'   : UE access link (not part of the infrastructure graph)
    edges = []
    n_core_links = 0
    for lid, link in topology.links.items():
        u, v = link.endpoints
        if not getattr(link, 'is_up', True):
            continue
        if u in core or v in core:
            edges.append((u, v, 'core'))
            n_core_links += 1
        elif ntype[u] == 'UE' or ntype[v] == 'UE':
            edges.append((u, v, 'uu'))
        elif lid in cut_lids:
            edges.append((u, v, 'cut'))
        else:
            edges.append((u, v, 'up'))

    # ── post-severance INFRASTRUCTURE graph (UEs excluded by construction) ──
    G = nx.Graph()
    G.add_nodes_from(infra_surv)
    for u, v, kindtag in edges:
        if kindtag == 'up':
            G.add_edge(u, v)
    comps = sorted((set(c) for c in nx.connected_components(G)),
                   key=len, reverse=True)
    frag_of = {nid: i for i, c in enumerate(comps) for nid in c}
    comp_sizes = [len(c) for c in comps]

    # ── UE -> serving anchors -> fragment(s) ──
    ue_anchors = {}
    for u, v, kindtag in edges:
        if kindtag != 'uu':
            continue
        for ue, anc in ((u, v), (v, u)):
            if ntype[ue] == 'UE' and anc in infra_set:
                ue_anchors.setdefault(ue, set()).add(anc)
    ue_frags = {ue: {frag_of[a] for a in anchors if a in frag_of}
                for ue, anchors in ue_anchors.items()}

    # ── UE-to-UE flows and how many span fragments ──
    flows = B['replicate_flows'](topology)
    total_flows = len(flows)
    cross_flows = sum(1 for s, t in flows
                      if ue_frags.get(s) and ue_frags.get(t)
                      and ue_frags[s].isdisjoint(ue_frags[t]))
    share = cross_flows / max(1, total_flows)

    # ── cross-check against the builder's own printed line ──
    mismatch = None
    if builder_stats is not None:
        want = (builder_stats['fragments'], builder_stats['cross'],
                builder_stats['total'], builder_stats['cut_links'])
        got = (len(comps), cross_flows, total_flows, len(cut_lids))
        if want != got:
            mismatch = (f"builder said fragments/cross/total/cut_links="
                        f"{want}, re-derivation got {got}")

    # ── Scenario B: where the rescue force lands ──
    # rescue_force_arrival calls Topology.add_ue_dynamically(ue_id, zone),
    # which attaches each new UE to 2-3 SURVIVING O-RUs, preferring those
    # whose coverage_area matches the requested zone.  Dynamically added UEs
    # get NO x_pos/y_pos (Node defaults 0,0), so their true geographic
    # position does not exist in the model — we therefore mark the candidate
    # serving O-RUs instead of inventing UE coordinates.
    rescue_anchors = []
    if kind == 'rescue_ops':
        rus = [nid for nid in infra_surv if ntype[nid] == 'O-RU']
        zone_rus = [nid for nid in rus if coverage.get(nid) == rescue_zone]
        rescue_anchors = zone_rus if zone_rus else rus

    return dict(
        seed=seed, kind=kind, group=group,
        label=label or f"seed {seed}",
        scenario_seed=scenario_seed,
        severance_tick=severance_tick, rescue_tick=rescue_tick,
        rescue_zone=rescue_zone if kind == 'rescue_ops' else None,
        rescue_num_ues=rescue_num_ues if kind == 'rescue_ops' else None,
        pos=pos, ntype=ntype, coverage=coverage,
        multihaul=multihaul, core=core, ue_ids=ue_ids,
        infra_surv=infra_surv, edges=edges,
        comps=comps, frag_of=frag_of, comp_sizes=comp_sizes,
        n_fragments=len(comps), n_cut_links=len(cut_lids),
        n_core_links=n_core_links,
        ue_frags=ue_frags,
        total_flows=total_flows, cross_flows=cross_flows, cross_share=share,
        n_multihaul=len(multihaul),
        n_multihaul_surviving=len(multihaul & infra_set),
        rescue_anchors=rescue_anchors,
        topo_info=topo_info,
        builder_stats=builder_stats, mismatch=mismatch,
    )


# ══════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════

def scenario_title(inst):
    tag = 'A' if inst['kind'] == 'recovery' else 'B'
    name = 'fragmenting recovery' if inst['kind'] == 'recovery' else 'rescue ops'
    return f"Scenario {tag} ({name}) — seed {inst['seed']}"


def annotation_text(inst, verbose=False):
    sizes = '/'.join(str(s) for s in inst['comp_sizes'])
    lines = [
        f"seed {inst['seed']}   scenario: "
        f"{'A / recovery' if inst['kind'] == 'recovery' else 'B / rescue_ops'}",
        f"severance tick: t={inst['severance_tick']}",
    ]
    if inst['rescue_tick'] is not None:
        lines.append(f"rescue arrival: t={inst['rescue_tick']} "
                     f"({inst['rescue_num_ues']} UEs -> {inst['rescue_zone']})")
    lines += [
        f"fragments: {inst['n_fragments']}  (sizes {sizes})",
        f"cut links: {inst['n_cut_links']} partition "
        f"+ {inst['n_core_links']} core",
        f"cross-fragment flows: {inst['cross_flows']}/{inst['total_flows']}"
        f"  ({inst['cross_share']:.1%})",
        f"MultiHaul relay sites: {inst['n_multihaul_surviving']}",
    ]
    if verbose:
        ti = inst['topo_info']
        lines.append(f"topology: {ti['infra']} infra + {ti['ues']} UEs, "
                     f"{ti['links']} links")
        if inst['scenario_seed'] != inst['seed']:
            lines.append(f"event seed: {inst['scenario_seed']}")
    return '\n'.join(lines)


def draw_map(ax, inst, compact=False, show_ue_links=False):
    """Render one scenario map onto `ax`."""
    pos, ntype = inst['pos'], inst['ntype']
    frag_of = inst['frag_of']
    lw_scale = 0.75 if compact else 1.0

    ax.set_facecolor('#FAFAFA')

    # ── links (drawn first, under the nodes) ──
    if show_ue_links:
        for u, v, tag in inst['edges']:
            if tag == 'uu':
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color='#C8D0D8', lw=0.25 * lw_scale, alpha=0.35,
                        zorder=1)
    for u, v, tag in inst['edges']:
        if tag == 'up':
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=SURVIVING_LINK_COLOR, lw=0.9 * lw_scale,
                    alpha=0.85, zorder=2, solid_capstyle='round')
    for u, v, tag in inst['edges']:
        if tag == 'core':
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=CUT_LINK_COLOR, lw=0.7 * lw_scale, alpha=0.45,
                    ls=(0, (1, 2)), zorder=3)
    for u, v, tag in inst['edges']:
        if tag == 'cut':
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=CUT_LINK_COLOR, lw=1.7 * lw_scale, alpha=0.95,
                    ls=(0, (4, 2.2)), zorder=4)

    # ── UEs (light, tinted by the fragment serving them) ──
    ue_x, ue_y, ue_c = [], [], []
    for ue in inst['ue_ids']:
        fr = inst['ue_frags'].get(ue, set())
        ue_x.append(pos[ue][0])
        ue_y.append(pos[ue][1])
        ue_c.append(frag_color(next(iter(fr))) if len(fr) == 1 else '#8A8A8A')
    if ue_x:
        ax.scatter(ue_x, ue_y, s=7 if compact else 11, c=ue_c, marker='.',
                   alpha=0.42, linewidths=0, zorder=3)

    # ── rescue-force attachment anchors (Scenario B) ──
    for nid in inst['rescue_anchors']:
        ax.scatter([pos[nid][0]], [pos[nid][1]],
                   s=270 if compact else 360, facecolors='none',
                   edgecolors='#7B1FA2', linewidths=1.6 * lw_scale,
                   marker='o', zorder=8)

    # ── infrastructure nodes ──
    for nid in inst['infra_surv']:
        st = NODE_STYLE.get(ntype[nid],
                            dict(marker='o', size=40, z=5, label=ntype[nid]))
        c = frag_color(frag_of[nid])
        if nid in inst['multihaul']:
            ax.scatter([pos[nid][0]], [pos[nid][1]],
                       s=MULTIHAUL_STYLE['size'] * (0.62 if compact else 1.0),
                       marker=MULTIHAUL_STYLE['marker'], c=[c],
                       edgecolors='black', linewidths=0.8 * lw_scale,
                       zorder=MULTIHAUL_STYLE['z'])
        else:
            ax.scatter([pos[nid][0]], [pos[nid][1]],
                       s=st['size'] * (0.62 if compact else 1.0),
                       marker=st['marker'], c=[c],
                       edgecolors='#22282E', linewidths=0.45 * lw_scale,
                       zorder=st['z'])

    # ── downed core nodes ──
    for nid in inst['core']:
        st = NODE_STYLE.get(ntype[nid], dict(marker='X', size=55, z=4))
        ax.scatter([pos[nid][0]], [pos[nid][1]],
                   s=st['size'] * (0.62 if compact else 1.0),
                   marker='X', c=[CORE_COLOR], edgecolors='#5A5A5A',
                   linewidths=0.45 * lw_scale, alpha=0.85, zorder=4)

    # ── annotation box ──
    ax.text(0.015, 0.985, annotation_text(inst, verbose=not compact),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=6.4 if compact else 8.6, family='DejaVu Sans',
            linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.42', facecolor='white',
                      edgecolor='#9AA3AB', alpha=0.93, linewidth=0.7))

    ax.set_xlim(-250, GEO_SIZE + 250)
    ax.set_ylim(-250, GEO_SIZE + 250)
    ax.set_aspect('equal')
    ax.set_title(scenario_title(inst), fontsize=9 if compact else 12,
                 fontweight='bold', pad=6)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)
        ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor('#C4CACF')


def legend_handles(max_fragments=4):
    """Shared legend: node types, link states, fragment colours."""
    h = []
    for key in ('O-RU', 'O-DU', 'O-CU-CP', 'O-CU-UP', 'EdgeUPF',
                'Near-RT-RIC', 'Relay'):
        st = NODE_STYLE[key]
        h.append(Line2D([], [], marker=st['marker'], color='none',
                        markerfacecolor='#6E7B87', markeredgecolor='#22282E',
                        markersize=7, label=st['label']))
    h.append(Line2D([], [], marker='*', color='none',
                    markerfacecolor='#6E7B87', markeredgecolor='black',
                    markersize=15, label='MultiHaul relay-capable site'))
    h.append(Line2D([], [], marker='X', color='none',
                    markerfacecolor=CORE_COLOR, markeredgecolor='#5A5A5A',
                    markersize=8, label='Core node (SMO/AMF/UPF) — down'))
    h.append(Line2D([], [], marker='.', color='none',
                    markerfacecolor='#8A8A8A', markeredgecolor='none',
                    markersize=9, label='UE (tinted by serving fragment)'))
    h.append(Line2D([], [], marker='o', color='none', markerfacecolor='none',
                    markeredgecolor='#7B1FA2', markeredgewidth=1.6,
                    markersize=12, label='Rescue-force attachment O-RU (Scen. B)'))
    h.append(Line2D([], [], color=SURVIVING_LINK_COLOR, lw=1.6,
                    label='Surviving infrastructure link'))
    h.append(Line2D([], [], color=CUT_LINK_COLOR, lw=2.0, ls=(0, (4, 2.2)),
                    label='SEVERED link (fragmenting cut)'))
    h.append(Line2D([], [], color=CUT_LINK_COLOR, lw=1.2, ls=(0, (1, 2)),
                    label='Link cut by sever_core'))
    for i in range(max_fragments):
        h.append(Patch(facecolor=frag_color(i), edgecolor='#22282E',
                       label=f'Fragment {i + 1} (by size)'))
    return h


def save_individual(inst, outdir, dpi=300):
    fig, ax = plt.subplots(figsize=(9.2, 9.6))
    draw_map(ax, inst, compact=False, show_ue_links=True)
    fig.legend(handles=legend_handles(max(inst['n_fragments'], 1)),
               loc='lower center', ncol=4, fontsize=7.4, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle('UNITY-6G disaster severance map — infrastructure geometry, '
                 'cut links and post-severance fragments',
                 fontsize=10.5, y=0.985)
    fig.tight_layout(rect=(0, 0.135, 1, 0.965))
    tag = 'scenA' if inst['kind'] == 'recovery' else 'scenB'
    if inst['group'] == 'train':
        name = (f"map_train_{tag}_seed{inst['seed']}"
                f"_ev{inst['scenario_seed']}.png")
    else:
        name = f"map_{tag}_seed{inst['seed']}.png"
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def save_contact_sheet(instances, path, suptitle, ncols, dpi=300):
    """Grid of compact maps with a shared legend and the standing caption.

    Layout is set explicitly (no tight_layout / bbox_inches='tight'): a fixed
    band of BAND_IN inches is reserved at the bottom for the shared legend and
    the caption, and TITLE_IN inches at the top for the suptitle, so nothing
    collides or gets cropped regardless of grid size.
    """
    BAND_IN, TITLE_IN = 2.1, 0.62
    PANEL_W, PANEL_H = 4.35, 4.95

    n = len(instances)
    nrows = int(math.ceil(n / ncols))
    figw = PANEL_W * ncols
    figh = PANEL_H * nrows + BAND_IN + TITLE_IN
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw, figh), squeeze=False)

    for i, inst in enumerate(instances):
        draw_map(axes[i // ncols][i % ncols], inst, compact=True)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    bottom = BAND_IN / figh
    fig.subplots_adjust(left=0.012, right=0.988,
                        top=1.0 - TITLE_IN / figh, bottom=bottom,
                        wspace=0.06, hspace=0.17)

    max_frags = max([i['n_fragments'] for i in instances] + [1])
    fig.legend(handles=legend_handles(max_frags), loc='upper center',
               ncol=6, fontsize=8.2, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, bottom - 0.008))
    fig.suptitle(suptitle, fontsize=14, fontweight='bold',
                 y=1.0 - 0.20 / figh, va='top')

    fig.text(0.5, 0.16 / figh, _wrap(CAPTION, 150), ha='center', va='bottom',
             fontsize=8.0, color='#33383D', linespacing=1.5)

    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return path


def _wrap(text, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
    if cur:
        lines.append(cur)
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════
#  SCENARIO SELECTION
# ══════════════════════════════════════════════════════════════════════

def eval_instances(seeds, kinds):
    out = []
    for kind in kinds:
        for seed in seeds:
            print(f"[MAPS] deriving EVAL  seed={seed} kind={kind} ...")
            out.append(derive_instance(seed, kind, group='eval'))
    return out


def training_instances(n_samples):
    """Sample real training episodes from train_on_comparison's own plan.

    Reproduces episode_spec(): shuffled seed pool from Random(base*9176+3),
    alternating scenario kinds, event_seed = base_seed*100000 + episode index,
    and _randomized_event_ticks() for the per-episode event timing.  These are
    genuine training instances — the point of the figure is that they come
    from the SAME generator as the evaluation, on DISJOINT seeds.
    """
    import random as _random
    plan = load_training_plan()
    B = load_builders()
    if plan is None:
        # Explicit fallback (documented in the report, not silently invented).
        pool = list(range(42, 50)) + list(range(100, 140))
        kinds = ('recovery', 'rescue_ops')
        ticks, base = 800, 42
        rand_ticks = None
        held_out = set(EVAL_SEEDS)
    else:
        pool, kinds = plan['seed_pool'], plan['kinds']
        ticks, base = plan['episode_ticks'], plan['base_seed']
        rand_ticks = plan['randomized_ticks']
        held_out = plan['held_out']

    plan_rng = _random.Random(base * 9176 + 3)
    shuffled = list(pool)
    plan_rng.shuffle(shuffled)

    out = []
    for ep_idx in range(1, n_samples + 1):
        topo_seed = shuffled[(ep_idx - 1) % len(shuffled)]
        kind = kinds[(ep_idx - 1) % len(kinds)]
        event_seed = base * 100000 + ep_idx
        assert topo_seed not in held_out, (
            f"training seed {topo_seed} overlaps the held-out eval seeds")
        rng = _random.Random(event_seed)
        if rand_ticks is not None:
            sev, rescue, _bh = rand_ticks(kind, ticks, rng)
        else:                       # mirrors _randomized_event_ticks exactly
            if kind == 'rescue_ops':
                sev = 0
                lo = max(60, ticks // 4)
                rescue = rng.randint(lo, max(lo + 1, (2 * ticks) // 3))
            else:
                lo = max(30, ticks // 8)
                sev, rescue = rng.randint(lo, max(lo + 1, ticks // 4)), None
        print(f"[MAPS] deriving TRAIN ep={ep_idx} seed={topo_seed} "
              f"kind={kind} sev={sev} event_seed={event_seed} ...")
        out.append(derive_instance(
            topo_seed, kind, severance_tick=sev, rescue_tick=rescue,
            scenario_seed=event_seed, group='train',
            label=f"train ep{ep_idx}"))
    return out


# ══════════════════════════════════════════════════════════════════════
#  CSV + VERIFICATION
# ══════════════════════════════════════════════════════════════════════

CSV_FIELDS = ['seed', 'kind', 'severance_tick', 'rescue_tick', 'n_fragments',
              'n_cut_links', 'cross_fragment_flows', 'total_flows',
              'cross_fragment_share', 'n_multihaul_sites', 'component_sizes',
              'group', 'scenario_seed', 'n_core_links_cut',
              'n_multihaul_surviving', 'n_infra_surviving', 'n_ues']


def write_csv(instances, path):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i in instances:
            w.writerow({
                'seed': i['seed'],
                'kind': i['kind'],
                'severance_tick': i['severance_tick'],
                'rescue_tick': '' if i['rescue_tick'] is None else i['rescue_tick'],
                'n_fragments': i['n_fragments'],
                'n_cut_links': i['n_cut_links'],
                'cross_fragment_flows': i['cross_flows'],
                'total_flows': i['total_flows'],
                'cross_fragment_share': f"{i['cross_share']:.4f}",
                'n_multihaul_sites': i['n_multihaul'],
                'component_sizes': '|'.join(str(s) for s in i['comp_sizes']),
                'group': i['group'],
                'scenario_seed': i['scenario_seed'],
                'n_core_links_cut': i['n_core_links'],
                'n_multihaul_surviving': i['n_multihaul_surviving'],
                'n_infra_surviving': len(i['infra_surv']),
                'n_ues': len(i['ue_ids']),
            })
    return path


def verify_against_expected(instances):
    """Compare Scenario-A evaluation instances with the published values.

    Never adjusts anything — reports agreement or disagreement.
    """
    problems = []
    checked = 0
    print("\n" + "=" * 78)
    print("VERIFICATION — Scenario A, evaluation seeds, vs published values")
    print("=" * 78)
    print(f"{'seed':>5} {'frags':>6} {'exp':>4} {'share':>8} {'exp':>8} "
          f"{'cut':>5} {'exp':>5}   status")
    for i in instances:
        if i['group'] != 'eval' or i['kind'] != 'recovery':
            continue
        exp = EXPECTED_SCEN_A.get(i['seed'])
        if exp is None:
            continue
        checked += 1
        e_fr, e_sh, e_cut = exp
        ok = (i['n_fragments'] == e_fr
              and abs(i['cross_share'] - e_sh) < 0.001
              and i['n_cut_links'] == e_cut)
        if not ok:
            problems.append(
                f"seed {i['seed']}: got fragments={i['n_fragments']}, "
                f"share={i['cross_share']:.1%}, cut_links={i['n_cut_links']}; "
                f"expected {e_fr}, {e_sh:.1%}, {e_cut}")
        print(f"{i['seed']:>5} {i['n_fragments']:>6} {e_fr:>4} "
              f"{i['cross_share']:>7.1%} {e_sh:>7.1%} "
              f"{i['n_cut_links']:>5} {e_cut:>5}   "
              f"{'MATCH' if ok else '*** MISMATCH ***'}")
    for i in instances:
        if i['mismatch']:
            problems.append(f"seed {i['seed']} ({i['kind']}): internal "
                            f"cross-check failed — {i['mismatch']}")
    if checked == 0:
        print("  (no Scenario-A evaluation seeds in this run — nothing to check)")
    if problems:
        print("\n*** DISCREPANCIES — reported, NOT corrected: ***")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nAll checked values match the published figures exactly.")
    print("=" * 78 + "\n")
    return problems


def print_table(instances):
    print("\n" + "=" * 108)
    print("SEVERANCE SCENARIO SUMMARY (all values measured from the builders)")
    print("=" * 108)
    print(f"{'group':>6} {'seed':>5} {'kind':>11} {'sev':>5} {'resc':>5} "
          f"{'frags':>6} {'sizes':>16} {'cut':>5} {'core':>5} "
          f"{'cross/total':>12} {'share':>7} {'MH':>4}")
    print("-" * 108)
    for i in instances:
        print(f"{i['group']:>6} {i['seed']:>5} {i['kind']:>11} "
              f"{i['severance_tick']:>5} "
              f"{('-' if i['rescue_tick'] is None else i['rescue_tick']):>5} "
              f"{i['n_fragments']:>6} "
              f"{'/'.join(str(s) for s in i['comp_sizes']):>16} "
              f"{i['n_cut_links']:>5} {i['n_core_links']:>5} "
              f"{str(i['cross_flows']) + '/' + str(i['total_flows']):>12} "
              f"{i['cross_share']:>6.1%} {i['n_multihaul']:>4}")
    print("=" * 108 + "\n")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def parse_seeds(spec):
    seeds = []
    for part in str(spec).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part[1:]:
            a, b = part.split('-', 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Geographic maps of every UNITY-6G severance scenario.')
    ap.add_argument('--seeds', type=str, default=','.join(map(str, EVAL_SEEDS)),
                    help='evaluation seeds, e.g. "50,51,52" or "50-54" '
                         '(default: 50-54)')
    ap.add_argument('--kind', choices=['recovery', 'rescue_ops', 'both'],
                    default='both', help='scenario kind(s) (default: both)')
    ap.add_argument('--training-samples', type=int, default=8,
                    help='number of training-episode instances to map '
                         '(default: 8; 0 disables the training sheet)')
    ap.add_argument('--outdir', type=str,
                    default=os.path.join(PROJECT_ROOT, 'output', 'maps'),
                    help='directory for the individual PNGs '
                         '(default: output/maps)')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--no-individual', action='store_true',
                    help='skip the per-scenario PNGs (contact sheets only)')
    args = ap.parse_args(argv)

    outdir = os.path.abspath(args.outdir)
    summary_dir = os.path.dirname(outdir) or outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    kinds = (['recovery', 'rescue_ops'] if args.kind == 'both'
             else [args.kind])

    t0 = time.time()
    load_builders()

    instances = eval_instances(seeds, kinds)
    train = training_instances(args.training_samples) \
        if args.training_samples > 0 else []
    all_inst = instances + train

    print_table(all_inst)
    problems = verify_against_expected(all_inst)

    written = []
    if not args.no_individual:
        for inst in all_inst:
            written.append(save_individual(inst, outdir, dpi=args.dpi))
            print(f"[MAPS] wrote {written[-1]}")

    # Evaluation contact sheet: Scenario A row(s) then Scenario B row(s).
    eval_sorted = ([i for i in instances if i['kind'] == 'recovery']
                   + [i for i in instances if i['kind'] == 'rescue_ops'])
    eval_path = os.path.join(summary_dir, 'FINAL_severance_maps_eval.png')
    save_contact_sheet(
        eval_sorted, eval_path,
        'UNITY-6G evaluation severance scenarios — Scenario A (fragmenting '
        'recovery, top) and Scenario B (rescue ops, bottom), held-out seeds '
        + ', '.join(map(str, seeds)),
        ncols=max(1, len(seeds)), dpi=args.dpi)
    written.append(eval_path)
    print(f"[MAPS] wrote {eval_path}")

    train_path = None
    if train:
        train_path = os.path.join(summary_dir,
                                  'FINAL_severance_maps_training.png')
        ncols = 4 if len(train) >= 4 else len(train)
        save_contact_sheet(
            train, train_path,
            'UNITY-6G TRAINING severance scenarios — same generator, '
            'disjoint seeds (42-49 / 100-139), randomised event timing',
            ncols=ncols, dpi=args.dpi)
        written.append(train_path)
        print(f"[MAPS] wrote {train_path}")

    csv_path = os.path.join(summary_dir, 'severance_scenarios.csv')
    write_csv(all_inst, csv_path)
    written.append(csv_path)
    print(f"[MAPS] wrote {csv_path}")

    print(f"\n[MAPS] {len(all_inst)} scenario instances mapped, "
          f"{len(written)} files written in {time.time() - t0:.1f}s")
    if problems:
        print("[MAPS] NOTE: verification reported discrepancies (above). "
              "Values were NOT adjusted.")
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
