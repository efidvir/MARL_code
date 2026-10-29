#!/usr/bin/env python3
"""WP12(a): UE-transit probe for the data-plane path selection.

Question (docs/RERUN_SPEC.md WP1.7 / WP12a): `Simulator._forward_traffic`
chooses every data-plane path with `nx.shortest_path` over
`topology.graph`, a graph that contains the UE nodes and their Uu edges.
Every UE is attached to 2-3 O-RUs, so a path may run O-RU -> UE -> O-RU.
The reachability gate `_can_route_ue_to_ue` only demands that SOME
infrastructure-only path exists; it does not constrain the path taken.
This probe measures how much delivered volume actually travels over a
path with an INTERIOR UE node, per tick and per arm.

It is a read-only diagnostic: nothing under sixg_sim/ or the harness is
edited.  `install()` monkeypatches, from outside the engine, at import time:

  * `networkx.shortest_path` / `networkx.shortest_simple_paths` return a
    `TaggedPath` (a list subclass) while `_forward_traffic` is running;
  * `TaggedPath.__iter__` records which path is being walked
    (`consume_path` walks the delivering path once for energy accounting,
    calling `Node.update_energy(volume, 0)` on every node in path order);
  * `sixg_sim.topology.Node.update_energy` attributes that volume to the
    path being walked -- one record per delivering `consume_path` call;
  * `Simulator._forward_traffic` opens/closes the recording window and
    summarises the tick (the believed graph is still installed inside the
    window, so the summary sees the graph the paths were chosen on);
  * the link-state service's `believed_graph` records whether the arm's
    routing view diverged from physical truth this tick
    (`sixg_sim.link_state_service.LinkStateService` since WP3; the
    harness's `MARLLinkStateDetector` before it, which the probe still
    patches when it finds it -- D12-FIXES ported the probe to the final
    engine, D12F.6);
  * `run_timeline_comparison.run_physics_pass` attaches the per-tick series
    to the pass result (`data['ue_transit_probe']`), so the harness's own
    pickle carries them.  The pool forks, so the parent's patches are
    inherited by every worker.

Self-check: the probe's per-tick UE-to-UE sum must equal the engine's
`last_tick_ue_to_ue_volume` (it is the same delivered volume, summed by a
different route).  The report prints the largest mismatch.

Usage (server, evaluation environment set by diag/run_wp12.sh):
  python3 diag/ue_transit_probe.py run --seeds 82,91 --ticks 1200 \
      --arms marl_static,ospf --out wp12_ue_transit.pkl
  python3 diag/ue_transit_probe.py report output/wp12_ue_transit.pkl
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────
#  Bookkeeping core (importable without the engine; unit-tested)
# ─────────────────────────────────────────────────────────────────────────
class ProbeState:
    """Per-process recording state."""

    def __init__(self):
        self.active = False       # inside Simulator._forward_traffic
        self.cur = None           # TaggedPath currently being iterated
        self.seq = 0              # iteration sequence number
        self.records = {}         # seq -> [path, volume, next_index]
        self.last_believed = None  # believed_graph() result this tick
        self.log = []             # per-tick summaries of the current pass
        self.examples = []        # (tick, path) samples of UE-transit paths
        # post-event samples: (tick, phase, path) with phase 'not-in-view'
        # (a bridge is up but absent from the routing view), 'in-view' or
        # 'no-bridge'; two budgets so that the not-in-view phase, which
        # carries the phase-locked transit, is always represented
        self.examples_post = []
        # interior-UE count of each recorded example, in lockstep with the
        # two example lists, computed with the SAME predicate that recorded
        # the example (`sim.topology.is_ue_node`).  The report must not
        # re-derive it from the node id: `is_ue_node` also matches
        # `Rescue_UE_*` (through its `'_UE_' in node_id` fallback), so a
        # narrower id test would report a path recorded as UE-transit
        # because of an interior rescue UE as having 0 interior UEs, while
        # the per-tick transit volume still counts it.
        self.example_interior_ues = []
        self.examples_post_interior_ues = []
        self.transit_ues = set()  # UE ids seen as interior nodes (pass)
        self.errors = 0

    def reset_tick(self):
        self.records = {}
        self.cur = None

    def reset_pass(self):
        self.reset_tick()
        self.log = []
        self.examples = []
        self.examples_post = []
        self.example_interior_ues = []
        self.examples_post_interior_ues = []
        self.transit_ues = set()
        self.errors = 0
        self.last_believed = None


S = ProbeState()


class TaggedPath(list):
    """A path returned by nx.shortest_path inside _forward_traffic.

    Iterating it announces "this path is being walked now"; consume_path
    walks the delivering path with `for n_id in path_nodes` right before
    calling Node.update_energy on each node."""
    __slots__ = ()

    def __iter__(self):
        S.seq += 1
        S.cur = self
        return list.__iter__(self)


def make_update_energy_hook(orig, state=S):
    """Wrap Node.update_energy: attribute `traffic_load` to the path being
    walked.  A record is opened on the FIRST node of the current path and
    advanced node by node; a call that does not continue the current path
    (e.g. a single-node `[sink]` path, which is a plain list and never
    announces itself) cannot open or extend a record."""
    def update_energy(node, traffic_load, control_bytes):
        if state.active and state.cur is not None:
            rec = state.records.get(state.seq)
            if rec is None:
                if node.id == state.cur[0]:
                    state.records[state.seq] = [list(state.cur),
                                                float(traffic_load), 1]
            else:
                path, _vol, idx = rec
                if idx < len(path) and node.id == path[idx]:
                    rec[2] = idx + 1
        return orig(node, traffic_load, control_bytes)
    return update_energy


def classify_records(records, is_ue):
    """Summarise one tick's records.

    Returns a dict with, per kind ('u2u': UE endpoints both sides, 'gen': UE
    source to a sink, 'infra': everything else), the delivered volume, the
    volume over paths with an interior UE, the path counts, plus the set of
    interior UE ids, example transit paths and the number of records whose
    walk was incomplete (attribution anomaly)."""
    kinds = ('u2u', 'gen', 'infra')
    out = {'vol': dict.fromkeys(kinds, 0.0),
           'transit': dict.fromkeys(kinds, 0.0),
           'n': dict.fromkeys(kinds, 0),
           'n_transit': dict.fromkeys(kinds, 0),
           'transit_ues': set(), 'examples': [], 'incomplete': 0}
    for _seq, (path, vol, idx) in sorted(records.items()):
        if idx != len(path):
            out['incomplete'] += 1
        if is_ue(path[0]) and is_ue(path[-1]):
            kind = 'u2u'
        elif is_ue(path[0]):
            kind = 'gen'
        else:
            kind = 'infra'
        out['vol'][kind] += vol
        out['n'][kind] += 1
        interior_ues = [n for n in path[1:-1] if is_ue(n)]
        if interior_ues:
            out['transit'][kind] += vol
            out['n_transit'][kind] += 1
            out['transit_ues'].update(interior_ues)
            if len(out['examples']) < 3:
                out['examples'].append(list(path))
    return out


EXAMPLE_BUDGET = 24          # pre-event / whole-pass budget (first ticks)
EXAMPLE_BUDGET_NIV = 12      # post-event, bridge up but not in view
EXAMPLE_BUDGET_POST = 12     # post-event, other phases


def name_is_ue(node_id):
    """`Topology.is_ue_node`'s name-based fallback, verbatim
    (`sixg_sim/topology.py`): 'UE_*' or '*_UE_*', so `Rescue_UE_5` counts.

    Used only where the live topology is not available: as the default of
    `record_examples`, and when reading a pickle written before the
    interior-UE counts were recorded.  Every UE of this model is named
    `UE_n` or `Rescue_UE_n`, so on this study's pickles it agrees with
    `is_ue_node` exactly."""
    node_id = str(node_id)
    return node_id.startswith('UE_') or '_UE_' in node_id


def record_examples(state, tick, island, relay_up, relay_not_in_view,
                    examples, is_ue=None):
    """Keep example transit paths.  `examples` are this tick's paths (up to
    3).  The whole-pass budget fills from the first ticks (so for a
    Scenario-A pass it holds pre-event paths); the post-event budgets are
    filled separately per phase so that the not-in-view phase is always
    documented.  Returns the phase label of the tick.

    `is_ue` is the predicate that classified this tick's records (the live
    `sim.topology.is_ue_node`); the interior-UE count of every kept example
    is computed with it HERE and stored alongside, so the report never has
    to guess the predicate back from node ids.  It defaults to
    `name_is_ue`, which is that predicate's own name-based fallback."""
    if is_ue is None:
        is_ue = name_is_ue
    if not island:
        phase = 'pre-event'
    elif relay_not_in_view > 0:
        phase = 'not-in-view'
    elif relay_up > 0:
        phase = 'in-view'
    else:
        phase = 'no-bridge'
    for p in examples:
        n_interior = sum(1 for n in p[1:-1] if is_ue(n))
        if len(state.examples) < EXAMPLE_BUDGET:
            state.examples.append((tick, list(p)))
            state.example_interior_ues.append(n_interior)
        if not island:
            continue
        n_niv = sum(1 for _t, ph, _p in state.examples_post
                    if ph == 'not-in-view')
        n_other = len(state.examples_post) - n_niv
        if phase == 'not-in-view' and n_niv < EXAMPLE_BUDGET_NIV:
            state.examples_post.append((tick, phase, list(p)))
            state.examples_post_interior_ues.append(n_interior)
        elif phase != 'not-in-view' and n_other < EXAMPLE_BUDGET_POST:
            state.examples_post.append((tick, phase, list(p)))
            state.examples_post_interior_ues.append(n_interior)
    return phase


# how an interior-UE histogram was obtained.  It is reported per pass,
# in the JSON and on the report line, because the two predicates are not
# the same predicate and a reader cannot otherwise tell which one ran.
INTERIOR_UE_SOURCES = ('recorded', 'name-fallback', 'count-length-mismatch')


def interior_ues(examples, is_ue=None, counts=None):
    """({number of interior UEs: count}, source) over example paths
    (the path is the last element of each tuple).

    `counts` are the per-example interior-UE counts recorded at collection
    time with the live `is_ue_node`; when they are present and line up with
    `examples` they are used as they stand ('recorded'), so the report
    cannot classify an interior node differently from the probe that
    recorded it.  `is_ue` (default `name_is_ue`) is the fallback for a
    pickle written before those counts existed ('name-fallback'), and it
    must stay as wide as `is_ue_node`, or a path recorded as UE-transit
    because of an interior `Rescue_UE_*` is reported with 0 interior UEs.

    A `counts` list that does NOT line up with `examples` is a defect of
    whatever wrote the pickle, not a pickle vintage: it is reported as
    'count-length-mismatch' and counted as an anomaly by `summarise_pass`,
    and the answer falls back to the name rule.  It used to fall through
    silently, which made a mis-sized count list indistinguishable from a
    correct one in the report and in the JSON.

    The source travels with the number.  The histogram alone does not say
    which predicate produced it, and on this study's committed evidence
    the answer is the fallback on every line (see D12.3 item 1)."""
    if counts is not None:
        if len(counts) == len(examples):
            hist = {}
            for k in counts:
                k = int(k)
                hist[k] = hist.get(k, 0) + 1
            return hist, 'recorded'
        source = 'count-length-mismatch'
    else:
        source = 'name-fallback'
    if is_ue is None:
        is_ue = name_is_ue
    hist = {}
    for ex in examples:
        path = ex[-1]
        k = sum(1 for n in path[1:-1] if is_ue(n))
        hist[k] = hist.get(k, 0) + 1
    return hist, source


def interior_ue_histogram(examples, is_ue=None, counts=None):
    """The histogram alone; `interior_ues` also returns its
    source, which is what the report and the JSON record."""
    return interior_ues(examples, is_ue=is_ue, counts=counts)[0]


# ─────────────────────────────────────────────────────────────────────────
#  Engine-side hooks
# ─────────────────────────────────────────────────────────────────────────
def _summarise_tick(sim):
    from sixg_sim.topology import LinkType
    G = sim.topology.graph          # the graph the paths were chosen on
    is_ue = sim.topology.is_ue_node
    c = classify_records(S.records, is_ue)
    relay_up = relay_not_in_view = 0
    for lk in sim.topology.links.values():
        if lk.link_type is not LinkType.TRANSPORT_RELAY:
            continue
        if not getattr(lk, 'is_up', True):
            continue
        u, v = lk.endpoints
        if is_ue(u) or is_ue(v):
            continue                # IAB access link, not a bridge
        relay_up += 1
        if not G.has_edge(u, v):
            relay_not_in_view += 1
    dual_ue = sum(1 for n in sim.topology.nodes
                  if is_ue(n) and n in G and G.degree(n) >= 2)
    tick = int(sim.current_tick)
    rec = {
        'tick': tick,
        'island': int(bool(sim.island_mode)),
        'believed': int(S.last_believed is not None),
        'relay_up': relay_up,
        'relay_not_in_view': relay_not_in_view,
        'dual_ue': dual_ue,
        'u2u': c['vol']['u2u'], 'u2u_transit': c['transit']['u2u'],
        'n_u2u': c['n']['u2u'], 'n_u2u_transit': c['n_transit']['u2u'],
        'gen': c['vol']['gen'], 'gen_transit': c['transit']['gen'],
        'n_gen': c['n']['gen'], 'n_gen_transit': c['n_transit']['gen'],
        'infra': c['vol']['infra'], 'infra_transit': c['transit']['infra'],
        'n_transit_ues': len(c['transit_ues']),
        'incomplete': c['incomplete'],
        'engine_u2u': float(getattr(sim, 'last_tick_ue_to_ue_volume', 0.0)),
        'flows': len(getattr(sim, 'ue_to_ue_flows', ())),
    }
    S.log.append(rec)
    S.transit_ues.update(c['transit_ues'])
    if c['examples']:
        record_examples(S, tick, rec['island'], relay_up, relay_not_in_view,
                        c['examples'], is_ue)


def _columns(log):
    if not log:
        return {}
    keys = list(log[0].keys())
    return {k: [r.get(k) for r in log] for k in keys}


def install():
    """Apply every monkeypatch.  Idempotent."""
    if getattr(install, 'done', False):
        return
    import networkx as nx
    import sixg_sim.topology as topo_mod
    import sixg_sim.simulation as sim_mod
    import run_timeline_comparison as rtc

    orig_sp = nx.shortest_path
    orig_ssp = nx.shortest_simple_paths

    def shortest_path(*a, **k):
        p = orig_sp(*a, **k)
        if S.active and isinstance(p, list):
            return TaggedPath(p)
        return p

    def shortest_simple_paths(*a, **k):
        gen = orig_ssp(*a, **k)
        if not S.active:
            return gen

        def _tagged():
            for p in gen:
                yield TaggedPath(p)
        return _tagged()

    nx.shortest_path = shortest_path
    nx.shortest_simple_paths = shortest_simple_paths

    topo_mod.Node.update_energy = make_update_energy_hook(
        topo_mod.Node.update_energy)

    orig_fwd = sim_mod.Simulator._forward_traffic

    def _forward_traffic(sim, traffic_arrivals):
        S.reset_tick()
        S.active = True
        try:
            return orig_fwd(sim, traffic_arrivals)
        finally:
            S.active = False
            try:
                _summarise_tick(sim)
            except Exception as exc:      # never break the pass
                S.errors += 1
                if S.errors <= 3:
                    print(f"[UE-TRANSIT-PROBE] summary failed at tick "
                          f"{getattr(sim, 'current_tick', '?')}: {exc!r}",
                          flush=True)
            S.reset_tick()
            S.last_believed = None
    sim_mod.Simulator._forward_traffic = _forward_traffic

    det_cls = getattr(rtc, 'MARLLinkStateDetector', None)
    if det_cls is None:
        # WP3 replaced the harness detector by the link-state service
        from sixg_sim.link_state_service import LinkStateService as det_cls
    orig_bg = det_cls.believed_graph

    def believed_graph(det):
        g = orig_bg(det)
        S.last_believed = g
        return g
    det_cls.believed_graph = believed_graph

    orig_rpp = rtc.run_physics_pass

    def run_physics_pass(mode, topology_seed, checkpoint_path,
                         scenario_name='recovery'):
        S.reset_pass()
        data = orig_rpp(mode, topology_seed, checkpoint_path, scenario_name)
        if isinstance(data, dict):
            data['ue_transit_probe'] = _columns(S.log)
            data['ue_transit_probe_examples'] = list(S.examples)
            data['ue_transit_probe_examples_post'] = list(S.examples_post)
            data['ue_transit_probe_meta'] = {
                'arm': mode, 'scenario': scenario_name,
                'seed': topology_seed, 'ticks': len(S.log),
                'transit_ues': sorted(S.transit_ues),
                # interior-UE counts of the recorded examples, in lockstep
                # with the two example lists and computed with the live
                # `is_ue_node` (see record_examples)
                'example_interior_ues': list(S.example_interior_ues),
                'example_post_interior_ues':
                    list(S.examples_post_interior_ues),
                'summary_errors': S.errors,
            }
        return data
    rtc.run_physics_pass = run_physics_pass
    install.done = True


# ─────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


def summarise_pass(run, arm, scenario, seed, severance_tick):
    p = run.get('ue_transit_probe') or {}
    if not p:
        return None
    _meta = run.get('ue_transit_probe_meta') or {}
    n = len(p['tick'])
    idx_post = [i for i in range(n) if p['island'][i]]
    idx_pre = [i for i in range(n) if not p['island'][i]]

    def tot(key, idx):
        return float(sum(p[key][i] for i in idx))
    u2u_post = tot('u2u', idx_post)
    tr_post = tot('u2u_transit', idx_post)
    gen_post = tot('gen', idx_post)
    gtr_post = tot('gen_transit', idx_post)
    mism = max((abs(p['u2u'][i] - p['engine_u2u'][i]) for i in range(n)),
               default=0.0)
    ticks_tr = [p['tick'][i] for i in idx_post if p['u2u_transit'][i] > 0]
    ticks_niv = [i for i in idx_post if p['relay_not_in_view'][i] > 0]
    ticks_iv = [i for i in idx_post if p['relay_up'][i] > 0
                and p['relay_not_in_view'][i] == 0]
    ticks_nobridge = [i for i in idx_post if p['relay_up'][i] == 0]

    h_first, src_first = interior_ues(
        run.get('ue_transit_probe_examples') or [],
        counts=_meta.get('example_interior_ues'))
    h_post, src_post = interior_ues(
        run.get('ue_transit_probe_examples_post') or [],
        counts=_meta.get('example_post_interior_ues'))

    def phase(idx):
        return {'ticks': len(idx),
                'u2u_mean': _mean(p['u2u'][i] for i in idx),
                'transit_mean': _mean(p['u2u_transit'][i] for i in idx),
                # delivered u2u volume that did NOT transit a UE: what the
                # arm would have delivered under an infrastructure-only path
                # view, before any re-routing of the transit volume
                'nontransit_mean': _mean(p['u2u'][i] - p['u2u_transit'][i]
                                         for i in idx),
                'transit_ticks': sum(1 for i in idx if p['u2u_transit'][i] > 0),
                'high_transit_ticks': sum(1 for i in idx
                                          if p['u2u_transit'][i] >= 100.0)}
    out = {
        'arm': arm, 'scenario': scenario, 'seed': seed, 'ticks': n,
        'severance_tick': severance_tick,
        'post_ticks': len(idx_post),
        'u2u_post_total': u2u_post,
        'u2u_transit_post_total': tr_post,
        'u2u_transit_post_fraction': (tr_post / u2u_post if u2u_post else 0.0),
        'u2u_transit_post_ticks': len(ticks_tr),
        'u2u_transit_first_tick': (ticks_tr[0] if ticks_tr else None),
        'u2u_transit_post_max_per_tick': max(
            (p['u2u_transit'][i] for i in idx_post), default=0.0),
        'u2u_transit_post_mean_per_tick': _mean(
            p['u2u_transit'][i] for i in idx_post),
        'u2u_transit_pre_total': tot('u2u_transit', idx_pre),
        'u2u_pre_total': tot('u2u', idx_pre),
        'gen_post_total': gen_post,
        'gen_transit_post_total': gtr_post,
        'gen_transit_post_fraction': (gtr_post / gen_post if gen_post else 0.0),
        'infra_transit_post_total': tot('infra_transit', idx_post),
        'n_transit_ues': len(_meta.get('transit_ues', [])),
        'dual_ue_mean_post': _mean(p['dual_ue'][i] for i in idx_post),
        'engine_u2u_max_mismatch': mism,
        'incomplete_records': int(sum(p['incomplete'])),
        'summary_errors': _meta.get('summary_errors', 0),
        # phase analysis: bridge physically up but absent from the routing
        # view in use (the detection ticks of a new bridge) versus bridge
        # installed versus no bridge at all
        'phase': {
            'bridge_not_in_view': phase(ticks_niv),
            'bridge_in_view': phase(ticks_iv),
            'no_bridge': phase(ticks_nobridge),
        },
        'examples': [(t, path) for t, path in
                     (run.get('ue_transit_probe_examples') or [])[:6]],
        'examples_post': [(t, ph, path) for t, ph, path in
                          (run.get('ue_transit_probe_examples_post') or [])],
        'interior_ues_per_example': {'first_ticks': h_first,
                                     'post_event': h_post},
        # which predicate produced each histogram above, per pass: the
        # number is meaningless to a reader who cannot tell
        'interior_ues_source': {'first_ticks': src_first,
                                'post_event': src_post},
        # a stored count list that does not line up with its examples is
        # an anomaly, recorded like summary_errors rather than silently
        # fallen through
        'interior_ue_count_anomalies': sum(
            1 for x in (src_first, src_post) if x == 'count-length-mismatch'),
    }
    # transit ticks grouped into runs (start, length) for the phase-lock view
    runs = []
    for t in ticks_tr:
        if runs and t == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([t, 1])
    out['u2u_transit_runs'] = runs[:40]
    out['u2u_transit_run_count'] = len(runs)
    return out


def report(pickle_path, out_dir=None):
    import pickle
    with open(pickle_path, 'rb') as f:
        P = pickle.load(f)
    prov = P.get('provenance') or {}
    sev = int(prov.get('severance_tick', 200))
    rows = []
    for scen_key, scenario, ev in (('recovery_multi', 'recovery', sev),
                                   ('rescue_multi', 'rescue_ops', 0)):
        for arm, per_seed in sorted((P.get(scen_key) or {}).items()):
            for seed, run in sorted(per_seed.items()):
                if not run:
                    continue
                s = summarise_pass(run, arm, scenario, seed, ev)
                if s is not None:
                    rows.append(s)
    lines = []
    lines.append("WP12(a) UE-transit probe report: %s" % pickle_path)
    lines.append("tree %s  digest %s  hashseed %s  ticks %s  severance %s"
                 % (prov.get('tree_head'), str(prov.get('tree_sha256'))[:12],
                    prov.get('pythonhashseed'), prov.get('total_ticks'), sev))
    lines.append("")
    hdr = ("%-11s %-10s %4s | %9s %9s %6s %5s %5s | %9s %6s | %8s %5s %5s | %s"
           % ('arm', 'scenario', 'seed', 'u2u_post', 'transit', 'frac',
              'ticks', 'runs', 'gen_post', 'gfrac', 'mismatch', 'inc',
              'nUE', 'pre-event transit'))
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for s in rows:
        lines.append(
            "%-11s %-10s %4d | %9.0f %9.1f %6.3f %5d %5d | %9.0f %6.3f | "
            "%8.2g %5d %5d | %.1f of %.0f"
            % (s['arm'], s['scenario'], s['seed'], s['u2u_post_total'],
               s['u2u_transit_post_total'], s['u2u_transit_post_fraction'],
               s['u2u_transit_post_ticks'], s['u2u_transit_run_count'],
               s['gen_post_total'], s['gen_transit_post_fraction'],
               s['engine_u2u_max_mismatch'], s['incomplete_records'],
               s['n_transit_ues'], s['u2u_transit_pre_total'],
               s['u2u_pre_total']))
    lines.append("")
    lines.append("Phase view (post-event ticks): bridge physically up but not "
                 "in the routing view / bridge in view / no bridge; per phase: "
                 "mean delivered u2u per tick, of which over a UE-transit path "
                 "(transit) and over infrastructure-only paths (non-transit); "
                 "ticks with transit >= 100 units")
    for s in rows:
        ph = s['phase']
        a, b, c = (ph['bridge_not_in_view'], ph['bridge_in_view'],
                   ph['no_bridge'])
        lines.append("  %-11s %-10s %4d: not-in-view %4d ticks u2u %7.1f "
                     "transit %6.1f non-transit %7.1f (>=100: %3d) | in-view "
                     "%4d ticks u2u %7.1f transit %6.1f non-transit %7.1f "
                     "(>=100: %3d) | no-bridge %4d ticks u2u %7.1f transit "
                     "%6.1f non-transit %7.1f (>=100: %3d) | dual-UE mean %.0f"
                     % (s['arm'], s['scenario'], s['seed'],
                        a['ticks'], a['u2u_mean'], a['transit_mean'],
                        a['nontransit_mean'], a['high_transit_ticks'],
                        b['ticks'], b['u2u_mean'], b['transit_mean'],
                        b['nontransit_mean'], b['high_transit_ticks'],
                        c['ticks'], c['u2u_mean'], c['transit_mean'],
                        c['nontransit_mean'], c['high_transit_ticks'],
                        s['dual_ue_mean_post']))
    lines.append("")
    lines.append("Phase-lock check (the mapping's '+70 to +130 units/tick "
                 "while a bridge is up but not yet in the routing view'): "
                 "u2u gap = not-in-view minus in-view, split into the "
                 "UE-transit and the infrastructure-only share")
    for s in rows:
        ph = s['phase']
        a, b = ph['bridge_not_in_view'], ph['bridge_in_view']
        if a['ticks'] and b['ticks']:
            lines.append("  %-11s %-10s %4d: gap %+7.1f = transit %+7.1f + "
                         "non-transit %+7.1f  (%d not-in-view ticks)"
                         % (s['arm'], s['scenario'], s['seed'],
                            a['u2u_mean'] - b['u2u_mean'],
                            a['transit_mean'] - b['transit_mean'],
                            a['nontransit_mean'] - b['nontransit_mean'],
                            a['ticks']))
        else:
            lines.append("  %-11s %-10s %4d: not applicable (no bridge "
                         "phases: %d not-in-view / %d in-view ticks)"
                         % (s['arm'], s['scenario'], s['seed'], a['ticks'],
                            b['ticks']))
    lines.append("")
    lines.append("Transit runs (post-event tick+length), first 12 per pass:")
    for s in rows:
        lines.append("  %-11s %-10s %4d: %s"
                     % (s['arm'], s['scenario'], s['seed'],
                        ' '.join('%d+%d' % (a, b)
                                 for a, b in s['u2u_transit_runs'][:12])
                        or '(none)'))
    lines.append("")
    lines.append("Example UE-transit paths from the FIRST ticks of each pass "
                 "(tick: path; pre-event for Scenario A):")
    for s in rows:
        for t, path in s['examples'][:3]:
            lines.append("  %-11s %-10s %4d  t=%4d: %s"
                         % (s['arm'], s['scenario'], s['seed'], t,
                            ' > '.join(path)))
    lines.append("")
    lines.append("Example UE-transit paths from POST-EVENT ticks (tick phase: "
                 "path; up to 2 per phase per pass; phase = bridge up but "
                 "not in the routing view / in view / no bridge):")
    for s in rows:
        shown = {}
        for t, ph, path in s['examples_post']:
            if shown.get(ph, 0) >= 2:
                continue
            shown[ph] = shown.get(ph, 0) + 1
            lines.append("  %-11s %-10s %4d  t=%4d %-11s: %s"
                         % (s['arm'], s['scenario'], s['seed'], t, ph,
                            ' > '.join(path)))
    lines.append("")
    lines.append("Interior UEs per recorded example path {count: paths} "
                 "(first-ticks budget | post-event budget), each with the "
                 "source of the count:")
    lines.append("  recorded = the per-example count stored by the probe "
                 "with the predicate that classified that tick's records "
                 "(is_ue_node); name-fallback = the pickle carries no such "
                 "counts (written before they existed) and the report "
                 "recounted the interior nodes with name_is_ue, is_ue_node's "
                 "own name rule; count-length-mismatch = counts were stored "
                 "but do not line up with the examples -- an anomaly, "
                 "counted in interior_ue_count_anomalies, answered by the "
                 "name rule.")
    for s in rows:
        h = s['interior_ues_per_example']
        src = s['interior_ues_source']
        lines.append("  %-11s %-10s %4d: %s [%s] | %s [%s]"
                     % (s['arm'], s['scenario'], s['seed'],
                        dict(sorted(h['first_ticks'].items())),
                        src['first_ticks'],
                        dict(sorted(h['post_event'].items())),
                        src['post_event']))
    n_anom = sum(s['interior_ue_count_anomalies'] for s in rows)
    srcs = sorted({x for s in rows for x in s['interior_ues_source'].values()})
    lines.append("  sources over %d passes: %s; count-length-mismatch "
                 "anomalies: %d" % (len(rows), ', '.join(srcs), n_anom))
    text = '\n'.join(lines)
    print(text)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(pickle_path))[0]
        with open(os.path.join(out_dir, base + '_report.json'), 'w') as f:
            json.dump({'pickle': pickle_path, 'provenance': {
                k: prov.get(k) for k in ('tree_head', 'tree_sha256',
                                         'pythonhashseed', 'total_ticks',
                                         'severance_tick', 'arms', 'seeds')},
                'passes': rows}, f, indent=1, default=str)
        with open(os.path.join(out_dir, base + '_report.txt'), 'w') as f:
            f.write(text + '\n')
        print("\nwritten: %s/%s_report.{json,txt}" % (out_dir, base))
    return rows


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────
def cmd_run(args):
    import multiprocessing
    os.chdir(ROOT)
    if multiprocessing.get_start_method(allow_none=True) not in (None, 'fork'):
        raise SystemExit("the probe relies on fork so that the workers "
                         "inherit the monkeypatches")
    multiprocessing.set_start_method('fork', force=True)
    install()
    import run_timeline_comparison as rtc
    print("[UE-TRANSIT-PROBE] patches installed; arms=%s seeds=%s ticks=%s"
          % (args.arms, args.seeds, args.ticks), flush=True)
    overrides = {'total_ticks': int(args.ticks)} if args.ticks else {}
    rtc.main(overrides=overrides, seed_list=args.seeds, out_file=args.out,
             force_overwrite=True, arms=args.arms, no_figures=True)
    pk = os.path.join(rtc.OUTPUT_DIR, args.out)
    report(pk, out_dir=os.path.join(rtc.OUTPUT_DIR, 'wp12'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the harness with the probe installed')
    r.add_argument('--seeds', default='82,91')
    r.add_argument('--ticks', type=int, default=1200)
    r.add_argument('--arms', default='marl_static,ospf')
    r.add_argument('--out', default='wp12_ue_transit.pkl')
    p = sub.add_parser('report', help='summarise a probe pickle')
    p.add_argument('pickle')
    p.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        cmd_run(args)
    else:
        report(args.pickle, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
