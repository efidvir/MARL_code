# -*- coding: utf-8 -*-
"""D12.1 -- no data-plane transit through user devices, for every arm
(docs/DECISIONS.md D12.1, D12F.3; docs/TRACKB_HANDOFF.md 7.1).

A UE may be a flow's ENDPOINT, never an interior hop.  Every arm forwards
through the one shared `Simulator._forward_traffic` (tick_step phase 7): the
five routing/SDN baselines on the true graph, the MARL-path arms on the
link-state service's installed view, which tick_step swaps in for exactly
that call.  So the property is checked here in both forwarding modes, on
hand-built topologies in which the UE shortcut is SHORTER than the
infrastructure path (so an unconstrained shortest path takes it):

  * UE-to-UE flows and the general traffic to an island sink never put a
    UE inside a path, and the stitching UE's links carry nothing of the
    flow;
  * a flow whose endpoints lie in two infrastructure fragments that only a
    dual-anchored UE joins in the forwarding view is NOT delivered, and is
    delivered once an infrastructure or relay link joins them;
  * a structural guard: every shortest-path call of the forwarding code and
    of the harness's per-tick path metrics takes a path view; the
    exceptions are named, with their reasons.

Fast: no simulator pass, no checkpoint.
"""
import contextlib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,  # noqa: E402
                               InterfaceType, TrafficClass)

A, B, C, D, E, F = ('O-RU_1', 'O-DU_2', 'O-DU_3', 'O-DU_4', 'O-RU_5',
                    'EdgeUPF_6')
S, T, M = 'UE_1', 'UE_2', 'UE_9'      # source, target, the stitching UE


def _link(lid, u, v, cap=500.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=True)


def _uu(lid, ue, anchor):
    return _link(lid, ue, anchor, 100.0, LinkType.WIRELESS, InterfaceType.Uu)


def _sim(infra_links, extra_relay=None):
    """Infrastructure A..F (F the island sink, an EdgeUPF on E), UE S on A,
    UE T on E, and UE M on BOTH A and E: S-A-M-E-T is four hops while any
    infrastructure path from A to E is longer, so an unconstrained search
    transits M."""
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    t = Topology()
    for nid, nt in ((A, NodeType.O_RU), (B, NodeType.O_DU), (C, NodeType.O_DU),
                    (D, NodeType.O_DU), (E, NodeType.O_RU), (F, NodeType.EDGEUPF),
                    (S, NodeType.UE), (T, NodeType.UE), (M, NodeType.UE)):
        t.add_node(Node(id=nid, node_type=nt))
    for u, v in infra_links:
        t.add_link(_link('Mesh_%s_%s' % (u, v), u, v))
    t.add_link(_link('Mesh_E_F', E, F))
    t.add_link(_uu('Uu_S_A', S, A))
    t.add_link(_uu('Uu_T_E', T, E))
    t.add_link(_uu('Uu_M_A', M, A))
    t.add_link(_uu('Uu_M_E', M, E))
    scen = Scenario(name='d12', duration_ticks=50, events=[], traffic_profiles={})
    sim = Simulator(t, scen, SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 1
    if extra_relay is not None:
        u, v = extra_relay
        lid = sim.transport_relay_model.create_link(u, v, 50.0, 15.0)
        sim.topology.links[lid] = _link(lid, u, v, 50.0, LinkType.TRANSPORT_RELAY)
        sim.topology.sync_pair(u, v)
    sim.island_mode = True
    sim.marl_ue_routing_enabled = True
    sim.topology.invalidate_infrastructure_cache()
    return sim


@contextlib.contextmanager
def _recording_paths():
    """Every path networkx returns while the block runs (the forwarding
    code reads nx.shortest_path / shortest_simple_paths off the module at
    call time, as diag/ue_transit_probe.py relies on)."""
    import networkx as nx
    rec = []
    osp, ossp = nx.shortest_path, nx.shortest_simple_paths

    def sp(*a, **k):
        p = osp(*a, **k)
        if isinstance(p, list):
            rec.append(list(p))
        return p

    def ssp(*a, **k):
        for p in ossp(*a, **k):
            rec.append(list(p))
            yield p
    nx.shortest_path, nx.shortest_simple_paths = sp, ssp
    try:
        yield rec
    finally:
        nx.shortest_path, nx.shortest_simple_paths = osp, ossp


def _forward(sim, flows, general=(), view=None):
    """Phase 7 of tick_step in isolation: forward `flows` (and 5 units of
    best-effort general traffic from each node in `general`) on the true
    graph, or on `view` swapped in as tick_step installs the link-state
    service's view.  Returns (UE-to-UE volume, recorded paths)."""
    sim.ue_to_ue_flows = list(flows)
    sim._generate_traffic()                 # resets queues and utilisation
    arr = {n: {TrafficClass.BEST_EFFORT: 5.0} for n in general}
    true_graph = sim.topology.graph
    with _recording_paths() as rec:
        try:
            if view is not None:
                sim.topology.graph = view
            sim._forward_traffic(arr)
        finally:
            sim.topology.graph = true_graph
    return float(sim.last_tick_ue_to_ue_volume), rec


def _interior_ues(path, sim):
    return [n for n in path[1:-1] if sim.topology.is_ue_node(n)]


def _carried(sim, lid):
    return float(sim._link_carried_ue.get(lid, 0.0))


def test_no_path_of_any_forwarding_mode_has_an_interior_ue():
    """UE-to-UE flow S->T and S's general traffic to the island sink F, on
    the true graph (the routing and SDN arms) and on an installed view equal
    to it (the MARL-path arms): the shortest path through the stitching UE
    M is four hops, the infrastructure path six; the flow must take the
    latter, and M's links must carry none of it."""
    chain = [(A, B), (B, C), (C, D), (D, E)]
    checked = 0
    for mode in ('true graph', 'installed view'):
        sim = _sim(chain)
        view = sim.topology.graph.copy() if mode == 'installed view' else None
        vol, rec = _forward(sim, [(S, T)], view=view)
        assert vol > 0.0, (mode, 'an infrastructure path exists: delivered')
        assert rec, (mode, 'no path was searched')
        for p in rec:
            assert not _interior_ues(p, sim), (mode, 'UE inside a path', p)
            checked += 1
        u2u = [p for p in rec if p[0] == S and p[-1] == T]
        assert u2u and u2u[0] == [S, A, B, C, D, E, T], (mode, u2u)
        assert _carried(sim, 'Uu_M_A') == 0.0 and _carried(sim, 'Uu_M_E') == 0.0, (
            mode, 'the stitching UE carried the flow')
        assert _carried(sim, 'Mesh_%s_%s' % (C, D)) > 0.0, mode
        # the general traffic of S to the island sink F
        sim2 = _sim(chain)
        view2 = sim2.topology.graph.copy() if mode == 'installed view' else None
        _vol, rec2 = _forward(sim2, [], general=(S,), view=view2)
        sink = [p for p in rec2 if p[0] == S and p[-1] == F]
        assert sink and sink[0] == [S, A, B, C, D, E, F], (mode, rec2)
        for p in rec2:
            assert not _interior_ues(p, sim2), (mode, 'UE inside a sink path', p)
            checked += 1
        assert _carried(sim2, 'Uu_M_A') == 0.0 and _carried(sim2, 'Uu_M_E') == 0.0
        assert _carried(sim2, 'Mesh_E_F') > 0.0, (mode, 'S reached its sink')
    assert checked >= 4, checked


def test_a_flow_across_fragments_joined_only_by_a_ue_is_not_delivered():
    """Two infrastructure fragments {A, B} and {D, E} (plus C hanging off
    B).  The flow S (on A) -> T (on E):

      * no infrastructure or relay link joins them: the reachability gate
        refuses it (0), as before D12.1;
      * a relay B=D joins them physically, so the gate admits the flow, but
        the arm's INSTALLED view does not hold the relay yet -- the MARL
        bridge-detection window of D12.1: only the dual-anchored UE M joins
        the two fragments there.  Before D12.1 the flow went S-A-M-E-T and
        was delivered through a user device; now it is not delivered;
      * once the view holds the relay, it is delivered over it."""
    frags = [(A, B), (B, C), (D, E)]
    sim = _sim(frags)
    vol, _rec = _forward(sim, [(S, T)])
    assert vol == 0.0, 'no infrastructure path: the gate refuses the flow'

    sim = _sim(frags, extra_relay=(B, D))
    assert sim._can_route_ue_to_ue(S, T), 'the relay joins the fragments'
    view = sim.topology.graph.copy()
    view.remove_edge(B, D)                      # not installed yet
    vol, rec = _forward(sim, [(S, T)], view=view)
    assert vol == 0.0, ('delivered through the dual-anchored UE while the '
                        'relay is outside the installed view', rec)
    assert not any(_interior_ues(p, sim) for p in rec), rec
    assert _carried(sim, 'Uu_M_A') == 0.0 and _carried(sim, 'Uu_M_E') == 0.0
    assert sim._diag_flow_noroute == 1, sim._diag_flow_noroute

    vol, rec = _forward(sim, [(S, T)], view=sim.topology.graph.copy())
    assert vol > 0.0, 'the relay is in the view: delivered over it'
    u2u = [p for p in rec if p[0] == S and p[-1] == T]
    assert u2u and u2u[0] == [S, A, B, D, E, T], u2u
    vol, rec = _forward(sim, [(S, T)])          # the true graph
    assert vol > 0.0 and not any(_interior_ues(p, sim) for p in rec), rec


def test_the_path_view_admits_only_the_flows_own_endpoints():
    """Topology.path_view: the endpoints and every infrastructure node, no
    other UE; the infrastructure set is the graph's non-UE nodes."""
    import networkx as nx
    sim = _sim([(A, B), (B, C), (C, D), (D, E)])
    topo = sim.topology
    infra = topo.infra_node_set()
    assert infra == frozenset({A, B, C, D, E, F}), sorted(infra)
    v = topo.path_view((S, T), infra=infra)
    assert set(v.nodes) == {A, B, C, D, E, F, S, T}
    assert M not in v and not v.has_edge(M, A)
    assert nx.shortest_path(topo.graph, S, T) == [S, A, M, E, T]   # the old one
    assert nx.shortest_path(v, S, T) == [S, A, B, C, D, E, T]
    v1 = topo.path_view((S,), infra=infra)
    assert T not in v1 and S in v1
    try:
        nx.shortest_path(v1, S, T)
    except nx.NodeNotFound:
        pass
    else:
        raise AssertionError('a UE outside the endpoints was reachable')
    # infra omitted: computed from the graph in use
    assert set(topo.path_view((S, T)).nodes) == set(v.nodes)


_HASH_PROBE = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
import networkx as nx
from sixg_sim.topology import Topology, Node, Link, NodeType, LinkType, InterfaceType
t = Topology()
infra = ['O-RU_%d' % i for i in range(12)]
for n in infra:
    t.add_node(Node(id=n, node_type=NodeType.O_RU))
ues = ['UE_%d' % i for i in range(30)]
for u in ues:
    t.add_node(Node(id=u, node_type=NodeType.UE))
k = 0
def L(u, v, lt=LinkType.MULTIHAUL_MESH, it=InterfaceType.BACKHAUL):
    global k
    k += 1
    t.add_link(Link(id='L%d' % k, endpoints=(u, v), capacity=1.0, latency=1,
                    link_type=lt, interface_type=it, is_up=True))
# a ladder: many equal-length infrastructure paths
for i in range(0, 10, 2):
    L(infra[i], infra[i + 1]); L(infra[i], infra[i + 2]); L(infra[i + 1], infra[i + 3])
for j, u in enumerate(ues):
    L(u, infra[j % 12], LinkType.WIRELESS, InterfaceType.Uu)
    L(u, infra[(j * 7 + 3) % 12], LinkType.WIRELESS, InterfaceType.Uu)
inf = t.infra_node_set()
for s in ues[:15]:
    for d in ues[15:]:
        try:
            p = nx.shortest_path(t.path_view((s, d), infra=inf), s, d)
        except nx.NetworkXNoPath:
            p = None
        print(s, d, p)
'''


def test_path_selection_does_not_depend_on_the_hash_seed():
    """Ties between equal-length infrastructure paths are broken in the
    graph's own adjacency order, never in salted set order: the same 225
    flows choose the same paths under PYTHONHASHSEED 0, 1 and 2."""
    outs = []
    for seed in ('0', '1', '2'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, '-c', _HASH_PROBE, ROOT],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout)
    lines = outs[0].splitlines()
    assert len(lines) == 225, len(lines)
    assert sum(1 for ln in lines if not ln.endswith('None')) >= 100, lines[:5]
    assert outs[0] == outs[1] == outs[2]


# ── structural guard ──────────────────────────────────────────────────────
_CALL = re.compile(r'nx\.(shortest_path|shortest_simple_paths|'
                   r'shortest_path_length|has_path)\(\s*([^,\n]+)')

# The calls that do NOT take a path view, each with its reason.  Anything
# else that searches `topology.graph` directly fails the guard.
_ALLOWED = {
    ('sixg_sim/simulation.py', 'max_graph'):
        'the best-case "optimal routing" sketch for the visualiser '
        '(_compute_optimal_scenario_routing), no traffic',
    ('sixg_sim/simulation.py', 'g'):
        "_compute_island_upper_bound: sixg_sim/main.py's log line only",
}
# AODV's route-establishment check and its RREP billing were named
# exceptions until D12F.8; they take the path view since.


def test_every_data_plane_path_search_takes_a_path_view():
    found = []
    for rel in ('sixg_sim/simulation.py', 'run_timeline_comparison.py'):
        src = open(os.path.join(ROOT, rel), encoding='utf-8').read()
        for m in _CALL.finditer(src):
            arg = m.group(2).strip()
            if 'path_view(' in arg or arg in ('_pv', '_sv'):
                found.append((rel, 'view'))
                continue
            assert (rel, arg) in _ALLOWED, (
                '%s: a path search on %r that is not a path view and not a '
                'named exception' % (rel, arg))
            found.append((rel, arg))
    views = [f for f in found if f[1] == 'view']
    # simulation.py: UE-to-UE path, its alternates, the sink path;
    # the harness: the SDN FlowMod path length, the hop metric, the latency
    # metric, AODV's route establishment and its RREP billing (D12F.8)
    assert len(views) == 8, found
    assert len(found) == len(views) + len(_ALLOWED), found
