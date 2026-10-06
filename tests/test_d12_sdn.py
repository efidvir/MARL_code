# -*- coding: utf-8 -*-
"""D12.2 -- the SDN baseline's discovery (docs/DECISIONS.md D12.2, D12F.4;
docs/TRACKB_HANDOFF.md 7.2 (a), (b), (c), (c2) and its two riders, (d)).

The real `SDNController` on small fake topologies (no simulator):

  * discovery is a cumulative BFS that terminates on a bipartite fragment
    after ecc(leader) + 1 rounds, with a round timer per fragment;
  * fragments are infrastructure components, UEs leaves of every fragment
    holding one of their anchors, never walked;
  * the operational-phase re-queue programs a flow only when both
    endpoints are anchored in ONE discovered fragment; a relay that joins
    two components lets their cross flows in after the recompute; a switch
    without a controller programs nothing (d), until it joins a led
    fragment and is discovered;
  * Scenario B: the pre-converged controller seeds every fragment's
    discovered set, elected or not (c2, rider 1), and a forced recompute
    REMAPS that state onto the new decomposition (rider 2), so the
    rescue-arrival flows are programmed -- also when the recompute
    renumbers the fragments.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _NT:
    def __init__(self, v):
        self.value = v


class _Node:
    def __init__(self, t):
        self.node_type = _NT(t)
        self.is_survivor = True


class _Link:
    def __init__(self, a, b):
        self.endpoints = (a, b)
        self.is_up = True


class _Topo:
    def __init__(self):
        self.nodes = {}
        self.links = {}

    def is_ue_node(self, nid):
        n = self.nodes.get(nid)
        return bool(n) and n.node_type.value == 'UE'


class _Ev:
    def __init__(self, tick, et):
        self.tick = tick
        self.event_type = et


class _Sim:
    def __init__(self, topo, flows, severance_tick):
        self.topology = topo
        self.ue_to_ue_flows = list(flows)
        self.scenario = type('S', (), {'events': [_Ev(severance_tick, 'sever_core')]})()
        self.current_tick = 0


def _topo(switches, links, ue_anchors):
    t = _Topo()
    for s in switches:
        t.nodes[s] = _Node('O-RU')
    for u in ue_anchors:
        t.nodes[u] = _Node('UE')
    for a, b in links:
        t.links['l_%s_%s' % (a, b)] = _Link(a, b)
    for u, anchors in ue_anchors.items():
        for a in anchors:
            t.links['uu_%s_%s' % (u, a)] = _Link(u, a)
    return t


def _ctrl(sim):
    import run_timeline_comparison as R
    return R.SDNController(sim)


def _run(ctrl, ticks, hist=None):
    for tick in ticks:
        ctrl.step(tick)
        if hist is not None:
            hist.append((tick, ctrl.phase, len(ctrl.programmed_flows),
                         {f: set(fr) for f, fr in ctrl.discovery_frontier.items()}))


# ── (a) cumulative BFS, per-fragment timer ────────────────────────────────
TREE = ['P0', 'P1', 'P2', 'P3', 'P4', 'Q1', 'Q2']
TREE_LINKS = [('P0', 'P1'), ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4'),
              ('P2', 'Q1'), ('Q1', 'Q2')]
TREE_UES = {'ua': ['P0'], 'ub': ['P4'], 'uc': ['Q2']}


def test_discovery_terminates_on_a_bipartite_tree_after_ecc_plus_one_rounds():
    """A tree is bipartite: the pre-D12.2 fixed-point test never ended on
    it.  Leader P2 (up-link degree 3), ecc 2: layers {P1, P3, Q1} and
    {P0, P4, Q2}, completion in the third round, and the three flows
    programmed that tick: election done + ecc + 1.  The walk never enters
    a UE."""
    import run_timeline_comparison as R
    sim = _Sim(_topo(TREE, TREE_LINKS, TREE_UES),
               [('ua', 'ub'), ('ub', 'uc'), ('ua', 'uc')], R.SEVERANCE_TICK)
    c = _ctrl(sim)
    hist = []
    _run(c, range(R.SEVERANCE_TICK, R.SEVERANCE_TICK + 120), hist)
    assert c.fragment_leaders == {0: 'P2'}, c.fragment_leaders
    e = c.election_done_tick[0]
    layers = [h[3].get(0) for h in hist if e <= h[0] <= e + 2]
    assert layers == [{'P2'}, {'P1', 'P3', 'Q1'}, {'P0', 'P4', 'Q2'}], layers
    first = next(t for t, _p, n, _f in hist if n > 0)
    assert first == e + 2 + 1, (first, e)
    assert c.programmed_flows == {('ua', 'ub'), ('ub', 'uc'), ('ua', 'uc')}
    assert c.discovery_done[0] is True and c.discovered[0] == set(TREE)
    assert c.phase == 'operational', c.phase
    seen = set().union(*(f for h in hist for f in h[3].values()))
    assert seen <= set(TREE), seen - set(TREE)


def test_two_fragments_discover_in_parallel_on_their_own_timers():
    """Two components of different eccentricity: their elections run in
    parallel and discovery starts when the last one ends (the model's
    phase gate, unchanged); from there each completes after ITS OWN
    ecc + 1 rounds and neither waits for the other (the pre-D12.2 shared
    round timer let the first fragment's rounds starve the second)."""
    import run_timeline_comparison as R
    sw = TREE + ['T1', 'T2', 'T3']
    links = TREE_LINKS + [('T1', 'T2'), ('T2', 'T3'), ('T3', 'T1')]
    ues = dict(TREE_UES, ut1=['T1'], ut2=['T3'])
    sim = _Sim(_topo(sw, links, ues), [('ua', 'ub'), ('ut1', 'ut2')],
               R.SEVERANCE_TICK)
    c = _ctrl(sim)
    hist = []
    _run(c, range(R.SEVERANCE_TICK, R.SEVERANCE_TICK + 120), hist)
    assert sorted(c.fragment_switches.values(), key=len) == [
        {'T1', 'T2', 'T3'}, set(TREE)]
    e = {f: c.election_done_tick[f] for f in c.fragment_leaders}
    tri = [f for f in c.fragment_switches if 'T1' in c.fragment_switches[f]][0]
    tree = 1 - tri
    # the 3-switch triangle elects 1 tick before the 7-switch tree
    # (round(2 + 0.1 N)); discovery starts at the later tick for both
    assert e[tree] == e[tri] + 1, e
    start = max(e.values())
    prog = {}
    for t, _p, n, _f in hist:
        prog.setdefault(n, t)
    assert prog[1] == start + 1 + 1, (prog, e)     # the triangle, ecc 1
    assert prog[2] == start + 2 + 1, (prog, e)     # the tree, ecc 2
    assert c.programmed_flows == {('ua', 'ub'), ('ut1', 'ut2')}


# ── (b) infrastructure components, UEs as leaves ──────────────────────────
def test_fragments_are_infrastructure_components_and_ues_their_leaves():
    """{A, B, C} and {D, E, F} joined ONLY by the dual-anchored UE m: two
    fragments, m a leaf of both (in fragment_members, never in a walk set),
    the anchors as the up-link peers that are switches."""
    import run_timeline_comparison as R
    t = _topo(['A', 'B', 'C', 'D', 'E', 'F'],
              [('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'E'), ('E', 'F'), ('F', 'D')],
              {'u1': ['B'], 'm': ['A', 'F'], 'u3': ['E']})
    sim = _Sim(t, [('u1', 'm'), ('m', 'u3'), ('u1', 'u3')], R.SEVERANCE_TICK)
    c = _ctrl(sim)
    c._compute_fragments()
    assert c.fragment_switches == {0: {'A', 'B', 'C'}, 1: {'D', 'E', 'F'}}
    assert c.fragment_members == {0: {'A', 'B', 'C', 'u1', 'm'},
                                  1: {'D', 'E', 'F', 'm', 'u3'}}
    assert c.anchors == {'u1': ('B',), 'm': ('A', 'F'), 'u3': ('E',)}
    assert set(c.fragment_leaders) == {0, 1}
    _run(c, range(R.SEVERANCE_TICK, R.SEVERANCE_TICK + 120))
    assert c.programmed_flows == {('u1', 'm'), ('m', 'u3')}
    for f in (0, 1):
        assert c.discovered[f] == c.fragment_switches[f], (f, c.discovered)


# ── (c) and (d) Scenario A: the operational re-queue ──────────────────────
def _scenario_a():
    """Components {A, B, C} and {D, E, F}, and the isolated switch I (below
    MIN_FRAGMENT_FOR_ELECTION: no controller).  Flows: one inside each
    component, one across them, one served only by I."""
    import run_timeline_comparison as R
    t = _topo(['A', 'B', 'C', 'D', 'E', 'F', 'I'],
              [('A', 'B'), ('B', 'C'), ('D', 'E'), ('E', 'F')],
              {'u1': ['A'], 'u2': ['C'], 'u3': ['D'], 'u4': ['F'],
               'i1': ['I'], 'i2': ['I']})
    flows = [('u1', 'u2'), ('u3', 'u4'), ('u1', 'u4'), ('i1', 'i2')]
    return _Sim(t, flows, R.SEVERANCE_TICK), R.SEVERANCE_TICK


def test_the_operational_requeue_waits_for_a_common_discovered_fragment():
    """(c): the cross flow and the flow served only by the leaderless switch
    I are NOT programmed, although the operational re-queue sees them every
    tick.  A relay C=D joins the two components: the recompute merges two
    discovered fragments and the cross flow is programmed at once.  A link
    E-I then brings I under a controller: the merged fragment holds an
    undiscovered switch, its walk continues, and I's flow is programmed
    once the walk has covered it ((d) holds only while I has no
    controller)."""
    sim, sev = _scenario_a()
    c = _ctrl(sim)
    _run(c, range(sev, sev + 60))
    assert c.phase == 'operational', c.phase
    assert c.programmed_flows == {('u1', 'u2'), ('u3', 'u4')}
    assert 'I' not in c.fragment_leaders.values()
    assert not any('I' in d for d in c.discovered.values())
    assert c.flow_queue == []
    # a relay joins the two components
    sim.topology.links['relay_C_D'] = _Link('C', 'D')
    _run(c, range(sev + 60, sev + 62))
    assert ('u1', 'u4') in c.programmed_flows, c.programmed_flows
    assert ('i1', 'i2') not in c.programmed_flows
    merged = [f for f, sw in c.fragment_switches.items() if 'A' in sw]
    assert len(merged) == 1 and c.fragment_switches[merged[0]] >= set('ABCDEF')
    assert c.discovery_done[merged[0]] is True
    # every per-fragment dict is keyed by the new decomposition only
    for d in (c.discovery_done, c.discovered, c.discovery_frontier,
              c.fragment_leaders, c.election_done_tick, c._discovery_round_tick):
        assert set(d) <= set(c.fragment_switches), (d, c.fragment_switches)
    # I joins the led fragment: one more discovery round reaches it
    sim.topology.links['l_E_I'] = _Link('E', 'I')
    hist = []
    _run(c, range(sev + 62, sev + 70), hist)
    assert ('i1', 'i2') in c.programmed_flows, hist
    assert any(p == 'discovery' for _t, p, _n, _f in hist), hist
    f_i = [f for f, sw in c.fragment_switches.items() if 'I' in sw][0]
    assert c.discovered[f_i] == c.fragment_switches[f_i]


# ── (c2) and its riders: Scenario B ────────────────────────────────────────
def _scenario_b():
    """Scenario B (severance at tick 0): the isolated switch X first in node
    order (fragment 0, below MIN_FRAGMENT_FOR_ELECTION) and the component
    {A, B, C} (fragment 1).  Flows at creation: one in each."""
    t = _topo(['X', 'A', 'B', 'C'], [('A', 'B'), ('B', 'C')],
              {'x1': ['X'], 'x2': ['X'], 'u1': ['A'], 'u2': ['C']})
    return _Sim(t, [('x1', 'x2'), ('u1', 'u2')], 0)


def _arrive(sim, ue, anchors, flows):
    sim.topology.nodes[ue] = _Node('UE')
    for a in anchors:
        sim.topology.links['uu_%s_%s' % (ue, a)] = _Link(ue, a)
    sim.ue_to_ue_flows += flows


def test_scenario_b_seeds_every_fragment_and_programs_the_rescue_arrivals():
    """(c2) and rider 1: mark_pre_converged marks EVERY fragment discovered
    with its switches as the discovered set -- the leaderless X included.
    Rescue UEs arrive at tick 30: the re-queue programs the flows inside a
    fragment, the one served only by X included (the pre-converged
    controller is converged by construction, so (d) does not apply), and not
    the cross flow."""
    sim = _scenario_b()
    c = _ctrl(sim)
    assert c.severance_tick == 0
    c.mark_pre_converged()
    assert c.phase == 'operational'
    assert c.programmed_flows == {('x1', 'x2'), ('u1', 'u2')}
    assert c.fragment_switches == {0: {'X'}, 1: {'A', 'B', 'C'}}
    assert set(c.fragment_leaders) == {1}, c.fragment_leaders   # X elects nobody
    assert c.discovery_done == {0: True, 1: True}
    assert c.discovered == {0: {'X'}, 1: {'A', 'B', 'C'}}
    _run(c, range(0, 30))
    assert len(c.programmed_flows) == 2
    _arrive(sim, 'r1', ['B'], [('r1', 'u2')])
    _arrive(sim, 'r2', ['X'], [('r2', 'x1'), ('r1', 'r2')])
    hist = []
    _run(c, range(30, 40), hist)
    assert ('r1', 'u2') in c.programmed_flows
    assert ('r2', 'x1') in c.programmed_flows, 'rider 1: a leaderless fragment of a pre-converged controller'
    assert ('r1', 'r2') not in c.programmed_flows, 'no infrastructure path'
    assert hist[0][2] == 4, hist[0]          # both programmed at tick 30
    assert c._walk_started is False


def test_scenario_b_survives_a_recompute_that_renumbers_the_fragments():
    """Rider 2: at the arrival tick a link X-A joins X to {A, B, C}; the
    forced recompute renumbers (two fragments 0 = {X}, 1 = {A, B, C} become
    one fragment 0) and must carry the pre-converged state over, not clear
    it: the arrival flow r1 -> x1, anchored at B and X, is programmed.  A
    rebuild-on-force would leave no fragment marked discovered and drop it
    (the 222 -> 310 / 292 check of acceptance item 2)."""
    sim = _scenario_b()
    c = _ctrl(sim)
    c.mark_pre_converged()
    _run(c, range(0, 30))
    sim.topology.links['l_X_A'] = _Link('X', 'A')
    _arrive(sim, 'r1', ['B'], [('r1', 'x1'), ('r1', 'u1')])
    hist = []
    _run(c, range(30, 35), hist)
    # programmed in the arrival tick itself: the pre-converged controller
    # never walks (a controller that had to re-discover the merged fragment
    # would program them three rounds later)
    assert hist[0][2] == 4, hist
    assert c._walk_started is False
    assert c.fragment_switches == {0: {'X', 'A', 'B', 'C'}}, c.fragment_switches
    assert c.discovery_done == {0: True}, c.discovery_done
    assert c.discovered == {0: {'X', 'A', 'B', 'C'}}, c.discovered
    assert {('r1', 'x1'), ('r1', 'u1')} <= c.programmed_flows, c.programmed_flows
    for d in (c.discovery_done, c.discovered, c.discovery_frontier,
              c.fragment_leaders, c.election_done_tick, c._discovery_round_tick):
        assert set(d) <= {0}, d
