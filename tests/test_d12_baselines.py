# -*- coding: utf-8 -*-
"""D12F.8 -- the leaf model for the OLSR, B.A.T.M.A.N. and AODV control
planes (docs/DECISIONS.md D12F.8; author's decision of 2026-10-03).

The real controllers on a small fake topology (no simulator).  Two
infrastructure components, A = A0-A1-A2 and B = B0-B1, are joined only by
the UE `us`, anchored in A1 and in B0; `ua1`, `ua2` sit in A, `ub1` in B.
Before D12F.8 these three controllers ran their fragment BFS over every
survivor, UEs included, so `us` stitched A and B into one fragment: the
cross flow ua1 -> ub1 entered their routable-flow sets (OLSR, B.A.T.M.A.N.)
or their RREQ queue and was "established" over the UE-inclusive graph
(AODV), and OLSR's and B.A.T.M.A.N.'s convergence ramps were timed over the
stitched fragment.  The data plane never delivered such a flow (it is
forwarded on the path view behind the infrastructure gate, D12.1); what the
controllers decided was WHEN a flow became routable and what they billed.

  * every routing baseline (OSPF, OLSR, B.A.T.M.A.N., AODV) computes the
    same decomposition: infrastructure components, UEs leaves of each
    fragment holding a live anchor, a fresh numbering on a recompute;
  * OLSR, B.A.T.M.A.N. and AODV never mark the cross flow routable, and do
    route the flows inside one fragment (so the test cannot pass vacuously);
  * AODV establishes no route through a UE even when its gate is bypassed;
  * AODV's overhead: a UE that hears a RREQ does not rebroadcast it, and no
    RREP is billed for a route that only exists through a UE.
"""
import os
import sys

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim.topology import Topology as _RealTopology  # noqa: E402


class _NT:
    def __init__(self, v):
        self.value = v


class _Node:
    def __init__(self, t, survivor=True):
        self.node_type = _NT(t)
        self.is_survivor = survivor


class _Link:
    def __init__(self, a, b, up=True):
        self.endpoints = (a, b)
        self.is_up = up


class _Topo:
    # the real path view (D12.1): a UE may appear only as a flow endpoint
    infra_node_set = _RealTopology.infra_node_set
    path_view = _RealTopology.path_view

    def __init__(self):
        self.nodes = {}
        self.links = {}
        self.graph = nx.Graph()

    def is_ue_node(self, nid):
        n = self.nodes.get(nid)
        return bool(n) and n.node_type.value == 'UE'

    def add_node(self, nid, t, survivor=True):
        self.nodes[nid] = _Node(t, survivor)
        self.graph.add_node(nid)

    def add_link(self, a, b, up=True):
        lid = 'l_%s_%s' % (a, b)
        self.links[lid] = _Link(a, b, up)
        if up:
            self.graph.add_edge(a, b, link_id=lid)


class _Ev:
    def __init__(self, tick, et):
        self.tick = tick
        self.event_type = et


class _Sim:
    def __init__(self, topo, flows):
        self.topology = topo
        self.ue_to_ue_flows = list(flows)
        # severance at tick 0: the controllers' rescue-ops timing
        self.scenario = type('S', (), {'events': [_Ev(0, 'sever_core')]})()
        self.phy_mac_states = {n: object() for n in topo.nodes}
        self.current_tick = 0


CROSS = ('ua1', 'ub1')          # A -> B, joined only by the UE `us`
INSIDE = [('ua1', 'ua2'), ('us', 'ub1')]   # inside A; inside B (us is B's leaf too)


def _world(stitch=True, bridge=False):
    t = _Topo()
    for n in ('A0', 'A1', 'A2', 'B0', 'B1', 'Z9'):
        t.add_node(n, 'O-RU')
    t.add_node('C0', 'Core')                       # the dead core: never a member
    for u in ('ua1', 'ua2', 'ub1', 'us', 'ux'):
        t.add_node(u, 'UE')
    t.add_node('ud', 'UE', survivor=False)         # a casualty: never a member
    for a, b in (('A0', 'A1'), ('A1', 'A2'), ('B0', 'B1'), ('C0', 'A0')):
        t.add_link(a, b)
    t.add_link('ua1', 'A0')
    t.add_link('ua2', 'A2')
    t.add_link('ub1', 'B1')
    t.add_link('us', 'A1')
    if stitch:
        t.add_link('us', 'B0')
    t.add_link('ud', 'A0')
    t.add_link('ux', 'B1', up=False)               # a down Uu link: no anchor
    if bridge:
        t.add_link('A2', 'B0')                     # an infrastructure link joins A and B
    return t


def _ctrls():
    import run_timeline_comparison as R
    return [('OSPF', R.OSPFController), ('OLSR', R.OLSRController),
            ('B.A.T.M.A.N.', R.BATMANController), ('AODV', R.AODVController)]


EXPECTED = {0: {'A0', 'A1', 'A2', 'ua1', 'ua2', 'us'},
            1: {'B0', 'B1', 'ub1', 'us'},
            2: {'Z9'}}


def test_every_routing_baseline_uses_the_infrastructure_fragments_with_ue_leaves():
    """OSPF, OLSR, B.A.T.M.A.N. and AODV compute ONE decomposition: the
    infrastructure components (the core and the casualty excluded, a down
    Uu link no anchor) numbered by lowest node id, `us` a leaf of both A
    and B.  A forced recompute after an infrastructure link joins A and B
    gives a fresh numbering with no stale fragment id (the pre-D12F.8 BFS
    kept the old higher ids)."""
    import run_timeline_comparison as R
    assert R.infra_fragments_with_ue_leaves(_world()) == EXPECTED
    checked = 0
    for name, cls in _ctrls():
        topo = _world()
        c = cls(_Sim(topo, [CROSS] + INSIDE))
        c._compute_fragments()
        assert c.fragment_members == EXPECTED, (name, c.fragment_members)
        topo.add_link('A2', 'B0')
        c._compute_fragments(force=True)
        assert c.fragment_members == {
            0: {'A0', 'A1', 'A2', 'B0', 'B1', 'ua1', 'ua2', 'ub1', 'us'},
            1: {'Z9'}}, (name, c.fragment_members)
        checked += 1
    assert checked == 4, checked


def _run(ctrl, ticks):
    seen = {}
    for tick in ticks:
        ctrl.sim.current_tick = tick
        ctrl.step(tick)
        for f in ctrl.get_routable_flows():
            seen.setdefault(f, tick)
    return seen


def test_no_baseline_routes_a_flow_joined_only_by_a_ue():
    """OLSR, B.A.T.M.A.N. and AODV over 400 ticks: the cross flow is never
    routable; both flows inside one fragment become routable (each count
    is checked, so the test cannot pass on a controller that routes
    nothing).  With an infrastructure link between A and B the cross flow
    does become routable (the positive control of the gate itself)."""
    checked = 0
    for name, cls in ((n, c) for n, c in _ctrls() if n != 'OSPF'):
        seen = _run(cls(_Sim(_world(), [CROSS] + INSIDE)), range(0, 400))
        assert CROSS not in seen, (name, seen)
        for f in INSIDE:
            assert f in seen, (name, f, seen)
        bridged = _run(cls(_Sim(_world(bridge=True), [CROSS] + INSIDE)), range(0, 400))
        assert CROSS in bridged, (name, bridged)
        checked += 1
    assert checked == 3, checked


def test_aodv_establishes_no_route_through_a_ue_even_past_its_gate():
    """The RREQ/RREP check on its own: the fragment gate is bypassed (one
    stitched fragment, as before D12F.8) and both flows are put in
    discovery at tick 0.  The flow inside A is established; the cross flow
    is not, because the only path through `us` is not a path a RREQ can
    take (it is relayed by infrastructure only)."""
    import run_timeline_comparison as R
    c = R.AODVController(_Sim(_world(), [CROSS, INSIDE[0]]))
    c._compute_fragments = lambda force=False: None
    c.fragment_members = {0: {'A0', 'A1', 'A2', 'B0', 'B1', 'ua1', 'ua2', 'ub1', 'us'}}
    c._fragments_computed = True
    c._active_rreqs = {CROSS: 0, INSIDE[0]: 0}
    established = set()
    for tick in range(0, R.AODV_RREQ_TIMEOUT + R.AODV_RREP_TIMEOUT + 3):
        c.step(tick)
        established |= set(c.route_cache)
    assert INSIDE[0] in established, established
    assert CROSS not in established, established


def _aodv_bill(topo, flow, tick):
    import run_timeline_comparison as R
    sim = _Sim(topo, [flow])
    ctrl = R.AODVController(sim)
    ctrl._active_rreqs = {flow: 0}
    ctrl.phase = 'rrep_path'
    sim._aodv_ctrl = ctrl
    cp = R.ControlPlaneMessageSim(sim, 'aodv')
    survivors = [n for n, nd in topo.nodes.items() if nd.is_survivor]
    return cp._sim_aodv(tick, survivors, len(survivors))


def test_aodv_overhead_a_ue_does_not_rebroadcast_and_no_rrep_through_a_ue():
    """The RREQ flood from ua1: `us` hears it at depth 3 but relays
    nothing, so at every depth the bill equals the bill on the same
    topology without the us-B0 link (before D12F.8 `us` rebroadcast into B
    and the two differed from depth 3 on).  In the RREP window the cross
    flow, routable only through `us`, bills no RREP; the flow inside A
    bills one per tick."""
    import run_timeline_comparison as R
    compared = 0
    for tick in range(1, R.AODV_RREQ_TIMEOUT):
        a = _aodv_bill(_world(stitch=True), CROSS, tick)
        b = _aodv_bill(_world(stitch=False), CROSS, tick)
        assert a == b, (tick, a, b)
        compared += 1
    assert compared >= 3, compared
    # depth 3 reaches `us`: the flood is not empty there (non-vacuous)
    assert _aodv_bill(_world(), CROSS, 2)[0] > 0
    rrep_tick = R.AODV_RREQ_TIMEOUT
    assert _aodv_bill(_world(), CROSS, rrep_tick) == (0, 0)
    assert _aodv_bill(_world(), INSIDE[0], rrep_tick)[0] == 1
