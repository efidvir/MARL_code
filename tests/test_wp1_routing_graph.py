"""WP1 -- routing-graph consistency (RERUN_SPEC WP1, items 1-6).

Covers the representative rule and sync_pair, the single relay removal path at
BOTH teardown sites (with node ids that contain underscores, which is what the
old `rid.split('_')` recovery broke on), unique relay link ids, the
interface-based relay predicate, consume_path's refusal to grant capacity over
an unresolved hop, and the strict end-of-tick invariant.

Runs under pytest and under tests/run_tests.py.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                               InterfaceType, is_transport_relay)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

# Node ids deliberately contain underscores: '{type}_{n}' is what the
# topology builders produce, and it is exactly what broke the old id-parsing
# teardown ('O-DU_18'.split('_', 1) -> ('O-DU', '18')).
A = 'O-DU_18'
B = 'Relay_7'
C = 'EdgeUPF_3'
UE = 'UE_42'
UE2 = 'UE_43'


def _topo():
    t = Topology()
    for nid, ntype in ((A, NodeType.O_DU), (B, NodeType.RELAY),
                       (C, NodeType.EDGEUPF), (UE, NodeType.UE)):
        t.add_node(Node(id=nid, node_type=ntype))
    return t


def _link(lid, u, v, cap=100.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL, up=True):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=up)


def _relay(lid, u, v, cap=30.0, up=True):
    return _link(lid, u, v, cap, LinkType.TRANSPORT_RELAY,
                 InterfaceType.BACKHAUL, up)


def _iab(lid, u, ue, cap=20.0, up=True):
    """An integrated-access link: TRANSPORT_RELAY type, Uu interface."""
    return _link(lid, u, ue, cap, LinkType.TRANSPORT_RELAY,
                 InterfaceType.Uu, up)


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


# ----------------------------------------------------------------------
# 1. the predicate (item 4)
# ----------------------------------------------------------------------

def test_is_transport_relay_excludes_iab_access_links():
    assert is_transport_relay(_relay('TR_%s_%s_1' % (A, B), A, B))
    # same link TYPE, access interface -> not a transport relay
    assert not is_transport_relay(_iab('ric_ue_access_%s_%s' % (A, UE), A, UE))
    assert not is_transport_relay(_link('Backhaul_1', A, B))
    assert not is_transport_relay(None)


def test_no_engine_predicate_tests_the_link_type_alone():
    """Every relay predicate in the engine goes through is_transport_relay."""
    src = _src('sixg_sim', 'simulation.py')
    # The two remaining type-only tests are deliberate: both sit in the
    # Priority-2 scan, which skips every relay PRODUCT (IAB access links
    # included) when it looks for an underlying physical hop to repurpose.
    # Radio-class tests (`lt is LinkType.TRANSPORT_RELAY` in the interference
    # emitter, the island upper bound) are not relay predicates and are not
    # matched here.
    allowed = 2
    n = sum(src.count(p) for p in (
        "link_type', None) == LinkType.TRANSPORT_RELAY",
        "link_type', None) != LinkType.TRANSPORT_RELAY",
        "link_type', None) is LinkType.TRANSPORT_RELAY",
        "link_type', None) is not LinkType.TRANSPORT_RELAY",
        "link_type == LinkType.TRANSPORT_RELAY",
    ))
    assert n == allowed, 'type-only relay predicates in simulation.py: %d' % n


# ----------------------------------------------------------------------
# 2. pair_representative and sync_pair (item 2)
# ----------------------------------------------------------------------

def test_pair_representative_prefers_relay_then_lowest_id():
    t = _topo()
    phys = _link('Backhaul_9', A, B, cap=800.0)
    rel = _relay('TR_%s_%s_2' % (A, B), A, B, cap=30.0)
    t.add_link(phys)
    t.add_link(rel)
    # a live relay REPRESENTS the hop, whatever the ids sort like
    assert t.pair_representative(A, B).id == rel.id
    assert t.pair_representative(B, A).id == rel.id     # order-independent

    # two physical links: lowest id wins, not dict insertion order
    t2 = _topo()
    t2.add_link(_link('Backhaul_9', A, B))
    t2.add_link(_link('Backhaul_10', A, B))
    assert t2.pair_representative(A, B).id == 'Backhaul_10'  # '1' < '9'

    # an IAB access link is NOT preferred over a physical link of its pair
    t3 = _topo()
    t3.add_link(_link('Uu_1', UE, A, lt=LinkType.WIRELESS,
                      iface=InterfaceType.Uu))
    t3.links['ric_ue_access_%s_%s' % (A, UE)] = _iab(
        'ric_ue_access_%s_%s' % (A, UE), A, UE)
    assert t3.pair_representative(A, UE).id == 'Uu_1'        # 'U' < 'r'

    # a pair with no links at all
    assert t.pair_representative(A, C) is None


def test_pair_representative_skips_down_links_and_honours_predicate():
    t = _topo()
    t.add_link(_link('Backhaul_9', A, B, cap=800.0))
    t.add_link(_relay('TR_%s_%s_2' % (A, B), A, B, up=False))
    # the relay is down, so the physical link represents the hop
    assert t.pair_representative(A, B).id == 'Backhaul_9'

    # a believed-state predicate substitutes its own notion of "up"
    believed = {'TR_%s_%s_2' % (A, B): True, 'Backhaul_9': False}
    rep = t.pair_representative(
        A, B, up_predicate=lambda lk: believed.get(lk.id, lk.is_up))
    assert rep.id == 'TR_%s_%s_2' % (A, B)

    # nothing up at all -> no representative
    for lk in t.links.values():
        lk.is_up = False
    assert t.pair_representative(A, B) is None


def test_sync_pair_sets_refreshes_and_removes_the_edge():
    t = _topo()
    phys = _link('Backhaul_9', A, B, cap=800.0)
    t.add_link(phys)
    rel = _relay('TR_%s_%s_2' % (A, B), A, B, cap=30.0)

    # a relay appears on top of the physical hop: it takes the edge over
    t.links[rel.id] = rel
    assert t.sync_pair(A, B) is rel
    assert t.graph[A][B]['link_id'] == rel.id
    assert t.graph[A][B]['capacity'] == 30.0

    # the relay goes away: the edge falls back to the physical link
    del t.links[rel.id]
    t.sync_pair(A, B)
    assert t.graph[A][B]['link_id'] == 'Backhaul_9'
    assert t.graph[A][B]['capacity'] == 800.0

    # the last link goes away: the edge goes with it
    del t.links['Backhaul_9']
    assert t.sync_pair(A, B) is None
    assert not t.graph.has_edge(A, B)
    # ... and syncing a pair that never had a link is a harmless no-op
    assert t.sync_pair(A, C) is None
    assert not t.graph.has_edge(A, C)


def test_sync_pair_uses_the_popped_links_own_endpoints():
    """The regression the whole item is about: endpoints come from the Link."""
    t = _topo()
    rel = _relay('TR_%s_%s_1' % (A, B), A, B)
    t.add_link(rel)
    assert t.graph.has_edge(A, B)
    popped = t.links.pop(rel.id)
    # the old code stripped 'TR_' and split at the first '_' -> ('O-DU', '18')
    n1, n2 = rel.id[len('TR_'):].split('_', 1)
    assert (n1, n2) != (A, B), 'id parsing must not be able to recover these'
    t.sync_pair(popped.endpoints[0], popped.endpoints[1])
    assert not t.graph.has_edge(A, B)


# ----------------------------------------------------------------------
# 3. update_link_statuses and the invariant (items 2, 6)
# ----------------------------------------------------------------------

def test_update_link_statuses_removes_orphan_edges_and_repoints():
    t = _topo()
    t.add_link(_link('Backhaul_9', A, B, cap=800.0))
    rel = _relay('TR_%s_%s_3' % (A, B), A, B, cap=30.0)
    t.add_link(rel)
    # simulate the defect: pop the link, leave the edge behind
    del t.links[rel.id]
    assert t.graph[A][B]['link_id'] == rel.id
    assert t.check_consistency(), 'a stale edge must be reported'

    t.update_link_statuses()
    assert t.graph[A][B]['link_id'] == 'Backhaul_9'
    assert t.graph[A][B]['capacity'] == 800.0
    assert t.graph_repairs == 1
    assert t.check_consistency() == []


def test_update_link_statuses_drops_an_edge_whose_pair_has_no_link():
    t = _topo()
    rel = _relay('TR_%s_%s_4' % (A, C), A, C)
    t.add_link(rel)
    del t.links[rel.id]
    t.update_link_statuses()
    assert not t.graph.has_edge(A, C)
    assert t.graph_repairs == 1
    assert t.check_consistency() == []


def test_update_link_statuses_maintenance_is_not_a_repair():
    """is_up transitions and capacity changes are ordinary maintenance."""
    t = _topo()
    lk = _link('Backhaul_9', A, B, cap=800.0)
    t.add_link(lk)
    lk.is_up = False
    t.update_link_statuses()
    assert not t.graph.has_edge(A, B)
    lk.is_up = True
    lk.capacity = 240.0                       # e.g. a degradation event
    t.update_link_statuses()
    assert t.graph[A][B]['link_id'] == 'Backhaul_9'
    assert t.graph[A][B]['capacity'] == 240.0  # the view follows the Link
    assert t.graph_repairs == 0
    assert t.check_consistency() == []


def test_check_consistency_catches_each_violation_class():
    t = _topo()
    t.add_link(_link('Backhaul_9', A, B))
    assert t.check_consistency() == []

    # (a) edge resolving to nothing
    t.graph.add_edge(A, C, link_id='TR_ghost_1', capacity=1.0)
    assert any('unknown link' in v for v in t.check_consistency())
    t.graph.remove_edge(A, C)

    # (b) edge with no link_id
    t.graph.add_edge(A, C)
    assert any('no link_id' in v for v in t.check_consistency())
    t.graph.remove_edge(A, C)

    # (c) edge pointing at a DOWN link
    t.links['Backhaul_9'].is_up = False
    assert any('down link' in v for v in t.check_consistency())
    t.links['Backhaul_9'].is_up = True

    # (d) edge pointing at the wrong link of the pair
    rel = _relay('TR_%s_%s_5' % (A, B), A, B)
    t.links[rel.id] = rel
    assert any('representative' in v for v in t.check_consistency())
    t.sync_pair(A, B)
    assert t.check_consistency() == []

    # (e) edge pointing at a link of ANOTHER pair
    t.graph[A][B]['link_id'] = 'Backhaul_9'
    t.links['Backhaul_9'].endpoints = (A, C)
    assert any('endpoints' in v for v in t.check_consistency())
    t.links['Backhaul_9'].endpoints = (A, B)
    t.sync_pair(A, B)
    assert t.check_consistency() == []

    # (f) an up infrastructure link with no edge at all
    t.graph.remove_edge(A, B)
    assert any('has no edge' in v for v in t.check_consistency())


def test_check_consistency_ignores_a_missing_ue_edge():
    """(2) of the invariant is about infrastructure pairs only."""
    t = _topo()
    t.add_link(_link('Uu_1', UE, A, lt=LinkType.WIRELESS,
                     iface=InterfaceType.Uu))
    t.graph.remove_edge(UE, A)
    assert t.check_consistency() == []


# ----------------------------------------------------------------------
# 4. unique relay ids (item 3)
# ----------------------------------------------------------------------

def test_relay_link_ids_are_unique_per_creation():
    from sixg_sim.transport_relay_model import TransportRelayModel
    m = TransportRelayModel()
    first = m.create_link(A, B, 30.0)
    m.clear_node_links(A)
    second = m.create_link(A, B, 30.0)
    assert first != second, 'a re-created bridge must not alias the old id'
    assert first.startswith('TR_%s_%s_' % (A, B))
    assert second.startswith('TR_%s_%s_' % (A, B))
    # a fresh model restarts the sequence, so ids are reproducible per run
    assert TransportRelayModel().create_link(A, B, 30.0) == first


# ----------------------------------------------------------------------
# 5. no code parses a link id any more (item 1)
# ----------------------------------------------------------------------

def test_no_source_file_parses_a_relay_link_id():
    # built from parts so this scanner does not match its own source
    needles = ('replace(' + '"TR_"', 'replace(' + "'TR_'")
    bad = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in ('.git', '__pycache__', 'output', 'release',
                                'logs', 'checkpoints')
                   and not d.startswith('torchinductor')]
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding='utf-8', errors='replace') as fh:
                for n, line in enumerate(fh, 1):
                    if any(nd in line for nd in needles):
                        bad.append('%s:%d' % (os.path.relpath(path, ROOT), n))
    assert not bad, 'link-id parsing survives at: %s' % bad


def test_both_teardown_sites_call_only_the_helper():
    """Every id clear_node_links returns goes through _remove_relay_link."""
    src = _src('sixg_sim', 'simulation.py')
    tree = ast.parse(src)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # for <rid> in <ids>:  <one call>
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            continue
        call = node.body[0].value
        if (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == '_remove_relay_link'):
            sites.append(node.lineno)
    assert len(sites) == 2, (
        'expected exactly the two teardown sites, found %r' % sites)
    assert src.count('clear_node_links(') == 2


def test_no_writer_bypasses_the_edge_rule():
    """Relay/IAB creation and removal never call add_edge/remove_edge
    themselves; the only direct edge removals left in the engine are the
    scenario-event handlers, which end in update_link_statuses."""
    sim_src = _src('sixg_sim', 'simulation.py')
    har_src = _src('run_timeline_comparison.py')
    assert sim_src.count('self.topology.graph.add_edge(') == 0
    assert har_src.count('sim.topology.graph.add_edge(') == 0
    assert har_src.count('sim.topology.graph.remove_edge(') == 0
    assert har_src.count('sim.topology.sync_pair(') == 2       # IAB up / down
    # The believed graph (moved into the engine by WP2, replaced by the
    # link-state service in WP3) uses the ONE rule.  It calls
    # `_representative_of` on the `_links_by_pair` index rather than
    # `pair_representative`, because it needs the representative of EVERY
    # pair on EVERY tick and pair_representative rescans the whole link table
    # per call -- but that is the same function pair_representative itself
    # delegates to, with belief substituted for `is_up`.  What must stay
    # impossible is the service inventing a rule of its own.
    det_src = _src('sixg_sim', 'link_state_service.py')
    assert '_representative_of(' in det_src
    assert '_links_by_pair()' in det_src
    assert har_src.count('pair_representative(') == 0


# ----------------------------------------------------------------------
# 6. the simulator-level paths
# ----------------------------------------------------------------------

def _sim():
    """A Simulator over a small hand-built topology.

    Two infrastructure components: {A, B} joined by a physical hop, and {C}
    on its own.  Enough for a bridge, a Priority-2 relay, and a repoint.
    """
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario

    t = _topo()
    t.nodes[A].has_multihaul = True
    t.nodes[B].has_multihaul = True
    t.nodes[C].has_multihaul = True
    t.add_link(_link('Backhaul_9', A, B, cap=800.0))
    t.add_link(_link('Uu_1', UE, A, cap=100.0, lt=LinkType.WIRELESS,
                     iface=InterfaceType.Uu))
    scenario = Scenario(name='wp1', duration_ticks=10, events=[],
                        traffic_profiles={})
    sim = Simulator(t, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 5
    return sim


def _two_ue_sim():
    """_sim() plus a second UE anchored at C, for one UE-to-UE flow."""
    sim = _sim()
    sim.topology.add_node(Node(id=UE2, node_type=NodeType.UE))
    sim.topology.add_link(_link('Uu_2', UE2, C, cap=100.0,
                                lt=LinkType.WIRELESS,
                                iface=InterfaceType.Uu))
    return sim


def _make_relay(sim, u, v, cap=30.0, up=True):
    """Create a relay link exactly as the engine does."""
    lid = sim.transport_relay_model.create_link(u, v, cap, 15.0)
    sim.topology.links[lid] = _relay(lid, u, v, cap=cap, up=up)
    if up:
        sim.topology.sync_pair(u, v)
    return lid


def test_remove_relay_link_drops_the_edge_and_all_bookkeeping():
    sim = _sim()
    lid = _make_relay(sim, A, C)                     # a bridge: no other link
    sim._relay_link_ready_tick[lid] = 7
    sim._relay_bridge_since[lid] = 3
    sim._relay_link_load[lid] = 0.5
    sim._relay_link_passes = {lid: 2}                # WP4's dict, if present
    sim._relay_route_since = {lid: 4}                # WP3's dict, if present
    assert sim.topology.graph.has_edge(A, C)

    assert sim._remove_relay_link(lid) is not None

    assert lid not in sim.topology.links
    assert not sim.topology.graph.has_edge(A, C)
    assert lid not in sim._relay_link_ready_tick
    assert lid not in sim._relay_bridge_since
    assert lid not in sim._relay_link_load
    assert lid not in sim._relay_link_passes
    assert lid not in sim._relay_route_since
    assert sim._relay_links_removed == 1
    assert sim._link_tombstones[lid]['endpoints'] == (A, C)
    assert sim._link_tombstones[lid]['capacity'] == 30.0
    assert sim.topology.check_consistency() == []
    # an unknown id is a no-op, not an error, and is not counted
    assert sim._remove_relay_link('TR_nobody_1') is None
    assert sim._relay_links_removed == 1


def test_remove_relay_link_restores_the_physical_hop():
    """Priority-2: the relay repurposed an existing hop; it must come back."""
    sim = _sim()
    lid = _make_relay(sim, A, B, cap=30.0)
    assert sim.topology.graph[A][B]['link_id'] == lid
    sim._remove_relay_link(lid)
    assert sim.topology.graph[A][B]['link_id'] == 'Backhaul_9'
    assert sim.topology.graph[A][B]['capacity'] == 800.0
    assert sim.topology.check_consistency() == []


def test_teardown_site_step_phy_mac_with_underscore_node_ids():
    """Teardown site 1: a relay-mode flip in _step_phy_mac."""
    from sixg_sim.agent import PHYMACAction
    from sixg_sim.phy_mac_state import RelayMode, MCS_OFFSET_CENTER

    sim = _sim()
    lid = _make_relay(sim, A, B, cap=30.0)
    ps = sim.phy_mac_states[A]
    ps.relay_mode = RelayMode.CAPACITY_BOOST
    ps.relay_link_active = True
    ps.relay_peer_node = B
    ps.relay_link_capacity_mbps = 30.0

    action = PHYMACAction(
        tx_power_step=2,
        mcs_emergency_idx=MCS_OFFSET_CENTER,
        mcs_general_idx=MCS_OFFSET_CENTER,
        prb_emergency_frac=ps.prb_emergency_fraction,
        prb_relay_frac=ps.prb_relay_fraction,
        prb_general_frac=ps.prb_general_fraction,
        relay_mode_idx=list(RelayMode).index(RelayMode.OFF),
        send_postcard=False,
        postcard_content=None,
    )
    sim._step_phy_mac(A, action)

    assert lid not in sim.topology.links
    assert sim.topology.graph[A][B]['link_id'] == 'Backhaul_9'
    assert ps.relay_link_active is False
    assert sim._relay_links_removed == 1
    assert sim.topology.check_consistency() == []


def test_teardown_removes_a_link_whose_far_end_acts():
    """clear_node_links returns links where the node is the TARGET too."""
    sim = _sim()
    lid = _make_relay(sim, B, C)          # created by B, torn down by C
    assert sim.topology.graph.has_edge(B, C)
    removed = sim.transport_relay_model.clear_node_links(C)
    assert removed == [lid]
    for rid in removed:
        sim._remove_relay_link(rid)
    assert not sim.topology.graph.has_edge(B, C)
    assert sim.topology.check_consistency() == []


def test_teardown_site_repoint_release_with_underscore_node_ids():
    """Teardown site 2: the release inside _maybe_repoint_to_fragment."""
    from sixg_sim.phy_mac_state import RelayMode

    sim = _sim()
    lid = _make_relay(sim, A, B, cap=30.0)   # intra-component: not a bridge
    ps = sim.phy_mac_states[A]
    ps.relay_mode = RelayMode.CAPACITY_BOOST
    ps.relay_link_active = True
    ps.relay_peer_node = B
    sim.island_mode = True
    # C is in its own infra component, so it is a candidate; the link budget
    # is not what this test is about.
    sim.transport_relay_model.find_relay_candidates = (
        lambda **kw: [(C, 30.0, 15.0)])

    assert sim._maybe_repoint_to_fragment(A, ps) is True
    assert lid not in sim.topology.links
    assert sim.topology.graph[A][B]['link_id'] == 'Backhaul_9'
    assert ps.relay_link_active is False
    assert sim._relay_links_removed == 1
    assert sim.topology.check_consistency() == []


# ----------------------------------------------------------------------
# 7. consume_path (item 5)
# ----------------------------------------------------------------------

def _run_one_flow(sim, src, dst):
    """Forward exactly one UE-to-UE flow and return the delivered volume."""
    sim.ue_to_ue_flows = [(src, dst)]
    sim.island_mode = True
    sim.marl_ue_routing_enabled = True
    for lk in sim.topology.links.values():
        lk.reset_utilization()
    sim._forward_traffic(sim._generate_traffic())
    return float(getattr(sim, 'last_tick_ue_to_ue_volume', 0.0))


def test_consume_path_never_grants_capacity_over_an_unresolved_hop():
    """A hop whose edge resolves to no Link delivers 0 for that flow.

    The defect: the old loop did `continue`, so the hop imposed no capacity
    limit at all and an orphan edge became a free shortcut.  The shape here
    is the defect's form (b): the pair still has a live Link (so the
    link-based flow gate passes and the path is chosen), but the edge names
    a link that does not exist.
    """
    sim = _two_ue_sim()
    lid = _make_relay(sim, B, C, cap=50.0)
    sim.topology.graph[B][C]['link_id'] = 'TR_%s_%s_99' % (B, C)   # ghost
    assert sim.count_stale_edges() == 1

    delivered = _run_one_flow(sim, UE, UE2)
    assert delivered == 0.0
    assert sim._diag_orphan_hops >= 1
    assert sim._diag_stale_u2u_tick > 0.0
    assert sim._diag_stale_edges_tick == 1
    # nothing was charged to any link on the path
    assert all(lk.current_utilization == 0.0
               for lk in sim.topology.links.values())
    assert sim.topology.links[lid].current_utilization == 0.0


def test_consume_path_refuses_an_edge_with_no_link_id():
    sim = _two_ue_sim()
    _make_relay(sim, B, C, cap=50.0)
    del sim.topology.graph[B][C]['link_id']      # edge names nothing
    assert sim.count_stale_edges() == 1
    delivered = _run_one_flow(sim, UE, UE2)
    assert delivered == 0.0
    assert sim._diag_orphan_hops >= 1
    assert sim._diag_stale_u2u_tick > 0.0


def test_consume_path_still_delivers_over_a_healthy_path():
    """Control for the two tests above: the same shape, with a real link."""
    sim = _two_ue_sim()
    lid = _make_relay(sim, B, C, cap=50.0)
    delivered = _run_one_flow(sim, UE, UE2)
    assert delivered > 0.0
    assert sim.topology.links[lid].current_utilization > 0.0
    assert sim._diag_orphan_hops == 0
    assert sim._diag_stale_u2u_tick == 0.0
    assert sim._diag_stale_edges_tick == 0
    assert sim.count_stale_edges() == 0


# ----------------------------------------------------------------------
# 8. the strict end-of-tick invariant (item 6)
# ----------------------------------------------------------------------

def test_check_graph_consistency_raises_on_a_stale_edge():
    sim = _sim()
    sim.check_graph_consistency(where='unit')          # clean: no raise
    lid = _make_relay(sim, A, C)
    del sim.topology.links[lid]                        # leave the edge behind
    try:
        sim.check_graph_consistency(where='unit')
    except AssertionError as exc:
        assert 'routing-graph invariant violated' in str(exc)
        assert 'unit' in str(exc)
        assert lid in str(exc)
    else:
        raise AssertionError('a stale edge must stop the run')


def test_drivers_check_the_invariant_every_tick():
    """The check is wired into the shared per-tick step (WP2), on the TRUE
    graph, and both drivers tick only through that step."""
    har = _src('run_timeline_comparison.py')
    trn = _src('train_on_comparison.py')
    step = _src('sixg_sim', 'tick_step.py')
    assert step.count('sim.check_graph_consistency(') == 1
    assert har.count('check_graph_consistency(') == 0
    assert trn.count('check_graph_consistency(') == 0
    assert har.count('sim.step_tick(') == 1
    assert trn.count('sim.step_tick(') == 2
    # ... and it runs BEFORE the believed graph is installed and before
    # traffic is generated for the pass
    i_check = step.index('sim.check_graph_consistency(')
    assert i_check < step.index('sim.topology.graph = believed')
    assert i_check < step.index('traffic = sim._generate_traffic()')
    # the per-tick counters are recorded in the result pickles
    for key in ("'stale_edges':", "'stale_u2u_volume':", "'orphan_hops':",
                "'graph_repairs':", "'relay_links_removed':"):
        assert key in har, key


def test_reference_topology_starts_consistent_without_parallel_links():
    """The representative rule can only matter on pairs with several links;
    the comparison topologies have none, so it cannot move the routing
    and SDN arms (section-14 invariance for WP1)."""
    import collections
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    for seed in (82, 91):
        t, _info = R.build_comparison_topology(seed)
        by_pair = collections.Counter(tuple(sorted(l.endpoints))
                                      for l in t.links.values())
        assert max(by_pair.values()) == 1, 'parallel links in seed %d' % seed
        assert t.check_consistency() == []
