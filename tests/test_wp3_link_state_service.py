"""WP3 -- the link-state service for the cross-layer arms (RERUN_SPEC WP3).

Fast checks on hand-built topologies (seconds), with every timing figure
hand-computed in the test rather than read back from the implementation:

  1. the constants really are the OSPF baseline's;
  2. detection timing -- an existing link's failure at EXACTLY 3 ticks, a new
     link's adjacency at 3 ticks, an administrative teardown immediately at
     the owning endpoint, and a never-seen link id DOWN after island onset;
  3. the installation formula on a 5-node path with known eccentricities,
     including the MinLSInterval throttle, the idealised-timing switch and
     the core node that must NOT lengthen an eccentricity (D3.17);
  4. tombstones -- a flow over a physically-down but not-yet-uninstalled link
     delivers EXACTLY 0, and it is not counted as a stale edge; a relay
     re-created on that pair before the removal installs carries nothing
     (D3.16); and the forwarding view never offers more capacity than the
     hop physically has (D3.15);
  5. billing -- exactly zero bytes before the event for EVERY arm, the
     BFD/hello/LSA arithmetic afterwards, and the flag that says whose
     per-type breakdown is real (D3.18);
  6. an integration run through Simulator.step_tick.

Runs under pytest and under tests/run_tests.py.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                               InterfaceType)
from sixg_sim import link_state_service as LS


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

# Node ids deliberately contain underscores ('{type}_{n}', what the topology
# builders produce).
A = 'O-DU_18'
B = 'Relay_7'
C = 'EdgeUPF_3'
CORE = 'AMF_1'
UE = 'UE_42'
UE2 = 'UE_43'


def _link(lid, u, v, cap=100.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL, up=True):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=up)


def _relay(lid, u, v, cap=30.0, up=True):
    return _link(lid, u, v, cap, LinkType.TRANSPORT_RELAY,
                 InterfaceType.BACKHAUL, up)


def _uu(lid, ue, anchor, cap=100.0, up=True):
    return _link(lid, ue, anchor, cap, LinkType.WIRELESS,
                 InterfaceType.Uu, up)


def _simulator(topology, ticks=200):
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    scenario = Scenario(name='wp3', duration_ticks=ticks, events=[],
                        traffic_profiles={})
    sim = Simulator(topology, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 0
    return sim


def _chain_sim():
    """Five routers in a PATH plus one core node hanging off N0:

        CORE -- N0 -- N1 -- N2 -- N3 -- N4

    Eccentricities in the infrastructure graph WITHOUT the core edge (which
    is what the tests that cut the core use) are 4, 3, 2, 3, 4.
    """
    t = Topology()
    t.add_node(Node(id=CORE, node_type=NodeType.AMF))
    for i in range(5):
        t.add_node(Node(id='O-DU_%d' % i, node_type=NodeType.O_DU))
    t.add_link(_link('Core_0', CORE, 'O-DU_0', cap=1000.0))
    for i in range(4):
        t.add_link(_link('Mesh_%d' % i, 'O-DU_%d' % i, 'O-DU_%d' % (i + 1),
                         cap=500.0))
    return _simulator(t)


def _service(sim, install=None):
    return LS.LinkStateService(sim, install=install)


def _pair(u, v):
    return tuple(sorted((u, v)))


def _set_link_up(sim, lid, up):
    lk = sim.topology.links[lid]
    lk.is_up = up
    sim.topology.sync_pair(lk.endpoints[0], lk.endpoints[1])


def _add_link(sim, lid, u, v, cap=500.0):
    sim.topology.links[lid] = _link(lid, u, v, cap=cap)
    sim.topology.sync_pair(u, v)
    return lid


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


def _harness():
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    return R


# ----------------------------------------------------------------------
# 1. the constants are the OSPF baseline's
# ----------------------------------------------------------------------

def test_constants_match_the_ospf_baseline():
    """RERUN_SPEC WP3: "Constants are those of the OSPF baseline model
    (_sim_ospf)".  They are duplicated in the service (the harness imports
    sixg_sim, so the service cannot import the harness); this is what stops
    the two copies drifting.  docs/LINK_STATE_SERVICE.md writes them out for
    the pre-registration."""
    R = _harness()
    assert LS.HELLO_INTERVAL == R.OSPF_HELLO_INTERVAL == 10
    assert LS.FLOOD_HOP_TICKS == R.OSPF_FLOOD_HOP_TICKS == 1
    assert LS.SPF_DELAY_TICKS == R.OSPF_SPF_DELAY == 5
    assert LS.MIN_LS_INTERVAL == R.OSPF_MIN_LS_INTERVAL == 5
    C_ = R.ControlPlaneMessageSim
    assert LS.HELLO_BASE == C_.OSPF_HELLO_BASE == 44
    assert LS.HELLO_PER_NBR == C_.OSPF_HELLO_PER_NBR == 4
    assert LS.LSA_BASE == C_.OSPF_LSA_BASE == 24
    assert LS.LSA_PER_LINK == C_.OSPF_LSA_PER_LINK == 12
    # BFD: 1 packet/s per direction, detect multiplier 3, 66 B on the wire
    assert LS.BFD_DETECT_TICKS == 3
    assert LS.BFD_BYTES == 66
    assert LS.BFD_PACKETS_PER_PAIR_PER_TICK == 2
    # the hello offset is bit-for-bit the baseline's staggering rule
    for nid in ('O-DU_0', 'Relay_7', 'EdgeUPF_3'):
        assert (LS.node_offset(nid, LS.HELLO_INTERVAL)
                == C_._get_node_offset(None, nid, LS.HELLO_INTERVAL))
    # and the routers billed are the ones the baseline bills
    sim = _chain_sim()
    assert sorted(LS.LinkStateService(sim)._routers()) == sorted(
        C_(sim, 'marl')._get_survivor_nodes())


def test_the_api_names_wp6_binds_to_exist():
    """The AGENT track built WP6 against these exact names, behind a one-line
    adapter (sixg_sim/agent.py: link_state_service / local_isolation_state).
    Renaming any of them turns that merge into a rewrite."""
    sim = _chain_sim()
    svc = LS.make_link_state_service(sim)
    assert sim.link_state_service is svc
    for name in ('step', 'get_routable_flows', 'believed_graph',
                 'tick_messages', 'installed_since', 'isolation_state'):
        assert callable(getattr(svc, name)), name
    assert isinstance(svc.visible, dict)
    svc.step(0)
    iso, since = svc.isolation_state('O-DU_0')
    assert iso is False and since is None
    assert svc.isolation_state('nobody') == (False, None)


# ----------------------------------------------------------------------
# 2. detection timing
# ----------------------------------------------------------------------

def test_existing_link_failure_is_detected_after_exactly_three_ticks():
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)                       # bootstrap: converged initial topology
    assert svc.visible['Mesh_1'] is True

    _set_link_up(sim, 'Mesh_1', False)          # physically down before tick 5
    for t in (5, 6, 7):
        svc.step(t)
        assert svc._link_belief['Mesh_1'] is True, (
            'BFD declared the session down after %d tick(s), not 3' % (t - 5))
    svc.step(8)
    assert svc._link_belief['Mesh_1'] is False
    assert svc._pair_confirmed[_pair('O-DU_1', 'O-DU_2')] is False


def test_a_flap_that_reverts_inside_the_detect_window_is_never_confirmed():
    """What a detect multiplier is for."""
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)
    _set_link_up(sim, 'Mesh_1', False)
    svc.step(5)
    svc.step(6)
    _set_link_up(sim, 'Mesh_1', True)
    svc.step(7)
    svc.step(8)
    svc.step(9)
    assert svc._link_belief['Mesh_1'] is True
    assert svc.detections == 0


def test_a_new_links_adjacency_comes_up_after_the_bfd_session():
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)

    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    pair = _pair('O-DU_0', 'O-DU_4')
    svc.step(1)
    # NO FREE ADOPTION: the link is physically up, the service believes it
    # down and starts a session.
    assert sim.topology.links['Mesh_ring'].is_up is True
    assert svc._link_belief['Mesh_ring'] is False
    assert svc._pair_confirmed[pair] is False
    for t in (2, 3):
        svc.step(t)
        assert svc._link_belief['Mesh_ring'] is False, t
    svc.step(4)
    assert svc._link_belief['Mesh_ring'] is True
    assert svc._pair_confirmed[pair] is True


def test_a_never_seen_link_is_adopted_at_once_before_island_onset():
    """The control for the test above: with the core still reachable the
    service is not in island mode and a new link converges with the rest of
    the (still centrally routed) network."""
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = False
    svc.step(0)
    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    svc.step(1)
    assert svc._link_belief['Mesh_ring'] is True
    assert svc._pair_confirmed[_pair('O-DU_0', 'O-DU_4')] is True


def test_an_administrative_teardown_is_immediate_at_the_owning_endpoint():
    """A router that shut its own interface does not wait for a keepalive to
    time out.  Compared directly against the same teardown with no recorded
    administrator, which must still take the full BFD timeout."""
    for admin, expect_immediate in ((True, True), (False, False)):
        sim = _chain_sim()
        svc = _service(sim)
        sim.island_mode = True
        svc.step(0)
        lid = sim.transport_relay_model.create_link('O-DU_0', 'O-DU_2',
                                                    30.0, 15.0)
        sim.topology.links[lid] = _relay(lid, 'O-DU_0', 'O-DU_2')
        sim.topology.sync_pair('O-DU_0', 'O-DU_2')
        pair = _pair('O-DU_0', 'O-DU_2')
        for t in (1, 2, 3, 4):
            svc.step(t)
        assert svc._pair_confirmed[pair] is True          # BFD session is up

        sim.current_tick = 5
        sim._remove_relay_link(lid, admin_node='O-DU_0' if admin else None)
        svc.step(5)
        if expect_immediate:
            assert svc._pair_confirmed[pair] is False, (
                'an administrative teardown by an endpoint still waited')
        else:
            assert svc._pair_confirmed[pair] is True
            svc.step(6)
            svc.step(7)
            assert svc._pair_confirmed[pair] is True
            svc.step(8)
            assert svc._pair_confirmed[pair] is False


def test_an_up_transition_never_takes_the_administrative_shortcut():
    """D3.2: every relay link is CREATED by one of its endpoints, so reading
    "known at once" as "believed up at once" would make the spec's own
    "a new link's adjacency up after the BFD session (3 ticks)" dead for
    exactly the links it is about.  The shortcut is down-only."""
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)
    lid = sim.transport_relay_model.create_link('O-DU_0', 'O-DU_3', 30.0, 15.0)
    sim.topology.links[lid] = _relay(lid, 'O-DU_0', 'O-DU_3')
    sim.topology.sync_pair('O-DU_0', 'O-DU_3')
    sim.current_tick = 1
    sim.record_link_admin_event(lid, 'O-DU_0', True)     # created by O-DU_0
    svc.step(1)
    assert svc._link_belief[lid] is False
    svc.step(2)
    svc.step(3)
    assert svc._link_belief[lid] is False
    svc.step(4)
    assert svc._link_belief[lid] is True


def test_access_adjacencies_are_not_in_the_link_state_domain():
    """D3.3: a pair with a UE endpoint follows physical truth -- no BFD, no
    flood, no SPF delay.  The anchor observes its own radio link directly and
    is the only node whose forwarding depends on it."""
    t = Topology()
    t.add_node(Node(id=CORE, node_type=NodeType.AMF))
    t.add_node(Node(id=A, node_type=NodeType.O_DU))
    t.add_node(Node(id=UE, node_type=NodeType.UE))
    t.add_link(_link('Core_0', CORE, A, cap=1000.0))
    t.add_link(_uu('Uu_1', UE, A))
    sim = _simulator(t)
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)
    _set_link_up(sim, 'Uu_1', False)
    svc.step(1)
    assert svc._link_belief['Uu_1'] is False
    assert svc._pair_installed[_pair(UE, A)] is False
    _set_link_up(sim, 'Uu_1', True)
    svc.step(2)
    assert svc._pair_installed[_pair(UE, A)] is True
    # ... and it originated no Router-LSA on the way
    assert svc._last_orig == {}


# ----------------------------------------------------------------------
# 3. the installation formula
# ----------------------------------------------------------------------

def test_installation_formula_on_a_hand_computed_topology():
    """CORE--N0--N1--N2--N3--N4 with the core edge cut, so the infrastructure
    graph is the 5-node path N0..N4 and the eccentricities are 4, 3, 2, 3, 4.

        t_o    = max(t_known[o], last_orig[o] + MIN_LS_INTERVAL)
        t_inst = max_o(t_o + FLOOD_HOP_TICKS * ecc_o) + SPF_DELAY_TICKS

    Ring-closing link N0--N4, created at tick 1:
        BFD session up (confirmed) at td = 1 + 3 = 4
        t_o(N0) = t_o(N4) = 4          (neither has originated before)
        ecc(N0) = ecc(N4) = 4          (in the INSTALLED graph: the path)
        t_inst  = max(4 + 1*4, 4 + 1*4) + 5 = 13
    """
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)                       # bootstrap with the core already cut
    assert svc._eccentricity('O-DU_0') == 4
    assert svc._eccentricity('O-DU_2') == 2
    assert svc._eccentricity('O-DU_4') == 4

    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    ring = _pair('O-DU_0', 'O-DU_4')
    for t in range(1, 5):
        svc.step(t)
    assert svc._pair_confirmed[ring] is True          # td = 4
    assert svc._pending_install[ring] == (True, 13)
    assert svc._last_orig['O-DU_0'] == 4 and svc._last_orig['O-DU_4'] == 4

    for t in range(5, 13):
        svc.step(t)
        assert svc._pair_installed[ring] is False, (
            'installed at %d, before t_inst = 13' % t)
        assert svc.believed_graph() is not None, t
    svc.step(13)
    assert svc._pair_installed[ring] is True
    assert svc.installed_since('Mesh_ring') == 13
    assert svc.believed_graph() is None          # installed == truth again


def test_the_installation_formula_ignores_non_router_transit_nodes():
    """The same topology with the core link STILL UP (D3.17).

    `ecc_o` is measured on the graph a Router-LSA can actually be flooded
    across, which is the graph of nodes that run the routing protocol on the
    mesh -- the same `_is_router` set that decides who originates an LSA and
    who is billed a hello.  The AMF hanging off N0 runs none, so it must not
    lengthen anyone's eccentricity:

        ecc(N4) = 4  (N4..N0), NOT 5 (N4..N0..AMF)
        ring link N0--N4 created at tick 1, confirmed at td = 4
        t_inst  = max(4 + 1*4, 4 + 1*4) + 5 = 13, NOT 14

    Every other installation test cuts the core before stepping, so without
    this case the definition of the flood graph is never pinned.
    """
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)                                   # bootstrap, core still up
    assert svc._pair_installed[_pair(CORE, 'O-DU_0')] is True
    assert svc._is_router(CORE) is False          # AMF: no routing on the mesh
    assert svc._eccentricity('O-DU_4') == 4       # not 5, through the AMF
    assert svc._eccentricity('O-DU_0') == 4
    assert svc._eccentricity('O-DU_2') == 2

    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    ring = _pair('O-DU_0', 'O-DU_4')
    for t in range(1, 5):
        svc.step(t)
    assert svc._pair_confirmed[ring] is True                  # td = 4
    assert svc._pending_install[ring] == (True, 13)


def test_the_min_ls_interval_throttle_delays_a_second_origination():
    """Same topology.  After the ring link's origination at tick 4, cut
    N0--N1 at tick 5:

        confirmed at td = 5 + 3 = 8
        t_o(N0) = max(8, last_orig[N0] + 5) = max(8, 4 + 5) = 9   <- throttled
        t_o(N1) = max(8, -)                 = 8
        ecc(N0) = 4, ecc(N1) = 3   (installed graph is still the path:
                                    the ring install matures at 13)
        t_inst  = max(9 + 4, 8 + 3) + 5 = 18

    Since D13.1 the throttled LSA is ORIGINATED at 9, not booked at 8 for
    tick 9: at the end of step 8 N0 has not announced the cut yet, and the
    installation tick is known only once it has.
    """
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    for t in range(1, 5):
        svc.step(t)
    assert svc._last_orig['O-DU_0'] == 4

    _set_link_up(sim, 'Mesh_0', False)               # N0--N1, down at tick 5
    cut = _pair('O-DU_0', 'O-DU_1')
    for t in range(5, 9):
        svc.step(t)
    assert svc._pair_confirmed[cut] is False          # td = 8
    assert svc._last_orig['O-DU_0'] == 4              # throttled by MinLS ...
    assert svc._last_orig['O-DU_1'] == 8
    assert svc.awaiting_announcement(cut)             # ... N0 still owes it
    assert svc.install_tick(cut) is None
    svc.step(9)
    assert svc._last_orig['O-DU_0'] == 9              # ... until tick 9
    assert not svc.awaiting_announcement(cut)
    assert svc._pending_install[cut] == (False, 18)
    assert svc.install_tick(cut) == 18


HUB = 'O-DU_0'


def _star_sim(leaves=6):
    """A hub router with `leaves` router leaves, plus the core (an AMF, which
    runs no routing protocol on the mesh) hanging off the hub:

        CORE -- HUB -- L1, L2, ..., Ln          (Mesh_i = HUB--L_i)

    In the installed router graph ecc(HUB) = 1 and ecc(L_i) = 2."""
    t = Topology()
    t.add_node(Node(id=CORE, node_type=NodeType.AMF))
    t.add_node(Node(id=HUB, node_type=NodeType.O_DU))
    for i in range(1, leaves + 1):
        t.add_node(Node(id='O-DU_%d' % i, node_type=NodeType.O_DU))
    t.add_link(_link('Core_0', CORE, HUB, cap=1000.0))
    for i in range(1, leaves + 1):
        t.add_link(_link('Mesh_%d' % i, HUB, 'O-DU_%d' % i, cap=500.0))
    return _simulator(t)


def test_k_simultaneous_failures_at_one_router_install_within_one_origination():
    """D13.1 (reviewer item 4, RFC 2328 S12.4).  k = 4 of the hub's links
    fail together before tick 5; BFD confirms all four at td = 8.

    A Router-LSA describes ALL of a router's links, so the hub announces the
    four changes in ONE origination at 8, and each leaf announces its own:

        t_o(HUB) = 8, ecc(HUB) = 1     -> arrival 9
        t_o(L_i) = 8, ecc(L_i) = 2     -> arrival 10
        t_inst   = max(9, 10) + 5 = 15   for ALL FOUR transitions

    Per-transition origination (the rule before D13.1) serialised the hub at
    MinLSInterval: originations at 8, 13, 18, 23 and installs at 15, 19,
    24, 29 -- the last one 5 x (k - 1) = 15 ticks late.  The bill follows
    the originations: one hub flood at 8 and none at 13, 18 or 23.
    """
    sim = _star_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)
    assert svc._eccentricity(HUB) == 1
    assert svc._eccentricity('O-DU_1') == 2
    cuts = [_pair(HUB, 'O-DU_%d' % i) for i in range(1, 5)]
    for i in range(1, 5):
        _set_link_up(sim, 'Mesh_%d' % i, False)
    for t in range(5, 8):
        svc.step(t)
    assert svc.originations == 0
    svc.step(8)
    assert all(svc._pair_confirmed[c] is False for c in cuts)
    # ONE LSA by the hub, one by each of the four leaves
    assert svc.originations == 5, svc.originations
    assert svc._last_orig[HUB] == 8
    assert svc.announced == 8          # 4 transitions x 2 endpoints
    ticks = [svc.install_tick(c) for c in cuts]
    assert ticks == [15, 15, 15, 15], (
        'k simultaneous transitions at one router must install within ONE '
        'origination; install ticks %r' % ticks)
    # the bill: the hub's single flood at 8 (its component after the cuts is
    # {CORE, HUB, L5, L6}: 3 pairs, 6 directed adjacencies); the cut-off
    # leaves flood nothing
    assert svc.tick_messages(8)['lsa_msgs'] == 6
    for t in range(9, 15):
        svc.step(t)
        assert all(svc._pair_installed[c] is True for c in cuts), t
        assert svc.tick_messages(t)['lsa_msgs'] == 0, t
    svc.step(15)
    assert all(svc._pair_installed[c] is False for c in cuts)
    for t in range(16, 30):
        svc.step(t)
        assert svc.tick_messages(t)['lsa_msgs'] == 0, (
            'a Router-LSA was originated at %d: the hub owed nothing' % t)
    assert svc.originations == 5
    assert svc._unannounced == {} and svc._awaiting == {}


def test_a_transition_learned_inside_the_min_ls_interval_waits():
    """D13.1.  After the hub's origination at 8 (four cuts, as above), two
    more of its links fail INSIDE its MinLSInterval window: Mesh_5 before
    tick 7 (confirmed 10) and Mesh_6 before tick 8 (confirmed 11).  The hub
    may not originate again before 8 + 5 = 13, and then carries BOTH in one
    LSA; the leaves announce their own at once:

        L5: t_o = 10, ecc 2 -> 12     L6: t_o = 11, ecc 2 -> 13
        HUB: t_o = max(10, 8 + 5) = 13, ecc 1 -> 14
        t_inst(L5) = max(14, 12) + 5 = 19,  t_inst(L6) = max(14, 13) + 5 = 19

    (ecc measured at each origination on the INSTALLED graph, which is still
    the full star: the first four cuts install at 15.)"""
    sim = _star_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)
    for i in range(1, 5):
        _set_link_up(sim, 'Mesh_%d' % i, False)
    for t in range(5, 7):
        svc.step(t)
    _set_link_up(sim, 'Mesh_5', False)              # pending from 7
    svc.step(7)
    _set_link_up(sim, 'Mesh_6', False)              # pending from 8
    svc.step(8)
    assert svc._last_orig[HUB] == 8
    c5, c6 = _pair(HUB, 'O-DU_5'), _pair(HUB, 'O-DU_6')
    for t in (9, 10, 11, 12):
        svc.step(t)
        assert svc._last_orig[HUB] == 8, (
            'the hub re-originated at %d, inside its MinLSInterval' % t)
    assert svc._last_orig['O-DU_5'] == 10
    assert svc._last_orig['O-DU_6'] == 11
    assert svc.awaiting_announcement(c5) and svc.awaiting_announcement(c6)
    n_before = svc.originations
    svc.step(13)
    assert svc._last_orig[HUB] == 13
    assert svc.originations == n_before + 1, 'ONE LSA carries both'
    assert (svc.install_tick(c5), svc.install_tick(c6)) == (19, 19)
    for t in range(14, 19):
        svc.step(t)
        assert svc._pair_installed[c5] and svc._pair_installed[c6], t
    svc.step(19)
    assert not svc._pair_installed[c5] and not svc._pair_installed[c6]
    assert svc._unannounced == {} and svc._awaiting == {}


def test_the_origination_rule_is_per_router_in_the_source():
    """The per-transition scheduling may not come back beside the per-router
    one: `_schedule_install` only queues, and `_originate` is the one place
    that books an origination for billing."""
    src = _src('sixg_sim', 'link_state_service.py')
    body = src[src.index('    def _schedule_install'):
               src.index('    def _originate')]
    assert '_lsa_schedule' not in body and '_last_orig[' not in body
    orig = src[src.index('    def _originate'):
               src.index('    def install_tick')]
    assert orig.count('self._lsa_schedule.setdefault(tick, []).append(o)') == 1
    assert 'self._originate(tick)' in src


def test_a_stale_administrative_record_cannot_date_a_later_origination():
    """A Priority-2 relay REPURPOSES an existing hop, so both links sit on
    the same pair.  Tearing the relay down administratively therefore does
    NOT take the pair down -- the physical link still holds it up -- and the
    administrative record is never consumed by a transition.  If a later
    failure of the physical link then claimed that record, its Router-LSA
    would be dated to a tick that has already been billed and would never be
    charged at all.  The record is usable only by a transition in its OWN
    tick.

    Relay torn down at tick 5; Mesh_1 (the physical hop) fails at tick 20 and
    is confirmed at 23.  Both endpoints must originate at 23, and with
    ecc(N1) = 3 and ecc(N2) = 2 on the path,
        t_inst = max(23 + 3, 23 + 2) + 5 = 31.
    """
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    pair = _pair('O-DU_1', 'O-DU_2')

    lid = sim.transport_relay_model.create_link('O-DU_1', 'O-DU_2',
                                                30.0, 15.0)
    sim.topology.links[lid] = _relay(lid, 'O-DU_1', 'O-DU_2')
    sim.topology.sync_pair('O-DU_1', 'O-DU_2')
    for t in range(1, 5):
        svc.step(t)
    assert svc._link_belief[lid] is True
    assert svc._last_orig == {}, 'the pair never changed state'

    sim.current_tick = 5
    sim._remove_relay_link(lid, admin_node='O-DU_1')
    svc.step(5)
    assert svc._pair_confirmed[pair] is True, 'the physical hop holds it up'
    svc.step(6)
    assert svc._pair_admin == {}, 'an unconsumed record outlived its tick'

    _set_link_up(sim, 'Mesh_1', False)
    for t in range(20, 24):
        svc.step(t)
    assert svc._pair_confirmed[pair] is False
    assert svc._last_orig['O-DU_1'] == 23
    assert svc._last_orig['O-DU_2'] == 23
    assert svc._pending_install[pair] == (False, 31)
    # the LSAs are billed at 23, not lost to a tick that has gone by
    assert svc.tick_messages(23)['lsa_msgs'] > 0


def test_a_transition_that_reverts_before_installation_is_cancelled():
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    _set_link_up(sim, 'Mesh_1', False)
    cut = _pair('O-DU_1', 'O-DU_2')
    for t in range(1, 5):
        svc.step(t)
    assert svc._pair_confirmed[cut] is False and cut in svc._pending_install
    _set_link_up(sim, 'Mesh_1', True)
    for t in range(5, 9):
        svc.step(t)
    assert svc._pair_confirmed[cut] is True
    assert cut not in svc._pending_install
    assert svc._pair_installed[cut] is True


def test_installation_off_is_the_idealised_timing_sensitivity():
    """RERUN_SPEC WP3's pre-registered evaluation-only sensitivity
    (MARL_LS_INSTALL=0): detection and BILLING are unchanged, only the effect
    is immediate."""
    sim = _chain_sim()
    svc = _service(sim, install=False)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    ring = _pair('O-DU_0', 'O-DU_4')
    for t in range(1, 5):
        svc.step(t)
    assert svc._pair_confirmed[ring] is True
    assert svc._pair_installed[ring] is True          # no flood/SPF wait
    assert ring not in svc._pending_install
    # ... but the Router-LSAs were still originated, so the bill is the same
    assert svc._last_orig['O-DU_0'] == 4
    assert svc.last_messages['lsa_msgs'] > 0


# ----------------------------------------------------------------------
# 4. tombstones
# ----------------------------------------------------------------------

def _tombstone_sim():
    """UE -- A == B -- UE2, where A==B is a TRANSPORT_RELAY, plus a LONGER
    physical detour A -- C -- B.

    The detour matters.  The engine gates a UE-to-UE flow on PHYSICAL
    infrastructure reachability before it ever selects a path, so a flow
    whose ONLY path is the dead hop is refused before consume_path sees it
    and the tombstone branch is never exercised.  With the detour the flow
    is admitted, the believed graph's shortest path is still the two-hop
    dead one, and the tombstone is what actually refuses the bytes -- which
    is the real case: an uninstalled removal blackholes traffic a converged
    network would have carried the long way round."""
    t = Topology()
    for nid, ntype in ((A, NodeType.O_DU), (B, NodeType.RELAY),
                       (C, NodeType.EDGEUPF), (UE, NodeType.UE),
                       (UE2, NodeType.UE)):
        t.add_node(Node(id=nid, node_type=ntype))
    t.add_link(_uu('Uu_1', UE, A))
    t.add_link(_uu('Uu_2', UE2, B))
    t.add_link(_link('Mesh_AC', A, C, cap=500.0))
    t.add_link(_link('Mesh_CB', C, B, cap=500.0))
    sim = _simulator(t)
    lid = sim.transport_relay_model.create_link(A, B, 50.0, 15.0)
    sim.topology.links[lid] = _relay(lid, A, B, cap=50.0)
    sim.topology.sync_pair(A, B)
    sim.island_mode = True
    sim.marl_ue_routing_enabled = True
    return sim, lid


def _assert_stale_ids_are_physically_down(sim, svc, where=''):
    """The contract `stale_link_ids()` states and that tick_step and
    consume_path rely on: an id in it is in the forwarding view ALTHOUGH the
    Link is physically DOWN.  A physically-UP id in that set means the view
    named a live link as a hop it is refusing to carry -- or, worse, named a
    live link the service does not believe up, which forwards at full
    capacity (the free-adoption bypass).  Asserted on every forwarding test
    through `_forward_on_the_view`, and per tick in the integration run."""
    for lid in svc.stale_link_ids():
        lk = sim.topology.links.get(lid)
        assert lk is None or not bool(getattr(lk, 'is_up', True)), (
            'stale_link_ids() named the physically-UP link %s %s'
            % (lid, where))


def _assert_the_carrier_is_stamped(sim, svc, where=''):
    """D3.23 and D3.30.  The service's own "is this link in my forwarding
    view?" API may not contradict the view it publishes, in EITHER direction,
    so this asserts the stamp of every carrier the view names -- including the
    physically-DOWN ones, which the first version of this helper skipped and
    was therefore structurally blind to.

    The contract (`installed_since`, docs/LINK_STATE_SERVICE.md section 4):

      * a carrier that is physically UP really carries bits, so it is stamped
        (D3.23).  WP4 counts a bridge's teardown grace with exactly that
        stamp, so a carrier without one is a relay that has been carrying its
        pair since creation and is about to be booked as never installed and
        never used;
      * a carrier that is physically DOWN -- the hop BLACKHOLES -- is stamped
        iff the service still BELIEVES it up and the Link is still there
        (D3.30).  That is the undetected-failure case: belief may not consult
        truth the service has not detected, so the link stays in the service's
        own view for the detection window while the flow delivers 0.0.  Only a
        hop with nothing believed up (a tombstone, or the physically-down
        `_down_representative`) is genuinely unstamped.

    It also asserts D3.33 -- the two copies of one stamp agree, in both
    directions: `installed_since(lid) is None` implies
    `lid not in sim._relay_route_since`."""
    G = svc.believed_graph()
    if G is None:
        G = sim.topology.graph
    for u, v, data in G.edges(data=True):
        lid = data.get('link_id')
        if lid is None:
            continue
        lk = sim.topology.links.get(lid)
        if lk is not None and bool(getattr(lk, 'is_up', True)):
            want, why = True, 'the carrier is physically up (D3.23)'
        else:
            want = lk is not None and bool(svc._link_belief.get(lid, False))
            why = ('the blackholing carrier is %sbelieved up (D3.30)'
                   % ('' if want else 'not '))
        assert bool(svc.visible.get(lid)) is want, (
            'the view names %s as the carrier of %s-%s and %s, but visible '
            'is %r %s' % (lid, u, v, why, svc.visible.get(lid), where))
        assert (svc.installed_since(lid) is not None) is want, (
            'the view names %s as the carrier of %s-%s and %s, but '
            'installed_since is %r %s'
            % (lid, u, v, why, svc.installed_since(lid), where))
    _rs = getattr(sim, '_relay_route_since', None)
    if isinstance(_rs, dict):
        for lid in _rs:
            assert svc.installed_since(lid) is not None, (
                'sim._relay_route_since holds %s but the service does not '
                'have it in the installed view %s' % (lid, where))


def _view_edge(sim, svc, u, v):
    """The (u, v) edge the arm actually forwards over: the believed graph's
    when the installed view differs from truth, the true graph's otherwise."""
    G = svc.believed_graph()
    if G is None:
        G = sim.topology.graph
    return dict(G[u][v]) if G.has_edge(u, v) else None


def _forward_on_the_view(sim, svc, src, dst):
    """Phase 7 of tick_step, in isolation: forward one UE-to-UE flow over the
    service's INSTALLED view and return the delivered volume."""
    _assert_stale_ids_are_physically_down(sim, svc)
    _assert_the_carrier_is_stamped(sim, svc)
    sim.ue_to_ue_flows = [(src, dst)]
    for lk in sim.topology.links.values():
        lk.reset_utilization()
    believed = svc.believed_graph()
    if believed is None:
        sim._forward_traffic(sim._generate_traffic())
    else:
        true_graph = sim.topology.graph
        sim._marl_stale_link_ids = svc.stale_link_ids()
        sim._marl_tombstone_link_ids = svc.tombstoned_link_ids()
        try:
            sim.topology.graph = believed
            sim._forward_traffic(sim._generate_traffic())
        finally:
            sim.topology.graph = true_graph
            sim._marl_stale_link_ids = frozenset()
            sim._marl_tombstone_link_ids = frozenset()
    return float(getattr(sim, 'last_tick_ue_to_ue_volume', 0.0))


def test_a_flow_over_a_tombstoned_hop_delivers_exactly_zero():
    """The hook WP1 left marked in consume_path.  The relay link is popped at
    tick 5 by its own endpoint A, so the pair is confirmed down AT ONCE (the
    administrative rule) -- but the removal still has to be flooded and
    installed, so until t_inst the believed graph STILL holds the hop, and
    it must carry nothing over it.
    """
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0     # control

    sim.current_tick = 5
    assert sim._remove_relay_link(lid, admin_node=A) is not None
    assert lid in sim._link_tombstones
    assert not sim.topology.graph.has_edge(A, B)             # true graph: gone
    svc.step(5)

    assert svc._pair_confirmed[_pair(A, B)] is False         # administrative
    assert svc._pair_installed[_pair(A, B)] is True          # not installed
    # A announced its own teardown at once; B learns of it only after its
    # BFD timeout, so the installation tick is not known yet (D13.1)
    assert svc._last_orig.get(A) == 5 and B not in svc._last_orig
    assert svc.awaiting_announcement(_pair(A, B))
    G = svc.believed_graph()
    assert G is not None and G.has_edge(A, B)
    assert G[A][B]['link_id'] == lid                         # the tombstone
    assert svc.tombstoned_link_ids() == frozenset({lid})

    delivered = _forward_on_the_view(sim, svc, UE, UE2)
    assert delivered == 0.0, (
        'the believed graph routed over a physically-down link')
    assert sim._diag_tombstone_hops >= 1
    assert sim._diag_tombstone_vol_tick > 0.0
    # ... and it is NOT a stale edge: stale_edges keeps meaning "defect"
    assert sim._diag_orphan_hops == 0
    assert sim._diag_stale_u2u_tick == 0.0
    assert sim._diag_stale_edges_tick == 0
    assert sim._diag_tombstone_edges_tick == 1
    # the flow gate still admitted the flow -- the loss is a blackhole, which
    # is what an uninstalled removal costs in the field
    assert svc.get_routable_flows() == [(UE, UE2)]
    # ... B announces at 5 + BFD_DETECT_TICKS, which dates the install ...
    for t in (6, 7, 8):
        svc.step(t)
    assert svc._last_orig.get(B) == 8
    t_inst = svc.install_tick(_pair(A, B))
    assert t_inst is not None and t_inst > 8
    # ... and once the removal IS installed, the detour carries the flow
    for t in range(9, t_inst + 1):
        svc.step(t)
    assert svc.believed_graph() is None
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0


def test_the_tombstone_is_purged_once_its_removal_is_installed():
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)
    sim.current_tick = 5
    sim._remove_relay_link(lid, admin_node=A)
    t = 5
    while svc.install_tick(_pair(A, B)) is None:    # until B has announced
        svc.step(t)
        assert lid in sim._link_tombstones, t
        assert svc._pair_installed[_pair(A, B)] is True, t
        t += 1
        assert t < 20, 'the removal was never announced by both ends'
    t_inst = svc.install_tick(_pair(A, B))
    for t in range(t, t_inst):
        svc.step(t)
        assert lid in sim._link_tombstones, t
        assert svc._pair_installed[_pair(A, B)] is True, t
    svc.step(t_inst)
    assert svc._pair_installed[_pair(A, B)] is False
    assert lid not in sim._link_tombstones
    assert svc.believed_graph() is None


def test_an_undetected_failure_blackholes_instead_of_rerouting():
    """The other half of the tombstone case, and the commoner one: the Link
    object still EXISTS and has physically failed, and the service has not
    detected it yet.  It is still the hop, it delivers 0.0, and the
    alternate-path retry -- which exists to work around SATURATION -- must
    NOT reroute over the detour, because no control plane could have
    installed that detour yet.  Rerouting here would hand back exactly the
    omniscience the detection floor removes.
    """
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0     # control

    _set_link_up(sim, lid, False)                # failed, NOT torn down
    for t in (5, 6, 7):
        svc.step(t)
        assert svc.visible[lid] is True, t       # still believed up
        assert svc.stale_link_ids() == frozenset({lid}), t
        assert svc.tombstoned_link_ids() == frozenset(), t
        G = svc.believed_graph()
        assert G is not None and G[A][B]['link_id'] == lid
        assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0, t
        assert sim._diag_orphan_hops == 0 and sim._diag_stale_edges_tick == 0
    # BFD declares the session down at tick 8 -- but the DETOUR is still
    # not usable, because the failure has not been flooded and installed
    # yet.  The hop stays in the installed view, now naming a link nothing
    # believes up, and it keeps blackholing.  That gap is the whole point
    # of modelling installation: detection is not convergence.
    svc.step(8)
    assert svc.visible[lid] is False
    assert svc.stale_link_ids() == frozenset({lid})
    assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0
    t_inst = svc._pending_install[_pair(A, B)][1]
    assert t_inst > 8
    for t in range(9, t_inst + 1):
        svc.step(t)
    assert svc._pair_installed[_pair(A, B)] is False
    assert svc.stale_link_ids() == frozenset()
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0   # the detour


def test_a_relay_recreated_before_the_removal_installs_carries_nothing():
    """THE CASE THE FIRST WP3 REVIEW FOUND (D3.16).  A relay is torn down
    administratively at tick 5, so the pair is confirmed down at once but its
    removal is not installed for several more ticks -- and a NEW relay is
    created on the SAME pair at tick 6, before that.

    The new link is physically up, believed DOWN (no free adoption after
    island onset), never installed, and `installed_since` is None for it.
    So the hop must still name the tombstone of the link whose removal is
    pending and deliver exactly 0.0.  Naming the live new link instead --
    which `_down_representative` did while its predicate accepted any Link of
    the pair -- forwarded traffic over a brand-new link id at full capacity
    for its whole BFD bring-up, the bypass this work package exists to close.
    """
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0     # control
    pair = _pair(A, B)

    sim.current_tick = 5
    assert sim._remove_relay_link(lid, admin_node=A) is not None
    svc.step(5)
    assert svc._pair_confirmed[pair] is False       # administrative, at once
    assert svc._pair_installed[pair] is True        # ... but not installed
    # ... and not even datable yet: the remote end B announces the removal
    # only after its BFD timeout, at 8 (D13.1)
    assert svc.awaiting_announcement(pair)

    # a SECOND relay on the same pair, one tick after the first was popped
    sim.current_tick = 6
    lid2 = sim.transport_relay_model.create_link(A, B, 40.0, 15.0)
    assert lid2 != lid
    sim.topology.links[lid2] = _relay(lid2, A, B, cap=40.0)
    sim.topology.sync_pair(A, B)
    assert sim.topology.graph[A][B]['link_id'] == lid2      # true graph: live

    for t in (6, 7, 8):                       # its whole BFD bring-up
        svc.step(t)
        assert svc._link_belief.get(lid2) is False, t
        assert svc.visible.get(lid2) is False, t
        assert svc.installed_since(lid2) is None, t
        _assert_stale_ids_are_physically_down(sim, svc, 'at tick %d' % t)
        G = svc.believed_graph()
        assert G is not None and G[A][B]['link_id'] == lid, (
            'tick %d: the hop named %s instead of the tombstone %s'
            % (t, G[A][B]['link_id'], lid))
        assert svc.tombstoned_link_ids() == frozenset({lid}), t
        assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0, (
            'tick %d: a link the service neither believes up nor has '
            'installed carried traffic' % t)
    t_inst = svc.install_tick(pair)
    assert t_inst is not None and t_inst > 9, t_inst   # B announced at 8

    # tick 9: the BFD session comes up, the pair is confirmed up again before
    # its removal ever installed, so the pending removal is cancelled and the
    # new relay becomes the hop -- with ITS capacity, not the old one's.
    svc.step(9)
    assert svc._link_belief.get(lid2) is True
    assert svc.installed_since(lid2) == 9
    assert svc._pending_install == {}
    assert lid not in sim._link_tombstones
    assert _view_edge(sim, svc, A, B)['link_id'] == lid2
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0


def test_the_view_never_offers_more_capacity_than_the_true_hop():
    """THE SECOND CASE THE REVIEW FOUND (D3.15).  A Priority-2 relay
    REPURPOSES a pair that already carries a live physical link, so for its
    whole BFD bring-up the relay is the TRUE representative of the hop while
    the service does not yet believe it up.

    `consume_path` reads `link.capacity` off whichever Link the hop names, so
    naming the believed (underlying) link there would forward at the
    pre-relay capacity -- 500 instead of 30 here -- over a hop that
    physically carries only the relay share.  Belief decides whether the hop
    EXISTS; physics decides which Link carries it.
    """
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    svc.step(0)
    u, v = 'O-DU_0', 'O-DU_1'                       # Mesh_0, capacity 500
    assert sim.topology.graph[u][v]['capacity'] == 500.0

    rid = sim.transport_relay_model.create_link(u, v, 30.0, 15.0)
    sim.topology.links[rid] = _relay(rid, u, v, cap=30.0)
    sim.topology.sync_pair(u, v)
    assert sim.topology.graph[u][v] == {'link_id': rid, 'capacity': 30.0}

    for t in (1, 2, 3):                             # the bring-up window
        svc.step(t)
        assert svc._link_belief.get(rid) is False, t     # not believed up
        edge = _view_edge(sim, svc, u, v)
        assert edge['link_id'] == rid, (t, edge)
        assert edge['capacity'] == 30.0, (
            'tick %d: the view offered %s over a hop that physically carries '
            '30.0' % (t, edge['capacity']))
        # D3.23: the stamp follows the link the view NAMES.  The relay is
        # carrying the hop from tick 1, so it is in the forwarding view from
        # tick 1 -- which is what WP4 counts its teardown grace with.
        assert svc.visible.get(rid) is True, t
        assert svc.installed_since(rid) == 1, t
        assert sim._relay_route_since.get(rid) == 1, t
    # the pair never stopped being a hop, so nothing about the view diverged
    assert svc._view_changes == []
    svc.step(4)
    assert svc._link_belief.get(rid) is True
    assert _view_edge(sim, svc, u, v)['capacity'] == 30.0
    # believing it up does not re-date a link that was already carrying it
    assert svc.installed_since(rid) == 1


def test_a_relay_created_during_an_undetected_failure_carries_nothing():
    """THE CASE THE SECOND WP3 REVIEW FOUND (D3.22).  The ordering D3.16 does
    NOT cover, because here the pair's own link never left `topology.links`:

      t=5  the relay carrying A==B fails PHYSICALLY (no administrative
           record, so BFD owes its full 3 ticks).  The service still believes
           it up, so the hop stays and blackholes -- correct.
      t=6  a NEW relay is created on the SAME pair, inside that window.

    The true representative is now the new link, and substituting it for the
    believed one (D3.15, unguarded) resolved the hop onto a Link with
    `_link_belief` False, `visible` False and `installed_since` None, dropped
    the failed id from `stale_link_ids()` and delivered at full capacity --
    cancelling the undetected-failure blackhole and adopting a brand-new link
    id for free, which is what RERUN_SPEC WP3 forbids twice ("a new link's
    adjacency comes up after the BFD session, 3 ticks"; "after island onset a
    never-seen link id starts DOWN -- no free adoption").

    The substitution may only ever REDUCE what a hop offers, so it applies
    only while the BELIEVED representative is itself physically up.
    """
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0     # control
    pair = _pair(A, B)

    # a PHYSICAL failure: the Link object stays, so there is no
    # administrative shortcut and BFD owes the full detect multiplier
    sim.current_tick = 5
    _set_link_up(sim, lid, False)
    svc.step(5)
    assert svc._link_belief[lid] is True            # undetected, still
    assert svc._pair_confirmed[pair] is True
    assert svc.stale_link_ids() == frozenset({lid})
    assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0    # the blackhole

    # a SECOND relay on the same pair, inside the detection window
    sim.current_tick = 6
    lid2 = sim.transport_relay_model.create_link(A, B, 40.0, 15.0)
    assert lid2 != lid
    sim.topology.links[lid2] = _relay(lid2, A, B, cap=40.0)
    sim.topology.sync_pair(A, B)
    assert sim.topology.graph[A][B]['link_id'] == lid2       # true graph: live

    for t in (6, 7):                    # the rest of the detection window
        svc.step(t)
        assert svc._link_belief.get(lid2) is False, t
        assert svc.visible.get(lid2) is False, t
        assert svc.installed_since(lid2) is None, t
        assert lid2 not in getattr(sim, '_relay_route_since', {}), t
        edge = _view_edge(sim, svc, A, B)
        assert edge['link_id'] == lid, (
            'tick %d: the hop named the unadopted new link %s instead of the '
            'failed one it still believes up' % (t, edge['link_id']))
        assert lid in svc.stale_link_ids(), (
            'tick %d: the failed id left stale_link_ids(), so consume_path '
            'never sees the blackhole' % t)
        assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0, (
            'tick %d: a link with no completed BFD session and no installed '
            'Router-LSA carried traffic' % t)

    # t=8: BFD confirms the failure.  Nothing of the pair is believed up, so
    # the hop takes its physically-DOWN representative (D3.16) and still
    # blackholes -- the new link is STILL not adopted.
    svc.step(8)
    assert svc._link_belief[lid] is False
    assert svc._link_belief.get(lid2) is False
    assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0

    # t=9: the new link's own BFD session completes.  NOW it is the hop, with
    # its own capacity -- three ticks after it appeared, as specified.
    svc.step(9)
    assert svc._link_belief.get(lid2) is True
    assert svc.installed_since(lid2) == 9
    assert _view_edge(sim, svc, A, B)['link_id'] == lid2
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0


def test_the_undetected_failure_carrier_keeps_its_stamp():
    """D3.30.  THE CASE THE THIRD REVIEW FOUND.  The D3.23 contract said in
    four places that "a hop that blackholes carries nothing and is not
    stamped".  That is true of the tombstone / `_down_representative` case
    only.  In the UNDETECTED-failure case the service still believes the
    carrier up, so the belief-side half of `visible` stamps it and the hop
    keeps `installed_since` and its `_relay_route_since` mirror for the whole
    detection window while delivering exactly 0.0.

    That is the RIGHT behaviour and the wrong documentation: making the stamp
    skip a link because it has physically failed would let `visible` -- a
    belief-side API -- consult truth the service has not detected, which is
    the omniscience the detection floor exists to remove.  So the contract now
    says what the code does, and this test pins it in both directions.

    Its consequence for WP4 is stated in `installed_since`'s docstring and in
    docs/LINK_STATE_SERVICE.md section 4: `_relay_link_passes` counts
    forwarding passes in which the link is INSTALLED IN THE VIEW (RERUN_SPEC
    WP4's own words), not passes in which it delivered, so a bridge that fails
    undetected accrues passes and is not booked as an idle-bridge teardown.
    """
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)
    assert svc.installed_since(lid) == 0
    assert sim._relay_route_since.get(lid) == 0
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0      # control

    _set_link_up(sim, lid, False)               # failed, NOT torn down
    for t in (5, 6, 7):                         # the detection window
        svc.step(t)
        # the hop blackholes ...
        assert svc.stale_link_ids() == frozenset({lid}), t
        assert _view_edge(sim, svc, A, B)['link_id'] == lid, t
        assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0, t
        # ... and the carrier is STILL in the service's own forwarding view,
        # because the service still believes it up
        assert svc._link_belief[lid] is True, t
        assert svc.visible[lid] is True, t
        assert svc.installed_since(lid) == 0, t
        assert sim._relay_route_since.get(lid) == 0, t

    # BFD confirms the failure: NOW nothing of the pair is believed up, the
    # hop takes its physically-down representative, and both copies of the
    # stamp clear together.
    svc.step(8)
    assert svc._link_belief[lid] is False
    assert svc.visible[lid] is False
    assert svc.installed_since(lid) is None
    assert lid not in sim._relay_route_since
    assert _forward_on_the_view(sim, svc, UE, UE2) == 0.0


def test_the_engine_mirror_follows_the_view_out_as_well_as_in():
    """D3.33.  `Simulator._relay_route_since` is the engine's copy of
    `installed_since`, and WP4 reads the ENGINE copy ("a protected link that
    is physically up but not yet installed is blocked", RERUN_SPEC WP4).  Two
    copies of one stamp that disagree are the stale-edge defect again, in the
    bookkeeping:

      * the exit branch popped `_installed_since` and left the engine copy, so
        every link that ever left the forwarding view looked to WP4 like a
        link installed since the tick it first entered, forever;
      * the write was applied to EVERY visible link id, so the relay-scoped
        dict filled with mesh and Uu access ids that `_remove_relay_link` --
        its only other writer -- will never pop.

    The invariant: `installed_since(lid) is None` implies
    `lid not in sim._relay_route_since`, and only relay links are ever in it.
    """
    sim, lid = _tombstone_sim()
    svc = _service(sim)
    svc.step(0)

    # only the RELAY is mirrored, although mesh and access links are visible
    assert set(sim._relay_route_since) == {lid}, sim._relay_route_since
    for other in ('Mesh_AC', 'Mesh_CB', 'Uu_1', 'Uu_2'):
        assert svc.visible.get(other) is True, other
        assert svc.installed_since(other) is not None, other
        assert other not in sim._relay_route_since, other

    # OUT of the view: a physical failure, detected at tick 8
    _set_link_up(sim, lid, False)
    for t in (5, 6, 7, 8):
        svc.step(t)
        assert (svc.installed_since(lid) is None) is (
            lid not in sim._relay_route_since), t
    assert svc.installed_since(lid) is None
    assert lid not in sim._relay_route_since

    # and BACK IN: the link recovers, its BFD session comes up again and the
    # pair's return is installed.  Both copies are rewritten together, and
    # with the CURRENT tick -- a re-entering link does not inherit tick 0.
    _set_link_up(sim, lid, True)
    back = None
    for t in range(9, 60):
        svc.step(t)
        assert (svc.installed_since(lid) is None) is (
            lid not in sim._relay_route_since), t
        if svc.installed_since(lid) is not None:
            back = t
            break
    assert back is not None and back > 8, back
    assert svc.installed_since(lid) == sim._relay_route_since[lid] == back


def test_a_parallel_believed_up_link_carries_when_the_representative_fails():
    """D3.31.  The D3.22 guard tested `topo._link_is_up(rep)` on the
    relay-preferred BELIEVED representative alone.  A pair carrying two links
    the service believes up, whose representative is the one that has
    physically failed, then blackholed for the whole detection window --
    although the service believed the PARALLEL link up and that link was
    physically carrying the pair in the true graph.  A hop refused on evidence
    the service does not have is the same error as a hop carried on evidence
    it does not have; the rule is the PAIR, not the representative.

    Latent on the evaluated topology (relay links go down administratively,
    where belief drops at once, or with their node, which downs both links of
    the pair; the static topology builds one link per infrastructure pair), so
    closing it moves no series -- but nothing pinned it, which is why it
    needed rediscovering.
    """
    sim, lid = _tombstone_sim()
    # a parallel mesh link on the SAME pair, present from the first step so
    # the service adopts both with the pre-converged initial topology
    sim.topology.links['Mesh_AB'] = _link('Mesh_AB', A, B, cap=500.0)
    sim.topology.sync_pair(A, B)
    svc = _service(sim)
    svc.step(0)
    assert svc._link_belief[lid] is True
    assert svc._link_belief['Mesh_AB'] is True
    # relay-first: the RELAY represents the pair in belief and in truth
    assert _view_edge(sim, svc, A, B)['link_id'] == lid

    # the representative fails PHYSICALLY: no administrative record, so BFD
    # owes its full 3 ticks and the service still believes it up
    _set_link_up(sim, lid, False)
    for t in (5, 6, 7):
        svc.step(t)
        assert svc._link_belief[lid] is True, t          # undetected
        assert svc._link_belief['Mesh_AB'] is True, t
        edge = _view_edge(sim, svc, A, B)
        assert edge['link_id'] == 'Mesh_AB', (
            'tick %d: the hop named %s, so it blackholed although the service '
            'believes the parallel link up and that link is carrying the pair'
            % (t, edge['link_id']))
        assert edge['capacity'] == 500.0, (t, edge)
        assert lid not in svc.stale_link_ids(), t
        assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0, t
    # the failed link is still BELIEVED up, so it keeps its own stamp (D3.30)
    assert svc.visible[lid] is True
    assert svc.installed_since(lid) == 0

    # tick 8: BFD confirms the failure.  Mesh_AB is now both the believed and
    # the true representative and nothing changes for the flow.
    svc.step(8)
    assert svc._link_belief[lid] is False
    assert svc.visible[lid] is False
    assert lid not in sim._relay_route_since
    assert _view_edge(sim, svc, A, B)['link_id'] == 'Mesh_AB'
    assert _forward_on_the_view(sim, svc, UE, UE2) > 0.0


def test_count_stale_edges_is_unchanged_on_the_true_graph():
    """The tombstone exclusion is scoped to the believed-graph swap, so a
    tombstoned id that appears on the TRUE graph is still the WP1 orphan."""
    sim, lid = _tombstone_sim()
    sim.current_tick = 5
    sim._remove_relay_link(lid, admin_node=A)
    sim.topology.graph.add_edge(A, B, link_id=lid, capacity=50.0)   # orphan
    assert sim.count_stale_edges() == 1
    assert sim._diag_tombstone_edges_tick == 0


# ----------------------------------------------------------------------
# 5. billing
# ----------------------------------------------------------------------

def test_pre_event_billing_is_exactly_zero_for_every_arm():
    """RERUN_SPEC WP3: billing runs "from island onset only, with pre-event
    billing exactly zero for all arms"."""
    R = _harness()
    sim = _chain_sim()
    # (a) the cross-layer arms, through the service itself
    svc = _service(sim)
    sim.island_mode = False
    for t in range(0, 6):
        svc.step(t)
        bill = svc.tick_messages(t)
        assert bill['bytes'] == 0 and bill['msgs'] == 0, (t, bill)
        for typ in LS.MESSAGE_TYPES:
            assert bill[typ + '_bytes'] == 0, (t, typ)
    # (b) every arm, through the harness's own accounting
    sim.link_state_service = svc
    for mode in ('marl', 'ospf', 'sdn', 'olsr', 'batman', 'aodv'):
        cps = R.ControlPlaneMessageSim(sim, mode)
        for t in range(0, min(6, R.SEVERANCE_TICK)):
            assert cps.compute_tick_messages(t) == (0, 0), (mode, t)


def test_the_phantom_floor_and_the_change_detection_proxy_are_gone():
    """IMPL_FACTS 5: 13 surviving agent-less sites were billed a 200 B frame
    EVERY tick (a constant 0.0208 Mb/s floor, 52 % of the reported MARL-RIC
    figure), any link flip billed all 35 survivors, and a 2 % draw billed the
    rest.  None of those rules may survive anywhere in _sim_marl."""
    har = _src('run_timeline_comparison.py')
    body = har[har.index('    def _sim_marl('):har.index('    def _sim_ospf(')]
    # past the docstring: the prose explains what was removed, the CODE
    # is what must not still do it
    code = body[body.index(chr(34) * 3, body.index(chr(34) * 3) + 3) + 3:]
    for gone in ('prev_state', 'topo_changed', 'rng.random()',
                 'MARL_POSTCARD_BYTES', '_get_node_state_snapshot',
                 'random.Random('):
        assert gone not in code, gone
    assert 'tick_messages(tick)' in code


def test_the_bill_is_bfd_plus_hellos_plus_lsas_and_nothing_else():
    """Hand-computed on the 5-node path with the core cut: 4 infrastructure
    adjacency pairs, so BFD is 4 x 2 packets x 66 B = 528 B per tick."""
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    # hellos: one broadcast per router at its own crc32 offset, 44 + 4 x
    # infra-neighbour count.  Over one full HelloInterval every router sends
    # exactly once.  Degrees on the path are 1, 2, 2, 2, 1.
    hello_msgs = hello_bytes = 0
    bill = None
    for t in range(0, LS.HELLO_INTERVAL):
        svc.step(t)
        b = svc.tick_messages(t)
        if bill is None:
            bill = b
        assert b['bfd_msgs'] == 8
        assert b['bfd_bytes'] == 8 * 66 == 528
        assert b['lsa_msgs'] == 0          # nothing has transitioned
        assert b['postcard_msgs'] == 0     # no site sends a heartbeat here
        assert b['bytes'] == (b['bfd_bytes'] + b['hello_bytes']
                              + b['lsa_bytes'] + b['postcard_bytes'])
        hello_msgs += b['hello_msgs']
        hello_bytes += b['hello_bytes']
    assert hello_msgs == 5
    assert hello_bytes == 5 * 44 + 4 * 8 == 252


def test_bfd_is_billed_while_a_session_is_still_coming_up():
    """D3.24.  A BFD session sends 1 packet/s per direction FROM THE MOMENT IT
    STARTS -- which is precisely what makes the 3-tick bring-up cost what it
    costs.  Billing only pairs already CONFIRMED up let every new relay
    bridge run its whole session for free: 3 x 2 x 66 B = 396 B per link, off
    the bill of the arms the bring-up delay is timing."""
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    assert svc.tick_messages(0)['bfd_msgs'] == 8          # 4 confirmed pairs

    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    ring = _pair('O-DU_0', 'O-DU_4')
    extra = 0
    for t in (1, 2, 3):                                   # the session itself
        svc.step(t)
        assert svc._pair_confirmed[ring] is False, t      # not up yet
        assert 'Mesh_ring' in svc._pending_link, t        # but running
        b = svc.tick_messages(t)
        assert b['bfd_msgs'] == 10, (t, b['bfd_msgs'])
        extra += b['bfd_bytes'] - 528
    assert extra == 3 * LS.BFD_PACKETS_PER_PAIR_PER_TICK * LS.BFD_BYTES == 396
    svc.step(4)                                           # confirmed up
    assert svc._pair_confirmed[ring] is True
    assert svc.tick_messages(4)['bfd_msgs'] == 10
    # a pair with neither a confirmed adjacency nor a session runs no BFD
    assert svc._pair_confirmed[_pair(CORE, 'O-DU_0')] is False
    assert 'Core_0' not in svc._pending_link


def test_a_router_lsa_floods_its_component_once_per_directed_adjacency():
    """One message per DIRECTED believed-up infra adjacency in the
    originator's component, each 24 + 12 x (originator infra degree) bytes.

    On the path with the core cut there are 4 pairs = 8 directed adjacencies.
    The ring link's two originators are N0 and N4, degree 1 each at the
    origination tick (the ring adjacency is confirmed on that same tick, so
    each has degree 2), so the sizes are hand-checked against the service's
    own believed degree rather than assumed.
    """
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    _add_link(sim, 'Mesh_ring', 'O-DU_0', 'O-DU_4')
    for t in range(1, 4):
        svc.step(t)
        assert svc.tick_messages(t)['lsa_msgs'] == 0, t
    svc.step(4)                                    # td: both ends originate
    bill = svc.tick_messages(4)
    deg = svc._infra_neighbours()
    # 5 pairs are believed up now (the path's 4 plus the ring)
    assert bill['lsa_msgs'] == 2 * 10
    assert bill['lsa_bytes'] == (10 * (24 + 12 * deg['O-DU_0'])
                                 + 10 * (24 + 12 * deg['O-DU_4']))
    assert svc.tick_messages(5)['lsa_msgs'] == 0   # no refresh, no LSAck


def test_the_hello_model_states_where_it_differs_from_the_baseline():
    """D3.25.  RERUN_SPEC WP3 says hellos are billed "as `_sim_ospf` (same
    interval and crc32 offset)", and the two are NOT identical: the baseline
    drops to OSPF_HELLO_FAST_INTERVAL = 1 while its measured control plane is
    converging, and re-phases each node's crc32 offset on that interval,
    while the service has no convergence state and bills HELLO_INTERVAL = 10
    at its steady-state phase throughout.  That is a defensible modelling
    choice and an indefensible silence, so the asymmetry is disclosed with
    the others rather than left inside a constant."""
    R = _harness()
    assert R.OSPF_HELLO_FAST_INTERVAL == 1 != LS.HELLO_INTERVAL
    svc_src = _src('sixg_sim', 'link_state_service.py')
    doc = _src('docs', 'LINK_STATE_SERVICE.md')
    omissions = doc[doc.index('## 6.'):doc.index('## 7.')]
    for text in (svc_src, omissions):
        low = text.lower()
        assert 'fast-hello' in low or 'fast hello' in low
        assert 'convergence' in low
    # and the service's BILL really does use the steady-state interval only
    bill = svc_src[svc_src.index('    def _bill('):
                   svc_src.index('    # helpers')]
    assert 'HELLO_INTERVAL' in bill
    assert 'FAST' not in bill and 'converg' not in bill


def test_the_billing_table_describes_the_rule_the_service_applies():
    """The pre-registration reads docs/LINK_STATE_SERVICE.md section 5, so
    the BFD row there must state the rule `_bill` applies -- including the
    session in progress D3.24 added."""
    doc = _src('docs', 'LINK_STATE_SERVICE.md')
    billing = doc[doc.index('## 5.'):doc.index('## 6.')]
    assert 'session' in billing.lower()
    bfd_row = [ln for ln in billing.splitlines() if ln.startswith('| BFD')]
    assert len(bfd_row) == 1, billing
    assert str(LS.BFD_BYTES) in bfd_row[0]
    assert 'confirmed' in bfd_row[0] and 'progress' in bfd_row[0]


def test_the_bfd_rule_discloses_that_it_departs_from_the_spec():
    """D3.34.  D3.24 changed the BFD billing rule from the BINDING
    specification's "per believed-up infrastructure adjacency pair per tick"
    to "per pair running a session" and recorded no divergence anywhere.  The
    pre-registration is generated from RERUN_SPEC, and the change moves
    `overhead` (descriptive) AND `energy` (comparative) for every MARL-path
    arm, so the departure has to be visible at every place a reader of the
    rule arrives: the spec sentence it contradicts is still there, unedited,
    and the three documents that describe the implemented rule all name it.

    This is the discipline D3.25 already applies to the hello interval; the
    BFD row was the one that silently diverged."""
    spec = _src('docs', 'RERUN_SPEC.md')
    # RERUN_SPEC is binding and is NOT edited: the sentence must still be
    # there, which is exactly why the divergence has to be recorded.
    assert 'believed-up infrastructure adjacency pair per tick' in spec
    svc_src = _src('sixg_sim', 'link_state_service.py')
    doc = _src('docs', 'LINK_STATE_SERVICE.md')
    billing = doc[doc.index('## 5.'):doc.index('## 6.')]
    decisions = _src('docs', 'DECISIONS.md')
    d324 = decisions[decisions.index('**D3.24'):decisions.index('**D3.25')]
    for where, text in (('link_state_service.py', svc_src),
                        ('LINK_STATE_SERVICE.md section 5', billing),
                        ('DECISIONS.md D3.24', d324)):
        low = text.lower()
        assert 'depart' in low, where
        assert 'rerun_spec' in low, where
        assert 'believed-up' in low, where
        assert 'rerun_map' in low or 'rerun_details' in low, where
    # and the consequence for the pre-registration is stated, not implied
    for text in (billing, d324):
        assert 'pre-registration' in text.lower()
        assert 'energy' in text.lower()


def test_the_energy_weights_are_the_ospf_per_message_cpu_costs():
    sim = _chain_sim()
    svc = _service(sim)
    sim.island_mode = True
    _set_link_up(sim, 'Core_0', False)
    svc.step(0)
    bill = svc.tick_messages(0)
    expect = ((LS.CPU_MJ_BFD * bill['bfd_msgs']
               + LS.CPU_MJ_HELLO * bill['hello_msgs']
               + LS.CPU_MJ_LSA * bill['lsa_msgs']
               + LS.CPU_MJ_POSTCARD * bill['postcard_msgs']) / bill['msgs'])
    assert abs(bill['cpu_mj_per_msg'] - expect) < 1e-12
    assert LS.CPU_MJ_BFD == LS.CPU_MJ_HELLO == 0.5
    assert LS.CPU_MJ_LSA == 2.0
    assert LS.CPU_MJ_POSTCARD == 0.3


def test_overhead_is_exported_descriptively():
    """RERUN_SPEC WP3: "Overhead is reported DESCRIPTIVELY, never as a
    comparative KPI - make sure nothing in the code or the exported keys
    implies otherwise"."""
    har = _src('run_timeline_comparison.py')
    svc_src = _src('sixg_sim', 'link_state_service.py')
    # the keys are neutral names for message types, not verdicts
    for key in ('oh_bfd', 'oh_hello', 'oh_lsa', 'oh_postcard'):
        assert "'" + key + "'" in har or '_OH_TYPES' in har, key
    assert 'DESCRIPTIVE' in svc_src and 'comparative KPI' in svc_src
    assert 'DESCRIPTIVE' in har or 'descriptive' in har
    # nothing names an overhead winner, ratio or ranking
    for banned in ('overhead_win', 'overhead_ratio', 'overhead_rank',
                   'best_overhead', 'overhead_advantage'):
        assert banned not in har, banned
        assert banned not in svc_src, banned
    # and the figure legend describes what is actually billed on the MARL
    # path.  The postcard component was the WP5 hook at WP3 and is now the
    # DCC ledger's 60-B frames (WP5), so a legend that calls the column
    # "postcards, 200B/msg" is a false description of the billing model.
    for stale in ('postcards, 200B/msg', 'postcards, same DCC'):
        assert stale not in har, stale
    # D3.26: and no baseline may justify ITS OWN billing rule by pointing at
    # the MARL postcard model WP3 deleted (the 200-B change-detection proxy;
    # the WP5 heartbeat is billed from the DCC ledger, not by _sim_marl), so
    # "billed exactly as the MARL postcard is" documents a model that does
    # not exist.  The phrases
    # are spelled with a placeholder so this scan cannot match itself -- the
    # trap the first run of the legend check fell into.
    marl_pc = 'MARL' + ' postcard'
    for stale in (marl_pc, 'delta-based postcards'):
        assert stale not in har, stale


def test_the_overhead_legend_quotes_the_sizes_its_own_models_bill():
    """D3.26.  The overhead panel is a publication figure, and five of its
    eight legend entries quoted byte figures their own `_sim_*` branch does
    not use: OSPF "Hello 64B + LSA 128B" against OSPF_HELLO_BASE = 44 and
    OSPF_LSA_BASE = 24, OLSR "40B/80B" against 60 and 60, SDN "LLDP 64B"
    against SDN_LLDP_BYTES = 120, AODV "RREQ 48B + RREP 44B" against 24 and
    20.  Every quoted number is now read from the constant, and this test
    checks it against the constant AND checks that the constant is the one
    the corresponding branch reads."""
    import re
    R = _harness()
    labels = R.overhead_legend_labels()
    C = R.ControlPlaneMessageSim
    expect = {
        'ospf': (('OSPF_HELLO_BASE', 'OSPF_LSA_BASE'), '_sim_ospf'),
        'sdn': (('SDN_FLOWMOD_BYTES', 'SDN_LLDP_BYTES'), '_sim_sdn'),
        'olsr': (('OLSR_HELLO_BASE', 'OLSR_TC_BASE'), '_sim_olsr'),
        'batman': (('BATMAN_OGM_BYTES',), '_sim_batman'),
        'aodv': (('AODV_RREQ_BYTES', 'AODV_RREP_BYTES'), '_sim_aodv'),
    }
    har = _src('run_timeline_comparison.py')
    order = ['_sim_marl', '_sim_ospf', '_sim_sdn', '_sim_olsr', '_sim_batman',
             '_sim_aodv']
    bodies = {}
    for name in order:
        # EACH branch ends at the next method OR at the end of
        # ControlPlaneMessageSim, never at the end of the file.  Bounding the
        # LAST entry by len(har) made the `_sim_aodv` slice ~5700 lines long
        # and swallowed `overhead_legend_labels()` itself, so the second half
        # of this check ("the constant is the one the branch reads") was
        # satisfied by the legend's own text and was vacuous for AODV.
        start = har.index('    def %s(' % name)
        ends = [x for x in (har.find('\n    def ', start + 1),
                            har.find('\nclass ', start + 1)) if x != -1]
        bodies[name] = har[start:min(ends)] if ends else har[start:]
    # the bound really is the class, not the file: the legend function is
    # defined far below and must not be inside any branch
    for name, body in bodies.items():
        assert 'def overhead_legend_labels' not in body, name
        assert len(body) < 20000, (name, len(body))
    for arm, (names, branch) in expect.items():
        quoted = [int(x) for x in re.findall(r'(\d+)B', labels[arm])]
        assert quoted == [getattr(C, n) for n in names], (
            '%s legend %r does not quote %s' % (arm, labels[arm], names))
        for n in names:
            assert n in bodies[branch], (arm, n, branch)
    # the MARL-path entries quote ONE size: the DCC heartbeat frame (WP5,
    # D5.4), read from the constant the DCC ledger bills per frame -- every
    # message the link-state service itself bills is variable-size
    from sixg_sim.postcards import POSTCARD_ENCODED_BYTES
    for arm in ('marl', 'marl_freeze', 'xldet'):
        assert [int(x) for x in re.findall(r'(\d+)B', labels[arm])] == [
            POSTCARD_ENCODED_BYTES], labels[arm]
        assert 'BFD' in labels[arm] and 'Router-LSA' in labels[arm]
        assert 'heartbeat' in labels[arm]
    # the figure really uses this dict, and holds no second copy of it
    assert 'labels_oh = overhead_legend_labels()' in har
    assert har.count('overhead_legend_labels()') == 2      # def + the one use


def test_an_all_zero_baseline_breakdown_cannot_be_read_as_a_decomposition():
    """D3.18.  Only the MARL code path fills `last_breakdown`: the five
    baseline models bill their own message types (OLSR TC, OGM, RREQ/RREP,
    OpenFlow/LLDP), which the four names bfd/hello/lsa/postcard do not
    describe.  Their oh_* series are therefore all-zero ALONGSIDE a non-zero
    'overhead', and a consumer must be able to tell "this protocol sent
    nothing" from "this protocol's model produces no breakdown"."""
    R = _harness()
    sim = _chain_sim()
    svc = _service(sim)
    sim.link_state_service = svc
    assert R.ControlPlaneMessageSim(sim, 'marl').breakdown_available is True
    for mode in ('ospf', 'sdn', 'olsr', 'batman', 'aodv'):
        cps = R.ControlPlaneMessageSim(sim, mode)
        assert cps.breakdown_available is False, mode
        # the situation the flag exists for: real bytes, empty breakdown
        total = 0
        bd_bytes = 0
        for t in range(R.SEVERANCE_TICK, R.SEVERANCE_TICK + 12):
            _m, b = cps.compute_tick_messages(t)
            total += b
            bd_bytes += sum(cps.last_breakdown[ty + '_bytes']
                            for ty in LS.MESSAGE_TYPES)
        assert bd_bytes == 0, mode
        # AODV is REACTIVE: with no _aodv_ctrl on this stub topology it has
        # no route discovery to bill, so only the four proactive models can
        # show the mismatch here.  On a real pass it shows it too (the WP3
        # review measured it on 236 of 600 recovery ticks).
        if mode != 'aodv':
            assert total > 0, mode
    har = _src('run_timeline_comparison.py')
    assert "'oh_breakdown_available'" in har
    assert 'breakdown_available' in har


# ----------------------------------------------------------------------
# 6. integration
# ----------------------------------------------------------------------

def test_integration_run_through_step_tick():
    """A full engine run over the path topology: the core link is cut at tick
    10, and the service must (a) keep the graph invariant, (b) never produce a
    stale edge, (c) bill exactly nothing before island onset and something
    after it, (d) raise the per-site isolation flag, and (e) leave belief
    equal to truth once everything has installed."""
    sim = _chain_sim()
    svc = LS.make_link_state_service(sim)
    assert svc is not None
    onset = None
    billed_pre = 0
    billed_post = 0
    for tick in range(0, 60):
        if tick == 10:
            _set_link_up(sim, 'Core_0', False)
        sim.step_tick(tick, None, forwarding=svc, where='wp3-integration')
        bill = svc.tick_messages(tick)
        if svc.island_onset_tick is None:
            billed_pre += bill['bytes']
        else:
            onset = onset if onset is not None else tick
            billed_post += bill['bytes']
        assert sim._diag_stale_edges_tick == 0, tick
        assert sim._diag_orphan_hops == 0, tick
        _assert_stale_ids_are_physically_down(sim, svc, 'at tick %d' % tick)
        _assert_the_carrier_is_stamped(sim, svc, 'at tick %d' % tick)
    assert billed_pre == 0, 'billed before island onset'
    assert billed_post > 0, 'billed nothing after island onset'
    assert onset is not None and onset >= 10
    # every site's component lost its core node
    for site in sim.agents:
        iso, since = svc.isolation_state(site)
        assert iso is True, site
        assert since is not None and since >= onset, site
    # and the view has converged back onto the truth
    assert svc.believed_graph() is None
    assert svc._pending_install == {}
    assert svc._awaiting == {} and svc._unannounced == {}
    assert svc.originations > 0
    assert sim._link_tombstones == {}


def test_the_service_is_stepped_by_both_drivers():
    har = _src('run_timeline_comparison.py')
    trn = _src('train_on_comparison.py')
    step = _src('sixg_sim', 'tick_step.py')
    assert har.count('make_link_state_service(sim)') == 1
    assert trn.count('make_link_state_service(sim)') == 2
    # only step_tick steps it, in both drivers.  The baseline controllers'
    # own .step(tick) calls live in the before_forward hook and are theirs.
    assert 'forwarding.step(tick)' in step
    for src in (har, trn):
        assert '_forwarding.step(' not in src
        assert 'link_state_service.step(' not in src
    # and phase 7 uses the service's own id sets
    assert 'forwarding.stale_link_ids()' in step
    assert 'forwarding.tombstoned_link_ids()' in step
