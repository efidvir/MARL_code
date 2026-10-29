"""WP4 -- bridge stability (flapping) and churn instrumentation (RERUN_SPEC WP4).

Fast checks on hand-built topologies, plus one real harness pass per seed:

  1. the constants, and the reason vocabularies the recording is written
     against;
  2. THE HEADLINE: the teardown grace counts USABLE PASSES, not ticks -- a
     bridge physically up for fifty ticks but installed for two passes is
     still held, and the same bridge with three passes is released;
  3. one test per hold reason (lockout, not_installed, grace, busy), and the
     PHYSICAL-STAMP fallback for an arm with no link-state service, which is
     what keeps SDN bit-identical;
  4. the churn counters: creations, teardowns by reason, bridge teardowns,
     IDLE-bridge teardowns counted separately (passes == 0), executed
     relay-mode flips, and the per-link lifetime records with their
     censoring;
  5. the decision digest, and the per-site postcard ledger (a silent site is
     on it with 0; its denominator is not derived from the senders);
  6. the pickle contract: every key the Track B analysis helpers read, with
     the name, the type and the length they expect
     (docs/TRACKB_HANDOFF.md section 1), cross-checked against series the
     WP4 code does not write (relay_count, relay_mode_nodes, postcards);
  7. an integration run on seeds 82 and 91 whose churn series are non-zero,
     and a Scenario B pass (severed at tick 0: the whole run is post-event).

The central claim also has an end-to-end form through the real teardown
site (`test_the_central_claim_through_the_teardown_site`), and
tools/wp4_negative_controls.py re-runs these unit tests against copies of
the engine with one WP4 rule reverted at a time (WP4-VERIFY D4.15).

Sections 6 and 7 run the real harness, so this file needs the evaluated
environment before the first sixg_sim import; `REQUIRES_ENV` makes
tests/run_tests.py re-run it in a child process that has it.

Runs under pytest and under tests/run_tests.py.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_ENV = {
    'MARL_TRANSPORT_POWER': 'planning', 'MARL_RELAY_STATE': 'reconcile',
    'MARL_RELAY_REPOINT': 'fragment_aware', 'MARL_POSTCARD_ALWAYS': '1',
    'MARL_REWARD_PROFILE': 'mission',
    'MARL_RELAY_ENTROPY_FLOOR': '0.01', 'MARL_STEER_MODEL': 'actionable',
}
REQUIRES_ENV = dict(EVAL_ENV)
if 'sixg_sim.agent' not in sys.modules:
    for _k, _v in EVAL_ENV.items():
        os.environ.setdefault(_k, _v)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                               InterfaceType)

TICKS = 300
SEEDS = (82, 91)
# The checkpoint the real passes load: MINTED from this tree (trackAgent
# merge).  WP6's strict schema gate refuses the tracked pre-WP6
# ce_ck_s1234/policy_final.pt; the recording these tests check does not
# depend on which weights the actor holds.
CKPT = os.path.join(ROOT, 'output', 'tmp_tests', 'wp4', 'policy_schema_current.pt')
CKPT_TORCH_SEED = 1234

# The per-tick series of the WP4 recording, and the type each Track B helper
# expects (docs/TRACKB_HANDOFF.md section 1; confirmatory_analysis constants
# EVENT_SERIES, DIAG_SERIES, LIFETIMES_KEY, REMOVED_BY_REASON_KEY, HOLDS_KEY,
# OVERHEAD_BY_TYPE_KEY, ENERGY_CTRL_KEY on branch trackB).  A key under
# another name is INVISIBLE to the analysis, so the names are pinned here.
EVENT_SERIES = ('relay_created', 'relay_removed', 'bridge_teardowns',
                'idle_bridge_teardowns', 'relay_mode_flips')
DICT_SERIES = ('relay_removed_by_reason', 'relay_teardown_holds')

A0, A1 = 'O-DU_0', 'O-DU_1'          # component A
B0, B1 = 'O-DU_2', 'O-DU_3'          # component B
CORE = 'AMF_1'
RID = 'TR_O-DU_1_O-DU_2_1'


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _link(lid, u, v, cap=500.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL, up=True):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=up)


def _two_fragment_sim():
    """Two two-node infrastructure fragments and nothing joining them, so a
    relay link between A1 and B0 is a genuine sole-path bridge."""
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    t = Topology()
    t.add_node(Node(id=CORE, node_type=NodeType.AMF))
    for nid in (A0, A1, B0, B1):
        t.add_node(Node(id=nid, node_type=NodeType.O_DU))
    t.add_link(_link('Mesh_A', A0, A1))
    t.add_link(_link('Mesh_B', B0, B1))
    scenario = Scenario(name='wp4', duration_ticks=400, events=[],
                        traffic_profiles={})
    sim = Simulator(t, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 0
    return sim


def _add_bridge(sim, rid=RID, up=True, cap=30.0, tick=None):
    """Create the sole-path relay link the way the engine does: the Link, the
    lifetime record, and the pair sync."""
    if tick is not None:
        sim.current_tick = tick
    sim.topology.links[rid] = _link(rid, A1, B0, cap=cap,
                                    lt=LinkType.TRANSPORT_RELAY,
                                    iface=InterfaceType.BACKHAUL, up=up)
    sim._record_relay_link_created(rid, sim.topology.links[rid])
    if up:
        sim.topology.sync_pair(A1, B0)
    sim.topology.invalidate_infrastructure_cache()
    return rid


def _with_service(sim):
    """Give the simulator a link-state service, which is what switches the
    grace over to usable passes.  A stand-in object is enough: the engine
    asks only whether one is attached (Simulator._relay_grace_in_passes)."""
    sim.link_state_service = object()
    return sim


def _installed(sim, rid, since=0, passes=0):
    sim._relay_route_since[rid] = since
    sim._relay_link_passes[rid] = passes


def _holds(sim):
    return dict(sim._relay_teardown_holds)


def _hold_delta(sim, before):
    return {r: sim._relay_teardown_holds[r] - before.get(r, 0)
            for r in sim.RELAY_HOLD_REASONS
            if sim._relay_teardown_holds[r] - before.get(r, 0)}


def _blocked(sim, node=A1):
    """(blocked?, {reason: holds added by this call})."""
    before = _holds(sim)
    out = sim._relay_teardown_blocked(node)
    return out, _hold_delta(sim, before)


# ----------------------------------------------------------------------
# 1. constants and vocabularies
# ----------------------------------------------------------------------
def test_the_constants_are_the_pre_registered_ones():
    from sixg_sim.simulation import Simulator as S
    assert S.RELAY_BRIDGE_GRACE_PASSES == 3
    assert S.RELAY_BRIDGE_BUSY_UTIL == 0.02
    # unchanged by WP4
    assert S.RELAY_REPOINT_TICKS == 3
    assert S.RELAY_BRIDGE_GRACE_TICKS == 3
    assert S.RELAY_TEARDOWN_REASONS == ('mode_change', 'repoint', 'unspecified')
    assert S.RELAY_HOLD_REASONS == ('lockout', 'not_installed', 'grace', 'busy')


def test_the_engine_teardown_sites_name_their_reason():
    """Both engine teardown sites pass a reason, so nothing real is booked as
    'unspecified' (which exists only for a caller that names none)."""
    with open(os.path.join(ROOT, 'sixg_sim', 'simulation.py'),
              encoding='utf-8') as fh:
        src = fh.read()
    assert src.count("reason='mode_change'") == 1
    assert src.count("reason='repoint'") == 1
    # and every removal still goes through the one helper (WP1)
    assert src.count('self._remove_relay_link(') == 2


# ----------------------------------------------------------------------
# 2. THE HEADLINE: grace in usable passes, not in ticks
# ----------------------------------------------------------------------
def test_grace_counts_installed_passes_and_not_ticks():
    """A bridge physically up for fifty ticks whose forwarding view has used
    it TWICE is still inside its grace; the tick rule would have released it
    forty-seven ticks ago.  This is the flapping defect (rerun_map stale_flap
    item 5): under the evaluation ordering the tick grace expired before the
    link was ever in the view, so the bridge was judged idle against load
    snapshots taken while it could not carry a byte."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    sim.current_tick = 50
    # the link has been up since tick 0 -- the TICK rule is long expired
    sim._relay_bridge_since[RID] = 0
    assert sim.current_tick - sim._relay_bridge_since[RID] > sim.RELAY_BRIDGE_GRACE_TICKS
    # ... but the arm's forwarding view has only used it twice
    _installed(sim, RID, since=45, passes=2)
    blocked, added = _blocked(sim)
    assert blocked is True
    assert added == {'grace': 1}
    # the third usable pass releases it (it is idle: no load snapshot)
    sim._relay_link_passes[RID] = 3
    blocked, added = _blocked(sim)
    assert blocked is False
    assert added == {}


def _register_bridge(sim, tick=0, cap=30.0):
    """A sole-path relay link A1-B0 made the way the Priority-1 formation
    branch makes one -- transport-model session, Link, lifetime record, pair
    sync -- already past its beam training, with A1 in CAPACITY_BOOST and
    holding it.  Returns the engine's own link id."""
    from sixg_sim.phy_mac_state import RelayMode
    sim.current_tick = tick
    rid = sim.transport_relay_model.create_link(A1, B0, cap, 20.0)
    sim.topology.links[rid] = _link(rid, A1, B0, cap=cap,
                                    lt=LinkType.TRANSPORT_RELAY,
                                    iface=InterfaceType.BACKHAUL, up=True)
    sim._record_relay_link_created(rid, sim.topology.links[rid])
    sim.topology.sync_pair(A1, B0)
    sim.topology.invalidate_infrastructure_cache()
    ps = sim.phy_mac_states[A1]
    ps.relay_mode = RelayMode.CAPACITY_BOOST
    ps.relay_link_active = True
    ps.relay_peer_node = B0
    return rid


def test_the_central_claim_through_the_teardown_site():
    """THE NEGATIVE CONTROL FOR WP4's CENTRAL CLAIM, end to end (WP4-VERIFY
    D4.15).  Nothing is hand-set on the pass counter: a bridge is physically
    up from tick 0 and bridging from tick 0, the forwarding view does not
    install it until tick 48, and its usable passes are accrued by the
    engine's own end-of-pass snapshot.  At tick 50 it has been physically up
    for FIFTY ticks and installed for TWO passes, and the site's policy asks
    to leave relay mode through the real teardown site (_step_phy_mac):

      * the teardown is REFUSED -- the link survives, the relay mode is held
        with it, and the hold is booked under 'grace';
      * one more usable pass later, the same request is GRANTED, and the
        teardown is booked as a mode_change bridge teardown that is NOT idle.

    Before installation the same request is refused under 'not_installed'.
    Under the pre-WP4 rule (ticks since link-up) the tick-50 request would
    have been granted forty-seven ticks after the grace ran out, which is
    what tools/wp4_negative_controls.py shows by running this very test
    against a copy of the engine with the grace counted in ticks."""
    from sixg_sim.phy_mac_state import RelayMode
    from sixg_sim.agent import PHYMACAction
    sim = _with_service(_two_fragment_sim())
    rid = _register_bridge(sim, tick=0)
    ps = sim.phy_mac_states[A1]
    off = PHYMACAction(relay_mode_idx=list(RelayMode).index(RelayMode.OFF))

    # ticks 0..47: physically up and bridging, NOT in the forwarding view
    for t in range(0, 48):
        sim.current_tick = t
        sim._service_relay_links()
        sim._snapshot_relay_link_passes()
    assert sim._relay_bridge_since[rid] == 0
    assert sim._relay_link_passes.get(rid, 0) == 0
    assert rid in sim.distinct_bridge_link_ids()

    # a request while it is still uninstalled: held, 'not_installed'
    sim.current_tick = 30
    before = _holds(sim)
    sim._step_phy_mac(A1, off)
    assert rid in sim.topology.links
    assert ps.relay_mode == RelayMode.CAPACITY_BOOST
    assert _hold_delta(sim, before) == {'not_installed': 1}

    # the view installs it at tick 48 (what the service mirrors into
    # _relay_route_since); the engine's snapshot counts two passes
    for t in (48, 49):
        sim.current_tick = t
        sim._relay_route_since[rid] = 48
        sim._service_relay_links()
        sim._snapshot_relay_link_passes()
    assert sim._relay_link_passes[rid] == 2

    # tick 50: up for 50 ticks, installed for 2 passes -> REFUSED
    sim.current_tick = 50
    assert sim.current_tick - sim._relay_bridge_since[rid] >= 10 * sim.RELAY_BRIDGE_GRACE_TICKS
    before = _holds(sim)
    sim._step_phy_mac(A1, off)
    assert rid in sim.topology.links, 'the bridge was torn down inside its grace'
    assert sim.topology.links[rid].is_up is True
    assert ps.relay_mode == RelayMode.CAPACITY_BOOST
    assert _hold_delta(sim, before) == {'grace': 1}
    assert sim._relay_links_removed == 0
    sim._snapshot_relay_link_passes()                 # the tick-50 pass
    assert sim._relay_link_passes[rid] == 3

    # tick 51: three usable passes, idle -> GRANTED
    sim.current_tick = 51
    before = _holds(sim)
    sim._step_phy_mac(A1, off)
    assert rid not in sim.topology.links
    assert ps.relay_mode == RelayMode.OFF
    assert _hold_delta(sim, before) == {}
    assert sim._relay_links_removed == 1
    assert sim._relay_removed_by_reason['mode_change'] == 1
    assert sim._relay_bridge_teardowns == 1
    assert sim._relay_idle_bridge_teardowns == 0
    rec = sim._relay_lifetimes[rid]
    assert (rec['created'], rec['up'], rec['removed'], rec['cause'],
            rec['passes'], rec['was_bridge'], rec['protected']) == (
                0, 0, 51, 'mode_change', 3, True, True)


def test_an_uninstalled_bridge_torn_down_is_an_idle_bridge_teardown():
    """The M8 idle-bridge count through the real teardown site: a sole-path
    bridge that never entered the view is held under 'not_installed' for as
    long as it is protected; once a parallel relay makes it redundant it
    loses protection and goes, and it is booked as a bridge teardown with no
    usable pass."""
    from sixg_sim.phy_mac_state import RelayMode
    from sixg_sim.agent import PHYMACAction
    sim = _with_service(_two_fragment_sim())
    rid = _register_bridge(sim, tick=0)
    ps = sim.phy_mac_states[A1]
    off = PHYMACAction(relay_mode_idx=list(RelayMode).index(RelayMode.OFF))
    for t in range(0, 20):
        sim.current_tick = t
        sim._service_relay_links()
        sim._snapshot_relay_link_passes()
    # a second relay across the same cut, between the other two nodes.  The
    # merge forest keeps the id that sorts first (distinct_bridge_link_ids),
    # so this one is named to sort before `rid`, which becomes redundant.
    second = 'TR_A_0'
    assert second < rid
    sim.topology.links[second] = _link(second, A0, B1, cap=30.0,
                                       lt=LinkType.TRANSPORT_RELAY,
                                       iface=InterfaceType.BACKHAUL)
    sim.topology.sync_pair(A0, B1)
    sim.topology.invalidate_infrastructure_cache()
    sim.current_tick = 20
    sim._service_relay_links()
    assert rid in sim._relay_bridge_since              # still a bridge ...
    assert rid not in sim.distinct_bridge_link_ids()   # ... but redundant
    sim._step_phy_mac(A1, off)
    assert rid not in sim.topology.links
    assert sim._relay_bridge_teardowns == 1
    assert sim._relay_idle_bridge_teardowns == 1
    assert sim._relay_lifetimes[rid]['passes'] == 0
    assert sim._relay_lifetimes[rid]['protected'] is False


def test_a_link_up_for_three_ticks_with_three_passes_is_released():
    """The mirror image: the pass rule is not a longer timer.  Three usable
    passes release the bridge however few ticks it has been up."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=10)
    sim.current_tick = 11
    sim._relay_bridge_since[RID] = 10
    _installed(sim, RID, since=10, passes=3)
    blocked, added = _blocked(sim)
    assert blocked is False
    assert added == {}


# ----------------------------------------------------------------------
# 3. one test per hold reason, and the fallback
# ----------------------------------------------------------------------
def test_hold_reason_lockout():
    """A beam still in training: the radio physically cannot be re-pointed."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, up=False, tick=0)
    sim._relay_link_ready_tick[RID] = 3
    sim.current_tick = 1
    blocked, added = _blocked(sim)
    assert blocked is True
    assert added == {'lockout': 1}


def test_hold_reason_not_installed():
    """Physically up, protected, and NOT in the arm's forwarding view: it has
    had no opportunity to be useful, so it is held (RERUN_SPEC WP4)."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    sim.current_tick = 40
    sim._relay_bridge_since[RID] = 0
    assert RID not in sim._relay_route_since
    blocked, added = _blocked(sim)
    assert blocked is True
    assert added == {'not_installed': 1}


def test_hold_reason_grace():
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    sim.current_tick = 5
    _installed(sim, RID, since=3, passes=1)
    blocked, added = _blocked(sim)
    assert blocked is True
    assert added == {'grace': 1}


def test_hold_reason_busy():
    """Make-before-break: past its grace, but measurably loaded."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0, cap=100.0)
    sim.current_tick = 20
    _installed(sim, RID, since=5, passes=9)
    sim._relay_link_load[RID] = sim.RELAY_BRIDGE_BUSY_UTIL
    blocked, added = _blocked(sim)
    assert blocked is True
    assert added == {'busy': 1}
    # and it is released the moment it goes idle -- a load condition, not a
    # timer
    sim._relay_link_load[RID] = sim.RELAY_BRIDGE_BUSY_UTIL / 2
    blocked, added = _blocked(sim)
    assert blocked is False
    assert added == {}


def test_an_arm_without_a_service_keeps_the_physical_stamp():
    """SDN (and the MARL_DETECT_TICKS=0 ablation) have no installation stamp,
    so the grace stays what it was: ticks since link-up.  This is why the
    section-14 matrix can require SDN to be bit-identical across WP4.  The
    pass counters are irrelevant there -- a link with ZERO passes and no
    entry in the forwarding view is released once its three ticks are up."""
    sim = _two_fragment_sim()
    assert sim._relay_grace_in_passes() is False
    _add_bridge(sim, tick=0)
    sim._relay_bridge_since[RID] = 0
    assert RID not in sim._relay_route_since
    assert sim._relay_link_passes.get(RID, 0) == 0
    sim.current_tick = 2                     # two ticks up: held
    blocked, added = _blocked(sim)
    assert blocked is True
    assert added == {'grace': 1}
    sim.current_tick = 3                     # three ticks up: released
    blocked, added = _blocked(sim)
    assert blocked is False
    assert added == {}


def test_an_arm_without_a_service_counts_passes_on_the_true_graph():
    """D4.17.  With no link-state service the forwarding view IS the true
    graph, so a physically-up relay link that the graph's edge names had a
    usable pass.  At c5ffe50 the counter stayed empty for such an arm, and
    every SDN bridge teardown was booked as IDLE.  The grace of this arm must
    still ignore the counter (physical stamp): passes are recording here."""
    sim = _two_fragment_sim()
    assert sim._relay_grace_in_passes() is False
    _add_bridge(sim, tick=0)
    assert sim.topology.graph[A1][B0]['link_id'] == RID
    sim._snapshot_relay_link_passes()
    sim._snapshot_relay_link_passes()
    assert sim._relay_link_passes[RID] == 2
    # physically down: not usable, not counted
    sim.topology.links[RID].is_up = False
    sim.topology.sync_pair(A1, B0)
    sim._snapshot_relay_link_passes()
    assert sim._relay_link_passes[RID] == 2
    sim.topology.links[RID].is_up = True
    sim.topology.sync_pair(A1, B0)
    sim.topology.invalidate_infrastructure_cache()
    # the grace still reads ticks, not passes: many passes, two ticks -> held
    sim._relay_link_passes[RID] = 10
    sim._relay_bridge_since[RID] = 0
    sim.current_tick = 2
    blocked, added = _blocked(sim)
    assert blocked is True and added == {'grace': 1}
    # and its bridge teardown is not idle
    sim.current_tick = 3
    sim._remove_relay_link(RID, reason='mode_change')
    assert sim._relay_bridge_teardowns == 1
    assert sim._relay_idle_bridge_teardowns == 0


def test_a_redundant_relay_link_gets_no_protection():
    """Unchanged by WP4: only a sole-path (tree-edge) bridge is protected."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    second = 'TR_O-DU_1_O-DU_2_2'
    sim.topology.links[second] = _link(second, A1, B0, cap=30.0,
                                       lt=LinkType.TRANSPORT_RELAY,
                                       iface=InterfaceType.BACKHAUL)
    sim._record_relay_link_created(second, sim.topology.links[second])
    sim.topology.invalidate_infrastructure_cache()
    sim.current_tick = 1
    protected = sim.distinct_bridge_link_ids()
    assert len(protected) == 1, protected


# ----------------------------------------------------------------------
# 4. the churn counters
# ----------------------------------------------------------------------
def test_usable_passes_are_counted_from_the_installed_view():
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    # nothing installed: no pass is usable
    sim._snapshot_relay_link_passes()
    assert sim._relay_link_passes.get(RID, 0) == 0
    sim._relay_route_since[RID] = 0
    for _ in range(3):
        sim._snapshot_relay_link_passes()
    assert sim._relay_link_passes[RID] == 3
    # a link that has left the topology loses its credit, so an id that came
    # back could never inherit it
    sim.topology.links.pop(RID)
    sim._relay_route_since.pop(RID)
    sim._snapshot_relay_link_passes()
    assert RID not in sim._relay_link_passes


def test_idle_bridge_teardowns_are_counted_separately():
    """M8's two counters.  A bridge torn down with ZERO usable passes never
    carried anything, and is the flapping signature; one with passes is an
    ordinary bridge teardown and must NOT be booked as idle."""
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    sim.current_tick = 7
    sim._relay_bridge_since[RID] = 0          # it IS a bridge at teardown
    sim._remove_relay_link(RID, reason='mode_change')
    assert sim._relay_bridge_teardowns == 1
    assert sim._relay_idle_bridge_teardowns == 1
    assert sim._relay_links_removed == 1
    assert sim._relay_removed_by_reason['mode_change'] == 1

    sim2 = _with_service(_two_fragment_sim())
    _add_bridge(sim2, tick=0)
    sim2.current_tick = 7
    sim2._relay_bridge_since[RID] = 0
    sim2._relay_link_passes[RID] = 4          # it carried the hop
    sim2._remove_relay_link(RID, reason='mode_change')
    assert sim2._relay_bridge_teardowns == 1
    assert sim2._relay_idle_bridge_teardowns == 0


def test_a_non_bridge_teardown_is_neither():
    sim = _with_service(_two_fragment_sim())
    _add_bridge(sim, tick=0)
    sim.current_tick = 7
    assert RID not in sim._relay_bridge_since
    sim._remove_relay_link(RID, reason='repoint')
    assert sim._relay_bridge_teardowns == 0
    assert sim._relay_idle_bridge_teardowns == 0
    assert sim._relay_removed_by_reason['repoint'] == 1


def test_removals_by_reason_sum_to_the_removal_count():
    sim = _two_fragment_sim()
    for i, reason in enumerate(('mode_change', 'repoint', None)):
        rid = 'TR_O-DU_1_O-DU_2_%d' % i
        sim.current_tick = i
        _add_bridge(sim, rid=rid)
        sim._remove_relay_link(rid, reason=reason)
    assert sim._relay_links_removed == 3
    assert sum(sim._relay_removed_by_reason.values()) == 3
    assert sim._relay_removed_by_reason['unspecified'] == 1
    # a removal of an id that is not there moves no counter at all
    sim._remove_relay_link('TR_nobody_0', reason='mode_change')
    assert sim._relay_links_removed == 3
    assert sum(sim._relay_removed_by_reason.values()) == 3


def test_the_lifetime_record_is_the_handoff_record():
    """docs/TRACKB_HANDOFF.md section 2: created / up / removed / cause /
    was_bridge / load, `removed` a real tick and None while the link is still
    up, `up` stamped when the beam finished training."""
    sim = _with_service(_two_fragment_sim())
    sim.current_tick = 4
    _add_bridge(sim, up=False)                       # created inside training
    rec = sim._relay_lifetimes[RID]
    assert rec['created'] == 4 and rec['up'] is None
    assert rec['removed'] is None and rec['cause'] is None
    assert rec['was_bridge'] is False
    # at-teardown fields are None while the link is up (D4.18)
    assert rec['load'] is None and rec['passes'] is None
    assert rec['protected'] is None
    # the beam comes up at tick 7
    sim._relay_link_ready_tick[RID] = 7
    sim.current_tick = 7
    sim._service_relay_links()
    assert sim.topology.links[RID].is_up is True
    assert rec['up'] == 7
    assert rec['was_bridge'] is True                 # it joins the fragments
    # still up at the end of the run: `removed` stays None, which is what
    # right-censors M11 (it must never be the run length)
    assert rec['removed'] is None
    sim.current_tick = 30
    sim._relay_link_passes[RID] = 5
    sim._relay_link_load[RID] = 0.25
    sim._remove_relay_link(RID, reason='repoint')
    assert rec['removed'] == 30 and rec['cause'] == 'repoint'
    assert abs(rec['load'] - 0.25) < 1e-12
    assert rec['passes'] == 5
    assert rec['was_bridge'] is True
    assert rec['protected'] is True                  # the sole path


def test_was_bridge_survives_the_link_ceasing_to_bridge():
    """`was_bridge` means EVER a bridge (M11's population); the bridge
    TEARDOWN counter is the other question and reads the live table."""
    sim = _with_service(_two_fragment_sim())
    sim.current_tick = 0
    _add_bridge(sim)
    sim._service_relay_links()
    rec = sim._relay_lifetimes[RID]
    assert rec['was_bridge'] is True
    assert RID in sim._relay_bridge_since
    # the fragments merge by another route: it is no longer a bridge
    sim.topology.links['Mesh_X'] = _link('Mesh_X', A0, B1)
    sim.topology.sync_pair(A0, B1)
    sim.topology.invalidate_infrastructure_cache()
    sim.current_tick = 1
    sim._service_relay_links()
    assert RID not in sim._relay_bridge_since
    assert rec['was_bridge'] is True
    sim._remove_relay_link(RID, reason='mode_change')
    assert sim._relay_bridge_teardowns == 0          # not a bridge AT teardown
    assert rec['was_bridge'] is True                 # but it was one


def test_relay_mode_flips_count_the_executed_mode():
    from sixg_sim.phy_mac_state import RelayMode
    sim = _two_fragment_sim()
    sim._snapshot_relay_mode_flips()                 # first pass: no flip
    assert sim._relay_mode_flips == 0
    sim.phy_mac_states[A1].relay_mode = RelayMode.CAPACITY_BOOST
    sim.phy_mac_states[B0].relay_mode = RelayMode.LOCAL_REROUTE
    sim._snapshot_relay_mode_flips()
    assert sim._relay_mode_flips == 2
    sim._snapshot_relay_mode_flips()                 # nothing moved
    assert sim._relay_mode_flips == 2
    sim.phy_mac_states[A1].relay_mode = RelayMode.OFF
    sim._snapshot_relay_mode_flips()
    assert sim._relay_mode_flips == 3


def test_both_pass_counters_are_taken_at_the_end_of_the_forwarding_pass():
    """Where they are taken is the whole of their meaning.  The relay-mode
    flips must be counted AFTER the engine's teardown holds have reverted a
    mode (M9 is the EXECUTED mode), and the usable passes must be counted
    after the pass that used them -- which is the end of _forward_traffic, the
    one point both tick orderings agree on (it is where the load and Block A
    snapshots already are).  Nothing else may call either."""
    with open(os.path.join(ROOT, 'sixg_sim', 'simulation.py'),
              encoding='utf-8') as fh:
        src = fh.read()
    for name in ('_snapshot_relay_link_passes', '_snapshot_relay_mode_flips'):
        assert src.count('self.%s()' % name) == 1, name
        assert src.count('def %s(self)' % name) == 1, name
    tail = src.split('self._snapshot_relay_link_load()')[-1]
    head = tail.split('def _collect_metrics')[0]
    assert 'self._snapshot_relay_link_passes()' in head
    assert 'self._snapshot_relay_mode_flips()' in head
    # no driver takes them itself: the engine does, inside the forwarding
    # pass step_tick makes (tick_step names them in its phase-8 comment only)
    for part in (('sixg_sim', 'tick_step.py'), ('run_timeline_comparison.py',),
                 ('train_on_comparison.py',)):
        with open(os.path.join(ROOT, *part), encoding='utf-8') as fh:
            assert 'sim._snapshot_relay' not in fh.read(), part


# ----------------------------------------------------------------------
# 5. the decision digest
# ----------------------------------------------------------------------
def test_the_decision_digest_moves_with_the_executed_decision():
    from sixg_sim.phy_mac_state import RelayMode
    from sixg_sim.tick_step import decision_digest
    sim = _two_fragment_sim()
    d0 = decision_digest(sim)
    assert isinstance(d0, str) and len(d0) == 64
    assert decision_digest(sim) == d0                # deterministic
    sim.phy_mac_states[A1].relay_mode = RelayMode.CAPACITY_BOOST
    d1 = decision_digest(sim)
    assert d1 != d0
    sim.phy_mac_states[A1].relay_mode = RelayMode.OFF
    assert decision_digest(sim) == d0
    sim.phy_mac_states[B0].tx_power_dbm += 1.0       # an executed tx power
    assert decision_digest(sim) != d0


# ----------------------------------------------------------------------
# 5b. the per-site postcard ledger (postcards_by_site, n_agent_sites)
# ----------------------------------------------------------------------
def test_the_postcard_ledger_keeps_silent_and_departed_sites():
    """One entry per site that hosted an agent at any post-event tick: a
    site that never sent appears with 0, a site whose agent left mid-window
    stays with what it sent, a site that was an agent only BEFORE the event
    is not on it, and sends before the event are not counted."""
    from sixg_sim.tick_step import PostEventPostcardLedger
    led = PostEventPostcardLedger(t_sev=2)
    pre = {'S1', 'S2', 'S3', 'S4'}
    led.update(0, pre, {'S1': 1})
    led.update(1, pre, {'S1': 1, 'S4': 1})
    led.update(2, {'S1', 'S2', 'S3'}, {'S1': 2, 'S3': 1, 'S4': 1})
    led.update(3, {'S1', 'S2'}, {'S1': 3, 'S3': 1, 'S4': 1})
    by_site, n = led.record()
    assert by_site == {'S1': 2, 'S2': 0, 'S3': 1}, by_site
    assert n == 3
    assert list(by_site) == sorted(by_site)
    assert led.roster == frozenset({'S1', 'S2', 'S3'})


def test_the_postcard_ledger_denominator_is_not_derived_from_the_senders():
    """D4.13: `n_agent_sites` is the ROSTER size, read from the agent table.
    A post-event sender that was never an agent site is kept in the dict
    (its sends are not dropped) and makes the dict one longer than the
    roster, so the analysis's coverage check (len(postcards_by_site) ==
    n_agent_sites) FAILS on it.  Deriving both numbers from one set made that
    check pass by construction whatever the engine did."""
    from sixg_sim.tick_step import PostEventPostcardLedger
    led = PostEventPostcardLedger(t_sev=0)
    led.update(0, {'S1'}, {'S1': 1, 'X': 1})
    by_site, n = led.record()
    assert by_site == {'S1': 1, 'X': 1}
    assert n == 1
    assert len(by_site) != n


def test_the_postcard_ledger_refuses_a_counter_that_goes_backwards():
    from sixg_sim.tick_step import PostEventPostcardLedger
    led = PostEventPostcardLedger(t_sev=0)
    led.update(0, {'S1'}, {'S1': 2})
    try:
        led.update(1, {'S1'}, {'S1': 1})
    except ValueError:
        return
    raise AssertionError('a reset cumulative counter was accepted')


# ----------------------------------------------------------------------
# 6 and 7. the real harness: the pickle contract and the churn series
# ----------------------------------------------------------------------
_cache = {}


def _pass(seed, scenario='recovery', arm='marl_static'):
    """One real TICKS-long pass, cached.  Every pass in this file uses the
    same TICKS, so the module globals `_apply_runtime_overrides` sets do not
    depend on the order the tests run in."""
    key = (arm, seed, scenario)
    if key in _cache:
        return _cache[key]
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    import sixg_sim.mappo_trainer as _T
    import sixg_sim.simulation as _S
    if not (all(os.environ.get(k) == v for k, v in EVAL_ENV.items())
            and _T._RELAY_H_FLOOR == float(EVAL_ENV['MARL_RELAY_ENTROPY_FLOOR'])
            and _S.STEER_MODEL == EVAL_ENV['MARL_STEER_MODEL']):
        # the reward is frozen since WP8, so it is no longer the proxy for
        # "imported under the evaluated environment" (D5.8 used it)
        raise AssertionError(
            'CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'sixg_sim was imported outside the evaluated environment; run '
            'this file through tests/run_tests.py')
    if 'ckpt' not in _cache:
        from sixg_sim.agent import mint_schema_current_checkpoint
        mint_schema_current_checkpoint(CKPT, torch_seed=CKPT_TORCH_SEED,
                                       extra={'test': 'wp4_bridge_stability'})
        _cache['ckpt'] = CKPT
    R._apply_runtime_overrides({'total_ticks': TICKS})
    out = R.run_physics_pass(arm, seed, CKPT, scenario)
    _cache[key] = (out, R)
    return _cache[key]


def _is_int_series(v, n):
    return (isinstance(v, list) and len(v) == n
            and all(isinstance(x, int) and not isinstance(x, bool) for x in v))


def _check_postcard_ledger(out, t_sev):
    """postcards_by_site against the `postcards` series it must add up to:
    the per-site sends after the event sum to the aggregate sends after the
    event, tick window for tick window.  Two recordings made from different
    counters (the engine's per-site ledger and last_postcards_sent)."""
    by_site = out['postcards_by_site']
    assert isinstance(by_site, dict) and by_site
    assert all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
               for v in by_site.values()), by_site
    assert list(by_site) == sorted(by_site)
    assert isinstance(out['n_agent_sites'], int) and out['n_agent_sites'] > 0
    assert len(by_site) == out['n_agent_sites'], (len(by_site), out['n_agent_sites'])
    assert sum(by_site.values()) == sum(out['postcards'][t_sev:]), (
        sum(by_site.values()), sum(out['postcards'][t_sev:]))


def _check_lifetimes_against_relay_count(out):
    """D4.16: the lifetime records re-derive the `relay_count` series, which
    the harness reads from the KPI snapshot (active_relay_links after the
    step), tick by tick: a link is up at tick t iff its record has
    up <= t and (removed is None or removed > t).  This is what makes the
    censoring (`removed` None only for a link really up at the end), the
    `up` stamp and every teardown checkable against something the lifetime
    code does not write."""
    lifet = out['relay_lifetimes']
    rc = out['relay_count']
    for t in range(len(rc)):
        live = sum(1 for r in lifet
                   if r['up'] is not None and r['up'] <= t
                   and (r['removed'] is None or r['removed'] > t))
        assert live == rc[t], (t, live, rc[t])


def _check_flips_against_mode_nodes(out):
    """`relay_mode_flips` against `relay_mode_nodes` (sites in a non-OFF
    relay mode, from the same post-step KPI snapshot): the count of such
    sites cannot move by k between two ticks unless at least k sites changed
    their executed mode -- or appeared (a site's FIRST executed mode is not a
    flip: it has no previous one; Scenario B's NeNB), which `node_count`
    bounds -- and no flip is recorded on the first tick."""
    fl, rmn = out['relay_mode_flips'], out['relay_mode_nodes']
    nc = out['node_count']
    assert fl[0] == 0
    for t in range(1, len(fl)):
        assert fl[t] + abs(nc[t] - nc[t - 1]) >= abs(rmn[t] - rmn[t - 1]), (
            t, fl[t], rmn[t - 1], rmn[t], nc[t - 1], nc[t])


def test_the_pickle_carries_every_series_the_analysis_reads():
    """Name, type and length of every WP4 key, on a real pass.  A key under
    another name is invisible to the analysis, and a series shorter than the
    run silently shortens the M1-M7/M10 windows, so both are pinned."""
    out, R = _pass(SEEDS[0])
    n = R.TOTAL_TICKS
    for key in EVENT_SERIES:
        assert key in out, key
        assert _is_int_series(out[key], n), (key, type(out[key]))
        assert all(x >= 0 for x in out[key]), key     # differences of a
        #                                               monotone counter
    for key in DICT_SERIES:
        assert isinstance(out[key], dict), key
        assert out[key], key                          # never an empty dict
        for name, s in out[key].items():
            assert isinstance(name, str), (key, name)
            assert _is_int_series(s, n), (key, name)
    # the split sums to the total, tick by tick
    tot = out['relay_removed']
    by_reason = out['relay_removed_by_reason']
    assert set(by_reason) == set(R.Simulator.RELAY_TEARDOWN_REASONS)
    for t in range(n):
        assert sum(s[t] for s in by_reason.values()) == tot[t], t
    assert set(out['relay_teardown_holds']) == set(R.Simulator.RELAY_HOLD_REASONS)
    # idle bridge teardowns are a SUBSET of bridge teardowns
    for t in range(n):
        assert out['idle_bridge_teardowns'][t] <= out['bridge_teardowns'][t], t
    # lifetimes: a list of records with the handoff's fields
    lifet = out['relay_lifetimes']
    assert isinstance(lifet, list)
    for rec in lifet:
        assert isinstance(rec, dict)
        for f in ('link_id', 'created', 'up', 'removed', 'cause',
                  'was_bridge', 'load'):
            assert f in rec, f
        assert isinstance(rec['created'], int) and 0 <= rec['created'] < n
        assert rec['up'] is None or 0 <= rec['up'] < n
        # NEVER the run length: `removed == T` would turn a censored record
        # into a completed lifetime of exactly T - up (handoff section 2)
        assert rec['removed'] is None or 0 <= rec['removed'] < n, rec
        assert isinstance(rec['was_bridge'], bool)
    # one creation record per created link
    assert len(lifet) == sum(out['relay_created'])
    assert sum(1 for r in lifet if r['removed'] is not None) == sum(out['relay_removed'])
    # the per-site postcard ledger, complete
    _check_postcard_ledger(out, R.SEVERANCE_TICK)
    # the lifetime records against an INDEPENDENT recording (D4.16)
    _check_lifetimes_against_relay_count(out)
    _check_flips_against_mode_nodes(out)
    # S8 and S11
    assert isinstance(out['decision_digest'], list)
    assert len(out['decision_digest']) == n
    assert all(isinstance(d, str) and len(d) == 64 for d in out['decision_digest'])
    assert isinstance(out['energy_ctrl'], list) and len(out['energy_ctrl']) == n
    assert all(isinstance(x, float) and x >= 0.0 for x in out['energy_ctrl'])
    # the per-type overhead breakdown is present for an arm that HAS one, and
    # decomposes the total it is written beside
    assert out['oh_breakdown_available'] is True
    ot = out['overhead_by_type']
    assert isinstance(ot, dict) and ot
    for name, s in ot.items():
        assert isinstance(s, list) and len(s) == n, name


def test_the_overhead_breakdown_is_absent_rather_than_zero_for_a_baseline():
    """D3.18, under the analysis's name: an arm whose protocol model produces
    no decomposition must record NO `overhead_by_type` at all -- all-zero
    series under that name would be read as a measurement of zero."""
    ospf, R = _pass(SEEDS[0], arm='ospf')
    assert ospf['oh_breakdown_available'] is False
    assert 'overhead_by_type' not in ospf
    # ... but its EVENT series exist: a routing arm forms no relay link, so
    # the relay-LINK series are all zero ("absent" would mean "not recorded"
    # in every report) ...
    n = R.TOTAL_TICKS
    for key in EVENT_SERIES:
        assert key in ospf and _is_int_series(ospf[key], n), key
    for key in ('relay_created', 'relay_removed', 'bridge_teardowns',
                'idle_bridge_teardowns'):
        assert sum(ospf[key]) == 0, key
    assert ospf['relay_lifetimes'] == []
    # ... while its relay-MODE flips are real and NOT zero (D4.12): the
    # OSPF/OLSR arms switch every site to LOCAL_REROUTE once the modelled
    # control plane reports the mesh converged (lock_phy_mac_non_mcs), and
    # back when it does not.  That is an executed relay-mode change, so M9
    # counts it for the routing arms too; the c5ffe50 version of this test
    # asserted zero and failed on exactly this.
    assert sum(ospf['relay_mode_flips']) > 0
    assert max(ospf['relay_mode_nodes']) > 0
    _check_flips_against_mode_nodes(ospf)


def test_the_churn_series_are_non_zero_on_the_invariance_seeds():
    """The recording measures something on the seeds the invariance matrix
    uses.  Relay links are created and torn down, bridges form, and the
    engine holds teardowns -- if any of these were silently zero the WP4
    metrics would be reported as "no churn" instead of "not recorded"."""
    n_protected_teardowns = 0
    for seed in SEEDS:
        out, R = _pass(seed)
        created = sum(out['relay_created'])
        removed = sum(out['relay_removed'])
        assert created > 0, seed
        assert removed > 0, seed
        assert sum(out['relay_mode_flips']) > 0, seed
        assert len(out['relay_lifetimes']) == created, seed
        assert sum(r['was_bridge'] for r in out['relay_lifetimes']) > 0, seed
        # the engine held at least one teardown, and every hold is booked
        # under a reason of the vocabulary
        holds = out['relay_teardown_holds']
        assert sum(sum(s) for s in holds.values()) > 0, seed
        # and the post-event window really is where the churn is (actions are
        # locked before the event, so nothing can churn there)
        t_sev = R.SEVERANCE_TICK
        assert sum(out['relay_created'][:t_sev]) == 0, seed
        assert sum(out['relay_mode_flips'][:t_sev]) == 0, seed
        # the independent cross-checks on every seed, not one representative
        _check_lifetimes_against_relay_count(out)
        _check_flips_against_mode_nodes(out)
        _check_postcard_ledger(out, t_sev)
        # THE CENTRAL CLAIM ON A REAL PASS (D4.18): this arm has a link-state
        # service, so no SOLE-PATH bridge of it may be torn down with fewer
        # than RELAY_BRIDGE_GRACE_PASSES usable passes
        prot = [r for r in out['relay_lifetimes']
                if r['removed'] is not None and r['protected']]
        short = [r for r in prot
                 if r['passes'] < R.Simulator.RELAY_BRIDGE_GRACE_PASSES]
        assert not short, (seed, short[:3])
        n_protected_teardowns += len(prot)
    # D12-FIXES (D12F.7): since D12.1 -- no data-plane transit through UEs
    # -- the marl_static passes on these seeds tear no sole-path bridge down
    # any more (they tore one down on seed 82 before), at 300, 600 and 1200
    # ticks.  The claim is therefore ALSO checked on the uniform-random
    # control, which runs behind the same link-state service and tears
    # bridges down all the time (70-84 protected teardowns per 600-tick
    # Scenario-A pass of the D12 references), so that it stays exercised.
    per_arm = {'marl_static': n_protected_teardowns}
    for seed in SEEDS:
        out, R = _pass(seed, arm='random')
        prot = [r for r in out['relay_lifetimes']
                if r['removed'] is not None and r['protected']]
        short = [r for r in prot
                 if r['passes'] < R.Simulator.RELAY_BRIDGE_GRACE_PASSES]
        assert not short, ('random', seed, short[:3])
        per_arm['random'] = per_arm.get('random', 0) + len(prot)
        n_protected_teardowns += len(prot)
    # ... and the claim was exercised: sole-path bridges were torn down
    assert n_protected_teardowns > 0, per_arm


def test_scenario_b_records_from_tick_zero():
    """Scenario B is severed at tick 0, so its postcard window is the whole
    run (the ledger is checked against sum(postcards[0:])), and the lifetime
    and flip cross-checks hold there too.  Its roster is the same agent-site
    set as Scenario A's: both scenarios lose the core to `sever_core` and no
    agent site joins or leaves afterwards (the rescue force brings UEs, not a
    radio site -- a NeNB appears only on a 'nenb_deployment' event, which
    neither scenario schedules), so roster growth and departure are exercised
    by the ledger's unit tests, not here."""
    out, R = _pass(SEEDS[0], scenario='rescue_ops')
    n = R.TOTAL_TICKS
    for key in EVENT_SERIES:
        assert _is_int_series(out[key], n), key
    _check_postcard_ledger(out, 0)
    _check_lifetimes_against_relay_count(out)
    _check_flips_against_mode_nodes(out)
    a, _ = _pass(SEEDS[0])
    assert list(out['postcards_by_site']) == list(a['postcards_by_site'])
    assert out['n_agent_sites'] == a['n_agent_sites']
