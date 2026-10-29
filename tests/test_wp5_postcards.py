"""WP5 -- control postcards (RERUN_SPEC WP5, reviewer item 2).

Fast checks on hand-built topologies, then one real run through each driver:

  1. the constants and the declared 60-byte wire layout;
  2. THE ORIGINAL DEFECT, directly: the DCC send counter resets itself on a
     new tick with no reset call from anyone, and a site's second heartbeat
     ten ticks later is admitted -- plus a negative control in which the
     self-reset is disabled and the post-pass invariant catches it;
  3. the frame model: one frame per LIVE infrastructure link incident to
     the sender (parallel links included, down links and UE access links
     excluded), no fan-out cap, only agent-hosting receivers enqueue, 60 B
     per frame, a per-tick ledger, and the link-state service billing it;
  4. the census leak is gone: the postcard carries the sender's OWN
     unserved-UE count, never the network-wide isolated-UE census -- checked
     on the helper AND through the simulator's own wiring
     (_update_phy_mac_observations -> PHYMACState -> the heartbeat on the
     wire) -- and the cooperative-relay field is "a fresh neighbour is
     relaying and reports unserved UEs";
  5. the receiver latch: 30-tick expiry, Block B from every fresh entry, the
     freshness field in [0, 1], `postcard_received` = arrived THIS tick
     (idempotent within a tick); plus the D5.11 merge check (the master
     actor vector does not carry freshness; the trackAgent schema must);
  6. the send invariant (PostcardSendAudit): at most one send per site per
     tick, more than one per site after the event, never vacuous;
  7. the heartbeat is an environment rule: every agent site, every 10
     ticks, identical for the MARL, random and XL-DET executors, and a
     heartbeat stamp from a later tick (an agent carried over from another
     episode) cannot silence a site;
  8. no Block E / fragment-label / learning-exchanger reference survives
     anywhere in the tree (source scan with named, reasoned exemptions),
     including READS of the removed census field; the removed switches
     (MARL_POSTCARD_FRAGMENTS, MARL_POSTCARD_ALWAYS=0,
     MARL_POSTCARD_DRAIN=legacy) are refused;
  9. the bridge-peer planner's re-pointing: on by default, `off` is the
     evaluation-only sensitivity the training driver refuses;
 10. the real drivers: a harness pass (run_physics_pass) and both training
     driver loops (run_eval_episode, run_comparison_worker_episode), each
     showing that no heartbeat was refused, every agent site sent more than
     one postcard after the event, and the frames billed equal an
     independent recount of the live incident infrastructure links.

Section 10 runs the comparison environment, so this file declares the
evaluated environment in REQUIRES_ENV and tests/run_tests.py re-runs it in a
child process that has it.

Runs under pytest and under tests/run_tests.py.
"""
import dataclasses
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_ENV = {
    'MARL_TRANSPORT_POWER': 'planning', 'MARL_RELAY_STATE': 'reconcile',
    'MARL_RELAY_REPOINT': 'fragment_aware', 'MARL_POSTCARD_ALWAYS': '1',
    'MARL_REWARD_PROFILE': 'mission', 'MARL_RELAY_ENTROPY_FLOOR': '0.01',
    'MARL_STEER_MODEL': 'actionable',
}
REQUIRES_ENV = dict(EVAL_ENV)
if 'sixg_sim.agent' not in sys.modules:
    for _k, _v in EVAL_ENV.items():
        os.environ.setdefault(_k, _v)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                               InterfaceType)
from sixg_sim import postcards as PC

TICKS = 300
SEED = 82
# The checkpoint the real passes load.  MINTED from this tree (trackAgent
# merge): WP6's strict schema gate refuses the tracked pre-WP6
# ce_ck_s1234/policy_final.pt, and these tests check the postcard machinery,
# which does not depend on which weights the actor holds.
CKPT = os.path.join(ROOT, 'output', 'tmp_tests', 'wp5', 'policy_schema_current.pt')
CKPT_TORCH_SEED = 1234

S, A, C, R_, E = 'O-DU_1', 'O-DU_2', 'O-DU_3', 'Relay_4', 'EdgeUPF_5'
CORE = 'AMF_1'
U1, U2, U3 = 'UE_1', 'UE_2', 'UE_3'


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _link(lid, u, v, cap=500.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL, up=True):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=up)


def _star_topology():
    """S has: two live links to A (a mesh hop and a relay on the same pair),
    a live microwave hop to the agent-less EdgeUPF E, a DOWN mesh hop to C
    and a live Uu link to a UE.  Its postcard therefore leaves on exactly
    three frames, two of which reach an agent (A)."""
    t = Topology()
    for nid, nt in ((S, NodeType.O_DU), (A, NodeType.O_DU),
                    (C, NodeType.O_DU), (E, NodeType.EDGEUPF),
                    (U1, NodeType.UE)):
        t.add_node(Node(id=nid, node_type=nt))
    t.add_link(_link('Mesh_SA', S, A))
    t.add_link(_link('TR_S_A_1', S, A, lt=LinkType.TRANSPORT_RELAY))
    t.add_link(_link('MW_SE', S, E, lt=LinkType.MICROWAVE_PTP))
    t.add_link(_link('Mesh_SC', S, C, up=False))
    t.sync_pair(S, C)                 # the graph holds no edge for a down hop
    t.add_link(_link('Mesh_AC', A, C))
    t.add_link(_link('Uu_1', U1, S, cap=100.0, lt=LinkType.WIRELESS,
                     iface=InterfaceType.Uu))
    return t


def _sim(topology, ticks=60):
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    scenario = Scenario(name='wp5', duration_ticks=ticks, events=[],
                        traffic_profiles={})
    sim = Simulator(topology, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 0
    return sim


def _acting_topology():
    """A core-less mesh of four radio sites (three O-DUs and a relay), one
    agent-less EdgeUPF and three UEs, so the island is up from tick 0."""
    t = Topology()
    for nid, nt in ((S, NodeType.O_DU), (A, NodeType.O_DU),
                    (C, NodeType.O_DU), (R_, NodeType.RELAY),
                    (E, NodeType.EDGEUPF), (U1, NodeType.UE),
                    (U2, NodeType.UE), (U3, NodeType.UE)):
        t.add_node(Node(id=nid, node_type=nt))
    t.add_link(_link('Mesh_SA', S, A))
    t.add_link(_link('Mesh_AC', A, C))
    t.add_link(_link('Mesh_CR', C, R_))
    t.add_link(_link('MW_RE', R_, E, lt=LinkType.MICROWAVE_PTP))
    for i, (ue, site) in enumerate(((U1, S), (U2, C), (U3, R_))):
        t.add_link(_link('Uu_%d' % i, ue, site, cap=100.0,
                         lt=LinkType.WIRELESS, iface=InterfaceType.Uu))
    return t


def _pc(sender, tick, relay=False, sinr=0.5, share=0.2, tx=0.4,
        unserved=0):
    from sixg_sim.agent import ControlPostcard
    return ControlPostcard(sender_id=sender, relay_mode_active=relay,
                           best_sinr_to_neighbour=sinr,
                           unserved_ue_count=unserved,
                           prb_avail_for_relay=share, policy_version=0,
                           timestamp=tick, tx_power_dbm_norm=tx)


def _independent_frame_count(topology, sender):
    """Live infrastructure links incident to `sender`, recounted here
    without the DCC's own helper."""
    n = 0
    for lk in topology.links.values():
        if not getattr(lk, 'is_up', False) or sender not in lk.endpoints:
            continue
        a, b = lk.endpoints
        if a == b:
            continue
        types = [getattr(topology.nodes.get(x), 'node_type', None) for x in (a, b)]
        if NodeType.UE in types:
            continue
        n += 1
    return n


# ----------------------------------------------------------------------
# 1. constants and the wire layout
# ----------------------------------------------------------------------
def test_the_constants_are_the_pre_registered_ones():
    assert PC.POSTCARD_PERIOD == 10
    assert PC.POSTCARD_LATCH_TICKS == 30
    assert PC.POSTCARD_ENCODED_BYTES == 60
    assert sum(n for _, n in PC.POSTCARD_WIRE_LAYOUT) == 60
    from sixg_sim.control_plane import DisasterControlChannel
    assert DisasterControlChannel(_star_topology()).frame_bytes == 60


def test_every_postcard_field_travels_in_a_declared_slot():
    """A field with no wire slot would travel for free; a slot for a field
    that no longer exists would be billed for nothing."""
    from sixg_sim.agent import ControlPostcard
    fields = {f.name for f in dataclasses.fields(ControlPostcard)}
    slots = {name for name, _ in PC.POSTCARD_WIRE_LAYOUT}
    assert set(PC.POSTCARD_FIELD_SLOTS) == fields, (
        set(PC.POSTCARD_FIELD_SLOTS) ^ fields)
    assert set(PC.POSTCARD_FIELD_SLOTS.values()) <= slots


# ----------------------------------------------------------------------
# 2. the original defect: the per-tick counter resets itself
# ----------------------------------------------------------------------
def test_the_dcc_counter_resets_itself_on_a_new_tick():
    """The counter used to be cleared only by reset_tick_counters(), which
    only Simulator.run_simulation calls and no driver runs: every send after
    a site's first was refused.  Nothing here calls a reset."""
    from sixg_sim.control_plane import DisasterControlChannel
    dcc = DisasterControlChannel(_star_topology())
    assert dcc.send_postcard(_pc(S, 5), 5) is True
    # a second send in the SAME tick is refused (the limiter still works)
    assert dcc.send_postcard(_pc(S, 5), 5) is False
    assert dcc.ledger(5)['refused'] == 1 and dcc.ledger(5)['sends'] == 1
    # the next heartbeat, ten ticks later, is admitted -- the defect refused it
    assert dcc.send_postcard(_pc(S, 15), 15) is True
    assert dcc.send_postcard(_pc(S, 25), 25) is True
    assert dcc.ledger(25)['sends'] == 1 and dcc.ledger(25)['refused'] == 0
    assert dcc.total_ledger['sends'] == 3
    # the ledger of a tick on which nothing was sent is zero
    assert dcc.ledger(16) == {'sends': 0, 'frames': 0, 'bytes': 0,
                              'deliveries': 0, 'refused': 0}


def test_negative_control_the_invariant_catches_a_counter_that_never_resets():
    """Disable the self-reset -- i.e. put the evaluated defect back -- and
    run the real heartbeat through step_tick: every site's second heartbeat
    is refused, and the post-pass invariant fails the pass."""
    sim = _sim(_acting_topology(), ticks=40)
    sim._detect_island_mode = lambda: True
    dcc = sim.control_plane.dcc
    dcc._roll_tick(0)
    dcc._roll_tick = lambda tick: None            # the defect
    for t in range(25):
        sim.step_tick(t, lambda tick, obs: sim._execute_agent_actions(obs))
    assert dcc.total_ledger['refused'] > 0
    try:
        sim.postcard_audit.check()
    except PC.PostcardInvariantError as e:
        assert 'at most one postcard after the event' in str(e)
    else:
        raise AssertionError('the invariant passed a one-postcard-per-site '
                             'episode')


# ----------------------------------------------------------------------
# 3. the frame model, the ledger and the bill
# ----------------------------------------------------------------------
def test_frames_equal_the_live_incident_infrastructure_links():
    from sixg_sim.control_plane import DisasterControlChannel
    topo = _star_topology()
    dcc = DisasterControlChannel(topo)
    assert dcc.send_postcard(_pc(S, 7), 7, receivers={S, A, C})
    led = dcc.ledger(7)
    # Mesh_SA, TR_S_A_1 (a parallel relay on the same pair: its own frame),
    # MW_SE.  NOT Mesh_SC (down) and NOT Uu_1 (access link).  No cap.
    assert led['frames'] == 3 == _independent_frame_count(topo, S)
    assert led['bytes'] == 3 * 60
    # only agent-hosting receivers enqueue: A twice, E (no agent) never,
    # C never (its link is down)
    assert led['deliveries'] == 2
    assert len(dcc.message_queues[A]) == 2
    assert len(dcc.message_queues[E]) == 0
    assert len(dcc.message_queues[C]) == 0
    assert all(m.delivery_tick == 8 for m in dcc.message_queues[A])
    # nothing is delivered before its tick, everything on it
    assert dcc.receive_messages(A, 7) == []
    assert len(dcc.receive_messages(A, 8)) == 2


def test_a_site_with_no_live_infrastructure_link_sends_no_frame():
    """The heartbeat is still ISSUED (the limiter admits it and the per-site
    record counts it), but nothing leaves the site and nothing is billed."""
    from sixg_sim.control_plane import DisasterControlChannel
    topo = _star_topology()
    dcc = DisasterControlChannel(topo)
    assert dcc.send_postcard(_pc(C, 3), 3, receivers={S, A, C}) is True
    # C's only live infrastructure link is Mesh_AC
    assert dcc.ledger(3)['frames'] == 1
    topo.links['Mesh_AC'].is_up = False
    assert dcc.send_postcard(_pc(C, 13), 13, receivers={S, A, C}) is True
    assert dcc.ledger(13) == {'sends': 1, 'frames': 0, 'bytes': 0,
                              'deliveries': 0, 'refused': 0}


def test_the_heartbeat_enqueues_only_at_agent_sites_and_the_service_bills_it():
    from sixg_sim import link_state_service as LS
    sim = _sim(_acting_topology(), ticks=40)
    sim._detect_island_mode = lambda: True
    # attached as the drivers attach it: the per-site action lock (D13.2)
    # reads each site's own flag from it, so the core-less island acts from
    # tick 1 (the flag of the tick-0 step) and the heartbeats fall on 1, 11
    svc = LS.make_link_state_service(sim)
    with _Spy() as spy:
        for t in range(13):
            sim.step_tick(t, lambda tick, obs: sim._execute_agent_actions(obs),
                          forwarding=svc)
            led = sim.control_plane.dcc.ledger(t)
            bill = svc.tick_messages(t)
            assert bill['postcard_msgs'] == led['frames'], t
            assert bill['postcard_bytes'] == led['bytes'] == 60 * led['frames'], t
            calls = [c for c in spy.calls if c[0] == t]
            # billed frames == the ledger == an independent recount of each
            # sender's live incident infrastructure links at its send
            assert led['frames'] == sum(c[3] for c in calls) == sum(c[4] for c in calls), t
            if t in (1, 11):
                assert led['sends'] == len(sim.agents) == len(calls) == 4
                assert led['frames'] > 0
            else:
                assert led['sends'] == 0 and bill['postcard_msgs'] == 0, t
    # E hosts no agent: nothing was ever queued for it
    assert E not in sim.agents
    assert not sim.control_plane.dcc.message_queues.get(E)
    # and a MARL-path arm with NO service (MARL_DETECT_TICKS=0) is billed
    # the same postcards from the ledger
    only = LS.postcard_only_messages(sim, 11)
    assert only['postcard_bytes'] == only['bytes'] == sim.control_plane.dcc.ledger(11)['bytes']
    assert only['bytes'] > 0


# ----------------------------------------------------------------------
# 4. the census leak
# ----------------------------------------------------------------------
def test_the_unserved_count_is_the_sites_own():
    ue_serving = {'u1': [S], 'u2': [S, A], 'u3': [A], 'u4': [A], 'u5': [C]}
    flows = [('u1', 'u3'), ('u2', 'u4'), ('u9', 'u8')]
    delivered = {'u3'}
    got = PC.site_unserved_ue_counts(ue_serving, flows, delivered, [S, A, C])
    # S: u1 (demand, not delivered) + u2 (dual-connected, not delivered)
    # A: u2 + u4 (u3 was delivered); C: u5 has no demand
    assert got == {S: 2, A: 2, C: 0}
    # a UE attached to NO site (u9, u8: flow endpoints, never delivered) is
    # counted nowhere -- that is the part of the census no site can see
    assert sum(got.values()) == 4


def test_the_postcard_carries_the_senders_own_count_not_the_census():
    from sixg_sim.agent import ControlPostcard
    sim = _sim(_star_topology())
    sim.phy_mac_states[S].unserved_ue_count = 2
    sim.phy_mac_states[A].unserved_ue_count = 3
    agent = sim.agents[S]
    sim._isolated_ue_count = 77                   # the network-wide census
    pc1 = sim._heartbeat_postcard(S, agent)
    sim._isolated_ue_count = 5
    pc2 = sim._heartbeat_postcard(S, agent)
    assert pc1.unserved_ue_count == 2 and pc2 == pc1
    assert sim._heartbeat_postcard(A, sim.agents[A]).unserved_ue_count == 3
    fields = {f.name for f in dataclasses.fields(ControlPostcard)}
    assert 'isolated_ue_count' not in fields
    # the count saturates at one byte
    sim.phy_mac_states[S].unserved_ue_count = 1000
    assert sim._heartbeat_postcard(S, agent).unserved_ue_count == 255


def _census_topology():
    """Two agent sites S and A (one live mesh hop) and six UEs:

      UE_1  at S          flow, DELIVERED on the last pass
      UE_2  at S          flow, not delivered          -> unserved at S
      UE_3  at S          no flow (no demand)
      UE_4  at S AND A    flow, not delivered          -> unserved at both
      UE_5  at NO site    flow endpoint                -> the census only
      UE_6  at A          flow, DELIVERED on the last pass
    """
    t = Topology()
    t.add_node(Node(id=S, node_type=NodeType.O_DU))
    t.add_node(Node(id=A, node_type=NodeType.O_DU))
    for i in range(1, 7):
        t.add_node(Node(id='UE_%d' % i, node_type=NodeType.UE))
    t.add_link(_link('Mesh_SA', S, A))
    for ue, site in (('UE_1', S), ('UE_2', S), ('UE_3', S), ('UE_4', S),
                     ('UE_4', A), ('UE_6', A)):
        t.add_link(_link('Uu_%s_%s' % (ue, site), ue, site, cap=100.0,
                         lt=LinkType.WIRELESS, iface=InterfaceType.Uu))
    return t


def test_the_simulator_writes_each_sites_own_unserved_count():
    """The WIRING, not the helper: Simulator._update_phy_mac_observations
    computes each site's count from ue_serving (its own live Uu links),
    ue_to_ue_flows and _delivered_ue_endpoints, writes it into PHYMACState,
    and the heartbeat carries that value onto the wire."""
    from sixg_sim.agent import PHYMACAction
    sim = _sim(_census_topology())
    assert set(sim.agents) == {S, A}
    sim.ue_to_ue_flows = [('UE_1', 'UE_6'), ('UE_2', 'UE_4'), ('UE_5', 'UE_2')]
    sim._delivered_ue_endpoints = {'UE_1', 'UE_6'}
    sim._update_phy_mac_observations()
    ps = sim.phy_mac_states
    # S: UE_2 + UE_4 (UE_1 was delivered, UE_3 has no demand);
    # A: UE_4, dual-attached (UE_6 was delivered)
    assert (ps[S].unserved_ue_count, ps[A].unserved_ue_count) == (2, 1)
    # the network-wide census is a different number: UE_5 is attached to no
    # site, so no site can see it
    assert sim._isolated_ue_count == 1
    assert sim._heartbeat_postcard(S, sim.agents[S]).unserved_ue_count == 2
    assert sim._heartbeat_postcard(A, sim.agents[A]).unserved_ue_count == 1
    # the real send carries it over the live hop to the neighbour's queue
    sim.control_plane.set_island_mode(True)
    assert sim._send_postcard_heartbeat(S, sim.agents[S], PHYMACAction())
    got = sim.control_plane.get_received_postcards(A, sim.current_tick + 1)
    assert [pc.sender_id for pc in got] == [S], got
    assert got[0].unserved_ue_count == 2
    # the inputs are read afresh on every call: once every UE with demand
    # was delivered, no site is unserved
    sim._delivered_ue_endpoints = {'UE_1', 'UE_2', 'UE_4', 'UE_5', 'UE_6'}
    sim._update_phy_mac_observations()
    assert (ps[S].unserved_ue_count, ps[A].unserved_ue_count) == (0, 0)
    # a UE counts only where it is attached over a LIVE Uu link
    sim._delivered_ue_endpoints = {'UE_1', 'UE_6'}
    sim.topology.links['Uu_UE_4_%s' % A].is_up = False
    sim._update_phy_mac_observations()
    assert (ps[S].unserved_ue_count, ps[A].unserved_ue_count) == (2, 0)


def test_the_cooperative_field_needs_a_relaying_neighbour_with_unserved_ues():
    from sixg_sim.agent import NeighborRadioSummary
    cases = [
        ([_pc(A, 0, relay=True, unserved=2)], 1.0),
        ([_pc(A, 0, relay=True, unserved=0)], 0.0),
        ([_pc(A, 0, relay=False, unserved=4)], 0.0),
        ([_pc(A, 0, relay=False, unserved=4),
          _pc(C, 0, relay=True, unserved=0)], 0.0),
        ([_pc(A, 0, relay=False, unserved=0),
          _pc(C, 0, relay=True, unserved=1)], 1.0),
    ]
    for pcs, want in cases:
        latch = PC.NeighbourLatch()
        latch.ingest(S, pcs, 0)
        nb = PC.fill_block_b(NeighborRadioSummary(), latch.fresh(S, 0),
                             latch.arrived(S, 0))
        assert nb.cooperative_relay_assigned == want, (pcs, want)


# ----------------------------------------------------------------------
# 5. the receiver latch and Block B
# ----------------------------------------------------------------------
def test_the_latch_expires_after_30_ticks():
    latch = PC.NeighbourLatch()
    latch.ingest(S, [_pc(A, 99)], 100)
    for t in range(100, 130):
        assert [e[0] for e in latch.fresh(S, t)] == [A], t
    assert latch.fresh(S, 130) == []
    assert latch.fresh(S, 99) == []          # nothing is fresh before arrival
    # a new postcard renews the entry
    latch.ingest(S, [_pc(A, 128)], 129)
    assert [e[1] for e in latch.fresh(S, 150)] == [21]
    assert latch.fresh(S, 159) == []


def test_block_b_from_the_simulator_latches_and_expires():
    sim = _sim(_star_topology())
    sim.current_tick = 100
    nb = sim._build_neighbor_radio_summary(S, [_pc(A, 99, relay=True, tx=0.8)])
    assert nb.postcard_received == 1.0
    assert nb.neighbor_count_norm == 0.1 and nb.any_relay_active == 1.0
    assert nb.postcard_freshness_norm == 1.0
    # a second build in the same tick (AgentMonitorPlotter) gives the same
    assert sim._build_neighbor_radio_summary(S) == nb
    sim.current_tick = 101
    nb = sim._build_neighbor_radio_summary(S)
    assert nb.postcard_received == 0.0            # nothing arrived THIS tick
    assert nb.neighbor_count_norm == 0.1          # ... but the entry is fresh
    assert nb.interferer_count_norm == 1.0 and nb.neighbour_avg_tx_power_norm == 0.8
    assert abs(nb.postcard_freshness_norm - (1 - 1 / 30)) < 1e-12
    sim.current_tick = 129
    assert sim._build_neighbor_radio_summary(S).neighbor_count_norm == 0.1
    sim.current_tick = 130
    from sixg_sim.agent import NeighborRadioSummary
    assert sim._build_neighbor_radio_summary(S) == NeighborRadioSummary()


def test_the_freshness_field_is_in_range():
    latch = PC.NeighbourLatch()
    latch.ingest(S, [_pc(A, 0)], 0)
    from sixg_sim.agent import NeighborRadioSummary
    prev = None
    for t in range(0, 45):
        nb = PC.fill_block_b(NeighborRadioSummary(), latch.fresh(S, t),
                             latch.arrived(S, t))
        f = nb.postcard_freshness_norm
        assert 0.0 <= f <= 1.0, (t, f)
        if prev is not None:
            assert f <= prev, t
        prev = f
        if t == 0:
            assert f == 1.0
        if t >= 30:
            assert f == 0.0
    # never heard anything: 0
    assert PC.fill_block_b(NeighborRadioSummary(), [], False).postcard_freshness_norm == 0.0


def test_block_b_is_built_from_every_fresh_entry():
    from sixg_sim.agent import NeighborRadioSummary
    latch = PC.NeighbourLatch()
    latch.ingest(S, [_pc(A, 0, relay=True, sinr=0.2, share=0.4, tx=0.9)], 1)
    latch.ingest(S, [_pc(C, 5, relay=False, sinr=0.6, share=0.0, tx=0.3)], 6)
    nb = PC.fill_block_b(NeighborRadioSummary(), latch.fresh(S, 6),
                         latch.arrived(S, 6))
    assert nb.postcard_received == 1.0
    assert nb.neighbor_count_norm == 0.2
    assert abs(nb.avg_sinr - 0.4) < 1e-12 and nb.best_relay_sinr == 0.6
    assert abs(nb.avg_prb_util - 0.2) < 1e-12     # mean relay share reported
    assert nb.any_relay_active == 1.0 and nb.coordination_quality == 0.5
    assert abs(nb.neighbour_avg_tx_power_norm - 0.6) < 1e-12
    assert nb.interferer_count_norm == 0.5
    assert nb.postcard_freshness_norm == 1.0      # the freshest entry is new
    # A's entry (heard at 1) expires at 31, C's (heard at 6) at 36
    nb = PC.fill_block_b(NeighborRadioSummary(), latch.fresh(S, 31),
                         latch.arrived(S, 31))
    assert nb.neighbor_count_norm == 0.1 and nb.any_relay_active == 0.0


_SCHEMA_ASSIGN = re.compile(r'^[ \t]*_BLOCK_B_FIELDS[ \t]*(:[^=\n]*)?=', re.M)


def test_merge_check_the_actor_reads_freshness_only_through_the_trackagent_schema():
    """MERGE CHECK (D5.11).  On master the actor's Block B is still the 16
    pre-WP5 columns of observation_to_tensor: postcard_freshness_norm is
    written by the latch but NOT read by the actor, and the five never-written
    columns are read at their defaults.  Freshness reaches the actor only with
    branch trackAgent's schema (`_BLOCK_B_FIELDS`, WP6).  So:

      * no `_BLOCK_B_FIELDS` in the tree (master): the actor vector must not
        move with postcard_freshness_norm, OBS_DIM is the master 60, and the
        open item is recorded in docs/DECISIONS.md;
      * `_BLOCK_B_FIELDS` present (after the merge): it must carry
        b_neighbour_freshness sourced from postcard_freshness_norm, and
        exactly one actor column must move with the field -- by identity, so
        at latch age 3 the column reads 1 - 3/30 = 0.9 (the trackAgent sign),
        not 3/30."""
    import glob
    from sixg_sim.agent import NeighborRadioSummary, OBS_DIM
    schema = None
    for path in sorted(glob.glob(os.path.join(ROOT, 'sixg_sim', '**', '*.py'),
                                 recursive=True)):
        with open(path, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        if _SCHEMA_ASSIGN.search(text):
            schema = (os.path.relpath(path, ROOT), text)
            break
    sim = _sim(_star_topology())
    obs = sim._build_agent_observations()[S]
    agent = sim.agents[S]
    latch = PC.NeighbourLatch()
    latch.ingest(S, [_pc(A, 0)], 0)
    nb = PC.fill_block_b(NeighborRadioSummary(), latch.fresh(S, 3),
                         latch.arrived(S, 3))
    assert abs(nb.postcard_freshness_norm - 0.9) < 1e-12      # 1 - 3/30

    def actor(fresh):
        o = dataclasses.replace(obs, neighbor_radio=dataclasses.replace(
            nb, postcard_freshness_norm=fresh))
        return agent.observation_to_tensor(o)

    x0, x1 = actor(0.0), actor(nb.postcard_freshness_norm)
    moved = [i for i in range(len(x0)) if float(x0[i]) != float(x1[i])]
    if schema is None:
        assert OBS_DIM == 60, OBS_DIM
        assert moved == [], (
            'the actor now reads postcard_freshness_norm (columns %r) but the '
            'tree has no _BLOCK_B_FIELDS: if this is the trackAgent merge, '
            'the schema must be found here; if not, D5.11 is out of date'
            % moved)
        dec = open(os.path.join(ROOT, 'docs', 'DECISIONS.md'),
                   encoding='utf-8').read()
        assert 'MERGE CHECK (D5.11)' in dec
        return
    rel, text = schema
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if 'b_neighbour_freshness' in l]
    assert hits, '%s: _BLOCK_B_FIELDS carries no b_neighbour_freshness' % rel
    assert any('postcard_freshness_norm' in ' '.join(lines[i:i + 2])
               for i in hits), (
        '%s: b_neighbour_freshness is not sourced from '
        'postcard_freshness_norm' % rel)
    assert len(moved) == 1, moved
    assert float(x0[moved[0]]) == 0.0
    assert abs(float(x1[moved[0]]) - 0.9) < 1e-6, float(x1[moved[0]])


# ----------------------------------------------------------------------
# 6. the send invariant
# ----------------------------------------------------------------------
def test_the_audit_raises_on_two_sends_by_one_site_in_one_tick():
    audit = PC.PostcardSendAudit()
    try:
        audit.record(4, True, [S], {S: 1}, {S: 3})
    except PC.PostcardInvariantError as e:
        assert 'at most one per site per tick' in str(e)
    else:
        raise AssertionError('two sends in one tick passed')


def test_the_audit_requires_more_than_one_send_per_site_after_the_event():
    audit = PC.PostcardSendAudit()
    cum = {}
    for t in range(40):
        before = dict(cum)
        if t in (0, 10, 20, 30):
            cum[S] = cum.get(S, 0) + 1
        if t == 0:
            cum[A] = cum.get(A, 0) + 1         # A sends once, then falls silent
        audit.record(t, True, [S, A], before, cum)
    rep = audit.report()
    assert rep['applicable'] and rep['sites_checked'] == 2
    assert rep['failing_sites'] == [A] and rep['max_per_tick'] == 1
    try:
        audit.check()
    except PC.PostcardInvariantError as e:
        assert 'O-DU_2=1' in str(e)
    else:
        raise AssertionError('a one-postcard site passed')


def test_the_audit_is_not_vacuous():
    # acting ticks but no site ever present: raises, never a silent pass
    audit = PC.PostcardSendAudit()
    for t in range(20):
        audit.record(t, True, [], {}, {})
    try:
        audit.check()
    except PC.PostcardInvariantError as e:
        assert 'nothing was checked' in str(e)
    else:
        raise AssertionError('an audit that checked nothing passed')
    # an arm whose action step never ran (routing, SDN) is NOT applicable
    audit = PC.PostcardSendAudit()
    for t in range(50):
        audit.record(t, False, [S, A], {}, {})
    assert audit.check()['applicable'] is False
    # ... and a send on such a tick is itself a violation
    try:
        audit.record(50, False, [S], {}, {S: 1})
    except PC.PostcardInvariantError:
        pass
    else:
        raise AssertionError('a send outside the action step passed')
    # a site present for one period or less owes one heartbeat: listed only
    audit = PC.PostcardSendAudit()
    cum = {}
    for t in range(30):
        before = dict(cum)
        if t in (0, 26):
            cum[S] = cum.get(S, 0) + 1          # S sends at 0 and 26
        audit.record(t, True, [S] + ([A] if t >= 25 else []), before, cum)
    rep = audit.check()                        # A was present 5 ticks, silent
    assert rep['sites_short_presence'] == [A] and rep['min_per_site'] == 2
    assert rep['sites_checked'] == 1


# ----------------------------------------------------------------------
# 7. the heartbeat is an environment rule
# ----------------------------------------------------------------------
def _heartbeat_run(configure, ticks=35):
    sim = _sim(_acting_topology(), ticks=ticks + 5)
    sim._detect_island_mode = lambda: True
    configure(sim)
    sends = {}
    observed = {}

    def act(tick, obs):
        observed[tick] = obs
        sim._execute_agent_actions(obs)

    for t in range(ticks):
        before = dict(sim._postcards_by_site)
        sim.step_tick(t, act)
        for s, n in sim._postcards_by_site.items():
            if n - before.get(s, 0):
                sends.setdefault(s, []).append(t)
    return sim, sends, observed


def test_the_heartbeat_is_the_same_environment_rule_for_every_agent_arm():
    """No head output adds or suppresses a postcard: the MARL executor with
    EVERY head pinned (WP7 deleted the postcard head itself at the trackAgent
    merge, and pinning a head that does not exist is refused), the
    uniform-random control and XL-DET all send exactly at 1, 11, 21, 31:
    the core-less island's own flags are set by the tick-0 step, so every
    site's action lock opens at tick 1 (D13.2)."""
    def pinned(sim):
        from sixg_sim.agent import DISCRETE_HEADS
        sim.pinned_action_heads = set(DISCRETE_HEADS) | {'prb'}

    def rand(sim):
        sim.random_action_policy = True

    def xldet(sim):
        sim.heuristic_action_policy = True

    for name, cfg in (('marl_pinned', pinned), ('random', rand),
                      ('xldet', xldet), ('marl', lambda sim: None)):
        sim, sends, _ = _heartbeat_run(cfg)
        assert set(sends) == set(sim.agents), (name, sends)
        for s in sim.agents:
            assert sends[s] == [1, 11, 21, 31], (name, s, sends[s])
        rep = sim.postcard_audit.check()
        assert rep['applicable'] and rep['min_per_site'] == 4, (name, rep)
        assert rep['max_per_tick'] == 1
        assert sim.control_plane.dcc.total_ledger['refused'] == 0, name


def test_a_heartbeat_stamp_from_a_later_tick_does_not_silence_a_site():
    """An agent object carried into a new Simulator (or episode) whose ticks
    restart at 0 still holds the last_postcard_tick of the run it came from.
    The due-test used to read `now - last < POSTCARD_PERIOD`, so a stamp in
    the FUTURE counted as 'not due' and the site stayed silent until the old
    tick was passed; the per-pass audit would then have aborted the run.  A
    future stamp now counts as due: the site keeps the 1, 11, 21, 31
    schedule (its action lock opens at tick 1, D13.2)."""
    def carried_over(sim):
        for ag in sim.agents.values():
            ag.last_postcard_tick = 500

    sim, sends, _ = _heartbeat_run(carried_over)
    assert set(sends) == set(sim.agents), sends
    for s in sim.agents:
        assert sends[s] == [1, 11, 21, 31], (s, sends[s])
    rep = sim.postcard_audit.check()
    assert rep['min_per_site'] == 4 and rep['max_per_tick'] == 1, rep


def test_block_b_is_live_between_heartbeats_on_the_real_path():
    """Postcards sent at tick 1 (the first acting tick, D13.2) arrive at
    tick 2; Block B holds them over the next ticks (the 1-in-10 square wave
    a feed-forward actor would otherwise see) while postcard_received marks
    only the arrival tick."""
    sim, sends, observed = _heartbeat_run(lambda sim: None, ticks=15)
    assert 0 not in observed, 'a site acted at tick 0, before its own flag'
    nb2 = observed[2][A].neighbor_radio
    nb6 = observed[6][A].neighbor_radio
    assert nb2.postcard_received == 1.0 and nb6.postcard_received == 0.0
    # A hears S and C (its two agent neighbours)
    assert nb2.neighbor_count_norm == 0.2 == nb6.neighbor_count_norm
    assert nb6.postcard_freshness_norm == 1 - 4 / 30
    assert observed[12][A].neighbor_radio.postcard_received == 1.0


# ----------------------------------------------------------------------
# 8. no Block E reference survives anywhere
# ----------------------------------------------------------------------
BLOCK_E_IDENTIFIERS = (
    'foreign_fragments_norm', 'foreign_unbridged', 'fragment_id',
    'holds_bridge', '_apply_fragment_cue', '_fragment_label_of',
    '_fragment_labels_now', 'FRAG_CUE_LATCH_TICKS', '_frag_cue_latch',
    'POSTCARD_FRAGMENTS', 'LearningPostcard', 'learning_exchanger',
    'learning_postcard',
    # The removed census FIELD of ControlPostcard, as a bare identifier: an
    # assignment (`isolated_ue_count=`), an attribute read
    # (`pc.isolated_ue_count`) and a defaulted read
    # (`getattr(pc, 'isolated_ue_count', 0)`, which would silently return 0)
    # are all caught.  The word boundaries on BOTH sides leave the two census
    # names that legitimately remain, listed in CENSUS_ALLOWED.
    'isolated_ue_count',
)
# The only census identifiers that may remain, each named with its reason.
# The bare-identifier pattern does not match them; the scan asserts that it
# does not (so the allowance is explicit, not an accident of the regex) and
# that each still exists (so the allowance is not stale).
CENSUS_ALLOWED = {
    'isolated_ue_count_norm': (
        'sixg_sim/agent.py',
        'ConnectivityState: the census attribute that fed the critic-only '
        'column g_isolated_ue_count (masked for the actor; the column was '
        'removed at WP13, D13.3, as constant 0.0 -- the attribute is still '
        'computed)'),
    '_isolated_ue_count': (
        'sixg_sim/simulation.py',
        'Simulator: the census value that column is computed from'),
}
ALL = '*'
# (path, identifiers allowed there, reason).  Every exemption is named.
EXEMPT = {
    'sixg_sim/agent.py': (
        {'POSTCARD_FRAGMENTS'},
        'the import-time tripwire that REFUSES the removed switch'),
    'tools/run_invariance_pass.sh': (
        {'POSTCARD_FRAGMENTS'},
        'a comment recording that the launcher no longer sets the switch'),
    'test_mission_config.py': (
        ALL, 'asserts that the switch is refused and the fields are gone'),
    'tests/test_wp5_postcards.py': (ALL, 'this scan'),
    'confirmatory/run_amendment1.sh': (
        {'POSTCARD_FRAGMENTS'},
        'the frozen A1 chain launcher: cd ~/MARL_run_6, verifies '
        'FREEZE_A1.sha256 and runs the frozen tree, never this one'),
    'confirmatory/FREEZE_A1.sha256': (
        {'learning_postcard'}, 'the frozen A1 tree hash list'),
    'smoke2.sh': (
        {'POSTCARD_FRAGMENTS'},
        'A1 smoke launcher: cd ~/MARL_run_6, runs the frozen tree'),
    'smoke_tests.sh': (
        {'POSTCARD_FRAGMENTS'},
        'A1 smoke launcher: cd ~/MARL_run_6, runs the frozen tree'),
    # trackAgent merge: the frozen shared-file declaration of that branch
    # names the one module path it edited (one line) before WP5 deleted it.
    'tools/shared_file_edits.py': (
        {'learning_postcard'},
        'generator of the FROZEN trackAgent shared-file declaration (pinned '
        'at bde51a9): its per-file reason table names the deleted module path'),
    'tests/data/shared_file_edits_bde51a9.tsv': (
        {'learning_postcard'},
        'the FROZEN hunk table of branch trackAgent at bde51a9 (the path it '
        'edited; the module is deleted in the merged tree)'),
}


def _tree_files():
    """The tree's tracked files: `git ls-files` where there is history (the
    laptop), else the sync manifest `.synced_files` the server tree carries
    (tools: scratchpad sync_run9.sh)."""
    try:
        out = subprocess.run(['git', 'ls-files'], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        names = [l for l in out.stdout.decode('utf-8').splitlines() if l]
        if out.returncode == 0 and names:
            return names
    except OSError:
        pass
    manifest = os.path.join(ROOT, '.synced_files')
    with open(manifest, encoding='utf-8') as fh:
        return [l.strip() for l in fh if l.strip()]


def _block_e_patterns():
    """One pattern per identifier, word boundaries on BOTH sides."""
    return {i: re.compile(r'(?<![A-Za-z0-9_])' + re.escape(i)
                          + r'(?![A-Za-z0-9_])')
            for i in BLOCK_E_IDENTIFIERS}


def test_the_block_e_scan_catches_reads_of_the_removed_census_field():
    """The scan used to look for `isolated_ue_count=` only, so a READ of the
    removed ControlPostcard field -- in particular the defaulted
    getattr(pc, 'isolated_ue_count', 0), which returns 0 without failing --
    passed it.  Self-check of the pattern on both sides."""
    pat = _block_e_patterns()['isolated_ue_count']
    for caught in ('isolated_ue_count=3', 'x = pc.isolated_ue_count',
                   "getattr(pc, 'isolated_ue_count', 0)",
                   'd["isolated_ue_count"]', 'isolated_ue_count'):
        assert pat.search(caught), caught
    for name, (rel, _why) in CENSUS_ALLOWED.items():
        assert not pat.search('x = obj.%s + 1' % name), name
        text = open(os.path.join(ROOT, rel), encoding='utf-8',
                    errors='replace').read()
        assert re.search(r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])'
                         % re.escape(name), text), (
            'CENSUS_ALLOWED is stale: %s no longer appears in %s' % (name, rel))


def test_no_block_e_reference_survives_anywhere():
    names = _tree_files()
    pats = _block_e_patterns()
    scanned, exempt_hits, bad = 0, 0, []
    for rel in names:
        if rel.startswith('docs/'):
            continue                   # the decision record is history
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path) or os.path.getsize(path) > 8 * 2 ** 20:
            continue
        try:
            text = open(path, encoding='utf-8').read()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        allowed = EXEMPT.get(rel, (set(), ''))[0]
        for ident, pat in pats.items():
            n = len(pat.findall(text))
            if not n:
                continue
            if allowed == ALL or ident in allowed:
                exempt_hits += n
            else:
                bad.append('%s: %s x%d' % (rel, ident, n))
    assert not bad, 'Block E / fragment-label / exchanger references:\n  ' + '\n  '.join(bad)
    # the scan really read the tree: many files, and it SAW the tripwire
    assert scanned > 80, scanned
    assert exempt_hits > 0
    src = open(os.path.join(ROOT, 'sixg_sim', 'agent.py'), encoding='utf-8').read()
    assert 'REMOVED by RERUN_SPEC WP5' in src
    # every exemption still names a file that exists
    for rel in EXEMPT:
        assert rel in names, rel


def test_the_actor_vector_and_the_postcard_carry_no_block_e():
    from sixg_sim import agent as A_
    from sixg_sim.agent import (OBS_DIM, ACTOR_OBS_DIM, ACTOR_OBS_NAMES,
                                ControlPostcard, NeighborRadioSummary)
    # WP6 (trackAgent merge): the actor is the 26-field schema; Block E is
    # not among its names
    assert OBS_DIM == ACTOR_OBS_DIM == 26
    assert not [n for n in ACTOR_OBS_NAMES
                if n.startswith('e_') or 'foreign' in n or 'fragment' in n]
    assert not hasattr(A_, 'POSTCARD_FRAGMENTS')
    pc = {f.name for f in dataclasses.fields(ControlPostcard)}
    assert not {'fragment_id', 'holds_bridge'} & pc
    nb = {f.name for f in dataclasses.fields(NeighborRadioSummary)}
    assert not {n for n in nb if n.startswith('foreign')}
    from sixg_sim.simulation import Simulator
    for gone in ('_fragment_labels_now', '_fragment_label_of',
                 '_apply_fragment_cue', 'FRAG_CUE_LATCH_TICKS'):
        assert not hasattr(Simulator, gone), gone
    sim = _sim(_star_topology())
    assert not hasattr(sim, 'learning_exchanger')
    obs = sim._build_agent_observations()
    x = sim.agents[S].observation_to_tensor(obs[S])
    assert tuple(x.shape) == (ACTOR_OBS_DIM,)


def test_the_removed_switches_are_refused():
    """MARL_POSTCARD_FRAGMENTS=1 and MARL_POSTCARD_DRAIN=legacy are refused
    at import; MARL_POSTCARD_ALWAYS=0 (the heartbeat 'off') is refused at
    Simulator construction.  `legacy` drained every queue for the
    coordinator before the receiver latch, so Block B was inert again while
    sends still happened -- PostcardSendAudit could not have seen it."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('MARL_POSTCARD_FRAGMENTS', 'MARL_POSTCARD_DRAIN')}
    env['MARL_POSTCARD_FRAGMENTS'] = '1'
    r = subprocess.run([sys.executable, '-c', 'import sixg_sim.agent'],
                       cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    assert r.returncode != 0
    assert b'REMOVED by RERUN_SPEC WP5' in r.stderr
    env.pop('MARL_POSTCARD_FRAGMENTS')
    for drain, refused in (('legacy', True), ('shared', False)):
        r = subprocess.run(
            [sys.executable, '-c', 'import sixg_sim.simulation'],
            cwd=ROOT, env=dict(env, MARL_POSTCARD_DRAIN=drain),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if refused:
            assert r.returncode != 0, drain
            assert b'MARL_POSTCARD_DRAIN' in r.stderr, r.stderr[-400:]
            assert b'REMOVED by RERUN_SPEC WP5' in r.stderr, r.stderr[-400:]
        else:
            assert r.returncode == 0, (drain, r.stderr[-800:])
    saved = os.environ.get('MARL_POSTCARD_ALWAYS')
    os.environ['MARL_POSTCARD_ALWAYS'] = '0'
    try:
        try:
            _sim(_star_topology())
        except ValueError as e:
            assert 'no \'off\'' in str(e)
        else:
            raise AssertionError('MARL_POSTCARD_ALWAYS=0 was accepted')
    finally:
        if saved is None:
            os.environ.pop('MARL_POSTCARD_ALWAYS', None)
        else:
            os.environ['MARL_POSTCARD_ALWAYS'] = saved


# ----------------------------------------------------------------------
# 9. the bridge-peer planner's re-pointing
# ----------------------------------------------------------------------
def test_repointing_is_on_by_default_and_off_is_the_evaluation_only_sensitivity():
    from sixg_sim.simulation import relay_repoint_setting
    assert relay_repoint_setting({}) == 'fragment_aware'
    assert relay_repoint_setting({'MARL_RELAY_REPOINT': ''}) == 'fragment_aware'
    assert relay_repoint_setting({'MARL_RELAY_REPOINT': 'fragment_aware'}) == 'fragment_aware'
    assert relay_repoint_setting({'MARL_RELAY_REPOINT': 'off'}) == 'off'
    for bad in ('legacy', 'on', '1'):
        try:
            relay_repoint_setting({'MARL_RELAY_REPOINT': bad})
        except ValueError:
            pass
        else:
            raise AssertionError('MARL_RELAY_REPOINT=%r accepted' % bad)
    saved = os.environ.get('MARL_RELAY_REPOINT')
    try:
        os.environ.pop('MARL_RELAY_REPOINT', None)
        assert _sim(_star_topology())._relay_repoint is True
        os.environ['MARL_RELAY_REPOINT'] = 'off'
        assert _sim(_star_topology())._relay_repoint is False
        # the training driver refuses the sensitivity
        import train_on_comparison as T
        try:
            T._refuse_evaluation_only_switches()
        except SystemExit as e:
            assert 'evaluation-only' in str(e)
        else:
            raise AssertionError('the training driver accepted '
                                 'MARL_RELAY_REPOINT=off')
    finally:
        if saved is None:
            os.environ.pop('MARL_RELAY_REPOINT', None)
        else:
            os.environ['MARL_RELAY_REPOINT'] = saved
    assert os.path.exists(os.path.join(ROOT, 'docs', 'BRIDGE_PEER_PLANNER.md'))


# ----------------------------------------------------------------------
# 10. the real drivers
# ----------------------------------------------------------------------
_cache = {}


class _Spy:
    """Wraps DisasterControlChannel.send_postcard for one driver run and
    records (tick, sender, admitted, frames billed, frames recounted)."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        from sixg_sim import control_plane as CP
        self._cls = CP.DisasterControlChannel
        self._orig = self._cls.send_postcard
        spy = self

        def send(dcc, postcard, current_tick, receivers=None):
            expect = _independent_frame_count(dcc.topology, postcard.sender_id)
            before = dcc.total_ledger['frames']
            ok = spy._orig(dcc, postcard, current_tick, receivers)
            spy.calls.append((int(current_tick), postcard.sender_id, ok,
                              dcc.total_ledger['frames'] - before, expect))
            return ok
        self._cls.send_postcard = send
        return self

    def __exit__(self, *exc):
        self._cls.send_postcard = self._orig
        return False

    def check(self, label):
        assert self.calls, '%s: no postcard was sent at all' % label
        refused = [c for c in self.calls if not c[2]]
        assert not refused, ('%s: %d heartbeat(s) refused, e.g. %r -- the '
                             'per-tick counter did not reset'
                             % (label, len(refused), refused[:3]))
        wrong = [c for c in self.calls if c[3] != c[4]]
        assert not wrong, '%s: frames != live incident links: %r' % (label, wrong[:3])
        per_site = {}
        for c in self.calls:
            per_site[c[1]] = per_site.get(c[1], 0) + 1
        assert min(per_site.values()) > 1, (label, per_site)
        per_tick = {}
        for c in self.calls:
            key = (c[0], c[1])
            per_tick[key] = per_tick.get(key, 0) + 1
        assert max(per_tick.values()) == 1, label
        return per_site


def _harness():
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    if not _imported_under_eval_env():
        raise AssertionError(
            'CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'sixg_sim was imported outside the evaluated environment; run '
            'this file through tests/run_tests.py')
    _mint_checkpoint()
    R._apply_runtime_overrides({'total_ticks': TICKS})
    return R


def _imported_under_eval_env():
    """The engine variables are in os.environ and the two import-time
    settings (entropy floor, steering model) have the evaluated values.  The
    reward is frozen since WP8, so it is no longer the proxy (D5.8 used it)."""
    import sixg_sim.mappo_trainer as _T
    import sixg_sim.simulation as _S_
    return (all(os.environ.get(k) == v for k, v in EVAL_ENV.items())
            and _T._RELAY_H_FLOOR == float(EVAL_ENV['MARL_RELAY_ENTROPY_FLOOR'])
            and _S_.STEER_MODEL == EVAL_ENV['MARL_STEER_MODEL'])


def _mint_checkpoint():
    """Mint the schema-current checkpoint once per process (deterministic in
    CKPT_TORCH_SEED; the global torch RNG is left untouched)."""
    if 'ckpt' not in _cache:
        from sixg_sim.agent import mint_schema_current_checkpoint
        mint_schema_current_checkpoint(CKPT, torch_seed=CKPT_TORCH_SEED,
                                       extra={'test': 'wp5_postcards'})
        _cache['ckpt'] = CKPT
    return _cache['ckpt']


def _adapted_actor():
    """The actor weights of the minted checkpoint (the name is historical:
    nothing is adapted since WP6's strict gate)."""
    import torch
    ck = torch.load(_mint_checkpoint(), map_location='cpu', weights_only=False)
    return ck['state_dict']


def test_the_counter_resets_per_tick_under_the_evaluation_harness():
    R = _harness()
    with _Spy() as spy:
        out = R.run_physics_pass('marl_static', SEED, CKPT, 'recovery')
    per_site = spy.check('harness')
    t_sev = R.SEVERANCE_TICK
    by_site = out['postcards_by_site']
    assert len(by_site) == out['n_agent_sites'] > 0
    assert min(by_site.values()) > 1, by_site
    assert set(per_site) == set(by_site)
    rep = out['postcard_audit']
    assert rep['applicable'] and rep['max_per_tick'] == 1
    assert rep['min_per_site'] > 1 and not rep['failing_sites']
    pcs, frames, byts = out['postcards'], out['postcard_frames'], out['postcard_bytes']
    assert len(pcs) == len(frames) == len(byts) == TICKS
    assert sum(pcs) == len(spy.calls) == sum(by_site.values())
    assert sum(pcs[:t_sev]) == 0                      # nobody sends pre-event
    assert sum(1 for x in pcs if x) >= 10             # a heartbeat, not one wave
    assert max(pcs) <= out['n_agent_sites']
    assert sum(frames) == sum(c[3] for c in spy.calls) > 0
    assert all(b == 60 * f for b, f in zip(byts, frames))
    # the link-state service bills exactly the ledger
    oh = out['overhead_by_type']['postcard']  # wp7-scan-ok: WP5 DCC message type, not an action head
    assert all(abs(o - b * 8 / 1e6) < 1e-12 for o, b in zip(oh, byts))
    _cache['harness'] = out


def test_the_counter_resets_per_tick_under_the_training_driver_eval_loop():
    R = _harness()
    import train_on_comparison as T
    with _Spy() as spy:
        res = T.run_eval_episode(_adapted_actor(), SEED, 'recovery', TICKS,
                                 event_seed=0, severance_tick=R.SEVERANCE_TICK,
                                 scenario_seed=SEED, random_seed=SEED)
    spy.check('training driver (run_eval_episode)')
    rep = res['postcard_audit']
    assert rep['applicable'] and rep['min_per_site'] > 1
    assert rep['max_per_tick'] == 1
    # D13.2: the first acting tick is the tick after the first site's OWN
    # flag -- BFD, one coalesced Router-LSA, flooding and SPF after the
    # severance (D13.1 measured 13-14 ticks; the global trigger used to make
    # it the severance tick itself, and per-transition LSAs 31-51 ticks)
    from sixg_sim.link_state_service import BFD_DETECT_TICKS
    onset = rep['onset_tick']
    assert R.SEVERANCE_TICK + BFD_DETECT_TICKS < onset <= R.SEVERANCE_TICK + 30, (
        'first postcard at tick %r, severance at %d' % (onset, R.SEVERANCE_TICK))


def test_the_counter_resets_per_tick_under_the_training_worker_loop():
    """The PPO collection loop itself (run_comparison_worker_episode): the
    episode raises inside on an invariant violation, and the spy shows no
    heartbeat was refused."""
    _harness()
    import train_on_comparison as T
    from sixg_sim.agent import CriticNetwork
    from sixg_sim.reward_spec import KAPPA
    # WP6: the critic takes the 37-wide critic vector (CriticNetwork()'s
    # default); WP8: the worker refuses any mixing weight but the frozen KAPPA.
    args = {'episode_idx': 0, 'topology_seed': 42, 'scenario_kind': 'recovery',
            'episode_ticks': 120, 'event_seed': 7,
            'actor_state_dict': _adapted_actor(),
            'critic_state_dict': CriticNetwork().state_dict(),
            'global_reward_mix': KAPPA, 'torch_seed': 1234}
    with _Spy() as spy:
        res = T.run_comparison_worker_episode(args)
    spy.check('training driver (run_comparison_worker_episode)')
    assert res['severance_tick'] >= 30
