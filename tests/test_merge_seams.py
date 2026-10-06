# -*- coding: utf-8 -*-
"""The five seams of the trackAgent merge (docs/DECISIONS.md, the merge
record), each tested on the code path it wires:

  (a) Block D local isolation reads the WP3 link-state service
      (`isolation_state`, the PREVIOUS step), and an arm with no service
      applies the same rule to the true graph at the same phase; Block C's
      `c_live_infra_neighbours` reads the site's own believed-up adjacencies.
  (b) WP8's churn term is scored on the EXECUTED relay mode: after a WP4
      teardown hold `ps.relay_mode` is the incumbent, not the sampled mode.
  (c) every Block B actor field is fed from WP5's fresh-entry receiver latch
      (the own-SINR-drop field is the site's own measurement, by definition).
  (d) the heartbeat is WP5's environment rule (period, limiter, one send per
      site per tick) and WP7 contributed only the head deletion.
  (e) the strict checkpoint gate is not weakened, and the policy-independent
      arms never need or open a checkpoint.

Small hand-built topologies and one short harness subprocess (e); run with

    PYTHONHASHSEED=0 nice -n 19 python3 tests/run_tests.py merge_seams

Every check counts what it compared and fails when that count is zero.
"""
import os
import pickle
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                               InterfaceType)

CORE = 'AMF_1'


def _link(lid, u, v, cap=500.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL, up=True):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=up)


def _simulator(topology, ticks=200):
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    scenario = Scenario(name='merge_seams', duration_ticks=ticks, events=[],
                        traffic_profiles={})
    sim = Simulator(topology, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 0
    return sim


def _chain_sim():
    """CORE -- O-DU_0 -- O-DU_1 -- O-DU_2 -- O-DU_3 -- O-DU_4."""
    t = Topology()
    t.add_node(Node(id=CORE, node_type=NodeType.AMF))
    for i in range(5):
        t.add_node(Node(id='O-DU_%d' % i, node_type=NodeType.O_DU))
    t.add_link(_link('Core_0', CORE, 'O-DU_0', cap=1000.0))
    for i in range(4):
        t.add_link(_link('Mesh_%d' % i, 'O-DU_%d' % i, 'O-DU_%d' % (i + 1)))
    return _simulator(t)


def _set_link_up(sim, lid, up):
    lk = sim.topology.links[lid]
    lk.is_up = up
    sim.topology.sync_pair(lk.endpoints[0], lk.endpoints[1])


def _run(sim, ticks, forwarding, cuts=(), act_fn=None):
    """Drive `sim` through step_tick; `cuts` = {tick: [link ids]} taken down
    before that tick.  Returns per-tick (observation Block D + neighbour
    count, service/engine state after the step, island flag)."""
    from sixg_sim.agent import local_isolation_state
    cuts = dict(cuts)
    obs_rec, state_rec, island = [], [], []

    def _after_observe(tick, observations):
        obs_rec.append({
            aid: (o.local_isolated, o.ticks_since_local_isolation,
                  int(round(o.connectivity.handover_candidate_count_norm * 10)))
            for aid, o in observations.items()})

    for tick in range(ticks):
        for lid in cuts.get(tick, ()):
            _set_link_up(sim, lid, False)
        sim.step_tick(tick, act_fn, forwarding=forwarding,
                      hooks={'after_observe': _after_observe},
                      where='test/merge_seams')
        state_rec.append({s: local_isolation_state(sim, s) for s in sim.agents})
        island.append(bool(sim.island_mode))
    return obs_rec, state_rec, island


def _assert_obs_reads_previous_step(obs_rec, state_rec, label):
    n = 0
    for tick in range(1, len(obs_rec)):
        for aid, (iso, dt, _nb) in obs_rec[tick].items():
            s_iso, s_since = state_rec[tick - 1][aid]
            assert iso == s_iso, (label, tick, aid, iso, s_iso)
            assert dt == ((tick - s_since) if s_iso else 0), (
                label, tick, aid, dt, s_since)
            n += 1
    assert n > 0, '%s: nothing compared' % label
    assert not any(iso for iso, _, _ in obs_rec[0].values()), (
        '%s: tick 0 has no previous step, the flag must be 0' % label)
    return n


# ══════════════════════════════════════════════════════════════════════
# (a) local isolation and the site's own believed-up neighbours
# ══════════════════════════════════════════════════════════════════════
def test_seam_a_block_d_is_the_services_previous_step():
    from sixg_sim.link_state_service import make_link_state_service
    from sixg_sim.agent import link_state_service
    sim = _chain_sim()
    svc = make_link_state_service(sim)
    assert svc is not None, 'MARL_DETECT_TICKS=0 is set: nothing to test'
    assert link_state_service(sim) is svc
    obs, state, island = _run(sim, 40, svc, cuts={10: ['Core_0']})
    n = _assert_obs_reads_previous_step(obs, state, 'service')
    assert n == 39 * len(sim.agents)
    # the state compared against IS the service's own
    for s in sim.agents:
        assert state[-1][s] == svc.isolation_state(s)
    first_island = island.index(True)
    first_flag = next(t for t in range(40) if any(i for i, _, _ in obs[t].values()))
    assert first_island >= 10
    # the deleted stand-in followed the global trigger and showed the flag at
    # first_island + 1; the service pays BFD detection + installation first
    assert first_flag > first_island + 1, (first_island, first_flag)
    # and every site is isolated once the cut is installed
    assert all(i for i, _, _ in obs[-1].values())


def test_seam_a_no_service_uses_the_true_graph_at_the_same_phase():
    sim = _chain_sim()
    obs, state, island = _run(sim, 20, None, cuts={10: ['Core_0']})
    _assert_obs_reads_previous_step(obs, state, 'true graph')
    # the true graph has no detection delay: the step of tick 10 sees the
    # cut, so the observation of tick 11 shows it, and not before
    assert not any(i for i, _, _ in obs[10].values())
    assert all(i for i, _, _ in obs[11].values())
    assert all(dt == 1 for _, dt, _ in obs[11].values())


def test_seam_a_live_infra_neighbours_are_the_sites_own_believed_adjacencies():
    """O-DU_2 has two infrastructure neighbours.  Mesh_1 (O-DU_1 -- O-DU_2)
    fails before tick 20: the site's own BFD session holds the adjacency for
    BFD_DETECT_TICKS steps, so c_live_infra_neighbours stays 2 for exactly
    that window and then drops to 1.  The true-graph count (the old rule)
    drops at once -- the difference is the WP3 seam D6.4 left open."""
    from sixg_sim.link_state_service import make_link_state_service, BFD_DETECT_TICKS
    sim = _chain_sim()
    svc = make_link_state_service(sim)
    obs, _state, _isl = _run(sim, 30, svc, cuts={5: ['Core_0'], 20: ['Mesh_1']})
    site = 'O-DU_2'
    counts = [obs[t][site][2] for t in range(30)]
    assert counts[19] == 2
    # observations at 20 .. 20+BFD read steps 19 .. 19+BFD, when the
    # session is still believed up
    held = [t for t in range(20, 30) if counts[t] == 2]
    assert held == list(range(20, 20 + BFD_DETECT_TICKS + 1)), counts
    assert counts[20 + BFD_DETECT_TICKS + 1] == 1, counts
    assert svc.believed_neighbours(site) == ['O-DU_3']
    # the same failure on an arm with NO service: the true graph, at once
    sim2 = _chain_sim()
    obs2, _, _ = _run(sim2, 22, None, cuts={5: ['Core_0'], 20: ['Mesh_1']})
    assert obs2[19][site][2] == 2 and obs2[20][site][2] == 1


# ══════════════════════════════════════════════════════════════════════
# (b) the churn term reads the EXECUTED relay mode after a WP4 hold
# ══════════════════════════════════════════════════════════════════════
A0, A1, B0, B1 = 'O-DU_0', 'O-DU_1', 'O-DU_2', 'O-DU_3'


def _bridge_sim(passes):
    """Two fragments joined by one sole-path relay bridge held by A1 in
    CAPACITY_BOOST; the arm has a service (grace counts usable passes) and
    the bridge has been installed for `passes` passes.  The link is made
    the way the Priority-1 formation branch makes one (transport-model
    session, Link, lifetime record, pair sync).  Returns (sim, ps, rid)."""
    from sixg_sim.phy_mac_state import RelayMode
    t = Topology()
    t.add_node(Node(id=CORE, node_type=NodeType.AMF))
    for nid in (A0, A1, B0, B1):
        t.add_node(Node(id=nid, node_type=NodeType.O_DU))
    t.add_link(_link('Mesh_A', A0, A1))
    t.add_link(_link('Mesh_B', B0, B1))
    sim = _simulator(t, ticks=400)
    rid = sim.transport_relay_model.create_link(A1, B0, 30.0, 20.0)
    sim.topology.links[rid] = _link(rid, A1, B0, cap=30.0,
                                    lt=LinkType.TRANSPORT_RELAY)
    sim._record_relay_link_created(rid, sim.topology.links[rid])
    sim.topology.sync_pair(A1, B0)
    sim.topology.invalidate_infrastructure_cache()
    sim.link_state_service = object()        # grace in usable passes (WP4)
    sim.current_tick = 50
    sim._relay_bridge_since[rid] = 0
    sim._relay_route_since[rid] = 40
    sim._relay_link_passes[rid] = passes
    ps = sim.phy_mac_states[A1]
    ps.relay_mode = RelayMode.CAPACITY_BOOST
    ps.relay_link_active = True
    ps.relay_peer_node = B0
    return sim, ps, rid


def _off_action():
    from sixg_sim.agent import PHYMACAction, relay_enum_index
    return PHYMACAction(tx_power_step=2, mcs_emergency_idx=2,
                        mcs_general_idx=2, prb_emergency_frac=0.3,
                        prb_relay_frac=0.0, prb_general_frac=0.7,
                        relay_mode_idx=relay_enum_index(0), relay_head_idx=0)


def test_seam_b_a_wp4_hold_leaves_the_executed_mode_in_ps_relay_mode():
    from sixg_sim.phy_mac_state import RelayMode
    from sixg_sim import reward_spec as RS
    sim, ps, RID = _bridge_sim(passes=1)     # inside the 3-pass grace
    assert sim._relay_teardown_blocked(A1), 'the fixture does not force a hold'
    holds_before = dict(sim._relay_teardown_holds)
    sim._snapshot_relay_mode_flips()          # baseline for the WP4 counter
    flips_before = sim._relay_mode_flips
    action = _off_action()
    assert list(RelayMode)[action.relay_mode_idx] == RelayMode.OFF   # SAMPLED
    sim._step_phy_mac(A1, action)
    # EXECUTED: the hold wrote the incumbent back
    assert ps.relay_mode == RelayMode.CAPACITY_BOOST, ps.relay_mode
    assert RS.executed_relay_mode(ps) == RelayMode.CAPACITY_BOOST
    assert RID in sim.topology.links, 'the held bridge was torn down'
    added = {r: sim._relay_teardown_holds[r] - holds_before.get(r, 0)
             for r in sim.RELAY_HOLD_REASONS}
    assert sum(added.values()) >= 1 and added['grace'] >= 1, added
    # WP4's own executed-flip record agrees: nothing changed
    sim._snapshot_relay_mode_flips()
    assert sim._relay_mode_flips == flips_before
    # and the agent-local churn term is not charged for the hold
    agent = sim.agents[A1]
    agent._prev_executed_relay_mode = RelayMode.CAPACITY_BOOST
    obs = sim._build_agent_observations()[A1]
    agent.calculate_reward(obs, action, obs)
    assert agent._prev_executed_relay_mode == RelayMode.CAPACITY_BOOST
    assert agent._reward_terms['relay_churn'] == 0.0, agent._reward_terms


def test_seam_b_without_a_hold_the_executed_mode_is_the_sampled_one():
    from sixg_sim.phy_mac_state import RelayMode
    from sixg_sim import reward_spec as RS
    sim, ps, RID = _bridge_sim(passes=10)    # grace served, link idle
    assert not sim._relay_teardown_blocked(A1), 'the fixture still holds'
    sim._step_phy_mac(A1, _off_action())
    assert ps.relay_mode == RelayMode.OFF
    assert RS.executed_relay_mode(ps) == RelayMode.OFF
    assert RID not in sim.topology.links, 'an unblocked teardown did not happen'
    assert sim._relay_removed_by_reason.get('mode_change', 0) == 1
    # ... and the churn term IS charged for an executed change
    agent = sim.agents[A1]
    agent._prev_executed_relay_mode = RelayMode.CAPACITY_BOOST
    obs = sim._build_agent_observations()[A1]
    agent.calculate_reward(obs, _off_action(), obs)
    assert agent._reward_terms['relay_churn'] == -RS.FROZEN.local_relay_churn < 0.0


def test_seam_b_the_reward_and_the_flip_counter_read_one_field():
    """Both consumers of 'the executed mode' read PHYMACState.relay_mode:
    reward_spec.executed_relay_mode (WP8) and the WP4 M9 flip snapshot.  So
    the churn the policy is charged for and the churn M9 reports cannot
    disagree about an engine hold."""
    import inspect
    from sixg_sim import reward_spec as RS
    from sixg_sim.simulation import Simulator
    body = inspect.getsource(RS.executed_relay_mode).split('"""')[-1]
    snap = inspect.getsource(Simulator._snapshot_relay_mode_flips).split('"""')[-1]
    assert "'relay_mode'" in body and "'relay_mode'" in snap


# ══════════════════════════════════════════════════════════════════════
# (c) every Block B field comes from WP5's fresh-entry latch
# ══════════════════════════════════════════════════════════════════════
def _pc(sender, relay, sinr, share, tx, unserved=0):
    from sixg_sim.agent import ControlPostcard
    return ControlPostcard(sender_id=sender, relay_mode_active=relay,
                           best_sinr_to_neighbour=sinr, unserved_ue_count=unserved,
                           prb_avail_for_relay=share, policy_version=0,
                           timestamp=0, tx_power_dbm_norm=tx)


def _expected_block_b(fresh, horizon, now, own_drop):
    """Block B computed independently from the list of FRESH (arrival tick,
    postcard) pairs -- the definitions RERUN_SPEC WP6 names."""
    pcs = [pc for _t, pc in fresh]
    n = len(pcs)
    relaying = [pc for pc in pcs if pc.relay_mode_active]
    youngest = min(now - t for t, _ in fresh)
    return {
        'b_neighbour_relay_share': sum(pc.prb_avail_for_relay for pc in pcs) / n,
        'b_neighbour_sinr_mean': sum(pc.best_sinr_to_neighbour for pc in pcs) / n,
        'b_neighbour_sinr_best': max(pc.best_sinr_to_neighbour for pc in pcs),
        'b_fresh_neighbour_count': min(1.0, n / 10.0),
        'b_any_neighbour_relaying': 1.0 if relaying else 0.0,
        'b_neighbour_freshness': 1.0 - youngest / float(horizon),
        'b_relaying_neighbour_fraction': len(relaying) / n,
        'b_neighbour_tx_power_mean': sum(pc.tx_power_dbm_norm for pc in pcs) / n,
        'b_own_sinr_drop': own_drop,
        'b_interferer_fraction': sum(1 for pc in pcs if pc.tx_power_dbm_norm > 0.5) / n,
    }


def test_seam_c_every_block_b_column_is_the_fresh_latch():
    """Through the simulator's own path: the DCC delivers nothing here, the
    postcards are ingested into Simulator._postcard_latch with their arrival
    ticks, and the ACTOR vector's ten Block B columns must equal the values
    computed independently from the FRESH entries only (an entry of age >=
    30 is expired and must not count)."""
    from sixg_sim.agent import _BLOCK_B_FIELDS, ACTOR_OBS_NAMES, actor_feature_values
    from sixg_sim import postcards as PC
    sim = _chain_sim()
    site = 'O-DU_2'
    latch = sim._postcard_latch
    assert latch.horizon == PC.POSTCARD_LATCH_TICKS == 30
    now = 45
    arrivals = [
        (10, _pc('O-DU_0', True, 0.90, 0.80, 0.95)),      # age 35: EXPIRED
        (20, _pc('O-DU_1', True, 0.40, 0.30, 0.70, unserved=2)),   # age 25
        (40, _pc('O-DU_3', False, 0.70, 0.10, 0.20)),     # age 5
        (43, _pc('O-DU_4', False, 0.55, 0.05, 0.60)),     # age 2
    ]
    for t, pc in arrivals:
        latch.ingest(site, [pc], t)
    sim.current_tick = now
    obs = sim._build_agent_observations()[site]
    own_drop = obs.neighbor_radio.experienced_sinr_drop_norm
    vec = dict(zip(ACTOR_OBS_NAMES, actor_feature_values(obs)))
    fresh = [(t, pc) for t, pc in arrivals if now - t < 30]
    assert len(fresh) == 3
    want = _expected_block_b(fresh, 30, now, own_drop)
    names = [n for n, _ in _BLOCK_B_FIELDS]
    assert names == list(want), (names, list(want))
    n = 0
    for name in names:
        assert abs(vec[name] - want[name]) < 1e-12, (name, vec[name], want[name])
        n += 1
    assert n == 10
    # the expired entry really was excluded: with it the best SINR would be 0.9
    assert vec['b_neighbour_sinr_best'] == 0.70
    # postcard_received (XL-DET's contract) is 0: nothing arrived THIS tick
    assert obs.neighbor_radio.postcard_received == 0.0


def test_seam_c_no_block_b_column_is_silently_constant():
    """Each of the nine latch-fed columns MOVES when the latch content moves
    (two different fresh sets), and the tenth (own SINR drop) moves with the
    site's own measurement instead -- so no column is constant by wiring.

    The own drop is the engine's measurement (`_compute_icic_state`: the
    site's SINR at the previous observation minus now, over 10 dB, clamped
    to [0, 1]), so the two inputs set the site's PREVIOUS SINR: none (no
    drop, 0.0) and 1000 dB (a drop far beyond 10 dB, 1.0).  Until WP13 the
    test wrote `sinr_delta`, which the engine overwrites, and its last
    assertion ran only if the two drops happened to differ (D13.5)."""
    from sixg_sim.agent import ACTOR_OBS_NAMES, actor_feature_values, _BLOCK_B_FIELDS
    names = [n for n, _ in _BLOCK_B_FIELDS]

    def vec(arrivals, prev_sinr_db, now=45):
        sim = _chain_sim()
        for t, pc in arrivals:
            sim._postcard_latch.ingest('O-DU_2', [pc], t)
        sim.current_tick = now
        if prev_sinr_db is not None:
            sim._prev_sinr['O-DU_2'] = float(prev_sinr_db)
        obs = sim._build_agent_observations()['O-DU_2']
        return dict(zip(ACTOR_OBS_NAMES, actor_feature_values(obs))), obs

    v1, o1 = vec([(44, _pc('O-DU_1', False, 0.2, 0.0, 0.3))], None)
    v2, o2 = vec([(30, _pc('O-DU_1', True, 0.6, 0.5, 0.8)),
                  (40, _pc('O-DU_3', True, 0.9, 0.3, 0.9))], 1000.0)
    moved = [n for n in names if v1[n] != v2[n]]
    latch_fed = [n for n in names if n != 'b_own_sinr_drop']
    assert set(latch_fed) <= set(moved), sorted(set(latch_fed) - set(moved))
    assert len(latch_fed) == 9
    # the inputs took effect: no drop, then the clamped maximum ...
    assert o1.neighbor_radio.experienced_sinr_drop_norm == 0.0
    assert o2.neighbor_radio.experienced_sinr_drop_norm == 1.0
    # ... and the actor column follows the site's own measurement,
    # unconditionally
    assert v1['b_own_sinr_drop'] == 0.0 and v2['b_own_sinr_drop'] == 1.0
    assert 'b_own_sinr_drop' in moved
    assert len(moved) == 10, moved


# ══════════════════════════════════════════════════════════════════════
# (d) the heartbeat: WP5's rule, WP7's head deletion, nothing else
# ══════════════════════════════════════════════════════════════════════
def test_seam_d_no_postcard_head_and_one_period_definition():
    import glob
    import re
    from sixg_sim import agent as A
    from sixg_sim import postcards as PC
    assert 'postcard' not in A.DISCRETE_HEADS  # wp7-scan-ok: WP5 DCC message type, not an action head
    assert PC.POSTCARD_PERIOD == 10
    assert not hasattr(A, 'POSTCARD_HEARTBEAT_TICKS')
    assert not hasattr(A, 'POSTCARD_FRESHNESS_TICKS')
    pat = re.compile(r'POSTCARD_HEARTBEAT_TICKS|POSTCARD_FRESHNESS_TICKS')
    n = 0
    for p in sorted(glob.glob(os.path.join(ROOT, 'sixg_sim', '**', '*.py'),
                              recursive=True)) + [
            os.path.join(ROOT, 'train_on_comparison.py'),
            os.path.join(ROOT, 'run_timeline_comparison.py')]:
        with open(p, encoding='utf-8', errors='replace') as fh:
            src = fh.read()
        n += 1
        assert not pat.search(src), '%s still names a second period' % p
    assert n > 20
    # the simulator sends the heartbeat in ONE place, after _step_phy_mac
    with open(os.path.join(ROOT, 'sixg_sim', 'simulation.py'),
              encoding='utf-8') as fh:
        sim_src = fh.read()
    body = sim_src.split('    def _execute_agent_actions(')[1].split('\n    def ')[0]
    assert body.count('self._send_postcard_heartbeat(') == 1
    assert body.index('self._step_phy_mac(node_id, action)') < \
        body.index('self._send_postcard_heartbeat(')
    assert 'ControlPostcard(' not in body, 'a postcard is built in the action step'


def test_seam_d_the_action_builders_never_send_a_postcard():
    import torch
    from sixg_sim import agent as A
    sim = _chain_sim()
    sim.island_mode = True
    obs = sim._build_agent_observations()
    n = 0
    for aid, agent in sim.agents.items():
        agent.last_postcard_tick = -100
        logits = {h: torch.zeros(1, k) for h, k in zip(A.DISCRETE_HEADS, A.HEAD_SIZES)}
        logits['prb'] = torch.zeros(1, 3)
        act = agent.action_from_logits_row(logits, 0, obs[aid], False)
        assert act.send_postcard is False and act.postcard_content is None
        act2 = agent.compute_action(obs[aid])
        assert act2.send_postcard is False and act2.postcard_content is None
        assert agent.last_postcard_tick == -100, 'a builder touched the heartbeat stamp'
        n += 1
    assert n == len(sim.agents) > 0


def test_seam_d_the_heartbeat_keeps_wp5s_period_and_limiter():
    """Island from tick 0; the action step runs every tick: every agent site
    sends at 0, 10, 20 and 30 (one per site per tick, the audit agrees), and a
    second call in the same tick is refused by the DCC limiter."""
    sim = _chain_sim()
    _set_link_up(sim, 'Core_0', False)
    sends = {}

    def act(tick, observations):
        before = dict(sim._postcards_by_site)
        sim._execute_agent_actions(observations)
        for s, c in sim._postcards_by_site.items():
            if c > before.get(s, 0):
                sends.setdefault(s, []).append(tick)

    for tick in range(35):
        sim.step_tick(tick, act, forwarding=None, where='test/merge_seams/d')
    assert set(sends) == set(sim.agents) and sends, sends
    onset = min(t for v in sends.values() for t in v)
    for s, ticks in sends.items():
        assert ticks == [onset + 10 * k for k in range(len(ticks))], (s, ticks)
        assert len(ticks) >= 3, (s, ticks)
    rep = sim.postcard_audit.check()
    assert rep['max_per_tick'] == 1 and rep['sites_checked'] == len(sim.agents)
    # the limiter: a second heartbeat attempt in the same tick sends nothing
    site = sorted(sim.agents)[0]
    ag = sim.agents[site]
    ag.last_postcard_tick = -100
    first = sim._send_postcard_heartbeat(site, ag, _off_action())
    ag.last_postcard_tick = -100
    second = sim._send_postcard_heartbeat(site, ag, _off_action())
    assert first is True and second is False


# ══════════════════════════════════════════════════════════════════════
# (e) the strict gate stays strict; policy-independent arms need nothing
# ══════════════════════════════════════════════════════════════════════
def test_seam_e_the_gate_still_refuses_the_pre_schema_checkpoint():
    import torch
    from sixg_sim.agent import assert_checkpoint_schema
    ck = torch.load(os.path.join(ROOT, 'ce_ck_s1234', 'policy_final.pt'),
                    map_location='cpu', weights_only=False)
    try:
        assert_checkpoint_schema(ck, where='merge_seams')
    except ValueError as e:
        assert 'schema' in str(e)
    else:
        raise AssertionError('the WP6 gate accepted a pre-schema checkpoint')


def _harness_module():
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    return R


def test_seam_e_only_checkpoint_arms_resolve_a_checkpoint():
    R = _harness_module()
    indep = ['xldet', 'random', 'sdn', 'ospf', 'olsr', 'batman', 'aodv']
    missing = {'MARL_EVAL_CHECKPOINT': os.path.join(ROOT, 'no', 'such.pt')}
    assert R._resolve_eval_checkpoint(indep, missing) == (None, [])
    assert R._resolve_eval_checkpoint(indep, {}) == (None, [])
    n = 0
    for arm in indep:
        assert R._task_checkpoint(arm, '/x/policy_final.pt') is None, arm
        n += 1
    assert n == 7
    assert set(R.CHECKPOINT_ARMS) == {'marl', 'marl_freeze', 'marl_static'}
    for arm in R.CHECKPOINT_ARMS:
        assert R._task_checkpoint(arm, '/x/p.pt') == '/x/p.pt'
        try:
            R._resolve_eval_checkpoint(['ospf', arm], missing)
        except SystemExit as e:
            assert 'not found' in str(e)
        else:
            raise AssertionError('%s ran without its checkpoint' % arm)
    ck = os.path.join(ROOT, 'ce_ck_s1234', 'policy_final.pt')
    assert R._resolve_eval_checkpoint(['marl_static'],
                                      {'MARL_EVAL_CHECKPOINT': ck}) == (ck, [ck])


def test_seam_e_policy_independent_arms_run_with_no_usable_checkpoint():
    """End to end, through the harness's own main(): xldet and random with
    MARL_EVAL_CHECKPOINT naming a file that does not exist.  Before the merge
    fix main() refused to start; if any pass opened the path it would fail.
    Short passes (120 ticks, one seed), the evaluated environment."""
    out_rel = os.path.join('tmp_tests', 'merge_seams_e.pkl')
    out_abs = os.path.join(ROOT, 'output', out_rel)
    if os.path.exists(out_abs):
        os.remove(out_abs)
    env = {k: v for k, v in os.environ.items() if not k.startswith('MARL_')}
    env.update({
        'MARL_TRANSPORT_POWER': 'planning', 'MARL_RELAY_STATE': 'reconcile',
        'MARL_RELAY_REPOINT': 'fragment_aware', 'MARL_POSTCARD_ALWAYS': '1',
        'MARL_REWARD_PROFILE': 'mission', 'MARL_RELAY_ENTROPY_FLOOR': '0.01',
        'MARL_STEER_MODEL': 'actionable', 'PYTHONHASHSEED': '0',
        'MARL_POOL_WORKERS': os.environ.get('MARL_POOL_WORKERS', '4'),
        'MARL_EVAL_CHECKPOINT': os.path.join(ROOT, 'no', 'such', 'policy.pt'),
    })
    log = os.path.join(ROOT, 'output', 'tmp_tests', 'merge_seams_e.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w', encoding='utf-8') as fh:
        rc = subprocess.call(
            [sys.executable, 'run_timeline_comparison.py', '--seed-list', '82',
             '--ticks', '120', '--arms', 'xldet,random', '--no-figures',
             '--out', out_rel, '--force-overwrite'],
            cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    text = open(log, encoding='utf-8', errors='replace').read()
    assert rc == 0, 'harness failed (rc=%d); see %s' % (rc, log)
    assert 'Checkpoint: none' in text, 'the harness resolved a checkpoint'
    with open(out_abs, 'rb') as fh:
        P = pickle.load(fh)
    n = 0
    for sk in ('recovery_multi', 'rescue_multi'):
        for arm in ('xldet', 'random'):
            assert arm in (P.get(sk) or {}), (sk, arm, list(P.get(sk) or {}))
            n += 1
    assert n == 4


# ══════════════════════════════════════════════════════════════════════
# the merged engine REFUSES these settings at import or construction; no
# launcher, stage environment or test environment may still make them
# ══════════════════════════════════════════════════════════════════════
def _refused(name, value):
    v = str(value).strip().lower()
    return {
        'MARL_POSTCARD_FRAGMENTS': v in ('1', 'true', 'on', 'yes'),
        'MARL_POSTCARD_ALWAYS': v in ('0', 'false', 'off', 'no'),
        'MARL_POSTCARD_DRAIN': v == 'legacy',
        'MARL_OBS_LOCALITY': True,
        'MARL_RELAY_MASK': v != 'off',
        'MARL_REWARD_PROFILE': v not in ('', 'mission'),
    }.get(name, False)


# Files that make a refused setting ON PURPOSE, each with its reason.
_REFUSED_SETTING_EXEMPT = {
    'confirmatory/run_amendment1.sh':
        'the frozen A1 chain launcher: cd ~/MARL_run_6 and runs that frozen '
        'tree, never this one (editing it would break FREEZE_A1.sha256)',
    'smoke2.sh': 'A1 smoke launcher: runs the frozen ~/MARL_run_6 tree',
    'smoke_tests.sh': 'A1 smoke launcher: runs the frozen ~/MARL_run_6 tree',
    'test_mission_config.py':
        'sets MARL_POSTCARD_FRAGMENTS=1 in a subprocess to prove the refusal',
    'tests/test_wp5_postcards.py':
        'sets each removed postcard switch in a subprocess to prove the refusal',
    'test_relay_mask.py':
        'sets MARL_RELAY_MASK in a subprocess to prove the refusal',
    'tests/test_merge_seams.py':
        'this scan: its refusal table (_refused) names every switch',
}


def _refused_settings(rel, text):
    """[(line, name, value)] of every refused setting `rel` makes.  Python:
    AST only (dict literal, keyword argument, env[...] = ...), so prose in a
    comment or docstring is not a setting.  Shell: non-comment lines."""
    import ast
    import re
    names = ('MARL_POSTCARD_FRAGMENTS', 'MARL_POSTCARD_ALWAYS',
             'MARL_POSTCARD_DRAIN', 'MARL_OBS_LOCALITY', 'MARL_RELAY_MASK',
             'MARL_REWARD_PROFILE')
    hits = []
    if rel.endswith('.py'):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Dict):
                pairs = list(zip(node.keys, node.values))
            elif isinstance(node, ast.Call):
                pairs = [(ast.Constant(kw.arg), kw.value) for kw in node.keywords
                         if kw.arg]
            elif isinstance(node, ast.Assign):
                pairs = [(t.slice, node.value) for t in node.targets
                         if isinstance(t, ast.Subscript)]
            for k, v in pairs:
                if (isinstance(k, ast.Constant) and k.value in names
                        and isinstance(v, ast.Constant)
                        and _refused(k.value, v.value)):
                    hits.append((getattr(v, 'lineno', 0), k.value, v.value))
    else:
        pat = re.compile(r'\b(%s)=([^\s"\';]*)' % '|'.join(names))
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('#'):
                continue
            for m in pat.finditer(line):
                if _refused(m.group(1), m.group(2)):
                    hits.append((i, m.group(1), m.group(2)))
    return hits


def test_no_launcher_or_environment_sets_a_refused_switch():
    """The Block E scan in test_wp5_postcards matches identifiers with word
    boundaries, so it cannot see `MARL_POSTCARD_FRAGMENTS=1` -- which is how
    the run chain's POLICY_ENV (confirmatory/chain_plan.py), the WP12
    launcher and two WP6/WP7 test environments still set a switch the merged
    engine refuses at import.  This scan looks for the SETTING."""
    from sixg_sim import runinfo
    listed, _src = runinfo._tree_files(ROOT)
    assert listed, 'no tracked-file list (git ls-files / .synced_files)'
    scanned, bad = 0, []
    exempt_hits = dict.fromkeys(_REFUSED_SETTING_EXEMPT, 0)
    for rel in sorted(set(f.replace(chr(92), '/') for f in listed)):
        if not rel.endswith(('.py', '.sh')) or rel.startswith('output/'):
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        scanned += 1
        for line, name, value in _refused_settings(rel, text):
            if rel in _REFUSED_SETTING_EXEMPT:
                exempt_hits[rel] += 1
            else:
                bad.append('%s:%d %s=%s' % (rel, line, name, value))
    assert not bad, 'refused settings are still made: %r' % bad
    assert scanned > 80, scanned
    # the scan can see what it looks for: the exempted files really do it
    stale = sorted(r for r, n in exempt_hits.items() if n == 0)
    assert not stale, ('exemptions for files that make no refused setting '
                       '(drop them): %r' % stale)
    # and it would have caught the defect this merge fixed
    assert _refused_settings('x.py', "E = {'MARL_POSTCARD_FRAGMENTS': '1'}")
    assert _refused_settings('x.sh', 'env MARL_POSTCARD_FRAGMENTS=1 python3 a.py')
    assert not _refused_settings('x.sh', '# MARL_POSTCARD_FRAGMENTS=1 was removed')
