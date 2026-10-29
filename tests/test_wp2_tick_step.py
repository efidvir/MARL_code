"""WP2 -- one per-tick step for training and evaluation (RERUN_SPEC WP2).

Fast checks (hand-built topologies, seconds): the fixed phase order of
Simulator.step_tick and its hook points; the pre-isolation action lock; the
shared backhaul-degradation helper; the end-of-pass per-link load snapshot
that feeds the two Block A backhaul fields; the action signature; the
detector's move into the engine; and a source scan that both drivers step
the simulator only through step_tick.

The 300-tick parity test, the Block A variation test and the training smoke
run against the comparison environment and live in test_wp2_parity.py.

Runs under pytest and under tests/run_tests.py.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                               InterfaceType)
from sixg_sim import tick_step as TS


# ----------------------------------------------------------------------
# helpers (same tiny topology family as the WP1 tests)
# ----------------------------------------------------------------------

A = 'O-DU_18'
B = 'Relay_7'
C = 'EdgeUPF_3'
UE = 'UE_42'


def _link(lid, u, v, cap=100.0, lt=LinkType.MULTIHAUL_MESH,
          iface=InterfaceType.BACKHAUL, up=True):
    return Link(id=lid, endpoints=(u, v), capacity=cap, latency=1,
                link_type=lt, interface_type=iface, is_up=up)


def _sim():
    """A Simulator over a small hand-built topology: {A, B} joined by a
    physical mesh hop, {C} alone, one UE camped on A."""
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    t = Topology()
    for nid, ntype in ((A, NodeType.O_DU), (B, NodeType.RELAY),
                       (C, NodeType.EDGEUPF), (UE, NodeType.UE)):
        t.add_node(Node(id=nid, node_type=ntype))
    t.nodes[A].has_multihaul = True
    t.nodes[B].has_multihaul = True
    t.nodes[C].has_multihaul = True
    t.add_link(_link('Backhaul_9', A, B, cap=800.0))
    t.add_link(_link('Microwave_1', B, C, cap=200.0, lt=LinkType.MICROWAVE_PTP))
    t.add_link(_link('Uu_1', UE, A, cap=100.0, lt=LinkType.WIRELESS,
                     iface=InterfaceType.Uu))
    scenario = Scenario(name='wp2', duration_ticks=10, events=[],
                        traffic_profiles={})
    sim = Simulator(t, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 0
    return sim


def _src(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


class _Recorder:
    """Replaces the engine phases on one Simulator instance with
    trace-recording stubs, so the ORDER of the step can be asserted without
    running any physics."""

    def __init__(self, sim, island, unlocked=None, observations=None):
        self.trace = []
        self.island = island
        self.sim = sim
        tr = self.trace
        # D13.2: the per-site action lock.  By default every agent site's own
        # flag is set exactly when the global trigger is (the order tests
        # below are about the phases, not about which site acts); a test
        # about the per-site lock passes its own `unlocked`.
        if unlocked is None:
            unlocked = sorted(sim.agents) if island else []
        self.unlocked = list(unlocked)
        obs = dict(observations or {})
        sim._process_events = lambda t: tr.append('events')
        sim._detect_island_mode = lambda: (tr.append('island'), island)[1]
        sim._build_agent_observations = lambda: (tr.append('observe'),
                                                 dict(obs))[1]
        sim.update_site_action_locks = lambda t: (tr.append('site_locks'),
                                                  list(self.unlocked))[1]
        sim.check_graph_consistency = lambda where='': tr.append('invariant')
        sim._generate_traffic = lambda: (tr.append('generate'), {})[1]
        sim._forward_traffic = lambda traffic: tr.append('forward')
        sim.control_plane.set_island_mode = lambda v: tr.append('set_island')
        sim.topology.invalidate_infrastructure_cache = (
            lambda: tr.append('invalidate'))
        self.saved = (TS.auto_link_adaptation, TS.lock_phy_mac_pre_event,
                      TS.apply_backhaul_degradation)
        TS.auto_link_adaptation = lambda s: tr.append('auto_mcs')
        TS.lock_phy_mac_pre_event = lambda s, sites=None: tr.append(
            'lock' if sites is None else ('lock_sites', tuple(sites)))
        TS.apply_backhaul_degradation = lambda s, p='': tr.append('degrade')

    def restore(self):
        (TS.auto_link_adaptation, TS.lock_phy_mac_pre_event,
         TS.apply_backhaul_degradation) = self.saved


class _FakeView:
    """A forwarding view with the link-state service API (WP3).

    `stale` and `tombstoned` are the two id sets the SERVICE owns and
    step_tick simply installs around the one forwarding call: ids the
    installed view holds as a hop although the Link object is physically
    down, and ids whose Link object has already been popped."""

    def __init__(self, trace, believed=None, stale=(), tombstoned=()):
        self.trace = trace
        self.believed = believed
        self.visible = {}
        self.stale = frozenset(stale)
        self.tombstoned = frozenset(tombstoned)

    def step(self, tick):
        self.trace.append('view_step')

    def get_routable_flows(self):
        self.trace.append('flow_gate')
        return []

    def believed_graph(self):
        self.trace.append('believed')
        return self.believed

    def stale_link_ids(self):
        return self.stale

    def tombstoned_link_ids(self):
        return self.tombstoned


def _hooks(trace):
    return {
        'after_island': lambda t: trace.append('h_after_island'),
        'island_config': lambda t: trace.append('h_island_config'),
        'after_observe': lambda t, o: trace.append('h_after_observe'),
        'before_forward': lambda t: trace.append('h_before_forward'),
    }


# ----------------------------------------------------------------------
# 1. the fixed order and the hook points
# ----------------------------------------------------------------------

def test_step_order_with_the_lock_open():
    sim = _sim()
    rec = _Recorder(sim, island=True)
    try:
        sim.topology.links['Backhaul_9'].current_utilization = 5.0
        view = _FakeView(rec.trace)
        res = sim.step_tick(3, lambda t, o: rec.trace.append('act'),
                            forwarding=view, degrade_tick=3,
                            hooks=_hooks(rec.trace), where='t')
    finally:
        rec.restore()
    assert rec.trace == [
        'events', 'degrade', 'island', 'set_island', 'h_after_island',
        'auto_mcs', 'h_island_config', 'observe', 'h_after_observe',
        'site_locks', 'act', 'invalidate', 'h_before_forward', 'view_step',
        'flow_gate', 'invariant', 'generate', 'believed', 'forward'], rec.trace
    assert sim.current_tick == 3
    # per-tick reset happened before anything else
    assert sim.topology.links['Backhaul_9'].current_utilization == 0.0
    assert res.acted and res.lock_open and not res.believed_view
    assert res.acting_sites == tuple(sorted(sim.agents)) and res.acting_sites
    # the flow gate was restored after forwarding
    assert not hasattr(sim, '_proto_all_flows')


def test_step_order_with_the_lock_closed():
    """Pre-isolation: the fixed pre-event configuration replaces both the
    arm configuration hook and the agents' action; the view is still
    stepped but neither gates flows nor installs a believed graph."""
    sim = _sim()
    rec = _Recorder(sim, island=False)
    try:
        view = _FakeView(rec.trace, believed=object())
        res = sim.step_tick(1, lambda t, o: rec.trace.append('act'),
                            forwarding=view, degrade_tick=7,
                            hooks=_hooks(rec.trace), where='t')
    finally:
        rec.restore()
    assert rec.trace == [
        'events', 'island', 'set_island', 'h_after_island',
        'auto_mcs', 'lock', 'observe', 'h_after_observe', 'site_locks',
        'invalidate', 'h_before_forward', 'view_step',
        'invariant', 'generate', 'forward'], rec.trace
    assert not res.acted and not res.lock_open and not res.believed_view
    assert res.acting_sites == ()


def test_step_order_with_some_sites_still_locked():
    """D13.2: after the global trigger, a site whose OWN lock is still
    closed gets the fixed pre-event configuration at phase 5 and is left out
    of the observations the driver's action step receives; the unlocked
    site acts."""
    sim = _sim()
    sites = sorted(sim.agents)
    assert len(sites) >= 2, sites
    got = []
    rec = _Recorder(sim, island=True, unlocked=sites[1:],
                    observations={s: 'obs-' + s for s in sites})
    try:
        res = sim.step_tick(3, lambda t, o: (rec.trace.append('act'),
                                             got.append(dict(o))),
                            forwarding=_FakeView(rec.trace),
                            hooks=_hooks(rec.trace), where='t')
    finally:
        rec.restore()
    assert rec.trace == [
        'events', 'island', 'set_island', 'h_after_island',
        'auto_mcs', 'h_island_config', 'observe', 'h_after_observe',
        'site_locks', ('lock_sites', (sites[0],)), 'act', 'invalidate',
        'h_before_forward', 'view_step', 'flow_gate', 'invariant',
        'generate', 'believed', 'forward'], rec.trace
    assert got == [{s: 'obs-' + s for s in sites[1:]}], got
    assert res.acted and res.acting_sites == tuple(sites[1:])


def test_believed_graph_is_installed_for_forwarding_only():
    """When the view's belief differs from the truth, forwarding runs on
    the believed graph and the true graph is back afterwards, even if the
    engine raises."""
    sim = _sim()
    rec = _Recorder(sim, island=True)
    true_graph = sim.topology.graph
    seen = []
    sim._forward_traffic = lambda traffic: seen.append(sim.topology.graph)
    try:
        believed = true_graph.copy()
        view = _FakeView(rec.trace, believed=believed)
        view.visible = {'Backhaul_9': True}
        res = sim.step_tick(2, None, forwarding=view)
        assert seen == [believed]
        assert sim.topology.graph is true_graph
        assert res.believed_view
        assert sim._marl_stale_link_ids == frozenset()
        assert sim._marl_tombstone_link_ids == frozenset()

        def boom(traffic):
            raise RuntimeError('engine')
        sim._forward_traffic = boom
        try:
            sim.step_tick(3, None, forwarding=view)
        except RuntimeError:
            pass
        else:
            raise AssertionError('engine error swallowed')
        assert sim.topology.graph is true_graph
    finally:
        rec.restore()


def test_stale_and_tombstone_ids_come_from_the_view(): 
    """WP3: the two id sets consume_path needs are the SERVICE's, not a
    rule step_tick re-derives.  They are installed around the one
    forwarding call and cleared afterwards, so every other arm -- and the
    TRUE graph -- sees empty sets."""
    sim = _sim()
    rec = _Recorder(sim, island=True)
    seen = {}
    sim._forward_traffic = lambda traffic: seen.update(
        stale=sim._marl_stale_link_ids,
        tombstoned=sim._marl_tombstone_link_ids)
    try:
        sim.topology.links['Microwave_1'].is_up = False
        view = _FakeView(rec.trace, believed=sim.topology.graph.copy(),
                         stale={'Microwave_1'}, tombstoned={'TR_gone_1'})
        sim.step_tick(2, None, forwarding=view)
    finally:
        rec.restore()
    assert seen['stale'] == frozenset({'Microwave_1'})
    assert seen['tombstoned'] == frozenset({'TR_gone_1'})
    assert sim._marl_stale_link_ids == frozenset()
    assert sim._marl_tombstone_link_ids == frozenset()


# ----------------------------------------------------------------------
# 2. the pre-isolation action lock (WP2.5) and its WP6 hook
# ----------------------------------------------------------------------

def test_the_global_trigger_is_not_the_agents_action_lock():
    """D13.2 (RERUN_SPEC WP2 step 5, reviewer item 6): the GLOBAL island
    trigger still decides the arms without agents -- the pre-event
    configuration of every radio site until it fires, the baseline arms' own
    configuration after -- but it is no longer what lets an agent act.  The
    docstring says so, and step_tick reaches the driver's action step only
    through the per-site lock."""
    sim = _sim()
    sim.island_mode = False
    assert sim.action_lock_open() is False
    sim.island_mode = True
    assert sim.action_lock_open() is True
    src = _src('sixg_sim', 'simulation.py')
    body = src[src.index('def action_lock_open'):
               src.index('def site_action_unlocked')]
    assert "NOT the agents' action lock" in body and 'D13.2' in body
    assert 'NOT switched over' not in body and 'WP6 HOOK' not in body
    step = _src('sixg_sim', 'tick_step.py')
    body = step[step.index('def step_tick('):]
    calls = [l.strip() for l in body.splitlines()
             if l.strip().startswith('act_fn(tick,')
             and not l.strip().startswith('act_fn(tick, observations)  ')]
    assert calls == ['act_fn(tick, {aid: obs for aid, obs in '
                     'observations.items()'], calls
    assert 'if lock_open and act_fn is not None' not in body
    assert body.count('sim.update_site_action_locks(tick)') == 1


def _core_sim():
    """CORE (an AMF) -- A == B, a UE on each radio site: A reaches the core
    over Core_0, B only through A (Mesh_AB)."""
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    t = Topology()
    for nid, ntype in (('AMF_1', NodeType.AMF), (A, NodeType.O_DU),
                       (B, NodeType.RELAY), (UE, NodeType.UE),
                       ('UE_43', NodeType.UE)):
        t.add_node(Node(id=nid, node_type=ntype))
    t.add_link(_link('Core_0', 'AMF_1', A, cap=1000.0,
                     lt=LinkType.FIBER))
    t.add_link(_link('Mesh_AB', A, B, cap=800.0))
    t.add_link(_link('Uu_1', UE, A, cap=100.0, lt=LinkType.WIRELESS,
                     iface=InterfaceType.Uu))
    t.add_link(_link('Uu_2', 'UE_43', B, cap=100.0, lt=LinkType.WIRELESS,
                     iface=InterfaceType.Uu))
    scenario = Scenario(name='wp13', duration_ticks=30, events=[],
                        traffic_profiles={})
    sim = Simulator(t, scenario,
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 0
    return sim


def _perturb(ps):
    from sixg_sim.phy_mac_state import RelayMode
    ps.tx_power_dbm = 30.0
    ps.relay_mode = RelayMode.LOCAL_REROUTE
    ps.prb_relay_fraction = 0.4


def _assert_pre_event(ps, where):
    from sixg_sim.phy_mac_state import RelayMode, MACScheduler
    assert ps.tx_power_dbm == TS.PRE_EVENT_TX_POWER_DBM, where
    assert ps.relay_mode == RelayMode.OFF, where
    assert ps.prb_relay_fraction == 0.0, where
    assert ps.mac_scheduler == MACScheduler.ROUND_ROBIN, where
    assert abs(ps.prb_emergency_fraction - 0.3) < 1e-9, where


def test_a_site_whose_own_flag_is_not_set_keeps_the_pre_event_action():
    """D13.2.  The global trigger has fired (stubbed on), B is cut off from
    the core and A is not: from the tick after B's own flag is set, B acts
    and A -- although ANOTHER site is isolated and the global trigger says
    "island" -- keeps the fixed pre-event configuration, takes no action,
    sends no heartbeat and is not in the observations the action step
    receives.  When A's own link to the core goes, A acts from the next
    tick.  With a GLOBAL lock A would act from the first tick."""
    sim = _core_sim()
    assert set(sim.agents) == {A, B}, sim.agents
    sim._detect_island_mode = lambda: True
    sim.random_action_policy = True
    seen = {}

    def act(tick, observations):
        seen[tick] = sorted(observations)
        sim._execute_agent_actions(observations)

    lk = sim.topology.links['Mesh_AB']
    lk.is_up = False
    sim.topology.sync_pair(A, B)
    checked = 0
    for tick in range(0, 8):
        _perturb(sim.phy_mac_states[A])
        res = sim.step_tick(tick, act, forwarding=None, where='t')
        want = [] if tick == 0 else [B]
        assert list(res.acting_sites) == want, (tick, res.acting_sites)
        assert seen.get(tick, []) == want, (tick, seen.get(tick))
        # A is locked: back on the pre-event configuration after the step,
        # never acted, never sent a heartbeat
        _assert_pre_event(sim.phy_mac_states[A], 'A at tick %d' % tick)
        assert sim.agents[A].last_action is None, tick
        assert sim._postcards_by_site.get(A, 0) == 0, tick
        assert sim.site_action_unlocked(A) is False, tick
        checked += 1
    assert checked == 8
    assert sim.site_action_unlocked(B) is True
    assert sim._site_unlock_tick == {B: 1}
    assert sim.agents[B].last_action is not None
    assert sim._postcards_by_site.get(B, 0) >= 1
    # A's own link to the core goes: its flag is set by the tick-8 step, so
    # it acts from tick 9 -- and B keeps acting (a latch, not a re-lock)
    core = sim.topology.links['Core_0']
    core.is_up = False
    sim.topology.sync_pair('AMF_1', A)
    res8 = sim.step_tick(8, act, forwarding=None, where='t')
    assert list(res8.acting_sites) == [B]
    res9 = sim.step_tick(9, act, forwarding=None, where='t')
    assert list(res9.acting_sites) == [A, B] == seen[9]
    assert sim._site_unlock_tick == {B: 1, A: 9}
    assert sim.agents[A].last_action is not None
    rep = sim.postcard_audit.report()
    assert rep['max_per_tick'] == 1


def test_a_locked_sites_postcard_is_refused_by_the_audit():
    """The audit is per site since D13.2: a send by a site whose own action
    step did not run raises, even on a tick on which another site acted."""
    from sixg_sim import postcards as PC
    audit = PC.PostcardSendAudit()
    audit.record(5, True, [B], {B: 0}, {B: 1})          # B acted and sent
    try:
        audit.record(6, True, [B], {A: 0}, {A: 1})      # A sent, locked
    except PC.PostcardInvariantError as e:
        assert 'its own action step did not run' in str(e)
    else:
        raise AssertionError('a locked site sent a postcard unnoticed')
    assert audit.present == {B: 1}


def test_lock_applies_the_fixed_pre_event_configuration():
    """With the lock closed the agents do not act and every site is at the
    operator configuration (23 dBm, relay OFF, RR, 30/70 access split)."""
    from sixg_sim.phy_mac_state import RelayMode, MACScheduler
    sim = _sim()
    for ps in sim.phy_mac_states.values():
        ps.tx_power_dbm = 30.0
        ps.relay_mode = RelayMode.LOCAL_REROUTE
        ps.prb_relay_fraction = 0.4
    sim._detect_island_mode = lambda: False
    acted = []
    sim.step_tick(0, lambda t, o: acted.append(t))
    assert acted == []
    assert sim.phy_mac_states, 'tiny topology has no PHY/MAC sites'
    for ps in sim.phy_mac_states.values():
        assert ps.tx_power_dbm == TS.PRE_EVENT_TX_POWER_DBM
        assert ps.relay_mode == RelayMode.OFF
        assert ps.prb_relay_fraction == 0.0
        assert ps.mac_scheduler == MACScheduler.ROUND_ROBIN
        assert abs(ps.prb_emergency_fraction - 0.3) < 1e-9
    # ... and with the lock open the driver's action step runs
    sim._detect_island_mode = lambda: True
    sim.step_tick(1, lambda t, o: acted.append(t))
    assert acted == [1]


# ----------------------------------------------------------------------
# 3. the shared backhaul-degradation helper
# ----------------------------------------------------------------------

def test_degradation_helper_classes_floor_and_idempotence():
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.scenario import Scenario
    t = Topology()
    for nid, ntype in (('a', NodeType.O_DU), ('b', NodeType.RELAY),
                       ('c', NodeType.EDGEUPF), ('d', NodeType.O_DU),
                       ('e', NodeType.O_DU)):
        t.add_node(Node(id=nid, node_type=ntype))
    t.add_link(_link('mw', 'a', 'b', cap=1000.0, lt=LinkType.MICROWAVE))
    t.add_link(_link('ptp', 'b', 'c', cap=100.0, lt=LinkType.MICROWAVE_PTP))
    t.add_link(_link('mesh', 'c', 'd', cap=2000.0, lt=LinkType.MULTIHAUL_MESH))
    t.add_link(_link('fib', 'd', 'e', cap=5000.0, lt=LinkType.FIBER))
    t.add_link(_link('down', 'a', 'e', cap=900.0, lt=LinkType.MICROWAVE,
                     up=False))
    t.links['TR_a_c_1'] = _link('TR_a_c_1', 'a', 'c', cap=30.0,
                                lt=LinkType.TRANSPORT_RELAY)
    t.sync_pair('a', 'c')
    sim = Simulator(t, Scenario(name='d', duration_ticks=5, events=[],
                                traffic_profiles={}),
                    SimulationConfig(verbose=False, random_seed=0))
    sim.current_tick = 4
    n = TS.apply_backhaul_degradation(sim, log_prefix='[T] ')
    assert n == 3
    assert t.links['mw'].capacity == 300.0
    assert t.links['ptp'].capacity == 50          # floor
    assert t.links['mesh'].capacity == 600.0
    assert t.links['fib'].capacity == 5000.0      # untouched
    assert t.links['down'].capacity == 900.0      # down links are skipped
    assert t.links['TR_a_c_1'].capacity == 30.0   # relay bridges untouched
    assert t.graph['a']['b']['capacity'] == 300.0  # graph edge refreshed
    assert sim._bh_degraded is True
    assert TS.apply_backhaul_degradation(sim) == 0  # once per episode
    assert t.links['mw'].capacity == 300.0
    assert TS.BH_DEGRADE_TYPES == (LinkType.MICROWAVE, LinkType.MICROWAVE_PTP,
                                   LinkType.MULTIHAUL_MESH)


def test_no_driver_keeps_a_private_degradation():
    har = _src('run_timeline_comparison.py')
    trn = _src('train_on_comparison.py')
    for src in (har, trn):
        assert '_BH_DEGRADE_TYPES' not in src
        assert '_apply_backhaul_degradation' not in src
        assert 'link.capacity * 0.30' not in src
    step = _src('sixg_sim', 'tick_step.py')
    assert step.count('apply_backhaul_degradation(sim, log_prefix)') == 1


# ----------------------------------------------------------------------
# 4. the end-of-pass per-link load snapshot -> Block A backhaul fields
# ----------------------------------------------------------------------

def test_link_load_snapshot_feeds_the_block_a_backhaul_fields():
    sim = _sim()
    ps_a = sim.phy_mac_states[A]
    ps_b = sim.phy_mac_states[B]
    # no pass yet: the defaults
    sim._update_phy_mac_observations()
    assert ps_a.backhaul_utilization == 0.0
    assert ps_a.backhaul_capacity_mbps == 1000.0
    # a forwarding pass leaves 400 of 800 on the A-B hop and 150 of 200 on
    # the B-C hop; the snapshot is taken at the end of the pass ...
    sim.topology.links['Backhaul_9'].current_utilization = 400.0
    sim.topology.links['Microwave_1'].current_utilization = 150.0
    sim._snapshot_link_loads()
    assert sim._link_load_snapshot['Backhaul_9'] == (0.5, 400.0)
    assert sim._link_load_snapshot['Microwave_1'] == (0.75, 50.0)
    assert 'Uu_1' in sim._link_load_snapshot
    # ... the live counters are then reset at the top of the next tick,
    # and the observation still reads the pass
    for l in sim.topology.links.values():
        l.reset_utilization()
    sim._update_phy_mac_observations()
    assert ps_a.backhaul_utilization == 0.5
    assert ps_a.backhaul_capacity_mbps == 400.0
    assert ps_b.backhaul_utilization == 0.75          # max over own links
    assert ps_b.backhaul_capacity_mbps == 50.0        # min over own links
    # the Uu access link is in the snapshot but never counts as backhaul:
    # A's fields above come from Backhaul_9 alone (Uu_1 carried nothing
    # and would have read 100.0 available)


def test_unmeasured_links_do_not_enter_the_block_a_fields():
    """A link that was DOWN at the end of the last pass has no entry and
    contributes nothing, even if it is up at observation time; a link that
    is down now is skipped as before."""
    sim = _sim()
    ps_b = sim.phy_mac_states[B]
    sim.topology.links['Microwave_1'].is_up = False
    sim.topology.links['Backhaul_9'].current_utilization = 80.0
    sim._snapshot_link_loads()
    assert 'Microwave_1' not in sim._link_load_snapshot
    sim.topology.links['Microwave_1'].is_up = True        # up now
    sim.topology.links['Microwave_1'].current_utilization = 199.0
    sim._update_phy_mac_observations()
    assert ps_b.backhaul_utilization == 0.1               # 80/800 only
    assert ps_b.backhaul_capacity_mbps == 720.0
    # snapshot present, link down now: skipped
    sim.topology.links['Backhaul_9'].is_up = False
    sim._update_phy_mac_observations()
    assert ps_b.backhaul_utilization == 0.0
    assert ps_b.backhaul_capacity_mbps == 1000.0


def test_forward_traffic_takes_the_snapshot_at_the_end():
    src = _src('sixg_sim', 'simulation.py')
    body = src[src.index('    def _forward_traffic('):
               src.index('    def _collect_metrics(')]
    i = body.index('self._snapshot_relay_link_load()')
    j = body.index('self._snapshot_link_loads()')
    assert i < j
    assert body.count('self._snapshot_link_loads()') == 1
    obs = src[src.index('    def _update_phy_mac_observations('):
              src.index('    # ── MultiHaul beam-steering mechanics')]
    code = '\n'.join(l for l in obs.splitlines()
                     if not l.strip().startswith('#'))
    # the two live-counter reads of the old code are gone ...
    assert 'link.current_utilization /' not in code
    assert 'link.available_capacity()' not in code
    # ... and the fields come from the snapshot
    assert '_link_load_snapshot' in code
    assert '_snap[0]' in code and '_snap[1]' in code


# ----------------------------------------------------------------------
# 5. action signature
# ----------------------------------------------------------------------

def test_action_signature_is_zero_until_an_agent_acts_then_stable():
    from sixg_sim.agent import PHYMACAction
    sim = _sim()
    for ag in sim.agents.values():
        ag.last_action = None
    assert TS.action_signature(sim) == 0
    aid = sorted(sim.agents)[0]
    sim.agents[aid].last_action = PHYMACAction(
        tx_power_step=2, mcs_emergency_idx=2, mcs_general_idx=2,
        prb_emergency_frac=0.3, prb_relay_frac=0.0, prb_general_frac=0.7,
        relay_mode_idx=0)
    s1 = TS.action_signature(sim)
    assert s1 != 0 and s1 == TS.action_signature(sim)
    sim.agents[aid].last_action.relay_mode_idx = 3
    assert TS.action_signature(sim) != s1


# ----------------------------------------------------------------------
# 6. the detector lives in the engine; both drivers step only through
#    step_tick
# ----------------------------------------------------------------------

def test_forwarding_view_lives_in_the_engine():
    """WP2 moved the MARL-path forwarding view into the engine; WP3 replaced
    it with sixg_sim/link_state_service.py.  The harness must still import it
    rather than define one of its own."""
    from sixg_sim import link_state_service as D
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    assert R.LinkStateService is D.LinkStateService
    assert R.MARL_BFD_DETECT_TICKS == D.MARL_BFD_DETECT_TICKS
    assert R.make_link_state_service is D.make_link_state_service
    har = _src('run_timeline_comparison.py')
    assert 'class LinkStateService' not in har
    assert 'LinkStateService(sim)' not in har
    sim = _sim()
    det = D.make_link_state_service(sim)
    if D.MARL_BFD_DETECT_TICKS > 0:
        assert isinstance(det, D.LinkStateService)
        det.step(0)
        assert set(det.visible) == set(sim.topology.links)
        assert sim.link_state_service is det and sim._marl_detect is det
    else:
        assert det is None


def test_the_legacy_worker_loop_is_marked_and_unused():
    """sixg_sim/worker.py still holds a second, now-divergent per-tick loop.
    It is off the evaluated path (its only importer is sixg_sim/main.py),
    but WP2's whole point is that no second tick order can quietly become
    the reference again, so the loop must carry the marker that says so."""
    wrk = _src('sixg_sim', 'worker.py')
    assert 'LEGACY TICK LOOP -- NOT THE EVALUATED PATH (WP2)' in wrk, (
        'the legacy loop in sixg_sim/worker.py lost its marker: either port '
        'it to sim.step_tick or restore the comment (docs/DECISIONS.md D2.11)')
    i = wrk.index('LEGACY TICK LOOP')
    j = wrk.index('for tick in range(ep_scenario.duration_ticks):')
    assert 0 < j - i < 1200, 'the marker is not at the head of the loop'
    # ... and nothing on the evaluated path pulls it in, or still points at
    # it as the implementation to copy (the trainer's docstring used to say
    # it mirrored run_worker_episode "exactly", which WP2 ended)
    for name in ('run_timeline_comparison.py', 'train_on_comparison.py'):
        src = _src(name)
        assert 'sixg_sim.worker import' not in src, name
        assert 'run_worker_episode(' not in src, name
        assert 'Mirrors sixg_sim/worker.py' not in src, name


def test_step_tick_without_a_forwarding_view():
    """The MARL_DETECT_TICKS=0 ablation (the factory -> None) and the
    baseline arms, which gate their own flows in the before_forward hook:
    `forwarding=None` must mean no view step, no flow gate, no believed
    graph and forwarding on the TRUE graph -- in island mode as well.

    Driven directly rather than through the environment variable, so this
    path is covered in every run (RERUN_SPEC WP3 pre-registers an
    'installation off / idealised timing' sensitivity that uses it)."""
    sim = _sim()
    rec = _Recorder(sim, island=True)
    graphs = []
    sim._forward_traffic = lambda traffic: (rec.trace.append('forward'),
                                            graphs.append(sim.topology.graph))
    true_graph = sim.topology.graph
    sim.ue_to_ue_flows = [(UE, 'UE_43')]
    try:
        res = sim.step_tick(4, lambda t, o: rec.trace.append('act'),
                            forwarding=None, degrade_tick=None,
                            hooks=_hooks(rec.trace), where='t')
    finally:
        rec.restore()
    assert rec.trace == [
        'events', 'island', 'set_island', 'h_after_island',
        'auto_mcs', 'h_island_config', 'observe', 'h_after_observe',
        'site_locks', 'act', 'invalidate', 'h_before_forward',
        'invariant', 'generate', 'forward'], rec.trace
    # no gate: the full flow set reached forwarding and nothing was buffered
    assert sim.ue_to_ue_flows == [(UE, 'UE_43')]
    assert not hasattr(sim, '_proto_all_flows')
    # the true graph, and no believed view reported
    assert graphs == [true_graph]
    assert sim.topology.graph is true_graph
    assert res.believed_view is False and res.acted and res.lock_open


def test_the_factory_is_none_under_the_zero_tick_ablation():
    """The factory's None branch, exercised whatever MARL_DETECT_TICKS is set
    to in this process (the constant is read at call time)."""
    from sixg_sim import link_state_service as D
    sim = _sim()
    saved = D.MARL_BFD_DETECT_TICKS
    try:
        D.MARL_BFD_DETECT_TICKS = 0
        assert D.make_link_state_service(sim) is None
        D.MARL_BFD_DETECT_TICKS = 3
        assert isinstance(D.make_link_state_service(sim), D.LinkStateService)
    finally:
        D.MARL_BFD_DETECT_TICKS = saved
    # ... and the drivers must not assume a view exists.  Attaching it is the
    # FACTORY's job now (it sets sim.link_state_service, the name WP6's
    # adapter binds to, and sim._marl_detect, which the harness's
    # latency/outage block reads) and it returns None under the ablation, so
    # a driver that assigned the attribute itself would set it to None.
    har = _src('run_timeline_comparison.py')
    trn = _src('train_on_comparison.py')
    for src in (har, trn):
        assert 'sim._marl_detect = ' not in src, (
            'a driver assigns the forwarding view itself; the factory owns it')
    assert "hasattr(sim, '_marl_detect')" in har


def test_drivers_step_only_through_step_tick():
    har = _src('run_timeline_comparison.py')
    trn = _src('train_on_comparison.py')
    assert har.count('sim.step_tick(') == 1
    assert trn.count('sim.step_tick(') == 2
    for frag in ('sim._process_events(', 'sim._forward_traffic(',
                 'sim._generate_traffic(', 'sim._build_agent_observations(',
                 'sim._detect_island_mode(', '.reset_utilization()',
                 'sim.check_graph_consistency(', 'sim.topology.graph = ',
                 '_auto_link_adaptation(sim)\n', 'sim.island_mode = '):
        assert frag not in har, frag
        assert frag not in trn, frag
    # the shared MARL-path action step is the one both drivers call
    assert har.count('def marl_action_step(') == 1
    assert har.count('marl_action_step(') >= 2
    assert trn.count('marl_action_step(sim, tick, observations') == 2
    assert 'sim._execute_agent_actions(' not in trn
    assert har.count('sim._execute_agent_actions(') == 1   # inside marl_action_step
    # both drivers create the forwarding view the same way
    assert har.count('make_link_state_service(sim)') == 1
    assert trn.count('make_link_state_service(sim)') == 2
    # the step itself holds the physics, in the documented order (sequential
    # search inside the function body: the module docstring names the same
    # calls earlier)
    step = _src('sixg_sim', 'tick_step.py')
    body = step[step.index('def step_tick('):]
    order = ['link.reset_utilization()', 'sim._process_events(tick)',
             'apply_backhaul_degradation(sim, log_prefix)',
             'sim.island_mode = sim._detect_island_mode()',
             "_hook('after_island', tick)", 'auto_link_adaptation(sim)',
             'lock_phy_mac_pre_event(sim)', "_hook('island_config', tick)",
             'observations = sim._build_agent_observations()',
             "_hook('after_observe', tick, observations)",
             'acting = sim.update_site_action_locks(tick)',
             'lock_phy_mac_pre_event(sim, sites=_locked)',
             'act_fn(tick, {aid: obs for aid, obs in observations.items()',
             'sim.topology.invalidate_infrastructure_cache()',
             "_hook('before_forward', tick)", 'forwarding.step(tick)',
             'forwarding.get_routable_flows()',
             'sim.check_graph_consistency(where=where)',
             'traffic = sim._generate_traffic()',
             'forwarding.believed_graph()', 'sim._forward_traffic(traffic)']
    cur = 0
    for s in order:
        i = body.find(s, cur)
        assert i >= 0, 'out of order or missing in step_tick: %r' % s
        cur = i + len(s)
