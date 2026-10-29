"""One per-tick simulation step shared by the training and evaluation drivers.

RERUN_SPEC WP2.  `step_tick` fixes the order of the per-tick phases for
`train_on_comparison.py` (both loops) and for the harness physics pass
(`run_timeline_comparison.run_physics_pass`), so the two drivers cannot
drift apart again.  `Simulator.step_tick` is a thin method over it.

  1. per-tick resets: link utilisation.  The DCC send counters named by
     RERUN_SPEC WP2.1 reset THEMSELVES on the first send of a new tick
     (WP5: DisasterControlChannel._roll_tick), so no driver can forget them
     again -- forgetting them was the one-postcard-per-episode defect;
  2. scenario events; backhaul degradation through ONE helper
     (`apply_backhaul_degradation`: MICROWAVE, MICROWAVE_PTP and
     MULTIHAUL_MESH, 30 % of capacity, floor 50 Mb/s);
  3. isolation: the engine's global island-mode detector
     (`Simulator._detect_island_mode`), read through
     `Simulator.action_lock_open()` by the arms WITHOUT agents only (the
     pre-event configuration of every radio site until it fires, the
     baseline arms' own `island_config` after).  The per-site local-isolation
     state WP6's Block D observes -- and, since D13.2, each agent site's
     ACTION LOCK -- is updated at phase 6, not here: inside the WP3
     link-state service's step, or, for an arm with no service, by
     `agent.advance_local_isolation` on the true graph;
  4. the shared PHY/MAC baseline (auto CQI->MCS link adaptation, every arm)
     and the observations, built from END-OF-PREVIOUS-PASS snapshots
     (`Simulator._link_load_snapshot` feeds the two Block A backhaul fields);
  5. actions, PER SITE (D13.2, RERUN_SPEC WP2 step 5): an agent site is
     LOCKED to the fixed pre-event radio configuration
     (`lock_phy_mac_non_mcs(sim, PRE_EVENT_TX_POWER_DBM, sites=...)`) until
     ITS OWN local-isolation flag is set -- the flag its own observation
     carries, i.e. the state after the PREVIOUS step
     (`Simulator.update_site_action_locks`, a latch for the episode) -- the
     same rule in training and evaluation; the driver's
     `act_fn(tick, observations)` receives the observations of the UNLOCKED
     sites only, so a locked site takes no action, sends no heartbeat and
     leaves no PPO transition (its configuration is environment dynamics,
     not the policy's choice).  The action step also sends the DCC postcard
     heartbeat (WP5); the tick's sends are then fed to
     `Simulator.postcard_audit` with the sites that acted (at most one per
     site per tick and none by a locked site, raised at once; > 1 per site
     after the event, checked by the driver at the end of the pass with
     `postcard_audit.check()`);
  6. link-state step (`forwarding.step(tick)`, which also updates the
     per-site local isolation) and its flow gate in island mode -- or, with
     no forwarding service, the true-graph local-isolation update
     (`agent.advance_local_isolation`); then the routing-graph invariant
     (WP1) on the TRUE graph;
  7. traffic generation and forwarding on the forwarding view
     (`forwarding.believed_graph()` whenever the INSTALLED view differs from
     truth);
  8. end-of-pass snapshots: relay-link load (make-before-break), per-link
     load (Block A), and the WP4 relay pass counters and executed relay-mode
     flips, all taken at the end of `Simulator._forward_traffic`.

Driver work that is NOT engine physics -- the harness's adapt-freeze
bookkeeping, the baseline arms' fixed PHY/MAC configuration and protocol
controllers, the learning bookkeeping -- runs through named hooks at fixed
points of the step, so its position relative to the physics is the same in
every driver:

  hooks['after_island'](tick)        after isolation, before the PHY/MAC
                                     baseline (freeze / temperature
                                     bookkeeping; touches no engine state)
  hooks['island_config'](tick)       in place of the pre-event lock when the
                                     GLOBAL trigger has fired, BEFORE
                                     observations: the baseline arms'
                                     per-arm configuration and the SDN
                                     controller's relay activation
  hooks['after_observe'](tick, obs)  after observations, before actions:
                                     reward completion, PPO bookkeeping
  hooks['before_forward'](tick)      after actions and the infrastructure
                                     cache invalidation, before the graph
                                     invariant and forwarding: the baseline
                                     controllers' step and flow gates (the
                                     MARL-path detector is stepped by the
                                     step itself at this same point)

`forwarding` is the arm's forwarding view -- a
`sixg_sim.link_state_service.LinkStateService` (WP3) with `step`,
`get_routable_flows`, `believed_graph` and `visible`, plus `tick_messages`,
`installed_since` and `isolation_state` -- or None for the true graph without
a flow gate (the baseline arms, whose controllers gate flows in
`before_forward`; the MARL_DETECT_TICKS=0 ablation).
"""
from collections import namedtuple
import hashlib
import zlib

from .topology import LinkType
from .phy_mac_state import (MCSLevel, RelayMode, MACScheduler,
                            auto_cqi_mcs_idx)

# ── Post-disaster backhaul degradation (one helper for both drivers) ──────
# A natural disaster brings rain fade, debris and antenna misalignment on
# the WIRELESS backhaul.  This DEGRADES capacity (it does not sever),
# creating congestion an arm may try to bypass.  Applied equally to ALL arms
# and to both drivers -- it is a physical event, not a handicap.
#
# WHICH LINK TYPES.  The 18 GHz PtP microwave hops AND the 60 GHz
# MULTIHAUL_MESH hops: ITU-R P.838-3 gives 25 mm/h specific attenuation of
# ~2.3 dB/km at 18 GHz against ~10.0 dB/km at 60 GHz, so the mesh class is
# the MORE rain-sensitive one.  (The training driver used to match the
# string 'Microwave' and so left every mesh hop at full capacity; the
# evaluation harness degraded all three classes.  One helper, one rule.)
#
# RESIDUAL LIMITATION, STATED RATHER THAN HIDDEN.  A single 70 % capacity
# cut is applied to both classes even though 60 GHz should lose more; it is
# the conservative direction for the 60 GHz mesh and keeps the event a single
# documented magnitude.  Dynamically formed TRANSPORT_RELAY bridges are NOT
# degraded: their capacity comes from the live TG budget at formation time
# (can_form_relay_link, rain_mm_h = 0.0), so a rain penalty applied only at
# this one event would be inconsistent with the budget that sized them.
BH_DEGRADE_TYPES = (LinkType.MICROWAVE,
                    LinkType.MICROWAVE_PTP,
                    LinkType.MULTIHAUL_MESH)
BH_DEGRADE_KEEP_FRACTION = 0.30     # capacity kept after the event
BH_DEGRADE_FLOOR = 50               # capacity floor (Mb/s units of Link.capacity)

# The fixed pre-event radio configuration every arm runs until the action
# lock opens (run_physics_pass: "Pre-disaster: identical radio baseline for
# ALL approaches").
PRE_EVENT_TX_POWER_DBM = 23.0


def apply_backhaul_degradation(sim, log_prefix=''):
    """Degrade every UP wireless backhaul link of the three static classes
    to max(BH_DEGRADE_FLOOR, BH_DEGRADE_KEEP_FRACTION x capacity), once per
    episode (`sim._bh_degraded`).  Returns the number of links degraded.

    Byte-for-byte the evaluation harness's former inline block: same link
    walk (topology.links insertion order), same expression, same graph-edge
    capacity refresh and cache invalidation.
    """
    if getattr(sim, '_bh_degraded', False):
        return 0
    sim._bh_degraded = True
    degraded_count = 0
    degraded_by_type = {}
    for lid, link in sim.topology.links.items():
        if not link.is_up:
            continue
        lt = getattr(link, 'link_type', None)
        if lt in BH_DEGRADE_TYPES:
            link.capacity = max(BH_DEGRADE_FLOOR,
                                link.capacity * BH_DEGRADE_KEEP_FRACTION)
            ep0, ep1 = link.endpoints
            if sim.topology.graph.has_edge(ep0, ep1):
                sim.topology.graph[ep0][ep1]['capacity'] = link.capacity
            degraded_count += 1
            degraded_by_type[lt.value] = degraded_by_type.get(lt.value, 0) + 1
    if degraded_count > 0:
        sim.topology.invalidate_infrastructure_cache()
        print(f"{log_prefix}[BH-DEGRADE] t={sim.current_tick}: "
              f"{degraded_count} wireless backhaul links degraded to "
              f"{BH_DEGRADE_KEEP_FRACTION:.0%} capacity (rain fade/debris) "
              f"{degraded_by_type}")
    return degraded_count


def auto_link_adaptation(sim):
    """3GPP TS 38.214 -- automatic CQI-to-MCS link adaptation.

    A NATIVE gNB/DU function at L1/L2, independent of any RIC, SDN
    controller or routing protocol: every arm gets it, every tick.
    sinr_average stands in for the reported CQI.  Uses the SAME CQI->MCS
    mapping (phy_mac_state.auto_cqi_mcs_idx) that Simulator._step_phy_mac
    takes as the reference point for the agents' bounded MCS-offset actions.
    (Moved from run_timeline_comparison._auto_link_adaptation, WP2.)
    """
    for nid, ps in sim.phy_mac_states.items():
        best_mcs = list(MCSLevel)[auto_cqi_mcs_idx(ps.sinr_average)]
        ps.mcs_general = best_mcs
        ps.mcs_emergency = best_mcs


def lock_phy_mac_non_mcs(sim, tx_power_dbm, allow_local_reroute=False,
                         sites=None):
    """Lock TX power, PRB split and scheduler -- but NOT MCS, which
    auto_link_adaptation() sets.  With allow_local_reroute the OSPF/OLSR
    arms get LOCAL_REROUTE on their existing mesh links once converged.
    `sites` restricts the lock to those radio sites (D13.2: the per-site
    action lock); None, the default, locks every site.
    (Moved from run_timeline_comparison._lock_phy_mac_non_mcs, WP2.)
    """
    if sites is None:
        states = sim.phy_mac_states.values()
    else:
        states = [sim.phy_mac_states[s] for s in sites
                  if s in sim.phy_mac_states]
    for ps in states:
        ps.tx_power_dbm = tx_power_dbm
        if allow_local_reroute and sim.island_mode:
            if ps.relay_mode == RelayMode.OFF:
                ps.relay_mode = RelayMode.LOCAL_REROUTE
            ps.prb_relay_fraction = max(ps.prb_relay_fraction, 0.2)
        elif ps.relay_mode not in (RelayMode.CAPACITY_BOOST,):
            # Don't reset relay mode if SDN already activated CAPACITY_BOOST
            ps.relay_mode = RelayMode.OFF
            ps.prb_relay_fraction = 0.0
        ps.mac_scheduler = MACScheduler.ROUND_ROBIN
        # ACCESS pool only: prb_relay_fraction is a share of the DEDICATED
        # transport radio and does not deduct from the Uu access carrier
        # (PHYMACState.normalise_prb), so the operator-configured 30/70
        # emergency/general access split is independent of relay mode.
        ps.prb_emergency_fraction = 0.3
        ps.prb_general_fraction = 0.7
        ps.normalise_prb()


def lock_phy_mac_pre_event(sim, sites=None):
    """The fixed pre-event configuration (RERUN_SPEC WP2.5): 23 dBm, relay
    OFF, round-robin, 30/70 access split; MCS from auto link adaptation.
    Every radio site, or only `sites` (the locked agent sites, D13.2)."""
    lock_phy_mac_non_mcs(sim, PRE_EVENT_TX_POWER_DBM, sites=sites)


def action_signature(sim):
    """A per-tick signature of every agent's last executed action (crc32 of
    the sorted (agent, heads, PRB split) rows), 0 while no agent has acted.
    Recorded by both drivers; the WP2 parity test and the WP13
    hash-invariance smoke compare it tick by tick."""
    rows = []
    for aid in sorted(sim.agents):
        act = getattr(sim.agents[aid], 'last_action', None)
        if act is None:
            continue
        rows.append((aid, int(act.tx_power_step), int(act.mcs_emergency_idx),
                     int(act.mcs_general_idx), int(act.relay_mode_idx),
                     float(act.prb_emergency_frac), float(act.prb_relay_frac),
                     float(act.prb_general_frac)))
    if not rows:
        return 0
    return zlib.crc32(repr(rows).encode('utf-8'))


def decision_digest(sim):
    """The sha256 hex digest of this tick's EXECUTED joint decision
    (`decision_digest`, docs/TRACKB_HANDOFF.md section 1, S8).

    Canonical form: one row per radio site, sorted by site id, joined with
    '\\n'.  Each row carries, in this order,

        site id
      | the site's sampled head indices -- tx power, MCS offset emergency,
        MCS offset general -- or -1 for each when no action was sampled
      | the EXECUTED relay mode
      | the EXECUTED tx power in dBm, the EXECUTED emergency and general MCS
        levels, and the EXECUTED PRB split (emergency, relay, general), all
        rounded to 6 decimals.

    WHY THE EXECUTED STATE AND NOT THE HEAD INDICES ALONE (a deviation from
    the handoff's wording, recorded in DECISIONS D4.6).  The handoff asks for
    `(site_id, tx_power_index, mcs_offset_emergency_index,
    mcs_offset_general_index, executed_relay_mode, prb_split)`, and says the
    PRE-EVENT LOCKED CONFIGURATION must digest too.  Those two requirements
    cannot both be met by head indices: before the action lock opens no head
    is sampled at all, so a head-index-only digest would be constant over the
    whole pre-event window and blind to a locked configuration that differed
    between the two runs.  The executed radio state is defined on every tick
    of every arm, it is what the site actually did, and it is read AFTER the
    engine's teardown holds have reverted a relay mode -- which is the point
    of "executed".  The head indices are kept beside it, so the digest is
    strictly stronger than either alone: two runs agreeing on it agree on
    every sampled head AND on every executed radio setting.

    S8 compares the digest series of two runs under PYTHONHASHSEED 0 and 1;
    nothing parses the canonical form, so its exact shape is engine-side, but
    it may not change silently -- `tests/test_wp4_bridge_stability.py` pins
    it.  Cost: one small hash per tick.
    """
    rows = []
    for site in sorted(sim.phy_mac_states):
        ps = sim.phy_mac_states[site]
        act = getattr(getattr(sim, 'agents', {}).get(site, None),
                      'last_action', None)
        if act is None:
            tx_i, mcs_e_i, mcs_g_i = -1, -1, -1
        else:
            tx_i = int(act.tx_power_step)
            mcs_e_i = int(act.mcs_emergency_idx)
            mcs_g_i = int(act.mcs_general_idx)
        mode = getattr(ps, 'relay_mode', None)
        rows.append('%s|%d|%d|%d|%s|%.6f|%s|%s|%.6f|%.6f|%.6f' % (
            site, tx_i, mcs_e_i, mcs_g_i,
            getattr(mode, 'value', mode),
            float(getattr(ps, 'tx_power_dbm', 0.0)),
            getattr(getattr(ps, 'mcs_emergency', None), 'name', 'NONE'),
            getattr(getattr(ps, 'mcs_general', None), 'name', 'NONE'),
            float(getattr(ps, 'prb_emergency_fraction', 0.0)),
            float(getattr(ps, 'prb_relay_fraction', 0.0)),
            float(getattr(ps, 'prb_general_fraction', 0.0))))
    return hashlib.sha256('\n'.join(rows).encode('utf-8')).hexdigest()


class PostEventPostcardLedger:
    """The per-site post-event postcard record, `postcards_by_site` and its
    completeness denominator `n_agent_sites` (docs/TRACKB_HANDOFF.md
    section 1; WP4-VERIFY D4.13).

    The driver calls `update` once per tick, after the tick's step, with the
    set of sites hosting an agent at that moment and the engine's CUMULATIVE
    per-site send counter (`Simulator._postcards_by_site`, never reset).

    Two things are kept apart ON PURPOSE, because the WP9 per-site assert is
    only a check if they are:

      * the ROSTER -- every site that hosted an agent at ANY tick
        t >= t_sev.  It is read from the agent table, never from the send
        ledger, so a site that never sent is on it, and a site whose agent
        left mid-window (a node taken down) stays on it.  `n_agent_sites` is
        its size and nothing else.
      * the SENDS -- per-tick differences of the cumulative counter, summed
        over t >= t_sev.

    `postcards_by_site` holds one entry for every roster site (0 when it sent
    nothing after the event) PLUS any site that sent after the event without
    being on the roster.  The second group should be empty; when it is not,
    `len(postcards_by_site) != n_agent_sites` and the analysis's coverage
    check fails on it, instead of the two numbers agreeing by construction
    (which is what deriving both from one set did: D4.13).

    A cumulative counter that DECREASES raises: a reset would silently
    drop sends from the record."""

    def __init__(self, t_sev):
        self.t_sev = int(t_sev)
        self._prev = {}
        self._post = {}
        self._roster = set()

    def update(self, tick, agent_sites, cumulative_by_site):
        now = {s: int(n) for s, n in (cumulative_by_site or {}).items()}
        for site, n in now.items():
            if n < self._prev.get(site, 0):
                raise ValueError(
                    'postcard ledger: the cumulative send counter of %r fell '
                    'from %d to %d at tick %d; it must never be reset'
                    % (site, self._prev[site], n, tick))
        if tick >= self.t_sev:
            self._roster.update(agent_sites)
            for site, n in now.items():
                d = n - self._prev.get(site, 0)
                if d:
                    self._post[site] = self._post.get(site, 0) + d
        self._prev = now

    @property
    def roster(self):
        return frozenset(self._roster)

    def record(self):
        """(postcards_by_site, n_agent_sites): the dict in sorted site order
        (deterministic, so two pickles compare key for key) and the roster
        size."""
        sites = sorted(self._roster | set(self._post))
        return ({s: int(self._post.get(s, 0)) for s in sites},
                len(self._roster))


TickResult = namedtuple('TickResult',
                        'observations acted believed_view lock_open '
                        'acting_sites')


def step_tick(sim, tick, act_fn, *, forwarding=None, degrade_tick=None,
              hooks=None, where='', log_prefix=''):
    """Run one tick of `sim` in the fixed WP2 order (module docstring).

    act_fn(tick, observations)  the arm's agent action step (None for arms
                                without agents); called only when some
                                site's own action lock is open, with the
                                observations of the UNLOCKED sites only
                                (D13.2).
    forwarding                  the arm's forwarding view (detector /
                                service) or None.
    degrade_tick                tick of the backhaul degradation event
                                (None: never).
    hooks                       {'after_island', 'island_config',
                                 'after_observe', 'before_forward'} -> callable.
    where                       label for the graph-invariant message.
    log_prefix                  prefix for engine log lines.
    Returns TickResult(observations, acted, believed_view, lock_open,
    acting_sites) -- `lock_open` is the GLOBAL trigger, `acting_sites` the
    sorted agent sites whose own lock was open (D13.2).
    """
    hooks = hooks or {}

    def _hook(name, *args):
        fn = hooks.get(name)
        if fn is not None:
            fn(*args)

    sim.current_tick = tick

    # 1. per-tick resets.  Link utilisation is cleared here (and again by
    #    _generate_traffic right before forwarding); the observation builder
    #    reads the end-of-pass snapshot, never the live counter.
    #
    #    The DCC send counters (RERUN_SPEC WP2.1) are NOT reset here: since
    #    WP5 they are keyed on the current tick and reset themselves on the
    #    first send of a new tick (DisasterControlChannel._roll_tick), which
    #    covers every driver, including any that never calls this step.
    for link in sim.topology.links.values():
        link.reset_utilization()

    # 2. scenario events, then the backhaul degradation event
    sim._process_events(tick)
    if degrade_tick is not None and tick == degrade_tick:
        apply_backhaul_degradation(sim, log_prefix)

    # 3. isolation: the engine's GLOBAL island detector, which only the arms
    #    WITHOUT agents read (phase 4).  The per-site local isolation WP6
    #    observes, and each agent site's action lock reads, is updated at
    #    phase 6 (module docstring).
    sim.island_mode = sim._detect_island_mode()
    sim.control_plane.set_island_mode(sim.island_mode)
    _hook('after_island', tick)

    # 4. shared PHY/MAC baseline; the GLOBAL pre-event configuration of the
    #    arms without agents; observations
    auto_link_adaptation(sim)
    lock_open = sim.action_lock_open()
    if not lock_open:
        # Pre-event: the identical fixed radio configuration for every arm,
        # in training and evaluation alike (RERUN_SPEC WP2.5).
        lock_phy_mac_pre_event(sim)
    else:
        _hook('island_config', tick)
    observations = sim._build_agent_observations()
    _hook('after_observe', tick, observations)

    # 5. actions, PER SITE (D13.2).  An agent site acts once ITS OWN local
    #    isolation flag -- the one its observation above carries -- has been
    #    set; until then it stays on the fixed pre-event configuration and
    #    the driver's act_fn never sees it, so it takes no action, sends no
    #    heartbeat and leaves no PPO transition.  The action step also sends
    #    the DCC postcard heartbeat (WP5), so the per-site send counter is
    #    read around it and the tick's sends go to the postcard audit with
    #    the sites that acted; it raises at once on a second send by one
    #    site in one tick, or on a send by a locked site.
    acted = False
    acting = []
    _pc_before = dict(getattr(sim, '_postcards_by_site', None) or {})
    if act_fn is not None:
        acting = sim.update_site_action_locks(tick)
        _acting = set(acting)
        if lock_open:
            # after the global trigger no global lock is applied, so the
            # sites whose own lock is still closed get it here
            _locked = [s for s in sorted(getattr(sim, 'agents', None) or ())
                       if s not in _acting]
            if _locked:
                lock_phy_mac_pre_event(sim, sites=_locked)
        if acting:
            act_fn(tick, {aid: obs for aid, obs in observations.items()
                          if aid in _acting})
            acted = True
    _audit = getattr(sim, 'postcard_audit', None)
    if _audit is not None:
        _audit.record(tick, acted, list(acting),
                      _pc_before, getattr(sim, '_postcards_by_site', None))
    # topology may have changed (events, relay lifecycle, IAB attach)
    sim.topology.invalidate_infrastructure_cache()

    # 6. link-state step and flow gate; the baseline controllers run in the
    #    hook at the same point.  Stepped EVERY tick, pre-event included, so
    #    the severance itself is a transition the detector has to catch.
    _hook('before_forward', tick)
    if forwarding is not None:
        forwarding.step(tick)
        if sim.island_mode:
            sim._proto_all_flows = list(sim.ue_to_ue_flows)
            sim.ue_to_ue_flows = forwarding.get_routable_flows()
    else:
        # WP6 Block D for an arm with NO link-state service (the baselines,
        # and the MARL_DETECT_TICKS=0 sensitivity): the same per-site rule on
        # the true graph, at the same phase, so observations read the
        # previous step either way.  Read-only on the topology.
        from .agent import advance_local_isolation
        advance_local_isolation(sim, tick)
    # END-OF-TICK ROUTING-GRAPH INVARIANT (WP1, strict), on the TRUE graph:
    # nothing below this line touches it.
    sim.check_graph_consistency(where=where)

    # 7. traffic generation and forwarding on the forwarding view.  The
    #    believed graph is installed for exactly this one call (try/finally:
    #    an engine exception must not leave it in place), and the link ids
    #    the view holds stale let consume_path tell a blackhole from
    #    congestion.
    traffic = sim._generate_traffic()
    believed = None
    if forwarding is not None and sim.island_mode:
        believed = forwarding.believed_graph()
    if believed is None:
        sim._forward_traffic(traffic)
    else:
        true_graph = sim.topology.graph
        # The two id sets consume_path needs to tell an EXPECTED dead hop from
        # a lifecycle defect, both owned by the service (WP3):
        #   stale      -- the Link object exists and is physically down
        #   tombstoned -- the Link object has already been popped and its
        #                 removal is not installed yet
        # Both deliver exactly 0.0 and both mark the path blackholed; only an
        # id in NEITHER set is an orphan hop, which the WP1 invariant forbids.
        sim._marl_stale_link_ids = forwarding.stale_link_ids()
        sim._marl_tombstone_link_ids = forwarding.tombstoned_link_ids()
        try:
            sim.topology.graph = believed
            sim._forward_traffic(traffic)
        finally:
            sim.topology.graph = true_graph
            sim._marl_stale_link_ids = frozenset()
            sim._marl_tombstone_link_ids = frozenset()
    # restore the full flow list after the flow gate
    if hasattr(sim, '_proto_all_flows'):
        sim.ue_to_ue_flows = sim._proto_all_flows
        del sim._proto_all_flows

    # 8. end-of-pass snapshots are taken inside _forward_traffic
    #    (_snapshot_relay_link_load, _snapshot_link_loads, and WP4's
    #    _snapshot_relay_link_passes and _snapshot_relay_mode_flips).
    return TickResult(observations, acted, believed is not None, lock_open,
                      tuple(acting))
