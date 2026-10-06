# -*- coding: utf-8 -*-
"""WP6 -- the schema against the real environment: Scenario B sets the local
isolation flag from tick 1, the two vectors come out at the fixed widths in
a live pass, and a short end-to-end smoke runs.

Since the trackAgent merge the pass attaches the WP3 link-state service the
way both drivers do (make_link_state_service, stepped by step_tick), and the
local isolation flag every observation carries is checked against the
service's own per-site state from the PREVIOUS step.

Builds the comparison topology, so it is slower than test_wp6_obs_schema and
belongs on <compute-server>:

    env MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile \\
        MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 \\
        MARL_REWARD_PROFILE=mission \\
        MARL_RELAY_ENTROPY_FLOOR=0.01 MARL_STEER_MODEL=actionable \\
        PYTHONHASHSEED=0 MARL_POOL_WORKERS=2 \\
        nice -n 19 python3 tests/run_tests.py wp6_smoke
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_ENV = {
    'MARL_TRANSPORT_POWER': 'planning', 'MARL_RELAY_STATE': 'reconcile',
    'MARL_RELAY_REPOINT': 'fragment_aware', 'MARL_POSTCARD_ALWAYS': '1',
    'MARL_REWARD_PROFILE': 'mission',
    'MARL_RELAY_ENTROPY_FLOOR': '0.01', 'MARL_STEER_MODEL': 'actionable',
}
for _k, _v in EVAL_ENV.items():
    os.environ.setdefault(_k, _v)

SEED = 82
TICKS = 40
# Scenario A needs a longer pass since the trackAgent merge: the flag is the
# link-state service's, and the service withdraws the severed core
# adjacencies only after BFD detection, flooding and SPF.  Until WP13 the
# long part was one Router-LSA per pair transition per originator at
# MIN_LS_INTERVAL, which isolated every site only at about tick 63 on seed 82
# with the severance at tick 10 (docs/DECISIONS.md, D-MRG.13 item 2); since
# D13.1 a router carries all its transitions in one Router-LSA and the flag
# rises much earlier, but the pass keeps its length.  The stand-in this
# replaced followed the global trigger and rose at tick 11.
TICKS_A = 100
TICKS_BY_KIND = {'rescue_ops': TICKS, 'recovery': TICKS_A}

_cache = {}


def _pass(kind, ticks=None):
    """One short environment pass; returns the per-tick observation records."""
    ticks = TICKS_BY_KIND[kind] if ticks is None else ticks
    key = (kind, ticks)
    if key in _cache:
        return _cache[key]
    import torch
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        from run_timeline_comparison import (build_comparison_topology,
                                             build_comparison_scenario)
    finally:
        sys.argv = argv
    from sixg_sim.simulation import SimulationConfig, Simulator

    topology, _info = build_comparison_topology(SEED, log_prefix='[WP6] ')
    scenario = build_comparison_scenario(
        kind, SEED, topology=topology, ticks=ticks,
        severance_tick=(0 if kind == 'rescue_ops' else 10),
        rescue_tick=ticks - 1, scenario_seed=SEED)
    sim = Simulator(topology, scenario,
                    SimulationConfig(enable_island_detection=True,
                                     verbose=False, random_seed=SEED))
    sim._current_scenario_type = getattr(scenario, 'scenario_type', kind)
    sim.random_action_policy = True
    sim.marl_ue_routing_enabled = True
    torch.manual_seed(SEED)
    # The arm's forwarding view, exactly as both drivers build it.
    from sixg_sim.link_state_service import make_link_state_service
    svc = make_link_state_service(sim)
    assert svc is not None, 'MARL_DETECT_TICKS=0 is set: no service to test'

    records = []
    service_state = []          # per tick, AFTER the step: svc's own state

    def _collect(tick, observations):
        records.append({aid: (o.local_isolated, o.ticks_since_local_isolation)
                        for aid, o in observations.items()})
        _cache.setdefault('last_obs', {}).update(observations)

    def _act(tick, observations):
        sim._execute_agent_actions(observations)

    island_ticks = []
    acting = []                 # per tick: the sites whose own lock was open
    for tick in range(ticks):
        res = sim.step_tick(tick, _act, forwarding=svc,
                            hooks={'after_observe': _collect},
                            where='wp6_smoke', log_prefix='[WP6] ')
        acting.append(tuple(res.acting_sites))
        service_state.append({aid: svc.isolation_state(aid)
                              for aid in sim.agents})
        if sim.island_mode:
            island_ticks.append(tick)

    _cache[key] = records
    _cache[('service',) + key] = service_state
    _cache[('island',) + key] = island_ticks
    _cache[('acting',) + key] = acting
    return records


def _assert_observation_is_the_services_previous_step(kind):
    """Every observation's Block D pair at tick t is the service's state
    after the step of tick t-1: flag equal, timer = t - isolated_since."""
    rec = _pass(kind)
    svc_state = _cache[('service', kind, TICKS_BY_KIND[kind])]
    assert len(rec) == len(svc_state) == TICKS_BY_KIND[kind]
    n_checked = 0
    for tick in range(1, len(rec)):
        for aid, (iso, dt) in rec[tick].items():
            s_iso, s_since = svc_state[tick - 1][aid]
            assert iso == s_iso, (kind, tick, aid, iso, s_iso)
            assert dt == ((tick - s_since) if s_iso else 0), (
                kind, tick, aid, dt, s_since)
            n_checked += 1
    assert n_checked > 0, 'no observation was compared with the service'
    return n_checked


def test_scenario_b_sets_the_flag_from_tick_1():
    rec = _pass('rescue_ops')
    assert len(rec) == TICKS

    # tick 0: no previous step, so the flag is down everywhere
    assert not any(iso for iso, _dt in rec[0].values()), \
        'tick 0 has no previous service step; the flag must be 0'

    # tick 1 onwards: every agent site is isolated, timer counting from 0
    for tick in range(1, TICKS):
        flags = {aid: v for aid, v in rec[tick].items()}
        assert all(iso for iso, _dt in flags.values()), (
            'Scenario B: every site must be locally isolated from tick 1; '
            'tick %d has %d sites with the flag down'
            % (tick, sum(1 for iso, _ in flags.values() if not iso)))
        for aid, (iso, dt) in flags.items():
            assert dt == tick, (aid, tick, dt)


def test_scenario_a_flag_is_down_before_the_event():
    rec = _pass('recovery')
    for tick in range(0, 10):
        assert not any(iso for iso, _dt in rec[tick].values()), (
            'Scenario A must not be locally isolated before the severance '
            '(tick %d)' % tick)
    # and it comes up after it (every site: the whole RAN lost the core)
    assert rec[-1] and all(iso for iso, _dt in rec[-1].values()), \
        'Scenario A: not every site became locally isolated after the severance'


def test_the_flag_is_the_link_state_services_previous_step():
    """Seam (a) of the trackAgent merge, on the real environment: in both
    scenarios every observed Block D pair equals the WP3 service's per-site
    state from the step before, and in Scenario A the flag rises only after
    the service has installed the severance -- strictly later than the
    engine's global island trigger, which the deleted stand-in followed."""
    n = (_assert_observation_is_the_services_previous_step('rescue_ops')
         + _assert_observation_is_the_services_previous_step('recovery'))
    assert n > 0
    rec = _pass('recovery')
    first_island = _cache[('island', 'recovery', TICKS_A)][0]
    flagged = [t for t in range(len(rec))
               if any(iso for iso, _ in rec[t].values())]
    assert flagged, 'Scenario A: the flag never rose within %d ticks' % TICKS_A
    first_flag = flagged[0]
    # the stand-in showed the flag at first_island + 1; the service needs
    # BFD detection (3 ticks) plus installation on top of that
    assert first_flag > first_island + 1 + 3, (first_island, first_flag)


def _assert_each_site_acts_from_its_own_flag(kind):
    """D13.2 on the real environment: at every tick a site is among the
    acting sites iff its OWN observed flag has been set at that tick or
    before (the per-site latch), whatever the global trigger says."""
    rec = _pass(kind)
    acting = _cache[('acting', kind, TICKS_BY_KIND[kind])]
    island = set(_cache[('island', kind, TICKS_BY_KIND[kind])])
    assert len(acting) == len(rec)
    seen_flag = set()
    n_pairs = n_locked_in_island = n_acting_outside_island = 0
    for tick in range(len(rec)):
        seen_flag |= {aid for aid, (iso, _dt) in rec[tick].items() if iso}
        for aid in rec[tick]:
            assert (aid in acting[tick]) == (aid in seen_flag), (
                kind, tick, aid, aid in acting[tick], aid in seen_flag)
            n_pairs += 1
            if tick in island and aid not in acting[tick]:
                n_locked_in_island += 1
            if tick not in island and aid in acting[tick]:
                n_acting_outside_island += 1
    assert n_pairs > 0, 'no (tick, site) pair was checked'
    # a site's own flag is never set before the global trigger in these
    # scenarios, so every heartbeat goes over the DCC (control_plane.py)
    assert n_acting_outside_island == 0, (kind, n_acting_outside_island)
    return n_pairs, n_locked_in_island


def test_each_site_acts_from_its_own_flag_not_the_global_trigger():
    """D13.2.  Scenario B: nobody acts at tick 0 (no previous step, no own
    flag), every site from tick 1.  Scenario A: no site acts before its own
    flag -- so the sites stay locked for the ticks between the global
    trigger and the service's installation of their isolation, which the
    global lock would have opened at once."""
    n_b, _ = _assert_each_site_acts_from_its_own_flag('rescue_ops')
    acting_b = _cache[('acting', 'rescue_ops', TICKS)]
    assert acting_b[0] == ()
    assert all(len(a) > 0 for a in acting_b[1:])
    n_a, locked_a = _assert_each_site_acts_from_its_own_flag('recovery')
    assert locked_a > 0, (
        'Scenario A: no site was ever locked after the global trigger; the '
        'per-site lock was not exercised')
    assert n_a + n_b > 0


def test_live_vectors_have_the_fixed_widths():
    from sixg_sim.agent import (ACTOR_OBS_DIM, CRITIC_OBS_DIM, RLAgent,
                                actor_feature_values, critic_feature_values)
    _pass('recovery')
    obs_all = _cache['last_obs']
    assert obs_all, 'no observations were collected'
    n = 0
    for aid, o in list(obs_all.items())[:8]:
        a = actor_feature_values(o)
        c = critic_feature_values(o)
        assert len(a) == ACTOR_OBS_DIM
        assert len(c) == CRITIC_OBS_DIM
        assert c[:ACTOR_OBS_DIM] == a
        assert all(v == v for v in c), 'NaN in the critic vector at %s' % aid
        n += 1
    assert n >= 1


def test_smoke_training_step_runs_under_the_new_schema():
    """Two tiny training episodes: collection, GAE and the PPO update all
    run on the WP6 vectors as WP13 left them (actor 26 in, critic 37 in)."""
    import torch
    import train_on_comparison as T
    from sixg_sim.agent import (CriticNetwork, PolicyNetwork, CRITIC_OBS_DIM,
                                OBS_DIM)
    from sixg_sim.reward_spec import KAPPA as _KAPPA

    actor = PolicyNetwork()
    critic = CriticNetwork()
    assert actor.fc1.in_features == OBS_DIM == 26
    assert critic.fc1.in_features == CRITIC_OBS_DIM == 37

    out = T.run_comparison_worker_episode({
        'episode_idx': 0,
        'topology_seed': SEED,
        'scenario_kind': 'rescue_ops',
        'episode_ticks': TICKS,
        'event_seed': SEED,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        # WP8-FIX: the mixing weight is FROZEN at reward_spec.KAPPA and the
        # worker now REFUSES any other value (require_frozen_mix), because a
        # driver that mixes at one kappa while every artefact of the run
        # records another is the drift this WP exists to close.  This smoke
        # test never cared about the weight; it cares that a pool fills.
        'global_reward_mix': _KAPPA,
        'torch_seed': SEED,
    })
    pool = out['pool_data']
    assert pool['obs'], 'no transitions were collected'
    assert pool['obs'][0].shape == (OBS_DIM,), pool['obs'][0].shape
    assert pool['global_obs'][0].shape == (CRITIC_OBS_DIM,), \
        pool['global_obs'][0].shape
    assert len(pool['advantages']) == len(pool['obs'])
    assert all(r == r for r in pool['returns']), 'NaN return'


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
