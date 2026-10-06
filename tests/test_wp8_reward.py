# -*- coding: utf-8 -*-
"""WP8 -- the reward, as a contract.

RERUN_SPEC WP8.  The team objective (eq:reward) and the agent-local reward
(eq:local) are ONE explicit object, `sixg_sim/reward_spec.py`, with named
coefficients and no environment reads; the churn term is scored on the
EXECUTED relay mode so WP4 engine holds are environment dynamics; PPO scores
the SAMPLED action; terms shown constant are DELETED, not zero-rated; and the
frozen coefficient set is dumped for the pre-registration.

Most of this file is FAST (no simulator).  The two checks that need the real
environment -- the engine-hold churn contract and the training smoke -- build
one comparison simulator each and are bounded to a few hundred ticks.

NOTHING IN THIS FILE SKIPS.  Every check either runs or fails.

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp8
"""
import ast
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEED = 82
TICKS = 120


def _py_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in ('__pycache__', '.git', 'output', 'logs')]
        for fn in filenames:
            if fn.endswith('.py'):
                out.append(os.path.join(dirpath, fn))
    assert len(out) > 40, 'the tree walk found only %d files' % len(out)
    return sorted(out)


# ── 1. the object, on hand-constructed states, against hand-computed values ─
def test_team_objective_on_a_hand_constructed_state():
    """Every term of eq:reward, computed by hand from the frozen rates."""
    from sixg_sim.reward_spec import (FROZEN, GlobalRewardInputs,
                                      global_reward)

    t = global_reward(GlobalRewardInputs(
        island_mode=True,
        delivered_fraction=0.42,
        general_delivered_fraction=0.31,
        churn_count=3,
        relay_throughput_mbps=180.0,
        reunify_held=7.5,
        bridges_lost=2,
        restored_ue_paid=17,
    ))

    # Hand-computed, independently of the implementation:
    #   delivery          100.0 x 0.42                 = 42.00
    #   full_delivery     0.42 < 0.95                  =  0.00
    #   general            10.0 x 0.31                 =  3.10
    #   relay_churn       -1.0 x 3                     = -3.00
    #   relay_throughput  min(5.0, 180 / 50)           =  3.60
    #   reunify           held                         =  7.50
    #   bridge_loss       -15.0 x 2                    = -30.00
    #   reach             min(5.0, 0.10 x 17)          =  1.70
    expect = {
        'delivery': 42.00, 'full_delivery': 0.0, 'general': 3.10,
        'relay_churn': -3.00, 'relay_throughput': 3.60, 'reunify': 7.50,
        'bridge_loss': -30.00, 'reach': 1.70,
        'normal_delivery': 0.0, 'normal_spare_capacity': 0.0,
    }
    got = t.as_dict()
    for k, v in expect.items():
        assert abs(got[k] - v) < 1e-9, (k, got[k], v)
    assert abs(t.total - 24.90) < 1e-9, t.total
    assert abs(got['total'] - 24.90) < 1e-9, got['total']

    # the two caps BIND, and they bind at the frozen values
    capped = global_reward(GlobalRewardInputs(
        island_mode=True, relay_throughput_mbps=500.0, restored_ue_paid=100))
    assert abs(capped.relay_throughput - FROZEN.relay_throughput_cap) < 1e-9
    assert abs(capped.reach - FROZEN.reach_cap) < 1e-9

    # the full-delivery bonus fires exactly AT the threshold, not above it
    at = global_reward(GlobalRewardInputs(island_mode=True,
                                          delivered_fraction=0.95))
    below = global_reward(GlobalRewardInputs(island_mode=True,
                                             delivered_fraction=0.9499))
    assert at.full_delivery == FROZEN.full_delivery_bonus
    assert below.full_delivery == 0.0

    # pre-event mode is a DIFFERENT, much smaller objective
    pre = global_reward(GlobalRewardInputs(
        island_mode=False, delivered_fraction=0.5, spare_capacity_sites=40,
        churn_count=99, bridges_lost=99))
    assert abs(pre.normal_delivery - 1.5) < 1e-9
    assert abs(pre.normal_spare_capacity - 0.4) < 1e-9
    assert abs(pre.total - 1.9) < 1e-9, pre.total


def test_the_dead_recovery_ramp_is_gone_and_cannot_come_back():
    """The recovery ramp and its completion bonus were CONSTANT 0.0 in both
    passes of tools/reward_term_dump.py, and they are structurally dead, not
    merely unreached: both are gated on `Simulator._recovery_started_tick`,
    which only a `restore_core` event sets, and `build_comparison_scenario`
    REPLACES the event list with one that carries no such event in either
    scenario.  Both drivers use that builder, so they are identically zero in
    training and in evaluation."""
    import inspect
    from dataclasses import fields as dc_fields
    from sixg_sim.reward_spec import (FROZEN, GLOBAL_TERM_NAMES,
                                      GlobalRewardInputs, GlobalRewardTerms)
    from sixg_sim.simulation import Simulator

    names = ({f.name for f in dc_fields(FROZEN)}
             | {f.name for f in dc_fields(GlobalRewardInputs)}
             | {f.name for f in dc_fields(GlobalRewardTerms)}
             | set(GLOBAL_TERM_NAMES))
    for dead in ('recovery_ramp', 'recovery_ramp_ticks',
                 'recovery_complete', 'recovery_complete_bonus',
                 'recovery_complete_delivered', 'recovery_complete_ramp',
                 'recovery_completed_now', 'recovery_started',
                 'ticks_since_recovery'):
        assert dead not in names, (
            '%r is back in the reward: it is gated on a restore_core event '
            'that the comparison scenarios do not contain' % dead)

    # and the reward no longer reads the simulator's recovery latch at all
    src = inspect.getsource(Simulator.compute_global_connectivity_reward)
    for attr in ('_recovery_started_tick', '_recovery_complete',
                 '_ticks_since_recovery'):
        assert attr not in src, (
            'the reward reads %r again; the latch is engine bookkeeping for '
            'scenarios the evaluated system never runs' % attr)


def test_the_full_delivery_bonus_is_kept_and_the_reason_is_recorded():
    """ACCEPTED LIMITATION, named in the WP report and in docs/DECISIONS.md
    D8.12.  `tools/reward_term_dump.py` also reports `full_delivery` as
    CONSTANT 0.0 in both passes -- but unlike the recovery ramp it is not
    structurally dead: its input is the live, varying delivered fraction
    (max 0.4714 in Scenario A and 0.5279 in Scenario B under the uniform
    reference policy over 600 ticks) and only the 0.95 THRESHOLD is not
    crossed.  Removing a bonus a better policy could earn would change the
    objective rather than clean it, so it stays, and the fact that the
    reference policy never earns it is disclosed."""
    from sixg_sim.reward_spec import FROZEN, GLOBAL_TERM_NAMES

    assert 'full_delivery' in GLOBAL_TERM_NAMES
    assert FROZEN.full_delivery_bonus == 25.0
    assert FROZEN.full_delivery_threshold == 0.95
    doc = os.path.join(ROOT, 'docs', 'DECISIONS.md')
    with open(doc, encoding='utf-8') as f:
        text = f.read()
    assert 'D8.12' in text, (
        'the accepted limitation is not recorded in docs/DECISIONS.md')
    assert 'full_delivery' in text


def test_the_reunification_term_on_hand_computed_partitions():
    from sixg_sim.reward_spec import reunify_points

    # 4 raw fragments down to 1: 3 eliminated x 15, no residual, and the full
    # bonus at its design scale (3 / (4-1) = 1.0) x 60.
    assert abs(reunify_points(1, 4) - 105.0) < 1e-9
    # 4 down to 2: 2 eliminated x 15, one residual component x -2, no bonus.
    assert abs(reunify_points(2, 4) - 28.0) < 1e-9
    # 2 down to 1: 1 eliminated x 15, plus 60 x (1 / 3).
    assert abs(reunify_points(1, 2) - 35.0) < 1e-9
    # nothing merged, and a partition the baseline never had: residual only.
    assert abs(reunify_points(3, 3) + 4.0) < 1e-9
    # an UNPARTITIONED network pays nothing at all
    assert abs(reunify_points(1, 1)) < 1e-9


def test_agent_local_reward_on_a_hand_constructed_state():
    from sixg_sim.reward_spec import FROZEN, LocalRewardInputs, local_reward

    t = local_reward(LocalRewardInputs(
        delivered_share_ue_to_ue=0.05,
        delivered_share_general=0.02,
        in_relay_mode=True,
        relay_carried_mbps=80.0,
        bridge_counterfactual_fragments=2,
        relay_mode_changed=True,
        mcs_stepped_down=True,
        tx_power_step_db=3.0,
    ))
    #   delivery               100.0 x 0.05          =  5.00
    #   general_delivery        10.0 x 0.02          =  0.20
    #   relay_throughput       min(3.0, 80/50)       =  1.60
    #   bridge_counterfactual   25.0 x 2             = 50.00
    #   relay_churn                                  = -1.00
    #   mcs_overreach                                = -1.00
    #   energy                 -max(0, 3/6) x 1.0    = -0.50
    expect = {'delivery': 5.0, 'general_delivery': 0.2,
              'relay_throughput': 1.6, 'bridge_counterfactual': 50.0,
              'relay_churn': -1.0, 'mcs_overreach': -1.0, 'energy': -0.5}
    got = t.as_dict()
    for k, v in expect.items():
        assert abs(got[k] - v) < 1e-9, (k, got[k], v)
    assert abs(t.total - 54.3) < 1e-9, t.total

    # relay throughput pays ONLY when this node is itself relaying, and only
    # for traffic actually carried -- it cannot be farmed by idling in a
    # relay mode, nor earned by a node that carries traffic without relaying.
    assert local_reward(LocalRewardInputs(
        in_relay_mode=True, relay_carried_mbps=0.0)).relay_throughput == 0.0
    assert local_reward(LocalRewardInputs(
        in_relay_mode=False, relay_carried_mbps=900.0)).relay_throughput == 0.0
    assert abs(local_reward(LocalRewardInputs(
        in_relay_mode=True, relay_carried_mbps=900.0)).relay_throughput
        - FROZEN.local_relay_throughput_cap) < 1e-9

    # turning the transmitter DOWN is never a bonus: the cost is
    # consumption-proportional and floored at zero.
    assert local_reward(LocalRewardInputs(tx_power_step_db=-6.0)).energy == 0.0
    assert abs(local_reward(LocalRewardInputs(tx_power_step_db=6.0)).energy
               + FROZEN.local_energy) < 1e-9


def test_the_mixing_function_is_kappa_zero_point_six():
    """The ARITHMETIC of the mixing.  `reward_spec.mix` is the function the
    live drivers call -- `train_on_comparison.run_comparison_worker_episode`
    and `sixg_sim.worker.run_worker_episode` both mix through it -- and
    test_the_live_driver_mixes_at_the_frozen_kappa below is the check that
    they do, on a real episode.  Before WP8-FIX this function had ZERO
    callers in the tree while both drivers mixed with `args.global_reward_mix`
    (its own default literal), so this test pinned a contract the production
    path did not use."""
    from sixg_sim.reward_spec import FROZEN, KAPPA, mix

    assert FROZEN.kappa == 0.6, FROZEN.kappa
    assert KAPPA == 0.6
    # r_i = (1 - kappa) * local + kappa * team, hand-computed on the two
    # states above: 0.4 x 54.3 + 0.6 x 24.90
    assert abs(mix(54.3, 24.90) - (0.4 * 54.3 + 0.6 * 24.90)) < 1e-9
    assert abs(mix(54.3, 24.90) - 36.66) < 1e-9
    assert mix(1.0, 0.0) == 0.4 and mix(0.0, 1.0) == 0.6


def test_the_frozen_kappa_cannot_be_moved_by_a_flag():
    """The tripwire, in the shape the drivers use it.

    `--global-reward-mix` was an INDEPENDENT literal 0.6 in
    `train_on_comparison.py` and in `sixg_sim/main.py`, and it was the weight
    the workers actually mixed with, while the [REWARD] banner, the JSON
    summary and `reward_id()` (which hashes kappa) all reported
    `FROZEN.kappa`.  `--global-reward-mix 0.3` therefore trained at 0.3 and
    recorded 0.6 in every artefact of the run."""
    import argparse
    import inspect
    import train_on_comparison as T
    from sixg_sim.reward_spec import EPISODE_TICKS, KAPPA, require_frozen_mix

    assert require_frozen_mix(None) == KAPPA        # unset -> the frozen one
    assert require_frozen_mix(KAPPA) == KAPPA
    for bad in (0.3, 0.0, 1.0, 0.61):
        raised = False
        try:
            require_frozen_mix(bad)
        except ValueError:
            raised = True
        assert raised, ('require_frozen_mix(%r) did not raise: the flag can '
                        'still move a published number' % bad)

    # the driver's own resolution, on the namespace argparse produces
    ns = T.resolve_frozen_run_shape(
        argparse.Namespace(episode_ticks=None, global_reward_mix=None))
    assert ns.global_reward_mix == KAPPA, ns.global_reward_mix
    assert ns.episode_ticks == EPISODE_TICKS, ns.episode_ticks

    # and the PRODUCTION path calls it: a resolution nothing invokes is the
    # failure this test exists to prevent
    src = inspect.getsource(T.main)
    assert 'resolve_frozen_run_shape(' in src, (
        'train_on_comparison.main() does not call resolve_frozen_run_shape: '
        'the flag defaults are unbound again')

    # no independent literal may come back as either flag's default
    tree = ast.parse(open(os.path.join(ROOT, 'train_on_comparison.py'),
                          encoding='utf-8').read())
    seen = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, 'attr', '') == 'add_argument'
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        flag = node.args[0].value
        if flag not in ('--global-reward-mix', '--episode-ticks'):
            continue
        for kw in node.keywords:
            if kw.arg == 'default':
                seen[flag] = kw.value
    for flag in ('--global-reward-mix', '--episode-ticks'):
        assert flag in seen, 'train_on_comparison.py no longer defines %s' % flag
        d = seen[flag]
        assert isinstance(d, ast.Constant) and d.value is None, (
            '%s carries its own default %r again; it must be None and be '
            'filled from sixg_sim.reward_spec by resolve_frozen_run_shape()'
            % (flag, getattr(d, 'value', ast.dump(d))))


def test_the_team_relay_throughput_term_keeps_the_pre_wp8_division():
    """The team relay-throughput term DIVIDES by 50, because the code WP8
    replaced divided by 50.

    `x / 50.0` and `x * (1.0 / 50.0)` are not the same double: they differ in
    the last ulp for about 14% of inputs.  Spelling the term as a
    multiplication by a rounded reciprocal would have changed the training
    signal in a second place, which would falsify the round's central
    verification claim ("the training path changes in exactly ONE numerical
    term, the agent-local churn cost").  The agent-local counterpart keeps
    its MULTIPLICATION for the same reason: that is what ITS pre-WP8 code
    wrote."""
    import random as _random
    from sixg_sim.reward_spec import (FROZEN, GlobalRewardInputs,
                                      LocalRewardInputs, global_reward,
                                      local_reward)

    assert FROZEN.relay_throughput_mbps_per_point == 50.0
    assert FROZEN.local_relay_throughput_per_mbps == 1.0 / 50.0

    rng = _random.Random(20260924)
    xs = [142.97908624570675] + [rng.uniform(0.0, 240.0) for _ in range(4000)]
    distinguishing = 0
    for x in xs:
        if x / 50.0 > FROZEN.relay_throughput_cap:
            continue                      # the cap binds; both spellings tie
        t = global_reward(GlobalRewardInputs(island_mode=True,
                                             relay_throughput_mbps=x))
        assert t.relay_throughput == x / 50.0, (x, t.relay_throughput)
        if x / 50.0 != x * (1.0 / 50.0):
            distinguishing += 1
            assert t.relay_throughput != x * (1.0 / 50.0), x

        loc = local_reward(LocalRewardInputs(in_relay_mode=True,
                                             relay_carried_mbps=x))
        if x * (1.0 / 50.0) <= FROZEN.local_relay_throughput_cap:
            assert loc.relay_throughput == x * (1.0 / 50.0), x

    # NON-VACUITY: if no draw distinguished the two spellings, the assertions
    # above proved nothing about which one the term uses.
    assert distinguishing > 100, (
        'only %d of %d draws distinguish x/50 from x*(1/50); this test is '
        'not measuring what it claims' % (distinguishing, len(xs)))
    # the documented example, exactly
    assert (142.97908624570675 / 50.0) != (142.97908624570675 * (1.0 / 50.0))


# ── 2. the churn term is scored on the EXECUTED mode ─────────────────────
def test_local_churn_follows_the_executed_mode_not_the_sampled_action():
    """A hand-constructed engine hold: the agent's ACTION asks for a
    different relay mode on every call while the engine keeps reverting the
    state to the incumbent.  The churn cost must be zero throughout, because
    a held teardown is environment dynamics, not a policy choice."""
    from sixg_sim.agent import (RLAgent, AgentObservation,
                                NeighborRadioSummary, ConnectivityState,
                                PHYMACAction, relay_enum_index)
    from sixg_sim.phy_mac_state import PHYMACState, RelayMode
    from sixg_sim.traffic import SliceDictionary

    ps = PHYMACState(node_id='site_0')
    ps.relay_mode = RelayMode.OFF            # what the ENGINE executed
    obs = AgentObservation(
        node_id='site_0', is_island=True, energy_soc=0.8, phy_mac=ps,
        neighbor_radio=NeighborRadioSummary(),
        connectivity=ConnectivityState(),
        ticks_since_severance=0, current_tick=0)
    agent = RLAgent('site_0', SliceDictionary(), is_training=True)

    def _act(head):
        return PHYMACAction(relay_head_idx=head,
                            relay_mode_idx=relay_enum_index(head))

    # Tick 1 establishes the incumbent; ticks 2..5 SAMPLE a different mode
    # every time while the executed state never moves.
    charged = []
    for head in (0, 2, 1, 2, 0):
        comps = agent.calculate_reward(obs, _act(head), obs)
        charged.append(agent._reward_terms['relay_churn'])
        assert comps.total_reward == comps.total_reward   # not NaN
    assert charged == [0.0] * 5, (
        'the churn term moved while the EXECUTED mode was pinned: %r -- it '
        'is being scored on the sampled action' % (charged,))

    # Now the ENGINE actually applies a change: the cost appears, once.
    ps.relay_mode = RelayMode.CAPACITY_BOOST
    agent.calculate_reward(obs, _act(2), obs)
    from sixg_sim.reward_spec import FROZEN
    assert (agent._reward_terms['relay_churn']
            == -FROZEN.local_relay_churn), agent._reward_terms
    # and it is not charged again while the mode holds
    agent.calculate_reward(obs, _act(0), obs)
    assert agent._reward_terms['relay_churn'] == 0.0, agent._reward_terms


def test_team_churn_counts_executed_modes_under_a_forced_engine_hold():
    """The same contract on the LIVE simulator: with every relay teardown
    blocked, the sampled relay actions keep changing and the team objective's
    churn term stays at zero."""
    import run_timeline_comparison as R
    from sixg_sim.simulation import Simulator, SimulationConfig

    topo, _info = R.build_comparison_topology(SEED, log_prefix='[WP8] ')
    scen = R.build_comparison_scenario('recovery', SEED, topology=topo,
                                       ticks=TICKS, severance_tick=20,
                                       scenario_seed=SEED)
    sim = Simulator(topo, scen, SimulationConfig(enable_island_detection=True,
                                                 verbose=False,
                                                 random_seed=SEED))
    sim._current_scenario_type = 'recovery'
    sim.random_action_policy = True          # exercises the widest action set
    sim.marl_ue_routing_enabled = True
    sim.island_mode = True
    sim.control_plane.set_island_mode(True)

    # THE ENGINE HOLD: every relay teardown is blocked, so `_step_phy_mac`
    # reverts the requested mode to the incumbent on every site, every tick.
    sim._relay_teardown_blocked = lambda node_id: True

    sampled_heads = set()
    churn_seen = []
    for tick in range(1, 41):
        sim.current_tick = tick
        sim._forward_traffic(sim._generate_traffic())
        obs = sim._build_agent_observations()
        sim._execute_agent_actions(obs)
        sim.compute_global_connectivity_reward()
        churn_seen.append(sim._reward_terms['relay_churn'])
        for a in sim.agents.values():
            act = getattr(a, 'last_action', None)
            if act is not None:
                sampled_heads.add(int(act.relay_head_idx))

    # NON-VACUITY, named: the sampled action really did vary, so a churn term
    # scored on the sampled action would have been non-zero.
    assert len(sampled_heads) > 1, (
        'the uniform arm sampled a single relay head (%r): this run cannot '
        'distinguish the executed mode from the sampled action'
        % sorted(sampled_heads))
    assert set(churn_seen) == {0.0}, (
        'the team churn term fired while every teardown was held: %r'
        % sorted(set(churn_seen)))


def test_ppo_scores_the_sampled_action_not_the_executed_mode():
    """The stored action a PPO minibatch is built from is the SAMPLED head
    index.  With every teardown held, the executed mode is pinned while the
    stored actions still vary -- so PPO is scoring what the policy chose."""
    import run_timeline_comparison as R
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.reward_spec import executed_relay_mode

    topo, _info = R.build_comparison_topology(SEED, log_prefix='[WP8] ')
    scen = R.build_comparison_scenario('recovery', SEED, topology=topo,
                                       ticks=TICKS, severance_tick=20,
                                       scenario_seed=SEED)
    sim = Simulator(topo, scen, SimulationConfig(enable_island_detection=True,
                                                 verbose=False,
                                                 random_seed=SEED))
    sim._current_scenario_type = 'recovery'
    sim.random_action_policy = True
    sim.marl_ue_routing_enabled = True
    sim.island_mode = True
    sim.control_plane.set_island_mode(True)
    sim._relay_teardown_blocked = lambda node_id: True

    stored, executed = set(), set()
    for tick in range(1, 21):
        sim.current_tick = tick
        sim._forward_traffic(sim._generate_traffic())
        obs = sim._build_agent_observations()
        sim._execute_agent_actions(obs)
        for nid, a in sim.agents.items():
            act = getattr(a, 'last_action', None)
            if act is None:
                continue
            # exactly what sixg_sim/worker.py stores into the PPO pool
            stored.add(int(act.relay_head_idx))
            ps = sim.phy_mac_states.get(nid)
            if ps is not None:
                executed.add(executed_relay_mode(ps))

    assert len(stored) > 1, (
        'the stored PPO action never varied, so this run proves nothing: %r'
        % sorted(stored))
    assert len(executed) == 1, (
        'the engine hold did not pin the executed mode: %r'
        % sorted(m.name for m in executed))


# ── 3. no removed term survives anywhere in the tree ─────────────────────
# Identifiers that existed ONLY to carry a term WP8 deleted, or to select a
# coefficient set from the environment.  AST references, not line text: a
# comment is not in the tree at all and prose never equals a bare identifier,
# so the scan structurally ignores rationale while catching live code.
REMOVED_IDENTIFIERS = {
    'QOS_POINTS', 'RELAY_CONTRIB_POINTS', 'RELAY_CONTRIB_PER_NODE',
    'BRIDGE_HELD_POINTS', 'BRIDGE_CF_POINTS', 'RELAY_CHURN_POINTS',
    'MCS_OVERREACH_POINTS', 'ENERGY_POINTS', 'DELIVERY_POINTS',
    'GEN_DELIVERY_POINTS', 'RELAY_TPUT_CAP', 'RELAY_TPUT_CAP_GLOBAL',
    'RELAY_TPUT_PER_MBPS', 'RELAY_CHURN_PENALTY', 'BRIDGE_LOSS_PENALTY',
    'REUNIFY_FULL_BONUS', 'REUNIFY_PER_FRAGMENT', 'REUNIFY_RESIDUAL',
    'REUNIFY_REF_FRAGMENTS', 'REACH_RESTORED_PER_UE', 'REACH_RESTORED_CAP',
    'REACH_REQUIRE_TRAFFIC',
    # the simulator's own copy of the PUBLISHED reunification cadence,
    # deleted by D8.23 in favour of `coef.reunify_eval_every`
    'REUNIFY_EVAL_EVERY',
    'reward_profile', 'RewardProfile', 'routed_flow_cliff',
    'relay_terms_use_delivered_ledger', 'reach_require_traffic',
    'relay_contrib_per_node', 'relay_contrib_points', 'bridge_held_points',
    'bridge_cf_points', 'delivery_points', 'gen_delivery_points',
}


def _removed_hits(src, rel):
    hits = set()
    for node in ast.walk(ast.parse(src, filename=rel)):
        if isinstance(node, ast.Name) and node.id in REMOVED_IDENTIFIERS:
            hits.add((node.lineno, node.id))
        elif (isinstance(node, ast.Attribute)
              and node.attr in REMOVED_IDENTIFIERS):
            hits.add((node.lineno, node.attr))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value in REMOVED_IDENTIFIERS):
            hits.add((node.lineno, node.value))
        elif isinstance(node, ast.alias):
            for part in (node.name or '').split('.'):
                if part in REMOVED_IDENTIFIERS:
                    hits.add((getattr(node, 'lineno', 0), part))
    return hits


def test_no_removed_reward_term_is_referenced_anywhere():
    found = {}
    me = os.path.abspath(__file__)
    for path in _py_files():
        if os.path.abspath(path) == me:
            continue               # this file NAMES them, on purpose
        with open(path, encoding='utf-8') as f:
            src = f.read()
        hits = _removed_hits(src, path)
        if hits:
            found[os.path.relpath(path, ROOT)] = sorted(hits)
    assert not found, (
        'a term or coefficient WP8 removed is still referenced in live code: '
        '%r' % found)


# The two sites in `sixg_sim/simulation.py` where a deleted reward constant
# is still named IN A COMMENT, and which this branch is NOT allowed to edit:
# `peer_reachable_ue_stats` is the graph/reachability measurement region and
# `_forward_traffic` is the forwarding region, both owned by the engine
# track (D8.17).  The scan below refuses any OTHER such comment, so the set
# cannot grow, and it is written as a subset test so that repointing these
# two -- which the owning track can do in one line at merge time -- keeps it
# green.  Replacement fields: `reach_per_ue` / `reach_cap` in
# sixg_sim/reward_spec.py.
STALE_COMMENT_REGIONS = {
    'peer_reachable_ue_stats': 'graph/reachability measurement region',
    '_forward_traffic': 'forwarding region',
}


def test_no_deleted_reward_constant_survives_in_a_simulator_comment():
    """The AST scan above is blind to comments, and five of them still named
    reward constants WP8 had deleted -- the exact documentation drift this WP
    exists to end, on the file where the reward used to live.  Three are
    repointed at `sixg_sim/reward_spec.py`'s field names; the two in regions
    this branch may not edit are enumerated above and may not multiply."""
    import io
    import tokenize

    path = os.path.join(ROOT, 'sixg_sim', 'simulation.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    spans = []
    for node in ast.walk(ast.parse(src, filename=path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    spans.sort()

    def enclosing(line):
        name = '<class or module body>'
        for start, end, nm in spans:
            if start <= line <= end:
                name = nm          # innermost wins: spans are sorted by start
        return name

    upper = sorted(n for n in REMOVED_IDENTIFIERS if n.upper() == n)
    assert len(upper) > 15, upper            # the scan really has a haystack
    hits = []
    with open(path, encoding='utf-8') as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type != tokenize.COMMENT:
                continue
            for name in upper:
                if name in tok.string:
                    hits.append((tok.start[0], name, enclosing(tok.start[0])))

    unexpected = [h for h in hits if h[2] not in STALE_COMMENT_REGIONS]
    assert not unexpected, (
        'sixg_sim/simulation.py names a DELETED reward constant in a comment '
        'outside the two accepted regions: %r.  Repoint it at the '
        'sixg_sim/reward_spec.py field that replaced it.' % (unexpected,))
    assert len(hits) <= len(STALE_COMMENT_REGIONS), (
        'the accepted stale-comment sites multiplied: %r' % (hits,))
    # the scan can see a comment at all -- otherwise "no hits" means nothing
    with open(path, encoding='utf-8') as f:
        comments = [t for t in tokenize.generate_tokens(f.readline)
                    if t.type == tokenize.COMMENT]
    assert len(comments) > 500, len(comments)


def test_the_removed_module_is_gone():
    """`sixg_sim/reward_profile.py` selected one of eight coefficient sets
    from MARL_REWARD_PROFILE at import time, and the EVALUATED set was not
    its default -- so the rates depended on whether a shell script had
    exported the variable.  The module is deleted."""
    assert not os.path.exists(os.path.join(ROOT, 'sixg_sim',
                                           'reward_profile.py'))
    raised = None
    try:
        __import__('sixg_sim.reward_profile')
    except ImportError as e:
        raised = e
    assert raised is not None, 'sixg_sim.reward_profile still imports'


def test_the_environment_cannot_move_a_coefficient():
    """The tripwire: the evaluated scripts still export
    MARL_REWARD_PROFILE=mission and must keep working, while any other value
    RAISES rather than being silently ignored."""
    import subprocess
    stub = ('import sixg_sim.reward_spec as R; '
            'print(R.FROZEN.kappa, R.reward_id())')
    for value, ok in (('mission', True), ('', True), ('full', False),
                      ('prev', False), ('reunify_boost', False)):
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = '0'
        if value:
            env['MARL_REWARD_PROFILE'] = value
        else:
            env.pop('MARL_REWARD_PROFILE', None)
        r = subprocess.run([sys.executable, '-c', stub], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if ok:
            assert r.returncode == 0, (value, r.stderr[-600:])
            assert '0.6' in r.stdout, (value, r.stdout)
        else:
            assert r.returncode != 0, (
                'MARL_REWARD_PROFILE=%r was accepted: a coefficient set can '
                'still be chosen from the environment' % value)
            assert 'FROZEN' in r.stderr, (value, r.stderr[-600:])


# ── 4. the dumped coefficient set matches the object, field by field ─────
FROZEN_DOC = os.path.join(ROOT, 'docs', 'REWARD_FROZEN.md')


def test_the_dumped_coefficient_file_matches_the_object_field_by_field():
    from sixg_sim.reward_spec import FROZEN, coefficients_text, REMOVED_TERMS
    from dataclasses import fields as dc_fields

    assert os.path.exists(FROZEN_DOC), (
        '%s is missing: regenerate with '
        '`python3 -m sixg_sim.reward_spec > docs/REWARD_FROZEN.md`'
        % FROZEN_DOC)
    with open(FROZEN_DOC, encoding='utf-8') as f:
        on_disk = f.read()
    assert on_disk == coefficients_text(), (
        'docs/REWARD_FROZEN.md is stale: it is the pre-registration copy of '
        'the frozen coefficients and it no longer equals the object.  '
        'Regenerate it.')

    # FIELD BY FIELD, from the file's own table rows -- not by re-rendering.
    rows = {}
    for line in on_disk.splitlines():
        line = line.strip()
        if line.startswith('| `') and line.endswith('|'):
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) == 2 and parts[0].startswith('`'):
                rows[parts[0].strip('`')] = parts[1]
    for f in dc_fields(FROZEN):
        assert f.name in rows, (
            'coefficient %r is not in docs/REWARD_FROZEN.md' % f.name)
        assert rows[f.name] == repr(getattr(FROZEN, f.name)), (
            f.name, rows[f.name], repr(getattr(FROZEN, f.name)))
    for name, _where, _why in REMOVED_TERMS:
        assert ('`%s`' % name) in on_disk, (
            'removed term %r is not listed in the pre-registration text'
            % name)

    # ...and the converse disclosure: a term that is KEPT although the
    # reference policy never earned it.  Without it a reader of the
    # pre-registration cannot learn that a published +25 bonus was never paid
    # in any measured pass.
    from sixg_sim.reward_spec import KEPT_UNEARNED_TERMS
    assert KEPT_UNEARNED_TERMS, 'the kept-but-unearned disclosure is empty'
    assert 'Terms KEPT that the reference policy never earns' in on_disk
    for name, _where, cond, measured in KEPT_UNEARNED_TERMS:
        assert ('`%s`' % name) in on_disk, (
            'kept-but-unearned term %r is not in the pre-registration text'
            % name)
        assert cond in on_disk and measured in on_disk, name
        # the measured evidence, not just the claim
        assert 'reward_terms_wp8.json' in measured, name
    kept = dict((n, m) for n, _w, _c, m in KEPT_UNEARNED_TERMS)
    assert '0.4714' in kept['full_delivery'] and '0.5279' in kept['full_delivery']


def test_no_coefficient_is_zero():
    """RERUN_SPEC WP8: "do not keep a term multiplied by zero".  Every RATE
    in the frozen set must be non-zero -- a zero rate is a term a reader has
    to evaluate before learning that it does nothing, and it is how three of
    the removed terms survived two repricings."""
    from sixg_sim.reward_spec import FROZEN
    from dataclasses import fields as dc_fields

    # Fields that are not rates: thresholds, reference counts and cadences.
    NOT_A_RATE = {'full_delivery_threshold', 'recovery_ramp_ticks',
                  'recovery_complete_delivered', 'recovery_complete_ramp',
                  'reunify_ref_fragments', 'reunify_eval_every',
                  'normal_spare_prb_threshold', 'local_energy_ref_db',
                  'kappa'}
    for f in dc_fields(FROZEN):
        v = getattr(FROZEN, f.name)
        assert v != 0, ('coefficient %r is zero: delete the term instead'
                        % f.name)
        if f.name not in NOT_A_RATE:
            assert v > 0, ('rate %r is negative; the SIGN of a cost belongs '
                           'in the term, not in its rate' % f.name)


def test_the_reward_id_is_wired_into_provenance():
    """Every checkpoint and result pickle records WHICH reward trained it."""
    from sixg_sim import runinfo
    from sixg_sim.reward_spec import reward_id

    rec = runinfo.provenance()
    assert rec['reward_id'] == reward_id(), rec.get('reward_id')
    assert rec['reward_id'].startswith('rew1-'), rec['reward_id']
    # it MOVES when a coefficient moves, which is the whole point
    import sixg_sim.reward_spec as RS
    saved = RS.FROZEN
    try:
        RS.FROZEN = RS.RewardCoefficients(delivery=99.0)
        assert RS.reward_id() != rec['reward_id']
    finally:
        RS.FROZEN = saved
    assert RS.reward_id() == rec['reward_id']


def test_the_episode_shape_is_disclosed():
    """RERUN_SPEC WP8: 800-tick training episodes, truncation treated as
    terminal, disclosed in docs/."""
    from sixg_sim.reward_spec import EPISODE_TICKS, TRUNCATION_IS_TERMINAL

    assert EPISODE_TICKS == 800
    assert TRUNCATION_IS_TERMINAL is True
    with open(FROZEN_DOC, encoding='utf-8') as f:
        doc = f.read()
    assert '800 ticks' in doc and 'TERMINAL' in doc, (
        'the episode shape is not disclosed in docs/REWARD_FROZEN.md')

    # The BEHAVIOUR is checked by
    # test_truncation_is_terminal_in_the_trainers_behaviour below.  What is
    # kept here is only a SECONDARY tripwire, and it reads the CODE, not the
    # docstring: `assert 'done' in inspect.getsource(finish_episode)` used to
    # be the only check there was, and the word `done` occurs in that
    # method's own docstring, so it held whatever the code did (D8.25).
    import inspect
    from sixg_sim.mappo_trainer import MAPPOTrainer
    fn = ast.parse(_dedent(
        inspect.getsource(MAPPOTrainer.finish_episode))).body[0]
    stripped = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                               and isinstance(fn.body[0].value, ast.Constant)
                               and isinstance(fn.body[0].value.value, str)
                               ) else fn.body
    assert len(stripped) < len(fn.body), (
        'finish_episode no longer opens with a docstring, so the strip below '
        'is removing a statement instead')
    code = chr(10).join(ast.dump(st) for st in stripped)
    assert "'dones'" in code, (
        'finish_episode does not touch the done flags at all: ' + code[:400])


# ── 5. the live drivers evaluate the object, and a short training smoke ──
def test_the_simulator_and_the_agent_hold_no_reward_rate():
    """The measurement stays in the environment; the rates live in one
    object.  This is the structural half of "what is published and what runs
    cannot drift apart": if an arithmetic literal creeps back into either
    reward method, it is no longer true that the dumped file describes the
    running system."""
    import inspect
    from sixg_sim.simulation import Simulator
    from sixg_sim.agent import RLAgent

    for fn in (Simulator.compute_global_connectivity_reward,
               RLAgent.calculate_reward):
        src = inspect.getsource(fn)
        body = ast.parse(_dedent(src)).body[0]
        lits = []
        for node in ast.walk(body):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, (int, float))
                    and not isinstance(node.value, bool)):
                if node.value not in (0, 1, 0.0, 1.0):
                    lits.append((node.lineno, node.value))
        assert not lits, (
            '%s carries numeric literals that are neither 0 nor 1: %r.  '
            'Every rate belongs in sixg_sim/reward_spec.py.'
            % (fn.__qualname__, lits))
        assert 'reward_spec' in src, (
            '%s does not delegate to reward_spec' % fn.__qualname__)


def _dedent(src):
    lines = src.splitlines()
    pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()),
              default=0)
    return '\n'.join(l[pad:] if len(l) >= pad else l for l in lines) + '\n'


def test_the_live_driver_mixes_at_the_frozen_kappa():
    """r_i = 0.4 * local + 0.6 * team ON THE PATH THAT TRAINS.

    One real (short) training episode through
    `train_on_comparison.run_comparison_worker_episode`, with the three live
    call sites recorded as they fire: the team objective the simulator
    computes, the agent-local reward the agent computes, and the scalar that
    reaches `MAPPOTrainer.collect` -- the number PPO is handed.  Each triple
    is checked against a kappa mixture computed here, in this file, from the
    literal 0.4 and 0.6.

    This replaces a check that pinned `reward_spec.mix`, which at the time
    had NO callers: both drivers mixed with `args.global_reward_mix`, whose
    default was an independent literal.  The last clause of the test is that
    a driver handed any other weight now FAILS instead of training on it.
    """
    import train_on_comparison as T
    from sixg_sim.agent import CriticNetwork, PolicyNetwork, RLAgent
    from sixg_sim.mappo_trainer import MAPPOTrainer
    from sixg_sim.simulation import Simulator
    from sixg_sim.reward_spec import KAPPA

    team = {'v': None}
    local = {}
    seen = []
    real_team = Simulator.compute_global_connectivity_reward
    real_local = RLAgent.calculate_reward
    real_collect = MAPPOTrainer.collect

    def spy_team(self, *a, **k):
        v = real_team(self, *a, **k)
        team['v'] = float(v)
        return v

    def spy_local(self, *a, **k):
        comps = real_local(self, *a, **k)
        local[self.node_id] = float(comps.total_reward)
        return comps

    def spy_collect(self, agent_id, obs_tensor, action_indices, prb_frac,
                    log_prob, value, reward, *a, **k):
        if agent_id in local and team['v'] is not None:
            seen.append((agent_id, local[agent_id], team['v'], float(reward)))
        return real_collect(self, agent_id, obs_tensor, action_indices,
                            prb_frac, log_prob, value, reward, *a, **k)

    actor, critic = PolicyNetwork(), CriticNetwork()
    job = {
        'episode_idx': 0,
        'topology_seed': SEED,
        'scenario_kind': 'rescue_ops',   # severed at t=0: transitions early
        'episode_ticks': 80,
        'event_seed': SEED,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'global_reward_mix': KAPPA,
        'torch_seed': SEED,
    }
    Simulator.compute_global_connectivity_reward = spy_team
    RLAgent.calculate_reward = spy_local
    MAPPOTrainer.collect = spy_collect
    try:
        T.run_comparison_worker_episode(dict(job))
    finally:
        Simulator.compute_global_connectivity_reward = real_team
        RLAgent.calculate_reward = real_local
        MAPPOTrainer.collect = real_collect

    # NON-VACUITY, named: a run that collected nothing checks nothing.
    assert len(seen) >= 20, (
        'only %d transitions reached MAPPOTrainer.collect: the live mixing '
        'was never exercised, so this test proves nothing' % len(seen))

    moved = 0
    for aid, loc, tm, r in seen:
        expect = 0.4 * loc + 0.6 * tm
        assert abs(r - expect) < 1e-9, (
            'the driver mixed %r + %r into %r, not %r (kappa=%r)'
            % (loc, tm, r, expect, KAPPA))
        if abs(loc - tm) > 1e-9:
            moved += 1
            # a mixture is only distinguishable from its parts when they differ
            assert abs(r - loc) > 1e-12 and abs(r - tm) > 1e-12, (aid, loc, tm, r)
    assert moved >= 10, (
        'the local and team rewards were equal on %d of %d transitions, so '
        'the weight could not be read off them' % (len(seen) - moved, len(seen)))

    # ...and no other weight can reach that arithmetic
    bad = dict(job)
    bad['global_reward_mix'] = 0.3
    raised = False
    try:
        T.run_comparison_worker_episode(bad)
    except ValueError as exc:
        raised = True
        assert 'FROZEN' in str(exc), str(exc)
    assert raised, (
        'the worker accepted global_reward_mix=0.3: a run can train at one '
        'kappa while every artefact records another')


def test_training_smoke_on_the_frozen_reward():
    """Two tiny training episodes end to end on the live drivers: the reward
    the worker mixes is finite, the pool fills, and a REAL PPO update runs --
    `MAPPOTrainer.update`, the path that actually trains, not a convenience
    method."""
    import torch
    import train_on_comparison as T
    from sixg_sim.agent import (CriticNetwork, PolicyNetwork, RLAgent,
                                CRITIC_OBS_DIM)
    from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
    from sixg_sim.reward_spec import KAPPA
    from sixg_sim.traffic import SliceDictionary

    actor, critic = PolicyNetwork(), CriticNetwork()
    out = T.run_comparison_worker_episode({
        'episode_idx': 0,
        'topology_seed': SEED,
        'scenario_kind': 'rescue_ops',
        'episode_ticks': TICKS,
        'event_seed': SEED,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'global_reward_mix': KAPPA,       # the frozen kappa, not a guess
        'torch_seed': SEED,
    })
    pool = out['pool_data']
    assert pool['obs'], 'no transitions were collected'
    assert all(math.isfinite(r) for r in pool['returns']), 'non-finite return'
    assert all(math.isfinite(a) for a in pool['advantages'])
    # a reward that never moved would train nothing
    assert len(set(round(float(r), 6) for r in pool['returns'])) > 1, (
        'every return is identical: the reward carries no signal at all')

    sd_ = SliceDictionary()
    agents = {'a%d' % i: RLAgent('a%d' % i, sd_, is_training=True)
              for i in range(2)}
    cfg = MAPPOConfig(mini_batch=32, n_epochs=2)
    trainer = MAPPOTrainer(agents, CriticNetwork(CRITIC_OBS_DIM), cfg)
    n = min(len(pool['obs']), 256)
    aids = list(agents)
    for i in range(n):
        trainer.collect(
            aids[i % 2],
            torch.as_tensor(pool['obs'][i], dtype=torch.float32),
            pool['actions'][i], list(pool['prb_acts'][i]),
            float(pool['log_probs'][i]), 0.0, 0.0, i == n - 1,
            global_obs=torch.as_tensor(pool['global_obs'][i],
                                       dtype=torch.float32))
    trainer.finish_episode()
    metrics = trainer.update()
    assert metrics['updates'] > 0, metrics
    assert math.isfinite(metrics['policy_loss'])
    assert math.isfinite(metrics['value_loss'])
    assert math.isfinite(metrics['entropy'])


# ── 5b. a published coefficient with no reader; the episode length ───────
_COEF_BASES = frozenset(('coef', '_coef', 'FROZEN', 'COEF', 'coefficients',
                         'RewardCoefficients'))


def _attr_base_tail(node):
    """The LAST identifier of the expression an attribute is read off.

    `coef.delivery` -> 'coef'; `self._coef.delivery` -> '_coef';
    `RS.FROZEN.kappa` -> 'FROZEN'; `RewardCoefficients().kappa` -> the call's
    own tail.  Anything else -> None."""
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _coefficient_reads(tree):
    """Every `<coefficient-object>.<field>` read in one parsed module.

    Returns {field_name: [lineno, ...]}.  Deliberately NARROW: a read counts
    only when the base expression is one of the names this tree spells a
    coefficient set with, so `GlobalRewardTerms.delivery` -- a RESULT field
    that happens to share a name with a RATE -- is not mistaken for a reader
    of the rate."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if _attr_base_tail(node.value) in _COEF_BASES:
                out.setdefault(node.attr, []).append(node.lineno)
        elif (isinstance(node, ast.Call)
              and getattr(node.func, 'id', '') == 'getattr'
              and len(node.args) >= 2
              and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)
              and _attr_base_tail(node.args[0]) in _COEF_BASES):
            out.setdefault(node.args[1].value, []).append(node.lineno)
    return out


def _rel(path):
    return os.path.relpath(path, ROOT).replace(chr(92), '/')


def _production_py_files():
    """The RUNNING system: every .py in the tree that is neither a test nor a
    diagnostic tool.  A coefficient whose only reader is a test or a dump is
    a coefficient the evaluated runs do not use."""
    out = []
    for path in _py_files():
        rel = _rel(path)
        if rel.startswith('tests/') or rel.startswith('tools/'):
            continue
        if os.path.basename(rel).startswith('test_'):
            continue
        out.append(path)
    return out


def test_every_published_coefficient_has_a_reader():
    """A published, HASHED coefficient that no running code reads is this
    WP's own drift, pointing the other way.

    `reunify_eval_every` was rendered into docs/REWARD_FROZEN.md and hashed
    into `reward_id()` while the cadence that ran was an independent
    `Simulator.REUNIFY_EVAL_EVERY = 5`: editing the simulator's copy moved
    the reward trajectory without moving the published number or the hash
    (D8.23).  Every field of `RewardCoefficients` must be READ by the
    production path -- not merely rendered, which is what the
    pre-registration table does, from a list of strings."""
    from dataclasses import fields as dc_fields
    from sixg_sim.reward_spec import FROZEN

    # the scanner itself, on a snippet whose answer is known: it must SEE a
    # real read and must NOT see a same-named field of a result object
    probe = ast.parse('a = coef.delivery + terms.delivery + _coef.kappa'
                      + chr(10) + 'b = getattr(FROZEN, "reach_cap")'
                      + chr(10) + 'c = delivery' + chr(10))
    seen = _coefficient_reads(probe)
    assert sorted(seen) == ['delivery', 'kappa', 'reach_cap'], sorted(seen)
    assert len(seen['delivery']) == 1, (
        'the scan counted terms.delivery as a read of the RATE delivery')

    names = [f.name for f in dc_fields(FROZEN)]
    assert len(names) >= 25, 'only %d coefficients?' % len(names)

    readers = dict((n, []) for n in names)
    n_files = n_reads = 0
    for path in _production_py_files():
        with open(path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
        n_files += 1
        for field, lines in _coefficient_reads(tree).items():
            if field in readers:
                readers[field].extend('%s:%d' % (_rel(path), ln)
                                      for ln in lines)
                n_reads += len(lines)

    # WHAT WAS CHECKED, so a pass cannot be vacuous
    assert n_files >= 10, 'the production scan covered only %d files' % n_files
    assert n_reads >= len(names), (
        'only %d coefficient reads found in %d production files'
        % (n_reads, n_files))

    unread = sorted(n for n in names if not readers[n])
    assert not unread, (
        '%d of %d published coefficients are READ NOWHERE in the production '
        'path: %r.  They are rendered into docs/REWARD_FROZEN.md and hashed '
        'into reward_id() while something else governs the behaviour they '
        'name -- either bind the running value to the field, or delete the '
        'field, its entry in the render list and therefore the hash payload '
        '(D8.23).  [scanned %d files, %d reads]'
        % (len(unread), len(names), unread, n_files, n_reads))

    # and the field this contract was written for is bound where claimed
    assert any(r.startswith('sixg_sim/simulation.py:')
               for r in readers['reunify_eval_every']), (
        'reunify_eval_every is read, but not by the simulator: %r'
        % (readers['reunify_eval_every'],))


def test_no_second_copy_of_the_reunification_cadence_can_reappear():
    """The other half of D8.23: binding the field is durable only if a second
    independent copy cannot come back beside it."""
    offenders = []
    for path in _py_files():
        rel = _rel(path)
        if rel.startswith('tests/') or rel == 'sixg_sim/reward_spec.py':
            continue
        with open(path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for t in targets:
                name = getattr(t, 'id', getattr(t, 'attr', '')) or ''
                if 'REUNIFY_EVAL' in name.upper():
                    offenders.append('%s:%d %s' % (rel, node.lineno, name))
    assert not offenders, (
        'an independent copy of the reunification cadence is back: %r.  The '
        'cadence IS the published field reward_spec.reunify_eval_every, and '
        'the simulator reads it off the coefficient object (D8.23)'
        % offenders)


# A file that carries the literal 800 for something that is NOT a training
# episode.  Named, with the reason, so the scan below has no silent holes;
# the test fails if the file goes away or stops carrying the literal.
_EPISODE_TICKS_EXEMPT = {
    'sixg_sim/gen_demo_kpis.py':
        "TICKS = 800 is the length of a SYNTHETIC demonstration series this "
        "script fabricates for a convergence-shape plot -- it runs no "
        "simulator, trains nothing and is on no evaluated path (untouched "
        "since the baseline 9dec943).  Binding it to the published training "
        "horizon would assert a relationship that does not exist.",
}


def _episode_site_values():
    """{site: the episode length that site RESOLVES to}, read off the running
    code (carry-forward defect 4 of the trackAgent review): a site that
    imports the constant and then ignores it resolves to something else.

      * train_on_comparison.py: `resolve_frozen_run_shape` on a namespace
        with the flag unset (the path main() takes);
      * make_untrained_checkpoints.py: the module's EPISODE_TICKS, and the
        name the metadata dict writes under 'episode_ticks';
      * sixg_sim/main.py: the post-parse block, AST-checked, since main()
        parses the command line (the flag defaults to None and is then
        assigned from reward_spec.EPISODE_TICKS; no getattr fallback);
      * plot_severance_maps.py: load_training_plan()['episode_ticks'].
    """
    import argparse
    out = {}
    argv = sys.argv
    sys.argv = ['train_on_comparison.py']
    try:
        import train_on_comparison as T
    finally:
        sys.argv = argv
    ns = T.resolve_frozen_run_shape(argparse.Namespace(
        episode_ticks=None, global_reward_mix=None))
    out['train_on_comparison.py'] = ns.episode_ticks

    import make_untrained_checkpoints as M
    with open(os.path.join(ROOT, 'make_untrained_checkpoints.py'),
              encoding='utf-8') as f:
        mtree = ast.parse(f.read())
    meta = [kv for node in ast.walk(mtree) if isinstance(node, ast.Dict)
            for kv in zip(node.keys, node.values)
            if isinstance(kv[0], ast.Constant) and kv[0].value == 'episode_ticks']
    assert meta and all(isinstance(v, ast.Name) and v.id == 'EPISODE_TICKS'
                        for _k, v in meta), 'make_untrained_checkpoints ' \
        'writes episode_ticks from something other than EPISODE_TICKS'
    out['make_untrained_checkpoints.py'] = M.EPISODE_TICKS

    with open(os.path.join(ROOT, 'sixg_sim', 'main.py'), encoding='utf-8') as f:
        main_tree = ast.parse(f.read())
    assigned = [n for n in ast.walk(main_tree)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute)
                and n.targets[0].attr == 'episode_ticks'
                and isinstance(n.value, ast.Name)
                and n.value.id == 'EPISODE_TICKS']
    default_none = [kw for n in ast.walk(main_tree)
                    if isinstance(n, ast.Call)
                    and getattr(n.func, 'attr', '') == 'add_argument'
                    and any(isinstance(a, ast.Constant) and a.value == '--episode-ticks'
                            for a in n.args)
                    for kw in n.keywords if kw.arg == 'default'
                    and isinstance(kw.value, ast.Constant) and kw.value.value is None]
    fallbacks = [n for n in ast.walk(main_tree)
                 if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'getattr'
                 and len(n.args) == 3 and isinstance(n.args[1], ast.Constant)
                 and n.args[1].value == 'episode_ticks']
    assert len(assigned) == 1 and len(default_none) == 1 and not fallbacks, (
        'sixg_sim/main.py: expected --episode-ticks default=None, ONE '
        'args.episode_ticks = EPISODE_TICKS, and no getattr fallback; found '
        '%d, %d, %d' % (len(assigned), len(default_none), len(fallbacks)))
    from sixg_sim.reward_spec import EPISODE_TICKS
    out['sixg_sim/main.py'] = EPISODE_TICKS if assigned else None

    import plot_severance_maps as P
    plan = P.load_training_plan()
    assert plan is not None, 'plot_severance_maps.load_training_plan() failed'
    out['plot_severance_maps.py'] = plan['episode_ticks']
    return out


def _stray_episode_literals(tree, value):
    """Every place in `tree` that binds `value` to an episode length: a
    keyword episode_ticks=/ticks=, an argparse default of a tick/episode
    option, a getattr(..., '<tick/episode name>', value) fallback, and an
    assignment -- plain or tuple-unpacking -- to a TICK/EPISODE-named target.
    (Carry-forward defect 1: `default=800` inside add_argument and
    `ticks, base = 800, 42` were invisible to the earlier scan.)"""
    hits = []

    def tickish(s):
        s = str(s).upper()
        return 'TICK' in s or 'EPISODE' in s

    def is_val(v):
        return isinstance(v, ast.Constant) and v.value == value \
            and not isinstance(v.value, bool)

    for node in ast.walk(tree):
        if (isinstance(node, ast.keyword)
                and node.arg in ('episode_ticks', 'ticks')
                and is_val(node.value)):
            hits.append((node.value.lineno, '%s=%r' % (node.arg, value)))
        if isinstance(node, ast.Call):
            fn = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
            if fn == 'add_argument' and any(
                    isinstance(a, ast.Constant) and tickish(a.value)
                    for a in node.args):
                for kw in node.keywords:
                    if kw.arg == 'default' and is_val(kw.value):
                        hits.append((node.lineno, 'add_argument default=%r'
                                     % value))
            if (fn == 'getattr' and len(node.args) == 3
                    and isinstance(node.args[1], ast.Constant)
                    and tickish(node.args[1].value) and is_val(node.args[2])):
                hits.append((node.lineno, 'getattr(..., %r, %r)'
                             % (node.args[1].value, value)))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                pairs = []
                if isinstance(t, (ast.Tuple, ast.List)) and isinstance(
                        node.value, (ast.Tuple, ast.List)):
                    pairs = list(zip(t.elts, node.value.elts))
                else:
                    pairs = [(t, node.value)]
                for tgt, val in pairs:
                    nm = getattr(tgt, 'id', getattr(tgt, 'attr', '')) or ''
                    if is_val(val) and tickish(nm):
                        hits.append((node.lineno, '%s = %r' % (nm, value)))
    return hits


def test_the_stray_episode_literal_scan_sees_every_spelling():
    """The scan must be able to fail on each spelling it claims to catch --
    including the two the trackAgent review found invisible to it."""
    src = (
        "parser.add_argument('--episode-ticks', type=int, default=800)\n"
        "ticks, base = 800, 42\n"
        "x = getattr(args, 'episode_ticks', 800)\n"
        "run(episode_ticks=800)\n"
        "EPISODE_LEN_TICKS = 800\n"
        "parser.add_argument('--width', default=800)\n"      # not a tick
        "w, h = 800, 600\n"                                   # not a tick
    )
    hits = _stray_episode_literals(ast.parse(src), 800)
    assert sorted(l for l, _ in hits) == [1, 2, 3, 4, 5], hits


def test_the_episode_length_has_one_source():
    """RERUN_SPEC WP8 discloses an 800-tick training episode.  Four files
    need that number and, until D8.24, three of them carried their own
    literal -- only `train_on_comparison.py` was bound, while
    docs/REWARD_FROZEN.md claimed the binding of the tree.  Every site named
    in `EPISODE_TICKS_SITES` must RESOLVE to the constant (not merely name
    it), and no production file -- the named sites included -- may carry a
    literal bound to an episode length."""
    from sixg_sim.reward_spec import EPISODE_TICKS, EPISODE_TICKS_SITES

    assert len(EPISODE_TICKS_SITES) >= 4, EPISODE_TICKS_SITES
    values = _episode_site_values()
    checked = 0
    for rel, how in EPISODE_TICKS_SITES:
        path = os.path.join(ROOT, *rel.split('/'))
        assert os.path.exists(path), 'EPISODE_TICKS_SITES names %r' % rel
        assert len(how) > 20, rel
        assert rel in values, (
            '%s is named in EPISODE_TICKS_SITES but this test has no way to '
            'resolve the value it uses' % rel)
        assert values[rel] == EPISODE_TICKS, (
            '%s resolves the episode length to %r, not reward_spec.'
            'EPISODE_TICKS = %r' % (rel, values[rel], EPISODE_TICKS))
        checked += 1
    assert checked == len(EPISODE_TICKS_SITES) == len(values) and checked > 0

    # An exemption is NAMED, with its reason, and checked for staleness
    # below -- not a silent hole in the scan.
    for rel, why in _EPISODE_TICKS_EXEMPT.items():
        path = os.path.join(ROOT, *rel.split('/'))
        assert os.path.exists(path), (
            '_EPISODE_TICKS_EXEMPT names %r, which is not in the tree' % rel)
        assert len(why) > 60, rel
        with open(path, encoding='utf-8') as f:
            body = f.read()
        assert str(EPISODE_TICKS) in body, (
            '%s no longer carries the literal it was excused for -- drop its '
            '_EPISODE_TICKS_EXEMPT entry' % rel)

    stray = []
    scanned = 0
    named_scanned = 0
    named = set(r for r, _ in EPISODE_TICKS_SITES)
    for path in _production_py_files():
        rel = _rel(path)
        if rel in _EPISODE_TICKS_EXEMPT or rel == 'sixg_sim/reward_spec.py':
            continue
        with open(path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
        scanned += 1
        named_scanned += rel in named
        for line, what in _stray_episode_literals(tree, EPISODE_TICKS):
            stray.append('%s:%d %s' % (rel, line, what))
    assert scanned > 5, 'the stray-literal scan covered %d files' % scanned
    assert named_scanned == len(named), (
        'the scan skipped a named site: %d of %d scanned'
        % (named_scanned, len(named)))
    assert not stray, (
        'an independent copy of the episode length is back: %r.  Bind it to '
        'sixg_sim.reward_spec.EPISODE_TICKS and add the file to '
        'EPISODE_TICKS_SITES (D8.24)' % stray)

    # the document says exactly this, of exactly these files
    with open(FROZEN_DOC, encoding='utf-8') as f:
        doc = f.read()
    for rel, _how in EPISODE_TICKS_SITES:
        assert ('`%s`' % rel) in doc, (
            'docs/REWARD_FROZEN.md does not name the bound site %r' % rel)
    assert 'the disclosed episode length and the' not in doc, (
        'docs/REWARD_FROZEN.md still carries the WP8 sentence that claimed '
        'of the tree what was true only of train_on_comparison.py (D8.24)')
    # carry-forward defect 2: the document claims only what the list and
    # this test support, and names the exempted site instead of a universal
    # "every place in the tree"
    flat = ' '.join(doc.split())
    assert 'every place in the tree' not in flat, (
        'docs/REWARD_FROZEN.md still makes the universal claim')
    assert 'Every site listed below reads it from there' in flat
    for rel in _EPISODE_TICKS_EXEMPT:
        assert ('`%s`' % rel) in doc, (
            'docs/REWARD_FROZEN.md does not name the exempted site %r' % rel)


def test_truncation_is_terminal_in_the_trainers_behaviour():
    """RERUN_SPEC WP8: "truncation treated as terminal".  Checked on
    BEHAVIOUR.

    Two identical three-step trajectories go through `MAPPOTrainer.collect`:
    one whose last transition the CALLER marks done=True, one left open and
    closed by `finish_episode`.  If truncation is terminal the two flush
    identical returns and advantages -- and both must differ from the
    bootstrapped alternative, which is the control that proves the data can
    tell the two apart.  The check this replaces asserted that the string
    'done' appeared in `finish_episode`'s source, and that string is in its
    docstring, so it held whatever the code did (D8.25)."""
    import torch
    from sixg_sim.agent import (ACTOR_OBS_DIM, CRITIC_OBS_DIM, CriticNetwork,
                                DISCRETE_HEADS, RLAgent)
    from sixg_sim.mappo_trainer import (MAPPOConfig, MAPPOTrainer,
                                        REWARD_SCALE, compute_gae)
    from sixg_sim.reward_spec import TRUNCATION_IS_TERMINAL
    from sixg_sim.traffic import SliceDictionary

    assert TRUNCATION_IS_TERMINAL is True

    REWARDS = [10.0, 20.0, 30.0]
    # a LARGE value on the last state, so bootstrapping past the truncation
    # would be impossible to miss
    VALUES = [1.0, 2.0, 100.0]
    ACTION = dict((h, 0) for h in DISCRETE_HEADS)
    PRB = [0.5, 0.5]

    def _run(explicit_done):
        agents = {'a0': RLAgent('a0', SliceDictionary(), is_training=True)}
        trainer = MAPPOTrainer(agents, CriticNetwork(CRITIC_OBS_DIM),
                               MAPPOConfig())
        n = len(REWARDS)
        for i in range(n):
            trainer.collect(
                'a0', torch.zeros(ACTOR_OBS_DIM), ACTION, list(PRB),
                0.0, VALUES[i], REWARDS[i],
                bool(explicit_done and i == n - 1),
                global_obs=torch.zeros(CRITIC_OBS_DIM), step=i)
        trainer.finish_episode()
        return ([float(x) for x in trainer._pool.returns],
                [float(x) for x in trainer._pool.advantages])

    ret_trunc, adv_trunc = _run(explicit_done=False)
    ret_term, adv_term = _run(explicit_done=True)

    assert len(ret_trunc) == len(REWARDS), ret_trunc
    assert ret_trunc == ret_term, (
        'truncation is NOT treated as terminal: closing the episode with '
        'finish_episode() flushed %r, an explicit done=True on the last '
        'transition flushed %r' % (ret_trunc, ret_term))
    assert adv_trunc == adv_term, (adv_trunc, adv_term)

    # THE CONTROL: what the same data gives if the tail is bootstrapped.  If
    # this equalled the observed value, the assertion above would also hold
    # for a trainer that never marked the last transition done.
    cfg = MAPPOConfig()
    scaled = [r / REWARD_SCALE for r in REWARDS]
    _adv_b, ret_boot = compute_gae(scaled, [False] * len(REWARDS), VALUES,
                                   VALUES[-1], cfg.gamma, cfg.gae_lambda)
    boot_last = float(ret_boot[-1])
    assert abs(boot_last - ret_trunc[-1]) > 1.0, (
        'the probe cannot tell a bootstrapped tail from a terminal one '
        '(bootstrapped %r, observed %r): it would pass on a trainer that '
        'never marked the last transition done' % (boot_last, ret_trunc[-1]))

    # and the observed value IS the terminal one, computed independently
    _adv_t, ret_ok = compute_gae(scaled, [False, False, True], VALUES, 0.0,
                                 cfg.gamma, cfg.gae_lambda)
    assert ret_trunc == [float(x) for x in ret_ok], (ret_trunc, ret_ok)


# ── 6. this file cannot go quiet ─────────────────────────────────────────
def test_no_wp8_test_can_silently_skip():
    """A contract test that returns early on a missing precondition reports a
    pass while checking nothing.  No test function in this file may contain a
    bare `return`, a call to the test framework's skip/xfail markers, or a
    `sys.exit`.

    The detection is an AST scan of CALL NAMES ('skip', 'skipTest', 'xfail',
    'importorskip', 'exit'), not a substring search, so this file
    deliberately does not spell the dotted marker idioms out: the WP6
    tripwire tests/test_wp6_parity.py::test_no_test_file_skips_quietly greps
    every test file in the tree for those literals, and a file that names one
    even in prose trips it (D7.9)."""
    MARKERS = {'skip', 'skipTest', 'xfail', 'importorskip', 'exit'}
    bad = []
    path = os.path.abspath(__file__)
    tree = ast.parse(open(path, encoding='utf-8').read(), filename=path)
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef)
                and node.name.startswith('test_')):
            continue
        # the test BODY only: a nested helper's own `return` is how a helper
        # returns a value, not a way for the test to skip
        nested = set()
        for inner in ast.walk(node):
            if inner is node:
                continue
            if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)):
                for k in ast.walk(inner):
                    nested.add(id(k))
        for sub in ast.walk(node):
            if id(sub) in nested:
                continue
            if isinstance(sub, ast.Return):
                bad.append('%s returns early at line %d'
                           % (node.name, sub.lineno))
            if isinstance(sub, ast.Call):
                f = sub.func
                name = getattr(f, 'attr', getattr(f, 'id', ''))
                if name in MARKERS:
                    bad.append('%s calls %r at line %d'
                               % (node.name, name, sub.lineno))
    assert not bad, bad

    # and the file really does define the contracts it claims to
    names = {n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
    for required in ('test_team_objective_on_a_hand_constructed_state',
                     'test_agent_local_reward_on_a_hand_constructed_state',
                     'test_local_churn_follows_the_executed_mode_not_the_'
                     'sampled_action',
                     'test_team_churn_counts_executed_modes_under_a_forced_'
                     'engine_hold',
                     'test_ppo_scores_the_sampled_action_not_the_executed_'
                     'mode',
                     'test_no_removed_reward_term_is_referenced_anywhere',
                     'test_the_dead_recovery_ramp_is_gone_and_cannot_come_'
                     'back',
                     'test_the_dumped_coefficient_file_matches_the_object_'
                     'field_by_field',
                     'test_the_frozen_kappa_cannot_be_moved_by_a_flag',
                     'test_the_live_driver_mixes_at_the_frozen_kappa',
                     'test_the_team_relay_throughput_term_keeps_the_pre_wp8_'
                     'division',
                     'test_every_published_coefficient_has_a_reader',
                     'test_no_second_copy_of_the_reunification_cadence_can_'
                     'reappear',
                     'test_the_episode_length_has_one_source',
                     'test_truncation_is_terminal_in_the_trainers_behaviour',
                     'test_training_smoke_on_the_frozen_reward'):
        assert required in names, 'the WP8 contract %r is gone' % required


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
