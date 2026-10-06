# -*- coding: utf-8 -*-
"""The online-learning path of the evaluation driver (FIX-PREFREEZE items 3
and 4; docs/DECISIONS.md D-FIX.3, D-FIX.4, D8.11, D6.41).

The adapt-then-freeze variant `marl_freeze` (chain S9, secondary S7) adapts
its policy online during the pass, inside `run_physics_pass`'s
`_hook_after_observe`, before it freezes.  Two defects lived there:

  * D8.11 -- it mixed `0.5 * local + 0.5 * (0.4 reachable_ue_fraction + 0.3
    intra_island_reachability + 0.3 ue_to_ue_routed_norm)`, neither the
    frozen kappa = 0.6 nor the team objective: the variant trained on a
    reward that is not the pre-registered one;
  * D6.41 -- `try: mappo_trainer.collect(...) except Exception: pass`, so a
    refused transition (WP6's width check, any error) was dropped in silence.

Both tests run one real, short Scenario A pass of the arm (seed 82, 200
ticks: severance at 50, the guarded freeze from 100) with the live call
sites recorded, as tests/test_wp8_reward.py does for the training driver.
tools/prefreeze_negative_controls.py re-introduces each defect and requires
the matching test to fail.

The real pass needs the evaluated environment before the first sixg_sim
import; `REQUIRES_ENV` makes tests/run_tests.py re-run this file in a child
process that has it.
"""
import math
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
REQUIRES_ENV = dict(EVAL_ENV)
if 'sixg_sim.agent' not in sys.modules:
    for _k, _v in EVAL_ENV.items():
        os.environ.setdefault(_k, _v)

TICKS = 200
SEED = 82
CKPT = os.path.join(ROOT, 'output', 'tmp_tests', 'prefreeze',
                    'policy_schema_current.pt')

_R = {}


def _driver():
    """The evaluation driver, imported under the evaluated environment, with
    a schema-current minted checkpoint and the short-run overrides."""
    if 'R' in _R:
        return _R['R']
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    import sixg_sim.simulation as _S
    if not (all(os.environ.get(k) == v for k, v in EVAL_ENV.items())
            and _S.STEER_MODEL == EVAL_ENV['MARL_STEER_MODEL']):
        raise AssertionError(
            'CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips '
            '(tests/test_wp6_parity.py): sixg_sim was imported outside the '
            'evaluated environment; run this file through tests/run_tests.py')
    from sixg_sim.agent import mint_schema_current_checkpoint
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    mint_schema_current_checkpoint(CKPT, torch_seed=1234,
                                   extra={'test': 'prefreeze_online_arm'})
    R._apply_runtime_overrides({'total_ticks': TICKS})
    assert R.SEVERANCE_TICK == 50 and R.TOTAL_TICKS == TICKS, (
        R.SEVERANCE_TICK, R.TOTAL_TICKS)
    _R['R'] = R
    return R


def test_the_adapt_then_freeze_arm_trains_on_the_frozen_reward():
    """D8.11, FIX-PREFREEZE item 3.  Every reward the online arm hands to
    `MAPPOTrainer.collect` is `reward_spec.mix(local, team)` = 0.4 * local +
    0.6 * team, with `local` the agent's own frozen local reward and `team`
    the simulator's team objective of THAT tick -- the reward the replicates
    were trained on (train_on_comparison.py, sixg_sim/worker.py).  Checked
    against a mixture computed here from the literal 0.4 / 0.6 and against
    the frozen object, on every collected transition (at least 20)."""
    R = _driver()
    from sixg_sim.agent import RLAgent
    from sixg_sim.mappo_trainer import MAPPOTrainer
    from sixg_sim.simulation import Simulator
    from sixg_sim import reward_spec

    assert reward_spec.KAPPA == 0.6
    team = {'v': None, 'calls': 0}
    local = {}
    seen = []
    real_team = Simulator.compute_global_connectivity_reward
    real_local = RLAgent.calculate_reward
    real_collect = MAPPOTrainer.collect

    def spy_team(self, *a, **k):
        v = real_team(self, *a, **k)
        team['v'] = float(v)
        team['calls'] += 1
        return v

    def spy_local(self, *a, **k):
        comps = real_local(self, *a, **k)
        local[self.node_id] = float(comps.total_reward)
        return comps

    def spy_collect(self, agent_id, obs_tensor, action_indices, prb_frac,
                    log_prob, value, reward, *a, **k):
        seen.append((agent_id, local.get(agent_id), team['v'], float(reward)))
        return real_collect(self, agent_id, obs_tensor, action_indices,
                            prb_frac, log_prob, value, reward, *a, **k)

    Simulator.compute_global_connectivity_reward = spy_team
    RLAgent.calculate_reward = spy_local
    MAPPOTrainer.collect = spy_collect
    try:
        out = R.run_physics_pass('marl_freeze', SEED, CKPT, 'recovery')
    finally:
        Simulator.compute_global_connectivity_reward = real_team
        RLAgent.calculate_reward = real_local
        MAPPOTrainer.collect = real_collect

    assert out and len(out['conn']) == TICKS, 'the pass did not complete'
    # NON-VACUITY: a pass that collected nothing checked nothing.
    assert len(seen) >= 20, 'only %d transitions were collected' % len(seen)
    assert team['calls'] > 0, 'the team objective was never computed'
    bad = []
    for aid, loc, tm, r in seen:
        if loc is None or tm is None:
            bad.append((aid, 'no local or team reward before the collect'))
            continue
        want = 0.4 * loc + 0.6 * tm
        if not math.isclose(r, want, rel_tol=1e-12, abs_tol=1e-12):
            bad.append((aid, loc, tm, r, want))
        elif r != reward_spec.mix(loc, tm):
            bad.append((aid, 'not the frozen object', loc, tm, r))
    assert not bad, '%d of %d collected rewards are not the frozen mixture: %r' % (
        len(bad), len(seen), bad[:5])
    # the mixture is told apart from the old 0.5/0.5 shape: the team and the
    # local terms differ on the samples that were checked
    assert any(abs(loc - tm) > 1e-9 for _a, loc, tm, _r in seen), (
        'local and team rewards were equal on every sample: the check could '
        'not tell a 0.4/0.6 mixture from another')

def test_a_refused_transition_in_the_online_arm_is_loud():
    """D6.41, FIX-PREFREEZE item 4.  A `collect` that raises must end the
    pass with that error.  The call sat inside `try: ... except Exception:
    pass`, so WP6's own width check -- or any other failure -- dropped the
    transition in silence and the pass completed as if nothing had
    happened."""
    R = _driver()
    from sixg_sim.mappo_trainer import MAPPOTrainer

    calls = {'n': 0}
    real_collect = MAPPOTrainer.collect

    class ProbeRefusal(RuntimeError):
        pass

    def refusing_collect(self, *a, **k):
        calls['n'] += 1
        raise ProbeRefusal('FIX-PREFREEZE probe: collect refused this transition')

    MAPPOTrainer.collect = refusing_collect
    raised = None
    try:
        R.run_physics_pass('marl_freeze', SEED, CKPT, 'recovery')
    except ProbeRefusal as exc:
        raised = exc
    finally:
        MAPPOTrainer.collect = real_collect
    assert calls['n'] >= 1, 'collect was never reached: the test checked nothing'
    assert raised is not None, (
        'collect raised %d time(s) and the pass completed: the failure was '
        'swallowed' % calls['n'])
    assert calls['n'] == 1, 'the pass went on after the first refusal (%d)' % calls['n']
    assert 'collect refused' in str(raised)
