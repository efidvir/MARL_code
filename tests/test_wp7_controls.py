# -*- coding: utf-8 -*-
"""WP7 -- the two controls and the training smoke, against the REAL engine.

RERUN_SPEC WP7:
  * uniform-random control: uniform over the NEW action space (each discrete
    head uniform, Dirichlet(1,1,1)), REDRAWN EVERY TICK.  Pre-registered.
  * untrained control: regenerated from the new network definition through
    the training initialisation path, carrying the new schema id and loading
    under the strict gate.
  * a short training smoke on the new action space.

Builds the comparison topology, so it is slower than
tests/test_wp7_action_space.py and belongs on <compute-server>:

    env MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile \\
        MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 \\
        MARL_REWARD_PROFILE=mission \\
        MARL_RELAY_ENTROPY_FLOOR=0.01 MARL_STEER_MODEL=actionable \\
        PYTHONHASHSEED=0 MARL_POOL_WORKERS=2 \\
        nice -n 19 python3 tests/run_tests.py wp7_controls

NOTHING IN THIS FILE SKIPS; see the tripwire in test_wp7_action_space.py.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tmp_base():
    """Parent of every scratch directory of this file: output/tmp_tests
    inside the tree (the seed audit skips it by design), never the system
    default, which falls back to the current directory -- the tree root,
    where the suite runs -- when `/` is full (D12F.2)."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return base


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
TICKS = 60

_cache = {}


def _random_arm_actions(ticks=TICKS):
    """Run the uniform-random control arm through the REAL batched sampler
    and return the per-tick actions it produced."""
    if 'rand' in _cache:
        return _cache['rand']
    import torch
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        from run_timeline_comparison import (build_comparison_topology,
                                             build_comparison_scenario)
    finally:
        sys.argv = argv
    from sixg_sim.simulation import SimulationConfig, Simulator

    topology, _info = build_comparison_topology(SEED, log_prefix='[WP7] ')
    scenario = build_comparison_scenario(
        'rescue_ops', SEED, topology=topology, ticks=ticks,
        severance_tick=0, rescue_tick=ticks - 1, scenario_seed=SEED)
    sim = Simulator(topology, scenario,
                    SimulationConfig(enable_island_detection=True,
                                     verbose=False, random_seed=SEED))
    sim._current_scenario_type = 'rescue_ops'
    sim.random_action_policy = True
    sim.marl_ue_routing_enabled = True
    torch.manual_seed(SEED)

    per_tick = []
    acting_ticks = []

    def _act(tick, observations):
        sim._execute_agent_actions(observations)
        acting_ticks.append(tick)
        per_tick.append({aid: sim.agents[aid].last_action
                         for aid in observations
                         if getattr(sim.agents[aid], 'last_action', None)
                         is not None})

    for tick in range(ticks):
        sim.step_tick(tick, _act, forwarding=None,
                      where='wp7_controls', log_prefix='[WP7] ')

    _cache['rand'] = per_tick
    _cache['acting_ticks'] = acting_ticks
    return per_tick


def test_the_uniform_random_control_draws_from_the_new_action_space():
    """Every discrete head uniform over ITS OWN new cardinality, the split
    uniform on the 2-simplex, and a fresh draw every tick."""
    import math
    from sixg_sim.agent import (DISCRETE_HEADS, HEAD_SIZES, N_RELAY_MODES,
                                relay_enum_index)

    per_tick = _random_arm_actions()
    # D13.2: a site acts once ITS OWN local-isolation flag is set, which the
    # tick-0 step sets in Scenario B -- so the action step runs on every tick
    # from 1, not from 0 (the global trigger used to open the lock at 0)
    assert _cache['acting_ticks'] == list(range(1, TICKS)), (
        _cache['acting_ticks'][:5], len(_cache['acting_ticks']))
    assert len(per_tick) == TICKS - 1, len(per_tick)
    flat = [a for tick in per_tick for a in tick.values()]
    assert len(flat) > 200, ('only %d actions: the arm produced too few '
                             'samples to say anything' % len(flat))

    # ── the support of each head is the NEW one ──────────────────────────
    tx = [int(a.tx_power_step) for a in flat]
    me = [int(a.mcs_emergency_idx) for a in flat]
    mg = [int(a.mcs_general_idx) for a in flat]
    rh = [int(a.relay_head_idx) for a in flat]
    assert set(tx) == set(range(5)), sorted(set(tx))
    assert set(me) == set(range(5)), sorted(set(me))
    assert set(mg) == set(range(5)), sorted(set(mg))
    assert set(rh) == set(range(N_RELAY_MODES)) == {0, 1, 2}, sorted(set(rh))
    # the executed mode is the head's mode; the two dropped aliases
    # (NORMAL, D2D_PEER_RELAY) can no longer be drawn at all
    assert set(int(a.relay_mode_idx) for a in flat) == {0, 2, 3}, (
        'the random arm drew a RelayMode outside the WP7 head: %r'
        % sorted(set(int(a.relay_mode_idx) for a in flat)))
    for a in flat:
        assert a.relay_mode_idx == relay_enum_index(a.relay_head_idx)

    # ── and it is UNIFORM, not merely supported ──────────────────────────
    # A two-sided binomial check per outcome at ~5 sigma: with n >= 200 this
    # catches a head that is 1/5 uniform where it should be 1/3, without
    # being flaky on a fair draw.
    def uniformity(sample, k, label):
        n = len(sample)
        p = 1.0 / k
        sd = math.sqrt(n * p * (1 - p))
        for v in range(k):
            c = sample.count(v)
            assert abs(c - n * p) <= 5.0 * sd, (
                '%s is not uniform over %d outcomes: %d of %d draws took '
                'value %d (expected %.1f +- %.1f)'
                % (label, k, c, n, v, n * p, sd))

    for name, sample, k in (('tx_power', tx, 5), ('mcs_emrg', me, 5),
                            ('mcs_gen', mg, 5), ('relay', rh, 3)):
        uniformity(sample, k, name)
    assert list(zip(DISCRETE_HEADS, HEAD_SIZES)) == [
        ('tx_power', 5), ('mcs_emrg', 5), ('mcs_gen', 5), ('relay', 3)]

    # ── the split is Dirichlet(1,1,1): uniform on the simplex ────────────
    prb = [a.prb_sampled for a in flat]
    assert all(p is not None and len(p) == 3 for p in prb)
    for p in prb:
        assert abs(sum(p) - 1.0) < 1e-5, p
        assert all(x >= 0.0 for x in p), p
    # Dirichlet(1,1,1) has mean 1/3 on every coordinate; softmax-of-zeros,
    # the wrong answer this arm is written against, would put EVERY draw
    # exactly at (1/3, 1/3, 1/3).
    for j in range(3):
        col = [p[j] for p in prb]
        mean = sum(col) / len(col)
        assert abs(mean - 1.0 / 3.0) < 0.06, (j, mean)
        assert max(col) - min(col) > 0.5, (
            'coordinate %d of the split barely moves (%.3f..%.3f): this is '
            'not a draw from Dirichlet(1,1,1)' % (j, min(col), max(col)))

    # ── redrawn EVERY TICK, per agent ────────────────────────────────────
    sites = sorted(set(k for t in per_tick for k in t))
    moved = 0
    for site in sites:
        series = [t[site] for t in per_tick if site in t]
        if len(series) < 10:
            continue
        sig = [(a.tx_power_step, a.mcs_emergency_idx, a.mcs_general_idx,
                a.relay_head_idx, tuple(round(x, 9) for x in a.prb_sampled))
               for a in series]
        changes = sum(1 for i in range(1, len(sig)) if sig[i] != sig[i - 1])
        assert changes >= 0.5 * (len(sig) - 1), (
            'site %s repeated its action on %d of %d consecutive ticks: the '
            'control is not redrawn every tick'
            % (site, len(sig) - 1 - changes, len(sig) - 1))
        moved += 1
    assert moved >= 3, 'only %d sites had a long enough series' % moved


def test_the_random_control_definition_is_written_down():
    """Pre-registered means WRITTEN DOWN, in docs as well as in code."""
    docs = os.path.join(ROOT, 'docs')
    dec = open(os.path.join(docs, 'DECISIONS.md'), encoding='utf-8').read()
    assert 'D7.4' in dec, 'the random control has no decision record'
    for phrase in ('Dirichlet(1,1,1)', 'redrawn every tick'):
        assert phrase in dec, (
            'the pre-registered random-control definition does not state %r'
            % phrase)
    spec = open(os.path.join(docs, 'RERUN_SPEC.md'), encoding='utf-8').read()
    assert 'Uniform-random control' in spec

    # and the same definition is in the code that implements it
    src = open(os.path.join(ROOT, 'sixg_sim', 'simulation.py'),
               encoding='utf-8').read()
    assert 'PRE-REGISTERED DEFINITION' in src
    assert 'Dirichlet(1,1,1)' in src


def test_the_untrained_control_regenerates_and_loads_under_the_strict_gate():
    """The untrained control is the NEW network, through the TRAINING
    initialisation path, stamped with the new schema id."""
    import torch
    import make_untrained_checkpoints as M
    from sixg_sim.agent import (PolicyNetwork, SCHEMA_ID, DISCRETE_HEADS,
                                assert_checkpoint_schema,
                                load_policy_state_dict)
    from sixg_sim import runinfo

    sd, canon, n_agents = M.initial_actor(1234)
    assert n_agents > 1, n_agents

    # it IS the new network definition: same keys, same shapes, no dead head
    ref = PolicyNetwork().state_dict()
    assert list(sd) == list(ref), (list(sd), list(ref))
    assert all(tuple(sd[k].shape) == tuple(ref[k].shape) for k in sd)
    for dead in ('handover', 'scheduler', 'postcard', 'iops'):  # wp7-scan-ok: the NEGATIVE assertion has to name them
        assert not any(k.startswith(dead) for k in sd), (dead, list(sd))
    assert sd['relay_head.weight'].shape[0] == 3, sd['relay_head.weight'].shape

    # deterministic in the torch seed, as the generator's docstring claims
    sd2, _c2, _n2 = M.initial_actor(1234)
    assert M.fingerprint(sd) == M.fingerprint(sd2), 'initialisation drifted'
    sd3, _c3, _n3 = M.initial_actor(2345)
    assert M.fingerprint(sd) != M.fingerprint(sd3), 'the torch seed is ignored'

    # it carries THE schema id and passes the strict gate, on disk
    import tempfile
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as td:
        path = os.path.join(td, 'policy_init.pt')
        payload = {'state_dict': sd, 'version': 0,
                   'metadata': {'episode': 0, 'untrained': True,
                                'torch_seed': 1234, 'canonical_seed': canon,
                                'provenance': runinfo.provenance(extra={
                                    'driver': 'make_untrained_checkpoints',
                                    'torch_seed': 1234})}}
        torch.save(payload, path)
        back = torch.load(path, map_location='cpu', weights_only=False)
        got = assert_checkpoint_schema(back, where='wp7 untrained control')
        assert got == SCHEMA_ID, (got, SCHEMA_ID)
        assert (back['metadata']['provenance']['schema_id']
                == SCHEMA_ID), back['metadata']['provenance']
        net = PolicyNetwork()
        load_policy_state_dict(net, back['state_dict'])
        import torch as _t
        out = net(_t.zeros(2, net.fc1.in_features))
        assert set(out) == set(DISCRETE_HEADS) | {'prb'}, sorted(out)


def test_a_pre_wp7_checkpoint_is_refused_rather_than_adapted():
    """The other half of the contract: the gate must REFUSE the old action
    space, and refuse it by schema id, not by a shape error out of
    load_state_dict."""
    import torch
    from sixg_sim.agent import (PolicyNetwork, assert_checkpoint_schema,
                                load_policy_state_dict)

    net = PolicyNetwork()
    stale_sd = dict(net.state_dict())
    stale_sd['relay_head.weight'] = torch.zeros(5, net.relay_head.in_features)
    stale_sd['relay_head.bias'] = torch.zeros(5)
    stale_sd['postcard_head.weight'] = torch.zeros(2, net.relay_head.in_features)
    stale_sd['postcard_head.bias'] = torch.zeros(2)
    stale = {'state_dict': stale_sd, 'version': 0,
             'metadata': {'provenance': {'schema_id': 'obs1-6c50fa877249bff6'}}}

    gate_raised = None
    try:
        assert_checkpoint_schema(stale, where='wp7 stale')
    except ValueError as e:
        gate_raised = e
    assert gate_raised is not None, 'the gate accepted the pre-WP7 schema id'
    assert 'schema' in str(gate_raised)

    # and even past the gate the loader refuses it rather than adapting
    load_raised = None
    try:
        load_policy_state_dict(PolicyNetwork(), stale_sd)
    except (RuntimeError, ValueError) as e:
        load_raised = e
    assert load_raised is not None, (
        'a 5-wide relay head with a postcard head loaded into the WP7 network')


def test_training_smoke_on_the_new_action_space():
    """Two tiny training episodes end to end: collection, GAE and the PPO
    update all run, and every stored action dict is exactly DISCRETE_HEADS
    with the relay entry inside the 3-mode head."""
    import train_on_comparison as T
    from sixg_sim.agent import (CriticNetwork, PolicyNetwork, DISCRETE_HEADS,
                                CRITIC_OBS_DIM, OBS_DIM, N_RELAY_MODES)
    from sixg_sim.reward_spec import KAPPA as _KAPPA

    actor = PolicyNetwork()
    critic = CriticNetwork()
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
    assert pool['global_obs'][0].shape == (CRITIC_OBS_DIM,)
    assert len(pool['advantages']) == len(pool['obs'])
    assert all(r == r for r in pool['returns']), 'NaN return'

    acts = pool['actions']
    assert acts, 'no actions were stored'
    for a in acts:
        assert set(a) == set(DISCRETE_HEADS), sorted(a)
        assert 0 <= a['relay'] < N_RELAY_MODES, a['relay']
        assert 0 <= a['tx_power'] < 5 and 0 <= a['mcs_emrg'] < 5
        assert 0 <= a['mcs_gen'] < 5

    # the stored behaviour-policy log-probs are real numbers, not the zeroed
    # fallback the WP6 leftovers used to produce on any failure
    lps = list(pool['log_probs'])
    assert lps and not all(v == 0.0 for v in lps), (
        'every stored log-prob is 0.0: the collection path fell back')
    assert all(v == v and v < 0.0 for v in lps), (
        'a stored log-prob is NaN or non-negative: %r'
        % [v for v in lps if not (v == v and v < 0.0)][:5])

    # and a real PPO update runs on them
    import torch
    from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
    from sixg_sim.agent import RLAgent
    from sixg_sim.traffic import SliceDictionary
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
            acts[i], list(pool['prb_acts'][i]),
            float(pool['log_probs'][i]), 0.0, 0.0, i == n - 1,
            global_obs=torch.as_tensor(pool['global_obs'][i],
                                       dtype=torch.float32))
    trainer.finish_episode()
    metrics = trainer.update()
    assert metrics['updates'] > 0, metrics
    assert metrics['entropy'] == metrics['entropy'], 'NaN entropy'
    import math
    ceiling = 3 * math.log(5) + math.log(3)
    assert 0.0 < metrics['entropy'] <= ceiling + 1e-6, (
        'the entropy term is not a sum over the FOUR heads: %r'
        % metrics['entropy'])
    assert math.isfinite(metrics['policy_loss'])
    assert math.isfinite(metrics['value_loss'])


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
