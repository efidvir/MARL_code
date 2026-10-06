"""WP-DET -- determinism of the training path.

The pre-registration promises that a run can be reproduced.  Before this work
package it could not: a worker episode with identical topology, identical
actor weights, identical seeds and even identical sampled actions produced
different numbers, and four identical-configuration training runs produced
four different policies.

WHAT THE DEFECT ACTUALLY WAS (measured on <compute-server> at head a2b0c03, not
inferred).  `run_comparison_worker_episode` built its simulator with

    SimulationConfig(enable_island_detection=True, verbose=False)

i.e. `random_seed=None`.  `Simulator.__init__` then constructed
`TrafficGenerator(seed=None)` -> `numpy.random.RandomState(None)`, which
draws its state from OS ENTROPY.  So every training episode -- including two
episodes with the same topology seed, the same event seed and the same torch
seed -- saw a DIFFERENT offered load from tick 0 on.  The evaluation harness
had already been fixed (run_timeline_comparison passes
`random_seed=int(topology_seed)`, with a comment explaining exactly this);
the training driver never was.

It was NOT a set/dict iterated in `id()` order: every set on the
observation, reward and action paths holds node-id or link-id STRINGS (or
small ints), and an AST scan of the tree for iterated set-valued names found
no container of Node or Link OBJECTS.  The audit is recorded in
docs/DECISIONS.md (D-DET.2).

WHAT THESE TESTS PIN DOWN
  1. two episodes with identical seeds give identical observation, action and
     reward series -- at the engine level (per-tick, per-agent, including the
     delivered-volume ledger, which is where the defect first showed) and
     through the real training worker (the collected PPO transitions);
  2. the training environment is seeded at all, and seeded with the same rule
     the evaluation harness uses;
  3. the batch is folded into the rollout pool in episode order, not in
     worker-completion order;
  4. the drivers pin themselves to one thread instead of trusting the
     caller's environment.

Runtime: about two minutes on <compute-server> (two short episodes twice).

    env MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile \\
        MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 \\
        MARL_REWARD_PROFILE=mission \\
        MARL_RELAY_ENTROPY_FLOOR=0.01 MARL_STEER_MODEL=actionable \\
        PYTHONHASHSEED=0 nice -n 19 python3 tests/run_tests.py wpdet
"""
import hashlib
import os
import re
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
# Read by tests/run_tests.py: the reward profile (MARL_REWARD_PROFILE) is
# fixed at the FIRST sixg_sim import, so a whole-suite run re-runs this file
# in a child process that has the evaluated environment.
REQUIRES_ENV = dict(EVAL_ENV)
if 'sixg_sim.agent' not in sys.modules:
    for _k, _v in EVAL_ENV.items():
        os.environ.setdefault(_k, _v)

TOPO_SEED = 131
EVENT_SEED = 700001
TORCH_SEED = 1235
# Short on purpose.  The engine trace forces the severance at tick
# SEVERANCE_TICK so the action locks open early and most of the episode is
# post-event; the worker trace has to live with the driver's own randomised
# event timing (severance >= 30), hence its longer episode.  Since D13.2 a
# site acts only once its OWN local-isolation flag is set -- BFD, one
# Router-LSA, flooding and SPF after the severance, about 14 ticks (D13.1)
# -- so both episodes are longer than they were under the global lock
# (24 and 40 ticks), to keep a post-unlock stretch of about 20 ticks.
TRACE_TICKS = 40
SEVERANCE_TICK = 4
WORKER_TICKS = 64

_cache = {}


def _skip_without_env():
    if os.environ.get('MARL_REWARD_PROFILE') != 'mission':
        raise AssertionError(
            'CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'needs the evaluated environment (MARL_REWARD_PROFILE=mission and '
            'friends) before the first sixg_sim import; tests/run_tests.py '
            'supplies it through REQUIRES_ENV')


# ══════════════════════════════════════════════════════════════════════
# 1. Engine-level episode trace
# ══════════════════════════════════════════════════════════════════════

def _episode_trace(tag):
    """One short episode through the evaluated training tick order, recorded
    per tick.

    Returns a list of (tick, dict) rows.  The row carries the OBSERVATION
    (every agent's unmasked vector -- unmasked so a difference cannot hide in
    a column the actor zeroes), the ACTION (the WP2 action signature plus the
    per-agent sampled heads) and the REWARD (per-agent), and, because the
    defect showed up in the physics a tick before it reached any observation,
    the per-link delivered-UE ledger and the flow counters as well.
    """
    import random as _random
    import torch

    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario,
                                         marl_action_step)
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.link_state_service import make_link_state_service
    from sixg_sim.tick_step import action_signature
    from sixg_sim.agent import RLAgent

    torch.set_num_threads(1)
    torch.manual_seed(TORCH_SEED)
    _random.seed(TORCH_SEED)

    topology, _info = build_comparison_topology(TOPO_SEED)
    scenario = build_comparison_scenario(
        'recovery', TOPO_SEED, topology=topology, ticks=TRACE_TICKS,
        severance_tick=SEVERANCE_TICK, rescue_tick=None,
        scenario_seed=EVENT_SEED)
    config = SimulationConfig(enable_island_detection=True, verbose=False,
                              random_seed=int(TOPO_SEED))
    sim = Simulator(topology, scenario, config)
    torch.manual_seed(TORCH_SEED)          # as the worker does
    sim._current_scenario_type = getattr(scenario, 'scenario_type', 'recovery')
    forwarding = make_link_state_service(sim)
    if forwarding is not None:
        sim._marl_detect = forwarding
    sim.marl_ue_routing_enabled = True

    rows = []
    pending = {}

    def _after_observe(tick, observations):
        row = {}
        for aid in sorted(observations):
            agent = sim.agents.get(aid)
            if not isinstance(agent, RLAgent):
                continue
            # the CRITIC vector (the 26 actor fields followed by the 11
            # global ones, WP6 as WP13 left it): every column the engine builds, so a
            # difference cannot hide in a column the actor does not read
            row['obs:' + aid] = [
                float(x) for x in agent.critic_features(
                    observations[aid]).tolist()]
        # reward for the action taken on the PREVIOUS tick, exactly as the
        # worker's _complete_pending computes it
        try:
            global_r = float(sim.compute_global_connectivity_reward())
        except Exception:
            global_r = 0.0
        row['reward:global'] = [global_r]
        for aid, tr in sorted(pending.items()):
            agent = sim.agents.get(aid)
            if agent is None or aid not in observations:
                continue
            comps = agent.calculate_reward(tr[0], tr[1], observations[aid])
            row['reward:' + aid] = [float(comps.total_reward)]
        pending.clear()
        # physics: where the defect first appeared
        row['carried'] = sorted(
            '%s|%.17g' % (lid, v)
            for lid, v in getattr(sim, '_link_carried_ue', {}).items())
        row['flows'] = [int(sim.ue_to_ue_success_count),
                        int(sim.ue_to_ue_failure_count)]
        rows.append((tick, row))

    def _act(tick, observations):
        marl_action_step(sim, tick, observations, log_prefix='[WPDET]   ')
        for aid in observations:
            agent = sim.agents.get(aid)
            act = getattr(agent, 'last_action', None)
            if act is not None and hasattr(agent, 'calculate_reward'):
                pending[aid] = (observations[aid], act)
        rows[-1][1]['action_sig'] = [int(action_signature(sim))]
        rows[-1][1]['actions'] = sorted(
            '%s|%d|%d|%d|%d|%.17g|%.17g|%.17g'
            % (aid, act.tx_power_step, act.mcs_emergency_idx,
               act.mcs_general_idx, act.relay_mode_idx,
               act.prb_emergency_frac, act.prb_relay_frac,
               act.prb_general_frac)
            for aid, act in ((a, getattr(sim.agents[a], 'last_action', None))
                             for a in observations)
            if act is not None)

    for tick in range(scenario.duration_ticks):
        sim.step_tick(tick, _act, forwarding=forwarding,
                      degrade_tick=SEVERANCE_TICK,
                      hooks={'after_observe': _after_observe},
                      where='test/wpdet/' + tag)
    return rows


def _first_difference(a, b):
    """(tick, key, index, va, vb) of the first difference, or None."""
    if len(a) != len(b):
        return (-1, 'row count', 0, len(a), len(b))
    for (ta, ra), (tb, rb) in zip(a, b):
        if ta != tb:
            return (ta, 'tick', 0, ta, tb)
        if set(ra) != set(rb):
            return (ta, 'key set', 0, sorted(set(ra) - set(rb)),
                    sorted(set(rb) - set(ra)))
        for key in sorted(ra):
            va, vb = ra[key], rb[key]
            if va == vb:
                continue
            if len(va) == len(vb):
                for i, (x, y) in enumerate(zip(va, vb)):
                    if x != y:
                        return (ta, key, i, x, y)
            return (ta, key, -1, va, vb)
    return None


def _traces():
    if 'traces' not in _cache:
        _cache['traces'] = (_episode_trace('a'), _episode_trace('b'))
    return _cache['traces']


def test_episode_series_are_reproducible():
    """Two episodes with identical seeds give identical observation, action
    and reward series (RERUN_SPEC WP-DET.6).

    NOTE what this does and does not pin down.  It builds its own seeded
    SimulationConfig, so it tests the ENGINE's determinism -- it passes on
    the pre-fix tree as well and is NOT the detector for the defect this
    work package fixed.  `test_worker_episode_is_reproducible` below drives
    the real training worker and is that detector.
    """
    _skip_without_env()
    a, b = _traces()
    assert len(a) == TRACE_TICKS, 'trace is %d ticks, expected %d' % (
        len(a), TRACE_TICKS)
    diff = _first_difference(a, b)
    assert diff is None, (
        'two identically-seeded episodes diverged: tick %s field %s[%s]: '
        '%r != %r' % diff)


def test_trace_is_not_vacuous():
    """The trace must actually contain acting ticks, non-trivial
    observations and delivered traffic -- otherwise it could not detect a
    divergence."""
    _skip_without_env()
    a, _b = _traces()
    from sixg_sim.link_state_service import BFD_DETECT_TICKS
    acted = [t for t, r in a if r.get('action_sig', [0])[0] != 0]
    ran = [t for t, r in a if 'action_sig' in r]
    # Until the first site's OWN flag is set (D13.2) the per-site locks keep
    # `_act` from running at all, so those rows carry no action key; from
    # that tick on some site acts on every tick (the lock is a latch).
    assert ran, 'the action locks never opened after the event'
    first = ran[0]
    assert first > SEVERANCE_TICK + BFD_DETECT_TICKS, (
        'agents acted at tick %d, before any site could have seen its own '
        'isolation (severance %d + BFD %d)'
        % (first, SEVERANCE_TICK, BFD_DETECT_TICKS))
    assert ran == list(range(first, TRACE_TICKS)), (
        'the action step skipped a tick after the first unlock: %r' % ran)
    assert len(acted) >= TRACE_TICKS - first - 2 and len(acted) >= 15, (
        'agents acted on only %d of %d ticks (first acting tick %d)'
        % (len(acted), TRACE_TICKS, first))
    obs_rows = [k for k in a[-1][1] if k.startswith('obs:')]
    assert len(obs_rows) >= 20, 'only %d agents observed' % len(obs_rows)
    assert any(len(r['carried']) > 0 for _t, r in a), \
        'no traffic was ever delivered: the ledger cannot detect anything'
    # the observation series must vary over time, or "identical" is trivial
    first = a[0][1][obs_rows[0]]
    assert any(a[t][1][obs_rows[0]] != first for t in range(1, len(a))), \
        'the observation vector never changes during the episode'


# ══════════════════════════════════════════════════════════════════════
# 2. The real training worker
# ══════════════════════════════════════════════════════════════════════

def _worker_digest():
    """Run ONE episode through the real training worker and digest every
    collected transition (observation, action, log-prob, advantage,
    return)."""
    import numpy as np
    import torch

    from train_on_comparison import run_comparison_worker_episode
    from run_timeline_comparison import build_comparison_topology, \
        build_comparison_scenario
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.agent import CriticNetwork, RLAgent

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    topology, _ = build_comparison_topology(42)
    scenario = build_comparison_scenario('recovery', 42, topology=topology,
                                         ticks=WORKER_TICKS,
                                         severance_tick=5)
    sim = Simulator(topology, scenario,
                    SimulationConfig(enable_island_detection=True,
                                     verbose=False, random_seed=42))
    agent = next(a for a in sim.agents.values() if isinstance(a, RLAgent))
    actor_sd = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
    critic_sd = {k: v.cpu()
                 for k, v in CriticNetwork().state_dict().items()}
    del sim, topology, scenario

    res = run_comparison_worker_episode(dict(
        episode_idx=1, topology_seed=TOPO_SEED, scenario_kind='recovery',
        episode_ticks=WORKER_TICKS, event_seed=EVENT_SEED,
        actor_state_dict=actor_sd, critic_state_dict=critic_sd,
        global_reward_mix=0.6, torch_seed=TORCH_SEED))
    pool = res['pool_data']
    h = hashlib.sha256()
    for arr in pool['obs']:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    for arr in pool['global_obs']:
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    h.update(repr(pool['actions']).encode('utf-8'))
    h.update(repr(pool['prb_acts']).encode('utf-8'))
    for key in ('log_probs', 'advantages', 'returns'):
        h.update(repr([round(float(x), 12) for x in pool[key]]).encode('utf-8'))
    return {'sha': h.hexdigest(),
            'n': len(pool['obs']),
            'reward': repr(res['avg_ep_reward'])}


def test_worker_episode_is_reproducible():
    """The real training worker, called twice with identical arguments,
    collects identical transitions.  This is the function whose output feeds
    the MAPPO update, so this is the property `policy_final.pt` rests on."""
    _skip_without_env()
    a = _worker_digest()
    b = _worker_digest()
    assert a['n'] > 0, 'the worker collected no transitions: the episode ' \
                       'never opened its action lock, so nothing is tested'
    assert a == b, ('two identical worker episodes differ:\n  %r\n  %r'
                    % (a, b))


def test_training_environment_is_seeded_like_the_harness():
    """The training simulator's traffic model must be seeded, with the same
    rule the evaluation harness uses (the topology seed).  Two simulators
    built the worker's way must start from the same RNG state."""
    _skip_without_env()
    import inspect
    import train_on_comparison as toc
    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario)
    from sixg_sim.simulation import SimulationConfig, Simulator

    src = inspect.getsource(toc.run_comparison_worker_episode)
    assert 'random_seed=int(topo_seed)' in src, (
        'the training worker no longer seeds its SimulationConfig from the '
        'topology seed; an unseeded TrafficGenerator is entropy-seeded and '
        'the run stops being reproducible')

    def _state():
        topology, _ = build_comparison_topology(TOPO_SEED)
        scenario = build_comparison_scenario(
            'recovery', TOPO_SEED, topology=topology, ticks=8,
            severance_tick=2, scenario_seed=EVENT_SEED)
        sim = Simulator(topology, scenario,
                        SimulationConfig(enable_island_detection=True,
                                         verbose=False,
                                         random_seed=int(TOPO_SEED)))
        st = sim.traffic_generator.rng.get_state()
        return hashlib.sha256(st[1].tobytes()).hexdigest(), int(st[2])

    assert _state() == _state(), \
        'the traffic generator does not start from a reproducible state'

    # ... and an UNSEEDED one does not -- i.e. this test can fail.
    def _unseeded():
        import numpy as np
        st = np.random.RandomState(None).get_state()
        return hashlib.sha256(st[1].tobytes()).hexdigest()

    assert _unseeded() != _unseeded(), \
        'numpy.random.RandomState(None) is reproducible on this build, so ' \
        'the guard above proves nothing'


# ══════════════════════════════════════════════════════════════════════
# 3. Stable batch order (WP-DET.2)
# ══════════════════════════════════════════════════════════════════════

def test_order_batch_results_sorts_by_episode_index():
    import train_on_comparison as toc
    out_of_order = [{'episode_idx': 7, 'tag': 'c'},
                    {'episode_idx': 3, 'tag': 'a'},
                    {'episode_idx': 5, 'tag': 'b'}]
    assert [r['tag'] for r in toc.order_batch_results(out_of_order)] == \
        ['a', 'b', 'c']
    # stable for an already-ordered batch, and a no-op on an empty one
    assert toc.order_batch_results([]) == []
    ordered = [{'episode_idx': i} for i in range(1, 17)]
    assert toc.order_batch_results(ordered) == ordered
    # string indices sort numerically, not lexically (9 before 10)
    mixed = [{'episode_idx': '10'}, {'episode_idx': '9'}]
    assert [int(r['episode_idx']) for r in toc.order_batch_results(mixed)] \
        == [9, 10]


def test_rollout_pool_is_extended_in_episode_order():
    """Source guard: nothing may be folded into the rollout pool inside the
    `as_completed` loop -- that loop only collects."""
    src = open(os.path.join(ROOT, 'train_on_comparison.py'),
               encoding='utf-8').read()
    m = re.search(r'for future in concurrent\.futures\.as_completed\('
                  r'futures\):(.*?)\n            for res in ', src, re.S)
    assert m, 'the batch collection loop has moved; update this guard'
    body = m.group(1)
    assert '_pool.' not in body and 'kpi_tracker' not in body, (
        'the rollout pool is still extended in worker-completion order:\n%s'
        % body)
    # the collection loop may print (liveness), but must touch nothing else
    assert 'train_summary' not in body and 'avg_ep_reward' not in body, (
        'the collection loop still does bookkeeping in completion order:'
        + body)
    assert 'order_batch_results(_batch_results)' in src, (
        'the batch is no longer folded in through order_batch_results()')
    # and the extension itself sits inside the ordered loop
    after = src.split('for res in order_batch_results(_batch_results):', 1)[1]
    assert 'mappo_trainer._pool.obs.extend' in after.split('\n        # ')[0], \
        'the pool extension is not inside the ordered loop'


# ══════════════════════════════════════════════════════════════════════
# 4. Thread pinning (WP-DET.5)
# ══════════════════════════════════════════════════════════════════════

def test_driver_pins_its_own_threads():
    """The drivers pin themselves to one thread rather than relying on the
    caller's environment: a multi-threaded torch/BLAS reduces in a
    thread-count-dependent order and changes the updated weights."""
    import train_on_comparison  # noqa: F401  (import sets the variables)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(var) == '1', \
            '%s is %r after importing the driver' % (var, os.environ.get(var))
    src = open(os.path.join(ROOT, 'train_on_comparison.py'),
               encoding='utf-8').read()
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert "os.environ.setdefault('%s'" % var not in src, (
            '%s is still set with setdefault: the caller can override the '
            'pin and the run stops being reproducible' % var)
    # the parent update and the worker both pin torch
    assert src.count('torch.set_num_threads(1)') >= 2, (
        'torch.set_num_threads(1) must be called in main() (the central '
        'MAPPO update) and in the worker')
    head = src.split('import sys', 1)[0]
    assert "os.environ['OMP_NUM_THREADS'] = '1'" in head, (
        'the thread pin must run before torch/numpy are imported')


def test_worker_pins_threads_before_importing_torch():
    import inspect
    import train_on_comparison as toc
    src = inspect.getsource(toc.run_comparison_worker_episode)
    i_env = src.index("os.environ['OMP_NUM_THREADS'] = '1'")
    i_torch = src.index('import torch')
    assert i_env < i_torch, 'the worker imports torch before pinning threads'
    assert 'torch.set_num_threads(1)' in src


# ══════════════════════════════════════════════════════════════════════
# 5. The gate's own criterion (WP-DET.3, audit D-DET.11)
#
# The gate is the only thing standing between "the training path is
# reproducible" and a claim nobody checked.  Its criterion is the sha256 of
# the POLICY WEIGHTS, not of the checkpoint FILE -- the file carries WP0's
# provenance and differs between any two runs by construction.  The script
# shipped for a while with the FILE digest written into the file the report
# labels as the weights digest, so the gate, re-run, failed on its own
# printed criterion while the record said it passed.  These guards are that
# defect's permanent detector.
# ══════════════════════════════════════════════════════════════════════

def _gate_src():
    return open(os.path.join(ROOT, 'tools', 'det_gate.sh'),
                encoding='utf-8').read()


def test_det_gate_computes_the_weights_digest():
    """policy.sha256 -- the file the report calls the gate criterion -- must
    be written by tools/policy_digest.py, and the FILE digest must go
    somewhere else."""
    src = _gate_src()
    m = re.search(r'policy_digest\.py[^\n]*--weights-only[^\n]*'
                  r'(?:\n[^\n]*?)?>\s*"\$d/policy\.sha256"', src)
    assert m, ('tools/det_gate.sh does not write policy.sha256 with '
               'tools/policy_digest.py --weights-only: the gate is not '
               'comparing the weights at all')
    assert re.search(r'sha256sum "\$d/ck/policy_final\.pt"[^\n]*>\s*'
                     r'"\$d/file\.sha256"', src), (
        'the checkpoint FILE digest is not written to file.sha256; the '
        'report prints it beside the criterion, so it has to exist')
    # ... and the file digest must NOT be what policy.sha256 holds.
    assert not re.search(r'sha256sum "\$d/ck/policy_final\.pt"[^\n]*>\s*'
                         r'"\$d/policy\.sha256"', src), (
        'policy.sha256 is being written from sha256sum of the checkpoint '
        'FILE: that is the digest that differs by provenance, and labelling '
        'it as the weights digest makes the gate fail on its own criterion')


def test_det_gate_reports_a_verdict():
    """A gate that only prints a table is not a gate: `report` must decide,
    and the decision must include the non-collapse check."""
    src = _gate_src()
    assert 'verdict()' in src and 'GATE: PASS' in src and 'GATE: FAIL' in src, \
        'tools/det_gate.sh report prints no PASS/FAIL verdict'
    assert 'share a policy digest' in src, (
        'the verdict does not check that two torch seeds give DIFFERENT '
        'policies; without it a gate on collapsed runs passes vacuously')
    # a crashed run must not be able to inherit the previous run's checkpoint
    assert 'stale' in src and 'rm -f "$d/ck/policy_final.pt"' in src, (
        'run_one does not clear the previous policy_final.pt; `rm -rf` on '
        'the run directory has been observed to fail, and a crashed run '
        'would then be digested as the previous run')
    # ... and a run that died AFTER writing the checkpoint must not pass
    # either.  `report` is a separate process from `run_one`, so the exit
    # status has to reach it on disk.
    assert 'echo "$rc" > "$d/rc"' in src, (
        'run_one does not record the training run\'s exit status; `report` '
        'cannot then tell a clean run from one that crashed after writing '
        'policy_final.pt and its last [MAPPO] line')
    assert 'training exited non-zero' in src, (
        'the verdict ignores the recorded exit status')
    assert 'no torch seed has two runs to compare' in src, (
        'the verdict does not check that anything was actually COMPARED: a '
        'single run per seed agrees with itself and would print PASS')


def test_weights_digest_covers_every_tensor():
    """Every tensor, its name, its dtype and its shape must reach the hash:
    a digest that missed any of them could pass two different policies."""
    import torch
    from tools.policy_digest import weights_sha256

    def ck(sd):
        return {'state_dict': sd, 'version': 0}

    base = {'a.weight': torch.zeros(2, 3), 'b.bias': torch.zeros(4),
            'z.weight': torch.zeros(3, 2)}
    d0, n_t, n_p = weights_sha256(ck(base))
    assert (n_t, n_p) == (3, 16), (n_t, n_p)

    for name in base:                       # every tensor, not just the first
        pert = {k: v.clone() for k, v in base.items()}
        pert[name].view(-1)[-1] = 1.0
        assert weights_sha256(ck(pert))[0] != d0, \
            'a change in %s does not reach the digest' % name

    renamed = {('c.bias' if k == 'b.bias' else k): v for k, v in base.items()}
    assert weights_sha256(ck(renamed))[0] != d0, 'tensor NAMES are not hashed'
    reshaped = dict(base, **{'a.weight': torch.zeros(3, 2)})
    assert weights_sha256(ck(reshaped))[0] != d0, 'tensor SHAPES are not hashed'
    retyped = dict(base, **{'b.bias': torch.zeros(4, dtype=torch.float64)})
    assert weights_sha256(ck(retyped))[0] != d0, 'tensor DTYPES are not hashed'
    # ... and the digest does not depend on the dict's insertion order
    shuffled = {k: base[k] for k in reversed(list(base))}
    assert weights_sha256(ck(shuffled))[0] == d0, \
        'the digest depends on the state dict insertion order'


def test_weights_digest_refuses_a_vacuous_checkpoint():
    """sha256 of nothing is a CONSTANT, so a weightless checkpoint would make
    every run 'agree'.  It must be refused, not digested."""
    import torch
    from tools.policy_digest import weights_sha256

    for bad in ({'state_dict': {}, 'version': 0}, {}, {'version': 0}):
        try:
            weights_sha256(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError('a weightless checkpoint was digested: %r'
                                 % (bad,))
    # a checkpoint whose OTHER tensor dict the digest would drop is refused.
    # BOTH shapes of that: an unrecognised key, and a SECOND recognised one.
    # The second is the one that actually bites -- D-DET.13's first open item
    # is "the critic's final weights are never digested", so the next WP to
    # touch this saves a second state dict, and only `state_dict` would have
    # been hashed while this tool's docstring promised a refusal.
    for two in ({'state_dict': {'a': torch.zeros(2)},
                 'critic_state_dict': {'b': torch.zeros(2)}},
                {'state_dict': {'a': torch.zeros(2)},
                 'model_state_dict': {'b': torch.zeros(2)}}):
        try:
            weights_sha256(two)
        except SystemExit:
            pass
        else:
            raise AssertionError(
                'a partial digest was taken over a checkpoint holding two '
                'tensor dicts (%s): two checkpoints differing only in the '
                'one that was dropped would compare EQUAL'
                % ', '.join(sorted(two)))


def _indent(line):
    return len(line) - len(line.lstrip())


def _loop_bodies(src, needle):
    """The body of every `for ...<needle>...:` loop in `src`, by indentation
    (a line belongs to the body while it is blank or indented deeper than the
    `for` line itself)."""
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if needle not in line or not line.lstrip().startswith('for '):
            continue
        head = _indent(line)
        body = []
        for nxt in lines[i + 1:]:
            if nxt.strip() and _indent(nxt) <= head:
                break
            body.append(nxt)
        yield '\n'.join(body)


# Directories the tree-wide scan below does not walk.  `output/` holds
# extracted copies of OTHER heads (the negative-control trees), which would
# be reported as offences in the current tree; `tests/` holds this guard
# itself, whose own `for body in _loop_bodies(src, 'as_completed(')` line
# would be scanned as if it were a worker loop.
_SCAN_SKIP_DIRS = {'.git', '__pycache__', 'output', 'logs', 'tests'}


def _tree_py_files():
    """Every .py file of the tree a driver could import or run."""
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs
                         if d not in _SCAN_SKIP_DIRS and not d.startswith('.'))
        for name in sorted(files):
            if name.endswith('.py'):
                yield os.path.join(base, name)


def test_no_unordered_pool_consumer_survives():
    """Tree-wide (WP-DET.2): an `as_completed` loop may collect, but nothing
    may be folded into a rollout pool inside one unless that loop is marked
    as off the evaluated path.

    Scoped by INDENTATION to the `as_completed` loop's own body, so the
    ordered `for res in order_batch_results(...)` fold that follows it in
    train_on_comparison.py is not mistaken for part of it.

    The scan walks the WHOLE tree (it used to glob only the root and
    `sixg_sim/`, so `sixg_sim/iops/`, `tools/` and any future package were
    outside it while the docstring said "tree-wide").
    """
    offenders = []
    n_loops = 0
    for path in _tree_py_files():
        src = open(path, encoding='utf-8', errors='replace').read()
        for body in _loop_bodies(src, 'as_completed('):
            n_loops += 1
            if '_pool.' not in body:
                continue
            if 'LEGACY' in body or 'off the evaluated path' in body:
                continue
            offenders.append(os.path.relpath(path, ROOT))
    assert not offenders, (
        'rollout pools are still extended in worker-completion order in: %s'
        % ', '.join(sorted(set(offenders))))
    # ... and the guard is not vacuous: it must still find the two loops the
    # tree is known to have.  Counted over the SCAN, not over a hard-coded
    # pair of files, so a third consumer anywhere trips this instead of
    # slipping past a guard that keeps reporting the two it was told about.
    assert n_loops == 2, (
        'expected exactly two as_completed loops in the scanned tree (the '
        'trainer and the legacy one), found %d -- a new one must be checked '
        'by hand and this count updated' % n_loops)
