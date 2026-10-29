"""WP2 -- driver parity, Block A variation and the training smoke.

These run the comparison environment (minutes, <compute-server> only) and MUST run
in a process whose sixg_sim modules were imported under the evaluated
environment: two settings are fixed at the FIRST sixg_sim import (the
entropy floor MARL_RELAY_ENTROPY_FLOOR in sixg_sim.mappo_trainer and the
steering model MARL_STEER_MODEL in sixg_sim.simulation; since WP8 the reward
is frozen and MARL_REWARD_PROFILE is only a tripwire that accepts `mission`),
and the engine variables are read from os.environ when a Simulator is built.
By the time a whole-suite run reaches this file the wp0/wp1 files have
already imported sixg_sim.  `REQUIRES_ENV` below tells tests/run_tests.py to
re-run this file in a child process with that environment, so the plain

    PYTHONHASHSEED=0 MARL_POOL_WORKERS=8 nice -n 19 python3 tests/run_tests.py

runs these tests too and no separate invocation is needed.  Running the file
on its own still works, with or without the environment exported:

    env MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile \\
        MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 \\
        MARL_REWARD_PROFILE=mission \\
        MARL_RELAY_ENTROPY_FLOOR=0.01 MARL_STEER_MODEL=actionable \\
        PYTHONHASHSEED=0 MARL_POOL_WORKERS=2 \\
        nice -n 19 python3 tests/run_tests.py wp2_parity

1. Parity: the same seed and checkpoint through the training driver's tick
   step (train_on_comparison.run_eval_episode, learning disabled) and the
   harness step (run_physics_pass, marl_static) for 300 ticks give identical
   action, delivered and fragment series.  Run for BOTH scenarios: A
   ('recovery', severed mid-run, so every site's action lock is closed
   before the event and opens on the site's own local-isolation flag after
   it -- D13.2, the per-site lock, in BOTH drivers) and B ('rescue_ops',
   pre-severed at tick 0, so every site's own flag is set by the tick-0 step
   and the agents act from tick 1; the degradation event fires at tick 0 --
   the branch run_eval_episode gained in WP2).
2. The two Block A backhaul fields vary during evaluation (they were 0 /
   full capacity on every evaluation tick before WP2).
3. A 2-episode training smoke with a tiny episode length.

The checkpoint fed to both drivers is MINTED from this tree (see CKPT
below).  WP6's strict schema gate refuses the shipped pre-WP6
ce_ck_s1234/policy_final.pt, and check 1 is a RERUN_SPEC invariant: it
builds a schema-current checkpoint for itself instead of going
unverified until a retrain lands.
"""
import os
import subprocess
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
# Read by tests/run_tests.py: these must be in the environment BEFORE the
# first sixg_sim import, so when this process was started without them the
# runner re-runs this file in a child process that has them.  `python3
# tests/run_tests.py` (no pattern) therefore runs these tests rather than
# failing them, and they cannot be silently skipped into a green suite.
REQUIRES_ENV = dict(EVAL_ENV)
if 'sixg_sim.agent' not in sys.modules:
    # nothing has fixed the import-time settings yet: a direct `pytest tests/test_wp2_parity.py`
    # or `python3 tests/run_tests.py wp2_parity` still works without the
    # wrapper.  (Not an unconditional setdefault: once sixg_sim is imported
    # the variables change nothing here and would only leak into later files.)
    for _k, _v in EVAL_ENV.items():
        os.environ.setdefault(_k, _v)

TICKS = 300
SEED = 82
TMP = os.path.join(ROOT, 'output', 'tmp_tests', 'wp2')
# The checkpoint both drivers are given.  MINTED FROM THIS TREE, not the
# shipped ce_ck_s1234/policy_final.pt: RERUN_SPEC WP6 made the observation
# schema gate strict, and it correctly REFUSES that file (fc1 62 wide, no
# provenance block -- it predates the WP6 schema and has to be retrained).
# The WP2 invariant is that the two DRIVERS agree tick for tick on ONE set
# of weights; it does not depend on WHICH weights, so the test builds a
# schema-current checkpoint for itself rather than dying at the gate and
# leaving a mandatory spec invariant unverified.  Point CKPT back at a
# ce_ck_* replicate once a retrain on this tree has produced a stamped one.
CKPT = os.path.join(TMP, 'policy_schema_current.pt')
CKPT_TORCH_SEED = 1234

_cache = {}


def _checkpoint():
    """Mint (once) the schema-current checkpoint both drivers load.

    Deterministic in CKPT_TORCH_SEED, and it leaves the global torch RNG
    state untouched, so it cannot shift either driver's draws.

    A file left over from an older schema is RE-MINTED rather than handed to
    the gate: a cached artefact of a previous tree failing the gate is a
    confusing way to report 'the schema changed'.
    """
    import torch

    from sixg_sim.agent import (SCHEMA_ID, checkpoint_schema_id,
                                mint_schema_current_checkpoint)
    if os.path.exists(CKPT):
        try:
            stale = checkpoint_schema_id(
                torch.load(CKPT, map_location='cpu',
                           weights_only=False)) != SCHEMA_ID
        except Exception:
            stale = True
        if stale:
            os.remove(CKPT)
    if not os.path.exists(CKPT):
        mint_schema_current_checkpoint(CKPT, torch_seed=CKPT_TORCH_SEED,
                                       extra={'test': 'wp2_parity'})
    return CKPT


def _imported_under_eval_env():
    """True when this process runs under EVAL_ENV: the engine variables are
    in os.environ (read when a Simulator is built) and the two import-time
    settings have the evaluated values.  Since WP8 the reward is frozen, so
    it is no longer the proxy (D5.8 used it); since WP5/WP6 neither is
    OBS_DIM nor the fragment-cue switch."""
    import sixg_sim.mappo_trainer as _T
    import sixg_sim.simulation as _S
    return (all(os.environ.get(k) == v for k, v in EVAL_ENV.items())
            and _T._RELAY_H_FLOOR == float(EVAL_ENV['MARL_RELAY_ENTROPY_FLOOR'])
            and _S.STEER_MODEL == EVAL_ENV['MARL_STEER_MODEL'])


def _harness(scenario='recovery'):
    """One 300-tick marl_static pass through run_physics_pass, cached per
    scenario ('recovery' = A, 'rescue_ops' = B)."""
    key = 'pass:' + scenario
    if key in _cache:
        return _cache[key]
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    if not _imported_under_eval_env():
        # Safety net only.  tests/run_tests.py re-runs this file in a child
        # process with REQUIRES_ENV applied, so this is reached only when the
        # file is driven some other way (bare pytest, no environment): a
        # LOUD failure with the command to use -- never a quiet skip, which
        # tests/test_wp6_parity.py forbids in every test file.
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 
            'sixg_sim was imported outside the evaluated environment; run '
            'this file on its own (module docstring) or through '
            'tests/run_tests.py')
    # min()-based, so re-applying the same total_ticks is idempotent
    R._apply_runtime_overrides({'total_ticks': TICKS})
    out = R.run_physics_pass('marl_static', SEED, _checkpoint(), scenario)
    _cache[key] = out
    _cache['severance'] = R.SEVERANCE_TICK
    _cache['rescue'] = R.RESCUE_ARRIVAL_TICK
    return out


def _eval_series(kind, severance_tick):
    """The same episode through the TRAINING driver's tick step, learning
    disabled (train_on_comparison.run_eval_episode)."""
    import torch
    import train_on_comparison as T
    ck = torch.load(_checkpoint(), map_location='cpu', weights_only=False)
    # The minted checkpoint is schema-current, so both drivers load the SAME
    # weights strictly (WP6); there is no adaptation step any more (D5.8's
    # 62 -> 60 truncation went with the WP6 gate).
    sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    return T.run_eval_episode(sd, SEED, kind, TICKS, event_seed=0,
                              severance_tick=severance_tick,
                              scenario_seed=SEED, random_seed=SEED)['series']


def _assert_series_parity(s, h, label, severance_tick, expect_partition):
    """s = training-driver series, h = harness pass output.

    FRAGMENTS ARE COMPARED ON THE UNGATED COMPONENT COUNT.  The harness
    reports `fragments = 1` on every non-island tick by definition
    ("Pre-disaster: single connected network via core") -- and, less
    visibly, so does the ENGINE: `get_island_kpis()['island_fragments']` is
    `count_infra_components(True) if island_mode else 1`, so the training
    driver's series was pinned to 1 pre-event too.  Both sides of the old
    comparison were therefore constants over Scenario A's first 50 of 300
    ticks, and no pre-event divergence between the drivers could fail the
    test.  Both drivers now record `count_infra_components(True)` itself
    (harness: `fragments_phys`; trainer: `series['fragments']`), which is a
    measurement on every tick, and the pre-event window is asserted
    separately: one whole network, so both must report 1.
    """
    assert len(s['action_sig']) == len(h['action_sig']) == TICKS, label
    for key in ('action_sig', 'delivered'):
        first = next((i for i in range(TICKS) if s[key][i] != h[key][i]), None)
        assert first is None, '%s: %s series differ from tick %s' % (
            label, key, first)
    hp = h['fragments_phys']
    assert len(hp) == TICKS, '%s: fragments_phys length' % label
    first = next((i for i in range(TICKS) if s['fragments'][i] != hp[i]), None)
    assert first is None, '%s: fragment series differ from tick %s' % (
        label, first)
    # pre-event window: one whole network, so BOTH drivers must measure one
    # component (a real measurement now, not the engine's pinned 1)
    pre = range(0, min(severance_tick, TICKS))
    assert all(s['fragments'][i] == 1 for i in pre), \
        '%s: training driver measures a partition before the event' % label
    assert all(hp[i] == 1 for i in pre), \
        '%s: harness measures a partition before the event' % label
    # in Scenario A the compared series is not a constant -- the event
    # really does partition the infrastructure graph -- so fragment
    # parity is a real constraint there.  Scenario B is NOT a partition:
    # build_comparison_scenario gives 'rescue_ops' a bare sever_core,
    # which removes the core but "leaves the surviving infra graph fully
    # connected", so its fragment series is 1 on every tick by design and
    # the parity weight there sits on the action and delivered series.
    if expect_partition:
        assert max(hp[severance_tick:]) > 1, (
            '%s: no partition after the event -- fragment parity is '
            'vacuous' % label)
    else:
        assert set(hp) == {1}, (
            '%s: rescue_ops partitioned the infrastructure graph; the '
            'scenario changed, so this test must too' % label)
    # and it IS the series the study reports, wherever that one is not pinned
    pinned = [i for i in range(TICKS)
              if h['fragments'][i] != hp[i] and h['fragments'][i] != 1]
    assert not pinned, ('%s: reported and measured fragment series disagree '
                        'on ticks %s' % (label, pinned[:5]))


def test_parity_training_step_vs_harness_step():
    """Scenario A: severance mid-run, so every site's action lock is closed
    before the event and opens on its OWN flag after it (D13.2)."""
    h = _harness('recovery')
    sev = _cache['severance']
    _assert_series_parity(_eval_series('recovery', sev), h, 'scenario A',
                          severance_tick=sev, expect_partition=True)
    # the lock: nobody acts before the event, the agents act after it
    assert all(a == 0 for a in h['action_sig'][:sev]), 'acted pre-event'
    assert any(a != 0 for a in h['action_sig'][sev:]), 'never acted'
    # ... and not at the GLOBAL trigger (the severance tick itself): a site
    # acts only once the link-state service has DETECTED and INSTALLED its
    # own isolation, which takes at least BFD_DETECT_TICKS + 1 ticks.  The
    # series are identical above, so this holds in both drivers.
    from sixg_sim.link_state_service import BFD_DETECT_TICKS
    first = next(i for i in range(TICKS) if h['action_sig'][i] != 0)
    assert first > sev + BFD_DETECT_TICKS, (
        'the agents acted at tick %d, %d tick(s) after the severance: that is '
        'the global trigger, not each site\'s own flag' % (first, first - sev))
    assert max(h['delivered']) > 0.0


def test_parity_scenario_b_pre_severed():
    """Scenario B: the network is already severed at tick 0, so every
    site's own flag is set by the tick-0 step and its action lock opens at
    tick 1 (D13.2), and the backhaul degradation fires at tick 0
    (run_eval_episode's `kind == 'rescue_ops'` branch, added by WP2).  The
    two drivers must still produce identical series."""
    h = _harness('rescue_ops')
    _assert_series_parity(_eval_series('rescue_ops', 0), h, 'scenario B',
                          severance_tick=0, expect_partition=False)
    # pre-severed: every site's own flag is set by the tick-0 step, so the
    # agents act from tick 1 -- not at tick 0, which has no previous step
    # (D13.2; the global lock let them act at tick 0)
    assert h['action_sig'][0] == 0, 'acted at tick 0, before any own flag'
    assert h['action_sig'][1] != 0, 'no action on tick 1 of B'
    assert max(h['delivered']) > 0.0
    # ... and the Block A backhaul fields carry the last pass here too
    assert max(h['obs_bh_util_max'][1:]) > 0.0
    assert len(set(h['obs_bh_cap_min'][1:])) > 2


def test_block_a_backhaul_fields_vary_in_evaluation():
    h = _harness('recovery')
    sev = _cache['severance']
    util = h['obs_bh_util_max']
    cap = h['obs_bh_cap_min']
    assert len(util) == len(cap) == TICKS
    # tick 0 has no pass behind it: defaults
    assert util[0] == 0.0 and cap[0] == 1000.0
    # from tick 1 on the fields carry the last pass: utilisation is no
    # longer identically 0 and the available capacity moves
    assert max(util[1:]) > 0.0
    assert len(set(util[1:])) > 2
    assert len(set(cap[1:])) > 2
    # ... in the pre-event period as well as after the event
    assert max(util[1:sev]) > 0.0 and max(util[sev:]) > 0.0


def test_training_smoke_two_episodes():
    os.makedirs(TMP, exist_ok=True)
    ck_dir = os.path.join(TMP, 'ck')
    out_dir = os.path.join(TMP, 'out')
    policy = os.path.join(ck_dir, 'policy_smoke.pt')
    env = dict(os.environ)
    env.update(EVAL_ENV)
    env.setdefault('PYTHONHASHSEED', '0')
    env['MARL_POOL_WORKERS'] = env.get('MARL_POOL_WORKERS', '2')
    cmd = [sys.executable, 'train_on_comparison.py', '--episodes', '2',
           '--episode-ticks', '120', '--batch-size', '2', '--seed-list', '42',
           '--eval-episodes', '1', '--baseline-episodes', '1',
           '--checkpoint-dir', ck_dir, '--output-dir', out_dir,
           '--save-policy-path', policy]
    log_path = os.path.join(TMP, 'train_smoke.log')
    with open(log_path, 'w', encoding='utf-8') as log:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
    log = open(log_path, encoding='utf-8', errors='replace').read()
    assert rc == 0, 'training smoke failed (rc=%d); see %s' % (rc, log_path)
    assert os.path.exists(policy), 'no policy written'
    assert 'EVALUATION PHASE' in log and 'BASELINE PHASE' in log
    assert 'Traceback' not in log
    # the degradation helper and the action step ran in the workers
    assert '[BH-DEGRADE]' in log
