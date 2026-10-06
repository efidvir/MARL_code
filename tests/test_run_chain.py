# -*- coding: utf-8 -*-
"""The plan's own properties, asserted against the object the launcher runs.

Every requirement COMPUTE_PLAN.md places on the chain that can be checked
without compute is checked here, against confirmatory/chain_plan.py -- the
same module confirmatory/run_final.sh reads.  The launcher's *behaviour*
(resume, the restart rule, the counting gates, the shed order) is exercised
in tests/test_run_chain_launcher.py.
"""
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'confirmatory')):
    if p not in sys.path:
        sys.path.insert(0, p)

import chain_plan as CP


def _mkdtemp(prefix):
    """Scratch directory inside the tree (output/tmp_tests), the WP0
    pattern, never the system default: tempfile.gettempdir() falls back to
    the CURRENT DIRECTORY when every candidate temp directory refuses a
    write (`/` full, as on 2026-10-01), and the suite runs from the tree
    root -- which is how eleven tmp* directories, some holding 3-byte stub
    pickles laid out like the real profile's output/final/, landed in
    ~/MARL_run_9 (D12F.2).  The seed audit skips tmp_tests by design."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def _rmtree(d):
    """Remove a scratch directory, retrying for up to 10 s: the tree is on
    NFS, where a file a just-finished child process still holds open is
    silly-renamed to .nfsXXXX and keeps its directory from being removed
    until the server lets go (the negative-control runners retry the same
    way).  The hygiene test (tests/test_zz_tree_hygiene.py) found one such
    leftover after the first removal attempt."""
    import time as _time
    for _ in range(20):
        shutil.rmtree(d, ignore_errors=True)
        if not os.path.exists(d):
            return
        _time.sleep(0.5)


@contextlib.contextmanager
def _scratch(prefix):
    """_mkdtemp, removed when the block ends, also when the test fails."""
    d = _mkdtemp(prefix)
    try:
        yield d
    finally:
        _rmtree(d)


def _real():
    return CP.build('real')


# ---------------------------------------------------------------- workload
def test_real_profile_reproduces_the_planned_pass_count():
    invs = _real()
    total = sum(i['expected_passes'] for i in invs)
    # 1056 planned passes + the 4 of the analysis-S8 hash-seed repeat (S6h,
    # FIX-PREFREEZE item 2a)
    assert total == 1060, 'plan workload is 1060 passes, got %d' % total
    assert len(invs) == 37, len(invs)
    per_stage = {}
    for i in invs:
        per_stage[i['stage']] = per_stage.get(i['stage'], 0) + i['expected_passes']
    assert per_stage['S1'] == 340, per_stage      # dev 280 + stress 60
    assert per_stage['S3'] == 56, per_stage       # validation
    assert per_stage['S4'] == 60, per_stage       # dev 40 + stress 20
    assert per_stage['S6'] == 320, per_stage      # confirmatory
    assert per_stage['S6h'] == 4, per_stage       # analysis S8: 2 seeds x 2 scenarios
    assert per_stage['S8'] == 160, per_stage      # untrained control
    assert per_stage['S9'] == 40, per_stage
    assert per_stage['S10'] == 40 and per_stage['S11'] == 40, per_stage


def test_every_invocation_is_at_most_64_passes():
    for profile in ('real', 'rehearsal'):
        for i in CP.build(profile):
            assert i['expected_passes'] <= 64, (profile, i['id'], i['expected_passes'])


def test_splits_are_on_whole_arm_seed_block_boundaries():
    """No invocation may hold a partial arm: every arm runs every seed of the
    block in both scenarios, so a partition can neither duplicate nor drop a
    pass."""
    for profile in ('real', 'rehearsal'):
        for i in CP.build(profile):
            if not i['expected_passes']:
                continue
            assert i['expected_passes'] == len(i['arms']) * len(i['seeds']) * 2
            assert len(set(i['expected_keys'])) == i['expected_passes']
            for arm in i['arms']:
                got = [k for k in i['expected_keys'] if k.startswith(arm + '|')]
                assert len(got) == len(i['seeds']) * 2, (i['id'], arm)


def test_no_pass_is_scheduled_twice_under_the_same_condition():
    """Within one stage AND one environment/checkpoint condition, a pass may
    appear exactly once.  (S1 legitimately runs the same arm and seed twice --
    once clean, once under the interferer -- which is a different condition,
    keyed here by the environment block and the checkpoint.)"""
    for profile in ('real', 'rehearsal'):
        seen = {}
        for i in CP.build(profile):
            cond = (i['stage'], tuple(sorted(i['env'].items())),
                    i['ckpt_kind'], i['ckpt_value'], i.get('ticks'))
            for k in i['expected_keys']:
                key = (cond, k)
                assert key not in seen, 'duplicate %s in %s and %s' % (k, seen.get(key), i['id'])
                seen[key] = i['id']


def test_confirmatory_runs_every_arm_in_the_same_invocations():
    """[M] and [N] share five invocations, so no confirmatory pass can be
    duplicated or lost and every arm shares one tree and one schema id."""
    s6 = [i for i in _real() if i['stage'] == 'S6']
    assert len(s6) == 5, len(s6)
    for i in s6:
        assert sorted(i['arms']) == sorted(CP.CONFIRMATORY_ARMS)
        assert i['expected_passes'] == 64
    seeds = sorted(s for i in s6 for s in i['seeds'])
    assert seeds == list(range(200, 220)), seeds


# ---------------------------------------------------------------- ordering
def test_dependency_graph_is_acyclic_and_matches_the_plan():
    done = set()
    for stage, _t, deps, order in CP.STAGES:
        for d in deps + order:
            assert d in done, '%s depends on %s, which does not run before it' % (stage, d)
        done.add(stage)
    deps = {s: d for s, _t, d, _o in CP.STAGES}
    assert deps['S3'] == ['S2']                       # validation needs training
    assert deps['S5b'] == ['S5']                       # dev-set analysis dry run
    assert 'S5b' in deps['S6'] and 'S3b' in deps['S6']  # freeze 2, via S5b
    assert deps['S7'] == ['S6']                        # primary needs all of [M]+[N]
    for s in ('S8', 'S9', 'S10', 'S11'):
        assert 'S5' in deps[s], s                      # never before FREEZE_FINAL
        assert s not in deps['S7'], 'the secondary family must not gate [R]'


def test_policy_independent_dev_and_stress_are_front_loaded():
    order = {s: n for n, (s, _t, _d, _o) in enumerate(CP.STAGES)}
    assert order['S1'] < order['S2'], 'S1 must be scheduled ahead of training'
    after = dict((s, o) for s, _t, _d, o in CP.STAGES)
    assert after['S2'] == ['S1'], 'the ordering constraint must be in the graph'
    s1 = [i for i in _real() if i['stage'] == 'S1']
    arms = sorted({a for i in s1 for a in i['arms']})
    assert 'marl_static' not in arms and 'marl_freeze' not in arms, arms
    assert sum(i['expected_passes'] for i in s1) == 340


def test_training_is_exclusive_and_nothing_else_is():
    for i in _real():
        assert i['exclusive'] == (i['kind'] == 'train'), i['id']


def test_training_flags_and_worker_count():
    tr = [i for i in _real() if i['kind'] == 'train']
    assert len(tr) == 4, len(tr)
    for i in tr:
        a = i['train_args']
        assert '--eval-episodes' in a and a[a.index('--eval-episodes') + 1] == '0'
        assert '--baseline-episodes' in a and a[a.index('--baseline-episodes') + 1] == '0'
        assert i['workers'] == 16, 'training worker count is configuration, not a knob'
        assert '--torch-seed' in a
    seeds = [i['train_args'][i['train_args'].index('--torch-seed') + 1] for i in tr]
    assert sorted(seeds) == ['1234', '2345', '3456', '4567'], seeds


def test_adapt_then_freeze_is_a_single_serial_invocation():
    s9 = [i for i in _real() if i['stage'] == 'S9']
    assert len(s9) == 1, 'marl_freeze invocations must never run two at once'
    assert s9[0]['serial_group'] == 'marl_freeze'


def test_evaluation_runs_at_24_workers_in_the_real_profile():
    for i in _real():
        if i['kind'] == 'eval':
            assert i['workers'] == 24, i['id']


# ---------------------------------------------------------------- shedding
def test_the_analysis_script_is_exercised_before_the_confirmatory_stage():
    """The rehearsal ran every S7 gate green and then confirmatory_analysis.py
    itself died.  S5b runs the same script on the development set -- its
    designed validation input, complete at the end of S4 -- and S6 does not
    start until it has produced a report."""
    invs = {i['id']: i for i in _real()}
    dev = invs['s5b_dev_analysis']
    assert dev['stage'] == 'S5b' and dev['step'] == 'dev'
    assert dev['script'] == 'confirmatory/stage_analysis.sh'
    order = {s: n for n, (s, _t, _d, _o) in enumerate(CP.STAGES)}
    assert order['S4'] < order['S5b'] < order['S6'] < order['S7']


def test_shed_groups_cover_the_plan_and_nothing_that_gates_the_primary():
    assert CP.SHED_ORDER == ['sensitivity', 'adapt_freeze', 'untrained']
    gated = {'S8', 'S9', 'S10', 'S11'}
    assert set(sum(CP.SHED_STAGES.values(), [])) == gated
    for i in _real():
        if i['stage'] in gated:
            assert i['shed_group'], i['id']
        elif i['stage'] in ('S1', 'S2', 'S3', 'S4', 'S6', 'S7'):
            assert not i['shed_group'], '%s must never be sheddable' % i['id']


# ---------------------------------------------------------------- hygiene
def test_rehearsal_output_never_lands_in_a_real_run_path():
    real_paths = {i['artifact'] for i in _real()}
    for i in CP.build('rehearsal'):
        art = i['artifact']
        assert art not in real_paths, art
        if art.startswith('output/'):
            assert art.startswith('output/rehearsal/'), art
        else:
            assert 'rehearsal' in art, art


def test_environment_block_pins_the_engine_and_the_policy():
    for i in CP.build('real'):
        if i['kind'] in ('eval', 'train'):
            for k, v in CP.ENGINE_ENV.items():
                if k == 'MARL_RELAY_REPOINT' and i['stage'] == 'S11':
                    assert i['env'][k] == 'off'     # the sensitivity itself
                    continue
                assert i['env'][k] == v, (i['id'], k)
            for k, v in CP.POLICY_ENV.items():
                assert i['env'][k] == v, (i['id'], k)


def test_the_two_sensitivities_differ_only_by_one_variable():
    base = {i['id']: i for i in _real()}
    a, b = base['s10_sens_install'], base['s11_sens_repoint']
    assert a['env']['MARL_LS_INSTALL'] == '0'
    assert b['env']['MARL_RELAY_REPOINT'] == 'off'
    assert a['arms'] == b['arms'] == ['marl_static']
    assert a['seeds'] == b['seeds'] == list(range(200, 220))
    # each differs from the confirmatory S6 environment by its ONE switch
    s6 = base['s6_cf_b0']['env']
    for inv, (k, v) in ((a, ('MARL_LS_INSTALL', '0')),
                        (b, ('MARL_RELAY_REPOINT', 'off'))):
        delta = {x for x in set(inv['env']) | set(s6)
                 if inv['env'].get(x) != s6.get(x)}
        assert delta == {k}, (inv['id'], sorted(delta))
        assert inv['env'][k] == v


def _s10_env_file(profile):
    """The env file the launcher writes for S10, produced by the REAL
    chain_write_env from the plan's --emit block (into a scratch directory)."""
    with _scratch('rc_s10env_') as d:
        out = subprocess.run(
            ['bash', '-c',
             'set -u; . confirmatory/chain_lib.sh; '
             'eval "$("$3" confirmatory/chain_plan.py --profile "$1" --emit s10_sens_install)"; '
             'CHAIN_PROFILE="$1"; CHAIN_ENVDIR="$2"; '
             'chain_write_env s10_sens_install "$INV_ENV" output/sel.pt 24; '
             'cat "$CHAIN_ENVFILE"', '_', profile, d, sys.executable],
            cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, (out.stdout, out.stderr)
        return out.stdout


def test_s10_runs_installation_off_not_the_omniscient_ablation():
    """FIX-PREFREEZE item 1.  The pre-registered evaluation-only S10 is
    "installation off": MARL_LS_INSTALL=0 (RERUN_SPEC WP3,
    docs/LINK_STATE_SERVICE.md section 7, D3.9).  MARL_DETECT_TICKS=0, which
    the chain ran until this fix, removes the link-state service altogether.
    Checked on the plan of every profile, on the sourceable --emit block and
    on the environment FILE the launcher's own chain_write_env writes -- the
    one the job script and the watchdog's restart source."""
    if shutil.which('bash') is None:
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'needs bash')
    checked = 0
    for profile in sorted(CP.PROFILES):
        inv = CP.by_id(profile, 's10_sens_install')
        assert inv['env'].get('MARL_LS_INSTALL') == '0', (profile, inv['env'])
        assert 'MARL_DETECT_TICKS' not in inv['env'], (profile, inv['env'])
        emitted = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'confirmatory', 'chain_plan.py'),
             '--profile', profile, '--emit', 's10_sens_install'],
            capture_output=True, text=True, check=True).stdout
        kv = dict(l.split('=', 1) for l in emitted.strip().splitlines())
        words = kv['INV_ENV'].strip("'").split()
        assert 'MARL_LS_INSTALL=0' in words, (profile, words)
        assert not [w for w in words if w.startswith('MARL_DETECT_TICKS')], words
        lines = _s10_env_file(profile).splitlines()
        assert 'export MARL_LS_INSTALL=0' in lines, (profile, lines)
        assert not [l for l in lines if 'MARL_DETECT_TICKS' in l], (profile, lines)
        checked += 1
    assert checked == len(CP.PROFILES) >= 4, checked
    # no invocation of any profile runs the omniscient ablation
    for profile in sorted(CP.PROFILES):
        for i in CP.build(profile):
            assert 'MARL_DETECT_TICKS' not in i['env'], (profile, i['id'])


def test_plan_cli_emits_a_sourceable_block():
    out = subprocess.run([sys.executable,
                          os.path.join(ROOT, 'confirmatory', 'chain_plan.py'),
                          '--profile', 'real', '--emit', 's6_cf_b0'],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    kv = dict(l.split('=', 1) for l in out.stdout.strip().splitlines())
    assert kv['INV_STAGE'] == 'S6'
    assert kv['INV_EXPECTED_PASSES'] == '64'
    assert kv['INV_EXCLUSIVE'] == '0'
    assert 'marl_static' in kv['INV_ARMS']


# ------------------------------------------------- the training memory gate
def test_the_training_memory_gate_admits_the_second_replicate_of_a_wave():
    """COMPUTE_PLAN.md states BOTH training thresholds for a WAVE: two
    concurrent replicates need 26 GB and abort below 22 GB (19.8 GB PSS /
    29.9 GB sum-RSS).  Once replicate 1 of the wave is resident, MemAvailable
    has already fallen by its footprint.  The 26 GB gate was fixed to add that
    footprint back; the 22 GB ABORT was not, and it short-circuits the gate --
    so every wave ended after one replicate.  These are the plan's own
    real-scale figures on <compute-server> (MemTotal 62.5 GB, MemAvailable 27.3 GB
    before the wave, SwapTotal 0).
    """
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import preflight
    g = preflight.GATES['train']

    # replicate 1 of the wave: nothing of ours is resident yet
    gate1, ok1, abort1 = preflight.memory_verdict(27.3, 0.0, g)
    assert ok1 and not abort1, (gate1, ok1, abort1)

    # replicate 2: replicate 1 holds half of the wave's 19.8 GB PSS, so raw
    # MemAvailable has fallen to 17.4 GB -- below the 22 GB floor.
    gate2, ok2, abort2 = preflight.memory_verdict(27.3 - 9.9, 9.9, g)
    assert not abort2, 'the abort floor must be a WAVE figure, not the raw one'
    assert ok2, 'the second replicate of a wave must be admitted: %r' % (gate2,)
    assert abs(gate2 - 27.3) < 1e-9, gate2


def test_the_training_abort_floor_still_fires_when_the_host_is_really_full():
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import preflight
    g = preflight.GATES['train']
    _, ok, abort = preflight.memory_verdict(10.0, 0.0, g)
    assert abort and not ok, 'a genuinely full host must still ABORT'
    # and a wave that is ALREADY over budget aborts even with our own jobs
    # counted back in
    _, ok, abort = preflight.memory_verdict(4.0, 8.0, g)
    assert abort and not ok


def test_evaluation_has_no_abort_floor_and_still_gates():
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import preflight
    g = preflight.GATES['eval']
    _, ok, abort = preflight.memory_verdict(19.0, 0.0, g)
    assert ok and not abort
    _, ok, abort = preflight.memory_verdict(17.0, 0.0, g)
    assert not ok and not abort, 'eval below 18 GB fails but never aborts'


# ------------------------------------------------------- the ablation gate
def _fake_pickle(path, arms, seeds, series):
    import pickle
    rec = {'recovery_multi': {}, 'rescue_multi': {}, 'seeds': list(seeds),
           'provenance': {'tree_head': 'faketest', 'tree_sha256': 'f' * 64,
                          'schema_id': 'fake', 'pythonhashseed': '0'},
           'pass_stats': {}}
    for arm in arms:
        for seed in seeds:
            for sk in ('recovery_multi', 'rescue_multi'):
                rec[sk].setdefault(arm, {})[seed] = dict(series)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(rec, f)


def _ablation(abl, base, series, counter='', arms='marl_static'):
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'stage_report.py'),
           '--ablation-diff', abl, '--baseline', base,
           '--ablation-series', series, '--ablation-arms', arms]
    if counter:
        cmd += ['--ablation-counter', counter]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_an_ablation_that_changed_nothing_is_not_recorded_as_a_pass():
    """The rehearsal's re-pointing sensitivity came out byte-identical to its
    baseline on all 45 series of all 8 passes and every gate in the chain
    reported it complete.  Counting keys proves the invocation RAN; it says
    nothing about whether the switch took."""
    with _scratch('rc_ablation_') as d:
        base = os.path.join(d, 'base.pkl')
        same = os.path.join(d, 'same.pkl')
        diff = os.path.join(d, 'diff.pkl')
        live = os.path.join(d, 'live.pkl')
        _fake_pickle(base, ['marl_static'], [200, 201],
                     {'repoints': 0, 'relay_bridges': [0] * 5, 'conn': [1.0] * 5})
        _fake_pickle(same, ['marl_static'], [200, 201],
                     {'repoints': 0, 'relay_bridges': [0] * 5, 'conn': [1.0] * 5})
        _fake_pickle(diff, ['marl_static'], [200, 201],
                     {'repoints': 0, 'relay_bridges': [0] * 5, 'conn': [0.5] * 5})
        # a baseline in which the mechanism DID fire
        _fake_pickle(live, ['marl_static'], [200, 201],
                     {'repoints': 7, 'relay_bridges': [1] * 5, 'conn': [1.0] * 5})

        r = _ablation(diff, base, 'repoints,relay_bridges,conn', 'repoints')
        assert r.returncode == 0, r.stdout
        assert 'PASS' in r.stdout and 'differing=4' in r.stdout, r.stdout

        # identical AND the mechanism never fired in the baseline -> NOT EXERCISED
        r = _ablation(same, base, 'repoints,relay_bridges,conn', 'repoints')
        assert r.returncode == 3, (r.returncode, r.stdout)
        assert 'NOT EXERCISED' in r.stdout and 'differing=0' in r.stdout, r.stdout

        # identical although the mechanism DID fire -> the switch did not take
        live2 = os.path.join(d, 'live2.pkl')
        _fake_pickle(live2, ['marl_static'], [200, 201],
                     {'repoints': 7, 'relay_bridges': [1] * 5, 'conn': [1.0] * 5})
        r = _ablation(live2, live, 'repoints,relay_bridges,conn', 'repoints')
        assert r.returncode == 1, (r.returncode, r.stdout)
        assert 'FAIL' in r.stdout and 'differing=0' in r.stdout, r.stdout
        assert 'switch did not take' in r.stdout, r.stdout

        # no declared counter: identical is always a failure
        r = _ablation(same, base, 'repoints,relay_bridges,conn')
        assert r.returncode == 1, (r.returncode, r.stdout)


def test_the_ablation_gate_fails_when_it_compared_nothing():
    with _scratch('rc_ablation0_') as d:
        a = os.path.join(d, 'a.pkl')
        b = os.path.join(d, 'b.pkl')
        _fake_pickle(a, ['marl_static'], [200], {'conn': [1.0]})
        _fake_pickle(b, ['marl_static'], [999], {'conn': [1.0]})
        r = _ablation(a, b, 'conn')
        assert r.returncode == 1 and 'share NO' in r.stdout, r.stdout
        r = _ablation(a, b, '')
        assert r.returncode == 1 and 'nothing to' in r.stdout, r.stdout


def test_every_ablation_invocation_declares_what_the_switch_must_move():
    for profile in ('real', 'rehearsal'):
        for i in CP.build(profile):
            if i['stage'] in ('S10', 'S11'):
                assert i['ablation_of'], i['id']
                assert i['ablation_series'], i['id']
            elif i['kind'] == 'eval':
                assert not i.get('ablation_of'), i['id']


# ---------------------------------------------------- the ledger tree pin
def test_the_tree_pin_comes_from_the_first_invocation_of_the_ledger():
    import json
    with _scratch('rc_treepin_') as d:
        led = os.path.join(d, 'l.jsonl')
        rows = [{'event': 'chain_start'},
                {'event': 'preflight', 'inv': 'a', 'tree_sha256': 'aaa'},
                {'event': 'preflight', 'inv': 'b', 'tree_sha256': 'aaa'},
                {'event': 'chain_end', 'rc': 0},
                {'event': 'chain_start'},
                {'event': 'preflight', 'inv': 'a', 'tree_sha256': 'bbb'}]
        with open(led, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + chr(10))
        out = subprocess.run([sys.executable,
                              os.path.join(ROOT, 'tools', 'stage_report.py'),
                              '--ledger-tree-pin', led],
                             capture_output=True, text=True)
        assert out.stdout.strip() == 'aaa', (
            'the pin spans the WHOLE ledger: a resume after a stop is part of the '
            'same run, and a tree change in between is what the pin is for. %s'
            % out.stdout)
        empty = os.path.join(d, 'e.jsonl')
        open(empty, 'w').close()
        out = subprocess.run([sys.executable,
                              os.path.join(ROOT, 'tools', 'stage_report.py'),
                              '--ledger-tree-pin', empty],
                             capture_output=True, text=True)
        assert out.stdout.strip() == '-', out.stdout


def test_the_timing_report_scopes_to_one_chain_run():
    """--timings used to sum every end record in the file, so a ledger holding
    an abandoned attempt and the run that finally worked gave a total that
    matched neither and could not be quoted from the tool."""
    import json
    with _scratch('rc_timings_') as d:
        led = os.path.join(d, 'l.jsonl')
        rows = [{'event': 'chain_start'},
                {'event': 'end', 'inv': 'x', 'stage': 'S1', 'status': 'fail',
                 'wall_s': 1000, 'passes': 0, 'artifact': ''},
                {'event': 'chain_end', 'rc': 1},
                {'event': 'chain_start'},
                {'event': 'end', 'inv': 'x', 'stage': 'S1', 'status': 'ok',
                 'wall_s': 7, 'passes': 0, 'artifact': ''},
                {'event': 'chain_end', 'rc': 0}]
        with open(led, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + chr(10))

        def run(scope):
            return subprocess.run(
                [sys.executable, os.path.join(ROOT, 'tools', 'stage_report.py'),
                 '--timings', led, '--profile', 'rehearsal', '--run', scope],
                capture_output=True, text=True, cwd=ROOT).stdout

        last = run('last')
        assert '1 completed invocations of 2 in the file' in last, last
        assert '1007' not in last, 'the abandoned run must not be summed in'
        assert '1000 s' in last, 'but it must still be stated, not silently dropped'
        allrun = run('all')
        assert '2 completed invocations of 2 in the file' in allrun, allrun
        assert '1007' in allrun, allrun
        assert '1 completed invocations of 2 in the file' in run('ok'), run('ok')


# ------------------------------------------- reduced-scale correctness rules
def test_the_steady_selection_window_scales_with_the_tick_count():
    """stage_select.sh used an absolute W = 500.  At the rehearsal's 600 ticks
    that window is ticks 100-600 and the severance is at tick 100, so the
    primary criterion scored the disruption, every replicate tied and the
    tie-break alone chose the checkpoint."""
    import shutil
    if shutil.which('bash') is None:
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'needs bash')
    for profile, usable in (('real', True), ('rehearsal', True), ('test', False)):
        r = subprocess.run(['bash', 'confirmatory/stage_select.sh',
                            '--window', profile],
                           capture_output=True, text=True, cwd=ROOT)
        assert ('usable=%s' % usable) in r.stdout, (profile, r.stdout, r.stderr)
        assert (r.returncode == 0) == usable, (profile, r.returncode, r.stdout)
    r = subprocess.run(['bash', 'confirmatory/stage_select.sh',
                        '--window', 'rehearsal'],
                       capture_output=True, text=True, cwd=ROOT)
    # 600 ticks -> W = 83, window 517-600, after the rescue arrival at 300
    assert 'ticks=600 W=83 window=517-600 severance=100 rescue=300' in r.stdout, r.stdout


def test_the_rehearsal_actually_exercises_individual_replicate_chaining():
    """Plan requirement 5 -- exactly `concurrent` alive, the next launched the
    moment one exits -- is only executed when there are MORE replicates than
    may run at once.  With two seeds and concurrent=2 both launch back to back
    and the stage gang-waits, which is the behaviour the plan replaces."""
    for profile in ('real', 'rehearsal'):
        P = CP.PROFILES[profile]
        assert len(P['torch_seeds']) > P['train']['concurrent'], profile


def test_the_analysis_dry_run_reads_a_set_it_can_actually_analyse():
    """confirmatory_analysis.py holds absolute tick windows (PRE_SURGE ends at
    1400) and cannot run below 1400 ticks.  Every profile must therefore point
    S5b at invocations that run at the NOMINAL tick count."""
    for profile in ('real', 'rehearsal'):
        P = CP.PROFILES[profile]
        prefixes = [x for x in P['dev_analysis_prefixes'].split(',') if x]
        assert prefixes, profile
        invs = CP.build(profile)
        matched = [i for i in invs
                   if any(i['id'].startswith(x) for x in prefixes)]
        assert matched, (profile, prefixes)
        arms = {a for i in matched for a in i['arms']}
        for need in ('marl_static', 'xldet', 'sdn', 'ospf', 'olsr', 'batman',
                     'aodv', 'random'):
            assert need in arms, (profile, need, sorted(arms))
        for i in matched:
            assert i['ticks'] in (None, P['nominal_ticks']), (profile, i['id'],
                                                              i['ticks'])
            assert len(i['seeds']) >= 2, (profile, i['id'])


def test_the_primary_analysis_reads_a_set_it_can_actually_analyse():
    """WP13.  The same holds for S7: confirmatory_analysis.py refuses a set
    below the nominal tick count ("series ... has 600 entries, expected 3600
    (run length)"), so the 2026-09-24 rehearsal could only fail there.  Every
    profile points S7 at S6 invocations that run at the NOMINAL tick count,
    with all eight arms, at least two seeds, and seeds the S0 manifest covers;
    a full-length profile analyses the whole confirmatory stage."""
    n = 0
    for profile in ('real', 'rehearsal', 'dryrun'):
        P = CP.PROFILES[profile]
        prefixes = [x for x in P['cf_analysis_prefixes'].split(',') if x]
        assert prefixes, profile
        invs = CP.build(profile)
        matched = [i for i in invs
                   if any(i['id'].startswith(x) for x in prefixes)]
        assert matched, (profile, prefixes)
        assert all(i['stage'] == 'S6' for i in matched), profile
        for i in matched:
            assert sorted(i['arms']) == sorted(CP.CONFIRMATORY_ARMS), i['id']
            assert i['ticks'] in (None, P['nominal_ticks']), (profile, i['id'],
                                                              i['ticks'])
            assert set(i['seeds']) <= set(P['cf_seeds']), (profile, i['id'])
        seeds = sorted(s for i in matched for s in i['seeds'])
        assert len(seeds) >= 2 and len(seeds) == len(set(seeds)), (profile, seeds)
        if P['ticks'] is None:
            s6 = sorted(i['id'] for i in invs if i['stage'] == 'S6')
            assert sorted(i['id'] for i in matched) == s6, (profile, s6)
            assert seeds == sorted(P['cf_seeds']), profile
        n += 1
    assert n == 3


def test_an_ablation_is_compared_with_the_baseline_at_its_own_tick_count():
    """WP13.  A reduced profile's S6 also holds the full-length subset for the
    same (arm, seed, scenario) keys.  The S10/S11 ablation gate compares its
    600-tick passes with S6; mixing in a 3600-tick baseline would make every
    series differ trivially and report the switch as exercised.  The launcher
    asks the plan for the baseline pickles AT THE ABLATION'S TICK COUNT."""
    src = open(os.path.join(ROOT, 'confirmatory', 'run_final.sh'),
               encoding='utf-8').read()
    assert '--artifacts "$INV_ABLATION_OF" --same-ticks-as "$id"' in src
    n = 0
    for profile in ('rehearsal', 'real'):
        invs = {i['id']: i for i in CP.build(profile)}
        for iid, inv in sorted(invs.items()):
            if not inv.get('ablation_of'):
                continue
            out = subprocess.run(
                [sys.executable, os.path.join(ROOT, 'confirmatory', 'chain_plan.py'),
                 '--profile', profile, '--artifacts', inv['ablation_of'],
                 '--same-ticks-as', iid],
                capture_output=True, text=True, check=True).stdout.strip()
            arts = [a for a in out.split(',') if a]
            assert arts, (profile, iid)
            by_art = {i['artifact']: i for i in invs.values()}
            for a in arts:
                assert by_art[a]['ticks'] == inv['ticks'], (profile, iid, a)
            if profile == 'rehearsal':
                assert not any('cffull' in a for a in arts), arts
            n += 1
    assert n == 4, n


def _fake_py(tmp, log):
    """A stand-in for $PY: the plan runs for real, the analysis and the gate
    tools only record how they were called."""
    path = os.path.join(tmp, 'fakepy')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('#!%s\n' % sys.executable)
        fh.write('''import json, os, sys
LOG = %r
a = sys.argv[1:]
if a and (a[0] == '-' or a[0].endswith('chain_plan.py')):
    os.execv(sys.executable, [sys.executable] + a)
with open(LOG, 'a') as fh:
    fh.write(json.dumps(a) + '\\n')
if a and a[0] == 'confirmatory_analysis.py':
    out = a[a.index('--out') + 1]
    json.dump({'primary': {'P1': {}}, 'secondary': {},
               'seeds': {'A': [200], 'B': [200]}}, open(out, 'w'))
sys.exit(0)
''' % log)
    os.chmod(path, 0o755)
    return path


def _run_primary(profile, make):
    import json
    import shutil
    with _scratch('rc_primary_') as tmp:
        os.makedirs(os.path.join(tmp, 'confirmatory'))
        for f in ('stage_analysis.sh', 'chain_plan.py'):
            shutil.copy(os.path.join(ROOT, 'confirmatory', f),
                        os.path.join(tmp, 'confirmatory', f))
        for art in make:
            p = os.path.join(tmp, art)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'wb') as fh:
                fh.write(b'pkl')
        log = os.path.join(tmp, 'calls.log')
        env = dict(os.environ, PY=_fake_py(tmp, log))
        r = subprocess.run(['bash', os.path.join(tmp, 'confirmatory',
                                                 'stage_analysis.sh'),
                            '--run', profile, 's7_analysis'],
                           cwd=tmp, env=env, capture_output=True, text=True)
        calls = []
        if os.path.exists(log):
            calls = [json.loads(l) for l in open(log) if l.strip()]
        keys = os.path.join(tmp, 'logs', 'chain_%s' % profile, 'S6.keys')
        keyset = open(keys).read().split() if os.path.exists(keys) else []
        return r, calls, keyset


def test_the_primary_step_analyses_exactly_the_profiles_confirmatory_input():
    """WP13.  confirmatory/stage_analysis.sh's primary step, run for real
    with the analysis and gate tools stubbed: in the rehearsal it hands the
    full-length subset, and only it, to the key-union gate and to
    confirmatory_analysis.py; in the real profile the whole of S6; and a
    missing input pickle stops it."""
    if not sys.platform.startswith('linux'):
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'the stage script needs bash and a POSIX shebang')
    reh = {i['id']: i for i in CP.build('rehearsal')}
    r, calls, keyset = _run_primary('rehearsal', [
        reh['s6_cf_b0']['artifact'], reh['s6_cffull_b0']['artifact']])
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    ana = [c for c in calls if c and c[0] == 'confirmatory_analysis.py']
    assert len(ana) == 1, calls
    assert ana[0][ana[0].index('--pickles') + 1] == reh['s6_cffull_b0']['artifact']
    union = [c for c in calls if '--key-union' in c]
    assert len(union) == 1 and union[0][union[0].index('--key-union') + 1] \
        == reh['s6_cffull_b0']['artifact'], calls
    assert sorted(keyset) == sorted(reh['s6_cffull_b0']['expected_keys'])
    assert len(keyset) == 32, len(keyset)
    # the full-length input missing: the step fails instead of analysing the
    # reduced set
    r2, calls2, _k = _run_primary('rehearsal', [reh['s6_cf_b0']['artifact']])
    assert r2.returncode != 0, r2.stdout[-1000:]
    assert not [c for c in calls2 if c and c[0] == 'confirmatory_analysis.py']
    # the real profile: all five confirmatory invocations
    real = [i for i in CP.build('real') if i['stage'] == 'S6']
    r3, calls3, keys3 = _run_primary('real', [i['artifact'] for i in real])
    assert r3.returncode == 0, (r3.stdout[-2000:], r3.stderr[-2000:])
    ana3 = [c for c in calls3 if c and c[0] == 'confirmatory_analysis.py']
    assert len(ana3) == 1
    assert sorted(ana3[0][ana3[0].index('--pickles') + 1].split(',')) \
        == sorted(i['artifact'] for i in real)
    assert len(keys3) == 320, len(keys3)


# ------------------- the secondary analyses the chain computes (FIX-PREFREEZE 2)
def test_the_plan_declares_the_inputs_of_s8_s12_and_secondary_s7():
    """PREREGISTRATION_R2 section 10.  Analysis S8 needs a PYTHONHASHSEED=1
    repeat of the selected checkpoint on the two lowest confirmatory seeds,
    paired with the S6 invocation at the same tick count that holds them;
    analysis S12 needs one pickle per untrained actor and secondary S7 the
    adapt-then-freeze pickle, both at the NOMINAL tick count on exactly the
    seeds the primary analysis reads (the analysis refuses anything else).
    Counts what it checked."""
    n = 0
    for profile in ('real', 'rehearsal', 'dryrun', 'test'):
        P = CP.PROFILES[profile]
        invs = {i['id']: i for i in CP.build(profile)}
        prim = sorted(P.get('cf_full_seeds') or P['cf_seeds'])
        # S8: the hash pair
        h1 = [i for i in invs.values() if i.get('analysis_role') == 'hash1']
        assert len(h1) == 1, (profile, h1)
        h1 = h1[0]
        assert h1['stage'] == 'S6h' and h1['hashseed'] == '1', h1
        assert h1['arms'] == ['marl_static'] and h1['ckpt_kind'] == 'selected'
        assert h1['seeds'] == sorted(P['cf_seeds'])[:2], (profile, h1['seeds'])
        h0 = invs[h1['hash_pair_of']]
        assert h0['stage'] == 'S6' and h0['hashseed'] == '0', h0['id']
        assert h0['ticks'] == h1['ticks'] and set(h1['seeds']) <= set(h0['seeds'])
        assert {k: v for k, v in h0['env'].items()} == h1['env'], profile
        assert not h1['shed_group'], 'the S8 repeat is never shed'
        assert [i['id'] for i in invs.values() if i.get('hashseed', '0') != '0'] == [h1['id']]
        # S12: one control per untrained actor
        un = [i for i in invs.values() if i.get('analysis_role') == 'untrained']
        assert sorted(i['control_name'] for i in un) == sorted(
            'untrained_s%d' % s for s in P['torch_seeds']), profile
        # secondary S7: the adapt-then-freeze pickle
        fz = [i for i in invs.values() if i.get('analysis_role') == 'freeze']
        assert len(fz) == 1 and fz[0]['arms'] == ['marl_freeze'], profile
        for i in un + fz:
            if profile != 'test':     # the paper profile runs 60 ticks throughout
                assert i['ticks'] in (None, P['nominal_ticks']), (profile, i['id'])
            assert sorted(i['seeds']) == prim, (profile, i['id'], i['seeds'])
        for i in un:
            assert i['stage'] == 'S8' and i['ckpt_kind'] == 'literal', i['id']
            assert i['ckpt_value'].endswith('untrained_s%s/policy_init.pt'
                                            % i['control_name'].split('_s')[-1])
        assert fz[0]['stage'] == 'S9' and fz[0]['ckpt_kind'] == 'selected'
        if P['ticks'] is None:
            # a full-length profile reads its own S8/S9 invocations
            assert sorted(i['id'] for i in un) == sorted(
                i['id'] for i in invs.values() if i['stage'] == 'S8')
            assert fz[0]['id'] == 's9_freeze'
        deps = {s: d for s, _t, d, _o in CP.STAGES}
        assert {'S7', 'S6h', 'S8', 'S9'} <= set(deps['S12']), deps['S12']
        n += 1
    assert n == 4
    # the CLI the stage script asks
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'confirmatory', 'chain_plan.py'),
                          '--profile', 'real', '--analysis-role', 'untrained'],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ['s8_untrained_s%d' % s for s in CP.PROFILES['real']['torch_seeds']], out


def _write_fake_pickle(path, inv, hashseed=None):
    import pickle
    rec = {'recovery_multi': {}, 'rescue_multi': {}, 'seeds': list(inv['seeds']),
           'provenance': {'tree_head': 'faketest', 'tree_sha256': 'f' * 64,
                          'schema_id': 'fake', 'checkpoint_sha256': 'c' * 64,
                          'pythonhashseed': hashseed or inv.get('hashseed', '0')},
           'pass_stats': {}}
    for key in inv['expected_keys']:
        arm, scen, seed = key.split('|')
        tgt = rec['recovery_multi'] if scen == 'recovery' else rec['rescue_multi']
        tgt.setdefault(arm, {})[int(seed)] = {'conn': [1.0]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(rec, fh)


def _fake_py_secondary(tmp, log):
    """$PY for the S12 step: the plan and the inline checks run for real; the
    gate tools, the S8 check, the analysis and the report comparison only
    record how they were called and answer per FAKE_* environment variables."""
    path = os.path.join(tmp, 'fakepy')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('#!%s\n' % sys.executable)
        fh.write('''import json, os, sys
LOG = %r
a = sys.argv[1:]
if a and (a[0] == '-' or a[0].endswith('chain_plan.py')):
    os.execv(sys.executable, [sys.executable] + a)
with open(LOG, 'a') as fh:
    fh.write(json.dumps(a) + '\\n')
tool = os.path.basename(a[0]) if a else ''
def arg(k, d=None):
    return a[a.index(k) + 1] if k in a else d
if tool == 'analysis_manifest.py':
    print('derived_manifest.json')
elif tool == 'hash_invariance_check.py':
    seeds = [int(s) for s in arg('--seeds', '').split(',') if s]
    json.dump({'n_compared': int(os.environ.get('FAKE_S8_N', '12')),
               'same_hashseed': False, 'seeds': seeds,
               'scenarios': ['recovery', 'rescue_ops'],
               'decision_series': {'runs_with_key': 4}, 'identical': True},
              open(arg('--out'), 'w'))
    print('SUMMARY compared=12 different=0 missing=0 same_hashseed=False -> IDENTICAL')
    sys.exit(int(os.environ.get('FAKE_S8_RC', '0')))
elif tool == 'confirmatory_analysis.py':
    pk = arg('--pickles', '').split(',')
    rep = {'primary': {'P1': {}}, 'seeds': {'A': [1], 'B': [1]}, 'secondary': {}}
    if any('s9_' in p for p in pk):
        for k in ('A', 'B'):
            for q in ('ach', 'components'):
                rep['secondary']['S7_%%s_%%s_marl_static_minus_marl_freeze' %% (k, q)] = {'n': 2}
    rep['S12'] = {}
    for item in [x for x in arg('--untrained', '').split(',') if x]:
        rep['S12'][item.split('=')[0]] = {
            'summary': {'A': {'n': 2}, 'B': {'n': 2}}, 'paired': {'A': {}, 'B': {}}}
    if '--s8' in a:
        rep['S8'] = json.load(open(arg('--s8')))
    for drop in [x for x in os.environ.get('FAKE_DROP', '').split(',') if x]:
        rep.pop(drop, None)
    json.dump(rep, open(arg('--out'), 'w'))
elif tool == 'compare_reports.py':
    print('SUMMARY leaves=10 mismatches=0 -> IDENTICAL')
    sys.exit(int(os.environ.get('FAKE_CMP_RC', '0')))
sys.exit(0)
''' % log)
    os.chmod(path, 0o755)
    return path


def _run_secondary(profile, skip=(), shed=(), env=None):
    """Run confirmatory/stage_analysis.sh --run PROFILE s12_secondary in a
    scratch tree holding the profile's pickles (minus `skip`), the chain's
    primary report, and a ledger recording `shed` stages."""
    import json
    tmp = _mkdtemp('rc_secondary_')
    try:
        os.makedirs(os.path.join(tmp, 'confirmatory'))
        for f in ('stage_analysis.sh', 'chain_plan.py'):
            shutil.copy(os.path.join(ROOT, 'confirmatory', f),
                        os.path.join(tmp, 'confirmatory', f))
        invs = CP.build(profile)
        for i in invs:
            if i['artifact'].endswith('.pkl') and i['id'] not in skip \
                    and i['stage'] in ('S6', 'S6h', 'S8', 'S9', 'S10', 'S11'):
                _write_fake_pickle(os.path.join(tmp, i['artifact']), i)
        prim = [i for i in invs if i['id'] == 's7_analysis'][0]['artifact']
        json.dump({'primary': {'P1': {}}}, open(os.path.join(tmp, prim), 'w'))
        with open(os.path.join(tmp, 'confirmatory', 'chain_%s.jsonl' % profile), 'w') as fh:
            for st in shed:
                fh.write(json.dumps({'event': 'shed', 'stage': st}) + chr(10))
        log = os.path.join(tmp, 'calls.log')
        e = dict(os.environ, PY=_fake_py_secondary(tmp, log))
        e.update(env or {})
        r = subprocess.run(['bash', os.path.join(tmp, 'confirmatory', 'stage_analysis.sh'),
                            '--run', profile, 's12_secondary'],
                           cwd=tmp, env=e, capture_output=True, text=True)
        calls = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
        rep_path = os.path.join(tmp, [i for i in invs if i['id'] == 's12_secondary'][0]['artifact'])
        rep = json.load(open(rep_path)) if os.path.exists(rep_path) else None
        return r, calls, rep
    finally:
        _rmtree(tmp)


def _calls_of(calls, tool):
    return [c for c in calls if c and os.path.basename(c[0]) == tool]


def test_s12_computes_s8_s12_and_secondary_s7_from_the_chains_own_pickles():
    """FIX-PREFREEZE item 2.  The chain's S12 step, run for real with the
    tools stubbed: it compares the declared hash pair, hands every untrained
    control (NAME=PKL) and the adapt-then-freeze pickle to ONE further run of
    the pre-registered analysis over exactly S7's primary input, embeds the S8
    result, checks by counting that S7-secondary, S8 and S12 are in the
    report, and requires the chain's primary report to be reproduced."""
    if not sys.platform.startswith('linux'):
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'the stage script needs bash and a POSIX shebang')
    for profile in ('rehearsal', 'real'):
        invs = {i['id']: i for i in CP.build(profile)}
        r, calls, rep = _run_secondary(profile)
        assert r.returncode == 0, (profile, r.stdout[-3000:], r.stderr[-2000:])
        hc = _calls_of(calls, 'hash_invariance_check.py')
        assert len(hc) == 1, calls
        h1 = invs['s6h_hash1']
        assert hc[0][1:3] == [invs[h1['hash_pair_of']]['artifact'], h1['artifact']], hc
        assert '--require-decision' in hc[0]
        assert hc[0][hc[0].index('--seeds') + 1] == ','.join(str(s) for s in h1['seeds'])
        hp = [c for c in calls if '--hash-pair' in c]
        assert len(hp) == 1 and hp[0][hp[0].index('--expect-hashseeds') + 1] == '0,1', hp
        ana = _calls_of(calls, 'confirmatory_analysis.py')
        assert len(ana) == 1, calls
        pk = ana[0][ana[0].index('--pickles') + 1].split(',')
        prim = [i['artifact'] for i in invs.values() if i['stage'] == 'S6'
                and any(i['id'].startswith(x) for x in
                        CP.PROFILES[profile]['cf_analysis_prefixes'].split(','))]
        fz = [i['artifact'] for i in invs.values() if i.get('analysis_role') == 'freeze']
        assert sorted(pk) == sorted(prim + fz), (profile, pk)
        un = ana[0][ana[0].index('--untrained') + 1].split(',')
        assert sorted(un) == sorted('%s=%s' % (i['control_name'], i['artifact'])
                                    for i in invs.values()
                                    if i.get('analysis_role') == 'untrained'), un
        assert ana[0][ana[0].index('--marl-arm') + 1] == 'marl_static'
        assert ana[0][ana[0].index('--s8') + 1].endswith('s8_hash.json')
        cmp_ = _calls_of(calls, 'compare_reports.py')
        assert len(cmp_) == 1 and cmp_[0][1] == invs['s7_analysis']['artifact'], cmp_
        assert not _calls_of(calls, 'untrained_analysis.py'), 'the first study script is not called'
        a = rep['analyses']
        assert a['S8_hash_invariance']['state'] == 'identical', a
        assert a['S12_untrained']['state'] == 'computed'
        assert a['S12_untrained']['controls'] == len(CP.PROFILES[profile]['torch_seeds'])
        assert a['S7_secondary_marl_freeze']['state'] == 'computed', a
        assert a['full_report']['state'] == 'computed', a
        assert a['full_report']['primary_report_reproduced'] == 'identical', a
        assert 'S6h=complete(1)' in rep['stages'], rep['stages']
        assert 'secondary S7 computed: 4 comparison(s)' in r.stdout
        assert 'analysis S12 computed: %d untrained control(s)' % len(
            CP.PROFILES[profile]['torch_seeds']) in r.stdout
        assert 'analysis S8 embedded: 12 series compared' in r.stdout


def test_s12_fails_loudly_when_an_input_is_missing_or_an_analysis_is_absent():
    """Each analysis counts what it got: a missing input is a FAILURE unless
    the ledger records the operator shedding that stage, an analysis absent
    from the report is a FAILURE, a vacuous S8 check is a FAILURE, and a full
    report that does not reproduce the primary report is a FAILURE.  A
    difference found by the S8 check is the analysis's RESULT, not a
    pipeline failure."""
    if not sys.platform.startswith('linux'):
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'the stage script needs bash and a POSIX shebang')
    p = 'rehearsal'
    n = 0
    # the adapt-then-freeze pickle missing, not shed
    r, calls, rep = _run_secondary(p, skip=('s9_freezefull',))
    assert r.returncode != 0 and 'secondary S7' in r.stdout, r.stdout[-2000:]
    assert not _calls_of(calls, 'confirmatory_analysis.py'), calls
    n += 1
    # ... and SHED by the operator (both S9 pickles absent): recorded, not computed
    r, calls, rep = _run_secondary(p, skip=('s9_freeze', 's9_freezefull'), shed=('S9',))
    assert r.returncode == 0, r.stdout[-3000:]
    assert rep['analyses']['S7_secondary_marl_freeze']['state'] == 'shed'
    pk = _calls_of(calls, 'confirmatory_analysis.py')[0]
    assert not any('s9_' in x for x in pk[pk.index('--pickles') + 1].split(','))
    n += 1
    # one untrained control missing
    r, calls, rep = _run_secondary(p, skip=('s8_untrainedfull_s2345',))
    assert r.returncode != 0 and 'untrained control pickle(s) missing' in r.stdout, r.stdout[-2000:]
    assert not _calls_of(calls, 'confirmatory_analysis.py')
    n += 1
    # the hash-seed repeat missing (S6h is never shed)
    r, calls, rep = _run_secondary(p, skip=('s6h_hash1',))
    assert r.returncode != 0 and 'analysis S8 input missing' in r.stdout, r.stdout[-2000:]
    assert not _calls_of(calls, 'hash_invariance_check.py')
    n += 1
    # a vacuous S8 check (exit 2), and one that compared nothing
    for env in ({'FAKE_S8_RC': '2'}, {'FAKE_S8_N': '0'}):
        r, calls, rep = _run_secondary(p, env=env)
        assert r.returncode != 0, (env, r.stdout[-2000:])
        n += 1
    # a found difference is recorded as the S8 result
    r, calls, rep = _run_secondary(p, env={'FAKE_S8_RC': '1'})
    assert r.returncode == 0, r.stdout[-2000:]
    assert rep['analyses']['S8_hash_invariance']['state'] == 'NOT_IDENTICAL'
    n += 1
    # an analysis given its input but absent from the report
    for drop in ('S12', 'S8', 'secondary'):
        r, calls, rep = _run_secondary(p, env={'FAKE_DROP': drop})
        assert r.returncode != 0, (drop, r.stdout[-2000:])
        n += 1
    # the full report does not reproduce the chain's primary report
    r, calls, rep = _run_secondary(p, env={'FAKE_CMP_RC': '1'})
    assert r.returncode != 0 and 'does not reproduce' in r.stdout, r.stdout[-2000:]
    n += 1
    assert n == 11, n


# ------------------------------------------- delay a wave, do not abandon it
def test_a_busy_host_refuses_a_training_wave_retryably():
    """COMPUTE_PLAN.md failure handling B: above ~20 cores of tenant load a
    training wave is DELAYED, not run -- a perturbed draw is worse than a late
    one (D0.8).  The refusal must be exit 1 (retry later is legitimate), never
    exit 2, which is the hard memory floor and is not retried."""
    if not sys.platform.startswith('linux'):
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'the preflight reads /proc')
    env = dict(os.environ)
    env.update({'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'preflight.py'),
                        '--kind', 'train', '--tree', ROOT, '--inv', 's2_probe',
                        '--profile', 'rehearsal', '--train-load-ceiling', '0.0001'],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert 'FAIL tenant_load' in r.stdout, r.stdout
    assert 'DELAY the wave' in r.stdout, r.stdout
    # and with a ceiling nothing can exceed, that check passes
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'preflight.py'),
                        '--kind', 'train', '--tree', ROOT, '--inv', 's2_probe',
                        '--profile', 'rehearsal', '--train-load-ceiling', '1e9'],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert 'FAIL tenant_load' not in r.stdout, r.stdout


def test_the_load_ceiling_only_applies_when_a_wave_starts():
    """Once replicates of this chain are resident they ARE the load; retesting
    the ceiling would refuse every later replicate of the same wave, which is
    the same double-charging the memory gate was fixed for."""
    if not sys.platform.startswith('linux'):
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'the preflight reads /proc')
    import json
    env = dict(os.environ)
    env.update({'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    with _scratch('rc_preflight_') as _pfd:
        out = os.path.join(_pfd, 'pf.json')
        subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'preflight.py'),
                        '--kind', 'train', '--tree', ROOT, '--inv', 's2_probe',
                        '--profile', 'rehearsal', '--train-load-ceiling', '0.0001',
                        '--json-out', out, '--quiet'],
                       capture_output=True, text=True, cwd=ROOT, env=env)
        rec = json.load(open(out))
        chk = [c for c in rec['checks'] if c['check'] == 'tenant_load']
        assert len(chk) == 1, rec['checks']
        assert chk[0]['fatal'] is True and rec['train_wave_starting'] is True, rec


def test_the_training_stage_delays_instead_of_stopping_on_the_first_refusal():
    """A structural guard for the defect the rehearsal found: stage_train.sh
    had no retry at all -- run_final.sh's evaluation path had one -- so a
    70 MB shortfall against the 26 GB wave gate ended the whole training stage
    after two of three replicates."""
    src = open(os.path.join(ROOT, 'confirmatory', 'stage_train.sh')).read()
    assert 'until chain_preflight train' in src, 'the refusal must be retried'
    assert 'TRAIN_PREFLIGHT_RETRIES' in src and 'TRAIN_PREFLIGHT_WAIT' in src
    assert 'reason=wave_delayed' in src, 'and every delay must be recorded'
    assert 'reason=preflight_abort' in src, 'the hard floor is still not retried'
    launcher = open(os.path.join(ROOT, 'confirmatory', 'run_final.sh')).read()
    assert 'TRAIN_PREFLIGHT_RETRIES="$PREFLIGHT_RETRIES"' in launcher
    assert 'TRAIN_PREFLIGHT_WAIT="$PREFLIGHT_WAIT"' in launcher


def test_every_profile_states_a_training_load_ceiling():
    for profile in ('real', 'rehearsal', 'test'):
        assert CP.PROFILES[profile]['train_load_ceiling'] > 0, profile
    assert CP.PROFILES['real']['train_load_ceiling'] == 20.0


def test_only_the_real_profile_names_the_preregistered_confirmatory_seeds():
    """The confirmatory block 200-219 must stay untouched until the real run.
    On 2026-09-24/25 the reduced rehearsal ran pipeline passes on seeds
    200-203 because its profile named them; this pins that no profile other
    than 'real' can do so again.  Counts what it checked, so it cannot pass
    vacuously on an empty profile table."""
    reg = set(range(200, 220))
    checked, offenders = 0, []
    for name, prof in CP.PROFILES.items():
        for key, val in prof.items():
            if isinstance(val, list) and val and all(isinstance(x, int) for x in val):
                checked += 1
                hit = sorted(set(val) & reg)
                if hit and name != 'real':
                    offenders.append((name, key, hit))
    assert checked >= 6, 'scanned only %d seed lists' % checked
    assert 'real' in CP.PROFILES and set(CP.PROFILES['real']['cf_seeds']) == reg
    assert not offenders, 'non-real profiles name pre-registered seeds: %r' % offenders


# --- the manifest each analysis step is checked against (tools/analysis_manifest.py)

def _am():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'analysis_manifest', os.path.join(ROOT, 'tools', 'analysis_manifest.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _manifest(path, seeds, kinds=('recovery', 'rescue_ops')):
    import json
    inst = [{'scenario_kind': k, 'topology_seed': s} for s in seeds for k in kinds]
    json.dump({'instances': inst}, open(path, 'w'))
    return path


def test_analysis_manifest_passes_the_real_profiles_manifests_unchanged():
    """The real profile's declared seeds equal its manifests, so the analysis
    must receive exactly the manifest paths it always did -- nothing derived."""
    am = _am()
    with _scratch('rc_manifest_') as d:
        dev = ['benchmark_manifest.json', 'benchmark_manifest_70_79.json']
        path, payload = am.resolve('real', 'dev', [os.path.join(ROOT, x) for x in dev],
                                   os.path.join(d, 'dev.json'))
        assert payload is None and path == ','.join(os.path.join(ROOT, x) for x in dev)
        assert not os.path.exists(os.path.join(d, 'dev.json'))
        cf = _manifest(os.path.join(d, 'cf.json'), range(200, 220))
        path, payload = am.resolve('real', 'primary', [cf], os.path.join(d, 'p.json'))
        assert payload is None and path == cf


def test_analysis_manifest_restricts_a_reduced_profile_to_its_declared_seeds():
    """The rehearsal analyses only its full-length subset; the derived manifest
    holds exactly those seeds, in every scenario kind of the source."""
    am = _am()
    with _scratch('rc_manifest_') as d:
        dev = [os.path.join(ROOT, x) for x in ('benchmark_manifest.json', 'benchmark_manifest_70_79.json')]
        path, payload = am.resolve('rehearsal', 'dev', dev, os.path.join(d, 'dev.json'))
        got = sorted({(i['scenario_kind'], int(i['topology_seed'])) for i in payload['instances']})
        kinds = sorted({i['scenario_kind'] for i in payload['instances']})
        assert len(kinds) == 2, kinds
        assert got == sorted((k, s) for k in kinds for s in (50, 51)), got
        assert payload['declared_seeds'] == [50, 51] and os.path.exists(path)
        cf = _manifest(os.path.join(d, 'cf.json'), (220, 221, 222, 223))
        path, payload = am.resolve('rehearsal', 'primary', [cf], os.path.join(d, 'p.json'))
        assert sorted({int(i['topology_seed']) for i in payload['instances']}) == [220, 221]


def test_analysis_manifest_refuses_a_declared_seed_the_manifest_lacks():
    """The expected set comes from the PLAN; a plan naming a seed the manifest
    does not hold is an error, never a silently smaller check."""
    am = _am()
    with _scratch('rc_manifest_') as d:
        cf = _manifest(os.path.join(d, 'cf.json'), (220, 222))
        try:
            am.resolve('rehearsal', 'primary', [cf], os.path.join(d, 'p.json'))
        except am.SystemExit2 as e:
            assert '221' in str(e), e
        else:
            raise AssertionError('a declared seed absent from the manifest was accepted')
        one_kind = _manifest(os.path.join(d, 'k.json'), (220, 221, 222, 223), kinds=('recovery',))
        path, payload = am.resolve('rehearsal', 'primary', [one_kind], os.path.join(d, 'k2.json'))
        assert {i['scenario_kind'] for i in payload['instances']} == {'recovery'}


def test_both_analysis_steps_take_the_plan_declared_manifest():
    """The S5b and S7 calls of confirmatory_analysis.py must go through the
    helper; the hard-coded development manifests that broke the merged tree's
    rehearsal must not come back."""
    src = open(os.path.join(ROOT, 'confirmatory', 'stage_analysis.sh')).read()
    # S5b, S7 and (FIX-PREFREEZE item 2) S12's full report
    assert src.count('tools/analysis_manifest.py') == 3, 'every step must use the helper'
    assert src.count('--manifests "$DEVMAN"') == 1
    assert src.count('--manifests "$CFMAN"') == 2, 'S7 and the S12 full report'
    calls = [l for l in src.splitlines()
             if ('confirmatory_analysis.py' in l or l.strip().startswith('ARGS=('))
             and '--manifests' in l and not l.strip().startswith('echo')]
    assert len(calls) == 3, calls
    assert not any('benchmark_manifest.json' in l for l in calls), calls
