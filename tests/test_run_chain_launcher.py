# -*- coding: utf-8 -*-
"""The launcher's BEHAVIOUR, exercised without compute.

Under the 'test' profile the stage scripts are replaced by tests/fake_stage.sh
(run_final.sh accepts CHAIN_SCRIPT_OVERRIDE for that profile and no other), so
the rules that matter at 3 a.m. can actually be run:

  * an invocation that produced an EMPTY pickle FAILS the gate -- the gate is
    not vacuous;
  * a FAILED invocation is re-run exactly once, UNCHANGED, and the second
    failure stops the chain;
  * an invocation that SUCCEEDED is never re-run, and a resumed chain redoes
    nothing else;
  * a successful invocation whose artefact later changes stops the chain
    instead of being quietly redone;
  * a job that exits 0 without writing anything FAILS even when a valid
    artefact from a previous attempt is lying there;
  * an artefact this attempt did not write FAILS the freshness gate;
  * a hard dependency is skipped only by an explicit, recorded waiver;
  * an ablation identical to its baseline is recorded NOT EXERCISED, never as
    a pass;
  * shedding out of the plan's order is refused.

These tests need bash and /proc (the preflight reads MemAvailable), so they
skip on anything that is not Linux.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _mkdtemp(prefix):
    """Scratch directory inside the tree (output/tmp_tests; the seed audit
    skips it by design), never the system default: tempfile.gettempdir()
    falls back to the current directory -- the tree root, where the suite
    runs -- when `/` is full (D12F.2).  Every caller removes it."""
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
PROFILE = 'test'
LEDGER = os.path.join(ROOT, 'confirmatory', 'chain_%s.jsonl' % PROFILE)
FAKE = os.path.join(ROOT, 'logs', 'fake')


def _skip_unless_linux():
    if not sys.platform.startswith('linux') or shutil.which('bash') is None:
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips (tests/test_wp6_parity.py): ' 'the launcher tests need bash and /proc')


# Directories the 'test' profile owns outright, and the confirmatory/ files
# it writes.  These artefacts are synthetic -- fake_stage.sh writes a tiny
# pickle (fake provenance, empty arms, a 'seeds' label list) to EVERY artifact
# path, even one named .json -- and they used to be left in the production
# tree, where the seed audit cannot verify a binary file named .json and
# refused a verdict (2026-09-30).  The old test_manifest_200_201.json also
# carried the pre-registered seeds as a label, from before the test profile
# moved to 220-221 (31b034c).  No tracked file in confirmatory/ starts with
# 'test_' or ends in '_test.<ext>', so the rule cannot touch a real file.
_TEST_DIRS = ('output/test', 'logs/chain_test', 'logs/fake',
              'confirmatory/env_test', 'confirmatory/preflight_test')
_TEST_FILES = ('confirmatory/chain_test.jsonl', 'confirmatory/FREEZE_FINAL_test.txt',
               'confirmatory/SELECTED_test.txt', 'logs/chain_test.log')


def _is_test_profile_artefact(rel):
    rel = rel.replace(os.sep, '/')
    if any(rel == d or rel.startswith(d + '/') for d in _TEST_DIRS):
        return True
    if rel in _TEST_FILES:
        return True
    return rel.startswith('confirmatory/test_')


def _cleanup():
    for d in _TEST_DIRS:
        shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)
    for f in _TEST_FILES:
        fp = os.path.join(ROOT, f)
        if os.path.exists(fp):
            os.remove(fp)
    conf = os.path.join(ROOT, 'confirmatory')
    for n in (os.listdir(conf) if os.path.isdir(conf) else []):
        fp = os.path.join(conf, n)
        if n.startswith('test_') and os.path.isfile(fp):
            os.remove(fp)


# Clean up AFTER the last test as well as before each one, so the 'test'
# profile never leaves fixtures in the production tree.
import atexit  # noqa: E402
atexit.register(_cleanup)


def _reset():
    _cleanup()
    os.makedirs(FAKE, exist_ok=True)
    # the S1 placeholder checkpoint the driver would demand
    ck = os.path.join(ROOT, 'output', 'test', 'untrained_s1234')
    os.makedirs(ck, exist_ok=True)
    open(os.path.join(ck, 'policy_init.pt'), 'wb').write(b'not a real checkpoint')


def _script(inv, *directives):
    with open(os.path.join(FAKE, inv + '.script'), 'w') as f:
        f.write('\n'.join(directives) + '\n')


def _run(*args, **kw):
    env = dict(os.environ)
    env.update({'CHAIN_SCRIPT_OVERRIDE': 'tests/fake_stage.sh',
                'CHAIN_FAKE_DIR': 'logs/fake',
                'CHAIN_NO_WATCHDOG': '1',
                'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    env.update(kw.pop('env', {}))
    cmd = ['bash', 'confirmatory/run_final.sh', '--profile', PROFILE,
           '--no-watchdog', '--preflight-retries', '1', '--preflight-wait', '1'] + list(args)
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def _ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def _ends(inv):
    return [r for r in _ledger() if r.get('inv') == inv and r.get('event') == 'end']


def test_dry_run_lists_every_stage_in_dependency_order():
    _skip_unless_linux()
    _reset()
    r = _run('--dry-run')
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    order = [l.split('STAGE', 1)[1].split()[0]
             for l in r.stdout.splitlines() if '---- STAGE' in l]
    import sys as _s
    _s.path.insert(0, os.path.join(ROOT, 'confirmatory'))
    import chain_plan
    assert order == chain_plan.STAGE_ORDER, order


def test_empty_pickle_fails_the_gate_and_is_re_run_once_then_stops():
    _skip_unless_linux()
    _reset()
    # every attempt produces a pickle with no passes at all
    _script('s1_dev_b0', 'empty', 'empty')
    r = _run('--stages', 'S0 S1')
    assert r.returncode != 0, 'an empty pickle must not pass the gate'
    ends = _ends('s1_dev_b0')
    assert len(ends) == 2, 'exactly one unchanged re-run: %s' % ends
    assert all(e['status'] == 'fail' for e in ends), ends
    assert all(e['exit'] == 0 for e in ends), 'the process exited 0; the GATE caught it'
    assert any(r.get('reason') == 'failed_twice' for r in _ledger()
               if r.get('event') == 'stop')
    assert 'do NOT change arms or seeds' in r.stdout


def test_technical_failure_is_re_run_unchanged_and_then_succeeds():
    _skip_unless_linux()
    _reset()
    _script('s1_dev_b0', 'fail', 'ok')
    r = _run('--stages', 'S0 S1')
    ends = _ends('s1_dev_b0')
    assert len(ends) == 2, ends
    assert ends[0]['status'] == 'fail' and ends[1]['status'] == 'ok', ends
    starts = [x for x in _ledger() if x.get('inv') == 's1_dev_b0' and x.get('event') == 'start']
    assert len({s['cmd'] for s in starts}) == 1, 'the re-run must be UNCHANGED'
    assert len({s['envfile'] for s in starts}) == 1, 'same env block'
    assert ends[1]['passes'] == 56 or ends[1]['passes'] > 0, ends


def test_resume_skips_what_succeeded_and_redoes_nothing_else():
    _skip_unless_linux()
    _reset()
    r1 = _run('--stages', 'S0 S1')
    assert r1.returncode == 0, r1.stdout[-3000:]
    first = [e['inv'] for e in _ledger() if e.get('event') == 'end']
    r2 = _run('--stages', 'S0 S1')
    assert r2.returncode == 0, r2.stdout[-3000:]
    skipped = [e['inv'] for e in _ledger() if e.get('event') == 'skip']
    assert sorted(set(skipped)) == sorted(set(first)), (skipped, first)
    ends = [e for e in _ledger() if e.get('event') == 'end']
    assert len(ends) == len(first), 'the second pass must not re-run anything'
    assert 'RESUME' in r2.stdout


def test_a_changed_artifact_stops_the_chain_instead_of_being_redone():
    _skip_unless_linux()
    _reset()
    assert _run('--stages', 'S0 S1').returncode == 0
    art = [e['artifact'] for e in _ledger() if e.get('event') == 'end'
           and e['artifact'].endswith('.pkl')][0]
    with open(os.path.join(ROOT, art), 'ab') as f:
        f.write(b'tampered')
    r = _run('--stages', 'S1')
    assert r.returncode != 0
    assert any(x.get('reason') == 'artifact_changed' for x in _ledger()
               if x.get('event') == 'stop'), _ledger()[-3:]


def test_shedding_out_of_order_is_refused():
    _skip_unless_linux()
    _reset()
    bad = _run('--shed', 'untrained', '--dry-run')
    assert bad.returncode != 0
    assert 'REFUSING' in (bad.stdout + bad.stderr)
    good = _run('--shed', 'sensitivity,adapt_freeze,untrained', '--dry-run')
    assert good.returncode == 0, good.stdout[-2000:]
    assert 'SHED  S10' in good.stdout and 'SHED  S8' in good.stdout


def test_a_stage_does_not_start_before_its_dependencies():
    _skip_unless_linux()
    _reset()
    r = _run('--stages', 'S6')       # S5/S3b never ran
    assert r.returncode != 0
    assert 'hard dependency' in r.stdout
    assert any(x.get('reason') == 'dependency_incomplete' for x in _ledger()
               if x.get('event') == 'stop')


def test_training_may_not_be_scheduled_before_the_front_loaded_stage():
    _skip_unless_linux()
    _reset()
    # S0 complete, S1 not: S2 must refuse (ordering), and say how to override
    r = _run('--stages', 'S0 S2')
    assert r.returncode != 0
    assert '--waive-order S2' in r.stdout
    assert any(x.get('reason') == 'order_violated' for x in _ledger()
               if x.get('event') == 'stop')


def test_a_stage_that_exits_0_without_writing_fails_even_with_a_stale_artefact():
    """THE campaign defect, in the launcher itself.

    Run 1 leaves a valid pickle.  The operator invalidates the invocation, and
    run 2's job exits 0 having written nothing.  The completeness gate used to
    load run 1's pickle, count 28 present / 0 missing, and record a full
    success with run 1's sha256 -- a check that reported success without
    having verified anything.  The launcher now removes the target before the
    launch AND requires an mtime at or after it.
    """
    _skip_unless_linux()
    _reset()
    assert _run('--stages', 'S0 S1').returncode == 0
    inv = 's1_dev_b0'
    art = os.path.join(ROOT, [e['artifact'] for e in _ends(inv)][-1])
    assert os.path.exists(art), art
    before = os.path.getmtime(art)

    assert _run('--invalidate', inv).returncode == 0
    _script(inv, 'nofile', 'nofile')
    r = _run('--stages', 'S1')
    assert r.returncode != 0, 'a job that wrote nothing must not pass'
    ends = _ends(inv)[-2:]
    assert len(ends) == 2, ends
    assert all(e['status'] == 'fail' for e in ends), ends
    assert all(e['exit'] == 0 for e in ends), 'the process exited 0; the GATE caught it'
    assert all(e['passes'] == 0 for e in ends), ends
    assert not os.path.exists(art), 'the stale artefact must have been removed'
    assert 'does not exist' in r.stdout, r.stdout
    assert before  # the pre-existing artefact really was there


def test_a_back_dated_artefact_fails_the_freshness_gate():
    """The second half of the same rule: an artefact that IS present and whose
    key count is complete, but which this attempt did not write."""
    _skip_unless_linux()
    _reset()
    _script('s1_dev_b0', 'stale', 'stale')
    r = _run('--stages', 'S0 S1')
    assert r.returncode != 0, 'a back-dated artefact must not pass the gate'
    assert 'was NOT written by this attempt' in r.stdout, r.stdout[-3000:]
    ends = _ends('s1_dev_b0')
    assert len(ends) == 2 and all(e['status'] == 'fail' for e in ends), ends
    assert all(e['gate'] == 1 and e['exit'] == 0 for e in ends), ends


def test_a_hard_dependency_is_skipped_only_by_an_explicit_recorded_waiver():
    """A stage can be unrunnable for a reason outside this track (S5b runs the
    pre-registered analysis, which cannot execute below 1400 ticks).  The
    alternative to a recorded waiver is an operator editing the ledger."""
    _skip_unless_linux()
    _reset()
    # the selection S6 needs; the dependency check is what is under test here
    sel = os.path.join(ROOT, 'confirmatory', 'SELECTED_%s.txt' % PROFILE)
    with open(sel, 'w') as f:
        f.write('output/test/untrained_s1234/policy_init.pt' + chr(10))

    bad = _run('--stages', 'S6')
    assert bad.returncode != 0
    assert '--waive-dep S6' in bad.stdout, bad.stdout[-2000:]
    assert any(x.get('reason') == 'dependency_incomplete'
               for x in _ledger() if x.get('event') == 'stop')

    good = _run('--stages', 'S6', '--waive-dep', 'S6')
    assert good.returncode == 0, good.stdout[-3000:]
    notes = [x for x in _ledger()
             if x.get('event') == 'note' and x.get('reason') == 'dep_waived']
    assert sorted({n['dep'] for n in notes}) == ['S3b', 'S5b'], notes
    assert 'UNPROVEN' in good.stdout
    ends = _ends('s6_cf_b0')
    assert ends and ends[-1]['status'] == 'ok', ends
    os.remove(sel)


def test_the_ablation_gate_runs_inside_the_chain():
    """S11 is an ablation of S6.  fake_stage writes both with the same series,
    so the ablation is identical to its baseline and the declared counter
    (repoints) is absent -- the chain must record NOT EXERCISED and must not
    report the sensitivity as tested."""
    _skip_unless_linux()
    _reset()
    sel = os.path.join(ROOT, 'confirmatory', 'SELECTED_%s.txt' % PROFILE)
    with open(sel, 'w') as f:
        f.write('output/test/untrained_s1234/policy_init.pt' + chr(10))
    r = _run('--stages', 'S6 S11', '--waive-dep', 'S6,S11',
             '--waive-order', 'S11')
    assert r.returncode == 0, r.stdout[-4000:]
    assert '[ABLATION]' in r.stdout, r.stdout[-3000:]
    end = _ends('s11_sens_repoint')[-1]
    assert end['ablation'] in ('NOT_EXERCISED', 'exercised'), end
    assert end['ablation'] == 'NOT_EXERCISED', (
        'identical output with a zero baseline counter is NOT EXERCISED: %r' % end)
    assert 'must not be reported as' in r.stdout
    os.remove(sel)


def test_cleanup_covers_every_artifact_the_test_profile_can_write():
    """Driven by the PLAN, not by a hand list: every artifact path the 'test'
    profile declares, plus the ledger and its env/preflight directories, must
    be covered by the cleanup rule.  When the plan grows a new output location
    this fails until the cleanup covers it -- the hand-written list it replaces
    missed three report files.  Also proves the rule spares every tracked file."""
    sys.path.insert(0, os.path.join(ROOT, 'confirmatory'))
    import chain_plan as CP
    arts = sorted({i['artifact'] for i in CP.build('test') if i.get('artifact')})
    assert len(arts) >= 10, 'the plan declared only %d test artifacts' % len(arts)
    extra = ['confirmatory/chain_test.jsonl', 'confirmatory/env_test/x.sh',
             'confirmatory/preflight_test/x.json']
    uncovered = [a for a in arts + extra if not _is_test_profile_artefact(a)]
    assert not uncovered, 'cleanup does not cover: %r' % uncovered
    # the tracked-file list: git on the laptop worktree, .synced_files on the
    # server (which has no history).  One of the two must exist, so this check
    # cannot silently not run where the suite actually runs.
    tracked = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True,
                             text=True).stdout.split()
    synced = os.path.join(ROOT, '.synced_files')
    if not tracked and os.path.exists(synced):
        tracked = open(synced).read().split()
    assert len(tracked) >= 50, 'no tracked-file list to check against (%d)' % len(tracked)
    hit = [t for t in tracked if _is_test_profile_artefact(t)]
    assert not hit, 'the cleanup rule would delete tracked files: %r' % hit


# ------------------------------------- the environment of an invocation (item 5)
_STRAY = {'MARL_STRAY_PROBE': 'leaked', 'MARL_DETECT_TICKS': '0',
          'UU_CA_MAX_EXTRA_CC': '3', 'PROBE_TICK': '7'}


def _marl_file(inv):
    out = {}
    for ln in open(os.path.join(ROOT, 'confirmatory', 'env_%s' % PROFILE,
                                '%s.marl_env' % inv)):
        if ln.strip():
            k, _, v = ln.strip().partition('=')
            out[k] = v
    return out


def test_a_stray_variable_of_the_operators_shell_never_reaches_an_invocation():
    """FIX-PREFREEZE item 5.  The launcher is started from a shell holding a
    stray MARL_* variable, the omniscient-ablation switch and two other
    variables the code reads.  Every invocation's job script re-executes
    under `env -i` and sources only its env file, so what each one RECORDS
    (the stand-in stage writes the real os.environ, as runinfo.provenance
    does) is exactly the plan's MARL_* set and none of the stray names; the
    preflight's job-environment check ran and passed for every invocation,
    and the completeness gate compared the recorded set."""
    _skip_unless_linux()
    _reset()
    import pickle
    sys.path.insert(0, os.path.join(ROOT, 'confirmatory'))
    import chain_plan as CP
    r = _run('--stages', 'S0 S1', env=dict(_STRAY))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    invs = [i for i in CP.build(PROFILE) if i['stage'] in ('S0', 'S1')]
    checks = [x for x in _ledger() if x.get('event') == 'envcheck']
    assert sorted({c['inv'] for c in checks}) == sorted(i['id'] for i in invs), checks
    assert all(c['rc'] == 0 for c in checks), checks
    assert '[PREFLIGHT-ENV] PASS' in r.stdout
    n = 0
    for i in invs:
        prov = pickle.load(open(os.path.join(ROOT, i['artifact']), 'rb'))['provenance']
        want = _marl_file(i['id'])
        assert want and prov['marl_env'] == want, (i['id'], prov['marl_env'], want)
        assert set(want) >= {'MARL_CHAIN_INV', 'MARL_POOL_WORKERS'}
        for k, v in i['env'].items():
            assert want[k] == v, (i['id'], k)
        assert not set(_STRAY) & set(prov['env_names']), (i['id'], prov['env_names'])
        assert prov['pythonhashseed'] == '0'
        n += 1
    assert n == len(invs) == 4, n
    # the gate's own line: the recorded set was compared, not assumed
    assert r.stdout.count('marl_env: ') >= 2 and 'problem(s)' in r.stdout


def _job_with(envstr, inv='probe'):
    """An env file + job script written by the REAL chain_write_env /
    chain_write_job into a scratch directory; returns (dir, job, marl_file)."""
    d = _mkdtemp('rcl_jobenv_')
    # separate directories, as in the launcher (confirmatory/env_<p>/ and
    # logs/chain_<p>/): both files are named <inv>.sh
    os.makedirs(os.path.join(d, 'env'))
    os.makedirs(os.path.join(d, 'job'))
    out = subprocess.run(
        ['bash', '-c',
         'set -u; . confirmatory/chain_lib.sh; CHAIN_PROFILE=test; '
         'CHAIN_ENVDIR="$1/env"; CHAIN_JOBDIR="$1/job"; '
         'chain_write_env "$2" "$3" "" 1 0; chain_write_job "$2" DONE true; '
         'echo "$CHAIN_JOB"; echo "$CHAIN_MARLENV"', '_', d, inv, envstr],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    job, marl = out.stdout.split()
    return d, job, marl


def _job_env_check(job, marl, hashseed='0', env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, 'tools', 'preflight.py'), '--job-env', job,
         '--expect-marl-file', marl, '--expect-hashseed', hashseed, '--inv', 'probe'],
        cwd=ROOT, env=e, capture_output=True, text=True)


def test_the_job_environment_check_refuses_a_leaking_or_wrong_environment():
    """The preflight's --job-env mode runs the job script with --print-env
    from the CALLER's environment.  The launcher's own job passes with stray
    variables in the caller's shell; a job without the `env -i` re-execution
    leaks them and FAILS; so do a wrong hash seed, a MARL_* variable the plan
    sets but the job does not, and an empty expected set."""
    _skip_unless_linux()
    d, job, marl = _job_with('MARL_A=1 MARL_B=two')
    try:
        ok = _job_env_check(job, marl, env=_STRAY)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert '[PREFLIGHT-ENV] PASS probe checks=4 failed=0' in ok.stdout, ok.stdout
        # a job that does not clear the caller's environment
        src = open(job).read()
        assert src.count('exec /usr/bin/env -i') == 1, src
        leaky = os.path.join(d, 'leaky.sh')
        open(leaky, 'w').write('\n'.join(l for l in src.splitlines()
                                          if 'exec /usr/bin/env -i' not in l) + '\n')
        bad = _job_env_check(leaky, marl, env=_STRAY)
        assert bad.returncode == 1, bad.stdout
        assert 'MARL_STRAY_PROBE' in bad.stdout and 'UU_CA_MAX_EXTRA_CC' in bad.stdout, bad.stdout
        # the plan's hash seed for this invocation is not what the job sets
        bad = _job_env_check(job, marl, hashseed='1')
        assert bad.returncode == 1 and 'pythonhashseed' in bad.stdout, bad.stdout
        # the plan sets a variable the job does not
        extra = os.path.join(d, 'extra.marl_env')
        open(extra, 'w').write(open(marl).read() + 'MARL_C=3\n')
        bad = _job_env_check(job, extra)
        assert bad.returncode == 1 and 'MARL_C' in bad.stdout, bad.stdout
        # nothing expected: a check against nothing is a failure
        empty = os.path.join(d, 'empty.marl_env')
        open(empty, 'w').close()
        bad = _job_env_check(job, empty)
        assert bad.returncode == 1, bad.stdout
    finally:
        _rmtree(d)


def test_the_completeness_gate_requires_the_recorded_marl_env_to_be_the_plans():
    """`stage_report.py --verify --expect-marl-env-file` (and
    `--check-marl-env` for a checkpoint): the run's RECORDED MARL_* set must
    equal the plan's -- a stray variable, a missing one, a different value,
    no record at all, or an empty plan set is a FAILURE."""
    import pickle
    d = _mkdtemp('rcl_marlenv_')
    try:
        plan = {'MARL_A': '1', 'MARL_CHAIN_INV': 'x', 'MARL_POOL_WORKERS': '4'}
        mf = os.path.join(d, 'x.marl_env')
        open(mf, 'w').write(''.join('%s=%s\n' % kv for kv in sorted(plan.items())))
        keys = os.path.join(d, 'x.keys')
        open(keys, 'w').write('marl_static|recovery|220\nmarl_static|rescue_ops|220\n')

        def pk(name, marl_env):
            prov = {'tree_head': 'h', 'tree_sha256': 'f' * 64, 'schema_id': 's',
                    'pythonhashseed': '0'}
            if marl_env is not None:
                prov['marl_env'] = marl_env
            rec = {'recovery_multi': {'marl_static': {220: {'conn': [1.0]}}},
                   'rescue_multi': {'marl_static': {220: {'conn': [1.0]}}},
                   'provenance': prov, 'pass_stats': {}}
            p = os.path.join(d, name)
            pickle.dump(rec, open(p, 'wb'))
            return p

        def verify(p, marl_file=mf):
            return subprocess.run(
                [sys.executable, os.path.join(ROOT, 'tools', 'stage_report.py'),
                 '--verify', p, '--keys-file', keys, '--expect-marl-env-file', marl_file],
                capture_output=True, text=True)

        def check(p, marl_file=mf):
            return subprocess.run(
                [sys.executable, os.path.join(ROOT, 'tools', 'stage_report.py'),
                 '--check-marl-env', p, '--expect-marl-env-file', marl_file],
                capture_output=True, text=True)

        n = 0
        good = pk('good.pkl', dict(plan))
        for run in (verify, check):
            r = run(good)
            assert r.returncode == 0, r.stdout
            assert '3 variable(s) compared with the plan, 0 problem(s)' in r.stdout, r.stdout
            n += 1
        for name, env, word in (('stray.pkl', dict(plan, MARL_DETECT_TICKS='0'), 'STRAY'),
                                ('missing.pkl', {'MARL_A': '1', 'MARL_CHAIN_INV': 'x'}, 'MISSING'),
                                ('different.pkl', dict(plan, MARL_A='2'), 'DIFFERENT'),
                                ('none.pkl', None, 'no marl_env')):
            p = pk(name, env)
            for run in (verify, check):
                r = run(p)
                assert r.returncode == 1 and word in r.stdout, (name, r.stdout)
                n += 1
        empty = os.path.join(d, 'empty.marl_env')
        open(empty, 'w').close()
        for run in (verify, check):
            r = run(good, empty)
            assert r.returncode == 1 and 'EMPTY' in r.stdout, r.stdout
            n += 1
        assert n == 12
    finally:
        _rmtree(d)
