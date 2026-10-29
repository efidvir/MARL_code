# -*- coding: utf-8 -*-
"""Tree hygiene of the test suite and of the negative-control runners
(D12-FIXES item FIRST, docs/DECISIONS.md D12F.2).

Two defects left files in ~/MARL_run_9 that the seed audit then had to
explain:

* eleven `tmp*` directories in the tree ROOT, some holding 3-byte stub
  pickles laid out like the real profile's `output/final/s6_cf_b*.pkl`.
  Tests called `tempfile.mkdtemp()` with no `dir=`; `tempfile.gettempdir()`
  falls back to the CURRENT DIRECTORY when every candidate temp directory
  refuses a write (`/` was full on 2026-10-01), and the suite runs from the
  tree root.  Several of those tests never removed their directory either.
* seven copies under `output/prefreeze_negctl/`, left by a negative-control
  run that was killed by a signal: SIGTERM and SIGHUP end a Python process
  without running its `finally`.

This file is named to run LAST in `tests/run_tests.py` (files and functions
run in sorted order), so its final test checks the tree root after the
whole suite.  Every check counts what it checked and fails on zero.
"""
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TMP_TESTS = os.path.join(ROOT, 'output', 'tmp_tests')


def _linux_or_fail(what):
    if not sys.platform.startswith('linux'):
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run '
                             'FAILS, it never skips (tests/test_wp6_parity.py): '
                             + what)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the negative-control runner: signals and stale copies ──────────────────
_SELFTEST = r'''
import os, sys, time
sys.path.insert(0, sys.argv[1])
import wp13_negative_controls as NC
marker, scratch, mode = sys.argv[2], sys.argv[3], sys.argv[4]

def fake_copy(d):
    with open(os.path.join(d, 'copied.txt'), 'w') as fh:
        fh.write('x')
    return 1

def fake_run_all(d, plan):
    if mode == 'hang':
        with open(marker, 'w') as fh:
            fh.write(d)
        time.sleep(120)
    return {}

NC._copy_tree = fake_copy
NC._run_all = fake_run_all
tests = {'tests/x.py': ('test_a',)}
table = {'ctl': ('x.py', 'a', 'b', {'tests/x.py': ('test_a',)}, 'self-test')}
sys.exit(NC.main(['ctl'], tests=tests, table=table, title='SELFTEST',
                 scratch=scratch))
'''


def _selftest(scratch, mode, marker):
    return subprocess.Popen(
        [sys.executable, '-c', _SELFTEST, os.path.join(ROOT, 'tools'), marker,
         scratch, mode],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_negctl_runner_removes_its_copies_when_it_is_killed():
    """A run ended by SIGTERM or SIGHUP while a control is running removes
    its scratch copies and exits 128 + signum.  Before D12F.2 both signals
    killed the process without unwinding: the copy stayed, exit -signum."""
    _linux_or_fail('POSIX signals')
    checked = 0
    for sig in (signal.SIGTERM, signal.SIGHUP):
        scratch = 'negctl_selftest_%d_%d' % (os.getpid(), int(sig))
        base = os.path.join(ROOT, 'output', scratch)
        os.makedirs(TMP_TESTS, exist_ok=True)
        marker = os.path.join(TMP_TESTS, scratch + '.started')
        if os.path.exists(marker):
            os.remove(marker)
        p = _selftest(scratch, 'hang', marker)
        try:
            t0 = time.time()
            while not os.path.exists(marker) and time.time() - t0 < 60:
                if p.poll() is not None:
                    break
                time.sleep(0.2)
            assert os.path.exists(marker), ('the runner never reached its '
                                            'first control', p.poll())
            copy = open(marker).read().strip()
            assert os.path.isdir(copy) and copy.startswith(base), copy
            assert os.path.exists(os.path.join(copy, '.negctl_owner')), copy
            p.send_signal(sig)
            out, _ = p.communicate(timeout=60)
            assert p.returncode == 128 + int(sig), (sig, p.returncode, out[-2000:])
            assert not os.path.exists(copy), ('copy left after %s' % sig.name,
                                              copy, out[-2000:])
            left = os.listdir(base) if os.path.isdir(base) else []
            assert left == [], (sig.name, left)
            assert 'scratch copy not removed' not in out, out[-2000:]
            checked += 1
        finally:
            if p.poll() is None:
                p.kill()
                p.communicate()
            shutil.rmtree(base, ignore_errors=True)
            if os.path.exists(marker):
                os.remove(marker)
    assert checked == 2, checked


def test_negctl_runner_sweeps_only_the_copies_of_a_dead_runner():
    """At start the runner removes the copies whose owner process is gone
    (a SIGKILL cannot be caught), keeps a live runner's copies and copies
    with no owner record, and reports all three counts."""
    _linux_or_fail('POSIX processes')
    scratch = 'negctl_sweeptest_%d' % os.getpid()
    base = os.path.join(ROOT, 'output', scratch)
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)
    host = socket.gethostname()
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    planted = {}
    for name, owner in (('stale_x', '%d %s\n' % (dead.pid, host)),
                        ('live_x', '%d %s\n' % (os.getpid(), host)),
                        ('legacy_x', None)):
        d = os.path.join(base, name)
        os.makedirs(d)
        with open(os.path.join(d, 'f.txt'), 'w') as fh:
            fh.write('x')
        if owner:
            with open(os.path.join(d, '.negctl_owner'), 'w') as fh:
                fh.write(owner)
        planted[name] = d
    try:
        p = _selftest(scratch, 'empty', os.path.join(TMP_TESTS, 'unused'))
        out, _ = p.communicate(timeout=120)
        line = [l for l in out.splitlines() if l.startswith('# scratch sweep')]
        assert len(line) == 1, out[-2000:]
        assert ('1 stale copies of a dead runner removed, 1 of a live runner '
                'kept, 1 without an owner record kept') in line[0], line
        assert not os.path.exists(planted['stale_x']), 'stale copy kept'
        assert os.path.isdir(planted['live_x']), 'a live runner copy was removed'
        assert os.path.isdir(planted['legacy_x']), 'an unowned copy was removed'
        # the head of this fake run is not green: the run itself fails, and
        # its own head copy is removed all the same
        assert p.returncode == 1, (p.returncode, out[-2000:])
        assert sorted(os.listdir(base)) == ['legacy_x', 'live_x'], os.listdir(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ── the suite's own scratch directories ────────────────────────────────────
# The tests that called tempfile.mkdtemp() / TemporaryDirectory() with no
# dir= before D12F.2 (those that run in a few seconds), by file.
CONVERTED = {
    'tests/test_run_chain.py': (
        'test_s10_runs_installation_off_not_the_omniscient_ablation',
        'test_an_ablation_that_changed_nothing_is_not_recorded_as_a_pass',
        'test_the_ablation_gate_fails_when_it_compared_nothing',
        'test_the_tree_pin_comes_from_the_first_invocation_of_the_ledger',
        'test_the_timing_report_scopes_to_one_chain_run',
        'test_the_primary_step_analyses_exactly_the_profiles_confirmatory_input',
        'test_the_load_ceiling_only_applies_when_a_wave_starts',
        # (the three analysis_manifest tests use _scratch too, but read the
        # tree's benchmark_manifest*.json, which a negative-control copy
        # does not hold; they run in the suite as before)
    ),
    'tests/test_run_chain_launcher.py': (
        'test_the_job_environment_check_refuses_a_leaking_or_wrong_environment',
        'test_the_completeness_gate_requires_the_recorded_marl_env_to_be_the_plans',
    ),
    'tests/test_wp13_integration.py': (
        'test_the_combiner_removes_only_what_is_constant_in_every_pass',
        'test_the_combiner_refuses_an_empty_a_short_or_a_mixed_set',
    ),
}


def test_scratch_directories_stay_in_tmp_tests_and_are_removed():
    """With tempfile's default directory pointed at a sentinel -- what
    gettempdir() returns on a full `/`, where it falls back to the current
    directory, the tree root -- the converted tests create NOTHING there:
    every scratch directory is made under output/tmp_tests and is gone when
    its test ends.  Counts the tests run and the directories created."""
    _linux_or_fail('the converted tests need bash and /proc')
    os.makedirs(TMP_TESTS, exist_ok=True)
    sentinel = tempfile.mkdtemp(prefix='hygiene_sentinel_', dir=TMP_TESTS)
    created = []
    orig_mkdtemp = tempfile.mkdtemp
    orig_tempdir = tempfile.tempdir

    def recording_mkdtemp(*a, **k):
        d = orig_mkdtemp(*a, **k)
        created.append(os.path.abspath(d))
        return d
    ran = 0
    problems = []
    try:
        tempfile.mkdtemp = recording_mkdtemp
        tempfile.tempdir = sentinel
        for i, (rel, names) in enumerate(sorted(CONVERTED.items())):
            mod = _load(rel, 'hygiene_mod_%d' % i)
            for name in names:
                before = len(created)
                getattr(mod, name)()
                ran += 1
                mine = created[before:]
                if not mine:
                    problems.append('%s::%s created no scratch directory'
                                    % (rel, name))
                for d in mine:
                    # directly in output/tmp_tests: the sentinel is itself
                    # a directory there, so a prefix test would pass a
                    # directory made in the sentinel
                    if os.path.dirname(d) != os.path.abspath(TMP_TESTS):
                        problems.append('%s::%s made %s outside output/tmp_tests'
                                        % (rel, name, d))
                    if os.path.exists(d):
                        problems.append('%s::%s left %s' % (rel, name, d))
    finally:
        tempfile.mkdtemp = orig_mkdtemp
        tempfile.tempdir = orig_tempdir
        in_sentinel = sorted(os.listdir(sentinel))
        shutil.rmtree(sentinel, ignore_errors=True)
    if in_sentinel:
        problems.append('created in the default temp directory: %s' % in_sentinel)
    n_expected = sum(len(v) for v in CONVERTED.values())
    assert ran == n_expected and ran > 0, (ran, n_expected)
    assert len(created) >= ran, (len(created), ran)
    assert not problems, '\n'.join(problems)


# ── the tree root after the suite ───────────────────────────────────────────
def _tmp_dirs_in(root):
    """(entries checked, tmp* directories found) in `root` itself."""
    names = sorted(os.listdir(root))
    found = [n for n in names
             if n.startswith('tmp') and os.path.isdir(os.path.join(root, n))]
    return len(names), found


def test_the_root_guard_finds_a_planted_tmp_directory():
    """The guard below on a planted root: it reports exactly the planted
    tmp* directory (not a tmp* FILE, not output/tmp_tests one level down)
    and counts every entry it looked at."""
    os.makedirs(TMP_TESTS, exist_ok=True)
    fake = tempfile.mkdtemp(prefix='hygiene_root_', dir=TMP_TESTS)
    try:
        for d in ('sixg_sim', 'tests', 'tmpab12cd', os.path.join('output', 'tmp_tests')):
            os.makedirs(os.path.join(fake, d))
        open(os.path.join(fake, 'tmpfile.txt'), 'w').close()
        n, found = _tmp_dirs_in(fake)
        assert n == 5, n
        assert found == ['tmpab12cd'], found
        os.rmdir(os.path.join(fake, 'tmpab12cd'))
        n, found = _tmp_dirs_in(fake)
        assert (n, found) == (4, []), (n, found)
    finally:
        shutil.rmtree(fake, ignore_errors=True)


def test_zz_no_tmp_directory_is_left_in_the_tree_root():
    """LAST test of the suite: no `tmp*` directory in the tree root.  It
    counts the entries it checked and requires the tree's own directories
    among them, so it cannot pass by looking at an empty or wrong place."""
    n, found = _tmp_dirs_in(ROOT)
    names = set(os.listdir(ROOT))
    assert n > 0 and {'sixg_sim', 'tests', 'tools'} <= names, (n, sorted(names)[:20])
    assert not found, ('tmp* directories in the tree root after the suite '
                       '(%d entries checked): %s' % (n, found))
    print('tree root: %d entries checked, 0 tmp* directories' % n)
