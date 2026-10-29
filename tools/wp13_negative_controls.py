#!/usr/bin/env python3
"""Negative controls for the WP13 behaviour fixes and the open WP5 record.

    python3 tools/wp13_negative_controls.py [--keep] [--list] [MUTATION ...]

The WP4 tool's method (tools/wp4_negative_controls.py), for tests in several
files: for every mutation, copy the tree's Python and config sources into a
scratch directory under output/wp13_negctl/ (never outside the tree),
re-introduce ONE defect -- the behaviour before the fix -- by an exact
string replacement whose anchor must occur EXACTLY ONCE, run the named tests
against the copy in a fresh interpreter (one child per test file, so a
file's own environment block applies before its first sixg_sim import), and
report which fail.  The unmutated copy runs first and must be fully green, so
a failure under a mutation is attributable to that mutation alone.

A control is CAUGHT when every test it lists under `must_fail` fails.  The
run exits 0 only when the head is green, at least one control ran, every
mutation applied (anchor found exactly once) and every control was caught.
Every line of the report counts what it checked.

The controls (docs/DECISIONS.md D13.x):

  per_transition_lsa      D13.1  revert Router-LSA origination to ONE LSA
                                 PER PAIR TRANSITION (serialised at
                                 MIN_LS_INTERVAL): the k-failure and the
                                 MinLSInterval-window tests must fail.
  global_action_lock      D13.2  revert the agents' action lock to the
                                 GLOBAL island trigger: the per-site lock
                                 test (a site whose own flag is not set keeps
                                 the pre-event action) must fail.
  heartbeat_no_lower_bound D5.19 revert the lower bound of the heartbeat
                                 due-test (D5.13): a stamp from a LATER tick
                                 silences the site again.
  drain_legacy_accepted   D5.19  accept MARL_POSTCARD_DRAIN=legacy again
                                 (D5.14).
  own_drop_constant       D13.5  wire b_own_sinr_drop to a constant: the
                                 now-unconditional seam (c) assertion fails
                                 (under the old conditional it passed).
  short_pass_removal      D13.3  remove g_pending_iops too, as the 600-tick
                                 passes of the merge suggested (it MOVES in
                                 Scenario B at production length): the
                                 evidence tests must fail.
  rehearsal_s7_reduced    D13.6  point the rehearsal's S7 at the reduced
                                 600-tick S6 again: the plan test and the
                                 stage-script test must fail.
  ablation_mixed_ticks    D13.6  compare an ablation with EVERY S6 pickle,
                                 the full-length subset included: the
                                 launcher test must fail.

A mutation is one edit (file, anchor, replacement) or, when the anchor is a
tuple of (file, anchor, replacement) triples, several edits applied
together; every anchor must occur exactly once.

SCRATCH HYGIENE (D12F.2).  The copies are removed in a `finally` -- which a
signal used to bypass: SIGTERM and SIGHUP (an ssh session that drops, a
tool-call timeout) end a Python process WITHOUT unwinding, so a run killed
that way left its copies behind (seven of them under
output/prefreeze_negctl/ on 2026-10-02).  Now SIGTERM and SIGHUP are turned
into SystemExit for the duration of the run, so the `finally` runs; every
copy carries an owner record (`.negctl_owner`: pid and host of the runner
that made it); and a run first removes the copies whose owner process is
gone on this host (a SIGKILL cannot be caught), keeps those of a live
runner, and counts and reports both, plus any copy with no owner record,
which it never deletes.
"""
import argparse
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# test file -> the tests run against every copy
TESTS = {
    'tests/test_wp3_link_state_service.py': (
        'test_installation_formula_on_a_hand_computed_topology',
        'test_the_min_ls_interval_throttle_delays_a_second_origination',
        'test_k_simultaneous_failures_at_one_router_install_within_one_origination',
        'test_a_transition_learned_inside_the_min_ls_interval_waits',
        'test_a_stale_administrative_record_cannot_date_a_later_origination',
        'test_a_flow_over_a_tombstoned_hop_delivers_exactly_zero',
    ),
    'tests/test_wp2_tick_step.py': (
        'test_lock_applies_the_fixed_pre_event_configuration',
        'test_a_site_whose_own_flag_is_not_set_keeps_the_pre_event_action',
        'test_a_locked_sites_postcard_is_refused_by_the_audit',
    ),
    'tests/test_wp5_postcards.py': (
        'test_the_heartbeat_is_the_same_environment_rule_for_every_agent_arm',
        'test_a_heartbeat_stamp_from_a_later_tick_does_not_silence_a_site',
        'test_the_removed_switches_are_refused',
    ),
    'tests/test_merge_seams.py': (
        'test_seam_c_every_block_b_column_is_the_fresh_latch',
        'test_seam_c_no_block_b_column_is_silently_constant',
    ),
    'tests/test_wp13_integration.py': (
        'test_the_schema_removed_exactly_the_fields_constant_in_every_pass',
        'test_the_short_pass_constants_that_move_at_production_length_stay',
        'test_the_pre_wp13_schema_id_is_refused_and_minting_stamps_the_new_one',
    ),
    'tests/test_run_chain.py': (
        'test_the_primary_analysis_reads_a_set_it_can_actually_analyse',
        'test_an_ablation_is_compared_with_the_baseline_at_its_own_tick_count',
        'test_the_primary_step_analyses_exactly_the_profiles_confirmatory_input',
    ),
}

# name -> (file, anchor, replacement, {test file: tests that MUST fail}, what)
MUTATIONS = {
    'per_transition_lsa': (
        'sixg_sim/link_state_service.py',
        '            carried = [(p, i) for tk, p, i in queue if tk <= tick]\n'
        '            rest = [e for e in queue if e[0] > tick]\n',
        '            _j = min(range(len(queue)), key=lambda j: queue[j][0])'
        '  # NEGCTL\n'
        '            carried = [queue[_j][1:]]\n'
        '            rest = queue[:_j] + queue[_j + 1:]\n',
        {'tests/test_wp3_link_state_service.py': (
            'test_k_simultaneous_failures_at_one_router_install_within_one_origination',
            'test_a_transition_learned_inside_the_min_ls_interval_waits')},
        'one Router-LSA per pair TRANSITION, serialised at MIN_LS_INTERVAL '
        '(the rule before D13.1): k simultaneous failures at one router '
        'install 5(k-1) ticks apart'),
    'global_action_lock': (
        'sixg_sim/tick_step.py',
        '        acting = sim.update_site_action_locks(tick)\n',
        '        acting = sorted(sim.agents) if lock_open else []  # NEGCTL\n',
        {'tests/test_wp2_tick_step.py': (
            'test_a_site_whose_own_flag_is_not_set_keeps_the_pre_event_action',)},
        'every agent site acts once the GLOBAL island trigger fires (the lock '
        'before D13.2): a site whose own flag is not set acts because another '
        'site is isolated'),
    'heartbeat_no_lower_bound': (
        'sixg_sim/simulation.py',
        'if last is not None and 0 <= (self.current_tick - last) < '
        'POSTCARD_PERIOD:',
        'if last is not None and (self.current_tick - last) < '
        'POSTCARD_PERIOD:  # NEGCTL',
        {'tests/test_wp5_postcards.py': (
            'test_a_heartbeat_stamp_from_a_later_tick_does_not_silence_a_site',)},
        'the heartbeat due-test without its lower bound (before D5.13): a '
        'stamp from a LATER tick counts as not due and silences the site'),
    'drain_legacy_accepted': (
        'sixg_sim/simulation.py',
        "if POSTCARD_DRAIN != 'shared':",
        "if POSTCARD_DRAIN not in ('shared', 'legacy'):  # NEGCTL",
        {'tests/test_wp5_postcards.py': (
            'test_the_removed_switches_are_refused',)},
        'MARL_POSTCARD_DRAIN=legacy accepted again (before D5.14): the '
        'coordinator drains every queue before the receiver latch'),
    'own_drop_constant': (
        'sixg_sim/agent.py',
        'lambda o: o.neighbor_radio.experienced_sinr_drop_norm),',
        'lambda o: 0.0),  # NEGCTL',
        {'tests/test_merge_seams.py': (
            'test_seam_c_no_block_b_column_is_silently_constant',)},
        'b_own_sinr_drop wired to a constant: the seam (c) test, whose last '
        'assertion was conditional until D13.5, must see it'),
    'short_pass_removal': (
        None,
        (('sixg_sim/agent.py',
          "    ('g_island_size', lambda o: o.connectivity.island_size_norm),\n"
          "    ('g_pending_iops', lambda o: o.connectivity.pending_iops_norm),\n",
          "    ('g_island_size', lambda o: o.connectivity.island_size_norm),"
          "  # NEGCTL\n"),
         ('sixg_sim/agent.py',
          "    'g_multi_island_bridge',       # 0.0 on every sample\n)\n",
          "    'g_multi_island_bridge',       # 0.0 on every sample\n"
          "    'g_pending_iops',  # NEGCTL\n)\n"),
         ('sixg_sim/agent.py',
          'assert len(CRITIC_GLOBAL_FIELDS) == 11, len(CRITIC_GLOBAL_FIELDS)',
          'assert len(CRITIC_GLOBAL_FIELDS) == 10, len(CRITIC_GLOBAL_FIELDS)'
          '  # NEGCTL')),
        None,
        {'tests/test_wp13_integration.py': (
            'test_the_schema_removed_exactly_the_fields_constant_in_every_pass',
            'test_the_short_pass_constants_that_move_at_production_length_stay')},
        'g_pending_iops removed as well, as the 600-tick merge passes '
        'suggested: it is constant in Scenario A but MOVES in Scenario B at '
        'production length, so the evidence must refuse the removal'),
    'rehearsal_s7_reduced': (
        'confirmatory/chain_plan.py',
        "        'cf_analysis_prefixes': 's6_cffull_b',\n",
        "        'cf_analysis_prefixes': 's6_cf_b',  # NEGCTL\n",
        {'tests/test_run_chain.py': (
            'test_the_primary_analysis_reads_a_set_it_can_actually_analyse',
            'test_the_primary_step_analyses_exactly_the_profiles_confirmatory_input')},
        "the rehearsal's S7 analyses the reduced 600-tick S6 again (the "
        '2026-09-24 rehearsal): confirmatory_analysis.py refuses that set'),
    'ablation_mixed_ticks': (
        'confirmatory/run_final.sh',
        '--artifacts "$INV_ABLATION_OF" --same-ticks-as "$id")',
        '--artifacts "$INV_ABLATION_OF")  # NEGCTL',
        {'tests/test_run_chain.py': (
            'test_an_ablation_is_compared_with_the_baseline_at_its_own_tick_count',)},
        'the ablation gate reads every S6 pickle, the 3600-tick subset '
        'included, as the baseline of a 600-tick ablation'),
}

RUNNER = r'''
import importlib.util, sys
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location('t_negctl', sys.argv[2])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import sixg_sim
assert sixg_sim.__file__.startswith(sys.argv[1]), sixg_sim.__file__
for name in sys.argv[3:]:
    try:
        getattr(mod, name)()
    except BaseException as exc:
        print('RESULT FAIL %s %s' % (name, type(exc).__name__))
    else:
        print('RESULT PASS %s' % name)
'''


def _copy_tree(dst):
    """Copy the tree's sources (the synced file list when present, else git's
    list, else a walk) into dst: Python, shell and config files only."""
    listing = os.path.join(ROOT, '.synced_files')
    files = None
    if os.path.exists(listing):
        with open(listing) as fh:
            files = [l.strip() for l in fh if l.strip()]
    else:
        try:
            out = subprocess.run(['git', 'ls-files'], cwd=ROOT,
                                 capture_output=True, text=True, check=True)
            files = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        except Exception:
            files = None
    if files is None:
        files = []
        for base, dirs, names in os.walk(ROOT):
            rel = os.path.relpath(base, ROOT)
            if rel.split(os.sep)[0] in ('.git', 'output', 'logs'):
                dirs[:] = []
                continue
            files += [os.path.normpath(os.path.join(rel, n)) for n in names]
    n = 0
    for rel in files:
        if not (rel.endswith('.py') or rel.endswith('.sh')
                or rel.startswith('config' + os.sep)
                or rel.startswith('config/') or rel.startswith('docs/')
                or rel.startswith('tests/data/')):
            continue
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(src, out)
        n += 1
    return n


OWNER = '.negctl_owner'


def _make_copy_dir(base, prefix):
    """A scratch directory under `base` with the owner record written
    first, so that even a copy cut short by SIGKILL can be attributed."""
    d = tempfile.mkdtemp(prefix=prefix, dir=base)
    with open(os.path.join(d, OWNER), 'w') as fh:
        fh.write('%d %s\n' % (os.getpid(), socket.gethostname()))
    return d


def _owner_alive(d):
    """(pid, alive) of the runner that made scratch copy `d`; alive is None
    when `d` has no readable owner record or was made on another host."""
    try:
        with open(os.path.join(d, OWNER)) as fh:
            pid_s, host = fh.read().split()[:2]
        pid = int(pid_s)
    except (OSError, ValueError):
        return None, None
    if host != socket.gethostname():
        return pid, None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return pid, False
    except PermissionError:
        return pid, True
    return pid, True


def _sweep_stale(base):
    """Remove the scratch copies under `base` whose owner process is gone.
    Returns (removed, live, unowned): copies of a live runner and copies
    without an owner record are kept and reported, never deleted."""
    removed, live, unowned = [], [], []
    if not os.path.isdir(base):
        return removed, live, unowned
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        pid, alive = _owner_alive(d)
        if alive is None:
            unowned.append(d)
        elif alive:
            live.append(d)
        else:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed, live, unowned


@contextlib.contextmanager
def _signals_end_the_run():
    """SIGTERM and SIGHUP raise SystemExit(128 + signum) inside the block,
    so a `finally` around the run still removes the scratch copies; the
    previous handlers are restored afterwards."""
    def _exit(signum, _frame):
        raise SystemExit(128 + signum)
    old = {}
    for s in (signal.SIGTERM, signal.SIGHUP):
        old[s] = signal.signal(s, _exit)
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _mutate(tree, rel, anchor, repl):
    path = os.path.join(tree, rel)
    with open(path, 'rb') as fh:
        data = fh.read()
    a, r = anchor.encode('utf-8'), repl.encode('utf-8')
    if b'\r\n' in data and b'\r\n' not in a:
        # a CRLF file: the anchor is written with LF
        a = a.replace(b'\n', b'\r\n')
        r = r.replace(b'\n', b'\r\n')
    count = data.count(a)
    if count != 1:
        return 'anchor found %d times in %s (expected exactly 1)' % (count, rel)
    with open(path, 'wb') as fh:
        fh.write(data.replace(a, r))
    return None


def _run(tree, test_file, tests):
    env = dict(os.environ)
    env.setdefault('PYTHONHASHSEED', '0')
    env.pop('MARL_RUN_TESTS_CHILD', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    p = subprocess.run(
        [sys.executable, '-c', RUNNER, tree, os.path.join(tree, test_file)]
        + list(tests), cwd=tree, env=env, capture_output=True, text=True)
    res = {}
    for line in p.stdout.splitlines():
        if line.startswith('RESULT '):
            _, verdict, name = line.split()[:3]
            res[name] = verdict
    if p.returncode != 0 or len(res) != len(tests):
        sys.stdout.write(p.stdout[-3000:])
        sys.stdout.write(p.stderr[-3000:])
        raise SystemExit('test runner failed on %s %s (exit %d, %d/%d results)'
                         % (tree, test_file, p.returncode, len(res),
                            len(tests)))
    return res


def _run_all(tree, tests_by_file):
    res = {}
    for tf in sorted(tests_by_file):
        for name, verdict in _run(tree, tf, tests_by_file[tf]).items():
            res[(tf, name)] = verdict
    return res


def _plan(mutations, tests=None, table=None):
    """test file -> tests: the static list plus every must_fail test of the
    selected mutations (so a control's tests always run on the head too)."""
    tests = TESTS if tests is None else tests
    table = MUTATIONS if table is None else table
    plan = {tf: list(ts) for tf, ts in tests.items()}
    for m in mutations:
        for tf, ts in table[m][3].items():
            cur = plan.setdefault(tf, [])
            cur += [t for t in ts if t not in cur]
    return plan


def main(argv, tests=None, table=None, title='WP13', scratch='wp13_negctl',
         doc=None):
    """Run the controls of `table` (default: this module's MUTATIONS) with
    the head tests of `tests` (default: TESTS), copying into
    output/<scratch>/.  tools/prefreeze_negative_controls.py calls this with
    its own tables."""
    tests = TESTS if tests is None else tests
    MUTATIONS = table if table is not None else globals()['MUTATIONS']
    ap = argparse.ArgumentParser(description=(doc or __doc__).split('\n\n')[0])
    ap.add_argument('mutations', nargs='*', default=None)
    ap.add_argument('--keep', action='store_true',
                    help='keep the scratch copies under output/%s/' % scratch)
    ap.add_argument('--list', action='store_true',
                    help='list the controls and exit')
    a = ap.parse_args(argv)
    names = a.mutations or sorted(MUTATIONS)
    if a.list:
        for n in sorted(MUTATIONS):
            print('%-26s %s' % (n, MUTATIONS[n][4]))
        return 0
    unknown = [m for m in names if m not in MUTATIONS]
    if unknown:
        ap.error('unknown mutation(s): %s' % ', '.join(unknown))
    plan = _plan(names, tests, MUTATIONS)
    n_tests = sum(len(v) for v in plan.values())
    base = os.path.join(ROOT, 'output', scratch)
    os.makedirs(base, exist_ok=True)
    head = open(os.path.join(ROOT, '.synced_head')).read().strip() \
        if os.path.exists(os.path.join(ROOT, '.synced_head')) else '?'
    print('# %s negative controls: tree %s head %s, %d controls, %d tests '
          'in %d files' % (title, ROOT, head, len(names), n_tests, len(plan)))
    removed, live, unowned = _sweep_stale(base)
    print('# scratch sweep of %s: %d stale copies of a dead runner removed, '
          '%d of a live runner kept, %d without an owner record kept'
          % (base, len(removed), len(live), len(unowned)))
    for d in removed:
        print('#   removed stale copy %s' % d)
    for d in live + unowned:
        print('#   WARNING: kept %s (%s)' % (d, 'live runner' if d in live
                                             else 'no owner record'))
    with _signals_end_the_run():
        return _main_run(a, names, plan, MUTATIONS, base)


def _main_run(a, names, plan, MUTATIONS, base):
    """The body of main(): the head copy, then one copy per control; every
    copy is removed in the `finally` unless --keep."""
    ok = True
    caught = 0
    scratch = []
    try:
        d = _make_copy_dir(base, 'head_')
        scratch.append(d)
        n = _copy_tree(d)
        res = _run_all(d, plan)
        red = sorted('%s::%s' % k for k, v in res.items() if v != 'PASS')
        print('HEAD  %d files copied, %d/%d tests pass%s'
              % (n, len(res) - len(red), len(res),
                 ('  FAILING: ' + ', '.join(red)) if red else ''))
        if red or len(res) == 0:
            print('VERDICT FAIL: the head is not green (or ran nothing), no '
                  'control is attributable')
            return 1
        for name in names:
            rel, anchor, repl, must_fail, what = MUTATIONS[name]
            d = _make_copy_dir(base, name + '_')
            scratch.append(d)
            _copy_tree(d)
            edits = (anchor if isinstance(anchor, tuple)
                     else ((rel, anchor, repl),))
            err = None
            for e_rel, e_anchor, e_repl in edits:
                err = _mutate(d, e_rel, e_anchor, e_repl)
                if err:
                    break
            if err:
                ok = False
                print('%-26s NOT APPLIED: %s' % (name, err))
                continue
            res = _run_all(d, plan)
            failed = sorted(k for k, v in res.items() if v != 'PASS')
            want = [(tf, t) for tf, ts in must_fail.items() for t in ts]
            missed = [k for k in want if k not in failed]
            verdict = 'caught' if (want and not missed) else 'NOT CAUGHT'
            ok = ok and bool(want) and not missed
            caught += int(bool(want) and not missed)
            print('%-26s %s  (%s)' % (name, verdict, what))
            print('    ran %d tests, fails %d: %s'
                  % (len(res), len(failed),
                     ', '.join(t for _f, t in failed) or '-'))
            print('    must fail %d, failed %d' % (len(want),
                                                    len(want) - len(missed)))
            if missed:
                print('    expected to fail but passed: %s'
                      % ', '.join(t for _f, t in missed))
    finally:
        if not a.keep:
            for d in scratch:
                for _ in range(10):
                    shutil.rmtree(d, ignore_errors=True)
                    if not os.path.exists(d):
                        break
                    time.sleep(1.0)
                if os.path.exists(d):
                    print('WARNING: scratch copy not removed: %s' % d)
            try:
                os.rmdir(base)            # only when empty
            except OSError:
                pass
    ok = ok and caught == len(names) and caught > 0
    print('VERDICT %s: %d/%d controls caught'
          % ('PASS' if ok else 'FAIL', caught, len(names)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
