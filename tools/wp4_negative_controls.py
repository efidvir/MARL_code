#!/usr/bin/env python3
"""Negative controls for WP4 (bridge stability), WP4-VERIFY D4.15.

    python3 tools/wp4_negative_controls.py [--keep] [MUTATION ...]

For every mutation: copy the tree's Python sources into a scratch directory
under output/wp4_negctl/ (never outside the tree), re-introduce ONE defect by
an exact, single-occurrence string replacement, run the WP4 unit tests
against the copy in a fresh interpreter, and report which tests fail.  The
unmutated copy is run first and must be fully green, so a failure under a
mutation is attributable to that mutation and nothing else.

A control PASSES when every test it names under `must_fail` fails.  The run
exits 0 only when the head is green, every mutation applied (its anchor text
was found exactly once -- an anchor that drifted would otherwise leave the
copy unmutated and "pass" nothing) and every control passed.

Only the fast unit tests of tests/test_wp4_bridge_stability.py are run (the
harness passes are not needed to exercise the teardown rule), so the whole
run takes seconds.  The scratch copies are deleted afterwards unless --keep.

The central control is `grace_in_ticks`: the engine's own teardown rule with
the grace counted in TICKS SINCE LINK-UP instead of usable passes.  The WP4
claim is that a bridge physically up for many ticks but installed for fewer
than RELAY_BRIDGE_GRACE_PASSES passes is held; under the tick rule the same
bridge is released, and the tests that pin the claim must fail.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# scratch hygiene shared with the WP13 / FIX-PREFREEZE runner (D12F.2):
# owner records, the stale-copy sweep, SIGTERM/SIGHUP -> SystemExit
import wp13_negative_controls as _HYG  # noqa: E402
TEST_FILE = 'tests/test_wp4_bridge_stability.py'

# The unit tests run against every copy (the harness-pass tests are left out:
# they need the checkpoint and minutes per pass, and no mutation here needs
# them to be caught).
UNIT_TESTS = (
    'test_the_constants_are_the_pre_registered_ones',
    'test_the_engine_teardown_sites_name_their_reason',
    'test_grace_counts_installed_passes_and_not_ticks',
    'test_the_central_claim_through_the_teardown_site',
    'test_an_uninstalled_bridge_torn_down_is_an_idle_bridge_teardown',
    'test_a_link_up_for_three_ticks_with_three_passes_is_released',
    'test_hold_reason_lockout',
    'test_hold_reason_not_installed',
    'test_hold_reason_grace',
    'test_hold_reason_busy',
    'test_an_arm_without_a_service_keeps_the_physical_stamp',
    'test_an_arm_without_a_service_counts_passes_on_the_true_graph',
    'test_a_redundant_relay_link_gets_no_protection',
    'test_usable_passes_are_counted_from_the_installed_view',
    'test_idle_bridge_teardowns_are_counted_separately',
    'test_a_non_bridge_teardown_is_neither',
    'test_removals_by_reason_sum_to_the_removal_count',
    'test_the_lifetime_record_is_the_handoff_record',
    'test_was_bridge_survives_the_link_ceasing_to_bridge',
    'test_relay_mode_flips_count_the_executed_mode',
    'test_both_pass_counters_are_taken_at_the_end_of_the_forwarding_pass',
    'test_the_decision_digest_moves_with_the_executed_decision',
    'test_the_postcard_ledger_keeps_silent_and_departed_sites',
    'test_the_postcard_ledger_denominator_is_not_derived_from_the_senders',
    'test_the_postcard_ledger_refuses_a_counter_that_goes_backwards',
)

# name -> (file, anchor, replacement, tests that MUST fail, what it reverts)
MUTATIONS = {
    'grace_in_ticks': (
        'sixg_sim/simulation.py',
        'in_passes = self._relay_grace_in_passes()',
        'in_passes = False  # NEGCTL grace_in_ticks',
        ('test_grace_counts_installed_passes_and_not_ticks',
         'test_the_central_claim_through_the_teardown_site',
         'test_hold_reason_not_installed'),
        'the grace is counted in ticks since link-up for EVERY arm (the '
        'pre-WP4 rule): a bridge up for 50 ticks is released however few '
        'usable passes it has had'),
    'no_not_installed_hold': (
        'sixg_sim/simulation.py',
        'if l.id not in self._relay_route_since:',
        'if False:  # NEGCTL no_not_installed_hold',
        ('test_hold_reason_not_installed',
         'test_the_central_claim_through_the_teardown_site'),
        'a protected bridge that is physically up but not installed is no '
        'longer held for that reason'),
    'passes_ignore_installation': (
        'sixg_sim/simulation.py',
        "stamps = getattr(self, '_relay_route_since', None)",
        "stamps = {l.id: 0 for l in self.active_relay_links()}  # NEGCTL",
        ('test_usable_passes_are_counted_from_the_installed_view',
         'test_the_central_claim_through_the_teardown_site'),
        'a pass is counted for every physically-up relay link, installed '
        'in the forwarding view or not'),
    'idle_ignores_passes': (
        'sixg_sim/simulation.py',
        'if _passes == 0:',
        'if True:  # NEGCTL idle_ignores_passes',
        ('test_idle_bridge_teardowns_are_counted_separately',
         'test_the_central_claim_through_the_teardown_site'),
        'every bridge teardown is booked as idle'),
    'sdn_counts_passes': (
        'sixg_sim/simulation.py',
        "return getattr(self, 'link_state_service', None) is not None",
        'return True  # NEGCTL sdn_counts_passes',
        ('test_an_arm_without_a_service_keeps_the_physical_stamp',),
        'an arm without a link-state service (SDN) also counts passes, so '
        'its bridges are held for ever under not_installed'),
    'no_service_counts_nothing': (
        'sixg_sim/simulation.py',
        "if g.has_edge(a, b) and g[a][b].get('link_id') == l.id:",
        'if False:  # NEGCTL no_service_counts_nothing',
        ('test_an_arm_without_a_service_counts_passes_on_the_true_graph',),
        'an arm without a service counts no usable pass (the c5ffe50 '
        'behaviour), so every SDN bridge teardown is booked as idle'),
    'roster_from_senders': (
        'sixg_sim/tick_step.py',
        'self._roster.update(agent_sites)',
        'self._roster.update(s for s in now if now[s] > self._prev.get(s, 0))'
        '  # NEGCTL',
        ('test_the_postcard_ledger_keeps_silent_and_departed_sites',
         'test_the_postcard_ledger_denominator_is_not_derived_from_the_senders'),
        'the per-site postcard roster is built from the senders, so a '
        'silent site is absent instead of 0'),
}

RUNNER = r'''
import importlib.util, sys, traceback
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location('t_wp4', sys.argv[2])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import sixg_sim
assert sixg_sim.__file__.startswith(sys.argv[1]), sixg_sim.__file__
for name in sys.argv[3:]:
    try:
        getattr(mod, name)()
    except Exception as exc:
        print('RESULT FAIL %s %s' % (name, type(exc).__name__))
    else:
        print('RESULT PASS %s' % name)
'''


def _copy_tree(dst):
    """Copy the tree's sources (the synced file list when present) into dst:
    Python and config files only; checkpoints, pickles and outputs stay."""
    listing = os.path.join(ROOT, '.synced_files')
    if os.path.exists(listing):
        with open(listing) as fh:
            files = [l.strip() for l in fh if l.strip()]
    else:
        files = []
        for base, dirs, names in os.walk(ROOT):
            rel = os.path.relpath(base, ROOT)
            if rel.split(os.sep)[0] in ('.git', 'output', 'logs'):
                dirs[:] = []
                continue
            files += [os.path.normpath(os.path.join(rel, n)) for n in names]
    n = 0
    for rel in files:
        if not (rel.endswith('.py') or rel.startswith('config' + os.sep)
                or rel.startswith('config/')):
            continue
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(src, out)
        n += 1
    return n


def _mutate(tree, rel, anchor, repl):
    path = os.path.join(tree, rel)
    with open(path, 'rb') as fh:
        data = fh.read()
    a, r = anchor.encode('utf-8'), repl.encode('utf-8')
    count = data.count(a)
    if count != 1:
        return 'anchor found %d times in %s (expected exactly 1)' % (count, rel)
    with open(path, 'wb') as fh:
        fh.write(data.replace(a, r))
    return None


def _run(tree, tests):
    env = dict(os.environ)
    env.setdefault('PYTHONHASHSEED', '0')
    env.pop('MARL_RUN_TESTS_CHILD', None)
    # no __pycache__ in the scratch copy: it is deleted afterwards, and a
    # bytecode directory left behind inside a ~/MARL* tree is debris the seed
    # audit would have to walk (one survived the first run at 18606cb)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    p = subprocess.run(
        [sys.executable, '-c', RUNNER, tree, os.path.join(tree, TEST_FILE)]
        + list(tests), cwd=tree, env=env, capture_output=True, text=True)
    res = {}
    for line in p.stdout.splitlines():
        if line.startswith('RESULT '):
            _, verdict, name = line.split()[:3]
            res[name] = verdict
    if p.returncode != 0 or len(res) != len(tests):
        sys.stdout.write(p.stdout[-3000:])
        sys.stdout.write(p.stderr[-3000:])
        raise SystemExit('test runner failed on %s (exit %d, %d/%d results)'
                         % (tree, p.returncode, len(res), len(tests)))
    return res


def main(argv):
    """Run the controls with SIGTERM/SIGHUP turned into SystemExit, so the
    `finally` of the body removes the scratch copies even when the run is
    killed (D12F.2)."""
    with _HYG._signals_end_the_run():
        return _main_body(argv)


def _main_body(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('mutations', nargs='*', default=sorted(MUTATIONS))
    ap.add_argument('--keep', action='store_true',
                    help='keep the scratch copies under output/wp4_negctl/')
    a = ap.parse_args(argv)
    unknown = [m for m in a.mutations if m not in MUTATIONS]
    if unknown:
        ap.error('unknown mutation(s): %s' % ', '.join(unknown))
    base = os.path.join(ROOT, 'output', 'wp4_negctl')
    os.makedirs(base, exist_ok=True)
    head = open(os.path.join(ROOT, '.synced_head')).read().strip() \
        if os.path.exists(os.path.join(ROOT, '.synced_head')) else '?'
    print('# WP4 negative controls: tree %s head %s, %d unit tests'
          % (ROOT, head, len(UNIT_TESTS)))
    removed, live, unowned = _HYG._sweep_stale(base)
    print('# scratch sweep of %s: %d stale copies of a dead runner removed, '
          '%d of a live runner kept, %d without an owner record kept'
          % (base, len(removed), len(live), len(unowned)))
    ok = True
    scratch = []
    try:
        d = _HYG._make_copy_dir(base, 'head_')
        scratch.append(d)
        n = _copy_tree(d)
        res = _run(d, UNIT_TESTS)
        red = sorted(t for t, v in res.items() if v != 'PASS')
        print('HEAD  %d files copied, %d/%d tests pass%s'
              % (n, len(res) - len(red), len(res),
                 ('  FAILING: ' + ', '.join(red)) if red else ''))
        if red:
            print('VERDICT FAIL: the head is not green, no control is '
                  'attributable')
            return 1
        for name in a.mutations:
            rel, anchor, repl, must_fail, what = MUTATIONS[name]
            d = _HYG._make_copy_dir(base, name + '_')
            scratch.append(d)
            _copy_tree(d)
            err = _mutate(d, rel, anchor, repl)
            if err:
                ok = False
                print('%-28s NOT APPLIED: %s' % (name, err))
                continue
            res = _run(d, UNIT_TESTS)
            failed = sorted(t for t, v in res.items() if v != 'PASS')
            missed = [t for t in must_fail if t not in failed]
            verdict = 'caught' if not missed else 'NOT CAUGHT'
            ok = ok and not missed
            print('%-28s %s  (%s)' % (name, verdict, what))
            print('    fails %d: %s' % (len(failed), ', '.join(failed) or '-'))
            if missed:
                print('    expected to fail but passed: %s' % ', '.join(missed))
    finally:
        if not a.keep:
            # The tree is on NFS: a file the just-finished child still had
            # open is silly-renamed to .nfsXXXX and cannot be removed until
            # the server lets go of it, so retry for a few seconds.
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
    print('VERDICT %s' % ('PASS: every control caught' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
