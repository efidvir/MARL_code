#!/usr/bin/env python3
"""Minimal test runner (the server has no pytest).

  python3 tests/run_tests.py [PATTERN ...]

Imports tests/test_*.py (or the files matching the given substrings), runs
every top-level function named test_* without arguments, and exits non-zero
if any fails.  The same files also run unchanged under pytest.

TEST FILES THAT NEED THEIR OWN ENVIRONMENT.  A file whose sixg_sim imports
depend on environment variables (the comparison environment: the reward
profile is fixed by MARL_REWARD_PROFILE at the first sixg_sim import) declares
them in a module-level `REQUIRES_ENV` dict.  Setting them here would be too
late -- an earlier test file has already imported sixg_sim in this process --
so such a file is RE-RUN IN A CHILD PROCESS with the environment applied,
and the child's results are folded into the summary.  `python3
tests/run_tests.py` with no pattern therefore runs every test, whatever the
caller's environment (WP13: "unit tests updated and passing").

A test that raises unittest.SkipTest is reported as skipped, not failed.
"""
import glob
import importlib.util
import os
import subprocess
import sys
import time
import traceback
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The environment THIS PROCESS was started with.  A test module's own
# os.environ.setdefault() must not be able to make _needs_child() believe the
# variable was exported before the first sixg_sim import.
_ORIG_ENV = dict(os.environ)
CHILD_FLAG = 'MARL_RUN_TESTS_CHILD'


def _needs_child(mod):
    """The REQUIRES_ENV entries this process was NOT started with ({} if the
    file can run here, or if we already are the child)."""
    req = getattr(mod, 'REQUIRES_ENV', None)
    if not req or os.environ.get(CHILD_FLAG):
        return {}
    return {k: v for k, v in req.items() if _ORIG_ENV.get(k) != v}


def _run_in_child(path, missing):
    """Re-run one test file in a child process with `missing` applied.
    Returns (passed, failed, skipped)."""
    name = os.path.basename(path)
    env = dict(os.environ)
    env.update(missing)
    env[CHILD_FLAG] = '1'
    print('----  %s needs %s before the first sixg_sim import: '
          're-running it in a child process'
          % (name, ' '.join('%s=%s' % kv for kv in sorted(missing.items()))))
    sys.stdout.flush()
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), name],
                          cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    out = (proc.stdout or b'').decode('utf-8', 'replace')
    for line in out.splitlines():
        print('  | ' + line)
    sys.stdout.flush()
    counts = [l for l in out.splitlines() if l.startswith('COUNTS ')]
    if not counts:
        print('FAIL  %s (child produced no COUNTS line, rc=%d)'
              % (name, proc.returncode))
        return 0, 1, 0
    got = dict(kv.split('=', 1) for kv in counts[-1].split()[1:])
    return int(got['passed']), int(got['failed']), int(got['skipped'])


def main(argv):
    files = sorted(glob.glob(os.path.join(HERE, 'test_*.py')))
    if argv:
        files = [f for f in files if any(p in os.path.basename(f) for p in argv)]
    n_pass = n_fail = n_skip = 0
    skipped = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except unittest.SkipTest as exc:
            n_skip += 1
            skipped.append('%s (import): %s' % (name, exc))
            print('skip  %s (import)  %s' % (name, exc))
            continue
        except Exception:
            n_fail += 1
            print('FAIL  %s (import)' % name)
            traceback.print_exc()
            continue
        missing = _needs_child(mod)
        if missing:
            p, f, s = _run_in_child(path, missing)
            n_pass += p
            n_fail += f
            n_skip += s
            if s:
                skipped.append('%s (%d skipped in the child)' % (name, s))
            continue
        for fn_name in sorted(n for n in vars(mod) if n.startswith('test_')):
            fn = getattr(mod, fn_name)
            if not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
            except unittest.SkipTest as exc:
                n_skip += 1
                skipped.append('%s::%s: %s' % (name, fn_name, exc))
                print('skip  %s::%s  %s' % (name, fn_name, exc))
            except Exception:
                n_fail += 1
                print('FAIL  %s::%s' % (name, fn_name))
                traceback.print_exc()
            else:
                n_pass += 1
                print('ok    %s::%s  (%.1fs)' % (name, fn_name, time.time() - t0))
    for s in skipped:
        print('SKIPPED  %s' % s)
    # machine-readable (read back by _run_in_child) and human summary
    print('COUNTS passed=%d failed=%d skipped=%d' % (n_pass, n_fail, n_skip))
    print('%d passed, %d failed%s'
          % (n_pass, n_fail, (', %d skipped' % n_skip) if n_skip else ''))
    return 1 if (n_fail or not n_pass) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
