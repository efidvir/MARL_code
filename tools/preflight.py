#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight for ONE invocation of the final-rerun chain (WP RUN-CHAIN).

Recorded and gated before every invocation, per COMPUTE_PLAN.md "failure
handling / A. monitoring and preconditions":

  uptime (load average), MemAvailable, /home free, no leftover job of OURS,
  the tree provenance digest, and the thread-pinning environment.

HARD GATES
  kind=eval   MemAvailable >= 18 GB  and  /home free >= 100 GB
  kind=train  MemAvailable >= 26 GB  and  /home free >= 100 GB,
              and an immediate ABORT (exit 2) below 22 GB.
Both training thresholds are WAVE figures (COMPUTE_PLAN.md states them for
two concurrent replicates), so both are tested against MemAvailable plus the
PSS already held by this chain's own expected jobs -- see memory_verdict().
There is NO SWAP on this host, so an overshoot is an OOM kill, not a
slowdown, and an evaluation pickle is written only at the END of an
invocation: a gate that passes when it should not costs the whole batch.

THE CAMPAIGN'S RECURRING DEFECT is a check that reports success without
having verified anything.  So every check here appends a record to a list,
the summary prints how many checks ran, and the exit status is FAILURE when
that count is zero or when any check could not be evaluated (an unreadable
/proc/meminfo is a failure, never a pass).

  python3 tools/preflight.py --kind eval --tree . --inv s6_cf_b0 \
      --json-out confirmatory/preflight/s6_cf_b0.json [--requested-workers 24]

exit 0 = pass, 1 = a gate failed (retry later is legitimate),
2 = abort level (training below the 22 GB floor; do not retry in a loop).

THE JOB'S ENVIRONMENT (FIX-PREFREEZE item 5), a separate mode run after the
invocation's env file and job script are written and before the launch:

  python3 tools/preflight.py --job-env logs/chain_P/INV.sh \
      --expect-marl-file confirmatory/env_P/INV.marl_env --expect-hashseed 0 \
      --inv INV --json-out confirmatory/preflight_P/INV.env.HHMMSS.json

runs the job script with --print-env FROM THE CALLER'S ENVIRONMENT (so a
job that does not clear it is caught) and requires: the MARL_* set equal to
the plan's, the declared PYTHONHASHSEED, the thread pins at 1, and no other
variable than the system ones the env file records, the pins and bash's
own.  It counts the variables it checked; zero is a FAILURE.
"""
import argparse
import json
import os
import subprocess
import sys
import time

GB = 1024.0 ** 3

# Hard gates, straight from COMPUTE_PLAN.md.
GATES = {
    'eval':  {'mem_gb': 18.0, 'disk_gb': 100.0, 'abort_mem_gb': None},
    'train': {'mem_gb': 26.0, 'disk_gb': 100.0, 'abort_mem_gb': 22.0},
    # Short single-process steps (manifest, selection, analysis, dumps).
    'tool':  {'mem_gb': 4.0,  'disk_gb': 20.0,  'abort_mem_gb': None},
}
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
JOB_PATTERNS = ('run_timeline_comparison.py', 'train_on_comparison.py')
# COMPUTE_PLAN.md failure handling B: "If tenant load is already above ~20
# cores when a training wave is due to start: DELAY the wave rather than run
# it.  A perturbed, slow training block is worse than a late one."  The
# trained weights depend on rollout-worker completion order (D0.8), so this is
# a scientific precondition, not a courtesy.  It applies when the WAVE starts;
# once replicates of this chain are resident they are part of the load and
# re-testing it would refuse every later replicate of the same wave.
TRAIN_LOAD_CEILING = 20.0


def memory_verdict(mem_gb, ours_pss_gb, gate):
    """The memory decision for ONE invocation, as a pure function so it can be
    unit-tested at the plan's real-scale figures instead of only at whatever
    the host happens to have free.

    Both the admission gate and the ABORT floor are stated by COMPUTE_PLAN.md
    for a WAVE ("two concurrent replicates need 26 GB, abort below 22 GB";
    19.8 GB PSS / 29.9 GB sum-RSS).  Once the first replicate of the wave is
    resident, MemAvailable has already fallen by that replicate's footprint,
    so BOTH thresholds have to be tested against the same wave-level figure --
    what MemAvailable would be if this chain's own expected jobs were not yet
    running.  Testing the raw figure against the 22 GB floor (as this function
    used to) aborts the second replicate of every wave, which is the same
    double-charging the 26 GB gate was already fixed for.

    Returns (mem_gate_gb, ok, abort).
    """
    mem_gate = mem_gb + ours_pss_gb
    abort = (gate['abort_mem_gb'] is not None
             and mem_gate < gate['abort_mem_gb'])
    ok = (not abort) and mem_gate >= gate['mem_gb']
    return mem_gate, ok, abort


class Checks(object):
    """A list of checks that knows how many of them actually ran."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, value, detail='', fatal=True):
        self.items.append({'check': name, 'ok': bool(ok), 'value': value,
                           'detail': detail, 'fatal': bool(fatal)})
        return ok

    @property
    def n(self):
        return len(self.items)

    @property
    def failed(self):
        return [c for c in self.items if not c['ok'] and c['fatal']]

    @property
    def warned(self):
        return [c for c in self.items if not c['ok'] and not c['fatal']]


def read_loadavg():
    with open('/proc/loadavg') as f:
        parts = f.read().split()
    return [float(x) for x in parts[:3]]


def read_mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError('MemAvailable missing from /proc/meminfo')


def read_free_gb(path):
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / GB


def _ancestry():
    """pids of this process and its ancestors (never counted as leftovers)."""
    out, pid = set(), os.getpid()
    for _ in range(40):
        out.add(pid)
        try:
            with open('/proc/%d/stat' % pid) as f:
                stat = f.read()
            ppid = int(stat[stat.rfind(')') + 2:].split()[1])
        except Exception:
            break
        if ppid <= 1 or ppid in out:
            break
        pid = ppid
    return out


def chain_inv_of(pid):
    """The chain invocation a process belongs to, from MARL_CHAIN_INV in its
    environment (every invocation's env file exports it)."""
    try:
        with open('/proc/%d/environ' % pid, 'rb') as f:
            for kv in f.read().split(b'\0'):
                if kv.startswith(b'MARL_CHAIN_INV='):
                    return kv.split(b'=', 1)[1].decode('utf-8', 'replace')
    except Exception:
        return None
    return None


def pss_kb(pid):
    """Proportional set size of one process, the fair way to charge shared
    pages (torch/numpy) to a forked pool."""
    try:
        with open('/proc/%d/smaps_rollup' % pid) as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1])
    except Exception:
        pass
    try:
        with open('/proc/%d/statm' % pid) as f:
            return int(f.read().split()[1]) * (os.sysconf('SC_PAGE_SIZE') // 1024)
    except Exception:
        return 0


def scan_jobs(tree):
    """Every simulator/trainer process on the host, split into OURS (cwd under
    `tree`) and FOREIGN (another tenant, or another track's tree).

    The distinction matters: this machine is shared with three other code
    tracks and other tenants.  A foreign job is load, which is recorded and
    which moves the worker recommendation; only a job of OURS is a leftover
    that must be cleared before we launch.
    """
    tree = os.path.realpath(tree)
    mine, foreign, skip = [], [], _ancestry()
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in skip:
            continue
        try:
            with open('/proc/%d/cmdline' % pid, 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
        except Exception:
            continue
        if not cmd or not any(p in cmd for p in JOB_PATTERNS):
            continue
        if 'preflight.py' in cmd or 'pgrep' in cmd:
            continue
        try:
            cwd = os.path.realpath('/proc/%d/cwd' % pid)
        except Exception:
            cwd = ''
        rec = {'pid': pid, 'cwd': cwd, 'cmd': cmd[:220],
               'chain_inv': chain_inv_of(pid), 'pss_kb': pss_kb(pid)}
        if cwd == tree or cwd.startswith(tree + os.sep):
            mine.append(rec)
        else:
            foreign.append(rec)
    return mine, foreign


def provenance(tree):
    sys.path.insert(0, os.path.abspath(tree))
    from sixg_sim import runinfo          # noqa: E402  (path set above)
    rec = runinfo.provenance(root=os.path.abspath(tree))
    return {k: rec.get(k) for k in ('tree_head', 'tree_head_source', 'tree_sha256',
                                    'tree_files', 'schema_id', 'pythonhashseed',
                                    'provenance_complete')}


def recommend_workers(requested, load1, mem_gb, profile):
    """Worker count for the NEXT invocation (never changed mid-invocation).

    real, dryrun   COMPUTE_PLAN.md: evaluation at 24; at a batch boundary drop
                   to 16 below 12 GB MemAvailable and to 8 below 8 GB.
                   'dryrun' is included because it IS the production workload:
                   its whole purpose is to measure the real memory profile and
                   wall clock at 24 workers, and capping it at 16 would measure
                   something the real run will never do.
    other profiles the operating rule for a SHARED machine, which is what the
                   reduced rehearsal runs on: 16 while the 1-minute load is
                   below 12, 8 otherwise -- then the same memory ceilings.
    """
    reasons = []
    cap = int(requested)
    if profile not in ('real', 'dryrun'):
        load_cap = 16 if load1 < 12.0 else 8
        if load_cap < cap:
            reasons.append('load1=%.2f -> %d' % (load1, load_cap))
            cap = load_cap
    if mem_gb < 8.0 and cap > 8:
        reasons.append('MemAvailable %.1f GB < 8 -> 8' % mem_gb)
        cap = 8
    elif mem_gb < 12.0 and cap > 16:
        reasons.append('MemAvailable %.1f GB < 12 -> 16' % mem_gb)
        cap = 16
    return cap, '; '.join(reasons) or 'requested %d unchanged' % requested


# The variables a job's environment may hold besides the plan's MARL_* set:
# what chain_write_env records (CHAIN_SYSTEM_VARS in confirmatory/chain_lib.sh),
# the pins it writes, and what bash itself exports.
ENV_SYSTEM = ('HOME', 'PATH', 'USER', 'LOGNAME', 'SHELL', 'LANG', 'LC_ALL',
              'TMPDIR', 'PY')
ENV_PINS = ('PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED')
ENV_BASH = ('PWD', 'OLDPWD', 'SHLVL', '_')


def read_marl_file(path):
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                k, _, v = ln.partition('=')
                out[k] = v
    return out


def job_environment(job):
    """The environment the job script gives its processes, obtained by
    running it with --print-env from THIS process's environment."""
    # bounded: a job that hangs is a failure, not a wait
    p = subprocess.run(['bash', job, '--print-env'], capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError('job --print-env exited %d: %s'
                           % (p.returncode, p.stderr.decode('utf-8', 'replace')[-300:]))
    env = {}
    for kv in p.stdout.split(b'\0'):
        if kv:
            k, _, v = kv.decode('utf-8', 'replace').partition('=')
            env[k] = v
    return env


def env_check(env, expected, hashseed):
    """Checks for one job environment against the plan.  Returns Checks."""
    c = Checks()
    marl = {k: v for k, v in env.items() if k.startswith('MARL_')}
    stray = sorted(set(marl) - set(expected))
    missing = sorted(set(expected) - set(marl))
    diff = sorted(k for k in set(marl) & set(expected) if marl[k] != expected[k])
    c.add('marl_env', bool(expected) and not (stray or missing or diff),
          len(set(marl) | set(expected)),
          'the job\'s MARL_* set vs the plan\'s %d: stray %s, missing %s, '
          'different %s' % (len(expected), stray, missing,
                            ['%s=%s (plan %s)' % (k, marl[k], expected[k]) for k in diff]))
    hs = env.get('PYTHONHASHSEED')
    c.add('pythonhashseed', hs == str(hashseed), hs,
          'the plan declares %s for this invocation' % hashseed)
    bad = {v: env.get(v) for v in THREAD_VARS if env.get(v) != '1'}
    c.add('thread_pinning', not bad, {v: env.get(v) for v in THREAD_VARS},
          'OMP/MKL/OPENBLAS must be exactly 1')
    allowed = set(expected) | set(ENV_SYSTEM) | set(ENV_PINS) | set(ENV_BASH)
    other = sorted(k for k in env if k not in allowed)
    c.add('no_other_variable', not other, len(env),
          'variables outside the plan, the pins and the recorded system set: %s'
          % other if other else '%d variable(s), all of the plan, the pins or '
          'the recorded system set' % len(env))
    return c


def env_main(a):
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'inv': a.inv, 'kind': 'job_env', 'profile': a.profile,
           'job': a.job_env, 'expect_marl_file': a.expect_marl_file,
           'expect_hashseed': a.expect_hashseed}
    try:
        expected = read_marl_file(a.expect_marl_file)
        env = job_environment(a.job_env)
        c = env_check(env, expected, a.expect_hashseed)
        rec['n_variables'] = len(env)
        rec['marl_env'] = {k: v for k, v in sorted(env.items()) if k.startswith('MARL_')}
    except Exception as exc:
        c = Checks()
        c.add('job_env', False, None, 'could not obtain the job environment: %s' % exc)
    rec['checks'] = c.items
    rec['n_checks'] = c.n
    rec['n_failed'] = len(c.failed)
    ok = c.n > 0 and not c.failed
    rec['verdict'] = 'PASS' if ok else ('FAIL(no checks ran)' if c.n == 0 else 'FAIL')
    if a.json_out:
        d = os.path.dirname(os.path.abspath(a.json_out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json_out, 'w') as f:
            json.dump(rec, f, indent=1, sort_keys=True)
    if not a.quiet:
        print('[PREFLIGHT-ENV] %s %s checks=%d failed=%d variables=%s marl=%d'
              % (rec['verdict'], a.inv, c.n, len(c.failed), rec.get('n_variables', '-'),
                 len(rec.get('marl_env') or {})))
        for ck in c.failed:
            print('           FAIL %s = %r  (%s)' % (ck['check'], ck['value'], ck['detail']))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default=None, choices=sorted(GATES))
    ap.add_argument('--job-env', default=None, metavar='JOB',
                    help='check the environment JOB gives its processes '
                         '(FIX-PREFREEZE item 5) instead of the host gates')
    ap.add_argument('--expect-marl-file', default=None)
    ap.add_argument('--expect-hashseed', default='0')
    ap.add_argument('--tree', default='.')
    ap.add_argument('--inv', default='-')
    ap.add_argument('--stage', default='-')
    ap.add_argument('--profile', default='rehearsal')
    ap.add_argument('--requested-workers', type=int, default=0)
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--expect-inv', default='',
                    help='comma-separated invocation ids this chain has '
                         'deliberately left running (a training wave keeps two '
                         'replicates alive); anything else of ours is a leftover')
    ap.add_argument('--freeze-file', default=None,
                    help='FREEZE_FINAL file: once it exists, the tree digest '
                         'must match the one sealed in it')
    ap.add_argument('--train-load-ceiling', type=float, default=TRAIN_LOAD_CEILING,
                    help='delay a training WAVE while the 1-minute load is at '
                         'or above this (COMPUTE_PLAN.md: ~20 cores of tenant '
                         'load).  Only applied when no replicate of this chain '
                         'is alive, i.e. when the wave is due to start.')
    ap.add_argument('--expect-tree', default='',
                    help='tree_sha256 recorded for the FIRST invocation of this '
                         'run.  Until FREEZE_FINAL is sealed at S5 nothing else '
                         'pins the tree, so a change anywhere across S0-S4 would '
                         'otherwise only be caught by S5b/S6 cross-provenance -- '
                         'after the 340 policy-independent passes, the training '
                         'block, validation and selection have been spent.')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    if a.job_env:
        if not a.expect_marl_file:
            ap.error('--job-env needs --expect-marl-file')
        return env_main(a)
    if not a.kind:
        ap.error('--kind is required')

    g = GATES[a.kind]
    c = Checks()
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'inv': a.inv, 'stage': a.stage, 'kind': a.kind, 'profile': a.profile,
           'tree': os.path.realpath(a.tree)}

    # 1. load average (recorded; informational, and it moves the worker cap)
    try:
        load = read_loadavg()
        rec['loadavg'] = load
        c.add('loadavg', True, load, 'uptime 1/5/15-minute load', fatal=False)
    except Exception as exc:
        load = [99.0, 99.0, 99.0]
        c.add('loadavg', False, None, 'unreadable: %s' % exc)

    # 4-before-2: the job scan has to run first, because the memory gate is
    #    stated for a WAVE, not for one process (see below).
    expected = {x.strip() for x in (a.expect_inv or '').split(',') if x.strip()}
    scan_error = None
    mine = foreign = []
    try:
        mine, foreign = scan_jobs(a.tree)
    except Exception as exc:
        scan_error = exc
    ours_expected = [j for j in mine if j.get('chain_inv') in expected]
    leftovers = [j for j in mine if j.get('chain_inv') not in expected]
    ours_pss_gb = sum(j.get('pss_kb', 0) for j in ours_expected) / (1024.0 * 1024.0)

    # 2. MemAvailable.
    #    COMPUTE_PLAN.md states the training gate for a WAVE: two concurrent
    #    replicates need 26 GB (19.8 GB PSS / 29.9 GB sum-RSS).  Once the
    #    first replicate of the wave is resident, MemAvailable has already
    #    fallen by its footprint, so testing the raw figure again for the
    #    SECOND replicate double-charges the wave for memory it is already
    #    using and can never pass.  The gate therefore tests what MemAvailable
    #    would be if this chain's own expected jobs were not yet running,
    #    while the hard ABORT floor stays on the raw figure.
    try:
        mem = read_mem_available_gb()
        mem_gate, mem_ok, mem_abort = memory_verdict(mem, ours_pss_gb, g)
        rec['mem_available_gb'] = round(mem, 2)
        rec['our_expected_pss_gb'] = round(ours_pss_gb, 2)
        rec['mem_gate_gb'] = round(mem_gate, 2)
        if mem_abort:
            c.add('mem_available_gb', False, round(mem_gate, 2),
                  'ABORT: the wave figure %.1f GB (raw MemAvailable %.1f GB + '
                  '%.1f GB already held by the expected jobs of this chain) is '
                  'below the %.0f GB floor'
                  % (mem_gate, mem, ours_pss_gb, g['abort_mem_gb']))
            rec['abort'] = True
        else:
            c.add('mem_available_gb', mem_ok, round(mem_gate, 2),
                  'need >= %.0f GB for the whole wave (no swap on this host); '
                  'raw MemAvailable %.1f GB + %.1f GB already held by the '
                  'expected jobs of this chain' % (g['mem_gb'], mem, ours_pss_gb))
    except Exception as exc:
        mem = 0.0
        c.add('mem_available_gb', False, None, 'unreadable: %s' % exc)

    # 2b. tenant load, for a training WAVE that is about to start
    if a.kind == 'train':
        starting = not ours_expected
        rec['train_load_ceiling'] = a.train_load_ceiling
        rec['train_wave_starting'] = starting
        if starting:
            c.add('tenant_load', load[0] < a.train_load_ceiling, round(load[0], 2),
                  'DELAY the wave rather than run it: the plan refuses to start '
                  'a training block above ~%.0f cores of load, because a '
                  'perturbed training draw is worse than a late one (D0.8)'
                  % a.train_load_ceiling)
        else:
            c.add('tenant_load', True, round(load[0], 2),
                  'wave already running (%d replicate(s) of this chain alive); '
                  'the load ceiling applies when a wave STARTS'
                  % len(ours_expected), fatal=False)

    # 3. /home free
    try:
        free = read_free_gb(os.path.expanduser('~'))
        rec['home_free_gb'] = round(free, 1)
        c.add('home_free_gb', free >= g['disk_gb'], round(free, 1),
              'need >= %.0f GB (shared NFS mount; a quota breach already '
              'destroyed a batch on 2026-09-22)' % g['disk_gb'])
    except Exception as exc:
        c.add('home_free_gb', False, None, 'unreadable: %s' % exc)

    # 4. no UNEXPECTED job of ours
    #    A training wave deliberately keeps two replicates alive, so the
    #    caller names the invocations it has legitimately launched; anything
    #    else of ours in this tree is a leftover and is a hard failure.
    #    Each invocation's env file exports MARL_CHAIN_INV, which is how a
    #    process is tied back to the invocation that owns it.
    try:
        if scan_error is not None:
            raise scan_error
        rec['our_jobs'] = mine
        rec['our_expected_jobs'] = len(ours_expected)
        rec['expect_inv'] = sorted(expected)
        rec['foreign_jobs'] = len(foreign)
        rec['foreign_job_cwds'] = sorted({j['cwd'] for j in foreign})
        c.add('our_jobs', not leftovers, len(leftovers),
              ('leftover simulator/trainer processes in %s: %s'
               % (rec['tree'], [(j['pid'], j.get('chain_inv')) for j in leftovers]))
              if leftovers else
              'no leftover job of ours in %s (%d expected job(s) of this chain '
              'still running: %s)' % (rec['tree'], len(ours_expected),
                                      sorted(expected) or 'none'))
        c.add('foreign_jobs', True, len(foreign),
              'other tracks/tenants running (recorded, not gated): %s'
              % ', '.join(rec['foreign_job_cwds'][:4]), fatal=False)
    except Exception as exc:
        c.add('our_jobs', False, None, 'process scan failed: %s' % exc)

    # 5. tree provenance digest
    try:
        prov = provenance(a.tree)
        rec['provenance'] = prov
        c.add('tree_provenance', bool(prov.get('tree_sha256')),
              (prov.get('tree_sha256') or '')[:16],
              'head=%s schema=%s complete=%s' % (prov.get('tree_head'),
                                                 prov.get('schema_id'),
                                                 prov.get('provenance_complete')))
    except Exception as exc:
        c.add('tree_provenance', False, None, 'provenance failed: %s' % exc)

    # 5a. the tree may not move DURING a run, even before freeze 2 exists.
    #     The freeze file is written by S5; everything from S0 to S4 runs
    #     before it, and the other three code tracks are still landing work
    #     packages in this tree.  So the first invocation of a run pins the
    #     digest in the ledger and every later invocation is compared to it:
    #     a 12-hour-late cross-provenance failure becomes a batch-boundary stop.
    if a.expect_tree and a.expect_tree != '-':
        got = (rec.get('provenance') or {}).get('tree_sha256')
        c.add('tree_pinned', got == a.expect_tree, (got or '')[:16],
              'the tree must not change during a run: this run started on '
              '%s (recorded in the chain ledger)' % a.expect_tree[:16])

    # 5b. once freeze 2 is sealed, the tree may not move again
    if a.freeze_file:
        sealed = None
        if os.path.exists(a.freeze_file):
            for line in open(a.freeze_file):
                if line.startswith('tree_sha256'):
                    parts = line.split()
                    sealed = parts[1] if len(parts) > 1 else None
                    break
        if sealed:
            got = (rec.get('provenance') or {}).get('tree_sha256')
            c.add('freeze_final', got == sealed, (got or '')[:16],
                  'the tree must not change after FREEZE_FINAL (%s sealed %s)'
                  % (a.freeze_file, (sealed or '')[:16]))
        else:
            c.add('freeze_final', True, None,
                  'freeze 2 not sealed yet (%s)' % a.freeze_file, fatal=False)

    # 6. environment that is scientifically load-bearing
    hs = os.environ.get('PYTHONHASHSEED')
    c.add('pythonhashseed', hs == '0', hs, "must be exactly 0")
    bad = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != '1'}
    c.add('thread_pinning', not bad, {v: os.environ.get(v) for v in THREAD_VARS},
          'OMP/MKL/OPENBLAS must be exactly 1: raising them changes the actor '
          'weights bit-wise AND is slower (COMPUTE_PLAN risks)')

    # worker recommendation (never applied mid-invocation)
    if a.requested_workers:
        w, why = recommend_workers(a.requested_workers, load[0],
                                   rec.get('mem_available_gb', 0.0), a.profile)
        rec['requested_workers'] = a.requested_workers
        rec['recommended_workers'] = w
        rec['worker_reason'] = why
        c.add('worker_cap', w >= 1, w, why, fatal=False)

    rec['checks'] = c.items
    rec['n_checks'] = c.n
    rec['n_failed'] = len(c.failed)
    rec['n_warned'] = len(c.warned)

    # A preflight that checked nothing is a FAILURE, not a pass.
    verdict = 'PASS'
    code = 0
    if c.n == 0:
        verdict, code = 'FAIL(no checks ran)', 1
    elif rec.get('abort'):
        verdict, code = 'ABORT', 2
    elif c.failed:
        verdict, code = 'FAIL', 1
    rec['verdict'] = verdict

    rec['tree_sha256'] = (rec.get('provenance') or {}).get('tree_sha256')

    if a.json_out:
        d = os.path.dirname(os.path.abspath(a.json_out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json_out, 'w') as f:
            json.dump(rec, f, indent=1, sort_keys=True)

    if not a.quiet:
        print('[PREFLIGHT] %s %s kind=%s checks=%d failed=%d warned=%d '
              'load1=%.2f mem=%.1fGB home=%.0fGB ours=%d foreign=%d workers=%s'
              % (verdict, a.inv, a.kind, c.n, len(c.failed), len(c.warned),
                 load[0], rec.get('mem_available_gb', -1),
                 rec.get('home_free_gb', -1), len(rec.get('our_jobs', [])),
                 rec.get('foreign_jobs', -1),
                 rec.get('recommended_workers', '-')))
        for ck in c.failed + c.warned:
            print('           %s %s = %r  (%s)'
                  % ('FAIL' if ck['fatal'] else 'warn', ck['check'],
                     ck['value'], ck['detail']))
    return code


if __name__ == '__main__':
    sys.exit(main())
