"""Run-environment helpers shared by every driver (WP0).

* ``pool_workers``: the MARL_POOL_WORKERS cap applied to every process pool.
* ``provenance``: the record stored in every result pickle and checkpoint
  (tree HEAD, tree digest, PYTHONHASHSEED, schema id, MARL_* environment);
  ``provenance_complete`` is False, with a warning on stderr, when the tree
  cannot be tied to a commit.
* ``check_hashseed``: loud warning when PYTHONHASHSEED is not '0'.
* ``pass_meter_start`` / ``pass_meter_stop``: wall time, CPU time and peak
  RSS of one physics pass inside a (re-used) pool worker.

Nothing in this module reads or writes simulation state, so it cannot change
the behaviour of any arm.
"""
import fnmatch
import hashlib
import os
import platform
import socket
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# WP6 landed: sixg_sim.agent.SCHEMA_ID is the hash of the two ordered
# observation field lists and their dimensions, and schema_id() below
# returns it.  This constant survives only as the value reported when
# sixg_sim.agent cannot be imported at all (a provenance record written
# from a tree without the package), which is itself a fault worth seeing in
# the artefact rather than a silent empty string.
SCHEMA_ID_PLACEHOLDER = 'agent-module-unimportable'

POOL_ENV = 'MARL_POOL_WORKERS'


# ── pool cap ───────────────────────────────────────────────────────────────
def pool_workers(default, env=None):
    """Worker count for a process pool: min(default, $MARL_POOL_WORKERS).

    Unset or empty variable -> ``default`` (the historical behaviour).  A
    value that is not a positive integer is an error, not a silent fallback.
    ``default`` may be 0 (nothing to schedule); the result is then 0.
    """
    env = os.environ if env is None else env
    default = int(default)
    raw = str(env.get(POOL_ENV, '') or '').strip()
    if not raw:
        return default
    try:
        cap = int(raw)
    except ValueError:
        raise ValueError('%s=%r is not an integer' % (POOL_ENV, raw))
    if cap < 1:
        raise ValueError('%s=%r must be >= 1' % (POOL_ENV, raw))
    return min(default, cap)


# ── hash seed ──────────────────────────────────────────────────────────────
def check_hashseed(tag, env=None, stream=None):
    """Print a loud warning unless PYTHONHASHSEED is exactly '0'.

    Returns True when the value is '0'.  The study's energy/latency
    accounting and its provenance checks assume PYTHONHASHSEED=0.
    """
    env = os.environ if env is None else env
    value = env.get('PYTHONHASHSEED')
    if value == '0':
        return True
    stream = sys.stderr if stream is None else stream
    bar = '!' * 78
    lines = [
        bar,
        '!! WARNING [%s]: PYTHONHASHSEED=%r, expected \'0\'.' % (tag, value),
        '!! Salted set/dict iteration order makes the energy and latency',
        '!! accounting irreproducible, and this run\'s provenance will not',
        '!! match the study pickles.  Re-run with PYTHONHASHSEED=0.',
        bar,
    ]
    print('\n'.join(lines), file=stream, flush=True)
    return False


# ── provenance ─────────────────────────────────────────────────────────────
def _read_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError:
        return None


def _git(root, *args):
    try:
        out = subprocess.run(['git', '-C', root] + list(args),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=20, check=True)
        return out.stdout.decode('utf-8', 'replace')
    except Exception:
        return None


def tree_head(root=None):
    """(head, source): ``.synced_head`` if present (server copy), else git."""
    root = root or PROJECT_ROOT
    synced = _read_text(os.path.join(root, '.synced_head'))
    if synced is not None and synced.strip():
        return synced.strip(), '.synced_head'
    head = _git(root, 'rev-parse', '--short', 'HEAD')
    if head:
        dirty = _git(root, 'status', '--porcelain', '--untracked-files=no')
        return head.strip() + ('-dirty' if dirty else ''), 'git'
    return None, 'unavailable'


def _tree_files(root):
    listing = _read_text(os.path.join(root, '.synced_files'))
    if listing is not None:
        return [l for l in listing.splitlines() if l.strip()], '.synced_files'
    listing = _git(root, 'ls-files')
    if listing is not None:
        return [l for l in listing.splitlines() if l.strip()], 'git ls-files'
    return None, 'unavailable'


_DIGEST_CACHE = {}

# Tracked files that are run artefacts, not code: the harness overwrites
# <checkpoint>_adapted_recovery[_freeze]_seed<N>.pt whenever a learning arm
# runs that seed (ce_ck_s1234/policy_final_adapted_recovery_freeze_seed58.pt
# is tracked and rewritten by smoke_tests.sh), which would change the digest
# during and between runs.  Matched against the file's basename.
DIGEST_EXCLUDE = ('*_adapted_recovery*',)


def _digest_excluded(rel):
    base = rel.replace('\\', '/').rsplit('/', 1)[-1]
    return any(fnmatch.fnmatchcase(base, pat) for pat in DIGEST_EXCLUDE)


def tree_digest(root=None):
    """sha256 over (path, sha256(content)) of every tracked file except the
    run artefacts matched by DIGEST_EXCLUDE.

    Detects server-side edits that the synced HEAD alone would not show.
    Cached per process (a pool worker computes it at most once).
    """
    root = os.path.abspath(root or PROJECT_ROOT)
    if root in _DIGEST_CACHE:
        return _DIGEST_CACHE[root]
    files, source = _tree_files(root)
    if files is None:
        res = {'tree_sha256': None, 'tree_files': 0,
               'tree_files_source': source, 'tree_files_missing': [],
               'tree_files_excluded': []}
    else:
        excluded = sorted(f for f in files if _digest_excluded(f))
        files = [f for f in files if not _digest_excluded(f)]
        h = hashlib.sha256()
        missing = []
        for rel in sorted(files):
            p = os.path.join(root, rel)
            try:
                fh = hashlib.sha256()
                with open(p, 'rb') as f:
                    for chunk in iter(lambda: f.read(1 << 20), b''):
                        fh.update(chunk)
                digest = fh.hexdigest()
            except OSError:
                missing.append(rel)
                digest = 'MISSING'
            h.update(rel.encode('utf-8') + b'\0' + digest.encode() + b'\n')
        res = {'tree_sha256': h.hexdigest(), 'tree_files': len(files),
               'tree_files_source': source, 'tree_files_missing': missing,
               'tree_files_excluded': excluded}
    _DIGEST_CACHE[root] = res
    return res


def file_sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def schema_id():
    try:
        from sixg_sim import agent as _agent
    except Exception:
        return SCHEMA_ID_PLACEHOLDER
    return getattr(_agent, 'SCHEMA_ID', SCHEMA_ID_PLACEHOLDER)


def reward_id():
    """Digest of the frozen reward coefficient set (RERUN_SPEC WP8)."""
    from sixg_sim.reward_spec import reward_id as _reward_id
    return _reward_id()


def marl_env(env=None):
    env = os.environ if env is None else env
    return {k: env[k] for k in sorted(env) if k.startswith('MARL_')}


def _version(mod):
    m = sys.modules.get(mod)
    return getattr(m, '__version__', None) if m is not None else None


def provenance(extra=None, root=None):
    """The provenance record stored in result pickles and checkpoints."""
    root = root or PROJECT_ROOT
    head, head_source = tree_head(root)
    rec = {
        'provenance_version': 1,
        'tree_root': os.path.abspath(root),
        'tree_head': head,
        'tree_head_source': head_source,
        'schema_id': schema_id(),
        # RERUN_SPEC WP8: which REWARD produced this artefact.  Not part
        # of schema_id -- that answers "can this checkpoint be loaded";
        # this answers "was it trained by this reward".
        'reward_id': reward_id(),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'marl_env': marl_env(),
        'argv': list(sys.argv),
        'python': platform.python_version(),
        'torch': _version('torch'),
        'numpy': _version('numpy'),
        'networkx': _version('networkx'),
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    rec.update(tree_digest(root))
    rec['provenance_complete'] = bool(rec['tree_head'] and rec['tree_sha256'])
    warn_incomplete(rec)
    if extra:
        rec.update(extra)
    return rec


_WARNED_INCOMPLETE = set()


def warn_incomplete(rec, stream=None):
    """Warn (once per process and tree) when a provenance record cannot be
    tied to a commit: no .synced_head/.synced_files and no git work tree,
    e.g. a `git archive` export.  Returns True when a warning applies."""
    missing = [k for k in ('tree_head', 'tree_sha256') if not rec.get(k)]
    if not missing:
        return False
    key = (rec.get('tree_root'), tuple(missing))
    if key not in _WARNED_INCOMPLETE:
        _WARNED_INCOMPLETE.add(key)
        stream = sys.stderr if stream is None else stream
        print('!! WARNING [provenance]: %s unavailable for tree %s (no '
              '.synced_head/.synced_files and no git work tree); records '
              'written by this run cannot be matched to a commit '
              '(provenance_complete=False).'
              % (' and '.join(missing), rec.get('tree_root')),
              file=stream, flush=True)
    return True


def provenance_summary(rec):
    """One-line human summary of a provenance record."""
    if not rec:
        return 'provenance: (none)'
    return ('provenance: head=%s (%s) tree_sha256=%s schema=%s '
            'PYTHONHASHSEED=%r MARL_*=%d%s'
            % (rec.get('tree_head'), rec.get('tree_head_source'),
               (rec.get('tree_sha256') or '-')[:16], rec.get('schema_id'),
               rec.get('pythonhashseed'), len(rec.get('marl_env') or {}),
               '' if rec.get('provenance_complete', True)
               else '  [INCOMPLETE: not tied to a commit]'))


# ── per-pass resource meter ────────────────────────────────────────────────
def _proc_status_kb(field):
    txt = _read_text('/proc/self/status')
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith(field + ':'):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _pss_kb():
    txt = _read_text('/proc/self/smaps_rollup')
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith('Pss:'):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _ru_maxrss_kb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v // 1024) if sys.platform == 'darwin' else int(v)
    except Exception:
        return None


def _cpu_s():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime
    except Exception:
        return time.process_time()


def pass_meter_start():
    """Reset the kernel's peak-RSS mark (Linux) and start the clocks.

    Pool workers are re-used across passes, so the process-lifetime peak
    (ru_maxrss) would mix passes; writing '5' to /proc/self/clear_refs
    resets VmHWM so the peak read at the end belongs to this pass alone.
    """
    reset = False
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        reset = True
    except OSError:
        pass
    return {'t0': time.time(), 'cpu0': _cpu_s(), 'hwm_reset': reset}


def pass_meter_stop(token):
    hwm = _proc_status_kb('VmHWM') if token.get('hwm_reset') else None
    if hwm is not None:
        peak, source = hwm, 'VmHWM (reset at pass start)'
    else:
        peak, source = _ru_maxrss_kb(), 'ru_maxrss (worker lifetime)'
    rss = _proc_status_kb('VmRSS')
    pss = _pss_kb()
    mb = (lambda kb: None if kb is None else round(kb / 1024.0, 1))
    return {
        'pid': os.getpid(),
        'wall_s': round(time.time() - token['t0'], 2),
        'cpu_s': round(_cpu_s() - token['cpu0'], 2),
        'peak_rss_mb': mb(peak),
        'peak_rss_source': source,
        'rss_end_mb': mb(rss),
        'pss_end_mb': mb(pss),
    }
