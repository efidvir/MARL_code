#!/usr/bin/env python3
"""Compare per-tick series of two harness pickles (run_timeline_comparison.py).

For every (scenario, arm, seed, key) the series of REF and NEW are compared
element by element and reported as IDENTICAL or DIFFERENT (with the first
differing tick, i.e. list index, and the values there).  Floats must match
bit for bit (NaN == NaN; 0.0 != -0.0); ints and floats compare by value, and
a series whose values are equal but whose numeric element type changed
(int / float / bool) is listed in a NOTE.

Usage:
  series_diff.py REF.pkl NEW.pkl [--arms a,b] [--scenarios recovery,rescue_ops]
                 [--seeds 82,91] [--keys conn,delivered,...|all|common]
                 [--allow-removed k1,k2] [--quiet]

Defaults: arms = union of the arms present in either pickle; scenarios =
both; seeds = union of the pickles' 'seeds'; keys = DEFAULT_KEYS.

Key selections:
  explicit list  exactly these keys; one missing on either side is MISSING.
  all            every key either run carries, plus DEFAULT_KEYS; a key on
                 one side only is MISSING.
  common         every key REF carries, plus DEFAULT_KEYS, must be present
                 and identical in NEW (a key REF has and NEW lacks is
                 MISSING).  Keys only NEW carries are new series: they are
                 not compared and are listed in a NOTE.
--allow-removed  keys that NEW may lack although REF has them (listed in a
                 NOTE instead of failing).  DEFAULT_KEYS cannot be allowed:
                 they carry the section-14 series (delivered, fragments, ...).

Exit status: 0 all identical, 1 any difference or missing series, 2 usage.
"""
import argparse
import math
import os
import pickle
import struct
import sys

try:
    import numpy as _np
except ImportError:          # pragma: no cover - numpy is always on the server
    _np = None

DEFAULT_KEYS = ('conn', 'delivered', 'fragments', 'relay_bridges',
                'relay_count', 'link_count', 'energy', 'latency')

SCENARIOS = {'recovery': 'recovery_multi', 'rescue_ops': 'rescue_multi'}
_SCEN_ALIASES = {'recovery': 'recovery', 'a': 'recovery',
                 'recovery_multi': 'recovery',
                 'rescue_ops': 'rescue_ops', 'rescue': 'rescue_ops',
                 'b': 'rescue_ops', 'rescue_multi': 'rescue_ops'}

MISSING = object()


def _is_float(x):
    if isinstance(x, float):
        return True
    return _np is not None and isinstance(x, _np.floating)


def _is_number(x):
    if isinstance(x, (int, float)):
        return True
    return _np is not None and isinstance(x, (_np.integer, _np.floating))


def _num_class(x):
    """'bool' / 'int' / 'float' for numbers (numpy included), else None."""
    if isinstance(x, bool) or (_np is not None and isinstance(x, _np.bool_)):
        return 'bool'
    if _is_float(x):
        return 'float'
    if _is_number(x):
        return 'int'
    return None


def _as_list(x):
    if hasattr(x, 'tolist') and not _is_number(x):      # numpy arrays
        return x.tolist()
    return x


def same(a, b):
    """Strict equality used for 'bit-identical'."""
    a, b = _as_list(a), _as_list(b)
    if _is_float(a) and _is_float(b):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return struct.pack('<d', fa) == struct.pack('<d', fb)
    if _is_number(a) and _is_number(b):
        if _is_float(a) or _is_float(b):
            fa, fb = float(a), float(b)
            if math.isnan(fa) or math.isnan(fb):
                return False
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return (list(a.keys()) == list(b.keys())
                and all(same(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def compare_series(ref, new):
    """Return None when identical, else a dict describing the difference."""
    if ref is MISSING or new is MISSING:
        return {'kind': 'missing',
                'ref_missing': ref is MISSING, 'new_missing': new is MISSING}
    if hasattr(ref, 'tolist'):
        ref = ref.tolist()
    if hasattr(new, 'tolist'):
        new = new.tolist()
    if not isinstance(ref, (list, tuple)) or not isinstance(new, (list, tuple)):
        if same(ref, new):
            return None
        return {'kind': 'scalar', 'ref': ref, 'new': new}
    first = None
    n_diff = 0
    for i, (x, y) in enumerate(zip(ref, new)):
        if not same(x, y):
            n_diff += 1
            if first is None:
                first = i
    if first is None and len(ref) == len(new):
        return None
    d = {'kind': 'series', 'len_ref': len(ref), 'len_new': len(new),
         'n_diff': n_diff, 'first_tick': first}
    if first is None:            # equal common prefix, different lengths
        d['first_tick'] = min(len(ref), len(new))
    else:
        d['ref'] = ref[first]
        d['new'] = new[first]
    return d


def type_change(ref, new):
    """For two values that compare identical: where the numeric element class
    (bool / int / float) differs -- the first list index, 'scalar', or None.
    Only top-level elements are inspected."""
    if ref is MISSING or new is MISSING:
        return None
    ref, new = _as_list(ref), _as_list(new)
    if isinstance(ref, (list, tuple)) and isinstance(new, (list, tuple)):
        for i, (x, y) in enumerate(zip(ref, new)):
            cx, cy = _num_class(x), _num_class(y)
            if cx is not None and cy is not None and cx != cy:
                return i
        return None
    cx, cy = _num_class(ref), _num_class(new)
    if cx is not None and cy is not None and cx != cy:
        return 'scalar'
    return None


def _fmt(v):
    if _is_float(v):
        return repr(float(v))
    s = repr(v)
    return s if len(s) <= 60 else s[:57] + '...'


def describe(d):
    if d is None:
        return 'IDENTICAL'
    if d['kind'] == 'missing':
        where = [w for w, m in (('ref', d['ref_missing']),
                                ('new', d['new_missing'])) if m]
        return 'MISSING in %s' % '+'.join(where)
    if d['kind'] == 'scalar':
        return 'DIFFERENT (scalar) ref=%s new=%s' % (_fmt(d['ref']), _fmt(d['new']))
    s = 'DIFFERENT first_tick=%d' % d['first_tick']
    if 'ref' in d:
        s += ' ref=%s new=%s' % (_fmt(d['ref']), _fmt(d['new']))
    s += ' n_diff=%d/%d' % (d['n_diff'], min(d['len_ref'], d['len_new']))
    if d['len_ref'] != d['len_new']:
        s += ' len ref=%d new=%d' % (d['len_ref'], d['len_new'])
    return s


def _csv(s):
    return [x.strip() for x in str(s).split(',') if x.strip()]


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _run_of(P, scen, arm, seed):
    runs = (P.get(SCENARIOS[scen]) or {}).get(arm)
    if runs is None:
        return MISSING
    run = runs.get(seed)
    if run is None:
        run = runs.get(str(seed))
    return MISSING if run is None else run


def _keys_for(r_run, n_run, keys, allow_removed=()):
    """Resolve the key selection for one run pair.

    Returns (compare, added, removed):
      compare  keys to compare; a key absent on either side is MISSING
      added    keys only NEW carries, skipped under 'common' (new series)
      removed  keys REF carries and NEW lacks that `allow_removed` tolerates

    keys: an explicit list, or ['all'] / ['common'] (see the module doc).
    Under all/common, DEFAULT_KEYS are always compared.  A pair where both
    runs are missing yields the single pseudo key '*'.
    """
    added = []
    if keys in (['all'], ['common']):
        if r_run is MISSING and n_run is MISSING:
            return ['*'], [], []
        rk = [] if r_run is MISSING else list(r_run.keys())
        nk = [] if n_run is MISSING else list(n_run.keys())
        compare = rk + [k for k in DEFAULT_KEYS if k not in rk]
        extra = [k for k in nk if k not in compare]
        if keys == ['all'] or r_run is MISSING or n_run is MISSING:
            compare += extra      # a lost run: every key it would carry
        else:
            added = extra
    else:
        compare = list(keys)
    removed = []
    if allow_removed and r_run is not MISSING and n_run is not MISSING:
        removed = [k for k in compare
                   if k in allow_removed and k in r_run and k not in n_run]
        compare = [k for k in compare if k not in removed]
    return compare, added, removed


def diff_pickles(ref, new, arms=None, scenarios=None, seeds=None, keys=None,
                 one_sided=None, allow_removed=(), removed=None,
                 type_notes=None):
    """Yield (scenario, arm, seed, key, diff-or-None).

    Optional tally dicts (key -> count), filled while iterating:
      one_sided   keys only NEW carries, skipped under keys=['common']
      removed     REF keys NEW lacks, tolerated through `allow_removed`
      type_notes  identical series whose numeric element type changed
    """
    scenarios = scenarios or list(SCENARIOS)
    keys = list(keys or DEFAULT_KEYS)
    allow_removed = tuple(allow_removed or ())
    if arms is None:
        arms = []
        for P in (ref, new):
            for sk in SCENARIOS.values():
                for a in (P.get(sk) or {}):
                    if a not in arms:
                        arms.append(a)
    if seeds is None:
        seeds = sorted(set(ref.get('seeds') or []) | set(new.get('seeds') or []))
    for scen in scenarios:
        for arm in arms:
            for seed in seeds:
                r_run = _run_of(ref, scen, arm, seed)
                n_run = _run_of(new, scen, arm, seed)
                run_keys, added, gone = _keys_for(r_run, n_run, keys,
                                                  allow_removed)
                for tally, ks in ((one_sided, added), (removed, gone)):
                    if tally is not None:
                        for k in ks:
                            tally[k] = tally.get(k, 0) + 1
                for key in run_keys:
                    r = MISSING if r_run is MISSING else r_run.get(key, MISSING)
                    n = MISSING if n_run is MISSING else n_run.get(key, MISSING)
                    d = compare_series(r, n)
                    if (d is None and type_notes is not None
                            and type_change(r, n) is not None):
                        type_notes[key] = type_notes.get(key, 0) + 1
                    yield scen, arm, seed, key, d


def _prov_line(tag, P):
    p = P.get('provenance')
    if not p:
        return '%s: no provenance record (pre-WP0 pickle)' % tag
    # records written before the flag existed: judge by the fields
    complete = p.get('provenance_complete',
                     bool(p.get('tree_head') and p.get('tree_sha256')))
    return ('%s: head=%s tree_sha256=%s schema=%s PYTHONHASHSEED=%r ticks=%s%s'
            % (tag, p.get('tree_head'), (p.get('tree_sha256') or '-')[:16],
               p.get('schema_id'), p.get('pythonhashseed'),
               p.get('total_ticks'),
               '' if complete else ' [INCOMPLETE: not tied to a commit]'))


def _tally(d):
    return ', '.join('%s(x%d)' % kv for kv in sorted(d.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ref')
    ap.add_argument('new')
    ap.add_argument('--arms', default=None)
    ap.add_argument('--scenarios', default=None,
                    help='recovery,rescue_ops (aliases: A,B)')
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--keys', default=None,
                    help='comma list, or "all" (every key either run '
                         'carries) or "common" (every REF key must be in '
                         'NEW; NEW-only keys are noted). '
                         'Default: ' + ','.join(DEFAULT_KEYS))
    ap.add_argument('--allow-removed', default=None,
                    help='comma list of REF keys NEW may lack (not '
                         'DEFAULT_KEYS)')
    ap.add_argument('--quiet', '-q', action='store_true',
                    help='print only non-identical series and the summary')
    a = ap.parse_args(argv)

    for p in (a.ref, a.new):
        if not os.path.isfile(p):
            print('series_diff: no such file: %s' % p, file=sys.stderr)
            return 2
    scenarios = None
    if a.scenarios:
        scenarios = []
        for s in _csv(a.scenarios):
            if s.lower() not in _SCEN_ALIASES:
                print('series_diff: unknown scenario %r' % s, file=sys.stderr)
                return 2
            scenarios.append(_SCEN_ALIASES[s.lower()])
    seeds = [int(s) for s in _csv(a.seeds)] if a.seeds else None
    arms = _csv(a.arms) if a.arms else None
    keys = _csv(a.keys) if a.keys else None
    allow_removed = _csv(a.allow_removed) if a.allow_removed else []
    protected = [k for k in allow_removed if k in DEFAULT_KEYS]
    if protected:
        print('series_diff: --allow-removed cannot name required keys %s '
              '(section-14 series); regenerate the reference instead'
              % ','.join(protected), file=sys.stderr)
        return 2

    ref, new = load(a.ref), load(a.new)
    print('REF %s' % a.ref)
    print('NEW %s' % a.new)
    print(_prov_line('  ref', ref))
    print(_prov_line('  new', new))

    n = n_same = n_diff = n_missing = 0
    bad_arms = {}
    one_sided, removed, type_notes = {}, {}, {}
    for scen, arm, seed, key, d in diff_pickles(
            ref, new, arms, scenarios, seeds, keys, one_sided,
            allow_removed, removed, type_notes):
        n += 1
        if d is None:
            n_same += 1
        elif d['kind'] == 'missing':
            n_missing += 1
        else:
            n_diff += 1
        if d is not None:
            bad_arms.setdefault(arm, 0)
            bad_arms[arm] += 1
        if d is not None or not a.quiet:
            print('%-10s %-12s seed=%-4s %-15s %s'
                  % (scen, arm, seed, key, describe(d)))
    print('SUMMARY series=%d identical=%d different=%d missing=%d%s'
          % (n, n_same, n_diff, n_missing,
             ('  non-identical arms: ' + ', '.join(
                 '%s(%d)' % kv for kv in sorted(bad_arms.items())))
             if bad_arms else ''))
    if one_sided:
        print('NOTE keys added in NEW (not compared; regenerate the '
              'reference to cover them): %s' % _tally(one_sided))
    if removed:
        print('NOTE keys removed in NEW (allowed by --allow-removed): %s'
              % _tally(removed))
    if type_notes:
        print('NOTE numeric element type changed, values equal '
              '(int/float/bool): %s' % _tally(type_notes))
    if n == 0:
        print('series_diff: nothing compared (check --arms/--seeds)',
              file=sys.stderr)
        return 1
    return 0 if (n_diff == 0 and n_missing == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
