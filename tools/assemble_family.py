#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the pickle set of a second confirmatory family (R3-lite family E).

The pre-registered analysis (confirmatory_analysis.py, md5 pinned) tests ONE
learned arm, `marl_static`, against the comparators found in the same pickle
set.  R3-lite registers two families on one confirmatory block: family 1
(marl_static = replicate 1234, run inside the S6 invocations with the seven
comparators) and family E (marl_static = the four-replicate ensemble, run in
its own invocation, chain stage S9x).  The analysis file may not change, so
family E is analysed by handing the SAME analysis a derived pickle set in
which

  * every comparator pass is the pass of the S6 pickles, untouched, and
  * every `marl_static` pass is the ensemble's pass of the same instance.

This tool builds that set and proves it is what it claims to be.  For every
comparator pickle it writes one derived pickle (same seeds, same arms, same
layout).  It REFUSES, leaving nothing behind, when

  * a seed of a comparator pickle has no pass in the arm source, in either
    scenario, or the arm source holds a seed no comparator pickle holds;
  * a (scenario, seed) of the arm occurs in two source pickles, a seed occurs
    in two comparator pickles, two seed keys of one pickle denote the same
    seed, or a seed key is not an integer;
  * an arm-source pickle does not record exactly the expected ordered list
    of checkpoint sha256 (--expect-checkpoint-sha256);
  * the provenance of any two input pickles differs in a field that
    identifies the code, the schema, the hash seed, the length of a pass,
    the library versions or the engine environment;
  * the offered-demand series of an arm pass is not that of the reference
    comparator (--reference-arm, default xldet) on the same instance -- the
    pass was not run on the same instance -- or either series is missing,
    not one-dimensional, not of the pass length, not finite, or constant
    (a series that carries no information identifies no instance);
  * the comparator set does not hold the reference arm, a comparator pickle
    holds no seed, or the assembled set is not exactly the expected key set
    (--expect-keys: one `arm|scenario|seed` per line, the plan's keys);
  * an output would be written onto an input, outside --out-dir, or into an
    --out-dir that already holds something; or --record names an input or an
    output.

It writes into a private directory beside --out-dir, re-reads every derived
pickle there and compares a canonical digest of every run with the digest
of the input it came from (comparators against the comparator pickles, the
arm against the arm source), checks that no input file changed while it
ran, and only then renames the private directory to --out-dir in one step.
The record (--record) lists the sha256 of every input and output file, the
counts and this tool's own sha256; a run that fails for any reason leaves a
record with status REFUSED, never an older one.  pass_stats are copied with
their passes and are not verified (no analysis reads them).

  python3 tools/assemble_family.py \\
      --comparators output/r3lite/s6_cf_b0.pkl,...  \\
      --arm-source  output/r3lite/s9x_ensemble.pkl \\
      --arm marl_static --expect-checkpoint-sha256 SHA,SHA,SHA,SHA \\
      --expect-keys logs/chain_r3lite/S7e.keys \\
      --out-dir output/r3lite/family_e --prefix e_ --record FILE.json

Exit 0: assembled and verified.  Exit 1: refused (the reasons are printed,
--out-dir does not exist or is as it was).  Exit 2: usage.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import signal
import struct
import sys
import tempfile

SCEN_KEYS = ('recovery_multi', 'rescue_multi')
SCEN_NAMES = {'recovery_multi': 'recovery', 'rescue_multi': 'rescue_ops'}
# provenance fields that must be identical across every input pickle
SAME_FIELDS = ('provenance_version', 'tree_head', 'tree_sha256', 'schema_id', 'reward_id',
               'pythonhashseed', 'total_ticks', 'severance_tick', 'rescue_arrival_tick',
               'overrides', 'python', 'torch', 'numpy', 'networkx', 'driver')
# engine-environment variables that legitimately differ between invocations
ENV_VOLATILE = ('MARL_CHAIN_INV', 'MARL_EVAL_CHECKPOINT', 'MARL_POOL_WORKERS')
OFFERED_KEY = 'offered'
PRIVATE_PREFIX = '.assembling_'
_NAN8 = struct.pack('>d', float('nan'))


class Refused(Exception):
    """An input the tool will not assemble; the message is the reason."""


class _Collected(Exception):
    """Raised when the reasons are already in the list of failures."""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _canon(h, x):
    """Feed a canonical byte form of x to the hash: independent of pickle
    memoisation, of dict insertion order and of the container's identity.
    Every float is eight bytes; every NaN is the one canonical NaN."""
    import numpy as np
    if x is None:
        h.update(b'N')
    elif isinstance(x, bool):
        h.update(b'T' if x else b'F')
    elif isinstance(x, int):
        h.update(b'i' + str(x).encode() + b';')
    elif isinstance(x, float):
        h.update(b'f' + (_NAN8 if math.isnan(x) else struct.pack('>d', x)))
    elif isinstance(x, str):
        b = x.encode('utf-8', 'surrogatepass')
        h.update(b's' + str(len(b)).encode() + b':' + b)
    elif isinstance(x, (bytes, bytearray)):
        h.update(b'b' + str(len(x)).encode() + b':' + bytes(x))
    elif isinstance(x, dict):
        h.update(b'{' + str(len(x)).encode() + b':')
        items = [(type(k).__name__, repr(k), _digest(k), _digest(v)) for k, v in x.items()]
        for tn, rk, dk, dv in sorted(items):
            h.update(dk.encode() + b'=' + dv.encode() + b',')
        h.update(b'}')
    elif isinstance(x, list) and x and all(type(v) is float for v in x):
        # the per-tick series: one buffer instead of one call per tick (IEEE
        # big-endian doubles; signed zero keeps its bits, NaN is canonical)
        a = np.array(x, dtype='>f8')
        if np.isnan(a).any():
            a[np.isnan(a)] = np.nan
        h.update(b'Lf' + str(len(x)).encode() + b':' + a.tobytes())
    elif isinstance(x, list) and x and all(type(v) is int for v in x):
        h.update(b'Li' + str(len(x)).encode() + b':' + ','.join(map(str, x)).encode() + b';')
    elif isinstance(x, (list, tuple)):
        h.update((b'[' if isinstance(x, list) else b'(') + str(len(x)).encode() + b':')
        for v in x:
            _canon(h, v)
        h.update(b']')
    elif isinstance(x, (set, frozenset)):
        h.update(b'<' + str(len(x)).encode() + b':')
        for d in sorted(_digest(v) for v in x):
            h.update(d.encode() + b',')
        h.update(b'>')
    elif isinstance(x, np.ndarray):
        h.update(b'A' + str(x.dtype).encode() + b'|' + str(x.shape).encode() + b'|')
        if x.dtype == object:
            _canon(h, x.tolist())
        else:
            a = np.ascontiguousarray(x)
            if a.dtype.kind in 'fc' and np.isnan(a).any():
                a = a.copy()
                a[np.isnan(a)] = np.nan
            h.update(a.tobytes())
    elif isinstance(x, np.generic):
        h.update(b'G' + str(x.dtype).encode() + b'|')
        _canon(h, x.item())
    else:
        h.update(b'P' + type(x).__name__.encode() + b'|' + pickle.dumps(x, protocol=4))


def _digest(x):
    h = hashlib.sha256()
    _canon(h, x)
    return h.hexdigest()


def run_digest(run):
    return _digest(run)


def offered_problem(a, b, total_ticks):
    """None when the two offered-demand series identify the same instance;
    otherwise the reason they do not."""
    import numpy as np
    out = []
    for name, x in (('the arm pass', a), ('the reference pass', b)):
        if x is None:
            return 'the offered series of %s is missing' % name
        if isinstance(x, (str, bytes, dict, set, frozenset)):
            return 'the offered series of %s is not a numeric series' % name
        try:
            raw = np.asarray(x)
            if raw.dtype.kind not in 'fiu':       # no bools, no objects, no strings
                return 'the offered series of %s is not a numeric series' % name
            v = raw.astype(np.float64)
        except (TypeError, ValueError):
            return 'the offered series of %s is not a numeric series' % name
        if v.ndim != 1 or v.size == 0:
            return 'the offered series of %s is not one-dimensional or is empty' % name
        if isinstance(total_ticks, int) and not isinstance(total_ticks, bool) and v.size != total_ticks:
            return ('the offered series of %s has %d entries, the pass length is %d'
                    % (name, v.size, total_ticks))
        if not np.isfinite(v).all():
            return 'the offered series of %s holds a non-finite value' % name
        if np.unique(v).size < 2:
            return 'the offered series of %s is constant and identifies no instance' % name
        out.append(v)
    if out[0].size != out[1].size:
        return 'the two offered series differ in length (%d, %d)' % (out[0].size, out[1].size)
    if out[0].tobytes() != out[1].tobytes():
        return 'the two offered series differ'
    return None


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _seed_of(key):
    """The integer seed a pickle key denotes, or Refused."""
    import numpy as np
    if isinstance(key, bool):
        raise Refused('seed key %r is not an integer' % (key,))
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, str) and key.isdigit():
        return int(key)
    raise Refused('seed key %r is not an integer' % (key,))


def _runs(P, sk, arm, where=''):
    """{int seed: run} of an arm in one scenario of a pickle (empty runs
    dropped, as the analysis's own loader does).  Two keys that denote the
    same seed are refused: the analysis's loader would silently keep one."""
    out = {}
    block = P.get(sk) if isinstance(P, dict) else None
    per = (block or {}).get(arm) if isinstance(block, dict) else None
    if per is None:
        return out
    if not isinstance(per, dict):
        raise Refused('%s %s %s is not a seed table' % (where, sk, arm))
    seen = set()
    for k, r in per.items():
        s = _seed_of(k)
        if s in seen:
            raise Refused('%s %s %s: two keys denote seed %d' % (where, sk, arm, s))
        seen.add(s)
        if r:
            out[s] = r
    return out


def _arms(P):
    arms = set()
    for sk in SCEN_KEYS:
        block = P.get(sk) if isinstance(P, dict) else None
        if block is None:
            continue
        if not isinstance(block, dict):
            raise Refused('scenario table %s is not a table' % sk)
        arms.update(block)
    return sorted(arms)


def _prov(P, path):
    pr = P.get('provenance') if isinstance(P, dict) else None
    if not isinstance(pr, dict) or not pr:
        raise Refused('%s carries no provenance record' % path)
    env = pr.get('marl_env')
    if env is not None and not isinstance(env, dict):
        raise Refused('%s: marl_env is not a table' % path)
    return pr


def _ckpt_list(pr):
    c = pr.get('checkpoint_sha256')
    if c is None:
        return []
    return [c] if isinstance(c, str) else list(c)


def _prov_view(pr):
    v = {k: pr.get(k) for k in SAME_FIELDS}
    v['marl_env'] = {k: x for k, x in sorted((pr.get('marl_env') or {}).items())
                     if k not in ENV_VOLATILE}
    return v


def _read_keys(path):
    """The expected key set of the assembled family: {(arm, scenario name, seed)}."""
    want = set()
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split('|')
            if len(parts) != 3 or not parts[2].isdigit() or parts[1] not in SCEN_NAMES.values():
                raise Refused('%s: unreadable key %r' % (path, ln))
            key = (parts[0], parts[1], int(parts[2]))
            if key in want:
                raise Refused('%s: key %r occurs twice' % (path, ln))
            want.add(key)
    if not want:
        raise Refused('%s holds no key' % path)
    return want


def _check_locations(comparators, arm_source, out_dir, prefix, record):
    """Refuse every layout in which a write could land on an input, outside
    --out-dir, or on another output."""
    if (not prefix or os.sep in prefix or (os.altsep and os.altsep in prefix)
            or '/' in prefix or '\\' in prefix or '..' in prefix or prefix.startswith('.')):
        raise Refused('--prefix %r must be a plain file-name prefix' % prefix)
    rp = os.path.realpath
    ins = [rp(p) for p in list(comparators) + list(arm_source)]
    od = rp(out_dir)
    if os.path.lexists(out_dir) and (not os.path.isdir(out_dir) or os.listdir(out_dir)):
        raise Refused('--out-dir %s exists and is not an empty directory' % out_dir)
    for p in ins:
        if os.path.dirname(p) == od or p == od or p.startswith(od + os.sep):
            raise Refused('--out-dir %s holds an input (%s)' % (out_dir, p))
    names = [prefix + os.path.basename(p) for p in comparators]
    if len(set(names)) != len(names):
        raise Refused('two comparator pickles have the same file name; their derived '
                      'pickles would be one file')
    outs = [os.path.join(od, n) for n in names]
    if record:
        rr = rp(record)
        if rr in ins or rr in outs or rr == od or rr.startswith(od + os.sep):
            raise Refused('--record %s names an input, an output or a path under --out-dir' % record)
    return names


def assemble(comparators, arm_source, arm, expect_sha, out_dir, prefix, reference_arm='xldet',
             record=None, expect_keys=None, say=print):
    """Returns (ok, record dict).  --out-dir is created only by a run that
    verified everything; a refused run leaves it absent (or empty, as it was)."""
    rec = {'tool': os.path.basename(__file__), 'tool_sha256': file_sha256(os.path.abspath(__file__)),
           'arm': arm, 'reference_arm': reference_arm,
           'expect_checkpoint_sha256': list(expect_sha),
           'comparators': [], 'arm_source': [], 'outputs': [], 'status': 'REFUSED',
           'pass_stats': 'copied with their passes, not verified'}
    fails = []
    private = committed = None
    try:
        if not comparators or not arm_source or not expect_sha:
            raise Refused('no comparator pickle, no arm-source pickle or no expected checkpoint named')
        if any(not p for p in list(comparators) + list(arm_source)):
            raise Refused('an empty path among the inputs')
        if len(set(map(os.path.realpath, comparators))) != len(comparators):
            raise Refused('a comparator pickle is named twice')
        if len(set(map(os.path.realpath, arm_source))) != len(arm_source):
            raise Refused('an arm-source pickle is named twice')
        for p in list(comparators) + list(arm_source):
            if not (os.path.isfile(p) and os.path.getsize(p) > 0):
                raise Refused('missing or empty input %s' % p)
        if reference_arm == arm:
            raise Refused('the reference arm %r is the arm being assembled' % arm)
        names = _check_locations(comparators, arm_source, out_dir, prefix, record)
        want_keys = _read_keys(expect_keys) if expect_keys else None
        in_sha = {p: file_sha256(p) for p in list(comparators) + list(arm_source)}

        # ---- the arm source: {sk: {seed: (run, path)}}
        src = {sk: {} for sk in SCEN_KEYS}
        views = []
        src_pass_stats = {}
        src_checkpoint = None
        for p in arm_source:
            P = _load(p)
            pr = _prov(P, p)
            got = _ckpt_list(pr)
            if got != list(expect_sha):
                fails.append('%s records checkpoint sha256 %s, expected exactly %s'
                             % (p, [str(g)[:12] for g in got], [e[:12] for e in expect_sha]))
            if src_checkpoint is None:
                src_checkpoint = pr.get('checkpoint')
            views.append((p, _prov_view(pr)))
            seeds_here = set()
            for sk in SCEN_KEYS:
                for s, r in _runs(P, sk, arm, p).items():
                    if s in src[sk]:
                        fails.append('arm %s %s seed %d occurs in %s and in %s'
                                     % (arm, sk, s, src[sk][s][1], p))
                    src[sk][s] = (r, p)
                    seeds_here.add(s)
            ps = P.get('pass_stats')
            for k, v in (ps.items() if isinstance(ps, dict) else ()):
                if str(k).split('|')[0] == arm:
                    src_pass_stats[k] = v
            rec['arm_source'].append({'path': p, 'sha256': in_sha[p], 'seeds': sorted(seeds_here),
                                      'checkpoint_sha256': got})
            if not seeds_here:
                fails.append('%s holds no pass of arm %s' % (p, arm))

        # ---- provenance: one code, one schema, one hash seed, one pass length, one engine env
        ref_path, ref_view = views[0]
        for p, v in views[1:]:
            for k in sorted(set(v) | set(ref_view)):
                if v.get(k) != ref_view.get(k):
                    fails.append('provenance %r differs: %s has %r, %s has %r'
                                 % (k, ref_path, ref_view.get(k), p, v.get(k)))
        if any(ref_view.get(k) in (None, '') for k in ('tree_head', 'tree_sha256', 'schema_id',
                                                       'pythonhashseed', 'total_ticks')):
            fails.append('an identifying provenance field is missing in %s' % ref_path)
        total_ticks = ref_view.get('total_ticks')

        # ---- the comparator pickles, ONE AT A TIME: each is loaded, checked,
        #      written as its derived pickle into a private directory and
        #      dropped, so that the memory held is the arm source plus one
        #      comparator pickle whatever the size of the block.  Nothing
        #      leaves the private directory unless every check of every
        #      pickle passed.
        parent = os.path.dirname(os.path.abspath(out_dir))
        os.makedirs(parent, exist_ok=True)
        private = tempfile.mkdtemp(prefix=PRIVATE_PREFIX, dir=parent)
        expect = {}
        seen_seed = {}
        have_keys, per_arm_counts = set(), {}
        n_off = n_cmp = n_arm = 0
        for p, name in zip(comparators, names):
            P = _load(p)
            v = _prov_view(_prov(P, p))
            for k in sorted(set(v) | set(ref_view)):
                if v.get(k) != ref_view.get(k):
                    fails.append('provenance %r differs: %s has %r, %s has %r'
                                 % (k, ref_path, ref_view.get(k), p, v.get(k)))
            arms = _arms(P)
            per_arm = {(sk, a): _runs(P, sk, a, p) for sk in SCEN_KEYS for a in arms}
            seeds = sorted(set(s for a in arms if a != arm for sk in SCEN_KEYS for s in per_arm[sk, a]))
            if not seeds:
                fails.append('%s holds no comparator pass' % p)
            for s in seeds:
                if s in seen_seed:
                    fails.append('seed %d occurs in comparator pickles %s and %s' % (s, seen_seed[s], p))
                seen_seed[s] = p
            if reference_arm not in arms:
                fails.append('%s holds no reference arm %r' % (p, reference_arm))
            rec['comparators'].append({'path': p, 'sha256': in_sha[p], 'seeds': seeds, 'arms': arms})
            # same instance: offered demand of the arm pass == that of the reference comparator
            for sk in SCEN_KEYS:
                refruns = per_arm.get((sk, reference_arm), {})
                for s in seeds:
                    if s not in refruns:
                        fails.append('%s %s seed %d: no %s pass' % (p, sk, s, reference_arm))
                        continue
                    if s not in src[sk]:
                        continue
                    a_run, b_run = src[sk][s][0], refruns[s]
                    why = ('a pass is not a record' if not (isinstance(a_run, dict) and isinstance(b_run, dict))
                           else offered_problem(a_run.get(OFFERED_KEY), b_run.get(OFFERED_KEY), total_ticks))
                    if why:
                        fails.append('%s seed %d: the offered series of arm %s (%s) is not that of '
                                     '%s in %s: %s' % (sk, s, arm, src[sk][s][1], reference_arm, p, why))
                    else:
                        n_off += 1
            # the keys this pickle contributes to the assembled set
            for sk in SCEN_KEYS:
                for a in sorted(set(arms) | {arm}):
                    held = ([s for s in seeds if s in src[sk]] if a == arm else list(per_arm[sk, a]))
                    for s in held:
                        have_keys.add((a, SCEN_NAMES[sk], s))
                        per_arm_counts[a] = per_arm_counts.get(a, 0) + 1
            # the derived pickle, while nothing has failed
            if not fails and all(s in src[sk] for sk in SCEN_KEYS for s in seeds):
                D = dict(P)
                for sk in SCEN_KEYS:
                    D[sk] = dict(P.get(sk) or {})
                    D[sk][arm] = {s: src[sk][s][0] for s in seeds}
                pr = dict(P['provenance'])
                pr['checkpoint_sha256'] = list(expect_sha)
                pr['checkpoint'] = src_checkpoint
                pr['assembled_family'] = {
                    'tool_sha256': rec['tool_sha256'], 'arm': arm,
                    'comparators_from': {'path': p, 'sha256': in_sha[p]},
                    'arm_from': sorted({src[sk][s][1] for sk in SCEN_KEYS for s in seeds}),
                }
                D['provenance'] = pr
                old_ps = P.get('pass_stats')
                ps = {k: x for k, x in (old_ps.items() if isinstance(old_ps, dict) else ())
                      if str(k).split('|')[0] != arm}
                for k, x in src_pass_stats.items():
                    parts = str(k).split('|')
                    if len(parts) == 3 and parts[2].isdigit() and int(parts[2]) in seeds:
                        ps[k] = x
                D['pass_stats'] = ps
                with open(os.path.join(private, name), 'wb') as f:
                    pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
                for sk in SCEN_KEYS:
                    for a in arms:
                        if a == arm:
                            continue
                        for s, r in per_arm[sk, a].items():
                            expect[(name, sk, a, s)] = run_digest(r)
                    for s in seeds:
                        expect[(name, sk, arm, s)] = run_digest(src[sk][s][0])
                del D
            del P, per_arm

        # ---- seed sets: the arm source covers exactly the comparator seeds, in both scenarios
        comp_seeds = set(seen_seed)
        for sk in SCEN_KEYS:
            have = set(src[sk])
            if comp_seeds - have:
                fails.append('arm %s %s: no pass for seeds %s' % (arm, sk, sorted(comp_seeds - have)))
            if have - comp_seeds:
                fails.append('arm %s %s: seeds %s are in no comparator pickle'
                             % (arm, sk, sorted(have - comp_seeds)))
        if not comp_seeds:
            fails.append('the comparator pickles hold no seed')
        if n_off == 0:
            fails.append('no offered-demand pair was compared')

        # ---- completeness: the assembled set is exactly the plan's key set
        rec['runs_per_arm'] = dict(sorted(per_arm_counts.items()))
        if want_keys is not None:
            miss, extra = sorted(want_keys - have_keys), sorted(have_keys - want_keys)
            if miss:
                fails.append('the assembled set lacks %d expected key(s), first %s'
                             % (len(miss), '|'.join(map(str, miss[0]))))
            if extra:
                fails.append('the assembled set holds %d key(s) the plan does not expect, first %s'
                             % (len(extra), '|'.join(map(str, extra[0]))))
            rec['expect_keys'] = {'path': expect_keys, 'sha256': file_sha256(expect_keys),
                                  'n': len(want_keys)}
        else:
            rec['expect_keys'] = None      # completeness NOT checked here (the caller's key gate does)
        if fails:
            raise _Collected()

        # ---- verify every derived pickle from disk, then commit with one rename
        out_sha = {}
        for name in names:
            t = os.path.join(private, name)
            # hashed BEFORE it is loaded: what is verified below is the file
            # with this digest, and the committed file must have it too
            out_sha[name] = file_sha256(t)
            D = _load(t)
            keys_here = set()
            for sk in SCEN_KEYS:
                for a in _arms(D):
                    for s, r in _runs(D, sk, a, t).items():
                        key = (name, sk, a, s)
                        keys_here.add(key)
                        if key not in expect:
                            fails.append('derived %s holds an unexpected run %s' % (name, key[1:]))
                        elif run_digest(r) != expect[key]:
                            fails.append('derived %s: run %s differs from its input' % (name, key[1:]))
                        elif a == arm:
                            n_arm += 1
                        else:
                            n_cmp += 1
            want_here = {k for k in expect if k[0] == name}
            if want_here - keys_here:
                fails.append('derived %s lacks %d run(s)' % (name, len(want_here - keys_here)))
            if _ckpt_list(D.get('provenance') or {}) != list(expect_sha):
                fails.append('derived %s does not record the expected checkpoint list' % name)
            if file_sha256(t) != out_sha[name]:
                fails.append('derived %s changed while it was being verified' % name)
        if n_cmp == 0 or n_arm == 0:
            fails.append('nothing was verified (comparator runs %d, arm runs %d)' % (n_cmp, n_arm))
        if n_arm != 2 * len(comp_seeds):
            fails.append('arm runs verified %d, expected %d (2 scenarios x %d seeds)'
                         % (n_arm, 2 * len(comp_seeds), len(comp_seeds)))
        for p, h in in_sha.items():
            if file_sha256(p) != h:
                fails.append('input %s changed while it was being assembled' % p)
        if fails:
            raise _Collected()
        if os.path.lexists(out_dir):
            os.rmdir(out_dir)                 # an EMPTY directory, checked above; anything else raises
        os.rename(private, out_dir)
        private, committed = None, out_dir
        for name in names:
            o = os.path.join(out_dir, name)
            if file_sha256(o) != out_sha[name]:
                fails.append('output %s is not the file that was verified' % o)
            rec['outputs'].append({'path': o, 'sha256': out_sha[name]})
        if fails:
            raise _Collected()
    except BaseException as e:
        if private and os.path.isdir(private):
            shutil.rmtree(private, ignore_errors=True)
        # a set that was committed and then found not to be the verified one
        # is not left behind either: after a refusal --out-dir does not exist
        if committed and os.path.isdir(committed):
            shutil.rmtree(committed, ignore_errors=True)
        if not isinstance(e, Exception):
            raise                              # an interrupt: cleaned up, then passed on
        if isinstance(e, Refused):
            fails.append(str(e))
        elif not isinstance(e, _Collected):
            fails.append('%s: %s' % (type(e).__name__, e))
        if not fails:
            fails.append('refused without a recorded reason')
        rec['outputs'] = []
        return False, dict(rec, failures=fails)

    rec.update(status='PASS', seeds=sorted(comp_seeds),
               verify={'comparator_runs': n_cmp, 'arm_runs': n_arm, 'offered_pairs': n_off,
                       'derived_pickles': len(names)},
               provenance=ref_view)
    say('ASSEMBLE VERIFY PASS arm=%s derived=%d comparator_runs=%d arm_runs=%d offered_pairs=%d seeds=%d%s'
        % (arm, len(names), n_cmp, n_arm, n_off, len(comp_seeds),
           (' expected_keys=%d' % len(want_keys)) if want_keys is not None else ' expected_keys=unchecked'))
    return True, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--comparators', required=True, metavar='PKL[,PKL]')
    ap.add_argument('--arm-source', required=True, metavar='PKL[,PKL]')
    ap.add_argument('--arm', default='marl_static')
    ap.add_argument('--reference-arm', default='xldet')
    ap.add_argument('--expect-checkpoint-sha256', required=True, metavar='SHA[,SHA]')
    ap.add_argument('--expect-keys', default=None, metavar='FILE',
                    help='the expected key set of the assembled family (arm|scenario|seed per line)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix', default='e_')
    ap.add_argument('--record', required=True, metavar='JSON')
    a = ap.parse_args(argv)
    comps = a.comparators.split(',')
    srcs = a.arm_source.split(',')
    shas = a.expect_checkpoint_sha256.split(',')
    if '' in comps or '' in srcs or '' in shas or not a.prefix:
        ap.error('an empty element in --comparators, --arm-source or --expect-checkpoint-sha256, or an empty --prefix')
    def write(rec):
        with open(a.record, 'w') as f:
            json.dump(rec, f, indent=1, sort_keys=True, default=str)
    rp = os.path.realpath
    od = rp(a.out_dir)
    if rp(a.record) in [rp(p) for p in comps + srcs] or rp(a.record) == od \
            or rp(a.record).startswith(od + os.sep):
        print('ASSEMBLE REFUSED: --record %s names an input or a path under --out-dir' % a.record)
        return 1
    # the record of THIS run replaces any older one before anything else happens
    os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
    write({'status': 'REFUSED', 'failures': ['the run did not finish'], 'outputs': []})

    # A TERMINATION is an interrupt: without this, SIGTERM (the watchdog, an
    # operator) kills the process where it stands and the private directory
    # with the derived pickles written so far stays on disk.  As an exception
    # it passes through assemble()'s clean-up and then ends the run.
    def _terminated(signum, _frame):
        raise KeyboardInterrupt('signal %d' % signum)
    try:
        signal.signal(signal.SIGTERM, _terminated)
    except (ValueError, OSError):              # not the main thread: nothing to install
        pass
    try:
        ok, rec = assemble(comps, srcs, a.arm, shas, a.out_dir, a.prefix, a.reference_arm,
                           record=a.record, expect_keys=a.expect_keys)
    except Exception as e:                     # a malformed input is a refusal, with a record
        ok, rec = False, {'status': 'REFUSED', 'outputs': [],
                          'failures': ['%s: %s' % (type(e).__name__, e)]}
    write(rec)
    if not ok:
        for m in rec.get('failures', []):
            print('ASSEMBLE REFUSED: %s' % m)
        print('ASSEMBLE REFUSED: %d reason(s); nothing was written to %s'
              % (len(rec.get('failures', [])), a.out_dir))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
