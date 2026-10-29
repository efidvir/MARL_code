#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chain ledger, per-invocation gates, and the timing report (WP RUN-CHAIN).

Every mode here COUNTS what it looked at and FAILS when that count is zero.
The recurring defect of this campaign is a check that reports success without
having verified anything, so "no keys found" and "no pickles compared" are
failures, never quiet passes.

MODES
  --ledger-append LEDGER --kv k=v ...     append one JSONL record
  --ledger-status LEDGER --inv ID         latest verdict for an invocation
                                          prints: STATUS sha256 attempts
  --verify PKL --keys-file F              key-completeness gate for one pickle
  --key-union PKL[,PKL] --keys-file F     union gate across an entire stage
  --cross-provenance PKL[,PKL]            tree_head / tree_sha256 / schema_id /
                                          pythonhashseed identical everywhere
  --check-marl-env ART --expect-marl-env-file F
                                          the MARL_* environment ART recorded
                                          (provenance marl_env of a pickle or
                                          a checkpoint) is EXACTLY the plan's
                                          (F: one K=V per line, written by
                                          chain_write_env); --verify takes
                                          --expect-marl-env-file too
  --hash-pair A,B [--expect-hashseeds 0,1] the analysis-S8 pair: same tree,
                                          schema and checkpoint, the two
                                          DECLARED hash seeds (counts the
                                          fields compared; zero is a FAIL)
  --timings LEDGER [--profile P]          measured wall clock per stage, the
          [--run last|all|ok]             measured CPU-s per pass per arm, and
                                          the extrapolation to the real run.
                                          --run scopes to ONE chain run:
                                          last (default) = since the last
                                          chain_start, ok = the last run that
                                          ended rc=0, all = the whole ledger.
  --ledger-tree-pin LEDGER                tree_sha256 of this run's FIRST
                                          invocation, or '-' -- what preflight
                                          pins the tree to before FREEZE_FINAL
  --ablation-diff PKL --baseline PKL[,..] an ablation must DIFFER from the run
          --ablation-series a,b           it is an ablation of.  Counts the
          [--ablation-counter c]          passes that differ on the named
          [--ablation-arms a,b]           series and FAILS on zero; exit 3 =
                                          NOT EXERCISED (the counter is zero in
                                          the baseline too, so there was
                                          nothing to switch off)
  --verify ... --expect-checkpoint-sha256 SHA[,SHA]
                                          R3-lite: the pickle must RECORD the
                                          pinned checkpoint digest(s)
  --paired-descriptive PKL[,PKL] --baseline PKL[,..] --paired-seeds a,b,..
          [--paired-arm marl_static] [--paired-ref-arm xldet]
          [--paired-label NAME] [--expect-checkpoint-sha256 ..] --json-out F
                                          R3-lite: DESCRIPTIVE paired
                                          comparison (the ensemble against
                                          XL-DET) with the pre-registered
                                          metric helpers; fails unless it
                                          compared exactly the declared seeds
                                          in both scenarios
  --audit-gate REPORT --freeze-file F --expect-window LO,HI
          --expect-planning-marker PATH --tree T [--audit-ledger LEDGER]
                                          R3-lite: the freeze-2 seed audit as
                                          a gate before the first confirmatory
                                          pass: the report must be the audit
                                          of this sealed tree with zero
                                          result-data hits (exit 3: hits left
                                          to classify)

Exit status is 0 only when every requested gate passed.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCEN_KEYS = (('recovery_multi', 'recovery'), ('rescue_multi', 'rescue_ops'))
# The only arms COMPUTE_PLAN.md has a MEASURED 3600-tick per-pass cost for.
MEASURED_AT_3600 = ('marl_static', 'xldet', 'ospf')
# Per-pass cost is affine in the tick count, and the fixed part belongs to the
# CODE PATH, not to the arm: every routing/SDN arm builds the same topology and
# flow table, and every marl-family arm runs the same observation/actuation
# stack.  So each family is anchored on the arm of that family the plan
# measured at 3600 ticks, instead of one global intercept that fits none.
FAMILY = {
    'ospf': 'routing', 'sdn': 'routing', 'olsr': 'routing',
    'batman': 'routing', 'aodv': 'routing',
    'marl_static': 'marl', 'marl_freeze': 'marl', 'marl': 'marl',
    'random': 'marl', 'xldet': 'xldet',
}
FAMILY_ANCHOR = {'routing': 'ospf', 'marl': 'marl_static', 'xldet': 'xldet'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def present_keys(P):
    """(arm|scenario|seed) keys whose pass produced data.

    A pass that failed inside the pool leaves either no entry or a falsy one;
    both must count as MISSING, otherwise an invocation that silently lost
    half its passes would look complete.
    """
    keys = set()
    for sk, sname in SCEN_KEYS:
        for arm, per_seed in (P.get(sk) or {}).items():
            for seed, rec in (per_seed or {}).items():
                if rec:
                    keys.add('%s|%s|%d' % (arm, sname, int(seed)))
    return keys


def read_keys_file(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_marl_env_file(path):
    """The MARL_* set chain_write_env declares for an invocation."""
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                k, _, v = ln.partition('=')
                out[k] = v
    return out


def recorded_provenance(path):
    """The provenance record of a result pickle (top level) or of a
    checkpoint (metadata.provenance, or __provenance__)."""
    try:
        obj = load_pickle(path)
    except Exception:
        import torch                      # a torch checkpoint (zip format)
        obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        return None
    for get in (lambda o: o.get('provenance'),
                lambda o: (o.get('metadata') or {}).get('provenance'),
                lambda o: o.get('__provenance__')):
        p = get(obj)
        if isinstance(p, dict):
            return p
    return None


def marl_env_verdict(prov, expected):
    """(ok, n_compared, problems): the recorded marl_env must equal the
    plan's MARL_* set exactly -- no stray variable, none missing, no value
    different.  Nothing to compare is a FAILURE (FIX-PREFREEZE item 5)."""
    problems = []
    if not expected:
        return False, 0, ['the expected MARL_* set is EMPTY: nothing to compare']
    got = (prov or {}).get('marl_env')
    if not isinstance(got, dict):
        return False, 0, ['no marl_env in the provenance record']
    n = 0
    for k in sorted(set(got) | set(expected)):
        n += 1
        if k not in expected:
            problems.append('STRAY %s=%s (not in the plan)' % (k, got[k]))
        elif k not in got:
            problems.append('MISSING %s (the plan sets %s)' % (k, expected[k]))
        elif str(got[k]) != str(expected[k]):
            problems.append('DIFFERENT %s=%s (the plan sets %s)' % (k, got[k], expected[k]))
    return (n > 0 and not problems), n, problems


def cmd_check_marl_env(a):
    if not a.expect_marl_env_file or not os.path.exists(a.check_marl_env):
        print('[MARL-ENV] FAIL %s: missing artefact or no --expect-marl-env-file'
              % a.check_marl_env)
        return 1
    ok, n, problems = marl_env_verdict(recorded_provenance(a.check_marl_env),
                                       read_marl_env_file(a.expect_marl_env_file))
    print('[MARL-ENV] %s %s: %d variable(s) compared with the plan, %d problem(s)'
          % ('PASS' if ok else 'FAIL', a.check_marl_env, n, len(problems)))
    for p in problems[:20]:
        print('           %s' % p)
    return 0 if ok else 1


def cmd_verify(a):
    exp = read_keys_file(a.keys_file)
    if not exp:
        print('[VERIFY] FAIL %s: the expected-key list is EMPTY, so this gate '
              'would have checked nothing' % a.verify)
        return 1
    if not os.path.exists(a.verify):
        print('[VERIFY] FAIL %s: pickle does not exist' % a.verify)
        return 1
    if os.path.getsize(a.verify) == 0:
        print('[VERIFY] FAIL %s: pickle is empty (0 bytes)' % a.verify)
        return 1
    P = load_pickle(a.verify)
    got = present_keys(P)
    exp_set = set(exp)
    missing = sorted(exp_set - got)
    extra = sorted(got - exp_set)
    prov = P.get('provenance') or {}
    stats = P.get('pass_stats') or {}
    ok = True
    if not got:
        ok = False
    if missing or extra:
        ok = False
    # The hash seed the PLAN declares for this invocation: '0' everywhere
    # but the analysis-S8 repeat (S6h), which must record '1'.
    if prov.get('pythonhashseed') != str(a.expect_hashseed):
        ok = False
    if not prov.get('tree_sha256'):
        ok = False
    # The run's recorded MARL_* environment is EXACTLY the plan's (item 5).
    env_n, env_problems = None, []
    if a.expect_marl_env_file:
        env_ok, env_n, env_problems = marl_env_verdict(
            prov, read_marl_env_file(a.expect_marl_env_file))
        ok = ok and env_ok
    # R3-lite: the checkpoint digest(s) the PLAN pins for the invocation
    # (member by member, in MARL_EVAL_CHECKPOINT order) are the ones the run
    # RECORDED opening (provenance checkpoint_sha256).  Only with the flag;
    # a pin with nothing recorded to compare is a FAILURE.
    pin_want, pin_got, pin_ok = None, None, None
    if a.expect_checkpoint_sha256:
        pin_want = [x for x in a.expect_checkpoint_sha256.split(',') if x]
        raw = prov.get('checkpoint_sha256')
        pin_got = ([raw] if isinstance(raw, str) else list(raw or []))
        pin_ok = bool(pin_want) and pin_got == pin_want
        ok = ok and pin_ok
    print('[VERIFY] %s %s  expected=%d present=%d missing=%d extra=%d '
          'pass_stats=%d hashseed=%r (plan %r) head=%s tree=%s'
          % ('PASS' if ok else 'FAIL', a.verify, len(exp_set), len(got),
             len(missing), len(extra), len(stats),
             prov.get('pythonhashseed'), str(a.expect_hashseed),
             prov.get('tree_head'),
             (prov.get('tree_sha256') or '')[:16]))
    if env_n is not None:
        print('         marl_env: %d variable(s) compared with the plan, %d problem(s)'
              % (env_n, len(env_problems)))
        for p in env_problems[:12]:
            print('         ENV %s' % p)
    if pin_ok is not None:
        print('         checkpoint_sha256: %d recorded, %d pinned -> %s'
              % (len(pin_got), len(pin_want), 'MATCH' if pin_ok else 'MISMATCH'))
        if not pin_ok:
            print('         PINNED   %s' % ','.join(pin_want))
            print('         RECORDED %s' % ','.join(str(x) for x in pin_got))
    for k in missing[:12]:
        print('         MISSING %s' % k)
    for k in extra[:12]:
        print('         UNEXPECTED %s' % k)
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump({'pickle': a.verify, 'expected': len(exp_set),
                       'present': len(got), 'missing': missing,
                       'extra': extra, 'ok': ok,
                       'provenance': {k: prov.get(k) for k in
                                      ('tree_head', 'tree_sha256', 'schema_id',
                                       'pythonhashseed')}}, f, indent=1)
    return 0 if ok else 1


def cmd_key_union(a):
    paths = [p for p in a.key_union.split(',') if p]
    exp = set(read_keys_file(a.keys_file))
    if not paths or not exp:
        print('[KEY-UNION] FAIL: %d pickles, %d expected keys -- nothing to check'
              % (len(paths), len(exp)))
        return 1
    union, dupes, per = set(), [], {}
    for p in paths:
        if not os.path.exists(p):
            print('[KEY-UNION] FAIL: missing pickle %s' % p)
            return 1
        k = present_keys(load_pickle(p))
        per[p] = len(k)
        dupes.extend(sorted(union & k))
        union |= k
    missing = sorted(exp - union)
    extra = sorted(union - exp)
    ok = bool(union) and not missing and not extra and not dupes
    print('[KEY-UNION] %s pickles=%d expected=%d union=%d missing=%d extra=%d '
          'duplicated=%d' % ('PASS' if ok else 'FAIL', len(paths), len(exp),
                             len(union), len(missing), len(extra), len(dupes)))
    for p in paths:
        print('            %-56s %d keys' % (p, per[p]))
    for k in (missing[:12] + dupes[:12]):
        print('            OFFENDING %s' % k)
    return 0 if ok else 1


def cmd_cross_provenance(a):
    paths = [p for p in a.cross_provenance.split(',') if p]
    if not paths:
        print('[PROVENANCE] FAIL: no pickles given')
        return 1
    fields = ('tree_head', 'tree_sha256', 'schema_id', 'pythonhashseed')
    seen, bad = {}, []
    for p in paths:
        if not os.path.exists(p):
            print('[PROVENANCE] FAIL: missing pickle %s' % p)
            return 1
        prov = (load_pickle(p).get('provenance') or {})
        sig = tuple(prov.get(f) for f in fields)
        seen.setdefault(sig, []).append(p)
        if prov.get('pythonhashseed') != '0' or not prov.get('tree_sha256'):
            bad.append(p)
    ok = len(seen) == 1 and not bad
    print('[PROVENANCE] %s pickles=%d distinct_signatures=%d'
          % ('PASS' if ok else 'FAIL', len(paths), len(seen)))
    for sig, ps in seen.items():
        print('             head=%s tree=%s schema=%s hashseed=%s  <- %d pickle(s)'
              % (sig[0], (sig[1] or '')[:16], sig[2], sig[3], len(ps)))
    if len(paths) == 1:
        print('             NOTE: one pickle only -- signature recorded, no '
              'cross-check was possible')
    return 0 if ok else 1


HASH_PAIR_SAME = ('tree_head', 'tree_sha256', 'schema_id', 'checkpoint_sha256')


def cmd_hash_pair(a):
    """The analysis-S8 pair must differ in the hash seed and in NOTHING
    else that could explain a difference: the same tree, schema id and
    checkpoint.  tools/hash_invariance_check.py requires the two seeds to
    differ; this requires them to be the two the PLAN declares, and the rest
    to be equal and present.  Counts the fields it compared; zero is a FAIL.
    """
    paths = [p for p in a.hash_pair.split(',') if p]
    want = [x for x in (a.expect_hashseeds or '').split(',') if x]
    if len(paths) != 2 or len(want) != 2:
        print('[HASH-PAIR] FAIL: need two pickles and two declared hash seeds, '
              'got %d and %d' % (len(paths), len(want)))
        return 1
    provs = []
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print('[HASH-PAIR] FAIL: missing or empty pickle %s' % p)
            return 1
        provs.append(load_pickle(p).get('provenance') or {})
    bad, n = [], 0
    for f in HASH_PAIR_SAME:
        x, y = provs[0].get(f), provs[1].get(f)
        n += 1
        if not x or not y or x != y:
            bad.append('%s: %r vs %r' % (f, x, y))
    got = [provs[0].get('pythonhashseed'), provs[1].get('pythonhashseed')]
    n += 1
    if got != want or got[0] == got[1]:
        bad.append('pythonhashseed: %r, the plan declares %r' % (got, want))
    ok = n > 0 and not bad
    print('[HASH-PAIR] %s %s vs %s: %d provenance fields compared, %d differ '
          '(hash seeds %s)' % ('PASS' if ok else 'FAIL', paths[0], paths[1], n,
                               len(bad), '/'.join(str(x) for x in got)))
    for x in bad:
        print('            %s' % x)
    return 0 if ok else 1


# ------------------------------------------------- R3-lite: ensemble vs XL-DET
PAIRED_SCEN = (('A', 'recovery_multi'), ('B', 'rescue_multi'))
PAIRED_SAME = ('tree_head', 'tree_sha256', 'schema_id', 'pythonhashseed')


def _arm_runs(paths, arm):
    """({'A': {seed: run}, 'B': {...}}, [provenance per pickle], problems)
    for one arm, merged over pickles.  Only that arm's runs are kept in
    memory (an eight-arm confirmatory pickle is large).  A seed present in
    two pickles is a PROBLEM, not a silent overwrite."""
    runs = {'A': {}, 'B': {}}
    provs, problems = [], []
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            problems.append('missing or empty pickle %s' % p)
            continue
        P = load_pickle(p)
        provs.append(dict(P.get('provenance') or {}, _path=p))
        for k, key in PAIRED_SCEN:
            for seed, r in ((P.get(key) or {}).get(arm) or {}).items():
                if not r:
                    continue
                if int(seed) in runs[k]:
                    problems.append('%s|%s|%d is in more than one pickle' % (arm, k, int(seed)))
                runs[k][int(seed)] = r
        del P
    return runs, provs, problems


def cmd_paired_descriptive(a):
    """DESCRIPTIVE paired comparison of one arm of one set of pickles (the
    R3-lite ensemble, chain S9x) with a reference arm of another set (the
    XL-DET passes of S6) on the SAME instances, both scenarios:
    achievability, energy per delivered unit and relay-link teardowns (M7).

    It is OUTSIDE every Holm family and decides no hypothesis
    (docs/PREREGISTRATION_R3.md section 10).  The metric helpers and the
    paired statistics are IMPORTED from confirmatory_analysis.py -- the
    pre-registered definitions, never re-implemented here.

    It counts what it compared and FAILS when that is not exactly the
    expected set: the seeds the plan declares, in both scenarios, in both
    arms, at the nominal run length, on one tree, with the pinned
    checkpoint digests recorded by the test pickles."""
    test_paths = [p for p in (a.paired_descriptive or '').split(',') if p]
    ref_paths = [p for p in (a.baseline or '').split(',') if p]
    want = sorted(int(s) for s in (a.paired_seeds or '').split(',') if s.strip())
    label = a.paired_label
    if not test_paths or not ref_paths or not want:
        print('[PAIRED] FAIL %s: %d test pickle(s), %d reference pickle(s), %d '
              'expected seed(s) -- nothing to compare'
              % (label, len(test_paths), len(ref_paths), len(want)))
        return 1
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import confirmatory_analysis as CA      # the pre-registered definitions

    test, tprov, problems = _arm_runs(test_paths, a.paired_arm)
    ref, rprov, more = _arm_runs(ref_paths, a.paired_ref_arm)
    problems += more

    # one tree, one schema, hash seed 0, across BOTH sets
    sigs = {}
    for pr in tprov + rprov:
        sigs.setdefault(tuple(pr.get(f) for f in PAIRED_SAME), []).append(pr['_path'])
    if len(sigs) != 1:
        problems.append('%d distinct provenance signatures across the two sets: %s'
                        % (len(sigs), sorted(sigs.values())))
    for pr in tprov + rprov:
        if pr.get('pythonhashseed') != '0' or not pr.get('tree_sha256'):
            problems.append('provenance of %s: pythonhashseed=%r tree_sha256=%r'
                            % (pr['_path'], pr.get('pythonhashseed'),
                               (pr.get('tree_sha256') or '')[:16]))
    # the pinned checkpoint digests, recorded by every test pickle
    pins = [x for x in (a.expect_checkpoint_sha256 or '').split(',') if x]
    recorded = None
    for pr in tprov:
        raw = pr.get('checkpoint_sha256')
        got = [raw] if isinstance(raw, str) else list(raw or [])
        recorded = got if recorded is None else recorded
        if pins and got != pins:
            problems.append('%s records checkpoint_sha256 %s, the plan pins %s'
                            % (pr['_path'], got, pins))
        if got != recorded:
            problems.append('the test pickles record different checkpoints')

    metrics = (('ach', +1, lambda r, k: CA.steady(r['conn'])),
               ('jpu', -1, lambda r, k: CA.jpu(r)),
               ('m7', +1, lambda r, k: CA.m7_teardowns_per_1000(r, CA.T_SEV[k])))
    rep = {'label': label, 'descriptive': True,
           'note': ('DESCRIPTIVE ONLY: outside every Holm family, decides no '
                    'hypothesis; p-values are unadjusted and reported for '
                    'description'),
           'arm': a.paired_arm, 'ref_arm': a.paired_ref_arm,
           'difference': '%s[%s] minus %s' % (a.paired_arm, label, a.paired_ref_arm),
           'expected_seeds': want, 'test_pickles': test_paths,
           'ref_pickles': ref_paths, 'checkpoint_sha256': recorded,
           'checkpoint_sha256_pinned': pins or None,
           'provenance': {f: (tprov[0].get(f) if tprov else None) for f in PAIRED_SAME},
           'T_TOTAL': CA.T_TOTAL, 'scenarios': {}}
    n_pairs = 0
    for k, _key in PAIRED_SCEN:
        ts, rs = sorted(test[k]), sorted(ref[k])
        if ts != want or rs != want:
            problems.append('scenario %s: %s has seeds %s, %s has seeds %s, the '
                            'plan declares %s' % (k, a.paired_arm, ts,
                                                  a.paired_ref_arm, rs, want))
            continue
        short = [(s, len(x[s]['conn'])) for x in (test[k], ref[k]) for s in want
                 if len(x[s]['conn']) != CA.T_TOTAL]
        if short:
            problems.append('scenario %s: run length is not %d ticks: %s'
                            % (k, CA.T_TOTAL, short[:6]))
            continue
        blk = {'n': len(want), 'seeds': want}
        for name, expect, fn in metrics:
            tv = [fn(test[k][s], k) for s in want]
            rv = [fn(ref[k][s], k) for s in want]
            if any(v is None for v in tv + rv):
                blk[name] = 'not recorded'
                continue
            d = [x - y for x, y in zip(tv, rv)]
            st_ = CA.paired(d, expect)
            blk[name] = {
                'mean_%s' % a.paired_arm: sum(tv) / len(tv),
                'mean_%s' % a.paired_ref_arm: sum(rv) / len(rv),
                'paired': {q: st_[q] for q in ('n', 'mean', 'sd', 'ci_lo', 'ci_hi',
                                               'wins', 'sign_p', 'wilcoxon_p',
                                               'n_nonzero', 'expect')},
                'per_seed_difference': {str(s): x for s, x in zip(want, d)},
            }
        rep['scenarios'][k] = blk
        n_pairs += len(want)
    rep['n_pairs'] = n_pairs
    rep['problems'] = problems
    ok = not problems and n_pairs == 2 * len(want) and n_pairs > 0
    rep['ok'] = ok
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump(rep, f, indent=1, sort_keys=True, default=str)
    print('[PAIRED] %s %s: %s minus %s, %d instance pair(s) over %d expected '
          'seed(s) x 2 scenarios, %d problem(s)  (DESCRIPTIVE, no Holm family)'
          % ('PASS' if ok else 'FAIL', label, a.paired_arm, a.paired_ref_arm,
             n_pairs, len(want), len(problems)))
    for k in sorted(rep['scenarios']):
        for name in ('ach', 'jpu', 'm7'):
            b = rep['scenarios'][k].get(name)
            if isinstance(b, dict):
                p = b['paired']
                print('         %s %-3s  mean diff %+.4f  CI [%+.4f, %+.4f]  wins %d/%d  '
                      'W %.5f' % (k, name, p['mean'], p['ci_lo'], p['ci_hi'],
                                  p['wins'], p['n'], p['wilcoxon_p']))
            else:
                print('         %s %-3s  %s' % (k, name, b))
    for x in problems[:12]:
        print('         PROBLEM %s' % x)
    return 0 if ok else 1


def _series_of(P, arms, scen_key, series):
    """{(arm, scenario, seed): {series_name: value}} for the named series."""
    out = {}
    for sk, sname in SCEN_KEYS:
        for arm, per_seed in (P.get(sk) or {}).items():
            if arms and arm not in arms:
                continue
            for seed, rec in (per_seed or {}).items():
                if not rec:
                    continue
                out[(arm, sname, int(seed))] = dict(
                    (name, rec.get(name)) for name in series)
    return out


def cmd_ablation_diff(a):
    """An ABLATION invocation must differ from the run it is an ablation of.

    Counting the ablation's own keys proves it RAN.  It does not prove the
    switch took: in the rehearsal, s11 (MARL_RELAY_REPOINT=off) came out
    byte-identical to its baseline on all 45 series of all 8 passes and every
    gate in the chain reported it complete.  This gate compares them.

    exit 0  the ablation moved the series it is supposed to move
    exit 3  NOT EXERCISED -- identical, AND the declared counter is zero in the
            baseline as well, so the mechanism never fired and there was
            nothing to switch off.  Valid data; no evidence.
    exit 1  identical although the mechanism DID fire in the baseline (the
            switch did not take), or nothing could be compared at all.
    """
    series = [x.strip() for x in (a.ablation_series or '').split(',') if x.strip()]
    arms = {x.strip() for x in (a.ablation_arms or '').split(',') if x.strip()}
    bases = [x for x in (a.baseline or '').split(',') if x]
    if not series or not bases:
        print('[ABLATION] FAIL: %d series, %d baseline pickle(s) -- nothing to '
              'compare' % (len(series), len(bases)))
        return 1
    if not os.path.exists(a.ablation_diff):
        print('[ABLATION] FAIL: %s does not exist' % a.ablation_diff)
        return 1
    abl = _series_of(load_pickle(a.ablation_diff), arms, None, series)
    base = {}
    counter_total = 0.0
    for b in bases:
        if not os.path.exists(b):
            print('[ABLATION] FAIL: baseline %s does not exist' % b)
            return 1
        P = load_pickle(b)
        base.update(_series_of(P, arms, None, series))
        if a.ablation_counter:
            for v in _series_of(P, arms, None, [a.ablation_counter]).values():
                x = v.get(a.ablation_counter)
                if isinstance(x, list):
                    counter_total += sum(y for y in x
                                         if isinstance(y, (int, float)))
                elif isinstance(x, (int, float)):
                    counter_total += x
    shared = sorted(set(abl) & set(base))
    if not shared:
        print('[ABLATION] FAIL: %s and its baseline share NO (arm, scenario, '
              'seed) key; nothing was compared' % a.ablation_diff)
        return 1
    n_diff = 0
    moved = {}
    for k in shared:
        hit = [nm for nm in series if abl[k].get(nm) != base[k].get(nm)]
        if hit:
            n_diff += 1
            for nm in hit:
                moved[nm] = moved.get(nm, 0) + 1
    verdict = 'PASS' if n_diff else ('NOT EXERCISED'
                                     if a.ablation_counter and not counter_total
                                     else 'FAIL')
    print('[ABLATION] %s %s vs %d baseline pickle(s): compared=%d passes, '
          'series=%d, differing=%d %s'
          % (verdict, a.ablation_diff, len(bases), len(shared), len(series),
             n_diff, dict(sorted(moved.items())) or ''))
    if a.ablation_counter:
        print('           baseline total of %r (the mechanism this ablation '
              'switches off) = %g' % (a.ablation_counter, counter_total))
    if n_diff:
        return 0
    if a.ablation_counter and not counter_total:
        print('           The ablated run is identical to its baseline AND the '
              'mechanism never fired in the baseline either, so this '
              'sensitivity tested nothing.  Recorded as NOT EXERCISED: the '
              'passes are valid, the ablation is not evidence.')
        return 3
    print('           The ablated run is identical to its baseline although the '
          'mechanism DID run there.  The switch did not take.')
    return 1


def _utc_stamp(s):
    """Seconds since the epoch of a UTC stamp as the chain ('2026-10-11T08:00:00Z')
    or the seed audit ('2026-10-11 08:00:00 UTC') writes it; None otherwise."""
    import calendar
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S UTC'):
        try:
            return calendar.timegm(time.strptime(str(s).strip(), fmt))
        except ValueError:
            pass
    return None


def cmd_audit_gate(a):
    """R3-lite: the freeze-2 seed audit as a gate of the launcher.

    docs/PREREGISTRATION_R3.md section 13 makes the audit a STOP CONDITION
    ("any result-data hit stops the run before S6"), and nothing in the chain
    used to read its report: `run_final.sh --profile r3lite` without --stages
    ran S0 to S12 straight through.  This reads the report the committed
    driver wrote (tools/run_seed_audit.sh, SEED_AUDIT_GATE=r3-freeze2) and
    passes only when it is THE audit of THIS sealed tree: the plan's block,
    the result-data gate requested and passed, full coverage, the freeze-2
    planning marker (this run's own S0 manifest), started after the seal, on
    the sealed head, from this tree.

    Exit 0: the report is clean, or its remaining text, file-name and history
    hits were acknowledged as classified by a recorded operator action for
    exactly this report (`run_final.sh --audit-ok`, a ledger note carrying
    the report's sha256).  Exit 3: valid, result-data gate passed, but hits
    remain unclassified.  Exit 1: anything else.  Every check is counted; a
    run that checked nothing fails.  No acknowledgement can turn a failed
    result-data gate or a coverage failure into a pass."""
    checks = []

    def chk(name, ok, detail=''):
        checks.append({'check': name, 'ok': bool(ok), 'detail': str(detail)})
        print('[AUDIT-GATE] %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                                          ('  (%s)' % detail) if detail else ''))

    # ---- the seal the audit must follow
    sealed_t = tree_head = sealed_sha = None
    lines = []
    try:
        with open(a.freeze_file) as f:
            lines = f.read().splitlines()
    except (OSError, TypeError):
        pass
    for ln in lines:
        parts = ln.split()
        if parts[:1] == ['FREEZE_FINAL'] and 'sealed' in parts[:-1]:
            sealed_t = _utc_stamp(parts[parts.index('sealed') + 1])
        elif parts[:1] == ['tree_head'] and len(parts) > 1:
            tree_head = parts[1]
        elif parts[:1] == ['tree_sha256'] and len(parts) > 1:
            sealed_sha = parts[1]
    chk('freeze 2 is sealed: seal time, tree_head and tree_sha256 read',
        sealed_t is not None and bool(tree_head) and bool(sealed_sha), a.freeze_file)

    # ---- the report
    rep, err = None, ''
    try:
        with open(a.audit_gate) as f:
            rep = json.load(f)
    except (OSError, ValueError) as e:
        err = str(e)[:120]
    chk('the audit report exists and parses', isinstance(rep, dict),
        a.audit_gate + (': ' + err if err else ''))
    if not isinstance(rep, dict):
        rep = {}
    v = rep.get('verdict') or {}
    req = rep.get('requested') or {}
    g = v.get('result_data_gate') or {}
    try:
        want = [int(x) for x in (a.expect_window or '').split(',')]
    except ValueError:
        want = []
    chk('it audited the block of the plan', len(want) == 2 and rep.get('window') == want
        and g.get('window') == want, '%s vs %s' % (rep.get('window'), want))
    chk('the result-data gate was requested and passed: zero seeds of the block in '
        'any pickle, non-planning JSON file or checkpoint',
        req.get('gate_result_data') is True and g.get('pass') is True
        and not g.get('files_with_result_data_hits')
        and v.get('window_hits_in_result_data') == [],
        'pass=%r, %d file(s) with hits' % (g.get('pass'), len(g.get('files_with_result_data_hits') or [])))
    chk('coverage was complete', v.get('coverage_ok') is True and not v.get('coverage_failures'),
        '%d coverage failure(s)' % len(v.get('coverage_failures') or []))
    chk("it is the freeze-2 audit: its planning markers name this run's S0 manifest",
        bool(a.expect_planning_marker)
        and a.expect_planning_marker in (req.get('planning_markers') or []),
        a.expect_planning_marker)
    started = _utc_stamp(rep.get('started') or '')
    chk('it started after the seal', sealed_t is not None and started is not None
        and started >= sealed_t, 'started %r' % rep.get('started'))
    sh = (rep.get('synced_head') or {}).get('value')
    chk('it ran on the sealed head', bool(tree_head) and sh == tree_head,
        '%r vs %r' % (sh, tree_head))
    rt = req.get('tree')
    chk('it ran from this tree', bool(rt) and os.path.realpath(rt) == os.path.realpath(a.tree or '.'),
        '%r' % rt)

    bad = [c for c in checks if not c['ok']]
    sha = sha256_file(a.audit_gate) if os.path.isfile(a.audit_gate or '') else '-'
    left = {'text': v.get('window_hits_in_text_non_planning') or [],
            'file names': v.get('window_hits_in_file_names') or [],
            'history': v.get('window_hits_in_git_history') or []}
    noted = []
    if not checks or bad:
        state, code = 'FAIL', 1
    elif v.get('clean') is True:
        state, code = 'PASS', 0
    else:
        if a.audit_ledger:
            noted = [r for r in read_ledger(a.audit_ledger)
                     if r.get('event') == 'note' and r.get('reason') == 'audit_classified'
                     and r.get('report_sha256') == sha]
        state, code = ('CLASSIFIED', 0) if noted else ('UNCLASSIFIED', 3)
    print('[AUDIT-GATE] %s %s: %d check(s), %d failed; report sha256 %s; clean=%r; hits left '
          'to classify: %s'
          % (state, a.audit_gate, len(checks), len(bad), sha[:16], v.get('clean'),
             ', '.join('%s %d' % (k, len(x)) for k, x in left.items())))
    if state == 'CLASSIFIED':
        print('[AUDIT-GATE] classified by a recorded operator action (%s): the remaining hits '
              'were acknowledged as classified in writing for exactly this report'
              % noted[-1].get('utc'))
    elif state == 'UNCLASSIFIED':
        print('[AUDIT-GATE] the result-data gate passed, but text, file-name or history hits '
              'remain.  Classify every one in writing (PREREGISTRATION_R3 section 13(b)), then '
              'record it: bash confirmatory/run_final.sh --profile <profile> --audit-ok')
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump({'state': state, 'report': a.audit_gate, 'report_sha256': sha,
                       'freeze_file': a.freeze_file, 'sealed_tree_sha256': sealed_sha,
                       'sealed_tree_head': tree_head, 'checks': checks,
                       'n_checks': len(checks), 'n_failed': len(bad),
                       'clean': v.get('clean'), 'hits_left': left,
                       'classified_by_note': bool(noted)}, f, indent=1, sort_keys=True)
            f.write('\n')
    return code


def cmd_ledger_tree_pin(a):
    """The tree digest of the FIRST invocation of the current run.

    Nothing else pins the tree before FREEZE_FINAL is sealed at S5, and S5
    comes after the 340 policy-independent passes, the training block,
    validation and selection.  Printing '-' means this run has no record yet
    (the first invocation sets it).
    """
    # The WHOLE ledger, not the last chain run.  A ledger is one run of the
    # chain; a resume after a stop is part of that run, and a tree change
    # between the stop and the resume is exactly what this is here to catch --
    # the rehearsal's own resume crossed a chain_start boundary, and if the
    # pin restarted there a code change made in between would have gone
    # unnoticed until the cross-provenance gate hours later.  Starting a new
    # run means archiving the ledger, which is an explicit operator action.
    for r in read_ledger(a.ledger_tree_pin):
        t = r.get('tree_sha256')
        if t and t != '-':
            print(t)
            return 0
    print('-')
    return 0


def cmd_ledger_append(a):
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for kv in a.kv:
        k, _, v = kv.partition('=')
        if v not in ('', None):
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        rec[k] = v
    d = os.path.dirname(os.path.abspath(a.ledger_append))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(a.ledger_append, 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')
    return 0


def read_ledger(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def _scope_run(recs, how):
    """Scope a ledger to ONE chain run.

    --timings used to sum every 'end' record in the file, so a ledger holding
    an abandoned attempt and the run that finally worked produced a total that
    matched neither, and the figures quoted in docs/RUN_CHAIN.md could not be
    reproduced by the command cited for them.
    """
    if how == 'all':
        return recs
    starts = [i for i, r in enumerate(recs) if r.get('event') == 'chain_start']
    if how == 'ok':
        ends = [i for i, r in enumerate(recs)
                if r.get('event') == 'chain_end' and r.get('rc') in (0, '0')]
        if ends:
            last_end = ends[-1]
            prev = [i for i in starts if i < last_end]
            return recs[(prev[-1] if prev else 0):last_end + 1]
    return recs[starts[-1]:] if starts else recs


def cmd_ledger_status(a):
    """Latest verdict for one invocation: STATUS sha256 attempts.

    STATUS is SUCCESS only for a recorded successful END.  The caller must
    still confirm the artifact's sha256 matches before skipping the work.
    """
    hist = [r for r in read_ledger(a.ledger_status) if r.get('inv') == a.inv]
    # An explicit operator INVALIDATE wipes the slate for that invocation:
    # everything recorded before it stops counting, so the work is redone and
    # the restart budget starts again.  It is the ONLY supported way to redo
    # an invocation that once succeeded, and it leaves a record.
    last_inv = max([i for i, r in enumerate(hist)
                    if r.get('event') == 'invalidate'] or [-1])
    recs = [r for r in hist[last_inv + 1:] if r.get('event') in ('end', 'skip')]
    attempts = len([r for r in recs if r.get('event') == 'end'])
    if not recs:
        print('NONE - 0')
        return 0
    last = recs[-1]
    status = 'SUCCESS' if last.get('status') == 'ok' else 'FAILED'
    print('%s %s %d' % (status, last.get('sha256') or '-', attempts))
    return 0


# ------------------------------------------------------------------ timings
def _load_plan(profile):
    sys.path.insert(0, os.path.join(ROOT, 'confirmatory'))
    import chain_plan
    return chain_plan, chain_plan.build(profile)


# ------------------------------------------- the ledger as a record of ORDER
def _inv_history(recs, inv):
    """[(record index, record)] of one invocation SINCE its last operator
    invalidate (which wipes the slate, exactly as in cmd_ledger_status)."""
    hist = [(i, r) for i, r in enumerate(recs) if r.get('inv') == inv]
    last = max([k for k, (_i, r) in enumerate(hist) if r.get('event') == 'invalidate'] or [-1])
    return hist[last + 1:]


def _attempts(recs, inv):
    """The attempts of `inv` since its last invalidate, in ledger order:
    [{'start': record or None, 'end': record or None}].

    An attempt is a START.  A start without an end is an attempt all the
    same -- the launcher or the host died while it ran, and it ran.  (An end
    without a start can only come from a ledger written by hand; it is kept
    as an attempt of its own so that nothing recorded is dropped.)"""
    out = []
    for _i, r in _inv_history(recs, inv):
        if r.get('event') == 'start':
            out.append({'start': r, 'end': None})
        elif r.get('event') == 'end':
            if out and out[-1]['end'] is None:
                out[-1]['end'] = r
            else:
                out.append({'start': None, 'end': r})
    return out


def terminal_failure(recs, inv):
    """The cause, as one line, when the ledger records that the
    pre-registered restart rule is EXHAUSTED for `inv`; else None.

    The rule is one unchanged re-run.  It is exhausted when the LAST attempt
    is closed and failed, and either an earlier attempt exists that did not
    succeed (it failed, or it never ended), or that last attempt already
    contains the watchdog's unchanged re-run (run_final.sh stops there:
    'watchdog_used_the_rerun').  It is derived from the START and END
    records -- what was run and how it ended -- and never from a `stop` line
    alone: a chain can stop for many reasons that are not a failure of the
    work (a preflight gate, a refused lock, a pin), and none of them may
    release a waiver.  An attempt that is still OPEN is never terminal: its
    job may be running.

    THE LAUNCHER USES THE SAME FUNCTION (--restart-state): an invocation this
    function calls terminal is never started again, so that 'family E is not
    completed' and 'S9x can still be run' cannot both be true."""
    att = _attempts(recs, inv)
    if not att or att[-1]['end'] is None or att[-1]['end'].get('status') == 'ok':
        return None

    def what(a, k):
        if a['end'] is None:
            return 'attempt %s did not end' % ((a['start'] or {}).get('attempt') or k)
        r = a['end']
        return 'attempt %s exit %s gate %s' % (r.get('attempt'), r.get('exit'), r.get('gate'))
    last = att[-1]
    if len(att) >= 2 and (att[-2]['end'] is None or att[-2]['end'].get('status') != 'ok'):
        return 'failed twice (%s; %s)' % (what(att[-2], len(att) - 1), what(last, len(att)))
    try:
        wd = int(last['end'].get('watchdog_restarts') or 0)
    except (TypeError, ValueError):
        wd = 0
    if wd >= 1:
        return "failed after the watchdog's one unchanged re-run (%s)" % what(last, len(att))
    return None


def restart_state(recs, inv):
    """(attempts started, state of the last one, terminal cause or None) for
    `inv` since its last invalidate.  The state is 'none' (never started),
    'open' (a start without an end), 'ok' or 'fail'."""
    att = _attempts(recs, inv)
    if not att:
        return 0, 'none', None
    e = att[-1]['end']
    state = 'open' if e is None else ('ok' if e.get('status') == 'ok' else 'fail')
    return len(att), state, terminal_failure(recs, inv)


def cmd_restart_state(a):
    """--restart-state LEDGER --inv ID: one line, `ATTEMPTS LAST TERMINAL
    CAUSE` (TERMINAL 0 or 1; CAUSE '-' when not terminal).  What
    run_final.sh reads before it starts an invocation."""
    if not a.inv:
        print('- - - no --inv given')
        return 2
    n, state, cause = restart_state(read_ledger(a.restart_state), a.inv)
    print('%d %s %d %s' % (n, state, 1 if cause else 0, cause or '-'))
    return 0


def cmd_stage_started(a):
    """--stage-started LEDGER --profile P --from-stage ST: `STARTED <inv>`
    (exit 0) when the ledger holds a start or an end of any invocation of
    stage ST or of a later stage of the profile's graph, since that
    invocation's last invalidate; `NOT_STARTED` (exit 1) otherwise; exit 2
    when ST is not a stage of the graph.  The launcher refuses to invalidate
    ANYTHING of a restricted profile once this says STARTED."""
    cp, invs = _load_plan(a.profile)
    order = cp.stage_order(a.profile)
    if a.from_stage not in order:
        print('ERROR %r is not a stage of profile %s' % (a.from_stage, a.profile))
        return 2
    recs = read_ledger(a.stage_started)
    for i in invs:
        if order.index(i['stage']) >= order.index(a.from_stage) and _attempts(recs, i['id']):
            print('STARTED %s' % i['id'])
            return 0
    print('NOT_STARTED')
    return 1


def cmd_terminal_failure(a):
    """`TERMINAL <stage> <inv>: <cause>` and exit 0 when an invocation of the
    stage has exhausted the restart rule; `NOT_TERMINAL ...` and exit 1
    otherwise (exit 2: the stage has no invocation, or no stage was named).
    The launcher honours a failure-only order waiver, and S12 records a
    family as not completed, on exit 0 ONLY."""
    if not a.stage:
        print('NOT_TERMINAL no --stage given')
        return 2
    _cp, invs = _load_plan(a.profile)
    ids = [i['id'] for i in invs if i['stage'] == a.stage]
    if not ids:
        print('NOT_TERMINAL %s: profile %s has no invocation of that stage' % (a.stage, a.profile))
        return 2
    recs = read_ledger(a.terminal_failure)
    for inv in ids:
        cause = terminal_failure(recs, inv)
        if cause:
            print('TERMINAL %s %s: %s' % (a.stage, inv, cause))
            return 0
    print('NOT_TERMINAL %s: the ledger records no exhausted restart rule for %s'
          % (a.stage, ', '.join(ids)))
    return 1


def _success_span(recs, inv):
    """Where the ledger records the SUCCESSFUL run of `inv`: the index and
    UTC stamp of its last successful end and of the start of that attempt,
    and the sha256 recorded for its artefact.  None when it did not
    succeed."""
    h = _inv_history(recs, inv)
    ends = [(i, r) for i, r in h if r.get('event') == 'end']
    if not ends or ends[-1][1].get('status') != 'ok':
        return None
    i_end, r_end = ends[-1]
    starts = [(i, r) for i, r in h if r.get('event') == 'start' and i < i_end]
    if not starts:
        return None
    i_start, r_start = starts[-1]
    # the FIRST start since the last invalidate: an earlier attempt that
    # failed has still run, and may have computed and printed what the
    # successful one did
    i_first, r_first = starts[0]
    return {'inv': inv, 'start_record': i_start, 'start_utc': r_start.get('utc'),
            'first_start_record': i_first, 'first_start_utc': r_first.get('utc'),
            'end_record': i_end, 'end_utc': r_end.get('utc'),
            'sha256': r_end.get('sha256'), 'artifact': r_end.get('artifact'),
            'attempt': r_end.get('attempt')}


def cmd_ledger_order(a):
    """--ledger-order LEDGER --first A[,A2..] --then B [--expect-then-sha256 SHA]:
    PASS only when every invocation is recorded successful, the successful
    END of each A precedes the FIRST START of B in the ledger -- the first
    attempt of B, failed or not, since B's last invalidate (the ledger is
    append-only: record order is time order) -- and, when asked, the sha256
    the ledger holds for B's artefact is SHA.

    S7e uses it so that 'the passes of family E were complete before family
    1 was analysed' is read from the record and not asserted by the step
    that needs it to be true."""
    recs = read_ledger(a.ledger_order)
    firsts = [x for x in (a.first or '').split(',') if x]
    out = {'ledger': a.ledger_order, 'first': firsts, 'then': a.then, 'ok': False,
           'problems': []}
    A = [_success_span(recs, x) for x in firsts]
    B = _success_span(recs, a.then or '')
    out['first_span'], out['then_span'] = A, B
    if not recs:
        out['problems'].append('the ledger %s holds no record' % a.ledger_order)
    if not firsts or not a.then:
        out['problems'].append('nothing to order: --first and --then are both required')
    for x, sp in zip(firsts, A):
        if sp is None:
            out['problems'].append('%s is not recorded successful' % x)
    if B is None:
        out['problems'].append('%s is not recorded successful' % a.then)
    before = bool(firsts) and B is not None and all(
        sp is not None and sp['end_record'] < B['first_start_record'] for sp in A)
    if B is not None:
        for x, sp in zip(firsts, A):
            if sp is not None and not sp['end_record'] < B['first_start_record']:
                out['problems'].append(
                    '%s ended (record %d, %s) AFTER %s was first started (record %d, %s)'
                    % (x, sp['end_record'], sp['end_utc'], a.then, B['first_start_record'],
                       B['first_start_utc']))
        if a.expect_then_sha256 and B.get('sha256') != a.expect_then_sha256:
            out['problems'].append("the ledger's sha256 of %s (%s) is not %s"
                                   % (a.then, str(B.get('sha256'))[:16], a.expect_then_sha256[:16]))
    out['first_end_before_then_start'] = before
    out['ok'] = before and not out['problems']
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump(out, f, indent=1, sort_keys=True)
    if out['ok']:
        print('[LEDGER-ORDER] PASS %s ended at record(s) %s (last %s), before %s was first started '
              'at record %d (%s)%s'
              % (','.join(firsts), ','.join(str(sp['end_record']) for sp in A),
                 max(str(sp['end_utc']) for sp in A), a.then, B['first_start_record'],
                 B['first_start_utc'],
                 '; the ledger holds the expected sha256 of ' + a.then if a.expect_then_sha256 else ''))
        return 0
    print('[LEDGER-ORDER] FAIL %d problem(s)' % len(out['problems']))
    for p_ in out['problems']:
        print('               %s' % p_)
    return 1


def cmd_ledger_artifacts(a):
    """--ledger-artifacts LEDGER --profile P --stage S6,S9x --json-out F:
    {artifact path: {inv, status, sha256}} for every pickle the plan
    declares for those stages -- what the ledger recorded when each
    invocation ended.  Exit 1 when any of them is not recorded successful."""
    _cp, invs = _load_plan(a.profile)
    want = [s for s in (a.stage or '').split(',') if s]
    recs = read_ledger(a.ledger_artifacts)
    out, bad = {}, 0
    for i in invs:
        if i['stage'] not in want or not i['artifact'].endswith('.pkl'):
            continue
        s = _success_span(recs, i['id'])
        out[i['artifact']] = {'inv': i['id'], 'stage': i['stage'],
                              'status': 'SUCCESS' if s else 'NOT_SUCCESSFUL',
                              'sha256': (s or {}).get('sha256')}
        bad += 0 if s else 1
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump(out, f, indent=1, sort_keys=True)
    print('[LEDGER-ARTIFACTS] %s %d pickle(s) of stage(s) %s, %d not recorded successful'
          % ('PASS' if out and not bad else 'FAIL', len(out), ','.join(want) or '-', bad))
    return 0 if out and not bad else 1


def timings(a):
    all_recs = read_ledger(a.timings)
    recs = _scope_run(all_recs, a.run)
    ends = [r for r in recs if r.get('event') == 'end']
    all_ends = [r for r in all_recs if r.get('event') == 'end']
    if not ends:
        print('[TIMINGS] FAIL: the ledger holds no completed invocation')
        return 1
    chain_plan, real_invs = _load_plan('real')
    _, prof_invs = _load_plan(a.profile)
    by_id = {i['id']: i for i in prof_invs}

    # ---- measured wall clock per stage
    per_stage = {}
    for r in ends:
        st = r.get('stage', '?')
        s = per_stage.setdefault(st, {'wall_s': 0.0, 'inv': 0, 'passes': 0,
                                      'ok': 0, 'fail': 0})
        s['wall_s'] += float(r.get('wall_s') or 0)
        s['inv'] += 1
        s['passes'] += int(r.get('passes') or 0)
        s['ok' if r.get('status') == 'ok' else 'fail'] += 1

    scope = {'last': 'the LAST chain run in the ledger',
             'ok': 'the last chain run that ended rc=0',
             'all': 'EVERY run in the ledger'}[a.run]
    print('MEASURED WALL CLOCK  (profile %s, --run %s = %s: %d completed '
          'invocations of %d in the file)'
          % (a.profile, a.run, scope, len(ends), len(all_ends)))
    print('  %-5s %-4s %-6s %-9s %s' % ('stage', 'inv', 'passes', 'wall_s', 'wall'))
    total = 0.0
    # the stage graph of the profile being reported (R3-lite has its own;
    # for the R2 profiles this is chain_plan.STAGE_ORDER, as before)
    for st in chain_plan.stage_order(a.profile):
        if st not in per_stage:
            continue
        s = per_stage[st]
        total += s['wall_s']
        print('  %-5s %-4d %-6d %-9.0f %.2f h%s'
              % (st, s['inv'], s['passes'], s['wall_s'], s['wall_s'] / 3600.0,
                 '' if not s['fail'] else '   (%d failed invocation(s))' % s['fail']))
    print('  %-5s %-4d %-6d %-9.0f %.2f h' % ('TOTAL', len(ends),
                                              sum(s['passes'] for s in per_stage.values()),
                                              total, total / 3600.0))
    if a.run != 'all':
        other = sum(float(r.get('wall_s') or 0) for r in all_ends) - total
        print('  (+ %.0f s = %.2f h in earlier/abandoned runs of this ledger, '
              'excluded by --run %s; --run all includes them)'
              % (other, other / 3600.0, a.run))

    # ---- measured CPU-s per pass, per arm AND PER TICK COUNT
    #
    # A profile is no longer uniformly scaled: the rehearsal carries a
    # full-length development subset so the pre-registered analysis can run at
    # all (REQUEST R11 in docs/RUN_CHAIN.md).  Averaging a 600-tick pass and a
    # 3600-tick pass of the same arm into one number inflates the mean and
    # then extrapolates the inflated figure AGAIN -- it put marl_static at
    # 227 CPU-s "at 600 ticks" when the 600-tick passes alone cost 192.  So
    # the per-pass costs are bucketed by the TICK COUNT OF THE INVOCATION that
    # produced them, which is also what turns the full-length subset into a
    # direct measurement at 3600 instead of an extrapolation.
    nominal = chain_plan.PROFILES[a.profile]['nominal_ticks']
    ticks_of_artifact = {}
    for i in prof_invs:
        if i['artifact'].endswith('.pkl'):
            ticks_of_artifact[i['artifact']] = i['ticks'] or nominal
    cpu = {}
    for r in ends:
        if r.get('status') != 'ok':
            continue
        art = r.get('artifact') or ''
        if not art.endswith('.pkl') or not os.path.exists(art):
            continue
        t = ticks_of_artifact.get(art)
        if t is None:
            continue
        try:
            P = load_pickle(art)
        except Exception:
            continue
        for key, st in (P.get('pass_stats') or {}).items():
            arm = key.split('|')[0]
            if st.get('cpu_s'):
                cpu.setdefault((arm, t), []).append(float(st['cpu_s']))
    if not cpu:
        print('  FAIL: no per-pass CPU statistics found in any pickle')
        return 1
    means = {k: sum(v) / len(v) for k, v in cpu.items()}
    tick_counts = sorted({t for _a, t in means})
    arms_seen = sorted({arm for arm, _t in means})

    # An arm measured at BOTH tick counts in this run pins its own affine
    # intercept -- no plan constant and no cross-arm assumption is involved.
    setup = {}
    for arm in arms_seen:
        pair = [t for t in tick_counts if (arm, t) in means]
        if len(pair) >= 2:
            lo, hi = pair[0], pair[-1]
            m_lo, m_hi = means[(arm, lo)], means[(arm, hi)]
            # cpu(t) = setup + rate*t
            rate = (m_hi - m_lo) / float(hi - lo)
            setup[arm] = m_hi - rate * hi
    # For an arm measured at one tick count only, fall back to the median of
    # the measured intercepts (same code path, stated as such).
    med = (sorted(setup.values())[len(setup) // 2] if setup else 0.0)

    print('')
    print('MEASURED CPU-s PER PASS, bucketed by the tick count of the '
          'invocation that produced it')
    print('  %-12s %s %12s %12s %8s' % ('arm', ''.join(
        '%7s@%-5d' % ('n/mean', t) for t in tick_counts),
        'setup', 'plan 3600', 'ratio'))
    scaled = {}
    for arm in arms_seen:
        cells = ''
        for t in tick_counts:
            k = (arm, t)
            cells += ('%4d/%-8.1f' % (len(cpu[k]), means[k])) if k in means else '%13s' % '-'
        su = setup.get(arm, med)
        if (arm, nominal) in means:
            val = means[(arm, nominal)]           # MEASURED at full length
            how = 'measured'
        else:
            t = max(t for t in tick_counts if (arm, t) in means)
            rate = (means[(arm, t)] - su) / float(t)
            val = su + rate * nominal
            how = 'affine'
        scaled[arm] = val
        plan = chain_plan.ARM_CPU_S_3600.get(arm)
        print('  %-12s %s %12.0f %12s %7s  (%s)'
              % (arm, cells, su, '%.0f' % plan if plan else 'ASSUMED',
                 '%.2f' % (val / plan) if plan else '-', how))
    print('  setup is this arm\'s OWN affine intercept where it was run at two '
          'tick counts in this ledger (%d of %d arms), else the median %.0f '
          'CPU-s of those that were.' % (len(setup), len(arms_seen), med))
    print('  "measured" means the x3600 figure IS the mean of full-length '
          'passes in this ledger, not an extrapolation.')

    # ---- extrapolation to the real workload
    print('')
    print('EXTRAPOLATION TO THE REAL WORKLOAD (%d passes), cost model '
          'wall = CPU-s / (cores_obtained x 3600)' % sum(i['expected_passes']
                                                         for i in real_invs))
    print('  NOTE: every CPU-s here was measured on a SHARED host; the load '
          'average over this ledger is recorded per invocation in '
          'confirmatory/preflight_%s/.' % a.profile)
    # The pessimistic reading: cost strictly proportional to ticks, i.e. no
    # fixed per-pass setup at all.  Kept as the upper bound of the range.
    short = min(tick_counts)
    linear = {}
    for arm in arms_seen:
        if (arm, short) in means:
            linear[arm] = means[(arm, short)] * (float(nominal) / short)
        else:
            linear[arm] = scaled[arm]
    stage_cpu, stage_cpu_lin = {}, {}
    for i in real_invs:
        if not i['expected_passes']:
            continue
        n = len(i['seeds']) * 2
        for tgt, tbl in ((stage_cpu, scaled), (stage_cpu_lin, linear)):
            c = sum(tbl.get(arm, chain_plan.ARM_CPU_S_3600.get(arm, 1000.0))
                    for arm in i['arms']) * n
            tgt[i['stage']] = tgt.get(i['stage'], 0.0) + c
    for cores, label in ((chain_plan.CORES_OBTAINED, 'typical tenant load'),
                         (18.26, 'quiet host')):
        lo = sum(stage_cpu.values()) / (cores * 3600.0)
        hi = sum(stage_cpu_lin.values()) / (cores * 3600.0)
        print('  evaluation stages at 24 workers (%s, %.2f cores obtained): '
              '%.2f h affine / %.2f h linear-in-ticks' % (label, cores, lo, hi))
    print('  per stage at %.2f cores:' % chain_plan.CORES_OBTAINED)
    for st in chain_plan.STAGE_ORDER:
        if st in stage_cpu:
            print('    %-5s %8.2f h' % (st, stage_cpu[st] /
                                        (chain_plan.CORES_OBTAINED * 3600.0)))
    # training wall: replicates run 2 at a time
    tr = [i for i in real_invs if i['kind'] == 'train']
    if tr:
        waves = (len(tr) + 1) // 2
        print('  training (measured separately; %d replicates in %d waves of 2): '
              'plan 3.48 h per wave -> %.2f h' % (len(tr), waves, waves * 3.48))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify')
    ap.add_argument('--key-union')
    ap.add_argument('--cross-provenance')
    ap.add_argument('--hash-pair')
    ap.add_argument('--check-marl-env')
    ap.add_argument('--expect-marl-env-file', default='',
                    help='with --verify or --check-marl-env: the plan\'s '
                         'MARL_* set (K=V per line, chain_write_env)')
    ap.add_argument('--expect-hashseeds', default='0,1')
    ap.add_argument('--expect-hashseed', default='0',
                    help='with --verify: the PYTHONHASHSEED the plan declares '
                         'for the invocation (default 0)')
    ap.add_argument('--keys-file')
    ap.add_argument('--json-out')
    ap.add_argument('--ledger-append')
    ap.add_argument('--kv', nargs='*', default=[])
    ap.add_argument('--ledger-status')
    ap.add_argument('--inv')
    ap.add_argument('--timings')
    ap.add_argument('--run', default='last', choices=('last', 'ok', 'all'),
                    help='which chain run --timings reports on (default: the '
                         'last one in the ledger)')
    ap.add_argument('--ledger-tree-pin')
    ap.add_argument('--ablation-diff')
    ap.add_argument('--baseline')
    ap.add_argument('--ablation-series', default='')
    ap.add_argument('--ablation-counter', default='')
    ap.add_argument('--ablation-arms', default='')
    ap.add_argument('--profile', default='rehearsal')
    ap.add_argument('--sha256')
    ap.add_argument('--expect-checkpoint-sha256', default='',
                    help='with --verify or --paired-descriptive: the sha256 '
                         'the plan pins for the invocation\'s checkpoint(s), '
                         'comma-separated in MARL_EVAL_CHECKPOINT order; the '
                         'pickle must record exactly these')
    ap.add_argument('--paired-descriptive', metavar='PKL[,PKL]',
                    help='DESCRIPTIVE paired comparison of --paired-arm in '
                         'these pickles with --paired-ref-arm in --baseline '
                         'on --paired-seeds (R3-lite: the ensemble against '
                         'XL-DET); --json-out for the result')
    ap.add_argument('--paired-arm', default='marl_static')
    ap.add_argument('--paired-ref-arm', default='xldet')
    ap.add_argument('--paired-seeds', default='')
    ap.add_argument('--paired-label', default='ensemble')
    ap.add_argument('--audit-gate', metavar='REPORT',
                    help='R3-lite: the freeze-2 seed-audit report as a gate of '
                         'the launcher (exit 0 pass, 3 hits left unclassified, '
                         '1 fail); with --freeze-file, --expect-window LO,HI, '
                         '--expect-planning-marker, --tree, [--audit-ledger]')
    ap.add_argument('--freeze-file', default='')
    ap.add_argument('--expect-window', default='')
    ap.add_argument('--expect-planning-marker', default='')
    ap.add_argument('--tree', default='.')
    ap.add_argument('--audit-ledger', default='',
                    help='the chain ledger: a note reason=audit_classified with '
                         "the report's sha256 acknowledges classified hits")
    ap.add_argument('--terminal-failure', metavar='LEDGER',
                    help='with --profile and --stage: exit 0 and "TERMINAL ..." '
                         'when the restart rule is exhausted for an invocation '
                         'of the stage')
    ap.add_argument('--stage', default='',
                    help='a stage id (--terminal-failure) or a comma list '
                         '(--ledger-artifacts)')
    ap.add_argument('--ledger-order', metavar='LEDGER',
                    help='with --first A --then B: the successful end of A '
                         'precedes the start of the successful attempt of B')
    ap.add_argument('--first', default='')
    ap.add_argument('--then', default='')
    ap.add_argument('--expect-then-sha256', default='')
    ap.add_argument('--ledger-artifacts', metavar='LEDGER',
                    help='with --profile, --stage and --json-out: the sha256 '
                         'the ledger recorded for every pickle of the stages')
    ap.add_argument('--restart-state', metavar='LEDGER',
                    help='with --inv: "ATTEMPTS LAST TERMINAL CAUSE" of the '
                         'restart rule for that invocation')
    ap.add_argument('--stage-started', metavar='LEDGER',
                    help='with --profile and --from-stage: exit 0 when an '
                         'invocation of that stage or a later one has started')
    ap.add_argument('--from-stage', default='')
    a = ap.parse_args()

    if a.restart_state:
        return cmd_restart_state(a)
    if a.stage_started:
        return cmd_stage_started(a)
    if a.terminal_failure:
        return cmd_terminal_failure(a)
    if a.ledger_order:
        return cmd_ledger_order(a)
    if a.ledger_artifacts:
        return cmd_ledger_artifacts(a)
    if a.sha256:
        print(sha256_file(a.sha256) if os.path.exists(a.sha256) else '-')
        return 0
    if a.audit_gate:
        return cmd_audit_gate(a)
    if a.ledger_append:
        return cmd_ledger_append(a)
    if a.ledger_status:
        return cmd_ledger_status(a)
    if a.ledger_tree_pin:
        return cmd_ledger_tree_pin(a)
    if a.ablation_diff:
        return cmd_ablation_diff(a)
    if a.paired_descriptive:
        return cmd_paired_descriptive(a)
    if a.verify:
        return cmd_verify(a)
    if a.key_union:
        return cmd_key_union(a)
    if a.cross_provenance:
        return cmd_cross_provenance(a)
    if a.hash_pair:
        return cmd_hash_pair(a)
    if a.check_marl_env:
        return cmd_check_marl_env(a)
    if a.timings:
        return timings(a)
    ap.error('nothing to do')


if __name__ == '__main__':
    sys.exit(main())
