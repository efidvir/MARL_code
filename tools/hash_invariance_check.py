#!/usr/bin/env python3
"""S8 hash-invariance check (PREREGISTRATION_R2 section 10).

Two harness pickles of the same tree, checkpoint and instances, run under
PYTHONHASHSEED 0 and 1, must hold identical decision, delivered and fragment
series.  Compared bit-exactly: every per-tick series and per-pass scalar of a
run except the accounting series ACCOUNTING_KEYS ('energy', 'latency'), whose
between-invocation jitter is a known property of the energy/latency
accounting (AMENDMENT_1.md 2.2).  The accounting series are compared as well
and reported as jitter (max relative difference of the run means); with
--strict they must be identical too.  Element comparison is the bit-exact
rule of tools/series_diff.py (NaN == NaN; 0.0 != -0.0).

Decisions themselves: the harness of the frozen tree records no per-tick
action series (its _pass_out holds consequences only: relay_mode_nodes,
relay_count, fragments, delivered, ...).  The per-tick decision digest
DECISION_KEY ('decision_digest', requested from the engine track in
docs/TRACKB_HANDOFF.md: sha256 of the executed joint action of the tick) is
compared like any other series when present; the result records under
'decision_series' whether every compared MARL-path run carried it.  Without
it the check covers decision CONSEQUENCES only, and a hash-order-dependent
decision whose only visible trace is in energy/latency would pass as
identical (the accounting jitter report cannot separate it from the known
accounting jitter).  --require-decision makes a MARL-path run without the
digest a missing series (NOT IDENTICAL) -- the setting for the study's S8.

The PRECONDITION of the check is that the two pickles were run under
DIFFERENT hash seeds.  Both provenance records must be present, both must
carry `pythonhashseed`, and the two values must differ; otherwise the pair
is a repeat-run pair (or a pair whose seeds are simply not knowable) and
NOTHING about hash invariance was verified, whatever the series say.  That
precondition is part of the verdict, not an advisory note: `identical`
requires `same_hashseed is False`, and a pair that fails it is reported in
'precondition_not_established' and exits 2 (WP9 defect list, D1 residual).

Usage:
  hash_invariance_check.py A.pkl B.pkl [--arms a,b] [--seeds 82,91]
        [--scenarios A,B] [--keys k1,k2] [--strict] [--require-decision]
        [--out S8.json] [--quiet]

Exit 0 identical, 1 differences or missing runs/series, 2 usage OR a check
that verified nothing: it compared nothing (an empty selection, pickles
without the runs, or --require-decision with no decision series anywhere),
or the differing-hash-seed precondition is not established.  `identical` is
then false and 'nothing_compared' / 'precondition_not_established' in the
JSON say why, because a check that verified nothing must never read as a
pass.  The JSON written with --out is what confirmatory_analysis.py embeds
with --s8.
"""
import argparse
import json
import math
import numbers
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import series_diff as SD  # noqa: E402

ACCOUNTING_KEYS = ('energy', 'latency')
PROV_FIELDS = ('tree_head', 'tree_sha256', 'schema_id', 'pythonhashseed', 'checkpoint_sha256', 'total_ticks')
DECISION_KEY = 'decision_digest'          # per-tick digest of the executed joint action (engine request)
MARL_PATH_ARMS = ('marl_static', 'marl_freeze', 'marl', 'xldet', 'random')
ACCOUNTING_NOTE = ('energy/latency are compared as jitter only (not --strict): a hash-order-dependent '
                   'decision whose only visible trace is in energy/latency is not detected unless the '
                   'per-tick decision digest is recorded and identical')


def _mean(v):
    v = [float(x) for x in v if isinstance(x, numbers.Real) and not math.isnan(x)]
    return sum(v) / len(v) if v else None


def _rel(a, b):
    if a is None or b is None:
        return None if a is b else float('inf')
    return abs(a - b) / abs(a) if a != 0 else (0.0 if b == 0 else float('inf'))


def check(A, B, arms=None, scenarios=None, seeds=None, keys=None, strict=False, require_decision=False,
          decision_key=DECISION_KEY, marl_path_arms=MARL_PATH_ARMS):
    """Compare pickles A and B (loaded dicts).  Returns the S8 result dict.
    require_decision: a MARL-path run without `decision_key` in both pickles
    is a missing series (the study setting).

    `identical` is true only when something was actually compared: an empty
    selection (n_compared == 0), or --require-decision with no decision
    series in any run, gives identical=False and a reason in
    'nothing_compared'.

    It is also true only when the check COULD have detected a hash-order
    dependence: the two pickles must record different PYTHONHASHSEEDs
    (`same_hashseed is False`).  Equal seeds, a missing provenance record or
    a missing `pythonhashseed` give identical=False and a reason in
    'precondition_not_established'."""
    scenarios = scenarios or list(SD.SCENARIOS)
    if arms is None:
        arms = []
        for P in (A, B):
            for sk in SD.SCENARIOS.values():
                for a in (P.get(sk) or {}):
                    if a not in arms:
                        arms.append(a)
    if seeds is None:
        seeds = sorted(set(A.get('seeds') or []) | set(B.get('seeds') or []))
    diffs, missing, jitter = [], [], {}
    n_compared = 0
    dec = {'key': decision_key, 'runs_with_key': 0, 'runs_without_key': 0, 'arms_without_key': [],
           'required': bool(require_decision), 'coverage': None}
    for scen in scenarios:
        for arm in arms:
            for seed in seeds:
                ra, rb = SD._run_of(A, scen, arm, seed), SD._run_of(B, scen, arm, seed)
                if ra is SD.MISSING or rb is SD.MISSING:
                    if ra is SD.MISSING and rb is SD.MISSING:
                        continue
                    missing.append({'scenario': scen, 'arm': arm, 'seed': seed, 'key': '*',
                                    'where': 'A' if ra is SD.MISSING else 'B'})
                    continue
                if arm in marl_path_arms:
                    if decision_key in ra and decision_key in rb:
                        dec['runs_with_key'] += 1
                    else:
                        dec['runs_without_key'] += 1
                        if arm not in dec['arms_without_key']:
                            dec['arms_without_key'].append(arm)
                        if require_decision and decision_key not in ra and decision_key not in rb:
                            missing.append({'scenario': scen, 'arm': arm, 'seed': seed, 'key': decision_key,
                                            'where': 'both'})
                run_keys = list(keys) if keys else sorted(set(ra) | set(rb))
                if decision_key in ra or decision_key in rb:
                    if decision_key not in run_keys:
                        run_keys.append(decision_key)       # always compared, even with --keys
                for key in run_keys:
                    x = ra.get(key, SD.MISSING)
                    y = rb.get(key, SD.MISSING)
                    if x is SD.MISSING or y is SD.MISSING:
                        missing.append({'scenario': scen, 'arm': arm, 'seed': seed, 'key': key,
                                        'where': 'A' if x is SD.MISSING else 'B'})
                        continue
                    if key in ACCOUNTING_KEYS and not strict and isinstance(x, (list, tuple)):
                        r = _rel(_mean(x), _mean(y))
                        j = jitter.setdefault(key, {'max_rel_mean_diff': 0.0, 'n_series': 0, 'n_identical': 0})
                        j['n_series'] += 1
                        if SD.compare_series(x, y) is None:
                            j['n_identical'] += 1
                        if r is not None:
                            j['max_rel_mean_diff'] = max(j['max_rel_mean_diff'], r)
                        continue
                    n_compared += 1
                    d = SD.compare_series(x, y)
                    if d is not None:
                        diffs.append({'scenario': scen, 'arm': arm, 'seed': seed, 'key': key,
                                      'diff': SD.describe(d)})
    pa, pb = A.get('provenance') or {}, B.get('provenance') or {}
    prov = {'A': {f: pa.get(f) for f in PROV_FIELDS}, 'B': {f: pb.get(f) for f in PROV_FIELDS}}
    # same_hashseed is a three-valued answer: True (both recorded, equal),
    # False (both recorded, different -- the only value that makes this a
    # hash-invariance check) and None (not knowable: a pickle without a
    # provenance record, or with no `pythonhashseed` in it).
    sa, sb = (pa.get('pythonhashseed') if pa else None), (pb.get('pythonhashseed') if pb else None)
    seed_known = sa is not None and sb is not None and str(sa) != '' and str(sb) != ''
    same_seed = (str(sa) == str(sb)) if seed_known else None
    same_tree = all(pa.get(f) == pb.get(f) for f in ('tree_head', 'tree_sha256', 'schema_id')) if (pa and pb) else None
    if dec['runs_with_key'] and not dec['runs_without_key']:
        dec['coverage'] = 'decisions (per-tick digest) and their consequences'
    elif dec['runs_with_key']:
        dec['coverage'] = 'mixed: some MARL-path runs carry no per-tick decision series'
    else:
        dec['coverage'] = 'consequences only: no per-tick decision series recorded (see docs/TRACKB_HANDOFF.md)'
    # A check that compared nothing is NOT a pass.  `not diffs and not missing`
    # is vacuously true for n_compared == 0: a --seeds/--arms value that
    # matches no run (a typo), pickles without a top-level 'seeds' key, or a
    # scenario selection that is empty all produced "IDENTICAL" and exit 0 for
    # a pre-registered check that verified nothing, and confirmatory_analysis
    # --s8 then printed "S8 hash invariance: identical" (WP9 defect list, D1).
    vacuous = []
    if n_compared == 0:
        vacuous.append('nothing compared: no run/key pair matched the selection '
                       '(arms=%s, seeds=%s, scenarios=%s; %d run(s) missing from one pickle). '
                       'Check the --arms/--seeds/--scenarios values and that both pickles carry '
                       'the runs and a top-level "seeds" key.'
                       % (arms, seeds, scenarios, len(missing)))
    # --require-decision with no decision series anywhere.  When MARL-path runs
    # ARE present without the digest, `missing` already carries that finding
    # (NOT IDENTICAL, exit 1) -- the check did its job.  When the selection
    # holds no MARL-path run at all, nothing about decisions was verified.
    no_decision = require_decision and not dec['runs_with_key']
    if no_decision and not dec['runs_without_key']:
        vacuous.append('nothing verified: --require-decision, but the selection (arms=%s) holds no '
                       'MARL-path run at all, so no %r series was compared'
                       % (arms, decision_key))
    # The precondition of S8: the two runs must have used DIFFERENT hash
    # seeds.  Identical series from a pair that shared a seed (or whose seeds
    # are unknown) say nothing about hash-order dependence -- they are a
    # repeat-run result.  This was an advisory `note` outside the verdict and
    # outside the exit code, so output/wp9_validation/s8_smoke_repeat.json
    # (two pickles, neither recording a hash seed) read as an S8 pass (WP9
    # defect list, D1 residual).
    precondition = []
    if same_seed is None:
        precondition.append('hash-seed precondition not established: A records pythonhashseed %r and B %r '
                            '-- both pickles must carry a provenance record with pythonhashseed for the '
                            'pair to be a hash-invariance pair at all.  Nothing about hash invariance was '
                            'verified; identical series here mean only that the two runs agree.' % (sa, sb))
    elif same_seed:
        precondition.append('hash-seed precondition not established: both pickles record PYTHONHASHSEED %r, '
                            'so this is a REPEAT-RUN check, not a hash-invariance check.  A hash-order '
                            'dependence cannot show up in a pair that shared the seed.' % (sa,))
    return {'identical': (not diffs and not missing and not vacuous and not no_decision
                          and not precondition),
            'strict': bool(strict), 'nothing_compared': vacuous,
            'precondition_not_established': precondition,
            'n_compared': n_compared, 'n_different': len(diffs), 'n_missing': len(missing),
            'arms': arms, 'seeds': seeds, 'scenarios': scenarios,
            'differences': diffs[:200], 'missing': missing[:200],
            'accounting_jitter': jitter, 'provenance': prov,
            'decision_series': dec,
            'accounting_note': None if strict else ACCOUNTING_NOTE,
            'same_hashseed': same_seed, 'same_tree': same_tree,
            'note': ('both pickles record the same PYTHONHASHSEED: this is a repeat-run check, '
                     'not a hash-invariance check' if same_seed else
                     'the PYTHONHASHSEED of at least one pickle is not recorded: the pair cannot be '
                     'shown to be a hash-invariance pair' if same_seed is None else None)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--arms', default=None)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--scenarios', default=None, help='recovery,rescue_ops (aliases A,B)')
    ap.add_argument('--keys', default=None, help='comma list; default every key either run carries')
    ap.add_argument('--strict', action='store_true', help='energy/latency must be identical too')
    ap.add_argument('--require-decision', action='store_true',
                    help='a MARL-path run without the per-tick %s series is a missing series' % DECISION_KEY)
    ap.add_argument('--out', default=None)
    ap.add_argument('--quiet', '-q', action='store_true')
    a = ap.parse_args(argv)
    for p in (a.a, a.b):
        if not os.path.isfile(p):
            print('hash_invariance_check: no such file: %s' % p, file=sys.stderr)
            return 2
    scen = None
    if a.scenarios:
        scen = []
        for s in SD._csv(a.scenarios):
            if s.lower() not in SD._SCEN_ALIASES:
                print('hash_invariance_check: unknown scenario %r' % s, file=sys.stderr)
                return 2
            scen.append(SD._SCEN_ALIASES[s.lower()])
    res = check(SD.load(a.a), SD.load(a.b),
                arms=SD._csv(a.arms) if a.arms else None,
                scenarios=scen,
                seeds=[int(s) for s in SD._csv(a.seeds)] if a.seeds else None,
                keys=SD._csv(a.keys) if a.keys else None, strict=a.strict,
                require_decision=a.require_decision)
    res['files'] = {'A': a.a, 'B': a.b}
    print('A %s\nB %s' % (a.a, a.b))
    for tag in ('A', 'B'):
        print('  %s provenance: %s' % (tag, res['provenance'][tag]))
    if res['note']:
        print('  NOTE ' + res['note'])
    if not a.quiet:
        for d in res['differences']:
            print('DIFFERENT %(scenario)-10s %(arm)-12s seed=%(seed)-4s %(key)-18s %(diff)s' % d)
        for m in res['missing']:
            print('MISSING   %(scenario)-10s %(arm)-12s seed=%(seed)-4s %(key)-18s only in %(where)s' % m)
    for key, j in res['accounting_jitter'].items():
        print('  accounting %-8s identical %d/%d series, max relative run-mean difference %.3e'
              % (key, j['n_identical'], j['n_series'], j['max_rel_mean_diff']))
    d = res['decision_series']
    print('  decision series %s: %d MARL-path run(s) with it, %d without -> coverage: %s'
          % (d['key'], d['runs_with_key'], d['runs_without_key'], d['coverage']))
    if res['accounting_note']:
        print('  NOTE ' + res['accounting_note'])
    for reason in res['nothing_compared']:
        print('  VACUOUS ' + reason)
    for reason in res['precondition_not_established']:
        print('  PRECONDITION ' + reason)
    print('SUMMARY compared=%d different=%d missing=%d same_hashseed=%r -> %s' % (
        res['n_compared'], res['n_different'], res['n_missing'], res['same_hashseed'],
        'IDENTICAL' if res['identical'] else
        'NOT A HASH-INVARIANCE PAIR (precondition not established)' if res['precondition_not_established'] else
        'NOT IDENTICAL (nothing compared)' if res['nothing_compared'] else 'NOT IDENTICAL'))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
        print('written', a.out)
    if res['nothing_compared'] or res['precondition_not_established']:
        return 2
    return 0 if res['identical'] else 1


if __name__ == '__main__':
    sys.exit(main())
