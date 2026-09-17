# -*- coding: utf-8 -*-
"""Amendment-1 sensitivity analysis (confirmatory seeds 80-99).

XL-DET was run twice on the confirmatory instances: once in the original job
(the pre-registered baseline source) and once in the same invocation as
marl_static.  This script reports

  1. reproducibility: whether XL-DET's decision series (everything except the
     energy and latency accounting) are bit-identical between the two runs;
  2. the size of the energy/latency accounting jitter between invocations,
     per instance and in run mean;
  3. P7 (energy per delivered unit, MARL-RIC minus XL-DET, Scenario A)
     recomputed against the co-run XL-DET, which shares marl_static's
     invocation and therefore its accounting jitter.

It is descriptive and outside the Holm family; the pre-registered P7 is the
one computed by confirmatory_analysis.py against the original job.
Run from ~/MARL_run_6.
"""
import math
import os
import pickle
import statistics as st

import confirmatory_analysis as CA

R5 = os.path.expanduser('~/MARL_run_5')
ORIG = os.path.join(R5, 'output', 'cf_eval_80_99.pkl')
CORUN = ['output/cf_corun_80_89.pkl', 'output/cf_corun_90_99.pkl']
JITTER_KEYS = ('energy', 'latency')


def isnum(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def series_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isnum(x) and isnum(y):
            if x != y:
                return False
        elif isnum(x) != isnum(y):
            return False
    return True


def main():
    orig = pickle.load(open(ORIG, 'rb'))
    co = {'recovery_multi': {}, 'rescue_multi': {}}
    for p in CORUN:
        P = pickle.load(open(p, 'rb'))
        for sk in co:
            for arm in ('xldet', 'marl_static'):
                for s, r in P[sk].get(arm, {}).items():
                    co[sk].setdefault(arm, {})[int(s)] = r
    print('1. XL-DET decision-series reproducibility (original job vs co-run)')
    all_ok = True
    for sk in ('recovery_multi', 'rescue_multi'):
        seeds = sorted(set(orig[sk]['xldet']) & set(co[sk]['xldet']))
        bad = []
        for s in seeds:
            a, b = orig[sk]['xldet'][s], co[sk]['xldet'][s]
            keys = [k for k in a if isinstance(a[k], list) and k not in JITTER_KEYS]
            diff = [k for k in keys if not series_equal(a[k], b[k])]
            if diff:
                bad.append((s, diff))
        all_ok &= not bad
        print('   %-15s %d instances, %d decision series each: %s' % (
            sk, len(seeds), len(keys), 'ALL BIT-IDENTICAL' if not bad else 'DIFFERENCES %s' % bad[:5]))
    print('   -> %s' % ('reproducible' if all_ok else 'NOT reproducible'))

    print('\n2. energy / latency accounting jitter between invocations (XL-DET)')
    for sk in ('recovery_multi', 'rescue_multi'):
        seeds = sorted(set(orig[sk]['xldet']) & set(co[sk]['xldet']))
        for k in JITTER_KEYS:
            rel = []
            for s in seeds:
                a = [x for x in orig[sk]['xldet'][s][k] if isnum(x)]
                b = [x for x in co[sk]['xldet'][s][k] if isnum(x)]
                if a and b:
                    rel.append(abs(st.mean(a) - st.mean(b)) / abs(st.mean(a)))
            print('   %-15s %-8s run-mean relative difference: mean %.2e  max %.2e' % (sk, k, st.mean(rel), max(rel)))

    print('\n3. P7 against the co-run XL-DET (Scenario A, energy per delivered unit)')
    sa = sorted(set(co['recovery_multi']['xldet']) & set(co['recovery_multi']['marl_static']))
    d = [CA.jpu(co['recovery_multi']['marl_static'][s]) - CA.jpu(co['recovery_multi']['xldet'][s]) for s in sa]
    r = CA.paired(d, -1)
    print('   n=%d  mean %+.4f  95%% CI [%+.4f, %+.4f]  lower on %d/%d  sign p=%.4f  Wilcoxon p=%.5f' % (
        r['n'], r['mean'], r['ci_lo'], r['ci_hi'], r['wins'], r['n'], r['sign_p'], r['wilcoxon_p']))
    d0 = [CA.jpu(co['recovery_multi']['marl_static'][s]) - CA.jpu(orig['recovery_multi']['xldet'][s]) for s in sa]
    r0 = CA.paired(d0, -1)
    print('   (against the original-job XL-DET, as pre-registered: mean %+.4f, CI [%+.4f, %+.4f], Wilcoxon p=%.5f)' % (
        r0['mean'], r0['ci_lo'], r0['ci_hi'], r0['wilcoxon_p']))


if __name__ == '__main__':
    main()
