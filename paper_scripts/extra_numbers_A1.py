# -*- coding: utf-8 -*-
"""Descriptive quantities quoted in the manuscript that make_results.py does
not print: per-arm service-loss medians (Scenario A) on the development and
confirmatory sets, and the paired statistics of the interferer stress test
(development seeds 50-59).  Reads the same KPI dumps as make_results.py and
uses the tests of confirmatory_analysis.py.  None of this is a pre-registered
primary or secondary analysis.

Usage: python paper/extra_numbers_A1.py > paper/extra_numbers_A1.txt
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from confirmatory_analysis import paired  # noqa: E402

A1 = os.path.join(ROOT, 'auth_results', 'A1')
ARMS = ['marl_static', 'marl_freeze', 'xldet', 'sdn', 'ospf', 'olsr', 'batman', 'aodv', 'random']
T0 = 200


def sli(r):
    conn = r['conn']['series']; off = r['offered']['series']
    pre = [c for c in conn[:T0] if c is not None]
    pre_off = [o for o in off[:T0] if o is not None]
    post_off = [o for o in off[T0:] if o is not None]
    a_ref = st.mean(pre) * st.mean(pre_off) / st.mean(post_off)
    return sum(max(0.0, a_ref - (c if c is not None else 0.0)) for c in conn[T0:])


def wmean(series, lo, hi):
    v = [x for x in series[lo:hi] if x is not None]
    return sum(v) / len(v)


def show(name, d, expect):
    p = paired(d, expect)
    print('  %-44s mean %+.2f  sd %.2f  95%% CI [%+.2f, %+.2f]  %s on %d/%d  sign p=%.3f  Wilcoxon p=%.4f' % (
        name, p['mean'], p['sd'], p['ci_lo'], p['ci_hi'], 'ahead' if expect > 0 else 'lower',
        p['wins'], p['n'], p['sign_p'], p['wilcoxon_p']))


print('== Service-loss integral medians, Scenario A (pp.s; per-instance, post-event ticks) ==')
for label, fn, tag in (('development 50-59,70-79', 'ce_results_A1_devstress.json', 'dev'),
                       ('confirmatory 80-99', 'ce_results_A1_cf.json', 'cf')):
    J = json.load(open(os.path.join(A1, fn)))[tag]['scenarios']['recovery_multi']
    row = []
    for a in ARMS:
        if a in J:
            v = [sli(r) for r in J[a].values()]
            row.append('%s %.0f (n=%d)' % (a, st.median(v), len(v)))
    print('  %s: %s' % (label, ' | '.join(row)))

print('\n== Interferer stress test (development seeds 50-59, onset t=2000) ==')
S = json.load(open(os.path.join(A1, 'ce_results_A1_devstress.json')))['stress']['scenarios']['recovery_multi']
seeds = sorted(S['xldet'], key=int)
pre = {a: [wmean(S[a][s]['conn']['series'], 1500, 2000) for s in seeds] for a in S}
post = {a: [wmean(S[a][s]['conn']['series'], 2000, 3600) for s in seeds] for a in S}
loss = {a: [pre[a][i] - post[a][i] for i in range(len(seeds))] for a in S}
comps_pre = {a: st.mean(wmean(S[a][s]['fragments']['series'], 1500, 2000) for s in seeds) for a in S}
comps_post = {a: st.mean(wmean(S[a][s]['fragments']['series'], 2000, 3600) for s in seeds) for a in S}
print('  seeds:', seeds)
for a in ('marl_static', 'marl_freeze', 'xldet', 'sdn', 'ospf'):
    print('  %-12s pre %.1f  post %.1f  loss %.1f +- %.1f  components pre %.2f post %.2f' % (
        a, st.mean(pre[a]), st.mean(post[a]), st.mean(loss[a]), st.stdev(loss[a]), comps_pre[a], comps_post[a]))
show('post-onset achievability, static - XL-DET', [post['marl_static'][i] - post['xldet'][i] for i in range(10)], +1)
show('loss, static - XL-DET', [loss['marl_static'][i] - loss['xldet'][i] for i in range(10)], +1)
show('post-onset achievability, static - SDN', [post['marl_static'][i] - post['sdn'][i] for i in range(10)], +1)
per_seed_comp = {s: (round(wmean(S['marl_static'][s]['fragments']['series'], 1500, 2000), 2),
                     round(wmean(S['marl_static'][s]['fragments']['series'], 2000, 3600), 2)) for s in seeds}
print('  static components pre/post per seed:', per_seed_comp)

print('\n== Detection floor (Scenario A): mean achievability over ticks [200, 200+k), zero-delivery ticks after t0 ==')
for label, fn, tag in (('development 50-59,70-79', 'ce_results_A1_devstress.json', 'dev'),
                       ('confirmatory 80-99', 'ce_results_A1_cf.json', 'cf')):
    J = json.load(open(os.path.join(A1, fn)))[tag]['scenarios']['recovery_multi']
    for a in ('marl_static', 'marl_freeze', 'xldet', 'sdn', 'ospf'):
        runs = list(J[a].values())
        row = []
        for k in (3, 4):
            row.append('[200,%d) %.1f' % (200 + k, st.mean(wmean(r['conn']['series'], 200, 200 + k) for r in runs)))
        zero_c = sum(1 for r in runs if any((c is not None and c == 0) for c in r['conn']['series'][200:]))
        zero_d = sum(1 for r in runs if any((d is not None and d == 0) for d in r['delivered']['series'][200:]))
        pre = st.mean(wmean(r['conn']['series'], 0, 200) for r in runs)
        print('  %-24s %-12s %s  pre %.1f  instances with a zero-achievability tick %d, zero-delivered tick %d' % (
            label, a, '  '.join(row), pre, zero_c, zero_d))
