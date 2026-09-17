# -*- coding: utf-8 -*-
"""Pre-registered analysis of the confirmatory test set (topology seeds 80-99).

This file is written, validated and hashed BEFORE the confirmatory evaluation
runs.  Its sha256 is recorded in confirmatory/PREREGISTRATION.md and in
confirmatory/FREEZE.sha256.  Validation = reproducing the development-set
figures already in the manuscript (seeds 50-59 + 70-79), which exercises every
code path on data whose answers are known.

Definitions are those of paper/make_results.py:
  steady(x)   mean of the numeric values among the last 500 ticks
  J/unit      per instance: whole-run energy / whole-run delivered volume
  pooled J/u  sum of energy / sum of delivered over the instance set
  pre-surge   Scenario B: mean achievability over ticks [900, 1400)
  best routing per instance: max achievability / min J/unit over the four
              routing protocols (the conservative comparator for MARL-RIC)

Primary family (nine tests, Holm-corrected at alpha = 0.05, decision test =
exact two-sided Wilcoxon signed-rank on the paired per-instance differences,
zeros dropped, mid-ranks for ties).  A hypothesis is CONFIRMED iff the Holm-
adjusted p < 0.05 AND the signed-rank statistic lies in the expected direction.

Usage:
  python3 confirmatory_analysis.py --pickles PKL[,PKL] --manifests JSON[,JSON]
                                   --out REPORT.json
"""
import argparse
import json
import math
import pickle
import statistics as st

W = 500
MARL, XL, SDN = 'marl_freeze', 'xldet', 'sdn'
ROUTING = ('ospf', 'olsr', 'batman', 'aodv')
ARMS = (MARL, XL, SDN) + ROUTING + ('random',)
SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
KIND = {'A': 'recovery', 'B': 'rescue_ops'}
T_EVENT = {'A': 200, 'B': 1400}
PRE_SURGE = (900, 1400)
ALPHA = 0.05


# ------------------------------------------------------------------ helpers
def num(xs):
    return [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]


def steady(xs):
    v = num(xs[-W:])
    return sum(v) / len(v) if v else float('nan')


def wmean(xs, lo, hi):
    v = num(xs[lo:hi])
    return sum(v) / len(v) if v else float('nan')


def jpu(r):
    return sum(num(r['energy'])) / sum(num(r['delivered']))


def sli(r, t0):
    conn, off = r['conn'], r['offered']
    pre = num(conn[:t0]); pre_off = num(off[:t0]); post_off = num(off[t0:])
    a_ref = st.mean(pre) * st.mean(pre_off) / st.mean(post_off)
    return sum(max(0.0, a_ref - (c if isinstance(c, (int, float)) else 0.0)) for c in conn[t0:])


def sign_p(d):
    n = sum(1 for x in d if x != 0)
    if n == 0:
        return 1.0
    k = min(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(d):
    """Exact two-sided signed-rank test; zeros dropped, mid-ranks for ties.
    Returns (p, W_plus, W_minus, n_nonzero)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0.0, 0.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    wm = sum(r for r, x in zip(ranks, d) if x < 0)
    doubled = [int(round(2 * r)) for r in ranks]
    dist = {0: 1}
    for r in doubled:
        nd = {}
        for s, c in dist.items():
            nd[s] = nd.get(s, 0) + c
            nd[s + r] = nd.get(s + r, 0) + c
        dist = nd
    centre = sum(doubled) / 2
    dev = abs(2 * wp - centre)
    p = sum(c for s, c in dist.items() if abs(s - centre) >= dev - 1e-9) / 2 ** n
    return min(1.0, p), wp, wm, n


def t_ci(d):
    n = len(d)
    mu = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    try:
        from scipy.stats import t as tdist
        q = tdist.ppf(0.975, n - 1)
    except Exception:
        q = {9: 2.262, 10: 2.228, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}.get(n - 1, 2.093)
    h = q * sd / math.sqrt(n)
    return mu, sd, mu - h, mu + h


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(d, expect):
    p, wp, wm, nz = wilcoxon(d)
    mu, sd, lo, hi = t_ci(d)
    wins = sum(1 for x in d if (x > 0 if expect > 0 else x < 0))
    try:
        from scipy.stats import wilcoxon as swil
        sp = float(swil(d, zero_method='wilcox', alternative='two-sided').pvalue)
    except Exception:
        sp = None
    return dict(n=len(d), mean=mu, sd=sd, ci_lo=lo, ci_hi=hi, wins=wins,
                sign_p=sign_p(d), wilcoxon_p=p, wilcoxon_p_scipy=sp,
                W_plus=wp, W_minus=wm, n_nonzero=nz,
                direction_ok=(wp > wm) if expect > 0 else (wm > wp), expect=expect)


# ------------------------------------------------------------------ load
def load(pickles, manifests):
    runs = {k: {a: {} for a in ARMS} for k in SCEN}
    for p in pickles:
        P = pickle.load(open(p, 'rb'))
        for k, sk in SCEN.items():
            for a in ARMS:
                for s, r in P[sk].get(a, {}).items():
                    if r:
                        runs[k][a][int(s)] = r
    opt = {}
    for m in manifests:
        for inst in json.load(open(m))['instances']:
            opt[(inst['scenario_kind'], int(inst['topology_seed']))] = inst
    return runs, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickles', required=True)
    ap.add_argument('--manifests', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs, opt = load(a.pickles.split(','), a.manifests.split(','))

    rep = {'seeds': {}, 'excluded': {}, 'descriptive': {}, 'primary': {}, 'secondary': {}}
    seeds = {}
    for k in SCEN:
        common = set.intersection(*[set(runs[k][arm]) for arm in ARMS])
        every = set.union(*[set(runs[k][arm]) for arm in ARMS])
        seeds[k] = sorted(common)
        rep['seeds'][k] = seeds[k]
        rep['excluded'][k] = sorted(every - common)     # harness failures only

    def ach(k, arm, s):
        return steady(runs[k][arm][s]['conn'])

    def best_routing_ach(k, s):
        return max(ach(k, r, s) for r in ROUTING)

    def best_routing_jpu(k, s):
        return min(jpu(runs[k][r][s]) for r in ROUTING)

    # ---- descriptive, per arm
    for k in SCEN:
        S = seeds[k]
        d = {}
        for arm in ARMS:
            R = runs[k][arm]
            comps = [steady(R[s]['fragments']) for s in S]
            optk = [opt[(KIND[k], s)]['attainable_optimum'] for s in S]
            d[arm] = dict(
                ach_mean=st.mean(ach(k, arm, s) for s in S), ach_sd=st.stdev(ach(k, arm, s) for s in S),
                ach_median=st.median(ach(k, arm, s) for s in S),
                ach_min=min(ach(k, arm, s) for s in S), ach_max=max(ach(k, arm, s) for s in S),
                thr_mean=st.mean(steady(R[s]['delivered']) for s in S),
                energy_mean=st.mean(steady(R[s]['energy']) for s in S),
                energy_sd=st.stdev(steady(R[s]['energy']) for s in S),
                jpu_pooled=sum(sum(num(R[s]['energy'])) for s in S) / sum(sum(num(R[s]['delivered'])) for s in S),
                jpu_mean=st.mean(jpu(R[s]) for s in S),
                comps_mean=st.mean(comps),
                comps_reunifiable=(st.mean(c for c, o in zip(comps, optk) if o == 1)
                                   if any(o == 1 for o in optk) else None),
                at_optimum=sum(1 for c, o in zip(comps, optk) if c <= o + 1e-9),
                below_floor=sum(1 for s in S if ach(k, arm, s) < 10.0),
                pre_level=st.mean(wmean(R[s]['conn'], *(PRE_SURGE if k == 'B' else (0, T_EVENT['A']))) for s in S),
            )
            if k == 'A':
                v = [sli(R[s], T_EVENT['A']) for s in S]
                d[arm].update(sli_mean=st.mean(v), sli_median=st.median(v))
        rep['descriptive'][k] = d
        rep['descriptive'][k + '_optima'] = {
            'n': len(S),
            'reunifiable': sum(1 for s in S if opt[(KIND[k], s)]['attainable_optimum'] == 1),
            'mean_optimum': st.mean(opt[(KIND[k], s)]['attainable_optimum'] for s in S),
            'mean_raw': st.mean(opt[(KIND[k], s)]['raw_components'] for s in S),
        }

    # ---- primary family
    fam = []
    for k in ('A', 'B'):
        S = seeds[k]
        fam.append(('P%d_%s_ach_vs_xldet' % (1 if k == 'A' else 4, k),
                    [ach(k, MARL, s) - ach(k, XL, s) for s in S], +1))
        fam.append(('P%d_%s_ach_vs_sdn' % (2 if k == 'A' else 5, k),
                    [ach(k, MARL, s) - ach(k, SDN, s) for s in S], +1))
        fam.append(('P%d_%s_ach_vs_best_routing' % (3 if k == 'A' else 6, k),
                    [ach(k, MARL, s) - best_routing_ach(k, s) for s in S], +1))
    S = seeds['A']
    fam.append(('P7_A_jpu_vs_xldet', [jpu(runs['A'][MARL][s]) - jpu(runs['A'][XL][s]) for s in S], -1))
    fam.append(('P8_A_jpu_vs_sdn', [jpu(runs['A'][MARL][s]) - jpu(runs['A'][SDN][s]) for s in S], -1))
    fam.append(('P9_A_jpu_vs_best_routing', [jpu(runs['A'][MARL][s]) - best_routing_jpu('A', s) for s in S], -1))
    res = [paired(d, e) for _, d, e in fam]
    adj = holm([r['wilcoxon_p'] for r in res])
    for (name, _, _), r, h in zip(fam, res, adj):
        r['wilcoxon_p_holm'] = h
        r['confirmed'] = bool(h < ALPHA and r['direction_ok'])
        rep['primary'][name] = r

    # ---- secondary (not in the Holm family)
    sec = rep['secondary']
    for k in ('A', 'B'):
        S = seeds[k]
        sec['S1_%s_components_marl_minus_xldet' % k] = paired(
            [steady(runs[k][MARL][s]['fragments']) - steady(runs[k][XL][s]['fragments']) for s in S], +1)
    SB = seeds['B']

    def change(arm, s):
        c = runs['B'][arm][s]['conn']
        return steady(c) - wmean(c, *PRE_SURGE)

    def pre(arm, s):
        return wmean(runs['B'][arm][s]['conn'], *PRE_SURGE)

    sec['S2_B_change_vs_xldet'] = paired([change(MARL, s) - change(XL, s) for s in SB], +1)
    sec['S2_B_change_vs_sdn'] = paired([change(MARL, s) - change(SDN, s) for s in SB], +1)
    sec['S2_B_change_vs_best_routing'] = paired(
        [change(MARL, s) - max(change(r, s) for r in ROUTING) for s in SB], +1)
    sec['S3_B_pre_level_vs_xldet'] = paired([pre(MARL, s) - pre(XL, s) for s in SB], +1)
    sec['S3_B_pre_level_vs_sdn'] = paired([pre(MARL, s) - pre(SDN, s) for s in SB], +1)
    sec['S3_B_pre_level_vs_best_routing'] = paired([pre(MARL, s) - max(pre(r, s) for r in ROUTING) for s in SB], +1)
    SA = seeds['A']
    sec['S4_A_abs_energy_vs_xldet'] = paired(
        [steady(runs['A'][MARL][s]['energy']) - steady(runs['A'][XL][s]['energy']) for s in SA], +1)
    sec['S5_A_sli_vs_xldet'] = paired(
        [sli(runs['A'][MARL][s], T_EVENT['A']) - sli(runs['A'][XL][s], T_EVENT['A']) for s in SA], -1)
    sec['S6_B_jpu_vs_xldet'] = paired([jpu(runs['B'][MARL][s]) - jpu(runs['B'][XL][s]) for s in SB], -1)
    pre_a = {arm: [wmean(runs['A'][arm][s]['conn'], 0, T_EVENT['A']) for s in SA] for arm in ARMS}
    sec['shared_world_A_pre_event_spread_pp'] = max(
        max(abs(pre_a[x][i] - pre_a[MARL][i]) for x in ARMS) for i in range(len(SA)))

    json.dump(rep, open(a.out, 'w'), indent=1)

    # ---- console report
    print('instances: A=%d B=%d   excluded (harness failures): %s' % (
        len(seeds['A']), len(seeds['B']), rep['excluded']))
    for k in SCEN:
        o = rep['descriptive'][k + '_optima']
        print('\nScenario %s  (optimum 1 on %d/%d, mean optimum %.2f, mean raw %.2f)' % (
            k, o['reunifiable'], o['n'], o['mean_optimum'], o['mean_raw']))
        print('  %-12s %13s %6s %7s %8s %8s %6s %6s' % ('arm', 'ach', 'thr', 'E/tick', 'J/u pool', 'comps', 'atopt', 'pre'))
        for arm in ARMS:
            x = rep['descriptive'][k][arm]
            print('  %-12s %5.1f +- %4.1f %6.0f %7.1f %8.4f %8.2f %3d/%-2d %6.1f' % (
                arm, x['ach_mean'], x['ach_sd'], x['thr_mean'], x['energy_mean'], x['jpu_pooled'],
                x['comps_mean'], x['at_optimum'], len(seeds[k]), x['pre_level']))
    print('\nPRIMARY (Holm over 9, exact Wilcoxon):')
    for name, r in rep['primary'].items():
        print('  %-28s %+8.4f  CI [%+.4f, %+.4f]  wins %2d/%d  sign %.4f  W %.5f (scipy %s)  Holm %.5f  %s' % (
            name, r['mean'], r['ci_lo'], r['ci_hi'], r['wins'], r['n'], r['sign_p'], r['wilcoxon_p'],
            'n/a' if r['wilcoxon_p_scipy'] is None else '%.5f' % r['wilcoxon_p_scipy'],
            r['wilcoxon_p_holm'], 'CONFIRMED' if r['confirmed'] else 'not confirmed'))
    print('\nSECONDARY:')
    for name, r in sec.items():
        if isinstance(r, dict):
            print('  %-32s %+9.4f  CI [%+.4f, %+.4f]  wins %2d/%d  sign %.4f  W %.5f' % (
                name, r['mean'], r['ci_lo'], r['ci_hi'], r['wins'], r['n'], r['sign_p'], r['wilcoxon_p']))
        else:
            print('  %-32s %s' % (name, r))
    print('\nwritten', a.out)


if __name__ == '__main__':
    main()
