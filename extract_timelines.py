# -*- coding: utf-8 -*-
"""Per-tick timelines for the manuscript's timeline figures.

For every arm and both scenarios, bins each instance's achievability, latency
and energy series into BIN-tick means, then summarises across instances
(mean, median, 25th and 75th percentile per bin).  Descriptive only.

Usage: python3 extract_timelines.py --pickles A.pkl[,B.pkl] --out timelines.json
"""
import argparse
import json
import math
import pickle
import statistics as st

BIN = 10
ARMS = ('marl_static', 'marl_freeze', 'xldet', 'sdn', 'ospf', 'olsr', 'batman', 'aodv', 'random')
SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
KEYS = ('conn', 'latency', 'energy', 'fragments')


def num(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def binned(series, n_bins):
    out = []
    for b in range(n_bins):
        v = [x for x in series[b * BIN:(b + 1) * BIN] if num(x)]
        out.append(sum(v) / len(v) if v else None)
    return out


def q(vals, p):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * p
    f = math.floor(k); c = math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickles', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs = {k: {arm: {} for arm in ARMS} for k in SCEN}
    for p in a.pickles.split(','):
        P = pickle.load(open(p, 'rb'))
        for k, sk in SCEN.items():
            for arm in ARMS:
                for s, r in P[sk].get(arm, {}).items():
                    if r:
                        runs[k][arm][int(s)] = r
    out = {'bin_ticks': BIN, 'scenarios': {}}
    for k in SCEN:
        present = [arm for arm in ARMS if runs[k][arm]]
        seeds = sorted(set.intersection(*[set(runs[k][arm]) for arm in present]))
        T = min(len(runs[k][arm][s]['conn']) for arm in present for s in seeds)
        nb = T // BIN
        sc = {'seeds': seeds, 'n_bins': nb, 'arms': {}}
        for arm in present:
            sc['arms'][arm] = {}
            for key in KEYS:
                per = [binned(runs[k][arm][s][key], nb) for s in seeds]
                stats = {'mean': [], 'median': [], 'q25': [], 'q75': [], 'n': []}
                for b in range(nb):
                    col = [row[b] for row in per if row[b] is not None]
                    stats['n'].append(len(col))
                    stats['mean'].append(round(st.mean(col), 4) if col else None)
                    stats['median'].append(round(q(col, 0.5), 4) if col else None)
                    stats['q25'].append(round(q(col, 0.25), 4) if col else None)
                    stats['q75'].append(round(q(col, 0.75), 4) if col else None)
                sc['arms'][arm][key] = stats
        out['scenarios'][k] = sc
        print('scenario %s: %d instances, %d bins x %d ticks' % (k, len(seeds), nb, BIN))
    json.dump(out, open(a.out, 'w'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
