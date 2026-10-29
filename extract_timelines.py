# -*- coding: utf-8 -*-
"""Per-tick timelines for the manuscript's timeline figures.

For every arm and both scenarios, bins each instance's series into BIN-tick
values, then summarises across instances (mean, median, 25th and 75th
percentile per bin).  Descriptive only.

  KEYS      state series, binned as means: achievability, latency, energy,
            fragments, fragments_raw, relay_count, relay_bridges
  SUM_KEYS  event series of the WP4 recording (relay_created, relay_removed,
            bridge_teardowns, idle_bridge_teardowns, relay_mode_flips) and the
            stale-edge diagnostics, binned as sums per BIN ticks

A key that not every instance of an arm carries is skipped and listed under
that arm's 'not_recorded' (never filled in).  The key names, the numeric test
and the numeric conversion (numpy scalars and arrays are accepted, counted
and converted to plain Python numbers, so the JSON is always serialisable)
come from confirmatory_analysis.py so that one definition exists.

Seeds: the instance set of a scenario is the intersection of the seeds of
the KNOWN arms (ARMS) that are present; an arm the pickles carry under
another name is summarised over its own seeds within that set and listed in
'unknown_arms', so a partial or unknown arm never shrinks the timeline of
the others.  Seeds an arm has beyond the set are recorded in 'seeds_dropped'.

Usage: python3 extract_timelines.py --pickles A.pkl[,B.pkl] --out timelines.json
"""
import argparse
import json
import math
import pickle
import statistics as st

from confirmatory_analysis import EVENT_SERIES, DIAG_SERIES, is_num, py_num

BIN = 10
ARMS = ('marl_static', 'marl_freeze', 'xldet', 'sdn', 'ospf', 'olsr', 'batman', 'aodv', 'random')
SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
KEYS = ('conn', 'latency', 'energy', 'fragments', 'fragments_raw', 'relay_count', 'relay_bridges')
SUM_KEYS = EVENT_SERIES + DIAG_SERIES


def num(x):
    """A usable numeric entry (Python or numpy number, not NaN/None)."""
    return is_num(x)


def binned(series, n_bins, how='mean'):
    """Bin a per-tick series into n_bins values (mean or sum per BIN ticks).

    Every entry goes through confirmatory_analysis.py_num: the sum of a bin of
    np.int64 entries stays np.int64, which json.dump refuses, so an event
    series recorded with numpy ints (the contract of TRACKB_HANDOFF section 1)
    otherwise crashed the whole timeline export at write time (WP9 defect
    list, D3)."""
    out = []
    for b in range(n_bins):
        v = [py_num(x) for x in series[b * BIN:(b + 1) * BIN] if num(x)]
        if not v:
            out.append(None)
        else:
            out.append(sum(v) / len(v) if how == 'mean' else sum(v))
    return out


def q(vals, p):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * p
    f = math.floor(k); c = math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def load_runs(pickles):
    runs = {k: {} for k in SCEN}
    for p in pickles:
        P = pickle.load(open(p, 'rb'))
        for k, sk in SCEN.items():
            for arm, d in P.get(sk, {}).items():
                for s, r in d.items():
                    if r:
                        runs[k].setdefault(arm, {})[int(s)] = r
    return runs


def extract(runs):
    out = {'bin_ticks': BIN, 'scenarios': {},
           'binning': dict([(k, 'mean') for k in KEYS] + [(k, 'sum') for k in SUM_KEYS])}
    for k in SCEN:
        known = [arm for arm in ARMS if runs[k].get(arm)]
        unknown = sorted(a for a in runs[k] if a not in ARMS and runs[k][a])
        present = known + unknown
        if not present:
            out['scenarios'][k] = {'seeds': [], 'n_bins': 0, 'arms': {}, 'unknown_arms': [], 'seeds_dropped': {}}
            continue
        # the seed set and the run length come from the known arms only
        basis = known or unknown
        seeds = sorted(set.intersection(*[set(runs[k][arm]) for arm in basis]))
        T = min(len(runs[k][arm][s]['conn']) for arm in basis for s in seeds)
        nb = T // BIN
        sc = {'seeds': seeds, 'n_bins': nb, 'arms': {}, 'unknown_arms': unknown, 'seeds_dropped': {}}
        for arm in present:
            arm_seeds = [s for s in seeds if s in runs[k][arm]]
            dropped = sorted(set(runs[k][arm]) - set(arm_seeds))
            if dropped:
                sc['seeds_dropped'][arm] = dropped
                print('scenario %s: arm %s has seeds outside the instance set, dropped: %s' % (k, arm, dropped))
            sc['arms'][arm] = {'not_recorded': []}
            for key, how in [(x, 'mean') for x in KEYS] + [(x, 'sum') for x in SUM_KEYS]:
                if not arm_seeds or not all(runs[k][arm][s].get(key) is not None for s in arm_seeds):
                    sc['arms'][arm]['not_recorded'].append(key)
                    continue
                per = [binned(runs[k][arm][s][key], nb, how) for s in arm_seeds]
                stats = {'mean': [], 'median': [], 'q25': [], 'q75': [], 'n': []}
                for b in range(nb):
                    col = [row[b] for row in per if row[b] is not None]
                    # py_num again on the way out: nothing but plain Python
                    # numbers ever reaches json.dump
                    stats['n'].append(len(col))
                    stats['mean'].append(round(py_num(st.mean(col)), 4) if col else None)
                    stats['median'].append(round(py_num(q(col, 0.5)), 4) if col else None)
                    stats['q25'].append(round(py_num(q(col, 0.25)), 4) if col else None)
                    stats['q75'].append(round(py_num(q(col, 0.75)), 4) if col else None)
                sc['arms'][arm][key] = stats
        out['scenarios'][k] = sc
        print('scenario %s: %d instances, %d bins x %d ticks' % (k, len(seeds), nb, BIN))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickles', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = extract(load_runs(a.pickles.split(',')))
    json.dump(out, open(a.out, 'w'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
