# -*- coding: utf-8 -*-
"""4-seed A/B: reunify_boost2 vs the baseline (full) policy vs XL-DET.

Unlike compare_boost.py (single s1234 seed), this aggregates the learned arms
as mean +/- s.d. ACROSS the four independently trained checkpoints, which is
the only honest way to answer the Pareto question — the s1234 preview was the
lucky seed (baseline freeze 1.29 components on s1234 vs 1.65 across all four).

Reads, from --dir (default output):
  auth_eval_s{1234,2345,3456,4567}.pkl   -> baseline (full) marl arms; XL-DET (s1234)
  boost2_eval_s{1234,2345,3456,4567}.pkl -> reunify_boost2 marl arms
Reports the 7 reunifiable recovery seeds.

Usage:  python3 compare_boost2.py [--dir output]
"""
import argparse
import math
import os
import pickle

SEEDS_CK = ['1234', '2345', '3456', '4567']
NON_REUNIFIABLE = {52, 53, 59}
STEADY_WINDOW = 500


def steady(series, key):
    if key in ('adm_hold', 'adm_throttle'):
        return float(series) if isinstance(series, (int, float)) else float('nan')
    if not isinstance(series, list) or not series:
        return float('nan')
    tail = [v for v in series[-STEADY_WINDOW:]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    return sum(tail) / len(tail) if tail else float('nan')


def seed_mean(pkl, arm, keys, keep):
    """Mean over the kept benchmark seeds, for one checkpoint's recovery run."""
    rec = pkl['recovery_multi'].get(arm, {})
    out = {}
    for k in keys:
        vals = [steady(run.get(k), k) for s, run in rec.items()
                if s in keep and run]
        vals = [v for v in vals if not math.isnan(v)]
        out[k] = sum(vals) / len(vals) if vals else float('nan')
    return out


def across_ck(prefix, arm, keys, keep, d):
    """mean +/- s.d. of each key across the four checkpoints for one arm."""
    per = []
    for ck in SEEDS_CK:
        p = os.path.join(d, '%s_s%s.pkl' % (prefix, ck))
        if not os.path.exists(p):
            continue
        per.append(seed_mean(pickle.load(open(p, 'rb')), arm, keys, keep))
    agg = {}
    for k in keys:
        xs = [m[k] for m in per if not math.isnan(m[k])]
        if not xs:
            agg[k] = (float('nan'), float('nan'), 0); continue
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
        agg[k] = (mu, sd, len(xs))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='output')
    args = ap.parse_args()
    d = args.dir
    keep = set(range(50, 60)) - NON_REUNIFIABLE
    keys = ['conn', 'fragments', 'relay_bridges', 'adm_hold']

    # XL-DET is policy-independent — take it from the full-width s1234 pickle.
    base_s1234 = pickle.load(open(os.path.join(d, 'auth_eval_s1234.pkl'), 'rb'))
    xld = seed_mean(base_s1234, 'xldet', keys, keep)

    base = across_ck('auth_eval', 'marl_freeze', keys, keep, d)
    boost2 = across_ck('boost2_eval', 'marl_freeze', keys, keep, d)

    def cell(agg, k, p=2):
        mu, sd, n = agg[k]
        return '--' if math.isnan(mu) else '%.*f+/-%.*f' % (p, mu, p, sd)

    print("=" * 72)
    print("REUNIFIABLE RECOVERY (7 seeds) -- 4-seed A/B, marl_freeze arm")
    print("=" * 72)
    print("%-22s %13s %13s %13s" % ('arm', 'achiev%', 'components', 'bridges'))
    print("%-22s %13.1f %13.2f %13.2f   (n=%d seeds)" %
          ('XL-DET', xld['conn'], xld['fragments'], xld['relay_bridges'],
           len(keep)))
    print("%-22s %13s %13s %13s   (n=%d ck)" %
          ('MARL-freeze (full)', cell(base, 'conn', 1), cell(base, 'fragments'),
           cell(base, 'relay_bridges'), base['conn'][2]))
    print("%-22s %13s %13s %13s   (n=%d ck)" %
          ('MARL-freeze (boost2)', cell(boost2, 'conn', 1), cell(boost2, 'fragments'),
           cell(boost2, 'relay_bridges'), boost2['conn'][2]))

    bm, bsd, bn = boost2['fragments']
    am = boost2['conn'][0]
    print()
    print("VERDICT (boost2 MARL-freeze, 4-seed mean, vs XL-DET):")
    print("  components:    %.2f+/-%.2f  vs XL-DET %.2f  (%s)" %
          (bm, bsd, xld['fragments'],
           'MARL <= XL-DET' if bm <= xld['fragments'] + 1e-6 else
           'XL-DET still better by %.2f' % (bm - xld['fragments'])))
    print("  achievability: %.1f%%  vs XL-DET %.1f%%  (%s)" %
          (am, xld['conn'],
           'MARL ahead by %.1f pp' % (am - xld['conn']) if am >= xld['conn']
           else 'XL-DET ahead by %.1f pp' % (xld['conn'] - am)))
    pareto = (bm <= xld['fragments'] + 1e-6) and (am >= xld['conn'] - 1e-6)
    print("  => 4-seed mean: MARL %s XL-DET on BOTH axes" %
          ('PARETO-DOMINATES' if pareto else 'does NOT yet dominate'))
    print("  (baseline full 4-seed: components %.2f, achiev %.1f%% -- boost2 moved "
          "components by %.2f)" %
          (base['fragments'][0], base['conn'][0],
           base['fragments'][0] - bm))


if __name__ == '__main__':
    main()
