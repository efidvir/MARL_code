# -*- coding: utf-8 -*-
"""A/B the reunify_boost policy against the baseline policy and XL-DET.

The one axis XL-DET beat the learned policy on was residual component count on
the reunifiable recovery instances (baseline MARL 1.29 vs XL-DET 1.14), while
MARL led on delivered throughput.  Lever 2 (the reunify_boost reward profile)
aims to close the component gap WITHOUT surrendering the throughput lead, so
that MARL Pareto-dominates XL-DET.

This reads:
  output/auth_eval_s1234.pkl   -> XL-DET and the BASELINE (full-profile) MARL
  output/boost_eval_s1234.pkl  -> the reunify_boost MARL (marl, marl_freeze)
and prints components + achievability + bridges for the 7 reunifiable recovery
seeds, so the Pareto question is answerable at a glance.

Usage:  python3 compare_boost.py [--dir output]
"""
import argparse
import math
import os
import pickle

NON_REUNIFIABLE = {52, 53, 59}
STEADY_WINDOW = 500


def steady(series):
    if not isinstance(series, list) or not series:
        return float('nan')
    tail = [v for v in series[-STEADY_WINDOW:]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    return sum(tail) / len(tail) if tail else float('nan')


def arm_means(pkl, arm, seeds_keep, keys):
    """Mean over kept seeds of the steady-state value of each key for one arm."""
    rec = pkl['recovery_multi'].get(arm, {})
    out = {}
    for k in keys:
        vals = []
        for s, run in rec.items():
            if s in seeds_keep and run:
                v = run.get(k)
                vals.append(float(v) if (k in ('adm_hold', 'adm_throttle')
                                         and isinstance(v, (int, float)))
                            else steady(v))
        vals = [v for v in vals if not math.isnan(v)]
        out[k] = sum(vals) / len(vals) if vals else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='output')
    args = ap.parse_args()

    base_p = os.path.join(args.dir, 'auth_eval_s1234.pkl')
    boost_p = os.path.join(args.dir, 'boost_eval_s1234.pkl')
    for p in (base_p, boost_p):
        if not os.path.exists(p):
            raise SystemExit('missing %s' % p)
    base = pickle.load(open(base_p, 'rb'))
    boost = pickle.load(open(boost_p, 'rb'))
    seeds = set(base['seeds']) - NON_REUNIFIABLE
    keys = ['conn', 'fragments', 'relay_bridges', 'adm_hold']

    rows = [
        ('XL-DET',              arm_means(base,  'xldet',       seeds, keys)),
        ('MARL-freeze (full)',  arm_means(base,  'marl_freeze', seeds, keys)),
        ('MARL-online (full)',  arm_means(base,  'marl',        seeds, keys)),
        ('MARL-freeze (boost)', arm_means(boost, 'marl_freeze', seeds, keys)),
        ('MARL-online (boost)', arm_means(boost, 'marl',        seeds, keys)),
    ]

    print("=" * 74)
    print("REUNIFIABLE RECOVERY (7 seeds)  --  boost A/B")
    print("=" * 74)
    print("%-22s %10s %10s %10s %10s" %
          ('arm', 'achiev%', 'components', 'bridges', 'holds'))
    for name, m in rows:
        print("%-22s %10.1f %10.2f %10.2f %10.0f" %
              (name, m['conn'], m['fragments'], m['relay_bridges'], m['adm_hold']))

    xld = rows[0][1]
    bf = rows[3][1]   # boost freeze
    print()
    print("VERDICT (boost MARL-freeze vs XL-DET):")
    dc = xld['fragments'] - bf['fragments']   # +ve => boost has FEWER components (better)
    da = bf['conn'] - xld['conn']             # +ve => boost delivers MORE (better)
    print("  components: boost %.2f vs XL-DET %.2f  (%s by %.2f)" %
          (bf['fragments'], xld['fragments'],
           'boost better' if dc > 0 else 'XL-DET better', abs(dc)))
    print("  achievability: boost %.1f%% vs XL-DET %.1f%%  (%s by %.1f pp)" %
          (bf['conn'], xld['conn'],
           'boost better' if da > 0 else 'XL-DET better', abs(da)))
    pareto = (bf['fragments'] <= xld['fragments'] + 1e-6) and (bf['conn'] >= xld['conn'] - 1e-6)
    print("  => MARL %s XL-DET on BOTH axes" %
          ('PARETO-DOMINATES' if pareto else 'does NOT yet dominate'))
    # vs the baseline policy, to confirm the boost actually moved reunification
    bfull = rows[1][1]
    print("  (baseline full-profile freeze: components %.2f, achiev %.1f%% -- "
          "boost moved components by %.2f)" %
          (bfull['fragments'], bfull['conn'], bfull['fragments'] - bf['fragments']))


if __name__ == '__main__':
    main()
