# -*- coding: utf-8 -*-
"""Merge the authoritative evaluation pickles into paper-ready numbers.

DESIGN OF THE STUDY THIS MERGES.  Four policies were trained independently
(torch seeds 1234/2345/3456/4567, identical data).  Only the learned arms
read a checkpoint, so the sweep ran ONE full-width pass (s1234: all nine
arms) plus three narrow passes (s2345/s3456/s4567: marl + marl_freeze only).
This script recombines them:

  * non-learned arms (xldet, random, ospf, sdn, olsr, batman, aodv):
      per-seed steady-state values from the full-width pickle,
      reported mean +/- s.d. over the 10 benchmark seeds;
  * learned arms (marl, marl_freeze):
      per-seed steady-state values per checkpoint; reported as the
      across-checkpoint mean of seed-means +/- the s.d. ACROSS CHECKPOINTS
      (training-seed variance), alongside the per-seed spread within each
      checkpoint (topology variance).  The two variances answer different
      reviewer questions and must not be pooled silently.

Conditioning: the benchmark manifest marks seeds 52, 53 and 59 as not fully
reunifiable under the actionable-bridge model.  Every KPI is reported for
(a) all 10 seeds and (b) the 7 reunifiable seeds, because reunification
KPIs on instances where reunification is impossible measure the scenario,
not the policy.

Usage:  python3 merge_auth_eval.py [--dir output] [--json merged_results.json]
"""
import argparse
import json
import math
import os
import pickle

CHECKPOINTS = ['1234', '2345', '3456', '4567']
LEARNED_ARMS = ['marl', 'marl_freeze']
STEADY_WINDOW = 500          # matches STEADY_STATE_WINDOW_TICKS in the driver
NON_REUNIFIABLE = {52, 53, 59}   # from benchmark_manifest.json

# KPI -> (pretty name, scale, how to aggregate a run's series)
#   'steady' = mean over the last STEADY_WINDOW ticks (NaN-aware)
#   'last'   = final value
#   'total'  = scalar already cumulative over the run
KPIS = {
    'conn':            ('Achievability %',        1.0,   'steady'),
    'fragments':       ('Components',             1.0,   'steady'),
    'fragments_raw':   ('Raw components',         1.0,   'steady'),
    'relay_bridges':   ('Cross-fragment bridges', 1.0,   'steady'),
    'latency':         ('Latency ms',             1.0,   'steady'),
    'outage':          ('Outage %',               1.0,   'steady'),
    'energy':          ('Energy J/tick',          1.0,   'steady'),
    'overhead':        ('Control OH Mbps',        1.0,   'steady'),
    'reach_restored':  ('UEs reach-restored',     1.0,   'steady'),
    'stranded_ues':    ('Stranded UEs',           1.0,   'steady'),
    'adm_hold':        ('Adm. HOLD events',       1.0,   'total'),
    'adm_throttle':    ('Adm. THROTTLE events',   1.0,   'total'),
    'adm_held_volume': ('Volume zeroed by HOLD',  1.0,   'total'),
}


def steady(series):
    if not series:
        return float('nan')
    tail = series[-STEADY_WINDOW:]
    vals = [v for v in tail if isinstance(v, (int, float)) and not (
        isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float('nan')


def agg(run, key, how):
    v = run.get(key)
    if how == 'total':
        return float(v) if isinstance(v, (int, float)) else float('nan')
    if not isinstance(v, list):
        return float('nan')
    return steady(v) if how == 'steady' else (
        float(v[-1]) if v else float('nan'))


def mean_sd(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float('nan'), float('nan'), 0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
        if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='output')
    ap.add_argument('--json', default='merged_results.json')
    args = ap.parse_args()

    data = {}
    for ck in CHECKPOINTS:
        p = os.path.join(args.dir, 'auth_eval_s%s.pkl' % ck)
        if not os.path.exists(p):
            print('MISSING %s -- merging what exists' % p)
            continue
        with open(p, 'rb') as f:
            data[ck] = pickle.load(f)
    if not data:
        raise SystemExit('no auth_eval pickles found in %s' % args.dir)
    base_ck = CHECKPOINTS[0]
    if base_ck not in data:
        raise SystemExit('the full-width pickle (s%s) is required' % base_ck)

    seeds = data[base_ck]['seeds']
    out = {'seeds': seeds, 'checkpoints': sorted(data), 'scenarios': {}}

    for scen_key, scen_name in (('recovery_multi', 'recovery'),
                                ('rescue_multi', 'rescue_ops')):
        scen_out = {}
        arms = sorted(data[base_ck][scen_key].keys())
        for arm in arms:
            arm_out = {}
            learned = arm in LEARNED_ARMS
            cks = [c for c in sorted(data) if arm in data[c][scen_key]] \
                if learned else [base_ck]
            for kpi, (label, scale, how) in KPIS.items():
                # per-checkpoint list of per-seed values
                per_ck = {}
                for c in cks:
                    runs = data[c][scen_key].get(arm, {})
                    per_ck[c] = {s: agg(runs[s], kpi, how) * scale
                                 for s in seeds if runs.get(s)}
                for cond, keep in (('all', set(seeds)),
                                   ('reunifiable',
                                    set(seeds) - NON_REUNIFIABLE)):
                    # seed-mean per checkpoint, then across checkpoints
                    ck_means = []
                    for c in cks:
                        m, _, n = mean_sd([v for s, v in per_ck[c].items()
                                           if s in keep])
                        if n:
                            ck_means.append(m)
                    m_ck, sd_ck, n_ck = mean_sd(ck_means)
                    # per-seed spread inside the base checkpoint
                    _, sd_seed, n_seed = mean_sd(
                        [v for s, v in per_ck[cks[0]].items() if s in keep])
                    arm_out.setdefault(kpi, {})[cond] = {
                        'mean': m_ck,
                        'sd_across_checkpoints': sd_ck if learned else None,
                        'n_checkpoints': n_ck if learned else 1,
                        'sd_across_seeds': sd_seed,
                        'n_seeds': n_seed,
                    }
            scen_out[arm] = arm_out
        out['scenarios'][scen_name] = scen_out

    with open(args.json, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote %s' % args.json)

    # ── console report ────────────────────────────────────────────────────
    for scen_name, scen_out in out['scenarios'].items():
        for cond in ('all', 'reunifiable'):
            print()
            print('=' * 78)
            print('%s  --  %s seeds' % (scen_name.upper(), cond))
            print('=' * 78)
            hdr = ['arm'] + ['achiev%', 'comps', 'bridges', 'outage%',
                             'holds', 'thr']
            print('%-12s %11s %9s %9s %9s %9s %7s' % tuple(hdr))
            for arm, arm_out in sorted(scen_out.items()):
                def fmt(kpi, prec=1):
                    d = arm_out.get(kpi, {}).get(cond)
                    if not d or math.isnan(d['mean']):
                        return '--'
                    sd = d['sd_across_checkpoints']
                    if sd is not None and not math.isnan(sd):
                        return '%.*f+/-%.*f' % (prec, d['mean'], prec, sd)
                    return '%.*f(%.*f)' % (prec, d['mean'], prec,
                                           d['sd_across_seeds'])
                print('%-12s %11s %9s %9s %9s %9s %7s' % (
                    arm, fmt('conn'), fmt('fragments', 2),
                    fmt('relay_bridges', 2), fmt('outage'),
                    fmt('adm_hold', 0), fmt('adm_throttle', 0)))
            print()
            print('  learned arms: mean +/- s.d. ACROSS the independently '
                  'trained checkpoints;')
            print('  other arms:   mean over seeds with (s.d. across seeds) '
                  'in parentheses.')


if __name__ == '__main__':
    main()
