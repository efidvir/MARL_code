# -*- coding: utf-8 -*-
"""Post-hoc, descriptive analysis of the untrained-actor control
(see confirmatory/UNTRAINED_CONTROL.md).  Written before any untrained run.

Reads, for the confirmatory seeds 80-99:
  - the untrained-actor runs: output/untrained_s<SEED>_80_99.pkl (arm name
    'marl_static'; the frozen executor running an untrained actor);
  - the trained frozen executor: ~/MARL_run_6/output/cf_static_80_89.pkl and
    cf_static_90_99.pkl (arm 'marl_static');
  - the baselines: ~/MARL_run_5/output/cf_eval_80_99.pkl.
Every comparison is paired per instance and descriptive; there is no decision
rule and no multiplicity correction.  Untrained pickles are loaded one by one
because they share the arm name of the trained executor.

Usage: python3 untrained_analysis.py [--seeds 1234,2345,3456,4567] --out REPORT.json
"""
import argparse
import json
import os
import pickle
import statistics as st

from confirmatory_analysis import steady, wmean, jpu, paired

HOME = os.path.expanduser('~')
SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
KIND = {'A': 'recovery', 'B': 'rescue_ops'}
ROUTING = ('ospf', 'olsr', 'batman', 'aodv')
SEEDS = list(range(80, 100))


def arm_runs(paths, arm):
    out = {k: {} for k in SCEN}
    for p in paths:
        P = pickle.load(open(p, 'rb'))
        for k, sk in SCEN.items():
            for s, r in P.get(sk, {}).get(arm, {}).items():
                if r:
                    out[k][int(s)] = r
    return out


def fp_ok(runs):
    """Static-executor audit: start and end fingerprints equal on every pass."""
    fps, bad = set(), 0
    for k in SCEN:
        for r in runs[k].values():
            a, b = r.get('static_fp_start'), r.get('static_fp_end')
            if a is None or a != b:
                bad += 1
            fps.add(a)
    return bad, fps


def summary(runs, opt):
    row = {}
    for k in SCEN:
        R = runs[k]
        ss = [s for s in SEEDS if s in R]
        row[k] = dict(
            n=len(ss),
            ach=st.mean(steady(R[s]['conn']) for s in ss),
            ach_sd=st.stdev(steady(R[s]['conn']) for s in ss),
            comps=st.mean(steady(R[s]['fragments']) for s in ss),
            bridges=st.mean(steady(R[s]['relay_bridges']) for s in ss) if all('relay_bridges' in R[s] for s in ss) else None,
            at_opt=sum(1 for s in ss if round(steady(R[s]['fragments']), 6) <= opt[(KIND[k], s)]['attainable_optimum']),
            jpu_pooled=sum(sum(x for x in R[s]['energy'] if isinstance(x, (int, float))) for s in ss)
            / sum(sum(x for x in R[s]['delivered'] if isinstance(x, (int, float))) for s in ss),
        )
        if k == 'B':
            row[k]['pre'] = st.mean(wmean(R[s]['conn'], 900, 1400) for s in ss)
    return row


def diff(a, b, k, key=lambda r: steady(r['conn'])):
    return [key(a[k][s]) - key(b[k][s]) for s in SEEDS]


def fmt_p(name, d):
    p = paired(d, +1)
    return ('  %-58s mean %+7.2f  sd %5.2f  95%% CI [%+.2f, %+.2f]  positive on %2d/%d  sign p %.4f  Wilcoxon p %.5f'
            % (name, p['mean'], p['sd'], p['ci_lo'], p['ci_hi'], p['wins'], p['n'], p['sign_p'], p['wilcoxon_p'])), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1234,2345,3456,4567')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    opt = {}
    for inst in json.load(open('benchmark_manifest_80_99.json'))['instances']:
        opt[(inst['scenario_kind'], int(inst['topology_seed']))] = inst
    base_pkl = [HOME + '/MARL_run_5/output/cf_eval_80_99.pkl']
    trained = arm_runs([HOME + '/MARL_run_6/output/cf_static_80_89.pkl',
                        HOME + '/MARL_run_6/output/cf_static_90_99.pkl'], 'marl_static')
    base = {arm: arm_runs(base_pkl, arm) for arm in ('xldet', 'sdn', 'random') + ROUTING}
    report = {'trained_frozen_s1234': summary(trained, opt), 'untrained': {}, 'paired': {}}
    print('== trained frozen executor (s1234), confirmatory set ==')
    print('  ', json.dumps(report['trained_frozen_s1234']))
    for arm in ('random', 'xldet'):
        report[arm] = summary(base[arm], opt)
    unt = {}
    for sd in [int(x) for x in a.seeds.split(',') if x]:
        path = 'output/untrained_s%d_80_99.pkl' % sd
        if not os.path.exists(path):
            print('-- untrained s%d: no result file (%s)' % (sd, path))
            continue
        runs = arm_runs([path], 'marl_static')
        bad, fps = fp_ok(runs)
        missing = {k: sorted(set(SEEDS) - set(runs[k])) for k in SCEN}
        print('\n== untrained actor, torch seed %d ==  fingerprint audit: %d bad passes, %d distinct fingerprints; missing %s'
              % (sd, bad, len(fps), missing))
        if any(missing.values()):
            print('   incomplete -- skipped')
            continue
        unt[sd] = runs
        report['untrained'][sd] = dict(summary(runs, opt), fp_bad=bad, n_fp=len(fps))
        print('  ', json.dumps(report['untrained'][sd]))
        pr = {}
        for k in SCEN:
            line, p = fmt_p('%s achievability: trained frozen - untrained s%d' % (k, sd), diff(trained, runs, k))
            print(line); pr['%s_ach_trained_minus_untrained' % k] = p
            line, p = fmt_p('%s achievability: untrained s%d - uniform random' % (k, sd), diff(runs, base['random'], k))
            print(line); pr['%s_ach_untrained_minus_random' % k] = p
            best = {s: max(steady(base[r][k][s]['conn']) for r in ROUTING) for s in SEEDS}
            d = [steady(runs[k][s]['conn']) - best[s] for s in SEEDS]
            line, p = fmt_p('%s achievability: untrained s%d - best routing' % (k, sd), d)
            print(line); pr['%s_ach_untrained_minus_best_routing' % k] = p
            line, p = fmt_p('%s components: untrained s%d - trained frozen' % (k, sd),
                            diff(runs, trained, k, key=lambda r: steady(r['fragments'])))
            print(line); pr['%s_comps_untrained_minus_trained' % k] = p
        line, p = fmt_p('A energy/unit: untrained s%d - trained frozen' % sd,
                        diff(runs, trained, 'A', key=jpu))
        print(line); pr['A_jpu_untrained_minus_trained'] = p
        report['paired'][sd] = pr
    if len(unt) > 1:
        print('\n== mean over the %d untrained actors (per instance) ==' % len(unt))
        mean_unt = {k: {s: st.mean(steady(unt[sd][k][s]['conn']) for sd in unt) for s in SEEDS} for k in SCEN}
        pr = {}
        for k in SCEN:
            d = [steady(trained[k][s]['conn']) - mean_unt[k][s] for s in SEEDS]
            line, p = fmt_p('%s achievability: trained frozen - mean untrained' % k, d)
            print(line); pr['%s_ach_trained_minus_mean_untrained' % k] = p
            print('  %s achievability by untrained actor: %s' % (k, {sd: round(report['untrained'][sd][k]['ach'], 2) for sd in unt}))
        report['paired']['mean'] = pr
    json.dump(report, open(a.out, 'w'), indent=1, default=str)
    print('\nwritten', a.out)


if __name__ == '__main__':
    main()
