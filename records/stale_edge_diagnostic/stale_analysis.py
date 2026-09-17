# -*- coding: utf-8 -*-
"""Post-hoc analysis of the stale relay-edge diagnostic (DIAG_STALE_EDGES.md).

  python3 stale_analysis.py fix   -> pre-registered analysis with the affected
                                     arms from R-fix; paired R-fix minus frozen
  python3 stale_analysis.py diag  -> identity check against the frozen runs and
                                     stale-edge usage in them
"""
import glob
import os
import pickle
import statistics as st
import subprocess
import sys

from confirmatory_analysis import steady, jpu, paired

H = os.path.expanduser('~')
SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
ARMS = ('marl_static', 'xldet', 'random')
SEEDS = list(range(80, 100))
FROZEN = {
    'marl_static': [H + '/MARL_run_6/output/cf_static_80_89.pkl', H + '/MARL_run_6/output/cf_static_90_99.pkl'],
    'xldet': [H + '/MARL_run_5/output/cf_eval_80_99.pkl'],
    'random': [H + '/MARL_run_5/output/cf_eval_80_99.pkl'],
}


def load(paths, arm):
    out = {k: {} for k in SCEN}
    for p in paths:
        P = pickle.load(open(p, 'rb'))
        for k, sk in SCEN.items():
            for s, r in P.get(sk, {}).get(arm, {}).items():
                if r:
                    out[k][int(s)] = r
    return out


def num(v):
    return [x for x in v if isinstance(x, (int, float))]


def line(name, d, expect=+1):
    p = paired(d, expect)
    return ('  %-52s mean %+8.3f  95%% CI [%+.3f, %+.3f]  >0 on %2d/%d  Wilcoxon p %.4f'
            % (name, p['mean'], p['ci_lo'], p['ci_hi'], sum(1 for x in d if x > 0), len(d), p['wilcoxon_p']))


def main(mode):
    tag = mode
    paths = sorted(glob.glob('output/%s_*.pkl' % tag))
    print('%s pickles: %s' % (tag, paths))
    new = {a: load(paths, a) for a in ARMS}
    old = {a: load(FROZEN[a], a) for a in ARMS}
    for a in ARMS:
        for k in SCEN:
            miss = sorted(set(SEEDS) - set(new[a][k]))
            if miss:
                print('MISSING %s %s: %s' % (a, k, miss))
    if mode == 'diag':
        print('\n== identity of R-diag decisions with the frozen runs ==')
        for a in ARMS:
            for k in SCEN:
                keys = ('conn', 'delivered', 'fragments', 'relay_bridges', 'relay_count', 'link_count')
                bad = [(s, key) for s in SEEDS if s in new[a][k] and s in old[a][k]
                       for key in keys if new[a][k][s][key] != old[a][k][s][key]]
                print('  %-12s %s: %d instance-series differ %s' % (a, k, len(bad), bad[:6]))
        print('\n== stale-edge use in the evaluated (frozen) behaviour ==')
        for a in ARMS:
            for k in SCEN:
                R = new[a][k]
                se_all, se_ss, share = [], [], []
                for s in SEEDS:
                    if s not in R:
                        continue
                    r = R[s]
                    se = num(r['stale_edges'])
                    se_all.append(sum(se) / len(se))
                    se_ss.append(steady(r['stale_edges']))
                    u2u = [c / 100.0 * o for c, o in zip(r['conn'], r['offered'])
                           if isinstance(c, (int, float)) and isinstance(o, (int, float))]
                    sv = num(r['stale_u2u_volume'])
                    share.append(100.0 * sum(sv) / max(1e-9, sum(u2u)))
                if not se_all:
                    continue
                print('  %-12s %s: stale edges mean %.2f (steady %.2f); share of delivered u2u volume '
                      'crossing a stale edge: mean %.1f%%, min %.1f%%, max %.1f%%' % (
                          a, k, st.mean(se_all), st.mean(se_ss), st.mean(share), min(share), max(share)))
        return
    print('\n== R-fix minus frozen, paired per instance ==')
    for a in ARMS:
        for k in SCEN:
            ss = [s for s in SEEDS if s in new[a][k] and s in old[a][k]]
            if not ss:
                continue
            d_ach = [steady(new[a][k][s]['conn']) - steady(old[a][k][s]['conn']) for s in ss]
            d_cmp = [steady(new[a][k][s]['fragments']) - steady(old[a][k][s]['fragments']) for s in ss]
            d_brg = [steady(new[a][k][s]['relay_bridges']) - steady(old[a][k][s]['relay_bridges']) for s in ss]
            d_jpu = [jpu(new[a][k][s]) - jpu(old[a][k][s]) for s in ss]
            print(' %s %s (n=%d): fixed ach %.2f vs frozen %.2f; comps %.2f vs %.2f' % (
                a, k, len(ss), st.mean(steady(new[a][k][s]['conn']) for s in ss),
                st.mean(steady(old[a][k][s]['conn']) for s in ss),
                st.mean(steady(new[a][k][s]['fragments']) for s in ss),
                st.mean(steady(old[a][k][s]['fragments']) for s in ss)))
            print(line('achievability (pp)', d_ach))
            print(line('components', d_cmp))
            print(line('bridges', d_brg))
            print(line('energy per unit', d_jpu))
    # pre-registered analysis with the affected arms replaced (later pickles
    # overwrite earlier ones arm by arm in confirmatory_analysis.load)
    pk = ','.join([H + '/MARL_run_5/output/cf_eval_80_99.pkl'] + paths)
    print('\n== pre-registered analysis, affected arms from R-fix (post hoc) ==', flush=True)
    subprocess.run([sys.executable, 'confirmatory_analysis.py', '--pickles', pk,
                    '--manifests', 'benchmark_manifest_80_99.json', '--marl-arm', 'marl_static',
                    '--out', 'confirmatory/stale_fix_prereg.json'], check=False)


if __name__ == '__main__':
    main(sys.argv[1])
