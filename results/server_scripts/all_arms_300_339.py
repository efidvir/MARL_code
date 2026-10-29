#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Every arm on the R3-lite topology seeds 300-339 x Scenarios A, B: per-arm
mean and 95 % t-interval over the 40 instances, and paired differences of
replicate 1234 and of the four-replicate ensemble against every comparator.

DATA (all read-only, no pass is run here):
  ~/MARL_r3lite/output/r3lite/s6_cf_b0..9.pkl   R3-lite S6, 8 arms x 40 seeds x A,B
        (marl_static = replicate 1234, xldet, sdn, random, ospf, olsr, batman, aodv)
  ~/MARL_r3lite/output/r3lite/s9x_ensemble.pkl  R3-lite S9x, marl_static = ensemble
  <server-data>/r4/cb/passes/             R4 Check B passes, config tu1 (T*)
        grid <server-data>/r4/gate_a/CB_filled.json

METRICS: the functions of confirmatory_analysis.py (md5 pinned in chain_plan
R3_ANALYSIS_MD5), through tools/r4_checkb_pair.run_metrics exactly as Check B:
  ach        CA.steady(conn)  (mean of the last 500 ticks of the per-tick
             user-to-user achievability min(100, 100 delivered/offered))
  m7         CA.m7_teardowns_per_1000(run, CA.T_SEV[k])
  energy     CA.steady(energy)
  jpu        CA.jpu(run) = sum(energy) / sum(delivered)  (per instance)
  components CA.steady(fragments)
Validity: the frozen run-level rules (r4_checkb_pair.marl_validity for the
R3-lite runs, r4_comparator_summary.pass_validity for the T* passes).

CROSS-CHECKS (fail loudly, never vacuous: each records how many values it
compared): the R3-lite family-1 report (descriptive means, P3/P6 against the
best routing protocol), the family-E report (ensemble descriptive means) and
Check B's per-seed values (T* and replicate 1234).

--per-seed-out PATH (v11-direct extension, 2026-10-20).  Written only after
the outputs above are complete and their three cross-checks have passed, and
it changes nothing in them.  It exports the per-instance values of every arm
(the 11 above plus replicate 1234's untrained actor, R3-lite S8 pickle
s8_untrained_s1234.pkl, the S12 input) and every scenario, lists aligned with
`seeds`, plus the per-instance energy/delivery sums (pooled J/unit) and the
attainable optimum of the R3-lite manifest.  Its own cross-checks, each
counted and exact (1e-9): the three above, re-run on the exported lists;
P1-P9 and E-P1-E-P9 means and wins of the two reports; ach_min, ach_max,
below_floor and at_optimum of the reports; the untrained arm against S12 of
the full report; a JSON round trip.  Any failure: the file is renamed
*.FAILED and the exit code is 5.
"""
import argparse
import glob
import json
import math
import os
import pickle
import statistics as st
import sys

TREE = os.path.expanduser(os.environ.get('R4_TREE', '~/r4_work/tree'))
for p in (TREE, os.path.join(TREE, 'tools'), os.path.join(TREE, 'confirmatory')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                              # noqa: E402
import confirmatory_analysis as CA              # noqa: E402
import r4_comparator_summary as SUM             # noqa: E402
import r4_checkb_pair as CBP                    # noqa: E402
import chain_plan as CP                         # noqa: E402

R3 = os.path.expanduser('~/MARL_r3lite')
S6_GLOB = os.path.join(R3, 'output/r3lite/s6_cf_b*.pkl')
S9X = os.path.join(R3, 'output/r3lite/s9x_ensemble.pkl')
REP1 = os.path.join(R3, 'confirmatory/r3lite_cf_report.json')
REPE = os.path.join(R3, 'confirmatory/r3lite_cf_report_family_e.json')
CB_OUT = '<server-data>/r4/cb'
CB_GRID = '<server-data>/r4/gate_a/CB_filled.json'
CB_JSON = '<server-data>/r4/cb/checkb_300_339.json'
TSTAR = 'tu1'
# v11-direct extension (--per-seed-out)
S8U = os.path.join(R3, 'output/r3lite/s8_untrained_s1234.pkl')
REPF = os.path.join(R3, 'confirmatory/r3lite_cf_report_full.json')
MANIFEST = os.path.join(R3, 'confirmatory/r3lite_manifest_300_339.json')
UNTRAINED = 'untrained_s1234'
FLOOR = 10.0          # CA descriptive below_floor: ach < 10.0
SEEDS = list(range(300, 340))
FIELDS = ('ach', 'm7', 'energy', 'jpu', 'components')
# the good direction of each metric (for `wins` and CA.paired's expect)
GOOD = {'ach': +1, 'm7': -1, 'energy': -1, 'jpu': -1, 'components': -1}
S6_ARMS = {'marl_static': 'ck1234', 'xldet': 'xldet', 'sdn': 'sdn', 'random': 'random',
           'ospf': 'ospf', 'olsr': 'olsr', 'batman': 'batman', 'aodv': 'aodv'}
ROUTING = ('ospf', 'olsr', 'batman', 'aodv')
ARM_ORDER = ('ck1234', 'ensemble', 'tstar_tu1', 'xldet', 'sdn',
             'ospf', 'olsr', 'batman', 'aodv', 'best_routing', 'random')
COMPARATORS = ('tstar_tu1', 'xldet', 'sdn', 'best_routing', 'ospf', 'olsr', 'batman', 'aodv', 'random')
TOL = 1e-9


def fnum(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def load_r3lite(paths, armmap, expect_reps):
    """{name: {(k, s): metrics}}, provenance records, violations, pass counts."""
    out = {n: {} for n in armmap.values()}
    sums = {n: {} for n in armmap.values()}
    provs, bad, counts = {}, [], {'runs': 0, 'pass_stats_ok': 0, 'pass_stats_total': 0}
    for p in paths:
        with open(p, 'rb') as f:
            P = pickle.load(f)
        prov = P.get('provenance') or {}
        rec, pbad = CBP.learned_provenance(prov, expect_reps)
        rec['arms'] = prov.get('arms')
        rec['argv'] = prov.get('argv')
        provs[p] = rec
        bad += ['%s: %s' % (os.path.basename(p), b) for b in pbad]
        ps = P.get('pass_stats') or {}
        counts['pass_stats_total'] += len(ps)
        counts['pass_stats_ok'] += sum(1 for v in ps.values() if v.get('ok'))
        for k, sk in CA.SCEN.items():
            for arm, name in armmap.items():
                for s, run in ((P.get(sk) or {}).get(arm) or {}).items():
                    s = int(s)
                    if s not in SEEDS or not run:
                        continue
                    if (k, s) in out[name]:
                        raise SystemExit('%s %s seed %d twice' % (name, k, s))
                    why = CBP.marl_validity(run, arm, k, s)
                    if why:
                        bad.append('%s %s %s s%d invalid: %s' % (os.path.basename(p), name, k, s, why))
                        continue
                    out[name][(k, s)] = CBP.run_metrics(run, k)
                    sums[name][(k, s)] = (sum(CA.num(run['energy'])), sum(CA.num(run['delivered'])),
                                          CA.steady(run['delivered']))
                    counts['runs'] += 1
        del P
    return out, sums, provs, bad, counts


def describe(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    if not v:
        return {'n': 0}
    mu, sd, lo, hi = CA.t_ci(v)
    return {'n': len(v), 'mean': mu, 'sd': sd, 'ci_lo': lo, 'ci_hi': hi,
            'median': st.median(v), 'min': min(v), 'max': max(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--per-seed-out', default=None,
                    help='also write the per-instance values of every arm (v11-direct)')
    a = ap.parse_args()
    rng = np.random.default_rng(1)

    s6 = sorted(glob.glob(S6_GLOB))
    if len(s6) != 10:
        raise SystemExit('expected 10 S6 pickles, found %d' % len(s6))
    M, SUMS, prov, bad, cnt6 = load_r3lite(s6, S6_ARMS, [1234])
    E, ESUMS, eprov, ebad, cnte = load_r3lite([S9X], {'marl_static': 'ensemble'},
                                              list(CP.R2_CHECKPOINT_SHA256))
    M.update(E)
    SUMS.update(ESUMS)
    prov.update(eprov)
    bad += ebad

    # T*: the Check B passes of config tu1, loaded by Check B's own loader
    comp, ctrees = CBP.load_comparators(CB_OUT, [CB_GRID], [TSTAR], set(SEEDS))
    with open(CB_JSON, encoding='utf-8') as f:
        cbj = json.load(f)
    gate_tree = cbj['gate_a_tree_sha256']
    cbad = CBP.comparator_tree_violations(ctrees, gate_tree)
    bad += ['T*: %s' % b for b in cbad]
    M['tstar_tu1'] = {}
    tstar_n = 0
    for key, (m, why, label, tree) in comp.get(TSTAR, {}).items():
        if why:
            bad.append('T* %s invalid: %s' % (label, why))
            continue
        M['tstar_tu1'][key] = {f: m[f] for f in FIELDS}
        tstar_n += 1
    tstar_meta = {}
    for p in sorted(glob.glob(os.path.join(CB_OUT, 'passes', 'xldet_tu__*.meta.json'))):
        with open(p, encoding='utf-8') as f:
            mj = json.load(f)
        t = mj['task']
        tstar_meta.setdefault('engine_env', t['engine_env'])
        if t['engine_env'] != tstar_meta['engine_env']:
            bad.append('T* engine env differs in %s' % p)
        tstar_meta.setdefault('spec_sha256', t['spec_sha256'])
        tstar_meta.setdefault('spec', t['spec'])
        tstar_meta.setdefault('trees', set()).add((mj['tree_head'], mj['tree_sha256']))
        tstar_meta['n_meta'] = tstar_meta.get('n_meta', 0) + 1
        if mj.get('tree_changed_during_pass') is not False or not mj.get('ok'):
            bad.append('T* meta %s: not ok or tree changed' % p)
    tstar_meta['trees'] = sorted([list(t) for t in tstar_meta.get('trees', set())])
    if tstar_meta.get('engine_env') != CBP.r3lite_eval_env():
        bad.append('T* engine env %s is not r3lite_eval' % tstar_meta.get('engine_env'))

    # best routing protocol per instance (CA's P3/P6/P9 definitions; min for the rest)
    M['best_routing'] = {}
    for k in 'AB':
        for s in SEEDS:
            if all((k, s) in M[r] for r in ROUTING):
                rr = [M[r][(k, s)] for r in ROUTING]
                M['best_routing'][(k, s)] = {
                    'ach': max(x['ach'] for x in rr), 'jpu': min(x['jpu'] for x in rr),
                    'energy': min(x['energy'] for x in rr),
                    'components': min(x['components'] for x in rr),
                    'm7': min(x['m7'] for x in rr if x['m7'] is not None)}

    # completeness: every arm on every (k, s)
    for name in ARM_ORDER:
        for k in 'AB':
            miss = [s for s in SEEDS if (k, s) not in M[name]]
            if miss:
                bad.append('%s %s: missing seeds %s' % (name, k, miss))
    if bad:
        print('VIOLATIONS:\n  ' + '\n  '.join(bad))
        raise SystemExit(2)

    # ---- descriptive
    desc = {}
    for k in 'AB':
        desc[k] = {}
        for name in ARM_ORDER:
            R = M[name]
            d = {f: describe([fnum(R[(k, s)][f]) for s in SEEDS]) for f in FIELDS}
            if name in SUMS:
                es = sum(SUMS[name][(k, s)][0] for s in SEEDS)
                ds = sum(SUMS[name][(k, s)][1] for s in SEEDS)
                d['jpu_pooled'] = es / ds
                d['thr_mean'] = st.mean(SUMS[name][(k, s)][2] for s in SEEDS)
            desc[k][name] = d

    # ---- paired: replicate 1234 and the ensemble against every comparator
    pairs = {}
    for L in ('ck1234', 'ensemble'):
        for Cn in COMPARATORS + (('ck1234',) if L == 'ensemble' else ()):
            blk = {}
            for k in 'AB':
                b = {}
                for f in FIELDS:
                    d = []
                    for s in SEEDS:
                        x, y = M[L][(k, s)][f], M[Cn][(k, s)][f]
                        if x is None or y is None:
                            continue
                        d.append(float(x) - float(y))
                    if len(d) < 2:
                        b[f] = {'n': len(d)}
                        continue
                    r = CA.paired(d, GOOD[f])
                    if f == 'ach':
                        r['boot95'] = SUM.boot_ci(d, 10000, rng)
                    b[f] = r
                blk[k] = b
            pairs['%s - %s' % (L, Cn)] = blk

    # ---- cross-checks against the recorded reports (exact)
    xc = {}
    with open(REP1, encoding='utf-8') as f:
        r1 = json.load(f)
    with open(REPE, encoding='utf-8') as f:
        re_ = json.load(f)
    n, worst = 0, 0.0
    for k in 'AB':
        for arm, name in S6_ARMS.items():
            rd = r1['descriptive'][k][arm]
            for mine, theirs in ((desc[k][name]['ach']['mean'], rd['ach_mean']),
                                 (desc[k][name]['ach']['sd'], rd['ach_sd']),
                                 (desc[k][name]['jpu']['mean'], rd['jpu_mean']),
                                 (desc[k][name]['jpu_pooled'], rd['jpu_pooled']),
                                 (desc[k][name]['components']['mean'], rd['comps_mean']),
                                 (desc[k][name]['energy']['mean'], rd['energy_mean'])):
                worst = max(worst, abs(mine - theirs))
                n += 1
    for P, k in (('P3_A_ach_vs_best_routing', 'A'), ('P6_B_ach_vs_best_routing', 'B'),
                 ('P1_A_ach_vs_xldet', 'A'), ('P4_B_ach_vs_xldet', 'B'),
                 ('P2_A_ach_vs_sdn', 'A'), ('P5_B_ach_vs_sdn', 'B')):
        Cn = 'best_routing' if 'routing' in P else ('xldet' if 'xldet' in P else 'sdn')
        mine = pairs['ck1234 - %s' % Cn][k]['ach']
        for f in ('mean', 'ci_lo', 'ci_hi', 'wilcoxon_p'):
            worst = max(worst, abs(mine[f] - r1['primary'][P][f]))
            n += 1
    mine = pairs['ck1234 - best_routing']['A']['jpu']
    worst = max(worst, abs(mine['mean'] - r1['primary']['P9_A_jpu_vs_best_routing']['mean']))
    n += 1
    xc['family1_report'] = {'file': REP1, 'values_compared': n, 'max_abs_diff': worst,
                            'pass': bool(n == 8 * 2 * 6 + 6 * 4 + 1 and worst < TOL)}
    n, worst = 0, 0.0
    for k in 'AB':
        rd = re_['descriptive'][k]['marl_static']
        for mine, theirs in ((desc[k]['ensemble']['ach']['mean'], rd['ach_mean']),
                             (desc[k]['ensemble']['jpu']['mean'], rd['jpu_mean']),
                             (desc[k]['ensemble']['jpu_pooled'], rd['jpu_pooled']),
                             (desc[k]['ensemble']['components']['mean'], rd['comps_mean'])):
            worst = max(worst, abs(mine - theirs))
            n += 1
        for arm, name in S6_ARMS.items():
            if arm == 'marl_static':
                continue
            worst = max(worst, abs(desc[k][name]['ach']['mean']
                                   - re_['descriptive'][k][arm]['ach_mean']))
            n += 1
    xc['family_e_report'] = {'file': REPE, 'values_compared': n, 'max_abs_diff': worst,
                             'pass': bool(n == 2 * (4 + 7) and worst < TOL)}
    n, worst = 0, 0.0
    cbp = cbj['pairs']['ck1234 - %s' % TSTAR]
    for k in 'AB':
        for s in SEEDS:
            ps = cbp[k]['per_seed'][str(s)]
            for f in FIELDS:
                for side, name in (('comparator', 'tstar_tu1'), ('learned', 'ck1234')):
                    x, y = M[name][(k, s)][f], ps[side][f]
                    worst = max(worst, abs(float(x) - float(y)))
                    n += 1
    cbe = cbj['pairs']['ensemble - %s' % TSTAR]
    for k in 'AB':
        worst = max(worst, abs(pairs['ensemble - tstar_tu1'][k]['ach']['mean'] - cbe[k]['ach_diff_mean']))
        n += 1
    xc['checkb_report'] = {'file': CB_JSON, 'values_compared': n, 'max_abs_diff': worst,
                           'pass': bool(n == 2 * 40 * 5 * 2 + 2 and worst < TOL)}
    if not all(v['pass'] for v in xc.values()):
        print('CROSS-CHECK FAILED: %s' % json.dumps(xc, indent=1))
        raise SystemExit(3)

    rep = {
        'what': 'every arm on R3-lite topology seeds 300-339 x Scenarios A (recovery) and B '
                '(rescue_ops), 3600 ticks; per-arm mean and 95 % t-interval over the 40 instances; '
                'paired differences learned - comparator per instance',
        'seeds': SEEDS,
        'metrics': {'ach': 'CA.steady(conn): mean of the last %d ticks of per-tick user-to-user '
                           'achievability min(100, 100 delivered/offered)' % CA.W,
                    'm7': 'CA.m7_teardowns_per_1000(run, CA.T_SEV[k]) relay-link teardowns per 1000 '
                          'post-event ticks',
                    'energy': 'CA.steady(energy) per-tick energy, last %d ticks' % CA.W,
                    'jpu': 'CA.jpu(run) = sum(energy)/sum(delivered) per instance; jpu_pooled = '
                           'sum over instances / sum over instances (the confirmatory J/u pool)',
                    'components': 'CA.steady(fragments) connected components, last %d ticks' % CA.W,
                    'best_routing': 'per instance: max ach, min jpu/energy/components/m7 over '
                                    'OSPF, OLSR, BATMAN, AODV (CA P3/P6/P9 definitions for ach/jpu)',
                    'ci': 'CA.t_ci (Student t, n-1 dof); paired = CA.paired (exact Wilcoxon, sign '
                          'test, t CI), ach also bootstrap 95 %% (B=10000, seed 1); wins = instances '
                          'where the learned arm is better in the metric good direction %s' % GOOD},
        'descriptive': desc,
        'paired': pairs,
        'cross_checks': xc,
        'provenance': {
            'analysis_code': {'confirmatory_analysis_md5': CP.R3_ANALYSIS_MD5,
                              'tree_imported_from': TREE,
                              'loaders': 'tools/r4_checkb_pair.py (marl_validity, run_metrics, '
                                         'load_comparators, learned_provenance)'},
            'r3lite_pickles': {p: v for p, v in prov.items()},
            'r3lite_pass_counts': {'s6': cnt6, 's9x': cnte},
            'tstar': {'config': TSTAR, 'grid': CB_GRID, 'out': CB_OUT, 'valid_passes': tstar_n,
                      'gate_a_tree_sha256': gate_tree, **tstar_meta},
            'engine_env_r3lite_eval': CBP.r3lite_eval_env(),
            'mixed_tree_note': CBP.MIXED_REASON,
            'no_new_passes': 'the routing-protocol passes on 300-339 already exist in the R3-lite '
                             'S6 pickles (all 8 arms, 64 passes per block); nothing was re-run',
        },
    }
    with open(a.out_json, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(SUM._jsonable(rep), f, indent=1, sort_keys=True, default=str)
        f.write('\n')

    # ---- markdown
    LBL = {'ck1234': 'MARL replicate 1234', 'ensemble': 'MARL ensemble (4 replicates)',
           'tstar_tu1': 'XL-DET tuned (T* = tu1)', 'xldet': 'XL-DET', 'sdn': 'SDN',
           'ospf': 'OSPF', 'olsr': 'OLSR', 'batman': 'BATMAN', 'aodv': 'AODV',
           'best_routing': 'best routing (per instance)', 'random': 'random'}

    def ci(d, fmt='%.1f'):
        if not d or d.get('n', 0) == 0:
            return 'n/a'
        return (fmt + ' [' + fmt + ', ' + fmt + ']') % (d['mean'], d['ci_lo'], d['ci_hi'])

    L = ['# All arms on R3-lite seeds 300-339 (40 instances per scenario, 3600 ticks)', '',
         'Mean [95 % t-interval] over the 40 instances. Achievability = mean of the last 500 ticks of '
         'min(100, 100 delivered/offered), user-to-user (pp). M7 = relay-link teardowns per 1000 '
         'post-event ticks. J/unit = energy per delivered unit (per-instance mean; pooled in '
         'parentheses). Components = connected components, last 500 ticks.', '']
    for k, nm in (('A', 'Scenario A (recovery)'), ('B', 'Scenario B (rescue operations)')):
        L += ['## %s' % nm, '',
              '| Arm | Achievability (pp) | M7 (/1000 ticks) | J/unit | Components |',
              '|---|---|---|---|---|']
        for name in ARM_ORDER:
            d = desc[k][name]
            jp = ci(d['jpu'], '%.4f') + ((' (%.4f)' % d['jpu_pooled']) if 'jpu_pooled' in d else '')
            L.append('| %s | %s | %s | %s | %s |' % (LBL[name], ci(d['ach']), ci(d['m7'], '%.2f'),
                                                     jp, ci(d['components'], '%.2f')))
        L.append('')
    L += ['## Paired differences (learned minus comparator, per instance, n = 40)', '',
          'Achievability difference in pp: mean [95 % t-interval], wins/40, exact Wilcoxon p. '
          'J/unit difference: mean [95 % t-interval] (negative = learned arm uses less energy per '
          'delivered unit).', '']
    for L_ in ('ck1234', 'ensemble'):
        L += ['### %s' % LBL[L_], '',
              '| Comparator | A: d ach | A wins | A p | B: d ach | B wins | B p | A: d J/unit | B: d J/unit |',
              '|---|---|---|---|---|---|---|---|---|']
        for Cn in COMPARATORS + (('ck1234',) if L_ == 'ensemble' else ()):
            b = pairs['%s - %s' % (L_, Cn)]

            def pc(r, fmt='%+.2f'):
                return (fmt + ' [' + fmt + ', ' + fmt + ']') % (r['mean'], r['ci_lo'], r['ci_hi'])
            L.append('| %s | %s | %d | %.2g | %s | %d | %.2g | %s | %s |' % (
                LBL[Cn], pc(b['A']['ach']), b['A']['ach']['wins'], b['A']['ach']['wilcoxon_p'],
                pc(b['B']['ach']), b['B']['ach']['wins'], b['B']['ach']['wilcoxon_p'],
                pc(b['A']['jpu'], '%+.4f'), pc(b['B']['jpu'], '%+.4f')))
        L.append('')
    L += ['## Provenance', '',
          '- Routing protocols, SDN, random, XL-DET, replicate 1234: R3-lite confirmatory S6 pickles '
          '`~/MARL_r3lite/output/r3lite/s6_cf_b0..9.pkl` (tree c6c5de5, engine pin 6b53f3f), '
          '%d runs read, pass_stats %d/%d ok.' % (cnt6['runs'], cnt6['pass_stats_ok'], cnt6['pass_stats_total']),
          '- Ensemble: `s9x_ensemble.pkl` (R3-lite S9x), %d runs, pass_stats %d/%d ok.'
          % (cnte['runs'], cnte['pass_stats_ok'], cnte['pass_stats_total']),
          '- T* (tu1): R4 Check B passes `<server-data>/r4/cb/passes/xldet_tu__%s__*` '
          '(tree %s), %d valid passes.' % (tstar_meta.get('spec_sha256', '')[:12],
                                          tstar_meta.get('trees'), tstar_n),
          '- Engine environment of every pass: `%s`.' % ' '.join(
              '%s=%s' % kv for kv in sorted(CBP.r3lite_eval_env().items())),
          '- Metric code: confirmatory_analysis.py md5 %s via tools/r4_checkb_pair.run_metrics.'
          % CP.R3_ANALYSIS_MD5,
          '- Cross-checks (exact, tolerance 1e-9): family-1 report %d values, max |diff| %.1e; '
          'family-E report %d values, %.1e; Check B report %d values, %.1e.' % (
              xc['family1_report']['values_compared'], xc['family1_report']['max_abs_diff'],
              xc['family_e_report']['values_compared'], xc['family_e_report']['max_abs_diff'],
              xc['checkb_report']['values_compared'], xc['checkb_report']['max_abs_diff']), '']
    with open(a.out_md, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    if a.per_seed_out:
        return per_seed_export(a.per_seed_out, M, SUMS, desc, xc, r1, re_, cbj, rep['provenance'])
    return 0


# ---------------------------------------------------------------- v11-direct
def load_untrained(path):
    """Replicate 1234's untrained actor on 300-339 (R3-lite S8 pickle, S12
    input): the same run-level validity rules and metrics as the learned runs;
    provenance = the r3lite_eval engine environment, 3600 ticks and exactly
    the pinned initial-actor digest chain_plan.R2_UNTRAINED_SHA256[1234]."""
    with open(path, 'rb') as f:
        P = pickle.load(f)
    prov = P.get('provenance') or {}
    env = dict(prov.get('marl_env') or {})
    core = {k: v for k, v in env.items() if k not in CBP.RUN_SPECIFIC_ENV}
    want = [CP.R2_UNTRAINED_SHA256[1234]]
    digests = list(prov.get('checkpoint_sha256') or [])
    bad = []
    if core != CBP.r3lite_eval_env():
        bad.append('untrained: engine environment %s is not r3lite_eval' % core)
    if prov.get('total_ticks') != CA.T_TOTAL:
        bad.append('untrained: total_ticks %r' % prov.get('total_ticks'))
    if digests != want:
        bad.append('untrained: checkpoint digests %s, expected %s' % (digests, want))
    ps = P.get('pass_stats') or {}
    counts = {'runs': 0, 'pass_stats_total': len(ps),
              'pass_stats_ok': sum(1 for v in ps.values() if v.get('ok'))}
    out, sums = {}, {}
    for k, sk in CA.SCEN.items():
        for s, run in ((P.get(sk) or {}).get('marl_static') or {}).items():
            s = int(s)
            if s not in SEEDS or not run:
                continue
            if (k, s) in out:
                raise SystemExit('untrained %s seed %d twice' % (k, s))
            why = CBP.marl_validity(run, 'marl_static', k, s)
            if why:
                bad.append('untrained %s s%d invalid: %s' % (k, s, why))
                continue
            out[(k, s)] = CBP.run_metrics(run, k)
            sums[(k, s)] = (sum(CA.num(run['energy'])), sum(CA.num(run['delivered'])),
                            CA.steady(run['delivered']))
            counts['runs'] += 1
    rec = {'pickle': path, 'tree_head': prov.get('tree_head'), 'tree_sha256': prov.get('tree_sha256'),
           'marl_env': env, 'checkpoint': prov.get('checkpoint'), 'checkpoint_sha256': digests,
           'pinned': want, 'total_ticks': prov.get('total_ticks'),
           'created_utc': prov.get('created_utc'), 'arms': prov.get('arms'), 'counts': counts}
    del P
    return out, sums, rec, bad


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def per_seed_export(path, M, SUMS, desc, xc_main, r1, re_, cbj, prov_main):
    U, USUMS, urec, ubad = load_untrained(S8U)
    for k in 'AB':
        miss = [s for s in SEEDS if (k, s) not in U]
        if miss:
            ubad.append('untrained %s: missing seeds %s' % (k, miss))
    if ubad:
        print('PER-SEED VIOLATIONS:\n  ' + '\n  '.join(ubad))
        raise SystemExit(4)
    MM = dict(M)
    MM[UNTRAINED] = U
    SS = dict(SUMS)
    SS[UNTRAINED] = USUMS
    arms = ARM_ORDER + (UNTRAINED,)
    with open(MANIFEST, encoding='utf-8') as f:
        man = json.load(f)
    opt = {(i['scenario_kind'], int(i['topology_seed'])): i for i in man['instances']}
    per = {}
    for k in 'AB':
        per[k] = {}
        for name in arms:
            d = {f: [fnum(MM[name][(k, s)][f]) for s in SEEDS] for f in FIELDS}
            if name in SS:
                d['energy_sum'] = [float(SS[name][(k, s)][0]) for s in SEEDS]
                d['delivered_sum'] = [float(SS[name][(k, s)][1]) for s in SEEDS]
            per[k][name] = d
    optima = {k: {'attainable_optimum': [int(opt[(CA.KIND[k], s)]['attainable_optimum']) for s in SEEDS],
                  'raw_components': [int(opt[(CA.KIND[k], s)]['raw_components']) for s in SEEDS]}
              for k in 'AB'}
    with open(REPF, encoding='utf-8') as f:
        rf = json.load(f)
    xc = {}

    def chk(name, pairs_, exact_int=False, extra=None):
        n, worst, bad = 0, 0.0, []
        for lab, mine, theirs in pairs_:
            if mine is None or theirs is None:
                bad.append('%s: missing (%r, %r)' % (lab, mine, theirs))
                continue
            dif = abs(float(mine) - float(theirs))
            worst = max(worst, dif)
            n += 1
            if (exact_int and mine != theirs) or dif >= TOL:
                bad.append('%s: %r vs %r' % (lab, mine, theirs))
        xc[name] = {'values_compared': n, 'max_abs_diff': worst, 'mismatches': bad[:20],
                    'pass': bool(n > 0 and not bad and (extra is None or n == extra))}
        if extra is not None:
            xc[name]['expected_count'] = extra

    def diffs(L, Cn, k, f):
        return [a_ - b_ for a_, b_ in zip(per[k][L][f], per[k][Cn][f])]

    # (1) the three exact cross-checks of the main run, re-run on the EXPORTED lists
    pr = []
    for k in 'AB':
        for arm, name in S6_ARMS.items():
            rd = r1['descriptive'][k][arm]
            v = per[k][name]
            pr += [('%s %s ach_mean' % (k, name), st.mean(v['ach']), rd['ach_mean']),
                   ('%s %s ach_sd' % (k, name), st.stdev(v['ach']), rd['ach_sd']),
                   ('%s %s jpu_mean' % (k, name), st.mean(v['jpu']), rd['jpu_mean']),
                   ('%s %s jpu_pooled' % (k, name), sum(v['energy_sum']) / sum(v['delivered_sum']),
                    rd['jpu_pooled']),
                   ('%s %s comps_mean' % (k, name), st.mean(v['components']), rd['comps_mean']),
                   ('%s %s energy_mean' % (k, name), st.mean(v['energy']), rd['energy_mean'])]
    for P, k, Cn in (('P3_A_ach_vs_best_routing', 'A', 'best_routing'),
                     ('P6_B_ach_vs_best_routing', 'B', 'best_routing'),
                     ('P1_A_ach_vs_xldet', 'A', 'xldet'), ('P4_B_ach_vs_xldet', 'B', 'xldet'),
                     ('P2_A_ach_vs_sdn', 'A', 'sdn'), ('P5_B_ach_vs_sdn', 'B', 'sdn')):
        r = CA.paired(diffs('ck1234', Cn, k, 'ach'), +1)
        for f in ('mean', 'ci_lo', 'ci_hi', 'wilcoxon_p'):
            pr.append(('%s %s' % (P, f), r[f], r1['primary'][P][f]))
    pr.append(('P9 mean', st.mean(diffs('ck1234', 'best_routing', 'A', 'jpu')),
               r1['primary']['P9_A_jpu_vs_best_routing']['mean']))
    chk('x1_family1_report_on_export', pr, extra=8 * 2 * 6 + 6 * 4 + 1)
    pr = []
    for k in 'AB':
        rd = re_['descriptive'][k]['marl_static']
        v = per[k]['ensemble']
        pr += [('%s ensemble ach_mean' % k, st.mean(v['ach']), rd['ach_mean']),
               ('%s ensemble jpu_mean' % k, st.mean(v['jpu']), rd['jpu_mean']),
               ('%s ensemble jpu_pooled' % k, sum(v['energy_sum']) / sum(v['delivered_sum']), rd['jpu_pooled']),
               ('%s ensemble comps_mean' % k, st.mean(v['components']), rd['comps_mean'])]
        for arm, name in S6_ARMS.items():
            if arm != 'marl_static':
                pr.append(('%s %s ach_mean (E report)' % (k, name), st.mean(per[k][name]['ach']),
                           re_['descriptive'][k][arm]['ach_mean']))
    chk('x2_family_e_report_on_export', pr, extra=2 * (4 + 7))
    pr = []
    for k in 'AB':
        ps = cbj['pairs']['ck1234 - %s' % TSTAR][k]['per_seed']
        for i, s in enumerate(SEEDS):
            for f in FIELDS:
                for side, name in (('comparator', 'tstar_tu1'), ('learned', 'ck1234')):
                    pr.append(('%s s%d %s %s' % (k, s, name, f), per[k][name][f][i], ps[str(s)][side][f]))
        pr.append(('%s ensemble - T* ach mean' % k, st.mean(diffs('ensemble', 'tstar_tu1', k, 'ach')),
                   cbj['pairs']['ensemble - %s' % TSTAR][k]['ach_diff_mean']))
    chk('x3_checkb_report_on_export', pr, extra=2 * 40 * 5 * 2 + 2)

    # (2) P1-P9 and E-P1-E-P9: mean and wins of every hypothesis, from the export
    pr = []
    HYP = (('P1_A_ach_vs_xldet', 'A', 'ach', 'xldet'), ('P2_A_ach_vs_sdn', 'A', 'ach', 'sdn'),
           ('P3_A_ach_vs_best_routing', 'A', 'ach', 'best_routing'),
           ('P4_B_ach_vs_xldet', 'B', 'ach', 'xldet'), ('P5_B_ach_vs_sdn', 'B', 'ach', 'sdn'),
           ('P6_B_ach_vs_best_routing', 'B', 'ach', 'best_routing'),
           ('P7_A_jpu_vs_xldet', 'A', 'jpu', 'xldet'), ('P8_A_jpu_vs_sdn', 'A', 'jpu', 'sdn'),
           ('P9_A_jpu_vs_best_routing', 'A', 'jpu', 'best_routing'))
    wins_tab = {}
    for fam, L_, rep_ in (('family1', 'ck1234', r1), ('familyE', 'ensemble', re_)):
        for P, k, f, Cn in HYP:
            d = diffs(L_, Cn, k, f)
            e = rep_['primary'][P]['expect']
            w = sum(1 for x in d if x * e > 0)
            wins_tab['%s %s' % (fam, P)] = w
            pr.append(('%s %s mean' % (fam, P), st.mean(d), rep_['primary'][P]['mean']))
            pr.append(('%s %s wins' % (fam, P), w, rep_['primary'][P]['wins']))
    chk('x4_primary_means_and_wins', pr, extra=2 * 9 * 2)

    # (3) worst topology, best topology, below-floor and at-optimum counts
    pr = []
    for k in 'AB':
        o = optima[k]['attainable_optimum']
        for arm, name, rep_ in [(a_, n_, r1) for a_, n_ in S6_ARMS.items()] + [('marl_static', 'ensemble', re_)]:
            rd = rep_['descriptive'][k][arm]
            v = per[k][name]
            pr += [('%s %s ach_min' % (k, name), min(v['ach']), rd['ach_min']),
                   ('%s %s ach_max' % (k, name), max(v['ach']), rd['ach_max']),
                   ('%s %s below_floor' % (k, name), sum(1 for x in v['ach'] if x < FLOOR), rd['below_floor']),
                   ('%s %s at_optimum' % (k, name),
                    sum(1 for c, oo in zip(v['components'], o) if c <= oo + 1e-9), rd['at_optimum'])]
        pr.append(('%s mean_optimum' % k, st.mean(o), r1['descriptive'][k + '_optima']['mean_optimum']))
        pr.append(('%s mean_raw' % k, st.mean(optima[k]['raw_components']),
                   r1['descriptive'][k + '_optima']['mean_raw']))
        u = per[k][UNTRAINED]
        pr.append(('%s untrained at_optimum' % k,
                   sum(1 for c, oo in zip(u['components'], o) if c <= oo + 1e-9),
                   rf['S12'][UNTRAINED]['summary'][k]['at_optimum']))
    chk('x5_min_max_floor_optimum', pr, extra=2 * (9 * 4 + 2 + 1))

    # (4) the untrained actor against S12 of the full report
    pr = []
    s12 = rf['S12'][UNTRAINED]
    for k in 'AB':
        u = per[k][UNTRAINED]
        sm = s12['summary'][k]
        pr += [('%s untrained ach_mean' % k, st.mean(u['ach']), sm['ach_mean']),
               ('%s untrained comps_mean' % k, st.mean(u['components']), sm['comps_mean']),
               ('%s untrained jpu_pooled' % k, sum(u['energy_sum']) / sum(u['delivered_sum']), sm['jpu_pooled']),
               ('%s untrained M7 mean' % k, st.mean(u['m7']), sm['M7']['mean']),
               ('%s untrained n' % k, len(u['ach']), sm['n'])]
        pp = s12['paired'][k]
        for key, (x, y, f) in (('ach_trained_minus_untrained', ('ck1234', UNTRAINED, 'ach')),
                               ('ach_untrained_minus_random', (UNTRAINED, 'random', 'ach')),
                               ('ach_untrained_minus_best_routing', (UNTRAINED, 'best_routing', 'ach')),
                               ('components_untrained_minus_trained', (UNTRAINED, 'ck1234', 'components')),
                               ('jpu_untrained_minus_trained', (UNTRAINED, 'ck1234', 'jpu'))):
            if key not in pp:
                continue
            d = diffs(x, y, k, f)
            ref = pp[key]
            r = CA.paired(d, ref['expect'])
            for g in ('mean', 'ci_lo', 'ci_hi', 'wins', 'n'):
                pr.append(('%s %s %s' % (k, key, g), r[g], ref[g]))
    nexp = sum(5 + 5 * sum(1 for key in ('ach_trained_minus_untrained', 'ach_untrained_minus_random',
                                         'ach_untrained_minus_best_routing',
                                         'components_untrained_minus_trained',
                                         'jpu_untrained_minus_trained') if key in s12['paired'][k])
               for k in 'AB')
    chk('x6_untrained_vs_S12', pr, extra=nexp)

    out = {
        'what': 'per-instance values of every arm on R3-lite topology seeds 300-339 x Scenarios A '
                '(recovery) and B (rescue_ops), 3600 ticks; every list is aligned with `seeds`',
        'seeds': SEEDS, 'fields': list(FIELDS),
        'arms': list(arms),
        'metrics': {'ach': 'CA.steady(conn), pp (last %d ticks)' % CA.W,
                    'm7': 'CA.m7_teardowns_per_1000(run, CA.T_SEV[k])',
                    'energy': 'CA.steady(energy) per-tick energy, last %d ticks' % CA.W,
                    'jpu': 'CA.jpu(run) = sum(energy)/sum(delivered) per instance',
                    'components': 'CA.steady(fragments)',
                    'energy_sum, delivered_sum': 'per-instance whole-run sums (pooled J/unit = '
                                                 'sum(energy_sum)/sum(delivered_sum)); not for '
                                                 'tstar_tu1 or best_routing',
                    'best_routing': 'per instance: max ach, min jpu/energy/components/m7 over the '
                                    'four routing protocols',
                    'untrained_s1234': "replicate 1234's untrained (initial) actor, R3-lite S8 pickle, "
                                       'the S12 control',
                    'optima': 'attainable optimum and raw (relay-free) component count per instance, '
                              'R3-lite manifest'},
        'per_seed': per, 'optima': optima,
        'wins': wins_tab,
        'cross_checks_main': xc_main,
        'cross_checks': xc,
        'provenance': {'main': prov_main, 'untrained': urec,
                       'reports_md5': {p: _md5(p) for p in (REP1, REPE, REPF, CB_JSON, MANIFEST)},
                       'script': os.path.abspath(__file__), 'script_md5': _md5(os.path.abspath(__file__))},
    }
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(SUM._jsonable(out), f, indent=None, sort_keys=True, default=str)
        f.write('\n')
    # (5) round trip: every exported value read back exactly
    with open(path, encoding='utf-8') as f:
        back = json.load(f)
    n, bad = 0, []
    for k in 'AB':
        for name in arms:
            for f_, vals in per[k][name].items():
                got = back['per_seed'][k][name][f_]
                if len(got) != len(SEEDS):
                    bad.append('%s %s %s: %d values' % (k, name, f_, len(got)))
                for x, y in zip(vals, got):
                    n += 1
                    if x != y:
                        bad.append('%s %s %s: %r != %r' % (k, name, f_, x, y))
    rt = {'values_compared': n, 'mismatches': bad[:20], 'pass': bool(n > 0 and not bad)}
    allok = rt['pass'] and all(v['pass'] for v in xc.values()) and all(v['pass'] for v in xc_main.values())
    for name, v in sorted(xc.items()):
        print('PER-SEED CROSS-CHECK %-32s %s: %d values (expected %s), max |diff| %.1e%s' % (
            name, 'PASS' if v['pass'] else 'FAIL', v['values_compared'], v.get('expected_count'),
            v['max_abs_diff'], ('' if not v['mismatches'] else ' ' + '; '.join(v['mismatches'][:5]))))
    print('PER-SEED ROUND TRIP %s: %d values' % ('PASS' if rt['pass'] else 'FAIL', n))
    for name, v in sorted(xc_main.items()):
        print('MAIN CROSS-CHECK %-18s %s: %d values, max |diff| %.1e' % (
            name, 'PASS' if v['pass'] else 'FAIL', v['values_compared'], v['max_abs_diff']))
    if not allok:
        os.replace(path, path + '.FAILED')
        print('PER-SEED EXPORT FAILED -> %s.FAILED' % path)
        return 5
    print('PER-SEED EXPORT OK -> %s (%d arms x 2 scenarios x 40 seeds)' % (path, len(arms)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
