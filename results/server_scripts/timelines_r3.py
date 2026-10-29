#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time course of every arm on R3-lite topology seeds 300-339 (Fig. 7 of the
direct paper): per-instance binned series, summarised across the 40
instances per bin (mean, median, 25th and 75th percentile).  Read-only,
descriptive; no pass is run.

An equivalent reader to extract_timelines.py (same binning function, same
linear-interpolation quantile, same numeric conversion from
confirmatory_analysis.py), with three differences that the cross-checks need:
the summary values are NOT rounded (extract_timelines.py rounds to 4
decimals, which would break a 1e-9 comparison), the two event series are
binned per 100 ticks, and the arm names are those of all_arms_300_339.json.

DATA (read-only):
  ~/MARL_r3lite/output/r3lite/s6_cf_b0..9.pkl   S6: marl_static = replicate 1234,
        xldet, sdn, random, ospf, olsr, batman, aodv
  ~/MARL_r3lite/output/r3lite/s9x_ensemble.pkl  S9x: marl_static = the ensemble
Validity and provenance: exactly as all_arms_300_339.py (r4_checkb_pair
marl_validity on every run; learned_provenance with replicates [1234] for S6
and the four pinned replicates for S9x).

SERIES
  conn              per-tick user-to-user achievability (pp), 10-tick bin means
  fragments         per-tick connected components, 10-tick bin means
  relay_removed     relay-link teardowns per tick, 100-tick bin sums
  bridge_teardowns  bridge teardowns per tick, 100-tick bin sums
Bin b of width w covers ticks [b*w, (b+1)*w), t = 0..3599.

CROSS-CHECKS (exact, tolerance 1e-9; each counts what it compared and fails
when it compared nothing or fewer than expected):
  (i)   every arm and scenario has exactly the 40 seeds 300-339, once each;
        every series has 3600 entries;
  (ii)  the mean of the last 50 bins of the `conn` mean curve equals
        all_arms ach.mean, and of `fragments` equals components.mean;
  (iii) the 100-tick bin sums of relay_removed add up to the whole-run sum of
        every instance, and minus the teardowns at ticks 0..T_SEV they give
        M7 = 1000 * sum(relay_removed[T_SEV+1:]) / (T - T_SEV - 1) equal to
        CA.m7_teardowns_per_1000 per instance and to all_arms m7.mean per arm;
        the bridge_teardowns bin sums add up to the whole-run sums.
Exit 0 = all pass; 3 = a cross-check failed (the output is renamed *.FAILED).
"""
import argparse
import glob
import hashlib
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

import confirmatory_analysis as CA              # noqa: E402
import r4_checkb_pair as CBP                    # noqa: E402
import chain_plan as CP                         # noqa: E402

R3 = os.path.expanduser('~/MARL_r3lite')
S6_GLOB = os.path.join(R3, 'output/r3lite/s6_cf_b*.pkl')
S9X = os.path.join(R3, 'output/r3lite/s9x_ensemble.pkl')
ALL_ARMS = '<server-data>/r4/baselines_300_339/all_arms_300_339.json'
SEEDS = list(range(300, 340))
S6_ARMS = {'marl_static': 'ck1234', 'xldet': 'xldet', 'sdn': 'sdn', 'random': 'random',
           'ospf': 'ospf', 'olsr': 'olsr', 'batman': 'batman', 'aodv': 'aodv'}
ARM_ORDER = ('ck1234', 'ensemble', 'xldet', 'sdn', 'ospf', 'olsr', 'batman', 'aodv', 'random')
ROUTING = ('ospf', 'olsr', 'batman', 'aodv')
SERIES = {'conn': ('mean', 10), 'fragments': ('mean', 10),
          'relay_removed': ('sum', 100), 'bridge_teardowns': ('sum', 100)}
LAST = CA.W // SERIES['conn'][1]          # 50 bins = the steady window of 500 ticks
TOL = 1e-9


def binned(series, width, how):
    n = len(series) // width
    out = []
    for b in range(n):
        v = [CA.py_num(x) for x in series[b * width:(b + 1) * width] if CA.is_num(x)]
        out.append(None if not v else (sum(v) / len(v) if how == 'mean' else sum(v)))
    return out


def q(vals, p):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load(paths, armmap, expect_reps, store, inst, bad, provs):
    for p in paths:
        with open(p, 'rb') as f:
            P = pickle.load(f)
        prov = P.get('provenance') or {}
        rec, pbad = CBP.learned_provenance(prov, expect_reps)
        provs[p] = rec
        bad += ['%s: %s' % (os.path.basename(p), b) for b in pbad]
        for k, sk in CA.SCEN.items():
            for arm, name in armmap.items():
                for s, run in ((P.get(sk) or {}).get(arm) or {}).items():
                    s = int(s)
                    if s not in SEEDS or not run:
                        continue
                    key = (k, name, s)
                    if key in store:
                        bad.append('%s %s seed %d twice' % (name, k, s))
                        continue
                    why = CBP.marl_validity(run, arm, k, s)
                    if why:
                        bad.append('%s %s %s s%d invalid: %s' % (os.path.basename(p), name, k, s, why))
                        continue
                    row = {}
                    for ser, (how, w) in SERIES.items():
                        x = run.get(ser)
                        if x is None or len(x) != CA.T_TOTAL:
                            bad.append('%s %s s%d: %s has %s entries' % (name, k, s, ser,
                                                                        None if x is None else len(x)))
                            continue
                        row[ser] = binned(x, w, how)
                    rr = run['relay_removed']
                    ts = CA.T_SEV[k]
                    inst[key] = {
                        'steady_conn': CA.steady(run['conn']),
                        'steady_fragments': CA.steady(run['fragments']),
                        'rr_total': sum(CA.num(rr)),
                        'rr_pre': sum(CA.num(rr[:ts + 1])),
                        'bt_total': sum(CA.num(run['bridge_teardowns'])),
                        'm7_ca': CA.m7_teardowns_per_1000(run, ts),
                        'T': len(rr), 'pickle': os.path.basename(p)}
                    store[key] = row
        del P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--all-arms', default=ALL_ARMS)
    a = ap.parse_args()
    s6 = sorted(glob.glob(S6_GLOB))
    if len(s6) != 10:
        raise SystemExit('expected 10 S6 pickles, found %d' % len(s6))
    store, inst, bad, provs = {}, {}, [], {}
    load(s6, S6_ARMS, [1234], store, inst, bad, provs)
    load([S9X], {'marl_static': 'ensemble'}, list(CP.R2_CHECKPOINT_SHA256), store, inst, bad, provs)

    xc = {}
    # (i) populations
    pops, n_pop = [], 0
    for k in 'AB':
        for name in ARM_ORDER:
            got = sorted(s for (kk, nn, s) in store if kk == k and nn == name)
            n_pop += 1
            if got != SEEDS:
                pops.append('%s %s: %d seeds, missing %s, extra %s' % (
                    k, name, len(got), sorted(set(SEEDS) - set(got)), sorted(set(got) - set(SEEDS))))
    xc['i_populations'] = {'populations': n_pop, 'expected_populations': 2 * len(ARM_ORDER),
                           'instances': len(store), 'expected_instances': 2 * len(ARM_ORDER) * 40,
                           'problems': pops,
                           'pass': bool(not pops and n_pop == 2 * len(ARM_ORDER)
                                        and len(store) == 2 * len(ARM_ORDER) * 40)}
    if bad or pops:
        print('VIOLATIONS:\n  ' + '\n  '.join(bad + pops))
        raise SystemExit(2)

    # summaries
    out_sc = {}
    for k in 'AB':
        arms = {}
        for name in ARM_ORDER:
            d = {}
            for ser, (how, w) in SERIES.items():
                per = [store[(k, name, s)][ser] for s in SEEDS]
                nb = CA.T_TOTAL // w
                stats = {'mean': [], 'median': [], 'q25': [], 'q75': [], 'n': []}
                for b in range(nb):
                    col = [row[b] for row in per if row[b] is not None]
                    stats['n'].append(len(col))
                    stats['mean'].append(st.mean(col) if col else None)
                    stats['median'].append(q(col, 0.5))
                    stats['q25'].append(q(col, 0.25))
                    stats['q75'].append(q(col, 0.75))
                d[ser] = stats
            arms[name] = d
        out_sc[k] = {'seeds': SEEDS, 'arms': arms}

    # (ii) last window against all_arms
    with open(a.all_arms, encoding='utf-8') as f:
        aa = json.load(f)
    n, worst, mis = 0, 0.0, []
    for k in 'AB':
        for name in ARM_ORDER:
            for ser, field in (('conn', 'ach'), ('fragments', 'components')):
                m = out_sc[k]['arms'][name][ser]['mean'][-LAST:]
                if len(m) != LAST or any(x is None for x in m):
                    mis.append('%s %s %s: incomplete last window' % (k, name, ser))
                    continue
                mine = sum(m) / LAST
                theirs = aa['descriptive'][k][name][field]['mean']
                dif = abs(mine - theirs)
                worst = max(worst, dif)
                n += 1
                if dif >= TOL:
                    mis.append('%s %s %s: %r vs %r' % (k, name, ser, mine, theirs))
    xc['ii_last_window_vs_all_arms'] = {'values_compared': n, 'expected': 2 * len(ARM_ORDER) * 2,
                                        'last_bins': LAST, 'max_abs_diff': worst, 'mismatches': mis,
                                        'pass': bool(n == 2 * len(ARM_ORDER) * 2 and not mis)}

    # (iii) teardowns: bin sums -> whole-run sums -> M7
    n_tot, n_inst, n_arm, worst_i, worst_a, mis = 0, 0, 0, 0.0, 0.0, []
    m7_means = {}
    for k in 'AB':
        ts = CA.T_SEV[k]
        for name in ARM_ORDER:
            vals = []
            for s in SEEDS:
                row, ii = store[(k, name, s)], inst[(k, name, s)]
                rsum = sum(x for x in row['relay_removed'] if x is not None)
                bsum = sum(x for x in row['bridge_teardowns'] if x is not None)
                n_tot += 2
                if rsum != ii['rr_total'] or bsum != ii['bt_total']:
                    mis.append('%s %s s%d: bin sums %r/%r vs run sums %r/%r' % (
                        k, name, s, rsum, bsum, ii['rr_total'], ii['bt_total']))
                m7 = 1000.0 * (rsum - ii['rr_pre']) / (ii['T'] - ts - 1)
                vals.append(m7)
                d = abs(m7 - ii['m7_ca'])
                worst_i = max(worst_i, d)
                n_inst += 1
                if d >= TOL:
                    mis.append('%s %s s%d: M7 from bins %r vs CA %r' % (k, name, s, m7, ii['m7_ca']))
            mean7 = st.mean(vals)
            m7_means['%s %s' % (k, name)] = mean7
            theirs = aa['descriptive'][k][name]['m7']['mean']
            d = abs(mean7 - theirs)
            worst_a = max(worst_a, d)
            n_arm += 1
            if d >= TOL:
                mis.append('%s %s: mean M7 %r vs all_arms %r' % (k, name, mean7, theirs))
    ne_i = 2 * len(ARM_ORDER) * 40
    xc['iii_teardowns_vs_m7'] = {'bin_sum_checks': n_tot, 'instance_m7_checks': n_inst,
                                 'arm_m7_mean_checks': n_arm,
                                 'expected': [2 * ne_i, ne_i, 2 * len(ARM_ORDER)],
                                 'max_abs_diff_instance': worst_i, 'max_abs_diff_arm_mean': worst_a,
                                 'm7_means': m7_means, 'mismatches': mis[:30],
                                 'pass': bool(not mis and n_tot == 2 * ne_i and n_inst == ne_i
                                              and n_arm == 2 * len(ARM_ORDER))}

    # descriptive: do the four routing protocols coincide?
    coin = {}
    for k in 'AB':
        for ser in ('conn', 'fragments'):
            ref = out_sc[k]['arms']['ospf'][ser]['mean']
            coin['%s %s' % (k, ser)] = max(abs(x - y) for r in ROUTING[1:]
                                           for x, y in zip(out_sc[k]['arms'][r][ser]['mean'], ref))

    rep = {
        'what': 'time course on R3-lite topology seeds 300-339 x Scenarios A (recovery) and B '
                '(rescue_ops), 3600 ticks: per-instance binned series summarised across the 40 '
                'instances per bin (mean, median, q25, q75 by linear interpolation, n)',
        'seeds': SEEDS, 'arms': list(ARM_ORDER),
        'arm_names': {'ck1234': 'replicate 1234 (marl_static, S6)',
                      'ensemble': 'four-replicate ensemble (marl_static, S9x)'},
        'series': {s: {'binning': how, 'bin_ticks': w, 'n_bins': CA.T_TOTAL // w}
                   for s, (how, w) in SERIES.items()},
        'bin_definition': 'bin b of width w covers ticks [b*w, (b+1)*w), t = 0..%d' % (CA.T_TOTAL - 1),
        'units': {'conn': 'pp (per-tick min(100, 100 delivered/offered), user-to-user)',
                  'fragments': 'connected components', 'relay_removed': 'relay-link teardowns per bin',
                  'bridge_teardowns': 'bridge teardowns per bin'},
        'events': {'A': {'severance': CA.T_EVENT['A']}, 'B': {'severance': CA.T_SEV['B'],
                                                            'surge': CA.T_EVENT['B']}},
        'scenarios': out_sc,
        'routing_max_abs_diff_of_mean_curves_vs_ospf': coin,
        'cross_checks': xc,
        'provenance': {'pickles': provs, 'all_arms_json': a.all_arms, 'all_arms_md5': md5(a.all_arms),
                       'analysis_code': {'confirmatory_analysis_md5': CP.R3_ANALYSIS_MD5,
                                         'confirmatory_analysis_file_md5': md5(CA.__file__),
                                         'tree_imported_from': TREE},
                       'script': os.path.abspath(__file__), 'script_md5': md5(os.path.abspath(__file__)),
                       'pythonhashseed': os.environ.get('PYTHONHASHSEED')},
    }
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(rep, f, sort_keys=True)
        f.write('\n')
    ok = all(v['pass'] for v in xc.values())
    for name, v in sorted(xc.items()):
        print('TIMELINE CROSS-CHECK %-28s %s %s' % (name, 'PASS' if v['pass'] else 'FAIL',
                                                   json.dumps({kk: vv for kk, vv in v.items()
                                                               if kk not in ('m7_means', 'mismatches')})))
        if v.get('mismatches'):
            print('   mismatches: %s' % v['mismatches'][:10])
    print('routing coincidence (max |mean - ospf mean| over bins): %s' % coin)
    if not ok:
        os.replace(a.out, a.out + '.FAILED')
        print('TIMELINES FAILED -> %s.FAILED' % a.out)
        return 3
    print('TIMELINES OK -> %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
