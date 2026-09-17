# -*- coding: utf-8 -*-
"""Per-instance steady-state values (final 500 ticks) for every arm, both
scenarios, development and confirmatory sets, from the Amendment-1 KPI dumps.
Columns: achievability %, energy per delivered unit (whole run, J/unit),
components, relay bridges, energy J/tick, latency ms, and (Scenario A) the
service-loss integral pp.s.  Descriptive listing for figure coordinates.

Usage: python paper/per_instance_A1.py > paper/per_instance_A1.txt
"""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
A1 = os.path.join(os.path.dirname(HERE), 'auth_results', 'A1')
ARMS = ['marl_static', 'marl_freeze', 'xldet', 'sdn', 'ospf', 'olsr', 'batman', 'aodv', 'random']
T0 = {'recovery_multi': 200, 'rescue_multi': 1400}


def f(x):
    return float('nan') if x is None else x


def sli(r, t0):
    import statistics as st
    conn = r['conn']['series']; off = r['offered']['series']
    pre = [c for c in conn[:t0] if c is not None]
    a_ref = st.mean(pre) * st.mean([o for o in off[:t0] if o is not None]) / st.mean([o for o in off[t0:] if o is not None])
    return sum(max(0.0, a_ref - (c if c is not None else 0.0)) for c in conn[t0:])


for label, fn, tag in (('DEVELOPMENT', 'ce_results_A1_devstress.json', 'dev'), ('CONFIRMATORY', 'ce_results_A1_cf.json', 'cf')):
    J = json.load(open(os.path.join(A1, fn)))[tag]['scenarios']
    for scen in ('recovery_multi', 'rescue_multi'):
        print('\n==== %s  %s ====' % (label, 'Scenario A' if scen == 'recovery_multi' else 'Scenario B'))
        for a in ARMS:
            R = J[scen][a]
            print('-- %s' % a)
            print('   seed   ach    J/unit  comps  bridges  E/tick  lat_ms' + ('   SLI' if scen == 'recovery_multi' else ''))
            for s in sorted(R, key=int):
                r = R[s]
                row = '   %-5s %5.1f  %.4f  %5.2f  %5.2f   %6.1f  %5.2f' % (
                    s, f(r['conn'].get('steady')), f(r['energy']['total']) / (r['delivered'].get('total') or float('nan')),
                    f(r['fragments'].get('steady')), f(r['relay_bridges'].get('steady')), f(r['energy'].get('steady')),
                    f(r['latency'].get('steady')))
                if scen == 'recovery_multi':
                    row += '  %7.0f' % sli(r, T0[scen])
                print(row)
