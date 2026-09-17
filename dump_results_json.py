# -*- coding: utf-8 -*-
"""Per-run KPI dump for the manuscript generator (paper/make_results.py).

Merges result pickles arm by arm into named tags and writes, for every
(tag, scenario, arm, seed), the steady-state / total / final value of every
series, the full per-tick series of the quantities the generator needs, and
the scalar fields.  Later pickles never overwrite an arm already filled by an
earlier pickle in the same tag, so a tag's baselines always come from the
first pickle that holds them.

Usage (from ~/MARL_run_6):
  python3 dump_results_json.py --out ce_results_A1.json \
     --tag dev=PKL,PKL,... --tag cf=PKL,... --tag stress=PKL,...
"""
import argparse
import json
import math
import pickle

W = 500
SERIES_KEYS = ('conn', 'offered', 'fragments', 'peer_reach', 'delivered', 'energy', 'latency')


def num(v):
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def summarize(run):
    d = {}
    for k, v in run.items():
        if isinstance(v, list):
            vals = [x for x in v if num(x)]
            if not vals:
                d[k] = {'empty': len(v)}
                continue
            tail = [x for x in v[-W:] if num(x)]
            d[k] = {'steady': (sum(tail) / len(tail) if tail else None), 'total': sum(vals),
                    'final': vals[-1], 'n': len(v)}
            if k in SERIES_KEYS:
                d[k]['series'] = [(round(x, 4) if num(x) else None) for x in v]
        elif num(v):
            d[k] = {'scalar': v}
        elif isinstance(v, str):
            d[k] = {'text': v}
        elif isinstance(v, dict):
            d[k] = {'dict': {str(a): (b if isinstance(b, (int, float, str)) else str(b)[:60]) for a, b in v.items()}}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', action='append', required=True, help='name=pkl,pkl,...')
    a = ap.parse_args()
    out = {}
    for spec in a.tag:
        name, _, paths = spec.partition('=')
        tag = {'sources': {}, 'scenarios': {'recovery_multi': {}, 'rescue_multi': {}}}
        for p in [x for x in paths.split(',') if x]:
            P = pickle.load(open(p, 'rb'))
            got = []
            for sk in ('recovery_multi', 'rescue_multi'):
                for arm, runs in P.get(sk, {}).items():
                    dest = tag['scenarios'][sk].setdefault(arm, {})
                    added = 0
                    for s, r in runs.items():
                        if r and str(s) not in dest:     # first source wins per (arm, seed)
                            dest[str(s)] = summarize(r)
                            added += 1
                    if added:
                        got.append('%s/%s:%d' % (sk.split('_')[0], arm, added))
            tag['sources'][p] = got
        out[name] = tag
        print('tag %-8s arms/seeds: %s' % (name, {sk: {a: len(v) for a, v in sorted(arms.items())}
                                                   for sk, arms in tag['scenarios'].items()}))
    json.dump(out, open(a.out, 'w'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
