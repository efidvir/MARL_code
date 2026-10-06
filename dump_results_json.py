# -*- coding: utf-8 -*-
"""Per-run KPI dump for the manuscript generator (paper/make_results.py).

Merges result pickles arm by arm into named tags and writes, for every
(tag, scenario, arm, seed), the steady-state / total / final value of every
series, the full per-tick series of the quantities the generator needs, the
scalar fields, and (WP9) a 'stability' block with the metrics M1-M11 computed
by the helpers of confirmatory_analysis.py (a metric whose series the run
does not carry is null, never a lower bound).  Later pickles never overwrite
an arm already filled by an earlier pickle in the same tag, so a tag's
baselines always come from the first pickle that holds them; each tag lists
which pickle supplied which arm and every pickle's provenance record.

New WP4 recording, passed through when present: the event series
(relay_created, relay_removed, bridge_teardowns, idle_bridge_teardowns,
relay_mode_flips), stale_edges / stale_u2u_volume, energy_ctrl, the
{name: per-tick} members (overhead_by_type, relay_removed_by_reason,
relay_teardown_holds) as 'by_key', and relay_lifetimes as 'records'.
Numeric entries follow confirmatory_analysis (numpy scalars and arrays are
accepted and converted, so the JSON is always serialisable).

Usage (from the analysis tree):
  python3 dump_results_json.py --out ce_results.json [--manifests M.json,...]
     --tag dev=PKL,PKL,... --tag cf=PKL,... --tag stress=PKL,...
"""
import argparse
import json
import pickle

import confirmatory_analysis as CA

W = 500
SERIES_KEYS = (('conn', 'offered', 'fragments', 'peer_reach', 'delivered', 'energy', 'latency',
                'fragments_raw', 'relay_count', 'relay_bridges', 'relay_mode_nodes', 'postcards',
                'overhead', CA.ENERGY_CTRL_KEY) + CA.EVENT_SERIES + CA.DIAG_SERIES)
SCEN_OF = {'recovery_multi': 'A', 'rescue_multi': 'B'}


def num(v):
    """A usable numeric entry (Python or numpy number, not NaN/None)."""
    return CA.is_num(v)


def _series_summary(k, v, with_series=None):
    vals = CA.num(v)
    if not vals:
        return {'empty': len(v)}
    tail = CA.num(v[-W:])
    d = {'steady': (sum(tail) / len(tail) if tail else None), 'total': sum(vals),
         'final': vals[-1], 'n': len(v)}
    if with_series if with_series is not None else (k in SERIES_KEYS):
        d['series'] = [(round(CA.py_num(x), 4) if num(x) else None) for x in v]
    return d


def summarize(run):
    d = {}
    for k, v in run.items():
        if hasattr(v, 'tolist'):
            v = v.tolist()          # numpy array -> list, numpy scalar -> Python scalar
        if k == CA.LIFETIMES_KEY and isinstance(v, (list, tuple)):
            d[k] = {'records': [CA.lifetime_record(x) for x in v], 'n': len(v)}
        elif isinstance(v, list):
            if v and all(isinstance(x, (list, tuple, dict)) for x in v):
                d[k] = {'records': [list(x) if isinstance(x, tuple) else x for x in v], 'n': len(v)}
            else:
                d[k] = _series_summary(k, v)
        elif num(v):
            d[k] = {'scalar': CA.py_num(v)}
        elif isinstance(v, str):
            d[k] = {'text': v}
        elif isinstance(v, dict):
            # {name: per-tick series} only when the values really are series.
            # `hasattr(x, 'tolist')` alone is true for every numpy SCALAR too
            # (np.int64 included), so an all-numpy {site: count} dict -- the
            # requested postcards_by_site, or a topology_info of numpy scalars
            # -- took the by_key branch and aborted the whole dump at
            # list(np.int64(4).tolist()) (WP9 defect list, D4).  A numpy scalar
            # is a real number (CA.is_real) and belongs in the 'dict' branch,
            # which converts it with py_num.
            if v and all(isinstance(x, (list, tuple)) or (hasattr(x, 'tolist') and not CA.is_real(x))
                         for x in v.values()):
                d[k] = {'by_key': {str(a): _series_summary(k, list(b.tolist() if hasattr(b, 'tolist') else b), True)
                                   for a, b in v.items()}}
            else:
                d[k] = {'dict': {str(a): (CA.py_num(b) if CA.is_real(b) else b if isinstance(b, str) else str(b)[:60])
                                 for a, b in v.items()}}
    return d


def stability_block(run, sk, inst):
    """M1-M11 of one run (scenario key sk, manifest instance inst or None)."""
    k = SCEN_OF[sk]
    o = inst['attainable_optimum'] if inst else None
    f = run.get('fragments')
    T = len(f) if f is not None else CA.T_TOTAL
    blk = CA.stability_metrics(run, CA.T_SEV[k], o, T)
    blk['severance_tick'] = CA.T_SEV[k]
    blk['attainable_optimum'] = o
    return blk


def load_manifests(paths):
    opt = {}
    for m in paths:
        for inst in json.load(open(m))['instances']:
            opt[(inst['scenario_kind'], int(inst['topology_seed']))] = inst
    return opt


def build(tags, opt):
    """tags = [(name, [pickle paths])] -> the output dict."""
    out = {}
    for name, paths in tags:
        tag = {'sources': {}, 'provenance': {}, 'scenarios': {'recovery_multi': {}, 'rescue_multi': {}}}
        for p in paths:
            P = pickle.load(open(p, 'rb'))
            prov = P.get('provenance')
            tag['provenance'][p] = ({f: prov.get(f) for f in CA.PROVENANCE_FIELDS + ('total_ticks', 'severance_tick')}
                                    if isinstance(prov, dict) else None)
            got = []
            for sk in ('recovery_multi', 'rescue_multi'):
                for arm, runs in P.get(sk, {}).items():
                    dest = tag['scenarios'][sk].setdefault(arm, {})
                    added = 0
                    for s, r in runs.items():
                        if r and str(s) not in dest:     # first source wins per (arm, seed)
                            d = summarize(r)
                            d['stability'] = stability_block(r, sk, opt.get((CA.KIND[SCEN_OF[sk]], int(s))))
                            dest[str(s)] = d
                            added += 1
                    if added:
                        got.append('%s/%s:%d' % (sk.split('_')[0], arm, added))
            tag['sources'][p] = got
        out[name] = tag
        print('tag %-8s arms/seeds: %s' % (name, {sk: {a: len(v) for a, v in sorted(arms.items())}
                                                   for sk, arms in tag['scenarios'].items()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', action='append', required=True, help='name=pkl,pkl,...')
    ap.add_argument('--manifests', default='', help='benchmark manifests (attainable optimum for M3/M4/M6)')
    a = ap.parse_args()
    tags = []
    for spec in a.tag:
        name, _, paths = spec.partition('=')
        tags.append((name, [x for x in paths.split(',') if x]))
    opt = load_manifests([m for m in a.manifests.split(',') if m])
    out = build(tags, opt)
    json.dump(out, open(a.out, 'w'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
