#!/usr/bin/env python3
"""Run the FINISHED Track B analysis helpers on an engine pickle (WP4-VERIFY
D4.16): the contract check between what this tree emits and what
confirmatory_analysis.py on branch trackB reads.

    python3 tools/wp4_trackb_contract.py --analysis ~/MARL_run_9b/confirmatory_analysis.py \
        --manifest benchmark_manifest_80_99.json output/inv_<wp>_<head>.pkl

The analysis module is imported from the given path and used AS IS -- this
tool re-implements none of its logic.  For every (scenario, arm, seed) run of
the pickle it calls `stability_metrics` (M1-M11), and it calls
`check_invariants` (the WP9 hard asserts) over all of them.

The invariance pickles are 600- or 1200-tick runs whose Scenario A event is
rescaled (output/ref/README.md), so the severance tick and run length are
taken from the pickle's provenance and the manifest instances' severance
tick is set to it; the attainable optimum `o` is the manifest's.

VERDICT.  PASS requires all of:
  * every M1-M11 value is computed (not None) for every run of every arm --
    a None here means the helper found a series missing ("not recorded");
  * `check_invariants` reports NO failure -- except, for a pickle without
    the pre-registered MARL arm (the 1200-tick routing/SDN reference), the
    assert that this arm is analysed.  Any other failure (run length, NaN,
    stale edges, lifetime records, per-site coverage, provenance, and since
    WP5 the per-site and aggregate POSTCARD asserts, which failed by design
    at WP4 (D4.8) and are now the WP5 acceptance check) fails the verdict;
  * the checks actually ran: run_length.n_series_checked > 0, the lifetime
    check saw the key in every run, and -- when the pickle holds MARL-path
    runs -- the per-site postcard mode is 'per-site' with the site coverage
    compared in every one of them (a pickle with none, such as the 1200-tick
    routing/SDN reference, must show no postcard run checked at all).
"""
import argparse
import importlib.util
import json
import pickle
import re
import sys

SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
KIND = {'A': 'recovery', 'B': 'rescue_ops'}
METRICS = ('M1', 'M2', 'M3', 'M4', 'M4ss', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11')

# check_invariants failures that are expected.  Until WP5 these were the
# per-site and aggregate postcard asserts (the one-postcard-per-episode
# defect, D4.8); WP5 fixed it, so NO failure is expected any more and a
# postcard assert failing is a verdict FAIL.
EXPECTED_FAILURES = ()


def _load_analysis(path):
    spec = importlib.util.spec_from_file_location('confirmatory_analysis_trackB', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fmt(v):
    if isinstance(v, float):
        return '%.4g' % v
    if isinstance(v, dict):
        keep = {k: v[k] for k in ('value', 'censored', 'bridge', 'idle', 'median',
                                   'n', 'n_censored', 'n_invalid') if k in v}
        return json.dumps(keep, sort_keys=True)
    return repr(v)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--manifest', action='append', required=True)
    ap.add_argument('pickle')
    a = ap.parse_args(argv)
    CA = _load_analysis(a.analysis)
    P = pickle.load(open(a.pickle, 'rb'))
    prov = P.get('provenance') or {}
    T = int(prov.get('total_ticks') or 0)
    ts_a = prov.get('severance_tick')
    if not T or ts_a is None:
        print('VERDICT FAIL: the pickle records no total_ticks/severance_tick '
              'in its provenance')
        return 1
    t_sev = {'A': int(ts_a), 'B': 0}
    arms, runs = [], {k: {} for k in SCEN}
    for k, sk in SCEN.items():
        for arm, per_seed in (P.get(sk) or {}).items():
            if arm not in arms:
                arms.append(arm)
            runs[k][arm] = {int(s): r for s, r in per_seed.items() if r}
    seeds = sorted({s for k in SCEN for arm in runs[k] for s in runs[k][arm]})
    opt = {}
    for m in a.manifest:
        for inst in json.load(open(m))['instances']:
            s = int(inst['topology_seed'])
            if s in seeds:
                inst = dict(inst)
                inst['severance_tick'] = t_sev['A'] if inst['scenario_kind'] == 'recovery' else 0
                opt[(inst['scenario_kind'], s)] = inst
    print('# analysis %s' % a.analysis)
    print('# pickle %s: T=%d severance_tick(A)=%d arms=%s seeds=%s tree_head=%s'
          % (a.pickle, T, t_sev['A'], arms, seeds, prov.get('tree_head')))
    missing_opt = [(KIND[k], s) for k in SCEN for s in seeds if (KIND[k], s) not in opt]
    if missing_opt:
        print('VERDICT FAIL: no manifest instance for %s' % missing_opt)
        return 1

    ok = True
    n_runs = 0
    none_metrics = []
    for k in SCEN:
        for arm in arms:
            for s in sorted(runs[k].get(arm, {})):
                r = runs[k][arm][s]
                o = opt[(KIND[k], s)]['attainable_optimum']
                m = CA.stability_metrics(r, t_sev[k], o, T)
                n_runs += 1
                nones = [x for x in METRICS if m.get(x) is None]
                if nones:
                    none_metrics.append((k, arm, s, nones))
                print('%s %-12s %3d  %s' % (k, arm, s, '  '.join(
                    '%s=%s' % (x, _fmt(m.get(x))) for x in METRICS)))
    if none_metrics:
        ok = False
        for e in none_metrics:
            print('NOT RECORDED  %s %s seed %s: %s' % e)

    inv = CA.check_invariants(runs, opt, [(a.pickle, prov, arms)], arms,
                              t_sev=t_sev, t_total=T, marl_arm='marl_static')
    allowed = list(EXPECTED_FAILURES)
    if CA.PREREG_MARL_ARM not in arms:
        # a routing/SDN-only pickle (the 1200-tick reference) cannot carry
        # the tested arm; that one assert is about the analysis's arm set,
        # not about anything the engine records
        allowed.append(re.compile(r"^the pre-registered arm '%s' is not among the "
                                  r"analysed arms " % re.escape(CA.PREREG_MARL_ARM)))
        print('NOTE  this pickle carries no %s run: the pre-registered-arm assert '
              'does not apply to it' % CA.PREREG_MARL_ARM)
    expected, unexpected = [], []
    for f in inv['failures']:
        (expected if any(p.match(f) for p in allowed) else unexpected).append(f)
    c = inv['checks']
    print('check_invariants: %d failure(s): %d expected (not applicable to this pickle), %d unexpected'
          % (len(inv['failures']), len(expected), len(unexpected)))
    for f in unexpected:
        print('UNEXPECTED  %s' % f)
    for f in expected[:6]:
        print('expected    %s' % f)
    if len(expected) > 6:
        print('expected    ... %d more' % (len(expected) - 6))
    pc = c['postcards']
    print('run_length: %d series over %d runs, %d violations; lifetimes: key in %d runs, '
          '%d invalid; postcards: mode=%s runs_checked=%d with_per_site=%d '
          'with_site_coverage=%d incomplete=%d min_per_site=%s; stale: key in %d runs'
          % (c['run_length']['n_series_checked'], c['run_length']['runs_checked'],
             len(c['run_length']['violations']), c['lifetimes']['runs_with_key'],
             len(c['lifetimes']['violations']), pc['mode'], pc['runs_checked'],
             pc['runs_with_per_site'], pc['runs_with_site_coverage'],
             len(pc['incomplete_per_site']), pc['min_per_site'],
             c['stale_edges']['runs_with_key']))
    # The postcard asserts apply to MARL-path runs only; a pickle without any
    # (the 1200-tick routing/SDN reference) must show that none was checked,
    # and one with them must show every one checked per site with coverage.
    n_marl = sum(1 for k in SCEN for arm in runs[k] if arm in CA.MARL_PATH_ARMS
                 for _ in runs[k][arm])
    if n_marl:
        postcards_ran = (pc['mode'] == 'per-site'
                         and pc['runs_checked'] == n_marl
                         and pc['runs_with_site_coverage'] == n_marl)
    else:
        postcards_ran = pc['runs_checked'] == 0
    print('MARL-path runs: %d (postcard asserts %s)'
          % (n_marl, 'apply' if n_marl else 'do not apply'))
    ran = (c['run_length']['n_series_checked'] > 0
           and c['lifetimes']['runs_with_key'] == n_runs
           and postcards_ran)
    if not ran:
        print('NOT EXERCISED: a check saw nothing to check (see the counts above)')
    ok = ok and not unexpected and ran and n_runs > 0
    print('VERDICT %s (%d runs)' % ('PASS' if ok else 'FAIL', n_runs))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
