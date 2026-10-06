#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine constancy-dump reports into the per-field evidence (WP13 item 3).

    python3 tools/constancy_combine.py output/wp13/constancy_<head>_*.json \\
        [--require-ticks 3600] [--require-passes 6] \\
        [--require-scenarios recovery,rescue_ops] [--require-seeds 3] \\
        [--json out.json]

`tools/constancy_dump.py` writes one report per (seed, scenario set).  The
schema decision needs the INTERSECTION over every pass: a field is removed
only if it is constant in EVERY pass read (the same value everywhere, too --
a field that is 0 in one pass and 1 in another is two constants but still
carries information across instances).  A field that moves anywhere stays.

The tool counts what it read and fails (exit 1) when it read nothing, when
a pass collected no sample, when the passes disagree on the schema or the
field list, or when a --require-* condition is not met -- so an empty glob,
a truncated run or a mixed-tree set cannot produce a removal list.
"""
import argparse
import json
import sys


def load_passes(paths):
    passes = []
    for path in paths:
        with open(path, encoding='utf-8') as fh:
            rep = json.load(fh)
        got = rep.get('passes') or []
        if not got:
            raise SystemExit('constancy_combine: %s holds no pass' % path)
        for p in got:
            p = dict(p)
            p['_file'] = path
            passes.append(p)
    return passes


def combine(passes):
    """Returns the per-field evidence and the verdict lists."""
    if not passes:
        raise SystemExit('constancy_combine: no pass read')
    schemas = sorted({p.get('schema_id') for p in passes})
    if len(schemas) != 1:
        raise SystemExit('constancy_combine: the passes come from different '
                         'schemas %r' % schemas)
    names = None
    for p in passes:
        n = int(p.get('samples_per_field') or 0)
        if n <= 0:
            raise SystemExit('constancy_combine: pass %s/%s in %s collected '
                             'no sample' % (p.get('scenario'), p.get('seed'),
                                            p['_file']))
        f = sorted(p['fields'])
        if names is None:
            names = f
        elif f != names:
            raise SystemExit('constancy_combine: the passes disagree on the '
                             'field list (%s)' % p['_file'])
        for name in f:
            s = p['fields'][name]
            if s is None or int(s.get('n') or 0) != n:
                raise SystemExit('constancy_combine: field %s has %r samples '
                                 'in pass %s/%s, expected %d'
                                 % (name, (s or {}).get('n'), p.get('scenario'),
                                    p.get('seed'), n))
    evidence = {}
    for name in names:
        per = []
        for p in passes:
            s = p['fields'][name]
            per.append({'scenario': p.get('scenario'), 'seed': p.get('seed'),
                        'ticks': p.get('ticks'), 'n': s['n'],
                        'constant': bool(s['constant']),
                        'min': s['min'], 'max': s['max'],
                        'variance': s['variance'],
                        'distinct': s['distinct'],
                        'constant_fraction': s['constant_fraction']})
        k = sum(1 for e in per if e['constant'])
        values = sorted({e['min'] for e in per if e['constant']})
        dead = (k == len(per) and len(values) == 1)
        evidence[name] = {'passes': len(per), 'constant_in': k,
                          'constant_values': values, 'dead': dead,
                          'samples': sum(e['n'] for e in per),
                          'per_pass': per}
    dead = sorted(n for n, e in evidence.items() if e['dead'])
    somewhere = sorted(n for n, e in evidence.items()
                       if e['constant_in'] and not e['dead'])
    return schemas[0], evidence, dead, somewhere


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('reports', nargs='+')
    ap.add_argument('--require-ticks', type=int, default=None)
    ap.add_argument('--require-passes', type=int, default=None)
    ap.add_argument('--require-scenarios', default=None,
                    help='comma-separated; every one must appear')
    ap.add_argument('--require-seeds', type=int, default=None,
                    help='at least this many distinct seeds per scenario')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)

    passes = load_passes(a.reports)
    schema, evidence, dead, somewhere = combine(passes)
    problems = []
    if a.require_passes is not None and len(passes) != a.require_passes:
        problems.append('%d passes read, %d required'
                        % (len(passes), a.require_passes))
    if a.require_ticks is not None:
        short = ['%s/%s:%s' % (p.get('scenario'), p.get('seed'), p.get('ticks'))
                 for p in passes if p.get('ticks') != a.require_ticks]
        if short:
            problems.append('passes not %d ticks long: %s'
                            % (a.require_ticks, ', '.join(short)))
    scen = sorted({p.get('scenario') for p in passes})
    if a.require_scenarios:
        missing = [s for s in a.require_scenarios.split(',') if s not in scen]
        if missing:
            problems.append('scenario(s) missing: %s' % ', '.join(missing))
    seeds_by = {s: sorted({p.get('seed') for p in passes
                           if p.get('scenario') == s}) for s in scen}
    if a.require_seeds is not None:
        few = ['%s has %d' % (s, len(v)) for s, v in seeds_by.items()
               if len(v) < a.require_seeds]
        if few:
            problems.append('fewer than %d seeds: %s'
                            % (a.require_seeds, '; '.join(few)))

    print('# constancy_combine: %d report(s), %d pass(es), schema %s'
          % (len(a.reports), len(passes), schema))
    for s in scen:
        print('#   %s: seeds %s' % (s, ', '.join(str(x) for x in seeds_by[s])))
    print('# samples per field per pass: %s'
          % ', '.join('%s/%s=%d' % (p.get('scenario'), p.get('seed'),
                                    p['samples_per_field']) for p in passes))
    print('%-32s %8s  %-10s %s' % ('field', 'const in', 'value',
                                   'per pass: variance / distinct'))
    for name in sorted(evidence):
        e = evidence[name]
        val = ('%g' % e['constant_values'][0]) if e['dead'] else (
            'several' if e['constant_in'] else '-')
        per = '; '.join('%s/%s %.2e/%d' % (x['scenario'][:4], x['seed'],
                                           x['variance'], x['distinct'])
                        for x in e['per_pass'])
        print('%-32s %4d/%-3d  %-10s %s' % (name, e['constant_in'],
                                            e['passes'], val, per))
    n_fields = len(evidence)
    n_samples = sum(e['samples'] for e in evidence.values())
    print('\nfields checked %d, passes %d, samples %d'
          % (n_fields, len(passes), n_samples))
    print('CONSTANT IN EVERY PASS (same value; remove): %s'
          % (', '.join(dead) or 'none'))
    print('CONSTANT IN SOME PASS ONLY (keep): %s'
          % (', '.join(somewhere) or 'none'))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump({'schema_id': schema, 'reports': a.reports,
                       'passes': len(passes), 'seeds_by_scenario': seeds_by,
                       'constant_in_every_pass': dead,
                       'constant_in_some_pass': somewhere,
                       'fields': evidence}, fh, indent=1, sort_keys=True)
    if problems or n_fields == 0 or n_samples == 0:
        for pr in problems:
            print('REQUIREMENT NOT MET: %s' % pr)
        print('VERDICT FAIL')
        return 1
    print('VERDICT PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
