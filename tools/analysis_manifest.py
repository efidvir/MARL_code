"""The manifest(s) the pre-registered analysis is checked against, for one
analysis step of one run-chain profile.

confirmatory_analysis.py demands that every arm holds EXACTLY the manifest's
seeds.  Which seeds a step analyses is a property of the PLAN, not of what
happens to be on disk:

    dev step      dev_full_seeds if the profile declares any, else dev_seeds
    primary step  cf_full_seeds  if the profile declares any, else cf_seeds

A reduced profile (the rehearsal) analyses only its FULL-LENGTH subset,
because the analysis holds absolute tick windows and refuses shorter runs.
Before this helper the dev step always passed the complete development
manifests, so on the merged tree -- which carries Track B's completeness
assert -- the rehearsal's two-seed subset failed with 18 seeds "missing"
(2026-09-29), and the primary step would have failed the same way.

The expected set is taken from the plan and NEVER from the pickles, so the
analysis's completeness check stays meaningful: an invocation that silently
dropped a seed still fails it.

  * declared set == every seed in the source manifests  -> the source
    manifest paths are printed UNCHANGED (the real profile's path, byte for
    byte what it was);
  * declared set is a strict subset                     -> a derived manifest
    holding only those instances is written to --out and its path printed;
  * a declared seed absent from the source manifests for any scenario kind,
    an empty declaration, or a derived manifest with no instances -> exit 2.

usage: analysis_manifest.py --profile P --step dev|primary --out PATH --manifests M1[,M2]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'confirmatory'))
import chain_plan as CP  # noqa: E402

FIELDS = {'dev': ('dev_full_seeds', 'dev_seeds'),
          'primary': ('cf_full_seeds', 'cf_seeds')}


def declared_seeds(profile, step):
    prof = CP.PROFILES[profile]
    first, fallback = FIELDS[step]
    seeds = list(prof.get(first) or []) or list(prof.get(fallback) or [])
    return sorted(int(s) for s in seeds)


def resolve(profile, step, manifests, out):
    seeds = declared_seeds(profile, step)
    if not seeds:
        raise SystemExit2('profile %r declares no %s seeds' % (profile, step))
    instances, by_kind = [], {}
    for m in manifests:
        for inst in json.load(open(m))['instances']:
            instances.append(inst)
            by_kind.setdefault(inst['scenario_kind'], set()).add(int(inst['topology_seed']))
    if not instances:
        raise SystemExit2('source manifests %s hold no instances' % manifests)
    for kind, have in sorted(by_kind.items()):
        miss = sorted(set(seeds) - have)
        if miss:
            raise SystemExit2('profile %r %s step declares seeds %s absent from the %s '
                              'instances of %s' % (profile, step, miss, kind, manifests))
    every = set().union(*by_kind.values())
    if set(seeds) == every:
        return ','.join(manifests), None
    keep = [i for i in instances if int(i['topology_seed']) in set(seeds)]
    if not keep:
        raise SystemExit2('derived manifest would hold no instances')
    payload = {
        'note': ('DERIVED for the %s analysis step of profile %r: the source '
                 'manifests restricted to the seeds this step declares. The '
                 'expected set comes from the plan, never from the pickles.'
                 % (step, profile)),
        'profile': profile, 'step': step,
        'declared_seeds': seeds,
        'source_manifests': list(manifests),
        'instances': keep,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='\n') as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write('\n')
    return out, payload


class SystemExit2(Exception):
    pass


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--profile', required=True, choices=sorted(CP.PROFILES))
    ap.add_argument('--step', required=True, choices=sorted(FIELDS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifests', required=True)
    a = ap.parse_args(argv)
    try:
        path, payload = resolve(a.profile, a.step, a.manifests.split(','), a.out)
    except SystemExit2 as e:
        print('[analysis_manifest] REFUSE: %s' % e, file=sys.stderr)
        return 2
    if payload is None:
        print('[analysis_manifest] %s/%s: declared seeds equal the manifests; passing them unchanged'
              % (a.profile, a.step), file=sys.stderr)
    else:
        print('[analysis_manifest] %s/%s: derived manifest for seeds %s (%d instances) -> %s'
              % (a.profile, a.step, payload['declared_seeds'], len(payload['instances']), path),
              file=sys.stderr)
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
