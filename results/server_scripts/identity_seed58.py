#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bit-identity of the Fig. 6 process-map pass (replicate 1234, development
seed 58, Scenario A, 3600 ticks, r4 tree) against the recorded development
pass of the same checkpoint on the same seed (Phase 0, dev_ck1234_b.pkl,
tree 6b53f3f).

  python3 identity_seed58.py REF.pkl NEW_RUN.pkl DUMP.json OUT.json

Compares, with series_diff.same (floats bit for bit, NaN == NaN):
  * the achievability (`conn`) and component (`fragments`, `fragments_raw`)
    series tick by tick (3600 entries each, counted);
  * every key of the reference run (the new run may add none but R4 keys);
  * the steady achievability and components of the dump JSON against
    confirmatory_analysis.steady of the reference run;
  * the checkpoint digest of both passes (replicate 1234's pinned digest).
Exit 0 = IDENTICAL.
"""
import hashlib
import json
import os
import pickle
import sys

TREE = os.path.expanduser(os.environ.get('R4_TREE', '~/r4_work/tree'))
for p in (TREE, os.path.join(TREE, 'tools'), os.path.join(TREE, 'confirmatory')):
    if p not in sys.path:
        sys.path.insert(0, p)

from series_diff import same, compare_series      # noqa: E402
import confirmatory_analysis as CA                # noqa: E402
import chain_plan as CP                           # noqa: E402

SEED, ARM, SC = 58, 'marl_static', 'recovery_multi'
R4_KEYS = {'r4_diag', 'comparator'}


def dp_probe(dump):
    return int((dump.get('provenance') or {}).get('probe_tick', -1))


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def main():
    ref_p, new_p, dump_p, out_p = sys.argv[1:5]
    with open(ref_p, 'rb') as f:
        REF = pickle.load(f)
    with open(new_p, 'rb') as f:
        NEW = pickle.load(f)
    with open(dump_p, encoding='utf-8') as f:
        dump = json.load(f)
    ref = REF[SC][ARM][SEED]
    new = NEW[SC][ARM][SEED]
    fails = []
    series = {}
    for key in ('conn', 'fragments', 'fragments_raw'):
        a, b = ref.get(key), new.get(key)
        if a is None or b is None:
            fails.append('%s missing (ref %s, new %s)' % (key, a is None, b is None))
            continue
        n_same = sum(1 for x, y in zip(a, b) if same(x, y))
        d = compare_series(a, b)
        series[key] = {'ticks_ref': len(a), 'ticks_new': len(b), 'ticks_bit_identical': n_same,
                       'identical': d is None, 'first_difference': d}
        if d is not None or len(a) != CA.T_TOTAL or n_same != CA.T_TOTAL:
            fails.append('%s differs: %s' % (key, d))
    keys = sorted(ref.keys())
    differing = [k for k in keys if k not in new or not same(ref[k], new[k])]
    extra = sorted(set(new) - set(ref))
    if differing:
        fails.append('keys differing: %s' % differing)
    if not set(extra) <= R4_KEYS:
        fails.append('unexpected extra keys in the new run: %s' % extra)
    if len(keys) < 50:
        fails.append('only %d keys compared' % len(keys))
    steady = {'achievability_ref': CA.steady(ref['conn']), 'achievability_dump': dump['steady_achievability'],
              'components_ref': CA.steady(ref['fragments']), 'components_dump': dump['steady_components']}
    for a_, b_ in (('achievability_ref', 'achievability_dump'), ('components_ref', 'components_dump')):
        if not same(float(steady[a_]), float(steady[b_])):
            fails.append('steady %s %r != %s %r' % (a_, steady[a_], b_, steady[b_]))
    pin = CP.R2_CHECKPOINT_SHA256[1234]
    ref_ck = list((REF.get('provenance') or {}).get('checkpoint_sha256') or [])
    new_ck = (dump.get('provenance') or {}).get('checkpoint_sha256')
    if ref_ck != [pin] or new_ck != pin:
        fails.append('checkpoint digests ref %s new %s, pinned %s' % (ref_ck, new_ck, pin))
    end = [s for s in dump['snapshots'] if s['tag'] == 'end']
    snaps = {s['tag']: {'tick': s['tick'], 'components': s['components'],
                        'components_raw': s['components_raw'], 'distinct_bridges': s['distinct_bridges'],
                        'relay_links': len(s['relay_links'])} for s in dump['snapshots']}
    # dump_endstate.py's 'pre' probe (tick 150) never fires: the v10 dump of
    # replicate 4567 (~/MARL_run_9/output/final/endstate_seed58.json) also
    # carries only 'cut' (tick 260) and 'end' (PROBE_TICK); the figure uses those two
    tags = [s['tag'] for s in dump['snapshots']]
    if len(end) != 1 or tags.count('cut') != 1 or snaps.get('cut', {}).get('tick') != 260 \
            or snaps['end']['tick'] != dp_probe(dump):
        fails.append('snapshots %s' % tags)
    rp, dp = REF.get('provenance') or {}, dump.get('provenance') or {}
    rep = {'verdict': 'IDENTICAL' if not fails else 'DIFFERENT', 'failures': fails,
           'seed': SEED, 'arm': ARM, 'scenario': 'A (recovery)',
           'series': series, 'keys_compared': len(keys), 'keys_differing': differing,
           'extra_keys_new': extra, 'decision_digest_ticks': len(ref.get('decision_digest') or []),
           'steady': steady, 'snapshots': snaps,
           'ref': {'pickle': os.path.abspath(ref_p), 'md5': md5(ref_p), 'tree_head': rp.get('tree_head'),
                   'tree_sha256': rp.get('tree_sha256'), 'marl_env': rp.get('marl_env'),
                   'checkpoint': rp.get('checkpoint'), 'checkpoint_sha256': ref_ck,
                   'created_utc': rp.get('created_utc')},
           'new': {'run_pickle': os.path.abspath(new_p), 'dump_json': os.path.abspath(dump_p),
                   'dump_md5': md5(dump_p), 'tree_root': dp.get('tree_root'), 'tree_head': dp.get('tree_head'),
                   'tree_sha256': dp.get('tree_sha256'), 'marl_env': dp.get('marl_env'),
                   'pythonhashseed': dp.get('pythonhashseed'), 'checkpoint': dp.get('checkpoint'),
                   'checkpoint_sha256': new_ck, 'probe_tick': dp.get('probe_tick'),
                   'created_utc': dp.get('created_utc')},
           'pinned_checkpoint_sha256': pin}
    with open(out_p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(rep, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    print(json.dumps({k: rep[k] for k in ('verdict', 'failures', 'keys_compared', 'steady', 'snapshots')},
                     indent=1, default=str))
    for k, v in series.items():
        print('%s: %d/%d ticks bit-identical' % (k, v['ticks_bit_identical'], v['ticks_ref']))
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
