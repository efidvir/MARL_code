"""WP9: tools/compare_reports.py (report reproduction check) and
tools/hash_invariance_check.py (S8) on synthetic inputs."""
import importlib.util
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_wp9_synth as SY  # noqa: E402


def _load(name):
    path = os.path.join(ROOT, 'tools', name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR = _load('compare_reports')
HI = _load('hash_invariance_check')


def test_compare_reports_leaf_rules():
    ref = {'primary': {'P1': {'mean': 1.5, 'confirmed': True, 'n': 20, 'p_scipy': None}},
           'seeds': {'A': [1, 2, 3]}, 'secondary': {'S1': {'mean': float('nan')}}}
    new = json.loads(json.dumps(ref))
    assert CR.compare(ref, new) == ([], [])
    new['primary']['P1']['mean'] = 1.5 + 1e-9
    m, n = CR.compare(ref, new)
    assert [x[0] for x in m] == ['/primary/P1/mean'] and n == []
    assert CR.compare(ref, new, rtol=1e-6) == ([], [])
    new['primary']['P1']['mean'] = 1.5
    new['primary']['P1']['extra'] = 7                       # new output: a note, not a failure
    new['stability'] = {'S9a': 1}
    m, n = CR.compare(ref, new)
    assert m == [] and n == ['/primary/P1/extra', '/stability']
    del new['primary']['P1']['confirmed']                   # lost output: a failure
    m, _ = CR.compare(ref, new)
    assert m == [('/primary/P1/confirmed', True, '<missing>')]
    new['primary']['P1']['confirmed'] = 1                   # bool vs int is a mismatch
    m, _ = CR.compare(ref, new)
    assert m and m[0][0] == '/primary/P1/confirmed'
    new['primary']['P1']['confirmed'] = True
    new['seeds']['A'] = [1, 2]
    m, _ = CR.compare(ref, new)
    assert m == [('/seeds/A', 'len 3', 'len 2')]
    assert CR.compare({'x': None}, {'x': None}) == ([], [])
    assert CR.compare({'x': None}, {'x': 0})[0]
    assert CR.count_leaves(ref) == 8


def test_compare_reports_cli_exit_codes():
    tmp = SY.mkdtemp('wp9_cmp_')
    ref = {'seeds': {'A': [1]}, 'primary': {'P1': {'mean': 0.5, 'wilcoxon_p_scipy': 0.1}}, 'secondary': {}}
    new = json.loads(json.dumps(ref))
    new['S10'] = {}
    a, b = os.path.join(tmp, 'ref.json'), os.path.join(tmp, 'new.json')
    json.dump(ref, open(a, 'w'))
    json.dump(new, open(b, 'w'))
    assert CR.main([a, b]) == 0
    new['primary']['P1']['wilcoxon_p_scipy'] = 0.2
    json.dump(new, open(b, 'w'))
    assert CR.main([a, b]) == 1
    assert CR.main([a, b, '--ignore', 'wilcoxon_p_scipy']) == 0
    assert CR.main([a, b, '--sections', 'seeds']) == 0
    assert CR.main([a, os.path.join(tmp, 'nope.json')]) == 2


def _pair(tmp, mutate_b=None, prov_b=None):
    arms, seeds = ('marl_static', 'xldet'), [1, 2]
    a = SY.make_pickle(os.path.join(tmp, 'h0.pkl'), arms, seeds, prov=SY.provenance(pythonhashseed='0'))
    b = SY.make_pickle(os.path.join(tmp, 'h1.pkl'), arms, seeds, mutate=mutate_b,
                       prov=prov_b or SY.provenance(pythonhashseed='1'))
    return a, b


def test_hash_invariance_identical_and_accounting_jitter():
    tmp = SY.mkdtemp('wp9_hash_')
    a, b = _pair(tmp)
    res = HI.check(pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb')))
    assert res['identical'] and res['n_different'] == 0 and res['n_missing'] == 0
    assert res['same_hashseed'] is False and res['same_tree'] is True and res['note'] is None
    assert res['precondition_not_established'] == []      # seeds 0 and 1: a real hash-invariance pair
    assert res['n_compared'] > 0 and set(res['accounting_jitter']) == {'energy', 'latency'}
    assert res['accounting_jitter']['energy']['n_identical'] == res['accounting_jitter']['energy']['n_series'] == 8

    def jitter(sk, arm, s, run):
        run['energy'] = [e * (1 + 1e-6) for e in run['energy']]
        return run
    a, b = _pair(tmp, mutate_b=jitter)
    res = HI.check(pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb')))
    assert res['identical']                                     # accounting jitter is not a decision change
    j = res['accounting_jitter']['energy']
    assert j['n_identical'] == 0 and 0 < j['max_rel_mean_diff'] < 1e-5
    res = HI.check(pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb')), strict=True)
    assert not res['identical'] and all(d['key'] == 'energy' for d in res['differences'])


def test_hash_invariance_detects_decision_changes_and_missing_runs():
    tmp = SY.mkdtemp('wp9_hash2_')

    def flip(sk, arm, s, run):
        if arm == 'xldet' and s == 2 and sk == 'recovery_multi':
            run['fragments'][-1] += 1
            run['relay_lifetimes'] = run['relay_lifetimes'] + [('TR_9_9_9', 1, 2, 3, 'x', False, 0.0)]
        if arm == 'marl_static' and s == 1 and sk == 'rescue_multi':
            return None
        return run
    a, b = _pair(tmp, mutate_b=flip)
    res = HI.check(pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb')))
    assert not res['identical']
    keys = sorted((d['arm'], d['seed'], d['key']) for d in res['differences'])
    assert keys == [('xldet', 2, 'fragments'), ('xldet', 2, 'relay_lifetimes')]
    assert res['missing'] == [{'scenario': 'rescue_ops', 'arm': 'marl_static', 'seed': 1, 'key': '*', 'where': 'B'}]
    # restricting the keys / arms / seeds
    res = HI.check(pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb')), keys=['conn'], arms=['xldet'])
    assert res['identical'] and res['n_compared'] == 4
    res = HI.check(pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb')), seeds=[2], scenarios=['recovery'])
    assert res['n_different'] == 2 and res['n_missing'] == 0


def test_hash_invariance_cli_and_same_seed_note():
    tmp = SY.mkdtemp('wp9_hash3_')
    a, b = _pair(tmp, prov_b=SY.provenance(pythonhashseed='0'))
    out = os.path.join(tmp, 's8.json')
    # both pickles ran under PYTHONHASHSEED=0: a repeat-run pair, NOT a pass
    assert HI.main([a, b, '--out', out, '--quiet']) == 2
    res = json.load(open(out))
    assert res['identical'] is False and res['same_hashseed'] is True and 'repeat-run' in res['note']
    assert res['files'] == {'A': a, 'B': b}
    assert HI.main([a, os.path.join(tmp, 'missing.pkl')]) == 2
    assert HI.main([a, b, '--scenarios', 'Q']) == 2


def test_hash_invariance_refuses_a_pair_that_cannot_show_hash_invariance():
    """D1 residual: `identical` required n_compared > 0 but not that the two
    pickles ran under DIFFERENT hash seeds, and `same_hashseed` was an advisory
    `note` outside both the verdict and the exit code.  A repeat-run pair --
    and, as in output/wp9_validation/s8_smoke_repeat.json, a pair where neither
    pickle records a seed at all -- therefore read as an S8 pass (identical
    True, exit 0) for a check that verified nothing about hash order."""
    tmp = SY.mkdtemp('wp9_hash6_')

    def verdict(A, B, **kw):
        r = HI.check(A, B, **kw)
        return r['identical'], r['same_hashseed'], r['precondition_not_established']

    # (a) both pickles record pythonhashseed '0' -- the repeat-run pair
    a, b = _pair(tmp, prov_b=SY.provenance(pythonhashseed='0'))
    A, B = pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb'))
    ident, seed, why = verdict(A, B)
    assert ident is False and seed is True and len(why) == 1
    assert 'REPEAT-RUN' in why[0] and "'0'" in why[0]
    assert HI.check(A, B)['n_compared'] > 0 and HI.check(A, B)['n_different'] == 0   # the series DO agree
    assert HI.main([a, b, '--quiet']) == 2

    # (b) neither pickle records a provenance hash seed.  Three shapes of
    # "not knowable", all of them exit 2: no provenance record at all (the
    # frozen pickles), a provenance record without the field, and one where
    # the field is present but null.
    c = SY.make_pickle(os.path.join(tmp, 'noprov0.pkl'), ('marl_static',), [1, 2], prov=None)
    d_ = SY.make_pickle(os.path.join(tmp, 'noprov1.pkl'), ('marl_static',), [1, 2], prov=None)
    C, D = pickle.load(open(c, 'rb')), pickle.load(open(d_, 'rb'))
    ident, seed, why = verdict(C, D)
    assert ident is False and seed is None and len(why) == 1 and 'not established' in why[0]
    assert HI.check(C, D)['n_compared'] > 0 and HI.check(C, D)['n_missing'] == 0
    assert 'not recorded' in HI.check(C, D)['note']
    assert HI.main([c, d_, '--quiet']) == 2

    noseed = SY.provenance()
    noseed.pop('pythonhashseed')
    e = SY.make_pickle(os.path.join(tmp, 'noseed.pkl'), ('marl_static',), [1, 2], prov=noseed)
    E = pickle.load(open(e, 'rb'))
    assert verdict(E, E)[:2] == (False, None)
    nullseed = SY.provenance(pythonhashseed=None)
    f = SY.make_pickle(os.path.join(tmp, 'nullseed.pkl'), ('marl_static',), [1, 2], prov=nullseed)
    F = pickle.load(open(f, 'rb'))
    assert verdict(F, F)[:2] == (False, None)
    # one side recorded, the other not, is still not knowable
    assert verdict(pickle.load(open(a, 'rb')), E)[:2] == (False, None)

    # the fix must not make a genuine hash-invariance pair fail
    g, h = _pair(SY.mkdtemp('wp9_hash6b_'))
    G, H = pickle.load(open(g, 'rb')), pickle.load(open(h, 'rb'))
    assert verdict(G, H) == (True, False, [])
    assert HI.main([g, h, '--quiet']) == 0
    # ... and the embedded --s8 JSON carries the refusal for confirmatory_analysis
    out = os.path.join(tmp, 's8_repeat.json')
    assert HI.main([a, b, '--out', out, '--quiet']) == 2
    j = json.load(open(out))
    assert j['identical'] is False and j['precondition_not_established'] and j['same_hashseed'] is True
    json.dumps(j)


def test_hash_invariance_never_passes_when_nothing_was_compared():
    """D1: `identical = not diffs and not missing` is vacuously true for
    n_compared == 0, so a --seeds/--arms typo, pickles without a 'seeds' key or
    --require-decision with no digest anywhere reported IDENTICAL (exit 0) for
    a pre-registered check that verified nothing."""
    tmp = SY.mkdtemp('wp9_hash5_')
    a, b = _pair(tmp)
    A, B = pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb'))
    ok = HI.check(A, B)
    assert ok['identical'] and ok['nothing_compared'] == [] and ok['n_compared'] > 0
    # a seed that matches no run (a typo)
    res = HI.check(A, B, seeds=[999])
    assert res['n_compared'] == 0 and res['identical'] is False and len(res['nothing_compared']) == 1
    assert 'nothing compared' in res['nothing_compared'][0]
    assert HI.main([a, b, '--seeds', '999', '--quiet']) == 2
    # an arm that matches no run (a typo)
    res = HI.check(A, B, arms=['marl_statc'])
    assert res['n_compared'] == 0 and res['identical'] is False and res['nothing_compared']
    assert HI.main([a, b, '--arms', 'marl_statc', '--quiet']) == 2
    # pickles with no top-level 'seeds' key: the seed list is empty, so nothing
    # is compared -- even though the runs differ
    A2 = {k: v for k, v in A.items() if k != 'seeds'}
    B2 = {k: v for k, v in B.items() if k != 'seeds'}
    B2['recovery_multi']['marl_static'][1]['conn'] = [0.0] * SY.T_TEST
    res = HI.check(A2, B2)
    assert res['seeds'] == [] and res['n_compared'] == 0 and res['identical'] is False
    assert 'seeds' in res['nothing_compared'][0]
    json.dumps(res)
    # the JSON confirmatory_analysis embeds with --s8 carries the reason
    out = os.path.join(tmp, 's8_vacuous.json')
    assert HI.main([a, b, '--seeds', '999', '--out', out, '--quiet']) == 2
    assert json.load(open(out))['identical'] is False and json.load(open(out))['nothing_compared']
    # --require-decision with no MARL-path run at all: runs ARE compared, but
    # no decision series is, so the study setting verified nothing either
    arms, seeds = ('sdn',), [1, 2]
    c = SY.make_pickle(os.path.join(tmp, 'sdn0.pkl'), arms, seeds, prov=SY.provenance(pythonhashseed='0'))
    d_ = SY.make_pickle(os.path.join(tmp, 'sdn1.pkl'), arms, seeds, prov=SY.provenance(pythonhashseed='1'))
    C, D = pickle.load(open(c, 'rb')), pickle.load(open(d_, 'rb'))
    assert HI.check(C, D)['identical']                                  # without the flag: a real pass
    res = HI.check(C, D, require_decision=True)
    assert res['n_compared'] > 0 and res['n_missing'] == 0 and res['identical'] is False
    assert res['decision_series']['runs_with_key'] == 0
    assert 'require-decision' in res['nothing_compared'][0]
    assert HI.main([c, d_, '--require-decision', '--quiet']) == 2
    assert HI.main([c, d_, '--quiet']) == 0


def test_hash_invariance_decision_coverage_and_require_decision():
    tmp = SY.mkdtemp('wp9_hash4_')
    a, b = _pair(tmp)
    A, B = pickle.load(open(a, 'rb')), pickle.load(open(b, 'rb'))
    res = HI.check(A, B)
    d = res['decision_series']
    assert d['key'] == 'decision_digest' and d['runs_with_key'] == 0 and d['runs_without_key'] == 8
    assert sorted(d['arms_without_key']) == ['marl_static', 'xldet'] and d['coverage'].startswith('consequences only')
    assert res['identical'] and res['accounting_note'] and 'energy/latency' in res['accounting_note']
    # the study setting: no digest -> missing series, NOT IDENTICAL
    res = HI.check(A, B, require_decision=True)
    assert not res['identical'] and res['n_missing'] == 8
    assert all(m['key'] == 'decision_digest' and m['where'] == 'both' for m in res['missing'])
    assert res['decision_series']['required'] is True
    assert HI.main([a, b, '--require-decision', '--quiet']) == 1
    assert HI.main([a, b, '--quiet']) == 0
    # digest recorded and identical: covered; a differing digest is a difference even under --keys
    arms, seeds = ('marl_static', 'xldet'), [1, 2]

    def dig(sk, arm, s, run):
        run['decision_digest'] = ['%s-%s-%d' % (arm, sk, t) for t in range(SY.T_TEST)]
        return run

    def dig_flip(sk, arm, s, run):
        run = dig(sk, arm, s, run)
        if arm == 'xldet' and s == 2 and sk == 'recovery_multi':
            run['decision_digest'][7] = 'other'
        return run
    c = SY.make_pickle(os.path.join(tmp, 'h0d.pkl'), arms, seeds, mutate=dig, prov=SY.provenance(pythonhashseed='0'))
    d_ = SY.make_pickle(os.path.join(tmp, 'h1d.pkl'), arms, seeds, mutate=dig, prov=SY.provenance(pythonhashseed='1'))
    e = SY.make_pickle(os.path.join(tmp, 'h1e.pkl'), arms, seeds, mutate=dig_flip, prov=SY.provenance(pythonhashseed='1'))
    res = HI.check(pickle.load(open(c, 'rb')), pickle.load(open(d_, 'rb')), require_decision=True)
    assert res['identical'] and res['decision_series']['coverage'].startswith('decisions')
    assert res['decision_series']['runs_with_key'] == 8 and res['decision_series']['runs_without_key'] == 0
    res = HI.check(pickle.load(open(c, 'rb')), pickle.load(open(e, 'rb')), keys=['conn'])
    assert not res['identical'] and [(x['arm'], x['seed'], x['key']) for x in res['differences']] == [('xldet', 2, 'decision_digest')]
    assert res['differences'][0]['diff'] and res['n_compared'] == 16      # conn + digest per run
    # strict mode drops the accounting note
    assert HI.check(pickle.load(open(c, 'rb')), pickle.load(open(d_, 'rb')), strict=True)['accounting_note'] is None
