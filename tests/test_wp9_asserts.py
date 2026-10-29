"""WP9: the hard asserts of confirmatory_analysis.analyse() on synthetic
pickles -- a clean set passes strictly, every assert fires on a corrupted set,
--no-assert turns the failures into a report section, S9/S10/S11/S12 blocks
are produced, and legacy pickles (no WP4 series, no provenance) report
'not recorded' rather than a lower bound.  Review fixes: provenance fields
must be present (not merely identical), numpy scalars/arrays are checked and
never dropped, the {name: series} members are NaN-checked, the postcard
check is per site when recorded and labelled aggregate otherwise, the
adapt-then-freeze arm must be complete once present, and the checkpoint
sha256 must agree across the pickles carrying the tested arm."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import confirmatory_analysis as CA  # noqa: E402
import test_wp9_synth as SY  # noqa: E402

np = SY.np
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
MARL_ARMS = ('marl_static', 'xldet', 'random')
BASE_ARMS = ('sdn', 'ospf', 'olsr', 'batman', 'aodv')
# numeric per-tick series of one synthetic run (the same for every arm): 19
# top-level (conn, offered, delivered, energy, fragments, fragments_raw,
# relay_count, relay_bridges, postcards, latency, overhead, relay_created,
# relay_removed, bridge_teardowns, idle_bridge_teardowns, relay_mode_flips,
# stale_edges, stale_u2u_volume, energy_ctrl) + 8 {name: series} members
# (overhead_by_type x4, relay_removed_by_reason x2, relay_teardown_holds x2).
# relay_lifetimes (records), the scalars and the fingerprint strings are not
# series.  This pins checks.run_length's coverage count (D4).
N_SERIES_PER_RUN = 27


def _set(tmp, mutate_a=None, mutate_b=None, prov_b='default', events=True):
    a = SY.make_pickle(os.path.join(tmp, 'marl.pkl'), MARL_ARMS, SEEDS, mutate=mutate_a, events=events)
    b = SY.make_pickle(os.path.join(tmp, 'base.pkl'), BASE_ARMS, SEEDS, mutate=mutate_b, prov=prov_b, events=events)
    m = SY.make_manifest(os.path.join(tmp, 'manifest.json'), SEEDS)
    return [a, b], [m]


def _analyse(pickles, manifests, **kw):
    kw.setdefault('marl_arm', 'marl_static')
    kw.setdefault('t_sev', SY.T_SEV_TEST)
    kw.setdefault('t_total', SY.T_TEST)
    with SY.short_run_patch():
        return CA.analyse(pickles, manifests, **kw)


def test_clean_set_passes_strict_and_produces_every_block():
    tmp = SY.mkdtemp('wp9_clean_')
    pk, mf = _set(tmp)
    rep = _analyse(pk, mf, strict=True)
    assert rep['asserts']['enforced'] and rep['asserts']['n_failures'] == 0, rep['asserts']['failures']
    assert rep['seeds'] == {'A': SEEDS, 'B': SEEDS} and rep['excluded'] == {'A': [], 'B': []}
    assert len(rep['primary']) == 9 and all(set(r) >= {'wilcoxon_p_holm', 'confirmed'} for r in rep['primary'].values())
    assert [k[:2] for k in rep['secondary']][:11] == ['S1', 'S1', 'S2', 'S2', 'S2', 'S3', 'S3', 'S3', 'S4', 'S5', 'S6']
    assert list(rep['stability']) == ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
    for eid, r in rep['stability'].items():
        assert r['status'] in ('assessable', CA.NOT_ASSESSABLE), (eid, r['status'])
        assert r['n'] == len(SEEDS)
    # marl flaps 3/2 every two ticks: M2 > 0 on every instance, XL-DET has none
    s10 = rep['S10']
    assert s10['A']['marl_static']['M2']['min'] > 0 and s10['A']['xldet']['M2']['max'] == 0
    assert s10['A']['marl_static']['monotone_reunification'] is False
    assert s10['A']['ospf']['monotone_reunification'] is True
    # WP4 series present: M7/M8/M9/M11 recorded, no lower-bound keys
    blk = s10['A']['marl_static']
    assert blk['M7']['n_recorded'] == len(SEEDS) and blk['M7']['median'] > 0
    assert blk['M8_bridge']['n_recorded'] == len(SEEDS) and blk['M8_idle']['n_recorded'] == len(SEEDS)
    assert blk['M9']['median'] > 0 and blk['M11']['n_bridges'] == 2 * len(SEEDS) and blk['M11']['n_censored'] == len(SEEDS)
    # M11 per run: bridge 1 completed (3 ticks), bridge 2 censored -> KM median 3 on every run
    assert blk['M11']['n_completed'] == len(SEEDS) and blk['M11']['median'] == 3 and blk['M11']['n_runs_km_undefined'] == 0
    assert blk['M11_completed_only (descriptive)']['median'] == 3
    assert blk['M11_censored_at_T (lower bound)']['median'] > 3
    assert not any('lower_bound_' in k for k in blk)
    # optimum 1 on even seeds (never reached: censored), 2 on odd seeds (marl ends at 2: settled)
    assert blk['M6']['n_censored'] == len(SEEDS) // 2
    assert s10['A']['xldet']['M6']['n_censored'] == len(SEEDS) // 2
    assert s10['A']['xldet']['per_seed']['1']['M6'] == {'value': 3, 'censored': False}   # 3 -> 2 at ts + 3
    assert s10['A']['xldet']['per_seed']['2']['M6'] == {'value': SY.T_TEST - 5, 'censored': True}
    assert 'surge_window_20' in s10['B']['marl_static'] and 'surge_window_20' not in s10['A']['marl_static']
    # per-seed metrics are exported for the manuscript tables
    ps = blk['per_seed'][str(SEEDS[0])]
    assert set(ps) == set(CA.STABILITY_METRICS)
    # S11 reads the by-type breakdown and the postcards
    s11 = rep['S11']['A']['marl_static']
    assert set(s11['by_type_post_event']) == {'bfd', 'hello', 'lsa', 'postcard'}  # wp7-scan-ok: WP5 DCC message type, not an action head
    assert s11['postcards_total']['median'] == 3 * SY.N_SITES
    assert s11['overhead_total_post_event']['n_recorded'] == len(SEEDS)
    assert s11['energy_ctrl_total']['median'] > 0
    assert rep['S12'] == {} and 'S8' not in rep
    chk = rep['asserts']['checks']
    assert chk['postcards']['runs_checked'] == len(MARL_ARMS) * len(SEEDS) * 2
    assert chk['stale_edges']['runs_with_key'] == (len(MARL_ARMS) + len(BASE_ARMS)) * len(SEEDS) * 2 * 2
    assert chk['nan']['exempt_counts'] == {'latency': 4 * len(SEEDS) * 2}     # routing arms, one NaN each
    assert chk['provenance']['fields']['tree_sha256'] == ['f' * 64]
    assert chk['provenance']['missing_fields'] == {}
    assert chk['provenance']['checkpoint_sha256'] == {'marl.pkl': ['c' * 64]}     # only the pickle carrying marl_static
    assert chk['postcards']['mode'] == 'aggregate' and chk['postcards']['runs_with_per_site'] == 0
    assert chk['nan']['non_numeric'] == []
    assert rep['config']['T_SEV'] == SY.T_SEV_TEST and rep['config']['marl_arm'] == 'marl_static'
    json.dumps(rep)


def test_every_hard_assert_fires_and_no_assert_reports_them():
    tmp = SY.mkdtemp('wp9_broken_')

    def mut_a(sk, arm, s, run):
        if arm == 'xldet' and s == 2 and sk == 'recovery_multi':
            run['postcards'] = [SY.N_SITES if t == SY.T_SEV_TEST['A'] else 0 for t in range(SY.T_TEST)]  # 1/site
        if arm == 'random' and s == 5 and sk == 'rescue_multi':
            run['stale_edges'][7] = 2
        return run

    def mut_b(sk, arm, s, run):
        if arm == 'aodv' and s == 3 and sk == 'rescue_multi':
            return None                                                     # missing seed
        if arm == 'sdn' and s == 1 and sk == 'rescue_multi':
            run['conn'][3] = float('nan')                                   # NaN in a KPI series
        if arm == 'ospf' and s == 4 and sk == 'recovery_multi':
            run['conn'] = run['conn'][:-1]                                  # run length
        return run

    pk, mf = _set(tmp, mut_a, mut_b, prov_b=SY.provenance(tree_sha256='0' * 64, pythonhashseed='1'))
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('strict analysis must stop on the corrupted set')
    for needle in ('arm aodv scenario B: missing seeds [3]', 'tree_sha256 differs', 'pythonhashseed differs',
                   'PYTHONHASHSEED is not 0', 'stale_edges sums to 2', 'postcards after the event',
                   'aggregate: mean per site must be > 1', 'engine-side WP5 invariant',
                   'series conn has 1 NaN/None', 'series conn has 39 entries, expected 40 (run length)'):
        assert needle in msg, (needle, msg)
    rep = _analyse(pk, mf, strict=False)
    inv = rep['asserts']
    assert inv['enforced'] is False and inv['n_failures'] == len(inv['failures']) >= 8
    assert inv['checks']['arms_and_seeds']['missing'] == {'B/aodv': [3]}
    assert inv['checks']['stale_edges']['violations'] == [['B', 'random', 5, 'stale_edges', 2]]
    assert inv['checks']['postcards']['violations'] == [['A', 'xldet', 2, SY.N_SITES, SY.N_SITES]]
    assert inv['checks']['nan']['violations'] == [['B', 'sdn', 1, 'conn', 1]]
    assert inv['checks']['run_length']['violations'] == [['A', 'ospf', 4, 'conn', 39]]
    # the analysis went on with the intersection (the pre-WP9 behaviour)
    assert rep['seeds']['B'] == [s for s in SEEDS if s != 3] and rep['excluded']['B'] == [3]
    assert len(rep['primary']) == 9


def test_manifest_severance_mismatch_is_an_assert():
    tmp = SY.mkdtemp('wp9_manifest_')
    pk, _ = _set(tmp)
    mf = [SY.make_manifest(os.path.join(tmp, 'bad_manifest.json'), SEEDS, ts={'A': 7, 'B': 0})]
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        assert 'severance_tick 7, expected 5' in str(e)
    else:
        raise AssertionError('a manifest with another severance tick must fail')


def test_legacy_pickles_report_not_recorded_never_a_lower_bound():
    tmp = SY.mkdtemp('wp9_legacy_')
    a = SY.make_pickle(os.path.join(tmp, 'marl.pkl'), MARL_ARMS, SEEDS, prov=None, events=False)
    b = SY.make_pickle(os.path.join(tmp, 'base.pkl'), BASE_ARMS, SEEDS, prov=None, events=False)
    m = SY.make_manifest(os.path.join(tmp, 'manifest.json'), SEEDS)
    try:
        _analyse([a, b], [m], strict=True)
    except AssertionError as e:
        assert 'no provenance record' in str(e)
    else:
        raise AssertionError('legacy pickles must fail the provenance assert')
    rep = _analyse([a, b], [m], strict=False)
    fails = rep['asserts']['failures']
    assert len(fails) == 2 and all('no provenance record' in f for f in fails), fails
    assert rep['asserts']['checks']['stale_edges']['runs_with_key'] == 0
    for eid in ('S9e', 'S9f'):
        assert rep['stability'][eid]['status'] == CA.NOT_RECORDED
    for eid in ('S9a', 'S9b', 'S9c', 'S9d'):
        assert rep['stability'][eid]['status'] != CA.NOT_RECORDED
    blk = rep['S10']['A']['marl_static']
    for m_ in ('M7', 'M8_bridge', 'M8_idle', 'M9', 'M11'):
        assert blk[m_].get('status') == CA.NOT_RECORDED, m_
    lb = [k for k in blk if k.startswith('M7_lower_bound')]
    assert len(lb) == 1 and 'post hoc' in lb[0] and blk[lb[0]]['n_recorded'] == len(SEEDS)
    assert blk['M1']['n_recorded'] == len(SEEDS) and blk['M4ss']['n_recorded'] == len(SEEDS)
    assert rep['S11']['A']['marl_static']['by_type_post_event'] == {'status': CA.NOT_RECORDED}
    assert rep['S11']['A']['marl_static']['energy_ctrl_total']['status'] == CA.NOT_RECORDED
    # the helper itself gives None on such a run (no lower bound anywhere)
    assert CA.m7_teardowns_per_1000(SY.make_run(SY.T_TEST, 1, 'marl_static', 'A', events=False), 5) is None


def test_untrained_controls_s12_and_provenance_across_controls():
    tmp = SY.mkdtemp('wp9_untrained_')
    pk, mf = _set(tmp)

    def weaker(sk, arm, s, run):
        run['conn'] = [c - 3.0 for c in run['conn']]
        return run
    u1 = SY.make_pickle(os.path.join(tmp, 'untrained_s1.pkl'), ('marl_static',), SEEDS, mutate=weaker)
    u2 = SY.make_pickle(os.path.join(tmp, 'untrained_s2.pkl'), ('marl_static',), SEEDS, mutate=weaker)
    rep = _analyse(pk, mf, strict=True, untrained=[('untrained_s1', u1), ('untrained_s2', u2)])
    s12 = rep['S12']
    assert set(s12) == {'untrained_s1', 'untrained_s2', 'mean_over_controls'}
    blk = s12['untrained_s1']
    assert blk['summary']['A']['n'] == len(SEEDS) and blk['fingerprint_audit']['A'] == {'bad_passes': 0, 'n': len(SEEDS)}
    pr = blk['paired']['A']
    assert set(pr) == {'ach_trained_minus_untrained', 'ach_untrained_minus_random',
                       'ach_untrained_minus_best_routing', 'components_untrained_minus_trained',
                       'jpu_untrained_minus_trained'}
    assert 'jpu_untrained_minus_trained' not in blk['paired']['B']
    assert abs(pr['ach_trained_minus_untrained']['mean'] - 3.0) < 1e-9
    assert abs(s12['mean_over_controls']['A']['ach_trained_minus_mean_untrained']['mean'] - 3.0) < 1e-9
    assert blk['summary']['A']['M7']['n_recorded'] == len(SEEDS)
    # a control from another tree fails the provenance assert
    u3 = SY.make_pickle(os.path.join(tmp, 'untrained_s3.pkl'), ('marl_static',), SEEDS,
                        prov=SY.provenance(tree_head='deadbee'))
    try:
        _analyse(pk, mf, strict=True, untrained=[('untrained_s3', u3)])
    except AssertionError as e:
        assert 'tree_head differs' in str(e)
    else:
        raise AssertionError('control provenance must be checked')
    # a control missing an instance fails too
    u4 = SY.make_pickle(os.path.join(tmp, 'untrained_s4.pkl'), ('marl_static',), SEEDS[:-1])
    try:
        _analyse(pk, mf, strict=True, untrained=[('untrained_s4', u4)])
    except AssertionError as e:
        assert 'control untrained_s4 scenario A: missing seeds [8]' in str(e)
    else:
        raise AssertionError('control completeness must be checked')


def test_parse_controls_and_s8_embedding():
    assert CA.parse_controls('u1=/x/a.pkl,/y/untrained_s2.pkl') == [('u1', '/x/a.pkl'), ('untrained_s2', '/y/untrained_s2.pkl')]
    assert CA.parse_controls('') == []
    tmp = SY.mkdtemp('wp9_s8_')
    pk, mf = _set(tmp)
    rep = _analyse(pk, mf, strict=True, s8={'identical': True, 'n_compared': 3})
    assert rep['S8'] == {'identical': True, 'n_compared': 3}


def test_print_report_runs_on_a_full_report():
    tmp = SY.mkdtemp('wp9_print_')
    pk, mf = _set(tmp)
    rep = _analyse(pk, mf, strict=False, s8={'identical': False})
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        CA.print_report(rep)
    out = buf.getvalue()
    for needle in ('PRIMARY: arm marl_static', 'S9 STABILITY FAMILY', 'S10 STABILITY', 'HARD ASSERTS: NOT enforced',
                   'S8 hash invariance: NOT identical'):
        assert needle in out, needle
    assert not any(math.isnan(v) for v in (rep['primary'][k]['mean'] for k in rep['primary']))


def test_provenance_fields_must_be_present_not_merely_identical():
    tmp = SY.mkdtemp('wp9_provmissing_')
    prov = SY.provenance()
    del prov['tree_sha256']                 # absent in every record: str(None) would be "identical"
    prov['schema_id'] = None
    a = SY.make_pickle(os.path.join(tmp, 'marl.pkl'), MARL_ARMS, SEEDS, prov=dict(prov))
    b = SY.make_pickle(os.path.join(tmp, 'base.pkl'), BASE_ARMS, SEEDS, prov=dict(prov))
    m = SY.make_manifest(os.path.join(tmp, 'manifest.json'), SEEDS)
    try:
        _analyse([a, b], [m], strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('a provenance field missing from every pickle must fail strict mode')
    assert "tree_sha256 missing (None/empty) in ['marl.pkl', 'base.pkl']" in msg, msg
    assert "schema_id missing (None/empty) in ['marl.pkl', 'base.pkl']" in msg, msg
    assert 'differs across pickles' not in msg
    rep = _analyse([a, b], [m], strict=False)
    c = rep['asserts']['checks']['provenance']
    assert c['missing_fields'] == {'tree_sha256': ['marl.pkl', 'base.pkl'], 'schema_id': ['marl.pkl', 'base.pkl']}
    assert c['fields']['tree_sha256'] == ['None'] and rep['asserts']['n_failures'] == 2
    # an empty string is missing too, and a field missing in one pickle is both missing and different
    prov2 = SY.provenance(schema_id='')
    a2 = SY.make_pickle(os.path.join(tmp, 'marl2.pkl'), MARL_ARMS, SEEDS, prov=prov2)
    b2 = SY.make_pickle(os.path.join(tmp, 'base2.pkl'), BASE_ARMS, SEEDS)
    rep = _analyse([a2, b2], [m], strict=False)
    fails = rep['asserts']['failures']
    assert any("schema_id missing (None/empty) in ['marl2.pkl']" in f for f in fails), fails
    assert any('schema_id differs across pickles' in f for f in fails), fails


def test_numpy_scalars_and_arrays_are_checked_never_dropped():
    if np is None:
        return
    tmp = SY.mkdtemp('wp9_numpy_')

    def mut_a(sk, arm, s, run):
        if arm == 'random' and s == 5 and sk == 'rescue_multi':
            run['stale_edges'] = [np.int64(0)] * SY.T_TEST      # a list of numpy scalars
            run['stale_edges'][7] = np.int64(5)
        if arm == 'xldet' and s == 3 and sk == 'recovery_multi':
            c = np.array(run['conn'], dtype=float)              # an ndarray with one NaN
            c[4] = np.nan
            run['conn'] = c
        if arm == 'marl_static' and s == 1:
            bt = list(run['overhead_by_type']['bfd'])             # NaN inside a {name: series} member
            bt[6] = float('nan')
            run['overhead_by_type']['bfd'] = bt
            run['relay_removed_by_reason']['repoint'] = np.array(run['relay_removed_by_reason']['repoint'], dtype=float)
        if arm == 'marl_static' and s == 2 and sk == 'recovery_multi':
            run['weird'] = [1, 'x', None, 2]                     # a mixed series: non-numeric entries are a violation
            run['relay_removed'] = np.array(run['relay_removed'])
            run['n_radio_sites'] = np.int64(SY.N_SITES)
        return run

    pk, mf = _set(tmp, mut_a)
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('numpy-typed violations must stop strict mode')
    for needle in ('B random seed 5: stale_edges sums to 5', 'A xldet seed 3: series conn has 1 NaN/None',
                   'A marl_static seed 1: series overhead_by_type/bfd has 1 NaN/None',
                   'B marl_static seed 1: series overhead_by_type/bfd has 1 NaN/None',
                   'A marl_static seed 2: series weird has 1 non-numeric', 'series weird has 1 NaN/None',
                   'A marl_static seed 2: series weird has 4 entries, expected 40 (run length)'):
        assert needle in msg, (needle, msg)
    rep = _analyse(pk, mf, strict=False)
    chk = rep['asserts']['checks']
    assert chk['stale_edges']['violations'] == [['B', 'random', 5, 'stale_edges', 5]]
    assert ['A', 'xldet', 3, 'conn', 1] in chk['nan']['violations']
    assert chk['nan']['non_numeric'] == [['A', 'marl_static', 2, 'weird', 1]]
    # the 4-entry 'weird' series is a run-length violation as well (D6)
    assert chk['run_length']['violations'] == [['A', 'marl_static', 2, 'weird', 4]]
    assert rep['asserts']['n_failures'] == 7, rep['asserts']['failures']
    json.dumps(rep)                                              # numpy never leaks into the report
    # the metrics on the numpy-typed run equal those on the plain run
    plain = SY.make_run(SY.T_TEST, 2, 'marl_static', 'A')
    with SY.short_run_patch():
        a = CA.stability_metrics(mut_a('recovery_multi', 'marl_static', 2, SY.make_run(SY.T_TEST, 2, 'marl_static', 'A')),
                                 SY.T_SEV_TEST['A'], 1, SY.T_TEST)
        b = CA.stability_metrics(plain, SY.T_SEV_TEST['A'], 1, SY.T_TEST)
    assert a == b and a['M7'] > 0
    # M7 on the numpy-scalar series is the true count, not 0
    rr = [np.int64(x) for x in plain['relay_removed']]
    assert CA.m7_teardowns_per_1000({'relay_removed': rr}, SY.T_SEV_TEST['A']) == b['M7']


def test_postcards_per_site_when_recorded_and_aggregate_labelled_otherwise():
    tmp = SY.mkdtemp('wp9_postcards_')

    def mut_a(sk, arm, s, run):
        if arm == 'xldet' and sk == 'recovery_multi':
            run['postcards_by_site'] = {'s1': 5, 's2': (1 if s == 2 else 4), 's3': 6}   # seed 2: one site sent once
        if arm == 'random' and sk == 'recovery_multi':
            run['postcards_min_per_site'] = 0 if s == 3 else 3                          # seed 3: a silent site
        return run

    pk, mf = _set(tmp, mut_a)
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('a site with <= 1 postcard must fail the per-site check')
    assert 'A xldet seed 2: a site sent 1 postcards after the event (per-site check' in msg, msg
    assert 'A random seed 3: a site sent 0 postcards after the event (per-site check' in msg, msg
    rep = _analyse(pk, mf, strict=False)
    post = rep['asserts']['checks']['postcards']
    assert post['per_site_violations'] == [['A', 'xldet', 2, 1, SY.N_SITES], ['A', 'random', 3, 0, SY.N_SITES]]
    assert post['violations'] == []                        # the aggregate mean passes everywhere
    assert post['mode'] == 'mixed' and post['min_per_site'] == 0 and post['runs_with_per_site'] == 2 * len(SEEDS)
    assert post['min_mean_per_site'] > 1 and rep['asserts']['n_failures'] == 2
    # per site everywhere: mode 'per-site'
    def everywhere(sk, arm, s, run):
        if arm in SY.MARL_PATH:
            run['postcards_by_site'] = {'s%d' % i: 4 for i in range(SY.N_SITES)}
        return run
    pk, mf = _set(SY.mkdtemp('wp9_postcards2_'), everywhere)
    rep = _analyse(pk, mf, strict=True)
    assert rep['asserts']['checks']['postcards']['mode'] == 'per-site' and rep['asserts']['checks']['postcards']['min_per_site'] == 4
    assert CA._postcards_min_per_site({'postcards_by_site': {}}) is None
    assert CA._postcards_min_per_site({'postcards_min_per_site': None}) is None


def test_postcards_per_site_must_cover_every_site():
    """D2: a site that never sent is simply ABSENT from a postcards_by_site the
    engine fills from senders, so the minimum over the entries present passes
    -- the very defect the per-site check exists for.  The check must compare
    the number of entries with the run's site count."""
    tmp = SY.mkdtemp('wp9_postcards3_')

    def mut_a(sk, arm, s, run):
        if arm in SY.MARL_PATH:
            run['postcards_by_site'] = {'s%d' % i: 4 + i for i in range(SY.N_SITES)}
        if arm == 'marl_static' and s == 1 and sk == 'recovery_multi':
            run['postcards_by_site'] = {'s1': 5, 's2': 7}        # one of three sites never sent: absent
        if arm == 'random' and s == 2 and sk == 'recovery_multi':
            run.pop('postcards_by_site')                          # the scalar form: no completeness claim
            run['postcards_min_per_site'] = 3
        return run

    pk, mf = _set(tmp, mut_a)
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('an incomplete postcards_by_site must fail the per-site check')
    assert ('A marl_static seed 1: postcards_by_site holds 2 site(s) but the run records 3 site(s)'
            in msg), msg
    rep = _analyse(pk, mf, strict=False)
    post = rep['asserts']['checks']['postcards']
    assert post['incomplete_per_site'] == [['A', 'marl_static', 1, 2, SY.N_SITES]]
    assert post['per_site_violations'] == []                      # min over the entries present is 5: the old check passed
    assert rep['asserts']['n_failures'] == 1, rep['asserts']['failures']
    # the scalar postcards_min_per_site cannot be checked for completeness and is not flagged
    assert post['runs_with_per_site'] == 2 * len(SEEDS) * len(MARL_ARMS)
    assert post['runs_with_site_coverage'] == post['runs_with_per_site'] - 1
    # the helpers themselves (min, entries recorded, badly recorded entries)
    assert CA._postcards_per_site({'postcards_by_site': {'a': 2, 'b': 3}}) == (2, 2, [])
    assert CA._postcards_per_site({'postcards_min_per_site': 3}) == (3, None, [])
    assert CA._postcards_per_site({'postcards_by_site': {}}) == (None, None, [])
    assert CA._postcards_min_per_site({'postcards_by_site': {'a': 9}}) == 9
    # the agent-site count wins over the radio-site count when the run records it
    assert CA._site_count({'n_agent_sites': 7, 'n_radio_sites': 3}) == 7
    assert CA._site_count({'n_radio_sites': 3}) == 3 and CA._site_count({}) is None


def test_postcards_per_site_entry_that_is_not_a_count_is_a_labelled_failure():
    """D3: a neighbouring recording error decided the fate of the per-site
    check.  One None in postcards_by_site made min() raise an unlabelled
    TypeError from inside check_invariants -- aborting the whole confirmatory
    analysis instead of producing the labelled per-site failure the assert
    exists for -- and an all-None dict made min() return None, which dropped
    the run from the per-site population and degraded the mode to 'aggregate'
    with no message at all."""
    # the helper: the minimum is taken over the real numbers, the rest is named
    assert CA._postcards_per_site({'postcards_by_site': {'s1': 5, 's2': None}}) == (5, 2, [['s2', 'None']])
    allnone = CA._postcards_per_site({'postcards_by_site': {'s1': None, 's2': None, 's3': None}})
    assert allnone[0] is None and allnone[1] == 3
    assert sorted(allnone[2]) == [['s1', 'None'], ['s2', 'None'], ['s3', 'None']]
    assert CA._postcards_per_site({'postcards_by_site': {'s1': None}}) == (None, 1, [['s1', 'None']])
    mixed = CA._postcards_per_site({'postcards_by_site': {'s1': 4, 's2': 'many', 's3': float('nan')}})
    assert mixed[0] == 4 and mixed[1] == 3 and sorted(x[0] for x in mixed[2]) == ['s2', 's3']
    assert CA._postcards_min_per_site({'postcards_by_site': {'s1': 5, 's2': None}}) == 5

    tmp = SY.mkdtemp('wp9_postcards4_')

    def mut_a(sk, arm, s, run):
        if arm in SY.MARL_PATH:
            run['postcards_by_site'] = {'s%d' % i: 4 + i for i in range(SY.N_SITES)}
        if arm == 'marl_static' and s == 1 and sk == 'recovery_multi':
            run['postcards_by_site'] = {'s0': 5, 's1': None, 's2': 7}          # one entry not a count
        if arm == 'xldet' and s == 2 and sk == 'recovery_multi':
            run['postcards_by_site'] = {'s%d' % i: None for i in range(SY.N_SITES)}   # every entry
        return run

    pk, mf = _set(tmp, mut_a)
    # the analysis must RUN (no TypeError escaping check_invariants) ...
    rep = _analyse(pk, mf, strict=False)
    post = rep['asserts']['checks']['postcards']
    assert post['per_site_non_numeric'] == [['A', 'marl_static', 1, 's1', 'None'],
                                            ['A', 'xldet', 2, 's0', 'None'],
                                            ['A', 'xldet', 2, 's1', 'None'],
                                            ['A', 'xldet', 2, 's2', 'None']], post['per_site_non_numeric']
    assert rep['asserts']['n_failures'] == 4, rep['asserts']['failures']
    # ... and the all-None run is no longer silent: it fails with a message
    for needle in ("A marl_static seed 1: postcards_by_site site 's1' holds None, expected a count",
                   "A xldet seed 2: postcards_by_site site 's0' holds None, expected a count"):
        assert needle in rep['asserts']['failures'], rep['asserts']['failures']
    # the surviving entries of the partly bad dict still carry the check
    assert post['min_per_site'] == 4 and post['per_site_violations'] == []
    json.dumps(rep)
    # ... and strict mode stops on it, labelled, instead of raising TypeError
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        assert "postcards_by_site site 's1' holds None, expected a count" in str(e), str(e)
    else:
        raise AssertionError('a postcards_by_site entry that is not a count must stop strict mode')


def test_every_numeric_series_has_the_run_length():
    """D6: the run-length assert covered `conn` only, so a short fragments /
    fragments_raw / event series silently shortened the M1-M7/M10 windows."""
    tmp = SY.mkdtemp('wp9_length_')

    def mut_a(sk, arm, s, run):
        if arm == 'marl_static' and s == 1 and sk == 'recovery_multi':
            run['fragments'] = run['fragments'][:30]              # conn is still full length
            run['fragments_raw'] = run['fragments_raw'][:30]
        if arm == 'xldet' and s == 2 and sk == 'rescue_multi':
            run['relay_removed'] = run['relay_removed'][:-1]      # an event series
        if arm == 'random' and s == 3 and sk == 'recovery_multi':
            run['overhead_by_type']['bfd'] = run['overhead_by_type']['bfd'][:7]   # a {name: series} member
        return run

    pk, mf = _set(tmp, mut_a)
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('a short series other than conn must stop strict mode')
    for needle in ('A marl_static seed 1: series fragments has 30 entries, expected 40 (run length)',
                   'A marl_static seed 1: series fragments_raw has 30 entries',
                   'B xldet seed 2: series relay_removed has 39 entries',
                   'A random seed 3: series overhead_by_type/bfd has 7 entries'):
        assert needle in msg, (needle, msg)
    rep = _analyse(pk, mf, strict=False)
    v = rep['asserts']['checks']['run_length']
    assert v['expected'] == SY.T_TEST
    # D4: the block recorded no coverage, so {'violations': [], 'expected': N}
    # was indistinguishable from "no series was ever examined" -- the very shape
    # this round exists to remove, in the check whose whole result on the frozen
    # data is an empty violations list.  n_series_checked / runs_checked back it.
    n_runs = 2 * len(SEEDS) * (len(MARL_ARMS) + len(BASE_ARMS))
    assert v['runs_checked'] == n_runs
    assert v['n_series_checked'] == N_SERIES_PER_RUN * n_runs, v['n_series_checked']
    assert sorted(v['violations']) == sorted([['A', 'marl_static', 1, 'fragments', 30],
                                              ['A', 'marl_static', 1, 'fragments_raw', 30],
                                              ['B', 'xldet', 2, 'relay_removed', 39],
                                              ['A', 'random', 3, 'overhead_by_type/bfd', 7]])
    assert rep['asserts']['n_failures'] == 4, rep['asserts']['failures']
    # the metric on the short window is what the assert exists to refuse: it is
    # still computed (--no-assert), over [ts, 30) rather than [ts, 40)
    r = rep['S10']['A']['marl_static']['per_seed']['1']
    assert r['M4'] is not None
    # the counter must be the real per-run series count, not a constant: it is
    # what makes a future breakage of _numeric_list visible (an unchecked key
    # drops the count) instead of silent (violations stays empty either way)
    clean = SY.make_run(SY.T_TEST, 1, 'marl_static', 'A')
    counted = [k for k, val in clean.items()
               if k not in CA.DICT_SERIES_KEYS and CA._numeric_list(val) is not None]
    members = [name for k in CA.DICT_SERIES_KEYS for name in (clean.get(k) or {})
               if CA._numeric_list(clean[k][name]) is not None]
    assert len(counted) + len(members) == N_SERIES_PER_RUN, (sorted(counted), sorted(members))
    assert 'relay_lifetimes' not in counted and 'n_radio_sites' not in counted
    assert 'fragments' in counted and 'bfd' in members
    # a run with no series at all contributes nothing to either counter
    rep2 = _analyse(*_set(SY.mkdtemp('wp9_length2_')), strict=False)
    assert rep2['asserts']['checks']['run_length']['runs_checked'] == n_runs
    assert rep2['asserts']['checks']['run_length']['violations'] == []
    assert rep2['asserts']['checks']['run_length']['n_series_checked'] == N_SERIES_PER_RUN * n_runs


def test_bridge_lifetime_records_without_a_lifetime_are_a_violation():
    """D7: a bridge record with no start tick was dropped silently (n did not
    count it) and `removed == T` (the engine writing T instead of None) was
    read as a completed lifetime.  D7 residual: the START tick was not
    validated at all, so `up == T` / `up > T` -- the same off-by-one, on the
    other field -- produced a zero or NEGATIVE censored lifetime that M11
    counted in n and fed to the medians, with n_invalid 0 and
    checks.lifetimes.violations empty."""
    tmp = SY.mkdtemp('wp9_lifetimes_')

    def mut_a(sk, arm, s, run):
        if arm == 'marl_static' and s == 1 and sk == 'recovery_multi':
            run['relay_lifetimes'] = list(run['relay_lifetimes']) + [
                {'link_id': 'TR_NO_START', 'created': None, 'up': None, 'removed': 9,
                 'cause': 'mode_change', 'was_bridge': True, 'load': 0.0},
                {'link_id': 'TR_REMOVED_AT_T', 'created': 6, 'up': 7, 'removed': SY.T_TEST,
                 'cause': 'mode_change', 'was_bridge': True, 'load': 0.0}]
        if arm == 'marl_static' and s == 3 and sk == 'recovery_multi':
            run['relay_lifetimes'] = list(run['relay_lifetimes']) + [
                {'link_id': 'TR_UP_AT_T', 'created': 6, 'up': SY.T_TEST, 'removed': None,
                 'cause': 'mode_change', 'was_bridge': True, 'load': 0.0},
                {'link_id': 'TR_UP_PAST_T', 'created': 6, 'up': SY.T_TEST + 5, 'removed': None,
                 'cause': 'mode_change', 'was_bridge': True, 'load': 0.0},
                {'link_id': 'TR_UP_NEGATIVE', 'created': -10, 'up': -10, 'removed': None,
                 'cause': 'mode_change', 'was_bridge': True, 'load': 0.0}]
        return run

    pk, mf = _set(tmp, mut_a)
    try:
        _analyse(pk, mf, strict=True)
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError('a bridge record without a lifetime must stop strict mode')
    assert "relay_lifetimes bridge record 'TR_NO_START': no start tick" in msg, msg
    assert "relay_lifetimes bridge record 'TR_REMOVED_AT_T': removed 40 >= T=40" in msg, msg
    for link, tick in (('TR_UP_AT_T', 40), ('TR_UP_PAST_T', 45), ('TR_UP_NEGATIVE', -10)):
        assert ("relay_lifetimes bridge record %r: start tick %d outside [0, T=40)" % (link, tick)) in msg, msg
    rep = _analyse(pk, mf, strict=False)
    lt = rep['asserts']['checks']['lifetimes']
    assert lt['runs_with_key'] == 2 * len(SEEDS) * (len(MARL_ARMS) + len(BASE_ARMS))
    assert len(lt['violations']) == 5
    assert [x[:4] for x in lt['violations']][0] == ['A', 'marl_static', 1, 'TR_NO_START']
    assert sorted(x[3] for x in lt['violations']) == ['TR_NO_START', 'TR_REMOVED_AT_T', 'TR_UP_AT_T',
                                                      'TR_UP_NEGATIVE', 'TR_UP_PAST_T']
    assert rep['asserts']['n_failures'] == 5, rep['asserts']['failures']
    # no bad record is counted by M11, and the report says how many were refused
    per = rep['S10']['A']['marl_static']['per_seed']
    assert per['1']['M11']['n'] == per['2']['M11']['n'] == per['3']['M11']['n'] == 2
    assert per['1']['M11']['n_invalid'] == 2 and per['3']['M11']['n_invalid'] == 3
    # unchanged by the bad records: no 0, no negative and no over-long censored
    # observation reaches the medians or n_censored
    for seed in ('1', '3'):
        assert per[seed]['M11']['median'] == per['2']['M11']['median']
        assert per[seed]['M11']['median_censored_at_T'] == per['2']['M11']['median_censored_at_T']
        assert per[seed]['M11']['n_censored'] == per['2']['M11']['n_censored']
    assert rep['S10']['A']['marl_static']['M11']['n_invalid_records'] == 5


def test_s11_by_type_absent_in_one_run_is_not_recorded_not_zero():
    """D8: a message type absent from one run of an arm contributed sum([]) = 0
    to that arm's per-type statistics instead of 'not recorded'."""
    tmp = SY.mkdtemp('wp9_s11_')

    def mut_a(sk, arm, s, run):
        if arm == 'marl_static' and s in (1, 2) and sk == 'recovery_multi':
            run['overhead_by_type'].pop('lsa')         # this run never recorded the type
        return run

    rep = _analyse(*_set(tmp, mut_a), strict=True)
    blk = rep['S11']['A']['marl_static']['by_type_post_event']
    assert set(blk) == {'bfd', 'hello', 'lsa', 'postcard'}          # the union over the seeds  # wp7-scan-ok: WP5 DCC message type, not an action head
    assert blk['lsa']['n'] == len(SEEDS) and blk['lsa']['n_recorded'] == len(SEEDS) - 2
    assert blk['lsa']['min'] > 0                                   # no zero from the two runs without the type
    assert blk['bfd']['n_recorded'] == len(SEEDS)
    assert CA._by_type_post_event({'overhead_by_type': {'bfd': [1.0, 2.0]}}, 'lsa', 0) is None
    assert CA._by_type_post_event({'overhead_by_type': {'bfd': [1.0, 2.0]}}, 'bfd', 1) == 2.0
    assert CA._by_type_post_event({}, 'bfd', 0) is None
    # D8 residual: only the absent KEY returned None.  A key whose post-event
    # slice holds no recorded value -- an empty list, a series shorter than
    # t_sev, a slice of NaN -- returned sum([]) == 0 and put that zero into the
    # per-type median/mean exactly as before the fix.  The run-length assert
    # covers the truncated case in strict mode only, and the frozen pickles are
    # analysed with --no-assert, where the 0 still entered S11.
    assert CA._by_type_post_event({'overhead_by_type': {'lsa': []}}, 'lsa', 0) is None
    assert CA._by_type_post_event({'overhead_by_type': {'lsa': [1.0, 2.0]}}, 'lsa', 5) is None
    assert CA._by_type_post_event({'overhead_by_type': {'lsa': [1.0, 2.0]}}, 'lsa', 2) is None
    assert CA._by_type_post_event({'overhead_by_type': {'lsa': [1.0, float('nan')]}}, 'lsa', 1) is None
    assert CA._by_type_post_event({'overhead_by_type': {'lsa': [1.0, 2.0]}}, 'lsa', 1) == 2.0
    # and through the analysis: a truncated per-type series contributes
    # "not recorded", never a 0 that drags the per-type minimum down
    def mut_b(sk, arm, s, run):
        if arm == 'marl_static' and s == 3 and sk == 'recovery_multi':
            run['overhead_by_type']['hello'] = []                       # recorded, but nothing in it
        if arm == 'marl_static' and s == 4 and sk == 'recovery_multi':
            run['overhead_by_type']['hello'] = run['overhead_by_type']['hello'][:2]   # shorter than t_sev
        return run

    rep = _analyse(*_set(SY.mkdtemp('wp9_s11b_'), mut_b), strict=False)
    blk = rep['S11']['A']['marl_static']['by_type_post_event']
    assert blk['hello']['n'] == len(SEEDS) and blk['hello']['n_recorded'] == len(SEEDS) - 2
    assert blk['hello']['min'] > 0, blk['hello']


def test_marl_freeze_must_be_complete_once_present():
    tmp = SY.mkdtemp('wp9_freeze_')

    def partial(sk, arm, s, run):
        return None if (arm == 'marl_freeze' and s == 8 and sk == 'recovery_multi') else run

    a = SY.make_pickle(os.path.join(tmp, 'marl.pkl'), MARL_ARMS, SEEDS)
    f = SY.make_pickle(os.path.join(tmp, 'freeze.pkl'), ('marl_freeze',), SEEDS, mutate=partial)
    b = SY.make_pickle(os.path.join(tmp, 'base.pkl'), BASE_ARMS, SEEDS)
    m = SY.make_manifest(os.path.join(tmp, 'manifest.json'), SEEDS)
    try:
        _analyse([a, f, b], [m], strict=True)
    except AssertionError as e:
        assert 'arm marl_freeze scenario A: missing seeds [8]' in str(e), str(e)
    else:
        raise AssertionError('a partial adapt-then-freeze set must fail strict mode')
    rep = _analyse([a, f, b], [m], strict=False)
    assert rep['asserts']['checks']['arms_and_seeds']['missing'] == {'A/marl_freeze': [8]}
    assert rep['secondary']['S7_A_ach_marl_static_minus_marl_freeze']['n'] == len(SEEDS) - 1   # the intersection, reported
    assert rep['secondary']['S7_B_ach_marl_static_minus_marl_freeze']['n'] == len(SEEDS)
    # complete: passes, S7 over every seed
    f2 = SY.make_pickle(os.path.join(tmp, 'freeze2.pkl'), ('marl_freeze',), SEEDS)
    rep = _analyse([a, f2, b], [m], strict=True)
    assert rep['secondary']['S7_A_ach_marl_static_minus_marl_freeze']['n'] == len(SEEDS)
    # absent altogether: no S7, no assert (it is optional)
    rep = _analyse([a, b], [m], strict=True)
    assert not any(k.startswith('S7') for k in rep['secondary'])


def test_checkpoint_sha256_identical_across_pickles_carrying_the_tested_arm():
    tmp = SY.mkdtemp('wp9_ckpt_')
    m = SY.make_manifest(os.path.join(tmp, 'manifest.json'), SEEDS)
    x = SY.make_pickle(os.path.join(tmp, 'xr.pkl'), ('xldet', 'random'), SEEDS)
    b = SY.make_pickle(os.path.join(tmp, 'base.pkl'), BASE_ARMS, SEEDS,
                       prov=SY.provenance(checkpoint_sha256=['b' * 64]))      # baselines: not compared
    lo = SY.make_pickle(os.path.join(tmp, 'marl_1_4.pkl'), ('marl_static',), SEEDS[:4])
    hi_other = SY.make_pickle(os.path.join(tmp, 'marl_5_8.pkl'), ('marl_static',), SEEDS[4:],
                              prov=SY.provenance(checkpoint_sha256=['d' * 64]))
    try:
        _analyse([lo, hi_other, x, b], [m], strict=True)
    except AssertionError as e:
        assert 'checkpoint_sha256 differs across the pickles carrying arm marl_static' in str(e), str(e)
    else:
        raise AssertionError('two checkpoints across the MARL batches must fail')
    hi_same = SY.make_pickle(os.path.join(tmp, 'marl_5_8b.pkl'), ('marl_static',), SEEDS[4:])
    rep = _analyse([lo, hi_same, x, b], [m], strict=True)
    assert rep['asserts']['checks']['provenance']['checkpoint_sha256'] == {
        'marl_1_4.pkl': ['c' * 64], 'marl_5_8b.pkl': ['c' * 64]}
    # a MARL-arm pickle without the field fails; a control with another checkpoint does not
    p = SY.provenance()
    del p['checkpoint_sha256']
    hi_none = SY.make_pickle(os.path.join(tmp, 'marl_5_8c.pkl'), ('marl_static',), SEEDS[4:], prov=p)
    try:
        _analyse([lo, hi_none, x, b], [m], strict=True)
    except AssertionError as e:
        assert "checkpoint_sha256 missing in ['marl_5_8c.pkl']" in str(e), str(e)
    else:
        raise AssertionError('a MARL-arm pickle without checkpoint_sha256 must fail')
    u = SY.make_pickle(os.path.join(tmp, 'untrained.pkl'), ('marl_static',), SEEDS,
                       prov=SY.provenance(checkpoint_sha256=['e' * 64]))
    rep = _analyse([lo, hi_same, x, b], [m], strict=True, untrained=[('untrained_s1', u)])
    assert rep['asserts']['n_failures'] == 0 and 'untrained_s1' in rep['S12']
