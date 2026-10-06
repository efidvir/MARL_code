"""WP9: dump_results_json.py and extract_timelines.py pass the WP4 series
through, import the metric helpers (no duplicated formulas) and never fill in
a series that is not recorded."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import confirmatory_analysis as CA  # noqa: E402
import dump_results_json as DJ  # noqa: E402
import extract_timelines as ET  # noqa: E402
import test_wp9_synth as SY  # noqa: E402

np = SY.np


def test_dump_summarize_passes_new_keys_through():
    run = SY.make_run(SY.T_TEST, 1, 'marl_static', 'A')
    d = DJ.summarize(run)
    for key in ('conn', 'fragments', 'fragments_raw', 'relay_count', 'relay_bridges', 'postcards', 'overhead',
                'energy_ctrl', 'stale_edges') + CA.EVENT_SERIES:
        assert 'series' in d[key] and len(d[key]['series']) == SY.T_TEST, key
        assert d[key]['total'] == sum(run[key]) or abs(d[key]['total'] - sum(run[key])) < 1e-6
    assert d['relay_lifetimes']['n'] == 3
    recs = d['relay_lifetimes']['records']
    assert recs[1] == {'link_id': 'TR_2_3_2', 'created': 13, 'up': 16, 'removed': None, 'cause': None,
                       'was_bridge': True, 'load': 0.1}                   # tuple normalised through CA
    bt = d['overhead_by_type']['by_key']
    assert set(bt) == {'bfd', 'hello', 'lsa', 'postcard'} and 'series' in bt['bfd'] and bt['bfd']['total'] == 60.0 * 35  # wp7-scan-ok: WP5 DCC message type, not an action head
    assert set(d['relay_removed_by_reason']['by_key']) == {'mode_change', 'repoint'}
    assert 'series' in d['relay_teardown_holds']['by_key']['grace']
    assert d['n_radio_sites'] == {'scalar': SY.N_SITES} and d['static_fp_start'] == {'text': 'fp-marl_static'}
    # a series that is only NaN/None is reported empty, as before
    assert DJ.summarize({'x': [None, float('nan')]}) == {'x': {'empty': 2}}


def test_dump_stability_block_uses_the_helpers():
    run = SY.make_run(SY.T_TEST, 1, 'marl_static', 'A')
    inst = {'attainable_optimum': 2}
    with SY.patched(T_SEV={'A': 5, 'B': 0}):
        blk = DJ.stability_block(run, 'recovery_multi', inst)
        assert blk['severance_tick'] == 5 and blk['attainable_optimum'] == 2
        assert blk['M1'] == CA.m1_fragment_changes(run, 5) and blk['M2'] == CA.m2_policy_splits(run, 5)
        assert blk['M7'] == CA.m7_teardowns_per_1000(run, 5) and blk['M11'] == CA.m11_median_bridge_lifetime(run, SY.T_TEST)
        assert blk['M4'] == CA.m4_fraction_at_optimum(run, 5, 2)
        blk = DJ.stability_block(run, 'recovery_multi', None)       # no manifest: optimum-based metrics null
        assert blk['M3'] is None and blk['M4'] is None and blk['M6'] is None and blk['M1'] is not None
        legacy = SY.make_run(SY.T_TEST, 1, 'ospf', 'B', events=False)
        blk = DJ.stability_block(legacy, 'rescue_multi', inst)
        assert blk['severance_tick'] == 0 and blk['M7'] is None and blk['M9'] is None and blk['M1'] == 0


def test_dump_build_first_source_wins_and_records_provenance():
    tmp = SY.mkdtemp('wp9_dump_')
    seeds = [1, 2]
    a = SY.make_pickle(os.path.join(tmp, 'a.pkl'), ('marl_static', 'xldet'), seeds)

    def other(sk, arm, s, run):
        run['conn'] = [0.0] * SY.T_TEST
        return run
    b = SY.make_pickle(os.path.join(tmp, 'b.pkl'), ('xldet', 'ospf'), seeds, mutate=other, prov=None)
    m = SY.make_manifest(os.path.join(tmp, 'm.json'), seeds)
    with SY.patched(T_SEV={'A': 5, 'B': 0}):
        out = DJ.build([('cf', [a, b])], DJ.load_manifests([m]))
    tag = out['cf']
    assert tag['sources'][a] == ['recovery/marl_static:2', 'recovery/xldet:2', 'rescue/marl_static:2', 'rescue/xldet:2']
    assert tag['sources'][b] == ['recovery/ospf:2', 'rescue/ospf:2']
    assert tag['scenarios']['recovery_multi']['xldet']['1']['conn']['steady'] > 0      # first source won
    assert tag['provenance'][a]['tree_head'] == 'abc1234' and tag['provenance'][b] is None
    st_ = tag['scenarios']['recovery_multi']['marl_static']['1']['stability']
    assert st_['attainable_optimum'] == 2 and st_['M2'] > 0 and st_['M7'] is not None
    json.dumps(out)                                              # serialisable


def test_timelines_bin_sums_for_event_series_and_list_not_recorded():
    tmp = SY.mkdtemp('wp9_tl_')
    seeds = [1, 2, 3]

    def drop(sk, arm, s, run):
        if arm == 'ospf':
            run.pop('relay_mode_flips')
            run.pop('relay_lifetimes', None)
        return run
    a = SY.make_pickle(os.path.join(tmp, 'a.pkl'), ('marl_static', 'ospf'), seeds, mutate=drop)
    out = ET.extract(ET.load_runs([a]))
    assert out['bin_ticks'] == 10 and out['binning']['relay_removed'] == 'sum' and out['binning']['conn'] == 'mean'
    sc = out['scenarios']['A']
    assert sc['seeds'] == seeds and sc['n_bins'] == SY.T_TEST // 10
    marl = sc['arms']['marl_static']
    assert marl['not_recorded'] == []
    run = SY.make_run(SY.T_TEST, 1, 'marl_static', 'A')
    # every instance has the same event series here, so mean == median == the per-bin sum
    for b in range(sc['n_bins']):
        assert marl['relay_removed']['mean'][b] == sum(run['relay_removed'][b * 10:(b + 1) * 10])
        assert marl['relay_removed']['n'][b] == len(seeds)
    assert abs(marl['fragments']['mean'][1] - sum(run['fragments'][10:20]) / 10) < 1e-9
    assert sc['arms']['ospf']['not_recorded'] == ['relay_mode_flips']
    assert 'relay_mode_flips' not in sc['arms']['ospf'] and 'relay_removed' in sc['arms']['ospf']
    assert set(marl) - {'not_recorded'} == set(ET.KEYS) | set(ET.SUM_KEYS)
    # legacy pickle: only the state series
    b_ = SY.make_pickle(os.path.join(tmp, 'legacy.pkl'), ('xldet',), seeds, events=False, prov=None)
    sc = ET.extract(ET.load_runs([b_]))['scenarios']['B']
    assert sorted(sc['arms']['xldet']['not_recorded']) == sorted(ET.SUM_KEYS)
    assert ET.binned([1, None, float('nan'), 2] + [None] * 6, 1, 'sum') == [3]
    assert ET.binned([None] * 10, 1, 'mean') == [None]


def test_timelines_arm_order_and_unknown_arms():
    tmp = SY.mkdtemp('wp9_tl2_')
    a = SY.make_pickle(os.path.join(tmp, 'a.pkl'), ('zzz_new_arm', 'xldet', 'marl_static'), [1])
    sc = ET.extract(ET.load_runs([a]))['scenarios']['A']
    assert list(sc['arms']) == ['marl_static', 'xldet', 'zzz_new_arm']
    assert sc['unknown_arms'] == ['zzz_new_arm'] and sc['seeds_dropped'] == {}


def test_timelines_unknown_or_partial_arm_never_shrinks_the_seed_set():
    tmp = SY.mkdtemp('wp9_tl3_')
    seeds = [1, 2, 3]

    def drop(sk, arm, s, run):
        return None if (arm == 'zzz_new_arm' and s == 3) else run
    a = SY.make_pickle(os.path.join(tmp, 'a.pkl'), ('marl_static', 'xldet', 'zzz_new_arm'), seeds, mutate=drop)
    sc = ET.extract(ET.load_runs([a]))['scenarios']['A']
    assert sc['seeds'] == seeds and sc['unknown_arms'] == ['zzz_new_arm'] and sc['seeds_dropped'] == {}
    assert sc['arms']['marl_static']['conn']['n'][0] == 3 and sc['arms']['xldet']['conn']['n'][0] == 3
    assert sc['arms']['zzz_new_arm']['conn']['n'][0] == 2               # over its own seeds within the set
    # a known arm with a seed the others lack: the intersection drops it and records that
    def extra(sk, arm, s, run):
        return None if (arm == 'xldet' and s == 3) else run
    b = SY.make_pickle(os.path.join(tmp, 'b.pkl'), ('marl_static', 'xldet'), seeds, mutate=extra)
    sc = ET.extract(ET.load_runs([b]))['scenarios']['A']
    assert sc['seeds'] == [1, 2] and sc['seeds_dropped'] == {'marl_static': [3]}
    assert sc['arms']['marl_static']['conn']['n'][0] == 2
    # only unknown arms: they form the basis themselves
    c = SY.make_pickle(os.path.join(tmp, 'c.pkl'), ('zzz_new_arm',), seeds)
    sc = ET.extract(ET.load_runs([c]))['scenarios']['A']
    assert sc['seeds'] == seeds and sc['unknown_arms'] == ['zzz_new_arm']


def _numpy_run(seed=1, arm='marl_static', k='A'):
    """A contract-compliant run written with numpy types throughout: arrays,
    numpy-scalar lists, an all-numpy {site: count} dict and numpy lifetime
    ticks (TRACKB_HANDOFF section 1: all of these are accepted)."""
    run = SY.make_run(SY.T_TEST, seed, arm, k)
    run['conn'] = np.array(run['conn'])
    run['relay_removed'] = [np.int64(x) for x in run['relay_removed']]
    run['relay_created'] = np.array(run['relay_created'], dtype=np.int64)
    run['bridge_teardowns'] = [np.int64(x) for x in run['bridge_teardowns']]
    run['stale_edges'] = np.array(run['stale_edges'], dtype=np.int64)
    run['n_radio_sites'] = np.int64(SY.N_SITES)
    run['overhead_by_type']['bfd'] = np.array(run['overhead_by_type']['bfd'])
    run['relay_lifetimes'] = [{'link_id': 'TR_1_2_1', 'created': np.int64(5), 'up': np.int64(8),
                               'removed': np.int64(11), 'cause': 'mode_change', 'was_bridge': True,
                               'load': np.float64(0.0)}]
    return run


def test_dump_and_timelines_accept_numpy_and_stay_serialisable():
    if np is None:
        return
    run = _numpy_run()
    run['topology_info'] = {'n': np.int64(4), 'name': 'x'}
    d = DJ.summarize(run)
    json.dumps(d)
    assert d['n_radio_sites'] == {'scalar': SY.N_SITES}
    assert d['relay_removed']['total'] == sum(int(x) for x in run['relay_removed']) and d['relay_removed']['total'] > 0
    assert d['conn']['n'] == SY.T_TEST and d['overhead_by_type']['by_key']['bfd']['total'] == 60.0 * 35
    assert d['topology_info'] == {'dict': {'n': 4, 'name': 'x'}}
    assert d['relay_lifetimes']['records'][0]['created'] == 5          # numpy ticks converted (D5)
    with SY.patched(T_SEV={'A': 5, 'B': 0}):
        blk = DJ.stability_block(run, 'recovery_multi', {'attainable_optimum': 2})
    json.dumps(blk)
    plain = SY.make_run(SY.T_TEST, 1, 'marl_static', 'A')
    plain['relay_lifetimes'] = [dict(run['relay_lifetimes'][0], created=5, up=8, removed=11, load=0.0)]
    with SY.patched(T_SEV={'A': 5, 'B': 0}):
        assert blk == DJ.stability_block(plain, 'recovery_multi', {'attainable_optimum': 2})
    assert ET.binned([np.int64(1), np.float64('nan'), np.int64(2)], 1, 'sum') == [3]
    assert ET.num(np.int64(3)) and not ET.num(np.float64('nan')) and DJ.num(np.float32(1.0))
    # D3: the binned sum of numpy ints must be a Python number, not np.int64
    b = ET.binned([np.int64(1)] * 10, 1, 'sum') + ET.binned(np.array([2.0] * 10), 1, 'mean')
    assert b == [10, 2.0] and [type(x) for x in b] == [int, float]
    json.dumps(b)


def test_timelines_extract_on_numpy_runs_is_json_serialisable():
    """D3: the event series of the WP4 recording may be numpy (the handoff
    accepts np.int64 arrays and lists); binned sums stayed np.int64 and
    json.dump raised at the very end of the export."""
    if np is None:
        return
    tmp = SY.mkdtemp('wp9_tl_np_')

    def to_numpy(sk, arm, s, run):
        if arm != 'marl_static':
            return run
        run['relay_removed'] = np.array(run['relay_removed'], dtype=np.int64)      # ndarray
        run['relay_created'] = [np.int64(x) for x in run['relay_created']]         # list of numpy scalars
        run['bridge_teardowns'] = [np.int64(x) for x in run['bridge_teardowns']]
        run['stale_edges'] = np.array(run['stale_edges'], dtype=np.int64)
        run['conn'] = np.array(run['conn'])
        return run

    a = SY.make_pickle(os.path.join(tmp, 'np.pkl'), ('marl_static', 'xldet'), [1, 2, 3], mutate=to_numpy)
    out = ET.extract(ET.load_runs([a]))
    json.dumps(out)                                                   # the defect: TypeError here
    marl = out['scenarios']['A']['arms']['marl_static']
    assert marl['not_recorded'] == []
    plain = ET.extract(ET.load_runs([SY.make_pickle(os.path.join(tmp, 'plain.pkl'),
                                                    ('marl_static', 'xldet'), [1, 2, 3])]))
    assert marl == plain['scenarios']['A']['arms']['marl_static']     # same values as the plain run
    for key in ('relay_removed', 'relay_created', 'conn'):
        for stat in ('mean', 'median', 'q25', 'q75'):
            for x in marl[key][stat]:
                assert x is None or type(x) in (int, float), (key, stat, type(x))


def test_dump_summarize_all_numpy_dict_is_a_dict_not_a_series_set():
    """D4: `hasattr(x, 'tolist')` is true for numpy SCALARS too, so an
    all-numpy {site: count} dict took the {name: series} branch and aborted the
    whole dump at list(np.int64(4).tolist())."""
    if np is None:
        return
    d = DJ.summarize({'conn': [1.0] * 10,
                      'postcards_by_site': {'s1': np.int64(4), 's2': np.int64(5)},
                      'topology_info': {'n': np.int64(4), 'm': np.float64(1.5)},
                      'mixed': {'n': np.int64(4), 'name': 'x'},
                      'series_set': {'bfd': np.array([1.0, 2.0]), 'lsa': [3.0, 4.0]}})
    json.dumps(d)
    assert d['postcards_by_site'] == {'dict': {'s1': 4, 's2': 5}}
    assert d['topology_info'] == {'dict': {'n': 4, 'm': 1.5}}
    assert d['mixed'] == {'dict': {'n': 4, 'name': 'x'}}
    assert set(d['series_set']['by_key']) == {'bfd', 'lsa'}           # real series still take by_key
    assert d['series_set']['by_key']['bfd']['total'] == 3.0
    assert [type(x) for x in d['postcards_by_site']['dict'].values()] == [int, int]
    # the same dict inside a full run: the dump of a contract-compliant pickle
    run = _numpy_run()
    run['postcards_by_site'] = {'s%d' % i: np.int64(4 + i) for i in range(SY.N_SITES)}
    out = DJ.summarize(run)
    json.dumps(out)
    assert out['postcards_by_site'] == {'dict': {'s0': 4, 's1': 5, 's2': 6}}
    # and the analysis reads it as a per-site record with a site count
    assert CA._postcards_per_site(run) == (4, SY.N_SITES, [])   # (min, sites recorded, bad entries)
