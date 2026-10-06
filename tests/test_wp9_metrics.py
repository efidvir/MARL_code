"""WP9: stability metric helpers M1-M11 on tiny synthetic series with known
answers, the 'not recorded' rule (None whenever a needed series is absent),
the numeric helpers (numpy scalars are counted, never dropped) and the
Kaplan-Meier rule of M11.
"""
import json
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

# ts = 2, T = 10, o = 1
F = [1, 1, 3, 3, 2, 2, 3, 1, 1, 2]
RAW = [1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
TS, O = 2, 1


def _run(**kw):
    return dict(kw)


def test_m1_counts_changes_after_the_event():
    # t in [3, 10): 3->2 (t=4), 2->3 (t=6), 3->1 (t=7), 1->2 (t=9)
    assert CA.m1_fragment_changes(_run(fragments=F), TS) == 4
    # the severance change itself (t=2) is excluded; ts=1 includes it
    assert CA.m1_fragment_changes(_run(fragments=F), 1) == 5
    assert CA.m1_fragment_changes(_run(fragments=[2] * 6), 0) == 0
    assert CA.m1_fragment_changes(_run(), TS) is None


def test_m2_policy_splits_require_raw_not_increasing():
    # rises at t=6 (2->3) and t=9 (1->2) while raw stays 3
    assert CA.m2_policy_splits(_run(fragments=F, fragments_raw=RAW), TS) == 2
    # a rise that coincides with a raw increase is a real split, not counted
    f = [1, 2, 3]
    assert CA.m2_policy_splits(_run(fragments=f, fragments_raw=[1, 2, 3]), 0) == 0
    assert CA.m2_policy_splits(_run(fragments=f, fragments_raw=[1, 1, 1]), 0) == 2
    assert CA.m2_policy_splits(_run(fragments=F), TS) is None          # no fragments_raw
    assert CA.m2_policy_splits(_run(fragments_raw=RAW), TS) is None


def test_m3_optimum_losses():
    # f[t-1] <= 1 < f[t]: only t=9 (1 -> 2)
    assert CA.m3_optimum_losses(_run(fragments=F), TS, O) == 1
    assert CA.m3_optimum_losses(_run(fragments=[1, 2, 1, 2, 1, 3]), 0, 1) == 3
    assert CA.m3_optimum_losses(_run(fragments=F), TS, None) is None
    assert CA.m3_optimum_losses(_run(), TS, O) is None


def test_m4_and_m4ss_fraction_at_optimum():
    # [2, 10): f <= 1 at t = 7, 8 -> 2/8
    assert abs(CA.m4_fraction_at_optimum(_run(fragments=F), TS, O) - 0.25) < 1e-12
    # explicit window [6, 10): t = 7, 8 -> 0.5
    assert abs(CA.m4_fraction_at_optimum(_run(fragments=F), TS, O, lo=6, hi=10) - 0.5) < 1e-12
    assert abs(CA.m4ss_steady_at_optimum(_run(fragments=F), O, w=4) - 0.5) < 1e-12
    # W longer than the run: the whole run
    assert abs(CA.m4ss_steady_at_optimum(_run(fragments=F), O, w=500) - 0.4) < 1e-12   # t = 0, 1, 7, 8
    assert CA.m4_fraction_at_optimum(_run(fragments=F), TS, None) is None
    assert CA.m4_fraction_at_optimum(_run(), TS, O) is None
    assert CA.m4_fraction_at_optimum(_run(fragments=F), 10, O) is None   # empty window


def test_m6_settling_time_and_censoring():
    # last t >= 2 with f > 1 is t = 9 -> 9 + 1 - 2 = 8, censored (f[T-1] > o)
    assert CA.m6_settling_time(_run(fragments=F), TS, O) == {'value': 8, 'censored': True}
    # settles: last above at t = 2 -> value 3, not censored
    assert CA.m6_settling_time(_run(fragments=[2, 2, 2, 1, 1]), 0, 1) == {'value': 3, 'censored': False}
    # never above the optimum after ts
    assert CA.m6_settling_time(_run(fragments=[3, 1, 1, 1]), 1, 1) == {'value': 0, 'censored': False}
    assert CA.m6_settling_time(_run(fragments=F), TS, None) is None
    assert CA.m6_settling_time(_run(), TS, O) is None


def test_m7_teardowns_per_1000_post_event_ticks():
    rr = [0, 0, 1, 0, 2, 0, 0, 0, 0, 1]
    # sum over [3, 10) = 3 ; window length 10 - 2 - 1 = 7
    assert abs(CA.m7_teardowns_per_1000(_run(relay_removed=rr), TS) - 1000.0 * 3 / 7) < 1e-9
    assert CA.m7_teardowns_per_1000(_run(relay_count=[0, 1, 0]), TS) is None   # net series is not enough
    assert CA.m7_teardowns_per_1000(_run(relay_removed=rr), 9) is None         # empty window


def test_m8_bridge_and_idle_bridge_teardowns():
    assert CA.m8_bridge_teardowns(_run(bridge_teardowns=[0, 1, 0, 1], idle_bridge_teardowns=[0, 0, 0, 1])) == \
        {'bridge': 2, 'idle': 1}
    assert CA.m8_bridge_teardowns(_run(bridge_teardowns=[1, 1])) == {'bridge': 2, 'idle': None}
    assert CA.m8_bridge_teardowns(_run(relay_bridges=[0, 1, 0])) is None


def test_m9_mode_flips_per_1000_ticks():
    fl = [0, 0, 1, 1, 0, 2, 0, 0, 0, 0]
    assert abs(CA.m9_mode_flips_per_1000(_run(relay_mode_flips=fl)) - 400.0) < 1e-9
    assert CA.m9_mode_flips_per_1000(_run(relay_mode_nodes=[0, 1])) is None


def test_m10_steady_window_changes():
    # [6, 10): changes at t = 6, 7, 9
    assert CA.m10_steady_fragment_changes(_run(fragments=F), w=4) == 3
    assert CA.m10_steady_fragment_changes(_run(fragments=F), w=500) == 5   # whole run, t >= 1
    assert CA.m10_steady_fragment_changes(_run()) is None


def test_km_median_rules():
    # no censoring: the smallest event time at which S(t) <= 0.5
    assert CA.km_median([5, 1, 3], [True, True, True]) == 3            # S(1) = 2/3, S(3) = 1/3
    assert CA.km_median([1, 2, 3, 4], [True] * 4) == 2                 # S(2) = 3/4 * 2/3 = 0.5 exactly
    # a record censored at an event time is still at risk at that time
    assert CA.km_median([2, 2, 2], [True, False, True]) == 2           # S(2) = 1 - 2/3
    # censoring dominates: S never reaches 0.5 -> undefined, never a substitute
    assert CA.km_median([1, 50, 60, 70], [True, False, False, False]) is None   # S(1) = 0.75
    # censored records lengthen the at-risk set without adding events
    assert CA.km_median([10, 30, 20, 40], [True, True, False, False]) == 30     # S(10) = .75, S(30) = .75 * 2/3
    assert CA.km_median([], []) is None


def test_m11_kaplan_meier_median_with_censoring():
    recs = [{'link_id': 'a', 'created': 5, 'up': 10, 'removed': 40, 'cause': 'mode', 'was_bridge': True, 'load': 0.0},
            ('b', 5, 8, None, None, True, 0.1),                  # tuple record, still up at T
            {'link_id': 'c', 'created': 1, 'up': 2, 'removed': 3, 'was_bridge': False}]
    # lifetimes: a = 30 completed, b censored at 100 - 8 = 92: S(30) = 1 - 1/2 -> KM median 30;
    # completed-only median 30; the lower-bound rule (censored at T) would say 61
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=recs), T=100)
    assert m == {'median': 30, 'median_completed': 30, 'median_censored_at_T': 61.0,
                 'n': 2, 'n_completed': 1, 'n_censored': 1, 'n_invalid': 0, 'invalid': []}, m
    # T falls back to the fragments length (b censored at 42)
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=recs, fragments=[1] * 50))
    assert m == {'median': 30, 'median_completed': 30, 'median_censored_at_T': 36.0,
                 'n': 2, 'n_completed': 1, 'n_censored': 1, 'n_invalid': 0, 'invalid': []}, m
    # every bridge censored: the KM median is undefined (None), n_censored says why
    only_c = [('b', 5, 8, None, None, True, 0.1), ('d', 1, 4, None, None, True, 0.2)]
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=only_c), T=100)
    assert m == {'median': None, 'median_completed': None, 'median_censored_at_T': 94.0,
                 'n': 2, 'n_completed': 0, 'n_censored': 2, 'n_invalid': 0, 'invalid': []}, m
    assert CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[]), T=10) == {
        'median': None, 'median_completed': None, 'median_censored_at_T': None, 'n': 0,
        'n_completed': 0, 'n_censored': 0, 'n_invalid': 0, 'invalid': []}
    # numpy ticks: lifetime_record converts them, so nothing downstream carries
    # a numpy scalar and the block stays JSON-serialisable (defect list D5)
    if np is not None:
        nrecs = [{'link_id': 'a', 'created': np.int64(5), 'up': np.int64(10), 'removed': np.int64(40),
                  'cause': 'mode', 'was_bridge': True, 'load': np.float64(0.25)},
                 ('b', np.int64(5), np.int64(8), None, None, True, np.float32(0.1))]
        m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=nrecs), T=100)
        assert m['median'] == 30 and m['n'] == 2 and m['n_completed'] == 1
        assert [type(m[k]) for k in ('median', 'median_censored_at_T')] == [int, float], m
        json.dumps(m)
        rec = CA.lifetime_record(nrecs[0])
        assert [type(rec[k]) for k in ('created', 'up', 'removed', 'load')] == [int, int, int, float]
        assert rec['was_bridge'] is True and rec['link_id'] == 'a'
        json.dumps(CA.lifetime_record(nrecs[1]))
    assert CA.m11_median_bridge_lifetime(_run(relay_bridges=[0, 1]), T=10) is None
    d = CA.lifetime_record(('x', 1, 2, 3, 'cause', True, 0.5))
    assert d == {'link_id': 'x', 'created': 1, 'up': 2, 'removed': 3, 'cause': 'cause', 'was_bridge': True, 'load': 0.5}
    try:
        CA.lifetime_record(42)
    except TypeError:
        pass
    else:
        raise AssertionError('lifetime_record must reject a scalar record')


def test_m11_refuses_records_that_carry_no_lifetime():
    """D7: a bridge record whose start tick is missing was dropped silently and
    one with removed == T was counted as a completed lifetime of T - start."""
    good = {'link_id': 'a', 'created': 5, 'up': 10, 'removed': 40, 'was_bridge': True}
    no_start = {'link_id': 'b', 'created': None, 'up': None, 'removed': 5, 'was_bridge': True}
    at_T = {'link_id': 'c', 'created': 2, 'up': 3, 'removed': 100, 'was_bridge': True}
    before = {'link_id': 'd', 'created': 20, 'up': 20, 'removed': 19, 'was_bridge': True}
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[good, no_start, at_T, before]), T=100)
    assert m['n'] == 1 and m['n_completed'] == 1 and m['median'] == 30
    assert m['n_invalid'] == 3 and [x[0] for x in m['invalid']] == ['b', 'c', 'd']
    assert 'no start tick' in m['invalid'][0][1]
    assert 'removed 100 >= T=100' in m['invalid'][1][1] and 'before the start tick' in m['invalid'][2][1]
    # a teardown on the last tick is valid (censoring is `removed is None`)
    last = {'link_id': 'e', 'created': 2, 'up': 3, 'removed': 99, 'was_bridge': True}
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[last]), T=100)
    assert m['n_invalid'] == 0 and m['n_completed'] == 1 and m['median'] == 96
    # a non-bridge record is not a violation, it is simply not M11's business
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[dict(no_start, was_bridge=False)]), T=100)
    assert m['n'] == 0 and m['n_invalid'] == 0
    t, e, bad = CA.bridge_lifetimes([CA.lifetime_record(good)], 100)
    assert (t, e, bad) == ([30], [True], [])

    # D7 residual: the START tick was never range-checked, so `up == T` -- the
    # same off-by-one the round refused for `removed == T` -- was accepted as a
    # censored observation and T - start entered the metric as 0; `up > T` made
    # it NEGATIVE and `up < 0` made it longer than the run.  n_invalid stayed 0.
    at_T = {'link_id': 'f', 'created': 10, 'up': 100, 'removed': None, 'was_bridge': True}
    past_T = {'link_id': 'g', 'created': 10, 'up': 105, 'removed': None, 'was_bridge': True}
    neg = {'link_id': 'h', 'created': -10, 'up': -10, 'removed': None, 'was_bridge': True}
    nan_up = {'link_id': 'i', 'created': None, 'up': float('nan'), 'removed': None, 'was_bridge': True}
    text = {'link_id': 'j', 'created': None, 'up': 'early', 'removed': None, 'was_bridge': True}
    for rec in (at_T, past_T, neg, nan_up, text):
        m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[rec]), T=100)
        assert m['n'] == 0 and m['n_censored'] == 0 and m['n_invalid'] == 1, (rec, m)
        assert m['median'] is None and m['median_censored_at_T'] is None, (rec, m)
        assert m['invalid'][0][0] == rec['link_id'] and 'start tick' in m['invalid'][0][1], m['invalid']
        assert 'outside [0, T=100)' in m['invalid'][0][1], m['invalid']
    # the zero/negative lifetime never reaches the reported median again
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[at_T, past_T, good]), T=100)
    assert m['n'] == 1 and m['n_invalid'] == 2 and m['median'] == 30 and m['median_censored_at_T'] == 30
    # the last valid start tick is T - 1, and 0 is valid
    ok_last = {'link_id': 'k', 'created': 0, 'up': 99, 'removed': None, 'was_bridge': True}
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[ok_last]), T=100)
    assert m['n_invalid'] == 0 and m['n_censored'] == 1 and m['median_censored_at_T'] == 1
    ok_zero = {'link_id': 'l', 'created': 0, 'up': 0, 'removed': 4, 'was_bridge': True}
    m = CA.m11_median_bridge_lifetime(_run(relay_lifetimes=[ok_zero]), T=100)
    assert m['n_invalid'] == 0 and m['n_completed'] == 1 and m['median'] == 4


def test_pre_registered_constants_are_the_ones_in_the_plan():
    """The study constants are pinned here, against PREREGISTRATION_R2 sections
    7.2, 8, 9 and 10: an accidental edit before freeze 1 must break a test, not
    a report (the metric tests all run on synthetic overrides -- T_SEV_TEST,
    TS = 2, T = 10 -- so nothing else pins them)."""
    assert CA.T_SEV == {'A': 200, 'B': 0}            # severance tick; post-event window [T_SEV, T)
    assert CA.T_TOTAL == 3600                        # ticks per evaluated pass
    assert CA.W == 500                               # steady-state window [T - W, T)
    assert CA.MIN_NONZERO == 6                       # S9: fewer non-zero differences -> not assessable
    assert CA.ALPHA == 0.05                          # Holm over 9 (P) and over 6 (S9)
    assert CA.S9_ENDPOINTS == (('S9a', 'A', 'M2', +1), ('S9b', 'B', 'M2', +1),
                               ('S9c', 'A', 'M4', -1), ('S9d', 'B', 'M4', -1),
                               ('S9e', 'A', 'M7', +1), ('S9f', 'B', 'M7', +1))
    assert len(CA.S9_ENDPOINTS) == 6 and [e[0] for e in CA.S9_ENDPOINTS] == ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
    # the other pre-registered constants of the same sections
    assert CA.PREREG_MARL_ARM == 'marl_static' and CA.SURGE_B == 1400
    assert CA.T_EVENT == {'A': 200, 'B': 1400} and CA.PRE_SURGE == (900, 1400)
    assert CA.EPS == 1e-9 and CA.NAN_EXEMPT == ('latency',)
    assert CA.STABILITY_METRICS == ('M1', 'M2', 'M3', 'M4', 'M4ss', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11')
    assert CA.KIND == {'A': 'recovery', 'B': 'rescue_ops'} and CA.SCEN == {'A': 'recovery_multi', 'B': 'rescue_multi'}


def test_stability_metrics_dict_and_not_recorded():
    r = _run(fragments=F, fragments_raw=RAW)
    m = CA.stability_metrics(r, TS, O, T=10)
    assert set(m) == set(CA.STABILITY_METRICS)
    assert (m['M1'], m['M2'], m['M3']) == (4, 2, 1)
    assert m['M6'] == {'value': 8, 'censored': True}
    for key in ('M7', 'M8', 'M9', 'M11'):
        assert m[key] is None, key             # WP4 series absent: not recorded, no lower bound
    full = _run(fragments=F, fragments_raw=RAW, relay_removed=[0, 0, 1, 0, 2, 0, 0, 0, 0, 1],
                bridge_teardowns=[0] * 10, idle_bridge_teardowns=[0] * 10, relay_mode_flips=[1] * 10,
                relay_lifetimes=[])
    m = CA.stability_metrics(full, TS, O, T=10)
    assert m['M7'] is not None and m['M8'] == {'bridge': 0, 'idle': 0} and m['M9'] == 1000.0
    assert m['M11'] == {'median': None, 'median_completed': None, 'median_censored_at_T': None,
                        'n': 0, 'n_completed': 0, 'n_censored': 0, 'n_invalid': 0, 'invalid': []}


def test_lower_bounds_are_separate_and_labelled():
    r = _run(relay_count=[0, 2, 1, 3, 1], relay_bridges=[0, 1, 0, 2, 1])
    # decrements: t=2 (1), t=4 (2) -> 3 over 4 post-event ticks
    assert abs(CA.m7_lower_bound_relay_count(r, 0) - 750.0) < 1e-9
    assert CA.m8_lower_bound_relay_bridges(r) == 2
    assert CA.m7_teardowns_per_1000(r, 0) is None        # the metric itself stays unrecorded
    assert CA.m7_lower_bound_relay_count(_run(), 0) is None


def test_dstats_and_quantiles():
    d = CA.dstats([3, None, 1, 2, 10])
    assert d['n'] == 5 and d['n_recorded'] == 4 and d['median'] == 2.5 and d['min'] == 1 and d['max'] == 10
    assert d['q25'] == 1.75 and d['q75'] == 4.75
    assert CA.dstats([None, None]) == {'status': CA.NOT_RECORDED, 'n': 2, 'n_recorded': 0}
    assert CA.quantile([], 0.5) is None


def test_series_accessors_accept_numpy_like_and_dicts():
    class Arr:
        def __init__(self, v):
            self.v = v

        def tolist(self):
            return list(self.v)
    r = {'fragments': Arr([1, 2, 2]), 'overhead_by_type': {'bfd': Arr([1.0, 2.0]), 'lsa': [0.0, 1.0]}}
    assert CA.series(r, 'fragments') == [1, 2, 2]
    assert CA.series(r, 'missing') is None
    assert CA.dict_series(r, 'overhead_by_type') == {'bfd': [1.0, 2.0], 'lsa': [0.0, 1.0]}
    assert CA.dict_series(r, 'fragments') is None


def test_numeric_helpers_count_every_numeric_type():
    assert CA.is_num(1) and CA.is_num(1.5) and CA.is_num(True)
    assert not CA.is_num(None) and not CA.is_num(float('nan')) and not CA.is_num('3')
    assert CA.is_real(float('nan')) and not CA.is_real(None) and not CA.is_real('3') and not CA.is_real([1])
    assert CA.num([1, None, float('nan'), 2.5, 'x']) == [1, 2.5]
    assert CA.py_num('x') == 'x' and CA.py_num(3) == 3 and CA.py_num(None) is None
    if np is None:
        return
    xs = [np.int64(3), np.float32(1.5), np.float64('nan'), None, np.int32(4)]
    got = CA.num(xs)
    assert got == [3, 1.5, 4] and [type(x) for x in got] == [int, float, int], got
    # a series of numpy scalars is a numeric series, never all-missing
    rr = [np.int64(x) for x in [0, 0, 1, 0, 2, 0, 0, 0, 0, 1]]
    assert abs(CA.m7_teardowns_per_1000(_run(relay_removed=rr), TS) - 1000.0 * 3 / 7) < 1e-9
    m8 = CA.m8_bridge_teardowns(_run(bridge_teardowns=np.array([0, 1, 0, 1]),
                                     idle_bridge_teardowns=[np.int64(0), np.int64(1)]))
    assert m8 == {'bridge': 2, 'idle': 1} and type(m8['bridge']) is int and type(m8['idle']) is int
    assert CA.m9_mode_flips_per_1000(_run(relay_mode_flips=np.array([1, 0, 1, 0]))) == 500.0
    assert CA.m8_lower_bound_relay_bridges(_run(relay_bridges=[np.int64(0), np.int64(2), np.int64(1)])) == 1
    # series() and dict_series() hand out Python numbers (JSON-serialisable)
    s = CA.series({'a': np.array([1, 2, 3])}, 'a')
    assert s == [1, 2, 3] and all(type(x) is int for x in s)
    s = CA.series({'a': [np.float64(0.5), None, np.int64(2)]}, 'a')
    assert s == [0.5, None, 2] and type(s[0]) is float and type(s[2]) is int
    d = CA.dict_series({'d': {'k': np.array([0.5, 1.0]), 'j': [np.int64(1)]}}, 'd')
    assert d == {'k': [0.5, 1.0], 'j': [1]}
    json.dumps([s, d, m8, CA.stability_metrics(_run(fragments=np.array(F), fragments_raw=np.array(RAW),
                                                    relay_removed=rr), TS, O, T=10)])
    # numpy fragments give the same counts as the plain list
    assert CA.m1_fragment_changes(_run(fragments=np.array(F)), TS) == 4
    assert CA.m4_fraction_at_optimum(_run(fragments=np.array(F)), TS, O) == 0.25
