"""WP9: the S9 stability family -- Holm ordering, the 'not assessable' rule
(fewer than six non-zero differences), 'not recorded' endpoints and the
expected-direction rule."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import confirmatory_analysis as CA  # noqa: E402


def test_holm_step_down_ordering():
    # sorted: 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> running max 0.06)
    adj = CA.holm([0.01, 0.04, 0.03, 0.005])
    assert [round(x, 10) for x in adj] == [0.03, 0.06, 0.06, 0.02], adj
    # monotone in the sorted order and capped at 1
    adj = CA.holm([0.5, 0.9, 0.2])
    assert [round(x, 10) for x in adj] == [1.0, 1.0, 0.6], adj
    assert CA.holm([]) == []


def test_min_nonzero_rule_boundary():
    n = 20
    pos = list(range(1, n + 1))                                # S9a: all positive, distinct
    five = [1, 2, 3, 4, 5] + [0] * (n - 5)                     # S9b: 5 non-zero -> not assessable
    missing = [None] * n                                       # S9c: not recorded
    neg = [-x for x in pos]                                    # S9d: expected < 0, confirmed
    wrong = neg                                                # S9e: expected > 0 but negative
    six = [1, 2, 3, 4, 5, 6] + [0] * (n - 6)                   # S9f: exactly 6 -> assessable
    fam = CA.stability_family({'S9a': pos, 'S9b': five, 'S9c': missing, 'S9d': neg, 'S9e': wrong, 'S9f': six})
    assert list(fam) == ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
    a, b, c, d, e, f = (fam[k] for k in ('S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f'))
    assert a['status'] == 'assessable' and a['confirmed'] and a['n_nonzero'] == 20
    assert a['family_size'] == 6 and a['wilcoxon_p_holm'] is not None
    assert b['status'] == CA.NOT_ASSESSABLE and b['confirmed'] is False and b['wilcoxon_p_holm'] is None
    assert b['n_nonzero'] == 5 and b['holm_input_p'] == 1.0 and b['wilcoxon_p'] == 0.0625
    assert c['status'] == CA.NOT_RECORDED and c['confirmed'] is False and c['n_not_recorded'] == 20
    assert d['status'] == 'assessable' and d['confirmed'] and d['direction_ok']
    assert e['status'] == 'assessable' and not e['confirmed'] and not e['direction_ok']
    assert f['status'] == 'assessable' and f['n_nonzero'] == 6 and abs(f['wilcoxon_p'] - 2 / 64) < 1e-12
    # Holm over 6 with two p = 1 entries: the assessable ones are adjusted with m = 6
    assert abs(a['wilcoxon_p_holm'] - min(1.0, 6 * a['wilcoxon_p'])) < 1e-12
    for k in ('S9b', 'S9c'):
        assert fam[k]['holm_input_p'] == 1.0


def test_family_ignores_unassessable_p_in_the_step_down():
    n = 20
    strong = [x for x in range(1, n + 1)]
    fam = CA.stability_family({'S9a': strong, 'S9b': [1, 0, 0, 0] + [0] * 16, 'S9c': [None] * n,
                               'S9d': [-x for x in strong], 'S9e': strong, 'S9f': strong})
    ps = [fam[k]['holm_input_p'] for k in fam]
    assert ps[1] == 1.0 and ps[2] == 1.0
    # the four assessable endpoints share the same raw p; Holm gives (6, 5, 4, 3) x p in rank order
    raw = fam['S9a']['wilcoxon_p']
    got = sorted(fam[k]['wilcoxon_p_holm'] for k in ('S9a', 'S9d', 'S9e', 'S9f'))
    assert abs(got[-1] - 6 * raw) < 1e-12 and all(abs(g - 6 * raw) < 1e-12 for g in got)   # running max


def test_empty_and_partial_diffs_are_not_recorded():
    fam = CA.stability_family({'S9a': [], 'S9b': None, 'S9c': [1, None, 2], 'S9d': [1] * 20,
                               'S9e': [1] * 20, 'S9f': [1] * 20})
    assert fam['S9a']['status'] == CA.NOT_RECORDED and fam['S9a']['n'] == 0
    assert fam['S9b']['status'] == CA.NOT_RECORDED and fam['S9b']['n_not_recorded'] is None
    assert fam['S9c']['status'] == CA.NOT_RECORDED and fam['S9c']['n_not_recorded'] == 1
    # all-tied differences: exact p = 2 / 2^20, direction ok -> confirmed after Holm over 6
    assert fam['S9d']['confirmed'] is False          # expected < 0 but all positive
    assert fam['S9e']['confirmed'] and fam['S9f']['confirmed']


def test_paired_and_wilcoxon_basics():
    p, wp, wm, nz = CA.wilcoxon([1, -1, 2, 0])
    assert nz == 3 and wp + wm == 6
    assert CA.wilcoxon([0, 0]) == (1.0, 0.0, 0.0, 0)
    r = CA.paired([1, 2, 3, 4, 5, 6], +1)
    assert r['n_nonzero'] == 6 and r['direction_ok'] and abs(r['wilcoxon_p'] - 2 / 64) < 1e-12
    assert r['wins'] == 6 and r['sign_p'] == 2 / 64
