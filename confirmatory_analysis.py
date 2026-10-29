# -*- coding: utf-8 -*-
"""Pre-registered analysis of a confirmatory instance set.

This file is written, validated and hashed BEFORE the confirmatory evaluation
runs.  Validation = reproducing the frozen Amendment-1 reports
(~/MARL_run_6/confirmatory/cf_report_A1.json and dev_report_A1.json) for
every P1-P9 and S1-S7 value from the frozen pickles (tools/compare_reports.py),
plus the manuscript's M1 fragment-change counts (docs/WP9_ANALYSIS.md).

Definitions are those of paper/make_results.py:
  steady(x)   mean of the numeric values among the last 500 ticks
  J/unit      per instance: whole-run energy / whole-run delivered volume
  pooled J/u  sum of energy / sum of delivered over the instance set
  pre-surge   Scenario B: mean achievability over ticks [900, 1400)
  best routing per instance: max achievability / min J/unit over the four
              routing protocols (the conservative comparator for MARL-RIC)

Primary family (nine tests, Holm-corrected at alpha = 0.05, decision test =
exact two-sided Wilcoxon signed-rank on the paired per-instance differences,
zeros dropped, mid-ranks for ties).  A hypothesis is CONFIRMED iff the Holm-
adjusted p < 0.05 AND the signed-rank statistic lies in the expected direction.

Stability (RERUN_SPEC WP4/WP9; PREREGISTRATION_R2 sections 7-10):
  metric helpers M1..M11 (module level, imported by dump_results_json.py and
  extract_timelines.py so that one definition exists), the S9 stability family
  (own Holm family of six, MARL minus XL-DET, "not assessable" below six
  non-zero differences), S10 descriptive stability per arm, S11 descriptive
  billed overhead, S12 untrained-actor control, S8 hash invariance (result of
  tools/hash_invariance_check.py, embedded with --s8).

Hard asserts (RERUN_SPEC WP9): every arm has every manifest seed (the
adapt-then-freeze arm too when it is present at all); provenance fields
present and identical across pickles, the selected checkpoint's sha256
identical across the pickles that carry the tested MARL arm; stale_edges == 0;
more than one postcard per site per episode after the event (per site when
`postcards_by_site` / `postcards_min_per_site` is recorded, else the aggregate
mean per site); no NaN/None/non-numeric entry in any per-tick series, the
{name: series} members included; run length.  --no-assert turns them into a
report section (legacy pickles of the frozen tree).

Numeric entries: Python int/float and numpy integer/floating scalars are
all accepted (`is_num`, `num`, `series`); numpy arrays are converted with
tolist().  A numeric type is never silently dropped from a sum.

Usage:
  python3 confirmatory_analysis.py --pickles PKL[,PKL] --manifests JSON[,JSON]
      --out REPORT.json [--marl-arm ARM] [--no-assert]
      [--untrained NAME=PKL[,NAME=PKL]] [--untrained-arm marl_static]
      [--s8 hash_invariance.json] [--nan-exempt latency]

Amendment 1 (see confirmatory/AMENDMENT_1.md): --marl-arm names the arm the
primary hypotheses test (default 'marl_freeze', the original pre-registration;
the amended and the R2 analyses use 'marl_static').  Pickles are merged arm by
arm, so the static arm's pickle can be combined with the baseline pickle.  When
the tested arm is not 'marl_freeze' and that arm is present, secondary S7
compares the two MARL executors.
"""
import argparse
import json
import math
import numbers
import os
import pickle
import statistics as st

W = 500
MARL, XL, SDN = 'marl_freeze', 'xldet', 'sdn'
# The pre-registered MARL-RIC arm (pre-registration R2 sections 6, 8, 10 and
# 12).  MARL above is only the module default that analyse() overwrites from
# --marl-arm; `marl_freeze` is the SECONDARY adapt-then-freeze variant (S7),
# so an analysis run without the flag would produce a complete, plausible
# P1-P9 and S9 report for the wrong arm under the label MARL-RIC.  The hard
# asserts refuse any other value.
PREREG_MARL_ARM = 'marl_static'
ROUTING = ('ospf', 'olsr', 'batman', 'aodv')
ARMS = (MARL, XL, SDN) + ROUTING + ('random',)
SCEN = {'A': 'recovery_multi', 'B': 'rescue_multi'}
KIND = {'A': 'recovery', 'B': 'rescue_ops'}
T_EVENT = {'A': 200, 'B': 1400}
PRE_SURGE = (900, 1400)
ALPHA = 0.05

# ---------------------------------------------------------------- WP9 constants
T_SEV = {'A': 200, 'B': 0}     # severance tick: the post-event window is [T_SEV, T)
T_TOTAL = 3600                  # ticks per evaluated pass
SURGE_B = 1400                  # Scenario B surge (descriptive surge window only)
MIN_NONZERO = 6                 # S9: fewer non-zero differences -> "not assessable"
NOT_RECORDED = 'not recorded'
NOT_ASSESSABLE = 'not assessable'
# arms on the MARL code path (link-state service, control postcards)
MARL_PATH_ARMS = ('marl_static', 'marl_freeze', 'marl', 'xldet', 'random')
# Per-tick event series of the WP4 recording (per-tick differences of the
# cumulative simulator counters).  Pickles of the frozen tree lack them; every
# metric that needs one returns None ("not recorded"), never a lower bound.
EVENT_SERIES = ('relay_created', 'relay_removed', 'bridge_teardowns',
                'idle_bridge_teardowns', 'relay_mode_flips')
DIAG_SERIES = ('stale_edges', 'stale_u2u_volume')
LIFETIMES_KEY = 'relay_lifetimes'              # list of per-link records
LIFETIME_FIELDS = ('link_id', 'created', 'up', 'removed', 'cause', 'was_bridge', 'load')
LIFETIME_NUMERIC = ('created', 'up', 'removed', 'load')   # converted with py_num (numpy ticks)
REMOVED_BY_REASON_KEY = 'relay_removed_by_reason'   # {reason: per-tick list}
HOLDS_KEY = 'relay_teardown_holds'                  # {reason: per-tick list}
OVERHEAD_BY_TYPE_KEY = 'overhead_by_type'           # {type: per-tick bytes}
ENERGY_CTRL_KEY = 'energy_ctrl'
DICT_SERIES_KEYS = (REMOVED_BY_REASON_KEY, HOLDS_KEY, OVERHEAD_BY_TYPE_KEY)
PROVENANCE_FIELDS = ('tree_head', 'tree_sha256', 'schema_id', 'pythonhashseed')
NAN_EXEMPT = ('latency',)   # mean latency is undefined on a tick with no delivery
STABILITY_METRICS = ('M1', 'M2', 'M3', 'M4', 'M4ss', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11')
S9_ENDPOINTS = (('S9a', 'A', 'M2', +1), ('S9b', 'B', 'M2', +1),
                ('S9c', 'A', 'M4', -1), ('S9d', 'B', 'M4', -1),
                ('S9e', 'A', 'M7', +1), ('S9f', 'B', 'M7', +1))
EPS = 1e-9


# ------------------------------------------------------------------ helpers
def is_real(x):
    """True for any real number: Python int/float (bool included, as the
    frozen script accepted it) and numpy integer/floating scalars, which
    register with numbers.Real.  NaN is a real number here; None, strings
    and records are not."""
    return isinstance(x, numbers.Real)


def is_num(x):
    """A usable numeric entry: a real number (numpy scalars included) that
    is not NaN.  None and NaN are not."""
    return isinstance(x, numbers.Real) and not math.isnan(x)


def py_num(x):
    """The plain Python int/float of a real number (every numpy scalar
    converted, np.float64 included although it subclasses float, so that
    sums and reports are JSON-serialisable and type-stable); bool and any
    non-numeric value unchanged."""
    if isinstance(x, bool) or type(x) in (int, float) or not isinstance(x, numbers.Real):
        return x
    return int(x) if isinstance(x, numbers.Integral) else float(x)


def num(xs):
    """The numeric entries of a series as Python numbers: every element that
    is a real number and not NaN (numpy scalars included and converted);
    None and NaN dropped.  No numeric type is ever dropped silently."""
    return [py_num(x) for x in xs if is_num(x)]


def steady(xs):
    v = num(xs[-W:])
    return sum(v) / len(v) if v else float('nan')


def wmean(xs, lo, hi):
    v = num(xs[lo:hi])
    return sum(v) / len(v) if v else float('nan')


def jpu(r):
    return sum(num(r['energy'])) / sum(num(r['delivered']))


def sli(r, t0):
    conn, off = r['conn'], r['offered']
    pre = num(conn[:t0]); pre_off = num(off[:t0]); post_off = num(off[t0:])
    a_ref = st.mean(pre) * st.mean(pre_off) / st.mean(post_off)
    return sum(max(0.0, a_ref - (c if is_real(c) else 0.0)) for c in conn[t0:])


def sign_p(d):
    n = sum(1 for x in d if x != 0)
    if n == 0:
        return 1.0
    k = min(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(d):
    """Exact two-sided signed-rank test; zeros dropped, mid-ranks for ties.
    Returns (p, W_plus, W_minus, n_nonzero)."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0.0, 0.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    wm = sum(r for r, x in zip(ranks, d) if x < 0)
    doubled = [int(round(2 * r)) for r in ranks]
    dist = {0: 1}
    for r in doubled:
        nd = {}
        for s, c in dist.items():
            nd[s] = nd.get(s, 0) + c
            nd[s + r] = nd.get(s + r, 0) + c
        dist = nd
    centre = sum(doubled) / 2
    dev = abs(2 * wp - centre)
    p = sum(c for s, c in dist.items() if abs(s - centre) >= dev - 1e-9) / 2 ** n
    return min(1.0, p), wp, wm, n


def t_ci(d):
    n = len(d)
    mu = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    try:
        from scipy.stats import t as tdist
        q = tdist.ppf(0.975, n - 1)
    except Exception:
        q = {9: 2.262, 10: 2.228, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}.get(n - 1, 2.093)
    h = q * sd / math.sqrt(n)
    return mu, sd, mu - h, mu + h


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def paired(d, expect):
    p, wp, wm, nz = wilcoxon(d)
    mu, sd, lo, hi = t_ci(d)
    wins = sum(1 for x in d if (x > 0 if expect > 0 else x < 0))
    try:
        from scipy.stats import wilcoxon as swil
        sp = float(swil(d, zero_method='wilcox', alternative='two-sided').pvalue)
    except Exception:
        sp = None
    return dict(n=len(d), mean=mu, sd=sd, ci_lo=lo, ci_hi=hi, wins=wins,
                sign_p=sign_p(d), wilcoxon_p=p, wilcoxon_p_scipy=sp,
                W_plus=wp, W_minus=wm, n_nonzero=nz,
                direction_ok=(wp > wm) if expect > 0 else (wm > wp), expect=expect)


# ------------------------------------------------------ WP9: series access
def series(r, key):
    """Per-tick series `key` of run `r` as a list of Python numbers (numpy
    arrays through tolist(), numpy scalars converted, None/NaN kept); None
    when the run does not carry it.  Callers must report None as 'not
    recorded' -- no substitute."""
    if not isinstance(r, dict) or r.get(key) is None:
        return None
    v = r[key]
    if hasattr(v, 'tolist'):
        v = v.tolist()
    return [py_num(x) for x in v]


def dict_series(r, key):
    """A {name: per-tick list} member (overhead_by_type, relay_removed_by_reason,
    relay_teardown_holds) as a plain dict of lists, or None."""
    v = r.get(key) if isinstance(r, dict) else None
    if not isinstance(v, dict):
        return None
    out = {}
    for name, s in v.items():
        if hasattr(s, 'tolist'):
            s = s.tolist()
        out[str(name)] = [py_num(x) for x in s]
    return out


def lifetime_record(rec):
    """Normalise one relay_lifetimes entry (dict, or a tuple in LIFETIME_FIELDS
    order) to a dict with every LIFETIME_FIELDS key (missing -> None).  The
    numeric fields (LIFETIME_NUMERIC: the tick fields and the load) go through
    py_num, so a record written with numpy ints/floats yields plain Python
    numbers and every value derived from it -- M11, the stability block, the
    dump -- stays JSON-serialisable and type-stable."""
    if isinstance(rec, dict):
        out = {k: rec.get(k) for k in LIFETIME_FIELDS}
    elif isinstance(rec, (list, tuple)):
        vals = list(rec) + [None] * (len(LIFETIME_FIELDS) - len(rec))
        out = dict(zip(LIFETIME_FIELDS, vals[:len(LIFETIME_FIELDS)]))
    else:
        raise TypeError('relay_lifetimes record must be a dict or a tuple, got %r' % type(rec).__name__)
    for k in LIFETIME_NUMERIC:
        out[k] = py_num(out[k])
    return out


def lifetimes(r):
    """relay_lifetimes of run `r` as a list of normalised dicts, or None."""
    v = r.get(LIFETIMES_KEY) if isinstance(r, dict) else None
    if v is None:
        return None
    return [lifetime_record(x) for x in v]


def _fragment_changes(f, lo, hi):
    """Number of ticks t in [lo, hi) (t >= 1) with f[t] != f[t-1]."""
    return sum(1 for t in range(max(lo, 1), min(hi, len(f))) if f[t] != f[t - 1])


# ------------------------------------------------- WP9: metric helpers M1-M11
# ts = severance tick (T_SEV: A 200, B 0); o = attainable optimum of the
# instance (benchmark manifest); T = length of the series.  Every helper
# returns None when a series it needs is absent.

def m1_fragment_changes(r, ts):
    """M1: fragment-count changes after the event = number of ticks t in
    [ts+1, T) with fragments[t] != fragments[t-1].   Needs: fragments."""
    f = series(r, 'fragments')
    if f is None:
        return None
    return _fragment_changes(f, ts + 1, len(f))


def m2_policy_splits(r, ts):
    """M2: policy-induced splits = number of ticks t in [ts+1, T) with
    fragments[t] > fragments[t-1] while fragments_raw[t] <= fragments_raw[t-1]
    (the relay-inclusive count rises although the raw components did not).
    Needs: fragments, fragments_raw."""
    f, raw = series(r, 'fragments'), series(r, 'fragments_raw')
    if f is None or raw is None:
        return None
    T = min(len(f), len(raw))
    return sum(1 for t in range(ts + 1, T) if f[t] > f[t - 1] and raw[t] <= raw[t - 1])


def m3_optimum_losses(r, ts, o):
    """M3: optimum losses = number of ticks t in [ts+1, T) with
    fragments[t-1] <= o < fragments[t].   Needs: fragments (+ manifest o)."""
    f = series(r, 'fragments')
    if f is None or o is None:
        return None
    return sum(1 for t in range(ts + 1, len(f)) if f[t - 1] <= o + EPS < f[t])


def m4_fraction_at_optimum(r, ts, o, lo=None, hi=None):
    """M4: fraction of post-event ticks at (or below) the attainable optimum =
    mean over t in [ts, T) of 1[fragments[t] <= o].  With lo/hi, the window
    [lo, hi) instead (M4ss uses [T-W, T)).   Needs: fragments (+ manifest o)."""
    f = series(r, 'fragments')
    if f is None or o is None:
        return None
    lo = ts if lo is None else lo
    hi = len(f) if hi is None else min(hi, len(f))
    lo = max(0, lo)
    if hi <= lo:
        return None
    return sum(1 for t in range(lo, hi) if f[t] <= o + EPS) / (hi - lo)


def m4ss_steady_at_optimum(r, o, w=W):
    """M4ss: M4 over the steady-state window [T-W, T).   Needs: fragments."""
    f = series(r, 'fragments')
    if f is None:
        return None
    return m4_fraction_at_optimum(r, 0, o, lo=max(0, len(f) - w), hi=len(f))


def m6_settling_time(r, ts, o):
    """M6: settling time = (last tick t >= ts with fragments[t] > o) + 1 - ts;
    0 when the count never exceeds o after ts.  Censored (value T - ts,
    'censored': True) when fragments[T-1] > o.  Returns
    {'value': int, 'censored': bool} or None.   Needs: fragments (+ o)."""
    f = series(r, 'fragments')
    if f is None or o is None:
        return None
    T = len(f)
    if T <= ts:
        return None
    above = [t for t in range(ts, T) if f[t] > o + EPS]
    if not above:
        return {'value': 0, 'censored': False}
    return {'value': above[-1] + 1 - ts, 'censored': bool(f[T - 1] > o + EPS)}


def m7_teardowns_per_1000(r, ts):
    """M7: relay-link teardowns per 1000 post-event ticks =
    1000 * sum(relay_removed[ts+1:T]) / (T - ts - 1).   Needs: relay_removed
    (WP4 recording; None on pickles of the frozen tree)."""
    rr = series(r, 'relay_removed')
    if rr is None:
        return None
    T = len(rr)
    if T - ts - 1 <= 0:
        return None
    return 1000.0 * sum(num(rr[ts + 1:T])) / (T - ts - 1)


def m8_bridge_teardowns(r):
    """M8: bridge teardowns and idle-bridge teardowns (passes == 0) per run =
    whole-run sums of bridge_teardowns and idle_bridge_teardowns.  Returns
    {'bridge': n or None, 'idle': n or None}; None when both are absent.
    Needs: bridge_teardowns, idle_bridge_teardowns (WP4 recording)."""
    b, i = series(r, 'bridge_teardowns'), series(r, 'idle_bridge_teardowns')
    if b is None and i is None:
        return None
    return {'bridge': (sum(num(b)) if b is not None else None),
            'idle': (sum(num(i)) if i is not None else None)}


def m9_mode_flips_per_1000(r):
    """M9: relay-mode flips (executed mode) per 1000 ticks =
    1000 * sum(relay_mode_flips) / T over the whole run (actions are locked
    before the event, so pre-event flips are 0).   Needs: relay_mode_flips
    (WP4 recording)."""
    fl = series(r, 'relay_mode_flips')
    if fl is None or not fl:
        return None
    return 1000.0 * sum(num(fl)) / len(fl)


def m10_steady_fragment_changes(r, w=W):
    """M10: M1 over the steady-state window = number of ticks t in [T-W, T)
    with fragments[t] != fragments[t-1].   Needs: fragments."""
    f = series(r, 'fragments')
    if f is None:
        return None
    return _fragment_changes(f, len(f) - w, len(f))


def km_median(times, events):
    """Kaplan-Meier (product-limit) median of right-censored lifetimes: the
    smallest observed event time t at which S(t) <= 0.5, S(t) = prod over
    event times t_i <= t of (1 - d_i / n_i), d_i = completed lifetimes equal
    to t_i, n_i = records with lifetime >= t_i (a record censored at t_i is
    still at risk at t_i).  None when S never falls to 0.5 (too many censored
    records) or when there is no record.  times[i] = observed lifetime;
    events[i] = True for a completed lifetime, False for a censored one."""
    if not times:
        return None
    s = 1.0
    for t in sorted(set(t for t, e in zip(times, events) if e)):
        at_risk = sum(1 for x in times if x >= t)
        d = sum(1 for x, e in zip(times, events) if e and x == t)
        s *= 1.0 - d / at_risk
        if s <= 0.5 + EPS:
            return t
    return None


def bridge_lifetimes(recs, T):
    """Split the `was_bridge` records of a normalised relay_lifetimes list
    into (times, events, invalid).

    times[i]/events[i] are the observed lifetime and its completion flag of a
    USABLE record (see m11).  `invalid` holds the records that carry no
    lifetime at all and must not be counted -- neither as a completed nor as
    a censored one, because either reading would be a made-up number:

      * no start tick (`created` and `up` both None): the record cannot say
        when the bridge came up, so `removed - up` is unknown; dropping it
        silently would shrink n without a trace (the WP9 defect list, D7);
      * start tick outside [0, T), or not a real number at all (NaN, a
        string): `up == T` is the same off-by-one as `removed == T` -- the
        engine writing T instead of a real tick -- and `T - start` is then
        a censored lifetime of 0; `up > T` makes it NEGATIVE and `up < 0`
        makes it longer than the run.  Each of those is a made-up number
        that M11 would report as a median bridge lifetime, so the start is
        range-checked exactly as `removed` is (D7 residual);
      * `removed` outside [start, T): `removed >= T` is the engine writing T
        (or a later tick) instead of None for a bridge that is still up at
        the end of the run -- counting it turns a censored record into a
        completed lifetime of exactly T - start (TRACKB_HANDOFF section 2:
        `removed` must be None while the link is up, never T); `removed <
        start` is a record whose ticks are inconsistent.

    Each entry is [link_id, reason, created, up, removed]; check_invariants
    turns them into hard-assert failures."""
    times, events, invalid = [], [], []
    for x in recs:
        if not x.get('was_bridge'):
            continue
        start = x.get('up') if x.get('up') is not None else x.get('created')
        end = x.get('removed')
        if start is None:
            invalid.append([x.get('link_id'), 'no start tick (created and up are both None)',
                            x.get('created'), x.get('up'), end])
            continue
        if not is_num(start) or not (0 <= start < T):
            invalid.append([x.get('link_id'),
                            'start tick %r outside [0, T=%r) (a bridge that came up at or after the '
                            'end of the run, or before it began, carries no lifetime)' % (start, T),
                            x.get('created'), x.get('up'), end])
            continue
        if end is not None and not (start <= end < T):
            invalid.append([x.get('link_id'),
                            ('removed %r >= T=%r (a bridge still up at the end of the run must have '
                             'removed None)' % (end, T)) if end >= T else
                            'removed %r before the start tick %r' % (end, start),
                            x.get('created'), x.get('up'), end])
            continue
        if end is None:
            times.append(T - start)
            events.append(False)
        else:
            times.append(end - start)
            events.append(True)
    return times, events, invalid


def m11_median_bridge_lifetime(r, T=None):
    """M11: median bridge lifetime.  Over the run's relay_lifetimes records
    with was_bridge set, lifetime = removed - up (up falls back to created);
    a bridge still up at the end of the run is right-censored at T - up.
    The pre-registered headline 'median' is the Kaplan-Meier median
    (km_median: smallest completed lifetime at which the product-limit
    survival estimate <= 0.5; None when it never does, i.e. when censoring
    dominates -- never a substitute value).  Also reported, descriptively:
    'median_completed' = plain median over completed lifetimes only (biased
    upwards when long bridges are censored) and 'median_censored_at_T' =
    plain median with censored records entered at T - up (a lower bound; the
    rule of the first WP9 draft).  Returns {'median', 'median_completed',
    'median_censored_at_T', 'n', 'n_completed', 'n_censored', 'n_invalid',
    'invalid'} or None.  A bridge record without a start tick, with a start
    tick outside [0, T), or with `removed` outside [start, T), carries no
    lifetime: it is NOT counted in n
    and is reported under 'invalid' (bridge_lifetimes; the hard asserts turn
    it into a failure) instead of being dropped silently or read as a
    completed lifetime.
    Needs: relay_lifetimes (WP4 recording); T = run length (default: the
    run's fragments length or T_TOTAL)."""
    recs = lifetimes(r)
    if recs is None:
        return None
    if T is None:
        f = series(r, 'fragments')
        T = len(f) if f else T_TOTAL
    times, events, invalid = bridge_lifetimes(recs, T)
    done = [t for t, e in zip(times, events) if e]
    return {'median': km_median(times, events),
            'median_completed': (st.median(done) if done else None),
            'median_censored_at_T': (st.median(times) if times else None),
            'n': len(times), 'n_completed': len(done), 'n_censored': len(times) - len(done),
            'n_invalid': len(invalid), 'invalid': invalid[:20]}


def stability_metrics(r, ts, o, T=None):
    """All stability metrics of one run: {'M1': ..., 'M11': ...}; a metric
    whose series is absent is None."""
    return {
        'M1': m1_fragment_changes(r, ts),
        'M2': m2_policy_splits(r, ts),
        'M3': m3_optimum_losses(r, ts, o),
        'M4': m4_fraction_at_optimum(r, ts, o),
        'M4ss': m4ss_steady_at_optimum(r, o),
        'M6': m6_settling_time(r, ts, o),
        'M7': m7_teardowns_per_1000(r, ts),
        'M8': m8_bridge_teardowns(r),
        'M9': m9_mode_flips_per_1000(r),
        'M10': m10_steady_fragment_changes(r),
        'M11': m11_median_bridge_lifetime(r, T),
    }


# Post hoc lower bounds from the net series of the frozen tree.  They are NOT
# the metrics (a teardown followed by a re-creation in the same tick and links
# still in beam training are invisible); they are reported only under their
# own '*_lower_bound_*' names, and only when the true series is absent.
def _by_type_post_event(r, ty, ts):
    """Post-event total of one `overhead_by_type` member of a run, or None
    when the run records no value for that type after the event (S11; 'not
    recorded', never 0).

    'Not recorded' is not only the absent KEY.  A key whose post-event slice
    holds nothing -- an empty list, or a series shorter than ts, or a slice
    of NaN/None -- has no recorded value either, and sum([]) == 0 put a zero
    into the per-type median/mean exactly as the absent key did before D8.
    The run-length assert catches a truncated series in strict mode only,
    and the frozen (and every legacy) pickle must be analysed with
    --no-assert, where the assert records the violation and the 0 would
    still enter S11 (WP9 defect list, D5/D8 residual)."""
    ds = dict_series(r, OVERHEAD_BY_TYPE_KEY)
    if not ds or ty not in ds:
        return None
    vals = num(ds[ty][ts:])
    return sum(vals) if vals else None


def m7_lower_bound_relay_count(r, ts):
    """Lower bound on M7 from relay_count decrements: 1000 * sum over t in
    [ts+1, T) of max(0, relay_count[t-1] - relay_count[t]) / (T - ts - 1)."""
    rc = series(r, 'relay_count')
    if rc is None or len(rc) - ts - 1 <= 0:
        return None
    T = len(rc)
    return 1000.0 * sum(max(0, rc[t - 1] - rc[t]) for t in range(ts + 1, T)) / (T - ts - 1)


def m8_lower_bound_relay_bridges(r):
    """Lower bound on M8 (bridge teardowns) from relay_bridges decrements."""
    rb = series(r, 'relay_bridges')
    if rb is None:
        return None
    return sum(max(0, rb[t - 1] - rb[t]) for t in range(1, len(rb)))


# ------------------------------------------------- WP9: descriptive statistics
def quantile(vals, p):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * p
    f = math.floor(k); c = math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def dstats(vals):
    """Median, quartiles, mean, min, max over the recorded (non-None) values;
    {'status': 'not recorded'} when nothing is recorded.  Counts are
    heavy-tailed, so the median and IQR are the headline numbers."""
    vals = list(vals)
    rec = [x for x in vals if x is not None]
    if not rec:
        return {'status': NOT_RECORDED, 'n': len(vals), 'n_recorded': 0}
    return dict(n=len(vals), n_recorded=len(rec), mean=st.mean(rec), median=quantile(rec, 0.5),
                q25=quantile(rec, 0.25), q75=quantile(rec, 0.75), min=min(rec), max=max(rec))


# ------------------------------------------------- WP9: S9 stability family
def stability_family(diffs, alpha=ALPHA, min_nonzero=MIN_NONZERO, endpoints=S9_ENDPOINTS):
    """S9: Holm over the six endpoints of `endpoints`.  diffs[id] is the list
    of paired per-instance differences (MARL minus XL-DET), or None / a list
    containing None when the metric is not recorded.

    Rules: exact two-sided Wilcoxon; an endpoint with fewer than `min_nonzero`
    non-zero differences is NOT ASSESSABLE (the minimum exact two-sided p for
    n <= 5 is 0.0625); a not-recorded or not-assessable endpoint enters the
    Holm step-down with p = 1 (family size stays 6, the other endpoints are
    not touched by a p that cannot fall below 0.0625) and can never be
    confirmed.  Confirmed iff Holm-adjusted p < alpha and expected direction."""
    res, ps = {}, []
    for eid, k, metric, expect in endpoints:
        d = diffs.get(eid)
        base = dict(scenario=k, metric=metric, expect=expect, family_size=len(endpoints))
        if d is None or not d or any(x is None for x in d):
            n_missing = None if d is None else sum(1 for x in d if x is None)
            r = dict(base, status=NOT_RECORDED, n=(len(d) if d else 0), n_not_recorded=n_missing)
            p_in = 1.0
        else:
            r = paired(d, expect)
            r.update(base)
            if r['n_nonzero'] < min_nonzero:
                r['status'] = NOT_ASSESSABLE
                p_in = 1.0
            else:
                r['status'] = 'assessable'
                p_in = r['wilcoxon_p']
        r['holm_input_p'] = p_in
        res[eid] = r
        ps.append(p_in)
    adj = holm(ps)
    for (eid, _, _, _), h in zip(endpoints, adj):
        r = res[eid]
        if r['status'] == 'assessable':
            r['wilcoxon_p_holm'] = h
            r['confirmed'] = bool(h < alpha and r['direction_ok'])
        else:
            r['wilcoxon_p_holm'] = None
            r['confirmed'] = False
    return res


# ------------------------------------------------- WP9: hard asserts
def _numeric_list(v):
    """The per-tick series `v` as a list when it is one: a non-empty list,
    tuple or numpy array (tolist()) that holds at least one real number or
    None entry.  None for anything else (scalars, strings, dicts, lists of
    records such as relay_lifetimes).  Non-numeric entries of a series are
    kept so that the caller can count them (a mixed series is a violation,
    never a series to skip)."""
    if v is None or isinstance(v, (str, bytes, dict)):
        return None
    if hasattr(v, 'tolist'):
        v = v.tolist()
    if not isinstance(v, (list, tuple)) or not v:
        return None
    if not any(x is None or is_real(x) for x in v):
        return None
    return list(v)


def _nan_count(v):
    return sum(1 for x in v if x is None or (is_real(x) and math.isnan(x)))


def _non_numeric_count(v):
    return sum(1 for x in v if x is not None and not is_real(x))


def _site_count(r):
    """The number of agent sites a run claims: `n_agent_sites` when the
    recording carries it, else `n_radio_sites` (the postcard denominator).
    None when neither is a usable number."""
    for key in ('n_agent_sites', 'n_radio_sites'):
        v = r.get(key) if isinstance(r, dict) else None
        if is_num(v):
            return py_num(v)
    return None


def _postcards_per_site(r):
    """(minimum, number of sites the run reports a count for, badly recorded
    entries) of the per-site post-event postcard record; (None, None, []) when
    the run has neither `postcards_by_site` ({site: sends after the event})
    nor `postcards_min_per_site` (the WP5 invariant's minimum).

    The second element is the length of `postcards_by_site`, and None when
    the minimum comes from the scalar `postcards_min_per_site` (which cannot
    say how many sites it covers).  The caller MUST compare it with the run's
    site count: a min over the entries present is blind to exactly the defect
    the per-site check exists for -- a site that never sent and is therefore
    absent from a dict the engine fills from senders only (WP9 defect list,
    D2).

    The third element is [[site, repr(value)], ...] for every entry that is
    not a usable count (None, NaN, a string, a record).  The minimum is taken
    over the numeric entries only and is None when there is no numeric entry
    at all.  A neighbouring recording error must not decide the fate of this
    check by accident: min() over a dict holding a None raised an unlabelled
    TypeError from inside check_invariants and aborted the whole analysis,
    and an all-None dict made min() return None, which silently dropped the
    run from the per-site population and degraded the mode to 'aggregate'
    with no message (WP9 defect list, D3).  Both are now labelled per-site
    violations."""
    by_site = r.get('postcards_by_site') if isinstance(r, dict) else None
    if hasattr(by_site, 'items') and len(by_site):
        good, bad = [], []
        for site, v in by_site.items():
            if is_num(v):
                good.append(py_num(v))
            else:
                bad.append([str(site), repr(v)])
        return (min(good) if good else None), len(by_site), bad
    v = r.get('postcards_min_per_site') if isinstance(r, dict) else None
    return ((py_num(v), None, []) if is_num(v) else (None, None, []))


def _postcards_min_per_site(r):
    """The per-site minimum alone (see _postcards_per_site)."""
    return _postcards_per_site(r)[0]


def check_invariants(runs, opt, provenances, arms, t_sev=None, t_total=T_TOTAL,
                     nan_exempt=NAN_EXEMPT, marl_path_arms=MARL_PATH_ARMS, extra_runs=None,
                     marl_arm=None):
    """RERUN_SPEC WP9 hard asserts.  Returns {'failures': [...], 'checks': {...}};
    the caller raises when strict.

    runs[k][arm][seed] = run dict; opt[(kind, seed)] = manifest instance;
    provenances = [(label, record-or-None[, arms the pickle contributed])];
    extra_runs = {name: runs-like {k: {seed: run}}} (untrained controls)
    checked for the run-level invariants; marl_arm = the tested MARL arm
    (default arms[0]) whose checkpoint sha256 must agree across the pickles
    that carry it (controls, which run other checkpoints, are excluded)."""
    t_sev = t_sev or T_SEV
    marl_arm = marl_arm or arms[0]
    fails, checks = [], {}

    # 0. the tested arm is the pre-registered one.  --marl-arm selects the
    #    arm reported as MARL-RIC; the report must not be produced for any
    #    other arm (pre-registration R2 section 10).
    checks['prereg_marl_arm'] = {'expected': PREREG_MARL_ARM, 'marl_arm': marl_arm,
                                 'arms': list(arms)}
    if marl_arm != PREREG_MARL_ARM:
        fails.append('marl_arm is %r but the pre-registered MARL-RIC arm is %r: '
                     'pass --marl-arm %s' % (marl_arm, PREREG_MARL_ARM, PREREG_MARL_ARM))
    if PREREG_MARL_ARM not in arms:
        fails.append('the pre-registered arm %r is not among the analysed arms %s'
                     % (PREREG_MARL_ARM, list(arms)))

    # 1. every arm has exactly the manifest seeds
    man = {k: sorted(s for (kind, s) in opt if kind == KIND[k]) for k in SCEN}
    c = {'manifest_seeds': man, 'missing': {}, 'extra': {}}
    for k in SCEN:
        for arm in arms:
            have = set(runs[k].get(arm, {}))
            miss = sorted(set(man[k]) - have)
            extra = sorted(have - set(man[k]))
            if miss:
                c['missing']['%s/%s' % (k, arm)] = miss
                fails.append('arm %s scenario %s: missing seeds %s' % (arm, k, miss))
            if extra:
                c['extra']['%s/%s' % (k, arm)] = extra
                fails.append('arm %s scenario %s: seeds outside the manifest %s' % (arm, k, extra))
    for name, U in (extra_runs or {}).items():
        for k in SCEN:
            miss = sorted(set(man[k]) - set(U[k]))
            if miss:
                c['missing']['%s/%s' % (k, name)] = miss
                fails.append('control %s scenario %s: missing seeds %s' % (name, k, miss))
    # the manifest's own severance tick must be the pre-registered one
    c['manifest_severance_mismatch'] = []
    for (kind, s), inst in sorted(opt.items()):
        k = [kk for kk in SCEN if KIND[kk] == kind]
        if k and 'severance_tick' in inst and inst['severance_tick'] != t_sev[k[0]]:
            c['manifest_severance_mismatch'].append([kind, s, inst['severance_tick']])
            fails.append('manifest %s seed %s: severance_tick %r, expected %d' % (kind, s, inst['severance_tick'], t_sev[k[0]]))
    checks['arms_and_seeds'] = c

    # 2. provenance identical across pickles (D0.2 fields), PYTHONHASHSEED 0,
    #    pre-registered run length and severance tick when recorded
    c = {'records': {}, 'fields': {}, 'missing_fields': {}, 'checkpoint_sha256': {}}
    entries = [(e[0], e[1], (e[2] if len(e) > 2 else None)) for e in provenances]
    for label, rec, _ in entries:
        if not isinstance(rec, dict):
            c['records'][label] = None
            fails.append('provenance: %s has no provenance record' % label)
            continue
        c['records'][label] = {f: rec.get(f) for f in PROVENANCE_FIELDS + ('total_ticks', 'severance_tick')}
        if not rec.get('provenance_complete', True):
            fails.append('provenance: %s is INCOMPLETE (not tied to a commit)' % label)
        if rec.get('total_ticks') not in (None, t_total):
            fails.append('provenance: %s total_ticks=%r, expected %d' % (label, rec.get('total_ticks'), t_total))
        if rec.get('severance_tick') not in (None, t_sev['A']):
            fails.append('provenance: %s severance_tick=%r, expected %d' % (label, rec.get('severance_tick'), t_sev['A']))
    present = [(label, rec) for label, rec, _ in entries if isinstance(rec, dict)]
    for f in PROVENANCE_FIELDS:
        # every field must be present (a missing field is not "identical")
        # and identical across every record
        absent = [label for label, rec in present if rec.get(f) in (None, '')]
        if absent:
            c['missing_fields'][f] = absent
            fails.append('provenance: %s missing (None/empty) in %s' % (f, absent))
        vals = sorted(set(str(rec.get(f)) for _, rec in present))
        c['fields'][f] = vals
        if len(vals) > 1:
            fails.append('provenance: %s differs across pickles: %s' % (f, vals))
    if present and any(str(rec.get('pythonhashseed')) != '0' for _, rec in present):
        fails.append('provenance: PYTHONHASHSEED is not 0 in every pickle: %s' % c['fields']['pythonhashseed'])
    # the selected checkpoint (harness field checkpoint_sha256: one sha256 per
    # --checkpoint path) must be the same in every pickle that carries the
    # tested MARL arm; pickles whose contributed arms are unknown (controls)
    # are not compared
    for label, rec, arms_here in entries:
        if isinstance(rec, dict) and arms_here and marl_arm in arms_here:
            v = rec.get('checkpoint_sha256')
            if isinstance(v, (list, tuple)):
                v = sorted(str(x) for x in v)
            c['checkpoint_sha256'][label] = v
    ck = c['checkpoint_sha256']
    if ck:
        empty = [label for label, v in ck.items() if not v]
        if empty:
            fails.append('provenance: checkpoint_sha256 missing in %s (pickles carrying arm %s)' % (empty, marl_arm))
        if len(set(json.dumps(v, sort_keys=True) for v in ck.values())) > 1:
            fails.append('provenance: checkpoint_sha256 differs across the pickles carrying arm %s: %s' % (marl_arm, ck))
    checks['provenance'] = c

    # 3-6. run-level invariants
    def all_runs():
        # (scenario, arm label, seed, run, on the MARL code path?)
        for k in SCEN:
            for arm in runs[k]:
                for s, r in runs[k][arm].items():
                    yield k, arm, s, r, arm in marl_path_arms
        for name, U in (extra_runs or {}).items():     # controls run the MARL executor
            for k in SCEN:
                for s, r in U[k].items():
                    yield k, name, s, r, True

    stale = {'runs_with_key': 0, 'violations': []}
    post = {'runs_checked': 0, 'runs_with_per_site': 0, 'violations': [], 'per_site_violations': [],
            'incomplete_per_site': [], 'per_site_non_numeric': [], 'runs_with_site_coverage': 0,
            'min_mean_per_site': None, 'min_per_site': None, 'mode': None,
            'note': ('per-site: postcards_by_site must hold one entry per site (a site that never '
                     'sent appears with 0), every entry must be a count (a None/NaN/non-numeric '
                     'entry is reported under per_site_non_numeric) and min over it > 1; the '
                     'scalar postcards_min_per_site '
                     'is accepted as the minimum of the engine-side WP5 invariant but cannot be '
                     'checked for completeness; aggregate: mean per site (sum(postcards[t_sev:]) / '
                     'n_radio_sites) > 1 -- the per-site check is then the engine-side WP5 invariant')}
    nan = {'violations': [], 'non_numeric': [], 'exempt_counts': {}}
    length = {'violations': [], 'expected': t_total, 'n_series_checked': 0, 'runs_checked': 0,
              'note': 'every numeric per-tick series of a run, top-level and {name: series} member, '
                      'has exactly t_total entries; n_series_checked / runs_checked say how much was '
                      'compared, so an empty violations list is backed by a count instead of being '
                      'indistinguishable from "no series was ever examined" (and a future breakage '
                      'of _numeric_list shows up as a drop in the count, not as silence)'}
    lifet = {'runs_with_key': 0, 'violations': [],
             'note': 'relay_lifetimes bridge records that carry no lifetime (no start tick, a start '
                     'tick outside [0, t_total), or removed outside [start, t_total)); they are not '
                     'counted by M11'}

    def series_check(k, arm, s, key, v):
        """Per-tick series checks: run length, non-numeric entries, NaN/None."""
        lst = _numeric_list(v)
        if lst is None:
            return
        # run length: EVERY numeric per-tick series, not only conn.  A short
        # fragments / fragments_raw / event series silently shortens the
        # M1-M7/M10 windows (m2 truncates to min(len(f), len(raw)), m4 divides
        # by the shortened window), which is a partial-window value reported
        # as if it were the metric -- worse than "not recorded" (WP9 defect
        # list, D6).
        length['n_series_checked'] += 1
        if len(lst) != t_total:
            length['violations'].append([k, arm, s, key, len(lst)])
            fails.append('%s %s seed %s: series %s has %d entries, expected %d (run length)'
                         % (k, arm, s, key, len(lst), t_total))
        n_other = _non_numeric_count(lst)
        if n_other:
            nan['non_numeric'].append([k, arm, s, key, n_other])
            fails.append('%s %s seed %s: series %s has %d non-numeric entries' % (k, arm, s, key, n_other))
        n_bad = _nan_count(lst)
        if n_bad:
            if key in nan_exempt:
                nan['exempt_counts'][key] = nan['exempt_counts'].get(key, 0) + n_bad
            else:
                nan['violations'].append([k, arm, s, key, n_bad])
                fails.append('%s %s seed %s: series %s has %d NaN/None entries' % (k, arm, s, key, n_bad))

    for k, arm, s, r, marl_path in all_runs():
        for key in DIAG_SERIES:
            v = series(r, key)
            if v is not None:
                stale['runs_with_key'] += 1
                tot = sum(num(v))
                if tot != 0:
                    stale['violations'].append([k, arm, s, key, tot])
                    fails.append('%s %s seed %s: %s sums to %r, expected 0' % (k, arm, s, key, tot))
        pc = series(r, 'postcards')
        if pc is not None and marl_path:
            n_sites = py_num(r.get('n_radio_sites'))
            post['runs_checked'] += 1
            if not n_sites:
                post['violations'].append([k, arm, s, 'n_radio_sites missing'])
                fails.append('%s %s seed %s: postcards recorded but n_radio_sites missing' % (k, arm, s))
            else:
                tot = sum(num(pc[t_sev[k]:]))
                per_site = tot / n_sites
                post['min_mean_per_site'] = (per_site if post['min_mean_per_site'] is None
                                             else min(post['min_mean_per_site'], per_site))
                site_min, n_recorded, bad_sites = _postcards_per_site(r)
                # a badly recorded entry is a labelled per-site failure, not a
                # crash (min() over a None) and not a silent loss of the check
                # (an all-None dict left site_min None; WP9 defect list, D3)
                for site, val in bad_sites:
                    post['per_site_non_numeric'].append([k, arm, s, site, val])
                    fails.append('%s %s seed %s: postcards_by_site site %r holds %s, expected a count'
                                 % (k, arm, s, site, val))
                if site_min is not None:
                    post['runs_with_per_site'] += 1
                    post['min_per_site'] = site_min if post['min_per_site'] is None else min(post['min_per_site'], site_min)
                    if site_min <= 1:
                        post['per_site_violations'].append([k, arm, s, site_min, n_sites])
                        fails.append('%s %s seed %s: a site sent %s postcards after the event (per-site check; need > 1 per site)'
                                     % (k, arm, s, site_min))
                    # the minimum is only a per-site check when it covers every
                    # site: a silent site absent from postcards_by_site is
                    # exactly the WP5 defect this check exists for
                    n_agent = _site_count(r)
                    if n_recorded is not None:
                        post['runs_with_site_coverage'] += 1
                        if n_agent is None or n_recorded != n_agent:
                            post['incomplete_per_site'].append([k, arm, s, n_recorded, n_agent])
                            fails.append('%s %s seed %s: postcards_by_site holds %d site(s) but the run records %s '
                                         'site(s): the per-site postcard check needs one entry per agent site '
                                         '(a site that never sent must appear with 0)'
                                         % (k, arm, s, n_recorded, n_agent))
                if tot <= n_sites:
                    post['violations'].append([k, arm, s, tot, n_sites])
                    fails.append('%s %s seed %s: %d postcards after the event for %d sites '
                                 '(aggregate: mean per site must be > 1; the per-site check is the engine-side WP5 invariant)'
                                 % (k, arm, s, tot, n_sites))
        n_before = length['n_series_checked']
        for key, v in r.items():
            if key in DICT_SERIES_KEYS:
                continue
            series_check(k, arm, s, key, v)
        for key in DICT_SERIES_KEYS:
            ds = r.get(key)
            if hasattr(ds, 'items'):
                for name, v in ds.items():
                    series_check(k, arm, s, '%s/%s' % (key, name), v)
        if length['n_series_checked'] > n_before:
            length['runs_checked'] += 1
        recs = lifetimes(r)
        if recs is not None:
            lifet['runs_with_key'] += 1
            for bad in bridge_lifetimes(recs, t_total)[2]:
                lifet['violations'].append([k, arm, s] + list(bad))
                fails.append('%s %s seed %s: relay_lifetimes bridge record %r: %s'
                             % (k, arm, s, bad[0], bad[1]))
    if post['runs_checked']:
        post['mode'] = ('per-site' if post['runs_with_per_site'] == post['runs_checked']
                        else 'aggregate' if post['runs_with_per_site'] == 0 else 'mixed')
    checks['stale_edges'] = stale
    checks['postcards'] = post
    checks['nan'] = nan
    checks['run_length'] = length
    checks['lifetimes'] = lifet
    return {'failures': fails, 'checks': checks, 'n_failures': len(fails)}


# ------------------------------------------------------------------ load
def load_all(pickles, manifests, extra_arms=()):
    """Like load(), plus the list of (label, provenance record or None, the
    arms of the analysis that pickle contributed)."""
    runs = {k: {a: {} for a in ARMS + tuple(extra_arms)} for k in SCEN}
    prov = []
    for p in pickles:
        P = pickle.load(open(p, 'rb'))
        got = set()
        for k, sk in SCEN.items():
            for a in runs[k]:
                for s, r in P[sk].get(a, {}).items():
                    if r:
                        runs[k][a][int(s)] = r
                        got.add(a)
        label = os.path.basename(p)
        if any(label == e[0] for e in prov):
            label = p
        prov.append((label, P.get('provenance'), sorted(got)))
    opt = {}
    for m in manifests:
        for inst in json.load(open(m))['instances']:
            opt[(inst['scenario_kind'], int(inst['topology_seed']))] = inst
    return runs, opt, prov


def load(pickles, manifests, extra_arms=()):
    runs, opt, _ = load_all(pickles, manifests, extra_arms)
    return runs, opt


def load_control(path, arm):
    """One untrained-control pickle: {k: {seed: run}} for `arm`, plus its
    provenance record."""
    P = pickle.load(open(path, 'rb'))
    U = {k: {} for k in SCEN}
    for k, sk in SCEN.items():
        for s, r in P.get(sk, {}).get(arm, {}).items():
            if r:
                U[k][int(s)] = r
    return U, P.get('provenance')


def parse_controls(spec):
    """'NAME=PATH,NAME=PATH' or 'PATH,PATH' (name = file stem) -> [(name, path)]."""
    out = []
    for item in [x for x in (spec or '').split(',') if x]:
        name, _, path = item.partition('=')
        if not path:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        out.append((name, path))
    return out


# ------------------------------------------------------------------ analysis
def analyse(pickles, manifests, marl_arm='marl_freeze', strict=True, untrained=(),
            untrained_arm='marl_static', s8=None, nan_exempt=NAN_EXEMPT, t_sev=None,
            t_total=T_TOTAL):
    """The whole report as a dict.  strict=True raises AssertionError listing
    every failed hard assert before any statistic is computed."""
    global MARL, ARMS
    MARL = marl_arm
    ARMS = (MARL, XL, SDN) + ROUTING + ('random',)
    t_sev = t_sev or T_SEV
    extra = ('marl_freeze',) if MARL != 'marl_freeze' else ()
    runs, opt, prov = load_all(pickles, manifests, extra)
    # the adapt-then-freeze arm (secondary S7) is optional, but once any of
    # its runs is present it must be complete like every other arm
    freeze_present = bool(extra) and any(runs[k]['marl_freeze'] for k in SCEN)
    has_freeze = bool(extra) and all(runs[k]['marl_freeze'] for k in SCEN)
    controls = {}
    for name, path in untrained:
        U, p = load_control(path, untrained_arm)
        controls[name] = U
        prov.append((name, p, None))

    # ---- hard asserts (RERUN_SPEC WP9) before anything else
    inv = check_invariants(runs, opt, prov, ARMS + (('marl_freeze',) if freeze_present else ()),
                           t_sev=t_sev, t_total=t_total, nan_exempt=nan_exempt,
                           extra_runs=controls, marl_arm=MARL)
    inv['enforced'] = bool(strict)
    if strict and inv['failures']:
        raise AssertionError('hard asserts failed (%d):\n  %s' % (
            len(inv['failures']), '\n  '.join(inv['failures'])))

    rep = {'seeds': {}, 'excluded': {}, 'descriptive': {}, 'primary': {}, 'secondary': {}}
    seeds = {}
    for k in SCEN:
        common = set.intersection(*[set(runs[k][arm]) for arm in ARMS])
        every = set.union(*[set(runs[k][arm]) for arm in ARMS])
        seeds[k] = sorted(common)
        rep['seeds'][k] = seeds[k]
        rep['excluded'][k] = sorted(every - common)     # harness failures only

    def ach(k, arm, s):
        return steady(runs[k][arm][s]['conn'])

    def best_routing_ach(k, s):
        return max(ach(k, r, s) for r in ROUTING)

    def best_routing_jpu(k, s):
        return min(jpu(runs[k][r][s]) for r in ROUTING)

    # ---- descriptive, per arm
    for k in SCEN:
        S = seeds[k]
        d = {}
        for arm in ARMS:
            R = runs[k][arm]
            comps = [steady(R[s]['fragments']) for s in S]
            optk = [opt[(KIND[k], s)]['attainable_optimum'] for s in S]
            d[arm] = dict(
                ach_mean=st.mean(ach(k, arm, s) for s in S), ach_sd=st.stdev(ach(k, arm, s) for s in S),
                ach_median=st.median(ach(k, arm, s) for s in S),
                ach_min=min(ach(k, arm, s) for s in S), ach_max=max(ach(k, arm, s) for s in S),
                thr_mean=st.mean(steady(R[s]['delivered']) for s in S),
                energy_mean=st.mean(steady(R[s]['energy']) for s in S),
                energy_sd=st.stdev(steady(R[s]['energy']) for s in S),
                jpu_pooled=sum(sum(num(R[s]['energy'])) for s in S) / sum(sum(num(R[s]['delivered'])) for s in S),
                jpu_mean=st.mean(jpu(R[s]) for s in S),
                comps_mean=st.mean(comps),
                comps_reunifiable=(st.mean(c for c, o in zip(comps, optk) if o == 1)
                                   if any(o == 1 for o in optk) else None),
                at_optimum=sum(1 for c, o in zip(comps, optk) if c <= o + 1e-9),
                below_floor=sum(1 for s in S if ach(k, arm, s) < 10.0),
                pre_level=st.mean(wmean(R[s]['conn'], *(PRE_SURGE if k == 'B' else (0, T_EVENT['A']))) for s in S),
            )
            if k == 'A':
                v = [sli(R[s], T_EVENT['A']) for s in S]
                d[arm].update(sli_mean=st.mean(v), sli_median=st.median(v))
        rep['descriptive'][k] = d
        rep['descriptive'][k + '_optima'] = {
            'n': len(S),
            'reunifiable': sum(1 for s in S if opt[(KIND[k], s)]['attainable_optimum'] == 1),
            'mean_optimum': st.mean(opt[(KIND[k], s)]['attainable_optimum'] for s in S),
            'mean_raw': st.mean(opt[(KIND[k], s)]['raw_components'] for s in S),
        }

    # ---- primary family
    fam = []
    for k in ('A', 'B'):
        S = seeds[k]
        fam.append(('P%d_%s_ach_vs_xldet' % (1 if k == 'A' else 4, k),
                    [ach(k, MARL, s) - ach(k, XL, s) for s in S], +1))
        fam.append(('P%d_%s_ach_vs_sdn' % (2 if k == 'A' else 5, k),
                    [ach(k, MARL, s) - ach(k, SDN, s) for s in S], +1))
        fam.append(('P%d_%s_ach_vs_best_routing' % (3 if k == 'A' else 6, k),
                    [ach(k, MARL, s) - best_routing_ach(k, s) for s in S], +1))
    S = seeds['A']
    fam.append(('P7_A_jpu_vs_xldet', [jpu(runs['A'][MARL][s]) - jpu(runs['A'][XL][s]) for s in S], -1))
    fam.append(('P8_A_jpu_vs_sdn', [jpu(runs['A'][MARL][s]) - jpu(runs['A'][SDN][s]) for s in S], -1))
    fam.append(('P9_A_jpu_vs_best_routing', [jpu(runs['A'][MARL][s]) - best_routing_jpu('A', s) for s in S], -1))
    res = [paired(d, e) for _, d, e in fam]
    adj = holm([r['wilcoxon_p'] for r in res])
    for (name, _, _), r, h in zip(fam, res, adj):
        r['wilcoxon_p_holm'] = h
        r['confirmed'] = bool(h < ALPHA and r['direction_ok'])
        rep['primary'][name] = r

    # ---- secondary (not in the Holm family)
    sec = rep['secondary']
    for k in ('A', 'B'):
        S = seeds[k]
        sec['S1_%s_components_marl_minus_xldet' % k] = paired(
            [steady(runs[k][MARL][s]['fragments']) - steady(runs[k][XL][s]['fragments']) for s in S], +1)
    SB = seeds['B']

    def change(arm, s):
        c = runs['B'][arm][s]['conn']
        return steady(c) - wmean(c, *PRE_SURGE)

    def pre(arm, s):
        return wmean(runs['B'][arm][s]['conn'], *PRE_SURGE)

    sec['S2_B_change_vs_xldet'] = paired([change(MARL, s) - change(XL, s) for s in SB], +1)
    sec['S2_B_change_vs_sdn'] = paired([change(MARL, s) - change(SDN, s) for s in SB], +1)
    sec['S2_B_change_vs_best_routing'] = paired(
        [change(MARL, s) - max(change(r, s) for r in ROUTING) for s in SB], +1)
    sec['S3_B_pre_level_vs_xldet'] = paired([pre(MARL, s) - pre(XL, s) for s in SB], +1)
    sec['S3_B_pre_level_vs_sdn'] = paired([pre(MARL, s) - pre(SDN, s) for s in SB], +1)
    sec['S3_B_pre_level_vs_best_routing'] = paired([pre(MARL, s) - max(pre(r, s) for r in ROUTING) for s in SB], +1)
    SA = seeds['A']
    sec['S4_A_abs_energy_vs_xldet'] = paired(
        [steady(runs['A'][MARL][s]['energy']) - steady(runs['A'][XL][s]['energy']) for s in SA], +1)
    sec['S5_A_sli_vs_xldet'] = paired(
        [sli(runs['A'][MARL][s], T_EVENT['A']) - sli(runs['A'][XL][s], T_EVENT['A']) for s in SA], -1)
    sec['S6_B_jpu_vs_xldet'] = paired([jpu(runs['B'][MARL][s]) - jpu(runs['B'][XL][s]) for s in SB], -1)
    if has_freeze:
        for k in ('A', 'B'):
            S = [s for s in seeds[k] if s in runs[k]['marl_freeze']]
            sec['S7_%s_ach_%s_minus_marl_freeze' % (k, MARL)] = paired(
                [ach(k, MARL, s) - steady(runs[k]['marl_freeze'][s]['conn']) for s in S], +1)
            sec['S7_%s_components_%s_minus_marl_freeze' % (k, MARL)] = paired(
                [steady(runs[k][MARL][s]['fragments']) - steady(runs[k]['marl_freeze'][s]['fragments'])
                 for s in S], -1)
            rep['descriptive'][k]['marl_freeze (secondary)'] = dict(
                ach_mean=st.mean(steady(runs[k]['marl_freeze'][s]['conn']) for s in S),
                comps_mean=st.mean(steady(runs[k]['marl_freeze'][s]['fragments']) for s in S),
                jpu_pooled=sum(sum(num(runs[k]['marl_freeze'][s]['energy'])) for s in S)
                / sum(sum(num(runs[k]['marl_freeze'][s]['delivered'])) for s in S))
    pre_a = {arm: [wmean(runs['A'][arm][s]['conn'], 0, T_EVENT['A']) for s in SA] for arm in ARMS}
    sec['shared_world_A_pre_event_spread_pp'] = max(
        max(abs(pre_a[x][i] - pre_a[MARL][i]) for x in ARMS) for i in range(len(SA)))

    # ================================================================ WP9
    rep['config'] = {'W': W, 'T_SEV': dict(t_sev), 'T_TOTAL': t_total, 'ALPHA': ALPHA,
                     'MIN_NONZERO': MIN_NONZERO, 'marl_arm': MARL, 'arms': list(ARMS),
                     'nan_exempt': list(nan_exempt), 'pickles': list(pickles),
                     'manifests': list(manifests), 'S9_endpoints': [list(e) for e in S9_ENDPOINTS]}
    rep['asserts'] = inv

    # ---- per-run stability metrics (all arms, all seeds)
    def opt_of(k, s):
        return opt[(KIND[k], s)]['attainable_optimum']

    stab = {k: {arm: {s: stability_metrics(runs[k][arm][s], t_sev[k], opt_of(k, s), t_total)
                      for s in seeds[k]} for arm in ARMS} for k in SCEN}

    # ---- S9 stability family (Holm over 6): M2, M4, M7; MARL minus XL-DET
    diffs = {}
    for eid, k, metric, expect in S9_ENDPOINTS:
        d = []
        for s in seeds[k]:
            a, b = stab[k][MARL][s][metric], stab[k][XL][s][metric]
            d.append(None if a is None or b is None else a - b)
        diffs[eid] = d
    rep['stability'] = stability_family(diffs)

    # ---- S10 descriptive stability per arm
    s10 = {}
    for k in SCEN:
        s10[k] = {}
        for arm in ARMS:
            per = stab[k][arm]
            S = seeds[k]
            blk = {'per_seed': {str(s): per[s] for s in S}}
            for m in ('M1', 'M2', 'M3', 'M4', 'M4ss', 'M7', 'M9', 'M10'):
                blk[m] = dstats(per[s][m] for s in S)
            m6 = [per[s]['M6'] for s in S]
            blk['M6'] = dstats((x['value'] if x else None) for x in m6)
            blk['M6']['n_censored'] = sum(1 for x in m6 if x and x['censored'])
            blk['M8_bridge'] = dstats((x['bridge'] if x else None) for x in (per[s]['M8'] for s in S))
            blk['M8_idle'] = dstats((x['idle'] if x else None) for x in (per[s]['M8'] for s in S))
            m11 = [per[s]['M11'] for s in S]
            blk['M11'] = dstats((x['median'] if x else None) for x in m11)          # Kaplan-Meier per run
            blk['M11']['rule'] = 'Kaplan-Meier median per run; None when censoring keeps S(t) above 0.5'
            blk['M11']['n_bridges'] = sum(x['n'] for x in m11 if x)
            blk['M11']['n_completed'] = sum(x['n_completed'] for x in m11 if x)
            blk['M11']['n_censored'] = sum(x['n_censored'] for x in m11 if x)
            blk['M11']['n_runs_km_undefined'] = sum(1 for x in m11 if x and x['n'] and x['median'] is None)
            blk['M11']['n_invalid_records'] = sum(x.get('n_invalid', 0) for x in m11 if x)
            blk['M11_completed_only (descriptive)'] = dstats((x['median_completed'] if x else None) for x in m11)
            blk['M11_censored_at_T (lower bound)'] = dstats((x['median_censored_at_T'] if x else None) for x in m11)
            blk['monotone_reunification'] = (all(per[s]['M2'] == 0 for s in S)
                                             if all(per[s]['M2'] is not None for s in S) else None)
            if blk['M7'].get('status') == NOT_RECORDED:
                blk['M7_lower_bound_relay_count (post hoc, frozen-tree pickles)'] = dstats(
                    m7_lower_bound_relay_count(runs[k][arm][s], t_sev[k]) for s in S)
            if blk['M8_bridge'].get('status') == NOT_RECORDED:
                blk['M8_lower_bound_relay_bridges (post hoc, frozen-tree pickles)'] = dstats(
                    m8_lower_bound_relay_bridges(runs[k][arm][s]) for s in S)
            if k == 'B':
                # descriptive surge window [1400, T): M1-M4 with ts = surge tick
                sw = {}
                for s in S:
                    r = runs[k][arm][s]
                    sw[s] = {'M1': m1_fragment_changes(r, SURGE_B), 'M2': m2_policy_splits(r, SURGE_B),
                             'M3': m3_optimum_losses(r, SURGE_B, opt_of(k, s)),
                             'M4': m4_fraction_at_optimum(r, SURGE_B, opt_of(k, s))}
                blk['surge_window_%d' % SURGE_B] = {m: dstats(sw[s][m] for s in S) for m in ('M1', 'M2', 'M3', 'M4')}
            s10[k][arm] = blk
    rep['S10'] = s10

    # ---- S11 descriptive billed overhead (by type when recorded)
    s11 = {}
    for k in SCEN:
        s11[k] = {}
        for arm in ARMS:
            S = seeds[k]
            R = runs[k][arm]
            ts = t_sev[k]
            blk = {}
            tot = [(sum(num(series(R[s], 'overhead')[ts:])) if series(R[s], 'overhead') is not None else None) for s in S]
            blk['overhead_total_post_event'] = dstats(tot)
            blk['overhead_per_tick_post_event'] = dstats((x / (t_total - ts) if x is not None else None) for x in tot)
            types = sorted(set().union(*[set((dict_series(R[s], OVERHEAD_BY_TYPE_KEY) or {})) for s in S]))
            # `types` is the union over the seeds of the arm: a run that does
            # not record a type has NOT recorded it (None -> dstats counts it
            # under n_recorded), it did not bill zero bytes of it.  Summing
            # the absent key as 0 pulled the per-type median/mean towards 0
            # for every seed that never carried the type (WP9 defect list, D8).
            blk['by_type_post_event'] = {
                ty: dstats(_by_type_post_event(R[s], ty, ts) for s in S)
                for ty in types} if types else {'status': NOT_RECORDED}
            blk['postcards_total'] = dstats((sum(num(series(R[s], 'postcards'))) if series(R[s], 'postcards') is not None else None) for s in S)
            blk['energy_ctrl_total'] = dstats((sum(num(series(R[s], ENERGY_CTRL_KEY))) if series(R[s], ENERGY_CTRL_KEY) is not None else None) for s in S)
            s11[k][arm] = blk
    rep['S11'] = s11

    # ---- S12 untrained-actor control (descriptive, paired per instance)
    s12 = {}
    for name, U in controls.items():
        blk = {'summary': {}, 'paired': {}, 'fingerprint_audit': {}}
        for k in SCEN:
            S = [s for s in seeds[k] if s in U[k]]
            if not S:
                blk['summary'][k] = {'status': NOT_RECORDED}
                continue
            comps = [steady(U[k][s]['fragments']) for s in S]
            blk['summary'][k] = dict(
                n=len(S), ach_mean=st.mean(steady(U[k][s]['conn']) for s in S),
                comps_mean=st.mean(comps),
                at_optimum=sum(1 for s, c in zip(S, comps) if c <= opt_of(k, s) + 1e-9),
                jpu_pooled=sum(sum(num(U[k][s]['energy'])) for s in S) / sum(sum(num(U[k][s]['delivered'])) for s in S),
                M2=dstats(m2_policy_splits(U[k][s], t_sev[k]) for s in S),
                M4=dstats(m4_fraction_at_optimum(U[k][s], t_sev[k], opt_of(k, s)) for s in S),
                M7=dstats(m7_teardowns_per_1000(U[k][s], t_sev[k]) for s in S))
            bad = sum(1 for s in S if U[k][s].get('static_fp_start') is None
                      or U[k][s].get('static_fp_start') != U[k][s].get('static_fp_end'))
            blk['fingerprint_audit'][k] = {'bad_passes': bad, 'n': len(S)}
            pr = {}
            pr['ach_trained_minus_untrained'] = paired([ach(k, MARL, s) - steady(U[k][s]['conn']) for s in S], +1)
            pr['ach_untrained_minus_random'] = paired([steady(U[k][s]['conn']) - ach(k, 'random', s) for s in S], +1)
            pr['ach_untrained_minus_best_routing'] = paired(
                [steady(U[k][s]['conn']) - best_routing_ach(k, s) for s in S], +1)
            pr['components_untrained_minus_trained'] = paired(
                [steady(U[k][s]['fragments']) - steady(runs[k][MARL][s]['fragments']) for s in S], +1)
            if k == 'A':
                pr['jpu_untrained_minus_trained'] = paired([jpu(U[k][s]) - jpu(runs[k][MARL][s]) for s in S], +1)
            blk['paired'][k] = pr
        s12[name] = blk
    if len(controls) > 1:
        mean_blk = {}
        for k in SCEN:
            S = [s for s in seeds[k] if all(s in U[k] for U in controls.values())]
            if S:
                mean_blk[k] = {'n_controls': len(controls), 'ach_trained_minus_mean_untrained': paired(
                    [ach(k, MARL, s) - st.mean(steady(U[k][s]['conn']) for U in controls.values()) for s in S], +1)}
        s12['mean_over_controls'] = mean_blk
    rep['S12'] = s12

    # ---- S8 hash invariance (result of tools/hash_invariance_check.py)
    if s8 is not None:
        rep['S8'] = s8
    return rep


# ------------------------------------------------------------------ console
def _fmt_ds(x):
    if not x or x.get('status') == NOT_RECORDED:
        return '%-22s' % NOT_RECORDED
    return '%7.3g [%7.3g, %7.3g]' % (x['median'], x['q25'], x['q75'])


def print_report(rep, seeds=None):
    seeds = seeds or rep['seeds']
    MARL = rep.get('config', {}).get('marl_arm', 'marl_freeze')
    arms = rep.get('config', {}).get('arms') or list(rep['descriptive']['A'])
    sec = rep['secondary']
    print('instances: A=%d B=%d   excluded (harness failures): %s' % (
        len(seeds['A']), len(seeds['B']), rep['excluded']))
    for k in SCEN:
        o = rep['descriptive'][k + '_optima']
        print('\nScenario %s  (optimum 1 on %d/%d, mean optimum %.2f, mean raw %.2f)' % (
            k, o['reunifiable'], o['n'], o['mean_optimum'], o['mean_raw']))
        print('  %-12s %13s %6s %7s %8s %8s %6s %6s' % ('arm', 'ach', 'thr', 'E/tick', 'J/u pool', 'comps', 'atopt', 'pre'))
        for arm in arms:
            x = rep['descriptive'][k][arm]
            print('  %-12s %5.1f +- %4.1f %6.0f %7.1f %8.4f %8.2f %3d/%-2d %6.1f' % (
                arm, x['ach_mean'], x['ach_sd'], x['thr_mean'], x['energy_mean'], x['jpu_pooled'],
                x['comps_mean'], x['at_optimum'], len(seeds[k]), x['pre_level']))
        x = rep['descriptive'][k].get('marl_freeze (secondary)')
        if x:
            print('  %-12s %5.1f          (adapt-then-freeze, secondary)  J/u pool %.4f  comps %.2f' % (
                'marl_freeze', x['ach_mean'], x['jpu_pooled'], x['comps_mean']))
    print('\nPRIMARY: arm %s (Holm over 9, exact Wilcoxon):' % MARL)
    for name, r in rep['primary'].items():
        print('  %-28s %+8.4f  CI [%+.4f, %+.4f]  wins %2d/%d  sign %.4f  W %.5f (scipy %s)  Holm %.5f  %s' % (
            name, r['mean'], r['ci_lo'], r['ci_hi'], r['wins'], r['n'], r['sign_p'], r['wilcoxon_p'],
            'n/a' if r['wilcoxon_p_scipy'] is None else '%.5f' % r['wilcoxon_p_scipy'],
            r['wilcoxon_p_holm'], 'CONFIRMED' if r['confirmed'] else 'not confirmed'))
    print('\nSECONDARY:')
    for name, r in sec.items():
        if isinstance(r, dict):
            print('  %-32s %+9.4f  CI [%+.4f, %+.4f]  wins %2d/%d  sign %.4f  W %.5f' % (
                name, r['mean'], r['ci_lo'], r['ci_hi'], r['wins'], r['n'], r['sign_p'], r['wilcoxon_p']))
        else:
            print('  %-32s %s' % (name, r))
    if 'stability' not in rep:
        return
    print('\nS9 STABILITY FAMILY: %s minus xldet (Holm over %d, exact Wilcoxon; < %d non-zero differences = not assessable):'
          % (MARL, len(S9_ENDPOINTS), MIN_NONZERO))
    for eid, r in rep['stability'].items():
        if r['status'] == NOT_RECORDED:
            print('  %-4s %s %-4s  %s' % (eid, r['scenario'], r['metric'], NOT_RECORDED))
            continue
        line = '  %-4s %s %-4s %+9.4f  CI [%+.4f, %+.4f]  wins %2d/%d  nonzero %2d  W %.5f' % (
            eid, r['scenario'], r['metric'], r['mean'], r['ci_lo'], r['ci_hi'], r['wins'], r['n'],
            r['n_nonzero'], r['wilcoxon_p'])
        if r['status'] == NOT_ASSESSABLE:
            line += '  ' + NOT_ASSESSABLE
        else:
            line += '  Holm %.5f  %s' % (r['wilcoxon_p_holm'], 'CONFIRMED' if r['confirmed'] else 'not confirmed')
        print(line)
    print('\nS10 STABILITY (descriptive; median [q25, q75] over instances):')
    for k in SCEN:
        print('  Scenario %s' % k)
        print('  %-12s %-24s %-24s %-24s %-24s %-24s %s' % ('arm', 'M1', 'M2', 'M4', 'M7', 'M9', 'M6 censored'))
        for arm in arms:
            b = rep['S10'][k][arm]
            print('  %-12s %s %s %s %s %s %d/%d' % (
                arm, _fmt_ds(b['M1']), _fmt_ds(b['M2']), _fmt_ds(b['M4']), _fmt_ds(b['M7']), _fmt_ds(b['M9']),
                b['M6']['n_censored'], b['M6'].get('n', 0)))
    inv = rep.get('asserts', {})
    print('\nHARD ASSERTS: %s, %d failure(s)%s' % (
        'enforced' if inv.get('enforced') else 'NOT enforced (--no-assert)', inv.get('n_failures', 0),
        ':' if inv.get('failures') else ''))
    for f in inv.get('failures', [])[:40]:
        print('  ' + f)
    if len(inv.get('failures', [])) > 40:
        print('  ... %d more' % (len(inv['failures']) - 40))
    if rep.get('S12'):
        print('\nS12 UNTRAINED CONTROLS (descriptive): %s' % ', '.join(n for n in rep['S12'] if n != 'mean_over_controls'))
    if rep.get('S8') is not None:
        s8 = rep['S8']
        print('S8 hash invariance: %s (%s series compared, same_hashseed=%r)' % (
            'identical' if s8.get('identical') else 'NOT identical', s8.get('n_compared'),
            s8.get('same_hashseed')))
        # a check that compared nothing, or a pair that is not a
        # hash-invariance pair at all (equal or unknown PYTHONHASHSEEDs), is
        # never "identical" -- and the console line must say so rather than
        # leave it to a `note` nobody prints (WP9 defect list, D1 and its
        # residual).  same_hashseed must be False for this block to mean
        # anything: True is a repeat-run pair, None is unknowable.
        for reason in (s8.get('nothing_compared') or []):
            print('  S8 VACUOUS: ' + reason)
        for reason in (s8.get('precondition_not_established') or []):
            print('  S8 PRECONDITION NOT ESTABLISHED: ' + reason)
        if s8.get('note'):
            print('  S8 NOTE: ' + s8['note'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickles', required=True)
    ap.add_argument('--manifests', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--marl-arm', default=PREREG_MARL_ARM,
                    help='the arm reported as MARL-RIC; the hard asserts refuse any '
                         'value other than the pre-registered %s' % PREREG_MARL_ARM)
    ap.add_argument('--no-assert', action='store_true',
                    help='legacy pickles: report the hard-assert failures instead of stopping')
    ap.add_argument('--untrained', default='', help='NAME=PKL[,NAME=PKL] untrained-actor controls (S12)')
    ap.add_argument('--untrained-arm', default='marl_static')
    ap.add_argument('--s8', default=None, help='JSON written by tools/hash_invariance_check.py')
    ap.add_argument('--nan-exempt', default=','.join(NAN_EXEMPT),
                    help='series whose NaN entries mean "undefined this tick"')
    a = ap.parse_args()
    s8 = json.load(open(a.s8)) if a.s8 else None
    rep = analyse(a.pickles.split(','), a.manifests.split(','), marl_arm=a.marl_arm,
                  strict=not a.no_assert, untrained=parse_controls(a.untrained),
                  untrained_arm=a.untrained_arm, s8=s8,
                  nan_exempt=tuple(x for x in a.nan_exempt.split(',') if x))
    json.dump(rep, open(a.out, 'w'), indent=1)
    print_report(rep)
    print('\nwritten', a.out)


if __name__ == '__main__':
    main()
