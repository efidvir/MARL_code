# WP9 — Analysis and reporting (Track B)

Status: implemented 2026-09-22 on branch `trackB`; review fixes of the same
day (section 10); validated on <compute-server> against the frozen Amendment-1
reports (section 8).  Binding definitions for PREREGISTRATION_R2 sections
7–10.  Files: `confirmatory_analysis.py`, `dump_results_json.py`,
`extract_timelines.py`, `tools/compare_reports.py`,
`tools/hash_invariance_check.py`, `tools/wp9_validate.sh`,
`tests/test_wp9_*.py`.

## 1. Unchanged parts

P1–P9 (MARL = `--marl-arm`, windows, comparators, exact two-sided Wilcoxon
with zeros dropped and mid-ranks, Holm over 9, decision rule) and S1–S7 are
the code of the frozen tree, moved unchanged into `analyse()`.  The report's
`seeds`, `excluded`, `descriptive`, `primary` and `secondary` sections are
byte-for-byte the old ones (section 6).  New material lives in new top-level
keys only: `config`, `asserts`, `stability` (S9), `S10`, `S11`, `S12`, `S8`.

## 2. Notation and windows

`f` = `fragments`, `r` = `fragments_raw`, `o` = the instance's
`attainable_optimum` from the benchmark manifest, `T` = run length (3600),
`t_s` = severance tick (`T_SEV`: A 200, B 0), `W` = 500 (steady-state window
`[T-W, T)`).  The post-event window is `[t_s, T)`; change counts use
`t ∈ [t_s+1, T)` so that the severance transition itself is not counted.
Scenario B is counted from the severance at tick 0 (monotonicity view); the
surge window `[1400, T)` is reported descriptively only (`S10 ... surge_window_1400`).

## 3. Metric helpers (`confirmatory_analysis.py`, importable)

| Metric | Definition | Needs |
|---|---|---|
| M1 `m1_fragment_changes(r, ts)` | #{t ∈ [t_s+1, T): f[t] ≠ f[t-1]} | fragments |
| M2 `m2_policy_splits(r, ts)` | #{t ∈ [t_s+1, T): f[t] > f[t-1] and r[t] ≤ r[t-1]} | fragments, fragments_raw |
| M3 `m3_optimum_losses(r, ts, o)` | #{t ∈ [t_s+1, T): f[t-1] ≤ o < f[t]} | fragments, o |
| M4 `m4_fraction_at_optimum(r, ts, o)` | mean over t ∈ [t_s, T) of 1[f[t] ≤ o] | fragments, o |
| M4ss `m4ss_steady_at_optimum(r, o)` | M4 over [T-W, T) | fragments, o |
| M6 `m6_settling_time(r, ts, o)` | (last t ≥ t_s with f[t] > o) + 1 − t_s; 0 if never above; censored (value T−t_s, flag) when f[T-1] > o | fragments, o |
| M7 `m7_teardowns_per_1000(r, ts)` | 1000 · Σ relay_removed[t_s+1:T] / (T − t_s − 1) | relay_removed |
| M8 `m8_bridge_teardowns(r)` | whole-run sums of bridge_teardowns and idle_bridge_teardowns | both series |
| M9 `m9_mode_flips_per_1000(r)` | 1000 · Σ relay_mode_flips / T (whole run; actions are locked before the event) | relay_mode_flips |
| M10 `m10_steady_fragment_changes(r)` | #{t ∈ [T-W, T): f[t] ≠ f[t-1]} | fragments |
| M11 `m11_median_bridge_lifetime(r, T)` | **Kaplan–Meier median** over bridge records (was_bridge): lifetime = removed − up (up falls back to created), a bridge still up at T is right-censored at T − up; `median` = smallest completed lifetime t with the product-limit estimate S(t) ≤ 0.5 (`km_median`; a record censored at t is still at risk at t), `None` when S never reaches 0.5 (censoring dominates — reported, never substituted). Also returned, descriptive only: `median_completed` (plain median of completed lifetimes; biased upwards under censoring) and `median_censored_at_T` (plain median with censored records at T − up; a lower bound), with `n`, `n_completed`, `n_censored` | relay_lifetimes |

Every helper returns `None` when a series it needs is absent; the report
shows it as `not recorded`.  There is no silent fallback.

Numeric entries: `is_num` / `num` / `series` accept Python int/float and
numpy integer/floating scalars (`numbers.Real`) and convert numpy arrays with
`tolist()`; every numpy scalar is converted to a plain Python number
(`py_num`) so that sums, metrics and the JSON report are type-stable.  A
numeric type is never dropped from a sum (a per-tick list of `np.int64`
counts in full), and `extract_timelines.num` / `dump_results_json.num` are
the same test.  Two explicitly
named post-hoc lower bounds exist for the frozen-tree pickles only
(`m7_lower_bound_relay_count`, `m8_lower_bound_relay_bridges`; net-series
decrements) and appear in S10 under keys containing `lower_bound ... post hoc`
solely when the true metric is not recorded.

`stability_metrics(r, ts, o, T)` returns all eleven for one run and is what
`dump_results_json.py` writes per run (`stability` block).

## 4. S9 stability family

Endpoints (`S9_ENDPOINTS`): S9a A/M2 (>0), S9b B/M2 (>0), S9c A/M4 (<0),
S9d B/M4 (<0), S9e A/M7 (>0), S9f B/M7 (>0); differences are MARL − XL-DET
per instance; exact two-sided Wilcoxon; Holm over 6 (`stability_family`).

Rules implemented:
- **Not assessable** when fewer than `MIN_NONZERO = 6` differences are
  non-zero (the exact two-sided minimum for n ≤ 5 is 0.0625).
- **Not recorded** when a metric is `None` for any instance (e.g. M7 on
  pickles without the WP4 series).
- A not-assessable or not-recorded endpoint enters the Holm step-down with
  p = 1: the family size stays 6, it can never be confirmed, and the other
  endpoints' adjusted p-values are not touched by a p that cannot fall below
  0.0625.  (`holm_input_p` records what entered; `wilcoxon_p_holm` is null
  for such endpoints.)
- Confirmed iff Holm-adjusted p < 0.05 and the statistic has the expected
  direction.
- Interpretation flags: `S10[k][arm]['monotone_reunification']` is true only
  when M2 = 0 on every instance.

## 5. Other sections

- **S10** per scenario and arm: median, quartiles, mean, min, max of M1, M2,
  M3, M4, M4ss, M6 (+ `n_censored`), M7, M8 (bridge, idle), M9, M10, M11
  (+ `n_bridges`, `n_censored`), the per-seed values, and for B the surge
  window.  Counts are heavy-tailed, so medians/IQR are the headline numbers.
- **S11** descriptive overhead per arm: post-event total and per-tick billed
  overhead, by message type when `overhead_by_type` is recorded, postcards
  sent per run, `energy_ctrl` total when recorded.
- **S12** untrained-actor controls (`--untrained NAME=PKL,...`, arm
  `--untrained-arm`, default `marl_static`): per control the summary
  (achievability, components, at-optimum, pooled J/unit, M2/M4/M7), the
  static-fingerprint audit, and paired descriptive comparisons
  (trained − untrained, untrained − random, untrained − best routing,
  components, J/unit in A) plus trained − mean over controls.
- **S8** hash invariance: `tools/hash_invariance_check.py A.pkl B.pkl --require-decision --out s8.json`
  compares every per-tick series and per-pass scalar bit-exactly except the
  accounting series energy/latency (reported as jitter; `--strict` makes them
  count).  `--s8 s8.json` embeds the result.  The two pickles must come from
  the selected checkpoint on two instances under PYTHONHASHSEED 0 and 1; the
  tool REFUSES a pair whose two pickles do not record two DIFFERENT
  `pythonhashseed` values (equal, missing, or no provenance record at
  all): `identical` is false, the reason is in
  `precondition_not_established` and the exit code is 2, because such a
  pair is a repeat-run check and cannot expose a hash-order dependence
  however many series agree (§12, D1).
  **What is compared:** the harness of the frozen tree records no per-tick
  action series (`_pass_out` holds consequences only: `relay_mode_nodes`,
  `relay_count`, `fragments`, `delivered`, …).  The pre-registration's
  "identical decision series" therefore needs the per-tick digest
  `decision_digest` requested in `docs/TRACKB_HANDOFF.md` §1 (sha256 of the
  executed joint action of the tick); the tool compares it like any other
  series (also under `--keys`) and records under `decision_series` whether
  every compared MARL-path run carried it.  **Without the digest S8 checks
  decision consequences only**, and a hash-order-dependent tx-power or MCS
  decision whose only visible trace is in energy/latency would pass as
  identical, because the energy/latency jitter report cannot separate such a
  change from the known accounting jitter (`accounting_note` in the result).
  `--require-decision` (the study setting) makes a MARL-path run without the
  digest a missing series, i.e. NOT IDENTICAL.

## 6. Hard asserts (`check_invariants`)

Evaluated before any statistic; `--no-assert` records them under `asserts`
instead of stopping (legacy pickles).  Any failure lists every violation.

1. every arm of the analysis has exactly the manifest's seeds in both
   scenarios (missing or extra instances fail); every control likewise; the
   adapt-then-freeze arm `marl_freeze` (secondary S7) is optional, but as
   soon as any of its runs is present it must be complete too (a partial set
   fails; with `--no-assert` S7 uses the intersection and reports `n`); the
   manifest's own `severance_tick` equals `T_SEV`;
2. provenance: every pickle carries a record; each of `tree_head`,
   `tree_sha256`, `schema_id`, `pythonhashseed` is **present** (None or an
   empty string in any record fails — a field absent from every record is
   not "identical") and identical across pickles and controls (D0.2 fields);
   `pythonhashseed == '0'`; `total_ticks == 3600` and `severance_tick == 200`
   when recorded; no `INCOMPLETE` record; the harness field
   `checkpoint_sha256` (one sha256 per `--checkpoint` path) is present and
   identical across every pickle that contributed the tested MARL arm (the
   five four-seed batches), controls excluded (they run other checkpoints);
3. `stale_edges` and `stale_u2u_volume` sum to 0 in every run that carries
   them (numpy-typed series count in full);
4. postcards, for every run of a MARL-path arm (marl_static, marl_freeze,
   marl, xldet, random, the controls) that carries `postcards`: **per site**
   when the run records `postcards_by_site` ({site: sends after the event})
   or `postcards_min_per_site` — min over sites > 1 (the RERUN_SPEC WP5/WP9
   condition) **and**, for `postcards_by_site`, one entry per agent site
   (`len(postcards_by_site) == n_agent_sites`, falling back to
   `n_radio_sites`; `checks.postcards.incomplete_per_site`).  Completeness is
   part of the assert, not only of the handoff request: a site that never
   sent is exactly what the check exists to catch, and a dict the engine
   fills from senders leaves it out, so the minimum over the entries present
   passes.  The scalar `postcards_min_per_site` carries no site list and is
   accepted without a completeness claim (`runs_with_site_coverage` counts
   the runs where completeness could be checked).  Otherwise the
   **aggregate** Σ postcards[t_s:] > n_radio_sites (mean per site > 1),
   labelled as such in the failure text and in `checks.postcards.mode`; the
   per-site check is then the engine-side WP5 invariant.  Both are evaluated
   when both are possible;
5. no NaN/None entry in any numeric per-tick series — top-level lists,
   numpy arrays and the members of the `{name: series}` keys
   (`overhead_by_type`, `relay_removed_by_reason`, `relay_teardown_holds`,
   reported as `key/name`) — except `NAN_EXEMPT = ('latency',)` where NaN
   means "nothing delivered this tick"; the exempt counts are reported
   (`--nan-exempt` overrides); a series holding a non-numeric entry (a
   string, a record) next to numbers is a violation, never a series to skip;
6. run length: **every** numeric per-tick series of a run equals `T_TOTAL`
   (3600) — the top-level lists and numpy arrays and the `{name: series}`
   members, not `conn` alone (`checks.run_length.violations` records
   `[scenario, arm, seed, key, len]`).  A short `fragments`,
   `fragments_raw` or event series passed every assert before and silently
   shortened the M1–M7/M10 windows (m2 truncates to min(len(f), len(raw)),
   m4 divides by the shortened window), i.e. produced a partial-window
   number reported as the metric rather than "not recorded".  The block
   also records how much was compared — `checks.run_length.n_series_checked`
   and `runs_checked` — so that an empty `violations` list is backed by a
   count instead of being indistinguishable from "no series was ever
   examined": 9000 series over 360 runs = 25 per run on the frozen cf set
   (§12.1);
7. `relay_lifetimes`: every `was_bridge` record carries a lifetime — a start
   tick (`up`, falling back to `created`) that is a real number in
   **[0, T_TOTAL)**, and `removed` either None (still up: right-censored) or
   in [start, T_TOTAL).  A record without a start tick was dropped silently
   by M11 (n did not count it) and `removed == T` (the engine writing T
   instead of None, TRACKB_HANDOFF §2) was counted as a completed lifetime
   of exactly T − start; the start tick went unchecked entirely, so the same
   engine bug on the other field — `up == T` — was accepted as a censored
   observation worth 0 ticks, `up > T` as a **negative** lifetime and
   `up < 0` as one longer than the run, all of them entering n, n_censored,
   `km_median` and `median_censored_at_T` with `n_invalid` 0.  All of them
   are now failures (`checks.lifetimes.violations`) and none is counted by
   M11, which reports them as `n_invalid` / `invalid` (S10:
   `M11.n_invalid_records`).

On the frozen pickles of the A1 test, (2) fails (no provenance record) and
(4) fails (one postcard per site per episode — the defect the rerun fixes;
the aggregate form, since those pickles have no per-site count); this is
expected and is why `--no-assert` exists.

## 7. Validation on the frozen data (<compute-server>, `~/MARL_run_9b`)

Commands (`PYTHONHASHSEED=0 nice -n 19`; `bash tools/wp9_validate.sh` runs
all of them plus the M1/assert checks and the regression of the export and
timeline scripts, writing to `output/wp9_validation/`):

```
python3 confirmatory_analysis.py --no-assert --marl-arm marl_static \
  --pickles ~/MARL_run_5/output/cf_eval_80_99.pkl,~/MARL_run_6/output/cf_static_80_89.pkl,~/MARL_run_6/output/cf_static_90_99.pkl \
  --manifests benchmark_manifest_80_99.json --out output/wp9_validation/cf_report_wp9.json
python3 tools/compare_reports.py ~/MARL_run_6/confirmatory/cf_report_A1.json output/wp9_validation/cf_report_wp9.json
python3 confirmatory_analysis.py --no-assert --marl-arm marl_static \
  --pickles ~/MARL_run_5/output/ce_eval_50_59.pkl,~/MARL_run_5/output/ce_eval_70_79.pkl,~/MARL_run_6/output/dev_static_50_59.pkl,~/MARL_run_6/output/dev_static_70_79.pkl \
  --manifests benchmark_manifest.json,benchmark_manifest_70_79.json --out output/wp9_validation/dev_report_wp9.json
python3 tools/compare_reports.py ~/MARL_run_6/confirmatory/dev_report_A1.json output/wp9_validation/dev_report_wp9.json
```

## 8. Validation record (<compute-server>, 2026-09-22, tree 19af437, PYTHONHASHSEED=0, nice 19)

Outputs kept in `~/MARL_run_9b/output/wp9_validation/` (not tracked).

| check | result |
|---|---|
| `tests/run_tests.py wp9` | 36 passed, 0 failed |
| cf report vs `cf_report_A1.json` (seeds, excluded, descriptive, primary, secondary) | 665 leaves, 0 mismatches (exact, rtol 0) |
| dev report vs `dev_report_A1.json` | 665 leaves, 0 mismatches |
| M1, marl_static, Scenario A, seeds 80–99 | 7, 4, 1043, 4, 3, 1, 5, 3, 2, 39, 5, 895, 3, 2, 3, 923, 1, 4, 3, 15 (= manuscript) |
| M2 marl_static A | 3, 2, 521, 1, 1, 0, 2, 1, 1, 19, 2, 448, 1, 1, 1, 461, 0, 2, 1, 7 (median 1.5; XL-DET 1 on 8/20) |
| M4ss marl_static A | 1.0 on 11/20, 0.43 on seeds 91 and 95, 0 on 7/20 |
| M10 marl_static A | 154, 143, 143 on seeds 82, 91, 95; 0 elsewhere |
| M6 marl_static A | censored on 8/20; seed 95 settles at 3399 |
| M7 lower bound (relay_count decrements, post hoc) marl_static A | median 477.1 (XL-DET 0, random 1120) — M7 itself: not recorded |
| S9 on the A1 data (descriptive here, the family is for seeds 200–219) | S9a A/M2 assessable (15 non-zero, Holm 0.0003, direction as expected); S9c A/M4 assessable (17 non-zero, Holm 0.0001); S9b, S9d not assessable (4 non-zero); S9e, S9f not recorded |
| hard asserts with `--no-assert` on the A1 pickles | 163 failures = 3 × no provenance record + 160 × postcards ≤ 1 per site after the event (4 MARL-path arms × 20 × 2), i.e. exactly the two known properties of the frozen tree; 0 NaN violations (9,269 exempt latency NaN in routing arms); 0 stale-edge keys; run length 3600 everywhere |
| `extract_timelines.py` on the cf pickles vs `timelines_cf.json` | 129,643 leaves identical; 72 new keys (fragments_raw, relay_count, relay_bridges, not_recorded per arm) |
| `dump_results_json.py` on the cf pickles vs `ce_results_A1_cf.json` | 0 mismatches over every old leaf; new keys: `series` for the added SERIES_KEYS, `stability`, `provenance` (66 s) |
| `tools/hash_invariance_check.py cf_static_80_89 cf_corun_80_89 --arms marl_static` | 660 series identical (a repeat-run smoke of the tool; the real S8 pair needs the retrained checkpoint) |

Not validated (needs the engine track): M7–M9, M11, S11 by type and the
provenance/stale-edge/postcard asserts on real pickles, because no pickle
of the corrected tree exists yet; the unit tests cover them on synthetic
runs with the key contract of `docs/TRACKB_HANDOFF.md`.

### 8.1 Re-validation after the review fixes (<compute-server>, 2026-09-22 12:40–12:48 UTC, `tools/wp9_validate.sh`, PYTHONHASHSEED=0, nice 19)

Log `~/MARL_run_9b/output/wp9_validation/validate_20260922T123956Z.log`;
files as committed in the review-fix commit (section 10), copied to
`~/MARL_run_9b` before the sync.

| check | result |
|---|---|
| `tests/run_tests.py wp9` | 46 passed, 0 failed (10 new tests, section 10) |
| cf report vs `cf_report_A1.json` (seeds, excluded, descriptive, primary, secondary) | 665 leaves, 0 mismatches (exact, rtol 0) |
| dev report vs `dev_report_A1.json` | 665 leaves, 0 mismatches |
| M1, marl_static, Scenario A, seeds 80–99 | 7, 4, 1043, 4, 3, 1, 5, 3, 2, 39, 5, 895, 3, 2, 3, 923, 1, 4, 3, 15 (= manuscript) |
| hard asserts with `--no-assert` on the A1 pickles | 163 failures = 3 × no provenance record + 160 × aggregate postcard check (mode `aggregate`, min mean per site 0.95); 0 NaN violations, **0 non-numeric entries** (the new mixed-series check fires nowhere on the real pickles), 9,269 exempt latency NaN; `missing_fields` {} (no record present, so nothing to test); 0 stale-edge keys; run length 3600 everywhere |
| M11 on the A1 pickles | not recorded (no `relay_lifetimes`), rule text `Kaplan-Meier median per run` |
| `extract_timelines.py` vs the previous validated output `timelines_cf_wp9.json` | 226,983 leaves identical; 4 new keys (`unknown_arms`, `seeds_dropped` per scenario) |
| `dump_results_json.py` vs the previous validated output `ce_results_cf_wp9.json` | 16,898,503 leaves identical, 0 new keys (M11 is null on these pickles, so its new fields do not appear) |
| `tools/hash_invariance_check.py cf_static_80_89 cf_corun_80_89 --arms marl_static` | 660 series identical; `decision_series.coverage` = "consequences only: no per-tick decision series recorded" (the frozen tree; the study pair is run with `--require-decision`) |

## 9. How to run on the confirmatory set (seeds 200–219)

```
PYTHONHASHSEED=0 nice -n 19 python3 tools/hash_invariance_check.py output/s8_h0.pkl output/s8_h1.pkl --require-decision --out confirmatory/s8.json
PYTHONHASHSEED=0 nice -n 19 python3 confirmatory_analysis.py --marl-arm marl_static   --pickles output/cf_200_203.pkl,... --manifests benchmark_manifest_200_219.json   --untrained untrained_s1234=output/untrained_s1234_200_219.pkl,... --s8 confirmatory/s8.json   --out confirmatory/cf_report_R2.json > confirmatory/cf_report_R2.txt
```

Without `--no-assert` any failed hard assert stops the run and lists every
violation; the pre-registered response is one unchanged re-run.

## 10. Review fixes (2026-09-22)

Defects of the WP9 review and what changed (all in Track B files; the two
engine-side requests are in `docs/TRACKB_HANDOFF.md` §1/§5):

1. **Provenance fields absent everywhere passed as "identical"** (medium).
   `check_invariants` now fails for every `PROVENANCE_FIELDS` entry that is
   None or empty in any present record (`checks.provenance.missing_fields`),
   in addition to the identity test.  Test
   `test_provenance_fields_must_be_present_not_merely_identical`.
2. **numpy scalars dropped by `num()`; ndarrays skipped by the NaN assert**
   (medium).  `is_real` / `is_num` / `py_num` / `num` accept every
   `numbers.Real` (numpy integer and floating scalars included) and convert
   to plain Python numbers; `series()` / `dict_series()` do the same;
   `_numeric_list` runs the NaN check on `tolist()` output and a series with
   non-numeric entries next to numbers is a violation
   (`checks.nan.non_numeric`).  `extract_timelines.num` and
   `dump_results_json.num` are the same test; the dump converts numpy
   scalars so the JSON stays serialisable.  Tests
   `test_numeric_helpers_count_every_numeric_type`,
   `test_numpy_scalars_and_arrays_are_checked_never_dropped`,
   `test_dump_and_timelines_accept_numpy_and_stay_serialisable`.
3. **S8 compared consequences only while the handoff said nothing else was
   needed** (medium).  `decision_digest` (per-tick sha256 of the executed
   joint action) requested in the handoff with its canonical form and hook;
   `tools/hash_invariance_check.py` always compares it when present, reports
   `decision_series.coverage`, states the consequence of the energy/latency
   jitter rule (`accounting_note`) and has `--require-decision` (the study
   setting).  Section 5 and PREREGISTRATION_R2 §10 say what S8 covers
   without the digest.  Test
   `test_hash_invariance_decision_coverage_and_require_decision`.
4. **Postcard assert was an aggregate mean presented as per-site** (medium).
   Per-site check when `postcards_by_site` or `postcards_min_per_site` is
   recorded (min over sites > 1, requested in the handoff); the aggregate
   check is labelled "aggregate: mean per site must be > 1; the per-site
   check is the engine-side WP5 invariant" in the failure text, in
   `checks.postcards.mode` and in section 6.  Test
   `test_postcards_per_site_when_recorded_and_aggregate_labelled_otherwise`.
5. **M11 median biased low by censored bridges entered at T − up** (low).
   Kaplan–Meier median (`km_median`) is the pre-registered value, null when
   censoring keeps S(t) above 0.5; the completed-only median and the old
   lower-bound median are reported next to it as descriptive companions
   (section 3, PREREGISTRATION_R2 §7.2, S10 keys `M11`,
   `M11_completed_only (descriptive)`, `M11_censored_at_T (lower bound)`).
   Tests `test_km_median_rules`, `test_m11_kaplan_meier_median_with_censoring`.
6. **`extract_timelines.py` took the seed intersection over unknown arms
   too** (low).  The instance set and run length come from the known arms
   only; unknown arms are summarised over their own seeds within that set
   and listed in `unknown_arms`; seeds an arm has beyond the set are
   recorded in `seeds_dropped` and printed.  Test
   `test_timelines_unknown_or_partial_arm_never_shrinks_the_seed_set`.
7. **`marl_freeze` completeness and checkpoint identity not asserted** (low).
   When any `marl_freeze` run is present the arm must be complete; the
   harness field `checkpoint_sha256` must be present and identical across
   the pickles that contributed the tested MARL arm (`load_all` now records
   which arms each pickle contributed; controls are excluded).  Tests
   `test_marl_freeze_must_be_complete_once_present`,
   `test_checkpoint_sha256_identical_across_pickles_carrying_the_tested_arm`.
8. **`{name: series}` members never NaN-checked** (low).  The NaN loop walks
   `overhead_by_type`, `relay_removed_by_reason` and `relay_teardown_holds`
   member by member (reported as `key/name`).  Covered by test 2's case.

Reproduction of the frozen reports is unchanged by every fix (section 8.1).

## 11. Defect round (2026-09-23): safety nets that verified nothing

Nine defects from `defects_WP9.txt`, four of them in the checks themselves
(D1, D2, D4, D6 — a pass that was vacuous, blind, a crash or absent).  Scope:
`confirmatory_analysis.py`, `extract_timelines.py`, `dump_results_json.py`,
`tools/hash_invariance_check.py` and their tests.  No engine file was
touched.

| # | what was wrong | fix | test |
|---|---|---|---|
| D1 | S8 reported IDENTICAL (exit 0) with `n_compared == 0`: a `--seeds`/`--arms` typo, pickles without a top-level `seeds` key, or an empty scenario selection gave a vacuous pass of a pre-registered check, and `--s8` then printed "S8 hash invariance: identical" | `identical` now also requires that something was compared (and, under `--require-decision`, that a decision series was); `nothing_compared` in the JSON carries the reason, the CLI prints `VACUOUS …` and exits **2** | `test_hash_invariance_never_passes_when_nothing_was_compared` |
| D2 | the per-site postcard check took the min over the entries **present**, so a silent site absent from `postcards_by_site` passed — the WP5 defect class the check was added for | the per-site branch requires one entry per agent site (`n_agent_sites`, else `n_radio_sites`); mismatch is a labelled failure under `checks.postcards.incomplete_per_site` | `test_postcards_per_site_must_cover_every_site` |
| D3 | `extract_timelines.extract()` crashed at `json.dump` on any numpy-int event series (binned sums stay `np.int64`) — a contract-compliant recording (TRACKB_HANDOFF §1) killed the whole timeline export | `binned()` converts every entry with `py_num`, and the per-bin mean/median/q25/q75 are converted again on the way out | `test_timelines_extract_on_numpy_runs_is_json_serialisable`, extended `…accept_numpy_and_stay_serialisable` |
| D4 | `dump_results_json.summarize()` treated an all-numpy-scalar dict (`postcards_by_site`, a numeric `topology_info`) as a {name: series} member, because `hasattr(np.int64, 'tolist')` is true, and aborted the whole dump at `list(4)` | the by_key branch now needs `isinstance(x, (list, tuple))` or (`tolist` **and not** `CA.is_real(x)`); numpy-scalar dicts take the `dict` branch through `py_num` | `test_dump_summarize_all_numpy_dict_is_a_dict_not_a_series_set` |
| D5 | numpy lifetime ticks flowed through `lifetime_record()` unconverted, so M11 / `km_median` returned numpy scalars and the stability block and report JSON failed to serialise | `lifetime_record()` applies `py_num` to `created, up, removed, load` (`LIFETIME_NUMERIC`) | numpy case in `test_m11_kaplan_meier_median_with_censoring` |
| D6 | the run-length assert checked `conn` only; a short `fragments`/event series silently shortened the M1–M7/M10 windows | every numeric per-tick series (top-level and `{name: series}` member) must have `t_total` entries; violations record the key | `test_every_numeric_series_has_the_run_length` |
| D7 | a bridge record with no start tick was dropped silently by M11; `removed == T` was counted as a completed lifetime | `bridge_lifetimes()` refuses both (no start tick; `removed` outside `[start, T)`), M11 reports `n_invalid`/`invalid` and the hard asserts fail on them | `test_m11_refuses_records_that_carry_no_lifetime`, `test_bridge_lifetime_records_without_a_lifetime_are_a_violation` |
| D8 | S11 `by_type_post_event`: a type absent from one run contributed `sum([])` = 0 instead of "not recorded" | `_by_type_post_event()` returns None when the run does not carry the type; `dstats` then reports `n_recorded` | `test_s11_by_type_absent_in_one_run_is_not_recorded_not_zero` |
| D9 | nothing pinned the pre-registered constants (`T_SEV['B']`, `T_TOTAL`, `W`, `MIN_NONZERO`, `ALPHA`, `S9_ENDPOINTS`); the metric tests all run on synthetic overrides | a constants test asserts them value for value against PREREGISTRATION_R2 §§7.2, 8, 9 | `test_pre_registered_constants_are_the_ones_in_the_plan` |

Decisions where the implementation differs from the literal required fix:

- **D1, `--require-decision`.**  The required fix makes `identical` require
  `decision_series.runs_with_key > 0`, which is implemented.  The *exit
  code 2 / "nothing compared"* label, however, is raised only when the
  selection holds **no MARL-path run at all**.  When MARL-path runs are
  present without the digest the tool already records one missing
  `decision_digest` series per run and exits 1: that is a finding, not a
  vacuous check, and it is the expected outcome on today's pickles (the
  engine request in TRACKB_HANDOFF §1 is still open).  Collapsing it into
  "nothing compared" would have hidden a real result behind a usage error.
- **D2, which site count.**  The completeness comparison prefers
  `n_agent_sites` when the recording carries it and falls back to
  `n_radio_sites` (the aggregate check's denominator), as the defect text
  allows; the scalar `postcards_min_per_site` is accepted without a
  completeness claim, because it carries no site list — `checks.postcards`
  reports `runs_with_site_coverage` so the report says how many runs could
  be checked at all.
- **D7, where the violation is raised.**  M11 does not raise: it returns
  `n_invalid`/`invalid` and excludes the records, and `check_invariants`
  turns them into hard-assert failures (`checks.lifetimes`).  A metric
  helper that raises would make `--no-assert` — the mode in which the
  frozen pickles are analysed — unable to produce a report at all, and the
  hard asserts are the pre-registered place for "this recording is wrong".

Reproduction of the frozen reports is unchanged by every fix: the new
asserts fire nowhere on the A1 pickles (every numeric series there is 3600
long; no pickle carries `relay_lifetimes`, `postcards_by_site` or
`overhead_by_type`), and the report sections they touch (`asserts`, `S10`,
`S11`) are sections the frozen reports do not contain.  Record in §11.1.

### 11.1 Validation record (<compute-server>, 2026-09-23, `tools/wp9_validate.sh`, PYTHONHASHSEED=0, nice 19, 8 workers)

Full script at tree `68f3615`; the two frozen-report reproductions and both
test suites repeated at `5147128` (a local rename only) with the same result.

| check | result |
|---|---|
| cf report vs `~/MARL_run_6/confirmatory/cf_report_A1.json` | **665 leaves, 0 mismatches, 0 new keys** in `seeds`/`excluded`/`descriptive`/`primary`/`secondary`; new top-level keys only (`config`, `asserts`, `stability`, `S10`, `S11`, `S12`) |
| dev report vs `dev_report_A1.json` | **665 leaves, 0 mismatches, 0 new keys** |
| M1 counts, `marl_static`, A, seeds 80–99 | MATCH the manuscript list |
| hard asserts with `--no-assert` on the A1 pickles | 163 failures = 3 × no provenance record + 160 × aggregate postcard check — unchanged by this round; **0 run-length violations** (the broadened D6 assert fires nowhere: every numeric series of all 720 frozen runs is 3600 long), 0 lifetime violations (no pickle carries `relay_lifetimes`), postcards mode `aggregate`, min mean per site 0.9545 |
| `extract_timelines.py` vs the previous validated output | 226,983 leaves, 0 mismatches |
| `dump_results_json.py` vs the previous validated output | 16,898,503 leaves, 0 mismatches, 0 new keys (M11 is null on these pickles, so `n_invalid` adds no leaf) |
| S8 smoke on the repeat-run pair | compared=660, 0 different, 0 missing → IDENTICAL (a real pass: `n_compared` > 0) — **this row was wrong, see §12**: `n_compared` > 0 makes the comparison non-empty, not the *pair* a hash-invariance pair.  Neither of those two pickles records a `pythonhashseed` at all, so nothing about hash order was verified and "IDENTICAL" was the same vacuous pass D1 was raised against.  Round 3 makes the differing-seed precondition part of the verdict; the step now reads exit **2**, "precondition not established" |
| S8 vacuity, real pickles | `--seeds 999` → exit **2** "nothing compared"; `--arms marl_statc` → exit **2**; `--arms marl_static` → exit 0; `--arms marl_static --require-decision` → exit **1** (the digest is genuinely missing: a finding, not a vacuous check) |
| `tests/run_tests.py wp9` / `wp12` | 57 passed / 43 passed, 0 failed (44 wp12 at `5147128`: the concurrent WP12 process, which shares this tree, added one) |

## 12. Defect round 3 (2026-09-23): the residuals of round 2's own safety nets

Six defects from `r3_WP9.txt`.  Two mediums (D1, D2 below) are residuals of
the nets §11 added: each net stopped one instance of a failure class and
left the symmetric one open — `n_compared > 0` without the differing-seed
precondition, and `removed` range-checked while the start tick was not.
Three lows are the same shape one level down (a check that cannot survive a
neighbouring recording error, a check that records no coverage, a "not
recorded" rule that only covers the absent key), and one is in the script
that certifies the round.  Scope: `confirmatory_analysis.py`,
`tools/hash_invariance_check.py`, `tools/wp9_validate.sh`, their tests and
these docs.  No engine file was touched.

| # | severity | what was wrong | fix | test |
|---|---|---|---|---|
| D1 | MEDIUM | S8's differing-hash-seed **precondition** was never part of the verdict: `same_hashseed` set an advisory `note` that nothing printed, `identical` required only `n_compared > 0`, and `print_report`'s S8 block printed `identical`/`n_compared`/`nothing_compared` only.  `output/wp9_validation/s8_smoke_repeat.json` — two pickles, *neither* recording a `pythonhashseed` — therefore read `identical True, 660 series compared`, exit 0, and the pre-registered S8 line said "identical" for a pair that could not have exposed a hash-order dependence at all | `identical` now also requires `same_hashseed is False` (both provenance records present, both carrying `pythonhashseed`, values differing).  Equal or unknown seeds put the reason in the new `precondition_not_established` list, make `identical` false and exit **2**.  `same_hashseed` is three-valued (True / False / None = not knowable).  The report's S8 line prints `same_hashseed`, the refusal and the `note` | `test_hash_invariance_refuses_a_pair_that_cannot_show_hash_invariance` (both seeds `'0'`; no provenance record; provenance without the field; the field present but null; one side only), updated `test_hash_invariance_cli_and_same_seed_note` |
| D2 | MEDIUM | `bridge_lifetimes()` validated `removed` against `[start, T)` but never validated the **start tick**.  `up`/`created` outside `[0, T)` was accepted as a usable censored observation and `times.append(T - start)` produced a zero (`up == T`, the same off-by-one the round refused for `removed == T`), a **negative** (`up > T`) or an over-long (`up < 0`) lifetime, counted in `n`/`n_censored` and fed to `km_median` and `median_censored_at_T`, with `n_invalid` 0 and `checks.lifetimes.violations` empty — the new net asserting "every bridge record carries a lifetime" about a record that carries none | after `start` is resolved, a start that is not a real number in `[0, T)` (NaN and non-numerics included) gets its own `invalid` entry, `start tick %r outside [0, T=%r)`, exactly as the no-start and `removed` kinds do: excluded from `times`/`events`, reported by M11 as `n_invalid`, and turned into a hard-assert failure by `check_invariants` | `test_m11_refuses_records_that_carry_no_lifetime` (`up == T`, `up > T`, `up < 0`, NaN, a string; and the valid boundaries 0 and T−1), `test_bridge_lifetime_records_without_a_lifetime_are_a_violation` |
| D3 | LOW | `_postcards_per_site` could not survive a neighbouring recording error: one `None` in `postcards_by_site` made `min()` raise an unlabelled `TypeError` from inside `check_invariants`, aborting the whole confirmatory analysis instead of producing the labelled per-site failure the assert exists for; an all-`None` dict was worse — `min()` returned `None`, the run left `runs_with_per_site`/`runs_with_site_coverage` and the mode degraded to `aggregate` with no message | the minimum is taken over the entries that are real numbers; every other entry is named in the third return value and reported as `checks.postcards.per_site_non_numeric` plus a failure `postcards_by_site site %r holds %r, expected a count`.  The helper returns `(min, entries recorded, bad entries)` | `test_postcards_per_site_entry_that_is_not_a_count_is_a_labelled_failure` (one `None`, an all-`None` dict, a mixed dict; through the analysis in both strict and `--no-assert` mode) |
| D4 | LOW | the run-length check recorded no coverage, so `checks.run_length: {'violations': [], 'expected': 3600}` was indistinguishable from "no series was ever examined" — the defect class this round exists to remove, in the check whose entire result on the frozen data is an empty violations list.  Its two sibling checks added in the same commit *do* report coverage | `n_series_checked` is incremented in `series_check` next to the length comparison and `runs_checked` per run that contributed at least one series; both are reported in the block.  An empty violations list is now backed by a count, and a future breakage of `_numeric_list` shows up as a drop in it instead of as silence | `test_every_numeric_series_has_the_run_length` (exact counts on the clean and the corrupted set, plus an independent recount of the per-run series list) |
| D5 | LOW | `_by_type_post_event` returned `None` only when the type **key** was absent.  A key present with an empty post-event slice — an empty list, a series shorter than `t_sev`, a slice of NaN — still returned `sum(num([])) == 0`, so the run contributed a zero to the per-type median/mean exactly as before D8.  The run-length assert covers the truncated case in strict mode only, and the frozen (and every legacy) pickle is analysed with `--no-assert` | `vals = num(ds[ty][ts:]); return sum(vals) if vals else None` — an empty or truncated per-type series is "not recorded", never 0 | `test_s11_by_type_absent_in_one_run_is_not_recorded_not_zero` (empty list, short series, all-NaN slice, and through the analysis: `n_recorded` drops, the per-type `min` does not) |
| D6 | LOW | step 5 of `tools/wp9_validate.sh` — the timelines/dump export regression whose `rc=0` the round-2 report cites as evidence — was wrapped in `if [ -f "$OUT/…json" ]`.  With the reference absent (a fresh tree, a cleaned output directory, a different `WP9_OUT`) the comparison was skipped entirely and the script still printed `done rc=0`: success reported having verified nothing, the same class as D1, in the script that certifies the round | a missing reference prints `STEP 5 NOT VERIFIED: no reference at …` and sets `rc=1`.  Re-baselining is now an explicit, separate act: `WP9_REBASELINE=1 bash tools/wp9_validate.sh` adopts this run's outputs as the references *after* the comparison has been printed | exercised in §12.1 (both references renamed away → `rc=1` and the NOT VERIFIED lines; restored → the comparison lines return) |

Decisions where the implementation is not the literal required fix, and the
accepted residue, are in `docs/DECISIONS.md` under the round-3 WP9 entry.

### 12.1 Validation record (<compute-server>, 2026-09-23, PYTHONHASHSEED=0, nice 19, MARL_POOL_WORKERS=8)

Every figure below was measured at tree `72d0b2c` (the round-3 code commit);
the commit that adds this record changes documentation only, and the two
frozen-report reproductions and both test suites were repeated at it with
identical results.  Full log: `output/wp9_validate_r3.log` on the server.

| check | result |
|---|---|
| cf report vs `~/MARL_run_6/confirmatory/cf_report_A1.json` | **665 leaves, 0 mismatches, 0 new keys** in `seeds`/`excluded`/`descriptive`/`primary`/`secondary`; new top-level keys only (`config`, `asserts`, `stability`, `S10`, `S11`, `S12`) |
| dev report vs `dev_report_A1.json` | **665 leaves, 0 mismatches, 0 new keys** |
| M1 counts, `marl_static`, A, seeds 80–99 | MATCH the manuscript list |
| hard asserts with `--no-assert` on the A1 pickles | cf **163** = 3 × no provenance record + 160 × aggregate postcard check; dev **164** = 4 + 160 — unchanged by this round, same two kinds and no third |
| **D4** `checks.run_length` coverage | `n_series_checked` **9000**, `runs_checked` **360**, 0 violations — 25 numeric series per run, which is exactly the figure the defect list had to establish by an independent scan because the block recorded none.  The empty violations list is now backed by a count |
| **D2** `checks.lifetimes` on the frozen data | `runs_with_key` 0: no A1 pickle carries `relay_lifetimes`, and a direct scan of the three cf pickles finds **0 bridge records** and therefore 0 start ticks outside [0, 3600).  The start-tick refusal cannot move a frozen number; it is exercised on synthetic records by the two tests |
| **D3** `checks.postcards.per_site_non_numeric` | `[]` on both sets (no frozen pickle records `postcards_by_site`); mode `aggregate`, 160 runs checked, unchanged |
| **D5** empty per-type slices on the frozen data | a scan of the three cf pickles finds **0** `overhead_by_type` members whose post-event slice holds no recorded value, so "not recorded instead of 0" cannot move a frozen S11 number either |
| `extract_timelines.py` vs the reference | 226,983 leaves, 0 mismatches, **new-keys = 0** — the stale baseline the previous round left as an open issue is re-baselined (`WP9_REBASELINE=1`) and the plain run now reads 0 |
| `dump_results_json.py` vs the reference | 16,898,503 leaves, 0 mismatches, 0 new keys |
| **D6** step 5 with the reference absent | `WP9_OUT=output/wp9_r3probe bash tools/wp9_validate.sh` (a fresh directory, the defect's own scenario) → two `STEP 5 NOT VERIFIED: no reference at …` lines and **`done rc=1`**.  Before the fix the identical run printed no comparison line at all and `done rc=0` |
| **D1** S8 smoke on the repeat-run pair | `compared=660, different=0, missing=0, same_hashseed=None` → **NOT A HASH-INVARIANCE PAIR (precondition not established)**, exit **2**.  The same pair read `IDENTICAL`, exit 0 before |
| **D1** the `--s8` console line | `S8 hash invariance: NOT identical (660 series compared, same_hashseed=None)` followed by `S8 PRECONDITION NOT ESTABLISHED: …` and `S8 NOTE: …`.  Before: `S8 hash invariance: identical (660 series compared)` and nothing else |
| `tools/wp9_validate.sh`, normal run | `done rc=0` |
| `tests/run_tests.py wp9` / `wp12` | **59 passed / 49 passed**, 0 failed (wp9 was 57 before this round: +2 new tests, four existing ones extended) |
