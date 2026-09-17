# Pre-registration — confirmatory test of MARL-RIC on fresh topologies

**Status:** written and hashed before any confirmatory instance was generated or
evaluated. The sha256 of this file, of the analysis script and of the frozen
code and checkpoint are recorded in `FREEZE.sha256`. Its creation time is the
UTC time in that file and in the Overleaf git history (`records/`).

## 1. Why this test exists

The manuscript's twenty evaluation topologies (seeds 50–59 and 70–79) are not a
clean confirmatory set. Seeds 50–59 were used to diagnose reunification
failures, and those diagnoses motivated the bridge-credit repricing and the
fragment-coordination observation block (Block E). All twenty were then
evaluated under an earlier transport model before the transport
link-management rules were adopted. From now on those twenty instances are the
**development set**. This document defines a **confirmatory set** that has
never been generated, inspected or evaluated. It is evaluated once, under a
configuration frozen before its instances exist.

## 2. Confirmatory instance set

- **Topology seeds:** 80–99, 20 instances. Each is used with the matching
  scenario seed in both scenarios: A, the fragmenting core severance, and B,
  the emergency-services surge on a severed network.
- **Evidence that the seeds are untouched** (audit of 2026-09-16):
  - No result file on the evaluation server contains seeds 80–99. The seeds
    seen across all 40 result files are 50–66 and 70–79.
  - No script, log or manifest in any run directory, or in the local
    repository, refers to them.
  - Training used topology seeds 42–49 and 100–139, with scenario seeds
    700000 + episode, so no (topology, damage) pair from this set was ever
    generated.
  - Validation used seeds 60–66.
- **Generation.** The attainable optimum of every instance is computed by
  `build_benchmark_manifest.py --seeds 80-99` after this file is hashed and
  before any arm runs. The resulting manifest is hashed into `FREEZE.sha256`
  before the evaluation starts.
- **Exclusions.** None. All 20 instances are analysed in both scenarios,
  whatever their attainable optimum. An instance is dropped only if the
  harness cannot construct or run it. Any such drop is reported with its cause.
  A failed run may be repeated once, unchanged.

## 3. Frozen configuration

No part of the following may change between this file's hash and the end of
the analysis.

- **Code.** Every `.py` file in `~/MARL_run_5` and `~/MARL_run_5/sixg_sim` on
  `cersrv-029`, as hashed in `FREEZE.sha256`. This is the exact code that
  produced the development-set results; the only later file is
  `dump_endstate.py`, a diagnostic that the harness does not import.
- **Deployed policy.** `ce_ck_s1234/policy_final.pt`, sha256 `bbee30e4e9a4d26432753341d72e64661eea63c67139d290a2e831479f01e01f`.
  It was selected on validation seeds 60–66 by the pre-fixed rule, and no
  other replicate is evaluated here.
- **Engine flags.** Applied to every arm:
  `MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1`.
- **Policy flags.**
  `MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01 MARL_STEER_MODEL=actionable`.
- **Arms.** `xldet, sdn, ospf, olsr, batman, aodv, random, marl_freeze`, in one
  invocation of `run_timeline_comparison.py --seed-list 80-99 --no-figures`,
  3 600 ticks per run.
- **Analysis.** `confirmatory_analysis.py`, sha256 in `FREEZE.sha256`.
  Before this file was written, the script was validated by reproducing every
  development-set figure already in the manuscript (`dev_validation_report.json`).

## 4. Metrics

These are the definitions of `paper/make_results.py`, unchanged.

- **Achievability.** Delivered over offered user-to-user volume. Steady state is
  the mean over the final 500 ticks.
- **Energy per delivered unit.**
  - Per instance: whole-run radio energy divided by whole-run delivered
    volume. This value is used in the paired tests.
  - Pooled: the same ratio summed over the whole set. Reported as description.
- **Components.** The steady-state count of relay-inclusive infrastructure
  components. "At optimum" means the count is at or below the instance's
  attainable optimum.
- **Pre-surge level (Scenario B).** Mean achievability over ticks [900, 1400).
  This is a settled 500-tick window after the tick-0 severance and before the
  responder arrival at tick 1400.
- **Best routing (per instance).**
  - For achievability: the maximum over OSPF, OLSR, B.A.T.M.A.N. and AODV.
  - For energy per unit: the minimum over the same four.

  Both are the conservative comparator for MARL-RIC.

## 5. Primary hypotheses

There are nine primary hypotheses. The decision test is the exact two-sided
Wilcoxon signed-rank test on the 20 paired per-instance differences, with zeros
dropped and mid-ranks for ties. A Holm correction is applied across all nine at
α = 0.05. Each hypothesis is **confirmed** if and only if its Holm-adjusted p is
below 0.05 **and** its signed-rank statistic lies in the expected direction.

For every hypothesis the report also gives the paired mean difference with its
95 % t-interval, the win count and the exact sign test. These are descriptive.
When they disagree with the decision test, the disagreement is reported next
to the decision. They never override it.

| ID | Scenario | Quantity | Comparison | Expected |
|---|---|---|---|---|
| P1 | A | achievability | MARL-RIC − XL-DET | > 0 |
| P2 | A | achievability | MARL-RIC − SDN | > 0 |
| P3 | A | achievability | MARL-RIC − best routing | > 0 |
| P4 | B | achievability | MARL-RIC − XL-DET | > 0 |
| P5 | B | achievability | MARL-RIC − SDN | > 0 |
| P6 | B | achievability | MARL-RIC − best routing | > 0 |
| P7 | A | energy per unit | MARL-RIC − XL-DET | < 0 |
| P8 | A | energy per unit | MARL-RIC − SDN | < 0 |
| P9 | A | energy per unit | MARL-RIC − best routing | < 0 |

## 6. Secondary analyses

These are reported in full and are outside the Holm family.

- **S1 — reunification.** At-optimum counts per arm in both scenarios, and the
  paired MARL-RIC − XL-DET component difference. The development set leads us
  to expect XL-DET ≥ MARL-RIC. No claim of a MARL-RIC reunification advantage
  will be made unless S1 shows one.
- **S2 — surge response (Scenario B).** The paired difference in change from
  the pre-surge level, (post − pre)_MARL − (post − pre)_baseline, against
  XL-DET, SDN and the best routing protocol. The best routing protocol here is
  the one with the largest change.
  **Interpretation rule:** the manuscript may say that MARL-RIC *responds to the
  surge* better only if S2 against that baseline has Wilcoxon p < 0.05 in the
  positive direction. Otherwise the Scenario B margin is described as an
  allocation advantage on the severed network that persists through the surge.
- **S3 — pre-surge level (Scenario B).** MARL-RIC minus each baseline.
- **S4 — absolute energy per tick (Scenario A).** MARL-RIC − XL-DET.
- **S5 — service-loss integral (Scenario A).** Medians per arm, and the paired
  MARL-RIC − XL-DET difference.
- **S6 — energy per unit (Scenario B).** MARL-RIC − XL-DET.
- **Shared-world check.** The largest pre-event achievability difference
  between arms in Scenario A. It is expected to be 0.

## 7. How the results enter the manuscript

- The 80–99 set becomes the **confirmatory test set**. Every inferential
  statement in the abstract and conclusion (p-values, confidence intervals,
  "established", "significant") is taken from it.
- Seeds 50–59 and 70–79 become the **development set**. Their detailed figures
  and tables stay in the paper, labelled as development results, with their
  history disclosed. Their p-values are presented as descriptive of that set
  and not as confirmation.
- **A primary hypothesis that is not confirmed** is reported as not confirmed,
  with its estimate and interval, and any matching claim is removed from the
  abstract and conclusion. This applies whatever the development set showed.
- **Nothing is changed after this set has been seen.** That covers code,
  checkpoint, flags, instances, metrics, windows, comparators, tests,
  correction and exclusions. Any analysis not listed here is labelled
  *post hoc* wherever it appears.
