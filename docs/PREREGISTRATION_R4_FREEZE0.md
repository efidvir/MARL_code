# R4 — Freeze 0: development protocol

Status: FROZEN on 19 October 2026. This file is committed on branch `r4` and pushed, as a whole file, to the public repository before the first R4 comparator pass. No R4 comparator result has been read except the one disclosed in section 2.5.

**What the freeze commit fixes.** The freeze is two commits. The code commit C holds every tool and grid file named here. The freeze commit F, on top of C, adds this file and changes nothing else (`git diff --stat C F` lists this file only). At that point:
- **The code commit C** is `25d8d2c922754d748c94bd5b8eeb8f69ec7f9a7c`.
- **The grid files,** by sha256 (also in `tools/r4_grids/SHA256SUMS`):

| File | sha256 |
|---|---|
| `GA.json` (30 configurations, 1,200 passes) | `3821a1d03fbbd61b4d27736be0a7d81ef9f9c7f6b13120ea2af820b12abe2f27` |
| `GA_refs.json` (XL-DET, 40 passes) | `09c78af7c70ceec1a2fad5fd0c54107ee8e26d3f5e60c8330d0c3824b0840314` |
| `GA_confirm.json` (template) | `549b0eaef5dc39b40c83e177e99200bbd6c2367e7955254a36501daba9e2d34d` |
| `CB.json` (template) | `5fef564f6ac5e40a09181075c0ea4b9d0cf8801ca731d7a1d98a6418e6f8361d` |
| `GA_record.json` | `3425e1d7d19df3f1834c83571362c1df84b79c8984c115d3ce364745c295f151` |
| `GA_proxy_fit.json` | `598074ec6374499038bc76a00c0ebc149fdefc07004a252e06d7ce7b17595eef` |

- **TREE_HASH,** the pipeline tree hash of C, is `e2abf1061c4377fa70148cc97b1d1b7a15158c4878b610e827cec1d5c91fb0a2`. It was computed on the compute host after C was synced there (238 files, `sixg_sim/runinfo.tree_digest`), and every pass records it as `tree_sha256`.

The compute host keeps the tree synced from C. This file is not synced to it, so every Gate A, confirmation and Check B pass carries exactly TREE_HASH. A pass from any other tree is refused by `tools/r4_gate_a.py` and `tools/r4_checkb_pair.py`.

R4 is a new study. It asks whether site-local policies trained with the digital twin deliver more than a **fairly tuned** engineered controller, with non-inferior relay churn, reunification, energy and life-safety delivery. It is the fifth confirmatory look in this lineage: R1 (superseded), R2, and R3-lite with its two families. R3-lite was the last attempt in the manuscript version that reports it (v10). R4 is reported in a later version and counted as its own look, whatever its outcome.

This file fixes what must be fixed before any R4 result is read:
- the roles of the seeds;
- the comparator family, its search and the selection rule (Gate A);
- the exploratory check of the existing policies against the tuned comparator (Check B);
- the hypotheses and margins of any confirmatory test.

No MARL policy is trained for R4 until the author decides so after Gate A and Check B. The rules for any training (pilot bands, gates, the learning recipe, the number of replicates and test instances, the stage design, the analysis plan and the power calculation) are frozen in a later annex before the first training pass. Freeze 1 comes before the confirmatory replicates are trained. Freeze 2 comes before any test block is read and fixes the analysis code, the manifest and the seed audit. Each freeze is pushed to the public repository before the step it guards, and the host's record of the push time is its external timestamp.

## 1. Seed roles

| Role | Seeds | Use |
|---|---|---|
| Training pool | 42–49, 100–139 (R2's 48) and 1000–1071 (72 new) | training episodes only |
| Development | 50–59, 70–79 (Gate A); 80–89 (descriptive confirmation of Gate A); 90–99 (reserve) | tuning, screening, gates |
| Validation | 300–339 | Check B and later go/no-go readings (exploratory; already read by R3-lite) |
| Confirmatory test | 700–759 (stage 1; any further stages are named in the training annex) | read once, after Freeze 2 |
| Out-of-distribution test | 760–779 | read once, after Freeze 2 |
| Untouched | 340–359, 400–439 | R3-lite's reserve; not used by R4 |

**Seed audit, 19 October 2026.** A read-only pre-audit of every working tree on the compute host found the following.
- **Seeds 700–779:** no result-data hit. The 80 text mentions are test fixtures and audit records that name the range.
- **Seeds 1000–1071:** one family of result-data hits, in earlier audit-proof records. This is immaterial for a training-only block.

The freeze audit is repeated before Freeze 2.

**Seeds 80–89 are audited before the confirmation passes.** The confirmation grid runs only after a read-only audit of 80–89 on the compute host, recorded with its verdict. Disclosed now: two unit tests of the existing harness (`tests/test_wp0_harness.py`, `tests/test_merge_seams.py`) run 120–300-tick passes of XL-DET, random, OSPF, AODV, marl_static and marl_freeze on seed 82, and have done so since R2. No comparator has run on 80–89.

**The grid driver enforces the roles.** `tools/r4_comparator_grid.py run` refuses:
- any pass on 80–89, 300–339 or 700–899 unless the seeds are named with `--allow-protected` and a reason with `--override-note` (both recorded in the run record and in every pass they cover);
- any pass on 340–359 or 400–439, always;
- a grid marked as a template (`GA_confirm.json`, `CB.json`), which only `tools/r4_gate_a.py` fills.

## 2. Comparators (Gate A)

**Information parity.** A comparator may read any field of the R2 actor observation, the 26 site-local fields that the MARL actor reads, plus the static site type (O-RU, O-DU, Relay). The site type is a configuration fact. No comparator reads global state or the twin at run time.
- The new rules read their inputs through the actor's own extractors: own UE count from `a_ue_count` (saturating at 100), own SINR from `a_sinr_mean` (which reveals it clipped to −5…30 dB), and the site's own Tx (the field behind `a_tx_power`, which XL-DET reads too).
- SE is computed from that SINR with the pilot's SE function rather than read from the observation, because the observed SE field depends on the MCS currently in force.
- A test replaces every other observation field with an object that raises when touched.

**P0 (done): the engine's SINR is the replica's.** The thresholds below are fitted on the twin's analytic SINR replica, so the live field had to match it:
- on the 40 saved Phase 0 states restored in the engine: largest difference 3.6e-14 dB;
- live, XL-DET and a Ts rule on seed 50, Scenarios A and B, 300 ticks: `sinr_average` matched the replica on all 26,400 site-ticks at the moment controllers read their observations (largest difference 3.6e-14 dB), and UE counts matched exactly;
- so the 1e-9 dB criterion holds and no threshold is refitted on the observed field.

**Common rules for every variant.**
- MCS, relay and PRB decisions are XL-DET's; only transmit power differs.
- A target power is pursued as the power closest to it that the engine's ±3/±6 dB steps and 10/33 dBm clamps can reach from the site's current power (ties: fewer steps, then the lower power). Before a site touches a clamp this is every integer from 10 to 33 dBm. After it, only part of them (a site that touched 10 dBm lives on 10 + 3k and 33 − 3k).
- Ts and TL setpoints lie on the start lattice {10, 11, 14, 17, 20, 23, 26, 29, 32, 33} dBm. Residual leaves and targets reached after a clamp can be other integers (for example 13, 15, 21, 24, 27 or 30 dBm).
- A feature rule (TU, TS, the TUL tree, DPS) is re-evaluated on every acting tick. A new selection is adopted once it has been made on 10 consecutive acting ticks: one DCC postcard period, a fixed engineering choice, not fitted. TU and TUL adopt their first selection at once; the trees step toward their base setpoint until their first adoption.

**The family.** Every engineered site-local rule that is built, or simulated at full length, is eligible as T*. Leaving one out would weaken the comparator by omission.
- **XL-DET:** unchanged. Its decision series is bit-identical to the frozen R3-lite code. It is itself a Gate A candidate.
- **Ts:** one fixed setpoint per site type.
- **TU:** a per-type setpoint chosen by the site's own UE count (1 or at most 2 thresholds per type).
- **TS:** a per-type setpoint indexed by own SINR or SE (depth-1 and depth-2 trees).
- **L:** XL-DET's own SINR/interference loop over the full 10–33 dBm range with fixed damping (hold time), hysteresis, deadband and SINR-target variants.
- **TL:** the loop clamped to a band of ±6 or ±12 dB around per-type setpoints, on the setpoints' own 3 dB lattice.
- **TUL:** the TU setpoint with a trim: XL-DET's loop within ±6 or ±12 dB, or a residual tree on own SINR/SE within ±6 or ±12 dB.
- **DPS:** depth-2 trees over own UE count, SINR/SE and own offset, with residual leaves within ±6 or ±12 dB or absolute leaves over the full lattice.

**Behaviour fixtures are not candidates.** A configuration run only as a behaviour fixture in passes of at most 300 ticks is excluded, because it was never a proposed comparator:
- `tl` (band ±3 around 32/11/10) and `l_hold5` (hold 5 ticks) of the main smoke;
- the feature smoke's `dps_const` and `tu_const`, which reproduce the 32/11/10 Ts rule bit for bit; that rule is listed.

`GA_record.json` lists them.

### 2.1 How the configurations are fixed, before any result is read

All of this uses only the 40 saved Phase 0 development states (seeds 50–59 and 70–79, both scenarios, tick 1800) and the twin's analytic SINR objective J = Σ max(1, nUE) · SE(SINR). The code is `tools/r4_proxy_fit.py` (fits) and `tools/r4_make_ga_grids.py` (grid files); their outputs are `GA_proxy_fit.json` and `GA_record.json`.

**Guard 2 is imposed on the proxy.** The unconstrained fits of the first build maximised J by giving up low-UE sites. On the proxy they would have raised the number of sites below the SE threshold by +5 to +8 against XL-DET, failing Gate A's guard 2 almost surely. So every family is fitted under the proxy of that guard:
- **The proxy.** B(P) is the number of alive sites whose replica SINR at powers P is below −3 dB (QPSK 1/3, the engine's `sites_no_se` definition). A rule's guard-2 proxy on a set of states is, per scenario, the mean over that scenario's states of B(rule as run) − B(XL-DET's saved Tx). "As run" is the stationary schedule: synchronous re-evaluation from the base powers, averaged over steps 3–5. A rule is **proxy-feasible** when its proxy is at most +2 in both scenarios.
- **Ts triples.** Only proxy-feasible triples are candidates: for the shortlist, for every fold's best triple and for the base triple of every fit.
- **Fitted families.** Each fitting option is fitted at every λ of the ladder 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32, 1000; the fitter maximises J − λ · (B − B_XL-DET) on the training states. The fit kept is the one with the highest plain training J among the fits that are proxy-feasible on the training states (ties: the smaller λ). If none is, it is the one with the smallest worst-scenario proxy, recorded as infeasible.
- **Fitting options.** Where a family has two (one-shot fit or stationary fit), each is fitted leave-one-seed-out: 20 folds of 19 seeds × 2 scenarios, λ chosen within each fold. The option chosen is, among those whose pooled held-out proxy is at most +2 in both scenarios, the one with the higher mean held-out J (ties: the first listed); if none is, the one with the smaller worst-scenario held-out proxy. Only then is it refitted on all 40 states. LOSO is used for nothing else.
- **TU\*** (the TU setpoint inside TUL) is the better of TU1 and TU2 by the same rule.

**The Ts shortlist** (at most 9, plus the simulated rule below):
- the 6 best proxy-feasible triples by mean J over the 40 states (for a fixed triple the leave-one-seed-out mean is that mean);
- the leave-one-seed-out-best rule: in each of the 20 folds, the best triple among those proxy-feasible on the fold's 38 training states; the triple picked most often (ties: the higher mean J);
- the type-mean rule: the oracle plans' mean power per type over alive sites on all 40 states, rounded to the nearest lattice value;
- uniform 23 dBm;
- duplicates removed.

**The rule simulated before Freeze 0.** Ts O-RU 32 / O-DU 11 / Relay 10 dBm was run at 3600 ticks (the cost measurement, disclosed below). It is listed (`ts_meas`) whatever its proxy (+6.9 in A, +6.0 in B).

**Fitted families.** 2 TU (`tu1`, `tu2`), 2 TS (`tsq1`, `tsq2`), 2 TUL trees (`tul_tree6`, `tul_tree12`) and 3 DPS (`dps6`, `dps12`, `dpsf`).

**Fixed configurations,** chosen without reference to any outcome:
- **L (6):**
  - `l_h10`: hold 10 ticks;
  - `l_y2`: hysteresis 2 dB, deadband 1 dB;
  - `l_h10y2`: both;
  - `l_h30y3`: hold 30 ticks, hysteresis 3 dB, deadband 2 dB;
  - `l_t14`: SINR target 14 dB and ample 25 dB (XL-DET's 17/28 lowered by 3 dB), ICIC floor 9 dB (target − 5, XL-DET's relation), hold 10 ticks;
  - `l_t20`: target 20 dB and ample 30 dB (raised by 3 dB, ample capped at the engine's 30 dB SINR ceiling), ICIC floor 15 dB, hold 10 ticks.
- **TL (4):** ±6 and ±12 dB around the leave-one-seed-out-best proxy-feasible Ts triple, each with XL-DET's loop and with the `l_h10y2` loop.
- **TUL loop (2):** TU\*'s setpoint with XL-DET's loop within ±6 and ±12 dB.

**The result.** 30 configurations in `GA.json`, the cap, listed in the family order:
- **Ts (9).** 424 of the 1,000 lattice triples are proxy-feasible. In O-RU/O-DU/Relay order, the shortlist is:
  - 32/20/20, the best feasible triple and the leave-one-seed-out-best rule (picked in 14 of 20 folds, 33/23/20 in the other 6);
  - 33/23/20, 29/17/17, 33/20/23, 32/17/23 and 29/14/20;
  - the type-mean rule 29/20/23;
  - uniform 23 dBm;
  - and `ts_meas`, 32/11/10.
- **TU\*** is TU1.
- **Fitted rules.** All nine are proxy-feasible on their training states, held out and in sample. λ is 8 to 12, and the stationary fit was chosen for every family with two fitting options.
- **`dps12` and `dpsf` are the same controller.** The two fits have the same splits, and their leaves give the same powers once the 10 dBm clamp applies. Both are listed under their own names and hashes, so their passes are identical; the exact-tie rule gives `dps12`.
- **Fixed rules.** The six L configurations, four TL around 32/20/20, and the two TUL loops around TU1.

`GA_record.json` holds every specification and hash, every λ and option chosen, and every proxied configuration's guard-2 proxy. The six best triples without the constraint (all with O-RU at 33 dBm, proxy +5.0 to +7.3 sites) and the unconstrained ladder rungs are recorded there but not listed.

**Cross-check.** Before a grid file is written, every fitted specification and every shortlist triple is re-run on the 40 states by an independent implementation: the controllers' own selection code, the vendored replica and the stationary schedule. Its J and guard-2 proxy must equal the fit's on every state within 1e-9.

**Families considered and not run.** A search of further site-local rules preceded this file. It was done on the analytic proxy and on existing run data only. Four further families were considered and are not in `GA.json`:
- **An own-SINR offset table around the TU setpoint.** This has the same structure as `tul_tree6` and `tul_tree12`: per type, four own-SINR bins with offsets around TU\*, the bins fitted rather than fixed at quartiles. So the family is represented.
- **A richer table over own UE count × own SINR (16 cells per type).** Under cross-validation on the proxy it scored below the best engineered rule (−1.83 pp in A, −1.27 pp in B). The depth-2 trees `dps6`, `dps12` and `dpsf` represent its coarser form.
- **A neighbour-state (Block B) trim.** Fitted in sample, it chose no trim.
- **A demand-gated power floor for sites whose microwave or mesh backhaul carries user-to-user traffic.** This needs a demand signal that is not among the 26 fields. It exploits the backhaul-scaling property disclosed below. On the proxy, the central pool it could reach is about +1 pp in Scenario A (wide interval) and 0 in Scenario B. Its absence is reported as a limitation of T\*, and the corrected-engine condition below covers it.

**The engine is the one v10 used.** In it, the capacity of 18 GHz microwave and 60 GHz mesh backhaul hops is scaled by the 3.5 GHz access spectral efficiency of their end sites (`_MCS_SCALED_LINK_TYPES` in `sixg_sim/simulation.py`). Satellite, fibre and transport-relay hops are not scaled, because they have their own link budgets. The same reasoning applies to microwave and mesh hops.

The author decided on 19 October 2026 to keep this engine for Gate A and Check B. The reason is continuity: like-for-like with the R2 and R3-lite results and with the existing learned passes. The property is disclosed here. A corrected-engine condition, with these hops unscaled in every arm, is to be run later as a sensitivity check and reported whatever its outcome.

**Search disclosure.** Every analysis made while preparing R4 is released with R4's results: the odds analyses, the head-by-head analysis of replicate 1234 against XL-DET, the non-power headroom analysis and the system-model probes.

### 2.2 Simulation

Every configuration of `GA.json`, and XL-DET (`GA_refs.json`), is simulated on development seeds 50–59 and 70–79 × Scenarios A and B at 3,600 ticks: 1,240 passes (1,200 + 40). The engine includes exact memoisation of path searches (bit-identical outputs, verified at full length against the frozen tree).

### 2.3 Gate A rule

Applied mechanically by `tools/r4_gate_a.py --tree-sha256 TREE_HASH` to the report of `tools/r4_comparator_summary.py` on `GA.json` and `GA_refs.json`.

**Eligibility.** A configuration is eligible only when:
- it has all 20 development seeds paired with XL-DET in each scenario;
- every pass of the configuration and of the reference meets the frozen data-validity rules of `confirmatory_analysis.check_invariants` (3,600 entries per series, no NaN, valid postcards, complete R4 instrumentation); a failing pass leaves its seed unpaired;
- every pass comes from TREE_HASH with no tree change during the pass.

The report must be made with the frozen thresholds, the 500-tick window, 3,600 ticks and one tree. Gate A recomputes every score and guard from the per-seed values and stops if they disagree with the report.

**Guards,** each in Scenario A and in Scenario B:
1. **life-safety:** the life-safety class's delivered/offered fraction, pooled over the last 500 ticks, × 100; the mean paired difference against XL-DET is at least −2 pp;
2. **sites below SE:** the mean over the last 500 ticks of the number of sites whose `sinr_average` is below −3 dB; the mean paired difference is at most +2;
3. **M7:** relay teardowns per 1,000 post-event ticks; the configuration's mean over the 20 seeds is at most 1.1 × XL-DET's mean.

**Score.** The mean over Scenarios A and B of the mean paired achievability difference against XL-DET over the 20 seeds. Achievability is R2's metric: the mean over the last 500 ticks of min(100, 100 × delivered/offered), user-to-user.

**Candidates.** XL-DET, with score 0, and every eligible configuration that passes all three guards in both scenarios.

**Choice.** Every candidate within 0.5 pp of the best score is tied with it. T* is the tied candidate of the simplest family, in the order XL-DET < Ts < TU < TS < L < TL < TUL < DPS; within that family, the highest score; an exact tie goes to the earlier configuration in `GA.json`.
- So **T* is XL-DET whenever no configuration scores more than 0.5 pp above it**, including when none is eligible or none passes the guards.
- **The runner-up** is the candidate the same rule selects with T* removed; XL-DET can be it. There is none only when T* is XL-DET and no configuration is a candidate.

**After the choice.** T* and the runner-up are run on seeds 80–89 × A, B as a descriptive check (`GA_confirm.json`, filled by the tool), after the seed audit of 80–89. That check does not change the choice. T*'s specification and hash are frozen at this point; any later change to T* would require retraining.

**Reading, not a stop.** H_rem is the hindsight oracle's paired achievability over T* on seeds 50–59 and 70–79. It is reported. Whether any training follows is decided by the author after Check B.

### 2.4 Blind logs until this file is pushed

Until this file is pushed, no achievability, life-safety delivery or M7 of any pass is computed for reading, printed or logged:
- every pass log goes through `tools/r4_blind.py`, a whitelist that keeps pass statistics, the driver's own lines and tracebacks only;
- unit-test logs go through its redact mode, which withholds the harness's outcome lines and keeps the test runner's;
- no summary is run on any pass.

Behaviour checks on 300-tick smoke passes (setpoints reached, bands held, selections recomputed, XL-DET identity) are made on the pass pickles without computing any outcome. From the push on, logs are no longer filtered.

### 2.5 Disclosure

**One comparator result has been seen.** It came from a cost-measurement pass of the Ts rule O-RU 32 / O-DU 11 / Relay 10 dBm on development seed 50, Scenario A, at 3,600 ticks:
- +11.1 pp achievability over XL-DET;
- life-safety delivery fell from 0.183 to 0.078;
- sites below the SE threshold rose from 2.2 to 7.0;
- so it fails two Gate A guards.

No other comparator result was read.

**Outcomes were written, unread, before the blind-log rule.** All of the following are on the compute host. Only the number of matching lines and the sha256 of each file were computed, and no line was displayed.
- **Moved unread** to `~/r4_work/QUARANTINE_prefreeze0/` (`MANIFEST.json`: 63 files, 800 outcome lines, each file's original path and sha256):
  - progress lines of the 300-tick smoke passes of the Phase 1 build and its fix (seeds 50–51: XL-DET, the Ts rules 32/11/10 and 20/29/23, TL band 3 around 32/11/10, `l_hold5`, XL-DET-L at defaults);
  - progress lines of 60-tick passes (seed 50 A: XL-DET, Ts 32/11/10, TL band 3);
  - the cost-measurement logs and summary (above);
  - progress lines of the 3,600-tick identity passes (XL-DET on seeds 50 and 60, XL-DET-L at defaults and marl_static on seed 50) and of a 600-tick XL-DET profiling pass (seed 50 A);
  - three summaries the fix phase computed on 19 October at about 13:18. The first gave paired achievability, guards and M7 at 300 ticks on seeds 50–51 for 32/11/10, 20/29/23 (now `ts_typemean`), TL band 3 and `l_hold5`, with a guard verdict and a ranking. The second compared XL-DET-L at defaults against XL-DET at 3,600 ticks on seeds 50 and 60; they are bit-identical. The third, a deliberately failing run, found every pass unusable.
  - three unit-test logs with harness outcome lines.
- **Redacted in place** (the outcome lines replaced by a marker): the unit-test logs of the Gate A build and the log of the test harness pass on seed 82.
- **Left in place** in the r4-perf profiling session's own directory (`~/r4_work/prof/`): progress lines of 600–900-tick XL-DET and marl_static profiling passes on seeds 50–51, and of its test runs. No comparator ran there.

R3-lite's own published reports in the copy of the frozen tree are R3-lite results and are not R4 outcomes.

## 3. Check B: the existing policies against T* (exploratory)

T* and the runner-up are run on validation seeds 300–339 × A, B (`CB.json`, filled by the Gate A tool; when T* or the runner-up is XL-DET, it is the plain XL-DET arm). Two existing policies are then compared with T* by seed and scenario:
- replicate 1234;
- the four-replicate ensemble.

The existing passes for both policies on 300–339 come from R3-lite and are reused unchanged. The comparison is also made on development seeds 50–59 and 70–79, where marl_static passes for both exist:
- `~/MARL_r3/r3out/phase0/out/dev_ck1234_{a..d}.pkl` (replicate 1234);
- `~/MARL_r3/r3out/phase0/out/ens_dev.pkl` (the ensemble);
- all made by the R2 tree 6b53f3f.

**What is paired** (`tools/r4_checkb_pair.py`), per seed and scenario, d = learned − comparator:
- achievability, with the mean, a bootstrap 95 % interval and the paired tests;
- the mean differences of M7, steady energy, energy per delivered unit and component count;
- Gate A's M7 guard applied to the learned policy: its mean M7 against 1.1 × the comparator's, with the ratio.

**What cannot be paired.** The existing learned pickles carry no R4 instrumentation, so the life-safety and sites-below-SE guards cannot be applied to the learned side. They are reported for the comparator side only.

**Provenance, enforced by the tool.**
- Every comparator pass comes from TREE_HASH, the Gate A tree; no flag admits another.
- The learned pickles come from older trees, so pairing them needs `--allow-mixed-tree-pairing`, which the report records. The engine is bit-identical across those trees (XL-DET and marl_static at 3,600 ticks against the frozen R3-lite tree).
- Every learned pickle must carry the R3-lite evaluation engine environment, 3,600 ticks and only checkpoints pinned by R3-lite: R2's replicates, `confirmatory/chain_plan.py` `R2_CHECKPOINT_SHA256`. Replicate 1234 must be exactly 1234; the ensemble exactly 1234 + 2345 + 3456 + 4567. Each pickle's provenance is recorded.

Check B is exploratory: it tests no hypothesis and spends no confirmatory alpha. If the author later decides to test an existing policy against T* confirmatorily, that test is pre-registered, with its own fresh seeds, before any of those seeds is read. The report then states that the policy was selected after the fact on R2's data and was confirmed against XL-DET in R3-lite.

## 4. Confirmatory hypotheses and margins (any confirmatory R4 test)

These were chosen by the author on 2026-10-15, before any R4 pass.

**Estimands.**
- For learned replicate r and test instance i, d_{r,i} = ach(policy_r, i) − ach(XL-DET-T*, i).
- θ_c is the mean over instances of the mean over all K replicates, conditional on the trained set.
- θ_r is the recipe-level mean under crossed random effects.
- All K replicates trained after Freeze 1 count. None is dropped or selected.

**Primary family: fixed sequence.** Each step must hold in Scenario A and in Scenario B (intersection-union, one-sided α = 0.025 each). Testing stops at the first failure.
1. θ_c > 1 pp
2. θ_r > 1 pp
3. θ_c > 3 pp
4. θ_r > 3 pp

**Non-inferiority family, against XL-DET-T*.** Intersection-union of one-sided 97.5 % bounds in each scenario, with no multiplicity adjustment.

| Endpoint | Margin, policy − XL-DET-T* |
|---|---|
| M7, relay teardowns per 1,000 post-event ticks | ≤ +0.40 |
| M4, fraction of post-event time at the attainable component count | ≥ −0.03 |
| Absolute energy, steady-state J per tick | ≤ +5 % |
| Life-safety delivery, last 500 ticks | ≥ −2 pp |

Margins are never moved.

## 5. Reporting

Everything is reported whatever the outcome:
- every comparator configuration simulated, with its specification;
- every gate result and reading;
- Check B in full;
- every repeated or failed invocation, with its cause.

If the study stops at any point, it is reported as stopped, with what was computed. Any test block not yet read stays unread.
