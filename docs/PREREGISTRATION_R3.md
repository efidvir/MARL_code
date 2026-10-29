# Pre-registration R3-lite — confirmatory test of two existing R2 policies on a fresh instance block

**Status: THE AUTHOR GAVE HIS GO-AHEAD ON 2026-10-12 (§17 says what he had
been sent, what he was asked, and gives his words and their times;
`docs/DECISIONS.md` D-R3.5 records it). Written 2026-10-11, revised the
same day after an adversarial review of the build and of the text
(D-R3.2), amended on 2026-10-11/12 by the author's
decisions recorded in D-R3.3 — forty instances instead of twenty, and the
ensemble of the four R2 replicates as a second confirmatory family — and
revised on 2026-10-12 after an independent review of that amendment and a
review of its fixes (D-R3.4). D-R3.3 quotes the questions put to him on
the design; D-R3.2 and D-R3.4 record the reviews that changed the text.**
Nothing has been run on the confirmatory block of this
study: no pass, no smoke, no rehearsal, no manifest. The text is frozen by
the commit that is synced for the run (**freeze 1**, §13); its hash, the
tree digest and the records of §13 are appended in §17 and never changed
afterwards.

**What this study is, in one paragraph.** R2 (`docs/PREREGISTRATION_R2.md`,
seeds 200–219, code `6b53f3f`) is finished and its result stands. R3-lite
tests TWO existing policies of R2's own training run, each identified by
the sha256 of files that already exist, against the same seven comparators,
with the same engine, metrics, hypotheses, tests and decision rule, on
forty topology seeds that nothing has ever been run on. **Family 1, the
primary family,** is replicate torch seed 1234. **Family E, the second
family,** is the ensemble of the four replicates. Each family has its own
nine hypotheses and its own Holm correction; they are never pooled, and
neither can replace the other (§8, §14). Nothing is retrained, no engine
file changes, and the analysis script is the R2 script byte for byte. The
new decisions are which policies are under test and on how many instances;
how and when they were taken is disclosed in §1.4 and §3.

`§` means a section of this document unless `R2 §` is written, which means
`docs/PREREGISTRATION_R2.md`. This document is self-contained for everything
that decides a hypothesis (instances, arms, metrics, tests, decision rules,
freeze procedure, claims). The description of the simulator and of each arm's
model is R2 §2, which is part of this pre-registration by reference: it
describes the code at `6b53f3f`, and §2 below pins that code file by file.

The launch procedure is `docs/RUN_CHAIN.md` §7 and the R3-lite section of
`docs/FREEZE_CHECKLIST.md`. The decision records are `docs/DECISIONS.md`
D-R3.1 (the decision and the build), D-R3.2 (the review of both and what
it changed), D-R3.3 (the author's decisions that gave the study its
present form: two families, forty instances), D-R3.4 (the review of
that amendment and what it changed) and D-R3.5 (the author's go-ahead and
his confirmation of the seed-audit lines). **"The lead"**, throughout this
text and those records, is the party that designed this study, wrote its
code and these documents in the working session with the author, ran the
computations, and put each question to him with a recommended option.

## 1. Why this test exists

### 1.1 The R2 result, which is not changed by anything here

R2 ran its pre-registered chain on 2026-10-05/06 (tree `6b53f3f`, tree digest
`9e8657b3dd2e6bab…`, report `~/MARL_run_9/confirmatory/final_cf_report.json`,
sha256 `3f54eb1e87e59777…`). The MARL-RIC arm was `marl_static` executing
replicate torch seed **4567** (sha256 `7e640ce090b4a414…`), chosen at R2's
stage S3b by R2's pre-registered rule: the fewest mean steady components on
the seven validation instances of Scenario A.

| ID | Scen. | Quantity | MARL-RIC − … | paired mean | 95 % t-interval | wins | exact Wilcoxon p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| P1 | A | achievability (pp) | XL-DET | −0.35 | [−4.28, +3.58] | 8/20 | 0.596 | 0.596 | not confirmed |
| P2 | A | achievability (pp) | SDN | +2.62 | [+0.06, +5.18] | 18/20 | 0.0056 | 0.028 | **confirmed** |
| P3 | A | achievability (pp) | best routing | +14.62 | [+10.36, +18.88] | 20/20 | < 1e-5 | 2e-5 | **confirmed** |
| P4 | B | achievability (pp) | XL-DET | +3.40 | [−1.30, +8.09] | 13/20 | 0.294 | 0.589 | not confirmed |
| P5 | B | achievability (pp) | SDN | +5.15 | [+2.98, +7.31] | 18/20 | 2e-5 | 1.3e-4 | **confirmed** |
| P6 | B | achievability (pp) | best routing | +7.15 | [+4.64, +9.66] | 20/20 | < 1e-5 | 2e-5 | **confirmed** |
| P7 | A | J per unit | XL-DET | +0.0145 | [−0.0057, +0.0347] | 7/20 | 0.154 | 0.461 | not confirmed |
| P8 | A | J per unit | SDN | +0.0168 | [+0.0042, +0.0294] | 6/20 | 0.0121 | 0.048 | not confirmed — **reversed** (MARL-RIC used more energy per unit) |
| P9 | A | J per unit | best routing | −0.0403 | [−0.0654, −0.0152] | 16/20 | 0.0032 | 0.019 | **confirmed** |

Five of nine primary hypotheses were confirmed (P2, P3, P5, P6, P9). Against
the engineered cross-layer controller XL-DET the learned policy was **not
distinguishable from XL-DET**: none of P1, P4 and P7 was confirmed (P1 −0.35
pp, 95 % interval [−4.28, +3.58]; P4 +3.40 pp, [−1.30, +8.09]; P7 +0.0145 J
per unit, [−0.0057, +0.0347]). That is a failure to show a difference, not a
demonstration of equality: R2 specified no equivalence margin and tested
none, and its intervals admit a difference of several points in either
direction. Stability family: S9e (A, M7) +0.84 teardowns per 1000 ticks,
Holm p 1.5e-4, and S9f (B, M7) +2.15, Holm p 5e-5, both in the pre-registered
"MARL-RIC worse" direction; S9a–S9d not assessable (0–2 non-zero
differences). By R2's own rule the learned policy's **bridging may not be
called stable**.

That is the R2 result. It is reported in the paper as the R2 result, with
its policy, its selection rule and its verdicts, whatever R3-lite finds.
R3-lite does not re-analyse, re-weight, pool with or replace it.

### 1.2 What was looked at after R2 (R3 Phase 0), and what it showed

After R2, development and validation instances only were used to diagnose
that result (R3 Phase 0, 2026-10-07 to 2026-10-11, tree `~/MARL_r3` at `6b53f3f`,
records under `~/MARL_r3/r3out/phase0/`; a seed guard refused every seed
outside development 50–59, 70–79 and validation 60–66). The part of it that
matters here:

- The four R2 replicates differ widely as policies, although they come from
  one training configuration and differ in the torch seed only.
- R2's rule ranked them by residual component count in Scenario A, on which
  they differ little (1.29, 1.49, 1.29, 1.14 mean steady components on the
  seven validation instances for 1234, 2345, 3456, 4567) and which does not
  measure delivery; it picked 4567, which on validation is **below** XL-DET
  in Scenario A (−2.47 pp, 1/7 instances) and not distinguishable from it
  in B (+1.83 pp, 3/7 instances).
- Replicate **1234** of the same training run is above XL-DET on development
  and on validation in both scenarios (table in §3.2).

**What Phase 0 showed about the comparator, which this study does not
address.** "XL-DET" in this document always means **XL-DET as implemented at
`6b53f3f`, a fixed engineered reference**. Phase 0 found that it is
improvable by a static power setting: a per-site power plan computed in
hindsight for each instance (it uses information no arm has at run time)
raises XL-DET's own achievability by +14.13 pp in Scenario A (bootstrap 95 %
interval [+11.10, +17.41]) and by +18.46 pp in B ([+15.07, +22.03]), on 20
of 20 development instances in each (`~/MARL_r3/r3out/phase0/PHASE0_REPORT.md`
v1.1, its sections 1 and 3.5). Whether a rule that needs no run-time
information — one fixed power per site type — would recover part of that
gain was **not simulated** (on the planner's model objective, which is a
proxy and not achievability, such a rule captures 83 % and 72 % of the
plan's gain; its section 3.7). On development seeds 50–59 (n = 10),
replicate 1234 is not distinguishable from an untuned XL-DET given one more
level of run-time information (Phase 0's level L1) in Scenario A (+2.83 pp,
[−2.31, +7.69]) and is above it in B (+4.94 pp, [+1.23, +8.71];
uncorrected, exploratory; its section 6). R3-lite changes no engine file, tunes no
comparator and trains nothing, so none of this is tested here. The
consequence is fixed now and repeated in §14: a confirmed P1 or P4 is a
statement about the selected policy against this fixed reference. It is
**not evidence that learning, rather than a power setting, is responsible**
for the difference, and it is not evidence that the policy delivers more
than a tuned engineered controller.

Phase 0 also examined an information ladder and the actor's advantage
signal. Neither is part of this study.

### 1.3 Why a new confirmatory test, and why on new instances

Choosing replicate 1234 **after** seeing how all four replicates perform on
the development and validation instances is a selection on those instances.
Its advantage there is therefore an optimistic estimate and is not evidence
by itself (the best of several candidates on a set is expected to look
better on that set than it is). The development seeds are, moreover, the development seeds of
R2. The only way to learn whether the advantage is real is a test on
instances that played no part in the choice, with everything else fixed in
advance. That test is this document. The author decided it on 2026-10-11
(D-R3.1); version 9 of the manuscript is held until it has run.

**Every confirmatory look at the question, counted.** "Does a learned
policy deliver more than XL-DET?" (the hypotheses against XL-DET on
achievability, P1 in Scenario A and P4 in Scenario B) has been, or will
be, put to a confirmatory test four times in three studies:

| # | study | date | instances | simulator | policy, and how it was chosen | P1 (A) | P4 (B) |
|---|---|---|---|---|---|---|---|
| 1 | the first test | 16 September 2026 | seeds 80–99, n = 20 | the uncorrected simulator, which a post-test audit found to differ materially from the described system (R2 §1) | that study's policy | not confirmed | confirmed on that simulator; **superseded by R2 as the paper's confirmatory evidence** because of the simulator, and reported as history with its verdict (R2 §1, §14) |
| 2 | R2 | 5–6 October 2026 | seeds 200–219, n = 20 | corrected (`6b53f3f`) | replicate 4567, by R2's pre-registered rule | not confirmed | not confirmed |
| 3 | R3-lite, family 1 | this study | seeds 300–339, n = 40 | `6b53f3f`, unchanged | replicate 1234, by the paired-achievability rule, formulated after the results on the selection instances were seen (§3) | to be tested | to be tested |
| 4 | R3-lite, family E | this study | the same forty instances | `6b53f3f`, unchanged | the ensemble of the four replicates, the pick of the guarded rule Phase 0 recommended **at g = 2** (g was not fixed; §3.3 item 7) | to be tested | to be tested |

**One numbering is used throughout this text and in the paper: four looks,
numbered as in the table.** Look 1 is the first test, look 2 is R2, and
this study, the third study of the question, holds looks 3 (family 1) and 4
(family E); on the corrected simulator that is one look by R2 and two by
this study. Each is a valid test of its own policy on instances that
played no part in choosing it, and each family is corrected within itself
only (§8); because the families are not corrected for one another, **the
count is disclosed instead, wherever a confirmed hypothesis against XL-DET
is quoted** (§14, §15). The block of the first test was analysed again
after that test — that study's own amendment, and the post hoc stale-edge
re-analysis R2 reports (R2 §1, §14) — and those are re-analyses of a block
already looked at, not looks on new instances: they are not counted, and
seeds 80–99 are fresh for nothing. R3-lite as a whole is **the last
confirmatory attempt at it in this manuscript**, for both families
jointly: the author accepted that commitment on 2026-10-11 (D-R3.3 item
3).

### 1.4 History of this design

The first draft of this document (D-R3.1, reviewed in D-R3.2; commit
`7694035`) registered one arm, replicate 1234, on twenty instances, seeds
300–319, and ran the ensemble as a descriptive secondary arm that "decides
nothing", could be reported "with intervals only", could "not appear in the
abstract or the conclusion" and could "not replace the primary arm after
the fact". On 2026-10-11, before freeze 1 and before any pass, manifest or
rehearsal on the block, the author, on the lead's recommendation (§3.3
item 7), decided two things (D-R3.3):

1. **The ensemble becomes a second confirmatory family** with its own
   hypotheses, its own Holm correction and its own stability family, beside
   replicate 1234, which stays the primary family. The earlier draft's
   prohibition is kept in the form that still has a meaning: the ensemble
   cannot replace the primary family, be promoted over it after the fact,
   or be reported without it (§14). What changed is that the ensemble's
   tests are registered before any data exist, instead of being forbidden.
2. **The block is forty instances, seeds 300–339**, instead of twenty. Seeds
   320–339 were part of the reserve of the first draft; they left it by
   this decision, before any pass on any of them, for the power reason of
   §5.1, which is the lead's — not to rescue a result, since none existed.

Both changes were made with knowledge of the selection data of §3.2 and
with no knowledge of any result on the block: there is none. **What the
author had seen when he chose, exactly** (D-R3.3): for the first change,
the Phase 0 report v1.1 and one bootstrap power figure for twenty
instances (the "88 %" of §3.3 item 7); for the second, the first draft's
power table for twenty instances and the lead's figures for twenty, thirty
and forty from an ad hoc run of the same method (6000 draws; the figures
for twenty and forty differ from §5.1 by at most one point in the second
decimal, and those for thirty are not reproduced in §5.1, which has no
row for thirty). **The tables of
§5.1 for forty instances and the rows for the ensemble were computed
afterwards**; they were in the text he was sent before his go-ahead (§17),
which is not an approval section by section, and they are not part of what
he saw when he selected. **In
each of the four questions put to him on the design the author selected
the option the lead had marked as recommended** (D-R3.3 quotes all four),
and he did so again in the question about the seed-audit lines (§13.1).

## 2. System under test: the R2 code, pinned

- **Engine, arms, training code, analysis: the tree `6b53f3f`, unchanged.**
  `confirmatory/ENGINE_PIN_6b53f3f.txt` is the output of `git ls-tree -r
  6b53f3f` for the 62 files that make an arm what it is: `sixg_sim/` (40
  files), `config/` (12), `run_timeline_comparison.py`,
  `train_on_comparison.py`, `confirmatory_analysis.py`,
  `build_benchmark_manifest.py`, `make_untrained_checkpoints.py`,
  `tools/hash_invariance_check.py`, `tools/series_diff.py`,
  `tools/compare_reports.py`, `tools/analysis_manifest.py` and
  `confirmatory/stage_eval.sh` (the script that launches every evaluation
  invocation). `chain_plan.engine_pin_check` recomputes each file's git blob
  id from the bytes on disk and also refuses a file ADDED to `sixg_sim/` or
  `config/`. It runs in `tests/test_r3lite_plan.py` (every suite run) and
  again inside the chain at freeze 2 (§13), where a difference stops the run
  before any confirmatory pass.
- **What differs from `6b53f3f` in the R3-lite tree:** the plan
  (`confirmatory/chain_plan.py`: two new profiles and their stage graph; the
  four R2 profiles are bit-for-bit unchanged, pinned by digests in
  `tests/test_r3lite_plan.py`), the launcher and stage scripts
  (`confirmatory/run_final.sh`, `chain_lib.sh`, `stage_prep.sh`,
  `stage_analysis.sh`, the new `stage_record.sh`), `tools/stage_report.py`,
  `tools/preflight.py`, `tools/seed_audit.py`, `tools/run_seed_audit.sh`,
  the new `tools/assemble_family.py` (§8), tests and documents. None of
  these is imported by the simulator, and none of them changes a pass: the
  assembler copies passes between pickles and verifies the copies.
- **The simulator and the arms' models** are described in R2 §2 (routing-graph
  consistency, the shared per-tick step and per-site action lock, the
  link-state service, bridge stability, control postcards, the observation
  schema `obs1-542eb7fc0d277d1d`, the four action heads, the frozen reward,
  the corrected UE-transit and SDN models, XL-DET's corrected interference
  brake). Every disclosure made there about a comparator applies unchanged.
- **Scenarios.** A (`recovery`): severance at tick 200 of 3600. B
  (`rescue_ops`): severed from tick 0, demand surge at tick 1400. Scenario
  seed = topology seed (`build_comparison_scenario`), as in R2.

## 3. The policy under test and how it was chosen

### 3.1 The policy

- **Arm:** `marl_static` — the frozen executor of R2 (argmax of the discrete
  heads, Dirichlet mean of the PRB head, no learning and no parameter change
  during a pass; a change raises `STATIC CHECK` inside the harness).
- **Weights:** `policy_final.pt` of replicate **torch seed 1234** of the R2
  training run, **sha256
  `54eaa5623148cfd44faf7e69bfe7825c396541b33d9357a07d782d559de5c4bc`**
  (`~/MARL_run_9/output/final/ck_s1234/policy_final.pt`, 101 221 bytes,
  written 2026-10-05 by chain stage S2 of R2). Trained by `train_on_comparison.py
  --episodes 400 --episode-ticks 800 --seed-range 42-49,100-139 --torch-seed
  1234 --base-seed 7 --batch-size 16 --eval-episodes 0 --baseline-episodes 0`
  on training topologies 42–49 and 100–139 (event seed of episode *k* =
  700000 + *k*), under the R2 engine and reward. No instance of §5 was
  generated in training.
- **No retraining, no fine-tuning, no change of the file.** Chain stage S0
  copies it out of the frozen R2 tree and accepts the copy only with that
  sha256 (`confirmatory/stage_prep.sh`, step `import`); stage S3b verifies
  the tracked record `confirmatory/SELECTED_r3lite.txt` against the plan's
  pin and the file on disk; the launcher refuses every invocation whose
  checkpoint file does not have the pinned digest; and every result pickle
  must record that digest in its provenance (`tools/stage_report.py --verify
  --expect-checkpoint-sha256`).

### 3.1b The second policy: the ensemble of the four replicates (family E)

- **Arm:** `marl_static`, the same frozen executor, with
  `MARL_EVAL_CHECKPOINT` equal to the comma-separated list of the four
  imported `policy_final.pt` files of R2's training run **in the order of
  torch seeds 1234, 2345, 3456, 4567**.
- **What the ensemble is, exactly as the code computes it**
  (`sixg_sim/agent.py`, class `EnsemblePolicy`, a file of the engine pin;
  `run_timeline_comparison.py` builds it from the list in the given order):
  at every decision the four actors are evaluated on the same observation
  and **the outputs of every head are averaged elementwise over the four
  members** — the logits of the discrete heads, and the parameters of the
  PRB head before the softplus. `marl_static` then acts as for a single
  policy: the argmax of each averaged discrete head and the Dirichlet mean
  of the averaged PRB parameters. No member is retrained or changed; the
  ensemble has no parameters of its own.
- **Weights:** the four files, each identified by its sha256, in that
  order: `54eaa5623148cfd44faf7e69bfe7825c396541b33d9357a07d782d559de5c4bc`
  (1234), `c86ef14f051a92871db8b5675ac22f6bf0c736d45a36e22dd6b55d33486617e4`
  (2345), `53e927f87643835b96b7154e8a5538c640c7489799926db66a593e9b76241740`
  (3456), `7e640ce090b4a414de5abb0aa5847288b599c0547eca768213de1e335406dfa9`
  (4567, R2's pick). They are imported, pinned and verified exactly as the
  file of §3.1; the launcher refuses the ensemble invocation unless all
  four files have their pinned digests, and its pickle must record the four
  digests in that order.
- This is the arm R3 Phase 0 ran on the development and validation
  instances (`ens_dev.pkl`, `ens_val.pkl`: the same four files in the same
  order).
- **Where the ensemble comes from, and where it has been run.** The
  construction is older than R2: `EnsemblePolicy` was written in the first
  study, in September 2026, because replicates of one training recipe froze
  into different configurations (its docstring in the pinned
  `sixg_sim/agent.py` says so). It was evaluated then on development seeds
  50–59 and validation seeds 60–66 with that study's replicates on the
  uncorrected simulator, in the adapting arms (three pickles in the first
  study's tree, 12 and 14 September). **No ensemble was run on seeds 80–99
  or on seeds 200–219**: the first test ran one checkpoint and R2 one
  replicate, and no environment file of the R2 chain carries a checkpoint
  list (checked on the host on 2026-10-12 by listing every ensemble pickle
  and log of every tree and reading the seed keys of the three pickles; no
  result was opened). The ensemble influenced no decision of R2. The four
  files averaged here are R2's replicates, and their only passes before
  this study are Phase 0's, on the selection instances.

### 3.2 The selection rule and the numbers it was applied to

**Rule (paired-achievability rule).** For each candidate, the per-instance
steady-state achievability of `marl_static` executing it minus XL-DET's on
the same instance, in Scenario A and in Scenario B; the candidate's score is
**min_AB = the smaller of the two scenario means** of that paired difference
(maximin), over the **development and validation instances together**
(seeds 50–59, 70–79 and 60–66: 27 instances per scenario). The candidate
with the largest score is selected. Implemented in
`~/MARL_r3/r3out/phase0/tools/select_replicate.py` (sha256
`3dbfb268365a5394…`, metric helpers imported from `confirmatory_analysis.py`
through `p0lib.py`, sha256 `84ecaa6df789596b…`), `--criterion min`; its
self-test reproduces R2's development report and R2's own selection.

**Data.** Existing passes only, all at 3600 ticks in the R2 chain
environment on tree `6b53f3f`: XL-DET on development from R2's S1 pickles
and on validation from Phase 0 (`val_xldet_sdn.pkl`); replicate 4567 from
R2's S4 and S3; replicates 1234, 2345, 3456 on validation from R2's S3 and on
development from Phase 0 (`dev_ck1234_a…d.pkl`, `dev_ck2345.pkl`,
`dev_ck3456.pkl`); the ensemble of the four from Phase 0 (`ens_dev.pkl`,
`ens_val.pkl`). Every pickle's provenance was checked (R2 environment, tree
head, tick count, hash seed 0, checkpoint sha256).

**Result** (`results/select_replicate_S1_min.json`, sha256
`a61ba7f3cb1e5824…`; paired difference against XL-DET in percentage points,
bootstrap 95 % interval, instances won, exact two-sided Wilcoxon p):

| rank | candidate | Scenario A (n = 27) | Scenario B (n = 27) | mean_AB | **min_AB** |
|---|---|---|---|---|---|
| 1 | **replicate 1234** | +8.89 [+5.73, +12.10], 23/27, p 1.4e-5 | +13.46 [+10.46, +16.33], 25/27, p 1.0e-7 | +11.18 | **+8.89** |
| 2 | ensemble of the four | +5.63 [+2.82, +8.41], 23/27, p 1.6e-4 | +9.73 [+6.50, +12.97], 23/27, p 9.5e-6 | +7.68 | +5.63 |
| 3 | replicate 3456 | −0.01 [−3.06, +3.03], 13/27, p 0.84 | +3.39 [−0.46, +7.31], 14/27, p 0.14 | +1.69 | −0.01 |
| 4 | replicate 4567 (R2's pick) | −0.75 [−3.10, +1.59], 11/27, p 0.32 | +0.43 [−2.24, +3.14], 14/27, p 0.86 | −0.16 | −0.75 |
| 5 | replicate 2345 | −13.18 [−16.80, −9.70], 2/27, p 1.5e-7 | −7.15 [−10.29, −3.93], 5/27, p 5.4e-4 | −10.17 | −13.18 |

The same order results under the mean criterion and on each subset:

| set | candidate | A | B | min_AB |
|---|---|---|---|---|
| development (n = 20) | replicate 1234 | +10.11 [+6.29, +14.03], 17/20, p 1.05e-4 | +12.86 [+9.97, +15.71], 19/20, p 3.81e-6 | +10.11 |
| | ensemble | +6.26 [+2.53, +9.86], 16/20, p 0.003 | +9.16 [+5.32, +12.96], 16/20, p 4.0e-4 | +6.26 |
| | replicate 3456 | +0.48, 10/20 | +3.15, 11/20 | +0.48 |
| | replicate 4567 | −0.15, 10/20 | −0.06, 11/20 | −0.15 |
| | replicate 2345 | −13.85, 2/20 | −7.76, 3/20 | −13.85 |
| validation (n = 7) | replicate 1234 | +5.39 [+0.58, +9.37], 6/7, p 0.078 | +15.19 [+7.13, +22.10], 6/7, p 0.031 | +5.39 |
| | ensemble | +3.84 [+1.96, +5.74], 7/7, p 0.016 | +11.35 [+5.15, +17.57], 7/7, p 0.016 | +3.84 |
| | replicate 3456 | −1.41, 3/7 | +4.08, 3/7 | −1.41 |
| | replicate 4567 | −2.47, 1/7 | +1.83, 3/7 | −2.47 |
| | replicate 2345 | −11.28, 0/7 | −5.43, 2/7 | −11.28 |

**Everything that was examined.** Five candidates — the four replicates and
their ensemble — and **six rule runs**: two criteria (the smaller of the two
scenario means, `min`; their mean, `mean`) on three instance sets. All six
give the same order: 1234, ensemble, 3456, 4567, 2345.

| instance set | criterion | score of replicate 1234 | order | result file (`results/…`), sha256 |
|---|---|---|---|---|
| development + validation (n = 27) | min (**the registered rule**) | +8.89 | 1234, ens, 3456, 4567, 2345 | `select_replicate_S1_min.json`, `a61ba7f3cb1e5824…` |
| development + validation | mean | +11.18 | the same | `select_replicate_S1_mean.json`, `12bf7114aea69e7b…` |
| development (n = 20) | min | +10.11 | the same | `select_replicate_dev_min.json`, `0df0306727209e5c…` |
| development | mean | +11.48 | the same | `select_replicate_dev_mean.json`, `69643404954205c6…` |
| validation (n = 7) | min | +5.39 | the same | `select_replicate_val_min.json`, `cd8ddbcd80024eeb…` |
| validation | mean | +10.29 | the same | `select_replicate_val_mean.json`, `966d330ed2c168a3…` |

The other five runs are the sensitivity of the choice to the criterion and
to the instance set; they decide nothing. These six are all the runs of the
tool on record (`results/select_replicate_*`). One further rule was
formulated and not registered: the guarded rule of §3.3 item 7, which
selects a different arm.

**Selected for family 1: replicate 1234**, the pick of the registered rule.
**Registered as family E: the ensemble**, ranked second by the registered
rule and first by the guarded one **at g = 2** (g was not fixed: with g ≥ 9
the guarded rule selects replicate 1234 too; §3.3 item 7). No other candidate is run
on the block: replicates 2345, 3456 and 4567 are run there only as members
of the ensemble.

### 3.3 Disclosures about the selection

1. **The rule was chosen after the results were seen.** The
   paired-achievability rule was formulated and applied after the development
   and validation results of all four replicates (and of the ensemble) were
   known. It is not a pre-specified rule; it is the reason this study needs
   fresh instances (§1.3). The p-values and intervals in §3.2 describe the
   selection data and are not evidence for any hypothesis of §8.
2. **The development seeds are R2's development seeds, and the validation
   seeds R2's validation seeds.** Both were also used before R2 for pipeline
   tests (R2 §4).
3. **R2's rule and the new rule disagree.** R2's rule (components, Scenario A,
   validation) picks 4567; on the same validation data the new rule picks
   1234. Had R2 been pre-registered with the new rule, 1234 would have been
   its arm. It was not, and R2's result is the result for 4567.
4. **One replicate of four.** The four replicates span −13.2 to +8.9 pp
   against XL-DET in Scenario A. Whatever R3-lite finds is a statement about
   this policy, selected this way; it is not a statement that the training
   method reliably produces such a policy (§14).
5. **Relay churn.** On development, replicate 1234 tears relay links down
   8.2 times as often as XL-DET in Scenario A and 7.2 times in B (M7: 12.18
   against 1.49 per 1000 post-event ticks in A, 10.84 against 1.50 in B;
   higher on 20/20 instances in both). The registered rule ignores this. It
   is the subject of item 7, of the stability family (§9) and of an
   expectation stated in advance (§11).
6. **What was and was not tuned on the selection instances.** The policy,
   the executor, the engine and the checkpoint were not changed after R2's
   freeze. One thing had been tuned on some of these instances before R2:
   XL-DET's interference brake was repaired after three variants were
   compared on development seeds 50–53 (R2 §2 item 11). The variant kept is
   the strongest comparator of the three, so the direction is conservative
   for P1 and P4, but the comparator was chosen on those seeds.
7. **Phase 0 recommended a different rule, and under it a different arm is
   selected.** The Phase 0 report (v1.1, its sections 1 and 6, item 1) made
   going ahead conditional on pre-registering "selection by paired delivery
   **with a relay-churn guard**": rank by the paired difference on
   validation, keep only candidates whose M7 is at most g times XL-DET's
   mean in each scenario (and with no instance above 50 teardowns per 1000
   ticks and J per unit not significantly above XL-DET's), and require the
   pick to hold on development. It asked the author to fix g "on principle"
   beforehand, because g decides between the two candidates: **with g = 2
   the guarded rule selects the ensemble** (M7 0.96 and 1.00 times XL-DET's
   on development; +6.26 / +9.16 pp), and replicate 1234 (8.2 and 7.2 times)
   passes only with g ≥ 9. The report's summary table says of the ensemble:
   "This is the candidate the recommended rule (item 1) selects." **The
   author took two decisions on 2026-10-11, both before any pass on the
   block.** First (D-R3.1): to test replicate 1234 by the paired rule.
   D-R3.1 records that decision and nothing about a guard: no decision
   about the guard and no reason for leaving it out is on record (D-R3.2
   finding 3). Then (D-R3.3 item 1), after the Phase 0 report v1.1 and one
   bootstrap power figure had been put to him together with four written
   options: to keep replicate 1234 as the primary arm **and to register the
   pick of the guarded rule at g = 2, the ensemble, as a second
   confirmatory family with its own correction.** The option he selected
   reads, word for word as the lead put it to him: *"1234 primary +
   combined second (Recommended) — Seed 1234 is primary (largest margin,
   about 8x relay churn). The combined policy is a second pre-registered
   family with its own correction (stable bridging, 88% chance on both
   scenarios). 20 seeds, about 12 h of compute."* **It was the option the
   lead had marked as recommended, as was the option the author selected in
   each of the three other questions** (D-R3.3). The three options he did
   not select were: the ensemble as the primary family on thirty instances;
   the ensemble as the primary family on twenty; and replicate 1234 alone,
   with the ensemble descriptive. **The reasoning behind the design is the
   lead's**: the lead drew up the options and recommended
   one in each question, and the author accepted the recommendation each
   time. On 2026-10-12 the lead asked him for a typed approval and offered
   him the chance to add a reason of his own for testing both policies; he
   replied: *"What do you mean? This is something we did together. The
   logic to running this came from you."* No reason is attributed to him
   here; D-R3.3 quotes the questions and the answers in full, and §17 and
   D-R3.5 record this exchange. Two remarks on the wording of the option quoted
   above, which was the lead's and is not a result: "stable bridging" describes the
   ensemble's relay churn on the development instances (M7 equal to
   XL-DET's) and is **not** a property this study may claim for it (§9,
   §11 item 5); and "88%" was a bootstrap figure for twenty instances that
   §5.1 replaces. The facts are disclosed as they are: the guard was
   proposed before the choice; it was not adopted as a selection rule; the
   arm it selects is tested here as a family of its own, registered before
   any data; and the churn the guard was meant to exclude is tested for
   both policies by their stability families, with the expected outcomes
   written down in advance (§11 items 4 and 5).
8. **A NO-GO condition of Phase 0 that the registered choice does not
   meet.** The same report lists "no eligible candidate" as a reason not to
   run R3 at all, where eligible means: paired achievability above XL-DET
   with an interval above zero in both scenarios on the 27 instances, the
   churn guard, and J per unit not significantly above XL-DET's. The
   ensemble meets all three; replicate 1234 meets them only with g ≥ 9.
   Family 1 therefore asks whether replicate 1234 delivers more than XL-DET
   on fresh instances and reports its churn. Whether a policy that passes
   the guard on the selection data does so is asked by family E, and what
   may be said about that policy's churn on the fresh instances is limited
   by §9 and by what its stability family can assess (§11 item 5, §14).

## 4. Development and validation evidence: cited, not re-run

R3-lite has no development, stress or validation stage. The evidence that
exists is §3.2 (and the Phase 0 tables `results/0ab_tables.md`, sha256
`5ea91345c12b070b…`, and `results/0ab_supplement.md`, sha256
`1074f3256eef5644…`); it is descriptive, it was used for the selection, and
it enters no test. The technical purposes R2's development stages served are
covered as follows: the engine is byte-identical to the one R2's gates
passed on (§2); the analysis script is byte-identical to the one that ran on
R2's development and confirmatory sets (§8); the pipeline of this study is
rehearsed end to end on two development seeds (profile `r3lite_rehearsal`,
§13). The stages R2 ran and R3-lite does not are listed, with reasons, in
§16; the plan records each of them and the chain writes them to its ledger
at every start.

## 5. Confirmatory instance set

- **Topology seeds 300–339**, with matching scenario seeds, both scenarios:
  **n = 40 instances per scenario**, twice R2's twenty; the same forty
  instances for both families.
  `PROFILES['r3lite']['cf_seeds']` in `confirmatory/chain_plan.py`. The plan
  builder refuses profile `r3lite` on any other set and refuses the
  rehearsal profile on any seed of a reserved block; the tests pin that no
  other profile names a seed of this block, that no profile names the
  blocks held in reserve (340–359, 400–439), and that only R2's `real`
  names R2's block (200–219) (`tests/test_r3lite_plan.py`,
  `tests/test_run_chain.py`).
- **Why forty (D-R3.3 item 2).** The first draft registered twenty (seeds
  300–319), R2's size. The power figures of §5.1 show what that would have
  cost if the effect on fresh instances is smaller than on the selection
  data, as the validation instances suggest for Scenario A: for replicate
  1234, P1 at the validation-level estimate reaches p < 0.05/9 | p < 0.05
  with probability 0.32 | 0.69 on twenty instances and 0.77 | 0.95 on forty;
  for the ensemble, E-P1 at the development estimate itself has 0.52 | 0.85
  on twenty and 0.93 | 0.99 on forty. The block size was fixed from these
  figures and from the selection data alone. Seeds 320–339 left the reserve
  by this decision, before any pass on them; what stays in reserve is
  340–359 and 400–439, and it is not held for a further attempt at this
  question (§14).
- **Never used.** The block was reserved when R3 was planned (the Phase 0
  seed guard refused 300–359 and 400–439) and nothing has run on it: no pass,
  no rehearsal, no smoke test, no manifest. The seed audit of §13 must show
  **zero result-data hits** for all forty seeds before the run starts; that
  is the gate, and it is stricter than R2's (R2's block had a disclosed
  rehearsal exposure on four seeds).
- **Generation.** `build_benchmark_manifest.py --seeds 300-339`, run by chain
  stage S0 at the start of the run. The manifest holds topology-derived
  planning data only (attainable optimum, reunifiability, raw component
  count, severance tick) and no arm's result; as in R2 (D-FIX.9) building it
  is planning, not exposure. Its sha256 is sealed at freeze 2.
- **Exclusions.** None. A failed invocation is repeated once, unchanged
  (§13); any drop is reported with its cause.

### 5.1 Power, from the selection data (stated before the run)

Paired differences against XL-DET on the 20 development instances
(`results/select_replicate_dev_min.json`). **Replicate 1234:** Scenario A
mean +10.11 pp, paired SD 9.06 (d_z 1.12, median +9.90, range −7.20 to
+29.16); Scenario B mean +12.86 pp, paired SD 6.76 (d_z 1.90, median +10.71,
range −0.33 to +25.11); on validation (n = 7) A +5.39 (SD 6.47), B +15.19
(SD 11.10). **The ensemble:** Scenario A mean +6.26 pp, paired SD 8.60 (d_z
0.73, median +6.16, range −15.28 to +23.70); Scenario B mean +9.16 pp,
paired SD 8.92 (d_z 1.03, median +9.46, range −9.50 to +29.54); on
validation A +3.84 (SD 2.75), B +11.35 (SD 9.06).

Probability that the exact two-sided Wilcoxon test on n = 40 differences
gives p below a threshold in the expected direction, differences drawn from
a normal distribution with the stated mean and SD (20 000 draws, generator
start 20261011). p < 0.05/9 is sufficient for confirmation under Holm
whatever the other eight p-values of the family are; p < 0.05 is necessary.
**The script and its output are `docs/POWER_R3LITE.md`** (the script
extracted from it has sha256
`8d0fe85828978ec121da51c01cb6687c6a6b29e70867d4160f818f9af91cc4b2`); every
number below is that script's, run on 2026-10-11. The same file prints the
tables for twenty instances, the first draft's block.

**Family 1, replicate 1234, n = 40**

| assumed true effect | P1 (A): p < 0.05/9 | p < 0.05 | P4 (B): p < 0.05/9 | p < 0.05 | **P1 and P4 both**, p < 0.05/9 |
|---|---|---|---|---|---|
| the development estimate | > 0.999 | > 0.999 | > 0.999 | > 0.999 | > 0.999 |
| half the development estimate | 0.69 | 0.92 | 0.997 | > 0.999 | 0.70 |
| a third of it | 0.28 | 0.61 | 0.83 | 0.97 | 0.25 |
| the validation estimate (A +5.39, B +15.19), development SD | 0.77 | 0.95 | > 0.999 | > 0.999 | 0.76 |
| the development estimate, R2's confirmatory paired SD (8.40, 10.04) | > 0.999 | > 0.999 | > 0.999 | > 0.999 | — |

**Family E, the ensemble, n = 40**

| assumed true effect | E-P1 (A): p < 0.05/9 | p < 0.05 | E-P4 (B): p < 0.05/9 | p < 0.05 | **E-P1 and E-P4 both**, p < 0.05/9 |
|---|---|---|---|---|---|
| the development estimate | 0.93 | 0.99 | 0.999 | > 0.999 | 0.93 |
| half the development estimate | 0.26 | 0.59 | 0.60 | 0.87 | 0.21 |
| a third of it | 0.09 | 0.31 | 0.22 | 0.54 | 0.05 |
| the validation estimate (A +3.84, B +11.35), development SD | 0.44 | 0.77 | > 0.999 | > 0.999 | 0.44 |
| the development estimate, R2's confirmatory paired SD (8.40, 10.04) | 0.95 | 0.99 | 0.995 | > 0.999 | — |

The last column draws the two scenarios' differences jointly, with the
correlation seen across the development instances (0.33 for replicate 1234,
0.50 for the ensemble). Smallest true effect detected with 80 % (90 %)
probability at p < 0.05/9 on forty instances: 5.7 (6.3) pp in A and 4.2
(4.7) pp in B for replicate 1234; 5.4 (6.0) pp in A and 5.6 (6.2) pp in B
for the ensemble (d_z 0.62 / 0.70 in each).

**Where the rows "half" and "a third" come from.** They are stress levels,
not estimates, and they are not derived from the spread of the candidates.
A bootstrap of the selection itself (instances resampled, the registered
rule re-applied to the five candidates, the pick scored on the instances
left out; same file) finds almost no optimism from choosing the best of
five on these instances: replicate 1234 is picked in 99.4 % of resamples,
and its estimate in a resample exceeds its estimate on the instances left
out by 0.0 pp in A and −0.1 pp in B on average. What resampling these
instances cannot measure is the larger risk: the development instances were
used throughout R2's development (§3.3 item 2), and forty new topologies
need not resemble them. The one partly independent figure there is, the
validation estimate in Scenario A, is 53 % of the development one for
replicate 1234 and 61 % for the ensemble. The rows below the first are
therefore the ones to read.

**What forty instances buy, and what they do not.** On twenty instances
(the first draft; same script, same draws) the figures for replicate 1234
were 0.94 | 0.996 for P1 at the development estimate, 0.27 | 0.63 at half
of it and 0.32 | 0.69 at the validation-level estimate, and "P1 and P4
both" 0.25 at half; for the ensemble, E-P1 had 0.52 | 0.85 at the
development estimate itself. Forty instances double the run (§6: 804
passes, about a day on the shared host). They do not make every row safe:

Reading, fixed now ("adequately powered" means a strict power of at least
0.80, the first figure of each pair). **Family 1: P4 is adequately powered
even at a third of the development effect; P1 is adequately powered at the
development estimate, is marginal at the validation-level estimate (0.77 |
0.95: that estimate, +5.39 pp, is just below the smallest effect detected
with 80 %), and is not adequately powered at a third of the development
estimate. Family E: adequately powered at the development estimates; below
them E-P1 is under-powered (0.26 | 0.59 at half, 0.44 | 0.77 at the
validation level) and E-P4 is marginal at half (0.60 | 0.87).** An
achievability hypothesis against XL-DET (P1, P4 and their family-E
counterparts) that is not confirmed with a positive estimate is therefore
to be reported as inconclusive, not as evidence of no difference (§14) —
for family E in particular. The smallest attainable exact two-sided p at n = 40 is 1.8e-12.

## 6. Arms and runs

Chain stages of `confirmatory/chain_plan.py`, profile `r3lite` (stage graph
`STAGES_R3LITE`), every pass 3600 ticks, `PYTHONHASHSEED=0`, `nice -n 19`,
`MARL_POOL_WORKERS` = 24, one harness invocation per row
(`run_timeline_comparison.py --seed-list <seeds> --arms <arms> --no-figures
--out r3lite/<id>.pkl`):

- **S0, preparation.** `s0_manifest` (§5) and `s0_import`: the five R2 files
  of §3.1, §3.1b and S8 below copied out of `~/MARL_run_9/output/final/` (read-only) into
  `output/r3lite/`, each accepted only with its pinned sha256.
- **S3b, selection record.** `s3b_record` verifies
  `confirmatory/SELECTED_r3lite.txt` (seven checks; it computes nothing).
- **S5, freeze 2.** `s5_freeze2` seals `confirmatory/FREEZE_FINAL_r3lite.txt`
  (§13).
- **S6, confirmatory, 10 invocations** `s6_cf_b0` … `s6_cf_b9` of 4 seeds × 8
  arms × 2 scenarios = 64 passes each (640): the **eight arms of R2, in R2's
  order and R2's block splitting**: XL-DET (`xldet`), uniform random
  (`random`), SDN/OpenFlow (`sdn`), OSPF, OLSR, B.A.T.M.A.N., AODV (`ospf`,
  `olsr`, `batman`, `aodv`) and MARL-RIC (`marl_static`, replicate 1234).
  The plan test compares these ten invocations with R2's five field by
  field: arms, environment, scenarios, tick count, worker count and script
  are identical; block *j* is R2's block *j* mod 5 with every seed shifted
  by +100 (*j* < 5) or +120 (*j* ≥ 5), and the checkpoint is pinned. **These
  passes are the data of family 1 and the comparator passes of both
  families.**
- **S9x, the confirmatory passes of family E, 1 invocation**
  (`s9x_ensemble`, 80 passes, **right after S6 and before any analysis**):
  `marl_static` executing the ensemble of §3.1b on all forty instances, both
  scenarios, in the environment of S6, written to its own pickle, which the
  family-1 analysis never reads. It depends on the seal only, never on a
  result: **S9x is never shed, skipped or repeated for any content of any
  report.** The launcher starts S7 only after it (an ordering, not a
  dependency: §13 says what happens if it cannot be completed).
- **S7, the analysis of family 1 — the primary result** (§8, §9, §10).
- **S7e, the analysis of family E** (§8, §9), after S7 and never before it:
  the same analysis on an assembled pickle set (§8, "How family E is
  analysed").
- **S6h, the hash-seed repeat, 1 invocation** (`s6h_hash1`, 4 passes;
  ordered after the analysis of family 1, and in the registered run after
  both analyses): `marl_static`, replicate 1234, seeds 300 and 301, both
  scenarios, `PYTHONHASHSEED=1` (§10 S8).
- **S8, untrained-actor control, 1 invocation** (`s8_untrained_s1234`, 80
  passes; ordered as S6h): `marl_static` executing the untrained
  initial actor of torch seed 1234, the file R2 generated and ran
  (`~/MARL_run_9/output/final/untrained_s1234/policy_init.pt`, sha256
  `59e927c85d679d79c1054c92bf3315af9556c5148a94534cbbd9ed66a887a6cc`),
  imported byte-identical.
- **S12, secondary analyses** (§10). It depends on family 1 and its two
  secondary stages only, and is ordered after S7e: nothing family 1 needs
  depends on family E, so that the contingency of §13 can run to its end.

Total 804 passes: 640 + 80 + 4 + 80. The two single-arm invocations hold the
whole block in one pickle each (80 passes; a multi-arm block holds 64),
because the analysis reads one pickle per control and the family-E set is
assembled from one ensemble pickle (`chain_plan.R3_SINGLE_ARM_MAX_PASSES`).
Nothing is shed: profile `r3lite` declares no shed group and the launcher
refuses `--shed` for it. The stage numbers of the chain and the analysis
labels S1–S12 of §9–§10 are different namespaces, as in R2 (chain S8 = the
untrained control; analysis S8 = hash invariance); chain S7e is the
family-E analysis and has nothing to do with the stability endpoint S9e.

## 7. Metrics

Unchanged from R2 §7; the definitions are those of the helpers in
`confirmatory_analysis.py`. For a run record *r* of one arm on one instance:
`num(xs)` keeps the numeric, non-NaN entries of a series; `W` = 500.

### 7.1 Primary metrics

- **`steady(xs)`** = mean of `num(xs[-W:])`, the last 500 ticks.
- **Achievability** (per instance) = `steady(r['conn'])`, in percent of
  offered traffic delivered.
- **Energy per delivered unit** (per instance) = `jpu(r)` =
  `sum(num(r['energy'])) / sum(num(r['delivered']))` over the whole run.
- **Components** = `steady(r['fragments'])`; an instance is at optimum iff
  `steady(fragments) ≤ attainable_optimum + 1e-9` (the manifest's value).
- **`wmean(xs, lo, hi)`** = mean of `num(xs[lo:hi])`; pre-surge level
  (Scenario B) = `wmean(conn, 900, 1400)`; pre-event level (A) =
  `wmean(conn, 0, 200)`; surge change (B) = `steady(conn) − wmean(conn, 900,
  1400)`.
- **Service-loss integral** (A) = `sli(r, 200)` as in R2 §7.1.
- **Best routing** (per instance): the maximum achievability, or the minimum
  J/unit, over `ospf, olsr, batman, aodv`.

### 7.2 Stability metrics

f = `fragments`, r = `fragments_raw`, o = `attainable_optimum`, t_s =
severance tick (`T_SEV`: A 200, B 0), T = 3600 (asserted). Change counts use
t ∈ [t_s+1, T).

| ID | helper | definition |
|---|---|---|
| **M2** policy-induced splits | `m2_policy_splits(r, ts)` | \|{ t ∈ [t_s+1, T) : f[t] > f[t−1] and r[t] ≤ r[t−1] }\| |
| **M4** fraction of post-event ticks at the attainable optimum | `m4_fraction_at_optimum(r, ts, o)` | mean over t ∈ [t_s, T) of 1[f[t] ≤ o] |
| **M7** relay-link teardowns per 1000 post-event ticks | `m7_teardowns_per_1000(r, ts)` | 1000 × Σ_{t ∈ [t_s+1, T)} `relay_removed`[t] / (T − t_s − 1), all teardown reasons |

M1, M3, M4ss, M6, M8, M9, M10 and M11 are defined in R2 §7.2 and reported
descriptively (S10) as medians with inter-quartile ranges.

## 8. Primary hypotheses: two families, each Holm over its own 9, α = 0.05

**The nine hypotheses, the decision test and the rule of R2 §8, unchanged,
applied twice: once per family.** In **family 1** MARL-RIC = `marl_static`
executing replicate 1234 and the hypotheses are named **P1–P9**. In
**family E** MARL-RIC = `marl_static` executing the ensemble of §3.1b and
the same nine hypotheses are named **E-P1–E-P9**. In each family: exact
two-sided Wilcoxon signed-rank test (`wilcoxon(d)`: zeros dropped,
mid-ranks for ties, exact enumeration of the signed-rank distribution) on
the 40 paired per-instance differences; Holm step-down over that family's
nine p-values (`holm`). A hypothesis is **confirmed** iff its Holm-adjusted
p < 0.05 **and** the signed-rank statistic lies in the expected direction
(`W+ > W−` for "> 0", `W− > W+` for "< 0"). The comparator passes are the
same passes in both families (below); the table gives the hypotheses of
family 1, and family E's are the same rows with the prefix `E-`.

| ID | Scen. | Quantity | Comparison | Expected |
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

The descriptive statistics are the paired mean, the 95 % t-interval
(`t_ci`), wins and the exact sign test (`sign_p`). When they disagree with
the decision test the disagreement is reported; they never override it. A
Holm-adjusted p < 0.05 in the direction opposite to the expected one is
reported as such ("reversed"), as P8 was in R2. As in R2 §8, P2, P5 and P8
are read together with the disclosures about the SDN model (R2 §2 item 10).

**The analysis code is pinned.** `confirmatory_analysis.py`, md5
`e6142b2798937b79bafe0314c98dbba0` (sha256
`9d2bdaa2e7ffc360f1555ec915f0eebb14e25c2b7f2fb29a06776b78dbf04311`, git blob
`6b020853ac01a45614089d61dc8b5b798fb7e86e`): the file that produced the R2
report. It is invoked exactly as in R2, `--marl-arm marl_static`, without
`--no-assert` (§12). The chain refuses to hand it a pickle unless the file
on disk has that md5 (`confirmatory/stage_analysis.sh`, every analysis
step; again at freeze 2). Per family: one test and one decision report.
The analysis reads family 1's data three times — S7, the assembly control
of S7e and the full report of S12 — and the second and third are
reproductions that must equal the S7 report leaf by leaf (below and §10);
they are not further tests and nothing is decided by them. There is no
interim look — the launcher's gates between S6 and S7 count keys and
compare provenance and never open a metric — and no second analysis of
either family's endpoints.

**How family E is analysed** (chain stage S7e, `confirmatory/stage_analysis.sh`
step `family_e`). The analysis tests one learned arm, `marl_static`, against
the comparators of the same pickle set, and the file may not change. For
family E it is therefore handed an **assembled pickle set**: for each of the
ten S6 pickles, one derived pickle in which every comparator pass is the S6
pass, untouched, and every `marl_static` pass is the ensemble's pass of the
same instance from the S9x pickle. The set is built by
`tools/assemble_family.py` (a new tool of this study, part of the frozen
tree; its sha256 is in `FREEZE_R3_<head>.sha256` and is recorded again by
the chain in the family-E record). The tool refuses, writing nothing: a
seed of the block without an ensemble pass in either scenario, a seed twice,
a seed outside the block; a source pickle that does not record exactly the
four pinned digests in order; any difference between the input pickles in
tree head, tree digest, schema, reward id, hash seed, pass length, library
versions or engine environment; and a pass whose offered-demand series is
not that of XL-DET on the same instance. After writing, it re-reads every
derived pickle and compares a canonical digest of every run with the digest
of the input it came from. The order of the step is fixed in the script:

1. **Family 1 first, and the passes of family E before it — both read from
   the chain ledger, not asserted by the step.** The S7 report must exist
   and must be the file whose sha256 the ledger recorded when S7 ended; and
   every S9x invocation must have ended before the successful attempt of S7
   started (`tools/stage_report.py --ledger-order`; the ledger is
   append-only, so record order is time order). A set in which family E was
   run after family 1 had been analysed is refused, whatever it contains.
   The report's sha256 is then written to the ledger before anything of
   family E is computed, and is checked again at the end. The launcher runs
   S7e only after S7 has ended ok.
2. **Positive control of the assembly path.** The set assembled from the S6
   pickles alone (`marl_static` = replicate 1234, the pinned digest of §3.1)
   is analysed by the same script and **must reproduce the S7 report leaf
   by leaf**, in both directions (`tools/compare_reports.py`, sections
   seeds, excluded, descriptive, primary, secondary, stability, S10, S11),
   **by count**: the two comparisons must each have compared leaves, and
   the same number. If it does not, family E is not computed and the
   failure is reported.
3. **Family E.** The set assembled from the S6 pickles and the S9x pickle
   (the four pinned digests) passes the provenance and key-union gates (8
   arms × 2 scenarios × 40 seeds, none missing, none twice) and is analysed
   by the same script, strict mode, same manifest: report
   `confirmatory/r3lite_cf_report_family_e.json`.

A run of the step that does not reach its last line leaves no family-E
report, no text of it, no record and no assembled pickle behind, and
removes those of any earlier run when it starts. Chain S12 then binds the
family-E report to its data once more: it re-hashes the ten S6 pickles and
the S9x pickle and requires each to have the sha256 the assembler recorded
when it read it and the sha256 the ledger recorded when the invocation
ended, and it requires the per-instance listing of §10, computed from the
same passes by another route, to agree with E-P1 and E-P4 of the report in
instances, wins and mean.

Checked on real data before this text was frozen (development instances
only; first on 2026-10-11 with an earlier version of the tool, and again on
2026-10-12 **with the tool as frozen**, whose sha256 is recorded with the
check in D-R3.4): the set assembled from R2's development comparator
pickles and Phase 0's `ens_dev.pkl` verified 280 comparator runs and 40
ensemble runs against their inputs, passed the unchanged analysis in strict
mode, and reproduced Phase 0's ensemble means (+6.2584 and +9.1591 pp).

**Multiplicity, stated plainly.** Each family controls its own family-wise
error rate at 0.05 by Holm over its nine hypotheses. The two families are
**not** corrected for each other. They share every comparator pass, and the
ensemble contains replicate 1234 as one of its four members, so their
outcomes are positively dependent; the probability that at least one of the
two families confirms a hypothesis against XL-DET when neither policy
delivers more is therefore **at most 0.10** — the sum of the two families'
rates, each at most 0.05 — and not less than the larger of the two; it is
not controlled at 0.05. Three rules follow, and §14 applies them: (i) every
claim names its policy and its family; (ii) a sentence about **"the two
policies tested here"** together — never about "learned policies" or the
training method in general without the spread of the four replicates
beside it (§14) — may be written for a hypothesis only if
it is confirmed in **both** families, or confirmed in one with a
Holm-adjusted p below **0.025** (the Bonferroni split over the two
families, fixed here), and then it still names both outcomes;
(iii) the number of confirmatory looks of §1.3 is stated wherever a
confirmed hypothesis against XL-DET is quoted. No test compares the two
families, and "confirmed in one and not in the other" is not evidence of a
difference between the two policies.

## 9. Stability family S9, one per policy (each Holm over its own 6, α = 0.05)

The family, the test and the code are R2 §9's, unchanged; the wording
differs from R2's text in the places marked below and listed in §16.
MARL-RIC − XL-DET, per scenario, exact two-sided Wilcoxon test (`paired(d,
expect)`), Holm over the six (`S9_ENDPOINTS`). **It is computed once for
each policy**: for replicate 1234 by S7 (endpoints **S9a–S9f**) and for the
ensemble by S7e on the assembled set (the same six endpoints, named
**E-S9a–E-S9f**), each with its own Holm correction over six. The endpoint
ids are names of these families only; the chain stage that runs the
ensemble's passes is S9x and the one that analyses family E is S7e.

| ID | Scen. | Quantity | Expected |
|---|---|---|---|
| S9a | A | M2 | > 0 |
| S9b | B | M2 | > 0 |
| S9c | A | M4 | < 0 |
| S9d | B | M4 | < 0 |
| S9e | A | M7 | > 0 |
| S9f | B | M7 | > 0 |

Rules (`stability_family`, `MIN_NONZERO` = 6), as in R2:

- An endpoint with **fewer than six non-zero differences** is **not
  assessable**; it enters the Holm step-down with p = 1 (the family size
  stays 6) and can never be confirmed. An endpoint whose metric is not
  recorded on some instance is "not recorded" and is treated the same way
  in the step-down; as in R2 this cannot happen on a study pickle (the hard
  asserts require the series), and chain S12 fails if the primary report
  shows one (below).
- An endpoint is **confirmed in the expected direction** iff its
  Holm-adjusted p < 0.05 and the statistic has the expected sign (the
  expected signs say "MARL-RIC splits more, is at the optimum less, tears
  down more"); a Holm-adjusted p < 0.05 with the opposite sign is reported as
  "MARL-RIC better" on that endpoint. *(Wording corrected against R2: R2 §9
  printed "MARL-RIC worse" for the opposite sign, a slip, since the expected
  sign is the "worse" one. The code sets `direction_ok` and carries no
  label; it is unchanged.)*
- **Reunification may be called monotone** for an arm only if its M2 = 0 on
  every instance of the scenario.
- **What may be said about the stability of a learned policy's bridging**
  (per policy, from its own family; one scope, used in §11 and §14 too:
  **all six endpoints**, M2, M4 and M7 in both scenarios). If any of its
  six endpoints is confirmed in the expected ("worse") direction, its
  bridging may not be called stable, and the paper says which endpoint,
  with the factor for M7. An endpoint confirmed in the opposite direction
  is reported as "MARL-RIC better on that endpoint", and no more is made of
  it. If no endpoint is confirmed worse, the paper may say that its
  bridging was **not shown worse than XL-DET by S9** in a scenario, and no
  more — **and only if at least one of the M2 and M7 endpoints of that
  scenario had at least 8 non-zero differences.** Eight, not the six that
  make an endpoint "assessable". Holm multiplies an endpoint's p by seven
  minus its rank among the six. Standing alone — the other endpoints not
  significant, which is the case the sentence "not shown worse" is written
  for — an endpoint with six non-zero differences (smallest exact two-sided
  p 1/32) or seven (1/64) cannot reach an adjusted p below 0.05 whatever
  their signs (6/32 = 0.19, 6/64 = 0.094), and one with eight reaches it
  only if all eight have the same sign (6/128 = 0.047). With other
  endpoints of the family significant the multiplier is smaller: six
  non-zero differences then suffice only if all five other endpoints have
  smaller p-values, and seven only if at least three do. An endpoint below
  eight is assessable in name and, on its own, could not have shown
  anything; "not shown worse" written on it would be a check that cannot
  fail. The threshold is a rule about sentences, chosen for that case; the
  analysis and its decisions are unchanged. For a scenario in which no M2 or M7 endpoint reaches eight, the
  paper writes that **S9 could not assess the policy's bridging in that
  scenario**. **The number of non-zero differences of every M2 and M7
  endpoint is printed beside every stability statement**, not only the
  number of assessable endpoints. *(Wording narrowed against R2, whose §9
  allowed "may be called stable" in that case: a two-sided test that finds
  no difference, with no equivalence margin specified, is the absence of a
  detected difference, not a demonstration of stability; and an endpoint
  that was not assessable detected nothing at all. The decision rule and
  the code are unchanged; only the sentences the paper may write are.)*
  "Not assessable" endpoints neither support nor refute anything. The word
  "stable" is not used as a label for either policy.
- No other stability comparison is inferential; S10 is descriptive.
- As in R2 §9: XL-DET instances that settle away from the optimum under the
  grace rule (R2 §2 item 4) are part of the comparator's data and are
  reported, not removed.

Family 1's is computed by S7 from the ten primary pickles, family E's by
S7e from the ten assembled pickles. Chain S12 computes nothing about
either: it counts that all six endpoints are in each report and copies
each endpoint's status (`assessable`, `not assessable`, `not recorded`) and
verdict into the secondary report, so that the report says which endpoints
were actually assessed. "Not assessable" is an outcome and is only
recorded. An endpoint that is `not recorded` (a metric series missing from
the pickles) is a technical failure of the pipeline: S12 fails on it and it
is reported as such; it changes nothing in the two reports, which S7 and
S7e have already written.

## 10. Secondary analyses (outside every Holm family)

Computed by the same run of the analysis script as the primary result, as in
R2 §10 and with R2's definitions: **S1** reunification (at-optimum counts,
paired component difference against XL-DET), **S2** surge response (B),
**S3** pre-surge level (B), **S4** absolute energy per tick (A), **S5**
service-loss integral (A), **S6** energy per unit in B, **S10** the
descriptive stability metrics of every arm, **S11** billed control overhead
by message type, and the shared-world check (largest pre-event achievability
difference between arms in A, expected 0). The one claim rule R2 §10
attaches to these applies unchanged: the paper may say that the policy
*responds to the surge* better than a baseline only if S2 against that
baseline has Wilcoxon p < 0.05 in the positive direction.

Computed by chain S12 (`confirmatory/stage_analysis.sh`, step `secondary`):

- **S8, hash invariance.** Replicate 1234 on seeds 300 and 301, both
  scenarios, under `PYTHONHASHSEED` 0 (the S6 invocation `s6_cf_b0`) and 1
  (`s6h_hash1`) must give identical decision, delivered and fragment series
  (`tools/hash_invariance_check.py --require-decision`); the pair's
  provenance must agree in tree, schema and checkpoint and differ in the
  hash seed only (`tools/stage_report.py --hash-pair`). A difference is the
  analysis's result and is reported; a comparison that compared nothing
  fails the stage.
- **S12, the untrained-actor control for replicate 1234** (chain S8) against
  the trained policy, random and the best routing protocol: achievability,
  components, J/unit, descriptive (`confirmatory_analysis.py --untrained
  untrained_s1234=<pickle>`, in one further run of the script over exactly
  the primary input, which must reproduce every leaf of the primary report:
  `tools/compare_reports.py`, sections seeds, excluded, descriptive, primary,
  secondary, stability, S10, S11).
- **The per-instance listing of the ensemble against XL-DET** (the S9x
  pickle against the XL-DET passes of S6, the same 40 instances, both
  scenarios): the paired differences in achievability, J/unit and M7 with
  the paired mean, the 95 % t-interval and wins, per instance and in
  summary (`tools/stage_report.py --paired-descriptive`, which imports the
  metric helpers and `paired` from `confirmatory_analysis.py` and fails
  unless it compared exactly the 40 declared seeds in both scenarios, on
  one tree, with the four pinned checkpoint digests). **The listing decides
  nothing**: every inferential statement about the ensemble is a hypothesis
  of family E (§8, §9), computed by S7e. The tool also writes unadjusted
  Wilcoxon p-values into its JSON record; they are not printed in the paper
  and are not a result. **The listing exists only beside a completed
  family E**: S12 requires its achievability contrasts to agree with E-P1
  and E-P4 of the family-E report (§8), and if family E is not completed
  (§13) the listing is not computed and nothing is reported in its place.
- **The secondary analyses of family E.** The run of the analysis script
  that produces the family-E report computes S1–S6, S10 and S11 for the
  ensemble as it does for replicate 1234; they are reported descriptively
  under the same rules. **Not run for the ensemble:** the hash-invariance
  repeat and an untrained control (its members' untrained actors are four
  different networks; the ensemble of them is not an initial state of any
  training run). Both are stated as not run wherever the ensemble's result
  is reported.

  **How the two families are presented, fixed now**, because the ensemble is
  the arm Phase 0's guarded rule selects at g = 2 (g was not fixed; §3.3
  item 7) and is expected to
  show less relay churn than the primary arm (§11 item 5), which would make
  it the natural thing to promote if P1 or P4 fails:
  1. **Both families are reported in every outcome**, in full: nine
     hypotheses and six stability endpoints each, in one table with the
     same rows, family 1 first.
  2. **Family 1 is the primary family** wherever one result is named first:
     in the abstract, the conclusion and every summary sentence. A sentence
     that reports a confirmed hypothesis of family E reports the outcome of
     the same hypothesis of family 1 in the same sentence, confirmed or not.
  3. **The ensemble does not replace replicate 1234**, whatever P1 and P4
     show: a failed P1 or P4 is reported as failed, and family E is then
     described as "the pre-registered second family", never as the study's
     main finding and never without the failed family-1 hypothesis beside
     it.
  4. The words used for family E are those of §14: "confirmed" or "not
     confirmed" per hypothesis, with the count of confirmatory looks.

**Dropped against R2, and why** (also §16): R2's **secondary S7**
(`marl_static` against the adapt-then-freeze variant) — the variant is not
part of the R3-lite question and was reported by R2; R2's evaluation-only
**sensitivities** ("installation off", "re-pointing off") — properties of the
engine with R2's policy, reported by R2, and the engine is unchanged; three
of R2's four **untrained controls** — they are the initial actors of
replicates that are not under test.

**Hard asserts of the analysis** (`check_invariants`; the script refuses to
report otherwise), for each family: as R2 §10 — the reported arm is
`marl_static`; every arm has exactly the manifest's 40 seeds in both
scenarios; one tree head, tree digest, schema id and hash seed (0) across
the pickles analysed; one `checkpoint_sha256` record across the ten pickles
analysed (the one digest of §3.1 for family 1; the ordered list of four of
§3.1b for family E); `stale_edges` and `stale_u2u_volume` zero; more than
one postcard per site after the event on every MARL-path run; no NaN or
short series (3600 ticks); every bridge record carries a valid lifetime.
Before each analysis the chain requires the key union of its ten pickles to
equal 8 arms × 2 scenarios × 40 seeds, with none missing and none twice.
**`scipy` is a precondition of every analysis step, checked by the chain
and not by the reader**: the analysis takes the t quantile of its
descriptive intervals from it and falls back, without saying so, to a table
that ends at 20 instances. `confirmatory/stage_analysis.sh` therefore
refuses to run the analysis unless the interpreter that will run it
imports `scipy.stats` and returns the t quantile for 39 degrees of freedom
(scipy 1.11.4 on the run host). No report is rejected or repeated on
account of a null `wilcoxon_p_scipy` field: that field is null by
construction wherever a test has no non-zero difference (two stability
endpoints of R2's own report), and the decision test, which is exact and
does not use scipy, is not affected.

## 11. Expectations stated in advance

Written before any pass on the block, from the development data of §3.2 and
§3.3. They are predictions to be compared with the result, not criteria.

1. **P1 and P4 (replicate 1234 against XL-DET):** positive in both
   scenarios, more reliably in B. The development estimates (+10.1, +12.9
   pp) may be inflated by the selection; a shrinkage towards the validation
   estimate in A (+5.4) would not be surprising.
2. **P2, P3, P5, P6 (against SDN and the best routing protocol):** confirmed,
   as in R2 with a weaker policy (development means: replicate 1234 61.3 /
   55.2 percent in A / B, SDN 46.7 / 39.5).
3. **P7–P9 (energy per unit, A):** negative (development: 0.166 J per unit
   against 0.207 for XL-DET and 0.206 for SDN). In particular P8, which was
   reversed in R2, is expected in the hypothesised direction.
4. **Stability: replicate 1234 is expected to tear relay links down more
   often than XL-DET** — 8.2 times as often in Scenario A and 7.2 times in B
   on development (§3.3 item 5), on every instance. S9e and S9f are
   therefore expected to be
   **confirmed in the "MARL-RIC worse" direction**, and by the rule of §9
   **the family is expected NOT to allow the policy's bridging to be called
   stable**. If P1 and P4 are confirmed, the expected overall picture is
   "delivers more, with more relay churn", and it will be reported as that.
5. **Family E, the ensemble.** *Achievability:* expected between XL-DET and
   replicate 1234 (development +6.3 / +9.2 pp against XL-DET), so E-P1 and
   E-P4 are expected positive, E-P4 more reliably; with smaller effects than
   replicate 1234's, a non-confirmed E-P1 with a positive estimate would not
   be surprising (§5.1). E-P2, E-P3, E-P5, E-P6 are expected confirmed and
   E-P7–E-P9 negative (development J per unit 0.179 against 0.207 for
   XL-DET). *Stability:* its relay churn on development equals XL-DET's (M7
   1.43 against 1.49 in A, 1.50 against 1.50 in B), **and for that very
   reason its stability family is expected to assess little or nothing.** On
   the 20 development instances, analysed by the unchanged script on the
   assembled set, all six endpoints were not assessable: non-zero
   differences against XL-DET on 1 instance (E-S9a, M2, A), 0 (E-S9b), 1
   (E-S9c, M4, A), 0 (E-S9d), 4 (E-S9e, M7, A: one higher, three lower) and
   2 (E-S9f, M7, B), against the six required. On forty instances E-S9e may
   become assessable; the others are expected not to. Whether E-S9e then
   has the eight non-zero differences that §9 requires before anything may
   be read from it is open (four of twenty on development). The expected
   statement for the ensemble is therefore "S9 could not assess its
   bridging" in Scenario B, and in A either the same or at best "not shown
   worse than XL-DET by S9" (§9) — **not** that its bridging is stable, and
   the paper will not say so.
6. **Hash invariance:** identical series, as in R2.

## 12. Frozen evaluation environment and launcher settings

As R2 §12. Every invocation runs a job script that re-executes itself under
`env -i` and sources a per-invocation environment file written once by
`confirmatory/chain_lib.sh` (`confirmatory/env_r3lite/<inv>.sh`) and reused
verbatim by a re-run. Before the launch the preflight requires the job's
`MARL_*` set to equal the plan's, and after the run the completeness gate
requires the pickle's recorded `marl_env` to equal it exactly.

| variable | value (profile `r3lite`) |
|---|---|
| `PYTHONHASHSEED` | `0` (`1` on `s6h_hash1` only) |
| `OMP/MKL/OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `PYTHONUNBUFFERED` | `1` |
| `MARL_POOL_WORKERS` | **24** for every evaluation invocation (the shared-machine load rule is not applied to `r3lite`; the memory ceilings of the preflight remain), 1 for tool steps |
| `MARL_TRANSPORT_POWER` | `planning` |
| `MARL_RELAY_STATE` | `reconcile` |
| `MARL_RELAY_REPOINT` | `fragment_aware` |
| `MARL_POSTCARD_ALWAYS` | `1` |
| `MARL_REWARD_PROFILE` | `mission` |
| `MARL_RELAY_ENTROPY_FLOOR` | `0.01` |
| `MARL_STEER_MODEL` | `actionable` |
| `MARL_EVAL_CHECKPOINT` | replicate 1234 (S6, S6h); its untrained initial actor (S8); the four replicates, comma-separated, in the order 1234, 2345, 3456, 4567 (S9x) |
| every other variable of R2 §12 | unset, as in R2; no stress, sensitivity or ablation variable is set by any stage |

That is the environment of R2's stage S6, variable for variable (the plan
test compares the two dictionaries).

Preflight gates before every invocation (`tools/preflight.py`): evaluation
≥ 18 GiB MemAvailable and ≥ 100 GB free on `/home`; tool steps ≥ 4 GiB / 20
GB; no leftover simulator process of ours in the tree; hash seed and thread
pins; the tree digest equal to the one recorded by the run's first preflight,
and after S5 equal to the one sealed in `FREEZE_FINAL_r3lite.txt`. Before
the first stage at or after S6 the launcher also requires the freeze-2 seed
audit of the sealed tree (§13). The hash-seed and thread-pin checks are made
in the launcher's own process, so the launcher runs under an environment
file of its own (`logs/freeze/r3lite_chain.env`: hash seed 0, the thread
pins, no `MARL_*` variable; the lines of R2's `real_chain.env`) and is
registered with the host's watchdog, as R2's was (FREEZE_CHECKLIST R.2.1).

The analysis invocations, as the chain runs them:

```
# S7 (family 1, the primary result), after the md5 pin, --cross-provenance and --key-union
nice -n 19 python3 confirmatory_analysis.py \
  --pickles output/r3lite/s6_cf_b0.pkl,...,output/r3lite/s6_cf_b9.pkl \
  --manifests confirmatory/r3lite_manifest_<block>.json \
  --marl-arm marl_static --out confirmatory/r3lite_cf_report.json
# S7e (family E), after S7: the order read from the ledger and the sha256 of the S7 report
# written to it; then the control, then family E
python3 tools/stage_report.py --ledger-order confirmatory/chain_r3lite.jsonl --first s9x_ensemble \
  --then s7_analysis --expect-then-sha256 <sha256 of the S7 report> \
  --json-out confirmatory/r3lite_family_e_ledger_order.json
python3 tools/assemble_family.py --comparators <the ten S6 pickles> --arm-source <the ten S6 pickles> \
  --arm marl_static --expect-checkpoint-sha256 <the pin of replicate 1234> \
  --expect-keys logs/chain_r3lite/S7e.keys \
  --out-dir output/r3lite/family_ctl --prefix c_ --record confirmatory/r3lite_family_ctl_assembly.json
nice -n 19 python3 confirmatory_analysis.py --pickles output/r3lite/family_ctl/c_s6_cf_b0.pkl,... \
  --manifests confirmatory/r3lite_manifest_<block>.json --marl-arm marl_static \
  --out confirmatory/r3lite_cf_report_family_ctl.json
python3 tools/compare_reports.py confirmatory/r3lite_cf_report.json confirmatory/r3lite_cf_report_family_ctl.json \
  --sections seeds,excluded,descriptive,primary,secondary,stability,S10,S11      # and the reverse
python3 tools/assemble_family.py --comparators <the ten S6 pickles> --arm-source output/r3lite/s9x_ensemble.pkl \
  --arm marl_static --expect-checkpoint-sha256 <the four pins, in order> \
  --expect-keys logs/chain_r3lite/S7e.keys \
  --out-dir output/r3lite/family_e --prefix e_ --record confirmatory/r3lite_family_e_assembly.json
nice -n 19 python3 confirmatory_analysis.py --pickles output/r3lite/family_e/e_s6_cf_b0.pkl,... \
  --manifests confirmatory/r3lite_manifest_<block>.json --marl-arm marl_static \
  --out confirmatory/r3lite_cf_report_family_e.json
# S12 (secondary analyses)
python3 tools/hash_invariance_check.py output/r3lite/s6_cf_b0.pkl output/r3lite/s6h_hash1.pkl \
  --arms marl_static --seeds <the two lowest seeds of the block> --require-decision \
  --quiet --out confirmatory/r3lite_s8_hash.json
nice -n 19 python3 confirmatory_analysis.py \
  --pickles output/r3lite/s6_cf_b0.pkl,...,output/r3lite/s6_cf_b9.pkl \
  --manifests confirmatory/r3lite_manifest_<block>.json --marl-arm marl_static \
  --s8 confirmatory/r3lite_s8_hash.json --out confirmatory/r3lite_cf_report_full.json \
  --untrained untrained_s1234=output/r3lite/s8_untrained_s1234.pkl
python3 tools/compare_reports.py confirmatory/r3lite_cf_report.json confirmatory/r3lite_cf_report_full.json \
  --sections seeds,excluded,descriptive,primary,secondary,stability,S10,S11
python3 tools/stage_report.py --paired-descriptive output/r3lite/s9x_ensemble.pkl \
  --baseline output/r3lite/s6_cf_b0.pkl,...,output/r3lite/s6_cf_b9.pkl \
  --paired-arm marl_static --paired-ref-arm xldet --paired-seeds <the block> \
  --paired-label ensemble --expect-checkpoint-sha256 <the four pins> \
  --json-out confirmatory/r3lite_ensemble_vs_xldet.json
```

## 13. Freeze procedure

- **Before freeze 1.** At the head to be synced, in the run tree:
  (i) the full test suite passes on the server, 0 failed and 0 skipped (it
  includes the engine pin, the analysis md5, the digests of the four R2
  profiles and every R3-lite plan property);
  (ii) **the run-level reproduction**: full-length passes of XL-DET and SDN
  on a development instance and of the policy under test on a validation
  instance and on that development instance are re-run in this tree on
  this host in R2's environment, and every series is identical in three
  comparisons (`tools/series_diff.py --keys all`; FREEZE_CHECKLIST R.1.1):
  two against the pickles the R2 chain wrote (XL-DET and SDN; replicate
  1234 on the validation instance) and one against Phase 0's pass of
  replicate 1234 on the development instance. The
  engine pin shows that the files are R2's; this shows that the host still
  produces R2's numbers from them. A difference stops the freeze;
  (iii) the rehearsal `run_final.sh --profile r3lite_rehearsal` (two
  development seeds, launched as the registered run is launched) has ended
  `CHAIN END profile=r3lite_rehearsal rc=0` at that head, or at a head that
  differs from it in `docs/` only, **and has passed the acceptance check of
  FREEZE_CHECKLIST R.2.4 by count**: 14 of 14 invocations ok on one tree
  digest; the S12 states `identical` (hash invariance), `computed` (the one
  untrained control), `dropped` (secondary S7), `reported(6)` (the stability
  family of family 1, with each endpoint's status), the full report
  `computed` and the primary report reproduced `identical`; **family E
  `computed`**, with its record naming the family-1 report by sha256, the
  assembly control `identical`, nine hypotheses and six stability endpoints
  in its report; the ensemble listing `ok` with 4 instance pairs; every
  evaluation pickle verified with the pinned checkpoint digest recorded (`N
  recorded, N pinned -> MATCH`); **and the ensemble's code path reproduced**:
  the rehearsal's two full-length ensemble instances compared, every series
  identical, with Phase 0's ensemble passes of the same instances
  (`ens_dev.pkl`) — the counterpart of (ii) for the invocation family E
  depends on, and a difference stops the freeze in the same way. The rehearsal is the only step that runs,
  on real pickles, the analysis with one untrained control, the hash check,
  the report reproduction, the assembly of the control set and of the
  family-E set with the analysis of each, the ensemble listing and the
  post-run pickle gate; its numbers are throwaway and are not opened. **Any
  change outside `docs/` after it invalidates it.**
- **Freeze 1 (before S0).** (a) The commit synced to the run tree, with
  `FREEZE_R3_<head>.sha256` = the sha256 of every tracked file of that
  commit (the files of `.synced_files`), written under `logs/freeze/` on the
  server and copied to `records/R3/` of the manuscript repository with the
  UTC time. **The external anchor (D-R3.3 item 4):** before S0 starts, **two
  whole files, the list of redactions that makes the first reproducible,
  and nothing else** are pushed to the author's public
  repository `github.com/efidvir/MARL_code` as a branch and an annotated
  tag: this pre-registration (with internal host names and local paths
  replaced by placeholders, as in the published R2 release, and the list of
  those replacements beside it) and `FREEZE_R3_<head>.sha256`. That is the
  option the author selected, and FREEZE_CHECKLIST R.3 publishes the same
  two files. No extract of another document is published: a third party can
  check a file's digest only against the whole file. The documents this
  text cites — the decision record, the power document, the checklist, the
  code — are not published at freeze 1; the sha256 of each is in the digest
  file, so each can be checked against it when it is released with the
  results. (The redaction rules are those of the published R2 release:
  host names, and absolute paths that carry a user or a machine name.
  Home-relative paths such as `~/MARL_run_9/…` were not replaced there and
  are not replaced here. If no rule matches this text at freeze 1, nothing
  is replaced: the list of redactions says so, says how the file is
  checked, and the published file is the frozen file byte for byte, with
  the sha256 the digest file lists for it. §17 records which was the
  case.) What is
  recorded in §17 is what a third party can check: the commit and tag
  object ids; the sha256 of the published (redacted) text and of the
  unredacted text as listed in the digest file; the redaction list, which
  is published with it, so that the published text can be regenerated from
  the frozen one; and **the time GitHub's servers recorded for the push**
  (the repository's push event from the GitHub API, saved verbatim). That
  server-recorded time is the **third-party timestamp**: an annotated tag's
  own date is declared by whoever creates it and is not evidence. A file
  copied into a repository the author controls does not by itself show a
  reviewer that the text predates the data; the host's record of when it
  arrived does, for as long as the host keeps it, and the author may add an
  independent archive capture of the tag. The push is the lead's action on
  the author's go-ahead (§17); the chain neither pushes nor checks it. (b)
  **The seed audit of the block with zero result-data hits:**
  `MARL_TREE=<run tree> SEED_AUDIT_GATE=r3-freeze1 bash
  tools/run_seed_audit.sh 300 339`, over every `$HOME/MARL*` tree and a
  history dump of all refs captured at that head. The gate: **exit 5 (a seed
  of the block in a pickle, a non-planning JSON file or a checkpoint) or
  exit 4 (coverage refused) stops the freeze**; exit 0 passes; exit 3 (a
  text, file-name or history mention left in the verdict) passes only after
  every such hit has been put, in writing, into one of the classes of
  §13.1 (expected: exit 3, left by the two commit messages of groups 6 and
  7 of §13.1, with the other 19 of its 21 lines listed as matched by the
  five patterns, and no other line). (c) This file, with §17 filled in for
  freeze 1, is committed
  locally; nothing tracked is synced to the run tree after its first
  preflight.
- **Segment 1: `--stages S0,S3b,S5`.** S0 builds the manifest and imports
  the pinned files; S3b verifies the selection record; S5 re-verifies the
  manifest (exactly the block, both scenario kinds), the five imported
  files, the selection record, the analysis md5 and the engine pin, and
  seals `confirmatory/FREEZE_FINAL_r3lite.txt`.
- **Freeze 2 (after S5, before any confirmatory arm runs).**
  `FREEZE_FINAL_r3lite.txt` — the tree digest, **the manifest's sha256 and
  the checkpoint sha256s** — is copied to §17 with the UTC time, and the
  seed audit is re-run with `SEED_AUDIT_GATE=r3-freeze2`. **The only planning
  data it may find beyond freeze 1 is this run's own S0 manifest**
  (`confirmatory/r3lite_manifest_<block>.json`, whose path the driver reads
  from the plan), the chain's own log and ledger lines naming it, and the
  freeze-1 audit's own log and report, which stay at their fixed paths in
  the run tree and are class (b) of §13.1 by their names. Any
  result-data hit stops the run before S6. **The launcher enforces this**
  (`confirmatory/run_final.sh`, `tools/stage_report.py --audit-gate`; the
  plan declares the gate and the builder refuses profile `r3lite` without
  it): no stage at or after S6 starts unless the freeze-2 report exists, is
  the audit of this block made from this tree, on the sealed head, after
  the seal, with the result-data gate requested and passed and coverage
  complete. If that report still lists a text, file-name or history hit
  (exit 3), S6 starts only after the comparison of §13.1 — the lead's, of
  the report with the 21 lines the author confirmed, and for any other line
  the author's own classification — has been recorded in the chain ledger
  by the operator action `run_final.sh
  --profile r3lite --audit-ok`, which is bound to that report by its sha256
  and is refused for a report with a result-data hit or a coverage failure.
  The freeze-2 audit is not re-run once S6 has started (it would then find
  this run's own pickles, as it must).
- **Segment 2: `--from S6`.** S6, S9x, S7, S7e, S6h, S8, S12, unattended.
  The passes of both families (S6, S9x) are complete before either is
  analysed; S7e verifies that order in the ledger before it computes
  anything (§8).
- **The run may stop only on a technical failure**: a preflight gate, a
  missing or incomplete pickle, a provenance or environment mismatch, a
  checkpoint whose digest is not the pinned one, a hard assert of the
  analysis, a refusal of the assembler, an assembly control that does not
  reproduce the family-1 report, a ledger that does not show the registered
  order of the two families, a tree that changed. No stopping rule
  looks at an outcome, and the launcher has no way to: nothing between S6
  and S7 opens a metric, and S7e's only reading of the S7 report is its
  sha256, its structure (the names of the hypotheses and the lists of
  instances, which family E's report must share) and a leaf-by-leaf
  equality with the control report.
- **A failed invocation is repeated once, unchanged** (same job script, same
  environment file, same tree, same seeds and arms). An invocation that
  succeeded is never re-run; a second failure stops the chain and is
  escalated; a partial set is never analysed and no pass is substituted from
  another invocation or tree. **The count is the ledger's, not a launch's**
  (`tools/stage_report.py --restart-state`): an attempt is a start, and an
  attempt that was started and never ended — the launcher or the host died
  while it ran — is an attempt; a re-issued command gives an invocation
  only what is left of its two. The same reading of the ledger decides
  whether the rule is exhausted for an invocation, stops the launcher from
  ever starting it again, releases the two waivers of the item after next
  and makes S12 record family E as not completed, so those four can never
  disagree.
- **What the operator cannot do** (profile `r3lite`; the plan declares it
  and the launcher enforces it before anything starts). `--shed` is refused
  for every stage; `--invalidate`, which would make a finished invocation
  run again, is refused for every invocation of S6 and of every later
  stage, and — once an invocation of S6 has been started — for every
  invocation of the chain, the seal of S5 included; `--waive-dep` is
  refused altogether; `--waive-order` is refused on every edge but the two
  of the next item; `--max-attempts` is refused with any value but two.
  Each refusal is written to the chain ledger, and every `note`, `stop`
  and `invalidate` line of the ledger is reported in §17 — an attempt is
  part of the record of the run.
- **If family E cannot be completed.** S9x and S7e are never shed, skipped
  or repeated for any content of any report. Family E is **not completed**
  when, and only when, the chain ledger records that the one unchanged
  repeat is used up for an invocation of S9x or of S7e — two attempts, the
  last of them ended in failure. That state is **read from the ledger** by
  the launcher and by S12 (`tools/stage_report.py --terminal-failure`);
  nobody declares it, and an invocation in that state is never started
  again.
  Then:
  1. **Family 1 is analysed and reported alone, with every one of its
     secondary analyses.** After a failed S9x: `run_final.sh --profile
     r3lite --stages S7,S6h,S8,S12 --waive-order S7,S12`. After a failed
     S7e: `--stages S6h,S8,S12 --waive-order S12`. The stage graph is built
     for this: S6h, S8 and S12 depend on family 1 only. The two order
     waivers (S7 after S9x, S12 after S7e) are the only ones the plan
     declares; the launcher honours them **only in that state of the
     ledger**, records them with the ledger's cause, and refuses them in
     every other — in particular while S9x has simply not been run, which
     is the state in which family 1 could be analysed and read before
     anyone decides about family E.
  2. **S12 records family E as `not_completed`, with the cause, and
     computes no per-instance listing of the ensemble.** No
     ensemble-against-XL-DET figure of any kind is reported from this run:
     the listing would otherwise be the only one, outside the analysis
     registered for it.
  3. **The paper reports family E as "not completed", with the cause**
     (§14), and what exists of it is deposited: the ensemble pickle, as far
     as it was written (S12 names it with its sha256 and never opens it),
     the assembler, the pinned analysis command and the failing log are
     released with the results, and the paper says that family E can be
     computed from them with the pinned command by anyone.
  4. **Family E is not re-run in another form.** S9x is not repeated, for
     this manuscript, by anyone. If it was S7e that failed, for a reason
     that lies in the tooling and not in the passes, the same command may
     later be completed on the same pickles; that is reported as a
     completion after a recorded failure, with the failure and its cause,
     under every rule of §14 for family E — never as if the chain had
     produced it, and never without the family-1 outcome beside it.
  Family 1's result is unaffected by all of this: it never reads an
  ensemble pass.
- **Everything is reported whatever the outcome**: all nine primary
  hypotheses and the six stability endpoints of each family, every
  secondary analysis of §10, every excluded or repeated invocation with its
  cause, and the comparison with the expectations of §11. If the run is
  stopped for a technical reason after S6 has started, that is reported
  with what had been computed.
- Nothing else in this file changes after freeze 1. Any analysis not listed
  here is labelled post hoc wherever it appears.

### 13.1 Classes of a seed-audit hit, fixed before the audit is run

Every hit of the freeze-1 and freeze-2 audits belongs to exactly one class.
The classes and the hits already known are written down here so that
"fresh" is not decided hit by hit after the fact.

| class | what it is | consequence |
|---|---|---|
| **(a) result data** | a seed of the block in a pickle, a non-planning JSON file or a checkpoint | exit 5; the freeze (or the run, at freeze 2) stops. No marker, pattern or acknowledgement can reclassify it |
| **(b) planning reference** | the plan (`confirmatory/chain_plan.py`), the tests that pin it, the planning documents, the audit's own reports and logs; at freeze 2 also this run's own S0 manifest and the chain's log and ledger lines that name it; and the messages of the commits that wrote the block into the plan and into this text (groups 6 and 7 below) | classified in the driver by explicit path markers, recorded in the report (`planning_markers`, `planning_hits`); does not count. A commit message has no path: it cannot be classified by a marker, stays in the verdict (exit 3) and is dealt with as the end of this section says |
| **(c) coincidence** | the integer is a tick count or a loop bound and not an instance: the line either names another seed explicitly or names no seed, and the number in the window is a duration or a range limit | classified per file and line with the reason; does not count once classified |
| **(d) negative statement** | the line states that the block was **not** read or used | classified per file and line; does not count once classified |
| **(e) anything else** | — | an exposure: the freeze stops and it is reported plainly |

**Who classifies, and when.** As first written, this paragraph read:
classes (c) and (d) are assigned **by the author, in writing, before the
first S0 step**: one line per hit (path, line, class, reason), with the
report's sha256, the author's name and the UTC time, copied into §17;
whoever runs the audit does not classify; at freeze 2 the same applies
before S6, and the launcher does not start S6 on a report with hits left
until that classification has been recorded (`--audit-ok`, above). **That
is what holds for any line outside the 21 lines of the two tables below.
For those 21 lines it was replaced before freeze 1, and the end of this
section says by what and why it is less than this.**

**Hits known before freeze 1.** A pre-audit of the first draft's block
(300–319) was run on 2026-10-11 09:35–09:51 UTC over all sixteen
`$HOME/MARL*` roots at head `f7f4ac9` (the freeze-1 options, without the
history dump; report sha256 `5d9c6dcc42648fd1…`, kept outside every scanned
tree and not freeze evidence). Result: **result-data gate PASS, no
file-name hit, and 15 text lines in five groups, none of which is a use of
the block.** **The block is now 300–339 (D-R3.3)**, and the integers
320–339 were not covered by that pre-audit, so it was repeated for the
whole block on 2026-10-12 07:15–07:35 UTC at head `ae4b909` (seventeen
roots, the run tree `~/MARL_r3lite` among them; 15 671 files, 839 pickles,
2 971 checkpoints; the freeze-1 options without the history dump, so its
own verdict is "coverage refused" by design; report sha256
`f177be281dc34352…`, kept outside every scanned tree and not freeze
evidence). Result: **no seed of 300–339 in any pickle, JSON result file or
checkpoint; no file-name hit; the seeds 320–339 added no line at all**; the
five patterns matched 17 lines — the 15 of the first pre-audit and the same
two lines of `tests/test_d12_baselines.py` in the run tree's copy; and one
unclassified text line, in a log of a failed test run of the lead in the
run tree itself (`logs/amend/t_plan.log`, a traceback quoting the first
draft's manifest name), which was moved out of every scanned tree the same
day. The five groups are classified now.
The driver passes one line pattern per group to the tool
(`--false-positive`, under the R3 gates only; `tools/run_seed_audit.sh`
reads exactly these five and refuses to run with another count); a line
that matches leaves the verdict and is **listed, with its path and line
number, in the log and in the report** (`false_positive_hits`,
`false_positive_patterns`).

| # | class | pattern | files and lines on 2026-10-11 | why it is not a use of the block |
|---|---|---|---|---|
| 1 | (c) | `[CROSS] + INSIDE)), range(0, 400))` | `tests/test_d12_baselines.py` lines 176 and 180, in `~/MARL_r3`, in `~/MARL_run_9` and in the run tree `~/MARL_r3lite` (6 lines); with the history dump, the same two lines once more as lines of the dump (2 lines) | `range(0, 400)` is the tick range of a synthetic routing test; the test builds no topology from a seed |
| 2 | (c) | `(seed 82, 300 ticks, the arm that writes and reloads` | `output/ref/README.md` line 103 and its four backup copies (lines 95, 97, 99, 101) in `~/MARL_run_9`; `output/ref/README.md` line 95 in `~/MARL_run_9b` (6 lines) | a smoke run on seed 82 that lasted 300 ticks |
| 3 | (c) | `marl_freeze smoke, seed 82, 300 ticks` | `output/ref/invariance_WP0_3712e7d.txt` line 35 in `~/MARL_run_9` and in `~/MARL_run_9b` (2 lines) | the same smoke run |
| 4 | (c) | `0c DONE (team term in the actor advantage)` | `r3out/phase0/PROGRESS.md` line 61 in `~/MARL_r3` (1 line) | the line ends "one real training-worker episode, ck4567 actor, seed 50, 300 ticks": a duration on a development seed |
| 5 | (d) | `Nothing for seeds 200-219, 300-359 or 400-439 was read` | `r3out/phase0/PHASE0_REPORT.md` line 7 and `PHASE0_REPORT_v1.md` line 6 in `~/MARL_r3` (2 lines) | it says the block was not read |

Rules for these five: (i) at each freeze the report's
`false_positive_hits` are compared with this table (by whom, and on what
authority, is said at the end of this section); a matching line in a file the table
does not name is accepted only if it is the same line (a copy or a backup of
a listed file, or the same line in the history dump) and is added to the
written classification, otherwise it is class (e); (ii) a pattern classifies
a text line only — a pickle, a JSON file or a checkpoint is class (a)
whatever any line says; (iii) **no pattern is added after freeze 1**: a text
hit that is not in this table is classified in writing as (c) or (d) by the
author, with its reason, or it is class (e). The line numbers are those of
2026-10-11 and may move; the classification follows the line's content.
The first two pre-audits ran without the history dump; the third, with it,
found the two lines of groups 6 and 7.

**The pre-audit with the history dump, and the two lines it adds.** The
pre-audit was repeated with a history dump of all refs on 2026-10-12
10:46–11:04 UTC at head `7830541` (the freeze-1 options; the dump and every
output outside the scanned trees; seventeen roots, 15 724 files, 839
pickles, 2 976 checkpoints, 285 commits of history; report sha256
`0ba858c8b066cac9…`; not freeze evidence). Result-data gate PASS, coverage
complete, no file-name hit. (The report also lists, outside its verdict,
two file names of `~/MARL_run_5` that carry the number 300 as a parameter
value and not as a seed — `output/hystval_d300_s1234.pkl` and
`logs/hystval_d300.log`; the tool marks them as not seed-like, and the
content of the pickle is covered by the result-data gate. They are in
every pre-audit report and are expected in both freeze reports.) The five
patterns matched 19 lines: the 17 of
the table and the two lines of `tests/test_d12_baselines.py` once more, as
lines of the history dump. **Two history lines were left in the verdict
(exit 3).** Both are commit messages of this repository — of the two
commits that wrote the enlarged block into the plan and into this text:

| # | class | line | where | why it is not a use of the block |
|---|---|---|---|---|
| 6 | (b) | `The confirmatory block is seeds 300-339 (40 instances; the reserve is 340-359` | the message of commit `f6b8611` (the code of the amendment, D-R3.3) | it says what the plan's block is; the commit changes the plan and runs nothing |
| 7 | (b) | `itself; forty instances, seeds 300-339; the count of every confirmatory look` | the message of commit `ae4b909` (the documents of the amendment) | it summarises this text |

A commit message has no path, so the driver's path markers cannot classify
it as the planning reference it is. No pattern is added for these two
lines: the driver reads exactly five, and it is not changed after the
rehearsal. They therefore stay in the verdict of both freeze audits, whose
exit status will be 3, and are dealt with as the paragraphs below say.

**The author's confirmation, given in advance for both freezes
(2026-10-12 11:46:30 UTC).** The lead put the 21 lines to the author in
these words: "The seed audit will list 21 text lines that contain a number
between 300 and 339. None uses the fresh seeds: 17 are tick counts or a
loop bound in old notes and one test ("seed 82, 300 ticks", "range(0,
400)") plus the sentence "Nothing for seeds … 300-359 … was read"; 2 are
that same test in the code history; 2 are our own commit messages naming
the block ("forty instances, seeds 300-339"). No result file, pickle or
checkpoint contains these seeds. The procedure says you, not I, confirm
this. Confirm for both audits (before sealing and before the run)?" He
selected: "Confirm (Recommended) — These 21 lines are not uses of seeds
300–339. Applies to both audits as long as they list exactly these lines;
if anything else shows up, I stop and ask you." The other option offered
was to be shown the lines first.

**What that means for the procedure of this section, stated as it is
done.** The author confirmed, on the lead's description and without being
shown the lines themselves (he chose that option), that these 21 lines are
not uses of the block. The assignment of the seven groups to the classes
(b), (c) and (d) and the reasons in the two tables are the lead's wording.
The author does not himself compare the freeze reports with the tables. At
each freeze the lead compares the report's `false_positive_hits` and every
text and history hit left in its verdict, line by line, with the 21 lines
of groups 1–7 (group 1 counted with its two lines in the history dump; a
line number may move; the content of the line and the file or commit it
belongs to may not), and copies that comparison into §17 with the report's
sha256. The freeze goes on only if the report lists exactly those lines.
Any other line — a new text or history hit, a file-name hit, or a known
line in a file or commit the tables do not name, a new copy of a listed
file included (rule (i) above would have accepted such a copy on a written
classification; the confirmation covers exactly these lines, so it is put
to the author) — stops the freeze until the author has classified it
himself, in writing,
as the paragraph "Who classifies, and when" first required; and a
result-data hit stops it whatever anyone says. At freeze 2 the same holds,
and the acknowledgement the launcher requires for a report with hits left
(`--audit-ok`) is recorded on that comparison.

**This is less than the procedure first written in this section asked of
the author.** There he was to write one line per hit himself, at each
freeze, against the report's sha256. For the 21 lines it is replaced by
his single confirmation of 11:46:30 UTC, given before either freeze audit
had run, and by the lead's comparison. The lead, who runs the audit, is
therefore also the one who finds that a report "lists exactly those
lines". What keeps that from being a matter of trust: the comparison is of
content, line against line, and can be repeated by anyone from the two
audit reports, which are kept with their sha256 in §17 and are released
with the results (§17, "After the run"; under the redaction rules of §13,
since they carry paths of the compute host), and from the two tables of
this section.

## 14. What will be claimed, for each outcome

The claims below are the strongest the tests can support; none may be
exceeded. Three phrases have a fixed meaning in every row.

- **"The selected learned policy"** (family 1) means: replicate 1234 of R2's
  four training replicates, chosen by the paired-achievability rule without
  a churn guard, on the development and validation instances, after their
  results were seen (§3.3). **"The ensemble"** (family E) means: the
  logit-averaged ensemble of the four replicates (§3.1b), the pick of the
  guarded rule at g = 2 (g was not fixed; §3.3 item 7), registered as a
  second family after the same results were
  seen. Every row below is written for family 1; **the same row applies to
  family E with "the ensemble" for "the selected learned policy" and
  E-P*n* for P*n***, and the rules of the paragraph after the table govern
  how the two are combined.
- **"XL-DET"** means, and each row that names it carries this caveat (C):
  **XL-DET as implemented at `6b53f3f`, a fixed engineered reference which
  Phase 0 showed to be improvable by a static power setting** (a per-site
  plan computed in hindsight raises XL-DET's own achievability by +14.1 pp
  in Scenario A and +18.5 pp in B on the development instances; a rule with
  one fixed power per site type was not simulated; §1.2). **The result is
  not evidence that learning, rather than a power setting, is responsible,
  nor that the policy beats a tuned engineered controller.**
- **"Counted"** (N) means: the sentence, or the one next to it, states the
  count of §1.3 in its one numbering — that "a learned policy delivers more
  than XL-DET" has been put to a confirmatory test four times in three
  studies; that this study holds the third and fourth looks, one per
  policy, each in its own family; that the second, R2 on the corrected
  simulator, did not confirm it for the policy R2's rule selected; and that
  the first, on the uncorrected simulator, was superseded as confirmatory
  evidence and is reported as history.

| outcome on the fresh block, per family | what the paper may say |
|---|---|
| **P1 and P4 both confirmed** | (N) The selected learned policy delivers a larger share of the offered traffic than XL-DET (C) on fresh topologies, in both scenarios, by the reported paired means and intervals. |
| P4 confirmed, P1 not | The same statement, with (N) and (C), for Scenario B only. For Scenario A: not confirmed; with a positive estimate, "inconclusive" (§5.1), never "equal". |
| P1 confirmed, P4 not | The same statement, with (N) and (C), for Scenario A only; Scenario B not confirmed, and with a positive estimate "inconclusive", never "equal". |
| neither confirmed | (N) No advantage over XL-DET (C) was demonstrated for the selected policy on fresh instances. The advantage seen on the selection instances was not reproduced at this sample size; the power of the test at smaller effects is given with it (§5.1), so a positive estimate is reported as inconclusive, never as "equal". **R2's result for replicate 4567 is unchanged and is neither strengthened nor weakened**: a null result for one policy says nothing about another. The paper claims no advantage over XL-DET for that policy. |
| P1 or P4 significant in the opposite direction | XL-DET (C) delivers more than the selected learned policy in that scenario; reported as that. |
| P2, P3, P5, P6 | Per hypothesis, as confirmed or not — or reversed (Holm-adjusted p < 0.05 the other way), reported as that: more than SDN / than the best routing protocol in that scenario. R2's outcome (replicate 4567, R2's block) and this outcome (replicate 1234, this block) are reported side by side. **No inference about a difference between the two policies is made**: they are not compared on the same instances, and "confirmed in one study and not in the other" is not a test of a difference. |
| P7, P8, P9 | Per hypothesis: lower energy per delivered unit than XL-DET (C), with (N) / than SDN / than the best routing protocol in Scenario A, or not confirmed, or reversed. |
| any of S9a–S9f confirmed in the "worse" direction (expected for replicate 1234: S9e and S9f) | The policy is worse than XL-DET on that endpoint, named — for M7: it tears relay links down more often than XL-DET, with the factor; its bridging is **not** called stable. Together with a confirmed P1/P4: "delivers more, with more relay churn" — both halves in the same sentence wherever the first appears in the abstract or conclusion. |
| an S9 endpoint confirmed in the opposite direction | "MARL-RIC better on that endpoint" (§9), the endpoint named. Nothing about the stability of the policy's bridging in general is concluded from it. |
| no S9 endpoint confirmed worse, and an M2 or M7 endpoint of the scenario has at least 8 non-zero differences | Its bridging was "not shown worse than XL-DET by S9" in that scenario (§9), with the number of non-zero differences of every M2 and M7 endpoint — not "stable" and not "as stable as XL-DET"; reunification may be called monotone only if M2 = 0 on every instance. |
| no M2 or M7 endpoint of a scenario has 8 non-zero differences (expected for the ensemble, §11 item 5) | "S9 could not assess the policy's bridging in that scenario", with the counts of non-zero differences. Nothing about stability is claimed for it, in either direction. |
| hash invariance not identical | Reported as a reproducibility limitation with the differing series. |

**How the two families are combined** (the rules of §8 and §10, applied to
the hypotheses against XL-DET; the same holds for every other hypothesis):

| family 1 (replicate 1234) | family E (the ensemble) | what the paper may say |
|---|---|---|
| confirmed | confirmed | Each statement for its own policy, family 1 first. A sentence about **the two policies tested here** together — replicate 1234 and the four-replicate ensemble, named — may be written for that hypothesis, with (N) and (C). It does not say "learned policies" or "the training method" without the spread of the four replicates of §3.2 beside it ("not claimed" item 1): the two tested policies are not independent, since the ensemble contains replicate 1234, and the other three replicates did not exceed XL-DET on the selection data. |
| confirmed | not confirmed | The statement for replicate 1234, with (N) and (C). In the same sentence or the next: the ensemble's hypothesis was not confirmed (with its estimate and interval; for E-P1 and E-P4, "inconclusive" if positive — point 3 below). A sentence about the two policies together only if family 1's Holm-adjusted p is below 0.025, and then with both outcomes named. |
| not confirmed | confirmed | **The primary family did not confirm the hypothesis** — stated first, in the same sentence. Then: the pre-registered second family confirmed it for the ensemble, with (N) and (C). The ensemble is not described as the study's main finding, and no sentence reports it without the failed family-1 hypothesis beside it. A sentence about the two policies together only if family E's Holm-adjusted p is below 0.025, and then still with the family-1 outcome beside it. |
| not confirmed | not confirmed | No advantage over XL-DET (C) was demonstrated for either policy, with (N). |
| reversed (Holm-adjusted p < 0.05 in the opposite direction) | any outcome | Family 1's statement first: XL-DET (C) delivers more than replicate 1234 in that scenario. Then family E's outcome as its own row says, with the family-1 reversal in the same sentence. No sentence about the two policies together. |
| confirmed or not confirmed | reversed | Family 1's outcome first, as its row says; then: XL-DET (C) delivers more than the ensemble in that scenario. No sentence about the two policies together. |
| any outcome | not completed (§13) | Family 1 as its row says; family E "not completed", with the cause, and what was deposited. No ensemble-against-XL-DET figure from this run is reported (the listing of §10 is not computed), and no sentence about the two policies together is written: one of them was not tested. |
| not completed (S6 or S7 stopped for good) | — | Nothing is claimed for either policy: family E's comparators are family 1's passes, so neither family can be analysed. The run is reported as stopped, with the cause and with what had been computed (§13). |

No row allows a comparison of the two policies with each other: neither
"the ensemble is more stable than replicate 1234" nor "replicate 1234
delivers more than the ensemble" is tested here. Their per-instance values
may be shown side by side, labelled descriptive.

**Five points the tables leave open, settled now** (found while the
manuscript text that does not depend on the results was being prepared,
before freeze 1; each makes a rule above applicable and none widens what
may be claimed, except the first, which is a concession to the length of
an abstract and is marked as one):

1. **Short forms of (N) and (C), for the abstract only.** The abstract has
   250 words, and the full forms of (N) and (C) take more than half of
   them. In the abstract, and nowhere else, they may be given in these
   words, fixed now; the main text and the conclusion carry the full forms.
   (N): *"This study holds the third and fourth, one per policy, of four
   confirmatory looks, in three studies, at whether a learned policy
   delivers more than XL-DET; the second study did not confirm this for
   the policy its rule selected."* (C): *"XL-DET is a fixed engineered
   reference whose achievability a per-site power plan computed in
   hindsight raises on the development instances (a rule with one fixed
   power per site type was not simulated); the result does not show that
   learning, rather than a power setting, is responsible, nor that the
   policy beats a tuned engineered controller."* The short (N) is the same
   whichever family the sentence is about. Of the full form it leaves out
   that each look of this study is in its own family, that the second
   study is R2 on the corrected simulator, and that the first look, on the
   uncorrected simulator, was superseded and is reported as history. The
   short (C) leaves out the code version, the name of the examination, the
   two figures, and the general statement that a static power setting
   improves XL-DET, of which it keeps the one case that was simulated.
   Nothing
   else of §14 is shortened for the abstract: both families, family 1
   first, and the relay-churn half of any sentence that has one.
2. **Stability, one scenario worse and the other not.** "Not shown worse
   than XL-DET by S9" is available for a policy only when none of its six
   endpoints is confirmed worse (§9). When an endpoint is confirmed worse
   in one scenario, the endpoints of the other scenario are reported as
   they are, each with its status and its number of non-zero differences,
   and the phrase "not shown worse" is not used for that scenario. The two
   other rows of the table still apply to it: an endpoint confirmed in the
   opposite direction is "MARL-RIC better on that endpoint", and if none of
   its M2 and M7 endpoints has 8 non-zero differences the paper writes that
   S9 could not assess the policy's bridging in that scenario (§9).
3. **A hypothesis against XL-DET that is neither confirmed nor reversed.**
   For P1 and P4 and their family-E counterparts with a positive estimate,
   the rows above hold: "not confirmed", and "inconclusive" (§5.1). With an
   estimate that is zero or negative, and for P7 and E-P7 whatever the sign
   of the estimate, it is reported as "not confirmed", with the estimate
   and its interval, and the paper says that no advantage was shown.
   "Inconclusive" is used for no other hypothesis: each of those is
   "confirmed", "not confirmed" or "reversed". "Equal" and "worse" are not
   used.
4. **"The factor" for M7** is the ratio of the two arms' mean M7 over the
   instances of the scenario, as in §3.3 item 5 (12.18 against 1.49 is
   8.2). Medians may be printed beside it, labelled as medians.
5. **Names.** The paper uses its own names for what this text calls by
   internal ones — "the released code" for the commit id, "the examination
   after the second test" for Phase 0, and its own terms for the two
   families, provided those terms keep family 1 as the primary family and
   family E as the pre-registered second family, as §10 and the table
   above require. What is fixed here is the content of (N), (C) and of each
   row, not those labels.

What may **not** be claimed under any outcome:

1. **That the training method produces a policy that beats XL-DET.** One of
   four replicates was selected; the others ranged from clearly worse than
   XL-DET to not distinguishable from it on the selection data. The supported statement is about the selected
   policy and the stated selection procedure. Any sentence about the method
   must give the spread of the four replicates beside it.
2. **That R2 was wrong or is superseded.** R2 tested the replicate its
   pre-registered rule chose and did not distinguish it from XL-DET (no
   hypothesis against XL-DET confirmed; P1 interval [−4.28, +3.58] pp, P4
   [−1.30, +8.09] pp; no equivalence was tested). Both studies are reported,
   each with its own policy, selection rule, block and Holm family; their
   p-values are not pooled and neither family is re-corrected for the other.
   Because the two families are not corrected for each other, the number of
   attempts is disclosed instead (item 6).
3. **The ensemble as a replacement for, or a rescue of, the primary
   family.** Family E is a pre-registered family and its confirmed
   hypotheses may be reported as confirmed, under the rules above. It may
   not be promoted over family 1 after the fact, reported without family 1,
   called the study's main finding, or used to restate a failed P1 or P4 as
   a success. No inferential statement about the ensemble may be made from
   anything but the family-E report (the per-instance listing of §10
   decides nothing). No other candidate — replicate 2345, 3456 or 4567 — is
   run on the block as an arm of its own.
4. **Anything about instances, scenarios or comparators outside those
   tested**, or about the mechanism by which replicate 1234 does better. In
   particular, not: that the policy beats the engineered state of the art,
   a tuned XL-DET, or any controller with a static power plan; nor that
   learning, rather than a power setting, is responsible (§1.2).
5. **A stability claim the families of §9 do not allow.** In particular:
   that the ensemble's bridging is stable, or as stable as XL-DET's — its
   stability family is expected to assess little (§11 item 5), and "not
   shown worse" or "could not be assessed" is all §9 allows; and that "a
   learned policy with acceptable relay churn delivers more than XL-DET" as
   one claim: a confirmed E-P1 or E-P4 is a statement about delivery, and
   the ensemble's churn is reported beside it as measured, with what S9
   could and could not assess.
6. **A further attempt in this manuscript, or an uncounted one.**
   (a) R3-lite is the third study of "a learned policy delivers more than
   XL-DET" and holds the third and fourth confirmatory looks at it, one per
   family (§1.3, one numbering): the second look, R2 on the corrected
   simulator, did not confirm it for the policy R2's rule selected, and the
   first, on the uncorrected simulator, was superseded as confirmatory
   evidence and is reported as history. (b) R3-lite is **the last confirmatory attempt at this
   question in this manuscript, for both families jointly** — the author's
   decision of 2026-10-11 (D-R3.3 item 3). Any later test of it — another
   replicate, another ensemble, a retrained policy — is a new
   pre-registered study, counted in its own text and in any paper that
   reports it. (c) The abstract and the conclusion state the count (N)
   wherever a confirmed hypothesis against XL-DET is quoted. (d) The blocks
   held in reserve (340–359, 400–439) are not used to rescue a failed
   hypothesis by choosing another arm or by adding instances.
7. **That the paired-achievability rule, or any selection rule, outperforms
   R2's rule.** Replicate 4567, R2's pick, is not run on the fresh block, so
   the two rules are never compared on fresh instances, and the amounts by
   which replicate 1234 exceeds replicate 4567 on development (+10.3 and
   +12.9 pp) are selection data. The paper may say that the two rules
   disagree on the selection data, and no more.

## 15. How the results enter the manuscript

- Version 9 of the manuscript is held until this run has ended and §17 is
  complete.
- Every inferential statement about replicate 1234 comes from family 1
  (§8–§9) and every inferential statement about the ensemble from family E,
  both on seeds 300–339. Every inferential statement about R2's policy
  comes from R2 on seeds 200–219 and is unchanged.
- The paper states the history in order: the first test (seeds 80–99, the
  uncorrected simulator; superseded as confirmatory evidence, reported with
  its verdict); R2 (the corrected
  simulator, the pre-registered rule, five of nine confirmed, the learned
  policy not distinguishable from XL-DET, with the intervals); the Phase 0
  diagnosis (the variance between replicates; a disagreement between the
  two selection rules on the selection data; the guarded rule Phase 0
  recommended and the arm it selects at g = 2 (g was not fixed); the
  finding that XL-DET is improvable
  by a static power setting); and R3-lite, with its two families and the
  count of §1.3 — with the disclosures of §3.3 and the design history of
  §1.4 in the main text, not in a footnote. The disclosures of §3.3 are
  its facts about the selection: what was chosen, when, on which data, what
  the rules were and which arm each selects, and what was tuned. The
  design history of §1.4 is, for the paper, what was changed, when, and
  what had been seen when it was changed. Whose recommendation and whose
  reasoning the design is (§1.4, §3.3 item 7) is recorded in this text and
  is not among the sentences this section requires of the paper; how the
  paper acknowledges the lead's part in the work is the author's to word,
  within the journal's rules.
- Wherever a confirmed hypothesis against XL-DET is quoted — abstract and
  conclusion included — the same sentence or the next carries the count (N)
  and caveat (C) of §14 (in the abstract, in the short forms of §14,
  point 1).
- The two families appear as §10 and §14 fix: both in every outcome, in
  one table with the same rows, family 1 first; family E never without the
  outcome of the same hypothesis in family 1.
- An unconfirmed hypothesis is reported as unconfirmed and its claim is not
  made. The expectations of §11 are reported next to the results, including
  the ones that failed.
- The development and validation numbers of §3.2 are reported as the
  selection data, labelled as such, never as evidence.

## 16. What R3-lite does not run, and where its text differs from R2's

### 16.1 Chain stages

Recorded in the plan (`chain_plan.DROPPED_R3LITE`), written to the chain
ledger at every start and into the secondary report (`dropped_by_plan`).

| R2 chain stage | in R3-lite | reason |
|---|---|---|
| S1 development + stress, policy-independent arms | not run | the engine is byte-identical to R2's; the evidence is R2's S1 and Phase 0, cited (§4) |
| S2 training | not run | the policy is an existing R2 replicate, imported byte-identical |
| S3 validation passes | not run | cited from R2's S3 and Phase 0 (§3.2) |
| S3b selection | **reduced** to a record | the selection was made before this text was written (§3); S3b verifies the record and the digest |
| S4 development + stress `marl_static`, end-state dump | not run | cited from Phase 0 (§3.2) |
| S5 development gate + FREEZE_FINAL | **reduced** to FREEZE_FINAL | there are no development pickles to gate; the seal covers the tree, the manifest and the checkpoints |
| S5b analysis dry run on the development set | not run | the analysis script is byte-identical to the one that ran on R2's sets (§8); the rehearsal runs it end to end |
| S8 untrained control, four actors | one actor (1234) | the other three belong to replicates not under test |
| S9 adapt-then-freeze variant (secondary S7) | not run | not part of the R3-lite question; reported by R2 |
| S10, S11 sensitivities | not run | engine properties reported by R2 with R2's policy; the engine is unchanged |
| — | **new: S9x** the ensemble's passes (chain S9x) | the confirmatory data of family E, run right after S6 and before any analysis (§6) |
| — | **new: S7e** the analysis of family E | the same analysis on the assembled set, after S7 (§8) |

### 16.2 Checks of R2's freeze

R2's freeze 1 had three checks besides the suite and the rehearsal
(`docs/FREEZE_CHECKLIST.md`, R2 steps b.2–b.4). None is run as R2 ran it:

| R2 freeze check | in R3-lite | reason |
|---|---|---|
| b.2 invariance pass against the stored reference pickles (all eight arms, 600 and 1200 ticks) | **replaced** by the run-level reproduction of §13 | those references served the development of the engine, each engine change with its own reference. Here no engine file changes (62 files pinned by blob id), so the open question is not "did a change alter an arm" but "does this host still produce R2's numbers", and that is answered more directly by re-running full-length passes of three arms on two instances and requiring every series identical in three comparisons — two against the pickles the R2 chain wrote, one against Phase 0's pass of replicate 1234 (§13 (ii)) |
| b.3 analysis-input contract on the invariance pickle | not run | it checked that pickle against the analysis's input contract; the analysis script is pinned by md5 and the rehearsal runs it end to end on real pickles |
| b.4 determinism gate | not run | it establishes that training gives one policy per torch seed; nothing is trained here, and the weights are pinned by sha256 |

What the pin and the reproduction do not cover is stated: the reproduction
is three arms on two instances (twelve passes), not all eight arms; a host change that
altered only an arm it does not run (the four routing protocols, the random
control) would not be seen by it. Those arms share the engine with the three
that are run, and the completeness gate compares every pickle's recorded
tree digest and environment with the plan's.

### 16.3 Where this text differs from R2's

Sections 7–10 say "as in R2". These are the places where the words are not
R2's:

| where | R2's text | here | why |
|---|---|---|---|
| §9, opposite-sign outcome | reported as "MARL-RIC worse" | reported as "MARL-RIC better" | R2's label was a slip: the expected sign is the "worse" one. The code has no label (`direction_ok`); unchanged |
| §9 and §14, no endpoint shows it worse | "bridging may be called stable" | "not shown worse than XL-DET by S9" | no equivalence margin was specified, so non-significance is not a demonstration of stability. Decision rule and code unchanged; the allowed sentence is narrower |
| chain stage of the ensemble | — (R2 had none) | S9x | an earlier draft of this plan named it S9e, which is the id of a stability endpoint |
| §8–§9, a second family | — (R2 had one learned arm) | hypotheses E-P1–E-P9 and endpoints E-S9a–E-S9f for the ensemble, each family Holm-corrected within itself | D-R3.3. The analysis file is unchanged; in the family-E report the keys are the script's own (P1…, S9a…) and the prefix E- is this text's and the paper's |
| §9, a scenario with no M2 or M7 endpoint of at least 8 non-zero differences | not addressed | "S9 could not assess the policy's bridging in that scenario" | "not shown worse" would be vacuous where no M2 or M7 endpoint of the scenario could, on its own, have reached significance (§9: eight non-zero differences, not the six of "assessable") |
| §1.1 and §14, R2's result against XL-DET | — | "not distinguishable from XL-DET", with the intervals | R2 tested no equivalence; a word that means equality is not used |
| §8, §9, §10: three passages of R2 | present in R2 | restored here by reference | P2/P5/P8 read with the SDN disclosures (R2 §8); XL-DET instances that settle away from the optimum are reported, not removed (R2 §9); the rule for "responds to the surge" (R2 §10, S2) |

## 17. Records appended at the freezes and after the run

*(nothing above this section changes after freeze 1)*

**The author's go-ahead, entered with the freeze-1 commit: what he was
sent, what he was asked, what he wrote** (2026-10-12, times UTC, from the
record of the working session).

- 09:45:30 — the lead sent him this file as it stood at commit `7830541`
  (§1–§16 as they are here, except for the passages listed below).
- 09:45:44 — the lead asked him for one typed line, in his own words,
  approving the five decisions of D-R3.3 and that text, named the four
  places where the text had changed that day (the rule of eight non-zero
  differences in §9; "the two policies tested here" in §8 and §14; what
  the operator cannot do, and what is reported and deposited if family E
  is not completed, in §13), and offered him the chance to add a reason of
  his own for testing both policies.
- 10:36:07 — Efi Dvir, the author: *"What do you mean? This is something we
  did together. The logic to running this came from you."*
- 10:37:14 — the lead replied that this was right, that it was not asking
  him for reasons, and that it needed only a plain "go" for two things it
  would not do on its own account: publishing this text under his name on
  his public repository, with the commitment that this is the last
  confirmatory attempt in the manuscript, and using the block, which can
  be used once. It said that the record would state whose reasoning the
  design is, with his sentence, and that the published text says that each
  option he had chosen was the recommended one.
- 10:38:28 — the author: *"Go ahead as planned. Do it as fast as you can.
  Utilizing the server that you have at your disposal. You can use all
  cores. Run everything as fast as possible, then optimized and just
  deliver what I've asked."*

**What this is, and what it is not.** It is his go-ahead for the two
things the lead had asked it for — the publication under his name, with
its commitment, of the text he had been sent (the published file adds
what is listed below, which he has not been shown), and the use of the
block — and so for freeze 1, the external anchor and the run as planned.
It is one line, and the lead does not present it as a reading of this
text section by section. He asked for speed and said that all cores could
be used: the registered environment (§12, 24 workers) was not changed, and
nothing registered here was changed to go faster.

**What this commit adds to the text he had been sent, and what of it he
saw.** He was shown none of the wording listed here. What he saw are the
questions he answered and the lead's reply of 10:37:14. At 12:45:01 the
lead wrote to him, in a status message of the working session, that his
sentence of 10:36:07 is quoted in this published text, asking him to say
before the push if he did not want that, and that the audit reports list
the two file names of §13.1, which he was not asked to confirm. No answer
had come when this text was frozen.

- *Whose reasoning the design is.* §3.3 item 7, which like this entry
  carries his sentence of 10:36:07 in this published text (he had been
  told that the record would carry it); and the clauses of §1.4 that say
  "on the lead's recommendation" and "which is the lead's", that the
  tables for forty instances were in the text he was sent, and that he
  took the recommended option in the four questions on the design and
  again in the question on the seed-audit lines.
- *The seed audit.* In §13.1: the paragraph "Who classifies, and when" as
  it now reads; rule (i) of the rules for the five patterns, which no
  longer says that the author compares the report with the table, and the
  sentence on the pre-audits after rule (iii); the pre-audit with the
  history dump, with its sentence on two file names that carry 300 as a
  parameter value; groups 6 and 7, with the commit messages in the row of
  class (b); the count added to group 1; the confirmation of the 21 lines;
  and the two paragraphs after it on who compares and why that is less
  than first written. In §13: the expected exit status and, in the
  paragraph on freeze 2, "the comparison of §13.1" where "the author's
  classification" stood, and the freeze-1 audit's own log and report
  among the planning data the second audit may find. His answer of
  11:46:30 to the question quoted in §13.1 covers one thing, that the 21
  lines are not uses of the block; the classes, the reasons and the
  procedure are the lead's wording.
- *The rest.* The status line; D-R3.5 in the list of decision records, and
  the definition of "the lead" before §1 (by its part in the work: asked
  at 12:04:31 whether the text should say who the lead is, the author
  answered that there was no need to name the lead; the wording is the
  lead's); in §5.1 and in the table of §14 on the two families,
  "inconclusive" said of P1 and P4 and their family-E counterparts only,
  as point 3 of §14 has it; in §13, one parenthesis on the
  redaction rules and "go-ahead" where "approval" stood; the five points
  added at the end of the tables of §14 (short forms of two caveats for
  the abstract; three cases the tables left open; names); the sentences of
  §15 on what §1.4 and §3.3 are for the paper and on the short forms; and,
  in this section, this entry, the entries "Freeze 1" and "Freeze 2" below
  — "Freeze 1" now asks for his go-ahead, and for his confirmation with
  the lead's comparison, where it asked for his approval and his signed
  classification; "Freeze 2" asks for the comparison and, for any line
  outside the 21, his own classification — and the last sentence of
  "After the run".

**The records, by freeze** (filled in as each is reached):

- Freeze 1: head, UTC time, `FREEZE_R3_<head>.sha256` digest; **the author's
  go-ahead**: entered above; **the external anchor** (§13: URL, commit and tag ids, the
  two sha256 of the text, the redaction list's sha256, and the push time
  GitHub's API reports — the third-party timestamp); the suite result at
  the frozen head (counts, the log's sha256); the
  three SUMMARY lines of the run-level reproduction; the rehearsal's head,
  its last `CHAIN END` line, its acceptance line and the SUMMARY line of
  the ensemble's reproduction; the seed-audit report
  path and sha256 with its verdict block; **the author's confirmation of
  11:46:30 UTC (§13.1) and the lead's line-by-line comparison of the
  freeze-1 report with the 21 lines**; and, for any line outside them, the
  author's own written classification (path, line, class, reason; name and
  UTC time).
- Freeze 2: `FREEZE_FINAL_r3lite.txt` verbatim with the UTC time; the
  freeze-2 seed-audit verdict block and report sha256; the launcher's
  `[AUDIT-GATE]` line, the comparison of the freeze-2 report with the 21
  lines, for any line outside them the author's own written
  classification, and the `--audit-ok` ledger note that records it.
- After the run: the chain's `CHAIN END` line, the sha256 of every pickle and
  report — the family-1 report, the family-E report, its record
  (`r3lite_family_e_record.json`) and the two assembly records — every
  repeated invocation with its cause, and **every `note`, `stop` and
  `invalidate` line of the chain ledger** (refused operator actions and
  honoured waivers included); if family E was not completed, that and its
  cause. The two seed-audit reports are released with the results, under
  the redaction rules of §13, each with the sha256 of the unredacted
  report as recorded here.
