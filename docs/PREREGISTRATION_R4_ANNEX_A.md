# R4 — Freeze 0, Annex A: the success margin, the test size and the residual pilot

Status: FROZEN on 20 October 2026. This annex is committed on branch `r4` and pushed, as a whole file, to the public repository before the first pilot training pass.

It amends section 4 of Freeze 0 (`docs/PREREGISTRATION_R4_FREEZE0.md`, pushed 2026-10-19 15:23:07 UTC) and adds the rules for the first R4 training. Everything in Freeze 0 not named here is unchanged.

## A.1 What is known when this annex is frozen

**Gate A** was applied mechanically on 20 October 2026.
- **T\* is `tu1`.** It sets one transmit power per site type and chooses the O-RU and Relay setpoint by the site's own UE count; MCS, relay and PRB are XL-DET's. Specification sha256 `2fd2b275835aa2bd69a9498d7e696223b5b59128636ef9359b6cc8b6a30afd00`.
- **Its margin over XL-DET** on development seeds 50–59 and 70–79 is +14.8 pp (A) and +17.0 pp (B). It passes all three guards.
- **The runner-up** is `tsq2`.

**Check B** (exploratory) found both existing learned policies below T\* on validation seeds 300–339.
- Replicate 1234: −7.4 pp (A), −2.2 pp (B).
- The ensemble: −6.4 pp (A), −5.8 pp (B).

**Training.** No MARL policy has been trained for R4, and no learned residual has been evaluated.

**Gate B** (section A.4) passed on 20 October 2026 on the code named in A.3. The logs are kept with the run records.

## A.2 The author's decision on the success margin (20 October 2026)

Taken before any R4 learned result exists. The author's aim is a learned controller that delivers more than the tuned engineered controller, "even by a little". A learned correction on top of T\* is expected to add only a small amount: about 0.5 pp on the twin's analytic proxy. So the primary family of Freeze 0 section 4 becomes the following fixed sequence. Each step must hold in Scenario A and in Scenario B (intersection-union, one-sided α = 0.025 each), and testing stops at the first failure:
1. θ_c > 0 pp
2. θ_r > 0 pp
3. θ_c > 1 pp
4. θ_r > 1 pp
5. θ_c > 3 pp
6. θ_r > 3 pp

**What does not change.**
- The non-inferiority margins: M7 ≤ +0.40, M4 ≥ −0.03, energy ≤ +5 %, life-safety ≥ −2 pp. They are against XL-DET-T\*, intersection-union, one-sided 97.5 %, with no multiplicity adjustment.
- The headline claim requires step 1 and every non-inferiority endpoint.
- The size of the effect is reported beside every claim.

**Test size.** n = 120 instances per scenario, from seeds 700–759 and 780–839. 780–839 are used only if their seed audit (run 20 October 2026, recorded) shows no result-data hit. If it does, the next unused block of 60 is used, named before Freeze 1. The test is a single fixed-size look, not sequential. The out-of-distribution block 760–779 is unchanged. K = 8 confirmatory replicates, all counted.

## A.3 The learned arm: MARL-R, a learned correction on top of T\*

**Executor.**
- Each acting site computes T\*'s target power exactly as `xldet_tu` does, with the same code and the same per-site dwell state.
- It adds a learned offset from {−12, −9, −6, −3, 0, +3, +6, +9, +12} dB.
- The executed target is clip(target + offset, 10, 33) dBm, reached with the engine's power steps.
- The policy chooses an offset once every 10 acting ticks and holds it for those 10 ticks. T\*'s own 10-tick adoption rule then adopts it.
- The policy samples from its own random generator, so the environment's random streams are untouched.
- Relay, MCS and PRB decisions are XL-DET's own, pinned and untrained.
- Evaluation uses the argmax of the offset head.

**Information.**
- The actor reads the 26 R2 actor fields, the site-type one-hot, its own current offset and T\*'s current target. A comparator could read every one of these.
- The twin is used only in training.

**Initialisation.** The offset head is initialised so that its argmax is 0, so the untrained policy reproduces T\* bit for bit (Gate B, below).

**Credit** (training only). Two variants:
- **R-dJ.** The per-agent reward is c · [J(P_t) − J(P_t with site i at T\*'s command)], with c = 0.1.
  - J = Σ_i max(1, nUE_i) · SE(SINR_i), computed from the live engine state with the replica SINR model.
  - "T\*'s command" is the power the site would hold had it followed T\* from its first acting tick. A site at offset 0 gets zero credit.
  - Each decision is credited with the ticks during which it is in force, [k+10, k+20), discounted by 0.99¹⁰ per decision with λ = 0.95.
  - There is no cross-agent centring.
- **R-comb′.** R-dJ plus a delivery-based team stream (the tick's fleet user-to-user delivery change) at β = 0.5. Each stream is standardised separately, and the team stream is not centred away.

**Common to both.**
- A KL penalty toward the prior P(offset 0) = 0.8, at 0.02, replacing the entropy bonus.
- A small fixed charge on power held above T\*'s command: 0.01 pp per dB per tick.
- A per-agent critic, used in training only: the 37 R2 critic fields, the 5 residual extras and 2 clock fields, with one value head per stream.
- All constants are named in one regime id, `resid-regime1-3e27134ae7d89853`.
- Time-limit truncation bootstrapped with V(s_T).

**Training regime.**
- 800-tick episodes, 16 per round (8 A, 8 B).
- Topologies are drawn independently and stratified per round from the 120 training seeds (42–49, 100–139, 1000–1071), so that every topology appears in both scenario classes.
- Events: A severance ~U[100, 300]; B rescue ~U[200, 600]; fragmenting cuts in B with probability 0.5 (training only).
- A separate data stream for each replicate.
- Learning rates: actor 1e-4, critic 3e-4, constant.
- A checkpoint every 5 rounds.
- Every setting is recorded in the run's configuration file.

The code is the commit `a8ef81e2af77a9c48a36222917762541a6597245`. Its tree hash on the compute host is `29e7fc959b274b3a2a926e400f18ed9e20cb5b87590dc4a9543f40288c12b175` (248 files).

An independent review of this code is running in parallel with the pilot. If it requires a change to any code used in training or evaluation, the pilot is rerun on the corrected commit, and both runs are reported.

## A.4 Gate B: before any training

All must pass, and each must be able to fail:
1. **Identity.** The untrained MARL-R equals T\* bit for bit: 2 development seeds × A, B × 300 ticks, and one 3,600-tick pass. A forced-offset run (+3 dB everywhere) must differ.
2. **Replica accuracy.** The replica SINR equals the live engine's within 1e-9 dB (20 ticks × 3 episodes).
3. **Credit sign.** On forked single-site power raises in the live engine, the credit's sign must agree with the engine-measured change in J on at least 95 % of pairs with |ΔJ| ≥ 0.5, and its neighbour term must be negative wherever the neighbours lose SE.
   - **Result:** 100 % agreement (417 pairs, largest difference 8e-14), and the neighbour term is negative on 100 % of the 434 neighbour-loss pairs. A selfish own-SE credit agrees on 0 %.
   - **Disclosed:** the criterion first drafted, dJ_i < 0 on 95 % of neighbour-loss pairs, holds on only 67.5 % of them. That draft criterion was wrong, not the code: where the site's own gain outweighs its neighbours' loss, J rises and the credit is rightly positive.
4. **Pinned heads** receive exactly zero gradient.
5. **Truncation.** The time-limit bootstrap is used.
6. **Determinism.** The same replicate seed twice gives identical weights after one round.
7. **No change to existing controllers.** XL-DET and T\* behave unchanged.

## A.5 The pilot (development seeds only)

**Runs.**
- 2 variants (R-dJ, R-comb′) × 2 replicates × 40 rounds.
- Replicate seeds (replicate seed, data seed), the same for both variants: (4101, 9101) and (4102, 9102).

**Evaluation.**
- The round-40 checkpoint of each replicate, argmax executor, 3,600 ticks.
- Development seeds 50–59, 70–79 and 90–99 × A, B.
- Paired with T\* on the same seeds and scenario. The Gate A passes cover 50–59 and 70–79; new T\* passes cover 90–99, on the Gate A tree or after an identity check.

**Reading.** For variant v and scenario s, d̄_v,s is the mean over the variant's 2 replicates and the 30 seeds of (MARL-R − T\*) achievability.

**Choice of variant.** The variant with the larger min over s of d̄_v,s. If they are within 0.2 pp, the choice is R-dJ.

**GO** to Freeze 1 requires, for the chosen variant, both of:
- d̄_A > 0 and d̄_B > 0;
- all four safety readings:
  - each replicate's mean (MARL-R − T\*) is at least −1 pp in both scenarios;
  - offset changes are at most 20 per site per 1,000 ticks;
  - the point estimates of M7 difference, energy ratio and life-safety difference against T\* are within the non-inferiority margins;
  - Gate B passed.

**Otherwise NO-GO.** No further R4 training is done, and R4 is reported as stopped after the pilot, with every number.

**After a GO.** The predicted probability of passing step 1 at n = 120 and K = 8 is reported, with θ = 0.7 × d̄ and the variance read from the pilot. It informs the author and does not change the rule.

## A.6 After a GO

**Freeze 1 fixes:**
- the variant;
- N = 40 rounds;
- the eight replicate seeds;
- the code and tree hashes;
- the analysis code of the confirmatory test (θ_c and θ_r as in Freeze 0; the six-step sequence above; the non-inferiority family);
- the seed lists.

**Then:**
1. The eight replicates are trained.
2. G3, a futility reading on validation seeds 300–339: θ_c's point estimate must exceed 0 in both scenarios, with the non-inferiority point estimates inside their margins. Otherwise the study stops and the test block is not read.
3. Freeze 2 (analysis code hash, manifest, seed audit).
4. The test.

All of it is reported whatever the outcome.
