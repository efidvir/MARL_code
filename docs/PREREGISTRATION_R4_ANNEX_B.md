# R4 — Freeze 0, Annex B: the first pilot run stopped, and the pilot design amended

Status: FROZEN on 20 October 2026. This annex is committed on branch `r4` and pushed, as a whole file, to the public repository before the first training pass of the rerun.

It amends Annex A (`docs/PREREGISTRATION_R4_ANNEX_A.md`, pushed 2026-10-20 08:59:24 UTC). Everything in Freeze 0 and Annex A not named here is unchanged, including:
- the success margin (θ_c > 0 pp in both scenarios first);
- the non-inferiority margins;
- the test size;
- the pilot's GO rule;
- the evaluation seeds and the pilot replicate seeds.

## B.1 The first pilot run was stopped before any evaluation

The four pilot runs started at 08:59:42 UTC on code commit `a8ef81e` and were stopped at 09:57:38 UTC, at round 8 of 40. No pilot policy was evaluated, and no achievability, life-safety delivery or M7 of any pilot policy was ever computed. There were two reasons.

**1. The independent review found defects in the training code.** Annex A.3 requires a rerun on the corrected commit when this happens. The defects:
- **(L5)** For a site with no working infrastructure link, the counterfactual of the credit did not move the site's serving distance with its power, as the engine does.
- **(L3)** The shadow command was reset after any gap in a site's acting ticks, erasing that site's credit.
- **(L4)** Training failed when a site stopped acting before the end of an episode.

**2. The training diagnostics showed that the design could not change the evaluated policy.**
- The offset head was initialised at the prior P(0) = 0.8, a logit gap of 3.47 nats over every other offset.
- Training was penalised toward that prior and sampled at temperature 1.
- Evaluation takes the argmax.

At round 8 the argmax was offset 0 on every probed state (`argmax_nonzero` = 0.000), and the sampled non-zero share stayed at the prior's 20 %. So the evaluated policy would have reproduced T\* exactly, a tie fixed by the initialisation and not by what was learned. These diagnostics are offsets and losses only; they carry no outcome.

The records of the stopped run (configurations, round records up to round 8, logs) are kept and released.

## B.2 The amended pilot design

The amendment is decided before any pilot outcome exists.

1. **Initialisation and prior.**
   - The offset head's bias is the log of the prior P(0) = 0.30, with 0.0875 on each of the 8 non-zero offsets (logit gap 1.23 nats).
   - The final-layer weights are as before, so the untrained argmax is still offset 0. The untrained policy still reproduces T\* bit for bit, which Gate B checks.
2. **KL penalty** toward that same prior, at 0.02 (the value is unchanged).
3. **Training temperature.**
   - T = 1.0 for rounds 1–10, then linear to T = 0.4 at round 40, so that the policy trained approaches the argmax policy that is evaluated.
   - Sampling and the PPO probability ratio both use the tempered distribution.
   - Evaluation stays argmax.
4. **Argmax probes** every 5 rounds on 8 fixed training topologies. They record the offsets only.
5. **Stop rules for each run.** Two advantage checks at round 10 stop a run only if they fail:
   - (c2) the sign of the final actor advantage on decisions that the engine measures as harmful, against its control;
   - (i) both reward streams reach the advantage, each with a control that must fail.

   The critic's explained variance is recorded as a reading and never stops a run.

**What is unchanged:**
- the corrected training code (`9908451` with this amendment);
- the credit variants R-dJ and R-comb′;
- the training regime;
- the 40 rounds;
- the replicate seeds (4101, 9101) and (4102, 9102);
- the evaluation (round-40 checkpoint, argmax, 3,600 ticks, development seeds 50–59, 70–79 and 90–99 × A, B, paired with T\*);
- the GO rule of Annex A.5.

The code is the commit `b323c866f8fc549804fc780688c04700b96a5146`, regime `resid-regime3-416702b5ddd13e5d`. Its tree hash on the compute host is `4644819ff46b3a454bea3f87559f3f01b6d5cb7619610fdde744d101d5931d73` (252 files).

**Implementation choices that this annex fixes:**
- The KL penalty is computed on the untempered distribution. On the tempered one, it would pull the logits toward T · log(prior) and work against the temperature schedule.
- The PPO update recomputes the log-probabilities at the round's temperature. It refuses a batch whose temperature differs, and it stops if the recomputed log-probabilities differ from the rollout's by more than 1e-4.

**Gate B passed on that tree on 20 October 2026, every check.**
- **(a) Identity:** the untrained policy equals T\* at 300 and 3,600 ticks, and a forced offset differs.
- **(b) Replica accuracy:** within 2.8e-14 dB.
- **(b2)** Credit for an isolated site agrees with the engine (144 sites).
- **(c) Credit sign:** 1.000 agreement on 408 pairs.
- **(c2) Final-advantage sign on engine-harmful decisions:** R-dJ 0.907 against its control's 0.584; R-comb′ 0.902 against 0.614.
- **(d)–(g)** pass; unit tests 158 passed, 0 failed.
- **(i)** Both streams reach the advantage.
- **(h)** is a reading, not a check: the critic's explained variance at round 11 was 0.169 (R-dJ) and 0.160 (R-comb′).

The thresholds of (c2) are those frozen with Annex A. A numerical check at the new prior's 70 % share of non-zero decisions found that they keep their meaning.

## B.3 Reporting

- Both the stopped run and the rerun are reported.
- The GO/NO-GO decision is taken on the rerun only, under Annex A.5's unchanged rule.
- If the rerun reads NO-GO, no further pilot is run, and R4 is reported as stopped after the pilot.
