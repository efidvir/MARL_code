# R4 — record of the study as run, and its stop

Recorded on 21 October 2026. R4 ran under Freeze 0 (pushed 2026-10-19 15:23:07 UTC), Annex A (2026-10-20 08:59:24 UTC) and Annex B (2026-10-20 13:16:43 UTC). It stopped after its pilot, before Freeze 1.

No R4 confirmatory test was run. Seeds 700–899 were never read. Everything below is development or validation data and spends no confirmatory alpha.

## 1. Gate A: the tuned engineered comparator (development seeds 50–59 and 70–79, n = 20 per scenario)

Applied mechanically on 2026-10-20 to 30 configurations; 25 were eligible.
- **T\***: the rule that sets one transmit power per site type and chooses the O-RU and Relay setpoint by the site's own UE count.
- **Its margin over XL-DET:** +14.8 pp (Scenario A) and +17.0 pp (Scenario B) in steady-state achievability.
- **Guards:** all three pass.
  - Life-safety delivery: −1.1 / −0.1 pp.
  - Sites below the SE threshold: +1.2 / +1.2.
  - M7: 1.43 against 1.49 (A) and 1.50 against 1.50 (B).
- **Runner-up:** the per-type setpoint indexed by own SINR.
- **The confirmation on seeds 80–89** (descriptive) and the full table of configurations are in the run records.

## 2. Check B: the existing learned policies against T\* (exploratory, validation seeds 300–339, n = 40 per scenario)

Learned policy minus T\*, in steady-state achievability, mean with 95 % interval:

| Policy | Scenario A | Scenario B |
|---|---|---|
| Replicate 1234 | −7.4 pp [−11.1, −3.7] | −2.2 pp [−3.5, −1.0] |
| The four-replicate ensemble | −6.4 pp [−8.0, −4.7] | −5.8 pp [−7.0, −4.6] |

## 3. The pilot of the learned correction on top of T\*

**The first run** (code `a8ef81e`) was stopped at round 8 of 40, before any evaluation. Annex B gives the reasons.

**The rerun:**
- Code `b323c86`, regime `resid-regime3-416702b5ddd13e5d`; Gate B passed in full.
- Training: 2026-10-20 13:17–19:30 UTC; all four runs passed the round-10 checks and reached round 40.
- Evaluation: the round-40 checkpoints with the argmax executor, 3,600 ticks, development seeds 50–59, 70–79 and 90–99 × A, B (n = 30 per scenario), paired with T\*. It finished at 21:58 UTC with 300 passes, 0 failed.

**Results, learned correction minus T\*** (pp of steady-state achievability):

| Variant | Replicate | Scenario A | Scenario B |
|---|---|---|---|
| Difference-of-objective credit | 4101 | +1.74 [+0.51, +2.92] | +1.42 [+0.15, +2.72] |
| Difference-of-objective credit | 4102 | +0.09 [−1.63, +1.78] | −0.17 [−1.87, +1.54] |
| Difference-of-objective credit | **mean** | **+0.92** | **+0.63** |
| With the delivery team stream | 4101 | +0.83 [−1.10, +2.66] | +0.38 [−1.39, +2.18] |
| With the delivery team stream | 4102 | −0.62 [−2.55, +1.28] | −1.20 [−3.15, +0.76] |
| With the delivery team stream | **mean** | **+0.10** | **−0.41** |

## 4. Annex A.5 reading

**Variant chosen:** the difference-of-objective credit, by the larger min over scenarios of d̄ (+0.63 against −0.41).

**GO criteria, all met.**
- **Margins:** d̄_A = +0.92 > 0 and d̄_B = +0.63 > 0.
- **Safety readings:**
  - Each replicate's mean is at least −1 pp in both scenarios (lowest −0.17).
  - Adopted-offset changes average 2.2 and 3.4 per site per 1,000 acting ticks for the two replicates. On about 1 % of site-passes a site oscillates, at up to 99 per 1,000.
  - Non-inferiority point estimates against T\*, all within the margins:
    - M7 difference: +0.015 (A) and 0.000 (B), margin +0.40.
    - Energy: +1.8 % and +1.9 %, margin +5 %.
    - Life-safety delivery: −1.09 pp (A) and −1.97 pp (B), margin −2 pp.
  - Gate B passed.

**Also observed (not a criterion).** The learned correction lowers power at sites without users (O-DU and Relay, to about 12 dBm against 20 for T\*). Sites below the SE threshold therefore rise by +6.3 (A) and +5.5 (B).

## 5. The decision to stop (21 October 2026, about 08:05 UTC)

The pilot met the go rule. We decided not to proceed to Freeze 1 and the confirmatory test.

**Why:** the life-safety point estimate in Scenario B, −1.97 pp, sits at the −2 pp non-inferiority margin. A one-sided 97.5 % bound on the test block would therefore almost certainly fall below the margin, so the non-inferiority family and the headline joint claim would very likely fail. The estimated probability of a clean pass was about 5–10 %, against the cost of eight further training runs and 120 test instances per scenario.

R4 ends here. It is reported as stopped after its pilot, with every number above, in the next version of the paper. There, the learned correction's gain over T\* is presented as an exploratory development result, together with its life-safety and coverage cost.
