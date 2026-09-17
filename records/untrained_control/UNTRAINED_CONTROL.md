# Untrained-actor control: a post-hoc supplementary check

**Status.** This check was written, hashed and committed to the Overleaf
`records/` history before any full-length run of it. It is post hoc: the
confirmatory results (`records/results_A1/`) had already been read. It is
descriptive only and changes no pre-registered hypothesis, test or decision.

## Why

An external reviewer asked for a randomly initialised but deterministic neural
actor. Its purpose is to separate what the actor architecture and its
initialisation contribute from what training contributes.

## What is run

- **Executor.** The frozen executor `marl_static` from Amendment 1, with the
  harness unchanged. `run_timeline_comparison.py` has md5
  `7ec6ecde30efb75d56b6fcbf8480dc91`, the file hashed in `FREEZE_A1.sha256`.
  The tree is a copy of `~/MARL_run_6` in `~/MARL_run_7`, and `sixg_sim/` is
  byte-identical.
- **Actor.** The shared actor before any training update, one per replicate
  torch seed (1234, 2345, 3456, 4567).
  - `make_untrained_checkpoints.py` builds it by the path
    `train_on_comparison.py` runs before its first episode:
    1. seed torch and random with the replicate seed;
    2. build the canonical simulator on training seed 42 (800-tick episode);
    3. construct `MAPPOTrainer`, which installs the first agent's
       `PolicyNetwork` as the shared actor.
  - No gradient step is taken.
  - Two independent constructions per seed gave identical parameters.
  - The checkpoints use the same format as `policy_final.pt`. Their SHA-256
    values are listed in `UNTRAINED_CHECKPOINTS.sha256`.
- **Execution.** Deterministic: the argmax of every categorical head and the
  Dirichlet mean. Parameter fingerprints are audited at the start and end of
  every pass.
- **Instances.** The confirmatory seeds 80–99, both scenarios, with the same
  engine and policy environment as the Amendment-1 runs and
  `PYTHONHASHSEED=0`.
- **Order.** s1234 first, since it is the starting point of the selected
  replicate. Then s2345, s3456 and s4567.
- **Other arms.** Neither the trained frozen executor nor any baseline is
  re-run. The trained arm comes from `~/MARL_run_6/output/cf_static_*.pkl`,
  and the baselines from `~/MARL_run_5/output/cf_eval_80_99.pkl`.

**Smoke test (disclosed).** Before this file was written, one shortened pass
was run to check the pipeline: s1234, seed 80, 1,600 ticks. Its end values
were displayed: Scenario A achievability 31.1% with 4.0 components, and
Scenario B 43.1% with 3.0 components. That pass is not part of the analysis.

## Analysis (`untrained_analysis.py`, written and validated before the runs)

The script reproduced the trained arm's confirmatory values, and returned
zero differences when a copy of the trained runs stood in for an untrained
actor.

**For each untrained actor, and for the mean of the four:**
- Scenario A and B steady-state achievability;
- components;
- bridges;
- instances at the attainable optimum;
- pooled energy per delivered unit;
- the Scenario B pre-surge level.

**Paired per instance (mean, 95% t-interval, sign test, exact Wilcoxon):**
- achievability, trained frozen minus untrained;
- achievability, untrained minus uniform random;
- achievability, untrained minus the best routing protocol;
- components, untrained minus trained;
- Scenario A energy per unit, untrained minus trained;
- trained minus the per-instance mean of the four untrained actors.

**Rules.**
- There is no decision rule, no Holm correction and no exclusion.
- Every run is reported, whatever it shows, as a post-hoc supplementary
  control.
- Energy comparisons with the other arms cross harness invocations, so the
  accounting jitter of `AMENDMENT_1.md` §2.2 applies.
