# Amendment 1 to the confirmatory pre-registration

**Status:** written, hashed and externally timestamped after the original
evaluation job started. It is timestamped before any result for seeds 80–99
was read. At the time of writing, that job's result file did not exist, and
neither its log nor any partial output had been read. Its hash is recorded in
`FREEZE_A1.sha256`, and it is committed to the Overleaf git history
(`records/`).

## 1. What was found

An audit of the frozen code, made while the original confirmatory evaluation
was running, found that the arm pre-registered as MARL-RIC (`marl_freeze`)
does not implement the controller the manuscript describes. The manuscript
says the twin-trained parameters are executed unchanged from the first tick,
with no model selection at test time. The code differs in three ways.

- **Online learning during the event.** From the severance until 400 ticks
  after the last scenario event, the arm runs pooled PPO updates with an EWC
  anchor and samples its actions. After that point it freezes and executes
  deterministically. A partitioned network could not perform the pooled
  update.
- **A guard that reads the test metric.** At the freeze point, a guard checks
  the arm's own trailing achievability on the topology under test. The guard
  can postpone the freeze. It can also restore the policy snapshot that scored
  best on that topology.
- **Scenario B carries over Scenario A's weights.** Scenario B starts from the
  weights the arm adapted during its Scenario A pass on the same topology.

The development-set logs (seeds 50–59 and 70–79) record at least:

- 34 freeze events, 33 of which passed the guard and 1 of which was forced;
- 16 postponed freezes;
- 17 restored best-scoring snapshots;
- a Scenario A → B weight hand-over on every instance.

These counts are lower bounds. The harness's worker processes buffer their
output, and the pool terminates them at the end. The logs hold 158 of 160 and
160 of 160 pass-start lines, so a few trailing lines may be missing.

## 2. What changes

### 2.1 The controller under test

Every primary hypothesis P1–P9 now tests **`marl_static`**. This arm executes
the loaded, twin-trained parameters unchanged from tick 0.

- **Execution is deterministic.** Every categorical head takes its argmax, and
  the resource split takes the Dirichlet mean.
- **No adaptation.** There is no online update, no exploration temperature and
  no EWC anchor.
- **No self-selection.** There is no freeze guard and no snapshot restore.
- **One checkpoint for both scenarios.** The Scenario B pass runs in the first
  scheduling phase from the base checkpoint.
- **Shared code path.** It shares with every other MARL-family arm the
  observations, the three-tick detection floor, the relay mechanics and the
  engine flags.
- **Invariants checked at run time.** The arm stops with an error if any agent
  starts sampling at any tick, or if the policy parameters are not
  bit-identical at the end of a pass.

### 2.2 The code change

The code is the frozen tree with two additions:

- the `marl_static` arm (`run6_static/marl_static.diff`, 78 lines added and 8
  changed);
- a guard that skips the harness's second scheduling phase when it has no
  tasks.

Every other file is byte-identical to the frozen tree. This was checked
recursively over the whole package.

The earlier freeze had missed the `sixg_sim/iops/` subpackage. Those files
date from April–May 2026 and are unchanged. `FREEZE_A1.sha256` hashes the
complete package recursively.

The static arm also stores the parameter fingerprints taken at the start and
at the end of each pass in its returned data (`static_fp_start`,
`static_fp_end`). The invariant can therefore be audited from the result
files, without relying on the logs.

The new arm was tested before this amendment was hashed (`smoke_A1.txt`,
`smoke2_A1.txt`):

- **Static behaviour.** The static setup and parameter checks pass, and the
  audit fingerprints match in both scenarios. The arm never freezes and never
  uses EWC, and its Scenario B pass runs in the first phase.
- **Determinism.** Repeated runs agree bit for bit on every series except
  energy and latency.
- **Baselines.** The baseline arms produce the same output under the patched
  and the original harness: every series agrees bit for bit except energy and
  latency.
- **The energy and latency jitter.** Both series carry an accounting jitter
  that is present in the original harness too. It is fixed once per harness
  invocation: every arm in the same invocation shares it, and it changes
  between invocations. That points to the per-process hash seed. Across the
  two smoke comparisons, per-tick energy differed by up to 2.3 %, run-mean
  energy by up to 0.35 %, and run-mean latency by up to 1.3 %.
  Achievability, components, bridges and delivered volume are unaffected.
  The Amendment-1 runs set `PYTHONHASHSEED=0` so that their own
  energy/latency accounting is reproducible. The decision series do not
  depend on the hash seed.

### 2.3 Replicate selection

The deployed replicate is **re-selected with the static executor**, using the
original rule and set. The rule is applied to all four trained replicates
(torch seeds 1234, 2345, 3456, 4567) on validation seeds 60–66, Scenario A:
choose the lowest mean steady-state component count, and break ties by the
higher mean achievability. The selection is recorded in `A1_selection.txt`
before any seed 80–99 pass of the static arm starts. Test data plays no part
in it.

### 2.4 Where each arm's results come from

- **`marl_static` on seeds 80–99.** Each instance is evaluated once, in both
  scenarios, with the selected checkpoint and the frozen engine and policy
  flags. The run is split into two batches of ten seeds (80–89 and 90–99) to
  stay within server memory alongside the original job. Splitting does not
  change any pass.
- **Baseline arms on seeds 80–99** (XL-DET, SDN, OSPF, OLSR, B.A.T.M.A.N.,
  AODV, uniform random). Taken unchanged from the original confirmatory job
  (`cf_eval_80_99.pkl`). Their code path is untouched, and the smoke test
  shows their output is bit-identical under both harnesses.
- **`marl_freeze` on seeds 80–99.** Also from the original job. It is reported
  as a **secondary, disclosed variant** and is never labelled the proposed
  controller.

### 2.5 Analysis

`confirmatory_analysis.py` gains two options:

- `--marl-arm`, which selects the arm the primary hypotheses test;
- merging of the static arm's result file with the baseline file.

It also gains **S7**: the paired difference between `marl_static` and
`marl_freeze` in achievability and in components, for both scenarios.

**Energy sensitivity (S8).** `marl_static` and the pre-registered baselines
come from different invocations, so the paired energy comparisons P7–P9
include the cross-invocation accounting jitter described in §2.2. To bound
it, XL-DET is also run in the same invocation as `marl_static` on seeds 80–99.
`A1_sensitivity.py` then reports three things:

- whether XL-DET's decision series reproduce bit for bit between the original
  job and the co-run;
- the size of the energy and latency jitter at full length;
- P7 recomputed against the co-run XL-DET.

S8 is descriptive and outside the Holm family. The primary analysis still
takes every baseline from the original job: it reads a copy of each co-run
result file that contains only `marl_static`.

With the default arm, the updated script reproduces the original
development-set validation report byte for byte (`dev_regression.json`).

The confirmatory analysis is run with `--marl-arm marl_static`.

Hypotheses P1–P9, the metrics, the windows, the comparators, the tests, the
Holm family, the decision rule, the S1–S6 analyses and the no-exclusion rule
are all unchanged.

### 2.6 The development set

The development set is re-run with `marl_static` and the re-selected
checkpoint. From then on, every development-set figure and table in the
manuscript describes the same controller as the confirmatory test. The
existing `marl_freeze` development results are kept and reported as the
disclosed variant. The Scenario A interferer stress test and the per-instance
end-state figure are also re-run with `marl_static`.

## 3. Unchanged

Everything in `PREREGISTRATION.md` that this document does not change remains
in force. That includes the rule that nothing is changed after the confirmatory
results are seen.
