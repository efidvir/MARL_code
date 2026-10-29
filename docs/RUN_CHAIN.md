# RUN_CHAIN — the staged launcher for the final corrected rerun

Owner: RUN-CHAIN track. Files owned here: `confirmatory/run_final.sh`,
`confirmatory/chain_plan.py`, `confirmatory/chain_lib.sh`,
`confirmatory/stage_*.sh`, `tools/preflight.py`, `tools/stage_report.py`,
`tests/test_run_chain*.py`, `tests/fake_stage.sh`, this document.
**No simulator, driver or analysis file is modified by this track.** Every
change those files need is recorded below as a REQUEST and worked around.

Binding source: `COMPUTE_PLAN.md` (19.4 h end to end, primary result at
T+14.7 h). This document says how each of its requirements is implemented and
where the implementation deviates.

---

## 1. Running it

```bash
cd ~/MARL_run_9d
bash confirmatory/run_final.sh --profile rehearsal --dry-run   # what would run
bash confirmatory/run_final.sh --profile rehearsal             # the rehearsal
bash confirmatory/run_final.sh --profile real                  # the real run
bash confirmatory/run_final.sh --profile real --from S6        # resume/continue
bash confirmatory/run_final.sh --profile real --shed sensitivity
bash confirmatory/run_final.sh --profile real --waive-dep S6   # recorded waiver
python3 tools/stage_report.py --timings confirmatory/chain_real.jsonl --profile real
python3 tools/stage_report.py --timings confirmatory/chain_real.jsonl --profile real --run all
```

`--timings` reports on ONE chain run: `--run last` (the default, everything
since the last `chain_start`), `--run ok` (the last run that ended `rc=0`) or
`--run all` (the whole ledger). It always states how many invocations it
summed and how much wall clock the runs it excluded hold, so a figure quoted
from this document can be reproduced by the command printed beside it.

Profiles (`confirmatory/chain_plan.py`):

| profile | seeds | ticks | training | eval workers | output |
|---|---|---|---|---|---|
| `real` | dev 50–59+70–79, val 60–66, confirmatory 200–219 | 3600 | 4 replicates, 400 ep, 800 ticks | 24 | `output/final/` |
| `rehearsal` | dev 50,51,70,71 (+ **full-length** dev subset 50,51); val 60,61; confirmatory 220–223, unregistered (+ **full-length** confirmatory subset 220,221, WP13). Until `31b034c` this profile named the first four of the pre-registered block instead (§6; PREREGISTRATION_R2 §5) | 600, subsets 3600 | **3** replicates, 32 ep, 200 ticks, 2 alive | 24 capped by load | `output/rehearsal/` |
| `dryrun` | as `real`, except confirmatory 220–239 (unregistered) | 3600 | as `real` | 24 | `output/dryrun/` |
| `test` | paper only; confirmatory 220,221 since `31b034c` | 60 | paper only | 4 | `output/test/` |

Two further profiles, `r3lite` and `r3lite_rehearsal` (R3-lite, 2026-10-11),
have their own stage graph and are described in §7; everything in §1–§6 is
about the four profiles above, which R3-lite leaves unchanged.

The rehearsal's numbers are **throwaway**. They live under `output/rehearsal/`
and `confirmatory/rehearsal_*`; `tests/test_run_chain.py` asserts that no
rehearsal artefact path can collide with a real-run path.

Everything the chain starts runs under `nice -n 19` with `PYTHONHASHSEED=0`
and `OMP/MKL/OPENBLAS_NUM_THREADS=1`, written into a per-invocation env file
(`confirmatory/env_<profile>/<inv>.sh`) that is sourced by the job script and
by any re-run.

## 2. Stages and invocations

`chain_plan.py` is the plan as data: `run_final.sh` and the tests read the
same object, so the properties below are asserted, not asserted-to-be.

| stage | plan node | invocations (real) | passes | notes |
|---|---|---|---|---|
| S0 | [C] manifest, [B] untrained checkpoints | 2 | 0 | frozen tree only |
| S1 | [F] dev + [H] stress, policy-independent | 6 | 340 | **ahead of training** |
| S2 | [D] training, EXCLUSIVE | 4 | 0 | 2 alive, chained individually |
| S3 | [E] validation | 4 | 56 | one per replicate |
| S3b | selection rule | 1 | 0 | one rule, one application |
| S4 | [G]+[I]+[J] | 3 | 60 | + end-state dump |
| S5 | [K] dev gate + [L] FREEZE_FINAL | 1 | 0 | technical only |
| S5b | analysis dry run on the development set | 1 | 0 | gates S6 |
| S6 | [M]+[N] confirmatory | 5 | 320 | 8 arms per invocation |
| S7 | [R] primary analysis | 1 | 0 | gated on the full key set |
| S6h | analysis-S8 repeat, PYTHONHASHSEED=1 | 1 | 4 | after S7, never shed (FIX-PREFREEZE item 2) |
| S8 | [O] untrained control | 4 | 160 | sheddable |
| S9 | [P] adapt-then-freeze | 1 | 40 | serial, sheddable |
| S10/S11 | [Q1]/[Q2] sensitivities | 1 + 1 | 40 + 40 | sheddable first |
| S12 | [S] secondary: S7-secondary, S8, S12 + descriptive | 1 | 0 | fails on a missing input unless shed |
| | | **37** | **1060** | the plan's 1056 + the S8 repeat |

Splitting: `block_size(arms) = 64 // (len(arms) * 2)`; every arm of an
invocation runs every seed of its block in both scenarios, so a partition can
neither duplicate nor drop a pass, and no invocation exceeds 64 passes. Each
evaluation invocation writes its own pickle and its sha256 goes in the ledger.

## 3. How each plan requirement is implemented

**Preflight (`tools/preflight.py`), before every invocation, recorded and
gated.** Load average, MemAvailable, `/home` free, leftover jobs of ours, the
tree provenance digest, the tree pin, `PYTHONHASHSEED`, thread pinning. Hard
gates: evaluation ≥ 18 GB MemAvailable and ≥ 100 GB free; training ≥ 26 GB
with an immediate abort below 22 GB (exit 2, no retry loop). Every check
appends a record; the summary prints `checks=N failed=K`; **zero checks is a
failure**, and an unreadable `/proc/meminfo` is a failure, not a pass. Each
run is written to `confirmatory/preflight_<profile>/<inv>.<time>.json`.

*Unit:* every memory figure here and in the preflight's messages is GiB
(MemAvailable kB / 1024², the unit the performance track measured the wave
in), although it is written "GB": the training gate is 26 GiB = 27.9 GB
(COMPUTE_PLAN.md §6, D-FIX.14).

*Both* training thresholds are **wave** figures. COMPUTE_PLAN.md (§6) states 26 GB
and the 22 GB abort for *two concurrent replicates* (19.8 GB PSS / 29.9 GB
sum-RSS), and once replicate 1 of a wave is resident, MemAvailable has already
fallen by its footprint. `preflight.memory_verdict()` therefore tests both
against MemAvailable **plus** the PSS already held by this chain's own
expected jobs. Applying the abort to the raw figure (as it did until now)
aborted the second replicate of every wave: at the plan's real-scale numbers
on this host — MemAvailable 27.3 GB before the wave, ~9.9 GB PSS resident
after replicate 1 — the raw figure is ~17.4 GB, below the 22 GB floor, on
every wave. `tests/test_run_chain.py` drives the function with exactly those
numbers.

*The tree is pinned from the FIRST invocation of a run*, not from
FREEZE_FINAL. `FREEZE_FINAL_<profile>.txt` is written by S5, which comes after
the 340 policy-independent passes, the ~7 h training block, validation and
selection; until then nothing stopped the tree moving underneath the chain,
and three other code tracks are landing work packages in it. Preflight now
records `tree_sha256` in the ledger, `--ledger-tree-pin` reads back the first
one of the current run, and any later invocation whose digest differs fails at
the **batch boundary** instead of at the S5b/S6 cross-provenance gate twelve
hours later.

*Leftover jobs* are the ones whose `/proc/<pid>/cwd` is inside **our** tree.
Other tracks' trees (`~/MARL_run_9`, `_9b`, `_9c`) and other tenants are
counted, recorded and fed into the worker recommendation, but they do not gate
— this machine is shared and gating on them would never launch.

**Worker counts.** The `real` profile uses the plan's 24 for evaluation and 16
for training, with the plan's batch-boundary reductions (16 below 12 GB
MemAvailable, 8 below 8 GB). Every other profile additionally applies the
operating rule for a shared machine: 16 while the 1-minute load is below 12,
8 otherwise. The worker count is chosen at the batch boundary and never
changed inside an invocation. A re-run keeps the original env file, worker
count included, because the pre-registered rule is "unchanged".

**Training is exclusive, in code.** `chain_lib.sh` implements a lock with two
modes: every evaluation invocation takes it *shared*, the training stage takes
it *exclusive* for its whole duration. Exclusive acquisition fails while any
shared holder exists; shared acquisition fails while the exclusive lock is
held. On top of that, the training preflight refuses to launch while any job
of ours is alive. Stale locks are cleared only when their owner pid is gone,
and the clearing is logged.

**Training replicates are chained individually** (`stage_train.sh`): exactly
`concurrent` (2) alive, the next launched the moment one exits, never
gang-waiting per wave. Every replicate gets `--eval-episodes 0
--baseline-episodes 0`.

**A wave is DELAYED, never abandoned.** COMPUTE_PLAN.md failure handling B:
"If tenant load is already above ~20 cores when a training wave is due to
start: DELAY the wave rather than run it. A perturbed, slow training block is
worse than a late one." Both preconditions are gates — the wave memory figure
and a per-profile load ceiling (`train_load_ceiling`, 20.0 for `real`) — and
both are *retried*, each wait recorded as `reason=wave_delayed`. The load
ceiling applies only when the wave STARTS (no replicate of this chain alive),
because once replicates are resident they are themselves the load. Only an
exit-2 abort, i.e. below the hard memory floor, is never retried.
`--preflight-retries` and `--preflight-wait` set how long the chain will wait
and apply to evaluation invocations and to the training wave alike; the
defaults (3 × 300 s) are tuned for the rehearsal and an operator should raise
them for the real run, where the plan expects to wait for a quiet window.

**Restarting a run, and the ledger.** One ledger
(`confirmatory/chain_<profile>.jsonl`) is one run of the chain; a resume after
a stop is part of that run, which is why the tree pin spans the whole file.
Starting a genuinely new run means archiving the ledger together with
`output/<profile>/`, `FREEZE_FINAL_<profile>.txt`, `SELECTED_<profile>.txt`
and the profile's reports — an explicit operator action. The rehearsal's
earlier attempts are kept under `archive_rehearsal/` on the server.

**Restart rule.** A FAILED invocation is re-run exactly once, by re-executing
the same job script with the same env file — same seeds, same arms, same
tree, same `PYTHONHASHSEED=0`. A second failure stops the chain and prints the
escalation text. An invocation recorded SUCCESSFUL is never re-run. If the
watchdog already performed its own restart, the chain does not add a second
one. Nothing in the launcher can re-scope an invocation: arms and seeds come
from `chain_plan.py`, not from the failure.

**Resume.** Re-invoking the launcher skips any invocation whose ledger record
says `ok` **and** whose artefact still hashes to the recorded sha256. If the
artefact no longer matches, the chain STOPS (`reason=artifact_changed`) rather
than redoing it: a completed invocation that changed under us is an
escalation, not a retry.

**Shedding.** `--shed sensitivity[,adapt_freeze[,untrained]]`, an explicit
operator action, recorded in the ledger as `event=shed`. Nothing sheds
automatically. Shedding out of the plan's order is refused before anything
runs. None of the sheddable stages gates S7.

**Waiving a dependency.** `--waive-dep <STAGE>` lets a stage start although a
*hard* dependency has not completed. It exists because a stage can be
unrunnable for a reason outside this track — S5b runs the pre-registered
analysis, which cannot execute below 1400 ticks (REQUEST R11) — and the
alternative an operator reaches for at 3 a.m. is editing the ledger by hand,
which leaves no record. The waiver names the dependent stage, is written to
the ledger as `event=note reason=dep_waived`, and the launcher prints that
whatever the skipped stage would have established is **unproven** for that
chain. `--waive-order` remains the equivalent for the schedule edges.

**Watchdog.** Invocations whose modelled length reaches the profile's
`watchdog_min_hours` are launched through `~/MARL_watch/register_chain.sh`
(argument order: `NAME CWD "CMD" LOGFILE DONE_MARKER EXPECTED_HOURS [ENVFILE]
[STALL_MIN]`), so the existing watchdog can restart one once. The launcher
then polls the pidfile, **deregisters immediately** when the process exits,
and reads `~/MARL_watch/state.json` to record whether the watchdog had already
used the one restart. The done marker is printed by the wrapper only on exit 0
— never the trainer's own "TRAINING COMPLETE", which is printed before the
run is actually over.

**The artefact must be the one THIS attempt wrote.** Before every launch,
`chain_launch` removes the invocation's artefact and records the launch epoch;
the gate then requires the artefact to exist *and* to carry an mtime at or
after that epoch (2 s of grace for NFS timestamp rounding). Without both
halves a job that exits 0 having written nothing is verified against the
**previous attempt's** pickle, counts 28 present / 0 missing and is recorded
as a full success — the campaign's recurring defect, in the launcher itself.
`stage_train.sh` does the same for each replicate's `policy_final.pt`. Two
launcher tests cover it: a job that writes nothing with a valid stale artefact
present, and a job that writes a *complete* pickle with a back-dated mtime.

**An ablation must differ from the run it ablates.**
`tools/stage_report.py --ablation-diff` compares S10/S11 against the S6
pickles on the series the switch is supposed to move, counts the passes that
differ, and refuses to call a zero difference a pass. Three outcomes, all
recorded in the ledger as `ablation=`: `exercised`; **`NOT_EXERCISED`** —
identical *and* the declared baseline counter is zero, so the mechanism never
fired and there was nothing to switch off (valid data, no evidence);
`FAILED` — identical although the mechanism did run in the baseline, i.e. the
switch did not take. S11 declares `repoints` as that counter, S10 declares
none because the detection floor is always engaged in the baseline.

**Gates that count.** After every evaluation invocation,
`tools/stage_report.py --verify` loads the pickle and compares the
(arm, scenario, seed) keys that actually carry data against the expected list
from `chain_plan.py`. Zero keys, a missing key, an unexpected key, a wrong
`pythonhashseed` or a missing tree digest all fail. Before any analysis,
`--cross-provenance` (tree_head / tree_sha256 / schema_id / pythonhashseed
identical) and `--key-union` (union over the stage's pickles equals the
expected set exactly, no duplicates) must pass. Every one of these reports the
number of things it compared and fails when that number is zero.

## 4. REQUESTS to the other tracks

These are defects or gaps in files this track does not own. Each is worked
around in the launcher; the workaround is named.

**R1 — `run_timeline_comparison.py` demands a checkpoint that the invocation
never reads.** `main()` exits unless `MARL_EVAL_CHECKPOINT` exists, even when
every selected arm is policy-independent (`xldet`, `random` and the routing
arms all take branches that never load a policy). S1 runs *before* training,
so there is no trained checkpoint to name. *Workaround:* S1 carries an
untrained-init checkpoint (produced by S0 [B]) as a placeholder, recorded in
the pickle's provenance. *Request:* skip the existence check when no
policy-reading arm is selected, or accept `--no-checkpoint`.

**R2 — `make_untrained_checkpoints.py` cannot be told where to write.** It
writes `untrained_s<seed>/` into the current directory and falls back to
`~/MARL_run_6/ce_ck_s1234/policy_final.pt` for its key/shape reference.
Without a redirect a rehearsal would write into the path the real run uses.
*Workaround:* `stage_prep.sh` moves the result to
`output/<profile>/untrained_s<seed>/`. *Request:* add `--out-dir`.

**R3 — `confirmatory_analysis.py` has no completeness gate.** `load()` merges
pickles and the seed set is the *intersection* over arms, with the difference
reported as "harness failures"; a partial or mixed-provenance set is analysed
silently. COMPUTE_PLAN.md lists this gate as required work. *Workaround:*
`stage_analysis.sh` runs the cross-provenance and key-union gates first and
refuses to call the analysis otherwise. *Request:* move the gate inside the
analysis entry point so it cannot be bypassed at 3 a.m.

**R4 — `untrained_analysis.py` is pinned to the old study.** It reads
`output/untrained_s<seed>_80_99.pkl`, `~/MARL_run_6/output/cf_static_*.pkl`
and `~/MARL_run_5/output/cf_eval_80_99.pkl`, i.e. the 80–99 confirmatory set
and two frozen trees. It cannot run against the pre-registered confirmatory
block in this tree.
*Workaround:* S12 produces its own descriptive report and records the
untrained analysis as **not run**, with its exit code, rather than skipping it
silently. *Request:* parameterise its inputs.

**R5 — `MARL_STRESS_INTERFERENCE_TICK` does not scale with `--ticks`.** It is
an absolute tick, so a shortened run at the production value (2000) produces a
"stress" arm that never meets the interferer and is indistinguishable from the
clean arm. *Workaround:* the rehearsal profile scales it (333 at 600 ticks).
*Request:* scale it with the tick override as the severance and rescue ticks
already are, or refuse a value beyond the run length.

**R6 — `dump_endstate.py` has no tick override.** It calls `run_physics_pass`
directly, so it always runs a full-length pass, and `PROBE_TICK` (default
3400) is absolute. The end-state dump therefore costs the same in a reduced
rehearsal as in the real run. *Workaround:* none needed for correctness; the
cost is budgeted. *Request:* accept `--ticks` and scale `PROBE_TICK`.

**R7 — the "installation off" sensitivity has no switch.** RERUN_SPEC WP3
pre-registers "installation off (idealised timing)" as an evaluation-only
sensitivity, but no environment variable turns the installed view off. The
closest existing knob is `MARL_DETECT_TICKS=0`, which the driver describes as
"oracle-grade failure detection" — the same idea, and that is what S10 uses.
*Request:* confirm against the pre-registration text before the real run, or
add the intended switch. Until then S10 is **an assumption, recorded here.**
*Correction (2026-10-01):* the intended switch exists since WP3 —
`MARL_LS_INSTALL=0` (`sixg_sim/link_state_service.py`,
`docs/LINK_STATE_SERVICE.md` §7: detection and billing unchanged, installation
immediate). *Resolved (FIX-PREFREEZE item 1, D-FIX.1):* `chain_plan.py`
now sets `MARL_LS_INSTALL=0` on S10; it set `MARL_DETECT_TICKS=0` until
then, which removes the service altogether.

**R8 — the watchdog's restart does not restore `PYTHONHASHSEED`.**
`watchdog.py` rebuilds the command as `cd CWD && source ENVFILE && nohup nice
-n 19 CMD`; unlike `register_chain.sh` it does not export `PYTHONHASHSEED=0`
itself. A registered job without an env file would be restarted with cron's
environment, and an "unchanged re-run" would silently differ in exactly the
variable the A1-era sensitivity analysis was forced by. *Workaround:* every
invocation is registered WITH a complete env file. *Request (watchdog owner):*
export the pinned variables in the restart path too.

**R10 — nothing in the other tracks' scripts scales with `--ticks`.** See R5
and R6; both are instances of the same gap, and both were found by running the
chain at reduced scale rather than by reading.

**R11 — `confirmatory_analysis.py` holds ABSOLUTE tick windows and cannot run
below 1400 ticks.** `W = 500`, `PRE_SURGE = (900, 1400)` and
`T_EVENT = {'A': 200, 'B': 1400}` are constants, while
`run_timeline_comparison.py` SCALES the scenario ticks with `--ticks`
(`SEVERANCE_TICK = min(200, max(50, T//6))`,
`RESCUE_ARRIVAL_TICK = min(1400, max(SEV+50, T//2))`). Below 1400 ticks
`wmean(conn, 900, 1400)` averages an empty slice, the NaN reaches
`statistics.stdev` and Python 3.12 raises
`AttributeError: 'float' object has no attribute 'numerator'`. Between 1400
and 3600 ticks it does not crash — the windows simply point at the wrong phase
of the run and the report is quietly wrong, which is worse. *Workaround:* the
rehearsal profile carries a small **full-length** development subset
(`dev_full_seeds`, two seeds, every dev arm, both scenarios, 3600 ticks) and
`dev_analysis_prefixes` points S5b at it, so the pre-registered analysis is
executed on data it can actually analyse. *Request:* scale the windows with
the tick count, as the driver already does, or refuse a tick count the windows
do not fit.  *Since the trackB merge the analysis REFUSES the set* ("series
... has 600 entries, expected 3600 (run length)"), so on the merged tree a
reduced-scale S7 can only fail, by that refusal, before any statistic is
computed.  *Correction (2026-10-01):* the S7 failures of 2026-09-24 were on
the PRE-merge trees and were a different failure, which this paragraph used
to call the same one ("as it did on 2026-09-24"). There the analysis loaded
the 600-tick set, computed the descriptive block, the whole primary family and
the S1 secondary tests in memory, and then crashed at the first secondary test
that reads the
pre-surge window (`S2_B_change_vs_xldet`, `confirmatory_analysis.py:277`: the
NaN of `wmean(conn, 900, 1400)` reaching `statistics.stdev`) -- a crash, not
a refusal. It wrote no report and printed no statistic (§6; the logs are
quoted in `docs/SEED_AUDIT_*.md` §6).  *Workaround for S7 (WP13,
commit `0367856`; recorded in DECISIONS.md D13.6 only since FIX-PREFREEZE
item 7, and its two negative controls first ran at the FIX-PREFREEZE head,
D-FIX.11):* the rehearsal's S6 also carries a two-seed
**full-length** confirmatory subset (`cf_full_seeds` 220, 221 since
`31b034c`; all 8 arms,
both scenarios, 3600 ticks, one invocation `s6_cffull_b0` of 32 passes) and
`cf_analysis_prefixes` points S7 at it; the S10/S11 ablation gate compares
with the S6 pickles at the ablation's own tick count
(`chain_plan.py --artifacts S6 --same-ticks-as <id>`), so the full-length
subset never serves as the baseline of a 600-tick ablation.  The full-length
profiles analyse the whole of S6, as before.

**R9 — `~/MARL_watch/state.json` keys `first_seen` and `restarts` by chain
name for ever.** Reusing a name later makes `hours_run` meaningless and can
consume the restart budget of an unrelated job. *Workaround:* names are unique
per profile and invocation (`rc_<profile>_<inv>`), and entries are
deregistered on completion.

## 5. Deviations from COMPUTE_PLAN.md

1. **S8 is four invocations, not three.** `MARL_EVAL_CHECKPOINT` is a
   per-invocation variable, so the four untrained checkpoints cannot share an
   invocation; 4 × 40 passes is the finest split that keeps whole
   (arm, seed-block) boundaries, and it is still ≤ 64.
2. **The plan's "six invocations" for S1 is reproduced exactly** (5 dev blocks
   of 4 seeds × 7 arms × 2 = 56, plus one stress invocation of 60), but the
   dev blocks cross the 59→70 seed gap. Blocks are contiguous in the ordered
   seed list, not in seed arithmetic.
3. **S5 is automated as a technical gate.** The plan's human freeze-2 decision
   is unchanged; what the launcher automates is the mechanical part — key
   completeness, cross-provenance, manifest and checkpoint digests — sealed
   into `confirmatory/FREEZE_FINAL_<profile>.txt`. The operator still has to
   decide.
4. **The manifest is generated at S0**, which the plan calls a protocol
   strengthening requiring the author's explicit approval. It is generated
   into a profile-specific path and its digest is sealed at S5; if the author
   prefers the RERUN_SPEC default, run S0's manifest invocation at S5 instead
   (`--stages S0` late) — there is no schedule impact either way.
5. **The `real` profile does NOT apply the work package's 16/8 worker rule for
   a shared machine.** The work package says to keep `MARL_POOL_WORKERS` at 16
   while the 1-minute load is below 12 and at 8 otherwise.
   `preflight.recommend_workers` applies that rule to every profile *except*
   `real`, which launches evaluation at COMPUTE_PLAN.md's 24 whatever the
   load. This is a deliberate choice of the plan over the operating rule: the
   plan's 19.4 h headline, its cost model (16.88 cores obtained at 24 workers
   under typical tenant load, 18.26 on a quiet host) and its ranked action 1
   ("run every evaluation invocation at 24 instead of 8, ~22 h saved") are all
   stated at 24. The memory ceilings that *are* applied to `real` (16 below
   12 GB MemAvailable, 8 below 8 GB) are strictly weaker than the evaluation
   gate's own 18 GB precondition, so they can never fire; the real protection
   against sustained foreign load is the **shed order**, not a worker
   reduction. The lever is worth a factor of two — the same rehearsal S1
   invocation took 873 s at 16 workers with the host load at 7 and 1734 s at 8
   workers with it at 33–36 — so if the author prefers the operating rule,
   delete the `profile != 'real'` guard in `recommend_workers` and add ~2 h to
   the schedule. Rehearsal and test profiles *do* apply it, which is why the
   rehearsal timings below are recorded at 16 and 8.
6. **The rehearsal profile is not uniformly reduced.** Its development set
   carries an extra two-seed subset at the full 3600 ticks (see R11) because
   the pre-registered analysis cannot run at 600. That subset is the single
   largest item in the rehearsal's wall clock and it exists so the S5b → S6
   edge is executed rather than assumed.
7. **The rehearsal profile runs three training replicates, not two.** Plan
   requirement 5 — exactly two alive, the next launched the moment one exits —
   is only *executed* when there are more replicates than may run at once.
   With two seeds and `concurrent = 2` both launch back to back and the stage
   then gang-waits, which is precisely the behaviour the plan replaces, so the
   earlier rehearsal could not have proved it. `tests/test_run_chain.py`
   asserts `len(torch_seeds) > train.concurrent` for every profile.

## 6. Rehearsal record (2026-09-24, <compute-server>, `~/MARL_run_9d`, tree `79e15b2`)

The whole chain was run end to end at reduced scale on ONE tree — every one
of the 28 preflight records carries the same `tree_sha256`
`fe8eaa471027b6dd…`. **Every number under `output/rehearsal/` is throwaway**;
600-tick passes and 4 confirmatory seeds cannot support any claim. The one
part that is *not* reduced is the two-seed **full-length development subset**
(R11), which exists so the pre-registered analysis can be executed at all.

Reproduce the table with
`python3 tools/stage_report.py --timings confirmatory/chain_rehearsal.jsonl --profile rehearsal --run all`;
`--run last` gives the final segment alone (7 invocations, 0.50 h) and states
the 2.98 h it excluded.

| stage | inv | passes | wall | notes |
|---|---|---|---|---|
| S0 | 2 | 0 | 24 s | manifest + untrained checkpoints |
| S1 | 3 | 96 | 4501 s | of which `s1_devfull_b0` **3145 s** (28 passes at 3600 ticks) |
| S2 | 3 | 0 | 1005 s | 3 replicates, 2 alive; **one recorded 300 s wave delay** |
| S3 | 3 | 12 | 722 s | one invocation per replicate |
| S3b | 1 | 0 | 3 s | selection rule — **NOT EXERCISED**, see below |
| S4 | 4 | 16 | 2637 s | `s4_devfull_static` 1444 s, `s4_endstate` 693 s |
| S5 | 1 | 0 | 30 s | dev gate + FREEZE_FINAL seal |
| S5b | 1 | 0 | 22 s | **the analysis ran**: 9 primary tests, 12 secondary |
| S6 | 1 | 64 | 1744 s | 8 arms, key-union 64/64 |
| S7 | 2 | 0 | 35 s | **FAILED twice**, chain stopped — see below |
| S8 | 3 | 24 | 709 s | untrained control |
| S9 | 1 | 8 | 513 s | adapt-then-freeze |
| S10 | 1 | 8 | 273 s | ablation **exercised** (8/8 passes differ on 5/5 series) |
| S11 | 1 | 8 | 252 s | ablation **NOT EXERCISED** (0/8 differ) |
| S12 | 1 | 0 | 52 s | 48 passes over 6 pickles |
| **TOTAL** | **28** | **236** | **12 522 s = 3.48 h** | 16 or 8 workers per the load rule |

### What the rehearsal executed, on the real code

The manifest and untrained checkpoints; the placeholder-checkpoint workaround
for S1; load-based 16/8 worker selection; the sha256 ledger and the
artefact-freshness gate on every invocation; the tree pin (28 records, one
digest); watchdog registration, polling and deregistration; the exclusive
lock; **three** training replicates with at most two alive and
`--eval-episodes 0 --baseline-episodes 0`; the **wave delay** (see below); the
dev gate and the FREEZE_FINAL seal; **S5b — the pre-registered analysis, on
the real script, producing a real report**; the confirmatory 8-arm invocation
with its cross-provenance and key-union gates; the S7 escalation; the
untrained control and the adapt-then-freeze arm; the **ablation gate** on both
sensitivities; the secondary descriptive report with `untrained_analysis.py`
recorded as NOT RUN (exit 90); and the recorded operator waivers
(`--waive-order` four times, for S8–S11 after S7).

### What it did NOT establish, and is recorded as such

**S7 is rehearsed in structure only.** Its gates ran and passed
(cross-provenance; key-union 64/64) and `confirmatory_analysis.py` then died
with `AttributeError: 'float' object has no attribute 'numerator'` at
`confirmatory_analysis.py:277`, exactly as R11 predicts for a 600-tick set.
The launcher did the right thing: one unchanged re-run, a second failure, and
the escalation text without analysing a partial set. `rehearsal_cf_report.json`
does not exist and no claim rests on it. The primary analysis can only be
exercised at full length, which is why **S5b** now runs the same script on the
full-length development subset first — and it passed, so the S5b → S6 edge is
executed, not assumed.

**S11 is rehearsed in structure only.** Its 8 passes are valid and complete,
and they are byte-identical to their S6 baseline on all four declared series:
at 600 ticks the `repoints` counter totals **0 in the baseline as well**, so
fragment-aware re-pointing never fired and the ablation had nothing to switch
off. The chain records `ablation=NOT_EXERCISED` and says so out loud. S10, by
contrast, differs on 8/8 passes and all five declared series.

**The selection rule is NOT EXERCISED.** With the window now scaled
(last 83 of 600 ticks, i.e. 517–600, after the rescue arrival at 300 —
previously it was ticks 100–600 and straddled the severance at 100) all three
replicates still score components exactly `4.000000`, so the primary criterion
again had zero discriminating power and the tie-break alone chose the
checkpoint. `stage_select.sh` now prints that and writes it into
`SELECTED_rehearsal.txt`. The window fix was still necessary; it is simply not
what made the rule degenerate at this scale.

**"Start the next replicate the moment ONE exits" is only partly shown.**
With three seeds and `concurrent = 2` the `while ALIVE >= CONCURRENT` loop was
entered for the first time (with two seeds it never is), but replicates 1 and
2 finished one second apart, so it released at *zero* alive, not at one:
`confirmatory/preflight_rehearsal/s2_train_s3456.160743.json` records
`expect_inv []`. The wave-level memory gate *was* exercised with a mixed set —
replicate 2's preflight records `expect_inv ['s2_train_s1234']`,
`our_expected_pss_gb 0.5`, raw 27.3 GB → wave figure 27.8 GB. At real scale
the replicates are 3.5 h with a historical 5–10 min within-wave spread, so the
release at one-alive will matter there.

### Defects the rehearsal exposed in the launcher

1. **A training wave was abandoned, not delayed.** `MemAvailable` was
   25.93 GB against the 26 GB wave gate — 70 MB short, for one sample — and
   `stage_train.sh` had *no retry at all*, while `run_final.sh`'s evaluation
   path had one. The stage stopped after two of three replicates and the whole
   chain with it. COMPUTE_PLAN.md says to **delay** a wave, not abandon it.
   Fixed: the stage now retries (`--preflight-retries` / `--preflight-wait`,
   recorded as `reason=wave_delayed`), an exit-2 abort is still never retried,
   and preflight additionally implements the plan's other wave precondition —
   refuse to *start* a wave above ~20 cores of tenant load. In the rerun this
   fired for real: one 300 s delay, then the wave started and all three
   replicates completed.
2. **The timing report averaged two tick scales together.** With a
   full-length subset in the profile, `--timings` mixed 600-tick and
   3600-tick passes into one mean and then extrapolated the inflated figure
   again (`marl_static` came out at 227 CPU-s "at 600 ticks" when its 600-tick
   passes alone cost 180). Fixed: per-pass costs are bucketed by the tick
   count of the invocation that produced them, an arm run at two tick counts
   pins its own affine intercept, and an arm run at full length is reported as
   **measured** rather than extrapolated.
3. **The tree pin restarted on a resume.** It was scoped to the last
   `chain_start`, so a code change made between a stop and a resume would not
   have been caught by the pin — only by the cross-provenance gate hours
   later. Fixed: the pin spans the whole ledger, which is one run of the
   chain; starting a new run means archiving the ledger, an explicit operator
   action (`archive_rehearsal/` on the server holds runs 1 and 2).

### Measured per-pass cost, and what it says about the plan

Eight of the nine arms were run at **both** 600 and 3600 ticks in this
ledger, so their 3600-tick figures are measured, not extrapolated, and each
arm pins its own affine intercept. All of it was measured with 8–16 pool
workers on a shared host at load 20–40; COMPUTE_PLAN.md's own anchors are
**1-worker** figures at load 22–54, so pool contention inflates these numbers
and the absolute comparison to the plan is not like for like. The arm-to-arm
ratios and the wall-clock extrapolation are the parts that carry over.

| arm | n/mean @600 | n/mean @3600 | plan 3600 | ratio |
|---|---|---|---|---|
| marl_static | 72 / 179.6 | 4 / **1085.5** | 1369 (measured) | 0.79 |
| xldet | 20 / 234.5 | 4 / **1470.3** | 1619 (measured) | 0.91 |
| ospf | 20 / 174.4 | 4 / **1117.0** | 642 (measured) | 1.74 |
| aodv | 16 / 172.1 | 4 / **1156.6** | 650 (assumed) | 1.78 |
| olsr | 16 / 177.1 | 4 / **1161.8** | 650 (assumed) | 1.79 |
| batman | 16 / 192.4 | 4 / **1247.2** | 650 (assumed) | 1.92 |
| sdn | 20 / 207.7 | 4 / **1395.7** | 650 (assumed) | 2.15 |
| random | 16 / 238.9 | 4 / **1551.2** | 1369 (assumed) | 1.13 |
| marl_freeze | 8 / 224.9 | — | 2054 (assumed) | 0.70 (affine) |

**The plan's largest stated soft spot is confirmed, in its relative form.**
"sdn/olsr/batman/aodv ≈ ospf" holds against *this run's own* ospf: aodv
1.04×, olsr 1.04×, batman 1.12×, sdn 1.25×. What does not hold is the
absolute level: every routing arm, ospf included, costs 1.7–2.2× the plan's
figure here, and `random` and `marl_freeze` are the two arms the plan guessed
in the wrong direction by more than 10 %.

**Extrapolated to the real 1056-pass workload** (cost model
`wall = CPU-s / (cores_obtained × 3600)`):

| | typical tenant load (16.88 cores) | quiet host (18.26 cores) |
|---|---|---|
| evaluation stages at 24 workers | **21.31 h** | **19.70 h** |
| + training (4 replicates, 2 waves) | 6.96 h | 6.96 h |
| + analysis, inter-stage, human gate | ~1.0 h | ~1.0 h |
| **end to end** | **~29.3 h** | **~27.7 h** |
| of which, to the PRIMARY result (S7) | **~24.0 h** | **~22.8 h** |

Per stage at 16.88 cores: S1 7.30, S3 1.00, S4 1.07, S6 6.70, S8 2.86,
S9 0.95, S10 0.71, S11 0.71.

**Verdict: the 19.4 h headline does not hold on today's code.** It never
claimed to — COMPUTE_PLAN.md states plainly that "the WP14 perf pack is
load-bearing for the target" and gives 28.8 h of compute / 29.6 h end to end
as the pre-WP14 baseline. What this rehearsal adds is that **today's tree sits
on that pre-WP14 baseline almost exactly** (measured ~29.3 h against the
plan's 29.6 h) and shows none of the pack's assumed reductions: the plan
budgets marl-family −35 %, xldet −60 % and routing/SDN −50 %, and the measured
figures are marl_static −21 %, xldet −9 % and the routing family *dearer* than
the pre-WP14 anchor. Until WP14 lands and is measured at full length, the
schedule to plan for is ~29 h end to end with the primary result at ~T+24 h,
not 19.4 h with the primary at T+14.7 h.

**What buys the window back.** Shedding in the plan's order removes
S10+S11 (1.42 h), then S9 (0.95 h), then S8 (2.86 h) — 5.23 h in total,
bringing the end to end to ~24.1 h typical / ~22.5 h quiet. None of it gates
the primary analysis, and the primary result is unaffected at ~T+24 h either
way. The plan's own advice stands and is now quantified: S1 runs first
precisely so this can be re-measured at T+7 h on the real batch, while there
is still time to shed.

---

## 7. R3-lite (profiles `r3lite` and `r3lite_rehearsal`)

Added 2026-10-11 (`docs/DECISIONS.md` D-R3.1), revised the same day after
the review of the build (D-R3.2), amended for the author's decisions of
D-R3.3 — a block of forty instances and a second confirmatory family — and
revised on 2026-10-12 after the reviews of that amendment (D-R3.4).
Pre-registration: `docs/PREREGISTRATION_R3.md`. **Not frozen and not
launched by the work packages that wrote it; nothing has run on its
confirmatory block.**

R3-lite is a new confirmatory test of EXISTING policies of R2's training
run on a fresh block of topology seeds: replicate torch seed 1234 (family
1, the primary family) and the ensemble of the four replicates (family E).
It reuses this launcher unchanged in everything it guarantees (dependency
order, preflight, the clean environment, the restart rule, resume, the
ledger; 64-pass invocations for every multi-arm block) and differs from the
R2 profiles in five ways.

**1. Its own stage graph.** `chain_plan.build()` dispatches on a profile's
`graph` field; only the two R3-lite profiles carry one, and the four R2
profiles are bit-for-bit what they were at `6b53f3f` (their `build()` output,
`--emit` blocks and profile dictionaries are pinned by sha256 in
`tests/test_r3lite_plan.py`). `--stages`, `--stage-deps`, `--shed-group-of`
and the new `--graph`, `--dropped`, `--secondary-stages`, `--shed-order`,
`--imports`, `--selected`, `--audit-gate` read the graph of the profile.

| stage | invocations | what it does |
|---|---|---|
| S0 | `s0_manifest`, `s0_import` | the block's manifest; **import** of five R2 files (four `policy_final.pt`, one untrained `policy_init.pt`) out of `~/MARL_run_9/output/final/` (read-only), each accepted only with the sha256 pinned in `chain_plan.py` (`stage_prep.sh`, step `import`: every source verified before anything is copied; an existing destination with another digest is refused, never overwritten) |
| S3b | `s3b_record` | selection RECORD (`stage_record.sh`, step `select`): the tracked `confirmatory/SELECTED_<profile>.txt`, the plan's pin and the checkpoint on disk name the same replicate with the same sha256; seven checks; **computes nothing** |
| S5 | `s5_freeze2` | FREEZE_FINAL (`stage_record.sh`, step `freeze2`): manifest = exactly the plan's seeds in both scenario kinds; the five imported digests; the selection record; `confirmatory_analysis.py` md5; the engine pin (62 files byte-identical to `6b53f3f`); then seals the tree digest with the manifest and checkpoint digests. No development gate: there are no development pickles |
| S6 | `s6_cf_b0..9` | R2's confirmatory stage, twice: the same 8 arms in the same order, the same 4-seed blocks, environment and script (each block compared field by field with profile `real` in the tests). The data of family 1 and the comparator passes of both families. **Gated by the freeze-2 seed audit** (item 4) |
| S9x | `s9x_ensemble` | **new: the confirmatory passes of family E**, right after S6 and before any analysis: `marl_static` on the ensemble of the four R2 replicates, all forty instances, one pickle (`ckpt_kind` `ensemble`: `MARL_EVAL_CHECKPOINT` is the comma-separated list of the four imported files; `chain_ckpt_abs` in `chain_lib.sh` makes each member absolute, and one path is written exactly as before). Named S9x, not S9e: S9a–S9f are the endpoints of the analysis's stability family, and a test pins that no chain stage or invocation carries an endpoint's name |
| S7 | `s7_analysis` | family 1: the unchanged `confirmatory_analysis.py --marl-arm marl_static`, after the md5 pin. Ordered after S9x (item 5) |
| S7e | `s7e_family_e` | **new: family E** (`stage_analysis.sh`, step `family_e`; item 5): the same analysis on a pickle set assembled by `tools/assemble_family.py` |
| S6h | `s6h_hash1` | hash-seed repeat on the two lowest seeds of the block (replicate 1234) |
| S8 | `s8_untrained_s1234` | untrained control, the selected replicate's initial actor only, all forty instances, one pickle |
| S12 | `s12_secondary` | analysis S8 (hash), analysis S12 (the one untrained control), the per-instance listing of the ensemble against the XL-DET passes of S6 (`tools/stage_report.py --paired-descriptive`; descriptive), the stability family of the primary report — all six endpoints present, **each one's status and verdict copied into the secondary report** (`S9_stability_family.endpoints`); `not recorded` or an entry without a valid status fails the stage, `not assessable` is recorded as the outcome it is — and **family E**: its report and its record must exist, the record must name this primary report and this family-E report by sha256 and carry a passed assembly control, and its six stability endpoints are copied the same way (`analyses.family_E`) |

**2. What R2 ran and R3-lite does not is DROPPED BY THE PLAN, with a
recorded reason** (`chain_plan.DROPPED_R3LITE`: S1, S2, S3, S4, S5b, S9,
S10, S11). `run_final.sh` writes one `DROPPED <stage>` line and one
`event=dropped` ledger record per stage at every chain start;
`stage_analysis.sh` names them, records secondary S7 as `dropped` (not as a
missing input, and not as shed), and writes `dropped_by_plan` into the
secondary report. Nothing can be shed: the profile declares no shed group
and `--shed` is refused for it before the chain starts.

**3. The checkpoints are pinned, not produced.** Every evaluation invocation
of an R3-lite profile carries `ckpt_sha256` (emitted as `INV_CKPT_SHA256`,
empty for every R2 invocation). Before the launch `run_final.sh` hashes the
checkpoint file(s) the invocation resolved and stops on a difference
(`reason=checkpoint_sha256_mismatch`); after the run the completeness gate
requires the pickle to RECORD the pinned digest(s) (`--verify
--expect-checkpoint-sha256`). `tools/preflight.py` treats `r3lite` like
`real` for the worker count (24, no shared-machine load rule).

**4. The freeze-2 seed audit is a gate of the launcher.** PREREGISTRATION_R3
§13 makes it a stop condition before S6. It used to live in the checklist
only: `run_final.sh --profile r3lite` without `--stages`/`--from` ran S0 to
S12 straight through and read no audit report. Now profile `r3lite` declares
the gate in the plan (`audit_gate` `r3-freeze2`, `audit_before` `S6`;
`chain_plan.audit_gate`, `--audit-gate`; the builder refuses the profile
without it), and the launcher evaluates it once per launch, before the first
stage at or after S6 that the launch would run — so `--from S6h` or
`--stages S7` cannot step around it. `tools/stage_report.py --audit-gate`
makes nine counted checks on the report the committed driver wrote
(`output/seed_audit_<lo>_<hi>_r3-freeze2.json`): freeze 2 is sealed; the
report parses; it audited the plan's block; the result-data gate was
requested and passed; coverage was complete; its planning markers name this
run's S0 manifest (so it is the freeze-2 audit and not freeze 1's); it
started after the seal; on the sealed head; from this tree.

| report | launcher |
|---|---|
| all nine checks pass and the verdict is clean (audit exit 0) | `[AUDIT-GATE] PASS`, ledger `event=audit_gate status=pass` with the report's sha256; the stage starts |
| all nine pass, but text, file-name or history hits remain (audit exit 3) | `STOP … reason=audit_unclassified`. After the author has classified every hit in writing (PREREGISTRATION_R3 §13.1): `run_final.sh --profile r3lite --audit-ok` records a ledger note bound to that report's sha256; the next launch prints `[AUDIT-GATE] CLASSIFIED` and goes on. A report that changes afterwards is unclassified again |
| any check fails: no report, another block or tree or head, made before the seal, a result-data hit, refused coverage | `STOP … reason=audit_gate_failed`. `--audit-ok` is refused for it |

The R2 profiles and `r3lite_rehearsal` declare no gate (the rehearsal's
seeds are development seeds, which are in result data by design, so no
audit of them can pass); the gate is exercised by the suite on a scratch
chain whose plan gives the rehearsal profile `r3lite`'s gate. Do not re-run
the freeze-2 audit after S6 has started: the report sits at a fixed path,
and a later run would — correctly — find this run's own pickles.

**5. Two confirmatory families on one block (D-R3.3).** The analysis script
tests one learned arm, `marl_static`, and may not change. Family E is
therefore analysed on an **assembled pickle set**: `tools/assemble_family.py`
writes, for each S6 pickle, a derived pickle in which every comparator pass
is the S6 pass and every `marl_static` pass is the ensemble's pass of the
same instance. It verifies every run of the derived set from disk against
its input (a canonical digest of the run) and refuses, leaving nothing
behind: a seed without an ensemble pass in either scenario, a seed twice or
outside the block, a source pickle that does not record exactly the pinned
ordered checkpoint list, any provenance difference between the inputs (tree
head and digest, schema, reward id, hash seed, pass length, library
versions, engine environment; the invocation name, the checkpoint path and
the worker count may differ), and a pass whose offered-demand series is not
XL-DET's on that instance. Chain S7e (`stage_analysis.sh`, step `family_e`)
fixes the order: (1) the S7 report must exist, must be the file whose
sha256 the ledger recorded for S7, and every S9x invocation must have
ended before the successful attempt of S7 started — read from the ledger
(`stage_report.py --ledger-order`), after which the report's sha256 is
written to the ledger before anything of family E is computed; (2)
**positive control** — the set assembled from the S6 pickles alone
(replicate 1234's pin) must give, through the same analysis, the S7 report
leaf by leaf, in both directions and by count (leaves compared, the same
number each way); (3) family E is assembled (the four pins, in order),
gated (provenance, key union) and analysed, strict, with the same manifest.
It writes `confirmatory/<prefix>cf_report_family_e.json`, the two assembly
records, `…family_e_ledger_order.json` and `…family_e_record.json`, which
names both reports by sha256 and carries the order as the ledger shows it.
The step removes its own outputs when it starts and, unless it reaches its
last line, when it ends: a failed S7e leaves no family-E report, text,
record or assembled pickle. The stage graph makes the ensemble's passes
confirmatory DATA: S9x depends on the seal only and runs right after S6;
S7 is ORDERED after S9x, not dependent on it; S7e depends on S7 and S9x;
S6h and S8 are ordered after S7; S12 depends on S7, S6h and S8 and is
ORDERED after S7e. Nothing family 1 needs depends on family E, so a family
E that cannot be completed leaves family 1 analysable and reportable to
the end.

**Operator actions of the R3-lite graph** (`chain_plan.ORDER_WAIVERS_R3LITE`,
profile fields `no_invalidate_from`, `declared_waivers_only`; refused before
the chain starts, each refusal a ledger `note` with
`reason=operator_action_refused`):

| action | profile `r3lite` | `r3lite_rehearsal` |
|---|---|---|
| `--shed` | refused | refused |
| `--invalidate <inv>` | refused for every invocation of S6 or a later stage; and for every invocation once an invocation of S6 has been started | allowed |
| `--max-attempts N` | refused unless N = 2 | any positive integer |
| `--waive-dep` | refused | allowed |
| `--waive-order S7` (edge S7 after S9x) | honoured only after a terminal failure of S9x | the same |
| `--waive-order S12` (edge S12 after S7e) | honoured only after a terminal failure of S7e or S9x | the same |
| `--waive-order` on any other edge | refused | allowed |

A **terminal failure** is what `stage_report.py --terminal-failure LEDGER
--profile P --stage ST` reports with exit 0: an invocation of the stage
whose last attempt is closed and failed, and which either has an earlier
attempt that did not succeed (it failed, or it was started and never
ended) or already contains the watchdog's unchanged re-run in that last
attempt. A `stop` line alone, one failed attempt, an attempt that is still
open, or a failure that was invalidated or followed by a success is not
one. **The launcher reads the same state** (`--restart-state LEDGER --inv
ID`: attempts started, how the last ended, terminal or not) before it
starts an invocation: it continues the attempt count from the ledger, never
starts an invocation the ledger calls terminal, and, when the budget is
used and the last attempt has a start and no end, closes that attempt as
failed (`exit=killed`, `closed_by=launcher`) once a preflight shows no job
of ours alive. If the plan cannot say how a waiver is treated, nothing is
waived. The contingency of PREREGISTRATION_R3 §13 is
therefore `--stages S7,S6h,S8,S12 --waive-order S7,S12` after a failed S9x
and `--stages S6h,S8,S12 --waive-order S12` after a failed S7e; S12 reads
the same state from the ledger, records family E as `not_completed` with
the cause, computes no ensemble listing and ends `rc=0`. In every other
state a missing ensemble pickle or family-E report fails S12, and with a
complete family E S12 also re-hashes the S6 and S9x pickles against the
assembly record and the ledger and cross-checks the listing with E-P1 and
E-P4. The two single-arm invocations
(`s9x_ensemble`, `s8_untrained_s1234`) hold the whole block in one pickle,
80 passes each (`chain_plan.R3_SINGLE_ARM_MAX_PASSES`): the analysis reads
one pickle per control and the assembler one ensemble pickle; a test pins
that exactly these two exceed 64.

| profile | seeds | ticks | checkpoints | eval workers | output |
|---|---|---|---|---|---|
| `r3lite` | the R3 confirmatory block (`chain_plan.R3_CONFIRMATORY_SEEDS`); the builder refuses any other set | 3600 | imported, sha256-pinned | 24 | `output/r3lite/`, `confirmatory/r3lite_*` |
| `r3lite_rehearsal` | development seeds 52, 53 (+ a full-length subset on the same two); the builder refuses any seed of a reserved block | 600, subsets 3600 | the same five files, imported into its own directory | 24 capped by load | `output/r3lite_rehearsal/`, `confirmatory/r3lite_rehearsal_*`, its own `SELECTED_r3lite_rehearsal.txt` |

Workload of `r3lite`: 20 invocations, 804 passes (S6 640, S9x 80, S6h 4, S8
80). Expected wall clock from R2's measured invocations at 24 workers on the
shared host (ledger `~/MARL_run_9/confirmatory/chain_real.jsonl`): S6 10 ×
1.5–1.75 h = 15–17.5 h, S9x about 2.5–3 h (Phase 0: 1608 CPU-s per ensemble
pass against 1369 for one actor), the two analyses with the two assemblies
about 0.5 h, S6h 0.5 h, S8 about 2.5 h, S12 0.2 h — **about 21–24 h end to
end, both analyses about 19–20 h after S6 starts**. The rehearsal: 14
invocations, 84 passes, 1.7 h of modelled compute, 2–3 h on the shared host.

**Launching.** The commands, in order, with what must be true after each,
are the R3-lite section of `docs/FREEZE_CHECKLIST.md` (R.0–R.6). Three
things in it are easy to get wrong and are fatal to a launch:

- **The launcher needs its own environment.** `tools/preflight.py` checks
  `PYTHONHASHSEED` (exactly 0) and `OMP/MKL/OPENBLAS_NUM_THREADS` (exactly
  1) in the process that runs it. A non-interactive ssh shell on <compute-server>
  has none of them, so a bare `nohup bash confirmatory/run_final.sh …`
  fails its first preflight, permanently, and is retried for the whole
  preflight budget. Every launch — the rehearsal and both segments — goes
  through `~/MARL_watch/register_chain.sh` with
  `logs/freeze/r3lite_chain.env` (R.2.1), which also gives the unattended
  segments a top-level watchdog entry, and with `--preflight-retries 12
  --preflight-wait 600`. The launcher tests cannot catch a missing
  environment (they set the variables themselves), so the checklist text is
  pinned by a test instead.
- **A failed first preflight, and a dry run, both pin the tree.**
  `chain_preflight` writes the tree digest to the ledger whether the
  preflight passed or not, and `run_final.sh --dry-run` writes a
  `chain_start` record and preflight records like a real start. Either
  fixes the tree digest for that profile's ledger: a later sync of any
  tracked file (documents included) then makes every preflight refuse
  (`tree_pinned`). **Do not use `--dry-run` in the run tree** to look at the
  plan; `python3 confirmatory/chain_plan.py --profile <p> --summary` shows
  it and writes nothing. To start a profile again at another head, archive
  its ledger, env and preflight directories and outputs first (R.2.3 does
  this for the rehearsal).
- **Two segments, one ledger.** `--stages S0,S3b,S5`, then the freeze-2
  audit, then `--from S6`. A single launch without `--stages`/`--from`
  stops at S6 on the audit gate.

```bash
cd ~/<run tree>
python3 confirmatory/chain_plan.py --profile r3lite --summary            # the plan, no compute, writes nothing
python3 confirmatory/chain_plan.py --profile r3lite_rehearsal --summary
python3 confirmatory/chain_plan.py --engine-pin-check                    # 62 files against 6b53f3f
python3 confirmatory/chain_plan.py --profile r3lite --audit-gate         # the gated stage, the gate, the report path
# launches: docs/FREEZE_CHECKLIST.md R.2.3 (rehearsal), R.4 (segment 1), R.6 (segment 2) -- never bare
```

**Seed audit.** `tools/run_seed_audit.sh LO HI` always took the block as a
parameter (default R2's). For the R3 block it is run with
`SEED_AUDIT_GATE=r3-freeze1` (before S0) or `r3-freeze2` (before S6), which
adds `--gate-result-data` (exit 5 on any seed of the block in a pickle, a
non-planning JSON file or a checkpoint), explicit `--planning-marker` paths
for the plan, its tests and the planning documents — at freeze 2 also this
run's own S0 manifest, whose path the driver reads from the plan — and the
**five pre-registered classification patterns** of PREREGISTRATION_R3 §13.1
(`--false-positive`; four coincidences, where the number in the window is a
tick count or a loop bound, and one negative statement). The driver reads
exactly five and refuses to run with another count; a line they match
leaves the verdict and is listed with its path and line in the log and the
report. They were written from a pre-audit of 2026-10-11 (result-data gate
PASS, 15 benign text lines); nothing is added to them after freeze 1. The
driver refuses the gate for any block other than the one profile `r3lite`
declares, and includes every `~/MARL_r3*` tree among the roots.

**Tests of family E** (`tests/test_r3lite_family.py`, 12 tests): the
canonical run digest; the assembler on synthetic pickles — what it builds
and records, the control set, each of its refusals (fourteen altered inputs
and seven further cases, each leaving the output directory empty), and a
tampered read-back; chain S7e driven through the stage script with the REAL
assembler and stubs for the analysis and the gate tools — the order of the
step, the arguments of every call, the derived pickles compared run by run
with their inputs, and thirty-five stops (preconditions, the order read
from the ledger, the control by count in both directions, the step's own
gates, outputs left by an earlier run), in each of which nothing of family
E is assembled or reported and no text of its analysis reaches the log;
and a negative control of the scipy precondition.

**Tests** (`tests/test_r3lite_plan.py`, 39 tests, each with a negative
control): the block and the reserved blocks; R2's confirmatory stage; the
key set; dropped stages; the pins (plan, SELECTED files, import, gate);
artefact collisions across all six profiles; the R2 profiles' digests; the
analysis md5 and the stage names against the endpoint names; the engine
pin; the import, record and freeze steps run for real in a scratch tree
against the real R2 files; in a scratch copy of the chain, for the rehearsal
profile only: a launcher dry run of the whole graph, segment 1 (S0, S3b, S5)
run for real through the launcher and resumed, the launcher's
checkpoint-digest stop driven for real with the evaluation script replaced
by a stub that never starts the harness, and **the launcher's audit gate
driven for real** (the scratch plan gives the rehearsal profile `r3lite`'s
gate; twelve cases on synthetic reports, with no evaluation invocation
reachable); `--audit-gate` on synthetic reports with thirteen negative
controls; S7 and S12 with the tools stubbed, including the status of each
stability endpoint; the audit gate and the classification patterns on a
synthetic window; and two tests of the documents the freeze is run from
(every launch in the checklist goes through the watchdog with the launcher
environment and the long budget; the pre-registration carries the
disclosures the review required). No test runs an arm on any seed (segment
1 builds the manifest of the two development seeds, a few seconds of
topology analysis), and none writes a file naming a seed of the R3 block.

**What the tests do not cover, and only the rehearsal does**
(FREEZE_CHECKLIST R.2): the unchanged analysis with ONE `--untrained`
control and no adapt-then-freeze pickle (R2 always had four controls),
`--s8`, the reproduction of the primary report by `compare_reports.py`,
chain S7e on full-length pickles (the assembly control reproducing the S7
report through the real analysis, and the analysis of the assembled
family-E set), `--paired-descriptive` on full-length pickles, and the
launcher's post-run `--verify … --expect-checkpoint-sha256` on a real
pickle. The suite stubs all of them. One of them has run on real pickles
outside a chain: on 2026-10-11 the set assembled from R2's development
comparator pickles and Phase 0's `ens_dev.pkl` (280 comparator runs and 40
ensemble runs verified, 40 offered-demand pairs) passed the unchanged
analysis in strict mode and gave Phase 0's ensemble means. The rehearsal's
acceptance is by count (R.2.4), and R2's rehearsal needed several
iterations.

**Per-root floors of the audit.** `tools/seed_audit_floors.txt` names no
`MARL_r3*` root. Freeze 1 therefore runs with the general per-root floor
and prints the count of every `MARL_r3*` root it scanned; freeze 2 passes
those counts back as `SEED_AUDIT_MIN_ROOTS="<basename>=<count> …"`, one
entry per scanned `MARL_r3*` root (the checklist's R.3 prints the line).

**The run tree.** `~/MARL_r3` (2026-10-11, head `7c941be`) fails five tests
of the full suite, 523 passed and 5 failed, all five being whole-tree scans
of WP7, WP8 and WP-DET that find R3 Phase 0's untracked copy of the
simulator under `r3out/orc/pilot/sixg_sim/` (a second `DISCRETE_HEADS`,
deleted-head references, a second reward cadence and episode length, a
third `as_completed` loop in its `main.py`). The same three files pass in a
copy of the head that holds the tracked files only. None of the five is an
R3-lite test and none is caused by an R3-lite change, but freeze 1 needs
`0 failed`. **Decided (D-R3.3 item 5): the run tree is a fresh one,
`~/MARL_r3lite`**, which holds one copy of the simulator and only what the
chain writes; `~/MARL_r3` keeps the Phase 0 material and is scanned by the
audit as a root of its own (`docs/FREEZE_CHECKLIST.md` R.0.2).
