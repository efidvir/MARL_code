# Compute plan — final corrected rerun (skeleton)

Status: skeleton written by the documentation track (WPDOC, 2026-09-22).
The dependency graph, pass counts and the rules below are fixed by
`docs/RERUN_SPEC.md` §15 and `docs/PREREGISTRATION_R2.md`. Every cost figure
marked ⟨perf⟩ is to be measured and filled in by the performance track
after its engine changes land; the WP0 measurements (D0.9, D0.11 in
`docs/DECISIONS.md`) are the only numbers available today and are quoted as
such.

Host rules (unchanged): <compute-server>, `nice -n 19`, `PYTHONHASHSEED=0`,
`MARL_POOL_WORKERS=8` at most for **all** project compute together (the cap
binds every use, D0.9), detached runs under `nohup` with a log file, no run
from the laptop. The machine is shared: check `uptime` before every launch.

---------------------------------------------------------------------------

## 1. Stages and what each one needs

```
 F1  freeze 1 ─────────────────────────────────────────────────────────────┐
     code + PREREGISTRATION_R2 + FREEZE_R2.sha256 + schema id             │
     needs: WP1-WP13 done, WP9 validated (reproduces A1 reports),           │
            WP12 decisions written, constancy dump (final actor list),      │
            SEED_AUDIT_200_219 clean                                       │
        │                                                                   │
        ├──► T   training, 4 replicates (torch 1234/2345/3456/4567)         │
        │       needs: F1 only                                              │
        │        │                                                          │
        │        └──► V   validation eval, seeds 60-66, marl_static,        │
        │                 one pickle per replicate (as each replicate ends) │
        │                  │                                                │
        │                  └──► SEL  selection (fixed rule) → checkpoint    │
        │                            sha256 appended to the record          │
        │                             │                                     │
        ├──► U   untrained controls (make_untrained_checkpoints.py)         │
        │       needs: F1 only (network definition, training init path)    │
        │                             │                                     │
        ├──► D-base  development set, seeds 50-59 + 70-79, arms that need   │
        │            NO policy: xldet, sdn, ospf, olsr, batman, aodv,       │
        │            random                                                 │
        │       needs: F1 only                                              │
        │                             │                                     │
        │            D-marl  development set, arms that need the policy:    │
        │                    marl_static, marl_freeze (own invocation),     │
        │                    untrained (needs U), sensitivities             │
        │       needs: SEL (marl_static, marl_freeze, sensitivities), U     │
        │                             │                                     │
        │            D-gate  technical checklist on D-base + D-marl          │
        │                    (abort → fix → new F1 → back to T)              │
        │                             │                                     │
        ├──► M   manifest 200-219 from the frozen tree (build_benchmark_    │
        │       manifest.py --seeds 200-219), digest appended = freeze 2    │
        │       needs: F1, D-gate passed (spec §15: "after the development │
        │              run"); nothing else                                  │
        │                             │                                     │
        │            C-base  confirmatory, 5 batches of 4 seeds, the 7      │
        │                    no-policy arms (+ marl_static in the same       │
        │                    invocation when SEL is done)                    │
        │       needs: M                                                    │
        │            C-marl  marl_static (if not co-run), marl_freeze        │
        │                    (sequential invocation), untrained x4,          │
        │                    sensitivities x2, S8 hash-invariance smoke      │
        │       needs: M, SEL, U                                            │
        │                             │                                     │
        └──► A   analysis (confirmatory_analysis.py, timelines, export)     │
                needs: every C pickle present, hard asserts pass            │
```

Which arms need the trained policy:

| arm | policy needed | earliest start | note |
|---|---|---|---|
| ospf, olsr, batman, aodv, sdn | no | F1 (dev), M (confirmatory) | own protocol models; invariance-locked from WP1 on |
| xldet | no | F1 / M | MARL code path, link-state service, no checkpoint |
| random | no | F1 / M | uniform over the new action space, redrawn every tick |
| untrained (x4) | no trained policy; needs U | F1 / M | initial actors from the training init path |
| marl_static | yes (selected) | SEL / M | primary arm |
| marl_freeze | yes (selected) | SEL / M | Phase-2 file dependency: one invocation at a time |
| sensitivity: installation off | yes | SEL / M | evaluation-only switch, descriptive |
| sensitivity: re-pointing off | yes | SEL / M | evaluation-only switch, descriptive |
| S8 hash-invariance smoke | yes | SEL / M | 2 instances x PYTHONHASHSEED 0 and 1 |

Critical path: F1 → T → V → SEL → D-marl → D-gate → M → C-marl → A.
D-base can overlap T, and C-base can overlap C-marl, but only inside the
shared 8-worker cap, so overlapping changes the order, not the total.

## 2. Pass counts (one pass = one arm x one scenario x one seed)

Full length 3 600 ticks unless stated. "Batch" = one harness invocation
(`--seed-list` of 4 seeds, 8 arms, 64 passes) as in the A1 chain.

| stage | passes | composition |
|---|---|---|
| V validation | 56 | 4 replicates x 7 seeds x 2 scenarios, marl_static |
| D-base | 280 | 20 seeds x 2 scenarios x 7 arms |
| D-marl | 40 + 40 + 160 + 80 = 320 | marl_static 40, marl_freeze 40, untrained 4 x 40, sensitivities 2 x 40 |
| M manifest | 40 instances | topology + severance drive + optimum solver, no full pass |
| SA seed audit | 2 runs | `tools/seed_audit.py` at freeze 1 and freeze 2: one process, no pool worker, 8-12 min per run depending on machine load (7 000 files over the 14 `$HOME/MARL*` trees, 2.5 GB of pickles, 2 600 checkpoints, plus the trackB history dump). Each run needs the dump captured on the laptop first (seconds) |
| C-base | 280 | 20 seeds x 2 scenarios x 7 arms (5 batches) |
| C-marl | 40 + 40 + 160 + 80 + 8 = 328 | marl_static, marl_freeze, untrained x 4, sensitivities x 2, S8 smoke (2 seeds x 2 hash seeds) |
| **evaluation total** | **1 264** | |
| T training | 4 x 400 episodes x 800 ticks | = 1.28 M simulated ticks per replicate plus PPO updates |

Re-run allowance: one unchanged re-run per technical failure (pre-registered);
budget ⟨perf⟩ % on top.

## 3. Cost model (to be completed by the performance track)

Only WP0 numbers exist (D0.9/D0.11, host under load 8-18, hyper-threads):

| quantity | WP0 measurement | after perf track |
|---|---|---|
| CPU per evaluation tick | 0.21-0.30 CPU-s | ⟨perf⟩ |
| CPU per 3 600-tick pass | 13-18 CPU-min | ⟨perf⟩ |
| wall per 64-pass batch at 8 workers | 1.8-2.5 h | ⟨perf⟩ |
| peak RSS per evaluation pass | 0.52-0.55 GB | ⟨perf⟩ |
| marl_freeze worker | PSS ≈ 0.35 GB at 300 ticks; budget 1 GB | ⟨perf⟩ |
| CPU per training episode (800 ticks, 22 agents) | not measured (WP13 smoke) | ⟨perf⟩ |
| PPO update per batch of 16 episodes | not measured | ⟨perf⟩ |
| training peak RSS (workers + parent rollout pool) | not measured (WP13 smoke) | ⟨perf⟩ |

Derived at the WP0 rates and 7.7 effective cores (to be replaced):

- evaluation, 1 264 passes x 13-18 CPU-min ≈ 274-379 CPU-h ≈ **36-49 h wall**;
- training, if a training tick costs what an evaluation tick costs
  (0.21-0.30 CPU-s): 1.28 M ticks ≈ 75-107 CPU-h per replicate before PPO,
  ≈ 10-14 h wall per replicate at 8 workers, **40-56 h for four run one after
  another** (two at once at 4 workers each: same total). ⟨perf: measure⟩;
- manifest: minutes; analysis: minutes.

Rough total at WP0 rates: **3.5-4.5 days** of wall time at the 8-worker cap,
training dominating. The performance track's per-pass and per-episode
figures replace this table before freeze 1; the plan's order does not
depend on them.

## 4. Scheduling rules and open decisions

1. **One 8-worker budget for everything.** Two training replicates at once
   run at 4 workers each; no evaluation batch runs beside a training run
   unless the sum stays at 8. Record the worker count of every run in its
   log header (provenance `marl_env` carries `MARL_POOL_WORKERS`).
2. **marl_freeze is one invocation at a time** (Phase-2 file dependency,
   WP10); it may use 8 workers inside that invocation.
3. **Batches of 4 seeds** for the confirmatory and development sets, as in
   the A1 chain, one output pickle per batch; the analysis merges them and
   asserts every arm and seed present.
4. **No old pickle is reused.** Every arm of every set is run from the
   frozen tree with `PYTHONHASHSEED=0`.
5. **Training determinism (D0.8).** Training is not bit-reproducible under
   the current driver (rollout pool filled in completion order). The engine
   track decides before freeze 1 whether to sort by episode index; either
   way the replicate is identified by its checkpoint sha256 in the record.
6. ⟨perf⟩ Whether the per-pass cost after the performance work changes the
   4-seed batch size or the batch order; whether a training replicate fits
   the 800-tick episode budget in the measured time.
7. ⟨perf⟩ Disk: one 64-pass full-length pickle ≈ 19-20 MB (A1 chain); the
   run-9 outputs need ≈ 0.5 GB; checkpoints ≈ 0.2 MB each plus per-agent
   adapted files.

## 5. Log of measured costs (append as runs complete)

| date | stage | workers | passes | wall | CPU-h | peak RSS | note |
|---|---|---|---|---|---|---|---|
| 2026-09-17 | invariance 600 t, 8 arms | 8 | 32 | 10.9 min | 1.36 | 0.55 GB/pass | D0.9 |
| 2026-09-17 | 1200 t routing+SDN | 8 | 20 | 14.6-14.8 min | – | 0.55 GB/pass | D0.11 |
| 2026-09-22 | seed audit 200-219, all trees, `--pt --git` | 1 (no pool) | – | 9.3-10.3 min | ≈ 0.16 | < 1 GB | `docs/SEED_AUDIT_200_219.md` |
| 2026-09-23 | seed audit 200-219, 14 trees, `--pt --git --git-log`, coverage floors | 1 (no pool) | – | 7.7 min (load 23-44) | ≈ 0.13 | < 1 GB | CLEAN, exit 0; `docs/SEED_AUDIT_200_219.md` |
| 2026-09-23 | seed audit 220-239, same roots | 1 (no pool) | – | 7.3 min | ≈ 0.12 | < 1 GB | no hit; coverage refused (4 files vanished under `~/MARL_run_9c`) |
| 2026-10-02 | invariance 600 t, 8 arms, before D12 (`54b10ba`) | 16 | 32 | 8.1 min | 1.66 | 0.75 GB/pass | D12F.5 (load 4-9) |
| 2026-10-02 | 1200 t, 8 arms, before D12 (`54b10ba`) | 16 | 32 | 16.1 min | 3.40 | 0.76 GB/pass | D12F.5 |
| 2026-10-02 | invariance 600 t, 8 arms, after D12.1 + D12.2 (`aada41e`) | 8 | 32 | 13.0 min | 1.62 | 0.76 GB/pass | D12F.5 (load 16-24) |
| 2026-10-02 | 1200 t, 8 arms, after D12.1 + D12.2 (`aada41e`) | 8 | 32 | 28.9 min | 3.61 | 0.76 GB/pass | D12F.5 |
| 2026-10-02 | controlled: 8 passes (marl_static, xldet, sdn, ospf; seed 82; A and B; 600 t), before and after side by side | 8 + 8 | 8 + 8 | 4.5 min each | 0.45 -> 0.50 | – | D12F.7: CPU +11 % in all, Scenario A +20-31 %, B -1 to +4 % |

## 6. Memory gates and their unit (review of 2026-10-02, D-FIX.14)

The memory gates that `tools/preflight.py` enforces, `docs/RUN_CHAIN.md` §3
describes and `docs/PREREGISTRATION_R2.md` §3/§12 pre-register are the
performance track's (its compute plan of 2026-09-23, kept beside this
repository on the laptop as `scratchpad/COMPUTE_PLAN.md`, 49 724 bytes,
sha256 ending `0f097b367127fae6`; stage S2a and "Failure handling" A):

| invocation | gate (MemAvailable) | abort | derived from |
|---|---|---|---|
| evaluation, 24 workers | ≥ 18 | – | 13.2 sum-RSS / 4.6 PSS peak |
| training WAVE, 2 replicates × 16 workers | ≥ 26 (MemAvailable + the PSS of the chain's own running replicates) | < 22 | 29.9 sum-RSS / 19.8 PSS |
| tool step | ≥ 4 | – | – |

**Every figure in this table is GiB (2^30 bytes), although every document
writes "GB".** `tools/preflight.py` divides the `/proc/meminfo` kB value by
1024² (`read_mem_available_gb`); the performance track read MemAvailable the
same way (`~/MARL_perf/tr2/sampler.py`, `int(...) / 1048576.0`) and its
per-process RSS/PSS as kB / 1024 (`psdetail.py`, `tools/rss_monitor.py`), so
the 19.8 and the 26 are in the unit the gate compares them in (the per-worker
0.52 "GB" PSS is 519 MiB / 1000, within 2.4 % of GiB). The training gate is
therefore 26 GiB = 27.9 GB decimal, and the code implements the plan as
written. Reading the plan's 26 as decimal GB (24.2 GiB) would LOWER the
gate by 1.8 GiB (7 %); that is a change of a pre-registered gate, not a unit
correction, and it is the author's (PREREGISTRATION_R2 §16 item 10).

What the host offered (MemAvailable in GiB, read by the preflight): 25.86-26.32
at the training wave of the `a33fdbc` rehearsal (2026-09-30 09:01-09:18 UTC);
24.52-25.50 at every one of the 95 refused training preflights of the
`02049e3` rehearsal from 2026-10-01 17:29 to 2026-10-02 08:57 UTC
(D-FIX.13), with the author's own four NVIDIA AODT infrastructure
containers resident on the host (`aodt_efid-{temporal,clickhouse,nessie,minio}-1`,
about 14 GiB, Temporal alone 11.5 GiB; unrelated to this study) and one
orphaned MARL pool worker of an earlier job of this campaign still alive
(pid 2675988, ppid 1, 8 d 20 h old, cwd `~/MARL_run_9`, 0 CPU, 0.47 GiB
RSS). Then, from `~/MARL_watch/DISK_LOG.md`: at 2026-10-02 09:00:09 UTC that
orphaned worker was killed, and at 09:04:26 UTC, ON THE AUTHOR'S
INSTRUCTION, the four AODT containers were stopped (not removed);
MemAvailable went from about 25.5 to 39.1 GiB and the waiting resume's next
preflight admitted the wave at 09:07:33 UTC. The gate was not changed.
**Corrected 2026-10-02 (D12-FIXES item ZERO):** this paragraph first said
"39.1 when a tenant released memory at 09:07 UTC" and that "nothing of this
work" was running; both were wrong -- no other tenant released anything,
the memory was freed by stopping the author's own containers and killing a
worker this campaign had leaked. **Resolution (PREREGISTRATION_R2 §16
item 10, the author's):** the 26 GiB gate stays for every profile; host
memory was freed instead; the AODT containers stay stopped until the real
run ends, and `docs/FREEZE_CHECKLIST.md` d.0 checks, before every launch,
that MemAvailable is at least 26 GiB and that no `aodt_efid-*` container is
running. The real run's training stage has the same gate and a two-hour
retry budget (`docs/FREEZE_CHECKLIST.md` d.2); on the host state before
09:00 UTC it would have stopped at S2 as well.
