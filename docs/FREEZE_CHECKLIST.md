# FREEZE_CHECKLIST — launching the real run of the final corrected rerun

Written 2026-10-01 for the author/operator, from the code at `a33fdbc` and
`docs/PREREGISTRATION_R2.md` (the pre-registration; "§" below means its
sections). The steps are ORDERED; each ends with what must be true before the
next starts. **State on 2026-10-01 (revised the same day, server reachable
again):** step 0.1 is satisfied (below); the seed audit has been MEASURED at
the rehearsed head `a33fdbc` (`docs/SEED_AUDIT_*.md` §6) and returned
exit 4 (INCOMPLETE) because of the launcher-test stand-ins of step c.0; no
other step has been executed. **State on 2026-10-02:** FIX-PREFREEZE
changed code (`1a486a7`..`02049e3`), so steps 0.1 and a.1 now name the
rehearsal on `02049e3`. That rehearsal ended `CHAIN END profile=rehearsal
rc=0` at 14:29 UTC after waiting 15.6 h at S2 on the 26 GiB training memory
gate (§16 items 9 and 10), so step 0.1 is satisfied again; the documents
committed after `02049e3` were synced after it ended. **State on
2026-10-02 evening (D12-FIXES):** the author decided on 2026-10-01 to fix
D12.1 and D12.2 ("Fix both", §16 item 1), which changed code again
(`62a85ae`..`aada41e`), so the `02049e3` rehearsal no longer covers the
head: steps 0.1 and a.1 must name the rehearsal at the D12 code head
(`docs/DECISIONS.md` D12F.9), b.2 compares with the D12 references, and
d.0 checks host memory before every launch. **State on 2026-10-03
(D12F.8, D12F.9):** the author decided to give OLSR, B.A.T.M.A.N. and AODV
the leaf model before the freeze ("Fix before freeze", D12F.8), which
changed code once more (`c6e2aff`..`ad01ff0`), and to run freeze 2 as an
automatic gate ("Automatic gate": the author's go for segment 2 is given in
advance, conditional on the freeze-2 seed audit allowing no new result-data
hit but the real run's own S0 manifest). Steps 0-d are executed, unattended
and in this order, by `~/MARL_watch/jobs/final_and_launch.sh <final head>`
(outside the repository): 0 and d.0 (`host_check`, the d.0 script, a short
host waited for up to 6 h); 0.1 at the final head (`logs/d12/suite_<head>.log`,
`logs/d12/rehearsal_<head>.log`, the ledger check below); a-c through
`~/MARL_watch/jobs/freeze_and_launch.sh <head> --check-only` (head, no
real-profile artefact, suite, seed audit over the history dump uploaded and
checksummed for that head); a.4 (`logs/freeze/FREEZE_R2_<head>.sha256`, the
schema and reward ids); b.2 against the **D12F.8 references**
`output/ref/inv_ref_d12f8_ad01ff0{,_t1200}.pkl` (every arm, 16 PASS lines);
b.3; b.4; d.1 (`logs/freeze/real_chain.env`, the plan 37/1060); d.3; d.4
(records in `logs/freeze/freeze2_<head>.txt` with the UTC time, the audit
re-run and gated); d.5. Any failed step stops the pipeline before the next.
The step texts below are the procedure it follows; where they name
`02049e3` or the `aada41e` references, read the final head and the D12F.8
references. **State on 2026-10-04 (XL-1):** the first real-run
attempt at `bbfd7b0` was held before S6 and then stopped at S2 (no
confirmatory pass; archived, PREREGISTRATION_R2 section 13) because its
development output showed XL-DET's Scenario-B defect (D-XL.1); XL-1 changed
code (`d7436c2`, `9ac7646`). The pipeline is run again at the XL-1 head,
with b.2 against the **XL1 references** `output/ref/inv_ref_xl1_9ac7646{,_t1200}.pkl`
and freeze 2 still automatic on the author's advance go. **Step d launches the study and is NOT to be run until steps 0–c
have passed and the author has taken the decisions of §16.**

Conventions, for every step:

- From the laptop, in Git Bash, with
  `export PATH="/usr/bin:/bin:/mingw64/bin:<local-bin>:$PATH"`.
  Every git command carries `-c core.autocrlf=false`. Nothing heavier than
  git, ssh and a checksum runs on the laptop.
- Every ssh is `ssh -o BatchMode=yes <compute-server> ...` (key only). Long server
  jobs are started detached (`nohup ... > log 2>&1 < /dev/null &`) and polled.
- On the server, after the sync of step a, nothing writes a TRACKED file
  under `~/MARL_run_9` until the run's last `CHAIN END`. Only `logs/`,
  `output/` and `audit_history/` (untracked) are written. The chain pins the
  tree digest at its first preflight; a tracked change stops it.
- Records appended to the pre-registration and to the seed-audit record are
  committed in the LOCAL repository (and the manuscript repository) and are
  NOT synced while the run is in progress.
- The confirmatory block is never written out literally in this file: this
  file is not one of the seed audit's planning paths, so the commands read
  the block from `confirmatory/chain_plan.py` (`PROFILES['real']['cf_seeds']`).

```bash
SP='<local-workdir>'
RUN9="$SP/run9"
G="git -C $RUN9 -c core.autocrlf=false"
```

---------------------------------------------------------------------------

## 0. Preconditions

0.1 **The rehearsal on `02049e3` has ended, rc=0, with every invocation of
its plan OK and the secondary analyses computed.** FIX-PREFREEZE changed
code (`1a486a7`..`02049e3`, §16 item 9), so the rehearsal on `a33fdbc` no
longer covers the head; the one that does is
`logs/prefreeze_fix/rehearsal_02049e3.log` (D-FIX.13: launched 2026-10-01
16:00 UTC, resumed twice on the same ledger; the log holds one `CHAIN END`
per launch, and only the LAST counts).

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
grep "CHAIN END" logs/prefreeze_fix/rehearsal_02049e3.log | tail -1
P="python3 confirmatory/chain_plan.py --profile rehearsal"
INVS=$(for s in $($P --stages); do $P --ids "$s"; done)
python3 - confirmatory/chain_rehearsal.jsonl $INVS <<'PY'
import json, sys
last, ends, invs = {}, [], sys.argv[2:]
for ln in open(sys.argv[1]):
    r = json.loads(ln)
    if r.get('event') == 'end':
        last[r['inv']] = r
    elif r.get('event') == 'chain_end':
        ends.append(r.get('rc'))
ok = [i for i in invs if (last.get(i) or {}).get('status') == 'ok']
s10 = (last.get('s10_sens_install') or {}).get('ablation')
print('plan invocations %d, ended ok %d, not ok %s, last chain_end rc %s, S10 ablation %s'
      % (len(invs), len(ok), sorted(set(invs) - set(ok)), ends[-1:], s10))
sys.exit(0 if invs and len(ok) == len(invs) and ends and ends[-1] == 0
         and s10 == 'exercised' else 1)
PY
echo "LEDGER EXIT=$?"
grep -c '^export MARL_LS_INSTALL=0$' confirmatory/env_rehearsal/s10_sens_install.sh
grep -c 'MARL_DETECT_TICKS' confirmatory/env_rehearsal/s10_sens_install.sh
grep -E '\] (S8 computed|secondary S7 computed|analysis S12 computed|analysis S8 embedded|secondary report ->)' \
  logs/chain_rehearsal/s12_secondary.log
EOF
```

Must print: the last line `... CHAIN END profile=rehearsal rc=0`; `plan
invocations 33, ended ok 33, not ok [], last chain_end rc [0], S10 ablation
exercised` and `LEDGER EXIT=0`; `1` then `0` (S10 runs `MARL_LS_INSTALL=0`,
not the omniscient `MARL_DETECT_TICKS=0`, and its ablation was compared with
the tick-matched S6 baseline and differs); and the five S12 lines:
`S8 computed` and `analysis S8 embedded` with a non-zero series count and
`same_hashseed=False`, `secondary S7 computed` with 4 comparisons,
`analysis S12 computed: 3 untrained control(s)`, and the `secondary report`
line with `S12 computed, secondary S7 computed`, an S8 state that is
neither `missing` nor `VACUOUS`, `full report computed` and `primary
reproduced identical`. Read counts
and states only, never a value. Any other ending is a defect to fix before
the freeze (a code change means a new rehearsal).

**Satisfied, checked 2026-10-02 14:31 UTC** (this script, as written):
`[2026-10-02T14:29:24Z] CHAIN END profile=rehearsal rc=0`; `plan
invocations 33, ended ok 33, not ok [], last chain_end rc [0], S10 ablation
exercised`, `LEDGER EXIT=0`; `1`, `0`; and the five S12 lines (S8: 268
series, `same_hashseed=False`; secondary S7: 4 comparisons; analysis S12:
3 untrained controls; the report: S12 and secondary S7 `computed`, full
report `computed`, primary reproduced `identical`). The wait at S2 (15.6 h
on the 26 GiB training memory gate, the host at 24.5-25.5 GiB) is D-FIX.13,
D-FIX.14 and §16 item 10. Re-run the script anyway before step a: it costs
nothing. The `a33fdbc` rehearsal (`logs/prefreeze/rehearsal_a33fdbc.log`,
`CHAIN END profile=rehearsal rc=0` at 2026-09-30T12:32:54Z) is the record
of the head before FIX-PREFREEZE only.

0.2 **No simulator, trainer, chain or test process of ours in the tree.**

```bash
ssh -o BatchMode=yes <compute-server> 'for p in $(pgrep -u "$USER" -f "run_timeline_comparison[.]py|train_on_comparison[.]py|run_final[.]sh|run_tests[.]py"); do
  printf "%s %s %s\n" "$p" "$(readlink /proc/$p/cwd)" "$(tr "\0" " " < /proc/$p/cmdline | cut -c1-120)"; done'
```

No line may show a cwd under `~/MARL_run_9`. (The `[.]` keeps the pattern
from matching its own command line; other trees' jobs are foreign load.)

0.3 **No artefact of an earlier `real` chain** (a stale `FREEZE_FINAL` seal
or ledger would make every preflight refuse):

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && ls -d confirmatory/*real* \
  confirmatory/env_real confirmatory/preflight_real output/final logs/chain_real* 2>/dev/null; echo END'
```

Must print only `END`.

0.4 **The author's decisions of §16 are recorded** in the local repository
(§5 block decision, D12.1/D12.2, the S10 switch, S7/S8/S12, the two-segment
launch, worker counts). If any decision changes code, stop here: the code
change needs its own record, invariance, rehearsal and a new freeze.

## a. Sync the final tree

a.1 **The tree to sync is the rehearsed code plus documents only.**

```bash
$G status --porcelain                     # must print nothing
$G diff --name-only 02049e3 HEAD          # must list docs/ files only
$G diff --name-only 02049e3 HEAD | grep -vc '^docs/'   # must print 0
HEAD_SHORT=$($G rev-parse --short HEAD); echo "$HEAD_SHORT"
```

The rehearsed head is `02049e3`, the last code commit of FIX-PREFREEZE
(step 0.1; §16 item 9 supersedes item 8, whose `b43316a` test-file
difference `02049e3` contains). If anything outside `docs/` changed since
`02049e3`, the rehearsal does not cover this head: STOP. **As of 2026-10-02
the commits after `02049e3` are documents only** (D-FIX.6 to D-FIX.14 and
the records of §15/§16); they were NOT synced while the rehearsal ran,
because the rehearsal's ledger pins `tree_sha256`, which covers `docs/`
(D-FIX.13), and were synced after its `CHAIN END` on 2026-10-02. The sync of
step a.2 then sends the freeze head itself.

a.2 **Sync** (git-tracked files only; `.gitattributes` is `* -text`, so the
archive carries the committed bytes):

```bash
cd "$SP" && bash sync_run9.sh             # "synced <head> to ~/MARL_run_9 (N files)"
```

a.3 **Verify the server tree is the commit, byte for byte.**

```bash
ssh -o BatchMode=yes <compute-server> 'cat ~/MARL_run_9/.synced_head'      # = $HEAD_SHORT
( cd "$RUN9" && $G ls-files -z | xargs -0 sha256sum | awk '{print $1}' | sha256sum )
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && tr "\n" "\0" < .synced_files | xargs -0 sha256sum | awk "{print \$1}" | sha256sum'
```

The two digests must be equal.

a.4 **Record the identity of the frozen tree** (freeze-1 record, §13):

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
mkdir -p logs/freeze
H=$(cat .synced_head)
tr '\n' '\0' < .synced_files | xargs -0 sha256sum > logs/freeze/FREEZE_R2_$H.sha256
sha256sum logs/freeze/FREEZE_R2_$H.sha256
date -u '+%Y-%m-%dT%H:%M:%SZ'
PYTHONHASHSEED=0 python3 - <<'PY'
import sys
sys.path.insert(0, '.')
from sixg_sim import runinfo
from sixg_sim import reward_spec
p = runinfo.provenance()
print('tree_head', p.get('tree_head'), 'tree_sha256', p.get('tree_sha256'))
print('schema_id', p.get('schema_id'), 'reward_id', reward_spec.reward_id())
PY
EOF
```

Expected: `schema_id obs1-542eb7fc0d277d1d`, `reward_id
rew1-7f89883ae19733f3`. Copy `FREEZE_R2_<head>.sha256`, its sha256 and the
UTC time to `records/R2/` of the manuscript repository and into §13
(locally).

## b. Confirm the suite, invariance and determinism at that head

Run b.1, b.2, b.3 and b.4 one after the other, each ALONE in the tree (the
launcher tests' preflight refuses while another simulator of this tree runs,
D-MRG.11).

b.1 **Full suite**, 0 failed and 0 skipped:

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
for v in $(compgen -e | grep '^MARL_'); do unset "$v"; done
H=$(cat .synced_head)
nohup bash -c 'PYTHONHASHSEED=0 MARL_POOL_WORKERS=16 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 tests/run_tests.py; echo "# exit=$?"' \
  > logs/freeze/suite_$H.log 2>&1 < /dev/null &
EOF
# poll until the exit line appears:
ssh -o BatchMode=yes <compute-server> 'tail -3 ~/MARL_run_9/logs/freeze/suite_$(cat ~/MARL_run_9/.synced_head).log'
```

Pass: `COUNTS passed=<n> failed=0 skipped=0` and `# exit=0`. Record `n`
(the last recorded count is 446 at `2ac9b08`; WP13 added tests since).

b.2 **Invariance against the D12-FIXES references** (D12.1 and D12.2
changed every arm by design, so the WP13 references no longer describe the
engine; `output/ref/README.md` "D12-FIXES references", D12F.5-D12F.7). The
freeze head's engine must be the one the references were made with
(`aada41e`; every commit after it changes documents, tests or tools only),
so EVERY arm must be identical at both lengths, the MARL-path arms read on
the references' own checkpoint:

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
H=$(cat .synced_head)
CK=$PWD/output/ref/inv_ref_d12_aada41e.minted_s1234.pt
nohup bash -c "INV_CHECKPOINT=$CK PYTHONHASHSEED=0 MARL_POOL_WORKERS=8 tools/run_invariance_pass.sh inv_freeze_$H.pkl \
  && INV_CHECKPOINT=$CK INV_TICKS=1200 PYTHONHASHSEED=0 MARL_POOL_WORKERS=8 \
     tools/run_invariance_pass.sh inv_freeze_${H}_t1200.pkl; echo \"# exit=\$?\"" \
  > logs/freeze/inv_$H.log 2>&1 < /dev/null &
EOF
# when logs/freeze/inv_<head>.log ends "# exit=0":
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 && H=$(cat .synced_head)
tools/invariance_check.sh --wp WP0 output/ref/inv_ref_d12_aada41e.pkl output/inv_freeze_$H.pkl; echo "EXIT=$?"
tools/invariance_check.sh --wp WP0 output/ref/inv_ref_d12_aada41e_t1200.pkl output/inv_freeze_${H}_t1200.pkl; echo "EXIT=$?"
grep -n -i "THE CURRENT REFERENCE\|D12-FIXES references" output/ref/README.md | head
EOF
```

Pass: both `INVARIANCE PASS`, `EXIT=0`, every one of the eight arms
`identical == series` (600 ticks: ospf olsr batman aodv sdn 264,
marl_static 276, xldet random 268; the 1200-tick reference now holds all
eight arms too); quote the full output (D7.15). The superseded post-WP13
references (`output/ref/inv_ref_wp13_c7d2322*`) stay as the record of the
code before D12.

b.3 **The Track B contract on the new pickle:**

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && H=$(cat .synced_head) && \
  python3 tools/wp4_trackb_contract.py --analysis confirmatory_analysis.py \
    --manifest benchmark_manifest_80_99.json output/inv_freeze_$H.pkl; echo "EXIT=$?"'
```

Pass: `VERDICT PASS`, `EXIT=0`.

b.4 **The determinism gate on the final tree** (§3; last executed before
WP6–WP13). The gate's report reads every directory under `output/det_gate/`,
so the old runs are moved aside first:

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
H=$(cat .synced_head)
[ -d output/det_gate ] && mv output/det_gate output/det_gate_before_$H
nohup bash -c 'bash tools/det_gate.sh serial && bash tools/det_gate.sh concurrent \
  && bash tools/det_gate.sh one w08_s2345 8 2345; bash tools/det_gate.sh report; echo "# exit=$?"' \
  > logs/freeze/detgate_$H.log 2>&1 < /dev/null &
EOF
ssh -o BatchMode=yes <compute-server> 'tail -12 ~/MARL_run_9/logs/freeze/detgate_$(cat ~/MARL_run_9/.synced_head).log'
```

Pass: `PASS seed=1234: <k> runs, one policy (...)` over the 1-, 8- and
16-worker serial runs and the four concurrent replicates, `PASS: distinct
torch seeds give distinct policies`, `GATE: PASS`, `# exit=0`. Record the
new weights digest of seed 1234 (it differs from the pre-WP6 `f4a807cc…`,
because the network changed). `tools/det_gate.sh` is not executable in the
tree (mode 100644, as committed), hence `bash tools/det_gate.sh`: called
directly it stops at once with "Permission denied" and `# exit=126`
(D12-FIXES found this when it ran the gate, D12F.7).

Any failure in b stops the freeze.

## c. Re-run the seed audit at that head

Nothing may write into any `~/MARL*` tree while it runs (a vanishing file
refuses the verdict). Steps 0.2 and b must be finished, and no other session
may sync, capture a history dump, run a suite or run this driver until c.2
has ended: on 2026-10-01 a sync, a replaced dump and a suite of another
session overlapped the audit, which is therefore not freeze evidence
(`docs/SEED_AUDIT_*.md` §6.3). Check with step 0.2's command, and
`cat ~/MARL_watch/state.json` for registered jobs of this tree, first.

c.0 **The launcher-test stand-ins — ONLY with the author's written
consent (§16 item 7).** The audit measured on 2026-09-30 and 2026-10-01
returned exit 4 because the launcher tests leave synthetic stand-ins in the
scanned trees: 244-byte pickles named `confirmatory/test_manifest_*.json`
(the JSON parse fails) and 238-byte pickles at
`output/test/untrained_s1234/policy_init.pt` (the checkpoint load fails),
in `~/MARL_run_9` and in the record tree `~/MARL_run_9d`
(`docs/SEED_AUDIT_*.md` §6.5 lists each with its sha256). Step b.1
re-creates one in `~/MARL_run_9` unless the tests' cleanup of `b43316a`
removes it, so do this after b and check first:

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~ && ls -l MARL_run_9/confirmatory/test_* MARL_run_9/output/test \
  MARL_run_9d/confirmatory/test_* MARL_run_9d/output/test 2>/dev/null; ls -d ~/archive_marl_test_fixtures_* 2>/dev/null; echo END'
```

If anything is listed, record and archive it OUTSIDE every scanned root (a
directory directly under `$HOME` matches no `$HOME/MARL*` pattern), never
delete it:

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
set -u
A=$HOME/archive_marl_test_fixtures_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$A"
cd ~ && for f in MARL_run_9/confirmatory/test_* MARL_run_9/output/test MARL_run_9d/confirmatory/test_* MARL_run_9d/output/test; do
  [ -e "$f" ] || continue
  find "$f" -type f -exec sh -c 'printf "%s %s %s\n" "$1" "$(stat -c %s "$1")" "$(sha256sum < "$1" | cut -c1-64)"' _ {} \; >> "$A/MANIFEST.txt"
  mkdir -p "$A/$(dirname "$f")" && mv "$f" "$A/$f"
done
cat "$A/MANIFEST.txt"; echo "archived to $A"
EOF
```

Copy `MANIFEST.txt` into the new dated section of the seed-audit record
(c.4). The sha256 of the files that were there on 2026-10-01 must match
§6.5 of that record; a file not in that table is reported, not archived
silently. (On 2026-10-01 another session's job,
`~/MARL_watch/jobs/freeze_checks.sh`, was seen to archive the `~/MARL_run_9d`
stand-ins into a tarball under `$HOME` and delete them; if it ran, its
output in `~/MARL_run_9/logs/freeze_checks.log` is the record of that step
and must be quoted in c.4, and the consent question still stands.)

c.1 **History dump of ALL refs, from the laptop, at the synced head:**

```bash
$G log --all -p --no-color | ssh -o BatchMode=yes <compute-server> \
  'mkdir -p ~/MARL_run_9/audit_history && cat > ~/MARL_run_9/audit_history/trackB_git_log.txt'
$G log --all -p --no-color | sha256sum
$G rev-list --all | wc -l
ssh -o BatchMode=yes <compute-server> 'sha256sum ~/MARL_run_9/audit_history/trackB_git_log.txt; \
  grep -c "^commit " ~/MARL_run_9/audit_history/trackB_git_log.txt'
```

The two sha256 must be equal, and the commit counts equal. Capture it
immediately before c.2, and after c.2 check that the report scanned THIS
dump (`git_logs[0].sha256` and `.commits` in the report, printed by c.3): on
2026-10-01 another session replaced the dump four minutes into the run, and
the run scanned the replacement.

c.2 **The committed driver, over the merged tree, read-only apart from its
untracked outputs** (block bounds read from the plan):

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
read LO HI < <(python3 -c "import sys; sys.path.insert(0, 'confirmatory'); import chain_plan as c; s = c.PROFILES['real']['cf_seeds']; print(min(s), max(s))")
H=$(cat .synced_head)
nohup env MARL_TREE=$HOME/MARL_run_9 bash tools/run_seed_audit.sh "$LO" "$HI" \
  > logs/freeze/seed_audit_driver_$H.log 2>&1 < /dev/null &
EOF
# about 8-12 min; then:
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && tail -2 logs/freeze/seed_audit_driver_$(cat .synced_head).log; \
  ls -l output/seed_audit_*.json logs/seed_audit_*.log; sha256sum output/seed_audit_*.json'
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && PYTHONHASHSEED=0 nice -n 19 python3 tests/run_tests.py wp12_seed_audit | tail -3'
```

Expected exit status **3 (TOUCHED) only if c.0 left no stand-in in any
scanned tree**; with the stand-ins present the result is **4 (INCOMPLETE)**,
as measured on 2026-09-30 and 2026-10-01, and the freeze stops (fix through
c.0, then re-run c.1–c.2). Any other coverage failure also stops it. The
driver writes `output/seed_audit_<lo>_<hi>.json` and
`logs/seed_audit_<lo>_<hi>.log` (planning-classified names) at FIXED paths
that the next run overwrites: copy both at once, with their sha256, to
`audit_history/run_<UTC start>/` (excluded from the walk):

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && D=audit_history/run_$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p $D && \
  cp -p output/seed_audit_*.json logs/seed_audit_*.log audit_history/trackB_git_log.txt $D/ && sha256sum $D/*'
```

The tool's tests must pass.

c.3 **Classify EVERY hit, and hide none** (§13; the report lists them under
`verdict`, `text_hits`, `filename_hits`, the per-file `hits` of the pickle,
JSON and checkpoint sections, `planning_hits`, `false_positive_hits`,
`broad_tokens['other']`, `errors` and `coverage`). Print the paths, never a
pickle's metrics — and never the text of a log line, which can carry a
result: the script below prints a line's text only for source and
documents, and only paths, line numbers and values for anything under
`logs/`, `output/` or `archive_rehearsal/` or with a log-like suffix:

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
read LO HI < <(python3 -c "import sys; sys.path.insert(0, 'confirmatory'); import chain_plan as c; s = c.PROFILES['real']['cf_seeds']; print(min(s), max(s))")
python3 - "output/seed_audit_${LO}_${HI}.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
v = r['verdict']
print('VERDICT', {k: v[k] for k in sorted(v) if not isinstance(v[k], (dict, list))})
print('coverage failures', v.get('coverage_failures'))
print('errors', r.get('errors'))
print('synced_head', r.get('synced_head'))
# the history dump(s) the run scanned: the report key is 'git_logs' (a list);
# 'git_log' does not exist and printed None (FIX-PREFREEZE item 10)
gl = r.get('git_logs') or []
for g in gl:
    print('git_logs', g.get('path'), 'sha256', g.get('sha256'), 'commits',
          g.get('commits'), 'lines', g.get('lines'), 'scanned', g.get('scanned'))
if not any(g.get('scanned') and g.get('commits') for g in gl):
    print('GIT HISTORY NOT SCANNED (%d dump record(s)): the freeze stops' % len(gl))
def safe(e):
    p = e['path']
    if any(s in p for s in ('/logs/', '/output/', 'archive_rehearsal')) or \
       p.endswith(('.log', '.out', '.jsonl', '.txt', '.tsv')):
        return '(log or output line: text not printed)'
    return e['snippet'][:100]
for sec in ('pickles', 'json', 'pt'):
    for path, e in sorted((r.get(sec) or {}).items()):
        if isinstance(e, dict) and e.get('hits'):
            print('DATA', sec, path, 'hits', e['hits'], 'planning', e.get('planning'))
        if isinstance(e, dict) and e.get('error'):
            print('FAILED', sec, path, str(e['error'])[:120])
for e in r['text_hits']:
    print('TEXT', e['path'], e['line'], e['values'], e['kinds'], safe(e))
for e in r.get('git_hits', []):
    print('GIT ', e['path'], e['line'], e['values'], e['kinds'], safe(e))
for e in r['filename_hits']:
    print('NAME', e)
print('planning hits', len(r['planning_hits']), 'false positives', len(r['false_positive_hits']))
for e in r['broad_tokens']['other']:
    print('BROAD', e['path'], e['line'], e['token'], e['planning'])
PY
EOF
```

(History lines appear twice, under TEXT and under GIT; a commit message line
belongs to no file and is never planning, so the `31b034c` message that
describes the exposure is one of the lines to classify.)

Classes (§13; the measured groups, with counts, are §6.6 of the seed-audit
record): (a) result data of the two reduced rehearsals of 2026-09-24 on the
four rehearsal seeds, all in `~/MARL_run_9d` (13 pickles, 3 manifests, 8
adapted-checkpoint names, 1 094 log/ledger lines), verified from the
ledgers and the pickles' provenance — 600-tick passes, the 2-batch 200-tick
toy policy, trees `0592f26` and `79e15b2`, S7 crashed twice in each; (b)
planning references — the real profile in `confirmatory/chain_plan.py`, the
tests that pin it, the planning documents, the superseded run-chain code of
`~/MARL_run_9d`, the audit's own reports, the `agent.py` docstring false
positive (and, at freeze 2, the real run's own S0 manifest and the chain's
lines naming it), each classified with a stated reason; a classification
run of the tool with explicit `--false-positive` patterns may be added
beside the driver's run, never instead of it; (c) the two groups the author
classifies (§16 item 7): the launcher-test logs that name the old stand-in
manifest, and the `output/wp13_*` copies of the tree; (d) anything else,
in particular any group not listed in §6.6 of the record: STOP and report it
plainly. The logs written by steps a–b (`logs/freeze/`) are scanned like
any other file; a passing suite prints no seed of the block, and a hit in
one of them is class (d) unless the line is a test fixture that its own
test file names. The rehearsal on `a33fdbc` and the dry run touched only
unregistered seeds (measured 2026-10-01, record §6.7); the rehearsal on
`02049e3` (step 0.1) runs on the same unregistered seeds 220-223
(`PROFILES['rehearsal']['cf_seeds']`) and was not yet measured by the
audit: its artefacts are new files in this scan, and a hit in one of them is
class (d).

c.4 **Record** the command, the report path and its sha256, every hit with
its class, and the disclosure paragraph of §5 as a new dated section of the
seed-audit record (`docs/SEED_AUDIT_*.md`) and the verdict block in §13 —
locally, not synced.

Any hit of class (d), a group of class (c) the author has not classified,
or a coverage failure, stops the freeze.

## d. Launch the real chain — DO NOT RUN before 0–c pass and §16 is decided

The block decision (§5) is final from here: the first invocation
(`s0_manifest`) builds the block's manifest.

d.0 **Host memory, immediately before EVERY launch (segment 1, segment 2,
a resume).** The 26 GiB training gate stays for every profile (§16 item 10,
the author's decision of 2026-10-02); the host offered 24.5-25.5 GiB for
15.6 h while the author's own four NVIDIA AODT infrastructure containers
(`aodt_efid-{temporal,clickhouse,nessie,minio}-1`, about 14 GiB) were
running, and they were stopped (not removed) on the author's instruction at
2026-10-02 09:04:26 UTC (`~/MARL_watch/DISK_LOG.md`). They stay stopped
until the real run ends; do NOT start them. The check counts what it read
and fails on a reading it could not make:

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
python3 - <<'PY'
import subprocess, sys
fails, checked = [], 0
mem = None
for ln in open('/proc/meminfo'):
    if ln.startswith('MemAvailable:'):
        mem = int(ln.split()[1]) / 1024.0 ** 2      # kB -> GiB, the gate's unit
if mem is None:
    fails.append('MemAvailable not read')
else:
    checked += 1
    print('MemAvailable %.2f GiB (gate 26 GiB)' % mem)
    if mem < 26.0:
        fails.append('MemAvailable %.2f GiB < 26' % mem)
p = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}\t{{.State}}'],
                   capture_output=True, text=True)
rows = [l.split('\t') for l in p.stdout.splitlines() if '\t' in l]
if p.returncode != 0 or not rows:
    fails.append('docker ps listed nothing (rc %d): cannot verify' % p.returncode)
else:
    checked += 1
    aodt = [(n, s) for n, s in rows if n.startswith('aodt_efid-')]
    running = [n for n, s in aodt if s == 'running']
    print('containers listed %d, aodt_efid-* found %d, running %d: %s'
          % (len(rows), len(aodt), len(running), ' '.join('%s=%s' % r for r in aodt)))
    if running:
        fails.append('aodt_efid-* running: %s' % running)
print('HOST CHECK %s: %d checks read%s' % ('PASS' if checked == 2 and not fails else 'FAIL',
      checked, ('; ' + '; '.join(fails)) if fails else ''))
sys.exit(0 if checked == 2 and not fails else 1)
PY
echo "EXIT=$?"
EOF
```

Pass: `HOST CHECK PASS: 2 checks read`, `EXIT=0`, the four `aodt_efid-*`
listed as `exited`. A FAIL on memory is a wait (the chain's own preflight
would refuse the wave in the same state), never a reason to start or stop
anything else on the host; a running `aodt_efid-*` container goes back to
the author.

d.1 **The launcher's own environment.** The preflight checks
`PYTHONHASHSEED` and the thread pins in the process that runs it and refuses
every invocation without them (a permanent refusal). Every invocation
starts from an empty environment and its env file (FIX-PREFREEZE item 5,
§12), so a stray `MARL_*` of this shell cannot reach it; clearing them here
is kept as a second guard. `TMPDIR` is passed on (recorded in each env
file): if `/` is full on the host (it was on 2026-10-01), point it at a
directory under `~/MARL_run_9/output/`:

```bash
ssh -o BatchMode=yes <compute-server> 'cat > ~/MARL_run_9/logs/freeze/real_chain.env' <<'EOF'
# launcher environment of the real chain (sourced by register_chain.sh and by
# the watchdog's restart path)
for v in $(compgen -e | grep '^MARL_'); do unset "$v"; done
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
EOF
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && python3 confirmatory/chain_plan.py --profile real'
```

The plan summary must read 37 invocations and 1060 passes (the plan's 1056
and the 4-pass analysis-S8 repeat S6h of FIX-PREFREEZE item 2). (Do NOT use
`run_final.sh --profile real --dry-run`: it writes a `chain_start` record and
preflight records into the real ledger and pins its tree.)

d.2 **Preflight retry budget, chosen for the MEMORY gate only:**
`--preflight-retries 12 --preflight-wait 600` (two hours per invocation,
and per training replicate, which inherits the same budget). Memory on this
host recovers as other tenants finish, and the plan prefers a late wave to a
perturbed one. **Known issue:** the budget applies to EVERY refusal, so a
permanent one (tree digest, `FREEZE_FINAL` seal, hash seed, thread pins, a
leftover job of ours) is also retried for two hours before the chain stops —
watch for it (step e) instead of waiting it out.

d.3 **Segment 1, S0 to S5b** (the two-segment launch gives freeze 2 its
pause, §13), detached and registered with the watchdog:

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && \
  grep "CHAIN END" logs/prefreeze_fix/rehearsal_02049e3.log | tail -1 | grep -q "profile=rehearsal rc=0" && \
  ~/MARL_watch/register_chain.sh rc_real_seg1 "$HOME/MARL_run_9" \
    "bash confirmatory/run_final.sh --profile real --stages S0,S1,S2,S3,S3b,S4,S5,S5b --preflight-retries 12 --preflight-wait 600" \
    logs/freeze/rc_real_seg1.log "CHAIN END profile=real rc=0" 20 logs/freeze/real_chain.env 600'
```

Arguments: NAME CWD CMD LOGFILE DONE_MARKER EXPECTED_HOURS ENVFILE STALL_MIN.
The done marker is printed only when the segment ends with rc=0. The stall
threshold is 600 min because the launcher is silent while it waits on a
training wave (two replicates of ~3.5 h each); the watchdog only flags a
stall. Expected, from the dry run's elapsed times (Duration, below): S1
7.4 h, S2 7.6 h (two waves of two replicates), S3 1.7 h, S4 1.7 h, S5/S5b
minutes; the segment 18.4 h.

d.4 **Freeze 2, between the segments**, after `CHAIN END profile=real rc=0`
in `logs/freeze/rc_real_seg1.log` and `logs/chain_real.log`:

- copy `confirmatory/FREEZE_FINAL_real.txt` and
  `confirmatory/SELECTED_real.txt` (and the untrained checkpoints' sha256,
  `sha256sum output/final/untrained_s*/policy_init.pt`) into §13, locally;
- check the development report exists:
  `tail -5 confirmatory/final_dev_report.txt` (S5b, hard asserts passed);
- re-run step c (fresh history dump, the driver, the classification): the
  only new hits allowed are the real run's own S0 manifest
  (`python3 confirmatory/chain_plan.py --profile real --field s0_manifest artifact`)
  and the lines of the chain's own logs and ledger that name it or the block
  (`logs/chain_real/s0_manifest.log` prints the block it builds) — class (b);
  a hit anywhere else, in particular in any S1–S5b artefact, is class (d);
- the author's go.

d.5 **Segment 2, S6 to S12:**

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && \
  ~/MARL_watch/register_chain.sh rc_real_seg2 "$HOME/MARL_run_9" \
    "bash confirmatory/run_final.sh --profile real --from S6 --preflight-retries 12 --preflight-wait 600" \
    logs/freeze/rc_real_seg2.log "CHAIN END profile=real rc=0" 16 logs/freeze/real_chain.env 600'
```

The ledger is one run across both segments (the tree pin spans it). If the
author waives freeze 2, a single `--profile real` launch under the name
`rc_real` with EXPECTED_HOURS 34 replaces d.3–d.5, and the waiver is
recorded in §13.

**Duration, measured (read 2026-10-01 from the dry run's ledger).** The
full-scale dry run (profile `dryrun`: the real profile's stages, arms,
training and worker counts on unregistered confirmatory seeds; tree
`b728ce8`, without the performance package; ledger
`~/MARL_run_9d/confirmatory/chain_dryrun.jsonl`, log
`~/MARL_run_9d/logs/chain_dryrun.log`) ran from `2026-09-27T06:47:46Z` to
`2026-09-28T14:53:39Z`: **32.1 h end to end**, `CHAIN END profile=dryrun
rc=0`. (Its first attempt, 2026-09-25 11:22–22:06 UTC, stopped in training;
`logs/chain_dryrun_attempt1.jsonl`, not counted.) Two figures must not be
confused:

- `tools/stage_report.py --timings confirmatory/chain_dryrun.jsonl --profile
  dryrun --run all` prints `TOTAL 36 1056 142093 39.47 h`. That is the SUM
  of the 36 invocations' wall clocks, so invocations that run at the same
  time are counted separately: the four training replicates run two at a
  time, and S2 sums to 15.02 h although it took **7.6 h** of elapsed time
  (first replicate start 14:12:03Z, last replicate end 21:46:13Z, two waves
  of about 3.7 h). An earlier review read the 39.47 h as the run's wall
  clock; the elapsed time is the 32.1 h above.
- Elapsed per segment, from the ledger's own stage records: **S0–S5b
  18.4 h** (S1 7.4 h, S2 7.6 h, S3 1.7 h, S4 1.7 h, S0/S3b/S5/S5b minutes)
  and **S6–S12 13.7 h** (S6 6.6 h, S8 4.0 h, S9 1.2 h, S10 1.0 h, S11
  1.0 h, S7/S12 minutes).

The watchdog compares a registered job's ELAPSED hours since registration
with 1.5 × EXPECTED_HOURS (`~/MARL_watch/watchdog.py`), so EXPECTED_HOURS 20
(segment 1), 16 (segment 2) and 34 (one segment) stay as they are: they sit
above the measured 18.4 h, 13.7 h and 32.1 h, and the flag fires at 30 h,
24 h and 51 h. A real run slower than the dry run by more than about 60 %
in segment 1 or 75 % in segment 2 is flagged, which is the intent.
`docs/RUN_CHAIN.md` §6 extrapolated ~29.3 h from the 2026-09-24 rehearsal,
with the primary result at ~T+24 h; the dry run puts S7 at T+24.9 h
(07:44Z on 2026-09-28).

## e. What to watch, and the restart rule

e.1 **Progress** (read-only):

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && tail -25 logs/chain_real.log'
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && python3 tools/stage_report.py --timings confirmatory/chain_real.jsonl --profile real'
ssh -o BatchMode=yes <compute-server> 'cat ~/MARL_watch/status/HEALTH.md'
ssh -o BatchMode=yes <compute-server> 'free -g | head -2; uptime; df -h /home | tail -1'
```

Per invocation: `logs/chain_real/<inv>.log` (harness output),
`<inv>.rss.tsv` (memory), `confirmatory/preflight_real/<inv>.*.json`. Every
END line must read `OK ... exit=0 gate=0`; S5 prints `FREEZE_FINAL sealed`;
S5b and S7 print `report: <n> primary test(s)`; S10 must read
`ablation=exercised`; S11 may read `NOT_EXERCISED` (recorded, not a pass).

e.2 **Preflight refusals.** `grep -E "wait |DELAY |PREFLIGHT\] (FAIL|ABORT)|^ +FAIL "
logs/chain_real.log | tail` names the failed check. `mem_available_gb`,
`home_free_gb` and `tenant_load` are transient: let the budget run.
`tree_pinned`, `freeze_final`, `pythonhashseed`, `thread_pinning`, or an
`our_jobs` leftover that is not a job of this chain just finishing, are
permanent: stop the segment (the pid is in `logs/freeze/rc_real_seg<k>.pid`;
stop any still-running invocation of THIS chain, identified by
`MARL_CHAIN_INV` in `/proc/<pid>/environ`), fix the cause without touching a
tracked file, and re-issue the same segment command: the ledger resumes and
skips every invocation already recorded OK with an unchanged sha256.

e.3 **Provenance of the launcher environment**, once the first S1 pickle
exists (a development pickle, not a confirmatory one). Since FIX-PREFREEZE
item 5 the chain checks this itself for every invocation (the
`[PREFLIGHT-ENV] PASS` line before each launch, `envcheck` in the ledger,
and `marl_env: <n> variable(s) compared with the plan, 0 problem(s)` after
each evaluation); this manual look is a cross-check:

```bash
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9 && python3 -c "
import pickle; p = pickle.load(open(\"output/final/s1_dev_b0.pkl\", \"rb\"))[\"provenance\"]
print(p[\"tree_head\"], p[\"tree_sha256\"][:16], p[\"schema_id\"], p[\"pythonhashseed\"])
print(sorted(p[\"marl_env\"].items()))"'
```

`marl_env` must hold exactly the §12 set for an S1 invocation
(`MARL_TRANSPORT_POWER`, `MARL_RELAY_STATE`, `MARL_RELAY_REPOINT`,
`MARL_POSTCARD_ALWAYS`, `MARL_REWARD_PROFILE`, `MARL_RELAY_ENTROPY_FLOOR`,
`MARL_STEER_MODEL`, `MARL_POOL_WORKERS`, `MARL_EVAL_CHECKPOINT`,
`MARL_CHAIN_INV`) and nothing else; anything else is a stop.

e.4 **The restart rule (pre-registered; enforced by the launcher).** A
FAILED invocation is re-run exactly once, unchanged (same job script, same
env file, same tree, same seeds and arms); if the watchdog already restarted
it, the chain does not add a second. A second failure stops the chain with
the escalation text: do NOT analyse a partial set, do NOT change arms or
seeds to fill the gap, do NOT substitute a pass from another invocation or
tree. An invocation recorded OK is never re-run; if its artefact no longer
matches the ledger's sha256 the chain stops (`artifact_changed`): escalate.
A training replicate that dies restarts from episode 0 with the same torch
seed when the segment is re-issued; replicates that succeeded are not re-run.
`--invalidate`, `--waive-dep` and `--waive-order` are operator actions that
the ledger records; use none of them without writing down why.

e.5 **The watchdog and the top-level launcher.** The watchdog restarts a
registered job that died without its done marker at most once. If it
restarts a segment while an invocation of that segment is still running, the
new launcher's preflight sees that job as a leftover and waits (within the
budget); once it exits, the invocation — which has no END record — is run
again from the start (the result is deterministic; the cost is its wall
time). If the budget runs out first, the segment stops and is re-issued by
hand after the job ends.

e.6 **Shedding**, only if the window demands it, only in the plan's order,
always explicit and recorded: `--shed sensitivity`, then
`--shed sensitivity,adapt_freeze`, then
`--shed sensitivity,adapt_freeze,untrained`. None of them gates the primary
analysis (S7).

e.7 **Never** sync, edit a tracked file on the server, or start another
simulator in `~/MARL_run_9` before the last `CHAIN END profile=real rc=0`.

## f. After the chain: the pre-registered analyses the chain does not run

**Since FIX-PREFREEZE item 2 the chain computes all three** (stage S6h
runs the `PYTHONHASHSEED=1` repeat; S12 runs the S8 check and the full
report, `confirmatory/final_s8_hash.json` and
`confirmatory/final_cf_report_full.json`, PREREGISTRATION_R2 §10), and the
rehearsal exercises them. The commands below are kept as the record of
what the stage does and as a manual fallback for a stage the operator
shed; they neither change nor re-run any chain invocation. Read
`tail -20 logs/chain_real/s12_secondary.log` for the presence lines
(`S8 computed`, `analysis S12 computed`, `secondary S7 computed`,
`primary reproduced: SUMMARY ... IDENTICAL`).

f.1 **S8, the `PYTHONHASHSEED=1` side** (one `marl_static` invocation on the
two lowest seeds of the block, both scenarios, 4 passes, the selected
checkpoint, the S6 environment):

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
read A B < <(python3 -c "import sys; sys.path.insert(0, 'confirmatory'); import chain_plan as c; s = sorted(c.PROFILES['real']['cf_seeds']); print(s[0], s[1])")
export S8_SEEDS="$A,$B"
nohup bash -c 'source confirmatory/env_real/s6_cf_b0.sh && export PYTHONHASHSEED=1 MARL_CHAIN_INV=post_s8_hash1 && nice -n 19 python3 run_timeline_comparison.py --seed-list "$S8_SEEDS" --arms marl_static --no-figures --out final/post_s8_hash1.pkl; echo "# exit=$?"' \
  > logs/freeze/post_s8_hash1.log 2>&1 < /dev/null &
EOF
```

f.2 **S8 comparison** (the `PYTHONHASHSEED=0` side is `s6_cf_b0`):

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
read A B < <(python3 -c "import sys; sys.path.insert(0, 'confirmatory'); import chain_plan as c; s = sorted(c.PROFILES['real']['cf_seeds']); print(s[0], s[1])")
python3 tools/hash_invariance_check.py output/final/s6_cf_b0.pkl output/final/post_s8_hash1.pkl \
  --arms marl_static --seeds "$A,$B" --require-decision --out confirmatory/final_s8.json; echo "EXIT=$?"
EOF
```

Pass: IDENTICAL, `EXIT=0`, `same_hashseed` false, `n_compared` > 0.

f.3 **The full report** (the chain's S6 pickles plus the S9 arm for S7, the
four S8 controls for S12, and the S8 JSON):

```bash
ssh -o BatchMode=yes <compute-server> 'bash -s' <<'EOF'
cd ~/MARL_run_9 || exit 1
P="python3 confirmatory/chain_plan.py --profile real"
PK="$($P --artifacts S6),$($P --field s9_freeze artifact)"
UN=""
for s in 1234 2345 3456 4567; do UN="${UN:+$UN,}untrained_s$s=$($P --field s8_untrained_s$s artifact)"; done
MAN=$($P --field s0_manifest artifact)
PYTHONHASHSEED=0 nice -n 19 python3 confirmatory_analysis.py --marl-arm marl_static \
  --pickles "$PK" --manifests "$MAN" --untrained "$UN" --s8 confirmatory/final_s8.json \
  --out confirmatory/final_cf_report_full.json > confirmatory/final_cf_report_full.txt 2>&1
echo "EXIT=$?"
python3 tools/compare_reports.py confirmatory/final_cf_report.json confirmatory/final_cf_report_full.json; echo "SAME_AS_CHAIN=$?"
EOF
```

Pass: `EXIT=0` (every hard assert, now including the S9 arm and the four
controls) and `SAME_AS_CHAIN=0`: every leaf of the chain's primary report is
reproduced; only the S7-secondary, S8 and S12 entries are new (reported as
NOTE). A mismatch is a defect, not a result.
