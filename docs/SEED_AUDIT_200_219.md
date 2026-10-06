# Seed audit — topology seeds 200–219 (fresh confirmatory set)

**Read §6 first.** §1–§5 below are the record of 2026-09-23, whose verdict
(CLEAN, "no seed in 200–219 has ever been generated, evaluated, trained on
…") was true on that date and is NOT true any more: the reduced rehearsals
of 2026-09-24 ran pipeline passes on seeds 200–203. §6 is the audit
measured at the final head on 2026-10-01 (and the measured run of
2026-09-30 that preceded it), with every hit classified, the disclosure
paragraph and the decision on the block as it stands.

Date: 2026-09-23 (first record 2026-09-22; revised three times on
2026-09-23, after each of the three reviews of that day). Host: <compute-server>
(every scan ran there, `nice -n 19`, `PYTHONHASHSEED=0`, read-only, one
process, no pool worker). Tool: `tools/seed_audit.py` at commit `d2dc46a`
(branch `trackB`, sha256
`b43b8c839fd05218a656f7d0d5653a7bba17d3e1950316e452beaa1627798ade`,
identical on laptop and server; its unit tests
`tests/test_wp12_seed_audit.py`, 33 tests, all passing on the server in the
same commit). Recorded run: report
`~/MARL_run_9b/output/seed_audit_200_219.json`, log
`~/MARL_run_9b/logs/seed_audit_200_219.log`, driver
`tools/run_seed_audit.sh` — **in this tree and committed**, so the
command that ran is the command that can be reviewed (§2.1). Earlier runs
are superseded: those of 2026-09-22 are kept beside the report
(`*.run1.*`, `*.crashed.log`, `*.run2.*`); the run of 2026-09-23 08:31 UTC,
whose driver was the uncommitted `~/seed_audit_evidence/run_audits.sh`
(§2.1), and the run of 10:30 UTC that the previous revision recorded were
both **overwritten** at the fixed report path by the runs below — their
figures survive only in this file's git history, which is why each revision
quotes the whole summary block rather than pointing at the file. The
contaminated attempt of 11:42 UTC is described in §1 below. All of them
agree on the substance.

**What the review of 2026-09-23 changed.** The earlier record rested on a
tool whose verdict looked only at the hit lists: a missing root, a failed
pickle or checkpoint, an aborted `--pt` scan and a `git log` that failed all
left the verdict CLEAN and the exit status 0, so an audit that scanned
*nothing* passed the freeze gate. Coverage is now half the verdict (§2.3),
`--git` checks git's return code (the `.git` of `~/MARL_code` is an empty
directory and `git log` fails there), the trackB history is scanned from a
captured dump instead of being silently skipped, the 160-character
value truncation of the **text** path is gone (that round did not touch
the container path, and this record claimed otherwise until the second
review below), the large-JSON regex path reads seed *lists*, file names of
this study's own outputs no longer enter the inventory, and two trees that the frozen root
list had missed (`~/MARL_run_9c`, created 2026-09-23 10:14, and
`~/MARL_watch`) are scanned. The substantive finding did not change.

**What the second review of 2026-09-23 changed.** Six further defects, four
of them in the audit's own safety nets — checks that reported success
having verified nothing, which in a study whose whole point is to be
checkable is worse than a missing check.

1. **The container path still read the verdict off a truncated copy.** The
   window test for JSON, pickles and checkpoints re-read the *report's*
   summary of a value: `str(v)[:120]`, and then at most the first 50
   integers of that already-cut string. A seed named past character 120 of
   a seed-key string — `"seed_spec": "… 80-99 superseded, 205
   confirmatory"`, 136 characters with `205` at offset 120 — was dropped
   with no `errors` entry, no `long_values` entry and no coverage warning,
   in both a JSON file and a pickle, and the tree read CLEAN with exit 0.
   The 160-character removal announced in the paragraph above had been
   applied to the text path only, so this record's own claim was wrong for
   the path that reads the result data. `walk_seeds` now accumulates every
   integer it meets and the hit test reads that set; the 400-integer /
   120-character summaries remain, but only in the report.
2. **The large-JSON regex fallback was weaker than the parsed path, and
   said so nowhere.** A string-valued seed key was not matched at all and a
   nested list lost everything after its first `]`; such a file was not
   even listed as carrying seed content, so nothing showed it had been
   scanned with less power. It now reads a string, a number and a list
   literal with its nested lists, every JSON file records which path it
   took (`json[<file>].scan`), and a test pins the two paths against each
   other.
3. **Symbolic links were dropped silently** by a bare `continue` while §2
   claimed nothing but the skipped directory names was left out, and
   `check_home_roots` excluded a symlinked `$HOME/MARL*` tree from both the
   required roots and the walk: a future `~/MARL_run_10 -> /data/…` would
   have been neither required nor scanned, coverage ok. Links are now
   counted and resolved, a broken or outside-the-roots target is a named
   warning, and a symlinked `$HOME/MARL*` directory is a root like any
   other.
4. **An empty root was a warning, not a failure**, so a tree moved away or
   an unmounted path passed with exit 0 whenever the other roots still met
   the *aggregate* floors — and the slack over them is larger than several
   whole roots. It is now a coverage failure, and per-root floors
   (`--min-per-root`, `--min-root NAME=N` from
   `tools/seed_audit_floors.txt`) close the rest of that gap.
5. **§2.1 misdescribed its own driver**, which enumerated fourteen literal
   roots while the text said "verbatim … by pattern" — the one thing
   `docs/PREREGISTRATION_R2.md` §13 forbids at a freeze. The driver is now
   `tools/run_seed_audit.sh`, in the tree and committed.
6. **§2.2 understated the evidence gap** between the history dump and the
   scanned tree.

Items 5 and 6 are corrected in place below, each with the wrong sentence
quoted; items 1–4 each carry a test in `tests/test_wp12_seed_audit.py`.
The run of §1 was re-done from
scratch with the corrected tool and the in-tree driver. The substantive
finding did not change.

**What the third review of 2026-09-23 changed.** Six further defects. The
first is the most serious this record has carried, because the check it
broke is the one the pre-registration leans on hardest.

1. **A seed in a torch checkpoint never reached the verdict** (§2 step 5).
   `scan_pt` computed the hit and wrote it to the report; `finish()` built
   `window_hits_in_result_data` from the pickle and JSON sections only. A
   safety net over 2 615 files could not fail: the audit printed "CLEAN —
   no seed in 200–219 in any result, name, script, log or history",
   `result data hits: none`, `coverage: ok`, and exited 0 while the same
   run's report carried `pt[<file>]['hits'] == [208]`. No §5 gate item
   would have caught it, `docs/PREREGISTRATION_R2.md` §5 and
   `docs/DECISIONS.md` both asserted the opposite, and the test that looked
   like it covered the path passed for an unrelated reason — a seed-named
   file in the same fixture directory produced a *file-name* hit, so
   deleting that file made the assertion fail. Fixed, with the checkpoint's
   seeds also entering the inventory of §4, and demonstrated on a planted
   seed: `VERDICT: TOUCHED`, `result data hits: [208]`, exit 3.
2. **The container walk still had two silent caps** (§2 step 3): it
   returned at depth > 6 and stopped after the 1000th container element of
   a list, dropping a seed key below either with no `errors` entry, no
   coverage warning and no coverage failure — which left the *parsed* JSON
   path weaker than the large-JSON regex fallback, the inverse of the
   divergence item 2 of the previous review fixed. Both caps are gone; the
   one cost bound that remains is recorded as an error and refuses the
   verdict.
3. **§2.2's "one command" could not see the gap it described.** `git log
   --stat 22609ec..trackB` lists `trackB` commits only, and the same
   paragraph identified the other-track commits as the part of the gap that
   matters; the count was wrong (six, not four) and two of the six were
   named nowhere. Corrected with an all-refs command and the full table.
4. **The driver could lose every per-root floor in silence**
   (`tools/run_seed_audit.sh`): a `while read … done < "$FLOORS"` loop
   under `set -u` and no `set -e` left the floors array empty if the file
   went missing, and the run continued with no `--min-root` at all and
   could still exit 0. The tool now reads the file itself
   (`--min-root-file`), records it, and refuses the verdict if it cannot;
   the driver also fails loudly (`exit 2`).
5. **§2.2's one checkable pointer had stopped being true**: it named the
   live `~/MARL_run_9b/.synced_head` as `22609ec`, and the tree was
   re-synced afterwards, so the reader following §5 item 5 got a mismatch
   and the HEAD equality was no longer reconstructable. The value is now
   frozen in the report (`synced_head`).
6. **The tool's docstring contradicted §2 step 2**, listing "JSON files too
   large to parse" among the text-scanned files; an oversized JSON takes
   the regex fallback and is never line-scanned.

Items 1, 2, 4 and 5 each carry a test; items 3, 5 and 6 are corrected in
place. Both runs of §1 were re-done from scratch with the corrected tool
and the committed driver, at a fresh history dump taken at the synced HEAD.
The substantive finding did not change.

## 1. Verdict

**No seed in 200–219 has ever been generated, evaluated, trained on,
named in a script, log, manifest or file name, or committed.** The range is
untouched and remains the confirmatory set of `docs/PREREGISTRATION_R2.md`
§5. No alternative range is needed.

Recorded run (tool at `d2dc46a`, driver `tools/run_seed_audit.sh`, exit
status 0):

```
Seed audit 200-219 on <compute-server>, 2026-09-23 11:58:09 UTC -> 2026-09-23 12:05:17 UTC (428 s)
files 7206 | text scanned 3322 | JSON parsed 352 + regex-scanned 13 | pickles 236 (loaded 215, duplicates 21, failed 0) | checkpoints 2615 (failed 0, skipped 0) | binary skipped 179

Per root: MARL_run_2=1090, MARL_run_3=1052, MARL_run_4=120, MARL_run_5=827, MARL_run_6=158, MARL_run_7=114, MARL_run_8=118, MARL_run_9=780, MARL_run_9b=329, MARL_run_9c=154, MARL_code=282, MARL_code_remote=690, MARL_perf=1478, MARL_watch=14
Per-root floors from <server-home>/MARL_run_9b/tools/seed_audit_floors.txt: 14 parsed, sha256 04ab537a5b47d0b9 (applied)
Tree .synced_head: d2dc46a (2026-09-23 11:40:16 UTC, read)
Skipped directory names: .git, .ipynb_checkpoints, .venv, __pycache__, audit_history, node_modules, site-packages, tmp_tests, venv
Symbolic links met (never followed): 50 (6 directories); targets outside every scanned root or broken: 0
$HOME/MARL* directories: 14 (MARL_code, MARL_code_remote, MARL_perf, MARL_run_2, MARL_run_3, MARL_run_4, MARL_run_5, MARL_run_6, MARL_run_7, MARL_run_8, MARL_run_9, MARL_run_9b, MARL_run_9c, MARL_watch)
Git history: 0 root(s) scanned of 1 with a .git entry; dumps: trackB_git_log.txt (90 commits, 101310 lines, sha256 2d91820893487f84)

VERDICT: CLEAN - no seed in 200-219 in any result, name, script, log or history
  result data hits   : none
  text hits (non-plan): none
  file-name hits     : none
  git-history hits   : none
  classified false positives (--false-positive): [200]
  coverage           : ok
    ~ git log failed in <server-home>/MARL_code (rc=128) and the root is declared --git-no-repo: its history is NOT auditable
```

`errors: []`; `verdict.coverage_ok: true` with no failure; every pickle and
every checkpoint opened, and **every checkpoint's hits now counted**
(`pt`: 58 entries carrying seed metadata, 0 of them with a hit — §2 step 5,
the defect this run is the first to be able to fail on); every one of the 14
`$HOME/MARL*` directories among the roots (given by pattern, §2.1); every
aggregate **and** per-root count floor met, the floors read by the tool from
`tools/seed_audit_floors.txt` (sha256 `04ab537a5b47d0b9…`, 14 parsed,
`min_root_file.ok: true`); all 50 symbolic links resolve inside the scanned
roots. The one warning is the empty `.git` directory of `~/MARL_code`
(§2.4), which §5 item 2 names as the only warning a freeze may carry.

**The fallback range 220–239** was audited on its own immediately afterwards
(`bash tools/run_seed_audit.sh 220 239`, same roots, same tool, same dump,
log `~/MARL_run_9b/logs/seed_audit_220_239.log`, 12:09:45–12:17:12 UTC,
7 226 files, 2 615 checkpoints, 0 `pt` hits): **CLEAN, exit 0,
`errors: []`, coverage ok** — no result-data, text, file-name or
git-history hit, and no classified false positive either. A clean 220–239
run is only needed if 200–219 is ever compromised, and it is repeated then
from the frozen tree like the 200–219 run.

**One superseded run, and why it is named here.** The first attempt of this
revision (11:42:06–11:48:47 UTC) reported `VERDICT: TOUCHED`, result-data
hits `[205, 207, 208]` and exit 3. Every one of those hits came from
**this round's own defect proofs**, written to
`~/MARL_run_9b/output/r3_proof/` — inside a scanned root, under names that
the planning rule (any path containing `seed_audit`) does not cover — so
the reports of the planted 208 checkpoint and the planted 205/207 deep and
wide JSON fixtures were read back as ordinary result data. No pickle and no
checkpoint of the real trees was among them. The proofs were moved to
`~/MARL_run_9b/audit_history/r3_proof/` (excluded from the walk with
`--skip-dir`, where the history dump lives) and the run above is the
re-run. It is recorded rather than quietly discarded for two reasons: it is
the rule of the tool's own docstring — an output that is neither
planning-named nor outside every scanned root contaminates the next run —
and it is an end-to-end demonstration that the fixed tool does refuse a
tree that carries a window seed, this time without anything planted for it
to find.

## 2. What was scanned, and how

Roots (14 directories — every directory matching `$HOME/MARL*`, which the
tool now enforces, §2.3): `~/MARL_run_2` … `~/MARL_run_9` (engine track),
`~/MARL_run_9b` (the synced HEAD of the local repository `run9_trackB`),
`~/MARL_run_9c` (the engine track's newer tree, created 2026-09-23 10:14),
`~/MARL_code`, `~/MARL_code_remote`, `~/MARL_perf` (performance track,
including its `tree*`, `perfB` and `base` copies) and `~/MARL_watch` (the
rerun watchdog's job, status and chain files). Directory names `venv`,
`.venv`, `.git`, `node_modules`, `__pycache__`, `site-packages`,
`.ipynb_checkpoints`, `tmp_tests` and `audit_history` are not walked, and
**symbolic links are not followed** — `os.walk` does not descend into a
symlinked directory and a symlinked file is not opened. Nothing else is
skipped. Every link met is now counted and resolved in the report
(`counts.symlinks`, `symlinks`, `symlinks_outside_roots`), and one whose
target is broken or lies outside every scanned root is a named coverage
warning, which §5 item 2 does not permit — so content reachable only through
a link cannot be missed silently. In this run all 50 links (six of them
directory links: `MARL_run_3/ca_v2/{sixg_sim,config}`,
`MARL_run_6/_orig_harness/{sixg_sim,config}` and
`MARL_perf/base/_orig_harness/{sixg_sim,config}`) resolve **inside** the
scanned roots, so their targets are walked on their own and no content is
missed. A symlinked `$HOME/MARL*` tree (`~/MARL_run_10 -> /data/…`) is a
directory like any other for the `$HOME/MARL*` check: it must be a root, and
the run is refused if it is not. `audit_history` holds the git-history dump
of §2.2, which must not be walked as ordinary file content.
`tmp_tests` is the unit tests' git-ignored scratch directory (D0.10): the
audit's own touched-tree test writes seed 210 there on purpose and removes it
again, so a concurrent test run must not make a real tree look touched.
`--skip-dir` can add names; every skipped name is recorded in the report.

Per file:

1. **File names** (every file): every integer token of the base name is
   checked against 200–219; tokens in seed-like positions (`seed<N>`,
   `_s<N>`, `_<a>_<b>.pkl|json|log|txt`) count as seed uses, other tokens
   are listed for inspection. Names of this study's own planning outputs
   (`seed_audit_200_219.json`, its logs) are bucketed as planning and do
   **not** enter the inventory of §4.
2. **Text** (`*.py *.sh *.log *.txt *.md *.yaml *.yml *.csv *.tsv *.sha256
   *.diff *.out *.nohup *.launcher *.ini *.cfg *.toml *.tex *.html *.js
   *.bak *.rst *.patch`, `.gitignore`/`.gitattributes`, files without an
   extension that contain no NUL byte; a JSON file above the parse limit is
   **not** text-scanned — it takes the fallback of step 3, and a JSON file
   that fails to parse is text-scanned and recorded as such):
   on every line that mentions `seed`, carries a `--seed-list`/`--seeds`/
   `--seed-range` spec, a `range(a, b)` call, or names a `*_<a>_<b>.ext`
   file, the seed values are extracted by pattern (key=value, `seed N`, CLI
   specs with comma lists and `a-b` ranges, `range()` bounds, seed-named
   files) and range-checked. Value strings are parsed up to 20 000
   characters and reaching that limit is a hard error, so no seed list can
   lose its tail (the earlier 160-character cap could). A `range(a, b)` on a
   line that never says `seed` is window-checked too, unless its loop
   variable names a tick, episode or graph position (`for tick in
   range(1, 361)`; the list is `BENIGN_LOOP_VARS` in the tool — generic
   names such as `i`, `k`, `s` stay counted): such loops go to their own
   bucket, listed in the report and never counted. Every other standalone
   200–219 token on a `seed` line is kept with its left context and
   classified (§3.4), so a value the patterns did not understand cannot
   hide. A line that merely **names one of this study's planning outputs**
   (`MARL_run_9b/logs/seed_audit_200_219.log`, which the watchdog's
   `status/HEALTH.md` quotes as the newest log of the tree) is planning for
   those values only; without that rule the audit's own log name, quoted
   elsewhere, would make its own window look touched.
3. **JSON** (parsed): every value under a key containing `seed` and every
   dict keyed by integers (result exports are keyed scenario → arm → seed;
   manifests list `topology_seed`), **at any nesting depth and at any
   position in a list** — see the depth and breadth note below, which until
   the third review of 2026-09-23 this sentence did not hold. A file above 64 MiB
   (`ce_results*.json`, `ce_results_A1*.json` and their copies) is not
   parsed; its **fallback** scan reads the same two things by regex — a seed
   key's value whether it is a **string** (`"seed_range": "200-219"`), a
   number or a **list literal, nested lists included**
   (`"seeds": [[200, 201], [202]]`), and every integer key — and the report
   records per file which path it took (`json[<file>].scan` = `parsed` /
   `regex-fallback` / `text (parse failed)`), with the fallback files listed
   in the summary. The earlier fallback matched only a bare integer or a
   list up to its first `]`, so a string-valued seed key was not matched at
   all and a nested list lost its tail — and the file was not even listed as
   carrying seed content, so nothing showed that it had been scanned with
   less power. The two paths are now pinned against each other by
   `tests/test_wp12_seed_audit.py`
   (`test_regex_fallback_matches_the_parsed_path`).

   The integers that decide the verdict are never read off the report's
   summary of a value, and this holds for **every** container path — JSON
   (step 3), pickles (step 4) and checkpoints (step 5) share one walk
   (`walk_seeds`), which collects every integer it meets in a set of its
   own. The summaries stay short by design (400 integers, 120 characters)
   and are the report's copy only; the hit test reads the value whole, and
   a **string** value is read with the same extractor a text line uses, so
   `"seed_range": "190-230"` names 190…230 rather than only 190 and 230,
   and a seed named at character 120 or at character 12 000 of a seed-key
   string is found. Until the second review of 2026-09-23 the test read
   the summary, and such a seed was dropped in silence; the two paths are
   now pinned by `test_seed_key_string_value_is_not_truncated` and
   `test_container_int_caps_do_not_hide_a_seed`.

   **Depth and breadth: the limit that remains, stated.** That walk has
   **no depth limit and no breadth limit**. Until the third review of
   2026-09-23 it returned at depth > 6 and stopped after the 1000th
   container element of a list, and a seed key below either cap was dropped
   with no `errors` entry, no `long_values` entry, no coverage warning and
   no coverage failure — the file was not even listed as carrying seed
   content. A 1440-tick harness run makes a list of more than 1000 per-tick
   dicts an ordinary shape, and the large-JSON regex fallback of this step
   has never had either limit, so the **parsed** path — the stronger one by
   construction — was the weaker of the two: the inverse of the divergence
   the previous round's defect 2 fixed. The walk is now iterative (no
   nesting depth can exhaust the interpreter stack) and remembers the
   containers it has entered by identity, so a cyclic or shared structure
   is walked once and cannot loop. The **one** bound left is a cost bound,
   `WALK_NODE_LIMIT` = 4 000 000 distinct containers in a single file;
   reaching it appends an `errors` entry naming the file and the path, which
   refuses coverage (exit 4) exactly as the text path's 20 000-character
   `VALUE_CHARS` limit does. No run of this study has come near it. Silence
   is the one option the premise of this audit rules out. Pinned by
   `test_container_walk_has_no_depth_or_breadth_cap` (a seed at list element
   1500 and at dict depth 9, in a JSON **and** a pickle, with the parsed and
   the `--json-max`-forced fallback paths asserted equal) and
   `test_container_walk_survives_a_cycle_and_records_a_stopped_walk`.
4. **Pickles** (`*.pkl`, 229 files): loaded with a class-stubbing
   `Unpickler` (no engine import needed; any class becomes a placeholder
   whose state is still walked), deduplicated by sha256 (21 byte-identical
   copies), and every int-keyed dict **at any depth** plus the `seeds` and
   `provenance` entries are collected (the walk's depth-6 cut is gone; see
   the depth and breadth note in step 3). A pickle that fails to load
   refuses the verdict.
5. **Checkpoints** (`*.pt`, `--pt`): the 2 615 files that are not per-agent
   adapted copies (`*_seed<N>.pt`, whose seed is in the name and is covered
   by step 1) are opened with `torch.load` through the same stubbing
   unpickler and their metadata is searched for keys containing `seed`
   (`torch_seed`, `canonical_seed` = 42, provenance `seed_list` = None; none
   in the window). A checkpoint that fails to load, and an abort of the
   scan because torch is unavailable, refuse the verdict.

   **A checkpoint hit refuses the verdict.** A checkpoint is result data,
   exactly as a pickle and a JSON export are: its hits are part of
   `verdict.window_hits_in_result_data` and its seed values are part of the
   inventory of §4. Until the third review of 2026-09-23 neither was true.
   `scan_pt` computed `pt[<file>]['hits']` and wrote it to the report, and
   `finish()` built the data-hit union from the pickle and JSON sections
   **only**, so a seed of the window in a checkpoint's metadata was printed
   in the report, printed in the Markdown summary, and then dropped on the
   way to the verdict: the run announced "CLEAN — no seed in 200–219 in any
   result, name, script, log or history", reported `result data hits: none`
   and `coverage: ok`, and exited 0. This is a safety net over 2 615 files
   that cannot fail, and no gate item of §5 would have caught it: items 1
   and 6 name the result-data, text, file-name, git-history, tick-loop and
   `broad_tokens` buckets and never the checkpoint list. Demonstrated on the
   reviewer's own fixture — a directory holding one file,
   `policy_final.pt` = `{'metadata': {'topology_seed': 208, 'torch_seed':
   1234}, 'state': {}}`, with no seed-named file, no text and no JSON — which
   now reports `VERDICT: TOUCHED`, `result data hits: [208]` and **exit 3**
   where it reported CLEAN and exit 0. Pinned by
   `test_checkpoint_metadata_is_scanned_with_pt`, which asserts
   `window_hits_in_result_data == [208]` with the checkpoint in a directory
   containing no seed-named file — the previous version of that test
   asserted only `not clean`, and passed because `adapted_seed209.pt` sat
   beside it and produced a **file-name** hit; deleting that file made the
   assertion fail.
6. **Git history** (`--git`, `--git-log`): see §2.2 and §2.4. Every line of
   a history is attributed to the file of its `diff --git` header, so the
   diffs of the planning documents are planning exactly as the files are,
   and a launcher's history is not.

Planning documents of this study (`docs/RERUN_SPEC.md`, the
pre-registration, the compute plan, the handoff, `docs/WP9_ANALYSIS.md`,
this file, the audit tool, its test and its own outputs) name the range on
purpose; hits in them are reported in a separate "planning" bucket and never
count as uses.

### 2.1 Commands

The recorded run is the driver `tools/run_seed_audit.sh`, which is **in
this tree and committed**: the command that ran is therefore the command
that can be reviewed, and it cannot drift from this section unnoticed. It
is invoked as

```
# 1. on the laptop: sync the committed tree, then capture the trackB
#    history AT THE SYNCED HEAD (git only; the repository lives here,
#    every scan runs on the server)
SYNC_TREE=run9_trackB SYNC_REMOTE=MARL_run_9b bash sync_run9.sh
cd <worktree run9_trackB>
git -c core.autocrlf=false log --all -p --no-color | ssh -o BatchMode=yes <compute-server> \
  'mkdir -p ~/MARL_run_9b/audit_history && cat > ~/MARL_run_9b/audit_history/trackB_git_log.txt'
git -c core.autocrlf=false log --all -p --no-color | sha256sum   # must equal the server's
git rev-parse --short HEAD                                       # record it with the verdict

# 2. on <compute-server>: the driver, from the tree
ssh -o BatchMode=yes <compute-server> 'bash ~/MARL_run_9b/tools/run_seed_audit.sh 200 219'

# 3. unit tests of the tool (server)
ssh -o BatchMode=yes <compute-server> 'cd ~/MARL_run_9b && \
  PYTHONHASHSEED=0 nice -n 19 python3 tests/run_tests.py wp12_seed_audit'
```

and the tool invocation the driver builds is

```
cd ~/MARL_run_9b
export PYTHONHASHSEED=0
ROOTS="$HOME/MARL_run_* $HOME/MARL_code $HOME/MARL_code_remote $HOME/MARL_perf $HOME/MARL_watch"
nice -n 19 python3 tools/seed_audit.py --lo 200 --hi 219 --roots $ROOTS \
  --out output/seed_audit_200_219.json --pt --git \
  --git-no-repo "$HOME/MARL_code" \
  --git-log "$HOME/MARL_run_9b/audit_history/trackB_git_log.txt" \
  --skip-dir audit_history \
  --false-positive "seed 42, 200 island ticks" \
  --min-files 6077 --min-text 2455 --min-json 224 --min-pickles 184 --min-pt 2563 \
  --min-per-root 10 --min-root-file tools/seed_audit_floors.txt \
  --tree "$HOME/MARL_run_9b" \
  -v > logs/seed_audit_200_219.log 2>&1
```

The per-root floors are passed as a **file the tool reads itself**, not as
`--min-root NAME=N` arguments the driver expands: the tool records the
file's path, sha256 and every floor it parsed under `min_root_file` in the
report, and refuses the verdict if it cannot read or parse it (§5 item 3).
`--tree` names the tree whose `.synced_head` is copied into the report
(§2.2). The driver still guards the file before invoking the tool
(`[ -r "$FLOORS" ]` and a non-zero count of `NAME=N` lines, `exit 2`
otherwise), so a floors file that goes missing fails loudly at both levels.

**Roots are given by pattern, never as a literal list.**
`docs/PREREGISTRATION_R2.md` §13 requires it at a freeze: the shell expands
the pattern at the moment of the run, so a tree created after the command
was written (`~/MARL_run_9c` was) is included automatically and the tool's
own `$HOME/MARL*` check can only pass. Note that the report's `argv` records
the *expansion* (fourteen absolute paths) and therefore cannot by itself
show whether a pattern or a literal list was typed — which is exactly why
the driver has to be a committed file rather than a command quoted in a
document.

**Correction to the previous revision of this section.** It said: "The
recorded run is the driver script `~/seed_audit_evidence/run_audits.sh`,
which is the command below verbatim (roots enumerated by pattern, so a tree
created later is included automatically and the tool's own `$HOME/MARL*`
check can only pass)". That was wrong twice over: the driver lived outside
the repository, and it enumerated all fourteen roots literally
(`R="$HOME/MARL_run_2 $HOME/MARL_run_3 … $HOME/MARL_watch"`), so the only
evidence run of the record did the one thing §13 forbids at a freeze while
the document described it as doing the opposite. The run it described is
superseded by the run of §1, which used the in-tree driver and the pattern;
`~/seed_audit_evidence/` is kept unchanged as a record of the superseded
runs and is no longer used.

**Where the report, the log and the dump live.** The report and the log are
written inside a scanned root (`~/MARL_run_9b/output/`, `logs/`), while the
tool's docstring says a report belongs "outside the scanned roots". That is
the general rule; the tool states the exception this study relies on and
the exception is the reason the paths are safe: a path containing
`seed_audit` is planning-classified, the report is written after the walk
so it cannot influence its own run, and the next run buckets it as planning
rather than as a seed use (§3.3, §3.4). The `--git-log` dump has **no** such
exception — it carries the non-planning diffs of ordinary source files,
which the walk would read as file content — so it lives in
`~/MARL_run_9b/audit_history/`, which `--skip-dir audit_history` excludes
from the walk and which is read only through `--git-log`. The previous
revision kept the dump in `~/seed_audit_evidence/`, outside every root,
which is equally safe; it moved into the tree so that the driver needs no
path outside `~/MARL_run_9b`.

The one `--false-positive` entry is the docstring classified in §3.2; the
tool reports it in its own bucket (`window_hits_classified_false_positive`)
so the allowlist is visible in the report. `--git-no-repo` declares
`~/MARL_code`, whose `.git` is an empty directory (§2.4). The aggregate
`--min-*` floors are the counts of the first record of 2026-09-22 (6 077
files, 2 455 text files, 224 JSON files, 184 pickles, 2 563 checkpoints);
a run that scans fewer than any of them is refused. They cannot see a
single root that lost its content — the slack over `--min-files` is about a
thousand files, more than the whole of `~/MARL_run_9c` or `~/MARL_watch` —
so `--min-per-root 10` and the per-root floors of
`tools/seed_audit_floors.txt` (`--min-root NAME=N`, the counts of the
recorded run, with `~/MARL_run_9c` held at 100 while the engine track
rewrites that tree) are passed as well; a refusal there means a root
shrank, and the cause is to be found before any floor is lowered. At
freeze 1 every floor is re-set from the frozen counts.
`--no-home-check` exists only for an ad-hoc single-tree audit and is
**never** passed at a freeze.

Exit status: **0** CLEAN, **3** TOUCHED (a hit), **4** INCOMPLETE (coverage
refused).

A first pass with plain `grep`, before the tool existed, shows why a
structured scanner was needed:

```
nice -n 19 grep -rInE --exclude-dir=venv --exclude-dir=.git --exclude-dir=node_modules \
  --exclude-dir=__pycache__ -e "seed" ~/MARL_run_2 ... ~/MARL_perf \
  | grep -cE "(^|[^0-9.])(20[0-9]|21[0-9])([^0-9.]|$)"
# -> 2593 lines, almost all training-episode indices ("Worker Episode 205
#    [seed=134 ...]"), event ticks ("sev=200", "rescue=208") and link counts
```

### 2.2 The local repository

The local repository is the git worktree `run9_trackB` (branch `trackB`,
shared history with `master` and `trackAgent`). Its committed tree is what
`~/MARL_run_9b` holds (`sync_run9.sh` extracts `git archive HEAD`), so steps
1–5 cover its files; its **history** is covered by the dump. The dump of the
recorded run was captured at trackB HEAD `d2dc46a` — **the same HEAD the
scanned `~/MARL_run_9b` was synced from** — 90 commits over all refs,
101 310 lines, sha256
`2d91820893487f840fc65cef6c8383b69bc32614c1d47cae63151eb48822042f`,
identical on laptop and server, and the tool scanned it line by line with
the text patterns, attributing every line to its `diff --git` file.

That HEAD equality is now **recorded with the verdict** and not left to a
live file: the report carries a `synced_head` field
(`{path, value, mtime_utc}`) that the tool copies from `$TREE/.synced_head`
at the start of the run, and the Markdown summary prints it. The previous
revision of this section instead pointed the reviewer at the live file
(`~/MARL_run_9b/.synced_head` = `22609ec`), written in the present tense as
something a reader could check. It stopped being true: the tree was re-synced
after the record was written, so by 14:00 local the file read `083f230`, and
a reviewer following §5 item 5 got a mismatch and could not confirm the
equality at all — nothing else in the tree pinned the audit-time value. The
claim was true of the historical state (the run of 2026-09-23 10:30–10:42
UTC did scan a tree synced from `22609ec`, from a dump captured at
`22609ec`), but it was no longer reconstructable from the server. §5 item 5
now compares the report's recorded field.

**The gap between the dump and this record, stated exactly.** A record is
written after the run it records, so the repository always gains commits
after the dump is captured, and the gap is never empty. At capture the dump
held **every** commit the repository had — 90 of 90, over all three refs
(`git rev-list --all | wc -l` = 90 = `grep -c '^commit '` of the dump) — so
the gap is exactly the commits made after it, which is at minimum this
record's own commit and whatever the other two tracks land meanwhile.

The command that enumerates that gap must look at **all refs**, because two
of the three tracks do not commit on `trackB`:

```sh
# every commit the repository has that the dump does not (exact, by construction)
comm -13 <(grep '^commit ' ~/MARL_run_9b/audit_history/trackB_git_log.txt \
             | awk '{print $2}' | sort) \
         <(git rev-list --all | sort)                     # in the worktree
# the same set, conveniently, with files:  <dump time> = the report's `started`
git log --stat --all --since='<dump time>' --not d2dc46a
```

`git log --stat d2dc46a..trackB` is the **trackB-only subset** of that set,
and quoting it as "the whole claim is one command" — as the previous
revision did, with `22609ec..trackB` — is precisely the false smallness this
paragraph exists to prevent: it cannot list the other-track commits that the
same paragraph identifies as the part of the gap that matters. Run against
the previous dump it returned three docs-only `trackB` commits, and a
reviewer who took it at its word would have concluded that the whole
post-dump gap was documentation.

How large the difference is, measured once: run at 2026-09-23 15:31 local,
just after this revision was committed, the `comm` command above returned
**13** commits absent from the dump — `e4bf26f` on `trackB` (this record)
and twelve on `master` and `trackAgent` (`WP6-FIX3`, `WP-DET`), while
`git log --oneline d2dc46a..trackB` returned **one**. The figures are a
dated observation and will differ whenever it is re-run; the point they fix
is the ratio, not the number. The tool that produced the verdict of §1 is
**not** in that gap: `d2dc46a` was the dump's HEAD, so `tools/seed_audit.py`
at the audited sha256 is inside the scanned history, and every commit after
it touches `docs/` in this worktree or belongs to another track.

**The gap of the previous dump (HEAD `22609ec`, 79 commits), named in
full.** The previous revision said "Four did here" and named two of them
plus an ellipsis. Six followed, all after **both** runs had finished
(10:42:12 and 10:51:09 UTC):

| commit | ref | time (UTC) | files | what it is |
|---|---|---|---|---|
| `7b11a90` | `trackAgent` | 10:52:00 | `docs/DECISIONS.md` | the agent track's own round (`WP6-FIX2`) |
| `82942b1` | `master` | 10:52:29 | `docs/DECISIONS.md` | the engine track's own round (`WP2-VERIFY`) |
| `f238c6a` | `trackB` | 10:54:18 | `docs/DECISIONS.md`, `docs/PREREGISTRATION_R2.md`, `docs/SEED_AUDIT_200_219.md`, `docs/TRACKB_HANDOFF.md` | this round's record |
| `3e25338` | `trackB` | 10:58:53 | `docs/DECISIONS.md`, `docs/SEED_AUDIT_200_219.md` | its validation block |
| `083f230` | `trackB` | 11:00:17 | `docs/DECISIONS.md`, `docs/SEED_AUDIT_200_219.md` | the revision that introduced the wrong command |
| `a2b0c03` | `master` | 11:02:45 | `docs/DECISIONS.md` | the engine track again (`WP2-VERIFY`) |

`a2b0c03` was named nowhere in the previous revision and `083f230` was
elided behind the ellipsis; the count "four" omitted both. The three
other-track commits belong to `master` and `trackAgent` and are the reason a
freeze audit is run when no track is writing (`docs/TRACKB_HANDOFF.md` §6);
their *files* reach the audit through the trees those tracks sync, their
*diffs* only through the next dump — which is this run's, and which contains
all six. None of the six adds code or data: every one of them touches
`docs/` only.

**Correction to the first revision of 2026-09-23.** Of its own dump
(HEAD `adcb5a5`, 38 commits, 91 508 lines) it said: "The only commits made
after it are this record and the pre-registration update it describes, both
planning documents." That was false. Seven commits followed `adcb5a5`
before that run's tree state:

| commit | time (+03:00) | files | planning only? |
|---|---|---|---|
| `e57c815` | 10:56 | `tools/seed_audit.py`, `tests/test_wp12_seed_audit.py` | no — the audit tool |
| `69e2c48` | 11:12 | `tools/seed_audit.py`, `confirmatory_analysis.py`, `docs/PREREGISTRATION_R2.md`, two test files | no — the tool **and** the analysis |
| `ba49e99` | 11:13 | `tests/test_wp9_prereg_arm.py` | test only |
| `12c17a9` | 11:14 | `tests/test_wp12_seed_audit.py` | test only |
| `8e916c6` | 11:15 | `tests/test_wp12_seed_audit.py` | test only |
| `aa824cf` | 11:29 | `tools/seed_audit.py`, `tests/test_wp12_seed_audit.py` | no — the audit tool |
| `cc134cb` | 11:30 | `tests/test_wp12_seed_audit.py` | test only |

The dump was captured at 10:54 and the run started at 11:31, so the scanned
tree was seven commits past the dump — the sentence also contradicted §1 of
the same revision, which named the tool at `cc134cb`. The **conclusion**
still held: a dump captured later over the same history scans CLEAN, and
the next revision's dump (`22609ec`, 79 commits, which contains all seven)
did so, as does this one (`d2dc46a`, 90 commits).
But the reader was told the evidence gap was smaller than it was, which is
the kind of claim this record exists to make checkable. §5 now requires the
dump to be captured at the HEAD the scanned tree was synced from, and the
commits between that HEAD and the appended verdict block to be named.

Result: **no commit ever added a seed of 200–219 to anything but a planning
document.** The window appears in the history of `docs/COMPUTE_PLAN.md`,
`docs/PREREGISTRATION_R2.md`, `docs/RERUN_SPEC.md`,
`docs/SEED_AUDIT_200_219.md`, `docs/TRACKB_HANDOFF.md`,
`docs/WP9_ANALYSIS.md`, `tests/test_wp12_seed_audit.py` and
`tools/seed_audit.py` (all planning, all bucketed as such) and nowhere else.
The manual scan of 2026-09-22 (`git log --all -p` split by `diff --git`
header, HEAD `32bf886`) reached the same conclusion and additionally
identified the two non-planning lines that name a window *token*: the
`sixg_sim/agent.py` docstring of §3.2 and
`sixg_sim/topology.py`'s `def generate_large_topology(num_nodes: int = 200,
seed: int = 42)`, a node-count default. Both were introduced by the baseline
commit `d94a936`.

### 2.3 Coverage is half the verdict

The verdict is CLEAN only when there is no hit **and** the run is complete.
A run is refused (verdict INCOMPLETE, exit 4) when any of these holds:

- a `--roots` entry does not exist, could not be walked, **or held no file
  at all** — an empty root is a failure, not a warning: a tree moved away, an
  unmounted path or a root whose whole content is skipped used to pass with
  exit 0 because the *aggregate* floors were still met by the other roots
  (this run has ~1 000 files of slack, more than the whole of `~/MARL_run_9c`
  or `~/MARL_watch`), and only a human reading a `~` line in the summary
  caught it;
- a per-root floor was not met (`--min-root NAME=N`, read by the tool from
  `tools/seed_audit_floors.txt` via `--min-root-file`, and
  `--min-per-root`), or a `--min-root` names a root that was not scanned at
  all, **or the floors file itself was missing, unreadable, unparsable or
  empty** — a run with no per-root floor is refused rather than passed;
- a directory matching `$HOME/MARL*` is not among the roots (unless
  `--no-home-check`) — the check that `~/MARL_run_9c` defeated;
- `errors` is non-empty (a file that vanished mid-walk, a JSON that failed
  to parse, a git invocation that failed, a value string at the parse
  limit, **a container walk that stopped at `WALK_NODE_LIMIT`** — §2 step 3);
- a pickle or a checkpoint failed to load, or `--pt` was aborted because
  torch is unavailable;
- `--git` was requested and no history was scanned at all;
- a `--git-log` dump was missing or empty;
- a `--min-*` coverage floor was not met.

Recorded as coverage **warnings** (which §5 item 2 allows only for the one
named case, so any other stops a freeze): the unauditable `.git` of
`~/MARL_code` (§2.4), and a symbolic link whose target is broken or outside
every scanned root (§2).

The report carries `verdict.coverage_ok`, `verdict.coverage_failures` and
`verdict.coverage_warnings`, and the Markdown summary prints them.

### 2.4 The one thing that cannot be audited

`~/MARL_code/.git` is an **empty directory**: `git -C ~/MARL_code log` fails
with "fatal: not a git repository", and no other server root is a repository
at all (`~/MARL_run_9b` is a `git archive` export, not a worktree). The
history of `~/MARL_code` therefore cannot be read, and the earlier claim
"git-history hits: none" was vacuous for it. This is now declared
explicitly with `--git-no-repo ~/MARL_code`, which records it as a named
warning in the report instead of letting it pass silently; the tree's
*files* are scanned normally (282 files). The only git history that exists
on any scanned path is the trackB worktree's, and it is scanned from the
dump of §2.2.

## 3. Findings

### 3.1 Result data, manifests, checkpoints, file names, history

No seed in 200–219 in any pickle, JSON file, checkpoint, file name or
commit. Nine file names carry a token of the window: eight are the audit's
own outputs (`docs/SEED_AUDIT_200_219.md`, `output/seed_audit_200_219*.json`,
`logs/seed_audit_200_219*.log`; all marked planning) and the ninth is
`MARL_perf/logs/prof200.log`, a 200-tick profiling run of the performance
track (topology seed 117, event seed 700001), not a seed.

**The checkpoint half of that sentence is enforced for the first time in
this run.** 2 615 checkpoints were opened, 58 of them carry seed metadata
(`torch_seed`, `canonical_seed` = 42, provenance `seed_list` = None) and
none carries a value in the window — `report.pt` has 58 entries and zero
non-empty `hits`. Every earlier run of this record produced the same 58
entries and the same zero, so the *finding* has never depended on the
defect; what changed is that the verdict would now refuse a non-zero one
instead of printing CLEAN over it (§2 step 5).

### 3.2 Text: one classified false positive

The only non-planning line in any script, log or document that names a
value of the window next to the word "seed" is the docstring

```
comparison environment (seed 42, 200 island ticks, 35 agents) the
```

in `sixg_sim/agent.py` (present in every copy of the engine: 23 lines across
the run trees, the run-9 reference exports, `~/MARL_run_9c` and the
performance track's trees). The scanner's comma-list rule reads "42, 200" as
a list; the text is "200 island ticks". It is passed as `--false-positive`
and appears in the report under `window_hits_classified_false_positive`.

### 3.3 Classified by the tool: tick loops and planning documents

- **Tick/episode loops** (`range_nonseed_benign`, 114 lines, not counted):
  `for tick in range(1, 361)` in `test_mission_config.py` and
  `for tick in range(1, 261)` (twice) in `test_relay_repoint.py`, in each
  copy of the engine's test files, plus a few lines of the audit's own test
  and tool (`tick`, `t`, `episode` loops used as fixtures). Loop variables:
  `tick` 105, `t` 7, `episode` 2 — no other name, which §5 item 6 requires.
- **Planning documents** (296 lines in 33 paths): `docs/RERUN_SPEC.md` §15
  ("Fresh confirmatory seeds 200-219 (audited unused)") in every copy of the
  run-9 tree (`~/MARL_run_9`, its three reference exports under `output/`,
  the eight `~/MARL_perf/tree*` copies, `~/MARL_run_9b` and `~/MARL_run_9c`),
  and in `~/MARL_run_9b` the pre-registration, the compute plan, the handoff,
  `docs/WP9_ANALYSIS.md`, this file, the audit tool, its test and the earlier
  audit logs; plus the corresponding files inside the trackB history dump.
  `~/MARL_watch/status/HEALTH.md`, which quotes the newest log file of each
  tree, is planning for that value by the rule of §2 step 2.

### 3.4 Other 200–219 tokens on "seed" lines

Histogram by left context (tokens that no seed pattern claimed):

```
{"Episode N": 929, "links=": 997, "rescue=": 387, "sev=": 112, "num_nodes": 31,
 "line number": 9, "range end": 7, "diff hunk": 1, "other": 725}
```

`Episode N` is a training-episode index (`Worker Episode 205 [seed=134
kind=…]`), `sev=`/`rescue=` are event ticks of training episodes,
`links=` is `peak_links=` in the harness's boost summary, `num_nodes` is the
default argument of `generate_large_topology(num_nodes=200, seed=42)`,
`line number`, `range end` and `diff hunk` are `path:200:`, `range(1, 200)`
and `@@ -200` contexts. Of the 725 "other" tokens, **722 sit in this study's
own material** (`~/MARL_run_9b`, any path naming `seed_audit`, and the same
files again inside the trackB history dump): its earlier audit logs and
unit-test logs, this file, the pre-registration, the audit tool and its
test. The three remaining tokens are

- `MARL_run_7/logs/untrained_s4567_80_99.log:103` — `TG-site
  nearest-neighbour spacing p10/median/p90 = 209/658/1047 m`, a distance in
  metres on an arm-log line that names the arm's seed;
- `MARL_perf/tree_ps2/logs/W16_quotakill.log:1831` — `640 Mbps@QAM256
  aggregate extra, shared over 209 Uu links`, a link count on an
  interleaved performance-log line that also names a seed;
- `MARL_run_9c/logs/inv_wp6fix2_xldet_e5715ec.log:19` (**new in the run of
  §1**, a file the engine track wrote into `~/MARL_run_9c` after the
  previous record) — `provenance: head=e5715ec (.synced_head)
  tree_sha256=8dc2c0212ed18498 schema=obs1-… PYTHONHASHSEED=0`: the token
  is the substring `0212` of a **hexadecimal tree digest**, on a line that
  counts as a seed line because it names `PYTHONHASHSEED`. Not a seed, and
  not a value any extractor claimed — which is what the `other` bucket is
  for. The tool is deliberately left as it is here: a rule that dropped
  digits found inside a hex run would also drop a real seed written next
  to one, and this bucket is reviewed at every freeze (§5 item 6).

Nothing remains unclassified.

## 4. Every seed ever used or referenced

From the recorded run (`all_seeds` in the report; ranges compressed):

| source | values |
|---|---|
| result pickles, JSON exports **and checkpoint metadata** (every int key of any dict **at any depth**, every value under a `seed` key, ≤ 100000) | 0–3, 7, 9–10, 42–66, 70–139, 1234, 2111, 2345, 3222, 3456, 4333, 4567, 5444 |
| scripts, logs, manifests, documents (values named next to "seed", CLI specs, `range()`; planning lines excluded), ≤ 1000 | 0–8, 42–66, 70–139, 149, 250, 300, 400, 600, 900, 999 |
| the same, > 1000 (event and torch seeds) | 1234–1235, 1800, 2111, 2345, 3222, 3456, 4333, 4567, 5444, 12345, 700001–700399 (odd), 750000–750004 (even), 4200001–4200399 (odd), 4250000–4250004 (even) |
| file names (`seed<N>`, `_s<N>`, `_<a>_<b>.ext`) | 1–2, 8, 50–66, 70–99, 1234, 2345, 3456, 4567 |

What they are:

- **Topology seeds actually used: 42–66 and 70–139.** Training topologies
  42–49 and 100–139 (`--seed-range "42-49,100-139"`, 48 seeds;
  `launch_authoritative.sh` and every other training launcher of
  `~/MARL_run_5`; the `train_on_comparison_summary.json` files list them),
  the harness's default held-out test seeds 50–54 (`DEFAULT_TEST_SEEDS`,
  printed by every training run as "Held-out eval seeds (excluded)"), the
  development set 50–59 and 70–79 (`benchmark_manifest.json`,
  `benchmark_manifest_70_79.json`, `ce_eval_*`, `dev_static_*`,
  `ext_eval_*`, `stress_*`), the validation set 60–66 (`val_eval_*`,
  `A1_val_*`, `hystval_*`), the superseded confirmatory set 80–99
  (`benchmark_manifest_80_99.json`, `cf_eval_80_99.pkl`, `cf_corun_*`,
  `cf_static_*`, `untrained_*_80_99`, `diag_*`/`fix_*` of run 8), and the
  smoke and tooling seeds 58 (behavioural smokes), 82 and 91 (WP0
  invariance matrix, memory measurements, performance-track profiling and
  the WP12 UE-transit probe), 87 and 98 (WP12 SDN probe), 117
  (performance-track profiling).
- **Event (scenario) seeds** `base_seed × 100000 + episode`: 700001–700400
  (`--base-seed 7`) for the evaluated replicates, 4200001–4200400 for the
  earlier rounds under the default base seed 42 (the logs print every
  second episode, hence the odd values); 750000–750004 and
  4250000–4250004 (five even values each) belong to the same two launch
  families (base seeds 7 and 42) and appear only in training logs.
  **Torch seeds** 1234, 2345, 3456, 4567 (1235 and 12345 occur on seed
  lines of older scripts and logs only, never in a result file; 1800 is
  the tick count of the WP0 memory run on a seed line; 2111, 3222, 4333 and
  5444 are the `torch_seed` of the performance track's four training
  replicates, `MARL_perf/tr2/out/M4_1..M4_4/o/train_on_comparison_summary.json`). The untrained controls' canonical topology
  seed is 42.
- **The `8` in the file-name column** is not a seed either: it is
  `~/MARL_run_9b/output/wp9_review/cf_s8.json`, a WP9 review fixture whose
  `_s8` suffix matches the seed-like name pattern. It is the only file name
  in any scanned root that yields it.
- **Int keys that are not seeds** (data column only): 0–3, 7 and 9–10 are
  agent, node and worker indices of
  `merged_results.json`/`merged_s1234_preview.json` and of the per-tick
  dicts in the harness pickles. They are listed because the scanner treats
  every int-keyed dict as a potential seed table. The previous revision
  listed a much larger set here (0–31, 4456, and "22 four- and five-digit
  integers (3936 … 79040)") and explained it as flow and node identifiers.
  Those were not keys at all: they were the **digits of floating-point
  values** under keys containing `seed`, read through `str(v)` by the old
  value path — `sd_across_seeds = 6.064136589897359` contributed 6,
  064136589897359 and so on, 43 such fragments from `merged_results.json`
  alone. A float is no longer read for digits (a seed stored as `205.0`
  still is), so the inventory now lists only integers that something
  actually keyed or named.
- **Text values that name no seed** (text column, ≤ 1000, with their first
  source): 0 (`PYTHONHASHSEED=0`, `AMENDMENT_1.md`), 1 (`--seeds 1 --ticks
  600` sanity-check comment in the harness), 2–6 and 8 (the synthetic
  `SEEDS = [1, …, 8]` of `tests/test_wp9_asserts.py`), 7 ("Multi-seed 7-arm
  recovery comparison"), **149** (`docs/DECISIONS.md:424`, "seed 82, 7 149
  of 169 889 units" — a thousands-separated unit count on a seed line),
  the tick counts **250, 300, 400, 600, 900** on prose lines that also name
  a seed (`mappo_trainer.py:717`, `output/ref/README.md:95`,
  `agent.py:899`, `UNTRAINED_CONTROL.md:46`, `DIAG_STALE_EDGES.md:38`), and
  **999** (`output/probe_wp9.py:70`, `HC.check(A, B, seeds=[999])`, a WP9
  probe fixture).

Consequently the only integers between 140 and 999 that appear anywhere are
149, 250, 300, 400, 600, 900 and 999 — each named above, none of them a
topology seed — together with the tokens classified in §3.3 and §3.4;
**200–219 is untouched**. The next untouched 20-seed range, if one were ever
needed, is 220–239: its own audit run of 2026-09-23 12:09 UTC is CLEAN with
coverage ok (§1). It is re-run from the frozen tree before the range is
used, exactly as 200–219 is.

## 5. Re-running the audit

The audit is repeated from the frozen tree at freeze 1 (before training, and
before the chain's S0 builds the block's manifest) and at freeze 2 (after
S5b, before any confirmatory arm runs; the manifest built at S0 exists by
then and is a planning reference, not an exposure — §6.11), with the commands of
§2.1 — **including the fresh capture of the trackB history dump**, whose
sha256 and commit count are recorded with the verdict. The verdict block of
each run is appended to `docs/PREREGISTRATION_R2.md` §13.

A freeze may proceed only when the run satisfies **all** of:

1. exit status 0, verdict CLEAN (no result-data, text, file-name or
   git-history hit). **`verdict.window_hits_in_result_data` covers all
   three container sources — pickles, JSON *and* torch checkpoints.** Until
   2026-09-23 it covered the first two, so a seed of the window in a
   checkpoint's metadata was written to `pt[<file>]['hits']` in the report,
   printed in the summary and left out of every verdict field: the run said
   CLEAN and exited 0 (§2 step 5). Read `pt` in the report as well as the
   verdict line: no entry there may carry a non-empty `hits`;
2. `errors: []` and `verdict.coverage_ok: true`, with no coverage failure of
   §2.3 (the coverage warning naming `~/MARL_code` is expected and is the
   only one allowed);
3. counts at least those of the first record of 2026-09-22 — **6 077 files,
   2 455 text files, 224 JSON files, 184 pickles, 2 563 checkpoints** —
   which the aggregate `--min-*` flags of §2.1 enforce, **and** every
   per-root floor of `tools/seed_audit_floors.txt` plus `--min-per-root`,
   which the aggregate floors cannot stand in for: the run of §1 has about
   a thousand files of slack over `--min-files`, more than the whole of
   `~/MARL_run_9c` or `~/MARL_watch`. Every per-root floor is re-set from
   the frozen counts at freeze 1, when no track writes while the audit
   runs. **This item is checked from the report, not inferred from item
   8**: the tool reads the floors file itself (`--min-root-file`) and
   records `min_root_file` = `{path, sha256, floors, ok}`, and a file that
   is missing, unreadable, unparsable or empty is a coverage failure. Check
   `min_root_file.ok` is `true`, that its `sha256` is that of the committed
   `tools/seed_audit_floors.txt`, and that `floors` holds one entry per
   root. Until 2026-09-23 the driver expanded that file into `--min-root`
   arguments in a `while read … done < "$FLOORS"` loop under `set -u` and
   no `set -e`: a missing, renamed or unreadable file left the array empty,
   the failed redirect was never checked, and the audit ran with **no**
   per-root floor at all and could still exit 0 — the only trace was the
   absence of `--min-root` from the report's `argv`, which nothing checked;
4. every directory matching `$HOME/MARL*` among the roots (the tool refuses
   otherwise; `--no-home-check` must not be passed);
5. the history scan actually ran: a `--git-log` dump with a non-zero commit
   count, captured from the real `run9_trackB` worktree **at the same HEAD
   the scanned `~/MARL_run_9b` was synced from** — compare the **report's
   own `synced_head.value`**, not the live `~/MARL_run_9b/.synced_head`,
   which the next sync overwrites (§2.2) — its sha256 and commit count
   recorded and identical on laptop and server, and **every commit made
   between that HEAD and the appended verdict block named**, with the files
   it touched. That set is enumerated over **all refs**, because `master`
   and `trackAgent` commit into the same repository:

   ```sh
   comm -13 <(grep '^commit ' <DUMP> | awk '{print $2}' | sort) \
            <(git rev-list --all | sort)          # exact, by construction
   git log --stat --all --since='<report.started>' --not <synced_head>
   ```

   A `<HEAD>..trackB` range is the trackB-only subset and does **not**
   satisfy this item on its own (§2.2: a record is written after its run,
   so the gap is never empty and must be stated rather than assumed away);
6. no new line in the text-hit list, in the false-positive list, in the
   file-name list or in the tick-loop bucket that is not in this document,
   no loop variable other than `tick`, `t` or `episode`, and **no new entry
   in `broad_tokens['other']` that is not classified in §3.4** (the bucket
   that catches values the extractors did not parse);
7. the unit tests of the tool pass on the server in the frozen commit
   (`tests/run_tests.py wp12_seed_audit`, 33 tests);
8. the run was driven by the committed `tools/run_seed_audit.sh`, whose
   roots are the `$HOME/MARL*` **pattern** — the report's `argv` shows only
   the shell's expansion of it, so the driver, not the report, is the
   evidence for this item (§2.1). It is evidence for **this item only**:
   item 3 no longer rests on it, because the per-root floors are now
   machine-checked from `min_root_file` in the report;
9. no output of the audit, or of any diagnostic run made while preparing
   the freeze, sits inside a scanned root under a name the planning rule
   does not cover (any path containing `seed_audit`). Such a file is read
   back as ordinary result data and makes the next run TOUCHED on its own
   contents — which is what the superseded run of §1 did. Write them under
   `audit_history/` (excluded with `--skip-dir`) or outside every root.

10. no RECORD DOCUMENT written while preparing the freeze names a literal
   seed of the audited window unless that document is itself matched by
   `PLANNING_MARKERS` in `tools/seed_audit.py`. Item 9 covers audit and
   diagnostic OUTPUTS; this covers the prose we write about them, which is
   the same failure by a different route. It happened: commit `e4bf26f` put
   a window seed into `docs/DECISIONS.md`, which is not a planning document,
   and the committed tree then returned TOUCHED on its own record while the
   substantive finding (the window never used) was unaffected. Write such a
   seed as `<N>`, name it only in the test file that needs it, or add the
   document to `PLANNING_MARKERS` — but note that the last option
   permanently buckets any future real leak in that file as planning, which
   is why it was not taken here.

Any of these failing stops the freeze until the cause is removed or
classified here.

## 6. The audit at the final head — measured 2026-09-30 and 2026-10-01

Written 2026-10-01 from the server (<compute-server> reachable again) and the
local repository. It supersedes the verdict of §1. Nothing in this section
was taken from a metric: this session read pickles only for their
`provenance` and `seeds` fields (the audit tool walks them for seed values
and reports no metric), logs only for their launcher, gate and traceback
lines, and no line of a rehearsal log that could carry a result is quoted.

### 6.1 Result in one paragraph

**Exit 4 (INCOMPLETE), measured twice** (2026-09-30 10:37 UTC and
2026-10-01 12:28 UTC, §6.2–§6.4). The coverage refusal has one cause:
five synthetic stand-ins written by the launcher tests of the `test`
profile into the production trees (three 244-byte pickles under a `.json`
name that the JSON parser rejects, two 238-byte pickles under a `.pt` name
that torch rejects; §6.5). Every hit is classified in §6.6. **Result data
of the window exists for seeds 200–203 only, all of it in `~/MARL_run_9d`,
all of it from the two reduced rehearsals of 2026-09-24** (13 pickles, 3
manifests, 8 adapted-checkpoint file names, 1 094 log/ledger lines). Seeds
204–219 carry no result data anywhere. No checkpoint among 2 793 carries a
window seed in its metadata. No RESULT pickle or JSON in `~/MARL_run_9`
(the tree of the real run) carries a window seed; the one file there that
named the block was the synthetic launcher-test stand-in of §6.5, a pickle
under a `.json` name, recorded as an error (§6.12). Everything else is a planning
reference, the audit's own output, a known false positive, or one of two
groups reported for the AUTHOR's classification (§6.6 (e): the launcher-test
stand-ins with their logs, and copies of the tree under
`~/MARL_run_9/output/wp13_*`). Under the brief's rule those two groups are
unexpected until the author classifies them; nothing else is.

The run of 2026-10-01 is **not freeze evidence** for a second reason: a
sync and a full test suite, started by another session, wrote into
`~/MARL_run_9` while it ran (§6.3).

### 6.2 Commands

```
# laptop, 2026-10-01 12:28 UTC, local repository at master 83e3126
git -c core.autocrlf=false log --all -p --no-color > trackB_git_log.txt
sha256sum trackB_git_log.txt   # 95f7e252bc303b8ad10201a241f648c9bc29d61bdecb9d8c6da4d97454682f24
                               # 7 380 825 B, 159 205 lines, 226 commits = git rev-list --all | wc -l
ssh -o BatchMode=yes <compute-server> 'cat > ~/MARL_run_9/audit_history/trackB_git_log.txt' < trackB_git_log.txt
ssh -o BatchMode=yes <compute-server> 'sha256sum ~/MARL_run_9/audit_history/trackB_git_log.txt'   # identical

# <compute-server>, started 12:28:28 UTC, after checking CHAIN END profile=rehearsal rc=0
cd ~/MARL_run_9 && nohup bash -c "cd ~/MARL_run_9 && MARL_TREE=\$HOME/MARL_run_9 bash tools/run_seed_audit.sh 200 219" \
  > audit_history/run_seed_audit_200_219.nohup 2>&1 < /dev/null &
```

Tool `tools/seed_audit.py` sha256 `b43b8c839fd05218…` (the audited tool of
§1, unchanged at `a33fdbc` and at `b43316a`), driver
`tools/run_seed_audit.sh` `1b04b0e0aa988c53…`, floors file `04ab537a5b47d0b9…`
(14 parsed, `min_root_file.ok: true`). The tool's unit tests (§5 item 7)
were NOT run by this session.

Outputs (the driver writes to fixed paths that the next run overwrites, as
the preamble of this record already notes; copies were made at once):

| run | report (bytes, sha256) | log (sha256) | dump scanned | copies |
|---|---|---|---|---|
| 2026-09-30 10:37:24–10:49:21 UTC | 3 561 054 B, `aa310f483504d22519340c9335a2b6e272a77ad5116b21a7bf76aa955f20bebd` | `88d3bb8bd820b3663e1016de179bba2da274ffaaff20fafdec5d5b0579064262` | 223 commits (the count of all refs at `a33fdbc`), `28b7c489fedf3b16…` | `~/MARL_run_9/audit_history/run_20260930T103724Z/` |
| 2026-10-01 12:28:28–12:39:18 UTC | 4 321 256 B, `3e39a99d6371868cdad9c9c62e5f56cb3c34b55a5d83d22249511b246a81c67c` | `844f99b02cf7bdb7f0a23930880322d023baf49fec4c82a063b99fff46b4ade1` | 227 commits, `18c50273a78cebb2…` (= all refs at `b43316a`, §6.3) | `~/MARL_run_9/audit_history/run_20261001T122828Z/` |

Both driver exit lines read `EXIT_200_219=4`. The summary block of the
2026-10-01 run, abridged (the skipped-directory and symbolic-link lines are
left out — unchanged from §1 — and three long lists are shortened with
`...`):

```
Seed audit 200-219 on <compute-server>, 2026-10-01 12:28:28 UTC -> 2026-10-01 12:39:18 UTC (650 s)
files 12504 | text scanned 6684 | JSON parsed 1894 + regex-scanned 14 | pickles 399 (loaded 360, duplicates 39, failed 0) | checkpoints 2793 (failed 2, skipped 0) | binary skipped 189
Per root: MARL_run_2=1090, MARL_run_3=1052, MARL_run_4=120, MARL_run_5=827, MARL_run_6=158, MARL_run_7=114, MARL_run_8=118, MARL_run_9=4143, MARL_run_9b=401, MARL_run_9c=252, MARL_run_9d=1755, MARL_code=282, MARL_code_remote=690, MARL_perf=1478, MARL_watch=24
Per-root floors from <server-home>/MARL_run_9/tools/seed_audit_floors.txt: 14 parsed, sha256 04ab537a5b47d0b9 (applied)
Tree .synced_head: a33fdbc (2026-09-29 22:09:29 UTC, read)
$HOME/MARL* directories: 15 (MARL_code, MARL_code_remote, MARL_perf, MARL_run_2, ..., MARL_run_9d, MARL_watch)
Git history: 0 root(s) scanned of 1 with a .git entry; dumps: trackB_git_log.txt (227 commits, 159306 lines, sha256 18c50273a78cebb2)

VERDICT: INCOMPLETE - coverage refused, this run is not evidence
  result data hits   : [200, 201, 202, 203]
  text hits (non-plan): [200, ..., 219]
  file-name hits     : [200, 201, 202, 203]
  git-history hits   : [200, ..., 219]
  classified false positives (--false-positive): [200]
  coverage           : REFUSED
    ! 3 error(s) recorded (see "errors"); the first is: MARL_run_9/confirmatory/test_manifest_200_201.json: json parse failed (Expecting value: line 1 column 1 (char 0)); scanned as text
    ! 2 checkpoint(s) failed to load
    ~ git log failed in <server-home>/MARL_code (rc=128) and the root is declared --git-no-repo: its history is NOT auditable
```

Hit counts of the two runs: text 1 265 lines in 73 paths (09-30) and
1 266 in 74 (10-01; the extra line is §6.6 (e1)'s history line of
`b43316a`), file names 49 and 50 (the extra one is the 09-30 report itself
under `output/`), planning lines 979 and 998 (the planning documents'
history added by the freeze commits), false positives 27 and 27, unclaimed
tokens on seed lines (`broad_tokens['other']`) 2 963 in the 10-01 run.

### 6.3 What else wrote into the tree while the 2026-10-01 run ran

This session did not cause any of the following and stopped none of it
(the processes are not its own); it is recorded because §5 requires that
nothing writes into a scanned tree during the run.

- 12:31:43 UTC: commit `b43316a` ("FREEZE: the launcher tests leave
  nothing in the production tree …", `tests/test_run_chain_launcher.py`
  only) was made in the same local repository by another session.
- 12:32:09 UTC: `~/MARL_run_9` was synced to it (`.synced_head` = `b43316a`).
- 12:32:12 UTC: the history dump was replaced by a 227-commit dump
  (`18c50273…`, verified on the laptop to be `git log --all -p` at
  `b43316a`). The run read the dump at its end, so it scanned that dump,
  not the 226-commit one captured for it.
- 12:32:41 UTC: `~/MARL_watch/jobs/freeze_checks.sh` started the full
  suite in `~/MARL_run_9`; its launcher tests write and delete the
  test-profile stand-ins. Its later steps archive and delete the stand-ins
  of `~/MARL_run_9d` (to `~/archive_marl_test_fixtures_<date>.tgz`) and run
  this driver again, which overwrites the fixed report and log paths.

The walk (the file inventory) finished at 12:29:58 UTC, before all of
these, and the report's `synced_head` is `a33fdbc`. Contents read later
can come from the new state: `MARL_run_9/output/test/untrained_s1234/policy_init.pt`
failed to load with `invalid load key 'n'` on 10-01 against `Invalid magic
number` on 09-30, so it was rewritten during the run. The 2026-09-30 run
had no such overlap with this session's knowledge, but it ran during the
rehearsal on `a33fdbc` (07:43–12:32 UTC), which also writes into
`~/MARL_run_9`.

### 6.4 The run of 2026-09-30, not cited before

It ran at head `a33fdbc` (10:37:24–10:49:21 UTC, 718 s, 12 409 files,
2 793 checkpoints, 223-commit dump) and returned **exit 4** with the same
three errors, the same two checkpoint failures (both `Invalid magic number`)
and the same hit classes as §6.6. Earlier freeze documents said the last
measured verdict was that of 2026-09-23 and predicted "exit 3 (TOUCHED)";
the record now cites this run. Its report and log are kept in
`audit_history/run_20260930T103724Z/` with the sha256 of §6.2.

### 6.5 The coverage failure: five launcher-test stand-ins

| path | bytes | sha256 | written (UTC) | content |
|---|---|---|---|---|
| `MARL_run_9/confirmatory/test_manifest_200_201.json` | 244 | `e23a97e21712a88c…` | 2026-09-29 12:42 (before `31b034c`) | pickle: `seeds [200, 201]`, empty `recovery_multi`/`rescue_multi`, provenance `tree_head 'faketest'`, `schema_id 'fake'` |
| `MARL_run_9/confirmatory/test_manifest_220_221.json` | 244 | `d66fca9d30fb9cc5…` | 2026-09-30 07:22 | the same with `seeds [220, 221]` |
| `MARL_run_9d/confirmatory/test_manifest_200_201.json` | 244 | `e23a97e21712a88c…` | 2026-09-24 18:56 | identical bytes to the first row |
| `MARL_run_9/output/test/untrained_s1234/policy_init.pt` | 238 | `5051c43a5d26feb6…` (09-30) | 2026-09-30 07:22 | a pickle of the same kind of stand-in record, not a torch file |
| `MARL_run_9d/output/test/untrained_s1234/policy_init.pt` | 238 | `5051c43a5d26feb6…` | 2026-09-24 18:56 | identical bytes |

They are not empty JSON files (an earlier summary said so): each is a
protocol-4 pickle, and the JSON parser fails on its first byte. Origin:
`tests/test_run_chain_launcher.py` runs `confirmatory/run_final.sh
--profile test` in the production tree with `CHAIN_SCRIPT_OVERRIDE=tests/fake_stage.sh`,
and `tests/fake_stage.sh` pickles a stand-in record (constant series of 1.0,
or no passes) to every invocation's artifact path, including S0's manifest
(a `.json` name) and the untrained checkpoint (a `.pt` name). No simulation
runs. **This is a code defect** (test artefacts written into the
production tree rather than under the audit-skipped `output/tmp_tests/`);
this record changes no code. `b43316a` (§6.3) is another session's attempt
to make the tests remove them; whether the suite at that head leaves none
behind is for the freeze checklist step b to show.

How they are to be handled before the gate can pass — the AUTHOR decides,
because they are untracked files inside the scanned trees and one of those
trees (`~/MARL_run_9d`) is a record:

1. **Archive them outside every scanned root** (for example a dated
   directory or tarball directly under `$HOME`, which matches no
   `$HOME/MARL*` pattern), with the path, size and sha256 of every file
   recorded first (this table already records them), then re-run the
   audit. This is the recommended handling: it removes nothing from the
   record and weakens no check. `docs/FREEZE_CHECKLIST.md` step c.0.
2. Or keep them and amend the gate to accept exactly these five errors,
   identified by path and sha256. Not recommended: the suite re-creates the
   `220_221` file with each run, so the accepted set would drift.

### 6.6 Every hit, by class

Counts are lines (text) or files. Paths are relative to `$HOME`.

**(a) Result data of the two reduced rehearsals of 2026-09-24 — the known
exposure.** Seeds 200–203 only; every file in `~/MARL_run_9d`.

- 13 pickles, every one `total_ticks 600`, `schema_id 'placeholder-pre-WP6'`:
  - attempt 1 (archived in `archive_rehearsal/run1_20260924T123836Z/output/rehearsal/`,
    `tree_head 0592f26`): `s6_cf_b0.pkl` (64 passes), `s8_untrained_s1234.pkl`,
    `s8_untrained_s2345.pkl` (8 each), `s9_freeze.pkl`, `s10_sens_install.pkl`,
    `s11_sens_repoint.pkl` (8 each) — 104 passes;
  - attempt 3 (`output/rehearsal/`, `tree_head 79e15b2`): `s6_cf_b0.pkl` (64),
    `s8_untrained_s1234/s2345/s3456.pkl` (8 each), `s9_freeze.pkl`,
    `s10_sens_install.pkl`, `s11_sens_repoint.pkl` (8 each) — 112 passes.
- 3 manifests `rehearsal_manifest_200_203.json` (S0 of attempts 1, 2 and 3:
  `archive_rehearsal/run1_…/confirmatory/`, `archive_rehearsal/run2_…/confirmatory/`,
  `confirmatory/`) — topology-derived quantities only.
- 8 file-name hits: `ck_s2345/policy_final_adapted_recovery_freeze_seed200.pt`
  … `seed203.pt` in `output/rehearsal/` and in the attempt-1 archive — the
  copies of the toy policy that the adapt-then-freeze arm (S9) adapted
  online on those instances. No later step uses them; their metadata holds
  no window seed (the checkpoint scan found none in any of 2 793 files).
- 1 094 log and ledger lines in 27 files: attempt 3 — `logs/chain_rehearsal/`
  `s0_manifest.log` (10), `s5_gate.log` (1), `s6_cf_b0.log` (291),
  `s8_untrained_s1234/s2345/s3456.log` (39 each), `s9_freeze.log` (47),
  `s10_sens_install.log` (39), `s11_sens_repoint.log` (39),
  `logs/chain_rehearsal.log` (1), `logs/rehearsal_run3.out` (1),
  `confirmatory/chain_rehearsal.jsonl` (2), `confirmatory/FREEZE_FINAL_rehearsal.txt` (1)
  — 549; attempt 1 (archive) — the same stage logs (531); attempt 2
  (archive) — S0 only (14). The unclaimed tokens on seed lines of the same
  logs (`broad_tokens['other']`) belong here too.

Verified from the ledgers (`confirmatory/chain_rehearsal.jsonl` and the
archived copies), the job scripts and the pickles' provenance, not from
the brief:

| claim | measured |
|---|---|
| when | **2026-09-24 only** (attempt 1 08:15–11:24 UTC, attempt 2 12:38–14:30, attempt 3 14:40–18:17; all on the 24th in local time too). "2026-09-24/25" in earlier text is wrong. |
| which code | attempt 1 `0592f26`, attempt 3 `79e15b2` (attempt 2, `dd8a18e`, never reached a window-seed pass): pre-merge run-chain trees without `sixg_sim/link_state_service.py`, `sixg_sim/postcards.py` or `sixg_sim/reward_spec.py` (the frozen reward object), with the environment-selected `sixg_sim/reward_profile.py`, observation schema `placeholder-pre-WP6` |
| length | `total_ticks 600` in every window-seed pickle (the study: 3 600) |
| policy | `train_on_comparison.py --episodes 32 --episode-ticks 200 --batch-size 16 --seed-range 42-49,100-139` (two batches: `policy_comparison_shared_v0/v1`), torch seeds 1234, 2345 (+3456 in attempt 3); the window-seed passes used the `s2345` replicate and the untrained actors |
| passes on 200–203 | 216 in all (104 + 112), every one at 600 ticks |
| primary analysis (S7) | failed twice in each attempt (10:56 UTC and 17:41 UTC, ledger `failed_twice`). `confirmatory_analysis.py --marl-arm marl_static` (unchanged since `d94a936`) passed its provenance and key-union gates (64/64), computed the descriptive block, **the whole primary family P1–P9** and the S1 secondary tests **in memory**, then crashed at `confirmatory_analysis.py:277`, the first secondary test that reads the pre-surge window (`S2_B_change_vs_xldet`), with `AttributeError: 'float' object has no attribute 'numerator'` in `statistics.stdev` (the NaN of the 900–1400 pre-surge window of a 600-tick series). It **wrote no report and printed no statistic**: the JSON dump and every print come after line 277, `rehearsal_cf_report.json` does not exist, and `rehearsal_cf_report.txt` (identical in both attempts) holds two scipy warnings and the traceback only. A crash, not a refusal: the refusal of 600-tick series came with the trackB merge. |
| other analyses | S12 wrote a pass-count and provenance report; `untrained_analysis.py` found no input (hard-wired to the first study's files) and is recorded NOT RUN (exit 90). The ablation gate of S10/S11 compared each ablation with its S6 baseline (identical or not); `docs/RUN_CHAIN.md` §6 records that S11 changed nothing at 600 ticks. No other output of these passes was analysed. |

**(b) Planning references — legitimate, they name the block the real run
WILL use.**

- `MARL_run_9/confirmatory/chain_plan.py:127` (`'cf_seeds': list(range(200, 220))`,
  the real profile); `MARL_run_9/tests/test_run_chain.py:91, 207, 712, 761`
  (the pins of the real profile and the guard that no other profile names
  the block); the same lines in the history dump (chain_plan 8 lines,
  test_run_chain 4), where the `-` lines are the pre-`31b034c` rehearsal and
  test profiles being removed.
- `MARL_run_9d/confirmatory/chain_plan.py:120, 143, 227` and
  `MARL_run_9d/tests/test_run_chain.py:91, 207`: the superseded run-chain
  code (`b728ce8`) whose rehearsal profile named 200–203 and test profile
  200–201 — the cause of (a), kept as a record.
- `docs/RUN_CHAIN.md:265` in `~/MARL_run_9`, `~/MARL_run_9d` and the
  history ("It cannot run against seeds 200–219 in this tree", about the
  dry-run profile): a planning statement in a document that is not
  planning-marked (§5 item 10). The local text is reworded on 2026-10-01 so
  that the next sync removes the file hit; the history line stays.
- The commit message of `31b034c` (one history line, the record of the
  exposure itself).
- The planning documents and the audit's own files: 998 lines in 111 paths,
  bucketed by the tool's planning markers.

**(c) The audit's own outputs.** `MARL_run_9/output/seed_audit_200_219.json`
(the 09-30 report, at scan time), `MARL_run_9b/output/seed_audit_200_219.json`,
`.run1.json`, `.run2.json` and `seed_audit_220_239.json`: planning-named JSON
files that list the window by design; the tool keeps them out of the
result-data verdict. The 18 planning-named file names (`SEED_AUDIT_200_219.md`
copies, `seed_audit_*` reports and logs) likewise. Also the two history
lines naming `adapted_seed209.pt` in `docs/DECISIONS.md` (added by
`e4bf26f`, removed by `f03c08f`): the WPDOC record of a file name planted
in a scratch fixture of the tool's own unit tests (§6.12; omitted from
this list until FIX-PREFREEZE item 10).

**(d) False positives.** The `sixg_sim/agent.py` docstring "(seed 42, 200
island ticks, 35 agents)" of §3.2: 27 lines matched by the driver's
`--false-positive "seed 42, 200 island ticks"`. In the merged tree the
docstring is re-wrapped so that `(seed 42, 200` ends line 1444 and "island
ticks" starts the next line; the pattern no longer matches, and the same
docstring is counted as a text hit in `MARL_run_9/sixg_sim/agent.py:1444`,
`MARL_run_9c/sixg_sim/agent.py`, the seven `wp13_*` copies and one history
line. Same false positive; the driver's pattern (code) is not changed here.

**(e) Reported for the AUTHOR's classification.**

- (e1) **Launcher-test stand-ins and their logs** (no simulation): the five
  files of §6.5 (two of them named for 200–201: 2 seed-like file-name hits);
  `MARL_run_9/logs/chain_test.log` (38 lines) and `MARL_run_9d/logs/chain_test.log`
  (49), `MARL_run_9d/confirmatory/chain_test.jsonl` (2),
  `MARL_run_9d/logs/chain_test/s0_manifest.log` (1) — every one a launcher
  or gate line naming the `200_201` stand-in manifest of the pre-`31b034c`
  test profile; and one history line of `b43316a`'s
  `tests/test_run_chain_launcher.py` comment naming that file. Proposed
  class: test fixture, not a use. (`~/MARL_run_9/logs/chain_test.log` was
  overwritten by the 2026-10-01 suite after the run read it.)
- (e2) **Copies of the tree under `MARL_run_9/output/`**: `wp13_ref_823620d`,
  `wp13_stage_0367856`, `wp13_stage_a82fd96`, `wp13_stage_b3894e9` (with two
  negative-control copies under `output/wp13_negctl/`) and
  `wp13_stage_e84c468` — 28 files, 50 lines: in each copy the real
  profile's lines of `confirmatory/chain_plan.py` (3–4), `tests/test_run_chain.py`
  (2), `docs/RUN_CHAIN.md` (1) and the agent.py docstring (1). Planning
  content under paths the planning rule does not cover. Proposed class:
  planning (copies made for the WP13 staging and negative controls).

**(f) Tokens on seed lines that no extractor claimed** (2 963; the bucket
§5 item 6 reviews). Apart from (a) and the planning paths: the severance
tick 200 (`tree … ticks 260 severance 200` and `discovery from t=200` in
the suite logs, `severance=200` in `logs/wp13/lat_*_sev200.*`, the
`severance_tick(A)=200` header of the contract logs, the WP12 report
headers), tree digests that contain `217`, `206` or `212`, link counts
(`uu_boost_links … ref=209` in `logs/wp13/sdiff_*`, `created: 202` in a
relay-lifetime diff), the planning file name `docs/SEED_AUDIT_200_219.md`
in `.synced_files` and in `docs/DECISIONS.md`, comments that name the block
(`chain_plan.py` "instead of the pre-registered 200-219",
`HANDOVER.md`), a test fixture (`tests/test_run_chain.py:565`, a synthetic
report with `'seeds': {'A': [200], 'B': [200]}`) and the §3.4 items. None
is a seed use. The 18 file names `logs/wp13/lat_*_sev200.*` and
`MARL_perf/logs/prof200.log` are tick counts, not seeds (not seed-like).

**Nothing else.** No hit falls outside (a)–(f).

### 6.7 What else was confirmed

- **The rehearsal on `a33fdbc`** (`CHAIN START` 2026-09-30T07:43:40Z,
  `CHAIN END profile=rehearsal rc=0` at 12:32:54Z, 25 RUN lines, no SHED
  or WAIVE) wrote only pickles with `tree_head a33fdbc`, `schema_id
  obs1-542eb7fc0d277d1d` and confirmatory seeds 220–223 (and the
  full-length 220–221); its manifest is `rehearsal_manifest_220_223.json`.
- **The rehearsal of 2026-09-29 on the merged tree** started at 14:55:49
  UTC, after `31b034c` (14:27:34 UTC), and built `rehearsal_manifest_220_223.json`
  (archived in `output/rehearsal_archive_a33fdbc_20260930T074340Z/`); its
  ledgers name no window file. The "possible" 2026-09-29 manifest for
  200–203 of the earlier pre-registration text did not happen.
- **The full-scale dry run** (`~/MARL_run_9d`, tree `b728ce8`, completed
  2026-09-27T06:47Z–09-28T14:53Z; first attempt `24e86d0`, 2026-09-25)
  used development and validation seeds and confirmatory 220–239 only
  (`dryrun_manifest_220_239.json`; provenance of every `output/dryrun*`
  pickle).
- **Only the `real` profile names 200–219** at `a33fdbc`
  (`PROFILES['real']['cf_seeds']`; rehearsal 220–223, dry run 220–239,
  test 220–221), pinned by
  `tests/test_run_chain.py::test_only_the_real_profile_names_the_preregistered_confirmatory_seeds`.

### 6.8 DISCLOSURE (text for the paper)

"The confirmatory block (topology seeds 200–219, with matching scenario
seeds) was fixed before training. Four of its seeds (200–203) were used on
24 September 2026, before the simulator corrections were complete, in two
attempts of a reduced-scale rehearsal of the run pipeline: 216 simulation
passes of 600 ticks instead of 3 600, with policies trained for two
16-episode batches of 200 ticks, on superseded versions of the simulator
without the corrected control plane, observation, action space and reward.
The rehearsal's confirmatory analysis crashed in all four of its attempts,
at a secondary test, after it had computed the primary tests in memory and
before it wrote or printed any result; no statistic from those passes was
reported or examined, and beyond the pipeline's mechanical checks (pass
counts, provenance, and whether each ablation changed its series at all)
none of their outputs was analysed. The block was kept as pre-registered.
The other sixteen seeds carry no result data from any earlier run."

### 6.9 The decision on the block, as it stands (2026-10-01)

The analysis lead **RECOMMENDED keeping 200–219 as pre-registered and
disclosing the rehearsal exposure (§6.8).** The author was asked, did not
choose explicitly, and instructed the work to proceed. This is recorded as
the lead's recommendation, not as the author's decision. It can still be
reversed until the real run touches the block, which it does at its first
invocation (`s0_manifest` builds the block's manifest).

Reasoning of the recommendation: sixteen of the twenty seeds were never
touched; the four that were saw only a reduced pipeline test with a toy
policy on superseded code, and no statistic from it was ever written,
printed or examined. Switching the pre-registered block after the fact
would raise the sharper question of what had been seen. Note for a
reversal: 220–239 is no longer an untouched fallback (dry run 220–239,
rehearsals 220–223, test profile 220–221); a replacement block needs its
own clean audit.

### 6.10 The gate at freeze 1 and freeze 2, revised

§5 item 1 (exit 0, CLEAN) cannot be met any more. A freeze may proceed when
a run of the committed driver, made while nothing else writes into any
scanned tree, satisfies: **exit 3 (TOUCHED)**; `errors: []` and
`coverage_ok: true` (which requires §6.5 handled first); §5 items 2–5 and
7–10 otherwise unchanged; the report's history dump sha256 equal to the
one captured for it; and every hit inside the groups of §6.6 (a)–(d), the
groups of (e) classified by the author, and (f) unchanged — any new group
is unexpected and stops the freeze. At freeze 2 the real run's own S0
manifest and the chain's lines naming it are added to (b) (§6.11). The verdict
block of each run is appended to `docs/PREREGISTRATION_R2.md` §13, and the
report and log are copied out of the fixed paths at once.

### 6.11 The manifest built at S0 is planning, not exposure (2026-10-01)

Recorded as decision D-FIX.9 (`docs/DECISIONS.md`, FIX-PREFREEZE item 9),
and stated the same way in `docs/PREREGISTRATION_R2.md` §5 and §15 item 9.
The chain builds the block's manifest at its first invocation, `s0_manifest`
(`build_benchmark_manifest.py --seeds <block>`), before training and before
freeze 2. The manifest holds topology-derived planning data only: per
instance the severance tick, the raw component count of the survivor graph,
the attainable optimum and whether it is 1 (`fully_reunifiable`), and the
solver's and the actuation's candidate-site counts, computed from the
topology and the scenario's own events without running any arm, and no
arm's result. Building it is therefore planning, not an exposure of the
block. In consequence:

- freeze 1's audit runs before S0, as §5 states;
- at freeze 2 the real run's own S0 manifest and the chain's lines naming it
  or the block (`logs/chain_real/s0_manifest.log`, the ledger, the
  launcher log) are class (b) planning references (§6.10); a hit anywhere
  else, any S1-S5b artefact in particular, is class (d) and stops the run;
- §5's earlier wording "before the manifest for 200-219 is built" is
  replaced; the audit is not required before the manifest exists.

### 6.12 Corrections to §6.1 and §6.6 (2026-10-01, FIX-PREFREEZE item 10)

- **§6.1 overstated one statement.** "No pickle or JSON in `~/MARL_run_9`
  carries a window seed" holds for result data only. At the time of both
  measured runs `~/MARL_run_9/confirmatory/test_manifest_200_201.json` was a
  244-byte PICKLE under a `.json` name (the launcher-test stand-in of §6.5)
  whose `seeds` label held `[200, 201]`, with no passes and provenance
  `faketest`; the scan could not parse it as JSON and recorded it under
  `errors`, so it was never classified as result data. It was a synthetic
  test fixture, not a use of the block. The 2026-10-01 suite at `b43316a`
  removed it with the other test-profile fixtures of `~/MARL_run_9` (its
  path, size and sha256 are in §6.5; `~/MARL_run_9/logs/freeze_checks.log`
  13:00:38 UTC records that none was left); the `~/MARL_run_9d` copies were
  archived to `~/archive_marl_test_fixtures_20261001.tgz` (sha256
  `59d7e5b280d016903df1501bfb8685006893634731f65d62936cc56dda7bf540`).
  `docs/PREREGISTRATION_R2.md` §5 carries the same correction.
- **§6.6 omitted two history lines.** The history dump holds two lines
  naming `adapted_seed209.pt`, both in `docs/DECISIONS.md`: added by
  `e4bf26f` (2026-09-23) and removed by `f03c08f` (2026-09-24), the WPDOC
  record describing a file name PLANTED in a temporary fixture directory of
  `tools/seed_audit.py`'s own unit tests (`tests/test_wp12_seed_audit.py`)
  to prove that a seed-like file name is reported. Class (c), the audit's
  own; neither names a file that ever existed outside a test's scratch
  directory, and the current `docs/DECISIONS.md` no longer names it.
