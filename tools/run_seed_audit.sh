#!/bin/bash
# WPDOC seed audit -- the recorded driver.
#
#   ssh -o BatchMode=yes <compute-server> 'bash ~/MARL_run_9b/tools/run_seed_audit.sh 200 219'
#
# It lives in the tree, so what ran is what is committed.  The earlier driver
# (~/seed_audit_evidence/run_audits.sh) lived outside the repository and had
# drifted from the document that quoted it: it enumerated all fourteen roots
# LITERALLY while docs/SEED_AUDIT_200_219.md §2.1 said it used the pattern.
#
# Roots are given BY PATTERN, which docs/PREREGISTRATION_R2.md §13 requires at
# a freeze: a tree created after this file was written (~/MARL_run_9c was) is
# then included automatically and the tool's own $HOME/MARL* check can only
# pass.  Never a literal list.
#
# The history dump must NOT sit inside a scanned root -- its diffs of ordinary
# source files would be read as file content instead of as history -- so it is
# written to ~/MARL_run_9b/audit_history, which is excluded from the walk with
# --skip-dir and scanned only through --git-log, with every line attributed to
# the file of its `diff --git` header.  Capture it from the laptop worktree at
# the HEAD that was synced, before this script runs:
#
#   git log --all -p --no-color | ssh -o BatchMode=yes <compute-server> \
#     'mkdir -p ~/MARL_run_9b/audit_history && cat > ~/MARL_run_9b/audit_history/trackB_git_log.txt'
#   git log --all -p --no-color | sha256sum      # must equal the server's
#
# THE AUDITED BLOCK IS A PARAMETER: `run_seed_audit.sh LO HI`, default 200 219
# (R2's confirmatory block), and with no SEED_AUDIT_GATE the command below is
# the R2 command, option for option.
#
# THE R3 BLOCK (docs/PREREGISTRATION_R3.md section 13).  The R3 gate is
# stricter than R2's and is requested explicitly:
#
#   MARL_TREE=$HOME/MARL_r3lite SEED_AUDIT_GATE=r3-freeze1 bash tools/run_seed_audit.sh 300 339
#   MARL_TREE=$HOME/MARL_r3lite SEED_AUDIT_GATE=r3-freeze2 bash tools/run_seed_audit.sh 300 339
#
#   r3-freeze1  before S0: ZERO result-data hits (--gate-result-data: a seed
#               of the block in any pickle, non-planning JSON file or
#               checkpoint is exit 5).  The plan itself -- the r3lite profile
#               in confirmatory/chain_plan.py, the tests that pin it and the
#               planning documents -- is classified as planning by explicit
#               --planning-marker paths, recorded in the report.
#   r3-freeze2  before S6: the same, and the ONLY additional planning data is
#               this run's own S0 manifest (its path is read from the plan,
#               never typed here) with the chain log and ledger that name it.
#               Anything else -- any pickle in particular -- is exit 5.
#   Exit 0 passes.  Exit 3 (a text, file-name or history mention left) passes
#   only once every such hit is classified in writing BY THE AUTHOR, before
#   the first S0 step, in one of the classes of PREREGISTRATION_R3 section
#   13(b): (b) planning reference, (c) coincidence -- the integer is a tick
#   count or a loop bound, not an instance -- or (d) negative statement -- the
#   line says the block was NOT read.  Anything else is an exposure.  Exit 4
#   and exit 5 stop the freeze.  The gate refuses any other block than the one
#   the r3lite profile declares.
#
#   The benign text hits KNOWN BEFORE FREEZE 1 (the pre-audit of 2026-10-11)
#   are pre-registered below as --false-positive line patterns, classes (c)
#   and (d).  A matching hit leaves the verdict but is LISTED, with its path
#   and line, in the log ("Hits classified as false positives") and in the
#   report (false_positive_hits, false_positive_patterns): recorded, never
#   hidden.  A pattern classifies a text line only; a pickle, a JSON file or
#   a checkpoint is result data whatever any line says.  No pattern is added
#   after freeze 1: a new hit is classified in writing or stops the freeze.
set -u
LO=${1:-200}
HI=${2:-219}
TREE=${MARL_TREE:-$HOME/MARL_run_9b}
cd "$TREE" || exit 1
export PYTHONHASHSEED=0

ROOTS="$HOME/MARL_run_* $HOME/MARL_code $HOME/MARL_code_remote $HOME/MARL_perf $HOME/MARL_watch"
# The R3 tree(s), when they exist: the tool's own $HOME/MARL* check refuses a
# run that leaves one out, and a pattern that matches nothing must not be
# handed to it as a literal root.
for d in "$HOME"/MARL_r3*; do
  [ -d "$d" ] && ROOTS="$ROOTS $d"
done
DUMP="${SEED_AUDIT_DUMP:-$TREE/audit_history/trackB_git_log.txt}"
FLOORS="$TREE/tools/seed_audit_floors.txt"
LOG="logs/seed_audit_${LO}_${HI}.log"
OUT="output/seed_audit_${LO}_${HI}.json"

# ---- the R3 gate (off unless SEED_AUDIT_GATE is set)
GATE="${SEED_AUDIT_GATE:-}"
GATE_ARGS=()
case "$GATE" in
  '') ;;
  r3-freeze1|r3-freeze2)
    BLOCK=$(python3 -c "import sys; sys.path.insert(0, 'confirmatory'); import chain_plan as c; s = c.PROFILES['r3lite']['cf_seeds']; print(s[0], s[-1], len(s))") \
      || { echo "run_seed_audit: cannot read the r3lite profile from $TREE/confirmatory/chain_plan.py" >&2; exit 2; }
    if [ "$BLOCK" != "$LO $HI $((HI - LO + 1))" ]; then
      echo "run_seed_audit: SEED_AUDIT_GATE=$GATE audits the block of the r3lite profile ($BLOCK: first last count), not $LO-$HI" >&2
      exit 2
    fi
    GATE_ARGS+=(--gate-result-data)
    # the plan, the tests that pin it, the planning documents (class b)
    for m in confirmatory/chain_plan.py tests/test_r3lite_plan.py docs/RUN_CHAIN.md \
             docs/DECISIONS.md docs/FREEZE_CHECKLIST.md; do
      GATE_ARGS+=(--planning-marker "$m")
    done
    if [ "$GATE" = r3-freeze2 ]; then
      MAN=$(python3 confirmatory/chain_plan.py --profile r3lite --field s0_manifest artifact) \
        || { echo "run_seed_audit: cannot read the S0 manifest path from the plan" >&2; exit 2; }
      case "$MAN" in
        confirmatory/*.json) ;;
        *) echo "run_seed_audit: unexpected S0 manifest path '$MAN'" >&2; exit 2 ;;
      esac
      # this run's own S0 manifest and the chain's lines naming it; nothing else
      for m in "$MAN" "$(basename "$MAN")" logs/chain_r3lite/s0_manifest \
               logs/chain_r3lite.log confirmatory/chain_r3lite.jsonl; do
        GATE_ARGS+=(--planning-marker "$m")
      done
    fi
    # PRE-REGISTERED classification of the benign text hits known before
    # freeze 1 (docs/PREREGISTRATION_R3.md section 13(b), which lists the
    # files and lines each pattern was written for).  One line SNIPPET
    # substring per row; the tool matches the first 200 characters of the
    # stripped line.
    #   (c) coincidence: a tick count or a loop bound, not an instance
    #       1  tests/test_d12_baselines.py: _run(..., range(0, 400)) tick loops
    #       2  output/ref/README*.md (R2 trees): "seed 82, <N> ticks" smoke run
    #       3  output/ref/invariance_WP0_*.txt: the same smoke run
    #       4  r3out/phase0/PROGRESS.md: task 0c, one episode of <N> ticks
    #   (d) negative statement: the line says the block was NOT read
    #       5  r3out/phase0/PHASE0_REPORT*.md: "Nothing for seeds ... was read"
    NFP=0
    while IFS= read -r pat; do
      [ -n "$pat" ] || continue
      GATE_ARGS+=(--false-positive "$pat"); NFP=$((NFP + 1))
    done <<'R3_PREREGISTERED_FALSE_POSITIVES'
[CROSS] + INSIDE)), range(0, 400))
(seed 82, 300 ticks, the arm that writes and reloads
marl_freeze smoke, seed 82, 300 ticks
0c DONE (team term in the actor advantage)
Nothing for seeds 200-219, 300-359 or 400-439 was read
R3_PREREGISTERED_FALSE_POSITIVES
    [ "$NFP" -eq 5 ] || { echo "run_seed_audit: $NFP pre-registered classification pattern(s) read, expected 5" >&2; exit 2; }
    # each freeze keeps its own log and report (the names stay planning
    # paths of the audit: they contain `seed_audit`)
    LOG="logs/seed_audit_${LO}_${HI}_${GATE}.log"
    OUT="output/seed_audit_${LO}_${HI}_${GATE}.json"
    ;;
  *) echo "run_seed_audit: unknown SEED_AUDIT_GATE '$GATE' (r3-freeze1, r3-freeze2, or unset)" >&2; exit 2 ;;
esac
# extra per-root floors for roots the committed floors file does not name
# (SEED_AUDIT_MIN_ROOTS="MARL_r3=250 ..."), each passed as --min-root
for spec in ${SEED_AUDIT_MIN_ROOTS:-}; do
  GATE_ARGS+=(--min-root "$spec")
done

# The per-root floors are now read BY THE TOOL (--min-root-file), which
# records the file, its sha256 and every parsed floor in the report and
# refuses the verdict (exit 4) if it is missing, unreadable, unparsable or
# empty.  This script used to expand the file into --min-root arguments in a
# `while read ... done < "$FLOORS"` loop; with `set -u` but no `set -e`, a
# missing or renamed file left the array empty, the failed redirect was never
# checked, and the audit ran with NO per-root floor at all and could still
# exit 0 -- the mechanism meant to guarantee the floors were applied could
# stop applying them without failing.  The guard below keeps the driver's own
# failure loud even if the tool's flag is ever dropped from the command.
[ -r "$FLOORS" ] || { echo "run_seed_audit: floors file missing or unreadable: $FLOORS" >&2; exit 2; }
NFLOORS=$(grep -c -E '^[[:space:]]*[^#[:space:]][^=]*=[0-9]+[[:space:]]*$' "$FLOORS" || true)
[ "${NFLOORS:-0}" -gt 0 ] || { echo "run_seed_audit: no NAME=N floor in $FLOORS" >&2; exit 2; }

mkdir -p logs output
nice -n 19 python3 tools/seed_audit.py --lo "$LO" --hi "$HI" --roots $ROOTS \
  --out "$OUT" --pt --git \
  --git-no-repo "$HOME/MARL_code" \
  --git-log "$DUMP" \
  --skip-dir audit_history \
  --false-positive "seed 42, 200 island ticks" \
  --min-files 6077 --min-text 2455 --min-json 224 --min-pickles 184 --min-pt 2563 \
  --min-per-root 10 --min-root-file "$FLOORS" --tree "$TREE" \
  ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
  -v > "$LOG" 2>&1
rc=$?
echo "EXIT_${LO}_${HI}=$rc" >> "$LOG"
echo "EXIT_${LO}_${HI}=$rc"
exit $rc
