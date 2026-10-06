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
set -u
LO=${1:-200}
HI=${2:-219}
TREE=${MARL_TREE:-$HOME/MARL_run_9b}
cd "$TREE" || exit 1
export PYTHONHASHSEED=0

ROOTS="$HOME/MARL_run_* $HOME/MARL_code $HOME/MARL_code_remote $HOME/MARL_perf $HOME/MARL_watch"
DUMP="$TREE/audit_history/trackB_git_log.txt"
FLOORS="$TREE/tools/seed_audit_floors.txt"
LOG="logs/seed_audit_${LO}_${HI}.log"

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
  --out "output/seed_audit_${LO}_${HI}.json" --pt --git \
  --git-no-repo "$HOME/MARL_code" \
  --git-log "$DUMP" \
  --skip-dir audit_history \
  --false-positive "seed 42, 200 island ticks" \
  --min-files 6077 --min-text 2455 --min-json 224 --min-pickles 184 --min-pt 2563 \
  --min-per-root 10 --min-root-file "$FLOORS" --tree "$TREE" \
  -v > "$LOG" 2>&1
rc=$?
echo "EXIT_${LO}_${HI}=$rc" >> "$LOG"
echo "EXIT_${LO}_${HI}=$rc"
exit $rc
