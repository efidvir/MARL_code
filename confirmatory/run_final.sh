#!/bin/bash
# THE STAGED LAUNCHER for the final corrected rerun (WP RUN-CHAIN).
#
#   bash confirmatory/run_final.sh --profile rehearsal
#   bash confirmatory/run_final.sh --profile real --from S6
#   bash confirmatory/run_final.sh --profile real --shed sensitivity
#   bash confirmatory/run_final.sh --profile rehearsal --dry-run
#   bash confirmatory/run_final.sh --profile rehearsal --waive-dep S6
#   bash confirmatory/run_final.sh --profile r3lite --audit-ok
#
# WHAT IT GUARANTEES (COMPUTE_PLAN.md; see docs/RUN_CHAIN.md for the mapping)
#  * stages run in DEPENDENCY order, and a stage does not start until every
#    invocation of every stage it depends on is recorded successful;
#  * every stage is split into invocations of at most 64 passes along whole
#    (arm, seed-block) boundaries (80 for the two single-arm invocations of
#    R3-lite, chain_plan.R3_SINGLE_ARM_MAX_PASSES), each writing its own
#    pickle whose sha256 goes in the chain ledger -- a crash costs at most
#    one invocation;
#  * PREFLIGHT before every invocation, recorded and gated (memory, disk,
#    leftover jobs of ours, tree digest, thread pinning, PYTHONHASHSEED);
#  * training is EXCLUSIVE, enforced by a lock that evaluation invocations
#    must take in shared mode, not by a comment;
#  * RESTART RULE: a FAILED invocation is re-run exactly once, UNCHANGED
#    (same job script, same env file, same tree).  An invocation that
#    SUCCEEDED is never re-run, and no invocation is ever re-scoped;
#  * RESUME: an invocation whose artifact exists with the sha256 recorded in
#    the ledger is skipped, and nothing else is redone;
#  * SHED: dropping the secondary stages is an explicit operator action
#    (--shed), recorded in the ledger, and only in the plan's order;
#  * AUDIT GATE (R3-lite): no stage at or after the one the plan gates (S6)
#    starts without a passing freeze-2 seed audit of the sealed tree; hits
#    left to classify need the recorded operator action --audit-ok;
#  * OPERATOR ACTIONS THE PLAN RESTRICTS (R3-lite): for the confirmatory
#    profile --invalidate is refused from S6 on (and for everything once S6
#    has started), --waive-dep altogether and --max-attempts with any value
#    but 2; in both R3-lite profiles --waive-order on the edges "S7 after
#    S9x" and "S12 after S7e" is honoured ONLY when the ledger records that
#    the restart rule is exhausted for the stage the plan names there
#    (chain_plan.ORDER_WAIVERS_R3LITE).  Every refusal is recorded;
#  * THE RESTART RULE IS THE LEDGER'S: attempts are counted from the ledger
#    (a start without an end is an attempt), and an invocation the ledger
#    calls terminal (tools/stage_report.py --restart-state) is never started
#    again -- by the same function that releases those waivers.
#
# Exit status: 0 the requested stages all completed; 1 the chain stopped.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/chain_lib.sh"

PROFILE=""; ONLY_STAGES=""; FROM_STAGE=""; DRY=0; SHED=""; WAIVE=""; INVALIDATE=""
WAIVE_DEP=""; AUDIT_OK=0
MAX_ATTEMPTS=2; PREFLIGHT_RETRIES=3; PREFLIGHT_WAIT=300

usage () { sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)      PROFILE="$2"; shift 2 ;;
    --stages)       ONLY_STAGES="$2"; shift 2 ;;
    --from)         FROM_STAGE="$2"; shift 2 ;;
    --shed)         SHED="$2"; shift 2 ;;
    --waive-order)  WAIVE="$2"; shift 2 ;;
    --waive-dep)    WAIVE_DEP="$2"; shift 2 ;;
    --invalidate)   INVALIDATE="$2"; shift 2 ;;
    --audit-ok)     AUDIT_OK=1; shift ;;
    --dry-run)      DRY=1; shift ;;
    --no-watchdog)  CHAIN_NO_WATCHDOG=1; export CHAIN_NO_WATCHDOG; shift ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --preflight-retries) PREFLIGHT_RETRIES="$2"; shift 2 ;;
    --preflight-wait)    PREFLIGHT_WAIT="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$PROFILE" ] || usage
# An attempt count is a positive integer for every profile: '[ 3 -gt x ]' is
# an error that bash counts as false, and the restart loop below would then
# re-run a failing job for ever.
case "$MAX_ATTEMPTS" in
  ''|*[!0-9]*|0*) echo "REFUSING: --max-attempts $MAX_ATTEMPTS: not a positive integer" >&2; exit 2 ;;
esac
cd "$CHAIN_TREE" || exit 1
chain_setup "$PROFILE"
trap 'chain_lock_release' EXIT

# ---------------------------------------------------------------- restricted operator actions
# PREREGISTRATION_R3 section 13: "an invocation that succeeded is never
# re-run", and S9x and S7e "are never shed, skipped or repeated for any
# content of any report".  --shed was refused for the R3-lite graph from the
# start; --invalidate, the two waivers and --max-attempts were not, and each
# of them is a way to redo or to step around a confirmatory invocation after
# a report has been read.  What a profile's plan restricts is refused before
# anything is started, and the refusal itself is written to the ledger: an
# attempt is a fact a reader of the run is entitled to.  The R2 profiles and
# the R3-lite rehearsal restrict nothing and pass straight through.
refuse_action () {        # $1 the action as typed   $2.. why
  local what="$1"; shift
  echo "REFUSING: $what: $*" >&2
  chain_say "REFUSED $what (recorded): $*"
  chain_ledger event=note reason=operator_action_refused "action=$what" "why=$*"
  exit 2
}
# RESTRICTED: the plan of this profile declares its waivers.  Anything but a
# clear 'free' from the plan counts as restricted (a plan that cannot be read
# restricts).
RESTRICTED=0
[ "$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --dep-waiver)" = free ] || RESTRICTED=1
# The restart rule is one unchanged re-run: two attempts, no more and no
# fewer.  More would repeat a confirmatory invocation beyond the rule; fewer
# would stop a stage for good after a single failure.
if [ "$MAX_ATTEMPTS" != 2 ] && [ "$RESTRICTED" = 1 ]; then
  refuse_action "--max-attempts $MAX_ATTEMPTS" "the registered restart rule of profile $PROFILE is one unchanged re-run (two attempts)"
fi

# ---------------------------------------------------------------- shed order
# "Shed stages in this order -- S10/S11 sensitivity, S9 adapt-then-freeze,
# S8 untrained control."  A later group may not be shed while an earlier one
# is still scheduled: that is what makes the order a rule and not a comment.
SHED_GROUPS=" $(echo "$SHED" | tr ',' ' ') "
if [ -n "$SHED" ]; then
  "$PY" - "$HERE" "$PROFILE" $(echo "$SHED" | tr ',' ' ') <<'PYX' || { [ "$RESTRICTED" = 0 ] || chain_ledger event=note reason=operator_action_refused "action=--shed $SHED" "why=the plan of profile $PROFILE sheds nothing, or not in that order (message on stderr)"; exit 2; }
import sys
sys.path.insert(0, sys.argv[1])
import chain_plan
# The shed groups are a property of the PROFILE's stage graph: the R2
# profiles shed in COMPUTE_PLAN's order, an R3-lite profile sheds nothing
# (docs/PREREGISTRATION_R3.md: every listed analysis is run and reported).
want, order = set(sys.argv[3:]), chain_plan.shed_order(sys.argv[2])
if want and not order:
    sys.exit('REFUSING: --shed %s: profile %s sheds nothing (its plan declares '
             'no shed group)' % (', '.join(sorted(want)), sys.argv[2]))
unknown = want - set(order)
if unknown:
    sys.exit('REFUSING: unknown shed group(s) %s; known: %s'
             % (', '.join(sorted(unknown)), ', '.join(order)))
for i, g in enumerate(order):
    missing = [o for o in order[:i] if o not in want]
    if g in want and missing:
        sys.exit('REFUSING: --shed %s while %s is still scheduled. The plan sheds '
                 'in this order: %s' % (g, ', '.join(missing), ', '.join(order)))
print('shed order OK: %s' % ', '.join(g for g in order if g in want))
PYX
fi

# ---------------------------------------------------------------- the waivers and --invalidate
# (the restricted operator actions, continued: see refuse_action above)
if [ -n "$WAIVE_DEP" ] && [ "$RESTRICTED" = 1 ]; then
  refuse_action "--waive-dep $WAIVE_DEP" "the plan of profile $PROFILE declares no waivable dependency"
fi
if [ -n "$WAIVE" ]; then
  WAIVABLE=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --waivable-order-stages) || exit 2
  if [ "$WAIVABLE" != '*' ]; then
    for w in $(echo "$WAIVE" | tr ',' ' '); do
      case " $WAIVABLE " in
        *" $w "*) ;;
        *) refuse_action "--waive-order $w" "the plan of profile $PROFILE declares an order waiver only for: ${WAIVABLE:-no stage}, and only after a terminal failure" ;;
      esac
    done
  fi
fi

# An explicit operator action, recorded: everything the ledger holds for these
# invocations stops counting, so they are redone on the next run and their
# restart budget starts again.  Nothing else is touched.  This exists because
# the alternative -- editing the ledger by hand -- leaves no record.
if [ -n "$INVALIDATE" ]; then
  # all of them are checked before any one is applied
  for id in $(echo "$INVALIDATE" | tr ',' ' '); do
    IWHY=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --invalidate-refusal "$id") || exit 2
    [ -z "$IWHY" ] || refuse_action "--invalidate $id" "$IWHY"
  done
  # ... and once a confirmatory stage has STARTED, nothing of a restricted
  # profile is invalidated any more -- not the stages before it either:
  # redoing S5 would seal FREEZE_FINAL again after confirmatory passes exist.
  # Fail closed: anything but a clear NOT_STARTED refuses.
  if [ "$RESTRICTED" = 1 ]; then
    NOINV=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --profile-field no_invalidate_from 2>/dev/null) || NOINV=""
    STARTED=""
    [ -z "$NOINV" ] || STARTED=$("$PY" "$CHAIN_REPORT_PY" --stage-started "$CHAIN_LEDGER" --profile "$PROFILE" --from-stage "$NOINV")
    case "$STARTED" in
      NOT_STARTED*) ;;
      *) refuse_action "--invalidate $INVALIDATE" "the ledger shows that ${NOINV:-a confirmatory stage} or a later stage has started (${STARTED:-the ledger or the plan could not be read}): nothing of this chain is invalidated any more, the seal of S5 included" ;;
    esac
  fi
  for id in $(echo "$INVALIDATE" | tr ',' ' '); do
    chain_say "INVALIDATE $id (operator action; its previous result no longer counts)"
    chain_ledger event=invalidate "inv=$id" "reason=operator"
  done
  exit 0
fi

# ---------------------------------------------------------------- audit gate
# R3-lite (docs/PREREGISTRATION_R3.md section 13): the freeze-2 seed audit is
# a STOP CONDITION -- "any result-data hit stops the run before S6" -- and it
# used to live in the checklist only: without --stages/--from this launcher
# ran S0 to S12 straight through and never read an audit report.  A profile
# whose plan declares a gate (r3lite: before S6) now cannot start the gated
# stage, or any stage after it, unless the report the committed driver wrote
# is the audit of THIS sealed tree and passed (tools/stage_report.py
# --audit-gate).  A report that passed the result-data gate but still lists
# text, file-name or history hits needs ONE recorded operator action,
# --audit-ok, taken after every such hit has been classified in writing; it
# is bound to that report by its sha256 and can never excuse a result-data
# hit or a coverage failure.  The R2 profiles and the rehearsal declare no
# gate and reach none of this.
AUDIT_BEFORE=""; AUDIT_GATE=""; AUDIT_REPORT=""
read -r AUDIT_BEFORE AUDIT_GATE AUDIT_REPORT <<< "$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --audit-gate)"
audit_gate_eval () {      # [$1 ledger, or empty: ignore recorded acknowledgements] -> 0 pass, 3 unclassified, else fail
  local lo hi man
  read -r lo hi <<< "$($PY -c "import sys; sys.path.insert(0, sys.argv[1]); import chain_plan as c; s = sorted(c.PROFILES[sys.argv[2]]['cf_seeds']); print(s[0], s[-1])" "$HERE" "$PROFILE")"
  man=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --field s0_manifest artifact)
  "$PY" "$CHAIN_REPORT_PY" --audit-gate "$CHAIN_TREE/$AUDIT_REPORT" \
      --freeze-file "$CHAIN_DIR/FREEZE_FINAL_${PROFILE}.txt" \
      --expect-window "$lo,$hi" --expect-planning-marker "$man" --tree "$CHAIN_TREE" \
      --audit-ledger "${1:-}" --json-out "$CHAIN_PREDIR/audit_gate.$(date -u +%H%M%S).json" | tee -a "$CHAIN_LOG"
  return "${PIPESTATUS[0]}"
}
if [ "$AUDIT_OK" = 1 ]; then
  if [ -z "$AUDIT_BEFORE" ]; then
    echo "REFUSING: --audit-ok: profile $PROFILE declares no seed-audit gate" >&2
    exit 2
  fi
  audit_gate_eval ""
  case $? in
    0) chain_say "AUDIT-OK not recorded: the audit report is clean, there is nothing to acknowledge" ;;
    3) ASHA=$(chain_sha "$CHAIN_TREE/$AUDIT_REPORT")
       chain_say "AUDIT-OK $AUDIT_REPORT sha256 ${ASHA:0:16} (operator action, recorded): every remaining"
       chain_say "         text, file-name and history hit of this report has been classified in"
       chain_say "         writing (PREREGISTRATION_R3 section 13(b)); the result-data gate passed"
       chain_ledger event=note reason=audit_classified "report=$AUDIT_REPORT" \
                    "report_sha256=$ASHA" "gate=$AUDIT_GATE" ;;
    *) chain_say "REFUSING --audit-ok: the audit report is missing, is not the audit of this sealed"
       chain_say "         tree, or did not pass the result-data gate or coverage (see [AUDIT-GATE]"
       chain_say "         above).  That is a stop, not something to acknowledge."
       chain_ledger event=stop reason=audit_ok_refused "report=$AUDIT_REPORT"
       exit 1 ;;
  esac
  exit 0
fi
AUDIT_ARMED=0; AUDIT_DONE=0

STAGES_ALL=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --stages)
chain_say "=========================================================="
chain_say "CHAIN START profile=$PROFILE tree=$CHAIN_TREE dry_run=$DRY"
chain_say "stages: $STAGES_ALL"
[ -n "$SHED" ] && chain_say "OPERATOR SHED: $SHED (explicit, recorded; none of these gate the primary analysis)"
chain_ledger event=chain_start "stages=$STAGES_ALL" "shed=$SHED" "dry_run=$DRY" \
             "tree=$CHAIN_TREE" "argv=$*"
# A profile whose stage graph leaves R2 stages out (R3-lite) says so at every
# chain start, in the log and in the ledger, with the plan's recorded reason:
# a stage is absent because the plan DROPPED it, never silently.  The R2
# profiles drop nothing and print nothing here.
while IFS=$'\t' read -r DST DWHY; do
  [ -n "$DST" ] || continue
  chain_say "DROPPED $DST (by the plan of profile $PROFILE, recorded): $DWHY"
  chain_ledger event=dropped "stage=$DST" "reason=$DWHY"
done < <($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --dropped)

stage_shed_group () {     # $1 stage -> its shed group, or empty
  $PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --shed-group-of "$1"
}

inv_status () {           # $1 inv -> "STATUS sha attempts"
  "$PY" "$CHAIN_REPORT_PY" --ledger-status "$CHAIN_LEDGER" --inv "$1"
}

stage_complete () {       # $1 stage -> 0 when every invocation succeeded
  local st="$1" id s
  local ids; ids=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --ids "$st")
  [ -n "$ids" ] || return 1
  for id in $ids; do
    s=$(inv_status "$id" | awk '{print $1}')
    [ "$s" = SUCCESS ] || return 1
  done
  return 0
}

# ---------------------------------------------------------------- one invocation
run_invocation () {       # $1 invocation id
  local id="$1" attempt=1 status sha attempts kind pfkind
  eval "$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --emit "$id")"
  # Test hook: ONLY the 'test' profile may replace the stage script, so the
  # restart/resume/gate rules can be exercised without compute.  The real and
  # rehearsal profiles cannot reach this.
  if [ "$PROFILE" = test ] && [ -n "${CHAIN_SCRIPT_OVERRIDE:-}" ]; then
    INV_SCRIPT="$CHAIN_SCRIPT_OVERRIDE"
  fi

  # --- shed?
  if [ -n "$INV_SHED_GROUP" ] && [ "${SHED_GROUPS/ $INV_SHED_GROUP /}" != "$SHED_GROUPS" ]; then
    chain_say "  SHED   $id ($INV_SHED_GROUP) - operator decision, recorded, not automatic"
    chain_ledger event=shed "inv=$id" "stage=$INV_STAGE" "group=$INV_SHED_GROUP"
    return 0
  fi

  # --- resume: never redo work that succeeded
  read -r status sha attempts <<< "$(inv_status "$id")"
  if [ "$status" = SUCCESS ]; then
    local now; now=$(chain_sha "$CHAIN_TREE/$INV_ARTIFACT")
    if [ "$now" = "$sha" ] && [ "$now" != '-' ]; then
      chain_say "  RESUME $id already complete (sha256 ${sha:0:16} matches) - skipped"
      chain_ledger event=skip "inv=$id" "stage=$INV_STAGE" "status=ok" "sha256=$sha" \
                   "artifact=$INV_ARTIFACT" "reason=resume"
      return 0
    fi
    chain_say "  STOP   $id is recorded SUCCESSFUL but its artifact no longer matches"
    chain_say "         ledger ${sha:0:16} vs on disk ${now:0:16} ($INV_ARTIFACT)"
    chain_say "         This is not something to re-run: a completed invocation changed"
    chain_say "         under us.  Escalate; do not analyse this set."
    chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=artifact_changed" \
                 "ledger_sha256=$sha" "disk_sha256=$now"
    return 1
  fi
  case "$INV_KIND" in
    eval)  pfkind=eval ;;
    train) pfkind=train ;;
    *)     pfkind=tool ;;
  esac

  # --- THE RESTART RULE, read from the ledger -- not counted by this launch.
  #
  # tools/stage_report.py --restart-state answers three things for the
  # invocation since its last invalidate: how many attempts were STARTED (a
  # start without an end is an attempt: the launcher or the host died while
  # it ran, and it ran), how the last one ended, and whether the rule is
  # exhausted.  It is the SAME function that releases a failure-only waiver
  # and that makes S12 record a family as not completed, so an invocation
  # the ledger calls terminal is never started again -- "family E is not
  # completed" and "S9x can still be run" cannot both be true.  Counting
  # from 1 in every launch, as this function used to, gave an invocation
  # with a failed attempt on record a fresh pair: a third execution, which
  # "repeated once, unchanged" does not allow.
  local rs_n rs_last rs_term rs_cause
  rs_state () {
    rs_n=""; rs_last=""; rs_term=""; rs_cause=""
    read -r rs_n rs_last rs_term rs_cause <<< "$("$PY" "$CHAIN_REPORT_PY" --restart-state "$CHAIN_LEDGER" --inv "$id")"
    case "$rs_n" in ''|*[!0-9]*) return 1 ;; esac
    case "$rs_last:$rs_term" in none:0|open:0|ok:0|fail:0|fail:1) return 0 ;; *) return 1 ;; esac
  }
  if ! rs_state; then
    chain_say "  STOP   $id: the restart state could not be read from the ledger ('$rs_n $rs_last $rs_term')."
    chain_say "         Nothing is started on a ledger that cannot be read."
    chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=restart_state_unreadable"
    return 1
  fi
  # An attempt that has a start and no end used the budget, but the ledger
  # cannot say that it FAILED until it is closed -- and it is closed only when
  # nothing of it can still be running: no process carries this invocation's
  # job script (that covers an analysis as well as a simulation), and a
  # preflight finds no job of ours alive in this tree.  Needed only when the
  # budget is used up: otherwise the next attempt simply follows.
  inv_job_alive () {      # $1 inv -> 0 when a process still runs that invocation's job script
    local want="$CHAIN_JOBDIR/$1.sh" p a
    for p in /proc/[0-9]*; do
      [ "${p#/proc/}" = "$$" ] && continue
      [ -r "$p/cmdline" ] || continue
      while IFS= read -r -d '' a; do
        [ "$a" = "$want" ] && return 0
      done < "$p/cmdline" 2>/dev/null
    done
    return 1
  }
  if [ "$rs_last" = open ] && [ "$rs_n" -ge "$MAX_ATTEMPTS" ] && [ "$DRY" = 0 ]; then
    if inv_job_alive "$id"; then
      chain_say "  STOP   $id: attempt $rs_n has a start and no end, and a process still runs its job"
      chain_say "         script ($CHAIN_JOBDIR/$id.sh).  Nothing is closed and nothing is started;"
      chain_say "         re-issue the command when that attempt has ended."
      chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=attempt_open" "attempts=$rs_n" "job=alive"
      return 1
    elif chain_preflight "$pfkind" "$id" "$INV_STAGE" "$INV_WORKERS"; then
      chain_say "  CLOSE  $id attempt $rs_n has a start and no end in the ledger, and the preflight"
      chain_say "         finds no job of ours alive: recorded as a failed attempt (exit=killed)"
      chain_ledger event=end "inv=$id" "stage=$INV_STAGE" "kind=$INV_KIND" "attempt=$rs_n" \
          status=fail exit=killed gate=- wall_s=0 sha256=- "artifact=$INV_ARTIFACT" passes=0 \
          watchdog_restarts=0 closed_by=launcher
      if ! rs_state; then
        chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=restart_state_unreadable"
        return 1
      fi
    else
      chain_say "  STOP   $id: attempt $rs_n has a start and no end, and the preflight does not show the"
      chain_say "         tree free of our jobs.  Nothing is closed and nothing is started; re-issue the"
      chain_say "         command when that attempt has ended."
      chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=attempt_open" "attempts=$rs_n"
      return 1
    fi
  fi
  if [ "$rs_n" -ge "$MAX_ATTEMPTS" ] || { [ "$MAX_ATTEMPTS" = 2 ] && [ "$rs_term" = 1 ]; }; then
    chain_say "  STOP   $id has already failed $rs_n time(s): the pre-registered"
    chain_say "         budget is one unchanged re-run.  Stopping the chain."
    [ "$rs_term" = 1 ] && chain_say "         The ledger: $rs_cause"
    chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=restart_budget_used" \
                 "attempts=$rs_n" "terminal=$rs_term"
    return 1
  fi
  attempt=$((rs_n + 1))

  # --- preflight, gated
  local tries=0
  until chain_preflight "$pfkind" "$id" "$INV_STAGE" "$INV_WORKERS"; do
    local rc=$?
    if [ "$rc" -eq 2 ]; then
      chain_say "  ABORT  $id: preflight abort level (below the training memory floor)"
      chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=preflight_abort"
      return 1
    fi
    tries=$((tries + 1))
    if [ "$tries" -ge "$PREFLIGHT_RETRIES" ]; then
      chain_say "  STOP   $id: preflight failed $tries times; not launching"
      chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=preflight_failed"
      return 1
    fi
    chain_say "  wait   $id: preflight gate not met; retrying in ${PREFLIGHT_WAIT}s ($tries/$PREFLIGHT_RETRIES)"
    sleep "$PREFLIGHT_WAIT"
  done

  # --- resolve the checkpoint
  local ckpt=""
  case "$INV_CKPT_KIND" in
    # 'ensemble' (R3-lite S9x): a comma-separated list of checkpoints, which
    # the harness loads as a logit-averaged ensemble.
    literal|replicate|ensemble) ckpt="$INV_CKPT_VALUE" ;;
    selected)
      local selfile="confirmatory/SELECTED_${PROFILE}.txt"
      if [ ! -s "$selfile" ] && [ "$DRY" = 1 ]; then
        ckpt="<selected-by-S3b-at-run-time>"
      elif [ ! -s "$selfile" ]; then
        chain_say "  STOP   $id needs the selected checkpoint but $selfile is missing"
        chain_ledger event=stop "inv=$id" "reason=no_selected_checkpoint"
        return 1
      fi
      ckpt=$(head -1 "$selfile")
      ;;
    none|'') ckpt="" ;;
  esac
  if [ -n "$ckpt" ] && [ "$DRY" = 0 ]; then
    # every member of the checkpoint argument (one path, or the ensemble's
    # comma-separated list) must exist
    local ckp ckmiss=""
    local OLDIFS="$IFS"; IFS=','
    for ckp in $ckpt; do
      [ -e "$CHAIN_TREE/$ckp" ] || ckmiss="${ckmiss:+$ckmiss,}$ckp"
    done
    IFS="$OLDIFS"
    if [ -n "$ckmiss" ]; then
      chain_say "  STOP   $id: checkpoint $ckmiss does not exist"
      chain_ledger event=stop "inv=$id" "reason=missing_checkpoint" "checkpoint=$ckmiss"
      return 1
    fi
  fi
  # --- the PINNED checkpoint digest (R3-lite).  An invocation whose plan
  # entry carries ckpt_sha256 may only run on checkpoint files with exactly
  # those digests, member by member and in order.  The R2 profiles carry no
  # pin (their checkpoints are trained and selected inside the run) and skip
  # this block.  A pin that cannot be compared is a STOP, never a pass.
  if [ -n "${INV_CKPT_SHA256:-}" ] && [ "$DRY" = 0 ]; then
    local got="" ckq
    local OLDIFS2="$IFS"; IFS=','
    for ckq in $ckpt; do
      got="${got:+$got,}$(chain_sha "$CHAIN_TREE/$ckq")"
    done
    IFS="$OLDIFS2"
    if [ -z "$ckpt" ] || [ "$got" != "$INV_CKPT_SHA256" ]; then
      chain_say "  STOP   $id: checkpoint digest is not the pinned one"
      chain_say "         pinned  $INV_CKPT_SHA256"
      chain_say "         on disk ${got:-<no checkpoint resolved>} (${ckpt:-none})"
      chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=checkpoint_sha256_mismatch" \
                   "pinned=$INV_CKPT_SHA256" "disk=$got" "checkpoint=$ckpt"
      return 1
    fi
    chain_say "  PIN    $id checkpoint sha256 matches the plan's pin ($(echo "$INV_CKPT_SHA256" | tr ',' '\n' | wc -l) file(s))"
  fi

  # --- lock (training excludes everything of ours, and vice versa)
  local mode=shared
  [ "$INV_EXCLUSIVE" = 1 ] && mode=exclusive
  if ! chain_lock_acquire "$mode" "$id"; then
    chain_ledger event=stop "inv=$id" "reason=lock_refused" "mode=$mode"
    return 1
  fi

  # --- env block and job script (written ONCE; a re-run reuses them verbatim)
  chain_write_env "$id" "$INV_ENV" "$ckpt" "$CHAIN_WORKERS" "${INV_HASHSEED:-0}"
  local cmd
  cmd=$(CHAIN_PROFILE="$PROFILE" bash "$INV_SCRIPT" --print-command "$PROFILE" "$id" "$ckpt") || {
    chain_say "  STOP   $id: $INV_SCRIPT could not build a command"
    chain_ledger event=stop "inv=$id" "reason=no_command"
    chain_lock_release
    return 1
  }
  chain_write_job "$id" "$INV_MARKER" "$cmd"

  # --- the environment the job will run with is EXACTLY the plan's (item 5)
  # Deterministic, so it is not retried: a mismatch is a defect to fix.
  if ! chain_env_check "$id" "${INV_HASHSEED:-0}"; then
    chain_say "  STOP   $id: the job's environment is not the plan's (see [PREFLIGHT-ENV] above)"
    chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=env_mismatch"
    chain_lock_release
    return 1
  fi

  if [ "$DRY" = 1 ]; then
    chain_say "  DRY    $id [$INV_STAGE/$INV_KIND] passes=$INV_EXPECTED_PASSES workers=$CHAIN_WORKERS"
    chain_say "         env: $(grep -c ^export "$CHAIN_ENVFILE") exported variables, checkpoint=${ckpt:-none}"
    chain_say "         cmd: $cmd"
    chain_lock_release
    return 0
  fi

  CHAIN_LAUNCH_ARTIFACT="$INV_ARTIFACT"
  while :; do
    chain_say "  RUN    $id [$INV_STAGE/$INV_KIND] attempt=$attempt passes=$INV_EXPECTED_PASSES workers=$CHAIN_WORKERS"
    chain_ledger event=start "inv=$id" "stage=$INV_STAGE" "kind=$INV_KIND" \
        "attempt=$attempt" "workers=$CHAIN_WORKERS" "expected_passes=$INV_EXPECTED_PASSES" \
        "artifact=$INV_ARTIFACT" "envfile=${CHAIN_ENVFILE#$CHAIN_TREE/}" \
        "job=${CHAIN_JOB#$CHAIN_TREE/}" "cmd=$cmd" "checkpoint=$ckpt"
    chain_launch "$id" "$INV_HOURS" "$INV_MARKER"

    # --- verify what came out, by counting it
    #
    # FRESHNESS FIRST.  chain_launch removed the target before starting, and
    # the artefact must carry an mtime at or after that launch.  Without both
    # halves, a job that exits 0 without writing anything is verified against
    # the PREVIOUS attempt's artefact and recorded as a full success -- the
    # exact shape of defect this campaign keeps producing.
    local gate_rc=0 passes=0 abl=''
    if [ "$CHAIN_RC" = 0 ]; then
      if ! chain_artifact_is_fresh "$INV_ARTIFACT" "$CHAIN_T0"; then
        chain_say "  GATE   $id exited 0 but $CHAIN_FRESH_WHY"
        gate_rc=1
      elif [ -n "$INV_ARTIFACT" ]; then
        chain_say "  GATE   $id $CHAIN_FRESH_WHY"
      fi
    fi
    if [ "$CHAIN_RC" = 0 ] && [ "$gate_rc" = 0 ] && [ "$INV_EXPECTED_PASSES" -gt 0 ]; then
      $PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --keys "$id" > "$CHAIN_JOBDIR/$id.keys"
      # R3-lite: the pickle must also RECORD the pinned checkpoint digest(s)
      # (provenance checkpoint_sha256); empty for the R2 profiles, where the
      # flag is not passed and the gate is what it was.
      local pinargs=()
      [ -n "${INV_CKPT_SHA256:-}" ] && pinargs=(--expect-checkpoint-sha256 "$INV_CKPT_SHA256")
      $PY "$CHAIN_REPORT_PY" --verify "$CHAIN_TREE/$INV_ARTIFACT" \
          --keys-file "$CHAIN_JOBDIR/$id.keys" \
          --expect-hashseed "${INV_HASHSEED:-0}" \
          --expect-marl-env-file "$CHAIN_MARLENV" ${pinargs[@]+"${pinargs[@]}"} | tee -a "$CHAIN_LOG"
      gate_rc=${PIPESTATUS[0]}
      passes=$INV_EXPECTED_PASSES
      # An ABLATION invocation must DIFFER from the run it is an ablation of.
      # Counting its keys proves it ran; it does not prove the switch took.
      if [ "$gate_rc" = 0 ] && [ -n "$INV_ABLATION_OF" ]; then
        local basepkls
        basepkls=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --artifacts "$INV_ABLATION_OF" --same-ticks-as "$id")
        $PY "$CHAIN_REPORT_PY" --ablation-diff "$CHAIN_TREE/$INV_ARTIFACT" \
            --baseline "$basepkls" --ablation-arms "$INV_ARMS" \
            --ablation-series "$INV_ABLATION_SERIES" \
            --ablation-counter "$INV_ABLATION_COUNTER" | tee -a "$CHAIN_LOG"
        case "${PIPESTATUS[0]}" in
          0) abl=exercised ;;
          3) abl=NOT_EXERCISED
             chain_say "  NOTE   $id is recorded NOT EXERCISED: its output is identical to"
             chain_say "         its baseline and the mechanism it switches off never fired"
             chain_say "         in the baseline either.  The passes are valid; the ABLATION"
             chain_say "         is not evidence of anything and must not be reported as"
             chain_say "         tested." ;;
          *) abl=FAILED; gate_rc=1 ;;
        esac
      fi
    fi
    # A checkpoint written by a tool step (S0's untrained actors) records
    # its environment too: it must be the plan's (item 5).
    if [ "$CHAIN_RC" = 0 ] && [ "$gate_rc" = 0 ] && [ "$INV_EXPECTED_PASSES" -eq 0 ] \
       && [ "${INV_ARTIFACT%.pt}" != "$INV_ARTIFACT" ]; then
      $PY "$CHAIN_REPORT_PY" --check-marl-env "$CHAIN_TREE/$INV_ARTIFACT" \
          --expect-marl-env-file "$CHAIN_MARLENV" | tee -a "$CHAIN_LOG"
      gate_rc=${PIPESTATUS[0]}
    fi
    local sha256; sha256=$(chain_sha "$CHAIN_TREE/$INV_ARTIFACT")
    local st=ok
    { [ "$CHAIN_RC" != 0 ] || [ "$gate_rc" != 0 ]; } && st=fail
    chain_ledger event=end "inv=$id" "stage=$INV_STAGE" "kind=$INV_KIND" \
        "attempt=$attempt" "status=$st" "exit=$CHAIN_RC" "gate=$gate_rc" \
        "wall_s=$CHAIN_WALL_S" "sha256=$sha256" "artifact=$INV_ARTIFACT" \
        "passes=$passes" "workers=$CHAIN_WORKERS" "registered=$CHAIN_REGISTERED" \
        "watchdog_restarts=${CHAIN_WD_RESTARTS:-0}" "ablation=$abl"
    chain_say "  ${st^^}     $id exit=$CHAIN_RC gate=$gate_rc wall=${CHAIN_WALL_S}s sha=${sha256:0:16} watchdog_restarts=${CHAIN_WD_RESTARTS:-0}"

    [ "$st" = ok ] && { chain_lock_release; return 0; }

    # --- the pre-registered restart rule
    if [ "${CHAIN_WD_RESTARTS:-0}" -ge 1 ]; then
      chain_say "  STOP   $id: the watchdog already performed the one unchanged re-run"
      chain_ledger event=stop "inv=$id" "reason=watchdog_used_the_rerun"
      chain_lock_release; return 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
      chain_say "  STOP   $id failed twice. Escalate: do NOT analyse a partial set,"
      chain_say "         do NOT change arms or seeds to fill the gap, and do NOT"
      chain_say "         substitute a pass from another invocation or tree."
      chain_ledger event=stop "inv=$id" "reason=failed_twice"
      chain_lock_release; return 1
    fi
    chain_say "  RERUN  $id UNCHANGED (pre-registered: one re-run on technical failure)"
    chain_say "         same job script, same env file, same tree, same seeds and arms"
  done
}

# ---------------------------------------------------------------- stage loop
RUNNING=0
rc_all=0
for ST in $STAGES_ALL; do
  # the gated stage and every stage after it (armed here, before any skip,
  # so that --from/--stages past the gated stage cannot step around it)
  [ -n "$AUDIT_BEFORE" ] && [ "$ST" = "$AUDIT_BEFORE" ] && AUDIT_ARMED=1
  if [ -n "$FROM_STAGE" ] && [ "$RUNNING" = 0 ]; then
    [ "$ST" = "$FROM_STAGE" ] && RUNNING=1 || { chain_say "SKIP  $ST (before --from $FROM_STAGE)"; continue; }
  fi
  if [ -n "$ONLY_STAGES" ]; then
    case " $(echo "$ONLY_STAGES" | tr ',' ' ') " in
      *" $ST "*) ;;
      *) chain_say "SKIP  $ST (not in --stages $ONLY_STAGES)"; continue ;;
    esac
  fi
  G=$(stage_shed_group "$ST")
  if [ -n "$G" ] && [ "${SHED_GROUPS/ $G /}" != "$SHED_GROUPS" ]; then
    chain_say "SHED  $ST (group $G) - explicit operator decision, recorded"
    chain_ledger event=shed "stage=$ST" "group=$G"
    continue
  fi

  # --- the seed-audit gate, once per launch, before the first stage at or
  #     after the gated one that this launch would run
  if [ "$AUDIT_ARMED" = 1 ] && [ "$AUDIT_DONE" = 0 ]; then
    if [ "$DRY" = 1 ]; then
      chain_say "PLAN  seed-audit gate $AUDIT_GATE before $ST: $AUDIT_REPORT (not evaluated in a dry run)"
    else
      chain_say "AUDIT gate $AUDIT_GATE before $ST: $AUDIT_REPORT"
      audit_gate_eval "$CHAIN_LEDGER"
      arc=$?
      if [ "$arc" != 0 ]; then
        if [ "$arc" = 3 ]; then
          chain_say "STOP  $ST: the freeze-2 seed audit lists hits that are not classified."
          chain_say "      Classify every one in writing, then record it with --audit-ok and"
          chain_say "      re-issue this command.  No confirmatory pass has been started by it."
          chain_ledger event=stop "stage=$ST" "reason=audit_unclassified" "report=$AUDIT_REPORT"
        else
          chain_say "STOP  $ST: no passing freeze-2 seed audit of this sealed tree"
          chain_say "      ($AUDIT_REPORT; see [AUDIT-GATE] above).  PREREGISTRATION_R3 section 13:"
          chain_say "      the audit runs after FREEZE_FINAL is sealed and before any confirmatory"
          chain_say "      pass.  Run it (docs/FREEZE_CHECKLIST.md R.5); do not work around this."
          chain_ledger event=stop "stage=$ST" "reason=audit_gate_failed" "report=$AUDIT_REPORT"
        fi
        rc_all=1; break
      fi
      chain_ledger event=audit_gate "stage=$ST" "status=pass" "gate=$AUDIT_GATE" \
                   "report=$AUDIT_REPORT" "report_sha256=$(chain_sha "$CHAIN_TREE/$AUDIT_REPORT")"
    fi
    AUDIT_DONE=1
  fi

  eval "$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --stage-deps "$ST")"
  [ "$DRY" = 1 ] && chain_say "PLAN  $ST deps=[${DEPS:--}] order_after=[${ORDER_AFTER:--}]"
  for d in $DEPS; do
    [ "$DRY" = 1 ] && break
    if ! stage_complete "$d"; then
      DG=$(stage_shed_group "$d")
      if [ -n "$DG" ] && [ "${SHED_GROUPS/ $DG /}" != "$SHED_GROUPS" ]; then continue; fi
      # A hard dependency may be waived ONLY by an explicit operator action
      # that names the dependent stage, and it is recorded in the ledger --
      # the same shape as --waive-order.  It exists because a stage can be
      # unrunnable for a reason outside this track (S5b runs the pre-registered
      # analysis, which cannot execute below 1400 ticks: REQUEST R11), and the
      # alternative an operator reaches for is editing the ledger by hand,
      # which leaves no record at all.
      DEP_WAIVED=0
      case " $(echo "$WAIVE_DEP" | tr ',' ' ') " in *" $ST "*) DEP_WAIVED=1 ;; esac
      if [ "$DEP_WAIVED" = 1 ]; then
        chain_say "WAIVE $ST hard dependency $d (EXPLICIT operator waiver, recorded)"
        chain_say "      $d has NOT run.  Whatever $d would have established is"
        chain_say "      UNPROVEN for this chain and must be reported as unproven."
        chain_ledger event=note "stage=$ST" "reason=dep_waived" "dep=$d"
        continue
      fi
      chain_say "STOP  $ST: hard dependency $d is not complete"
      chain_say "      Re-run with --waive-dep $ST only if $d is known to be"
      chain_say "      unrunnable for a reason outside this chain; it is recorded."
      chain_ledger event=stop "stage=$ST" "reason=dependency_incomplete" "dep=$d"
      rc_all=1; break 2
    fi
  done
  for o in $ORDER_AFTER; do
    [ "$DRY" = 1 ] && break
    if ! stage_complete "$o"; then
      WAIVED=0
      case " $(echo "$WAIVE" | tr ',' ' ') " in *" $ST "*) WAIVED=1 ;; esac
      # What a waiver of THIS edge is worth is the plan's to say:
      #   free      recorded and honoured (every R2 edge, as always);
      #   failure   honoured only when the ledger records that the restart
      #             rule is exhausted for one of the stages the plan names --
      #             so that "S7 without S9x" exists after S9x failed for
      #             good and in no other state, above all not while S9x has
      #             simply not been run yet;
      #   refused   never honoured.
      WRULE=""; WNEED=""
      read -r WRULE WNEED <<< "$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --order-waiver "$ST" "$o")"
      # FAIL CLOSED: `read` leaves WRULE empty when the plan prints nothing,
      # and an empty rule must never fall through to the unrestricted arm.
      case "$WRULE" in
        free|failure|refused) ;;
        *) chain_say "STOP  $ST: the plan did not say how a waiver of the edge after $o is treated"
           chain_say "      ('$WRULE').  Nothing is waived on a plan that cannot be read."
           chain_ledger event=stop "stage=$ST" "reason=order_waiver_unreadable" "after=$o"
           rc_all=1; break 2 ;;
      esac
      if [ "$WRULE" = failure ]; then
        WCAUSE=""
        for wn in $WNEED; do
          if WT=$("$PY" "$CHAIN_REPORT_PY" --terminal-failure "$CHAIN_LEDGER" --profile "$PROFILE" --stage "$wn"); then
            WCAUSE="$WT"; break
          fi
        done
        if [ "$WAIVED" = 1 ] && [ -n "$WCAUSE" ]; then
          chain_say "WAIVE $ST ordering after $o (operator waiver, recorded; honoured ONLY because the"
          chain_say "      ledger records a terminal failure -- $WCAUSE)"
          chain_ledger event=note "stage=$ST" "reason=order_waived" "after=$o" \
                       "rule=failure_only" "cause=$WCAUSE"
        elif [ "$WAIVED" = 1 ]; then
          chain_say "STOP  $ST: --waive-order $ST is REFUSED on the edge after $o.  The plan honours it"
          chain_say "      only after a terminal failure of [$WNEED] (the one unchanged re-run used"
          chain_say "      up), and the ledger records none.  $o has to run first; nothing was started."
          chain_ledger event=stop "stage=$ST" "reason=order_waiver_refused" "after=$o" "needs=$WNEED"
          rc_all=1; break 2
        else
          chain_say "STOP  $ST must run after $o (plan of profile $PROFILE)."
          if [ -n "$WCAUSE" ]; then
            chain_say "      The ledger records a terminal failure -- $WCAUSE."
            chain_say "      To go on without $o: --waive-order $ST (recorded; see PREREGISTRATION_R3 section 13)."
          else
            chain_say "      $o is not complete and has not failed terminally: it has to run first."
          fi
          chain_ledger event=stop "stage=$ST" "reason=order_violated" "after=$o"
          rc_all=1; break 2
        fi
      elif [ "$WRULE" = refused ]; then
        chain_say "STOP  $ST must run after $o: the plan of profile $PROFILE declares no waiver for"
        chain_say "      this edge.  $o has to run first."
        chain_ledger event=stop "stage=$ST" "reason=order_violated" "after=$o"
        rc_all=1; break 2
      elif [ "$WAIVED" = 1 ]; then
        chain_say "WAIVE $ST ordering after $o (explicit operator waiver, recorded)"
        chain_ledger event=note "stage=$ST" "reason=order_waived" "after=$o"
      else
        chain_say "STOP  $ST must run after $o (plan: the 340 policy-independent"
        chain_say "      dev+stress passes go AHEAD of training so a defect in 7 of"
        chain_say "      the 8 arms shows up before 7 h of training is committed)."
        chain_say "      Re-run with --waive-order $ST if that is really intended."
        chain_ledger event=stop "stage=$ST" "reason=order_violated" "after=$o"
        rc_all=1; break 2
      fi
    fi
  done

  chain_say "---- STAGE $ST ----"
  chain_ledger event=stage_start "stage=$ST"

  if [ "$ST" = S2 ]; then
    # The training stage owns its own concurrency (exactly two replicates
    # alive, chained individually) and takes the exclusive lock for its whole
    # duration.  Nothing of ours may overlap it.
    if [ "$DRY" = 1 ]; then
      chain_say "  DRY    S2 -> confirmatory/stage_train.sh $PROFILE"
    else
      if ! chain_lock_acquire exclusive "S2"; then rc_all=1; break; fi
      CHAIN_PROFILE="$PROFILE" \
      TRAIN_PREFLIGHT_RETRIES="$PREFLIGHT_RETRIES" \
      TRAIN_PREFLIGHT_WAIT="$PREFLIGHT_WAIT" \
      bash confirmatory/stage_train.sh "$PROFILE"
      trc=$?
      chain_lock_release
      [ "$trc" = 0 ] || { chain_say "STOP  S2 training stage failed"; rc_all=1; break; }
    fi
  else
    for ID in $($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --ids "$ST"); do
      if ! run_invocation "$ID"; then rc_all=1; break 2; fi
    done
  fi
  chain_ledger event=stage_end "stage=$ST"
done

chain_say "CHAIN END profile=$PROFILE rc=$rc_all"
chain_ledger event=chain_end "rc=$rc_all"
exit $rc_all
