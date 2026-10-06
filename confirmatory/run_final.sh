#!/bin/bash
# THE STAGED LAUNCHER for the final corrected rerun (WP RUN-CHAIN).
#
#   bash confirmatory/run_final.sh --profile rehearsal
#   bash confirmatory/run_final.sh --profile real --from S6
#   bash confirmatory/run_final.sh --profile real --shed sensitivity
#   bash confirmatory/run_final.sh --profile rehearsal --dry-run
#   bash confirmatory/run_final.sh --profile rehearsal --waive-dep S6
#
# WHAT IT GUARANTEES (COMPUTE_PLAN.md; see docs/RUN_CHAIN.md for the mapping)
#  * stages run in DEPENDENCY order, and a stage does not start until every
#    invocation of every stage it depends on is recorded successful;
#  * every stage is split into invocations of at most 64 passes along whole
#    (arm, seed-block) boundaries, each writing its own pickle whose sha256
#    goes in the chain ledger -- a crash costs at most one invocation;
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
#    (--shed), recorded in the ledger, and only in the plan's order.
#
# Exit status: 0 the requested stages all completed; 1 the chain stopped.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/chain_lib.sh"

PROFILE=""; ONLY_STAGES=""; FROM_STAGE=""; DRY=0; SHED=""; WAIVE=""; INVALIDATE=""
WAIVE_DEP=""
MAX_ATTEMPTS=2; PREFLIGHT_RETRIES=3; PREFLIGHT_WAIT=300

usage () { sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)      PROFILE="$2"; shift 2 ;;
    --stages)       ONLY_STAGES="$2"; shift 2 ;;
    --from)         FROM_STAGE="$2"; shift 2 ;;
    --shed)         SHED="$2"; shift 2 ;;
    --waive-order)  WAIVE="$2"; shift 2 ;;
    --waive-dep)    WAIVE_DEP="$2"; shift 2 ;;
    --invalidate)   INVALIDATE="$2"; shift 2 ;;
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
cd "$CHAIN_TREE" || exit 1
chain_setup "$PROFILE"
trap 'chain_lock_release' EXIT

# ---------------------------------------------------------------- shed order
# "Shed stages in this order -- S10/S11 sensitivity, S9 adapt-then-freeze,
# S8 untrained control."  A later group may not be shed while an earlier one
# is still scheduled: that is what makes the order a rule and not a comment.
SHED_GROUPS=" $(echo "$SHED" | tr ',' ' ') "
if [ -n "$SHED" ]; then
  "$PY" - "$HERE" $(echo "$SHED" | tr ',' ' ') <<'PYX' || exit 2
import sys
sys.path.insert(0, sys.argv[1])
import chain_plan
want, order = set(sys.argv[2:]), chain_plan.SHED_ORDER
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

# An explicit operator action, recorded: everything the ledger holds for these
# invocations stops counting, so they are redone on the next run and their
# restart budget starts again.  Nothing else is touched.  This exists because
# the alternative -- editing the ledger by hand -- leaves no record.
if [ -n "$INVALIDATE" ]; then
  for id in $(echo "$INVALIDATE" | tr ',' ' '); do
    chain_say "INVALIDATE $id (operator action; its previous result no longer counts)"
    chain_ledger event=invalidate "inv=$id" "reason=operator"
  done
  exit 0
fi

STAGES_ALL=$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --stages)
chain_say "=========================================================="
chain_say "CHAIN START profile=$PROFILE tree=$CHAIN_TREE dry_run=$DRY"
chain_say "stages: $STAGES_ALL"
[ -n "$SHED" ] && chain_say "OPERATOR SHED: $SHED (explicit, recorded; none of these gate the primary analysis)"
chain_ledger event=chain_start "stages=$STAGES_ALL" "shed=$SHED" "dry_run=$DRY" \
             "tree=$CHAIN_TREE" "argv=$*"

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
  if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
    chain_say "  STOP   $id has already failed $attempts time(s): the pre-registered"
    chain_say "         budget is one unchanged re-run.  Stopping the chain."
    chain_ledger event=stop "inv=$id" "stage=$INV_STAGE" "reason=restart_budget_used" \
                 "attempts=$attempts"
    return 1
  fi

  # --- preflight, gated
  case "$INV_KIND" in
    eval)  pfkind=eval ;;
    train) pfkind=train ;;
    *)     pfkind=tool ;;
  esac
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
    literal|replicate) ckpt="$INV_CKPT_VALUE" ;;
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
  if [ -n "$ckpt" ] && [ "$DRY" = 0 ] && [ ! -e "$CHAIN_TREE/$ckpt" ]; then
    chain_say "  STOP   $id: checkpoint $ckpt does not exist"
    chain_ledger event=stop "inv=$id" "reason=missing_checkpoint" "checkpoint=$ckpt"
    return 1
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
      $PY "$CHAIN_REPORT_PY" --verify "$CHAIN_TREE/$INV_ARTIFACT" \
          --keys-file "$CHAIN_JOBDIR/$id.keys" \
          --expect-hashseed "${INV_HASHSEED:-0}" \
          --expect-marl-env-file "$CHAIN_MARLENV" | tee -a "$CHAIN_LOG"
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
      if [ "$WAIVED" = 1 ]; then
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
