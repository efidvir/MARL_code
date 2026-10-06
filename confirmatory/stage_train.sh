#!/bin/bash
# S2: the TRAINING stage.  Called once, by confirmatory/run_final.sh, which
# already holds the EXCLUSIVE chain lock for the whole stage.
#
#   bash confirmatory/stage_train.sh PROFILE
#
# WHY THIS STAGE IS DIFFERENT
#  * Nothing of ours may overlap it.  train_on_comparison.py extends the
#    shared rollout pool in as_completed order (DECISIONS.md D0.8), so
#    minibatch composition -- and therefore the trained weights -- depend on
#    worker completion timing.  Contention perturbs the checkpoints.  The
#    exclusive lock is held by the caller; this script additionally refuses
#    to launch a replicate while any job of ours is alive (preflight).
#  * Replicates are chained INDIVIDUALLY: the next one starts the moment one
#    exits, keeping exactly two alive, instead of gang-waiting per wave.
#  * Every replicate gets --eval-episodes 0 --baseline-episodes 0: 26.8 min
#    per replicate of strictly serial time after the final checkpoint save,
#    and nothing reads the summary those phases populate.
#  * MARL_POOL_WORKERS stays at the evaluated 16.  It is configuration, not a
#    tuning knob, and it is recorded as such.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/chain_lib.sh"
PROFILE="${1:-}"
[ -n "$PROFILE" ] || { echo "usage: $0 PROFILE" >&2; exit 2; }
cd "$CHAIN_TREE" || exit 1
chain_setup "$PROFILE"

# How long the stage will WAIT for the host before giving up on a wave.  The
# caller (run_final.sh) passes its own values; the defaults match it.
TRAIN_PREFLIGHT_RETRIES=${TRAIN_PREFLIGHT_RETRIES:-6}
TRAIN_PREFLIGHT_WAIT=${TRAIN_PREFLIGHT_WAIT:-300}
CONCURRENT=$("$PY" -c "import sys;sys.path.insert(0,'$HERE');import chain_plan;print(chain_plan.PROFILES['$PROFILE']['train']['concurrent'])")
PRE=$("$PY" "$CHAIN_PLAN_PY" --profile "$PROFILE" --profile-field out_prefix)
IDS=$("$PY" "$CHAIN_PLAN_PY" --profile "$PROFILE" --ids S2)
chain_say "S2 TRAINING: replicates [$IDS], at most $CONCURRENT alive at a time (EXCLUSIVE)"

declare -A PIDFILE_OF ARTIFACT_OF START_OF NAME_OF T0_OF
LAUNCHED=""; FAILED=0; ALIVE=0
TRAIN_WORKERS=$("$PY" -c "import sys;sys.path.insert(0,'$HERE');import chain_plan;print(chain_plan.PROFILES['$PROFILE']['train']['workers'])")

# NOT a subshell: reap() has to be able to update FAILED in this shell.
poll () {
  local id pid
  ALIVE=0
  for id in $LAUNCHED; do
    [ -f "$CHAIN_JOBDIR/$id.done" ] && continue
    pid=$(cat "${PIDFILE_OF[$id]}" 2>/dev/null || echo '')
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      ALIVE=$((ALIVE + 1))
    else
      reap "$id"
    fi
  done
}

reap () {                 # $1 invocation id
  local id="$1" rc sha wall now
  [ -f "$CHAIN_JOBDIR/$id.done" ] && return 0
  touch "$CHAIN_JOBDIR/$id.done"
  rc=$(cat "$CHAIN_JOBDIR/$id.rc" 2>/dev/null || echo 98)
  now=$(date +%s); wall=$((now - ${START_OF[$id]}))
  sha=$(chain_sha "$CHAIN_TREE/${ARTIFACT_OF[$id]}")
  local st=ok
  # The checkpoint must have been written by THIS replicate.  launch() removed
  # it before starting, so an existing file with an older mtime would be a
  # previous attempt's policy being credited to a run that wrote nothing.
  if ! chain_artifact_is_fresh "${ARTIFACT_OF[$id]}" "${T0_OF[$id]}"; then
    chain_say "  GATE $id: $CHAIN_FRESH_WHY"
    st=fail
  fi
  { [ "$rc" != 0 ] || [ "$sha" = '-' ]; } && st=fail
  # the checkpoint records the environment it was trained in: the plan's
  if [ "$st" = ok ]; then
    "$PY" "$CHAIN_REPORT_PY" --check-marl-env "$CHAIN_TREE/${ARTIFACT_OF[$id]}" \
        --expect-marl-env-file "$CHAIN_ENVDIR/$id.marl_env" | tee -a "$CHAIN_LOG"
    [ "${PIPESTATUS[0]}" = 0 ] || st=fail
  fi
  local restarts
  restarts=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('restarts',{}).get(sys.argv[2],0))" "$CHAIN_WATCH_STATE" "${NAME_OF[$id]}" 2>/dev/null || echo 0)
  rm -f "$HOME/MARL_watch/chains/${NAME_OF[$id]}.json"
  chain_ledger event=end "inv=$id" stage=S2 kind=train attempt=1 "status=$st" \
      "exit=$rc" "wall_s=$wall" "sha256=$sha" "artifact=${ARTIFACT_OF[$id]}" \
      "passes=0" "workers=$TRAIN_WORKERS" "registered=1" "watchdog_restarts=$restarts"
  chain_say "  ${st} $id exit=$rc wall=${wall}s policy=${sha:0:16}"
  [ "$st" = ok ] || FAILED=$((FAILED + 1))
}

launch () {               # $1 invocation id
  local id="$1" status sha attempts
  eval "$($PY "$CHAIN_PLAN_PY" --profile "$PROFILE" --emit "$id")"

  read -r status sha attempts <<< "$("$PY" "$CHAIN_REPORT_PY" --ledger-status "$CHAIN_LEDGER" --inv "$id")"
  if [ "$status" = SUCCESS ]; then
    local now; now=$(chain_sha "$CHAIN_TREE/$INV_ARTIFACT")
    if [ "$now" = "$sha" ] && [ "$now" != '-' ]; then
      chain_say "  RESUME $id already trained (policy sha ${sha:0:16}) - not re-run"
      chain_ledger event=skip "inv=$id" stage=S2 status=ok "sha256=$sha" \
                   "artifact=$INV_ARTIFACT" reason=resume
      touch "$CHAIN_JOBDIR/$id.done"
      return 0
    fi
    chain_say "  STOP $id recorded successful but its policy file changed - escalate"
    chain_ledger event=stop "inv=$id" stage=S2 reason=artifact_changed
    FAILED=$((FAILED + 1)); return 1
  fi
  if [ "$attempts" -ge 2 ]; then
    chain_say "  STOP $id already failed twice; a replicate restarts from episode 0"
    chain_say "       with the same --torch-seed, and that budget is used."
    chain_ledger event=stop "inv=$id" stage=S2 reason=restart_budget_used
    FAILED=$((FAILED + 1)); return 1
  fi

  # The replicates this stage has deliberately left running.  Without this
  # the second replicate of a wave sees the first as a "leftover job of ours"
  # and the whole training stage stops after one replicate -- which is exactly
  # what the first rehearsal attempt did at T+0.4 h.
  local alive_ids=""
  for a_id in $LAUNCHED; do
    [ -f "$CHAIN_JOBDIR/$a_id.done" ] && continue
    alive_ids="${alive_ids:+$alive_ids,}$a_id"
  done
  # DELAY THE WAVE, do not abandon it.  COMPUTE_PLAN.md failure handling B:
  # "If tenant load is already above ~20 cores when a training wave is due to
  # start: DELAY the wave rather than run it.  A perturbed, slow training block
  # is worse than a late one."  The same applies to the memory gate: the
  # rehearsal stopped the whole training stage because MemAvailable was
  # 25.93 GB against a 26 GB wave gate -- 70 MB short, for a moment -- and the
  # stage had no retry at all, only run_final.sh's evaluation path did.
  # An ABORT (exit 2, below the hard floor) is different and is not retried.
  local tries=0 rc
  until chain_preflight train "$id" S2 "$INV_WORKERS" "$alive_ids"; do
    rc=$?
    if [ "$rc" -eq 2 ]; then
      # A dip below the hard floor is TRANSIENT while one of this chain's own
      # replicates is still finishing: its resident memory is about to be
      # reclaimed.  The full-scale dry run of 2026-09-25 died exactly here.
      # Replicate 3 was launched the moment replicate 2 exited, while
      # replicate 1 still held ~10 GB, so the wave figure read 19.9 GB against
      # the 22 GB floor and the ENTIRE training stage aborted -- at hour 7 of
      # the chain, with 2 of 4 replicates trained.  Replicate 1 exited two
      # minutes later.  The floor itself is right (there is no swap on this
      # host, so an overshoot is an OOM kill), but "not retrying" turned a
      # two-minute wait into a dead training block.
      #
      # So an ABORT is now waited out on the SAME bounded budget as a plain
      # refusal, and only becomes fatal once that budget is spent.  The floor
      # still prevents launching into memory that is genuinely absent: if it is
      # still breached after every retry, the stage fails as before.
      if [ "$tries" -lt "$TRAIN_PREFLIGHT_RETRIES" ]; then
        tries=$((tries + 1))
        chain_say "  DELAY $id: below the hard memory floor; this is transient while"
        chain_say "        our own replicates finish, so waiting ${TRAIN_PREFLIGHT_WAIT}s ($tries/$TRAIN_PREFLIGHT_RETRIES)"
        chain_ledger event=note "inv=$id" stage=S2 reason=wave_abort_transient "try=$tries"
        sleep "$TRAIN_PREFLIGHT_WAIT"
        continue
      fi
      chain_say "  ABORT $id: still below the hard training memory floor after"
      chain_say "        $tries wait(s) over $((tries * TRAIN_PREFLIGHT_WAIT))s; the memory is genuinely absent"
      chain_ledger event=stop "inv=$id" stage=S2 reason=preflight_abort "tries=$tries"
      FAILED=$((FAILED + 1)); return 1
    fi
    tries=$((tries + 1))
    if [ "$tries" -ge "$TRAIN_PREFLIGHT_RETRIES" ]; then
      chain_say "  STOP $id: training preflight refused $tries time(s) over "
      chain_say "       $((tries * TRAIN_PREFLIGHT_WAIT))s (>= 26 GB MemAvailable for the wave,"
      chain_say "       abort < 22 GB, load below the profile's ceiling).  The wave is"
      chain_say "       DELAYED, not failed: re-run the chain when the host is quieter;"
      chain_say "       the replicates that succeeded are not re-run."
      chain_ledger event=stop "inv=$id" stage=S2 reason=preflight_failed "tries=$tries"
      FAILED=$((FAILED + 1)); return 1
    fi
    chain_say "  DELAY $id: training precondition not met; waiting ${TRAIN_PREFLIGHT_WAIT}s ($tries/$TRAIN_PREFLIGHT_RETRIES)"
    chain_ledger event=note "inv=$id" stage=S2 reason=wave_delayed "try=$tries"
    sleep "$TRAIN_PREFLIGHT_WAIT"
  done
  TRAIN_WORKERS="$CHAIN_WORKERS"

  local ckdir="output/${PRE}ck_s${INV_TORCH_SEED}" outdir="output/${PRE}out_s${INV_TORCH_SEED}"
  mkdir -p "$ckdir" "$outdir"
  chain_write_env "$id" "$INV_ENV" "" "$TRAIN_WORKERS"
  local cmd="nice -n 19 $PY train_on_comparison.py $INV_TRAIN_ARGS --output-dir $outdir --checkpoint-dir $ckdir --save-policy-path $ckdir/policy_final.pt"
  chain_write_job "$id" "$INV_MARKER" "$cmd"
  # the replicate's environment is EXACTLY the plan's (FIX-PREFREEZE item 5)
  if ! chain_env_check "$id" 0; then
    chain_say "  STOP $id: the job's environment is not the plan's (see [PREFLIGHT-ENV] above)"
    chain_ledger event=stop "inv=$id" stage=S2 reason=env_mismatch
    FAILED=$((FAILED + 1)); return 1
  fi
  chain_clear_artifact "$INV_ARTIFACT"
  T0_OF[$id]=$CHAIN_T0

  chain_say "  LAUNCH $id (torch-seed $INV_TORCH_SEED, workers $TRAIN_WORKERS, expected ${INV_HOURS} h)"
  chain_ledger event=start "inv=$id" stage=S2 kind=train attempt=1 \
      "workers=$TRAIN_WORKERS" "artifact=$INV_ARTIFACT" "cmd=$cmd" \
      "envfile=${CHAIN_ENVFILE#$CHAIN_TREE/}" "job=${CHAIN_JOB#$CHAIN_TREE/}"

  local name="rc_${PROFILE}_${id}" stall reg_hours
  stall=$("$PY" -c "print(max(30, int(float('$INV_HOURS')*60*0.5)))")
  reg_hours=$("$PY" -c "print(max(0.5, float('$INV_HOURS')))")
  rm -f "$CHAIN_JOBDIR/$id.rc" "$CHAIN_JOBDIR/$id.done"
  "$CHAIN_REGISTER" "$name" "$CHAIN_TREE" "bash $CHAIN_JOB" "$CHAIN_JOBLOG" \
      "$INV_MARKER" "$reg_hours" "${CHAIN_ENVFILE#$CHAIN_TREE/}" "$stall" >> "$CHAIN_LOG" 2>&1
  PIDFILE_OF[$id]="$CHAIN_TREE/${CHAIN_JOBLOG%.log}.pid"
  ARTIFACT_OF[$id]="$INV_ARTIFACT"
  START_OF[$id]=$(date +%s)
  NAME_OF[$id]="$name"
  LAUNCHED="$LAUNCHED $id"
  sleep 5
}

for ID in $IDS; do
  poll
  while [ "$ALIVE" -ge "$CONCURRENT" ]; do sleep 20; poll; done
  launch "$ID" || true
  [ "$FAILED" -gt 0 ] && break
done

poll
while [ "$ALIVE" -gt 0 ]; do sleep 20; poll; done
for ID in $LAUNCHED; do reap "$ID"; done

if [ "$FAILED" -gt 0 ]; then
  chain_say "S2 TRAINING: $FAILED replicate(s) failed."
  chain_say "  A replicate that dies mid-run cannot be resumed into the same draw (the"
  chain_say "  rollout pool lives in the parent's memory).  Re-run the chain: the failed"
  chain_say "  replicate restarts from episode 0 with the same --torch-seed and is"
  chain_say "  recorded as a restart; the replicates that succeeded are NOT re-run."
  exit 1
fi
chain_say "S2 TRAINING COMPLETE: $(echo $LAUNCHED | wc -w) replicate(s)"
exit 0
