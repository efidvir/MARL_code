#!/bin/bash
# Shared machinery for the final-rerun chain (WP RUN-CHAIN).
# Sourced by confirmatory/run_final.sh and by the stage scripts.  Nothing in
# here runs on its own.
#
# Everything a stage does goes through these functions so that there is ONE
# ledger, ONE environment block, ONE lock discipline and ONE way of talking
# to the watchdog.

CHAIN_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAIN_PLAN_PY="$CHAIN_TREE/confirmatory/chain_plan.py"
CHAIN_REPORT_PY="$CHAIN_TREE/tools/stage_report.py"
CHAIN_PREFLIGHT_PY="$CHAIN_TREE/tools/preflight.py"
CHAIN_REGISTER="$HOME/MARL_watch/register_chain.sh"
CHAIN_WATCH_STATE="$HOME/MARL_watch/state.json"
PY=${PY:-python3}

chain_setup () {          # $1 profile
  CHAIN_PROFILE="$1"
  CHAIN_DIR="$CHAIN_TREE/confirmatory"
  CHAIN_LEDGER="$CHAIN_DIR/chain_${CHAIN_PROFILE}.jsonl"
  CHAIN_LOG="$CHAIN_TREE/logs/chain_${CHAIN_PROFILE}.log"
  CHAIN_PREDIR="$CHAIN_DIR/preflight_${CHAIN_PROFILE}"
  CHAIN_ENVDIR="$CHAIN_DIR/env_${CHAIN_PROFILE}"
  CHAIN_LOCK="$CHAIN_DIR/.lock_${CHAIN_PROFILE}"
  CHAIN_JOBDIR="$CHAIN_TREE/logs/chain_${CHAIN_PROFILE}"
  mkdir -p "$CHAIN_TREE/logs" "$CHAIN_PREDIR" "$CHAIN_ENVDIR" "$CHAIN_JOBDIR" \
           "$CHAIN_LOCK.shared"
  CHAIN_OUT_PREFIX="$($PY "$CHAIN_PLAN_PY" --profile "$CHAIN_PROFILE" --profile-field out_prefix)"
  CHAIN_WD_MIN_H="$($PY "$CHAIN_PLAN_PY" --profile "$CHAIN_PROFILE" --profile-field watchdog_min_hours)"
  CHAIN_TRAIN_LOAD_CEILING="$($PY "$CHAIN_PLAN_PY" --profile "$CHAIN_PROFILE" --profile-field train_load_ceiling)"
  mkdir -p "$CHAIN_TREE/output/${CHAIN_OUT_PREFIX%/}"
}

chain_stamp () { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

chain_say () {
  local line="[$(chain_stamp)] $*"
  echo "$line"
  echo "$line" >> "$CHAIN_LOG"
}

chain_ledger () {         # key=value ...
  "$PY" "$CHAIN_REPORT_PY" --ledger-append "$CHAIN_LEDGER" \
        --kv "profile=$CHAIN_PROFILE" "$@" || true
}

chain_sha () {            # $1 path
  if [ -f "$1" ]; then sha256sum "$1" | awk '{print $1}'; else echo '-'; fi
}

# ---------------------------------------------------------------- locking
# Training is EXCLUSIVE: the trained weights depend on rollout-worker
# completion order (DECISIONS.md D0.8), so nothing of ours may run beside it.
# This is enforced here, in code, not in a comment.
chain_lock_gc () {        # drop lock files whose owner is gone
  local f pid
  if [ -f "$CHAIN_LOCK.excl/pid" ]; then
    pid=$(cat "$CHAIN_LOCK.excl/pid" 2>/dev/null || echo '')
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      chain_say "lock: clearing stale EXCLUSIVE lock (owner pid ${pid:-none} gone)"
      rm -rf "$CHAIN_LOCK.excl"
    fi
  fi
  for f in "$CHAIN_LOCK.shared"/*; do
    [ -e "$f" ] || continue
    pid=$(basename "$f")
    kill -0 "$pid" 2>/dev/null || { rm -f "$f"; }
  done
}

chain_shared_holders () { ls -1 "$CHAIN_LOCK.shared" 2>/dev/null | wc -l; }

chain_lock_acquire () {   # $1 shared|exclusive   $2 inv id
  chain_lock_gc
  if [ "$1" = exclusive ]; then
    if [ "$(chain_shared_holders)" -gt 0 ]; then
      chain_say "REFUSING $2: training is exclusive and $(chain_shared_holders) evaluation lock(s) are held"
      return 1
    fi
    if ! mkdir "$CHAIN_LOCK.excl" 2>/dev/null; then
      chain_say "REFUSING $2: another exclusive holder is present"
      return 1
    fi
    echo $$ > "$CHAIN_LOCK.excl/pid"
    echo "$2" > "$CHAIN_LOCK.excl/inv"
    CHAIN_HELD=exclusive
  else
    if [ -d "$CHAIN_LOCK.excl" ]; then
      chain_say "REFUSING $2: a training block holds the exclusive lock ($(cat "$CHAIN_LOCK.excl/inv" 2>/dev/null))"
      return 1
    fi
    touch "$CHAIN_LOCK.shared/$$"
    CHAIN_HELD=shared
  fi
  return 0
}

chain_lock_release () {
  case "${CHAIN_HELD:-}" in
    exclusive) rm -rf "$CHAIN_LOCK.excl" ;;
    shared)    rm -f "$CHAIN_LOCK.shared/$$" ;;
  esac
  CHAIN_HELD=
}

# ---------------------------------------------------------------- preflight
chain_preflight () {      # $1 kind  $2 inv  $3 stage  $4 workers  [$5 expected-alive]
  local kind="$1" inv="$2" stage="$3" want="$4" expect="${5:-}" rc pin tsha
  local out="$CHAIN_PREDIR/${inv}.$(date -u +%H%M%S).json"
  # The tree digest recorded for the FIRST invocation of this run.  Nothing
  # else pins the tree before FREEZE_FINAL is sealed at S5, and S5 comes after
  # the 340 policy-independent passes, the training block, validation and
  # selection have all been spent.  Three other code tracks are landing work
  # packages in this tree while the chain runs.
  pin=$("$PY" "$CHAIN_REPORT_PY" --ledger-tree-pin "$CHAIN_LEDGER" 2>/dev/null || echo '-')
  [ -n "$pin" ] || pin='-'
  "$PY" "$CHAIN_PREFLIGHT_PY" --kind "$kind" --tree "$CHAIN_TREE" --inv "$inv" \
        --stage "$stage" --profile "$CHAIN_PROFILE" --requested-workers "$want" \
        --expect-inv "$expect" --freeze-file "$CHAIN_DIR/FREEZE_FINAL_${CHAIN_PROFILE}.txt" \
        --expect-tree "$pin" --train-load-ceiling "$CHAIN_TRAIN_LOAD_CEILING" \
        --json-out "$out" | tee -a "$CHAIN_LOG"
  rc=${PIPESTATUS[0]}
  CHAIN_PREFLIGHT_JSON="$out"
  CHAIN_WORKERS=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('recommended_workers') or 0)" "$out" 2>/dev/null || echo 0)
  [ "${CHAIN_WORKERS:-0}" -ge 1 ] 2>/dev/null || CHAIN_WORKERS="$want"
  tsha=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('tree_sha256') or '-')" "$out" 2>/dev/null || echo '-')
  chain_ledger event=preflight "inv=$inv" "stage=$stage" "kind=$kind" \
      "rc=$rc" "record=$out" "workers=$CHAIN_WORKERS" "tree_sha256=$tsha" \
      "tree_pin=$pin"
  return $rc
}

# ---------------------------------------------------------------- env block
# The environment file is sourced by register_chain.sh AND by the watchdog's
# restart path.  The watchdog restart does NOT set PYTHONHASHSEED itself
# (watchdog.py builds `cd CWD && source ENVFILE && nohup nice -n 19 CMD`), so
# an unchanged re-run is only really unchanged if the env file carries every
# scientifically load-bearing variable.  That is why it is written here in
# full, once, and reused byte-identically by any re-run.
#
# THE WHOLE ENVIRONMENT (FIX-PREFREEZE item 5).  The job script re-executes
# itself under `env -i` before sourcing this file (chain_write_job), so the
# lines below are ALL an invocation's processes see: the system variables a
# process needs (CHAIN_SYSTEM_VARS, recorded here with their values at write
# time), the pins, and the plan's MARL_* set.  A stray MARL_* -- or any other
# variable the code reads (UU_CA_MAX_EXTRA_CC, PROBE_TICK, ...) -- in the
# operator's shell, in register_chain.sh's or in the watchdog's cannot
# reach it.  <inv>.marl_env holds the MARL_* set the run must record
# (provenance marl_env): the preflight's job-environment check and the
# completeness gate compare against it.
CHAIN_SYSTEM_VARS="HOME PATH USER LOGNAME SHELL LANG LC_ALL TMPDIR PY"

# The value of MARL_EVAL_CHECKPOINT for a checkpoint argument given relative
# to the tree.  One path -> "$CHAIN_TREE/<path>", exactly what was written
# before this function existed.  A comma-separated list (R3-lite's ensemble
# invocation: the harness reads a list as an ENSEMBLE) -> every member made
# absolute, joined by commas, in the given order.
chain_ckpt_abs () {       # $1 checkpoint path, or a comma-separated list of them
  local out="" p
  local IFS=','
  for p in $1; do
    [ -n "$p" ] && out="${out:+$out,}$CHAIN_TREE/$p"
  done
  printf '%s\n' "$out"
}

chain_write_env () {      # $1 inv  $2 "K=V K=V"  $3 checkpoint(s)  $4 workers  [$5 hash seed, default 0]
  local inv="$1" envstr="$2" ckpt="$3" workers="$4" hashseed="${5:-0}"
  local f="$CHAIN_ENVDIR/${inv}.sh" kv v
  if [ -n "$ckpt" ]; then ckpt="$(chain_ckpt_abs "$ckpt")"; fi
  {
    echo "# generated by confirmatory/run_final.sh for invocation $inv"
    echo "# profile=$CHAIN_PROFILE tree=$CHAIN_TREE"
    echo "# the job starts from an EMPTY environment (env -i); this file is all of it"
    for v in $CHAIN_SYSTEM_VARS; do
      [ -n "${!v:-}" ] && printf 'export %s=%q\n' "$v" "${!v}"
    done
    # 0 for every invocation but the analysis-S8 repeat (S6h, hash seed 1),
    # whose plan entry says so; nothing else can set it.
    echo "export PYTHONHASHSEED=$hashseed"
    echo "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
    echo "export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
    echo "export PYTHONUNBUFFERED=1"
    [ -n "$workers" ] && echo "export MARL_POOL_WORKERS=$workers"
    for kv in $envstr; do echo "export $kv"; done
    [ -n "$ckpt" ] && echo "export MARL_EVAL_CHECKPOINT=$ckpt"
    echo "export MARL_CHAIN_INV=$inv"
  } > "$f"
  {
    [ -n "$workers" ] && echo "MARL_POOL_WORKERS=$workers"
    for kv in $envstr; do echo "$kv"; done
    [ -n "$ckpt" ] && echo "MARL_EVAL_CHECKPOINT=$ckpt"
    echo "MARL_CHAIN_INV=$inv"
  } | LC_ALL=C sort > "$CHAIN_ENVDIR/${inv}.marl_env"
  CHAIN_ENVFILE="$f"
  CHAIN_MARLENV="$CHAIN_ENVDIR/${inv}.marl_env"
}

# The environment the job script ACTUALLY gives its processes, checked
# before the launch against the plan's set (tools/preflight.py
# --job-env): the job script is run with --print-env FROM THIS SHELL, so a
# job that did not clear the caller's environment is caught here.
chain_env_check () {      # $1 inv  [$2 hash seed] -> 0 when the job's environment is the plan's
  local inv="$1" hs="${2:-0}" rc
  local out="$CHAIN_PREDIR/${inv}.env.$(date -u +%H%M%S).json"
  "$PY" "$CHAIN_PREFLIGHT_PY" --job-env "$CHAIN_JOB" --expect-marl-file "$CHAIN_MARLENV" \
        --expect-hashseed "$hs" --inv "$inv" --profile "$CHAIN_PROFILE" \
        --json-out "$out" | tee -a "$CHAIN_LOG"
  rc=${PIPESTATUS[0]}
  chain_ledger event=envcheck "inv=$inv" "rc=$rc" "record=$out"
  return $rc
}

# ---------------------------------------------------------------- job script
chain_write_job () {      # $1 inv  $2 marker  $3 command line
  local inv="$1" marker="$2" cmd="$3"
  local f="$CHAIN_JOBDIR/${inv}.sh"
  {
    echo '#!/bin/bash'
    echo "# invocation $inv ($CHAIN_PROFILE) - regenerated by run_final.sh"
    echo 'set -u'
    echo '# Start from an EMPTY environment: the env file below is all of it'
    echo '# (FIX-PREFREEZE item 5).  Whoever runs this -- the launcher, the'
    echo '# watchdog, register_chain.sh -- passes nothing else on.'
    echo 'if [ "${CHAIN_CLEAN_ENV:-}" != 1 ]; then exec /usr/bin/env -i CHAIN_CLEAN_ENV=1 /bin/bash "$0" "$@"; fi'
    echo 'unset CHAIN_CLEAN_ENV'
    echo "cd $CHAIN_TREE || exit 97"
    echo "source $CHAIN_ENVFILE"
    echo 'if [ "${1:-}" = --print-env ]; then exec /usr/bin/env -0; fi'
    echo "echo \"=== \$(date -u '+%Y-%m-%dT%H:%M:%SZ') START $inv ===\""
    echo "echo \"    argv: $cmd\""
    echo "$cmd"
    echo 'rc=$?'
    echo "echo \$rc > $CHAIN_JOBDIR/${inv}.rc"
    echo "echo \"=== \$(date -u '+%Y-%m-%dT%H:%M:%SZ') END $inv exit=\$rc ===\""
    echo "if [ \$rc -eq 0 ]; then echo '$marker'; fi"
    echo 'exit $rc'
  } > "$f"
  chmod +x "$f"
  CHAIN_JOB="$f"
  CHAIN_JOBLOG="logs/chain_${CHAIN_PROFILE}/${inv}.log"
  CHAIN_JOBRC="$CHAIN_JOBDIR/${inv}.rc"
}

# How old an artefact may be and still count as "written by this attempt".
# The tree is on an NFS mount, so an mtime set by the file server can round or
# skew by a second against the launcher's own clock.  Two seconds of grace
# costs nothing against the failure this guards: an artefact from a PREVIOUS
# attempt, minutes or hours old, credited to a job that wrote nothing.
CHAIN_MTIME_GRACE=${CHAIN_MTIME_GRACE:-2}

chain_clear_artifact () {   # $1 artefact path relative to the tree; sets CHAIN_T0
  # The completeness gate has to verify what THIS attempt produced.  Without
  # this, a job that exits 0 without writing is verified against the previous
  # attempt's pickle and recorded as a full success -- the campaign's recurring
  # defect, a check that reports success without having verified anything.
  [ -n "${1:-}" ] && rm -f "$CHAIN_TREE/$1"
  CHAIN_T0=$(date +%s)
}

chain_artifact_is_fresh () {   # $1 artefact path  $2 launch epoch
  local f m
  CHAIN_FRESH_WHY=''
  [ -n "${1:-}" ] || return 0            # no artefact declared: nothing to check
  f="$CHAIN_TREE/$1"
  [ -e "$f" ] || { CHAIN_FRESH_WHY="$1 does not exist"; return 1; }
  m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  if [ "$m" -lt $(( $2 - CHAIN_MTIME_GRACE )) ]; then
    CHAIN_FRESH_WHY="$1 was NOT written by this attempt (mtime $m < launch $2)"
    return 1
  fi
  CHAIN_FRESH_WHY="$1 written by this attempt (mtime $m >= launch $2)"
  return 0
}

# Launch a job.  Long invocations go through ~/MARL_watch/register_chain.sh so
# the existing watchdog monitors them and can restart one once; short ones run
# in the foreground through the identical job script and env file.
# usage: chain_launch inv hours marker  -> sets CHAIN_RC, CHAIN_WALL_S, CHAIN_T0
# The caller sets CHAIN_LAUNCH_ARTIFACT first; it is removed before the launch.
chain_launch () {
  local inv="$1" hours="$2" marker="$3"
  local name="rc_${CHAIN_PROFILE}_${inv}"
  local t0 t1 pid reg=0 stall
  rm -f "$CHAIN_JOBRC"
  chain_clear_artifact "${CHAIN_LAUNCH_ARTIFACT:-}"
  t0=$CHAIN_T0
  if [ "${CHAIN_NO_WATCHDOG:-0}" != 1 ] && [ -x "$CHAIN_REGISTER" ] && \
     "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$hours" "$CHAIN_WD_MIN_H"; then
    reg=1
    stall=$("$PY" -c "print(max(20, int(float('$hours')*60*0.75)))")
    # What we REGISTER is a tolerance, not the estimate: the watchdog reports a
    # problem past 1.5x expected_hours into a shared HEALTH report, and a
    # reduced-scale invocation modelled at 0.03 h would trip it in two minutes.
    local reg_hours
    reg_hours=$("$PY" -c "print(max(0.5, float('$hours')))")
    chain_say "  registering with the watchdog: $name (modelled ${hours} h, registered ${reg_hours} h, stall ${stall} min)"
    "$CHAIN_REGISTER" "$name" "$CHAIN_TREE" "bash $CHAIN_JOB" "$CHAIN_JOBLOG" \
        "$marker" "$reg_hours" "${CHAIN_ENVFILE#$CHAIN_TREE/}" "$stall" \
        >> "$CHAIN_LOG" 2>&1
    local pidfile="$CHAIN_TREE/${CHAIN_JOBLOG%.log}.pid"
    sleep 2
    while :; do
      pid=$(cat "$pidfile" 2>/dev/null || echo '')
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break
      sleep 10
    done
    # Deregister IMMEDIATELY: a registered entry whose process is gone and
    # whose log has no done marker is restarted by the watchdog within ten
    # minutes, which would collide with our own (single, pre-registered)
    # unchanged re-run.
    CHAIN_WD_RESTARTS=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('restarts',{}).get(sys.argv[2],0))" "$CHAIN_WATCH_STATE" "$name" 2>/dev/null || echo 0)
    rm -f "$HOME/MARL_watch/chains/$name.json"
  else
    bash "$CHAIN_JOB" >> "$CHAIN_TREE/$CHAIN_JOBLOG" 2>&1
    CHAIN_WD_RESTARTS=0
  fi
  t1=$(date +%s)
  CHAIN_WALL_S=$((t1 - t0))
  CHAIN_RC=$(cat "$CHAIN_JOBRC" 2>/dev/null || echo 98)
  CHAIN_REGISTERED=$reg
}
