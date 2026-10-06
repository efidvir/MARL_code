#!/bin/bash
# WP12 diagnostics launcher (docs/RERUN_SPEC.md WP12; docs/DECISIONS.md D12.x).
#
#   diag/run_wp12.sh ue         [--seeds 82,91 --ticks 1200 --arms marl_static,ospf --out wp12_ue_transit.pkl]
#   diag/run_wp12.sh sdn-static [--seeds 80-99 --scenarios recovery,rescue_ops]
#   diag/run_wp12.sh sdn-run    [--seeds 87,98 --ticks 600 --out wp12_sdn_probe.pkl]
#   diag/run_wp12.sh ue-report  output/PKL      diag/run_wp12.sh sdn-report output/PKL
#
# Environment: the evaluation environment of confirmatory/run_amendment1.sh,
# as tools/run_invariance_pass.sh sets it (ENGINE + POLICY variables,
# MARL_STEER_MODEL=actionable, PYTHONHASHSEED=0, nice -n 19,
# MARL_POOL_WORKERS default 8).  Since the trackAgent merge: POLICY no longer
# sets MARL_POSTCARD_FRAGMENTS=1 (WP5 removed the cue; the engine refuses the
# switch), and the checkpoint is WP12_CHECKPOINT or, unset, one minted from
# this tree (torch seed 1234) into output/wp12/minted_s1234.pt -- WP6's strict
# gate refuses the pre-WP6 ce_ck_s1234/policy_final.pt the pre-freeze WP12
# runs used, so a re-run here is a new measurement, not a repeat.  (Every other
# MARL_* variable of the calling shell is removed.)  The probes only
# monkeypatch; no engine or harness file is edited.  Output goes to
# output/<out>.pkl and output/wp12/, the log to logs/wp12_<what>.log.
# Start the long ones detached:  nohup diag/run_wp12.sh ue > /dev/null 2>&1 &
set -eu
[ $# -ge 1 ] || { sed -n '2,15p' "$0" >&2; exit 2; }
WHAT="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"

for v in $(compgen -e | grep '^MARL_' || true); do
  [ "$v" = MARL_POOL_WORKERS ] || unset "$v"
done
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MARL_STEER_MODEL=actionable
export MARL_POOL_WORKERS="${MARL_POOL_WORKERS:-8}"
ENGINE="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1"
POLICY="MARL_REWARD_PROFILE=mission MARL_RELAY_ENTROPY_FLOOR=0.01"

cd "$TREE"
mkdir -p output/wp12 logs
CK="${WP12_CHECKPOINT:-}"
if [ -z "$CK" ]; then
  CK="$TREE/output/wp12/minted_s1234.pt"
  python3 -c "import sys; sys.path.insert(0, '.'); from sixg_sim.agent import mint_schema_current_checkpoint as m; m(sys.argv[1], torch_seed=1234, extra={'driver': 'diag/run_wp12.sh'})" "$CK" \
    || { echo "could not mint a checkpoint at $CK" >&2; exit 1; }
fi
[ -f "$CK" ] || { echo "missing checkpoint $CK" >&2; exit 1; }
# WP12_LOG_TAG=dev keeps a second invocation of the same command (e.g.
# sdn-static on another seed set) from overwriting the first log.
LOG="logs/wp12_${WHAT}${WP12_LOG_TAG:+_$WP12_LOG_TAG}.log"
{
  echo "# run_wp12 $WHAT $*: tree=$TREE head=$(cat .synced_head 2>/dev/null || echo '?')"
  echo "# env: $ENGINE $POLICY MARL_STEER_MODEL=$MARL_STEER_MODEL PYTHONHASHSEED=$PYTHONHASHSEED MARL_POOL_WORKERS=$MARL_POOL_WORKERS MARL_EVAL_CHECKPOINT=$CK"
  echo "# start $(date -u '+%F %T UTC')  load: $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$LOG"

set +e
case "$WHAT" in
  ue)
    env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$CK" nice -n 19 \
      python3 diag/ue_transit_probe.py run "$@" >> "$LOG" 2>&1 < /dev/null ;;
  ue-report)
    env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$CK" nice -n 19 \
      python3 diag/ue_transit_probe.py report "$@" --out-dir output/wp12 >> "$LOG" 2>&1 < /dev/null ;;
  sdn-static)
    env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$CK" nice -n 19 \
      python3 diag/sdn_discovery_probe.py static "$@" >> "$LOG" 2>&1 < /dev/null ;;
  sdn-run)
    env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$CK" nice -n 19 \
      python3 diag/sdn_discovery_probe.py run "$@" >> "$LOG" 2>&1 < /dev/null ;;
  sdn-report)
    env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$CK" nice -n 19 \
      python3 diag/sdn_discovery_probe.py report "$@" --out-dir output/wp12 >> "$LOG" 2>&1 < /dev/null ;;
  *)
    echo "unknown command $WHAT" >&2; exit 2 ;;
esac
RC=$?
set -e
echo "# end $(date -u '+%F %T UTC') exit=$RC" >> "$LOG"
tail -n 3 "$LOG"
exit $RC
