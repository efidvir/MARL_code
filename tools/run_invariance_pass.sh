#!/bin/bash
# Canonical invariance-matrix pass (docs/RERUN_SPEC.md section 14; WP0).
#
#   tools/run_invariance_pass.sh OUT.pkl [TREE_DIR]
#
# Runs the eight arms marl_static, xldet, random, sdn, ospf, olsr, batman,
# aodv on seeds 82 and 91, both scenarios, with --ticks 600, in the
# evaluation environment of confirmatory/run_amendment1.sh (ENGINE + POLICY
# variables, MARL_STEER_MODEL=actionable), with PYTHONHASHSEED=0, nice -n 19
# and MARL_POOL_WORKERS (default 8).  Every other MARL_* variable of the
# calling shell is removed.
#
# CHECKPOINT (trackAgent merge).  Only marl, marl_freeze and marl_static read
# one.  When one of them is selected, MARL_EVAL_CHECKPOINT is INV_CHECKPOINT
# if given, else a checkpoint MINTED from this tree with
# sixg_sim.agent.mint_schema_current_checkpoint(torch_seed=1234) into
# output/<OUT>.minted_s1234.pt: WP6's strict schema gate refuses the tracked
# pre-WP6 ce_ck_s1234/policy_final.pt, which every reference up to WP5 used,
# so a MARL-path arm is comparable across the merge only as "same tree rule,
# minted weights", never bit-for-bit.  With no such arm selected (routing,
# SDN, xldet, random) no checkpoint is set at all -- the harness neither needs
# nor opens one.
#
# NOTE the harness scales the event ticks with --ticks (D0.4/D0.5): at 600
# ticks Scenario A is severed at tick 100 (not the evaluated 200) and the
# Scenario B surge comes at tick 300.  The supplementary routing/SDN
# reference keeps the evaluated severance tick 200:
#   INV_TICKS=1200 INV_ARMS=sdn,ospf,olsr,batman,aodv \
#       tools/run_invariance_pass.sh inv_<wp>_<head>_t1200.pkl
#
# Writes TREE_DIR/output/OUT.pkl, TREE_DIR/logs/OUT.log (harness output and
# /usr/bin/time -v) and TREE_DIR/logs/OUT.rss.tsv (process-tree memory).
# TREE_DIR defaults to the tree holding this script.  A tree without
# sixg_sim/runinfo.py (pre-WP0) has no pool cap of its own; it is run through
# a shim that caps multiprocessing.cpu_count() at MARL_POOL_WORKERS instead.
#
# Overrides (the main reference uses the defaults; the 1200-tick reference
# uses INV_TICKS and INV_ARMS as above; anything else is a smoke):
#   INV_SEEDS, INV_TICKS, INV_ARMS, INV_CHECKPOINT; INV_FORCE=1 overwrites
#   an existing OUT.
set -eu
[ $# -ge 1 ] || { sed -n '2,29p' "$0" >&2; exit 2; }
OUT="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(cd "${2:-$HERE/..}" && pwd)"
BASE="${OUT%.pkl}"

for v in $(compgen -e | grep '^MARL_' || true); do
  [ "$v" = MARL_POOL_WORKERS ] || unset "$v"
done
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MARL_STEER_MODEL=actionable
export MARL_POOL_WORKERS="${MARL_POOL_WORKERS:-8}"
ENGINE="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1"
# MARL_POSTCARD_FRAGMENTS=1 was part of POLICY until WP5 removed the fragment
# cue; the engine now refuses it (sixg_sim/agent.py).
POLICY="MARL_REWARD_PROFILE=mission MARL_RELAY_ENTROPY_FLOOR=0.01"
SEEDS="${INV_SEEDS:-82,91}"
TICKS="${INV_TICKS:-600}"
ARMS="${INV_ARMS:-marl_static,xldet,random,sdn,ospf,olsr,batman,aodv}"

cd "$TREE"
mkdir -p output logs
if [ -e "output/$OUT" ] && [ "${INV_FORCE:-0}" != 1 ]; then
  echo "refusing: $TREE/output/$OUT exists (INV_FORCE=1 to overwrite)" >&2
  exit 1
fi
CK=""
case ",$ARMS," in
  *,marl,*|*,marl_freeze,*|*,marl_static,*)
    CK="${INV_CHECKPOINT:-}"
    if [ -z "$CK" ]; then
      CK="$TREE/output/$BASE.minted_s1234.pt"
      python3 -c "import sys; sys.path.insert(0, '.'); from sixg_sim.agent import mint_schema_current_checkpoint as m; m(sys.argv[1], torch_seed=1234, extra={'driver': 'tools/run_invariance_pass.sh'})" "$CK" \
        || { echo "could not mint a checkpoint at $CK" >&2; exit 1; }
    fi
    [ -f "$CK" ] || { echo "missing checkpoint $CK" >&2; exit 1; }
    ;;
esac
CKENV=()
[ -n "$CK" ] && CKENV=(MARL_EVAL_CHECKPOINT="$CK")

SHIM='
import multiprocessing, os, runpy, sys
_n = int(os.environ.get("MARL_POOL_WORKERS") or 0)
if _n > 0:
    _orig = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: min(_orig(), _n)
sys.argv = ["run_timeline_comparison.py"] + sys.argv[1:]
sys.path[0] = os.path.abspath(".")
runpy.run_path("run_timeline_comparison.py", run_name="__main__")
'
if [ -f sixg_sim/runinfo.py ]; then
  CMD=(python3 run_timeline_comparison.py)
  MODE="native pool cap"
else
  CMD=(python3 -c "$SHIM")
  MODE="pre-WP0 tree: cpu_count shim"
fi
TIMEV=()
[ -x /usr/bin/time ] && TIMEV=(/usr/bin/time -v)

{
  echo "# run_invariance_pass: tree=$TREE head=$(cat .synced_head 2>/dev/null || echo '?') ($MODE)"
  echo "# seeds=$SEEDS ticks=$TICKS arms=$ARMS MARL_POOL_WORKERS=$MARL_POOL_WORKERS"
  echo "# env: $ENGINE $POLICY MARL_STEER_MODEL=$MARL_STEER_MODEL PYTHONHASHSEED=$PYTHONHASHSEED MARL_EVAL_CHECKPOINT=${CK:-<unset: no selected arm reads one>}"
  echo "# start $(date -u '+%F %T UTC')"
} > "logs/$BASE.log"

env $ENGINE $POLICY "${CKENV[@]}" "${TIMEV[@]}" nice -n 19 "${CMD[@]}" \
    --seed-list "$SEEDS" --ticks "$TICKS" --arms "$ARMS" --no-figures --out "$OUT" \
    >> "logs/$BASE.log" 2>&1 < /dev/null &
PID=$!
python3 "$HERE/rss_monitor.py" "$PID" "logs/$BASE.rss.tsv" --interval 5 > "logs/$BASE.rss.txt" 2>&1 &
MON=$!
set +e
wait "$PID"
RC=$?
wait "$MON"
set -e
echo "# end $(date -u '+%F %T UTC') exit=$RC" >> "logs/$BASE.log"
cat "logs/$BASE.rss.txt" >> "logs/$BASE.log"
if [ $RC -eq 0 ] && [ -s "output/$OUT" ]; then
  sha256sum "output/$OUT" | tee -a "logs/$BASE.log"
fi
exit $RC
