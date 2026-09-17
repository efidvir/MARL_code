#!/bin/bash
# Stale relay-edge diagnostic (confirmatory/DIAG_STALE_EDGES.md):
#   R-fix : MARL_FIX_PHANTOM=1 MARL_DIAG_PHANTOM=1, marl_static/xldet/random, seeds 80-99
#   R-diag: MARL_DIAG_PHANTOM=1,                   same arms and seeds
# then the post-hoc analyses.
set -u
cd ~/MARL_run_8 || exit 1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MARL_STEER_MODEL=actionable
ENGINE="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1"
POLICY="MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01"
CK="$HOME/MARL_run_6/ce_ck_s1234/policy_final.pt"
ARMS="marl_static,xldet,random"
L=confirmatory/stale_chain.log
stamp() { date -u '+%F %T UTC'; }
echo "=== $(stamp)  STALE-EDGE DIAGNOSTIC START ===" >> "$L"
sha256sum -c --quiet confirmatory/STALE_FREEZE.sha256 >> "$L" 2>&1 || { echo "=== freeze mismatch -- refusing ===" >> "$L"; exit 1; }
echo "bbee30e4e9a4d26432753341d72e64661eea63c67139d290a2e831479f01e01f  $CK" | sha256sum -c --quiet >> "$L" 2>&1 || { echo "=== checkpoint mismatch -- refusing ===" >> "$L"; exit 1; }
run_batch () {  # $1 tag  $2 seeds  $3 extra env
  local out="$1_$(echo "$2" | tr - _).pkl"
  env $ENGINE $POLICY $3 MARL_EVAL_CHECKPOINT="$CK" python3 run_timeline_comparison.py \
      --seed-list "$2" --arms "$ARMS" --no-figures --out "$out" > "logs/${out%.pkl}.log" 2>&1
  local rc=$?
  echo "    $(stamp)  $out exit=$rc" >> "$L"
  if [ $rc -ne 0 ] || [ ! -s "output/$out" ]; then
    env $ENGINE $POLICY $3 MARL_EVAL_CHECKPOINT="$CK" python3 run_timeline_comparison.py \
        --seed-list "$2" --arms "$ARMS" --no-figures --out "$out" > "logs/${out%.pkl}.rerun.log" 2>&1
    echo "    $(stamp)  $out re-run exit=$?" >> "$L"
  fi
  sha256sum "output/$out" >> "$L" 2>&1
}
for s in 80-83 84-87 88-91 92-95 96-99; do run_batch fix "$s" "MARL_FIX_PHANTOM=1 MARL_DIAG_PHANTOM=1"; done
echo "=== $(stamp)  R-fix complete ===" >> "$L"
python3 stale_analysis.py fix > confirmatory/stale_fix_report.txt 2>&1
echo "    $(stamp)  fix analysis exit=$?" >> "$L"
for s in 80-83 84-87 88-91 92-95 96-99; do run_batch diag "$s" "MARL_DIAG_PHANTOM=1"; done
echo "=== $(stamp)  R-diag complete ===" >> "$L"
python3 stale_analysis.py diag > confirmatory/stale_diag_report.txt 2>&1
echo "=== $(stamp)  STALE-EDGE DIAGNOSTIC COMPLETE (diag analysis exit=$?) ===" >> "$L"
