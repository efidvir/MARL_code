#!/bin/bash
# CONFIRMATORY TEST -- seeds 80-99, evaluated once under the frozen configuration
# declared in confirmatory/PREREGISTRATION.md.  Refuses to run if any frozen
# file has changed since the freeze.
set -u
cd ~/MARL_run_5 || exit 1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MARL_STEER_MODEL=actionable
ENGINE="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1"
POLICY="MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01"
MCK="$PWD/ce_ck_s1234/policy_final.pt"
ARMS="xldet,sdn,ospf,olsr,batman,aodv,random,marl_freeze"
LOG=confirmatory/chain.log

echo "=== $(date -u '+%F %T UTC')  CONFIRMATORY CHAIN START ===" >> "$LOG"
if ! sha256sum -c --quiet confirmatory/FREEZE.sha256 >> "$LOG" 2>&1; then
  echo "=== FREEZE VIOLATED -- refusing to run ===" >> "$LOG"; exit 1
fi
echo "    freeze verified ($(wc -l < confirmatory/FREEZE.sha256) files)" >> "$LOG"
for v in MARL_PRB_POLICY MARL_AGENT_FLEET MARL_POSTCARD_DRAIN MARL_UE_RELAY MARL_OBS_LOCALITY MARL_DETECT_TICKS MARL_STRESS_INTERFERENCE_DBM; do
  if [ -n "${!v:-}" ]; then echo "REFUSING: $v is set" >> "$LOG"; exit 1; fi
done

run_eval () {
  env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$MCK" python3 run_timeline_comparison.py \
    --seed-list 80-99 --arms "$ARMS" --no-figures --out cf_eval_80_99.pkl \
    > logs/cf_eval_80_99.log 2>&1
}
echo "=== $(date -u '+%F %T UTC')  evaluation start (seeds 80-99, all arms, both scenarios) ===" >> "$LOG"
run_eval; rc=$?
echo "=== $(date -u '+%F %T UTC')  evaluation exit=$rc ===" >> "$LOG"
if [ $rc -ne 0 ] || [ ! -s output/cf_eval_80_99.pkl ]; then
  echo "=== technical failure: one unchanged re-run, as pre-registered ===" >> "$LOG"
  run_eval; rc=$?
  echo "=== $(date -u '+%F %T UTC')  re-run exit=$rc ===" >> "$LOG"
fi
sha256sum output/cf_eval_80_99.pkl >> "$LOG"

echo "=== $(date -u '+%F %T UTC')  pre-registered analysis ===" >> "$LOG"
python3 confirmatory_analysis.py --pickles output/cf_eval_80_99.pkl \
  --manifests confirmatory/benchmark_manifest_80_99.json \
  --out confirmatory/cf_report.json > confirmatory/cf_report.txt 2>&1
echo "=== $(date -u '+%F %T UTC')  analysis exit=$? ===" >> "$LOG"
echo "=== $(date -u '+%F %T UTC')  CONFIRMATORY CHAIN COMPLETE ===" >> "$LOG"
