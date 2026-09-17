#!/bin/bash
# Untrained-actor control (confirmatory/UNTRAINED_CONTROL.md): the unchanged
# frozen executor running each replicate's untrained shared actor on the
# confirmatory seeds 80-99, then the descriptive analysis.
set -u
cd ~/MARL_run_7 || exit 1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MARL_STEER_MODEL=actionable
ENGINE="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1"
POLICY="MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01"
L=confirmatory/untrained_chain.log
stamp() { date -u '+%F %T UTC'; }
echo "=== $(stamp)  UNTRAINED CONTROL START ===" >> "$L"
sha256sum -c --quiet confirmatory/UNTRAINED_CHECKPOINTS.sha256 >> "$L" 2>&1 || { echo "=== checkpoint hashes do not match -- refusing ===" >> "$L"; exit 1; }
echo "7ec6ecde30efb75d56b6fcbf8480dc91  run_timeline_comparison.py" | md5sum -c --quiet >> "$L" 2>&1 || { echo "=== harness changed -- refusing ===" >> "$L"; exit 1; }
run_one () {  # $1 torch seed
  local s="$1" out="untrained_s$1_80_99.pkl"
  env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$PWD/untrained_s$s/policy_init.pt" nice -n 5 python3 run_timeline_comparison.py \
      --seed-list 80-99 --arms marl_static --no-figures --out "$out" > "logs/${out%.pkl}.log" 2>&1
  local rc=$?
  echo "    $(stamp)  $out exit=$rc" >> "$L"
  if [ $rc -ne 0 ] || [ ! -s "output/$out" ]; then
    echo "    technical failure on $out: one unchanged re-run" >> "$L"
    env $ENGINE $POLICY MARL_EVAL_CHECKPOINT="$PWD/untrained_s$s/policy_init.pt" nice -n 5 python3 run_timeline_comparison.py \
        --seed-list 80-99 --arms marl_static --no-figures --out "$out" > "logs/${out%.pkl}.rerun.log" 2>&1
    echo "    $(stamp)  $out re-run exit=$?" >> "$L"
  fi
  sha256sum "output/$out" >> "$L" 2>&1
  python3 untrained_analysis.py --seeds "$DONE$s" --out "confirmatory/untrained_report_after_s$s.json" \
      > "confirmatory/untrained_report_after_s$s.txt" 2>&1
  echo "    $(stamp)  analysis after s$s exit=$?" >> "$L"
  DONE="$DONE$s,"
}
DONE=""
for s in 1234 2345 3456 4567; do run_one "$s"; done
python3 untrained_analysis.py --seeds 1234,2345,3456,4567 --out confirmatory/untrained_report.json \
    > confirmatory/untrained_report.txt 2>&1
echo "=== $(stamp)  UNTRAINED CONTROL COMPLETE (analysis exit=$?) ===" >> "$L"
