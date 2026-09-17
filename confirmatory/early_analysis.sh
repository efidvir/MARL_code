#!/bin/bash
# Orchestration only: run the pre-registered Amendment-1 analysis commands as
# soon as their inputs exist, instead of after the whole chain.  Same scripts,
# same arguments, separate *_early output names; the frozen chain later writes
# the canonical outputs from the same inputs (which must be identical).
set -u
cd ~/MARL_run_6 || exit 1
export PYTHONHASHSEED=0
R5="$HOME/MARL_run_5"
A1=confirmatory/A1_chain.log
L=confirmatory/early.log
stamp() { date -u '+%F %T UTC'; }
echo "=== $(stamp)  early analysis waiter start ===" >> "$L"

# 1. confirmatory: needs the split static pickles AND the original job
until grep -q "cf_static_90_99.pkl" "$A1" 2>/dev/null && grep -q "CONFIRMATORY CHAIN COMPLETE" "$R5/confirmatory/chain.log" 2>/dev/null; do sleep 120; done
echo "=== $(stamp)  inputs ready: confirmatory analysis ===" >> "$L"
python3 confirmatory_analysis.py \
    --pickles "$R5/output/cf_eval_80_99.pkl,output/cf_static_80_89.pkl,output/cf_static_90_99.pkl" \
    --manifests benchmark_manifest_80_99.json --marl-arm marl_static \
    --out confirmatory/cf_report_A1_early.json > confirmatory/cf_report_A1_early.txt 2>&1
echo "    $(stamp)  confirmatory analysis exit=$?" >> "$L"
python3 A1_sensitivity.py > confirmatory/A1_sensitivity_early.txt 2>&1
echo "    $(stamp)  sensitivity exit=$?" >> "$L"
python3 extract_timelines.py \
    --pickles "$R5/output/cf_eval_80_99.pkl,output/cf_static_80_89.pkl,output/cf_static_90_99.pkl" \
    --out confirmatory/timelines_cf_early.json >> "$L" 2>&1
python3 dump_results_json.py --out confirmatory/ce_results_A1_cf.json \
    --tag "cf=output/cf_static_80_89.pkl,output/cf_static_90_99.pkl,$R5/output/cf_eval_80_99.pkl" >> "$L" 2>&1
echo "=== $(stamp)  CONFIRMATORY EARLY DONE ===" >> "$L"

# 2. development set: needs both static development pickles
until grep -q "dev_static_70_79.pkl exit=" "$A1" 2>/dev/null; do sleep 120; done
echo "=== $(stamp)  inputs ready: development analysis ===" >> "$L"
python3 confirmatory_analysis.py \
    --pickles "$R5/output/ce_eval_50_59.pkl,$R5/output/ce_eval_70_79.pkl,output/dev_static_50_59.pkl,output/dev_static_70_79.pkl" \
    --manifests benchmark_manifest.json,benchmark_manifest_70_79.json --marl-arm marl_static \
    --out confirmatory/dev_report_A1_early.json > confirmatory/dev_report_A1_early.txt 2>&1
echo "    $(stamp)  development analysis exit=$?" >> "$L"
python3 extract_timelines.py \
    --pickles "$R5/output/ce_eval_50_59.pkl,$R5/output/ce_eval_70_79.pkl,output/dev_static_50_59.pkl,output/dev_static_70_79.pkl" \
    --out confirmatory/timelines_dev_static_early.json >> "$L" 2>&1
echo "=== $(stamp)  DEVELOPMENT EARLY DONE ===" >> "$L"

# 3. stress + full KPI dump for the manuscript generator
until grep -q "stress_static_50_59.pkl exit=" "$A1" 2>/dev/null; do sleep 120; done
python3 dump_results_json.py --out confirmatory/ce_results_A1.json \
    --tag "dev=output/dev_static_50_59.pkl,output/dev_static_70_79.pkl,$R5/output/ce_eval_50_59.pkl,$R5/output/ce_eval_70_79.pkl" \
    --tag "cf=output/cf_static_80_89.pkl,output/cf_static_90_99.pkl,$R5/output/cf_eval_80_99.pkl" \
    --tag "stress=output/stress_static_50_59.pkl,$R5/output/ce_stress_m75_50_59.pkl" >> "$L" 2>&1
echo "=== $(stamp)  KPI DUMP DONE ===" >> "$L"
