#!/bin/bash
# WP9 validation of confirmatory_analysis.py on the frozen Amendment-1 data
# (RERUN_SPEC WP9; docs/WP9_ANALYSIS.md sections 7-8).  Run on <compute-server> from
# the tree root:  PYTHONHASHSEED=0 nice -n 19 bash tools/wp9_validate.sh
#
# 1. cf report from the frozen pickles vs ~/MARL_run_6/confirmatory/cf_report_A1.json
# 2. dev report vs dev_report_A1.json
# 3. the manuscript's M1 counts for marl_static, Scenario A, seeds 80-99
# 4. the hard-assert failures on the legacy pickles (expected: no provenance
#    record x3 and the aggregate postcard check x160; nothing else)
# 5. extract_timelines.py and dump_results_json.py regression against the
#    outputs of the previous validated script version (old leaves identical;
#    new keys are notes).  A MISSING reference is a FAILURE (rc=1), never a
#    silent skip: a fresh tree, a cleaned output directory or a different
#    WP9_OUT used to skip the comparison and still print "done rc=0", so the
#    export regression reported success having verified nothing (WP9 defect
#    list, D6).  Re-baseline with WP9_REBASELINE=1 (see below).
# 6. hash-invariance smoke on a repeat-run pair.  EXPECTED OUTCOME: exit 2,
#    "precondition not established" -- neither frozen pickle records a
#    PYTHONHASHSEED, so the pair cannot show hash invariance.  The step is a
#    coverage/plumbing smoke and its rc is deliberately not folded into rc.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${WP9_OUT:-output/wp9_validation}
mkdir -p "$OUT"
export PYTHONHASHSEED=0
R5=~/MARL_run_5/output
R6=~/MARL_run_6/output
REF=~/MARL_run_6/confirmatory
PY="nice -n 19 python3"
CF="$R5/cf_eval_80_99.pkl,$R6/cf_static_80_89.pkl,$R6/cf_static_90_99.pkl"
DEV="$R5/ce_eval_50_59.pkl,$R5/ce_eval_70_79.pkl,$R6/dev_static_50_59.pkl,$R6/dev_static_70_79.pkl"
rc=0
echo "== tree $(cat .synced_head 2>/dev/null || git rev-parse --short HEAD 2>/dev/null) $(date -u +%FT%TZ) =="

echo "== 1. cf report =="
$PY confirmatory_analysis.py --no-assert --marl-arm marl_static --pickles "$CF" \
    --manifests benchmark_manifest_80_99.json --out "$OUT/cf_report_wp9.json" > "$OUT/cf_report_wp9.txt" || rc=1
$PY tools/compare_reports.py "$REF/cf_report_A1.json" "$OUT/cf_report_wp9.json" | tee "$OUT/compare_cf.txt" || rc=1

echo "== 2. dev report =="
$PY confirmatory_analysis.py --no-assert --marl-arm marl_static --pickles "$DEV" \
    --manifests benchmark_manifest.json,benchmark_manifest_70_79.json --out "$OUT/dev_report_wp9.json" > "$OUT/dev_report_wp9.txt" || rc=1
$PY tools/compare_reports.py "$REF/dev_report_A1.json" "$OUT/dev_report_wp9.json" | tee "$OUT/compare_dev.txt" || rc=1

echo "== 3/4. M1 counts and legacy asserts =="
$PY - "$OUT" <<'EOF' || rc=1
import json, sys
from collections import Counter
out = sys.argv[1]
rep = json.load(open(out + '/cf_report_wp9.json'))
ps = rep['S10']['A']['marl_static']['per_seed']
got = [ps[str(s)]['M1'] for s in range(80, 100)]
exp = [7, 4, 1043, 4, 3, 1, 5, 3, 2, 39, 5, 895, 3, 2, 3, 923, 1, 4, 3, 15]
print('M1 marl_static A 80-99:', got)
print('manuscript           :', exp)
print('M1', 'MATCH' if got == exp else 'MISMATCH')
a = rep['asserts']
kinds = Counter('no provenance record' if 'no provenance record' in f
                else 'postcards (aggregate)' if 'postcards after the event' in f
                else 'other: ' + f for f in a['failures'])
print('hard asserts (--no-assert):', a['n_failures'], 'failures', dict(kinds))
c = a['checks']
print('nan violations', len(c['nan']['violations']), '| non-numeric', len(c['nan']['non_numeric']),
      '| exempt', c['nan']['exempt_counts'])
print('stale-edge keys', c['stale_edges']['runs_with_key'], '| run-length violations', len(c['run_length']['violations']))
print('postcards mode', c['postcards']['mode'], '| runs checked', c['postcards']['runs_checked'],
      '| min mean per site', c['postcards']['min_mean_per_site'])
print('provenance missing_fields', c['provenance']['missing_fields'], '| checkpoint', c['provenance']['checkpoint_sha256'])
m11 = rep['S10']['A']['marl_static']['M11']
print('M11 marl_static A:', {k: m11[k] for k in ('status', 'rule', 'n_bridges', 'n_censored') if k in m11})
ok = got == exp and a['n_failures'] == 163 and set(kinds) == {'no provenance record', 'postcards (aggregate)'}
print('STEP 3/4', 'OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
EOF

echo "== 5. timelines / dump regression =="
$PY extract_timelines.py --pickles "$CF" --out "$OUT/timelines_cf_wp9_new.json" || rc=1
if [ -f "$OUT/timelines_cf_wp9.json" ]; then
  $PY tools/compare_reports.py "$OUT/timelines_cf_wp9.json" "$OUT/timelines_cf_wp9_new.json" \
      --sections bin_ticks,binning,scenarios | tail -3 | tee "$OUT/compare_timelines.txt" || rc=1
else
  echo "STEP 5 NOT VERIFIED: no reference at $OUT/timelines_cf_wp9.json -- the timelines regression"
  echo "  compared NOTHING.  Re-baseline with:  WP9_REBASELINE=1 bash tools/wp9_validate.sh"
  rc=1
fi
$PY dump_results_json.py --out "$OUT/ce_results_cf_wp9_new.json" --manifests benchmark_manifest_80_99.json \
    --tag "cf=$CF" || rc=1
if [ -f "$OUT/ce_results_cf_wp9.json" ]; then
  $PY tools/compare_reports.py "$OUT/ce_results_cf_wp9.json" "$OUT/ce_results_cf_wp9_new.json" \
      --sections cf | tail -3 | tee "$OUT/compare_dump.txt" || rc=1
else
  echo "STEP 5 NOT VERIFIED: no reference at $OUT/ce_results_cf_wp9.json -- the dump regression"
  echo "  compared NOTHING.  Re-baseline with:  WP9_REBASELINE=1 bash tools/wp9_validate.sh"
  rc=1
fi
# Re-baselining: adopt this run's outputs as the references, so that the NEXT
# run compares against them and reads new-keys=0.  Only ever after reading the
# comparison above -- WP9_REBASELINE=1 declares "the new keys are the intended
# output of this tree", it does not verify anything.
if [ "${WP9_REBASELINE:-0}" = "1" ]; then
  cp -f "$OUT/timelines_cf_wp9_new.json" "$OUT/timelines_cf_wp9.json"
  cp -f "$OUT/ce_results_cf_wp9_new.json" "$OUT/ce_results_cf_wp9.json"
  echo "re-baselined: $OUT/timelines_cf_wp9.json and $OUT/ce_results_cf_wp9.json now hold this run's output"
fi

echo "== 6. hash-invariance smoke (repeat-run pair; EXPECTED rc=2, precondition not established) =="
if [ -f "$R6/cf_corun_80_89.pkl" ]; then
  $PY tools/hash_invariance_check.py "$R6/cf_static_80_89.pkl" "$R6/cf_corun_80_89.pkl" --arms marl_static \
      --quiet --out "$OUT/s8_smoke_repeat.json" | tail -8
  echo "  (rc above is not folded into the script's rc: neither frozen pickle records a PYTHONHASHSEED,"
  echo "   so this pair is a repeat-run pair and can never establish S8 -- that is the point of the step)"
else
  echo "STEP 6 SKIPPED: no $R6/cf_corun_80_89.pkl"
fi
echo "== done rc=$rc $(date -u +%FT%TZ) =="
exit $rc
