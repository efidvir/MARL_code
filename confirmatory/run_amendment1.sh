#!/bin/bash
# AMENDMENT 1 chain (see confirmatory/AMENDMENT_1.md):
#   1. re-select the deployed replicate with the static executor on the
#      validation seeds (60-66), by the pre-fixed rule;
#   2. evaluate marl_static on the confirmatory seeds 80-99 (two batches);
#   3. re-run the development set (50-59, 70-79) with marl_static;
#   4. interferer stress test and end-state dump with marl_static;
#   5. timelines;
#   6. once the original confirmatory job (baseline arms) has finished, the
#      amended pre-registered analysis.
set -u
cd ~/MARL_run_6 || exit 1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
# fixed hash seed: makes the energy/latency accounting of these runs reproducible
# (the decision series are hash-seed invariant; see AMENDMENT_1.md 2.2)
export PYTHONHASHSEED=0
export MARL_STEER_MODEL=actionable
ENGINE="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1"
POLICY="MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01"
R5="$HOME/MARL_run_5"
L=confirmatory/A1_chain.log
stamp() { date -u '+%F %T UTC'; }

echo "=== $(stamp)  AMENDMENT-1 CHAIN START ===" >> "$L"
sha256sum -c --quiet confirmatory/FREEZE_A1.sha256 >> "$L" 2>&1 || { echo "=== FREEZE_A1 VIOLATED -- refusing ===" >> "$L"; exit 1; }
echo "    freeze verified ($(wc -l < confirmatory/FREEZE_A1.sha256) files)" >> "$L"

static_run () {  # $1 checkpoint  $2 seed list  $3 output name  $4 extra env  $5 arms
  local ck="$1" seeds="$2" out="$3" extra="${4:-}" arms="${5:-marl_static}"
  env $ENGINE $POLICY $extra MARL_EVAL_CHECKPOINT="$ck" nice -n 5 python3 run_timeline_comparison.py \
      --seed-list "$seeds" --arms "$arms" --no-figures --out "$out" > "logs/${out%.pkl}.log" 2>&1
  local rc=$?
  echo "    $(stamp)  $out exit=$rc" >> "$L"
  if [ $rc -ne 0 ] || [ ! -s "output/$out" ]; then
    echo "    technical failure on $out: one unchanged re-run (pre-registered)" >> "$L"
    env $ENGINE $POLICY $extra MARL_EVAL_CHECKPOINT="$ck" nice -n 5 python3 run_timeline_comparison.py \
        --seed-list "$seeds" --arms "$arms" --no-figures --out "$out" > "logs/${out%.pkl}.rerun.log" 2>&1
    echo "    $(stamp)  $out re-run exit=$?" >> "$L"
  fi
}

# ---------------------------------------------------------------- 1. selection
echo "=== $(stamp)  step 1: replicate re-selection, static executor, seeds 60-66 ===" >> "$L"
static_run "$PWD/ce_ck_s1234/policy_final.pt" 60-66 A1_val_s1234.pkl &
static_run "$PWD/ce_ck_s2345/policy_final.pt" 60-66 A1_val_s2345.pkl &
wait
static_run "$PWD/ce_ck_s3456/policy_final.pt" 60-66 A1_val_s3456.pkl &
static_run "$PWD/ce_ck_s4567/policy_final.pt" 60-66 A1_val_s4567.pkl &
wait
BEST=$(python3 - 2>> confirmatory/A1_selection.txt <<'PY'
import pickle, sys
W = 500
def score(p):
    rec = pickle.load(open(p, 'rb'))['recovery_multi']['marl_static']
    cs, ac = [], []
    for s, r in rec.items():
        if not r:
            continue
        f = [v for v in r['fragments'][-W:] if isinstance(v, (int, float))]
        c = [v for v in r['conn'][-W:] if isinstance(v, (int, float))]
        if f and c:
            cs.append(sum(f) / len(f)); ac.append(sum(c) / len(c))
    return (sum(cs) / len(cs), sum(ac) / len(ac), len(cs)) if cs else (9e9, 0.0, 0)
res = {s: score('output/A1_val_s%s.pkl' % s) for s in ('1234', '2345', '3456', '4567')}
print('rule: lowest mean steady components over validation seeds 60-66 (Scenario A); ties -> higher achievability', file=sys.stderr)
for s, (c, a, n) in res.items():
    print('  static s%s  components %.3f  achievability %.1f  (instances %d/7)' % (s, c, a, n), file=sys.stderr)
best = min(res.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
print('SELECTED: s%s' % best, file=sys.stderr)
print(best)
PY
)
[ -n "$BEST" ] || { echo "=== SELECTION FAILED ===" >> "$L"; exit 1; }
MCK="$PWD/ce_ck_s$BEST/policy_final.pt"
sha256sum "$MCK" >> confirmatory/A1_selection.txt
echo "=== $(stamp)  VALIDATION WINNER (static executor): s$BEST ===" >> "$L"
sed 's/^/    /' confirmatory/A1_selection.txt >> "$L"

# ---------------------------------------------------------------- 2. confirmatory
echo "=== $(stamp)  step 2: marl_static on confirmatory seeds 80-99 ===" >> "$L"
# XL-DET is co-run in the same invocation (sensitivity analysis, AMENDMENT_1.md 2.5);
# the primary analysis reads a marl_static-only copy of each pickle, so the
# pre-registered baselines still come from the original job.
static_run "$MCK" 80-89 cf_corun_80_89.pkl "" marl_static,xldet
static_run "$MCK" 90-99 cf_corun_90_99.pkl "" marl_static,xldet
python3 - >> "$L" 2>&1 <<'PY'
import pickle
for part in ('80_89', '90_99'):
    P = pickle.load(open('output/cf_corun_%s.pkl' % part, 'rb'))
    only = {k: ({'marl_static': v['marl_static']} if str(k).endswith('_multi') else v) for k, v in P.items()}
    pickle.dump(only, open('output/cf_static_%s.pkl' % part, 'wb'))
    print('    split %s -> marl_static only: %s' % (part, {k: sorted(v['marl_static']) for k, v in only.items() if str(k).endswith('_multi')}))
PY
sha256sum output/cf_corun_80_89.pkl output/cf_corun_90_99.pkl output/cf_static_80_89.pkl output/cf_static_90_99.pkl >> "$L" 2>&1

# ---------------------------------------------------------------- 3. development
echo "=== $(stamp)  step 3: marl_static on development seeds ===" >> "$L"
static_run "$MCK" 50-59 dev_static_50_59.pkl
static_run "$MCK" 70-79 dev_static_70_79.pkl

# ---------------------------------------------------------------- 4. stress + end state
echo "=== $(stamp)  step 4: stress test and end-state dump ===" >> "$L"
static_run "$MCK" 50-59 stress_static_50_59.pkl "MARL_STRESS_INTERFERENCE_DBM=-75 MARL_STRESS_INTERFERENCE_TICK=2000"
env $ENGINE $POLICY nice -n 5 python3 dump_endstate.py 58 "$MCK" marl_static output/endstate_static_seed58.json \
    > logs/endstate_static_seed58.log 2>&1
echo "    $(stamp)  end-state dump exit=$?" >> "$L"

# ---------------------------------------------------------------- 5. timelines (development)
python3 extract_timelines.py \
    --pickles "$R5/output/ce_eval_50_59.pkl,$R5/output/ce_eval_70_79.pkl,output/dev_static_50_59.pkl,output/dev_static_70_79.pkl" \
    --out confirmatory/timelines_dev_static.json >> "$L" 2>&1

# ---------------------------------------------------------------- 6. analysis
echo "=== $(stamp)  step 6: waiting for the original confirmatory job (baseline arms) ===" >> "$L"
until grep -q "CONFIRMATORY CHAIN COMPLETE" "$R5/confirmatory/chain.log" 2>/dev/null; do sleep 120; done
echo "=== $(stamp)  original job complete; amended pre-registered analysis ===" >> "$L"
python3 confirmatory_analysis.py \
    --pickles "$R5/output/cf_eval_80_99.pkl,output/cf_static_80_89.pkl,output/cf_static_90_99.pkl" \
    --manifests benchmark_manifest_80_99.json --marl-arm marl_static \
    --out confirmatory/cf_report_A1.json > confirmatory/cf_report_A1.txt 2>&1
echo "    $(stamp)  confirmatory analysis exit=$?" >> "$L"
python3 confirmatory_analysis.py \
    --pickles "$R5/output/ce_eval_50_59.pkl,$R5/output/ce_eval_70_79.pkl,output/dev_static_50_59.pkl,output/dev_static_70_79.pkl" \
    --manifests benchmark_manifest.json,benchmark_manifest_70_79.json --marl-arm marl_static \
    --out confirmatory/dev_report_A1.json > confirmatory/dev_report_A1.txt 2>&1
echo "    $(stamp)  development analysis exit=$?" >> "$L"
python3 A1_sensitivity.py > confirmatory/A1_sensitivity.txt 2>&1
echo "    $(stamp)  sensitivity analysis exit=$?" >> "$L"
python3 extract_timelines.py \
    --pickles "$R5/output/cf_eval_80_99.pkl,output/cf_static_80_89.pkl,output/cf_static_90_99.pkl" \
    --out confirmatory/timelines_cf.json >> "$L" 2>&1
echo "=== $(stamp)  AMENDMENT-1 CHAIN COMPLETE ===" >> "$L"
