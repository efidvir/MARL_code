#!/bin/bash
# Behavioural tests for the marl_static arm (short runs, seed 58).
set -u
cd ~/MARL_run_6 || exit 1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MARL_STEER_MODEL=actionable
ENV="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01"
CK="$PWD/ce_ck_s1234/policy_final.pt"
L=logs/smoke.log
echo "=== $(date -u '+%F %T UTC') smoke start ===" > $L

# pristine copy of the frozen harness for the baseline-identity test
mkdir -p _orig_harness/output
cp -p ~/MARL_run_5/run_timeline_comparison.py _orig_harness/
ln -sfn ../sixg_sim _orig_harness/sixg_sim
ln -sfn ../config _orig_harness/config
for f in *.py; do [ "$f" = run_timeline_comparison.py ] || ln -sfn ../$f _orig_harness/$f; done

( env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 600 \
    --arms marl_static,marl_freeze --no-figures --out smoke_t1.pkl > logs/smoke_t1.log 2>&1; echo "T1 exit=$?" >> $L ) &
( env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 600 \
    --arms marl_static --no-figures --out smoke_t2.pkl > logs/smoke_t2.log 2>&1; echo "T2 exit=$?" >> $L ) &
( env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 400 \
    --arms xldet,ospf,sdn,random,marl_freeze --no-figures --out smoke_base_new.pkl > logs/smoke_base_new.log 2>&1; echo "T3a exit=$?" >> $L ) &
( cd _orig_harness && env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 400 \
    --arms xldet,ospf,sdn,random,marl_freeze --no-figures --out smoke_base_orig.pkl > ../logs/smoke_base_orig.log 2>&1; echo "T3b exit=$?" >> ../$L ) &
wait
echo "=== $(date -u '+%F %T UTC') smoke runs done ===" >> $L
python3 - >> $L 2>&1 <<'PY'
import pickle, math, re
def load(p): return pickle.load(open(p, 'rb'))
def eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b): return True
    return a == b
ok = True
t1 = load('output/smoke_t1.pkl'); t2 = load('output/smoke_t2.pkl')
for scen in ('recovery_multi', 'rescue_multi'):
    a = t1[scen]['marl_static'][58]; b = t2[scen]['marl_static'][58]
    same = all(len(a[k]) == len(b[k]) and all(eq(x, y) for x, y in zip(a[k], b[k]))
               for k in ('conn', 'energy', 'fragments', 'delivered', 'latency'))
    print('T2 determinism %s: %s' % (scen, 'IDENTICAL' if same else 'DIFFERENT')); ok &= same
nb = load('output/smoke_base_new.pkl'); ob = load('_orig_harness/output/smoke_base_orig.pkl')
for scen in ('recovery_multi', 'rescue_multi'):
    for arm in ('xldet', 'ospf', 'sdn', 'random'):
        a = nb[scen][arm][58]; b = ob[scen][arm][58]
        keys = [k for k in a if isinstance(a[k], list)]
        same = all(len(a[k]) == len(b[k]) and all(eq(x, y) for x, y in zip(a[k], b[k])) for k in keys)
        print('T3 baseline identity %-14s %-7s: %s' % (scen, arm, 'IDENTICAL' if same else 'DIFFERENT')); ok &= same
log = open('logs/smoke_t1.log', errors='replace').read()
st = [l for l in log.splitlines() if 'MARL_STATIC' in l]
checks = {
  'static setup line x2': sum('STATIC DEPLOYMENT' in l for l in st) == 2,
  'static check OK x2': sum('STATIC CHECK OK' in l for l in st) == 2,
  'no FREEZE on static': not any('[FREEZE' in l for l in st),
  'no EWC / safe adaptation on static': not any(('[EWC]' in l or 'Safe adaptation' in l) for l in st),
  'no adapted save on static': not any('Saved' in l and 'adapted' in l for l in st),
  'freeze arm still freezes': any('[FREEZE]' in l for l in log.splitlines() if 'MARL_FREEZE' in l),
  'phase 2 holds only marl_freeze (1 task)': re.search(r'for 1 Phase-2 tasks', log) is not None,
  'detection floor covers marl_static': 'marl/marl_freeze/marl_static/random/xldet' in log,
}
for k, v in checks.items():
    print('T1 %-40s %s' % (k, 'PASS' if v else 'FAIL')); ok &= v
print('SMOKE RESULT:', 'ALL PASS' if ok else 'FAILURES PRESENT')
PY
echo "=== $(date -u '+%F %T UTC') smoke complete ===" >> $L
