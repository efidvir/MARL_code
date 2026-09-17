#!/bin/bash
# Final-binary smoke: static audit fields + baseline identity (seed 58, short).
set -u
cd ~/MARL_run_6 || exit 1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MARL_STEER_MODEL=actionable
ENV="MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01"
CK="$PWD/ce_ck_s1234/policy_final.pt"
L=logs/smoke2.log
echo "=== $(date -u '+%F %T UTC') smoke2 start ===" > $L
rm -f output/smoke2_*.pkl _orig_harness/output/smoke2_*.pkl
( env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 400 \
    --arms marl_static --no-figures --out smoke2_static.pkl > logs/smoke2_static.log 2>&1; echo "static exit=$?" >> $L ) &
( env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 400 \
    --arms xldet,ospf,sdn,random,marl_freeze --no-figures --out smoke2_base_new.pkl > logs/smoke2_base_new.log 2>&1; echo "base_new exit=$?" >> $L ) &
( cd _orig_harness && env $ENV MARL_EVAL_CHECKPOINT="$CK" nice -n 5 python3 run_timeline_comparison.py --seed-list 58 --ticks 400 \
    --arms xldet,ospf,sdn,random,marl_freeze --no-figures --out smoke2_base_orig.pkl > ../logs/smoke2_base_orig.log 2>&1; echo "base_orig exit=$?" >> ../$L ) &
wait
python3 - >> $L 2>&1 <<'PY'
import pickle, math
JITTER = {'energy': 1e-3, 'latency': 5e-2}      # measured run-to-run jitter bounds (relative)
def load(p): return pickle.load(open(p, 'rb'))
def isnum(x): return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))
def same(a, b, tol=None):
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if isnum(x) and isnum(y):
            if tol is None:
                if x != y: return False
            elif abs(x - y) > tol * max(1.0, abs(y)): return False
        elif not (x == y or (not isnum(x) and not isnum(y))): return False
    return True
ok = True
s = load('output/smoke2_static.pkl')
for scen in ('recovery_multi', 'rescue_multi'):
    r = s[scen]['marl_static'][58]
    good = r.get('static_fp_start') and r.get('static_fp_start') == r.get('static_fp_end')
    print('static audit %-14s start=%s end=%s -> %s' % (scen, str(r.get('static_fp_start'))[:16], str(r.get('static_fp_end'))[:16], 'PASS' if good else 'FAIL'))
    ok &= bool(good)
nb = load('output/smoke2_base_new.pkl'); ob = load('_orig_harness/output/smoke2_base_orig.pkl')
for scen in ('recovery_multi', 'rescue_multi'):
    for arm in ('xldet', 'ospf', 'sdn', 'random'):
        a = nb[scen][arm][58]; b = ob[scen][arm][58]
        keys = sorted(k for k in a if isinstance(a[k], list))
        exact = [k for k in keys if k not in JITTER]
        bad_exact = [k for k in exact if not same(a[k], b[k])]
        bad_tol = [k for k in JITTER if k in a and not same(a[k], b[k], JITTER[k])]
        scal = [k for k in a if not isinstance(a[k], list) and a[k] != b.get(k)]
        res = not bad_exact and not bad_tol and not scal and set(a) == set(b)
        print('baseline identity %-14s %-7s exact on %d series%s -> %s' % (
            scen, arm, len(exact), (' | differ: %s %s %s' % (bad_exact, bad_tol, scal)) if not res else '', 'PASS' if res else 'FAIL'))
        ok &= res
print('SMOKE2 RESULT:', 'ALL PASS' if ok else 'FAILURES PRESENT')
PY
echo "=== $(date -u '+%F %T UTC') smoke2 complete ===" >> $L
