#!/bin/bash
# S3b: apply the PRE-REGISTERED selection rule to the validation passes and
# write confirmatory/SELECTED_<profile>.txt (line 1 = the checkpoint path,
# relative to the tree; the rest is the audit trail).
#
# The rule is the one already used in confirmatory/run_amendment1.sh:
#   lowest mean steady components over the validation seeds (Scenario A),
#   ties broken by higher achievability.
# It is applied ONCE, to the complete set of replicates.  The script fails
# when a replicate is missing, when a replicate scored zero instances, or
# when fewer than one replicate could be scored at all -- a selection that
# "succeeds" without having compared anything is exactly the defect this
# campaign keeps producing.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_select.sh --run $PROFILE $INV"
  exit 0
fi
cd "$TREE" || exit 1

# --window PROFILE prints the steady window this profile would score and exits
# non-zero when the window is unusable at that tick count.  It exists so the
# scaling rule is testable without running a selection.
if [ "$MODE" = --window ]; then
  "$PY" - "$PROFILE" <<'PYW'
import sys
sys.path.insert(0, 'confirmatory')
import chain_plan
P = chain_plan.PROFILES[sys.argv[1]]
NOMINAL = P['nominal_ticks']
TICKS = P['ticks'] or NOMINAL
W = max(1, int(round(500.0 * TICKS / float(NOMINAL))))
SEV = min(200, max(50, TICKS // 6))
RESCUE = min(1400, max(SEV + 50, TICKS // 2))
print('ticks=%d W=%d window=%d-%d severance=%d rescue=%d usable=%s'
      % (TICKS, W, TICKS - W, TICKS, SEV, RESCUE, TICKS - W > RESCUE))
sys.exit(0 if TICKS - W > RESCUE else 1)
PYW
  exit $?
fi

"$PY" - "$PROFILE" <<'PYX'
import os, pickle, sys, hashlib
sys.path.insert(0, 'confirmatory')
import chain_plan

profile = sys.argv[1]
P = chain_plan.PROFILES[profile]

# THE STEADY WINDOW SCALES WITH THE RUN LENGTH.
# The pre-registered rule scores the last W ticks of the run, with W = 500 at
# the nominal 3600 ticks.  Held absolute, that window straddles the severance
# at any reduced tick count -- at 600 ticks it would cover ticks 100..600 and
# the severance is at tick 100 -- so the rule would score the disruption
# instead of the steady tail and would have no discriminating power at all.
# run_timeline_comparison.py scales the scenario ticks the same way
# (SEVERANCE_TICK = min(200, max(50, T//6)), RESCUE_ARRIVAL_TICK =
# min(1400, max(SEV+50, T//2))), so the window is scaled to match, and the
# script REFUSES to run at any tick count where the window would still reach
# back past the last scenario event.
NOMINAL = P['nominal_ticks']
TICKS = P['ticks'] or NOMINAL
W = max(1, int(round(500.0 * TICKS / float(NOMINAL))))
SEV = min(200, max(50, TICKS // 6))
RESCUE = min(1400, max(SEV + 50, TICKS // 2))
if TICKS - W <= RESCUE:
    sys.exit('[stage_select] FAIL: at %d ticks the steady window would start at '
             'tick %d, on or before the last scenario event (severance %d, '
             'rescue %d).  The selection rule cannot be applied at this scale; '
             'it is NOT EXERCISED, not passed.' % (TICKS, TICKS - W, SEV, RESCUE))
print('[stage_select] steady window: last %d of %d ticks (%d-%d); severance %d, '
      'rescue %d' % (W, TICKS, TICKS - W, TICKS, SEV, RESCUE))
rows, missing = {}, []
for s in P['torch_seeds']:
    pkl = 'output/%ss3_val_s%d.pkl' % (P['out_prefix'], s)
    if not os.path.exists(pkl):
        missing.append(pkl)
        continue
    rec = pickle.load(open(pkl, 'rb'))['recovery_multi'].get('marl_static', {})
    cs, ac = [], []
    for seed, r in rec.items():
        if not r:
            continue
        f = [v for v in r['fragments'][-W:] if isinstance(v, (int, float))]
        c = [v for v in r['conn'][-W:] if isinstance(v, (int, float))]
        if f and c:
            cs.append(sum(f) / len(f)); ac.append(sum(c) / len(c))
    rows[s] = (sum(cs) / len(cs) if cs else 9e9,
               sum(ac) / len(ac) if ac else 0.0, len(cs))

if missing:
    sys.exit('[stage_select] FAIL: missing validation pickle(s): %s' % ', '.join(missing))
if not rows:
    sys.exit('[stage_select] FAIL: no replicate could be scored')
empty = [s for s, v in rows.items() if v[2] == 0]
if empty:
    sys.exit('[stage_select] FAIL: replicate(s) %s scored 0 instances' % empty)

print('[stage_select] rule: lowest mean steady components over validation seeds '
      '%s (Scenario A); ties -> higher achievability' % P['val_seeds'])
comps = sorted(v[0] for v in rows.values())
DEGENERATE = len(comps) > 1 and comps[0] == comps[-1]
if DEGENERATE:
    print('[stage_select] WARNING: every replicate scored the SAME components '
          '(%.6f).  The primary criterion had ZERO discriminating power and the '
          'tie-break alone chose the checkpoint: the selection rule is NOT '
          'EXERCISED by this run and must not be reported as proven.' % comps[0])
for s in sorted(rows):
    c, a, n = rows[s]
    print('[stage_select]   replicate s%-6d components %8.3f  achievability %6.1f  instances %d/%d'
          % (s, c, a, n, len(P['val_seeds'])))
best = min(rows.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
ck = 'output/%sck_s%d/policy_final.pt' % (P['out_prefix'], best)
if not os.path.exists(ck):
    sys.exit('[stage_select] FAIL: selected checkpoint %s does not exist' % ck)
h = hashlib.sha256(open(ck, 'rb').read()).hexdigest()
out = 'confirmatory/SELECTED_%s.txt' % profile
with open(out, 'w') as f:
    f.write(ck + '\n')
    f.write('# selected replicate: torch seed %d\n' % best)
    f.write('# sha256 %s\n' % h)
    f.write('# rule: lowest mean steady components over validation seeds %s '
            '(Scenario A); ties -> higher achievability\n' % P['val_seeds'])
    f.write('# replicates scored: %d\n' % len(rows))
    f.write('# steady window: last %d of %d ticks; severance %d, rescue %d\n'
            % (W, TICKS, SEV, RESCUE))
    if DEGENERATE:
        f.write('# NOT EXERCISED: every replicate tied on the primary criterion;\n'
                '#               the tie-break alone chose this checkpoint\n')
    for s in sorted(rows):
        f.write('#   s%d components %.6f achievability %.4f instances %d\n'
                % (s, rows[s][0], rows[s][1], rows[s][2]))
print('[stage_select] SELECTED replicate s%d -> %s (sha256 %s)' % (best, ck, h[:16]))
print('[stage_select] wrote %s after scoring %d replicate(s)' % (out, len(rows)))
PYX
exit $?
