#!/bin/bash
# S5: the DEV GATE (technical only) and FREEZE_FINAL.
#
# TECHNICAL ONLY.  This gate aborts on a BUG -- a missing pass, an empty
# pickle, a mixed provenance, a checkpoint that cannot be found.  It never
# looks at an effect size and nothing here may be tuned on what it sees.
#
# It then seals FREEZE_FINAL: the tree digest, the manifest digest, the
# selected checkpoint digest and the sha256 of every dev/stress pickle, into
# confirmatory/FREEZE_FINAL_<profile>.txt.
#
# It FAILS when the number of artefacts it verified is zero.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_gate.sh --run $PROFILE $INV"
  exit 0
fi
cd "$TREE" || exit 1

FAIL=0
N=0
PKLS=""
KEYDIR="logs/chain_${PROFILE}"
mkdir -p "$KEYDIR"

echo "[stage_gate] DEV GATE for profile $PROFILE"
for ID in $("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --ids S1) \
          $("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --ids S4); do
  ART=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --field "$ID" artifact)
  case "$ART" in *.pkl) ;; *) echo "[stage_gate] (skipping non-pickle artefact of $ID: $ART)"; continue ;; esac
  "$PY" confirmatory/chain_plan.py --profile "$PROFILE" --keys "$ID" > "$KEYDIR/$ID.keys"
  if "$PY" tools/stage_report.py --verify "$ART" --keys-file "$KEYDIR/$ID.keys"; then
    N=$((N + 1)); PKLS="$PKLS,$ART"
  else
    FAIL=$((FAIL + 1))
  fi
done
PKLS="${PKLS#,}"

if [ "$N" -eq 0 ]; then
  echo "[stage_gate] FAIL: the gate verified ZERO pickles; there is nothing to gate"
  exit 1
fi
echo "[stage_gate] verified $N dev/stress pickle(s), $FAIL failure(s)"

"$PY" tools/stage_report.py --cross-provenance "$PKLS" || FAIL=$((FAIL + 1))

SEL="confirmatory/SELECTED_${PROFILE}.txt"
CK=$(head -1 "$SEL" 2>/dev/null || echo '')
[ -n "$CK" ] && [ -e "$CK" ] || { echo "[stage_gate] FAIL: no selected checkpoint ($SEL)"; FAIL=$((FAIL + 1)); }

MANIFEST=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --field s0_manifest artifact)
[ -s "$MANIFEST" ] || { echo "[stage_gate] FAIL: confirmatory manifest $MANIFEST is missing"; FAIL=$((FAIL + 1)); }

[ "$FAIL" -eq 0 ] || { echo "[stage_gate] DEV GATE FAILED ($FAIL problem(s)) - the chain stops here"; exit 1; }

OUT="confirmatory/FREEZE_FINAL_${PROFILE}.txt"
{
  echo "FREEZE_FINAL  profile=$PROFILE  sealed $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "tree: $TREE"
  "$PY" - <<'PYX'
import sys
sys.path.insert(0, '.')
from sixg_sim import runinfo
p = runinfo.provenance()
for k in ('tree_head', 'tree_head_source', 'tree_sha256', 'tree_files',
          'schema_id', 'pythonhashseed', 'provenance_complete'):
    print('%-20s %s' % (k, p.get(k)))
PYX
  echo "manifest             $(sha256sum "$MANIFEST")"
  echo "selected_checkpoint  $(sha256sum "$CK")"
  echo "dev/stress pickles verified: $N"
  for p in ${PKLS//,/ }; do echo "  $(sha256sum "$p")"; done
} > "$OUT"
echo "[stage_gate] FREEZE_FINAL sealed -> $OUT"
cat "$OUT"
exit 0
