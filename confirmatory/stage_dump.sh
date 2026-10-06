#!/bin/bash
# S4 [J]: the end-state dump for the process figure.
#
# NOTE (measured in the rehearsal): dump_endstate.py has no tick override --
# it calls run_physics_pass directly, so it always runs a FULL-LENGTH pass
# whatever the profile, and PROBE_TICK (default 3400) is an absolute tick.
# The step therefore costs the same in the rehearsal as in the real run:
# one single-core marl_static pass.  See the REQUEST in docs/RUN_CHAIN.md.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"; CKPT="${4:-}"
eval "$($PY "$TREE/confirmatory/chain_plan.py" --profile "$PROFILE" --emit "$INV")"
if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_dump.sh --run $PROFILE $INV"
  exit 0
fi
cd "$TREE" || exit 1

CK="${MARL_EVAL_CHECKPOINT:-}"
[ -n "$CK" ] || CK=$(head -1 "confirmatory/SELECTED_${PROFILE}.txt" 2>/dev/null || echo '')
[ -n "$CK" ] && [ -e "$CK" ] || { echo "[stage_dump] FAIL: no usable checkpoint"; exit 1; }

SEED="${INV_SEEDS%%,*}"
mkdir -p "$(dirname "$INV_ARTIFACT")"
echo "[stage_dump] seed=$SEED arm=marl_static probe_tick=${PROBE_TICK:-3400} -> $INV_ARTIFACT"
nice -n 19 "$PY" dump_endstate.py "$SEED" "$CK" marl_static "$INV_ARTIFACT"
RC=$?
[ $RC -eq 0 ] || exit $RC
# Count what the figure actually needs: the snapshots, their component
# labels and their bridges.  (A gate that only asks "is the file non-empty"
# is the defect this campaign keeps producing.)
"$PY" - "$INV_ARTIFACT" <<'PYX'
import json, sys
d = json.load(open(sys.argv[1]))
snaps = d.get('snapshots') or {}
if isinstance(snaps, dict):
    snaps = list(snaps.values())
nodes = sum(len((s or {}).get('nodes') or {}) for s in snaps)
links = sum(len((s or {}).get('links') or []) for s in snaps)
bridges = sum(len((s or {}).get('bridge_link_ids') or []) for s in snaps)
print('[stage_dump] dump: arm=%s seed=%s snapshots=%d nodes=%d links=%d '
      'bridge links=%d steady_components=%s'
      % (d.get('arm'), d.get('seed'), len(snaps), nodes, links, bridges,
         d.get('steady_components')))
if not snaps:
    sys.exit('[stage_dump] FAIL: the dump holds no snapshot')
if nodes == 0 or links == 0:
    sys.exit('[stage_dump] FAIL: a snapshot carries no node or link record')
if any((s or {}).get('components') is None for s in snaps):
    sys.exit('[stage_dump] FAIL: a snapshot carries no component count')
if not d.get('provenance'):
    sys.exit('[stage_dump] FAIL: the dump carries no provenance record')
PYX
exit $?
