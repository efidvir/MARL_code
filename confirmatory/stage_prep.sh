#!/bin/bash
# S0 preparation steps of the final-rerun chain.
#
#   manifest   build_benchmark_manifest.py over the confirmatory seeds [C]
#   untrained  make_untrained_checkpoints.py for every replicate seed [B]
#
# Both depend only on the frozen tree.  The untrained checkpoints are also
# the PLACEHOLDER checkpoint stage S1 carries.  It is no longer required:
# since the trackAgent merge run_timeline_comparison.main() requires and
# opens a checkpoint only when an arm that reads one (CHECKPOINT_ARMS:
# marl, marl_freeze, marl_static) is selected, and never hands one to a
# policy-independent arm (xldet and random take their own branches in
# run_physics_pass; the routing arms are not on the MARL path at all).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
eval "$($PY "$TREE/confirmatory/chain_plan.py" --profile "$PROFILE" --emit "$INV")"

if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_prep.sh --run $PROFILE $INV"
  exit 0
fi
cd "$TREE" || exit 1

case "$INV_STEP" in
  manifest)
    LO=${INV_SEEDS%%,*}; HI=${INV_SEEDS##*,}
    echo "[stage_prep] benchmark manifest for seeds $LO-$HI -> $INV_ARTIFACT"
    nice -n 19 "$PY" build_benchmark_manifest.py --seeds "$LO-$HI" --out "$INV_ARTIFACT"
    RC=$?
    if [ $RC -eq 0 ]; then
      "$PY" - "$INV_ARTIFACT" "$LO" "$HI" <<'PYX'
import json, sys
m = json.load(open(sys.argv[1]))
lo, hi = int(sys.argv[2]), int(sys.argv[3])
inst = m.get('instances') or []
want = 2 * (hi - lo + 1)          # both scenario kinds per seed
print('[stage_prep] manifest holds %d instances (expected %d)' % (len(inst), want))
if len(inst) != want:
    sys.exit('[stage_prep] FAIL: manifest instance count does not match the seed set')
seeds = sorted({int(i['topology_seed']) for i in inst})
if seeds != list(range(lo, hi + 1)):
    sys.exit('[stage_prep] FAIL: manifest seeds %s' % seeds)
PYX
      RC=$?
    fi
    exit $RC
    ;;
  untrained)
    SEEDS=$("$PY" -c "import sys;sys.path.insert(0,'$TREE/confirmatory');import chain_plan;print(' '.join(str(s) for s in chain_plan.PROFILES['$PROFILE']['torch_seeds']))")
    PRE=$("$PY" "$TREE/confirmatory/chain_plan.py" --profile "$PROFILE" --profile-field out_prefix)
    echo "[stage_prep] untrained-actor checkpoints for torch seeds: $SEEDS"
    nice -n 19 "$PY" make_untrained_checkpoints.py $SEEDS
    RC=$?
    [ $RC -eq 0 ] || exit $RC
    # make_untrained_checkpoints.py writes untrained_s<seed>/ in the CWD with
    # no way to redirect it (REQUEST in docs/RUN_CHAIN.md).  Move the result
    # under output/<profile>/ so a rehearsal can never land in, or overwrite,
    # a path the real run uses.
    N=0
    for s in $SEEDS; do
      if [ -s "untrained_s$s/policy_init.pt" ]; then
        mkdir -p "output/${PRE}untrained_s$s"
        mv -f "untrained_s$s/policy_init.pt" "output/${PRE}untrained_s$s/policy_init.pt"
        rmdir "untrained_s$s" 2>/dev/null
        N=$((N + 1))
        echo "[stage_prep] output/${PRE}untrained_s$s/policy_init.pt $(sha256sum "output/${PRE}untrained_s$s/policy_init.pt" | cut -c1-16)"
      else
        echo "[stage_prep] FAIL: untrained_s$s/policy_init.pt missing"
        exit 1
      fi
    done
    [ "$N" -gt 0 ] || { echo "[stage_prep] FAIL: no checkpoint was written"; exit 1; }
    echo "[stage_prep] wrote $N untrained checkpoint(s)"
    exit 0
    ;;
  *)
    echo "[stage_prep] unknown step: $INV_STEP" >&2
    exit 2
    ;;
esac
