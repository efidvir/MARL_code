#!/bin/bash
# ONE evaluation invocation of the final-rerun chain.
#
#   bash confirmatory/stage_eval.sh --print-command PROFILE INV [CKPT]
#   bash confirmatory/stage_eval.sh --run           PROFILE INV [CKPT]
#
# --print-command gives run_final.sh the exact line it writes into the job
# script (so the ledger, the job script and any unchanged re-run all carry
# the same text).  --run does the work: it creates the output directory,
# launches run_timeline_comparison.py under `nice -n 19`, and attaches
# tools/rss_monitor.py to the process tree for the whole invocation, as the
# plan's monitoring section requires.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
[ -n "$MODE" ] && [ -n "$PROFILE" ] && [ -n "$INV" ] || {
  echo "usage: $0 --print-command|--run PROFILE INV [CKPT]" >&2; exit 2; }

eval "$($PY "$TREE/confirmatory/chain_plan.py" --profile "$PROFILE" --emit "$INV")"

if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_eval.sh --run $PROFILE $INV"
  exit 0
fi
[ "$MODE" = --run ] || { echo "unknown mode $MODE" >&2; exit 2; }

cd "$TREE" || exit 1
OUTDIR="output/$(dirname "$INV_OUT")"
mkdir -p "$OUTDIR" "logs/chain_${PROFILE}"

ARGS=(--seed-list "$INV_SEEDS" --arms "$INV_ARMS" --no-figures --out "$INV_OUT")
[ -n "$INV_TICKS" ] && ARGS+=(--ticks "$INV_TICKS")

echo "[stage_eval] $INV  arms=$INV_ARMS seeds=$INV_SEEDS ticks=${INV_TICKS:-full}"
echo "[stage_eval] MARL_POOL_WORKERS=${MARL_POOL_WORKERS:-unset} PYTHONHASHSEED=${PYTHONHASHSEED:-unset}"
echo "[stage_eval] MARL_EVAL_CHECKPOINT=${MARL_EVAL_CHECKPOINT:-unset}"
echo "[stage_eval] argv: python3 run_timeline_comparison.py ${ARGS[*]}"

nice -n 19 "$PY" run_timeline_comparison.py "${ARGS[@]}" &
MAIN=$!
"$PY" tools/rss_monitor.py "$MAIN" "logs/chain_${PROFILE}/${INV}.rss.tsv" --interval 5 \
      > "logs/chain_${PROFILE}/${INV}.rss.log" 2>&1 &
wait "$MAIN"
RC=$?
echo "[stage_eval] $INV run_timeline_comparison exit=$RC"
exit $RC
