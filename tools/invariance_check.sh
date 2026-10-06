#!/bin/bash
# Invariance matrix check (docs/RERUN_SPEC.md section 14).
#
#   tools/invariance_check.sh REF.pkl NEW.pkl ARM [ARM ...]
#   tools/invariance_check.sh --wp WP3 REF.pkl NEW.pkl [ARM ...]
#
# Every ARM listed must be bit-identical between REF (previous commit) and
# NEW (this commit) on every series, scenario and seed.  With --wp the
# must-be-identical arms come from the section-14 matrix (extra ARM
# arguments are added to them); all other arms present in the pickles are
# reported as INFO (changes allowed; an arm absent from one pickle is
# reported as not run).
#
# Environment:
#   INV_KEYS           series to compare (default "common": every key REF
#                      carries, plus the series_diff DEFAULT_KEYS, must be
#                      present and identical in NEW; keys only NEW carries
#                      are listed in a NOTE).  A comma list or "all" also works.
#   INV_ALLOW_REMOVED  comma list of REF keys NEW may lack (deliberate
#                      removals; listed in a NOTE).  The DEFAULT_KEYS
#                      (conn, delivered, fragments, ...) cannot be listed.
#   INV_SEEDS          comma list of seeds (default: all seeds in the pickles)
#   INV_SCENARIOS      comma list of scenarios, recovery,rescue_ops or A,B
#                      (default: both)
#
# Exit status: 0 = PASS, 1 = FAIL (difference or missing series), 2 = usage.
#
# Matrix (section 14; "routing" = ospf olsr batman aodv):
#   WP0, WP9, WP11, WP-DET  all arms (no behaviour change).  WP-DET touches
#                           only the TRAINING driver plus one engine field
#                           that nothing reads (IOPSRegistration.local_token,
#                           formerly uuid4), so every evaluation arm must be
#                           bit-identical.
#   WP1 .. WP5, WP12, WP13  routing + sdn
#   WP6, WP7, WP8           routing + sdn + xldet.  XL-DET is identical only
#                           up to the relay-mode aliasing; the pickles carry no
#                           per-tick relay-mode codes yet, so every recorded
#                           series must match as-is (relay_mode_nodes counts
#                           sites in a non-OFF mode, which the aliasing keeps).
#                           Once an action series is recorded, compare it on
#                           the mapped modes.
#   WP10                    every arm except marl_freeze
#   D12.2                   routing + marl_static + xldet + random: the SDN
#                           baseline fix changes SDN (Scenario A by design;
#                           Scenario B must stay identical -- check it with
#                           INV_SCENARIOS) and nothing else.  (D12.1 has no
#                           row: it changes every arm by design, and its
#                           evidence is diag/run_wp12.sh ue, D12F.6.)
#   D12F.8                  ospf + sdn + marl_static + xldet + random: the
#                           leaf model for the OLSR, B.A.T.M.A.N. and AODV
#                           control planes changes those three and nothing
#                           else (OSPF's decomposition moved unchanged into
#                           the shared infra_fragments_with_ue_leaves).
#   XL1                     routing + sdn + marl_static + random: XL-DET's
#                           interference-brake fix (D-XL.1) changes XL-DET
#                           and nothing else.
#                           After D12-FIXES every WP compares against the
#                           D12 references (output/ref/README.md).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
WP=""
if [ "${1:-}" = "--wp" ]; then
  WP="$(echo "${2:-}" | tr '[:lower:]' '[:upper:]')"; shift 2 || true
fi
if [ $# -lt 2 ]; then
  sed -n '2,12p' "$0" >&2
  exit 2
fi
REF="$1"; NEW="$2"; shift 2
for f in "$REF" "$NEW"; do
  [ -f "$f" ] || { echo "invariance_check: no such file: $f" >&2; exit 2; }
done
KEYS="${INV_KEYS:-common}"
EXTRA_OPT=()
[ -n "${INV_SEEDS:-}" ] && EXTRA_OPT+=(--seeds "$INV_SEEDS")
[ -n "${INV_SCENARIOS:-}" ] && EXTRA_OPT+=(--scenarios "$INV_SCENARIOS")
[ -n "${INV_ALLOW_REMOVED:-}" ] && EXTRA_OPT+=(--allow-removed "$INV_ALLOW_REMOVED")

# arms present in each pickle (line 1: REF, line 2: NEW)
ARMS_BY_PICKLE="$("$PY" - "$REF" "$NEW" <<'PY'
import pickle, sys
for p in sys.argv[1:]:
    with open(p, 'rb') as f:
        P = pickle.load(f)
    arms = []
    for sk in ('recovery_multi', 'rescue_multi'):
        for a in (P.get(sk) or {}):
            if a not in arms:
                arms.append(a)
    print(' '.join(arms) or '-')
PY
)" || { echo "invariance_check: cannot read pickles" >&2; exit 2; }
REF_ARMS="$(echo "$ARMS_BY_PICKLE" | sed -n 1p)"
NEW_ARMS="$(echo "$ARMS_BY_PICKLE" | sed -n 2p)"
PRESENT="$(echo $REF_ARMS $NEW_ARMS | tr ' ' '\n' | awk 'NF && $0 != "-" && !seen[$0]++' | tr '\n' ' ')"
has() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

ROUTING="ospf olsr batman aodv"
MUST=""
case "$WP" in
  "") ;;
  WP0|WP9|WP11|WP-DET|WPDET) MUST="$PRESENT" ;;
  WP1|WP2|WP3|WP4|WP5|WP12|WP13) MUST="$ROUTING sdn" ;;
  WP6|WP7|WP8) MUST="$ROUTING sdn xldet" ;;
  WP10) MUST="$(for a in $PRESENT; do [ "$a" = marl_freeze ] || printf '%s ' "$a"; done)" ;;
  D12.2|D122) MUST="$ROUTING marl_static xldet random" ;;
  D12F.8|D12F8) MUST="ospf sdn marl_static xldet random" ;;
  XL1|XL-1) MUST="$ROUTING sdn marl_static random" ;;
  *) echo "invariance_check: unknown work package '$WP'" >&2; exit 2 ;;
esac
MUST="$(echo $MUST "$@" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
[ -n "${MUST// }" ] || { echo "invariance_check: no arms to check" >&2; exit 2; }

# where: which pickle lacks the arm ("" when both carry it)
where_missing() {
  local m=""
  has "$1" "$REF_ARMS" || m="REF"
  has "$1" "$NEW_ARMS" || m="${m:+$m+}NEW"
  echo "$m"
}

echo "=== invariance check ${WP:-(explicit arms)}  keys=$KEYS${INV_ALLOW_REMOVED:+  allow-removed=$INV_ALLOW_REMOVED}"
echo "    ref: $REF"
echo "    new: $NEW"
rc=0
for arm in $MUST; do
  out="$("$PY" "$HERE/series_diff.py" "$REF" "$NEW" --arms "$arm" --keys "$KEYS" "${EXTRA_OPT[@]}" --quiet)"
  st=$?
  [ $st -eq 2 ] && { echo "invariance_check: series_diff usage error" >&2; exit 2; }
  summ="$(echo "$out" | grep '^SUMMARY' | sed 's/^SUMMARY //')"
  miss="$(where_missing "$arm")"
  if [ $st -eq 0 ]; then
    echo "PASS  $arm  $summ"
  else
    rc=1
    if [ -n "$miss" ]; then
      echo "FAIL  $arm  not run in $miss (required${WP:+ for $WP})  ${summ:-series_diff exit $st}"
    else
      echo "FAIL  $arm  ${summ:-series_diff exit $st}"
      echo "$out" | grep -v -E '^(REF|NEW|SUMMARY|NOTE|  ref:|  new:)' | head -20 | sed 's/^/      /'
    fi
  fi
  echo "$out" | grep '^NOTE' | sed 's/^/      /'
done
for arm in $PRESENT; do
  has "$arm" "$MUST" && continue
  miss="$(where_missing "$arm")"
  if [ -n "$miss" ]; then
    echo "INFO  $arm  not run in $miss (not compared)"
    continue
  fi
  out="$("$PY" "$HERE/series_diff.py" "$REF" "$NEW" --arms "$arm" --keys "$KEYS" "${EXTRA_OPT[@]}" --quiet)"
  st=$?
  [ $st -eq 2 ] && { echo "invariance_check: series_diff usage error" >&2; exit 2; }
  summ="$(echo "$out" | grep '^SUMMARY' | sed 's/^SUMMARY //')"
  if [ $st -eq 0 ]; then
    echo "INFO  $arm  unchanged  $summ"
  else
    echo "INFO  $arm  changed (allowed for ${WP:-this check})  $summ"
  fi
  echo "$out" | grep '^NOTE' | sed 's/^/      /'
done
echo "$("$PY" "$HERE/series_diff.py" "$REF" "$NEW" --arms none --keys "$KEYS" 2>/dev/null | grep -E '^  (ref|new):')"
if [ $rc -eq 0 ]; then echo "INVARIANCE PASS"; else echo "INVARIANCE FAIL"; fi
exit $rc
