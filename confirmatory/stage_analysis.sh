#!/bin/bash
# S7 (primary) and S12 (secondary) analysis steps.
#
# Before ANY analysis, and regardless of which analysis it is:
#   (i)  the cross-pickle provenance assert -- tree_head, tree_sha256,
#        schema_id and pythonhashseed identical across every pickle merged;
#   (ii) the key-union assert -- the union of (arm, seed, scenario) keys
#        equals the expected confirmatory set exactly, no duplicates, none
#        missing.
# Both are counting gates: zero pickles or zero expected keys is a FAILURE.
# They live here because confirmatory_analysis.py is owned by the analysis
# track and does not yet refuse a partial set (REQUEST in docs/RUN_CHAIN.md).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
eval "$($PY "$TREE/confirmatory/chain_plan.py" --profile "$PROFILE" --emit "$INV")"
if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_analysis.sh --run $PROFILE $INV"
  exit 0
fi
cd "$TREE" || exit 1
KEYDIR="logs/chain_${PROFILE}"; mkdir -p "$KEYDIR"
MANIFEST=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --field s0_manifest artifact)

collect () {              # $1 stage -> comma list of existing pickles (global PKLS/NMISS)
  PKLS=""; NMISS=0
  for id in $("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --ids "$1"); do
    art=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --field "$id" artifact)
    case "$art" in *.pkl) ;; *) continue ;; esac
    if [ -s "$art" ]; then PKLS="$PKLS,$art"; else NMISS=$((NMISS + 1)); fi
  done
  PKLS="${PKLS#,}"
}

collect_prefix () {       # $1 stage  $2 id prefix -> PKLS (global), NMISS
  local st="$1" pre="$2" id art
  PKLS=""; NMISS=0
  for id in $("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --ids "$st"); do
    case "$id" in $pre*) ;; *) continue ;; esac
    art=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --field "$id" artifact)
    case "$art" in *.pkl) ;; *) continue ;; esac
    if [ -s "$art" ]; then PKLS="$PKLS,$art"; else NMISS=$((NMISS + 1)); fi
  done
  PKLS="${PKLS#,}"
}

# ---------------------------------------------------------------------------
# S5b: run the REAL analysis script on the DEVELOPMENT set before committing to
# the confirmatory stage.  confirmatory_analysis.py is written to be validated
# by reproducing the development-set figures, and the development set is
# complete at the end of S4 -- three hours before the confirmatory analysis
# would otherwise be attempted for the first time.  In the rehearsal the S7
# gates all passed and the analysis script itself then died; this stage is
# where that has to surface.
if [ "$INV_STEP" = dev ]; then
  # WHICH development pickles the analysis reads is a property of the PROFILE,
  # not a constant.  The pre-registered analysis holds absolute tick windows
  # (PRE_SURGE = 900..1400) and cannot run below 1400 ticks at all, so a
  # reduced-scale profile points this step at its own FULL-LENGTH development
  # subset instead of at its reduced dev set.  See REQUEST R11.
  PREFIXES=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --profile-field dev_analysis_prefixes)
  [ -n "$PREFIXES" ] || { echo "[stage_analysis] FAIL: profile $PROFILE names no development analysis input"; exit 1; }
  DEV=""; MISS=0; NPRE=0
  for pre in $(echo "$PREFIXES" | tr ',' ' '); do
    ST=$(echo "$pre" | sed -n 's/^\(s[0-9]*\)_.*/\1/p' | tr 'a-z' 'A-Z')
    collect_prefix "$ST" "$pre"
    [ -n "$PKLS" ] && DEV="${DEV:+$DEV,}$PKLS"
    MISS=$((MISS + NMISS))
    NPRE=$((NPRE + 1))
  done
  echo "[stage_analysis] development input: $NPRE prefix(es) [$PREFIXES] -> ${DEV:-none}"
  [ "$NPRE" -gt 0 ] || { echo "[stage_analysis] FAIL: zero prefixes examined"; exit 1; }
  [ -n "$DEV" ] || { echo "[stage_analysis] FAIL: no development pickle exists"; exit 1; }
  [ "$MISS" -eq 0 ] || { echo "[stage_analysis] FAIL: $MISS development invocation(s) produced no pickle"; exit 1; }
  "$PY" tools/stage_report.py --cross-provenance "$DEV" || exit 1
  echo "[stage_analysis] development-set dry run over: $DEV"
  # The seeds this step is checked against come from the PLAN (dev_full_seeds,
  # else dev_seeds), never from the pickles, so the completeness assert stays
  # meaningful.  The real profile gets the development manifests unchanged.
  DEVMAN=$("$PY" tools/analysis_manifest.py --profile "$PROFILE" --step dev \
      --out "${INV_ARTIFACT%.json}.manifest.json" \
      --manifests benchmark_manifest.json,benchmark_manifest_70_79.json) \
    || { echo "[stage_analysis] FAIL: no valid development manifest for profile $PROFILE"; exit 1; }
  echo "[stage_analysis] development manifest(s): $DEVMAN"
  nice -n 19 "$PY" confirmatory_analysis.py --pickles "$DEV" --manifests "$DEVMAN" \
      --marl-arm marl_static --out "$INV_ARTIFACT" > "${INV_ARTIFACT%.json}.txt" 2>&1
  RC=$?
  tail -30 "${INV_ARTIFACT%.json}.txt"
  if [ $RC -ne 0 ]; then
    echo "[stage_analysis] FAIL: the pre-registered analysis does not run on the"
    echo "                 development set (exit $RC).  Fix this BEFORE spending"
    echo "                 the confirmatory stage: at S7 the whole set is already"
    echo "                 committed and a second failure stops the chain."
    exit $RC
  fi
  "$PY" - "$INV_ARTIFACT" <<'PYX'
import json, sys
r = json.load(open(sys.argv[1]))
n = len(r.get('primary') or {})
seeds = {k: len(v) for k, v in (r.get('seeds') or {}).items()}
print('[stage_analysis] development report: %d primary test(s), instances %s' % (n, seeds))
if n == 0 or not seeds or min(seeds.values() or [0]) == 0:
    sys.exit('[stage_analysis] FAIL: the development report holds no usable result')
PYX
  exit $?
fi

if [ "$INV_STEP" = primary ]; then
  # WHICH confirmatory invocations the primary analysis reads is a property of
  # the PROFILE (cf_analysis_prefixes): the whole of S6 for a full-length
  # profile, the full-length subset of S6 for a reduced one -- the analysis
  # refuses a set below the nominal tick count (WP13; REQUEST R11 for S5b).
  PREFIXES=$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --profile-field cf_analysis_prefixes)
  [ -n "$PREFIXES" ] || { echo "[stage_analysis] FAIL: profile $PROFILE names no confirmatory analysis input"; exit 1; }
  CF=""; MISS=0; NPRE=0
  # the expected key set of exactly those invocations
  : > "$KEYDIR/S6.keys"
  for pre in $(echo "$PREFIXES" | tr ',' ' '); do
    collect_prefix S6 "$pre"
    [ -n "$PKLS" ] && CF="${CF:+$CF,}$PKLS"
    MISS=$((MISS + NMISS))
    NPRE=$((NPRE + 1))
    for id in $("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --ids S6); do
      case "$id" in $pre*) ;; *) continue ;; esac
      "$PY" confirmatory/chain_plan.py --profile "$PROFILE" --keys "$id" >> "$KEYDIR/S6.keys"
    done
  done
  PKLS="$CF"
  echo "[stage_analysis] confirmatory input: $NPRE prefix(es) [$PREFIXES] -> ${PKLS:-none}"
  [ "$NPRE" -gt 0 ] || { echo "[stage_analysis] FAIL: zero prefixes examined"; exit 1; }
  [ -n "$PKLS" ] || { echo "[stage_analysis] FAIL: no confirmatory pickle exists"; exit 1; }
  [ "$MISS" -eq 0 ] || { echo "[stage_analysis] FAIL: $MISS confirmatory invocation(s) produced no pickle"; exit 1; }
  [ -s "$KEYDIR/S6.keys" ] || { echo "[stage_analysis] FAIL: zero expected confirmatory keys"; exit 1; }
  "$PY" tools/stage_report.py --cross-provenance "$PKLS" || exit 1
  "$PY" tools/stage_report.py --key-union "$PKLS" --keys-file "$KEYDIR/S6.keys" || exit 1

  # As at S5b: the expected seeds come from the plan (cf_full_seeds, else
  # cf_seeds).  The real profile gets its sealed manifest unchanged.
  CFMAN=$("$PY" tools/analysis_manifest.py --profile "$PROFILE" --step primary \
      --out "${INV_ARTIFACT%.json}.manifest.json" --manifests "$MANIFEST") \
    || { echo "[stage_analysis] FAIL: no valid confirmatory manifest for profile $PROFILE"; exit 1; }
  echo "[stage_analysis] confirmatory manifest(s): $CFMAN"
  echo "[stage_analysis] confirmatory_analysis.py --marl-arm marl_static"
  nice -n 19 "$PY" confirmatory_analysis.py --pickles "$PKLS" --manifests "$CFMAN" \
      --marl-arm marl_static --out "$INV_ARTIFACT" \
      > "${INV_ARTIFACT%.json}.txt" 2>&1
  RC=$?
  tail -40 "${INV_ARTIFACT%.json}.txt"
  [ $RC -eq 0 ] || { echo "[stage_analysis] FAIL: confirmatory_analysis.py exit=$RC"; exit $RC; }
  "$PY" - "$INV_ARTIFACT" <<'PYX'
import json, sys
r = json.load(open(sys.argv[1]))
nprim = len(r.get('primary') or {})
nsec = len(r.get('secondary') or {})
seeds = {k: len(v) for k, v in (r.get('seeds') or {}).items()}
print('[stage_analysis] report: %d primary test(s), %d secondary, instances %s'
      % (nprim, nsec, seeds))
if nprim == 0 or not seeds or min(seeds.values() or [0]) == 0:
    sys.exit('[stage_analysis] FAIL: the report contains no usable primary result')
PYX
  exit $?
fi

if [ "$INV_STEP" = secondary ]; then
  # S12 of the chain: the pre-registered secondary analyses that need the
  # stages after S7 (PREREGISTRATION_R2 section 10), and the descriptive
  # report over S6h and S8-S11.
  #
  #   analysis S8   hash invariance: S6h (PYTHONHASHSEED=1) against its plan
  #                 pair in S6 (PYTHONHASHSEED=0), tools/hash_invariance_check.py
  #                 --require-decision, embedded with --s8;
  #   analysis S12  every untrained actor of chain S8 against the trained
  #                 policy, random and the best routing protocol:
  #                 confirmatory_analysis.py --untrained NAME=PKL,... ;
  #   secondary S7  marl_static against the adapt-then-freeze variant: the
  #                 chain-S9 pickle among --pickles.
  #
  # One further run of the pre-registered analysis computes all three, over
  # exactly the primary input of S7 plus those pickles, and must reproduce
  # every leaf of the chain's primary report.  Each analysis COUNTS its input
  # and fails when it is missing; a stage is excused only when the ledger
  # records that the operator SHED it (run_final.sh --shed).  untrained_
  # analysis.py is NOT called: it is the first study's script, hard-wired to
  # seeds 80-99 and the run_5/run_6 trees, and analysed nothing here (it was
  # recorded NOT RUN, exit 90).
  plan () { "$PY" confirmatory/chain_plan.py --profile "$PROFILE" "$@"; }
  LEDGER="confirmatory/chain_${PROFILE}.jsonl"
  BASE="${INV_ARTIFACT%secondary_report.json}"
  [ "$BASE" != "$INV_ARTIFACT" ] || { echo "[stage_analysis] FAIL: unexpected artifact name $INV_ARTIFACT"; exit 1; }
  was_shed () {           # $1 stage -> 0 when the ledger records the operator shedding it
    "$PY" - "$LEDGER" "$1" <<'PYX'
import json, sys
n = 0
try:
    for ln in open(sys.argv[1]):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get('event') == 'shed' and r.get('stage') == sys.argv[2]:
            n += 1
except OSError:
    pass
sys.exit(0 if n else 1)
PYX
  }
  FAILS=0
  fail () { echo "[stage_analysis] FAIL: $*"; FAILS=$((FAILS + 1)); }

  # ---- 1. the descriptive report: every invocation complete, per invocation
  ALL=""; ALL0=""; SUM=""
  for st in S6h S8 S9 S10 S11; do
    collect "$st"
    if [ -z "$PKLS" ]; then
      if was_shed "$st"; then SUM="$SUM $st=shed"; else SUM="$SUM $st=absent"; fi
      continue
    fi
    # PER INVOCATION, not per stage.  S8 runs the same (arm, seed, scenario)
    # keys once per untrained checkpoint, so a stage-level union gate reports
    # every key of the second invocation as a duplicate -- which is exactly
    # what it did in the rehearsal.  A repetition family is complete when EACH
    # invocation holds its whole key set; only a partitioned stage (S6) has a
    # union to check.  Each is checked against the hash seed its plan entry
    # declares (S6h: 1).
    nok=0; nbad=0
    for id in $(plan --ids "$st"); do
      art=$(plan --field "$id" artifact)
      case "$art" in *.pkl) ;; *) continue ;; esac
      hs=$(plan --field "$id" hashseed)
      plan --keys "$id" > "$KEYDIR/$id.keys"
      if "$PY" tools/stage_report.py --verify "$art" --keys-file "$KEYDIR/$id.keys" \
           --expect-hashseed "${hs:-0}"; then
        nok=$((nok + 1))
      else
        nbad=$((nbad + 1))
      fi
      [ "${hs:-0}" = 0 ] && [ -s "$art" ] && ALL0="$ALL0,$art"
    done
    if [ "$nok" -gt 0 ] && [ "$nbad" -eq 0 ]; then
      SUM="$SUM $st=complete($nok)"
    else
      SUM="$SUM $st=INCOMPLETE($nok-ok,$nbad-bad)"
    fi
    ALL="$ALL,$PKLS"
  done
  ALL="${ALL#,}"; ALL0="${ALL0#,}"
  [ -n "$ALL" ] || { echo "[stage_analysis] FAIL: every secondary stage is absent; nothing to describe"; exit 1; }
  collect S6
  # Everything run at PYTHONHASHSEED=0 shares one provenance signature with
  # S6; the hash-seed repeat differs in the seed BY DESIGN and is checked as
  # a pair below (same tree, schema and checkpoint; the declared seeds).
  "$PY" tools/stage_report.py --cross-provenance "${ALL0:+$ALL0,}$PKLS" || exit 1

  # ---- 2. analysis S8: the hash-invariance pair
  S8JSON="${BASE}s8_hash.json"; S8_STATE=missing; S8_RC=-
  H1=$(plan --analysis-role hash1)
  NH=$(echo $H1 | wc -w)
  if [ "$NH" != 1 ]; then
    fail "the plan declares $NH analysis-S8 repeat(s) (role hash1); expected exactly 1"
  else
    H0=$(plan --field "$H1" hash_pair_of)
    A1=$(plan --field "$H1" artifact); A0=$(plan --field "$H0" artifact)
    HSEEDS=$(plan --field "$H1" seeds | tr ' ' ',')
    echo "[stage_analysis] analysis S8 input: $A0 (PYTHONHASHSEED $(plan --field "$H0" hashseed)) vs $A1 (PYTHONHASHSEED $(plan --field "$H1" hashseed)), seeds [$HSEEDS], arm marl_static"
    if [ ! -s "$A0" ] || [ ! -s "$A1" ]; then
      fail "analysis S8 input missing ($A0 and/or $A1); the S6h stage is never shed"
    elif ! "$PY" tools/stage_report.py --hash-pair "$A0,$A1" \
         --expect-hashseeds "$(plan --field "$H0" hashseed),$(plan --field "$H1" hashseed)"; then
      fail "the analysis-S8 pair is not the declared pair (see [HASH-PAIR] above)"
    else
      nice -n 19 "$PY" tools/hash_invariance_check.py "$A0" "$A1" --arms marl_static \
          --seeds "$HSEEDS" --require-decision --quiet --out "$S8JSON" \
          > "${S8JSON%.json}.txt" 2>&1
      S8_RC=$?
      grep -E "^SUMMARY" "${S8JSON%.json}.txt" | sed 's/^/[stage_analysis] S8 /'
      case "$S8_RC" in
        0) S8_STATE=identical ;;
        # A difference IS the pre-registered analysis's finding (reported, as
        # the analysis prints it), not a pipeline failure.
        1) S8_STATE=NOT_IDENTICAL
           echo "[stage_analysis] S8 RESULT: NOT IDENTICAL -- recorded and reported as the analysis-S8 result" ;;
        *) fail "the S8 check verified nothing (exit $S8_RC: nothing compared or the hash-seed precondition not established)" ;;
      esac
      if [ "$S8_RC" = 0 ] || [ "$S8_RC" = 1 ]; then
        "$PY" - "$S8JSON" "$HSEEDS" <<'PYX' || { fail "the S8 result is vacuous"; S8_STATE=VACUOUS; }
import json, sys
r = json.load(open(sys.argv[1]))
want = sorted(int(s) for s in sys.argv[2].split(',') if s)
if len(want) != 2:
    sys.exit('[stage_analysis] FAIL: the plan declares %d seed(s) for the S8 pair, expected 2' % len(want))
ok = (r.get('n_compared', 0) > 0 and r.get('same_hashseed') is False
      and sorted(r.get('seeds') or []) == want and len(r.get('scenarios') or []) == 2
      and (r.get('decision_series') or {}).get('runs_with_key', 0) > 0)
print('[stage_analysis] S8 computed: %d series compared over seeds %s, %d scenarios, '
      '%d run(s) with the decision digest, same_hashseed=%r'
      % (r.get('n_compared', 0), r.get('seeds'), len(r.get('scenarios') or []),
         (r.get('decision_series') or {}).get('runs_with_key', 0), r.get('same_hashseed')))
sys.exit(0 if ok else 1)
PYX
      fi
    fi
  fi

  # ---- 3. the inputs of analysis S12 (untrained) and secondary S7 (freeze)
  roles_input () {        # $1 role -> R_ARG (NAME=PKL or PKL list), R_N, R_MISS, R_DECL
    local id art name
    R_ARG=""; R_N=0; R_MISS=0; R_DECL=0
    for id in $(plan --analysis-role "$1"); do
      R_DECL=$((R_DECL + 1))
      art=$(plan --field "$id" artifact)
      name=$(plan --field "$id" control_name)
      if [ -s "$art" ]; then
        R_ARG="${R_ARG:+$R_ARG,}${name:+$name=}$art"; R_N=$((R_N + 1))
      else
        R_MISS=$((R_MISS + 1))
      fi
    done
  }
  roles_input untrained
  UN="$R_ARG"; NU=$R_N; S12_STATE=computed
  echo "[stage_analysis] analysis S12 input: $R_N of $R_DECL untrained control(s) -> ${UN:-none}"
  if [ "$R_DECL" = 0 ]; then
    fail "the plan declares no untrained control for analysis S12"; S12_STATE=missing
  elif [ "$R_MISS" -gt 0 ]; then
    if [ "$R_N" = 0 ] && was_shed S8; then
      S12_STATE=shed; echo "[stage_analysis] analysis S12: chain S8 was SHED by the operator (ledger) -- recorded, not computed"
    else
      fail "$R_MISS untrained control pickle(s) missing for analysis S12"; S12_STATE=missing
    fi
  fi
  roles_input freeze
  FZ="$R_ARG"; NF=$R_N; S7S_STATE=computed
  echo "[stage_analysis] secondary S7 input: $R_N of $R_DECL adapt-then-freeze pickle(s) -> ${FZ:-none}"
  if [ "$R_DECL" = 0 ]; then
    fail "the plan declares no adapt-then-freeze pickle for secondary S7"; S7S_STATE=missing
  elif [ "$R_MISS" -gt 0 ]; then
    if [ "$R_N" = 0 ] && was_shed S9; then
      S7S_STATE=shed; echo "[stage_analysis] secondary S7: chain S9 was SHED by the operator (ledger) -- recorded, not computed"
    else
      fail "$R_MISS adapt-then-freeze pickle(s) missing for secondary S7"; S7S_STATE=missing
    fi
  fi

  # ---- 4. the full report: S7's primary input + the S9 arm + the controls + S8
  FULL="${BASE}cf_report_full.json"; FULL_STATE=not_run; CMP_STATE=not_run
  PREFIXES=$(plan --profile-field cf_analysis_prefixes)
  CF=""; MISS=0; NPRE=0
  for pre in $(echo "$PREFIXES" | tr ',' ' '); do
    collect_prefix S6 "$pre"
    [ -n "$PKLS" ] && CF="${CF:+$CF,}$PKLS"
    MISS=$((MISS + NMISS)); NPRE=$((NPRE + 1))
  done
  if [ "$NPRE" = 0 ] || [ -z "$CF" ] || [ "$MISS" -gt 0 ]; then
    fail "the primary analysis input is incomplete ($NPRE prefix(es), $MISS missing)"
  elif [ "$FAILS" -gt 0 ]; then
    echo "[stage_analysis] the full report is not run: $FAILS input failure(s) above"
  else
    CFMAN=$("$PY" tools/analysis_manifest.py --profile "$PROFILE" --step primary \
        --out "${FULL%.json}.manifest.json" --manifests "$MANIFEST") \
      || fail "no valid confirmatory manifest for profile $PROFILE"
    if [ "$FAILS" = 0 ]; then
      ARGS=(--pickles "$CF${FZ:+,$FZ}" --manifests "$CFMAN" --marl-arm marl_static
            --s8 "$S8JSON" --out "$FULL")
      [ -n "$UN" ] && ARGS+=(--untrained "$UN")
      echo "[stage_analysis] confirmatory_analysis.py --pickles $CF${FZ:+,$FZ} --manifests $CFMAN --marl-arm marl_static --s8 $S8JSON${UN:+ --untrained $UN}"
      rm -f "$FULL"
      nice -n 19 "$PY" confirmatory_analysis.py "${ARGS[@]}" > "${FULL%.json}.txt" 2>&1
      RC=$?
      if [ "$RC" != 0 ] || [ ! -s "$FULL" ]; then
        fail "confirmatory_analysis.py exit=$RC for the full report (${FULL%.json}.txt)"
        FULL_STATE=FAILED
      else
        # Presence, by counting -- never a value.
        if "$PY" - "$FULL" "$NU" "$NF" "$S12_STATE" "$S7S_STATE" <<'PYX'
import json, sys
rep = json.load(open(sys.argv[1]))
nu, nf, s12_state, s7_state = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
fails, n_checked = [], 0
# secondary S7
s7 = {k: v for k, v in (rep.get('secondary') or {}).items() if k.startswith('S7_')}
want = {'S7_%s_%s_marl_static_minus_marl_freeze' % (k, q)
        for k in ('A', 'B') for q in ('ach', 'components')}
if s7_state == 'computed':
    n_checked += 1
    miss = sorted(want - set(s7))
    n_inst = {k: (s7[k] or {}).get('n', 0) for k in s7}
    if nf == 0 or miss or not s7 or min(n_inst.values()) == 0:
        fails.append('secondary S7 not computed: %d/%d comparisons, missing %s, instances %s'
                     % (len(s7), len(want), miss, n_inst))
    print('[stage_analysis] secondary S7 computed: %d comparison(s), instances per comparison %s'
          % (len(s7), sorted(set(n_inst.values()))))
elif s7:
    fails.append('secondary S7 present although its input was %s' % s7_state)
# analysis S12
ctl = {k: v for k, v in (rep.get('S12') or {}).items() if k != 'mean_over_controls'}
if s12_state == 'computed':
    n_checked += 1
    per = {}
    for name, blk in ctl.items():
        per[name] = [((blk.get('summary') or {}).get(k) or {}).get('n', 0) for k in ('A', 'B')]
        if min(per[name]) == 0 or set((blk.get('paired') or {})) != {'A', 'B'}:
            fails.append('analysis S12 control %s has no comparison in both scenarios' % name)
    if nu == 0 or len(ctl) != nu:
        fails.append('analysis S12: %d control block(s) for %d control pickle(s)' % (len(ctl), nu))
    print('[stage_analysis] analysis S12 computed: %d untrained control(s), instances (A, B) %s'
          % (len(ctl), sorted({tuple(v) for v in per.values()})))
elif ctl:
    fails.append('analysis S12 present although its input was %s' % s12_state)
# analysis S8
s8 = rep.get('S8')
n_checked += 1
if not s8 or not s8.get('n_compared') or s8.get('same_hashseed') is not False:
    fails.append('analysis S8 not embedded with a non-vacuous comparison')
else:
    print('[stage_analysis] analysis S8 embedded: %d series compared, same_hashseed=%r'
          % (s8['n_compared'], s8['same_hashseed']))
if n_checked == 0:
    fails.append('nothing was checked')
for f in fails:
    print('[stage_analysis] FAIL: ' + f)
sys.exit(1 if fails else 0)
PYX
        then FULL_STATE=computed; else fail "the full report lacks an analysis it was given the input for"; FULL_STATE=INCOMPLETE; fi
        # Nothing the extra pickles add may move the chain's primary report.
        PRIM=$(plan --field s7_analysis artifact)
        if [ ! -s "$PRIM" ]; then
          fail "the chain's primary report $PRIM is missing"
        else
          "$PY" tools/compare_reports.py "$PRIM" "$FULL" \
              --sections seeds,excluded,descriptive,primary,secondary,stability,S10,S11 \
              > "${FULL%.json}.compare.txt" 2>&1
          CRC=$?
          grep -E "^SUMMARY" "${FULL%.json}.compare.txt" | sed 's/^/[stage_analysis] primary reproduced: /'
          if [ "$CRC" = 0 ]; then CMP_STATE=identical
          else CMP_STATE=DIFFERENT; fail "the full report does not reproduce every leaf of $PRIM (${FULL%.json}.compare.txt)"; fi
        fi
      fi
    fi
  fi

  "$PY" - "$INV_ARTIFACT" "$ALL" "$SUM" "$S8_STATE" "$S8_RC" "$S8JSON" "$S12_STATE" "$NU" \
        "$S7S_STATE" "$NF" "$FULL_STATE" "$FULL" "$CMP_STATE" "$FAILS" <<'PYX'
import json, pickle, sys, os
(out, pkls, summary, s8_state, s8_rc, s8_json, s12_state, nu, s7_state, nf,
 full_state, full, cmp_state, nfails) = sys.argv[1:15]
rep = {'stages': summary.split(), 'pickles': {},
       'analyses': {
           'S8_hash_invariance': {'state': s8_state, 'exit': s8_rc, 'json': s8_json},
           'S12_untrained': {'state': s12_state, 'controls': int(nu)},
           'S7_secondary_marl_freeze': {'state': s7_state, 'pickles': int(nf)},
           'full_report': {'state': full_state, 'path': full,
                           'primary_report_reproduced': cmp_state}},
       'untrained_analysis': ('not called: untrained_analysis.py is the first study\'s '
                              'script (seeds 80-99, run_5/run_6 trees); analysis S12 is '
                              'computed by confirmatory_analysis.py --untrained in the '
                              'full report'),
       'failures': int(nfails)}
n = 0
for p in pkls.split(','):
    if not p or not os.path.exists(p):
        continue
    P = pickle.load(open(p, 'rb'))
    keys = []
    for sk, sn in (('recovery_multi', 'recovery'), ('rescue_multi', 'rescue_ops')):
        for arm, per in (P.get(sk) or {}).items():
            keys += ['%s|%s|%s' % (arm, sn, s) for s, r in (per or {}).items() if r]
    rep['pickles'][p] = {'passes': len(keys),
                         'arms': sorted({k.split('|')[0] for k in keys}),
                         'provenance': {k: (P.get('provenance') or {}).get(k)
                                        for k in ('tree_head', 'tree_sha256',
                                                  'schema_id', 'pythonhashseed')}}
    n += len(keys)
rep['total_passes'] = n
json.dump(rep, open(out, 'w'), indent=1, sort_keys=True)
print('[stage_analysis] secondary report -> %s (%d passes over %d pickles; S8 %s, '
      'S12 %s, secondary S7 %s, full report %s, primary reproduced %s)'
      % (out, n, len(rep['pickles']), s8_state, s12_state, s7_state, full_state, cmp_state))
if n == 0:
    sys.exit('[stage_analysis] FAIL: the secondary family holds no passes')
PYX
  RC=$?
  [ "$FAILS" = 0 ] || { echo "[stage_analysis] FAIL: $FAILS secondary analysis failure(s)"; exit 1; }
  exit $RC
fi

echo "[stage_analysis] unknown step: $INV_STEP" >&2
exit 2
