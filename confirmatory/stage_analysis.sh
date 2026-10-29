#!/bin/bash
# S7 (primary) and S12 (secondary) analysis steps; for the R3-lite graph also
# S7e (step family_e: the second confirmatory family on an assembled set).
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

# R3-lite: the analysis script must be THE pre-registered file (md5 pinned in
# confirmatory/chain_plan.py, docs/PREREGISTRATION_R3.md section 8) before
# any step of an R3-lite profile hands it a pickle.  The R2 profiles pin it
# through their own freeze record and return here at once, unchanged.
analysis_pin () {
  [ "$("$PY" confirmatory/chain_plan.py --profile "$PROFILE" --graph)" = r3lite ] || return 0
  "$PY" - <<'PYX'
import hashlib, sys
sys.path.insert(0, 'confirmatory')
import chain_plan
try:
    with open('confirmatory_analysis.py', 'rb') as f:
        got = hashlib.md5(f.read()).hexdigest()
except OSError as e:
    sys.exit('[stage_analysis] FAIL: confirmatory_analysis.py unreadable (%s)' % e)
if got != chain_plan.R3_ANALYSIS_MD5:
    sys.exit('[stage_analysis] FAIL: confirmatory_analysis.py md5 %s is not the '
             'pre-registered %s' % (got, chain_plan.R3_ANALYSIS_MD5))
print('[stage_analysis] confirmatory_analysis.py md5 %s == the pre-registered pin' % got)
# The analysis takes its t quantiles and its second Wilcoxon from scipy, and
# when the import fails it falls back WITHOUT SAYING SO: a table of t
# quantiles that ends at 19 degrees of freedom, and a null in the scipy
# field.  On forty instances that fallback is wrong.  The import is therefore
# a precondition of every R3-lite analysis step, checked with the
# interpreter that will run the analysis -- not a rule for whoever reads the
# report afterwards.
try:
    import scipy
    from scipy.stats import t as _t, wilcoxon as _w
    _q = float(_t.ppf(0.975, 39))
except Exception as e:
    sys.exit('[stage_analysis] FAIL: scipy.stats is not usable by %s (%s: %s); the '
             'pre-registered analysis is not run without it'
             % (sys.executable, type(e).__name__, e))
if not 2.0 < _q < 2.1:
    sys.exit('[stage_analysis] FAIL: scipy returns t(0.975, 39) = %r' % _q)
print('[stage_analysis] scipy %s usable (t, wilcoxon)' % scipy.__version__)
PYX
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
  analysis_pin || exit 1
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

if [ "$INV_STEP" = family_e ]; then
  # S7e of the R3-lite chain: THE SECOND CONFIRMATORY FAMILY (family E).
  #
  # The pre-registered analysis tests one learned arm, marl_static, against
  # the comparators of the same pickle set, and the file is md5-pinned.  For
  # family E it is therefore handed an ASSEMBLED set: every comparator pass
  # is the S6 pass, untouched, and every marl_static pass is the ensemble's
  # pass of the same instance (chain S9x).  tools/assemble_family.py builds
  # the set and verifies every run of it against its input; it refuses a
  # missing, doubled or foreign seed, a checkpoint list that is not the
  # pinned one, a provenance difference, and a pass that was not run on the
  # same instance (offered demand).
  #
  # ORDER, fixed here and not left to the operator:
  #   1. the family-1 report of S7 must exist; its sha256 is printed and
  #      recorded BEFORE anything of family E is computed;
  #   2. POSITIVE CONTROL of the assembly path: the set assembled from the
  #      S6 pickles alone (marl_static = replicate 1234) must give, through
  #      the same analysis, the S7 report leaf by leaf;
  #   3. family E is assembled and analysed, strict mode, same manifest.
  # Each of the two families is Holm-corrected within its own report; this
  # step never pools them, compares them or chooses between them.
  plan () { "$PY" confirmatory/chain_plan.py --profile "$PROFILE" "$@"; }
  efail () { echo "[stage_analysis] FAIL: $*"; exit 1; }
  analysis_pin || exit 1
  [ "$(plan --graph)" = r3lite ] || efail "step family_e belongs to the R3-lite stage graph only"
  BASE="${INV_ARTIFACT%cf_report_family_e.json}"
  [ "$BASE" != "$INV_ARTIFACT" ] || efail "unexpected artifact name $INV_ARTIFACT"
  SECTIONS=seeds,excluded,descriptive,primary,secondary,stability,S10,S11
  LEDGER="confirmatory/chain_${PROFILE}.jsonl"
  OUTPRE=$(plan --profile-field out_prefix)
  [ -n "$OUTPRE" ] || efail "profile $PROFILE has no out_prefix"
  CTLDIR="output/${OUTPRE}family_ctl"; FAMDIR="output/${OUTPRE}family_e"
  CTLREC="${BASE}family_ctl_assembly.json"; CTLREP="${BASE}cf_report_family_ctl.json"
  FAMREC="${BASE}family_e_assembly.json"; FAMOUT="${BASE}family_e_record.json"
  ORDJSON="${BASE}family_e_ledger_order.json"

  # NOTHING OF AN EARLIER RUN OF THIS STEP SURVIVES ITS START, and nothing
  # that a run which did not reach its last line computed survives its end.
  # The launcher removes only the artefact it gates on; the record, the
  # assembled sets and the text of the analysis are this step's, and a stale
  # record beside a new report (or a report without a record) is exactly
  # what S12 must never be handed.  A failed run keeps what explains the
  # failure -- the assembler's records, the control report and the two
  # comparisons -- and neither a family-E report, nor its text, nor its
  # record, nor an assembled pickle.
  # (The assembler builds each set in a private directory beside its target,
  # output/<prefix>.assembling_*, and removes it itself -- unless it is
  # killed outright.  Those leftovers are this step's too.)
  #
  # REMOVED MEANS GONE.  The run tree is on a network filesystem.  There a
  # file that another process still holds open is not removed at once: the
  # client leaves a placeholder in the directory, `rm -rf` ends "Directory
  # not empty", and the directory stays until that process lets go.  The
  # host's on-access scanner holds a file for a moment after it was written,
  # read or executed (measured 2026-10-12: the control set's directory was
  # left behind by 19 of 200 failed runs of this step, and by a failed run
  # in the suite; DECISIONS.md D-R3.6).  So a removal here is repeated until
  # the paths are gone, for up to a minute, and its caller is told if they
  # are not.
  rm_gone () {             # $@ paths -> 0 when none of them exists any more
    local i p
    for i in $(seq 1 60); do
      rm -rf "$@" 2>/dev/null
      for p in "$@"; do
        [ -e "$p" ] || [ -L "$p" ] || continue
        sleep 1; continue 2
      done
      return 0
    done
    rm -rf "$@"            # once more, aloud: the message says what is left
    for p in "$@"; do
      if [ -e "$p" ] || [ -L "$p" ]; then return 1; fi
    done
    return 0
  }
  rm -f "$INV_ARTIFACT" "${INV_ARTIFACT%.json}.txt" "${INV_ARTIFACT%.json}.err" \
        "${INV_ARTIFACT%.json}.manifest.json" \
        "$FAMOUT" "$FAMREC" "$CTLREC" "$CTLREP" "${CTLREP%.json}.txt" \
        "${CTLREP%.json}.compare.txt" "${CTLREP%.json}.compare_rev.txt" "$ORDJSON"
  rm_gone "$CTLDIR" "$FAMDIR" "output/${OUTPRE}".assembling_* \
    || efail "an assembled set of an earlier run of this step cannot be removed; nothing was computed"
  FAME_DONE=0
  fame_cleanup () {
    [ "$FAME_DONE" = 1 ] && return 0
    rm -f "$INV_ARTIFACT" "${INV_ARTIFACT%.json}.txt" "$FAMOUT"
    if rm_gone "$CTLDIR" "$FAMDIR" "output/${OUTPRE}".assembling_*; then
      echo "[stage_analysis] family E: the step did NOT complete -- no family-E report, text, record or assembled set is left"
    else
      echo "[stage_analysis] family E: the step did NOT complete -- no family-E report, text or record is left; AN ASSEMBLED SET COULD NOT BE REMOVED (above) and the next start of this step refuses to go on until it is"
    fi
  }
  trap fame_cleanup EXIT

  # ---- 1. family 1 first, and the passes of family E before it: both READ
  #         FROM THE LEDGER.  The family-1 report must be the file whose
  #         sha256 the ledger recorded when S7 ended, and every invocation of
  #         S9x must have ended before S7 was FIRST started (an attempt of S7
  #         that failed has still run).
  #         A set in which family E was run after family 1 had been analysed
  #         is refused here, whatever it contains.
  PRIM=$(plan --field s7_analysis artifact)
  [ -s "$PRIM" ] || efail "the family-1 report $PRIM does not exist; family E is never computed before it"
  PRIM_SHA=$(sha256sum "$PRIM" | cut -d' ' -f1)
  S9XIDS=$(plan --ids S9x | tr ' ' ',')
  [ -n "$S9XIDS" ] || efail "the plan of profile $PROFILE declares no S9x invocation"
  "$PY" tools/stage_report.py --ledger-order "$LEDGER" --first "$S9XIDS" --then s7_analysis \
      --expect-then-sha256 "$PRIM_SHA" --json-out "$ORDJSON" \
    || efail "the ledger does not show the passes of family E ($S9XIDS) complete before family 1 was analysed, or does not hold this family-1 report ($ORDJSON)"
  "$PY" tools/stage_report.py --ledger-append "$LEDGER" --kv event=note \
      reason=family_1_report_recorded "inv=$INV" "stage=$INV_STAGE" "report=$PRIM" \
      "report_sha256=$PRIM_SHA" \
    || efail "the family-1 report could not be recorded in the ledger"
  echo "[stage_analysis] family 1 report $PRIM sha256 $PRIM_SHA (recorded before family E is computed)"

  # ---- 2. the inputs: the comparators are S7's primary input, the arm is role 'ensemble'
  PREFIXES=$(plan --profile-field cf_analysis_prefixes)
  [ -n "$PREFIXES" ] || efail "profile $PROFILE names no confirmatory analysis input"
  CF=""; MISS=0; NPRE=0; SELPIN=""
  : > "$KEYDIR/S7e.keys"
  for pre in $(echo "$PREFIXES" | tr ',' ' '); do
    collect_prefix S6 "$pre"
    [ -n "$PKLS" ] && CF="${CF:+$CF,}$PKLS"
    MISS=$((MISS + NMISS)); NPRE=$((NPRE + 1))
    for id in $(plan --ids S6); do
      case "$id" in $pre*) ;; *) continue ;; esac
      plan --keys "$id" >> "$KEYDIR/S7e.keys"
      pin=$(plan --field "$id" ckpt_sha256 | tr ' ' ',')
      [ -n "$pin" ] || efail "the plan declares no checkpoint pin for $id"
      if [ -z "$SELPIN" ]; then SELPIN="$pin"
      elif [ "$SELPIN" != "$pin" ]; then efail "the S6 invocations do not share one checkpoint pin ($SELPIN, $pin)"; fi
    done
  done
  echo "[stage_analysis] family E comparators: $NPRE prefix(es) [$PREFIXES] -> ${CF:-none}"
  [ "$NPRE" -gt 0 ] || efail "zero prefixes examined"
  [ -n "$CF" ] || efail "no confirmatory pickle exists"
  [ "$MISS" -eq 0 ] || efail "$MISS confirmatory invocation(s) produced no pickle"
  [ -s "$KEYDIR/S7e.keys" ] || efail "zero expected confirmatory keys"
  EIDS=$(plan --analysis-role ensemble)
  NE=$(echo $EIDS | wc -w)
  [ "$NE" = 1 ] || efail "the plan declares $NE ensemble pickle(s) for the analysis (role ensemble); expected exactly 1"
  ENSP=$(plan --field "$EIDS" artifact)
  EPINS=$(plan --field "$EIDS" ckpt_sha256 | tr ' ' ',')
  [ -s "$ENSP" ] || efail "the ensemble pickle $ENSP is missing (chain S9x is never shed)"
  [ -n "$EPINS" ] || efail "the plan declares no checkpoint pin for the ensemble"
  [ "$EPINS" != "$SELPIN" ] || efail "the ensemble's pin equals the selected replicate's pin: not two families"
  echo "[stage_analysis] family E arm: $ENSP (pinned checkpoints $(echo "$EPINS" | tr ',' '\n' | cut -c1-12 | tr '\n' ' '))"
  plan --keys "$EIDS" > "$KEYDIR/$EIDS.keys"
  "$PY" tools/stage_report.py --verify "$ENSP" --keys-file "$KEYDIR/$EIDS.keys" \
      --expect-hashseed 0 --expect-checkpoint-sha256 "$EPINS" || exit 1
  "$PY" tools/stage_report.py --cross-provenance "$CF,$ENSP" || exit 1
  "$PY" tools/stage_report.py --key-union "$CF" --keys-file "$KEYDIR/S7e.keys" || exit 1
  CFMAN=$("$PY" tools/analysis_manifest.py --profile "$PROFILE" --step primary \
      --out "${INV_ARTIFACT%.json}.manifest.json" --manifests "$MANIFEST") \
    || efail "no valid confirmatory manifest for profile $PROFILE"
  derived () {            # $1 dir  $2 prefix -> the derived pickles of $CF, comma list
    local out="" p
    for p in $(echo "$CF" | tr ',' ' '); do out="${out:+$out,}$1/$2$(basename "$p")"; done
    echo "$out"
  }

  # ---- 3. POSITIVE CONTROL: the assembly path on family 1 itself
  nice -n 19 "$PY" tools/assemble_family.py --comparators "$CF" --arm-source "$CF" \
      --arm marl_static --expect-checkpoint-sha256 "$SELPIN" \
      --expect-keys "$KEYDIR/S7e.keys" \
      --out-dir "$CTLDIR" --prefix c_ --record "$CTLREC" \
    || efail "the control set could not be assembled ($CTLREC)"
  CTLP=$(derived "$CTLDIR" c_)
  "$PY" tools/stage_report.py --key-union "$CTLP" --keys-file "$KEYDIR/S7e.keys" || exit 1
  nice -n 19 "$PY" confirmatory_analysis.py --pickles "$CTLP" --manifests "$CFMAN" \
      --marl-arm marl_static --out "$CTLREP" > "${CTLREP%.json}.txt" 2>&1 \
    || { tail -20 "${CTLREP%.json}.txt"; efail "confirmatory_analysis.py failed on the control set"; }
  "$PY" tools/compare_reports.py "$PRIM" "$CTLREP" --sections "$SECTIONS" \
      > "${CTLREP%.json}.compare.txt" 2>&1
  CRC=$?
  grep -E "^SUMMARY" "${CTLREP%.json}.compare.txt" | sed 's/^/[stage_analysis] assembly control: /'
  [ "$CRC" = 0 ] || efail "the assembly path does not reproduce the family-1 report leaf by leaf (${CTLREP%.json}.compare.txt); family E is not computed"
  "$PY" tools/compare_reports.py "$CTLREP" "$PRIM" --sections "$SECTIONS" \
      > "${CTLREP%.json}.compare_rev.txt" 2>&1 \
    || efail "the family-1 report holds a leaf the control report lacks (${CTLREP%.json}.compare_rev.txt); family E is not computed"
  # BY COUNT, both ways.  The comparison skips a section its reference lacks
  # and then reports 'leaves=0 ... IDENTICAL' with exit 0: a comparison of
  # nothing.  The control counts only if leaves were compared, and as many
  # one way as the other.
  LEAVES_FWD=$(sed -n 's/^SUMMARY leaves=\([0-9][0-9]*\) mismatches=0 -> IDENTICAL$/\1/p' "${CTLREP%.json}.compare.txt" | tail -1)
  LEAVES_REV=$(sed -n 's/^SUMMARY leaves=\([0-9][0-9]*\) mismatches=0 -> IDENTICAL$/\1/p' "${CTLREP%.json}.compare_rev.txt" | tail -1)
  { [ -n "$LEAVES_FWD" ] && [ -n "$LEAVES_REV" ] && [ "$LEAVES_FWD" -gt 0 ] && [ "$LEAVES_FWD" = "$LEAVES_REV" ]; } \
    || efail "the assembly control compared ${LEAVES_FWD:-no} leaf(s) one way and ${LEAVES_REV:-no} the other: nothing was proved; family E is not computed"
  echo "[stage_analysis] assembly control: the set assembled from S6 alone reproduces $PRIM ($LEAVES_FWD leaves, both directions)"
  rm_gone "$CTLDIR" \
    || echo "[stage_analysis] WARNING: the control set $CTLDIR could not be removed within a minute; it is the control's own copy of family 1 and no result is read from it"

  # ---- 4. FAMILY E
  nice -n 19 "$PY" tools/assemble_family.py --comparators "$CF" --arm-source "$ENSP" \
      --arm marl_static --expect-checkpoint-sha256 "$EPINS" \
      --expect-keys "$KEYDIR/S7e.keys" \
      --out-dir "$FAMDIR" --prefix e_ --record "$FAMREC" \
    || efail "the family-E set could not be assembled ($FAMREC)"
  EP=$(derived "$FAMDIR" e_)
  "$PY" tools/stage_report.py --cross-provenance "$EP" || exit 1
  "$PY" tools/stage_report.py --key-union "$EP" --keys-file "$KEYDIR/S7e.keys" || exit 1
  echo "[stage_analysis] confirmatory_analysis.py --marl-arm marl_static (family E: $EP)"
  nice -n 19 "$PY" confirmatory_analysis.py --pickles "$EP" --manifests "$CFMAN" \
      --marl-arm marl_static --out "$INV_ARTIFACT" \
      > "${INV_ARTIFACT%.json}.txt" 2> "${INV_ARTIFACT%.json}.err"
  RC=$?
  # NOTHING OF THE TEXT of the family-E analysis reaches the job log unless
  # the step passes every check below.  The analysis writes its tables to
  # stdout and can still fail after printing them, so a failed run shows only
  # the last lines of its STDERR (the error), which holds no table; the text
  # is removed with everything else of family E.
  [ $RC -eq 0 ] || { tail -20 "${INV_ARTIFACT%.json}.err"; efail "confirmatory_analysis.py exit=$RC for family E (${INV_ARTIFACT%.json}.err)"; }
  [ "$(sha256sum "$PRIM" | cut -d' ' -f1)" = "$PRIM_SHA" ] || efail "the family-1 report changed while family E was computed"
  "$PY" - "$INV_ARTIFACT" "$PRIM" "$PRIM_SHA" "$FAMREC" "$CTLREC" "$CTLREP" "$FAMOUT" \
        "$ORDJSON" "$LEAVES_FWD" "$CF" "$ENSP" <<'PYX' || exit 1
import hashlib, json, sys
fam, prim, prim_sha, famrec, ctlrec, ctlrep, out, ordjson, leaves, cf, ensp = sys.argv[1:12]
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
r, p = json.load(open(fam)), json.load(open(prim))
ar, cr = json.load(open(famrec)), json.load(open(ctlrec))
od = json.load(open(ordjson))
fails = []
# the order, as the ledger holds it (written by --ledger-order above)
then, first = od.get('then_span') or {}, od.get('first_span') or []
if not (od.get('ok') is True and od.get('first_end_before_then_start') is True
        and first and then.get('sha256') == prim_sha
        and all(isinstance(f, dict) and f.get('end_record') is not None
                and f['end_record'] < then.get('first_start_record', -1) for f in first)):
    fails.append('the ledger-order record does not show every pass of family E complete '
                 'before the analysis of family 1 started')
# the assembly records name exactly the pickles this step was given
want_cmp, want_arm = [x for x in cf.split(',') if x], [x for x in ensp.split(',') if x]
for name, a, src in (('family E', ar, want_arm), ('control', cr, want_cmp)):
    if sorted(c.get('path') for c in a.get('comparators') or []) != sorted(want_cmp) \
            or sorted(c.get('path') for c in a.get('arm_source') or []) != sorted(src):
        fails.append('the %s assembly record does not name the pickles of this chain' % name)
nprim = len(r.get('primary') or {})
seeds = {k: list(v) for k, v in (r.get('seeds') or {}).items()}
if nprim == 0 or nprim != len(p.get('primary') or {}):
    fails.append('family E holds %d primary test(s), family 1 %d' % (nprim, len(p.get('primary') or {})))
if not seeds or min(len(v) for v in seeds.values()) == 0 or seeds != {k: list(v) for k, v in (p.get('seeds') or {}).items()}:
    fails.append('family E instances %s are not family 1 instances'
                 % {k: len(v) for k, v in seeds.items()})
if sorted(r.get('primary') or {}) != sorted(p.get('primary') or {}):
    fails.append('the two families do not hold the same hypotheses')
st = r.get('stability') or {}
if [k for k in ('S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f') if not isinstance(st.get(k), dict) or not st[k]]:
    fails.append('family E lacks a stability endpoint')
for name, a in (('family E', ar), ('control', cr)):
    v = a.get('verify') or {}
    if a.get('status') != 'PASS' or not v.get('comparator_runs') or not v.get('arm_runs') \
            or not v.get('offered_pairs') or v.get('arm_runs') != 2 * len(a.get('seeds') or []):
        fails.append('the %s assembly record is not a counted PASS' % name)
if ar.get('expect_checkpoint_sha256') == cr.get('expect_checkpoint_sha256'):
    fails.append('family E and the control were assembled for the same checkpoint list')
print('[stage_analysis] family E report: %d primary test(s), instances %s, assembled from %d '
      'comparator run(s) + %d ensemble run(s), %d offered-demand pair(s)'
      % (nprim, {k: len(v) for k, v in seeds.items()},
         (ar.get('verify') or {}).get('comparator_runs', 0), (ar.get('verify') or {}).get('arm_runs', 0),
         (ar.get('verify') or {}).get('offered_pairs', 0)))
for f in fails:
    print('[stage_analysis] FAIL: ' + f)
if fails:
    sys.exit(1)
json.dump({'family_1_report': prim, 'family_1_report_sha256': prim_sha,
           # READ FROM THE LEDGER, not asserted: where every S9x invocation
           # ended and where the successful attempt of S7 started
           'ledger_order': {'record': ordjson, 'record_sha256': sha(ordjson),
                            'family_e_passes': [{k: f.get(k) for k in ('inv', 'end_record', 'end_utc', 'sha256')}
                                                for f in first],
                            'family_1_analysis': {k: then.get(k) for k in
                                                  ('inv', 'first_start_record', 'first_start_utc',
                                                   'start_record', 'start_utc', 'end_utc', 'sha256')},
                            'family_e_passes_complete_before_family_1_analysis':
                                od.get('first_end_before_then_start')},
           'assembly_control': {'record': ctlrec, 'record_sha256': sha(ctlrec),
                                'report': ctlrep, 'report_sha256': sha(ctlrep),
                                'reproduces_family_1_report': 'identical',
                                'leaves_compared_each_direction': int(leaves)},
           'family_e_assembly': {'record': famrec, 'record_sha256': sha(famrec),
                                 'tool_sha256': ar.get('tool_sha256'),
                                 'expect_checkpoint_sha256': ar.get('expect_checkpoint_sha256'),
                                 'verify': ar.get('verify')},
           'family_e_report': fam, 'family_e_report_sha256': sha(fam),
           'note': 'each family is Holm-corrected within its own report; nothing is pooled'},
          open(out, 'w'), indent=1, sort_keys=True)
print('[stage_analysis] family E record -> %s' % out)
PYX
  tail -40 "${INV_ARTIFACT%.json}.txt"
  FAME_DONE=1
  exit 0
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
  analysis_pin || exit 1
  # WHICH stages this step describes is a property of the profile's stage
  # graph (chain_plan.secondary_stages): S6h S8 S9 S10 S11 for the R2
  # profiles, as this loop always read; S6h S8 S9x for R3-lite.  The R2
  # stages an R3-lite profile does not run are DROPPED BY THE PLAN, each with
  # a recorded reason, and are named as such here -- never met as a silent
  # absence.
  SECSTAGES=$(plan --secondary-stages)
  [ -n "$SECSTAGES" ] || { echo "[stage_analysis] FAIL: the plan of profile $PROFILE names no secondary stage"; exit 1; }
  # R3-LITE: IS FAMILY E 'NOT COMPLETED'?  (PREREGISTRATION_R3 section 13.)
  # The answer is READ FROM THE LEDGER and from nowhere else: family E is
  # not completed when, and only when, the restart rule is exhausted for an
  # invocation of S9x (its passes) or of S7e (its analysis) -- two failed
  # attempts.  Then this step still computes and reports every secondary
  # analysis of family 1, records family E as not_completed with the
  # ledger's cause, does NOT compute the per-instance listing of the
  # ensemble (it would be the only ensemble-against-XL-DET figure of the
  # run, outside the analysis that was registered for it), and opens no
  # ensemble pickle.  In every other state a missing ensemble pickle or
  # family-E report is a FAILURE of this step, as before.
  NC_CAUSE=""; NC_STAGE=""
  if [ "$(plan --graph)" = r3lite ]; then
    for st in S9x S7e; do
      if NCT=$("$PY" tools/stage_report.py --terminal-failure "$LEDGER" --profile "$PROFILE" --stage "$st"); then
        NC_STAGE="$st"; NC_CAUSE="$NCT"; break
      fi
    done
    [ -z "$NC_CAUSE" ] || echo "[stage_analysis] FAMILY E NOT COMPLETED (read from the ledger): $NC_CAUSE"
  fi
  DROPPED=$(plan --dropped)
  if [ -n "$DROPPED" ]; then
    echo "[stage_analysis] stages DROPPED by the plan of profile $PROFILE (recorded, not computed):"
    echo "$DROPPED" | sed 's/^/[stage_analysis]   /'
  fi

  # ---- 1. the descriptive report: every invocation complete, per invocation
  ALL=""; ALL0=""; SUM=""
  for st in $SECSTAGES; do
    if [ -n "$NC_CAUSE" ] && [ "$st" = S9x ]; then
      # not described and not opened, whichever stage failed: the label names
      # the stage at which family E stopped (S9x itself, or S7e with the
      # ensemble's passes complete)
      SUM="$SUM $st=not_described(family_E_not_completed_at_$NC_STAGE)"; continue
    fi
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
    # No adapt-then-freeze pickle is declared.  That is a FAILURE unless the
    # plan itself DROPPED chain S9 for this profile and says why (R3-lite).
    S9WHY=$(plan --dropped-reason S9)
    if [ -n "$S9WHY" ]; then
      S7S_STATE=dropped
      echo "[stage_analysis] secondary S7: chain S9 is DROPPED by the plan of profile $PROFILE (recorded, not computed) -- $S9WHY"
    else
      fail "the plan declares no adapt-then-freeze pickle for secondary S7"; S7S_STATE=missing
    fi
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

  # ---- 5. R3-lite only: the per-instance listing of the ensemble against
  #         XL-DET, and the stability families of both confirmatory reports.
  #
  # Declared by the plan (analysis role 'ensemble', chain S9x): marl_static
  # executing the four-replicate ensemble, listed against the XL-DET passes
  # of S7's primary input on the same instances.  The listing is DESCRIPTIVE
  # and decides nothing: every inferential statement about the ensemble comes
  # from the family-E report of S7e (the pre-registered analysis on the
  # assembled set), whose presence and record this step counts below.  An R2
  # profile declares no such role and skips this block entirely (ENS_STATE
  # stays 'not_declared' and nothing is added to its report).
  ENS_STATE=not_declared; ENSJSON="${BASE}ensemble_vs_xldet.json"; STAB_STATE=""
  EIDS=$(plan --analysis-role ensemble)
  if [ -n "$EIDS" ]; then
    ENSP=""; EMISS=0; EPINS=""; ESEEDS=""; ENS_STATE=computed
    NE=$(echo $EIDS | wc -w)
    for id in $EIDS; do
      art=$(plan --field "$id" artifact)
      if [ -s "$art" ]; then ENSP="${ENSP:+$ENSP,}$art"; else EMISS=$((EMISS + 1)); fi
      EPINS=$(plan --field "$id" ckpt_sha256 | tr ' ' ',')
      ESEEDS=$(plan --field "$id" seeds | tr ' ' ',')
    done
    echo "[stage_analysis] ensemble input: ${ENSP:-none} (seeds [$ESEEDS]) against xldet of ${CF:-none}"
    if [ -n "$NC_CAUSE" ]; then
      ENS_STATE=not_completed
      rm -f "$ENSJSON" "${ENSJSON%.json}.txt"
      echo "[stage_analysis] ensemble listing: NOT COMPUTED and not reported -- family E is not completed"
    elif [ "$NE" != 1 ]; then
      fail "the plan declares $NE ensemble pickle(s) for the analysis (role ensemble); expected exactly 1"; ENS_STATE=missing
    elif [ "$EMISS" -gt 0 ] || [ -z "$ENSP" ]; then
      fail "$EMISS ensemble pickle(s) missing (chain S9x is never shed)"; ENS_STATE=missing
    elif [ -z "$EPINS" ] || [ -z "$ESEEDS" ]; then
      fail "the plan declares no checkpoint pin or no seeds for the ensemble"; ENS_STATE=missing
    elif [ "$NPRE" = 0 ] || [ -z "$CF" ] || [ "$MISS" -gt 0 ]; then
      fail "the ensemble has no complete XL-DET reference (the primary input is incomplete)"; ENS_STATE=missing
    else
      rm -f "$ENSJSON"
      nice -n 19 "$PY" tools/stage_report.py --paired-descriptive "$ENSP" --baseline "$CF" \
          --paired-arm marl_static --paired-ref-arm xldet --paired-seeds "$ESEEDS" \
          --paired-label ensemble --expect-checkpoint-sha256 "$EPINS" \
          --json-out "$ENSJSON" > "${ENSJSON%.json}.txt" 2>&1
      ERC=$?
      grep -E "^\[PAIRED\]|^         " "${ENSJSON%.json}.txt" | sed 's/^/[stage_analysis] /'
      if [ "$ERC" != 0 ] || [ ! -s "$ENSJSON" ]; then
        fail "the ensemble comparison did not complete (exit $ERC, ${ENSJSON%.json}.txt)"; ENS_STATE=FAILED
      elif ! "$PY" - "$ENSJSON" "$ESEEDS" <<'PYX'
import json, sys
r = json.load(open(sys.argv[1]))
want = sorted(int(s) for s in sys.argv[2].split(',') if s)
sc = r.get('scenarios') or {}
ok = (r.get('ok') is True and r.get('descriptive') is True and want
      and r.get('n_pairs') == 2 * len(want) and sorted(sc) == ['A', 'B']
      and all(isinstance(sc[k].get('ach'), dict)
              and sc[k]['ach']['paired']['n'] == len(want)
              and sorted(int(s) for s in sc[k]['ach']['per_seed_difference']) == want
              for k in sc))
print('[stage_analysis] ensemble comparison computed: %s instance pair(s) over %d seed(s) x %d '
      'scenario(s), descriptive=%r' % (r.get('n_pairs'), len(want), len(sc), r.get('descriptive')))
sys.exit(0 if ok else 1)
PYX
      then
        fail "the ensemble comparison is vacuous or incomplete ($ENSJSON)"; ENS_STATE=VACUOUS
      fi
    fi
    # The stability family S9 is computed by the primary analysis from the
    # primary pickles (PREREGISTRATION_R3 section 9); S12 COUNTS that the
    # chain's primary report carries all six endpoints AND records what each
    # one is: its status ('assessable', 'not assessable', 'not recorded') and
    # whether it was confirmed, copied from the primary report and never
    # recomputed.  Presence alone is not "reported": an endpoint whose metric
    # is NOT RECORDED was not assessed and fails this stage (the metric
    # series are missing from the pickles: a pipeline defect).  'not
    # assessable' (fewer than six non-zero differences) is a legitimate
    # outcome of the family and is only recorded.
    PRIM=$(plan --field s7_analysis artifact)
    STABJSON="$KEYDIR/s12_stability_endpoints.json"; rm -f "$STABJSON"
    STAB_RC=1
    if [ -s "$PRIM" ]; then
      "$PY" - "$PRIM" "$STABJSON" <<'PYX'
import json, sys
st = (json.load(open(sys.argv[1])).get('stability') or {})
want = ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
known = ('assessable', 'not assessable', 'not recorded')
have = [k for k in want if isinstance(st.get(k), dict) and st[k]]
ep = {}
for k in have:
    r = st[k]
    ep[k] = {'status': r.get('status'), 'confirmed': r.get('confirmed'),
             'scenario': r.get('scenario'), 'metric': r.get('metric'),
             'expect': r.get('expect'), 'n': r.get('n'), 'n_nonzero': r.get('n_nonzero')}
bad = sorted(k for k in have if ep[k]['status'] not in known
             or not isinstance(ep[k]['confirmed'], bool))
notrec = sorted(k for k in have if ep[k]['status'] == 'not recorded')
n_ass = sum(1 for k in have if ep[k]['status'] == 'assessable')
print('[stage_analysis] stability family in the primary report: %d of %d endpoint(s) %s'
      % (len(have), len(want), have))
for k in have:
    print('[stage_analysis]   %s %s %s: status=%r confirmed=%r'
          % (k, ep[k]['scenario'], ep[k]['metric'], ep[k]['status'], ep[k]['confirmed']))
print('[stage_analysis] stability family: %d assessable, %d not assessable, %d not recorded, '
      '%d without a valid status' % (n_ass, sum(1 for k in have if ep[k]['status'] == 'not assessable'),
                                     len(notrec), len(bad)))
json.dump({'endpoints': ep, 'n_reported': len(have), 'n_assessable': n_ass,
           'not_recorded': notrec, 'malformed': bad}, open(sys.argv[2], 'w'),
          indent=1, sort_keys=True)
sys.exit(1 if have != want else 4 if bad else 3 if notrec else 0)
PYX
      STAB_RC=$?
    fi
    case "$STAB_RC" in
      0) STAB_STATE="reported(6)" ;;
      3) STAB_STATE=NOT_RECORDED
         fail "a stability endpoint of the chain's primary report is NOT RECORDED: it was not assessed ($PRIM, $STABJSON)" ;;
      4) STAB_STATE=MALFORMED
         fail "a stability endpoint of the chain's primary report carries no valid status or verdict ($PRIM, $STABJSON)" ;;
      *) STAB_STATE=MISSING; fail "the chain's primary report does not carry the six S9 endpoints ($PRIM)" ;;
    esac
    # FAMILY E (chain S7e): its report and its record must exist, the record
    # must name THIS primary report by sha256, and its six stability
    # endpoints are copied with status and verdict exactly as family 1's.
    FAME_IDS=$(plan --ids S7e)
    if [ -n "$FAME_IDS" ]; then
      FAME_REPORT=$(plan --field "$FAME_IDS" artifact)
      FAME_RECORD="${FAME_REPORT%cf_report_family_e.json}family_e_record.json"
      FAME_STAB="$KEYDIR/s12_stability_endpoints_family_e.json"; rm -f "$FAME_STAB"
      LEDART="$KEYDIR/s12_ledger_artifacts.json"; rm -f "$LEDART"
      FAME_STATE=computed; FAME_CAUSE=""; FAME_NC_STAGE=""; FAME_DEPOSIT=""
      if [ -n "$NC_CAUSE" ]; then
        FAME_STATE=not_completed; FAME_CAUSE="$NC_CAUSE"; FAME_NC_STAGE="$NC_STAGE"
        if [ -e "$FAME_REPORT" ] || [ -e "$FAME_RECORD" ]; then
          FAME_STATE=INCONSISTENT
          fail "the ledger records family E as not completed, but a family-E report or record exists ($FAME_REPORT, $FAME_RECORD)"
        fi
        # what exists of family E's passes is NAMED with its digest, for the
        # deposit section 13 requires -- hashed, never opened
        for id in $(plan --ids S9x); do
          art=$(plan --field "$id" artifact)
          [ -s "$art" ] && FAME_DEPOSIT="${FAME_DEPOSIT:+$FAME_DEPOSIT,}$art=$(sha256sum "$art" | cut -d' ' -f1)"
        done
      elif [ ! -s "$FAME_REPORT" ] || [ ! -s "$FAME_RECORD" ]; then
        FAME_STATE=missing; fail "the family-E report or its record is missing ($FAME_REPORT, $FAME_RECORD)"
      elif ! { "$PY" tools/stage_report.py --ledger-artifacts "$LEDGER" --profile "$PROFILE" \
                   --stage S6,S9x --json-out "$LEDART"
               "$PY" - "$FAME_REPORT" "$FAME_RECORD" "$PRIM" "$FAME_STAB" "$LEDART" "$CF" "$ENSP" \
                       "$ENS_STATE" "$ENSJSON" <<'PYX'
import hashlib, json, math, os, sys
fam, rec, prim, out, ledart, cf, ensp, ens_state, ensjson = sys.argv[1:10]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()
R, r = json.load(open(rec)), json.load(open(fam))
fails = []
if R.get('family_1_report_sha256') != sha(prim):
    fails.append('the family-E record names another family-1 report than %s' % prim)
if R.get('family_e_report_sha256') != sha(fam):
    fails.append('the family-E record names another family-E report than %s' % fam)
if (R.get('assembly_control') or {}).get('reproduces_family_1_report') != 'identical':
    fails.append('the family-E record carries no passed assembly control')
if (R.get('ledger_order') or {}).get('family_e_passes_complete_before_family_1_analysis') is not True:
    fails.append('the family-E record does not show the passes of family E complete before '
                 'family 1 was analysed')
# STALENESS.  Family 1 is recomputed by this step on the pickles of S6 as
# they are now; family E is not, so its report is bound to the pickles it
# was computed from: every comparator pickle of S6 and the ensemble pickle
# of S9x must TODAY have the sha256 (i) the assembler recorded when it read
# it and (ii) the ledger recorded when the invocation ended.
asm = R.get('family_e_assembly') or {}
ap = asm.get('record')
n_bound = 0
want = sorted([x for x in cf.split(',') if x] + [x for x in ensp.split(',') if x])
if not ap or not os.path.exists(ap) or sha(ap) != asm.get('record_sha256'):
    fails.append('the assembly record of family E is missing or is not the one its record names')
else:
    A = json.load(open(ap))
    led = json.load(open(ledart)) if os.path.exists(ledart) else {}
    used = (A.get('comparators') or []) + (A.get('arm_source') or [])
    if sorted(c.get('path') for c in used) != want:
        fails.append('family E was not assembled from the pickles of this chain (%d named, %d expected)'
                     % (len(used), len(want)))
    for c in used:
        p = c.get('path')
        now = sha(p) if p and os.path.exists(p) else None
        l = led.get(p) or {}
        if now is None or now != c.get('sha256'):
            fails.append('%s is not the pickle family E was assembled from' % p)
        elif l.get('status') != 'SUCCESS' or l.get('sha256') != now:
            fails.append('%s is not the pickle the ledger recorded for %s' % (p, l.get('inv')))
        else:
            n_bound += 1
    if n_bound == 0 or n_bound != len(want):
        fails.append('only %d of %d input pickle(s) of family E are bound to the ledger' % (n_bound, len(want)))
# CROSS-CHECK with the per-instance listing computed above from the same
# passes by another route: the achievability contrasts against XL-DET must
# agree in instances, wins and mean with P1 and P4 of the family-E report.
n_x = 0
if ens_state == 'computed' and os.path.exists(ensjson):
    L = (json.load(open(ensjson)).get('scenarios') or {})
    prim_e = r.get('primary') or {}
    for scen, hyp in (('A', 'P1'), ('B', 'P4')):
        keys = [k for k in prim_e if k.split('_')[0] == hyp]
        lp = (L.get(scen) or {}).get('ach')
        lp = lp.get('paired') if isinstance(lp, dict) else None
        rp = prim_e[keys[0]] if len(keys) == 1 else None
        if not lp or not isinstance(rp, dict) or 'mean' not in rp:
            fails.append('%s of the family-E report cannot be compared with the listing' % hyp)
        elif not (isinstance(lp.get('mean'), (int, float)) and isinstance(rp.get('mean'), (int, float))
                  and math.isfinite(lp['mean']) and math.isfinite(rp['mean'])):
            fails.append('%s: a mean that is not a finite number cannot agree with anything' % hyp)
        elif lp.get('n') != rp.get('n') or lp.get('wins') != rp.get('wins') or not lp.get('n') \
                or not abs(float(lp['mean']) - float(rp['mean'])) <= 1e-9:
            fails.append('the listing and the family-E report disagree on %s (instances, wins or mean)' % hyp)
        else:
            n_x += 1
    if n_x != 2:
        fails.append('the listing confirms %d of 2 achievability contrasts of the family-E report' % n_x)
else:
    fails.append('no computed listing to cross-check the family-E report against (%s)' % ens_state)
print('[stage_analysis] family E: %d input pickle(s) bound to the assembly record and the ledger; '
      'listing agrees on %d of 2 achievability contrast(s)' % (n_bound, n_x))
st = r.get('stability') or {}
want = ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
known = ('assessable', 'not assessable', 'not recorded')
have = [k for k in want if isinstance(st.get(k), dict) and st[k]]
ep = {k: {'status': st[k].get('status'), 'confirmed': st[k].get('confirmed'),
          'scenario': st[k].get('scenario'), 'metric': st[k].get('metric'),
          'expect': st[k].get('expect'), 'n': st[k].get('n'), 'n_nonzero': st[k].get('n_nonzero')}
      for k in have}
bad = sorted(k for k in have if ep[k]['status'] not in known or not isinstance(ep[k]['confirmed'], bool))
notrec = sorted(k for k in have if ep[k]['status'] == 'not recorded')
if have != want:
    fails.append('the family-E report does not carry the six S9 endpoints')
if bad or notrec:
    fails.append('family-E stability endpoints without a valid status %s, not recorded %s' % (bad, notrec))
n_ass = sum(1 for k in have if ep[k]['status'] == 'assessable')
print('[stage_analysis] family E: %d primary test(s); stability family %d assessable, %d not assessable'
      % (len(r.get('primary') or {}), n_ass, sum(1 for k in have if ep[k]['status'] == 'not assessable')))
for k in have:
    print('[stage_analysis]   E-%s %s %s: status=%r confirmed=%r'
          % (k, ep[k]['scenario'], ep[k]['metric'], ep[k]['status'], ep[k]['confirmed']))
json.dump({'endpoints': ep, 'n_reported': len(have), 'n_assessable': n_ass,
           'not_recorded': notrec, 'malformed': bad, 'n_primary': len(r.get('primary') or {})},
          open(out, 'w'), indent=1, sort_keys=True)
for f in fails:
    print('[stage_analysis] FAIL: ' + f)
sys.exit(1 if fails or not (r.get('primary') or {}) else 0)
PYX
      }
      then
        FAME_STATE=INVALID; fail "the family-E report is not the recorded, complete, current report of S7e ($FAME_REPORT)"
      fi
      export FAME_STATE FAME_REPORT FAME_RECORD FAME_STAB FAME_CAUSE FAME_NC_STAGE FAME_DEPOSIT
    fi
  fi

  "$PY" - "$INV_ARTIFACT" "$ALL" "$SUM" "$S8_STATE" "$S8_RC" "$S8JSON" "$S12_STATE" "$NU" \
        "$S7S_STATE" "$NF" "$FULL_STATE" "$FULL" "$CMP_STATE" "$FAILS" \
        "$ENS_STATE" "$ENSJSON" "$STAB_STATE" "$DROPPED" "${STABJSON:-}" <<'PYX'
import json, pickle, sys, os
(out, pkls, summary, s8_state, s8_rc, s8_json, s12_state, nu, s7_state, nf,
 full_state, full, cmp_state, nfails) = sys.argv[1:15]
ens_state, ens_json, stab_state, dropped, stab_json = sys.argv[15:20]
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
# R3-lite only (an R2 profile declares neither, and its report is unchanged):
# the stages its plan dropped, with the plan's reasons; the descriptive
# ensemble comparison; the count of the stability family in the primary
# report.
if dropped.strip():
    rep['dropped_by_plan'] = dict(l.split('\t', 1) for l in dropped.splitlines() if '\t' in l)
if ens_state != 'not_declared':
    rep['analyses']['ensemble_vs_xldet'] = {
        'state': ens_state, 'json': ens_json,
        'note': 'descriptive per-instance listing; outside every Holm family; the '
                'inferential statements about the ensemble are those of the family-E report'}
    if ens_state == 'not_completed':
        rep['analyses']['ensemble_vs_xldet'].update(
            json=None, note='NOT COMPUTED and not reported: family E is not completed '
                            '(PREREGISTRATION_R3 section 13)')
    # the second confirmatory family (chain S7e), counted and copied, never recomputed
    if os.environ.get('FAME_STATE'):
        fe = {'state': os.environ['FAME_STATE'], 'report': os.environ.get('FAME_REPORT'),
              'record': os.environ.get('FAME_RECORD'),
              'source': 'the family-E report (computed by S7e on the assembled set)'}
        if os.environ['FAME_STATE'] in ('not_completed', 'INCONSISTENT'):
            # READ FROM THE LEDGER: which stage exhausted the restart rule, and how
            fe.update(report=None, record=None,
                      not_completed_stage=os.environ.get('FAME_NC_STAGE'),
                      cause=os.environ.get('FAME_CAUSE'),
                      source='the chain ledger (tools/stage_report.py --terminal-failure)',
                      passes_on_disk=dict(x.split('=', 1) for x in
                                          (os.environ.get('FAME_DEPOSIT') or '').split(',') if '=' in x),
                      note='family E is NOT COMPLETED: nothing of it is computed or reported '
                           'by this run; family 1 is reported alone')
        fsj = os.environ.get('FAME_STAB')
        if fsj and os.path.exists(fsj):
            sj = json.load(open(fsj))
            fe.update(n_primary=sj.get('n_primary'), stability_endpoints=sj.get('endpoints') or {},
                      n_assessable=sj.get('n_assessable'), not_recorded=sj.get('not_recorded') or [])
        rep['analyses']['family_E'] = fe
    rep['analyses']['S9_stability_family'] = {
        'state': stab_state, 'source': 'the primary report (computed by S7 from the primary pickles)'}
    # what each endpoint IS in the primary report: status and verdict, copied
    if stab_json and os.path.exists(stab_json):
        sj = json.load(open(stab_json))
        rep['analyses']['S9_stability_family'].update(
            endpoints=sj.get('endpoints') or {}, n_reported=sj.get('n_reported'),
            n_assessable=sj.get('n_assessable'), not_recorded=sj.get('not_recorded') or [])
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
if ens_state != 'not_declared':
    print('[stage_analysis] R3-lite: ensemble vs XL-DET %s, family E %s, stability family %s, '
          '%d stage(s) dropped by the plan and recorded'
          % (ens_state, os.environ.get('FAME_STATE') or 'not_declared', stab_state,
             len(rep.get('dropped_by_plan') or {})))
if n == 0:
    sys.exit('[stage_analysis] FAIL: the secondary family holds no passes')
PYX
  RC=$?
  [ "$FAILS" = 0 ] || { echo "[stage_analysis] FAIL: $FAILS secondary analysis failure(s)"; exit 1; }
  exit $RC
fi

echo "[stage_analysis] unknown step: $INV_STEP" >&2
exit 2
