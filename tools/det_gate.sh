#!/bin/bash
# RERUN_SPEC WP-DET.3 -- the determinism gate.
#
# A 2-batch training smoke (--episodes 32 --episode-ticks 200 --batch-size 16)
# must produce IDENTICAL [MAPPO] loss lines and an IDENTICAL POLICY for a
# given torch seed, at every worker count and when four replicates run
# concurrently.  Before WP-DET it did not: the training simulator's traffic
# model was entropy-seeded and the rollout pool was extended in
# worker-completion order.
#
# "IDENTICAL POLICY" is the sha256 of the WEIGHTS (tools/policy_digest.py),
# not of the checkpoint FILE.  The file also carries the provenance record
# WP0 requires -- process id, creation timestamp and argv -- which differs
# between two runs by construction.  Both digests are reported; the weights
# one is the criterion.
#
#   tools/det_gate.sh serial              # W=1, 8, 16 for one torch seed
#   tools/det_gate.sh concurrent          # four concurrent replicates
#   tools/det_gate.sh one TAG W SEED      # a single run (e.g. a second
#                                         # torch seed, to show the gate is
#                                         # not passing because every run
#                                         # collapsed to the same policy)
#   tools/det_gate.sh report              # compare whatever has been run
#
# `report` ends with an explicit VERDICT and EXIT STATUS (0 = PASS): within a
# torch seed every run must have EXITED ZERO and must carry a real weights
# digest, all equal, with identical [MAPPO] lines; at least one torch seed
# must have two runs to compare, or nothing was tested; across torch seeds
# the digests must DIFFER, or the runs collapsed onto one policy and
# "identical" would mean nothing.
#
# Everything lands in output/det_gate/.  Run from the tree root on
# <compute-server>.  Every run is nice -n 19 with PYTHONHASHSEED=0 and the
# evaluated engine/policy flags.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/output/det_gate"
mkdir -p "$OUT"

TORCH_SEED="${TORCH_SEED:-1234}"
EPISODES="${EPISODES:-32}"
TICKS="${TICKS:-200}"
BATCH="${BATCH:-16}"

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export MARL_TRANSPORT_POWER=planning
export MARL_RELAY_STATE=reconcile
export MARL_RELAY_REPOINT=fragment_aware
export MARL_POSTCARD_ALWAYS=1
export MARL_REWARD_PROFILE=mission
export MARL_RELAY_ENTROPY_FLOOR=0.01
export MARL_STEER_MODEL=actionable

run_one() {                       # run_one <tag> <workers> <torch_seed>
  local tag="$1" workers="$2" tseed="$3"
  local d="$OUT/$tag"
  rm -rf "$d" 2>/dev/null || true
  mkdir -p "$d/ck" "$d/out"
  # A previous run's artefacts must not survive into this one.  `rm -rf` has
  # been observed to fail here ("Directory not empty"), and a stale
  # policy_final.pt would then be digested as if this run had produced it --
  # a crashed run would report the previous run's PASS.  Remove the outputs
  # explicitly and refuse the run if any of them is still there.
  rm -f "$d/ck/policy_final.pt" "$d/policy.sha256" "$d/file.sha256" \
        "$d/mappo.txt" "$d/train.log" "$d/rc" 2>/dev/null || true
  if [ -e "$d/ck/policy_final.pt" ]; then
    echo "### $tag REFUSED: stale $d/ck/policy_final.pt could not be removed" >&2
    return 1
  fi
  echo "### $(date -u +%FT%TZ) $tag workers=$workers torch_seed=$tseed"
  MARL_POOL_WORKERS="$workers" nice -n 19 python3 train_on_comparison.py \
      --episodes "$EPISODES" --episode-ticks "$TICKS" --batch-size "$BATCH" \
      --torch-seed "$tseed" --eval-episodes 0 --baseline-episodes 0 \
      --checkpoint-dir "$d/ck" --output-dir "$d/out" \
      --save-policy-path "$d/ck/policy_final.pt" \
      > "$d/train.log" 2>&1
  local rc=$?
  # The run's OWN exit status, on disk.  `report`/`verdict` are normally run
  # later, in a separate process, and could not otherwise tell a clean run
  # from one that died AFTER writing policy_final.pt and its last [MAPPO]
  # line (the KPI save, the summary write): both leave a complete-looking
  # run directory, and the verdict would have called that a PASS.
  echo "$rc" > "$d/rc"
  grep -F '[MAPPO]' "$d/train.log" > "$d/mappo.txt"
  # THE CRITERION IS THE WEIGHTS DIGEST, so this is where it is computed.
  # `policy.sha256` holds tools/policy_digest.py --weights-only (sha256 over
  # every tensor of the state dict, in sorted key order); `file.sha256` holds
  # the file's own sha256, which differs between any two runs because of the
  # provenance record WP0 requires (pid, created_utc, argv, the run's own
  # output paths).  Writing the FILE digest into policy.sha256 -- which this
  # script used to do -- made the report label the file digest as the gate
  # criterion, i.e. made the gate fail on its own printed criterion every
  # time.  See docs/DECISIONS.md D-DET.9 and D-DET.11.
  if [ -f "$d/ck/policy_final.pt" ]; then
    sha256sum "$d/ck/policy_final.pt" | awk '{print $1}' > "$d/file.sha256"
    if ! python3 "$ROOT/tools/policy_digest.py" --weights-only \
            "$d/ck/policy_final.pt" > "$d/policy.sha256" 2> "$d/policy.err"; then
      echo "DIGEST_FAILED" > "$d/policy.sha256"
      sed 's/^/### policy_digest: /' "$d/policy.err" >&2
    fi
  else
    echo "MISSING" > "$d/policy.sha256"
    echo "MISSING" > "$d/file.sha256"
  fi
  echo "### $(date -u +%FT%TZ) $tag rc=$rc weights=$(cat "$d/policy.sha256") \
file=$(cat "$d/file.sha256") mappo_lines=$(wc -l < "$d/mappo.txt")"
  return $rc
}

report() {
  echo
  echo "=== WP-DET gate: [MAPPO] lines and the policy digest ==="
  printf '%-18s %-10s %-10s %-14s %s\n' TAG WEIGHTS8 FILE8 MAPPO EXIT
  for d in "$OUT"/*/; do
    tag="$(basename "$d")"
    [ -f "$d/policy.sha256" ] || continue
    printf '%-18s %-10s %-10s %-14s %s\n' "$tag" \
      "$(cut -c1-8 < "$d/policy.sha256")" \
      "$(cut -c1-8 < "$d/file.sha256" 2>/dev/null)" \
      "$(md5sum < "$d/mappo.txt" | cut -c1-12)" \
      "$(cat "$d/rc" 2>/dev/null || echo '?')"
  done
  echo
  echo "--- full weights sha256 (the gate criterion) ---"
  for d in "$OUT"/*/; do
    [ -f "$d/policy.sha256" ] || continue
    printf '%-18s %s  exit=%s\n' "$(basename "$d")" \
      "$(cat "$d/policy.sha256")" "$(cat "$d/rc" 2>/dev/null || echo '?')"
  done
  echo
  echo "--- file sha256 (differs by provenance: pid, timestamp, argv) ---"
  for d in "$OUT"/*/; do
    [ -f "$d/file.sha256" ] || continue
    printf '%-18s %s\n' "$(basename "$d")" "$(cat "$d/file.sha256")"
  done
  echo
  echo "--- pairwise [MAPPO] diffs against the first run (empty = identical) ---"
  local first="" tag=""
  for d in "$OUT"/*/; do
    [ -f "$d/mappo.txt" ] || continue
    if [ -z "$first" ]; then first="$d/mappo.txt"; tag="$(basename "$d")"; continue; fi
    echo "-- $tag vs $(basename "$d")"
    if diff "$first" "$d/mappo.txt"; then echo "   identical"; fi
  done
  echo
  echo "--- the loss lines themselves ($tag) ---"
  [ -n "$first" ] && cat "$first"
  verdict
}

# The gate's own PASS/FAIL, so it is a gate and not a table someone has to
# read correctly.  Runs are grouped by the torch seed in their tag
# (<name>_s<seed>).  Within a seed: every run must have exited ZERO and must
# carry a real weights digest, all of them equal, with identical [MAPPO]
# lines.  At least one seed must have TWO runs, or every line is a run
# agreeing with itself.  Across seeds: the digests must DIFFER -- if two
# torch seeds produced the same policy the runs collapsed onto one point and
# "identical" would be vacuous.
verdict() {
  local tmp rc=0 n=0 ncmp=0
  tmp="$(mktemp)" || return 1
  for d in "$OUT"/*/; do
    tag="$(basename "$d")"
    [ -f "$d/policy.sha256" ] || continue
    seed="$(printf '%s' "$tag" | sed -n 's/.*_s\([0-9][0-9]*\)$/\1/p')"
    [ -n "$seed" ] || seed="unknown"
    # field 5: the run's own exit status.  '?' means the run pre-dates the
    # `rc` file, so nothing is known about it -- reported, never silently
    # treated as success.
    printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "$tag" "$(cat "$d/policy.sha256")" \
      "$(md5sum < "$d/mappo.txt" 2>/dev/null | cut -c1-32)" \
      "$(cat "$d/rc" 2>/dev/null || echo '?')" >> "$tmp"
    n=$((n + 1))
  done
  echo
  echo "--- VERDICT (weights digest per torch seed) ---"
  if [ "$n" -eq 0 ]; then
    echo "FAIL: no runs to compare"
    rm -f "$tmp"; return 1
  fi
  for seed in $(cut -f1 "$tmp" | sort -u); do
    local ndig nmap bad crashed unknown nruns
    bad="$(awk -F'\t' -v s="$seed" '$1==s && ($3=="MISSING" || $3=="DIGEST_FAILED" || $3=="" || $4=="") {print $2}' "$tmp")"
    # A run whose training process exited non-zero is not evidence, however
    # complete its output directory looks.
    crashed="$(awk -F'\t' -v s="$seed" '$1==s && $5!="0" && $5!="?" {print $2"(exit="$5")"}' "$tmp")"
    unknown="$(awk -F'\t' -v s="$seed" '$1==s && $5=="?" {print $2}' "$tmp")"
    ndig=$(awk -F'\t' -v s="$seed" '$1==s {print $3}' "$tmp" | sort -u | wc -l)
    nmap=$(awk -F'\t' -v s="$seed" '$1==s {print $4}' "$tmp" | sort -u | wc -l)
    nruns=$(awk -F'\t' -v s="$seed" '$1==s' "$tmp" | wc -l)
    if [ -n "$bad" ]; then
      echo "FAIL seed=$seed: no usable digest for: $(echo $bad)"; rc=1
    elif [ -n "$crashed" ]; then
      echo "FAIL seed=$seed: training exited non-zero for: $(echo $crashed)"; rc=1
    elif [ "$ndig" -ne 1 ] || [ "$nmap" -ne 1 ]; then
      echo "FAIL seed=$seed: $ndig distinct weights digests, $nmap distinct [MAPPO] blocks"
      awk -F'\t' -v s="$seed" '$1==s {printf "      %-18s %s\n", $2, $3}' "$tmp"
      rc=1
    elif [ "$nruns" -lt 2 ]; then
      # One run agrees with itself.  Say so instead of printing a PASS that
      # was never a comparison (a second torch seed is run for the
      # non-collapse check below, and legitimately has one run).
      echo "NOTE seed=$seed: 1 run only -- nothing was compared within this \
seed ($(awk -F'\t' -v s="$seed" '$1==s {print $3; exit}' "$tmp"))"
    else
      ncmp=$((ncmp + 1))
      echo "PASS seed=$seed: $nruns runs, one policy \
($(awk -F'\t' -v s="$seed" '$1==s {print $3; exit}' "$tmp"))"
    fi
    if [ -n "$unknown" ]; then
      echo "      NOTE seed=$seed: exit status not recorded for: \
$(echo $unknown) (runs made before tools/det_gate.sh wrote \$d/rc)"
    fi
  done
  # WP-DET.3 is a statement about two runs agreeing.  If no torch seed ever
  # had two runs to compare, every line above is a tautology and the gate
  # must not report PASS.
  if [ "$ncmp" -eq 0 ]; then
    echo "FAIL: no torch seed has two runs to compare -- nothing was tested"
    rc=1
  fi
  # non-collapse: two torch seeds must not land on the same policy
  local dup
  dup="$(awk -F'\t' '$3 != "MISSING" && $3 != "DIGEST_FAILED" && $3 != "" {print $3"\t"$1}' "$tmp" \
         | sort -u | cut -f1 | sort | uniq -d)"
  if [ -n "$dup" ]; then
    echo "FAIL: two torch seeds share a policy digest ($dup): the gate is vacuous"
    rc=1
  elif [ "$(cut -f1 "$tmp" | sort -u | wc -l)" -lt 2 ]; then
    echo "NOTE: only one torch seed was run, so the non-collapse check did not run"
    echo "      (tools/det_gate.sh one <tag> <workers> <other-seed>)"
  else
    echo "PASS: distinct torch seeds give distinct policies (not collapsed)"
  fi
  rm -f "$tmp"
  [ $rc -eq 0 ] && echo "GATE: PASS" || echo "GATE: FAIL"
  return $rc
}

case "${1:-serial}" in
  serial)
    run_one "w01_s$TORCH_SEED"  1  "$TORCH_SEED"
    run_one "w08_s$TORCH_SEED"  8  "$TORCH_SEED"
    run_one "w16_s$TORCH_SEED" 16  "$TORCH_SEED"
    report
    ;;
  concurrent)
    # Four replicates at once.  Worker count per replicate is
    # CONC_WORKERS (default 2), so the four together stay inside the
    # 8-worker budget the campaign runs under.
    CW="${CONC_WORKERS:-2}"
    for i in 1 2 3 4; do
      run_one "conc${i}_s$TORCH_SEED" "$CW" "$TORCH_SEED" &
    done
    wait
    report
    ;;
  one)
    # tools/det_gate.sh one <tag> <workers> <torch_seed>
    run_one "${2:?tag}" "${3:?workers}" "${4:?torch_seed}"
    ;;
  report)
    report
    ;;
  *)
    echo "usage: $0 [serial|concurrent|one <tag> <workers> <seed>|report]" >&2
    exit 2;;
esac
