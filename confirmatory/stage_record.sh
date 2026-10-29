#!/bin/bash
# R3-lite: the two stages that RECORD and SEAL instead of computing.
#
#   select   (chain S3b)  verify the PRE-WRITTEN confirmatory/SELECTED_<profile>.txt
#   freeze2  (chain S5)   seal confirmatory/FREEZE_FINAL_<profile>.txt
#
# WHY THERE IS NO SELECTION HERE.  In R2, S3b applied the selection rule to
# validation passes it had just run.  In R3-lite the policy under test was
# chosen BEFORE the pre-registration was written (R3 Phase 0: the paired
# achievability rule on development + validation data, disclosed in
# docs/PREREGISTRATION_R3.md section 3) and is an existing R2 file.  So S3b
# computes nothing: it checks that the tracked SELECTED file, the plan's pins
# (confirmatory/chain_plan.py) and the checkpoint on disk all name the SAME
# replicate with the SAME sha256, and writes what it checked.  Every later
# invocation resolves its checkpoint from line 1 of that SELECTED file, exactly
# as in R2 (run_final.sh), and is refused unless the digest is the pinned one.
#
# freeze2 is R2's S5 without the development gate (those stages do not exist
# here): it re-verifies the manifest, the imported files, the selection record,
# the analysis script and the engine pin, and seals the tree digest together
# with the manifest and checkpoint digests.  From then on every preflight
# refuses a tree whose digest differs (tools/preflight.py reads the
# 'tree_sha256' line), and the seed audit of freeze 2 runs before S6.
#
# Both steps COUNT what they verified and fail on zero.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
[ -n "$MODE" ] && [ -n "$PROFILE" ] && [ -n "$INV" ] || {
  echo "usage: $0 --print-command|--run PROFILE INV" >&2; exit 2; }
eval "$($PY "$TREE/confirmatory/chain_plan.py" --profile "$PROFILE" --emit "$INV")"
if [ "$MODE" = --print-command ]; then
  echo "bash confirmatory/stage_record.sh --run $PROFILE $INV"
  exit 0
fi
[ "$MODE" = --run ] || { echo "unknown mode $MODE" >&2; exit 2; }
cd "$TREE" || exit 1

case "$INV_STEP" in
  select)
    echo "[stage_record] selection RECORD for profile $PROFILE -> $INV_ARTIFACT"
    "$PY" - "$PROFILE" "$INV_ARTIFACT" <<'PYX'
import hashlib, json, os, re, sys, time
sys.path.insert(0, 'confirmatory')
import chain_plan

profile, out = sys.argv[1], sys.argv[2]
if chain_plan.graph_of(profile) != 'r3lite':
    sys.exit('[stage_record] FAIL: profile %s has no pre-registered selection' % profile)
path, seed, pin = chain_plan.selected_checkpoint(profile)
sel = chain_plan.selected_file(profile)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


checks = []


def chk(name, ok, detail=''):
    checks.append({'check': name, 'ok': bool(ok), 'detail': str(detail)})
    print('[stage_record] %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                                        ('  (%s)' % detail) if detail else ''))


lines = []
if os.path.isfile(sel):
    with open(sel) as f:
        lines = f.read().splitlines()
chk('the SELECTED file exists and is not empty', bool(lines), sel)
first = lines[0].strip() if lines else ''
chk('line 1 is the plan\'s checkpoint path', first == path, '%r vs %r' % (first, path))
seeds = [int(m.group(1)) for l in lines
         for m in [re.match(r'#\s*selected replicate:\s*torch seed\s+(\d+)\s*$', l)] if m]
chk('exactly one "selected replicate" line, naming the plan\'s replicate',
    seeds == [seed], '%s vs %d' % (seeds, seed))
shas = [m.group(1) for l in lines
        for m in [re.match(r'#\s*sha256\s+([0-9a-f]{64})\s*$', l)] if m]
chk('exactly one "sha256" line, equal to the plan\'s pin', shas == [pin],
    '%s vs %s' % ([s[:16] for s in shas], pin[:16]))
on_disk = sha(path) if os.path.isfile(path) else None
chk('the checkpoint exists', on_disk is not None, path)
chk('the checkpoint on disk has the pinned sha256', on_disk == pin,
    '%s vs %s' % ((on_disk or '-')[:16], pin[:16]))
# The selection is part of freeze 1: the SELECTED file must be a TRACKED file
# of the synced tree, i.e. inside the tree digest, not something written on
# the server afterwards.
tracked = None
if os.path.isfile('.synced_files'):
    with open('.synced_files') as f:
        tracked = sel in {l.strip() for l in f}
chk('the SELECTED file is a tracked file of the synced tree (.synced_files)',
    tracked is True, 'no .synced_files' if tracked is None else sel)

bad = [c for c in checks if not c['ok']]
if not checks or bad:
    sys.exit('[stage_record] FAIL: %d of %d selection check(s) failed; nothing '
             'is recorded' % (len(bad), len(checks)))
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
with open(out, 'w') as f:
    json.dump({'profile': profile,
               'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'computed': 'nothing: this stage verifies a pre-registered selection',
               'selected_file': sel, 'selected_file_sha256': sha(sel),
               'checkpoint': path, 'torch_seed': seed,
               'sha256_pinned': pin, 'sha256_on_disk': on_disk,
               'audit_trail': [l for l in lines[1:] if l.startswith('#')],
               'checks': checks, 'n_checks': len(checks), 'ok': True},
              f, indent=1, sort_keys=True)
    f.write('\n')
print('[stage_record] SELECTED replicate s%d -> %s (sha256 %s), %d check(s) -> %s'
      % (seed, path, pin[:16], len(checks), out))
PYX
    exit $?
    ;;

  freeze2)
    OUT="$INV_ARTIFACT"
    echo "[stage_record] FREEZE_FINAL (freeze 2) for profile $PROFILE -> $OUT"
    BODY="$(mktemp "${TMPDIR:-/tmp}/freeze2_${PROFILE}.XXXXXX")" || exit 1
    "$PY" - "$PROFILE" > "$BODY" <<'PYX'
import hashlib, json, os, sys
sys.path.insert(0, 'confirmatory')
sys.path.insert(0, '.')
import chain_plan

profile = sys.argv[1]
if chain_plan.graph_of(profile) != 'r3lite':
    sys.exit('[stage_record] FAIL: profile %s has no freeze2 step' % profile)
P = chain_plan.PROFILES[profile]
invs = {i['id']: i for i in chain_plan.build(profile)}


def digest(p, algo='sha256'):
    h = hashlib.new(algo)
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


fails, n = [], 0


def need(ok, what):
    global n
    n += 1
    if not ok:
        fails.append(what)
    sys.stderr.write('[stage_record] %-4s %s\n' % ('ok' if ok else 'FAIL', what))


# 1. the confirmatory manifest: exactly the plan's seeds, both scenario kinds
man = invs['s0_manifest']['artifact']
cf = sorted(P['cf_seeds'])
inst = []
if os.path.isfile(man):
    inst = json.load(open(man)).get('instances') or []
by_kind = {}
for i in inst:
    by_kind.setdefault(i.get('scenario_kind'), set()).add(int(i['topology_seed']))
need(len(inst) == 2 * len(cf) and len(by_kind) == 2
     and all(sorted(v) == cf for v in by_kind.values()),
     'manifest %s holds exactly the plan\'s %d seeds in both scenario kinds '
     '(%d instances)' % (man, len(cf), len(inst)))
# 2. every imported file still has its pinned digest
items = chain_plan.imports_of(profile)
for it in items:
    got = digest(it['dest']) if os.path.isfile(it['dest']) else None
    it['got'] = got
    need(got == it['sha256'], '%s (%s, torch seed %s) has the pinned sha256'
         % (it['dest'], it['role'], it['torch_seed']))
need(os.path.isfile(invs['s0_import']['artifact']),
     'the import record %s exists' % invs['s0_import']['artifact'])
# 3. the selection record, and line 1 of the SELECTED file
path, seed, pin = chain_plan.selected_checkpoint(profile)
sel = chain_plan.selected_file(profile)
rec = invs['s3b_record']['artifact']
rec_ok = False
if os.path.isfile(rec):
    r = json.load(open(rec))
    rec_ok = (r.get('ok') is True and r.get('checkpoint') == path
              and r.get('sha256_on_disk') == pin and r.get('n_checks', 0) > 0)
need(rec_ok, 'the selection record %s verified replicate %d with the pinned sha256' % (rec, seed))
first = open(sel).readline().strip() if os.path.isfile(sel) else ''
need(first == path, 'line 1 of %s is %s' % (sel, path))
# 4. the analysis script is the pre-registered one
md5 = digest('confirmatory_analysis.py', 'md5') if os.path.isfile('confirmatory_analysis.py') else None
need(md5 == chain_plan.R3_ANALYSIS_MD5,
     'confirmatory_analysis.py md5 %s == the pinned %s' % (md5, chain_plan.R3_ANALYSIS_MD5))
# 5. every engine/analysis file is byte-identical to R2's
n_eng, eng_problems = chain_plan.engine_pin_check('.')
need(n_eng > 0 and not eng_problems,
     '%d engine/analysis file(s) identical to the R2 tree %s (%d problem(s): %s)'
     % (n_eng, chain_plan.R3_ENGINE_HEAD, len(eng_problems), eng_problems[:3]))

if n == 0 or fails:
    sys.exit('[stage_record] FAIL: %d of %d freeze-2 check(s) failed; nothing is sealed'
             % (len(fails), n))

from sixg_sim import runinfo
p = runinfo.provenance()
if not p.get('tree_sha256'):
    sys.exit('[stage_record] FAIL: no tree digest; nothing is sealed')
for k in ('tree_head', 'tree_head_source', 'tree_sha256', 'tree_files',
          'schema_id', 'pythonhashseed', 'provenance_complete'):
    print('%-20s %s' % (k, p.get(k)))
print('%-20s %s  %s' % ('manifest', digest(man), man))
print('%-20s %s  %s' % ('selected_checkpoint', digest(path), path))
print('%-20s %s  %s' % ('selected_file', digest(sel), sel))
for it in items:
    print('%-20s %s  %s' % ('imported_' + it['role'], it['got'], it['dest']))
print('%-20s %s  confirmatory_analysis.py' % ('analysis_script_md5', md5))
print('%-20s %s (%d files identical: %s)' % ('engine_head', chain_plan.R3_ENGINE_HEAD,
                                             n_eng, chain_plan.ENGINE_PIN_FILE))
print('freeze-2 checks verified: %d' % n)
PYX
    RC=$?
    if [ $RC -ne 0 ] || [ ! -s "$BODY" ]; then
      rm -f "$BODY"
      echo "[stage_record] FREEZE_FINAL NOT sealed (exit $RC) - the chain stops here"
      exit 1
    fi
    {
      echo "FREEZE_FINAL  profile=$PROFILE  sealed $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "tree: $TREE"
      cat "$BODY"
    } > "$OUT"
    rm -f "$BODY"
    grep -q '^tree_sha256 ' "$OUT" || { echo "[stage_record] FAIL: $OUT carries no tree_sha256 line"; rm -f "$OUT"; exit 1; }
    echo "[stage_record] FREEZE_FINAL sealed -> $OUT"
    cat "$OUT"
    exit 0
    ;;

  *)
    echo "[stage_record] unknown step: $INV_STEP" >&2
    exit 2
    ;;
esac
