#!/bin/bash
# S0 preparation steps of the final-rerun chain.
#
#   manifest   build_benchmark_manifest.py over the confirmatory seeds [C]
#   untrained  make_untrained_checkpoints.py for every replicate seed [B]
#   import     (R3-lite only) copy the sha256-pinned R2 checkpoints out of the
#              frozen R2 tree; nothing is trained or generated
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
  import)
    # R3-lite S0: the policy under test, the other three ensemble members and
    # the untrained initial actor are EXISTING R2 files.  They are copied OUT
    # of the frozen R2 tree (read-only: nothing is ever written there) into
    # output/<profile>/, and a copy is accepted only with the sha256 the plan
    # pins (confirmatory/chain_plan.py R2_CHECKPOINT_SHA256 /
    # R2_UNTRAINED_SHA256).  Every source is verified BEFORE anything is
    # copied, so a wrong source leaves no file behind; an existing
    # destination with a different digest is never overwritten.  The step
    # counts what it verified and fails on zero.
    echo "[stage_prep] import of the pinned R2 checkpoints -> $INV_ARTIFACT"
    "$PY" - "$PROFILE" "$INV_ARTIFACT" <<'PYX'
import hashlib, json, os, shutil, sys, time
sys.path.insert(0, 'confirmatory')
import chain_plan

profile, out = sys.argv[1], sys.argv[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def fail(msg):
    sys.exit('[stage_prep] FAIL: ' + msg)


if chain_plan.graph_of(profile) != 'r3lite':
    fail('profile %s has no import step' % profile)
items = chain_plan.imports_of(profile)
if not items:
    fail('the plan declares nothing to import')
tree = os.path.realpath('.')
pre = 'output/' + chain_plan.PROFILES[profile]['out_prefix']
# pass 1: every source and every destination path, before any copy
for it in items:
    it['src_abs'] = os.path.realpath(os.path.expanduser(it['src']))
    dest_abs = os.path.realpath(it['dest'])
    if os.path.isabs(it['dest']) or not it['dest'].startswith(pre) \
            or not dest_abs.startswith(tree + os.sep):
        fail('destination %s is not under %s of this tree' % (it['dest'], pre))
    if it['src_abs'].startswith(tree + os.sep):
        fail('source %s lies inside this tree; the import reads the frozen R2 '
             'tree only' % it['src'])
    if not os.path.isfile(it['src_abs']):
        fail('source %s does not exist' % it['src'])
    it['src_sha256'] = sha(it['src_abs'])
    if it['src_sha256'] != it['sha256']:
        fail('source %s has sha256 %s, the plan pins %s (%s, torch seed %s)'
             % (it['src'], it['src_sha256'], it['sha256'], it['role'], it['torch_seed']))
    if os.path.exists(it['dest']) and sha(it['dest']) != it['sha256']:
        fail('%s exists with a different sha256 (%s; the plan pins %s).  It is '
             'NOT overwritten: find out what wrote it' % (it['dest'], sha(it['dest']), it['sha256']))
# pass 2: copy, and verify the copy
n = 0
for it in items:
    if os.path.exists(it['dest']):
        it['action'] = 'already present, verified'
    else:
        os.makedirs(os.path.dirname(it['dest']), exist_ok=True)
        tmp = it['dest'] + '.import_tmp'
        shutil.copyfile(it['src_abs'], tmp)
        if sha(tmp) != it['sha256']:
            os.remove(tmp)
            fail('the copy of %s does not have the pinned sha256' % it['src'])
        os.replace(tmp, it['dest'])
        it['action'] = 'copied'
    it['dest_sha256'] = sha(it['dest'])
    it['size'] = os.path.getsize(it['dest'])
    if it['dest_sha256'] != it['sha256']:
        fail('%s does not have the pinned sha256 after the import' % it['dest'])
    n += 1
    print('[stage_prep] %-16s s%-5s %s  %s  (%s)'
          % (it['role'], it['torch_seed'], it['dest_sha256'][:16], it['dest'], it['action']))
if n != len(items) or n == 0:
    fail('verified %d of %d imports' % (n, len(items)))
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
with open(out, 'w') as f:
    json.dump({'profile': profile, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'import_root': chain_plan.PROFILES[profile]['import_root'],
               'verified': n, 'items': items}, f, indent=1, sort_keys=True)
    f.write('\n')
print('[stage_prep] imported and verified %d file(s) -> %s' % (n, out))
PYX
    exit $?
    ;;
  *)
    echo "[stage_prep] unknown step: $INV_STEP" >&2
    exit 2
    ;;
esac
