#!/bin/bash
# A stand-in stage script, usable ONLY under the 'test' profile (run_final.sh
# refuses CHAIN_SCRIPT_OVERRIDE for any other profile).  It lets the launcher
# tests exercise the restart rule, the resume rule and the counting gates
# without running any simulation.
#
# Behaviour per attempt comes from $CHAIN_FAKE_DIR/<inv>.script, one
# directive per line, consumed top-down:
#   ok      write a well-formed pickle holding exactly the expected passes
#   empty   write a well-formed pickle holding NO passes (the gate must fail)
#   nofile  exit 0 without writing anything (the gate must fail)
#   stale   write a COMPLETE, well-formed pickle but back-date its mtime by an
#           hour (the key count would pass; the freshness gate must fail)
#   fail    exit 9 without writing anything
# A missing file means "ok".
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/.." && pwd)"
PY=${PY:-python3}
MODE="${1:-}"; PROFILE="${2:-}"; INV="${3:-}"
if [ "$MODE" = --print-command ]; then
  echo "bash tests/fake_stage.sh --run $PROFILE $INV"
  exit 0
fi
cd "$TREE" || exit 1
"$PY" - "$PROFILE" "$INV" "${CHAIN_FAKE_DIR:-logs/fake}" <<'PYX'
import os, pickle, sys
sys.path.insert(0, 'confirmatory')
import chain_plan

profile, inv, fdir = sys.argv[1], sys.argv[2], sys.argv[3]
i = chain_plan.by_id(profile, inv)
script = os.path.join(fdir, inv + '.script')
directive = 'ok'
if os.path.exists(script):
    lines = [l.strip() for l in open(script) if l.strip()]
    if lines:
        directive = lines[0]
        open(script, 'w').write('\n'.join(lines[1:]) + ('\n' if lines[1:] else ''))
print('[fake_stage] %s directive=%s' % (inv, directive))
if directive == 'fail':
    sys.exit(9)
if directive == 'nofile':
    sys.exit(0)
art = i['artifact']
os.makedirs(os.path.dirname(art) or '.', exist_ok=True)
rec = {'recovery_multi': {}, 'rescue_multi': {}, 'seeds': i['seeds'],
       'provenance': {'tree_head': 'faketest', 'tree_sha256': 'f' * 64,
                      'schema_id': 'fake',
                      # what the job's environment really holds, recorded
                      # as runinfo.provenance does (FIX-PREFREEZE item 5):
                      # the hash seed and every MARL_* variable, plus the
                      # variable NAMES for the launcher tests to inspect
                      'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
                      'marl_env': {k: v for k, v in sorted(os.environ.items())
                                   if k.startswith('MARL_')},
                      'env_names': sorted(os.environ)},
       'pass_stats': {}}
if directive != 'empty':
    series = {'conn': [1.0] * 10, 'energy': [1.0] * 10, 'delivered': [1.0] * 10,
              'offered': [1.0] * 10, 'fragments': [1.0] * 10}
    for key in i['expected_keys']:
        arm, scen, seed = key.split('|')
        tgt = rec['recovery_multi'] if scen == 'recovery' else rec['rescue_multi']
        tgt.setdefault(arm, {})[int(seed)] = dict(series)
        rec['pass_stats'][key] = {'cpu_s': 1.0, 'wall_s': 1.0, 'pid': 0}
pickle.dump(rec, open(art, 'wb'))
if directive == 'stale':
    # Complete and well-formed, but written an hour ago as far as the file
    # system is concerned: the completeness gate alone would pass it.
    old = os.path.getmtime(art) - 3600
    os.utime(art, (old, old))
    print('[fake_stage] wrote %s and back-dated it by 3600 s' % art)
print('[fake_stage] wrote %s with %d passes' % (art, len(rec['pass_stats'])))
PYX
exit $?
