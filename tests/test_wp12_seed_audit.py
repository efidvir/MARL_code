"""WPDOC (pre-freeze diagnostics): tools/seed_audit.py on a synthetic tree.

Checks that the audit finds a seed of the window in every place the
pre-registration relies on (file names, seed lists in scripts, --seed-list
specs in logs, manifest topology_seed values, pickle seed keys), that it
loads a pickle holding classes the host cannot import, that planning
documents are bucketed separately, that tick/episode loops and the tests'
own scratch directory never make a tree look touched, and that a clean tree
yields a clean verdict.  Temporary files live under output/tmp_tests (server
rule: nothing outside the tree).
"""
import json
import os
import pickle
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import seed_audit as SA  # noqa: E402


def _mkdtemp(prefix):
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def _write(d, name, text):
    p = os.path.join(d, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    return p


GHOST_MODULE = 'ghost_module_the_audit_host_cannot_import'


def _dump_with_ghost(P, path, payload):
    """Pickle P plus an object of a class that lives in a temporary module.
    The module is removed after dumping, so a plain pickle.load fails and
    only the class-stubbing loader of the audit can read the file."""
    import types
    mod = types.ModuleType(GHOST_MODULE)

    class Ghost(object):
        pass

    Ghost.__module__ = GHOST_MODULE
    Ghost.__qualname__ = 'Ghost'
    mod.Ghost = Ghost
    sys.modules[GHOST_MODULE] = mod
    try:
        g = Ghost()
        g.payload = payload
        P = dict(P, ghost=g)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(P, f, protocol=4)
    finally:
        del sys.modules[GHOST_MODULE]


def _tree(d, touched):
    """A miniature run tree.  With touched=True, seed 210 appears in a script
    seed list, a log's --seed-list spec, a manifest, a pickle and a file name;
    with touched=False every seed is outside 200-219."""
    s = 210 if touched else 91
    _write(d, 'run_x.sh', 'python3 run_timeline_comparison.py --seed-list "80-89" --arms ospf\n'
                          'python3 build_benchmark_manifest.py --seeds 50-59\n'
                          'SEEDS="%d,82"\n' % s)
    _write(d, 'launch.py', 'seeds = [58, %d]\nfor s in range(42, 50):\n    pass\n'
                           'for s in range(%d, %d):\n    pass\n'
                           '# seed 42 with sev=200 and 200 island ticks, Episode 205\n'
                           % (s, s, s + 2))
    _write(d, 'logs/eval.log', '[OSPF - recovery] --- Starting Physics Pass: OSPF [recovery] seed=%d ---\n'
                               '  Worker Episode 205 [seed=134 kind=recovery sev=200]: avg_reward=1\n' % s)
    _write(d, 'docs/RERUN_SPEC.md', 'Fresh confirmatory seeds 200-219 (audited unused).\n')
    man = {'instances': [{'topology_seed': t, 'scenario_kind': 'recovery', 'attainable_optimum': 1}
                         for t in (80, 81, s)], 'created': 'x'}
    _write(d, 'benchmark_manifest_80_81.json', json.dumps(man))
    P = {'recovery_multi': {'ospf': {80: {'conn': [1.0]}, s: {'conn': [1.0]}}},
         'rescue_multi': {'ospf': {80: {'conn': [1.0]}, s: {'conn': [1.0]}}},
         'seeds': [80, s], 'provenance': {'seeds': [80, s], 'argv': ['x']}}
    _dump_with_ghost(P, os.path.join(d, 'output', 'cf_eval_80_81.pkl'), {50: 'x'})
    if touched:
        _write(d, 'output/smoke_seed210.log', 'nothing here\n')
    # an unrelated numeric token in a name must be reported but not as seed-like
    _write(d, 'notes_2024.txt', 'no seeds here\n')


def _audit(d, lo=200, hi=219):
    au = SA.Audit(lo, hi, [d], verbose=False)
    au.run()
    return au


def test_touched_tree_is_found_everywhere():
    d = _mkdtemp('wp12_sa_')
    try:
        _tree(d, touched=True)
        au = _audit(d)
        v = au.rep['verdict']
        assert not v['clean'], v
        assert v['window_hits_in_result_data'] == [210], v
        assert v['window_hits_in_file_names'] == [210], v
        kinds = {}
        for e in au.rep['text_hits']:
            kinds.setdefault(e['path'].split('/', 1)[1], set()).update(e['kinds'])
        assert 'kv' in kinds['run_x.sh'] and 'kv' in kinds['launch.py'], kinds
        assert 'kv' in kinds['logs/eval.log'], kinds
        # a seed loop that never says "seed" is still window-checked
        assert 'range()-nonseed' in kinds['launch.py'], kinds
        assert v['window_hits_in_text_non_planning'] == [210, 211], v
        # the planning document is bucketed apart and does not make the tree dirty
        plan = [e['path'] for e in au.rep['planning_hits']]
        assert plan and all('RERUN_SPEC' in p for p in plan), plan
        # manifest and pickle carry the seed
        js = [k for k, e in au.rep['json'].items() if e['hits'] == [210]]
        assert js and js[0].endswith('benchmark_manifest_80_81.json'), js
        pk = [k for k, e in au.rep['pickles'].items() if e['hits'] == [210]]
        assert pk and pk[0].endswith('cf_eval_80_81.pkl'), au.rep['pickles']
        e = au.rep['pickles'][pk[0]]
        assert e['error'] is None and e['seeds_key'] == [80, 210], e
        assert '80' in e['seed_values'] and '210' in e['seed_values'], e
        # a plain load cannot read the file (its class is gone) ...
        try:
            with open(os.path.join(d, 'output', 'cf_eval_80_81.pkl'), 'rb') as f:
                pickle.load(f)
        except Exception:
            pass
        else:
            raise AssertionError('plain pickle.load should fail on the ghost class')
        # ... but the stubbed load read it, including the ghost's own int-keyed payload
        assert 50 in au.seeds_data, au.seeds_data
        # benign tokens are classified, not reported as seed uses
        h = au.rep['broad_tokens']['histogram']
        assert h.get('sev=', 0) >= 2 and h.get('Episode N', 0) >= 1, h
        assert all(t['token'] not in (210,) for t in au.rep['broad_tokens']['other'])
        # the seed lists of the tree are enumerated (CLI spec, list, --seeds)
        assert set(range(80, 90)) <= au.seeds_text and set(range(50, 60)) <= au.seeds_text
        assert {58, 82, 134} <= au.seeds_text, au.seeds_text
        # range() on a line without "seed" is window-checked but not listed as a seed
        # (42 itself is named by the "seed 42" comment, so test 43-49)
        assert not (set(range(43, 50)) & au.seeds_text), au.seeds_text
        md = au.markdown()
        assert 'TOUCHED' in md and 'cf_eval_80_81.pkl' in md
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_clean_tree_is_clean():
    d = _mkdtemp('wp12_sa_')
    try:
        _tree(d, touched=False)
        au = _audit(d)
        v = au.rep['verdict']
        assert v['clean'], v
        assert au.rep['text_hits'] == [], au.rep['text_hits']
        assert au.rep['planning_hits'], 'planning mention must still be listed'
        assert not [k for k, e in au.rep['pickles'].items() if e['hits']]
        assert not [k for k, e in au.rep['json'].items() if e['hits']]
        assert au.rep['counts']['pickles_loaded'] == 1 and au.rep['counts']['pickles_failed'] == 0
        # the pickle seeds were read (80, 91) even though nothing is in the window
        e = list(au.rep['pickles'].values())[0]
        assert e['seeds_key'] == [80, 91], e
        assert 'CLEAN' in au.markdown()
        # a file name whose only token is not seed-like is listed but not counted
        assert au.rep['verdict']['window_hits_in_file_names'] == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_duplicate_pickles_are_loaded_once():
    d = _mkdtemp('wp12_sa_')
    try:
        os.makedirs(os.path.join(d, 'a'))
        os.makedirs(os.path.join(d, 'b'))
        P = {'recovery_multi': {'sdn': {205: {}}}, 'seeds': [205]}
        for sub in ('a', 'b'):
            with open(os.path.join(d, sub, 'same.pkl'), 'wb') as f:
                pickle.dump(P, f, protocol=4)
        au = _audit(d)
        c = au.rep['counts']
        assert c['pickles'] == 2 and c['pickles_loaded'] == 1 and c['pickles_duplicate'] == 1, c
        dup = [e for e in au.rep['pickles'].values() if e['duplicate_of']]
        assert len(dup) == 1 and dup[0]['hits'] == [205], dup
        assert au.rep['verdict']['window_hits_in_result_data'] == [205]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_false_positive_allowlist_and_sources():
    """A prose line "seed 42, 200 island ticks" names 200 next to "seed"; the
    scanner must report it, and an explicit --false-positive substring must
    move it to its own bucket so the verdict can be clean.  Text values keep
    their first source."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'sixg_sim/agent.py', '    comparison environment (seed 42, 200 island ticks, 35 agents) the\n')
        _write(d, 'run.sh', 'python3 x.py --seed-list 60-66\n')
        au = _audit(d)
        assert not au.rep['verdict']['clean']
        assert au.rep['verdict']['window_hits_in_text_non_planning'] == [200]
        au2 = SA.Audit(200, 219, [d], false_positives=['seed 42, 200 island ticks'])
        au2.run()
        v = au2.rep['verdict']
        assert v['clean'], v
        assert v['window_hits_classified_false_positive'] == [200], v
        assert len(au2.rep['false_positive_hits']) == 1 and au2.rep['text_hits'] == []
        # the false-positive line no longer feeds the "seeds referenced" list ...
        assert 200 not in au2.seeds_text and 42 not in au2.seeds_text, au2.seeds_text
        # ... while ordinary values keep their first source
        assert set(range(60, 67)) <= au2.seeds_text
        assert au2.rep['text_value_sources']['60'].startswith(os.path.basename(d) + '/run.sh:1')
        assert 'false positives' in au2.markdown()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_vanished_files_are_errors_not_crashes():
    """Another job's temporary files can disappear between the directory walk
    and the scan (the recorded run met exactly this); the audit must record an
    error and finish, and the rest of the tree must still be scanned."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'run.sh', 'python3 x.py --seed-list 205\n')
        with open(os.path.join(d, 'gone.pkl'), 'wb') as f:
            pickle.dump({'seeds': [1]}, f)
        au = SA.Audit(200, 219, [d])
        real_scan_name = au.scan_name

        def scan_name_then_delete(path):
            real_scan_name(path)
            if path.endswith('gone.pkl'):
                os.remove(path)
        au.scan_name = scan_name_then_delete
        au.run()
        assert au.rep['errors'] and 'gone.pkl' in au.rep['errors'][0], au.rep['errors']
        assert au.rep['counts']['pickles_loaded'] == 0
        assert au.rep['verdict']['window_hits_in_text_non_planning'] == [205]
        # a pickle path that never existed is an error too
        au2 = SA.Audit(200, 219, [d])
        au2.scan_pickle(os.path.join(d, 'never.pkl'))
        assert au2.rep['errors'] and 'never.pkl' in au2.rep['errors'][0]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_intkeyed_inventory():
    out = {'intkeyed': [{'path': '/recovery_multi/ospf', 'n': 2, 'keys': [82, 91]},
                        {'path': '/pass_stats', 'n': 1, 'keys': [0]}],
           'seedkeys': [{'path': '/seeds', 'value': [82, 91]}]}
    inv = SA._intkeyed_inventory(out)
    assert inv == ['/pass_stats: 0', '/recovery_multi/ospf: 82, 91', '/seeds: 82, 91'], inv


def test_value_parsing():
    f = SA._ints_from_value
    assert f('80-89') == set(range(80, 90))
    assert f('50,51,52,55') == {50, 51, 52, 55}
    assert f('42-49 and 100-139') == set(range(42, 50)) | set(range(100, 140))
    assert f('53 ---') == {53}
    assert f('200-219') == set(range(200, 220))
    assert SA._fmt_ranges([50, 51, 52, 53, 70, 72, 73]) == '50-53, 70, 72-73'
    assert SA._fmt_ranges([]) == '(none)'


def test_tick_loops_are_listed_not_counted():
    """`for tick in range(1, 361)` covers 200-219 but is a tick loop: it goes
    to the range_nonseed_benign bucket and never makes a tree TOUCHED.  A
    range() over a generic variable (`for s in range(205, 207)`) on a line
    without "seed" still counts, as before."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'test_x.py', 'for tick in range(1, 361):\n    pass\n'
                               'for t in range(0, 300):\n    pass\n'
                               'for episode in range(200, 210):\n    pass\n')
        au = _audit(d)
        v = au.rep['verdict']
        assert v['clean'], v
        assert au.rep['text_hits'] == [], au.rep['text_hits']
        rb = au.rep['range_nonseed_benign']
        assert [e['var'] for e in rb] == ['tick', 't', 'episode'], rb
        assert rb[0]['values'] == [200, 219] and rb[2]['values'] == [200, 209], rb
        md = au.markdown()
        assert 'listed, not counted' in md and '"tick": 1' in md, md
        # a generic loop variable is not benign
        _write(d, 'loop.py', 'for s in range(205, 207):\n    run(s)\n')
        au2 = _audit(d)
        assert not au2.rep['verdict']['clean']
        assert au2.rep['verdict']['window_hits_in_text_non_planning'] == [205, 206]
        assert au2.rep['text_hits'][0]['kinds'] == ['range()-nonseed']
        # a range() with no for-loop at all is window-checked
        _write(d, 'loop.py', 'seedish = list(range(210, 212))\n')
        au3 = _audit(d)
        assert au3.rep['verdict']['window_hits_in_text_non_planning'] == [210, 211]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_skip_dirs_default_and_option():
    """The unit tests' scratch directory (tmp_tests) is never walked, so a
    concurrent run of the touched-tree test cannot make a real tree look
    touched; --skip-dir extends the list and the report records it."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'output/tmp_tests/x/run.sh', 'python3 x.py --seed-list 210\n')
        _write(d, 'scratchy/run.sh', 'python3 x.py --seed-list 211\n')
        # one file outside every skipped directory: a root that walks NOTHING
        # is now a coverage failure of its own
        _write(d, 'keep.txt', 'no seed here\n')
        au = _audit(d)
        v = au.rep['verdict']
        assert v['window_hits_in_text_non_planning'] == [211], v
        assert 'tmp_tests' in au.rep['skip_dirs'], au.rep['skip_dirs']
        au2 = SA.Audit(200, 219, [d], skip_dirs=['scratchy'])
        au2.run()
        assert au2.rep['verdict']['clean'], au2.rep['verdict']
        assert 'scratchy' in au2.rep['skip_dirs'] and 'tmp_tests' in au2.rep['skip_dirs']
        assert 'scratchy' in au2.markdown()
        assert au2.rep['counts']['files'] == 1, au2.rep['counts']
        # CLI: --skip-dir reaches the report
        out = os.path.join(d, 'rep.json')
        rc = SA.main(['--lo', '200', '--hi', '219', '--roots', d, '--out', out,
                      '--skip-dir', 'scratchy'])
        assert rc == 0, rc
        with open(out) as f:
            rep = json.load(f)
        assert 'scratchy' in rep['skip_dirs'] and rep['verdict']['clean']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_planning_json_and_wp9_doc_are_planning():
    """The audit's own JSON report names the window; it must neither be a
    data hit nor enter the seeds-ever-used inventory.  docs/WP9_ANALYSIS.md
    (the analysis plan, which prints the future command line) is a planning
    document of this study."""
    d = _mkdtemp('wp12_sa_')
    try:
        # shaped like a real report: text_value_sources is keyed by seed strings
        _write(d, 'output/seed_audit_200_219.json',
               json.dumps({'window': [200, 219], 'all_seeds': {'x': '200-219'},
                           'text_value_sources': {'200': 'a.sh:1', '219': 'b.sh:2'}}))
        _write(d, 'benchmark_manifest_x.json',
               json.dumps({'instances': [{'topology_seed': 91}]}))
        _write(d, 'docs/WP9_ANALYSIS.md',
               '## 9. How to run on the confirmatory set (seeds 200-219)\n'
               'python3 confirmatory_analysis.py --pickles output/cf_200_203.pkl\n')
        au = _audit(d)
        v = au.rep['verdict']
        assert v['clean'], v
        assert v['window_hits_in_result_data'] == [], v
        assert 200 not in au.seeds_data and 219 not in au.seeds_data, au.seeds_data
        assert 91 in au.seeds_data
        plan = sorted(set(e['path'].split('/', 1)[1] for e in au.rep['planning_hits']))
        assert plan == ['docs/WP9_ANALYSIS.md'], plan
        assert au.rep['text_hits'] == []
        js = au.rep['json'][os.path.basename(d) + '/output/seed_audit_200_219.json']
        assert js['planning'] and js['hits'] == [200, 219], js
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# Coverage is half the verdict (review of 2026-09-23): a run that scanned
# nothing, lost a root, met an error or could not open a pickle, a checkpoint
# or a git history must not read CLEAN, because the freeze gate is exactly
# "the tool exits 0".
# --------------------------------------------------------------------------

def _run_repo(d, *args):
    """git in a temporary repository, with an identity of its own (the server
    has no global one) and no hooks."""
    import subprocess
    env = dict(os.environ, GIT_AUTHOR_NAME='audit test', GIT_AUTHOR_EMAIL='t@example.invalid',
               GIT_COMMITTER_NAME='audit test', GIT_COMMITTER_EMAIL='t@example.invalid')
    p = subprocess.run(['git', '-C', d] + list(args), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert p.returncode == 0, (args, p.stdout.decode('utf-8', 'replace')[:400])
    return p.stdout.decode('utf-8', 'replace')


def test_missing_root_refuses_the_verdict():
    """A mistyped or renamed root used to be appended to `errors` and then
    ignored: the run scanned nothing and still printed CLEAN with exit 0."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'run.sh', 'python3 x.py --seed-list 60-66\n')
        gone = os.path.join(d, 'no_such_tree_TYPO')
        au = SA.Audit(200, 219, [d, gone])
        au.run()
        v = au.rep['verdict']
        assert v['no_window_hits'], v          # nothing of the window anywhere
        assert not v['coverage_ok'], v
        assert not v['clean'], v               # ... and yet not CLEAN
        assert any('root not walked' in f and 'TYPO' in f for f in v['coverage_failures']), v
        md = au.markdown()
        assert 'INCOMPLETE' in md and 'coverage refused' in md, md
        # the CLI must refuse too, with its own exit code
        out = os.path.join(d, 'rep.json')
        rc = SA.main(['--lo', '200', '--hi', '219', '--roots', d, gone, '--out', out])
        assert rc == 4, rc
        with open(out) as f:
            rep = json.load(f)
        assert rep['verdict']['coverage_ok'] is False and rep['verdict']['clean'] is False
        # a run over the same tree without the bad root is clean again
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', d, '--out', out]) == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_failed_pickle_and_min_counts_refuse_the_verdict():
    d = _mkdtemp('wp12_sa_')
    try:
        with open(os.path.join(d, 'broken.pkl'), 'wb') as f:
            f.write(b'\x80\x04 not a pickle at all')
        au = _audit(d)
        v = au.rep['verdict']
        assert au.rep['counts']['pickles_failed'] == 1, au.rep['counts']
        assert v['no_window_hits'] and not v['clean'], v
        assert any('pickle' in f for f in v['coverage_failures']), v
        # coverage floors: a run that scanned less than the record is refused
        os.remove(os.path.join(d, 'broken.pkl'))
        _write(d, 'a.sh', '# nothing\n')
        au2 = SA.Audit(200, 219, [d], min_counts={'files': 100, 'pt': 5})
        au2.run()
        f2 = au2.rep['verdict']['coverage_failures']
        assert any('files 1 < 100' in x for x in f2), f2
        assert any('pt 0 < 5' in x for x in f2), f2
        assert not au2.rep['verdict']['clean']
        au3 = SA.Audit(200, 219, [d], min_counts={'files': 1})
        au3.run()
        assert au3.rep['verdict']['clean'], au3.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_home_roots_must_all_be_scanned():
    """Every tree beside the scanned ones must be a root: ~/MARL_run_9c was
    created after the freeze command's literal root list was written."""
    d = _mkdtemp('wp12_sa_')
    try:
        a = os.path.join(d, 'home', 'MARL_run_a')
        b = os.path.join(d, 'home', 'MARL_run_b')
        _write(a, 'x.sh', '# seeds 60-66\n')
        _write(b, 'y.sh', '# seeds 70-79\n')
        pat = os.path.join(d, 'home', 'MARL*')
        au = SA.Audit(200, 219, [a], home_glob=pat)
        au.run()
        v = au.rep['verdict']
        assert au.rep['home_marl_dirs'] == sorted([a, b]), au.rep['home_marl_dirs']
        assert au.rep['home_marl_unscanned'] == [b], au.rep['home_marl_unscanned']
        assert not v['clean'] and any('MARL_run_b' in f for f in v['coverage_failures']), v
        # both roots -> clean
        au2 = SA.Audit(200, 219, [a, b], home_glob=pat)
        au2.run()
        assert au2.rep['verdict']['clean'], au2.rep['verdict']
        # --no-home-check is the ad-hoc escape hatch
        au3 = SA.Audit(200, 219, [a], home_glob=pat, home_check=False)
        au3.run()
        assert au3.rep['verdict']['clean'], au3.rep['verdict']
        assert au3.rep['home_marl_unscanned'] == []
        # a tree that is not one of those directories does not arm the check
        au4 = SA.Audit(200, 219, [d], home_glob=pat)
        au4.run()
        assert au4.rep['home_marl_unscanned'] == [] and au4.rep['verdict']['clean']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_long_seed_list_is_not_truncated():
    """A value string used to be cut at 160 characters, so a seed of the
    window at the end of a long list was never extracted and the tree read
    CLEAN.  Reaching the (far larger) parse limit is now a hard error."""
    long_list = ', '.join(str(n) for n in range(100, 140)) + ', 205'
    assert len(long_list) > 160
    assert SA._ints_from_value(long_list) == set(range(100, 140)) | {205}
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'launch.py', 'SEEDS = [%s]\n' % long_list)
        au = _audit(d)
        v = au.rep['verdict']
        assert v['window_hits_in_text_non_planning'] == [205], v
        assert not v['clean'] and 205 in au.seeds_text and 139 in au.seeds_text
        assert not au.rep['broad_tokens']['other'], au.rep['broad_tokens']['other']
        # the parse limit itself is recorded, never silently applied
        huge = ', '.join(str(n) for n in range(10000, 10000 + SA.VALUE_CHARS // 5)) + ', 207'
        assert len(huge) > SA.VALUE_CHARS
        _write(d, 'huge.py', 'seeds = [%s]\n' % huge)
        au2 = _audit(d)
        assert au2.rep['long_values'] and au2.rep['long_values'][0]['chars'] >= SA.VALUE_CHARS
        assert not au2.rep['verdict']['coverage_ok'], au2.rep['verdict']
        assert any('parse limit' in f for f in au2.rep['verdict']['coverage_failures'])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_large_json_is_regex_scanned_for_lists_and_int_keys():
    """Files above --json-max are not parsed; the regex path must find a seed
    LIST (the normal shape), a STRING value, a NESTED list and an int-keyed
    table, not only a scalar.  Planting int and list values alone could not
    see the divergence this test now pins: the pattern accepted a list literal
    or a bare integer only, so a string-valued seed key went unmatched and a
    nested list stopped at its first `]`."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'big.json',
               json.dumps({'seeds': [200, 201, 202], 'torch_seed': 1234,
                           'provenance': {'seed_list': [205]},
                           'per_seed': {'207': 1.5},
                           'seed_range': '209-211',
                           'nested_seeds': [[212], [213]]}))
        want = [200, 201, 202, 205, 207, 209, 210, 211, 212, 213]
        key = os.path.join(os.path.basename(d), 'big.json')
        au = SA.Audit(200, 219, [d], json_max=100)
        au.run()
        e = au.rep['json'][key]
        assert au.rep['counts']['json_scanned'] == 1 and au.rep['counts']['json_parsed'] == 0
        assert e['scan'] == 'regex-fallback', e
        assert e['hits'] == want, e
        assert not au.rep['verdict']['clean'], au.rep['verdict']
        assert au.rep['verdict']['window_hits_in_result_data'] == want
        assert 1234 in au.seeds_data, au.seeds_data
        # parsed path (default limit) agrees -- exactly, not approximately
        au2 = _audit(d)
        assert au2.rep['counts']['json_parsed'] == 1
        assert au2.rep['json'][key]['scan'] == 'parsed'
        assert au2.rep['json'][key]['hits'] == e['hits'] == want
        assert au2.rep['verdict']['window_hits_in_result_data'] == \
            au.rep['verdict']['window_hits_in_result_data']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_checkpoint_metadata_is_scanned_with_pt():
    """--pt opened 2,563 checkpoints in the recorded run and had no test.  A
    seed planted in a checkpoint's metadata must be found, and an abort of
    the scan (torch unavailable) must refuse the verdict.

    The hit must reach `window_hits_in_result_data`, and the checkpoint must
    be the ONLY thing that can put it there.  Until 2026-09-23 `finish()`
    built the data-hit union from the pickle and JSON sections alone, so
    `pt[<file>]['hits'] == [208]` was written to the report and dropped on
    the way to the verdict: the run printed "CLEAN - no seed in 200-219 in
    any result, name, script, log or history" and exited 0.  This test's
    `assert not v['clean']` passed anyway, because the same directory held
    `adapted_seed209.pt`, whose FILE NAME produced a file-name hit -- delete
    that file and the assertion failed.  The 208 checkpoint therefore now
    sits in a directory of its own with no seed-named file in it."""
    try:
        import torch
    except Exception as e:
        raise AssertionError('torch is required for the --pt path of the audit: %r' % (e,))
    d = _mkdtemp('wp12_sa_')
    try:
        # no seed-named file anywhere under this root: nothing but the
        # checkpoint's metadata can produce a hit of the window
        os.makedirs(os.path.join(d, 'checkpoints'), exist_ok=True)
        torch.save({'metadata': {'topology_seed': 208, 'torch_seed': 1234},
                    'state': {'w': torch.zeros(2)}},
                   os.path.join(d, 'checkpoints', 'policy_final.pt'))
        torch.save({'metadata': {'topology_seed': 91}, 'state': {}},
                   os.path.join(d, 'checkpoints', 'other.pt'))
        au = SA.Audit(200, 219, [d], do_pt=True)
        au.run()
        v = au.rep['verdict']
        assert au.rep['counts']['pt'] == 2, au.rep['counts']
        assert au.rep['counts']['pt_failed'] == 0, au.rep['counts']
        pt = au.rep['pt'][os.path.basename(d) + '/checkpoints/policy_final.pt']
        assert pt['hits'] == [208], pt
        # the verdict, not only the report: a checkpoint is result data
        assert v['window_hits_in_result_data'] == [208], v
        assert v['window_hits_in_file_names'] == [], v
        assert v['window_hits_in_text_non_planning'] == [], v
        assert not v['no_window_hits'] and not v['clean'], v
        assert 'CLEAN' not in au.markdown().split('VERDICT:')[1].splitlines()[0]
        # the checkpoint's seeds also reach the inventory of section 4, as a
        # pickle's and a JSON's do (scan_pt fed seeds_data nowhere before)
        assert 208 in au.seeds_data and 1234 in au.seeds_data, au.seeds_data
        assert 91 in au.seeds_data, au.seeds_data
        assert 'policy_final.pt' in au.markdown()
        # without --pt the checkpoint is invisible, and NOTHING else in this
        # tree names the window -- so the tree reads clean
        au2 = _audit(d)
        assert au2.rep['counts']['pt'] == 0 and au2.rep['pt'] == {}
        assert au2.rep['verdict']['window_hits_in_result_data'] == []
        assert au2.rep['verdict']['clean'], au2.rep['verdict']
        # a per-agent adapted file carries its seed in the NAME (step 1) and
        # is not opened; kept in a separate root so it cannot stand in for
        # the metadata hit above
        d2 = _mkdtemp('wp12_sa_')
        try:
            torch.save({'metadata': {'topology_seed': 209}},
                       os.path.join(d2, 'adapted_seed209.pt'))
            au4 = SA.Audit(200, 219, [d2], do_pt=True)
            au4.run()
            assert au4.rep['counts']['pt'] == 0, au4.rep['counts']   # skipped by name
            assert au4.rep['verdict']['window_hits_in_file_names'] == [209]
            assert au4.rep['verdict']['window_hits_in_result_data'] == []
        finally:
            shutil.rmtree(d2, ignore_errors=True)
        # an aborted --pt scan (no torch) is a coverage failure, not silence
        au3 = SA.Audit(200, 219, [d], do_pt=True)
        real = SA.Audit.scan_pt

        def abort(self, path):
            self.rep['errors'].append('torch unavailable (test); --pt aborted')
            self.do_pt = False
            self.pt_aborted = True
        SA.Audit.scan_pt = abort
        try:
            au3.run()
        finally:
            SA.Audit.scan_pt = real
        v3 = au3.rep['verdict']
        assert au3.rep['counts']['pt_skipped'] >= 1, au3.rep['counts']
        assert not v3['coverage_ok'] and not v3['clean'], v3
        assert any('--pt aborted' in f for f in v3['coverage_failures']), v3
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_git_history_is_scanned_and_a_broken_repository_is_an_error():
    """--git was a silent no-op: stderr went to /dev/null and the return code
    was never checked, so an empty .git directory (~/MARL_code) produced "no
    hits" and the evidence "no seed was ever committed" rested on nothing.
    Hits must also accumulate over several roots, not only the last one."""
    d = _mkdtemp('wp12_sa_')
    try:
        ra, rb = os.path.join(d, 'repoA'), os.path.join(d, 'repoB')
        for r, seed in ((ra, 205), (rb, 206)):
            os.makedirs(r, exist_ok=True)
            _run_repo(r, 'init', '-q')
            _write(r, 'old_launch.sh', 'python3 run.py --seed-list %d\n' % seed)
            _run_repo(r, 'add', 'old_launch.sh')
            _run_repo(r, 'commit', '-qm', 'launcher')
            os.remove(os.path.join(r, 'old_launch.sh'))
            _run_repo(r, 'rm', '-q', '--cached', 'old_launch.sh')
            _run_repo(r, 'commit', '-qm', 'drop launcher')
        au = SA.Audit(200, 219, [ra, rb], do_git=True)
        au.run()
        v = au.rep['verdict']
        # the working trees are empty of the seed ...
        assert v['window_hits_in_file_names'] == [] and v['window_hits_in_result_data'] == []
        # ... but the history carries it, for BOTH roots
        assert v['window_hits_in_git_history'] == [205, 206], v
        assert sorted(set(e['path'] for e in au.rep['git_hits'])) ==             ['git:repoA:old_launch.sh', 'git:repoB:old_launch.sh']
        assert not v['clean'] and au.rep['counts']['git_roots_scanned'] == 2
        assert all(e['scanned'] and e['lines'] > 0 for e in au.rep['git_roots'])
        # a root with an empty .git directory is a hard error, not a clean pass
        rc = os.path.join(d, 'repoC')
        os.makedirs(os.path.join(rc, '.git'), exist_ok=True)
        _write(rc, 'note.txt', 'nothing here\n')
        au2 = SA.Audit(200, 219, [rc], do_git=True)
        au2.run()
        v2 = au2.rep['verdict']
        assert v2['no_window_hits'] and not v2['clean'], v2
        assert au2.rep['errors'] and 'git log failed' in au2.rep['errors'][0], au2.rep['errors']
        assert not v2['coverage_ok'], v2
        # --git over a root that is not a repository at all: nothing scanned
        au3 = SA.Audit(200, 219, [rc + '_plain'], do_git=True)
        os.makedirs(rc + '_plain', exist_ok=True)
        _write(rc + '_plain', 'n.txt', 'x\n')
        au3.run()
        assert au3.rep['git_roots'] and au3.rep['git_roots'][0]['reason'] == 'no .git entry'
        assert any('no root produced a git history' in f
                   for f in au3.rep['verdict']['coverage_failures']), au3.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_git_log_dump_is_scanned():
    """The trackB worktree lives on the laptop and every scan runs on the
    server: --git-log scans a captured `git log --all -p` dump and records
    its sha256, so the history claim is reproducible."""
    d = _mkdtemp('wp12_sa_')
    try:
        dump = os.path.join(d, 'trackB_git_log.txt')
        with open(dump, 'w', encoding='utf-8') as f:
            f.write('commit 1111111111111111111111111111111111111111\n'
                    '    WP0: launcher\n'
                    'diff --git a/launch.sh b/launch.sh\n'
                    '+python3 run.py --seed-list 212\n')
        tree = os.path.join(d, 'tree')      # the dump is outside the root
        _write(tree, 'clean.sh', '# seeds 60-66\n')
        au = SA.Audit(200, 219, [tree], git_logs=[dump])
        au.run()
        v = au.rep['verdict']
        assert v['window_hits_in_git_history'] == [212], v
        assert not v['clean'], v
        e = au.rep['git_logs'][0]
        assert e['scanned'] and e['commits'] == 1 and e['lines'] == 4 and len(e['sha256']) == 64
        assert 'trackB_git_log.txt' in au.markdown()
        # an empty or missing dump is a hard error
        empty = os.path.join(d, 'empty.txt')
        open(empty, 'w').close()
        au2 = SA.Audit(200, 219, [tree], git_logs=[empty, os.path.join(d, 'nope.txt')])
        au2.run()
        v2 = au2.rep['verdict']
        assert not v2['coverage_ok'], v2
        assert any('empty dump' in e for e in au2.rep['errors']), au2.rep['errors']
        assert any('not a file' in e for e in au2.rep['errors']), au2.rep['errors']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_planning_file_names_are_excluded_from_the_inventory():
    """seeds_names had no planning guard, so the audit's own report and log
    names put 200-219 into the "seeds ever used" file-name row."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'output/seed_audit_200_219.json', json.dumps({'window': [200, 219]}))
        _write(d, 'logs/seed_audit_200_219.log', 'VERDICT: CLEAN\n')
        _write(d, 'output/cf_eval_80_89.pkl.sha256', 'deadbeef\n')
        au = _audit(d)
        assert au.rep['verdict']['clean'], au.rep['verdict']
        assert not (set(range(200, 220)) & au.seeds_names), au.seeds_names
        assert set(range(80, 90)) <= au.seeds_names, au.seeds_names
        assert '200' not in au.rep['all_seeds']['file names (seed<N>, _s<N>, _<a>_<b>.ext)']
        # the file names are still listed, marked planning, for inspection
        fn = [e for e in au.rep['filename_hits'] if e['planning']]
        assert len(fn) == 2 and all(e['tokens'] for e in fn), au.rep['filename_hits']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_git_no_repo_declares_an_unauditable_history():
    """~/MARL_code carries an empty .git directory.  The exception must be
    declared on the command line and recorded as a warning naming the root,
    never pass silently as "no seed was ever committed"."""
    d = _mkdtemp('wp12_sa_')
    try:
        r = os.path.join(d, 'MARL_code_like')
        os.makedirs(os.path.join(r, '.git'), exist_ok=True)
        _write(r, 'agent.py', '# seeds 42-49\n')
        au = SA.Audit(200, 219, [r], do_git=True, git_no_repo=[r])
        au.run()
        v = au.rep['verdict']
        assert au.rep['errors'] == [], au.rep['errors']
        assert any('--git-no-repo' in w and 'MARL_code_like' in w
                   for w in v['coverage_warnings']), v
        assert au.rep['git_roots'][0]['declared_not_a_repo'] is True
        # it is still not a scanned history: the run has no git evidence at all
        assert any('no root produced a git history' in f
                   for f in v['coverage_failures']), v
        # with a dump beside it, the run is complete again
        dump = os.path.join(d, 'hist.txt')
        with open(dump, 'w', encoding='utf-8') as f:
            f.write('commit 2222222222222222222222222222222222222222\n+ seeds 60-66\n')
        au2 = SA.Audit(200, 219, [r], do_git=True, git_no_repo=[r], git_logs=[dump])
        au2.run()
        assert au2.rep['verdict']['clean'], au2.rep['verdict']
        assert au2.rep['verdict']['coverage_warnings'], au2.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_history_lines_are_attributed_to_their_file():
    """`git log -p` carries the diffs of the planning documents themselves,
    which name the window on purpose.  Every line must be attributed to the
    file of its `diff --git` header, so a planning document's history is
    planning and a launcher's history is a hit; a commit message belongs to
    no file and is never planning."""
    d = _mkdtemp('wp12_sa_')
    try:
        # the dump lives OUTSIDE the scanned root, as it does on the server
        tree = os.path.join(d, 'tree')
        _write(tree, 'clean.sh', '# seeds 60-66\n')
        dump = os.path.join(d, 'hist.txt')
        with open(dump, 'w', encoding='utf-8') as f:
            f.write(
                'commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n'
                '    WPDOC: pre-registration\n'
                'diff --git a/docs/PREREGISTRATION_R2.md b/docs/PREREGISTRATION_R2.md\n'
                '+The confirmatory set is seeds 200-219, both scenarios\n'
                'diff --git a/tools/seed_audit.py b/tools/seed_audit.py\n'
                "+  --lo 200 --hi 219 --out output/seed_audit_200_219.json\n"
                'commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n'
                '    WP3: launcher\n'
                'diff --git a/launch_confirmatory.sh b/launch_confirmatory.sh\n'
                '+python3 run_timeline_comparison.py --seed-list 216-219\n')
        au = SA.Audit(200, 219, [tree], git_logs=[dump])
        au.run()
        v = au.rep['verdict']
        # only the launcher's history counts
        assert v['window_hits_in_git_history'] == [216, 217, 218, 219], v
        assert not v['clean'], v
        paths = sorted(set(e['path'] for e in au.rep['git_hits']))
        assert paths == ['gitlog:hist.txt:launch_confirmatory.sh'], paths
        plan = sorted(set(e['path'] for e in au.rep['planning_hits']))
        assert plan == ['gitlog:hist.txt:docs/PREREGISTRATION_R2.md'], plan
        # the tool's own diff names seed_audit_200_219.json: a planning output
        # name, so it yields neither a hit nor an inventory entry
        assert not any('seed_audit.py' in e['path'] for e in au.rep['text_hits'])
        assert au.rep['git_logs'][0]['commits'] == 2, au.rep['git_logs']
        # the planning lines do not enter the "seeds ever referenced" list
        assert not (set(range(200, 216)) & au.seeds_text), au.seeds_text
        assert {216, 217, 218, 219} <= au.seeds_text, au.seeds_text
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_planning_output_named_elsewhere_is_planning():
    """The rerun watchdog's status page quotes the newest log file of every
    tree, so `MARL_run_9b/logs/seed_audit_220_239.log` appears in a file that
    is not itself a planning document.  A value that comes only from the NAME
    of one of this study's planning outputs is planning too -- otherwise the
    audit's own log name makes its own window look touched."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'status/HEALTH.md',
               '- `MARL_run_9b`: head e57c815, 0 python processes, '
               'newest: MARL_run_9b/logs/seed_audit_200_219.log\n')
        _write(d, 'status/RESULTS.md',
               '- newest result: MARL_run_9b/output/cf_eval_204_207.pkl\n')
        au = _audit(d)
        v = au.rep['verdict']
        # the audit's own log name: planning, and out of the inventory
        plan = [e for e in au.rep['planning_hits'] if 'HEALTH' in e['path']]
        assert plan and plan[0]['values'] == list(range(200, 220)), au.rep['planning_hits']
        assert plan[0]['kinds'] == ['range-name (planning output)'], plan
        assert not (set(range(200, 204)) | set(range(208, 220))) & au.seeds_text, au.seeds_text
        # a result file of the window named in the same way is still a hit
        assert v['window_hits_in_text_non_planning'] == [204, 205, 206, 207], v
        assert not v['clean'], v
        assert [e['path'].split('/', 1)[1] for e in au.rep['text_hits']] == \
            ['status/RESULTS.md'], au.rep['text_hits']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_seed_key_string_value_is_not_truncated():
    """DEFECT 1.  The report's copy of a value under a `seed` key is cut at 120
    characters, and the hit test used to be read off that copy (and then off
    only its first 50 integers), so a seed of the window named past character
    120 of a string value was never extracted: no hit, no error, no
    long_values entry, no coverage warning -- the tree read CLEAN.  The shape
    is not hypothetical: string-valued seed keys ("seed_range":
    "42-49,100-139") already exist in this study's outputs."""
    spec = ('topology seeds 42-49 and 100-139 for training, 50-59 and 70-79 '
            'for development, 60-66 for validation, 80-99 superseded, '
            '205 confirmatory')
    assert spec.index('205') >= 120, spec.index('205')   # past the report's copy
    payload = {'provenance': {'seed_spec': spec}}
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'prov.json', json.dumps(payload))
        with open(os.path.join(d, 'prov.pkl'), 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        au = _audit(d)
        v = au.rep['verdict']
        assert v['window_hits_in_result_data'] == [205], v
        assert not v['clean'], v
        # both containers, not just one of them
        js = au.rep['json'][os.path.join(os.path.basename(d), 'prov.json')]
        pk = au.rep['pickles'][os.path.join(os.path.basename(d), 'prov.pkl')]
        assert js['hits'] == [205] and pk['hits'] == [205], (js['hits'], pk['hits'])
        # the ranges named in the same string are read whole, as on a text line
        assert {42, 49, 100, 139, 70, 79, 60, 66, 80, 99} <= au.seeds_data, au.seeds_data
        # the report's copy stays short; only the hit test reads the full value
        assert len(js['detail']['seedkeys'][0]['value']) <= 120
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_container_int_caps_do_not_hide_a_seed():
    """DEFECT 1 in its other two shapes: a value under a `seed` key whose
    report copy is a 120-character string because the list is not all-integer,
    and an int-keyed table past its 400th key."""
    d = _mkdtemp('wp12_sa_')
    try:
        payload = {'seed_list': ['from the launcher'] + list(range(1000, 1400)) + [208]}
        _write(d, 'mixed.json', json.dumps(payload))
        with open(os.path.join(d, 'mixed.pkl'), 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        au = _audit(d)
        v = au.rep['verdict']
        assert v['window_hits_in_result_data'] == [208], v
        assert not v['clean'], v
        js = au.rep['json'][os.path.join(os.path.basename(d), 'mixed.json')]
        assert len(str(js['detail']['seedkeys'][0]['value'])) <= 130
        # an int-keyed table past its 400th key.  400 distinct keys cannot fit
        # below 219, so the cap is shown on a window that they can precede.
        keys = {str(k): 1 for k in range(0, 400)}
        keys['1500'] = 1
        _write(d, 'keys.json', json.dumps({'per_seed': keys}))
        au2 = SA.Audit(1500, 1600, [d])
        au2.run()
        assert au2.rep['verdict']['window_hits_in_result_data'] == [1500], \
            au2.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_regex_fallback_matches_the_parsed_path():
    """DEFECT 2.  A JSON file above --json-max is not parsed.  The fallback
    accepted only a list literal or a bare integer as a seed key's value, so a
    STRING value was not matched at all, and `[[200,201],[202]]` stopped at
    the first `]`.  Neither weakening was visible anywhere in the report: such
    a file was not even listed as having seed content.

    The two paths must now agree, hit for hit, and the report must say which
    path each file took."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'a_string_seedkey.json',
               json.dumps({'config': {'seed_range': '200-219',
                                      'topology_seed': '205'}}))
        _write(d, 'b_nested.json', json.dumps({'seeds': [[200, 201], [202]]}))
        _write(d, 'c_list.json', json.dumps({'seeds': [207], 'per_seed': {'208': 1}}))
        _write(d, 'd_none.json', json.dumps({'arms': ['ospf'], 'ticks': 361}))
        parsed = _audit(d)
        regex = SA.Audit(200, 219, [d], json_max=10)
        regex.run()
        assert parsed.rep['counts']['json_parsed'] == 4, parsed.rep['counts']
        assert regex.rep['counts']['json_scanned'] == 4, regex.rep['counts']
        for name, want in (('a_string_seedkey.json', list(range(200, 220))),
                           ('b_nested.json', [200, 201, 202]),
                           ('c_list.json', [207, 208]),
                           ('d_none.json', [])):
            k = os.path.join(os.path.basename(d), name)
            assert parsed.rep['json'][k]['hits'] == want, (name, parsed.rep['json'][k])
            assert regex.rep['json'][k]['hits'] == want, (name, regex.rep['json'][k])
            # every file says which path it took -- a weaker scan is visible
            assert regex.rep['json'][k]['scan'] == 'regex-fallback', name
            assert parsed.rep['json'][k]['scan'] == 'parsed', name
        assert regex.rep['verdict']['window_hits_in_result_data'] == \
            parsed.rep['verdict']['window_hits_in_result_data'] == list(range(200, 220))
        # the string-valued seed key is LISTED, not silently absent
        md = regex.markdown()
        assert 'a_string_seedkey.json' in md and 'b_nested.json' in md, md
        assert 'fallback path (4)' in md, md
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_symbolic_links_are_counted_and_home_symlinks_must_be_roots():
    """DEFECT 5.  "Directory names <SKIP_DIRS> are not walked; nothing else
    is" was false: every symbolic link was dropped by a bare `continue`, with
    no count and no record; os.walk does not descend into a symlinked
    directory; and check_home_roots excluded a symlinked $HOME/MARL* tree from
    both the required roots and the walk -- so `~/MARL_run_10 -> /data/...`
    would have been neither required nor scanned, coverage ok."""
    d = _mkdtemp('wp12_sa_')
    try:
        home = os.path.join(d, 'home')
        real = os.path.join(d, 'elsewhere', 'MARL_run_linked')
        a = os.path.join(home, 'MARL_run_a')
        _write(a, 'x.sh', '# seeds 60-66\n')
        _write(real, 'y.sh', '# --seed-list 214\n')
        inside = _write(a, 'inside_target.txt', 'no seed\n')
        try:
            os.symlink(real, os.path.join(home, 'MARL_run_linked'),
                       target_is_directory=True)
            os.symlink(inside, os.path.join(a, 'link_inside.txt'))
            os.symlink(os.path.join(d, 'nowhere.txt'),
                       os.path.join(a, 'link_broken.txt'))
        except (OSError, NotImplementedError, AttributeError) as e:
            print('   (symbolic links unavailable on this host: %r)' % (e,))
            return
        pat = os.path.join(home, 'MARL*')
        au = SA.Audit(200, 219, [a], home_glob=pat)
        au.run()
        v = au.rep['verdict']
        # the symlinked tree is a $HOME/MARL* directory and must be a root
        assert os.path.join(home, 'MARL_run_linked') in au.rep['home_marl_dirs'], \
            au.rep['home_marl_dirs']
        assert au.rep['home_marl_symlinks'] == [os.path.join(home, 'MARL_run_linked')], \
            au.rep['home_marl_symlinks']
        assert not v['clean'] and any('MARL_run_linked' in f
                                      for f in v['coverage_failures']), v
        # links met during the walk are counted and resolved, never silent
        assert au.rep['counts']['symlinks'] == 2, au.rep['counts']
        by = {os.path.basename(e['path']): e for e in au.rep['symlinks']}
        assert sorted(by) == ['link_broken.txt', 'link_inside.txt'], sorted(by)
        assert by['link_inside.txt']['inside_roots'] is True
        assert by['link_broken.txt']['broken'] is True
        assert any('link_broken.txt' in w for w in v['coverage_warnings']), v
        assert not any('link_inside.txt' in w for w in v['coverage_warnings']), v
        assert 'Symbolic links met' in au.markdown()
        # with BOTH trees as roots the audit is complete again -- and the
        # symlinked tree's own content is then scanned (seed 214 is found)
        au2 = SA.Audit(200, 219, [a, os.path.join(home, 'MARL_run_linked')],
                       home_glob=pat)
        au2.run()
        assert au2.rep['home_marl_unscanned'] == [], au2.rep['home_marl_unscanned']
        assert au2.rep['verdict']['window_hits_in_text_non_planning'] == [214], \
            au2.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_an_empty_root_refuses_the_verdict():
    """DEFECT 6.  A root that exists but holds no file was a coverage
    WARNING, so a tree moved away, an unmounted path, or a root whose whole
    content is skipped, still exited 0 whenever the other roots met the
    aggregate floors -- the recorded run has ~966 files of slack, more than
    the whole of ~/MARL_run_9c (239) or ~/MARL_watch (12).  Only a human
    reading a `~` line in the summary caught it."""
    d = _mkdtemp('wp12_sa_')
    try:
        full = os.path.join(d, 'full')
        empty = os.path.join(d, 'moved_away')
        _write(full, 'run.sh', 'python3 x.py --seed-list 60-66\n')
        _write(full, 'logs/eval.log', 'seed=61 ok\n')
        os.makedirs(empty, exist_ok=True)
        au = SA.Audit(200, 219, [full, empty])
        au.run()
        v = au.rep['verdict']
        assert v['no_window_hits'], v          # nothing of the window anywhere
        assert not v['coverage_ok'] and not v['clean'], v
        assert any('EMPTY (0 files)' in f and 'moved_away' in f
                   for f in v['coverage_failures']), v
        assert not any('moved_away' in w for w in v['coverage_warnings']), v
        # the aggregate floor cannot see it: the other root meets it alone
        au2 = SA.Audit(200, 219, [full, empty], min_counts={'files': 2})
        au2.run()
        assert not au2.rep['verdict']['clean'], au2.rep['verdict']
        # ... and the CLI refuses with its own exit code
        out = os.path.join(d, 'rep.json')
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', full, empty,
                        '--out', out]) == 4
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', full,
                        '--out', out]) == 0
        # per-root floors: a root that shrank is refused although the
        # aggregate floor is still met
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', full,
                        '--out', out, '--min-root', 'full=2']) == 0
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', full,
                        '--out', out, '--min-root', 'full=3']) == 4
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', full,
                        '--out', out, '--min-per-root', '3']) == 4
        assert SA.main(['--lo', '200', '--hi', '219', '--roots', full,
                        '--out', out, '--min-root', 'MARL_typo=1']) == 4
        with open(out) as f:
            rep = json.load(f)
        assert any('no scanned root is named MARL_typo' in x
                   for x in rep['verdict']['coverage_failures']), rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_float_digits_are_not_seeds_but_an_integral_float_is():
    """A float under a `seed` key used to be stringified and mined with a
    plain `\d+`, so `sd_across_seeds = 6.064136589897359` named 6 (and
    merged_results.json put 43 such digit fragments into the inventory of
    every seed ever used, where the record explained them as flow and node
    identifiers).  Digits of a float are not seeds -- but a seed stored AS a
    float still is, so 205.0 must be found."""
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'means.json', json.dumps(
            {'sd_across_seeds': 6.064136589897359, 'per_seed_mean': [0.203, 1.5]}))
        au = _audit(d)
        assert au.rep['verdict']['clean'], au.rep['verdict']
        assert 6 not in au.seeds_data and 203 not in au.seeds_data, au.seeds_data
        _write(d, 'odd.json', json.dumps({'topology_seed': 205.0}))
        au2 = _audit(d)
        assert au2.rep['verdict']['window_hits_in_result_data'] == [205], au2.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _deep(seed, levels=9):
    """{'topology_seed': seed} wrapped in `levels` dict levels."""
    o = {'topology_seed': seed}
    for i in range(levels):
        o = {'lvl%d' % i: o}
    return o


def test_container_walk_has_no_depth_or_breadth_cap():
    """R3 DEFECT 2.  `walk_seeds` returned at depth > 6 and stopped after the
    1000th container element of a list.  A seed key below either cap was
    dropped with NO `errors` entry, no `long_values` entry, no coverage
    warning and no coverage failure: the file was not even listed as carrying
    seed content, and the tree read "CLEAN ... coverage ok", exit 0.

    A 1440-tick harness run makes a list of more than 1000 per-tick dicts an
    ordinary shape.  The large-JSON regex fallback has neither limit, so the
    PARSED path -- the stronger one by construction -- was the weaker of the
    two; the two paths are pinned against each other here as
    test_regex_fallback_matches_the_parsed_path does for the other shapes."""
    d = _mkdtemp('wp12_sa_')
    try:
        # element 1500 of a list, and depth 9 of a dict nest
        list_cap = {'runs': [{'tick': i} for i in range(1500)] + [{'topology_seed': 205}]}
        deep_nest = _deep(207, levels=9)
        _write(d, 'list_cap.json', json.dumps(list_cap))
        _write(d, 'deep_nest.json', json.dumps(deep_nest))
        with open(os.path.join(d, 'list_cap.pkl'), 'wb') as f:
            pickle.dump(list_cap, f, protocol=4)
        with open(os.path.join(d, 'deep_nest.pkl'), 'wb') as f:
            pickle.dump(deep_nest, f, protocol=4)

        parsed = _audit(d)
        v = parsed.rep['verdict']
        assert v['window_hits_in_result_data'] == [205, 207], v
        assert not v['clean'], v
        base = os.path.basename(d)
        for name, want in (('list_cap.json', [205]), ('deep_nest.json', [207])):
            e = parsed.rep['json'][os.path.join(base, name)]
            assert e['hits'] == want, (name, e)
            assert e['scan'] == 'parsed', (name, e)
            assert e['n_seedkeys'] >= 1, (name, e)   # LISTED as carrying seed content
        for name, want in (('list_cap.pkl', [205]), ('deep_nest.pkl', [207])):
            e = parsed.rep['pickles'][os.path.join(base, name)]
            assert e['hits'] == want, (name, e)
        # the parsed path and the --json-max-forced regex fallback agree
        regex = SA.Audit(200, 219, [d], json_max=10)
        regex.run()
        for name in ('list_cap.json', 'deep_nest.json'):
            k = os.path.join(base, name)
            assert regex.rep['json'][k]['scan'] == 'regex-fallback', name
            assert regex.rep['json'][k]['hits'] == parsed.rep['json'][k]['hits'], name
        assert regex.rep['verdict']['window_hits_in_result_data'] == \
            parsed.rep['verdict']['window_hits_in_result_data'] == [205, 207]
        # and the walk itself reported nothing as truncated
        assert parsed.rep['errors'] == [], parsed.rep['errors']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_container_walk_survives_a_cycle_and_records_a_stopped_walk():
    """The depth limit was also what kept a cyclic pickle from looping.  With
    no depth limit the walk remembers the containers it entered, so a cycle
    is walked once; and the one bound that remains (WALK_NODE_LIMIT) is
    recorded as an error, which refuses coverage -- never silent."""
    d = _mkdtemp('wp12_sa_')
    try:
        cyc = {'topology_seed': 206, 'child': {}}
        cyc['child']['back'] = cyc            # a cycle
        cyc['child']['also'] = cyc['child']   # and a self-reference
        out = SA.new_walk_out()
        SA.walk_seeds(cyc, '', out)           # must terminate
        assert 206 in out['ints'], out['ints']
        assert out['truncated'] == [], out['truncated']

        # reaching the node limit is an `errors` entry, so `errors: []` and
        # `coverage_ok` cannot both hold on a walk that did not finish
        big = {'rows': [{'n': i} for i in range(50)] + [{'topology_seed': 205}]}
        with open(os.path.join(d, 'big.pkl'), 'wb') as f:
            pickle.dump(big, f, protocol=4)
        real = SA.WALK_NODE_LIMIT
        SA.WALK_NODE_LIMIT = 5
        try:
            au = _audit(d)
        finally:
            SA.WALK_NODE_LIMIT = real
        assert any('WALK_NODE_LIMIT' in e for e in au.rep['errors']), au.rep['errors']
        v = au.rep['verdict']
        assert not v['coverage_ok'] and not v['clean'], v
        assert any('errors' in f for f in v['coverage_failures']), v['coverage_failures']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_min_root_file_is_read_by_the_tool_and_a_lost_file_refuses_the_verdict():
    """R3 DEFECT 4.  The per-root floors were expanded by the driver in a
    `while read ... done < FLOORS` loop under `set -u` and no `set -e`: a
    missing, renamed or unreadable floors file left the array empty, the
    failed redirect was never checked, and the audit ran with NO --min-root
    at all and could still exit 0.  The only trace was the absence of
    --min-root entries from the report's argv, which nothing checked.

    The tool now reads the file, records what it read, and refuses the
    verdict when it cannot."""
    d = _mkdtemp('wp12_sa_')
    try:
        root = os.path.join(d, 'MARL_run_z')
        os.makedirs(root, exist_ok=True)
        for i in range(3):
            _write(root, 'f%d.txt' % i, 'nothing here\n')
        floors = os.path.join(d, 'floors.txt')
        with open(floors, 'w', encoding='utf-8') as f:
            f.write('# a comment\n\nMARL_run_z=2\n')
        au = SA.Audit(200, 219, [root], home_check=False, min_root_file=floors)
        au.run()
        mrf = au.rep['min_root_file']
        assert mrf['ok'] and mrf['floors'] == {'MARL_run_z': 2}, mrf
        assert mrf['sha256'] and mrf['path'] == os.path.abspath(floors), mrf
        assert au.rep['requested']['min_root'] == {'MARL_run_z': 2}, au.rep['requested']
        assert au.rep['verdict']['coverage_ok'] and au.rep['verdict']['clean'], au.rep['verdict']

        # a floor the tree does not meet still refuses, i.e. it was applied
        with open(floors, 'w', encoding='utf-8') as f:
            f.write('MARL_run_z=99\n')
        au2 = SA.Audit(200, 219, [root], home_check=False, min_root_file=floors)
        au2.run()
        assert not au2.rep['verdict']['coverage_ok'], au2.rep['verdict']
        assert any('MARL_run_z 3 < 99' in c for c in au2.rep['verdict']['coverage_failures']), \
            au2.rep['verdict']['coverage_failures']

        # missing file: a COVERAGE FAILURE, not a floor-less run that passes
        au3 = SA.Audit(200, 219, [root], home_check=False,
                       min_root_file=os.path.join(d, 'NOSUCHFILE.txt'))
        au3.run()
        v3 = au3.rep['verdict']
        assert not v3['coverage_ok'] and not v3['clean'], v3
        assert any('--min-root-file' in c and 'NOT applied' in c
                   for c in v3['coverage_failures']), v3['coverage_failures']

        # a file that parses to nothing is the same failure
        with open(floors, 'w', encoding='utf-8') as f:
            f.write('# only comments\n\n')
        au4 = SA.Audit(200, 219, [root], home_check=False, min_root_file=floors)
        au4.run()
        assert not au4.rep['verdict']['coverage_ok'], au4.rep['verdict']
        assert any('no per-root floor parsed' in e for e in au4.rep['errors']), au4.rep['errors']

        # and an unparsable line is not silently skipped
        with open(floors, 'w', encoding='utf-8') as f:
            f.write('MARL_run_z=2\nMARL_run_z 7\n')
        au5 = SA.Audit(200, 219, [root], home_check=False, min_root_file=floors)
        au5.run()
        assert not au5.rep['verdict']['coverage_ok'], au5.rep['verdict']
        assert any('unparsable line' in e for e in au5.rep['errors']), au5.rep['errors']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_committed_floors_file_parses_and_the_driver_guards_it():
    """The committed floors file must be readable by the same parser, and the
    committed driver must fail loudly rather than run without floors."""
    floors = os.path.join(ROOT, 'tools', 'seed_audit_floors.txt')
    d = _mkdtemp('wp12_sa_')
    try:
        _write(d, 'x.txt', 'nothing\n')
        au = SA.Audit(200, 219, [d], home_check=False, min_root_file=floors)
        au.run()
        mrf = au.rep['min_root_file']
        assert mrf['ok'], mrf
        assert len(mrf['floors']) >= 14, mrf['floors']
        assert mrf['floors'].get('MARL_run_9b'), mrf['floors']
    finally:
        shutil.rmtree(d, ignore_errors=True)
    with open(os.path.join(ROOT, 'tools', 'run_seed_audit.sh'), encoding='utf-8') as f:
        drv = f.read()
    assert '--min-root-file' in drv, 'the driver must let the tool read the floors'
    assert '[ -r "$FLOORS" ]' in drv, 'the driver must test the floors file before using it'
    assert 'NFLOORS' in drv and 'exit 2' in drv, 'a floor-less driver run must fail loudly'


def test_synced_head_is_recorded_in_the_report():
    """R3 DEFECT 5.  docs/SEED_AUDIT_200_219.md 2.2 pointed a reviewer at the
    LIVE `~/MARL_run_9b/.synced_head` to confirm that the history dump and the
    scanned tree were at the same HEAD.  The next sync overwrote it (22609ec
    -> 083f230) and the equality stopped being reconstructable at all.  The
    value the run saw is now frozen in the report."""
    d = _mkdtemp('wp12_sa_')
    try:
        tree = os.path.join(d, 'tree')
        os.makedirs(tree, exist_ok=True)
        _write(tree, '.synced_head', '22609ec\n')
        _write(d, 'x.txt', 'nothing\n')
        au = SA.Audit(200, 219, [d], home_check=False, tree=tree)
        au.run()
        sh = au.rep['synced_head']
        assert sh['value'] == '22609ec', sh
        assert sh['mtime_utc'] and sh['error'] is None, sh
        assert sh['path'] == os.path.join(os.path.abspath(tree), '.synced_head'), sh
        assert '22609ec' in au.markdown(), au.markdown()[:2000]
        # a tree without one is recorded, not a coverage failure
        au2 = SA.Audit(200, 219, [d], home_check=False, tree=os.path.join(d, 'notree'))
        au2.run()
        assert au2.rep['synced_head']['value'] is None
        assert au2.rep['synced_head']['error'], au2.rep['synced_head']
        assert au2.rep['verdict']['coverage_ok'], au2.rep['verdict']
    finally:
        shutil.rmtree(d, ignore_errors=True)
