"""WP0: tools/series_diff.py and tools/invariance_check.sh on synthetic pickles."""
import contextlib
import copy
import io
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import series_diff as SD  # noqa: E402

NAN = float('nan')


def _mkdtemp(prefix):
    """Temporary directory inside the tree (output/tmp_tests): the server
    rules forbid writing outside ~/MARL_run_9, /tmp included."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def _run(conn, **extra):
    r = {'conn': list(conn), 'delivered': [1.0, 2.0, 3.0, 4.0],
         'fragments': [1, 1, 2, 2], 'relay_bridges': [0, 0, 1, 1],
         'relay_count': [0, 0, 1, 1], 'link_count': [10, 10, 9, 9],
         'energy': [5.0, 5.0, 5.5, NAN], 'latency': [NAN, 1.0, 1.0, 1.0],
         'n_radio_sites': 22}
    r.update(extra)
    return r


def _pickle(arms=('ospf', 'marl_static'), seeds=(82, 91)):
    P = {'recovery_multi': {}, 'rescue_multi': {}, 'seeds': list(seeds)}
    for sk in ('recovery_multi', 'rescue_multi'):
        for a in arms:
            P[sk][a] = {s: _run([0.5, 0.6, 0.7, 0.8]) for s in seeds}
    return P


class _Tmp:
    def __enter__(self):
        self.d = _mkdtemp('wp0_sd_')
        return self

    def write(self, name, obj):
        p = os.path.join(self.d, name)
        with open(p, 'wb') as f:
            pickle.dump(obj, f)
        return p

    def __exit__(self, *exc):
        shutil.rmtree(self.d, ignore_errors=True)


def _main(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = SD.main(args)
    return rc, buf.getvalue()


def test_same_semantics():
    assert SD.same(NAN, NAN)
    assert not SD.same(NAN, 0.0)
    assert not SD.same(0.0, -0.0)
    assert SD.same(1, 1.0)
    assert SD.same(True, 1)
    assert not SD.same(1.0, 1.0000000000000002)
    assert SD.same([1, [2.0, NAN]], [1, [2.0, NAN]])
    assert not SD.same({'a': 1, 'b': 2}, {'b': 2, 'a': 1})   # order counts
    assert SD.same('x', 'x') and not SD.same('x', 'y')
    try:
        import numpy as np
        assert SD.same(np.float64(0.1), 0.1)
        assert SD.same(np.array([1.0, NAN]), [1.0, NAN])
        assert SD.same(np.int64(3), 3)
    except ImportError:
        pass


def test_compare_series_first_tick():
    assert SD.compare_series([1, 2, 3], [1, 2, 3]) is None
    d = SD.compare_series([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.5, 4.5])
    assert d['first_tick'] == 2 and d['n_diff'] == 2
    assert 'first_tick=2' in SD.describe(d)
    d = SD.compare_series([1, 2, 3], [1, 2])
    assert d['first_tick'] == 2 and d['len_ref'] == 3 and d['len_new'] == 2
    d = SD.compare_series(SD.MISSING, [1])
    assert SD.describe(d) == 'MISSING in ref'
    assert SD.compare_series(22, 22) is None
    assert SD.compare_series(22, 23)['kind'] == 'scalar'


def test_identical_pickles_exit_zero():
    with _Tmp() as t:
        a = t.write('a.pkl', _pickle())
        b = t.write('b.pkl', _pickle())
        rc, out = _main([a, b])
        assert rc == 0, out
        # 2 scenarios x 2 arms x 2 seeds x 8 default keys
        assert 'SUMMARY series=64 identical=64 different=0 missing=0' in out
        assert out.count('IDENTICAL') == 64
        rc, out = _main([a, b, '--quiet'])
        assert rc == 0 and 'IDENTICAL' not in out


def test_difference_reported_with_first_tick():
    with _Tmp() as t:
        P = _pickle()
        Q = copy.deepcopy(P)
        Q['rescue_multi']['marl_static'][91]['delivered'][3] = 4.25
        a, b = t.write('a.pkl', P), t.write('b.pkl', Q)
        rc, out = _main([a, b, '--quiet'])
        assert rc == 1
        line = [l for l in out.splitlines() if 'DIFFERENT' in l]
        assert len(line) == 1, out
        assert line[0].split()[:4] == ['rescue_ops', 'marl_static', 'seed=91',
                                       'delivered']
        assert 'first_tick=3' in line[0] and 'ref=4.0 new=4.25' in line[0]
        assert 'non-identical arms: marl_static(1)' in out
        # restricting the comparison to the unchanged arm / scenario passes
        assert _main([a, b, '--arms', 'ospf'])[0] == 0
        assert _main([a, b, '--scenarios', 'A'])[0] == 0
        assert _main([a, b, '--seeds', '82'])[0] == 0
        assert _main([a, b, '--keys', 'conn,energy'])[0] == 0
        assert _main([a, b, '--scenarios', 'B', '--seeds', '91'])[0] == 1


def test_missing_arm_seed_or_key_fails():
    with _Tmp() as t:
        P = _pickle()
        Q = copy.deepcopy(P)
        del Q['recovery_multi']['ospf'][82]
        del Q['rescue_multi']['marl_static'][82]['energy']
        a, b = t.write('a.pkl', P), t.write('b.pkl', Q)
        rc, out = _main([a, b, '--quiet'])
        assert rc == 1
        assert 'missing=9' in out, out     # 8 keys of the lost run + 1 key
        rc, out = _main([a, b, '--arms', 'sdn'])
        assert rc == 1 and 'MISSING in ref+new' in out
        rc, _ = _main([a, os.path.join(t.d, 'nope.pkl')])
        assert rc == 2


def test_key_selection_all_and_common():
    with _Tmp() as t:
        P = _pickle(arms=('ospf',), seeds=(82,))
        Q = copy.deepcopy(P)
        for sk in ('recovery_multi', 'rescue_multi'):
            Q[sk]['ospf'][82]['stale_edges'] = [0, 0, 0, 0]
        a, b = t.write('a.pkl', P), t.write('b.pkl', Q)
        rc, out = _main([a, b, '--keys', 'common', '--quiet'])
        assert rc == 0, out
        assert 'identical=18' in out          # 9 common keys x 2 scenarios
        assert 'NOTE keys added in NEW' in out
        assert 'stale_edges(x2)' in out
        rc, out = _main([a, b, '--keys', 'all', '--quiet'])
        assert rc == 1 and 'missing=2' in out
        Q['recovery_multi']['ospf'][82]['n_radio_sites'] = 21
        b = t.write('b.pkl', Q)
        rc, out = _main([a, b, '--keys', 'common', '--quiet'])
        assert rc == 1 and 'DIFFERENT (scalar) ref=22 new=21' in out


def test_common_keys_fail_when_new_drops_a_series():
    """Review defect: a REF series missing from NEW must fail under
    'common', whether it is a section-14 series or any other REF key."""
    with _Tmp() as t:
        P = _pickle(arms=('ospf',), seeds=(82,))
        for sk in ('recovery_multi', 'rescue_multi'):
            P[sk]['ospf'][82]['adm_rate'] = 0.5
        Q = copy.deepcopy(P)
        for sk in ('recovery_multi', 'rescue_multi'):
            del Q[sk]['ospf'][82]['delivered']
            del Q[sk]['ospf'][82]['fragments']
        a, b = t.write('a.pkl', P), t.write('b.pkl', Q)
        rc, out = _main([a, b, '--keys', 'common', '--quiet'])
        assert rc == 1, out
        assert 'identical=16 different=0 missing=4' in out, out
        assert 'delivered       MISSING in new' in out
        assert 'fragments       MISSING in new' in out
        assert 'NOTE' not in out
        # section-14 series may not be allow-listed
        rc, _ = _main([a, b, '--keys', 'common',
                       '--allow-removed', 'delivered'])
        assert rc == 2
        # a key missing on BOTH sides is still required (DEFAULT_KEYS)
        P2 = copy.deepcopy(Q)
        c = t.write('c.pkl', P2)
        rc, out = _main([c, b, '--keys', 'common', '--quiet'])
        assert rc == 1 and 'MISSING in ref+new' in out, out

        # a non-required REF key: fails unless explicitly allow-listed
        R = copy.deepcopy(P)
        for sk in ('recovery_multi', 'rescue_multi'):
            del R[sk]['ospf'][82]['adm_rate']
        d = t.write('d.pkl', R)
        rc, out = _main([a, d, '--keys', 'common', '--quiet'])
        assert rc == 1 and 'adm_rate        MISSING in new' in out, out
        rc, out = _main([a, d, '--keys', 'common', '--quiet',
                         '--allow-removed', 'adm_rate'])
        assert rc == 0, out
        assert 'identical=18 different=0 missing=0' in out
        assert ('NOTE keys removed in NEW (allowed by --allow-removed): '
                'adm_rate(x2)') in out
        # the allow-list does not hide a real difference of that key
        R2 = copy.deepcopy(P)
        R2['rescue_multi']['ospf'][82]['adm_rate'] = 0.75
        e = t.write('e.pkl', R2)
        rc, out = _main([a, e, '--keys', 'common', '--quiet',
                         '--allow-removed', 'adm_rate'])
        assert rc == 1 and 'DIFFERENT (scalar)' in out
        # 'all' honours the allow-list too
        rc, _ = _main([a, d, '--keys', 'all', '--allow-removed', 'adm_rate'])
        assert rc == 0


def test_provenance_line_marks_incomplete_records():
    P = {'provenance': {'tree_head': 'abc1234', 'tree_sha256': 'f' * 64,
                        'provenance_complete': True}}
    assert 'INCOMPLETE' not in SD._prov_line('ref', P)
    P['provenance']['provenance_complete'] = False
    assert 'INCOMPLETE' in SD._prov_line('ref', P)
    old = {'provenance': {'tree_head': None, 'tree_sha256': None}}  # no flag
    assert 'head=None' in SD._prov_line('ref', old)
    assert 'INCOMPLETE' in SD._prov_line('ref', old)
    assert 'pre-WP0' in SD._prov_line('ref', {})


def test_type_change_is_noted():
    assert SD.type_change([1, 2, 3], [1, 2.0, 3]) == 1
    assert SD.type_change([True, 0], [1, 0]) == 0
    assert SD.type_change([1.0, NAN], [1.0, NAN]) is None
    assert SD.type_change(22, 22.0) == 'scalar'
    assert SD.type_change('x', 'x') is None
    with _Tmp() as t:
        P = _pickle(arms=('ospf',), seeds=(82,))
        Q = copy.deepcopy(P)
        Q['recovery_multi']['ospf'][82]['relay_count'] = [0.0, 0.0, 1.0, 1.0]
        a, b = t.write('a.pkl', P), t.write('b.pkl', Q)
        rc, out = _main([a, b, '--quiet'])
        assert rc == 0, out                  # equal values: still identical
        assert ('NOTE numeric element type changed, values equal '
                '(int/float/bool): relay_count(x1)') in out
        rc, out = _main([a, a, '--quiet'])
        assert rc == 0 and 'NOTE' not in out


def _bash():
    for cand in ('bash', r'C:\Program Files\Git\bin\bash.exe'):
        try:
            subprocess.run([cand, '-c', 'true'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return cand
        except Exception:
            continue
    return None


def _inv(bash, *args, **env_extra):
    env = dict(os.environ, PYTHON=sys.executable)
    for k in ('INV_KEYS', 'INV_SEEDS', 'INV_ALLOW_REMOVED', 'INV_SCENARIOS'):
        env.pop(k, None)
    env.update(env_extra)
    p = subprocess.run([bash, os.path.join(TOOLS, 'invariance_check.sh')]
                       + [x.replace('\\', '/') for x in args],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return p.returncode, p.stdout.decode('utf-8', 'replace')


def test_invariance_check_matrix():
    bash = _bash()
    if bash is None:
        print('  (bash not available: invariance_check.sh not exercised)')
        return
    with _Tmp() as t:
        P = _pickle(arms=('marl_static', 'xldet', 'ospf', 'olsr', 'batman',
                          'aodv', 'sdn', 'random'))
        Q = copy.deepcopy(P)
        Q['recovery_multi']['marl_static'][82]['fragments'][2] = 3
        a, b = t.write('a.pkl', P), t.write('b.pkl', Q)
        rc, out = _inv(bash, '--wp', 'WP1', a, b)
        assert rc == 0, out
        assert 'INVARIANCE PASS' in out
        for arm in ('ospf', 'olsr', 'batman', 'aodv', 'sdn'):
            assert 'PASS  %s ' % arm in out, out
        assert 'INFO  marl_static  changed' in out
        assert 'INFO  xldet  unchanged' in out
        rc, out = _inv(bash, '--wp', 'wp0', a, b)
        assert rc == 1 and 'FAIL  marl_static' in out and 'first_tick=2' in out
        rc, out = _inv(bash, a, b, 'ospf', 'marl_static')
        assert rc == 1 and 'PASS  ospf' in out and 'FAIL  marl_static' in out
        rc, out = _inv(bash, '--wp', 'WP6', a, b)
        assert rc == 0 and 'PASS  xldet' in out
        Q['rescue_multi']['sdn'][91]['energy'][0] = 5.000001
        b = t.write('b.pkl', Q)
        rc, out = _inv(bash, '--wp', 'WP4', a, b)
        assert rc == 1 and 'FAIL  sdn' in out and 'INVARIANCE FAIL' in out
        rc, _ = _inv(bash, '--wp', 'WP99', a, b)
        assert rc == 2
        rc, _ = _inv(bash, a)
        assert rc == 2
        # D12.2 (D12-FIXES): SDN may change in Scenario A, nothing else may;
        # Scenario B of SDN is checked on its own with INV_SCENARIOS
        R = copy.deepcopy(P)
        R['recovery_multi']['sdn'][82]['delivered'][3] = 1.5
        a2, c = t.write('a2.pkl', P), t.write('c.pkl', R)
        rc, out = _inv(bash, '--wp', 'D12.2', a2, c)
        assert rc == 0 and 'INVARIANCE PASS' in out, out
        for arm in ('ospf', 'olsr', 'batman', 'aodv', 'marl_static', 'xldet', 'random'):
            assert 'PASS  %s ' % arm in out, (arm, out)
        assert 'INFO  sdn  changed' in out, out
        rc, out = _inv(bash, a2, c, 'sdn', INV_SCENARIOS='rescue_ops')
        assert rc == 0 and 'PASS  sdn ' in out, out
        rc, out = _inv(bash, a2, c, 'sdn', INV_SCENARIOS='recovery')
        assert rc == 1 and 'FAIL  sdn' in out, out
        R['rescue_multi']['ospf'][91]['conn'][0] = 0.25
        c = t.write('c.pkl', R)
        rc, out = _inv(bash, '--wp', 'D12.2', a2, c)
        assert rc == 1 and 'FAIL  ospf' in out, out
        # D12F.8: OLSR, B.A.T.M.A.N. and AODV may change, nothing else may
        S = copy.deepcopy(P)
        for arm in ('olsr', 'batman', 'aodv'):
            S['recovery_multi'][arm][82]['delivered'][3] = 2.5
        d = t.write('d.pkl', S)
        rc, out = _inv(bash, '--wp', 'D12F.8', a2, d)
        assert rc == 0 and 'INVARIANCE PASS' in out, out
        for arm in ('ospf', 'sdn', 'marl_static', 'xldet', 'random'):
            assert 'PASS  %s ' % arm in out, (arm, out)
        for arm in ('olsr', 'batman', 'aodv'):
            assert 'INFO  %s  changed' % arm in out, (arm, out)
        S['recovery_multi']['ospf'][82]['delivered'][3] = 2.5
        d = t.write('d.pkl', S)
        rc, out = _inv(bash, '--wp', 'D12F.8', a2, d)
        assert rc == 1 and 'FAIL  ospf' in out, out
        # XL1: XL-DET may change, nothing else may
        X = copy.deepcopy(P)
        X['rescue_multi']['xldet'][91]['delivered'][3] = 2.5
        x = t.write('x.pkl', X)
        rc, out = _inv(bash, '--wp', 'XL1', a2, x)
        assert rc == 0 and 'INVARIANCE PASS' in out, out
        for arm in ('ospf', 'olsr', 'batman', 'aodv', 'sdn', 'marl_static', 'random'):
            assert 'PASS  %s ' % arm in out, (arm, out)
        assert 'INFO  xldet  changed' in out, out
        X['rescue_multi']['sdn'][91]['delivered'][3] = 2.5
        x = t.write('x.pkl', X)
        rc, out = _inv(bash, '--wp', 'XL1', a2, x)
        assert rc == 1 and 'FAIL  sdn' in out, out


def test_invariance_check_removed_series_and_arms_not_run():
    bash = _bash()
    if bash is None:
        print('  (bash not available: invariance_check.sh not exercised)')
        return
    arms = ('marl_static', 'xldet', 'random', 'ospf', 'olsr', 'batman',
            'aodv', 'sdn')
    with _Tmp() as t:
        P = _pickle(arms=arms)
        for sk in ('recovery_multi', 'rescue_multi'):
            for arm in arms:
                for s in (82, 91):
                    P[sk][arm][s]['adm_rate'] = 0.5
        a = t.write('a.pkl', P)

        # review defect: NEW stops recording section-14 series -> FAIL
        Q = copy.deepcopy(P)
        for sk in ('recovery_multi', 'rescue_multi'):
            for s in (82, 91):
                del Q[sk]['ospf'][s]['delivered']
                del Q[sk]['ospf'][s]['fragments']
        b = t.write('b.pkl', Q)
        rc, out = _inv(bash, '--wp', 'WP1', a, b)
        assert rc == 1, out
        assert 'FAIL  ospf  series=40 identical=32 different=0 missing=8' in out
        assert 'delivered       MISSING in new' in out
        assert 'INVARIANCE FAIL' in out
        rc, out = _inv(bash, '--wp', 'WP1', a, b,
                       INV_ALLOW_REMOVED='delivered,fragments')
        assert rc == 2, out

        # a deliberate removal of another key: FAIL unless allow-listed,
        # and the NOTE is shown for the PASS arm
        R = copy.deepcopy(P)
        for sk in ('recovery_multi', 'rescue_multi'):
            for arm in arms:
                for s in (82, 91):
                    del R[sk][arm][s]['adm_rate']
                    R[sk][arm][s]['stale_edges'] = [0, 0, 0, 0]
        c = t.write('c.pkl', R)
        rc, out = _inv(bash, '--wp', 'WP1', a, c)
        assert rc == 1 and 'FAIL  sdn' in out, out
        rc, out = _inv(bash, '--wp', 'WP1', a, c, INV_ALLOW_REMOVED='adm_rate')
        assert rc == 0, out
        assert 'PASS  sdn  series=36 identical=36' in out, out
        assert 'NOTE keys removed in NEW (allowed by --allow-removed): adm_rate(x4)' in out
        assert 'NOTE keys added in NEW' in out and 'stale_edges(x4)' in out
        assert 'allow-removed=adm_rate' in out

        # arms missing from NEW: 'not run', never 'changed'
        S = copy.deepcopy(P)
        for sk in ('recovery_multi', 'rescue_multi'):
            del S[sk]['xldet']
            del S[sk]['batman']
        d = t.write('d.pkl', S)
        rc, out = _inv(bash, a, d, 'marl_static', 'random', 'sdn')
        assert rc == 0, out
        assert 'INFO  xldet  not run in NEW (not compared)' in out
        assert 'INFO  batman  not run in NEW (not compared)' in out
        assert 'changed (allowed' not in out
        rc, out = _inv(bash, '--wp', 'WP1', a, d)
        assert rc == 1 and 'FAIL  batman  not run in NEW (required for WP1)' in out
        rc, out = _inv(bash, '--wp', 'WP0', d, a)
        assert rc == 1 and 'FAIL  xldet  not run in REF (required for WP0)' in out


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
