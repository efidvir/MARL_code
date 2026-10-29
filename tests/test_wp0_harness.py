"""WP0: harness plumbing (pool cap, provenance and pass stats in the pickle)
with the physics pass stubbed out, plus a static scan that every process pool
in the repository is sized through runinfo.pool_workers."""
import ast
import glob
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _mkdtemp(prefix):
    """Temporary directory inside the tree (output/tmp_tests): the server
    rules forbid writing outside ~/MARL_run_9, /tmp included."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


class _FakePool:
    created = []

    def __init__(self, processes=None):
        _FakePool.created.append(processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, tasks):
        return [fn(t) for t in tasks]


_CALLS = []


def _fake_pass(mode, topology_seed, checkpoint_path, scenario_name='recovery'):
    _CALLS.append((mode, checkpoint_path))
    if mode == 'aodv' and scenario_name == 'rescue_ops' and topology_seed == 82:
        raise RuntimeError('stub failure')
    return {'conn': [float(topology_seed)], 'arm': mode,
            'scenario': scenario_name}


def test_main_pool_cap_provenance_and_pass_stats():
    import run_timeline_comparison as R
    saved = (multiprocessing.Pool, R.run_physics_pass, R.OUTPUT_DIR,
             R.CHECKPOINT_DIR)
    saved_env = {k: os.environ.get(k)
                 for k in ('MARL_POOL_WORKERS', 'MARL_EVAL_CHECKPOINT')}
    tmp = _mkdtemp('wp0_harness_')
    ck = os.path.join(ROOT, 'ce_ck_s1234', 'policy_final.pt')
    try:
        multiprocessing.Pool = _FakePool
        R.run_physics_pass = _fake_pass
        R.OUTPUT_DIR = tmp
        R.CHECKPOINT_DIR = tmp
        os.environ['MARL_EVAL_CHECKPOINT'] = ck
        os.environ['MARL_POOL_WORKERS'] = '3'
        _FakePool.created.clear()
        R.main(seed_list='82,91', arms='ospf,aodv', no_figures=True,
               out_file='wp0_capped.pkl')
        # 2 arms x 2 seeds x 2 scenarios = 8 phase-1 tasks, capped at 3;
        # no adapting arm -> no phase-2 pool at all
        assert _FakePool.created == [3], _FakePool.created
        with open(os.path.join(tmp, 'wp0_capped.pkl'), 'rb') as f:
            P = pickle.load(f)
        assert set(P) == {'recovery_multi', 'rescue_multi', 'seeds',
                          'provenance', 'pass_stats'}
        assert P['seeds'] == [82, 91]
        assert P['recovery_multi']['ospf'][91] == {
            'conn': [91.0], 'arm': 'ospf', 'scenario': 'recovery'}
        assert 82 not in P['rescue_multi']['aodv']       # failed pass dropped
        assert 91 in P['rescue_multi']['aodv']

        prov = P['provenance']
        assert prov['driver'] == 'run_timeline_comparison'
        assert prov['arms'] == ['ospf', 'aodv'] and prov['seeds'] == [82, 91]
        assert prov['pool_workers'] == {'phase1': 3, 'phase2': 0}
        assert prov['marl_env']['MARL_POOL_WORKERS'] == '3'
        assert prov['pythonhashseed'] == os.environ.get('PYTHONHASHSEED')
        assert prov['total_ticks'] == R.TOTAL_TICKS
        # trackAgent merge: ospf and aodv read no checkpoint, so none is
        # resolved, recorded or handed to a pass -- even with
        # MARL_EVAL_CHECKPOINT set
        assert prov['checkpoint'] is None
        assert prov['checkpoint_sha256'] == []
        for k in ('tree_head', 'tree_sha256', 'schema_id', 'argv'):
            assert k in prov
        assert _CALLS and all(c is None for _m, c in _CALLS), _CALLS

        stats = P['pass_stats']
        expect = {'%s|%s|%d' % (a, s, seed) for a in ('ospf', 'aodv')
                  for s in ('recovery', 'rescue_ops') for seed in (82, 91)}
        assert set(stats) == expect
        assert stats['aodv|rescue_ops|82']['ok'] is False
        assert stats['ospf|recovery|82']['ok'] is True
        assert all(v['phase'] == 1 for v in stats.values())
        assert all('peak_rss_mb' in v and 'wall_s' in v for v in stats.values())

        # an arm that DOES read the checkpoint: it is resolved, recorded
        # with its sha256, and handed to that arm's passes only
        del _CALLS[:]
        R.main(seed_list='82', arms='marl_static,ospf', no_figures=True,
               out_file='wp0_ck.pkl')
        with open(os.path.join(tmp, 'wp0_ck.pkl'), 'rb') as f:
            prov = pickle.load(f)['provenance']
        assert prov['checkpoint'] == ck
        assert len(prov['checkpoint_sha256']) == 1
        assert len(prov['checkpoint_sha256'][0]) == 64
        got = sorted(set(_CALLS))
        assert got == [('marl_static', ck), ('ospf', None)], got

        # unset cap: the historical sizing min(tasks, cpu_count)
        del os.environ['MARL_POOL_WORKERS']
        _FakePool.created.clear()
        R.main(seed_list='82,91', arms='ospf,aodv', no_figures=True,
               out_file='wp0_uncapped.pkl')
        assert _FakePool.created == [min(8, multiprocessing.cpu_count())]
    finally:
        (multiprocessing.Pool, R.run_physics_pass, R.OUTPUT_DIR,
         R.CHECKPOINT_DIR) = saved
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


class _Node:
    def __init__(self, node_type):
        self.node_type = type('NT', (), {'value': node_type})()


class _Agent:
    def __init__(self, net):
        self.policy_net = net


def _load_per_agent(R, path, agent_ids, nodes):
    """The per-agent branch of run_physics_pass's checkpoint loader, run on
    the same helpers: returns (agents, is_per_agent, n_keys, direct, fallback)."""
    import torch
    from sixg_sim.agent import PolicyNetwork
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    checkpoint = R._strip_reserved_keys(checkpoint)
    is_per_agent = R._is_per_agent_checkpoint(checkpoint)
    agents = {aid: _Agent(PolicyNetwork(R.OBS_DIM)) for aid in agent_ids}
    direct = fallback = 0
    if is_per_agent:
        fallback_sd = list(checkpoint.values())[0]
        for aid, agent in agents.items():
            sd, is_direct = R._per_agent_source(checkpoint, aid, nodes,
                                                fallback_sd)
            assert R._load_into_agent(agent, sd) is False
            if is_direct:
                direct += 1
            else:
                fallback += 1
    return agents, is_per_agent, len(checkpoint), direct, fallback


def _same_weights(net, sd):
    import torch
    own = net.state_dict()
    return (list(own) == list(sd)
            and all(torch.equal(own[k], sd[k]) for k in own))


def test_adapted_checkpoint_with_provenance_loads_per_agent():
    import torch
    import run_timeline_comparison as R
    from sixg_sim.agent import PolicyNetwork
    assert R.ADAPTED_PROVENANCE_KEY.startswith('__')
    torch.manual_seed(12345)
    src = {aid: PolicyNetwork(R.OBS_DIM).state_dict()
           for aid in ('gnb_0', 'relay_0')}
    prov = {'tree_head': 'abc1234', 'tree_sha256': '0' * 64,
            'marl_env': {'MARL_POOL_WORKERS': '8'}, 'argv': ['x']}
    nodes = {'gnb_0': _Node('gnb'), 'relay_0': _Node('relay'),
             'relay_7': _Node('relay'), 'ue_3': _Node('ue')}
    ids = ('gnb_0', 'relay_7', 'ue_3')      # direct / same type / last resort
    tmp = _mkdtemp('wp0_adapted_')
    try:
        # as written by run_physics_pass: provenance inserted LAST; also the
        # worst case with the provenance entry FIRST (it must never become
        # the last-resort fallback state dict)
        for label, order in (('trailing', ['gnb_0', 'relay_0', '__prov']),
                             ('leading', ['__prov', 'gnb_0', 'relay_0'])):
            ck = {}
            for k in order:
                if k == '__prov':
                    ck[R.ADAPTED_PROVENANCE_KEY] = prov
                else:
                    ck[k] = src[k]
            path = os.path.join(tmp, 'adapted_%s.pt' % label)
            torch.save(ck, path)
            raw = torch.load(path, map_location='cpu', weights_only=False)
            assert len(raw) == 3 and raw[R.ADAPTED_PROVENANCE_KEY] == prov
            agents, per_agent, n_keys, direct, fallback = _load_per_agent(
                R, path, ids, nodes)
            assert per_agent, label
            assert n_keys == 2, label
            assert (direct, fallback) == (1, 2), (label, direct, fallback)
            assert _same_weights(agents['gnb_0'].policy_net, src['gnb_0'])
            assert _same_weights(agents['relay_7'].policy_net, src['relay_0'])
            assert _same_weights(agents['ue_3'].policy_net, src['gnb_0'])

        # a legacy shared checkpoint (raw state dict) is not per-agent, and
        # stripping (a no-op for it) keeps every parameter
        legacy = src['gnb_0']
        stripped = R._strip_reserved_keys(legacy)
        assert list(stripped) == list(legacy)
        assert not R._is_per_agent_checkpoint(stripped)
        assert R._strip_reserved_keys('not-a-dict') == 'not-a-dict'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_physics_pass_loader_uses_the_tested_helpers():
    import inspect
    import run_timeline_comparison as R
    src = inspect.getsource(R.run_physics_pass)
    i_load = src.index("checkpoint = torch.load(checkpoint_path, map_location='cpu'")
    i_detect = src.index('is_per_agent = _is_per_agent_checkpoint(checkpoint)')
    assert 'checkpoint = _strip_reserved_keys(checkpoint)' in src[i_load:i_detect]
    assert '_per_agent_source(checkpoint, aid,' in src[i_detect:]


_POOL_CALLS = {'Pool', 'ProcessPoolExecutor'}


def _call_name(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _is_cap_call(node):
    return isinstance(node, ast.Call) and _call_name(node) == 'pool_workers'


def test_every_process_pool_is_capped():
    files = [p for p in glob.glob(os.path.join(ROOT, '**', '*.py'),
                                  recursive=True)
             if os.sep + 'tests' + os.sep not in p
             and os.sep + 'tools' + os.sep not in p
             and os.sep + 'output' + os.sep not in p]
    n_pools = 0
    bad = []
    for path in files:
        try:
            tree = ast.parse(open(path, encoding='utf-8').read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        capped_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_cap_call(node.value):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        capped_names.add(t.id)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _call_name(node) in _POOL_CALLS):
                continue
            n_pools += 1
            arg = None
            for kw in node.keywords:
                if kw.arg in ('processes', 'max_workers'):
                    arg = kw.value
            if arg is None and node.args:
                arg = node.args[0]
            ok = (arg is not None and (_is_cap_call(arg) or (
                isinstance(arg, ast.Name) and arg.id in capped_names)))
            if not ok:
                bad.append('%s:%d' % (os.path.relpath(path, ROOT), node.lineno))
    assert n_pools >= 4, n_pools     # harness x2, training driver, main.py
    assert not bad, 'process pools not sized via pool_workers: %s' % bad


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
