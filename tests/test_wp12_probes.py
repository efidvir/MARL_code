"""WP12: the two pre-freeze diagnostics.

(a) the UE-transit probe's attribution core, exercised with the exact
    iteration pattern of Simulator._forward_traffic.consume_path;
(b) the SDN-discovery predictor, and the real SDNController with a tiny
    fake topology.  Until D12.2 these tests reproduced the defect (a
    fragment without an odd cycle was never discovered, and a stuck
    fragment starved every fragment after it); since D12.2 (D12-FIXES,
    docs/DECISIONS.md D12F.4) they are INVERTED as docs/TRACKB_HANDOFF.md
    7.2 acceptance item 3 asks: a path is discovered in ecc(leader) rounds
    from its end node, and fragments discover in parallel;
(c) the control-plane UE stitching (review of D12.2): the probe's infra-only
    decomposition and its 'infra-only' fragment model separate a UE-stitched
    fragment, and since D12.2 the real SDNController does too -- two
    fragments, the UE a leaf of both, the cross flows not programmed, and
    the probe's decomposition of each live fragment equal to the fragment.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load(name):
    path = os.path.join(ROOT, 'diag', name + '.py')
    spec = importlib.util.spec_from_file_location('diag_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── (a) UE-transit probe core ──────────────────────────────────────────
class _FakeNode:
    def __init__(self, nid):
        self.id = nid
        self.calls = []

    def update_energy(self, traffic_load, control_bytes):
        self.calls.append((traffic_load, control_bytes))


def _consume(mod, nodes, path, volume):
    """The part of consume_path the probe listens to: walk the delivering
    path with `for n_id in path_nodes` and call update_energy on each."""
    hook = mod.make_update_energy_hook(_FakeNode.update_energy)
    for n_id in path:
        hook(nodes[n_id], volume, 0)


def test_ue_probe_attributes_volume_to_the_walked_path():
    mod = _load('ue_transit_probe')
    S = mod.S
    nodes = {n: _FakeNode(n) for n in
             ('UE_1', 'O-RU_1', 'UE_7', 'O-RU_2', 'UE_2', 'EdgeUPF_1',
              'O-DU_1')}
    S.reset_pass()
    S.active = True
    hook = mod.make_update_energy_hook(_FakeNode.update_energy)
    # 1. u2u flow over a UE-transit path (mimic: apply_marl_policy iterates
    #    the path once, consume_path iterates it for credit and for energy)
    p1 = mod.TaggedPath(['UE_1', 'O-RU_1', 'UE_7', 'O-RU_2', 'UE_2'])
    for _ in p1:                     # apply_marl_policy walk (no energy)
        pass
    for _ in p1:                     # credit walk (no energy)
        pass
    for n_id in p1:                  # energy walk
        hook(nodes[n_id], 12.5, 0)
    # 2. general traffic: the same path object consumed twice (two classes)
    p2 = mod.TaggedPath(['UE_2', 'O-RU_2', 'EdgeUPF_1'])
    for n_id in p2:
        hook(nodes[n_id], 3.0, 0)
    for n_id in p2:
        hook(nodes[n_id], 4.0, 0)
    # 3. a path that delivered nothing: walked by apply_marl_policy only
    p3 = mod.TaggedPath(['UE_1', 'O-RU_1', 'O-DU_1'])
    for _ in p3:
        pass
    # 4. a single-node plain-list path ([sink]) right after it: must not
    #    open a record on p3
    hook(nodes['EdgeUPF_1'], 9.0, 0)
    c = mod.classify_records(S.records, lambda n: n.startswith('UE_'))
    S.active = False
    assert c['n'] == {'u2u': 1, 'gen': 2, 'infra': 0}, c['n']
    assert abs(c['vol']['u2u'] - 12.5) < 1e-9
    assert abs(c['transit']['u2u'] - 12.5) < 1e-9
    assert abs(c['vol']['gen'] - 7.0) < 1e-9
    assert c['transit']['gen'] == 0.0
    assert c['transit_ues'] == {'UE_7'}
    assert c['incomplete'] == 0
    assert c['examples'] == [['UE_1', 'O-RU_1', 'UE_7', 'O-RU_2', 'UE_2']]
    # the original update_energy still ran for every call
    assert nodes['UE_7'].calls == [(12.5, 0)]
    assert nodes['EdgeUPF_1'].calls == [(3.0, 0), (4.0, 0), (9.0, 0)]


def test_ue_probe_inactive_outside_forwarding_and_incomplete_walk_flagged():
    mod = _load('ue_transit_probe')
    S = mod.S
    nodes = {n: _FakeNode(n) for n in ('UE_1', 'O-RU_1', 'UE_2')}
    S.reset_pass()
    S.active = False
    hook = mod.make_update_energy_hook(_FakeNode.update_energy)
    p = mod.TaggedPath(['UE_1', 'O-RU_1', 'UE_2'])
    for n_id in p:
        hook(nodes[n_id], 1.0, 0)
    assert S.records == {}
    S.active = True
    for n_id in p:
        if n_id == 'UE_2':
            break                    # walk abandoned half-way
        hook(nodes[n_id], 1.0, 0)
    c = mod.classify_records(S.records, lambda n: n.startswith('UE_'))
    S.active = False
    assert c['incomplete'] == 1 and c['n']['u2u'] == 1


# ── (b) SDN discovery: predictor ───────────────────────────────────────
def _adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def test_sdn_predictor_bipartite_fragment_never_reaches_fixed_point():
    mod = _load('sdn_discovery_probe')
    # A-B-C path with one UE on each end: no odd cycle
    adj = _adj([('A', 'B'), ('B', 'C'), ('u1', 'A'), ('u2', 'C')])
    nb = lambda n: sorted(adj.get(n, ()))
    members = set(adj)
    rounds, sizes = mod.lldp_rounds_to_fixed_point(nb, 'A', members, 50)
    assert rounds is None
    assert sizes[-4:] == [2, 3, 2, 3] or sizes[-4:] == [3, 2, 3, 2], sizes
    assert mod.is_bipartite_component(members, nb)
    # the same fragment with a chord A-C (triangle): fixed point reached
    adj2 = _adj([('A', 'B'), ('B', 'C'), ('A', 'C'), ('u1', 'A'), ('u2', 'C')])
    nb2 = lambda n: sorted(adj2.get(n, ()))
    rounds2, sizes2 = mod.lldp_rounds_to_fixed_point(nb2, 'A', set(adj2), 50)
    assert rounds2 is not None and sizes2[-1] == 5
    assert not mod.is_bipartite_component(set(adj2), nb2)


def test_sdn_predictor_first_stuck_fragment_starves_the_rest():
    mod = _load('sdn_discovery_probe')
    edges = [('A', 'B'), ('B', 'C'), ('u1', 'A'), ('u2', 'C'),        # bipartite
             ('D', 'E'), ('E', 'F'), ('F', 'D'), ('u3', 'D'), ('u4', 'F')]  # odd
    adj = _adj(edges)
    nb = lambda n: sorted(adj.get(n, ()))
    deg = lambda n: len(adj.get(n, ()))
    is_ue = lambda n: n.startswith('u')
    order = ['A', 'B', 'C', 'u1', 'u2', 'D', 'E', 'F', 'u3', 'u4']
    frags = mod.sdn_fragments(order, is_ue, nb, deg)
    assert [f['leader'] for f in frags] == ['A', 'D']
    v = mod.predict(frags, nb, [('u1', 'u2'), ('u3', 'u4')])
    assert v['stuck_fid'] == 0 and v['flows_programmable'] == 0
    assert v['never_recovers'] is True
    assert [f['status'] for f in frags] == ['never', 'starved']
    # odd-cycle fragment first: it completes and its flow is programmable
    frags2 = mod.sdn_fragments(order[5:] + order[:5], is_ue, nb, deg)
    v2 = mod.predict(frags2, nb, [('u1', 'u2'), ('u3', 'u4')])
    assert v2['stuck_fid'] == 1 and v2['flows_programmable'] == 1
    assert v2['never_recovers'] is False
    assert [f['status'] for f in frags2] == ['done', 'never']


# ── (b) SDN discovery: the real controller on a fake topology ──────────
class _NT:
    def __init__(self, v):
        self.value = v


class _Node:
    def __init__(self, t):
        self.node_type = _NT(t)
        self.is_survivor = True


class _Link:
    def __init__(self, a, b):
        self.endpoints = (a, b)
        self.is_up = True


class _Topo:
    def __init__(self):
        self.nodes = {}
        self.links = {}

    def is_ue_node(self, nid):
        n = self.nodes.get(nid)
        return bool(n) and n.node_type.value == 'UE'


class _Ev:
    def __init__(self, tick, et):
        self.tick = tick
        self.event_type = et


class _Scen:
    def __init__(self, events):
        self.events = events


class _Sim:
    def __init__(self, topo, flows, severance_tick):
        self.topology = topo
        self.ue_to_ue_flows = flows
        self.scenario = _Scen([_Ev(severance_tick, 'sever_core')])
        self.current_tick = 0


def _fake_sim(fragments, flows, severance_tick=200):
    """fragments: list of (infra ids, edges among them, {ue: anchor})."""
    t = _Topo()
    for infra, edges, ues in fragments:
        for n in infra:
            t.nodes[n] = _Node('O-RU')
        for u in ues:
            t.nodes[u] = _Node('UE')
        for a, b in edges:
            t.links['l_%s_%s' % (a, b)] = _Link(a, b)
        for u, anchor in ues.items():
            t.links['uu_%s' % u] = _Link(u, anchor)
    return _Sim(t, flows, severance_tick)


def _run_ctrl(sim, last_tick):
    import run_timeline_comparison as R
    ctrl = R.SDNController(sim)
    hist = []
    for tick in range(ctrl.severance_tick, last_tick + 1):
        ctrl.step(tick)
        hist.append((tick, ctrl.phase,
                     {fid: len(fr) for fid, fr in ctrl.discovery_frontier.items()},
                     len(ctrl.programmed_flows)))
    return ctrl, hist


def _first_tick(hist, n_programmed):
    """First tick at which at least n_programmed flows are programmed."""
    return next(t for t, _ph, _fr, n in hist if n >= n_programmed)


BIPARTITE = (['A', 'B', 'C'], [('A', 'B'), ('B', 'C')], {'u1': 'A', 'u2': 'C'})
ODD_CYCLE = (['D', 'E', 'F'], [('D', 'E'), ('E', 'F'), ('F', 'D')],
             {'u3': 'D', 'u4': 'F'})


def test_sdn_controller_discovers_a_bipartite_path_in_ecc_rounds():
    """INVERTED by D12.2 (it asserted the defect: the fixed-point test never
    ended on the path A-B-C).  The cumulative BFS discovers the path from
    its end node A (the leader: all three switches have up-link degree 2,
    A first) in ecc(A) = 2 rounds, completes in the third (which discovers
    nothing), and programs the flow in that same tick:
    first programming = election done + ecc + 1."""
    import run_timeline_comparison as R
    sim = _fake_sim([BIPARTITE], [('u1', 'u2')])
    ctrl, hist = _run_ctrl(sim, R.SEVERANCE_TICK + 400)
    assert ctrl.fragment_leaders == {0: 'A'}, ctrl.fragment_leaders
    assert ctrl.fragment_switches == {0: {'A', 'B', 'C'}}
    assert ctrl.fragment_members[0] == {'A', 'B', 'C', 'u1', 'u2'}
    assert ctrl.phase == 'operational', ctrl.phase
    assert ctrl.programmed_flows == {('u1', 'u2')}
    assert ctrl.discovery_done.get(0) is True
    assert ctrl.discovered[0] == {'A', 'B', 'C'}
    e = ctrl.election_done_tick[0]
    sizes = [h[2].get(0) for h in hist if e <= h[0] <= e + 3]
    assert sizes == [1, 1, 1, 1], sizes      # {A}, {B}, {C}, done
    assert _first_tick(hist, 1) == e + 2 + 1, (_first_tick(hist, 1), e)


def test_sdn_controller_discovers_a_fragment_with_an_odd_cycle():
    import run_timeline_comparison as R
    sim = _fake_sim([ODD_CYCLE], [('u3', 'u4')])
    ctrl, hist = _run_ctrl(sim, R.SEVERANCE_TICK + 400)
    assert ctrl.phase == 'operational', ctrl.phase
    assert ctrl.programmed_flows == {('u3', 'u4')}
    assert ctrl.discovery_done.get(0) is True
    first_prog = next(t for t, ph, _f, _n in hist if ph in ('programming',
                                                      'operational'))
    assert first_prog < R.SEVERANCE_TICK + 30, first_prog


def test_sdn_controller_fragments_discover_in_parallel_and_none_starves():
    """INVERTED by D12.2 (it asserted that a stuck first fragment starved
    the second through the shared round timer).  Each fragment has its own
    round timer: in either node order both complete, each after its own
    ecc + 1 rounds from the same election tick, and both flows are
    programmed."""
    import run_timeline_comparison as R
    for order in ([BIPARTITE, ODD_CYCLE], [ODD_CYCLE, BIPARTITE]):
        sim = _fake_sim(order, [('u1', 'u2'), ('u3', 'u4')])
        ctrl, hist = _run_ctrl(sim, R.SEVERANCE_TICK + 400)
        assert ctrl.phase == 'operational', ctrl.phase
        assert ctrl.programmed_flows == {('u1', 'u2'), ('u3', 'u4')}
        assert all(ctrl.discovery_done.get(f) for f in (0, 1))
        e = set(ctrl.election_done_tick.values())
        assert len(e) == 1, ctrl.election_done_tick       # in parallel
        e = e.pop()
        # the path completes at e + 3 (ecc 2), the triangle at e + 2 (ecc
        # 1); 3 flows per tick, so both are programmed by e + 3
        assert _first_tick(hist, 1) == e + 2, (_first_tick(hist, 1), e)
        assert _first_tick(hist, 2) == e + 3, (_first_tick(hist, 2), e)


# ── (b) the recommended cumulative-BFS iteration terminates everywhere ──
def test_sdn_bfs_rounds_terminates_on_bipartite_and_odd_fragments():
    mod = _load('sdn_discovery_probe')
    adj = _adj([('A', 'B'), ('B', 'C'), ('u1', 'A'), ('u2', 'C')])
    nb = lambda n: sorted(adj.get(n, ()))
    rounds, reached = mod.bfs_rounds(nb, 'A', set(adj))
    assert (rounds, reached) == (3, 5), (rounds, reached)   # A | B,u1 | C | u2
    # from the centre the eccentricity is 2
    assert mod.bfs_rounds(nb, 'B', set(adj)) == (2, 5)
    # the iteration in use never terminates on the same fragment
    assert mod.lldp_rounds_to_fixed_point(nb, 'A', set(adj), 50)[0] is None
    # a node outside `members` is not reached
    assert mod.bfs_rounds(nb, 'A', {'A', 'B', 'C'}) == (2, 3)
    # predict() carries the value per eligible fragment
    deg = lambda n: len(adj.get(n, ()))
    frags = mod.sdn_fragments(['A', 'B', 'C', 'u1', 'u2'],
                              lambda n: n.startswith('u'), nb, deg)
    mod.predict(frags, nb, [('u1', 'u2')])
    # every infra node has up-link degree 2, so the lexicographic tie-break
    # of the controller elects 'A' (an end of the path: eccentricity 3)
    assert frags[0]['leader'] == 'A'
    assert frags[0]['rounds'] is None and frags[0]['bfs_rounds'] == 3


# ── (a) report: per-phase decomposition of the delivered u2u volume ────
def _synthetic_run(ticks):
    """A probe log where a bridge is up on every post-event tick, absent
    from the routing view on even ticks; transit volume 200 on those ticks
    and 10 otherwise; total u2u 1000 on every tick."""
    log = {'tick': [], 'island': [], 'believed': [], 'relay_up': [],
           'relay_not_in_view': [], 'dual_ue': [], 'u2u': [],
           'u2u_transit': [], 'n_u2u': [], 'n_u2u_transit': [], 'gen': [],
           'gen_transit': [], 'n_gen': [], 'n_gen_transit': [], 'infra': [],
           'infra_transit': [], 'n_transit_ues': [], 'incomplete': [],
           'engine_u2u': [], 'flows': []}
    for t in range(ticks):
        post = t >= 4
        niv = post and t % 2 == 0
        log['tick'].append(t)
        log['island'].append(int(post))
        log['believed'].append(int(niv))
        log['relay_up'].append(int(post))
        log['relay_not_in_view'].append(int(niv))
        log['dual_ue'].append(70)
        log['u2u'].append(1000.0 if post else 800.0)
        log['u2u_transit'].append((200.0 if niv else 10.0) if post else 5.0)
        log['n_u2u'].append(200)
        log['n_u2u_transit'].append(20)
        log['gen'].append(500.0)
        log['gen_transit'].append(0.0)
        log['n_gen'].append(100)
        log['n_gen_transit'].append(0)
        log['infra'].append(0.0)
        log['infra_transit'].append(0.0)
        log['n_transit_ues'].append(3)
        log['incomplete'].append(0)
        log['engine_u2u'].append(log['u2u'][-1])
        log['flows'].append(222)
    return {'ue_transit_probe': log,
            'ue_transit_probe_examples': [(4, ['UE_1', 'O-RU_1', 'UE_9',
                                               'O-RU_2', 'UE_2'])],
            'ue_transit_probe_examples_post': [
                (6, 'not-in-view', ['UE_3', 'O-RU_1', 'UE_9', 'O-RU_2',
                                    'UE_4']),
                (7, 'in-view', ['UE_3', 'O-RU_1', 'UE_9', 'O-RU_3', 'UE_8',
                                'O-RU_2', 'UE_4'])],
            'ue_transit_probe_meta': {'transit_ues': ['UE_9', 'UE_8'],
                                      'summary_errors': 0}}


def test_ue_probe_report_phase_decomposition_and_phase_lock_line():
    mod = _load('ue_transit_probe')
    s = mod.summarise_pass(_synthetic_run(24), 'marl_static', 'recovery',
                           82, 4)
    a = s['phase']['bridge_not_in_view']
    b = s['phase']['bridge_in_view']
    assert (a['ticks'], b['ticks']) == (10, 10)
    assert abs(a['transit_mean'] - 200.0) < 1e-9
    assert abs(a['nontransit_mean'] - 800.0) < 1e-9
    assert abs(b['nontransit_mean'] - 990.0) < 1e-9
    assert (a['high_transit_ticks'], b['high_transit_ticks']) == (10, 0)
    assert s['post_ticks'] == 20 and s['u2u_transit_post_ticks'] == 20
    assert abs(s['u2u_transit_pre_total'] - 20.0) < 1e-9
    assert s['engine_u2u_max_mismatch'] == 0.0
    assert s['n_transit_ues'] == 2
    # the report renders the decomposition and the phase-lock line
    import pickle
    import tempfile
    P = {'provenance': {'severance_tick': 4, 'total_ticks': 24,
                        'tree_head': 'test', 'pythonhashseed': '0'},
         'recovery_multi': {'marl_static': {82: _synthetic_run(24)}},
         'rescue_multi': {}}
    base = os.path.join(ROOT, 'output', 'tmp_tests')   # inside the tree
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(prefix='wp12_report_', dir=base)
    pk = os.path.join(d, 'p.pkl')
    with open(pk, 'wb') as f:
        pickle.dump(P, f)
    rows = mod.report(pk, out_dir=d)
    assert len(rows) == 1
    text = ' '.join(open(os.path.join(d, 'p_report.txt')).read().split())
    assert 'gap +0.0 = transit +190.0 + non-transit -190.0' in text, text
    assert 'non-transit 800.0 (>=100: 10)' in text, text
    assert 'transit 200.0 non-transit 800.0' in text, text
    # post-event examples are reported with their phase, and the shape
    # statistic distinguishes the two budgets
    assert 't= 6 not-in-view: UE_3 > O-RU_1 > UE_9 > O-RU_2 > UE_4' in text
    assert 't= 7 in-view : UE_3 > O-RU_1 > UE_9 > O-RU_3 > UE_8' in text
    assert s['interior_ues_per_example'] == {'first_ticks': {1: 1},
                                             'post_event': {1: 1, 2: 1}}
    # the per-pass histogram line carries the source of the count
    assert ('82: {1: 1} [name-fallback] | {1: 1, 2: 1} [name-fallback]') in text, text
    assert 'sources over 1 passes: name-fallback; count-length-mismatch anomalies: 0' in text, text


def test_ue_probe_post_event_example_budgets():
    mod = _load('ue_transit_probe')
    S = mod.ProbeState()
    # the interior node is a RESCUE UE (the rescue_ops passes carry 20 of
    # them, arriving at the rescaled RESCUE_ARRIVAL_TICK); the probe
    # classifies it with `sim.topology.is_ue_node`, which matches
    # 'Rescue_UE_*' through its '_UE_' fallback
    path = ['UE_1', 'O-RU_1', 'Rescue_UE_5', 'O-RU_2', 'UE_2']
    # pre-event ticks fill only the first-ticks budget
    for t in range(10):
        ph = mod.record_examples(S, t, 0, 0, 0, [path, path, path])
        assert ph == 'pre-event'
    assert len(S.examples) == mod.EXAMPLE_BUDGET and S.examples_post == []
    # post-event: the not-in-view budget is reserved even when the other
    # phases come first and would fill everything
    for t in range(200, 240):
        mod.record_examples(S, t, 1, 1, 0, [path])          # in-view
    assert len(S.examples_post) == mod.EXAMPLE_BUDGET_POST
    assert all(ph == 'in-view' for _t, ph, _p in S.examples_post)
    for t in range(240, 260):
        ph = mod.record_examples(S, t, 1, 1, 1, [path])     # not-in-view
        assert ph == 'not-in-view'
    niv = [e for e in S.examples_post if e[1] == 'not-in-view']
    assert len(niv) == mod.EXAMPLE_BUDGET_NIV and niv[0][0] == 240
    assert len(S.examples_post) == (mod.EXAMPLE_BUDGET_POST
                                    + mod.EXAMPLE_BUDGET_NIV)
    assert mod.record_examples(S, 300, 1, 0, 0, [path]) == 'no-bridge'
    assert len(S.examples_post) == 24                     # budgets full
    assert len(S.examples) == mod.EXAMPLE_BUDGET          # unchanged
    # the count is recorded WITH the example, by the predicate that
    # classified the tick's records
    assert S.examples_post_interior_ues == [1] * 24
    assert S.example_interior_ues == [1] * mod.EXAMPLE_BUDGET
    assert mod.interior_ue_histogram(
        S.examples_post, counts=S.examples_post_interior_ues) == {1: 24}
    # the report's old predicate, `str(n).startswith('UE')`, is NARROWER
    # than the one that recorded the example and loses the rescue UE: it
    # would have reported a spurious {0: 24} bucket for paths the per-tick
    # transit volume counts as UE-transit
    assert mod.interior_ue_histogram(
        S.examples_post, lambda n: str(n).startswith('UE')) == {0: 24}
    # the fallback for pickles written before the counts existed is
    # `is_ue_node`'s own name rule, which does not lose it
    assert mod.interior_ue_histogram(S.examples_post) == {1: 24}
    assert mod.name_is_ue('Rescue_UE_5') and mod.name_is_ue('UE_9')
    assert not mod.name_is_ue('O-RU_2') and not mod.name_is_ue('UEX')


def test_ue_probe_interior_ue_histogram_uses_the_recording_predicate():
    """DEFECT: `summarise_pass` classified interior nodes with
    `str(n).startswith('UE')`, a narrower predicate than the
    `sim.topology.is_ue_node` the probe used when it recorded the example,
    so a path recorded as UE-transit because of an interior rescue UE was
    reported as having 0 interior UEs."""
    mod = _load('ue_transit_probe')
    run = _synthetic_run(24)
    rescue = ['UE_3', 'O-RU_1', 'Rescue_UE_7', 'O-RU_2', 'UE_4']
    run['ue_transit_probe_examples'] = [(0, list(rescue))]
    run['ue_transit_probe_examples_post'] = [(6, 'not-in-view', list(rescue)),
                                             (7, 'in-view', list(rescue))]
    # a pass whose probe stored the counts: they are used as they stand
    run['ue_transit_probe_meta'] = dict(run['ue_transit_probe_meta'],
                                        example_interior_ues=[1],
                                        example_post_interior_ues=[1, 1])
    s = mod.summarise_pass(run, 'marl_static', 'rescue_ops', 87, 0)
    assert s['interior_ues_per_example'] == {'first_ticks': {1: 1},
                                             'post_event': {1: 2}}
    # a pass recorded before the counts existed: the fallback must still be
    # as wide as is_ue_node
    legacy = dict(run)
    legacy['ue_transit_probe_meta'] = {'transit_ues': ['Rescue_UE_7'],
                                       'summary_errors': 0}
    s2 = mod.summarise_pass(legacy, 'marl_static', 'rescue_ops', 87, 0)
    assert s2['interior_ues_per_example'] == {'first_ticks': {1: 1},
                                              'post_event': {1: 2}}
    # a stored count that does not line up with the examples is ignored
    # rather than mis-zipped
    bad = dict(run)
    bad['ue_transit_probe_meta'] = dict(run['ue_transit_probe_meta'],
                                        example_post_interior_ues=[1])
    s3 = mod.summarise_pass(bad, 'marl_static', 'rescue_ops', 87, 0)
    assert s3['interior_ues_per_example']['post_event'] == {1: 2}
    # and the probe writes the counts into the pickle meta it produces
    S = mod.ProbeState()
    mod.record_examples(S, 5, 1, 1, 1, [rescue])
    assert S.example_interior_ues == [1] and S.examples_post_interior_ues == [1]


# ── (c) control-plane UE stitching ─────────────────────────────────────
# two bipartite infra components (paths A-B-C and D-E-F) joined ONLY by
# the dual-anchored UE u2 (A and F); u1 hangs on B, u3 on E, u4 on D
STITCH_EDGES = [('A', 'B'), ('B', 'C'), ('D', 'E'), ('E', 'F'),
                ('u1', 'B'), ('u2', 'A'), ('u2', 'F'), ('u3', 'E'),
                ('u4', 'D')]
STITCH_ORDER = ['A', 'B', 'C', 'u1', 'u2', 'D', 'E', 'F', 'u3', 'u4']
STITCH_FLOWS = [('u1', 'u2'),   # intra (component 0 via u2's anchor A)
                ('u2', 'u3'),   # intra (component 1 via u2's anchor F)
                ('u1', 'u3'),   # cross: no infrastructure path
                ('u4', 'u1')]   # cross


def test_sdn_infra_decomposition_separates_a_ue_stitched_fragment():
    mod = _load('sdn_discovery_probe')
    adj = _adj(STITCH_EDGES)
    nb = lambda n: sorted(adj.get(n, ()))
    deg = lambda n: len(adj.get(n, ()))
    is_ue = lambda n: n.startswith('u')
    dec = mod.infra_decomposition(STITCH_ORDER, is_ue, nb, STITCH_FLOWS)
    assert dec['comp_sizes'] == [3, 3]
    assert dec['comp_bipartite'] == [True, True]
    assert dec['ues_multi_anchored'] == ['u2']
    assert dec['ue_comps'] == {'u1': [0], 'u2': [0, 1], 'u3': [1], 'u4': [1]}
    assert dec['flows'] == {'intra': 2, 'cross': 2, 'unanchored': 0}
    assert dec['flows_intra_per_comp'] == {0: 1, 1: 1}
    # the controller as it is: ONE fragment of 6 infra / 4 UE
    st = mod.sdn_fragments(STITCH_ORDER, is_ue, nb, deg, model='ue-stitched')
    assert [(f['n_infra'], f['n_ue']) for f in st] == [(6, 4)]
    v = mod.predict(st, nb, STITCH_FLOWS)
    assert st[0]['bipartite'] and v['never_recovers']
    assert st[0]['flows_intra'] == 4          # the two cross flows included
    # the leaf model: two fragments, u2 a leaf of both, cross flows in none
    inf = mod.sdn_fragments(STITCH_ORDER, is_ue, nb, deg, model='infra-only')
    assert [(f['n_infra'], f['n_ue']) for f in inf] == [(3, 2), (3, 3)]
    assert 'u2' in inf[0]['members'] and 'u2' in inf[1]['members']
    assert inf[0]['walk'] == {'A', 'B', 'C'} and inf[1]['walk'] == {'D', 'E', 'F'}
    assert [f['leader'] for f in inf] == ['B', 'E']   # up-link degree 3 each
    v2 = mod.predict(inf, nb, STITCH_FLOWS)
    assert [f['flows_intra'] for f in inf] == [1, 1]
    # still bipartite without the UEs: the fixed-point test never ends
    assert [f['bipartite'] for f in inf] == [True, True]
    assert v2['never_recovers'] and v2['stuck_fid'] == 0
    # the cumulative iteration terminates: ecc from the leaders, and only
    # the two flows with an infrastructure path are programmable
    v3 = mod.predict(inf, nb, STITCH_FLOWS, iteration='bfs')
    assert [f['status'] for f in inf] == ['done', 'done']
    assert [f['bfs_rounds'] for f in inf] == [1, 1]   # B and E are centres
    assert v3['flows_programmable'] == 2 and not v3['never_recovers']
    # a flow that is intra in two fragments is programmable once
    both = [('u2', 'u2x')]
    adj2 = _adj(STITCH_EDGES + [('u2x', 'C'), ('u2x', 'D')])
    nb2 = lambda n: sorted(adj2.get(n, ()))
    inf2 = mod.sdn_fragments(STITCH_ORDER + ['u2x'], is_ue, nb2, deg,
                             model='infra-only')
    v4 = mod.predict(inf2, nb2, both, iteration='bfs')
    assert [f['flows_intra'] for f in inf2] == [1, 1]
    assert v4['flows_programmable'] == 1


def test_sdn_controller_keeps_ue_stitched_components_apart():
    """INVERTED by D12.2 (it asserted that the dual-anchored UE u2 stitched
    the two components into ONE fragment whose cross flows were programmed).
    Now: two fragments, {A, B, C} and {D, E, F}; u2 a leaf of both and
    walked by neither; the two flows with an infrastructure path programmed
    and the two cross flows not; and the probe's infra-only decomposition
    of each live fragment is exactly that fragment (one component, no UE
    anchored in two components inside it, no flow without a path)."""
    import run_timeline_comparison as R
    t = _Topo()
    for n in ('A', 'B', 'C', 'D', 'E', 'F'):
        t.nodes[n] = _Node('O-RU')
    for u in ('u1', 'u2', 'u3', 'u4'):
        t.nodes[u] = _Node('UE')
    for a, b in STITCH_EDGES + [('A', 'C')]:      # A-B-C-A: odd cycle
        t.links['l_%s_%s' % (a, b)] = _Link(a, b)
    sim = _Sim(t, list(STITCH_FLOWS), R.SEVERANCE_TICK)
    ctrl, hist = _run_ctrl(sim, R.SEVERANCE_TICK + 400)
    assert ctrl.fragment_switches == {0: {'A', 'B', 'C'}, 1: {'D', 'E', 'F'}}
    assert ctrl.fragment_members == {0: {'A', 'B', 'C', 'u1', 'u2'},
                                     1: {'D', 'E', 'F', 'u2', 'u3', 'u4'}}
    assert ctrl.anchors['u2'] == ('A', 'F'), ctrl.anchors
    assert ctrl.phase == 'operational', ctrl.phase
    assert ctrl.programmed_flows == {('u1', 'u2'), ('u2', 'u3')}
    for flow in (('u1', 'u3'), ('u4', 'u1')):
        assert flow not in ctrl.programmed_flows, flow
    mod = _load('sdn_discovery_probe')
    table = mod._fragment_table(ctrl)
    assert len(table) == 2
    for row, sizes in zip(table, ([3], [3])):
        assert row['infra_comp_sizes'] == sizes, row
        assert row['n_ues_multi_anchored'] == 0, row
        assert row['flows_cross'] == 0 and row['flows_unanchored'] == 0, row
        assert row['flows_with_infra_path'] == row['flows_intra'] == 1, row
        assert row['infra_comp_leaders'] == [row['leader']], row


def test_sdn_report_omits_the_ecc_clause_when_the_pickle_lacks_it():
    mod = _load('sdn_discovery_probe')
    import pickle
    import tempfile
    frag_old = {'fid': 0, 'n_members': 4, 'n_infra': 3, 'n_ue': 1,
                'leader': 'A', 'election_done_tick': 215, 'bipartite': True,
                'predicted_rounds': None, 'flows_intra': 1}
    frag_new = dict(frag_old, bfs_rounds=2, infra_comp_sizes=[3],
                    infra_comp_bipartite=[True], infra_comp_ecc=[2],
                    n_ues_multi_anchored=0, flows_with_infra_path=1,
                    flows_cross=0, flows_unanchored=0)
    # an isolated single switch: no election, so no leader, no walk and no
    # cumulative-BFS number.  The guard used to test only for the KEY, so
    # this rendered as "predicted rounds None (cumulative BFS would need
    # None)" -- the placeholder in a different form.
    frag_none = {'fid': 1, 'n_members': 1, 'n_infra': 1, 'n_ue': 0,
                 'leader': None, 'election_done_tick': None,
                 'bipartite': True, 'predicted_rounds': None,
                 'bfs_rounds': None, 'flows_intra': 0}
    log = [{'tick': t, 'phase': 'discovery', 'programmed': 0, 'queued': 0,
            'done': 0, 'frontier': {0: 2 if t % 2 else 3}}
           for t in range(200, 260)]
    run_old = {'delivered': [0.0] * 260,
               'sdn_probe': {'fragments': [frag_old], 'fragments_tick': 210,
                             'log': log,
                             'frontier_hist': {0: [r['frontier'][0]
                                                   for r in log]}}}
    run_new = dict(run_old, sdn_probe=dict(run_old['sdn_probe'],
                                           fragments=[frag_new, frag_none]))
    P = {'provenance': {'severance_tick': 200, 'total_ticks': 260,
                        'tree_head': 'test', 'pythonhashseed': '0'},
         'recovery_multi': {'sdn': {87: run_old, 98: run_new}},
         'rescue_multi': {}}
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(prefix='wp12_sdnreport_', dir=base)
    pk = os.path.join(d, 'p.pkl')
    with open(pk, 'wb') as f:
        pickle.dump(P, f)
    rows = mod.report(pk, out_dir=d)
    assert len(rows) == 2
    text = open(os.path.join(d, 'p_report.txt')).read()
    assert 'n/a' not in text and 'predates' not in text, text
    l87 = [l for l in text.splitlines() if 'fid 0' in l]
    assert len(l87) == 2
    assert 'cumulative BFS' not in l87[0] and '<- oscillating' in l87[0]
    assert '(cumulative BFS would need 2)' in l87[1]
    assert 'infra-only decomposition: components [3], bipartite T' in text
    # the would-be leaders are printed only when the pickle carries them
    assert "ecc of their would-be leaders [2]" in text
    frag_led = dict(frag_new, infra_comp_leaders=['A'])
    run_led = dict(run_old, sdn_probe=dict(run_old['sdn_probe'],
                                           fragments=[frag_led]))
    P2 = dict(P, recovery_multi={'sdn': {87: run_led}})
    d2 = tempfile.mkdtemp(prefix='wp12_sdnreport2_', dir=base)
    pk2 = os.path.join(d2, 'p.pkl')
    with open(pk2, 'wb') as f:
        pickle.dump(P2, f)
    mod.report(pk2, out_dir=d2)
    t2 = open(os.path.join(d2, 'p_report.txt')).read()
    assert "ecc of their would-be leaders ['A'] [2]" in t2, t2
    # the leaderless fragment: no placeholder anywhere in the report
    l1 = [l for l in text.splitlines() if 'fid 1' in l]
    assert len(l1) == 1, l1
    assert 'leader None' in l1[0] and 'no election' in l1[0], l1[0]
    assert 'predicted rounds' not in l1[0], l1[0]
    # the placeholder itself is gone from the whole report
    assert 'would need None' not in text, text
    # a leader WITH a None fixed-point prediction still prints one (87's
    # oscillating fragment, and fid 0 of every Scenario-A pass of the real
    # report): the suppression is about the missing election, not about a
    # None value, and the cumulative-BFS clause is still attached when the
    # pickle carries a number
    assert 'predicted rounds None' in l87[0], l87[0]
    assert 'predicted rounds None (cumulative BFS would need 2)' in l87[1]


# ── (d) the predictor's own safety nets ────────────────────────────────
def test_sdn_predict_does_not_report_a_disconnected_walk_as_done():
    """DEFECT: with iteration='bfs', `predict` set `completes = True`
    unconditionally and discarded the reached count, so a fragment whose
    walk set is not connected would be reported status='done' and ALL of
    its flows counted programmable although the BFS never reached part of
    it.  Neither of today's two fragment models can build such a walk, so
    nothing in the committed evidence would have caught it."""
    mod = _load('sdn_discovery_probe')
    # A-B and X-Y are two separate components; P-Q-R is a third
    adj = _adj([('A', 'B'), ('X', 'Y'), ('P', 'Q'), ('Q', 'R')])
    nb = lambda n: sorted(adj.get(n, ()))
    # the BFS itself reports the shortfall
    assert mod.bfs_rounds(nb, 'A', {'A', 'B', 'X', 'Y'}) == (1, 2)
    assert mod.bfs_walk(nb, 'A', {'A', 'B', 'X', 'Y'})[1] == {'A', 'B'}
    assert mod.bfs_rounds(nb, 'A', {'A', 'B'}) == (1, 2)

    def frag(fid, members, leader):
        return {'fid': fid, 'members': set(members), 'walk': set(members),
                'leader': leader}
    # fid 0's walk is deliberately disconnected; fid 1's is not
    frags = [frag(0, ['A', 'B', 'X', 'Y'], 'A'), frag(1, ['P', 'Q', 'R'], 'P')]
    flows = [('A', 'B'), ('X', 'Y'), ('P', 'R')]
    v = mod.predict(frags, nb, flows, iteration='bfs')
    assert frags[0]['bfs_reached'] == 2 and frags[0]['walk_covered'] is False
    assert frags[0]['status'] == 'partial-walk'
    # its flows are NOT counted programmable ...
    assert v['flows_programmable'] == 1            # only ('P', 'R')
    assert v['partial_walk_fids'] == [0]
    # ... and it starves nobody: a cumulative BFS does terminate, so the
    # shared round timer is released
    assert v['stuck_fid'] is None
    assert frags[1]['status'] == 'done'
    assert frags[1]['walk_covered'] is True and frags[1]['bfs_reached'] == 3
    assert v['never_recovers'] is False
    # a connected walk is unaffected (and the 'done' path still counts)
    frags2 = [frag(0, ['P', 'Q', 'R'], 'P')]
    v2 = mod.predict(frags2, nb, flows, iteration='bfs')
    assert frags2[0]['status'] == 'done' and v2['partial_walk_fids'] == []
    assert v2['flows_programmable'] == 1 and v2['stuck_fid'] is None
    # a fragment with no election carries the fields as None, not absent
    frags3 = [frag(0, ['A', 'B'], None)]
    mod.predict(frags3, nb, flows, iteration='bfs')
    assert frags3[0]['status'] == 'no-election'
    assert frags3[0]['walk_covered'] is None and frags3[0]['bfs_reached'] is None


# a component in which the node of highest UP-LINK degree (B: 4 parallel
# links to C plus A-B = 5) is NOT the node of most distinct peers
# (A: B, D, E = 3), and the two elect different leaders with different
# eccentricities (from B: 3, from A: 2)
PARALLEL_INFRA = ['A', 'B', 'C', 'D', 'E', 'F']
PARALLEL_LINKS = ([('B', 'C')] * 4
                  + [('A', 'B'), ('A', 'D'), ('A', 'E'), ('E', 'F')])


def _parallel_sim(flows=(), severance_tick=None):
    import run_timeline_comparison as R
    t = _Topo()
    for n in PARALLEL_INFRA:
        t.nodes[n] = _Node('O-RU')
    for i, (a, b) in enumerate(PARALLEL_LINKS):
        t.links['l%02d_%s_%s' % (i, a, b)] = _Link(a, b)
    sev = R.SEVERANCE_TICK if severance_tick is None else severance_tick
    return _Sim(t, list(flows), sev)


def test_sdn_leader_rule_counts_up_links_not_distinct_peers():
    """DEFECT: `_fragment_table` elected the would-be leader of each
    infra-only component with `max(comp, key=lambda n: len(neighbors(n)))`
    -- the number of distinct peers -- while `sdn_fragments`/`static_case`
    and the engine (`run_timeline_comparison.py:2199-2203`) count up LINKS
    incident to the node.  The model PERMITS parallel links between a node
    pair, so the two rules CAN elect different leaders and report different
    eccentricities, and handoff acceptance item 1 would then have compared
    the engine against a differently-elected predictor.  This fixture is
    such a pair; no topology of this study is (0 in 20 seeds, 0 in the 40
    post-severance cases -- `tools/leader_rule_census.py`), which is why
    the divergence has to be built here to be exercised at all."""
    mod = _load('sdn_discovery_probe')
    import run_timeline_comparison as R
    sim = _parallel_sim()
    adj = _adj(PARALLEL_LINKS)
    nb = lambda n: sorted(adj.get(n, ()))
    updeg = {}
    for a, b in PARALLEL_LINKS:
        updeg[a] = updeg.get(a, 0) + 1
        updeg[b] = updeg.get(b, 0) + 1
    assert updeg == {'A': 3, 'B': 5, 'C': 4, 'D': 1, 'E': 2, 'F': 1}
    assert {n: len(adj[n]) for n in adj} == {'A': 3, 'B': 2, 'C': 1,
                                            'D': 1, 'E': 2, 'F': 1}
    # the two rules disagree here, which is the whole point of the case
    assert mod.leader_of(PARALLEL_INFRA, lambda n: updeg.get(n, 0)) == 'B'
    assert max(sorted(PARALLEL_INFRA), key=lambda n: len(nb(n))) == 'A'
    assert mod.bfs_rounds(nb, 'B', set(PARALLEL_INFRA))[0] == 3
    assert mod.bfs_rounds(nb, 'A', set(PARALLEL_INFRA))[0] == 2

    # the engine elects B
    ctrl = R.SDNController(sim)
    ctrl._compute_fragments()
    assert ctrl.fragment_members[0] == set(PARALLEL_INFRA)
    assert ctrl.fragment_leaders[0] == 'B', ctrl.fragment_leaders

    # ... and so must the probe's table of the same live fragment
    table = mod._fragment_table(ctrl)
    assert table[0]['infra_comp_sizes'] == [6]
    assert table[0]['infra_comp_leaders'] == ['B']
    assert table[0]['infra_comp_ecc'] == [3]
    assert table[0]['leader'] == 'B' and table[0]['bfs_rounds'] == 3

    # ... and so must the static predictor, so acceptance item 1 compares
    # like with like
    frags = mod.sdn_fragments(PARALLEL_INFRA, lambda n: False, nb,
                              lambda n: updeg.get(n, 0))
    assert [f['leader'] for f in frags] == ['B']
    mod.predict(frags, nb, [], iteration='bfs')
    assert frags[0]['bfs_rounds'] == 3 and frags[0]['walk_covered'] is True
    # the lexicographic tie-break of `leader_of` is the engine's
    assert mod.leader_of(['B', 'A', 'C'], lambda n: 1) == 'A'
    assert mod.leader_of([], lambda n: 1) is None


def test_sdn_predict_derives_completion_on_the_fixed_point_branch_too():
    """DEFECT (round 3): the previous round derived completion from
    the walk only on the `bfs` branch.  The DEFAULT branch,
    iteration='fixed-point', still read `completes = rounds is not None` --
    "the iteration terminated" taken to mean "the walk was covered" --
    while `predict` computed `walk_covered` three lines earlier and threw
    it away.  The default iteration is what produced four of the seven
    committed static evidence files, so the guard was absent from the path
    carrying most of the study's static evidence.

    The case below is the one from the defect report: a fragment whose walk
    is {P,Q,R,X,Y} with leader P, an odd cycle P-Q-R-P (so the fixed-point
    iteration DOES terminate) and X-Y disconnected from it.  Old
    behaviour: status 'done', 2 programmable flows -- including ('X','Y'),
    a flow in the part the discovery never reached -- and never_recovers
    False.  A partial walk must read the same under both iterations."""
    mod = _load('sdn_discovery_probe')
    adj = _adj([('P', 'Q'), ('Q', 'R'), ('R', 'P'), ('X', 'Y'),
                ('S', 'T'), ('T', 'U'), ('U', 'S')])
    nb = lambda n: sorted(adj.get(n, ()))

    def frag(fid, members, leader):
        return {'fid': fid, 'members': set(members), 'walk': set(members),
                'leader': leader}

    # the fixed point IS reached (the odd cycle) and the walk is NOT covered
    assert mod.lldp_rounds_to_fixed_point(nb, 'P', {'P', 'Q', 'R', 'X', 'Y'})[0] == 3
    assert mod.bfs_rounds(nb, 'P', {'P', 'Q', 'R', 'X', 'Y'}) == (1, 3)

    flows = [('P', 'Q'), ('X', 'Y'), ('S', 'T')]
    frags = [frag(0, ['P', 'Q', 'R', 'X', 'Y'], 'P'),
             frag(1, ['S', 'T', 'U'], 'S')]
    v = mod.predict(frags, nb, flows)                 # default: fixed-point
    assert v['iteration'] == 'fixed-point'
    assert frags[0]['rounds'] == 3                    # it terminated ...
    assert frags[0]['bfs_reached'] == 3
    assert frags[0]['walk_covered'] is False          # ... without covering
    assert frags[0]['status'] == 'partial-walk', frags[0]['status']
    assert v['partial_walk_fids'] == [0]
    # its flows are NOT programmable: only ('S','T') of fid 1 is
    assert v['flows_programmable'] == 1
    # and it starves nobody -- its round timer did stop
    assert v['stuck_fid'] is None
    assert frags[1]['status'] == 'done'

    # the same fragment under the bfs branch: the identical verdict
    frags_b = [frag(0, ['P', 'Q', 'R', 'X', 'Y'], 'P'),
               frag(1, ['S', 'T', 'U'], 'S')]
    vb = mod.predict(frags_b, nb, flows, iteration='bfs')
    assert frags_b[0]['status'] == 'partial-walk'
    assert (vb['partial_walk_fids'], vb['flows_programmable'],
            vb['stuck_fid']) == ([0], 1, None)

    # a fixed point that never terminates is still 'never', and still
    # starves the fragments after it: coverage did not replace termination
    bip = _adj([('a', 'b'), ('b', 'c'), ('c', 'd'), ('m', 'n'), ('n', 'o')])
    nb2 = lambda n: sorted(bip.get(n, ()))
    fr2 = [frag(0, ['a', 'b', 'c', 'd'], 'a'), frag(1, ['m', 'n', 'o'], 'm')]
    v2 = mod.predict(fr2, nb2, [('a', 'd'), ('m', 'o')])
    assert fr2[0]['rounds'] is None and fr2[0]['walk_covered'] is True
    assert fr2[0]['status'] == 'never' and v2['stuck_fid'] == 0
    assert fr2[1]['status'] == 'starved' and v2['flows_programmable'] == 0
    assert v2['never_recovers'] is True and v2['partial_walk_fids'] == []

    # a connected walk under fixed-point is unaffected (the committed
    # static evidence): it completes and its flows count
    fr3 = [frag(0, ['P', 'Q', 'R'], 'P')]
    v3 = mod.predict(fr3, nb, flows)
    assert fr3[0]['status'] == 'done' and fr3[0]['walk_covered'] is True
    assert v3['flows_programmable'] == 1 and v3['partial_walk_fids'] == []


def test_sdn_static_verdict_column_names_a_partial_walk():
    """DEFECT (round 3): `cmd_static` printed 'ok' in the per-row
    verdict column for a row whose fragment terminated without covering
    itself; only the trailing PARTIAL WALK line named it, so the per-row
    prediction read clean."""
    mod = _load('sdn_discovery_probe')
    ok = {'never_recovers': False, 'stuck_fid': None, 'partial_walk_fids': []}
    assert mod.verdict_word(ok) == 'ok'
    assert mod.verdict_word(dict(ok, partial_walk_fids=[0])) == 'part-walk'
    # precedence: a starved-fragment row and a never-recovering row keep
    # their own words even when a partial walk is also present
    assert mod.verdict_word(dict(ok, stuck_fid=1,
                                 partial_walk_fids=[0])) == 'partial'
    assert mod.verdict_word(dict(ok, never_recovers=True, stuck_fid=1,
                                 partial_walk_fids=[0])) == 'NEVER'
    # a verdict from an older JSON without the key is still readable
    assert mod.verdict_word({'never_recovers': False,
                             'stuck_fid': None}) == 'ok'
    # the column is 9 characters wide in the row format
    assert len('part-walk') <= 9


def test_ue_probe_histogram_records_which_predicate_produced_it():
    """DEFECT (round 3): `interior_ue_histogram` chose silently
    between two different predicates.  With `counts` absent (every pickle
    written before the previous round, which is every pickle this study has
    committed evidence from) or present with a length that does not match
    `examples`, it fell back to `name_is_ue` with no marker, no counter and
    no field in the JSON -- so a report could not tell a reader whether the
    recorded-count check it describes had applied, and a mis-sized count
    list was indistinguishable from a correct one."""
    mod = _load('ue_transit_probe')
    rescue = ['UE_3', 'O-RU_1', 'Rescue_UE_7', 'O-RU_2', 'UE_4']

    # (1) the three sources, at the level of the histogram itself
    ex = [(0, list(rescue)), (1, list(rescue))]
    assert mod.interior_ues(ex, counts=[1, 1]) == ({1: 2}, 'recorded')
    assert mod.interior_ues(ex) == ({1: 2}, 'name-fallback')
    assert mod.interior_ues(ex, counts=[1]) == ({1: 2},
                                                'count-length-mismatch')
    # a stored count that differs from the name rule is used as it stands,
    # so 'recorded' is not a relabelling of the same computation
    assert mod.interior_ues(ex, counts=[2, 2]) == ({2: 2}, 'recorded')
    # the thin wrapper still answers with the histogram alone
    assert mod.interior_ue_histogram(ex, counts=[1, 1]) == {1: 2}
    assert set(mod.INTERIOR_UE_SOURCES) == {'recorded', 'name-fallback',
                                            'count-length-mismatch'}

    # (2) summarise_pass records the source per pass, and the anomaly
    run = _synthetic_run(24)
    run['ue_transit_probe_examples'] = [(0, list(rescue))]
    run['ue_transit_probe_examples_post'] = [(6, 'not-in-view', list(rescue)),
                                             (7, 'in-view', list(rescue))]
    run['ue_transit_probe_meta'] = dict(run['ue_transit_probe_meta'],
                                        example_interior_ues=[1],
                                        example_post_interior_ues=[1, 1])
    s = mod.summarise_pass(run, 'marl_static', 'rescue_ops', 87, 0)
    assert s['interior_ues_source'] == {'first_ticks': 'recorded',
                                        'post_event': 'recorded'}
    assert s['interior_ue_count_anomalies'] == 0

    # a pass written before the counts existed: the number is unchanged,
    # but the report now says which predicate produced it
    legacy = dict(run)
    legacy['ue_transit_probe_meta'] = {'transit_ues': ['Rescue_UE_7'],
                                       'summary_errors': 0}
    s2 = mod.summarise_pass(legacy, 'marl_static', 'rescue_ops', 87, 0)
    assert s2['interior_ues_per_example'] == {'first_ticks': {1: 1},
                                              'post_event': {1: 2}}
    assert s2['interior_ues_source'] == {'first_ticks': 'name-fallback',
                                         'post_event': 'name-fallback'}
    assert s2['interior_ue_count_anomalies'] == 0

    # a mis-sized count list is a RECORDED anomaly, not a silent
    # fall-through: the answer is the fallback's, and it is labelled
    bad = dict(run)
    bad['ue_transit_probe_meta'] = dict(run['ue_transit_probe_meta'],
                                        example_post_interior_ues=[1])
    s3 = mod.summarise_pass(bad, 'marl_static', 'rescue_ops', 87, 0)
    assert s3['interior_ues_per_example']['post_event'] == {1: 2}
    assert s3['interior_ues_source'] == {'first_ticks': 'recorded',
                                         'post_event': 'count-length-mismatch'}
    assert s3['interior_ue_count_anomalies'] == 1


def test_leader_rule_census_measures_the_parallel_link_claim():
    """DEFECT (round 3): D12.3 item 4 and the `leader_of` docstring
    asserted as fact that the model HAS parallel links, and that the
    committed evidence was safe only because no parallel link fell on a
    tied-degree candidate.  Neither was ever measured.
    `tools/leader_rule_census.py` measures both, and this test shows it
    would report a divergence if the data had one -- otherwise a census
    that always prints 0 would be the same kind of check as the ones
    D12.3 was raised about."""
    import importlib.util
    path = os.path.join(ROOT, 'tools', 'leader_rule_census.py')
    spec = importlib.util.spec_from_file_location('leader_rule_census', path)
    cen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cen)

    # (1) the fixture WITH a parallel pair: counted, and the two rules
    # diverge on it
    t = _Topo()
    for n in PARALLEL_INFRA:
        t.nodes[n] = _Node('O-RU')
    for i, (a, b) in enumerate(PARALLEL_LINKS):
        t.links['l%02d_%s_%s' % (i, a, b)] = _Link(a, b)
    r = cen.census_case(t, [], [])
    assert r['raw_parallel_pairs'] == [('B', 'C')]
    assert r['raw_max_multiplicity'] == 4
    assert r['post_parallel_pairs'] == [('B', 'C')]
    comp = [c for c in r['components'] if c['elected']]
    assert len(comp) == 1 and comp[0]['size'] == 6
    assert comp[0]['leader_uplink_degree'] == 'B'
    assert comp[0]['leader_distinct_peers'] == 'A'
    assert comp[0]['ecc_uplink_degree'] == 3
    assert comp[0]['ecc_distinct_peers'] == 2
    assert comp[0]['diverges'] is True

    # (2) the same graph with the parallel pair collapsed to one link:
    # no parallel pair, and the two rules agree
    t2 = _Topo()
    for n in PARALLEL_INFRA:
        t2.nodes[n] = _Node('O-RU')
    for i, (a, b) in enumerate(sorted(set(PARALLEL_LINKS))):
        t2.links['l%02d_%s_%s' % (i, a, b)] = _Link(a, b)
    r2 = cen.census_case(t2, [], [])
    assert r2['raw_parallel_pairs'] == [] and r2['post_parallel_pairs'] == []
    comp2 = [c for c in r2['components'] if c['elected']]
    assert len(comp2) == 1
    assert comp2[0]['leader_uplink_degree'] == 'A'
    assert comp2[0]['leader_distinct_peers'] == 'A'
    assert comp2[0]['diverges'] is False

    # (3) a cut link leaves the post-severance count able to differ from
    # the raw one: the census reports the graph the election runs on
    lid = sorted(k for k in t.links if k.endswith('_B_C'))[0]
    r3 = cen.census_case(t, [lid], [])
    assert r3['raw_parallel_pairs'] == [('B', 'C')]
    assert r3['post_max_multiplicity'] == 3
