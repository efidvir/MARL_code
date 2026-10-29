# -*- coding: utf-8 -*-
"""WP6 -- the observation schema: two vectors, fixed dimensions, strict
schema id, and no masking/padding path left anywhere in the tree.

Fast: no simulator is built.  Run with

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp6_obs_schema
"""
import ast
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── the field lists the spec fixes, verbatim and in order ────────────────
EXPECTED_ACTOR = [
    # Block A (11)
    'a_tx_power', 'a_spectral_efficiency', 'a_sinr_mean', 'a_ue_count',
    'a_emergency_fraction', 'a_coverage', 'a_own_link_max_util',
    'a_own_link_min_avail', 'a_relay_mode', 'a_relay_link_active',
    'a_sinr_min',
    # Block B (10)
    'b_neighbour_relay_share', 'b_neighbour_sinr_mean',
    'b_neighbour_sinr_best', 'b_fresh_neighbour_count',
    'b_any_neighbour_relaying', 'b_neighbour_freshness',
    'b_relaying_neighbour_fraction', 'b_neighbour_tx_power_mean',
    'b_own_sinr_drop', 'b_interferer_fraction',
    # Block C local (3)
    'c_own_relay_capacity', 'c_live_infra_neighbours',
    'c_anchored_flow_demand',
    # Block D (2)
    'd_local_isolation', 'd_ticks_since_local_isolation',
]

EXPECTED_GLOBAL = [
    # 11: g_reachable_ue_fraction, g_isolated_ue_count, g_nenb_count and
    # g_multi_island_bridge were removed at WP13 (D13.3, constant in every
    # pass of the production-length dump; tests/test_wp13_integration.py
    # ties the removal to that evidence)
    'g_active_relay_paths',
    'g_ue_to_ue_routed', 'g_intra_island_reachability', 'g_core_distance',
    'g_bridge_node_flag', 'g_island_size', 'g_pending_iops',
    'g_island_member_count', 'g_island_ue_load_balance', 'g_xn_mesh_density',
    'g_local_epc_health',
]


def _observation(**over):
    """A synthetic AgentObservation with every block populated."""
    from sixg_sim.agent import (AgentObservation, NeighborRadioSummary,
                                ConnectivityState)
    from sixg_sim.phy_mac_state import PHYMACState
    kw = dict(
        node_id='site_0', is_island=False, energy_soc=0.8,
        phy_mac=PHYMACState(node_id='site_0'),
        neighbor_radio=NeighborRadioSummary(),
        connectivity=ConnectivityState(),
        ticks_since_severance=0, current_tick=0,
    )
    kw.update(over)
    return AgentObservation(**kw)


# ── 1. the two vectors: exact length, exact names, exact order ───────────
def test_actor_vector_length_and_names():
    import torch
    from sixg_sim.agent import (ACTOR_OBS_FIELDS, ACTOR_OBS_DIM,
                                ACTOR_OBS_NAMES, actor_feature_values,
                                OBS_DIM)
    assert [n for n, _ in ACTOR_OBS_FIELDS] == EXPECTED_ACTOR
    assert list(ACTOR_OBS_NAMES) == EXPECTED_ACTOR
    assert ACTOR_OBS_DIM == len(ACTOR_OBS_FIELDS) == 26
    assert OBS_DIM == ACTOR_OBS_DIM, 'OBS_DIM must be the ACTOR width'

    values = actor_feature_values(_observation())
    assert len(values) == ACTOR_OBS_DIM
    assert all(isinstance(v, float) for v in values)

    from sixg_sim.agent import RLAgent
    from sixg_sim.traffic import SliceDictionary
    agent = RLAgent('site_0', SliceDictionary(), is_training=False)
    t = agent.observation_to_tensor(_observation())
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (ACTOR_OBS_DIM,), t.shape


def test_critic_vector_is_actor_then_global():
    from sixg_sim.agent import (CRITIC_GLOBAL_FIELDS, CRITIC_OBS_DIM,
                                CRITIC_OBS_NAMES, ACTOR_OBS_DIM,
                                actor_feature_values, critic_feature_values)
    assert [n for n, _ in CRITIC_GLOBAL_FIELDS] == EXPECTED_GLOBAL
    assert list(CRITIC_OBS_NAMES) == EXPECTED_ACTOR + EXPECTED_GLOBAL
    assert CRITIC_OBS_DIM == ACTOR_OBS_DIM + len(EXPECTED_GLOBAL) == 37

    obs = _observation()
    actor = actor_feature_values(obs)
    critic = critic_feature_values(obs)
    assert len(critic) == CRITIC_OBS_DIM
    assert critic[:ACTOR_OBS_DIM] == actor, \
        'the critic vector must START with the actor fields, unchanged'

    from sixg_sim.agent import RLAgent, CriticNetwork
    from sixg_sim.traffic import SliceDictionary
    agent = RLAgent('site_0', SliceDictionary(), is_training=False)
    t = agent.critic_features(obs)
    assert tuple(t.shape) == (CRITIC_OBS_DIM,), t.shape
    # the default critic takes exactly this width
    assert CriticNetwork().fc1.in_features == CRITIC_OBS_DIM


def test_no_locality_parameter_survives():
    import inspect
    from sixg_sim.agent import RLAgent
    sig = inspect.signature(RLAgent.observation_to_tensor)
    assert list(sig.parameters) == ['self', 'obs'], list(sig.parameters)


# ── 2. the strict loader ─────────────────────────────────────────────────
def test_strict_loader_rejects_mismatched_schema_id():
    from sixg_sim.agent import (SCHEMA_ID, PolicyNetwork,
                                assert_checkpoint_schema,
                                checkpoint_schema_id,
                                load_policy_state_dict)

    good = {'state_dict': PolicyNetwork().state_dict(), 'version': 0,
            'metadata': {'provenance': {'schema_id': SCHEMA_ID}}}
    assert checkpoint_schema_id(good) == SCHEMA_ID
    assert assert_checkpoint_schema(good, where='test') == SCHEMA_ID

    for bad in (
            {'metadata': {'provenance': {'schema_id': 'obs1-deadbeefdeadbeef'}}},
            {'metadata': {'provenance': {'schema_id': 'placeholder-pre-WP6'}}},
            {'metadata': {'episode': 3}},          # no id at all (pre-WP6)
            {'state_dict': {}},                    # bare
    ):
        try:
            assert_checkpoint_schema(bad, where='test')
        except ValueError as e:
            assert 'schema' in str(e), e
        else:
            raise AssertionError('accepted a mismatched schema id: %r' % bad)

    # and a width mismatch raises instead of being zero-padded/truncated
    import torch
    net = PolicyNetwork()
    sd = dict(net.state_dict())
    sd['fc1.weight'] = torch.zeros(net.fc1.out_features,
                                   net.fc1.in_features + 3)
    try:
        load_policy_state_dict(net, sd)
    except ValueError as e:
        assert 'input columns' in str(e), e
    else:
        raise AssertionError('the fc1 pad/truncate path is still alive')


def test_schema_id_is_wired_into_provenance():
    from sixg_sim import agent, runinfo
    assert agent.SCHEMA_ID.startswith('obs1-'), agent.SCHEMA_ID
    assert runinfo.schema_id() == agent.SCHEMA_ID
    assert runinfo.schema_id() != 'placeholder-pre-WP6'
    prov = runinfo.provenance(extra={'driver': 'test_wp6'})
    assert prov['schema_id'] == agent.SCHEMA_ID, prov['schema_id']
    assert 'placeholder' not in prov['schema_id']

    # the id really is a function of the two ordered lists -- and, since
    # RERUN_SPEC WP7 / D7.5, of the ACTION SPACE as well: the checkpoint
    # contract is both halves, and a same-width change to the heads (a
    # re-ordering of the relay modes, say) would otherwise be accepted in
    # silence.  The action half is exercised in tests/test_wp7_action_space.
    import hashlib
    payload = '\n'.join([
        'actor=%d' % agent.ACTOR_OBS_DIM,
        'critic=%d' % agent.CRITIC_OBS_DIM,
        'actor_fields=' + ','.join(agent.ACTOR_OBS_NAMES),
        'global_fields=' + ','.join(agent.CRITIC_GLOBAL_NAMES),
        'discrete_heads=' + ','.join(
            '%s:%d' % (h, n) for h, n in zip(agent.DISCRETE_HEADS,
                                             agent.HEAD_SIZES)),
        'relay_modes=' + ','.join(m.name for m in agent.RELAY_HEAD_MODES),
        'continuous_heads=prb:dirichlet:3',
    ])
    assert agent.SCHEMA_ID == 'obs1-' + hashlib.sha256(
        payload.encode('utf-8')).hexdigest()[:16]


# ── 3. Block D: the local-isolation timer ────────────────────────────────
def test_block_d_timer_rises_and_saturates():
    from sixg_sim.agent import (actor_feature_values, ACTOR_OBS_NAMES,
                                LOCAL_ISOLATION_TIMER_TICKS)
    i_flag = ACTOR_OBS_NAMES.index('d_local_isolation')
    i_timer = ACTOR_OBS_NAMES.index('d_ticks_since_local_isolation')
    assert LOCAL_ISOLATION_TIMER_TICKS == 200.0

    # before local isolation: both exactly 0
    v = actor_feature_values(_observation(local_isolated=False,
                                          ticks_since_local_isolation=0))
    assert v[i_flag] == 0.0 and v[i_timer] == 0.0

    # a large ticks_since value must NOT leak while the flag is down
    v = actor_feature_values(_observation(local_isolated=False,
                                          ticks_since_local_isolation=999))
    assert v[i_flag] == 0.0 and v[i_timer] == 0.0

    # after: the flag is 1 and the timer rises monotonically
    prev = -1.0
    for dt in (0, 1, 10, 50, 100, 199):
        v = actor_feature_values(_observation(local_isolated=True,
                                              ticks_since_local_isolation=dt))
        assert v[i_flag] == 1.0
        assert v[i_timer] == dt / 200.0, (dt, v[i_timer])
        assert v[i_timer] > prev
        prev = v[i_timer]

    # saturates at exactly 1.0 from 200 ticks on
    for dt in (200, 201, 5000):
        v = actor_feature_values(_observation(local_isolated=True,
                                              ticks_since_local_isolation=dt))
        assert v[i_timer] == 1.0, (dt, v[i_timer])


# ── 4. local isolation: the WP3 service, and the true-graph rule ─────────
# (trackAgent merge.)  WP6 read the flag through a stand-in latch over the
# engine's GLOBAL island trigger until WP3 landed.  After the merge the flag
# is the link-state service's per-site `isolation_state`; an arm with no
# service applies the same rule to the true graph.  The stand-in is deleted.
class _StubService(object):
    """A link-state service that answers `isolation_state` from a table."""

    def __init__(self, table):
        self.table = dict(table)
        self.calls = []

    def isolation_state(self, site_id):
        self.calls.append(site_id)
        return self.table.get(site_id, (False, None))


class _NoHookSim(object):
    """A simulator stand-in whose global trigger DISAGREES with the service:
    a reader of island_mode or of action_lock_open() would get it wrong."""

    def __init__(self, sites, svc=None):
        self.agents = {s: object() for s in sites}
        self.island_mode = True
        self.lock_calls = 0
        if svc is not None:
            self.link_state_service = svc

    def action_lock_open(self):
        self.lock_calls += 1
        return True


def test_the_adapter_returns_the_services_isolation_state():
    from sixg_sim.agent import (advance_local_isolation, link_state_service,
                                local_isolation_state)
    svc = _StubService({'s0': (True, 7), 's1': (False, None)})
    sim = _NoHookSim(['s0', 's1', 's2'], svc)
    assert link_state_service(sim) is svc
    assert local_isolation_state(sim, 's0') == (True, 7)
    # the global trigger says "isolated"; the service says s1 is not, and
    # the service wins
    assert local_isolation_state(sim, 's1') == (False, None)
    assert local_isolation_state(sim, 's2') == (False, None)
    assert svc.calls == ['s0', 's1', 's2'], svc.calls
    # nothing is advanced on the engine side while a service owns the state
    assert advance_local_isolation(sim, 5) == 0
    assert sim.lock_calls == 0, 'the adapter consulted the global action lock'
    assert not hasattr(sim, '_local_isolation')


def test_an_object_without_isolation_state_is_not_a_service():
    from sixg_sim.agent import link_state_service
    sim = _NoHookSim(['s0'])
    assert link_state_service(sim) is None
    sim.link_state_service = object()          # no isolation_state
    assert link_state_service(sim) is None


def _true_graph_sim():
    """CORE -- A -- B    C -- D  (C-D cut off from the core), a UE on D.
    A real Topology, no Simulator."""
    from sixg_sim.topology import Topology, Node, NodeType, Link, LinkType
    t = Topology()
    t.add_node(Node(id='CORE', node_type=NodeType.AMF))
    for n in ('A', 'B', 'C', 'D'):
        t.add_node(Node(id=n, node_type=NodeType.O_DU))
    t.add_node(Node(id='UE_0', node_type=NodeType.UE))

    def link(lid, a, b, lt=LinkType.MULTIHAUL_MESH):
        return Link(id=lid, endpoints=(a, b), link_type=lt, capacity=100.0)
    t.add_link(link('L_core', 'CORE', 'A'))
    t.add_link(link('L_ab', 'A', 'B'))
    t.add_link(link('L_cd', 'C', 'D'))
    # an access link to a UE that is itself attached to the core side must
    # NOT join D's component to anything
    t.add_link(link('L_ue_d', 'UE_0', 'D', LinkType.WIRELESS))
    t.add_link(link('L_ue_b', 'UE_0', 'B', LinkType.WIRELESS))

    class _S(object):
        pass
    sim = _S()
    sim.topology = t
    sim.agents = {n: object() for n in ('A', 'B', 'C', 'D')}
    sim.core_nodes = {'CORE'}
    return sim


def test_no_service_applies_the_same_rule_to_the_true_graph():
    from sixg_sim.agent import (advance_local_isolation, local_isolation_state,
                                true_graph_isolation)
    sim = _true_graph_sim()
    assert true_graph_isolation(sim) == {'A': False, 'B': False,
                                         'C': True, 'D': True}, (
        'a UE access link must not join a cut-off component to the core')
    # nothing computed yet: everything reads (False, None)
    assert local_isolation_state(sim, 'C') == (False, None)
    assert advance_local_isolation(sim, 4) == 4       # 4 sites evaluated
    assert local_isolation_state(sim, 'C') == (True, 4)
    assert local_isolation_state(sim, 'A') == (False, None)
    # idempotent within a tick
    sim.topology.links['L_core'].is_up = False
    assert advance_local_isolation(sim, 4) == 0
    assert local_isolation_state(sim, 'A') == (False, None)
    # the next tick sees the cut; C keeps its first stamp
    assert advance_local_isolation(sim, 5) == 4
    assert local_isolation_state(sim, 'A') == (True, 5)
    assert local_isolation_state(sim, 'C') == (True, 4), \
        'isolated_since must not be re-stamped while the state holds'
    # recovery clears it, and a second isolation re-stamps
    sim.topology.links['L_core'].is_up = True
    advance_local_isolation(sim, 6)
    assert local_isolation_state(sim, 'A') == (False, None)
    sim.topology.links['L_core'].is_up = False
    advance_local_isolation(sim, 9)
    assert local_isolation_state(sim, 'A') == (True, 9)


def test_the_block_d_pair_follows_the_adapter():
    """The actor's two Block D columns from a Scenario-B-style state: flag 0
    at tick 0 (no previous step), 1 from tick 1, timer from the step's tick."""
    from sixg_sim.agent import actor_feature_values, ACTOR_OBS_NAMES
    i_flag = ACTOR_OBS_NAMES.index('d_local_isolation')
    i_timer = ACTOR_OBS_NAMES.index('d_ticks_since_local_isolation')
    v0 = actor_feature_values(_observation(local_isolated=False, current_tick=0,
                                           ticks_since_local_isolation=0))
    assert v0[i_flag] == 0.0 and v0[i_timer] == 0.0
    v1 = actor_feature_values(_observation(local_isolated=True, current_tick=1,
                                           ticks_since_local_isolation=1))
    assert v1[i_flag] == 1.0, 'Scenario B must set the flag from tick 1'
    assert v1[i_timer] == 1 / 200.0


def test_the_stand_in_is_gone_and_the_observation_builder_advances_nothing():
    """The deleted stand-in may not come back: no `_wp2_isolation_*` state,
    no read of action_lock_open() in the adapter, and the observation
    builder only READS (the update is tick_step phase 6)."""
    src = open(os.path.join(ROOT, 'sixg_sim', 'agent.py'),
               encoding='utf-8').read()
    n_checked = 0
    for fn in ('link_state_service', 'local_isolation_state',
               'true_graph_isolation', 'advance_local_isolation'):
        body = src.split('def %s(' % fn)[1].split('\ndef ')[0]
        code = '\n'.join(l for l in body.splitlines()
                         if l.strip() and not l.strip().startswith('#'))
        code = code.split('"""')[-1] if code.count('"""') >= 2 else code
        assert 'action_lock_open' not in code, fn
        assert 'island_mode' not in code, fn
        n_checked += 1
    assert n_checked == 4
    assert '_wp2_isolation' not in src
    sim_src = open(os.path.join(ROOT, 'sixg_sim', 'simulation.py'),
                   encoding='utf-8').read()
    build = sim_src.split('def _build_agent_observations(')[1].split('\n    def ')[0]
    assert 'advance_local_isolation' not in build.replace(
        '# agent.advance_local_isolation', ''), (
        'the observation builder advances the isolation state again')
    assert 'local_isolation_state' in build
    step = open(os.path.join(ROOT, 'sixg_sim', 'tick_step.py'),
                encoding='utf-8').read()
    # The CALL SITES, read from the syntax tree: a text search also matches
    # the docstring and the comments, and since D13.2 the action step is
    # called with the unlocked sites' observations, not the literal
    # `act_fn(tick, observations)` -- the text search then found the
    # docstring's mention, placed the action step before the observations
    # and failed for a reason that is not a phase-order change.
    import ast
    fn = next(n for n in ast.walk(ast.parse(step))
              if isinstance(n, ast.FunctionDef) and n.name == 'step_tick')

    def _name(f):
        return f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None)

    def _calls(pred):
        return sorted(n.lineno for n in ast.walk(fn)
                      if isinstance(n, ast.Call) and pred(n.func))
    sites = {
        'observations': _calls(
            lambda f: _name(f) == '_build_agent_observations'),
        'act_fn': _calls(lambda f: isinstance(f, ast.Name)
                         and f.id == 'act_fn'),
        'forwarding.step': _calls(
            lambda f: isinstance(f, ast.Attribute) and f.attr == 'step'
            and isinstance(f.value, ast.Name) and f.value.id == 'forwarding'),
        'advance_local_isolation': _calls(
            lambda f: _name(f) == 'advance_local_isolation'),
    }
    for what, lines in sites.items():
        assert len(lines) == 1, (
            'step_tick must call %s exactly once, found %d call(s) at %r'
            % (what, len(lines), lines))
    i_obs, i_act, i_fwd, i_adv = (sites[k][0] for k in (
        'observations', 'act_fn', 'forwarding.step',
        'advance_local_isolation'))
    assert i_obs < i_act < i_fwd < i_adv, (
        'the isolation update must run at phase 6, after the observations '
        'and the action step, next to the service step: %r' % sites)


# ── 5. grep-style scan: no masking or padding path left in the tree ──────
_BANNED_NAMES = ('NONLOCAL_OBS_COLUMNS', 'OBS_LOCALITY', 'mask_nonlocal',
                 'OBS_COL_RELAY_CAPABLE',
                 # the shipped fc1 zero-padding migration tool (WP6-FIX2
                 # D6.12): deleted, and it may not come back under its own
                 # name or its two dimension constants.
                 'migrate_checkpoint', 'OLD_OBS_DIM', 'NEW_OBS_DIM')
_BANNED_TEXT = ("os.environ.get('MARL_OBS_LOCALITY', 'local')",
                'features = (features + [0.0] * OBS_DIM)[:OBS_DIM]',
                # the fc1 column-padding idiom itself, in either spelling
                'torch.cat([tensor, pad], dim=1)',
                'torch.cat([w, pad], dim=1)')


def _tree_sources():
    """Every .py file the TREE SHIPS, and nothing else.

    The file list comes from `.synced_files` on the server or `git ls-files`
    locally (sixg_sim.runinfo._tree_files), so run artefacts, scratch copies
    and a reference checkout of an older commit sitting inside the run
    directory are not scanned -- and, equally, a module cannot escape the
    scan by being dropped somewhere unusual, because anything tracked is in
    the list.  The glob is the fallback when neither listing exists.
    """
    from sixg_sim import runinfo
    listed, _source = runinfo._tree_files(ROOT)
    if listed:
        rels = [f.replace('\\', '/') for f in listed if f.endswith('.py')]
    else:
        rels = [os.path.relpath(p, ROOT).replace(os.sep, '/')
                for p in glob.glob(os.path.join(ROOT, '**', '*.py'),
                                   recursive=True)]
    for rel in rels:
        if rel.startswith(('output/', 'tests/test_wp6_')):
            continue
        path = os.path.join(ROOT, rel)
        if os.path.isfile(path):
            yield rel, path


def _code_lines(src):
    """The source with whole-line comments dropped.

    The removal is documented in prose right where it happened, so the scan
    must look at CODE.  A trailing comment on a code line is kept: it cannot
    revive the behaviour, and dropping it would need a tokeniser pass for no
    gain.
    """
    return '\n'.join(l for l in src.splitlines()
                     if not l.lstrip().startswith('#'))


def test_no_masking_or_padding_path_remains():
    """Like the WP0 pool-cap scan: an AST pass over every module in the tree.

    Names are checked in the CODE (ast.Name / attribute / keyword), not in
    comments, so the explanatory comments that say what was removed do not
    trip it -- but a live reference anywhere does.
    """
    offenders = []
    n_files = 0
    for rel, path in _tree_sources():
        src = open(path, encoding='utf-8').read()
        try:
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        n_files += 1
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.Name):
                names = (node.id,)
            elif isinstance(node, ast.Attribute):
                names = (node.attr,)
            elif isinstance(node, ast.keyword):
                names = (node.arg,)
            elif isinstance(node, ast.arg):
                names = (node.arg,)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                # a re-introduced `def migrate_checkpoint(...)` is a
                # definition, not a Name reference
                names = (node.name,)
            elif isinstance(node, ast.alias):
                # `import a.b.migrate_checkpoint` as well as `from a import x`
                names = tuple((node.name or '').split('.')) + \
                    ((node.asname,) if node.asname else ())
            elif isinstance(node, ast.ImportFrom):
                names = tuple((node.module or '').split('.'))
            for name in names:
                if name in _BANNED_NAMES:
                    offenders.append('%s:%d %s'
                                     % (rel, getattr(node, 'lineno', 0), name))
        code = _code_lines(src)
        for text in _BANNED_TEXT:
            if text in code:
                offenders.append('%s: %r' % (rel, text))
    assert n_files >= 20, n_files
    assert not offenders, (
        'WP6 removed the masking/padding path, but it is still referenced: '
        + '; '.join(sorted(offenders)))


def test_pad_truncate_and_zeroing_loop_are_gone_from_agent():
    src = open(os.path.join(ROOT, 'sixg_sim', 'agent.py'),
               encoding='utf-8').read()
    code = _code_lines(src)
    # the two exact lines the review named (agent.py:805 and agent.py:804)
    assert 'for _c in NONLOCAL_OBS_COLUMNS' not in code
    assert '[0.0] * OBS_DIM)[:OBS_DIM]' not in code
    # and no env switch may set an observation dimension any more
    bad = [l for l in code.splitlines()
           if ('OBS_DIM' in l or 'OBS_LOCALITY' in l) and 'os.environ' in l
           and 'raise' not in l and '_OBS_LOCALITY_ENV' not in l]
    assert not bad, bad


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
