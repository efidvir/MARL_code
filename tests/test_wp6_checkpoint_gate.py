# -*- coding: utf-8 -*-
"""WP6-FIX2 / WP6-FIX3 -- the schema gate on every checkpoint population.

WP6 added `assert_checkpoint_schema`.  The first adversarial review found
four populations of checkpoints that never ran it and one collection path
that still substituted the ACTOR vector for the critic input; the SECOND
review found a fifth population that this file had CLAIMED to cover --
`sixg_sim/main.py`'s auto-resume (WP6-FIX3 defect 2).  The claim is no
longer made by assertion: `test_every_policy_checkpoint_load_is_gated`
enumerates every non-test `torch.load` of a policy checkpoint in the tree,
so a sixth population cannot appear without failing this file.

The populations:

  * the evaluation driver (`run_timeline_comparison.py`), which produces the
    published marl_static / marl_freeze / untrained / random numbers, loaded
    every checkpoint through `load_policy_state_dict()` alone -- an fc1
    WIDTH comparison, which a same-width / different-order checkpoint passes;
  * `RLAgent.save_policy` wrote a bare state dict with no id, while
    `RLAgent.load_policy` asserts one: the pair was self-inconsistent;
  * `ModelRegistry` (the trainer's own on-disk store) neither stamped nor
    checked the id;
  * `MAPPOTrainer.collect` stored the 26-wide ACTOR tensor as the critic
    input whenever a caller omitted `global_obs`;
  * `sixg_sim/iops/migrate_checkpoint.py`, a shipped fc1 zero-padding
    migration tool, made the WP6 "no padding path remains" claim false;
  * `sixg_sim/main.py`'s auto-resume loaded the newest file under
    `--checkpoint-dir` straight into `mappo_trainer._ref_actor` through
    `load_policy_state_dict(..., strict=False)` -- an fc1 WIDTH comparison
    that tolerated missing and unexpected keys as well -- inside a blanket
    `except Exception` that would have swallowed the refusal.

One `torch.load` of a policy-shaped file is deliberately NOT gated, and the
scan names it: `sixg_sim/coordinator_agent.py` persists
`CoordinatorPolicyNet`, a 5-dim global-policy network that never touches
`ACTOR_OBS_FIELDS`, so `assert_checkpoint_schema` is the wrong gate for it.
Its own defect -- a blanket except that leaves the coordinator on random
weights -- belongs to whoever owns that file.

Fast: no simulator pass.  Run with

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp6_checkpoint_gate
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tmp_base():
    """Parent of every scratch directory of this file: output/tmp_tests
    inside the tree (the seed audit skips it by design), never the system
    default, which falls back to the current directory -- the tree root,
    where the suite runs -- when `/` is full (D12F.2)."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return base


if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _agent(node_id='site_0'):
    from sixg_sim.agent import RLAgent
    from sixg_sim.traffic import SliceDictionary
    return RLAgent(node_id, SliceDictionary(), is_training=False)


def _driver():
    """run_timeline_comparison, imported with a neutral argv."""
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    return R


def _src(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as f:
        return f.read()


# ── 1. RLAgent.save_policy / load_policy are one format ──────────────────
def test_save_policy_writes_the_stamped_format():
    """DEFECT 2: save_policy wrote a BARE state dict with no schema id, so
    every checkpoint it produced was refused by load_policy."""
    import torch
    from sixg_sim.agent import SCHEMA_ID, checkpoint_schema_id

    a = _agent()
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        p = os.path.join(d, 'policy.pt')
        a.save_policy(p)
        ck = torch.load(p, weights_only=False)

    assert isinstance(ck, dict), type(ck)
    assert set(ck) == {'state_dict', 'version', 'metadata'}, sorted(ck)
    assert 'provenance' in ck['metadata'], sorted(ck['metadata'])
    assert checkpoint_schema_id(ck) == SCHEMA_ID
    # the same shape train_on_comparison.py, make_untrained_checkpoints.py
    # and mint_schema_current_checkpoint already write
    assert ck['metadata']['provenance']['schema_id'] == SCHEMA_ID
    assert 'fc1.weight' in ck['state_dict']

    # ... and the driver must read it as the LEGACY single-policy format.
    # WP6-FIX3 review: without the non-dict `version` key this file passed
    # `all(isinstance(v, dict) for v in ck.values())` by accident, so a
    # checkpoint written by `sixg_sim/main.py --save-policy-path` entered
    # the evaluation driver through the PER-AGENT branch and was reported as
    # "per-agent, Direct match: 0, fallback: N".  It still loaded the right
    # weights, but only because dict insertion order put 'state_dict' first
    # and `fallback_sd = list(checkpoint.values())[0]` picked it up.  This is
    # the same assertion tests/test_wp6_parity.py makes for the minted file.
    R = _driver()
    assert not R._is_per_agent_checkpoint(R._strip_reserved_keys(ck)), (
        'a save_policy checkpoint is mis-read as a per-agent checkpoint')


def test_save_policy_load_policy_round_trip():
    import torch
    a = _agent('site_0')
    b = _agent('site_1')
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        p = os.path.join(d, 'policy.pt')
        a.save_policy(p)
        b.load_policy(p)          # used to raise: no id in a bare state dict
    for k, v in a.policy_net.state_dict().items():
        assert torch.equal(v, b.policy_net.state_dict()[k]), k
    assert b.policy_version >= 1000


def test_load_policy_still_refuses_a_foreign_schema():
    """The round trip must not have been bought by weakening the gate."""
    import torch
    a = _agent()
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        p = os.path.join(d, 'stale.pt')
        torch.save({'state_dict': a.policy_net.state_dict(),
                    'metadata': {'provenance': {'schema_id': 'obs1-0000'}}}, p)
        try:
            a.load_policy(p)
        except ValueError as e:
            assert 'schema' in str(e), e
        else:
            raise AssertionError('load_policy accepted a foreign schema id')

        p2 = os.path.join(d, 'bare.pt')
        torch.save(a.policy_net.state_dict(), p2)   # pre-WP6 spelling
        try:
            a.load_policy(p2)
        except ValueError as e:
            assert 'schema' in str(e), e
        else:
            raise AssertionError('load_policy accepted an unstamped checkpoint')


# ── 2. the trainer's own checkpoint store ────────────────────────────────
def test_model_registry_stamps_when_metadata_is_omitted():
    """DEFECT 3: `metadata or {}` let any caller write an unstamped
    checkpoint into the trainer's own store."""
    import torch
    from sixg_sim.agent import PolicyNetwork, SCHEMA_ID, checkpoint_schema_id
    from sixg_sim.mappo_trainer import ModelRegistry

    net = PolicyNetwork()
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        reg = ModelRegistry(d, 'agent/0 1')
        path = reg.save(net)                       # no metadata at all
        ck = torch.load(path, weights_only=False)
        assert checkpoint_schema_id(ck) == SCHEMA_ID, ck.get('metadata')

        # metadata WITHOUT a provenance block is still stamped
        path2 = reg.save(net, metadata={'episode': 7})
        ck2 = torch.load(path2, weights_only=False)
        assert checkpoint_schema_id(ck2) == SCHEMA_ID
        assert ck2['metadata']['episode'] == 7, 'caller metadata was dropped'

        # a caller's own provenance is left alone
        path3 = reg.save(net, metadata={'provenance': {'schema_id': SCHEMA_ID,
                                                       'driver': 'mine'}})
        ck3 = torch.load(path3, weights_only=False)
        assert ck3['metadata']['provenance']['driver'] == 'mine'

        # ... and a caller's provenance that LACKS the id is MERGED, not
        # replaced: the id is filled in and every key the caller set
        # survives (review of WP6-FIX2; the first spelling of this
        # replaced the whole block and silently dropped that record).
        path4 = reg.save(net, metadata={'provenance': {'driver': 'legacy',
                                                       'tree_head': 'abc123',
                                                       'arm': 'marl_static'}})
        ck4 = torch.load(path4, weights_only=False)
        prov4 = ck4['metadata']['provenance']
        assert checkpoint_schema_id(ck4) == SCHEMA_ID, prov4
        for k, v in (('driver', 'legacy'), ('tree_head', 'abc123'),
                     ('arm', 'marl_static')):
            assert prov4.get(k) == v, ('the caller provenance key %r was '
                                       'dropped: %r' % (k, prov4))
        # and the fill-in really is a full record, not just the id
        assert 'created_utc' in prov4 and 'python' in prov4, prov4

        # WP6-FIX3 defect 7: a caller-supplied id that is NOT this tree's
        # is REFUSED.  The weights being written are always this process's
        # policy_net, so a stale or foreign provenance block (one loaded
        # from an older checkpoint and re-used) would mislabel them, and
        # the loaders below would then refuse the file forever with
        # nothing at save time saying why.
        raised = None
        try:
            reg.save(net, metadata={'provenance': {
                'schema_id': 'obs1-deadbeefdeadbeef', 'driver': 'stale'}})
        except ValueError as e:
            raised = e
        assert raised is not None, (
            'ModelRegistry.save wrote this tree-s weights under a foreign '
            'schema id')
        assert 'obs1-deadbeefdeadbeef' in str(raised), str(raised)
        assert SCHEMA_ID in str(raised), str(raised)

        # the same id spelled correctly is still accepted (the refusal is
        # about a MISMATCH, not about supplying an id at all)
        path5 = reg.save(net, metadata={'provenance': {
            'schema_id': SCHEMA_ID, 'driver': 'explicit'}})
        ck5 = torch.load(path5, weights_only=False)
        assert ck5['metadata']['provenance']['driver'] == 'explicit'


def test_model_registry_loaders_run_the_schema_gate():
    import torch
    from sixg_sim.agent import PolicyNetwork
    from sixg_sim.mappo_trainer import ModelRegistry

    net = PolicyNetwork()
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        reg = ModelRegistry(d, 'a0')
        reg.save(net)
        reg.save(net)

        fresh = PolicyNetwork()
        v = ModelRegistry(d, 'a0').load_latest(fresh)
        assert v == 1, v
        for k, t in net.state_dict().items():
            assert torch.equal(t, fresh.state_dict()[k]), k
        assert ModelRegistry(d, 'a0').load_version(PolicyNetwork(), 0) is True

        # now poison v1's stamp: BOTH loaders must refuse it
        p = os.path.join(d, 'policy_a0_v1.pt')
        ck = torch.load(p, weights_only=False)
        ck['metadata']['provenance']['schema_id'] = 'obs1-notthistree'
        torch.save(ck, p)
        for call in (lambda r: r.load_latest(PolicyNetwork()),
                     lambda r: r.load_version(PolicyNetwork(), 1)):
            try:
                call(ModelRegistry(d, 'a0'))
            except ValueError as e:
                assert 'schema' in str(e), e
            else:
                raise AssertionError('a loader accepted a foreign schema id')


# ── 3. the collection path cannot store an ACTOR vector as critic input ──
def _trainer(agents):
    from sixg_sim.agent import CriticNetwork, CRITIC_OBS_DIM
    from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
    return MAPPOTrainer(agents, CriticNetwork(CRITIC_OBS_DIM), MAPPOConfig())


def test_collect_requires_a_critic_width_global_obs():
    """DEFECT 6: the silent actor-for-critic substitution in `collect`."""
    import torch
    from sixg_sim.agent import ACTOR_OBS_DIM, CRITIC_OBS_DIM

    agents = {'a': _agent('a')}
    tr = _trainer(agents)
    obs = torch.zeros(ACTOR_OBS_DIM)
    heads = {'relay': 0}
    prb = [0.34, 0.33, 0.33]

    # omitted entirely -> loud, not a silent fallback to the actor tensor
    try:
        tr.collect('a', obs, heads, prb, 0.0, 0.0, 0.0, False)
    except ValueError as e:
        assert 'global_obs' in str(e), e
    else:
        raise AssertionError('collect accepted a missing global_obs')

    # the actor tensor passed by mistake -> loud
    try:
        tr.collect('a', obs, heads, prb, 0.0, 0.0, 0.0, False, global_obs=obs)
    except ValueError as e:
        assert str(CRITIC_OBS_DIM) in str(e), e
    else:
        raise AssertionError('collect accepted an ACTOR-width critic input')

    # the real critic feature vector -> accepted and stored unchanged
    g = _agent('a').critic_features(_observation())
    assert tuple(g.shape) == (CRITIC_OBS_DIM,)
    tr.collect('a', obs, heads, prb, 0.0, 0.0, 0.0, False, global_obs=g)
    stored = tr._traj['a']['global_obs'][-1]
    assert tuple(stored.shape) == (CRITIC_OBS_DIM,), stored.shape


def _observation(**over):
    from sixg_sim.agent import (AgentObservation, NeighborRadioSummary,
                                ConnectivityState)
    from sixg_sim.phy_mac_state import PHYMACState
    kw = dict(node_id='a', is_island=False, energy_soc=0.8,
              phy_mac=PHYMACState(node_id='a'),
              neighbor_radio=NeighborRadioSummary(),
              connectivity=ConnectivityState(),
              ticks_since_severance=0, current_tick=0)
    kw.update(over)
    return AgentObservation(**kw)


def _end_of_branch(lines, i):
    """Index of the first line after the branch that line `i` sits in:
    the next line at or below its indentation that is not blank."""
    indent = len(lines[i]) - len(lines[i].lstrip())
    for k in range(i + 1, len(lines)):
        l = lines[k]
        if l.strip() and (len(l) - len(l.lstrip())) < indent:
            return k
    return len(lines)


# ── 4. the evaluation driver gates every checkpoint load ─────────────────
def test_every_torch_load_in_the_eval_driver_is_gated():
    """DEFECT 1: the strict loader was never invoked on the evaluation path.

    Source-level, because the two loads sit inside the pass builder: every
    `torch.load(` in the driver must be followed, within a few lines, by the
    WP6 gate -- and the gate must come BEFORE `_strip_reserved_keys`, which
    throws the reserved provenance block away.
    """
    lines = _src('run_timeline_comparison.py').splitlines()
    loads = [i for i, l in enumerate(lines) if 'torch.load(' in l]
    assert len(loads) == 2, ('expected the ensemble-member load and the '
                             'single-checkpoint load, found %d' % len(loads))
    strips = 0
    for i in loads:
        # the gate must be close to the load ...
        near = lines[i + 1:i + 6]
        assert any('assert_checkpoint_schema(' in l for l in near), (
            'ungated torch.load at run_timeline_comparison.py:%d' % (i + 1))
        gate = next(k for k, l in enumerate(near)
                    if 'assert_checkpoint_schema(' in l)
        # ... and it must pass the driver-specific reserved key.  WP6-FIX3
        # review: the gate is called INLINE now (the wrapper that used to
        # bind this argument was ten lines in a file this track does not
        # own), so nothing but this assertion keeps an adapted per-agent
        # checkpoint -- whose stamp lives under ADAPTED_PROVENANCE_KEY, not
        # under 'metadata' -- from being refused for having no stamp.
        assert any('reserved_key=ADAPTED_PROVENANCE_KEY' in l
                   for l in lines[i + 1:i + 8]), (
            'the gate at run_timeline_comparison.py:%d no longer reads the '
            'adapted provenance key' % (i + 1))
        # ... and BEFORE the strip, wherever in the branch the strip sits.
        # Searched to the end of the enclosing branch rather than over a
        # fixed window: a window that ends on the strip turns this check
        # into a no-op the moment a line is inserted (review of WP6-FIX2).
        branch = lines[i + 1:_end_of_branch(lines, i)]
        here = [k for k, l in enumerate(branch)
                if '_strip_reserved_keys(' in l]
        strips += len(here)
        for k in here:
            assert gate < k, (
                'the schema gate runs AFTER _strip_reserved_keys, which '
                'removes the provenance block it reads '
                '(run_timeline_comparison.py:%d)' % (i + 2 + k))
    # positively: the single-checkpoint branch DOES strip, so the ordering
    # assertion above is about a call that exists
    assert strips == 1, ('expected exactly one _strip_reserved_keys() call '
                         'after a torch.load, found %d' % strips)

    # and the dead pre-WP6 adaptation reporting is gone with it
    src = _src('run_timeline_comparison.py')
    assert '_obs_dim_adapted' not in src
    assert 'STALE OBS SCHEMA' not in src
    assert 'zero-pads the fc1' not in src


def test_eval_gate_reads_the_adapted_provenance_block():
    """An adapted per-agent checkpoint carries its provenance under the
    reserved '__provenance__' key, not under 'metadata': the stamp must
    survive the adaptation round trip.

    Called exactly as the driver now calls it -- `assert_checkpoint_schema`
    with `reserved_key=ADAPTED_PROVENANCE_KEY`.  The driver-local wrapper
    that used to bind that argument was deleted in review round 3: it was
    ten lines in an engine-owned file to bind one constant."""
    from sixg_sim.agent import SCHEMA_ID, PolicyNetwork, assert_checkpoint_schema
    R = _driver()

    def _gate(ck):
        return assert_checkpoint_schema(ck, where='t',
                                        reserved_key=R.ADAPTED_PROVENANCE_KEY)

    adapted = {'site_0': PolicyNetwork().state_dict()}
    adapted[R.ADAPTED_PROVENANCE_KEY] = {'schema_id': SCHEMA_ID,
                                         'driver': 'test'}
    assert _gate(adapted) == SCHEMA_ID

    adapted[R.ADAPTED_PROVENANCE_KEY] = {'schema_id': 'obs1-someothertree'}
    try:
        _gate(adapted)
    except ValueError as e:
        assert 'schema' in str(e), e
    else:
        raise AssertionError('the adapted-checkpoint gate accepted a '
                             'foreign schema id')

    # an ordinary stamped checkpoint still goes through the plain path
    good = {'state_dict': PolicyNetwork().state_dict(),
            'metadata': {'provenance': {'schema_id': SCHEMA_ID}}}
    assert _gate(good) == SCHEMA_ID
    try:
        _gate({'state_dict': {}})
    except ValueError as e:
        assert 'schema' in str(e), e
    else:
        raise AssertionError('an unstamped checkpoint was accepted')


# ── 5. the convergence script encodes the WP6 critic contract ────────────
def test_marl_convergence_script_uses_the_critic_width():
    """DEFECT 4: the shipped MAPPO convergence/schema-guard script built the
    centralised critic at the ACTOR width and fed it ACTOR vectors -- the
    pre-WP6 contract D6.6 was raised to remove -- and passed while doing so.
    """
    code = '\n'.join(l for l in _src('test_marl_convergence.py').splitlines()
                     if not l.lstrip().startswith('#'))
    assert 'CriticNetwork(CRITIC_OBS_DIM)' in code
    assert 'CriticNetwork(OBS_DIM)' not in code
    assert 'critic_features(' in code, \
        'the value/global_obs inputs must come from critic_features()'
    assert 'torch.stack(all_obs).mean' not in code


# ── 6. the fc1 zero-padding migration tool is gone ───────────────────────
def test_the_fc1_padding_migration_tool_is_deleted():
    """DEFECT 7: a tracked, shipped fc1 zero-padding migration tool made
    test_no_masking_or_padding_path_remains assert something false."""
    from sixg_sim import runinfo
    assert not os.path.exists(
        os.path.join(ROOT, 'sixg_sim', 'iops', 'migrate_checkpoint.py'))  # wp7-scan-ok: 'iops' here is the PACKAGE directory, not the head
    listed, _source = runinfo._tree_files(ROOT)
    if listed:
        bad = [f for f in listed if 'migrate_checkpoint' in f]
        assert not bad, bad
    import sixg_sim.iops as _iops
    assert not hasattr(_iops, 'migrate_checkpoint')


# ── 7. BOTH --resume paths (WP6-FIX3 defect 2, and its review) ───────────
# Every driver with a --resume that installs weights into the trainer, and
# the name its blanket handler binds the exception to.  The first review of
# the DEFECT-2 fix found it applied to main.py alone, while the call site it
# had been modelled ON -- train_on_comparison.py, the driver that actually
# performs the WP6 retrain -- still loaded with strict=False under a bare
# `except Exception`.  Both are checked here, by the same assertions.
RESUME_DRIVERS = {
    os.path.join('sixg_sim', 'main.py'): '_e',
    'train_on_comparison.py': 'e',
}


def _check_resume_path(rel, exc_name):
    src = _src(rel)
    lines = src.splitlines()
    loads = [i for i, l in enumerate(lines)
             if ('_torch.load(' in l or 'torch.load(' in l)
             and not l.strip().startswith('#')]
    assert len(loads) == 1, ('%s: expected exactly the resume load, found '
                             '%d' % (rel, len(loads)))
    i = loads[0]
    # a wide window: the gate carries a long comment, and the point is the
    # ORDER of gate vs load, not how close either sits to the torch.load
    near = lines[i + 1:i + 35]
    assert any('assert_checkpoint_schema(' in l for l in near), (
        'ungated torch.load at %s:%d' % (rel, i + 1))
    gate = next(k for k, l in enumerate(near)
                if 'assert_checkpoint_schema(' in l)
    load = next((k for k, l in enumerate(near)
                 if 'load_policy_state_dict(' in l and 'import' not in l),
                None)
    assert load is not None, '%s: the resume no longer loads anything' % rel
    assert gate < load, ('the gate runs AFTER the weights are installed '
                         '(%s)' % rel)

    # the resume block, from the load into its exception handling.  CODE
    # lines only: the comment there names strict=False to say it is gone.
    block_lines = [l for l in lines[i:i + 60]
                   if not l.strip().startswith('#')]
    block = '\n'.join(block_lines)
    assert 'strict=False' not in block, (
        '%s: the resume still loads with strict=False: a checkpoint missing '
        'keys leaves the resumed actor on its random initialisation for '
        'every key it does not carry' % rel)

    # and the enclosing except must not swallow the refusal.  RuntimeError
    # as well as ValueError (review of the DEFECT-2 fix): dropping
    # strict=False means torch's own load_state_dict raises RuntimeError --
    # not ValueError -- on a missing or unexpected key, so a re-raise that
    # covered ValueError alone let exactly the case the fix was written for
    # fall through to the blanket handler.  It only changed the silent
    # outcome from a partially-loaded actor to a fully-random one.
    blanket = 'except Exception as %s' % exc_name
    assert blanket in block, ('%s: the resume block no longer has the '
                              'blanket handler this test is about' % rel)
    reraise = 'except (ValueError, RuntimeError)'
    assert reraise in block, (
        '%s: the blanket except swallows the schema refusal (ValueError) or '
        'the strict loader (RuntimeError) into a [Resume] WARNING and a '
        'silent fresh start' % rel)
    assert block.index(reraise) < block.index(blanket), (
        '%s: the re-raise must come BEFORE the blanket handler' % rel)


def test_both_resume_paths_run_the_gate_strictly():
    """WP6-FIX3 DEFECT 2 and its review: main.py:379 and
    train_on_comparison.py:688 each loaded the newest / configured
    checkpoint straight into the trainer with no gate, with strict=False,
    inside a blanket except."""
    for rel, exc_name in sorted(RESUME_DRIVERS.items()):
        _check_resume_path(rel, exc_name)


# ── 8. no SIXTH population: every policy load in the tree, enumerated ────
# path -> how many gated `torch.load` sites it is expected to contain.
# A file that gains one, or loses one, fails this test: the claim in this
# module docstring is checked against the tree instead of asserted.
GATED_POLICY_LOADS = {
    'run_timeline_comparison.py': 2,      # ensemble member, single checkpoint
    'train_on_comparison.py': 1,          # --resume
    # D6.40: the untrained control's key/shape reference is now a MINTED
    # checkpoint read back off disk, so this driver loads one too -- and it
    # gates it, like every other population.
    'make_untrained_checkpoints.py': 1,
    os.path.join('sixg_sim', 'agent.py'): 1,          # RLAgent.load_policy
    os.path.join('sixg_sim', 'main.py'): 1,           # auto-resume
    os.path.join('sixg_sim', 'mappo_trainer.py'): 2,  # ModelRegistry loaders
}
# Loads of a policy-shaped file that are NOT ACTOR_OBS_FIELDS checkpoints,
# with the reason.  `assert_checkpoint_schema` is the wrong gate for these.
UNGATED_BY_DESIGN = {
    os.path.join('sixg_sim', 'coordinator_agent.py'):
        'CoordinatorPolicyNet is a 5-dim global-policy network that never '
        'reads ACTOR_OBS_FIELDS',
    # trackAgent merge: two master/trackB tools read checkpoint FILES as
    # data and never load them into a network, so the schema gate is the
    # wrong check for them (both must also read pre-schema files).
    os.path.join('tools', 'policy_digest.py'):
        'WP-DET: digests the weights tensors and the metadata of a '
        'checkpoint file; nothing is loaded into a PolicyNetwork',
    os.path.join('tools', 'seed_audit.py'):
        'Track B WPDOC: scans checkpoint METADATA for seed use (stub '
        'unpickler first); nothing is loaded into a PolicyNetwork',
    # FIX-PREFREEZE item 5 (D-FIX.5): the chain's gate compares the MARL_*
    # environment a checkpoint RECORDED with the plan's
    os.path.join('tools', 'stage_report.py'):
        'FIX-PREFREEZE: --check-marl-env reads the provenance METADATA '
        '(marl_env) of a checkpoint file; nothing is loaded into a '
        'PolicyNetwork',
}
GATE_NAMES = ('assert_checkpoint_schema(',)


def _tree_sources():
    """Every shipped (non-test) python source, as repo-relative paths.

    The whole tree, walked -- not a hand-kept list.  The first version of
    this function enumerated three root files plus `sixg_sim/**` while its
    docstring said "every", which is the same kind of false completeness
    claim it exists to prevent: about seventeen shipped root-level and
    `tools/` sources (confirmatory_analysis.py, extract_timelines.py,
    dump_results_json.py, untrained_analysis.py, tools/*.py, ...) were never
    scanned, so a sixth checkpoint population appearing in any of them would
    not have failed `test_every_policy_checkpoint_load_is_gated`.

    Excluded, and only these: `__pycache__`, dot-directories (`.git`),
    everything under `tests/`, and root-level `test_*.py` contract scripts.
    The test scripts load checkpoints on purpose -- several read the
    pre-WP6 replicate precisely to assert that the gate REFUSES it.

    SHIPPED means TRACKED (trackAgent merge): the list is the tree's own
    file list -- `git ls-files` on the laptop, the sync manifest
    `.synced_files` on the server (sixg_sim.runinfo._tree_files) -- and the
    filesystem walk is only the fallback when neither exists.  The walk alone
    also read run ARTEFACTS: the server tree's `output/` holds a scratch copy
    of an older tree (`output/merge2_tree/`, from the trackB merge), whose
    pre-WP6 modules made this test fail on files nothing ships.
    """
    from sixg_sim import runinfo
    listed, _source = runinfo._tree_files(ROOT)
    if listed:
        rels = [f.replace(chr(92), '/') for f in listed]
    else:
        rels = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = sorted(d for d in dirnames
                                 if d != '__pycache__' and not d.startswith('.')
                                 and d not in ('output', 'logs'))
            rels += [os.path.relpath(os.path.join(dirpath, fn), ROOT)
                     .replace(os.sep, '/') for fn in filenames]
    out = []
    for rel in sorted(set(rels)):
        if (not rel.endswith('.py') or rel.startswith('tests/')
                or os.path.basename(rel).startswith('test_')
                or rel.startswith(('output/', 'logs/'))):
            continue
        if not os.path.isfile(os.path.join(ROOT, *rel.split('/'))):
            continue
        out.append(os.path.join(*rel.split('/')))
    assert 'run_timeline_comparison.py' in out and len(out) > 20, (
        'the source walk found %d files -- it is not walking the tree'
        % len(out))
    return sorted(out)


def test_every_policy_checkpoint_load_is_gated():
    """WP6-FIX3 DEFECT 2, the general form.  The first fix round asserted in
    prose that it had covered 'EVERY checkpoint population in the tree' and
    had missed one.  This walks the tree instead."""
    found = {}
    for rel in _tree_sources():
        lines = _src(rel).splitlines()
        sites = [i for i, l in enumerate(lines)
                 if ('torch.load(' in l or '_torch.load(' in l)
                 and not l.strip().startswith('#')]
        if sites:
            found[rel] = sites

    expected = set(GATED_POLICY_LOADS) | set(UNGATED_BY_DESIGN)
    assert set(found) == expected, (
        'the set of shipped files that torch.load a checkpoint changed. '
        'new (ungated and undeclared): %s; gone (update the table): %s'
        % (sorted(set(found) - expected), sorted(expected - set(found))))

    for rel, want in GATED_POLICY_LOADS.items():
        lines = _src(rel).splitlines()
        sites = found[rel]
        assert len(sites) == want, (
            '%s has %d torch.load sites, the table expects %d -- a new '
            'checkpoint population must be gated AND declared here'
            % (rel, len(sites), want))
        for i in sites:
            near = lines[i + 1:i + 16]
            assert any(any(g in l for g in GATE_NAMES) for l in near), (
                'ungated torch.load at %s:%d' % (rel, i + 1))

    # the exception is real, and it is the one documented
    for rel, why in UNGATED_BY_DESIGN.items():
        assert os.path.exists(os.path.join(ROOT, rel)), rel
        assert why


# ── 9. the dead centralised trainer no longer ships the pre-WP6 contract ─
def test_centralized_trainer_builds_the_critic_vector():
    """WP6-FIX3 DEFECT 5.  CentralizedMARLTrainer.update_critic built its
    critic input by CONCATENATING every agent's 26-wide ACTOR vector and
    padding absent agents at the ACTOR width, then fed it to a critic every
    live site builds at CRITIC_OBS_DIM.  It is not instantiated anywhere, so
    it could not fire -- but it shipped exactly the contract D6.6 removed
    and would have raised a shape error the moment anyone used it."""
    import inspect

    from sixg_sim import agent as A

    src = inspect.getsource(A.CentralizedMARLTrainer.update_critic)
    assert 'critic_features(' in src, (
        'update_critic no longer builds the WP6 critic vector')
    assert 'observation_to_tensor(' not in src, (
        'update_critic still feeds the ACTOR vector to the critic')
    assert 'agent.obs_dim' not in src, (
        'update_critic still pads absent agents at the ACTOR width')
    assert 'CRITIC_OBS_DIM' in src, src

    # and it is still dead: no instantiation anywhere in the shipped tree
    sites = [rel for rel in _tree_sources()
             if 'CentralizedMARLTrainer(' in _src(rel)]
    assert not sites, ('CentralizedMARLTrainer is instantiated in %s -- it '
                       'is documented as dead code; wire it up properly or '
                       'update that docstring' % sites)


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
