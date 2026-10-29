# -*- coding: utf-8 -*-
"""WP7 -- the action space, as a contract.

RERUN_SPEC WP7.  Four discrete heads (tx power 5, MCS offset emergency 5,
MCS offset general 5, relay 3) plus a Dirichlet split over the 2-simplex.
The handover, scheduler, postcard and isolated-operation admission heads are
DELETED from the network, the trainer, the simulator and every driver, and
the relay head is collapsed from five modes to three.

These tests are FAST: no simulator is built here.  The live-environment
checks (the uniform-random control against the real batched sampler, the
regenerated untrained control and the training smoke) are in
tests/test_wp7_controls.py.

NOTHING IN THIS FILE SKIPS.  Every check either runs or fails; there is no
`if ... : return` that can quietly turn a contract test into a no-op.  The
tripwire at the bottom enforces that for this file and its sibling.

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp7
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Every head this WP deleted, and the attribute/state each one drove.
DELETED_HEADS = ['handover', 'scheduler', 'postcard', 'iops']
EXPECTED_HEADS = ['tx_power', 'mcs_emrg', 'mcs_gen', 'relay']
EXPECTED_SIZES = [5, 5, 5, 3]


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


def _py_files():
    """Every tracked .py file in the tree, absolute paths."""
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in ('__pycache__', '.git', 'output', 'logs')]
        for fn in filenames:
            if fn.endswith('.py'):
                out.append(os.path.join(dirpath, fn))
    assert len(out) > 40, 'the tree walk found only %d files' % len(out)
    return sorted(out)


# ── 1. the network: exactly four discrete heads plus the Dirichlet ───────
def test_policy_network_exposes_exactly_the_four_heads_plus_the_dirichlet():
    import torch
    from sixg_sim.agent import (PolicyNetwork, DISCRETE_HEADS, HEAD_SIZES,
                                ACTOR_OBS_DIM)

    assert DISCRETE_HEADS == EXPECTED_HEADS, DISCRETE_HEADS
    assert HEAD_SIZES == EXPECTED_SIZES, HEAD_SIZES

    net = PolicyNetwork()
    out = net(torch.zeros(3, ACTOR_OBS_DIM))
    assert set(out) == set(EXPECTED_HEADS) | {'prb'}, sorted(out)
    for h, n in zip(EXPECTED_HEADS, EXPECTED_SIZES):
        assert out[h].shape == (3, n), (h, out[h].shape, n)
    assert out['prb'].shape == (3, 3), out['prb'].shape

    # the deleted heads are gone as MODULES too, so no checkpoint of this
    # tree can carry their weights
    keys = list(net.state_dict())
    for dead in DELETED_HEADS:
        assert not any(k.startswith(dead) for k in keys), (dead, keys)
    assert sorted(k for k in keys if k.endswith('_head.weight')) == [
        'mcs_emrg_head.weight', 'mcs_gen_head.weight', 'prb_head.weight',
        'relay_head.weight', 'tx_power_head.weight'], keys

    # the PRB head is a Dirichlet, not a softmax point
    from sixg_sim.agent import prb_distribution, PRB_POLICY
    assert PRB_POLICY == 'dirichlet', PRB_POLICY
    d = prb_distribution(out['prb'])
    s = d.sample()
    assert s.shape == (3, 3)
    assert abs(float(s.sum(dim=-1)[0]) - 1.0) < 1e-5


def test_discrete_heads_has_exactly_one_definition_in_the_tree():
    """AST scan: exactly one module-level assignment to DISCRETE_HEADS, and
    it is in sixg_sim/agent.py.  Everything else imports it."""
    definitions = []
    for path in _py_files():
        with open(path, encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == 'DISCRETE_HEADS':
                    definitions.append((os.path.relpath(path, ROOT),
                                        node.lineno))
    assert len(definitions) == 1, (
        'DISCRETE_HEADS must be ASSIGNED exactly once in the tree; found %r'
        % (definitions,))
    assert definitions[0][0] == os.path.join('sixg_sim', 'agent.py'), (
        'the one definition must be in sixg_sim/agent.py, not %r'
        % (definitions[0],))

    # and the trainer's name is the same object, not a copy
    from sixg_sim.agent import DISCRETE_HEADS as A
    from sixg_sim.mappo_trainer import DISCRETE_HEADS as T
    assert T is A, 'mappo_trainer.DISCRETE_HEADS is a copy again'


# ── the deleted-head scan ────────────────────────────────────────────────
# Names that WERE action heads before WP7 deleted them.  A string constant
# EXACTLY equal to one of these is a live reference; prose never is, because
# prose says "the handover head", not "handover".
DELETED_HEAD_NAMES = {'handover', 'scheduler', 'postcard', 'iops'}

# Identifiers that only ever existed in order to carry a deleted head.
DELETED_IDENTIFIERS = {
    'handover_idx', 'scheduler_idx',
    'handover_head', 'scheduler_head', 'postcard_head', 'iops_head',
    '_iops_decision',
}

# Attributes WP7 deliberately KEEPS as dataclass fields (deleting a field is
# a WP of its own) but which no head may write any more: a read is fine, an
# assignment is a live head.
DELETED_WRITE_ATTRS = {'handover_triggered'}


# A per-LINE opt-out, so that a file which legitimately names a head at one
# site does not have to be allow-listed WHOLE (which is how a real reference
# hides).  Every use is counted and pinned below, so a merge cannot quietly
# add one.
SCAN_OK = '# wp7-scan-ok:'


def _scan_ok_lines(src):
    """Line numbers carrying the `# wp7-scan-ok: <reason>` marker."""
    out = set()
    for i, line in enumerate(src.splitlines(), start=1):
        at = line.find(SCAN_OK)
        if at >= 0 and line[at + len(SCAN_OK):].strip():
            out.add(i)
    return out


def _deleted_head_hits(src, rel):
    """Every AST reference to a deleted action head in `src`.

    AST, not line text.  A line-based scan cannot do this job, and the
    previous one (review round 5, MEDIUM) did not: the idiom WP7 had to
    remove is `{'handover': act.handover_idx}`, a line that is BOTH a string
    constant and a live attribute read, so a scan that skips every line
    carrying a string literal skips exactly the code it exists to catch.

    Comments and docstrings drop out for free here, structurally rather than
    by a filter: a comment is not in the tree at all, and a docstring is a
    string whose VALUE is prose, never the bare head name.
    """
    out = set()
    skip = _scan_ok_lines(src)
    for node in ast.walk(ast.parse(src, filename=rel)):
        ln = getattr(node, 'lineno', None)
        if ln is None or ln in skip:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in DELETED_HEAD_NAMES:
                out.add((ln, 'string %r' % node.value))
        elif isinstance(node, ast.Name):
            if node.id in DELETED_IDENTIFIERS:
                out.add((ln, 'name %s' % node.id))
        elif isinstance(node, ast.Attribute):
            if node.attr in DELETED_IDENTIFIERS:
                out.add((ln, 'attribute .%s' % node.attr))
            elif (node.attr in DELETED_WRITE_ATTRS
                  and isinstance(node.ctx, ast.Store)):
                out.add((ln, 'write to .%s' % node.attr))
        elif isinstance(node, ast.keyword):
            # `PHYMACAction(handover_idx=0, scheduler_idx=...)` -- a keyword
            # is not a Name, and this is the shape the routing-graph test
            # carried until this branch removed it.
            if node.arg in DELETED_IDENTIFIERS:
                out.add((ln, 'keyword %s=' % node.arg))
        elif isinstance(node, ast.arg):
            if node.arg in DELETED_IDENTIFIERS:
                out.add((ln, 'parameter %s' % node.arg))
        elif isinstance(node, ast.alias):
            for nm in (node.name, node.asname):
                if nm in DELETED_IDENTIFIERS:
                    out.add((ln, 'import %s' % nm))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            if node.name in DELETED_IDENTIFIERS:
                out.add((ln, 'def %s' % node.name))
    return sorted(out)


def test_the_deleted_head_scan_actually_fires():
    """Self-test for `_deleted_head_hits` (review round 5).

    The scan this replaces collected the line numbers spanned by EVERY
    string constant and dropped those whole lines, so 11 of its 17 patterns
    could never match and the one idiom WP7 existed to delete --
    `{'handover': act.handover_idx}`, the shape that stood at
    `run_timeline_comparison.py:4941`, `sixg_sim/worker.py:194` and
    `train_on_comparison.py:278` -- produced no hit at all.  A contract test
    that cannot fail is not a contract, so the scan is now tested itself."""
    probe = (
        "def f(act, ps):\n"
        "    act_heads = {'handover': act.handover_idx,\n"
        "                 'scheduler': act.scheduler_idx}\n"
        "    x = act._iops_decision\n"
        "    y = {\"postcard\": 1, \"iops\": 2}\n"
        "    z = PHYMACAction(handover_idx=0)\n"
        "    ps.handover_triggered = True\n"
        "    return act_heads, x, y, z\n"
    )
    hits = _deleted_head_hits(probe, 'probe.py')
    kinds = {k for _ln, k in hits}
    for expect in ("string 'handover'", "string 'scheduler'",
                   "string 'postcard'", "string 'iops'",
                   'attribute .handover_idx', 'attribute .scheduler_idx',
                   'attribute ._iops_decision', 'keyword handover_idx=',
                   'write to .handover_triggered'):
        assert expect in kinds, (expect, sorted(kinds))

    # THE regression the old scan could not see: the dict-literal line is
    # flagged ON ITS OWN LINE, by both the key and the attribute read -- not
    # merely reached through some other line of the probe.
    assert (2, "string 'handover'") in hits, hits
    assert (2, 'attribute .handover_idx') in hits, hits

    # ...and prose still does not fire, which is the property the old scan
    # was reaching for.  A docstring and a comment that discuss the deleted
    # heads are structurally exempt: the docstring's VALUE is not a head
    # name, and the comment is not in the tree.
    prose = ('"""The handover and scheduler heads are gone, and so are the\n'
             'postcard and iops heads: see D7.1."""\n'
             '# handover_idx and scheduler_idx were removed here\n'
             'OK = 1\n')
    assert _deleted_head_hits(prose, 'prose.py') == [], \
        _deleted_head_hits(prose, 'prose.py')


def test_no_reference_to_a_deleted_head_survives():
    """No code anywhere names a deleted head.  Comments and docstrings are
    exempt on purpose -- this WP's whole point is that the reasons are
    written down, and a scan that matched its own prose would be unusable --
    but the exemption is STRUCTURAL (see `_deleted_head_hits`), not a line
    filter that also hides the code."""
    # Files that legitimately still carry the NAME for a non-head reason.
    allowed = {
        # PHYMACState keeps the two unread fields (deleting a dataclass
        # field is a WP of its own and no head writes them now)
        os.path.join('sixg_sim', 'phy_mac_state.py'),
        # the drawing palette has a 'postcard' colour; postcards still exist
        os.path.join('sixg_sim', 'visualizer.py'),
        # this file names every deleted head, on purpose
        os.path.join('tests', 'test_wp7_action_space.py'),
    }
    # Every `# wp7-scan-ok:` exemption in the tree, pinned.  A whole-file
    # allow-list is how a real reference hides, so the opt-out is per LINE
    # and every one of them is named here.  A merge that adds an exemption
    # fails this test and has to justify it.
    expected_markers = {
        # 'iops' there is the sixg_sim/iops PACKAGE directory, not the head
        # (449 and 233 since D12-FIXES item FIRST added the 12-line
        # _tmp_base helper to both files, D12F.2)
        (os.path.join('tests', 'test_wp6_checkpoint_gate.py'), 449),
        # the negative assertion "no key starts with a dead head name" has
        # to spell the dead head names out
        (os.path.join('tests', 'test_wp7_controls.py'), 233),
        # trackAgent merge: since WP5 'postcard' is also the name of the DCC
        # MESSAGE TYPE -- the link-state service bills it, the harness
        # records overhead_by_type['postcard'], Track B's analysis reads it.
        # None of these is the deleted action head; each line is marked.
        (os.path.join('sixg_sim', 'link_state_service.py'), 243),
        (os.path.join('sixg_sim', 'link_state_service.py'), 245),
        (os.path.join('tests', 'test_wp5_postcards.py'), 1113),
        (os.path.join('tests', 'test_wp9_asserts.py'), 93),
        (os.path.join('tests', 'test_wp9_asserts.py'), 623),
        (os.path.join('tests', 'test_wp9_export.py'), 34),
        (os.path.join('tests', 'test_wp9_synth.py'), 105),
        # the merge's own seam test asserts the head is NOT in DISCRETE_HEADS
        (os.path.join('tests', 'test_merge_seams.py'), 405),
    }

    scanned = 0
    hits = []
    markers = set()
    for path in _py_files():
        rel = os.path.relpath(path, ROOT)
        with open(path, encoding='utf-8') as f:
            src = f.read()
        if rel in allowed:
            continue          # already exempt whole; this file DEFINES SCAN_OK
        markers.update((rel, ln) for ln in _scan_ok_lines(src))
        scanned += 1
        for ln, what in _deleted_head_hits(src, rel):
            hits.append('%s:%d  %s' % (rel, ln, what))
    assert not hits, (
        'references to a DELETED action head survive in code:\n  '
        + '\n  '.join(hits))
    assert markers == expected_markers, (
        'the set of `%s` exemptions changed:\n  added %r\n  removed %r'
        % (SCAN_OK, sorted(markers - expected_markers),
           sorted(expected_markers - markers)))
    # a scan that silently stopped finding files would also report no hits
    assert scanned > 40, 'the scan only read %d files' % scanned


# ── 2. the relay head: three modes, and the documented mapping ───────────
def test_the_relay_mapping_is_the_documented_one():
    from sixg_sim.agent import (RELAY_HEAD_MODES, RELAY_HEAD_TO_ENUM,
                                N_RELAY_MODES, relay_enum_index,
                                relay_head_index)
    from sixg_sim.phy_mac_state import RelayMode

    assert N_RELAY_MODES == 3
    assert RELAY_HEAD_MODES == [RelayMode.OFF, RelayMode.LOCAL_REROUTE,
                                RelayMode.CAPACITY_BOOST]
    # the mapping recorded in docs/DECISIONS.md D7.2
    assert RELAY_HEAD_TO_ENUM == [0, 2, 3]
    assert [relay_enum_index(h) for h in range(3)] == [0, 2, 3]
    # the two dropped aliases fold onto OFF, which is what the engine did
    assert relay_head_index(1) == 0, 'NORMAL must fold onto OFF'
    assert relay_head_index(4) == 0, 'D2D_PEER_RELAY must fold onto OFF'
    # round trip
    for h in range(3):
        assert relay_head_index(relay_enum_index(h)) == h

    # the mapping is written down where a reader will look for it
    dec = open(os.path.join(ROOT, 'docs', 'DECISIONS.md'),
               encoding='utf-8').read()
    assert 'D7.2' in dec, 'the relay-mode mapping has no decision record'
    for name in ('LOCAL_REROUTE', 'CAPACITY_BOOST', 'D2D_PEER_RELAY'):
        assert name in dec, name


def test_the_action_carries_both_indices_and_they_agree():
    from sixg_sim.agent import (PHYMACAction, RLAgent, relay_enum_index)
    from sixg_sim.traffic import SliceDictionary

    a = PHYMACAction()
    assert a.relay_head_idx == 0 and a.relay_mode_idx == 0, (
        'the dataclass default must be OFF on both indices')
    assert not hasattr(a, 'handover_idx')
    assert not hasattr(a, 'scheduler_idx')

    agent = RLAgent('site_0', SliceDictionary(), is_training=True)
    seen = set()
    for t in range(40):
        act = agent.compute_action(_observation(current_tick=t))
        assert act.relay_mode_idx == relay_enum_index(act.relay_head_idx), (
            act.relay_head_idx, act.relay_mode_idx)
        assert 0 <= act.relay_head_idx <= 2
        seen.add(act.relay_head_idx)
        assert not hasattr(act, '_iops_decision')
    assert len(seen) >= 2, (
        'a sampling actor produced only %r relay head values in 40 draws'
        % sorted(seen))


# ── 3. the schema id covers the action space ─────────────────────────────
def test_schema_id_covers_the_action_space_not_only_the_observation():
    """RERUN_SPEC WP7 / D7.5.  Two trees that agree on every observation
    field but disagree about the heads must not share a schema id."""
    import sixg_sim.agent as A

    base = A._compute_schema_id()
    assert base == A.SCHEMA_ID, 'SCHEMA_ID is not what _compute_schema_id says'

    def perturbed(**over):
        saved = {k: getattr(A, k) for k in over}
        try:
            for k, v in over.items():
                setattr(A, k, v)
            return A._compute_schema_id()
        finally:
            for k, v in saved.items():
                setattr(A, k, v)

    # a head removed
    assert perturbed(DISCRETE_HEADS=A.DISCRETE_HEADS[:-1],
                     HEAD_SIZES=A.HEAD_SIZES[:-1]) != base
    # a head renamed, same widths
    assert perturbed(DISCRETE_HEADS=['tx_power', 'mcs_emrg', 'mcs_gen',
                                     'relay_mode']) != base
    # a head's cardinality changed
    assert perturbed(HEAD_SIZES=[5, 5, 5, 5]) != base
    # the relay modes RE-ORDERED, which no width check could catch
    assert perturbed(RELAY_HEAD_MODES=list(reversed(A.RELAY_HEAD_MODES))) != base

    # the CONTINUOUS head's parameterisation (D7.12).  MARL_PRB_POLICY
    # =softmax is a supported setting under which the PRB head is a
    # deterministic softmax point instead of a Dirichlet sample: a different
    # action distribution behind an IDENTICAL tensor shape, so no width or
    # key check downstream can separate the two.  The id must.
    assert A.PRB_POLICY == 'dirichlet', (
        'run this test in the default environment (MARL_PRB_POLICY unset)')
    assert perturbed(PRB_POLICY='softmax') != base
    # and it is the LIVE value, not a constant: re-asserting the default
    # reproduces the id exactly.
    assert perturbed(PRB_POLICY='dirichlet') == base

    # and the observation half still matters
    assert perturbed(ACTOR_OBS_NAMES=A.ACTOR_OBS_NAMES[:-1]) != base


def test_the_strict_gate_refuses_a_checkpoint_from_the_old_action_space():
    """A pre-WP7 checkpoint carries the pre-WP7 schema id, so the gate
    refuses it with a message about the schema rather than with a tensor
    shape error out of load_state_dict."""
    from sixg_sim.agent import assert_checkpoint_schema, SCHEMA_ID

    stale = {'state_dict': {}, 'version': 0,
             'metadata': {'provenance': {'schema_id': 'obs1-deadbeefdeadbeef'}}}
    raised = None
    try:
        assert_checkpoint_schema(stale, where='wp7 test')
    except ValueError as e:
        raised = e
    assert raised is not None, 'the gate accepted a foreign schema id'
    assert SCHEMA_ID in str(raised)


# ── 4. the entropy bonus is the sum over the four heads ──────────────────
def build_single_minibatch_trainer(T=32, seed=0):
    """A MAPPOTrainer holding exactly T transitions and configured so that
    `update()` runs ONE minibatch in ONE epoch.

    Why that shape.  `update()` reports `entropy` as the MEAN over minibatch
    steps, and it takes an optimiser step between them, so with more than one
    step the reported number is an average over CHANGING weights and cannot
    be reproduced exactly.  With a single step it is exactly the term the
    loss used, evaluated on the weights as they stood before `update()` was
    called -- which is what makes the assertions below exact rather than
    bounds.  Returns (trainer, obs_list, weights_before).
    """
    import copy
    import torch
    from sixg_sim.agent import (RLAgent, CriticNetwork, DISCRETE_HEADS,
                                ACTOR_OBS_DIM, CRITIC_OBS_DIM, N_RELAY_MODES)
    from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
    from sixg_sim.traffic import SliceDictionary

    torch.manual_seed(seed)
    sd_ = SliceDictionary()
    agents = {'a%d' % i: RLAgent('a%d' % i, sd_, is_training=True)
              for i in range(2)}
    cfg = MAPPOConfig(mini_batch=T, n_epochs=1)
    trainer = MAPPOTrainer(agents, CriticNetwork(CRITIC_OBS_DIM), cfg)

    aids = list(agents)
    obs_list = [torch.randn(ACTOR_OBS_DIM) for _ in range(T)]
    widths = {h: (N_RELAY_MODES if h == 'relay' else 5) for h in DISCRETE_HEADS}
    for i in range(T):
        acts = {h: i % widths[h] for h in DISCRETE_HEADS}
        trainer.collect(aids[i % len(aids)], obs_list[i], acts,
                        [0.5, 0.3, 0.2], -3.0, 0.0, 1.0, i >= T - len(aids),
                        global_obs=torch.randn(CRITIC_OBS_DIM))
    trainer.finish_episode()
    weights_before = copy.deepcopy(trainer._ref_actor.state_dict())
    return trainer, obs_list, weights_before


def test_entropy_is_summed_over_the_four_discrete_heads_on_the_live_update():
    """The coefficient is unchanged; what changed is how many heads the sum
    runs over.  This pins the SUM (not a mean, not a subset) ON THE PATH THAT
    ACTUALLY TRAINS.

    WP7 review defect 3: this contract used to be asserted of
    `RLAgent.evaluate_actions`, which no driver and no trainer ever called,
    and which scored the PRB head with the OLD softmax cross-entropy while
    the live update uses the Dirichlet density.  The production path is
    `MAPPOTrainer.update`; that is what is pinned here, and
    `evaluate_actions` has been deleted (D8.1) so it cannot be pinned again.
    """
    import math
    import torch
    from torch.distributions import Categorical
    from sixg_sim.agent import (PolicyNetwork, DISCRETE_HEADS,
                                apply_relay_mask)
    from sixg_sim.mappo_trainer import _prb_dist, _PRB_POLICY

    T = 32
    trainer, obs_list, weights_before = build_single_minibatch_trainer(T)
    metrics = trainer.update()
    assert metrics['updates'] == 1, (
        'the fixture did not produce exactly one minibatch step: %r' % metrics)

    # Recompute the term on the pre-update weights, head by head.
    net = PolicyNetwork()
    net.load_state_dict(weights_before)
    obs_t = torch.stack(obs_list)
    with torch.no_grad():
        logits = apply_relay_mask(net(obs_t), obs_t)
    per_head = {h: float(Categorical(logits=logits[h]).entropy().mean())
                for h in DISCRETE_HEADS}
    expect = sum(per_head.values())
    prb_H = 0.0
    if 'prb' in logits and _PRB_POLICY == 'dirichlet':
        with torch.no_grad():
            prb_H = float(_prb_dist(logits['prb']).entropy().mean())
    expect += prb_H

    got = float(metrics['entropy'])
    assert abs(got - expect) < 1e-5, (
        'MAPPOTrainer.update entropy %r != the four-head sum plus the '
        'Dirichlet term %r (per head %r, prb %r)'
        % (got, expect, per_head, prb_H))

    # EXACTLY four discrete heads: dropping any one of them, or counting one
    # twice, moves the asserted value by more than the tolerance -- so the
    # equality above genuinely detects a head leaving or joining the sum.
    assert len(DISCRETE_HEADS) == 4, DISCRETE_HEADS
    for h, v in per_head.items():
        assert abs(v) > 1e-3, (
            'head %r contributes %r nats, so its presence in the sum is not '
            'detectable by the equality check' % (h, v))
        assert abs(got - (expect - v)) > 1e-4, h
        assert abs(got - (expect + v)) > 1e-4, h
    # the Dirichlet term is part of the live sum and is NOT bounded by ln3
    # (its differential entropy is negative near alpha ~ 1), which is exactly
    # why the old ln5+ln5+ln5+ln3 ceiling could not detect a missing head.
    discrete_ceiling = 3 * math.log(5) + math.log(3)
    assert sum(per_head.values()) <= discrete_ceiling + 1e-6, (
        per_head, discrete_ceiling)

    # and the coefficient itself is UNCHANGED by this WP: compared against
    # the frozen MAPPOConfig at the branch base, field by field, so a value
    # quietly retuned to compensate for the smaller head set would show up.
    from sixg_sim.mappo_trainer import MAPPOConfig
    cfg = MAPPOConfig()
    for field, expected in FROZEN_ENTROPY_SETTINGS.items():
        assert getattr(cfg, field) == expected, (
            'WP7 changed the entropy setting %r: %r, was %r at %s.  The WP '
            'may drop heads from the sum; it may not retune the coefficient.'
            % (field, getattr(cfg, field), expected, FROZEN_XLDET_COMMIT))


def test_the_convenience_copy_of_the_ppo_scoring_path_is_gone():
    """TRIPWIRE for WP7 review defect 3.  `RLAgent.evaluate_actions` was a
    second, uncalled implementation of "log-probs and entropy for a batch",
    and it had drifted away from the live update's PRB term.  A contract
    pinned against it asserts nothing about training.  If it (or another
    method of that shape) comes back on RLAgent, this fails and the contract
    above has to be re-aimed deliberately."""
    from sixg_sim.agent import RLAgent
    assert not hasattr(RLAgent, 'evaluate_actions'), (
        'RLAgent.evaluate_actions is back.  The PPO scoring path is '
        'MAPPOTrainer.update; a convenience copy on the agent is how the '
        'entropy contract came to be pinned against a dead function (D8.1).')


# ── 5. XL-DET: the SAME decisions, only the relay name changed ───────────
# The branch base, and the XL-DET source as it stood there.  The file is
# VENDORED (tests/data/) rather than read out of git, because the server tree
# is populated with `git archive` and has no .git directory -- a test that
# needed git would have to skip there, and a contract test that can skip is
# the WP6 defect this campaign already paid for once.  The digest is the
# output of `git show 9dec943:sixg_sim/heuristic_controller.py | sha256sum`
# and is recorded in docs/DECISIONS.md D7.3, so the vendored copy cannot be
# quietly edited into agreement with the new rule.
FROZEN_XLDET_COMMIT = '9dec943'
FROZEN_XLDET_SHA256 = (
    'eba46aef5192f4ede226e38afb53f6e297791b99a9eb9987ec222c31538d1ac1')
FROZEN_XLDET_PATH = os.path.join(ROOT, 'tests', 'data',
                                 'xldet_frozen_9dec943.py')

# MAPPOConfig's entropy settings at the branch base, read off
# `git show 9dec943:sixg_sim/mappo_trainer.py`.  WP7 changes how many heads
# the entropy SUM runs over; it must not change the coefficient itself.
FROZEN_ENTROPY_SETTINGS = {
    'entropy_coef': 0.02,
    'entropy_coef_start': 0.02,
    'entropy_decay': 0.985,
    'entropy_min': 0.001,
    'deploy_entropy': 5e-3,
}


def _frozen_xldet():
    """The XL-DET module exactly as it was BEFORE this WP, loaded as a real
    module inside the sixg_sim package so its relative imports resolve.

    This is the INDEPENDENT side of the decision-equality test: the old rule
    is re-EXECUTED from the frozen source, not re-described here.
    """
    import hashlib
    import importlib.util

    raw = open(FROZEN_XLDET_PATH, 'rb').read()
    got = hashlib.sha256(raw).hexdigest()
    assert got == FROZEN_XLDET_SHA256, (
        'tests/data/xldet_frozen_9dec943.py is not the file at %s: sha256 %s, '
        'expected %s.  The decision-equality test compares against THAT '
        'file; a modified copy would compare the new rule with itself.'
        % (FROZEN_XLDET_COMMIT, got, FROZEN_XLDET_SHA256))

    name = 'sixg_sim._frozen_xldet'
    if name in sys.modules:
        return sys.modules[name]
    import sixg_sim                                     # noqa: F401
    spec = importlib.util.spec_from_file_location(name, FROZEN_XLDET_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _StubAgent(object):
    """Just an object XL-DET can hang its `_xldet` state on."""


def _observation_series(n=120):
    """A series that exercises every branch of every XL-DET rule: the SINR
    ladder (tx_head's three outcomes and the loop-gain give-back), the
    postcard latch and its staleness expiry, and both relay outcomes."""
    import math
    out = []
    for t in range(n):
        sinr = 6.0 + 22.0 * (0.5 + 0.5 * math.sin(t / 7.0))
        o = _observation(current_tick=t)
        o.phy_mac.sinr_average = sinr
        o.phy_mac.tx_power_dbm = 20.0 + 3.0 * ((t // 3) % 4)
        o.phy_mac.relay_link_active = bool((t // 11) % 2)
        o.neighbor_radio.postcard_received = 1.0 if t % 10 == 0 else 0.0
        o.neighbor_radio.neighbour_avg_tx_power_norm = 0.3 + 0.5 * ((t // 5) % 2)
        o.neighbor_radio.interferer_count_norm = 0.8 * ((t // 4) % 2)
        o.neighbor_radio.avg_sinr = 0.4
        o.neighbor_radio.experienced_sinr_drop_norm = 0.1 * (t % 5)
        o.connectivity.relay_capable = 1.0 if (t // 13) % 2 else 0.0
        out.append(o)
    return out


def test_xldet_decisions_are_unchanged_apart_from_the_relay_aliasing():
    import copy
    import sixg_sim.heuristic_controller as NEW
    from sixg_sim.agent import RELAY_HEAD_MODES

    OLD = _frozen_xldet()
    old_modes = list(__import__('sixg_sim.phy_mac_state',
                                fromlist=['RelayMode']).RelayMode)

    series = _observation_series()
    a_old, a_new = _StubAgent(), _StubAgent()

    n_boost = n_reroute = n_xl1 = n_tx = 0
    for o in series:
        # the two implementations must never see each other's state
        d_old = OLD.decide(copy.deepcopy(o), a_old)
        d_new = NEW.decide(copy.deepcopy(o), a_new)

        assert len(d_old) == 7, ('the frozen rule returned %d slots' % len(d_old))
        assert len(d_new) == 5, ('the WP7 rule must return 5 slots, got %d'
                                 % len(d_new))

        # tx power, both MCS offsets and the PRB point: byte-for-byte, except
        # the ONE tx difference XL-1 introduced (docs/DECISIONS.md D-XL.1):
        # below the half-rate cliff the interference brake no longer holds a
        # site, so where the frozen rule held (TX_HOLD) the new one climbs.
        # The controller state is built from the observations alone, so the
        # two rules stay in step after such a tick.
        n_tx += 1
        if d_new[0] != d_old[0]:
            g = float(getattr(o.phy_mac, 'sinr_average', 0.0))
            assert (g < NEW.ICIC_FLOOR_DB and d_new[0] == NEW.TX_UP
                    and d_old[0] == NEW.TX_HOLD), ('tx', o.current_tick, d_new[0], d_old[0], g)
            n_xl1 += 1
        assert d_new[1] == d_old[1], ('mcs_e', o.current_tick)
        assert d_new[2] == d_old[2], ('mcs_g', o.current_tick)
        assert d_new[4] == d_old[6], ('prb', o.current_tick, d_new[4], d_old[6])

        # relay: THE ONLY ALLOWED DIFFERENCE is the index space.  The mode
        # the two rules NAME must be the same RelayMode.
        mode_new = RELAY_HEAD_MODES[d_new[3]]
        mode_old = old_modes[d_old[3]]
        assert mode_new == mode_old, (
            'XL-DET chose a DIFFERENT relay mode at tick %d: %s (head %d) '
            'vs %s (enum %d).  The WP7 mapping is wrong.'
            % (o.current_tick, mode_new, d_new[3], mode_old, d_old[3]))

        n_boost += (mode_new.name == 'CAPACITY_BOOST')
        n_reroute += (mode_new.name == 'LOCAL_REROUTE')

    # the comparison is only worth anything if both relay outcomes occurred
    assert n_boost > 0 and n_reroute > 0, (n_boost, n_reroute)
    # ... and the XL-1 exception must not have swallowed the tx comparison
    assert n_tx > 0 and n_xl1 < n_tx, (n_xl1, n_tx)
    print('  tx compared on %d ticks, %d of them the XL-1 climb below the cliff' % (n_tx, n_xl1))
    # ... and if the tx rule actually moved
    txs = set(NEW.decide(copy.deepcopy(o), _StubAgent())[0] for o in series)
    assert len(txs) >= 2, txs


def test_xldets_prb_point_still_holds_against_the_engine():
    """WP7 reshaped decide()'s return tuple; the PRB point it returns is
    still a fixed point of the engine's two projections, and the engine's
    admission thresholds still admit every class at that point.  (The
    separate "XL-DET is blind to the global columns" check lives in
    test_xldet.py, which poisons the real observation objects.)"""
    import sixg_sim.heuristic_controller as XL

    prb, modes = XL.verify_prb_point()
    assert len(prb) == 3, prb
    assert abs(sum(prb) - 1.0) < 1e-9, prb
    assert modes, 'verify_prb_point returned no admission modes'
    # the engine's own thresholds must still ADMIT every class at that point
    assert all(str(getattr(m, 'name', m)).upper() == 'ADMIT'
               for m in modes.values()), modes
    assert len(modes) == 4, modes


# ── 6. the tripwire: no contract test in this WP may silently skip ───────
def test_no_wp7_test_can_silently_skip():
    """A contract test that returns early on a missing precondition reports
    a pass while checking nothing.  Round 3 of WP6 shipped three of those.
    No test function in the WP7 files may contain a bare `return`, a call
    to the test framework's skip/xfail markers, or a `try/except` that
    swallows its own assertion.

    The detection below is an AST scan of the call names ('skip', 'skipTest',
    'xfail', 'exit'), NOT a substring search, so this file deliberately does
    not spell the dotted `pytest`-skip idiom out: the WP6 tripwire
    tests/test_wp6_parity.py::test_no_test_file_skips_quietly greps every
    test file in the tree for those literals, and a file that names one in
    prose trips it (D7.9)."""
    bad = []
    for fn in ('test_wp7_action_space.py', 'test_wp7_controls.py'):
        path = os.path.join(ROOT, 'tests', fn)
        assert os.path.exists(path), 'missing WP7 test file %s' % fn
        tree = ast.parse(open(path, encoding='utf-8').read(), filename=path)
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef)
                    and node.name.startswith('test_')):
                continue
            # the test BODY only: a nested helper's own `return` is how a
            # helper returns a value, not a way for the test to skip
            nested = set()
            for inner in ast.walk(node):
                if inner is node:
                    continue
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Lambda)):
                    for k in ast.walk(inner):
                        nested.add(id(k))
            for sub in ast.walk(node):
                if id(sub) in nested:
                    continue
                if isinstance(sub, ast.Return):
                    bad.append('%s::%s returns early at line %d'
                               % (fn, node.name, sub.lineno))
                if isinstance(sub, ast.Call):
                    f = sub.func
                    dotted = getattr(f, 'attr', getattr(f, 'id', ''))
                    if dotted in ('skip', 'skipTest', 'xfail', 'exit'):
                        bad.append('%s::%s calls %s at line %d'
                                   % (fn, node.name, dotted, sub.lineno))
                if isinstance(sub, ast.ExceptHandler):
                    names = []
                    t = sub.type
                    if t is None:
                        names = ['<bare>']
                    elif isinstance(t, ast.Name):
                        names = [t.id]
                    elif isinstance(t, ast.Tuple):
                        names = [getattr(e, 'id', '?') for e in t.elts]
                    if any(n in ('<bare>', 'Exception', 'BaseException',
                                 'AssertionError', 'ImportError')
                           for n in names):
                        bad.append('%s::%s swallows %r at line %d'
                                   % (fn, node.name, names, sub.lineno))
    assert not bad, 'WP7 tests can skip or swallow:\n  ' + '\n  '.join(bad)


if __name__ == '__main__':
    for _n, _f in sorted(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
            print('ok', _n)
