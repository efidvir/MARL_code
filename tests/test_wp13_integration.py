# -*- coding: utf-8 -*-
"""WP13 -- the observation fields removed after the production-length
constancy dump (D13.3), the schema id that removal changes (D13.4), and the
tool that turns the dump reports into the per-field evidence.

Fast: no simulator is built.  Run with

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp13_integration
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _rmtree(d):
    """Remove a scratch directory, retrying for up to 10 s: the tree is on
    NFS, where a file a just-finished child process still holds open is
    silly-renamed to .nfsXXXX and keeps its directory from being removed
    until the server lets go (the negative-control runners retry the same
    way).  The hygiene test (tests/test_zz_tree_hygiene.py) found one such
    leftover after the first removal attempt."""
    import time as _time
    for _ in range(20):
        shutil.rmtree(d, ignore_errors=True)
        if not os.path.exists(d):
            return
        _time.sleep(0.5)


@contextlib.contextmanager
def _scratch(prefix):
    """A scratch directory inside the tree (output/tmp_tests; the seed audit
    skips it by design), removed when the block ends, also on a failure.
    Never the system default: tempfile.gettempdir() falls back to the
    current directory -- the tree root, where the suite runs -- when `/` is
    full (D12F.2)."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(prefix=prefix, dir=base)
    try:
        yield d
    finally:
        _rmtree(d)

# The evidence the removal rests on: tools/constancy_combine.py over the six
# 3600-tick passes at c148cc9 (seeds 82, 50, 71 x recovery, rescue_ops, with
# D13.1 and D13.2 applied), committed with the tree.
EVIDENCE = os.path.join(ROOT, 'docs', 'wp13', 'constancy_c148cc9.json')
# The id every checkpoint written between the WP7 head change and WP13
# carries (docs/DECISIONS.md D7.5, D-MRG.11).
PRE_WP13_SCHEMA_ID = 'obs1-48327117f57a86a2'


def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'constancy_combine', os.path.join(ROOT, 'tools', 'constancy_combine.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(values):
    n = len(values)
    mean = sum(values) / n
    return {'n': n, 'min': min(values), 'max': max(values), 'mean': mean,
            'variance': sum((v - mean) ** 2 for v in values) / n,
            'distinct': len(set(values)), 'modal_value': values[0],
            'constant_fraction': max(values.count(v) for v in values) / n,
            'constant': min(values) == max(values)}


def _report(path, passes, schema='obs1-test'):
    rep = {'passes': []}
    for scen, seed, ticks, fields in passes:
        n = len(next(iter(fields.values())))
        rep['passes'].append({
            'scenario': scen, 'seed': seed, 'ticks': ticks,
            'schema_id': schema, 'samples_per_field': n,
            'fields': {k: _stats(v) for k, v in fields.items()}})
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh)
    return path


# ── 1. the combiner: the intersection, and nothing from an empty set ──────
def test_the_combiner_removes_only_what_is_constant_in_every_pass():
    """dead: the same value in every pass.  kept: constant in some passes
    only, constant everywhere but at DIFFERENT values (it tells instances
    apart), and anything that moves."""
    t = _tool()
    with _scratch('wp13_comb_') as d:
        a = _report(os.path.join(d, 'a.json'), [
            ('recovery', 1, 100, {'dead': [0.0] * 4, 'some': [0.0] * 4,
                                  'twovals': [0.0] * 4, 'moves': [0.0, 1.0, 0.5, 0.5]}),
            ('rescue_ops', 1, 100, {'dead': [0.0] * 4, 'some': [0.0, 0.2, 0.0, 0.0],
                                    'twovals': [1.0] * 4, 'moves': [1.0] * 4})])
        b = _report(os.path.join(d, 'b.json'), [
            ('recovery', 2, 100, {'dead': [0.0] * 4, 'some': [0.0] * 4,
                                  'twovals': [1.0] * 4, 'moves': [0.3] * 4})])
        rc = t.main([a, b, '--require-passes', '3', '--require-ticks', '100',
                     '--json', os.path.join(d, 'out.json')])
        assert rc == 0
        with open(os.path.join(d, 'out.json'), encoding='utf-8') as fh:
            out = json.load(fh)
        assert out['constant_in_every_pass'] == ['dead'], out['constant_in_every_pass']
        assert sorted(out['constant_in_some_pass']) == ['moves', 'some', 'twovals']
        assert out['fields']['dead']['constant_in'] == 3
        assert out['fields']['twovals']['constant_in'] == 3
        assert out['fields']['twovals']['dead'] is False
        assert out['passes'] == 3 and len(out['fields']) == 4


def test_the_combiner_refuses_an_empty_a_short_or_a_mixed_set():
    t = _tool()
    with _scratch('wp13_comb_') as d:
        ok = _report(os.path.join(d, 'ok.json'),
                     [('recovery', 1, 100, {'f': [0.0, 1.0]})])
        refused = 0
        # a report with no pass
        empty = os.path.join(d, 'empty.json')
        with open(empty, 'w', encoding='utf-8') as fh:
            json.dump({'passes': []}, fh)
        try:
            t.main([empty])
        except SystemExit as e:
            assert 'no pass' in str(e), e
            refused += 1
        else:
            raise AssertionError('a report with no pass was accepted')
        # two schemas
        other = _report(os.path.join(d, 'other.json'),
                        [('recovery', 2, 100, {'f': [0.0, 1.0]})], schema='obs1-x')
        try:
            t.main([ok, other])
        except SystemExit as e:
            assert 'different schemas' in str(e), e
            refused += 1
        else:
            raise AssertionError('passes from two schemas were combined')
        # a pass with no sample
        zero = os.path.join(d, 'zero.json')
        with open(zero, 'w', encoding='utf-8') as fh:
            json.dump({'passes': [{'scenario': 'recovery', 'seed': 3, 'ticks': 100,
                                   'schema_id': 'obs1-test',
                                   'samples_per_field': 0, 'fields': {}}]}, fh)
        try:
            t.main([ok, zero])
        except SystemExit as e:
            assert 'no sample' in str(e), e
            refused += 1
        else:
            raise AssertionError('a pass with no sample was accepted')
        # the requirements: a 100-tick set is not production length
        assert t.main([ok, '--require-ticks', '3600']) == 1
        assert t.main([ok, '--require-passes', '6']) == 1
        assert t.main([ok, '--require-scenarios', 'recovery,rescue_ops']) == 1
        assert t.main([ok, '--require-seeds', '3']) == 1
        assert t.main([ok]) == 0
        assert refused == 3


# ── 2. the schema removed exactly what the evidence shows constant ────────
def _evidence():
    assert os.path.isfile(EVIDENCE), 'the WP13 evidence file is missing'
    with open(EVIDENCE, encoding='utf-8') as fh:
        return json.load(fh)


def test_the_evidence_is_production_length_both_scenarios_three_seeds():
    ev = _evidence()
    assert ev['passes'] == 6, ev['passes']
    assert sorted(ev['seeds_by_scenario']) == ['recovery', 'rescue_ops']
    for scen, seeds in ev['seeds_by_scenario'].items():
        assert sorted(seeds) == [50, 71, 82], (scen, seeds)
    n = 0
    for name, e in ev['fields'].items():
        assert e['passes'] == 6, name
        for p in e['per_pass']:
            assert p['ticks'] == 3600, (name, p)
            assert p['n'] > 0, (name, p)
        n += 1
    # the pre-WP13 schema: 26 actor + 15 critic-global fields
    assert n == 41, n
    assert ev['schema_id'] == PRE_WP13_SCHEMA_ID, ev['schema_id']


def test_the_schema_removed_exactly_the_fields_constant_in_every_pass():
    """D13.3.  Every removed field is constant, at one value, in all six
    passes; every field that stays moves in at least one; the two sets make
    up the whole pre-WP13 schema."""
    from sixg_sim.agent import (ACTOR_OBS_NAMES, CRITIC_GLOBAL_NAMES,
                                REMOVED_CONSTANT_FIELDS)
    ev = _evidence()
    dead = set(ev['constant_in_every_pass'])
    removed = set(REMOVED_CONSTANT_FIELDS)
    kept = list(ACTOR_OBS_NAMES) + list(CRITIC_GLOBAL_NAMES)
    assert removed == dead, (
        'removed %s, constant in every pass %s'
        % (sorted(removed), sorted(dead)))
    assert not removed & set(kept), sorted(removed & set(kept))
    checked = 0
    for name in kept:
        e = ev['fields'].get(name)
        assert e is not None, '%s is not in the evidence' % name
        assert not e['dead'], '%s is constant in every pass but kept' % name
        assert e['constant_in'] < e['passes'] or len(e['constant_values']) > 1
        checked += 1
    for name in removed:
        e = ev['fields'][name]
        assert e['dead'] and e['constant_in'] == e['passes'] == 6, name
        assert len(e['constant_values']) == 1, name
        checked += 1
    assert checked == len(ev['fields']) == 41, checked
    assert len(ACTOR_OBS_NAMES) == 26 and len(CRITIC_GLOBAL_NAMES) == 11


def test_the_short_pass_constants_that_move_at_production_length_stay():
    """The 600-tick passes of the merge (D-MRG.11) also showed
    a_emergency_fraction and g_pending_iops constant.  At 3600 ticks both
    MOVE in Scenario B (the rescue force arrives at tick 1400 with emergency
    UEs), so the evidence keeps them -- and so does the schema."""
    from sixg_sim.agent import ACTOR_OBS_NAMES, CRITIC_GLOBAL_NAMES
    ev = _evidence()
    n = 0
    for name, where in (('a_emergency_fraction', ACTOR_OBS_NAMES),
                        ('g_pending_iops', CRITIC_GLOBAL_NAMES)):
        e = ev['fields'][name]
        assert not e['dead'], name
        moved = [p for p in e['per_pass'] if not p['constant']]
        assert moved and all(p['scenario'] == 'rescue_ops' for p in moved), (
            name, [(p['scenario'], p['seed']) for p in moved])
        assert len(moved) == 3, (name, len(moved))
        assert name in where, '%s moves but was removed' % name
        n += 1
    assert n == 2


# ── 3. the schema id moved, and the old one is refused ────────────────────
def test_the_pre_wp13_schema_id_is_refused_and_minting_stamps_the_new_one():
    """D13.4.  Removing fields changes SCHEMA_ID; every checkpoint written
    before carries the old id and the strict gate refuses it, while a
    checkpoint minted by this tree (the untrained controls' generator uses
    the same helper) carries the new id and passes.  The actor stays 26
    wide, so an old actor checkpoint would LOAD by shape: the id is the only
    thing that can refuse it, which is why it covers the critic list too."""
    import torch
    from sixg_sim.agent import (SCHEMA_ID, ACTOR_OBS_DIM, CRITIC_OBS_DIM,
                                assert_checkpoint_schema,
                                checkpoint_schema_id,
                                mint_schema_current_checkpoint)
    assert SCHEMA_ID.startswith('obs1-') and SCHEMA_ID != PRE_WP13_SCHEMA_ID
    assert (ACTOR_OBS_DIM, CRITIC_OBS_DIM) == (26, 37)
    old = {'state_dict': {}, 'version': 0,
           'metadata': {'provenance': {'schema_id': PRE_WP13_SCHEMA_ID}}}
    try:
        assert_checkpoint_schema(old, where='wp13')
    except ValueError as e:
        assert PRE_WP13_SCHEMA_ID in str(e) and SCHEMA_ID in str(e), e
    else:
        raise AssertionError('the pre-WP13 schema id was accepted')
    with _scratch('wp13_mint_') as _md:
        path = os.path.join(_md, 'minted.pt')
        mint_schema_current_checkpoint(path, torch_seed=1234)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint_schema_id(ck) == SCHEMA_ID
        assert assert_checkpoint_schema(ck, where='wp13') == SCHEMA_ID
        assert tuple(ck['state_dict']['fc1.weight'].shape)[1] == ACTOR_OBS_DIM == 26
        from sixg_sim.agent import CriticNetwork
        assert CriticNetwork().fc1.in_features == CRITIC_OBS_DIM == 37


def test_the_untrained_control_generator_mints_under_the_current_id():
    """make_untrained_checkpoints.py stamps SCHEMA_ID as imported at run
    time (not a literal), through mint_schema_current_checkpoint."""
    src = open(os.path.join(ROOT, 'make_untrained_checkpoints.py'),
               encoding='utf-8').read()
    assert 'mint_schema_current_checkpoint' in src
    assert 'SCHEMA_ID' in src
    assert PRE_WP13_SCHEMA_ID not in src
    assert 'obs1-' not in src, 'a literal schema id in the generator'
