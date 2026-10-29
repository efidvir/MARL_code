#!/usr/bin/env python3
"""R3-lite, the SECOND CONFIRMATORY FAMILY (family E, D-R3.3).

Family E is analysed by the unchanged pre-registered analysis on a pickle set
assembled by tools/assemble_family.py: every comparator pass is the S6 pass,
every marl_static pass is the ensemble's pass of the same instance.  These
tests pin

  * the assembler: what it builds, what it verifies, and every input it must
    refuse (each refusal is a negative control of the positive case);
  * the step `family_e` of confirmatory/stage_analysis.sh (chain S7e): the
    order (family-1 report first, its sha256 recorded; the assembly path
    proved on family 1 itself; only then family E), the inputs it hands the
    tools, and every way it must stop.

No simulator pass is run and no seed of a confirmatory or reserved block is
named: the pickles are synthetic and the profile is the rehearsal profile.
"""
import contextlib
import copy
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, 'confirmatory'), os.path.join(ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
import chain_plan as CP            # noqa: E402
import assemble_family as AF       # noqa: E402

REH = 'r3lite_rehearsal'
CK_A = ['a' * 64]
CK_E = ['a' * 64, 'b' * 64, 'c' * 64, 'd' * 64]
SCEN = AF.SCEN_KEYS
TOOL = os.path.join(ROOT, 'tools', 'assemble_family.py')


def _mkdtemp(prefix):
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


@contextlib.contextmanager
def _scratch(prefix):
    d = _mkdtemp(prefix)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _linux_or_fail(what):
    if not sys.platform.startswith('linux') or shutil.which('bash') is None:
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run FAILS, '
                             'it never skips: ' + what)


# ------------------------------------------------------------ synthetic pickles
def _run(arm, sk, seed, tag=0.0):
    """A pass: the offered series depends on the INSTANCE only (scenario and
    seed), everything else on the arm as well."""
    k = SCEN.index(sk)
    return {'offered': [float(seed), float(seed) + k, 7.5],
            'conn': [float(len(arm)) + seed + k + tag, float('nan'), 3.25],
            'fragments': [1, 2, seed],
            'relay_lifetimes': [{'link_id': 'l%d' % seed, 'created': 1, 'load': 0.5 + tag}],
            'label': '%s|%s|%d' % (arm, sk, seed)}


def _prov(ckpt, **over):
    p = {'provenance_version': 1, 'tree_head': 'faketest', 'tree_sha256': 'f' * 64,
         'schema_id': 'obs-fake', 'reward_id': 'rew-fake', 'pythonhashseed': '0',
         # the synthetic passes are three ticks long (see _run)
         'total_ticks': 3, 'severance_tick': 200, 'rescue_arrival_tick': 1400,
         'overrides': {}, 'python': '3.12', 'torch': '2', 'numpy': '1', 'networkx': '3',
         'driver': 'run_timeline_comparison',
         'marl_env': {'MARL_CHAIN_INV': 'x', 'MARL_EVAL_CHECKPOINT': '/p', 'MARL_POOL_WORKERS': '24',
                      'MARL_POSTCARD_ALWAYS': '1', 'MARL_STEER_MODEL': 'actionable'},
         'checkpoint': '/p', 'checkpoint_sha256': list(ckpt)}
    p.update(over)
    return p


def _pk(path, arms, seeds, ckpt, tag=0.0, prov=None, runs=None):
    P = {sk: {a: {s: _run(a, sk, s, tag) for s in seeds} for a in arms} for sk in SCEN}
    if runs:
        runs(P)
    P['seeds'] = list(seeds)
    P['provenance'] = prov or _prov(ckpt)
    P['pass_stats'] = {'%s|%s|%d' % (a, n, s): {'wall_s': 1.0, 'from': os.path.basename(path)}
                       for a in arms for n in ('recovery', 'rescue_ops') for s in seeds}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(P, f)
    return path


ARMS8 = ['marl_static', 'xldet', 'sdn']


def _inputs(d, **kw):
    """Two comparator pickles (seeds 1,2 and 3,4; marl_static = one policy)
    and one arm-source pickle (marl_static = the ensemble, seeds 1-4)."""
    c0 = _pk(os.path.join(d, 'in', 's6_b0.pkl'), ARMS8, [1, 2], CK_A)
    c1 = _pk(os.path.join(d, 'in', 's6_b1.pkl'), ARMS8, [3, 4], CK_A)
    src = _pk(os.path.join(d, 'in', 'ens.pkl'), ['marl_static'], kw.get('src_seeds', [1, 2, 3, 4]),
              kw.get('src_ckpt', CK_E), tag=100.0, prov=kw.get('src_prov'), runs=kw.get('src_runs'))
    return [c0, c1], [src]


def _assemble(d, comps, srcs, expect=CK_E, prefix='e_', arm='marl_static', ref='xldet',
              out=None, rec=None, keys=None):
    out = out or os.path.join(d, 'out')
    rec = rec or os.path.join(d, 'record.json')
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, TOOL, '--comparators', ','.join(comps),
           '--arm-source', ','.join(srcs), '--arm', arm, '--reference-arm', ref,
           '--expect-checkpoint-sha256', ','.join(expect),
           '--out-dir', out, '--prefix', prefix, '--record', rec]
    if keys is not None:
        kf = os.path.join(d, 'expect.keys')
        with open(kf, 'w') as f:
            f.write('\n'.join(keys) + '\n')
        cmd += ['--expect-keys', kf]
    r = subprocess.run(cmd, capture_output=True, text=True)
    left = sorted(os.listdir(out)) if os.path.isdir(out) else []
    recd = None
    if os.path.isfile(rec):
        try:
            recd = json.load(open(rec))
        except ValueError:
            recd = None
    return r, recd, out, left


def _keys(arms, seeds):
    return ['%s|%s|%d' % (a, n, s) for a in arms for n in ('recovery', 'rescue_ops') for s in seeds]


def _private_left(d):
    """Private working directories the tool may have left beside --out-dir."""
    return [x for dp, dn, _f in os.walk(d) for x in dn if x.startswith(AF.PRIVATE_PREFIX)]


def _same(a, b):
    return AF.run_digest(a) == AF.run_digest(b)


# ============================================================ the run digest
def test_the_run_digest_is_canonical_and_sensitive():
    import numpy as np
    a = {'conn': [1.0, 2.5, float('nan')], 'n': [1, 2, 3], 'd': {'b': [1.0, 2], 'a': None},
         'arr': np.array([1.5, 2.5]), 'g': np.float64(2.0), 't': (1, 2), 's': 'x'}
    b = {'s': 'x', 't': (1, 2), 'g': np.float64(2.0), 'arr': np.array([1.5, 2.5]),
         'd': {'a': None, 'b': [1.0, 2]}, 'n': [1, 2, 3], 'conn': [1.0, 2.5, float('nan')]}
    assert _same(a, b), 'independent of dict order; NaN equals NaN'
    assert _same(pickle.loads(pickle.dumps(a)), a), 'independent of a pickle round trip'
    n = 0
    for key, new in (('conn', [1.0, 2.5, 0.0]), ('conn', [1.0, 2.5]), ('n', [1, 2, 3.0]),
                     ('n', [1, 23]), ('d', {'b': [1.0, 2], 'a': 0}), ('arr', np.array([1.5, 2.6])),
                     ('arr', np.array([1.5, 2.5], dtype='float32')), ('g', np.float64(2.5)),
                     ('t', [1, 2]), ('s', 'y')):
        c = dict(a)
        c[key] = new
        assert not _same(c, a), (key, new)
        n += 1
    assert n == 10
    assert not _same({'x': [0.0, 1.0]}, {'x': [-0.0, 1.0]}), 'signed zero keeps its bits'
    assert not _same({'x': []}, {'x': ()}) and not _same({'x': [1, 2]}, {'x': [12]})
    assert not _same({'a': 1}, {'a': 1, 'b': None})
    # every float is eight bytes and every NaN is the one canonical NaN, on
    # the per-tick fast path and on the general path alike: a NaN followed by
    # a string cannot spell another float (the collision the review found)
    import struct
    other = struct.unpack('>d', b'nans3:ab')[0]
    assert not _same([float('nan'), 'abN'], [other, None])
    neg_nan = struct.unpack('>d', struct.pack('>Q', 0xfff8000000000001))[0]
    assert _same([float('nan'), 1.0], [neg_nan, 1.0]) and _same([float('nan'), None], [neg_nan, None])
    assert _same(np.array([float('nan'), 1.0]), np.array([neg_nan, 1.0]))
    assert not _same([1, 0], [True, False]) and not _same(['1,2'], [1, 2]) and not _same([1, 2], ['1', '2'])
    assert not _same({'1': 1}, {1: 1}) and not _same([[1.0], [2.0]], [1.0, 2.0])
    assert not _same([1.0], [1]) and not _same('a', b'a')
    assert _same({'s': {3, 1, 2}}, {'s': {2, 3, 1}}) and not _same({'s': {1, 2}}, {'s': [1, 2]})


# ============================================================== the assembler
def test_the_assembler_builds_family_e_on_the_same_comparator_passes():
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        r, rec, out, left = _assemble(d, comps, srcs)
        assert r.returncode == 0, r.stdout + r.stderr
        assert 'ASSEMBLE VERIFY PASS arm=marl_static derived=2 comparator_runs=16 arm_runs=8 ' \
               'offered_pairs=8 seeds=4' in r.stdout, r.stdout
        assert left == ['e_s6_b0.pkl', 'e_s6_b1.pkl'], left
        S = pickle.load(open(srcs[0], 'rb'))
        n_cmp = n_arm = 0
        for c in comps:
            P = pickle.load(open(c, 'rb'))
            D = pickle.load(open(os.path.join(out, 'e_' + os.path.basename(c)), 'rb'))
            assert sorted(D) == sorted(P) and D['seeds'] == P['seeds']
            for sk in SCEN:
                assert sorted(D[sk]) == sorted(P[sk]) == sorted(ARMS8)
                for a in ('xldet', 'sdn'):
                    for s in P['seeds']:
                        assert _same(D[sk][a][s], P[sk][a][s]); n_cmp += 1
                for s in P['seeds']:
                    # the learned arm is the ENSEMBLE's pass, not the S6 policy's
                    assert _same(D[sk]['marl_static'][s], S[sk]['marl_static'][s]); n_arm += 1
                    assert not _same(D[sk]['marl_static'][s], P[sk]['marl_static'][s])
                assert sorted(D[sk]['marl_static']) == P['seeds'], 'no seed of another block'
            # the derived provenance: the comparators' code identity, the ensemble's checkpoints
            assert D['provenance']['checkpoint_sha256'] == CK_E
            assert D['provenance']['tree_sha256'] == P['provenance']['tree_sha256']
            af = D['provenance']['assembled_family']
            assert af['comparators_from'] == {'path': c, 'sha256': _sha256(c)} and af['arm_from'] == srcs
            # pass statistics follow the passes
            for k, v in D['pass_stats'].items():
                assert v['from'] == ('ens.pkl' if k.startswith('marl_static|') else os.path.basename(c)), k
            assert len(D['pass_stats']) == len(P['pass_stats'])
        assert (n_cmp, n_arm) == (16, 8)
        # the record: every input and output by sha256, the counts, the tool itself
        assert rec['status'] == 'PASS' and rec['seeds'] == [1, 2, 3, 4]
        assert rec['verify'] == {'comparator_runs': 16, 'arm_runs': 8, 'offered_pairs': 8,
                                 'derived_pickles': 2}
        assert rec['tool_sha256'] == _sha256(TOOL) and rec['expect_checkpoint_sha256'] == CK_E
        assert [(x['path'], x['sha256']) for x in rec['comparators']] == [(c, _sha256(c)) for c in comps]
        assert [(x['path'], x['sha256']) for x in rec['arm_source']] == [(s, _sha256(s)) for s in srcs]
        assert [(x['path'], x['sha256']) for x in rec['outputs']] == [
            (os.path.join(out, f), _sha256(os.path.join(out, f))) for f in left]
        # the inputs were not touched
        assert pickle.load(open(comps[0], 'rb'))['provenance']['checkpoint_sha256'] == CK_A


def test_the_control_set_assembled_from_the_comparators_alone_is_the_comparator_set():
    """The positive control of chain S7e: with the arm taken from the S6
    pickles themselves, every run of the derived set is the S6 run."""
    with _scratch('r3_fam_') as d:
        comps, _srcs = _inputs(d)
        r, rec, out, left = _assemble(d, comps, comps, expect=CK_A, prefix='c_')
        assert r.returncode == 0, r.stdout + r.stderr
        n = 0
        for c in comps:
            P = pickle.load(open(c, 'rb'))
            D = pickle.load(open(os.path.join(out, 'c_' + os.path.basename(c)), 'rb'))
            for sk in SCEN:
                for a in ARMS8:
                    for s in P['seeds']:
                        assert _same(D[sk][a][s], P[sk][a][s]); n += 1
            assert D['provenance']['checkpoint_sha256'] == CK_A
        assert n == 24 and rec['verify']['arm_runs'] == 8
        # NEGATIVE CONTROL: the control is refused for the ensemble's pins
        r, rec, out2, left = _assemble(os.path.join(d, 'x'), comps, comps, expect=CK_E, prefix='c_')
        assert r.returncode == 1 and 'records checkpoint sha256' in r.stdout and left == []


def test_the_assembler_refuses_every_input_that_is_not_the_registered_family():
    def drop(sk, arm, seed):
        return lambda P: P[sk][arm].pop(seed)

    def offered(P):
        P['rescue_multi']['marl_static'][3]['offered'][1] += 1e-9

    cases = [
        # (name, kwargs of _inputs / change, expected reason)
        ('a seed of the block has no ensemble pass', dict(src_seeds=[1, 2, 3]), 'no pass for seeds [4]'),
        ('one scenario of one seed is missing', dict(src_runs=drop('rescue_multi', 'marl_static', 2)),
         'rescue_multi: no pass for seeds [2]'),
        ('the source holds a seed outside the block', dict(src_seeds=[1, 2, 3, 4, 5]),
         'seeds [5] are in no comparator pickle'),
        ('the checkpoint list is not the pinned list', dict(src_ckpt=CK_E[:3]), 'records checkpoint sha256'),
        ('the checkpoint list is in another order', dict(src_ckpt=CK_E[::-1]), 'records checkpoint sha256'),
        ('one member of the ensemble differs', dict(src_ckpt=CK_E[:3] + ['e' * 64]), 'records checkpoint sha256'),
        ('another code tree', dict(src_prov=_prov(CK_E, tree_sha256='0' * 64)), "provenance 'tree_sha256' differs"),
        ('another hash seed', dict(src_prov=_prov(CK_E, pythonhashseed='1')), "provenance 'pythonhashseed' differs"),
        ('another pass length', dict(src_prov=_prov(CK_E, total_ticks=600)), "provenance 'total_ticks' differs"),
        ('another schema', dict(src_prov=_prov(CK_E, schema_id='obs-other')), "provenance 'schema_id' differs"),
        # the engine environment: another VALUE of a variable both sides set, a variable
        # only one side sets, and a variable only the other side sets
        ('another value of an engine variable',
         dict(src_prov=_prov(CK_E, marl_env={'MARL_POSTCARD_ALWAYS': '1', 'MARL_STEER_MODEL': 'other'})),
         "provenance 'marl_env' differs"),
        ('an engine variable the comparators do not set',
         dict(src_prov=_prov(CK_E, marl_env={'MARL_POSTCARD_ALWAYS': '1', 'MARL_STEER_MODEL': 'actionable',
                                             'MARL_TRANSPORT_POWER': 'planning'})),
         "provenance 'marl_env' differs"),
        ('an engine variable the ensemble pass does not set',
         dict(src_prov=_prov(CK_E, marl_env={'MARL_POSTCARD_ALWAYS': '1'})),
         "provenance 'marl_env' differs"),
        ('a pass that was not run on the same instance', dict(src_runs=offered),
         'the offered series of arm marl_static'),
    ]
    n = 0
    for name, kw, why in cases:
        with _scratch('r3_fam_') as d:
            comps, srcs = _inputs(d, **kw)
            r, rec, out, left = _assemble(d, comps, srcs)
            assert r.returncode == 1, (name, r.stdout)
            assert why in r.stdout, (name, r.stdout)
            assert 'ASSEMBLE REFUSED' in r.stdout and 'VERIFY PASS' not in r.stdout, name
            assert left == [], '%s: nothing may be left in the output directory (%s)' % (name, left)
            assert rec['status'] == 'REFUSED' and rec['failures'] and not rec['outputs'], name
            n += 1
    assert n == 14
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        # the same seed in two source pickles / in two comparator pickles / a pickle named twice
        dup = _pk(os.path.join(d, 'in', 'ens2.pkl'), ['marl_static'], [4], CK_E, tag=100.0)
        r, rec, out, left = _assemble(os.path.join(d, 'a'), comps, srcs + [dup])
        assert r.returncode == 1 and 'occurs in' in r.stdout and left == [], r.stdout
        c2 = _pk(os.path.join(d, 'in', 's6_b2.pkl'), ARMS8, [4], CK_A)
        r, rec, out, left = _assemble(os.path.join(d, 'b'), comps + [c2], srcs)
        assert r.returncode == 1 and 'seed 4 occurs in comparator pickles' in r.stdout and left == []
        r, rec, out, left = _assemble(os.path.join(d, 'c'), comps + [comps[0]], srcs)
        assert r.returncode == 1 and 'named twice' in r.stdout and left == []
        # no reference comparator; a missing input; a source without the arm
        r, rec, out, left = _assemble(os.path.join(d, 'e'), comps, srcs, ref='ospf')
        assert r.returncode == 1 and "holds no reference arm 'ospf'" in r.stdout and left == []
        r, rec, out, left = _assemble(os.path.join(d, 'f'), comps, srcs, ref='marl_static')
        assert r.returncode == 1 and 'is the arm being assembled' in r.stdout and left == []
        r, rec, out, left = _assemble(os.path.join(d, 'g'), comps, [os.path.join(d, 'in', 'none.pkl')])
        assert r.returncode == 1 and 'missing or empty input' in r.stdout and left == []
        noarm = _pk(os.path.join(d, 'in', 'noarm.pkl'), ['xldet'], [1, 2, 3, 4], CK_E)
        r, rec, out, left = _assemble(os.path.join(d, 'h'), comps, [noarm])
        assert r.returncode == 1 and 'holds no pass of arm marl_static' in r.stdout and left == []
        # a legitimate difference is NOT refused: the invocation name, the checkpoint path, the workers
        ok = _pk(os.path.join(d, 'in', 'ens_ok.pkl'), ['marl_static'], [1, 2, 3, 4], CK_E, tag=100.0,
                 prov=_prov(CK_E, marl_env={'MARL_CHAIN_INV': 'other', 'MARL_EVAL_CHECKPOINT': '/q,/r',
                                            'MARL_POOL_WORKERS': '8', 'MARL_POSTCARD_ALWAYS': '1',
                                            'MARL_STEER_MODEL': 'actionable'}))
        r, rec, out, left = _assemble(os.path.join(d, 'i'), comps, [ok])
        assert r.returncode == 0 and len(left) == 2, r.stdout
        # and the unchanged input still assembles (the refusals above are not a broken harness)
        r, rec, out, left = _assemble(os.path.join(d, 'j'), comps, srcs)
        assert r.returncode == 0 and len(left) == 2, r.stdout


def test_the_assembler_verifies_what_it_wrote_from_disk():
    """NEGATIVE CONTROL of the read-back: a derived pickle that does not hold
    the runs it was built from is refused and removed."""
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        real_load = AF._load
        seen = []

        def tampering_load(path):
            P = real_load(path)
            if AF.PRIVATE_PREFIX in path:
                seen.append(path)
                if len(seen) == 2:            # the second derived pickle, read back
                    P['recovery_multi']['xldet'][3]['conn'][0] += 1.0
            return P
        AF._load = tampering_load
        try:
            ok, rec = AF.assemble(comps, srcs, 'marl_static', CK_E, os.path.join(d, 'out'), 'e_',
                                  say=lambda *_a: None)
        finally:
            AF._load = real_load
        assert not ok and len(seen) == 2
        assert any('differs from its input' in f for f in rec['failures']), rec['failures']
        assert not os.path.exists(os.path.join(d, 'out')) and not _private_left(d), 'nothing is left behind'
        # an INTERRUPT in the read-back cleans up too, and is passed on
        def interrupting_load(path):
            if AF.PRIVATE_PREFIX in path:
                raise KeyboardInterrupt()
            return real_load(path)
        AF._load = interrupting_load
        try:
            try:
                AF.assemble(comps, srcs, 'marl_static', CK_E, os.path.join(d, 'out'), 'e_',
                            say=lambda *_a: None)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('the interrupt was swallowed')
        finally:
            AF._load = real_load
        assert not os.path.exists(os.path.join(d, 'out')) and not _private_left(d)
        # an input that changes while it is assembled is refused
        def touching_load(path):
            P = real_load(path)
            if AF.PRIVATE_PREFIX in path and not touched:
                touched.append(path)
                with open(comps[1], 'ab') as f:
                    f.write(b' ')
            return P
        touched = []
        keep = open(comps[1], 'rb').read()
        AF._load = touching_load
        try:
            ok, rec = AF.assemble(comps, srcs, 'marl_static', CK_E, os.path.join(d, 'out'), 'e_',
                                  say=lambda *_a: None)
        finally:
            AF._load = real_load
            with open(comps[1], 'wb') as f:
                f.write(keep)
        assert not ok and any('changed while it was being assembled' in f for f in rec['failures'])
        assert not os.path.exists(os.path.join(d, 'out')) and not _private_left(d)
        # ... and without the tampering the same call passes
        ok, rec = AF.assemble(comps, srcs, 'marl_static', CK_E, os.path.join(d, 'out'), 'e_',
                              say=lambda *_a: None)
        assert ok and rec['verify']['comparator_runs'] == 16


def test_the_assembler_cannot_write_onto_an_input_or_outside_its_directory():
    """Review of 2026-10-12: with a path in --prefix, an --out-dir that holds
    the inputs, or a --record that names a pickle, the first version
    overwrote its own inputs and still printed PASS."""
    n = 0
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        before = {p: _sha256(p) for p in comps + srcs}
        ind = os.path.dirname(comps[0])

        def untouched():
            return {p: _sha256(p) for p in comps + srcs} == before

        for prefix in ('../in/', os.path.join(d, 'zz_'), 'a/b', '..', '.hidden_', 'e\\'):
            r, rec, out, left = _assemble(os.path.join(d, 'p%d' % n), comps, srcs, prefix=prefix)
            assert r.returncode == 1 and 'must be a plain file-name prefix' in r.stdout, (prefix, r.stdout)
            assert untouched() and left == [] and rec['status'] == 'REFUSED'; n += 1
        # the output directory is the input directory, or holds an input
        r, rec, out, left = _assemble(os.path.join(d, 'o1'), comps, srcs, out=ind)
        assert r.returncode == 1 and 'is not an empty directory' in r.stdout and untouched(); n += 1
        r, rec, out, left = _assemble(os.path.join(d, 'o2'), comps, srcs, out=d)
        assert r.returncode == 1 and untouched(); n += 1
        # an output directory that already holds something is never merged into
        stale = os.path.join(d, 'o3', 'out')
        os.makedirs(stale)
        open(os.path.join(stale, 'e_stale_old.pkl'), 'wb').write(b'old')
        r, rec, out, left = _assemble(os.path.join(d, 'o3'), comps, srcs)
        assert r.returncode == 1 and 'is not an empty directory' in r.stdout and left == ['e_stale_old.pkl']; n += 1
        # ... an EMPTY one is accepted
        os.makedirs(os.path.join(d, 'o4', 'out'))
        r, rec, out, left = _assemble(os.path.join(d, 'o4'), comps, srcs)
        assert r.returncode == 0 and len(left) == 2 and untouched(); n += 1
        # the record names an input, or a path under the output directory
        r, rec, out, left = _assemble(os.path.join(d, 'r1'), comps, srcs, rec=srcs[0])
        assert r.returncode == 1 and 'names an input' in r.stdout and untouched() and left == []; n += 1
        r, rec, out, left = _assemble(os.path.join(d, 'r2'), comps, srcs, rec=comps[1])
        assert r.returncode == 1 and untouched() and left == []; n += 1
        r, rec, out, left = _assemble(os.path.join(d, 'r3'), comps, srcs,
                                      rec=os.path.join(d, 'r3', 'out', 'record.json'))
        assert r.returncode == 1 and 'under --out-dir' in r.stdout and untouched()
        assert not os.path.exists(os.path.join(d, 'r3', 'out')); n += 1
        # two comparator pickles with one file name would be one derived file
        twin = _pk(os.path.join(d, 'other', 's6_b0.pkl'), ARMS8, [3, 4], CK_A)
        r, rec, out, left = _assemble(os.path.join(d, 't1'), [comps[0], twin], srcs)
        assert r.returncode == 1 and 'the same file name' in r.stdout and left == []; n += 1
        # an empty element among the inputs is a usage error, not a silent drop
        r, rec, out, left = _assemble(os.path.join(d, 'u1'), [comps[0], '', comps[1]], srcs)
        assert r.returncode == 2 and left == []; n += 1
        assert n == 15 and untouched() and not _private_left(d)


def test_the_assembler_refuses_seed_keys_and_offered_series_that_identify_nothing():
    def two_keys(P):
        P['recovery_multi']['marl_static']['1'] = _run('marl_static', 'recovery_multi', 1, 555.0)

    def str_float_key(P):
        P['recovery_multi']['marl_static']['1.0'] = P['recovery_multi']['marl_static'].pop(1)

    def float_key(P):
        P['recovery_multi']['marl_static'][1.5] = _run('marl_static', 'recovery_multi', 1, 5.0)

    def offered(value, both=True):
        def f(P):
            P['rescue_multi']['marl_static'][3]['offered'] = value
        f.both = both
        f.value = value
        return f
    import numpy as np
    nan = float('nan')
    cases = [('two keys of one pickle denote one seed', two_keys, 'two keys denote seed 1'),
             ('a seed key that is a decimal string', str_float_key, 'is not an integer'),
             ('a seed key that is not an integer', float_key, 'is not an integer')]
    # an offered series that carries no information identifies no instance,
    # even when the two passes agree on it
    for name, value, why in (
            ('missing', None, 'is missing'),
            ('all NaN', [nan, nan, nan], 'non-finite'),
            ('constant', [0.0, 0.0, 0.0], 'is constant'),
            ('a string', 'abc', 'not a numeric series'),
            ('a table', {'a': 1}, 'not a numeric series'),
            ('booleans', [True, False, True], 'not a numeric series'),
            ('two-dimensional', np.array([[1.0, 2.0, 3.0]]), 'not one-dimensional'),
            ('not the pass length', [1.0, 2.0, 3.0, 4.0], 'the pass length is 3'),
            ('empty', [], 'not one-dimensional or is empty')):
        cases.append(('offered ' + name, offered(value), why))
    n = 0
    for name, change, why in cases:
        with _scratch('r3_fam_') as d:
            comps, srcs = _inputs(d, src_runs=change)
            if getattr(change, 'both', False):
                # the reference pass carries the same degenerate series: agreement is not identity
                P = pickle.load(open(comps[1], 'rb'))
                P['rescue_multi']['xldet'][3]['offered'] = change.value
                pickle.dump(P, open(comps[1], 'wb'))
            r, rec, out, left = _assemble(d, comps, srcs)
            assert r.returncode == 1 and why in r.stdout, (name, r.stdout)
            assert 'Traceback' not in r.stdout + r.stderr, (name, r.stderr)
            assert left == [] and rec['status'] == 'REFUSED' and not _private_left(d), name
            n += 1
    assert n == 12
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        # -0.0 against 0.0, and a float32 array against the same values as a list of floats
        P = pickle.load(open(srcs[0], 'rb'))
        P['recovery_multi']['marl_static'][2]['offered'] = np.array([2.0, 2.0, 7.5], dtype='float32')
        pickle.dump(P, open(srcs[0], 'wb'))
        r, rec, out, left = _assemble(os.path.join(d, 'a'), comps, srcs)
        assert r.returncode == 0, 'the same values in another container are the same instance: ' + r.stdout


def test_the_assembler_checks_the_assembled_set_against_the_plans_keys():
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        keys = _keys(ARMS8, [1, 2, 3, 4])
        r, rec, out, left = _assemble(os.path.join(d, 'k0'), comps, srcs, keys=keys)
        assert r.returncode == 0 and 'expected_keys=24' in r.stdout, r.stdout
        assert rec['expect_keys']['n'] == 24 and rec['runs_per_arm'] == {'marl_static': 8, 'sdn': 8, 'xldet': 8}
        # without --expect-keys the line and the record SAY that completeness was not checked
        r, rec, out, left = _assemble(os.path.join(d, 'k1'), comps, srcs)
        assert r.returncode == 0 and 'expected_keys=unchecked' in r.stdout and rec['expect_keys'] is None
        n = 0
        # a comparator pass that is empty, an arm absent from one scenario, a key too many, one too few
        for change, why in (
                (lambda P: P['recovery_multi']['sdn'].__setitem__(1, {}), 'lacks 1 expected key(s), first sdn|recovery|1'),
                (lambda P: P['recovery_multi']['sdn'].__setitem__(1, None), 'lacks 1 expected key(s)'),
                (lambda P: P['rescue_multi'].pop('sdn'), 'lacks 2 expected key(s)')):
            dd = os.path.join(d, 'c%d' % n)
            c0 = _pk(os.path.join(dd, 'in', 's6_b0.pkl'), ARMS8, [1, 2], CK_A, runs=change)
            r, rec, out, left = _assemble(dd, [c0, comps[1]], srcs, keys=keys)
            assert r.returncode == 1 and why in r.stdout and left == [], r.stdout
            # NEGATIVE CONTROL of the option: the same input passes when the keys are not given
            r2, rec2, out2, left2 = _assemble(os.path.join(dd, 'nokeys'), [c0, comps[1]], srcs)
            assert r2.returncode == 0 and len(left2) == 2, r2.stdout
            n += 1
        r, rec, out, left = _assemble(os.path.join(d, 'k2'), comps, srcs, keys=keys[:-1])
        assert r.returncode == 1 and 'holds 1 key(s) the plan does not expect' in r.stdout and left == []; n += 1
        r, rec, out, left = _assemble(os.path.join(d, 'k3'), comps, srcs, keys=keys + ['ospf|recovery|1'])
        assert r.returncode == 1 and 'lacks 1 expected key(s), first ospf|recovery|1' in r.stdout; n += 1
        for bad in (['xldet|recovery'], ['xldet|nowhere|1'], keys + [keys[0]], []):
            r, rec, out, left = _assemble(os.path.join(d, 'k4_%d' % n), comps, srcs, keys=bad)
            assert r.returncode == 1 and left == [] and 'Traceback' not in r.stderr, (bad, r.stdout); n += 1
        # a comparator pickle that holds no seed
        empty = _pk(os.path.join(d, 'in', 's6_empty.pkl'), ARMS8, [], CK_A)
        r, rec, out, left = _assemble(os.path.join(d, 'k5'), comps + [empty], srcs)
        assert r.returncode == 1 and 'holds no comparator pass' in r.stdout and left == []; n += 1
        assert n == 10


def test_a_run_of_the_assembler_that_fails_never_leaves_an_older_record():
    with _scratch('r3_fam_') as d:
        comps, srcs = _inputs(d)
        rec_path = os.path.join(d, 'x', 'record.json')
        r, rec, out, left = _assemble(os.path.join(d, 'x'), comps, srcs, rec=rec_path)
        assert r.returncode == 0 and rec['status'] == 'PASS'
        n = 0
        for change in (lambda P: P.__setitem__('provenance', ['not', 'a', 'record']),
                       lambda P: P['provenance'].__setitem__('marl_env', 'a string'),
                       lambda P: P.__setitem__('recovery_multi', ['not a table']),
                       lambda P: P['provenance'].__setitem__('checkpoint_sha256', 12345)):
            bad = _pk(os.path.join(d, 'bad%d' % n, 'ens.pkl'), ['marl_static'], [1, 2, 3, 4], CK_E,
                      tag=100.0, runs=None)
            P = pickle.load(open(bad, 'rb'))
            change(P)
            pickle.dump(P, open(bad, 'wb'))
            r, rec, out, left = _assemble(os.path.join(d, 'y%d' % n), comps, [bad], rec=rec_path)
            assert r.returncode == 1 and 'ASSEMBLE REFUSED' in r.stdout, r.stdout + r.stderr
            assert 'Traceback' not in r.stdout + r.stderr, r.stderr
            assert rec is not None and rec['status'] == 'REFUSED' and rec['failures'] and not rec['outputs']
            assert left == []
            n += 1
        assert n == 4
        # a single checkpoint recorded as a bare string is the list of that one string
        one = _pk(os.path.join(d, 'one', 's6_b0.pkl'), ARMS8, [1, 2], CK_A)
        P = pickle.load(open(one, 'rb'))
        P['provenance']['checkpoint_sha256'] = CK_A[0]
        pickle.dump(P, open(one, 'wb'))
        other = _pk(os.path.join(d, 'one', 's6_b1.pkl'), ARMS8, [3, 4], CK_A)
        r, rec, out, left = _assemble(os.path.join(d, 'z'), [one, other], [one, other], expect=CK_A, prefix='c_')
        assert r.returncode == 0 and len(left) == 2, r.stdout


# ===================================================== chain S7e: step family_e
def _fake_py(tmp, log):
    """$PY of the step: the plan, the inline checks, the ASSEMBLER and the
    LEDGER queries of tools/stage_report.py run for real; the gate tools,
    the analysis and the report comparison record how they were called and
    answer per FAKE_* environment variables.  FAKE_XPROV=N / FAKE_KEYUNION=N
    fail the Nth call of that gate; FAKE_CMP_FAIL_CALL=N fails the Nth
    comparison; FAKE_CMP_LEAVES=a[,b] is the leaf count the comparisons
    report (first, then second)."""
    path = os.path.join(tmp, 'fakepy')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('#!%s\n' % sys.executable)
        fh.write('''import json, os, sys
LOG = %r
REAL_SR = %r
a = sys.argv[1:]
tool = os.path.basename(a[0]) if a else ''
if a and (a[0] == '-' or tool == 'chain_plan.py'):
    os.execv(sys.executable, [sys.executable] + a)
with open(LOG, 'a') as fh:
    fh.write(json.dumps(a) + '\\n')
def arg(k, d=None):
    return a[a.index(k) + 1] if k in a else d
def nth(tool_, flag=None):
    """How many calls of this kind the log holds, this one included."""
    return len([1 for c in map(json.loads, open(LOG)) if c and os.path.basename(c[0]) == tool_
                and (flag is None or flag in c)])
if tool == 'assemble_family.py':
    os.execv(sys.executable, [sys.executable] + a)
if tool == 'analysis_manifest.py':
    print('derived_manifest.json')
elif tool == 'stage_report.py':
    if [k for k in ('--ledger-order', '--ledger-append', '--terminal-failure', '--ledger-artifacts')
            if k in a]:
        os.execv(sys.executable, [sys.executable, REAL_SR] + a[1:])
    if '--verify' in a and os.environ.get('FAKE_VERIFY') == 'fail':
        print('[VERIFY] FAIL')
        sys.exit(1)
    for flag, var in (('--cross-provenance', 'FAKE_XPROV'), ('--key-union', 'FAKE_KEYUNION')):
        if flag in a and os.environ.get(var) == str(nth(tool, flag)):
            print('[GATE] FAIL %%s' %% flag)
            sys.exit(1)
elif tool == 'confirmatory_analysis.py':
    out = arg('--out')
    # the analysis prints its tables to STDOUT -- and can still fail after printing them
    print('TEXT_OF_%%s P1 +1.2345 Holm 0.0123' %% ('FAMILY_E' if 'family_e' in out else 'CONTROL'))
    if os.environ.get('FAKE_CA_FAIL') and os.environ['FAKE_CA_FAIL'] in out:
        sys.stderr.write('Traceback (most recent call last): the analysis died\\n')
        sys.exit(3)
    st = {k: {'status': 'not assessable', 'confirmed': False, 'scenario': 'A', 'metric': 'M7',
              'expect': 1, 'n': 2, 'n_nonzero': 0} for k in ('S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f')}
    rep = {'primary': {'P1': {}, 'P4': {}}, 'seeds': {'A': [52, 53], 'B': [52, 53]},
           'secondary': {}, 'stability': st}
    if os.environ.get('FAKE_CA_E') == 'no_stability' and 'family_e' in out:
        rep.pop('stability')
    if os.environ.get('FAKE_CA_E') == 'other_seeds' and 'family_e' in out:
        rep['seeds'] = {'A': [52], 'B': [52]}
    json.dump(rep, open(out, 'w'))
elif tool == 'compare_reports.py':
    n = nth(tool)
    lv = os.environ.get('FAKE_CMP_LEAVES', '10').split(',')
    lv = lv[min(n, len(lv)) - 1]
    if os.environ.get('FAKE_CMP') == 'different' or os.environ.get('FAKE_CMP_FAIL_CALL') == str(n):
        print('SUMMARY leaves=%%s mismatches=1 -> DIFFERENT' %% lv)
        sys.exit(1)
    print('SUMMARY leaves=%%s mismatches=0 -> IDENTICAL' %% lv)
sys.exit(0)
''' % (log, os.path.join(ROOT, 'tools', 'stage_report.py')))
    os.chmod(path, 0o755)
    return path


def _write_ledger(path, invs, prim_path, mode='ok'):
    """A chain ledger as run_final.sh leaves it for S7e.  `mode`:
      ok        every S9x invocation ended, THEN S7 started and succeeded;
      reversed  S7 was analysed first, the S9x passes were run afterwards;
      one_late  one S9x invocation ended only after S7 had started;
      no_s9x    S9x was never recorded;   none  an empty ledger;
      othersha  the ledger holds another sha256 for the family-1 report;
      s7_failed S7's only recorded end is a failure."""
    L = []

    def run(inv, sha, status='ok'):
        L.append({'utc': 'T%04d' % len(L), 'event': 'start', 'inv': inv, 'stage': invs[inv]['stage'],
                  'attempt': 1})
        L.append({'utc': 'T%04d' % len(L), 'event': 'end', 'inv': inv, 'stage': invs[inv]['stage'],
                  'attempt': 1, 'status': status, 'exit': 0 if status == 'ok' else 7, 'gate': 0,
                  'sha256': sha, 'artifact': invs[inv]['artifact'], 'watchdog_restarts': 0})

    def sha_of(inv):
        p = os.path.join(os.path.dirname(os.path.dirname(path)), invs[inv]['artifact'])
        return _sha256(p) if os.path.exists(p) else '-'
    s9 = [i for i in invs if invs[i]['stage'] == 'S9x']
    psha = _sha256(prim_path) if os.path.exists(prim_path) else '-'
    assert len(s9) == 2
    if mode == 'reversed':
        run('s7_analysis', psha)
        for i in s9:
            run(i, sha_of(i))
    elif mode == 'one_late':
        run(s9[0], sha_of(s9[0]))
        run('s7_analysis', psha)
        run(s9[1], sha_of(s9[1]))
    elif mode == 'no_s9x':
        run('s7_analysis', psha)
    elif mode in ('ok', 'othersha', 's7_failed'):
        for i in s9:
            run(i, sha_of(i))
        run('s7_analysis', 'f' * 64 if mode == 'othersha' else psha,
            status='fail' if mode == 's7_failed' else 'ok')
    else:
        assert mode == 'none', mode
    with open(path, 'w') as f:
        for r in L:
            f.write(json.dumps(r, sort_keys=True) + '\n')


def _stage_pickle(path, inv, tag=0.0, ckpt=None):
    arms = sorted({k.split('|')[0] for k in inv['expected_keys']})
    return _pk(path, arms, sorted(inv['seeds']), ckpt if ckpt is not None else inv['ckpt_sha256'], tag=tag,
               prov=_prov(ckpt if ckpt is not None else inv['ckpt_sha256'],
                          marl_env={'MARL_CHAIN_INV': inv['id'], 'MARL_POSTCARD_ALWAYS': '1'}))


# `rm` on a filesystem that does not let go at once: the first @BUSY@ removals of
# an assembled set's directory empty it, leave a placeholder in it and end as
# the real one does on the run tree's network filesystem while another process
# still holds a file of the directory open.  Every removal of an EXISTING
# family directory is counted.
_BUSY_RM = '''#!/bin/bash
n=$(cat "@COUNT@" 2>/dev/null || echo 0)
busy=0
for a in "$@"; do
  case "$a" in
    */family_ctl|*/family_e)
      if [ -d "$a" ]; then
        n=$((n + 1)); echo "$n" > "@COUNT@"
        if [ "$n" -le @BUSY@ ]; then
          "@RM@" -rf "$a"/* 2>/dev/null
          : > "$a/.nfs0000000000000001"
          echo "rm: cannot remove '$a': Directory not empty" >&2
          busy=1
        fi
      fi ;;
  esac
done
[ "$busy" = 1 ] && exit 1
exec "@RM@" "$@"
'''


def _run_family_e(env=None, prim=True, skip=(), ens_ckpt=None, analysis=None, ledger='ok', plant=False,
                  noscipy=False, busy_rm=0):
    """confirmatory/stage_analysis.sh --run r3lite_rehearsal s7e_family_e in a
    scratch tree holding the rehearsal's synthetic pickles, (prim) the
    family-1 report and a chain ledger (`ledger`: a mode of _write_ledger).
    plant=True leaves the outputs of an EARLIER run of the step in place
    before it starts: a report, its text, a record and both assembled
    directories.  busy_rm=N runs the step with the `rm` of _BUSY_RM first on
    its PATH; out['rm_busy'] is then the number of removals of an existing
    family directory."""
    tmp = _mkdtemp('r3_s7e_')
    try:
        os.makedirs(os.path.join(tmp, 'confirmatory'))
        os.makedirs(os.path.join(tmp, 'tools'))
        for f in ('stage_analysis.sh', 'chain_plan.py'):
            shutil.copy(os.path.join(ROOT, 'confirmatory', f), os.path.join(tmp, 'confirmatory', f))
        shutil.copy(TOOL, os.path.join(tmp, 'tools', 'assemble_family.py'))
        data = open(os.path.join(ROOT, 'confirmatory_analysis.py'), 'rb').read()
        with open(os.path.join(tmp, 'confirmatory_analysis.py'), 'wb') as f:
            f.write(data if analysis is None else analysis(data))
        invs = {i['id']: i for i in CP.build(REH)}
        for i in invs.values():
            if not i['artifact'].endswith('.pkl') or i['id'] in skip:
                continue
            ens = i['stage'] == 'S9x'
            _stage_pickle(os.path.join(tmp, i['artifact']), i, tag=100.0 if ens else 0.0,
                          ckpt=ens_ckpt if (ens and ens_ckpt is not None) else None)
        prim_path = os.path.join(tmp, invs['s7_analysis']['artifact'])
        if prim:
            json.dump({'primary': {'P1': {}, 'P4': {}}, 'seeds': {'A': [52, 53], 'B': [52, 53]},
                       'stability': {}}, open(prim_path, 'w'))
        ledger_path = os.path.join(tmp, 'confirmatory', 'chain_%s.jsonl' % REH)
        _write_ledger(ledger_path, invs, prim_path, ledger)
        base = os.path.join(tmp, 'confirmatory', 'r3lite_rehearsal_')
        fam = os.path.join(tmp, 'output', 'r3lite_rehearsal', 'family_e')
        ctl = os.path.join(tmp, 'output', 'r3lite_rehearsal', 'family_ctl')
        owned = [base + 'cf_report_family_e.json', base + 'cf_report_family_e.txt',
                 base + 'family_e_record.json', fam, ctl]
        if plant:
            for p in owned[:3]:
                with open(p, 'w') as f:
                    f.write('{"stale": true}')
            for dd in (fam, ctl):
                os.makedirs(dd)
                with open(os.path.join(dd, 'stale.pkl'), 'wb') as f:
                    f.write(b'stale')
        log = os.path.join(tmp, 'calls.log')
        e = dict(os.environ, PY=_fake_py(tmp, log))
        if noscipy:
            # an interpreter whose `import scipy` fails: a package of that name, first on the path
            os.makedirs(os.path.join(tmp, 'noscipy', 'scipy'))
            with open(os.path.join(tmp, 'noscipy', 'scipy', '__init__.py'), 'w') as f:
                f.write("raise ImportError('scipy is disabled for this test')\n")
            e['PYTHONPATH'] = os.path.join(tmp, 'noscipy')
        e.update(env or {})
        rm_count = os.path.join(tmp, 'rm_busy.count')
        if busy_rm:
            os.makedirs(os.path.join(tmp, 'fakebin'))
            fake_rm = os.path.join(tmp, 'fakebin', 'rm')
            with open(fake_rm, 'w') as f:
                f.write(_BUSY_RM.replace('@COUNT@', rm_count).replace('@BUSY@', str(int(busy_rm)))
                        .replace('@RM@', shutil.which('rm')))
            os.chmod(fake_rm, 0o755)
            e['PATH'] = os.path.join(tmp, 'fakebin') + os.pathsep + e['PATH']
        r = subprocess.run(['bash', os.path.join(tmp, 'confirmatory', 'stage_analysis.sh'),
                            '--run', REH, 's7e_family_e'], cwd=tmp, env=e,
                           capture_output=True, text=True)
        calls = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
        out = {'invs': invs, 'prim_sha': _sha256(prim_path) if prim else None, 'files': {},
               'ledger': [json.loads(l) for l in open(ledger_path) if l.strip()],
               'rm_busy': int(open(rm_count).read()) if os.path.exists(rm_count) else 0,
               # what the step owns and a failed run must not leave behind
               'left': [os.path.relpath(p, tmp) for p in owned if os.path.exists(p)]}
        for name in ('cf_report_family_e.json', 'family_e_record.json', 'family_e_assembly.json',
                     'family_ctl_assembly.json', 'cf_report_family_ctl.json'):
            p = base + name
            out['files'][name] = json.load(open(p)) if os.path.exists(p) else None
        out['family_e_dir'] = sorted(os.listdir(fam)) if os.path.isdir(fam) else None
        out['family_ctl_dir_exists'] = os.path.isdir(ctl)
        fam_pkls = [x for x in out['family_e_dir'] or [] if x.endswith('.pkl') and x != 'stale.pkl']
        if fam_pkls:
            D = pickle.load(open(os.path.join(fam, fam_pkls[0]), 'rb'))
            S6 = pickle.load(open(os.path.join(tmp, invs['s6_cffull_b0']['artifact']), 'rb'))
            EN = pickle.load(open(os.path.join(tmp, invs['s9x_ensemblefull']['artifact']), 'rb'))
            out['derived_ok'] = all(
                _same(D[sk][a][s], (EN if a == 'marl_static' else S6)[sk][a][s])
                for sk in SCEN for a in D[sk] for s in D[sk][a])
            out['derived_arms'] = sorted(D['recovery_multi'])
            out['derived_ckpt'] = D['provenance']['checkpoint_sha256']
        return r, calls, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tool_calls(calls, tool):
    return [c for c in calls if c and os.path.basename(c[0]) == tool]


def _a(call, flag):
    return call[call.index(flag) + 1]


def test_s7e_computes_family_e_only_after_family_1_and_only_on_a_proved_assembly_path():
    _linux_or_fail('the stage script needs bash and a POSIX shebang')
    r, calls, out = _run_family_e()
    assert r.returncode == 0, r.stdout[-5000:] + r.stderr[-2000:]
    invs = out['invs']
    s6 = invs['s6_cffull_b0']['artifact']
    ens = invs['s9x_ensemblefull']['artifact']
    sel = invs['s6_cffull_b0']['ckpt_sha256']
    pins = invs['s9x_ensemblefull']['ckpt_sha256']
    assert len(sel) == 1 and len(pins) == 4 and sel != pins
    # 1. the family-1 report is recorded BEFORE anything of family E
    lines = r.stdout.splitlines()
    i_rec = [i for i, l in enumerate(lines) if 'recorded before family E is computed' in l]
    i_ctl = [i for i, l in enumerate(lines) if 'assembly control: the set assembled from S6 alone reproduces' in l]
    i_fam = [i for i, l in enumerate(lines) if 'confirmatory_analysis.py --marl-arm marl_static (family E' in l]
    assert len(i_rec) == len(i_ctl) == len(i_fam) == 1 and i_rec[0] < i_ctl[0] < i_fam[0], r.stdout[-3000:]
    assert out['prim_sha'] in lines[i_rec[0]]
    # 2. the tools, in order: control assembled and analysed and compared, THEN family E
    seq = [os.path.basename(c[0]) for c in calls
           if os.path.basename(c[0]) in ('assemble_family.py', 'confirmatory_analysis.py', 'compare_reports.py')]
    assert seq == ['assemble_family.py', 'confirmatory_analysis.py', 'compare_reports.py',
                   'compare_reports.py', 'assemble_family.py', 'confirmatory_analysis.py'], seq
    asm = _tool_calls(calls, 'assemble_family.py')
    ctl, fam = asm
    # both assemblies are checked against the plan's key set of the analysis input
    for c in (ctl, fam):
        assert _a(c, '--expect-keys') == 'logs/chain_r3lite_rehearsal/S7e.keys'
    assert out['files']['family_e_assembly.json']['expect_keys']['n'] == 32
    assert out['files']['family_ctl_assembly.json']['expect_keys']['n'] == 32
    # the control: the arm comes from the S6 pickles themselves, the selected replicate's pin
    assert _a(ctl, '--comparators') == _a(ctl, '--arm-source') == s6
    assert _a(ctl, '--expect-checkpoint-sha256').split(',') == sel and _a(ctl, '--prefix') == 'c_'
    # family E: the SAME comparators, the ensemble's pickle, the four pins in order
    assert _a(fam, '--comparators') == s6 and _a(fam, '--arm-source') == ens
    assert _a(fam, '--expect-checkpoint-sha256').split(',') == pins and _a(fam, '--prefix') == 'e_'
    assert _a(ctl, '--arm') == _a(fam, '--arm') == 'marl_static'
    assert _a(ctl, '--out-dir') != _a(fam, '--out-dir')
    # the analysis: the unchanged script, strict, marl_static, on the DERIVED sets only
    ana = _tool_calls(calls, 'confirmatory_analysis.py')
    for c, pre, d in ((ana[0], 'c_', 'family_ctl'), (ana[1], 'e_', 'family_e')):
        assert c[0] == 'confirmatory_analysis.py' and '--no-assert' not in c
        assert _a(c, '--marl-arm') == 'marl_static' and '--untrained' not in c and '--s8' not in c
        assert _a(c, '--pickles') == 'output/r3lite_rehearsal/%s/%s%s' % (d, pre, os.path.basename(s6))
        assert _a(c, '--manifests') == 'derived_manifest.json'
    assert _a(ana[1], '--out') == invs['s7e_family_e']['artifact']
    assert _a(ana[0], '--out') != invs['s7_analysis']['artifact'], 'the family-1 report is never rewritten'
    # the control is compared with the family-1 report in BOTH directions
    cmp_ = _tool_calls(calls, 'compare_reports.py')
    assert cmp_[0][1:3] == [invs['s7_analysis']['artifact'], _a(ana[0], '--out')]
    assert cmp_[1][1:3] == [_a(ana[0], '--out'), invs['s7_analysis']['artifact']]
    assert 'stability' in _a(cmp_[0], '--sections') and 'primary' in _a(cmp_[0], '--sections')
    # the gates: the ensemble pickle against its own key set and pins; provenance; key unions
    sr = _tool_calls(calls, 'stage_report.py')
    ver = [c for c in sr if '--verify' in c]
    assert len(ver) == 1 and _a(ver[0], '--verify') == ens and _a(ver[0], '--expect-hashseed') == '0'
    assert _a(ver[0], '--expect-checkpoint-sha256').split(',') == pins
    assert len([c for c in sr if '--key-union' in c]) == 3 and len([c for c in sr if '--cross-provenance' in c]) == 2
    # 3. what is left: the family-E set (really assembled), no control set, the record
    assert out['family_e_dir'] == ['e_' + os.path.basename(s6)] and not out['family_ctl_dir_exists']
    assert out['derived_ok'] and out['derived_ckpt'] == pins
    assert out['derived_arms'] == sorted(CP.CONFIRMATORY_ARMS)
    rec = out['files']['family_e_record.json']
    assert rec['family_1_report_sha256'] == out['prim_sha']
    # the ORDER is read from the ledger and recorded with where it was read: every S9x
    # invocation ended before the successful attempt of S7 started
    lo = rec['ledger_order']
    assert lo['family_e_passes_complete_before_family_1_analysis'] is True
    assert sorted(f['inv'] for f in lo['family_e_passes']) == ['s9x_ensemble', 's9x_ensemblefull']
    assert lo['family_1_analysis']['inv'] == 's7_analysis' and lo['family_1_analysis']['sha256'] == out['prim_sha']
    assert all(f['end_record'] < lo['family_1_analysis']['start_record'] for f in lo['family_e_passes'])
    assert 'family_1_report_recorded_before_family_e' not in rec, 'no self-attested boolean'
    led = [c for c in sr if '--ledger-order' in c]
    assert len(led) == 1 and _a(led[0], '--then') == 's7_analysis' \
        and _a(led[0], '--expect-then-sha256') == out['prim_sha'] \
        and sorted(_a(led[0], '--first').split(',')) == ['s9x_ensemble', 's9x_ensemblefull']
    assert calls.index(led[0]) < calls.index(asm[0]), 'the order is established before anything is assembled'
    # ... and the family-1 digest is IN the ledger, written by the step before family E exists
    notes = [x for x in out['ledger'] if x.get('reason') == 'family_1_report_recorded']
    assert len(notes) == 1 and notes[0]['report_sha256'] == out['prim_sha'] \
        and notes[0]['inv'] == 's7e_family_e' and notes[0]['event'] == 'note', notes
    app = [c for c in sr if '--ledger-append' in c]
    assert len(app) == 1 and calls.index(led[0]) < calls.index(app[0]) < calls.index(asm[0])
    assert rec['assembly_control']['leaves_compared_each_direction'] == 10
    assert rec['assembly_control']['reproduces_family_1_report'] == 'identical'
    # the TEXT of the family-E analysis is printed once, and only after the record is written
    # (every check of the step has passed by then)
    i_rec2 = [i for i, l in enumerate(lines) if '[stage_analysis] family E record ->' in l]
    i_txt = [i for i, l in enumerate(lines) if 'TEXT_OF_FAMILY_E' in l]
    assert len(i_rec2) == 1 and len(i_txt) == 1 and i_rec2[0] < i_txt[0], r.stdout[-2000:]
    assert rec['family_e_assembly']['expect_checkpoint_sha256'] == pins
    assert rec['family_e_assembly']['verify'] == {'comparator_runs': 28, 'arm_runs': 4,
                                                  'offered_pairs': 4, 'derived_pickles': 1}
    assert rec['family_e_assembly']['tool_sha256'] == _sha256(TOOL)
    assert 'nothing is pooled' in rec['note']
    assert out['files']['family_e_assembly.json']['status'] == 'PASS'
    assert out['files']['family_ctl_assembly.json']['expect_checkpoint_sha256'] == sel


def test_s7e_stops_before_family_e_whenever_a_precondition_fails():
    _linux_or_fail('the stage script needs bash and a POSIX shebang')
    n = 0

    def none_of_family_e(calls, out):
        asm = _tool_calls(calls, 'assemble_family.py')
        assert not [c for c in asm if _a(c, '--prefix') == 'e_'], 'family E was assembled'
        assert out['files']['cf_report_family_e.json'] is None and out['files']['family_e_record.json'] is None
        assert not out['family_e_dir']
        assert out['left'] == [], out['left']
        assert 'TEXT_OF_FAMILY_E' not in r.stdout + r.stderr, 'family-E text in the log of a failed step'

    # no family-1 report: nothing is assembled at all
    r, calls, out = _run_family_e(prim=False)
    assert r.returncode != 0 and 'family E is never computed before it' in r.stdout, r.stdout[-2000:]
    assert not _tool_calls(calls, 'assemble_family.py') and not _tool_calls(calls, 'confirmatory_analysis.py')
    none_of_family_e(calls, out); n += 1
    # the assembly path does not reproduce the family-1 report: family E is not computed
    r, calls, out = _run_family_e(env={'FAKE_CMP': 'different'})
    assert r.returncode != 0 and 'does not reproduce the family-1 report leaf by leaf' in r.stdout
    assert len(_tool_calls(calls, 'confirmatory_analysis.py')) == 1
    none_of_family_e(calls, out); n += 1
    # the analysis fails on the control set
    r, calls, out = _run_family_e(env={'FAKE_CA_FAIL': 'family_ctl'})
    assert r.returncode != 0 and 'failed on the control set' in r.stdout
    none_of_family_e(calls, out); n += 1
    # the ensemble pickle is missing (chain S9x is never shed)
    r, calls, out = _run_family_e(skip=('s9x_ensemblefull',))
    assert r.returncode != 0 and 'is missing (chain S9x is never shed)' in r.stdout
    assert not _tool_calls(calls, 'assemble_family.py'); none_of_family_e(calls, out); n += 1
    # the ensemble pickle fails its own completeness gate
    r, calls, out = _run_family_e(env={'FAKE_VERIFY': 'fail'})
    assert r.returncode != 0 and not _tool_calls(calls, 'assemble_family.py')
    none_of_family_e(calls, out); n += 1
    # a comparator pickle is missing
    r, calls, out = _run_family_e(skip=('s6_cffull_b0',))
    assert r.returncode != 0 and ('no confirmatory pickle exists' in r.stdout
                                  or 'produced no pickle' in r.stdout), r.stdout[-1500:]
    none_of_family_e(calls, out); n += 1
    # the ensemble pickle does not record the four pinned checkpoints: the REAL assembler refuses
    inv = CP.by_id(REH, 's9x_ensemblefull')
    for bad in (inv['ckpt_sha256'][::-1], inv['ckpt_sha256'][:1], inv['ckpt_sha256'][:3] + ['e' * 64]):
        r, calls, out = _run_family_e(ens_ckpt=bad)
        assert r.returncode != 0 and 'the family-E set could not be assembled' in r.stdout, r.stdout[-2000:]
        assert 'ASSEMBLE REFUSED' in r.stdout and 'records checkpoint sha256' in r.stdout
        assert len(_tool_calls(calls, 'confirmatory_analysis.py')) == 1, 'only the control was analysed'
        assert out['files']['cf_report_family_e.json'] is None and not out['family_e_dir']
        assert out['files']['family_e_assembly.json']['status'] == 'REFUSED'
        n += 1
    # the analysis fails on family E / leaves no stability family / other instances
    # NOTHING OF THE FAMILY-E TEXT reaches the log of a step that failed: the analysis has
    # printed its tables (the fake prints a marker line) and then died, or passed and one of
    # the step's own checks failed.  A failed analysis shows its STDERR only.
    r, calls, out = _run_family_e(env={'FAKE_CA_FAIL': 'family_e'})
    assert r.returncode != 0 and 'exit=3 for family E' in r.stdout
    assert 'the analysis died' in r.stdout and 'TEXT_OF_FAMILY_E' not in r.stdout + r.stderr, r.stdout[-1500:]
    assert out['files']['family_e_record.json'] is None and out['left'] == []; n += 1
    r, calls, out = _run_family_e(env={'FAKE_CA_E': 'no_stability'})
    assert r.returncode != 0 and 'family E lacks a stability endpoint' in r.stdout
    assert 'TEXT_OF_FAMILY_E' not in r.stdout + r.stderr
    assert out['files']['family_e_record.json'] is None and out['left'] == []; n += 1
    r, calls, out = _run_family_e(env={'FAKE_CA_E': 'other_seeds'})
    assert r.returncode != 0 and 'are not family 1 instances' in r.stdout
    assert 'TEXT_OF_FAMILY_E' not in r.stdout + r.stderr
    assert out['files']['family_e_record.json'] is None and out['left'] == []; n += 1
    # the analysis script is not the pre-registered file: nothing is called
    r, calls, out = _run_family_e(analysis=lambda data: data + b'\n# edited\n')
    assert r.returncode != 0 and 'is not the pre-registered' in r.stdout + r.stderr
    assert not calls; n += 1
    assert n == 13
    # THE ORDER, read from the ledger: family E is refused unless every S9x invocation ended
    # before the successful attempt of S7 started and the ledger holds THIS family-1 report
    for mode, why in (('reversed', 'AFTER s7_analysis was first started'),
                      ('one_late', 'AFTER s7_analysis was first started'),
                      ('no_s9x', 'is not recorded successful'), ('none', 'holds no record'),
                      ('othersha', "the ledger's sha256 of s7_analysis"),
                      ('s7_failed', 's7_analysis is not recorded successful')):
        r, calls, out = _run_family_e(ledger=mode)
        assert r.returncode != 0 and 'the ledger does not show the passes of family E' in r.stdout, \
            (mode, r.stdout[-2000:])
        assert why in r.stdout, (mode, r.stdout[-2000:])
        assert not _tool_calls(calls, 'assemble_family.py') and not _tool_calls(calls, 'confirmatory_analysis.py')
        assert not [x for x in out['ledger'] if x.get('reason') == 'family_1_report_recorded'], mode
        none_of_family_e(calls, out); n += 1
    # THE CONTROL, both directions and by count: the second comparison is as fatal as the
    # first, and a comparison of no leaf, or of unequal counts, proves nothing
    r, calls, out = _run_family_e(env={'FAKE_CMP_FAIL_CALL': '1'})
    assert r.returncode != 0 and 'does not reproduce the family-1 report leaf by leaf' in r.stdout
    none_of_family_e(calls, out); n += 1
    r, calls, out = _run_family_e(env={'FAKE_CMP_FAIL_CALL': '2'})
    assert r.returncode != 0 and 'holds a leaf the control report lacks' in r.stdout, r.stdout[-2000:]
    assert len(_tool_calls(calls, 'compare_reports.py')) == 2
    none_of_family_e(calls, out); n += 1
    for leaves in ('0', '10,9', '0,10'):
        r, calls, out = _run_family_e(env={'FAKE_CMP_LEAVES': leaves})
        assert r.returncode != 0 and 'nothing was proved' in r.stdout, (leaves, r.stdout[-2000:])
        none_of_family_e(calls, out); n += 1
    # THE GATES of the step itself: provenance (inputs, then the assembled set) and the three
    # key unions (inputs, control set, family-E set)
    for var, k, asm_e in (('FAKE_XPROV', '1', 0), ('FAKE_XPROV', '2', 1), ('FAKE_KEYUNION', '1', 0),
                          ('FAKE_KEYUNION', '2', 0), ('FAKE_KEYUNION', '3', 1)):
        r, calls, out = _run_family_e(env={var: k})
        assert r.returncode != 0 and '[GATE] FAIL' in r.stdout, (var, k, r.stdout[-2000:])
        asm = [c for c in _tool_calls(calls, 'assemble_family.py') if _a(c, '--prefix') == 'e_']
        assert len(asm) == asm_e, (var, k)
        assert len([c for c in _tool_calls(calls, 'confirmatory_analysis.py')
                    if 'family_e' in _a(c, '--out')]) == 0, 'family E was analysed after a failed gate'
        assert out['left'] == [] and out['files']['family_e_record.json'] is None, (var, k, out['left'])
        n += 1
    # NOTHING STALE SURVIVES: the outputs of an earlier run are gone whether the step stops
    # at its first gate, in the middle or at its very last check -- and the step says so
    for kw in (dict(prim=False), dict(env={'FAKE_VERIFY': 'fail'}), dict(env={'FAKE_CMP': 'different'}),
               dict(env={'FAKE_CA_FAIL': 'family_e'}), dict(env={'FAKE_CA_E': 'no_stability'}),
               dict(ledger='reversed')):
        r, calls, out = _run_family_e(plant=True, **kw)
        assert r.returncode != 0, kw
        assert out['left'] == [], (kw, out['left'])
        assert out['files']['cf_report_family_e.json'] is None and out['files']['family_e_record.json'] is None
        assert 'the step did NOT complete' in r.stdout, (kw, r.stdout[-1500:])
        n += 1
    assert n == 13 + 6 + 2 + 3 + 5 + 6
    # SCIPY is a precondition of the step, checked with the interpreter that would run the
    # analysis: without it nothing is called at all (NEGATIVE CONTROL of analysis_pin)
    r, calls, out = _run_family_e(noscipy=True)
    assert r.returncode != 0 and 'scipy.stats is not usable' in r.stdout + r.stderr, r.stdout[-1500:] + r.stderr[-800:]
    assert not calls and out['files']['cf_report_family_e.json'] is None
    # ... and a planted earlier run does not disturb a run that passes
    r, calls, out = _run_family_e(plant=True)
    assert r.returncode == 0 and out['family_e_dir'] == ['e_' + os.path.basename(
        CP.by_id(REH, 's6_cffull_b0')['artifact'])], r.stdout[-3000:]
    assert out['files']['cf_report_family_e.json'].get('stale') is None
    assert 'the step did NOT complete' not in r.stdout
    # the step belongs to the R3-lite graph only
    assert not [i for i in CP.build('real') if i.get('step') == 'family_e']
    # and the unchanged input passes (the stops above are not a broken harness)
    r, calls, out = _run_family_e()
    assert r.returncode == 0, r.stdout[-3000:]


def test_s7e_removes_its_assembled_sets_on_a_filesystem_that_lets_go_late():
    """2026-10-12, the suite at the frozen head: a failed run of the step left
    the control set's directory behind.  The run tree is on a network
    filesystem, where a file that another process still holds open (the
    host's on-access scanner does, for a moment after a file was written,
    read or executed) is not removed at once: `rm -rf` ends "Directory not
    empty" and the directory stays for a few seconds.  19 of 200 repeats of
    that failed run left it; on a local filesystem none can.  The step now
    repeats a removal until the paths are gone and says so when they are
    not.  Here its `rm` is one that leaves a placeholder in a family
    directory for its first N calls (_BUSY_RM).  With the single `rm -rf`
    the step had before, the first assertion on `left` fails."""
    _linux_or_fail('the stage script needs bash and a POSIX shebang')
    ctl = os.path.join('output', 'r3lite_rehearsal', 'family_ctl')
    fam = os.path.join('output', 'r3lite_rehearsal', 'family_e')
    # a failed run: the control set is gone when the step ends, because the removal was repeated
    r, calls, out = _run_family_e(env={'FAKE_CMP_FAIL_CALL': '1'}, busy_rm=3)
    assert r.returncode != 0 and 'does not reproduce the family-1 report leaf by leaf' in r.stdout
    assert out['left'] == [], out['left']
    # three refusals, then the removal -- at least: on the run tree's own filesystem the real
    # removal that follows may itself need another call, which is the very thing repeated
    assert out['rm_busy'] >= 4, out['rm_busy']
    assert 'no family-E report, text, record or assembled set is left' in r.stdout
    # a run that passes: no control set beside family E, family E itself complete, no warning
    r, calls, out = _run_family_e(busy_rm=2)
    assert r.returncode == 0, r.stdout[-3000:]
    assert not out['family_ctl_dir_exists'] and out['family_e_dir'] and out['rm_busy'] >= 3
    assert out['files']['family_e_record.json'] is not None and 'WARNING: the control set' not in r.stdout
    # a set that can NOT be removed: the step says so and does not claim that nothing is left;
    # the report and the record are gone all the same
    r, calls, out = _run_family_e(env={'FAKE_CMP_FAIL_CALL': '1'}, busy_rm=10 ** 6)
    assert r.returncode != 0 and out['left'] == [ctl], out['left']
    assert 'AN ASSEMBLED SET COULD NOT BE REMOVED' in r.stdout
    assert 'record or assembled set is left' not in r.stdout
    assert out['files']['cf_report_family_e.json'] is None and out['files']['family_e_record.json'] is None
    assert out['rm_busy'] > 60, out['rm_busy']            # a minute of attempts, not one
    # ... and a start that cannot clear the sets of an earlier run computes nothing beside them,
    # after removing the stale report, its text and the record
    r, calls, out = _run_family_e(plant=True, busy_rm=10 ** 6)
    assert r.returncode != 0 and 'cannot be removed; nothing was computed' in r.stdout, r.stdout[-1500:]
    assert not _tool_calls(calls, 'assemble_family.py') and not _tool_calls(calls, 'confirmatory_analysis.py')
    assert sorted(out['left']) == sorted([ctl, fam]), out['left']
    assert out['files']['cf_report_family_e.json'] is None and out['files']['family_e_record.json'] is None


def test_the_assembler_is_part_of_the_freeze_and_never_edits_the_analysis():
    """The assembler is a new tool (D-R3.3); the analysis file stays the
    pinned one, and the stage script hands the analysis nothing but derived
    pickles of the two prefixes."""
    with open(os.path.join(ROOT, 'confirmatory_analysis.py'), 'rb') as f:
        assert hashlib.md5(f.read()).hexdigest() == CP.R3_ANALYSIS_MD5
    src = open(os.path.join(ROOT, 'confirmatory', 'stage_analysis.sh'), encoding='utf-8').read()
    step = src.split('if [ "$INV_STEP" = family_e ]; then', 1)[1].split('\nif [ "$INV_STEP" = secondary ]', 1)[0]
    call = '"$PY" tools/assemble_family.py'
    assert step.count(call) == 2
    assert step.count('confirmatory_analysis.py --pickles') == 2 and '--no-assert' not in step
    assert step.index('PRIM_SHA=') < step.index('--ledger-order') < step.index('--ledger-append') \
        < step.index(call)
    assert 'trap fame_cleanup EXIT' in step and step.index('trap fame_cleanup EXIT') < step.index('PRIM_SHA=')
    assert "'family_1_report_recorded_before_family_e'" not in step
    assert step.index('--prefix c_') < step.index('"$PY" tools/compare_reports.py') < step.index('--prefix e_')
    assert 'nothing is pooled' in step and 'never pools them' in step
    tool = open(TOOL, encoding='utf-8').read()
    assert 'import confirmatory_analysis' not in tool and 'sixg_sim' not in tool
