# -*- coding: utf-8 -*-
"""R3-lite: the plan, its pins and its stage scripts (docs/PREREGISTRATION_R3.md).

R3-lite is a NEW pre-registered confirmatory test of an EXISTING policy (the
R2 replicate torch seed 1234) on a fresh block of topology seeds.  What can
be checked without compute is checked here, against the objects the launcher
runs (confirmatory/chain_plan.py and the stage scripts):

  * profile r3lite runs exactly the R3 confirmatory block, and no other
    profile names it or any reserved block;
  * its confirmatory stage IS R2's confirmatory stage (arms, block split,
    environment, scripts) -- only the seeds and the checkpoint differ;
  * the stages R2 ran and R3-lite does not are ABSENT AND RECORDED;
  * the checkpoints are sha256-pinned in the plan, in the tracked SELECTED
    files, at the import, before every launch and in every pickle gate;
  * the four R2 profiles are bit-for-bit what they were at 6b53f3f;
  * confirmatory_analysis.py and every engine file are byte-identical to
    R2's;
  * no artefact path of one profile can be written by another.

NO TEST HERE RUNS AN ARM ON ANY SEED, AND NONE WRITES A FILE THAT NAMES A
SEED OF THE R3 CONFIRMATORY BLOCK: every stage-script and launcher test uses
the r3lite_rehearsal profile (two development seeds) in a scratch copy of
the chain; the only computation is that profile's S0 manifest (topology
analysis of the two development seeds, no arm); where the launcher has to
reach an evaluation invocation, the evaluation script is a stub that never
starts the harness; and the seed-audit mechanics are exercised on a
synthetic window.

Vacuous checks are this campaign's recurring defect, so every test counts
what it checked and most carry a negative control: the same check, on a
tampered input, must FAIL.
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
for p in (ROOT, os.path.join(ROOT, 'confirmatory'), os.path.join(ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import chain_plan as CP

R3 = 'r3lite'
REH = 'r3lite_rehearsal'
BLOCK = list(range(300, 340))
OLD_BLOCK = list(range(300, 320))      # the first draft's block (before D-R3.3)
SHA_1234 = '54eaa5623148cfd44faf7e69bfe7825c396541b33d9357a07d782d559de5c4bc'
ANALYSIS_MD5 = 'e6142b2798937b79bafe0314c98dbba0'
R2_SOURCE = os.path.expanduser('~/MARL_run_9/output/final')


# ---------------------------------------------------------------- scratch
def _mkdtemp(prefix):
    """Scratch directory inside the tree (output/tmp_tests; the seed audit
    skips it by design), never the system default (D12F.2)."""
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def _rmtree(d):
    import time as _time
    for _ in range(20):
        shutil.rmtree(d, ignore_errors=True)
        if not os.path.exists(d):
            return
        _time.sleep(0.5)


@contextlib.contextmanager
def _scratch(prefix):
    d = _mkdtemp(prefix)
    try:
        yield d
    finally:
        _rmtree(d)


def _linux_or_fail(what):
    if not sys.platform.startswith('linux') or shutil.which('bash') is None:
        raise AssertionError('CANNOT RUN HERE -- a contract that cannot run '
                             'FAILS, it never skips (tests/test_wp6_parity.py): '
                             + what)


@contextlib.contextmanager
def _patched_profile(name, **changes):
    """PROFILES[name] with fields replaced, restored afterwards."""
    saved = copy.deepcopy(CP.PROFILES[name])
    try:
        CP.PROFILES[name].update(changes)
        yield
    finally:
        CP.PROFILES[name] = saved


def _raises_systemexit(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e) or 'SystemExit'
    return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _plan_cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, 'confirmatory', 'chain_plan.py')]
                          + list(args), capture_output=True, text=True)


# ============================================================ the seed blocks
def test_r3lite_runs_exactly_the_r3_confirmatory_block():
    P = CP.PROFILES[R3]
    assert P['cf_seeds'] == BLOCK, P['cf_seeds']
    assert CP.R3_CONFIRMATORY_SEEDS == BLOCK
    assert P['ticks'] is None and P['nominal_ticks'] == 3600, 'full-length passes'
    assert P['out_prefix'] == 'r3lite/'
    assert P['eval_workers'] == 24
    invs = CP.build(R3)
    s6 = [i for i in invs if i['stage'] == 'S6']
    assert len(s6) == 10, [i['id'] for i in s6]
    assert sorted(s for i in s6 for s in i['seeds']) == BLOCK
    # every seed any invocation of the profile names is a seed of the block
    named = sorted({s for i in invs for s in i['seeds']})
    assert named == BLOCK, named
    for i in invs:
        if i['kind'] == 'eval':
            assert i['ticks'] is None and i['workers'] == 24, i['id']
    # NEGATIVE CONTROLS: the builder itself refuses a block that is not
    # exactly the registered one, and a reduced-length confirmatory profile
    assert len(BLOCK) == 40 and BLOCK[0] == 300 and BLOCK[-1] == 339
    for bad in (BLOCK[:-1], BLOCK + [340], [s + 1 for s in BLOCK], BLOCK[1:],
                OLD_BLOCK, list(range(200, 220))):
        with _patched_profile(R3, cf_seeds=bad):
            assert _raises_systemexit(lambda: CP.build(R3)), bad
    with _patched_profile(R3, cf_full_seeds=[300, 301]):
        assert _raises_systemexit(lambda: CP.build(R3))
    assert CP.build(R3), 'the unpatched profile builds again'


def _seed_lists(prof):
    """Every list of ints of a profile dict, by key (the lists that can name
    topology seeds; torch seeds are four-digit and can never fall in a block)."""
    return {k: v for k, v in prof.items()
            if isinstance(v, list) and v and all(isinstance(x, int) for x in v)}


def _block_offenders(profiles):
    """[(profile, key, block, seeds)] for every seed list that names a seed
    of a reserved block it may not name."""
    allowed = {('real', 'R2 confirmatory block'), (R3, 'R3 confirmatory block')}
    out, n = [], 0
    for name, prof in profiles.items():
        for key, val in _seed_lists(prof).items():
            n += 1
            for block, hit in CP.reserved_hits(val):
                kind = block.rsplit(' ', 1)[0]
                if (name, kind) not in allowed:
                    out.append((name, key, block, hit))
    return out, n


def test_no_other_profile_names_the_r3_block_or_any_reserved_block():
    assert [b[1:] for b in CP.RESERVED_SEED_BLOCKS] == [(200, 219), (300, 339),
                                                        (340, 359), (400, 439)]
    # the blocks do not overlap and the R3 block is exactly the profile's
    spans = [set(range(lo, hi + 1)) for _n, lo, hi in CP.RESERVED_SEED_BLOCKS]
    assert sum(len(s) for s in spans) == len(set().union(*spans))
    assert sorted(spans[1]) == BLOCK == CP.R3_CONFIRMATORY_SEEDS
    offenders, n = _block_offenders(CP.PROFILES)
    assert n >= 12, 'scanned only %d seed lists' % n
    assert not offenders, offenders
    # the block is named by r3lite and by nothing else
    for name, prof in CP.PROFILES.items():
        hit = sorted({s for v in _seed_lists(prof).values() for s in v} & set(BLOCK))
        assert (hit == BLOCK) if name == R3 else (hit == []), (name, hit)
    # the two reserve blocks are named by nobody at all
    every = {s for prof in CP.PROFILES.values() for v in _seed_lists(prof).values() for s in v}
    assert not (every & (set(range(340, 360)) | set(range(400, 440))))
    # NEGATIVE CONTROL: the scanner sees a planted seed of each block
    for planted, where in ((305, REH), (335, REH), (210, REH), (340, R3), (350, R3),
                           (410, 'real'), (301, 'test'), (339, 'test')):
        fake = copy.deepcopy(CP.PROFILES)
        fake[where]['cf_seeds'] = list(fake[where]['cf_seeds']) + [planted]
        off, _n = _block_offenders(fake)
        assert any(o[0] == where and planted in o[3] for o in off), (planted, where, off)


def test_the_rehearsal_runs_development_seeds_only():
    P = CP.PROFILES[REH]
    dev = set(CP.PROFILES['real']['dev_seeds'])
    assert P['cf_seeds'] == [52, 53] and P['cf_full_seeds'] == [52, 53]
    assert set(P['cf_seeds']) <= dev, 'the stand-in seeds are development seeds'
    assert P['ticks'] == 600 and P['out_prefix'] == 'r3lite_rehearsal/'
    invs = CP.build(REH)
    named = sorted({s for i in invs for s in i['seeds']})
    assert named == [52, 53], named
    assert not CP.reserved_hits(named)
    n = 0
    for i in invs:
        blob = json.dumps(i, sort_keys=True)
        for s in BLOCK:
            # no id, title, path or key of the rehearsal names a block seed
            assert ('|%d"' % s) not in blob and ('_%d_' % s) not in blob, (i['id'], s)
            n += 1
    assert n == len(invs) * 40 and n > 0
    # NEGATIVE CONTROLS: the builder refuses the rehearsal on any reserved seed
    for bad in ([300, 301], [52, 319], [200, 201], [330, 331], [52, 339], [340, 341],
                [359, 52], [400, 401]):
        with _patched_profile(REH, cf_seeds=bad, cf_full_seeds=bad):
            why = _raises_systemexit(lambda: CP.build(REH))
            assert why and 'reserved block' in why, (bad, why)
    # ... and a full-length subset that is not part of its seeds
    with _patched_profile(REH, cf_full_seeds=[54]):
        assert _raises_systemexit(lambda: CP.build(REH))


# ======================================================== R2's comparators
def test_the_confirmatory_stage_is_r2s_confirmatory_stage():
    """Same eight arms in the same order, the same block splitting, the same
    environment, script and worker count as profile `real` -- the only
    differences are the seeds (+100) and the pinned checkpoint."""
    assert CP.CONFIRMATORY_ARMS == ['xldet', 'random', 'sdn', 'ospf', 'olsr',
                                    'batman', 'aodv', 'marl_static']
    real = [i for i in CP.build('real') if i['stage'] == 'S6']
    mine = [i for i in CP.build(R3) if i['stage'] == 'S6']
    # R2 ran 20 instances in 5 blocks of 4; R3-lite runs 40 in 10 blocks of 4.
    # Block j of R3-lite is R2's block j mod 5 with every seed shifted by
    # +100 (j < 5) or +120 (j >= 5): the same block, twice.
    assert len(real) == 5 and len(mine) == 10
    n = 0
    for j, b in enumerate(mine):
        a = real[j % 5]
        shift = 100 + 20 * (j // 5)
        assert b['id'] == 's6_cf_b%d' % j and a['id'] == 's6_cf_b%d' % (j % 5), (a['id'], b['id'])
        assert b['arms'] == a['arms'] == CP.CONFIRMATORY_ARMS
        assert b['seeds'] == [s + shift for s in a['seeds']] == list(range(300 + 4 * j, 304 + 4 * j)), \
            (a['seeds'], b['seeds'])
        assert b['env'] == a['env'], 'the evaluation environment is R2\'s'
        for k in ('script', 'scenarios', 'ticks', 'workers', 'kind', 'ckpt_kind',
                  'hashseed', 'exclusive', 'expected_passes', 'marker', 'ablation_of'):
            assert b[k] == a[k], (k, a[k], b[k])
        shifted = ['%s|%s|%d' % (k.split('|')[0], k.split('|')[1], int(k.split('|')[2]) + shift)
                   for k in a['expected_keys']]
        assert b['expected_keys'] == shifted
        assert b['expected_keys'] != a['expected_keys'], 'negative control: the seeds do differ'
        assert b['out'] == 'r3lite/%s.pkl' % b['id'] and b['artifact'] == 'output/' + b['out']
        n += 1
    assert n == 10
    # the primary analysis: the same script, step and arm list
    r7 = CP.by_id('real', 's7_analysis')
    m7 = CP.by_id(R3, 's7_analysis')
    for k in ('script', 'step', 'arms', 'kind', 'stage'):
        assert m7[k] == r7[k], k
    assert m7['artifact'] == 'confirmatory/r3lite_cf_report.json'
    assert CP.PROFILES[R3]['cf_analysis_prefixes'] == CP.PROFILES['real']['cf_analysis_prefixes']
    assert CP.DEPLOYED_ARM == 'marl_static'
    # the hash-seed repeat: the first two seeds of the block, pair = S6's first block
    h = CP.by_id(R3, 's6h_hash1')
    assert h['seeds'] == [300, 301] and h['hashseed'] == '1' and h['hash_pair_of'] == 's6_cf_b0'
    assert h['arms'] == ['marl_static'] and h['ckpt_kind'] == 'selected'
    assert h['env'] == CP.by_id(R3, 's6_cf_b0')['env']
    assert [i['id'] for i in CP.build(R3) if i.get('hashseed', '0') != '0'] == ['s6h_hash1']


def _key_problems(invs, arms, seeds):
    """(missing, duplicated, unexpected) of the union of the invocations'
    expected keys against arms x scenarios x seeds."""
    want = {'%s|%s|%d' % (a, s, seed) for a in arms for s in CP.SCENARIOS for seed in seeds}
    seen, dupes = set(), []
    for i in invs:
        for k in i['expected_keys']:
            if k in seen:
                dupes.append(k)
            seen.add(k)
    return sorted(want - seen), sorted(dupes), sorted(seen - want)


def test_the_expected_key_set_is_complete_and_duplicate_free():
    invs = CP.build(R3)
    s6 = [i for i in invs if i['stage'] == 'S6']
    assert _key_problems(s6, CP.CONFIRMATORY_ARMS, BLOCK) == ([], [], [])
    assert sum(len(i['expected_keys']) for i in s6) == 8 * 2 * 40 == 640
    assert len(s6) == 10 and all(len(i['seeds']) == 4 for i in s6)
    for i in invs:
        if i['expected_passes']:
            assert i['expected_passes'] == len(i['expected_keys']) \
                == len(set(i['expected_keys'])) == len(i['arms']) * len(i['seeds']) * 2, i['id']
            cap = CP.R3_SINGLE_ARM_MAX_PASSES if len(i['arms']) == 1 else CP.MAX_PASSES_PER_INVOCATION
            assert i['expected_passes'] <= cap, i['id']
    # exactly the two single-arm families hold the whole block in one pickle,
    # above the 64 of a multi-arm block; nothing else does
    assert CP.MAX_PASSES_PER_INVOCATION == 64 and CP.R3_SINGLE_ARM_MAX_PASSES == 80
    assert sorted(i['id'] for i in invs if i['expected_passes'] > 64) == \
        ['s8_untrained_s1234', 's9x_ensemble']
    assert all(i['expected_passes'] == 64 for i in s6)
    per = {}
    for i in invs:
        per[i['stage']] = per.get(i['stage'], 0) + i['expected_passes']
    assert per == {'S0': 0, 'S3b': 0, 'S5': 0, 'S6': 640, 'S9x': 80, 'S7': 0, 'S7e': 0,
                   'S6h': 4, 'S8': 80, 'S12': 0}, per
    assert sum(per.values()) == 804
    # the single-arm families hold every instance of the block once
    for iid in ('s8_untrained_s1234', 's9x_ensemble'):
        assert _key_problems([CP.by_id(R3, iid)], ['marl_static'], BLOCK) == ([], [], [])
    # NEGATIVE CONTROLS: the same check sees a lost block, a repeated block
    # and a foreign seed
    miss, dup, extra = _key_problems(s6[:-1], CP.CONFIRMATORY_ARMS, BLOCK)
    assert len(miss) == 64 and not dup and not extra
    miss, dup, extra = _key_problems(s6 + [s6[0]], CP.CONFIRMATORY_ARMS, BLOCK)
    assert not miss and len(dup) == 64
    miss, dup, extra = _key_problems(s6, CP.CONFIRMATORY_ARMS, BLOCK[1:])
    assert len(extra) == 16


# ================================================== dropped: absent, recorded
def test_the_dropped_stages_are_absent_and_recorded():
    order = CP.stage_order(R3)
    assert order == ['S0', 'S3b', 'S5', 'S6', 'S9x', 'S7', 'S7e', 'S6h', 'S8', 'S12'], order
    assert CP.stage_order(REH) == order, 'the rehearsal runs the same stage graph'
    dropped = CP.dropped_stages(R3)
    # exactly the R2 stages that are not in the R3-lite graph, each with a reason
    assert set(dropped) == set(CP.STAGE_ORDER) - set(order) \
        == {'S1', 'S2', 'S3', 'S4', 'S5b', 'S9', 'S10', 'S11'}, sorted(dropped)
    for st, why in dropped.items():
        assert isinstance(why, str) and len(why) > 20, (st, why)
    n = 0
    for profile in (R3, REH):
        invs = CP.build(profile)
        assert {i['stage'] for i in invs} == set(order), profile
        for i in invs:
            n += 1
            assert i['stage'] not in dropped, i['id']
            assert i['kind'] != 'train' and not i['exclusive'], i['id']
            assert 'marl_freeze' not in i['arms'] and 'marl' not in i['arms'], i['id']
            assert not i.get('ablation_of'), i['id']
            assert not i['shed_group'], '%s: R3-lite sheds nothing' % i['id']
            for k in ('MARL_STRESS_INTERFERENCE_DBM', 'MARL_LS_INSTALL', 'MARL_DETECT_TICKS'):
                assert k not in i['env'], (i['id'], k)
            if i['kind'] == 'eval':
                assert i['env'] == dict(CP.ENGINE_ENV, **dict(CP.POLICY_ENV, **CP.EVAL_ENV)), i['id']
        assert not [i for i in invs if i.get('analysis_role') == 'freeze']
        assert CP.shed_order(profile) == [] and CP.shed_stages(profile) == {}
        assert CP.secondary_stages(profile) == ['S6h', 'S8', 'S9x']
        assert CP.dropped_stages(profile) == dropped
    assert n == 20 + 14, n
    # the graph: acyclic, every dependency runs before its dependant, and the
    # secondary family never gates the primary analysis
    done = set()
    for st, _t, deps, after in CP.STAGES_R3LITE:
        for d in deps + after:
            assert d in done, (st, d)
        done.add(st)
    deps = {s: d for s, _t, d, _o in CP.STAGES_R3LITE}
    after = {s: o for s, _t, _d, o in CP.STAGES_R3LITE}
    assert deps['S7'] == ['S6'] and 'S5' in deps['S6'] and 'S3b' in deps['S6']
    for s in ('S6h', 'S8', 'S9x'):
        assert s not in deps['S7'] and 'S5' in deps[s], s
    # TWO CONFIRMATORY FAMILIES (D-R3.3).  The passes of family E (S9x) are
    # confirmatory DATA: they depend on the seal only, run right after S6 and
    # BEFORE any analysis; S7 follows them by ORDER (a recorded waiver lets
    # family 1 stand alone if they cannot be completed), never by dependency;
    # the family-E analysis needs the family-1 report AND the ensemble passes.
    assert deps['S9x'] == ['S5', 'S0'] and after['S9x'] == ['S6']
    assert after['S7'] == ['S9x'] and 'S9x' not in deps['S7']
    assert deps['S7e'] == ['S7', 'S9x'] and after['S7e'] == []
    assert order.index('S6') < order.index('S9x') < order.index('S7') < order.index('S7e') \
        < min(order.index(s) for s in ('S6h', 'S8', 'S12'))
    # THE CONTINGENCY RUNS TO THE END (review of 2026-10-12): nothing that
    # family 1 needs depends on family E.  Its secondary stages are ordered
    # after S7 only, and S12 depends on family 1 alone and is ORDERED after
    # S7e -- so that a family E that cannot be completed costs family 1
    # nothing, and S12 can record it as not completed.
    assert after['S6h'] == ['S7'] and after['S8'] == ['S7']
    assert deps['S12'] == ['S7', 'S6h', 'S8'] and after['S12'] == ['S7e']
    for s in ('S7', 'S6h', 'S8', 'S12'):
        assert 'S7e' not in deps[s] and 'S9x' not in deps[s], s
    # ... and the two order edges into family E's absence are FAILURE-ONLY:
    # a waiver is honoured only when the ledger records that the restart
    # rule is exhausted for a stage named here, in BOTH profiles
    assert CP.ORDER_WAIVERS_R3LITE == {('S7', 'S9x'): ['S9x'], ('S12', 'S7e'): ['S7e', 'S9x']}
    for (st, aft), need in CP.ORDER_WAIVERS_R3LITE.items():
        assert aft in after[st] and set(need) <= set(order), (st, aft)
        for profile in (R3, REH):
            assert CP.order_waiver(profile, st, aft) == ('failure', need), (profile, st)
            assert _plan_cli('--profile', profile, '--order-waiver', st, aft).stdout.split() == \
                ['failure'] + need
    # the CONFIRMATORY profile refuses every other operator action that could
    # redo or step around a confirmatory invocation; the rehearsal (throwaway
    # numbers, single stages driven by tests) and the R2 profiles do not
    edges = [(st, aft) for st, _t, _d, o in CP.STAGES_R3LITE for aft in o]
    assert sorted(edges) == [('S12', 'S7e'), ('S6h', 'S7'), ('S7', 'S9x'), ('S8', 'S7'), ('S9x', 'S6')]
    for st, aft in edges:
        if (st, aft) not in CP.ORDER_WAIVERS_R3LITE:
            assert CP.order_waiver(R3, st, aft) == ('refused', []), (st, aft)
            assert CP.order_waiver(REH, st, aft) == ('free', []), (st, aft)
    assert CP.waivable_order_stages(R3) == ['S12', 'S7'] and CP.waivable_order_stages(REH) is None
    assert CP.dep_waiver(R3) == 'refused' and CP.dep_waiver(REH) == 'free'
    assert CP.PROFILES[R3]['no_invalidate_from'] == 'S6' and CP.PROFILES[R3]['declared_waivers_only'] is True
    for k in ('no_invalidate_from', 'declared_waivers_only'):
        assert k not in CP.PROFILES[REH], k
    refused = allowed = 0
    for i in CP.build(R3):
        why = CP.invalidate_refusal(R3, i['id'])
        if order.index(i['stage']) >= order.index('S6'):
            assert why and i['id'] in why and 'never invalidated' in why, i['id']
            refused += 1
        else:
            assert why == '', i['id']
            allowed += 1
    assert (allowed, refused) == (4, 16), (allowed, refused)
    for i in CP.build(REH):
        assert CP.invalidate_refusal(REH, i['id']) == '', i['id']
    assert 'unknown invocation' in CP.invalidate_refusal(R3, 'nonsense')
    assert _plan_cli('--profile', R3, '--invalidate-refusal', 's9x_ensemble').stdout.strip().startswith(
        's9x_ensemble belongs to stage S9x')
    assert _plan_cli('--profile', R3, '--invalidate-refusal', 's0_manifest').stdout.strip() == ''
    assert _plan_cli('--profile', R3, '--waivable-order-stages').stdout.split() == ['S12', 'S7']
    assert _plan_cli('--profile', R3, '--dep-waiver').stdout.strip() == 'refused'
    for profile in ('real', 'rehearsal', 'test'):
        assert CP.order_waiver(profile, 'S7', 'S9x') == ('free', [])
        assert CP.waivable_order_stages(profile) is None and CP.dep_waiver(profile) == 'free'
        assert CP.invalidate_refusal(profile, 'anything') == ''
        assert _plan_cli('--profile', profile, '--waivable-order-stages').stdout.strip() == '*'
    for profile in (R3, REH):
        e = CP.by_id(profile, 's7e_family_e')
        p = CP.by_id(profile, 's7_analysis')
        assert e['stage'] == 'S7e' and e['kind'] == 'analysis' and e['step'] == 'family_e'
        assert e['script'] == p['script'] == 'confirmatory/stage_analysis.sh'
        assert e['arms'] == p['arms'] and e['seeds'] == p['seeds'] and e['scenarios'] == p['scenarios']
        assert e['artifact'] == p['artifact'].replace('cf_report.json', 'cf_report_family_e.json')
        assert e['artifact'] != p['artifact'] and e['expected_passes'] == 0
        assert [i['id'] for i in CP.build(profile) if i['stage'] == 'S7e'] == ['s7e_family_e']
    # the CLI the launcher and the stage scripts ask
    out = _plan_cli('--profile', R3, '--dropped')
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    assert sorted(l.split('\t')[0] for l in lines) == sorted(dropped), lines
    assert all(l.split('\t', 1)[1] == dropped[l.split('\t')[0]] for l in lines)
    assert _plan_cli('--profile', R3, '--stages').stdout.split() == order
    assert _plan_cli('--profile', R3, '--dropped-reason', 'S9').stdout.strip() == dropped['S9']
    assert _plan_cli('--profile', R3, '--secondary-stages').stdout.split() == ['S6h', 'S8', 'S9x']
    assert _plan_cli('--profile', R3, '--shed-order').stdout.strip() == ''
    assert _plan_cli('--profile', R3, '--graph').stdout.strip() == 'r3lite'
    # NEGATIVE CONTROL: an R2 profile drops nothing, keeps its graph, and a
    # stage that IS run has no dropped-reason
    for profile in ('real', 'rehearsal', 'dryrun', 'test'):
        assert CP.dropped_stages(profile) == {} and CP.graph_of(profile) == 'r2'
        assert CP.stage_order(profile) == CP.STAGE_ORDER
        assert CP.shed_order(profile) == CP.SHED_ORDER
        assert CP.secondary_stages(profile) == ['S6h', 'S8', 'S9', 'S10', 'S11']
    assert _plan_cli('--profile', 'real', '--dropped').stdout == ''
    assert _plan_cli('--profile', 'real', '--stages').stdout.split() == CP.STAGE_ORDER
    assert _plan_cli('--profile', R3, '--dropped-reason', 'S6').stdout.strip() == ''
    assert _plan_cli('--profile', 'real', '--shed-group-of', 'S8').stdout.strip() == 'untrained'
    assert _plan_cli('--profile', R3, '--shed-group-of', 'S8').stdout.strip() == ''


# ================================================================ the pins
def _selected_fields(path):
    import re
    with open(path) as f:
        lines = f.read().splitlines()
    seeds = [int(m.group(1)) for l in lines
             for m in [re.match(r'#\s*selected replicate:\s*torch seed\s+(\d+)\s*$', l)] if m]
    shas = [m.group(1) for l in lines
            for m in [re.match(r'#\s*sha256\s+([0-9a-f]{64})\s*$', l)] if m]
    return lines[0].strip(), seeds, shas


def test_the_selected_checkpoint_is_pinned_everywhere():
    assert CP.R2_CHECKPOINT_SHA256[1234] == SHA_1234
    assert sorted(CP.R2_CHECKPOINT_SHA256) == [1234, 2345, 3456, 4567]
    assert len(set(CP.R2_CHECKPOINT_SHA256.values())) == 4
    for v in list(CP.R2_CHECKPOINT_SHA256.values()) + list(CP.R2_UNTRAINED_SHA256.values()):
        assert len(v) == 64 and set(v) <= set('0123456789abcdef'), v
    assert CP.R2_UNTRAINED_SHA256[1234] not in CP.R2_CHECKPOINT_SHA256.values()
    n = 0
    for profile, pre in ((R3, 'r3lite/'), (REH, 'r3lite_rehearsal/')):
        P = CP.PROFILES[profile]
        assert P['selected_seed'] == 1234 and P['untrained_seeds'] == [1234]
        assert P['ensemble_seeds'] == [1234, 2345, 3456, 4567]
        path, seed, sha = CP.selected_checkpoint(profile)
        assert (path, seed, sha) == ('output/%sck_s1234/policy_final.pt' % pre, 1234, SHA_1234)
        # the tracked SELECTED file says the same thing
        sel = os.path.join(ROOT, CP.selected_file(profile))
        assert CP.selected_file(profile) == 'confirmatory/SELECTED_%s.txt' % profile
        first, seeds, shas = _selected_fields(sel)
        assert first == path and seeds == [1234] and shas == [SHA_1234], (profile, first, seeds, shas)
        for i in CP.build(profile):
            if i['kind'] != 'eval':
                assert 'ckpt_sha256' not in i, i['id']
                continue
            n += 1
            pins = i['ckpt_sha256']
            if i['stage'] in ('S6', 'S6h'):
                assert i['ckpt_kind'] == 'selected' and pins == [SHA_1234], i['id']
            elif i['stage'] == 'S8':
                assert i['ckpt_kind'] == 'literal', i['id']
                assert i['ckpt_value'] == 'output/%suntrained_s1234/policy_init.pt' % pre
                assert pins == [CP.R2_UNTRAINED_SHA256[1234]], i['id']
            elif i['stage'] == 'S9x':
                assert i['ckpt_kind'] == 'ensemble', i['id']
                assert i['ckpt_value'].split(',') == [
                    'output/%sck_s%d/policy_final.pt' % (pre, s) for s in (1234, 2345, 3456, 4567)]
                assert pins == [CP.R2_CHECKPOINT_SHA256[s] for s in (1234, 2345, 3456, 4567)]
            else:
                raise AssertionError('unexpected evaluation stage %s' % i['stage'])
            # what the launcher sources
            kv = dict(l.split('=', 1) for l in CP.emit(i).splitlines())
            assert kv['INV_CKPT_SHA256'] == ','.join(pins), i['id']
        # the import list IS the pin table
        imp = CP.imports_of(profile)
        assert len(imp) == 5, imp
        assert sorted((x['role'], x['torch_seed']) for x in imp) == [
            ('ensemble_member', 2345), ('ensemble_member', 3456), ('ensemble_member', 4567),
            ('selected', 1234), ('untrained', 1234)]
        for x in imp:
            want = (CP.R2_UNTRAINED_SHA256 if x['role'] == 'untrained'
                    else CP.R2_CHECKPOINT_SHA256)[x['torch_seed']]
            assert x['sha256'] == want and x['dest'].startswith('output/' + pre), x
            assert x['src'].startswith('~/MARL_run_9/output/final/'), x['src']
    assert n == 13 + 7, n    # r3lite: 10 S6 + S6h + S8 + S9x; the rehearsal: 2 + 1 + 2 + 2
    # an R2 invocation carries no pin and emits an EMPTY one (never a stale one)
    for i in CP.build('real'):
        assert 'ckpt_sha256' not in i
        assert 'INV_CKPT_SHA256=' + "''" in CP.emit(i).splitlines(), i['id']
    # NEGATIVE CONTROL: the SELECTED parser sees a wrong digest and a wrong path
    with _scratch('r3_sel_') as d:
        with open(os.path.join(ROOT, CP.selected_file(R3))) as f:
            good = f.read()
        bad = os.path.join(d, 'SELECTED.txt')
        with open(bad, 'w') as f:
            f.write(good.replace(SHA_1234, CP.R2_CHECKPOINT_SHA256[4567]).replace('ck_s1234', 'ck_s4567'))
        first, seeds, shas = _selected_fields(bad)
        assert first != CP.selected_checkpoint(R3)[0] and shas != [SHA_1234]


def test_the_ensemble_is_one_invocation_of_family_e_with_its_own_pickle():
    for profile in (R3, REH):
        invs = {i['id']: i for i in CP.build(profile)}
        role = [i for i in invs.values() if i.get('analysis_role') == 'ensemble']
        assert len(role) == 1, (profile, [i['id'] for i in role])
        e = role[0]
        prim = sorted(CP.PROFILES[profile].get('cf_full_seeds') or CP.PROFILES[profile]['cf_seeds'])
        assert e['stage'] == 'S9x' and e['arms'] == ['marl_static'], e['id']
        assert sorted(e['seeds']) == prim and e['ticks'] is None, e['id']
        assert e['artifact'] == 'output/%s%s.pkl' % (CP.PROFILES[profile]['out_prefix'], e['id'])
        # it can never be read by the primary analysis
        for pre in CP.PROFILES[profile]['cf_analysis_prefixes'].split(','):
            assert not e['id'].startswith(pre)
        assert e['artifact'] not in [i['artifact'] for i in invs.values() if i['stage'] == 'S6']
        assert e['env'] == invs['s6_cf_b0']['env'], 'the ensemble runs in the confirmatory environment'
        un = [i for i in invs.values() if i.get('analysis_role') == 'untrained']
        assert [i['control_name'] for i in un] == ['untrained_s1234'], profile
        assert sorted(un[0]['seeds']) == prim and un[0]['ticks'] is None
    assert CP.by_id(R3, 's9x_ensemble')['analysis_role'] == 'ensemble'
    assert CP.by_id(REH, 's9x_ensemble')['analysis_role'] == '', 'the reduced pickle is not analysed'
    assert CP.by_id(REH, 's9x_ensemblefull')['analysis_role'] == 'ensemble'
    assert _plan_cli('--profile', R3, '--analysis-role', 'ensemble').stdout.split() == ['s9x_ensemble']
    assert _plan_cli('--profile', 'real', '--analysis-role', 'ensemble').stdout.split() == []


def test_the_rehearsal_is_the_same_graph_at_reduced_scale():
    a = {i['id']: i for i in CP.build(R3)}
    b = {i['id']: i for i in CP.build(REH)}
    # every stage-level id of the real profile exists in the rehearsal (S6 has
    # one block instead of five), with the same stage, kind, script and step
    n = 0
    for iid, i in a.items():
        if iid.startswith('s6_cf_b') and iid != 's6_cf_b0':
            continue
        assert iid in b, iid
        for k in ('stage', 'kind', 'script', 'arms', 'ckpt_kind'):
            assert b[iid][k] == i[k], (iid, k)
        assert b[iid].get('step') == i.get('step'), iid
        n += 1
    assert n == len(a) - 9 and n >= 11
    assert sorted(set(b) - set(a)) == ['s6_cffull_b0', 's8_untrainedfull_s1234', 's9x_ensemblefull']
    for iid, i in b.items():
        if i['kind'] != 'eval':
            continue
        assert i['ticks'] == (None if 'full' in iid else 600), iid
        assert i['seeds'] == [52, 53], iid
    assert CP.PROFILES[REH]['cf_analysis_prefixes'] == 's6_cffull_b'
    assert b['s6h_hash1']['hash_pair_of'] == 's6_cf_b0' and b['s6h_hash1']['ticks'] == 600


# ============================================================ no collisions
def _owned_paths(profile):
    """Every path a chain of `profile` can write: the plan's artefacts, the
    files the stage scripts derive from them, and the launcher's own files."""
    out = set()
    for i in CP.build(profile):
        art = i['artifact']
        out.add(art)
        if art.endswith('.json'):
            stem = art[:-len('.json')]
            out.update({stem + '.txt', stem + '.manifest.json'})
        if art.endswith('secondary_report.json'):
            base = art[:-len('secondary_report.json')]
            for x in ('s8_hash.json', 's8_hash.txt', 'cf_report_full.json',
                      'cf_report_full.txt', 'cf_report_full.manifest.json',
                      'cf_report_full.compare.txt', 'ensemble_vs_xldet.json',
                      'ensemble_vs_xldet.txt'):
                out.add(base + x)
        if art.endswith('cf_report_family_e.json'):
            # what chain S7e writes beside the family-E report, and its two
            # directories of assembled pickles
            base = art[:-len('cf_report_family_e.json')]
            for x in ('family_e_record.json', 'family_e_assembly.json', 'family_ctl_assembly.json',
                      'family_e_ledger_order.json', 'cf_report_family_e.err',
                      'cf_report_family_ctl.json', 'cf_report_family_ctl.txt',
                      'cf_report_family_ctl.compare.txt', 'cf_report_family_ctl.compare_rev.txt'):
                out.add(base + x)
            pre = CP.PROFILES[profile]['out_prefix']
            out.update({'output/%sfamily_e/' % pre, 'output/%sfamily_ctl/' % pre})
    out.update({'confirmatory/chain_%s.jsonl' % profile, 'logs/chain_%s.log' % profile,
                'confirmatory/SELECTED_%s.txt' % profile,
                'confirmatory/FREEZE_FINAL_%s.txt' % profile})
    dirs = {'confirmatory/preflight_%s/' % profile, 'confirmatory/env_%s/' % profile,
            'logs/chain_%s/' % profile, 'confirmatory/.lock_%s.shared/' % profile,
            'confirmatory/.lock_%s.excl/' % profile,
            'output/%s' % CP.PROFILES[profile]['out_prefix']}
    return out, dirs


def _collisions(pa, pb):
    fa, da = pa
    fb, db = pb
    bad = sorted(fa & fb) + sorted(da & db)
    bad += sorted(f for f in fa for d in db if f.startswith(d))
    bad += sorted(f for f in fb for d in da if f.startswith(d))
    return bad


def test_no_artifact_path_collisions_between_profiles():
    owned = {p: _owned_paths(p) for p in CP.PROFILES}
    n = 0
    for a in sorted(owned):
        for b in sorted(owned):
            if a < b:
                assert not _collisions(owned[a], owned[b]), (a, b, _collisions(owned[a], owned[b])[:5])
                n += 1
    assert n == 15, n                    # six profiles, every pair
    # every pickle of a profile lies under its own output directory, and the
    # R3-lite reports carry the profile's own prefix
    for p in (R3, REH):
        pre = CP.PROFILES[p]['out_prefix']
        for i in CP.build(p):
            art = i['artifact']
            if art.startswith('output/'):
                assert art.startswith('output/' + pre), (p, art)
            else:
                assert art.startswith('confirmatory/' + pre.replace('/', '_')) \
                    or art == 'confirmatory/FREEZE_FINAL_%s.txt' % p, (p, art)
    assert CP.by_id(R3, 's0_manifest')['artifact'] == 'confirmatory/r3lite_manifest_300_339.json'
    assert CP.by_id(REH, 's0_manifest')['artifact'] == 'confirmatory/r3lite_rehearsal_manifest_52_53.json'
    # NEGATIVE CONTROLS: a rehearsal that wrote under r3lite's prefix, or kept
    # r3lite's report name, IS reported
    with _patched_profile(REH, out_prefix='r3lite/'):
        assert _collisions(_owned_paths(R3), _owned_paths(REH))
    fa, da = _owned_paths(R3)
    assert _collisions((fa, da), ({'output/r3lite/anything.pkl'}, set()))
    assert _collisions((fa, da), ({'confirmatory/r3lite_cf_report.txt'}, set()))


# =============================================== the R2 profiles are untouched
# Recorded on <compute-server> from the tree synced at 6b53f3f (the frozen R2 code),
# 2026-10-11: sha256 of json.dumps(build(p), sort_keys=True), of the '\n'-joined
# emit() blocks, and of json.dumps(PROFILES[p], sort_keys=True).
R2_PLAN_DIGESTS = {
    'real': (37, 1060,
             '50f698e01e0961f7e0c1e0900a2eefd8efd94b585e078603f169702e1b6d275d',
             'a885ca58a4bea5a8ab1567deb7da6ffc571df9932b8401666fe6e827e2a52093',
             '43a306be37eba854cf78992caebcc5ef4e8d8f9d419b2cafe3d7a5341565cf18'),
    'rehearsal': (33, 288,
                  '38b2225009dc05bcdc81648a39e24944fb1659d7e362b994c3e32bc29c1a5f5f',
                  'bde4c4493a744b3b2d040a34efd22f33176f177a3f7641cf193cbefa821b9b9f',
                  '31c74a8925dd1395299cbbf68b750780b9f50c5478c90540d1a9400894e6b955'),
    'dryrun': (37, 1060,
               '397b0e8fc59a27e7b38eee57660100978be2531b1cb7d6ea16921501c8b0836a',
               '48c7b97d5fe5c02d645735141758c8aa2298f0ec6c012e3f4ade6ab610b8242d',
               '184e4ba64fad6eea23469f53b51a60ea9e7337651b2a34a5f9bdaa38813e310f'),
    'test': (23, 100,
             'df17fd2bca0e7714d4582d2c89fdd83ff8b240b7e5bceab1b25220e6c6ddc506',
             'eb3bfbee978bb999a38e0d73347591756dcbe1647c439b9bb61e7a6a0c850190',
             '39c4a3b5082ca631e8a999065f24a6752515bcd2ca6a3beb9a8b8c1f589b9122'),
}
R2_CONSTANTS_DIGEST = '9e8d5fc98a1b0e12dca5b02f166891dea3f1c1d81862cd1d9ff4f7061654e17e'
# The one line emit() gained with R3-lite (always present, empty for R2).
NEW_EMIT_KEYS = ('INV_CKPT_SHA256',)


def _plan_digests(profile, build=None):
    b = build if build is not None else CP.build(profile)
    j = json.dumps(b, sort_keys=True)
    e = '\n'.join('\n'.join(l for l in CP.emit(i).splitlines()
                            if l.split('=', 1)[0] not in NEW_EMIT_KEYS) for i in b)
    prof = json.dumps(CP.PROFILES[profile], sort_keys=True)
    return (len(b), sum(i['expected_passes'] for i in b),
            hashlib.sha256(j.encode()).hexdigest(),
            hashlib.sha256(e.encode()).hexdigest(),
            hashlib.sha256(prof.encode()).hexdigest())


def test_the_r2_profiles_are_bit_for_bit_what_they_were():
    n = 0
    for profile, want in sorted(R2_PLAN_DIGESTS.items()):
        assert _plan_digests(profile) == want, profile
        n += 1
    assert n == 4
    const = hashlib.sha256(json.dumps(
        [CP.STAGES, CP.SHED_ORDER, CP.SHED_STAGES, CP.ENGINE_ENV, CP.POLICY_ENV,
         CP.EVAL_ENV, CP.CONFIRMATORY_ARMS], sort_keys=True).encode()).hexdigest()
    assert const == R2_CONSTANTS_DIGEST
    assert sorted(CP.PROFILES) == ['dryrun', 'r3lite', 'r3lite_rehearsal', 'real',
                                   'rehearsal', 'test']
    # the emit block gained exactly the declared key(s), nothing else
    keys = [l.split('=', 1)[0] for l in CP.emit(CP.by_id('real', 's6_cf_b0')).splitlines()]
    assert keys[-len(NEW_EMIT_KEYS):] == list(NEW_EMIT_KEYS) and len(keys) == 31, keys
    # NEGATIVE CONTROL: the digest moves when ONE field of ONE invocation moves
    tampered = CP.build('real')
    tampered[10]['workers'] = 23
    assert _plan_digests('real', tampered)[2] != R2_PLAN_DIGESTS['real'][2]
    tampered = CP.build('real')
    tampered[-1]['title'] += ' '
    assert _plan_digests('real', tampered)[3] != R2_PLAN_DIGESTS['real'][3]
    with _patched_profile('real', eval_workers=23):
        assert _plan_digests('real')[4] != R2_PLAN_DIGESTS['real'][4]


# ==================================== the analysis script and the engine files
def test_the_analysis_script_is_the_preregistered_file():
    with open(os.path.join(ROOT, 'confirmatory_analysis.py'), 'rb') as f:
        data = f.read()
    assert hashlib.md5(data).hexdigest() == ANALYSIS_MD5 == CP.R3_ANALYSIS_MD5
    assert b'\r\n' not in data, 'CRLF would be a different file'
    assert hashlib.md5(data + b'\n').hexdigest() != ANALYSIS_MD5      # negative control
    # the pre-registered arm and the families the pre-registration quotes
    import confirmatory_analysis as CA
    assert CA.PREREG_MARL_ARM == 'marl_static' and CA.ALPHA == 0.05 and CA.MIN_NONZERO == 6
    assert [e[0] for e in CA.S9_ENDPOINTS] == ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
    assert CA.T_TOTAL == 3600 and CA.W == 500
    # No chain stage of R3-lite carries the name of a stability ENDPOINT: the
    # ensemble stage is S9x, so that "S9e" means one thing (the endpoint A/M7)
    # in the plan, the ledger, the reports and the pre-registration.
    endpoints = {e[0].lower() for e in CA.S9_ENDPOINTS}
    n = 0
    for profile in (R3, REH):
        for st in CP.stage_order(profile):
            n += 1
            assert st.lower() not in endpoints, st
        for i in CP.build(profile):
            n += 1
            assert i['id'].split('_')[0].lower() not in endpoints, i['id']
    assert n == 2 * 10 + 20 + 14, n
    assert 's9e' in endpoints and 'S9x' in CP.stage_order(R3)       # negative control


ENGINE_PATHS = ('sixg_sim', 'run_timeline_comparison.py', 'train_on_comparison.py',
                'confirmatory_analysis.py')


def test_no_engine_file_differs_from_r2():
    """Every arm must be bit-identical to R2.  The pin file is the output of
    `git ls-tree -r 6b53f3f -- <paths>`; the check recomputes each blob id
    from the file on disk, so it runs on the server, where there is no git.
    Where the tree IS a git work tree, the pin file and the tree are also
    compared with git itself."""
    assert CP.R3_ENGINE_HEAD == '6b53f3f'
    assert CP.ENGINE_PIN_FILE == 'confirmatory/ENGINE_PIN_6b53f3f.txt'
    # the blob id function is git's (the id of the empty blob is well known)
    with _scratch('r3_blob_') as d:
        empty = os.path.join(d, 'empty')
        open(empty, 'wb').close()
        assert CP.git_blob_id(empty) == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    pins = CP.read_engine_pin(ROOT)
    assert len(pins) == 62, len(pins)
    for must in ('run_timeline_comparison.py', 'train_on_comparison.py',
                 'confirmatory_analysis.py', 'build_benchmark_manifest.py',
                 'sixg_sim/simulation.py', 'sixg_sim/agent.py', 'sixg_sim/runinfo.py',
                 'confirmatory/stage_eval.sh', 'tools/hash_invariance_check.py',
                 'tools/compare_reports.py'):
        assert must in pins, must
    assert sum(1 for p in pins if p.startswith('sixg_sim/')) == 40
    n, problems = CP.engine_pin_check(ROOT)
    assert n == 62 and not problems, problems[:10]
    cli = _plan_cli('--engine-pin-check')
    assert cli.returncode == 0 and '[ENGINE-PIN] PASS 62 file(s)' in cli.stdout, cli.stdout
    # git, where there is one: the diff the work package was required to keep empty
    if os.path.exists(os.path.join(ROOT, '.git')) and shutil.which('git'):
        g = subprocess.run(['git', '-c', 'core.autocrlf=false', 'diff', '--name-only',
                            CP.R3_ENGINE_HEAD, '--'] + list(ENGINE_PATHS),
                           cwd=ROOT, capture_output=True, text=True)
        assert g.returncode == 0 and g.stdout.strip() == '', g.stdout + g.stderr
        ls = subprocess.run(['git', '-c', 'core.autocrlf=false', 'ls-tree', '-r',
                             CP.R3_ENGINE_HEAD, '--'] + sorted(set(p.split('/')[0] if p.startswith(('sixg_sim/', 'config/')) else p for p in pins)),
                            cwd=ROOT, capture_output=True, text=True)
        got = dict((l.split('\t')[1], l.split()[2]) for l in ls.stdout.splitlines() if l.strip())
        assert got == pins, 'the pin file is not `git ls-tree -r 6b53f3f`'
    # NEGATIVE CONTROLS on a miniature tree: a changed byte, a missing file
    # and an ADDED file in a pinned directory are each reported
    with _scratch('r3_pin_') as d:
        os.makedirs(os.path.join(d, 'confirmatory'))
        os.makedirs(os.path.join(d, 'sixg_sim'))
        os.makedirs(os.path.join(d, 'config'))
        shutil.copy(os.path.join(ROOT, 'sixg_sim', 'runinfo.py'), os.path.join(d, 'sixg_sim', 'runinfo.py'))
        shutil.copy(os.path.join(ROOT, 'confirmatory_analysis.py'), os.path.join(d, 'confirmatory_analysis.py'))
        with open(os.path.join(d, CP.ENGINE_PIN_FILE), 'w') as f:
            for p in ('confirmatory_analysis.py', 'sixg_sim/runinfo.py'):
                f.write('100644 blob %s\t%s\n' % (pins[p], p))
        assert CP.engine_pin_check(d) == (2, [])
        with open(os.path.join(d, 'sixg_sim', 'runinfo.py'), 'ab') as f:
            f.write(b'\n')
        n2, pr = CP.engine_pin_check(d)
        assert n2 == 2 and len(pr) == 1 and pr[0].startswith('DIFFERENT sixg_sim/runinfo.py'), pr
        shutil.copy(os.path.join(ROOT, 'sixg_sim', 'runinfo.py'), os.path.join(d, 'sixg_sim', 'runinfo.py'))
        with open(os.path.join(d, 'sixg_sim', 'extra_module.py'), 'w') as f:
            f.write('x = 1\n')
        n3, pr = CP.engine_pin_check(d)
        assert pr == ['ADDED sixg_sim/extra_module.py (not in the R2 tree)'], pr
        os.remove(os.path.join(d, 'sixg_sim', 'extra_module.py'))
        os.remove(os.path.join(d, 'confirmatory_analysis.py'))
        n4, pr = CP.engine_pin_check(d)
        assert n4 == 1 and pr == ['MISSING confirmatory_analysis.py'], pr
        with open(os.path.join(d, CP.ENGINE_PIN_FILE), 'w') as f:
            f.write('')
        n5, pr = CP.engine_pin_check(d)
        assert n5 == 0 and pr[-1] == 'no pinned file was compared', pr


def test_r3lite_runs_at_the_registered_worker_count():
    import importlib.util
    spec = importlib.util.spec_from_file_location('preflight_r3', os.path.join(ROOT, 'tools', 'preflight.py'))
    PF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PF)
    assert PF.FULL_SCALE_PROFILES == ('real', 'dryrun', 'r3lite')
    # a busy shared host: the load rule caps the rehearsal, never the registered run
    assert PF.recommend_workers(24, 30.0, 64.0, R3)[0] == 24
    assert PF.recommend_workers(24, 30.0, 64.0, 'real')[0] == 24
    assert PF.recommend_workers(24, 30.0, 64.0, REH)[0] == 8          # negative control
    assert PF.recommend_workers(24, 1.0, 64.0, REH)[0] == 16
    assert PF.recommend_workers(24, 1.0, 10.0, R3)[0] == 16, 'the memory ceilings still apply'


# ===================================================== the launcher's pieces
def test_the_environment_file_carries_an_ensemble_as_a_list_of_absolute_paths():
    _linux_or_fail('chain_lib.sh needs bash')
    with _scratch('r3_env_') as d:
        def write(ckpt):
            out = subprocess.run(
                ['bash', '-c',
                 'set -u; . confirmatory/chain_lib.sh; CHAIN_PROFILE=x; CHAIN_ENVDIR="$1"; '
                 'chain_write_env probe "MARL_STEER_MODEL=actionable" "$2" 24; '
                 'echo "TREE=$CHAIN_TREE"; cat "$CHAIN_ENVFILE"; echo ---; cat "$CHAIN_MARLENV"',
                 '_', d, ckpt], cwd=ROOT, capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            tree = [l for l in out.stdout.splitlines() if l.startswith('TREE=')][0][5:]
            return tree, out.stdout
        ens = CP.by_id(REH, 's9x_ensemblefull')['ckpt_value']
        tree, txt = write(ens)
        want = ','.join('%s/%s' % (tree, p) for p in ens.split(','))
        assert want.count(',') == 3 and want.count(tree) == 4
        assert 'export MARL_EVAL_CHECKPOINT=%s' % want in txt.splitlines()
        assert 'MARL_EVAL_CHECKPOINT=%s' % want in txt.split('---')[1].splitlines()
        # ONE path is written exactly as before
        tree, txt = write('output/sel.pt')
        assert 'export MARL_EVAL_CHECKPOINT=%s/output/sel.pt' % tree in txt.splitlines()
        # none at all: no checkpoint line (negative control)
        tree, txt = write('')
        assert 'MARL_EVAL_CHECKPOINT' not in txt


def _write_run_pickle(path, arm, seeds, conn, ck_sha, hashseed='0', ticks=3600, energy=2.0,
                      removed=0.0, tree='t' * 64):
    rec = {'recovery_multi': {arm: {}}, 'rescue_multi': {arm: {}}, 'seeds': list(seeds),
           'provenance': {'tree_head': 'faketest', 'tree_sha256': tree, 'schema_id': 'fake',
                          'pythonhashseed': hashseed, 'checkpoint_sha256': list(ck_sha)},
           'pass_stats': {}}
    for key in ('recovery_multi', 'rescue_multi'):
        for s in seeds:
            rec[key][arm][s] = {'conn': [float(conn + s % 3)] * ticks,
                                'energy': [energy] * ticks, 'delivered': [1.0] * ticks,
                                'relay_removed': [removed] * ticks}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(rec, f)


def _stage_report(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'stage_report.py')]
                          + list(args), cwd=ROOT, capture_output=True, text=True)


def test_the_completeness_gate_requires_the_pinned_checkpoint_digest():
    with _scratch('r3_verify_') as d:
        inv = CP.by_id(REH, 's9x_ensemblefull')
        pins = inv['ckpt_sha256']
        pkl = os.path.join(d, 'e.pkl')
        _write_run_pickle(pkl, 'marl_static', inv['seeds'], 50.0, pins, ticks=4)
        keys = os.path.join(d, 'e.keys')
        with open(keys, 'w') as f:
            f.write('\n'.join(inv['expected_keys']) + '\n')
        ok = _stage_report('--verify', pkl, '--keys-file', keys,
                           '--expect-checkpoint-sha256', ','.join(pins))
        assert ok.returncode == 0 and 'checkpoint_sha256: 4 recorded, 4 pinned -> MATCH' in ok.stdout, ok.stdout
        # NEGATIVE CONTROLS: another checkpoint, the right ones in another
        # order, one member missing, nothing recorded
        for bad in (pins[::-1], pins[:3], [SHA_1234], []):
            _write_run_pickle(pkl, 'marl_static', inv['seeds'], 50.0, bad, ticks=4)
            r = _stage_report('--verify', pkl, '--keys-file', keys,
                              '--expect-checkpoint-sha256', ','.join(pins))
            assert r.returncode != 0 and 'MISMATCH' in r.stdout, (bad, r.stdout)
        # without the flag the gate is what it was (the R2 profiles)
        r = _stage_report('--verify', pkl, '--keys-file', keys)
        assert r.returncode == 0 and 'checkpoint_sha256' not in r.stdout, r.stdout


def test_the_ensemble_comparison_counts_what_it_compared():
    """tools/stage_report.py --paired-descriptive on synthetic full-length
    pickles: the paired difference is the planted one, and every way of
    comparing less than the declared set FAILS."""
    with _scratch('r3_paired_') as d:
        seeds = [52, 53]
        pins = CP.by_id(REH, 's9x_ensemblefull')['ckpt_sha256']
        ens = os.path.join(d, 'ens.pkl')
        ref = os.path.join(d, 'ref.pkl')
        out = os.path.join(d, 'out.json')
        _write_run_pickle(ens, 'marl_static', seeds, 55.0, pins, energy=1.0, removed=0.0)
        _write_run_pickle(ref, 'xldet', seeds, 50.0, [], energy=2.0, removed=1.0)

        def run(test=ens, baseline=ref, want='52,53', pin=','.join(pins)):
            if os.path.exists(out):
                os.remove(out)
            return _stage_report('--paired-descriptive', test, '--baseline', baseline,
                                 '--paired-seeds', want, '--paired-label', 'ensemble',
                                 '--expect-checkpoint-sha256', pin, '--json-out', out)
        r = run()
        assert r.returncode == 0, r.stdout + r.stderr
        rep = json.load(open(out))
        assert rep['ok'] is True and rep['descriptive'] is True and rep['n_pairs'] == 4
        assert sorted(rep['scenarios']) == ['A', 'B']
        for k in ('A', 'B'):
            a = rep['scenarios'][k]['ach']['paired']
            assert a['n'] == 2 and abs(a['mean'] - 5.0) < 1e-9, a
            j = rep['scenarios'][k]['jpu']['paired']
            assert abs(j['mean'] - (-1.0)) < 1e-9, j           # 1.0 - 2.0 J per unit
            m = rep['scenarios'][k]['m7']['paired']
            assert abs(m['mean'] - (-1000.0)) < 1e-6, m        # 0 against 1 per tick
            assert sorted(rep['scenarios'][k]['ach']['per_seed_difference']) == ['52', '53']
        assert rep['checkpoint_sha256'] == pins and rep['arm'] == 'marl_static' and rep['ref_arm'] == 'xldet'
        assert '[PAIRED] PASS ensemble' in r.stdout and 'DESCRIPTIVE' in r.stdout
        # NEGATIVE CONTROLS
        n = 0
        assert run(want='52,53,54').returncode != 0; n += 1          # a declared seed is missing
        assert run(want='52').returncode != 0; n += 1                # an undeclared seed is present
        assert run(pin=SHA_1234).returncode != 0; n += 1             # not the pinned ensemble
        assert run(want='').returncode != 0; n += 1                  # nothing declared
        short = os.path.join(d, 'short.pkl')
        _write_run_pickle(short, 'marl_static', seeds, 55.0, pins, ticks=600)
        assert run(test=short).returncode != 0; n += 1               # a reduced-length pass
        other = os.path.join(d, 'other.pkl')
        _write_run_pickle(other, 'xldet', seeds, 50.0, [], tree='u' * 64)
        assert run(baseline=other).returncode != 0; n += 1           # another tree
        assert run(test=ens + ',' + ens).returncode != 0; n += 1     # an instance twice
        h1 = os.path.join(d, 'h1.pkl')
        _write_run_pickle(h1, 'xldet', seeds, 50.0, [], hashseed='1')
        assert run(baseline=h1).returncode != 0; n += 1              # another hash seed
        assert run(baseline=os.path.join(d, 'nope.pkl')).returncode != 0; n += 1
        assert n == 9
        assert run().returncode == 0, 'the unchanged input passes again'


# ------------------------------------------------------- import, record, seal
def _stage_tree(d, engine=False):
    """A scratch tree holding the chain's own scripts and plan (copies).  With
    engine=True also the pinned engine files, as symbolic links into the real
    tree, and a .synced_* record so that the tree has a digest."""
    os.makedirs(os.path.join(d, 'confirmatory'))
    os.makedirs(os.path.join(d, 'tools'))
    for f in os.listdir(os.path.join(ROOT, 'confirmatory')):
        src = os.path.join(ROOT, 'confirmatory', f)
        if os.path.isfile(src) and (f.endswith(('.sh', '.py')) or f.startswith(('ENGINE_PIN_', 'SELECTED_r3lite'))):
            shutil.copy(src, os.path.join(d, 'confirmatory', f))
    for f in os.listdir(os.path.join(ROOT, 'tools')):
        src = os.path.join(ROOT, 'tools', f)
        if os.path.isfile(src) and f.endswith(('.py', '.sh', '.txt')):
            shutil.copy(src, os.path.join(d, 'tools', f))
    tracked = sorted('confirmatory/' + f for f in os.listdir(os.path.join(d, 'confirmatory'))) \
        + sorted('tools/' + f for f in os.listdir(os.path.join(d, 'tools')))
    if engine:
        pins = CP.read_engine_pin(ROOT)
        for top in sorted({p.split('/')[0] for p in pins}):
            if top in ('confirmatory', 'tools'):
                continue
            os.symlink(os.path.join(ROOT, top), os.path.join(d, top))
        tracked += sorted(p for p in pins if not p.startswith(('confirmatory/', 'tools/')))
    with open(os.path.join(d, '.synced_files'), 'w') as f:
        f.write('\n'.join(tracked) + '\n')
    with open(os.path.join(d, '.synced_head'), 'w') as f:
        f.write('scratch\n')
    return d


def _stage(d, script, profile, inv, env=None):
    e = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
    e.update(env or {})
    return subprocess.run(['bash', os.path.join(d, 'confirmatory', script), '--run', profile, inv],
                          cwd=d, env=e, capture_output=True, text=True)


def _need_r2_source():
    for it in CP.imports_of(REH):
        if not os.path.isfile(os.path.expanduser(it['src'])):
            raise AssertionError(
                'CANNOT RUN HERE -- a contract that cannot run FAILS, it never skips: the '
                'frozen R2 output %s is not on this host (the import is tested against '
                'the real files, read-only)' % it['src'])


def test_the_import_copies_only_files_with_the_pinned_digest():
    _linux_or_fail('the import step is a bash stage script')
    _need_r2_source()
    with _scratch('r3_import_') as top:
        d = _stage_tree(os.path.join(top, 'tree'))
        before = {it['src']: _sha256(os.path.expanduser(it['src'])) for it in CP.imports_of(REH)}
        r = _stage(d, 'stage_prep.sh', REH, 's0_import')
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        rec = json.load(open(os.path.join(d, CP.by_id(REH, 's0_import')['artifact'])))
        assert rec['verified'] == 5 and len(rec['items']) == 5, rec
        for it in CP.imports_of(REH):
            dest = os.path.join(d, it['dest'])
            assert _sha256(dest) == it['sha256'], it['dest']
        assert _sha256(os.path.join(d, 'output/r3lite_rehearsal/ck_s1234/policy_final.pt')) == SHA_1234
        assert {x['action'] for x in rec['items']} == {'copied'}
        # nothing was written under any other profile's directory, and the
        # frozen source is untouched
        assert sorted(os.listdir(os.path.join(d, 'output'))) == ['r3lite_rehearsal']
        assert before == {it['src']: _sha256(os.path.expanduser(it['src'])) for it in CP.imports_of(REH)}
        # a second run copies nothing and verifies everything again
        r = _stage(d, 'stage_prep.sh', REH, 's0_import')
        assert r.returncode == 0 and r.stdout.count('already present, verified') == 5, r.stdout
        # NEGATIVE CONTROL 1: a destination that is not the pinned file is
        # REFUSED and left as it is (never silently replaced)
        victim = os.path.join(d, 'output/r3lite_rehearsal/ck_s2345/policy_final.pt')
        with open(victim, 'ab') as f:
            f.write(b'x')
        tampered = _sha256(victim)
        r = _stage(d, 'stage_prep.sh', REH, 's0_import')
        assert r.returncode != 0 and 'NOT overwritten' in r.stderr + r.stdout, r.stdout + r.stderr
        assert _sha256(victim) == tampered
        # NEGATIVE CONTROL 2: a source with another digest imports NOTHING
        d2 = _stage_tree(os.path.join(top, 'tree2'))
        home = os.path.join(top, 'home')
        for it in CP.imports_of(REH):
            src = os.path.join(home, it['src'][2:])
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.copy(os.path.expanduser(it['src']), src)
        wrong = os.path.join(home, 'MARL_run_9/output/final/ck_s4567/policy_final.pt')
        with open(wrong, 'ab') as f:
            f.write(b'x')
        r = _stage(d2, 'stage_prep.sh', REH, 's0_import', env={'HOME': home})
        assert r.returncode != 0 and 'the plan pins' in r.stderr + r.stdout, r.stdout + r.stderr
        assert not os.path.exists(os.path.join(d2, 'output')), 'a refused import leaves no file'
        assert not os.path.exists(os.path.join(d2, CP.by_id(REH, 's0_import')['artifact']))
        # ... and with the source repaired the same tree imports
        shutil.copy(os.path.expanduser('~/MARL_run_9/output/final/ck_s4567/policy_final.pt'), wrong)
        r = _stage(d2, 'stage_prep.sh', REH, 's0_import', env={'HOME': home})
        assert r.returncode == 0, r.stdout + r.stderr
        # NEGATIVE CONTROL 3: a missing source
        os.remove(wrong)
        d3 = _stage_tree(os.path.join(top, 'tree3'))
        r = _stage(d3, 'stage_prep.sh', REH, 's0_import', env={'HOME': home})
        assert r.returncode != 0 and 'does not exist' in r.stderr + r.stdout
    # an R2 profile imports nothing
    assert _plan_cli('--profile', 'real', '--imports').returncode != 0
    assert len(json.loads(_plan_cli('--profile', REH, '--imports').stdout)) == 5


def test_the_selection_record_verifies_and_selects_nothing():
    _linux_or_fail('the record step is a bash stage script')
    _need_r2_source()
    with _scratch('r3_record_') as top:
        d = _stage_tree(os.path.join(top, 'tree'))
        art = os.path.join(d, CP.by_id(REH, 's3b_record')['artifact'])
        sel = os.path.join(d, CP.selected_file(REH))
        ck = os.path.join(d, CP.selected_checkpoint(REH)[0])
        # before the import there is no checkpoint: the record FAILS
        r = _stage(d, 'stage_record.sh', REH, 's3b_record')
        assert r.returncode != 0 and not os.path.exists(art), r.stdout
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        r = _stage(d, 'stage_record.sh', REH, 's3b_record')
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        rec = json.load(open(art))
        assert rec['ok'] is True and rec['n_checks'] == 7 and len(rec['checks']) == 7, rec
        assert rec['torch_seed'] == 1234 and rec['sha256_on_disk'] == rec['sha256_pinned'] == SHA_1234
        assert rec['computed'].startswith('nothing')
        assert rec['checkpoint'] == 'output/r3lite_rehearsal/ck_s1234/policy_final.pt'
        assert open(sel).readline().strip() == rec['checkpoint'], 'what run_final.sh resolves'
        good = open(sel).read()
        good_files = open(os.path.join(d, '.synced_files')).read()
        n = 0

        def must_fail(what):
            os.remove(art) if os.path.exists(art) else None
            r = _stage(d, 'stage_record.sh', REH, 's3b_record')
            assert r.returncode != 0, (what, r.stdout[-1500:])
            assert not os.path.exists(art), '%s: a failed record wrote its artefact' % what
            return r.stdout + r.stderr

        # NEGATIVE CONTROLS, each restored afterwards
        for what, text in (
                ('another digest', good.replace(SHA_1234, CP.R2_CHECKPOINT_SHA256[4567])),
                ('another replicate', good.replace('torch seed 1234', 'torch seed 4567')),
                ('another path', good.replace('ck_s1234', 'ck_s4567')),
                ('two digest lines', good + '# sha256 %s\n' % CP.R2_CHECKPOINT_SHA256[2345]),
                ('no digest line', '\n'.join(l for l in good.splitlines() if 'sha256 ' not in l) + '\n'),
                ('the r3lite path', good.replace('r3lite_rehearsal/', 'r3lite/'))):
            with open(sel, 'w') as f:
                f.write(text)
            must_fail(what)
            n += 1
        with open(sel, 'w') as f:
            f.write(good)
        with open(ck, 'ab') as f:                     # the file on disk is not the pinned one
            f.write(b'x')
        assert 'pinned sha256' in must_fail('a changed checkpoint'); n += 1
        os.remove(ck)
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        with open(os.path.join(d, '.synced_files'), 'w') as f:    # written on the server, not frozen
            f.write(''.join(l for l in good_files.splitlines(True) if 'SELECTED_' not in l))
        assert 'tracked file' in must_fail('an untracked SELECTED file'); n += 1
        os.remove(os.path.join(d, '.synced_files'))
        must_fail('no .synced_files'); n += 1
        with open(os.path.join(d, '.synced_files'), 'w') as f:
            f.write(good_files)
        assert n == 9
        r = _stage(d, 'stage_record.sh', REH, 's3b_record')
        assert r.returncode == 0, 'the restored tree records again: ' + r.stdout[-1500:]


def test_freeze_2_seals_the_tree_the_manifest_and_the_checkpoints():
    _linux_or_fail('the freeze step is a bash stage script')
    _need_r2_source()
    with _scratch('r3_freeze_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        out = os.path.join(d, CP.by_id(REH, 's5_freeze2')['artifact'])
        man = os.path.join(d, CP.by_id(REH, 's0_manifest')['artifact'])

        def manifest(seeds, kinds=('recovery', 'rescue_ops')):
            with open(man, 'w') as f:
                json.dump({'instances': [{'topology_seed': s, 'scenario_kind': k,
                                          'attainable_optimum': 1, 'severance_tick': 200}
                                         for k in kinds for s in seeds]}, f)

        def freeze():
            os.remove(out) if os.path.exists(out) else None
            return _stage(d, 'stage_record.sh', REH, 's5_freeze2')

        # nothing prepared: nothing is sealed
        r = freeze()
        assert r.returncode != 0 and not os.path.exists(out), r.stdout
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        assert _stage(d, 'stage_record.sh', REH, 's3b_record').returncode == 0
        manifest([52, 53])
        r = freeze()
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        txt = open(out).read()
        lines = dict((l.split()[0], l.split()[1:]) for l in txt.splitlines() if l.strip())
        assert txt.startswith('FREEZE_FINAL  profile=r3lite_rehearsal  sealed 20')
        # the line tools/preflight.py reads to pin the tree after freeze 2
        tree_line = [l for l in txt.splitlines() if l.startswith('tree_sha256')]
        assert len(tree_line) == 1 and len(tree_line[0].split()[1]) == 64, tree_line
        assert lines['manifest'][0] == _sha256(man)
        assert lines['selected_checkpoint'] == [SHA_1234, 'output/r3lite_rehearsal/ck_s1234/policy_final.pt']
        assert lines['imported_untrained'][0] == CP.R2_UNTRAINED_SHA256[1234]
        assert txt.count('imported_ensemble_member') == 3 and txt.count('imported_selected') == 1
        assert lines['analysis_script_md5'][0] == ANALYSIS_MD5
        assert lines['engine_head'][0] == '6b53f3f' and '(62 files identical' in txt
        assert 'freeze-2 checks verified: 11' in txt, txt
        # NEGATIVE CONTROLS: each is refused and seals nothing
        n = 0
        for what, seeds, kinds in (('a seed missing', [52], ('recovery', 'rescue_ops')),
                                   ('a foreign seed', [52, 53, 54], ('recovery', 'rescue_ops')),
                                   ('one scenario kind', [52, 53], ('recovery',))):
            manifest(seeds, kinds)
            r = freeze()
            assert r.returncode != 0 and not os.path.exists(out), what
            n += 1
        manifest([52, 53])
        ck = os.path.join(d, 'output/r3lite_rehearsal/ck_s3456/policy_final.pt')
        with open(ck, 'ab') as f:
            f.write(b'x')
        r = freeze()
        assert r.returncode != 0 and not os.path.exists(out); n += 1          # an ensemble member changed
        os.remove(ck)
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        # an analysis script that is not the pre-registered file
        link = os.path.join(d, 'confirmatory_analysis.py')
        os.remove(link)
        with open(os.path.join(ROOT, 'confirmatory_analysis.py'), 'rb') as f:
            data = f.read()
        with open(link, 'wb') as f:
            f.write(data + b'\n# edited\n')
        r = freeze()
        assert r.returncode != 0 and not os.path.exists(out); n += 1
        assert 'confirmatory_analysis.py md5' in r.stderr
        os.remove(link)
        os.symlink(os.path.join(ROOT, 'confirmatory_analysis.py'), link)
        # a selection record that did not verify
        rec = os.path.join(d, CP.by_id(REH, 's3b_record')['artifact'])
        good = open(rec).read()
        with open(rec, 'w') as f:
            f.write(good.replace('"ok": true', '"ok": false'))
        r = freeze()
        assert r.returncode != 0 and not os.path.exists(out); n += 1
        with open(rec, 'w') as f:
            f.write(good)
        assert n == 6
        assert freeze().returncode == 0, 'the restored tree seals again'


# ------------------------------------------------- the launcher, dry run only
def test_the_launcher_dry_run_walks_the_r3lite_graph_and_records_what_is_dropped():
    """run_final.sh --dry-run for the REHEARSAL profile, in a scratch copy of
    the chain (so that no ledger, env file or log of a real profile is ever
    touched): the stage order, the recorded drops, the refusal to shed, and
    the environment file of the ensemble invocation.  Nothing is launched."""
    _linux_or_fail('the launcher needs bash and /proc')
    with _scratch('r3_dry_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        env = dict(os.environ, CHAIN_NO_WATCHDOG='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

        def run(*args):
            return subprocess.run(['bash', os.path.join(d, 'confirmatory', 'run_final.sh'),
                                   '--profile', REH, '--no-watchdog', '--preflight-retries', '1',
                                   '--preflight-wait', '1'] + list(args),
                                  cwd=d, env=env, capture_output=True, text=True)
        # R3-lite sheds nothing: any --shed is refused before the chain starts
        for shed in ('untrained', 'sensitivity', 'sensitivity,adapt_freeze,untrained'):
            r = run('--dry-run', '--shed', shed)
            assert r.returncode == 2 and 'sheds nothing' in r.stderr + r.stdout, (shed, r.stdout, r.stderr)
        assert not os.path.exists(os.path.join(d, 'confirmatory', 'chain_%s.jsonl' % REH)), \
            'a refused --shed starts no chain'
        r = run('--dry-run')
        assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
        order = [l.split('STAGE', 1)[1].split()[0] for l in r.stdout.splitlines() if '---- STAGE' in l]
        assert order == CP.stage_order(REH), order
        dropped = [l.split('DROPPED', 1)[1].split()[0] for l in r.stdout.splitlines() if '] DROPPED ' in l]
        assert sorted(dropped) == sorted(CP.dropped_stages(REH)), dropped
        ledger = [json.loads(l) for l in open(os.path.join(d, 'confirmatory', 'chain_%s.jsonl' % REH))
                  if l.strip()]
        rec = {x['stage']: x['reason'] for x in ledger if x.get('event') == 'dropped'}
        assert rec == CP.dropped_stages(REH), rec
        assert not [x for x in ledger if x.get('event') == 'shed']
        dry = [l.split()[2] for l in r.stdout.splitlines() if '  DRY    ' in l and '[' in l]
        assert dry == [i['id'] for i in CP.build(REH)], dry
        assert 'S2' not in order and 'stage_train' not in r.stdout
        # the ensemble's environment: four absolute paths under the rehearsal's
        # own output directory, in replicate order
        envf = open(os.path.join(d, 'confirmatory', 'env_%s' % REH, 's9x_ensemblefull.sh')).read()
        ck = [l for l in envf.splitlines() if l.startswith('export MARL_EVAL_CHECKPOINT=')]
        assert len(ck) == 1, envf
        paths = ck[0].split('=', 1)[1].split(',')
        assert paths == ['%s/output/r3lite_rehearsal/ck_s%d/policy_final.pt' % (os.path.realpath(d), s)
                         for s in (1234, 2345, 3456, 4567)] or \
            paths == ['%s/output/r3lite_rehearsal/ck_s%d/policy_final.pt' % (d, s)
                      for s in (1234, 2345, 3456, 4567)], paths
        # the confirmatory invocation resolves its checkpoint from the SELECTED file
        e6 = open(os.path.join(d, 'confirmatory', 'env_%s' % REH, 's6_cffull_b0.sh')).read()
        assert e6.count('output/r3lite_rehearsal/ck_s1234/policy_final.pt') == 1, e6
        h = open(os.path.join(d, 'confirmatory', 'env_%s' % REH, 's6h_hash1.sh')).read()
        assert 'export PYTHONHASHSEED=1' in h.splitlines() and 'export PYTHONHASHSEED=0' in e6.splitlines()
        # nothing of the dry run names a seed of the R3 block or an r3lite path
        for dirpath, _dn, files in os.walk(os.path.join(d, 'confirmatory', 'env_%s' % REH)):
            for fn in files:
                body = open(os.path.join(dirpath, fn)).read()
                assert 'output/r3lite/' not in body, fn
        assert not os.path.exists(os.path.join(d, 'output', 'r3lite'))


def _launch(d, *args, **more_env):
    """confirmatory/run_final.sh of the scratch tree `d`, REHEARSAL profile,
    no watchdog, the launcher's own environment as the chain requires it
    (keyword arguments: further environment variables, e.g. PY=<wrapper>)."""
    env = dict(os.environ, CHAIN_NO_WATCHDOG='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env.update(more_env)
    return subprocess.run(['bash', os.path.join(d, 'confirmatory', 'run_final.sh'),
                           '--profile', REH, '--no-watchdog', '--preflight-retries', '1',
                           '--preflight-wait', '1'] + list(args),
                          cwd=d, env=env, capture_output=True, text=True)


def _ledger_of(d):
    p = os.path.join(d, 'confirmatory', 'chain_%s.jsonl' % REH)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def test_segment_1_of_the_rehearsal_runs_end_to_end_through_the_launcher():
    """The REAL launcher on the REAL stage scripts, rehearsal profile, stages
    S0, S3b and S5 only -- everything before the first evaluation pass: the
    manifest of the two DEVELOPMENT seeds (a few seconds of topology
    analysis, no arm is run), the import, the selection record and the
    freeze-2 seal, each through preflight, the clean environment, the
    freshness gate and the ledger.  Then a second run resumes all four."""
    _linux_or_fail('the launcher needs bash and /proc')
    _need_r2_source()
    with _scratch('r3_seg1_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        r = _launch(d, '--stages', 'S0,S3b,S5')
        assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
        assert 'CHAIN END profile=%s rc=0' % REH in r.stdout
        ran = [l.split('STAGE', 1)[1].split()[0] for l in r.stdout.splitlines() if '---- STAGE' in l]
        assert ran == ['S0', 'S3b', 'S5'], ran
        for st in ('S6', 'S9x', 'S7', 'S7e', 'S6h', 'S8', 'S12'):
            assert 'SKIP  %s (not in --stages' % st in r.stdout, st
        ends = [x for x in _ledger_of(d) if x.get('event') == 'end']
        assert [(x['inv'], x['status']) for x in ends] == [
            ('s0_manifest', 'ok'), ('s0_import', 'ok'), ('s3b_record', 'ok'), ('s5_freeze2', 'ok')], ends
        invs = {i['id']: i for i in CP.build(REH)}
        man = json.load(open(os.path.join(d, invs['s0_manifest']['artifact'])))
        got = sorted((i['scenario_kind'], int(i['topology_seed'])) for i in man['instances'])
        assert got == [('recovery', 52), ('recovery', 53), ('rescue_ops', 52), ('rescue_ops', 53)], got
        fz = open(os.path.join(d, invs['s5_freeze2']['artifact'])).read()
        assert 'selected_checkpoint  %s' % SHA_1234 in fz and 'freeze-2 checks verified: 11' in fz, fz
        assert ('manifest             %s' % _sha256(os.path.join(d, invs['s0_manifest']['artifact']))) in fz
        # the ledger's digest of each artefact is the file's
        for x in ends:
            assert x['sha256'] == _sha256(os.path.join(d, x['artifact'])), x['inv']
        # not one evaluation pass ran, and nothing was written for r3lite
        pk = [os.path.join(dp, f) for dp, _dn, fs in os.walk(d) for f in fs if f.endswith('.pkl')]
        assert pk == [], pk
        outs = os.listdir(os.path.join(d, 'output'))
        assert 'r3lite_rehearsal' in outs and 'r3lite' not in outs and 'final' not in outs, outs
        assert not [f for f in os.listdir(os.path.join(d, 'confirmatory'))
                    if f.startswith('r3lite_') and not f.startswith('r3lite_rehearsal_')]
        # resume: nothing is redone
        r2 = _launch(d, '--stages', 'S0,S3b,S5')
        assert r2.returncode == 0, r2.stdout[-3000:]
        assert r2.stdout.count('  RESUME ') == 4 and '  RUN    ' not in r2.stdout, r2.stdout[-3000:]
        # NEGATIVE CONTROL: a sealed tree that changes is refused at the next preflight
        with open(os.path.join(d, 'confirmatory', 'chain_plan.py'), 'a') as f:
            f.write('\n# edited after freeze 2\n')
        r3 = _launch(d, '--stages', 'S6', '--waive-dep', 'S6')
        assert r3.returncode != 0 and 'freeze_final' in r3.stdout and '  RUN    ' not in r3.stdout, \
            r3.stdout[-3000:]


_EVAL_STUB = '''#!/bin/bash
# TEST STUB (tests/test_r3lite_plan.py): stands in for confirmatory/stage_eval.sh
# in a scratch tree so that the launcher's checkpoint gates can be driven
# without running a simulation.  It never starts the harness.
if [ "${1:-}" = --print-command ]; then
  echo "bash confirmatory/stage_eval.sh --run ${2:-} ${3:-}"
  exit 0
fi
echo "[stub] no simulation in a test: ${3:-}"
exit 7
'''


def test_the_launcher_refuses_a_checkpoint_that_is_not_the_pinned_one():
    """The digest stop of run_final.sh, driven for real.  The evaluation
    script of the scratch tree is a stub that exits 7 without starting the
    harness, so the PASSING side of the pin can be observed too (the launcher
    goes on to launch, the stub fails, the restart rule fires) -- and no
    simulation runs on any seed."""
    _linux_or_fail('the launcher needs bash and /proc')
    _need_r2_source()
    with _scratch('r3_pin_launch_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        with open(os.path.join(d, 'confirmatory', 'stage_eval.sh'), 'w') as f:
            f.write(_EVAL_STUB)
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        ck = {s: os.path.join(d, 'output', 'r3lite_rehearsal', 'ck_s%d' % s, 'policy_final.pt')
              for s in (1234, 2345, 3456, 4567)}
        un = os.path.join(d, 'output', 'r3lite_rehearsal', 'untrained_s1234', 'policy_init.pt')

        def tamper(path):
            with open(path, 'ab') as f:
                f.write(b'x')

        def restore(path, seed, name='ck_s%d/policy_final.pt'):
            shutil.copy(os.path.join(R2_SOURCE, name % seed), path)

        def stops(reason):
            return [x for x in _ledger_of(d) if x.get('event') == 'stop' and x.get('reason') == reason]

        n = 0
        # --- the selected checkpoint (S6): one changed byte -> STOP before any launch
        tamper(ck[1234])
        r = _launch(d, '--stages', 'S6', '--waive-dep', 'S6')
        assert r.returncode != 0 and 'checkpoint digest is not the pinned one' in r.stdout, r.stdout[-4000:]
        assert '  RUN    ' not in r.stdout and 'PIN    ' not in r.stdout, r.stdout[-3000:]
        s = stops('checkpoint_sha256_mismatch')
        assert len(s) == 1 and s[0]['inv'] == 's6_cf_b0' and s[0]['pinned'] == SHA_1234 \
            and s[0]['disk'] == _sha256(ck[1234]) != SHA_1234, s
        assert not [x for x in _ledger_of(d) if x.get('event') == 'start'], 'nothing was launched'
        n += 1
        # ... restored: the pin passes, the launcher launches, the stub fails,
        # one unchanged re-run, then the chain stops
        restore(ck[1234], 1234)
        r = _launch(d, '--stages', 'S6', '--waive-dep', 'S6')
        assert r.returncode != 0, r.stdout[-3000:]
        assert "PIN    s6_cf_b0 checkpoint sha256 matches the plan's pin (1 file(s))" in r.stdout, r.stdout[-4000:]
        assert r.stdout.count('  RUN    s6_cf_b0') == 2 and 'RERUN  s6_cf_b0 UNCHANGED' in r.stdout
        assert 'STOP   s6_cf_b0 failed twice' in r.stdout
        starts = [x for x in _ledger_of(d) if x.get('event') == 'start']
        assert len(starts) == 2 and all(x['checkpoint'] == 'output/r3lite_rehearsal/ck_s1234/policy_final.pt'
                                        for x in starts), starts
        assert [x['exit'] for x in _ledger_of(d) if x.get('event') == 'end'] == [7, 7]
        n += 1
        # --- the ensemble (S9x): ONE changed member of four -> STOP
        tamper(ck[3456])
        r = _launch(d, '--stages', 'S9x', '--waive-dep', 'S9x', '--waive-order', 'S9x')
        assert r.returncode != 0 and 'checkpoint digest is not the pinned one' in r.stdout, r.stdout[-4000:]
        s = stops('checkpoint_sha256_mismatch')
        assert len(s) == 2 and s[1]['inv'] == 's9x_ensemble', s
        pinned, disk = s[1]['pinned'].split(','), s[1]['disk'].split(',')
        assert pinned == CP.by_id(REH, 's9x_ensemble')['ckpt_sha256'] and len(disk) == 4
        assert [a == b for a, b in zip(pinned, disk)] == [True, True, False, True], (pinned, disk)
        n += 1
        restore(ck[3456], 3456)
        r = _launch(d, '--stages', 'S9x', '--waive-dep', 'S9x', '--waive-order', 'S9x')
        assert "PIN    s9x_ensemble checkpoint sha256 matches the plan's pin (4 file(s))" in r.stdout, \
            r.stdout[-4000:]
        envf = open(os.path.join(d, 'confirmatory', 'env_%s' % REH, 's9x_ensemble.sh')).read()
        assert envf.count('policy_final.pt') == 4 and envf.count('MARL_EVAL_CHECKPOINT=') == 1
        n += 1
        # ... a MISSING member is a stop of its own
        os.remove(ck[4567])
        assert _launch(d, '--invalidate', 's9x_ensemble').returncode == 0
        r = _launch(d, '--stages', 'S9x', '--waive-dep', 'S9x', '--waive-order', 'S9x')
        assert r.returncode != 0 and 'ck_s4567/policy_final.pt does not exist' in r.stdout, r.stdout[-3000:]
        assert len(stops('missing_checkpoint')) == 1
        restore(ck[4567], 4567)
        n += 1
        # --- the untrained control (S8): a literal checkpoint is pinned too
        tamper(un)
        r = _launch(d, '--stages', 'S8', '--waive-dep', 'S8', '--waive-order', 'S8')
        assert r.returncode != 0 and 'checkpoint digest is not the pinned one' in r.stdout, r.stdout[-3000:]
        s = stops('checkpoint_sha256_mismatch')
        assert len(s) == 3 and s[2]['inv'] == 's8_untrained_s1234' \
            and s[2]['pinned'] == CP.R2_UNTRAINED_SHA256[1234], s
        n += 1
        assert n == 6
        # no simulation ran: the only job output is the stub's line
        logs = os.path.join(d, 'logs', 'chain_%s' % REH)
        body = ''.join(open(os.path.join(logs, f)).read() for f in os.listdir(logs) if f.endswith('.log'))
        assert '[stub] no simulation in a test' in body and 'run_timeline_comparison' not in body
        assert not [f for dp, _dn, fs in os.walk(d) for f in fs if f.endswith('.pkl')]


def _ledger_lines(*recs):
    return ''.join(json.dumps(dict(r, utc='T%04d' % k), sort_keys=True) + '\n' for k, r in enumerate(recs))


def _E(inv, status='ok', attempt=1, wd=0, sha='a' * 64):
    """(start, end) records of one attempt, as run_final.sh writes them."""
    return ({'event': 'start', 'inv': inv, 'attempt': attempt},
            {'event': 'end', 'inv': inv, 'attempt': attempt, 'status': status,
             'exit': 0 if status == 'ok' else 7, 'gate': 0, 'sha256': sha if status == 'ok' else '-',
             'artifact': CP.by_id(REH, inv)['artifact'] if inv in {i['id'] for i in CP.build(REH)} else '',
             'watchdog_restarts': wd})


def test_a_terminal_failure_is_an_exhausted_restart_rule_and_nothing_else():
    """tools/stage_report.py --terminal-failure: the ONLY thing that releases
    a failure-only waiver and the ONLY thing that makes S12 record family E
    as not completed.  It is read from the END records -- two failed
    attempts, or one that contains the watchdog's re-run -- and from nothing
    else: not from a `stop` line, not from one failure, not from a failure
    that was invalidated or followed by a success."""
    stop = {'event': 'stop', 'inv': 's9x_ensemble', 'reason': 'failed_twice'}
    inval = {'event': 'invalidate', 'inv': 's9x_ensemble', 'reason': 'operator'}
    F1, F2 = _E('s9x_ensemble', 'fail'), _E('s9x_ensemble', 'fail', attempt=2)
    OK = _E('s9x_ensemble')
    cases = [
        # (name, records, stage, expected exit, needle)
        ('an empty ledger', (), 'S9x', 1, 'NOT_TERMINAL S9x'),
        ('never attempted', _E('s6_cf_b0'), 'S9x', 1, 'NOT_TERMINAL'),
        ('one failed attempt', F1, 'S9x', 1, 'NOT_TERMINAL'),
        ('a stop line and no failed end', (stop, stop), 'S9x', 1, 'NOT_TERMINAL'),
        ('one failed attempt and a stop line', F1 + (stop,), 'S9x', 1, 'NOT_TERMINAL'),
        ('two failed attempts', F1 + F2, 'S9x', 0,
         'TERMINAL S9x s9x_ensemble: failed twice (attempt 1 exit 7 gate 0; attempt 2 exit 7 gate 0)'),
        ('one failed attempt that holds the watchdog re-run', _E('s9x_ensemble', 'fail', wd=1), 'S9x', 0,
         "TERMINAL S9x s9x_ensemble: failed after the watchdog's one unchanged re-run"),
        ('failed, then succeeded', F1 + OK, 'S9x', 1, 'NOT_TERMINAL'),
        # (not reachable through the launcher, which never re-runs a success: the rule needs
        # TWO attempts that did not succeed, not two attempts of which the last failed)
        ('succeeded, then failed', OK + _E('s9x_ensemble', 'fail', attempt=2), 'S9x', 1, 'NOT_TERMINAL'),
        # AN ATTEMPT IS A START: one that never ended (the launcher or the host died) has run
        ('a start that never ended, then a failed attempt', (F1[0],) + F2, 'S9x', 0,
         'TERMINAL S9x s9x_ensemble: failed twice (attempt 1 did not end; attempt 2 exit 7 gate 0)'),
        # ... but an attempt that is still OPEN is never terminal: its job may be running
        ('a failed attempt, then a start that has not ended', F1 + (F2[0],), 'S9x', 1, 'NOT_TERMINAL'),
        ('one start that has not ended', (F1[0],), 'S9x', 1, 'NOT_TERMINAL'),
        ('two starts and no end', (F1[0], F2[0]), 'S9x', 1, 'NOT_TERMINAL'),
        ('failed twice, invalidated', F1 + F2 + (inval,), 'S9x', 1, 'NOT_TERMINAL'),
        ('failed twice, invalidated, failed once', F1 + F2 + (inval,) + F1, 'S9x', 1, 'NOT_TERMINAL'),
        ('failed twice, invalidated, failed twice', F1 + F2 + (inval,) + F1 + F2, 'S9x', 0, 'failed twice'),
        ('the SECOND invocation of the stage failed twice',
         OK + _E('s9x_ensemblefull', 'fail') + _E('s9x_ensemblefull', 'fail', attempt=2), 'S9x', 0,
         'TERMINAL S9x s9x_ensemblefull: failed twice'),
        ('another stage failed twice', _E('s8_untrained_s1234', 'fail') + _E('s8_untrained_s1234', 'fail', 2),
         'S9x', 1, 'NOT_TERMINAL'),
        ('the analysis of family E failed twice', _E('s7e_family_e', 'fail') + _E('s7e_family_e', 'fail', 2),
         'S7e', 0, 'TERMINAL S7e s7e_family_e: failed twice'),
        ('a stage the profile does not have', F1 + F2, 'S2', 2, 'has no invocation of that stage'),
        ('no stage named', F1 + F2, '', 2, 'no --stage given'),
    ]
    n = 0
    with _scratch('r3_terminal_') as d:
        led = os.path.join(d, 'ledger.jsonl')
        for name, recs, stage, rc, needle in cases:
            with open(led, 'w') as f:
                f.write(_ledger_lines(*recs))
            r = _stage_report('--terminal-failure', led, '--profile', REH, '--stage', stage)
            assert r.returncode == rc, (name, r.stdout, r.stderr)
            assert needle in r.stdout and len(r.stdout.strip().splitlines()) == 1, (name, r.stdout)
            assert (r.returncode == 0) == r.stdout.startswith('TERMINAL '), name
            n += 1
        # a ledger that does not exist is an empty ledger
        r = _stage_report('--terminal-failure', os.path.join(d, 'none.jsonl'), '--profile', REH, '--stage', 'S9x')
        assert r.returncode == 1 and r.stdout.startswith('NOT_TERMINAL'); n += 1
        assert n == 22
        # THE LAUNCHER READS THE SAME STATE (--restart-state): attempts STARTED, how the last
        # one ended, and whether the rule is exhausted -- one line, and TERMINAL here is
        # TERMINAL there, case by case
        m = 0
        for name, recs, line in (
                ('never started', (), '0 none 0 -'),
                ('one failed attempt', F1, '1 fail 0 -'),
                ('one open attempt', (F1[0],), '1 open 0 -'),
                ('one success', OK, '1 ok 0 -'),
                ('failed, then succeeded', F1 + _E('s9x_ensemble', attempt=2), '2 ok 0 -'),
                ('failed twice', F1 + F2,
                 '2 fail 1 failed twice (attempt 1 exit 7 gate 0; attempt 2 exit 7 gate 0)'),
                ('failed, then open', F1 + (F2[0],), '2 open 0 -'),
                ('never ended, then failed', (F1[0],) + F2,
                 '2 fail 1 failed twice (attempt 1 did not end; attempt 2 exit 7 gate 0)'),
                ('the watchdog re-run inside one failed attempt', _E('s9x_ensemble', 'fail', wd=1),
                 "1 fail 1 failed after the watchdog's one unchanged re-run (attempt 1 exit 7 gate 0)"),
                ('failed twice, invalidated', F1 + F2 + (inval,), '0 none 0 -'),
                ('failed twice, invalidated, one open attempt', F1 + F2 + (inval,) + (F1[0],), '1 open 0 -')):
            with open(led, 'w') as f:
                f.write(_ledger_lines(*recs))
            r = _stage_report('--restart-state', led, '--inv', 's9x_ensemble')
            assert r.returncode == 0 and r.stdout.strip() == line, (name, r.stdout, r.stderr)
            t = _stage_report('--terminal-failure', led, '--profile', REH, '--stage', 'S9x')
            assert (t.returncode == 0) == (line.split()[2] == '1'), (name, t.stdout)
            if t.returncode == 0:
                assert t.stdout.strip() == 'TERMINAL S9x s9x_ensemble: ' + line.split(' ', 3)[3], name
            m += 1
        assert m == 11
        assert _stage_report('--restart-state', led).returncode == 2, 'no invocation named'
        # --stage-started: has ANY invocation of that stage or a later one been started?
        for name, recs, st, rc, out in (
                ('nothing', (), 'S6', 1, 'NOT_STARTED'),
                ('only stages before it', _E('s0_manifest') + _E('s5_freeze2'), 'S6', 1, 'NOT_STARTED'),
                ('an S6 invocation started and still open', _E('s5_freeze2') + (_E('s6_cf_b0')[0],), 'S6', 0,
                 'STARTED s6_cf_b0'),
                ('an S6 invocation ended', _E('s6_cffull_b0'), 'S6', 0, 'STARTED s6_cffull_b0'),
                ('a later stage', _E('s8_untrained_s1234', 'fail'), 'S6', 0, 'STARTED s8_untrained_s1234'),
                ('a stage of another graph', _E('s6_cf_b0'), 'S2', 2, 'ERROR')):
            with open(led, 'w') as f:
                f.write(_ledger_lines(*recs))
            r = _stage_report('--stage-started', led, '--profile', REH, '--from-stage', st)
            assert r.returncode == rc and r.stdout.startswith(out), (name, r.stdout, r.stderr)
            m += 1
        assert m == 17


def test_the_ledger_order_is_read_from_the_records_and_binds_the_report():
    """tools/stage_report.py --ledger-order (S7e: every pass of family E
    ended before the analysis of family 1 started, and the ledger holds THIS
    family-1 report) and --ledger-artifacts (S12: the pickles family E was
    assembled from are the ones the ledger recorded)."""
    A, A2, B = _E('s9x_ensemble', sha='1' * 64), _E('s9x_ensemblefull', sha='2' * 64), \
        _E('s7_analysis', sha='3' * 64)
    FIRST = 's9x_ensemble,s9x_ensemblefull'
    cases = [
        ('both ended before S7 started', A + A2 + B, FIRST, '3' * 64, 0, 'PASS'),
        ('no digest asked', A + A2 + B, FIRST, '', 0, 'PASS'),
        ('S7 first', B + A + A2, FIRST, '3' * 64, 1, 'AFTER s7_analysis was first started'),
        ('one S9x invocation after S7', A + B + A2, FIRST, '3' * 64, 1,
         's9x_ensemblefull ended (record 5'),
        ('S9x ended while S7 was running', (A[0], A[1], A2[0], B[0], A2[1], B[1]), FIRST, '3' * 64, 1,
         'AFTER s7_analysis was first started'),
        ('S9x not recorded', B, FIRST, '3' * 64, 1, 's9x_ensemble is not recorded successful'),
        ('S9x failed', _E('s9x_ensemble', 'fail') + A2 + B, FIRST, '3' * 64, 1,
         's9x_ensemble is not recorded successful'),
        ('S7 failed', A + A2 + _E('s7_analysis', 'fail'), FIRST, '3' * 64, 1,
         's7_analysis is not recorded successful'),
        ('another family-1 report', A + A2 + B, FIRST, '4' * 64, 1, "the ledger's sha256 of s7_analysis"),
        # an attempt of S7 that FAILED has still run (and may have printed what the successful
        # one did): the FIRST start of S7 is what every S9x end must precede
        ('S7 failed before S9x ended, succeeded after', _E('s7_analysis', 'fail') + A + A2
         + _E('s7_analysis', attempt=2, sha='3' * 64), FIRST, '3' * 64, 1,
         'AFTER s7_analysis was first started (record 0'),
        ('S7 failed after S9x ended, then succeeded', A + A2 + _E('s7_analysis', 'fail')
         + _E('s7_analysis', attempt=2, sha='3' * 64), FIRST, '3' * 64, 0, 'PASS'),
        ('an empty ledger', (), FIRST, '', 1, 'holds no record'),
        ('nothing to order', A + B, '', '', 1, 'nothing to order'),
        # a start of S7 that never ended, before S9x ended: it ran all the same
        ('S7 started and never ended before S9x ended, then succeeded', (B[0],) + A + A2
         + _E('s7_analysis', attempt=2, sha='3' * 64), FIRST, '3' * 64, 1,
         'AFTER s7_analysis was first started (record 0'),
    ]
    n = 0
    with _scratch('r3_order_') as d:
        led, out = os.path.join(d, 'ledger.jsonl'), os.path.join(d, 'order.json')
        for name, recs, first, sha, rc, needle in cases:
            with open(led, 'w') as f:
                f.write(_ledger_lines(*recs))
            if os.path.exists(out):
                os.remove(out)
            args = ['--ledger-order', led, '--first', first, '--then', 's7_analysis', '--json-out', out]
            r = _stage_report(*(args + (['--expect-then-sha256', sha] if sha else [])))
            assert r.returncode == rc and needle in r.stdout, (name, r.stdout, r.stderr)
            j = json.load(open(out))
            assert j['ok'] is (rc == 0) and bool(j['problems']) == (rc != 0), (name, j)
            if rc == 0:
                assert j['first_end_before_then_start'] is True and j['then_span']['sha256'] == '3' * 64
                assert [s['inv'] for s in j['first_span']] == first.split(',')
                assert all(s['end_record'] < j['then_span']['first_start_record']
                           <= j['then_span']['start_record'] for s in j['first_span'])
            n += 1
        assert n == 14
        # --ledger-artifacts: what the ledger recorded for every pickle of the named stages
        ids = [i['id'] for i in CP.build(REH) if i['stage'] in ('S6', 'S9x')]
        assert ids == ['s6_cf_b0', 's6_cffull_b0', 's9x_ensemble', 's9x_ensemblefull'], ids
        recs = ()
        for k, inv in enumerate(ids):
            recs += _E(inv, sha=str(k) * 64)
        with open(led, 'w') as f:
            f.write(_ledger_lines(*recs))
        r = _stage_report('--ledger-artifacts', led, '--profile', REH, '--stage', 'S6,S9x', '--json-out', out)
        assert r.returncode == 0 and '[LEDGER-ARTIFACTS] PASS 4 pickle(s)' in r.stdout, r.stdout
        j = json.load(open(out))
        assert {v['inv']: v['sha256'] for v in j.values()} == {inv: str(k) * 64 for k, inv in enumerate(ids)}
        assert sorted(j) == sorted(CP.by_id(REH, i)['artifact'] for i in ids)
        assert {v['status'] for v in j.values()} == {'SUCCESS'}
        # one of them failed, or was never run: exit 1, and the entry says so
        with open(led, 'w') as f:
            f.write(_ledger_lines(*(recs[:6] + _E(ids[3], 'fail'))))
        r = _stage_report('--ledger-artifacts', led, '--profile', REH, '--stage', 'S6,S9x', '--json-out', out)
        j = json.load(open(out))
        assert r.returncode == 1 and '1 not recorded successful' in r.stdout
        assert j[CP.by_id(REH, ids[3])['artifact']] == {'inv': ids[3], 'stage': 'S9x',
                                                        'status': 'NOT_SUCCESSFUL', 'sha256': None}
        r = _stage_report('--ledger-artifacts', led, '--profile', REH, '--stage', 'S2', '--json-out', out)
        assert r.returncode == 1 and 'FAIL 0 pickle(s)' in r.stdout, 'no pickle is not a pass'


def test_the_launcher_honours_the_two_order_waivers_only_after_a_terminal_failure():
    """The registered contingency ('if family E cannot be completed'), driven
    through the REAL launcher in a scratch chain of the rehearsal profile.
    The evaluation script is the stub that exits 7 without starting the
    harness, so S9x really fails and the ledger lines of its attempts are
    the launcher's own; the stages before it and the stages that would need
    pickles are recorded complete by hand (the scratch tree is not sealed:
    a sealed tree could not carry the stub), and nothing is simulated on any
    seed.

      * while S9x has not failed for good -- never attempted, or failed once,
        or stopped for another reason -- S7 does not start, waiver or not;
      * after the unchanged re-run has failed too, the waiver is honoured
        and recorded with the ledger's cause;
      * S7e can never start without S9x (a hard dependency);
      * S12 follows S7e by order, under the same failure-only rule."""
    _linux_or_fail('the launcher needs bash and /proc')
    _need_r2_source()
    with _scratch('r3_waive_launch_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        with open(os.path.join(d, 'confirmatory', 'stage_eval.sh'), 'w') as f:
            f.write(_EVAL_STUB)
        # the five pinned checkpoints, imported for real (the ensemble's pin is checked)
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        ledger = os.path.join(d, 'confirmatory', 'chain_%s.jsonl' % REH)
        invs = {i['id']: i for i in CP.build(REH)}

        def record(inv, status='ok', attempt=1):
            with open(ledger, 'a') as f:
                for x in _E(inv, status, attempt):
                    f.write(json.dumps(dict(x, utc='T', stage=invs[inv]['stage'])) + '\n')

        def record_stage(st):
            for i in invs.values():
                if i['stage'] == st:
                    record(i['id'])

        def of(event, **kv):
            return [x for x in _ledger_of(d) if x.get('event') == event
                    and all(x.get(k) == v for k, v in kv.items())]
        n = 0
        for st in ('S0', 'S3b', 'S5', 'S6'):
            record_stage(st)
        # --- 1. S9x NEVER ATTEMPTED: the state in which family 1 could be analysed and read
        #        before anyone decides whether to run family E.  Refused, waiver or not.
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and 'STOP  S7 must run after S9x' in r.stdout, r.stdout[-3000:]
        assert 'S9x is not complete and has not failed terminally: it has to run first' in r.stdout
        r = _launch(d, '--stages', 'S7', '--waive-order', 'S7')
        assert r.returncode != 0 and '--waive-order S7 is REFUSED on the edge after S9x' in r.stdout, r.stdout[-3000:]
        s = of('stop', reason='order_waiver_refused')
        assert len(s) == 1 and s[0]['stage'] == 'S7' and s[0]['after'] == 'S9x' and s[0]['needs'] == 'S9x', s
        assert not of('start', inv='s7_analysis') and not of('note', reason='order_waived')
        n += 1
        # --- 2. ONE failed attempt and a `stop` line: the restart rule is not exhausted
        record('s9x_ensemble', 'fail')
        with open(ledger, 'a') as f:
            f.write(json.dumps({'utc': 'T', 'event': 'stop', 'inv': 's9x_ensemble', 'stage': 'S9x',
                                'reason': 'failed_twice'}) + '\n')
        r = _launch(d, '--stages', 'S7', '--waive-order', 'S7')
        assert r.returncode != 0 and '--waive-order S7 is REFUSED on the edge after S9x' in r.stdout
        assert len(of('stop', reason='order_waiver_refused')) == 2 and not of('start', inv='s7_analysis')
        n += 1
        # --- 3. THE ONE RE-RUN THAT IS LEFT, for real (the stub), through the launcher.  The
        #        ledger already holds one failed attempt of this invocation, so the re-issued
        #        launch runs it exactly ONCE more, as attempt 2 -- not a fresh pair
        r = _launch(d, '--stages', 'S9x')
        assert r.returncode != 0 and 'STOP   s9x_ensemble failed twice' in r.stdout, r.stdout[-4000:]
        assert r.stdout.count('  RUN    s9x_ensemble') == 1 \
            and 'RUN    s9x_ensemble [S9x/eval] attempt=2' in r.stdout, r.stdout[-4000:]
        assert 'RERUN  s9x_ensemble UNCHANGED' not in r.stdout, 'the re-run was this launch itself'
        ends = of('end', inv='s9x_ensemble')
        assert [x['status'] for x in ends] == ['fail', 'fail'] and [x['attempt'] for x in ends] == [1, 2]
        assert ends[1]['exit'] == 7 and [x['attempt'] for x in of('start', inv='s9x_ensemble')] == [1, 2]
        # ... and a further launch does not run it a third time
        r = _launch(d, '--stages', 'S9x')
        assert r.returncode != 0 and 'has already failed 2 time(s)' in r.stdout and '  RUN    ' not in r.stdout
        assert len(of('end', inv='s9x_ensemble')) == 2 and len(of('stop', reason='restart_budget_used')) == 1
        # ... while a FRESH invocation still gets its attempt and its one unchanged re-run in
        #     one launch (the operator invalidates: the rehearsal profile allows it)
        assert _launch(d, '--invalidate', 's9x_ensemble').returncode == 0
        r = _launch(d, '--stages', 'S9x')
        assert r.stdout.count('  RUN    s9x_ensemble') == 2 and 'RERUN  s9x_ensemble UNCHANGED' in r.stdout
        assert 'STOP   s9x_ensemble failed twice' in r.stdout
        ends = of('end', inv='s9x_ensemble')[-2:]
        assert [x['status'] for x in ends] == ['fail', 'fail'] and [x['attempt'] for x in ends] == [1, 2]
        # ... now the launcher SAYS what the ledger records, and still does not go on unasked
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and 'STOP  S7 must run after S9x' in r.stdout
        assert 'The ledger records a terminal failure -- TERMINAL S9x s9x_ensemble: failed twice' in r.stdout
        assert 'To go on without S9x: --waive-order S7' in r.stdout and not of('start', inv='s7_analysis')
        n += 1
        # ... and with the recorded waiver S7 starts (it then fails: there is no pickle to analyse)
        r = _launch(d, '--stages', 'S7', '--waive-order', 'S7')
        assert 'WAIVE S7 ordering after S9x (operator waiver, recorded; honoured ONLY because the' in r.stdout, \
            r.stdout[-4000:]
        assert '---- STAGE S7 ----' in r.stdout and of('start', inv='s7_analysis')
        w = of('note', reason='order_waived')
        assert len(w) == 1 and w[0]['stage'] == 'S7' and w[0]['after'] == 'S9x' \
            and w[0]['rule'] == 'failure_only' \
            and w[0]['cause'].startswith('TERMINAL S9x s9x_ensemble: failed twice (attempt 1 exit 7'), w
        n += 1
        # --- 4. S7e: a HARD dependency on S9x; no waiver of the order reaches it
        record('s7_analysis')
        r = _launch(d, '--stages', 'S7e', '--waive-order', 'S7e')
        assert r.returncode != 0 and 'STOP  S7e: hard dependency S9x is not complete' in r.stdout, r.stdout[-3000:]
        assert not of('start', inv='s7e_family_e')
        n += 1
        # --- 5. the secondary stages of family 1 follow S7 only ...
        r = _launch(d, '--dry-run', '--stages', 'S6h,S8')
        assert r.returncode == 0 and 'PLAN  S6h deps=[S5 S3b] order_after=[S7]' in r.stdout \
            and 'PLAN  S8 deps=[S5 S0] order_after=[S7]' in r.stdout, r.stdout[-3000:]
        record_stage('S6h'); record_stage('S8')
        # ... and S12 follows S7e by ORDER, under the same failure-only rule: here S7e can
        # never run, because S9x failed for good
        r = _launch(d, '--stages', 'S12')
        assert r.returncode != 0 and 'STOP  S12 must run after S7e' in r.stdout, r.stdout[-3000:]
        assert 'TERMINAL S9x s9x_ensemble: failed twice' in r.stdout and not of('start', inv='s12_secondary')
        r = _launch(d, '--stages', 'S12', '--waive-order', 'S12')
        assert 'WAIVE S12 ordering after S7e (operator waiver, recorded; honoured ONLY because the' in r.stdout, \
            r.stdout[-4000:]
        assert '---- STAGE S12 ----' in r.stdout and of('start', inv='s12_secondary')
        w = of('note', reason='order_waived', stage='S12')
        assert len(w) == 1 and w[0]['after'] == 'S7e' and w[0]['cause'].startswith('TERMINAL S9x'), w
        n += 1
        assert n == 6
        # --- 6. FAIL CLOSED: if the plan cannot say how a waiver of the edge is treated (the
        #        call fails, or prints nothing), nothing is waived.  An empty answer used to
        #        fall through to the unrestricted arm.
        wrapper = os.path.join(top, 'py_no_order_waiver')
        with open(wrapper, 'w') as f:
            f.write('#!/bin/bash\nfor a in "$@"; do [ "$a" = --order-waiver ] && exit 1; done\n'
                    'exec %s "$@"\n' % sys.executable)
        os.chmod(wrapper, 0o755)
        before = len(of('start', inv='s12_secondary')), len(of('note', reason='order_waived'))
        r = _launch(d, '--stages', 'S12', '--waive-order', 'S12', PY=wrapper)
        assert r.returncode != 0 and 'the plan did not say how a waiver of the edge after S7e is treated' in r.stdout, \
            r.stdout[-3000:]
        assert len(of('stop', reason='order_waiver_unreadable')) == 1
        assert (len(of('start', inv='s12_secondary')), len(of('note', reason='order_waived'))) == before
        n += 1
        assert n == 7
        # no simulation ran: the only job output of an evaluation is the stub's line
        logs = os.path.join(d, 'logs', 'chain_%s' % REH)
        body = ''.join(open(os.path.join(logs, f)).read() for f in os.listdir(logs) if f.endswith('.log'))
        assert '[stub] no simulation in a test' in body and 'run_timeline_comparison' not in body
        assert not [f for dp, _dn, fs in os.walk(d) for f in fs if f.endswith('.pkl')]


def test_the_launcher_never_starts_again_what_the_ledger_calls_terminal():
    """ONE source for the restart rule (tools/stage_report.py restart_state):
    what releases a waiver and makes S12 record a family as not completed is
    what stops the launcher.  Driven through the real launcher on a scratch
    chain of the rehearsal profile with the evaluation stub; nothing is
    simulated.

      * one failed attempt that holds the watchdog's re-run is TERMINAL, and
        the launcher does not start the invocation again (it used to: the
        budget counted end records only);
      * an attempt that has a start and no end -- the launcher or the host
        died -- has run: the re-issued launch gives the ONE attempt left;
      * a failed attempt followed by an open one has used the budget: the
        launcher closes the open attempt as failed, once the preflight shows
        no job of ours alive, and stops; the ledger then says TERMINAL;
      * --max-attempts is a positive integer, for every profile."""
    _linux_or_fail('the launcher needs bash and /proc')
    _need_r2_source()
    with _scratch('r3_terminal_launch_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        with open(os.path.join(d, 'confirmatory', 'stage_eval.sh'), 'w') as f:
            f.write(_EVAL_STUB)
        assert _stage(d, 'stage_prep.sh', REH, 's0_import').returncode == 0
        ledger = os.path.join(d, 'confirmatory', 'chain_%s.jsonl' % REH)
        invs = {i['id']: i for i in CP.build(REH)}

        def write(*recs):
            with open(ledger, 'a') as f:
                for x in recs:
                    f.write(json.dumps(dict(x, utc='T', stage=invs[x['inv']]['stage'])) + '\n')

        def of(event, **kv):
            return [x for x in _ledger_of(d) if x.get('event') == event
                    and all(x.get(k) == v for k, v in kv.items())]

        def state(inv='s9x_ensemble'):
            return _stage_report('--restart-state', ledger, '--inv', inv).stdout.strip()

        def terminal():
            return _stage_report('--terminal-failure', ledger, '--profile', REH, '--stage', 'S9x')
        for st in ('S0', 'S3b', 'S5', 'S6'):
            for i in invs.values():
                if i['stage'] == st:
                    write(*_E(i['id']))
        n = 0
        # --- 1. the watchdog path: ONE failed end, with the watchdog's re-run inside it
        write(*_E('s9x_ensemble', 'fail', wd=1))
        assert terminal().returncode == 0 and state().startswith('1 fail 1 failed after the watchdog')
        r = _launch(d, '--stages', 'S9x')
        assert r.returncode != 0 and '  RUN    ' not in r.stdout, r.stdout[-3000:]
        assert 'STOP   s9x_ensemble has already failed 1 time(s)' in r.stdout
        assert "The ledger: failed after the watchdog's one unchanged re-run" in r.stdout
        s = of('stop', reason='restart_budget_used')
        assert len(s) == 1 and s[0]['inv'] == 's9x_ensemble' and s[0]['terminal'] == 1 and s[0]['attempts'] == 1, s
        assert len(of('start', inv='s9x_ensemble')) == 1, 'nothing was started'
        n += 1
        # --- 2. a START WITHOUT AN END is an attempt: the re-issued launch runs it once more
        assert _launch(d, '--invalidate', 's9x_ensemble').returncode == 0
        write(_E('s9x_ensemble')[0])
        assert state() == '1 open 0 -' and terminal().returncode == 1
        r = _launch(d, '--stages', 'S9x')
        assert r.stdout.count('  RUN    s9x_ensemble') == 1 \
            and 'RUN    s9x_ensemble [S9x/eval] attempt=2' in r.stdout, r.stdout[-4000:]
        assert 'STOP   s9x_ensemble failed twice' in r.stdout and 'RERUN  s9x_ensemble' not in r.stdout
        assert state() == '2 fail 1 failed twice (attempt 1 did not end; attempt 2 exit 7 gate 0)', state()
        assert terminal().returncode == 0
        r = _launch(d, '--stages', 'S9x')
        assert '  RUN    ' not in r.stdout and 'has already failed 2 time(s)' in r.stdout
        n += 1
        # --- 3. a failed attempt, then an attempt that never ended: the budget is used, and the
        #        launcher CLOSES the open attempt (no job of ours is alive) so that the ledger
        #        can say what happened
        assert _launch(d, '--invalidate', 's9x_ensemble').returncode == 0
        write(*_E('s9x_ensemble', 'fail'))
        write(_E('s9x_ensemble', attempt=2)[0])
        assert state() == '2 open 0 -' and terminal().returncode == 1, 'an open attempt is never terminal'
        # ... but NOT while a process still runs that invocation's job script: nothing is
        # closed, nothing is started (a stand-in process carries the job script's path)
        jobs = [os.path.join(x, 'logs', 'chain_%s' % REH, 's9x_ensemble.sh') for x in (os.path.realpath(d), d)]
        alive = subprocess.Popen(['bash', '-c', 'sleep 120; true'] + jobs)
        try:
            r = _launch(d, '--stages', 'S9x')
            assert r.returncode != 0 and 'CLOSE  ' not in r.stdout and '  RUN    ' not in r.stdout, r.stdout[-3000:]
            assert 'attempt 2 has a start and no end, and a process still runs its job' in r.stdout
            s = of('stop', reason='attempt_open')
            assert len(s) == 1 and s[0]['job'] == 'alive' and state() == '2 open 0 -', s
        finally:
            alive.kill()
            alive.wait()
        r = _launch(d, '--stages', 'S9x')
        assert r.returncode != 0 and '  RUN    ' not in r.stdout, r.stdout[-4000:]
        assert 'CLOSE  s9x_ensemble attempt 2 has a start and no end in the ledger' in r.stdout
        assert 'STOP   s9x_ensemble has already failed 2 time(s)' in r.stdout
        closed = [x for x in of('end', inv='s9x_ensemble') if x.get('closed_by') == 'launcher']
        assert len(closed) == 1 and closed[0]['status'] == 'fail' and closed[0]['exit'] == 'killed' \
            and closed[0]['attempt'] == 2, closed
        assert state() == '2 fail 1 failed twice (attempt 1 exit 7 gate 0; attempt 2 exit killed gate -)', state()
        assert terminal().returncode == 0
        # ... a dry run closes nothing
        assert _launch(d, '--invalidate', 's9x_ensemble').returncode == 0
        write(*_E('s9x_ensemble', 'fail'))
        write(_E('s9x_ensemble', attempt=2)[0])
        k = len(of('end', inv='s9x_ensemble'))
        r = _launch(d, '--dry-run', '--stages', 'S9x')
        assert len(of('end', inv='s9x_ensemble')) == k and 'CLOSE  ' not in r.stdout, 'a dry run writes no end'
        assert state() == '2 open 0 -'
        n += 1
        # --- 4. --max-attempts is a positive integer for EVERY profile ('[ 3 -gt x ]' is an
        #        error bash counts as false: the restart loop never ended)
        k = len(_ledger_of(d))
        for bad in ('x', '0', '-1', '2x', ''):
            r = _launch(d, '--stages', 'S9x', '--max-attempts', bad)
            assert r.returncode == 2 and 'not a positive integer' in r.stderr, (bad, r.stderr)
            n += 1
        assert len(_ledger_of(d)) == k, 'refused before the chain is set up'
        assert n == 3 + 5
        logs = os.path.join(d, 'logs', 'chain_%s' % REH)
        body = ''.join(open(os.path.join(logs, f)).read() for f in os.listdir(logs) if f.endswith('.log'))
        assert 'run_timeline_comparison' not in body
        assert not [f for dp, _dn, fs in os.walk(d) for f in fs if f.endswith('.pkl')]


def test_the_confirmatory_profile_refuses_invalidate_and_undeclared_waivers_and_records_it():
    """`--invalidate`, `--waive-dep` and `--waive-order` are operator actions
    that can redo or step around a confirmatory invocation after a report
    has been read.  The plan of profile r3lite restricts them
    (no_invalidate_from, declared_waivers_only); the launcher refuses what
    the plan restricts BEFORE anything starts and writes the refusal to the
    ledger.  Driven through the real launcher on a scratch plan that gives
    the REHEARSAL profile r3lite's two fields (development seeds; nothing is
    launched)."""
    _linux_or_fail('the launcher needs bash and /proc')
    with _scratch('r3_strict_launch_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        plan = os.path.join(d, 'confirmatory', 'chain_plan.py')
        with open(plan) as f:
            src = f.read()
        anchor = "        'out_prefix': 'r3lite_rehearsal/',\n"
        assert src.count(anchor) == 1
        with open(plan, 'w') as f:
            f.write(src.replace(anchor, anchor + "        'no_invalidate_from': %r, 'declared_waivers_only': %r,\n"
                                % (CP.PROFILES[R3]['no_invalidate_from'], CP.PROFILES[R3]['declared_waivers_only'])))

        def of(event, **kv):
            return [x for x in _ledger_of(d) if x.get('event') == event
                    and all(x.get(k) == v for k, v in kv.items())]
        n = 0
        # an invocation of S6 or later is never invalidated -- alone, or hidden in a list
        for arg, bad in (('s6_cf_b0', 's6_cf_b0'), ('s9x_ensemble', 's9x_ensemble'),
                         ('s7_analysis', 's7_analysis'), ('s7e_family_e', 's7e_family_e'),
                         ('s12_secondary', 's12_secondary'), ('s0_manifest,s9x_ensemble', 's9x_ensemble'),
                         ('nonsense', 'nonsense')):
            r = _launch(d, '--invalidate', arg)
            assert r.returncode == 2 and ('REFUSING: --invalidate %s' % bad) in r.stderr, (arg, r.stderr, r.stdout[-800:])
            assert not of('invalidate'), 'nothing of a refused list is applied'
            n += 1
        notes = of('note', reason='operator_action_refused')
        assert len(notes) == 7 and notes[0]['action'] == '--invalidate s6_cf_b0' \
            and 'never invalidated' in notes[0]['why'], notes
        # ... an invocation before S6 can be (segment 1 may be redone before the seal is used)
        r = _launch(d, '--invalidate', 's0_manifest')
        assert r.returncode == 0 and [x['inv'] for x in of('invalidate')] == ['s0_manifest'], r.stdout[-800:]
        n += 1
        # no dependency is waivable; no order edge but the two declared ones
        r = _launch(d, '--stages', 'S6', '--waive-dep', 'S6')
        assert r.returncode == 2 and 'REFUSING: --waive-dep S6' in r.stderr \
            and 'declares no waivable dependency' in r.stderr, r.stderr
        n += 1
        for st in ('S6h', 'S8', 'S9x', 'S7e', 'S7,S8'):
            r = _launch(d, '--stages', st.split(',')[-1], '--waive-order', st)
            assert r.returncode == 2 and 'REFUSING: --waive-order %s' % st.split(',')[-1] in r.stderr \
                and 'only for: S12 S7' in r.stderr, (st, r.stderr)
            n += 1
        # the restart rule is ONE unchanged re-run: no other attempt count can be asked for
        for k in ('1', '3', '9'):
            r = _launch(d, '--stages', 'S6', '--max-attempts', k)
            assert r.returncode == 2 and 'REFUSING: --max-attempts %s' % k in r.stderr and 'one unchanged re-run' in r.stderr, (k, r.stderr)
            n += 1
        # --shed is refused for the graph (as before), and for a restricted profile that
        # refusal is in the ledger too: "each refusal is recorded" holds for all of them
        r = _launch(d, '--dry-run', '--shed', 'untrained')
        assert r.returncode == 2 and 'sheds nothing' in r.stderr + r.stdout, r.stderr
        s = [x for x in of('note', reason='operator_action_refused') if x['action'].startswith('--shed')]
        assert len(s) == 1 and s[0]['action'] == '--shed untrained', s
        n += 1
        assert len(of('note', reason='operator_action_refused')) == 7 + 1 + 5 + 3 + 1
        assert not of('chain_start') and not of('start') and not of('preflight'), 'a refused action starts nothing'
        # the two declared stages may be NAMED (whether the waiver is honoured is the ledger's to say)
        r = _launch(d, '--dry-run', '--stages', 'S7', '--waive-order', 'S7,S12', '--max-attempts', '2')
        assert r.returncode == 0 and 'REFUSING' not in r.stderr, r.stdout[-1500:] + r.stderr
        n += 1
        # ONCE A CONFIRMATORY STAGE HAS STARTED nothing is invalidated any more -- not the
        # stages before it either: redoing S5 would seal FREEZE_FINAL again after S6 data exist
        with open(os.path.join(d, 'confirmatory', 'chain_%s.jsonl' % REH), 'a') as f:
            f.write(json.dumps({'utc': 'T', 'event': 'start', 'inv': 's6_cf_b0', 'stage': 'S6', 'attempt': 1}) + '\n')
        k = len(of('invalidate'))
        for arg in ('s0_manifest', 's5_freeze2', 's3b_record,s0_import'):
            r = _launch(d, '--invalidate', arg)
            assert r.returncode == 2 and 'REFUSING: --invalidate %s' % arg in r.stderr \
                and 'has started (STARTED s6_cf_b0)' in r.stderr, (arg, r.stderr)
            n += 1
        assert len(of('invalidate')) == k, 'nothing was invalidated'
        assert n == 7 + 1 + 1 + 5 + 3 + 1 + 1 + 3
    # the real rehearsal profile restricts nothing (its launcher tests drive single stages)
    assert _plan_cli('--profile', REH, '--dep-waiver').stdout.strip() == 'free'
    src = open(os.path.join(ROOT, 'confirmatory', 'run_final.sh'), encoding='utf-8').read()
    assert src.index('refuse_action "--waive-dep') < src.index('if [ -n "$INVALIDATE" ]; then') \
        < src.index('chain_say "CHAIN START'), 'the refusals come before the chain starts'


# ------------------------------------------------ S7 and S12, tools stubbed
def _fake_py(tmp, log):
    """$PY for the analysis steps: the plan and the inline checks run for
    real; the gate tools, the S8 check, the analysis, the report comparison
    and the ensemble comparison record how they were called and answer per
    FAKE_* environment variables.  The LEDGER queries of
    tools/stage_report.py (--terminal-failure, --ledger-artifacts, ...) run
    for real on the scratch ledger."""
    path = os.path.join(tmp, 'fakepy')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('#!%s\n' % sys.executable)
        fh.write('''import json, os, sys
LOG = %r
REAL_SR = %r
a = sys.argv[1:]
if a and (a[0] == '-' or a[0].endswith('chain_plan.py')):
    os.execv(sys.executable, [sys.executable] + a)
with open(LOG, 'a') as fh:
    fh.write(json.dumps(a) + '\\n')
tool = os.path.basename(a[0]) if a else ''
def arg(k, d=None):
    return a[a.index(k) + 1] if k in a else d
if tool == 'stage_report.py' and [k for k in ('--ledger-order', '--ledger-append', '--terminal-failure',
                                              '--ledger-artifacts') if k in a]:
    os.execv(sys.executable, [sys.executable, REAL_SR] + a[1:])
if tool == 'analysis_manifest.py':
    print('derived_manifest.json')
elif tool == 'hash_invariance_check.py':
    seeds = [int(s) for s in arg('--seeds', '').split(',') if s]
    json.dump({'n_compared': 12, 'same_hashseed': False, 'seeds': seeds,
               'scenarios': ['recovery', 'rescue_ops'],
               'decision_series': {'runs_with_key': 4}, 'identical': True},
              open(arg('--out'), 'w'))
    print('SUMMARY compared=12 different=0 missing=0 same_hashseed=False -> IDENTICAL')
elif tool == 'confirmatory_analysis.py':
    rep = {'primary': {'P1': {}}, 'seeds': {'A': [1], 'B': [1]}, 'secondary': {}, 'S12': {}}
    for item in [x for x in arg('--untrained', '').split(',') if x]:
        rep['S12'][item.split('=')[0]] = {
            'summary': {'A': {'n': 2}, 'B': {'n': 2}}, 'paired': {'A': {}, 'B': {}}}
    if '--s8' in a:
        rep['S8'] = json.load(open(arg('--s8')))
    json.dump(rep, open(arg('--out'), 'w'))
elif tool == 'compare_reports.py':
    print('SUMMARY leaves=10 mismatches=0 -> IDENTICAL')
elif tool == 'stage_report.py' and '--paired-descriptive' in a:
    seeds = [int(s) for s in arg('--paired-seeds', '').split(',') if s]
    mode = os.environ.get('FAKE_ENS', 'ok')
    if mode == 'fail':
        print('[PAIRED] FAIL ensemble')
        sys.exit(1)
    n = 0 if mode == 'vacuous' else 2 * len(seeds)
    blk = {'n': len(seeds), 'ach': {'paired': {'n': len(seeds) if n else 0,
                                               'mean': 2.0 if mode == 'othermean' else 1.0,
                                               'wins': 1 if mode == 'otherwins' else len(seeds)},
                                    'per_seed_difference': {str(s): 1.0 for s in seeds}}}
    json.dump({'ok': True, 'descriptive': mode != 'inferential', 'n_pairs': n,
               'scenarios': {'A': blk, 'B': blk}}, open(arg('--json-out'), 'w'))
    print('[PAIRED] PASS ensemble: marl_static minus xldet')
sys.exit(0)
''' % (log, os.path.join(ROOT, 'tools', 'stage_report.py')))
    os.chmod(path, 0o755)
    return path


def _ledger_records(tmp, invs, mode='ok', prim_path=None):
    """The chain ledger S12 finds, as run_final.sh writes it.  `mode`:
      ok                    every invocation before S12 ended successfully;
      s9x_failed2           the first S9x invocation failed twice, nothing after S6 but S7,
                            S6h and S8 ran (the registered contingency);
      s9x_watchdog          one failed attempt that contains the watchdog's re-run;
      s9x_failed1           ONE failed attempt: the restart rule is not exhausted;
      s9x_failed2_redone    failed twice, invalidated by the operator, then succeeded;
      s9x_stop_only         a `stop` line for S9x and no failed end at all;
      s7e_failed2           S9x and S7 succeeded, S7e failed twice;
      ledger_other_sha      as ok, but the ledger holds another sha256 for an S6 pickle."""
    L = []
    by = {i['id']: i for i in invs}

    def sha_of(inv):
        p = os.path.join(tmp, by[inv]['artifact'])
        return _sha256(p) if os.path.isfile(p) else '-'

    def run(inv, status='ok', attempt=1, wd=0, sha=None):
        L.append({'utc': 'T%04d' % len(L), 'event': 'start', 'inv': inv, 'stage': by[inv]['stage'],
                  'attempt': attempt})
        L.append({'utc': 'T%04d' % len(L), 'event': 'end', 'inv': inv, 'stage': by[inv]['stage'],
                  'attempt': attempt, 'status': status, 'exit': 0 if status == 'ok' else 7, 'gate': 0,
                  'sha256': sha or (sha_of(inv) if status == 'ok' else '-'),
                  'artifact': by[inv]['artifact'], 'watchdog_restarts': wd})

    def stage(st, **kw):
        for i in invs:
            if i['stage'] == st:
                run(i['id'], **kw)
    s9 = [i['id'] for i in invs if i['stage'] == 'S9x']
    for st in ('S0', 'S3b', 'S5'):
        stage(st)
    for i in invs:
        if i['stage'] == 'S6':
            run(i['id'], sha='0' * 64 if (mode == 'ledger_other_sha' and i['id'] == 's6_cffull_b0') else None)
    if mode in ('ok', 'ledger_other_sha', 's7e_failed2'):
        stage('S9x')
    elif mode == 's9x_failed2':
        run(s9[0], status='fail'); run(s9[0], status='fail', attempt=2)
        L.append({'utc': 'T', 'event': 'stop', 'inv': s9[0], 'reason': 'failed_twice'})
    elif mode == 's9x_watchdog':
        run(s9[0], status='fail', wd=1)
    elif mode == 's9x_failed1':
        run(s9[0], status='fail')
    elif mode == 's9x_stop_only':
        L.append({'utc': 'T', 'event': 'stop', 'inv': s9[0], 'stage': 'S9x', 'reason': 'failed_twice'})
        L.append({'utc': 'T', 'event': 'stop', 'inv': s9[0], 'stage': 'S9x', 'reason': 'preflight_failed'})
    elif mode == 's9x_failed2_redone':
        run(s9[0], status='fail'); run(s9[0], status='fail', attempt=2)
        L.append({'utc': 'T', 'event': 'invalidate', 'inv': s9[0], 'reason': 'operator'})
        stage('S9x')
    else:
        raise AssertionError(mode)
    run('s7_analysis', sha=_sha256(prim_path) if prim_path and os.path.isfile(prim_path) else None)
    if mode == 's7e_failed2':
        run('s7e_family_e', status='fail'); run('s7e_family_e', status='fail', attempt=2)
    elif mode in ('ok', 'ledger_other_sha', 's9x_failed2_redone'):
        run('s7e_family_e')
    stage('S6h'); stage('S8')
    return L


def _fake_pickle(path, inv):
    rec = {'recovery_multi': {}, 'rescue_multi': {}, 'seeds': list(inv['seeds']),
           'provenance': {'tree_head': 'faketest', 'tree_sha256': 'f' * 64, 'schema_id': 'fake',
                          'checkpoint_sha256': list(inv.get('ckpt_sha256') or []),
                          'pythonhashseed': inv.get('hashseed', '0')},
           'pass_stats': {}}
    for key in inv['expected_keys']:
        arm, scen, seed = key.split('|')
        tgt = rec['recovery_multi'] if scen == 'recovery' else rec['rescue_multi']
        tgt.setdefault(arm, {})[int(seed)] = {'conn': [1.0]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(rec, fh)


def _STAB_ENTRY(k, status=None, confirmed=None):
    """One endpoint of the stability family as confirmatory_analysis.
    stability_family writes it: S9e and S9f assessable and confirmed, the
    other four not assessable (R2's own outcome)."""
    worse = k in ('S9e', 'S9f')
    status = status or ('assessable' if worse else 'not assessable')
    ent = {'scenario': 'A' if k in ('S9a', 'S9c', 'S9e') else 'B',
           'metric': {'a': 'M2', 'b': 'M2', 'c': 'M4', 'd': 'M4', 'e': 'M7', 'f': 'M7'}[k[-1]],
           'expect': -1 if k in ('S9c', 'S9d') else 1, 'family_size': 6,
           'n': 2, 'status': status,
           'confirmed': (worse and status == 'assessable') if confirmed is None else confirmed}
    if status != 'not recorded':
        ent['n_nonzero'] = 2 if worse else 0
    return ent


_E_PRIMARY = {'P1_A_ach_vs_xldet': {'n': 2, 'wins': 2, 'mean': 1.0},
              'P4_B_ach_vs_xldet': {'n': 2, 'wins': 2, 'mean': 1.0}}


def _write_family_e(tmp, invs, prim_path, family_e=True):
    """The artefacts chain S7e leaves for S12: the family-E report, its
    record and the assembler's record.  `family_e`: True (as S7e writes
    them), False (absent), or a dict of defects: {'report': False} (no
    report), {'record': False}, {'prim_sha': 'x'} (the record names another
    family-1 report), {'fam_sha': 'x'}, {'control': 'DIFFERENT'},
    {'stability': <as _run_secondary>}, {'primary': {}} (a report without a
    primary test), {'order': False} (the record does not show family E's
    passes complete before family 1 was analysed), {'asm': False} (no
    assembly record), {'asm_sha': 'x'} (the record names another assembly
    record), {'foreign': True} (assembled from other pickles), {'short':
    True} (assembled without the ensemble pickle), {'p1': {...}} (another
    P1 than the listing's)."""
    if family_e is False:
        return
    d = family_e if isinstance(family_e, dict) else {}
    by = {i['id']: i for i in invs}
    fam = os.path.join(tmp, by['s7e_family_e']['artifact'])
    rec = fam.replace('cf_report_family_e.json', 'family_e_record.json')
    asm_rel = by['s7e_family_e']['artifact'].replace('cf_report_family_e.json', 'family_e_assembly.json')
    asm = os.path.join(tmp, asm_rel)
    prim_e = dict(_E_PRIMARY)
    if 'p1' in d:
        prim_e['P1_A_ach_vs_xldet'] = d['p1']
    rep = {'primary': d.get('primary', prim_e), 'seeds': {'A': [52, 53], 'B': [52, 53]},
           'stability': {}}
    stab = d.get('stability', True)
    over = stab if isinstance(stab, dict) else {}
    for k in ('S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f'):
        if not (stab is True or (not isinstance(stab, dict) and k in (stab or ())) or (over and True)):
            continue
        # family E's expected outcome: no endpoint assessable (too few non-zero differences)
        rep['stability'][k] = over.get(k, _STAB_ENTRY(k, status='not assessable'))
    if d.get('report', True):
        json.dump(rep, open(fam, 'w'))
    # the assembler's record: the comparator pickles of the primary input and
    # the ensemble pickle, each with the sha256 it had when it was read
    def entry(inv):
        p = os.path.join(tmp, by[inv]['artifact'])
        return {'path': 'elsewhere/' + by[inv]['artifact'] if d.get('foreign') else by[inv]['artifact'],
                'sha256': _sha256(p) if os.path.isfile(p) else '-'}
    if d.get('asm', True):
        json.dump({'status': 'PASS', 'comparators': [entry('s6_cffull_b0')],
                   'arm_source': [] if d.get('short') else [entry('s9x_ensemblefull')]}, open(asm, 'w'))
    if d.get('record', True):
        json.dump({'family_1_report': prim_path,
                   'family_1_report_sha256': d.get('prim_sha') or _sha256(prim_path),
                   'family_e_report': fam,
                   'family_e_report_sha256': d.get('fam_sha') or (_sha256(fam) if os.path.exists(fam) else ''),
                   'ledger_order': {'family_e_passes_complete_before_family_1_analysis': d.get('order', True)},
                   'family_e_assembly': {'record': asm_rel,
                                         'record_sha256': d.get('asm_sha')
                                         or (_sha256(asm) if os.path.exists(asm) else '')},
                   'assembly_control': {'reproduces_family_1_report': d.get('control', 'identical')}},
                  open(rec, 'w'))


def _run_secondary(skip=(), env=None, stability=True, analysis=None, family_e=True, ledger='ok',
                   after=None):
    """confirmatory/stage_analysis.sh --run r3lite_rehearsal s12_secondary in
    a scratch tree holding the rehearsal's (fake) pickles minus `skip`, the
    family-1 report, (family_e) the artefacts of chain S7e and a chain
    ledger (`ledger`: a mode of _ledger_records).  `after(tmp, invs)` runs
    last, once every artefact and the ledger are written -- to change a
    pickle after it was assembled and recorded."""
    profile = REH
    tmp = _mkdtemp('r3_secondary_')
    try:
        os.makedirs(os.path.join(tmp, 'confirmatory'))
        for f in ('stage_analysis.sh', 'chain_plan.py'):
            shutil.copy(os.path.join(ROOT, 'confirmatory', f), os.path.join(tmp, 'confirmatory', f))
        with open(os.path.join(ROOT, 'confirmatory_analysis.py'), 'rb') as f:
            data = f.read()
        with open(os.path.join(tmp, 'confirmatory_analysis.py'), 'wb') as f:
            f.write(data if analysis is None else analysis(data))
        invs = CP.build(profile)
        for i in invs:
            if i['artifact'].endswith('.pkl') and i['id'] not in skip:
                _fake_pickle(os.path.join(tmp, i['artifact']), i)
        prim = [i for i in invs if i['id'] == 's7_analysis'][0]['artifact']
        rep = {'primary': {'P1': {}}}
        if stability:
            # `stability`: True (all six, as the analysis writes them), a tuple
            # of endpoint ids (a partial family), or a dict {id: entry} that
            # replaces the entry of an endpoint (None removes its status)
            over = stability if isinstance(stability, dict) else {}
            rep['stability'] = {}
            for k in ('S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f'):
                if not (stability is True or k in stability or k not in over and over):
                    continue
                ent = _STAB_ENTRY(k)
                if k in over:
                    ent = over[k]
                rep['stability'][k] = ent
        json.dump(rep, open(os.path.join(tmp, prim), 'w'))
        _write_family_e(tmp, invs, os.path.join(tmp, prim), family_e)
        with open(os.path.join(tmp, 'confirmatory', 'chain_%s.jsonl' % profile), 'w') as f:
            for x in ([] if ledger is None else _ledger_records(tmp, invs, ledger, os.path.join(tmp, prim))):
                f.write(json.dumps(x, sort_keys=True) + '\n')
        if after:
            after(tmp, {i['id']: i for i in invs})
        log = os.path.join(tmp, 'calls.log')
        e = dict(os.environ, PY=_fake_py(tmp, log))
        for k in ('FAME_STATE', 'FAME_REPORT', 'FAME_RECORD', 'FAME_STAB', 'FAME_CAUSE',
                  'FAME_NC_STAGE', 'FAME_DEPOSIT'):
            e.pop(k, None)
        e.update(env or {})
        r = subprocess.run(['bash', os.path.join(tmp, 'confirmatory', 'stage_analysis.sh'),
                            '--run', profile, 's12_secondary'],
                           cwd=tmp, env=e, capture_output=True, text=True)
        calls = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
        rp = os.path.join(tmp, [i for i in invs if i['id'] == 's12_secondary'][0]['artifact'])
        return r, calls, (json.load(open(rp)) if os.path.exists(rp) else None)
    finally:
        _rmtree(tmp)


def _calls_of(calls, tool, flag=None):
    return [c for c in calls if c and os.path.basename(c[0]) == tool and (flag is None or flag in c)]


def test_s12_covers_what_exists_and_records_what_the_plan_dropped():
    _linux_or_fail('the stage script needs bash and a POSIX shebang')
    invs = {i['id']: i for i in CP.build(REH)}
    r, calls, rep = _run_secondary()
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    # the stages it describes: the plan's, each complete; no R2 stage is met
    # as a silent 'absent'
    assert rep['stages'] == ['S6h=complete(1)', 'S8=complete(2)', 'S9x=complete(2)'], rep['stages']
    assert rep['dropped_by_plan'] == CP.dropped_stages(REH)
    assert 'stages DROPPED by the plan of profile r3lite_rehearsal' in r.stdout
    a = rep['analyses']
    assert a['S8_hash_invariance']['state'] == 'identical'
    assert a['S12_untrained'] == {'state': 'computed', 'controls': 1}, a['S12_untrained']
    assert a['S7_secondary_marl_freeze'] == {'state': 'dropped', 'pickles': 0}
    assert 'chain S9 is DROPPED by the plan' in r.stdout
    assert a['ensemble_vs_xldet']['state'] == 'computed'
    assert a['ensemble_vs_xldet']['json'].endswith('r3lite_rehearsal_ensemble_vs_xldet.json')
    assert a['S9_stability_family']['state'] == 'reported(6)'
    # ... and WHAT each endpoint is, copied from the primary report: status
    # and verdict per endpoint, so the reader can tell which were assessed
    ep = a['S9_stability_family']['endpoints']
    assert sorted(ep) == ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f'], ep
    assert {k: (v['status'], v['confirmed']) for k, v in ep.items()} == {
        'S9a': ('not assessable', False), 'S9b': ('not assessable', False),
        'S9c': ('not assessable', False), 'S9d': ('not assessable', False),
        'S9e': ('assessable', True), 'S9f': ('assessable', True)}, ep
    assert a['S9_stability_family']['n_reported'] == 6
    assert a['S9_stability_family']['n_assessable'] == 2
    assert a['S9_stability_family']['not_recorded'] == []
    assert ep['S9e']['scenario'] == 'A' and ep['S9e']['metric'] == 'M7'
    assert "S9e A M7: status='assessable' confirmed=True" in r.stdout, r.stdout[-3000:]
    assert a['full_report']['state'] == 'computed'
    assert a['full_report']['primary_report_reproduced'] == 'identical'
    assert rep['failures'] == 0
    # THE SECOND CONFIRMATORY FAMILY: counted and copied from the report and
    # the record of chain S7e, never recomputed here; its stability endpoints
    # carry status and verdict like family 1's (here: none assessable)
    fe = a['family_E']
    assert fe['state'] == 'computed' and fe['n_primary'] == 2, fe
    # ... BOUND to the pickles it was computed from (the assembler's record and the
    # ledger, re-hashed here) and CROSS-CHECKED against the listing computed above
    assert 'family E: 2 input pickle(s) bound to the assembly record and the ledger; listing ' \
           'agrees on 2 of 2 achievability contrast(s)' in r.stdout, r.stdout[-3000:]
    la = _calls_of(calls, 'stage_report.py', '--ledger-artifacts')
    assert len(la) == 1 and la[0][la[0].index('--stage') + 1] == 'S6,S9x'
    tf = _calls_of(calls, 'stage_report.py', '--terminal-failure')
    assert [c[c.index('--stage') + 1] for c in tf] == ['S9x', 'S7e'], tf
    assert 'FAMILY E NOT COMPLETED' not in r.stdout
    assert fe['report'].endswith('r3lite_rehearsal_cf_report_family_e.json')
    assert fe['record'].endswith('r3lite_rehearsal_family_e_record.json')
    assert sorted(fe['stability_endpoints']) == ['S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f']
    assert {v['status'] for v in fe['stability_endpoints'].values()} == {'not assessable'}
    assert fe['n_assessable'] == 0 and fe['not_recorded'] == []
    assert "E-S9e A M7: status='not assessable' confirmed=False" in r.stdout, r.stdout[-3000:]
    assert 'inferential statements about the ensemble are those of the family-E report' \
        in a['ensemble_vs_xldet']['note']
    # ONE further run of the unchanged analysis over exactly the primary
    # input (no adapt-then-freeze pickle, no ensemble pickle), marl_static
    ana = _calls_of(calls, 'confirmatory_analysis.py')
    assert len(ana) == 1, calls
    prim = [invs['s6_cffull_b0']['artifact']]
    assert ana[0][ana[0].index('--pickles') + 1].split(',') == prim
    assert ana[0][ana[0].index('--marl-arm') + 1] == 'marl_static'
    assert ana[0][ana[0].index('--untrained') + 1] == \
        'untrained_s1234=' + invs['s8_untrainedfull_s1234']['artifact']
    assert '--no-assert' not in ana[0]
    # the ensemble: its own pickle against the XL-DET passes of the primary
    # input, the plan's seeds and the plan's four pins
    ens = _calls_of(calls, 'stage_report.py', '--paired-descriptive')
    assert len(ens) == 1, calls
    e = ens[0]
    assert e[e.index('--paired-descriptive') + 1] == invs['s9x_ensemblefull']['artifact']
    assert e[e.index('--baseline') + 1].split(',') == prim
    assert e[e.index('--paired-seeds') + 1] == '52,53'
    assert e[e.index('--paired-arm') + 1] == 'marl_static' and e[e.index('--paired-ref-arm') + 1] == 'xldet'
    assert e[e.index('--expect-checkpoint-sha256') + 1].split(',') == invs['s9x_ensemblefull']['ckpt_sha256']
    hc = _calls_of(calls, 'hash_invariance_check.py')
    assert len(hc) == 1 and hc[0][1:3] == [invs['s6_cf_b0']['artifact'], invs['s6h_hash1']['artifact']]
    # every pickle of the secondary family was gated with its own key set
    ver = _calls_of(calls, 'stage_report.py', '--verify')
    assert len(ver) == 5, ver


def test_s12_fails_loudly_on_every_missing_piece_of_r3lite():
    _linux_or_fail('the stage script needs bash and a POSIX shebang')
    n = 0
    # the ensemble pickle is missing (S9x is never shed)
    r, calls, rep = _run_secondary(skip=('s9x_ensemblefull',))
    assert r.returncode != 0 and 'ensemble pickle(s) missing' in r.stdout, r.stdout[-2000:]
    assert not _calls_of(calls, 'stage_report.py', '--paired-descriptive'); n += 1
    # the comparison tool fails / compares nothing / is not marked descriptive
    for mode in ('fail', 'vacuous', 'inferential'):
        r, calls, rep = _run_secondary(env={'FAKE_ENS': mode})
        assert r.returncode != 0, (mode, r.stdout[-2000:])
        assert rep['analyses']['ensemble_vs_xldet']['state'] in ('FAILED', 'VACUOUS'), mode
        n += 1
    # the primary report lacks the stability family, wholly or in part
    for stab in (False, ('S9a', 'S9b', 'S9c', 'S9d', 'S9e')):
        r, calls, rep = _run_secondary(stability=stab)
        assert r.returncode != 0 and 'six S9 endpoints' in r.stdout, r.stdout[-2000:]
        assert rep['analyses']['S9_stability_family']['state'] == 'MISSING'
        n += 1
    # an endpoint that is present but NOT RECORDED was not assessed: presence
    # is not "reported" (the check used to print reported(6) for it)
    r, calls, rep = _run_secondary(stability={'S9f': _STAB_ENTRY('S9f', status='not recorded')})
    assert r.returncode != 0 and 'NOT RECORDED' in r.stdout, r.stdout[-2000:]
    sf = rep['analyses']['S9_stability_family']
    assert sf['state'] == 'NOT_RECORDED' and sf['not_recorded'] == ['S9f'], sf
    assert sf['endpoints']['S9f']['status'] == 'not recorded' and sf['n_reported'] == 6
    n += 1
    # an endpoint without a status or without a verdict is not a report either
    for broken in ({'n': 2}, dict(_STAB_ENTRY('S9a'), status='fine'),
                   {k: v for k, v in _STAB_ENTRY('S9a').items() if k != 'confirmed'}):
        r, calls, rep = _run_secondary(stability={'S9a': broken})
        assert r.returncode != 0 and 'no valid status or verdict' in r.stdout, r.stdout[-2000:]
        assert rep['analyses']['S9_stability_family']['state'] == 'MALFORMED'
        n += 1
    # 'not assessable' is a legitimate outcome: all six not assessable passes
    r, calls, rep = _run_secondary(stability={
        k: _STAB_ENTRY(k, status='not assessable') for k in ('S9e', 'S9f')})
    assert r.returncode == 0, r.stdout[-2000:]
    sf = rep['analyses']['S9_stability_family']
    assert sf['state'] == 'reported(6)' and sf['n_assessable'] == 0, sf
    # the untrained control is missing
    r, calls, rep = _run_secondary(skip=('s8_untrainedfull_s1234',))
    assert r.returncode != 0 and 'untrained control pickle(s) missing' in r.stdout
    assert not _calls_of(calls, 'confirmatory_analysis.py'); n += 1
    # the hash-seed repeat is missing
    r, calls, rep = _run_secondary(skip=('s6h_hash1',))
    assert r.returncode != 0 and 'analysis S8 input missing' in r.stdout; n += 1
    # the analysis script is not the pre-registered file: NOTHING is analysed
    r, calls, rep = _run_secondary(analysis=lambda data: data + b'\n# edited\n')
    assert r.returncode != 0 and 'is not the pre-registered' in r.stdout + r.stderr, r.stdout[-1500:]
    assert not calls and rep is None; n += 1
    assert n == 13
    # THE SECOND FAMILY: S12 fails when chain S7e left no report, no record, a
    # record that names another family-1 or family-E report, no passed
    # assembly control, a report without a primary test, or a stability
    # family that is incomplete or not recorded
    m = 0
    for defect, msg in (
            (False, 'the family-E report or its record is missing'),
            ({'report': False}, 'the family-E report or its record is missing'),
            ({'record': False}, 'the family-E report or its record is missing'),
            ({'prim_sha': 'f' * 64}, 'names another family-1 report'),
            ({'fam_sha': 'f' * 64}, 'names another family-E report'),
            ({'control': 'DIFFERENT'}, 'no passed assembly control'),
            ({'primary': {}}, 'not the recorded, complete, current report of S7e'),
            ({'stability': {'S9f': _STAB_ENTRY('S9f', status='not recorded')}}, 'not recorded'),
            ({'stability': {'S9a': {'n': 2}}}, 'without a valid status'),
            # the order of the two families is not shown by the record
            ({'order': False}, 'does not show the passes of family E complete before'),
            ({'order': 'yes'}, 'does not show the passes of family E complete before'),
            # the assembler's record: absent, not the one the record names, of other pickles
            ({'asm': False}, 'the assembly record of family E is missing or is not the one'),
            ({'asm_sha': 'f' * 64}, 'the assembly record of family E is missing or is not the one'),
            ({'foreign': True}, 'was not assembled from the pickles of this chain'),
            ({'short': True}, 'was not assembled from the pickles of this chain'),
            # the listing computed here disagrees with the report: mean, wins, instances
            ({'p1': {'n': 2, 'wins': 2, 'mean': 1.5}}, 'disagree on P1'),
            ({'p1': {'n': 2, 'wins': 1, 'mean': 1.0}}, 'disagree on P1'),
            ({'p1': {'n': 3, 'wins': 2, 'mean': 1.0}}, 'disagree on P1'),
            # a mean that is not a number agrees with nothing (abs(nan - x) > eps is False)
            ({'p1': {'n': 2, 'wins': 2, 'mean': float('nan')}}, 'P1: a mean that is not a finite number'),
            ({'p1': {'n': 2, 'wins': 2, 'mean': '1.0'}}, 'P1: a mean that is not a finite number'),
            ({'p1': {}}, 'P1 of the family-E report cannot be compared')):
        r, calls, rep = _run_secondary(family_e=defect)
        assert r.returncode != 0 and msg in r.stdout, (defect, r.stdout[-2500:])
        if defect is not False and rep is not None:
            assert rep['analyses']['family_E']['state'] in ('missing', 'INVALID'), (defect, rep['analyses'])
        m += 1
    assert m == 21
    for mode in ('othermean', 'otherwins'):
        r, calls, rep = _run_secondary(env={'FAKE_ENS': mode})
        assert r.returncode != 0 and 'disagree on P1' in r.stdout and 'disagree on P4' in r.stdout, mode
        assert rep['analyses']['family_E']['state'] == 'INVALID'
        m += 1
    # STALENESS: a pickle that changed after family E was assembled from it, and a pickle
    # the ledger recorded with another digest -- each of S6 and S9x
    def touch(inv):
        def f(tmp, by):
            with open(os.path.join(tmp, by[inv]['artifact']), 'ab') as fh:
                fh.write(b'\n')
        return f
    for inv in ('s6_cffull_b0', 's9x_ensemblefull'):
        r, calls, rep = _run_secondary(after=touch(inv))
        assert r.returncode != 0 and 'is not the pickle family E was assembled from' in r.stdout, \
            (inv, r.stdout[-2500:])
        assert CP.by_id(REH, inv)['artifact'] + ' is not the pickle family E' in r.stdout
        assert rep['analyses']['family_E']['state'] == 'INVALID'
        m += 1
    r, calls, rep = _run_secondary(ledger='ledger_other_sha')
    assert r.returncode != 0 and 'is not the pickle the ledger recorded for s6_cffull_b0' in r.stdout, \
        r.stdout[-2500:]
    m += 1
    r, calls, rep = _run_secondary(ledger=None)
    assert r.returncode != 0 and 'is not the pickle the ledger recorded' in r.stdout; m += 1
    assert m == 27
    # and the unchanged input passes (the controls above are not a broken harness)
    r, calls, rep = _run_secondary()
    assert r.returncode == 0


def test_s12_reports_family_1_alone_only_when_the_ledger_records_family_e_as_not_completed():
    """PREREGISTRATION_R3 section 13, 'If family E cannot be completed'.
    The state is READ FROM THE LEDGER: the restart rule exhausted for an
    invocation of S9x or of S7e.  Then S12 completes with every secondary
    analysis of family 1, records family E as not_completed with the cause,
    computes NO ensemble listing and opens no ensemble pickle.  In every
    other state of the ledger a missing family E is a failure, as before."""
    _linux_or_fail('the stage script needs bash and a POSIX shebang')
    invs = {i['id']: i for i in CP.build(REH)}
    s9 = ('s9x_ensemble', 's9x_ensemblefull')
    n = 0
    # --- S9x failed twice (or once, with the watchdog's re-run inside): the contingency
    for mode, cause in (('s9x_failed2', 'failed twice (attempt 1 exit 7 gate 0; attempt 2 exit 7 gate 0)'),
                        ('s9x_watchdog', "failed after the watchdog's one unchanged re-run")):
        r, calls, rep = _run_secondary(skip=s9, family_e=False, ledger=mode)
        assert r.returncode == 0, (mode, r.stdout[-4000:] + r.stderr[-1500:])
        assert 'FAMILY E NOT COMPLETED (read from the ledger): TERMINAL S9x s9x_ensemble: ' + cause in r.stdout
        a = rep['analyses']
        fe = a['family_E']
        assert fe['state'] == 'not_completed' and fe['not_completed_stage'] == 'S9x', fe
        assert fe['cause'].startswith('TERMINAL S9x s9x_ensemble: ' + cause), fe['cause']
        assert fe['report'] is None and fe['record'] is None and fe['passes_on_disk'] == {}
        assert 'stability_endpoints' not in fe and 'n_primary' not in fe
        assert a['ensemble_vs_xldet']['state'] == 'not_completed' and a['ensemble_vs_xldet']['json'] is None
        assert 'NOT COMPUTED and not reported' in a['ensemble_vs_xldet']['note']
        assert rep['stages'] == ['S6h=complete(1)', 'S8=complete(2)',
                                 'S9x=not_described(family_E_not_completed_at_S9x)'], rep['stages']
        # everything of family 1 is there, exactly as in a complete run
        assert a['S8_hash_invariance']['state'] == 'identical' and a['S12_untrained']['state'] == 'computed'
        assert a['full_report']['state'] == 'computed' and a['full_report']['primary_report_reproduced'] == 'identical'
        assert a['S9_stability_family']['state'] == 'reported(6)' and rep['failures'] == 0
        # nothing of the ensemble was computed or opened
        assert not _calls_of(calls, 'stage_report.py', '--paired-descriptive')
        assert not _calls_of(calls, 'stage_report.py', '--ledger-artifacts')
        assert len(_calls_of(calls, 'stage_report.py', '--verify')) == 3, 'S6h and the two S8 pickles only'
        assert not [p for p in rep['pickles'] if 's9x' in p]
        n += 1
    # ... S9x failed for good but (partial) ensemble pickles are on disk: named with their
    # digest for the deposit, never opened, never listed
    r, calls, rep = _run_secondary(family_e=False, ledger='s9x_failed2')
    assert r.returncode == 0, r.stdout[-3000:]
    fe = rep['analyses']['family_E']
    assert fe['state'] == 'not_completed' and sorted(fe['passes_on_disk']) == sorted(invs[i]['artifact'] for i in s9)
    assert all(len(v) == 64 for v in fe['passes_on_disk'].values())
    assert not _calls_of(calls, 'stage_report.py', '--paired-descriptive')
    assert not [p for p in rep['pickles'] if 's9x' in p] \
        and 'S9x=not_described(family_E_not_completed_at_S9x)' in rep['stages']
    # NO TOOL IS HANDED AN ENSEMBLE PICKLE, although both are on disk: not the completeness
    # gate, not the provenance gate, not the listing (every call of every tool is logged)
    assert calls and not [c for c in calls if any('s9x_' in str(x) for x in c)], \
        [c for c in calls if any('s9x_' in str(x) for x in c)]
    n += 1
    # --- S9x and S7 succeeded, S7e failed twice: the same state, the cause names S7e
    r, calls, rep = _run_secondary(family_e=False, ledger='s7e_failed2')
    assert r.returncode == 0, r.stdout[-3000:]
    fe = rep['analyses']['family_E']
    assert fe['state'] == 'not_completed' and fe['not_completed_stage'] == 'S7e', fe
    assert fe['cause'].startswith('TERMINAL S7e s7e_family_e: failed twice'), fe['cause']
    assert sorted(fe['passes_on_disk']) == sorted(invs[i]['artifact'] for i in s9)
    assert rep['analyses']['ensemble_vs_xldet']['state'] == 'not_completed'
    assert not _calls_of(calls, 'stage_report.py', '--paired-descriptive')
    # the label says WHERE family E stopped: S9x itself is complete here, and is not described
    assert rep['stages'][-1] == 'S9x=not_described(family_E_not_completed_at_S7e)', rep['stages']
    assert calls and not [c for c in calls if any('s9x_' in str(x) for x in c)]
    n += 1
    # --- the ledger says not completed, but a family-E report exists: a contradiction, a FAILURE
    for mode in ('s9x_failed2', 's7e_failed2'):
        r, calls, rep = _run_secondary(ledger=mode)
        assert r.returncode != 0 and 'but a family-E report or record exists' in r.stdout, (mode, r.stdout[-2000:])
        assert rep['analyses']['family_E']['state'] == 'INCONSISTENT'
        n += 1
    # --- NEGATIVE CONTROLS: every state of the ledger that is NOT an exhausted restart rule
    # leaves a missing family E the failure it always was
    for mode in ('s9x_failed1', 's9x_stop_only', 'ok', None):
        r, calls, rep = _run_secondary(skip=s9, family_e=False, ledger=mode)
        assert r.returncode != 0, (mode, r.stdout[-2000:])
        assert 'FAMILY E NOT COMPLETED' not in r.stdout, mode
        assert 'ensemble pickle(s) missing' in r.stdout and 'the family-E report or its record is missing' in r.stdout
        assert rep['analyses']['family_E']['state'] == 'missing', mode
        n += 1
    # ... and a failure that the operator invalidated and that then succeeded is an ordinary run
    r, calls, rep = _run_secondary(ledger='s9x_failed2_redone')
    assert r.returncode == 0 and rep['analyses']['family_E']['state'] == 'computed', r.stdout[-3000:]
    assert 'FAMILY E NOT COMPLETED' not in r.stdout
    n += 1
    assert n == 2 + 1 + 1 + 2 + 4 + 1
    # an R2 profile never asks the question (no ledger query, no such state)
    src = open(os.path.join(ROOT, 'confirmatory', 'stage_analysis.sh'), encoding='utf-8').read()
    step = src.split('if [ "$INV_STEP" = secondary ]; then', 1)[1]
    call = '"$PY" tools/stage_report.py --terminal-failure'
    assert step.count(call) == 1
    assert step.index('if [ "$(plan --graph)" = r3lite ]; then') < step.index(call)


def test_the_primary_step_refuses_an_analysis_script_that_is_not_pinned():
    _linux_or_fail('the stage script needs bash and a POSIX shebang')

    def primary(analysis=None):
        tmp = _mkdtemp('r3_primary_')
        try:
            os.makedirs(os.path.join(tmp, 'confirmatory'))
            for f in ('stage_analysis.sh', 'chain_plan.py'):
                shutil.copy(os.path.join(ROOT, 'confirmatory', f), os.path.join(tmp, 'confirmatory', f))
            with open(os.path.join(ROOT, 'confirmatory_analysis.py'), 'rb') as f:
                data = f.read()
            with open(os.path.join(tmp, 'confirmatory_analysis.py'), 'wb') as f:
                f.write(data if analysis is None else analysis(data))
            for i in CP.build(REH):
                if i['stage'] == 'S6':
                    _fake_pickle(os.path.join(tmp, i['artifact']), i)
            log = os.path.join(tmp, 'calls.log')
            r = subprocess.run(['bash', os.path.join(tmp, 'confirmatory', 'stage_analysis.sh'),
                                '--run', REH, 's7_analysis'], cwd=tmp,
                               env=dict(os.environ, PY=_fake_py(tmp, log)),
                               capture_output=True, text=True)
            return r, ([json.loads(l) for l in open(log)] if os.path.exists(log) else [])
        finally:
            _rmtree(tmp)

    r, calls = primary()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    ana = _calls_of(calls, 'confirmatory_analysis.py')
    assert len(ana) == 1
    assert ana[0][ana[0].index('--pickles') + 1] == 'output/r3lite_rehearsal/s6_cffull_b0.pkl'
    assert ana[0][ana[0].index('--marl-arm') + 1] == 'marl_static' and '--no-assert' not in ana[0]
    assert 'md5 %s == the pre-registered pin' % ANALYSIS_MD5 in r.stdout
    # NEGATIVE CONTROL
    r, calls = primary(analysis=lambda data: data.replace(b'ALPHA = 0.05', b'ALPHA = 0.10'))
    assert r.returncode != 0 and not _calls_of(calls, 'confirmatory_analysis.py'), r.stdout[-1500:]


# ================================================================ seed audit
def _audit_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location('seed_audit_r3', os.path.join(ROOT, 'tools', 'seed_audit.py'))
    SA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SA)
    return SA


def test_the_audited_block_is_a_parameter_and_the_r3_gate_is_zero_result_data():
    """The mechanics, on a SYNTHETIC window (700-719) so that no file naming
    a seed of the R3 block is ever written: the plan and this run's own S0
    manifest are planning data by explicit markers; a pickle is result data
    whatever its path; the gate exits 5 on result data."""
    SA = _audit_mod()
    lo, hi = 700, 719
    with _scratch('r3_audit_') as top:
        tree = os.path.join(top, 'MARL_x')
        os.makedirs(os.path.join(tree, 'confirmatory'))
        os.makedirs(os.path.join(tree, 'logs', 'chain_p'))
        with open(os.path.join(tree, 'confirmatory', 'chain_plan.py'), 'w') as f:
            f.write("PROFILES = {'p': {'cf_seeds': list(range(700, 720))}}\n")
        man = 'confirmatory/p_manifest_700_719.json'
        with open(os.path.join(tree, man), 'w') as f:
            json.dump({'instances': [{'topology_seed': s, 'scenario_kind': 'recovery'}
                                     for s in range(700, 720)]}, f)
        with open(os.path.join(tree, 'logs', 'chain_p', 's0_manifest.log'), 'w') as f:
            f.write('[stage_prep] benchmark manifest for seeds 700-719 -> %s\n' % man)
        with open(os.path.join(tree, 'notes.txt'), 'w') as f:
            f.write('nothing to see\n')
        markers = ['confirmatory/chain_plan.py', man, os.path.basename(man), 'logs/chain_p/s0_manifest']

        def audit(**kw):
            au = SA.Audit(lo, hi, [tree], home_check=False, **kw)
            au.run()
            return au.rep

        # the default audit (R2's): the plan and the manifest are HITS, and the
        # report carries none of the R3 fields
        rep = audit()
        v = rep['verdict']
        assert v['window_hits_in_result_data'] == list(range(700, 720)), v
        assert v['window_hits_in_text_non_planning'] and not v['clean']
        assert 'result_data_gate' not in v and 'planning_markers' not in rep['requested']
        assert 'gate_result_data' not in rep['requested']
        # freeze 2 of the R3 gate: the same tree, the plan and the run's own
        # S0 manifest classified by markers -> zero result data, gate passes
        rep = audit(planning_markers=markers, gate_result_data=True)
        v = rep['verdict']
        g = v['result_data_gate']
        assert g['pass'] is True and g['files_with_result_data_hits'] == [], g
        assert len(g['planning_json_with_window_seeds']) == 1
        assert g['planning_json_with_window_seeds'][0].endswith(man), g
        assert v['window_hits_in_result_data'] == [] and v['clean'] is True, v
        assert rep['requested']['planning_markers'] == markers
        # freeze 1: the manifest marker is NOT given -> the manifest is result data
        rep = audit(planning_markers=markers[:1], gate_result_data=True)
        g = rep['verdict']['result_data_gate']
        assert g['pass'] is False and len(g['files_with_result_data_hits']) == 1, g
        # a PICKLE of the window is result data even under a planning path
        pk = os.path.join(tree, 'logs', 'chain_p', 's0_manifest.pkl')
        with open(pk, 'wb') as f:
            pickle.dump({'recovery_multi': {'xldet': {705: {'conn': [1.0]}}}, 'seeds': [705]}, f)
        rep = audit(planning_markers=markers, gate_result_data=True)
        g = rep['verdict']['result_data_gate']
        assert g['pass'] is False and [os.path.basename(x) for x in g['files_with_result_data_hits']] \
            == ['s0_manifest.pkl'], g
        assert rep['verdict']['window_hits_in_result_data'] == [705]
        # exit codes of the CLI: 5 on result data, 0 once it is gone, 3 for a
        # text mention left to classify, 4 on refused coverage
        out = os.path.join(top, 'rep.json')
        base = ['--lo', str(lo), '--hi', str(hi), '--roots', tree, '--out', out, '--no-home-check']
        mk = [x for m in markers for x in ('--planning-marker', m)]
        import io
        def cli(*extra):
            buf, old = io.StringIO(), sys.stdout
            sys.stdout = buf
            try:
                return SA.main(base + list(extra)), buf.getvalue()
            finally:
                sys.stdout = old
        rc, txt = cli('--gate-result-data', *mk)
        assert rc == 5 and 'RESULT-DATA GATE   : FAIL' in txt and 's0_manifest.pkl' in txt, txt[-1500:]
        assert cli(*mk)[0] == 3, 'without the gate flag the same hit keeps exit 3'
        os.remove(pk)
        rc, txt = cli('--gate-result-data', *mk)
        assert rc == 0 and 'RESULT-DATA GATE   : PASS' in txt, txt[-1500:]
        with open(os.path.join(tree, 'run.log'), 'w') as f:
            f.write('python3 run_timeline_comparison.py --seed-list 703,704 --arms xldet\n')
        assert cli('--gate-result-data', *mk)[0] == 3
        assert cli('--gate-result-data', '--min-files', '100000', *mk)[0] == 4
        assert json.load(open(out))['window'] == [lo, hi]
        # THE CLASSES OF A TEXT HIT (PREREGISTRATION_R3 section 13(b)).  A
        # tick count next to the word "seed" (class c, coincidence) and a
        # statement that the block was NOT read (class d) are text hits until
        # they are classified by an explicit --false-positive pattern; then
        # they leave the verdict and are LISTED with path and line.
        os.remove(os.path.join(tree, 'run.log'))
        with open(os.path.join(tree, 'README.md'), 'w') as f:
            f.write('3. smoke run (seed 82, 705 ticks, the arm that reloads)\n'
                    'Seeds read: 50-59. Nothing for seeds 700-719 was read.\n')
        fp = ['--false-positive', '(seed 82, 705 ticks', '--false-positive',
              'Nothing for seeds 700-719 was read']
        rc, txt = cli('--gate-result-data', *mk)
        assert rc == 3 and 'RESULT-DATA GATE   : PASS' in txt, txt[-1500:]
        assert len(json.load(open(out))['text_hits']) == 2
        rc, txt = cli('--gate-result-data', *(mk + fp))
        assert rc == 0, txt[-1500:]
        rep = json.load(open(out))
        assert rep['text_hits'] == [] and rep['verdict']['clean'] is True
        assert sorted((os.path.basename(e['path']), e['line']) for e in rep['false_positive_hits']) \
            == [('README.md', 1), ('README.md', 2)], rep['false_positive_hits']
        assert rep['false_positive_patterns'] == fp[1::2]
        assert rep['verdict']['window_hits_classified_false_positive'] == list(range(700, 720))
        assert 'Hits classified as false positives by --false-positive (2 lines)' in txt
        # NEGATIVE CONTROLS: one of the two patterns alone leaves the other
        # line a hit; a line no pattern names stays a hit; and a pattern can
        # never clear RESULT DATA (the pickle keeps exit 5)
        assert cli('--gate-result-data', *(mk + fp[:2]))[0] == 3
        with open(os.path.join(tree, 'run.log'), 'w') as f:
            f.write('python3 run_timeline_comparison.py --seed-list 703,704 --arms xldet\n')
        rc, txt = cli('--gate-result-data', *(mk + fp))
        assert rc == 3 and json.load(open(out))['verdict']['window_hits_in_text_non_planning'] == [703, 704]
        os.remove(os.path.join(tree, 'run.log'))
        with open(pk, 'wb') as f:
            pickle.dump({'recovery_multi': {'xldet': {705: {'conn': [1.0]}}}, 'seeds': [705]}, f)
        rc, txt = cli('--gate-result-data', *(mk + fp + ['--false-positive', 's0_manifest.pkl']))
        assert rc == 5 and 'RESULT-DATA GATE   : FAIL' in txt, txt[-1500:]
        os.remove(pk)
    # the R3 pre-registration is a planning document for every window
    assert 'docs/PREREGISTRATION_R3' in SA.PLANNING_MARKERS and 'docs/PREREGISTRATION_R2' in SA.PLANNING_MARKERS


def test_the_audit_driver_defaults_to_r2_and_gates_only_the_r3_block():
    _linux_or_fail('the driver is a bash script')
    with open(os.path.join(ROOT, 'tools', 'run_seed_audit.sh'), encoding='utf-8') as f:
        drv = f.read()
    assert 'LO=${1:-200}' in drv and 'HI=${2:-219}' in drv, 'the default block stays R2\'s'
    assert '--gate-result-data' in drv and '--planning-marker' in drv
    assert 'r3-freeze1|r3-freeze2)' in drv
    # the S0 manifest path is read from the plan, never typed into the driver
    assert '--profile r3lite --field s0_manifest artifact' in drv
    assert '300_319' not in drv and '300_339' not in drv and 'manifest_3' not in drv
    # THE PRE-REGISTERED CLASSIFICATIONS (PREREGISTRATION_R3 section 13(b)):
    # exactly five line patterns, read under the R3 gate only, counted by the
    # driver; the default (R2) command keeps its one pattern, option for option
    tag = 'R3_PREREGISTERED_FALSE_POSITIVES'
    assert drv.count("<<'%s'\n" % tag) == 1 and drv.count('\n%s\n' % tag) == 1
    pats = drv.split("<<'%s'\n" % tag, 1)[1].split('\n%s\n' % tag, 1)[0].splitlines()
    assert pats == ['[CROSS] + INSIDE)), range(0, 400))',
                    '(seed 82, 300 ticks, the arm that writes and reloads',
                    'marl_freeze smoke, seed 82, 300 ticks',
                    '0c DONE (team term in the actor advantage)',
                    'Nothing for seeds 200-219, 300-359 or 400-439 was read'], pats
    assert '[ "$NFP" -eq 5 ]' in drv
    gate_block = drv.split('r3-freeze1|r3-freeze2)', 1)[1].split('\n  *) echo', 1)[0]
    assert "<<'%s'" % tag in gate_block, 'the patterns are read inside the R3 gate only'
    assert drv.count('--false-positive "seed 42, 200 island ticks"') == 1
    code = [l for l in drv.splitlines() if '--false-positive' in l and not l.lstrip().startswith('#')]
    assert len(code) == 2, code     # the R2 pattern, and the loop over the five
    # each pattern fits the tool's snippet (the first 200 characters of a line)
    assert all(0 < len(p) <= 200 for p in pats)
    # every one of them is listed, with its class, in the pre-registration
    with open(os.path.join(ROOT, 'docs', 'PREREGISTRATION_R3.md'), encoding='utf-8') as f:
        prereg = f.read()
    for p in pats:
        assert '`%s`' % p in prereg, 'pattern not pre-registered in section 13: %r' % p

    def drive(gate, lo, hi):
        e = dict(os.environ, MARL_TREE=ROOT, SEED_AUDIT_GATE=gate)
        return subprocess.run(['bash', os.path.join(ROOT, 'tools', 'run_seed_audit.sh'), str(lo), str(hi)],
                              cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
    # each of these is refused by the driver BEFORE the audit tool is started
    r = drive('bogus', 200, 219)
    assert r.returncode == 2 and 'unknown SEED_AUDIT_GATE' in r.stderr, r.stderr
    for gate in ('r3-freeze1', 'r3-freeze2'):
        for lo, hi in ((200, 219), (300, 319), (300, 338), (301, 340), (300, 359), (52, 53)):
            r = drive(gate, lo, hi)
            assert r.returncode == 2 and 'audits the block of the r3lite profile' in r.stderr, \
                (gate, lo, hi, r.stderr)
            assert 'EXIT_' not in r.stdout, 'no audit ran'


# =================================== the freeze-2 audit as a gate of the launcher
def _audit_report(path, window, tree, head, marker, started, clean=True, gate_pass=True,
                  coverage=True, text_hits=(), data_files=(), gate=True):
    """A SYNTHETIC seed-audit report with the fields tools/seed_audit.py
    writes and tools/stage_report.py --audit-gate reads."""
    v = {'clean': clean, 'coverage_ok': coverage,
         'coverage_failures': [] if coverage else ['root not walked'],
         'window_hits_in_result_data': [] if gate_pass else [window[0]],
         'window_hits_in_text_non_planning': list(text_hits),
         'window_hits_in_file_names': [], 'window_hits_in_git_history': []}
    if gate:
        v['result_data_gate'] = {'window': list(window), 'pass': gate_pass,
                                 'files_with_result_data_hits': list(data_files),
                                 'planning_json_with_window_seeds': [marker]}
    rep = {'window': list(window), 'started': started, 'verdict': v,
           'requested': {'gate_result_data': gate, 'tree': tree,
                         'planning_markers': ['confirmatory/chain_plan.py', marker]},
           'synced_head': {'value': head}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(rep, f)


def test_the_audit_gate_accepts_only_the_passing_audit_of_the_sealed_tree():
    """tools/stage_report.py --audit-gate on synthetic reports (a synthetic
    window, so that no file naming a seed of the R3 block is written)."""
    with _scratch('r3_auditgate_') as d:
        fz = os.path.join(d, 'FREEZE_FINAL_x.txt')
        with open(fz, 'w') as f:
            f.write('FREEZE_FINAL  profile=x  sealed 2026-10-11T08:00:00Z\ntree: %s\n'
                    'tree_head            abc1234\ntree_sha256          %s\n' % (d, 'e' * 64))
        rp = os.path.join(d, 'output', 'audit.json')
        man = 'confirmatory/x_manifest_700_719.json'
        led = os.path.join(d, 'ledger.jsonl')
        base = dict(window=[700, 719], tree=d, head='abc1234', marker=man,
                    started='2026-10-11 08:05:00 UTC')

        def write(**kw):
            _audit_report(rp, **dict(base, **kw))

        def call(ledger='', freeze=fz, report=rp):
            return _stage_report('--audit-gate', report, '--freeze-file', freeze,
                                 '--expect-window', '700,719', '--expect-planning-marker', man,
                                 '--tree', d, '--audit-ledger', ledger)

        def note(sha):
            with open(led, 'a') as f:
                f.write(json.dumps({'event': 'note', 'reason': 'audit_classified',
                                    'report_sha256': sha, 'utc': 'now'}) + '\n')

        write()
        r = call()
        assert r.returncode == 0, r.stdout + r.stderr
        assert '[AUDIT-GATE] PASS' in r.stdout and '9 check(s), 0 failed' in r.stdout, r.stdout
        n = 0
        # NEGATIVE CONTROLS: each of these is a FAIL (exit 1), and a recorded
        # acknowledgement of exactly that report does not change it
        for kw in (dict(window=[700, 718]),                                  # another block
                   dict(gate_pass=False, clean=False, data_files=['x.pkl']),  # result data
                   dict(gate=False),                                          # gate not requested
                   dict(coverage=False, clean=False),                         # coverage refused
                   dict(marker='confirmatory/chain_plan.py'),                 # a freeze-1 report
                   dict(started='2026-10-11 07:59:59 UTC'),                   # before the seal
                   dict(started='yesterday'),                                 # no usable time
                   dict(head='def5678'),                                      # another head
                   dict(tree=os.path.join(d, 'elsewhere'))):                  # another tree
            write(**kw)
            note(_sha256(rp))
            r = call(ledger=led)
            assert r.returncode == 1 and '[AUDIT-GATE] FAIL' in r.stdout, (kw, r.stdout)
            n += 1
        os.remove(rp)
        assert call().returncode == 1; n += 1                                 # no report
        with open(rp, 'w') as f:
            f.write('{not json')
        assert call().returncode == 1; n += 1                                 # unparsable
        write()
        assert call(freeze=fz + '.none').returncode == 1; n += 1              # not sealed
        with open(fz + '.nohead', 'w') as f:
            f.write('FREEZE_FINAL  profile=x  sealed 2026-10-11T08:00:00Z\ntree_sha256 %s\n' % ('e' * 64))
        assert call(freeze=fz + '.nohead').returncode == 1; n += 1            # a seal without a head
        assert n == 13
        # HITS LEFT TO CLASSIFY: exit 3 until a ledger note names THIS report
        write(clean=False, text_hits=[705])
        r = call(ledger=led)
        assert r.returncode == 3 and '[AUDIT-GATE] UNCLASSIFIED' in r.stdout, r.stdout
        assert 'text 1' in r.stdout
        note('0' * 64)                                                        # another report's note
        assert call(ledger=led).returncode == 3
        note(_sha256(rp))
        r = call(ledger=led)
        assert r.returncode == 0 and '[AUDIT-GATE] CLASSIFIED' in r.stdout, r.stdout
        assert call().returncode == 3, 'without the ledger the note does not exist'
        # ... and a report that changes afterwards is unclassified again
        write(clean=False, text_hits=[705, 706])
        assert call(ledger=led).returncode == 3


def test_r3lite_declares_the_audit_gate_and_the_other_profiles_do_not():
    g = _plan_cli('--profile', R3, '--audit-gate').stdout.split()
    assert g == ['S6', 'r3-freeze2',
                 'output/seed_audit_%d_%d_r3-freeze2.json' % (BLOCK[0], BLOCK[-1])], g
    assert CP.audit_gate(R3) == tuple(g)
    # the path is the one the committed driver writes for that gate
    with open(os.path.join(ROOT, 'tools', 'run_seed_audit.sh'), encoding='utf-8') as f:
        drv = f.read()
    assert 'OUT="output/seed_audit_${LO}_${HI}_${GATE}.json"' in drv
    # S6 is the first stage that runs an arm, and every stage before it is
    # one of segment 1 (S0, S3b, S5)
    order = CP.stage_order(R3)
    assert order[:order.index('S6')] == ['S0', 'S3b', 'S5']
    assert not [i['id'] for i in CP.build(R3)
                if i['kind'] == 'eval' and order.index(i['stage']) < order.index('S6')]
    n = 0
    for profile in ('real', 'rehearsal', 'dryrun', 'test', REH):
        assert CP.audit_gate(profile) is None, profile
        assert _plan_cli('--profile', profile, '--audit-gate').stdout.strip() == '', profile
        n += 1
    assert n == 5
    # NEGATIVE CONTROLS: the builder refuses r3lite without the gate, with
    # another gate, or with the gate after the first confirmatory stage
    for changes in (dict(audit_gate=''), dict(audit_gate='r3-freeze1'), dict(audit_before='S7')):
        with _patched_profile(R3, **changes):
            assert _raises_systemexit(lambda: CP.build(R3)), changes
    with _patched_profile(R3, audit_before='S99'):
        assert _raises_systemexit(lambda: CP.audit_gate(R3))
    assert CP.build(R3), 'the unpatched profile builds again'


def test_the_launcher_stops_at_the_gated_stage_without_a_passing_freeze_2_audit():
    """The launcher's audit gate, driven for real in a scratch chain.  The
    scratch plan gives the REHEARSAL profile r3lite's gate (development
    seeds; the real rehearsal declares none), segment 1 runs for real and
    seals FREEZE_FINAL, and the audit reports are synthetic.  NO evaluation
    invocation can start in this test: wherever the gate is expected to
    PASS the launcher is pointed at S7, whose dependency S6 is incomplete,
    and where S6 itself is requested the selected checkpoint is moved
    aside first, so a broken gate would stop at the missing checkpoint
    and not at a simulation."""
    _linux_or_fail('the launcher needs bash and /proc')
    _need_r2_source()
    with _scratch('r3_gate_launch_') as top:
        d = _stage_tree(os.path.join(top, 'tree'), engine=True)
        plan = os.path.join(d, 'confirmatory', 'chain_plan.py')
        with open(plan) as f:
            src = f.read()
        anchor = "        'out_prefix': 'r3lite_rehearsal/',\n"
        assert src.count(anchor) == 1
        with open(plan, 'w') as f:
            f.write(src.replace(anchor, anchor + "        'audit_gate': 'r3-freeze2', "
                                                 "'audit_before': 'S6',\n"))

        def plan_cli(*args):
            return subprocess.run([sys.executable, plan, '--profile', REH] + list(args),
                                  capture_output=True, text=True).stdout.strip()
        before, gate, report = plan_cli('--audit-gate').split()
        assert (before, gate, report) == ('S6', 'r3-freeze2', 'output/seed_audit_52_53_r3-freeze2.json')
        man = plan_cli('--field', 's0_manifest', 'artifact')
        rp = os.path.join(d, report)

        def stops(reason):
            return [x for x in _ledger_of(d) if x.get('event') == 'stop' and x.get('reason') == reason]

        def launched():
            led = _ledger_of(d)
            return [x for x in led if x.get('event') in ('start', 'preflight')
                    and x.get('stage') not in ('S0', 'S3b', 'S5')]

        # segment 1, for real: manifest, import, selection record, the seal
        r = _launch(d, '--stages', 'S0,S3b,S5')
        assert r.returncode == 0 and 'CHAIN END profile=%s rc=0' % REH in r.stdout, r.stdout[-4000:]
        assert '[AUDIT-GATE]' not in r.stdout, 'segment 1 lies before the gate'
        fz = os.path.join(d, 'confirmatory', 'FREEZE_FINAL_%s.txt' % REH)
        assert os.path.isfile(fz)
        import time as _time
        now = _time.strftime('%Y-%m-%d %H:%M:%S UTC', _time.gmtime(_time.time() + 2))
        base = dict(window=[52, 53], tree=d, head='scratch', marker=man, started=now)
        n = 0
        # 1. NO REPORT: S6 does not start (the checkpoint is aside, see above)
        ck = os.path.join(d, 'output', 'r3lite_rehearsal', 'ck_s1234', 'policy_final.pt')
        os.rename(ck, ck + '.aside')
        try:
            for args in (('--from', 'S6'), ('--stages', 'S6'), ()):
                r = _launch(d, *args)
                assert r.returncode != 0 and 'no passing freeze-2 seed audit of this sealed tree' in r.stdout, \
                    (args, r.stdout[-3000:])
                assert '  RUN    ' not in r.stdout and not launched(), (args, launched())
                assert not stops('missing_checkpoint'), 'the gate stops BEFORE the invocation is reached'
                n += 1
            assert len(stops('audit_gate_failed')) == 3
        finally:
            os.rename(ck + '.aside', ck)
        # 2. ... and no later stage can step around the gate
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and 'STOP  S7: no passing freeze-2 seed audit' in r.stdout, r.stdout[-3000:]
        r = _launch(d, '--from', 'S12')
        assert r.returncode != 0 and 'STOP  S12: no passing freeze-2 seed audit' in r.stdout, r.stdout[-3000:]
        assert len(stops('audit_gate_failed')) == 5 and not stops('dependency_incomplete')
        n += 2
        # 3. --audit-ok cannot stand in for a missing report
        r = _launch(d, '--audit-ok')
        assert r.returncode == 1 and 'REFUSING --audit-ok' in r.stdout, r.stdout[-2000:]
        assert not [x for x in _ledger_of(d) if x.get('reason') == 'audit_classified']
        n += 1
        # 4. a report with a RESULT-DATA hit: a stop, and not acknowledgeable
        _audit_report(rp, **dict(base, gate_pass=False, clean=False, data_files=['x.pkl']))
        assert _launch(d, '--audit-ok').returncode == 1
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and len(stops('audit_gate_failed')) == 6, r.stdout[-3000:]
        assert not [x for x in _ledger_of(d) if x.get('reason') == 'audit_classified']
        n += 1
        # 5. a report made BEFORE the seal (the freeze-1 audit, or a stale one)
        _audit_report(rp, **dict(base, started='2026-01-01 00:00:00 UTC'))
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and 'FAIL it started after the seal' in r.stdout, r.stdout[-3000:]
        n += 1
        # 6. hits left to classify: a stop of its own, until the recorded action
        _audit_report(rp, **dict(base, clean=False, text_hits=[52]))
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and 'seed audit lists hits that are not classified' in r.stdout, \
            r.stdout[-3000:]
        assert len(stops('audit_unclassified')) == 1
        r = _launch(d, '--audit-ok')
        assert r.returncode == 0 and 'AUDIT-OK %s sha256 %s' % (report, _sha256(rp)[:16]) in r.stdout, \
            r.stdout[-2000:]
        notes = [x for x in _ledger_of(d) if x.get('reason') == 'audit_classified']
        assert len(notes) == 1 and notes[0]['report_sha256'] == _sha256(rp) and notes[0]['event'] == 'note'
        r = _launch(d, '--stages', 'S7')
        assert '[AUDIT-GATE] CLASSIFIED' in r.stdout, r.stdout[-3000:]
        # the gate passed: the launcher went on to the stage's own rule
        assert 'STOP  S7: hard dependency S6 is not complete' in r.stdout and r.returncode != 0
        assert len(stops('dependency_incomplete')) == 1
        n += 1
        # ... the acknowledgement is bound to that report: a changed one is unclassified again
        _audit_report(rp, **dict(base, clean=False, text_hits=[52, 53]))
        r = _launch(d, '--stages', 'S7')
        assert r.returncode != 0 and len(stops('audit_unclassified')) == 2, r.stdout[-3000:]
        n += 1
        # 7. the CLEAN report of this sealed tree passes, and is recorded
        _audit_report(rp, **base)
        r = _launch(d, '--stages', 'S7')
        assert '[AUDIT-GATE] PASS' in r.stdout and 'STOP  S7: hard dependency S6 is not complete' in r.stdout, \
            r.stdout[-3000:]
        passed = [x for x in _ledger_of(d) if x.get('event') == 'audit_gate']
        assert len(passed) == 2 and passed[-1]['status'] == 'pass' \
            and passed[-1]['report_sha256'] == _sha256(rp) and passed[-1]['gate'] == 'r3-freeze2', passed
        r = _launch(d, '--audit-ok')
        assert r.returncode == 0 and 'nothing to acknowledge' in r.stdout
        assert len([x for x in _ledger_of(d) if x.get('reason') == 'audit_classified']) == 1
        n += 1
        # 8. a dry run names the gate and evaluates nothing
        r = _launch(d, '--dry-run', '--stages', 'S7')
        assert r.returncode == 0 and 'PLAN  seed-audit gate r3-freeze2 before S7' in r.stdout, r.stdout[-3000:]
        assert '[AUDIT-GATE]' not in r.stdout
        n += 1
        assert n == 12
        # not one evaluation pass ran in this test
        assert not [x for x in _ledger_of(d) if x.get('event') == 'start'
                    and x.get('stage') not in ('S0', 'S3b', 'S5')]
        assert not [f for dp, _dn, fs in os.walk(d) for f in fs if f.endswith('.pkl')]
    # NEGATIVE CONTROL of the harness: the unpatched rehearsal has no gate
    # (the other launcher tests start S6 on it without an audit report)
    assert _plan_cli('--profile', REH, '--audit-gate').stdout.strip() == ''


# ====================================== the documents the freeze is run from
def _doc(name):
    with open(os.path.join(ROOT, 'docs', name), encoding='utf-8') as f:
        return f.read()


def _r3_checklist():
    parts = _doc('FREEZE_CHECKLIST.md').split('## R3-lite — freezing and launching profile `r3lite`')
    assert len(parts) == 2, 'one R3-lite section'
    return parts[1]


def test_the_checklist_launches_every_r3lite_chain_with_the_launcher_environment():
    """Review of 2026-10-11: steps R.2, R.4 and R.6 launched the chain as a
    bare `nohup bash confirmatory/run_final.sh ...`.  The preflight checks
    PYTHONHASHSEED and the thread pins in the LAUNCHER's own process, a
    non-interactive ssh shell has none of them, the refusal is permanent, and
    the failed preflight has by then pinned the tree digest in the ledger.
    The launcher tests cannot see this (they set the variables themselves),
    so the procedure text is checked here."""
    sec = _r3_checklist()
    import preflight as PF
    # the environment file: what the preflight requires of the launcher
    head = "cat > ~/$RT/logs/freeze/r3lite_chain.env\" <<'EOF'\n"
    assert sec.count(head) == 1
    env = sec.split(head, 1)[1].split('\nEOF\n', 1)[0]
    lines = [l for l in env.splitlines() if l and not l.startswith('#')]
    assert 'export PYTHONHASHSEED=0' in lines, env
    n = 0
    for v in PF.THREAD_VARS:
        assert any(l.startswith('export ') and ('%s=1' % v) in l.split() for l in lines), (v, env)
        n += 1
    assert n >= 3, PF.THREAD_VARS
    assert any(l.startswith('for v in $(compgen -e | grep') and 'unset' in l for l in lines), env
    assert 'export TMPDIR=<server-tmp>' in lines
    # every chain launch: registered with the watchdog, that file, the long budget
    launches = [l for l in sec.splitlines()
                if 'run_final.sh --profile r3lite' in l and '--audit-ok' not in l
                and not l.lstrip().startswith('#')]
    assert len(launches) == 3, launches
    for l in launches:
        assert l.strip().startswith('"bash confirmatory/run_final.sh --profile r3lite'), l
        assert '--preflight-retries 12 --preflight-wait 600"' in l, l
    reg = [l for l in sec.splitlines() if '~/MARL_watch/register_chain.sh ' in l and 'rc_r3lite_' in l]
    assert [l.split('register_chain.sh ')[1].split()[0] for l in reg] == \
        ['rc_r3lite_reh', 'rc_r3lite_seg1', 'rc_r3lite_seg2'], reg
    env_args = [l for l in sec.splitlines() if ' logs/freeze/r3lite_chain.env ' in l and 'CHAIN END' in l]
    assert len(env_args) == 3, env_args
    assert '"CHAIN END profile=r3lite_rehearsal rc=0"' in env_args[0]
    assert all('"CHAIN END profile=r3lite rc=0"' in l for l in env_args[1:])
    assert '--stages S0,S3b,S5 ' in launches[1] and '--from S6 ' in launches[2]
    # NEGATIVE CONTROLS: the pre-review forms are gone, and the dry-run trap is named
    assert 'nohup bash confirmatory/run_final.sh' not in sec
    assert not [l for l in sec.splitlines() if 'run_final.sh' in l and '--dry-run' in l
                and 'Do NOT' not in l and 'do not' not in l.lower()], 'no dry run is prescribed'
    assert 'Do NOT use `run_final.sh --dry-run`' in sec
    # the host check precedes every launch, and the tree is a parameter
    assert sec.count('R.2.0') >= 4
    assert 'RT=MARL_r3lite ' in sec and 'R3="$SP/r3a"' in sec and 'SYNC_TREE=r3a SYNC_REMOTE=$RT' in sec
    assert '~/MARL_r3/' not in sec.replace('$HOME/MARL_r3/r3out/phase0', '') and '~/MARL_r3lite' not in sec, \
        'the run tree is read from $RT; only the Phase 0 tree is named literally'
    # D-R3.3: the plan summaries, the acceptance by count, the order of segment 2
    assert '`profile r3lite_rehearsal: 14 invocations, 84 passes`' in ' '.join(sec.split())
    assert '`profile r3lite: 20 invocations, 804 passes`' in ' '.join(sec.split())
    for iid, want in (('r3lite_rehearsal', (14, 84)), ('r3lite', (20, 804))):
        invs = CP.build(iid)
        assert (len(invs), sum(i['expected_passes'] for i in invs)) == want, (iid, want)
    assert "'PASS' if n == 16 and not fails" in sec and 'len(invs) == 14' in sec
    assert sec.count('need(') == 16 + 1, 'sixteen acceptance checks and the helper itself'
    assert 'S9x (the ensemble' in ' '.join(sec.split())
    # THE CONTINGENCY (R3 section 13): the two commands that carry family 1 to the end when
    # family E cannot be completed -- as comments, since neither is launched unless the ledger
    # records the double failure; with the launcher budget of every other launch; and the
    # first draft's command, which stopped at S7e, is gone
    cont = [l for l in sec.splitlines() if l.startswith('#   bash confirmatory/run_final.sh --profile r3lite ')]
    assert len(cont) == 2, cont
    assert ' --stages S7,S6h,S8,S12 --waive-order S7,S12 ' in cont[0], cont[0]
    assert ' --stages S6h,S8,S12 --waive-order S12 ' in cont[1], cont[1]
    for l in cont:
        assert l.endswith(' --preflight-retries 12 --preflight-wait 600'), l
    assert '--from S7 --waive-order S7' not in sec
    # ... and each stage it names exists, and each waiver it asks for is one the plan declares
    for l in cont:
        stages = l.split('--stages ')[1].split()[0].split(',')
        waived = l.split('--waive-order ')[1].split()[0].split(',')
        assert set(stages) <= set(CP.stage_order(R3)) and set(waived) <= set(CP.waivable_order_stages(R3))
        assert 'S7e' not in stages and 'S9x' not in stages, 'family E is not run in another form'
        for st in stages:
            deps, order = CP.stage_deps(st, R3)
            assert not (set(deps) & {'S7e', 'S9x'}), (st, deps)
            for o in order:
                assert o in stages or o in ('S7', 'S6') or st in waived, (st, o)
    # the anchor publishes the two whole files R3 section 13 names, and no extract
    flat = ' '.join(sec.split())
    assert 'two whole files, the same two R3 §13 names' in flat
    assert 'entries D-R3.1–D-R3.3' not in flat and 'No extract of `docs/DECISIONS.md`' in flat
    # the ensemble code path is compared with Phase 0's passes of the same instances
    assert 'phase0/out/ens_dev.pkl output/r3lite_rehearsal/s9x_ensemblefull.pkl' in sec
    assert sorted(CP.by_id(REH, 's9x_ensemblefull')['seeds']) == [52, 53] and '--seeds 52,53 --keys all' in sec


def test_the_preregistration_states_what_the_review_required():
    """The disclosures and limits the review of 2026-10-11 required of
    docs/PREREGISTRATION_R3.md, pinned so that a later edit cannot drop one
    silently.  This checks that the statements are PRESENT; whether they are
    adequate is the author's and the reviewers' judgement."""
    doc = _doc('PREREGISTRATION_R3.md')
    flat = ' '.join(doc.split())
    n = 0
    for needle in (
            # the selection history: the guard Phase 0 recommended, and what it would have picked
            'relay-churn guard', 'the guarded rule selects the ensemble', 'NO-GO',
            # six rule runs, five candidates
            'six rule runs',
            # R2's result is not a tie
            'not distinguishable from XL-DET', 'neither strengthened nor weakened',
            # every confirmatory look at the question, counted; the last attempt
            'four times in three studies', 'the last confirmatory attempt at it in this manuscript',
            'for both families jointly',
            # the comparator is improvable
            'XL-DET as implemented at `6b53f3f`', 'hindsight', 'not simulated',
            'not evidence that learning, rather than a power setting',
            'the engineered state of the art',
            # the classes of an audit hit
            '(c) coincidence', '(d) negative statement', 'by the author, in writing',
            # the two families and how they are presented (D-R3.3)
            'Both families are reported in every outcome', 'Family 1 is the primary family',
            'The ensemble does not replace replicate 1234',
            'The primary family did not confirm the hypothesis',
            'not** corrected for each other', 'below **0.025**',
            'at most 0.10',
            # review of 2026-10-12 (D-R3.4): one numbering of the looks and R2's wording for the
            # first test; what the author had seen and that he took the recommended options;
            # the ensemble's own history; the qualifier of the guarded rule
            'One numbering is used throughout', "superseded by R2 as the paper's confirmatory evidence",
            'marked as recommended', 'No ensemble was run on seeds 80–99', 'at g = 2',
            # ... the joint sentence names the two policies; "not shown worse" needs an endpoint
            # that could have shown something
            'the two policies tested here', 'at least 8 non-zero differences',
            # ... the contingency is read from the ledger and reports no ensemble figure; what
            # the operator cannot do; what the anchor publishes
            'read from the ledger', 'computes no per-instance listing of the ensemble',
            'What the operator cannot do', 'two whole files, the list of redactions',
            # ... the restart rule is counted from the ledger; the joint sentence and the spread
            "The count is the ledger's, not a launch's", 'is never started again',
            'without the spread of the four replicates beside it',
            # the ensemble's stability family is expected to assess little
            'S9 could not assess', 'is not used as a label for either policy',
            # how family E is analysed, and what happens if it cannot be completed
            'tools/assemble_family.py', 'must reproduce the S7 report leaf by leaf',
            'If family E cannot be completed', 'never shed, skipped or repeated for any content of any report',
            # the design history and the enlarged block
            'History of this design', 'left the reserve',
            # power: the script, the joint probabilities, both families
            'docs/POWER_R3LITE.md', 'P1 and P4 both', 'E-P1 and E-P4 both',
            # no claim about the selection rule
            'outperforms R2\'s rule',
            # the external anchor
            'third-party timestamp'):
        assert needle in flat, 'missing from PREREGISTRATION_R3.md: %r' % needle
        n += 1
    assert n == 51
    # NEGATIVE CONTROLS: what the first draft said of the ensemble, and its
    # block, survive only as history (section 1.4), not as the design
    for gone in ('**It decides nothing.** No hypothesis of §8 or §9 is about the ensemble',
                 'n = 20 instances per scenario', 'Total 404 passes', 'seeds 300–319.',
                 'to be entered by the author',
                 # ... and what the review of 2026-10-12 had removed
                 'withdrawn', 'learned policies of this training run', 'between 0.05 and 0.10',
                 'to register the rule without a guard', '--from S7 --waive-order S7',
                 'is rejected and the analysis is repeated', 'in the lead\'s words as they were put to him',
                 'one test, one run', 'two whole files and nothing else',
                 'no M2 or M7 endpoint was assessable', "three of R2's own full-length passes"):
        assert gone not in flat, 'a statement the reviews removed survives: %r' % gone
    # the contingency commands of section 13 are the checklist's, and the stage graph's
    assert '--stages S7,S6h,S8,S12 --waive-order S7,S12' in flat and '--stages S6h,S8,S12 --waive-order S12' in flat
    # the option the author selected is quoted whole, as D-R3.3 records it
    dec = ' '.join(_doc('DECISIONS.md').split())
    quote = ('1234 primary + combined second (Recommended) — Seed 1234 is primary (largest margin, '
             'about 8x relay churn). The combined policy is a second pre-registered family with its own '
             'correction (stable bridging, 88% chance on both scenarios). 20 seeds, about 12 h of compute.')
    assert quote in dec and quote in flat.replace('*', '')
    # the block and the reserve are the plan's
    assert 'Topology seeds 300–339' in doc and 'held in reserve (340–359, 400–439)' in flat
    assert CP.R3_CONFIRMATORY_SEEDS == list(range(300, 340))
    # the power script the text pins is the script in docs/POWER_R3LITE.md
    pw = _doc('POWER_R3LITE.md').split('```python\n', 1)[1].split('\n```', 1)[0] + '\n'
    sha = hashlib.sha256(pw.encode('utf-8')).hexdigest()
    assert sha in doc, 'PREREGISTRATION_R3 section 5.1 does not pin the script of POWER_R3LITE.md (%s)' % sha
    # "tie" is not used for R2's result anywhere in the text (the one use of
    # the word that stays is the Wilcoxon test's handling of tied ranks)
    import re
    rest = flat.replace('mid-ranks for ties', '')
    assert flat.count('mid-ranks for ties') == 1
    assert not re.findall(r'\b(tie|tied|ties)\b', rest), \
        re.findall(r'.{40}\b(?:tie|tied|ties)\b.{40}', rest)
    # the candidates of the SELECTED record are the five of section 3.2
    with open(os.path.join(ROOT, 'confirmatory', 'SELECTED_r3lite.txt'), encoding='utf-8') as f:
        sel = f.read()
    assert '# candidates ranked: 5' in sel and 'relay-churn guard' in sel
    rows = [l.split() for l in sel.splitlines()
            if l.startswith('#   s') or l.startswith('#   ensemble')]
    assert [r[1] for r in rows] == ['s1234', 'ensemble', 's3456', 's4567', 's2345'], rows
    m = 0
    for r, lead in zip(rows, ('| 1 | **replicate 1234**', '| 2 | ensemble of the four',
                              '| 3 | replicate 3456', '| 4 | replicate 4567', '| 5 | replicate 2345')):
        line = [l for l in doc.splitlines() if l.startswith(lead)]
        assert len(line) == 1, lead
        # the same A, B and min scores, to the table's two decimals
        for v in r[2:5]:
            shown = ('%+.2f' % float(v)).replace('-', '−')
            assert shown in line[0], (lead, v, shown)
            m += 1
    assert m == 15
    # the chain stage of the ensemble is S9x; S9e is the endpoint only
    assert 'chain S9x' in flat and 'chain S9e' not in flat
