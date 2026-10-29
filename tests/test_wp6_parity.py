# -*- coding: utf-8 -*-
"""WP6-FIX3 -- the minted checkpoint, and a tripwire on the WP2 parity
invariant.

WP6's strict schema gate refuses `ce_ck_s1234/policy_final.pt` (fc1 62 wide,
no provenance block).  That is CORRECT -- the file predates the WP6
observation schema and has to be retrained -- but it is also the checkpoint
`tests/test_wp2_parity.py` fed to both drivers, so the RERUN_SPEC WP2
invariant

    the same seed and checkpoint through both drivers for 300 ticks give
    identical action, delivered and fragment series

stopped being executed anywhere: both parity tests died at the gate before
any comparison was made.  A specification requirement that silently stops
being tested is the exact failure this rerun exists to eliminate.

The fix has two halves:

  * `sixg_sim.agent.mint_schema_current_checkpoint` builds a schema-current
    checkpoint from this tree's network definition, and
    `tests/test_wp2_parity.py` now mints one for itself instead of dying.
    Checks 1 and 2 below are that helper's contract.
  * Check 3 is the TRIPWIRE: it reads `tests/test_wp2_parity.py` and fails
    if the 300-tick two-driver comparison, or any of the three series it
    asserts on, is removed again.  The parity pass itself is NOT duplicated
    here -- it is minutes of simulator time, and running it twice would buy
    nothing.
  * Checks 4 and 5 are that no test file anywhere in the tree can pass
    quietly: `tests/` AND the root-level contract scripts, which are the
    files that actually carry the environment guards.

Fast: no simulator pass.  Run

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp6_parity
"""
import glob
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


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

WP2 = os.path.join(HERE, 'test_wp2_parity.py')


def _src(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _driver():
    """run_timeline_comparison, imported with a neutral argv."""
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        import run_timeline_comparison as R
    finally:
        sys.argv = argv
    return R


# ── 1. the minted checkpoint is one this tree can actually load ──────────
def test_minted_checkpoint_is_schema_current_and_loads():
    import torch
    from sixg_sim.agent import (ACTOR_OBS_DIM, PolicyNetwork, SCHEMA_ID,
                                assert_checkpoint_schema,
                                checkpoint_schema_id,
                                load_policy_state_dict,
                                mint_schema_current_checkpoint)

    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        path = mint_schema_current_checkpoint(os.path.join(d, 'a', 'p.pt'),
                                              torch_seed=1234)
        assert os.path.exists(path), 'nothing written'
        ck = torch.load(path, map_location='cpu', weights_only=False)

        # stamped with THIS tree's id, and the gate accepts it
        assert checkpoint_schema_id(ck) == SCHEMA_ID, ck.get('metadata')
        assert assert_checkpoint_schema(ck, where=path) == SCHEMA_ID

        # the weights are this tree's actor, and they load STRICTLY
        assert tuple(ck['state_dict']['fc1.weight'].shape) == (128,
                                                               ACTOR_OBS_DIM)
        net = PolicyNetwork()
        load_policy_state_dict(net, ck['state_dict'])      # strict by default

        # deterministic in the torch seed, and it does not move the global RNG
        torch.manual_seed(99)
        before = torch.randn(4)
        torch.manual_seed(99)
        p2 = mint_schema_current_checkpoint(os.path.join(d, 'b.pt'),
                                            torch_seed=1234)
        after = torch.randn(4)
        assert torch.equal(before, after), (
            'minting moved the global torch RNG state, so it would shift the '
            'draws of any run started after it')
        sd2 = torch.load(p2, map_location='cpu',
                         weights_only=False)['state_dict']
        assert all(torch.equal(ck['state_dict'][k], sd2[k]) for k in sd2), (
            'the same torch seed gave different weights')

        # and the gate still refuses it once the stamp is removed
        stripped = dict(ck)
        stripped['metadata'] = {k: v for k, v in ck['metadata'].items()
                                if k != 'provenance'}
        refused = False
        try:
            assert_checkpoint_schema(stripped, where='stripped')
        except ValueError:
            refused = True
        assert refused, 'the gate accepted a checkpoint with no schema id'


# ── 2. the evaluation driver reads it as a single shared policy ──────────
def test_the_driver_reads_a_minted_checkpoint_as_the_legacy_format():
    """`run_physics_pass` chooses between the per-agent and the legacy
    single-policy branch with
    `all(isinstance(v, dict) for v in checkpoint.values())`.  A bare
    two-key {state_dict, metadata} file trips that test by accident and
    would be loaded through the per-agent fallback path; the minted format
    carries a non-dict `version`, so the choice is unambiguous."""
    import torch
    from sixg_sim.agent import mint_schema_current_checkpoint

    R = _driver()
    with tempfile.TemporaryDirectory(dir=_tmp_base()) as d:
        path = mint_schema_current_checkpoint(os.path.join(d, 'p.pt'),
                                              torch_seed=1234)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert set(ck) == {'state_dict', 'version', 'metadata'}, sorted(ck)
        stripped = R._strip_reserved_keys(ck)
        assert not R._is_per_agent_checkpoint(stripped), (
            'the minted checkpoint is mis-read as a per-agent checkpoint')


# ── 3. TRIPWIRE: the WP2 parity invariant is still exercised ─────────────
def test_wp2_parity_invariant_is_still_exercised():
    """RERUN_SPEC WP2: 'A parity test runs the same seed/checkpoint through
    both drivers for 300 ticks and requires identical action, delivered and
    fragment series.'  WP6 broke it by leaving the test on a checkpoint the
    new gate refuses, and nothing noticed, because a test that dies at an
    unrelated line still LOOKS like a test.  This check reads the file."""
    src = _src(WP2)

    assert 'TICKS = 300' in src, 'the 300-tick horizon is gone'
    assert 'def test_parity_training_step_vs_harness_step(' in src, (
        'the parity test function is gone')
    body = src.split('def test_parity_training_step_vs_harness_step(')[1]
    body = body.split('\ndef ')[0]
    # trackAgent merge: master's version (both scenarios) factors the
    # training-driver run and the series comparison into two helpers the
    # test calls.  Follow each helper the body CALLS -- a helper that exists
    # but is not called adds nothing.
    followed = 0
    for helper in ('_eval_series', '_assert_series_parity'):
        if (helper + '(') in body and ('def %s(' % helper) in src:
            body += src.split('def %s(' % helper)[1].split('\ndef ')[0]
            followed += 1
    assert followed in (0, 2), (
        'the parity test calls only one of its two helpers', followed)

    # both drivers, on the same seed and the same checkpoint
    assert 'run_physics_pass(' in src, 'the harness driver is no longer run'
    assert 'run_eval_episode(' in body, (
        'the training driver is no longer run in the parity test')
    assert 'SEED' in body, 'the parity test no longer pins the seed'

    # the three series RERUN_SPEC names, each actually compared
    for key in ('action_sig', 'delivered', 'fragments'):
        assert ("'%s'" % key) in body, (
            'the %r series is no longer compared in the parity test' % key)
    assert body.count('assert') >= 5, (
        'the parity test lost most of its assertions: %d left'
        % body.count('assert'))

    # it builds a schema-current checkpoint for itself rather than using a
    # pre-WP6 one the gate refuses (which is how the invariant went dark)
    assert 'mint_schema_current_checkpoint' in src, (
        'tests/test_wp2_parity.py no longer mints its own checkpoint, so it '
        'will die at the schema gate again')
    # the CKPT binding itself, not the prose around it, which names the
    # refused replicate on purpose
    ckpt_lines = [l for l in src.splitlines() if l.startswith('CKPT')]
    assert ckpt_lines, 'tests/test_wp2_parity.py no longer defines CKPT'
    assert not any('ce_ck' in l for l in ckpt_lines), (
        'CKPT points at a shipped pre-WP6 checkpoint again; the WP6 gate '
        'refuses those until they are retrained: %r' % ckpt_lines)


# ── 4. NO test file anywhere can pass quietly by skipping ────────────────
# A skip that pytest understands.  Never allowed, in either directory.
SKIP_IDIOMS = ('pytest.skip', 'pytest.mark.skip', 'unittest.skip',
               'SkipTest', 'pytest.importorskip')

# An early exit that is not a pytest skip.  The root-level contract scripts
# are standalone programs, not pytest modules, and they guard on environment
# variables the check cannot run without.  Each one that does is named here
# with the reason it is VISIBLE rather than quiet; a file that grows one
# without an entry fails.  (The first version of this test scanned `tests/`
# only, so these ten scripts -- the files that actually carry the guards --
# were outside the tripwire it claimed to be.)
EARLY_EXIT_IDIOMS = ('sys.exit(2)', '-- skipped')
ALLOWED_EARLY_EXITS = {
    'test_mission_config.py':
        'sys.exit(2) unless MARL_RELAY_ENTROPY_FLOOR=0.01: the entropy '
        'floor IS part of the thing under test (item 6), so without it '
        'there is nothing to run.  Exit 2 is non-zero -- a shell, a CI step '
        'and tests/run_tests.py all report it as a failure -- and the '
        'required command line is printed.  (WP5 removed the fragment cue, '
        'so MARL_POSTCARD_FRAGMENTS=1 is refused rather than required; WP8 '
        'removed MARL_REWARD_PROFILE from this guard: the reward '
        'coefficients are frozen, so there is no profile to require.)',
    'test_relay_repoint.py':
        'sys.exit(2) unless MARL_RELAY_REPOINT=fragment_aware, with the '
        'required command line printed.  Same shape, same visibility.',
}


def _scanned_test_files():
    """Every test file in the tree: tests/ AND the root-level contract
    scripts.  The root scripts are the ones that carry environment guards,
    so a scan of tests/ alone is a tripwire pointed away from the traffic."""
    me = os.path.abspath(__file__)
    paths = (glob.glob(os.path.join(HERE, 'test_*.py'))
             + glob.glob(os.path.join(ROOT, 'test_*.py')))
    # this file spells the idioms out in order to look for them
    return sorted(p for p in paths if os.path.abspath(p) != me)


def test_no_test_file_skips_quietly():
    """tests/run_tests.py has no skip concept -- an exception is a FAIL --
    but the same files also run under pytest, where a skip would turn a
    missing contract into a green run.  Neither is allowed here: a contract
    that cannot run must FAIL loudly."""
    offenders = []
    for path in _scanned_test_files():
        src = _src(path)
        for idiom in SKIP_IDIOMS:
            if idiom in src:
                offenders.append('%s: %s' % (os.path.basename(path), idiom))
    assert not offenders, (
        'a test can skip instead of failing: %s' % offenders)


def test_every_early_exit_guard_is_declared():
    """The root contract scripts exit early rather than skip.  That is
    allowed -- a non-zero exit with the reason printed is not a quiet pass --
    but only for the files named in ALLOWED_EARLY_EXITS, with the reason
    recorded, so the claim 'nothing skips quietly' is checked and not just
    asserted."""
    found = {}
    for path in _scanned_test_files():
        src = _src(path)
        hits = [i for i in EARLY_EXIT_IDIOMS if i in src]
        if hits:
            found[os.path.basename(path)] = hits

    undeclared = sorted(set(found) - set(ALLOWED_EARLY_EXITS))
    assert not undeclared, (
        'a test file exits early without an entry in ALLOWED_EARLY_EXITS: '
        '%s -- either make it fail loudly or record why the early exit is '
        'visible' % [(f, found[f]) for f in undeclared])

    # the table cannot go stale either: a named file must still exist and
    # must still carry the idiom it was excused for
    for name, why in ALLOWED_EARLY_EXITS.items():
        assert os.path.exists(os.path.join(ROOT, name)), (
            'ALLOWED_EARLY_EXITS names %s, which is not in the tree' % name)
        assert name in found, (
            '%s no longer exits early -- remove its ALLOWED_EARLY_EXITS '
            'entry' % name)
        assert len(why) > 40, name
