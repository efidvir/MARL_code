"""WPDOC/WP9: the analysis must report the PRE-REGISTERED arm.

`--marl-arm` selects the arm reported as MARL-RIC.  Its default used to be
`marl_freeze`, the secondary adapt-then-freeze variant (S7), so an analysis
run without the flag produced a complete, plausible P1-P9 and S9 report for
the wrong arm under the label MARL-RIC.  Pre-registration R2 section 10 makes
`config.marl_arm == marl_static` a hard assert and section 12 pins the
invocation; these tests hold both ends of that.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import confirmatory_analysis as CA  # noqa: E402
import test_wp9_synth as SY  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
MARL_ARMS = ('marl_static', 'marl_freeze', 'xldet', 'random')
BASE_ARMS = ('sdn', 'ospf', 'olsr', 'batman', 'aodv')


def _set(tmp):
    a = SY.make_pickle(os.path.join(tmp, 'marl.pkl'), MARL_ARMS, SEEDS)
    b = SY.make_pickle(os.path.join(tmp, 'base.pkl'), BASE_ARMS, SEEDS)
    m = SY.make_manifest(os.path.join(tmp, 'manifest.json'), SEEDS)
    return [a, b], [m]


def _analyse(pickles, manifests, **kw):
    kw.setdefault('t_sev', SY.T_SEV_TEST)
    kw.setdefault('t_total', SY.T_TEST)
    with SY.short_run_patch():
        return CA.analyse(pickles, manifests, **kw)


def test_pre_registered_arm_is_the_default_and_is_recorded():
    """The CLI default is the pre-registered arm, not the secondary one."""
    assert CA.PREREG_MARL_ARM == 'marl_static'
    src = open(os.path.join(ROOT, 'confirmatory_analysis.py'), encoding='utf-8').read()
    assert "ap.add_argument('--marl-arm', default=PREREG_MARL_ARM" in src, \
        'the --marl-arm default must be the pre-registered arm'
    tmp = SY.mkdtemp('wp9_arm_')
    pk, mf = _set(tmp)
    rep = _analyse(pk, mf, marl_arm='marl_static', strict=True)
    assert rep['config']['marl_arm'] == 'marl_static'
    assert 'marl_static' in rep['config']['arms']
    chk = rep['asserts']['checks']['prereg_marl_arm']
    assert chk['expected'] == 'marl_static' and chk['marl_arm'] == 'marl_static', chk
    assert rep['asserts']['n_failures'] == 0, rep['asserts']['failures']


def test_a_report_for_another_arm_is_refused():
    """The secondary arm under the MARL-RIC label is a hard-assert failure,
    so the script refuses to report instead of producing a plausible one."""
    tmp = SY.mkdtemp('wp9_arm_')
    pk, mf = _set(tmp)
    try:
        _analyse(pk, mf, marl_arm='marl_freeze', strict=True)
    except AssertionError as e:
        msg = str(e)
        assert 'marl_freeze' in msg and 'marl_static' in msg, msg
        assert '--marl-arm marl_static' in msg, msg
    else:
        raise AssertionError('analysing the secondary arm as MARL-RIC must be refused')
    # the report for the wrong arm would otherwise be complete and plausible:
    # with --no-assert it is produced, and the failure is recorded in it
    rep = _analyse(pk, mf, marl_arm='marl_freeze', strict=False)
    assert len(rep['primary']) == 9 and list(rep['stability']) == [
        'S9a', 'S9b', 'S9c', 'S9d', 'S9e', 'S9f'], 'the wrong-arm report is complete'
    fails = rep['asserts']['failures']
    assert any('pre-registered MARL-RIC arm' in f for f in fails), fails
    assert rep['config']['marl_arm'] == 'marl_freeze'
    assert rep['asserts']['checks']['prereg_marl_arm']['marl_arm'] == 'marl_freeze'
