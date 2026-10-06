#!/usr/bin/env python3
"""Negative controls for the FIX-PREFREEZE behaviour fixes (DECISIONS.md D-FIX.*).

    python3 tools/prefreeze_negative_controls.py [--keep] [--list] [MUTATION ...]

The method and the runner are those of tools/wp13_negative_controls.py (and
of tools/wp4_negative_controls.py before it): for every mutation, copy the
tree's sources into a scratch directory under output/prefreeze_negctl/
(removed at the end), re-introduce ONE defect -- the behaviour before the fix
-- by an exact string replacement whose anchor must occur EXACTLY ONCE, run
the named tests against the copy in a fresh interpreter, and report which
fail.  The unmutated copy runs first and must be fully green, so a failure
under a mutation is attributable to that mutation alone.

A control is CAUGHT when every test it lists under `must_fail` fails.  The
run exits 0 only when the head is green, at least one control ran, every
mutation applied and every control was caught.  Every line of the report
counts what it checked.

The controls:

  s10_omniscient          D-FIX.1  S10 back to MARL_DETECT_TICKS=0 (the
                                   omniscient ablation): the S10 test must
                                   fail.
  s12_no_untrained        D-FIX.2  the S12 step stops passing the untrained
                                   controls (--untrained) to the analysis.
  s12_no_freeze           D-FIX.2  the S12 step stops reading the chain-S9
                                   pickle (secondary S7 not computed).
  s12_s8_vacuous_passes   D-FIX.2  a vacuous S8 check (exit 2) is taken
                                   for a pass.
  freeze_old_reward       D-FIX.3  the online arm mixes 0.5 * local + 0.5 *
                                   the hand-rolled connectivity term again.
  freeze_silent_collect   D-FIX.4  `try: collect(...) except Exception:
                                   pass` restored around the online arm's
                                   collect call.
  job_env_not_cleared     D-FIX.5  the job script no longer re-executes
                                   under `env -i`: the caller's variables
                                   reach the invocation.
  marl_env_not_compared   D-FIX.5  the completeness gate ignores the
                                   recorded marl_env.
  job_env_check_vacuous   D-FIX.5  the job-environment check passes
                                   whatever it found.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import wp13_negative_controls as NC  # noqa: E402

# test file -> the tests run against every copy
TESTS = {
    'tests/test_run_chain.py': (
        'test_s10_runs_installation_off_not_the_omniscient_ablation',
        'test_the_two_sensitivities_differ_only_by_one_variable',
        'test_an_ablation_is_compared_with_the_baseline_at_its_own_tick_count',
        'test_the_plan_declares_the_inputs_of_s8_s12_and_secondary_s7',
        'test_s12_computes_s8_s12_and_secondary_s7_from_the_chains_own_pickles',
        'test_s12_fails_loudly_when_an_input_is_missing_or_an_analysis_is_absent',
    ),
    'tests/test_prefreeze_online_arm.py': (
        'test_the_adapt_then_freeze_arm_trains_on_the_frozen_reward',
        'test_a_refused_transition_in_the_online_arm_is_loud',
    ),
    'tests/test_run_chain_launcher.py': (
        'test_a_stray_variable_of_the_operators_shell_never_reaches_an_invocation',
        'test_the_job_environment_check_refuses_a_leaking_or_wrong_environment',
        'test_the_completeness_gate_requires_the_recorded_marl_env_to_be_the_plans',
    ),
}

# name -> (file, anchor, replacement, {test file: tests that MUST fail}, what)
MUTATIONS = {
    's10_omniscient': (
        'confirmatory/chain_plan.py',
        "env={'MARL_LS_INSTALL': '0'}, shed_group='sensitivity',",
        "env={'MARL_DETECT_TICKS': '0'}, shed_group='sensitivity',  # NEGCTL",
        {'tests/test_run_chain.py': (
            'test_s10_runs_installation_off_not_the_omniscient_ablation',
            'test_the_two_sensitivities_differ_only_by_one_variable')},
        'S10 runs MARL_DETECT_TICKS=0 again (before D-FIX.1): the link-state '
        'service removed, not installation off'),
    's12_no_untrained': (
        'confirmatory/stage_analysis.sh',
        '      [ -n "$UN" ] && ARGS+=(--untrained "$UN")\n',
        '      : # NEGCTL\n',
        {'tests/test_run_chain.py': (
            'test_s12_computes_s8_s12_and_secondary_s7_from_the_chains_own_pickles',)},
        'the untrained controls are not handed to the analysis (before '
        'D-FIX.2): analysis S12 is never computed'),
    's12_no_freeze': (
        'confirmatory/stage_analysis.sh',
        '      ARGS=(--pickles "$CF${FZ:+,$FZ}" --manifests "$CFMAN" --marl-arm marl_static\n',
        '      ARGS=(--pickles "$CF" --manifests "$CFMAN" --marl-arm marl_static\n',
        {'tests/test_run_chain.py': (
            'test_s12_computes_s8_s12_and_secondary_s7_from_the_chains_own_pickles',)},
        'the chain-S9 pickle is not read (before D-FIX.2): secondary S7 is '
        'never computed'),
    's12_s8_vacuous_passes': (
        'confirmatory/stage_analysis.sh',
        '        *) fail "the S8 check verified nothing (exit $S8_RC: nothing compared or the hash-seed precondition not established)" ;;\n',
        '        *) S8_STATE=identical ;;  # NEGCTL\n',
        {'tests/test_run_chain.py': (
            'test_s12_fails_loudly_when_an_input_is_missing_or_an_analysis_is_absent',)},
        'a vacuous S8 check (exit 2: nothing compared, or the hash-seed '
        'precondition not established) is recorded as identical'),
    'freeze_old_reward': (
        'run_timeline_comparison.py',
        '                r = _RS.mix(comps.total_reward, team_r)\n',
        '                r = (0.5 * comps.total_reward) + (0.5 * (  # NEGCTL\n'
        '                    0.4 * observations[aid].connectivity.reachable_ue_fraction +\n'
        '                    0.3 * observations[aid].connectivity.intra_island_reachability +\n'
        '                    0.3 * observations[aid].connectivity.ue_to_ue_routed_norm))\n',
        {'tests/test_prefreeze_online_arm.py': (
            'test_the_adapt_then_freeze_arm_trains_on_the_frozen_reward',)},
        'the online arm mixes 0.5 * local + 0.5 * (0.4 reach + 0.3 intra + '
        '0.3 routed) again (before D-FIX.3, D8.11)'),
    'freeze_silent_collect': (
        'run_timeline_comparison.py',
        "                mappo_trainer.collect(\n"
        "                    aid, tr['obs_t'], tr['heads'], tr['prb'],\n"
        "                    tr['log_prob'], tr['value'], r, False,\n"
        "                    global_obs=tr['global_obs'])\n",
        "                try:  # NEGCTL\n"
        "                    mappo_trainer.collect(\n"
        "                        aid, tr['obs_t'], tr['heads'], tr['prb'],\n"
        "                        tr['log_prob'], tr['value'], r, False,\n"
        "                        global_obs=tr['global_obs'])\n"
        "                except Exception:\n"
        "                    pass\n",
        {'tests/test_prefreeze_online_arm.py': (
            'test_a_refused_transition_in_the_online_arm_is_loud',)},
        'the online arm swallows a collect failure again (before D-FIX.4, '
        'D6.41)'),
    'job_env_not_cleared': (
        'confirmatory/chain_lib.sh',
        '    echo \'if [ "${CHAIN_CLEAN_ENV:-}" != 1 ]; then exec /usr/bin/env -i '
        'CHAIN_CLEAN_ENV=1 /bin/bash "$0" "$@"; fi\'\n',
        '    : # NEGCTL\n',
        {'tests/test_run_chain_launcher.py': (
            'test_a_stray_variable_of_the_operators_shell_never_reaches_an_invocation',
            'test_the_job_environment_check_refuses_a_leaking_or_wrong_environment')},
        'the job script sources its env file over the caller\'s environment '
        '(before D-FIX.5): a stray MARL_* reaches the invocation'),
    'marl_env_not_compared': (
        'tools/stage_report.py',
        '        ok = ok and env_ok\n',
        '        ok = ok  # NEGCTL\n',
        {'tests/test_run_chain_launcher.py': (
            'test_the_completeness_gate_requires_the_recorded_marl_env_to_be_the_plans',)},
        'the completeness gate does not compare the recorded marl_env (before '
        'D-FIX.5)'),
    'job_env_check_vacuous': (
        'tools/preflight.py',
        '    ok = c.n > 0 and not c.failed\n'
        "    rec['verdict'] = 'PASS' if ok else ('FAIL(no checks ran)' if c.n == 0 else 'FAIL')\n",
        '    ok = c.n > 0  # NEGCTL\n'
        "    rec['verdict'] = 'PASS' if ok else ('FAIL(no checks ran)' if c.n == 0 else 'FAIL')\n",
        {'tests/test_run_chain_launcher.py': (
            'test_the_job_environment_check_refuses_a_leaking_or_wrong_environment',)},
        'the job-environment check reports PASS whatever it found'),
}


_copy_tree = NC._copy_tree


def _copy_tree_with_listing(dst):
    """The WP13 copy, plus a `.synced_files` listing of the files the copy
    holds.  The launcher tests run confirmatory/run_final.sh IN the copy, and
    its preflight refuses a tree whose digest cannot be computed
    (runinfo.tree_digest reads `.synced_files`, else git; a scratch copy has
    neither): without the listing every launcher test fails on the head copy
    and no control is attributable."""
    n = _copy_tree(dst)
    rels = []
    for base, _dirs, names in os.walk(dst):
        rels += [os.path.relpath(os.path.join(base, nm), dst) for nm in names]
    # the runner's owner record is not a tree file (D12F.2)
    rels = [r for r in rels if r != NC.OWNER]
    with open(os.path.join(dst, '.synced_files'), 'w') as fh:
        fh.write('\n'.join(sorted(rels)) + '\n')
    return n


def main(argv):
    NC._copy_tree = _copy_tree_with_listing
    return NC.main(argv, tests=TESTS, table=MUTATIONS, title='FIX-PREFREEZE',
                   scratch='prefreeze_negctl', doc=__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
