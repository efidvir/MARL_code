#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE PLAN, as data: stages S0..S12 of the final corrected rerun, expanded
into the individual invocations confirmatory/run_final.sh executes.

This file is the single place where COMPUTE_PLAN.md's schedule is written
down in executable form.  run_final.sh and the stage scripts read it; the
tests in tests/test_run_chain.py read it too, so the properties the plan
requires (<= 64 passes per invocation, whole (arm, seed-block) boundaries,
the dependency graph, front-loading, exclusivity of training, the shed
order) are asserted against the same object the launcher runs.

  python3 confirmatory/chain_plan.py --profile rehearsal --stages
  python3 confirmatory/chain_plan.py --profile rehearsal --ids S1
  python3 confirmatory/chain_plan.py --profile rehearsal --emit s1_dev_b0
  python3 confirmatory/chain_plan.py --profile rehearsal --keys s1_dev_b0
  python3 confirmatory/chain_plan.py --profile real --json

PROFILES
  real       the workload of COMPUTE_PLAN.md: 20 confirmatory seeds, full
             3600-tick passes, 4 training replicates, evaluation at 24
             workers.  NOT run by this work package.
  rehearsal  the same stages, same order, same scripts, at 4 seeds / 600
             ticks / 2 training replicates of 2 batches.  Its numbers are
             THROWAWAY and are written under output/rehearsal/ so they can
             never be mistaken for, or land in, a real-run path.
  test       a paper profile used only by the unit tests (no compute).
"""
import argparse
import json
import os
import sys

# ---------------------------------------------------------------- arms
# Policy-INDEPENDENT arms: they need only the frozen code.  ('random' and
# 'xldet' run on the MARL code path but never consult the policy network --
# see run_timeline_comparison.run_physics_pass -- so they are independent of
# the trained weights.  Since the trackAgent merge the driver neither needs
# nor opens a checkpoint for an invocation of these arms only
# (run_timeline_comparison.CHECKPOINT_ARMS); these stages still carry the
# untrained checkpoint as a placeholder, which is harmless and keeps the
# stage records unchanged.)
INDEP_ARMS = ['xldet', 'random', 'sdn', 'ospf', 'olsr', 'batman', 'aodv']
STRESS_ARMS = ['xldet', 'sdn', 'ospf']
DEPLOYED_ARM = 'marl_static'          # the deployed executor (Amendment 1)
FREEZE_ARM = 'marl_freeze'            # adapt-then-freeze variant
CONFIRMATORY_ARMS = INDEP_ARMS + [DEPLOYED_ARM]      # 8 arms

SCENARIOS = ('recovery', 'rescue_ops')               # A and B
MAX_PASSES_PER_INVOCATION = 64

# Per-pass CPU seconds at 3600 ticks (COMPUTE_PLAN.md cost model).
# MEASURED: marl_static, xldet, ospf.  The rest are the plan's stated
# assumptions and are labelled as such wherever they are used.
ARM_CPU_S_3600 = {
    'marl_static': 1369.0, 'xldet': 1619.0, 'ospf': 642.0,
    'sdn': 650.0, 'olsr': 650.0, 'batman': 650.0, 'aodv': 650.0,
    'random': 1369.0, 'marl_freeze': 2054.0, 'marl': 1369.0,
}
CORES_OBTAINED = 16.88                # measured at MARL_POOL_WORKERS=24

# The arm-neutral engine configuration and the mission policy configuration,
# as in run_5/run_corrected_engine.sh and confirmatory/run_amendment1.sh,
# with ONE deliberate difference: MARL_POSTCARD_FRAGMENTS=1 is gone from
# POLICY_ENV.  RERUN_SPEC WP5 removed the fragment cue and the engine now
# REFUSES the switch at import (sixg_sim/agent.py), so a stage that still set
# it would fail before its first tick (D5.11 named this line; the trackAgent
# merge applies it).  MARL_REWARD_PROFILE=mission stays: since WP8 it is a
# tripwire that accepts only the frozen set, so it moves nothing.
ENGINE_ENV = {
    'MARL_TRANSPORT_POWER': 'planning',
    'MARL_RELAY_STATE': 'reconcile',
    'MARL_RELAY_REPOINT': 'fragment_aware',
    'MARL_POSTCARD_ALWAYS': '1',
}
POLICY_ENV = {
    'MARL_REWARD_PROFILE': 'mission',
    'MARL_RELAY_ENTROPY_FLOOR': '0.01',
}
EVAL_ENV = {'MARL_STEER_MODEL': 'actionable'}

# ---------------------------------------------------------------- stages
# deps        HARD data dependencies (COMPUTE_PLAN.md dependency graph).
# order_after SCHEDULE edges that are not data dependencies.  S2 (training)
#             depends only on the frozen tree, but the plan front-loads S1
#             ahead of it to get a defect signal on seven of the eight arms
#             before committing seven hours of training; that is an ordering
#             constraint the launcher enforces (waivable only by an explicit,
#             recorded operator action).
STAGES = [
    ('S0',  'prep: confirmatory manifest [C] + untrained checkpoints [B]', [], []),
    ('S1',  'dev + stress, policy-independent arms (340 passes) [F]+[H]', ['S0'], []),
    ('S2',  'TRAIN replicates, EXCLUSIVE [D]', ['S0'], ['S1']),
    ('S3',  'validation selection passes [E]', ['S2'], []),
    ('S3b', 'apply the pre-registered selection rule -> SELECTED checkpoint', ['S3'], []),
    ('S4',  'dev + stress marl_static, end-state dump [G]+[I]+[J]', ['S3b'], []),
    ('S5',  'DEV GATE + FREEZE_FINAL (technical) [K]+[L]', ['S1', 'S4'], []),
    ('S5b', 'analysis dry-run on the DEVELOPMENT set (its designed validation '
            'input) - proves confirmatory_analysis.py RUNS before the '
            'confirmatory stage is committed to', ['S5'], []),
    ('S6',  'CONFIRMATORY, 8 arms, one tree, 5 invocations [M]+[N]', ['S5b', 'S3b'], []),
    ('S7',  'CONFIRMATORY ANALYSIS (primary result) [R]', ['S6'], []),
    # Analysis S8 (hash invariance, PREREGISTRATION_R2 section 10) needs one
    # extra marl_static invocation under PYTHONHASHSEED=1 on the two lowest
    # confirmatory seeds, the PYTHONHASHSEED=0 side being S6's first block.
    # Its own stage so that it never gates the primary result (after S7)
    # and is not shed with the untrained control: chain S8 and analysis S8
    # are different namespaces (section 6).
    ('S6h', 'hash-invariance repeat of an S6 subset under PYTHONHASHSEED=1 '
            '(analysis S8)', ['S5', 'S3b'], ['S7']),
    ('S8',  'untrained-actor control [O]', ['S5', 'S0'], ['S7']),
    ('S9',  'adapt-then-freeze variant [P]', ['S5', 'S3b'], ['S7']),
    ('S10', 'sensitivity: installation off [Q1]', ['S5', 'S3b'], ['S7']),
    ('S11', 'sensitivity: re-pointing off [Q2]', ['S5', 'S3b'], ['S7']),
    ('S12', 'secondary analyses S7-secondary, S8, S12 + descriptive report [S]',
            ['S7', 'S6h', 'S8', 'S9', 'S10', 'S11'], []),
]
STAGE_ORDER = [s[0] for s in STAGES]

# COMPUTE_PLAN.md failure handling: "shed stages in this order -- S10/S11
# sensitivity runs, S9 adapt-then-freeze, S8 untrained control".  Shedding is
# an explicit operator action (--shed), never automatic, and a group may only
# be shed after every group before it in this list.
SHED_ORDER = ['sensitivity', 'adapt_freeze', 'untrained']
SHED_STAGES = {'sensitivity': ['S10', 'S11'],
               'adapt_freeze': ['S9'],
               'untrained': ['S8']}

PROFILES = {
    'real': {
        'out_prefix': 'final/',
        'ticks': None,                       # full length (3600)
        'dev_seeds': list(range(50, 60)) + list(range(70, 80)),
        'stress_seeds': list(range(50, 60)),
        'val_seeds': list(range(60, 67)),
        'cf_seeds': list(range(200, 220)),
        'endstate_seed': 58,
        'torch_seeds': [1234, 2345, 3456, 4567],
        'train': {'episodes': 400, 'episode_ticks': 800, 'batch_size': 16,
                  'base_seed': 7, 'seed_range': '42-49,100-139',
                  'workers': 16, 'concurrent': 2},
        'eval_workers': 24,
        'stress_dbm': '-75',
        'stress_tick': 2000,
        'watchdog_min_hours': 0.5,
        'nominal_ticks': 3600,
        # COMPUTE_PLAN.md: delay a training wave above ~20 cores of load.
        'train_load_ceiling': 20.0,
        # The real profile's own development set is already full length.
        'dev_full_seeds': [],
        'dev_analysis_prefixes': 's1_dev_b,s4_dev_static',
        # ... and so is its confirmatory set: S7 reads S6 itself.
        'cf_full_seeds': [],
        'cf_analysis_prefixes': 's6_cf_b',
    },
    'rehearsal': {
        'out_prefix': 'rehearsal/',
        'ticks': 600,
        'dev_seeds': [50, 51, 70, 71],
        'stress_seeds': [50, 51],
        'val_seeds': [60, 61],
        # NOT the pre-registered block 200-219: a rehearsal is a pipeline
        # test, and the confirmatory seeds must stay untouched until the
        # real run.  220-223 are unregistered (the dry run's block).
        # tests/test_run_chain.py pins that only 'real' names 200-219.
        'cf_seeds': [220, 221, 222, 223],
        'endstate_seed': 58,
        # THREE replicates with concurrent=2, so the wait-then-launch-the-next
        # path of stage_train.sh is actually executed: with two seeds and
        # concurrent=2 both replicates launch back to back and the stage then
        # gang-waits, which is the behaviour the plan explicitly replaces.
        # Three seeds also drive --expect-inv with a CHANGING set of alive
        # invocations, which is where the first rehearsal failure came from.
        'torch_seeds': [1234, 2345, 3456],
        'train': {'episodes': 32, 'episode_ticks': 200, 'batch_size': 16,
                  'base_seed': 7, 'seed_range': '42-49,100-139',
                  'workers': 16, 'concurrent': 2},
        # The pre-registered analysis holds ABSOLUTE tick windows
        # (PRE_SURGE = 900..1400, T_EVENT B = 1400) while the driver SCALES
        # the severance and rescue ticks with --ticks, so it cannot run below
        # 1400 ticks at all (REQUEST R11).  S5b therefore gets a small
        # FULL-LENGTH development subset of its own: two seeds, every dev arm,
        # both scenarios, at the nominal tick count.  It is the only part of
        # the rehearsal that is not reduced, and it exists so the S5b -> S6
        # edge is executed rather than assumed.
        'dev_full_seeds': [50, 51],
        'dev_analysis_prefixes': 's1_devfull,s4_devfull',
        # The same holds for S7 (WP13): the confirmatory analysis refuses a
        # 600-tick set outright ("series ... has 600 entries, expected 3600
        # (run length)"), so at reduced scale the primary step could only
        # fail -- the 2026-09-24 rehearsal stopped there, and a rehearsal that
        # cannot reach its primary result has not rehearsed it.  S6 therefore
        # carries a two-seed FULL-LENGTH confirmatory subset of its own (all
        # 8 arms, both scenarios, nominal ticks, seeds from cf_seeds so the
        # S0 manifest covers them), and S7 analyses exactly that.  The
        # reduced S6 invocations stay the baseline of the S10/S11 ablations,
        # which run at the reduced tick count too (--same-ticks-as).
        'cf_full_seeds': [220, 221],
        'cf_analysis_prefixes': 's6_cffull_b',
        # The rehearsal's training block is ~7 minutes of 2-batch toy training
        # whose weights are thrown away, so the plan's 20-core ceiling -- which
        # exists to protect a 3.5 h draw -- would only stop the rehearsal from
        # reaching anything downstream.  The CHECK still runs and is recorded;
        # only the threshold differs.  Recorded in docs/RUN_CHAIN.md section 5.
        'train_load_ceiling': 40.0,
        'eval_workers': 24,
        'stress_dbm': '-75',
        # MARL_STRESS_INTERFERENCE_TICK is an ABSOLUTE tick and does NOT
        # scale with --ticks, so a short rehearsal at the real value would
        # run a "stress" arm that never sees the interferer.  Scaled here.
        'stress_tick': 333,
        'watchdog_min_hours': 0.02,
        'nominal_ticks': 3600,
    },
    # FULL-SCALE DRY RUN.  Identical to 'real' in every dimension that costs
    # time -- full 3600-tick passes, the real 4 x 25-batch training block, 24
    # evaluation workers, the real dev/stress/validation seeds -- but on
    # confirmatory seeds 220-239 instead of the pre-registered 200-219, and
    # writing under output/dryrun/.
    #
    # WHY IT IS NOT A PEEK.  The pre-registered confirmatory set is 200-219 and
    # this profile never touches it.  Seeds 220-239 were audited CLEAN by
    # tools/seed_audit.py in the same run that cleared 200-219, which is why
    # that second range was audited at all.  Nothing computed here can enter
    # the confirmatory analysis: a different seed set, a different output
    # prefix, and the S5/S6 manifest gates key on the profile's own seeds.
    #
    # WHY RUN IT.  The reduced rehearsal could not exercise the production
    # memory profile at 24 workers, the real concurrent-replicate training
    # block, or the true per-stage wall clock.  This does, on otherwise idle
    # machine time, and its measured timings replace the plan's extrapolations.
    # Its RESULTS are throwaway -- the tree is not frozen and WP4, WP5 and
    # WP-PERF are still to land -- but its VALIDATION of the chain mechanics
    # and its timings are not.
    'dryrun': {
        'out_prefix': 'dryrun/',
        'ticks': None,                       # full length (3600)
        'dev_seeds': list(range(50, 60)) + list(range(70, 80)),
        'stress_seeds': list(range(50, 60)),
        'val_seeds': list(range(60, 67)),
        'cf_seeds': list(range(220, 240)),
        'endstate_seed': 58,
        'torch_seeds': [1234, 2345, 3456, 4567],
        'train': {'episodes': 400, 'episode_ticks': 800, 'batch_size': 16,
                  'base_seed': 7, 'seed_range': '42-49,100-139',
                  'workers': 16, 'concurrent': 2},
        'eval_workers': 24,
        'stress_dbm': '-75',
        'stress_tick': 2000,
        'watchdog_min_hours': 0.5,
        'nominal_ticks': 3600,
        'train_load_ceiling': 20.0,
        'dev_full_seeds': [],
        'dev_analysis_prefixes': 's1_dev_b,s4_dev_static',
        'cf_full_seeds': [],
        'cf_analysis_prefixes': 's6_cf_b',
    },
    'test': {
        'out_prefix': 'test/',
        'ticks': 60,
        'dev_seeds': [50, 51],
        'stress_seeds': [50],
        'val_seeds': [60],
        'cf_seeds': [220, 221],
        'endstate_seed': 58,
        'torch_seeds': [1234, 2345],
        'train': {'episodes': 2, 'episode_ticks': 60, 'batch_size': 1,
                  'base_seed': 7, 'seed_range': '42-49,100-139',
                  'workers': 2, 'concurrent': 2},
        'eval_workers': 4,
        'stress_dbm': '-75',
        'stress_tick': 33,
        'watchdog_min_hours': 99.0,
        'nominal_ticks': 3600,
        'train_load_ceiling': 9999.0,
        'dev_full_seeds': [],
        'dev_analysis_prefixes': 's1_dev_b,s4_dev_static',
        'cf_full_seeds': [],
        'cf_analysis_prefixes': 's6_cf_b',
    },
}


def block_size(arms):
    """Largest whole-seed block that keeps an invocation at <= 64 passes.

    Splitting is along whole (arm, seed-block) boundaries: every arm of the
    invocation runs every seed of the block, in both scenarios.  A crash then
    costs exactly one invocation and no pass can be duplicated or lost.
    """
    per_seed = max(1, len(arms) * len(SCENARIOS))
    return max(1, MAX_PASSES_PER_INVOCATION // per_seed)


def chunks(seeds, n):
    return [seeds[i:i + n] for i in range(0, len(seeds), n)]


def expected_keys(arms, seeds):
    return ['%s|%s|%d' % (a, s, seed)
            for a in arms for s in SCENARIOS for seed in sorted(seeds)]


def expected_hours(arms, seeds, ticks, nominal, workers_note=CORES_OBTAINED):
    cpu = sum(ARM_CPU_S_3600.get(a, 1000.0) for a in arms) * len(seeds) * len(SCENARIOS)
    cpu *= (float(ticks or nominal) / float(nominal))
    return round(cpu / (workers_note * 3600.0), 3)


def _eval(inv_id, stage, title, arms, seeds, P, ckpt_kind, ckpt_value='',
          env=None, shed_group='', serial_group='', out=None, ticks='profile',
          ablation_of='', ablation_series=(), ablation_counter='',
          hashseed='0', analysis_role='', control_name='', hash_pair_of=''):
    ticks = P['ticks'] if ticks == 'profile' else ticks
    out = out or (P['out_prefix'] + inv_id + '.pkl')
    e = dict(ENGINE_ENV)
    e.update(POLICY_ENV)
    e.update(EVAL_ENV)
    e.update(env or {})
    return {
        'id': inv_id, 'stage': stage, 'kind': 'eval', 'title': title,
        'script': 'confirmatory/stage_eval.sh',
        'arms': list(arms), 'seeds': list(seeds), 'scenarios': list(SCENARIOS),
        'ticks': ticks, 'out': out, 'artifact': 'output/' + out,
        'ckpt_kind': ckpt_kind, 'ckpt_value': ckpt_value,
        'env': e, 'workers': P['eval_workers'],
        'expected_passes': len(arms) * len(seeds) * len(SCENARIOS),
        'expected_keys': expected_keys(arms, seeds),
        'exclusive': False, 'serial_group': serial_group,
        'shed_group': shed_group,
        # An ABLATION invocation names the stage it is an ablation OF, the
        # series the switch is supposed to move, and the baseline counter that
        # proves the mechanism had anything to switch off.  A sensitivity whose
        # output is identical to its baseline is either a broken switch (FAIL)
        # or a mechanism that never fired (NOT EXERCISED) -- never a pass.
        'ablation_of': ablation_of,
        'ablation_series': list(ablation_series),
        'ablation_counter': ablation_counter,
        # PYTHONHASHSEED of the invocation: '0' everywhere except the analysis-S8
        # repeat (S6h).  chain_write_env exports it, the completeness gate
        # requires the pickle to record it.
        'hashseed': str(hashseed),
        # Which secondary analysis of S12 reads this invocation's pickle:
        # 'untrained' (analysis S12, one control per untrained actor, named
        # control_name), 'freeze' (secondary S7: marl_static vs marl_freeze),
        # 'hash1' (analysis S8; hash_pair_of names its PYTHONHASHSEED=0 side).
        'analysis_role': analysis_role,
        'control_name': control_name,
        'hash_pair_of': hash_pair_of,
        'expected_hours': expected_hours(arms, seeds, ticks, P['nominal_ticks']),
        'marker': 'INVOCATION COMPLETE',
    }


def build(profile):
    P = PROFILES[profile]
    pre = P['out_prefix']
    inv = []

    # ---------------------------------------------------------------- S0
    cf = P['cf_seeds']
    inv.append({
        'id': 's0_manifest', 'stage': 'S0', 'kind': 'tool',
        'title': 'benchmark manifest for the confirmatory seeds [C]',
        'script': 'confirmatory/stage_prep.sh', 'step': 'manifest',
        'arms': [], 'seeds': cf, 'scenarios': list(SCENARIOS),
        'ticks': None,
        'out': '', 'artifact': 'confirmatory/%smanifest_%d_%d.json' % (
            pre.replace('/', '_'), cf[0], cf[-1]),
        'ckpt_kind': 'none', 'ckpt_value': '',
        'env': dict(ENGINE_ENV, **dict(POLICY_ENV, **EVAL_ENV)),
        'workers': 1, 'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.02, 'marker': 'INVOCATION COMPLETE',
    })
    inv.append({
        'id': 's0_untrained', 'stage': 'S0', 'kind': 'tool',
        'title': 'untrained-actor checkpoints [B] (also the S1 placeholder)',
        'script': 'confirmatory/stage_prep.sh', 'step': 'untrained',
        'arms': [], 'seeds': [], 'scenarios': [],
        'ticks': None, 'out': '',
        'artifact': 'output/%suntrained_s%d/policy_init.pt' % (pre, P['torch_seeds'][0]),
        'ckpt_kind': 'none', 'ckpt_value': '',
        'env': dict(ENGINE_ENV, **POLICY_ENV),
        'workers': 1, 'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.05, 'marker': 'INVOCATION COMPLETE',
    })

    # ---------------------------------------------------------------- S1
    # The placeholder checkpoint.  run_timeline_comparison.main() used to
    # refuse to start unless MARL_EVAL_CHECKPOINT existed, even when every
    # arm of the invocation was policy-independent; since the trackAgent
    # merge it requires and opens a checkpoint only for CHECKPOINT_ARMS, so
    # the placeholder is no longer needed but is kept (it is never opened).
    ph = 'output/%suntrained_s%d/policy_init.pt' % (pre, P['torch_seeds'][0])
    for i, blk in enumerate(chunks(P['dev_seeds'], block_size(INDEP_ARMS))):
        inv.append(_eval('s1_dev_b%d' % i, 'S1',
                         'dev, policy-independent arms, seeds %s' % blk,
                         INDEP_ARMS, blk, P, 'literal', ph))
    # The full-length development subset (see the profile comment): same arms,
    # same scenarios, NOMINAL tick count, its own pickle.  A 600-tick pass and
    # a 3600-tick pass of the same (arm, seed, scenario) are different
    # conditions and never share an artefact.
    for i, blk in enumerate(chunks(P.get('dev_full_seeds') or [],
                                   block_size(INDEP_ARMS))):
        inv.append(_eval('s1_devfull_b%d' % i, 'S1',
                         'dev at FULL LENGTH for the analysis dry run, seeds %s' % blk,
                         INDEP_ARMS, blk, P, 'literal', ph, ticks=None))
    for i, blk in enumerate(chunks(P['stress_seeds'], block_size(STRESS_ARMS))):
        inv.append(_eval('s1_stress_b%d' % i, 'S1',
                         'stress, policy-independent arms, seeds %s' % blk,
                         STRESS_ARMS, blk, P, 'literal', ph,
                         env={'MARL_STRESS_INTERFERENCE_DBM': P['stress_dbm'],
                              'MARL_STRESS_INTERFERENCE_TICK': str(P['stress_tick'])}))

    # ---------------------------------------------------------------- S2
    tr = P['train']
    for s in P['torch_seeds']:
        inv.append({
            'id': 's2_train_s%d' % s, 'stage': 'S2', 'kind': 'train',
            'title': 'training replicate torch-seed %d (EXCLUSIVE)' % s,
            'script': 'confirmatory/stage_train.sh',
            'arms': [], 'seeds': [], 'scenarios': [], 'ticks': tr['episode_ticks'],
            'out': '', 'artifact': 'output/%sck_s%d/policy_final.pt' % (pre, s),
            'ckpt_kind': 'none', 'ckpt_value': '',
            'env': dict(ENGINE_ENV, **POLICY_ENV),
            'workers': tr['workers'], 'torch_seed': s,
            'train_args': ['--episodes', str(tr['episodes']),
                           '--episode-ticks', str(tr['episode_ticks']),
                           '--seed-range', tr['seed_range'],
                           '--torch-seed', str(s),
                           '--base-seed', str(tr['base_seed']),
                           '--batch-size', str(tr['batch_size']),
                           # COMPUTE_PLAN ranked action 4: 26.8 min per
                           # replicate of serial time after the final
                           # checkpoint save, and nothing reads the summary
                           # these phases populate.
                           '--eval-episodes', '0', '--baseline-episodes', '0'],
            'expected_passes': 0, 'expected_keys': [],
            'exclusive': True, 'serial_group': '', 'shed_group': '',
            'expected_hours': round(3.48 * (tr['episodes'] / 400.0)
                                    * (tr['episode_ticks'] / 800.0), 3),
            # The wrapper prints this only on exit 0.  It is deliberately NOT
            # the trainer's own "TRAINING COMPLETE", which appears in the log
            # before the final save and would make a later failure look done.
            'marker': 'INVOCATION COMPLETE',
        })

    # ---------------------------------------------------------------- S3
    for s in P['torch_seeds']:
        inv.append(_eval('s3_val_s%d' % s, 'S3',
                         'validation passes for replicate %d' % s,
                         [DEPLOYED_ARM], P['val_seeds'], P,
                         'replicate', 'output/%sck_s%d/policy_final.pt' % (pre, s)))
    inv.append({
        'id': 's3b_select', 'stage': 'S3b', 'kind': 'select',
        'title': 'pre-registered selection rule over the validation passes',
        'script': 'confirmatory/stage_select.sh',
        'arms': [DEPLOYED_ARM], 'seeds': P['val_seeds'], 'scenarios': list(SCENARIOS),
        'ticks': None, 'out': '',
        'artifact': 'confirmatory/SELECTED_%s.txt' % profile,
        'ckpt_kind': 'none', 'ckpt_value': '',
        'env': {}, 'workers': 1, 'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.01, 'marker': 'INVOCATION COMPLETE',
    })

    # ---------------------------------------------------------------- S4
    inv.append(_eval('s4_dev_static', 'S4', 'dev marl_static [G]',
                     [DEPLOYED_ARM], P['dev_seeds'], P, 'selected'))
    inv.append(_eval('s4_stress_static', 'S4', 'stress marl_static [I]',
                     [DEPLOYED_ARM], P['stress_seeds'], P, 'selected',
                     env={'MARL_STRESS_INTERFERENCE_DBM': P['stress_dbm'],
                          'MARL_STRESS_INTERFERENCE_TICK': str(P['stress_tick'])}))
    if P.get('dev_full_seeds'):
        inv.append(_eval('s4_devfull_static', 'S4',
                         'dev marl_static at FULL LENGTH for the analysis dry run',
                         [DEPLOYED_ARM], P['dev_full_seeds'], P, 'selected',
                         ticks=None))
    inv.append({
        'id': 's4_endstate', 'stage': 'S4', 'kind': 'dump',
        'title': 'end-state dump, seed %d [J]' % P['endstate_seed'],
        'script': 'confirmatory/stage_dump.sh',
        'arms': [DEPLOYED_ARM], 'seeds': [P['endstate_seed']], 'scenarios': [],
        'ticks': P['ticks'], 'out': '',
        'artifact': 'output/%sendstate_seed%d.json' % (pre, P['endstate_seed']),
        'ckpt_kind': 'selected', 'ckpt_value': '',
        'env': dict(ENGINE_ENV, **dict(POLICY_ENV, **EVAL_ENV)),
        'workers': 1, 'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.1, 'marker': 'INVOCATION COMPLETE',
    })

    # ---------------------------------------------------------------- S5
    inv.append({
        'id': 's5_gate', 'stage': 'S5', 'kind': 'gate',
        'title': 'DEV GATE (technical) + FREEZE_FINAL digest seal',
        'script': 'confirmatory/stage_gate.sh',
        'arms': [], 'seeds': [], 'scenarios': [], 'ticks': None, 'out': '',
        'artifact': 'confirmatory/FREEZE_FINAL_%s.txt' % profile,
        'ckpt_kind': 'none', 'ckpt_value': '', 'env': {}, 'workers': 1,
        'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.01, 'marker': 'INVOCATION COMPLETE',
    })

    # ---------------------------------------------------------------- S5b
    # WHY THIS STAGE EXISTS.  The rehearsal ran every gate of S7 green and
    # then confirmatory_analysis.py itself died -- at T+14.5 h in real time,
    # with the whole confirmatory set already spent.  The analysis is written
    # to be validated by reproducing the DEVELOPMENT-set figures, and the
    # development set is complete at the end of S4, three hours earlier.  So
    # the launcher runs the real analysis script on the real development
    # pickles here, and S6 does not start until it has produced a report.
    inv.append({
        'id': 's5b_dev_analysis', 'stage': 'S5b', 'kind': 'analysis',
        'title': 'run the pre-registered analysis on the development set',
        'script': 'confirmatory/stage_analysis.sh', 'step': 'dev',
        'arms': [DEPLOYED_ARM] + INDEP_ARMS, 'seeds': P['dev_seeds'],
        'scenarios': list(SCENARIOS), 'ticks': None, 'out': '',
        'artifact': 'confirmatory/%sdev_report.json' % pre.replace('/', '_'),
        'ckpt_kind': 'none', 'ckpt_value': '', 'env': {}, 'workers': 1,
        'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.05, 'marker': 'INVOCATION COMPLETE',
    })

    # ---------------------------------------------------------------- S6
    for i, blk in enumerate(chunks(P['cf_seeds'], block_size(CONFIRMATORY_ARMS))):
        inv.append(_eval('s6_cf_b%d' % i, 'S6',
                         'confirmatory, all 8 arms, seeds %s' % blk,
                         CONFIRMATORY_ARMS, blk, P, 'selected'))
    # The full-length confirmatory subset of a reduced profile (see the
    # rehearsal profile's comment): same arms, same scenarios, NOMINAL tick
    # count, its own pickle -- the input S7 analyses there.  Empty for the
    # full-length profiles.
    for i, blk in enumerate(chunks(P.get('cf_full_seeds') or [],
                                   block_size(CONFIRMATORY_ARMS))):
        inv.append(_eval('s6_cffull_b%d' % i, 'S6',
                         'confirmatory at FULL LENGTH for the primary '
                         'analysis, seeds %s' % blk,
                         CONFIRMATORY_ARMS, blk, P, 'selected', ticks=None))

    # ---------------------------------------------------------------- S7
    inv.append({
        'id': 's7_analysis', 'stage': 'S7', 'kind': 'analysis',
        'title': 'pre-registered confirmatory analysis (primary result)',
        'script': 'confirmatory/stage_analysis.sh', 'step': 'primary',
        'arms': CONFIRMATORY_ARMS, 'seeds': P['cf_seeds'], 'scenarios': list(SCENARIOS),
        'ticks': None, 'out': '',
        'artifact': 'confirmatory/%scf_report.json' % pre.replace('/', '_'),
        'ckpt_kind': 'none', 'ckpt_value': '', 'env': {}, 'workers': 1,
        'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.2, 'marker': 'INVOCATION COMPLETE',
    })

    # ---------------------------------------------------------------- S6h
    # Analysis S8, hash invariance (PREREGISTRATION_R2 section 10): the
    # selected checkpoint on the TWO LOWEST seeds of the confirmatory block,
    # both scenarios, under PYTHONHASHSEED=1; the PYTHONHASHSEED=0 side is
    # the S6 invocation at the same tick count that holds both seeds (S6's
    # first block).  S12 compares them with tools/hash_invariance_check.py
    # --require-decision and embeds the result with --s8.
    hs = sorted(P['cf_seeds'])[:2]
    h0 = [i for i in inv if i['stage'] == 'S6' and i['ticks'] == P['ticks']
          and set(hs) <= set(i['seeds']) and DEPLOYED_ARM in i['arms']]
    if len(hs) != 2 or len(h0) != 1:
        raise SystemExit('profile %r: no single S6 invocation holds the hash '
                         'pair seeds %s at its tick count' % (profile, hs))
    inv.append(_eval('s6h_hash1', 'S6h',
                     'hash-invariance repeat (PYTHONHASHSEED=1) of %s, seeds %s'
                     % (h0[0]['id'], hs),
                     [DEPLOYED_ARM], hs, P, 'selected', hashseed='1',
                     analysis_role='hash1', hash_pair_of=h0[0]['id']))

    # ---------------------------------------------------------------- S8
    # One invocation per untrained checkpoint: MARL_EVAL_CHECKPOINT is a
    # per-invocation environment variable, so checkpoints cannot be mixed
    # inside one invocation (COMPUTE_PLAN's "three invocations" for S8 is not
    # reachable; four of 40 passes each is, and is still <= 64).
    #
    # Analysis S12 (PREREGISTRATION_R2 section 10) compares each untrained
    # actor with the trained policy, random and the best routing protocol;
    # S12 of the chain hands every one of these pickles to
    # confirmatory_analysis.py --untrained untrained_s<seed>=<pickle>.  The
    # analysis refuses a set below the nominal tick count, so a REDUCED
    # profile (cf_full_seeds) also runs a full-length subset per untrained
    # actor on exactly the seeds its primary analysis reads (as S6 does for
    # S7), and it is that subset the analysis reads.
    full = bool(P.get('cf_full_seeds'))
    for s in P['torch_seeds']:
        inv.append(_eval('s8_untrained_s%d' % s, 'S8',
                         'untrained-actor control, init checkpoint %d' % s,
                         [DEPLOYED_ARM], P['cf_seeds'], P, 'literal',
                         'output/%suntrained_s%d/policy_init.pt' % (pre, s),
                         shed_group='untrained',
                         analysis_role='' if full else 'untrained',
                         control_name='untrained_s%d' % s))
    for s in (P['torch_seeds'] if full else []):
        inv.append(_eval('s8_untrainedfull_s%d' % s, 'S8',
                         'untrained-actor control at FULL LENGTH for the '
                         'analysis, init checkpoint %d' % s,
                         [DEPLOYED_ARM], P['cf_full_seeds'], P, 'literal',
                         'output/%suntrained_s%d/policy_init.pt' % (pre, s),
                         shed_group='untrained', ticks=None,
                         analysis_role='untrained',
                         control_name='untrained_s%d' % s))

    # ---------------------------------------------------------------- S9
    # Secondary S7 (marl_static vs marl_freeze) reads this pickle beside the
    # primary input; a reduced profile adds a full-length subset (as S8).
    inv.append(_eval('s9_freeze', 'S9', 'adapt-then-freeze variant [P]',
                     [FREEZE_ARM], P['cf_seeds'], P, 'selected',
                     shed_group='adapt_freeze', serial_group='marl_freeze',
                     analysis_role='' if full else 'freeze'))
    if full:
        inv.append(_eval('s9_freezefull', 'S9',
                         'adapt-then-freeze variant at FULL LENGTH for the '
                         'analysis',
                         [FREEZE_ARM], P['cf_full_seeds'], P, 'selected',
                         shed_group='adapt_freeze', serial_group='marl_freeze',
                         ticks=None, analysis_role='freeze'))

    # ------------------------------------------------------------- S10/S11
    # S10 is the PRE-REGISTERED evaluation-only sensitivity "installation off,
    # idealised timing": MARL_LS_INSTALL=0 (RERUN_SPEC WP3,
    # docs/LINK_STATE_SERVICE.md section 7, DECISIONS.md D3.9, the comment at
    # sixg_sim/link_state_service.MARL_LS_INSTALL).  A confirmed transition
    # takes effect at once; detection and billing are unchanged, so it
    # isolates the cost of the flood + SPF installation model alone.  It is
    # NOT MARL_DETECT_TICKS=0, which removes the link-state service entirely
    # (the omniscient ablation: true graph, no detection, no billing) and
    # which the chain ran here until FIX-PREFREEZE item 1.
    # Every severance installs at least one transition through the flood + SPF
    # model in the baseline, so there is no legitimate nothing-to-switch-off
    # case: no counter is declared and a zero difference is a FAILURE.
    inv.append(_eval('s10_sens_install', 'S10',
                     'sensitivity: installation off (idealised timing)',
                     [DEPLOYED_ARM], P['cf_seeds'], P, 'selected',
                     env={'MARL_LS_INSTALL': '0'}, shed_group='sensitivity',
                     ablation_of='S6',
                     ablation_series=('conn', 'delivered', 'energy',
                                      'relay_links_removed', 'action_sig')))
    # MARL_RELAY_REPOINT=off removes fragment-aware re-pointing.  If the
    # baseline never re-pointed (repoints == 0) the ablation had nothing to
    # switch off: that is NOT EXERCISED, recorded, and it is not a pass.
    inv.append(_eval('s11_sens_repoint', 'S11',
                     'sensitivity: re-pointing off',
                     [DEPLOYED_ARM], P['cf_seeds'], P, 'selected',
                     env={'MARL_RELAY_REPOINT': 'off'}, shed_group='sensitivity',
                     ablation_of='S6',
                     ablation_series=('repoints', 'relay_bridges',
                                      'relay_links_removed', 'conn'),
                     ablation_counter='repoints'))

    # ---------------------------------------------------------------- S12
    inv.append({
        'id': 's12_secondary', 'stage': 'S12', 'kind': 'analysis',
        'title': 'secondary analyses (S7-secondary, S8 hash, S12 untrained) '
                 'and the descriptive report over S6h and S8-S11',
        'script': 'confirmatory/stage_analysis.sh', 'step': 'secondary',
        'arms': [], 'seeds': P['cf_seeds'], 'scenarios': list(SCENARIOS),
        'ticks': None, 'out': '',
        'artifact': 'confirmatory/%ssecondary_report.json' % pre.replace('/', '_'),
        'ckpt_kind': 'none', 'ckpt_value': '', 'env': {}, 'workers': 1,
        'expected_passes': 0, 'expected_keys': [],
        'exclusive': False, 'serial_group': '', 'shed_group': '',
        'expected_hours': 0.1, 'marker': 'INVOCATION COMPLETE',
    })
    return inv


# ---------------------------------------------------------------- helpers
def by_id(profile, inv_id):
    for i in build(profile):
        if i['id'] == inv_id:
            return i
    raise SystemExit('unknown invocation %r in profile %r' % (inv_id, profile))


def stage_deps(stage):
    for s, _t, deps, order in STAGES:
        if s == stage:
            return deps, order
    raise SystemExit('unknown stage %r' % stage)


def shell_quote(s):
    s = str(s)
    if s == '':
        return "''"
    ok = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@%_-+=:,./')
    if all(ch in ok for ch in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def emit(i):
    """A block of KEY=value assignments for `source`ing from bash."""
    env = ' '.join('%s=%s' % (k, i['env'][k]) for k in sorted(i['env']))
    out = [
        ('INV_ID', i['id']), ('INV_STAGE', i['stage']), ('INV_KIND', i['kind']),
        ('INV_TITLE', i['title']), ('INV_SCRIPT', i['script']),
        ('INV_STEP', i.get('step', '')),
        ('INV_ARMS', ','.join(i['arms'])),
        ('INV_SEEDS', ','.join(str(s) for s in i['seeds'])),
        ('INV_TICKS', '' if i.get('ticks') in (None, '') else str(i['ticks'])),
        ('INV_OUT', i['out']), ('INV_ARTIFACT', i['artifact']),
        ('INV_CKPT_KIND', i['ckpt_kind']), ('INV_CKPT_VALUE', i['ckpt_value']),
        ('INV_ENV', env), ('INV_WORKERS', i['workers']),
        ('INV_EXPECTED_PASSES', i['expected_passes']),
        ('INV_EXCLUSIVE', '1' if i['exclusive'] else '0'),
        ('INV_SERIAL_GROUP', i['serial_group']),
        ('INV_SHED_GROUP', i['shed_group']),
        ('INV_HOURS', i['expected_hours']),
        ('INV_MARKER', i['marker']),
        ('INV_TRAIN_ARGS', ' '.join(i.get('train_args', []))),
        ('INV_TORCH_SEED', i.get('torch_seed', '')),
        ('INV_ABLATION_OF', i.get('ablation_of', '')),
        ('INV_ABLATION_SERIES', ','.join(i.get('ablation_series') or ())),
        ('INV_ABLATION_COUNTER', i.get('ablation_counter', '')),
        ('INV_HASHSEED', i.get('hashseed', '0')),
        ('INV_ANALYSIS_ROLE', i.get('analysis_role', '')),
        ('INV_CONTROL_NAME', i.get('control_name', '')),
        ('INV_HASH_PAIR_OF', i.get('hash_pair_of', '')),
    ]
    return '\n'.join('%s=%s' % (k, shell_quote(v)) for k, v in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='rehearsal', choices=sorted(PROFILES))
    ap.add_argument('--stages', action='store_true', help='list stage ids')
    ap.add_argument('--stage-deps', default=None, help='print deps/order of a stage')
    ap.add_argument('--ids', default=None, metavar='STAGE',
                    help='list invocation ids of a stage (or ALL)')
    ap.add_argument('--emit', default=None, metavar='ID')
    ap.add_argument('--keys', default=None, metavar='ID')
    ap.add_argument('--field', default=None, nargs=2, metavar=('ID', 'NAME'))
    ap.add_argument('--profile-field', default=None, metavar='NAME')
    ap.add_argument('--shed-group-of', default=None, metavar='STAGE')
    ap.add_argument('--artifacts', default=None, metavar='STAGE',
                    help='comma-separated pickle artefacts of a stage')
    ap.add_argument('--same-ticks-as', default=None, metavar='ID',
                    help='with --artifacts: only the invocations that run at '
                         'the tick count of invocation ID')
    ap.add_argument('--analysis-role', default=None, metavar='ROLE',
                    help='ids of the invocations S12 reads for ROLE '
                         '(untrained, freeze, hash1)')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()

    if a.profile_field:
        print(PROFILES[a.profile][a.profile_field])
    elif a.shed_group_of:
        g = [k for k, v in SHED_STAGES.items() if a.shed_group_of in v]
        print(g[0] if g else '')
    elif a.stages:
        print(' '.join(STAGE_ORDER))
    elif a.stage_deps:
        deps, order = stage_deps(a.stage_deps)
        print('DEPS=%s' % shell_quote(' '.join(deps)))
        print('ORDER_AFTER=%s' % shell_quote(' '.join(order)))
    elif a.artifacts:
        # --same-ticks-as: an ablation is compared with the baseline passes
        # at ITS OWN tick count.  A reduced profile's S6 also holds a
        # full-length subset (cf_full_seeds) for the same (arm, seed,
        # scenario) keys; mixing it in would compare a 600-tick ablation with
        # a 3600-tick baseline, whose series differ trivially.
        want = (None if a.same_ticks_as is None
                else by_id(a.profile, a.same_ticks_as).get('ticks'))
        print(','.join(i['artifact'] for i in build(a.profile)
                       if i['stage'] == a.artifacts
                       and i['artifact'].endswith('.pkl')
                       and (a.same_ticks_as is None
                            or i.get('ticks') == want)))
    elif a.analysis_role:
        print(' '.join(i['id'] for i in build(a.profile)
                       if i.get('analysis_role') == a.analysis_role))
    elif a.ids:
        want = None if a.ids == 'ALL' else a.ids
        print(' '.join(i['id'] for i in build(a.profile)
                       if want is None or i['stage'] == want))
    elif a.emit:
        print(emit(by_id(a.profile, a.emit)))
    elif a.keys:
        for k in by_id(a.profile, a.keys)['expected_keys']:
            print(k)
    elif a.field:
        v = by_id(a.profile, a.field[0]).get(a.field[1], '')
        print(' '.join(str(x) for x in v) if isinstance(v, list) else v)
    elif a.json:
        json.dump(build(a.profile), sys.stdout, indent=1)
        print('')
    else:
        invs = build(a.profile)
        tot_p = sum(i['expected_passes'] for i in invs)
        tot_h = sum(i['expected_hours'] for i in invs)
        print('profile %s: %d invocations, %d passes, %.2f h modelled compute'
              % (a.profile, len(invs), tot_p, tot_h))
        for s, title, deps, order in STAGES:
            mine = [i for i in invs if i['stage'] == s]
            print('  %-4s %-58s inv=%-2d passes=%-4d h=%5.2f deps=%s%s'
                  % (s, title[:58], len(mine),
                     sum(i['expected_passes'] for i in mine),
                     sum(i['expected_hours'] for i in mine),
                     ','.join(deps) or '-',
                     (' after=' + ','.join(order)) if order else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
