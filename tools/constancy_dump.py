#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation-field constancy dump (RERUN_SPEC WP6 / WP13).

WP6 fixed a 26-field actor vector and a 41-field critic vector, but the spec
makes that list final only after a constancy dump has shown that no field is
dead.  WP13 ran it at production length (3600 ticks, both scenarios, three
seeds) and removed the four critic-global fields constant in every pass
(docs/DECISIONS.md D13.3; tools/constancy_combine.py combines the reports):
the actor stays 26 wide, the critic is 37.  This is that dump: it runs a pass per scenario and reports, for every
field of `sixg_sim.agent.ACTOR_OBS_FIELDS` and `CRITIC_GLOBAL_FIELDS`,

    min, max, mean, variance, the modal value, and the FRACTION OF SAMPLES
    AT THAT MODAL VALUE (1.0 = the field never moved),

over every (tick, agent) sample of the run.  A field whose constant fraction
is 1.0 carries no information and must be removed from the list (or its
writer fixed) before the schema is frozen.

It does NOT need a checkpoint: the pass runs the environment with the
uniform-random reference policy (`Simulator.random_action_policy`), which is
the arm that exercises the widest range of actions and therefore the widest
range of observations.  Every per-tick phase is the WP2 `Simulator.step_tick`
order, and the pass attaches the WP3 link-state service exactly as both
drivers do (`make_link_state_service`, None only under the MARL_DETECT_TICKS=0
sensitivity), so the fields -- Block D's local isolation and Block C's live
infrastructure neighbours included -- are exactly what a driver would see.
(Before the trackAgent merge the pass ran with no service, which is not the
MARL code path any more.)  A pass that collects no sample raises instead of
reporting an empty field as "constant".

Usage (<compute-server>, inside the run tree):

    PYTHONHASHSEED=0 MARL_POOL_WORKERS=8 nice -n 19 \\
      python3 tools/constancy_dump.py --ticks 600 --seed 82 \\
      --out output/constancy_wp6.json

Add `--scenario recovery` or `--scenario rescue_ops` to run just one.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stats(values):
    n = len(values)
    if n == 0:
        return None
    lo = min(values)
    hi = max(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    modal_value, modal_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return {
        'n': n,
        'min': lo,
        'max': hi,
        'mean': mean,
        'variance': var,
        'distinct': len(counts),
        'modal_value': modal_value,
        'constant_fraction': modal_n / n,
        'constant': (lo == hi),
    }


def run_pass(scenario_kind, seed, ticks, log_prefix='[CONST] '):
    """One environment pass; returns {field_name: [value, ...]}."""
    import torch
    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario)
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.link_state_service import make_link_state_service
    from sixg_sim.agent import (ACTOR_OBS_FIELDS, CRITIC_GLOBAL_FIELDS,
                                SCHEMA_ID)

    fields = list(ACTOR_OBS_FIELDS) + list(CRITIC_GLOBAL_FIELDS)
    samples = {name: [] for name, _fn in fields}

    topology, _info = build_comparison_topology(seed, log_prefix=log_prefix)
    severance = min(200, max(50, ticks // 6))
    scenario = build_comparison_scenario(
        scenario_kind, seed, topology=topology, ticks=ticks,
        severance_tick=severance, scenario_seed=seed)
    config = SimulationConfig(enable_island_detection=True, verbose=False,
                              random_seed=int(seed))
    sim = Simulator(topology, scenario, config)
    sim._current_scenario_type = getattr(scenario, 'scenario_type',
                                         scenario_kind)
    sim.random_action_policy = True      # uniform reference policy
    sim.marl_ue_routing_enabled = True
    torch.manual_seed(seed)
    # the arm's forwarding view, exactly as both drivers build it
    forwarding = make_link_state_service(sim)

    def _collect(tick, observations):
        for obs in observations.values():
            for name, fn in fields:
                samples[name].append(float(fn(obs)))

    def _act(tick, observations):
        sim._execute_agent_actions(observations)

    for tick in range(scenario.duration_ticks):
        sim.step_tick(tick, _act, forwarding=forwarding,
                      hooks={'after_observe': _collect},
                      where='constancy_dump', log_prefix=log_prefix)

    empty = sorted(n for n, v in samples.items() if not v)
    if empty:
        raise SystemExit('constancy_dump: no sample collected for %s -- '
                         'nothing was measured' % ', '.join(empty))
    return samples, {'schema_id': SCHEMA_ID, 'severance_tick': severance,
                     'ticks': ticks, 'seed': seed,
                     'scenario': scenario_kind,
                     'link_state_service': forwarding is not None,
                     'samples_per_field': len(next(iter(samples.values())))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ticks', type=int, default=600)
    ap.add_argument('--seed', type=int, default=82)
    ap.add_argument('--scenario', action='append',
                    choices=['recovery', 'rescue_ops'])
    ap.add_argument('--out', default=None,
                    help='write the full report as JSON here')
    args = ap.parse_args(argv)

    from sixg_sim import runinfo
    from sixg_sim.agent import ACTOR_OBS_FIELDS, CRITIC_GLOBAL_FIELDS
    runinfo.check_hashseed('constancy_dump')
    ordered = [n for n, _ in ACTOR_OBS_FIELDS] + [n for n, _ in
                                                  CRITIC_GLOBAL_FIELDS]

    scenarios = args.scenario or ['recovery', 'rescue_ops']
    report = {'provenance': runinfo.provenance(
        extra={'driver': 'constancy_dump'}), 'passes': []}

    for kind in scenarios:
        samples, meta = run_pass(kind, args.seed, args.ticks)
        per_field = {name: _stats(vals) for name, vals in samples.items()}
        meta['fields'] = per_field
        report['passes'].append(meta)

        print('\n=== %s  seed %d  %d ticks  schema %s ==='
              % (kind, args.seed, args.ticks, meta['schema_id']))
        print('%-32s %9s %9s %11s %8s %8s  %s'
              % ('field', 'min', 'max', 'variance', 'distinct', 'const%',
                 'VERDICT'))
        for name in ordered:
            s = per_field[name]
            verdict = 'CONSTANT -- DEAD FIELD' if s['constant'] else (
                'near-constant' if s['constant_fraction'] > 0.999 else '')
            print('%-32s %9.4f %9.4f %11.3e %8d %7.1f%%  %s'
                  % (name, s['min'], s['max'], s['variance'], s['distinct'],
                     100.0 * s['constant_fraction'], verdict))

    # A field is DEAD only if it never moved in ANY pass (the intersection);
    # one that is constant in one scenario and not in another still carries
    # information and stays in the list.
    dead = sorted(name for name in ordered
                  if all(p['fields'][name]['constant']
                         for p in report['passes']))
    somewhere = sorted({name for p in report['passes']
                        for name, s in p['fields'].items() if s['constant']}
                       - set(dead))
    report['constant_in_every_pass'] = dead
    report['constant_in_some_pass'] = somewhere
    print('\nCONSTANT IN EVERY PASS (remove from the schema): %s'
          % (', '.join(dead) or 'none'))
    print('CONSTANT IN SOME PASS ONLY (keep; scenario-dependent): %s'
          % (', '.join(somewhere) or 'none'))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
