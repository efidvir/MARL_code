#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block D isolation latency after the severance (WP13 item 1, D13.1).

Runs Scenario A ('recovery') on the comparison topology with the WP3
link-state service attached as both drivers attach it, the uniform-random
reference policy acting, and reports, per agent site, the first tick at which
the service's own isolation flag is set (`LinkStateService.isolation_state`,
the value Block D observes one tick later) -- against the severance tick and
the engine's global island trigger.  Also the service's Router-LSA
origination count, which D13.1 changes.

    PYTHONHASHSEED=0 nice -n 19 python3 tools/isolation_latency.py \\
        --seed 82 --ticks 100 --severance 10 [--json out.json]

The report counts what it measured: the number of agent sites, how many were
flagged within the pass, and exits 2 when no site was flagged (nothing to
report a latency of).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_ENV = {
    'MARL_TRANSPORT_POWER': 'planning', 'MARL_RELAY_STATE': 'reconcile',
    'MARL_RELAY_REPOINT': 'fragment_aware', 'MARL_POSTCARD_ALWAYS': '1',
    'MARL_REWARD_PROFILE': 'mission', 'MARL_RELAY_ENTROPY_FLOOR': '0.01',
    'MARL_STEER_MODEL': 'actionable',
}


def measure(seed, ticks, severance):
    for k, v in EVAL_ENV.items():
        os.environ.setdefault(k, v)
    import torch
    argv = sys.argv
    sys.argv = ['run_timeline_comparison.py']
    try:
        from run_timeline_comparison import (build_comparison_topology,
                                             build_comparison_scenario)
    finally:
        sys.argv = argv
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.link_state_service import make_link_state_service

    topology, _info = build_comparison_topology(seed, log_prefix='[LAT] ')
    scenario = build_comparison_scenario(
        'recovery', seed, topology=topology, ticks=ticks,
        severance_tick=severance, rescue_tick=ticks - 1, scenario_seed=seed)
    sim = Simulator(topology, scenario,
                    SimulationConfig(enable_island_detection=True,
                                     verbose=False, random_seed=seed))
    sim._current_scenario_type = getattr(scenario, 'scenario_type',
                                         'recovery')
    sim.random_action_policy = True
    sim.marl_ue_routing_enabled = True
    torch.manual_seed(seed)
    svc = make_link_state_service(sim)
    if svc is None:
        raise SystemExit('MARL_DETECT_TICKS=0: no link-state service')

    def _act(tick, observations):
        sim._execute_agent_actions(observations)

    first_flag = {}
    first_island = None
    for tick in range(ticks):
        sim.step_tick(tick, _act, forwarding=svc, where='isolation_latency',
                      log_prefix='[LAT] ')
        if first_island is None and sim.island_mode:
            first_island = tick
        for site in sim.agents:
            if site not in first_flag and svc.isolation_state(site)[0]:
                first_flag[site] = tick
    sites = sorted(sim.agents)
    lat = sorted(first_flag[s] - severance for s in first_flag)
    rep = {
        'seed': seed, 'ticks': ticks, 'severance_tick': severance,
        'first_global_island_tick': first_island,
        'agent_sites': len(sites), 'flagged_sites': len(first_flag),
        'first_flag_tick_by_site': {s: first_flag.get(s) for s in sites},
        'latency_min': lat[0] if lat else None,
        'latency_median': lat[len(lat) // 2] if lat else None,
        'latency_max': lat[-1] if lat else None,
        'lsa_originations': getattr(svc, 'originations', None),
        'detections': svc.detections, 'installs': svc.installs,
    }
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seed', type=int, default=82)
    ap.add_argument('--ticks', type=int, default=100)
    ap.add_argument('--severance', type=int, default=10)
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    rep = measure(a.seed, a.ticks, a.severance)
    print('ISOLATION-LATENCY seed=%d severance=%d first_global_island=%s '
          'sites=%d flagged=%d latency min/median/max = %s/%s/%s '
          'lsa_originations=%s'
          % (rep['seed'], rep['severance_tick'],
             rep['first_global_island_tick'], rep['agent_sites'],
             rep['flagged_sites'], rep['latency_min'],
             rep['latency_median'], rep['latency_max'],
             rep['lsa_originations']))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=1, sort_keys=True)
    return 0 if rep['flagged_sites'] > 0 else 2


if __name__ == '__main__':
    sys.exit(main())
