#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-TERM constancy dump (RERUN_SPEC WP8).

WP6's `tools/constancy_dump.py` answers the same question for OBSERVATION
fields; this is its counterpart for the reward.  RERUN_SPEC WP8 says "terms
shown constant by the constancy dump (e.g. r_qos) are removed and listed in
the pre-registration", and a decision record is not evidence, so this is the
measurement.

It runs a pass per scenario with the uniform-random reference policy (no
checkpoint needed -- it is the arm that exercises the widest range of
actions, hence the widest range of reward states) through the WP2
`Simulator.step_tick` order, and reports, for

  * every term of `reward_spec.global_reward` (one sample per tick), and
  * every term of `reward_spec.local_reward` (one sample per tick per agent),

min, max, mean, variance, the number of distinct values, the modal value and
the FRACTION OF SAMPLES AT IT (1.0 = the term never moved).

It ALSO reports the PROBES for the terms WP8 removed, so that the removal is
evidenced rather than asserted:

  removed_r_qos          what the deleted life-safety term WOULD have paid
                         (2.0 x the node's LIFE_SAFETY admission_success_rate
                         when that slice has offered load, else 0.0)
  probe_ls_offered_load  the LIFE_SAFETY offered load itself, i.e. the reason
  removed_relay_contrib  the INDICATOR the deleted flat per-relay-node term
                         multiplied (relay mode on AND own link carried
                         traffic).  This one is NOT constant -- the term was
                         removed because its COEFFICIENT is zero in the
                         evaluated configuration, not because the indicator
                         never moved, and the dump says so instead of
                         pretending otherwise.
  removed_bridge_held    the INDICATOR the deleted local bridge-held term
                         multiplied (this node's own relay link is in the
                         de-duplicated bridge set).  Same status.

Usage (<compute-server>, inside the run tree):

    PYTHONHASHSEED=0 MARL_POOL_WORKERS=8 nice -n 19 \\
      python3 tools/reward_term_dump.py --ticks 600 --seed 82 \\
      --out output/reward_terms_wp8.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.constancy_dump import _stats          # noqa: E402  (same stats)

# The life-safety rate the DELETED r_qos term used, kept here (and only here)
# so the probe reproduces what the term would have paid.
_REMOVED_QOS_POINTS = 2.0


def _local_probe_values(agent, obs):
    """The removed-term probes for one agent observation."""
    from sixg_sim.topology import TrafficClass as TC
    ps = obs.phy_mac
    ls_load = 0.0
    qos = 0.0
    if obs.local_slices:
        ls = obs.local_slices.get(TC.LIFE_SAFETY)
        if ls is not None:
            ls_load = float(getattr(ls, 'offered_load', 0.0) or 0.0)
            if ls_load > 0.0 and hasattr(ls, 'admission_success_rate'):
                qos = _REMOVED_QOS_POINTS * float(ls.admission_success_rate)
    from sixg_sim.reward_spec import executed_relay_mode
    from sixg_sim.phy_mac_state import RelayMode
    carried = float(getattr(ps, 'relay_traffic_carried_mbps', 0.0) or 0.0)
    in_relay = executed_relay_mode(ps) in (RelayMode.LOCAL_REROUTE,
                                           RelayMode.CAPACITY_BOOST)
    return {
        'removed_r_qos': qos,
        'probe_ls_offered_load': ls_load,
        'removed_relay_contrib': 1.0 if (in_relay and carried > 0.0) else 0.0,
        'removed_bridge_held':
            1.0 if getattr(ps, 'relay_link_is_bridge', False) else 0.0,
    }


def run_pass(scenario_kind, seed, ticks, log_prefix='[RTERM] '):
    import torch
    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario)
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim import reward_spec as RS

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
    # The arm's forwarding view, exactly as both drivers build it (trackAgent
    # merge: before it the pass ran with no link-state service, which is not
    # the MARL code path any more).  None only under MARL_DETECT_TICKS=0.
    from sixg_sim.link_state_service import make_link_state_service
    forwarding = make_link_state_service(sim)

    g_samples = {n: [] for n in RS.GLOBAL_TERM_NAMES}
    g_samples['TOTAL'] = []
    l_samples = {n: [] for n in RS.LOCAL_TERM_NAMES}
    l_samples['TOTAL'] = []
    for probe in ('removed_r_qos', 'probe_ls_offered_load',
                  'removed_relay_contrib', 'removed_bridge_held'):
        l_samples[probe] = []

    state = {'obs': None}

    def _capture(tick, observations):
        state['obs'] = observations

    def _act(tick, observations):
        sim._execute_agent_actions(observations)

    for tick in range(scenario.duration_ticks):
        sim.step_tick(tick, _act, forwarding=forwarding,
                      hooks={'after_observe': _capture},
                      where='reward_term_dump', log_prefix=log_prefix)

        # TEAM objective: one sample per tick, from the live method, read off
        # the per-term diagnostic the method publishes.
        total = sim.compute_global_connectivity_reward()
        terms = dict(sim._reward_terms)
        for n in RS.GLOBAL_TERM_NAMES:
            g_samples[n].append(float(terms.get(n, 0.0)))
        g_samples['TOTAL'].append(float(total))

        # AGENT-LOCAL: one sample per tick per agent, from the live method.
        observations = state['obs'] or {}
        for aid, obs in observations.items():
            agent = sim.agents.get(aid)
            act = getattr(agent, 'last_action', None)
            if agent is None or act is None or not hasattr(agent,
                                                           'calculate_reward'):
                continue
            comps = agent.calculate_reward(obs, act, obs)
            lt = dict(agent._reward_terms)
            for n in RS.LOCAL_TERM_NAMES:
                l_samples[n].append(float(lt.get(n, 0.0)))
            l_samples['TOTAL'].append(float(comps.total_reward))
            for k, v in _local_probe_values(agent, obs).items():
                l_samples[k].append(float(v))

    meta = {'scenario': scenario_kind, 'seed': seed, 'ticks': ticks,
            'severance_tick': severance,
            'reward_id': RS.reward_id()}
    return g_samples, l_samples, meta


def _report(title, samples, ordered):
    print('\n--- %s ---' % title)
    print('%-26s %11s %11s %12s %8s %8s  %s'
          % ('term', 'min', 'max', 'variance', 'distinct', 'const%',
             'VERDICT'))
    out = {}
    for name in ordered:
        s = _stats(samples[name])
        out[name] = s
        if s is None:
            print('%-26s  (no samples)' % name)
            continue
        verdict = 'CONSTANT' if s['constant'] else (
            'near-constant' if s['constant_fraction'] > 0.999 else '')
        print('%-26s %11.4f %11.4f %12.3e %8d %7.1f%%  %s'
              % (name, s['min'], s['max'], s['variance'], s['distinct'],
                 100.0 * s['constant_fraction'], verdict))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ticks', type=int, default=600)
    ap.add_argument('--seed', type=int, default=82)
    ap.add_argument('--scenario', action='append',
                    choices=['recovery', 'rescue_ops'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    from sixg_sim import runinfo, reward_spec as RS
    runinfo.check_hashseed('reward_term_dump')

    scenarios = args.scenario or ['recovery', 'rescue_ops']
    report = {'provenance': runinfo.provenance(
        extra={'driver': 'reward_term_dump'}),
        'coefficients': RS.FROZEN.as_dict(),
        'removed_terms': [list(t) for t in RS.REMOVED_TERMS],
        'passes': []}

    g_order = list(RS.GLOBAL_TERM_NAMES) + ['TOTAL']
    l_order = list(RS.LOCAL_TERM_NAMES) + ['TOTAL', 'removed_r_qos',
                                           'probe_ls_offered_load',
                                           'removed_relay_contrib',
                                           'removed_bridge_held']

    for kind in scenarios:
        g, l, meta = run_pass(kind, args.seed, args.ticks)
        print('\n=== %s  seed %d  %d ticks  reward %s ==='
              % (kind, args.seed, args.ticks, meta['reward_id']))
        meta['global'] = _report('TEAM OBJECTIVE (per tick)', g, g_order)
        meta['local'] = _report('AGENT-LOCAL (per tick per agent)', l, l_order)
        report['passes'].append(meta)

    def _const_everywhere(section, name):
        return all(p[section][name] is not None and p[section][name]['constant']
                   for p in report['passes'])

    dead_g = [n for n in RS.GLOBAL_TERM_NAMES
              if _const_everywhere('global', n)]
    dead_l = [n for n in RS.LOCAL_TERM_NAMES if _const_everywhere('local', n)]
    report['constant_in_every_pass'] = {'global': dead_g, 'local': dead_l}
    print('\nSURVIVING TERMS CONSTANT IN EVERY PASS '
          '(they must be removed too): global=%s local=%s'
          % (', '.join(dead_g) or 'none', ', '.join(dead_l) or 'none'))
    for probe in ('removed_r_qos', 'probe_ls_offered_load',
                  'removed_relay_contrib', 'removed_bridge_held'):
        vals = [p['local'][probe] for p in report['passes']]
        print('  %-24s constant in every pass: %s   (max seen %r)'
              % (probe, all(v['constant'] for v in vals),
                 max(v['max'] for v in vals)))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1, sort_keys=True)
        print('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
