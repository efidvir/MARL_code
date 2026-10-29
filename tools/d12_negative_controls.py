#!/usr/bin/env python3
"""Negative controls for the D12-FIXES package (docs/DECISIONS.md D12F.*).

    python3 tools/d12_negative_controls.py [--keep] [--list] [MUTATION ...]

The method and the runner are those of tools/wp13_negative_controls.py (and
tools/prefreeze_negative_controls.py): for every mutation, copy the tree's
sources into a scratch directory under output/d12_negctl/ (removed at the
end, also when the run is killed by SIGTERM or SIGHUP), re-introduce ONE
defect -- the behaviour before the fix -- by an exact string replacement
whose anchor must occur EXACTLY ONCE, run the named tests against the copy
in a fresh interpreter, and report which fail.  The unmutated copy runs
first and must be fully green, so a failure under a mutation is
attributable to that mutation alone.

A control is CAUGHT when every test it lists under `must_fail` fails.  The
run exits 0 only when the head is green, at least one control ran, every
mutation applied and every control was caught.  Every line of the report
counts what it checked.

The controls (item FIRST, D12F.2):

  negctl_signals_ignored  the runner no longer turns SIGTERM/SIGHUP into
                          SystemExit: a killed run leaves its copies.
  negctl_no_sweep         the runner no longer removes the copies of a dead
                          runner at start.
  tmp_default_dir         a converted test's scratch directory goes back to
                          tempfile's default directory (the tree root when
                          `/` is full).
  tmp_not_removed         a converted test's scratch directory is no longer
                          removed when the test ends.
  root_guard_blind        the tree-root guard no longer sees a tmp*
                          directory.

The controls (D12.1, D12F.3):

  ue_transit_restored     the UE-to-UE path and its alternates are searched
                          on the UE-inclusive graph again: a flow transits
                          a dual-anchored UE.
  sink_transit_restored   the general-traffic sink path is searched on the
                          UE-inclusive graph again.

The controls (D12.2, D12F.4):

  sdn_fixed_point         discovery replaces the frontier by its open
                          neighbourhood and completes only at a fixed point
                          (never on a bipartite fragment).
  sdn_shared_timer        one LLDP round of any fragment restarts every
                          fragment's round timer (the shared timer).
  sdn_ue_walk             UEs are switches again: fragments and the walk
                          go through them (UE stitching).
  sdn_requeue_blind       the operational re-queue queues every
                          unprogrammed flow, eligible or not.
  sdn_no_seed             the Scenario-B pre-converged controller gets no
                          discovered set ((c2) missing, also in the remap).
  sdn_seed_elected_only   the pre-converged seeding skips the fragments
                          without a leader (rider 1 broken).
  sdn_rebuild_on_force    a forced recompute clears the per-fragment state
                          instead of remapping it (rider 2 broken).

The controls (D12F.8, the leaf model for the other routing baselines):

  olsr_ue_walk            OLSR's fragment BFS walks every survivor, UEs
                          included, again (a dual-anchored UE stitches two
                          components; a recompute keeps stale fragment ids).
  batman_ue_walk          the same for B.A.T.M.A.N.
  aodv_ue_walk            the same for AODV.
  aodv_establish_via_ue   AODV's route establishment searches the
                          UE-inclusive graph again.
  aodv_ue_rebroadcast     a UE that hears a RREQ rebroadcasts it again.
  aodv_rrep_via_ue        the RREP is billed along the UE-inclusive path
                          again.

The control (XL-1, XL-DET's interference brake, D-XL.1):

  xldet_brake_latch       the brake holds a site below its half-rate cliff
                          again.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import wp13_negative_controls as NC  # noqa: E402
import prefreeze_negative_controls as PF  # noqa: E402

HYG = 'tests/test_zz_tree_hygiene.py'
UET = 'tests/test_d12_ue_transit.py'
SDN = 'tests/test_d12_sdn.py'
WP12 = 'tests/test_wp12_probes.py'
BAS = 'tests/test_d12_baselines.py'
XL1 = 'tests/test_xl1_xldet_brake.py'
RTC = 'run_timeline_comparison.py'

# test file -> the tests run against every copy
TESTS = {
    HYG: (
        'test_negctl_runner_removes_its_copies_when_it_is_killed',
        'test_negctl_runner_sweeps_only_the_copies_of_a_dead_runner',
        'test_scratch_directories_stay_in_tmp_tests_and_are_removed',
        'test_the_root_guard_finds_a_planted_tmp_directory',
    ),
    UET: (
        'test_no_path_of_any_forwarding_mode_has_an_interior_ue',
        'test_a_flow_across_fragments_joined_only_by_a_ue_is_not_delivered',
        'test_the_path_view_admits_only_the_flows_own_endpoints',
        'test_path_selection_does_not_depend_on_the_hash_seed',
        'test_every_data_plane_path_search_takes_a_path_view',
    ),
    SDN: (
        'test_discovery_terminates_on_a_bipartite_tree_after_ecc_plus_one_rounds',
        'test_two_fragments_discover_in_parallel_on_their_own_timers',
        'test_fragments_are_infrastructure_components_and_ues_their_leaves',
        'test_the_operational_requeue_waits_for_a_common_discovered_fragment',
        'test_scenario_b_seeds_every_fragment_and_programs_the_rescue_arrivals',
        'test_scenario_b_survives_a_recompute_that_renumbers_the_fragments',
    ),
    WP12: (
        'test_sdn_controller_discovers_a_bipartite_path_in_ecc_rounds',
        'test_sdn_controller_discovers_a_fragment_with_an_odd_cycle',
        'test_sdn_controller_fragments_discover_in_parallel_and_none_starves',
        'test_sdn_controller_keeps_ue_stitched_components_apart',
        'test_sdn_leader_rule_counts_up_links_not_distinct_peers',
    ),
    BAS: (
        'test_every_routing_baseline_uses_the_infrastructure_fragments_with_ue_leaves',
        'test_no_baseline_routes_a_flow_joined_only_by_a_ue',
        'test_aodv_establishes_no_route_through_a_ue_even_past_its_gate',
        'test_aodv_overhead_a_ue_does_not_rebroadcast_and_no_rrep_through_a_ue',
    ),
    XL1: (
        'test_the_brake_cannot_hold_a_site_below_its_half_rate_cliff',
        'test_a_braked_fleet_climbs_back_from_a_step_down',
    ),
}

# The fragment BFS of OLSR, B.A.T.M.A.N. and AODV before D12F.8 (every
# survivor, UEs included, along every up link; no reset of the dict).
_OLD_BFS = (
    '        if self._fragments_computed and not force:\n'
    '            return\n'
    '        self._fragments_computed = True\n'
    '        survivor_nodes = [  # NEGCTL\n'
    '            nid for nid, n in self.sim.topology.nodes.items()\n'
    '            if n.is_survivor and n.node_type.value not in {"Core", "UPF", "AMF", "CoreUPF"}\n'
    '        ]\n'
    '        visited = set()\n'
    '        frag_id = 0\n'
    '        for start in survivor_nodes:\n'
    '            if start in visited:\n'
    '                continue\n'
    '            component = set()\n'
    '            queue = [start]\n'
    '            while queue:\n'
    '                nid = queue.pop(0)\n'
    '                if nid in visited:\n'
    '                    continue\n'
    '                visited.add(nid)\n'
    '                component.add(nid)\n'
    '                for lid, lnk in self.sim.topology.links.items():\n'
    "                    if not getattr(lnk, 'is_up', True):\n"
    '                        continue\n'
    '                    ep = lnk.endpoints\n'
    '                    peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)\n'
    '                    if peer and peer in survivor_nodes and peer not in visited:\n'
    '                        queue.append(peer)\n'
    '            self.fragment_members[frag_id] = component\n'
    '            frag_id += 1\n')
_NEW_BODY = (
    '        if self._fragments_computed and not force:\n'
    '            return\n'
    '        self._fragments_computed = True\n'
    '        self.fragment_members = infra_fragments_with_ue_leaves(self.sim.topology)\n')


def _walk(proto):
    head = ('        \"\"\"The %s control plane\'s fragments: OSPF\'s decomposition,\n' % proto)
    tail = '        the previous decomposition\'s higher fragment ids.\"\"\"\n'
    return (head, tail)

# name -> (file, anchor, replacement, {test file: tests that MUST fail}, what)
MUTATIONS = {
    'negctl_signals_ignored': (
        'tools/wp13_negative_controls.py',
        '    with _signals_end_the_run():\n'
        '        return _main_run(a, names, plan, MUTATIONS, base)\n',
        '    with contextlib.nullcontext():  # NEGCTL\n'
        '        return _main_run(a, names, plan, MUTATIONS, base)\n',
        {HYG: ('test_negctl_runner_removes_its_copies_when_it_is_killed',)},
        'SIGTERM/SIGHUP end the runner without unwinding (before D12F.2): '
        'its scratch copies stay'),
    'negctl_no_sweep': (
        'tools/wp13_negative_controls.py',
        '    removed, live, unowned = _sweep_stale(base)\n',
        '    removed, live, unowned = [], [], []  # NEGCTL\n',
        {HYG: ('test_negctl_runner_sweeps_only_the_copies_of_a_dead_runner',)},
        'the copies of a runner killed by SIGKILL are never removed'),
    'tmp_default_dir': (
        'tests/test_run_chain.py',
        '    return tempfile.mkdtemp(prefix=prefix, dir=base)\n',
        '    return tempfile.mkdtemp(prefix=prefix)  # NEGCTL\n',
        {HYG: ('test_scratch_directories_stay_in_tmp_tests_and_are_removed',)},
        'the chain tests make their scratch directories in tempfile\'s '
        'default directory again (before D12F.2): the tree root on a full /'),
    'tmp_not_removed': (
        'tests/test_run_chain.py',
        '        yield d\n    finally:\n        _rmtree(d)\n',
        '        yield d\n    finally:\n        pass  # NEGCTL\n',
        {HYG: ('test_scratch_directories_stay_in_tmp_tests_and_are_removed',)},
        'the chain tests no longer remove their scratch directories'),
    'root_guard_blind': (
        HYG,
        "             if n.startswith('tmp') and os.path.isdir(os.path.join(root, n))]\n",
        "             if n.startswith('tmp') and False]  # NEGCTL\n",
        {HYG: ('test_the_root_guard_finds_a_planted_tmp_directory',)},
        'the tree-root guard cannot see a tmp* directory'),
    'ue_transit_restored': (
        'sixg_sim/simulation.py',
        (('sixg_sim/simulation.py',
          '                    path = nx.shortest_path(_pv, source_ue, target_ue)',
          '                    path = nx.shortest_path(self.topology.graph, source_ue, target_ue)  # NEGCTL'),
         ('sixg_sim/simulation.py',
          '                                    _pv, source_ue, target_ue):',
          '                                    self.topology.graph, source_ue, target_ue):  # NEGCTL')),
        None,
        {UET: ('test_no_path_of_any_forwarding_mode_has_an_interior_ue',
               'test_a_flow_across_fragments_joined_only_by_a_ue_is_not_delivered',
               'test_every_data_plane_path_search_takes_a_path_view')},
        'the UE-to-UE path and its alternates on the UE-inclusive graph '
        '(before D12.1): a dual-anchored UE carries the flow'),
    'sink_transit_restored': (
        'sixg_sim/simulation.py',
        '            _sv = self.topology.path_view((src,), infra=_infra_nodes)',
        '            _sv = self.topology.graph  # NEGCTL',
        {UET: ('test_no_path_of_any_forwarding_mode_has_an_interior_ue',)},
        'the general-traffic sink path on the UE-inclusive graph (before '
        'D12.1)'),
    'sdn_fixed_point': (
        RTC,
        ((RTC,
          '                        if peer in walk and peer not in disc:\n'
          '                            layer.add(peer)\n'
          '                if not layer:\n',
          '                        if peer in walk:  # NEGCTL\n'
          '                            layer.add(peer)\n'
          '                if layer == self.discovery_frontier.get(fid):\n'),
         (RTC,
          '                    disc |= layer\n',
          '                    disc.clear(); disc |= layer  # NEGCTL\n')),
        None,
        {SDN: ('test_discovery_terminates_on_a_bipartite_tree_after_ecc_plus_one_rounds',
               'test_two_fragments_discover_in_parallel_on_their_own_timers'),
         WP12: ('test_sdn_controller_discovers_a_bipartite_path_in_ecc_rounds',
                'test_sdn_controller_fragments_discover_in_parallel_and_none_starves')},
        'the fixed-point termination test (before D12.2 a): a bipartite '
        'fragment is never discovered'),
    'sdn_shared_timer': (
        RTC,
        '            if (tick - self._discovery_round_tick[fid]) >= self.TICKS_PER_LLDP_ROUND:\n'
        '                self._discovery_round_tick[fid] = tick\n',
        '            if (tick - self._discovery_round_tick[fid]) >= self.TICKS_PER_LLDP_ROUND:\n'
        '                self._discovery_round_tick = {k: tick for k in self.fragment_leaders}  # NEGCTL\n',
        {SDN: ('test_two_fragments_discover_in_parallel_on_their_own_timers',)},
        'one round timer for all fragments (before D12.2 a): the first '
        'fragment\'s rounds hold the others back'),
    'sdn_ue_walk': (
        RTC,
        '            if is_ue(nid):\n'
        '                ues.add(nid)\n'
        '            else:\n'
        '                switches.append(nid)\n',
        '            switches.append(nid)  # NEGCTL: UEs walked as switches\n',
        {SDN: ('test_fragments_are_infrastructure_components_and_ues_their_leaves',),
         WP12: ('test_sdn_controller_keeps_ue_stitched_components_apart',)},
        'fragments and the LLDP walk go through UEs again (before D12.2 b): '
        'a dual-anchored UE stitches two components'),
    'sdn_requeue_blind': (
        RTC,
        '                eligible = [flow for flow in new_flows if self._eligible(flow)]\n',
        '                eligible = list(new_flows)  # NEGCTL\n',
        {SDN: ('test_the_operational_requeue_waits_for_a_common_discovered_fragment',)},
        'the operational re-queue programs every unprogrammed flow (before '
        'D12.2 c): flows with no common discovered fragment included'),
    'sdn_no_seed': (
        RTC,
        ((RTC,
          '            self.discovered[fid] = set(self.fragment_switches[fid])\n',
          '            pass  # NEGCTL: no (c2) seeding\n'),
         (RTC,
          '                if self._pre_converged and not self._walk_started:\n'
          '                    new[\'done\'][fid] = True\n'
          '                    new[\'discovered\'][fid] = set(sw)\n'
          '                continue\n',
          '                continue  # NEGCTL\n'),
         (RTC,
          '            if self._pre_converged and not self._walk_started:\n'
          '                new[\'done\'][fid] = True\n'
          '                new[\'discovered\'][fid] = set(sw)\n'
          '                continue\n'
          '            if disc:\n',
          '            if disc:  # NEGCTL\n')),
        None,
        {SDN: ('test_scenario_b_seeds_every_fragment_and_programs_the_rescue_arrivals',
               'test_scenario_b_survives_a_recompute_that_renumbers_the_fragments')},
        'the pre-converged controller has no discovered set (before D12.2 '
        'c2): the eligibility test drops every rescue-arrival flow'),
    'sdn_seed_elected_only': (
        RTC,
        ((RTC,
          '        for fid in self.fragment_members:\n'
          '            self.discovery_done[fid] = True\n'
          '            self.discovered[fid] = set(self.fragment_switches[fid])\n',
          '        for fid in self.fragment_leaders:  # NEGCTL\n'
          '            self.discovery_done[fid] = True\n'
          '            self.discovered[fid] = set(self.fragment_switches[fid])\n'),
         (RTC,
          '                if self._pre_converged and not self._walk_started:\n'
          '                    new[\'done\'][fid] = True\n',
          '                if self._pre_converged and not self._walk_started and fid in new[\'leaders\']:  # NEGCTL\n'
          '                    new[\'done\'][fid] = True\n')),
        None,
        {SDN: ('test_scenario_b_seeds_every_fragment_and_programs_the_rescue_arrivals',)},
        'the pre-converged seeding covers elected fragments only (rider 1): '
        'a flow served only by a leaderless fragment is dropped'),
    'sdn_rebuild_on_force': (
        RTC,
        '        self._remap(old, comps, updeg)\n',
        '        self.fragment_leaders = {}  # NEGCTL: rebuild, not remap\n'
        '        self.election_done_tick = {}\n'
        '        self.discovery_frontier = {}\n'
        '        self.discovered = {}\n'
        '        self.discovery_done = {}\n'
        '        self._discovery_round_tick = {}\n',
        {SDN: ('test_scenario_b_seeds_every_fragment_and_programs_the_rescue_arrivals',
               'test_scenario_b_survives_a_recompute_that_renumbers_the_fragments',
               'test_the_operational_requeue_waits_for_a_common_discovered_fragment')},
        'a forced recompute clears the per-fragment state (rider 2): the '
        'pre-converged controller loses its discovered sets'),
}


def _ue_walk(proto, must_fail, what):
    head, tail = _walk(proto)
    mid = ('        `infra_fragments_with_ue_leaves` -- infrastructure components over\n'
           '        live infrastructure links, every surviving UE a leaf of each\n'
           '        fragment holding one of its anchors (D12F.8).  Until D12F.8 this\n'
           '        BFS walked every survivor, UEs included, so a dual-anchored UE\n'
           '        stitched two components into one fragment, and a recompute kept\n')
    anchor = head + mid + tail + _NEW_BODY
    return (RTC, anchor, head + mid + tail + _OLD_BFS, must_fail, what)


MUTATIONS.update({
    'olsr_ue_walk': _ue_walk(
        'OLSR',
        {BAS: ('test_every_routing_baseline_uses_the_infrastructure_fragments_with_ue_leaves',
               'test_no_baseline_routes_a_flow_joined_only_by_a_ue')},
        "OLSR's fragments walk UEs again (before D12F.8): a dual-anchored UE "
        'stitches two components and the cross flow becomes routable'),
    'batman_ue_walk': _ue_walk(
        'B.A.T.M.A.N.',
        {BAS: ('test_every_routing_baseline_uses_the_infrastructure_fragments_with_ue_leaves',
               'test_no_baseline_routes_a_flow_joined_only_by_a_ue')},
        "B.A.T.M.A.N.'s fragments walk UEs again (before D12F.8): a "
        'dual-anchored UE stitches two components and the cross flow '
        'becomes routable'),
    'aodv_ue_walk': _ue_walk(
        'AODV',
        {BAS: ('test_every_routing_baseline_uses_the_infrastructure_fragments_with_ue_leaves',)},
        "AODV's fragments walk UEs again (before D12F.8): its RREQ queue "
        'takes the cross flow (the establishment check alone still refuses it)'),
    'aodv_establish_via_ue': (
        RTC,
        '                    nx.shortest_path(self.sim.topology.path_view((src, tgt)),\n'
        '                                     src, tgt)\n',
        '                    nx.shortest_path(self.sim.topology.graph, src, tgt)  # NEGCTL\n',
        {BAS: ('test_aodv_establishes_no_route_through_a_ue_even_past_its_gate',),
         UET: ('test_every_data_plane_path_search_takes_a_path_view',)},
        "AODV's route establishment on the UE-inclusive graph (before "
        'D12F.8): a route through a UE is established'),
    'aodv_ue_rebroadcast': (
        RTC,
        '                                if not is_ue(nbr):\n'
        '                                    next_frontier.add(nbr)\n',
        '                                if True:  # NEGCTL\n'
        '                                    next_frontier.add(nbr)\n',
        {BAS: ('test_aodv_overhead_a_ue_does_not_rebroadcast_and_no_rrep_through_a_ue',)},
        'a UE that hears a RREQ rebroadcasts it (before D12F.8)'),
    'aodv_rrep_via_ue': (
        RTC,
        '                    path_len = nx.shortest_path_length(\n'
        '                        self.sim.topology.path_view((src, tgt)), src, tgt)\n',
        '                    path_len = nx.shortest_path_length(graph, src, tgt)  # NEGCTL\n',
        {BAS: ('test_aodv_overhead_a_ue_does_not_rebroadcast_and_no_rrep_through_a_ue',),
         UET: ('test_every_data_plane_path_search_takes_a_path_view',)},
        'the RREP billed along the UE-inclusive path (before D12F.8)'),
    'xldet_brake_latch': (
        'sixg_sim/heuristic_controller.py',
        '    if g < SINR_TARGET_DB and (g < ICIC_FLOOR_DB or not _icic_braked(st, own_drop_db)):\n',
        '    if g < SINR_TARGET_DB and not _icic_braked(st, own_drop_db):  # NEGCTL\n',
        {XL1: ('test_the_brake_cannot_hold_a_site_below_its_half_rate_cliff',
               'test_a_braked_fleet_climbs_back_from_a_step_down')},
        "XL-DET's interference brake holds a site below its half-rate cliff "
        'again (before XL-1): a braked fleet stays at the power floor'),
})


def main(argv):
    # the copies need the `.synced_files` listing of the FIX-PREFREEZE
    # runner: the launcher tests run run_final.sh in the copy
    NC._copy_tree = PF._copy_tree_with_listing
    return NC.main(argv, tests=TESTS, table=MUTATIONS, title='D12-FIXES',
                   scratch='d12_negctl', doc=__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
