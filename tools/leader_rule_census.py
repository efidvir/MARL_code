#!/usr/bin/env python3
"""Census of parallel links, and of the two fragment-leader election rules,
over the real topologies of this study (WP12 / D12.3 item 4).

Why it exists.  `diag/sdn_discovery_probe.py` elects the would-be fragment
leader by up-LINK degree in both of its paths: `sdn_fragments` /
`static_case` (the static predictor) and `_fragment_table` (the per-fragment
table of the instrumented `sdn-run`, which is what
`docs/wp12/wp12_sdn_probe_report.txt` prints as `infra_comp_ecc` and
`infra_comp_leaders`).  Before the D12.3 round `_fragment_table` used a
different rule: the number of DISTINCT peers in the collapsed adjacency.
The two rules differ only where a node pair carries more than one link,
because `Topology.update_link_statuses` (`sixg_sim/topology.py:408-414`)
collapses parallel links into one graph edge while the engine's election
(`run_timeline_comparison.py:2199-2203`) counts link incidences.

D12.3 item 4 asserted, without checking, that "the model has parallel links
between one node pair" and that the committed evidence was safe only
because no parallel link fell on a tied-degree candidate.  That is the
claim this tool replaces with a measurement.  It also closes the gap that
`_fragment_table`'s change was verified only by a unit test: regenerating
the run report from the existing pickle replays a fragment table recorded
by the OLD rule and cannot exercise the new one, so the rule is exercised
here instead, on the post-severance topology of every (seed, scenario).

What it measures, per (seed, scenario) of the requested seed set:

  1. parallel links in the RAW topology: unordered node pairs carrying
     more than one link in `topology.links`, whatever their state;
  2. parallel links in the POST-SEVERANCE graph the election actually runs
     on: the same count over the links that survive the scenario's cuts,
     are up, and do not touch a core node (`sever_core` downs those);
  3. for every controller fragment (the `ue-stitched` model, i.e.
     `_compute_fragments` as it is) the infra-only decomposition of that
     fragment, and on every component of at least
     `MIN_FRAGMENT_FOR_ELECTION` switches -- exactly `_fragment_table`'s
     loop -- the leader and the leader's BFS eccentricity under BOTH rules:
       new: `leader_of(comp, up-link incidence)`   (the engine's rule)
       old: `max(sorted(comp), key=lambda n: len(neighbors(n)))`
            (distinct peers in the collapsed adjacency, the pre-fix rule)
     and counts the components where the two disagree on the leader or on
     its eccentricity.

A divergence count of 0 is what licenses the committed
`docs/wp12/wp12_sdn_probe_report.txt`; it is not an assumption, and it is
re-measured whenever this tool is run.  This is a static replica: it runs
no simulator, loads no checkpoint and writes nothing outside `--out`.

Usage:
  python3 tools/leader_rule_census.py [--seeds 80-99]
      [--scenarios recovery,rescue_ops] [--out output/wp12/leader_census.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diag'))

import sdn_discovery_probe as sp          # noqa: E402


def _pair(u, v):
    return (u, v) if u <= v else (v, u)


def census_case(topology, cut_link_ids, flows):
    """Both parallel-link counts and the two-rule comparison for one
    (seed, scenario).  Mirrors `static_case`'s graph construction and
    `_fragment_table`'s election loop."""
    core = {nid for nid, n in topology.nodes.items()
            if n.node_type.value in sp.CORE_TYPES}
    cut = set(cut_link_ids or ())

    # (1) parallel links in the raw topology, every link, any state
    raw_mult = {}
    for lk in topology.links.values():
        u, v = lk.endpoints
        raw_mult[_pair(u, v)] = raw_mult.get(_pair(u, v), 0) + 1

    # (2) the post-severance graph the election runs on
    adj = {}
    updeg = {}
    post_mult = {}
    for lid, lk in topology.links.items():
        if lid in cut or not getattr(lk, 'is_up', True):
            continue
        u, v = lk.endpoints
        if u in core or v in core:
            continue                      # sever_core downs these links
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        updeg[u] = updeg.get(u, 0) + 1    # link INCIDENCE, the engine's rule
        updeg[v] = updeg.get(v, 0) + 1
        post_mult[_pair(u, v)] = post_mult.get(_pair(u, v), 0) + 1

    node_ids = [nid for nid, n in topology.nodes.items()
                if n.is_survivor and nid not in core
                and n.node_type.value not in sp.SDN_EXCLUDED_TYPES]
    is_ue = topology.is_ue_node

    def neighbors(n):
        return sorted(adj.get(n, ()))

    # the controller's own fragments (ue-stitched = _compute_fragments as
    # it is), then _fragment_table's per-fragment infra decomposition
    frags = sp.sdn_fragments(node_ids, is_ue, neighbors,
                             lambda n: updeg.get(n, 0), model='ue-stitched')
    comps = []
    for fr in frags:
        members = set(fr['members'])
        intra = [(s, t) for s, t in flows
                 if s in members and t in members]
        dec = sp.infra_decomposition(sorted(members), is_ue, neighbors, intra)
        for comp in dec['comps']:
            if len(comp) < sp.MIN_FRAGMENT_FOR_ELECTION:
                comps.append({'fid': fr['fid'], 'size': len(comp),
                              'elected': False})
                continue
            new = sp.leader_of(comp, lambda n: updeg.get(n, 0))
            # the pre-fix rule of _fragment_table, verbatim: distinct peers
            # in the collapsed adjacency, same lexicographic tie-break
            old = max(sorted(comp), key=lambda n: len(neighbors(n)))
            ecc_new = sp.bfs_rounds(neighbors, new, set(comp))[0]
            ecc_old = sp.bfs_rounds(neighbors, old, set(comp))[0]
            comps.append({
                'fid': fr['fid'], 'size': len(comp), 'elected': True,
                'leader_uplink_degree': new, 'ecc_uplink_degree': ecc_new,
                'leader_distinct_peers': old, 'ecc_distinct_peers': ecc_old,
                'diverges': bool(new != old or ecc_new != ecc_old)})
    return {
        'raw_parallel_pairs': sorted(p for p, m in raw_mult.items() if m > 1),
        'raw_max_multiplicity': max(raw_mult.values()) if raw_mult else 0,
        'raw_links': len(topology.links),
        'post_parallel_pairs': sorted(p for p, m in post_mult.items() if m > 1),
        'post_max_multiplicity': (max(post_mult.values())
                                  if post_mult else 0),
        'post_links': sum(post_mult.values()),
        'n_fragments': len(frags),
        'components': comps,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', default='80-99')
    ap.add_argument('--scenarios', default='recovery,rescue_ops')
    ap.add_argument('--out', default='output/wp12/leader_census.json')
    args = ap.parse_args(argv)

    os.chdir(ROOT)
    import run_timeline_comparison as rtc
    seeds = rtc.parse_seed_list(args.seeds)
    scenarios = [s for s in args.scenarios.split(',') if s]

    rows = []
    for seed in seeds:
        topology, _info = rtc.build_comparison_topology(seed)
        flows = rtc._replicate_ue_flows(topology)
        for scen in scenarios:
            scenario = rtc.build_comparison_scenario(scen, seed,
                                                     topology=topology)
            cut = rtc.scenario_cut_link_ids(scenario)
            r = census_case(topology, cut, flows)
            r['seed'] = seed
            r['scenario'] = scen
            rows.append(r)

    elected = [c for r in rows for c in r['components'] if c['elected']]
    diverging = [(r['seed'], r['scenario'], c)
                 for r in rows for c in r['components']
                 if c.get('diverges')]
    seeds_with_raw = sorted({r['seed'] for r in rows
                             if r['raw_parallel_pairs']})
    cases_with_post = sorted((r['seed'], r['scenario']) for r in rows
                             if r['post_parallel_pairs'])

    lines = [
        "WP12 leader-rule census (D12.3 item 4): parallel links, and the "
        "up-link-degree election rule against the distinct-peer rule",
        "seeds %s, scenarios %s -- %d cases, %d topologies, no simulator"
        % (args.seeds, args.scenarios, len(rows), len(seeds)),
        "",
        "parallel links (node pairs carrying more than one link):",
        "  raw topology:        %d of %d seeds have any (max multiplicity "
        "%d over %d links)"
        % (len(seeds_with_raw), len(seeds),
           max((r['raw_max_multiplicity'] for r in rows), default=0),
           max((r['raw_links'] for r in rows), default=0)),
        "  post-severance graph: %d of %d cases have any (this is the graph "
        "the election runs on)" % (len(cases_with_post), len(rows)),
        "",
        "election rule over every fragment component with >= %d switches "
        "(the _fragment_table loop):" % sp.MIN_FRAGMENT_FOR_ELECTION,
        "  %d components elect a leader; divergences (leader or "
        "eccentricity): %d" % (len(elected), len(diverging)),
        "",
        "per case: seed scen | parallel raw/post | components "
        "(size leader ecc)",
    ]
    for r in rows:
        desc = '; '.join(
            ("%d %s %s" % (c['size'], c['leader_uplink_degree'],
                           c['ecc_uplink_degree']))
            if c['elected'] else "%d no-election" % c['size']
            for c in r['components'])
        lines.append("  %4d %-10s | %d/%d | %s"
                     % (r['seed'], r['scenario'],
                        len(r['raw_parallel_pairs']),
                        len(r['post_parallel_pairs']), desc))
    lines.append("")
    if diverging:
        lines.append("DIVERGENT COMPONENTS (the two rules disagree -- the "
                     "committed report is NOT safe as it stands):")
        for seed, scen, c in diverging:
            lines.append("  %4d %-10s fid %s size %d: up-link %s ecc %s "
                         "vs distinct-peer %s ecc %s"
                         % (seed, scen, c['fid'], c['size'],
                            c['leader_uplink_degree'], c['ecc_uplink_degree'],
                            c['leader_distinct_peers'],
                            c['ecc_distinct_peers']))
    else:
        lines.append("no divergent component: on this seed set the two "
                     "rules elect the same leader with the same "
                     "eccentricity everywhere, so the pre-fix "
                     "_fragment_table and the fixed one produce the same "
                     "infra_comp_leaders / infra_comp_ecc.")
    text = '\n'.join(lines)
    print(text)

    out = args.out
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'seeds': seeds, 'scenarios': scenarios,
                       'n_cases': len(rows),
                       'n_elected_components': len(elected),
                       'n_divergences': len(diverging),
                       'seeds_with_raw_parallel_links': seeds_with_raw,
                       'cases_with_post_parallel_links': cases_with_post,
                       'rows': rows}, f, indent=1)
        with open(os.path.splitext(out)[0] + '.txt', 'w') as f:
            f.write(text + '\n')
        print("\nwritten: %s{,.txt}" % os.path.splitext(out)[0])
    # exit 1 if the census refutes what the docs rest on
    return 1 if diverging else 0


if __name__ == '__main__':
    raise SystemExit(main())
