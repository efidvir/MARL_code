#!/usr/bin/env python3
"""WP12(b): why the SDN arm never recovers on confirmatory seeds 87 and 98.

Frozen fact (~/MARL_run_5/output/cf_eval_80_99.pkl): in Scenario A the SDN
arm delivers 0 on every one of the 3400 post-severance ticks of seeds 87
and 98 (relay links 0, controller never operational); every other seed
recovers 24-31 ticks after the severance, and the same two seeds recover
normally in Scenario B (sever_core only, no fragmenting cuts).

Hypothesis (from reading `SDNController.step` in run_timeline_comparison.py):
the LLDP discovery replaces the frontier by its open neighbourhood,
S <- N(S), and declares a fragment discovered only when N(S) == S.  On a
fragment without an odd cycle (a bipartite fragment: a tree, or a mesh whose
cycles are all even) N^k({leader}) alternates between the two colour classes
and never reaches a fixed point, so discovery never completes.  A single
`_discovery_round_tick` is shared by all fragments, so the fragment that is
stuck consumes every round and no later fragment advances either.  The
controller then stays in phase 'discovery' for ever, `programmed_flows`
stays empty and the harness forwards no UE-to-UE flow.

Second finding (review of D12.2): `_compute_fragments` and the LLDP walk
traverse UE nodes.  The recovery cuts partition the INFRASTRUCTURE into
2-4 components (FRAG_MIN_FRAGMENTS..FRAG_MAX_FRAGMENTS), but a UE attached
to O-RUs of two components stitches them into one controller "fragment",
so every recovery seed shows a single 34/150 fragment.  The controller then
"discovers" switches reachable only through a UE and programs flows whose
endpoints have no infrastructure path -- flows the data-plane gate
`Simulator._can_route_ue_to_ue` (infrastructure-only reachability between
the endpoints' anchor O-RUs) never forwards.  The static predictor therefore
reports, for every case, the infra-only decomposition (component sizes,
UEs anchored in more than one component, flows without an infrastructure
path) and can replay the controller under either fragment model:

  --fragment-model ue-stitched   (default) `_compute_fragments` as it is
  --fragment-model infra-only    the leaf model of D12.1/D12.2: fragments
                                 are infra components, UEs leaves attached
                                 to every fragment holding one of their
                                 anchors, the discovery walk never enters
                                 a UE
  --iteration fixed-point        (default) the discovery test as it is
  --iteration bfs                the cumulative BFS requested in D12.2

Three checks:

  static   replicate `SDNController._compute_fragments` and the discovery
           iteration on the post-severance topology of every seed (default
           80-99, both scenarios) WITHOUT running the simulator, and predict
           which seeds never recover.  The prediction (ue-stitched,
           fixed-point) must match the frozen set {87A, 98A} exactly.
  run      instrumented sdn re-run (monkeypatch only) on the given seeds:
           per-tick controller phase and per-fragment frontier sizes, the
           bipartiteness of each fragment as the controller computed it and
           its infra-only decomposition on the live topology.
  report   summarise a `run` pickle.

Read-only with respect to the engine: nothing under sixg_sim/ or the
harness is edited.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Simulator._identify_core_nodes: these become non-survivors at sever_core
# and every link incident to them goes down.
CORE_TYPES = {"Core", "SMO", "Non-RT-RIC", "AMF", "UPF"}
# SDNController._compute_fragments excludes these types explicitly
# (the others above are excluded through is_survivor == False).
SDN_EXCLUDED_TYPES = {"Core", "UPF", "AMF", "CoreUPF"}
MIN_FRAGMENT_FOR_ELECTION = 3


# ─────────────────────────────────────────────────────────────────────────
#  Pure helpers (unit-tested)
# ─────────────────────────────────────────────────────────────────────────
def lldp_rounds_to_fixed_point(neighbors, leader, members, max_rounds=1000):
    """Replicate the discovery iteration of SDNController.step.

    frontier <- {leader}; each round frontier <- N(frontier) restricted to
    `members`; stop when N(frontier) == frontier.  Returns (rounds, sizes)
    where rounds is None if no fixed point was reached within max_rounds;
    sizes is the sequence of frontier sizes (the oscillation is visible in
    it)."""
    frontier = {leader}
    sizes = [1]
    for r in range(1, max_rounds + 1):
        new = set()
        for nid in frontier:
            for peer in neighbors(nid):
                if peer in members:
                    new.add(peer)
        sizes.append(len(new))
        if new == frontier:
            return r, sizes
        frontier = new
    return None, sizes


FRAGMENT_MODELS = ('ue-stitched', 'infra-only')
ITERATIONS = ('fixed-point', 'bfs')


def infra_components(node_ids, is_ue, neighbors):
    """Connected components of the INFRASTRUCTURE-ONLY graph (a UE is never
    traversed), in first-appearance order of `node_ids`; each a sorted
    list.  This is the component structure the scenario generator counts
    (`_post_severance_infra_graph`) and the one the data-plane gate
    `Simulator._can_route_ue_to_ue` reasons over."""
    visited = set()
    comps = []
    node_set = set(node_ids)
    for start in node_ids:
        if start in visited or is_ue(start):
            continue
        comp = []
        queue = [start]
        visited.add(start)
        while queue:
            nid = queue.pop(0)
            comp.append(nid)
            for peer in neighbors(nid):
                if (peer in node_set and not is_ue(peer)
                        and peer not in visited):
                    visited.add(peer)
                    queue.append(peer)
        comps.append(sorted(comp))
    return comps


def ue_anchor_components(node_ids, is_ue, neighbors, comps):
    """For every UE in `node_ids`: the sorted list of indices (into
    `comps`) of the infra components its live infrastructure neighbours
    belong to -- the anchor set of `_can_route_ue_to_ue`."""
    comp_of = {n: i for i, c in enumerate(comps) for n in c}
    out = {}
    for n in node_ids:
        if is_ue(n):
            out[n] = sorted({comp_of[p] for p in neighbors(n) if p in comp_of})
    return out


def classify_flows(flows, ue_comps):
    """Data-plane deliverability of each (src, tgt) flow under the leaf
    model.  'intra': the two anchor sets share an infra component, i.e.
    an infrastructure-only path exists (exactly the `_can_route_ue_to_ue`
    gate); 'cross': they do not (no arm forwards the flow until a bridge
    joins the components); 'unanchored': an endpoint has no live
    infrastructure neighbour.  Returns (counts, flows per component that
    carry it as intra)."""
    counts = {'intra': 0, 'cross': 0, 'unanchored': 0}
    per_comp = {}
    for s, t in flows:
        a = set(ue_comps.get(s, ()))
        b = set(ue_comps.get(t, ()))
        if not a or not b:
            counts['unanchored'] += 1
        elif a & b:
            counts['intra'] += 1
            for c in sorted(a & b):
                per_comp[c] = per_comp.get(c, 0) + 1
        else:
            counts['cross'] += 1
    return counts, per_comp


def infra_decomposition(node_ids, is_ue, neighbors, flows):
    """The infra-only view of one post-severance topology: component
    sizes, bipartiteness of each component (over its infra nodes), the
    UEs anchored in more than one component (the stitches) and the flow
    classes.  Pure; used by the static predictor and by the instrumented
    run alike."""
    comps = infra_components(node_ids, is_ue, neighbors)
    ue_comps = ue_anchor_components(node_ids, is_ue, neighbors, comps)
    counts, per_comp = classify_flows(flows, ue_comps)
    multi = sorted(u for u, cs in ue_comps.items() if len(cs) > 1)
    return {
        'comp_sizes': [len(c) for c in comps],
        'comp_bipartite': [is_bipartite_component(set(c), neighbors)
                           for c in comps],
        'n_ue': sum(1 for n in node_ids if is_ue(n)),
        'ues_multi_anchored': multi,
        'n_ues_multi_anchored': len(multi),
        'n_ues_unanchored': sum(1 for cs in ue_comps.values() if not cs),
        'flows': counts,
        'flows_intra_per_comp': per_comp,
        'n_flows': len(flows),
        'comps': comps,
        'ue_comps': ue_comps,
    }


def leader_of(infra, degree):
    """The would-be fragment leader under the model's own election rule,
    written once for both probe paths: the lexicographically first
    infrastructure node of maximum UP-LINK degree.

    `degree` must count up LINKS incident to the node, exactly as the
    engine does (`run_timeline_comparison.py:2199-2203`,
    `sum(1 for l in links.values() if is_up and nid in l.endpoints)`), NOT
    the number of distinct peers.  The model PERMITS parallel links between
    a node pair -- `Topology.update_link_statuses`
    (`sixg_sim/topology.py:408-414`) collapses them into one graph edge --
    and on such a pair the two counts differ, so the two rules can elect
    different leaders and report different eccentricities.  That divergence
    is what handoff acceptance item 1 ("the engine's fragment table must
    coincide with the predictor's") would have compared.

    NO instance of this study contains one.  `tools/leader_rule_census.py`
    counts them and compares the two rules over the real data: 0 node pairs
    carry more than one link in the raw topology (20 of 20 seeds of 80-99)
    and 0 in the post-severance graph the election runs on (40 of 40
    cases), and over the 62 infra components of those 40 cases the two
    rules elect the SAME node with the SAME eccentricity every time (0
    divergences).  The rule is written once here because the two probe
    paths must not be ABLE to diverge, not because they do on the committed
    evidence; the census, not an assumption about tied degrees, is what
    says the committed numbers are safe.

    `max` over a SORTED sequence returns the first maximal element, which
    is the engine's `max(sorted(infra_in_frag), key=...)` tie-break."""
    infra = sorted(infra)
    if not infra:
        return None
    return max(infra, key=degree)


def sdn_fragments(node_ids, is_ue, neighbors, degree, model='ue-stitched'):
    """Fragments as the controller computes them.

    model='ue-stitched' replicates SDNController._compute_fragments as it
    is in the baseline (d94a936): BFS over every survivor node, UEs
    included, so a UE attached to two infrastructure components joins them
    into one fragment and the discovery walk passes through UEs.
    model='infra-only' is the leaf model requested in D12.2: fragments are
    the infra components, every UE is a leaf member of each fragment that
    holds one of its anchors (a dual-anchored UE belongs to two fragments)
    and the discovery walk never enters a UE.  Election rule in both
    (unchanged): >= MIN_FRAGMENT_FOR_ELECTION infra nodes; leader = the
    lexicographically first infra node of maximum up-link degree.  `fid`
    follows first appearance in `node_ids`.  Each fragment carries
    'members' (flow-eligibility set) and 'walk' (the nodes the discovery
    may traverse)."""
    if model not in FRAGMENT_MODELS:
        raise ValueError(model)
    frags = []
    node_set = set(node_ids)
    if model == 'ue-stitched':
        visited = set()
        for start in node_ids:
            if start in visited:
                continue
            comp = set()
            queue = [start]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                comp.add(nid)
                for peer in neighbors(nid):
                    if peer in node_set and peer not in visited:
                        queue.append(peer)
            infra = sorted(n for n in comp if not is_ue(n))
            frags.append((comp, comp, infra))
    else:
        comps = infra_components(node_ids, is_ue, neighbors)
        ue_comps = ue_anchor_components(node_ids, is_ue, neighbors, comps)
        for i, infra in enumerate(comps):
            leaves = {u for u, cs in ue_comps.items() if i in cs}
            frags.append((set(infra) | leaves, set(infra), infra))
    out = []
    for members, walk, infra in frags:
        leader = None
        if len(infra) >= MIN_FRAGMENT_FOR_ELECTION:
            leader = leader_of(infra, degree)
        out.append({'fid': len(out), 'members': members, 'walk': walk,
                    'n_members': len(members), 'n_infra': len(infra),
                    'n_ue': len(members) - len(infra), 'leader': leader})
    return out


def is_bipartite_component(members, neighbors):
    """2-colouring by BFS over the induced subgraph."""
    colour = {}
    for start in sorted(members):
        if start in colour:
            continue
        colour[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in neighbors(u):
                if v not in members:
                    continue
                if v not in colour:
                    colour[v] = 1 - colour[u]
                    queue.append(v)
                elif colour[v] == colour[u]:
                    return False
    return True


def bfs_walk(neighbors, leader, members):
    """Cumulative discovery from `leader` over the induced subgraph
    (discovered <- discovered u N(frontier), frontier = the newly
    discovered layer, stop when a round adds nothing).  Returns (rounds,
    reached set).

    It stops after `rounds` rounds whether or not it has covered
    `members`: on a members set that is NOT connected from the leader it
    terminates having reached only the leader's component.  Callers must
    test `reached` against `members` -- see `predict`."""
    seen = {leader}
    frontier = {leader}
    rounds = 0
    while frontier:
        new = set()
        for nid in frontier:
            for peer in neighbors(nid):
                if peer in members and peer not in seen:
                    seen.add(peer)
                    new.add(peer)
        if not new:
            break
        rounds += 1
        frontier = new
    return rounds, seen


def bfs_rounds(neighbors, leader, members):
    """Rounds a CUMULATIVE discovery needs: the BFS eccentricity of the
    leader in the induced subgraph.  This is the recommended replacement
    iteration (D12.2); it terminates on every fragment, and covers the
    whole fragment only when the fragment is connected from the leader.
    Returns (rounds, n_reached) -- `n_reached < len(members)` says the
    walk did not cover the fragment."""
    rounds, seen = bfs_walk(neighbors, leader, members)
    return rounds, len(seen)


def predict(frags, neighbors, flows, max_rounds=1000,
            iteration='fixed-point'):
    """Discovery outcome in fid order, with the shared-round-timer
    serialisation: the first eligible fragment whose iteration never
    terminates blocks every later fragment.  Flows of fragments completed
    before the block are programmable (counted once even when a flow is
    intra in two fragments of the infra-only model).  The walk is the
    fragment's 'walk' set (all members, or infra members only); with
    iteration='bfs' the cumulative iteration of D12.2 is assumed, which
    terminates on every fragment after `bfs_rounds` rounds.

    Completion is DERIVED, never assumed, on BOTH iterations.
    TERMINATION and COVERAGE are different things and the probe keeps them
    apart: the fixed-point test terminates only on a walk that contains an
    odd cycle, a cumulative BFS terminates on every walk, and NEITHER
    covers the fragment unless the walk set is connected from the leader.
    A walk that is not connected stops with part of the fragment
    undiscovered whichever iteration ran it.  Such a fragment gets status
    'partial-walk' under either iteration, contributes no programmable
    flows (the probe must not count a fragment's flows on a discovery that
    never reached part of it) and, unlike a non-terminating fixed point,
    does NOT block later fragments, because its round timer does stop.
    Only a fixed point that never terminates is 'never', and only 'never'
    starves the fragments after it.  Neither of today's two fragment models
    can produce a disconnected walk -- the ue-stitched walk is a BFS
    component and the infra-only walk is an infra component, both connected
    by construction -- so 'partial-walk' never fires on the committed
    evidence under either iteration; it exists so that a future fragment
    model cannot report a partial discovery as done."""
    if iteration not in ITERATIONS:
        raise ValueError(iteration)
    stuck = None
    partial_walk = []
    programmable = set()
    for fr in frags:
        walk = set(fr.get('walk', fr['members']))
        fr['eligible'] = fr['leader'] is not None
        fr['bipartite'] = is_bipartite_component(walk, neighbors)
        intra = [(s, t) for s, t in flows
                 if s in fr['members'] and t in fr['members']]
        fr['flows_intra'] = len(intra)
        if not fr['eligible']:
            fr['rounds'] = None
            fr['bfs_rounds'] = None
            fr['bfs_reached'] = None
            fr['walk_covered'] = None
            fr['status'] = 'no-election'
            continue
        rounds, sizes = lldp_rounds_to_fixed_point(
            neighbors, fr['leader'], walk, max_rounds)
        fr['rounds'] = rounds
        fr['bfs_rounds'], _reached = bfs_rounds(neighbors, fr['leader'], walk)
        fr['bfs_reached'] = _reached
        # the cumulative walk covers the fragment iff the walk set is
        # connected from the leader; nothing else may stand in for this
        fr['walk_covered'] = (_reached == len(walk))
        fr['size_tail'] = sizes[-6:]
        # the iteration stopped: a fixed point within max_rounds, or a
        # cumulative BFS, which stops on every walk
        terminates = ((rounds is not None) if iteration == 'fixed-point'
                      else True)
        # ... and it covered the fragment only if the walk is connected.
        # Both conditions, on both branches: a fixed point reached over a
        # disconnected walk is the same partial discovery as a BFS one.
        completes = terminates and fr['walk_covered']
        if stuck is not None:
            fr['status'] = 'starved'
        elif completes:
            fr['status'] = 'done'
            programmable.update(intra)
        elif terminates:
            # terminates, but reached only part of its own walk: its flows
            # are NOT programmable, and it starves nobody
            fr['status'] = 'partial-walk'
            partial_walk.append(fr['fid'])
        else:
            fr['status'] = 'never'
            stuck = fr['fid']
    return {'stuck_fid': stuck, 'flows_programmable': len(programmable),
            'never_recovers': (not programmable
                               and any(f['eligible'] for f in frags)),
            'partial_walk_fids': partial_walk,
            'iteration': iteration}


# ─────────────────────────────────────────────────────────────────────────
#  static: post-severance topology without a simulator
# ─────────────────────────────────────────────────────────────────────────
def static_case(topology, cut_link_ids, flows, model='ue-stitched',
                iteration='fixed-point'):
    core = {nid for nid, n in topology.nodes.items()
            if n.node_type.value in CORE_TYPES}
    cut = set(cut_link_ids or ())
    adj = {}
    deg = {}
    for lid, lk in topology.links.items():
        if lid in cut or not getattr(lk, 'is_up', True):
            continue
        u, v = lk.endpoints
        if u in core or v in core:
            continue                    # sever_core downs these links
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    node_ids = [nid for nid, n in topology.nodes.items()
                if n.is_survivor and nid not in core
                and n.node_type.value not in SDN_EXCLUDED_TYPES]
    is_ue = topology.is_ue_node

    def neighbors(n):
        return sorted(adj.get(n, ()))
    frags = sdn_fragments(node_ids, is_ue, neighbors, lambda n: deg.get(n, 0),
                          model=model)
    verdict = predict(frags, neighbors, flows, iteration=iteration)
    infra = infra_decomposition(node_ids, is_ue, neighbors, flows)
    return frags, verdict, infra


def verdict_word(v):
    """The per-row 'predicted' column of `cmd_static`.

    A row whose fragment's discovery terminated without covering the
    fragment is NOT 'ok': it is named here as well as in the trailing
    PARTIAL WALK line, so that the per-row verdict cannot read as a clean
    prediction.  'NEVER' (nothing programmable at all) and 'partial' (a
    non-terminating fragment starved the ones after it) keep their
    meanings and their precedence."""
    if v['never_recovers']:
        return 'NEVER'
    if v['stuck_fid'] is not None:
        return 'partial'
    if v.get('partial_walk_fids'):
        return 'part-walk'
    return 'ok'


def _infra_desc(infra):
    return "%s bip %s ue>1 %d cross %d/%d" % (
        infra['comp_sizes'],
        ''.join('T' if b else 'F' for b in infra['comp_bipartite']),
        infra['n_ues_multi_anchored'], infra['flows']['cross'],
        infra['n_flows'])


def cmd_static(args):
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
            frags, verdict, infra = static_case(
                topology, cut, flows, model=args.fragment_model,
                iteration=args.iteration)
            rows.append({'seed': seed, 'scenario': scen, 'cut_links': cut,
                         'n_flows': len(flows), 'verdict': verdict,
                         'infra': {k: v for k, v in infra.items()
                                   if k not in ('comps', 'ue_comps')},
                         'fragments': [
                             {k: (sorted(v) if isinstance(v, set) else v)
                              for k, v in fr.items()
                              if k not in ('members', 'walk')}
                             for fr in frags]})
    if args.fragment_model == 'ue-stitched':
        model_desc = ('SDNController._compute_fragments as it is: BFS over '
                      'UEs too, a dual-anchored UE stitches two infra '
                      'components')
    else:
        model_desc = ('D12.2 leaf model: fragments = infra components, UEs '
                      'leaves, the discovery walk never enters a UE')
    if args.iteration == 'fixed-point':
        iter_desc = ('frontier <- N(frontier), stop at N(S) == S; never = '
                     'no fixed point in 1000 rounds')
    else:
        iter_desc = 'cumulative BFS, complete after ecc(leader) rounds'
    lines = ["WP12(b) static SDN-discovery prediction (post-severance "
             "topology, no simulator)",
             "fragment model: %s (%s); iteration: %s (%s)"
             % (args.fragment_model, model_desc, args.iteration, iter_desc),
             "columns: infra-only components = components of the "
             "infrastructure graph after the cuts [sizes], bipartite per "
             "component (T/F), UEs anchored in >1 component (the stitches), "
             "flows without an infrastructure path (never forwarded by any "
             "arm: the _can_route_ue_to_ue gate) / all flows; fragments = "
             "the controller's fragments under the chosen model: infra/UE "
             "members, bipartite over the walk, rounds of the fixed-point "
             "test ('never' = none), ecc = cumulative-BFS rounds, flows "
             "with both endpoints in the fragment, status", ""]
    hdr = "%4s %-10s %4s | %-9s | %-36s | %s" % (
        'seed', 'scenario', 'cuts', 'predicted', 'infra-only components',
        'eligible fragments (fid: infra/UE bipartite rounds ecc flows '
        'status)')
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for r in rows:
        v = r['verdict']
        elig = [f for f in r['fragments'] if f['eligible']]
        desc = '; '.join("%d: %d/%d %s %s %s %d %s" % (
            f['fid'], f['n_infra'], f['n_ue'],
            'BIP' if f['bipartite'] else 'odd',
            'never' if f['rounds'] is None else f['rounds'],
            f['bfs_rounds'],
            f['flows_intra'], f['status']) for f in elig)
        lines.append("%4d %-10s %4d | %-9s | %-36s | %s"
                     % (r['seed'], r['scenario'], len(r['cut_links']),
                        verdict_word(v),
                        _infra_desc(r['infra']), desc))
    partial_walk = sorted((r['seed'], r['scenario']) for r in rows
                          if r['verdict'].get('partial_walk_fids'))
    never = sorted((r['seed'], r['scenario']) for r in rows
                   if r['verdict']['never_recovers'])
    partial = sorted((r['seed'], r['scenario']) for r in rows
                     if not r['verdict']['never_recovers']
                     and r['verdict']['stuck_fid'] is not None)
    stitched = sorted((r['seed'], r['scenario']) for r in rows
                      if r['infra']['n_ues_multi_anchored'] > 0)
    multi_frag = sorted((r['seed'], r['scenario']) for r in rows
                        if len(r['infra']['comp_sizes']) > 1)
    lines.append("")
    if partial_walk:
        # never fires on the two fragment models of this study (both walks
        # are connected by construction); printed only when it does, so the
        # committed sdn_static*.txt stay reproducible from this code
        lines.append("PARTIAL WALK (a fragment's discovery terminated "
                     "without covering the fragment): %s" % partial_walk)
    lines.append("predicted never-recovering: %s" % never)
    lines.append("predicted partially blocked (later fragments starved): %s"
                 % partial)
    lines.append("cases with >1 infra-only component: %d of %d; with a UE "
                 "anchored in >1 component (stitched): %d of %d"
                 % (len(multi_frag), len(rows), len(stitched), len(rows)))
    text = '\n'.join(lines)
    print(text)
    out_dir = os.path.join(ROOT, 'output', 'wp12')
    os.makedirs(out_dir, exist_ok=True)
    name = args.out_name
    with open(os.path.join(out_dir, name + '.json'), 'w') as f:
        json.dump({'seeds': seeds, 'scenarios': scenarios,
                   'fragment_model': args.fragment_model,
                   'iteration': args.iteration, 'rows': rows,
                   'never': never, 'partial': partial,
                   'stitched': stitched, 'multi_fragment': multi_frag,
                   **({'partial_walk': partial_walk} if partial_walk else {})},
                  f, indent=1)
    with open(os.path.join(out_dir, name + '.txt'), 'w') as f:
        f.write(text + '\n')
    print("\nwritten: %s/%s.{json,txt}" % (out_dir, name))


# ─────────────────────────────────────────────────────────────────────────
#  run: instrumented sdn passes through the harness (monkeypatch only)
# ─────────────────────────────────────────────────────────────────────────
class _Probe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fragments = None   # table logged at _compute_fragments
        # the FIRST table of the pass (D12-FIXES): since D12.2 the
        # operational re-queue recomputes the fragments on every tick that
        # holds an unprogrammed flow, so `fragments` ends as the LAST table
        # (after any relay bridge merged components); the post-severance
        # decomposition that handoff 7.2 acceptance item 1 compares with the
        # static predictor is this one
        self.fragments_first = None
        self.fragments_first_tick = None
        self.log = []           # per-tick rows after the severance
        self.frontier_hist = {}  # fid -> list of frontier sizes (per tick)


P = _Probe()


def _fragment_table(ctrl):
    sim = ctrl.sim
    links = sim.topology.links
    is_ue = sim.topology.is_ue_node
    adj = {}
    for lk in links.values():
        if not getattr(lk, 'is_up', True):
            continue
        u, v = lk.endpoints
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    def neighbors(n):
        return sorted(adj.get(n, ()))
    # up-LINK degree, as the engine's election rule counts it: parallel
    # links between one node pair count twice, distinct peers do not (see
    # leader_of).  `adj` above is the collapsed graph and must not be used
    # for the election.
    updeg = {}
    for lk in links.values():
        if not getattr(lk, 'is_up', True):
            continue
        for nid in lk.endpoints:
            updeg[nid] = updeg.get(nid, 0) + 1
    flows = list(getattr(sim, '_proto_all_flows', None)
                 or sim.ue_to_ue_flows)
    table = []
    for fid, members in ctrl.fragment_members.items():
        infra = [n for n in members if not is_ue(n)]
        leader = ctrl.fragment_leaders.get(fid)
        rounds = None
        ecc = None
        if leader is not None:
            rounds, _sizes = lldp_rounds_to_fixed_point(
                neighbors, leader, members, 1000)
            ecc, _reached = bfs_rounds(neighbors, leader, members)
        # the infra-only decomposition of THIS fragment on the live
        # topology: what the leaf model would have made of it
        intra_flows = [(s, t) for s, t in flows
                       if s in members and t in members]
        dec = infra_decomposition(sorted(members), is_ue, neighbors,
                                  intra_flows)
        ecc_infra = []
        infra_leaders = []
        for comp in dec['comps']:
            if len(comp) >= MIN_FRAGMENT_FOR_ELECTION:
                cl = leader_of(comp, lambda n: updeg.get(n, 0))
                infra_leaders.append(cl)
                ecc_infra.append(bfs_rounds(neighbors, cl, set(comp))[0])
            else:
                infra_leaders.append(None)
                ecc_infra.append(None)
        table.append({'fid': fid, 'n_members': len(members),
                      'n_infra': len(infra), 'n_ue': len(members) - len(infra),
                      'leader': leader,
                      'election_done_tick': ctrl.election_done_tick.get(fid),
                      'bipartite': is_bipartite_component(members, neighbors),
                      'predicted_rounds': rounds,
                      'bfs_rounds': ecc,
                      'flows_intra': len(intra_flows),
                      'infra_comp_sizes': dec['comp_sizes'],
                      'infra_comp_bipartite': dec['comp_bipartite'],
                      'infra_comp_ecc': ecc_infra,
                      'infra_comp_leaders': infra_leaders,
                      'n_ues_multi_anchored': dec['n_ues_multi_anchored'],
                      'flows_with_infra_path': dec['flows']['intra'],
                      'flows_cross': dec['flows']['cross'],
                      'flows_unanchored': dec['flows']['unanchored']})
    return table


def install():
    if getattr(install, 'done', False):
        return
    import run_timeline_comparison as rtc
    C = rtc.SDNController
    orig_cf = C._compute_fragments
    orig_step = C.step
    orig_rpp = rtc.run_physics_pass

    def _compute_fragments(ctrl, force=False):
        before = ctrl._fragments_computed
        orig_cf(ctrl, force=force)
        if force or not before:
            P.fragments = _fragment_table(ctrl)
            P.fragments_tick = getattr(ctrl.sim, 'current_tick', None)
            if P.fragments_first is None:
                P.fragments_first = P.fragments
                P.fragments_first_tick = P.fragments_tick
    C._compute_fragments = _compute_fragments

    def step(ctrl, tick):
        orig_step(ctrl, tick)
        if tick < ctrl.severance_tick:
            return
        sizes = {}
        for fid in ctrl.fragment_leaders:
            fr = ctrl.discovery_frontier.get(fid)
            sizes[fid] = (len(fr) if fr is not None else -1)
            P.frontier_hist.setdefault(fid, []).append(sizes[fid])
        P.log.append({'tick': int(tick), 'phase': ctrl.phase,
                      'programmed': len(ctrl.programmed_flows),
                      'queued': len(ctrl.flow_queue),
                      'done': sum(1 for fid in ctrl.fragment_leaders
                                  if ctrl.discovery_done.get(fid, False)),
                      'frontier': sizes})
    C.step = step

    def run_physics_pass(mode, topology_seed, checkpoint_path,
                         scenario_name='recovery'):
        P.reset()
        data = orig_rpp(mode, topology_seed, checkpoint_path, scenario_name)
        if isinstance(data, dict) and mode == 'sdn':
            data['sdn_probe'] = {
                'fragments': P.fragments,
                'fragments_tick': getattr(P, 'fragments_tick', None),
                'fragments_first': P.fragments_first,
                'fragments_first_tick': P.fragments_first_tick,
                'log': P.log,
                'frontier_hist': {int(k): v for k, v in P.frontier_hist.items()},
            }
        return data
    rtc.run_physics_pass = run_physics_pass
    install.done = True


def cmd_run(args):
    import multiprocessing
    os.chdir(ROOT)
    multiprocessing.set_start_method('fork', force=True)
    install()
    import run_timeline_comparison as rtc
    print("[SDN-PROBE] patches installed; seeds=%s ticks=%s"
          % (args.seeds, args.ticks), flush=True)
    overrides = {'total_ticks': int(args.ticks)} if args.ticks else {}
    rtc.main(overrides=overrides, seed_list=args.seeds, out_file=args.out,
             force_overwrite=True, arms='sdn', no_figures=True)
    report(os.path.join(rtc.OUTPUT_DIR, args.out),
           out_dir=os.path.join(rtc.OUTPUT_DIR, 'wp12'))


# ─────────────────────────────────────────────────────────────────────────
#  report
# ─────────────────────────────────────────────────────────────────────────
def summarise_pass(run, scenario, seed, severance_tick):
    pr = run.get('sdn_probe')
    if not pr:
        return None
    log = pr['log']
    delivered = run.get('delivered') or []
    post = [v for v in delivered[severance_tick:] if isinstance(v, (int, float))]
    first_delivery = next((i + severance_tick for i, v in
                           enumerate(delivered[severance_tick:]) if v and v > 0),
                          None)
    phases = [r['phase'] for r in log]
    first_disc = next((r['tick'] for r in log if r['phase'] == 'discovery'),
                      None)
    first_oper = next((r['tick'] for r in log if r['phase'] == 'operational'),
                      None)
    frag_rows = []
    for fr in (pr['fragments'] or []):
        hist = pr['frontier_hist'].get(fr['fid'], [])
        tail = hist[-8:]
        # period-2 oscillation: sizes alternate a,b,a,b over the last 20 rounds
        osc = (len(hist) >= 20 and len(set(hist[-20::2])) == 1
               and len(set(hist[-19::2])) == 1 and hist[-1] != hist[-2])
        frag_rows.append(dict(fr, frontier_tail=tail, oscillating=bool(osc),
                              rounds_logged=len(hist)))
    return {
        'scenario': scenario, 'seed': seed, 'severance_tick': severance_tick,
        'ticks': len(delivered),
        'delivered_post_sum': float(sum(post)),
        'first_delivery_tick': first_delivery,
        'final_phase': (phases[-1] if phases else None),
        'phase_counts': {p: phases.count(p) for p in sorted(set(phases))},
        'first_discovery_tick': first_disc,
        'first_operational_tick': first_oper,
        'programmed_final': (log[-1]['programmed'] if log else None),
        'fragments_tick': pr.get('fragments_tick'),
        'fragments': frag_rows,
        # absent from pickles written before D12-FIXES (one table per pass)
        'fragments_first_tick': pr.get('fragments_first_tick'),
        'fragments_first': pr.get('fragments_first'),
    }


def report(pickle_path, out_dir=None):
    import pickle
    with open(pickle_path, 'rb') as f:
        Pk = pickle.load(f)
    prov = Pk.get('provenance') or {}
    sev = int(prov.get('severance_tick', 200))
    rows = []
    for scen_key, scenario, ev in (('recovery_multi', 'recovery', sev),
                                   ('rescue_multi', 'rescue_ops', 0)):
        for seed, run in sorted(((Pk.get(scen_key) or {}).get('sdn') or {})
                                .items()):
            if not run:
                continue
            s = summarise_pass(run, scenario, seed, ev)
            if s is not None:
                rows.append(s)
    lines = ["WP12(b) instrumented SDN re-run: %s" % pickle_path,
             "tree %s  digest %s  hashseed %s  ticks %s  severance %s"
             % (prov.get('tree_head'), str(prov.get('tree_sha256'))[:12],
                prov.get('pythonhashseed'), prov.get('total_ticks'), sev), ""]
    for s in rows:
        lines.append("seed %d %-10s: delivered post-event %.0f, first delivery "
                     "%s, discovery from t=%s, operational from t=%s, final "
                     "phase %s, programmed flows %s, phases %s"
                     % (s['seed'], s['scenario'], s['delivered_post_sum'],
                        s['first_delivery_tick'], s['first_discovery_tick'],
                        s['first_operational_tick'], s['final_phase'],
                        s['programmed_final'], s['phase_counts']))
        for fr in s['fragments']:
            # the clause is suppressed when the pickle has no cumulative-BFS
            # number AND when it has one whose value is None (a fragment
            # with no elected leader was never walked) -- `'bfs_rounds' not
            # in fr` alone left "would need None" in the report, the
            # placeholder this section was asked to drop
            ecc = ('' if fr.get('bfs_rounds') is None else
                   " (cumulative BFS would need %s)" % fr['bfs_rounds'])
            if fr.get('leader') is None:
                # no election, so no discovery walk and nothing predicted
                rounds_txt = 'no election'
            else:
                rounds_txt = 'predicted rounds %s%s' % (
                    fr['predicted_rounds'], ecc)
            lines.append("    fid %d: %d infra / %d UE, leader %s, election "
                         "done t=%s, %s, %s, intra flows "
                         "%d, frontier tail %s%s"
                         % (fr['fid'], fr['n_infra'], fr['n_ue'], fr['leader'],
                            fr['election_done_tick'],
                            'BIPARTITE' if fr['bipartite'] else 'odd cycle',
                            rounds_txt,
                            fr['flows_intra'], fr['frontier_tail'],
                            '  <- oscillating' if fr['oscillating'] else ''))
            if 'infra_comp_sizes' in fr:
                # the would-be leaders themselves are printed when the
                # pickle carries them (handoff 7.2 acceptance item 1 asks
                # for leaders, not only their eccentricities); pickles
                # written before they were recorded print as before
                who = ('' if 'infra_comp_leaders' not in fr else
                       " %s" % (fr['infra_comp_leaders'],))
                lines.append("        infra-only decomposition: components "
                             "%s, bipartite %s, ecc of their would-be "
                             "leaders%s %s, UEs anchored in >1 component %d, "
                             "flows with an infrastructure path %d / "
                             "without %d / unanchored %d"
                             % (fr['infra_comp_sizes'],
                                ''.join('T' if b else 'F'
                                        for b in fr['infra_comp_bipartite']),
                                who, fr['infra_comp_ecc'],
                                fr['n_ues_multi_anchored'],
                                fr['flows_with_infra_path'],
                                fr['flows_cross'], fr['flows_unanchored']))
    first = [s for s in rows if s.get('fragments_first')]
    if first:
        lines.append('')
        lines.append("first fragment table of each pass (the post-severance "
                     "decomposition; the lines above show the last one):")
        for s in first:
            lines.append("seed %d %-10s at t=%s:" % (s['seed'], s['scenario'],
                                                  s['fragments_first_tick']))
            for fr in s['fragments_first']:
                lines.append("    fid %d: %d infra / %d UE, leader %s, intra "
                             "flows %d; infra-only components %s, leaders %s, "
                             "UEs anchored in >1 component %d, flows with an "
                             "infrastructure path %d / without %d"
                             % (fr['fid'], fr['n_infra'], fr['n_ue'],
                                fr['leader'], fr['flows_intra'],
                                fr.get('infra_comp_sizes'),
                                fr.get('infra_comp_leaders'),
                                fr.get('n_ues_multi_anchored', 0),
                                fr.get('flows_with_infra_path', 0),
                                fr.get('flows_cross', 0)))
    text = '\n'.join(lines)
    print(text)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(pickle_path))[0]
        with open(os.path.join(out_dir, base + '_report.json'), 'w') as f:
            json.dump({'pickle': pickle_path, 'provenance': {
                k: prov.get(k) for k in ('tree_head', 'tree_sha256',
                                         'pythonhashseed', 'total_ticks',
                                         'severance_tick', 'seeds')},
                'passes': rows}, f, indent=1, default=str)
        with open(os.path.join(out_dir, base + '_report.txt'), 'w') as f:
            f.write(text + '\n')
        print("\nwritten: %s/%s_report.{json,txt}" % (out_dir, base))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    st = sub.add_parser('static')
    st.add_argument('--seeds', default='80-99')
    st.add_argument('--scenarios', default='recovery,rescue_ops')
    st.add_argument('--out-name', default='sdn_static',
                    help='basename under output/wp12/ (default sdn_static)')
    st.add_argument('--fragment-model', choices=FRAGMENT_MODELS,
                    default='ue-stitched',
                    help="controller fragment model: 'ue-stitched' = "
                         "_compute_fragments as it is (default), "
                         "'infra-only' = the D12.2 leaf model")
    st.add_argument('--iteration', choices=ITERATIONS, default='fixed-point',
                    help="discovery termination: 'fixed-point' = the "
                         "test as it is (default), 'bfs' = the cumulative "
                         "iteration requested in D12.2")
    r = sub.add_parser('run')
    r.add_argument('--seeds', default='87,98')
    r.add_argument('--ticks', type=int, default=600)
    r.add_argument('--out', default='wp12_sdn_probe.pkl')
    p = sub.add_parser('report')
    p.add_argument('pickle')
    p.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)
    if args.cmd == 'static':
        cmd_static(args)
    elif args.cmd == 'run':
        cmd_run(args)
    else:
        report(args.pickle, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
