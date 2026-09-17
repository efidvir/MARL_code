# -*- coding: utf-8 -*-
"""Freeze the corrected disaster benchmark.

The severance generator, the runtime engine and the attainable-optimum solver
now share one definition of a steerable site (sixg_sim.simulation.
site_can_steer). Because the generator's "bridgeable" constraint previously
used an inflated definition, the damage instances themselves change -- so the
benchmark is regenerated and frozen BEFORE any arm is trained or evaluated on
it, with a manifest recorded so nobody can later be accused of regenerating
scenarios because a particular result was inconvenient.

The attainable optimum is computed by driving the scenario to its severance and
calling the solver on the LIVE post-event graph, rather than scraping a log
line, so that both scenario kinds are measured the same way.

Usage:  python build_benchmark_manifest.py [--seeds 50-59] [--out FILE]
"""
import argparse, io, json, os, sys, datetime, hashlib, contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def measure(R, Simulator, site_can_steer, nx, seed, kind):
    t = R.build_comparison_topology(seed)
    t = t[0] if isinstance(t, tuple) else t
    scen = R.build_comparison_scenario(kind, seed, topology=t)
    sim = Simulator(t, scen, R.SimulationConfig())

    # drive to just past the last severance-type event
    sev = max([getattr(e, 'tick', 0) for e in getattr(scen, 'events', [])
               if 'sever' in str(getattr(e, 'event_type', ''))] or [0])
    for tick in range(sev + 2):
        sim.current_tick = tick
        sim._process_events(tick)
    sim.island_mode = sim._detect_island_mode()

    sim.topology.invalidate_infrastructure_cache()
    g_all = sim.topology._build_infrastructure_graph()
    surv = [n for n in g_all.nodes
            if getattr(t.nodes.get(n), 'is_survivor', False)]
    # The downed core nodes sit in this graph as isolated vertices. They are
    # not fragments anyone can bridge, so both the component count and the
    # solver must see the SURVIVOR subgraph only.
    g = g_all.subgraph(surv).copy()
    raw = nx.number_connected_components(g)

    res = R.tg_feasible_component_optimum(sim.topology, [], graph=g, surv=surv)
    raw_solver, optimum, n_sites = (list(res) + [None, None, None])[:3]

    steerable = sum(1 for n in t.nodes.values() if site_can_steer(n))
    return {
        'topology_seed': seed,
        'scenario_kind': kind,
        'severance_tick': sev,
        'raw_components': int(raw),
        'attainable_optimum': int(optimum) if optimum is not None else None,
        'solver_candidate_sites': int(n_sites) if n_sites is not None else None,
        'actionable_steerable_sites': steerable,
        'fully_reunifiable': (optimum == 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='50-59')
    ap.add_argument('--kinds', default='recovery,rescue_ops')
    ap.add_argument('--out', default='benchmark_manifest.json')
    a = ap.parse_args()

    lo, _, hi = a.seeds.partition('-')
    seeds = list(range(int(lo), int(hi or lo) + 1))
    kinds = [k.strip() for k in a.kinds.split(',') if k.strip()]

    import networkx as nx
    import run_timeline_comparison as R
    from sixg_sim.simulation import Simulator, site_can_steer, STEER_MODEL

    print("STEER_MODEL =", STEER_MODEL)
    if STEER_MODEL != 'actionable':
        print("REFUSING: freeze the benchmark only under the corrected "
              "definition (unset MARL_STEER_MODEL).")
        return 2

    entries = []
    for seed in seeds:
        for kind in kinds:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                e = measure(R, Simulator, site_can_steer, nx, seed, kind)
            entries.append(e)
            print("  seed %-3d %-11s raw=%-2d optimum=%-2s sites=%-3s %s"
                  % (seed, kind, e['raw_components'],
                     e['attainable_optimum'], e['solver_candidate_sites'],
                     '' if e['fully_reunifiable'] else 'NOT-REUNIFIABLE'))

    payload = {
        'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'steer_model': STEER_MODEL,
        'note': ('Frozen under the unified actionable-bridge definition '
                 '(sixg_sim.simulation.site_can_steer). Any change to that '
                 'predicate invalidates this manifest. attainable_optimum is '
                 'the reference every reunification claim must be made '
                 'against; the literal 1 is correct only where '
                 'fully_reunifiable is true.'),
        'instances': entries,
    }
    blob = json.dumps(payload, indent=2, sort_keys=True)
    payload['manifest_sha256'] = hashlib.sha256(blob.encode()).hexdigest()[:32]
    io.open(a.out, 'w', encoding='utf-8').write(
        json.dumps(payload, indent=2, sort_keys=True))

    print()
    for kind in kinds:
        sel = [e for e in entries if e['scenario_kind'] == kind]
        ok = sum(1 for e in sel if e['fully_reunifiable'])
        mean_opt = sum(e['attainable_optimum'] for e in sel) / max(1, len(sel))
        print("  %-11s n=%-3d fully reunifiable %d/%d   mean optimum %.2f"
              % (kind, len(sel), ok, len(sel), mean_opt))
    print()
    print("manifest sha256 : %s" % payload['manifest_sha256'])
    print("written to      : %s" % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
