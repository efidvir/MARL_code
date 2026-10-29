# -*- coding: utf-8 -*-
"""Export the scenario geometry of one Scenario A instance for Fig. 6 (a), (b).

Read-only.  Runs the scenario builders of one engine tree (no simulation
tick is run) and writes a plain JSON that make_result_figs.py draws locally:
  * plot_severance_maps.derive_instance(seed, 'recovery') of that tree
    (positions, links classified up / cut / core, fragments, flows);
  * the actionable steerable sites (MultiHaul flag AND an agent), exactly as
    plot_process_figure.py computes them (Simulator construction, no tick);
  * the attainable optimum of the instance from the tree's benchmark manifest.

Usage: python3 geom_seed58.py <tree> <seed> <out.json>
"""
import contextlib
import io
import json
import os
import sys

TREE = os.path.realpath(os.path.expanduser(sys.argv[1]))
SEED = int(sys.argv[2])
OUT = sys.argv[3]
sys.path.insert(0, TREE)

import run_timeline_comparison as R          # noqa: E402  (cached: the builders below use this module)
import plot_severance_maps as PSM            # noqa: E402
from sixg_sim.simulation import Simulator    # noqa: E402

for mod in (R, PSM, sys.modules['sixg_sim.simulation']):
    f = os.path.realpath(mod.__file__)
    assert f.startswith(TREE + os.sep), (mod.__name__, f, TREE)

inst = PSM.derive_instance(SEED, 'recovery')
if inst['mismatch']:
    raise SystemExit('builder mismatch: %s' % inst['mismatch'])

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    _t = R.build_comparison_topology(SEED)
    topo = _t[0] if isinstance(_t, tuple) else _t
    scen = R.build_comparison_scenario('recovery', SEED, topology=topo)
    sim = Simulator(topo, scen, R.SimulationConfig())
actionable = sorted(n for n in sim.agents if getattr(topo.nodes[n], 'has_multihaul', False))
surv = set(inst['infra_surv'])
manifest = json.load(open(os.path.join(TREE, 'benchmark_manifest.json')))
opt = [i for i in manifest['instances']
       if i['scenario_kind'] == 'recovery' and i['topology_seed'] == SEED]
assert len(opt) == 1, len(opt)

head = None
for p in ('.synced_head',):
    q = os.path.join(TREE, p)
    if os.path.exists(q):
        head = open(q).read().strip()

out = {
    'what': 'scenario geometry of Scenario A (recovery), topology seed %d, from the builders of one tree; '
            'no simulation tick run' % SEED,
    'tree': TREE, 'tree_head': head,
    'seed': SEED, 'kind': 'recovery',
    'severance_tick': inst['severance_tick'],
    'pos': {str(k): [float(v[0]), float(v[1])] for k, v in inst['pos'].items()},
    'ntype': {str(k): str(v) for k, v in inst['ntype'].items()},
    'core': sorted(str(x) for x in inst['core']),
    'infra_surv': [str(x) for x in inst['infra_surv']],
    'ue_ids': [str(x) for x in inst['ue_ids']],
    'edges': [[str(u), str(v), k] for (u, v, k) in inst['edges'] if k != 'uu'],
    'n_uu_edges': sum(1 for e in inst['edges'] if e[2] == 'uu'),
    'frag_of': {str(k): int(v) for k, v in inst['frag_of'].items()},
    'comp_sizes': [int(x) for x in inst['comp_sizes']],
    'n_fragments': int(inst['n_fragments']),
    'n_cut_links': int(inst['n_cut_links']),
    'n_core_links': int(inst['n_core_links']),
    'cross_flows': int(inst['cross_flows']),
    'total_flows': int(inst['total_flows']),
    'cross_share': float(inst['cross_share']),
    'builder_stats': inst['builder_stats'],
    'n_multihaul_flags': int(inst['n_multihaul']),
    'multihaul_actionable_surviving': sorted(str(x) for x in set(actionable) & surv),
    'n_agents': len(sim.agents),
    'attainable_optimum': opt[0]['attainable_optimum'],
    'manifest_instance': opt[0],
}
with open(OUT, 'w', newline='\n') as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print('GEOM OK seed=%d tree=%s head=%s fragments=%d sizes=%s cut=%d core_links=%d flows=%d/%d steer=%d infra=%d opt=%s'
      % (SEED, TREE, head, out['n_fragments'], out['comp_sizes'], out['n_cut_links'], out['n_core_links'],
         out['cross_flows'], out['total_flows'], len(out['multihaul_actionable_surviving']),
         len(out['infra_surv']), out['attainable_optimum']))
