# -*- coding: utf-8 -*-
"""Process-map pass for Fig. 6 panel (c) of the direct paper: one recovery
pass of the frozen executor of replicate 1234 on development seed 58.

A copy of dump_endstate.py (same snapshot function, same patched
_execute_agent_actions, same probe ticks 150 / 260 / PROBE_TICK=3400, same
summary JSON) with two additions, both outside the engine:
  * the harness tree is imported from R4_TREE (default ~/r4_work/tree, whose
    engine is bit-identical to the frozen R3-lite tree), not from this
    file's directory;
  * the full run returned by run_physics_pass is also written as a harness
    pickle {'recovery_multi': {arm: {seed: run}}, 'rescue_multi': {},
    'provenance': ...} so that tools/r4_identity_check.py can compare it with
    the recorded development pass bit for bit.

Usage (server; the R3-lite evaluation engine environment must be set):
  python3 dump_endstate_v11.py <seed> <ckpt> <arm> <out.json> <out_run.pkl>
"""
import json
import os
import pickle
import sys

seed = int(sys.argv[1]); ck = os.path.abspath(sys.argv[2]); arm = sys.argv[3]; out = sys.argv[4]
out_pkl = sys.argv[5]
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')
TREE = os.path.expanduser(os.environ.get('R4_TREE', '~/r4_work/tree'))
sys.path.insert(0, TREE)

import run_timeline_comparison as R                    # noqa: E402
from sixg_sim import runinfo                           # noqa: E402
from sixg_sim.simulation import Simulator              # noqa: E402
from sixg_sim.topology import LinkType, NodeType       # noqa: E402,F401

PROBE_TICK = int(os.environ.get('PROBE_TICK', '3400'))
_orig = Simulator._execute_agent_actions
_done = {'x': False}


def snapshot(sim, tag):
    labels = sim._infra_component_labels(True) or {}
    raw = sim._infra_component_labels(False) or {}
    bridges = []
    for lid, l in sim.topology.links.items():
        if getattr(l, 'link_type', None) == LinkType.TRANSPORT_RELAY:
            ep = list(getattr(l, 'endpoints', ()))
            bridges.append({'id': str(lid), 'endpoints': [str(e) for e in ep],
                            'is_up': bool(getattr(l, 'is_up', True)),
                            'capacity_mbps': float(getattr(l, 'capacity_mbps', 0.0) or 0.0)})
    nodes = {}
    for nid, n in sim.topology.nodes.items():
        nodes[str(nid)] = {'x': float(n.x_pos), 'y': float(n.y_pos),
                           'type': str(getattr(n, 'node_type', '')).split('.')[-1],
                           'survivor': bool(getattr(n, 'is_survivor', True)),
                           'multihaul': bool(getattr(n, 'has_multihaul', False)),
                           'label': (None if labels.get(nid) is None else str(labels.get(nid))),
                           'raw_label': (None if raw.get(nid) is None else str(raw.get(nid)))}
    links = []
    for lid, l in sim.topology.links.items():
        lt = getattr(l, 'link_type', None)
        links.append({'id': str(lid), 'endpoints': [str(e) for e in getattr(l, 'endpoints', ())],
                      'type': (str(lt).split('.')[-1] if lt is not None else ''),
                      'is_up': bool(getattr(l, 'is_up', True))})
    rec = {'tag': tag, 'tick': sim.current_tick, 'seed': seed, 'arm': arm,
           'components': sim.count_infra_components(True), 'components_raw': sim.count_infra_components(False),
           'distinct_bridges': len(sim.distinct_bridge_link_ids()),
           'bridge_link_ids': [str(b) for b in sim.distinct_bridge_link_ids()],
           'relay_links': bridges, 'nodes': nodes, 'links': links}
    print('%s tick=%d comps=%d raw=%d bridges=%d relay_links=%d' % (
        tag, sim.current_tick, rec['components'], rec['components_raw'], rec['distinct_bridges'], len(bridges)), flush=True)
    return rec


SNAPS = []


def patched(self, observations):
    _orig(self, observations)
    if self.current_tick == 150 and not any(s['tag'] == 'pre' for s in SNAPS):
        SNAPS.append(snapshot(self, 'pre'))
    if self.current_tick == 260 and not any(s['tag'] == 'cut' for s in SNAPS):
        SNAPS.append(snapshot(self, 'cut'))
    if self.current_tick >= PROBE_TICK and not _done['x']:
        _done['x'] = True
        SNAPS.append(snapshot(self, 'end'))


Simulator._execute_agent_actions = patched
runinfo.check_hashseed('dump_endstate')
_provenance = runinfo.provenance(root=TREE, extra={
    'driver': 'dump_endstate_v11 (copy of dump_endstate.py + run pickle)', 'seed': seed, 'arm': arm,
    'checkpoint': ck, 'checkpoint_sha256': runinfo.file_sha256(ck), 'probe_tick': PROBE_TICK})
r = R.run_physics_pass(arm, seed, ck, 'recovery')
fr = [v for v in r['fragments'][-500:] if isinstance(v, (int, float))]
cn = [v for v in r['conn'][-500:] if isinstance(v, (int, float))]
summary = {'seed': seed, 'arm': arm, 'steady_components': sum(fr) / len(fr), 'steady_achievability': sum(cn) / len(cn),
           'snapshots': SNAPS, 'provenance': _provenance}
with open(out, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(summary, f)
with open(out_pkl, 'wb') as f:
    pickle.dump({'recovery_multi': {arm: {seed: r}}, 'rescue_multi': {}, 'seeds': [seed],
                 'provenance': _provenance}, f, protocol=pickle.HIGHEST_PROTOCOL)
print('steady comps %.2f achiev %.1f -> %s, run -> %s' % (
    summary['steady_components'], summary['steady_achievability'], out, out_pkl))
