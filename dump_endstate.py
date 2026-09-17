# -*- coding: utf-8 -*-
"""Dump the end state of one (arm, seed) recovery pass for the process figure.

Runs the real harness (run_physics_pass) under whatever engine env is set,
and at PROBE_TICK records everything the figure needs and nothing it can
re-derive: the transport-relay links that exist (the bridges), the
relay-inclusive component label of every infrastructure node, and the raw
(relay-free) label.  Node geometry, link cuts and pre-event topology are pure
functions of the seed (build_comparison_topology / _build_fragmenting_
severance) and are re-derived locally by plot_process_figure.py.

Usage (server):
  MARL_STEER_MODEL=actionable MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile \
  MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 MARL_REWARD_PROFILE=mission \
  MARL_POSTCARD_FRAGMENTS=1 python3 dump_endstate.py <seed> <ckpt> <arm> <out.json>
"""
import json
import os
import sys

seed = int(sys.argv[1]); ck = os.path.abspath(sys.argv[2]); arm = sys.argv[3]; out = sys.argv[4]
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
from sixg_sim.topology import LinkType, NodeType

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
r = R.run_physics_pass(arm, seed, ck, 'recovery')
fr = [v for v in r['fragments'][-500:] if isinstance(v, (int, float))]
cn = [v for v in r['conn'][-500:] if isinstance(v, (int, float))]
summary = {'seed': seed, 'arm': arm, 'steady_components': sum(fr) / len(fr), 'steady_achievability': sum(cn) / len(cn),
           'snapshots': SNAPS}
json.dump(summary, open(out, 'w'))
print('steady comps %.2f achiev %.1f -> %s' % (summary['steady_components'], summary['steady_achievability'], out))
