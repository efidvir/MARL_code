# -*- coding: utf-8 -*-
"""End-state bridge-feasibility probe for one (policy, seed).

HYPOTHESIS UNDER TEST.  On the selected policy's misses (e.g. seed 55: 21 of
22 sites in a relay mode, zero bridges) the sites are WILLING but no
cross-fragment peer is within the 60 GHz link budget, because that budget is
evaluated at the agent's FR1 access transmit power (ps.tx_power_dbm), which an
energy-aware policy drives toward the 10 dBm floor.  If the same peers become
feasible at full backhaul power (33 dBm), the failure is a cross-radio
modelling coupling, not a policy failure.

Replays the real harness (run_physics_pass) and, at tick 3400 (deep in the
frozen phase), dumps for every steerable site: relay mode, tx power, active
link, own fragment, and -- for sites without a bridge -- whether ANY node in a
different fragment is a feasible 60 GHz peer at (a) the current power and (b)
33 dBm.

Usage: MARL_STEER_MODEL=actionable [mission env] python3 diag_bridge_feasibility.py <seed> <ckpt> <arm>
"""
import os
import sys

seed = int(sys.argv[1]); ck = os.path.abspath(sys.argv[2]); arm = sys.argv[3] if len(sys.argv) > 3 else 'marl_freeze'
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_timeline_comparison as R
from sixg_sim.simulation import Simulator, BAND_TG
from sixg_sim.topology import NodeType, LinkType
from sixg_sim.phy_mac_state import RelayMode

PROBE_TICK = 3400
_orig = Simulator._execute_agent_actions
_done = {'x': False}


def probe(sim):
    labels = sim._infra_component_labels(True) or {}
    trm = sim.transport_relay_model
    print("=== FEASIBILITY PROBE t=%d seed=%d arm=%s ===" % (sim.current_tick, seed, arm))
    print("components (relay-inclusive): %d   raw: %d"
          % (sim.count_infra_components(True), sim.count_infra_components(False)))
    n_will = n_cb = n_feas_now = n_feas_33 = 0
    for nid, ps in sorted(sim.phy_mac_states.items()):
        node = sim.topology.nodes.get(nid)
        if node is None or not node.is_survivor or not getattr(node, 'has_multihaul', False):
            continue
        mode = str(getattr(ps, 'relay_mode', '')).split('.')[-1]
        mine = labels.get(nid)
        foreign = [n for n, lab in labels.items()
                   if lab != mine and n in sim.topology.nodes
                   and sim.topology.nodes[n].is_survivor
                   and sim.topology.nodes[n].node_type != NodeType.UE]
        def feasible(p):
            best = None
            for peer in foreign:
                try:
                    r = trm.can_form_relay_link(nid, peer, tx_power_dbm=p,
                                                interference_dbm=sim.aggregate_interference_dbm(peer, BAND_TG))
                except TypeError:
                    r = trm.can_form_relay_link(nid, peer, p)
                if r:
                    best = peer if best is None else best
            return best
        in_relay = mode in ('LOCAL_REROUTE', 'CAPACITY_BOOST')
        n_will += int(in_relay); n_cb += int(mode == 'CAPACITY_BOOST')
        # LOCK-IN CHECK: if this site holds a link, where does it point?
        peer_now = getattr(ps, 'relay_peer_node', None)
        if getattr(ps, 'relay_link_active', False) and peer_now:
            same = (labels.get(peer_now) == mine)
            print("      LINK %s -> %s  (%s)" % (nid, peer_now,
                  "INTRA-fragment: locks out bridging" if same else "cross-fragment bridge"))
        f_now = feasible(float(ps.tx_power_dbm)) if foreign else None
        f_33 = feasible(33.0) if foreign else None
        n_feas_now += int(f_now is not None); n_feas_33 += int(f_33 is not None)
        print("  %-10s mode=%-15s tx=%5.1f dBm active=%-5s frag=%-8s foreign_frags=%d  feasible@now=%-10s feasible@33=%s"
              % (nid, mode, float(ps.tx_power_dbm), getattr(ps, 'relay_link_active', False), str(mine),
                 len(set(labels[n] for n in foreign)), str(f_now), str(f_33)))
    print("SUMMARY: steerable sites in relay mode=%d (CAPACITY_BOOST=%d); sites with a feasible cross-fragment peer: at current power=%d, at 33 dBm=%d"
          % (n_will, n_cb, n_feas_now, n_feas_33), flush=True)


def patched(self, observations):
    _orig(self, observations)
    if self.current_tick >= PROBE_TICK and not _done['x']:
        _done['x'] = True
        probe(self)


Simulator._execute_agent_actions = patched
r = R.run_physics_pass(arm, seed, ck, 'recovery')
fr = [v for v in r['fragments'][-500:] if isinstance(v, (int, float))]
print("steady components (last 500):", round(sum(fr) / len(fr), 2))
