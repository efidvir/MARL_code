# -*- coding: utf-8 -*-
"""Why do MARL's bridges die?  Per-teardown forensic replay.

Hysteresis on the relay head latched the relay MODE (144 -> 0 flips) yet the
bridge formation/teardown counts did not move (93/85 -> 93/84 on the s1234
checkpoint).  So the teardowns are not caused by relay-mode switching.  This
script replays one evaluation pass with the REAL harness (run_physics_pass,
same arm plumbing, same freeze logic) and, every tick, diffs the set of live
TRANSPORT_RELAY links.  For each link that disappears it prints both
endpoints' state on the tick BEFORE the drop and the action they had just
issued -- relay mode, tx power (dBm), PRB relay share, SINR, link capacity and
utilisation -- and tags whether the drop happened INSIDE the agents' action
step (agent-caused) or BETWEEN steps (engine rule: link budget, make-before-
break, repoint outage, ...).  Formations are printed too.

Usage:  MARL_STEER_MODEL=actionable python3 diag_bridge_teardown.py <seed> <ck.pt> [hyst]
"""
import os
import sys

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 54
ck = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else 'auth_ck_s1234/policy_final.pt')
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')
if len(sys.argv) > 3:
    os.environ['MARL_RELAY_HYSTERESIS'] = sys.argv[3]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
from sixg_sim.topology import LinkType


def snap(sim):
    d = {}
    for lid, l in sim.topology.links.items():
        if (getattr(l, 'link_type', None) is LinkType.TRANSPORT_RELAY
                and getattr(l, 'is_up', True)):
            a, b = l.endpoints
            d[lid] = dict(a=a, b=b, cap=round(float(getattr(l, 'capacity', 0) or 0), 1),
                          util=round(float(getattr(l, 'current_utilization', 0) or 0), 1))
    return d


def ep(sim, n):
    ps = sim.phy_mac_states.get(n)
    ag = sim.agents.get(n)
    act = getattr(ag, 'last_action', None) if ag is not None else None
    if ps is None:
        return {'node': n, 'phy': None}
    return dict(node=n,
                mode=str(getattr(ps, 'relay_mode', '')).split('.')[-1],
                tx=round(float(getattr(ps, 'tx_power_dbm', 0)), 1),
                prb_r=round(float(getattr(ps, 'prb_relay_fraction', 0)), 3),
                sinr=round(float(getattr(ps, 'sinr_average', 0)), 1),
                active=bool(getattr(ps, 'relay_link_active', False)),
                act_relay=(int(act.relay_mode_idx) if act is not None else None),
                act_tx=(int(act.tx_power_step) if act is not None else None),
                act_prb_r=(round(float(act.prb_relay_frac), 3) if act is not None else None))


_orig = Simulator._execute_agent_actions
_prev = {'links': None, 'eps': {}}
_stats = {'drop_in_step': 0, 'drop_between': 0, 'form': 0, 'first_tick': None}


def _report_drop(sim, lid, v, pre_eps, where):
    t = sim.current_tick
    print(f"[DROP {where} t={t}] {lid}  {v['a']}<->{v['b']}  cap={v['cap']} util={v['util']}")
    print(f"    pre  A {pre_eps.get(v['a'])}")
    print(f"    pre  B {pre_eps.get(v['b'])}")
    print(f"    now  A {ep(sim, v['a'])}")
    print(f"    now  B {ep(sim, v['b'])}", flush=True)


def patched(self, observations):
    before = snap(self)
    before_eps = {n: ep(self, n) for v in before.values() for n in (v['a'], v['b'])}
    # drops that happened BETWEEN the previous action step and this one
    if _prev['links'] is not None:
        for lid, v in _prev['links'].items():
            if lid not in before:
                _stats['drop_between'] += 1
                _report_drop(self, lid, v, _prev['eps'], 'BETWEEN-STEPS(engine)')
    _orig(self, observations)
    after = snap(self)
    for lid, v in before.items():
        if lid not in after:
            _stats['drop_in_step'] += 1
            _report_drop(self, lid, v, before_eps, 'IN-STEP(agent)')
    for lid, v in after.items():
        if lid not in before:
            _stats['form'] += 1
            print(f"[FORM t={self.current_tick}] {lid}  {v['a']}<->{v['b']}  cap={v['cap']}", flush=True)
    _prev['links'] = after
    _prev['eps'] = {n: ep(self, n) for v in after.values() for n in (v['a'], v['b'])}


Simulator._execute_agent_actions = patched
print(f"=== forensic replay: seed={seed} arm=marl_freeze ck={os.path.basename(os.path.dirname(ck))} "
      f"hyst={os.environ.get('MARL_RELAY_HYSTERESIS','0')} ===", flush=True)
r = R.run_physics_pass('marl_freeze', seed, ck, 'recovery')
fr = [v for v in r['fragments'][-500:] if isinstance(v, (int, float))]
print("=== SUMMARY ===")
print("formations:", _stats['form'], " drops inside action step:", _stats['drop_in_step'],
      " drops between steps (engine):", _stats['drop_between'])
print("steady components (last 500):", round(sum(fr) / len(fr), 2))
