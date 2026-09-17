# -*- coding: utf-8 -*-
"""XL-DET contract tests.

XL-DET only earns its place in the paper if three things are true, and none of
them is obvious from reading the module:

  1. its pinned PRB point really is a fixed point of the engine's projections
     and really does admit every traffic class (verify_prb_point);
  2. its MCS rule really does beat the auto-CQI link adaptation every other arm
     gets for free -- otherwise it is not a competent baseline;
  3. it really is blind to the masked global columns -- otherwise it is a
     baseline with an unfair advantage, and beating it would prove nothing.

(3) is the one a code review cannot settle by eye, so it is tested by
CORRUPTION: every non-local field is set to NaN and the controller must return
byte-identical actions. A controller that reads any of them cannot pass.

Run:  python test_xldet.py
"""
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


import sixg_sim.heuristic_controller as XL
from sixg_sim.agent import NONLOCAL_OBS_COLUMNS
from sixg_sim.phy_mac_state import (MCS_OFFSET_CENTER, MCS_SPECTRAL_EFFICIENCY,
                                    MCSLevel, PHYMACState, auto_cqi_mcs_idx)

print("1. the PRB point holds against the engine's own constants")
try:
    prb, modes = XL.verify_prb_point()
    check(True, "verify_prb_point() passes")
    print("     post-projection point: %.6f / %.6f / %.6f" % tuple(prb))
    print("     admission: " + ", ".join("%s=%s" % (k.name, v)
                                         for k, v in modes.items()))
except AssertionError as e:
    check(False, "verify_prb_point(): %s" % e)

_acc = XL.PRB_EMERGENCY + XL.PRB_GENERAL
print("     access split: emergency %.4f, general %.4f"
      % (XL.PRB_EMERGENCY / _acc, XL.PRB_GENERAL / _acc))
check(abs(XL.PRB_EMERGENCY + XL.PRB_RELAY + XL.PRB_GENERAL - 1.0) < 1e-6,
      "the PRB point lies on the simplex")

print()
print("2. the MCS rule beats auto-CQI on delivered spectral efficiency")


def se_of(sinr, head_idx):
    """Effective SE the engine would deliver for this head at this SINR."""
    final = max(0, min(len(MCSLevel) - 1,
                       auto_cqi_mcs_idx(sinr) + (head_idx - MCS_OFFSET_CENTER)))
    ps = PHYMACState(node_id="t")
    ps.sinr_average = sinr
    ps.mcs_general = list(MCSLevel)[final]
    return ps.spectral_efficiency(for_emergency=False)


sinrs = [round(-6.0 + 0.5 * i, 1) for i in range(0, 81)]      # -6 .. +34 dB
wins = ties = losses = 0
worst = None
for s in sinrs:
    a = se_of(s, MCS_OFFSET_CENTER)          # follow auto-CQI
    x = se_of(s, XL.mcs_head(s))             # XL-DET's choice
    if x > a + 1e-9:
        wins += 1
    elif x < a - 1e-9:
        losses += 1
        if worst is None or (x - a) < worst[1]:
            worst = (s, x - a)
    else:
        ties += 1
print("     over %d SINR points: %d better, %d equal, %d worse"
      % (len(sinrs), wins, ties, losses))
if worst:
    print("     worst regression: %.1f dB, %.3f b/s/Hz" % worst)
check(losses == 0, "XL-DET's MCS is never worse than auto-CQI")
check(wins > 0, "XL-DET's MCS is strictly better somewhere")

print()
print("3. XL-DET is blind to the masked global columns")

import run_timeline_comparison as R
from sixg_sim.simulation import Simulator

_t = R.build_comparison_topology(59)
topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 59, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig())
sim.island_mode = True
sim.control_plane.set_island_mode(True)
sim.heuristic_action_policy = True

# advance a few ticks so observations are populated and postcards have flowed
for tick in range(1, 12):
    sim.current_tick = tick
    sim._forward_traffic(sim._generate_traffic())
    obs_all = sim._build_agent_observations()
    sim._execute_agent_actions(obs_all)

NONLOCAL_NAMES = set(NONLOCAL_OBS_COLUMNS.values())
CONN_NONLOCAL = [n for n in NONLOCAL_NAMES if not n.startswith('coordinator_')]

probe_ids = list(obs_all.keys())[:12]
clean, dirty = {}, {}
for aid in probe_ids:
    o = obs_all[aid]
    ag = sim.agents[aid]

    if hasattr(ag, '_xldet'):
        del ag._xldet
    clean[aid] = XL.decide(o, ag)

    o2 = copy.deepcopy(o)
    for name in CONN_NONLOCAL:
        if hasattr(o2.connectivity, name):
            setattr(o2.connectivity, name, float('nan'))
    o2.global_policy = [float('nan')] * 5
    if hasattr(ag, '_xldet'):
        del ag._xldet
    dirty[aid] = XL.decide(o2, ag)

corrupted = len(CONN_NONLOCAL) + 5
same = sum(1 for a in probe_ids if clean[a] == dirty[a])
print("     corrupted %d non-local fields; identical decisions on %d/%d agents"
      % (corrupted, same, len(probe_ids)))
check(same == len(probe_ids),
      "no masked column reaches XL-DET (NaN corruption changes nothing)")

print()
print("4. XL-DET is deterministic and actuates what it claims")
acts = [ag.last_action for ag in sim.agents.values()
        if getattr(ag, 'last_action', None) is not None]
check(len(acts) > 0, "the engine produced actions under XL-DET")

relay_frac = set(round(float(a.prb_relay_frac), 6) for a in acts)
print("     distinct relay PRB shares across the fleet: %s"
      % sorted(relay_frac))
check(relay_frac == {round(XL.PRB_RELAY, 6)},
      "every agent got the pinned relay share, unmoved by the projections")

modes = sorted(set(int(a.relay_mode_idx) for a in acts))
print("     relay modes in use: %s (LOCAL_REROUTE=%d, CAPACITY_BOOST=%d)"
      % (modes, XL.RELAY_LOCAL_REROUTE, XL.RELAY_CAPACITY_BOOST))
check(set(modes) <= {XL.RELAY_LOCAL_REROUTE, XL.RELAY_CAPACITY_BOOST},
      "relay head only ever selects a bridging mode")

boost = sum(1 for a in acts if int(a.relay_mode_idx) == XL.RELAY_CAPACITY_BOOST)
steerable = sum(1 for aid in obs_all
                if float(getattr(obs_all[aid].connectivity,
                                 'relay_capable', 0.0)) >= 0.5)
print("     CAPACITY_BOOST at %d agents; steerable sites: %d" % (boost, steerable))
check(boost == steerable,
      "CAPACITY_BOOST is selected at exactly the steerable sites")

# determinism: same observation, same controller state -> same action
rep = {}
for aid in probe_ids:
    ag = sim.agents[aid]
    if hasattr(ag, '_xldet'):
        del ag._xldet
    rep[aid] = XL.decide(obs_all[aid], ag)
check(all(rep[a] == clean[a] for a in probe_ids),
      "repeating a decision from the same state reproduces it exactly")

print()
print("5. the arm cannot be silently combined with another policy override")
sim.random_action_policy = True
try:
    sim._execute_agent_actions(obs_all)
    check(False, "combining XL-DET with random_action_policy must raise")
except RuntimeError:
    check(True, "combining XL-DET with random_action_policy raises")
finally:
    sim.random_action_policy = False

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
