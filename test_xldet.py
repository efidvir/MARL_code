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
import sixg_sim.agent as _AG
from sixg_sim.agent import CRITIC_GLOBAL_FIELDS
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

# WP6: the critic-only fields are CRITIC_GLOBAL_FIELDS, whose extractors
# all read ConnectivityState.  The 'g_' prefix is the schema name; the
# dataclass attribute is what this probe has to poison.
_G2ATTR = {
    'g_reachable_ue_fraction':     'reachable_ue_fraction',
    'g_isolated_ue_count':         'isolated_ue_count_norm',
    'g_active_relay_paths':        'active_relay_paths_norm',
    'g_ue_to_ue_routed':           'ue_to_ue_routed_norm',
    'g_intra_island_reachability': 'intra_island_reachability',
    'g_core_distance':             'core_distance_norm',
    'g_bridge_node_flag':          'bridge_node_flag',
    'g_island_size':               'island_size_norm',
    'g_pending_iops':              'pending_iops_norm',
    'g_island_member_count':       'island_member_count_norm',
    'g_island_ue_load_balance':    'island_ue_load_balance',
    'g_xn_mesh_density':           'xn_mesh_density',
    'g_local_epc_health':          'local_epc_health',
    'g_nenb_count':                'nenb_count_norm',
    'g_multi_island_bridge':       'multi_island_bridge',
}
# WP13 (D13.3) removed four of these columns from the schema as constant
# (g_reachable_ue_fraction, g_isolated_ue_count, g_nenb_count,
# g_multi_island_bridge).  Their ConnectivityState attributes still exist and
# XL-DET may read them no more than the others, so the probe poisons every
# attribute of the map -- the schema's globals AND the removed ones -- and
# fails if a schema global has no entry here.
_unmapped = [n for n, _fn in CRITIC_GLOBAL_FIELDS if n not in _G2ATTR]
if _unmapped:
    raise SystemExit('test_xldet: critic-global field(s) with no attribute '
                     'in _G2ATTR: %s' % ', '.join(_unmapped))
CONN_NONLOCAL = sorted(set(_G2ATTR.values()))
assert len(CONN_NONLOCAL) == 15, len(CONN_NONLOCAL)

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

# RERUN_SPEC WP7: XL.RELAY_* are HEAD indices into agent.RELAY_HEAD_MODES,
# so they are compared against relay_head_idx.  relay_mode_idx stays the
# RelayMode enum index and is what the engine actuates.
modes = sorted(set(int(a.relay_head_idx) for a in acts))
print("     relay head modes in use: %s (LOCAL_REROUTE=%d, CAPACITY_BOOST=%d)"
      % (modes, XL.RELAY_LOCAL_REROUTE, XL.RELAY_CAPACITY_BOOST))
check(set(modes) <= {XL.RELAY_LOCAL_REROUTE, XL.RELAY_CAPACITY_BOOST},
      "relay head only ever selects a bridging mode")
check(all(int(a.relay_mode_idx) == _AG.relay_enum_index(int(a.relay_head_idx))
          for a in acts),
      "the executed RelayMode is the head's mode under the WP7 mapping")

boost = sum(1 for a in acts if int(a.relay_head_idx) == XL.RELAY_CAPACITY_BOOST)
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
