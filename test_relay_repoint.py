# -*- coding: utf-8 -*-
"""Safety contract for fragment-aware re-pointing (MARL_RELAY_REPOINT).

  1. OFF BY DEFAULT: with the variable unset the attribute is False and no
     re-point is ever counted, even across a full severance with bridges.
  2. NEVER ABANDON A BRIDGE: with the flag on, a site whose active link is a
     tree-edge (sole-path) bridge must be refused by the helper -- this is
     the property that keeps reunification monotone.
  3. GUARDS: a non-steerable site is refused; a site inside the per-site
     cooldown is refused.
  4. LIVENESS (weak): across the same XL-DET-driven severance the helper is
     reached and counted only when a non-bridge link plus a feasible
     cross-fragment target coexist (count >= 0 is reported; the decisive
     functional check is the seed-55 probe on the server).

Run:  MARL_STEER_MODEL=actionable MARL_RELAY_REPOINT=fragment_aware python test_relay_repoint.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


if os.environ.get('MARL_RELAY_REPOINT') != 'fragment_aware':
    print("re-run with MARL_RELAY_REPOINT=fragment_aware MARL_STEER_MODEL=actionable"); sys.exit(2)

print("1. off by default (clean subprocess, through a real severance)")
code = r'''
import os, sys; sys.path.insert(0, %r)
os.environ.setdefault('MARL_STEER_MODEL','actionable')
import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
t=R.build_comparison_topology(54); topo=t[0] if isinstance(t,tuple) else t
scen=R.build_comparison_scenario('recovery',54,topology=topo)
sim=Simulator(topo,scen,R.SimulationConfig()); sim.island_mode=True; sim.control_plane.set_island_mode(True)
sim.heuristic_action_policy=True
for a in sim.agents.values(): a.is_training=False
for tick in range(1,261):
    sim.current_tick=tick; sim._process_events(tick); sim._forward_traffic(sim._generate_traffic())
    sim._execute_agent_actions(sim._build_agent_observations())
print("DEFAULT repoint_flag=%%s repoints=%%d bridges=%%d" %% (sim._relay_repoint, sim._diag_repoints, len(sim.distinct_bridge_link_ids())))
''' % HERE
env = {k: v for k, v in os.environ.items() if k != 'MARL_RELAY_REPOINT'}
r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
line = [l for l in r.stdout.splitlines() if l.startswith('DEFAULT')]
print("    ", line[-1] if line else r.stderr[-600:])
check(bool(line) and 'repoint_flag=False' in line[-1] and 'repoints=0' in line[-1],
      "unset env -> flag False and zero re-points across a bridged severance")

print()
print("2./3./4. flag on: guards and the never-abandon-a-bridge property")
import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
from sixg_sim.topology import LinkType
_t = R.build_comparison_topology(54); topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 54, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig()); sim.island_mode = True
sim.control_plane.set_island_mode(True)
sim.heuristic_action_policy = True                 # XL-DET forms bridges reliably
for a in sim.agents.values():
    a.is_training = False
for tick in range(1, 261):
    sim.current_tick = tick; sim._process_events(tick)
    sim._forward_traffic(sim._generate_traffic())
    sim._execute_agent_actions(sim._build_agent_observations())
check(sim._relay_repoint is True, "flag parsed (fragment_aware)")
bridges = sim.distinct_bridge_link_ids()
print("     bridges after severance: %d ; re-points counted so far: %d" % (len(bridges), sim._diag_repoints))
check(len(bridges) > 0, "bridges exist to test the protection against")

# a site holding a BRIDGE must be refused
holders = [n for n, ps in sim.phy_mac_states.items()
           if ps.relay_link_active and any(n in sim.topology.links[b].endpoints for b in bridges if b in sim.topology.links)]
refused = all(sim._maybe_repoint_to_fragment(n, sim.phy_mac_states[n]) is False for n in holders)
check(holders and refused, "every bridge-holding site is refused (%d checked)" % len(holders))
still = sim.distinct_bridge_link_ids()
check(still == bridges, "bridge set unchanged after the refused calls")

# cooldown guard: mark a site as just re-pointed -> refused regardless
some = next(iter(sim.phy_mac_states))
sim._last_repoint_tick[some] = sim.current_tick
check(sim._maybe_repoint_to_fragment(some, sim.phy_mac_states[some]) is False, "cooldown refuses")
del sim._last_repoint_tick[some]

# non-steerable guard: temporarily strip capability
node = sim.topology.nodes[some]; saved = getattr(node, 'has_multihaul', False)
node.has_multihaul = False
check(sim._maybe_repoint_to_fragment(some, sim.phy_mac_states[some]) is False, "non-steerable site refuses")
node.has_multihaul = saved

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
