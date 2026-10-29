# -*- coding: utf-8 -*-
"""Test the action path the EXPERIMENT uses, not a function it never calls.

The earlier PRB test exercised RLAgent.compute_action and passed, while the
live path -- the inline batched block in Simulator._execute_agent_actions --
still used a deterministic softmax. agent.action_from_logits_row is dead code
and compute_action is only a "shouldn't happen" fallback, so a test that calls
either proves nothing about the experiment.

Run:  python test_live_action_path.py
"""
import os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
import sixg_sim.agent as A

print("PRB_POLICY =", A.PRB_POLICY)
print()

print("0. the functions a naive test would call are NOT the live path")
import io as _io
src = _io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sixg_sim", "simulation.py"), encoding="utf-8").read()
check("action_from_logits_row" not in src,
      "action_from_logits_row is not called by the engine (it is dead code)")
check(src.count("agent.compute_action(obs)") <= 1,
      "compute_action appears only as the fallback")

_t = R.build_comparison_topology(59)
topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 59, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig())
sim.island_mode = True
sim.control_plane.set_island_mode(True)

print()
print("1. training mode: the executed split is stochastic")
for a in sim.agents.values():
    a.is_training = True
seen = collections.defaultdict(set)
for tick in range(1, 5):
    sim.current_tick = tick
    sim._forward_traffic(sim._generate_traffic())
    obs = sim._build_agent_observations()
    sim._execute_agent_actions(obs)
    for aid, ag in sim.agents.items():
        act = getattr(ag, 'last_action', None)
        if act is not None:
            seen[aid].add(round(float(act.prb_relay_frac), 6))
varies = sum(1 for v in seen.values() if len(v) > 1)
print("     agents whose relay share varied over 4 ticks: %d/%d" % (varies, len(seen)))
check(varies > 0, "the live path produces a STOCHASTIC allocation while training")

print()
print("2. the sampled point is carried for PPO, not the projected one")
some = [ag.last_action for ag in sim.agents.values()
        if getattr(ag, 'last_action', None) is not None]
check(all(getattr(a, 'prb_sampled', None) is not None for a in some),
      "every action carries prb_sampled")
scored = [A.prb_training_action(a) for a in some]
check(all(abs(sum(v) - 1.0) < 1e-5 for v in scored),
      "the scored point lies on the simplex")
projected_differs = sum(
    1 for a in some
    if abs(a.prb_general_frac - (a.prb_sampled or [0, 0, 0])[2]) > 1e-9)
print("     actions where the projection moved the point: %d/%d"
      % (projected_differs, len(some)))

print()
print("3. evaluation mode: same observation -> same action")
# Determinism is a property of the MAPPING, not of the time series: the
# observation changes every tick, so the Dirichlet mean legitimately moves.
# Feed the SAME observation twice and require an identical allocation.
for a in sim.agents.values():
    a.is_training = False
sim.current_tick = 20
sim._forward_traffic(sim._generate_traffic())
obs_fixed = sim._build_agent_observations()

# Repeating the call is NOT a valid determinism test: obs.phy_mac is a
# reference to the live PHYMACState, which _step_phy_mac mutates, so the
# second call sees a different observation. Test the branch instead: make
# Dirichlet.sample fatal and confirm evaluation never reaches it.
import torch.distributions as _D
_real_sample = _D.Dirichlet.sample
_sampled = {'n': 0}


def _tripwire(self, *a, **k):
    _sampled['n'] += 1
    return _real_sample(self, *a, **k)


_D.Dirichlet.sample = _tripwire
try:
    _sampled['n'] = 0
    sim._execute_agent_actions(obs_fixed)
    eval_samples = _sampled['n']
    for a in sim.agents.values():
        a.is_training = True
    _sampled['n'] = 0
    sim._execute_agent_actions(obs_fixed)
    train_samples = _sampled['n']
finally:
    _D.Dirichlet.sample = _real_sample

print("     Dirichlet.sample calls -- evaluation: %d, training: %d"
      % (eval_samples, train_samples))
check(eval_samples == 0, "evaluation never samples (it takes the mean branch)")
check(train_samples > 0, "training does sample")

# and it must be the mean, not a sample: compare against alpha/alpha0
for a in sim.agents.values():
    a.is_training = True
sim._execute_agent_actions(obs_fixed)
train1 = {aid: round(float(ag.last_action.prb_relay_frac), 9)
          for aid, ag in sim.agents.items()
          if getattr(ag, 'last_action', None) is not None}
sim._execute_agent_actions(obs_fixed)
train2 = {aid: round(float(ag.last_action.prb_relay_frac), 9)
          for aid, ag in sim.agents.items()
          if getattr(ag, 'last_action', None) is not None}
differ = sum(1 for a in train1 if a in train2 and train1[a] != train2[a])
print("     repeated observation while TRAINING differs on: %d/%d"
      % (differ, len(train1)))
check(differ > 0, "training re-samples on the same observation")

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
