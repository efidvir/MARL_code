# -*- coding: utf-8 -*-
"""Contract tests for the capability-conditioned relay mask (MARL_RELAY_MASK).

What must be true before a retrain is worth anything:
  1. OFF by default: with the variable unset, logits are untouched — every
     shipped result stays reproducible.
  2. The mask holds on the LIVE path: with it on, no agent whose own
     relay_capable bit is 0 ever executes CAPACITY_BOOST, over many sampled
     training ticks; steerable agents still can.
  3. PPO consistency: the collection-time log-prob of a sampled batch equals
     the update-path log-prob of the same (obs, action) under the same
     temperature — i.e. pi_old and pi_new describe the SAME masked
     distribution, so the ratio at epoch start is exactly 1.
  4. Physical honesty: the mask removes an alias, not a capability — verified
     in-engine by the fall-through in _step_phy_mac (CAPACITY_BOOST without
     has_multihaul selects targets exactly like LOCAL_REROUTE).

Run:  MARL_RELAY_MASK=capability MARL_STEER_MODEL=actionable python test_relay_mask.py
      (test 1 spawns its own unmasked subprocess)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


if os.environ.get('MARL_RELAY_MASK') != 'capability':
    print("re-run with MARL_RELAY_MASK=capability MARL_STEER_MODEL=actionable")
    sys.exit(2)

import torch
import sixg_sim.agent as A
from sixg_sim.agent import (OBS_COL_RELAY_CAPABLE, RELAY_CB_IDX,
                            apply_relay_mask)

print("1. default-off proven in a clean subprocess")
code = (
    "import os,sys,torch; sys.path.insert(0, %r); "
    "import sixg_sim.agent as A; "
    "obs=torch.zeros(4, 60); obs[:2, %d]=1.0; "
    "logits={'relay': torch.randn(4,5)}; "
    "out=A.apply_relay_mask(logits, obs); "
    "assert out is logits, 'must be an identity no-op when off'; "
    "print('SUBPROC-OK')"
) % (os.path.dirname(os.path.abspath(__file__)), OBS_COL_RELAY_CAPABLE)
env = {k: v for k, v in os.environ.items() if k != 'MARL_RELAY_MASK'}
r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                   text=True, env=env)
check("SUBPROC-OK" in r.stdout, "unset env -> apply_relay_mask is identity")

print()
print("2. unit behaviour of the mask itself")
obs = torch.zeros(4, 60)
obs[:2, OBS_COL_RELAY_CAPABLE] = 1.0          # agents 0,1 steerable; 2,3 not
logits = {'relay': torch.zeros(4, 5), 'tx_power': torch.zeros(4, 5)}
out = apply_relay_mask(logits, obs)
check(bool((out['relay'][:2] == 0).all()), "steerable rows untouched")
check(bool((out['relay'][2:, RELAY_CB_IDX] < -1e8).all()),
      "CAPACITY_BOOST logit masked at non-steerable rows")
check(bool((out['relay'][2:, [0, 1, 2, 4]] == 0).all()),
      "other relay actions untouched")
check(bool((logits['relay'] == 0).all()), "input tensor not modified in place")
check(out['tx_power'] is logits['tx_power'], "other heads passed through")

print()
print("3. live-path enforcement + steerable sites still boost")
import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
from sixg_sim.phy_mac_state import RelayMode

_t = R.build_comparison_topology(51)
topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 51, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig())
sim.island_mode = True
sim.control_plane.set_island_mode(True)
for a in sim.agents.values():
    a.is_training = True                       # stochastic: worst case for a mask

viol, boosts, nonsteer_seen = 0, 0, 0
for tick in range(1, 41):
    sim.current_tick = tick
    sim._forward_traffic(sim._generate_traffic())
    obs_all = sim._build_agent_observations()
    sim._execute_agent_actions(obs_all)
    for aid, ag in sim.agents.items():
        act = getattr(ag, 'last_action', None)
        o = obs_all.get(aid)
        if act is None or o is None:
            continue
        capable = float(getattr(o.connectivity, 'relay_capable', 0.0)) >= 0.5
        if int(act.relay_mode_idx) == RELAY_CB_IDX:
            boosts += 1
            if not capable:
                viol += 1
        if not capable:
            nonsteer_seen += 1
print("     40 sampled ticks: %d CAPACITY_BOOST draws, %d at non-steerable "
      "sites (%d non-steerable agent-ticks observed)"
      % (boosts, viol, nonsteer_seen))
check(nonsteer_seen > 0, "non-steerable agents were actually exercised")
check(viol == 0, "no non-steerable agent ever sampled CAPACITY_BOOST")
check(boosts > 0, "steerable agents still sample CAPACITY_BOOST")

print()
print("4. PPO consistency: collection log-prob == update-path log-prob")
from sixg_sim.mappo_trainer import MAPPOTrainer
ids = [aid for aid in sim.agents if obs_all.get(aid) is not None][:16]
obs_b = torch.stack([sim.agents[i].observation_to_tensor(obs_all[i]) for i in ids])
ref = sim.agents[ids[0]]
with torch.no_grad():
    bl = ref.policy_net(obs_b)
bl = apply_relay_mask(bl, obs_b)
acts, prbs = [], []
import torch.nn.functional as F
for k in range(len(ids)):
    row = {h: int(torch.distributions.Categorical(
        logits=bl[h][k]).sample()) for h in A.DISCRETE_HEADS if h in bl}
    row['handover'] = int(torch.distributions.Categorical(logits=bl['handover'][k]).sample()) if 'handover' in bl else 0
    row['scheduler'] = int(torch.distributions.Categorical(logits=bl['scheduler'][k]).sample()) if 'scheduler' in bl else 0
    acts.append(row)
    al = F.softplus(bl['prb'][k]) + 1.0
    prbs.append((al / al.sum()).tolist())

trainer = MAPPOTrainer.__new__(MAPPOTrainer)      # no full init needed
trainer.agents = {i: sim.agents[i] for i in ids}
trainer.device = torch.device('cpu')
lp_collect = trainer.compute_log_prob_batch(ids, obs_b, acts, prbs)

# update-path recomputation, exactly as MAPPOTrainer.update does it
logits_u = apply_relay_mask(ref.policy_net(obs_b), obs_b)
lp_up = torch.zeros(len(ids))
for h in A.DISCRETE_HEADS:
    if h in logits_u:
        d = torch.distributions.Categorical(logits=logits_u[h])
        lp_up = lp_up + d.log_prob(torch.tensor([a.get(h, 0) for a in acts]))
from sixg_sim.mappo_trainer import _prb_log_prob
lp_up = lp_up + _prb_log_prob(logits_u['prb'], torch.tensor(prbs))
gap = max(abs(a - b) for a, b in zip(lp_collect, lp_up.tolist()))
print("     max |collect - update| over %d agents: %.2e" % (len(ids), gap))
check(gap < 1e-4, "pi_old and pi_new agree exactly -> PPO ratio starts at 1")
check(all(v > -1e6 for v in lp_collect),
      "no stored action has a masked (-inf) log-prob")

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
