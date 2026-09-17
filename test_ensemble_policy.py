# -*- coding: utf-8 -*-
"""Contract tests for EnsemblePolicy (averaged-logit actor ensemble).

  1. drop-in: same head names and shapes as a single PolicyNetwork
  2. it is a real average: for each head, logits == mean of member logits
  3. all members receive gradient through the average (online adaptation
     of the freeze arm trains the ensemble as one actor)
  4. the ensemble is not degenerate: on real observations its relay argmax
     differs from at least one member's on some rows (else averaging would
     be a no-op), and it never emits NaN
  5. the relay mask still applies through the ensemble (mask helper sees an
     ordinary {head: logits} dict)

Run:  MARL_STEER_MODEL=actionable python test_ensemble_policy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')

import torch
from sixg_sim.agent import (OBS_DIM, EnsemblePolicy, PolicyNetwork,
                            load_policy_state_dict)

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


HERE = os.path.dirname(os.path.abspath(__file__))
paths = [os.path.join(HERE, p) for p in (
    'auth_ck_s1234_policy_final.pt', 'auth_ck_2345_policy_final.pt',
    'auth_ck_3456_policy_final.pt', 'auth_ck_4567_policy_final.pt')]
members = []
for p in paths:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    net = PolicyNetwork(OBS_DIM)
    load_policy_state_dict(net, sd)
    members.append(net)
ens = EnsemblePolicy(members)

print("1./2. drop-in shape + true average on a random batch")
x = torch.randn(32, OBS_DIM)
with torch.no_grad():
    single = members[0](x)
    out = ens(x)
    per = [m(x) for m in members]
check(set(out) == set(single), "same head names as a PolicyNetwork: %s" % sorted(out))
check(all(out[h].shape == single[h].shape for h in out), "same shapes per head")
maxdev = max(float((out[h] - torch.stack([o[h] for o in per]).mean(0)).abs().max())
             for h in out)
check(maxdev < 1e-6, "logits equal the member mean (max dev %.1e)" % maxdev)

print()
print("3. gradient reaches every member")
ens.zero_grad()
loss = sum(v.sum() for v in ens(x).values())
loss.backward()
got = [any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters())
       for m in members]
check(all(got), "all %d members received non-zero gradient" % len(members))
n_single = sum(p.numel() for p in members[0].parameters())
n_ens = sum(p.numel() for p in ens.parameters())
check(n_ens == 4 * n_single, "parameter count is 4x a single actor (%d vs %d)" % (n_ens, n_single))

print()
print("4. non-degenerate on real observations")
import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
_t = R.build_comparison_topology(55); topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 55, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig()); sim.island_mode = True
sim.control_plane.set_island_mode(True)
for tk in range(1, 6):
    sim.current_tick = tk
    sim._forward_traffic(sim._generate_traffic())
    obs = sim._build_agent_observations()
    sim._execute_agent_actions(obs)
ids = [i for i in sim.agents if obs.get(i) is not None]
ob = torch.stack([sim.agents[i].observation_to_tensor(obs[i]) for i in ids])
with torch.no_grad():
    e_arg = ens(ob)['relay'].argmax(1)
    m_arg = [m(ob)['relay'].argmax(1) for m in members]
disagree_members = int(sum((m_arg[0] != m_arg[k]).sum() for k in range(1, 4)))
diff_from_any = int(sum(int((e_arg != a).any()) for a in m_arg))
print("     members disagree with member-0 on %d agent-rows; ensemble differs from %d/4 members somewhere" % (disagree_members, diff_from_any))
check(diff_from_any > 0, "ensemble argmax is not identical to every member (averaging does something)")
check(not any(torch.isnan(v).any() for v in ens(ob).values()), "no NaN on real observations")

print()
print("5. relay mask composes with the ensemble")
os.environ['MARL_RELAY_MASK'] = 'capability'
import importlib, sixg_sim.agent as A
importlib.reload(A)
ob2 = ob.clone(); ob2[:, A.OBS_COL_RELAY_CAPABLE] = 0.0
masked = A.apply_relay_mask(ens(ob2), ob2)
check(bool((masked['relay'][:, A.RELAY_CB_IDX] < -1e8).all()), "mask reaches the ensemble's relay logits")

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
