# -*- coding: utf-8 -*-
"""Unit tests for the two ensemble plumbing fixes in run_timeline_comparison:
  _adapted_checkpoint_path  -- a comma-list checkpoint must map to a REAL file
                               path next to the first member (the old
                               str.replace produced a nonsense path and the
                               freeze arm's torch.save crashed every pass);
  _load_into_agent          -- the rescue pass rebuilds agents as plain
                               PolicyNetworks; when the recovery-adapted
                               state is ensemble-shaped it must install an
                               EnsemblePolicy of the right size and round-trip
                               the weights exactly; plain dicts untouched.
Run:  MARL_STEER_MODEL=actionable python test_ensemble_roundtrip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')

import torch
import run_timeline_comparison as R
from sixg_sim.agent import (OBS_DIM, EnsemblePolicy, PolicyNetwork,
                            load_policy_state_dict)

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def norm(p):
    return p.replace(os.sep, '/')


print("1. adapted-checkpoint path helper")
single = R._adapted_checkpoint_path('/x/auth_ck_s1234/policy_final.pt',
                                    '_adapted_recovery_freeze', 61)
ens_p = R._adapted_checkpoint_path(
    '/x/auth_ck_s1234/policy_final.pt,/x/auth_ck_s2345/policy_final.pt',
    '_adapted_recovery_freeze', 61)
print("     single   ->", norm(single))
print("     ensemble ->", norm(ens_p))
check(norm(single).endswith('auth_ck_s1234/policy_final_adapted_recovery_freeze_seed61.pt'),
      "single checkpoint keeps the historical naming")
check(norm(ens_p) == '/x/auth_ck_s1234/ensemble_adapted_recovery_freeze_seed61.pt',
      "ensemble maps to a real file next to the first member")
check(',' not in ens_p, "no comma survives into the path")

print()
print("2. ensemble-aware per-agent loader")
HERE = os.path.dirname(os.path.abspath(__file__))
members = []
for p in ('auth_ck_s1234_policy_final.pt', 'auth_ck_2345_policy_final.pt',
          'auth_ck_3456_policy_final.pt', 'auth_ck_4567_policy_final.pt'):
    ck = torch.load(os.path.join(HERE, p), map_location='cpu', weights_only=False)
    net = PolicyNetwork(OBS_DIM)
    load_policy_state_dict(net, ck['state_dict'])
    members.append(net)
ens = EnsemblePolicy(members)
sd = ens.state_dict()                    # what run_physics_pass saves per agent


class _Agent:
    pass


fresh = _Agent()
fresh.policy_net = PolicyNetwork(OBS_DIM)   # what the rescue pass constructs
flag = R._load_into_agent(fresh, sd)
check(isinstance(fresh.policy_net, EnsemblePolicy), "ensemble-shaped dict installs an EnsemblePolicy")
check(len(fresh.policy_net.members) == 4, "member count recovered from the keys (4)")
check(flag is False, "adapted-flag False for ensembles")
x = torch.randn(6, OBS_DIM)
with torch.no_grad():
    a = ens(x)
    b = fresh.policy_net(x)
dev = max(float((a[h] - b[h]).abs().max()) for h in a)
check(dev < 1e-6, "weights round-trip exactly (max dev %.1e)" % dev)

plain = _Agent()
plain.policy_net = PolicyNetwork(OBS_DIM)
R._load_into_agent(plain, members[0].state_dict())
check(type(plain.policy_net) is PolicyNetwork, "plain dict leaves a plain PolicyNetwork")
with torch.no_grad():
    dev2 = max(float((members[0](x)[h] - plain.policy_net(x)[h]).abs().max()) for h in a)
check(dev2 < 1e-6, "plain load still exact (max dev %.1e)" % dev2)

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
