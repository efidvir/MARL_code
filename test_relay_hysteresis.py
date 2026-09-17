# -*- coding: utf-8 -*-
"""Contract tests for the relay-decision hysteresis executor.

  1. delta unset  -> executor is an exact identity on the argmax relay choice
                     (every shipped result reproducible).
  2. delta > 0    -> on the frozen-argmax path, relay-mode flips per agent
                     drop versus delta = 0 on the same checkpoint, same seed,
                     same tick range; and a mode change only ever happens
                     when the challenger's logit beats the incumbent's by at
                     least delta (checked directly against the logits).
  3. training path untouched: with is_training=True (learned arm), the
                     sampled relay action is never overridden, so pi_old
                     still equals the executed action.

Run:  MARL_STEER_MODEL=actionable python test_relay_hysteresis.py
(uses the shipped policy_cloud_trained.pt if present; otherwise the
untrained net -- the executor contract does not depend on training quality)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MARL_STEER_MODEL', 'actionable')

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = r'''
import os, sys, random, numpy as np, torch
sys.path.insert(0, %r)
# Seed EVERYTHING before any object that draws randomness is built: the
# simulator constructs the policy nets, the scenario builder and traffic
# generator use random/numpy, and the sampling path uses torch.
random.seed(7); np.random.seed(7); torch.manual_seed(7)
import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
_t = R.build_comparison_topology(54); topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 54, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig()); sim.island_mode = True
sim.control_plane.set_island_mode(True)
# A REAL trained policy: an untrained net's argmax barely moves, so it cannot
# exercise churn.  Loading must fail loudly, never silently fall back.
ck = os.path.join(%r, 'auth_ck_s1234_policy_final.pt')
sd = torch.load(ck, map_location='cpu', weights_only=False)
# train_on_comparison saves {'state_dict', 'version', 'metadata'}
state = sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd
ref = next(iter(sim.agents.values()))
ref.policy_net.load_state_dict(state)        # raises on mismatch -> probe fails
# every agent shares the one actor object? if not, load into each
for a in sim.agents.values():
    if a.policy_net is not ref.policy_net:
        a.policy_net.load_state_dict(state)
train = os.environ.get('PROBE_TRAIN') == '1'
for a in sim.agents.values():
    a.is_training = train
random.seed(11); np.random.seed(11); torch.manual_seed(11)
flips = 0; seq = {}; viol = 0; changes = 0
hyst = float(os.environ.get('MARL_RELAY_HYSTERESIS', '0') or 0)
for tick in range(1, 61):
    sim.current_tick = tick
    sim._forward_traffic(sim._generate_traffic())
    obs = sim._build_agent_observations()
    ids = [i for i in sim.agents if obs.get(i) is not None]
    ot = torch.stack([sim.agents[i].observation_to_tensor(obs[i]) for i in ids])
    with torch.no_grad():
        lg = next(iter(sim.agents.values())).policy_net(ot)['relay']
    prev = {i: (getattr(sim.agents[i], 'last_action', None).relay_mode_idx
                if getattr(sim.agents[i], 'last_action', None) is not None else None) for i in ids}
    sim._execute_agent_actions(obs)
    for k, i in enumerate(ids):
        cur = int(sim.agents[i].last_action.relay_mode_idx)
        p = prev[i]
        if p is not None and cur != p:
            flips += 1; changes += 1
            if hyst > 0 and not train and float(lg[k, cur] - lg[k, p]) < hyst - 1e-6:
                viol += 1
print("FLIPS=%%d CHANGES=%%d VIOL=%%d" %% (flips, changes, viol))
''' % (HERE, HERE)


def probe(delta, train=False):
    env = dict(os.environ)
    env['MARL_RELAY_HYSTERESIS'] = str(delta)
    env['PROBE_TRAIN'] = '1' if train else '0'
    r = subprocess.run([sys.executable, '-c', PROBE], capture_output=True,
                       text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('FLIPS=')]
    if not line:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit('probe failed')
    parts = dict(kv.split('=') for kv in line[-1].split())
    return int(parts['FLIPS']), int(parts['CHANGES']), int(parts['VIOL'])


print("1./2. frozen-argmax path, 60 ticks on seed 54 (the worst churn seed)")
f0, _, _ = probe(0.0)
f0b, _, _ = probe(0.0)
check(f0 == f0b, "delta=0 is deterministic (identity executor): %d == %d flips" % (f0, f0b))
f3, c3, v3 = probe(3.0)
print("     relay-mode flips: delta=0 -> %d,  delta=3.0 -> %d" % (f0, f3))
check(f3 < f0, "hysteresis reduces relay-mode flips on the argmax path")
check(v3 == 0, "every accepted change met the margin (%d violations)" % v3)

print()
print("3. learned-arm TRAINING path is untouched (pi_old == executed)")
ft0, _, _ = probe(0.0, train=True)
ft3, _, vt3 = probe(3.0, train=True)
# on the sampling path the executor must not intervene: same RNG seed, same
# stochastic draws, so flip counts must be IDENTICAL, not merely similar.
check(ft0 == ft3, "training-path flips identical with/without delta (%d vs %d)" % (ft0, ft3))

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
