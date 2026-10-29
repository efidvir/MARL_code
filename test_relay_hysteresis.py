# -*- coding: utf-8 -*-
"""Contract tests for the relay-decision hysteresis executor.

  1. delta unset  -> executor is an exact identity on the argmax relay choice
                     (every shipped result reproducible).
  2. delta > 0    -> on the frozen-argmax path, relay-mode flips per agent
                     drop versus delta = 0 on the same checkpoint, same seed,
                     same tick range; and a mode change only ever happens
                     when the challenger's logit beats the incumbent's by at
                     least delta (checked directly against the logits).

                     THE MARGIN IS CHECKED WHERE IT CAN FIRE.  At delta = 3.0
                     this policy accepts no change at all, so a violation
                     count taken there is a count over an empty set and is
                     printed, not checked.  The margin rule is measured at
                     half the SMALLEST margin the delta = 0 run produced:
                     every change of that run must still clear it, so the
                     comparison is evaluated once per accepted change, and a
                     run that accepted none FAILS by name instead of
                     reporting "0 violations".
  3. training path untouched: with is_training=True (learned arm), the
                     sampled relay action is never overridden, so pi_old
                     still equals the executed action.

Run:  MARL_STEER_MODEL=actionable python test_relay_hysteresis.py
(uses auth_ck_s1234_policy_final.pt if present; otherwise a schema-current
checkpoint minted by sixg_sim.agent.mint_schema_current_checkpoint -- the
executor's INDEX contract does not depend on training quality, but checks 2
and 3 need an actor that churns and say so by name when it does not)

RERUN_SPEC WP7 re-based the relay head from the 5-wide RelayMode ENUM index
to the 3-wide HEAD index (RELAY_HEAD_MODES).  The probe below therefore
indexes the relay logit row with `relay_head_idx`, the same index the engine's
Simulator._relay_hysteresis compares, and carries a tripwire on the row width
so a further re-base fails with a message rather than an IndexError on
CAPACITY_BOOST and a wrong-pair comparison on LOCAL_REROUTE.
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
# RERUN_SPEC WP7 / WP8 Part A: the relay head is 3 WIDE and indexed by the
# HEAD index (RELAY_HEAD_MODES), while PHYMACAction.relay_mode_idx is still
# the 5-valued ENUM index.  Everything below that touches the logit row uses
# relay_head_idx; relay_mode_idx is kept only for human-readable reporting.
from sixg_sim.agent import (N_RELAY_MODES, RELAY_HEAD_MODES,
                            mint_schema_current_checkpoint)
# Prefer the trained replicate: an untrained net's argmax barely moves, so it
# cannot exercise churn.  When it is absent (it is not tracked in this tree),
# mint a schema-current checkpoint so the executor contract still RUNS
# instead of dying at a FileNotFoundError; the probe prints which it used and
# the caller decides which checks that supports.
ck = os.path.join(%r, 'auth_ck_s1234_policy_final.pt')
if os.path.exists(ck):
    src = 'trained'
    sd = torch.load(ck, map_location='cpu', weights_only=False)
else:
    src = 'minted'
    _mck = os.path.join(%r, 'output', 'tmp_tests', 'hyst_minted_policy.pt')
    mint_schema_current_checkpoint(_mck, torch_seed=1234)
    sd = torch.load(_mck, map_location='cpu', weights_only=False)
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
flips = 0; seq = {}; viol = 0; changes = 0; margins = []
hyst = float(os.environ.get('MARL_RELAY_HYSTERESIS', '0') or 0)
for tick in range(1, 61):
    sim.current_tick = tick
    sim._forward_traffic(sim._generate_traffic())
    obs = sim._build_agent_observations()
    ids = [i for i in sim.agents if obs.get(i) is not None]
    ot = torch.stack([sim.agents[i].observation_to_tensor(obs[i]) for i in ids])
    with torch.no_grad():
        lg = next(iter(sim.agents.values())).policy_net(ot)['relay']
    # TRIPWIRE.  The margin check below indexes this row, so the row's width
    # and the index space MUST agree.  At 9dec943 the head was 5 wide and the
    # ENUM index was the right one; WP7 made it 3 wide.  If either is re-based
    # again, fail here with a message instead of raising IndexError on one
    # value (CAPACITY_BOOST) and silently comparing the wrong pair on another.
    assert lg.shape[1] == N_RELAY_MODES == len(RELAY_HEAD_MODES), (
        'relay logit row is %%d wide, head index space is %%d'
        %% (lg.shape[1], N_RELAY_MODES))
    prev = {i: (getattr(sim.agents[i], 'last_action', None).relay_head_idx
                if getattr(sim.agents[i], 'last_action', None) is not None else None) for i in ids}
    sim._execute_agent_actions(obs)
    for k, i in enumerate(ids):
        _act = sim.agents[i].last_action
        cur = int(_act.relay_head_idx)
        assert 0 <= cur < N_RELAY_MODES, 'relay_head_idx %%d out of head range' %% cur
        p = prev[i]
        if p is not None and cur != p:
            flips += 1; changes += 1
            # THE HEAD-INDEXED COMPARISON, evaluated on EVERY accepted change.
            # It used to sit inside `if hyst > 0 and not train`, where no
            # probe this file runs ever reached it: at delta=0 `hyst > 0`
            # short-circuits, both training probes make `not train` False,
            # and at delta=3.0 this policy has no accepted change at all.  The
            # line WP7's re-base was about was therefore never executed, and
            # "0 margin violations" was a count over an empty set.
            margin = float(lg[k, cur] - lg[k, p])
            margins.append(margin)
            if hyst > 0 and not train and margin < hyst - 1e-6:
                viol += 1
print("SRC=%%s" %% src)
print("FLIPS=%%d CHANGES=%%d VIOL=%%d" %% (flips, changes, viol))
print("MARGINS=%%s" %% ",".join("%%.10g" %% m for m in margins))
''' % (HERE, HERE, HERE)


SRC = []


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
    for l in r.stdout.splitlines():
        if l.startswith('SRC='):
            SRC.append(l.split('=', 1)[1].strip())
    mline = [l for l in r.stdout.splitlines() if l.startswith('MARGINS=')]
    if not mline:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit('probe printed no margin line')
    raw = mline[-1].split('=', 1)[1].strip()
    margins = [float(x) for x in raw.split(',') if x]
    parts = dict(kv.split('=') for kv in line[-1].split())
    if len(margins) != int(parts['CHANGES']):
        raise SystemExit('probe reported %d changes and %d margins'
                         % (int(parts['CHANGES']), len(margins)))
    return (int(parts['FLIPS']), int(parts['CHANGES']), int(parts['VIOL']),
            margins)


print("1./2. frozen-argmax path, 60 ticks on seed 54 (the worst churn seed)")
f0, c0, v0, m0 = probe(0.0)
f0b, _, _, _ = probe(0.0)
print("     policy source: %s" % (SRC[0] if SRC else '?'))
check(f0 == f0b, "delta=0 is deterministic (identity executor): %d == %d flips" % (f0, f0b))
f3, c3, v3, m3 = probe(3.0)
print("     relay-mode flips: delta=0 -> %d,  delta=3.0 -> %d" % (f0, f3))
# NON-VACUITY, named separately so "0 flips" can never read as a pass: the
# margin check and the reduction check both need the argmax path to churn.
check(f0 > 0, "the argmax path actually churns at delta=0 (%d flips) -- "
              "without flips there is nothing for the margin to gate" % f0)
check(f3 < f0, "hysteresis reduces relay-mode flips on the argmax path")

# THE MARGIN RULE, measured where it can actually fire.  delta=3.0 accepts
# NOTHING on this policy, so its violation count is a count over an empty
# set: it is printed, not checked, because "0 violations out of 0 accepted
# changes" is not evidence that the head-indexed comparison works.  The rule
# is checked instead at a delta the executor still admits changes at --
# half the SMALLEST margin the argmax path produced, so every delta=0 change
# must still clear it and the comparison at `lg[k, cur] - lg[k, p]` is
# evaluated once per change.
print("     delta=3.0 accepted %d changes, so its %d violations are vacuous"
      % (c3, v3))
_mmin = min(m0) if m0 else 0.0
check(m0 and _mmin > 0,
      "every argmax change has a positive head-indexed logit margin "
      "(min %.6g over %d changes)" % (_mmin, len(m0)))
DELTA_M = 0.5 * _mmin
fm, cm, vm, mm = probe(DELTA_M)
check(cm > 0, "the margin rule is EXERCISED at delta=%.6g: %d accepted "
              "change(s) -- a zero here means the check below counts nothing"
              % (DELTA_M, cm))
check(vm == 0, "every accepted change met the margin at delta=%.6g "
               "(%d violations over %d accepted changes)" % (DELTA_M, vm, cm))

print()
print("3. learned-arm TRAINING path is untouched (pi_old == executed)")
ft0, _, _, _ = probe(0.0, train=True)
ft3, _, vt3, _ = probe(3.0, train=True)
# on the sampling path the executor must not intervene: same RNG seed, same
# stochastic draws, so flip counts must be IDENTICAL, not merely similar.
check(ft0 == ft3, "training-path flips identical with/without delta (%d vs %d)" % (ft0, ft3))
check(ft0 > 0, "the training/sampling path actually churns (%d flips)" % ft0)

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
