# -*- coding: utf-8 -*-
"""Contract tests for the MISSION configuration:
     MARL_RELAY_ENTROPY_FLOOR=<f>  (+ MARL_REWARD_PROFILE=mission, which the
                                    WP8 tripwire accepts and which must not
                                    move the frozen reward)

RERUN_SPEC WP8 FROZE the reward.  `MARL_REWARD_PROFILE` no longer selects a
coefficient set, `sixg_sim/reward_profile.py` is DELETED, and the EVALUATED
set -- the former `mission` profile, which is what every shipped script
exported -- is hard-wired in `sixg_sim/reward_spec.py` (the variable survives
as a tripwire that accepts only `mission`).  Item 1 is therefore INVERTED:
what used to be "the counterfactual credit is OFF unless the variable is set"
is now "the coefficients are the SAME with the variable unset", which is the
property the freeze exists to give.

RERUN_SPEC WP5 REMOVED the fragment cue (Block E) and the postcard fragment
labels, and with them the MARL_POSTCARD_FRAGMENTS switch, which the engine
now refuses.  Items 1, 3 and 4 below check the removal.

What must hold before a retrain is worth anything:
  1. THE ENVIRONMENT CANNOT MOVE A COEFFICIENT.  In a clean subprocess with
     none of the variables set, the frozen reward id and the per-site
     counterfactual credit are exactly the ones in force here; and a
     subprocess that sets MARL_POSTCARD_FRAGMENTS=1 is REFUSED at import.
  2. THE COUNTERFACTUAL IS EXACT.  On a live simulator with bridges formed,
     for every agent: bridge_counterfactual_fragments equals
     components(graph minus MY relay links) - components(graph), computed
     independently with networkx.  Redundant (non-tree) links score 0.
  3. NO CUE IS LEFT.  The simulator has no fragment-label service for
     postcards, the postcard carries no label or bridge bit, and Block B has
     no foreign-fragment fields (tests/test_wp5_postcards.py scans the tree).
  4. WP6: the observation layout is FIXED IN CODE -- ACTOR_OBS_DIM == 26,
     CRITIC_OBS_DIM == 37 (WP13 removed four constant critic globals,
     D13.3), no Block E cue column in the actor vector, and
     no environment variable changes either.
  5. WP6: a pre-WP6 checkpoint is REFUSED, not zero-padded or truncated --
     the schema drops interior columns, so adapting fc1 would mis-align every
     later column.
  6. The entropy floor changes the loss only when set and only for training.

Run:  MARL_REWARD_PROFILE=mission MARL_RELAY_ENTROPY_FLOOR=0.01 \
      MARL_STEER_MODEL=actionable python test_mission_config.py
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


# The one variable this script's own contract (item 6) is about.  Neither the
# fragment switch (WP5 refuses it) nor the reward profile (WP8 froze the
# coefficients) is required any more.
need = {'MARL_RELAY_ENTROPY_FLOOR': '0.01'}
if any(os.environ.get(k) != v for k, v in need.items()):
    print("re-run with", " ".join("%s=%s" % kv for kv in need.items()),
          "MARL_REWARD_PROFILE=mission MARL_STEER_MODEL=actionable")
    sys.exit(2)

import sixg_sim.reward_spec as RS

# -- 1. the reward does not depend on the environment (clean subprocess) -----
print("1. the frozen reward is the same with the environment unset")
code = (
    "import os,sys; sys.path.insert(0,%r); "
    "import sixg_sim.agent as A, sixg_sim.reward_spec as RS; "
    "assert A.OBS_DIM==26, A.OBS_DIM; "
    "assert not hasattr(A, 'POSTCARD_FRAGMENTS'); "
    "import sixg_sim.mappo_trainer as T; assert T._RELAY_H_FLOOR==0.0; "
    "print('REWARD-ID', RS.reward_id(), RS.FROZEN.local_bridge_counterfactual)"
) % HERE
env = {k: v for k, v in os.environ.items()
       if k not in ('MARL_REWARD_PROFILE', 'MARL_POSTCARD_FRAGMENTS',
                    'MARL_RELAY_ENTROPY_FLOOR')}
r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
want = "REWARD-ID %s %r" % (RS.reward_id(), RS.FROZEN.local_bridge_counterfactual)
check(want in r.stdout,
      "unset env -> actor obs 26, floor 0, and the SAME frozen reward (%s)"
      % RS.reward_id()
      + ("" if want in r.stdout else " -- got %r; %s"
         % (r.stdout.strip(), r.stderr[-800:])))
r2 = subprocess.run([sys.executable, '-c',
                     "import sys; sys.path.insert(0,%r); import sixg_sim.agent" % HERE],
                    capture_output=True, text=True,
                    env=dict(env, MARL_POSTCARD_FRAGMENTS='1'))
check(r2.returncode != 0 and 'REMOVED by RERUN_SPEC WP5' in r2.stderr,
      "MARL_POSTCARD_FRAGMENTS=1 is refused at import (the cue was removed)")

# ── 4. layout, fixed in code ────────────────────────────────────────────────
print()
print("4. layout (no flag any more)")
import torch
import networkx as nx
import sixg_sim.agent as A
from sixg_sim.agent import (OBS_DIM, ACTOR_OBS_DIM, CRITIC_OBS_DIM,
                            ACTOR_OBS_NAMES, SCHEMA_ID,
                            PolicyNetwork, assert_checkpoint_schema)
check(ACTOR_OBS_DIM == 26 and OBS_DIM == ACTOR_OBS_DIM,
      "actor observation is 26 wide (%d)" % ACTOR_OBS_DIM)
check(CRITIC_OBS_DIM == 37, "critic observation is 37 wide (%d)" % CRITIC_OBS_DIM)
check(not any(n.startswith('e_') or 'foreign' in n for n in ACTOR_OBS_NAMES),
      "no Block E cue column in the actor vector")
check(SCHEMA_ID.startswith('obs1-'), "schema id is hashed (%s)" % SCHEMA_ID)
check(RS.FROZEN.local_bridge_counterfactual == 25.0,
      "the frozen per-site counterfactual credit is 25.0 per fragment")
check(not any(f.name.startswith('bridge_held')
              for f in __import__('dataclasses').fields(RS.FROZEN)),
      "the bridge-held term is DELETED, not zero-rated")

# ── 5. the tracked pre-WP6 (62-wide) checkpoint is refused ──────────────────
print()
print("5. a pre-WP6 checkpoint is refused, not adapted")
# ce_ck_s1234/policy_final.pt is the tracked frozen executor, trained on
# the pre-WP6 62-dim masked schema (auth_ck_s1234_policy_final.pt, which
# this check used to read, is not in the tree).
ck = torch.load(os.path.join(HERE, 'ce_ck_s1234', 'policy_final.pt'),
                map_location='cpu', weights_only=False)
try:
    assert_checkpoint_schema(ck, where='test_mission_config')
except ValueError as _e:
    check('schema' in str(_e), "strict loader refuses the pre-WP6 checkpoint")
else:
    check(False, "strict loader accepted a pre-WP6 checkpoint")
from sixg_sim.agent import load_policy_state_dict
try:
    load_policy_state_dict(PolicyNetwork(), ck['state_dict'])
except (ValueError, RuntimeError) as _e:
    check(True, "fc1 width mismatch raises instead of zero-padding")
else:
    check(False, "the fc1 pad/truncate path is still alive")

# ── 2./3. live simulator: counterfactual + cue ──────────────────────────────
print()
print("2./3. live simulator: counterfactual credit, and no fragment cue left")
import run_timeline_comparison as R
from sixg_sim.simulation import Simulator
from sixg_sim.topology import LinkType, NodeType
from sixg_sim.phy_mac_state import RelayMode

_t = R.build_comparison_topology(54); topo = _t[0] if isinstance(_t, tuple) else _t
scen = R.build_comparison_scenario('recovery', 54, topology=topo)
sim = Simulator(topo, scen, R.SimulationConfig())
sim.island_mode = True
sim.control_plane.set_island_mode(True)
# Use XL-DET's actuation so bridges reliably FORM (it boosts everywhere);
# what we test is the accounting, not the policy.
sim.heuristic_action_policy = True
for a in sim.agents.values():
    a.is_training = False

# Drive THROUGH the scenario's own severance (tick 200 in 'recovery') using
# the simulator's real event processor, then on until a bridge exists.
formed_tick = None
for tick in range(1, 361):
    sim.current_tick = tick
    sim._process_events(tick)                    # applies the fragmenting cut
    sim._forward_traffic(sim._generate_traffic())
    obs = sim._build_agent_observations()
    sim._execute_agent_actions(obs)
    if tick > 200 and sim.distinct_bridge_link_ids() and formed_tick is None:
        formed_tick = tick
    if formed_tick and tick >= formed_tick + 12:
        break
print("     raw fragments after severance: %d" % sim.count_infra_components(False))
sim._forward_traffic(sim._generate_traffic())
obs = sim._build_agent_observations()          # refreshes ps.* accounting
bridges = sim.distinct_bridge_link_ids()
print("     bridges formed by tick %s: %d distinct" % (formed_tick, len(bridges)))
check(len(bridges) > 0, "at least one distinct bridge exists to test against")


def infra_graph(sim, drop_links=()):
    G = nx.Graph()
    for nid, n in sim.topology.nodes.items():
        if n.is_survivor and n.node_type != NodeType.UE:
            G.add_node(nid)
    for lid, l in sim.topology.links.items():
        if lid in drop_links or not getattr(l, 'is_up', True):
            continue
        a, b = l.endpoints
        if a in G and b in G:
            G.add_edge(a, b)
    return G


base_c = nx.number_connected_components(infra_graph(sim))
mism = 0; nonzero = 0; checked = 0
for nid, ps in sim.phy_mac_states.items():
    mine = [lid for lid, l in sim.topology.links.items()
            if getattr(l, 'link_type', None) is LinkType.TRANSPORT_RELAY
            and getattr(l, 'is_up', True) and nid in l.endpoints]
    cf_ref = nx.number_connected_components(infra_graph(sim, set(mine))) - base_c
    cf_eng = int(getattr(ps, 'bridge_counterfactual_fragments', 0))
    checked += 1
    if cf_eng != cf_ref:
        mism += 1
        print("     MISMATCH %s: engine %d vs networkx %d (links=%d)" % (nid, cf_eng, cf_ref, len(mine)))
    if cf_eng > 0:
        nonzero += 1
print("     %d agents checked, %d with positive counterfactual, %d mismatches" % (checked, nonzero, mism))
check(mism == 0, "engine counterfactual == networkx recount for every agent")
check(nonzero > 0, "some bridging site earns positive counterfactual credit")

# reward wiring: a bridging site must be paid local_bridge_counterfactual x count
rew_ok = True
for nid, ag in sim.agents.items():
    ps = sim.phy_mac_states.get(nid)
    act = getattr(ag, 'last_action', None)
    if ps is None or act is None or int(ps.bridge_counterfactual_fragments) == 0:
        continue
    o = obs.get(nid)
    if o is None:
        continue
    rc = ag.calculate_reward(o, act, o)
    expect = (RS.FROZEN.local_bridge_counterfactual * ps.bridge_counterfactual_fragments)
    # coordination_reward = r_relay + r_bridge; r_relay >= 0, so >= expect
    if rc.coordination_reward + 1e-6 < expect:
        rew_ok = False
        print("     reward too small at %s: %.2f < %.2f" % (nid, rc.coordination_reward, expect))
check(rew_ok, "calculate_reward pays the frozen rate x counterfactual at bridging sites")

# no cue is left (RERUN_SPEC WP5): no label service, no label on the postcard,
# no foreign-fragment field in Block B
import dataclasses
from sixg_sim.agent import ControlPostcard, NeighborRadioSummary
check(not any(hasattr(sim, n) for n in ('_fragment_labels_now', '_fragment_label_of',
                                         '_apply_fragment_cue')),
      "the simulator has no postcard fragment-label service")
_pc_fields = {f.name for f in dataclasses.fields(ControlPostcard)}
check(not ({'fragment_id', 'holds_bridge', 'isolated_ue_count'} & _pc_fields),
      "the postcard carries no fragment label, bridge bit or network census")
check('unserved_ue_count' in _pc_fields,
      "the postcard carries the sender's OWN unserved-UE count")
_nb_fields = {f.name for f in dataclasses.fields(NeighborRadioSummary)}
check(not any(n.startswith('foreign') for n in _nb_fields),
      "Block B has no foreign-fragment field")

# ── 6. entropy floor ────────────────────────────────────────────────────────
print()
print("6. entropy floor")
import sixg_sim.mappo_trainer as T
check(T._RELAY_H_FLOOR == 0.01, "floor read from env (%.3f)" % T._RELAY_H_FLOOR)

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
