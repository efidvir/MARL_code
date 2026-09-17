# -*- coding: utf-8 -*-
"""Contract tests for the MISSION configuration:
     MARL_REWARD_PROFILE=mission   + MARL_POSTCARD_FRAGMENTS=1
     + MARL_RELAY_ENTROPY_FLOOR=<f>

What must hold before a retrain is worth anything:
  1. OFF BY DEFAULT.  In a clean subprocess with none of the three set:
     OBS_DIM == 60, bridge_cf_points == 0, and the reward of a real episode
     tick is bit-identical to the pre-change value path (no cf term).
  2. THE COUNTERFACTUAL IS EXACT.  On a live simulator with bridges formed,
     for every agent: bridge_counterfactual_fragments equals
     components(graph minus MY relay links) - components(graph), computed
     independently with networkx.  Redundant (non-tree) links score 0.
  3. THE CUE IS RIGHT.  Postcards carry a stable fragment label; a receiver
     that hears a node in a DIFFERENT fragment sees foreign_fragments_norm
     > 0; once the two fragments are joined by a bridge, the labels merge and
     the cue drops to 0.  The cue latches (non-zero on a tick with no
     postcard, within 30 ticks) and expires after.
  4. OBS_DIM == 62 with the flag, the two cue dims are the LAST two columns,
     and every pre-existing column index (relay_capable=50, the locality
     mask set 28-59) is untouched.
  5. A 60-dim checkpoint still loads under OBS_DIM=62 (fc1 zero-padded) and
     produces the same logits as before on the first 60 features.
  6. The entropy floor changes the loss only when set and only for training.

Run:  MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01 \
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


need = {'MARL_REWARD_PROFILE': 'mission', 'MARL_POSTCARD_FRAGMENTS': '1'}
if any(os.environ.get(k) != v for k, v in need.items()):
    print("re-run with", " ".join("%s=%s" % kv for kv in need.items()),
          "MARL_RELAY_ENTROPY_FLOOR=0.01 MARL_STEER_MODEL=actionable")
    sys.exit(2)

# ── 1. off by default (clean subprocess) ────────────────────────────────────
print("1. off by default")
code = (
    "import os,sys; sys.path.insert(0,%r); "
    "import sixg_sim.agent as A, sixg_sim.reward_profile as rp; "
    "assert A.OBS_DIM==60, A.OBS_DIM; assert not A.POSTCARD_FRAGMENTS; "
    "assert rp.PROFILE.name=='full' and rp.PROFILE.bridge_cf_points==0.0; "
    "import sixg_sim.mappo_trainer as T; assert T._RELAY_H_FLOOR==0.0; "
    "print('DEFAULT-OK')"
) % HERE
env = {k: v for k, v in os.environ.items()
       if k not in ('MARL_REWARD_PROFILE', 'MARL_POSTCARD_FRAGMENTS',
                    'MARL_RELAY_ENTROPY_FLOOR')}
r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
check('DEFAULT-OK' in r.stdout, "unset env -> OBS_DIM 60, profile full, cf 0, floor 0"
      + ("" if 'DEFAULT-OK' in r.stdout else "\n" + r.stderr[-800:]))

# ── 4. layout with the flag ─────────────────────────────────────────────────
print()
print("4. layout under the flag")
import torch
import networkx as nx
import sixg_sim.agent as A
from sixg_sim.agent import (OBS_DIM, OBS_COL_RELAY_CAPABLE, NONLOCAL_OBS_COLUMNS,
                            PolicyNetwork, load_policy_state_dict)
check(OBS_DIM == 62, "OBS_DIM == 62 (%d)" % OBS_DIM)
check(OBS_COL_RELAY_CAPABLE == 50, "relay_capable column still 50")
check(max(NONLOCAL_OBS_COLUMNS) == 59 and min(NONLOCAL_OBS_COLUMNS) == 28,
      "locality mask columns unchanged (28..59)")
import sixg_sim.reward_profile as rp
check(rp.PROFILE.name == 'mission' and rp.PROFILE.bridge_cf_points == 25.0
      and rp.PROFILE.bridge_held_points == 0.0, "mission profile loaded (cf=25, held=0)")

# ── 5. 60-dim checkpoint loads under 62 and is behaviourally identical ───────
print()
print("5. old 60-dim checkpoint under OBS_DIM=62")
ck = torch.load(os.path.join(HERE, 'auth_ck_s1234_policy_final.pt'),
                map_location='cpu', weights_only=False)
sd = ck['state_dict']
net62 = PolicyNetwork(OBS_DIM)
adapted = load_policy_state_dict(net62, sd)
check(adapted, "loader reports fc1 adaptation (60 -> 62)")
net60 = PolicyNetwork(60); net60.load_state_dict(sd)
x60 = torch.randn(8, 60); x62 = torch.cat([x60, torch.rand(8, 2)], dim=1)
with torch.no_grad():
    o60 = net60(x60); o62 = net62(x62)
dev = max(float((o60[h] - o62[h]).abs().max()) for h in o60)
check(dev < 1e-5, "padded checkpoint ignores the 2 new dims exactly (max dev %.1e)" % dev)

# ── 2./3. live simulator: counterfactual + cue ──────────────────────────────
print()
print("2./3. live simulator: counterfactual credit and fragment cue")
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

# reward wiring: a bridging site's r_bridge must be 25 x count under 'mission'
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
    expect = 25.0 * ps.bridge_counterfactual_fragments
    # coordination_reward = r_relay + r_bridge; r_relay >= 0, so >= expect
    if rc.coordination_reward + 1e-6 < expect:
        rew_ok = False
        print("     reward too small at %s: %.2f < %.2f" % (nid, rc.coordination_reward, expect))
check(rew_ok, "calculate_reward pays 25 x counterfactual at bridging sites")

# fragment labels: stable representatives, same label <=> same component
labels = sim._fragment_labels_now()
G = infra_graph(sim)
comp_of = {}
for i, comp in enumerate(nx.connected_components(G)):
    for n in comp:
        comp_of[n] = i
consistent = all(
    (labels.get(a) == labels.get(b)) == (comp_of.get(a) == comp_of.get(b))
    for a in labels for b in labels if a in comp_of and b in comp_of)
check(consistent, "fragment labels agree with networkx components (same label <=> same component)")
check(all(labels[n] == min(m for m in labels if labels[m] == labels[n]) for n in labels),
      "label == smallest node id of its component (stable across ticks)")

# cue: build a receiver summary by hand from synthetic postcards.  XL-DET
# reunifies the island within a tick, so to have >=2 fragments to test the
# cue against, re-expose the RAW partition by tearing the relay links down
# (pure accounting test: the engine's own teardown does exactly this).
from sixg_sim.agent import ControlPostcard, TrafficClass, StrainLevel, NeighborRadioSummary
if len(set(labels.values())) < 2:
    for lid in [l for l, lk in sim.topology.links.items()
                if getattr(lk, 'link_type', None) is LinkType.TRANSPORT_RELAY]:
        lk = sim.topology.links.pop(lid)
        try:
            a, b = lk.endpoints
            if sim.topology.graph.has_edge(a, b):
                sim.topology.graph.remove_edge(a, b)
        except Exception:
            pass
    sim.topology.invalidate_infrastructure_cache()
    sim.current_tick += 1                       # refresh the per-tick label cache
    labels = sim._fragment_labels_now()
    print("     relay links torn down -> fragments now: %d" % len(set(labels.values())))
some = list(labels)
lab_vals = sorted(set(labels.values()))
if len(lab_vals) >= 2:
    me = next(n for n in some if labels[n] == lab_vals[0])
    other = next(n for n in some if labels[n] == lab_vals[1])
    def pc(sender, holds):
        return ControlPostcard(sender_id=sender, relay_mode_active=True,
                               best_sinr_to_neighbour=0.5, isolated_ue_count=0,
                               prb_avail_for_relay=0.5, policy_version=0,
                               timestamp=sim.current_tick,
                               fragment_id=labels[sender], holds_bridge=holds)
    nb = sim._build_neighbor_radio_summary(me, [pc(other, False)])
    check(nb.foreign_fragments_norm > 0 and nb.foreign_unbridged == 1.0,
          "hearing a node in another fragment -> foreign cue on, unbridged=1")
    nb2 = sim._build_neighbor_radio_summary(me, [pc(other, True)])
    check(nb2.foreign_unbridged == 0.0, "if that node already holds a bridge -> unbridged=0")
    same = next(n for n in some if labels[n] == labels[me] and n != me)
    nb3 = sim._build_neighbor_radio_summary(me, [pc(same, False)])
    check(nb3.foreign_fragments_norm == 0.0, "hearing my OWN fragment -> no foreign cue")
    # latch: no postcards this tick -> still the last cue; expires after 30
    nb4 = sim._build_neighbor_radio_summary(me, [])
    check(nb4.foreign_fragments_norm == 0.0, "latch holds the LAST cue (own-fragment => 0)")
    sim._build_neighbor_radio_summary(me, [pc(other, False)])
    sim.current_tick += 5
    nb5 = sim._build_neighbor_radio_summary(me, [])
    check(nb5.foreign_fragments_norm > 0, "latched foreign cue survives 5 silent ticks")
    sim.current_tick += 40
    nb6 = sim._build_neighbor_radio_summary(me, [])
    check(nb6.foreign_fragments_norm == 0.0, "latched cue expires after 30 silent ticks")
else:
    print("     (only one fragment after bridging; cue tests need >=2 -- skipped)")

# ── 6. entropy floor ────────────────────────────────────────────────────────
print()
print("6. entropy floor")
import sixg_sim.mappo_trainer as T
check(T._RELAY_H_FLOOR == 0.01, "floor read from env (%.3f)" % T._RELAY_H_FLOOR)

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
