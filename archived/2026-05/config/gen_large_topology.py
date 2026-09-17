"""
Generate a large realistic 6G disaster-recovery topology.
Produces:
  config/topology_large.yaml   — 213 nodes, ~520 links
  config/scenario_disaster.yaml — 800-tick multi-zone severance scenario

Run: python config/gen_large_topology.py
"""

import yaml
import math
import random

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Layout parameters
# ─────────────────────────────────────────────────────────────────────────────
ZONES = {
    "north": {"cx": 0.20, "cy": 0.80, "color": "#4a90d9"},
    "south": {"cx": 0.80, "cy": 0.20, "color": "#d94a4a"},
    "east":  {"cx": 0.80, "cy": 0.80, "color": "#4ad97a"},
    "west":  {"cx": 0.20, "cy": 0.20, "color": "#d9c24a"},
}
ZONE_NAMES = list(ZONES.keys())

GNB_PER_ZONE     = 5      # gNB macro sites per zone   → 20 total
RELAY_PER_ZONE   = 10     # IAB/relay nodes per zone   → 40 total
UE_PER_ZONE      = 30     # regular UEs per zone       → 120 total
RESCUE_PER_ZONE  = 5      # rescue-service UEs/zone    → 20 total

# ─────────────────────────────────────────────────────────────────────────────
# Build nodes
# ─────────────────────────────────────────────────────────────────────────────
nodes = []
links = []
link_set = set()
link_counter = [0]

def L(a, b, capacity, latency, ltype):
    key = tuple(sorted([a, b]))
    if key in link_set:
        return
    link_set.add(key)
    link_counter[0] += 1
    links.append({
        "id":        f"L{link_counter[0]:04d}",
        "endpoints": [a, b],
        "capacity":  capacity,
        "latency":   latency,
        "type":      ltype,
    })

# ── Core & National backbone ─────────────────────────────────────────────────
nodes.append({"id": "Core",     "type": "Core",    "initial_energy": 1.0})
nodes.append({"id": "NatUPF1",  "type": "Core",    "initial_energy": 1.0})
nodes.append({"id": "NatUPF2",  "type": "Core",    "initial_energy": 1.0})

# ── Edge UPFs (one per zone + one shared) ────────────────────────────────────
edge_upfs = {}
for zn in ZONE_NAMES:
    uid = f"EdgeUPF_{zn}"
    edge_upfs[zn] = uid
    nodes.append({"id": uid, "type": "EdgeUPF", "initial_energy": 1.0, "coverage_area": zn})

# Backbone: Core ↔ NatUPF ↔ EdgeUPF ring
L("Core", "NatUPF1", 10000, 2, "fiber")
L("Core", "NatUPF2", 10000, 2, "fiber")
for zn in ZONE_NAMES:
    np_id = "NatUPF1" if zn in ("north", "east") else "NatUPF2"
    L(np_id, edge_upfs[zn], 5000, 3, "fiber")

# ── gNB macro sites ──────────────────────────────────────────────────────────
gnbs_by_zone = {}
for zn in ZONE_NAMES:
    gnbs_by_zone[zn] = []
    for i in range(GNB_PER_ZONE):
        gid = f"gNB_{zn}_{i+1}"
        nodes.append({
            "id": gid, "type": "gNB-Site",
            "initial_energy": 1.0, "coverage_area": zn,
        })
        gnbs_by_zone[zn].append(gid)
        # Connect to zone EdgeUPF via fiber backhaul
        L(gid, edge_upfs[zn], 1000, 2, "fiber")

    # Inter-gNB microwave ring within zone
    zone_gnbs = gnbs_by_zone[zn]
    for i in range(len(zone_gnbs)):
        L(zone_gnbs[i], zone_gnbs[(i+1) % len(zone_gnbs)], 500, 3, "microwave")

# Inter-zone gNB links (border gNBs connect to adjacent zones)
border_pairs = [
    ("north", "east"),
    ("north", "west"),
    ("south", "east"),
    ("south", "west"),
    ("east",  "west"),
    ("north", "south"),
]
for za, zb in border_pairs:
    L(gnbs_by_zone[za][0], gnbs_by_zone[zb][-1], 300, 5, "microwave")

# ── Relay / IAB nodes ────────────────────────────────────────────────────────
relays_by_zone = {}
for zn in ZONE_NAMES:
    relays_by_zone[zn] = []
    for i in range(RELAY_PER_ZONE):
        rid = f"relay_{zn}_{i+1}"
        energy = round(random.uniform(0.6, 1.0), 2)
        nodes.append({
            "id": rid, "type": "Relay",
            "initial_energy": energy, "coverage_area": zn,
        })
        relays_by_zone[zn].append(rid)

    # Each relay connects to 2 random gNBs and 1 EdgeUPF in its zone
    zone_gnbs   = gnbs_by_zone[zn]
    zone_relays = relays_by_zone[zn]
    for i, rid in enumerate(zone_relays):
        g1 = zone_gnbs[i % len(zone_gnbs)]
        g2 = zone_gnbs[(i + 1) % len(zone_gnbs)]
        L(rid, g1, 400, 2, "iab")
        L(rid, g2, 300, 3, "iab")
        # Every other relay also links EdgeUPF
        if i % 3 == 0:
            L(rid, edge_upfs[zn], 600, 2, "fiber")

# ── UE nodes ─────────────────────────────────────────────────────────────────
ues_by_zone = {}
for zn in ZONE_NAMES:
    ues_by_zone[zn] = []
    zone_gnbs = gnbs_by_zone[zn]

    for i in range(UE_PER_ZONE):
        uid = f"UE_{zn}_{i+1:02d}"
        nodes.append({
            "id": uid, "type": "UE",
            "initial_energy": 1.0, "coverage_area": zn,
        })
        ues_by_zone[zn].append(uid)
        # Attach to a gNB (round-robin)
        gnb = zone_gnbs[i % len(zone_gnbs)]
        L(uid, gnb, 50, 1, "microwave")

    for i in range(RESCUE_PER_ZONE):
        uid = f"RescueUE_{zn}_{i+1}"
        nodes.append({
            "id": uid, "type": "UE",
            "initial_energy": 1.0,
            "coverage_area": zn,
            "is_rescue_service": True,
        })
        ues_by_zone[zn].append(uid)
        gnb = zone_gnbs[i % len(zone_gnbs)]
        L(uid, gnb, 80, 1, "microwave")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"Nodes: {len(nodes)}")
print(f"Links: {len(links)}")

topology = {"nodes": nodes, "links": links}

import os, pathlib
cfg_dir = pathlib.Path(__file__).parent
with open(cfg_dir / "topology_large.yaml", "w") as f:
    yaml.dump(topology, f, default_flow_style=False, allow_unicode=True)

print(f"Written -> {cfg_dir / 'topology_large.yaml'}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate disaster scenario
# ─────────────────────────────────────────────────────────────────────────────

# Build traffic profiles for every gNB and relay
traffic_profiles = {}
for zn in ZONE_NAMES:
    for gid in gnbs_by_zone[zn]:
        traffic_profiles[gid] = {
            "life_safety":  {"baseline_rate": 8.0,  "burst_probability": 0.05, "burst_multiplier": 2.0},
            "operations":   {"baseline_rate": 15.0, "burst_probability": 0.10, "burst_multiplier": 2.0},
            "telemetry":    {"baseline_rate": 20.0, "burst_probability": 0.12, "burst_multiplier": 1.8},
            "best_effort":  {"baseline_rate": 30.0, "burst_probability": 0.20, "burst_multiplier": 2.5},
        }
    for rid in relays_by_zone[zn]:
        traffic_profiles[rid] = {
            "life_safety":  {"baseline_rate": 2.0,  "burst_probability": 0.03, "burst_multiplier": 1.5},
            "operations":   {"baseline_rate": 5.0,  "burst_probability": 0.05, "burst_multiplier": 1.5},
            "telemetry":    {"baseline_rate": 8.0,  "burst_probability": 0.08, "burst_multiplier": 1.3},
            "best_effort":  {"baseline_rate": 12.0, "burst_probability": 0.12, "burst_multiplier": 1.8},
        }
for zn in ("north", "east"):
    uid = edge_upfs[zn]
    traffic_profiles[uid] = {
        "life_safety":  {"baseline_rate": 5.0,  "burst_probability": 0.03, "burst_multiplier": 1.3},
        "operations":   {"baseline_rate": 12.0, "burst_probability": 0.07, "burst_multiplier": 1.6},
        "telemetry":    {"baseline_rate": 18.0, "burst_probability": 0.10, "burst_multiplier": 1.5},
        "best_effort":  {"baseline_rate": 25.0, "burst_probability": 0.15, "burst_multiplier": 2.0},
    }

# Disaster events
events = [
    # Tick 80: sever the core connection (island mode)
    {"tick": 80,  "type": "sever_core", "parameters": {}},

    # Tick 100: fail inter-zone links north→east (split into 2 islands)
    {"tick": 100, "type": "fail_link",
     "parameters": {"link_id": [l["id"] for l in links
                                if l["endpoints"][0] in gnbs_by_zone["north"]
                                and l["endpoints"][1] in gnbs_by_zone["east"]][0]}},

    # Tick 120: rescue UE surge in south zone
    {"tick": 120, "type": "traffic_surge",
     "parameters": {"node_id": gnbs_by_zone["south"][0], "duration": 60, "multiplier": 3.0}},

    # Tick 160: energy depletion in a north relay
    {"tick": 160, "type": "energy_depletion",
     "parameters": {"node_id": relays_by_zone["north"][0]}},

    # Tick 250: fail another inter-zone link west→south
    {"tick": 250, "type": "fail_link",
     "parameters": {"link_id": [l["id"] for l in links
                                if l["endpoints"][0] in gnbs_by_zone["south"]
                                and l["endpoints"][1] in gnbs_by_zone["west"]][0]}},

    # Tick 400: second traffic surge (rescue operations)
    {"tick": 400, "type": "traffic_surge",
     "parameters": {"node_id": gnbs_by_zone["north"][1], "duration": 80, "multiplier": 2.5}},
]

scenario = {
    "name": "Large-Scale Disaster Recovery",
    "description": (
        "Multi-zone catastrophic failure: core severance + inter-zone link failures "
        "creating 3-4 isolated network islands. MARL must form IAB chains and register "
        "UEs via IOPS to restore emergency communications."
    ),
    "duration_ticks": 800,
    "events": events,
    "traffic_profiles": traffic_profiles,
}

with open(cfg_dir / "scenario_disaster.yaml", "w") as f:
    yaml.dump(scenario, f, default_flow_style=False, allow_unicode=True)

print(f"Written -> {cfg_dir / 'scenario_disaster.yaml'}")
