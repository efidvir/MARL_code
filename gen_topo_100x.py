"""Generate a 100x scaled O-RAN topology (~1280 nodes) for IOPS training."""
import yaml

nodes = []
links = []
link_id = 0

# ── Core layer (2 core nodes) ────────────────────────────────
nodes.append({"id": "core1", "type": "Core", "initial_energy": 1.0})
nodes.append({"id": "core2", "type": "Core", "initial_energy": 1.0})
link_id += 1
links.append({"id": f"L{link_id}", "endpoints": ["core1", "core2"],
              "capacity": 5000, "latency": 1, "type": "fiber"})

# ── Edge UPF layer (8 edge UPFs, one per zone) ──────────────
zone_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
for zi, zone in enumerate(zone_names):
    eid = f"edgeUPF_{zone}"
    nodes.append({"id": eid, "type": "EdgeUPF", "initial_energy": 1.0,
                  "coverage_area": f"zone{zone}"})
    # Redundant uplinks to both cores
    link_id += 1
    links.append({"id": f"L{link_id}", "endpoints": [eid, "core1"],
                  "capacity": 2000, "latency": 3, "type": "fiber"})
    link_id += 1
    links.append({"id": f"L{link_id}", "endpoints": [eid, "core2"],
                  "capacity": 1500, "latency": 4, "type": "fiber"})

# ── Per-zone infrastructure ──────────────────────────────────
GNB_PER_ZONE = 10
RELAY_PER_ZONE = 5
UE_PER_GNB = 13
RESCUE_PER_ZONE = 4

for zi, zone in enumerate(zone_names):
    edge = f"edgeUPF_{zone}"

    # gNB-Sites
    gnb_ids = []
    for g in range(1, GNB_PER_ZONE + 1):
        gnb_id = f"gNB_{zone}{g}"
        gnb_ids.append(gnb_id)
        nodes.append({"id": gnb_id, "type": "gNB-Site", "initial_energy": 1.0,
                      "coverage_area": f"zone{zone}"})
        # Backhaul to zone edge UPF
        link_id += 1
        links.append({"id": f"L{link_id}", "endpoints": [gnb_id, edge],
                      "capacity": 500, "latency": 2, "type": "fiber"})

    # Inter-gNB Xn links: ring topology + skip-1 cross links
    for g in range(len(gnb_ids)):
        next_g = (g + 1) % len(gnb_ids)
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [gnb_ids[g], gnb_ids[next_g]],
                      "capacity": 200, "latency": 1, "type": "iab"})
        # Cross link (skip-1 for denser mesh)
        skip = (g + 2) % len(gnb_ids)
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [gnb_ids[g], gnb_ids[skip]],
                      "capacity": 150, "latency": 2, "type": "microwave"})

    # Relay nodes
    for r in range(1, RELAY_PER_ZONE + 1):
        relay_id = f"relay_{zone}{r}"
        nodes.append({"id": relay_id, "type": "Relay", "initial_energy": 0.8,
                      "coverage_area": f"zone{zone}"})
        # Connect to 2 nearby gNBs
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [relay_id, gnb_ids[(r * 2 - 2) % len(gnb_ids)]],
                      "capacity": 300, "latency": 2, "type": "microwave"})
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [relay_id, gnb_ids[(r * 2 - 1) % len(gnb_ids)]],
                      "capacity": 250, "latency": 3, "type": "microwave"})

    # UEs per gNB
    ue_count = 0
    for g, gnb_id in enumerate(gnb_ids):
        for u in range(1, UE_PER_GNB + 1):
            ue_count += 1
            ue_id = f"UE_{zone}{ue_count}"
            nodes.append({"id": ue_id, "type": "UE", "initial_energy": 1.0,
                          "coverage_area": f"zone{zone}"})
            link_id += 1
            links.append({"id": f"L{link_id}", "endpoints": [ue_id, gnb_id],
                          "capacity": 50, "latency": 1, "type": "microwave"})

    # Rescue UEs
    for r in range(1, RESCUE_PER_ZONE + 1):
        rescue_id = f"Rescue_{zone}{r}"
        nodes.append({"id": rescue_id, "type": "UE", "initial_energy": 1.0,
                      "coverage_area": f"zone{zone}",
                      "is_rescue_service": True})
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [rescue_id, gnb_ids[r % len(gnb_ids)]],
                      "capacity": 80, "latency": 1, "type": "microwave"})

# ── Inter-zone links (connecting adjacent zones in octagon) ──
for zi in range(len(zone_names)):
    next_zi = (zi + 1) % len(zone_names)
    link_id += 1
    links.append({"id": f"L{link_id}",
                  "endpoints": [f"gNB_{zone_names[zi]}1",
                                f"gNB_{zone_names[next_zi]}1"],
                  "capacity": 300, "latency": 5, "type": "microwave"})
    # Second inter-zone link for redundancy
    link_id += 1
    links.append({"id": f"L{link_id}",
                  "endpoints": [f"gNB_{zone_names[zi]}5",
                                f"gNB_{zone_names[next_zi]}5"],
                  "capacity": 250, "latency": 6, "type": "microwave"})

# ── Write topology ───────────────────────────────────────────
topo = {"nodes": nodes, "links": links}
with open("config/topology_100x.yaml", "w") as f:
    yaml.dump(topo, f, default_flow_style=False, sort_keys=False)

print(f"Topology generated: {len(nodes)} nodes, {len(links)} links")
ntype = {}
for n in nodes:
    t = n["type"]
    if n.get("is_rescue_service"):
        t = "UE (Rescue)"
    ntype[t] = ntype.get(t, 0) + 1
for t, c in sorted(ntype.items()):
    print(f"  {t:15s}: {c}")
