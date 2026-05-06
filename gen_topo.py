"""Generate a 10x scaled O-RAN topology for IOPS training."""
import yaml

nodes = []
links = []
link_id = 0

# Core layer (2 core nodes)
nodes.append({"id": "core1", "type": "Core", "initial_energy": 1.0})
nodes.append({"id": "core2", "type": "Core", "initial_energy": 1.0})
link_id += 1
links.append({"id": f"L{link_id}", "endpoints": ["core1", "core2"],
              "capacity": 2000, "latency": 1, "type": "fiber"})

# Edge UPF layer (3 edge UPFs)
for e in range(1, 4):
    eid = f"edgeUPF{e}"
    zone_letter = chr(64 + e)  # A, B, C
    nodes.append({"id": eid, "type": "EdgeUPF", "initial_energy": 1.0,
                  "coverage_area": f"zone{zone_letter}"})
    link_id += 1
    links.append({"id": f"L{link_id}", "endpoints": [eid, "core1"],
                  "capacity": 1000, "latency": 3, "type": "fiber"})
    link_id += 1
    links.append({"id": f"L{link_id}", "endpoints": [eid, "core2"],
                  "capacity": 800, "latency": 4, "type": "fiber"})

# Per-zone: gNBs, relays, UEs
zones = ["A", "B", "C"]
gnb_per_zone = 6
relay_per_zone = 3
ue_per_gnb = 5
rescue_per_zone = 2

for zi, zone in enumerate(zones):
    edge = f"edgeUPF{zi + 1}"

    gnb_ids = []
    for g in range(1, gnb_per_zone + 1):
        gnb_id = f"gNB_{zone}{g}"
        gnb_ids.append(gnb_id)
        nodes.append({"id": gnb_id, "type": "gNB-Site", "initial_energy": 1.0,
                      "coverage_area": f"zone{zone}"})
        # Connect to edge UPF
        link_id += 1
        links.append({"id": f"L{link_id}", "endpoints": [gnb_id, edge],
                      "capacity": 500, "latency": 2, "type": "fiber"})

    # Inter-gNB Xn links (ring + cross links for mesh)
    for g in range(len(gnb_ids)):
        # Ring
        next_g = (g + 1) % len(gnb_ids)
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [gnb_ids[g], gnb_ids[next_g]],
                      "capacity": 200, "latency": 1, "type": "iab"})
        # Cross link (skip-1)
        if len(gnb_ids) > 3:
            skip = (g + 2) % len(gnb_ids)
            link_id += 1
            links.append({"id": f"L{link_id}",
                          "endpoints": [gnb_ids[g], gnb_ids[skip]],
                          "capacity": 150, "latency": 2, "type": "microwave"})

    # Relays
    for r in range(1, relay_per_zone + 1):
        relay_id = f"relay_{zone}{r}"
        nodes.append({"id": relay_id, "type": "Relay", "initial_energy": 0.8,
                      "coverage_area": f"zone{zone}"})
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [relay_id, gnb_ids[(r - 1) * 2 % len(gnb_ids)]],
                      "capacity": 300, "latency": 2, "type": "microwave"})
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [relay_id, gnb_ids[(r * 2 - 1) % len(gnb_ids)]],
                      "capacity": 250, "latency": 3, "type": "microwave"})

    # UEs per gNB
    ue_count = 0
    for g, gnb_id in enumerate(gnb_ids):
        for u in range(1, ue_per_gnb + 1):
            ue_count += 1
            ue_id = f"UE_{zone}{ue_count}"
            nodes.append({"id": ue_id, "type": "UE", "initial_energy": 1.0,
                          "coverage_area": f"zone{zone}"})
            link_id += 1
            links.append({"id": f"L{link_id}", "endpoints": [ue_id, gnb_id],
                          "capacity": 50, "latency": 1, "type": "microwave"})

    # Rescue UEs
    for r in range(1, rescue_per_zone + 1):
        rescue_id = f"Rescue_{zone}{r}"
        nodes.append({"id": rescue_id, "type": "UE", "initial_energy": 1.0,
                      "coverage_area": f"zone{zone}",
                      "is_rescue_service": True})
        link_id += 1
        links.append({"id": f"L{link_id}",
                      "endpoints": [rescue_id, gnb_ids[r % len(gnb_ids)]],
                      "capacity": 80, "latency": 1, "type": "microwave"})

# Inter-zone links (connecting zones)
for zi in range(len(zones)):
    next_zi = (zi + 1) % len(zones)
    link_id += 1
    links.append({"id": f"L{link_id}",
                  "endpoints": [f"gNB_{zones[zi]}1", f"gNB_{zones[next_zi]}1"],
                  "capacity": 300, "latency": 5, "type": "microwave"})

topo = {"nodes": nodes, "links": links}
with open("config/topology_10x.yaml", "w") as f:
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
