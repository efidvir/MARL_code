"""
Test topology generation only.
"""

import sys
from pathlib import Path

# Set up paths
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("Testing topology generation...")

try:
    from sixg_sim.topology import generate_large_topology, NodeType

    print("Generating topology...")
    topology = generate_large_topology(num_nodes=20, seed=42)  # Small test

    # Count node types
    node_type_counts = {}
    for node in topology.nodes.values():
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1

    print(f"Generated {len(topology.nodes)} nodes and {len(topology.links)} links:")
    for node_type, count in node_type_counts.items():
        print(f"  {node_type.value}: {count}")

    # Check UE connectivity
    ue_nodes = [node_id for node_id, node in topology.nodes.items() if node.node_type == NodeType.UE]
    infrastructure_nodes = [node_id for node_id, node in topology.nodes.items()
                           if node.node_type != NodeType.UE]

    print(f"\nUE Connectivity Check:")
    print(f"  UEs: {len(ue_nodes)}")
    print(f"  Infrastructure: {len(infrastructure_nodes)}")

    # Count UE connections
    ue_connection_count = 0
    for ue_id in ue_nodes:
        ue_links = [link for link in topology.links.values()
                   if ue_id in link.endpoints]
        ue_connection_count += len(ue_links)
        if len(ue_links) == 0:
            print(f"  WARNING: UE {ue_id} has no connections!")

    avg_connections = ue_connection_count / len(ue_nodes) if ue_nodes else 0
    print(f"  Total UE connections: {ue_connection_count}")
    print(f"  Average connections per UE: {avg_connections:.1f}")

    print("\nTopology generation successful!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
