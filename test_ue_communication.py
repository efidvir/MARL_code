"""
Simple test for UE-to-UE communication in simulation.
"""

import sys
from pathlib import Path

# Set up paths
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("Testing UE-to-UE communication...")

try:
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.topology import generate_large_topology, NodeType
    from sixg_sim.scenario import Scenario, ScenarioEvent
    from sixg_sim.traffic import NodeTrafficProfile, TrafficProfile, TrafficClass

    print("Generating small topology...")
    topology = generate_large_topology(num_nodes=5, seed=42)  # Very small test

    # Count node types
    node_type_counts = {}
    for node in topology.nodes.values():
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1

    print(f"Generated topology with {len(topology.nodes)} nodes and {len(topology.links)} links:")
    for node_type, count in node_type_counts.items():
        print(f"  {node_type.value}: {count} nodes")

    # Create minimal traffic profiles
    print("Creating traffic profiles...")
    traffic_profiles = {}
    default_oran_profile = {
        TrafficClass.LIFE_SAFETY: TrafficProfile(baseline_rate=5.0, burst_probability=0.02, burst_multiplier=1.5),
        TrafficClass.OPERATIONS: TrafficProfile(baseline_rate=15.0, burst_probability=0.05, burst_multiplier=2.0),
        TrafficClass.TELEMETRY: TrafficProfile(baseline_rate=20.0, burst_probability=0.08, burst_multiplier=1.8),
        TrafficClass.BEST_EFFORT: TrafficProfile(baseline_rate=20.0, burst_probability=0.12, burst_multiplier=2.2),
    }

    default_profiles = {
        NodeType.O_RU: default_oran_profile,
        NodeType.O_DU: default_oran_profile,
        NodeType.O_CU_CP: default_oran_profile,
        NodeType.O_CU_UP: default_oran_profile,
        NodeType.NEAR_RT_RIC: default_oran_profile,
        NodeType.SMO: default_oran_profile,
        NodeType.UPF: default_oran_profile,
        NodeType.AMF: default_oran_profile,
        NodeType.RELAY: default_oran_profile,
        NodeType.EDGEUPF: default_oran_profile,
        NodeType.UE: {
            TrafficClass.LIFE_SAFETY: TrafficProfile(baseline_rate=2.0, burst_probability=0.05, burst_multiplier=3.0),
            TrafficClass.OPERATIONS: TrafficProfile(baseline_rate=5.0, burst_probability=0.1, burst_multiplier=2.0),
            TrafficClass.TELEMETRY: TrafficProfile(baseline_rate=8.0, burst_probability=0.15, burst_multiplier=1.5),
            TrafficClass.BEST_EFFORT: TrafficProfile(baseline_rate=25.0, burst_probability=0.3, burst_multiplier=4.0),
        },
    }

    # Create profiles for all nodes
    for node_id, node in topology.nodes.items():
        profiles = default_profiles.get(node.node_type, default_profiles[NodeType.RELAY])
        traffic_profiles[node_id] = NodeTrafficProfile(node_id=node_id, profiles=profiles)

    # Create scenario with severance
    print("Creating scenario...")
    events = [
        ScenarioEvent(tick=50, event_type="sever_core", parameters={}),
    ]

    scenario = Scenario(
        name="UE Communication Test",
        duration_ticks=100,
        events=events,
        traffic_profiles=traffic_profiles
    )

    # Set up simulation
    print("Setting up simulation...")
    config = SimulationConfig(
        tick_duration_ms=100,
        random_seed=42,
        enable_island_detection=True,
        verbose=True,
        sample_interval=10,  # Print every 10 ticks
        live_plot=False,  # Disable plotting for test
    )

    simulator = Simulator(topology, scenario, config)

    # Run simulation
    print("Running simulation (short test)...")
    metrics = simulator.run_simulation()

    print("\n[SUCCESS] Simulation completed!")
    print(f"Simulated {len(metrics.metrics_history)} ticks")

    # Analyze UE-to-UE communication
    print("\nUE-to-UE Communication Analysis:")
    if metrics.metrics_history:
        for m in metrics.metrics_history[-10:]:  # Last 10 ticks
            ue2ue = m.ue_to_ue_stats
            enabled = ue2ue.get('enabled', False)
            marl_enabled = ue2ue.get('marl_routing_enabled', False)
            success = ue2ue.get('success_count', 0)
            total = ue2ue.get('total_flows', 0)
            success_rate = ue2ue.get('success_rate', 0.0) * 100

            status = "MARL-ENABLED" if marl_enabled else ("CORE-DEPENDENT" if enabled else "DISABLED")
            print(f"  t={m.tick}: {status} - {success}/{total} flows ({success_rate:.1f}%)")

    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()