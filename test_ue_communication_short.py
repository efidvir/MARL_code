"""
Very short test for UE-to-UE communication in simulation.
"""

import sys
from pathlib import Path

# Set up paths
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("Testing UE-to-UE communication (short test)...")

try:
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.topology import generate_large_topology, NodeType
    from sixg_sim.scenario import Scenario, ScenarioEvent
    from sixg_sim.traffic import NodeTrafficProfile, TrafficProfile, TrafficClass

    print("Generating tiny topology...")
    topology = generate_large_topology(num_nodes=2, seed=42)  # Very tiny test

    # Create minimal traffic profiles
    print("Creating traffic profiles...")
    traffic_profiles = {}
    default_oran_profile = {
        TrafficClass.LIFE_SAFETY: TrafficProfile(baseline_rate=5.0, burst_probability=0.02, burst_multiplier=1.5),
        TrafficClass.OPERATIONS: TrafficProfile(baseline_rate=15.0, burst_probability=0.05, burst_multiplier=2.0),
        TrafficClass.TELEMETRY: TrafficProfile(baseline_rate=20.0, burst_probability=0.08, burst_multiplier=1.8),
        TrafficClass.BEST_EFFORT: TrafficProfile(baseline_rate=20.0, burst_probability=0.12, burst_multiplier=2.2),
    }

    for node_id, node in topology.nodes.items():
        profiles = default_oran_profile
        traffic_profiles[node_id] = NodeTrafficProfile(node_id=node_id, profiles=profiles)

    # Create scenario with severance at tick 3
    print("Creating scenario...")
    events = [
        ScenarioEvent(tick=3, event_type="sever_core", parameters={}),
    ]

    scenario = Scenario(
        name="UE Communication Short Test",
        duration_ticks=10,
        events=events,
        traffic_profiles=traffic_profiles
    )

    # Set up simulation with verbose output
    print("Setting up simulation...")
    config = SimulationConfig(
        tick_duration_ms=100,
        random_seed=42,
        enable_island_detection=True,
        verbose=True,
        sample_interval=1,  # Print every tick
        live_plot=False,
    )

    simulator = Simulator(topology, scenario, config)

    # Run simulation
    print("Running simulation (very short test)...")
    metrics = simulator.run_simulation()

    print("\n[SUCCESS] Simulation completed!")
    print(f"Simulated {len(metrics.metrics_history)} ticks")

    # Show UE-to-UE communication analysis
    print("\nUE-to-UE Communication Analysis:")
    for m in metrics.metrics_history:
        ue2ue = m.ue_to_ue_stats
        enabled = ue2ue.get('enabled', False)
        marl_enabled = ue2ue.get('marl_routing_enabled', False)
        success = ue2ue.get('success_count', 0)
        total = ue2ue.get('total_flows', 0)
        success_rate = ue2ue.get('success_rate', 0.0) * 100

        if marl_enabled:
            status = "MARL-ENABLED"
        elif enabled:
            status = "CORE-DEPENDENT"
        else:
            status = "DISABLED"

        print(f"  t={m.tick}: {status} - {success}/{total} flows ({success_rate:.1f}%)")

    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()