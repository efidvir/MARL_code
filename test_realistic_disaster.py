"""
Test realistic disaster modeling with failed postcard transmissions.
"""

import sys
from pathlib import Path

# Set up paths
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("Testing realistic disaster modeling...")

try:
    from sixg_sim.simulation import Simulator, SimulationConfig
    from sixg_sim.topology import generate_large_topology, NodeType
    from sixg_sim.scenario import Scenario, ScenarioEvent
    from sixg_sim.traffic import NodeTrafficProfile, TrafficProfile, TrafficClass

    print("Generating topology...")
    topology = generate_large_topology(num_nodes=3, seed=42)  # Small but realistic topology

    # Create traffic profiles
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

    # Create scenario with early severance to see disaster effects
    print("Creating scenario...")
    events = [
        ScenarioEvent(tick=5, event_type="sever_core", parameters={}),
        ScenarioEvent(tick=15, event_type="traffic_surge",
                     parameters={"node_id": list(topology.nodes.keys())[0],
                               "duration": 20, "multiplier": 2.0}),
    ]

    scenario = Scenario(
        name="Realistic Disaster Test",
        duration_ticks=50,
        events=events,
        traffic_profiles=traffic_profiles
    )

    # Set up simulation with verbose output to see failures
    print("Setting up simulation...")
    config = SimulationConfig(
        tick_duration_ms=100,
        random_seed=42,
        enable_island_detection=True,
        verbose=True,
        sample_interval=5,  # Print every 5 ticks
        live_plot=False,
    )

    simulator = Simulator(topology, scenario, config)

    # Run simulation
    print("Running simulation (realistic disaster test)...")
    metrics = simulator.run_simulation()

    print("\n[SUCCESS] Simulation completed!")
    print(f"Simulated {len(metrics.metrics_history)} ticks")

    # Analyze failed transmissions
    print("\nFailed Transmission Analysis:")
    total_failed = 0
    failure_reasons = {}
    failure_priorities = {}

    for m in metrics.metrics_history:
        failed_transmissions = m.failed_postcard_transmissions
        total_failed += len(failed_transmissions)

        for failure in failed_transmissions:
            reason = failure.get('reason', 'unknown')
            priority = failure.get('priority', 'unknown')

            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            failure_priorities[priority] = failure_priorities.get(priority, 0) + 1

    print(f"Total failed transmissions: {total_failed}")
    print("Failure reasons:", failure_reasons)
    print("Failure priorities:", failure_priorities)

    # Show some example failures
    if total_failed > 0:
        print("\nExample failures:")
        for m in metrics.metrics_history[-5:]:  # Last 5 ticks
            for failure in m.failed_postcard_transmissions[-3:]:  # Last 3 failures per tick
                print(f"  Tick {m.tick}: {failure.get('source')} -> {failure.get('target')} "
                      f"(reason: {failure.get('reason')}, priority: {failure.get('priority')})")

    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()