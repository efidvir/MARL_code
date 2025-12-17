"""
Main simulation engine for 6G network simulation.

Coordinates topology, agents, traffic, and control plane through discrete time steps.
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from .topology import Topology, Node, Link, TrafficClass
from .traffic import TrafficGenerator, SliceDictionary
from .agent import BaseAgent, AgentObservation, EnergyTier, StrainLevel, create_agent_for_node, LocalSliceState
from .control_plane import ControlPlaneManager
from .scenario import Scenario, ScenarioEvent
from .metrics import MetricsCollector, TickMetrics


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    tick_duration_ms: int = 100  # Duration of each tick in milliseconds
    random_seed: Optional[int] = None
    enable_island_detection: bool = True
    control_message_limit: int = 10  # Max control bytes per tick
    verbose: bool = False  # Enable verbose logging
    sample_interval: int = 100  # How often to print progress when verbose


class Simulator:
    """Main simulation coordinator."""

    def __init__(self, topology: Topology, scenario: Scenario, config: SimulationConfig):
        self.topology = topology
        self.scenario = scenario
        self.config = config

        # Initialize components
        self.slice_dictionary = SliceDictionary()
        self.traffic_generator = TrafficGenerator(seed=config.random_seed)
        self.control_plane = ControlPlaneManager(topology)
        self.metrics = MetricsCollector()

        # Verbose/debug state
        self.last_postcards_sent: int = 0
        self.last_action_summary = {}

        # Create agents for each node
        self.agents: Dict[str, BaseAgent] = {}
        for node_id, node in topology.nodes.items():
            if node.is_survivor:  # Only create agents for survivor nodes
                self.agents[node_id] = create_agent_for_node(
                    node_id, node.node_type.value, self.slice_dictionary
                )

        # Set up traffic generator
        for profile in scenario.traffic_profiles.values():
            self.traffic_generator.add_node_profile(profile)

        # Simulation state
        self.current_tick = 0
        self.island_mode = False
        self.core_nodes = self._identify_core_nodes()

    def _identify_core_nodes(self) -> Set[str]:
        """Identify nodes that are part of the core network."""
        return {node_id for node_id, node in self.topology.nodes.items()
                if node.node_type.value == 'Core'}

    def _detect_island_mode(self) -> bool:
        """Check if network should enter island mode."""
        if not self.config.enable_island_detection:
            return False

        # Island mode if no surviving NON-core node can reach any core node
        survivor_nodes = [
            node_id for node_id, node in self.topology.nodes.items()
            if node.is_survivor and node.node_type.value != "Core"
        ]

        for survivor in survivor_nodes:
            if any(self.topology.has_path(survivor, core_node) for core_node in self.core_nodes):
                return False  # At least one path exists

        return True

    def _process_events(self, tick: int):
        """Process scenario events for current tick."""
        events = self.scenario.get_events_at_tick(tick)

        for event in events:
            self._execute_event(event)

    def _execute_event(self, event: ScenarioEvent):
        """Execute a scenario event."""
        if event.event_type == 'sever_core':
            # Cut all links to/from core nodes
            for link in self.topology.links.values():
                if (link.endpoints[0] in self.core_nodes or
                    link.endpoints[1] in self.core_nodes):
                    link.is_up = False
            print(f"Tick {event.tick}: Core severed - entering island mode")

        elif event.event_type == 'fail_link':
            link_id = event.parameters.get('link_id')
            if link_id in self.topology.links:
                self.topology.links[link_id].is_up = False
                print(f"Tick {event.tick}: Link {link_id} failed")
            else:
                print(f"Tick {event.tick}: fail_link ignored (link {link_id} not found)")

        elif event.event_type == 'restore_link':
            link_id = event.parameters.get('link_id')
            if link_id in self.topology.links:
                self.topology.links[link_id].is_up = True
                print(f"Tick {event.tick}: Link {link_id} restored")
            else:
                print(f"Tick {event.tick}: restore_link ignored (link {link_id} not found)")

        elif event.event_type == 'energy_depletion':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self.topology.nodes[node_id].energy_soc = 0.0
                self.topology.nodes[node_id].is_survivor = False
                print(f"Tick {event.tick}: Node {node_id} depleted")
            else:
                print(f"Tick {event.tick}: energy_depletion ignored (node {node_id} not found)")

        elif event.event_type == 'node_failure':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self.topology.nodes[node_id].is_survivor = False
                print(f"Tick {event.tick}: Node {node_id} failed")
            else:
                print(f"Tick {event.tick}: node_failure ignored (node {node_id} not found)")

        elif event.event_type == 'node_recovery':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self.topology.nodes[node_id].is_survivor = True
                print(f"Tick {event.tick}: Node {node_id} recovered")
            else:
                print(f"Tick {event.tick}: node_recovery ignored (node {node_id} not found)")

        # Update topology graph after link changes
        self.topology.update_link_statuses()

    def _generate_traffic(self) -> Dict[str, Dict[TrafficClass, float]]:
        """Generate traffic arrivals for current tick."""
        return self.traffic_generator.generate_traffic(self.current_tick)

    def _build_agent_observations(self) -> Dict[str, AgentObservation]:
        """Build observations for all agents."""
        observations = {}

        # Get control messages for this tick
        control_messages = {}
        for node_id in self.agents.keys():
            control_messages[node_id] = self.control_plane.get_received_postcards(
                node_id, self.current_tick
            )

        for node_id, agent in self.agents.items():
            node = self.topology.nodes[node_id]

            # Build local slice states
            local_slices = {}
            for traffic_class in TrafficClass:
                queue = node.queues[traffic_class]
                local_slices[traffic_class] = LocalSliceState(
                    traffic_class=traffic_class,
                    importance=self.slice_dictionary.get_importance_score(traffic_class),
                    current_queue_length=queue.queued_load,
                    offered_load=queue.offered_load,
                    admission_success_rate=0.9,  # Simplified
                    freshness_target=self.slice_dictionary.qos_profiles[traffic_class].target_delay,
                    current_freshness=5.0  # Simplified delay proxy
                )

            # Get energy tier
            energy_tier = EnergyTier(node.get_energy_tier())

            # Fuse neighbor summaries
            neighbor_summary = self.control_plane.fuse_neighbor_summaries(
                node_id, control_messages.get(node_id, []), self.current_tick
            )

            observations[node_id] = AgentObservation(
                node_id=node_id,
                is_island=node.is_island,
                energy_tier=energy_tier,
                local_slices=local_slices,
                neighbor_summary=neighbor_summary,
                current_tick=self.current_tick
            )

        return observations

    def _execute_agent_actions(self, observations: Dict[str, AgentObservation]):
        """Execute actions from all agents."""
        # Reset tick-local summaries
        self.last_postcards_sent = 0
        self.last_action_summary = {}

        for node_id, agent in self.agents.items():
            observation = observations[node_id]
            action = agent.compute_action(observation)

            # Send postcard if requested
            if action.send_postcard and action.postcard_content:
                self.control_plane.send_postcard(action.postcard_content, self.current_tick)
                self.last_postcards_sent += 1

            # Apply class actions (would affect admission control)
            # For now, just store for metrics
            for tclass, caction in action.class_actions.items():
                key = (tclass.value, caction.admission_mode.value)
                self.last_action_summary[key] = self.last_action_summary.get(key, 0) + 1

    def _forward_traffic(self, traffic_arrivals: Dict[str, Dict[TrafficClass, float]]):
        """Forward traffic through the network."""
        # Simplified forwarding: admit all traffic, forward to neighbors
        for node_id, arrivals in traffic_arrivals.items():
            if node_id not in self.topology.nodes:
                continue

            node = self.topology.nodes[node_id]
            total_traffic = sum(arrivals.values())

            # Update node queues
            for traffic_class, amount in arrivals.items():
                queue = node.queues[traffic_class]
                queue.offered_load = amount
                queue.admitted_load = amount  # Simplified: admit all
                queue.queued_load += amount
                queue.delivered_load = amount  # Simplified: deliver all

            # Update energy consumption
            control_bytes = 0  # Would track actual control traffic
            node.update_energy(total_traffic, control_bytes)

            # Reset queues for next tick
            node.reset_queues()

        # Reset link utilization
        for link in self.topology.links.values():
            link.reset_utilization()

    def _collect_metrics(self):
        """Collect metrics for current tick."""
        # Node states
        node_states = {}
        for node_id, node in self.topology.nodes.items():
            node_states[node_id] = {
                'is_survivor': node.is_survivor,
                'is_island': node.is_island,
                'energy_soc': node.energy_soc,
                'energy_tier': node.get_energy_tier()
            }

        # Link states
        link_states = {}
        for link_id, link in self.topology.links.items():
            link_states[link_id] = {
                'is_up': link.is_up,
                'utilization': link.current_utilization,
                'capacity': link.capacity
            }

        # Traffic stats (simplified)
        traffic_stats = {}
        for node_id, node in self.topology.nodes.items():
            node_traffic = {}
            for traffic_class in TrafficClass:
                queue = node.queues[traffic_class]
                node_traffic[traffic_class] = {
                    'offered_load': queue.offered_load,
                    'admitted_load': queue.admitted_load,
                    'queued_load': queue.queued_load,
                    'delivered_load': queue.delivered_load,
                    'dropped_load': queue.dropped_load
                }
            traffic_stats[node_id] = node_traffic

        # Control stats
        control_stats = {}
        for node_id in self.agents.keys():
            control_stats[node_id] = {
                'postcards_sent': 0,  # Placeholder (not yet tracked per-node)
                'postcards_received': len(self.control_plane.get_received_postcards(
                    node_id, self.current_tick))
            }

        self.metrics.record_tick(
            self.current_tick, node_states, link_states,
            traffic_stats, control_stats, self.island_mode
        )

    def _print_debug_summary(self, tick: int):
        """Print a concise summary of the current state for debugging/insight."""
        infra_survivors = sum(
            1 for n in self.topology.nodes.values()
            if n.node_type != n.node_type.UE and n.is_survivor
        )
        ue_survivors = sum(
            1 for n in self.topology.nodes.values()
            if n.node_type == n.node_type.UE and n.is_survivor
        )
        print(f"[DBG] t={tick} island={self.island_mode} "
              f"infra_alive={infra_survivors} ue_alive={ue_survivors} "
              f"postcards_sent={self.last_postcards_sent}")
        if self.last_action_summary:
            top_actions = list(self.last_action_summary.items())[:8]
            actions_str = ", ".join([f"{k[0]}:{k[1]}={v}" for k, v in top_actions])
            print(f"[DBG] recent_actions: {actions_str}")
        # Reset tick-local summaries
        self.last_postcards_sent = 0
        self.last_action_summary = {}

    def run_simulation(self) -> MetricsCollector:
        """Run the complete simulation."""
        print(f"Starting simulation for {self.scenario.duration_ticks} ticks")

        for tick in range(self.scenario.duration_ticks):
            self.current_tick = tick

            # Progress reporting for long simulations
            if tick % 100 == 0 and tick > 0:
                print(f"Completed tick {tick}/{self.scenario.duration_ticks}")

            # Process scenario events
            self._process_events(tick)

            # Check for island mode transition
            was_island = self.island_mode
            self.island_mode = self._detect_island_mode()
            self.control_plane.set_island_mode(self.island_mode)

            if self.island_mode and not was_island:
                print(f"Tick {tick}: Entering island mode")
                self.metrics.set_severance_tick(tick)
                # Mark island nodes
                for node in self.topology.nodes.values():
                    if node.is_survivor:
                        node.is_island = True
            elif (not self.island_mode) and was_island:
                print(f"Tick {tick}: Exiting island mode (core reachability restored)")

            # Generate and forward traffic
            traffic_arrivals = self._generate_traffic()
            self._forward_traffic(traffic_arrivals)

            # Build agent observations and execute actions
            observations = self._build_agent_observations()
            self._execute_agent_actions(observations)

            # Collect metrics
            self._collect_metrics()

            # Verbose progress snapshots
            if self.config.verbose and (tick % self.config.sample_interval == 0 or self.island_mode != was_island):
                self._print_debug_summary(tick)

            # Reset control plane for next tick
            self.control_plane.reset_for_tick()

            if tick % 50 == 0:
                print(f"Completed tick {tick}/{self.scenario.duration_ticks}")

        print("Simulation completed")
        return self.metrics


def run_simulation(topology_file: str, scenario_file: str,
                  output_dir: str = "output") -> MetricsCollector:
    """Convenience function to run a complete simulation."""
    print(f"Loading topology from {topology_file}...")
    from .topology import load_topology_from_yaml
    topology = load_topology_from_yaml(topology_file)
    print(f"Loaded {len(topology.nodes)} nodes, {len(topology.links)} links")

    print(f"Loading scenario from {scenario_file}...")
    from .scenario import load_scenario_from_yaml
    scenario = load_scenario_from_yaml(scenario_file, topology)
    print(f"Loaded scenario '{scenario.name}' with {len(scenario.events)} events")

    config = SimulationConfig(
        tick_duration_ms=100,
        random_seed=42,
        enable_island_detection=True
    )

    # Run simulation
    print("Initializing simulator...")
    simulator = Simulator(topology, scenario, config)
    print("Running simulation...")
    metrics = simulator.run_simulation()
    print(f"Simulation completed with {len(metrics.metrics_history)} ticks")

    # Export results
    print(f"Exporting results to {output_dir}...")
    import os
    os.makedirs(output_dir, exist_ok=True)
    metrics.export_to_csv(output_dir)
    print("Export completed.")

    return metrics
