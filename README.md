# 6G Network Simulation - Island Mode Proof of Concept

A discrete-event simulation of a 5G/early-6G RAN + transport network that demonstrates autonomous island-mode operation after core network severance. The simulation implements multi-agent cooperative recovery using semantic control overlays and constrained communication.

## Overview

This proof-of-concept implements:

- **Network Topology**: Graph-based model with nodes (gNB sites, relays, edge UPFs, core) and links (fiber, microwave, IAB)
- **Traffic Classes**: Four QoS classes (life-safety, operations, telemetry, best-effort) with semantic slice policies
- **Multi-Agent System**: Autonomous agents at each survivor node making local decisions
- **Island Mode**: Automatic detection and transition to decentralized operation when core connectivity is lost
- **Control Overlays**: IP overlay (normal mode) and Disaster Control Channel (DCC) for constrained communication
- **Recovery Metrics**: Quantitative analysis of survival time, energy usage, connectivity, and traffic delivery

## Architecture

```
sixg_sim/
├── topology.py      # Network graph, nodes, links
├── traffic.py       # Traffic generation, QoS profiles
├── agent.py         # HeuristicAgent with observation/action interfaces
├── control_plane.py # IP overlay and DCC implementations
├── simulation.py    # Main discrete-event engine
├── scenario.py      # Event definitions and configurations
├── metrics.py       # KPI collection and analysis
├── analysis.py      # Plotting and reporting functions
└── main.py          # CLI entry point
```

## Installation

```bash
# Install dependencies
pip install networkx numpy pandas matplotlib pyyaml

# Clone/download the code
cd sixg_sim/
```

## Quick Start

Run a basic simulation with severance and recovery:

```bash
python -m sixg_sim.main --topology config/topology_example.yaml --scenario config/scenario_severance.yaml --output-dir results/
```

This will:
1. Load a 5-node topology (2 gNBs, relay, edge UPF, core)
2. Run 300 ticks of simulation
3. Trigger core severance at tick 50
4. Generate traffic surges and link failures
5. Export metrics and generate analysis plots

## Configuration Files

### Topology Configuration

Define network nodes and links in YAML:

```yaml
nodes:
  - id: "gNB1"
    type: "GNBSite"
    initial_energy: 1.0
    coverage_area: "zoneA"

links:
  - id: "L1"
    endpoints: ["gNB1", "core"]
    capacity: 1000
    latency: 5
    type: "fiber"
```

**Node Types**: GNBSite, Relay, EdgeUPF, Core, SatelliteGateway, FieldGateway

**Link Types**: fiber, microwave, iab, d2d, satellite

### Scenario Configuration

Define simulation events and traffic patterns:

```yaml
name: "Island Mode Test"
duration_ticks: 300

events:
  - tick: 50
    type: "sever_core"
    parameters: {}
  - tick: 100
    type: "traffic_surge"
    parameters:
      node_id: "gNB1"
      duration: 30
      multiplier: 2.5

traffic_profiles:
  gNB1:
    life_safety:
      baseline_rate: 8.0
      burst_probability: 0.05
```

**Event Types**: sever_core, fail_link, restore_link, energy_depletion, traffic_surge, node_failure, node_recovery

## Key Concepts

### Island Mode Detection

The simulation automatically detects when survivor nodes lose connectivity to core/central controllers and switches to island mode:

- Agents prioritize life-safety traffic
- Control communication switches to constrained DCC
- Routing biases toward reliable local paths
- Energy conservation becomes critical

### Agent Decision Making

Each survivor node runs a `HeuristicAgent` that:

1. **Observes**: Local state, energy level, neighbor summaries via postcards
2. **Decides**: Admission policies, routing biases, control message sending
3. **Acts**: Protects critical traffic, throttles non-essential services

### Control Overlays

- **IP Overlay**: Normal operation with full connectivity
- **DCC**: Island mode with 1 postcard/tick limits and tiny message sizes

### Traffic Classes & QoS

| Class | Priority | Preemption | Stress Policy |
|-------|----------|------------|---------------|
| Life Safety | 4 | Yes | Always Admit |
| Operations | 3 | No | Always Admit |
| Telemetry | 2 | No | Throttle |
| Best Effort | 1 | No | Hold |

## Output Analysis

The simulation generates:

### Metrics Files
- `per_tick_metrics.csv`: Time-series data
- `simulation_summary.txt`: Key KPIs and statistics

### Plots
- Life-safety success ratio over time
- Node energy consumption
- Link utilization statistics
- Traffic delivery rates by class

### Key Metrics
- **Recovery Time**: Ticks to achieve 95% life-safety delivery after severance
- **Energy Usage**: Total consumption and depletion events
- **Connectivity**: Island formation and reachability
- **Traffic Stats**: Admission/delivery rates per class

## Example Output

```
============================================================
6G NETWORK SIMULATION SUMMARY
============================================================

RECOVERY METRICS:
  Severance occurred at tick: 50
  Recovery time: 25 ticks
  Life safety success ratio: 0.87

ENERGY METRICS:
  gNB1: 125.3 units used, final SoC: 0.92
  gNB2: 98.7 units used, final SoC: 0.95
  relay1: 245.1 units used, final SoC: 0.78

CONNECTIVITY METRICS:
  Final number of islands: 2

TRAFFIC METRICS:
  Life Safety:
    Total offered: 1847.2, delivered: 1608.9
    Admission rate: 0.98, delivery rate: 0.87
  Operations:
    Total offered: 2893.4, delivered: 2521.8
    Admission rate: 0.95, delivery rate: 0.87
```

## Extending the Simulation

### Adding New Agent Policies

Replace `HeuristicAgent` with RL-trained policies:

```python
class RLAgent(BaseAgent):
    def compute_action(self, observation: AgentObservation) -> AgentAction:
        # Use trained policy network
        action_vector = self.policy_net(observation_to_tensor(observation))
        return tensor_to_action(action_vector)
```

### Enhanced Control Plane

Add more sophisticated control overlays:

```python
class DTNOverlay(ControlOverlay):
    """Delay-tolerant networking for extreme disruptions"""
```

### Realistic Traffic Models

Implement time-varying patterns:

```python
class RealisticTrafficGenerator(TrafficGenerator):
    def generate_traffic(self, tick: int) -> Dict[str, Dict[TrafficClass, float]]:
        # Add diurnal patterns, event-driven surges, etc.
```

## Limitations & Simplifications

- Simplified PHY/MAC/RLC modeling
- No actual 3GPP protocol stacks
- Heuristic agents (no RL training)
- Limited control message semantics
- No UE-to-UE sidelink modeling
- Simplified energy consumption model

## Future Work

- Integration with MARL frameworks (Ray, PettingZoo)
- Real-time visualization dashboard
- Multi-scenario optimization
- Hardware-in-the-loop validation
- 3GPP standard compliance improvements

## License

This is a research proof-of-concept implementation for academic and experimental use.
