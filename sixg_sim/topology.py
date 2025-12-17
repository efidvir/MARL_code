"""
Network topology modeling for 6G simulation.

Defines Node and Link classes representing network entities and connections,
along with utilities for loading topology configurations from YAML/JSON.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json


class NodeType(Enum):
    """Enumeration of node types in the network topology."""
    GNBSITE = "GNBSite"  # O-RU + DU, with optional CU-lite
    DU = "DU"            # Distributed Unit (baseband processing)
    CU = "CU"            # Centralized Unit (higher layer processing)
    RELAY = "Relay"      # Microwave / IAB / generic backhaul relay
    EDGEUPF = "EdgeUPF"  # Edge UPF / MEC
    CORE = "Core"        # Central 5GC / SMO / RIC block
    UE = "UE"            # User Equipment (mobile devices)
    SATELLITEGATEWAY = "SatelliteGateway"
    FIELDGATEWAY = "FieldGateway"


class LinkType(Enum):
    """Enumeration of link types in the network."""
    FIBER = "fiber"
    MICROWAVE = "microwave"
    IAB = "iab"  # Integrated Access and Backhaul
    D2D = "d2d"  # Device-to-Device
    SATELLITE = "satellite"


class TrafficClass(Enum):
    """Traffic classes with priority ordering."""
    LIFE_SAFETY = "life_safety"      # Mission-critical, highest priority
    OPERATIONS = "operations"        # Network & operational control
    TELEMETRY = "telemetry"         # Monitoring, statistics, logs
    BEST_EFFORT = "best_effort"      # General user data


@dataclass
class TrafficQueue:
    """Represents a traffic queue for a specific class at a node."""
    traffic_class: TrafficClass
    offered_load: float = 0.0  # Total traffic offered this tick
    admitted_load: float = 0.0  # Traffic admitted to queue
    queued_load: float = 0.0    # Current queue length
    delivered_load: float = 0.0 # Traffic successfully delivered this tick
    dropped_load: float = 0.0   # Traffic dropped this tick

    def reset_tick(self):
        """Reset per-tick counters."""
        self.offered_load = 0.0
        self.admitted_load = 0.0
        self.delivered_load = 0.0
        self.dropped_load = 0.0


@dataclass
class Node:
    """Represents a network node with its properties and state."""
    id: str
    node_type: NodeType
    initial_energy: float = 1.0  # State of Charge (0.0 to 1.0)
    coverage_area: Optional[str] = None
    is_survivor: bool = True  # Becomes False if node fails or depletes
    is_island: bool = False   # True when disconnected from core

    # State
    energy_soc: float = field(init=False)  # Current SoC
    queues: Dict[TrafficClass, TrafficQueue] = field(init=False)
    base_energy_consumption: float = 10.0  # Base power consumption per tick
    traffic_energy_factor: float = 0.1     # Energy per unit traffic
    control_energy_factor: float = 0.01    # Energy per control byte

    def __post_init__(self):
        """Initialize mutable state."""
        self.energy_soc = self.initial_energy
        self.queues = {
            cls: TrafficQueue(cls)
            for cls in TrafficClass
        }

    def get_energy_tier(self) -> str:
        """Get energy state tier for agent observations."""
        if self.energy_soc > 0.7:
            return "high"
        elif self.energy_soc > 0.3:
            return "medium"
        else:
            return "low"

    def update_energy(self, traffic_load: float, control_bytes: float):
        """Update energy state based on consumption."""
        consumption = (self.base_energy_consumption +
                      self.traffic_energy_factor * traffic_load +
                      self.control_energy_factor * control_bytes)
        # Slow down consumption to sustain long simulations
        self.energy_soc = max(0.0, self.energy_soc - consumption / 10000.0)

        # Mark as non-survivor if energy depleted (only for UEs; infra assumed powered)
        if self.energy_soc <= 0.0 and self.node_type == NodeType.UE:
            self.is_survivor = False

    def reset_queues(self):
        """Reset all traffic queues for new tick."""
        for queue in self.queues.values():
            queue.reset_tick()


@dataclass
class Link:
    """Represents a network link with capacity and state."""
    id: str
    endpoints: Tuple[str, str]  # (source, destination) node IDs
    capacity: float  # Throughput units per tick
    latency: int = 1  # Ticks of latency
    link_type: LinkType = LinkType.FIBER
    is_up: bool = True  # Link status

    # State
    current_utilization: float = 0.0  # Current used capacity

    def available_capacity(self) -> float:
        """Return available capacity on this link."""
        return self.capacity - self.current_utilization if self.is_up else 0.0

    def can_carry(self, traffic_amount: float) -> bool:
        """Check if link can carry additional traffic."""
        return self.available_capacity() >= traffic_amount

    def add_traffic(self, traffic_amount: float) -> float:
        """Add traffic to link, return amount actually carried."""
        available = self.available_capacity()
        carried = min(traffic_amount, available)
        self.current_utilization += carried
        return carried

    def reset_utilization(self):
        """Reset utilization for new tick."""
        self.current_utilization = 0.0


class Topology:
    """Network topology representation using NetworkX graph."""

    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for links
        self.nodes: Dict[str, Node] = {}
        self.links: Dict[str, Link] = {}

    def add_node(self, node: Node):
        """Add a node to the topology."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node_type=node.node_type.value)

    def add_link(self, link: Link):
        """Add a link to the topology."""
        self.links[link.id] = link
        self.graph.add_edge(link.endpoints[0], link.endpoints[1],
                          link_id=link.id, capacity=link.capacity)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get list of neighboring node IDs."""
        return list(self.graph.neighbors(node_id))

    def has_path(self, source: str, target: str) -> bool:
        """Check if there's a path from source to target."""
        try:
            return nx.has_path(self.graph, source, target)
        except nx.NetworkXError:
            return False

    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get shortest path between nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXError:
            return None

    def update_link_statuses(self):
        """Update graph based on current link statuses."""
        for link in self.links.values():
            edge_data = self.graph.get_edge_data(link.endpoints[0], link.endpoints[1])
            if edge_data:
                # Remove edge if link is down
                if not link.is_up:
                    self.graph.remove_edge(link.endpoints[0], link.endpoints[1])
                # Add edge if link is up and not present
                elif not self.graph.has_edge(link.endpoints[0], link.endpoints[1]):
                    self.graph.add_edge(link.endpoints[0], link.endpoints[1],
                                      link_id=link.id, capacity=link.capacity)


def load_topology_from_yaml(file_path: str) -> Topology:
    """Load topology configuration from YAML file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    topology = Topology()

    # Load nodes
    for node_config in config.get('nodes', []):
        node = Node(
            id=node_config['id'],
            node_type=NodeType(node_config['type']),
            initial_energy=node_config.get('initial_energy', 1.0),
            coverage_area=node_config.get('coverage_area')
        )
        topology.add_node(node)

    # Load links
    for link_config in config.get('links', []):
        link = Link(
            id=link_config['id'],
            endpoints=tuple(link_config['endpoints']),
            capacity=link_config['capacity'],
            latency=link_config.get('latency', 1),
            link_type=LinkType(link_config.get('type', 'fiber')),
            is_up=link_config.get('is_up', True)
        )
        topology.add_link(link)

    return topology


def load_topology_from_json(file_path: str) -> Topology:
    """Load topology configuration from JSON file."""
    with open(file_path, 'r') as f:
        config = json.load(f)

    topology = Topology()

    # Load nodes
    for node_config in config.get('nodes', []):
        node = Node(
            id=node_config['id'],
            node_type=NodeType(node_config['type']),
            initial_energy=node_config.get('initial_energy', 1.0),
            coverage_area=node_config.get('coverage_area')
        )
        topology.add_node(node)

    # Load links
    for link_config in config.get('links', []):
        link = Link(
            id=link_config['id'],
            endpoints=tuple(link_config['endpoints']),
            capacity=link_config['capacity'],
            latency=link_config.get('latency', 1),
            link_type=LinkType(link_config.get('type', 'fiber')),
            is_up=link_config.get('is_up', True)
        )
        topology.add_link(link)

    return topology


def generate_large_topology(num_nodes: int = 200, seed: int = 42) -> Topology:
    """
    Generate a large-scale 200-node network topology.

    Creates a realistic 5G/6G network with:
    - gNB sites with integrated DU/CU functionality
    - Separate DU and CU nodes
    - Various relay types
    - Edge UPF/MEC nodes
    - Core network nodes
    - Multiple connectivity layers
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    topology = Topology()

    # Node type distribution - adjust for infrastructure + UEs
    infrastructure_nodes = num_nodes
    ue_nodes = 2000  # Fixed UE count for realism

    node_counts = {
        NodeType.GNBSITE: 80,    # gNB sites (integrated DU/CU-lite)
        NodeType.DU: 40,         # Distributed Units
        NodeType.CU: 20,         # Centralized Units
        NodeType.RELAY: 35,      # Various relay types
        NodeType.EDGEUPF: 20,    # Edge UPF/MEC
        NodeType.CORE: 5,        # Core network (5GC, SMO, RIC)
        NodeType.UE: ue_nodes,   # User Equipment
    }

    # Create nodes
    node_id = 0
    coverage_areas = [f"zone_{i}" for i in range(10)]  # 10 coverage zones

    for node_type, count in node_counts.items():
        for i in range(count):
            node_id += 1
            coverage_area = random.choice(coverage_areas)

        # Set energy levels based on node type and location
            base_energy = 1.0
            if node_type == NodeType.UE:
                base_energy = random.uniform(0.6, 1.0)  # UEs start reasonably charged
            elif node_type == NodeType.RELAY:
                base_energy = 0.8  # Relays have lower energy reserves
            elif node_type in [NodeType.CORE, NodeType.EDGEUPF]:
                base_energy = 0.95  # Critical infrastructure slightly higher
            elif coverage_area in ["zone_7", "zone_8", "zone_9"]:  # Remote areas
                base_energy = 0.75

            # Add some random variation (less for infrastructure, more for UEs)
            if node_type == NodeType.UE:
                energy_variation = random.uniform(-0.1, 0.1)
            else:
                energy_variation = random.uniform(-0.05, 0.05)
            initial_energy = max(0.5, min(1.0, base_energy + energy_variation))

            # Per-node energy consumption tuning
            base_consumption = 10.0
            traffic_factor = 0.1
            control_factor = 0.01

            if node_type == NodeType.UE:
                base_consumption = 0.5   # Very low base draw for UEs
                traffic_factor = 0.01    # Light traffic cost
                control_factor = 0.003
            elif node_type == NodeType.GNBSITE:
                base_consumption = 3.0
                traffic_factor = 0.03
            elif node_type == NodeType.DU:
                base_consumption = 2.5
                traffic_factor = 0.025
            elif node_type == NodeType.CU:
                base_consumption = 2.0
                traffic_factor = 0.02
            elif node_type == NodeType.RELAY:
                base_consumption = 1.5
                traffic_factor = 0.015
            elif node_type in [NodeType.EDGEUPF, NodeType.CORE]:
                base_consumption = 2.5
                traffic_factor = 0.02

            node = Node(
                id=f"{node_type.value}_{node_id}",
                node_type=node_type,
                initial_energy=initial_energy,
                coverage_area=coverage_area
            )
            # Apply tuned energy parameters
            node.base_energy_consumption = base_consumption
            node.traffic_energy_factor = traffic_factor
            node.control_energy_factor = control_factor
            topology.add_node(node)

    # Create connectivity layers
    # Layer 1: Access layer (gNB <-> DU connections)
    gnb_nodes = [node_id for node_id, node in topology.nodes.items()
                if node.node_type == NodeType.GNBSITE]
    du_nodes = [node_id for node_id, node in topology.nodes.items()
               if node.node_type == NodeType.DU]

    link_id = 0
    # Connect each gNB to 1-2 nearby DUs
    for gnb in gnb_nodes:
        num_connections = random.randint(1, 2)
        connected_dus = random.sample(du_nodes, min(num_connections, len(du_nodes)))
        for du in connected_dus:
            link_id += 1
            capacity = random.choice([100, 200, 500])  # Low latency, high capacity
            link = Link(
                id=f"access_{link_id}",
                endpoints=(gnb, du),
                capacity=capacity,
                latency=1,
                link_type=LinkType.FIBER,
                is_up=True
            )
            topology.add_link(link)

    # Layer 2: Distribution layer (DU <-> CU connections)
    cu_nodes = [node_id for node_id, node in topology.nodes.items()
               if node.node_type == NodeType.CU]

    # Connect DUs to CUs (each DU connects to 1-3 CUs)
    for du in du_nodes:
        num_connections = random.randint(1, 3)
        connected_cus = random.sample(cu_nodes, min(num_connections, len(cu_nodes)))
        for cu in connected_cus:
            link_id += 1
            capacity = random.choice([500, 800, 1000])
            link = Link(
                id=f"dist_{link_id}",
                endpoints=(du, cu),
                capacity=capacity,
                latency=random.randint(2, 5),
                link_type=LinkType.FIBER,
                is_up=True
            )
            topology.add_link(link)

    # Layer 3: Backhaul layer (CU/Relay <-> Edge UPF)
    edge_upf_nodes = [node_id for node_id, node in topology.nodes.items()
                     if node.node_type == NodeType.EDGEUPF]
    relay_nodes = [node_id for node_id, node in topology.nodes.items()
                  if node.node_type == NodeType.RELAY]
    cu_plus_relay = cu_nodes + relay_nodes

    # Connect CUs and relays to edge UPFs
    for source in cu_plus_relay:
        num_connections = random.randint(1, 2)
        connected_upfs = random.sample(edge_upf_nodes, min(num_connections, len(edge_upf_nodes)))
        for upf in connected_upfs:
            link_id += 1
            link_type = random.choice([LinkType.FIBER, LinkType.MICROWAVE, LinkType.IAB])
            capacity = 800 if link_type == LinkType.FIBER else 300
            latency = 1 if link_type == LinkType.FIBER else random.randint(3, 8)

            link = Link(
                id=f"backhaul_{link_id}",
                endpoints=(source, upf),
                capacity=capacity,
                latency=latency,
                link_type=link_type,
                is_up=True
            )
            topology.add_link(link)

    # Layer 4: Core connections (Edge UPF <-> Core)
    core_nodes = [node_id for node_id, node in topology.nodes.items()
                 if node.node_type == NodeType.CORE]

    # Connect edge UPFs to core nodes
    for upf in edge_upf_nodes:
        # Each edge UPF connects to 1-2 core nodes
        num_connections = random.randint(1, 2)
        connected_cores = random.sample(core_nodes, min(num_connections, len(core_nodes)))
        for core in connected_cores:
            link_id += 1
            link = Link(
                id=f"core_{link_id}",
                endpoints=(upf, core),
                capacity=2000,  # High capacity core links
                latency=random.randint(5, 10),
                link_type=LinkType.FIBER,
                is_up=True
            )
            topology.add_link(link)

    # Add some cross-connections and mesh connectivity
    # Connect relays to each other for redundancy
    for i, relay1 in enumerate(relay_nodes):
        for relay2 in relay_nodes[i+1:i+4]:  # Connect to next 3 relays
            if random.random() < 0.3:  # 30% chance of mesh connection
                link_id += 1
                link_type = random.choice([LinkType.MICROWAVE, LinkType.IAB])
                capacity = 200 if link_type == LinkType.IAB else 400
                latency = random.randint(2, 6)

                link = Link(
                    id=f"mesh_{link_id}",
                    endpoints=(relay1, relay2),
                    capacity=capacity,
                    latency=latency,
                    link_type=link_type,
                    is_up=True
                )
                topology.add_link(link)

    # Add UE connectivity - UEs connect to nearest infrastructure
    infrastructure_nodes = gnb_nodes + du_nodes + relay_nodes
    ue_nodes = [node_id for node_id, node in topology.nodes.items()
               if node.node_type == NodeType.UE]

    # Group UEs by coverage area for more realistic connectivity
    ues_by_area = {}
    for ue_id in ue_nodes:
        ue_node = topology.nodes[ue_id]
        area = ue_node.coverage_area
        if area not in ues_by_area:
            ues_by_area[area] = []
        ues_by_area[area].append(ue_id)

    # Connect UEs in each area to local infrastructure
    total_ue_connections = 0
    disconnected_ues = 0
    for area, area_ues in ues_by_area.items():
        # Find infrastructure in this area
        area_infrastructure = [
            node_id for node_id in infrastructure_nodes
            if topology.nodes[node_id].coverage_area == area
        ]

        # If no local infrastructure, connect to any available
        if not area_infrastructure:
            area_infrastructure = infrastructure_nodes[:min(20, len(infrastructure_nodes))]

        # If still none, mark UEs as disconnected
        if not area_infrastructure:
            disconnected_ues += len(area_ues)
            continue

        # Connect each UE to 2-3 infrastructure nodes for redundancy
        for ue_id in area_ues:
            num_connections = random.randint(2, min(3, len(area_infrastructure)))
            connected_infra = random.sample(area_infrastructure, num_connections)

            for infra_id in connected_infra:
                link_id += 1
                # UE links have moderate capacity and low latency (wireless)
                capacity = random.choice([50, 75, 100])  # Mbps for UE connections
                latency = random.randint(1, 3)  # Very low latency wireless

                link = Link(
                    id=f"ue_{link_id}",
                    endpoints=(ue_id, infra_id),
                    capacity=capacity,
                    latency=latency,
                    link_type=LinkType.FIBER,  # Representing wireless backhaul
                    is_up=True
                )
                topology.add_link(link)
                total_ue_connections += 1

    print(f"UE connectivity: {len(ue_nodes) - disconnected_ues}/{len(ue_nodes)} have infrastructure access")
    print(f"Created {total_ue_connections} UE-infrastructure connections")

    return topology
