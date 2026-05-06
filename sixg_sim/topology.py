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
import math
import yaml
import json


class NodeType(Enum):
    """Enumeration of O-RAN node types in the network topology."""
    # O-RAN Radio Access Network components
    O_RU = "O-RU"              # O-RAN Radio Unit (antenna + RF)
    O_DU = "O-DU"              # O-RAN Distributed Unit (lower PHY/MAC/RLC)
    O_CU_CP = "O-CU-CP"        # O-RAN Central Unit - Control Plane (RRC/PDCP-C)
    O_CU_UP = "O-CU-UP"        # O-RAN Central Unit - User Plane (PDCP-U/SDAP)
    O_CU = "O-CU"              # Combined O-CU (not split)
    
    # RAN Intelligent Controllers
    NEAR_RT_RIC = "Near-RT-RIC"  # Near Real-Time RIC (xApps, <1s control)
    NON_RT_RIC = "Non-RT-RIC"    # Non Real-Time RIC (rApps, >1s control, part of SMO)
    
    # Core and Management
    SMO = "SMO"                # Service Management and Orchestration
    UPF = "UPF"                # User Plane Function (5GC)
    AMF = "AMF"                # Access and Mobility Management Function (5GC)
    CORE = "Core"              # Generic 5G Core placeholder
    
    # Edge and Transport
    EDGEUPF = "EdgeUPF"        # Edge UPF / MEC host
    RELAY = "Relay"            # Microwave / transport relay
    
    # Legacy compatibility
    GNBSITE = "gNB-Site"       # Combined O-RU + O-DU site
    DU = "DU"                  # Alias for O-DU
    CU = "CU"                  # Alias for O-CU
    
    # End devices
    UE = "UE"                  # User Equipment
    
    # Special nodes
    SATELLITEGATEWAY = "SatelliteGateway"
    FIELDGATEWAY = "FieldGateway"


class InterfaceType(Enum):
    """O-RAN interface types between network elements."""
    # Fronthaul
    OPEN_FH = "Open-FH"        # Open Fronthaul (7.2x) between O-RU and O-DU
    
    # Midhaul
    F1_C = "F1-C"              # F1 Control Plane: O-CU-CP <-> O-DU
    F1_U = "F1-U"              # F1 User Plane: O-CU-UP <-> O-DU
    F1 = "F1"                  # Combined F1 (for non-split CU)
    
    # CU internal
    E1 = "E1"                  # E1: O-CU-CP <-> O-CU-UP
    
    # RIC interfaces
    E2 = "E2"                  # E2: Near-RT RIC <-> O-CU/O-DU (near-RT control)
    A1 = "A1"                  # A1: Non-RT RIC <-> Near-RT RIC (policy)
    
    # Management
    O1 = "O1"                  # O1: SMO <-> all O-RAN NFs (management)
    O2 = "O2"                  # O2: SMO <-> O-Cloud (cloud management)
    
    # Core interfaces (NG)
    N2 = "N2"                  # N2: gNB <-> AMF (control plane)
    N3 = "N3"                  # N3: gNB <-> UPF (user plane)
    N4 = "N4"                  # N4: SMF <-> UPF
    NG = "NG"                  # Generic NG interface
    
    # Inter-gNB
    Xn_C = "Xn-C"              # Xn Control Plane (inter-gNB handover signaling)
    Xn_U = "Xn-U"              # Xn User Plane (inter-gNB data forwarding)
    Xn = "Xn"                  # Combined Xn
    X2 = "X2"                  # Legacy X2 (LTE interworking)
    
    # Backhaul/Transport
    BACKHAUL = "Backhaul"      # Generic IP backhaul
    MICROWAVE = "Microwave"    # Microwave transport link
    SATELLITE = "Satellite"    # Satellite backhaul
    
    # Access
    Uu = "Uu"                  # Air interface: UE <-> O-RU/gNB


class LinkType(Enum):
    """Physical link types in the network."""
    FIBER = "fiber"
    MICROWAVE = "microwave"                # Generic microwave (backward compat)
    MICROWAVE_PTP = "microwave_ptp"        # Traditional Ceragon MW (fixed, pre-aimed)
    MULTIHAUL_MESH = "multihaul_mesh"      # Siklu 60GHz MultiHaul TG (dynamic beam steering)
    TRANSPORT_RELAY = "transport_relay"     # Dynamic relay (backward compat)
    D2D = "d2d"                            # Device-to-Device / sidelink
    SATELLITE = "satellite"
    WIRELESS = "wireless"                  # Generic wireless (for UE access)


# Backward-compatible mapping for YAML files that use old values
_LINK_TYPE_ALIASES = {
    "iab": "transport_relay",
    "microwave": "microwave_ptp",          # Default microwave → PtP
}


def _parse_link_type(raw: str) -> LinkType:
    """Parse a link type string, handling old aliases."""
    raw = raw.lower().strip()
    raw = _LINK_TYPE_ALIASES.get(raw, raw)
    return LinkType(raw)


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
    is_rescue_service: bool = False  # True for rescue/emergency service UEs (3GPP TS 22.179)
    emergency_state: bool = False    # True when UE is in MCPTT emergency state
    emergency_type: Optional[str] = None  # 'emergency', 'imminent_peril', 'general'
    has_multihaul: bool = False  # True if site has Siklu MultiHaul TG (60GHz beam steering)
    # Geographic position (meters) — used for radius-based disaster zones
    x_pos: float = 0.0
    y_pos: float = 0.0
    zone: Optional[str] = None  # Geographic zone: 'north', 'south', 'east', 'west', 'central'

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
        # UEs drain 10× slower so they survive full 500-tick episodes
        # (energy is an observation signal, not a training bottleneck)
        divisor = 100000.0 if self.node_type == NodeType.UE else 10000.0
        self.energy_soc = max(0.0, self.energy_soc - consumption / divisor)

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
    interface_type: InterfaceType = InterfaceType.BACKHAUL  # O-RAN interface type
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
        self.graph = nx.Graph()  # Undirected graph for bidirectional links
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

    def _build_infrastructure_graph(self):
        """Build and cache the infrastructure-only graph (no Uu links)."""
        if not hasattr(self, '_infra_graph_cache') or self._infra_graph_cache is None:
            self._infra_graph_cache = nx.Graph()
            for node in self.nodes:
                if not node.startswith('UE_'):
                    self._infra_graph_cache.add_node(node)
            for link in self.links.values():
                if link.is_up and not any(ep.startswith('UE_') for ep in link.endpoints):
                    self._infra_graph_cache.add_edge(link.endpoints[0], link.endpoints[1])
            self._cache_stale = False   # freshly rebuilt

        return self._infra_graph_cache

    def invalidate_infrastructure_cache(self):
        """Invalidate the infrastructure graph cache when topology changes."""
        self._infra_graph_cache = None
        self._cache_stale = True    # signal that bridge set needs recompute

    def has_infrastructure_path(self, source: str, target: str) -> bool:
        """Check if there's a path from source to target using only infrastructure links (no Uu air interface)."""
        try:
            infra_graph = self._build_infrastructure_graph()
            return nx.has_path(infra_graph, source, target)
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
    
    def add_ue_dynamically(self, ue_id: str, coverage_area: Optional[str] = None, 
                           connect_to_rus: Optional[List[str]] = None) -> bool:
        """
        Dynamically add a UE to the topology.
        
        Args:
            ue_id: Unique identifier for the UE
            coverage_area: Coverage area for the UE
            connect_to_rus: List of O-RU IDs to connect to (if None, auto-connect)
        
        Returns:
            True if UE was added successfully
        """
        if ue_id in self.nodes:
            return False  # UE already exists
        
        # Create UE node
        ue_node = Node(
            id=ue_id,
            node_type=NodeType.UE,
            initial_energy=1.0,
            coverage_area=coverage_area
        )
        self.add_node(ue_node)
        
        # Connect to O-RUs
        if connect_to_rus is None:
            # Auto-connect: find O-RUs in the same coverage area or nearby
            o_ru_nodes = [nid for nid, node in self.nodes.items() 
                         if node.node_type == NodeType.O_RU and node.is_survivor]
            if not o_ru_nodes:
                return True  # UE added but no O-RUs available
            
            # Filter by coverage area if specified
            if coverage_area:
                area_rus = [nid for nid in o_ru_nodes 
                           if self.nodes[nid].coverage_area == coverage_area]
                if area_rus:
                    o_ru_nodes = area_rus
            
            # Connect to 2-3 O-RUs (multi-connectivity)
            import random
            num_connections = min(random.randint(2, 3), len(o_ru_nodes))
            selected_rus = random.sample(o_ru_nodes, num_connections)
            connect_to_rus = selected_rus
        
        # Create Uu (air interface) links
        import random
        link_counter = len(self.links) + 1
        for ru_id in connect_to_rus:
            if ru_id not in self.nodes:
                continue
            
            link_id = f"Uu_{ue_id}_{ru_id}_{link_counter}"
            link_counter += 1
            
            capacity = random.choice([50, 100, 150, 200])  # Mbps
            latency = random.randint(1, 5)  # ms
            
            uu_link = Link(
                id=link_id,
                endpoints=(ue_id, ru_id),
                capacity=capacity,
                latency=latency,
                link_type=LinkType.WIRELESS,
                interface_type=InterfaceType.Uu,
                is_up=True
            )
            self.add_link(uu_link)
        
        return True
    
    def remove_ue(self, ue_id: str) -> bool:
        """
        Remove a UE from the topology (and all its links).
        
        Args:
            ue_id: ID of UE to remove
        
        Returns:
            True if UE was removed successfully
        """
        if ue_id not in self.nodes:
            return False
        
        # Remove all links connected to this UE
        links_to_remove = []
        for link_id, link in self.links.items():
            if ue_id in link.endpoints:
                links_to_remove.append(link_id)
        
        for link_id in links_to_remove:
            link = self.links[link_id]
            # Remove from graph
            if self.graph.has_edge(link.endpoints[0], link.endpoints[1]):
                self.graph.remove_edge(link.endpoints[0], link.endpoints[1])
            del self.links[link_id]
        
        # Remove node
        if ue_id in self.graph:
            self.graph.remove_node(ue_id)
        del self.nodes[ue_id]
        
        return True


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
            coverage_area=node_config.get('coverage_area'),
            x_pos=float(node_config.get('x_pos', 0.0)),
            y_pos=float(node_config.get('y_pos', 0.0)),
            zone=node_config.get('zone'),
            is_rescue_service=node_config.get('is_rescue_service', False),
        )
        topology.add_node(node)

    # Load links
    for link_config in config.get('links', []):
        link = Link(
            id=link_config['id'],
            endpoints=tuple(link_config['endpoints']),
            capacity=link_config['capacity'],
            latency=link_config.get('latency', 1),
            link_type=_parse_link_type(link_config.get('type', 'fiber')),
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
            link_type=_parse_link_type(link_config.get('type', 'fiber')),
            is_up=link_config.get('is_up', True)
        )
        topology.add_link(link)

    return topology


def generate_large_topology(num_nodes: int = 200, seed: int = 42) -> Topology:
    """
    Generate a large-scale O-RAN compliant network topology.

    Creates a realistic 5G O-RAN network with:
    - O-RU (Radio Units) at cell sites
    - O-DU (Distributed Units) for lower-layer processing
    - O-CU-CP/O-CU-UP (Central Units) for higher-layer processing
    - Near-RT RIC for real-time RAN optimization
    - SMO (Service Management and Orchestration) with Non-RT RIC
    - UPF (User Plane Function) and AMF (Access Management Function)
    - Proper O-RAN interface types (Open-FH, F1, E2, A1, O1, NG, Xn)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    topology = Topology()

    # O-RAN node distribution
    ue_count = 2000  # Fixed UE count

    node_counts = {
        # Radio Access Network
        NodeType.O_RU: 80,        # O-RAN Radio Units (cell sites)
        NodeType.O_DU: 40,        # O-RAN Distributed Units
        NodeType.O_CU_CP: 10,     # O-CU Control Plane
        NodeType.O_CU_UP: 10,     # O-CU User Plane
        # RAN Intelligent Controllers
        NodeType.NEAR_RT_RIC: 5,  # Near Real-Time RIC
        NodeType.SMO: 2,          # SMO (includes Non-RT RIC)
        # Core Network
        NodeType.UPF: 10,         # User Plane Function
        NodeType.AMF: 5,          # Access and Mobility Management
        # Transport
        NodeType.RELAY: 25,       # Transport relays
        NodeType.EDGEUPF: 8,      # Edge UPF / MEC
        # End devices
        NodeType.UE: ue_count,
    }

    # Coverage zones
    coverage_areas = [f"zone_{i}" for i in range(10)]
    node_id = 0

    # Energy parameters by node type
    energy_params = {
        NodeType.O_RU: (0.85, 3.0, 0.03, 0.01),      # (base_energy, consumption, traffic_factor, control_factor)
        NodeType.O_DU: (0.90, 2.5, 0.025, 0.01),
        NodeType.O_CU_CP: (0.95, 2.0, 0.015, 0.015),
        NodeType.O_CU_UP: (0.95, 2.2, 0.02, 0.01),
        NodeType.NEAR_RT_RIC: (0.98, 3.0, 0.02, 0.02),
        NodeType.SMO: (0.99, 4.0, 0.01, 0.03),
        NodeType.UPF: (0.95, 3.5, 0.03, 0.01),
        NodeType.AMF: (0.98, 2.5, 0.01, 0.02),
        NodeType.RELAY: (0.80, 1.5, 0.015, 0.01),
        NodeType.EDGEUPF: (0.95, 3.0, 0.025, 0.01),
        NodeType.UE: (0.80, 0.5, 0.01, 0.003),
    }

    # Create all nodes
    for node_type, count in node_counts.items():
        for i in range(count):
            node_id += 1
            coverage_area = random.choice(coverage_areas)

            base_energy, base_consumption, traffic_factor, control_factor = energy_params.get(
                node_type, (0.9, 2.0, 0.02, 0.01)
            )

            # Add variation
            if node_type == NodeType.UE:
                initial_energy = random.uniform(0.6, 1.0)
            else:
                initial_energy = max(0.5, min(1.0, base_energy + random.uniform(-0.05, 0.05)))

            node = Node(
                id=f"{node_type.value}_{node_id}",
                node_type=node_type,
                initial_energy=initial_energy,
                coverage_area=coverage_area
            )
            node.base_energy_consumption = base_consumption
            node.traffic_energy_factor = traffic_factor
            node.control_energy_factor = control_factor
            topology.add_node(node)

    # Helper to get nodes by type
    def get_nodes(ntype):
        return [nid for nid, n in topology.nodes.items() if n.node_type == ntype]

    o_ru_nodes = get_nodes(NodeType.O_RU)
    o_du_nodes = get_nodes(NodeType.O_DU)
    o_cu_cp_nodes = get_nodes(NodeType.O_CU_CP)
    o_cu_up_nodes = get_nodes(NodeType.O_CU_UP)
    near_rt_ric_nodes = get_nodes(NodeType.NEAR_RT_RIC)
    smo_nodes = get_nodes(NodeType.SMO)
    upf_nodes = get_nodes(NodeType.UPF)
    amf_nodes = get_nodes(NodeType.AMF)
    relay_nodes = get_nodes(NodeType.RELAY)
    edge_upf_nodes = get_nodes(NodeType.EDGEUPF)
    ue_nodes = get_nodes(NodeType.UE)

    link_id = 0

    # ========== Layer 1: Open Fronthaul (O-RU <-> O-DU) ==========
    for o_ru in o_ru_nodes:
        num_conn = random.randint(1, 2)
        for o_du in random.sample(o_du_nodes, min(num_conn, len(o_du_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"OpenFH_{link_id}",
                endpoints=(o_ru, o_du),
                capacity=random.choice([25000, 50000]),  # 25-50 Gbps eCPRI
                latency=1,  # <100us requirement
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.OPEN_FH,
                is_up=True
            ))

    # ========== Layer 2: F1 Interface (O-DU <-> O-CU) ==========
    # F1-C: O-DU <-> O-CU-CP (control plane)
    for o_du in o_du_nodes:
        num_conn = random.randint(1, 2)
        for o_cu_cp in random.sample(o_cu_cp_nodes, min(num_conn, len(o_cu_cp_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"F1-C_{link_id}",
                endpoints=(o_du, o_cu_cp),
                capacity=random.choice([1000, 2000]),
                latency=random.randint(1, 3),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.F1_C,
                is_up=True
            ))

    # F1-U: O-DU <-> O-CU-UP (user plane)
    for o_du in o_du_nodes:
        num_conn = random.randint(1, 2)
        for o_cu_up in random.sample(o_cu_up_nodes, min(num_conn, len(o_cu_up_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"F1-U_{link_id}",
                endpoints=(o_du, o_cu_up),
                capacity=random.choice([5000, 10000]),
                latency=random.randint(1, 3),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.F1_U,
                is_up=True
            ))

    # ========== Layer 3: E1 Interface (O-CU-CP <-> O-CU-UP) ==========
    for o_cu_cp in o_cu_cp_nodes:
        for o_cu_up in o_cu_up_nodes:
            if random.random() < 0.5:  # Not all CU-CP connect to all CU-UP
                link_id += 1
                topology.add_link(Link(
                    id=f"E1_{link_id}",
                    endpoints=(o_cu_cp, o_cu_up),
                    capacity=2000,
                    latency=1,
                    link_type=LinkType.FIBER,
                    interface_type=InterfaceType.E1,
                    is_up=True
                ))

    # ========== Layer 4: E2 Interface (Near-RT RIC <-> O-DU/O-CU) ==========
    # E2 to O-DUs
    for ric in near_rt_ric_nodes:
        for o_du in random.sample(o_du_nodes, min(15, len(o_du_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"E2_{link_id}",
                endpoints=(ric, o_du),
                capacity=500,
                latency=random.randint(2, 5),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.E2,
                is_up=True
            ))
        # E2 to O-CU-CP
        for o_cu_cp in random.sample(o_cu_cp_nodes, min(5, len(o_cu_cp_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"E2_{link_id}",
                endpoints=(ric, o_cu_cp),
                capacity=500,
                latency=random.randint(2, 5),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.E2,
                is_up=True
            ))

    # ========== Layer 5: A1 Interface (SMO/Non-RT RIC <-> Near-RT RIC) ==========
    for smo in smo_nodes:
        for ric in near_rt_ric_nodes:
            link_id += 1
            topology.add_link(Link(
                id=f"A1_{link_id}",
                endpoints=(smo, ric),
                capacity=200,
                latency=random.randint(5, 15),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.A1,
                is_up=True
            ))

    # ========== Layer 6: O1 Interface (SMO <-> all O-RAN NFs) ==========
    all_oran_nfs = o_ru_nodes + o_du_nodes + o_cu_cp_nodes + o_cu_up_nodes + near_rt_ric_nodes
    for smo in smo_nodes:
        for nf in random.sample(all_oran_nfs, min(50, len(all_oran_nfs))):
            link_id += 1
            topology.add_link(Link(
                id=f"O1_{link_id}",
                endpoints=(smo, nf),
                capacity=100,
                latency=random.randint(10, 30),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.O1,
                is_up=True
            ))

    # ========== Layer 7: NG Interface (O-CU <-> Core) ==========
    # N2: O-CU-CP <-> AMF (control)
    for o_cu_cp in o_cu_cp_nodes:
        for amf in random.sample(amf_nodes, min(2, len(amf_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"N2_{link_id}",
                endpoints=(o_cu_cp, amf),
                capacity=1000,
                latency=random.randint(3, 8),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.N2,
                is_up=True
            ))

    # N3: O-CU-UP <-> UPF (user plane)
    for o_cu_up in o_cu_up_nodes:
        for upf in random.sample(upf_nodes, min(3, len(upf_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"N3_{link_id}",
                endpoints=(o_cu_up, upf),
                capacity=10000,
                latency=random.randint(2, 5),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.N3,
                is_up=True
            ))

    # N4: AMF/SMF <-> UPF (simplified)
    for amf in amf_nodes:
        for upf in random.sample(upf_nodes, min(4, len(upf_nodes))):
            link_id += 1
            topology.add_link(Link(
                id=f"N4_{link_id}",
                endpoints=(amf, upf),
                capacity=1000,
                latency=random.randint(3, 8),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.N4,
                is_up=True
            ))

    # ========== Layer 8: Xn Interface (Inter-gNB / O-CU <-> O-CU) ==========
    for i, cu1 in enumerate(o_cu_cp_nodes):
        for cu2 in o_cu_cp_nodes[i+1:]:
            if random.random() < 0.4:
                link_id += 1
                topology.add_link(Link(
                    id=f"Xn_{link_id}",
                    endpoints=(cu1, cu2),
                    capacity=2000,
                    latency=random.randint(2, 5),
                    link_type=LinkType.FIBER,
                    interface_type=InterfaceType.Xn,
                    is_up=True
                ))

    # ========== Layer 9: Transport/Backhaul — Optimally Planned ==========
    #
    # The transport topology is PLANNED (not random). This means:
    #   - MultiHaul TG deployed at high-value relay locations (geographic
    #     crossroads where mesh provides maximum path redundancy)
    #   - Each relay connects to its NEAREST O-DU (shortest geographic path)
    #   - MultiHaul mesh connects k-nearest neighbours (range-limited 60GHz)
    #   - Static backup links placed at single-points-of-failure for 2-path
    #     redundancy on critical transport corridors
    #
    # This simulates a professional Ceragon network planning stage output.

    # --- Step 1: Assign geographic coordinates for proximity planning ---
    # Use the coverage_area zones as a proxy for geographic clustering
    # Assign (x,y) positions based on zone + jitter for realistic spacing
    area_centers = {}
    area_idx = 0
    all_infra = relay_nodes + o_du_nodes + edge_upf_nodes
    for nid in all_infra:
        area = topology.nodes[nid].coverage_area or f"zone_{area_idx}"
        if area not in area_centers:
            # Place zone centers on a grid
            grid_cols = max(3, int(len(set(topology.nodes[n].coverage_area for n in all_infra if topology.nodes[n].coverage_area)) ** 0.5) + 1)
            cx = (area_idx % grid_cols) * 2.0
            cy = (area_idx // grid_cols) * 2.0
            area_centers[area] = (cx, cy)
            area_idx += 1

    node_pos = {}
    area_node_count = {}
    for nid in all_infra:
        area = topology.nodes[nid].coverage_area or "default"
        cx, cy = area_centers.get(area, (0, 0))
        cnt = area_node_count.get(area, 0)
        # Spread nodes within zone in a small circle
        angle = cnt * 2.3999  # golden angle for uniform distribution
        r = 0.3 + 0.15 * cnt
        node_pos[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
        area_node_count[area] = cnt + 1

    def _dist(a, b):
        ax, ay = node_pos.get(a, (0, 0))
        bx, by = node_pos.get(b, (0, 0))
        return ((ax - bx)**2 + (ay - by)**2) ** 0.5

    # --- Step 2: MultiHaul placement — at geographic crossroads ---
    # Score each relay by "connectivity value" = how many other nodes
    # it can potentially bridge (high degree = crossroad)
    relay_scores = {}
    for relay in relay_nodes:
        # Count nearby O-DUs and other relays within planning range
        nearby = sum(1 for n in (o_du_nodes + relay_nodes)
                     if n != relay and _dist(relay, n) < 4.0)
        relay_scores[relay] = nearby

    # Top 40% relays by score get MultiHaul
    multihaul_relay_count = max(1, int(len(relay_nodes) * 0.40))
    sorted_relays = sorted(relay_nodes, key=lambda r: relay_scores.get(r, 0), reverse=True)
    multihaul_relays = set(sorted_relays[:multihaul_relay_count])
    for nid in multihaul_relays:
        topology.nodes[nid].has_multihaul = True

    # Top 20% O-DUs near MultiHaul relay clusters get MultiHaul
    du_mh_scores = {}
    for du in o_du_nodes:
        du_mh_scores[du] = sum(1 for r in multihaul_relays if _dist(du, r) < 3.0)
    multihaul_gnb_count = max(1, int(len(o_du_nodes) * 0.20))
    sorted_dus = sorted(o_du_nodes, key=lambda d: du_mh_scores.get(d, 0), reverse=True)
    multihaul_gnbs = set(sorted_dus[:multihaul_gnb_count])
    for nid in multihaul_gnbs:
        topology.nodes[nid].has_multihaul = True

    all_multihaul_nodes = multihaul_relays | multihaul_gnbs

    # --- Step 3: Relay ↔ O-DU links — each relay connects to nearest DUs ---
    connected_relay_dus = set()
    for relay in relay_nodes:
        # Sort O-DUs by distance, connect to 2 nearest
        nearest_dus = sorted(o_du_nodes, key=lambda d: _dist(relay, d))[:2]
        for o_du in nearest_dus:
            pair = tuple(sorted([relay, o_du]))
            if pair in connected_relay_dus:
                continue
            connected_relay_dus.add(pair)
            link_id += 1
            if relay in multihaul_relays and o_du in multihaul_gnbs:
                lt = LinkType.MULTIHAUL_MESH
                cap = 1000  # 60GHz — up to 1 Gbps
            else:
                lt = LinkType.MICROWAVE_PTP
                cap = 500   # Traditional MW PtP
            topology.add_link(Link(
                id=f"Backhaul_{link_id}",
                endpoints=(relay, o_du),
                capacity=cap,
                latency=max(1, int(_dist(relay, o_du) * 1.5)),
                link_type=lt,
                interface_type=InterfaceType.BACKHAUL,
                is_up=True
            ))

    # --- Step 4: Relay ↔ Edge UPF — nearest aggregation ---
    for relay in relay_nodes:
        nearest_eupfs = sorted(edge_upf_nodes, key=lambda e: _dist(relay, e))[:2]
        for eupf in nearest_eupfs:
            link_id += 1
            topology.add_link(Link(
                id=f"Backhaul_{link_id}",
                endpoints=(relay, eupf),
                capacity=800,
                latency=max(2, int(_dist(relay, eupf) * 2)),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.BACKHAUL,
                is_up=True
            ))

    # Edge UPF ↔ Core UPF
    for eupf in edge_upf_nodes:
        nearest_upfs = sorted(upf_nodes, key=lambda u: _dist(eupf, u))[:2]
        for upf in nearest_upfs:
            link_id += 1
            topology.add_link(Link(
                id=f"N9_{link_id}",
                endpoints=(eupf, upf),
                capacity=5000,
                latency=max(3, int(_dist(eupf, upf) * 2.5)),
                link_type=LinkType.FIBER,
                interface_type=InterfaceType.NG,
                is_up=True
            ))

    # --- Step 5: MultiHaul mesh — k-nearest neighbours (range-limited) ---
    # 60GHz has ~300m range → connect each MH node to its 3 nearest MH peers
    multihaul_list = sorted(all_multihaul_nodes)
    connected_mh = set()
    for mh1 in multihaul_list:
        nearest = sorted(
            [m for m in multihaul_list if m != mh1],
            key=lambda m: _dist(mh1, m)
        )[:3]  # k=3 nearest neighbours
        for mh2 in nearest:
            pair = tuple(sorted([mh1, mh2]))
            if pair in connected_mh:
                continue
            connected_mh.add(pair)
            link_id += 1
            topology.add_link(Link(
                id=f"MultiHaul_{link_id}",
                endpoints=(mh1, mh2),
                capacity=1000,   # 60GHz multi-gigabit
                latency=max(1, int(_dist(mh1, mh2) * 0.8)),
                link_type=LinkType.MULTIHAUL_MESH,
                interface_type=InterfaceType.MICROWAVE,
                is_up=True
            ))

    # --- Step 6: Static backup links — placed at single-points-of-failure ---
    # For each non-MultiHaul relay, check if it has only ONE transport path.
    # If so, add a MW PtP backup to its nearest non-directly-connected peer.
    non_mh_relays = [r for r in relay_nodes if r not in multihaul_relays]
    connected_backup = set()
    for r1 in non_mh_relays:
        # Find nodes r1 is already directly connected to
        direct_peers = set()
        for lid, link in topology.links.items():
            ep = link.endpoints
            if r1 in ep:
                peer = ep[1] if ep[0] == r1 else ep[0]
                direct_peers.add(peer)
        # If only connected to 1-2 infra peers, add backup for redundancy
        infra_peers = direct_peers & set(relay_nodes + o_du_nodes)
        if len(infra_peers) <= 2:
            # Find nearest non-connected relay/DU
            candidates = [n for n in (relay_nodes + o_du_nodes)
                         if n != r1 and n not in direct_peers]
            if candidates:
                nearest = min(candidates, key=lambda c: _dist(r1, c))
                pair = tuple(sorted([r1, nearest]))
                if pair not in connected_backup:
                    connected_backup.add(pair)
                    link_id += 1
                    topology.add_link(Link(
                        id=f"MW_Backup_{link_id}",
                        endpoints=(r1, nearest),
                        capacity=300,
                        latency=max(2, int(_dist(r1, nearest) * 2)),
                        link_type=LinkType.MICROWAVE_PTP,
                        interface_type=InterfaceType.MICROWAVE,
                        is_up=True
                    ))

    # ========== Layer 10: Uu Interface (UE <-> O-RU Air Interface) ==========
    # UEs connect to O-RUs via the air interface (Uu)
    
    # Group UEs by coverage area
    ues_by_area = {}
    for ue_id in ue_nodes:
        ue_node = topology.nodes[ue_id]
        area = ue_node.coverage_area
        if area not in ues_by_area:
            ues_by_area[area] = []
        ues_by_area[area].append(ue_id)

    # O-RUs by area
    o_rus_by_area = {}
    for o_ru_id in o_ru_nodes:
        o_ru_node = topology.nodes[o_ru_id]
        area = o_ru_node.coverage_area
        if area not in o_rus_by_area:
            o_rus_by_area[area] = []
        o_rus_by_area[area].append(o_ru_id)

    total_ue_connections = 0
    disconnected_ues = 0
    
    for area, area_ues in ues_by_area.items():
        # Find O-RUs in this area (primary coverage)
        area_o_rus = o_rus_by_area.get(area, [])
        
        # If no local O-RUs, try any available
        if not area_o_rus:
            area_o_rus = o_ru_nodes[:min(10, len(o_ru_nodes))]
        
        if not area_o_rus:
            disconnected_ues += len(area_ues)
            continue

        # Connect each UE to 2-3 O-RUs (multi-connectivity)
        for ue_id in area_ues:
            num_connections = random.randint(2, min(3, len(area_o_rus)))
            connected_o_rus = random.sample(area_o_rus, num_connections)

            for o_ru_id in connected_o_rus:
                link_id += 1
                # Air interface: variable capacity based on conditions
                capacity = random.choice([50, 100, 150, 200])  # Mbps
                latency = random.randint(1, 5)  # ms

                topology.add_link(Link(
                    id=f"Uu_{link_id}",
                    endpoints=(ue_id, o_ru_id),
                    capacity=capacity,
                    latency=latency,
                    link_type=LinkType.WIRELESS,
                    interface_type=InterfaceType.Uu,
                    is_up=True
                ))
                total_ue_connections += 1

    print(f"UE connectivity: {len(ue_nodes) - disconnected_ues}/{len(ue_nodes)} have O-RU access via Uu")
    print(f"Created {total_ue_connections} Uu (air interface) connections")

    # Ensure all up links are in the graph
    topology.update_link_statuses()

    return topology
