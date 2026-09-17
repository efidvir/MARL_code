"""
Network topology modeling for 6G simulation.

Defines Node and Link classes representing network entities and connections,
along with utilities for loading topology configurations from YAML/JSON.
"""

import networkx as nx
import numpy as np
from typing import ClassVar, Dict, List, Optional, Any, Tuple
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


def _parse_interface_type(raw: str) -> InterfaceType:
    """Parse an interface type string by enum value (e.g. 'Uu', 'Open-FH')
    or by enum name (e.g. 'OPEN_FH', 'uu')."""
    raw = str(raw).strip()
    try:
        return InterfaceType(raw)
    except ValueError:
        pass
    try:
        return InterfaceType[raw.upper().replace('-', '_')]
    except KeyError:
        raise ValueError(f"Unknown interface type: {raw!r}")


class TrafficClass(Enum):
    """Traffic classes with priority ordering."""
    LIFE_SAFETY = "life_safety"      # Mission-critical, highest priority
    OPERATIONS = "operations"        # Network & operational control
    TELEMETRY = "telemetry"         # Monitoring, statistics, logs
    BEST_EFFORT = "best_effort"      # General user data


@dataclass
class TrafficQueue:
    """Represents a traffic queue for a specific class at a node.

    BACKLOG SEMANTICS (corrected).  `queued_load` used to be incremented as
    `queued_load += delivered` in Simulator._forward_traffic and was never
    reset by reset_tick(), which made it a MONOTONIC CUMULATIVE-DELIVERY
    COUNTER — it went up every time traffic was successfully delivered, i.e.
    it was largest exactly when the queue was healthiest.  It was nonetheless
    fed to the policy as `LocalSliceState.current_queue_length`
    (simulation.py, _build_agent_observations), so the "queue length"
    observation was really "total bytes I have ever delivered", monotonically
    rising with tick index and carrying no congestion information at all.

    It is now a genuine BACKLOG, maintained by `update_backlog()`:

        backlog_{t} = clamp( backlog_{t-1}·(1 − LEAK) + (offered − delivered),
                             0, BUFFER_MBIT )

    a leaky-bucket / RED-style average queue length in Mbit (tick = 1 s, so
    Mbps·tick = Mbit).  It RISES while traffic is being dropped, DRAINS with
    time constant 1/LEAK = 5 ticks once drops stop, and SATURATES at a finite
    buffer, which is what a real node with finite memory does.

    Why a leaky bucket rather than a strict FIFO backlog: a strict backlog
    would require re-presenting the carried-over load to the router next tick,
    which changes offered-volume accounting and therefore every arm's headline
    delivery KPI.  The leak keeps the fix confined to the observation that was
    wrong.  Nothing here alters delivered/dropped accounting.
    """
    traffic_class: TrafficClass
    offered_load: float = 0.0  # Total traffic offered this tick
    admitted_load: float = 0.0  # Traffic admitted to queue
    queued_load: float = 0.0    # Current backlog (Mbit) — see class docstring
    delivered_load: float = 0.0 # Traffic successfully delivered this tick
    dropped_load: float = 0.0   # Traffic dropped this tick
    # Tick index of the most recent successful delivery on this slice, or -1 if
    # nothing has ever been delivered.  Feeds an honest
    # LocalSliceState.current_freshness (age of information) instead of the
    # hardcoded 5.0 the observation carried before.
    last_delivered_tick: int = -1

    # Leaky-bucket backlog parameters.
    #   LEAK: service/ageing rate per tick.  0.2 => 5-tick drain time
    #         constant, i.e. steady-state backlog = 5x the per-tick drop rate,
    #         the same order as the 3GPP PDB range for the non-critical
    #         slices in this model.
    #   BUFFER_MBIT: finite node buffer.  500 Mbit ~ 62 MB, a plausible
    #         per-slice buffer for an O-RU/O-DU class device, and it bounds
    #         the observation so a permanently starved slice reports
    #         "saturated" rather than an unbounded number.
    QUEUE_LEAK_PER_TICK: ClassVar[float] = 0.20
    QUEUE_BUFFER_MBIT:   ClassVar[float] = 500.0

    def reset_tick(self):
        """Reset per-tick counters.

        `queued_load` and `last_delivered_tick` are deliberately NOT reset:
        they are carried STATE (a backlog and an age), not per-tick counters.
        """
        self.offered_load = 0.0
        self.admitted_load = 0.0
        self.delivered_load = 0.0
        self.dropped_load = 0.0

    def update_backlog(self, tick: int):
        """Advance the leaky-bucket backlog from this tick's offered/delivered.

        Call ONCE per tick per queue, after offered_load / delivered_load have
        been set.  See the class docstring for the recurrence.
        """
        undelivered = max(0.0, self.offered_load - self.delivered_load)
        self.queued_load = min(
            self.QUEUE_BUFFER_MBIT,
            max(0.0, self.queued_load * (1.0 - self.QUEUE_LEAK_PER_TICK)
                + undelivered))
        if self.delivered_load > 0.0:
            self.last_delivered_tick = int(tick)

    def freshness_ticks(self, tick: int) -> float:
        """Age of information for this slice, in ticks.

        0 when something was delivered this tick; the number of ticks since the
        last successful delivery otherwise; and the current tick index itself
        when nothing has EVER been delivered (the honest answer to "how stale
        is my information" for a slice that has never been served).
        """
        if self.last_delivered_tick < 0:
            return float(max(0, tick))
        return float(max(0, tick - self.last_delivered_tick))


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
        # Note: Disabled battery depletion deactivation for UEs to keep offered traffic constant
        if self.energy_soc <= 0.0 and self.node_type == NodeType.UE:
            pass


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

    def is_ue_node(self, node_id: str) -> bool:
        """Return True if node_id is a UE (any UE, including dynamically added
        rescue UEs like 'Rescue_UE_N'). Uses the node type when the node is
        registered; falls back to name-based matching otherwise."""
        node = self.nodes.get(node_id)
        if node is not None:
            return node.node_type == NodeType.UE
        # Name-based fallback: 'UE_*' or '*_UE_*' (e.g. 'Rescue_UE_5')
        return node_id.startswith('UE_') or '_UE_' in node_id

    def _build_infrastructure_graph(self):
        """Build and cache the infrastructure-only graph (no Uu links)."""
        if not hasattr(self, '_infra_graph_cache') or self._infra_graph_cache is None:
            self._infra_graph_cache = nx.Graph()
            for node in self.nodes:
                if not self.is_ue_node(node):
                    self._infra_graph_cache.add_node(node)
            for link in self.links.values():
                if link.is_up and not any(self.is_ue_node(ep) for ep in link.endpoints):
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
        """Update graph based on current link statuses.

        An edge is present iff at least one link between its endpoints is up
        (parallel links share a single edge in this undirected graph).
        Handles full fail -> restore -> fail cycles symmetrically.
        """
        # Determine, per endpoint pair, whether any link is up
        up_link_for_pair: Dict[Tuple[str, str], Link] = {}
        all_pairs = set()
        for link in self.links.values():
            pair = tuple(sorted(link.endpoints))
            all_pairs.add(pair)
            if link.is_up and pair not in up_link_for_pair:
                up_link_for_pair[pair] = link

        changed = False
        for pair in all_pairs:
            u, v = pair
            up_link = up_link_for_pair.get(pair)
            if up_link is not None:
                # Re-add edge if link is up and edge absent (e.g. restored link)
                if not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v,
                                        link_id=up_link.id,
                                        capacity=up_link.capacity)
                    changed = True
            elif self.graph.has_edge(u, v):
                # Remove edge if all links between the pair are down
                self.graph.remove_edge(u, v)
                changed = True

        if changed:
            # Keep infrastructure connectivity cache consistent with the graph
            self.invalidate_infrastructure_cache()
    
    # Attachment radius for a dynamically joining UE, metres.  A UE can only
    # camp on a cell whose signal it can actually hear; 1500 m is the FR1
    # macro cell-edge distance at nominal power in this model (SINR ~16 dB at
    # 23 dBm, comfortably above the QPSK_1/3 threshold), so it is the outer
    # bound of a plausible attachment, not a free-for-all.
    UE_ATTACH_MAX_DIST_M = 1500.0
    UE_ATTACH_MAX_CELLS  = 2      # dual connectivity, matching the generated
                                  # topology's 1-2 nearest-O-RU attachment

    def add_ue_dynamically(self, ue_id: str, coverage_area: Optional[str] = None,
                           connect_to_rus: Optional[List[str]] = None,
                           x_pos: Optional[float] = None,
                           y_pos: Optional[float] = None,
                           rng: Optional[Any] = None) -> bool:
        """
        Dynamically add a UE to the topology.

        GEOMETRY AND ATTACHMENT (corrected).  This method previously created
        the UE with no coordinates at all — Node's x_pos/y_pos defaults left
        every dynamically added UE (all 20 rescue UEs of a `rescue_force_arrival`
        event) sitting at (0, 0), the corner of the 8 km x 8 km deployment — and
        then attached it to 2-3 cells chosen with `random.sample`, i.e. UNIFORMLY
        AT RANDOM over every surviving O-RU regardless of distance.  Two
        consequences, both of which flattered the results:
          * a rescue UE at (0, 0) is up to 11 km from the incident it was
            dispatched to, so `geo_disaster` radii, fragment membership and any
            distance-based capacity model saw the wrong geometry entirely;
          * random attachment gave every rescue UE a link to a cell it could not
            physically hear, and (because the samples were independent) spread
            the rescue population across fragments so that "rescue UE is
            attached" was essentially guaranteed for free.

        Now: the caller supplies real coordinates (see
        Simulator._handle_scenario_event's rescue_force_arrival, which scatters
        them around the incident zone), and attachment is NEAREST-CELL within
        UE_ATTACH_MAX_DIST_M — the physically correct rule, and the same rule
        build_comparison_topology uses for its static UE population.

        Args:
            ue_id: Unique identifier for the UE
            coverage_area: Coverage area for the UE
            connect_to_rus: explicit list of cell IDs to attach to.  When given
                it overrides nearest-cell selection (used by scenario events
                that pin an attachment deliberately).
            x_pos, y_pos: UE coordinates in metres.  When omitted, the UE is
                placed at the centroid of the cells in its coverage area (or of
                all surviving cells), which is a far better default than (0, 0)
                but callers with a real position should always pass one.
            rng: optional random.Random for the per-link capacity/latency draw,
                so a caller can keep the draw reproducible.

        Returns:
            True if UE was added successfully
        """
        if ue_id in self.nodes:
            return False  # UE already exists

        import random as _random
        _rng = rng if rng is not None else _random

        # Candidate serving cells: surviving radio access nodes.  O_RU is the
        # normal case; RELAY sites are included because a relay hosts its own
        # FR1 small cell (integrated access node — design decision 2), so it
        # is a legitimate attachment point.
        access_types = {NodeType.O_RU, NodeType.RELAY}
        cells = [nid for nid, node in self.nodes.items()
                 if node.node_type in access_types and node.is_survivor]
        area_cells = [nid for nid in cells
                      if coverage_area and self.nodes[nid].coverage_area == coverage_area]

        # ── Position ──────────────────────────────────────────────────────
        if x_pos is None or y_pos is None:
            ref = area_cells or cells
            if ref:
                x_pos = sum(self.nodes[c].x_pos for c in ref) / len(ref)
                y_pos = sum(self.nodes[c].y_pos for c in ref) / len(ref)
            else:
                x_pos, y_pos = 0.0, 0.0

        # Create UE node WITH coordinates
        ue_node = Node(
            id=ue_id,
            node_type=NodeType.UE,
            initial_energy=1.0,
            coverage_area=coverage_area,
            x_pos=float(x_pos),
            y_pos=float(y_pos),
        )
        self.add_node(ue_node)

        # ── Attachment: nearest cell(s) inside the attach radius ──────────
        if connect_to_rus is None:
            pool = area_cells or cells
            if not pool:
                return True  # UE added but no access node available

            def _d(cid):
                n = self.nodes[cid]
                return math.hypot(n.x_pos - ue_node.x_pos,
                                  n.y_pos - ue_node.y_pos)

            ranked = sorted(pool, key=_d)
            in_range = [c for c in ranked if _d(c) <= self.UE_ATTACH_MAX_DIST_M]
            if in_range:
                connect_to_rus = in_range[:self.UE_ATTACH_MAX_CELLS]
            else:
                # Nothing in range.  Attach to the single nearest cell rather
                # than to nothing: a UE out of coverage of every surviving cell
                # is what `_reachable_ue_fraction` is supposed to be able to
                # report, and the honest way to express "barely in coverage" is
                # one weak link whose capacity the SINR model will then scale
                # down (LinkType.WIRELESS is MCS-scaled in consume_path).
                connect_to_rus = ranked[:1]

        # Create Uu (air interface) links
        link_counter = len(self.links) + 1
        for ru_id in connect_to_rus:
            if ru_id not in self.nodes:
                continue

            link_id = f"Uu_{ue_id}_{ru_id}_{link_counter}"
            link_counter += 1

            capacity = _rng.choice([50, 100, 150, 200])  # Mbps
            latency = _rng.randint(1, 5)  # ms

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
        endpoints = tuple(link_config['endpoints'])
        # Interface type: honor explicit key, else infer — a link with a UE
        # endpoint is Uu (air interface), all others are backhaul. Matches the
        # convention used by the code-built topologies (InterfaceType.Uu on
        # UE<->O-RU links, BACKHAUL on transport).
        raw_iface = link_config.get('interface', link_config.get('interface_type'))
        if raw_iface is not None:
            interface_type = _parse_interface_type(raw_iface)
        elif any(topology.is_ue_node(ep) for ep in endpoints):
            interface_type = InterfaceType.Uu
        else:
            interface_type = InterfaceType.BACKHAUL
        link = Link(
            id=link_config['id'],
            endpoints=endpoints,
            capacity=link_config['capacity'],
            latency=link_config.get('latency', 1),
            link_type=_parse_link_type(link_config.get('type', 'fiber')),
            interface_type=interface_type,
            is_up=link_config.get('is_up', True)
        )
        topology.add_link(link)

    # Auto-generate geographic coordinates if missing (for dashboard Geo Map)
    has_coords = any(n.x_pos != 0.0 or n.y_pos != 0.0 for n in topology.nodes.values())
    if not has_coords and topology.nodes:
        import math
        import random
        # Map compass zones to angles (degrees)
        zone_angles = {
            "N": 90, "NE": 45, "E": 0, "SE": 315,
            "S": 270, "SW": 225, "W": 180, "NW": 135,
            "north": 90, "east": 0, "south": 270, "west": 180,
            "central": None, "core": None
        }
        
        # Center of the "city" map
        cx, cy = 5000.0, 5000.0
        radius = 3500.0
        
        for nid, node in topology.nodes.items():
            area = str(node.coverage_area or node.zone or "").lower().replace("zone", "")
            
            # Find matching zone
            angle = None
            for key, val in zone_angles.items():
                if key.lower() == area:
                    angle = val
                    break
            
            # If no match or central, put near center
            if angle is None:
                if 'core' in nid.lower() or 'upf' in nid.lower():
                    # Core nodes strictly in center
                    node.x_pos = cx + random.uniform(-300, 300)
                    node.y_pos = cy + random.uniform(-300, 300)
                else:
                    # Unknown nodes spread out
                    a = random.uniform(0, 360)
                    r = random.uniform(500, 4000)
                    node.x_pos = cx + r * math.cos(math.radians(a))
                    node.y_pos = cy + r * math.sin(math.radians(a))
            else:
                # Place in the specific geographic sector
                base_x = cx + radius * math.cos(math.radians(angle))
                base_y = cy + radius * math.sin(math.radians(angle))
                
                if 'upf' in nid.lower():
                    node.x_pos = base_x + random.uniform(-100, 100)
                    node.y_pos = base_y + random.uniform(-100, 100)
                elif 'gnb' in nid.lower():
                    node.x_pos = base_x + random.uniform(-600, 600)
                    node.y_pos = base_y + random.uniform(-600, 600)
                elif 'relay' in nid.lower():
                    node.x_pos = base_x + random.uniform(-800, 800)
                    node.y_pos = base_y + random.uniform(-800, 800)
                elif 'ue' in nid.lower():
                    node.x_pos = base_x + random.uniform(-1200, 1200)
                    node.y_pos = base_y + random.uniform(-1200, 1200)
                else:
                    node.x_pos = base_x + random.uniform(-500, 500)
                    node.y_pos = base_y + random.uniform(-500, 500)

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
