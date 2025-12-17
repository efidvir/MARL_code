"""
Control plane implementation for network simulation.

Handles control overlays (IP overlay and Disaster Control Channel) and
enforces constraints on control message exchange between agents.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from .topology import Topology, Node, TrafficClass
from .agent import ControlPostcard, NeighborSummary, StrainLevel, EnergyTier


@dataclass
class ControlMessage:
    """A control message in transit."""
    postcard: ControlPostcard
    source_node: str
    target_node: str
    delivery_tick: int  # When message will be delivered


class ControlOverlay:
    """Base class for control overlays."""

    def __init__(self, topology: Topology):
        self.topology = topology

    def can_exchange_messages(self, node_a: str, node_b: str) -> bool:
        """Check if two nodes can exchange control messages."""
        raise NotImplementedError

    def get_message_latency(self, source: str, target: str) -> int:
        """Get latency for message delivery between nodes."""
        raise NotImplementedError


class IPOverlay(ControlOverlay):
    """
    IP overlay control plane.

    Allows normal control message exchange when IP connectivity exists.
    """

    def can_exchange_messages(self, node_a: str, node_b: str) -> bool:
        """Check if IP path exists between nodes."""
        return self.topology.has_path(node_a, node_b)

    def get_message_latency(self, source: str, target: str) -> int:
        """Get IP routing latency (simplified)."""
        # Simple latency based on hop count
        try:
            path = self.topology.get_shortest_path(source, target)
            if path:
                return len(path) - 1  # Number of hops
            return 1000  # Very high if no path
        except:
            return 1000


class DisasterControlChannel(ControlOverlay):
    """
    Disaster Control Channel (DCC) for island mode.

    Very limited bandwidth control channel between neighboring nodes.
    """

    def __init__(self, topology: Topology, max_neighbors: int = 3,
                 max_rate_per_tick: float = 1.0):
        super().__init__(topology)
        self.max_neighbors = max_neighbors  # Max neighbors per node
        self.max_rate_per_tick = max_rate_per_tick  # Max postcards per tick per node
        self.message_queues: Dict[str, List[ControlMessage]] = defaultdict(list)
        self.sent_this_tick: Dict[str, int] = defaultdict(int)

    def can_exchange_messages(self, node_a: str, node_b: str) -> bool:
        """Check if DCC allows message exchange between neighbors."""
        neighbors_a = set(self.topology.get_neighbors(node_a))
        return node_b in neighbors_a and len(neighbors_a) <= self.max_neighbors

    def get_message_latency(self, source: str, target: str) -> int:
        """Get DCC latency (very low, but bounded)."""
        if self.can_exchange_messages(source, target):
            return 1  # Minimal latency for direct neighbors
        return 1000  # No direct connection

    def send_postcard(self, postcard: ControlPostcard, current_tick: int) -> bool:
        """Attempt to send a postcard via DCC."""
        sender = postcard.sender_id

        # Check rate limit
        if self.sent_this_tick[sender] >= self.max_rate_per_tick:
            return False

        # Send to eligible neighbors
        neighbors = self.topology.get_neighbors(sender)
        eligible_neighbors = [n for n in neighbors if self.can_exchange_messages(sender, n)]

        sent_count = 0
        for neighbor in eligible_neighbors[:self.max_neighbors]:
            message = ControlMessage(
                postcard=postcard,
                source_node=sender,
                target_node=neighbor,
                delivery_tick=current_tick + self.get_message_latency(sender, neighbor)
            )
            self.message_queues[neighbor].append(message)
            sent_count += 1

        if sent_count > 0:
            self.sent_this_tick[sender] += 1
            return True
        return False

    def receive_messages(self, node_id: str, current_tick: int) -> List[ControlPostcard]:
        """Get all messages delivered to node at current tick."""
        delivered = []
        queue = self.message_queues[node_id]

        # Find delivered messages
        remaining = []
        for message in queue:
            if message.delivery_tick <= current_tick:
                delivered.append(message.postcard)
            else:
                remaining.append(message)

        self.message_queues[node_id] = remaining
        return delivered

    def reset_tick_counters(self):
        """Reset per-tick counters."""
        self.sent_this_tick.clear()


class ControlPlaneManager:
    """Manages control overlays and message exchange."""

    def __init__(self, topology: Topology):
        self.topology = topology
        self.ip_overlay = IPOverlay(topology)
        self.dcc = DisasterControlChannel(topology)
        self.is_island_mode = False

    def set_island_mode(self, island_mode: bool):
        """Set whether network is in island mode."""
        self.is_island_mode = island_mode

    def can_send_control_message(self, source: str, target: str) -> bool:
        """Check if control message can be sent from source to target."""
        if not self.is_island_mode:
            return self.ip_overlay.can_exchange_messages(source, target)
        else:
            return self.dcc.can_exchange_messages(source, target)

    def send_postcard(self, postcard: ControlPostcard, current_tick: int) -> bool:
        """Send a postcard via appropriate overlay."""
        if not self.is_island_mode:
            # In normal mode, postcards are sent via IP overlay (simplified)
            return True  # Assume success for normal mode
        else:
            return self.dcc.send_postcard(postcard, current_tick)

    def get_received_postcards(self, node_id: str, current_tick: int) -> List[ControlPostcard]:
        """Get postcards received by node at current tick."""
        if not self.is_island_mode:
            return []  # No postcards in normal mode for this simulation
        else:
            return self.dcc.receive_messages(node_id, current_tick)

    def fuse_neighbor_summaries(self, node_id: str, received_postcards: List[ControlPostcard],
                               current_tick: int) -> NeighborSummary:
        """Fuse information from neighbor postcards into summary."""
        if not received_postcards:
            # Default summary when no postcards received
            return NeighborSummary(
                most_needy_class=TrafficClass.BEST_EFFORT,
                need_level=StrainLevel.OKAY,
                strain_level=StrainLevel.OKAY,
                latest_policy_version=0,
                neighbor_count=0
            )

        # Analyze received postcards
        neighbor_count = len(set(p.sender_id for p in received_postcards))

        # Find most common needy class
        needy_classes = [p.most_needy_class for p in received_postcards]
        most_needy_class = max(set(needy_classes), key=needy_classes.count)

        # Find highest need level
        need_levels = [p.need_level for p in received_postcards]
        highest_need = max(need_levels, key=lambda x: ['okay', 'degrading', 'near_limit'].index(x.value))

        # Determine overall strain
        strain_counts = {'okay': 0, 'degrading': 0, 'near_limit': 0}
        for level in need_levels:
            strain_counts[level.value] += 1

        # Strain is high if majority are degrading or near limit
        if strain_counts['near_limit'] > neighbor_count / 2:
            strain_level = StrainLevel.NEAR_LIMIT
        elif strain_counts['degrading'] + strain_counts['near_limit'] > neighbor_count / 2:
            strain_level = StrainLevel.DEGRADING
        else:
            strain_level = StrainLevel.OKAY

        # Get latest policy version
        latest_version = max((p.policy_version for p in received_postcards), default=0)

        return NeighborSummary(
            most_needy_class=most_needy_class,
            need_level=highest_need,
            strain_level=strain_level,
            latest_policy_version=latest_version,
            neighbor_count=neighbor_count
        )

    def reset_for_tick(self):
        """Reset control plane state for new tick."""
        self.dcc.reset_tick_counters()
