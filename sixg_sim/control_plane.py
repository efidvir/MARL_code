"""
Control plane implementation for network simulation.

Handles control overlays (IP overlay and Disaster Control Channel) and
enforces constraints on control message exchange between agents.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import random
from .topology import Topology, Node, NodeType, TrafficClass
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
    IP overlay control plane with realistic disaster degradation.

    Normal mode: Reliable IP connectivity
    Disaster mode: Degraded reliability, congestion, and priority-based delivery
    """

    def __init__(self, topology: Topology):
        super().__init__(topology)
        self.is_disaster_mode = False
        self.disaster_start_tick = 0
        self.message_failure_log = []  # Track failed transmissions for visualization

    def set_disaster_mode(self, disaster_mode: bool, current_tick: int = 0):
        """Enable disaster mode with degraded IP reliability."""
        self.is_disaster_mode = disaster_mode
        if disaster_mode and self.disaster_start_tick == 0:
            self.disaster_start_tick = current_tick

    def can_exchange_messages(self, node_a: str, node_b: str) -> bool:
        """Check if IP path exists between nodes."""
        return self.topology.has_path(node_a, node_b)

    def get_message_latency(self, source: str, target: str) -> int:
        """Get IP routing latency (increases during disasters)."""
        try:
            path = self.topology.get_shortest_path(source, target)
            if not path:
                return 1000

            base_latency = len(path) - 1  # Number of hops

            if self.is_disaster_mode:
                # Disaster degradation: increased latency, jitter
                disaster_time = max(1, 100)  # Assume current tick context
                degradation_factor = min(5.0, 1.0 + (disaster_time / 100.0))  # Up to 5x latency
                base_latency = int(base_latency * degradation_factor)
                # Add jitter (±50% during disasters)
                jitter = random.uniform(0.5, 1.5)
                base_latency = int(base_latency * jitter)

            return max(1, base_latency)
        except:
            return 1000

    def attempt_message_delivery(self, source: str, target: str, message_priority: str = "normal") -> bool:
        """
        Attempt to deliver a message with realistic failure modes during disasters.

        Priority levels: "critical" (life-safety), "high" (operations), "normal" (other)
        """
        # Handle broadcast case (target="broadcast")
        if target == "broadcast":
            # In broadcast, we try to reach multiple neighbors and succeed if at least one works
            neighbors = [n for n in self.topology.get_neighbors(source) if self.can_exchange_messages(source, n)]
            if not neighbors:
                self._log_failed_transmission(source, target, "no_neighbors", message_priority)
                return False

            # Try to reach at least one neighbor
            success_count = 0
            for neighbor in neighbors[:3]:  # Try up to 3 neighbors
                if self._attempt_direct_delivery(source, neighbor, message_priority):
                    success_count += 1

            success = success_count > 0
            if not success:
                self._log_failed_transmission(source, target, "all_broadcast_failed", message_priority)
            return success

        # Direct message delivery
        return self._attempt_direct_delivery(source, target, message_priority)

    def _attempt_direct_delivery(self, source: str, target: str, message_priority: str) -> bool:
        """Attempt direct message delivery between two nodes."""
        if not self.can_exchange_messages(source, target):
            self._log_failed_transmission(source, target, "no_path", message_priority)
            return False

        if not self.is_disaster_mode:
            return True  # Normal mode: assume success

        # Disaster mode: realistic failure probabilities
        disaster_duration = max(1, 100)  # Assume current tick context
        base_failure_rate = min(0.3, disaster_duration / 200.0)  # Up to 30% base failure rate

        # Priority affects success rate
        priority_multiplier = {
            "critical": 0.1,   # 10% of base failure rate for life-safety
            "high": 0.3,       # 30% of base failure rate for operations
            "normal": 1.0      # Full failure rate for other messages
        }

        effective_failure_rate = base_failure_rate * priority_multiplier.get(message_priority, 1.0)

        # Network congestion: higher failure rate as disaster progresses
        congestion_factor = min(2.0, 1.0 + (disaster_duration / 300.0))
        effective_failure_rate *= congestion_factor

        # Distance affects reliability (longer paths more likely to fail)
        path_length = len(self.topology.get_shortest_path(source, target) or [])
        distance_factor = 1.0 + (path_length - 1) * 0.1  # 10% additional failure per hop
        effective_failure_rate *= distance_factor

        # Final success check
        success = random.random() > effective_failure_rate

        if not success:
            failure_reason = "congestion" if random.random() < 0.6 else "timeout"
            self._log_failed_transmission(source, target, failure_reason, message_priority)

        return success

    def _log_failed_transmission(self, source: str, target: str, reason: str, priority: str):
        """Log failed transmission for visualization."""
        self.message_failure_log.append({
            'source': source,
            'target': target,
            'reason': reason,
            'priority': priority,
            'tick': 0  # Will be set by caller
        })
        # Keep only recent failures for memory efficiency
        if len(self.message_failure_log) > 1000:
            self.message_failure_log = self.message_failure_log[-500:]


class DisasterControlChannel(ControlOverlay):
    """
    Disaster Control Channel (DCC) for island mode (RERUN_SPEC WP5; the
    constants and the model are in sixg_sim/postcards.py).

    A point-to-point control channel over the infrastructure transport hops
    (fibre, microwave, 60 GHz mesh and relay links).  A postcard is sent as
    ONE FRAME PER LIVE INFRASTRUCTURE LINK incident to the sender
    (`Link.is_up`; UE access links carry no DCC), with no fan-out cap.  A
    frame is enqueued at its far end only when that node hosts an agent --
    no other node reads a DCC queue -- but every frame is billed.  Latency is
    one tick.

    The per-site limiter (at most `max_rate_per_tick` sends per site per
    tick) is keyed on the CURRENT TICK and resets itself on the first send of
    a new tick.  It used to be cleared only by `reset_tick_counters`, which
    only `Simulator.run_simulation` calls and no driver of the study runs, so
    each site sent ONE postcard per episode.  `reset_tick_counters` is kept
    and is now merely idempotent.

    The per-tick LEDGER (sends, frames, bytes, deliveries, refused) is what
    the link-state service bills as the 'postcard' message type.
    """

    def __init__(self, topology: Topology, max_rate_per_tick: float = 1.0):
        super().__init__(topology)
        from .postcards import POSTCARD_ENCODED_BYTES
        self.max_rate_per_tick = max_rate_per_tick  # Max postcards per tick per node
        self.frame_bytes = POSTCARD_ENCODED_BYTES
        self.message_queues: Dict[str, List[ControlMessage]] = defaultdict(list)
        self.sent_this_tick: Dict[str, int] = defaultdict(int)
        # The tick the limiter counters and the ledger belong to.
        self._counter_tick: Optional[int] = None
        self._tick_ledger: Dict[str, int] = self._zero_ledger()
        # Cumulative totals since construction (never reset).
        self.total_ledger: Dict[str, int] = self._zero_ledger()

    @staticmethod
    def _zero_ledger() -> Dict[str, int]:
        return {'sends': 0, 'frames': 0, 'bytes': 0, 'deliveries': 0,
                'refused': 0}

    def _roll_tick(self, current_tick: int) -> None:
        """Self-reset: the first DCC event of a new tick clears the per-site
        send counters and the per-tick ledger."""
        if current_tick != self._counter_tick:
            self.sent_this_tick.clear()
            self._tick_ledger = self._zero_ledger()
            self._counter_tick = current_tick

    def ledger(self, tick: int) -> Dict[str, int]:
        """The ledger of `tick`: what was sent on that tick (all zero for a
        tick on which nothing was sent)."""
        if tick != self._counter_tick:
            return self._zero_ledger()
        return dict(self._tick_ledger)

    def _is_ue(self, node_id: str) -> bool:
        """Return True if node_id is a UE (UEs do not participate in the DCC)."""
        node = self.topology.nodes.get(node_id)
        return node is not None and node.node_type == NodeType.UE

    def live_infra_links(self, node_id: str) -> List[Tuple[str, str]]:
        """[(link_id, far end)] for every LIVE infrastructure link incident to
        `node_id`, in `topology.links` insertion order: the links a postcard
        from `node_id` goes out on, one frame each.  Parallel links to one
        neighbour are separate interfaces and each carries a frame."""
        if self._is_ue(node_id):
            return []
        out = []
        for lid, link in self.topology.links.items():
            if not getattr(link, 'is_up', False):
                continue
            a, b = link.endpoints
            if a == node_id:
                peer = b
            elif b == node_id:
                peer = a
            else:
                continue
            if peer == node_id or self._is_ue(peer):
                continue
            out.append((lid, peer))
        return out

    def can_exchange_messages(self, node_a: str, node_b: str) -> bool:
        """True iff a live infrastructure link joins the two nodes (the DCC
        runs over infrastructure hops only; UE access links carry none)."""
        return any(peer == node_b for _, peer in self.live_infra_links(node_a))

    def get_message_latency(self, source: str, target: str) -> int:
        """Get DCC latency (very low, but bounded)."""
        if self.can_exchange_messages(source, target):
            return 1  # Minimal latency for direct neighbors
        return 1000  # No direct connection

    def send_postcard(self, postcard: ControlPostcard, current_tick: int,
                      receivers=None) -> bool:
        """Send one postcard: one frame per live infrastructure link of the
        sender, enqueued (delivery next tick) at every far end that is in
        `receivers` (the agent-hosting sites; None = every far end).

        Returns True when the limiter ADMITTED the send -- also when the
        sender has no live infrastructure link, in which case no frame
        leaves and the ledger says so (frames 0) -- and False only when the
        site already sent `max_rate_per_tick` postcards this tick."""
        self._roll_tick(current_tick)
        sender = postcard.sender_id

        # Rate limit: at most max_rate_per_tick sends per site per tick.
        if self.sent_this_tick[sender] >= self.max_rate_per_tick:
            self._tick_ledger['refused'] += 1
            self.total_ledger['refused'] += 1
            return False
        self.sent_this_tick[sender] += 1

        frames = 0
        deliveries = 0
        for _lid, peer in self.live_infra_links(sender):
            frames += 1
            if receivers is not None and peer not in receivers:
                continue
            self.message_queues[peer].append(ControlMessage(
                postcard=postcard,
                source_node=sender,
                target_node=peer,
                delivery_tick=current_tick + 1,
            ))
            deliveries += 1

        for led in (self._tick_ledger, self.total_ledger):
            led['sends'] += 1
            led['frames'] += frames
            led['bytes'] += frames * self.frame_bytes
            led['deliveries'] += deliveries
        return True

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
        """Reset per-tick counters.  Idempotent and no longer needed: the
        counters reset themselves on the first send of a new tick
        (_roll_tick).  Kept for Simulator.run_simulation."""
        self.sent_this_tick.clear()


class ControlPlaneManager:
    """Manages control overlays and message exchange with realistic disaster behavior."""

    def __init__(self, topology: Topology):
        self.topology = topology
        self.ip_overlay = IPOverlay(topology)
        self.dcc = DisasterControlChannel(topology)
        self.is_island_mode = False
        self.is_disaster_mode = False  # Separate from island mode - can have disaster without full island

    def set_island_mode(self, island_mode: bool):
        """Set whether network is in island mode."""
        self.is_island_mode = island_mode

    def set_disaster_mode(self, disaster_mode: bool, current_tick: int = 0):
        """Set disaster mode affecting IP overlay reliability."""
        self.is_disaster_mode = disaster_mode
        self.ip_overlay.set_disaster_mode(disaster_mode, current_tick)

    def can_send_control_message(self, source: str, target: str) -> bool:
        """Check if control message can be sent from source to target."""
        if not self.is_island_mode:
            return self.ip_overlay.can_exchange_messages(source, target)
        else:
            return self.dcc.can_exchange_messages(source, target)

    def send_postcard(self, postcard: ControlPostcard, current_tick: int,
                      receivers=None) -> bool:
        """Send a postcard via appropriate overlay with realistic failure modes.

        In island mode -- the only mode in which the study's drivers send
        postcards: a site's action lock opens on its own local-isolation
        flag (D13.2), which in the study's scenarios is never set before the
        global trigger (tests/test_wp6_smoke.py counts it) -- this is the DCC
        (one frame per live infrastructure link; `receivers` = the
        agent-hosting sites).  The IP-overlay branch is reached only by the
        legacy drivers."""
        if not self.is_island_mode:
            # Normal mode: Use IP overlay with disaster degradation if applicable
            priority = self._get_message_priority(postcard)
            return self.ip_overlay.attempt_message_delivery(
                postcard.sender_id, "broadcast", priority  # Broadcast to all reachable nodes
            )
        else:
            # Island mode: Use DCC (limited to neighbors)
            success = self.dcc.send_postcard(postcard, current_tick,
                                             receivers=receivers)
            if not success:
                # Log DCC failure too
                self.ip_overlay._log_failed_transmission(
                    postcard.sender_id, "neighbors", "dcc_limit", "high"
                )
            return success

    def _get_message_priority(self, postcard: ControlPostcard) -> str:
        """Determine message priority for delivery."""
        if postcard.need_level == StrainLevel.NEAR_LIMIT:
            return "critical"  # Life-safety level coordination needed
        elif postcard.need_level == StrainLevel.DEGRADING:
            return "high"  # Operations level coordination
        else:
            return "normal"  # Routine coordination

    def get_received_postcards(self, node_id: str, current_tick: int) -> List[ControlPostcard]:
        """Get postcards received by node at current tick."""
        if not self.is_island_mode:
            return []  # No postcards in normal mode for this simulation
        else:
            return self.dcc.receive_messages(node_id, current_tick)

    def fuse_neighbor_summaries(self, node_id: str, received_postcards: List[ControlPostcard],
                               current_tick: int) -> NeighborSummary:
        """Fuse information from neighbor postcards into summary, accounting for communication failures."""
        if not received_postcards:
            # No postcards received - could be due to communication failures
            # In disaster mode, this indicates poor connectivity
            default_strain = StrainLevel.DEGRADING if self.is_disaster_mode else StrainLevel.OKAY
            return NeighborSummary(
                most_needy_class=TrafficClass.BEST_EFFORT,
                need_level=default_strain,
                strain_level=default_strain,
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

    def get_failed_transmissions(self, since_tick: int = 0) -> List[Dict]:
        """Get failed transmission log for visualization."""
        return [f for f in self.ip_overlay.message_failure_log if f.get('tick', 0) >= since_tick]

    def update_failure_log_ticks(self, current_tick: int):
        """Update tick information in failure log."""
        for failure in self.ip_overlay.message_failure_log:
            if 'tick' not in failure or failure['tick'] == 0:
                failure['tick'] = current_tick

    def reset_for_tick(self):
        """Reset control plane state for new tick."""
        self.dcc.reset_tick_counters()
