"""
Agent implementation for autonomous network optimization.

Defines the agent interface and a heuristic agent that implements
rule-based policies for island-mode operation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
from .topology import TrafficClass, Node
from .traffic import SliceDictionary


class AdmissionMode(Enum):
    """Admission modes for traffic classes."""
    ADMIT = "admit"
    THROTTLE = "throttle"
    HOLD = "hold"


class EnergyTier(Enum):
    """Energy state tiers."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StrainLevel(Enum):
    """Network strain levels."""
    OKAY = "okay"
    DEGRADING = "degrading"
    NEAR_LIMIT = "near_limit"


@dataclass
class LocalSliceState:
    """Local slice state for agent observation."""
    traffic_class: TrafficClass
    importance: float  # 0.0 to 1.0
    current_queue_length: float
    offered_load: float
    admission_success_rate: float  # Fraction successfully admitted
    freshness_target: int  # Target delay in ticks
    current_freshness: float  # Current average delay proxy


@dataclass
class NeighborSummary:
    """Summary of neighbor states."""
    most_needy_class: TrafficClass
    need_level: StrainLevel
    strain_level: StrainLevel  # Overall neighbor strain
    latest_policy_version: int
    neighbor_count: int


@dataclass
class AgentObservation:
    """Complete observation structure for agent decision making."""
    node_id: str
    is_island: bool
    energy_tier: EnergyTier
    local_slices: Dict[TrafficClass, LocalSliceState]
    neighbor_summary: NeighborSummary
    current_tick: int


@dataclass
class ClassAction:
    """Action for a specific traffic class."""
    admission_mode: AdmissionMode
    priority_weight: float  # Relative weight for scheduling (0.0 to 2.0)


@dataclass
class LinkBias:
    """Routing bias for a link."""
    link_id: str
    bias_score: float  # Higher = prefer this link (0.0 to 2.0)


@dataclass
class ControlPostcard:
    """Limited control message sent to neighbors."""
    sender_id: str
    most_needy_class: TrafficClass
    need_level: StrainLevel
    policy_version: int
    timestamp: int


@dataclass
class AgentAction:
    """Complete action structure from agent."""
    class_actions: Dict[TrafficClass, ClassAction]
    link_biases: Dict[str, LinkBias]  # link_id -> bias
    send_postcard: bool
    postcard_content: Optional[ControlPostcard]


class BaseAgent(ABC):
    """Abstract base class for network agents."""

    def __init__(self, node_id: str, slice_dictionary: SliceDictionary):
        self.node_id = node_id
        self.slice_dictionary = slice_dictionary
        self.policy_version = 0

    @abstractmethod
    def compute_action(self, observation: AgentObservation) -> AgentAction:
        """Compute action based on current observation."""
        pass

    def update_policy_version(self):
        """Increment policy version when behavior changes."""
        self.policy_version += 1


class HeuristicAgent(BaseAgent):
    """
    Rule-based heuristic agent implementing island-mode protection policies.

    Always protects life-safety traffic, throttles telemetry and best-effort
    under stress, and biases routing away from strained neighbors.
    """

    def __init__(self, node_id: str, slice_dictionary: SliceDictionary):
        super().__init__(node_id, slice_dictionary)
        self.last_postcard_tick = -100  # Allow initial postcard

    def compute_action(self, observation: AgentObservation) -> AgentAction:
        """Compute heuristic action based on observation."""

        # Initialize actions
        class_actions = {}
        link_biases = {}

        # Determine stress level
        is_stressed = self._is_network_stressed(observation)

        # Set class actions based on priority and stress
        for traffic_class in TrafficClass:
            class_actions[traffic_class] = self._compute_class_action(
                traffic_class, observation, is_stressed
            )

        # Set link biases based on neighbor strain
        # Note: In real implementation, we'd have actual links to bias
        # For now, create placeholder biases
        link_biases = self._compute_link_biases(observation)

        # Decide whether to send postcard
        send_postcard = self._should_send_postcard(observation)
        postcard_content = None

        if send_postcard:
            postcard_content = ControlPostcard(
                sender_id=observation.node_id,
                most_needy_class=self._get_most_needy_class(observation),
                need_level=self._get_need_level(observation),
                policy_version=self.policy_version,
                timestamp=observation.current_tick
            )
            self.last_postcard_tick = observation.current_tick

        return AgentAction(
            class_actions=class_actions,
            link_biases=link_biases,
            send_postcard=send_postcard,
            postcard_content=postcard_content
        )

    def _is_network_stressed(self, observation: AgentObservation) -> bool:
        """Determine if network is under stress."""
        if observation.energy_tier == EnergyTier.LOW:
            return True

        if observation.is_island:
            return True

        # Check if any high-priority queues are building up
        life_safety_queue = observation.local_slices[TrafficClass.LIFE_SAFETY].current_queue_length
        operations_queue = observation.local_slices[TrafficClass.OPERATIONS].current_queue_length

        return life_safety_queue > 50 or operations_queue > 100

    def _compute_class_action(self, traffic_class: TrafficClass,
                            observation: AgentObservation, is_stressed: bool) -> ClassAction:
        """Compute action for a specific traffic class."""

        base_weight = self.slice_dictionary.get_importance_score(traffic_class)

        if traffic_class == TrafficClass.LIFE_SAFETY:
            # Always protect life-safety, boost priority in island mode
            weight = 2.0 if observation.is_island else 1.5
            mode = AdmissionMode.ADMIT

        elif traffic_class == TrafficClass.OPERATIONS:
            # Protect operations but allow some throttling
            weight = 1.2
            mode = AdmissionMode.ADMIT

        elif traffic_class == TrafficClass.TELEMETRY:
            if is_stressed:
                weight = 0.3
                mode = AdmissionMode.THROTTLE
            else:
                weight = 0.8
                mode = AdmissionMode.ADMIT

        elif traffic_class == TrafficClass.BEST_EFFORT:
            if is_stressed:
                weight = 0.1
                mode = AdmissionMode.HOLD
            else:
                weight = 0.5
                mode = AdmissionMode.THROTTLE

        return ClassAction(admission_mode=mode, priority_weight=weight)

    def _compute_link_biases(self, observation: AgentObservation) -> Dict[str, LinkBias]:
        """Compute routing biases for links."""
        # Placeholder implementation - in real scenario, we'd have actual link IDs
        # Bias away from strained neighbors
        biases = {}

        strain_multiplier = 1.0
        if observation.neighbor_summary.strain_level == StrainLevel.NEAR_LIMIT:
            strain_multiplier = 0.3
        elif observation.neighbor_summary.strain_level == StrainLevel.DEGRADING:
            strain_multiplier = 0.7

        # This would be populated with actual link IDs in real implementation
        # For now, return empty dict as placeholder
        return biases

    def _should_send_postcard(self, observation: AgentObservation) -> bool:
        """Decide whether to send a control postcard."""
        # Send postcard if:
        # 1. We haven't sent one recently (rate limiting)
        # 2. We're in island mode
        # 3. Network state has changed significantly

        ticks_since_last = observation.current_tick - self.last_postcard_tick
        min_interval = 10  # Minimum ticks between postcards

        if ticks_since_last < min_interval:
            return False

        # Always send in island mode if it's been long enough
        if observation.is_island:
            return True

        # Send if energy is low or neighbors are strained
        if (observation.energy_tier == EnergyTier.LOW or
            observation.neighbor_summary.strain_level in [StrainLevel.DEGRADING, StrainLevel.NEAR_LIMIT]):
            return True

        return False

    def _get_most_needy_class(self, observation: AgentObservation) -> TrafficClass:
        """Get the most needy traffic class based on current state."""
        # Find class with longest relative queue
        max_relative_queue = 0.0
        neediest_class = TrafficClass.LIFE_SAFETY

        for traffic_class, slice_state in observation.local_slices.items():
            # Normalize queue by expected capacity
            relative_queue = slice_state.current_queue_length / 100.0  # Assume capacity of 100
            importance = self.slice_dictionary.get_importance_score(traffic_class)
            weighted_need = relative_queue * importance

            if weighted_need > max_relative_queue:
                max_relative_queue = weighted_need
                neediest_class = traffic_class

        return neediest_class

    def _get_need_level(self, observation: AgentObservation) -> StrainLevel:
        """Get overall need level for postcard."""
        if observation.energy_tier == EnergyTier.LOW:
            return StrainLevel.NEAR_LIMIT

        life_safety_queue = observation.local_slices[TrafficClass.LIFE_SAFETY].current_queue_length
        if life_safety_queue > 75:
            return StrainLevel.NEAR_LIMIT
        elif life_safety_queue > 25:
            return StrainLevel.DEGRADING
        else:
            return StrainLevel.OKAY


def create_agent_for_node(node_id: str, node_type: str, slice_dictionary: SliceDictionary) -> BaseAgent:
    """Factory function to create appropriate agent for node type."""
    # For now, all nodes use the heuristic agent
    # Later, this could be extended to create different agent types
    return HeuristicAgent(node_id, slice_dictionary)
