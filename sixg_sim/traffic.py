"""
Traffic generation and modeling for network simulation.

Defines traffic arrival patterns and models for different traffic classes.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .topology import TrafficClass


@dataclass
class TrafficProfile:
    """Traffic profile for a specific class at a node."""
    baseline_rate: float  # Base arrival rate per tick
    burst_probability: float = 0.1  # Probability of burst events
    burst_multiplier: float = 2.0   # Multiplier during bursts
    surge_events: List[Dict[str, Any]] = None  # Scheduled surge events

    def __post_init__(self):
        if self.surge_events is None:
            self.surge_events = []

    def get_arrival_rate(self, tick: int) -> float:
        """Get arrival rate at given tick, considering surges."""
        rate = self.baseline_rate

        # Check for active surge events
        for surge in self.surge_events:
            if surge['start_tick'] <= tick <= surge['end_tick']:
                rate *= surge.get('multiplier', 2.0)
                break

        # Add random bursts
        if np.random.random() < self.burst_probability:
            rate *= self.burst_multiplier

        return rate


@dataclass
class NodeTrafficProfile:
    """Traffic profile for all classes at a node."""
    node_id: str
    profiles: Dict[TrafficClass, TrafficProfile]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NodeTrafficProfile':
        """Create from configuration dictionary."""
        profiles = {}
        for class_name, profile_config in config.items():
            if class_name == 'node_id':
                continue
            traffic_class = TrafficClass(class_name)
            profiles[traffic_class] = TrafficProfile(
                baseline_rate=profile_config['baseline_rate'],
                burst_probability=profile_config.get('burst_probability', 0.1),
                burst_multiplier=profile_config.get('burst_multiplier', 2.0),
                surge_events=profile_config.get('surge_events', [])
            )
        return cls(node_id=config['node_id'], profiles=profiles)


class TrafficGenerator:
    """Generates traffic arrivals for the network."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.node_profiles: Dict[str, NodeTrafficProfile] = {}

    def add_node_profile(self, profile: NodeTrafficProfile):
        """Add traffic profile for a node."""
        self.node_profiles[profile.node_id] = profile

    def generate_traffic(self, tick: int) -> Dict[str, Dict[TrafficClass, float]]:
        """
        Generate traffic arrivals for all nodes at given tick.

        Returns: {node_id: {traffic_class: arrival_amount}}
        """
        arrivals = {}

        for node_id, profile in self.node_profiles.items():
            node_arrivals = {}
            for traffic_class, class_profile in profile.profiles.items():
                rate = class_profile.get_arrival_rate(tick)
                # Use Poisson distribution for arrivals
                arrival_amount = self.rng.poisson(rate)
                node_arrivals[traffic_class] = float(arrival_amount)
            arrivals[node_id] = node_arrivals

        return arrivals


@dataclass
class QoSProfile:
    """QoS characteristics for a traffic class."""
    target_delay: int  # Target delay in ticks
    reliability_target: float  # Target reliability (0.0 to 1.0)
    priority: int  # Priority level (higher = more important)
    preemption_allowed: bool  # Can preempt other traffic
    admission_policy: str  # "always_admit", "throttle_under_stress", "hold"

    @classmethod
    def default_profiles(cls) -> Dict[TrafficClass, 'QoSProfile']:
        """Get default QoS profiles for all traffic classes."""
        return {
            TrafficClass.LIFE_SAFETY: cls(
                target_delay=5,
                reliability_target=0.99,
                priority=4,
                preemption_allowed=True,
                admission_policy="always_admit"
            ),
            TrafficClass.OPERATIONS: cls(
                target_delay=10,
                reliability_target=0.95,
                priority=3,
                preemption_allowed=False,
                admission_policy="always_admit"
            ),
            TrafficClass.TELEMETRY: cls(
                target_delay=50,
                reliability_target=0.90,
                priority=2,
                preemption_allowed=False,
                admission_policy="throttle_under_stress"
            ),
            TrafficClass.BEST_EFFORT: cls(
                target_delay=100,
                reliability_target=0.80,
                priority=1,
                preemption_allowed=False,
                admission_policy="hold"
            )
        }


@dataclass
class SliceDictionary:
    """Semantic slice dictionary used by agents for decision making."""
    qos_profiles: Dict[TrafficClass, QoSProfile]

    def __init__(self, qos_profiles: Optional[Dict[TrafficClass, QoSProfile]] = None):
        if qos_profiles is None:
            self.qos_profiles = QoSProfile.default_profiles()
        else:
            self.qos_profiles = qos_profiles

    def get_importance_score(self, traffic_class: TrafficClass) -> float:
        """Get importance score for a traffic class (0.0 to 1.0)."""
        profile = self.qos_profiles[traffic_class]
        return profile.priority / 4.0  # Normalize to 0-1

    def should_admit_under_stress(self, traffic_class: TrafficClass) -> bool:
        """Check if traffic class should be admitted under network stress."""
        profile = self.qos_profiles[traffic_class]
        return profile.admission_policy in ["always_admit", "throttle_under_stress"]

    def can_preempt(self, traffic_class: TrafficClass) -> bool:
        """Check if traffic class can preempt others."""
        return self.qos_profiles[traffic_class].preemption_allowed
