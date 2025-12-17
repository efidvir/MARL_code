"""
Scenario definition and loading for network simulation.

Defines events, traffic patterns, and scenario configurations.
"""

import yaml
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from .traffic import TrafficGenerator, NodeTrafficProfile
from .topology import Topology, NodeType


@dataclass
class ScenarioEvent:
    """An event that occurs at a specific tick in the simulation."""
    tick: int
    event_type: str
    parameters: Dict[str, Any]

    def __post_init__(self):
        """Validate event type."""
        valid_types = [
            'sever_core',           # Cut all links to core
            'fail_link',            # Set specific link to down
            'restore_link',         # Set specific link to up
            'energy_depletion',     # Force energy depletion at node
            'traffic_surge',        # Traffic surge at node
            'node_failure',         # Mark node as non-survivor
            'node_recovery'         # Mark node as survivor
        ]
        if self.event_type not in valid_types:
            raise ValueError(f"Invalid event type: {self.event_type}")


@dataclass
class Scenario:
    """Complete scenario definition."""
    name: str
    duration_ticks: int
    events: List[ScenarioEvent]
    traffic_profiles: Dict[str, NodeTrafficProfile]  # node_id -> profile
    description: Optional[str] = None

    def get_events_at_tick(self, tick: int) -> List[ScenarioEvent]:
        """Get all events scheduled for a specific tick."""
        return [event for event in self.events if event.tick == tick]


def load_scenario_from_yaml(file_path: str, topology: Topology) -> Scenario:
    """Load scenario from YAML file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load events
    events = []
    for event_config in config.get('events', []):
        event = ScenarioEvent(
            tick=event_config['tick'],
            event_type=event_config['type'],
            parameters=event_config.get('parameters', {})
        )
        events.append(event)

    # Load traffic profiles
    traffic_profiles = {}
    traffic_config = config.get('traffic_profiles', {})

    for node_id, node_config in traffic_config.items():
        if node_id not in topology.nodes:
            continue  # Skip profiles for nodes not in topology

        profile = NodeTrafficProfile.from_config({
            'node_id': node_id,
            **node_config
        })
        traffic_profiles[node_id] = profile

    return Scenario(
        name=config.get('name', 'Unnamed Scenario'),
        duration_ticks=config['duration_ticks'],
        events=events,
        traffic_profiles=traffic_profiles,
        description=config.get('description')
    )


def load_scenario_from_json(file_path: str, topology: Topology) -> Scenario:
    """Load scenario from JSON file."""
    with open(file_path, 'r') as f:
        config = json.load(f)

    # Load events
    events = []
    for event_config in config.get('events', []):
        event = ScenarioEvent(
            tick=event_config['tick'],
            event_type=event_config['type'],
            parameters=event_config.get('parameters', {})
        )
        events.append(event)

    # Load traffic profiles
    traffic_profiles = {}
    traffic_config = config.get('traffic_profiles', {})

    for node_id, node_config in traffic_config.items():
        if node_id not in topology.nodes:
            continue

        profile = NodeTrafficProfile.from_config({
            'node_id': node_id,
            **node_config
        })
        traffic_profiles[node_id] = profile

    return Scenario(
        name=config.get('name', 'Unnamed Scenario'),
        duration_ticks=config['duration_ticks'],
        events=events,
        traffic_profiles=traffic_profiles,
        description=config.get('description')
    )


def create_traffic_generator(scenario: Scenario, seed: Optional[int] = None) -> TrafficGenerator:
    """Create traffic generator from scenario."""
    generator = TrafficGenerator(seed=seed)

    for profile in scenario.traffic_profiles.values():
        generator.add_node_profile(profile)

    return generator


# Example scenario configurations
EXAMPLE_SCENARIO_CONFIG = {
    'name': 'Basic Severance Test',
    'description': 'Test basic island-mode operation after core severance',
    'duration_ticks': 500,
    'events': [
        {
            'tick': 100,
            'type': 'sever_core',
            'parameters': {}
        },
        {
            'tick': 200,
            'type': 'traffic_surge',
            'parameters': {
                'node_id': 'gNB1',
                'duration': 50,
                'multiplier': 3.0
            }
        },
        {
            'tick': 300,
            'type': 'fail_link',
            'parameters': {
                'link_id': 'L2'
            }
        }
    ],
    'traffic_profiles': {
        'gNB1': {
            'life_safety': {
                'baseline_rate': 5.0,
                'surge_events': []
            },
            'operations': {
                'baseline_rate': 10.0,
                'surge_events': []
            },
            'telemetry': {
                'baseline_rate': 15.0,
                'surge_events': []
            },
            'best_effort': {
                'baseline_rate': 20.0,
                'surge_events': []
            }
        },
        'gNB2': {
            'life_safety': {
                'baseline_rate': 3.0,
                'surge_events': []
            },
            'operations': {
                'baseline_rate': 8.0,
                'surge_events': []
            },
            'telemetry': {
                'baseline_rate': 12.0,
                'surge_events': []
            },
            'best_effort': {
                'baseline_rate': 15.0,
                'surge_events': []
            }
        }
    }
}
