"""
Scenario definition and loading for 6G network simulation.

Key additions vs original:
  - New event types: partial_sever, sever_zone, restore_core, geo_disaster
  - DiverseScenarioGenerator: domain-randomised scenario catalogue used each
    episode during MARL training to build a generalised, robust policy.
  - Curriculum schedule: episode 1-3 -> full_core only; 4-6 -> +partial_core;
    7-9 -> +zone_loss; 10+ -> +cascading + geo_disaster
  - geo_disaster: radius-based geographic destruction using x_pos/y_pos coords.
  - ScenarioGenerator kept as backward-compatible alias.
"""

import yaml
import json
import math
from typing import Dict, List, Optional, Any, Tuple
import random
from dataclasses import dataclass, field
from .traffic import TrafficGenerator, NodeTrafficProfile
from .topology import Topology


# ─────────────────────────────────────────────────────────────────────────────
# Core data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioEvent:
    """An event that occurs at a specific tick in the simulation."""
    tick: int
    event_type: str
    parameters: Dict[str, Any]

    _VALID_TYPES = {
        # Infrastructure failure
        'sever_core',            # Cut ALL links to/from every core node
        'partial_sever',         # Cut only a named subset of core nodes
        'sever_zone',            # Mark all nodes in a zone as non-survivor
        'geo_disaster',          # Radius-based geographic destruction
        # Recovery
        'restore_core',          # Re-enable core links + restore nodes
        # Link / node
        'fail_link',
        'restore_link',
        'energy_depletion',
        'node_failure',
        'node_recovery',
        # Traffic
        'traffic_surge',
        'ue_join',
        'ue_leave',
        'rescue_force_arrival',
        # 3GPP TS 22.179 MCPTT
        'mcppt_emergency_alert',
        'mcppt_emergency_call',
        # ETSI TS 22.346 V16.0.0 Multi-eNB IOPS
        'nenb_deployment',       # Rescue force deploys a Nomadic eNB
        'island_merge',          # Two islands connect via Transport relay
        'multi_enb_iops_init',   # Trigger Multi-eNB IOPS island formation
        'mcptt_group_call',      # MCPTT group call within island
    }

    def __post_init__(self):
        if self.event_type not in self._VALID_TYPES:
            raise ValueError(f"Invalid event type: '{self.event_type}'. "
                             f"Valid: {sorted(self._VALID_TYPES)}")


@dataclass
class Scenario:
    """Complete scenario definition."""
    name: str
    duration_ticks: int
    events: List[ScenarioEvent]
    traffic_profiles: Dict[str, NodeTrafficProfile]
    description: Optional[str] = None
    # One of: full_core | partial_core | zone_loss | cascading | multi_enb_iops
    scenario_type: str = 'full_core'

    def get_events_at_tick(self, tick: int) -> List[ScenarioEvent]:
        return [e for e in self.events if e.tick == tick]


# ─────────────────────────────────────────────────────────────────────────────
# YAML / JSON loaders (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario_from_yaml(file_path: str, topology: Topology) -> Scenario:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return _parse_scenario_config(config, topology)


def load_scenario_from_json(file_path: str, topology: Topology) -> Scenario:
    with open(file_path, 'r') as f:
        config = json.load(f)
    return _parse_scenario_config(config, topology)


def _parse_scenario_config(config: dict, topology: Topology) -> Scenario:
    events = [
        ScenarioEvent(
            tick=ec['tick'],
            event_type=ec['type'],
            parameters=ec.get('parameters', {})
        )
        for ec in config.get('events', [])
    ]

    traffic_profiles = {}
    for node_id, node_cfg in config.get('traffic_profiles', {}).items():
        if node_id not in topology.nodes:
            continue
        traffic_profiles[node_id] = NodeTrafficProfile.from_config(
            {'node_id': node_id, **node_cfg}
        )

    return Scenario(
        name=config.get('name', 'Unnamed Scenario'),
        duration_ticks=config['duration_ticks'],
        events=events,
        traffic_profiles=traffic_profiles,
        description=config.get('description'),
        scenario_type=config.get('scenario_type', 'full_core'),
    )


def create_traffic_generator(scenario: Scenario,
                             seed: Optional[int] = None) -> TrafficGenerator:
    gen = TrafficGenerator(seed=seed)
    for profile in scenario.traffic_profiles.values():
        gen.add_node_profile(profile)
    return gen


# ─────────────────────────────────────────────────────────────────────────────
# Zone inference (inferred from node-ID prefix — no YAML change required)
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_ZONES = ('north', 'south', 'east', 'west')


def _infer_zone(node_id: str) -> Optional[str]:
    """'relay_north_3' -> 'north'  |  'Core' -> None"""
    nid = node_id.lower()
    for z in _KNOWN_ZONES:
        if z in nid:
            return z
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum schedule
# (lo, hi inclusive, list of scenario types allowed in that episode range)
# ─────────────────────────────────────────────────────────────────────────────

_CURRICULUM: List[tuple] = [
    (1,   3,  ['full_core']),
    (4,   6,  ['full_core', 'zone_loss']),
    (7,   9,  ['full_core', 'zone_loss', 'multi_enb_iops']),
    (10, 999, ['full_core', 'zone_loss', 'cascading', 'multi_enb_iops',
              'geo_disaster']),
]

_TYPE_WEIGHTS = {
    'full_core':       0.20,
    'zone_loss':       0.20,
    'cascading':       0.15,
    'multi_enb_iops':  0.20,
    'geo_disaster':    0.25,
}


# ─────────────────────────────────────────────────────────────────────────────
# Diverse Scenario Generator
# ─────────────────────────────────────────────────────────────────────────────

class DiverseScenarioGenerator:
    """
    Domain-randomised scenario generator for 6G MARL disaster-recovery training.

    Each call to generate(episode, seed) produces a unique Scenario object
    drawn from the curriculum-appropriate catalogue.

    Scenario archetypes
    -------------------
    full_core    : ALL Core + EdgeUPF severed at once (hardest for agents)
    partial_core : Only Core nodes down; 1-2 EdgeUPF survive
    zone_loss    : One geographic zone + core fully lost
    cascading    : Core lost first, then relay nodes fail one-by-one

    Recovery
    --------
    From episode 3 onward, 70% of episodes include a restore_core event so
    that agents learn graceful handover back to normal mode.
    """

    SCENARIO_TYPES = ('full_core', 'zone_loss', 'cascading', 'multi_enb_iops',
                      'geo_disaster')

    def __init__(self, topology: Topology, base_duration: int = 800,
                 curriculum_start: int = 4):
        self.topology      = topology
        self.base_duration = base_duration

        # Pre-cache node lists by role
        self._core_nodes = [
            nid for nid, n in topology.nodes.items()
            if n.node_type.value in ('Core', 'AMF', 'SMF', 'UPF',
                                     'Non-RT-RIC', 'SMO')
        ]
        self._edge_upf_nodes = [
            nid for nid, n in topology.nodes.items()
            if n.node_type.value == 'EdgeUPF'
        ]
        self._relay_nodes = [
            nid for nid, n in topology.nodes.items()
            if n.node_type.value in ('Relay', 'gNB-Site', 'O-DU', 'O-RU')
        ]
        self._ue_nodes = [
            nid for nid, n in topology.nodes.items()
            if n.node_type.value == 'UE'
        ]
        self._zones = sorted({
            z for nid in topology.nodes
            for z in (_infer_zone(nid),) if z
        })
        # Geographic extent — for radius-based geo_disaster scenarios
        self._has_geo = any(n.x_pos != 0.0 or n.y_pos != 0.0
                            for n in topology.nodes.values())
        if self._has_geo:
            xs = [n.x_pos for n in topology.nodes.values()]
            ys = [n.y_pos for n in topology.nodes.values()]
            self._geo_center = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2)
            self._geo_extent = (max(xs)-min(xs), max(ys)-min(ys))
        else:
            self._geo_center = (0.0, 0.0)
            self._geo_extent = (1.0, 1.0)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, episode: int = 1, seed: Optional[int] = None) -> Scenario:
        """Return a diverse Scenario for the given training episode."""
        rng = random.Random(
            seed if seed is not None else random.randint(0, 2**31)
        )
        sc_type = self._pick_type(episode, rng)
        return self._build(sc_type, episode, rng)

    # Backward-compatible entry point
    def generate_random_scenario(self, seed: Optional[int] = None) -> Scenario:
        return self.generate(episode=1, seed=seed)

    # ── Type selection ─────────────────────────────────────────────────────────

    def _pick_type(self, episode: int, rng: random.Random) -> str:
        allowed = ['full_core']
        for lo, hi, types in _CURRICULUM:
            if lo <= episode <= hi:
                allowed = types
                break
        weights = [_TYPE_WEIGHTS.get(t, 0.25) for t in allowed]
        return rng.choices(allowed, weights=weights, k=1)[0]

    # ── Scenario assembly ──────────────────────────────────────────────────────

    def _build(self, sc_type: str, episode: int, rng: random.Random) -> Scenario:
        # Severance in first 20-25% of episode so island window is maximised.
        # Having severance late (e.g. tick 600/800) leaves only 200 island ticks
        # -- not enough for agents to observe routing outcomes and learn.
        max_sev = max(101, min(200, self.base_duration // 4))
        sev_tick = rng.randint(80, max_sev)
        events: List[ScenarioEvent] = []

        # Primary failure event
        if sc_type == 'full_core':
            self._add_full_core(events, sev_tick)
        elif sc_type == 'partial_core':
            self._add_partial_core(events, sev_tick, rng)
        elif sc_type == 'zone_loss':
            self._add_zone_loss(events, sev_tick, rng)
        elif sc_type == 'cascading':
            self._add_cascading(events, sev_tick, rng)
        elif sc_type == 'multi_enb_iops':
            self._add_multi_enb_iops(events, sev_tick, rng)
        elif sc_type == 'geo_disaster':
            self._add_geo_disaster(events, sev_tick, rng)

        # Post-severance emergency UEs + rescue force
        self._add_emergency_ues(events, sev_tick, rng)
        self._add_rescue_force(events, sev_tick, rng)

        # Recovery (episode >= 3, 70% chance)
        recovery_tick = -1
        include_recovery = (episode >= 3) and (rng.random() < 0.70)
        if include_recovery:
            min_rt  = sev_tick + 100
            max_rt  = min(sev_tick + 400, self.base_duration - 50)
            if min_rt < max_rt:
                recovery_tick = rng.randint(min_rt, max_rt)
                events.append(ScenarioEvent(
                    tick=recovery_tick,
                    event_type='restore_core',
                    parameters={'scenario_type': sc_type}
                ))

        traffic_profiles = self._build_traffic(rng)

        recovery_str = (f"recovery@{recovery_tick}"
                        if include_recovery else "no_recovery")
        return Scenario(
            name=f"Diverse-{sc_type}-ep{episode}",
            duration_ticks=self.base_duration,
            events=sorted(events, key=lambda e: e.tick),
            traffic_profiles=traffic_profiles,
            description=(f"ep={episode} type={sc_type} sev={sev_tick} "
                         f"{recovery_str}"),
            scenario_type=sc_type,
        )

    # ── Event builders ─────────────────────────────────────────────────────────

    def _add_full_core(self, events: list, sev_tick: int):
        """Sever ALL core + EdgeUPF simultaneously."""
        events.append(ScenarioEvent(
            tick=sev_tick,
            event_type='sever_core',
            parameters={'include_edge_upf': True}
        ))

    def _add_partial_core(self, events: list, sev_tick: int,
                          rng: random.Random):
        """Core down; 1-2 EdgeUPF survive to provide partial backhaul."""
        k = min(rng.randint(1, 2), len(self._edge_upf_nodes))
        surviving = set(rng.sample(self._edge_upf_nodes, k)) \
            if self._edge_upf_nodes else set()
        severed_upf = [n for n in self._edge_upf_nodes if n not in surviving]
        events.append(ScenarioEvent(
            tick=sev_tick,
            event_type='partial_sever',
            parameters={'nodes': self._core_nodes + severed_upf}
        ))

    def _add_zone_loss(self, events: list, sev_tick: int, rng: random.Random):
        """One geographic zone + core severed."""
        # Core must go down too, otherwise island_mode won't trigger
        events.append(ScenarioEvent(
            tick=sev_tick,
            event_type='sever_core',
            parameters={'include_edge_upf': True}
        ))
        if self._zones:
            zone = rng.choice(self._zones)
            events.append(ScenarioEvent(
                tick=sev_tick,
                event_type='sever_zone',
                parameters={'zone': zone}
            ))

    def _add_cascading(self, events: list, sev_tick: int, rng: random.Random):
        """Core first, then relay nodes fail one-by-one at ~10-tick intervals."""
        events.append(ScenarioEvent(
            tick=sev_tick,
            event_type='sever_core',
            parameters={'include_edge_upf': True}
        ))
        if self._relay_nodes:
            n = rng.randint(3, min(8, len(self._relay_nodes)))
            chosen = rng.sample(self._relay_nodes, n)
            for i, nid in enumerate(chosen):
                fail_t = sev_tick + (i + 1) * rng.randint(8, 15)
                if fail_t < self.base_duration - 10:
                    events.append(ScenarioEvent(
                        tick=fail_t,
                        event_type='node_failure',
                        parameters={'node_id': nid}
                    ))

    def _add_geo_disaster(self, events: list, sev_tick: int,
                          rng: random.Random):
        """
        Geographic radius-based disaster.
        
        Picks a random epicenter and destruction radius, then fails all
        nodes within that radius. This creates realistic disaster patterns
        where geographically nearby nodes are destroyed together.
        
        Three epicenter modes:
          - center_hit: earthquake at the central DC area
          - zone_hit:   flood/explosion at one zone's edge site
          - corridor:   infrastructure corridor (off-center strike)
        """
        if not self._has_geo:
            # Fallback for topology without geographic coordinates
            self._add_full_core(events, sev_tick)
            return
        
        mode = rng.choice(['center_hit', 'zone_hit', 'corridor'])
        cx, cy = self._geo_center
        extent = max(self._geo_extent[0], self._geo_extent[1])
        
        if mode == 'center_hit':
            # Strike at central DC area
            epicenter = (cx + rng.uniform(-200, 200),
                         cy + rng.uniform(-200, 200))
            radius = extent * rng.uniform(0.15, 0.25)
        elif mode == 'zone_hit':
            # Strike at a zone's edge site
            zone_nodes = [n for n in self.topology.nodes.values()
                         if n.node_type.value == 'EdgeUPF']
            if zone_nodes:
                target = rng.choice(zone_nodes)
                epicenter = (target.x_pos + rng.uniform(-100, 100),
                             target.y_pos + rng.uniform(-100, 100))
            else:
                epicenter = (cx, cy)
            radius = extent * rng.uniform(0.20, 0.30)
        else:  # corridor
            # Off-center strike between two zones
            angle = rng.uniform(0, 2 * math.pi)
            offset = extent * rng.uniform(0.15, 0.30)
            epicenter = (cx + offset * math.cos(angle),
                         cy + offset * math.sin(angle))
            radius = extent * rng.uniform(0.12, 0.20)
        
        # Find all nodes within the disaster radius
        destroyed_nodes = []
        for nid, node in self.topology.nodes.items():
            d = math.sqrt((node.x_pos - epicenter[0])**2 +
                         (node.y_pos - epicenter[1])**2)
            if d <= radius:
                destroyed_nodes.append(nid)
        
        if not destroyed_nodes:
            # No nodes hit — fallback to full_core
            self._add_full_core(events, sev_tick)
            return
        
        # Emit geo_disaster event (simulation.py will handle the node failures)
        events.append(ScenarioEvent(
            tick=sev_tick,
            event_type='geo_disaster',
            parameters={
                'epicenter_x': epicenter[0],
                'epicenter_y': epicenter[1],
                'radius': radius,
                'destroyed_nodes': destroyed_nodes,
                'mode': mode,
            }
        ))
        
        # Also sever core if core nodes are in the blast
        core_hit = any(nid in self._core_nodes for nid in destroyed_nodes)
        if core_hit:
            events.append(ScenarioEvent(
                tick=sev_tick,
                event_type='sever_core',
                parameters={'include_edge_upf': False}
            ))
        
        # Fail individual nodes that aren't core (core handled by sever_core)
        for nid in destroyed_nodes:
            if nid not in self._core_nodes:
                events.append(ScenarioEvent(
                    tick=sev_tick,
                    event_type='node_failure',
                    parameters={'node_id': nid}
                ))

    def _add_emergency_ues(self, events: list, sev_tick: int,
                           rng: random.Random):
        """10-40 % of UEs declare emergency post-severance."""
        if not self._ue_nodes:
            return
        n = max(1, int(len(self._ue_nodes) * rng.uniform(0.10, 0.40)))
        for ue in rng.sample(self._ue_nodes, n):
            t = sev_tick + rng.randint(1, 25)
            if t < self.base_duration - 5:
                events.append(ScenarioEvent(
                    tick=t,
                    event_type='mcppt_emergency_alert',
                    parameters={
                        'ue_id': ue,
                        'severity': rng.choice(['emergency', 'imminent_peril']),
                    }
                ))

    def _add_rescue_force(self, events: list, sev_tick: int,
                          rng: random.Random):
        """0-20 rescue UEs arrive 20-60 ticks after severance."""
        arrival = sev_tick + rng.randint(20, 60)
        n = rng.randint(0, 20)
        if n > 0 and arrival < self.base_duration - 10:
            events.append(ScenarioEvent(
                tick=arrival,
                event_type='rescue_force_arrival',
                parameters={'count': n}
            ))

    def _build_traffic(self, rng: random.Random) -> Dict[str, NodeTrafficProfile]:
        profiles = {}
        for nid, node in self.topology.nodes.items():
            ntv = node.node_type.value
            if ntv == 'UE':
                profiles[nid] = NodeTrafficProfile.from_config({
                    'node_id': nid,
                    'life_safety': {'baseline_rate': rng.uniform(3.0,  15.0),
                                    'surge_events': []},
                    'operations':  {'baseline_rate': rng.uniform(15.0, 30.0),
                                    'surge_events': []},
                    'telemetry':   {'baseline_rate': rng.uniform(30.0, 60.0),
                                    'surge_events': []},
                    'best_effort': {'baseline_rate': rng.uniform(45.0, 90.0),
                                    'surge_events': []},
                })
            else:
                profiles[nid] = NodeTrafficProfile.from_config({
                    'node_id': nid,
                    'life_safety': {'baseline_rate': 0.0, 'surge_events': []},
                    'operations':  {'baseline_rate': 0.0, 'surge_events': []},
                    'telemetry':   {'baseline_rate': 0.0, 'surge_events': []},
                    'best_effort': {'baseline_rate': 0.0, 'surge_events': []},
                })
        return profiles

    # ── Multi-eNB IOPS scenario builder (TS 22.346) ──────────────────────────

    def _add_multi_enb_iops(self, events: list, sev_tick: int,
                             rng: random.Random):
        """
        Multi-eNB IOPS scenario (ETSI TS 122 346 V16.0.0):
          1. Core severance
          2. Multi-eNB island formation
          3. NeNB deployment by rescue teams
          4. MCPTT group calls within island
        """
        # Phase 1: Core severance
        events.append(ScenarioEvent(
            tick=sev_tick,
            event_type='sever_core',
            parameters={'include_edge_upf': True}
        ))

        # Phase 2: Multi-eNB IOPS init (auto-detected, but explicit event helps logging)
        events.append(ScenarioEvent(
            tick=sev_tick + 1,
            event_type='multi_enb_iops_init',
            parameters={'reason': 'core_severance'}
        ))

        # Phase 3: NeNB deployments (1-3 Nomadic eNBs from rescue teams)
        n_nenb = rng.randint(1, 3)
        for i in range(n_nenb):
            deploy_tick = sev_tick + rng.randint(40, 120)
            if deploy_tick < self.base_duration - 50:
                events.append(ScenarioEvent(
                    tick=deploy_tick,
                    event_type='nenb_deployment',
                    parameters={
                        'nenb_index': i,
                        'coverage_area': (rng.choice(self._zones)
                                          if self._zones else None),
                    }
                ))

        # Phase 4: MCPTT group calls within island
        n_calls = rng.randint(1, 3)
        for i in range(n_calls):
            call_tick = sev_tick + rng.randint(50, 200)
            group_size = rng.randint(3, min(8, len(self._ue_nodes)))
            if call_tick < self.base_duration - 20 and self._ue_nodes:
                group_ues = rng.sample(
                    self._ue_nodes, min(group_size, len(self._ue_nodes)))
                events.append(ScenarioEvent(
                    tick=call_tick,
                    event_type='mcptt_group_call',
                    parameters={
                        'group_ues': group_ues,
                        'call_type': rng.choice(
                            ['mcptt_emergency', 'mcptt_private']),
                    }
                ))

        # Also add emergency UEs (some declare alerts)
        if self._ue_nodes:
            n_emrg = max(1, int(len(self._ue_nodes) * rng.uniform(0.15, 0.50)))
            for ue in rng.sample(self._ue_nodes, n_emrg):
                t = sev_tick + rng.randint(2, 30)
                if t < self.base_duration - 5:
                    events.append(ScenarioEvent(
                        tick=t,
                        event_type='mcppt_emergency_alert',
                        parameters={
                            'ue_id': ue,
                            'severity': rng.choice(
                                ['emergency', 'imminent_peril']),
                        }
                    ))


# Backward-compatible alias
class ScenarioGenerator(DiverseScenarioGenerator):
    """Kept for backward compatibility with earlier code."""
    def generate_random_scenario(self, seed: Optional[int] = None) -> Scenario:
        return self.generate(episode=1, seed=seed)
