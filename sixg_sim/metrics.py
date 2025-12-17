"""
Metrics collection and analysis for network simulation.

Tracks simulation state and computes recovery time, utilization, and other KPIs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from .topology import TrafficClass, Node, Link
from .scenario import Scenario


@dataclass
class TickMetrics:
    """Metrics collected at each simulation tick."""
    tick: int
    node_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    link_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    traffic_stats: Dict[str, Dict[TrafficClass, Dict[str, float]]] = field(default_factory=dict)
    control_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    island_mode_active: bool = False
    severance_tick: Optional[int] = None


@dataclass
class RecoveryMetrics:
    """Recovery time and success metrics."""
    life_safety_success_ratio: float
    recovery_time_ticks: Optional[int]  # Ticks to reach target after severance
    target_threshold: float = 0.95  # Target success ratio
    window_size: int = 20  # Ticks to average over

    @classmethod
    def calculate_recovery_time(cls, metrics_history: List[TickMetrics],
                              severance_tick: int, target_threshold: float = 0.95,
                              window_size: int = 20) -> Optional[int]:
        """Calculate recovery time after severance."""
        if not metrics_history or severance_tick is None:
            return None

        # Find metrics after severance
        post_severance_metrics = [m for m in metrics_history if m.tick >= severance_tick]

        for i in range(len(post_severance_metrics) - window_size + 1):
            window = post_severance_metrics[i:i + window_size]

            # Calculate average life-safety success ratio over window
            success_ratios = []
            for metric in window:
                total_life_safety = 0.0
                successful_life_safety = 0.0

                for node_stats in metric.traffic_stats.values():
                    if TrafficClass.LIFE_SAFETY in node_stats:
                        ls_stats = node_stats[TrafficClass.LIFE_SAFETY]
                        offered = ls_stats.get('offered_load', 0)
                        delivered = ls_stats.get('delivered_load', 0)
                        total_life_safety += offered
                        successful_life_safety += delivered

                if total_life_safety > 0:
                    ratio = successful_life_safety / total_life_safety
                    success_ratios.append(ratio)

            if success_ratios and np.mean(success_ratios) >= target_threshold:
                return window[0].tick - severance_tick

        return None  # Recovery not achieved


@dataclass
class EnergyMetrics:
    """Energy consumption and state metrics."""
    total_energy_used: Dict[str, float] = field(default_factory=dict)
    final_soc: Dict[str, float] = field(default_factory=dict)
    depletion_events: List[Tuple[str, int]] = field(default_factory=list)  # (node_id, tick)

    @classmethod
    def from_metrics_history(cls, metrics_history: List[TickMetrics],
                           topology_nodes: Dict[str, Node]) -> 'EnergyMetrics':
        """Calculate energy metrics from history."""
        metrics = cls()

        for node_id, node in topology_nodes.items():
            # Calculate total energy used (simplified - in reality would track per tick)
            metrics.total_energy_used[node_id] = (1.0 - node.energy_soc) * 1000.0  # Scale up
            metrics.final_soc[node_id] = node.energy_soc

            # Check for depletion events
            for tick_metric in metrics_history:
                if node_id in tick_metric.node_states:
                    soc = tick_metric.node_states[node_id].get('energy_soc', 1.0)
                    if soc <= 0.0:
                        metrics.depletion_events.append((node_id, tick_metric.tick))
                        break

        return metrics


@dataclass
class ConnectivityMetrics:
    """Connectivity and reachability metrics."""
    islands_over_time: List[Tuple[int, int]] = field(default_factory=list)  # (tick, num_islands)
    connectivity_matrix: Dict[int, Dict[str, Dict[str, bool]]] = field(default_factory=dict)
    gnb_reachability: Dict[int, float] = field(default_factory=dict)  # tick -> fraction reachable

    @classmethod
    def calculate_island_sizes(cls, metrics_history: List[TickMetrics],
                             topology_links: Dict[str, Link]) -> List[Tuple[int, int]]:
        """Calculate number of islands over time (simplified)."""
        islands = []

        for metric in metrics_history:
            # Count surviving nodes that are marked as island
            island_nodes = sum(1 for node_state in metric.node_states.values()
                             if node_state.get('is_island', False))
            islands.append((metric.tick, max(1, island_nodes)))

        return islands


@dataclass
class UtilizationMetrics:
    """Link and node utilization metrics."""
    link_utilization: Dict[str, Dict[str, float]] = field(default_factory=dict)  # link_id -> {avg, peak}
    node_utilization: Dict[str, Dict[str, float]] = field(default_factory=dict)  # node_id -> {avg_queue, etc}

    @classmethod
    def from_metrics_history(cls, metrics_history: List[TickMetrics]) -> 'UtilizationMetrics':
        """Calculate utilization metrics from history."""
        metrics = cls()

        # Link utilization
        link_usage = defaultdict(list)

        for metric in metrics_history:
            for link_id, link_state in metric.link_states.items():
                utilization = link_state.get('utilization', 0.0)
                link_usage[link_id].append(utilization)

        for link_id, usages in link_usage.items():
            metrics.link_utilization[link_id] = {
                'average': np.mean(usages),
                'peak': max(usages),
                'min': min(usages)
            }

        # Node utilization (queue lengths)
        node_queues = defaultdict(lambda: defaultdict(list))

        for metric in metrics_history:
            for node_id, traffic_stats in metric.traffic_stats.items():
                for traffic_class, class_stats in traffic_stats.items():
                    queue_len = class_stats.get('queued_load', 0.0)
                    node_queues[node_id][traffic_class.name].append(queue_len)

        for node_id, class_queues in node_queues.items():
            node_stats = {}
            for class_name, queues in class_queues.items():
                node_stats[f'{class_name}_avg_queue'] = np.mean(queues)
                node_stats[f'{class_name}_max_queue'] = max(queues)
            metrics.node_utilization[node_id] = node_stats

        return metrics


@dataclass
class TrafficMetrics:
    """Traffic statistics per class."""
    per_class_stats: Dict[TrafficClass, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_metrics_history(cls, metrics_history: List[TickMetrics]) -> 'TrafficMetrics':
        """Calculate traffic statistics from history."""
        metrics = cls()

        class_totals = defaultdict(lambda: defaultdict(float))

        for metric in metrics_history:
            for node_stats in metric.traffic_stats.values():
                for traffic_class, class_stats in node_stats.items():
                    for stat_name, value in class_stats.items():
                        class_totals[traffic_class][stat_name] += value

        # Convert to averages/rates
        num_ticks = len(metrics_history)
        for traffic_class in TrafficClass:
            if traffic_class in class_totals:
                stats = class_totals[traffic_class]
                metrics.per_class_stats[traffic_class] = {
                    'total_offered': stats.get('offered_load', 0.0),
                    'total_admitted': stats.get('admitted_load', 0.0),
                    'total_delivered': stats.get('delivered_load', 0.0),
                    'total_dropped': stats.get('dropped_load', 0.0),
                    'avg_offered_per_tick': stats.get('offered_load', 0.0) / num_ticks,
                    'admission_rate': (stats.get('admitted_load', 0.0) /
                                     max(stats.get('offered_load', 0.0), 1.0)),
                    'delivery_rate': (stats.get('delivered_load', 0.0) /
                                    max(stats.get('admitted_load', 0.0), 1.0))
                }

        return metrics


class MetricsCollector:
    """Collects and aggregates simulation metrics."""

    def __init__(self):
        self.metrics_history: List[TickMetrics] = []
        self.severance_tick: Optional[int] = None

    def record_tick(self, tick: int, node_states: Dict[str, Any],
                   link_states: Dict[str, Any], traffic_stats: Dict[str, Any],
                   control_stats: Dict[str, Any], island_mode: bool):
        """Record metrics for a simulation tick."""
        metric = TickMetrics(
            tick=tick,
            node_states=node_states,
            link_states=link_states,
            traffic_stats=traffic_stats,
            control_stats=control_stats,
            island_mode_active=island_mode,
            severance_tick=self.severance_tick
        )
        self.metrics_history.append(metric)

    def set_severance_tick(self, tick: int):
        """Mark the tick when severance occurred."""
        self.severance_tick = tick

    def get_recovery_metrics(self) -> RecoveryMetrics:
        """Calculate recovery metrics."""
        recovery_time = RecoveryMetrics.calculate_recovery_time(
            self.metrics_history, self.severance_tick
        )
        return RecoveryMetrics(
            life_safety_success_ratio=0.85,  # Placeholder - would calculate from history
            recovery_time_ticks=recovery_time
        )

    def get_energy_metrics(self, topology_nodes: Dict[str, Node]) -> EnergyMetrics:
        """Get energy consumption metrics."""
        return EnergyMetrics.from_metrics_history(self.metrics_history, topology_nodes)

    def get_connectivity_metrics(self, topology_links: Dict[str, Link]) -> ConnectivityMetrics:
        """Get connectivity and island metrics."""
        return ConnectivityMetrics(
            islands_over_time=ConnectivityMetrics.calculate_island_sizes(
                self.metrics_history, topology_links
            )
        )

    def get_utilization_metrics(self) -> UtilizationMetrics:
        """Get utilization metrics."""
        return UtilizationMetrics.from_metrics_history(self.metrics_history)

    def get_traffic_metrics(self) -> TrafficMetrics:
        """Get traffic statistics."""
        return TrafficMetrics.from_metrics_history(self.metrics_history)

    def export_to_csv(self, output_dir: str):
        """Export metrics to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Per-tick metrics
        tick_data = []
        for metric in self.metrics_history:
            row = {'tick': metric.tick, 'island_mode': metric.island_mode_active}
            tick_data.append(row)

        pd.DataFrame(tick_data).to_csv(f"{output_dir}/per_tick_metrics.csv", index=False)

        # Could add more detailed exports here
        print(f"Metrics exported to {output_dir}")
