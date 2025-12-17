"""
Analysis and visualization functions for simulation results.

Generates plots, summaries, and reports from simulation metrics.
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from .metrics import (
    MetricsCollector, RecoveryMetrics, EnergyMetrics,
    ConnectivityMetrics, UtilizationMetrics, TrafficMetrics
)
from .topology import TrafficClass


def plot_life_safety_success_over_time(metrics: MetricsCollector, output_file: str = None):
    """Plot life-safety traffic success ratio over time."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot generation")
        return

    ticks = []
    success_ratios = []
    island_mode_ticks = []

    for metric in metrics.metrics_history:
        ticks.append(metric.tick)
        if metric.island_mode_active:
            island_mode_ticks.append(metric.tick)

        # Calculate life-safety success for this tick
        total_offered = 0.0
        total_delivered = 0.0

        for node_stats in metric.traffic_stats.values():
            if TrafficClass.LIFE_SAFETY in node_stats:
                ls_stats = node_stats[TrafficClass.LIFE_SAFETY]
                total_offered += ls_stats.get('offered_load', 0)
                total_delivered += ls_stats.get('delivered_load', 0)

        ratio = total_delivered / max(total_offered, 1.0)
        success_ratios.append(ratio)

    plt.figure(figsize=(12, 6))
    plt.plot(ticks, success_ratios, 'b-', linewidth=2, label='Life Safety Success Ratio')

    if island_mode_ticks:
        plt.axvline(x=island_mode_ticks[0], color='r', linestyle='--',
                   label=f'Severance (Tick {island_mode_ticks[0]})')

    plt.xlabel('Simulation Tick')
    plt.ylabel('Success Ratio')
    plt.title('Life Safety Traffic Success Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.close()  # Don't show if no output file


def plot_energy_consumption(metrics: MetricsCollector, topology_nodes: Dict[str, Any],
                           output_file: str = None):
    """Plot energy state over time for key nodes."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping energy plot")
        return

    energy_history = {}

    # Extract energy history
    for metric in metrics.metrics_history:
        for node_id, node_state in metric.node_states.items():
            if node_id not in energy_history:
                energy_history[node_id] = []
            energy_history[node_id].append((metric.tick, node_state.get('energy_soc', 1.0)))

    plt.figure(figsize=(12, 6))

    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (node_id, history) in enumerate(energy_history.items()):
        if not history:
            continue
        ticks, socs = zip(*history)
        color = colors[i % len(colors)]
        plt.plot(ticks, socs, color=color, linewidth=2, label=f'Node {node_id}')

    plt.xlabel('Simulation Tick')
    plt.ylabel('State of Charge (SoC)')
    plt.title('Node Energy State Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Energy plot saved to {output_file}")
    else:
        plt.close()


def plot_link_utilization(utilization_metrics: UtilizationMetrics, output_file: str = None):
    """Plot average link utilization."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping link utilization plot")
        return

    link_ids = []
    avg_utils = []
    peak_utils = []

    for link_id, stats in utilization_metrics.link_utilization.items():
        link_ids.append(link_id)
        avg_utils.append(stats['average'])
        peak_utils.append(stats['peak'])

    x = np.arange(len(link_ids))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, avg_utils, width, label='Average Utilization', alpha=0.8)
    plt.bar(x + width/2, peak_utils, width, label='Peak Utilization', alpha=0.8)

    plt.xlabel('Link ID')
    plt.ylabel('Utilization (%)')
    plt.title('Link Utilization Statistics')
    plt.xticks(x, link_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Link utilization plot saved to {output_file}")
    else:
        plt.close()


def plot_traffic_statistics(traffic_metrics: TrafficMetrics, output_file: str = None):
    """Plot traffic statistics per class."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping traffic statistics plot")
        return

    classes = []
    admission_rates = []
    delivery_rates = []

    for traffic_class, stats in traffic_metrics.per_class_stats.items():
        classes.append(traffic_class.value.replace('_', ' ').title())
        admission_rates.append(stats.get('admission_rate', 0))
        delivery_rates.append(stats.get('delivery_rate', 0))

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, admission_rates, width, label='Admission Rate', alpha=0.8)
    plt.bar(x + width/2, delivery_rates, width, label='Delivery Rate', alpha=0.8)

    plt.xlabel('Traffic Class')
    plt.ylabel('Rate')
    plt.title('Traffic Statistics by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.1)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Traffic statistics plot saved to {output_file}")
    else:
        plt.close()


def generate_simulation_summary(metrics: MetricsCollector,
                              topology_nodes: Dict[str, Any],
                              output_file: str = None) -> str:
    """Generate a text summary of simulation results."""

    recovery = metrics.get_recovery_metrics()
    energy = metrics.get_energy_metrics(topology_nodes)
    connectivity = metrics.get_connectivity_metrics({})
    utilization = metrics.get_utilization_metrics()
    traffic = metrics.get_traffic_metrics()

    summary = []
    summary.append("=" * 60)
    summary.append("6G NETWORK SIMULATION SUMMARY")
    summary.append("=" * 60)

    # Recovery metrics
    summary.append("\nRECOVERY METRICS:")
    summary.append(f"  Severance occurred at tick: {metrics.severance_tick}")
    summary.append(f"  Recovery time: {recovery.recovery_time_ticks} ticks" if recovery.recovery_time_ticks else "  Recovery: Not achieved")
    summary.append(".2f")

    # Energy metrics
    summary.append("\nENERGY METRICS:")
    for node_id, energy_used in energy.total_energy_used.items():
        final_soc = energy.final_soc.get(node_id, 0.0)
        summary.append(f"  {node_id}: {energy_used:.1f} units used, final SoC: {final_soc:.2f}")

    if energy.depletion_events:
        summary.append(f"  Depletion events: {len(energy.depletion_events)} nodes depleted")

    # Connectivity metrics
    summary.append("\nCONNECTIVITY METRICS:")
    if connectivity.islands_over_time:
        final_islands = connectivity.islands_over_time[-1][1] if connectivity.islands_over_time else 0
        summary.append(f"  Final number of islands: {final_islands}")

    # Utilization metrics
    summary.append("\nUTILIZATION METRICS:")
    if utilization.link_utilization:
        avg_link_util = np.mean([stats['average'] for stats in utilization.link_utilization.values()])
        summary.append(".1f")
        max_link_util = max([stats['peak'] for stats in utilization.link_utilization.values()])
        summary.append(".1f")

    # Traffic metrics
    summary.append("\nTRAFFIC METRICS:")
    for traffic_class, stats in traffic.per_class_stats.items():
        class_name = traffic_class.value.replace('_', ' ').title()
        offered = stats.get('total_offered', 0)
        delivered = stats.get('total_delivered', 0)
        admission_rate = stats.get('admission_rate', 0)
        delivery_rate = stats.get('delivery_rate', 0)
        summary.append(f"  {class_name}:")
        summary.append(".1f")
        summary.append(".2f")
        summary.append(".2f")

    summary.append("\n" + "=" * 60)

    final_summary = "\n".join(summary)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(final_summary)

    return final_summary


def run_complete_analysis(metrics: MetricsCollector, topology_nodes: Dict[str, Any],
                         output_dir: str = "output"):
    """Run complete analysis suite and generate all outputs."""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    plot_life_safety_success_over_time(metrics, f"{output_dir}/life_safety_success.png")
    plot_energy_consumption(metrics, topology_nodes, f"{output_dir}/energy_consumption.png")
    plot_link_utilization(metrics.get_utilization_metrics(), f"{output_dir}/link_utilization.png")
    plot_traffic_statistics(metrics.get_traffic_metrics(), f"{output_dir}/traffic_statistics.png")

    # Generate summary
    summary = generate_simulation_summary(metrics, topology_nodes, f"{output_dir}/simulation_summary.txt")

    print("Analysis complete. Results saved to:", output_dir)
    print("\nSummary:")
    print(summary)
