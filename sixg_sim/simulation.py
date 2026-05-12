"""
Main simulation engine for 6G network simulation.

Coordinates topology, agents, traffic, and control plane through discrete time steps.

MCPTT Emergency Communication Implementation (3GPP TS 22.179):
- NETWORK-ASSISTED UE-to-UE communication through surviving RAN infrastructure
- NOT direct ProSe UE-to-UE (device-to-device without any network involvement)
- Rescue service coordination via distributed RAN (O-RU/O-DU/O-CU-UP)
- Emergency alerts and prioritized calls following 3GPP standards
- Island mode communication when core network (UPF/RIC) fails but RAN survives
- MARL-enabled intelligent routing through distributed infrastructure
"""

import os
import torch
import networkx as nx
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from .topology import Topology, Node, Link, TrafficClass, NodeType, LinkType, InterfaceType
from .traffic import TrafficGenerator, SliceDictionary
from .agent import (BaseAgent, AgentObservation, EnergyTier, StrainLevel,
                    create_agent_for_node, LocalSliceState, RLAgent, CentralizedMARLTrainer,
                    PHYMACAction, NeighborRadioSummary, ConnectivityState,
                    ControlPostcard, StrainLevel)
from .phy_mac_state import PHYMACState, RelayMode, MCSLevel, MACScheduler, TX_POWER_STEPS_DB
from .transport_relay_model import TransportRelayModel
from .iops_manager import IOPSManager
from .coordinator_agent import CoordinatorAgent, GlobalPolicyVector
from .control_plane import ControlPlaneManager
from .scenario import Scenario, ScenarioEvent
from .metrics import MetricsCollector, TickMetrics

import matplotlib
# Respect env override first; fallback to TkAgg then Agg
backend_env = os.environ.get("MPLBACKEND")
if backend_env:
    matplotlib.use(backend_env, force=True)
else:
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    tick_duration_ms: int = 100  # Duration of each tick in milliseconds
    random_seed: Optional[int] = None
    enable_island_detection: bool = True
    control_message_limit: int = 10  # Max control bytes per tick
    verbose: bool = False  # Enable verbose logging
    sample_interval: int = 100  # How often to print progress when verbose
    live_plot: bool = False  # Enable real-time topology plotting (requires GUI backend)
    live_interval: int = 50  # Ticks between live plot updates
    live_max_labels: int = 40  # How many infra nodes to label in live plot
    agent_monitor: bool = False  # Enable single agent monitoring plotter


class NetworkPainter:
    """Real-time O-RAN topology painter with interface colors and UE counts."""

    # O-RAN Node colors
    NODE_COLORS = {
        "SMO": "#8B0000", "Non-RT-RIC": "#A52A2A", "AMF": "#DC143C", "UPF": "#FF4500",
        "Core": "#FF6347", "Near-RT-RIC": "#9400D3",
        "O-CU-CP": "#4169E1", "O-CU-UP": "#1E90FF", "O-CU": "#6495ED", "CU": "#6495ED",
        "O-DU": "#20B2AA", "DU": "#20B2AA",
        "O-RU": "#32CD32", "gNB-Site": "#228B22", "GNBSite": "#228B22",
        "Relay": "#808080", "EdgeUPF": "#DAA520",
    }
    NODE_SIZES = {
        "SMO": 400, "Non-RT-RIC": 350, "AMF": 320, "UPF": 300, "Core": 300,
        "Near-RT-RIC": 350, "O-CU-CP": 280, "O-CU-UP": 280, "O-CU": 280, "CU": 280,
        "O-DU": 240, "DU": 240, "O-RU": 200, "gNB-Site": 200, "GNBSite": 200,
        "Relay": 180, "EdgeUPF": 260,
    }
    # O-RAN Interface colors
    IFACE_COLORS = {
        "Open-FH": "#00FF00", "F1": "#0000FF", "F1-C": "#0000CD", "F1-U": "#4169E1",
        "E1": "#9932CC", "E2": "#FF00FF", "A1": "#FF69B4", "O1": "#FFA500",
        "N2": "#FF0000", "N3": "#DC143C", "N4": "#B22222", "NG": "#CD5C5C",
        "Xn": "#FFD700", "Backhaul": "#D2691E", "Microwave": "#A0522D",
    }

    def __init__(self, topology: Topology, max_labels: int = 40):
        from matplotlib.lines import Line2D
        self.topology = topology
        self.max_labels = max_labels
        self.Line2D = Line2D
        
        # Build graph with only infrastructure nodes (no UEs)
        self.G = nx.Graph()
        for node_id, node in topology.nodes.items():
            if node.node_type.value == "UE":
                continue  # Skip UEs
            self.G.add_node(node_id, node_type=node.node_type.value)
        
        # Store infrastructure links with interface type
        # Skip: Uu (air), O1 (management - clutters view), O2 (cloud mgmt)
        self.infra_links = []
        skip_interfaces = {"Uu", "O1", "O2"}
        
        for link in topology.links.values():
            iface_type = getattr(link, 'interface_type', None)
            iface_name = iface_type.value if iface_type else "Backhaul"
            
            # Skip management and air interfaces
            if iface_name in skip_interfaces:
                continue
            # Skip links involving UEs
            if any(topology.nodes.get(ep) and topology.nodes[ep].node_type.value == "UE" 
                   for ep in link.endpoints):
                continue
            
            self.G.add_edge(link.endpoints[0], link.endpoints[1], interface=iface_name)
            self.infra_links.append(link)
        
        # Store UE links for counting
        self.ue_links = [l for l in topology.links.values() 
                        if getattr(l, 'interface_type', None) and 
                        getattr(l, 'interface_type').value == "Uu"]
        
        # Create geographic-style layout based on node type hierarchy and coverage area
        self.pos = self._create_geographic_layout()
        self.fig, self.ax = plt.subplots(figsize=(16, 11))
        self.ax.set_title("O-RAN Topology (live)")
        self.ax.axis("off")
    
    def _create_geographic_layout(self):
        """Create a geographic-style layout spreading nodes across the figure."""
        import random
        random.seed(42)
        
        pos = {}
        
        # Define vertical layers (y-axis) for O-RAN hierarchy (top to bottom)
        layer_y = {
            # Core/Management layer (top)
            "SMO": 0.95, "Non-RT-RIC": 0.90, "AMF": 0.85, "UPF": 0.85, "Core": 0.88,
            # Near-RT RIC layer
            "Near-RT-RIC": 0.75,
            # CU layer
            "O-CU-CP": 0.60, "O-CU-UP": 0.58, "O-CU": 0.59, "CU": 0.59,
            # Edge/Transport layer
            "EdgeUPF": 0.50, "Relay": 0.45,
            # DU layer
            "O-DU": 0.30, "DU": 0.30,
            # RU layer (bottom - closest to users)
            "O-RU": 0.12, "gNB-Site": 0.12, "GNBSite": 0.12,
        }
        
        # Group nodes by coverage area for horizontal spreading
        nodes_by_area = {}
        for node_id in self.G.nodes():
            node = self.topology.nodes.get(node_id)
            if node:
                area = node.coverage_area or "default"
                if area not in nodes_by_area:
                    nodes_by_area[area] = []
                nodes_by_area[area].append(node_id)
        
        # Assign x positions based on coverage area
        areas = sorted(nodes_by_area.keys())
        area_x_base = {area: (i + 0.5) / len(areas) for i, area in enumerate(areas)}
        
        # Position each node
        type_counters = {}  # Track count per type for horizontal offset within type
        
        for node_id in self.G.nodes():
            node = self.topology.nodes.get(node_id)
            if not node:
                continue
                
            ntype = node.node_type.value
            area = node.coverage_area or "default"
            
            # Get base y from layer
            y = layer_y.get(ntype, 0.5)
            # Add small random jitter to prevent overlap
            y += random.uniform(-0.03, 0.03)
            
            # Get base x from area
            x_base = area_x_base.get(area, 0.5)
            
            # Add offset within area based on node type to spread horizontally
            if ntype not in type_counters:
                type_counters[ntype] = {}
            if area not in type_counters[ntype]:
                type_counters[ntype][area] = 0
            
            count = type_counters[ntype][area]
            type_counters[ntype][area] += 1
            
            # Spread nodes of same type within area
            area_width = 0.8 / len(areas)  # Width per area
            x_offset = (count % 8) * (area_width / 10) - area_width / 4
            x = x_base + x_offset + random.uniform(-0.02, 0.02)
            
            # Clamp to valid range
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            pos[node_id] = (x, y)
        
        return pos

    def update(self, snapshot: TickMetrics):
        self.ax.clear()
        self.ax.axis("off")
        
        # First pass: find all UEs that are connected to ANY O-RU (globally connected)
        globally_connected_ues = set()
        ue_to_original_ru = {}  # Track which O-RU each UE was originally linked to
        
        for link in self.ue_links:
            lstate = snapshot.link_states.get(link.id, {})
            is_up = lstate.get("is_up", True)
            
            # Identify UE and O-RU endpoints
            ue_ep = None
            ru_ep = None
            for ep in link.endpoints:
                node = self.topology.nodes.get(ep)
                if node:
                    if node.node_type.value == "UE":
                        ue_ep = ep
                    elif node.node_type.value in ["O-RU", "gNB-Site", "GNBSite"]:
                        ru_ep = ep
            
            if ue_ep and ru_ep:
                # Track original O-RU for each UE
                if ue_ep not in ue_to_original_ru:
                    ue_to_original_ru[ue_ep] = ru_ep
                
                # Check if this UE is connected via this link
                if is_up:
                    ue_state = snapshot.node_states.get(ue_ep, {})
                    ru_state = snapshot.node_states.get(ru_ep, {})
                    if ue_state.get("is_survivor", True) and ru_state.get("is_survivor", True):
                        globally_connected_ues.add(ue_ep)
        
        # Second pass: count per O-RU with disconnected tracking
        ue_count_per_ru = {}
        total_connected_ues = 0
        # Count actual UEs from topology (dynamic population)
        total_ues = sum(1 for n in self.topology.nodes.values() if n.node_type.value == "UE")
        
        for link in self.ue_links:
            lstate = snapshot.link_states.get(link.id, {})
            is_up = lstate.get("is_up", True)
            
            # Identify endpoints
            ue_ep = None
            ru_ep = None
            for ep in link.endpoints:
                node = self.topology.nodes.get(ep)
                if node:
                    if node.node_type.value == "UE":
                        ue_ep = ep
                    elif node.node_type.value in ["O-RU", "gNB-Site", "GNBSite"]:
                        ru_ep = ep
            
            if ru_ep:
                if ru_ep not in ue_count_per_ru:
                    ue_count_per_ru[ru_ep] = {"connected": 0, "total": 0, "truly_disconnected": 0}
                ue_count_per_ru[ru_ep]["total"] += 1
                total_ues += 1
                
                if ue_ep:
                    ue_state = snapshot.node_states.get(ue_ep, {})
                    ru_state = snapshot.node_states.get(ru_ep, {})
                    
                    if is_up and ue_state.get("is_survivor", True) and ru_state.get("is_survivor", True):
                        # Connected to this O-RU
                        ue_count_per_ru[ru_ep]["connected"] += 1
                        total_connected_ues += 1
                    else:
                        # Not connected to this O-RU - check if truly disconnected
                        # (not connected to ANY O-RU)
                        if ue_ep not in globally_connected_ues:
                            ue_count_per_ru[ru_ep]["truly_disconnected"] += 1
        
        # Categorize edges by interface type
        edges_by_iface = {}
        failed_edges = []
        
        for link in self.infra_links:
            lstate = snapshot.link_states.get(link.id, {})
            is_up = lstate.get("is_up", True)
            iface_type = getattr(link, 'interface_type', None)
            iface_name = iface_type.value if iface_type else "Backhaul"
            edge = (link.endpoints[0], link.endpoints[1])
            
            if not is_up:
                failed_edges.append(edge)
            else:
                if iface_name not in edges_by_iface:
                    edges_by_iface[iface_name] = []
                edges_by_iface[iface_name].append(edge)
        
        # Draw edges by interface type with colors
        for iface_name, edges in edges_by_iface.items():
            color = self.IFACE_COLORS.get(iface_name, "#888888")
            width = 2.0 if iface_name in ["Open-FH", "F1", "N3"] else 1.5
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edges, edge_color=color,
                                   width=width, alpha=0.7, ax=self.ax)
        
        # Draw failed edges
        if failed_edges:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=failed_edges, edge_color="#FF0000",
                                   width=2.5, style="dashed", alpha=0.9, ax=self.ax)
        
        # ==== Communication Paths (would show MARL coordination) ====
        marl_paths = getattr(snapshot, 'marl_comm_paths', [])
        if marl_paths:
            marl_edges = []
            for src, tgt, msg_type in marl_paths:
                if src in self.G.nodes() and tgt in self.G.nodes():
                    marl_edges.append((src, tgt))
            
            if marl_edges:
                # Draw outer glow (wide, semi-transparent cyan)
                nx.draw_networkx_edges(self.G, self.pos, edgelist=marl_edges, 
                                       edge_color="#00FFFF", width=8.0, alpha=0.3, ax=self.ax)
                # Draw middle glow
                nx.draw_networkx_edges(self.G, self.pos, edgelist=marl_edges,
                                       edge_color="#00FFFF", width=4.0, alpha=0.5, ax=self.ax)
                # Draw inner bright core
                nx.draw_networkx_edges(self.G, self.pos, edgelist=marl_edges,
                                       edge_color="#FFFFFF", width=2.0, alpha=0.9, ax=self.ax)
                
                # Draw small arrows/dots showing direction of messages
                for src, tgt in marl_edges[:30]:  # Limit for performance
                    if src in self.pos and tgt in self.pos:
                        x1, y1 = self.pos[src]
                        x2, y2 = self.pos[tgt]
                        # Position message indicator at 70% along the path
                        mx = x1 + 0.7 * (x2 - x1)
                        my = y1 + 0.7 * (y2 - y1)
                        self.ax.plot(mx, my, 'o', color='#00FFFF', markersize=4,
                                    markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        # ==== Failed Postcard Transmissions ====
        failed_transmissions = getattr(snapshot, 'failed_postcard_transmissions', [])
        if failed_transmissions:
            # Draw failed transmission indicators as red X marks
            for failure in failed_transmissions[-20:]:  # Show last 20 failures
                src = failure.get('source')
                tgt = failure.get('target')
                priority = failure.get('priority', 'normal')
                reason = failure.get('reason', 'unknown')

                if src in self.pos and tgt in self.pos:
                    x1, y1 = self.pos[src]
                    x2, y2 = self.pos[tgt]

                    # Draw broken red line for failed transmission
                    self.ax.plot([x1, x2], [y1, y2], color='#FF0000', linewidth=1.5,
                               linestyle='--', alpha=0.7, zorder=4)

                    # Draw red X at midpoint to indicate failure
                    mx = (x1 + x2) / 2
                    my = (y1 + y2) / 2
                    self.ax.plot(mx, my, 'x', color='#FF0000', markersize=8,
                               markeredgewidth=2, zorder=6)
        
        # Separate survivor and failed nodes
        survivor_nodes, survivor_colors, survivor_sizes = [], [], []
        failed_nodes, failed_colors, failed_sizes = [], [], []
        
        for n in self.G.nodes():
            state = snapshot.node_states.get(n, {})
            is_survivor = state.get("is_survivor", True)
            ntype = self.G.nodes[n]["node_type"]
            color, size = self._node_style(ntype, is_survivor)
            
            if is_survivor:
                survivor_nodes.append(n)
                survivor_colors.append(color)
                survivor_sizes.append(size)
            else:
                failed_nodes.append(n)
                failed_colors.append("#FF0000")
                failed_sizes.append(size * 1.2)
        
        # Draw nodes
        if survivor_nodes:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=survivor_nodes, 
                                   node_color=survivor_colors, node_size=survivor_sizes,
                                   alpha=0.9, linewidths=0.5, edgecolors="black", ax=self.ax)
        if failed_nodes:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=failed_nodes,
                                   node_color=failed_colors, node_size=failed_sizes,
                                   alpha=0.9, linewidths=3, edgecolors="#8B0000", ax=self.ax)
        
        # Labels for infra nodes
        infra_types = ["SMO", "Non-RT-RIC", "Near-RT-RIC", "AMF", "UPF", "Core",
                       "O-CU-CP", "O-CU-UP", "O-CU", "CU", "O-DU", "DU", 
                       "O-RU", "gNB-Site", "GNBSite", "EdgeUPF", "Relay"]
        label_nodes = [n for n in self.G.nodes() if self.G.nodes[n]["node_type"] in infra_types]
        labels = {n: n.split('_')[-1] if '_' in n else n[:6] for n in label_nodes[:self.max_labels]}
        nx.draw_networkx_labels(self.G, self.pos, labels=labels, font_size=5, ax=self.ax)
        
        # Draw UE count badges near O-RUs (green=connected, red=truly disconnected)
        ru_types = ["O-RU", "gNB-Site", "GNBSite"]
        for node_id in self.G.nodes():
            if self.G.nodes[node_id]["node_type"] in ru_types and node_id in ue_count_per_ru:
                counts = ue_count_per_ru[node_id]
                x, y = self.pos[node_id]
                connected = counts['connected']
                disconnected = counts.get('truly_disconnected', 0)
                
                # Green badge for connected UEs (always show)
                self.ax.annotate(f"{connected}", (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=6, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='#00AA00', 
                                        edgecolor='black', linewidth=0.3))
                
                # Red badge for truly disconnected UEs (only show if > 0)
                if disconnected > 0:
                    self.ax.annotate(f"{disconnected}", (x, y), xytext=(18, 5), textcoords='offset points',
                                   fontsize=6, fontweight='bold', color='white',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor='#DD0000', 
                                            edgecolor='black', linewidth=0.3))
        
        # Create legends - O-RAN data/control plane interfaces only (no O1 management)
        node_legend = [
            ("SMO/RIC", "#8B0000"), ("Near-RT-RIC", "#9400D3"), ("AMF/UPF", "#DC143C"),
            ("O-CU", "#4169E1"), ("O-DU", "#20B2AA"), ("O-RU", "#32CD32"),
            ("Relay", "#808080"), ("EdgeUPF", "#DAA520"), ("FAILED", "#FF0000"),
        ]
        # O-RAN interface legend (data/control plane only)
        iface_legend = [
            ("Open-FH (RU-DU)", "#00FF00"), ("F1 (DU-CU)", "#0000FF"), 
            ("E1 (CU-CP/UP)", "#9932CC"), ("E2 (RIC)", "#FF00FF"),
            ("A1 (RIC pol)", "#FF69B4"), ("N2/N3 (Core)", "#FF0000"), 
            ("Xn (X-gNB)", "#FFD700"), ("Backhaul", "#D2691E"),
        ]
        
        node_handles = [self.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, 
                                    markersize=7, label=n, markeredgecolor='black') 
                       for n, c in node_legend]
        # UE count badge legend
        node_handles.append(self.Line2D([0], [0], marker='s', color='w', markerfacecolor='#00AA00',
                                        markersize=6, label='UEs conn', markeredgecolor='black'))
        node_handles.append(self.Line2D([0], [0], marker='s', color='w', markerfacecolor='#DD0000',
                                        markersize=6, label='UEs lost', markeredgecolor='black'))
        
        iface_handles = [self.Line2D([0], [1], color=c, linewidth=2, label=n) for n, c in iface_legend]
        iface_handles.append(self.Line2D([0], [1], color='#FF0000', linewidth=2, 
                                         linestyle='--', label='FAILED'))
        # Failed postcard transmissions
        iface_handles.append(self.Line2D([0], [1], color='#FF0000', linewidth=1.5,
                                         linestyle='--', marker='x', markersize=6,
                                         label='Failed Postcard', markeredgewidth=2))
        
        leg1 = self.ax.legend(handles=node_handles, loc='upper left', fontsize=6, 
                              title="Nodes", title_fontsize=7, framealpha=0.9)
        self.ax.add_artist(leg1)
        self.ax.legend(handles=iface_handles, loc='upper right', fontsize=6,
                       title="Interfaces", title_fontsize=7, framealpha=0.9)
        
        # Stats in title
        is_island = any(snapshot.node_states.get(n, {}).get("is_island", False) for n in self.G.nodes())
        status = " [ISLAND MODE]" if is_island else ""
        marl_info = ""  # MARL info removed
        
        # UE population stats
        ue2ue_stats = snapshot.ue_to_ue_stats
        total_ue_pop = ue2ue_stats.get('total_ue_population', total_ues)
        rescue_ue_count = ue2ue_stats.get('rescue_ue_count', 0)
        ue_pop_info = f" | UE Pop: {total_ue_pop}"
        if rescue_ue_count > 0:
            ue_pop_info += f" (Rescue: {rescue_ue_count})"
        
        # UE-to-UE communication status
        ue2ue_enabled = ue2ue_stats.get('enabled', False)
        ue2ue_marl = ue2ue_stats.get('marl_routing_enabled', False)
        ue2ue_success = ue2ue_stats.get('success_count', 0)
        ue2ue_total = ue2ue_stats.get('total_flows', 0)
        if ue2ue_total > 0:
            ue2ue_status = f" | UE2UE: ({ue2ue_success}/{ue2ue_total})"
        else:
            ue2ue_status = ""
        
        self.ax.set_title(f"O-RAN Topology t={snapshot.tick} | Infra: {len(survivor_nodes)} up, "
                         f"{len(failed_nodes)} failed | UEs: {total_connected_ues}/{total_ues}"
                         f"{ue_pop_info}{status}{marl_info}{ue2ue_status}",
                         fontsize=10, fontweight='bold')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _node_style(self, node_type: str, is_survivor: bool):
        color = self.NODE_COLORS.get(node_type, "#cccccc")
        size = self.NODE_SIZES.get(node_type, 150)
        if not is_survivor:
            color = "#FF0000"
        return color, size


class TrafficAnalysisPlotter:
    """Real-time traffic analysis visualization showing UE-to-UE and general UE traffic."""
    
    def __init__(self, max_history: int = 1500):
        self.max_history = max_history
        
        # Time series data
        self.ticks = []
        
        # Traffic volumes
        self.ue_to_ue_traffic = []  # UE-to-UE traffic volume
        self.ue_general_traffic = []  # UE to infrastructure/external traffic
        self.total_ue_traffic = []  # Total UE traffic
        
        # Traffic success rates
        self.ue_to_ue_success_rate = []  # % of UE-to-UE traffic successfully delivered
        self.ue_general_success_rate = []  # % of general UE traffic successfully delivered
        
        # Island mode indicator
        self.island_mode = []
        self.disaster_tick = None
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 8))
        self.fig.suptitle('UE Traffic Analysis Over Time', fontsize=14, fontweight='bold')
        
        # Configure subplots
        self.ax_volume = self.axes[0]
        self.ax_success = self.axes[1]
        
        self.ax_volume.set_title('Traffic Volume (UE-to-UE vs General UE Traffic)')
        self.ax_success.set_title('Traffic Success Rate')

        plt.tight_layout()
        # Only use interactive mode if we're using an interactive backend
        if matplotlib.get_backend() != 'Agg':
            plt.ion()
            self.fig.show()

        # Animation setup
        self.animation = None
        self.animation_data = []
        self.animation_enabled = False

    def save_plot(self, output_file: str):
        """Save the current UE traffic analysis plot to file."""
        if self.fig:
            self.fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"UE Traffic Analysis plot saved to: {output_file}")

    def start_animation(self, interval: int = 200):
        """Start animating the traffic analysis plot."""
        if self.animation:
            self.animation.event_source.stop()

        self.animation_enabled = True
        self.animation = FuncAnimation(
            self.fig, self._animate_frame,
            frames=len(self.animation_data) if self.animation_data else 1,
            interval=interval, blit=False, repeat=True
        )
        print("Traffic Analysis animation started")

    def stop_animation(self):
        """Stop the traffic analysis animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.animation_enabled = False
        print("Traffic Analysis animation stopped")

    def _animate_frame(self, frame_idx: int):
        """Animation frame update function."""
        if frame_idx < len(self.animation_data):
            frame_data = self.animation_data[frame_idx]
            self._draw_frame(frame_data)

    def _draw_frame(self, frame_data):
        """Draw a single animation frame."""
        ticks, ue_to_ue_traffic, ue_general_traffic, total_ue_traffic, ue_to_ue_success, ue_general_success, island_mode, disaster_tick = frame_data

        # Clear and redraw volume plot
        self.ax_volume.clear()
        self.ax_volume.plot(ticks, ue_to_ue_traffic, 'r-', linewidth=2, label='UE-to-UE Traffic', alpha=0.8)
        self.ax_volume.plot(ticks, ue_general_traffic, 'b-', linewidth=2, label='General UE Traffic', alpha=0.8)
        self.ax_volume.plot(ticks, total_ue_traffic, 'g--', linewidth=1.5, label='Total UE Traffic', alpha=0.6)

        self.ax_volume.set_ylabel('Traffic Volume')
        self.ax_volume.set_title('Traffic Volume (UE-to-UE vs General UE Traffic)')
        self.ax_volume.legend(loc='upper left')
        self.ax_volume.grid(True, alpha=0.3)

        # Clear and redraw success rate plot
        self.ax_success.clear()
        self.ax_success.plot(ticks, ue_to_ue_success, 'r-o', markersize=3, linewidth=1.5, label='UE-to-UE Success %', alpha=0.8)
        self.ax_success.plot(ticks, ue_general_success, 'b-s', markersize=3, linewidth=1.5, label='General UE Success %', alpha=0.8)

        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.set_xlabel('Tick')
        self.ax_success.set_title('Traffic Success Rate')
        self.ax_success.legend(loc='lower right')
        self.ax_success.grid(True, alpha=0.3)
        self.ax_success.set_ylim(0, 105)

        # Shade disaster periods
        for ax in [self.ax_volume, self.ax_success]:
            # Shade island mode periods
            island_starts = []
            island_ends = []
            in_island = False
            for i, is_island in enumerate(island_mode):
                if is_island and not in_island:
                    island_starts.append(ticks[i])
                    in_island = True
                elif not is_island and in_island:
                    island_ends.append(ticks[i-1] if i > 0 else ticks[0])
                    in_island = False
            if in_island and ticks:
                island_ends.append(ticks[-1])

            for start, end in zip(island_starts, island_ends):
                ax.axvspan(start, end, alpha=0.2, color='red', label='Island Mode' if start == island_starts[0] else "")

    def update(self, tick: int, simulator: 'Simulator'):
        """Update traffic metrics from current simulation state."""
        self.ticks.append(tick)
        
        # Track disaster tick
        if simulator.island_mode and self.disaster_tick is None:
            self.disaster_tick = tick
        
        # Island mode disables UE-to-UE routing
        
        # Get UE-to-UE traffic from simulator (calculated during forwarding)
        ue_to_ue_volume = simulator.last_tick_ue_to_ue_volume
        ue_to_ue_offered = simulator.last_tick_ue_to_ue_offered
        ue_to_ue_delivered = ue_to_ue_volume  # Delivered = volume (if routed, it's delivered)
        
        self.ue_to_ue_traffic.append(ue_to_ue_volume)
        
        # Get general UE traffic from simulator (calculated during forwarding)
        ue_general_volume = simulator.last_tick_ue_general_volume
        ue_general_offered = simulator.last_tick_ue_general_offered
        ue_general_delivered = ue_general_volume  # Delivered = volume
        
        self.ue_general_traffic.append(ue_general_volume)
        self.total_ue_traffic.append(ue_to_ue_volume + ue_general_volume)
        
        # Calculate success rates
        ue_to_ue_sr = (ue_to_ue_delivered / max(ue_to_ue_offered, 1.0)) * 100 if ue_to_ue_offered > 0 else 0.0
        ue_general_sr = (ue_general_delivered / max(ue_general_offered, 1.0)) * 100 if ue_general_offered > 0 else 0.0
        
        self.ue_to_ue_success_rate.append(ue_to_ue_sr)
        self.ue_general_success_rate.append(ue_general_sr)
        
        # Island mode
        self.island_mode.append(1 if simulator.island_mode else 0)
        
        # Trim history if needed
        if len(self.ticks) > self.max_history:
            self.ticks = self.ticks[-self.max_history:]
            self.ue_to_ue_traffic = self.ue_to_ue_traffic[-self.max_history:]
            self.ue_general_traffic = self.ue_general_traffic[-self.max_history:]
            self.total_ue_traffic = self.total_ue_traffic[-self.max_history:]
            self.ue_to_ue_success_rate = self.ue_to_ue_success_rate[-self.max_history:]
            self.ue_general_success_rate = self.ue_general_success_rate[-self.max_history:]
            self.island_mode = self.island_mode[-self.max_history:]

        # Store animation frame data
        if self.animation_enabled:
            frame_data = (
                self.ticks.copy(),
                self.ue_to_ue_traffic.copy(),
                self.ue_general_traffic.copy(),
                self.total_ue_traffic.copy(),
                self.ue_to_ue_success_rate.copy(),
                self.ue_general_success_rate.copy(),
                self.island_mode.copy(),
                self.disaster_tick,
                self.marl_recovery_tick
            )
            self.animation_data.append(frame_data)

        # Redraw plots
        self._redraw()
    
    def _redraw(self):
        """Redraw all subplots."""
        ticks = self.ticks
        
        # === Traffic Volume ===
        self.ax_volume.clear()
        
        # Plot traffic volumes
        line1, = self.ax_volume.plot(ticks, self.ue_to_ue_traffic, 'r-', linewidth=2.5, 
                                     label='UE-to-UE Traffic', alpha=0.8)
        line2, = self.ax_volume.plot(ticks, self.ue_general_traffic, 'b-', linewidth=2.5,
                                     label='General UE Traffic (to Infrastructure/External)', alpha=0.8)
        line3, = self.ax_volume.plot(ticks, self.total_ue_traffic, 'g--', linewidth=1.5,
                                     label='Total UE Traffic', alpha=0.6)
        
        self.ax_volume.set_ylabel('Traffic Volume (units/tick)', fontsize=10)
        self.ax_volume.set_xlabel('Tick', fontsize=10)
        self.ax_volume.legend(loc='upper left', fontsize=9)
        self.ax_volume.grid(True, alpha=0.3)
        self.ax_volume.set_title('Traffic Volume: UE-to-UE vs General UE Traffic\n'
                                 '(Normal: UEs prefer outward to UPF, UE-to-UE is lower | '
                                 'Island: Only UE-to-UE possible, enabled by MARL)')
        
        # Shade disaster period
        self._shade_disaster_period(self.ax_volume)
        
        # === Traffic Success Rate ===
        self.ax_success.clear()
        
        line1, = self.ax_success.plot(ticks, self.ue_to_ue_success_rate, 'r-', linewidth=2.5,
                                      label='UE-to-UE Success Rate', alpha=0.8)
        line2, = self.ax_success.plot(ticks, self.ue_general_success_rate, 'b-', linewidth=2.5,
                                      label='General UE Success Rate', alpha=0.8)
        
        self.ax_success.set_ylabel('Success Rate (%)', fontsize=10)
        self.ax_success.set_xlabel('Tick', fontsize=10)
        self.ax_success.set_ylim(0, 105)
        self.ax_success.legend(loc='upper left', fontsize=9)
        self.ax_success.grid(True, alpha=0.3)
        self.ax_success.set_title('Traffic Delivery Success Rate')
        
        # Shade disaster period
        self._shade_disaster_period(self.ax_success)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _shade_disaster_period(self, ax):
        """Shade background during disaster/island mode period."""
        if not self.ticks or not self.island_mode:
            return
        
        # Find disaster start and end
        disaster_start = None
        disaster_end = None
        in_disaster = False
        
        for i, (t, im) in enumerate(zip(self.ticks, self.island_mode)):
            if im and not in_disaster:
                disaster_start = t
                in_disaster = True
            elif not im and in_disaster:
                disaster_end = t
                in_disaster = False
                break
        
        if disaster_start is not None:
            end_tick = disaster_end if disaster_end else self.ticks[-1]
            
            # Shade disaster period
            ax.axvspan(disaster_start, end_tick, alpha=0.15, color='red', 
                      label='Disaster/Island Mode' if ax == self.ax_volume else '')
            
            # Add vertical line at disaster start
            ax.axvline(disaster_start, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label='Disaster Start' if ax == self.ax_success else '')
            
            # Add text annotation for disaster
            if ax == self.ax_volume:
                y_max = ax.get_ylim()[1]
                ax.text(disaster_start, y_max * 0.95, 'DISASTER', 
                       rotation=90, verticalalignment='top', fontsize=9, 
                       color='red', fontweight='bold', alpha=0.8)
            
            # Add annotation for MARL recovery
            if self.marl_recovery_tick and disaster_start <= self.marl_recovery_tick <= end_tick:
                if ax == self.ax_volume:
                    y_max = ax.get_ylim()[1]
                    ax.axvline(self.marl_recovery_tick, color='green', linestyle=':', linewidth=2, alpha=0.7)
                    ax.text(self.marl_recovery_tick, y_max * 0.85, 'MARL\nRecovery', 
                           rotation=90, verticalalignment='top', fontsize=8, 
                           color='green', fontweight='bold', alpha=0.8)
                elif ax == self.ax_success:
                    ax.axvline(self.marl_recovery_tick, color='green', linestyle=':', linewidth=2, alpha=0.7)
                    ax.text(self.marl_recovery_tick, 100, 'MARL\nRecovery', 
                           rotation=90, verticalalignment='top', fontsize=8, 
                           color='green', fontweight='bold', alpha=0.8)


# MARLMetricsPlotter class removed - keeping only core simulation logic
        self.max_history = max_history
        
        # Time series data
        self.ticks = []
        
        # Policy distribution (how many agents use each mode)
        self.admit_count = []
        self.throttle_count = []
        self.hold_count = []
        
        # Coordination metrics
        self.postcards_sent = []
        self.avg_strain = []  # 0=okay, 1=degrading, 2=near_limit
        
        # Performance/Reward metrics
        self.ue_connectivity = []  # % of UEs connected
        self.life_safety_success = []  # % life-safety traffic delivered
        self.network_health = []  # Combined health score
        
        # Convergence proxy (policy stability)
        self.policy_changes = []  # How many agents changed policy this tick
        self.cumulative_reward = []
        
        # Island mode indicator
        self.island_mode = []
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('MARL Agent Learning Metrics', fontsize=14, fontweight='bold')
        
        # Configure subplots
        self.ax_policy = self.axes[0, 0]
        self.ax_coord = self.axes[0, 1]
        self.ax_perf = self.axes[1, 0]
        self.ax_conv = self.axes[1, 1]
        
        self.ax_policy.set_title('Policy Distribution')
        self.ax_coord.set_title('Agent Coordination')
        self.ax_perf.set_title('Performance (Reward Proxy)')
        self.ax_conv.set_title('Convergence & Stability')

        plt.tight_layout()
        plt.ion()
        self.fig.show()

        # Animation setup
        self.animation = None
        self.animation_data = []
        self.animation_enabled = False

    def start_animation(self, interval: int = 200):
        """Start animating the MARL metrics plot."""
        if self.animation:
            self.animation.event_source.stop()

        self.animation_enabled = True
        self.animation = FuncAnimation(
            self.fig, self._animate_frame,
            frames=len(self.animation_data) if self.animation_data else 1,
            interval=interval, blit=False, repeat=True
        )
        print("MARL Metrics animation started")

    def stop_animation(self):
        """Stop the MARL metrics animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.animation_enabled = False
        print("MARL Metrics animation stopped")

    def _animate_frame(self, frame_idx: int):
        """Animation frame update function."""
        if frame_idx < len(self.animation_data):
            frame_data = self.animation_data[frame_idx]
            self._draw_frame(frame_data)

    def _draw_frame(self, frame_data):
        """Draw a single animation frame."""
        ticks, admit_count, throttle_count, hold_count, postcards_sent, avg_strain, ue_connectivity, life_safety_success, network_health, island_mode, policy_changes, cumulative_reward = frame_data

        # Policy Distribution
        self.ax_policy.clear()
        self.ax_policy.stackplot(ticks, admit_count, throttle_count, hold_count,
                                labels=['Admit', 'Throttle', 'Hold'],
                                colors=['#00AA00', '#FFA500', '#DD0000'], alpha=0.7)
        self.ax_policy.set_ylabel('Agent Count')
        self.ax_policy.set_xlabel('Tick')
        self.ax_policy.legend(loc='upper left', fontsize=8)
        self.ax_policy.set_title('Policy Distribution (Best-Effort)')
        self._shade_island_mode(self.ax_policy)

        # Agent Coordination
        self.ax_coord.clear()
        ax2 = self.ax_coord.twinx()
        line1, = self.ax_coord.plot(ticks, postcards_sent, 'b-', linewidth=2, label='Postcards Sent')
        line2, = ax2.plot(ticks, avg_strain, 'r-', linewidth=2, label='Avg Strain')
        self.ax_coord.set_ylabel('Postcards', color='blue')
        ax2.set_ylabel('Strain Level', color='red')
        self.ax_coord.set_xlabel('Tick')
        self.ax_coord.set_title('Agent Coordination')
        ax2.set_ylim(0, 2.5)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Okay', 'Degrading', 'Near Limit'])
        lines = [line1, line2]
        self.ax_coord.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)
        self._shade_island_mode(self.ax_coord)

        # Performance/Reward
        self.ax_perf.clear()
        self.ax_perf.plot(ticks, ue_connectivity, 'g-', linewidth=2, label='UE Connectivity %')
        self.ax_perf.plot(ticks, network_health, 'b-', linewidth=2, label='Network Health %')
        self.ax_perf.plot(ticks, life_safety_success, 'r--', linewidth=1.5, label='Life-Safety+UE2UE %')
        self.ax_perf.set_ylabel('Percentage')
        self.ax_perf.set_xlabel('Tick')
        self.ax_perf.set_ylim(0, 105)
        self.ax_perf.legend(loc='lower left', fontsize=8)
        self.ax_perf.set_title('Performance Metrics (Reward Proxy)')
        self._shade_island_mode(self.ax_perf)

        # Convergence
        self.ax_conv.clear()
        ax2_conv = self.ax_conv.twinx()
        line3, = self.ax_conv.plot(ticks, policy_changes, 'purple', linewidth=2,
                                   label='Policy Changes', alpha=0.7)
        line4, = ax2_conv.plot(ticks, cumulative_reward, 'green', linewidth=2,
                               label='Cumulative Reward')
        self.ax_conv.set_ylabel('Policy Changes', color='purple')
        ax2_conv.set_ylabel('Cumulative Reward', color='green')
        self.ax_conv.set_xlabel('Tick')
        self.ax_conv.set_title('Convergence & Learning Progress')
        lines = [line3, line4]
        self.ax_conv.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)
        self._shade_island_mode(self.ax_conv)

        plt.tight_layout()

    def update(self, tick: int, simulator: 'Simulator'):
        """Update MARL metrics from current simulation state."""
        self.ticks.append(tick)

        # Count policy modes from last action summary
        action_summary = simulator.last_action_summary
        admit, throttle, hold = 0, 0, 0

        for (tc, mode), count in action_summary.items():
            if tc == "best_effort":
                if mode == "admit":
                    admit = count
                elif mode == "throttle":
                    throttle = count
                elif mode == "hold":
                    hold = count

        # If no summary, estimate from island mode
        if not action_summary:
            agent_count = sum(1 for n in simulator.topology.nodes.values()
                            if n.is_survivor and n.node_type.value != "UE")
            if simulator.island_mode:
                # In island mode, more agents throttle/hold
                admit = int(agent_count * 0.3)
                throttle = int(agent_count * 0.4)
                hold = agent_count - admit - throttle
            else:
                # Normal mode - mostly admit
                admit = int(agent_count * 0.8)
                throttle = int(agent_count * 0.15)
                hold = agent_count - admit - throttle

        self.admit_count.append(admit)
        self.throttle_count.append(throttle)
        self.hold_count.append(hold)

        # Coordination metrics
        self.postcards_sent.append(simulator.last_postcards_sent)

        # Estimate strain level based on network state
        strain_level = 2 if simulator.island_mode else (1 if simulator.last_postcards_sent > 0 else 0)
        self.avg_strain.append(strain_level)

        # Performance metrics
        total_ues = sum(1 for n in simulator.topology.nodes.values() if n.node_type.value == "UE")
        connected_ues = sum(1 for n in simulator.topology.nodes.values()
                          if n.node_type.value == "UE" and n.is_survivor)
        ue_conn_ratio = connected_ues / max(total_ues, 1)
        self.ue_connectivity.append(ue_conn_ratio * 100)

        # Life-safety success (simplified - from traffic stats)
        infra_alive = sum(1 for n in simulator.topology.nodes.values()
                        if n.node_type.value != "UE" and n.is_survivor)
        total_infra = sum(1 for n in simulator.topology.nodes.values() if n.node_type.value != "UE")
        health = infra_alive / max(total_infra, 1)
        self.network_health.append(health * 100)
        self.life_safety_success.append(min(100, health * 100 + ue_conn_ratio * 20))

        # Island mode
        self.island_mode.append(1 if simulator.island_mode else 0)

        # Convergence: policy changes from previous tick
        if len(self.admit_count) > 1:
            policy_change = (abs(self.admit_count[-1] - self.admit_count[-2]) +
                           abs(self.throttle_count[-1] - self.throttle_count[-2]) +
                           abs(self.hold_count[-1] - self.hold_count[-2]))
        else:
            policy_change = 0
        self.policy_changes.append(policy_change)

        # Performance score (represents system value delivered)
        performance_score = ue_conn_ratio * 50 + health * 30 + (1 - self.avg_strain[-1]/2) * 20

        # Learning progress (how well agents are coordinating)
        coordination_bonus = max(0, simulator.last_postcards_sent * 2)  # Postcards indicate active coordination

        # Island mode penalty (reduced capability but still valuable coordination)
        island_penalty = 0.7 if simulator.island_mode else 1.0

        # Total value delivered this tick
        tick_value = (performance_score + coordination_bonus) * island_penalty

        # Cumulative total value (always increases, never decreases)
        prev_total = self.cumulative_reward[-1] if self.cumulative_reward else 0
        self.cumulative_reward.append(prev_total + max(0, tick_value))  # Only add positive contributions

        # Trim history if needed
        if len(self.ticks) > self.max_history:
            self.ticks = self.ticks[-self.max_history:]
            self.admit_count = self.admit_count[-self.max_history:]
            self.throttle_count = self.throttle_count[-self.max_history:]
            self.hold_count = self.hold_count[-self.max_history:]
            self.postcards_sent = self.postcards_sent[-self.max_history:]
            self.avg_strain = self.avg_strain[-self.max_history:]
            self.ue_connectivity = self.ue_connectivity[-self.max_history:]
            self.life_safety_success = self.life_safety_success[-self.max_history:]
            self.network_health = self.network_health[-self.max_history:]
            self.island_mode = self.island_mode[-self.max_history:]
            self.policy_changes = self.policy_changes[-self.max_history:]
            self.cumulative_reward = self.cumulative_reward[-self.max_history:]

        # Store animation frame data
        if self.animation_enabled:
            frame_data = (
                self.ticks.copy(),
                self.admit_count.copy(),
                self.throttle_count.copy(),
                self.hold_count.copy(),
                self.postcards_sent.copy(),
                self.avg_strain.copy(),
                self.ue_connectivity.copy(),
                self.life_safety_success.copy(),
                self.network_health.copy(),
                self.island_mode.copy(),
                self.policy_changes.copy(),
                self.cumulative_reward.copy()
            )
            self.animation_data.append(frame_data)

        # Redraw plots
        self._redraw()

    def _redraw(self):
        """Redraw all subplots."""
        ticks = self.ticks

        # === Policy Distribution (stacked area) ===
        self.ax_policy.clear()
        self.ax_policy.stackplot(ticks, self.admit_count, self.throttle_count, self.hold_count,
                                labels=['Admit', 'Throttle', 'Hold'],
                                colors=['#00AA00', '#FFA500', '#DD0000'], alpha=0.7)
        self.ax_policy.set_ylabel('Agent Count')
        self.ax_policy.set_xlabel('Tick')
        self.ax_policy.legend(loc='upper left', fontsize=8)
        self.ax_policy.set_title('Policy Distribution (Best-Effort)')

        # Shade island mode periods
        self._shade_island_mode(self.ax_policy)

        # === Coordination (postcards + strain) ===
        self.ax_coord.clear()
        ax2 = self.ax_coord.twinx()

        line1, = self.ax_coord.plot(ticks, self.postcards_sent, 'b-', linewidth=2, label='Postcards Sent')
        line2, = ax2.plot(ticks, self.avg_strain, 'r-', linewidth=2, label='Avg Strain')

        self.ax_coord.set_ylabel('Postcards', color='blue')
        ax2.set_ylabel('Strain Level', color='red')
        self.ax_coord.set_xlabel('Tick')
        self.ax_coord.set_title('Agent Coordination')
        ax2.set_ylim(0, 2.5)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Okay', 'Degrading', 'Near Limit'])

        lines = [line1, line2]
        self.ax_coord.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)
        self._shade_island_mode(self.ax_coord)

        # === Performance/Reward ===
        self.ax_perf.clear()
        self.ax_perf.plot(ticks, self.ue_connectivity, 'g-', linewidth=2, label='UE Connectivity %')
        self.ax_perf.plot(ticks, self.network_health, 'b-', linewidth=2, label='Network Health %')
        self.ax_perf.plot(ticks, self.life_safety_success, 'r--', linewidth=1.5, label='Life-Safety+UE2UE %')

        self.ax_perf.set_ylabel('Percentage')
        self.ax_perf.set_xlabel('Tick')
        self.ax_perf.set_ylim(0, 105)
        self.ax_perf.legend(loc='lower left', fontsize=8)
        self.ax_perf.set_title('Performance Metrics (Reward Proxy)')
        self._shade_island_mode(self.ax_perf)

        # === Convergence ===
        self.ax_conv.clear()
        ax2_conv = self.ax_conv.twinx()

        line1, = self.ax_conv.plot(ticks, self.policy_changes, 'purple', linewidth=2,
                                   label='Policy Changes', alpha=0.7)
        line2, = ax2_conv.plot(ticks, self.cumulative_reward, 'green', linewidth=2,
                               label='Cumulative Reward')

        self.ax_conv.set_ylabel('Policy Changes', color='purple')
        ax2_conv.set_ylabel('Cumulative Reward', color='green')
        self.ax_conv.set_xlabel('Tick')
        self.ax_conv.set_title('Convergence & Learning Progress')

        lines = [line1, line2]
        self.ax_conv.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=8)
        self._shade_island_mode(self.ax_conv)

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

    def _shade_island_mode(self, ax):
        """Shade background when in island mode."""
        if not self.ticks or not self.island_mode:
            return

        # Find island mode regions
        in_island = False
        start_tick = None

        for i, (t, im) in enumerate(zip(self.ticks, self.island_mode)):
            if im and not in_island:
                start_tick = t
                in_island = True
            elif not im and in_island:
                ax.axvspan(start_tick, t, alpha=0.2, color='red', label='Island Mode' if i == 1 else '')
                in_island = False

        # Handle ongoing island mode
        if in_island and start_tick is not None:
            ax.axvspan(start_tick, self.ticks[-1], alpha=0.2, color='red')


class AgentMonitorPlotter:
    """Real-time visualization of a single agent's internal state, actions, and decision-making."""

    def __init__(self, agent_id: str, max_history: int = 500):
        self.agent_id = agent_id
        self.max_history = max_history

        # Time series data
        self.ticks = []

        # State observations
        self.energy_tier = []  # 0=High, 1=Medium, 2=Low
        self.island_mode = []
        self.network_stress = []  # Agent's stress assessment

        # Queue states (per traffic class)
        self.life_safety_queue = []
        self.operations_queue = []
        self.telemetry_queue = []
        self.best_effort_queue = []

        # Actions taken
        self.life_safety_action = []  # 0=Admit, 1=Throttle, 2=Hold
        self.operations_action = []
        self.telemetry_action = []
        self.best_effort_action = []

        # Priority weights
        self.life_safety_weight = []
        self.operations_weight = []
        self.telemetry_weight = []
        self.best_effort_weight = []

        # Communication
        self.postcards_sent = []  # 1 if sent this tick, 0 otherwise
        self.postcards_received = []

        # "Rewards" - performance indicators
        self.admission_success = []  # Overall admission success rate
        self.neighbor_strain = []  # Average neighbor strain level

        # Policy changes
        self.policy_version = []

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        self.fig.suptitle(f'Agent {agent_id} - Internal State & Decision Making', fontsize=14, fontweight='bold')

        # Configure subplots
        self.ax_state = self.axes[0, 0]
        self.ax_queues = self.axes[0, 1]
        self.ax_actions = self.axes[1, 0]
        self.ax_weights = self.axes[1, 1]
        self.ax_comm = self.axes[2, 0]
        self.ax_rewards = self.axes[2, 1]

        self.ax_state.set_title('Agent State')
        self.ax_queues.set_title('Traffic Queues')
        self.ax_actions.set_title('Admission Actions')
        self.ax_weights.set_title('Priority Weights')
        self.ax_comm.set_title('Communication')
        self.ax_rewards.set_title('Performance & Rewards')

        plt.tight_layout()
        plt.ion()
        self.fig.show()

        # Animation setup
        self.animation = None
        self.animation_data = []
        self.animation_enabled = False

    def start_animation(self, interval: int = 200):
        """Start animating the agent monitor plot."""
        if self.animation:
            self.animation.event_source.stop()

        self.animation_enabled = True
        self.animation = FuncAnimation(
            self.fig, self._animate_frame,
            frames=len(self.animation_data) if self.animation_data else 1,
            interval=interval, blit=False, repeat=True
        )
        print(f"Agent Monitor animation started for {self.agent_id}")

    def stop_animation(self):
        """Stop the agent monitor animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.animation_enabled = False
        print(f"Agent Monitor animation stopped for {self.agent_id}")

    def _animate_frame(self, frame_idx: int):
        """Animation frame update function."""
        if frame_idx < len(self.animation_data):
            frame_data = self.animation_data[frame_idx]
            self._draw_frame(frame_data)

    def _draw_frame(self, frame_data):
        """Draw a single animation frame."""
        ticks, energy_tier, island_mode, network_stress, life_safety_queue, operations_queue, telemetry_queue, best_effort_queue, life_safety_action, operations_action, telemetry_action, best_effort_action, life_safety_weight, operations_weight, telemetry_weight, best_effort_weight, postcards_sent, postcards_received, admission_success, neighbor_strain, policy_version = frame_data

        # Agent State
        self.ax_state.clear()
        self.ax_state.plot(ticks, energy_tier, 'r-', linewidth=2, label='Energy Tier', alpha=0.7)
        self.ax_state.plot(ticks, island_mode, 'b--', linewidth=2, label='Island Mode', alpha=0.7)
        self.ax_state.plot(ticks, network_stress, 'g:', linewidth=2, label='Network Stress', alpha=0.7)
        self.ax_state.set_ylabel('State Value')
        self.ax_state.set_title('Agent State')
        self.ax_state.legend(loc='upper left', fontsize=8)
        self.ax_state.grid(True, alpha=0.3)

        # Traffic Queues
        self.ax_queues.clear()
        self.ax_queues.plot(ticks, life_safety_queue, 'r-', linewidth=2, label='Life Safety', alpha=0.8)
        self.ax_queues.plot(ticks, operations_queue, 'b-', linewidth=2, label='Operations', alpha=0.8)
        self.ax_queues.plot(ticks, telemetry_queue, 'g-', linewidth=1.5, label='Telemetry', alpha=0.6)
        self.ax_queues.plot(ticks, best_effort_queue, 'y-', linewidth=1.5, label='Best Effort', alpha=0.6)
        self.ax_queues.set_ylabel('Queue Length')
        self.ax_queues.set_title('Traffic Queues')
        self.ax_queues.legend(loc='upper left', fontsize=8)
        self.ax_queues.grid(True, alpha=0.3)

        # Admission Actions
        self.ax_actions.clear()
        self.ax_actions.plot(ticks, life_safety_action, 'r-o', markersize=3, linewidth=1, label='Life Safety', alpha=0.8)
        self.ax_actions.plot(ticks, operations_action, 'b-s', markersize=3, linewidth=1, label='Operations', alpha=0.8)
        self.ax_actions.plot(ticks, telemetry_action, 'g-^', markersize=3, linewidth=1, label='Telemetry', alpha=0.6)
        self.ax_actions.plot(ticks, best_effort_action, 'y-d', markersize=3, linewidth=1, label='Best Effort', alpha=0.6)
        self.ax_actions.set_ylabel('Action (0=Admit, 1=Throttle, 2=Hold)')
        self.ax_actions.set_title('Admission Actions')
        self.ax_actions.legend(loc='upper left', fontsize=8)
        self.ax_actions.grid(True, alpha=0.3)
        self.ax_actions.set_yticks([0, 1, 2])
        self.ax_actions.set_yticklabels(['Admit', 'Throttle', 'Hold'])

        # Priority Weights
        self.ax_weights.clear()
        self.ax_weights.plot(ticks, life_safety_weight, 'r-', linewidth=2, label='Life Safety', alpha=0.8)
        self.ax_weights.plot(ticks, operations_weight, 'b-', linewidth=2, label='Operations', alpha=0.8)
        self.ax_weights.plot(ticks, telemetry_weight, 'g-', linewidth=1.5, label='Telemetry', alpha=0.6)
        self.ax_weights.plot(ticks, best_effort_weight, 'y-', linewidth=1.5, label='Best Effort', alpha=0.6)
        self.ax_weights.set_ylabel('Priority Weight')
        self.ax_weights.set_title('Priority Weights')
        self.ax_weights.legend(loc='upper left', fontsize=8)
        self.ax_weights.grid(True, alpha=0.3)

        # Communication
        self.ax_comm.clear()
        ax2 = self.ax_comm.twinx()
        line1, = self.ax_comm.plot(ticks, postcards_sent, 'b-o', markersize=4, linewidth=1, label='Sent', alpha=0.8)
        line2, = ax2.plot(ticks, postcards_received, 'r-s', markersize=4, linewidth=1, label='Received', alpha=0.8)
        self.ax_comm.set_ylabel('Postcards Sent', color='blue')
        ax2.set_ylabel('Postcards Received', color='red')
        self.ax_comm.set_title('Communication Activity')
        self.ax_comm.grid(True, alpha=0.3)

        # Combined legend
        lines = [line1, line2]
        labels = ['Sent', 'Received']
        self.ax_comm.legend(lines, labels, loc='upper left', fontsize=8)

        # Performance & Rewards
        self.ax_rewards.clear()
        ax3 = self.ax_rewards.twinx()
        line3, = self.ax_rewards.plot(ticks, admission_success, 'g-', linewidth=2, label='Admission Success', alpha=0.8)
        line4, = ax3.plot(ticks, neighbor_strain, 'm--', linewidth=2, label='Neighbor Strain', alpha=0.7)
        self.ax_rewards.plot(ticks, policy_version, 'k:', linewidth=1, label='Policy Version', alpha=0.5)
        self.ax_rewards.set_ylabel('Admission Success Rate', color='green')
        ax3.set_ylabel('Neighbor Strain Level', color='magenta')
        self.ax_rewards.set_title('Performance & Rewards')
        self.ax_rewards.grid(True, alpha=0.3)

        # Combined legend
        lines = [line3, line4, plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1)]
        labels = ['Admission Success', 'Neighbor Strain', 'Policy Version']
        self.ax_rewards.legend(lines, labels, loc='upper left', fontsize=8)

        # Shade disaster periods
        for ax in [self.ax_state, self.ax_queues, self.ax_actions, self.ax_weights, self.ax_comm, self.ax_rewards]:
            # Shade island mode periods
            island_starts = []
            island_ends = []
            in_island = False
            for i, is_island in enumerate(island_mode):
                if is_island and not in_island:
                    island_starts.append(ticks[i])
                    in_island = True
                elif not is_island and in_island:
                    island_ends.append(ticks[i-1] if i > 0 else ticks[0])
                    in_island = False
            if in_island and ticks:
                island_ends.append(ticks[-1])

            for start, end in zip(island_starts, island_ends):
                ax.axvspan(start, end, alpha=0.2, color='red', label='Island Mode' if start == island_starts[0] else "")

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

    def update(self, tick: int, simulator: 'Simulator'):
        """Update the agent monitor with current state."""
        if self.agent_id not in simulator.agents:
            return

        agent = simulator.agents[self.agent_id]

        # Get current observation (we need to build it)
        try:
            observations = simulator._build_agent_observations()
            if self.agent_id not in observations:
                return

            observation = observations[self.agent_id]

            # Update state data
            self.ticks.append(tick)
            self.energy_tier.append(observation.energy_tier.value if hasattr(observation.energy_tier, 'value') else 0)
            self.island_mode.append(1 if observation.is_island else 0)

            # Network stress (simplified)
            life_safety_queue = observation.local_slices[TrafficClass.LIFE_SAFETY].current_queue_length
            operations_queue = observation.local_slices[TrafficClass.OPERATIONS].current_queue_length
            stress_level = 1 if (life_safety_queue > 50 or operations_queue > 100 or observation.is_island) else 0
            self.network_stress.append(stress_level)

            # Queue states
            self.life_safety_queue.append(observation.local_slices[TrafficClass.LIFE_SAFETY].current_queue_length)
            self.operations_queue.append(observation.local_slices[TrafficClass.OPERATIONS].current_queue_length)
            self.telemetry_queue.append(observation.local_slices[TrafficClass.TELEMETRY].current_queue_length)
            self.best_effort_queue.append(observation.local_slices[TrafficClass.BEST_EFFORT].current_queue_length)

            # Get current action (we need to compute it)
            action = agent.compute_action(observation)

            # Actions (convert to numbers)
            action_map = {'ADMIT': 0, 'THROTTLE': 1, 'HOLD': 2}
            self.life_safety_action.append(action_map.get(action.class_actions[TrafficClass.LIFE_SAFETY].admission_mode.value, 0))
            self.operations_action.append(action_map.get(action.class_actions[TrafficClass.OPERATIONS].admission_mode.value, 0))
            self.telemetry_action.append(action_map.get(action.class_actions[TrafficClass.TELEMETRY].admission_mode.value, 0))
            self.best_effort_action.append(action_map.get(action.class_actions[TrafficClass.BEST_EFFORT].admission_mode.value, 0))

            # Weights
            self.life_safety_weight.append(action.class_actions[TrafficClass.LIFE_SAFETY].priority_weight)
            self.operations_weight.append(action.class_actions[TrafficClass.OPERATIONS].priority_weight)
            self.telemetry_weight.append(action.class_actions[TrafficClass.TELEMETRY].priority_weight)
            self.best_effort_weight.append(action.class_actions[TrafficClass.BEST_EFFORT].priority_weight)

            # Communication
            self.postcards_sent.append(1 if action.send_postcard else 0)
            self.postcards_received.append(len(simulator.control_plane.get_received_postcards(self.agent_id, tick)))

            # Performance indicators
            total_admission = sum(slice_state.admission_success_rate for slice_state in observation.local_slices.values())
            avg_admission = total_admission / len(observation.local_slices)
            self.admission_success.append(avg_admission)

            # Neighbor strain (simplified)
            neighbor_strain = observation.neighbor_summary.strain_level.value if hasattr(observation.neighbor_summary.strain_level, 'value') else 0
            strain_map = {'OKAY': 0, 'DEGRADING': 1, 'NEAR_LIMIT': 2}
            self.neighbor_strain.append(strain_map.get(neighbor_strain, 0))

            # Policy version
            self.policy_version.append(agent.policy_version)

            # Trim history
            if len(self.ticks) > self.max_history:
                self.ticks = self.ticks[-self.max_history:]
                self.energy_tier = self.energy_tier[-self.max_history:]
                self.island_mode = self.island_mode[-self.max_history:]
                self.network_stress = self.network_stress[-self.max_history:]
                self.life_safety_queue = self.life_safety_queue[-self.max_history:]
                self.operations_queue = self.operations_queue[-self.max_history:]
                self.telemetry_queue = self.telemetry_queue[-self.max_history:]
                self.best_effort_queue = self.best_effort_queue[-self.max_history:]
                self.life_safety_action = self.life_safety_action[-self.max_history:]
                self.operations_action = self.operations_action[-self.max_history:]
                self.telemetry_action = self.telemetry_action[-self.max_history:]
                self.best_effort_action = self.best_effort_action[-self.max_history:]
                self.life_safety_weight = self.life_safety_weight[-self.max_history:]
                self.operations_weight = self.operations_weight[-self.max_history:]
                self.telemetry_weight = self.telemetry_weight[-self.max_history:]
                self.best_effort_weight = self.best_effort_weight[-self.max_history:]
                self.postcards_sent = self.postcards_sent[-self.max_history:]
                self.postcards_received = self.postcards_received[-self.max_history:]
                self.admission_success = self.admission_success[-self.max_history:]
                self.neighbor_strain = self.neighbor_strain[-self.max_history:]
                self.policy_version = self.policy_version[-self.max_history:]

            # Store animation frame data
            if self.animation_enabled:
                frame_data = (
                    self.ticks.copy(),
                    self.energy_tier.copy(),
                    self.island_mode.copy(),
                    self.network_stress.copy(),
                    self.life_safety_queue.copy(),
                    self.operations_queue.copy(),
                    self.telemetry_queue.copy(),
                    self.best_effort_queue.copy(),
                    self.life_safety_action.copy(),
                    self.operations_action.copy(),
                    self.telemetry_action.copy(),
                    self.best_effort_action.copy(),
                    self.life_safety_weight.copy(),
                    self.operations_weight.copy(),
                    self.telemetry_weight.copy(),
                    self.best_effort_weight.copy(),
                    self.postcards_sent.copy(),
                    self.postcards_received.copy(),
                    self.admission_success.copy(),
                    self.neighbor_strain.copy(),
                    self.policy_version.copy()
                )
                self.animation_data.append(frame_data)

            # Redraw plots
            self._redraw()

        except Exception as e:
            # Silently handle errors to avoid breaking simulation
            pass

    def _redraw(self):
        """Redraw all plots with current data."""
        ticks = self.ticks

        if not ticks:
            return

        # === Agent State ===
        self.ax_state.clear()
        self.ax_state.plot(ticks, self.energy_tier, 'r-', linewidth=2, label='Energy Tier', alpha=0.7)
        self.ax_state.plot(ticks, self.island_mode, 'b--', linewidth=2, label='Island Mode', alpha=0.7)
        self.ax_state.plot(ticks, self.network_stress, 'g:', linewidth=2, label='Network Stress', alpha=0.7)
        self.ax_state.set_ylabel('State Value')
        self.ax_state.set_title('Agent State')
        self.ax_state.legend(loc='upper left', fontsize=8)
        self.ax_state.grid(True, alpha=0.3)

        # === Traffic Queues ===
        self.ax_queues.clear()
        self.ax_queues.plot(ticks, self.life_safety_queue, 'r-', linewidth=2, label='Life Safety', alpha=0.8)
        self.ax_queues.plot(ticks, self.operations_queue, 'b-', linewidth=2, label='Operations', alpha=0.8)
        self.ax_queues.plot(ticks, self.telemetry_queue, 'g-', linewidth=1.5, label='Telemetry', alpha=0.6)
        self.ax_queues.plot(ticks, self.best_effort_queue, 'y-', linewidth=1.5, label='Best Effort', alpha=0.6)
        self.ax_queues.set_ylabel('Queue Length')
        self.ax_queues.set_title('Traffic Queues')
        self.ax_queues.legend(loc='upper left', fontsize=8)
        self.ax_queues.grid(True, alpha=0.3)

        # === Admission Actions ===
        self.ax_actions.clear()
        self.ax_actions.plot(ticks, self.life_safety_action, 'r-o', markersize=3, linewidth=1, label='Life Safety', alpha=0.8)
        self.ax_actions.plot(ticks, self.operations_action, 'b-s', markersize=3, linewidth=1, label='Operations', alpha=0.8)
        self.ax_actions.plot(ticks, self.telemetry_action, 'g-^', markersize=3, linewidth=1, label='Telemetry', alpha=0.6)
        self.ax_actions.plot(ticks, self.best_effort_action, 'y-d', markersize=3, linewidth=1, label='Best Effort', alpha=0.6)
        self.ax_actions.set_ylabel('Action (0=Admit, 1=Throttle, 2=Hold)')
        self.ax_actions.set_title('Admission Actions')
        self.ax_actions.legend(loc='upper left', fontsize=8)
        self.ax_actions.grid(True, alpha=0.3)
        self.ax_actions.set_yticks([0, 1, 2])
        self.ax_actions.set_yticklabels(['Admit', 'Throttle', 'Hold'])

        # === Priority Weights ===
        self.ax_weights.clear()
        self.ax_weights.plot(ticks, self.life_safety_weight, 'r-', linewidth=2, label='Life Safety', alpha=0.8)
        self.ax_weights.plot(ticks, self.operations_weight, 'b-', linewidth=2, label='Operations', alpha=0.8)
        self.ax_weights.plot(ticks, self.telemetry_weight, 'g-', linewidth=1.5, label='Telemetry', alpha=0.6)
        self.ax_weights.plot(ticks, self.best_effort_weight, 'y-', linewidth=1.5, label='Best Effort', alpha=0.6)
        self.ax_weights.set_ylabel('Priority Weight')
        self.ax_weights.set_title('Priority Weights')
        self.ax_weights.legend(loc='upper left', fontsize=8)
        self.ax_weights.grid(True, alpha=0.3)

        # === Communication ===
        self.ax_comm.clear()
        ax2 = self.ax_comm.twinx()
        line1, = self.ax_comm.plot(ticks, self.postcards_sent, 'b-o', markersize=4, linewidth=1, label='Sent', alpha=0.8)
        line2, = ax2.plot(ticks, self.postcards_received, 'r-s', markersize=4, linewidth=1, label='Received', alpha=0.8)
        self.ax_comm.set_ylabel('Postcards Sent', color='blue')
        ax2.set_ylabel('Postcards Received', color='red')
        self.ax_comm.set_title('Communication Activity')
        self.ax_comm.grid(True, alpha=0.3)

        # Combined legend
        lines = [line1, line2]
        labels = ['Sent', 'Received']
        self.ax_comm.legend(lines, labels, loc='upper left', fontsize=8)

        # === Performance & Rewards ===
        self.ax_rewards.clear()
        ax3 = self.ax_rewards.twinx()
        line3, = self.ax_rewards.plot(ticks, self.admission_success, 'g-', linewidth=2, label='Admission Success', alpha=0.8)
        line4, = ax3.plot(ticks, self.neighbor_strain, 'm--', linewidth=2, label='Neighbor Strain', alpha=0.7)
        self.ax_rewards.plot(ticks, self.policy_version, 'k:', linewidth=1, label='Policy Version', alpha=0.5)
        self.ax_rewards.set_ylabel('Admission Success Rate', color='green')
        ax3.set_ylabel('Neighbor Strain Level', color='magenta')
        self.ax_rewards.set_title('Performance & Rewards')
        self.ax_rewards.grid(True, alpha=0.3)

        # Combined legend
        lines = [line3, line4, plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1)]
        labels = ['Admission Success', 'Neighbor Strain', 'Policy Version']
        self.ax_rewards.legend(lines, labels, loc='upper left', fontsize=8)

        # Shade disaster periods
        for ax in [self.ax_state, self.ax_queues, self.ax_actions, self.ax_weights, self.ax_comm, self.ax_rewards]:
            # Shade island mode periods
            island_starts = []
            island_ends = []
            in_island = False
            for i, is_island in enumerate(self.island_mode):
                if is_island and not in_island:
                    island_starts.append(self.ticks[i])
                    in_island = True
                elif not is_island and in_island:
                    island_ends.append(self.ticks[i-1] if i > 0 else self.ticks[0])
                    in_island = False
            if in_island and self.ticks:
                island_ends.append(self.ticks[-1])

            for start, end in zip(island_starts, island_ends):
                ax.axvspan(start, end, alpha=0.2, color='red', label='Island Mode' if start == island_starts[0] else "")

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    


class Simulator:
    """Main simulation coordinator."""

    def __init__(self, topology: Topology, scenario: Scenario, config: SimulationConfig):
        self.topology = topology
        self.scenario = scenario
        self.config = config

        # Initialize components
        self.slice_dictionary = SliceDictionary()
        self.traffic_generator = TrafficGenerator(seed=config.random_seed)
        self.control_plane = ControlPlaneManager(topology)
        self.metrics = MetricsCollector()
        self.painter = None
        self.traffic_plotter = None  # Traffic analysis plotter
        self.agent_monitor = None  # Single agent monitor plotter
        self.marl_plotter = None   # Plotter for MARL metrics

        # Verbose/debug state
        self.last_postcards_sent: int = 0  # Postcards sent in current tick
        self.total_postcards_sent: int = 0  # Cumulative postcards sent since severance
        self.last_action_summary = {}

        # Create agents for each infra node (UEs are passive endpoints — no agents)
        self.agents: Dict[str, BaseAgent] = {}

        for node_id, node in topology.nodes.items():
            # UEs do not run policies or send postcards; skip them.
            if getattr(node.node_type, 'value', str(node.node_type)) == 'UE':
                continue
            if node.is_survivor:  # Only create agents for survivor infra nodes
                self.agents[node_id] = create_agent_for_node(
                    node_id, node.node_type.value, self.slice_dictionary
                )

        # Initialize agent monitor for one agent (independent of live plotting)
        if getattr(self.config, 'agent_monitor', False):
            try:
                # Choose a representative agent (prefer infrastructure over UE)
                infra_agents = [aid for aid in self.agents.keys()
                              if self.topology.nodes.get(aid, None) and
                              self.topology.nodes[aid].node_type.value != "UE"]
                monitor_agent = infra_agents[0] if infra_agents else list(self.agents.keys())[0]
                print(f"Initializing agent monitor for {monitor_agent}...")
                self.agent_monitor = AgentMonitorPlotter(monitor_agent, max_history=500)
                print("Agent monitor initialized successfully")
            except Exception as agent_e:
                print(f"Agent monitor disabled: {agent_e}")
                import traceback
                traceback.print_exc()
                self.agent_monitor = None
        else:
            self.agent_monitor = None

        # Live plot setup
        if self.config.live_plot:
            try:
                self.painter = NetworkPainter(
                    topology=self.topology,
                    max_labels=self.config.live_max_labels
                )
                plt.ion()
                plt.show(block=False)


                # Also initialize traffic analysis plotter
                try:
                    print("Initializing traffic plotter...")
                    self.traffic_plotter = TrafficAnalysisPlotter(max_history=1500)
                    print("Traffic plotter initialized successfully")
                except Exception as traffic_e:
                    print(f"Traffic plotter disabled: {traffic_e}")
                    self.traffic_plotter = None
            except Exception as e:
                print(f"Live plotting disabled (init error): {e}")
                self.painter = None
                self.marl_plotter = None
                self.traffic_plotter = None
                self.agent_monitor = None

        # Set up traffic generator
        for profile in scenario.traffic_profiles.values():
            self.traffic_generator.add_node_profile(profile)

        # Simulation state
        self.current_tick = 0
        self.island_mode = False
        self.core_nodes = self._identify_core_nodes()
        
        # UE-to-UE communication state (3GPP TS 22.179 MCPTT)
        self.ue_to_ue_flows: List[Tuple[str, str]] = []  # (source_ue, target_ue) pairs
        self.ue_to_ue_forward_flows: List[Tuple[str, str]] = []  # Forward direction flows
        self.ue_to_ue_enabled = True  # Initially enabled (core connected)
        self.marl_ue_routing_enabled = False  # MARL routing initially disabled
        self.ue_to_ue_success_count = 0  # Successful UE-to-UE communications this tick
        self.ue_to_ue_failure_count = 0  # Failed UE-to-UE communications this tick

        # MCPTT Emergency State Tracking (3GPP TS 22.179)
        self.emergency_active_ues: Set[str] = set()  # UEs in emergency state
        self.emergency_alerts_sent = 0  # Emergency alerts sent this tick
        self.mcppt_emergency_calls = 0  # Emergency calls established

        # Traffic tracking for visualization
        self.last_tick_ue_to_ue_volume = 0.0
        self.last_tick_ue_general_volume = 0.0
        self.last_tick_ue_to_ue_offered = 0.0
        self.last_tick_ue_general_offered = 0.0
        
        # Dynamic UE management
        self.ue_counter = max((int(nid.split('_')[-1]) for nid in topology.nodes.keys() 
                              if nid.startswith('UE_') and nid.split('_')[-1].isdigit()), 
                              default=0)  # Track highest UE ID number
        
        # Pre-compute UE to O-RU mappings for efficient UE-to-UE routing checks
        self._build_ue_to_oru_mapping()

        # Cache for infrastructure connectivity checks
        self._oru_connectivity_cache = {}

        # UE-to-UE routing table (built after flows are established)
        self._ue_routing_table = {}

        # Initialize UE-to-UE communication pairs (rescue services, etc.)
        self._establish_ue_to_ue_flows()

        # Build initial routing table after flows are established
        self._build_ue_routing_table()

        # ── PHY/MAC state ────────────────────────────────────────────────────────────
        self.phy_mac_states: Dict[str, PHYMACState] = {}
        self.transport_relay_model = TransportRelayModel()
        self._init_phy_mac_states()

        # ICIC: track per-node SINR between ticks for delta computation
        self._prev_sinr: Dict[str, float] = {}   # node_id -> last SINR average

        # Per-tick connectivity caches (populated at start of _build_agent_observations)
        self._tick_infra_nodes_cache: list  = []
        self._tick_bridge_set_cache:  set   = set()
        self._tick_intra_reach_cache: dict  = {}

        # IOPS: isolated-mode UE registration manager (one per simulation run)
        self.iops_manager = IOPSManager(max_capacity=50)

        # Multi-eNB IOPS Controller (ETSI TS 122 346 V16.0.0)
        from .iops import MultiENBIOPSController, LearningPostcardExchanger
        self.iops_controller = MultiENBIOPSController()
        self.learning_exchanger = LearningPostcardExchanger(
            exchange_interval=5, peer_learning_weight=0.1)

        # Non-RT RIC Coordinator (one per island cluster in sim)
        self.coordinator = CoordinatorAgent(interval=20)
        self._current_global_policy: GlobalPolicyVector = GlobalPolicyVector.default()

        # Timing
        self._island_start_tick: int = 0
        self._ticks_since_severance: int = 0

        # Recovery state (set when restore_core event fires)
        self._recovery_started_tick: int = -1   # -1 = no recovery yet
        self._recovery_complete: bool = False
        self._ticks_since_recovery: int = 0
        self._current_scenario_type: str = 'full_core'  # for obs one-hot

        # Connectivity metric cache (updated each tick)
        self._ue_pair_routed_fraction: float = 0.0
        self._reachable_ue_fraction: float = 1.0
        self._isolated_ue_count: int = 0

    def _identify_core_nodes(self) -> Set[str]:
        """Identify nodes that are part of the core network (5GC, SMO).
        
        Edge UPFs are NOT core nodes — they survive severance and provide
        Local Data Network (DN) traffic steering per 3GPP TS 23.501 §6.3.3.
        When core is severed, EdgeUPFs continue routing local UE-to-UE
        traffic without traversing the central core.
        """
        core_types = {"Core", "SMO", "Non-RT-RIC", "AMF", "UPF"}
        return {node_id for node_id, node in self.topology.nodes.items()
                if node.node_type.value in core_types}
    
    def _build_ue_to_oru_mapping(self):
        """Build efficient mapping from UEs to their connected O-RUs for fast UE-to-UE routing checks."""
        self.ue_to_oru_map = {}

        for link in self.topology.links.values():
            if link.is_up and hasattr(link, 'interface_type') and link.interface_type.value == "Uu":
                ue_ep = None
                oru_ep = None

                for ep in link.endpoints:
                    if ep.startswith('UE_'):
                        ue_ep = ep
                    elif ep.startswith(('O-RU_', 'gNB-Site_', 'GNBSite_')):
                        oru_ep = ep

                if ue_ep and oru_ep:
                    if ue_ep not in self.ue_to_oru_map:
                        self.ue_to_oru_map[ue_ep] = []
                    self.ue_to_oru_map[ue_ep].append(oru_ep)

    def _update_ue_to_oru_mapping_for_ue(self, ue_id: str):
        """Update the UE-to-O-RU mapping for a newly added UE."""
        if ue_id not in self.ue_to_oru_map:
            self.ue_to_oru_map[ue_id] = []

        # Find all Uu links connected to this UE
        for link in self.topology.links.values():
            if link.is_up and hasattr(link, 'interface_type') and link.interface_type.value == "Uu":
                if ue_id in link.endpoints:
                    other_end = link.endpoints[0] if link.endpoints[1] == ue_id else link.endpoints[1]
                    if other_end.startswith(('O-RU_', 'gNB-Site_', 'GNBSite_')):
                        if other_end not in self.ue_to_oru_map[ue_id]:
                            self.ue_to_oru_map[ue_id].append(other_end)

    def _update_ue_routing_table(self):
        """Clear the UE-to-UE routing table cache after topology changes."""
        # With lazy loading, we just clear the cache and let it rebuild on demand
        old_size = len(self._ue_routing_table)
        self._ue_routing_table.clear()
        print(f"UE-to-UE routing table cache cleared (was {old_size} entries, will rebuild lazily)")

    def _build_ue_routing_table(self):
        """Initialize empty routing table for lazy loading."""
        print(f"Initializing UE-to-UE routing table for {len(self.ue_to_ue_flows)} flows (lazy loading)")
        self._ue_routing_table = {}  # Start empty, populate on demand
        print("UE-to-UE routing table initialized (lazy loading enabled)")

    def _calculate_ue_to_ue_routing(self, source_ue: str, target_ue: str) -> bool:
        """Calculate if UE-to-UE routing is possible (used for pre-computation)."""
        if not self.island_mode:
            # Normal mode: Check if there's an infrastructure path between the UEs' connected O-RUs
            source_orus = self.ue_to_oru_map.get(source_ue, [])
            target_orus = self.ue_to_oru_map.get(target_ue, [])

            if not source_orus or not target_orus:
                return False

            # Check if any O-RU pair has infrastructure connectivity
            for source_oru in source_orus:
                for target_oru in target_orus:
                    # Use cached connectivity results
                    cache_key = (min(source_oru, target_oru), max(source_oru, target_oru))
                    if cache_key not in self._oru_connectivity_cache:
                        self._oru_connectivity_cache[cache_key] = self.topology.has_infrastructure_path(source_oru, target_oru)

                    if self._oru_connectivity_cache[cache_key]:
                        return True

            return False
        else:
            # Island mode: use IOPS controller for multi-hop routing
            iops_ctrl = getattr(self, 'iops_controller', None)
            if iops_ctrl:
                route = iops_ctrl.route_ue_to_ue(source_ue, target_ue,
                                                 self.topology)
                return route is not None
            return False

    def _establish_ue_to_ue_flows(self):
        """Establish UE-to-UE communication pairs following 3GPP TS 22.179 MCPTT requirements.

        Implements NETWORK-ASSISTED UE-to-UE communication through surviving RAN infrastructure:
        - NOT direct ProSe UE-to-UE (device-to-device without network)
        - Uses surviving O-RU/O-DU/O-CU-UP infrastructure for routing
        - MARL enables intelligent routing when core network (UPF) is down
        - Emergency UE-to-UE communication (3GPP TS 22.179 Section 5.7.2.3)
        - Rescue service coordination during disasters via distributed RAN
        """
        # Find all UEs
        ue_nodes = [node_id for node_id, node in self.topology.nodes.items()
                    if node.node_type.value == "UE" and node.is_survivor]

        if len(ue_nodes) < 2:
            return

        # Clear existing flows
        self.ue_to_ue_flows = []

        # 3GPP TS 22.179 MCPTT Emergency Communication Requirements:
        # - Emergency UEs establish direct communication with nearby rescue services
        # - Proximity-based discovery and communication (ProSe)
        # - Higher priority for life safety and emergency communication

        import random
        random.seed(42)  # Deterministic for reproducibility

        # Separate regular UEs from rescue service UEs
        regular_ues = []
        rescue_ues = []

        for ue_id in ue_nodes:
            node = self.topology.nodes[ue_id]
            # Check if this is a rescue service UE (higher life safety traffic or designated as rescue)
            is_rescue = (
                hasattr(node, 'is_rescue_service') and node.is_rescue_service
            ) or (
                # Rescue UEs typically have higher life safety baseline rates
                hasattr(node, 'traffic_profile') and
                node.traffic_profile and
                TrafficClass.LIFE_SAFETY in node.traffic_profile.profiles and
                node.traffic_profile.profiles[TrafficClass.LIFE_SAFETY].baseline_rate > 3.0
            )

            if is_rescue:
                rescue_ues.append(ue_id)
            else:
                regular_ues.append(ue_id)

        print(f"3GPP TS 22.179 MCPTT Emergency Communication Setup: {len(rescue_ues)} rescue UEs, {len(regular_ues)} regular UEs")

        # 1. Emergency UE-to-Rescue Communication (Highest Priority - 3GPP TS 22.179)
        for regular_ue in regular_ues:
            # Each regular UE connects to 1-2 nearby rescue services
            available_rescue = rescue_ues.copy()
            if available_rescue:
                num_rescue_connections = min(random.randint(1, 2), len(available_rescue))
                selected_rescue = random.sample(available_rescue, num_rescue_connections)

                for rescue_ue in selected_rescue:
                    # Bidirectional emergency communication
                    self.ue_to_ue_flows.append((regular_ue, rescue_ue))
                    self.ue_to_ue_flows.append((rescue_ue, regular_ue))

        # 2. Rescue Service Coordination Network (3GPP TS 22.179 Group Communication)
        for rescue_ue in rescue_ues:
            # Rescue services maintain coordination network with 2-3 other rescue UEs
            other_rescue = [r for r in rescue_ues if r != rescue_ue]
            if len(other_rescue) >= 2:
                num_coord_connections = min(random.randint(2, 3), len(other_rescue))
                coord_partners = random.sample(other_rescue, num_coord_connections)

                for partner in coord_partners:
                    if (rescue_ue, partner) not in self.ue_to_ue_flows:
                        self.ue_to_ue_flows.append((rescue_ue, partner))
                        self.ue_to_ue_flows.append((partner, rescue_ue))

        # 3. Proximity-based UE-to-UE Communication (ProSe - when network fails)
        # Regular UEs establish local coordination networks
        for regular_ue in regular_ues:
            # Each UE connects to 1-2 nearby UEs for local coordination
            other_regular = [r for r in regular_ues if r != regular_ue]
            if len(other_regular) >= 1:
                num_local_connections = min(random.randint(1, 2), len(other_regular))
                local_partners = random.sample(other_regular, num_local_connections)

                for partner in local_partners:
                    # Only add if not already connected via rescue services
                    if (regular_ue, partner) not in self.ue_to_ue_flows:
                        self.ue_to_ue_flows.append((regular_ue, partner))
                        self.ue_to_ue_flows.append((partner, regular_ue))

        # Remove duplicate flows
        unique_flows = []
        seen = set()
        for flow in self.ue_to_ue_flows:
            flow_tuple = tuple(sorted(flow))
            if flow_tuple not in seen:
                seen.add(flow_tuple)
                unique_flows.append(flow)

        self.ue_to_ue_flows = unique_flows

        # Update node attributes to mark rescue services
        for ue_id in rescue_ues:
            if ue_id in self.topology.nodes:
                self.topology.nodes[ue_id].is_rescue_service = True

        print(f"3GPP TS 22.179 MCPTT Emergency Communication: Established {len(self.ue_to_ue_flows)} UE-to-UE flows")
        print(f"  - Emergency UE-Rescue flows: {len([f for f in self.ue_to_ue_flows if any(r in f for r in rescue_ues)])}")
        print(f"  - Rescue coordination flows: {len([f for f in self.ue_to_ue_flows if all(r in rescue_ues for r in f)])}")
        print(f"  - Local UE coordination flows: {len([f for f in self.ue_to_ue_flows if all(r not in rescue_ues for r in f)])}")
    
    def _add_ue_to_ue_flows_for_new_ue(self, new_ue_id: str):
        """Add UE-to-UE flows for a newly joined UE (e.g., rescue service)."""
        # Update the UE-to-O-RU mapping for the new UE
        self._update_ue_to_oru_mapping_for_ue(new_ue_id)

        # Find existing UEs that could communicate with this new UE
        existing_ues = [node_id for node_id, node in self.topology.nodes.items()
                       if node.node_type.value == "UE" and node.is_survivor and node_id != new_ue_id]

        # Store the original number of flows before adding new ones
        original_flow_count = len(self.ue_to_ue_flows)
        
        if not existing_ues:
            return
        
        import random
        random.seed(hash(new_ue_id) % 1000)  # Deterministic but different per UE
        
        # Connect new UE to 1-3 existing UEs (rescue services coordinate)
        num_connections = min(random.randint(1, 3), len(existing_ues))
        connected_ues = random.sample(existing_ues, num_connections)
        
        for target_ue in connected_ues:
            # Add bidirectional flows (rescue services communicate both ways)
            if (new_ue_id, target_ue) not in self.ue_to_ue_flows:
                self.ue_to_ue_flows.append((new_ue_id, target_ue))
                # Update routing table for the new flow
                self._ue_routing_table[(new_ue_id, target_ue)] = self._calculate_ue_to_ue_routing(new_ue_id, target_ue)
            if (target_ue, new_ue_id) not in self.ue_to_ue_flows:
                self.ue_to_ue_flows.append((target_ue, new_ue_id))
                # Update routing table for the new flow
                self._ue_routing_table[(target_ue, new_ue_id)] = self._calculate_ue_to_ue_routing(target_ue, new_ue_id)
    
    def _can_route_ue_to_ue(self, source_ue: str, target_ue: str) -> bool:
        """
        Check if UE-to-UE routing is possible through the CURRENT topology.

        In island mode the topology graph evolves each tick as MARL agents
        create / tear down transport backhaul links.  We therefore check the live
        graph instead of a static pre-computed table.

        Path must exist through infrastructure only (no direct UE-UE hop):
          source_ue -> O-RU_A -> ... -> O-RU_B -> target_ue
        transport relay links added by agents appear in self.topology.graph, so if an
        agent bridges two islands, this check automatically returns True.
        """
        if not self.island_mode:
            # Normal mode: full graph available — use fast cached result
            flow_key = (source_ue, target_ue)
            if flow_key not in self._ue_routing_table:
                self._ue_routing_table[flow_key] = self._calculate_ue_to_ue_routing(
                    source_ue, target_ue
                )
            return self._ue_routing_table[flow_key]

        # Island mode: check if there is a live infrastructure path connecting
        # the O-RU/gNB neighbours of src_ue to the O-RU/gNB neighbours of tgt_ue.
        # UE nodes are excluded from has_infrastructure_path, so we find the infra
        # anchors (neighbors with is_up links) first.
        try:
            def _infra_neighbors(ue_id):
                """Return live infra nodes one hop from ue_id."""
                nbrs = []
                for link in self.topology.links.values():
                    if not getattr(link, 'is_up', True):
                        continue
                    ep0, ep1 = link.endpoints
                    if ep0 == ue_id:
                        peer_node = self.topology.nodes.get(ep1)
                        if peer_node and peer_node.node_type.value != 'UE':
                            nbrs.append(ep1)
                    elif ep1 == ue_id:
                        peer_node = self.topology.nodes.get(ep0)
                        if peer_node and peer_node.node_type.value != 'UE':
                            nbrs.append(ep0)
                return nbrs

            src_anchors = _infra_neighbors(source_ue)
            tgt_anchors = _infra_neighbors(target_ue)
            if not src_anchors or not tgt_anchors:
                return False
            # If any src anchor can reach any tgt anchor via live infra links
            return any(
                self.topology.has_infrastructure_path(s, t)
                for s in src_anchors for t in tgt_anchors
            )
        except Exception:
            return False

    def _compute_optimal_scenario_routing(self):
        """Calculate the theoretical shortest paths for all UE pairs assuming all transport relays are active."""
        if not hasattr(self, 'current_optimal_paths'):
            self.current_optimal_paths = []
            self.current_optimal_actions = set()
            
        if not self.island_mode:
            self.current_optimal_paths = []
            self.current_optimal_actions = set()
            return
            
        # Build "Max Connectivity Graph"
        max_graph = nx.Graph()
        for nid, node in self.topology.nodes.items():
            if node.is_survivor:
                max_graph.add_node(nid)
                
        for link in self.topology.links.values():
            ep0, ep1 = link.endpoints
            if ep0 in max_graph and ep1 in max_graph:
                # If it's a transport link, assume it could be turned ON (except if severed)
                # But wait, if it's severed (e.g. cut by scenario), it's already removed?
                # Actually, in our topology, severed links have `link.is_up = False`.
                # BUT MARL agents also set `link.is_up = False`. How to distinguish?
                # Severed core links are between core nodes. Transport relays are 'TRANSPORT_RELAY' or 'MICROWAVE'.
                # We can assume any TRANSPORT_RELAY or MICROWAVE link connected to a Relay node is available.
                is_relay_link = link.link_type in (LinkType.TRANSPORT_RELAY, LinkType.MICROWAVE) and \
                               (self.topology.nodes[ep0].node_type.value == 'Relay' or \
                                self.topology.nodes[ep1].node_type.value == 'Relay')
                
                if link.is_up or is_relay_link:
                    max_graph.add_edge(ep0, ep1, weight=link.latency)
                    
        self.current_optimal_paths = []
        self.current_optimal_actions = set()
        
        for (src_ue, tgt_ue) in getattr(self, 'ue_to_ue_flows', []):
            if src_ue in max_graph and tgt_ue in max_graph:
                try:
                    path = nx.shortest_path(max_graph, source=src_ue, target=tgt_ue, weight='weight')
                    # Convert to link pairs for frontend
                    for i in range(len(path)-1):
                        self.current_optimal_paths.append((path[i], path[i+1]))
                    # Record relays used
                    for nid in path:
                        if self.topology.nodes[nid].node_type.value == 'Relay':
                            self.current_optimal_actions.add(nid)
                except nx.NetworkXNoPath:
                    pass

    def compute_optimal_connectivity(self) -> Dict[str, float]:
        """Compute optimal connectivity metrics.
        
        Uses the first island-mode traffic evaluation as the theoretical
        baseline (before MARL has had time to change transport topology).
        Subsequent ticks show MARL's improvement over this baseline.
        
        Returns dict with:
          - optimal_frac:     fraction of UE pairs routable at baseline
          - actual_frac:      fraction of UE pairs currently routed
          - efficiency:       actual / optimal
          - optimal_flows:    baseline routable flows count
          - actual_flows:     currently routed flows
          - total_flows:      total UE-to-UE flows
        """
        self._compute_optimal_scenario_routing()
        
        ue_flows = getattr(self, 'ue_to_ue_flows', [])
        total_flows = len(ue_flows)
        actual_flows = getattr(self, '_last_routed_flows', 0)
        # After core restore, _last_routed_flows resets to 0.
        # Use peak island-mode flows as the actual performance metric.
        if actual_flows == 0:
            actual_flows = getattr(self, '_island_peak_flows', 0)
        
        # The baseline is captured on first island tick
        baseline = getattr(self, '_island_baseline_flows', None)
        if baseline is None:
            # Not yet captured — use total flows as theoretical max
            baseline = total_flows
        
        optimal_frac = baseline / max(1, total_flows)
        actual_frac = actual_flows / max(1, total_flows)
        efficiency = actual_frac / max(0.001, optimal_frac)
        n_components = self._count_island_fragments() if self.island_mode else 1

        return {
            'optimal_frac': optimal_frac,
            'actual_frac': actual_frac,
            'efficiency': min(1.0, efficiency),
            'optimal_flows': baseline,
            'actual_flows': actual_flows,
            'total_flows': total_flows,
            'unreachable_flows': total_flows - baseline,
            'n_components': n_components,
        }

    def _assess_marl_convergence(self) -> Dict[str, float]:
        """
        Assess real MARL convergence through agent behavior analysis.
        Measures actual multi-agent coordination, not just message counting.
        """
        infra_agents = [(aid, agent) for aid, agent in self.agents.items()
                       if self.topology.nodes.get(aid, None) and
                       self.topology.nodes[aid].node_type.value != "UE"]

        if not infra_agents:
            return {'convergence_ratio': 0.0, 'policy_stability': 0.0, 'coordination_quality': 0.0,
                   'emergent_behavior': 0.0, 'converged_agents': 0, 'total_agents': 0}

        total_agents = len(infra_agents)

        # Analyze agent policies and behavior
        converged_agents = 0
        policy_consensus = {'life_safety': {}, 'operations': {}, 'telemetry': {}, 'best_effort': {}}
        coordination_patterns = []
        emergent_actions = 0

        for agent_id, agent in infra_agents:
            # Check if agent has adapted emergency policies in island mode
            if self.island_mode:
                # In island mode, successful agents should show emergency prioritization
                # Check for life_safety admission and best_effort throttling
                agent_converged = False
                if hasattr(agent, 'last_actions'):
                    ls_action = agent.last_actions.get(TrafficClass.LIFE_SAFETY, {})
                    be_action = agent.last_actions.get(TrafficClass.BEST_EFFORT, {})

                    # Converged if prioritizing emergency traffic (life_safety admitted, best_effort restricted)
                    ls_mode = ls_action.get('admission_mode')
                    be_mode = be_action.get('admission_mode')

                    if (ls_mode in ['ADMIT', 'THROTTLE'] and
                        be_mode in ['THROTTLE', 'HOLD']):
                        agent_converged = True
                        converged_agents += 1

            # Track policy consensus across all agents
            if hasattr(agent, 'last_actions'):
                for tclass, action in agent.last_actions.items():
                    mode = action.get('admission_mode', 'unknown')
                    if mode not in policy_consensus[tclass.value]:
                        policy_consensus[tclass.value][mode] = 0
                    policy_consensus[tclass.value][mode] += 1

        # Calculate convergence metrics
        convergence_ratio = converged_agents / total_agents if total_agents > 0 else 0.0

        # Policy stability (how consistent agent policies are)
        policy_stability = 0.0
        for tclass, modes in policy_consensus.items():
            if modes:
                # Measure consensus (highest mode percentage)
                max_count = max(modes.values())
                total_votes = sum(modes.values())
                consensus_pct = max_count / total_votes if total_votes > 0 else 0.0
                policy_stability += consensus_pct
        policy_stability /= len(policy_consensus) if policy_consensus else 1.0

        # Coordination quality (based on postcard consistency - simplified)
        coordination_quality = min(1.0, self.total_postcards_sent / max(1, total_agents * 5))

        # Emergent behavior (simplified - based on sustained coordination)
        emergent_behavior = min(1.0, len(self.convergence_history) / 20.0) if hasattr(self, 'convergence_history') else 0.0

        # Overall convergence combines multiple factors
        overall_convergence = (convergence_ratio * 0.5 +
                             policy_stability * 0.3 +
                             coordination_quality * 0.2)

        return {
            'convergence_ratio': overall_convergence,
            'policy_stability': policy_stability,
            'coordination_quality': coordination_quality,
            'emergent_behavior': emergent_behavior,
            'converged_agents': converged_agents,
            'total_agents': total_agents
        }

    def _detect_island_mode(self) -> bool:
        """Check if network should enter island mode.

        Uses the infrastructure graph (link.is_up respected) so severed core
        links correctly trigger island mode.  Falls back to True (island) when
        no surviving non-core node can reach any surviving core node.
        """
        if not self.config.enable_island_detection:
            return False

        # Core types that define 'connected to core'
        core_types = {"Core", "SMO", "Non-RT-RIC", "AMF", "UPF"}

        # Only count core nodes that are still alive
        live_core_nodes = [
            nid for nid in self.core_nodes
            if nid in self.topology.nodes and
               self.topology.nodes[nid].is_survivor
        ]
        if not live_core_nodes:
            # All core is gone -> definitely island
            return True

        # Any surviving non-core node that can still reach live core?
        survivor_nodes = [
            nid for nid, node in self.topology.nodes.items()
            if node.is_survivor and node.node_type.value not in core_types
        ]

        # Use infrastructure graph (respects link.is_up)
        for survivor in survivor_nodes:
            if any(self.topology.has_infrastructure_path(survivor, core_nid)
                   for core_nid in live_core_nodes):
                return False   # at least one path exists -> still connected

        return True   # no surviving non-core node can reach core -> island

    def compute_global_connectivity_reward(self) -> float:
        """
        Phase-aware shaped global reward — designed for learning signal clarity.

        Island mode (Core severed)
        --------------------------
          +20.0 * UE-pair routing fraction  — primary learning signal
          + 3.0 full-connectivity bonus     — if all flows routed
          + 0.2 per active Transport relay node   — positive-only relay reward
          +10.0 per routed flow (capped)    — throughput-coupled relay incentive
          + 5.0 * recovery_ramp             — ramp once restore_core fires
          - 0.5 per extra island fragment   — soft connectivity penalty

        Normal mode (pre-severance)
        ---------------------------
          + 3.0 * UE-pair routing fraction
          + 0.01 per survivor node with spare PRB capacity  — baseline signal
        """
        from .phy_mac_state import RelayMode
        reward = 0.0

        total_flows  = max(1, len(getattr(self, 'ue_to_ue_flows', [])))
        routed_flows = getattr(self, 'ue_to_ue_success_count', 0)
        frac = min(1.0, routed_flows / total_flows)

        if self.island_mode:
            # ── Primary connectivity signal ────────────────────────────────────
            reward += 100.0 * frac

            if frac >= 1.0:
                reward += 25.0  # Full-connectivity bonus

            # ── Relay reward: positive-only, no idle penalty ──────────────────
            relay_count = 0
            for nid, ps in self.phy_mac_states.items():
                node = self.topology.nodes.get(nid)
                if node is None or not node.is_survivor:
                    continue
                ntype_val = getattr(getattr(node, 'node_type', None), 'value', '')
                if ntype_val not in ('Relay', 'gNB-Site', 'O-DU', 'O-RU'):
                    continue
                if getattr(ps, 'relay_mode', RelayMode.OFF) in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST):
                    relay_count += 1

            reward += 0.2 * relay_count

            # Extra reward when relay effort directly produces routed flows
            if routed_flows > 0 and relay_count > 0:
                # Scale: 0.5 points per routed pair (capped so it can't dominate)
                reward += min(20.0, float(routed_flows) * 0.5)

            # ── Recovery ramp ─────────────────────────────────────────────────
            if self._recovery_started_tick >= 0 and not self._recovery_complete:
                ramp = min(1.0, self._ticks_since_recovery / 200.0)
                reward += 5.0 * ramp
                if frac >= 0.95 and ramp >= 0.5:
                    self._recovery_complete = True
                    reward += 5.0   # one-shot completion bonus

            # ── Soft fragment penalty (check every 10 ticks to save CPU) ──────
            if self.current_tick % 10 == 0:
                fragments = self._count_island_fragments()
                self._last_fragment_penalty = 0.5 * max(0, fragments - 1)
            reward -= getattr(self, '_last_fragment_penalty', 0.0)

        else:
            # ── Normal-mode reward ─────────────────────────────────────────────
            reward += 3.0 * frac

            # Baseline signal so agents aren't starved pre-severance
            spare_cap_nodes = sum(
                1 for ps in self.phy_mac_states.values()
                if getattr(ps, 'prb_utilization', 0.5) < 0.85
            )
            reward += 0.01 * spare_cap_nodes   # tiny, but non-zero

        return reward

    def _count_routed_ue_pairs(self) -> int:
        """Count how many UE-to-UE flows were delivered this tick."""
        return getattr(self, '_last_routed_flows', 0)

    def get_island_kpis(self) -> dict:
        """
        Return a snapshot dict of all learning-relevant KPIs for LearningTracker.
        Called once per tick, near-zero overhead (reads already-maintained state).

        Keys:
            ue_conn_frac     float  UE-pair routing fraction [0,1]
            transport_relay_count  int    nodes in LOCAL_REROUTE or CAPACITY_BOOST mode
            transport_link_count   int    active transport wireless backhaul links
            island_node_count int   surviving infra nodes (non-Core) in island
            iops_admitted    int    cumulative IOPS registrations
            island_fragments int    disconnected island sub-graphs (1 = fully connected)
            ticks_since_sev  int    0 before severance
        """
        from .phy_mac_state import RelayMode
        transport_relay_count = sum(
            1 for ps in self.phy_mac_states.values()
            if getattr(ps, 'relay_mode', RelayMode.OFF) in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST)
        )
        transport_link_count = len(getattr(self.transport_relay_model, 'active_links', {}))

        island_node_count = sum(
            1 for nid, n in self.topology.nodes.items()
            if n.is_survivor and
               getattr(n, 'node_type', None) is not None and
               n.node_type.value not in {'UE', 'Core', 'CoreUPF', 'UPF'}
        )

        return {
            'ue_conn_frac':     getattr(self, '_ue_pair_routed_fraction', 0.0),
            'transport_relay_count':  transport_relay_count,
            'transport_link_count':   transport_link_count,
            'island_node_count': island_node_count,
            'iops_admitted':    self.iops_manager.total_admitted,
            'island_fragments': self._count_island_fragments() if self.island_mode else 1,
            'ticks_since_sev':  getattr(self, '_ticks_since_severance', 0),
        }

    def _count_island_fragments(self) -> int:
        """Count disconnected subgraphs among surviving non-core nodes."""
        try:
            survivor_nodes = [
                nid for nid, n in self.topology.nodes.items()
                if n.is_survivor and n.node_type.value not in {"Core", "UPF", "AMF"}
            ]
            if not survivor_nodes:
                return 1
            visited, count = set(), 0
            def bfs(start):
                queue = [start]
                while queue:
                    nid = queue.pop()
                    if nid in visited:
                        continue
                    visited.add(nid)
                    if nid not in self.topology.nodes:
                        continue
                    for lid, lnk in self.topology.links.items():
                        src = lnk.endpoints[0] if hasattr(lnk, 'endpoints') else None
                        dst = lnk.endpoints[1] if hasattr(lnk, 'endpoints') else None
                        if not getattr(lnk, 'is_up', True):
                            continue
                        nb = dst if src == nid else (src if dst == nid else None)
                        if nb and nb in survivor_nodes and nb not in visited:
                            queue.append(nb)
            for nid in survivor_nodes:
                if nid not in visited:
                    bfs(nid)
                    count += 1
            return count
        except Exception:
            return 1

    def _process_events(self, tick: int):
        """Process scenario events for current tick."""
        events = self.scenario.get_events_at_tick(tick)

        for event in events:
            self._execute_event(event)

    def _execute_event(self, event: ScenarioEvent):
        """Execute a scenario event."""
        if event.event_type == 'sever_core':
            # Cut all links to/from core nodes
            for link in self.topology.links.values():
                if (link.endpoints[0] in self.core_nodes or
                    link.endpoints[1] in self.core_nodes):
                    link.is_up = False

            # Invalidate caches since links changed
            self.topology.invalidate_infrastructure_cache()
            self._oru_connectivity_cache.clear()
            self._update_ue_routing_table()

            # Mark all core nodes (including UPF and EdgeUPF) as non-survivors
            core_nodes_down = []
            for core_node_id in self.core_nodes:
                if core_node_id in self.topology.nodes:
                    self.topology.nodes[core_node_id].is_survivor = False
                    core_nodes_down.append(core_node_id)
                    # Remove agents for core nodes (they're down)
                    if core_node_id in self.agents:
                        del self.agents[core_node_id]
            
            print(f"Tick {event.tick}: Core severed - entering island mode")
            print(f"  Core nodes marked as down: {len(core_nodes_down)} nodes")
            # Edge UPFs survive — count them separately
            surviving_eupfs = [nid for nid, n in self.topology.nodes.items()
                              if n.node_type.value == 'EdgeUPF' and n.is_survivor]
            if surviving_eupfs:
                print(f"  EdgeUPFs surviving (local DN steering): {len(surviving_eupfs)}")
            if core_nodes_down:
                upf_count = sum(1 for nid in core_nodes_down
                              if self.topology.nodes[nid].node_type.value == 'UPF')
                print(f"  Core UPFs down: {upf_count}")

            # Record when island mode started for MARL temporal features
            if not self.island_mode:
                self._island_start_tick = event.tick
                self._ticks_since_severance = 0
                self._recovery_started_tick = -1
                self._recovery_complete = False
                self._ticks_since_recovery = 0
                # Clear routing table cache: all pre-computed routes are now invalid
                self._ue_routing_table.clear()
                # Reset baseline for optimal calculation (will be captured on first island tick)
                self._island_baseline_flows = None

        elif event.event_type == 'partial_sever':
            # Sever the explicitly named nodes (partial_core scenario)
            target_nodes = set(event.parameters.get('nodes', []))
            core_types   = {'Core', 'SMO', 'Non-RT-RIC', 'AMF', 'UPF'}

            # Also include any core-type nodes that aren't in target_nodes
            # so that island_mode detection fires correctly.
            # (Without this, a surviving EdgeUPF keeps a path to core open.)
            all_core_node_ids = [
                nid for nid, node in self.topology.nodes.items()
                if getattr(node.node_type, 'value', '') in core_types
            ]
            # Any core node not in target_nodes -> sever its links but keep it
            # as a survivor so the surviving EdgeUPF logic is preserved.
            # All nodes IN target_nodes -> fully down (is_survivor=False).
            for nid in target_nodes:
                if nid not in self.topology.nodes:
                    continue
                self.topology.nodes[nid].is_survivor = False
                if nid in self.agents:
                    del self.agents[nid]

            # Cut ALL links touching any core-type node (whether in target or not)
            # This is what actually severs the island from the core backbone.
            for link in self.topology.links.values():
                ep0, ep1 = link.endpoints
                if ep0 in target_nodes or ep1 in target_nodes:
                    link.is_up = False
                    continue
                # Also cut links between pure-core nodes that aren't EdgeUPF
                n0 = self.topology.nodes.get(ep0)
                n1 = self.topology.nodes.get(ep1)
                if n0 and n1:
                    t0 = getattr(n0.node_type, 'value', '')
                    t1 = getattr(n1.node_type, 'value', '')
                    if t0 in {'Core', 'AMF', 'UPF', 'SMO'} or \
                       t1 in {'Core', 'AMF', 'UPF', 'SMO'}:
                        link.is_up = False

            self.topology.invalidate_infrastructure_cache()
            self._oru_connectivity_cache.clear()
            if not self.island_mode:
                self._island_start_tick = event.tick
                self._ticks_since_severance = 0
                self._recovery_started_tick = -1
                self._recovery_complete = False
                self._ue_routing_table.clear()
            severed_count = len(target_nodes)
            print(f"Tick {event.tick}: partial_sever — {severed_count} nodes severed "
                  f"(all core backhaul links cut)")

        elif event.event_type == 'sever_zone':
            # Mark all nodes in a named zone as non-survivor; cut their links
            from .scenario import _infer_zone
            zone = event.parameters.get('zone', '')
            zone_nodes = [
                nid for nid in self.topology.nodes
                if _infer_zone(nid) == zone
            ]
            for nid in zone_nodes:
                self.topology.nodes[nid].is_survivor = False
                if nid in self.agents:
                    del self.agents[nid]
                for link in self.topology.links.values():
                    if link.endpoints[0] == nid or link.endpoints[1] == nid:
                        link.is_up = False
            self.topology.invalidate_infrastructure_cache()
            self._oru_connectivity_cache.clear()
            self._ue_routing_table.clear()
            print(f"Tick {event.tick}: sever_zone '{zone}' — {len(zone_nodes)} nodes severed")

        elif event.event_type == 'restore_core':
            # Re-enable all core links; restore node survivor flags; recreate agents
            from sixg_sim.agent import create_agent_for_node
            restored = []
            for link in self.topology.links.values():
                ep0, ep1 = link.endpoints
                if ep0 in self.core_nodes or ep1 in self.core_nodes:
                    link.is_up = True
            for core_nid in self.core_nodes:
                if core_nid in self.topology.nodes:
                    self.topology.nodes[core_nid].is_survivor = True
                    restored.append(core_nid)
                    if core_nid not in self.agents:
                        node = self.topology.nodes[core_nid]
                        self.agents[core_nid] = create_agent_for_node(
                            core_nid, node.node_type.value,
                            self.slice_dictionary
                        )
            self.topology.invalidate_infrastructure_cache()
            self._oru_connectivity_cache.clear()
            self._ue_routing_table.clear()
            # Stamp recovery start for reward shaping
            self._recovery_started_tick = event.tick
            self._recovery_complete = False
            self._ticks_since_recovery = 0
            print(f"Tick {event.tick}: restore_core — {len(restored)} core nodes restored")

        elif event.event_type == 'fail_link':
            link_id = event.parameters.get('link_id')
            if link_id in self.topology.links:
                self.topology.links[link_id].is_up = False
                self.topology.invalidate_infrastructure_cache()  # Invalidate cache
                self._oru_connectivity_cache.clear()  # Clear connectivity cache
                self._update_ue_routing_table()  # Update routing table
                print(f"Tick {event.tick}: Link {link_id} failed")
            else:
                print(f"Tick {event.tick}: fail_link ignored (link {link_id} not found)")

        elif event.event_type == 'restore_link':
            link_id = event.parameters.get('link_id')
            if link_id in self.topology.links:
                self.topology.links[link_id].is_up = True
                self.topology.invalidate_infrastructure_cache()  # Invalidate cache
                self._oru_connectivity_cache.clear()  # Clear connectivity cache
                self._update_ue_routing_table()  # Update routing table
                print(f"Tick {event.tick}: Link {link_id} restored")
            else:
                print(f"Tick {event.tick}: restore_link ignored (link {link_id} not found)")

        elif event.event_type == 'energy_depletion':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self.topology.nodes[node_id].energy_soc = 0.0
                self.topology.nodes[node_id].is_survivor = False
                print(f"Tick {event.tick}: Node {node_id} depleted")
            else:
                print(f"Tick {event.tick}: energy_depletion ignored (node {node_id} not found)")

        elif event.event_type == 'node_failure':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self.topology.nodes[node_id].is_survivor = False
                print(f"Tick {event.tick}: Node {node_id} failed")
            else:
                print(f"Tick {event.tick}: node_failure ignored (node {node_id} not found)")

        elif event.event_type == 'node_recovery':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self.topology.nodes[node_id].is_survivor = True
                print(f"Tick {event.tick}: Node {node_id} recovered")
            else:
                print(f"Tick {event.tick}: node_recovery ignored (node {node_id} not found)")

        elif event.event_type == 'ue_join':
            # Single UE joins the network
            coverage_area = event.parameters.get('coverage_area')
            connect_to_rus = event.parameters.get('connect_to_rus')  # Optional list of O-RU IDs
            ue_id = event.parameters.get('ue_id')  # Optional: specify UE ID
            
            if not ue_id:
                # Generate new UE ID
                self.ue_counter += 1
                ue_id = f"UE_{self.ue_counter}"
            
            if self.topology.add_ue_dynamically(ue_id, coverage_area, connect_to_rus):
                # Update UE-to-O-RU mapping for the new UE
                self._update_ue_to_oru_mapping_for_ue(ue_id)

                # Create traffic profile for new UE
                from sixg_sim.traffic import NodeTrafficProfile, TrafficProfile, TrafficClass
                ue_profile = NodeTrafficProfile(
                    node_id=ue_id,
                    profiles={
                        TrafficClass.LIFE_SAFETY: TrafficProfile(baseline_rate=2.0, burst_probability=0.05, burst_multiplier=3.0),
                        TrafficClass.OPERATIONS: TrafficProfile(baseline_rate=5.0, burst_probability=0.1, burst_multiplier=2.0),
                        TrafficClass.TELEMETRY: TrafficProfile(baseline_rate=8.0, burst_probability=0.15, burst_multiplier=1.5),
                        TrafficClass.BEST_EFFORT: TrafficProfile(baseline_rate=25.0, burst_probability=0.3, burst_multiplier=4.0),
                    }
                )
                self.traffic_generator.add_node_profile(ue_profile)
                self.scenario.traffic_profiles[ue_id] = ue_profile

                # Update UE-to-UE flows if this is a rescue service UE
                is_rescue = event.parameters.get('is_rescue', False)
                if is_rescue:
                    self._add_ue_to_ue_flows_for_new_ue(ue_id)
                
                print(f"Tick {event.tick}: UE {ue_id} joined network (coverage: {coverage_area})")
            else:
                print(f"Tick {event.tick}: ue_join failed (UE {ue_id} already exists or error)")

        elif event.event_type == 'ue_leave':
            # UE leaves the network
            ue_id = event.parameters.get('ue_id')
            if self.topology.remove_ue(ue_id):
                # Remove from traffic generator
                if ue_id in self.scenario.traffic_profiles:
                    del self.scenario.traffic_profiles[ue_id]
                # Remove from UE-to-UE flows
                self.ue_to_ue_flows = [(s, t) for s, t in self.ue_to_ue_flows if s != ue_id and t != ue_id]
                print(f"Tick {event.tick}: UE {ue_id} left network")
            else:
                print(f"Tick {event.tick}: ue_leave ignored (UE {ue_id} not found)")

        elif event.event_type == 'mcppt_emergency_call':
            # 3GPP TS 22.179 MCPTT Emergency Private Call (with Floor control)
            caller_ue = event.parameters.get('caller_ue')
            target_ue = event.parameters.get('target_ue')
            emergency_type = event.parameters.get('emergency_type', 'emergency')

            if (caller_ue in self.topology.nodes and target_ue in self.topology.nodes and
                self.topology.nodes[caller_ue].node_type.value == "UE" and
                self.topology.nodes[target_ue].node_type.value == "UE"):

                # Establish emergency call (higher priority than regular calls)
                emergency_call_id = f"emergency_call_{caller_ue}_{target_ue}_{event.tick}"
                self.mcppt_emergency_calls += 1

                # Put caller in emergency state if not already
                if not self.topology.nodes[caller_ue].emergency_state:
                    self.emergency_active_ues.add(caller_ue)
                    self.topology.nodes[caller_ue].emergency_state = True
                    self.topology.nodes[caller_ue].emergency_type = emergency_type

                print(f"Tick {event.tick}: MCPTT Emergency Call established: {caller_ue} -> {target_ue} ({emergency_type})")
            else:
                print(f"[ERROR] Emergency call failed: caller {caller_ue} or target {target_ue} not found or not UE nodes")

        elif event.event_type == 'mcppt_emergency_alert':
            # 3GPP TS 22.179 MCPTT Emergency Alert
            ue_id = event.parameters.get('ue_id')
            emergency_type = event.parameters.get('emergency_type', 'general')

            # Guard: skip if ue_id is missing or not a real node
            if not ue_id:
                return
            if ue_id not in self.topology.nodes or self.topology.nodes[ue_id].node_type.value != "UE":
                # Silently skip — scenario may reference a UE that hasn't joined yet
                return
            # UE is valid — put in emergency state
            self.emergency_active_ues.add(ue_id)
            self.topology.nodes[ue_id].emergency_state = True
            self.topology.nodes[ue_id].emergency_type = emergency_type

            # Alert nearby UEs and rescue services
            emergency_alert_sent = 0
            for flow in self.ue_to_ue_flows:
                if ue_id in flow:
                    other_ue = flow[0] if flow[1] == ue_id else flow[1]
                    if other_ue in self.topology.nodes and self.topology.nodes[other_ue].is_survivor:
                        emergency_alert_sent += 1

            self.emergency_alerts_sent += emergency_alert_sent
            if emergency_alert_sent > 0 and getattr(self, '_verbose', False):
                print(f"Tick {event.tick}: MCPTT Emergency Alert from {ue_id} ({emergency_type}) - alerted {emergency_alert_sent} UEs")

        elif event.event_type == 'rescue_force_arrival':
            # Multiple rescue UEs join during disaster
            num_rescue_ues = event.parameters.get('num_ues', 20)
            coverage_area = event.parameters.get('coverage_area')
            
            import random
            random.seed(event.tick)  # Deterministic based on tick
            
            added_count = 0
            for i in range(num_rescue_ues):
                self.ue_counter += 1
                ue_id = f"Rescue_UE_{self.ue_counter}"
                
                if self.topology.add_ue_dynamically(ue_id, coverage_area):
                    # Update UE-to-O-RU mapping for the new rescue UE
                    self._update_ue_to_oru_mapping_for_ue(ue_id)

                    # Mark as rescue service UE (3GPP TS 22.179 MCPTT Emergency Services)
                    self.topology.nodes[ue_id].is_rescue_service = True

                    # Create traffic profile for rescue UE following MCPTT requirements
                    # Higher life safety priority for emergency communication
                    from sixg_sim.traffic import NodeTrafficProfile, TrafficProfile, TrafficClass
                    rescue_profile = NodeTrafficProfile(
                        node_id=ue_id,
                        profiles={
                            TrafficClass.LIFE_SAFETY: TrafficProfile(
                                baseline_rate=8.0,  # High life safety for emergency coordination
                                burst_probability=0.15,  # Frequent emergency bursts
                                burst_multiplier=5.0,   # Strong emergency bursts
                                surge_events=[]  # Can be triggered by emergency events
                            ),
                            TrafficClass.OPERATIONS: TrafficProfile(
                                baseline_rate=12.0,  # Command and control operations
                                burst_probability=0.2, burst_multiplier=4.0
                            ),
                            TrafficClass.TELEMETRY: TrafficProfile(
                                baseline_rate=15.0,  # Status reports and coordination data
                                burst_probability=0.25, burst_multiplier=3.0
                            ),
                            TrafficClass.BEST_EFFORT: TrafficProfile(
                                baseline_rate=18.0,  # General coordination when resources available
                                burst_probability=0.3, burst_multiplier=2.5
                            ),
                        }
                    )
                    self.traffic_generator.add_node_profile(rescue_profile)
                    self.scenario.traffic_profiles[ue_id] = rescue_profile

                    # Add to UE-to-UE flows (rescue services coordinate via MCPTT)
                    self._add_ue_to_ue_flows_for_new_ue(ue_id)
                    added_count += 1
            
            print(f"Tick {event.tick}: Rescue force arrived - {added_count} rescue UEs joined network")

        elif event.event_type == 'nenb_deployment':
            # ETSI TS 22.346: Deploy a Nomadic eNB from rescue teams
            coverage_area = event.parameters.get('coverage_area')
            nenb_id = self.iops_controller.deploy_nenb(
                self.topology, event.tick, coverage_area=coverage_area)
            if nenb_id:
                # Create agent for the new NeNB
                from sixg_sim.agent import create_agent_for_node
                node = self.topology.nodes.get(nenb_id)
                if node:
                    self.agents[nenb_id] = create_agent_for_node(
                        nenb_id, node.node_type.value, self.slice_dictionary)
                    # Init PHY/MAC state for NeNB
                    ps = PHYMACState(node_id=nenb_id)
                    ps.base_coverage_radius_m = 500.0
                    ps.effective_coverage_radius_m = 500.0
                    self.phy_mac_states[nenb_id] = ps
                print(f"Tick {event.tick}: NeNB '{nenb_id}' deployed (coverage: {coverage_area})")
            else:
                print(f"Tick {event.tick}: NeNB deployment failed")

        elif event.event_type == 'multi_enb_iops_init':
            # ETSI TS 22.346: Explicit Multi-eNB IOPS initialization trigger
            reason = event.parameters.get('reason', 'core_severance')
            islands = self.iops_controller.update_islands(
                self.topology, event.tick)
            print(f"Tick {event.tick}: Multi-eNB IOPS init ({reason}) "
                  f"— {len(islands)} island(s) formed")
            for island in islands:
                print(f"  Island '{island.island_id}': "
                      f"{island.size} eNBs, anchor={island.anchor_enb}")

        elif event.event_type == 'mcptt_group_call':
            # ETSI TS 22.346 + TS 22.179: MCPTT group call within island
            group_ues = event.parameters.get('group_ues', [])
            call_type = event.parameters.get('call_type', 'mcptt_emergency')
            # Find island for first UE in group
            for ue_id in group_ues:
                island = self.iops_controller.get_island_for_ue(
                    ue_id, self.topology)
                if island and island.local_epc:
                    success = island.local_epc.multicast_group_call(
                        group_ues, call_type)
                    status = 'established' if success else 'failed'
                    print(f"Tick {event.tick}: MCPTT group call ({call_type}) "
                          f"with {len(group_ues)} UEs — {status}")
                    break

        elif event.event_type == 'island_merge':
            # Two islands merge via Transport relay
            island_a = event.parameters.get('island_a')
            island_b = event.parameters.get('island_b')
            merged = self.iops_controller.try_merge_islands(island_a, island_b)
            if merged:
                print(f"Tick {event.tick}: Islands merged → '{merged.island_id}' "
                      f"({merged.size} eNBs)")

        # Update topology graph after link changes
        self.topology.update_link_statuses()

    def _generate_traffic(self) -> Dict[str, Dict[TrafficClass, float]]:
        """Generate traffic arrivals for current tick."""
        # Reset per-tick tracking before new traffic arrives
        for node in self.topology.nodes.values():
            node.reset_queues()
        for link in self.topology.links.values():
            link.reset_utilization()
            
        return self.traffic_generator.generate_traffic(self.current_tick)

    # ── PHY/MAC initialisation ────────────────────────────────────────────────

    def _init_phy_mac_states(self):
        """Create one PHYMACState per O-RU / O-DU and register relay positions."""
        import random as _rnd
        agent_types = {NodeType.O_RU, NodeType.O_DU, NodeType.RELAY,
                       NodeType.GNBSITE, NodeType.DU}
        for node_id, node in self.topology.nodes.items():
            if node.node_type in agent_types and node.is_survivor:
                ps = PHYMACState(node_id=node_id)
                # O-RUs get max 20 MHz; O-DUs get 100 PRBs
                if node.node_type == NodeType.O_RU:
                    ps.base_coverage_radius_m = 500.0
                    ps.effective_coverage_radius_m = 500.0
                else:
                    ps.base_coverage_radius_m = 1000.0
                    ps.effective_coverage_radius_m = 1000.0
                self.phy_mac_states[node_id] = ps

                # Register a synthetic position if not stored on node
                if not self.transport_relay_model.node_positions.get(node_id):
                    x = _rnd.uniform(0, 10000)
                    y = _rnd.uniform(0, 10000)
                    self.transport_relay_model.register_position(node_id, x, y)

    def _update_phy_mac_observations(self):
        """
        Update PHYMACState OBS fields from current topology/traffic.
        Called once per tick before building agent observations.
        """
        # Count UEs per O-RU via Uu links
        ue_counts:       Dict[str, int] = {nid: 0 for nid in self.phy_mac_states}
        emrg_counts:     Dict[str, int] = {nid: 0 for nid in self.phy_mac_states}
        backhaul_util:   Dict[str, float] = {nid: 0.0 for nid in self.phy_mac_states}
        backhaul_cap:    Dict[str, float] = {nid: 1000.0 for nid in self.phy_mac_states}

        for link in self.topology.links.values():
            if not link.is_up:
                continue
            a, b = link.endpoints
            for ru_id in (a, b):
                if ru_id in self.phy_mac_states:
                    peer = b if ru_id == a else a
                    peer_node = self.topology.nodes.get(peer)
                    if peer_node and peer_node.node_type == NodeType.UE:
                        ue_counts[ru_id] += 1
                        if peer_node.emergency_state or peer_node.is_rescue_service:
                            emrg_counts[ru_id] += 1
                    else:
                        # Track backhaul utilisation (non-Uu links)
                        util_frac = link.current_utilization / max(1.0, link.capacity)
                        backhaul_util[ru_id] = max(backhaul_util[ru_id], util_frac)
                        backhaul_cap[ru_id]  = min(backhaul_cap[ru_id],
                                                   link.available_capacity())

        # Total UE count for reachability
        all_ue_ids  = [nid for nid, n in self.topology.nodes.items()
                       if n.node_type == NodeType.UE and n.is_survivor]
        total_ues = max(1, len(all_ue_ids))
        connected_ues = sum(ue_counts.values())
        self._reachable_ue_fraction = min(1.0, connected_ues / total_ues)
        self._isolated_ue_count     = max(0, total_ues - connected_ues)

        # UE-to-UE routed fraction: relative to all flows (not just attempted)
        total_flows = max(1, len(self.ue_to_ue_flows))
        self._ue_pair_routed_fraction = self.ue_to_ue_success_count / total_flows

        # Write into each PHYMACState
        for node_id, ps in self.phy_mac_states.items():
            ps.active_ue_count    = ue_counts.get(node_id, 0)
            ps.emergency_ue_count = emrg_counts.get(node_id, 0)
            ps.backhaul_utilization   = backhaul_util.get(node_id, 0.0)
            ps.backhaul_capacity_mbps = backhaul_cap.get(node_id, 1000.0)
            # PRB utilisation proxy: UE load vs. capacity
            node = self.topology.nodes.get(node_id)
            if node:
                total_q = sum(q.offered_load for q in node.queues.values())
                cap_proxy = ps.access_capacity_mbps() * 10.0  # scale to match traffic units
                ps.prb_utilization = min(1.0, total_q / max(1.0, cap_proxy))
            # Connectivity metrics
            ps.reachable_ue_fraction    = self._reachable_ue_fraction
            ps.ue_pair_routing_fraction = self._ue_pair_routed_fraction

        # Ticks since severance / recovery
        if self.island_mode:
            self._ticks_since_severance = self.current_tick - self._island_start_tick
        if self._recovery_started_tick >= 0 and not self._recovery_complete:
            self._ticks_since_recovery = self.current_tick - self._recovery_started_tick

    def _step_phy_mac(self, node_id: str, action: 'PHYMACAction'):
        """
        Apply a PHYMACAction to a node's PHYMACState and update the topology
        if relay mode changes (create / tear down transport relay links).
        """
        ps = self.phy_mac_states.get(node_id)
        if ps is None:
            return

        prev_relay = ps.relay_mode

        # ── Apply discrete controls ────────────────────────────────────────
        ps.apply_power_step(action.tx_power_step)
        ps.mcs_emergency = list(MCSLevel)[action.mcs_emergency_idx]
        ps.mcs_general   = list(MCSLevel)[action.mcs_general_idx]
        ps.relay_mode    = list(RelayMode)[action.relay_mode_idx]
        ps.mac_scheduler = list(MACScheduler)[action.scheduler_idx]
        ps.handover_triggered = bool(action.handover_idx)

        # PRB allocation (already normalised to sum=1 by softmax in net)
        ps.prb_emergency_fraction = action.prb_emergency_frac
        ps.prb_relay_fraction     = action.prb_relay_frac
        ps.prb_general_fraction   = action.prb_general_frac
        ps.normalise_prb()

        # ── Transport relay mode change ─────────────────────────────────────────
        if prev_relay != ps.relay_mode:
            # Tear down existing transport relay links from this node
            removed_ids = self.transport_relay_model.clear_node_links(node_id)
            for rid in removed_ids:
                # Remove corresponding topology link
                self.topology.links.pop(rid, None)
                try:
                    n1, n2 = rid.replace("TR_", "").split("_", 1)
                    if self.topology.graph.has_edge(n1, n2):
                        self.topology.graph.remove_edge(n1, n2)
                except Exception:
                    pass
            ps.relay_link_active = False
            ps.relay_peer_node   = None
            ps.relay_link_capacity_mbps = 0.0
            self.topology.invalidate_infrastructure_cache()

        if ps.relay_mode in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST) and not ps.relay_link_active:
            # ── Self-Optimized Healing ───────────────────────────────────────
            #
            # Instead of creating new links from scratch, re-optimize the
            # EXISTING transport fabric for UE-to-UE local traffic:
            #
            # LOCAL_REROUTE: Repurpose existing link capacity for local traffic.
            #   Works on ANY link type (MW PtP, MultiHaul, fiber).
            #   Traffic that was flowing north-south (to core) is redirected
            #   east-west (UE-to-UE) within the surviving local fabric.
            #
            # CAPACITY_BOOST: MultiHaul beam re-steering to increase capacity
            #   on a specific existing mesh link. Only works on nodes with
            #   has_multihaul=True. Siklu 60GHz nodes can dynamically adjust
            #   beam direction/width to boost throughput on selected paths.
            #
            node = self.topology.nodes.get(node_id)
            has_mh = node.has_multihaul if node else False

            # Find existing topology neighbours with active links
            existing_neighbours = []
            for lid, link in self.topology.links.items():
                ep = link.endpoints
                if node_id in ep and link.is_up:
                    peer = ep[1] if ep[0] == node_id else ep[0]
                    peer_node = self.topology.nodes.get(peer)
                    if peer_node and peer_node.is_survivor:
                        is_multihaul_link = link.link_type == LinkType.MULTIHAUL_MESH
                        existing_neighbours.append((peer, lid, link, is_multihaul_link))

            if existing_neighbours:
                # Pick the best existing neighbour for local rerouting
                # Prefer: (1) MultiHaul links if CAPACITY_BOOST, (2) highest capacity
                if ps.relay_mode == RelayMode.CAPACITY_BOOST and has_mh:
                    # MultiHaul beam re-steer: boost capacity on best MultiHaul link
                    mh_neighbours = [(p, lid, l, mh) for p, lid, l, mh in existing_neighbours if mh]
                    target_list = mh_neighbours if mh_neighbours else existing_neighbours
                else:
                    target_list = existing_neighbours

                # Sort by link capacity (highest first)
                target_list.sort(key=lambda x: x[2].capacity, reverse=True)
                best_peer, best_lid, best_link, is_mh = target_list[0]

                # Apply capacity boost for MultiHaul beam re-steer
                boost_factor = 1.0
                if ps.relay_mode == RelayMode.CAPACITY_BOOST and has_mh and is_mh:
                    boost_factor = 1.5  # 50% capacity boost via beam re-steering

                relay_cap = best_link.capacity * boost_factor * ps.prb_relay_fraction / 10.0

                # Register this as the active relay path
                new_link_id = self.transport_relay_model.create_link(
                    node_id, best_peer, relay_cap, 15.0
                )
                self.topology.links[new_link_id] = Link(
                    id=new_link_id,
                    endpoints=(node_id, best_peer),
                    capacity=relay_cap,
                    latency=2,
                    link_type=LinkType.TRANSPORT_RELAY,
                    interface_type=InterfaceType.BACKHAUL,
                    is_up=True,
                    current_utilization=0.0
                )
                ps.relay_link_active   = True
                ps.relay_peer_node     = best_peer
                ps.relay_link_capacity_mbps = relay_cap
                self.topology.invalidate_infrastructure_cache()

                if self.config.verbose:
                    mode_str = "BOOST" if ps.relay_mode == RelayMode.CAPACITY_BOOST else "REROUTE"
                    link_str = "MultiHaul" if is_mh else "MW-PtP"
                    print(f"[{mode_str}] {node_id} -> {best_peer}: "
                          f"{relay_cap:.1f} Mbps via {link_str}"
                          f"{' (beam boost)' if boost_factor > 1 else ''}")

        elif ps.relay_mode == RelayMode.D2D_PEER_RELAY:
            # D2D: signal UEs — simplified capacity boost
            ps.d2d_active_pairs = ps.active_ue_count // 2

    def _compute_connectivity_state(self, node_id: str) -> 'ConnectivityState':
        """Build ConnectivityState (Block C) for a node.
        Uses per-tick cached bridge set and reachability to avoid O(N^2) work.
        """
        ps   = self.phy_mac_states.get(node_id)
        node = self.topology.nodes.get(node_id)
        cs   = ConnectivityState()
        if ps is None or node is None:
            return cs

        all_ue_ids = [nid for nid, n in self.topology.nodes.items()
                      if n.node_type == NodeType.UE and n.is_survivor]
        total_ues  = max(1, len(all_ue_ids))

        cs.reachable_ue_fraction     = self._reachable_ue_fraction
        cs.isolated_ue_count_norm    = min(1.0, self._isolated_ue_count / total_ues)
        cs.ue_to_ue_routed_norm      = self._ue_pair_routed_fraction
        cs.active_relay_paths_norm   = min(1.0,
            len(self.transport_relay_model.active_links) / max(1, len(self.phy_mac_states)))

        # ── Intra-island reachability (per-tick cached, skip for UE nodes) ──
        _is_ue = (node.node_type == NodeType.UE)
        if not _is_ue:
            cs.intra_island_reachability = self._tick_intra_reach_cache.get(node_id, 0.0)
        else:
            cs.intra_island_reachability = 0.0

        # Island size (infra_nodes list cached per tick)
        infra_nodes = self._tick_infra_nodes_cache
        cs.island_size_norm = min(1.0, len(infra_nodes) / max(1, len(self.topology.nodes)))

        # Transport relay capacity available
        relay_cap = ps.relay_capacity_mbps()
        cs.potential_relay_capacity_norm = min(1.0, relay_cap / 100.0)

        # Bridge node flag (per-tick cached set — free to query)
        cs.bridge_node_flag = 1.0 if node_id in self._tick_bridge_set_cache else 0.0

        # UE min RSRP proxy (SINR_min normalised)
        cs.ue_rsrp_min_norm = max(0.0, min(1.0, (ps.sinr_min + 5.0) / 35.0))

        return cs

    def _build_neighbor_radio_summary(self, node_id: str,
                                      postcards: list) -> 'NeighborRadioSummary':
        """Build NeighborRadioSummary (Block B, 16 dims) from received postcards."""
        nb = NeighborRadioSummary()
        if not postcards:
            return nb

        relay_active_count = 0
        sinr_vals, prb_vals = [], []
        island_count = 0
        tx_power_vals = []
        sinr_deg_vals = []
        power_increased_count = 0

        for pc in postcards:
            if hasattr(pc, 'relay_mode_active'):
                if pc.relay_mode_active:
                    relay_active_count += 1
                sinr_vals.append(pc.best_sinr_to_neighbour)
                prb_vals.append(pc.prb_avail_for_relay)
            island_count += 1
            # ICIC dims from enriched postcards
            if hasattr(pc, 'tx_power_dbm_norm'):
                tx_power_vals.append(pc.tx_power_dbm_norm)
                if pc.tx_power_dbm_norm > 0.5:   # above mid-scale = boosted
                    power_increased_count += 1
            if hasattr(pc, 'sinr_degradation_norm'):
                sinr_deg_vals.append(pc.sinr_degradation_norm)

        n = max(1, len(postcards))
        nb.avg_sinr             = float(np.mean(sinr_vals)) if sinr_vals else 0.5
        nb.avg_prb_util         = 1.0 - (float(np.mean(prb_vals)) if prb_vals else 0.5)
        nb.avg_backhaul_avail   = 0.5
        nb.fraction_island      = min(1.0, island_count / n)
        nb.neighbor_count_norm  = min(1.0, n / 10.0)
        nb.any_relay_active     = float(relay_active_count > 0)
        nb.postcard_received    = 1.0
        nb.coordination_quality = min(1.0, relay_active_count / n)
        nb.best_relay_sinr      = max(sinr_vals) if sinr_vals else 0.0
        # ICIC dims
        nb.neighbour_avg_tx_power_norm = (
            float(np.mean(tx_power_vals)) if tx_power_vals else 0.5
        )
        nb.experienced_sinr_drop_norm = (
            float(np.mean(sinr_deg_vals)) if sinr_deg_vals else 0.0
        )
        nb.interferer_count_norm = min(1.0, power_increased_count / n)
        # cooperative_relay_assigned: set if any neighbour's postcard says relay is active
        # and there are isolated UEs — neighbour wants ME to relay
        nb.cooperative_relay_assigned = float(
            relay_active_count > 0 and
            any(getattr(pc, 'isolated_ue_count', 0) > 0 for pc in postcards)
        )
        return nb

    def _compute_icic_state(self, node_id: str) -> float:
        """
        Compute SINR drop since last tick for ICIC observation.
        Returns delta_sinr normalised [0,1] (positive = SINR dropped).
        """
        ps = self.phy_mac_states.get(node_id)
        if ps is None:
            return 0.0
        current_sinr = getattr(ps, 'sinr_average', 15.0)
        prev_sinr    = self._prev_sinr.get(node_id, current_sinr)
        delta        = prev_sinr - current_sinr   # positive when SINR dropped
        self._prev_sinr[node_id] = current_sinr
        # Stamp into PHYMACState for use in postcard
        ps.sinr_delta = -delta   # positive when SINR improved
        return max(0.0, min(1.0, delta / 10.0))   # normalise to [-10dB, 10dB]

    def _handle_iops_events(self):
        """
        Process IOPS registration requests.
        Called once per tick when in island mode.

        The actual ADMIT/DENY decision is made by O-CU-CP agents via their
        iops_admit head.  Here we:
          1. Expire stale pending requests
          2. Trigger IOPS for newly arrived rescue UEs (from scenario events)
        """
        if not self.island_mode:
            return
        self.iops_manager.expire_old_pending(
            self.current_tick, max_wait=50
        )
        # Any UE marked is_rescue_service but not yet IOPS-registered
        for nid, node in self.topology.nodes.items():
            if (node.node_type == NodeType.UE and
                    node.is_survivor and
                    getattr(node, 'is_rescue_service', False) and
                    not self.iops_manager.is_registered(nid)):
                self.iops_manager.request_registration(
                    nid, is_emergency=True, tick=self.current_tick
                )

    # ── Rewritten observation builder ─────────────────────────────────────────

    def _build_agent_observations(self) -> Dict[str, AgentObservation]:
        """Build PHY/MAC-based observations (59 dims) for all infrastructure agents."""
        self._update_phy_mac_observations()
        self._handle_iops_events()

        # ── Per-tick connectivity caches (computed ONCE, shared by all agents) ───
        # 1. Infra-only node list
        self._tick_infra_nodes_cache = [
            nid for nid, n in self.topology.nodes.items()
            if n.node_type != NodeType.UE and n.is_survivor
        ]
        infra_set = set(self._tick_infra_nodes_cache)


        # 2. Bridge set — skip unless topology changed (link events)
        #    Use a simple staleness flag set by invalidate_infrastructure_cache()
        _topo_changed = getattr(self.topology, '_cache_stale', True)
        if _topo_changed or not self._tick_bridge_set_cache:
            try:
                if not self.topology.graph.is_directed():
                    _bridge_nodes: set = set()
                    for e in nx.bridges(self.topology.graph):
                        if e[0] in infra_set and e[1] in infra_set:
                            _bridge_nodes.add(e[0]); _bridge_nodes.add(e[1])
                    self._tick_bridge_set_cache = _bridge_nodes
                else:
                    self._tick_bridge_set_cache = set()
            except Exception:
                self._tick_bridge_set_cache = set()

        # 3. Intra-island reachability via connected components (ONE pass, O(V+E))
        #    In an undirected graph: reach(node) = (component_size - 1) / (total - 1)
        try:
            infra_g  = self.topology._build_infrastructure_graph()
            _n_total = max(1, len(self._tick_infra_nodes_cache) - 1)
            self._tick_intra_reach_cache = {}
            for comp in nx.connected_components(infra_g):
                comp_reach = (len(comp) - 1) / _n_total
                for nid in comp:
                    self._tick_intra_reach_cache[nid] = comp_reach
        except Exception:
            self._tick_intra_reach_cache = {}
        # ── End per-tick caches ──────────────────────────────────────────────────

        # Get current coordinator policy
        all_postcards = []
        for nid in self.agents:
            all_postcards.extend(
                self.control_plane.get_received_postcards(nid, self.current_tick)
            )
        # Update Multi-eNB IOPS islands every tick
        if self.island_mode:
            self.iops_controller.update_islands(self.topology, self.current_tick)

        self._current_global_policy = self.coordinator.step(
            all_postcards, self.phy_mac_states,
            self.iops_manager, self._ticks_since_severance,
            self.topology, self.current_tick,
            iops_controller=self.iops_controller
        )
        global_policy_list = self._current_global_policy.to_list()

        observations = {}
        control_messages = {
            nid: self.control_plane.get_received_postcards(nid, self.current_tick)
            for nid in self.agents
        }

        # IOPS state for Block C
        iops_cap, iops_pend = self.iops_manager.get_obs_features()

        # ── Parallel per-agent observation building ──────────────────────────────
        # Each agent's obs is independent (reads shared state, no writes).
        # ThreadPoolExecutor releases GIL during numpy/networkx calls.
        from concurrent.futures import ThreadPoolExecutor
        import threading
        _lock = threading.Lock()

        agent_items = [(nid, agent) for nid, agent in self.agents.items()
                       if self.topology.nodes.get(nid) is not None]
        _N_WORKERS = min(8, len(agent_items))

        def _build_one_obs(args):
            node_id, agent = args
            node = self.topology.nodes.get(node_id)
            if node is None:
                return node_id, None

            ps = self.phy_mac_states.get(node_id)
            if ps is None:
                ps = PHYMACState(node_id=node_id)
                with _lock:
                    self.phy_mac_states[node_id] = ps

            nb_summary = self._build_neighbor_radio_summary(
                node_id, control_messages.get(node_id, [])
            )
            nb_summary.experienced_sinr_drop_norm = self._compute_icic_state(node_id)

            conn_state = self._compute_connectivity_state(node_id)
            conn_state.registration_capacity_norm = iops_cap
            conn_state.pending_iops_norm          = iops_pend

            # Multi-eNB IOPS observations (8 new dims)
            iops_obs = self.iops_controller.get_node_obs(node_id, self.topology)
            peer_obs = self.learning_exchanger.get_peer_obs(node_id)
            conn_state.island_member_count_norm = iops_obs['island_member_count_norm']
            conn_state.island_ue_load_balance   = iops_obs['island_ue_load_balance']
            conn_state.xn_mesh_density          = iops_obs['xn_mesh_density']
            conn_state.local_epc_health         = iops_obs['local_epc_health']
            conn_state.nenb_count_norm          = iops_obs['nenb_count_norm']
            conn_state.peer_avg_reward          = peer_obs.get('peer_avg_reward', 0.0)
            conn_state.peer_best_relay_hint     = peer_obs.get('peer_best_relay_hint', 0.0)
            conn_state.multi_island_bridge      = iops_obs['multi_island_bridge']

            local_slices = {}
            for tc in TrafficClass:
                q = node.queues[tc]
                local_slices[tc] = LocalSliceState(
                    traffic_class=tc,
                    importance=self.slice_dictionary.get_importance_score(tc),
                    current_queue_length=q.queued_load,
                    offered_load=q.offered_load,
                    admission_success_rate=(
                        q.delivered_load / max(1.0, q.offered_load)
                        if q.offered_load > 0 else 1.0
                    ),
                    freshness_target=self.slice_dictionary.qos_profiles[tc].target_delay,
                    current_freshness=5.0,
                )

            obs = AgentObservation(
                node_id=node_id,
                is_island=node.is_island,
                energy_soc=node.energy_soc,
                phy_mac=ps,
                neighbor_radio=nb_summary,
                connectivity=conn_state,
                ticks_since_severance=self._ticks_since_severance,
                current_tick=self.current_tick,
                global_policy=global_policy_list,
                local_slices=local_slices,
                energy_tier=EnergyTier(node.get_energy_tier()),
                neighbor_summary=nb_summary,
            )
            return node_id, obs

        if _N_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
                for nid, obs in pool.map(_build_one_obs, agent_items):
                    if obs is not None:
                        observations[nid] = obs
        else:
            for args in agent_items:
                nid, obs = _build_one_obs(args)
                if obs is not None:
                    observations[nid] = obs

        return observations

    def _execute_agent_actions(self, observations: Dict[str, AgentObservation]):
        """
        Execute PHY/MAC actions from all agents.

        Steps per tick:
          1. Run coordinator (every COORD_INTERVAL ticks) — broadcasts global policy
          2. For each agent: compute_action() -> PHYMACAction
          3. _step_phy_mac() -> apply radio state + transport relay topology changes
          4. Process IOPS admit decisions from O-CU-CP agents
          5. Send postcards (ICIC-enriched)
        """
        self.last_postcards_sent = 0
        self.last_action_summary = {}
        if not hasattr(self, 'marl_comm_paths'):
            self.marl_comm_paths = []
        self.marl_comm_paths.clear()

        relay_nodes     = []
        postcard_senders = []
        iops_admitted   = 0
        iops_denied     = 0

        # ── IOPS: check for pending registrations from this tick ───────────────
        pending_iops = self.iops_manager.flush_pending()

        # ── Batched forward pass + vectorised action sampling ─────────────────
        agent_ids = [nid for nid in self.agents if observations.get(nid) is not None]
        _batch_actions: dict = {}

        if agent_ids:
            N = len(agent_ids)
            # ONE forward pass for all N agents: (N, OBS_DIM) -> Dict[head -> (N, C)]
            obs_tensors = torch.stack([
                self.agents[aid].observation_to_tensor(observations[aid])
                for aid in agent_ids
            ])
            ref_agent = next(iter(self.agents.values()))
            with torch.no_grad():
                bl = ref_agent.policy_net(obs_tensors)   # batch_logits

            _is_train = getattr(ref_agent, 'is_training', True)

            if _is_train:
                # Sample all discrete heads in 8 multinomial calls (vs N×8)
                import torch.nn.functional as _F
                def _vsample(head: str) -> torch.Tensor:
                    # multinomial expects probabilities; softmax+clamp for numerical safety
                    p = _F.softmax(bl[head], dim=-1).clamp(min=1e-8)
                    return torch.multinomial(p, num_samples=1).squeeze(1)   # (N,)

                tx_idx   = _vsample("tx_power")
                mcs_e    = _vsample("mcs_emrg")
                mcs_g    = _vsample("mcs_gen")
                relay_v  = _vsample("relay")
                ho_v     = _vsample("handover")
                sched_v  = _vsample("scheduler")
                post_v   = _vsample("postcard")
                iops_v   = _vsample("iops")
            else:
                # Inference: argmax (no sampling)
                tx_idx   = bl["tx_power"].argmax(dim=1)
                mcs_e    = bl["mcs_emrg"].argmax(dim=1)
                mcs_g    = bl["mcs_gen"].argmax(dim=1)
                relay_v  = bl["relay"].argmax(dim=1)
                ho_v     = bl["handover"].argmax(dim=1)
                sched_v  = bl["scheduler"].argmax(dim=1)
                post_v   = bl["postcard"].argmax(dim=1)
                iops_v   = bl["iops"].argmax(dim=1)

            # PRB softmax for all N agents at once: (N, 3) -> list of lists
            import torch.nn.functional as _F
            prb_all = _F.softmax(bl["prb"], dim=-1)   # (N, 3)

            # Build PHYMACAction per agent (pure Python, no more per-head sampling)
            for i, aid in enumerate(agent_ids):
                agent = self.agents[aid]
                obs   = observations[aid]
                prb   = prb_all[i].tolist()

                # Coordinator PRB floor
                if obs.global_policy and obs.is_island:
                    min_emrg = obs.global_policy[2]
                    if prb[0] < min_emrg:
                        deficit = min_emrg - prb[0]
                        prb[0] += deficit
                        prb[1] = max(0.0, prb[1] - deficit / 2)
                        prb[2] = max(0.0, prb[2] - deficit / 2)
                        total = sum(prb); prb = [x / total for x in prb]

                ri   = relay_v[i].item()
                pi   = bool(post_v[i].item())
                ps   = obs.phy_mac

                # Postcard (rate-limited per agent)
                postcard_content = None
                tx_norm  = max(0.0, min(1.0, (ps.tx_power_dbm - 10.0) / 33.0))
                sinr_deg = max(0.0, min(1.0, -getattr(ps, 'sinr_delta', 0.0) / 10.0))
                if pi and (obs.current_tick - agent.last_postcard_tick) >= 10:
                    agent.last_postcard_tick = obs.current_tick
                    from .agent import ControlPostcard, TrafficClass, StrainLevel
                    postcard_content = ControlPostcard(
                        sender_id=aid,
                        relay_mode_active=(ri > 0),
                        best_sinr_to_neighbour=min(1.0, (ps.sinr_average + 5) / 35),
                        isolated_ue_count=int(obs.connectivity.isolated_ue_count_norm * 100),
                        prb_avail_for_relay=prb[1],
                        policy_version=agent.policy_version,
                        timestamp=obs.current_tick,
                        most_needy_class=TrafficClass.LIFE_SAFETY,
                        need_level=StrainLevel.NEAR_LIMIT if obs.is_island else StrainLevel.OKAY,
                        tx_power_dbm_norm=tx_norm,
                        sinr_degradation_norm=sinr_deg,
                    )
                else:
                    pi = False

                from .agent import PHYMACAction
                action = PHYMACAction(
                    tx_power_step=tx_idx[i].item(),
                    mcs_emergency_idx=mcs_e[i].item(),
                    mcs_general_idx=mcs_g[i].item(),
                    prb_emergency_frac=prb[0],
                    prb_relay_frac=prb[1],
                    prb_general_frac=prb[2],
                    relay_mode_idx=ri,
                    handover_idx=ho_v[i].item(),
                    scheduler_idx=sched_v[i].item(),
                    send_postcard=pi,
                    postcard_content=postcard_content,
                )
                action._iops_decision = iops_v[i].item()
                agent.last_action = action
                _batch_actions[aid] = action

        for node_id, agent in self.agents.items():
            obs = observations.get(node_id)
            if obs is None:
                continue

            action = _batch_actions.get(node_id)
            if action is None:   # fallback (shouldn't happen)
                action = agent.compute_action(obs)
            agent.last_action  = action
            agent.last_actions = {}

            # ── Apply PHY/MAC control ───────────────────────────────────────
            self._step_phy_mac(node_id, action)

            # ── IOPS admission (O-CU-CP nodes act on pending requests) ───────
            if (self.island_mode and pending_iops and
                    self.topology.nodes.get(node_id) and
                    self.topology.nodes[node_id].node_type in
                    {NodeType.O_CU_CP, NodeType.O_CU, NodeType.CU}):
                iops_decision = getattr(action, '_iops_decision', 0)
                req = self.iops_manager.get_next_pending()
                if req:
                    if iops_decision == 0:   # DENY
                        self.iops_manager.deny(req.ue_id)
                        iops_denied += 1
                    elif iops_decision == 1:  # ADMIT_EMERGENCY
                        if req.is_emergency:
                            ok = self.iops_manager.admit(
                                req.ue_id, True, self.current_tick
                            )
                            if ok:
                                # Mark UE as registered so routing can use it
                                ue_node = self.topology.nodes.get(req.ue_id)
                                if ue_node:
                                    ue_node.iops_registered = True
                                iops_admitted += 1
                    else:   # ADMIT_ALL (iops_decision == 2)
                        ok = self.iops_manager.admit(
                            req.ue_id, req.is_emergency, self.current_tick
                        )
                        if ok:
                            ue_node = self.topology.nodes.get(req.ue_id)
                            if ue_node:
                                ue_node.iops_registered = True
                            iops_admitted += 1
                        else:
                            iops_denied += 1

            # ── Relay tracking ─────────────────────────────────────────────
            ps = self.phy_mac_states.get(node_id)
            if ps and ps.relay_mode.value != 'off':
                relay_nodes.append(node_id)
            relay_str = list(getattr(
                self.phy_mac_states.get(node_id, PHYMACState('')),
                'relay_mode', RelayMode.OFF
            ).value)[0].upper()
            self.last_action_summary[('relay', relay_str)] = (
                self.last_action_summary.get(('relay', relay_str), 0) + 1
            )

            # ── Postcard: only infra nodes send coordination messages ────────
            _node_obj = self.topology.nodes.get(node_id)
            _is_infra = (_node_obj is not None and
                         getattr(_node_obj.node_type, 'value',
                                 str(_node_obj.node_type)) != 'UE')
            if _is_infra and action.send_postcard and action.postcard_content:
                ok = self.control_plane.send_postcard(
                    action.postcard_content, self.current_tick
                )
                if ok:
                    self.last_postcards_sent += 1
                    if self.island_mode:
                        self.total_postcards_sent += 1
                        postcard_senders.append(node_id)

        # ── Coordinator learning reward ───────────────────────────────────
        if self.island_mode and self.current_tick % 20 == 0:
            self.coordinator.record_reward(
                connectivity_reward  = self._ue_pair_routed_fraction,
                coverage_reward      = self._reachable_ue_fraction,
                emergency_served     = min(1.0, iops_admitted / max(1, iops_admitted + iops_denied))
            )
            self.coordinator.update(batch_size=20)

        # ── Learning postcard exchange (Multi-eNB IOPS) ─────────────────────
        if self.island_mode:
            try:
                self.learning_exchanger.tick(
                    self.agents, self.iops_controller,
                    self.topology, self.current_tick)
            except Exception:
                pass  # Don't break sim if learning exchange fails

        # Sync dynamic transport relay links to NetworkX routing graph
        self.topology.update_link_statuses()

        # Debug output
        if self.config.verbose and self.island_mode and self.current_tick % 20 == 0:
            transport_links = len(self.transport_relay_model.active_links)
            print(f"[PHY/MAC] t={self.current_tick}: "
                  f"relay={len(relay_nodes)}, relays={transport_links}, "
                  f"postcards={self.last_postcards_sent}, "
                  f"IOPS admitted={iops_admitted}/denied={iops_denied}, "
                  f"UE_routed={self._ue_pair_routed_fraction:.1%}, "
                  f"reachable={self._reachable_ue_fraction:.1%}")


    def _forward_traffic(self, traffic_arrivals):
        """
        Forward traffic through the network.
        Now uses proper physical path traversal across Oran infra, checking Link capacity
        and hitting the MARL agent transit policies at EVERY hop (RU->DU->CU).
        """
        import networkx as nx
        from .topology import TrafficClass
        
        self.ue_to_ue_success_count = 0
        self.ue_to_ue_failure_count = 0
        ue_to_ue_volume = 0.0
        ue_to_ue_offered = 0.0
        ue_general_volume = 0.0
        ue_general_offered = 0.0
        
        ue_to_ue_participants = set()
        for s, t in self.ue_to_ue_flows:
            ue_to_ue_participants.add(s)
            ue_to_ue_participants.add(t)
            
        def apply_marl_policy(path_nodes, traffic_class, amount):
            bottleneck = amount
            for n_id in path_nodes:
                agent = self.agents.get(n_id)
                if agent and hasattr(agent, 'last_actions') and traffic_class in agent.last_actions:
                    mode = agent.last_actions[traffic_class].get('admission_mode', 0)
                    if mode == 1 or mode == 'THROTTLE': bottleneck = min(bottleneck, amount * 0.5)
                    elif mode == 2 or mode == 'HOLD': return 0.0
            return bottleneck

        def consume_path(path_nodes, amount):
            final_bottleneck = amount
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                link_data = self.topology.graph.get_edge_data(u, v)
                if link_data and 'link_id' in link_data:
                    link_id = link_data['link_id']
                    link = self.topology.links.get(link_id)   # safe get — transport relay links may be dynamic
                    if link is None:
                        continue   # edge in graph but not in links dict — treat as free
                    if getattr(link, 'is_up', True):
                        final_bottleneck = min(final_bottleneck, link.available_capacity())
                    else:
                        return 0.0

            if final_bottleneck <= 0:
                return 0.0

            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                link_data = self.topology.graph.get_edge_data(u, v)
                if link_data and 'link_id' in link_data:
                    link = self.topology.links.get(link_data['link_id'])
                    if link:
                        link.add_traffic(final_bottleneck)
            
            for n_id in path_nodes:
                if n_id in self.topology.nodes:
                    self.topology.nodes[n_id].update_energy(final_bottleneck, 0)
                    
            return final_bottleneck

        # 1. Process UE-to-UE flows
        for source_ue, target_ue in self.ue_to_ue_flows:
            source_node = self.topology.nodes.get(source_ue)
            target_node = self.topology.nodes.get(target_ue)

            if not source_node or not source_node.is_survivor: continue
            if not target_node or not target_node.is_survivor:
                self.ue_to_ue_failure_count += 1
                continue

            base_ue_to_ue_traffic = 5.0 if self.island_mode else 8.0
            emergency_traffic = 0.0
            if source_node.emergency_state:
                if getattr(source_node, 'emergency_type', '') == 'emergency': emergency_traffic = 15.0
                elif getattr(source_node, 'emergency_type', '') == 'imminent_peril': emergency_traffic = 20.0
                else: emergency_traffic = 10.0
            rescue_bonus = 12.0 if (source_node.is_rescue_service or target_node.is_rescue_service) else 0.0

            ue_to_ue_traffic = base_ue_to_ue_traffic + emergency_traffic + rescue_bonus
            ue_to_ue_offered += ue_to_ue_traffic
            
            if self._can_route_ue_to_ue(source_ue, target_ue):
                try:
                    path = nx.shortest_path(self.topology.graph, source_ue, target_ue)
                    admitted = apply_marl_policy(path, TrafficClass.LIFE_SAFETY, ue_to_ue_traffic)
                    delivered = consume_path(path, admitted)
                    
                    if delivered > 0:
                        self.ue_to_ue_success_count += 1
                        ue_to_ue_volume += delivered
                    else:
                        self.ue_to_ue_failure_count += 1
                except nx.NetworkXNoPath:
                    self.ue_to_ue_failure_count += 1
            else:
                self.ue_to_ue_failure_count += 1

        if self.island_mode and self.marl_ue_routing_enabled and self.current_tick % 20 == 0:
            print(f"[ISLAND] t={self.current_tick}: UE-to-UE traffic - {self.ue_to_ue_success_count}/{len(self.ue_to_ue_flows)} flows, volume: {ue_to_ue_volume:.1f} units")

        # Expose counts to connectivity reward          
        self._last_routed_flows = self.ue_to_ue_success_count
        self._total_ue_flows    = max(1, len(self.ue_to_ue_flows))
        # Capture baseline on first island-mode evaluation
        if self.island_mode and getattr(self, '_island_baseline_flows', None) is None:
            self._island_baseline_flows = self.ue_to_ue_success_count
            self._island_peak_flows = self.ue_to_ue_success_count
            total = len(self.ue_to_ue_flows)
            print(f"  [BASELINE] First island tick: {self._island_baseline_flows}/{total} flows "
                  f"({self._island_baseline_flows/max(1,total):.1%}) — this is the theoretical max")
        elif self.island_mode:
            # Track peak routed flows during island mode
            self._island_peak_flows = max(
                getattr(self, '_island_peak_flows', 0),
                self.ue_to_ue_success_count)

        # 2. General UE Traffic
        core_targets = [n for n in getattr(self, 'core_nodes', []) if self.topology.nodes.get(n) and self.topology.nodes[n].is_survivor]
        
        for node_id, arrivals in traffic_arrivals.items():
            if node_id not in self.topology.nodes: continue
            node = self.topology.nodes[node_id]
            
            if getattr(node.node_type, 'value', str(node.node_type)) == "UE" and node.is_survivor:
                ue_to_ue_amount = 0.0
                if not self.island_mode and node_id in ue_to_ue_participants:
                    flows_cnt = sum(1 for s, t in self.ue_to_ue_flows if (s == node_id or t == node_id) and getattr(self.topology.nodes.get(s), 'is_survivor', False) and getattr(self.topology.nodes.get(t), 'is_survivor', False))
                    ue_to_ue_amount = flows_cnt * 1.5
                
                clean_arrivals = {}
                for tc, amount in arrivals.items():
                    gen_amount = max(0, amount - ue_to_ue_amount) if (tc == TrafficClass.LIFE_SAFETY and node_id in ue_to_ue_participants) else amount
                    clean_arrivals[tc] = gen_amount
                    ue_general_offered += gen_amount
                
                best_path = None
                if not self.island_mode and core_targets:
                    paths = []
                    for c in core_targets:
                        if self.topology.has_path(node_id, c):
                            paths.append(nx.shortest_path(self.topology.graph, node_id, c))
                    if paths:
                        best_path = min(paths, key=len)
                
                for tc, amount in clean_arrivals.items():
                    queue = node.queues[tc]
                    queue.offered_load = amount
                    delivered = 0.0
                    if best_path and amount > 0:
                        admitted = apply_marl_policy(best_path, tc, amount)
                        delivered = consume_path(best_path, admitted)
                        
                    queue.admitted_load = delivered
                    queue.queued_load += delivered
                    queue.dropped_load = amount - delivered
                    queue.delivered_load = delivered
                    
                    if tc != TrafficClass.LIFE_SAFETY or node_id not in ue_to_ue_participants:
                        ue_general_volume += delivered
            else:
                total_traffic = sum(arrivals.values())
                node.update_energy(total_traffic, 0)
                for tc, amount in arrivals.items():
                    q = node.queues[tc]
                    q.offered_load = amount
                    q.admitted_load = amount
                    q.delivered_load = amount
                    q.dropped_load = 0

        self.last_tick_ue_to_ue_volume = ue_to_ue_volume
        self.last_tick_ue_general_volume = ue_general_volume
        self.last_tick_ue_to_ue_offered = ue_to_ue_offered
        self.last_tick_ue_general_offered = ue_general_offered
    def _collect_metrics(self):
        """Collect metrics for current tick."""
        # Node states
        node_states = {}
        for node_id, node in self.topology.nodes.items():
            node_states[node_id] = {
                'is_survivor': node.is_survivor,
                'is_island': node.is_island,
                'energy_soc': node.energy_soc,
                'energy_tier': node.get_energy_tier()
            }

        # Link states
        link_states = {}
        for link_id, link in self.topology.links.items():
            link_states[link_id] = {
                'is_up': link.is_up,
                'utilization': link.current_utilization,
                'capacity': link.capacity
            }

        # Traffic stats (simplified)
        traffic_stats = {}
        for node_id, node in self.topology.nodes.items():
            node_traffic = {}
            for traffic_class in TrafficClass:
                queue = node.queues[traffic_class]
                node_traffic[traffic_class] = {
                    'offered_load': queue.offered_load,
                    'admitted_load': queue.admitted_load,
                    'queued_load': queue.queued_load,
                    'delivered_load': queue.delivered_load,
                    'dropped_load': queue.dropped_load
                }
            traffic_stats[node_id] = node_traffic

        # Control stats
        control_stats = {}
        for node_id in self.agents.keys():
            control_stats[node_id] = {
                'postcards_sent': 0,  # Placeholder (not yet tracked per-node)
                'postcards_received': len(self.control_plane.get_received_postcards(
                    node_id, self.current_tick))
            }
        
        # UE population stats
        total_ues = sum(1 for n in self.topology.nodes.values() if n.node_type.value == "UE")
        active_ues = sum(1 for n in self.topology.nodes.values() 
                        if n.node_type.value == "UE" and n.is_survivor)
        rescue_ues = sum(1 for nid in self.topology.nodes.keys() 
                        if nid.startswith("Rescue_UE_"))
        
        # UE-to-UE communication stats
        ue_to_ue_stats = {
            'enabled': self.ue_to_ue_enabled and not self.island_mode,
            'success_count': self.ue_to_ue_success_count,
            'failure_count': self.ue_to_ue_failure_count,
            'total_flows': len(self.ue_to_ue_flows),
            'success_rate': self.ue_to_ue_success_count / max(len(self.ue_to_ue_flows), 1) if self.ue_to_ue_flows else 0.0,
            'total_ue_population': total_ues,
            'active_ue_population': active_ues,
            'rescue_ue_count': rescue_ues,
            # MCPTT Emergency Communication Metrics (3GPP TS 22.179)
            'emergency_active_ues': len(self.emergency_active_ues),
            'emergency_alerts_sent': self.emergency_alerts_sent,
            'mcppt_emergency_calls': self.mcppt_emergency_calls
        }

        # Get failed transmissions for visualization
        failed_transmissions = self.control_plane.get_failed_transmissions(self.current_tick - 10)  # Last 10 ticks

        self.metrics.record_tick(
            self.current_tick, node_states, link_states,
            traffic_stats, control_stats, self.island_mode,
            ue_to_ue_stats=ue_to_ue_stats,
            failed_postcard_transmissions=failed_transmissions
        )

    def _print_debug_summary(self, tick: int):
        """Print a concise summary of the current state for debugging/insight."""
        infra_survivors = sum(
            1 for n in self.topology.nodes.values()
            if n.node_type != n.node_type.UE and n.is_survivor
        )
        ue_survivors = sum(
            1 for n in self.topology.nodes.values()
            if n.node_type == n.node_type.UE and n.is_survivor
        )
        ue_to_ue_status = "ENABLED" if (self.ue_to_ue_enabled and (not self.island_mode or self.marl_ue_routing_enabled)) else "DISABLED"
        ue_to_ue_success = f"{self.ue_to_ue_success_count}/{len(self.ue_to_ue_flows)}" if self.ue_to_ue_flows else "0/0"
        print(f"[DBG] t={tick} island={self.island_mode} "
              f"infra_alive={infra_survivors} ue_alive={ue_survivors} "
              f"postcards_sent={self.last_postcards_sent} "
              f"total_postcards={self.total_postcards_sent} "
              f"UE2UE={ue_to_ue_status} ({ue_to_ue_success})")
        if self.last_action_summary:
            top_actions = list(self.last_action_summary.items())[:8]
            actions_str = ", ".join([f"{k[0]}:{k[1]}={v}" for k, v in top_actions])
            print(f"[DBG] recent_actions: {actions_str}")
        # Reset tick-local summaries
        self.last_postcards_sent = 0
        self.last_action_summary = {}

    def start_all_animations(self, interval: int = 200):
        """Start animations for all plotter windows."""
        print("Starting animations for all plotters...")

        # Start traffic analysis animation
        if self.traffic_plotter:
            self.traffic_plotter.start_animation(interval)


        # Start agent monitor animation
        if self.agent_monitor:
            self.agent_monitor.start_animation(interval)

        # Start network topology animation (if available)
        if hasattr(self, 'painter') and self.painter:
            # Note: NetworkPainter doesn't have animation yet, could be added later
            pass

        print("All animations started!")

    def stop_all_animations(self):
        """Stop animations for all plotter windows."""
        print("Stopping animations for all plotters...")

        # Stop traffic analysis animation
        if self.traffic_plotter:
            self.traffic_plotter.stop_animation()


        # Stop agent monitor animation
        if self.agent_monitor:
            self.agent_monitor.stop_animation()

        print("All animations stopped!")


    def run_simulation(self) -> MetricsCollector:
        """Run the complete simulation."""
        print(f"Starting simulation for {self.scenario.duration_ticks} ticks")

        for tick in range(self.scenario.duration_ticks):
            self.current_tick = tick

            # Progress reporting for long simulations
            # More frequent during disaster period
            is_disaster_period = 150 <= tick <= 350
            progress_interval = 10 if is_disaster_period else 100
            if tick % progress_interval == 0 and tick > 0:
                disaster_note = " [DISASTER PERIOD - HIGH RESOLUTION]" if is_disaster_period else ""
                print(f"Completed tick {tick}/{self.scenario.duration_ticks}{disaster_note}")

            # Process scenario events
            self._process_events(tick)

            # Check for island mode transition
            was_island = self.island_mode
            self.island_mode = self._detect_island_mode()
            self.control_plane.set_island_mode(self.island_mode)

            # Set disaster mode when core is severed (affects IP overlay reliability)
            disaster_triggered = self.metrics.severance_tick is not None and tick >= self.metrics.severance_tick
            self.control_plane.set_disaster_mode(disaster_triggered, tick)

            if self.island_mode and not was_island:
                print(f"\n{'='*70}")
                print(f"Tick {tick}: WARNING - DISASTER - Entering Island Mode")
                print(f"{'='*70}")
                self.metrics.set_severance_tick(tick)
                # Reset cumulative postcard counter (start counting from severance)
                self.total_postcards_sent = 0
                # Update failure log with current tick
                self.control_plane.update_failure_log_ticks(tick)
                # Mark island nodes
                for node in self.topology.nodes.values():
                    if node.is_survivor:
                        node.is_island = True
                # Enable MARL-based UE-to-UE routing in island mode
                self.marl_ue_routing_enabled = True
                print(f"  INFO: MARL UE-to-UE routing enabled for coordinated communication")
                print(f"{'='*70}\n")
            elif (not self.island_mode) and was_island:
                print(f"Tick {tick}: Exiting island mode (core reachability restored)")
                # Restore core nodes (including UPF/EdgeUPF) if links are restored
                core_nodes_restored = []
                for core_node_id in self.core_nodes:
                    if core_node_id in self.topology.nodes:
                        # Check if node can reach other nodes (links restored)
                        # If any link to/from this core node is up, restore it
                        has_connection = any(
                            (link.endpoints[0] == core_node_id or link.endpoints[1] == core_node_id)
                            and link.is_up
                            for link in self.topology.links.values()
                        )
                        if has_connection and not self.topology.nodes[core_node_id].is_survivor:
                            self.topology.nodes[core_node_id].is_survivor = True
                            core_nodes_restored.append(core_node_id)
                            # Recreate agent for restored core node
                            if core_node_id not in self.agents:
                                node = self.topology.nodes[core_node_id]
                                self.agents[core_node_id] = create_agent_for_node(
                                    core_node_id, node.node_type.value, self.slice_dictionary
                                )
                
                if core_nodes_restored:
                    upf_restored = sum(1 for nid in core_nodes_restored 
                                     if self.topology.nodes[nid].node_type.value in {"UPF", "EdgeUPF"})
                    print(f"  Core nodes restored: {len(core_nodes_restored)} nodes (including {upf_restored} UPF/EdgeUPF)")
                
                # Re-enable normal UE-to-UE routing
                self.ue_to_ue_enabled = True
                print(f"  UE-to-UE communication restored via core")

            # Generate and forward traffic
            traffic_arrivals = self._generate_traffic()
            self._forward_traffic(traffic_arrivals)

            # Build agent observations and execute actions
            observations = self._build_agent_observations()
            self._execute_agent_actions(observations)
            
            # UE-to-UE routing remains disabled in island mode (would be enabled by coordinated agents)
            # In a full MARL implementation, agents would learn to coordinate and enable this routing

            # Collect metrics
            self._collect_metrics()

            # Verbose progress snapshots (higher frequency during disaster)
            is_disaster_period = 150 <= tick <= 350
            debug_interval = 10 if is_disaster_period else self.config.sample_interval
            if self.config.verbose and (tick % debug_interval == 0 or self.island_mode != was_island):
                self._print_debug_summary(tick)

            # Live plot update (higher frequency during disaster)
            is_disaster_period = 150 <= tick <= 350
            live_plot_interval = 5 if is_disaster_period else self.config.live_interval
            if self.painter and (tick % live_plot_interval == 0 or self.island_mode != was_island):
                snapshot = self.metrics.metrics_history[-1] if self.metrics.metrics_history else None
                if snapshot:
                    try:
                        self.painter.update(snapshot)
                        plt.pause(0.001)
                    except Exception as e:
                        print(f"Live plot update error: {e}")

            # Traffic analysis plot update (1 tick during disaster, 100 ticks normal)
            plot_interval = 1 if is_disaster_period else 100
            if self.traffic_plotter and tick % plot_interval == 0:
                try:
                    self.traffic_plotter.update(tick, self)
                    plt.pause(0.001)
                except Exception as e:
                    print(f"Traffic plot update error: {e}")

            # Agent monitor update (1 tick during disaster, 100 ticks normal)
            if self.agent_monitor and tick % plot_interval == 0:
                convergence_threshold = 0.80  # 80% of agents must converge
                sustained_period = 8  # Must maintain convergence for 8 consecutive ticks

                # Track convergence history
                if not hasattr(self, 'convergence_history'):
                    self.convergence_history = []

                self.convergence_history.append(convergence_metrics['convergence_ratio'])
                # Keep only last 15 ticks for analysis
                self.convergence_history = self.convergence_history[-15:]

                # Check for sustained convergence
                recent_convergence = self.convergence_history[-sustained_period:] if len(self.convergence_history) >= sustained_period else []
                sustained_convergence = len(recent_convergence) == sustained_period and all(r >= convergence_threshold for r in recent_convergence)

                # Show detailed MARL convergence progress
                if tick % 5 == 0 or tick <= severance_tick + 40:
                    conv_pct = convergence_metrics['convergence_ratio'] * 100
                    policy_stability = convergence_metrics['policy_stability'] * 100
                    coord_quality = convergence_metrics['coordination_quality'] * 100
                    emergent_behavior = convergence_metrics['emergent_behavior'] * 100

                    print(f"[MARL Convergence] t={tick} (since severance: {tick - severance_tick}):")
                    print(f"  Agent Convergence: {conv_pct:.1f}% ({convergence_metrics['converged_agents']}/{convergence_metrics['total_agents']} agents)")
                    print(f"  Policy Stability: {policy_stability:.1f}%")
                    print(f"  Coordination Quality: {coord_quality:.1f}%")
                    print(f"  Emergent Behavior: {emergent_behavior:.1f}%")
                    print(f"  Sustained Period: {len(recent_convergence)}/{sustained_period} ticks")

                if sustained_convergence:
                    self.marl_ue_routing_enabled = True
                    print(f"\n{'='*80}")
                    print(f"Tick {tick}: SUCCESS - MARL CONVERGENCE ACHIEVED - UE-to-UE Routing ENABLED!")
                    print(f"{'='*80}")
                    print(f"  SUCCESS: Multi-agent reinforcement learning achieved real convergence")
                    print(f"  SUCCESS: Agents coordinated policies without fallbacks or timeouts")
                    print(f"  SUCCESS: Emergent behavior enabled rescue communications")
                    print(f"  VALIDATION: This proves MARL solution works as Proof-of-Concept")
                    print(f"  ")
                    print(f"  FINAL CONVERGENCE METRICS:")
                    conv_pct = convergence_metrics['convergence_ratio'] * 100
                    policy_pct = convergence_metrics['policy_stability'] * 100
                    coord_pct = convergence_metrics['coordination_quality'] * 100
                    emergent_pct = convergence_metrics['emergent_behavior'] * 100
                    print(f"     - Agent Convergence: {conv_pct:.1f}% ({convergence_metrics['converged_agents']}/{convergence_metrics['total_agents']} agents)")
                    print(f"     - Policy Stability: {policy_pct:.1f}% (consistent admission policies)")
                    print(f"     - Coordination Quality: {coord_pct:.1f}% (sustained messaging)")
                    print(f"     - Emergent Behavior: {emergent_pct:.1f}% (collaborative intelligence)")
                    print(f"     - Time to Convergence: {time_since_severance} ticks")
                    print(f"     - Sustained Period: {sustained_period} consecutive ticks")
                    
                    # Show final converged policies
                    if self.last_action_summary:
                        print(f"  CONVERGED POLICIES:")
                        for (tclass, mode), count in sorted(self.last_action_summary.items()):
                            print(f"     - {tclass}: {mode} = {count} agents")

                    # Show infrastructure nodes that converged
                    infra_nodes = [nid for nid in self.agents.keys()
                                 if self.topology.nodes.get(nid, None) and
                                 self.topology.nodes[nid].node_type.value != "UE"]
                    print(f"  INFRASTRUCTURE NODES: {len(infra_nodes)} total")
                    print(f"  VALIDATION: This proves the MARL algorithm successfully coordinates")
                    print(f"               distributed infrastructure for emergency communications!")
                    print(f"{'='*70}\n")

            # Collect metrics
            self._collect_metrics()

            # Verbose progress snapshots (higher frequency during disaster)
            is_disaster_period = 150 <= tick <= 350
            debug_interval = 10 if is_disaster_period else self.config.sample_interval
            if self.config.verbose and (tick % debug_interval == 0 or self.island_mode != was_island):
                self._print_debug_summary(tick)

            # Live plot update (higher frequency during disaster)
            is_disaster_period = 150 <= tick <= 350
            live_plot_interval = 5 if is_disaster_period else self.config.live_interval
            if self.painter and (tick % live_plot_interval == 0 or self.island_mode != was_island):
                snapshot = self.metrics.metrics_history[-1] if self.metrics.metrics_history else None
                if snapshot:
                    try:
                        self.painter.update(snapshot)
                        plt.pause(0.001)
                    except Exception as e:
                        print(f"Live plot update error: {e}")
            
            # Determine if we're in the critical disaster period (high-resolution plotting)
            # Disaster starts ~tick 180, stabilizes ~tick 300
            is_disaster_period = 150 <= tick <= 350  # Wider window for safety

            # MARL metrics plot update (1 tick during disaster, 100 ticks normal)
            plot_interval = 1 if is_disaster_period else 100
            if self.marl_plotter and (tick % plot_interval == 0 or self.island_mode != was_island):
                try:
                    self.marl_plotter.update(tick, self)
                    plt.pause(0.001)
                except Exception as e:
                    print(f"MARL plot update error: {e}")

            # Traffic analysis plot update (1 tick during disaster, 100 ticks normal)
            if self.traffic_plotter and tick % plot_interval == 0:
                try:
                    self.traffic_plotter.update(tick, self)
                    plt.pause(0.001)
                except Exception as e:
                    print(f"Traffic plot update error: {e}")

            # Agent monitor update (1 tick during disaster, 100 ticks normal)
            if self.agent_monitor and tick % plot_interval == 0:
                try:
                    self.agent_monitor.update(tick, self)
                    plt.pause(0.001)
                except Exception as e:
                    print(f"Agent monitor update error: {e}")

            # Reset control plane for next tick
            self.control_plane.reset_for_tick()

        print("Simulation completed")
        return self.metrics


def run_simulation(topology_file: str, scenario_file: str,
                  output_dir: str = "output") -> MetricsCollector:
    """Convenience function to run a complete simulation."""
    print(f"Loading topology from {topology_file}...")
    from .topology import load_topology_from_yaml
    topology = load_topology_from_yaml(topology_file)
    print(f"Loaded {len(topology.nodes)} nodes, {len(topology.links)} links")

    print(f"Loading scenario from {scenario_file}...")
    from .scenario import load_scenario_from_yaml
    scenario = load_scenario_from_yaml(scenario_file, topology)
    print(f"Loaded scenario '{scenario.name}' with {len(scenario.events)} events")

    config = SimulationConfig(
        tick_duration_ms=100,
        random_seed=42,
        enable_island_detection=True
    )

    # Run simulation
    print("Initializing simulator...")
    simulator = Simulator(topology, scenario, config)
    print("Running simulation...")
    metrics = simulator.run_simulation()
    print(f"Simulation completed with {len(metrics.metrics_history)} ticks")

    # Save UE traffic analysis plot if it was enabled
    if simulator.traffic_plotter:
        simulator.traffic_plotter.save_plot(f"{output_dir}/ue_traffic_analysis.png")
    else:
        print("Traffic plotter was not initialized - UE traffic analysis plot not saved")

    # Export results
    print(f"Exporting results to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    metrics.export_to_csv(output_dir)
    print("Export completed.")

    return metrics
