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
import math
import zlib
import random
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from .topology import Topology, Node, Link, TrafficClass, NodeType, LinkType, InterfaceType
from .traffic import TrafficGenerator, SliceDictionary
from .agent import (BaseAgent, AgentObservation, EnergyTier, StrainLevel,
                    create_agent_for_node, LocalSliceState, RLAgent, CentralizedMARLTrainer,
                    PHYMACAction, NeighborRadioSummary, ConnectivityState,
                    ControlPostcard, StrainLevel)
from .phy_mac_state import (PHYMACState, RelayMode, MCSLevel, MACScheduler,
                            TX_POWER_STEPS_DB, MCS_OFFSET_CENTER,
                            auto_cqi_mcs_idx)
from .transport_relay_model import (
    TransportRelayModel, RADIO_CLASS_BY_BAND,
    BAND_ACCESS, BAND_TG, BAND_MW,
    ACCESS_FR1, TG_MESH_60GHZ, MW_PTP, MW_MAX_HOP_M,
    access_channel_group, transport_channel_group,
    integrated_access_link_capacity, IAB_ACCESS_MIN_USABLE_MBPS,
)
from .iops_manager import IOPSManager
from .coordinator_agent import CoordinatorAgent, GlobalPolicyVector
from .control_plane import ControlPlaneManager
from .scenario import Scenario, ScenarioEvent
from .metrics import MetricsCollector, TickMetrics
from .reward_profile import PROFILE as _RP

import matplotlib
# Respect env override first; fallback to TkAgg then Agg
backend_env = os.environ.get("MPLBACKEND")
if backend_env:
    try:
        matplotlib.use(backend_env, force=True)
    except (ImportError, Exception):
        matplotlib.use("Agg", force=True)
else:
    try:
        matplotlib.use("TkAgg")
    except (ImportError, Exception):
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ── Which link types are subject to access-class MCS capacity scaling ─────────
#
# consume_path() scales a link's provisioned capacity by min(endpoint SE)/MAX_SE
# to model link adaptation.  That is only meaningful for links whose throughput
# is actually set by the FR1 access-class modulation:
#
#   INCLUDED  WIRELESS        the Uu air interface (UE <-> O-RU)
#             MULTIHAUL_MESH  60 GHz TG mesh hop
#             MICROWAVE_PTP   18 GHz MW PtP hop
#             MICROWAVE       legacy generic microwave alias
#             D2D             sidelink
#
#   EXCLUDED  FIBER           an optical link's 25 Gbps has nothing to do with
#                             either endpoint's radio SINR.  Scaling it meant a
#                             faded O-RU dimmed its own Open-Fronthaul (and at
#                             SINR < -3 dB switched it off entirely).
#             SATELLITE       its own link budget, not the FR1 access MCS.
#             TRANSPORT_RELAY capacity already derived from the relay's OWN
#                             budget at creation time (can_form_relay_link /
#                             mw_link_feasible); re-scaling double-counts link
#                             adaptation.
#
# Gating applies identically to every arm — it is shared physics, not an arm
# feature.
_MCS_SCALED_LINK_TYPES = frozenset({
    LinkType.WIRELESS,
    LinkType.MULTIHAUL_MESH,
    LinkType.MICROWAVE_PTP,
    LinkType.MICROWAVE,
    LinkType.D2D,
})


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
    
    


# ── WHICH NODES HOST AN AGENT ────────────────────────────────────────────────
# Exactly the node types that own radio hardware, i.e. the ones
# _init_phy_mac_states gives a PHYMACState to.  A node without a PHYMACState has
# nothing for a transmit-power, MCS or relay action to act upon, so an agent
# there contributes only inert samples -- to the PPO batch, and to the
# cross-agent mean that the advantage centring subtracts.  Keeping the two sets
# identical is what stops the acting fleet and the trained fleet from drifting
# apart (they had: 35 agents, 22 of them able to act).
#
#   MARL_AGENT_FLEET=radio      (default) one agent per radio-capable site
#   MARL_AGENT_FLEET=all_infra  the previous behaviour, kept so that results
#                               published before this fix reproduce
# Postcard delivery.  get_received_postcards is a DESTRUCTIVE read, so a
# second reader gets nothing.  'shared' drains once and gives the same set to
# the coordinator and to the agents; 'legacy' reproduces the earlier behaviour,
# in which the coordinator consumed every queue and all 16 Block B observation
# dimensions were dataclass constants.
POSTCARD_DRAIN = os.environ.get('MARL_POSTCARD_DRAIN', 'shared').strip().lower()
if POSTCARD_DRAIN not in ('shared', 'legacy'):
    raise ValueError(
        f"MARL_POSTCARD_DRAIN={POSTCARD_DRAIN!r} is not one of 'shared', 'legacy'")

RADIO_AGENT_TYPES = {NodeType.O_RU, NodeType.O_DU, NodeType.RELAY,
                     NodeType.GNBSITE, NodeType.DU}

AGENT_FLEET = os.environ.get('MARL_AGENT_FLEET', 'radio').strip().lower()
if AGENT_FLEET not in ('radio', 'all_infra'):
    raise ValueError(
        f"MARL_AGENT_FLEET={AGENT_FLEET!r} is not one of 'radio', 'all_infra'")


# Which sites can actually re-point a steerable head.
#   MARL_STEER_MODEL=actionable  (default) capability AND radio hardware
#   MARL_STEER_MODEL=flagged     the topology flag alone, as published
STEER_MODEL = os.environ.get('MARL_STEER_MODEL', 'actionable').strip().lower()
if STEER_MODEL not in ('actionable', 'flagged'):
    raise ValueError(
        f"MARL_STEER_MODEL={STEER_MODEL!r} is not one of 'actionable', 'flagged'")


def site_can_steer(node) -> bool:
    """SINGLE SOURCE OF TRUTH: can this site re-point a steerable head?

    Two conditions, and they were previously checked in three places under
    three different rules:
      * the topology grants the site a steerable head (has_multihaul), and
      * the site owns radio hardware the engine will actually step, i.e. a
        PHYMACState, which _init_phy_mac_states grants only to
        RADIO_AGENT_TYPES.

    A node carrying the flag but no radio hardware -- EdgeUPF in the
    evaluation topology -- is NOT steerable: _step_phy_mac never runs for it,
    so nothing it "decides" can reach the transport layer. Counting such a
    node as steerable inflates the planning statistics and, worse, makes the
    attainable-optimum solver claim reunifications no arm can perform.

    The runtime engine, the severance generator and the optimum solver must
    all call this, or they will disagree again.
    """
    if not getattr(node, 'has_multihaul', False):
        return False
    if STEER_MODEL == 'flagged':
        return True
    return getattr(node, 'node_type', None) in RADIO_AGENT_TYPES


def node_hosts_agent(node) -> bool:
    """True if this node should run a policy."""
    if not getattr(node, 'is_survivor', False):
        return False
    ntype = getattr(node, 'node_type', None)
    if getattr(ntype, 'value', str(ntype)) == 'UE':
        return False
    if AGENT_FLEET == 'all_infra':
        return True
    return ntype in RADIO_AGENT_TYPES


class Simulator:
    """Main simulation coordinator."""

    def __init__(self, topology: Topology, scenario: Scenario, config: SimulationConfig):
        self.topology = topology
        self.scenario = scenario
        self.config = config

        # Reproducibility: seed all RNG sources from the run seed so identical
        # seeds give identical runs across processes.
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)

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
            # One agent per radio-capable survivor; see node_hosts_agent().
            if node_hosts_agent(node):
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

        # ── Aggregate co-channel interference cache ──────────────────────
        # {(band, node_id): I_dbm} for the tick stamped in
        # _interference_tick.  Rebuilt at most once per tick by
        # _compute_aggregate_interference(); see the AGGREGATE CO-CHANNEL
        # INTERFERENCE block for the model.
        self._interference_cache: Dict[Tuple[str, str], float] = {}
        self._interference_tick: Optional[int] = None

        # Relay-churn tracking for compute_global_connectivity_reward:
        # node_id -> relay_mode at the previous reward computation.  Starts
        # empty each episode (a fresh Simulator is built per episode), so the
        # first tick never pays a churn penalty.
        self._prev_relay_modes: Dict[str, 'RelayMode'] = {}

        # ── MultiHaul beam-steering bookkeeping ──────────────────────────
        # link_id -> tick at which a re-pointed relay link finishes beam
        # training and may be brought up (see RELAY_REPOINT_TICKS).
        self._relay_link_ready_tick: Dict[str, int] = {}
        # link_id -> tick a relay link first became a cross-fragment BRIDGE
        # (drives the RELAY_BRIDGE_GRACE_TICKS teardown grace period).
        self._relay_bridge_since: Dict[str, int] = {}
        # link_id -> utilisation fraction on the LAST completed forwarding
        # pass (drives the RELAY_BRIDGE_BUSY_UTIL make-before-break rule).
        # Refreshed by _snapshot_relay_link_load() at the end of
        # _forward_traffic, so it is well defined under BOTH tick orderings
        # (engine run(): forward-then-act; comparison harness: act-then-forward).
        self._relay_link_load: Dict[str, float] = {}
        # Uniform-random reference policy (bridging-attribution ablation).
        # When True, _execute_agent_actions ignores the policy network and
        # samples every action head uniformly — see the flag's use there.
        self.random_action_policy: bool = False
        # ── Admission census (cumulative over the episode) ────────────────
        # See apply_marl_policy in _forward_traffic.  Counts how often the
        # policy's own PRB split zeroed or halved a flow, so the self-inflicted
        # share of an arm's losses can be separated from control quality.
        # Only arms that run agents can register here at all — that asymmetry
        # is precisely what these numbers exist to quantify.
        self._adm_calls: int = 0
        self._adm_hold: int = 0
        self._adm_throttle: int = 0
        self._adm_held_volume: float = 0.0
        self._adm_throttled_volume: float = 0.0
        # When True, _execute_agent_actions replaces the policy network with
        # the deterministic cross-layer controller in heuristic_controller —
        # the XL-DET baseline arm.  Mutually exclusive with the flag above.
        self.heuristic_action_policy: bool = False
        # ── ADAPTABILITY STRESS: external wideband interferer ──────────────
        # An out-of-distribution disaster condition NOT present in training or
        # the nominal benchmark: an uncoordinated external emitter (responder
        # radios, jamming, adjacent-channel debris) that raises co-channel
        # interference fleet-wide from a chosen tick onward.  It is pure
        # PHYSICS applied identically to every arm, so it isolates the POLICY
        # response: XL-DET's per-agent tx-power loop with a fixed brake cannot
        # coordinate a fleet-wide backoff, whereas a policy trained on the
        # ICIC postcard fields can in principle learn to.  Hard-gated: unset
        # env => stress_dbm is None => this branch never runs, so the nominal
        # results and every running job are provably unaffected.
        # ── TRANSPORT-RADIO POWER MODEL (MARL_TRANSPORT_POWER) ─────────────
        # MEASURED DEFECT.  Every 60 GHz TG / 18 GHz MW link budget in this
        # engine was evaluated at ps.tx_power_dbm -- the FR1 ACCESS transmit
        # power the policy controls.  An energy-aware policy drives access
        # power toward the 10 dBm floor, which silently shrank its BACKHAUL
        # beam reach from the planning 1068 m (@23 dBm) to roughly a quarter:
        # on seed 55 the selected policy had 21 of 22 sites in a relay mode
        # and formed zero bridges because no cross-fragment peer was inside
        # that budget, while XL-DET (holding 23-33 dBm) formed the one
        # feasible bridge.  Physically the MultiHaul head and the MW ODU are
        # separate radios with their own power amplifiers; a point-to-point
        # backhaul beam is not dialled down to save access-side energy.  This
        # is the same cross-radio coupling already corrected for PRBs
        # (normalise_prb: "they do not share spectrum, so they must not share
        # a PRB budget") -- they do not share a power amplifier either.
        #
        # 'planning' evaluates every transport budget and every transport-band
        # interference emission at TRANSPORT_PLANNING_TX_DBM, the same 23 dBm
        # the attainable-optimum solver assumes (FRAG_BRIDGE_TX_DBM in the
        # driver), so the benchmark's optimum definition is unchanged and
        # every arm -- XL-DET, SDN, routing, random, learned -- gets the same
        # backhaul reach.  Applied identically to all arms; the access budget
        # (PHYMACState.sinr_average etc.) is untouched.  Default: unset ->
        # legacy coupling, every shipped result bit-identical.
        # ── FRAGMENT-AWARE RE-POINTING (MARL_RELAY_REPOINT) ────────────────
        # MEASURED DEFECT.  A relay link is torn down ONLY on a relay-mode
        # change, and the fragment-aware bridging branch runs ONLY for a site
        # with no active link.  So a CAPACITY_BOOST site whose beam already
        # points at a same-fragment neighbour can never re-point to a
        # cross-fragment target: peer selection is fragment-aware at FIRST
        # formation and never again.  Seed 55, selected policy: 21 of 22
        # sites in CAPACITY_BOOST, 20 of them holding intra-fragment links,
        # the bridge peers feasible at current power, zero bridges, island
        # stuck at 2 components.  The engine's own comment says "beam steering
        # is the point"; this makes that true continuously, not once.
        #
        # 'fragment_aware': in island mode, a CAPACITY_BOOST site at a
        # steerable head whose current link is NOT a tree-edge bridge, that
        # is not inside a beam-training lockout, and for which a feasible
        # cross-fragment target exists, drops the non-bridge link and lets the
        # existing fragment-aware formation run in the same tick.  A per-site
        # cooldown (RELAY_REPOINT_COOLDOWN_TICKS) prevents oscillation, and
        # the abandoned peer's link state is cleared too (a link has two
        # ends).  Applied identically to every arm.  Default: unset ->
        # legacy behaviour, every shipped result bit-identical.
        # ── RELAY-LINK STATE RECONCILIATION (MARL_RELAY_STATE) ─────────────
        # MEASURED DEFECT.  The ONLY place PHYMACState.relay_link_active is
        # cleared is the acting node's own relay-mode change.  When the PEER
        # tears the shared link down (its own mode flip), or the link is
        # otherwise removed, this end keeps relay_link_active=True with no
        # link behind it -- and the fragment-aware bridging branch, gated on
        # `not relay_link_active`, then excludes the site for the rest of the
        # episode.  Instrumented on seed 55 (selected policy): 649 of the
        # 1308 re-point attempts were refused for exactly this reason, and
        # the island stayed at two components with 20 sites "holding" links
        # the topology no longer contained.  A link has two ends; this makes
        # the state say so.  'reconcile': each tick, a node whose flag claims
        # a link while no TRANSPORT_RELAY link (up OR pending beam-training)
        # involves it has its link state cleared.  Applied identically to
        # every arm.  Default: unset -> legacy, every shipped result
        # bit-identical.
        # ── DCC POSTCARD HEARTBEAT (MARL_POSTCARD_ALWAYS) ──────────────────
        # See the sender block in _execute_agent_actions.  Default off.
        self._postcard_always: bool = (os.environ.get('MARL_POSTCARD_ALWAYS', '0')
                                       .strip().lower() in ('1', 'true', 'on', 'yes'))
        _rs = os.environ.get('MARL_RELAY_STATE', '').strip().lower()
        if _rs not in ('', 'legacy', 'reconcile'):
            raise ValueError("MARL_RELAY_STATE=%r is not one of 'legacy', 'reconcile'" % _rs)
        self._relay_state_reconcile: bool = (_rs == 'reconcile')
        self._diag_phantom_resets: int = 0
        _rp = os.environ.get('MARL_RELAY_REPOINT', '').strip().lower()
        if _rp not in ('', 'off', 'legacy', 'fragment_aware'):
            raise ValueError("MARL_RELAY_REPOINT=%r is not one of 'off', 'fragment_aware'" % _rp)
        self._relay_repoint: bool = (_rp == 'fragment_aware')
        self._last_repoint_tick: Dict[str, int] = {}
        self._diag_repoints: int = 0
        _tp = os.environ.get('MARL_TRANSPORT_POWER', '').strip().lower()
        if _tp not in ('', 'legacy', 'planning'):
            raise ValueError("MARL_TRANSPORT_POWER=%r is not one of 'legacy', 'planning'" % _tp)
        self._transport_planning_power = (self.TRANSPORT_PLANNING_TX_DBM
                                          if _tp == 'planning' else None)
        _sd = os.environ.get('MARL_STRESS_INTERFERENCE_DBM')
        self._stress_i_dbm = (float(_sd) if _sd not in (None, '') else None)
        self._stress_i_tick = int(
            os.environ.get('MARL_STRESS_INTERFERENCE_TICK', '0') or '0')
        # Raw (relay-free) component count at the moment of severance — the
        # baseline the reunification reward measures progress against.
        self._post_severance_fragment_baseline: Optional[int] = None

        # Per-tick connectivity caches (populated at start of _build_agent_observations)
        self._tick_infra_nodes_cache: list  = []
        self._tick_bridge_set_cache:  set   = set()
        self._tick_intra_reach_cache: dict  = {}
        self._tick_core_dist_cache:   dict  = {}   # node_id -> hop count to nearest live core
        self._tick_ue_demand_cache:   dict  = {}   # anchor node_id -> # UE-to-UE flows anchored

        # Active traffic surges from 'traffic_surge' scenario events:
        # node_id -> (end_tick, multiplier)
        self._active_traffic_surges: Dict[str, Tuple[int, float]] = {}

        # Links taken down by node_failure / energy_depletion events
        # (node_id -> [link_id, ...]) so node_recovery can restore them.
        self._links_downed_by_node_failure: Dict[str, List[str]] = {}

        # Cached best-case island connectivity upper bound (None = recompute)
        self._island_bound_cache: Optional[int] = None

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
        # PEER reachability (see peer_reachable_ue_stats): fraction of surviving
        # UEs whose serving cell sits in the connected BODY of the survivor
        # network, and how many of those owe it to the policy's own bridges.
        self._peer_reachable_ue_fraction: float = 1.0
        self._stranded_ue_count:       int = 0
        self._reach_restored_ue_count: int = 0
        # Identities of the restored UEs (see peer_reachable_ue_stats) — used
        # by the reachability reward term when REACH_REQUIRE_TRAFFIC is on.
        self._reach_restored_ue_ids: frozenset = frozenset()
        # UEs that were an endpoint of a UE-to-UE flow which delivered
        # non-zero volume this tick; rebuilt every tick by _forward_traffic.
        self._delivered_ue_endpoints: set = set()
        self._peer_reach_stats:       dict = {}
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
    
    def _uu_link_endpoints(self, link):
        """Classify a Uu link's endpoints by NODE TYPE (not name prefix).

        Returns (ue_ep, anchor_ep) where anchor_ep is the non-UE radio anchor
        (O-RU, gNB-Site, NeNB, ...).  Works for both generated and YAML
        topologies, since topology loading sets interface_type "Uu" on UE
        access links regardless of node naming.
        """
        ue_ep, anchor_ep = None, None
        for ep in link.endpoints:
            ep_node = self.topology.nodes.get(ep)
            if ep_node is None:
                continue
            if getattr(ep_node.node_type, 'value', str(ep_node.node_type)) == 'UE':
                ue_ep = ep
            else:
                anchor_ep = ep
        return ue_ep, anchor_ep

    def _build_ue_to_oru_mapping(self):
        """Build efficient mapping from UEs to their connected O-RUs for fast UE-to-UE routing checks."""
        self.ue_to_oru_map = {}

        for link in self.topology.links.values():
            if link.is_up and hasattr(link, 'interface_type') and link.interface_type.value == "Uu":
                ue_ep, oru_ep = self._uu_link_endpoints(link)

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
                    ue_ep, other_end = self._uu_link_endpoints(link)
                    if ue_ep == ue_id and other_end:
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
        # Stable across processes: crc32 (unlike salted hash()) + run seed
        random.seed(zlib.crc32(new_ue_id.encode()) ^ (self.config.random_seed or 0))
        
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

    def _compute_island_upper_bound(self) -> int:
        """Count UE-to-UE flows physically routable in the best-case topology.

        Best case = current surviving topology (live links) PLUS every
        physically feasible transport relay link between surviving
        relay-capable infra nodes, using the same physics check the agents
        use when they actually form relay links
        (TransportRelayModel.can_form_relay_link) at max Tx power and full
        relay bandwidth.  This is the true upper bound that MARL performance
        is measured against.

        Cached in self._island_bound_cache; invalidated (set to None)
        whenever a scenario event changes the topology.
        """
        cached = getattr(self, '_island_bound_cache', None)
        if cached is not None:
            return cached

        # Base graph: surviving nodes + live links
        g = nx.Graph()
        for nid, n in self.topology.nodes.items():
            if n.is_survivor:
                g.add_node(nid)
        for link in self.topology.links.values():
            if not getattr(link, 'is_up', True):
                continue
            a, b = link.endpoints
            if a in g and b in g:
                g.add_edge(a, b)

        # Best-case relay links between relay-capable survivors
        relay_types = {NodeType.O_RU, NodeType.O_DU, NodeType.RELAY,
                       NodeType.GNBSITE, NodeType.DU}
        relay_nodes = [nid for nid, n in self.topology.nodes.items()
                       if n.is_survivor and n.node_type in relay_types]
        for nid in relay_nodes:
            if not self.transport_relay_model.node_positions.get(nid):
                n = self.topology.nodes[nid]
                self.transport_relay_model.register_position(
                    nid, getattr(n, 'x_pos', 0.0), getattr(n, 'y_pos', 0.0))
        for i, a in enumerate(relay_nodes):
            for b in relay_nodes[i + 1:]:
                if g.has_edge(a, b):
                    continue
                # Best case = the HARDWARE MAXIMUM Tx power, full relay
                # bandwidth, clear air, no interference.  Was 43.0 dBm, which
                # no radio in this model can reach (PHYMACState clamps to
                # tx_power_max_dbm = 33 dBm), so the "upper bound" was not
                # achievable by any arm and the achievability denominator was
                # inflated.  33 dBm gives the true best case; the extra
                # 10 dB used to buy ~2 km of TG reach against the ~1.53 km a
                # real radio can actually reach.
                feasible, cap, _sinr = self.transport_relay_model.can_form_relay_link(
                    a, b, tx_power_dbm=PHYMACState('').tx_power_max_dbm,
                    relay_bw_fraction=1.0)
                if feasible and cap > 0:
                    g.add_edge(a, b)

        # Count flows whose endpoints survive and are connected in best case
        routable = 0
        for s, t in getattr(self, 'ue_to_ue_flows', []):
            ns, nt = self.topology.nodes.get(s), self.topology.nodes.get(t)
            if not (ns and nt and ns.is_survivor and nt.is_survivor):
                continue
            if s in g and t in g and nx.has_path(g, s, t):
                routable += 1

        self._island_bound_cache = routable
        return routable

    def compute_optimal_connectivity(self) -> Dict[str, float]:
        """Compute optimal connectivity metrics.

        The theoretical maximum ('optimal') is the number of UE pairs that
        are physically routable in the current surviving topology assuming
        the best case: all feasible transport relay links active (see
        _compute_island_upper_bound).  Recomputed whenever a scenario event
        changes the topology.

        Returns dict with:
          - optimal_frac:     fraction of UE pairs routable at best case
          - actual_frac:      fraction of UE pairs currently routed
          - efficiency:       actual / optimal
          - optimal_flows:    best-case routable flows count
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

        # Real upper bound: physically routable flows in best-case topology
        baseline = self._compute_island_upper_bound()

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

    # ── Reunification reward constants (see compute_global_connectivity_reward) ──
    #
    # REPRICED.  The relay/reunification block was measured (per-head
    # isolation, Scenario A seed 42) to be reward-POSITIVE while KPI-NEGATIVE
    # when the relay head was freed on its own: +6.94 reward for -7.0 pp of
    # UE connectivity.  Relaying that neither carries traffic nor removes a
    # fragment must not out-earn delivering traffic, so the rates came down.
    # The two REUNIFY payouts are kept (they are strictly conditional on a
    # GENUINE fragment reduction, measured against the relay-free partition,
    # which is real recovery work) but scaled so the whole block can no
    # longer rival the delivery block: full reunification of the Scenario-A
    # 4-way partition now pays 3 x 6 + 25 = +43 against the delivery block's
    # +135, where it used to pay 3 x 10 + 40 = +70.
    REUNIFY_FULL_BONUS    = _RP.reunify_full_bonus   # 25.0 (was 40.0)
    REUNIFY_PER_FRAGMENT  = _RP.reunify_per_fragment # 6.0 per fragment eliminated (was 10.0)
    REUNIFY_RESIDUAL      =  2.0   # per component still remaining
    REUNIFY_REF_FRAGMENTS =  4     # Scenario-A design partition (FRAG_MAX_FRAGMENTS)
    REUNIFY_EVAL_EVERY    =  5     # ticks between component recomputations
    BRIDGE_LOSS_PENALTY   = 15.0   # per cross-fragment bridge torn down
    RELAY_CHURN_PENALTY   =  1.0   # per agent whose relay_mode changed
    # Relay activity rates — both now gated on DELIVERED UE traffic, not on
    # raw link utilisation (which also carried the node's own telemetry).
    # RELAY_CONTRIB_PER_NODE: REMOVED (2.0 -> 0.5 -> 0.0).  A FLAT payment
    # for "being a relay that carried something" cannot tell a relay moving
    # 1 Mbps from one moving 500 Mbps, so it is the one relay term that is
    # not proportional to what the relaying achieved.  At 0.5 it still left
    # freeing the relay head marginally reward-POSITIVE (+0.19) while it was
    # KPI-negative (-7.1 pp) when both pricings were scored on an identical
    # trajectory.  Relaying is now paid ONLY in proportion to the delivered
    # UE traffic it carries (RELAY_TPUT_CAP_GLOBAL below) or to a genuine
    # fragment elimination (the REUNIFY terms above).
    RELAY_CONTRIB_PER_NODE  = _RP.relay_contrib_per_node # 0.0 (was 2.0)
    RELAY_TPUT_CAP_GLOBAL   = _RP.relay_tput_cap_global  # 5.0 (was 15.0)

    def bridging_relay_link_ids(self) -> set:
        """Ids of active relay links whose endpoints sit in DIFFERENT
        components of the relay-free infra graph — i.e. the links currently
        doing reunification work (as opposed to LOCAL_REROUTE links that
        merely parallel an existing intra-fragment hop).

        RAW set: two relay links joining the SAME pair of fragments both
        appear here.  Use distinct_bridge_link_ids() for anything that counts
        bridges or decides teardown protection.
        """
        try:
            relay_links = self.active_relay_links()
            if not relay_links:
                return set()
            comp = self._relay_free_components()
            return {l.id for l in relay_links
                    if l.endpoints[0] in comp and l.endpoints[1] in comp
                    and comp[l.endpoints[0]] != comp[l.endpoints[1]]}
        except Exception:
            return set()

    def distinct_bridge_link_ids(self) -> set:
        """De-duplicated bridge set: one link id per INDEPENDENT fragment
        merge.

        Why this exists.  The raw bridging set counts LINKS, and several
        distinct links can perform the SAME merge, which inflated the
        reported bridge count above what the topology can even support (4
        "bridges" were reported on a topology with only 3 MultiHaul sites).
        Two mechanisms produce the duplicates:
          * the far endpoint of a bridge re-pointing back across the same cut
            (the Priority-2 branch in _step_phy_mac used to be able to do
            this — it is now blocked at source);
          * three or more relay links forming a cycle over the fragments, in
            which one of them adds no connectivity at all.

        The number that is physically meaningful is the number of merges,
        i.e. the number of TREE EDGES in the forest that the relay links
        induce over the raw fragments — exactly `raw_components -
        components_with_relays`.  This returns the link ids of those tree
        edges (union-find over link ids in sorted order, so the choice of
        representative is deterministic and stable across ticks).
        """
        try:
            relay_links = self.active_relay_links()
            if not relay_links:
                return set()
            # One relay-free component pass shared by the cross-component
            # filter and the union-find below (this runs on every relay-mode
            # flip, so it must not cost two graph traversals).
            comp = self._relay_free_components()
            bridging = {l.id for l in relay_links
                        if l.endpoints[0] in comp and l.endpoints[1] in comp
                        and comp[l.endpoints[0]] != comp[l.endpoints[1]]}
            if not bridging:
                return set()
            parent = {}

            def _find(x):
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            keep = set()
            for lid in sorted(bridging):
                link = self.topology.links.get(lid)
                if link is None:
                    continue
                ra, rb = _find(comp[link.endpoints[0]]), _find(comp[link.endpoints[1]])
                if ra == rb:
                    continue          # redundant: that merge already happened
                parent[ra] = rb
                keep.add(lid)
            return keep
        except Exception:
            return set()

    def compute_global_connectivity_reward(self) -> float:
        """
        Phase-aware shaped global reward — designed for learning signal clarity.

        The primary signal is VOLUME-WEIGHTED: it pays for the fraction of
        offered UE-to-UE traffic volume actually delivered this tick
        (self._ue_to_ue_delivered_fraction from _forward_traffic), which is
        exactly the steady-state achievability metric the deployment
        comparison evaluates.  The old binary routed fraction (a flow counts
        once >=50% of its volume got through) is kept only as a diagnostic in
        self._last_binary_routed_fraction — it gave no gradient toward
        delivering FULL volume and let the policy plateau at partial delivery.

        Island mode (Core severed)
        --------------------------
          +100.0 * volume-weighted UE-to-UE delivered fraction
                 — primary learning signal: delivered/offered volume over all
                   UE-to-UE flows (what the achievability comparison measures)
          + 25.0 full-delivery bonus — if delivered fraction >= 0.95
                 (FULL_DELIVERY_THRESHOLD; volume-based, replaces the old
                 all-flows-routed binary trigger)
          + 10.0 * general delivered fraction — delivered/offered volume of
                 general (non-UE-to-UE) UE traffic.  General classes are
                 admission-gated by prb_general_frac (see
                 _derive_admission_policies); without this term nothing in the
                 reward values general throughput and the policy starves
                 general PRB even on a healthy network.
          (no flat per-relay payment: RELAY_CONTRIB_PER_NODE is 0.0 — see
                 the REPRICING block below)
          -  1.0 per agent whose relay_mode changed since the previous tick
                 (RELAY_CHURN_PENALTY) — discourages relay flapping.
          + up to 5.0 relay-throughput bonus (RELAY_TPUT_CAP_GLOBAL) — 1 pt
                 per 50 Mbps of DELIVERED UE traffic carried on transport
                 relay links
          +  5.0 * recovery_ramp            — ramp once restore_core fires
                 (+5.0 one-shot completion bonus)

        Island reunification (the research objective) — NEW
        ---------------------------------------------------
          + 25.0 REUNIFY_FULL_BONUS    — island is ONE connected component
                 and the raw (relay-free) partition had more than one, i.e.
                 the policy's own bridges did the reunifying.  Scaled by
                 (raw - 1) / (REUNIFY_REF_FRAGMENTS - 1), capped at 1.0, so
                 the bonus is proportionate to the work the scenario demanded
          +  6.0 REUNIFY_PER_FRAGMENT  per fragment ELIMINATED relative to the
                 raw partition — graded, so 4->3->2->1 each pays
          -  2.0 REUNIFY_RESIDUAL      per component still remaining
          - 15.0 BRIDGE_LOSS_PENALTY   per cross-fragment bridge torn down
                 since the previous tick (sustained bridging)

        Stranded-user REACHABILITY (life-safety objective) — NEW
        --------------------------------------------------------
          + REACH_RESTORED_PER_UE (0.10) per surviving UE that moved from
                 "cannot reach any peer outside my island" into the connected
                 BODY of the survivor network AND carried traffic this tick,
                 capped at REACH_RESTORED_CAP (5.0).  Measured by
                 peer_reachable_ue_stats as
                     restored = reachable_op - reachable_raw
                 i.e. a DELTA against the relay-free graph, so it is zero for
                 any arm that forms no bridges and zero for a relay link that
                 merely parallels an existing intra-component hop.
                 REPRICED from 1.0/60.0 and made traffic-conditional — see the
                 REACH_RESTORED_PER_UE / REACH_REQUIRE_TRAFFIC constants for
                 the measurement that forced it and the resulting arithmetic.

        WHY this term had to be added.  Every other positive term in this
        reward is paid PER BYTE.  Measured on the 10-seed Scenario-A run
        (output/multiseed_data.pkl): achieved components == raw components -
        bridges exactly, so fragment reduction is purely a function of how
        many bridges form, and every seed's residual fragment was a SINGLETON
        or near-singleton (raw fragment sizes 21/11/2/1 infra nodes).  A
        volume-maximising policy correctly DECLINES to bridge a 1-node island:
        the island's traffic is ~1/35 of the offered volume, so the delivery
        block pays at most ~+3 for restoring it, while the bridge costs relay
        PRB, a churn flip and a beam-training outage.  That trade is right for
        a throughput objective and WRONG for disaster recovery, where a
        stranded island is somebody's only link and its value is not its
        throughput.  The policy was also not even reliably making the
        throughput-optimal call: on seed 58 under-bridging cost it delivery
        too (52.1 % achievability, its worst seed, below SDN's 53.1 %).

        MARGINAL TRADE — what a bridge costs vs what it now earns.
        Take the worst case for bridging: a 1-node island, ~2 % of the
        surviving UE population (~7 UEs of ~350).

          COST of relaying, in reward points
            access capacity   0.0   after the PRB-pool separation (see
                                    PHYMACState.normalise_prb).  This was by
                                    far the largest cost and it was an
                                    ARTEFACT: the old model normalised
                                    emergency + relay + general to 1.0, so
                                    prb_relay_fraction was subtracted from the
                                    Uu access pool.  Measured on the Scenario-A
                                    topology, that cost EVERY node in a relay
                                    mode ~19-20 % of its own access capacity
                                    (agent sample e=.45/r=.16/g=.39:
                                    50.4 -> 60.0 Mbps; a relay-heavy
                                    r=.40 sample: 36.0 -> 60.0 Mbps), and the
                                    node count is not 3 but the whole fleet:
                                    the MARL arms run 14-35 of 42 nodes in a
                                    relay mode, and the SDN/OSPF/OLSR
                                    baselines lock ALL 42 to LOCAL_REROUTE.
                                    So the conflation was suppressing ~20 % of
                                    fleet access capacity — up to about -20
                                    points off the 100 x delivered_frac term —
                                    on every arm that relays at all.  A 60 GHz
                                    MultiHaul TG bridge does not share
                                    spectrum with a 3.5 GHz Uu cell, so none
                                    of that cost was real.
            relay churn      -1.0   RELAY_CHURN_PENALTY, one mode flip
            beam training    -0.0 to -0.5  RELAY_REPOINT_TICKS (3) ticks in
                                    which the new link carries nothing; the
                                    site's pre-existing links are untouched.
            transport budget        still real, and unchanged: relay capacity
                                    is still prb_relay_fraction x the transport
                                    link budget, and prb_relay_fraction is
                                    bit-identical before and after the pool
                                    separation — so the fix removes a cost but
                                    grants NO extra bridging ability, to any arm.
            total after the fix  ~ -1.0 to -1.5

          EARNED by the bridge, in reward points
            delivered volume  + the island's traffic share x 100.  For a
                                    1-node island this is the ~+2 to +3 that
                                    the old reward was weighing against a
                                    much larger apparent PRB cost.
            reunification    +10.0   REUNIFY_PER_FRAGMENT, one fragment gone,
                                    but only re-evaluated every
                                    REUNIFY_EVAL_EVERY (5) ticks
            reachability     +1.0 per restored USER, EVERY tick   <- NEW

        SUPERSEDED SIZING RATIONALE (kept because the numbers below are the
        ones that were wrong, and the record of why matters).  This paragraph
        argued for REACH_RESTORED_PER_UE = 1.0 and REACH_RESTORED_CAP = 60.0.
        Both were cut — to 0.10 and 5.0, with a traffic condition added — after
        the term was measured to be the dominant reward-positive / KPI-negative
        driver: the policy trained against it reached 29.2 % UE connectivity
        against 30.1 % for doing nothing.  The reasoning below is sound about
        WHY the term should exist and wrong about HOW MUCH it should pay; the
        error is that it priced a stranded user's reachability ABOVE the
        delivered volume the evaluation actually scores, so a policy could farm
        it by over-relaying.  See the REACH_RESTORED_PER_UE constant for the
        replacement arithmetic.

        Original text: Sizing rationale for REACH_RESTORED_PER_UE = 1.0: one restored user is
        worth exactly one percentage point of delivered volume.  On the
        Scenario-A topology (42 infra nodes, 150 UEs) a UE's fair share of the
        +100 volume term is 100/150 = 0.67 points, so this prices a stranded
        user's reachability at ~1.5x the throughput that user would contribute,
        and it is paid on EVERY tick the user stays connected rather than only
        when the fragment count is re-evaluated.  That is a deliberate,
        stated life-safety premium rather than an accident of the volume
        weights.  REACH_RESTORED_CAP (60.0) binds once a bridge restores more
        than 60 of the 150 UEs; it keeps the term from swamping the volume
        signal and keeps the reward inside the range REWARD_SCALE assumes.

        LIMIT OF THIS TERM — stated because it is measured, not assumed.  The
        term prices USERS, so it is silent on a residual fragment that holds
        none.  Measured on Scenario A seed 50 (42 infra nodes, 150 UEs, raw
        partition 4 components):

            achieved comps   4      3      2      1
            stranded UEs    62     62      0      0
            reach %       58.7   58.7  100.0  100.0
            reward term      0      0    +60    +60   (cap binds at 60)

        So going 4 -> 2 components restores 62 of the 150 UEs (41 % of the
        population) and the term pays its capped +60 for HOLDING those bridges
        on every tick.  Going 2 -> 1 restores ZERO users on this seed: the last
        residual fragment is UE-free infrastructure (a bare relay / O-DU site).
        `attached_outside_body` in peer_reachable_ue_stats is the number that
        says which case a seed is in, and it must be read next to the component
        count before concluding anything about the last fragment.

        This is why the three fixes are complementary rather than redundant:
          * this reachability term prices HOLDING the bridges that carry users
            — the 4 -> 2 work, which is where 41 % of the population lives;
          * the PRB-pool separation is what makes the LAST, UE-free fragment
            worth its +10 REUNIFY_PER_FRAGMENT, by removing the ~20 % access
            capacity cost that used to offset it;
          * the deterministic bridge-target tie-break in
            _rank_bridge_candidates is what lets all three MultiHaul sites
            actually pick DISTINCT targets, without which the third merge is
            not reachable at all.

        Term balance (why these numbers)  — REPRICED, see below
        --------------------------------------------------------
        Scenario A severs the island into 4 raw fragments, so the
        reunification block spans [-6 (4 fragments, nothing bridged),
        +43 (1 component: 3 x 6 eliminated + 25 bonus, 0 residual)].
        The delivery block spans [0, +135] (100 x delivered_frac + 25
        full-delivery + 10 x general).  Therefore:
          * FULL reunification with ZERO traffic delivered  =>  +43
          * ZERO reunification with FULL traffic delivered   => +135
        so a bridge that carries nothing can never out-earn serving traffic —
        the volume term stays dominant, as required.  At the margin the two
        remain comparable and both learnable: +6 per fragment eliminated is
        worth a 6-percentage-point gain in delivered fraction, which is the
        right order for one restored island's traffic.  This is intentional —
        bridging is only valuable because it lets traffic flow, and the
        graded term is the credit-assignment shortcut that makes that
        discoverable before the traffic actually appears.
        The term this replaced was -0.5 per extra fragment (-1.5 at 4
        fragments, ~1 % of the volume term): no usable gradient at all.

        REPRICING — the relay/reunification block was OVER-PAID, measured
        ------------------------------------------------------------------
        The bound above ("a bridge that carries nothing can never out-earn
        serving traffic") was a statement about the EXTREMES, and it was
        true.  It was not true AT THE MARGIN, which is where a policy
        actually learns.  Measured with the per-head isolation harness
        (Simulator.pinned_action_heads: pin every action head to its
        do-nothing default, then free exactly one; Scenario A seed 42, 400
        ticks), freeing ONLY the relay head moved the global reward UP by
        +6.94 while it moved UE connectivity DOWN by 7.0 percentage points.
        Random relaying was reward-positive and KPI-negative — a
        reward-hacking surface — because three terms paid for relaying more
        or less regardless of what the relaying achieved:

          1. "+0.5 per routed flow, capped at 20, whenever ANY relay
             contributed" — REMOVED (rate -> 0).  The worst of the three: a
             cliff worth up to +20 that unlocked the instant one relay
             anywhere carried a byte, but whose SIZE was routed_flows, i.e.
             delivery the do-nothing policy was already achieving with no
             relays at all.  It double-paid for delivery the +100 term
             already prices, and mis-attributed it to relaying.
          2. "+2.0 per contributing relay node" -> +0.5.  With 14-20 nodes
             in a relay mode this was +28 to +40 per tick for the fact of
             relaying — comparable to a 30-point swing in delivered volume.
          3. relay-throughput cap 15.0 -> 5.0.

        Terms 2 and 3 are also now STRICTLY CONDITIONAL on delivered UE
        traffic: they read the per-link ledger _link_carried_ue built by
        _forward_traffic, not link.current_utilization, which also contained
        infrastructure telemetry the delivered fractions never counted.  A
        relay that carries only its own site's O&M traffic now earns zero.

        What was deliberately NOT cut: the REUNIFY payouts and the
        reachability term.  Both are already strictly conditional on a
        genuine outcome — a fragment eliminated relative to the RELAY-FREE
        partition, and a UE moved into the connected body — so neither can
        be earned by relaying that achieves nothing.  They were scaled down
        (40 -> 25, 10 -> 6) only to keep the whole block below the delivery
        block, not because they were unconditional.

        No new bonus was introduced anywhere in this repricing, no metric
        changed, and no arm's physical behaviour changed: every edit is
        either a constant or a switch of which already-computed byte counter
        a term reads.

        Behaviour on an UNFRAGMENTED network (Scenario B)
        ------------------------------------------------
        Scenario B severs the core but cuts no links, so most seeds sit at
        raw == 1.  Then eliminated == 0, residual == 0 (op == 1) and the
        completion bonus is gated on raw > 1 — the whole block evaluates to
        EXACTLY 0.0 and cannot distort Scenario B.  On the Scenario-B seeds
        where a single node is orphaned (raw == 2) the block is worth -2.0
        while the orphan stays cut off and +10 + 40/3 = +23.3 if it is
        reconnected: active, but scaled down to the size of the problem.

        Normal mode (pre-severance)
        ---------------------------
          + 3.0 * volume-weighted UE-to-UE delivered fraction
          + 0.01 per survivor node with spare PRB capacity  — baseline signal

        Expected magnitude: island-mode reward stays in roughly [-60, 250]
        (typically 0-220).  The capped reachability term contributes at most
        +5.0 to that upper bound (it was +60 before the repricing), so
        REWARD_SCALE=100.0 in
        mappo_trainer.py still keeps the scaled return inside ~[-0.6, 3.1]
        and the 0.4 local / 0.6 global mix in worker.py remains sane.
        NOTE: the reward changed => a RETRAIN is required; checkpoints
        trained against the old reward are not comparable.  The PRB-pool
        separation makes this doubly true: it raises access_capacity_mbps by
        ~20 % at every node in a relay mode, which lowers ps.prb_utilization by
        a similar amount — and prb_utilization is an OBSERVATION field
        (PHYMACState.to_block_a), so old checkpoints are off-distribution as
        well as mis-incentivised.  The action/observation SHAPES are unchanged,
        so checkpoints still load; their behaviour is simply no longer
        meaningful.
        """
        from .phy_mac_state import RelayMode
        FULL_DELIVERY_THRESHOLD = 0.95   # volume fraction for the +25 bonus
        reward = 0.0

        total_flows  = max(1, len(getattr(self, 'ue_to_ue_flows', [])))
        routed_flows = getattr(self, 'ue_to_ue_success_count', 0)
        # Binary routed fraction — DIAGNOSTIC ONLY (no longer the reward driver)
        frac = min(1.0, routed_flows / total_flows)
        self._last_binary_routed_fraction = frac

        # Volume-weighted delivered fraction (primary signal, computed in
        # _forward_traffic as delivered/offered UE-to-UE volume this tick)
        delivered_frac = max(0.0, min(1.0, float(
            getattr(self, '_ue_to_ue_delivered_fraction', 0.0))))

        # General (non-UE-to-UE) delivered volume fraction this tick
        gen_offered = float(getattr(self, 'last_tick_ue_general_offered', 0.0))
        gen_frac = (
            max(0.0, min(1.0,
                float(getattr(self, 'last_tick_ue_general_volume', 0.0))
                / gen_offered))
            if gen_offered > 0 else 0.0)

        # ── Relay churn tracking (always updated, penalised in island mode) ──
        # Compare each agent's relay_mode to its value at the previous call
        # (one call per tick in every training/eval loop).  The snapshot dict
        # is created empty in __init__, so a fresh Simulator (one per episode
        # in worker.py/main.py) starts with no penalty on its first tick.
        churn_count = 0
        _current_modes = {}
        for _nid, _ps in self.phy_mac_states.items():
            _mode = getattr(_ps, 'relay_mode', RelayMode.OFF)
            _current_modes[_nid] = _mode
            _prev = self._prev_relay_modes.get(_nid)
            if _prev is not None and _prev != _mode:
                churn_count += 1
        self._prev_relay_modes = _current_modes

        if self.island_mode:
            # ── Primary connectivity signal (volume-weighted) ─────────────────
            reward += 100.0 * delivered_frac

            if delivered_frac >= FULL_DELIVERY_THRESHOLD:
                reward += 25.0  # Full-delivery bonus (>=95% of offered volume)

            # ── General-traffic throughput (healthy-network value) ────────────
            reward += 10.0 * gen_frac

            # ── Relay churn penalty ───────────────────────────────────────────
            # Raised 0.5 -> RELAY_CHURN_PENALTY (1.0) now that a sustained
            # bridge is worth up to +70: a mode flip that risks a bridge must
            # not look like a rounding error next to it.  Fleet-wide per-tick
            # flapping (~40 agents) costs ~-40, clearly negative; a couple of
            # agents re-evaluating costs ~-2, cheap enough to keep exploring.
            reward -= self.RELAY_CHURN_PENALTY * churn_count

            # ── Relay-link activity: which nodes' relay links carried traffic ──
            # STRICTLY CONDITIONAL ON DELIVERED UE TRAFFIC.  This used to read
            # link.current_utilization, which also carries infrastructure
            # telemetry/O&M — traffic the delivered fractions above do NOT
            # count.  A relay could therefore look "contributing" while
            # moving nothing the objective pays for.  _link_carried_ue is the
            # per-link ledger of delivered UE volume built by _forward_traffic
            # (same bytes, same routing, no new physics).
            _carried_ue = getattr(self, '_link_carried_ue', {})
            # _RP.relay_terms_use_delivered_ledger is TRUE in every shipped
            # profile; the raw-utilisation branch exists only so the ablation
            # can reproduce the pre-repricing configuration exactly.
            _use_ledger = _RP.relay_terms_use_delivered_ledger
            relay_throughput = 0.0
            contributing_endpoints = set()
            for lid, link in self.topology.links.items():
                if (getattr(link, 'link_type', None) != LinkType.TRANSPORT_RELAY
                        or not link.is_up):
                    continue
                _vol = (_carried_ue.get(lid, 0.0) if _use_ledger
                        else float(getattr(link, 'current_utilization', 0.0)))
                if _vol > 0:
                    relay_throughput += _vol
                    contributing_endpoints.add(link.endpoints[0])
                    contributing_endpoints.add(link.endpoints[1])

            # ── Relay reward: pay ONLY relays that actually contributed ────────
            # (relay mode on AND their relay link carried nonzero traffic this
            # tick).  Prevents 'everyone toggles relay on' reward farming.
            relay_count = 0
            for nid, ps in self.phy_mac_states.items():
                node = self.topology.nodes.get(nid)
                if node is None or not node.is_survivor:
                    continue
                ntype_val = getattr(getattr(node, 'node_type', None), 'value', '')
                if ntype_val not in ('Relay', 'gNB-Site', 'O-DU', 'O-RU'):
                    continue
                if (getattr(ps, 'relay_mode', RelayMode.OFF) in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST)
                        and nid in contributing_endpoints):
                    relay_count += 1

            # REPRICED 2.0 -> 0.5 -> RELAY_CONTRIB_PER_NODE (0.0).  See the
            # REPRICING block of this method's docstring: at 2.0, with 14-20
            # nodes in a relay mode, this term alone was worth +28 to +40 per
            # tick — comparable to a 30-percentage-point swing in delivered
            # volume — for the mere fact of relaying.  At 0.5 it still left
            # the relay head marginally reward-positive while it was
            # KPI-negative, so the flat rate went to zero outright.
            reward += self.RELAY_CONTRIB_PER_NODE * relay_count

            # REMOVED — "+0.5 per routed flow whenever ANY relay contributed"
            # (formerly `if routed_flows > 0 and relay_count > 0:
            #            reward += min(20.0, routed_flows * 0.5)`).
            # Rate lowered to zero, and this is the single worst term of the
            # three.  It was a CLIFF: the instant one relay anywhere in the
            # fleet carried a byte, up to +20 unlocked — but the +20 was a
            # function of routed_flows, i.e. of delivery the do-nothing
            # policy was already producing without any relay.  It therefore
            # (a) paid a second time for delivery the +100 delivered-fraction
            # term above already pays for, and (b) attributed that payment to
            # relaying, which had not caused it.  That is exactly the
            # "relaying out-earns delivering" inversion, in its purest form.
            #
            # The term is reinstated ONLY under the `prev` / `deliv` ablation
            # profiles, whose sole purpose is to reproduce the pre-repricing
            # configuration for attribution.  Every shipped profile has
            # _RP.routed_flow_cliff == False, so this branch is dead code in
            # production — see sixg_sim/reward_profile.py.
            if (_RP.routed_flow_cliff and routed_flows > 0
                    and relay_count > 0):
                reward += min(20.0, routed_flows * 0.5)

            # Bonus for relay-routed throughput: rewards relay paths that
            # carry DELIVERED UE data (see _carried_ue above).  Cap REPRICED
            # 15.0 -> RELAY_TPUT_CAP_GLOBAL (5.0) so that a relay carrying
            # traffic is still worth more than one that is not, but cannot
            # rival the delivery block.
            if relay_throughput > 0:
                reward += min(self.RELAY_TPUT_CAP_GLOBAL,
                              relay_throughput / 50.0)  # 1 pt per 50 Mbps

            # ── Recovery ramp ─────────────────────────────────────────────────
            if self._recovery_started_tick >= 0 and not self._recovery_complete:
                ramp = min(1.0, self._ticks_since_recovery / 200.0)
                reward += 5.0 * ramp
                if delivered_frac >= 0.95 and ramp >= 0.5:
                    self._recovery_complete = True
                    reward += 5.0   # one-shot completion bonus

            # ── ISLAND REUNIFICATION (explicit research objective) ────────────
            #
            # The research goal is that the policy bridges the post-severance
            # islands back toward a SINGLE connected component.  The old term
            # (-0.5 per extra fragment, i.e. -1.5 at 4 fragments) was ~1.5% of
            # the +100 volume term, so it produced essentially no gradient and
            # the policy had no reason to prefer a cross-fragment bridge over
            # an intra-fragment capacity relay.  It is replaced by:
            #
            #   +REUNIFY_FULL_BONUS (40) when the island is ONE component and
            #        the raw partition had more than one — a strong, explicit
            #        preference for full reunification.
            #   +REUNIFY_PER_FRAGMENT (10) per fragment ELIMINATED relative to
            #        the raw (relay-free) partition, so partial progress is
            #        rewarded and the policy can climb 4 -> 3 -> 2 -> 1.
            #   -REUNIFY_RESIDUAL (2) per component still remaining, keeping
            #        "fewer is better" monotone even if a node failure creates
            #        a fragment the baseline never had.
            #
            # Computed on the SAME primitives the comparison harness reports
            # (count_infra_components), so reward and metric cannot drift.
            # Evaluated every REUNIFY_EVAL_EVERY ticks and held between
            # evaluations (component counting is O(N+L)).
            if (self.current_tick % self.REUNIFY_EVAL_EVERY == 0
                    or not hasattr(self, '_last_reunify_reward')):
                frag_op  = self.count_infra_components(include_relay_links=True)
                frag_raw = self.count_infra_components(include_relay_links=False)
                if self._post_severance_fragment_baseline is None:
                    self._post_severance_fragment_baseline = frag_raw
                eliminated = max(0, frag_raw - frag_op)
                r_re = self.REUNIFY_PER_FRAGMENT * eliminated
                r_re -= self.REUNIFY_RESIDUAL * max(0, frag_op - 1)
                if frag_op == 1 and frag_raw > 1:
                    # Scale the completion bonus by how much reunification
                    # work the scenario actually demanded, so the term is
                    # PROPORTIONATE across scenarios instead of paying a flat
                    # +40 for closing a single orphan.  At the Scenario-A
                    # design partition (REUNIFY_REF_FRAGMENTS = 4) the factor
                    # is 1.0 and the bonus is the full +40.
                    scale = min(1.0, (frag_raw - 1)
                                / float(max(1, self.REUNIFY_REF_FRAGMENTS - 1)))
                    r_re += self.REUNIFY_FULL_BONUS * scale
                self._last_reunify_reward = r_re
                self._last_fragment_count     = frag_op
                self._last_fragment_count_raw = frag_raw
            reward += getattr(self, '_last_reunify_reward', 0.0)

            # ── Bridge-loss penalty (sustained bridging) ──────────────────────
            # Tearing down a link that was the ONLY path between two fragments
            # costs more than the churn penalty alone: it directly undoes the
            # reunification term above.  Together with the (now physically
            # motivated) teardown protection in _step_phy_mac and the
            # RELAY_REPOINT_TICKS beam-training outage, this makes beam
            # thrashing strictly unprofitable.
            #
            # Counted over the DE-DUPLICATED bridge set, so dropping a
            # redundant parallel relay link — which removes no connectivity —
            # is not punished as if a fragment had been cut loose.
            _bridges_now = self.distinct_bridge_link_ids()
            _lost = len(getattr(self, '_prev_bridge_link_ids', set()) - _bridges_now)
            reward -= self.BRIDGE_LOSS_PENALTY * _lost
            self._prev_bridge_link_ids = _bridges_now

            # ── STRANDED-USER REACHABILITY ────────────────────────────────────
            # Life-safety objective: pay per USER re-joined to the connected
            # body of the survivor network, independent of the volume that
            # user then carries.  This is what makes bridging a singleton
            # island rational; see the REACHABILITY block of this docstring for
            # the marginal-trade arithmetic.
            #
            # Read from the per-tick stats computed in
            # _update_phy_mac_observations (which reuses the UE->O-RU map it
            # already built), so this adds no graph work to the reward path.
            # `restored` is a delta against the RELAY-FREE graph, so it is 0
            # unless the arm's own relay links moved users into the body.
            #
            # Timing: every training loop calls _build_agent_observations()
            # before this method, so the value is from THIS tick — but it is
            # sampled pre-action, i.e. it is the reachability the agents
            # actually observed when they chose, whereas the REUNIFY block
            # above recounts components post-action.  On the single tick a
            # bridge comes up or goes down the two therefore differ by one
            # tick.  That is deliberate: `restored` is a SUSTAINED quantity
            # paid every tick the users stay connected, so a one-tick lag is
            # immaterial, and crediting the state the policy conditioned on is
            # the better attribution.
            _restored = int(getattr(self, '_reach_restored_ue_count', 0))
            # CONDITIONALITY (see REACH_REQUIRE_TRAFFIC): count only the
            # restored users that actually carried traffic this tick, so the
            # term moves with the quantity the evaluation scores.  The rate
            # stays FLAT per user — a restored user on a quiet island is
            # worth exactly what a restored user on a busy one is worth,
            # which is the life-safety premium the term exists to express.
            _restored_paid = _restored
            if self.REACH_REQUIRE_TRAFFIC:
                _ids = getattr(self, '_reach_restored_ue_ids', None)
                _delivering = getattr(self, '_delivered_ue_endpoints', None)
                if _ids and _delivering:
                    _restored_paid = len(_ids & _delivering)
                else:
                    _restored_paid = 0
            _r_reach = (min(self.REACH_RESTORED_CAP,
                            self.REACH_RESTORED_PER_UE * _restored_paid)
                        if _restored_paid > 0 else 0.0)
            reward += _r_reach

            # ── Term-by-term diagnostic (no behaviour, no metric) ──────────
            # Written so the relay-vs-delivery balance can be audited
            # directly instead of inferred from the total.  Read by the
            # per-head isolation harness; nothing in the training or
            # evaluation path consumes it.
            self._reward_terms = {
                'delivery':      100.0 * delivered_frac,
                'full_delivery': 25.0 if delivered_frac >= FULL_DELIVERY_THRESHOLD else 0.0,
                'general':       10.0 * gen_frac,
                'relay_churn':   -self.RELAY_CHURN_PENALTY * churn_count,
                'relay_count':   relay_count,
                'relay_contrib': self.RELAY_CONTRIB_PER_NODE * relay_count,
                'relay_tput_raw': relay_throughput,
                'routed_flows':  routed_flows,
                'relay_tput':    (min(self.RELAY_TPUT_CAP_GLOBAL,
                                      relay_throughput / 50.0)
                                  if relay_throughput > 0 else 0.0),
                'reunify':       getattr(self, '_last_reunify_reward', 0.0),
                'bridge_loss':   -self.BRIDGE_LOSS_PENALTY * _lost,
                'reach':         _r_reach,
                'reach_restored':      _restored,
                'reach_restored_paid': _restored_paid,
            }

        else:
            # ── Normal-mode reward (volume-weighted, small scale by design) ───
            reward += 3.0 * delivered_frac

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
            transport_relay_link_count int  ACTIVE transport relay LINKS
                             (the physically comparable "relay" quantity)
            transport_relay_bridge_count int  subset of the above that joins
                             two raw fragments (true reunification bridges)
            relay_mode_node_count int  nodes whose relay_mode is
                             LOCAL_REROUTE/CAPACITY_BOOST — an INTENT, not a
                             link.  Deliberately a different key name.
            transport_relay_count  int  DEPRECATED alias of
                             transport_relay_link_count.  It used to mean
                             nodes-in-relay-mode, which made OSPF/OLSR report
                             42 "relays" while holding zero relay links.
            transport_link_count   int  active transport wireless backhaul links
                             (TransportRelayModel bookkeeping view)
            island_node_count int   surviving infra nodes (non-Core) in island
            iops_admitted    int    cumulative IOPS registrations
            island_fragments int    infra components INCLUDING relay links
                             (1 = fully reunified)
            island_fragments_raw int  infra components EXCLUDING relay links
                             (the post-severance partition baseline)
            ticks_since_sev  int    0 before severance
        """
        relay_links       = self.active_relay_links()
        relay_link_count  = len(relay_links)
        relay_mode_nodes  = self.count_relay_mode_nodes()
        bridge_count      = self.count_bridging_relay_links()
        transport_link_count = len(getattr(self.transport_relay_model, 'active_links', {}))

        island_node_count = sum(
            1 for nid, n in self.topology.nodes.items()
            if n.is_survivor and
               getattr(n, 'node_type', None) is not None and
               n.node_type.value not in {'UE', 'Core', 'CoreUPF', 'UPF'}
        )

        return {
            'ue_conn_frac':     getattr(self, '_ue_pair_routed_fraction', 0.0),
            'transport_relay_link_count':   relay_link_count,
            'transport_relay_bridge_count': bridge_count,
            'relay_mode_node_count':        relay_mode_nodes,
            # Back-compat key — now LINKS, not nodes-in-mode (see docstring)
            'transport_relay_count':  relay_link_count,
            'transport_link_count':   transport_link_count,
            'island_node_count': island_node_count,
            'iops_admitted':    self.iops_manager.total_admitted,
            'island_fragments': (self.count_infra_components(True)
                                 if self.island_mode else 1),
            'island_fragments_raw': (self.count_infra_components(False)
                                     if self.island_mode else 1),
            'ticks_since_sev':  getattr(self, '_ticks_since_severance', 0),
        }

    # ── Fragment / relay metric primitives ────────────────────────────────
    #
    # SINGLE SOURCE OF TRUTH for "how fragmented is the island" and "how many
    # transport relay bridges exist".  Every consumer (reward shaping,
    # get_island_kpis, the timeline comparison's per-tick record) must go
    # through these so that all arms report the SAME physical quantity.
    #
    # Definitions (deliberately explicit — these were previously ambiguous):
    #   INFRA SCOPE   surviving nodes whose node_type is not a core function
    #                 ({Core, UPF, AMF, CoreUPF}) and not a UE.  UEs are
    #                 excluded from the node set AND from traversal: a UE with
    #                 Uu links into two different infra fragments is NOT a
    #                 transport element and must not merge them (the same bug
    #                 that was fixed once in OSPFController._compute_fragments).
    #   FRAGMENTS     connected components of that infra scope over links that
    #                 are is_up, INCLUDING active TRANSPORT_RELAY links, so a
    #                 relay bridge genuinely reduces the count.
    #   RAW FRAGMENTS same, but with TRANSPORT_RELAY links removed — the
    #                 post-severance physical partition, i.e. the baseline a
    #                 bridging policy is measured against.

    #: node_type.value strings that are NOT part of the surviving island infra
    _NON_INFRA_NODE_TYPES = {"Core", "UPF", "AMF", "CoreUPF", "UE"}

    def _infra_scope_nodes(self) -> set:
        """Surviving infrastructure nodes (no core functions, no UEs)."""
        return {
            nid for nid, n in self.topology.nodes.items()
            if getattr(n, 'is_survivor', False)
            and getattr(getattr(n, 'node_type', None), 'value', 'UE')
            not in self._NON_INFRA_NODE_TYPES
        }

    def count_infra_components(self, include_relay_links: bool = True) -> int:
        """Connected components of the surviving infrastructure graph.

        Args:
            include_relay_links: if True (default) active TRANSPORT_RELAY
                links count as edges — this is the OPERATIONAL fragment count
                and it drops when the policy bridges islands.  If False, relay
                links are ignored — this is the RAW post-severance partition
                (the ceiling/baseline the policy is judged against).

        UE nodes are excluded from both the node set and from traversal, so
        Uu links can never merge two infra fragments.

        Delegates to `_infra_component_labels` so the fragment COUNT and the
        per-node component LABELS used by the bridge and reachability terms
        can never disagree.
        """
        try:
            labels = self._infra_component_labels(include_relay_links)
            if not labels:
                return 1
            return len(set(labels.values()))
        except Exception:
            return 1

    def active_relay_links(self) -> list:
        """Active TRANSPORT_RELAY links present in the topology (is_up).

        This is the physically meaningful "relay" quantity: a LINK that exists
        and can carry transport traffic.  It is NOT the number of nodes whose
        relay_mode happens to be LOCAL_REROUTE/CAPACITY_BOOST — a node can sit
        in relay mode forever without ever forming a link (this is exactly why
        the OSPF/OLSR arms used to report 42 "relays" with zero relay links).
        """
        return [l for l in self.topology.links.values()
                if getattr(l, 'link_type', None) == LinkType.TRANSPORT_RELAY
                and getattr(l, 'is_up', True)]

    def count_relay_mode_nodes(self) -> int:
        """Infra nodes whose relay_mode is LOCAL_REROUTE or CAPACITY_BOOST.

        Diagnostic only — reported under a key clearly distinct from the
        active-relay-LINK count.  Nodes in relay mode with no relay link are
        an intent, not a bridge.
        """
        from .phy_mac_state import RelayMode
        return sum(
            1 for ps in self.phy_mac_states.values()
            if getattr(ps, 'relay_mode', RelayMode.OFF)
            in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST))

    def _infra_component_labels(self, include_relay_links: bool = False) -> dict:
        """{node_id: component_id} over the surviving infrastructure graph.

        Args:
            include_relay_links: if False (default) active TRANSPORT_RELAY
                links are ignored — the RAW post-severance partition.  If True
                they count as edges — the OPERATIONAL partition.

        Component ids are assigned in the iteration order of
        `_infra_scope_nodes()`, so they are labels only: never compare ids
        across two calls.  Use `_component_anchor()` when a STABLE identity
        for a component is needed.

        Single implementation shared by every fragment/bridge/reachability
        consumer (count_infra_components, distinct_bridge_link_ids,
        peer_reachable_ue_stats) so they cannot disagree about what a
        component is.
        """
        infra = self._infra_scope_nodes()
        adj: Dict[str, list] = {nid: [] for nid in infra}
        for lnk in self.topology.links.values():
            if not getattr(lnk, 'is_up', True):
                continue
            if (not include_relay_links and
                    getattr(lnk, 'link_type', None) == LinkType.TRANSPORT_RELAY):
                continue
            eps = getattr(lnk, 'endpoints', None)
            if not eps or len(eps) != 2:
                continue
            a, b = eps
            if a in adj and b in adj and a != b:
                adj[a].append(b)
                adj[b].append(a)
        comp: Dict[str, int] = {}
        cid = 0
        for start in infra:
            if start in comp:
                continue
            stack = [start]
            while stack:
                nid = stack.pop()
                if nid in comp:
                    continue
                comp[nid] = cid
                stack.extend(nb for nb in adj[nid] if nb not in comp)
            cid += 1
        return comp

    def _relay_free_components(self) -> dict:
        """{node_id: component_id} over the infra graph with TRANSPORT_RELAY
        links removed — the raw post-severance partition.  Shared by every
        bridge-detection consumer so they cannot disagree."""
        return self._infra_component_labels(include_relay_links=False)

    @staticmethod
    def _component_anchor(comp_labels: dict, cid) -> Optional[str]:
        """STABLE identity of a component: the lexicographically smallest node
        id it contains.

        Component *ids* from `_infra_component_labels` are traversal-order
        labels and change whenever the graph changes.  Anything that has to
        recognise "the same component" across two graph variants (relay-free
        vs operational) or produce reproducible tie-breaks across nodes must
        key on the anchor, not the id.
        """
        members = [nid for nid, c in comp_labels.items() if c == cid]
        return min(members) if members else None

    # ── Reachability constants (see compute_global_connectivity_reward) ──
    #
    # A stranded user's value is NOT its throughput — that is why the term
    # exists and it is why it is kept.  But at 1.0 per restored UE, capped
    # at 60.0, it was no longer a life-safety PREMIUM on top of delivery: it
    # was the largest single positive term in the whole reward, and the
    # policy that maximised it lost to doing nothing on the KPI.
    #
    # DEMOTED TO A TIEBREAKER — measured, not assumed.  With the relay head
    # freed on its own from a do-nothing policy (Scenario A seed 42, 400
    # ticks, mean of 5 torch draws), the per-tick term decomposition was:
    #
    #     term            do_nothing   free:relay      delta
    #     delivery             39.20        33.87      -5.33
    #     relay_contrib         0.00         0.00       0.00   <- reprice worked
    #     reunify              -4.00         6.28     +10.28
    #     reach                 0.00        37.75     +37.75   <- dominant
    #
    # i.e. relaying DESTROYED 5.33 points of delivered volume and was paid
    # 37.75 for it.  Any policy gradient built on that sum learns to relay
    # for reachability points and to stop caring about delivery, which is
    # exactly the trained-below-do-nothing inversion that was observed
    # (trained 29.2 % vs do-nothing 30.1 % UE connectivity).
    #
    # THE NEW ARITHMETIC.  On the Scenario-A topology (42 infra nodes,
    # 150 surviving UEs):
    #
    #     max reach contribution     = REACH_RESTORED_CAP = +5.0
    #                                  (cap binds at 50 restored UEs;
    #                                   150 x 0.10 = 15.0 without the cap)
    #     delivery block, same span  = +100 x delivered_frac, +25 full-delivery
    #                                  bonus, +10 x general = up to +135
    #     delivery for the SAME users: restoring the 62 UEs that the measured
    #                                  4 -> 2 merge re-joins is 62/150 = 41.3 %
    #                                  of the UE population, i.e. up to
    #                                  +41.3 of the +100 volume term
    #
    # So the maximum the term can ever pay (+5.0) is 12 % of what delivering
    # the same users' traffic pays (+41.3), and 3.7 % of the delivery block's
    # span (+135).  Per user: 0.10 points against the 100/150 = 0.667 points
    # that user's fair share of the volume term is worth — the reachability
    # premium is now 15 % of the throughput it displaces, where it used to be
    # 150 %.  The term can therefore break a tie between two actions that
    # deliver the same volume, and can no longer outrank delivery.
    #
    # STILL WORTH BRIDGING A SINGLETON.  The marginal trade the term was
    # introduced for survives, because it never rested on this term alone:
    # a 1-node island costs ~-1.0 to -1.5 (one churn flip + beam training)
    # and earns its traffic share of the volume term (~+2 to +3) plus
    # REUNIFY_PER_FRAGMENT (+6) plus, now, ~+0.7 of reachability for its ~7
    # users.  The bridge stays clearly profitable; what changed is that it is
    # profitable because it RECONNECTS AND CARRIES, not because reconnecting
    # alone pays more than carrying.
    REACH_RESTORED_PER_UE = _RP.reach_per_ue   # 0.10 per UE (was 1.0)
    REACH_RESTORED_CAP    = _RP.reach_cap      # 5.0 cap    (was 60.0)

    # CONDITIONAL ON THE RESTORED USERS ACTUALLY CARRYING TRAFFIC.
    #
    # The evaluation KPI scores delivered volume and cell attachment.  The
    # term as written scored peer reachability, which is a DIFFERENT
    # quantity, and that divergence is the mechanism by which the term could
    # be reward-positive while KPI-negative: a bridge that re-joins users who
    # then send nothing moves the reward and not the metric.
    #
    # With this flag the term counts only those restored UEs that were an
    # endpoint of at least one UE-to-UE flow that DELIVERED non-zero volume
    # this tick (Simulator._delivered_ue_endpoints, a set built by
    # _forward_traffic from the same deliveries the +100 term is computed
    # from — no new physics, no new counter, no extra graph work).
    #
    # WHY THIS IS THE RIGHT CHOICE, and what it costs.  It aligns the term
    # with the metric without removing the disaster-recovery intent, because
    # the intent was never "pay for a graph edge" — it was "a stranded
    # island is somebody's only link, and its value is not its THROUGHPUT".
    # That distinction is preserved exactly: the term still pays a FLAT 0.10
    # per restored user whether that user carries 1 kbps or 10 Mbps, so a
    # low-traffic island is worth as much per user as a busy one, which is
    # the whole point.  What it no longer pays for is a user who is
    # reconnected on paper and carries nothing at all — and a user who can
    # be reached but whose calls do not get through is not a user whose
    # life-safety link has been restored.
    #
    # The cost is a tick of latency in one direction: a bridge that comes up
    # this tick pays nothing until traffic flows over it.  That is acceptable
    # because the term is SUSTAINED — it is paid on every tick the users stay
    # connected — so the credit is delayed, not lost, and REUNIFY_PER_FRAGMENT
    # (which is unconditional on traffic, by design) already covers the
    # discovery gradient for forming the bridge in the first place.
    REACH_REQUIRE_TRAFFIC = _RP.reach_require_traffic

    def current_peer_reach_stats(self) -> dict:
        """Peer-reachability stats for THIS tick, for any arm.

        `_update_phy_mac_observations` refreshes the cached stats once per tick,
        but it is only reached from `_build_agent_observations` — i.e. only on
        the agent-driven arms (MARL, marl_freeze, the random control).  The
        routing baselines (SDN / OSPF / OLSR / BATMAN / AODV) never call it, so
        reading the cached attributes directly would report their init defaults
        (100 % reachable, 0 stranded) as if measured.  This checks the freshness
        stamp and recomputes when stale, so every arm is measured on the same
        primitive.
        """
        if getattr(self, '_peer_reach_tick', None) == self.current_tick:
            cached = getattr(self, '_peer_reach_stats', None)
            if cached:
                return cached
        stats = self.peer_reachable_ue_stats()
        self._peer_reach_stats           = stats
        self._peer_reachable_ue_fraction = stats['peer_reachable_fraction']
        self._stranded_ue_count          = stats['stranded']
        self._reach_restored_ue_count    = stats['restored']
        self._reach_restored_ue_ids      = stats.get('restored_ues', frozenset())
        self._peer_reach_tick            = self.current_tick
        return stats

    def peer_reachable_ue_stats(self, ue_serving: Optional[dict] = None) -> dict:
        """How many surviving UEs can reach the BODY of the survivor network,
        and how many of those owe that reachability to the policy's bridges.

        Why this exists (disaster-recovery objective).  `_reachable_ue_fraction`
        answers "is this UE attached to a cell?" — it is an ACCESS metric and
        it cannot see fragmentation at all: a UE camped on a cell inside a
        one-node island is 100 % "reachable" by that measure while being able
        to call nobody.  What matters after a severance is whether a UE can
        reach anyone at all, and that is a property of the INFRA COMPONENT its
        serving cell sits in.

        Definitions
        -----------
        network body   the relay-free component holding the most attached UEs
                       (ties: most infra nodes, then smallest node id).  This
                       is the part of the network that survived on its own;
                       it is a property of the SEVERANCE, identical for every
                       arm, and it is pinned by its anchor node so the
                       relay-free and operational graphs agree on which
                       component it is.
        reachable_raw  attached UEs whose serving cell is in the body of the
                       RELAY-FREE graph — the reachability the severance left.
        reachable_op   attached UEs whose serving cell is in the body of the
                       OPERATIONAL graph (active TRANSPORT_RELAY links count).
                       The body can only GROW when relay links are added, so
                       reachable_op >= reachable_raw always.
        restored       reachable_op - reachable_raw.  Exactly the number of
                       UEs that moved from "cannot reach any peer outside my
                       island" into the connected body, and attributable to
                       the relay links because it is measured as a delta
                       against the relay-free graph.  Zero for every arm that
                       forms no bridges, and zero for a bridge that merely
                       parallels an existing intra-component hop.
        stranded       total surviving UEs - reachable_op (includes UEs with
                       no serving cell at all — a bridge cannot fix those, and
                       they correctly contribute nothing to `restored`).
        attached_outside_body
                       attached UEs whose serving cell is in a component OTHER
                       than the body.  Diagnostic, and the one number that says
                       whether an unclosed residual fragment actually holds
                       USERS: if it is 0 while achieved components > 1, the
                       leftover fragments are UE-free infrastructure (bare
                       relay / O-DU sites) and no per-UE term can motivate
                       closing them — only the reunification block can.

        Args:
            ue_serving: optional {ue_id: [serving_infra_node_id, ...]} — the
                UE->O-RU mapping (`ue_to_oru_map`).  Pass the copy
                `_update_phy_mac_observations` builds while it walks the Uu
                links to avoid a second pass.  Rebuilt from `ue_to_oru_map`, or
                from the link set, if omitted.

        Counting note: every count here is over DISTINCT UEs.  The per-O-RU
        `active_ue_count` / `ue_counts` numbers elsewhere in the simulator count
        Uu LINKS, and a multi-homed UE holds several — on the Scenario-A
        topology that inflates 150 UEs to 233 "UEs".  Using those here made
        `stranded` collapse to 0 whenever reachable_op exceeded the true UE
        population, which silently hid every stranded user.  A multi-homed UE
        is counted as reachable if ANY of its serving cells is in the body,
        which is the physically correct rule (it can reach the body over
        either cell).

        Cost: two O(N+L) component passes over the ~40-node infra graph.
        """
        out = {'total_ues': 0, 'attached_ues': 0, 'reachable_raw': 0,
               'reachable_op': 0, 'restored': 0, 'stranded': 0,
               'attached_outside_body': 0,
               'peer_reachable_fraction': 1.0, 'body_anchor': None,
               # IDENTITIES of the restored UEs, not just the count.  The
               # reachability reward term is conditional on those users
               # carrying traffic (Simulator.REACH_REQUIRE_TRAFFIC), which
               # needs to intersect them with the tick's delivered flows.
               # Built from the same two membership tests that produce
               # reachable_raw / reachable_op, so it cannot disagree with
               # `restored`: len(restored_ues) == restored, always.
               'restored_ues': frozenset()}
        try:
            total_ues = sum(
                1 for n in self.topology.nodes.values()
                if n.node_type == NodeType.UE and n.is_survivor)
            out['total_ues'] = total_ues
            if total_ues <= 0:
                return out

            if ue_serving is None:
                ue_serving = {}
                for link in self.topology.links.values():
                    if not getattr(link, 'is_up', True):
                        continue
                    a, b = link.endpoints
                    for ru_id in (a, b):
                        if ru_id not in self.phy_mac_states:
                            continue
                        ue_id = b if ru_id == a else a
                        ue_node = self.topology.nodes.get(ue_id)
                        if (ue_node is not None
                                and ue_node.node_type == NodeType.UE
                                and ue_node.is_survivor):
                            ue_serving.setdefault(ue_id, []).append(ru_id)

            comp_raw = self._infra_component_labels(include_relay_links=False)
            if not comp_raw:
                return out

            # DISTINCT surviving UEs holding at least one Uu link into the
            # infra scope.  `attached` is a set, so a multi-homed UE counts once.
            attached = {
                ue for ue, rus in ue_serving.items()
                if any(r in comp_raw for r in rus)}
            out['attached_ues'] = len(attached)

            # UE mass per raw component, over DISTINCT UEs.  A multi-homed UE
            # straddling two components contributes to both — that is right for
            # picking the body (either cell would serve it) and cannot
            # double-count `reachable_*`, which are set sizes.
            ue_mass:   Dict[int, int] = {}
            node_mass: Dict[int, int] = {}
            for nid, cid in comp_raw.items():
                node_mass[cid] = node_mass.get(cid, 0) + 1
            for ue in attached:
                for cid in {comp_raw[r] for r in ue_serving[ue] if r in comp_raw}:
                    ue_mass[cid] = ue_mass.get(cid, 0) + 1

            # The body: most attached UEs, then most nodes, then smallest
            # anchor — fully deterministic, no dependence on traversal order.
            body_cid = min(
                node_mass,
                key=lambda c: (-ue_mass.get(c, 0), -node_mass[c],
                               self._component_anchor(comp_raw, c) or ''))
            anchor = self._component_anchor(comp_raw, body_cid)
            out['body_anchor'] = anchor

            body_raw_nodes = {nid for nid, cid in comp_raw.items()
                              if cid == body_cid}
            _raw_set = {ue for ue in attached
                        if any(r in body_raw_nodes for r in ue_serving[ue])}
            out['reachable_raw'] = len(_raw_set)

            comp_op = self._infra_component_labels(include_relay_links=True)
            body_op = comp_op.get(anchor)
            if body_op is None:
                _op_set = _raw_set
                out['reachable_op'] = out['reachable_raw']
            else:
                body_op_nodes = {nid for nid, cid in comp_op.items()
                                 if cid == body_op}
                _op_set = {ue for ue in attached
                           if any(r in body_op_nodes for r in ue_serving[ue])}
                out['reachable_op'] = len(_op_set)

            # The body can only GROW when relay links are added, so
            # _op_set is a superset of _raw_set and the difference is
            # exactly the UEs the policy's own bridges re-joined.
            out['restored_ues'] = frozenset(_op_set - _raw_set)
            out['restored'] = max(0, out['reachable_op'] - out['reachable_raw'])
            out['stranded'] = max(0, total_ues - out['reachable_op'])
            out['attached_outside_body'] = max(
                0, out['attached_ues'] - out['reachable_op'])
            out['peer_reachable_fraction'] = min(
                1.0, out['reachable_op'] / float(total_ues))
            return out
        except Exception:
            return out

    def count_bridging_relay_links(self) -> int:
        """Number of INDEPENDENT fragment merges performed by relay links.

        This is the de-duplicated count (distinct_bridge_link_ids): relay
        links whose endpoints lie in different components of the relay-free
        infra graph, with parallel/cyclic duplicates removed so that two
        links joining the same pair of fragments count ONCE.  Equal by
        construction to (raw components) - (components with relay links).

        LOCAL_REROUTE links that duplicate an existing intra-fragment hop are
        excluded (they are not cross-component at all).
        """
        return len(self.distinct_bridge_link_ids())

    def _count_island_fragments(self) -> int:
        """Operational fragment count (see count_infra_components).

        Kept as the historical name; delegates to the shared primitive so the
        reward, the KPI snapshot and the comparison harness can never drift.
        """
        return self.count_infra_components(include_relay_links=True)

    def _take_node_down(self, node_id: str):
        """Mark a node as failed and take down all its links (same mechanism
        as sever_zone).  Downed link ids are tracked so node_recovery can
        restore them symmetrically."""
        self.topology.nodes[node_id].is_survivor = False
        if node_id in self.agents:
            del self.agents[node_id]
        downed = []
        for lid, link in self.topology.links.items():
            if link.is_up and node_id in link.endpoints:
                link.is_up = False
                downed.append(lid)
                ep0, ep1 = link.endpoints
                if self.topology.graph.has_edge(ep0, ep1):
                    self.topology.graph.remove_edge(ep0, ep1)
        self._links_downed_by_node_failure.setdefault(node_id, []).extend(downed)
        # Invalidate connectivity caches
        self.topology.invalidate_infrastructure_cache()
        self._oru_connectivity_cache.clear()
        self._ue_routing_table.clear()

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
                    # Also remove from the graph so shortest_path won't
                    # route through dead links
                    if self.topology.graph.has_edge(link.endpoints[0], link.endpoints[1]):
                        self.topology.graph.remove_edge(link.endpoints[0], link.endpoints[1])

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
                    if self.topology.graph.has_edge(ep0, ep1):
                        self.topology.graph.remove_edge(ep0, ep1)
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
                        if self.topology.graph.has_edge(ep0, ep1):
                            self.topology.graph.remove_edge(ep0, ep1)

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

        elif event.event_type == 'backhaul_degradation':
            # Post-disaster MW link degradation: rain fade, debris, misalignment
            # Reduces capacity on microwave/backhaul links (not fiber, not UE access)
            # This is a PHYSICAL event — applies equally regardless of protocol
            deg_frac = event.parameters.get('degradation_fraction', 0.70)
            reason = event.parameters.get('reason', 'disaster')
            degraded_count = 0
            for lid, link in self.topology.links.items():
                if not link.is_up:
                    continue
                # Only degrade MW backhaul transport links
                lt_str = str(getattr(link, 'link_type', ''))
                if any(t in lt_str for t in ['Microwave', 'microwave', 'MICROWAVE']):
                    old_cap = link.capacity
                    link.capacity = max(50, link.capacity * (1.0 - deg_frac))
                    # Update graph edge weight
                    ep0, ep1 = link.endpoints
                    if self.topology.graph.has_edge(ep0, ep1):
                        self.topology.graph[ep0][ep1]['capacity'] = link.capacity
                    degraded_count += 1
            if degraded_count > 0:
                self.topology.invalidate_infrastructure_cache()
                print(f"Tick {event.tick}: Backhaul degradation ({reason}) — "
                      f"{degraded_count} MW links at {(1-deg_frac)*100:.0f}% capacity")

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
                    node = self.topology.nodes[core_nid]
                    if core_nid not in self.agents and node_hosts_agent(node):
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
                self._take_node_down(node_id)
                print(f"Tick {event.tick}: Node {node_id} depleted")
            else:
                print(f"Tick {event.tick}: energy_depletion ignored (node {node_id} not found)")

        elif event.event_type == 'node_failure':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                self._take_node_down(node_id)
                print(f"Tick {event.tick}: Node {node_id} failed")
            else:
                print(f"Tick {event.tick}: node_failure ignored (node {node_id} not found)")

        elif event.event_type == 'node_recovery':
            node_id = event.parameters.get('node_id')
            if node_id in self.topology.nodes:
                node = self.topology.nodes[node_id]
                node.is_survivor = True
                # Restore links taken down by this node's failure — but only
                # those whose other endpoint is still alive.
                for lid in self._links_downed_by_node_failure.pop(node_id, []):
                    link = self.topology.links.get(lid)
                    if link is None:
                        continue
                    other = (link.endpoints[0] if link.endpoints[1] == node_id
                             else link.endpoints[1])
                    other_node = self.topology.nodes.get(other)
                    if other_node and other_node.is_survivor:
                        link.is_up = True
                # Re-create the node's agent (removed on failure)
                if node_id not in self.agents and node_hosts_agent(node):
                    self.agents[node_id] = create_agent_for_node(
                        node_id, node.node_type.value, self.slice_dictionary
                    )
                self.topology.invalidate_infrastructure_cache()
                self._oru_connectivity_cache.clear()
                self._ue_routing_table.clear()
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
            # Dedicated stream for the geometry, so adding positions does not
            # perturb the legacy module-level RNG consumption downstream.
            _geo_rng = random.Random(
                event.tick * 7919 + (self.config.random_seed or 0))

            # ── INCIDENT-ZONE PLACEMENT ───────────────────────────────────
            # Rescue UEs used to be created with NO coordinates, i.e. all of
            # them at (0, 0) — the corner of the 8 km x 8 km deployment, up to
            # 11 km from the incident they were dispatched to (see
            # Topology.add_ue_dynamically for the full rationale).  They are
            # now scattered around the incident zone, which is resolved in
            # descending order of directness:
            #   1. explicit event parameters (x_pos/y_pos or epicenter);
            #   2. the centroid of the infrastructure the disaster DESTROYED —
            #      the most defensible definition of "where the incident is",
            #      and available for every severance/geo_disaster scenario;
            #   3. the centroid of the named coverage area's cells;
            #   4. the centroid of all surviving cells (last resort).
            cx, cy = self._incident_zone_centre(event, coverage_area)
            radius = float(event.parameters.get(
                'deploy_radius_m', self.RESCUE_DEPLOY_RADIUS_M))

            added_count = 0
            self._rescue_arrived_tick = event.tick  # Track arrival for traffic model
            for i in range(num_rescue_ues):
                self.ue_counter += 1
                ue_id = f"Rescue_UE_{self.ue_counter}"

                # Uniform over the disc (sqrt keeps the density uniform rather
                # than clustering at the centre).
                _a = _geo_rng.uniform(0.0, 2.0 * math.pi)
                _r = radius * math.sqrt(_geo_rng.random())
                ux, uy = cx + _r * math.cos(_a), cy + _r * math.sin(_a)

                if self.topology.add_ue_dynamically(
                        ue_id, coverage_area,
                        x_pos=ux, y_pos=uy, rng=_geo_rng):
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

        elif event.event_type == 'traffic_surge':
            # Scale a node's generated traffic by `multiplier` for `duration` ticks
            node_id = event.parameters.get('node_id')
            duration = int(event.parameters.get('duration', 50))
            multiplier = float(event.parameters.get('multiplier', 2.0))
            if node_id in self.topology.nodes:
                self._active_traffic_surges[node_id] = (event.tick + duration, multiplier)
                print(f"Tick {event.tick}: traffic_surge on {node_id} — "
                      f"x{multiplier:g} for {duration} ticks")
            else:
                print(f"Tick {event.tick}: traffic_surge ignored (node {node_id} not found)")

        else:
            # Never silently ignore scenario events (e.g. geo_disaster has no
            # handler yet) — make the gap visible.
            print(f"Tick {event.tick}: WARNING - unhandled scenario event "
                  f"'{event.event_type}' (parameters: {event.parameters}) — ignored")

        # Update topology graph after link changes
        self.topology.update_link_statuses()
        # Scenario events may change nodes/links/flows — recompute the
        # best-case connectivity upper bound lazily on next use.
        self._island_bound_cache = None

    # Radius of the rescue staging area around the incident centre, metres.
    # 800 m is the scale of a real multi-agency incident ground (inner cordon
    # plus forward command post); it is deliberately smaller than the FR1
    # attach radius (Topology.UE_ATTACH_MAX_DIST_M = 1500 m) so a rescue team
    # deployed to a zone whose cells all died is genuinely orphaned rather than
    # silently reattached across the map.
    RESCUE_DEPLOY_RADIUS_M = 800.0

    def _incident_zone_centre(self, event: 'ScenarioEvent',
                              coverage_area: Optional[str] = None
                              ) -> Tuple[float, float]:
        """Where the incident is, in metres, for rescue-force placement.

        Resolution order (most direct evidence first):
          1. explicit event parameters: (x_pos, y_pos) or epicenter=(x, y);
          2. centroid of the infrastructure the disaster DESTROYED
             (is_survivor False) — the most defensible definition of "the
             incident", and available for every severance / geo_disaster /
             node_failure scenario in this family;
          3. centroid of the named coverage area's surviving cells;
          4. centroid of every surviving infra node.

        Returns (0, 0) only if the topology has no positioned nodes at all.
        """
        p = getattr(event, 'parameters', {}) or {}
        if p.get('x_pos') is not None and p.get('y_pos') is not None:
            return float(p['x_pos']), float(p['y_pos'])
        epi = p.get('epicenter')
        if isinstance(epi, (tuple, list)) and len(epi) >= 2:
            return float(epi[0]), float(epi[1])

        def _centroid(nids):
            pts = [(self.topology.nodes[n].x_pos, self.topology.nodes[n].y_pos)
                   for n in nids]
            pts = [(x, y) for x, y in pts if not (x == 0.0 and y == 0.0)]
            if not pts:
                return None
            return (sum(x for x, _ in pts) / len(pts),
                    sum(y for _, y in pts) / len(pts))

        destroyed = [nid for nid, n in self.topology.nodes.items()
                     if n.node_type != NodeType.UE and not n.is_survivor]
        c = _centroid(destroyed)
        if c is not None:
            return c

        if coverage_area:
            area = [nid for nid, n in self.topology.nodes.items()
                    if n.node_type != NodeType.UE and n.is_survivor
                    and n.coverage_area == coverage_area]
            c = _centroid(area)
            if c is not None:
                return c

        alive = [nid for nid, n in self.topology.nodes.items()
                 if n.node_type != NodeType.UE and n.is_survivor]
        c = _centroid(alive)
        return c if c is not None else (0.0, 0.0)

    def _generate_traffic(self) -> Dict[str, Dict[TrafficClass, float]]:
        """Generate traffic arrivals for current tick."""
        # Reset per-tick tracking before new traffic arrives
        for node in self.topology.nodes.values():
            node.reset_queues()
        for link in self.topology.links.values():
            link.reset_utilization()

        arrivals = self.traffic_generator.generate_traffic(self.current_tick)

        # Apply active traffic surges (scenario event 'traffic_surge')
        if self._active_traffic_surges:
            expired = [nid for nid, (end_tick, _mult) in self._active_traffic_surges.items()
                       if self.current_tick >= end_tick]
            for nid in expired:
                del self._active_traffic_surges[nid]
                print(f"Tick {self.current_tick}: traffic_surge on {nid} ended")
            for nid, (_end, mult) in self._active_traffic_surges.items():
                if nid in arrivals:
                    arrivals[nid] = {tc: amt * mult for tc, amt in arrivals[nid].items()}

        return arrivals

    # ── PHY/MAC initialisation ────────────────────────────────────────────────

    def _init_phy_mac_states(self):
        """Create one PHYMACState per O-RU / O-DU and register relay positions."""
        import random as _rnd
        agent_types = RADIO_AGENT_TYPES   # shared: see node_hosts_agent()
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

                # Register position: prefer the node's real topology geometry
                # (x_pos/y_pos, used by disaster zones) so the dynamic SINR
                # model is consistent with the rest of the sim; fall back to
                # a synthetic position only when the topology has none.
                if not self.transport_relay_model.node_positions.get(node_id):
                    x = getattr(node, 'x_pos', 0.0)
                    y = getattr(node, 'y_pos', 0.0)
                    if x == 0.0 and y == 0.0:
                        x = _rnd.uniform(0, 10000)
                        y = _rnd.uniform(0, 10000)
                    self.transport_relay_model.register_position(node_id, x, y)

    # ── AGGREGATE CO-CHANNEL INTERFERENCE ─────────────────────────────────────
    #
    # WHAT WAS WRONG.  This simulator previously computed
    #
    #     SINR_i = tx_power_i − FSPL(d_i) − (NOISE_FLOOR_DBM + 0.0)
    #
    # with `SINR_AMBIENT_INTERFERENCE_DB = 0.0`, i.e. THERMAL-NOISE-ONLY.  A
    # thermal-only SINR makes transmit power a free capacity multiplier: every
    # +6 dB step buys ~1 MCS row (up to 2× spectral efficiency) and costs the
    # policy only a token reward step penalty, while doing NOTHING to any
    # neighbour.  Raising power was therefore close to a dominant action, and
    # any capacity or delivery number produced under it is inflated.
    #
    # WHAT REPLACES IT.  A real aggregate-interference term:
    #
    #     SINR_i = S_i / (N + I_i)      (combined in the LINEAR domain)
    #     I_i    = Σ_{j ≠ i, co-channel, ACTIVE}  P_rx(j → i)
    #
    # where P_rx(j → i) uses the SAME path-loss/atmosphere function as the
    # wanted signal (TransportRadioClass.interference_power_dbm), plus the
    # two-end off-boresight antenna discrimination, plus the interferer's
    # activity factor.  This is an EXACT O(N²) pass, not an approximation:
    # with 42 infra nodes and three bands it is ~1.8 k distance evaluations
    # per tick, which is free next to the routing passes.
    #
    # BANDS ARE SEPARATE.  Access (FR1 3.5 GHz) and transport (60 GHz TG,
    # 18 GHz MW) interfere only WITHIN their own band — a 60 GHz mesh bridge
    # cannot desensitise a 3.5 GHz Uu carrier.  Each band has its own
    # transmitter set, its own antenna discrimination and its own channel
    # plan:
    #
    #   BAND_ACCESS (FR1, 3.5 GHz, reuse-3, 5 dB/end discrimination)
    #       Transmitters: every surviving node with a PHYMACState (each is a
    #       cell), at ACCESS_RESOURCE_UTILISATION (see below).  THIS is the
    #       band where interference bites: quasi-omni sector antennas, no
    #       atmospheric absorption, ~14 co-channel cells in an 8×8 km area.
    #
    #   BAND_TG (60 GHz, reuse-3, 15 dB/end discrimination)
    #       Transmitters: MultiHaul sites with an up MULTIHAUL_MESH or
    #       TRANSPORT_RELAY link.  Activity = prb_relay_fraction (a TDD TG
    #       frame only radiates in the slots it uses).  Interference here is
    #       tens of dB BELOW thermal: 15 dB/end pencil-beam discrimination
    #       plus 15 dB/km oxygen absorption make 60 GHz mesh genuinely
    #       noise-limited, which is the physically correct answer and is why
    #       the TG reach is hard-bounded by the budget rather than by I.
    #
    #   BAND_MW (18 GHz, reuse-4, 20 dB/end discrimination)
    #       Transmitters: nodes with an up MICROWAVE_PTP link, continuous
    #       carrier (activity 1.0).  Licensed PtP hops are frequency
    #       coordinated by the regulator, which is what reuse-4 represents.
    #
    # WHY A FREQUENCY-REUSE FACTOR EXISTS.  A 42-cell FR1 deployment in an
    # 8×8 km area is not single-frequency; hard reuse-3 / fractional frequency
    # reuse is the standard plan.  Modelling every cell as co-channel would
    # put the whole network in universal outage — an artefact of omitting the
    # channel plan, not a physical result.  Co-channel membership is assigned
    # by crc32 of the node id (transport_relay_model.access_channel_group /
    # transport_channel_group), so it is deterministic, process-independent
    # and IDENTICAL for every arm and every seed replay.
    #
    # POWER CONTROL IS NOW A GENUINE TRADE, so transmit power is deliberately
    # NOT locked: a node that raises power gains on its own wanted signal and
    # degrades every co-channel neighbour's SINR.  The cost is physical, not a
    # reward-shaping term.
    #
    # MEASUREMENT LAG.  The cache is built ONCE per tick, from the powers and
    # utilisations standing at the top of the tick, and every consumer in that
    # tick reads the same values.  That is how a real network works — CQI is
    # reported from a previous measurement window — and it also makes the pass
    # independent of the two tick orderings in this codebase (the engine's
    # run() forwards traffic before actions; run_timeline_comparison.py
    # executes actions first).
    # ACCESS_RESOURCE_UTILISATION — the co-channel RESOURCE-COLLISION factor
    # for the FR1 band: the fraction of an interferer's radiated power that
    # actually lands on the victim's PRBs.  It is a MODELLING CONSTANT, named
    # as such, and it is what makes the 42-cell aggregate a realistic level
    # rather than a worst-case one.  It stands for three effects that all
    # reduce co-channel collision below 100 %:
    #   * ICIC / fractional frequency reuse: reuse partners mute each other's
    #     edge sub-band (3GPP LTE/NR inter-cell interference coordination), so
    #     full-power co-channel overlap is a minority of the band;
    #   * duty cycle: a disaster-mode cell carries a handful of emergency and
    #     telemetry flows, not a full buffer;
    #   * SSB/CSI-RS reference signals set the floor — an idle cell still
    #     radiates.
    # CALIBRATION.  At 0.10 the resulting FR1 SINR distribution over the
    # evaluation topology (seeds 50-54, 42 infra nodes) is
    #     min −7.4 dB   mean +6.0 dB   max +30 dB   (I/N ≈ +13.5 dB)
    # which matches the 3GPP UMa downlink reference SINR CDF (TR 36.873 /
    # TR 38.901: 5th pct ≈ −3 dB, median ≈ +6 dB, 95th pct ≈ +22 dB).  That
    # match is the justification for the value; it is not tuned to any arm's
    # result and it is applied identically everywhere.
    # DELIBERATELY NOT COUPLED TO LIVE LOAD.  An earlier version used each
    # cell's measured prb_utilization here.  That is physically appealing but
    # introduces a positive feedback loop — load → interference → lower SINR →
    # lower capacity → higher load — whose fixed point is universal outage,
    # and it makes the interference level depend on which of the two tick
    # orderings in this codebase is running (offered_load is zero at
    # observation time under the harness ordering and non-zero under the
    # engine's).  A fixed reference utilisation is deterministic,
    # ordering-independent and cannot collapse.  Power control remains a
    # genuine trade regardless, because I scales with every neighbour's Tx
    # power, which is exactly the coupling item 1 requires.
    ACCESS_RESOURCE_UTILISATION   = 0.10
    INTERFERENCE_TG_ACTIVITY_FLOOR = 0.05  # TG beacon/keep-alive duty cycle
    INTERFERENCE_MIN_SEPARATION_M = 50.0   # near-field guard, matches SINR guard

    def _interference_emitters(self) -> Dict[str, list]:
        """Per-band list of (node_id, effective_tx_dbm, channel_group).

        `effective_tx_dbm` already carries the activity factor in dB, so the
        interference sum needs no further scaling.  A node appears in a band
        only if it actually operates a radio in that band this tick, which is
        what makes the sum an ACTIVE-transmitter sum rather than a
        "everything that exists" sum.
        """
        import math as _m
        trm = self.transport_relay_model
        out = {BAND_ACCESS: [], BAND_TG: [], BAND_MW: []}

        # Which nodes hold an up link of each transport type this tick?
        tg_nodes: Set[str] = set()
        mw_nodes: Set[str] = set()
        for link in self.topology.links.values():
            if not getattr(link, 'is_up', True):
                continue
            lt = getattr(link, 'link_type', None)
            if lt is LinkType.MULTIHAUL_MESH or lt is LinkType.TRANSPORT_RELAY:
                tg_nodes.update(link.endpoints)
            elif lt in (LinkType.MICROWAVE_PTP, LinkType.MICROWAVE):
                mw_nodes.update(link.endpoints)

        for node_id, ps in self.phy_mac_states.items():
            node = self.topology.nodes.get(node_id)
            if node is None or not node.is_survivor:
                continue
            if node_id not in trm.node_positions:
                continue    # unpositioned — cannot contribute a geometry term

            # ACCESS: every surviving cell radiates on its own carrier, at the
            # fixed co-channel resource-collision factor (see the constant's
            # docstring for why this is not coupled to live load).
            act = self.ACCESS_RESOURCE_UTILISATION
            out[BAND_ACCESS].append(
                (node_id, ps.tx_power_dbm + 10.0 * _m.log10(act),
                 access_channel_group(node_id)))

            # TG: only MultiHaul sites with a live 60 GHz link.  Activity here
            # IS the agent's own prb_relay_fraction — a direct action, not a
            # load measurement, so it introduces no feedback loop.
            if node_id in tg_nodes and getattr(node, 'has_multihaul', False):
                tg_act = max(self.INTERFERENCE_TG_ACTIVITY_FLOOR,
                             min(1.0, float(getattr(ps, 'prb_relay_fraction', 0.0))))
                out[BAND_TG].append(
                    (node_id, self._transport_tx_dbm(ps) + 10.0 * _m.log10(tg_act),
                     transport_channel_group(node_id, BAND_TG)))

            # MW: continuous carrier on a pre-aimed licensed hop.
            if node_id in mw_nodes:
                out[BAND_MW].append(
                    (node_id, self._transport_tx_dbm(ps),
                     transport_channel_group(node_id, BAND_MW)))
        return out

    def _compute_aggregate_interference(self):
        """Build this tick's exact aggregate-interference table.

        Fills self._interference_cache with {(band, node_id): I_dbm} and
        stamps self._interference_tick.  Also writes each node's OUTGOING
        interference contribution into PHYMACState.interference_caused_db, so
        "how much am I hurting my neighbours" is observable per node.

        Exact, not approximated: every ordered co-channel pair is evaluated.
        Cost is O(Σ_band N_band²) ≈ 1.8 k operations per tick on the 42-node
        comparison topology.
        """
        import math as _m
        if getattr(self, '_interference_tick', None) == self.current_tick:
            return
        trm = self.transport_relay_model
        emitters = self._interference_emitters()

        cache: Dict[Tuple[str, str], float] = {}
        caused_lin: Dict[str, float] = {}

        for band, members in emitters.items():
            if len(members) < 2:
                for rx_id, _p, _g in members:
                    cache[(band, rx_id)] = float('-inf')
                continue
            for rx_id, _rx_p, rx_grp in members:
                i_lin = 0.0
                for tx_id, tx_p, tx_grp in members:
                    if tx_id == rx_id or tx_grp != rx_grp:
                        continue   # different channel — not co-channel
                    d = trm.distance_m(rx_id, tx_id)
                    if d is None:
                        continue
                    p_dbm = trm.band_interference_power_dbm(
                        band, tx_p,
                        max(self.INTERFERENCE_MIN_SEPARATION_M, d))
                    p_lin = 10.0 ** (p_dbm / 10.0)
                    i_lin += p_lin
                    caused_lin[tx_id] = caused_lin.get(tx_id, 0.0) + p_lin
                cache[(band, rx_id)] = (10.0 * _m.log10(i_lin)
                                        if i_lin > 0.0 else float('-inf'))

        self._interference_cache = cache
        self._interference_tick = self.current_tick
        for node_id, ps in self.phy_mac_states.items():
            lin = caused_lin.get(node_id, 0.0)
            ps.interference_caused_db = (10.0 * _m.log10(lin) if lin > 0.0
                                         else -200.0)

    def aggregate_interference_dbm(self, node_id: str,
                                   band: str = BAND_ACCESS) -> Optional[float]:
        """Aggregate co-channel interference at `node_id` in `band`, dBm.

        Returns None when there is no co-channel interferer this tick, which
        every SINR helper interprets as "thermal-noise-limited".  Computes the
        per-tick table on first use and reuses it for the rest of the tick.
        """
        self._compute_aggregate_interference()
        v = self._interference_cache.get((band, node_id))
        if v is None or v == float('-inf'):
            return None
        return v

    # ── INTEGRATED ACCESS NODE (design decision 2) ────────────────────────────

    # Minimum usable integrated-access capacity: below this the link is not
    # worth provisioning (re-exported from transport_relay_model so callers
    # need only one import).
    IAB_ACCESS_MIN_USABLE_MBPS = IAB_ACCESS_MIN_USABLE_MBPS

    def integrated_access_capacity_mbps(
            self, site_id: str, ue_id: str, *,
            prb_fraction: Optional[float] = None,
            bandwidth_mhz: Optional[float] = None,
            coverage_radius_m: Optional[float] = None,
            for_emergency: bool = False,
            use_interference: bool = True) -> Tuple[float, float]:
        """Capacity of the ACCESS link from a relay-hosted FR1 small cell to a UE.

        DESIGN DECISION 2 — the relay is an INTEGRATED ACCESS NODE (IAB-style):
        it hosts its own FR1 small cell, so serving a UE directly is physically
        sound rather than a bookkeeping convenience.

        This is the ARM-AGNOSTIC entry point that replaces the harness's
        linear-in-distance placeholder

            dist_factor = max(0.3, 1.0 - dist / 1500.0)
            ue_cap      = 25.0 * ps.prb_relay_fraction * dist_factor

        which had no link budget behind it, never went into outage, ignored the
        site's Tx power and ignored interference entirely.

        Physics (see transport_relay_model.integrated_access_link_capacity):
          * ACCESS_FR1 budget — 3.5 GHz FSPL, 20 MHz, NF 5 dB, 2x8 dBi,
            2 dB implementation loss, 9.99 dB shadow-fade margin;
          * SINR = S/(N + I) with I the live per-tick aggregate FR1
            interference at the SERVING SITE's carrier (item 1);
          * SINR -> MCS -> SE through the site's own PHYMACState, i.e. the
            same OLLA step-down and near-threshold half-rate rule every other
            access link uses;
          * capacity = bandwidth x SE x prb_fraction;
          * hard coverage gate at the site's effective coverage radius, which
            itself scales with Tx power (PHYMACState.apply_power_step).

        Nothing here is MARL-specific.  Any arm — SDN, OSPF, OLSR, BATMAN,
        AODV, the random control, MARL — that decides to light up an
        integrated-access link gets exactly this capacity.

        Args:
            site_id:       the relay / infra node hosting the small cell.  Must
                           have a PHYMACState (i.e. be a radio site).
            ue_id:         the UE to be served.
            prb_fraction:  share of the site's ACCESS carrier given to this
                           link.  Defaults to the site's
                           PHYMACState.prb_relay_fraction, which is the
                           action the policy (or the arm's lock) already sets,
                           so "the agent allocated no relay PRB" still means
                           "no capacity to offer".
            bandwidth_mhz: access channel width; default ACCESS_FR1's 20 MHz.
            coverage_radius_m: coverage gate; default the site's
                           PHYMACState.effective_coverage_radius_m.
            for_emergency: use the emergency MCS row (rescue / life-safety UE).
            use_interference: include the aggregate interference term.  Leave
                           True; False is for A/B diagnostics only.

        Returns:
            (capacity_mbps, sinr_db).  capacity_mbps == 0.0 means unusable:
            unknown geometry, outside the coverage radius, or SINR below the
            most robust MCS.  sinr_db is -inf when geometry or coverage
            rejects the link.

        Example (harness use):
            cap, sinr = sim.integrated_access_capacity_mbps(site, ue_id)
            if cap >= sim.IAB_ACCESS_MIN_USABLE_MBPS:
                ...create the Uu link with capacity=cap...
        """
        ps = self.phy_mac_states.get(site_id)
        if ps is None:
            return 0.0, float('-inf')

        trm = self.transport_relay_model
        # UEs are not registered in the transport model's position table
        # (only infra nodes are), so resolve geometry from the topology and
        # fall back to the position table for the site.
        site = self.topology.nodes.get(site_id)
        ue = self.topology.nodes.get(ue_id)
        if site is None or ue is None:
            return 0.0, float('-inf')
        sx, sy = trm.node_positions.get(
            site_id, (getattr(site, 'x_pos', 0.0), getattr(site, 'y_pos', 0.0)))
        ux, uy = getattr(ue, 'x_pos', 0.0), getattr(ue, 'y_pos', 0.0)
        dist_m = math.hypot(sx - ux, sy - uy)

        i_dbm = (self.aggregate_interference_dbm(site_id, BAND_ACCESS)
                 if use_interference else None)
        radius = (ps.effective_coverage_radius_m if coverage_radius_m is None
                  else float(coverage_radius_m))
        share = (ps.prb_relay_fraction if prb_fraction is None
                 else float(prb_fraction))

        return integrated_access_link_capacity(
            dist_m, ps.tx_power_dbm, share,
            bandwidth_mhz=bandwidth_mhz,
            interference_dbm=i_dbm,
            coverage_radius_m=radius,
            rain_mm_h=trm.rain_mm_h,
            phy_mac_state=ps,
            for_emergency=for_emergency,
        )

    # ── Dynamic per-node SINR model constants ─────────────────────────────────
    #
    # SINR_i = S_i / (N + I_i)   in linear terms, where
    #
    #   S_i   = P_tx,i + 2·G_ant − FSPL(d_i) − impl_loss − link_margin
    #           (TransportRadioClass ACCESS_FR1: 3.5 GHz, 20 MHz, NF 5 dB,
    #           8 dBi/end, 2 dB implementation, 9.99 dB shadow-fade margin —
    #           a decomposition that reproduces the previous model's single
    #           effective −100 dBm floor EXACTLY, so the interference term is
    #           the only change to the access budget).
    #   N     = kTB + NF = −95.99 dBm at 20 MHz.
    #   I_i   = aggregate co-channel interference from
    #           aggregate_interference_dbm(node, BAND_ACCESS) — see the
    #           AGGREGATE CO-CHANNEL INTERFERENCE block above.
    #   d_i   = distance to the STRONGEST (nearest) active infra neighbour
    #           link of node i; if the node has no positioned infra
    #           neighbour, a serving-link proxy of 0.7 × effective coverage
    #           radius is used instead.
    #   Result clamped to [−10, +30] dB (practical CQI reporting range;
    #           also keeps Block A's (sinr+5)/35 normalisation in-range).
    #
    # Thermal-only reference points at 23 dBm (i.e. I = 0, unchanged from the
    # previous model): 700 m → 22.8 dB, 1 km → 19.7 dB, 2 km → 13.6 dB,
    # 4.6 km → 6.4 dB.  With the aggregate-interference term the FR1 band is
    # interference-limited and the measured spread drops by roughly 4-6 dB;
    # the exact figures for the evaluation topology are reported by the
    # standalone physics check.
    #
    # tx_power_dbm is the AGENT-controlled power (10–33 dBm, ±6 dB steps).
    # It still moves SINR dB-for-dB on the node's own wanted signal, but it
    # now ALSO raises I at every co-channel neighbour, so power control is a
    # real trade rather than a free multiplier.
    #
    # SINR_AMBIENT_INTERFERENCE_DB is retained at 0.0 and is DEPRECATED: it
    # was a flat dB pad on the noise floor, superseded by the per-node
    # aggregate term.  Leaving it at 0 keeps it out of the budget; setting it
    # non-zero would double-count interference.
    SINR_AMBIENT_INTERFERENCE_DB = 0.0    # DEPRECATED — see block above
    SINR_MIN_LINK_DIST_M         = 50.0   # near-field guard for FSPL
    SINR_CAP_DB                  = 30.0   # practical CQI ceiling
    SINR_FLOOR_DB                = -10.0  # deep-outage floor

    def _update_dynamic_sinr(self):
        """
        Recompute PHYMACState.sinr_average / sinr_min per node each tick from
        geometry, current tx power AND this tick's aggregate co-channel
        interference (see the two constant blocks above).

        sinr_average: S/(N+I) at the strongest active infra neighbour.
        sinr_min:     cell-edge S/(N+I) at max(neighbour dist, effective
                      coverage radius) — the weakest-UE / coverage proxy.

        One O(N²) interference pass + one pass over links + one over nodes;
        all downstream consumers (auto-CQI, Block A obs, relay feasibility,
        consume_path capacity scaling) pick the values up automatically, and
        identically for every arm.
        """
        trm = self.transport_relay_model
        # Interference first: sinr_average is S/(N+I), so I must be current.
        self._compute_aggregate_interference()

        # Strongest (nearest) active infra neighbour per PHY/MAC node
        best_dist: Dict[str, float] = {}
        for link in self.topology.links.values():
            if not getattr(link, 'is_up', True):
                continue
            a, b = link.endpoints
            for me, peer in ((a, b), (b, a)):
                if me not in self.phy_mac_states:
                    continue
                peer_node = self.topology.nodes.get(peer)
                if (peer_node is None or not peer_node.is_survivor or
                        peer_node.node_type == NodeType.UE):
                    continue
                d = trm.distance_m(me, peer)
                if d is None:
                    continue
                if me not in best_dist or d < best_dist[me]:
                    best_dist[me] = d

        for node_id, ps in self.phy_mac_states.items():
            i_dbm = self.aggregate_interference_dbm(node_id, BAND_ACCESS)
            # Adaptability stress: fold in the external interferer (see
            # __init__).  dB-domain power sum with the current co-channel
            # interference; where the node was thermal-limited (None) the
            # external source becomes the interference term.
            if (self._stress_i_dbm is not None
                    and self.current_tick >= self._stress_i_tick):
                _ext = self._stress_i_dbm
                i_dbm = (_ext if i_dbm is None else
                         10.0 * math.log10(10.0 ** (i_dbm / 10.0)
                                           + 10.0 ** (_ext / 10.0)))
            d = best_dist.get(node_id)
            if d is None:
                # Isolated / unpositioned: serving-link proxy at ~70 % of
                # the coverage radius (mean UE distance in a uniform cell).
                d = 0.7 * ps.effective_coverage_radius_m
            d = max(self.SINR_MIN_LINK_DIST_M, d)
            sinr = trm.access_sinr_db(ps.tx_power_dbm, d, i_dbm)
            ps.sinr_average = max(self.SINR_FLOOR_DB,
                                  min(self.SINR_CAP_DB, sinr))
            # Cell-edge (weakest UE) SINR at the coverage boundary
            d_edge = max(d, ps.effective_coverage_radius_m,
                         self.SINR_MIN_LINK_DIST_M)
            ps.sinr_min = max(
                self.SINR_FLOOR_DB,
                min(self.SINR_CAP_DB,
                    trm.access_sinr_db(ps.tx_power_dbm, d_edge, i_dbm)))

        # One-time sanity line: per-node SINR spread + interference level
        if not getattr(self, '_sinr_spread_logged', False) and self.phy_mac_states:
            vals = [ps.sinr_average for ps in self.phy_mac_states.values()]
            i_vals = [self.aggregate_interference_dbm(n, BAND_ACCESS)
                      for n in self.phy_mac_states]
            i_vals = [v for v in i_vals if v is not None]
            n_floor = ACCESS_FR1.noise_floor_dbm()
            print(f"[SINR] dynamic per-node SINR: min={min(vals):.1f} dB  "
                  f"mean={sum(vals) / len(vals):.1f} dB  max={max(vals):.1f} dB  "
                  f"({len(vals)} infra nodes)")
            if i_vals:
                print(f"[SINR] FR1 aggregate interference: "
                      f"min={min(i_vals):.1f} dBm  "
                      f"mean={sum(i_vals) / len(i_vals):.1f} dBm  "
                      f"max={max(i_vals):.1f} dBm  "
                      f"(thermal floor {n_floor:.1f} dBm -> "
                      f"I/N mean {sum(i_vals) / len(i_vals) - n_floor:+.1f} dB)")
            self._sinr_spread_logged = True

    def _update_phy_mac_observations(self):
        """
        Update PHYMACState OBS fields from current topology/traffic.
        Called once per tick before building agent observations.
        """
        # Beam re-point timers first: a relay link that finished training this
        # tick must be up before SINR/routing/fragment counting look at it.
        self._service_relay_links()

        # Dynamic SINR first: downstream fields (PRB utilisation via
        # access_capacity_mbps) depend on the fresh per-node SINR.
        self._update_dynamic_sinr()

        # Count UEs per O-RU via Uu links
        ue_counts:       Dict[str, int] = {nid: 0 for nid in self.phy_mac_states}
        emrg_counts:     Dict[str, int] = {nid: 0 for nid in self.phy_mac_states}
        backhaul_util:   Dict[str, float] = {nid: 0.0 for nid in self.phy_mac_states}
        backhaul_cap:    Dict[str, float] = {nid: 1000.0 for nid in self.phy_mac_states}
        # UE -> serving infra nodes, over DISTINCT UEs.  ue_counts below counts
        # Uu LINKS (a multi-homed UE contributes several), which is what the
        # per-cell load fields want but is wrong for anything counting PEOPLE —
        # see peer_reachable_ue_stats.
        ue_serving:      Dict[str, list] = {}

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
                        if peer_node.is_survivor:
                            ue_serving.setdefault(peer, []).append(ru_id)
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
        # DISTINCT UEs, not Uu LINKS.  This was `sum(ue_counts.values())`,
        # which counts Uu links: build_comparison_topology attaches each UE to
        # 1-2 nearest O-RUs, so a multi-homed UE contributed 2 and the "fraction
        # of UEs reachable" routinely exceeded 1.0 before the min() clamp — on
        # the Scenario-A topology 150 UEs counted as ~233.  The clamp hid it,
        # and it fed R_coverage at reward weight 0.15 (agent.py:837), so a
        # policy that changed nothing could look fully covering while a third of
        # the population was unattached.  `ue_serving` is already keyed by UE id
        # in the link walk above, so the distinct count is free.
        connected_ues = len(ue_serving)
        self._reachable_ue_fraction = min(1.0, connected_ues / total_ues)
        self._isolated_ue_count     = max(0, total_ues - connected_ues)

        # PEER reachability (disaster-recovery objective, not an access
        # metric): can a UE reach the BODY of the survivor network, or is it
        # camped on a cell inside an island?  Reuses the UE->O-RU map built in
        # the link walk above rather than re-walking every Uu link.  See
        # peer_reachable_ue_stats for the definitions.
        _reach = self.peer_reachable_ue_stats(ue_serving=ue_serving)
        self._peer_reach_stats           = _reach
        self._peer_reachable_ue_fraction = _reach['peer_reachable_fraction']
        self._stranded_ue_count          = _reach['stranded']
        self._reach_restored_ue_ids      = _reach.get('restored_ues', frozenset())
        self._reach_restored_ue_count    = _reach['restored']
        # Freshness stamp.  Only the agent-driven arms reach this method (it is
        # called from _build_agent_observations), so anything that reports these
        # numbers for the routing baselines must check the stamp and recompute
        # rather than read a stale default — see current_peer_reach_stats().
        self._peer_reach_tick            = self.current_tick

        # UE-to-UE routed fraction: relative to all flows (not just attempted)
        total_flows = max(1, len(self.ue_to_ue_flows))
        self._ue_pair_routed_fraction = self.ue_to_ue_success_count / total_flows

        # ── PER-NODE relay ATTRIBUTION (credit assignment) ─────────────────
        # Decompose, per node, the two relay terms the GLOBAL reward already
        # computes fleet-wide:
        #   * how much traffic THIS node's own transport relay links carried
        #     this tick (the local part of "+2.0 per CONTRIBUTING relay" and
        #     of the relay-throughput bonus), and
        #   * whether one of THIS node's own relay links is a member of the
        #     de-duplicated bridge set (the local part of the reunification
        #     and BRIDGE_LOSS_PENALTY terms).
        # link.current_utilization was set by _forward_traffic earlier in the
        # same tick, and distinct_bridge_link_ids() is computed once here and
        # reused, so this costs one pass over the (few) relay links.
        # WHICH BYTES COUNT.  This reads the per-link DELIVERED-UE-TRAFFIC
        # ledger (_link_carried_ue, built by _forward_traffic) rather than
        # link.current_utilization.  The two differ: current_utilization also
        # contains infrastructure-originated telemetry/O&M traffic, which the
        # global reward's delivered fractions do NOT count.  Paying a relay
        # for carrying its own site's telemetry was a way to look
        # "contributing" without moving a single user's byte, so the relay
        # terms are now strictly conditional on traffic the objective
        # actually prices.
        _carried_ledger = getattr(self, '_link_carried_ue', {})
        _relay_carried: Dict[str, float] = {}
        _relay_bridge_nodes: set = set()
        _relay_bridge_count: Dict[str, int] = {}   # per-site counterfactual
        try:
            _bridge_ids = self.distinct_bridge_link_ids()
        except Exception:
            _bridge_ids = set()
        for _lid, _link in self.topology.links.items():
            if getattr(_link, 'link_type', None) != LinkType.TRANSPORT_RELAY:
                continue
            if not getattr(_link, 'is_up', False):
                continue
            _util = float(_carried_ledger.get(_lid, 0.0) or 0.0)
            for _ep in _link.endpoints:
                if _util > 0:
                    _relay_carried[_ep] = _relay_carried.get(_ep, 0.0) + _util
                if _lid in _bridge_ids:
                    _relay_bridge_nodes.add(_ep)
                    _relay_bridge_count[_ep] = _relay_bridge_count.get(_ep, 0) + 1

        # Per-node delivered-volume shares (the local counterpart of the
        # global +100 x delivered_frac / +10 x general_frac terms).
        _share_u2u = getattr(self, '_node_delivery_share_u2u', {})
        _share_gen = getattr(self, '_node_delivery_share_gen', {})
        _raw_u2u   = getattr(self, '_node_carried_u2u', {})
        _raw_gen   = getattr(self, '_node_carried_gen', {})

        # Write into each PHYMACState
        for node_id, ps in self.phy_mac_states.items():
            ps.relay_traffic_carried_mbps = _relay_carried.get(node_id, 0.0)
            ps.relay_link_is_bridge       = node_id in _relay_bridge_nodes
            ps.bridge_counterfactual_fragments = _relay_bridge_count.get(node_id, 0)
            ps.delivered_share_ue_to_ue   = _share_u2u.get(node_id, 0.0)
            ps.delivered_share_general    = _share_gen.get(node_id, 0.0)
            ps.carried_ue_traffic_mbps    = (_raw_u2u.get(node_id, 0.0)
                                             + _raw_gen.get(node_id, 0.0))
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

    # ── MultiHaul beam-steering mechanics ─────────────────────────────────
    #
    # WHICH RADIOS THESE CONSTANTS APPLY TO (design decision 1: hybrid
    # transport).  ONLY the 60 GHz TG MESH class is steerable, so every
    # constant in this block applies to the TG class alone:
    #
    #   * TG mesh (LinkType.MULTIHAUL_MESH, sites with has_multihaul) is an
    #     ELECTRONICALLY steered phased array.  It can be re-pointed at any
    #     peer inside its ~1.07 km (23 dBm) / ~1.53 km (33 dBm) clear-air
    #     budget, and pays RELAY_REPOINT_TICKS of beam-training outage each
    #     time it is.  This is the only class that may form a NEW bridge.
    #   * MW PtP (LinkType.MICROWAVE_PTP) is MECHANICALLY aligned.  It is
    #     NOT re-pointable inside an episode at any cost — realigning a
    #     0.3 m parabola is a truck roll with an alignment crew — so it never
    #     enters the bridging branch, never consumes RELAY_REPOINT_TICKS, and
    #     appears only in the Priority-2 "repurpose an existing hop" branch,
    #     where agents may bring it up/down but never aim it.
    #
    # Two costs keep TG steering from being free, and both are what make a
    # "sustained bridge" a meaningful thing to learn:
    #
    #   RELAY_REPOINT_TICKS    beam slew + re-acquisition + link training.
    #       A newly pointed relay link exists but is DOWN for this many ticks
    #       (it carries no traffic and does not reduce the fragment count).
    #       3 ticks @ 1 s/tick is a deliberately generous stand-in for
    #       mechanical/phased re-point plus TG link-up.
    #   RELAY_BRIDGE_GRACE_TICKS + RELAY_BRIDGE_BUSY_UTIL  the teardown
    #       protection for a link that is currently the ONLY path between two
    #       fragments (see _relay_teardown_blocked for the full rationale).
    #
    # TEARDOWN PROTECTION — WHY IT IS NOT A BLANKET TIMER ANY MORE.
    #
    # This used to be `RELAY_BRIDGE_HOLD_TICKS = 20`: ANY sole-path relay link
    # was locked for 20 ticks after it came up, regardless of whether it was
    # carrying a single bit.  That number had no physical referent — it was
    # chosen because it produced sustained bridges — and it was strong enough
    # to do the reunification work BY ITSELF: a uniformly random policy that
    # happens to sample CAPACITY_BOOST once at a MultiHaul site locked the
    # resulting bridge for >= 23 ticks (3 beam-training + 20 hold), so the
    # island reunified without anything being learned.  A hysteresis that can
    # manufacture the headline result is not admissible evidence, so it is
    # replaced by the two effects that ARE physically defensible:
    #
    #   1. BEAM-TRAINING LOCKOUT (RELAY_REPOINT_TICKS).  A radio that is
    #      mid-acquisition physically cannot be re-pointed.  This is not a
    #      policy choice, it is the same 3 ticks already charged as outage.
    #   2. MAKE-BEFORE-BREAK ON A LOADED LINK (RELAY_BRIDGE_BUSY_UTIL).  A
    #      transport link that is actually carrying traffic is not torn down
    #      under it; that is ordinary operational practice (graceful
    #      shutdown / IS-IS overload bit / MPLS make-before-break), and it is
    #      re-evaluated EVERY tick from measured utilisation — the moment the
    #      link goes idle it is free to be dropped.  It is a load condition,
    #      not a timer, so it cannot lock an unused link at all.
    #   3. A SHORT GRACE PERIOD (RELAY_BRIDGE_GRACE_TICKS) equal to the
    #      re-point cost.  Rationale: utilisation is only observable AFTER a
    #      forwarding pass, so a link that has just come up has no measured
    #      load yet; giving it as many ticks as re-pointing costs is the
    #      minimum window in which "is this link useful?" is answerable at
    #      all, and it is symmetric with the cost the policy already pays.
    #
    # Worst-case protection for an IDLE bridge is now 3 (training, during
    # which it is down anyway) + 3 (grace) = 6 ticks, versus 23 before.  A
    # bridge that survives longer than that survives because it is carrying
    # traffic — i.e. because the policy put it somewhere useful.
    RELAY_REPOINT_TICKS      = 3
    RELAY_BRIDGE_GRACE_TICKS = 3      # == RELAY_REPOINT_TICKS by construction
    RELAY_BRIDGE_BUSY_UTIL   = 0.02   # fraction of link capacity counted as
                                      # "actively carrying traffic"

    def _service_relay_links(self):
        """Bring re-pointed relay links up once their beam-training timer
        expires.  Called once per tick (from _update_phy_mac_observations),
        so it runs identically for every arm."""
        changed = False
        for lid, ready in list(self._relay_link_ready_tick.items()):
            link = self.topology.links.get(lid)
            if link is None:
                self._relay_link_ready_tick.pop(lid, None)
                continue
            if self.current_tick >= ready:
                link.is_up = True
                self.topology.graph.add_edge(
                    link.endpoints[0], link.endpoints[1],
                    link_id=lid, capacity=link.capacity)
                self._relay_link_ready_tick.pop(lid, None)
                changed = True
        if changed:
            self.topology.invalidate_infrastructure_cache()

        # Stamp the tick at which each link BECAME a cross-fragment bridge, so
        # the RELAY_BRIDGE_GRACE_TICKS grace period measures dwell from link-up
        # rather than from the first teardown attempt.  Also forget stamps for
        # links that no longer bridge (e.g. the underlying fragments merged by
        # another route), so the hold cannot outlive its justification.
        if self._relay_bridge_since or self.active_relay_links():
            bridges = self.bridging_relay_link_ids()
            for lid in bridges:
                self._relay_bridge_since.setdefault(lid, self.current_tick)
            for lid in list(self._relay_bridge_since):
                if lid not in bridges:
                    self._relay_bridge_since.pop(lid, None)

    def relay_link_load_fraction(self, link_id: str) -> float:
        """Utilisation fraction of a relay link on the LAST completed
        forwarding pass (see _snapshot_relay_link_load).

        A live reading of `link.current_utilization` is NOT usable here: the
        two tick orderings in this codebase differ — the engine's own run()
        loop forwards traffic before executing agent actions, while the
        evaluation harness (run_timeline_comparison.py) executes actions
        first and forwards afterwards, and both reset utilisation at the top
        of the tick.  The snapshot gives every caller the same well-defined
        quantity: "what this link carried the last time traffic was actually
        forwarded".
        """
        return float(getattr(self, '_relay_link_load', {}).get(link_id, 0.0))

    def _snapshot_relay_link_load(self):
        """Record per-relay-link utilisation fraction after a forwarding pass.

        Called at the end of _forward_traffic so it runs identically for
        every arm and both tick orderings.  Entries for links that no longer
        exist are dropped so a stale 'busy' reading can never protect a link
        that was re-created later under the same id.
        """
        loads = {}
        for lid, lnk in self.topology.links.items():
            if getattr(lnk, 'link_type', None) is not LinkType.TRANSPORT_RELAY:
                continue
            cap = float(getattr(lnk, 'capacity', 0.0) or 0.0)
            if cap <= 0.0:
                continue
            loads[lid] = min(1.0,
                             float(getattr(lnk, 'current_utilization', 0.0)) / cap)
        self._relay_link_load = loads

    def _relay_teardown_blocked(self, node_id: str) -> bool:
        """True if this node holds a relay link that physically must not be
        dropped this tick.

        Three protections, each with a physical referent (the full rationale
        and the history of the 20-tick blanket hold it replaced are in the
        RELAY_REPOINT_TICKS / RELAY_BRIDGE_* constant block above):

          1. BEAM-TRAINING LOCKOUT.  A link still inside its
             RELAY_REPOINT_TICKS window is created is_up=False, so it is
             invisible to active_relay_links() and therefore to
             bridging_relay_link_ids() — which meant a freshly pointed bridge
             had ZERO protection during the exact 3 ticks it needs to survive
             in order to become a bridge at all.  Measured effect: the policy
             would create a cross-fragment link and flip its relay head on the
             next tick, destroying the beam mid-acquisition, over and over —
             many relay links, never a bridge.  A radio that is mid-acquisition
             physically cannot be re-pointed, so this protection starts at
             creation, not at link-up.
          2. GRACE PERIOD.  A NON-REDUNDANT bridge that has been up for fewer
             than RELAY_BRIDGE_GRACE_TICKS ticks (== the re-point cost) is
             held: its usefulness is not yet measurable.
          3. MAKE-BEFORE-BREAK.  A NON-REDUNDANT bridge whose last measured
             utilisation is at least RELAY_BRIDGE_BUSY_UTIL of its capacity is
             held while it stays loaded, and released the tick it goes idle.

        "NON-REDUNDANT" means the link is a tree edge of the relay merge
        forest (see distinct_bridge_link_ids): a second relay link joining a
        pair of fragments that some other relay link already joins is not a
        sole path, carries no unique connectivity, and gets no protection.
        """
        try:
            if any(node_id in getattr(self.topology.links.get(lid), 'endpoints', ())
                   for lid in self._relay_link_ready_tick):
                return True
            links = [l for l in self.active_relay_links()
                     if node_id in l.endpoints]
            if not links:
                return False
            # Only genuinely sole-path (non-redundant) bridges are protected.
            # Cheap — only runs on a relay-mode flip.
            protected = self.distinct_bridge_link_ids()
            for l in links:
                if l.id not in protected:
                    continue
                since = self._relay_bridge_since.get(l.id, self.current_tick)
                if self.current_tick - since < self.RELAY_BRIDGE_GRACE_TICKS:
                    return True    # grace: usefulness not yet observable
                if (self.relay_link_load_fraction(l.id)
                        >= self.RELAY_BRIDGE_BUSY_UTIL):
                    return True    # make-before-break: carrying live traffic
            return False
        except Exception:
            return False

    def _rank_bridge_candidates(self, node_id, candidates, infra_graph):
        """Fragment-aware re-ranking of relay peer candidates (beam steering).

        `candidates` is TransportRelayModel.find_relay_candidates output —
        (peer, capacity_mbps, sinr_db) sorted by capacity.  All of them are
        already outside this node's component; capacity alone cannot see two
        things that decide whether the island reunifies:

          1. two MultiHaul sites re-pointing at the SAME neighbouring island
             merge 4 fragments into 3 instead of 2;
          2. picking a fragment that OTHER sites could also have reached
             strands a fragment that only this site can reach (measured: on
             seed 53 pure capacity-first steering plateaus at 2 components
             although 1 is feasible).

        Sort key — MOST-CONSTRAINED-FIRST:

            (peer's component already joined by an active/pending bridge?,
             how many other steerable sites can also reach that component,
             -capacity,
             site-decorrelated deterministic rank of the target component,
             target component anchor)

        so an unclaimed fragment beats a claimed one, a fragment only this
        site can reach beats one everybody can reach, capacity breaks the
        next tie, and the last two keys resolve the rest.  Returns the
        re-ordered list.

        WHY THE LAST TWO KEYS EXIST (greedy self-collision).  `-capacity` was
        the final key, and it discriminates far less than it looks like it
        does: TransportRelayModel.can_form_relay_link returns

            cap = BANDWIDTH_MHZ * 1e6 * se * relay_bw_fraction / 1e6

        where `se` comes from the 5-entry MCS_SPECTRAL_EFFICIENCY table and
        `relay_bw_fraction` is fixed for the whole call.  Capacity is
        therefore QUANTISED to at most 5 distinct values per site, and every
        candidate that clears the same MCS threshold has a bit-identical
        capacity.  Exact ties are the normal case, not the exception.  On a
        tie `sorted` is stable, so the order fell through to the order
        find_relay_candidates produced, which is `topology.nodes.items()`
        order — IDENTICAL for every site.  Two MultiHaul sites in the same
        tier therefore preferred the SAME target component.  With only 3
        MultiHaul sites on the Scenario-A topology, one such collision is the
        difference between a full spanning assignment and a permanently
        stranded fragment: the union-find de-duplication in
        distinct_bridge_link_ids correctly counts the redundant second link as
        0 bridges, so the collision shows up as "3 sites in CAPACITY_BOOST,
        2 bridges, 1 fragment still stranded".

        The `claimed` tier already suppresses this whenever the sites are
        stepped SEQUENTIALLY within a tick and the first site's link
        registers (active, or pending in `_relay_link_ready_tick`).  The two
        new keys close the remaining cases — the first site's link creation
        being blocked, ranking falling back on an exception, or sites in
        different components sharing a tier — by giving each site a DIFFERENT
        deterministic preference order over equally-good targets:

            rank = crc32(f"{node_id}|{anchor}")

        keyed on the component's stable anchor (its smallest node id) rather
        than its traversal-order component id, and on the site's own id, so
        it is reproducible across processes (unlike the salted builtin
        `hash`), stable across ticks, and decorrelated between sites.  This is
        a COORDINATION fix in the engine, symmetric across every arm that
        forms relay links: nothing about it is MARL-specific and it does not
        change WHICH modes are available or when they fire.
        """
        try:
            import networkx as _nx_rank
            # Component id per node on the CURRENT infra graph (which already
            # includes any relay link that is up)
            comp_of = {}
            for i, comp in enumerate(_nx_rank.connected_components(infra_graph)):
                for nid in comp:
                    comp_of[nid] = i
            # Components already reached by an existing active relay bridge
            claimed = set()
            for l in self.active_relay_links():
                for ep in l.endpoints:
                    if ep in comp_of:
                        claimed.add(comp_of[ep])
            # Also claim the components pending a re-point (link created but
            # still training) so two sites don't race for the same island.
            for lid in self._relay_link_ready_tick:
                lk = self.topology.links.get(lid)
                if lk is None:
                    continue
                for ep in lk.endpoints:
                    if ep in comp_of:
                        claimed.add(comp_of[ep])
            my_comp = comp_of.get(node_id)
            claimed.discard(my_comp)

            # Scarcity: for each candidate component, how many OTHER
            # steerable (MultiHaul, not already linked) sites could reach it?
            cand_comps = {comp_of.get(p) for p, _c, _s in candidates}
            cand_comps.discard(None)
            others = [
                nid for nid, n in self.topology.nodes.items()
                if nid != node_id and getattr(n, 'is_survivor', False)
                and getattr(n, 'has_multihaul', False)
                and nid in self.phy_mac_states
                and not getattr(self.phy_mac_states[nid], 'relay_link_active', False)
            ]
            scarcity = {c: 0 for c in cand_comps}
            for o in others:
                oc = comp_of.get(o)
                o_ps = self.phy_mac_states[o]
                seen = set()
                for peer, pc in comp_of.items():
                    if pc == oc or pc not in scarcity or pc in seen:
                        continue
                    ok, _cap, _s = self.transport_relay_model.can_form_relay_link(
                        o, peer, self._transport_tx_dbm(o_ps),
                        max(0.15, getattr(o_ps, 'prb_relay_fraction', 0.15)),
                        self.aggregate_interference_dbm(peer, BAND_TG))
                    if ok:
                        seen.add(pc)
                for pc in seen:
                    scarcity[pc] += 1

            # Stable identity per candidate component: its smallest node id.
            # Component ids from `enumerate(connected_components(...))` are
            # traversal-order labels and must never be used as a tie-break.
            anchor_of: Dict[int, str] = {}
            for nid, c in comp_of.items():
                if c in cand_comps:
                    prev = anchor_of.get(c)
                    if prev is None or nid < prev:
                        anchor_of[c] = nid

            import zlib as _zlib
            _rank_cache: Dict[str, int] = {}

            def _site_rank(anchor: str) -> int:
                """Deterministic, site-decorrelated rank for a component.

                crc32 (not the builtin `hash`, which is salted per process)
                so the same seed replays identically across runs.
                """
                r = _rank_cache.get(anchor)
                if r is None:
                    r = _zlib.crc32(f"{node_id}|{anchor}".encode('utf-8'))
                    _rank_cache[anchor] = r
                return r

            def _key(c):
                peer, cap, _sinr = c
                pc = comp_of.get(peer)
                anchor = anchor_of.get(pc, '')
                return (1 if pc in claimed else 0, scarcity.get(pc, 0), -cap,
                        _site_rank(anchor), anchor)

            return sorted(candidates, key=_key)
        except Exception:
            return candidates

    def _step_phy_mac(self, node_id: str, action: 'PHYMACAction'):
        """
        Apply a PHYMACAction to a node's PHYMACState and update the topology
        if relay mode changes (create / tear down transport relay links).

        MCS semantics: the agent's mcs_*_idx heads are BOUNDED OFFSETS from
        the auto CQI→MCS link adaptation choice (3GPP TS 38.214), not
        absolute MCS indices:

            final_mcs_idx = clamp(auto_cqi_idx + (head_idx - MCS_OFFSET_CENTER),
                                  0, len(MCSLevel) - 1)

        Head index MCS_OFFSET_CENTER (offset 0) means "follow auto-CQI
        exactly"; negative offsets pick a more robust MCS, positive offsets
        a more aggressive one.  This gives the policy a physically sane
        default on ANY topology and makes deliberate deviations (conservative
        under interference, aggressive when queues build) the learned signal.

        If the resulting MCS is too aggressive for the current SINR the link
        does NOT collapse to zero: PHYMACState.spectral_efficiency applies
        an OLLA-style step-down to the highest feasible MCS, and the event
        is counted in ps.mcs_fallback_events (diagnostics).
        """
        ps = self.phy_mac_states.get(node_id)
        if ps is None:
            return

        prev_relay = ps.relay_mode

        # ── Apply discrete controls ────────────────────────────────────────
        ps.apply_power_step(action.tx_power_step)
        # MCS heads: bounded offset from the auto-CQI (SINR-appropriate) MCS
        _n_mcs    = len(MCSLevel)
        _auto_idx = auto_cqi_mcs_idx(ps.sinr_average)
        _e_idx = max(0, min(_n_mcs - 1,
                            _auto_idx + (action.mcs_emergency_idx - MCS_OFFSET_CENTER)))
        _g_idx = max(0, min(_n_mcs - 1,
                            _auto_idx + (action.mcs_general_idx - MCS_OFFSET_CENTER)))
        ps.mcs_emergency = list(MCSLevel)[_e_idx]
        ps.mcs_general   = list(MCSLevel)[_g_idx]
        # OLLA fallback diagnostics: count ticks where the applied MCS
        # exceeds what the current SINR supports (spectral_efficiency will
        # step down gracefully instead of returning 0 — see
        # PHYMACState.effective_mcs).  Once per tick per node.
        if (ps.effective_mcs(for_emergency=False)[1] or
                ps.effective_mcs(for_emergency=True)[1]):
            ps.mcs_fallback_events += 1
        ps.relay_mode    = list(RelayMode)[action.relay_mode_idx]
        ps.mac_scheduler = list(MACScheduler)[action.scheduler_idx]
        ps.handover_triggered = bool(action.handover_idx)

        # PRB allocation (already normalised to sum=1 by softmax in net)
        ps.prb_emergency_fraction = action.prb_emergency_frac
        ps.prb_relay_fraction     = action.prb_relay_frac
        ps.prb_general_fraction   = action.prb_general_frac
        ps.normalise_prb()
        # Relay-link state reconciliation (MARL_RELAY_STATE): clear a phantom
        # "active" flag whose link no longer exists, so the site can bridge.
        if self._relay_state_reconcile and ps.relay_link_active:
            if not any(node_id in getattr(l, 'endpoints', ())
                       for l in self.topology.links.values()
                       if getattr(l, 'link_type', None) == LinkType.TRANSPORT_RELAY):
                ps.relay_link_active = False
                ps.relay_peer_node = None
                ps.relay_link_capacity_mbps = 0.0
                self._diag_phantom_resets += 1

        # ── Transport relay mode change ─────────────────────────────────────────
        #
        # TEARDOWN PROTECTION (see _relay_teardown_blocked and the
        # RELAY_BRIDGE_* constant block): a relay link that currently joins
        # two otherwise-disconnected fragments is the only thing carrying
        # cross-island traffic.  Letting a single relay-head flip tear it down
        # mid beam-training destroyed bridges before they could exist at all.
        # The protection is now physically motivated and NARROW — beam-training
        # lockout, a grace period equal to the re-point cost, and
        # make-before-break while the link is measurably loaded — rather than
        # a blanket 20-tick lock on any sole-path link.  An idle bridge is
        # droppable within ~6 ticks of being pointed.
        if prev_relay != ps.relay_mode:
            if self._relay_teardown_blocked(node_id):
                # HARDWARE LOCKOUT: the beam cannot be re-pointed until the
                # hold expires, so the relay mode is held with it.  Reverting
                # the mode (rather than only the link) keeps PRB and energy
                # accounting honest — a node cannot hold a live bridge while
                # claiming to have switched its radio out of relay mode — and
                # keeps the churn penalty consistent, since nothing physically
                # changed this tick.
                ps.relay_mode = prev_relay
            else:
                # Tear down existing transport relay links from this node
                removed_ids = self.transport_relay_model.clear_node_links(node_id)
                for rid in removed_ids:
                    # Remove corresponding topology link
                    self.topology.links.pop(rid, None)
                    self._relay_link_ready_tick.pop(rid, None)
                    self._relay_bridge_since.pop(rid, None)
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

        # Fragment-aware re-pointing (MARL_RELAY_REPOINT): may release a
        # non-bridge link so that the formation branch below can bridge.
        if (self._relay_repoint and self.island_mode
                and ps.relay_mode == RelayMode.CAPACITY_BOOST
                and ps.relay_link_active):
            self._maybe_repoint_to_fragment(node_id, ps)

        if ps.relay_mode in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST) and not ps.relay_link_active:
            # ── MultiHaul Self-Healing & Transport Optimization ───────────
            #
            # CAPACITY_BOOST (MultiHaul beam re-steering):
            #   Priority 1: Bridge disconnected fragments by discovering
            #   physically nearby survivor nodes in different connected
            #   components. This is the key MARL advantage -- dynamic beam
            #   steering to restore post-severance connectivity.
            #   Priority 2: If no cross-fragment target, boost capacity
            #   on existing MultiHaul mesh link.
            #
            # LOCAL_REROUTE (Transport repurposing):
            #   Repurpose existing link capacity for east-west traffic.
            #   Works on any link type (MW PtP, fiber, MultiHaul).
            #
            node = self.topology.nodes.get(node_id)
            has_mh = node.has_multihaul if node else False

            # ── Priority 1 (CAPACITY_BOOST + MultiHaul): Bridge fragments ──
            #
            # BEAM STEERING IS THE POINT.  A MultiHaul site can re-point to
            # ANY peer inside its link budget, so the peer choice — not just
            # relay on/off — decides whether the island reunifies.  The peer
            # is selected ENGINE-SIDE (rather than as an extra policy head) so
            # that existing checkpoints keep their action-space shape; the
            # selection is made FRAGMENT-AWARE here, which is what actually
            # matters for reunification:
            #
            #   1. only peers in a DIFFERENT current component are considered
            #      (a bridge inside one's own component adds capacity but zero
            #      reunification);
            #   2. among those, peers in a component NOT already joined by
            #      another active bridge win — otherwise two MultiHaul sites
            #      both re-point at the same neighbouring island and 4
            #      fragments collapse to 3 instead of 2;
            #   3. capacity (i.e. SINR / distance) breaks the remaining tie.
            #
            # The agent still controls WHETHER to relay, its Tx power and its
            # relay PRB share, all three of which move the link budget and so
            # the set of reachable peers.
            bridged = False
            if ps.relay_mode == RelayMode.CAPACITY_BOOST and has_mh and self.island_mode:
                import networkx as _nx_relay
                # Find which INFRASTRUCTURE component this node belongs to
                # (Must use infra-only graph — UE nodes mask fragmentation)
                try:
                    infra_graph = self.topology._build_infrastructure_graph()
                    if node_id in infra_graph:
                        my_component = _nx_relay.node_connected_component(infra_graph, node_id)
                    else:
                        my_component = {node_id}
                except Exception:
                    my_component = {node_id}

                # Find nearby survivor infra nodes NOT in our component
                cross_fragment_candidates = [
                    nid for nid, n in self.topology.nodes.items()
                    if n.is_survivor
                    and nid not in my_component
                    and n.node_type != NodeType.UE
                    and nid in infra_graph
                ]

                if cross_fragment_candidates:
                    # Use physics-based relay feasibility check
                    for nid_c in [node_id] + cross_fragment_candidates:
                        n_c = self.topology.nodes.get(nid_c)
                        if n_c and not self.transport_relay_model.node_positions.get(nid_c):
                            self.transport_relay_model.register_position(
                                nid_c, n_c.x_pos, n_c.y_pos)

                    relay_candidates = self.transport_relay_model.find_relay_candidates(
                        source=node_id,
                        candidate_nodes=cross_fragment_candidates,
                        tx_power_dbm=self._transport_tx_dbm(ps),
                        relay_bw_fraction=max(0.15, ps.prb_relay_fraction),
                        # Per-RECEIVER aggregate 60 GHz interference: the peer
                        # is the receiver, so I is evaluated at the peer.  In
                        # practice 60 GHz mesh is noise-limited (pencil-beam
                        # discrimination + oxygen absorption), but the term is
                        # applied exactly rather than assumed negligible.
                        interference_dbm=lambda peer: (
                            self.aggregate_interference_dbm(peer, BAND_TG)),
                        # NO truncation before the fragment-aware re-ranking.
                        # find_relay_candidates sorts by capacity, i.e. by
                        # proximity, so any small cap (the old value was 3)
                        # silently drops every peer in the FAR fragments — the
                        # large nearby component fills the whole window and the
                        # distant singleton islands become invisible.  That was
                        # measured to cost a full component on seed 53.
                        max_candidates=len(cross_fragment_candidates)
                    )

                    if relay_candidates:
                        relay_candidates = self._rank_bridge_candidates(
                            node_id, relay_candidates, infra_graph)
                        best_target, best_cap, best_sinr = relay_candidates[0]
                        # UNPHYSICAL MULTIPLIER REMOVED.  This used to be
                        #     peer_mh = peer_node.has_multihaul
                        #     boost   = 1.5 if peer_mh else 1.0
                        #     relay_cap = best_cap * boost
                        # i.e. a bare 1.5x factor layered ON TOP of a capacity
                        # that can_form_relay_link had already derived from the
                        # full 60 GHz link budget (FSPL + oxygen + rain -> SINR
                        # -> MCS -> BW x SE x relay share).  Nothing in the
                        # budget was left unaccounted for, so the multiplier was
                        # pure inflation: it made a TG-to-TG bridge 50 % better
                        # than physics allows and it applied to every arm that
                        # bridges, so it inflated the whole comparison.  The
                        # derived budget is now used as-is.
                        relay_cap = best_cap

                        new_link_id = self.transport_relay_model.create_link(
                            node_id, best_target, relay_cap, best_sinr)
                        # RE-POINT COST: the beam needs RELAY_REPOINT_TICKS to
                        # slew, re-acquire and re-train before it carries
                        # traffic.  The link is created DOWN and brought up by
                        # _service_relay_links() once the timer expires, so a
                        # policy that thrashes beams pays real outage instead
                        # of re-pointing for free.
                        self.topology.links[new_link_id] = Link(
                            id=new_link_id,
                            endpoints=(node_id, best_target),
                            capacity=relay_cap,
                            latency=3,
                            link_type=LinkType.TRANSPORT_RELAY,
                            interface_type=InterfaceType.BACKHAUL,
                            is_up=(self.RELAY_REPOINT_TICKS <= 0),
                            current_utilization=0.0
                        )
                        if self.RELAY_REPOINT_TICKS <= 0:
                            self.topology.graph.add_edge(
                                node_id, best_target,
                                link_id=new_link_id, capacity=relay_cap)
                        else:
                            self._relay_link_ready_tick[new_link_id] = (
                                self.current_tick + self.RELAY_REPOINT_TICKS)
                        ps.relay_link_active = True
                        ps.relay_peer_node = best_target
                        ps.relay_link_capacity_mbps = relay_cap
                        self.topology.invalidate_infrastructure_cache()
                        bridged = True

                        if self.config.verbose:
                            print(f"[MULTIHAUL BRIDGE] {node_id} -> {best_target}: "
                                  f"{relay_cap:.1f} Mbps (SINR={best_sinr:.1f}dB) "
                                  f"-- cross-fragment beam-steer "
                                  f"(re-point {self.RELAY_REPOINT_TICKS} ticks)")

            # ── Priority 2 / LOCAL_REROUTE: Boost existing links ──────────
            if not bridged and not ps.relay_link_active:
                # BRIDGE DOUBLE-COUNTING FIX.  This branch repurposes capacity
                # on an EXISTING PHYSICAL link, so the scan must skip
                # TRANSPORT_RELAY links (they are themselves relay products,
                # not underlying transport).  Without the skip, the FAR
                # endpoint of a fresh cross-fragment bridge saw that bridge as
                # an ordinary "existing neighbour" and pointed a second relay
                # link straight back across the same cut.  That link:
                #   * added no connectivity (the cut was already bridged),
                #   * inflated the bridge counter — 4 "bridges" were reported
                #     on a topology with only 3 MultiHaul sites, and
                #   * silently overwrote the A–B edge attributes in the
                #     undirected routing graph (same node pair, new link_id).
                # Peers already reachable over a relay link are excluded for
                # the same reason: a parallel relay to the same peer is pure
                # duplication.
                _relay_peers = {
                    (l.endpoints[1] if l.endpoints[0] == node_id else l.endpoints[0])
                    for l in self.topology.links.values()
                    if getattr(l, 'link_type', None) is LinkType.TRANSPORT_RELAY
                    and getattr(l, 'is_up', True) and node_id in l.endpoints
                }
                existing_neighbours = []
                for lid, link in self.topology.links.items():
                    ep = link.endpoints
                    if link.link_type == LinkType.TRANSPORT_RELAY:
                        continue
                    if node_id in ep and link.is_up:
                        peer = ep[1] if ep[0] == node_id else ep[0]
                        if peer in _relay_peers:
                            continue
                        peer_node = self.topology.nodes.get(peer)
                        if peer_node and peer_node.is_survivor:
                            is_multihaul_link = link.link_type == LinkType.MULTIHAUL_MESH
                            existing_neighbours.append((peer, lid, link, is_multihaul_link))

                if existing_neighbours:
                    if ps.relay_mode == RelayMode.CAPACITY_BOOST and has_mh:
                        mh_neighbours = [(p, lid, l, mh) for p, lid, l, mh in existing_neighbours if mh]
                        target_list = mh_neighbours if mh_neighbours else existing_neighbours
                    else:
                        target_list = existing_neighbours

                    target_list.sort(key=lambda x: x[2].capacity, reverse=True)
                    best_peer, best_lid, best_link, is_mh = target_list[0]

                    # ── RELAY CAPACITY ON AN EXISTING HOP ──────────────────
                    #
                    # No beam is re-pointed here: this branch REPURPOSES a
                    # share of an already-aligned hop for east-west traffic,
                    # so it costs no RELAY_REPOINT_TICKS (that cost belongs to
                    # the steerable TG class in the Priority-1 branch above)
                    # and it is legal on an MW PtP hop, which may be brought
                    # up/down but never aimed at a new peer.
                    #
                    # The capacity is DERIVED FROM THE HOP'S OWN RADIO CLASS,
                    # not from a flat multiplier.  What used to be here was
                    #     boost_factor = 1.5 if (CAPACITY_BOOST and MultiHaul)
                    #     relay_cap = link.capacity * boost_factor * share
                    # justified in-comment as "1.5x boost ... rounded up for
                    # spatial multiplexing" — a 50 % capacity gift with no link
                    # budget behind it, granted to every arm that relays.  Now:
                    #   MULTIHAUL_MESH -> 60 GHz TG budget at the true hop
                    #                     length (FSPL + O2 + rain -> SINR ->
                    #                     MCS, capped at the TG modem ceiling)
                    #   MICROWAVE_PTP  -> 18 GHz MW budget at the true hop
                    #                     length, capped by the licensed
                    #                     channel width
                    #   FIBER / other  -> a share of the provisioned capacity
                    #                     (no radio budget applies)
                    # and in every case it is bounded by the hop's provisioned
                    # capacity, since a relay cannot conjure more than the hop
                    # physically carries.
                    _share = max(0.2, ps.prb_relay_fraction)
                    _lt = getattr(best_link, 'link_type', None)
                    _hop_d = self.transport_relay_model.distance_m(node_id, best_peer)
                    _derived = None
                    if _hop_d is not None and _lt is LinkType.MULTIHAUL_MESH:
                        _ok, _c, _s = self.transport_relay_model.can_form_relay_link(
                            node_id, best_peer, self._transport_tx_dbm(ps), _share,
                            self.aggregate_interference_dbm(best_peer, BAND_TG))
                        _derived = _c if _ok else 0.0
                    elif _hop_d is not None and _lt in (LinkType.MICROWAVE_PTP,
                                                        LinkType.MICROWAVE):
                        _ok, _c, _s = self.transport_relay_model.mw_link_feasible(
                            node_id, best_peer, self._transport_tx_dbm(ps), _share,
                            self.aggregate_interference_dbm(best_peer, BAND_MW))
                        _derived = _c if _ok else 0.0
                    if _derived is None:
                        _derived = best_link.capacity * _share     # fibre etc.
                    relay_cap = min(_derived, best_link.capacity)

                    # ZERO-CAPACITY GUARD.  A derived budget CAN now come back
                    # at 0 (e.g. a MULTIHAUL_MESH hop longer than the 60 GHz
                    # reach, which the old flat multiplier could never
                    # produce).  Creating the link anyway would add a
                    # zero-capacity edge to the routing graph, which carries no
                    # traffic yet counts as CONNECTIVITY — it would merge two
                    # components in the infra graph and deflate the reported
                    # fragment count for free.  A hop whose budget does not
                    # close simply does not come up.
                    if relay_cap > 0.0:
                        new_link_id = self.transport_relay_model.create_link(
                            node_id, best_peer, relay_cap, 15.0)
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
                        # Add to topology graph so traffic routing can use it
                        self.topology.graph.add_edge(
                            node_id, best_peer,
                            link_id=new_link_id, capacity=relay_cap)
                        ps.relay_link_active = True
                        ps.relay_peer_node = best_peer
                        ps.relay_link_capacity_mbps = relay_cap
                        self.topology.invalidate_infrastructure_cache()

                        if self.config.verbose:
                            mode_str = ("BOOST"
                                        if ps.relay_mode == RelayMode.CAPACITY_BOOST
                                        else "REROUTE")
                            link_str = "MultiHaul-TG" if is_mh else "MW-PtP"
                            print(f"[{mode_str}] {node_id} -> {best_peer}: "
                                  f"{relay_cap:.1f} Mbps via {link_str} "
                                  f"(derived from the hop's own link budget)")
                    elif self.config.verbose:
                        print(f"[RELAY-INFEASIBLE] {node_id} -> {best_peer}: "
                              f"{_lt} hop of {(_hop_d or 0.0):.0f} m does not "
                              f"close its link budget at "
                              f"{ps.tx_power_dbm:.0f} dBm")


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
        cs.potential_iab_capacity_norm = min(1.0, relay_cap / 100.0)

        # Bridge node flag (per-tick cached set — free to query)
        cs.bridge_node_flag = 1.0 if node_id in self._tick_bridge_set_cache else 0.0

        # Relay capability: does THIS site own a steerable MultiHaul radio?
        # This is the only observable that distinguishes the handful of nodes
        # where relay_mode=CAPACITY_BOOST can re-point a beam across a
        # fragment boundary (see _step_phy_mac: the bridging branch is gated
        # on `has_mh`) from every other node, where the same action costs PRB
        # and achieves nothing.  It is a static hardware property of the site
        # — exactly the kind of thing a real O-RU knows about itself from its
        # own inventory — not a hint about what to do with it.
        cs.relay_capable = 1.0 if getattr(node, 'has_multihaul', False) else 0.0

        # UE min RSRP proxy (SINR_min normalised)
        cs.ue_rsrp_min_norm = max(0.0, min(1.0, (ps.sinr_min + 5.0) / 35.0))

        # Core distance: hop count to nearest live core node (per-tick BFS cache).
        # 1.0 = core unreachable (island); 10+ hops saturates.
        core_d = self._tick_core_dist_cache.get(node_id)
        cs.core_distance_norm = 1.0 if core_d is None else min(1.0, core_d / 10.0)

        # Handover candidates: live infra neighbours in the routing graph
        ho_candidates = 0
        if node_id in self.topology.graph:
            for nb in self.topology.graph.neighbors(node_id):
                nb_node = self.topology.nodes.get(nb)
                if (nb_node is not None and nb_node.is_survivor and
                        nb_node.node_type != NodeType.UE):
                    ho_candidates += 1
        cs.handover_candidate_count_norm = min(1.0, ho_candidates / 10.0)

        # UE-pair demand anchored at this node (per-tick cache from flow set)
        cs.ue_pair_demand_norm = min(
            1.0, self._tick_ue_demand_cache.get(node_id, 0) / 20.0)

        return cs

    # ── Fragment coordination (MARL_POSTCARD_FRAGMENTS) ────────────────────
    FRAG_CUE_LATCH_TICKS = 30      # three DCC postcard periods

    def _fragment_labels_now(self) -> Dict[str, str]:
        """Stable per-node fragment labels for THIS tick (cached per tick).

        Uses the relay-INCLUSIVE infrastructure components and represents
        each component by its smallest node id, so a label sent in a
        postcard at tick t still means the same fragment when compared at
        tick t+k (a raw component index could be renumbered between ticks).
        """
        t = self.current_tick
        if getattr(self, '_frag_labels_tick', None) == t:
            return self._frag_labels
        try:
            raw = self._infra_component_labels(True) or {}
        except Exception:
            raw = {}
        groups: Dict[object, list] = {}
        for n, lab in raw.items():
            groups.setdefault(lab, []).append(n)
        rep = {lab: min(ns) for lab, ns in groups.items()}
        self._frag_labels = {n: rep[lab] for n, lab in raw.items()}
        self._frag_labels_tick = t
        return self._frag_labels

    def _fragment_label_of(self, node_id: str):
        from .agent import POSTCARD_FRAGMENTS as _PF
        if not _PF:
            return -1
        return self._fragment_labels_now().get(node_id, -1)

    def _apply_fragment_cue(self, node_id: str, nb, postcards: list) -> None:
        """Fill nb.foreign_fragments_norm / nb.foreign_unbridged from the
        postcards received THIS tick, latched for FRAG_CUE_LATCH_TICKS."""
        latch = getattr(self, '_frag_cue_latch', None)
        if latch is None:
            latch = self._frag_cue_latch = {}
        mine = self._fragment_labels_now().get(node_id)
        fresh = [pc for pc in (postcards or [])
                 if getattr(pc, 'fragment_id', -1) not in (-1, None)]
        cue = None
        if fresh and mine is not None:
            foreign: Dict[object, bool] = {}
            for pc in fresh:
                f = pc.fragment_id
                if f != mine:
                    foreign[f] = foreign.get(f, False) or bool(
                        getattr(pc, 'holds_bridge', False))
            cue = (self.current_tick,
                   min(1.0, len(foreign) / 3.0),
                   1.0 if any(not h for h in foreign.values()) else 0.0)
            latch[node_id] = cue
        else:
            old = latch.get(node_id)
            if old is not None and (self.current_tick - old[0]
                                    <= self.FRAG_CUE_LATCH_TICKS):
                cue = old
        if cue is not None:
            nb.foreign_fragments_norm = cue[1]
            nb.foreign_unbridged = cue[2]

    def _build_neighbor_radio_summary(self, node_id: str,
                                      postcards: list) -> 'NeighborRadioSummary':
        """Build NeighborRadioSummary (Block B, 16 dims) from received postcards."""
        nb = NeighborRadioSummary()
        # Fragment coordination cue (+2 dims) -- applied BEFORE the empty-list
        # early return because the cue is LATCHED: a node keeps what it last
        # heard for up to FRAG_CUE_LATCH_TICKS even on ticks with no postcard.
        from .agent import POSTCARD_FRAGMENTS as _PF
        if _PF:
            self._apply_fragment_cue(node_id, nb, postcards)
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
        """Build PHY/MAC-based observations (OBS_DIM dims) for all infrastructure agents."""
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

        # 4. Hop distance to nearest live core node (ONE multi-source BFS)
        self._tick_core_dist_cache = {}
        try:
            live_cores = [nid for nid in self.core_nodes
                          if nid in self.topology.nodes
                          and self.topology.nodes[nid].is_survivor
                          and nid in self.topology.graph]
            if live_cores:
                from collections import deque
                dq = deque((c, 0) for c in live_cores)
                _seen = set(live_cores)
                while dq:
                    _nid, _d = dq.popleft()
                    self._tick_core_dist_cache[_nid] = _d
                    for _nb in self.topology.graph.neighbors(_nid):
                        if _nb not in _seen:
                            _seen.add(_nb)
                            dq.append((_nb, _d + 1))
        except Exception:
            self._tick_core_dist_cache = {}

        # 5. UE-pair demand per anchor node (from flow set via UE→O-RU map)
        self._tick_ue_demand_cache = {}
        for _s, _t in getattr(self, 'ue_to_ue_flows', []):
            for _ue in (_s, _t):
                for _anchor in self.ue_to_oru_map.get(_ue, []):
                    self._tick_ue_demand_cache[_anchor] = (
                        self._tick_ue_demand_cache.get(_anchor, 0) + 1)
        # ── End per-tick caches ──────────────────────────────────────────────────

        # ── Deliver postcards ONCE, then share the result ────────────────────
        # get_received_postcards -> DisasterControlChannel.receive_messages is a
        # DESTRUCTIVE read (control_plane.py: `self.message_queues[node_id] =
        # remaining`).  Draining here for the coordinator and reading again
        # below for the agents meant the agents always got [] and every Block B
        # dimension was a dataclass constant.  One read, both consumers.
        if POSTCARD_DRAIN == 'legacy':
            # Reproduces the pre-fix behaviour: the coordinator drains every
            # queue and the agents are left with nothing.
            all_postcards = []
            for nid in self.agents:
                all_postcards.extend(
                    self.control_plane.get_received_postcards(
                        nid, self.current_tick))
            control_messages = {
                nid: self.control_plane.get_received_postcards(
                    nid, self.current_tick)
                for nid in self.agents
            }
        else:
            control_messages = {
                nid: self.control_plane.get_received_postcards(
                    nid, self.current_tick)
                for nid in self.agents
            }
            all_postcards = [pc for pcs in control_messages.values()
                             for pc in pcs]
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
        # control_messages was populated above, from the single drain.

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
                    # TRUE BACKLOG, not a cumulative-delivery counter — see
                    # TrafficQueue's docstring for what this used to be.
                    current_queue_length=q.queued_load,
                    offered_load=q.offered_load,
                    admission_success_rate=(
                        q.delivered_load / max(1.0, q.offered_load)
                        if q.offered_load > 0 else 1.0
                    ),
                    freshness_target=self.slice_dictionary.qos_profiles[tc].target_delay,
                    # HONEST AGE OF INFORMATION (was hardcoded 5.0, i.e. a
                    # constant that no action could move and that therefore
                    # carried zero information).  Now the number of ticks since
                    # this slice last had a successful delivery, measured by
                    # TrafficQueue.freshness_ticks, so it is comparable against
                    # freshness_target (the slice's QoS target delay).
                    current_freshness=q.freshness_ticks(self.current_tick),
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

    @staticmethod
    def _derive_admission_policies(action: 'PHYMACAction') -> Dict[TrafficClass, Dict[str, str]]:
        """Map the policy's PRB split onto per-traffic-class admission decisions.

        The PHY/MAC action space has no explicit admission head, so admission
        is derived from the PRB allocation the policy chose THIS tick:
          - LIFE_SAFETY follows prb_emergency_frac
          - OPERATIONS / TELEMETRY / BEST_EFFORT follow prb_general_frac with
            increasingly strict thresholds (lower-priority classes are
            throttled/held first as general PRB share shrinks).

        The resulting dict genuinely reflects the policy output (it varies
        agent-to-agent and tick-to-tick with the sampled PRB softmax) and is
        consumed by apply_marl_policy() in _forward_traffic ('ADMIT' passes,
        'THROTTLE' halves, 'HOLD' drops) and by _assess_marl_convergence().

        The shares are re-normalised over the ACCESS pool (emergency +
        general) exactly as PHYMACState.normalise_prb does, because admission
        is an access-side decision.  The raw heads come from a 3-way softmax
        that includes the TRANSPORT relay head, and since that head no longer
        draws from the access carrier (see normalise_prb), reading it raw
        would re-introduce the same spurious coupling: a site dedicating a
        large share of its 60 GHz transport radio to a bridge would start
        HOLDing its own OPERATIONS traffic for no physical reason.
        """
        _access = action.prb_emergency_frac + action.prb_general_frac
        if _access > 1e-6:
            emerg = action.prb_emergency_frac / _access
            gen   = action.prb_general_frac / _access
        else:
            emerg, gen = 0.3, 0.7

        def _mode(share: float, admit_thr: float, throttle_thr: float) -> str:
            if share >= admit_thr:
                return 'ADMIT'
            if share >= throttle_thr:
                return 'THROTTLE'
            return 'HOLD'

        return {
            TrafficClass.LIFE_SAFETY: {'admission_mode': _mode(emerg, 0.20, 0.05)},
            TrafficClass.OPERATIONS:  {'admission_mode': _mode(gen,   0.30, 0.10)},
            TrafficClass.TELEMETRY:   {'admission_mode': _mode(gen,   0.40, 0.15)},
            TrafficClass.BEST_EFFORT: {'admission_mode': _mode(gen,   0.55, 0.25)},
        }

    # Planning transmit power of the transport radios (60 GHz TG head, 18 GHz
    # MW ODU) -- equals FRAG_BRIDGE_TX_DBM, the power the attainable-optimum
    # solver and the [TG-PLAN] reach line assume.  See MARL_TRANSPORT_POWER.
    TRANSPORT_PLANNING_TX_DBM = 23.0

    RELAY_REPOINT_COOLDOWN_TICKS = 10    # one DCC period between re-points

    def _maybe_repoint_to_fragment(self, node_id: str, ps) -> bool:
        """Release this site's NON-bridge relay link if a feasible
        cross-fragment target exists, so the same-tick formation branch can
        bridge.  See MARL_RELAY_REPOINT in __init__.  Returns True if a link
        was released."""
        node = self.topology.nodes.get(node_id)
        if node is None or not getattr(node, 'has_multihaul', False):
            return False
        t = self.current_tick
        if t - self._last_repoint_tick.get(node_id, -10 ** 9) < self.RELAY_REPOINT_COOLDOWN_TICKS:
            return False
        # Beam-training lockout: a link mid-acquisition cannot be re-pointed.
        if any(node_id in getattr(self.topology.links.get(lid), 'endpoints', ())
               for lid in self._relay_link_ready_tick):
            return False
        mine = [l for l in self.active_relay_links() if node_id in l.endpoints]
        if not mine:
            return False
        # Only a link that is NOT a tree-edge bridge may be abandoned: a
        # sole-path bridge keeps its existing protection and is never
        # re-pointed away from the fragment it joins.
        bridges = self.distinct_bridge_link_ids()
        if any(l.id in bridges for l in mine):
            return False
        # Is there a feasible cross-fragment target at all?  Same candidate
        # construction and link-budget test as the formation branch.
        try:
            import networkx as _nx
            g = self.topology._build_infrastructure_graph()
            my_comp = (_nx.node_connected_component(g, node_id)
                       if node_id in g else {node_id})
        except Exception:
            return False
        cands = [nid for nid, n in self.topology.nodes.items()
                 if n.is_survivor and nid not in my_comp
                 and n.node_type != NodeType.UE and nid in g]
        if not cands:
            return False
        for nid_c in [node_id] + cands:
            n_c = self.topology.nodes.get(nid_c)
            if n_c and not self.transport_relay_model.node_positions.get(nid_c):
                self.transport_relay_model.register_position(nid_c, n_c.x_pos, n_c.y_pos)
        feasible = self.transport_relay_model.find_relay_candidates(
            source=node_id, candidate_nodes=cands,
            tx_power_dbm=self._transport_tx_dbm(ps),
            relay_bw_fraction=max(0.15, ps.prb_relay_fraction),
            interference_dbm=lambda peer: self.aggregate_interference_dbm(peer, BAND_TG),
            max_candidates=len(cands))
        if not feasible:
            return False
        # Release the non-bridge link(s) -- both ends.
        peers = set()
        for l in mine:
            peers.add(l.endpoints[1] if l.endpoints[0] == node_id else l.endpoints[0])
        removed = self.transport_relay_model.clear_node_links(node_id)
        for rid in removed:
            self.topology.links.pop(rid, None)
            self._relay_link_ready_tick.pop(rid, None)
            self._relay_bridge_since.pop(rid, None)
            try:
                n1, n2 = rid.replace("TR_", "").split("_", 1)
                if self.topology.graph.has_edge(n1, n2):
                    self.topology.graph.remove_edge(n1, n2)
            except Exception:
                pass
        ps.relay_link_active = False
        ps.relay_peer_node = None
        ps.relay_link_capacity_mbps = 0.0
        for p in peers:                       # the abandoned end is free too
            pps = self.phy_mac_states.get(p)
            if pps is not None and not any(p in l.endpoints for l in self.active_relay_links()):
                pps.relay_link_active = False
                pps.relay_peer_node = None
                pps.relay_link_capacity_mbps = 0.0
        self.topology.invalidate_infrastructure_cache()
        self._last_repoint_tick[node_id] = t
        self._diag_repoints += 1
        return True

    def _transport_tx_dbm(self, ps) -> float:
        """Tx power for a TRANSPORT link budget or transport-band emission.

        Under MARL_TRANSPORT_POWER=planning this is the transport radio's own
        planning power; otherwise (legacy) it is the FR1 access power the
        policy controls, reproducing the shipped coupling exactly.
        """
        p = self._transport_planning_power
        return float(p) if p is not None else float(ps.tx_power_dbm)

    def _relay_hysteresis(self, relay_logits: torch.Tensor,
                          relay_v: torch.Tensor,
                          agent_ids: List[str]) -> torch.Tensor:
        """Deployment-executor switching margin on the relay head.

        MEASURED FAILURE MODE (auth_eval_s1234, reunifiable recovery seeds):
        the frozen policy's relay choice oscillates — up to 27 bridge
        formations and 26 teardowns in one episode (seed 54) against XL-DET's
        2 and 1 — because the bridge itself moves the observations that
        drive the decision: bridge up -> obs shift -> choice flips -> bridge
        down -> obs revert -> flips back.  A limit cycle, not a preference:
        MARL's PEAK bridge counts match XL-DET's; it loses reunification to
        CHURN.

        THE RULE.  Keep the incumbent relay mode unless the challenger's
        logit exceeds the incumbent's by MARL_RELAY_HYSTERESIS nats (e.g.
        ln 2 = 0.69 means "at least twice as likely").  Standard switching
        control; the same latch XL-DET has by construction and the routing
        protocols have as hold-down timers.  It is an EXECUTOR property of
        the xApp, not a training change: the learned arms apply it only on
        the frozen argmax path (no log-prob recorded, no gradient), and the
        random control applies it on its sampling path (no PPO at all).
        Env-gated, default 0 = exact argmax; every shipped result is
        bit-identical with it unset.  The margin is SELECTED ON VALIDATION
        SEEDS disjoint from both training and the benchmark (see
        run_hyst_eval.sh), so the benchmark stays a genuine test set.
        """
        hyst = float(os.environ.get('MARL_RELAY_HYSTERESIS', '0') or 0)
        if hyst <= 0.0:
            return relay_v
        keep = []
        for i, aid in enumerate(agent_ids):
            prev = getattr(self.agents[aid], 'last_action', None)
            cha = int(relay_v[i])
            if prev is not None:
                inc = int(prev.relay_mode_idx)
                if cha != inc and float(relay_logits[i, cha]
                                        - relay_logits[i, inc]) < hyst:
                    cha = inc                               # margin not met
            keep.append(cha)
        return torch.tensor(keep, dtype=torch.long)

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
            # Capability-conditioned relay mask (MARL_RELAY_MASK) — a no-op
            # unless enabled.  Applied HERE, before the arm overrides below:
            # the random arm replaces bl outright (its uniform control is
            # defined over the raw action space), and XL-DET never samples.
            from .agent import apply_relay_mask as _relay_mask
            bl = _relay_mask(bl, obs_tensors)

            # ── UNIFORM-RANDOM REFERENCE POLICY (attribution ablation) ─────
            # Set self.random_action_policy=True to run this EXACT environment
            # — same observations, same mechanisms, same _step_phy_mac, same
            # teardown rules — with the policy replaced by uniform sampling.
            # It is the control arm for the claim "the MARL policy LEARNED to
            # bridge islands": if a uniform policy reunifies the island just
            # as often, the mechanism (not the learning) is doing the work.
            # Zero logits make every discrete head's softmax exactly uniform;
            # the PRB simplex is drawn from Dirichlet(1,1,1), i.e. uniform
            # over the simplex (softmax of zeros would be the deterministic
            # 1/3,1/3,1/3 point, which is not a random allocation).
            _rand_policy = bool(getattr(self, 'random_action_policy', False))
            if _rand_policy:
                bl = {k: torch.zeros_like(v) for k, v in bl.items()}

            _is_train = True if _rand_policy else getattr(ref_agent, 'is_training', True)

            if _is_train:
                # Sample all discrete heads in 8 multinomial calls (vs N×8)
                import torch.nn.functional as _F
                _temp = getattr(ref_agent, '_logit_temperature', 1.0)
                def _vsample(head: str) -> torch.Tensor:
                    # Apply temperature scaling before softmax for controlled exploration
                    p = _F.softmax(bl[head] / _temp, dim=-1).clamp(min=1e-8)
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

                # ── RELAY DECISION HYSTERESIS (deployment executor) ────────
                # MEASURED FAILURE MODE (auth_eval_s1234, reunifiable seeds):
                # at evaluation the argmax relay choice oscillates — up to 27
                # bridge formations and 26 teardowns in one episode (seed 54)
                # against XL-DET's 2 and 1 — because the bridge itself changes
                # the observations that drive the argmax: bridge up -> obs
                # move -> argmax flips -> bridge down -> obs revert -> flips
                # back.  A limit cycle, not a preference.  Reunification is
                # lost to CHURN, not to under-selection: MARL's PEAK bridge
                # counts match XL-DET's.
                #
                # THE EXECUTOR RULE.  Standard switching-control hysteresis on
                # the frozen policy's relay head: keep the incumbent relay
                # mode unless the challenger's logit exceeds the incumbent's
                # by MARL_RELAY_HYSTERESIS nats.  This is a deployment-
                # executor statement about the xApp (topology-affecting
                # actions carry a switching margin), NOT a training change:
                # it acts only on this argmax branch, where no log-prob is
                # recorded and no gradient flows, so PPO correctness is
                # untouched.  Env-gated, default 0.0 = exact argmax, every
                # shipped result bit-identical.  Other arms: XL-DET and the
                # routing baselines never pass through this branch, and the
                # uniform-random control replaces the logits above, so the
                # comparison stays like-for-like.
                relay_v = self._relay_hysteresis(bl["relay"], relay_v, agent_ids)

            if _rand_policy:
                # ATTRIBUTION CONTROL.  The random arm never reaches the argmax
                # branch, so give it the SAME executor rule here.  If the
                # executor alone rescued a uniform-random policy, the
                # reunification gain would belong to the executor and not to
                # learning; this is the control that settles that.  Safe on
                # the sampling path only because the random arm records no
                # log-prob and runs no PPO (the learned arms' sampling path is
                # deliberately left untouched to keep pi_old == executed).
                relay_v = self._relay_hysteresis(bl["relay"], relay_v, agent_ids)

            # ── PINNED-HEAD ABLATION (per-head credit isolation) ───────────
            # Set self.pinned_action_heads to a set of head names; each named
            # head is forced to its PHYMACAction DATACLASS DEFAULT for every
            # agent, i.e. the "do nothing" setting of that knob (tx 0 dB, MCS
            # follow auto-CQI, relay OFF, no postcard, IOPS deny, PRB
            # 0.30/0.00/0.70).  Pinning EVERY head gives the do-nothing
            # control arm; pinning all but one FREES exactly that head, which
            # is how the per-head reward-vs-KPI isolation is measured.
            #
            # This is a MEASUREMENT SCAFFOLD only, in the same spirit as
            # random_action_policy above: the attribute does not exist unless
            # a driver sets it, so every arm — MARL, the routing baselines and
            # the random control — is bit-identical when it is unset.  No
            # physics, capacity or admission rule is touched.
            _pinned = getattr(self, 'pinned_action_heads', None)
            if _pinned:
                from .agent import PHYMACAction as _PA
                _d = _PA()
                _defaults = {
                    'tx_power': _d.tx_power_step,
                    'mcs_emrg': _d.mcs_emergency_idx,
                    'mcs_gen':  _d.mcs_general_idx,
                    'relay':    _d.relay_mode_idx,
                    'handover': _d.handover_idx,
                    'scheduler': _d.scheduler_idx,
                    'postcard': 0,
                    'iops':     0,
                }
                _vars = {'tx_power': 'tx_idx', 'mcs_emrg': 'mcs_e',
                         'mcs_gen': 'mcs_g', 'relay': 'relay_v',
                         'handover': 'ho_v', 'scheduler': 'sched_v',
                         'postcard': 'post_v', 'iops': 'iops_v'}
                _pin_const = {}
                for _h, _dv in _defaults.items():
                    if _h in _pinned:
                        _pin_const[_vars[_h]] = torch.full((N,), int(_dv),
                                                           dtype=torch.long)
                tx_idx  = _pin_const.get('tx_idx',  tx_idx)
                mcs_e   = _pin_const.get('mcs_e',   mcs_e)
                mcs_g   = _pin_const.get('mcs_g',   mcs_g)
                relay_v = _pin_const.get('relay_v', relay_v)
                ho_v    = _pin_const.get('ho_v',    ho_v)
                sched_v = _pin_const.get('sched_v', sched_v)
                post_v  = _pin_const.get('post_v',  post_v)
                iops_v  = _pin_const.get('iops_v',  iops_v)

            # PRB allocation for all N agents at once: (N, 3).
            # THIS is the live path -- agent.compute_action is only the
            # "shouldn't happen" fallback below -- so the policy
            # parameterisation must be applied HERE, not only there.
            import torch.nn.functional as _F
            from .agent import PRB_POLICY as _PRB_POLICY
            if _PRB_POLICY == 'dirichlet':
                # alpha = softplus(logits) + 1; sample while training, mean at
                # evaluation.  Sampling keeps the action stochastic so the PPO
                # ratio is a density ratio; the mean is the deterministic
                # representative used at evaluation.
                _alpha = _F.softplus(bl["prb"]) + 1.0
                if _is_train:
                    prb_all = torch.distributions.Dirichlet(_alpha).sample()
                else:
                    prb_all = _alpha / _alpha.sum(dim=-1, keepdim=True)
            else:
                prb_all = _F.softmax(bl["prb"], dim=-1)   # legacy
            if _pinned and 'prb' in _pinned:
                from .agent import PHYMACAction as _PA2
                _dp = _PA2()
                prb_all = torch.tensor(
                    [[_dp.prb_emergency_frac, _dp.prb_relay_frac,
                      _dp.prb_general_frac]] * N, dtype=torch.float32)
            if _rand_policy:
                # Uniform over the 3-simplex (Dirichlet(1,1,1)), sampled as
                # normalised Exp(1) draws so no extra distribution object is
                # needed and the global torch RNG stays the only seed source.
                _e = torch.empty(N, 3).exponential_(1.0)
                prb_all = _e / _e.sum(dim=1, keepdim=True)

            # ── XL-DET: THE ENGINEERED NON-LEARNING CONTROL ARM ───────────
            # Set self.heuristic_action_policy=True to replace the learned
            # policy with sixg_sim.heuristic_controller -- a deterministic
            # cross-layer controller with the SAME actuation, the SAME
            # observations (it obeys the same locality mask) and the SAME
            # environment.  It answers the question the uniform-random control
            # cannot: not "is the action space enough?" but "is LEARNING
            # necessary?"  It is built to win if it can; see that module.
            #
            # Placed last on purpose: it must win over the pinned-head and
            # random-policy scaffolds above rather than race them, so a driver
            # that sets two flags gets a loud contradiction, not a blend.
            _xldet = bool(getattr(self, 'heuristic_action_policy', False))
            if _xldet:
                if _rand_policy or _pinned:
                    raise RuntimeError(
                        "heuristic_action_policy cannot be combined with "
                        "random_action_policy or pinned_action_heads: each "
                        "replaces the policy, so the combination has no "
                        "defined meaning.")
                from . import heuristic_controller as _XL
                _h = [_XL.decide(observations[aid], self.agents[aid])
                      for aid in agent_ids]
                _L = lambda k: torch.tensor([int(r[k]) for r in _h],
                                            dtype=torch.long)
                tx_idx  = _L(0)
                mcs_e   = _L(1)
                mcs_g   = _L(2)
                relay_v = _L(3)
                post_v  = _L(4)
                iops_v  = _L(5)
                prb_all = torch.tensor([r[6] for r in _h], dtype=torch.float32)
                # XL-DET does not actuate handover or scheduler.  Leaving them
                # on the network's own draw would make a "deterministic"
                # baseline stochastic and, worse, would let an UNTRAINED net
                # drive two heads of it.  Pin them to the PHYMACAction
                # defaults -- the same "do nothing" setting the pinned-head
                # ablation uses -- so the arm is fully specified by its rules.
                from .agent import PHYMACAction as _PAx
                _dx = _PAx()
                ho_v    = torch.full((N,), int(_dx.handover_idx), dtype=torch.long)
                sched_v = torch.full((N,), int(_dx.scheduler_idx), dtype=torch.long)

            # Build PHYMACAction per agent (pure Python, no more per-head sampling)
            for i, aid in enumerate(agent_ids):
                agent = self.agents[aid]
                obs   = observations[aid]
                prb   = prb_all[i].tolist()
                # The point PPO must score: as SAMPLED, before the coordinator
                # clamp and the general-traffic floor below, which are
                # environment and not policy.
                _prb_sampled = list(prb)

                # Coordinator PRB floor
                if obs.global_policy and obs.is_island:
                    min_emrg = obs.global_policy[2]
                    if prb[0] < min_emrg:
                        deficit = min_emrg - prb[0]
                        prb[0] += deficit
                        prb[1] = max(0.0, prb[1] - deficit / 2)
                        prb[2] = max(0.0, prb[2] - deficit / 2)
                        total = sum(prb); prb = [x / total for x in prb]

                # General-traffic PRB floor: prevent degenerate "zero
                # forwarding" policies (mirrors agent.compute_action)
                _PRB_GENERAL_FLOOR = 0.10
                if prb[2] < _PRB_GENERAL_FLOOR:
                    deficit = _PRB_GENERAL_FLOOR - prb[2]
                    prb[2] = _PRB_GENERAL_FLOOR
                    other_sum = prb[0] + prb[1]
                    if other_sum > 1e-6:
                        prb[0] -= deficit * (prb[0] / other_sum)
                        prb[1] -= deficit * (prb[1] / other_sum)
                    prb = [max(0.0, x) for x in prb]
                    total = sum(prb); prb = [x / total for x in prb]

                ri   = relay_v[i].item()
                pi   = bool(post_v[i].item())
                ps   = obs.phy_mac

                # Postcard (rate-limited per agent)
                postcard_content = None
                tx_norm  = max(0.0, min(1.0, (ps.tx_power_dbm - 10.0) / 33.0))
                sinr_deg = max(0.0, min(1.0, -getattr(ps, 'sinr_delta', 0.0) / 10.0))
                # MARL_POSTCARD_ALWAYS: the DCC postcard is a control-plane
                # heartbeat, sent at the rate limit regardless of the policy's
                # postcard head (which then only matters for nothing).  Measured
                # motivation: the selected policy sent ~no postcards after the
                # severance, so the fragment cue (Block E) never fired at the
                # only two sites able to bridge seed 55.  XL-DET already requests
                # one every tick; this gives every agent arm the same heartbeat.
                if (pi or self._postcard_always) and (obs.current_tick - agent.last_postcard_tick) >= 10:
                    pi = True
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
                        # fragment coordination (inert unless the receiver
                        # is enabled; see MARL_POSTCARD_FRAGMENTS)
                        fragment_id=self._fragment_label_of(aid),
                        holds_bridge=bool(getattr(ps, 'relay_link_is_bridge', False)),
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
                    prb_sampled=_prb_sampled,
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
            # Per-class admission decisions derived from this tick's policy
            # output — consumed by apply_marl_policy() when routing traffic.
            agent.last_actions = self._derive_admission_policies(action)

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

        A UE-to-UE flow counts as 'routed' (success) only when at least 50% of
        its offered volume this tick was delivered (SUCCESS_DELIVERY_FRACTION);
        partial deliveries below that threshold still count toward delivered
        volume but not toward routed_fraction.  The volume-weighted delivered
        fraction is exposed as self._ue_to_ue_delivered_fraction.

        PER-NODE DELIVERED-VOLUME ATTRIBUTION (credit assignment)
        ---------------------------------------------------------
        The dominant global reward term is +100 x volume-weighted delivered
        fraction, and until now NOTHING in the per-agent reward corresponded
        to it.  Per-node traffic queues carry no usable signal (measured
        cross-agent sd of node_offered_load = 0.0000), and because the
        actor's advantage has the cross-agent mean subtracted from it
        (COMA-style baseline, A_i -= mean_j A_j), any quantity that is
        identical across agents contributes EXACTLY ZERO to the actor's
        gradient.  Net effect: the policy was trained on the relay, energy
        and bridge terms but not on delivered volume — the very metric the
        evaluation scores.  This block is what makes delivery locally
        creditable.

        ATTRIBUTION RULE — "carried volume", credited to every infra node on
        the delivering path.  When consume_path() pushes V Mbps along
        path p, EVERY node of p that owns an agent (i.e. is in
        self.phy_mac_states; the two UE endpoints do not) is credited with
        the full V, and every link of p is credited with the full V.

        DOUBLE-COUNTING IS DELIBERATE, AND HERE IS WHY.  Summed over the
        nodes of an H-hop path the credit is H x V, not V, so this is NOT a
        partition of the delivered volume.  That is the correct choice for a
        DIFFERENCE reward.  The quantity a difference reward wants is
        D_i = G(z) - G(z_-i): what the objective would have LOST had agent i
        not acted.  Every node on a path is a but-for cause of that path's
        delivery — remove any single one of them and (absent an alternate
        path, which consume_path would then have had to find) the whole V is
        lost, not V/H.  Splitting the volume V/H per hop would instead
        under-credit exactly the nodes doing the most forwarding work and
        would make a long relayed path look individually worthless to each
        of its members, which is the opposite of the incentive wanted.  The
        cost of the choice is stated plainly: the per-agent delivery terms do
        not sum to the global delivery term.  They are not meant to — the
        global term is what the critic sees, this is what the actor's
        gradient sees, and the COMA baseline removes the common part of it
        anyway.

        Also note what is NOT credited: infrastructure-originated traffic
        (telemetry / O&M) is routed through the same consume_path but is
        credited to nobody, because it does not enter the global reward's
        delivered fractions either.  Local and global therefore price
        exactly the same bytes.

        Exposed as:
          self._node_carried_u2u  {node_id -> Mbps}  UE-to-UE volume through me
          self._node_carried_gen  {node_id -> Mbps}  general UE volume through me
          self._link_carried_ue   {link_id -> Mbps}  delivered UE volume on link
          self._node_delivery_share_u2u / _gen  the same, divided by the
              tick's OFFERED volume, i.e. in the same units as the global
              delivered fraction, so the per-agent reward can price a byte at
              the global rate (+100 per unit u2u fraction, +10 general).
        """
        SUCCESS_DELIVERY_FRACTION = 0.5
        import networkx as nx
        from .topology import TrafficClass

        # Per-node / per-link delivered-volume attribution (see docstring)
        self._node_carried_u2u: Dict[str, float] = {}
        self._node_carried_gen: Dict[str, float] = {}
        self._link_carried_ue:  Dict[str, float] = {}
        # UEs that were an endpoint of a UE-to-UE flow which delivered
        # non-zero volume this tick.  Pure bookkeeping over deliveries the
        # loop below already computes — no routing, no physics, no extra
        # graph work — and it is built on every arm, because _forward_traffic
        # is the shared traffic path.  Read by the reachability reward term
        # (see Simulator.REACH_REQUIRE_TRAFFIC); nothing else consumes it and
        # no metric is derived from it.
        self._delivered_ue_endpoints: set = set()

        self.ue_to_ue_success_count = 0
        self.ue_to_ue_failure_count = 0
        ue_to_ue_volume = 0.0
        ue_to_ue_offered = 0.0
        # ── Diagnostic counters: WHY did a flow deliver zero this tick? ──
        # (cheap; read by drivers for mechanism-level debugging)
        self._diag_flow_hold = 0    # zeroed by admission HOLD along the path
        self._diag_flow_cap = 0     # admitted > 0 but capacity bottleneck = 0
        self._diag_flow_noroute = 0  # no routable path
        ue_general_volume = 0.0
        ue_general_offered = 0.0
        
        ue_to_ue_participants = set()
        for s, t in self.ue_to_ue_flows:
            ue_to_ue_participants.add(s)
            ue_to_ue_participants.add(t)
            
        def apply_marl_policy(path_nodes, traffic_class, amount):
            """Admission along a path, and the census of what it costs.

            ASYMMETRY THIS MEASURES.  last_actions is populated only by
            _execute_agent_actions, so ONLY arms that run agents are subject to
            admission at all: the routing baselines (OSPF/OLSR/BATMAN/AODV/SDN)
            pass through untouched because their nodes have no last_actions.
            A single HOLD anywhere on a path zeroes the flow outright and a
            single THROTTLE halves it, so an arm whose PRB split wanders into
            HOLD is penalised by a channel its competitors cannot experience.

            That is a real property of the design, not a bug -- admission IS a
            policy decision -- but it must be MEASURED before the uniform-random
            control's poor showing can be attributed to poor control rather than
            to self-inflicted admission loss.  These counters are cumulative
            over the episode and are reported per arm; they change no capacity,
            no routing and no reward.
            """
            bottleneck = amount
            throttled  = False
            for n_id in path_nodes:
                agent = self.agents.get(n_id)
                if agent and hasattr(agent, 'last_actions') and traffic_class in agent.last_actions:
                    mode = agent.last_actions[traffic_class].get('admission_mode', 0)
                    if mode == 1 or mode == 'THROTTLE':
                        bottleneck = min(bottleneck, amount * 0.5)
                        throttled = True
                    elif mode == 2 or mode == 'HOLD':
                        self._adm_calls += 1
                        self._adm_hold  += 1
                        self._adm_held_volume += float(amount)
                        return 0.0
            self._adm_calls += 1
            if throttled:
                self._adm_throttle += 1
                self._adm_throttled_volume += float(amount) - float(bottleneck)
            return bottleneck

        MAX_SE = 4.0  # QAM256 spectral efficiency — link.capacity assumes this

        def consume_path(path_nodes, amount, credit=None):
            """Route traffic through path, applying MCS-dependent link capacity.

            link.capacity = maximum capacity at QAM256 + max power.
            effective_capacity = link.capacity × (current_SE / MAX_SE)

            Agent MCS/power → spectral efficiency → effective capacity.
            This applies identically to MARL and baseline passes.

            `credit` selects the per-node/per-link delivered-volume ledger to
            attribute this delivery to ('u2u', 'gen', or None for traffic the
            global reward does not price — see the ATTRIBUTION RULE in
            _forward_traffic's docstring).  Attribution is pure bookkeeping:
            it happens only on the path that actually moved bytes (the
            final_bottleneck <= 0 early return below fires first), and it
            changes no capacity, no utilisation and no routing decision, so
            every arm's physical behaviour is bit-identical.
            """
            final_bottleneck = amount
            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                link_data = self.topology.graph.get_edge_data(u, v)
                if link_data and 'link_id' in link_data:
                    link_id = link_data['link_id']
                    link = self.topology.links.get(link_id)
                    if link is None:
                        continue
                    if not getattr(link, 'is_up', True):
                        # BLACKHOLE vs SATURATION.  Normally graph edge
                        # presence already implies is_up, so this branch only
                        # fires when the routing view and physical truth
                        # disagree.  The MARL failure-detection floor
                        # (MARLLinkStateDetector in run_timeline_comparison.py)
                        # creates that disagreement DELIBERATELY: for the BFD
                        # detection interval the routing view still contains a
                        # link that has physically failed, so the arm keeps
                        # selecting the dead path and delivers nothing over it.
                        # Flag it so the alternate-path retry below — which
                        # exists to work around SATURATION — cannot silently
                        # reroute around a failure the control plane has not
                        # detected yet, which would give back the very
                        # omniscience the floor removes.  Scoped to link ids
                        # the floor is actually holding stale, so behaviour is
                        # bit-identical for every other arm and for MARL on
                        # every tick where belief matches truth.
                        if link_id in getattr(self, '_marl_stale_link_ids', ()):
                            self._marl_blackholed_path = True
                        return 0.0

                    # Scale link capacity by transmitting node's MCS efficiency
                    # Use worst-case (minimum) of both endpoints.
                    #
                    # EXCEPTION — TRANSPORT_RELAY links: their capacity was
                    # ALREADY derived from the relay link's OWN budget at
                    # creation time (can_form_relay_link: FSPL -> SINR ->
                    # best MCS -> capacity, or a relay fraction of the base
                    # link's capacity).  Re-scaling them by the endpoints'
                    # ACCESS-class MCS (mcs_general, driven by the node's
                    # access sinr_average) double-counts link adaptation:
                    # under the static 15 dB SINR this was a uniform
                    # constant for every node/arm, but with per-node dynamic
                    # SINR a physically feasible relay bridge terminating at
                    # a low-SINR node was zeroed despite its verified budget.
                    # Applies identically to every arm's relay links.
                    #
                    # WIRED LINKS ARE NOT RADIO-ADAPTED EITHER.  The scaling
                    # was previously applied to EVERY link type, including the
                    # 25 Gbps FIBER Open-Fronthaul and F1/E1/E2/N2/N3/N9
                    # links.  A single-mode fibre's capacity has nothing to do
                    # with either endpoint's FR1 spectral efficiency: a low-SINR
                    # O-RU was silently having its 25 Gbps fronthaul cut to
                    # 3.1 Gbps (SE 0.5/4.0), and at SINR below -3 dB to ZERO —
                    # a fibre going dark because the air interface faded.  That
                    # throttled every arm's fronthaul and it is not physics.
                    # The scaling is now gated to link types that actually run
                    # over the air and whose capacity is set by the access-class
                    # MCS.  SATELLITE is excluded too (its own budget, not FR1).
                    if getattr(link, 'link_type', None) not in _MCS_SCALED_LINK_TYPES:
                        effective_cap = link.capacity
                    else:
                        mcs_scale = 1.0
                        for node_id in (u, v):
                            ps = self.phy_mac_states.get(node_id)
                            if ps:
                                se = ps.spectral_efficiency(for_emergency=False)
                                mcs_scale = min(mcs_scale, se / MAX_SE)
                        effective_cap = link.capacity * mcs_scale
                    available = max(0.0, effective_cap - link.current_utilization)
                    final_bottleneck = min(final_bottleneck, available)

            if final_bottleneck <= 0:
                return 0.0

            for i in range(len(path_nodes)-1):
                u, v = path_nodes[i], path_nodes[i+1]
                link_data = self.topology.graph.get_edge_data(u, v)
                if link_data and 'link_id' in link_data:
                    link = self.topology.links.get(link_data['link_id'])
                    if link:
                        link.add_traffic(final_bottleneck)
                    if credit is not None and 'link_id' in link_data:
                        _lid = link_data['link_id']
                        self._link_carried_ue[_lid] = (
                            self._link_carried_ue.get(_lid, 0.0)
                            + final_bottleneck)

            # Per-node delivered-volume credit.  Restricted to nodes that own
            # an agent (phy_mac_states) — the UE endpoints of a flow are on
            # the path but have no policy, so crediting them would be dead
            # bookkeeping.
            if credit is not None:
                _acc = (self._node_carried_u2u if credit == 'u2u'
                        else self._node_carried_gen)
                for n_id in path_nodes:
                    if n_id in self.phy_mac_states:
                        _acc[n_id] = _acc.get(n_id, 0.0) + final_bottleneck

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

            # ── UE-to-UE traffic: constant profiles ──────────────────────
            # Controlled experiment: traffic rate is CONSTANT per flow type.
            # Only the rescue UE arrival changes offered load.
            # Per-flow RNG provides natural tick-to-tick variation around
            # the mean, but NO systematic trends, phases, or decay.
            # Stable per-flow RNG: crc32 (process-independent, unlike salted
            # hash()) mixed with the run's random_seed.
            _t_rng = random.Random(
                self.current_tick * 6997
                + zlib.crc32(f"{source_ue}|{target_ue}".encode()) % 7919
                + (self.config.random_seed or 0))

            is_rescue_src = getattr(source_node, 'is_rescue_service', False)
            is_rescue_tgt = getattr(target_node, 'is_rescue_service', False)
            is_rescue_flow = is_rescue_src or is_rescue_tgt

            if not self.island_mode:
                # Pre-disaster: normal cellular traffic
                ue_to_ue_traffic = 8.0 + _t_rng.gauss(0, 1.0)

            elif is_rescue_flow:
                # ── Rescue service traffic (constant after arrival) ──
                # MCPTT PTT voice (3GPP TS 22.179): 60% duty cycle
                ptt_active = _t_rng.random() < 0.6
                voice = 4.5 if ptt_active else 0.5   # AMR-WB / SID frames
                sensor = 3.0   # body-cam GPS + vitals telemetry
                # Video assessment: 20% chance per tick per flow
                video = _t_rng.uniform(10.0, 30.0) if _t_rng.random() < 0.20 else 0.0
                ue_to_ue_traffic = voice + sensor + video

            else:
                # ── Civilian UE traffic post-disaster (constant) ──
                # Always-on baseline: ETWS/CMAS + cell reselection + IoT
                always_on = 4.0
                # Voice: 50% chance of active call
                calling = _t_rng.random() < 0.50
                voice = 6.0 if calling else 2.0
                # Messaging (WhatsApp, SMS)
                messaging = 3.0 if _t_rng.random() < 0.40 else 1.0
                ue_to_ue_traffic = always_on + voice + messaging

            ue_to_ue_traffic = max(0.5, ue_to_ue_traffic)  # minimum SOS beacon
            ue_to_ue_offered += ue_to_ue_traffic
            
            if self._can_route_ue_to_ue(source_ue, target_ue):
                try:
                    path = nx.shortest_path(self.topology.graph, source_ue, target_ue)
                    admitted = apply_marl_policy(path, TrafficClass.LIFE_SAFETY, ue_to_ue_traffic)
                    self._marl_blackholed_path = False
                    delivered = consume_path(path, admitted, credit='u2u')

                    if (delivered <= 0 and self.island_mode
                            and not self._marl_blackholed_path):
                        # Shortest path saturated — try alternative paths
                        # (relay links may provide capacity on longer routes).
                        # NOT entered when the chosen path was blackholed by an
                        # undetected-down link: that flow is waiting on failure
                        # detection, and enumerating detours would be the
                        # instantaneous reroute the detection floor forbids.
                        try:
                            alt_count = 0
                            for alt_path in nx.shortest_simple_paths(
                                    self.topology.graph, source_ue, target_ue):
                                alt_count += 1
                                if alt_count == 1:
                                    continue  # skip first (same as shortest)
                                if alt_count > 4:
                                    break  # limit search
                                alt_admitted = apply_marl_policy(
                                    alt_path, TrafficClass.LIFE_SAFETY, ue_to_ue_traffic)
                                delivered = consume_path(
                                    alt_path, alt_admitted, credit='u2u')
                                if delivered > 0:
                                    break
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass

                    ue_to_ue_volume += max(0.0, delivered)
                    if delivered > 0:
                        self._delivered_ue_endpoints.add(source_ue)
                        self._delivered_ue_endpoints.add(target_ue)
                    # Diagnostic: classify zero-delivery cause
                    if delivered <= 0:
                        if admitted <= 0:
                            self._diag_flow_hold += 1
                        else:
                            self._diag_flow_cap += 1
                    # Success only if a meaningful share of the offered volume
                    # got through (see SUCCESS_DELIVERY_FRACTION in docstring)
                    if delivered >= SUCCESS_DELIVERY_FRACTION * ue_to_ue_traffic:
                        self.ue_to_ue_success_count += 1
                    else:
                        self.ue_to_ue_failure_count += 1
                except nx.NetworkXNoPath:
                    self.ue_to_ue_failure_count += 1
                    self._diag_flow_noroute += 1
            else:
                self.ue_to_ue_failure_count += 1
                self._diag_flow_noroute += 1

        if self.island_mode and self.marl_ue_routing_enabled and self.current_tick % 20 == 0:
            print(f"[ISLAND] t={self.current_tick}: UE-to-UE traffic - {self.ue_to_ue_success_count}/{len(self.ue_to_ue_flows)} flows, volume: {ue_to_ue_volume:.1f} units")

        # Expose counts to connectivity reward
        self._last_routed_flows = self.ue_to_ue_success_count
        self._total_ue_flows    = max(1, len(self.ue_to_ue_flows))
        if self.island_mode:
            # Track peak routed flows during island mode.  (The theoretical
            # max is NOT the first island tick's score — it is computed from
            # the best-case surviving topology in _compute_island_upper_bound.)
            self._island_peak_flows = max(
                getattr(self, '_island_peak_flows', 0),
                self.ue_to_ue_success_count)

        # 2. General UE Traffic
        core_targets = [n for n in getattr(self, 'core_nodes', []) if self.topology.nodes.get(n) and self.topology.nodes[n].is_survivor]

        # Island-local sinks: surviving EdgeUPFs provide local DN steering
        # (3GPP TS 23.501 §6.3.3) when the central core is unreachable.
        island_sinks = [nid for nid, n in self.topology.nodes.items()
                        if n.is_survivor and
                        getattr(n.node_type, 'value', '') == 'EdgeUPF'] if self.island_mode else []

        def _best_path_to_any(src, sinks):
            """Shortest live-graph path from src to any sink ([src] if src IS a sink)."""
            paths = []
            for sink in sinks:
                if sink == src:
                    return [src]
                if self.topology.has_path(src, sink):
                    try:
                        paths.append(nx.shortest_path(self.topology.graph, src, sink))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
            return min(paths, key=len) if paths else None

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
                    best_path = _best_path_to_any(node_id, core_targets)
                elif self.island_mode and island_sinks:
                    # Island mode: general UE traffic steers to a reachable
                    # island-local EdgeUPF through the same capacity-enforced
                    # mechanism.  Traffic with no in-island destination
                    # legitimately drops.
                    best_path = _best_path_to_any(node_id, island_sinks)
                
                for tc, amount in clean_arrivals.items():
                    queue = node.queues[tc]
                    queue.offered_load = amount
                    delivered = 0.0
                    # Credit this delivery to the path ONLY when it is a
                    # delivery the global reward's general fraction counts —
                    # the life-safety slice of a UE that is also a UE-to-UE
                    # endpoint is excluded below, so it is excluded here too.
                    # Local and global must price the same bytes.
                    _gen_counts = (tc != TrafficClass.LIFE_SAFETY
                                   or node_id not in ue_to_ue_participants)
                    if best_path and amount > 0:
                        admitted = apply_marl_policy(best_path, tc, amount)
                        delivered = consume_path(
                            best_path, admitted,
                            credit='gen' if _gen_counts else None)


                    queue.admitted_load = delivered
                    queue.dropped_load = amount - delivered
                    queue.delivered_load = delivered
                    # TRUE BACKLOG (was `queue.queued_load += delivered`, a
                    # monotonic cumulative-delivery counter that reset_tick
                    # never cleared — see TrafficQueue's docstring).  Must run
                    # AFTER offered/delivered are set.
                    queue.update_backlog(self.current_tick)

                    if tc != TrafficClass.LIFE_SAFETY or node_id not in ue_to_ue_participants:
                        ue_general_volume += delivered
            else:
                # Infrastructure-originated traffic (telemetry, O&M) must reach
                # a UPF/EdgeUPF like any other traffic — no free delivery.
                # Connected mode: nearest live core node.  Island mode: an
                # island-local EdgeUPF.  Failed nodes / no viable path -> drop.
                infra_path = None
                if node.is_survivor:
                    sinks = core_targets if (not self.island_mode and core_targets) else island_sinks
                    infra_path = _best_path_to_any(node_id, sinks) if sinks else None
                for tc, amount in arrivals.items():
                    q = node.queues[tc]
                    q.offered_load = amount
                    delivered = 0.0
                    if infra_path and amount > 0:
                        admitted = apply_marl_policy(infra_path, tc, amount)
                        delivered = consume_path(infra_path, admitted)
                    q.admitted_load = delivered
                    q.delivered_load = delivered
                    q.dropped_load = amount - delivered
                    # Same true-backlog update as the UE branch above.
                    q.update_backlog(self.current_tick)

        self.last_tick_ue_to_ue_volume = ue_to_ue_volume
        self.last_tick_ue_general_volume = ue_general_volume
        self.last_tick_ue_to_ue_offered = ue_to_ue_offered
        self.last_tick_ue_general_offered = ue_general_offered
        # Volume-weighted delivered fraction (complements the thresholded
        # routed_fraction indicator — see SUCCESS_DELIVERY_FRACTION)
        self._ue_to_ue_delivered_fraction = (
            ue_to_ue_volume / ue_to_ue_offered if ue_to_ue_offered > 0 else 0.0)

        # ── PER-NODE delivery SHARE, in global-reward units ────────────────
        # Divide each node's carried volume by the SAME denominator the
        # global delivered fraction uses, so "one unit of local share" and
        # "one unit of global delivered_frac" are the same thing and can be
        # paid at the same rate (+100 / +10).  A node's share is in [0, 1]:
        # it cannot carry more than the network delivered, which cannot
        # exceed what was offered.  See the ATTRIBUTION RULE in this method's
        # docstring for why the shares do not sum to the global fraction.
        self._node_delivery_share_u2u = (
            {n: v / ue_to_ue_offered
             for n, v in self._node_carried_u2u.items()}
            if ue_to_ue_offered > 0 else {})
        self._node_delivery_share_gen = (
            {n: v / ue_general_offered
             for n, v in self._node_carried_gen.items()}
            if ue_general_offered > 0 else {})
        # Freeze this pass's relay-link load so the make-before-break teardown
        # rule reads a well-defined value regardless of tick ordering.
        self._snapshot_relay_link_load()

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
                            node = self.topology.nodes[core_node_id]
                            if (core_node_id not in self.agents
                                    and node_hosts_agent(node)):
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
                # Report only actually-computed convergence quantities
                cm = self._assess_marl_convergence()
                print(f"[MARL] t={tick}: converged agents "
                      f"{cm['converged_agents']}/{cm['total_agents']} "
                      f"(ratio={cm['convergence_ratio']:.2f}, "
                      f"stability={cm['policy_stability']:.2f}, "
                      f"coordination={cm['coordination_quality']:.2f})")
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
