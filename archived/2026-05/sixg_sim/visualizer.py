"""
6G MARL Simulation Visualizer
==============================
Real-time animated visualization with:
  - Network topology graph (nodes coloured by type/state)
  - Severed links drawn in red
  - Transport relay links in amber (animated pulse)
  - Postcard / DCC flashes (yellow arc)
  - PHY/MAC annotations per infrastructure node
  - Training panel: loss curves, reward, UE-pair connectivity
  - Online panel: live connectivity & resource bar gauges
  - MP4 / GIF export via matplotlib
"""

import sys
# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import collections
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import matplotlib
matplotlib.use("Agg")               # headless – works in any environment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "border":      "#30363d",
    "text":        "#c9d1d9",
    "text_dim":    "#8b949e",
    "accent":      "#58a6ff",
    "green":       "#3fb950",
    "orange":      "#d29922",
    "red":         "#f85149",
    "purple":      "#bc8cff",
    "cyan":        "#39d3f0",
    "yellow":      "#e3b341",
    # Node types
    "core":        "#f85149",
    "gnb":         "#58a6ff",
    "relay":       "#d29922",
    "ue":          "#3fb950",
    "ue_rescue":   "#bc8cff",
    "severed":     "#484f58",
    "transport_link":    "#d29922",
    "d2d_link":    "#bc8cff",
    "normal_link": "#444c56",
    "severed_link":"#f85149",
    "postcard":    "#e3b341",
    "island_ring": "#f85149",
}

NODE_SIZE = {
    "core": 600, "gnb": 500, "relay": 380,
    "odu": 300, "oru": 260, "ue": 220,
}


# ──────────────────────────────────────────────────────────────────────────────
# Data containers written to by the simulation loop via callbacks
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TickSnapshot:
    """Per-tick state snapshot collected by VisualizerHook."""
    tick:               int
    episode:            int = 0
    phase:              str = "train"       # "train" | "online"
    island_mode:        bool = False
    severance_tick:     Optional[int] = None
    # Graph edges state
    severed_links:      List[Tuple] = field(default_factory=list)
    transport_links:          List[Tuple] = field(default_factory=list)     # (u,v,capacity_mbps)
    postcard_senders:   List[str]   = field(default_factory=list)
    postcard_receivers: List[str]   = field(default_factory=list)
    # Node states
    node_relay_modes:   Dict[str, str]   = field(default_factory=dict)
    node_tx_power:      Dict[str, float] = field(default_factory=dict)
    node_prb_relay:     Dict[str, float] = field(default_factory=dict)
    node_iops_reg:      Set[str] = field(default_factory=set)
    node_is_island:     Set[str] = field(default_factory=set)
    node_relay_peers:   Dict[str, str]   = field(default_factory=dict)  # nid → peer_nid
    node_relay_capacity:Dict[str, float] = field(default_factory=dict)  # nid → Mbps
    # Metrics
    ue_pairs_routed:    float = 0.0
    reachable_frac:     float = 1.0
    iops_admitted:      int   = 0
    # Learning metrics
    policy_loss:        float = float("nan")
    value_loss:         float = float("nan")
    entropy:            float = float("nan")
    episode_reward:     float = 0.0
    # Coordinator
    global_policy:      List[float] = field(default_factory=lambda: [0.0]*5)
    # Multi-eNB IOPS (ETSI TS 22.346)
    iops_island_count:  int   = 0
    iops_nenb_count:    int   = 0
    iops_peer_exchanges:int   = 0
    iops_mcptt_calls:   int   = 0
    iops_xn_density:    float = 0.0
    node_island_ids:    Dict[str, str] = field(default_factory=dict)
    scenario_type:      str   = ""
    # Learning postcard exchange pairs: [(sender, receiver), ...]
    learning_postcard_pairs: List[Tuple[str, str]] = field(default_factory=list)
    # Theoretical optimal routing
    optimal_paths:      List[Tuple[str, str]] = field(default_factory=list)
    optimal_actions:    List[str] = field(default_factory=list)

@dataclass
class EpisodeSummary:
    episode: int
    avg_reward: float
    policy_loss: float
    value_loss: float
    connectivity_rate: float   # fraction of ticks with any UE-pair connectivity


# ──────────────────────────────────────────────────────────────────────────────
# Hook object – attach to simulation loop
# ──────────────────────────────────────────────────────────────────────────────

class VisualizerHook:
    """
    Lightweight callback object.  Add one instance to the simulation and call:
      hook.on_tick(snap)      – called each sim tick
      hook.on_episode_end(ep) – called at end of training episode
    The animator reads from hook.snapshots deque.
    """

    MAX_SNAPSHOTS  = 2000
    MAX_EPISODES   = 200

    def __init__(self):
        self.snapshots: collections.deque = collections.deque(
            maxlen=self.MAX_SNAPSHOTS
        )
        self.episodes: List[EpisodeSummary] = []
        self._episode_rewards:    List[float] = []
        self._episode_policy_loss: List[float] = []
        self._episode_value_loss:  List[float] = []
        self._episode_conn_ticks:  int = 0
        self._episode_total_ticks: int = 0
        self.latest: Optional[TickSnapshot] = None

    # ── called by main loop every tick ───────────────────────────────────────
    def on_tick(self, snap: TickSnapshot):
        self.snapshots.append(snap)
        self.latest = snap
        self._episode_rewards.append(snap.episode_reward)
        self._episode_total_ticks += 1
        if snap.ue_pairs_routed > 0:
            self._episode_conn_ticks += 1
        if not math.isnan(snap.policy_loss):
            self._episode_policy_loss.append(snap.policy_loss)
        if not math.isnan(snap.value_loss):
            self._episode_value_loss.append(snap.value_loss)

    def on_episode_end(self, episode: int):
        avg_r  = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
        avg_pl = float(np.mean(self._episode_policy_loss)) if self._episode_policy_loss else float("nan")
        avg_vl = float(np.mean(self._episode_value_loss))  if self._episode_value_loss  else float("nan")
        conn_r = (self._episode_conn_ticks / max(1, self._episode_total_ticks))
        self.episodes.append(EpisodeSummary(
            episode=episode, avg_reward=avg_r,
            policy_loss=avg_pl, value_loss=avg_vl,
            connectivity_rate=conn_r,
        ))
        self._episode_rewards.clear()
        self._episode_policy_loss.clear()
        self._episode_value_loss.clear()
        self._episode_conn_ticks = 0
        self._episode_total_ticks = 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _smooth(vals: List[float], w: int = 5) -> List[float]:
    if len(vals) < 2:
        return list(vals)
    arr = np.array(vals, dtype=float)
    kernel = np.ones(w) / w
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


def _classify_node(node_id: str, node_obj=None) -> str:
    """Return visual class string for a node."""
    nid = node_id.lower()
    if node_obj is not None:
        nt = str(getattr(node_obj, "node_type", "")).lower()
        if "ue" in nt:
            return "ue_rescue" if getattr(node_obj, "is_rescue_service", False) else "ue"
        if "core" in nt or "upf" in nt or "amf" in nt:
            return "core"
        if "relay" in nt or "Transport Relay" in nt or nid.startswith("relay"):
            return "relay"
        if "gnb" in nt or "du" in nt or "ru" in nt:
            return "gnb"
    # fallback from name
    if any(x in nid for x in ("core", "upf", "amf", "ausf", "udm")):
        return "core"
    if "relay" in nid or nid.startswith("Transport Relay"):
        return "relay"
    if "ue" in nid:
        return "ue_rescue" if "rescue" in nid else "ue"
    return "gnb"


def _node_color(cls: str, is_island: bool, is_severed: bool) -> str:
    if is_severed:
        return PALETTE["severed"]
    c = PALETTE.get(cls, PALETTE["gnb"])
    if is_island and cls not in ("ue", "ue_rescue"):
        # Tint toward red slightly so island infra looks stressed
        return c
    return c


# ──────────────────────────────────────────────────────────────────────────────
# Main Animator class
# ──────────────────────────────────────────────────────────────────────────────

class SimulationAnimator:
    """
    Produces two animated figures:
      1. training_animation.mp4  – network + learning curves (training phase)
      2. online_animation.mp4    – network + live metrics (online/deploy phase)

    Usage::

        anim = SimulationAnimator(topology, output_dir="output")
        hook = anim.make_hook()          # attach hook to your sim loop
        # ... run simulation, calling hook.on_tick / hook.on_episode_end ...
        anim.render_training("output/training.mp4")
        anim.render_online("output/online.mp4")
    """

    def __init__(self, topology, output_dir: str = "output",
                 fps: int = 8, dpi: int = 120):
        self.topology   = topology
        self.output_dir = output_dir
        self.fps        = fps
        self.dpi        = dpi
        os.makedirs(output_dir, exist_ok=True)

        self._hook:   Optional[VisualizerHook] = None
        self._pos:    Optional[Dict]            = None   # nx layout positions
        self._node_class: Dict[str, str]        = {}

        # Pre-compute layout once
        if HAS_NX:
            self._build_layout()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        G = nx.Graph()
        for nid in self.topology.nodes:
            G.add_node(nid)
        for lid, link in self.topology.links.items():
            if hasattr(link, "source") and hasattr(link, "dest"):
                G.add_edge(link.source, link.dest, link_id=lid)

        # Try hierarchical spring layout
        try:
            self._pos = nx.spring_layout(G, seed=42, k=2.5)
        except Exception:
            self._pos = {n: (i % 5, i // 5) for i, n in enumerate(G.nodes)}

        for nid, node in self.topology.nodes.items():
            self._node_class[nid] = _classify_node(nid, node)

    def make_hook(self) -> VisualizerHook:
        self._hook = VisualizerHook()
        return self._hook

    # ──────────────────────────────────────────────────────────────────────────
    # Figure builders
    # ──────────────────────────────────────────────────────────────────────────

    def _make_training_figure(self):
        """3-column layout: network (left, tall) | losses (top-right) | rewards / connectivity (bottom-right)."""
        fig = plt.figure(figsize=(18, 10), facecolor=PALETTE["bg"])
        gs  = gridspec.GridSpec(
            3, 3,
            figure=fig,
            left=0.04, right=0.97, top=0.94, bottom=0.06,
            hspace=0.45, wspace=0.35,
            width_ratios=[1.8, 1, 1],
        )
        ax_net  = fig.add_subplot(gs[:, 0])         # full left column
        ax_pol  = fig.add_subplot(gs[0, 1:])        # top-right (spans 2 cols)
        ax_val  = fig.add_subplot(gs[1, 1])         # mid-right left
        ax_ent  = fig.add_subplot(gs[1, 2])         # mid-right right
        ax_rew  = fig.add_subplot(gs[2, 1])         # bottom-right left
        ax_conn = fig.add_subplot(gs[2, 2])         # bottom-right right

        for ax in (ax_net, ax_pol, ax_val, ax_ent, ax_rew, ax_conn):
            ax.set_facecolor(PALETTE["panel"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
            ax.tick_params(colors=PALETTE["text_dim"], labelsize=7)
            ax.title.set_color(PALETTE["text"])

        return fig, ax_net, ax_pol, ax_val, ax_ent, ax_rew, ax_conn

    def _make_online_figure(self):
        """2-column layout: network (left) | dashboard gauges (right)."""
        fig = plt.figure(figsize=(16, 9), facecolor=PALETTE["bg"])
        gs  = gridspec.GridSpec(
            4, 2,
            figure=fig,
            left=0.04, right=0.97, top=0.94, bottom=0.06,
            hspace=0.5, wspace=0.3,
            width_ratios=[1.8, 1],
        )
        ax_net   = fig.add_subplot(gs[:, 0])
        ax_conn  = fig.add_subplot(gs[0, 1])
        ax_iops  = fig.add_subplot(gs[1, 1])
        ax_relay = fig.add_subplot(gs[2, 1])
        ax_prb   = fig.add_subplot(gs[3, 1])

        for ax in (ax_net, ax_conn, ax_iops, ax_relay, ax_prb):
            ax.set_facecolor(PALETTE["panel"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
            ax.tick_params(colors=PALETTE["text_dim"], labelsize=7)
            ax.title.set_color(PALETTE["text"])

        return fig, ax_net, ax_conn, ax_iops, ax_relay, ax_prb

    # ──────────────────────────────────────────────────────────────────────────
    # Network drawing helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_network(self, ax, snap: TickSnapshot):
        ax.cla()
        ax.set_facecolor(PALETTE["panel"])
        ax.set_axis_off()

        if not HAS_NX or self._pos is None:
            ax.text(0.5, 0.5, "networkx not available", color=PALETTE["text"],
                    ha="center", va="center", transform=ax.transAxes)
            return

        pos   = self._pos
        nodes = self.topology.nodes
        links = self.topology.links

        severed_set = set(map(tuple, snap.severed_links))
        relay_set     = {(u, v) for u, v, *_ in snap.transport_links}

        # ── Draw links ──────────────────────────────────────────────────────
        for lid, link in links.items():
            if not hasattr(link, "source") or not hasattr(link, "dest"):
                continue
            u, v = link.source, link.dest
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Determine style
            edge = (u, v)
            edge_r = (v, u)
            is_severed = (edge in severed_set or edge_r in severed_set or
                          not getattr(link, "is_up", True))
            is_relay     = (edge in relay_set or edge_r in relay_set)

            if is_severed:
                lc, lw, ls, la = PALETTE["severed_link"], 1.5, "--", 0.6
            elif is_relay:
                lc, lw, ls, la = PALETTE["transport_link"], 2.5, "-", 0.9
            else:
                lc, lw, ls, la = PALETTE["normal_link"], 1.0, "-", 0.7

            ax.plot([x0, x1], [y0, y1], color=lc, lw=lw,
                    linestyle=ls, alpha=la, zorder=1)

        # ── Draw transport relay links (if not in topology.links) ────────────────────
        for entry in snap.transport_links:
            u, v = entry[0], entry[1]
            cap  = entry[2] if len(entry) > 2 else 10.0
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            # Draw thicker amber relay line
            ax.plot([x0, x1], [y0, y1], color=PALETTE["transport_link"],
                    lw=3.0, alpha=0.85, zorder=2,
                    path_effects=[pe.Stroke(linewidth=5,
                                            foreground=PALETTE["bg"], alpha=0.3),
                                  pe.Normal()])
            # Capacity label at midpoint
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my, f"{cap:.1f}M",
                    color=PALETTE["yellow"], fontsize=5.5,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc=PALETTE["panel"], ec="none", alpha=0.7))

        # ── Postcard arcs ────────────────────────────────────────────────
        for s_id in snap.postcard_senders:
            if s_id not in pos:
                continue
            sx, sy = pos[s_id]
            for r_id in snap.postcard_receivers:
                if r_id not in pos or r_id == s_id:
                    continue
                rx, ry = pos[r_id]
                # Draw a curved arc
                mid_x = (sx + rx) / 2 + (ry - sy) * 0.25
                mid_y = (sy + ry) / 2 + (rx - sx) * 0.25
                ax.annotate("", xy=(rx, ry), xytext=(sx, sy),
                             arrowprops=dict(
                                 arrowstyle="-|>", color=PALETTE["postcard"],
                                 lw=1.2, connectionstyle=f"arc3,rad=0.25"
                             ), zorder=7)

        # ── Draw nodes ──────────────────────────────────────────────────
        for nid, node in nodes.items():
            if nid not in pos:
                continue
            x, y  = pos[nid]
            cls   = self._node_class.get(nid, "gnb")
            is_sv = nid in snap.node_is_island
            is_dn = not getattr(node, "is_survivor", True)
            color = _node_color(cls, is_sv, is_dn)
            size  = NODE_SIZE.get(cls, 250)

            # Island ring
            if is_sv and not is_dn:
                ax.scatter([x], [y], s=size * 2.2,
                           c=PALETTE["island_ring"], alpha=0.25, zorder=3,
                           linewidths=0)

            ax.scatter([x], [y], s=size, c=color,
                       edgecolors=PALETTE["bg"] if not is_dn else PALETTE["border"],
                       linewidths=1.5, zorder=4, alpha=0.9 if not is_dn else 0.45)

            # IOPS star
            if nid in snap.node_iops_reg:
                ax.scatter([x], [y+0.06], s=60, c=PALETTE["purple"],
                           marker="*", zorder=8)

            # Label
            short = nid.split("_")[-1] if "_" in nid else nid
            ax.text(x, y - 0.09, short,
                    color=PALETTE["text"] if not is_dn else PALETTE["text_dim"],
                    fontsize=6, ha="center", va="top", zorder=9)

            # PHY/MAC annotation for infra nodes
            relay_mode = snap.node_relay_modes.get(nid, "")
            tx_pow     = snap.node_tx_power.get(nid)
            if relay_mode or tx_pow is not None:
                ann_parts = []
                if relay_mode and relay_mode.upper() not in ("OFF", ""):
                    ann_parts.append(f"↗{relay_mode[:3]}")
                if tx_pow is not None:
                    ann_parts.append(f"{tx_pow:+.0f}dB")
                if ann_parts:
                    ax.text(x + 0.07, y + 0.05, " ".join(ann_parts),
                            color=PALETTE["orange"], fontsize=5,
                            ha="left", va="bottom", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc=PALETTE["bg"], ec="none", alpha=0.6))

        # ── Legend ───────────────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(color=PALETTE["core"],        label="Core/UPF"),
            mpatches.Patch(color=PALETTE["gnb"],         label="gNB/O-RU"),
            mpatches.Patch(color=PALETTE["relay"],       label="Transport Relay"),
            mpatches.Patch(color=PALETTE["ue"],          label="UE"),
            mpatches.Patch(color=PALETTE["ue_rescue"],   label="Rescue UE"),
            mpatches.Patch(color=PALETTE["severed"],     label="Severed"),
            Line2D([0],[0], color=PALETTE["transport_link"],   lw=2, label="transport relay link"),
            Line2D([0],[0], color=PALETTE["severed_link"],lw=1.5,ls="--",label="Severed link"),
            Line2D([0],[0], color=PALETTE["postcard"],   lw=1.5,label="Postcard"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=5.5, framealpha=0.4,
                  facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"])

        # ── Status header ────────────────────────────────────────────────
        ph_str  = snap.phase.upper()
        ep_str  = f"Ep {snap.episode}" if snap.phase == "train" else "ONLINE"
        sev_str = "[ISLAND MODE]" if snap.island_mode else "[CONNECTED]"
        ax.set_title(
            f"6G MARL Simulation  ·  {ph_str}  {ep_str}  ·  Tick {snap.tick}  ·  {sev_str}\n"
            f"UE pairs routed: {snap.ue_pairs_routed:.0%}  "
            f"Reachable: {snap.reachable_frac:.0%}  "
            f"IOPS admitted: {snap.iops_admitted}",
            color=PALETTE["text"], fontsize=8, pad=6,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Learning panel drawing helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _style_ax(self, ax, title, ylabel, xlabel=""):
        ax.set_facecolor(PALETTE["panel"])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["border"])
        ax.set_title(title, color=PALETTE["text"], fontsize=7, pad=3)
        ax.set_ylabel(ylabel, color=PALETTE["text_dim"], fontsize=6)
        if xlabel:
            ax.set_xlabel(xlabel, color=PALETTE["text_dim"], fontsize=6)
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=6)
        ax.grid(True, color=PALETTE["border"], lw=0.4, alpha=0.5)

    def _draw_loss_panel(self, ax_pol, ax_val, ax_ent, snapshots):
        """Draw policy loss, value loss, entropy over time."""
        ticks = [s.tick for s in snapshots]
        ploss = [s.policy_loss if not math.isnan(s.policy_loss) else None for s in snapshots]
        vloss = [s.value_loss  if not math.isnan(s.value_loss)  else None for s in snapshots]
        ents  = [s.entropy     if not math.isnan(s.entropy)     else None for s in snapshots]

        def _plot_metric(ax, vals, label, color, title, ylabel):
            ax.cla()
            self._style_ax(ax, title, ylabel, "Tick")
            clean_t = [t for t, v in zip(ticks, vals) if v is not None]
            clean_v = [v for v in vals if v is not None]
            if len(clean_v) >= 2:
                smoothed = _smooth(clean_v, w=min(len(clean_v), 7))
                ax.plot(clean_t, clean_v,  color=color, alpha=0.25, lw=0.8)
                ax.plot(clean_t, smoothed,  color=color, lw=1.5, label=label)
                ax.legend(fontsize=6, facecolor=PALETTE["panel"],
                          edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])

        _plot_metric(ax_pol, ploss, "Policy Loss",  PALETTE["accent"],  "Policy Loss",  "Loss")
        _plot_metric(ax_val, vloss, "Value Loss",   PALETTE["orange"],  "Value/Critic Loss", "Loss")
        _plot_metric(ax_ent, ents,  "Entropy",      PALETTE["purple"],  "Policy Entropy",    "H")

    def _draw_reward_conn(self, ax_rew, ax_conn, snapshots, episodes):
        """Draw reward curve and connectivity fraction."""
        # Per-tick reward
        ticks = [s.tick for s in snapshots]
        rews  = [s.episode_reward for s in snapshots]
        conns = [s.ue_pairs_routed for s in snapshots]

        ax_rew.cla()
        self._style_ax(ax_rew, "Episode Reward", "Reward", "Tick")
        if len(rews) >= 2:
            sr = _smooth(rews, w=min(len(rews), 15))
            ax_rew.fill_between(ticks, 0, sr,
                                color=PALETTE["green"], alpha=0.2)
            ax_rew.plot(ticks, sr, color=PALETTE["green"], lw=1.5)

        ax_conn.cla()
        self._style_ax(ax_conn, "UE-pair Connectivity", "Fraction", "Tick")
        if len(conns) >= 2:
            sc = _smooth(conns, w=min(len(conns), 9))
            ax_conn.fill_between(ticks, 0, sc,
                                 color=PALETTE["cyan"], alpha=0.2)
            ax_conn.plot(ticks, sc, color=PALETTE["cyan"], lw=1.5)
            ax_conn.set_ylim(0, 1.05)
            ax_conn.axhline(1.0, color=PALETTE["green"],
                            lw=0.8, ls="--", alpha=0.6)

        # Episode markers (vertical lines)
        if episodes:
            ep_ticks = []
            for i, ep in enumerate(episodes):
                # Approximate tick where episode ended
                ep_tick_idx = min(i * 500, len(ticks) - 1)
                if ep_tick_idx < len(ticks):
                    ep_ticks.append(ticks[ep_tick_idx])
            for et in ep_ticks:
                ax_rew.axvline(et, color=PALETTE["border"],
                               lw=0.7, ls=":", alpha=0.8)
                ax_conn.axvline(et, color=PALETTE["border"],
                                lw=0.7, ls=":", alpha=0.8)

    def _draw_episode_convergence(self, ax_pol, episodes):
        """Overlay per-episode loss on the policy axis."""
        if not episodes:
            return
        ep_nums  = [e.episode  for e in episodes]
        rewards  = [e.avg_reward     for e in episodes]
        pol_loss = [e.policy_loss if not math.isnan(e.policy_loss) else 0
                    for e in episodes]
        conn     = [e.connectivity_rate for e in episodes]
        ax_pol.cla()
        self._style_ax(ax_pol, "Per-Episode Convergence", "Value", "Episode")
        ax2 = ax_pol.twinx()
        ax2.set_facecolor("none")
        ax2.tick_params(colors=PALETTE["text_dim"], labelsize=6)
        ax_pol.plot(ep_nums, rewards,  color=PALETTE["green"],  lw=1.5, label="Avg Reward")
        ax_pol.plot(ep_nums, conn,     color=PALETTE["cyan"],   lw=1.2, ls="--", label="Conn Rate")
        ax2.plot(ep_nums, pol_loss,    color=PALETTE["orange"], lw=1.2, ls=":", label="Policy Loss")
        ax_pol.legend(fontsize=6, loc="upper left",
                      facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
                      labelcolor=PALETTE["text"])
        ax2.legend(fontsize=6, loc="upper right",
                   facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
                   labelcolor=PALETTE["text"])

    # ──────────────────────────────────────────────────────────────────────────
    # Online dashboard helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_online_gauges(self, ax_conn, ax_iops, ax_relay, ax_prb,
                            snapshots, snap: TickSnapshot):
        """Draw live metric gauges for the online phase."""
        # Connectivity time-series
        ticks = [s.tick for s in snapshots]
        conns = [s.ue_pairs_routed for s in snapshots]
        reach = [s.reachable_frac  for s in snapshots]

        ax_conn.cla()
        self._style_ax(ax_conn, "UE-pair Connectivity", "%", "Tick")
        if conns:
            ax_conn.fill_between(ticks, 0, [c*100 for c in conns],
                                 color=PALETTE["green"], alpha=0.3)
            ax_conn.plot(ticks, [c*100 for c in conns],
                         color=PALETTE["green"], lw=1.5, label="Routed pairs")
            ax_conn.plot(ticks, [r*100 for r in reach],
                         color=PALETTE["cyan"], lw=1.0, ls="--", label="Reachable UEs")
            ax_conn.set_ylim(0, 110)
            ax_conn.legend(fontsize=6, facecolor=PALETTE["panel"],
                           edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])

        # IOPS bar
        ax_iops.cla()
        self._style_ax(ax_iops, "IOPS Registered UEs", "", "")
        iops_count = len(snap.node_iops_reg) if snap.node_iops_reg else 0
        ax_iops.barh(["IOPS Reg"], [iops_count], color=PALETTE["purple"])
        ax_iops.barh(["Admitted"], [snap.iops_admitted], color=PALETTE["green"])
        ax_iops.set_xlim(0, max(10, iops_count + 2))
        ax_iops.set_xlabel("UE count", color=PALETTE["text_dim"], fontsize=6)

        # Relay mode distribution
        ax_relay.cla()
        self._style_ax(ax_relay, "Relay Mode Distribution", "", "")
        modes = {"OFF": 0, "Transport Relay": 0, "D2D": 0}
        for rm in snap.node_relay_modes.values():
            rm_up = rm.upper()
            if "Transport Relay" in rm_up:
                modes["Transport Relay"] += 1
            elif "D2D" in rm_up or "PEER" in rm_up:
                modes["D2D"] += 1
            else:
                modes["OFF"] += 1
        colors_m = [PALETTE["gnb"], PALETTE["transport_link"], PALETTE["d2d_link"]]
        bars = ax_relay.bar(list(modes.keys()), list(modes.values()),
                            color=colors_m, edgecolor=PALETTE["border"])
        ax_relay.set_ylabel("Nodes", color=PALETTE["text_dim"], fontsize=6)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_relay.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                              str(int(h)), ha="center", va="bottom",
                              color=PALETTE["text"], fontsize=6)

        # PRB relay allocation bar
        ax_prb.cla()
        self._style_ax(ax_prb, "Avg PRB Relay Allocation", "%", "")
        prb_vals = list(snap.node_prb_relay.values())
        avg_prb  = float(np.mean(prb_vals)) * 100 if prb_vals else 0.0
        ax_prb.barh(["PRB Relay"], [avg_prb],
                    color=PALETTE["accent"], edgecolor=PALETTE["border"])
        ax_prb.set_xlim(0, 100)
        ax_prb.set_xlabel("% of PRB budget", color=PALETTE["text_dim"], fontsize=6)
        ax_prb.text(avg_prb + 1, 0, f"{avg_prb:.1f}%",
                    color=PALETTE["text"], fontsize=7, va="center")

        # Coordinator policy vector
        gp = snap.global_policy
        if any(v != 0 for v in gp):
            gp_labels = ["relay_prio", "power_bdgt", "emrg_prb", "coop_thrs", "iops_en"]
            ax_prb.cla()
            self._style_ax(ax_prb, "Coordinator Policy Vector", "", "")
            c_colors = [PALETTE["accent"], PALETTE["orange"], PALETTE["green"],
                        PALETTE["purple"], PALETTE["cyan"]]
            ax_prb.barh(gp_labels, gp, color=c_colors,
                        edgecolor=PALETTE["border"])
            ax_prb.set_xlim(0, 1.05)

    # ──────────────────────────────────────────────────────────────────────────
    # Render entry points
    # ──────────────────────────────────────────────────────────────────────────

    def render_training(self, output_path: Optional[str] = None,
                        phase_filter: str = "train"):
        """Produce training animation MP4 / GIF."""
        if self._hook is None:
            print("[Visualizer] No hook set, cannot render.")
            return

        snaps    = [s for s in self._hook.snapshots if s.phase == phase_filter]
        episodes = self._hook.episodes

        if not snaps:
            print(f"[Visualizer] No snapshots for phase={phase_filter}")
            return

        output_path = output_path or os.path.join(self.output_dir, "training_animation.mp4")
        print(f"[Visualizer] Rendering training animation ({len(snaps)} frames) -> {output_path}")

        fig, ax_net, ax_pol, ax_val, ax_ent, ax_rew, ax_conn = \
            self._make_training_figure()

        fig.suptitle("6G MARL — Digital Twin Training Phase",
                     color=PALETTE["text"], fontsize=12, y=0.98,
                     fontweight="bold")

        def _update(frame_idx):
            snap = snaps[frame_idx]
            # Use all snapshots up to this frame for curves
            hist = snaps[:frame_idx+1]
            ep_hist = [e for e in episodes if e.episode <= snap.episode]

            self._draw_network(ax_net, snap)
            if ep_hist:
                self._draw_episode_convergence(ax_pol, ep_hist)
            else:
                self._draw_loss_panel(ax_pol, ax_val, ax_ent, hist)
            self._draw_loss_panel(ax_val, ax_ent, ax_ent, hist)  # reuse
            self._draw_reward_conn(ax_rew, ax_conn, hist, ep_hist)

        # Stride to keep video short
        total = len(snaps)
        stride = max(1, total // min(total, 600))
        frames = list(range(0, total, stride))

        anim = FuncAnimation(fig, _update, frames=frames,
                             interval=1000//self.fps, blit=False)

        self._save_animation(anim, output_path, len(frames))
        plt.close(fig)

    def render_online(self, output_path: Optional[str] = None,
                      phase_filter: str = "online"):
        """Produce online / deployment animation MP4 / GIF."""
        if self._hook is None:
            print("[Visualizer] No hook set, cannot render.")
            return

        snaps = [s for s in self._hook.snapshots if s.phase == phase_filter]
        if not snaps:
            # fall back to all snapshots if no online-tagged ones
            snaps = list(self._hook.snapshots)
        if not snaps:
            print("[Visualizer] No snapshots found.")
            return

        output_path = output_path or os.path.join(self.output_dir, "online_animation.mp4")
        print(f"[Visualizer] Rendering online animation ({len(snaps)} frames) -> {output_path}")

        fig, ax_net, ax_conn, ax_iops, ax_relay, ax_prb = self._make_online_figure()
        fig.suptitle("6G MARL — Live Disaster Recovery (Online Phase)",
                     color=PALETTE["text"], fontsize=12, y=0.98,
                     fontweight="bold")

        def _update(frame_idx):
            snap = snaps[frame_idx]
            hist = snaps[:frame_idx+1]
            self._draw_network(ax_net, snap)
            self._draw_online_gauges(ax_conn, ax_iops, ax_relay, ax_prb, hist, snap)

        total  = len(snaps)
        stride = max(1, total // min(total, 600))
        frames = list(range(0, total, stride))

        anim = FuncAnimation(fig, _update, frames=frames,
                             interval=1000//self.fps, blit=False)

        self._save_animation(anim, output_path, len(frames))
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # Combined render (static per-episode summary frames + animated)
    # ──────────────────────────────────────────────────────────────────────────

    def render_all(self):
        """Convenience: render both training and online animations."""
        train_path  = os.path.join(self.output_dir, "training_animation.mp4")
        online_path = os.path.join(self.output_dir, "online_animation.mp4")
        self.render_training(train_path, phase_filter="train")
        self.render_online(online_path,  phase_filter="online")

        # Also export a static convergence summary PNG
        self._export_convergence_summary()

    def _export_convergence_summary(self):
        """Static PNG: 2×2 convergence plots across all episodes."""
        if not self._hook or not self._hook.episodes:
            return
        episodes = self._hook.episodes
        ep_nums  = [e.episode          for e in episodes]
        rewards  = [e.avg_reward       for e in episodes]
        pol_loss = [e.policy_loss if not math.isnan(e.policy_loss) else 0
                    for e in episodes]
        val_loss = [e.value_loss  if not math.isnan(e.value_loss)  else 0
                    for e in episodes]
        conn     = [e.connectivity_rate for e in episodes]

        fig, axs = plt.subplots(2, 2, figsize=(12, 7),
                                facecolor=PALETTE["bg"])
        fig.suptitle("MAPPO Training Convergence Summary",
                     color=PALETTE["text"], fontsize=13, fontweight="bold")

        plots = [
            (axs[0,0], ep_nums, rewards,  PALETTE["green"],  "Average Episode Reward",    "Reward"),
            (axs[0,1], ep_nums, conn,     PALETTE["cyan"],   "UE-pair Connectivity Rate", "Fraction"),
            (axs[1,0], ep_nums, pol_loss, PALETTE["orange"], "Policy Loss (per episode)", "Loss"),
            (axs[1,1], ep_nums, val_loss, PALETTE["accent"], "Value/Critic Loss",         "Loss"),
        ]
        for ax, x, y, c, title, ylabel in plots:
            ax.set_facecolor(PALETTE["panel"])
            for sp in ax.spines.values():
                sp.set_edgecolor(PALETTE["border"])
            ax.plot(x, y, color=c, lw=2.0, marker="o", ms=4, markerfacecolor="white")
            if len(y) >= 3:
                sm = _smooth(y, w=min(len(y), 5))
                ax.plot(x, sm, color=c, lw=0.8, ls="--", alpha=0.6)
            ax.set_title(title, color=PALETTE["text"], fontsize=9)
            ax.set_ylabel(ylabel, color=PALETTE["text_dim"], fontsize=8)
            ax.set_xlabel("Episode", color=PALETTE["text_dim"], fontsize=8)
            ax.tick_params(colors=PALETTE["text_dim"])
            ax.grid(True, color=PALETTE["border"], lw=0.4, alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(self.output_dir, "convergence_summary.png")
        fig.savefig(out, dpi=self.dpi, facecolor=PALETTE["bg"])
        plt.close(fig)
        print(f"[Visualizer] Convergence summary → {out}")

    # ──────────────────────────────────────────────────────────────────────────
    # Save helper
    # ──────────────────────────────────────────────────────────────────────────

    def _save_animation(self, anim: FuncAnimation, path: str, n_frames: int):
        ext = os.path.splitext(path)[-1].lower()
        try:
            if ext == ".gif":
                writer = PillowWriter(fps=self.fps)
                anim.save(path, writer=writer, dpi=self.dpi)
            else:
                # Try ffmpeg first, fall back to GIF
                try:
                    writer = FFMpegWriter(fps=self.fps, bitrate=1800,
                                         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
                    anim.save(path, writer=writer, dpi=self.dpi,
                              progress_callback=lambda i, n:
                                  print(f"\r  Encoding frame {i}/{n_frames}", end="", flush=True))
                    print()
                except Exception:
                    gif_path = path.replace(".mp4", ".gif")
                    print(f"  [ffmpeg unavailable] Saving as GIF → {gif_path}")
                    writer = PillowWriter(fps=self.fps)
                    anim.save(gif_path, writer=writer, dpi=self.dpi)
                    path = gif_path
            print(f"[Visualizer] Saved -> {path}")
        except Exception as ex:
            print(f"[Visualizer] Save failed: {ex}")


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot builder  (called from main.py / simulation loop)
# ──────────────────────────────────────────────────────────────────────────────

def build_snapshot(sim, tick: int, episode: int, phase: str,
                   policy_loss: float = float("nan"),
                   value_loss:  float = float("nan"),
                   entropy:     float = float("nan"),
                   episode_reward: float = 0.0,
                   ue_pairs_routed: float = None,
                   reachable_frac:  float = None,
                   iops_admitted:   int   = None) -> TickSnapshot:
    """
    Build a TickSnapshot from the live Simulator state.
    Call this once per tick inside the training loop.
    Optional scalars (ue_pairs_routed, reachable_frac, iops_admitted) can be
    passed explicitly; otherwise they are read from the sim object.
    """
    snap = TickSnapshot(tick=tick, episode=episode, phase=phase,
                        island_mode=getattr(sim, "island_mode", False),
                        policy_loss=policy_loss, value_loss=value_loss,
                        entropy=entropy, episode_reward=episode_reward,
                        optimal_paths=getattr(sim, "current_optimal_paths", []),
                        optimal_actions=list(getattr(sim, "current_optimal_actions", set())))

    topo = sim.topology

    # Severed links
    for lid, link in topo.links.items():
        if not getattr(link, "is_up", True):
            s, d = getattr(link, "source", ""), getattr(link, "dest", "")
            snap.severed_links.append((s, d))

    # Transport relay links
    relay_model = getattr(sim, "transport_relay_model", None)
    if relay_model:
        for entry in getattr(relay_model, "active_links", []):
            if hasattr(entry, "source") and hasattr(entry, "dest"):
                snap.transport_links.append((entry.source, entry.dest,
                                       getattr(entry, "capacity_mbps", 10.0)))
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                snap.transport_links.append((entry[0], entry[1],
                                       entry[2] if len(entry) > 2 else 10.0))

    # Postcard senders (from last tick)
    ctrl = getattr(sim, "control_plane", None)
    if ctrl:
        for nid in sim.agents:
            pcs = ctrl.get_received_postcards(nid, tick)
            if pcs:
                snap.postcard_receivers.append(nid)
                for pc in pcs:
                    sid = getattr(pc, "sender_id", None)
                    if sid:
                        snap.postcard_senders.append(sid)

    # PHY/MAC per-node state
    for nid, ps in getattr(sim, "phy_mac_states", {}).items():
        rm = getattr(getattr(ps, "relay_mode", None), "value", "off")
        snap.node_relay_modes[nid] = rm
        snap.node_tx_power[nid]    = getattr(ps, "tx_power_dbm", 0.0)
        snap.node_prb_relay[nid]   = getattr(ps, "prb_relay_frac", 0.0)
        # Relay re-routing path info
        peer = getattr(ps, "relay_peer_node", None)
        if peer and getattr(ps, "relay_link_active", False):
            snap.node_relay_peers[nid] = peer
            snap.node_relay_capacity[nid] = getattr(ps, "relay_link_capacity_mbps", 0.0)

    # Island nodes
    for nid, node in topo.nodes.items():
        if getattr(node, "is_island", False):
            snap.node_is_island.add(nid)

    # IOPS registered UEs
    iops_mgr = getattr(sim, "iops_manager", None)
    if iops_mgr:
        snap.node_iops_reg = set(getattr(iops_mgr, "_registered", {}).keys())

    # Scalar metrics (use caller-provided overrides when given)
    snap.ue_pairs_routed = (ue_pairs_routed if ue_pairs_routed is not None
                            else getattr(sim, "_ue_pair_routed_fraction", 0.0))
    snap.reachable_frac  = (reachable_frac  if reachable_frac  is not None
                            else getattr(sim, "_reachable_ue_fraction",   1.0))
    if iops_mgr and iops_admitted is None:
        snap.iops_admitted = getattr(iops_mgr, "total_admitted", 0)
    elif iops_admitted is not None:
        snap.iops_admitted = iops_admitted

    # Global coordinator policy
    gp = getattr(sim, "_current_global_policy", None)
    if gp is not None:
        snap.global_policy = gp.to_list() if hasattr(gp, "to_list") else list(gp)

    # Multi-eNB IOPS controller state
    iops_ctrl = getattr(sim, "iops_controller", None)
    if iops_ctrl:
        stats = iops_ctrl.stats()
        snap.iops_island_count   = stats.get('num_islands', 0)
        snap.iops_nenb_count     = stats.get('total_nenbs', 0)
        snap.iops_xn_density     = 0.0
        # Per-node island assignment
        for isl_id, isl_info in stats.get('islands', {}).items():
            for enb in isl_info.get('members', []):
                snap.node_island_ids[enb] = isl_id
            snap.iops_xn_density = max(
                snap.iops_xn_density, isl_info.get('xn_density', 0.0))

    # Peer learning exchanges
    exchanger = getattr(sim, "learning_exchanger", None)
    if exchanger:
        snap.iops_peer_exchanges = getattr(exchanger, 'total_exchanges', 0)
        # Capture sender→receiver pairs from last exchange tick
        for recv_id, postcards in getattr(exchanger, '_inbox', {}).items():
            for pc in postcards:
                sid = getattr(pc, 'sender_id', None)
                if sid:
                    snap.learning_postcard_pairs.append((sid, recv_id))

    # Scenario type
    snap.scenario_type = getattr(sim, '_current_scenario_type', '')

    return snap
