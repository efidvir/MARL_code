"""
Transport relay model for dynamic wireless backhaul.

Ceragon transport devices provide programmable wireless transport underlay
that supports dynamic relay behavior at the transport layer. This model
handles link feasibility, path loss estimation, and relay link lifecycle.

NOTE: This is NOT a 3GPP IAB (Integrated Access & Backhaul) RAN function.
Ceragon devices do not operate as native IAB-MT/IAB-DU nodes. Instead,
they provide SDN-controlled dynamic transport relay/rerouting capability.

Physics model (simplified free-space + basic link budget):
  Path loss:  PL = 20*log10(d_m) + 20*log10(f_MHz) - 27.55  (dB)
  Rx power:   P_rx = P_tx - PL
  SINR:       SINR = P_rx - noise_floor - interference
  Capacity:   C = BW * SE(SINR)   (TDD — same spectrum as access)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


@dataclass
class TransportRelayLink:
    """A temporary wireless transport backhaul link created in relay mode."""
    link_id:       str
    source_node:   str
    target_node:   str
    capacity_mbps: float
    sinr_db:       float
    active:        bool = True


# Backward-compatible alias
IABLink = TransportRelayLink


class TransportRelayModel:
    """
    Maintains the set of active transport relay links and evaluates link feasibility.

    Integration with topology:
      When create_link() is called, the caller is responsible for
      adding a corresponding Link object to Topology so that the
      NetworkX graph reflects the new wireless hop.  Similarly,
      remove_link() signals the caller to drop it from the graph.
    """

    # ── Physical constants ───────────────────────────────────────────────
    NOISE_FLOOR_DBM  = -100.0    # Thermal noise + NF at 20 MHz BW
    BANDWIDTH_MHZ    = 20.0
    CARRIER_FREQ_GHZ = 3.5       # FR1 mid-band
    MAX_RELAY_RANGE_M = 2000.0   # Practical max relay range (beyond this: infeasible)
    D2D_CAPACITY_MBPS = 2.0      # Fixed capacity per active D2D sidelink pair

    def __init__(self):
        self.active_links: Dict[str, TransportRelayLink] = {}
        # Optional: {node_id: (x_m, y_m)} for position-aware path loss
        self.node_positions: Dict[str, Tuple[float, float]] = {}

    # ── Position registration ────────────────────────────────────────────

    def register_position(self, node_id: str, x_m: float, y_m: float):
        self.node_positions[node_id] = (x_m, y_m)

    def distance_m(self, node_a: str, node_b: str) -> Optional[float]:
        """Euclidean distance between two nodes (None if unknown)."""
        if node_a not in self.node_positions or node_b not in self.node_positions:
            return None
        xa, ya = self.node_positions[node_a]
        xb, yb = self.node_positions[node_b]
        return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

    # ── Path loss and SINR ───────────────────────────────────────────────

    def free_space_path_loss_db(self, dist_m: float) -> float:
        """Free-space path loss: PL = 20*log10(d) + 20*log10(f_MHz) − 27.55."""
        if dist_m <= 1.0:
            return 0.0
        freq_mhz = self.CARRIER_FREQ_GHZ * 1000.0
        return 20.0 * math.log10(dist_m) + 20.0 * math.log10(freq_mhz) - 27.55

    def estimate_sinr_db(self, tx_power_dbm: float, dist_m: float,
                         interference_dbm: float = -120.0) -> float:
        """Estimate received SINR in dB."""
        pl     = self.free_space_path_loss_db(dist_m)
        p_rx   = tx_power_dbm - pl
        noise  = self.NOISE_FLOOR_DBM
        # Combine noise + interference in linear domain, then back to dB
        n_lin  = 10 ** (noise / 10.0)
        i_lin  = 10 ** (interference_dbm / 10.0)
        ni_dbm = 10.0 * math.log10(n_lin + i_lin)
        return p_rx - ni_dbm

    # ── Link feasibility ─────────────────────────────────────────────────

    def best_mcs_for_sinr(self, sinr_db: float) -> Tuple[float, float]:
        """
        Return (spectral_efficiency, sinr_threshold) for the best MCS
        that works at the given SINR.  Returns (0, 0) if no MCS works.
        """
        from .phy_mac_state import MCSLevel, MCS_SINR_THRESHOLD, MCS_SPECTRAL_EFFICIENCY
        best_se = 0.0
        best_thr = 0.0
        for mcs in reversed(list(MCSLevel)):
            if sinr_db >= MCS_SINR_THRESHOLD[mcs]:
                best_se  = MCS_SPECTRAL_EFFICIENCY[mcs]
                best_thr = MCS_SINR_THRESHOLD[mcs]
                break
        return best_se, best_thr

    def can_form_relay_link(self, source: str, target: str,
                          tx_power_dbm: float,
                          relay_bw_fraction: float = 0.20
                          ) -> Tuple[bool, float, float]:
        """
        Evaluate transport relay link feasibility between source and target.

        Returns:
            (feasible: bool, capacity_mbps: float, sinr_db: float)
        """
        dist = self.distance_m(source, target)

        if dist is None:
            # No position data — heuristic: assume feasible at moderate capacity
            heuristic_cap = self.BANDWIDTH_MHZ * relay_bw_fraction * 2.0  # ~8 Mbps at QAM16
            return True, heuristic_cap, 12.0

        if dist > self.MAX_RELAY_RANGE_M:
            return False, 0.0, -999.0

        sinr_db = self.estimate_sinr_db(tx_power_dbm, dist)
        se, _ = self.best_mcs_for_sinr(sinr_db)

        if se <= 0.0:
            return False, 0.0, sinr_db

        # Capacity = BW × SE × relay_bw_fraction  (TDD: relay shares spectrum)
        cap = self.BANDWIDTH_MHZ * 1e6 * se * relay_bw_fraction / 1e6
        return True, cap, sinr_db

    # Backward-compatible alias
    can_form_iab_link = can_form_relay_link

    # ── Link lifecycle ───────────────────────────────────────────────────

    def create_link(self, source: str, target: str,
                    capacity_mbps: float, sinr_db: float = 15.0) -> str:
        """Create a transport relay link and return its ID."""
        link_id = f"TR_{source}_{target}"
        self.active_links[link_id] = TransportRelayLink(
            link_id=link_id,
            source_node=source,
            target_node=target,
            capacity_mbps=capacity_mbps,
            sinr_db=sinr_db,
            active=True,
        )
        return link_id

    def remove_link(self, link_id: str):
        """Remove a transport relay link (returns its data or None)."""
        return self.active_links.pop(link_id, None)

    def update_link_capacity(self, link_id: str, new_cap: float):
        """Update capacity of an existing relay link (e.g. after power change)."""
        if link_id in self.active_links:
            self.active_links[link_id].capacity_mbps = new_cap

    def get_node_links(self, node_id: str) -> List[TransportRelayLink]:
        """Return all active transport relay links touching a node."""
        return [l for l in self.active_links.values()
                if l.active and node_id in (l.source_node, l.target_node)]

    def clear_node_links(self, node_id: str) -> List[str]:
        """Remove all relay links for a node; return list of removed link IDs."""
        to_remove = [lid for lid, l in self.active_links.items()
                     if node_id in (l.source_node, l.target_node)]
        for lid in to_remove:
            self.active_links.pop(lid, None)
        return to_remove

    # ── D2D helpers ──────────────────────────────────────────────────────

    def estimate_d2d_pair_capacity(self, ue_count: int) -> float:
        """
        Total sidelink capacity added by D2D relay mode.
        Simplified: each UE pair gets D2D_CAPACITY_MBPS Mbps of sidelink.
        """
        pairs = max(0, ue_count // 2)
        return pairs * self.D2D_CAPACITY_MBPS

    # ── Candidate discovery ──────────────────────────────────────────────

    def find_relay_candidates(self, source: str,
                            candidate_nodes: List[str],
                            tx_power_dbm: float,
                            relay_bw_fraction: float = 0.20,
                            max_candidates: int = 3) -> List[Tuple[str, float, float]]:
        """
        Find the best transport relay candidates for a source node.

        Returns list of (node_id, capacity_mbps, sinr_db) sorted by capacity desc.
        """
        results = []
        for target in candidate_nodes:
            if target == source:
                continue
            feasible, cap, sinr = self.can_form_relay_link(
                source, target, tx_power_dbm, relay_bw_fraction
            )
            if feasible and cap > 0:
                results.append((target, cap, sinr))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_candidates]

    # Backward-compatible alias
    find_iab_candidates = find_relay_candidates


# Backward-compatible aliases for imports
IABRelayModel = TransportRelayModel
