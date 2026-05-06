"""
PHY/MAC state machine for O-RU and O-DU nodes.

Models the radio resource parameters that MARL agents directly control,
replacing the Near-RT RIC xApp functions that are unavailable in island mode.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Enumerations ──────────────────────────────────────────────────────────────

class RelayMode(Enum):
    OFF              = "off"               # Link idle / not participating
    NORMAL           = "normal"            # Normal core-bound transport mode
    LOCAL_REROUTE    = "local_reroute"     # Repurpose capacity for UE-to-UE local traffic
    CAPACITY_BOOST   = "capacity_boost"    # MultiHaul beam re-steer to boost link capacity
    D2D_PEER_RELAY   = "d2d_peer_relay"    # Signal UEs to relay via sidelink (ProSe)
    # Backward-compatible alias
    TRANSPORT_RELAY  = "local_reroute"     # Maps to LOCAL_REROUTE


class MCSLevel(Enum):
    """Modulation and Coding Schemes ordered by robustness (most robust first)."""
    QPSK_1_3  = 0   # Very robust; survives SINR ≥ -3 dB
    QPSK_1_2  = 1   # Robust;      survives SINR ≥  0 dB
    QAM16     = 2   # Balanced;    survives SINR ≥  5 dB
    QAM64     = 3   # High rate;   survives SINR ≥ 12 dB
    QAM256    = 4   # Very high;   survives SINR ≥ 20 dB


class MACScheduler(Enum):
    ROUND_ROBIN     = "round_robin"
    PROP_FAIR       = "prop_fair"
    EMERGENCY_FIRST = "emergency_first"   # Life-safety UEs get all free PRBs first
    MAX_SINR        = "max_sinr"          # Maximise cell throughput (best-effort focus)


# ── Physics tables ────────────────────────────────────────────────────────────

# Minimum SINR required for each MCS to operate (dB)
MCS_SINR_THRESHOLD: Dict[MCSLevel, float] = {
    MCSLevel.QPSK_1_3: -3.0,
    MCSLevel.QPSK_1_2:  0.0,
    MCSLevel.QAM16:     5.0,
    MCSLevel.QAM64:    12.0,
    MCSLevel.QAM256:   20.0,
}

# Peak spectral efficiency (bits/s/Hz) for each MCS
MCS_SPECTRAL_EFFICIENCY: Dict[MCSLevel, float] = {
    MCSLevel.QPSK_1_3: 0.667,
    MCSLevel.QPSK_1_2: 1.0,
    MCSLevel.QAM16:    2.0,
    MCSLevel.QAM64:    3.0,
    MCSLevel.QAM256:   4.0,
}

# Tx power adjustment steps (dB) available as discrete agent actions
TX_POWER_STEPS_DB = [-6.0, -3.0, 0.0, 3.0, 6.0]

# Scheduling weight multipliers for emergency UEs under each scheduler
SCHEDULER_EMERGENCY_WEIGHT: Dict[MACScheduler, float] = {
    MACScheduler.ROUND_ROBIN:     1.0,
    MACScheduler.PROP_FAIR:       1.5,
    MACScheduler.EMERGENCY_FIRST: 4.0,
    MACScheduler.MAX_SINR:        0.5,
}


# ── PHY/MAC State dataclass ───────────────────────────────────────────────────

@dataclass
class PHYMACState:
    """
    PHY/MAC state for a single O-RU or O-DU node.

    Agent-controlled fields are marked [CTRL].
    Observed/derived fields are marked [OBS] and updated by the simulator
    each tick based on the current topology and UE associations.
    """

    node_id: str

    # ── [CTRL] Transmission power ──────────────────────────────────────────
    tx_power_dbm:     float = 23.0   # Current Tx power in dBm
    tx_power_min_dbm: float = 10.0   # Hardware minimum
    tx_power_max_dbm: float = 33.0   # Hardware maximum (macro O-RU)

    # ── [CTRL] MCS per UE class ────────────────────────────────────────────
    mcs_emergency: MCSLevel = MCSLevel.QPSK_1_2   # For life-safety / rescue UEs
    mcs_general:   MCSLevel = MCSLevel.QAM16       # For all other UEs

    # ── [CTRL] PRB allocation (must sum to 1.0; enforced by normalise()) ──
    prb_emergency_fraction: float = 0.30   # Fraction for emergency UEs
    prb_relay_fraction:     float = 0.00   # Fraction donated to Transport relay
    prb_general_fraction:   float = 0.70   # Fraction for general UEs

    # ── [CTRL] MAC scheduler and relay ────────────────────────────────────
    relay_mode:      RelayMode    = RelayMode.OFF
    mac_scheduler:   MACScheduler = MACScheduler.PROP_FAIR
    handover_triggered: bool      = False   # Signals edge UEs to attempt HO

    # ── [OBS] Radio measurements (updated per tick by simulator) ──────────
    sinr_average:     float = 15.0   # Mean UE SINR (dB)
    sinr_min:         float = 5.0    # Weakest UE SINR (dB) — coverage stress
    active_ue_count:  int   = 0
    emergency_ue_count: int = 0
    prb_utilization:  float = 0.0    # Fraction of non-relay PRBs in use

    # ── [OBS] Backhaul / fronthaul ─────────────────────────────────────────
    backhaul_utilization: float = 0.0     # [0,1] fraction of backhaul used
    backhaul_capacity_mbps: float = 1000.0   # Available midhaul/fronthaul (Mbps)

    # ── [OBS] Coverage ────────────────────────────────────────────────────
    base_coverage_radius_m:      float = 500.0
    effective_coverage_radius_m: float = 500.0

    # ── [OBS] Transport relay state ─────────────────────────────────────────────
    relay_link_active: bool          = False
    relay_peer_node:   Optional[str] = None
    relay_link_capacity_mbps: float       = 0.0

    # ── [OBS] D2D state ───────────────────────────────────────────────────
    d2d_active_pairs: int = 0

    # ── [OBS] Interference caused to neighbours ───────────────────────────
    interference_caused_db: float = 0.0

    # ── [OBS] Connectivity metrics (updated per tick) ──────────────────────
    reachable_ue_fraction:       float = 1.0   # UEs with active Uu link / total
    ue_pair_routing_fraction:    float = 0.0   # Routed pairs / required pairs
    is_bridge_node:              bool  = False  # Sole link between two clusters

    # ── Physics helpers ───────────────────────────────────────────────────

    def apply_power_step(self, step_index: int):
        """Apply a discrete Tx power step (0=−6 dB … 4=+6 dB)."""
        delta = TX_POWER_STEPS_DB[max(0, min(4, step_index))]
        self.tx_power_dbm = max(self.tx_power_min_dbm,
                                min(self.tx_power_max_dbm, self.tx_power_dbm + delta))
        # Update effective coverage: radius ∝ 10^(ΔP/20)  (free-space model)
        relative_power_db = self.tx_power_dbm - 23.0  # relative to nominal 23 dBm
        self.effective_coverage_radius_m = (
            self.base_coverage_radius_m * (10 ** (relative_power_db / 20.0))
        )

    def spectral_efficiency(self, for_emergency: bool) -> float:
        """
        Return achievable spectral efficiency (bits/s/Hz) for a UE class,
        accounting for SINR headroom.  Returns 0 if SINR < MCS threshold.
        """
        mcs = self.mcs_emergency if for_emergency else self.mcs_general
        threshold = MCS_SINR_THRESHOLD[mcs]
        se_peak   = MCS_SPECTRAL_EFFICIENCY[mcs]
        sinr      = self.sinr_average

        if sinr < threshold:
            return 0.0                           # Link fails — no throughput
        elif sinr < threshold + 5.0:
            return se_peak * 0.5                 # Degraded link — half rate
        return se_peak

    def access_capacity_mbps(self, bandwidth_mhz: float = 20.0) -> float:
        """
        Capacity available for UE traffic (Mbps), excluding relay PRBs.
        Uses a weighted average SE based on emergency/general UE mix.
        """
        se_emrg = self.spectral_efficiency(for_emergency=True)
        se_gen  = self.spectral_efficiency(for_emergency=False)
        total   = max(1, self.active_ue_count)
        w_emrg  = self.emergency_ue_count / total
        w_gen   = 1.0 - w_emrg

        # Emergency PRBs × emergency SE  +  general PRBs × general SE
        capacity = bandwidth_mhz * 1e6 * (
            self.prb_emergency_fraction * (w_emrg * se_emrg + (1 - w_emrg) * se_emrg) +
            self.prb_general_fraction   * se_gen
        ) / 1e6
        return max(0.0, capacity)

    def relay_capacity_mbps(self, bandwidth_mhz: float = 20.0) -> float:
        """
        Transport relay capacity (Mbps) under TDD model.
        The relay PRB fraction is carved from the same spectrum as access
        so UE capacity is reduced proportionally.
        """
        if self.relay_mode not in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST) or self.prb_relay_fraction <= 0:
            return 0.0
        se = MCS_SPECTRAL_EFFICIENCY.get(self.mcs_general, 2.0)
        return max(0.0, bandwidth_mhz * 1e6 * self.prb_relay_fraction * se / 1e6)

    def normalise_prb(self):
        """Ensure PRB fractions sum to exactly 1.0."""
        total = (self.prb_emergency_fraction +
                 self.prb_relay_fraction +
                 self.prb_general_fraction)
        if total > 1e-6:
            self.prb_emergency_fraction /= total
            self.prb_relay_fraction     /= total
            self.prb_general_fraction   /= total
        else:
            # Fallback: even split without relay
            self.prb_emergency_fraction = 0.3
            self.prb_relay_fraction     = 0.0
            self.prb_general_fraction   = 0.7

    # ── Observation serialisation ─────────────────────────────────────────

    def to_block_a(self) -> List[float]:
        """
        Block A — 12-dim Radio Unit Status vector for agent observation.
        All values normalised to [0, 1].
        """
        power_range = self.tx_power_max_dbm - self.tx_power_min_dbm
        relay_val = {RelayMode.OFF: 0.0,
                     RelayMode.NORMAL: 0.25,
                     RelayMode.LOCAL_REROUTE: 0.5,
                     RelayMode.CAPACITY_BOOST: 0.75,
                     RelayMode.D2D_PEER_RELAY: 1.0}.get(self.relay_mode, 0.0)

        return [
            (self.tx_power_dbm - self.tx_power_min_dbm) / max(1.0, power_range),
            self.prb_utilization,
            self.spectral_efficiency(for_emergency=False) / 4.0,   # normalise by max SE
            max(0.0, min(1.0, (self.sinr_average + 5.0) / 35.0)), # SINR [-5,30] → [0,1]
            min(1.0, self.active_ue_count / 100.0),
            min(1.0, self.emergency_ue_count / max(1, self.active_ue_count)),
            min(1.0, self.effective_coverage_radius_m / 2000.0),
            self.backhaul_utilization,
            min(1.0, self.backhaul_capacity_mbps / 1000.0),
            relay_val,
            1.0 if self.relay_link_active else 0.0,
            max(0.0, min(1.0, (self.sinr_min + 5.0) / 35.0)),
        ]
