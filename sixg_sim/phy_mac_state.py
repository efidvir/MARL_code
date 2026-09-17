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


# ── Auto CQI→MCS link adaptation (3GPP TS 38.214 §5.1.3.1) ───────────────────
#
# Native gNB/DU L1/L2 behaviour: the MAC scheduler maps the reported CQI
# (proxied here by sinr_average) to the highest-rate MCS whose SINR operating
# threshold is met.  This is the SHARED physical-layer baseline for every
# routing/control arm; agent MCS actions are interpreted as bounded OFFSETS
# from this choice (see MCS_OFFSET_CENTER below and Simulator._step_phy_mac).

def auto_cqi_mcs_idx(sinr_db: float) -> int:
    """Return the MCSLevel index the auto CQI→MCS link adaptation selects.

    Highest-rate MCS whose SINR threshold is satisfied; falls back to the
    most robust MCS (QPSK_1_3) when SINR is below every threshold.
    """
    best = 0
    for mcs in MCSLevel:
        if sinr_db >= MCS_SINR_THRESHOLD[mcs]:
            best = mcs.value
    return best


# Agent MCS heads are OFFSETS from the auto-CQI MCS index, centred on this
# value: head index h maps to offset (h - MCS_OFFSET_CENTER).  With the
# 5-way MCS head this yields offsets {-2, -1, 0, +1, +2}; offset 0 means
# "follow auto-CQI exactly", negative offsets are more robust (conservative
# under interference), positive offsets are more aggressive (drain queues).
# The final MCS index is clamped to the valid MCSLevel range.
MCS_OFFSET_CENTER = len(MCSLevel) // 2

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

    # ── [CTRL] Resource allocation ────────────────────────────────────────
    #
    # TWO SEPARATE POOLS — see normalise_prb() for why.
    #
    #   ACCESS pool (the FR1 Uu carrier, 20 MHz):
    #       prb_emergency_fraction + prb_general_fraction == 1.0
    #   TRANSPORT pool (the dedicated backhaul radio — 60 GHz MultiHaul TG,
    #   MW PtP or fibre; NOT the Uu carrier):
    #       prb_relay_fraction in [0, 1], independent of the access split
    prb_emergency_fraction: float = 0.30   # Share of ACCESS PRBs for emergency UEs
    prb_relay_fraction:     float = 0.00   # Share of the TRANSPORT radio given to relay
    prb_general_fraction:   float = 0.70   # Share of ACCESS PRBs for general UEs

    # ── [CTRL] MAC scheduler and relay ────────────────────────────────────
    relay_mode:      RelayMode    = RelayMode.OFF
    mac_scheduler:   MACScheduler = MACScheduler.PROP_FAIR
    handover_triggered: bool      = False   # Signals edge UEs to attempt HO

    # ── [OBS] Radio measurements (updated per tick by simulator) ──────────
    # sinr_average / sinr_min are DYNAMIC: recomputed every tick by
    # Simulator._update_dynamic_sinr() from node geometry, own tx_power_dbm
    # and the FSPL link budget (see simulation.py).  The values below are
    # only cold-start defaults for the first tick / standalone unit tests.
    sinr_average:     float = 15.0   # Link SINR at strongest active neighbour (dB)
    sinr_min:         float = 5.0    # Cell-edge (weakest UE) SINR (dB) — coverage stress
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

    # ── [OBS] PER-NODE relay ATTRIBUTION (credit assignment) ───────────────
    #
    # These are the node-local decomposition of two terms that
    # Simulator.compute_global_connectivity_reward already computes
    # fleet-wide, written per node by _update_phy_mac_observations so the
    # per-agent reward can pay each agent for the part of the shared
    # objective ITS OWN relay link produced.
    #
    # Why they are needed.  Measured on the Scenario-A comparison
    # environment (42 infra nodes, 200 island ticks), EVERY term of the old
    # local reward had a cross-agent standard deviation of 0.000000 except
    # the Tx-power energy COST — i.e. the only credit-assigned gradient any
    # agent received told it to turn its transmitter down.  These two fields
    # are the cheapest quantities that are (a) genuinely per-node and (b)
    # responsive to the node's OWN relay action: measured eta^2 of the relay
    # head on relay_link_active is 0.86.
    #
    # They are DERIVED, not new physics: relay_traffic_carried_mbps is the
    # utilisation the transport links incident to this node already report,
    # and relay_link_is_bridge is membership of the de-duplicated bridge set
    # the global reward already counts.  Nothing here changes the
    # environment, so every arm behaves exactly as before — only the MARL
    # training signal is re-attributed.
    relay_traffic_carried_mbps: float = 0.0   # traffic on MY transport relay links
    relay_link_is_bridge:       bool  = False # one of MY relay links is a distinct bridge
    # How many fragments MY relay links eliminate: the number of my links that
    # are tree edges of the forest the relay links induce over the raw
    # partition (Simulator.distinct_bridge_link_ids).  Removing a tree edge
    # splits exactly one merge, so this IS the per-site counterfactual
    # "components without my bridges minus components with them" -- a true
    # difference reward for bridging, and 0 for a redundant/cyclic link.
    bridge_counterfactual_fragments: int = 0

    # ── [OBS] PER-NODE DELIVERED-VOLUME ATTRIBUTION (credit assignment) ────
    #
    # The node-local counterpart of the DOMINANT global reward term
    # (+100 x volume-weighted delivered fraction).  Without these the primary
    # objective had no local counterpart at all, and since the actor's
    # advantage has the cross-agent mean subtracted from it (COMA baseline),
    # a term identical across agents contributes exactly zero to the actor's
    # gradient — so the policy was learning relay/energy/bridge behaviour but
    # NOT delivered volume, the metric the evaluation actually scores.
    #
    # Definition: the delivered UE traffic volume that traversed THIS node
    # this tick, divided by the tick's OFFERED volume — i.e. expressed in the
    # same units as the global delivered fraction, so a byte is priced
    # identically by the local and the global view.  Written by
    # _update_phy_mac_observations from the ledgers _forward_traffic builds
    # while it routes (see its ATTRIBUTION RULE docstring: every infra node
    # on a delivering path is credited with the full volume, deliberately,
    # because each is a but-for cause of that delivery).
    #
    # DERIVED, not new physics: these are the same bytes consume_path already
    # pushed through the same links.  Nothing in the environment changes, so
    # every arm behaves exactly as before.
    delivered_share_ue_to_ue: float = 0.0   # my carried u2u volume / offered u2u
    delivered_share_general:  float = 0.0   # my carried general volume / offered general
    carried_ue_traffic_mbps:  float = 0.0   # raw Mbps of delivered UE traffic through me

    # ── [OBS] D2D state ───────────────────────────────────────────────────
    d2d_active_pairs: int = 0

    # ── [OBS] Interference caused to neighbours ───────────────────────────
    interference_caused_db: float = 0.0

    # ── [OBS] Link-adaptation diagnostics ─────────────────────────────────
    # Number of ticks on which the applied MCS (auto-CQI + agent offset)
    # exceeded what the current SINR supports, forcing an OLLA-style
    # step-down in effective_mcs().  Incremented once per tick per node in
    # Simulator._step_phy_mac; exposed for metrics / diagnostics.
    mcs_fallback_events: int = 0

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

    def effective_mcs(self, for_emergency: bool) -> Tuple[Optional[MCSLevel], bool]:
        """
        OLLA-style link adaptation: return (effective_mcs, fell_back).

        If the selected MCS's SINR threshold exceeds the current
        sinr_average, real systems do NOT deliver zero throughput — outer
        loop link adaptation / HARQ steps the transmission down to the
        highest MCS that the SINR does support.  This helper mirrors that:

          * selected MCS feasible          -> (selected, False)
          * infeasible, lower MCS feasible -> (highest feasible, True)
          * SINR below even QPSK_1/3       -> (None, True)   — genuine outage

        The fell_back flag feeds the per-node mcs_fallback_events counter.
        """
        selected = self.mcs_emergency if for_emergency else self.mcs_general
        sinr = self.sinr_average
        if sinr >= MCS_SINR_THRESHOLD[selected]:
            return selected, False
        # Step down: highest MCS below the selected one whose threshold is met
        for mcs in reversed(list(MCSLevel)):
            if mcs.value < selected.value and sinr >= MCS_SINR_THRESHOLD[mcs]:
                return mcs, True
        return None, True                        # Below QPSK_1/3 — outage

    def spectral_efficiency(self, for_emergency: bool) -> float:
        """
        Return achievable spectral efficiency (bits/s/Hz) for a UE class,
        accounting for SINR headroom.

        Uses OLLA-style graceful fallback (see effective_mcs): if the
        selected MCS is infeasible at the current SINR, the highest feasible
        MCS is used instead of returning 0.0.  Returns 0.0 ONLY when SINR is
        below the QPSK_1/3 threshold (-3 dB) — a genuine link outage.
        Within 5 dB of the effective MCS's threshold the link runs at half
        rate (degraded, retransmission-limited).
        """
        mcs, _fell_back = self.effective_mcs(for_emergency)
        if mcs is None:
            return 0.0                           # Genuine outage — no MCS works
        threshold = MCS_SINR_THRESHOLD[mcs]
        se_peak   = MCS_SPECTRAL_EFFICIENCY[mcs]
        if self.sinr_average < threshold + 5.0:
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
        Transport relay capacity (Mbps) on the DEDICATED transport radio.

        prb_relay_fraction is the share of the transport carrier the node
        dedicates to the relay hop.  It is NOT carved out of the access
        spectrum — see normalise_prb().
        """
        if self.relay_mode not in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST) or self.prb_relay_fraction <= 0:
            return 0.0
        se = MCS_SPECTRAL_EFFICIENCY.get(self.mcs_general, 2.0)
        return max(0.0, bandwidth_mhz * 1e6 * self.prb_relay_fraction * se / 1e6)

    def normalise_prb(self):
        """Normalise the ACCESS pool to 1.0 and clamp the TRANSPORT share.

        PHYSICS CORRECTION (was: all three fractions normalised to sum 1.0).
        The old model put access traffic and the transport relay hop in ONE
        20 MHz pool, so raising prb_relay_fraction mechanically cut
        access_capacity_mbps at the same site.  That is the in-band 3GPP IAB
        model, and this simulator explicitly does NOT model IAB:
        transport_relay_model.py states "This is NOT a 3GPP IAB (Integrated
        Access & Backhaul) RAN function", and every link a relay hop actually
        runs on is a dedicated transport radio — LinkType.MULTIHAUL_MESH
        (60 GHz Siklu MultiHaul TG), MW PtP or fibre — never the FR1 Uu access
        carrier the UEs are camped on.  A 60 GHz TG bridge and a 3.5 GHz Uu
        cell do not share spectrum, so they must not share a PRB budget.

        Consequences, and why this is a correction rather than a favour:
          * it applies to EVERY arm that puts a node into a relay mode.  The
            SDN/OSPF/OLSR baselines are locked to LOCAL_REROUTE with
            prb_relay_fraction = 0.20 (_lock_phy_mac_non_mcs in
            run_timeline_comparison.py) and the random-action control samples
            the same relay head, so all of them gain the same access capacity
            back.  Nothing here is MARL-only.
          * a relay hop is still NOT free.  Its capacity is still bounded by
            prb_relay_fraction x the transport link budget
            (TransportRelayModel.can_form_relay_link), re-pointing a beam
            still costs RELAY_REPOINT_TICKS of outage, flipping relay_mode
            still costs RELAY_CHURN_PENALTY, and dropping a bridge still costs
            BRIDGE_LOSS_PENALTY.  What is removed is only the spurious
            cross-band coupling to access capacity.
          * prb_relay_fraction keeps its full meaning as an agent control: it
            scales relay link capacity through relay_bw_fraction.

        After this call: prb_emergency_fraction + prb_general_fraction == 1.0
        and prb_relay_fraction is clamped to [0, 1].
        """
        access = self.prb_emergency_fraction + self.prb_general_fraction
        if access > 1e-6:
            self.prb_emergency_fraction /= access
            self.prb_general_fraction   /= access
        else:
            # Degenerate access split (both heads ~0) — fall back to the
            # nominal 30/70 emergency/general split.
            self.prb_emergency_fraction = 0.3
            self.prb_general_fraction   = 0.7
        self.prb_relay_fraction = max(0.0, min(1.0, self.prb_relay_fraction))

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
