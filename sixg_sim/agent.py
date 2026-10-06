"""
PHY/MAC MARL agents for autonomous 6G island-mode operation.

Each surviving O-RU / O-DU agent replaces the missing Near-RT RIC and
Core functions by directly controlling:
  - Tx power (coverage / interference tradeoff)
  - MCS per UE class (bounded OFFSET from auto CQI→MCS link adaptation;
    offset 0 = follow auto-CQI, negative = more robust, positive = more
    aggressive — see Simulator._step_phy_mac)
  - PRB allocation split (emergency / relay / general), a Dirichlet draw on
    the 2-simplex, from which per-class ADMISSION follows
  - Relay mode, 3 options (OFF / LOCAL_REROUTE / CAPACITY_BOOST)

RERUN_SPEC WP7: that is the whole action space -- four discrete heads plus
the split.  The MAC-scheduler, handover, postcard and isolated-operation
admission heads are DELETED (they actuated nothing); the relay head lost the
two options that were aliases of OFF.  See the N_* constants below.

Algorithm: MAPPO (Multi-Agent PPO) with centralised training,
           decentralised execution (CTDE).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum

import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from .topology import TrafficClass, NodeType
from .traffic import SliceDictionary
from .phy_mac_state import (
    PHYMACState, RelayMode, MCSLevel,
    TX_POWER_STEPS_DB, MCS_SPECTRAL_EFFICIENCY,
)


# ── Re-export legacy enums used by simulation.py ─────────────────────────────
# (kept for backward compatibility with metrics / dashboard code)

class AdmissionMode(Enum):
    ADMIT    = "admit"
    THROTTLE = "throttle"
    HOLD     = "hold"


class EnergyTier(Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class StrainLevel(Enum):
    OKAY      = "okay"
    DEGRADING = "degrading"
    NEAR_LIMIT = "near_limit"


# ── Observation dataclasses ───────────────────────────────────────────────────

@dataclass
class NeighborRadioSummary:
    """Aggregated radio state from received DCC postcards (Block B — 16 dims)."""
    # ── Original 12 dims ──────────────────────────────────────────────────
    avg_prb_util:           float = 0.0
    avg_sinr:               float = 0.5
    avg_backhaul_avail:     float = 1.0
    fraction_island:        float = 0.0
    best_relay_sinr:        float = 0.0
    best_relay_hop_norm:    float = 0.0
    neighbor_count_norm:    float = 0.0
    any_overloaded:         float = 0.0
    any_relay_active:       float = 0.0
    postcard_received:      float = 0.0
    policy_version_norm:    float = 0.0
    coordination_quality:   float = 0.0
    # ── ICIC additions (+4 dims) ─────────────────────────────────────────
    neighbour_avg_tx_power_norm:  float = 0.5   # avg Tx power of postcards [0,1]
    experienced_sinr_drop_norm:   float = 0.0   # my SINR drop since last tick [0,1]
    interferer_count_norm:        float = 0.0   # fraction of neighbours that boosted power
    cooperative_relay_assigned:   float = 0.0   # a fresh neighbour relays AND reports unserved UEs
    # ── Postcard freshness (RERUN_SPEC WP5 receiver latch; WP6 Block B) ──
    # 1 - (age of the freshest latched neighbour entry) / 30: 1.0 on the
    # tick a postcard arrives, falling to 0.0 when the last entry expires,
    # and 0.0 when this site has heard nothing fresh.  Written by
    # sixg_sim.postcards.fill_block_b; the actor reads it as
    # `b_neighbour_freshness` (it replaced the binary `postcard_received`
    # column).  `postcard_received` stays "a fresh postcard arrived THIS
    # tick" (XL-DET reads it by name).  Every Block B field above is built
    # from the FRESH latch entries, not from this tick's arrivals alone.
    postcard_freshness_norm:      float = 0.0


@dataclass
class ConnectivityState:
    """UE and inter-island connectivity state (Block C — 23 dims)."""
    # ── Original 12 dims ────────────────────────────────────────────────
    reachable_ue_fraction:        float = 1.0
    isolated_ue_count_norm:       float = 0.0
    active_relay_paths_norm:      float = 0.0
    ue_to_ue_routed_norm:         float = 0.0
    intra_island_reachability:    float = 1.0
    core_distance_norm:           float = 0.0
    bridge_node_flag:             float = 0.0
    island_size_norm:             float = 1.0
    potential_iab_capacity_norm:  float = 0.0
    handover_candidate_count_norm: float = 0.0
    ue_rsrp_min_norm:             float = 1.0
    ue_pair_demand_norm:          float = 0.0
    # ── IOPS additions (+2 dims) ─────────────────────────────────────────
    registration_capacity_norm:   float = 0.0   # fraction of IOPS slots used
    pending_iops_norm:            float = 0.0   # pending IOPS requests [0,1]
    # ── Multi-eNB IOPS additions (+8 dims, TS 22.346 V16.0.0) ────────────
    island_member_count_norm:     float = 0.0   # eNBs in island / max_possible
    island_ue_load_balance:       float = 0.5   # my UE count / island avg
    xn_mesh_density:              float = 0.0   # fraction of possible Xn links
    local_epc_health:             float = 1.0   # 0=down, 1=healthy
    nenb_count_norm:              float = 0.0   # nomadic eNBs in island / max
    peer_avg_reward:              float = 0.0   # never written: the learning-postcard
    peer_best_relay_hint:         float = 0.0   #   exchanger was REMOVED (RERUN_SPEC WP5)
    multi_island_bridge:          float = 0.0   # 1 if bridges two islands
    # ── Relay capability (+1 dim) ────────────────────────────────────────
    # 1.0 if THIS site owns a steerable MultiHaul radio, i.e. it is one of
    # the few nodes at which relay_mode=CAPACITY_BOOST can actually re-point
    # a beam and bridge two post-severance fragments.  Without this dim the
    # observation of a MultiHaul site is indistinguishable from that of a
    # plain O-RU, and because the actor is parameter-shared (one actor for
    # all ~42 agents, see MAPPOTrainer) the policy CANNOT condition its
    # relay head on relay capability: the CAPACITY_BOOST gradient from the
    # 3 sites where it pays is averaged against the ~39 sites where it only
    # burns PRB, so the shared policy converges to "never CAPACITY_BOOST".
    relay_capable:                float = 0.0   # 1 if node.has_multihaul


@dataclass
class AgentObservation:
    """Complete PHY/MAC observation for one O-RU or O-DU agent (60 dims)."""
    node_id:          str
    is_island:        bool
    energy_soc:       float          # [0, 1]
    phy_mac:          PHYMACState    # Block A (12 dims)
    neighbor_radio:   NeighborRadioSummary   # Block B (16 dims)
    connectivity:     ConnectivityState      # Block C (14 dims)
    ticks_since_severance: int       # Block D
    current_tick:     int
    # ── Coordinator policy vector (Block D extra) ──────────────────────
    global_policy:    Optional[List[float]] = None   # 5 floats from coordinator
    # ── Legacy slice state (for reward compatibility) ───────────────────
    local_slices:     Optional[Dict] = None
    energy_tier:      Optional[EnergyTier] = None
    neighbor_summary: Optional[Any] = None
    # ── Block D (WP6): the LOCALLY detectable event ───────────────
    # `local_isolated` is True when THIS site's component in the link-state
    # service's believed-up infrastructure graph holds no core node, as of
    # the PREVIOUS service step (for an arm with no service, the true graph
    # at the same phase: agent.local_isolation_state and
    # agent.advance_local_isolation); `ticks_since_local_isolation` is the number
    # of ticks since that became true (0 when not isolated).  These replace
    # the scripted `ticks_since_severance` / `is_island` pair in the actor
    # vector; both of those stay on the dataclass for diagnostics and for
    # the PRB coordinator clamp.
    local_isolated:   bool = False
    ticks_since_local_isolation: int = 0


# ── Action dataclasses ────────────────────────────────────────────────────────

@dataclass
class ControlPostcard:
    """DCC postcard carrying the sender's own PHY/MAC summary to its
    neighbours: declared encoded size 60 B, field by field in
    sixg_sim.postcards.POSTCARD_WIRE_LAYOUT (RERUN_SPEC WP5).

    Every field is the SENDER's own state.  `unserved_ue_count` replaced the
    network-wide isolated-UE census the postcard used to carry (an oracle no
    site could observe); the fragment label and bridge-holder bit were
    removed with Block E."""
    sender_id:              str
    relay_mode_active:      bool
    best_sinr_to_neighbour: float    # normalised [0,1]
    unserved_ue_count:      int      # the SENDER's own unserved UEs (PHYMACState.unserved_ue_count)
    prb_avail_for_relay:    float    # [0,1] the sender's relay PRB share
    policy_version:         int
    timestamp:              int
    # ICIC additions
    tx_power_dbm_norm:      float        = 0.5   # [0,1] — (tx-10)/33
    sinr_degradation_norm:  float        = 0.0   # my SINR dropped by this much [0,1]
    # Legacy fields
    most_needy_class:       TrafficClass = TrafficClass.LIFE_SAFETY
    need_level:             StrainLevel  = StrainLevel.OKAY


@dataclass
class PHYMACAction:
    """
    Structured action output from one O-RU / O-DU agent.
    Each field maps to a discrete or continuous control knob.
    """
    # Tx power step index: 0=−6 dB, 1=−3 dB, 2=0 dB, 3=+3 dB, 4=+6 dB
    tx_power_step:      int = 2

    # MCS OFFSET indices — interpreted relative to the auto CQI→MCS choice:
    #   offset = idx - MCS_OFFSET_CENTER  →  {0:-2, 1:-1, 2:0, 3:+1, 4:+2}
    #   final_mcs = clamp(auto_cqi_idx + offset, 0, n_mcs-1)
    # idx=2 (offset 0) = follow auto-CQI exactly (see Simulator._step_phy_mac)
    mcs_emergency_idx:  int = 1   # default: one step more robust than auto-CQI
    mcs_general_idx:    int = 2   # default: follow auto-CQI

    # PRB allocation — the EXECUTED split, after the coordinator clamp and
    # the general-traffic floor; fractions summing to 1
    prb_emergency_frac: float = 0.30
    prb_relay_frac:     float = 0.00
    prb_general_frac:   float = 0.70

    # The split as SAMPLED from the policy, before those two projections.
    # This is what PPO scores; see prb_training_action().
    prb_sampled:        Optional[List[float]] = None

    # Relay ENUM index into list(RelayMode):
    #   0=OFF, 1=NORMAL, 2=LOCAL_REROUTE, 3=CAPACITY_BOOST, 4=D2D_PEER_RELAY
    # This stays an ENUM index so that _step_phy_mac, the SDN dummy action,
    # the harness CAPACITY_BOOST gate and PHYMACState are untouched by the
    # WP7 head shrink.
    relay_mode_idx:     int = 0

    # Relay HEAD index into RELAY_HEAD_MODES (0=OFF, 1=LOCAL_REROUTE,
    # 2=CAPACITY_BOOST) -- what the 3-way policy head emitted and therefore
    # what PPO scores.  RERUN_SPEC WP7: NORMAL and D2D_PEER_RELAY were
    # behaviourally identical to OFF (no relay link, no relay capacity, and
    # D2D's only write was never read), so three of five options aliased one
    # another; see relay_enum_index()/relay_head_index() below.
    relay_head_idx:     int = 0

    # Postcard.  RERUN_SPEC WP7: there is NO postcard action head any more.
    # These two fields are populated by the ENVIRONMENT (the DCC heartbeat in
    # Simulator._execute_agent_actions) and are never scored by PPO.
    send_postcard:      bool = False
    postcard_content:   Optional[ControlPostcard] = None

    # ── Legacy fields kept for simulation.py compatibility ─────────────
    class_actions:      Optional[Dict] = None   # populated after action applied
    link_biases:        Dict = None

    def __post_init__(self):
        if self.link_biases is None:
            self.link_biases = {}


# ── Legacy: keep AgentAction as alias so old imports don't break ──────────────
AgentAction = PHYMACAction

# Legacy structures kept for ControlPlane / Metrics code
@dataclass
class ClassAction:
    admission_mode: AdmissionMode = AdmissionMode.ADMIT
    priority_weight: float = 1.0


@dataclass
class LocalSliceState:
    traffic_class: TrafficClass
    importance: float = 0.5
    current_queue_length: float = 0.0
    offered_load: float = 0.0
    admission_success_rate: float = 1.0
    freshness_target: int = 10
    current_freshness: float = 5.0


@dataclass
class NeighborSummary:
    most_needy_class: TrafficClass = TrafficClass.LIFE_SAFETY
    need_level: StrainLevel = StrainLevel.OKAY
    strain_level: StrainLevel = StrainLevel.OKAY
    latest_policy_version: int = 0
    neighbor_count: int = 0


# ── NamedTuples for experience replay ────────────────────────────────────────

Experience = NamedTuple("Experience", [
    ("observation",      AgentObservation),
    ("action",           PHYMACAction),
    ("reward",           float),
    ("next_observation", AgentObservation),
    ("done",             bool),
    ("info",             Dict[str, Any]),
])

MARLExperience = NamedTuple("MARLExperience", [
    ("global_observation",      Dict[str, AgentObservation]),
    ("global_action",           Dict[str, PHYMACAction]),
    ("global_reward",           Dict[str, float]),
    ("next_global_observation", Dict[str, AgentObservation]),
    ("done",                    bool),
    ("info",                    Dict[str, Any]),
])

RewardComponents = NamedTuple("RewardComponents", [
    ("qos_reward",          float),
    ("energy_reward",       float),
    ("coordination_reward", float),
    ("stability_reward",    float),
    ("total_reward",        float),
    # PHY/MAC specific
    ("connectivity_reward", float),
    ("coverage_reward",     float),
    ("interference_penalty", float),
])


# ── Neural network architecture ───────────────────────────────────────────────

# ══ OBSERVATION SCHEMA (RERUN_SPEC WP6) ═════════════════════════
# TWO ordered name/extractor lists, one per consumer:
#
#   ACTOR_OBS_FIELDS      (26)  what a partitioned site can produce on its
#                               own.  observation_to_tensor() returns
#                               exactly these, in this order.
#   CRITIC_GLOBAL_FIELDS  (11)  whole-network quantities the centralised
#                               critic is entitled to under CTDE.
#                               critic_features() returns the actor fields
#                               FOLLOWED BY these.
#
# WHAT THIS REPLACED, and why (reviewer items 3 and 6, implementation audit):
#
#   * NONLOCAL_OBS_COLUMNS.  The actor vector used to be the full 62-dim
#     record with 23 critic-only columns silently zeroed before every
#     forward pass, in training and in evaluation.  A third of the actor's
#     first-layer weights were multiplied by zero and nothing in the tensor,
#     the checkpoint or the logs said so.  Two lists cannot be masked: the
#     actor never builds a column it may not have.
#   * MARL_OBS_LOCALITY.  An environment variable could switch the evaluated
#     actor between two different observation spaces.  Setting it is now an
#     error (RERUN_SPEC section 1: no env switch changes the evaluated
#     system).
#   * The pad/truncate `(features + [0.0] * OBS_DIM)[:OBS_DIM]`.  It turned
#     every schema mistake into a silently zero-padded or silently dropped
#     column.  Both vectors now ASSERT their length against the list.
#   * Fields the audit proved carry no information, dropped here:
#     PRB utilisation (always 0), relay-capable (1 at every agent site), the
#     5-dim coordinator vector (a constant neutral vector), the never-written
#     Block B fields (best relay hop, overload indicator, policy version),
#     the constant backhaul-availability field, the leaky cooperative-relay
#     gate (a thresholded network-wide UE census), the duplicate minimum-SINR
#     proxy, the relay-activity indicator (a fixed function of two Block A
#     fields), the tick index, the IOPS registration-capacity field, the two
#     learning-exchanger peer fields, and Block E (which the postcard
#     analysis shows can never fire).
#   * The scripted `ticks_since_severance` timer, replaced by the LOCALLY
#     detectable Block D pair (local isolation flag + its own timer).
#
# The final actor list is fixed only after the WP13 constancy dump
# (tools/constancy_dump.py reports min/max/variance and the constant
# fraction of every field); nothing below may be added without re-running it.
# WP13 ran it at production length (D13.3: 3600 ticks, both scenarios,
# seeds 82, 50, 71, with the per-router LSA origination D13.1 and the
# per-site action lock D13.2 applied) and removed the four critic-global
# fields that were constant, at one value, in every pass:
# REMOVED_CONSTANT_FIELDS below (per-field evidence:
# docs/wp13/constancy_c148cc9.json).  No actor field was constant: the
# 600-tick passes had also shown a_emergency_fraction and g_pending_iops
# constant, but both MOVE in Scenario B at production length (the rescue
# force of tick 1400 brings emergency UEs), so both stay.
#
# Each entry is (name, extractor).  The extractor takes an AgentObservation
# and returns a float.  THE ORDER IS THE WIRE FORMAT: it is hashed into
# SCHEMA_ID, every checkpoint records that id, and the strict loader refuses
# a checkpoint whose id differs.

# ── The fragment cue is REMOVED (RERUN_SPEC WP5) ──────────────────────────
# The evaluated configuration appended two columns (the "fragment cue") and
# put the sender's fragment label and bridge bit into every postcard, under
# a switch.  Those labels were computed by the simulator from its global
# link table -- no flooding was modelled, delayed or billed -- i.e. an
# oracle, and the cue could not fire in steady state anyway (a postcard only
# travels over a live link, so it never comes from a disjoint fragment).
# The columns (WP6 had already dropped them from the actor), the postcard
# fields and the switch are gone.  Setting the switch now asks for behaviour
# this tree does not have, so it is refused instead of silently ignored;
# unset (or 0) is the only accepted value.
_REMOVED_FRAGMENT_SWITCH = os.environ.get('MARL_POSTCARD_FRAGMENTS')
if (_REMOVED_FRAGMENT_SWITCH is not None
        and _REMOVED_FRAGMENT_SWITCH.strip().lower()
        not in ('', '0', 'false', 'off', 'no')):
    raise ValueError(
        "MARL_POSTCARD_FRAGMENTS=%r: the fragment cue and the postcard "
        "fragment labels were REMOVED by RERUN_SPEC WP5 (an oracle the actor "
        "could not have observed).  Unset the variable." % _REMOVED_FRAGMENT_SWITCH)

# MARL_OBS_LOCALITY is removed, not honoured-and-ignored.
_OBS_LOCALITY_ENV = os.environ.get('MARL_OBS_LOCALITY')
if _OBS_LOCALITY_ENV is not None:
    raise ValueError(
        "MARL_OBS_LOCALITY=%r: the switch was REMOVED by RERUN_SPEC WP6. "
        "The actor vector is ACTOR_OBS_FIELDS and the critic vector is "
        "ACTOR_OBS_FIELDS + CRITIC_GLOBAL_FIELDS; there is no masking to "
        "turn on or off." % _OBS_LOCALITY_ENV)

# Block D: the local-isolation timer saturates after this many ticks.
LOCAL_ISOLATION_TIMER_TICKS = 200.0

# Block B: the neighbour-freshness field decays from 1.0 to 0.0 over the WP5
# receiver-latch horizon, postcards.POSTCARD_LATCH_TICKS (3 x the heartbeat
# period).  It has ONE definition, in sixg_sim/postcards.py, and is applied
# there by fill_block_b; this module keeps no copy of it.

_RELAY_MODE_CODE = {
    RelayMode.OFF:             0.0,
    RelayMode.NORMAL:          0.25,
    RelayMode.LOCAL_REROUTE:   0.5,
    RelayMode.CAPACITY_BOOST:  0.75,
    RelayMode.D2D_PEER_RELAY:  1.0,
}


def _clip01(x):
    x = float(x)
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _local_isolation_timer(o):
    """min(1, dt / 200), and exactly 0.0 while the site is not isolated."""
    if not o.local_isolated:
        return 0.0
    dt = max(0, int(o.ticks_since_local_isolation))
    return min(1.0, dt / LOCAL_ISOLATION_TIMER_TICKS)


# ── Block A (11): own radio state, measured at the site ───────────────
_BLOCK_A_FIELDS = [
    ('a_tx_power', lambda o: (
        (o.phy_mac.tx_power_dbm - o.phy_mac.tx_power_min_dbm)
        / max(1.0, o.phy_mac.tx_power_max_dbm - o.phy_mac.tx_power_min_dbm))),
    ('a_spectral_efficiency',
     lambda o: o.phy_mac.spectral_efficiency(for_emergency=False) / 4.0),
    ('a_sinr_mean', lambda o: _clip01((o.phy_mac.sinr_average + 5.0) / 35.0)),
    ('a_ue_count', lambda o: min(1.0, o.phy_mac.active_ue_count / 100.0)),
    ('a_emergency_fraction', lambda o: min(
        1.0, o.phy_mac.emergency_ue_count / max(1, o.phy_mac.active_ue_count))),
    ('a_coverage',
     lambda o: min(1.0, o.phy_mac.effective_coverage_radius_m / 2000.0)),
    # The two backhaul fields are fed by the WP2 end-of-pass per-link load
    # snapshot: "what my own non-access links carried on the last pass".
    ('a_own_link_max_util', lambda o: o.phy_mac.backhaul_utilization),
    ('a_own_link_min_avail',
     lambda o: min(1.0, o.phy_mac.backhaul_capacity_mbps / 1000.0)),
    ('a_relay_mode',
     lambda o: _RELAY_MODE_CODE.get(o.phy_mac.relay_mode, 0.0)),
    ('a_relay_link_active',
     lambda o: 1.0 if o.phy_mac.relay_link_active else 0.0),
    ('a_sinr_min', lambda o: _clip01((o.phy_mac.sinr_min + 5.0) / 35.0)),
]

# ── Block B (10): neighbour state, from received DCC postcards ────────
_BLOCK_B_FIELDS = [
    ('b_neighbour_relay_share', lambda o: o.neighbor_radio.avg_prb_util),
    ('b_neighbour_sinr_mean', lambda o: o.neighbor_radio.avg_sinr),
    ('b_neighbour_sinr_best', lambda o: o.neighbor_radio.best_relay_sinr),
    ('b_fresh_neighbour_count',
     lambda o: o.neighbor_radio.neighbor_count_norm),
    ('b_any_neighbour_relaying', lambda o: o.neighbor_radio.any_relay_active),
    ('b_neighbour_freshness',
     lambda o: o.neighbor_radio.postcard_freshness_norm),
    ('b_relaying_neighbour_fraction',
     lambda o: o.neighbor_radio.coordination_quality),
    ('b_neighbour_tx_power_mean',
     lambda o: o.neighbor_radio.neighbour_avg_tx_power_norm),
    ('b_own_sinr_drop',
     lambda o: o.neighbor_radio.experienced_sinr_drop_norm),
    ('b_interferer_fraction',
     lambda o: o.neighbor_radio.interferer_count_norm),
]

# ── Block C local (3): the three genuinely local connectivity fields ───
_BLOCK_C_FIELDS = [
    ('c_own_relay_capacity',
     lambda o: o.connectivity.potential_iab_capacity_norm),
    # From the SITE'S OWN believed-up infrastructure links (WP3), never from
    # the whole-topology graph.
    ('c_live_infra_neighbours',
     lambda o: o.connectivity.handover_candidate_count_norm),
    ('c_anchored_flow_demand', lambda o: o.connectivity.ue_pair_demand_norm),
]

# ── Block D (2): the locally detectable event, replacing the scripted timer
_BLOCK_D_FIELDS = [
    ('d_local_isolation', lambda o: 1.0 if o.local_isolated else 0.0),
    ('d_ticks_since_local_isolation', _local_isolation_timer),
]

ACTOR_OBS_FIELDS = (_BLOCK_A_FIELDS + _BLOCK_B_FIELDS
                    + _BLOCK_C_FIELDS + _BLOCK_D_FIELDS)

# ── Critic-only globals (11): legitimate under centralised training ───
# (four removed at WP13, D13.3 -- see REMOVED_CONSTANT_FIELDS)
CRITIC_GLOBAL_FIELDS = [
    ('g_active_relay_paths',
     lambda o: o.connectivity.active_relay_paths_norm),
    ('g_ue_to_ue_routed', lambda o: o.connectivity.ue_to_ue_routed_norm),
    ('g_intra_island_reachability',
     lambda o: o.connectivity.intra_island_reachability),
    ('g_core_distance', lambda o: o.connectivity.core_distance_norm),
    ('g_bridge_node_flag', lambda o: o.connectivity.bridge_node_flag),
    ('g_island_size', lambda o: o.connectivity.island_size_norm),
    ('g_pending_iops', lambda o: o.connectivity.pending_iops_norm),
    ('g_island_member_count',
     lambda o: o.connectivity.island_member_count_norm),
    ('g_island_ue_load_balance',
     lambda o: o.connectivity.island_ue_load_balance),
    ('g_xn_mesh_density', lambda o: o.connectivity.xn_mesh_density),
    ('g_local_epc_health', lambda o: o.connectivity.local_epc_health),
]

# ── Removed at WP13 (D13.3) ───────────────────────────────────────────
# Constant, at ONE value, in every pass of the production-length constancy
# dump (tools/constancy_dump.py --ticks 3600, recovery and rescue_ops, seeds
# 82, 50 and 71, at c148cc9; combined by tools/constancy_combine.py into
# docs/wp13/constancy_c148cc9.json).  A field constant in some pass but
# moving in another stays.  Listed by name so that
# tests/test_wp13_integration.py can prove that exactly these -- and no field
# that moves anywhere -- left the schema.  The ConnectivityState attributes
# behind them still exist (the engine computes them, XL-DET must not read
# them either); only the critic columns are gone.
REMOVED_CONSTANT_FIELDS = (
    'g_reachable_ue_fraction',     # 1.0 on every sample
    'g_isolated_ue_count',         # 0.0 on every sample
    'g_nenb_count',                # 0.0 on every sample
    'g_multi_island_bridge',       # 0.0 on every sample
)

ACTOR_OBS_NAMES = tuple(n for n, _ in ACTOR_OBS_FIELDS)
CRITIC_GLOBAL_NAMES = tuple(n for n, _ in CRITIC_GLOBAL_FIELDS)
CRITIC_OBS_NAMES = ACTOR_OBS_NAMES + CRITIC_GLOBAL_NAMES

ACTOR_OBS_DIM = len(ACTOR_OBS_FIELDS)     # 26
CRITIC_OBS_DIM = len(CRITIC_OBS_NAMES)    # 37

assert ACTOR_OBS_DIM == 26, ACTOR_OBS_DIM
assert len(CRITIC_GLOBAL_FIELDS) == 11, len(CRITIC_GLOBAL_FIELDS)
assert len(set(CRITIC_OBS_NAMES)) == CRITIC_OBS_DIM, 'duplicate field name'
assert not set(REMOVED_CONSTANT_FIELDS) & set(CRITIC_OBS_NAMES), \
    'a removed constant field is back in the schema'

# `OBS_DIM` keeps its old meaning "the actor network's input width", so that
# PolicyNetwork(OBS_DIM) reads the same everywhere; the critic takes
# CRITIC_OBS_DIM.  Neither is read from the environment any more.
OBS_DIM = ACTOR_OBS_DIM


def _compute_schema_id():
    """Stable id of the checkpoint CONTRACT: the observation schema AND the
    action space.  Stored in every checkpoint and in every result pickle's
    provenance (sixg_sim.runinfo.schema_id).

    RERUN_SPEC WP7 (D7.5).  The id used to cover only the two observation
    field lists and their widths.  That was not the whole contract: a
    checkpoint also encodes an action space, and WP7 changes it (four heads
    instead of eight, relay 3-wide instead of 5-wide).  Two trees that agree
    on every observation field and disagree about what the relay head means
    would have produced the SAME id, so a stale checkpoint would have been
    refused by a tensor-shape error deep in `load_state_dict` -- or, for a
    same-width change such as a re-ordering of the relay modes, silently
    ACCEPTED.  The head names and cardinalities are therefore part of the
    payload, in the order the network emits them.

    Called AFTER the action-head constants AND `PRB_POLICY` below are
    defined; the definition sits here so the observation-side names read in
    one place.
    """
    payload = chr(10).join([
        'actor=%d' % ACTOR_OBS_DIM,
        'critic=%d' % CRITIC_OBS_DIM,
        'actor_fields=' + ','.join(ACTOR_OBS_NAMES),
        'global_fields=' + ','.join(CRITIC_GLOBAL_NAMES),
        'discrete_heads=' + ','.join(
            '%s:%d' % (h, n) for h, n in zip(DISCRETE_HEADS, HEAD_SIZES)),
        'relay_modes=' + ','.join(m.name for m in RELAY_HEAD_MODES),
        # The PARAMETERISATION, not a constant.  `MARL_PRB_POLICY=softmax` is
        # a supported setting under which the PRB head is a deterministic
        # softmax point rather than a Dirichlet sample: a different action
        # distribution with the SAME tensor shape, so nothing downstream --
        # not load_state_dict, not a width check -- can tell the two apart.
        # Hard-coding 'dirichlet' here made the id claim to cover the action
        # space while silently accepting a checkpoint trained under the other
        # parameterisation, which is exactly the same-width failure mode the
        # relay-mode re-ordering paragraph above is written against (D7.12).
        'continuous_heads=prb:%s:3' % PRB_POLICY,
    ])
    return 'obs1-' + hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def actor_feature_values(obs) -> List[float]:
    """The ACTOR vector as a plain list, in ACTOR_OBS_FIELDS order."""
    values = [float(fn(obs)) for _name, fn in ACTOR_OBS_FIELDS]
    assert len(values) == ACTOR_OBS_DIM, (
        'actor vector is %d wide, ACTOR_OBS_FIELDS has %d entries'
        % (len(values), ACTOR_OBS_DIM))
    return values


def critic_feature_values(obs) -> List[float]:
    """The CRITIC vector: the actor fields, then the global fields."""
    values = actor_feature_values(obs)
    values += [float(fn(obs)) for _name, fn in CRITIC_GLOBAL_FIELDS]
    assert len(values) == CRITIC_OBS_DIM, (
        'critic vector is %d wide, expected %d'
        % (len(values), CRITIC_OBS_DIM))
    return values


# ══ LOCAL ISOLATION (WP6 Block D), read from the WP3 link-state service ══
# Site i is isolated exactly when its component in the link-state service's
# BELIEVED-UP infrastructure graph (the INSTALLED view, access adjacencies
# excluded) contains no core node.  The service computes that per site once
# per tick inside its own step (`LinkStateService._update_isolation`, called
# from `step`, which tick_step runs at phase 6, AFTER the action step), and
# `local_isolation_state` returns `LinkStateService.isolation_state(site)`.
# The observation builder runs at phase 4 of the NEXT tick, so every
# observation reads the state of the PREVIOUS step:
#   * Scenario B (severed before tick 0): tick 0 reads nothing (the service
#     has not stepped yet), every isolated site shows the flag from tick 1,
#     and its timer counts from the tick-0 step;
#   * Scenario A: the flag rises only after the service has INSTALLED the
#     severance (BFD detection, LSA flooding, SPF delay), i.e. later than the
#     engine's global island trigger -- which is what "locally detectable"
#     means.
# Every arm on the MARL code path has the service (make_link_state_service,
# both drivers).  The one exception is the pre-registered sensitivity
# MARL_DETECT_TICKS=0 (S10, "installation off, idealised timing"), whose
# forwarding view IS the true graph: for a simulator with NO service the SAME
# rule is applied to the TRUE infrastructure graph (`true_graph_isolation`),
# advanced by tick_step at the same phase 6 (`advance_local_isolation`), so
# the timing contract -- updated once per tick, idempotent, observations read
# the previous step -- is the same.  (A simulator driven outside step_tick
# with no service is never advanced; its flag stays 0.)
#
# The former stand-in, the engine's GLOBAL island-mode trigger read through
# `Simulator.action_lock_open()` (D6.4, D6.39), is deleted: it flagged every
# site at once, one tick after the event, whether or not the site's own
# component still held a core node.  Since D13.2 the ACTIONS are gated by
# the same per-site flag too (tick_step phase 5,
# `Simulator.update_site_action_locks`): a site acts once its own flag is
# set; `action_lock_open()` is the global trigger of the arms without
# agents only.


def link_state_service(sim):
    """The arm's WP3 link-state service -- the object
    `make_link_state_service` attaches as `sim.link_state_service` -- or
    None when the arm has none (baselines, MARL_DETECT_TICKS=0)."""
    svc = getattr(sim, 'link_state_service', None)
    return svc if callable(getattr(svc, 'isolation_state', None)) else None


def local_isolation_state(sim, site_id):
    """(is_isolated, isolated_since_tick) for `site_id`, as of the PREVIOUS
    step.  `isolated_since_tick` is None when the site is not isolated.

    With a service: exactly `svc.isolation_state(site_id)`.  Without one:
    the true-graph state `advance_local_isolation` last computed."""
    svc = link_state_service(sim)
    if svc is not None:
        return svc.isolation_state(site_id)
    return (getattr(sim, '_local_isolation', None) or {}).get(
        site_id, (False, None))


def true_graph_isolation(sim):
    """{site: isolated} for every agent site, on the TRUE infrastructure
    graph: the service's rule (component holds no core node; links with a UE
    endpoint are access, not infrastructure) with physical truth in place of
    the installed view.  Used only when the arm has no link-state service."""
    topo = sim.topology
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    is_ue = topo.is_ue_node
    up = topo._link_is_up
    for lk in topo.links.values():
        if not up(lk):
            continue
        a, b = lk.endpoints
        if is_ue(a) or is_ue(b):
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    core_roots = {find(c) for c in (getattr(sim, 'core_nodes', None) or ())}
    return {site: find(site) not in core_roots for site in sim.agents}


def advance_local_isolation(sim, tick):
    """Advance the NO-SERVICE local-isolation state by one tick (tick_step
    phase 6, the point where an arm with a service steps it).  A no-op when
    the arm has a service -- the service owns the state -- and IDEMPOTENT
    per tick.  Returns the number of sites it evaluated (0 when it did
    nothing), so a caller can tell a real update from a skipped one."""
    if link_state_service(sim) is not None:
        return 0                                # WP3 owns the state
    if getattr(sim, '_local_isolation_tick', None) == tick:
        return 0                                # already advanced this tick
    state = getattr(sim, '_local_isolation', None)
    if state is None:
        state = sim._local_isolation = {}
    now = true_graph_isolation(sim)
    for site, iso in now.items():
        was = state.get(site)
        if was is None or was[0] != iso:
            state[site] = (iso, tick if iso else None)
    sim._local_isolation_tick = tick
    return len(now)

# ── ACTION HEAD SIZES (RERUN_SPEC WP7) ──────────────────────────────────────
# FOUR discrete heads plus the Dirichlet PRB split.  `handover`, `scheduler`,
# `postcard` and `iops` are GONE -- not untrained, not masked: deleted from
# the network, the trainer, the simulator and every driver.  Why each went:
#
#   handover   PHYMACState.handover_triggered was written and never read.
#   scheduler  PHYMACState.mac_scheduler was written and never read
#              (SCHEDULER_EMERGENCY_WEIGHT in phy_mac_state.py is defined and
#              never used), so the choice touched no capacity, queue or
#              admission calculation.
#   postcard   the DCC postcard is a control-plane HEARTBEAT: under the
#              evaluated configuration it was sent at the rate limit whatever
#              the head chose, and the stored 'postcard' action was the
#              SCHEDULE's value, not the head's -- so pi_old and pi_new were
#              scored on an action the head did not take.  That is a
#              spurious advantage-weighted gradient into the shared trunk,
#              and it does not cancel across agents.  It is now an
#              environment rule (postcards.POSTCARD_PERIOD, WP5).
#   iops       under MARL_AGENT_FLEET=radio, THE EVALUATED CONFIGURATION,
#              the admission decision was read only at O-CU nodes, which
#              that fleet never gives an agent; even an admission only set
#              Node.iops_registered, which nothing reads.  (Not
#              unconditional: MARL_AGENT_FLEET=all_infra does put agents on
#              CU-type nodes, and there the block ran -- see D7.13.)
#              Per-class admission at radio sites comes from the PRB split
#              through Simulator._derive_admission_policies, UNCHANGED.
N_TX_STEPS    = len(TX_POWER_STEPS_DB)           # 5
# MCS heads keep cardinality len(MCSLevel) but encode OFFSETS from the
# auto-CQI MCS: idx - MCS_OFFSET_CENTER ∈ {-2..+2}.
N_MCS         = len(MCSLevel)                    # 5

# ── RELAY HEAD: 3 MODES, NOT 5 (RERUN_SPEC WP7) ─────────────────────────────
# OFF, NORMAL and D2D_PEER_RELAY had the SAME engine effect -- no relay link,
# no relay capacity, and D2D's only write (ps.d2d_active_pairs) is never read.
# They differed only in the observation code and in churn counting, so a
# uniform policy put 3/5 of its relay mass on "no relay" and switching among
# the aliases was charged as churn although nothing physical moved.
#
# THE MAPPING, recorded here and in docs/DECISIONS.md (D7.2):
#     head 0 -> RelayMode.OFF             (enum index 0)   [was: OFF]
#     head 1 -> RelayMode.LOCAL_REROUTE   (enum index 2)   [was: LOCAL_REROUTE]
#     head 2 -> RelayMode.CAPACITY_BOOST  (enum index 3)   [was: CAPACITY_BOOST]
#     dropped: RelayMode.NORMAL (enum 1) and D2D_PEER_RELAY (enum 4), both
#              aliases of OFF in the engine.
# PHYMACAction.relay_mode_idx keeps the ENUM index, so _step_phy_mac, the SDN
# dummy action, the harness CAPACITY_BOOST gate and PHYMACState need no edit;
# PHYMACAction.relay_head_idx carries the HEAD index, and that is what PPO
# scores and what the relay hysteresis compares.
RELAY_HEAD_MODES = [RelayMode.OFF, RelayMode.LOCAL_REROUTE,
                    RelayMode.CAPACITY_BOOST]
N_RELAY_MODES = len(RELAY_HEAD_MODES)            # 3
_RELAY_ENUM_ORDER = list(RelayMode)
RELAY_HEAD_TO_ENUM = [_RELAY_ENUM_ORDER.index(m) for m in RELAY_HEAD_MODES]
RELAY_ENUM_TO_HEAD = {e: h for h, e in enumerate(RELAY_HEAD_TO_ENUM)}
assert RELAY_HEAD_TO_ENUM == [0, 2, 3], RELAY_HEAD_TO_ENUM


def relay_enum_index(head_idx: int) -> int:
    """Head index (0..2) -> index into list(RelayMode)."""
    return RELAY_HEAD_TO_ENUM[int(head_idx)]


def relay_head_index(enum_idx: int) -> int:
    """Index into list(RelayMode) -> head index.  The two dropped aliases
    (NORMAL, D2D_PEER_RELAY) map onto OFF, which is what the engine already
    did with them."""
    return RELAY_ENUM_TO_HEAD.get(int(enum_idx), 0)


# ── POSTCARD HEARTBEAT (no longer an action) ────────────────────────────────
# The DCC postcard period is postcards.POSTCARD_PERIOD (10 ticks), one rule
# applied by the simulator after a site's action is executed
# (Simulator._send_postcard_heartbeat, RERUN_SPEC WP5).  No action builder in
# this module sends a postcard, so this module keeps no copy of the period.


# ── THE DISCRETE ACTION HEADS — ONE DEFINITION, TREE-WIDE ───────────────────
# RERUN_SPEC WP7.  This list is THE definition; `sixg_sim/mappo_trainer.py`
# imports it rather than keeping a second copy that has to be kept
# "byte-identical in content" by a test.  Four heads, and every one of them
# changes the environment:
#
#   tx_power   Simulator._step_phy_mac sets the site's transmit power.
#   mcs_emrg   MCS OFFSET from the auto-CQI choice, emergency classes.
#   mcs_gen    MCS OFFSET from the auto-CQI choice, general classes.
#   relay      3 modes (OFF / LOCAL_REROUTE / CAPACITY_BOOST) — the HEAD
#              index, see RELAY_HEAD_MODES.  This is the bridging decision.
#
# There is no UNTRAINED_HEADS list any more: a head the actor emits and the
# trainer does not train is exactly the dead weight WP7 removes.  The heads
# that used to sit here ('postcard', 'iops') and there ('handover',
# 'scheduler') are DELETED from the network; the reasons are recorded at the
# N_* constants above and in docs/DECISIONS.md D7.1.
#
# CHECKPOINT COMPATIBILITY: deliberately broken.  The *_head keys for the
# four deleted heads are gone and the relay head changes shape, so every
# pre-WP7 checkpoint fails the strict loader.  That is intended -- all four
# replicates are retrained from this tree -- and the schema id covers the
# action space precisely so that the refusal is a clear message rather than a
# shape error (D7.5).
DISCRETE_HEADS = ["tx_power", "mcs_emrg", "mcs_gen", "relay"]
MCS_LEVELS     = list(MCSLevel)
RELAY_MODES    = list(RelayMode)
HEAD_SIZES     = [N_TX_STEPS, N_MCS, N_MCS, N_RELAY_MODES]
assert len(HEAD_SIZES) == len(DISCRETE_HEADS)
assert HEAD_SIZES == [5, 5, 5, 3], HEAD_SIZES

# ── PRB HEAD PARAMETERISATION ────────────────────────────────────────────────
# The PRB split is a point on the 2-simplex.  PPO needs the log-DENSITY of the
# action it sampled; a bare softmax of the logits is a deterministic map with no
# density, and the cross-entropy that stood in for its log-probability is not a
# density ratio.  The same three logits are therefore read as the concentration
# of a Dirichlet, which keeps prb_head at nn.Linear(hidden, 3) so every existing
# checkpoint still loads with unchanged shapes.
#
#   MARL_PRB_POLICY=dirichlet  (default) sampled while training, mean at eval
#   MARL_PRB_POLICY=softmax    the previous deterministic behaviour, kept so
#                              that results published before this fix reproduce
PRB_POLICY = os.environ.get('MARL_PRB_POLICY', 'dirichlet').strip().lower()
if PRB_POLICY not in ('dirichlet', 'softmax'):
    raise ValueError(
        f"MARL_PRB_POLICY={PRB_POLICY!r} is not one of 'dirichlet', 'softmax'")

# alpha >= 1 keeps the density unimodal, so the mean is the mode and the
# evaluation-time action is the policy's actual preference.
PRB_CONCENTRATION_FLOOR = 1.0

# The checkpoint contract covers the action space too, so the id is computed
# only now that the heads AND the continuous head's parameterisation are
# known (see _compute_schema_id).  PRB_POLICY has to be read first: it names
# a different action DISTRIBUTION over the same tensor shape, so it is part
# of the contract and not a free runtime switch (D7.5, amended D7.12).
SCHEMA_ID = _compute_schema_id()


def prb_distribution(prb_logits: torch.Tensor) -> Dirichlet:
    """Dirichlet over (emergency, relay, general) from the raw PRB logits."""
    return Dirichlet(F.softplus(prb_logits) + PRB_CONCENTRATION_FLOOR)


def prb_training_action(action) -> List[float]:
    """The simplex point PPO must score.

    The executed allocation has passed through the coordinator minimum-emergency
    clamp and the general-traffic floor.  Those are properties of the
    environment, not of the policy, so the sampled point is what carries the
    density.  Falls back to the executed triple for actions produced before this
    field existed.
    """
    raw = getattr(action, 'prb_sampled', None)
    if raw:
        return list(raw)
    return [action.prb_emergency_frac, action.prb_relay_frac,
            action.prb_general_frac]


class PolicyNetwork(nn.Module):
    """
    Actor network for PHY/MAC MARL.

    Inputs:  ACTOR_OBS_DIM (26) actor observation vector
             (Block A 11 + Block B 10 + Block C 3 + Block D 2)
    Outputs: RERUN_SPEC WP7 -- FOUR discrete heads, tx_power (5),
             mcs_emrg (5), mcs_gen (5) and relay (3), plus the 3-parameter
             Dirichlet PRB split over the 2-simplex.  The handover,
             scheduler, postcard and IOPS-admission heads are DELETED, not
             merely untrained: see the N_* block above for why each one went
             and docs/DECISIONS.md D7.1.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

        # Discrete heads -- exactly DISCRETE_HEADS, in that order
        self.tx_power_head   = nn.Linear(hidden, N_TX_STEPS)
        self.mcs_emrg_head   = nn.Linear(hidden, N_MCS)
        self.mcs_gen_head    = nn.Linear(hidden, N_MCS)
        self.relay_head      = nn.Linear(hidden, N_RELAY_MODES)

        # PRB 3-way split (Dirichlet concentration logits)
        self.prb_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor):
        """Forward pass. x: (batch, ACTOR_OBS_DIM)."""
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))

        return {
            "tx_power":   self.tx_power_head(h),
            "mcs_emrg":   self.mcs_emrg_head(h),
            "mcs_gen":    self.mcs_gen_head(h),
            "relay":      self.relay_head(h),     # [B, 3] — HEAD index
            "prb":        self.prb_head(h),       # [B, 3] — Dirichlet outside
        }


class EnsemblePolicy(nn.Module):
    """Averaged-logit ensemble of independently trained PolicyNetworks.

    WHY.  Four training replicates of the same recipe converge to the same
    training reward (Fig. learning-curve, s.d. collapses 13x) yet differ
    sharply in the reunification configuration they FREEZE into at
    deployment: post-freeze bridge churn is zero, so residual components are
    decided by a static argmax, and one replicate (torch seed 2345) froze
    into a non-bridging configuration on three of seven reunifiable
    instances while another (3456) matched the deterministic XL-DET baseline.
    Averaging the actors' logits is the standard variance-reduction tool for
    exactly this: a site where three replicates prefer CAPACITY_BOOST and one
    prefers OFF is out-voted toward bridging, and no single unlucky
    initialisation can carry the deployment on its own.

    HONESTY.  This is model AVERAGING, not model SELECTION on the benchmark:
    every member is an already-trained policy, nothing is fitted to any
    evaluation seed, and the ensemble is compared on the same held-out
    validation seeds as the single checkpoints before anything is reported
    on the test benchmark.  Inference cost is N x a 125 kB network — trivial.

    forward() returns the same {head: logits} dict a PolicyNetwork does, so
    every consumer (batched sampling, log-prob, PPO update, the relay mask)
    sees an ordinary actor.  PRB logits are averaged too, which keeps the
    Dirichlet parameterisation unchanged (softplus of the mean logit).
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(list(members))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = [m(x) for m in self.members]
        return {h: torch.stack([o[h] for o in outs], dim=0).mean(dim=0)
                for h in outs[0]}


def checkpoint_schema_id(ckpt, reserved_key=None) -> Optional[str]:
    """The observation-schema id recorded in a checkpoint, or None.

    Checkpoints are written as
    {'state_dict': ..., 'metadata': {'provenance': {'schema_id': ...}}}
    (train_on_comparison, mappo_trainer, make_untrained_checkpoints); the
    two flatter spellings are accepted so a hand-built dict also works.

    An ADAPTED per-agent checkpoint is written as
    {agent_id: state_dict, ..., <reserved_key>: provenance} and carries
    its provenance under a driver-chosen reserved key rather than under
    'metadata' (run_timeline_comparison.ADAPTED_PROVENANCE_KEY).  Pass
    that key and the id is read from there, so the stamp survives the
    adaptation round trip.
    """
    if not isinstance(ckpt, dict):
        return None
    if reserved_key is not None:
        prov = ckpt.get(reserved_key)
        if isinstance(prov, dict) and prov.get('schema_id'):
            return prov['schema_id']
    meta = ckpt.get('metadata') or {}
    prov = meta.get('provenance') or {} if isinstance(meta, dict) else {}
    for src in (prov, meta, ckpt):
        if isinstance(src, dict) and src.get('schema_id'):
            return src['schema_id']
    return None


def assert_checkpoint_schema(ckpt, where='', reserved_key=None) -> str:
    """STRICT schema gate (RERUN_SPEC WP6).

    Refuses any checkpoint whose recorded schema id is not this tree's
    SCHEMA_ID -- including one with no id at all, which is every checkpoint
    written before WP6.  There is no adaptation path: the WP6 schema drops
    INTERIOR columns, so zero-padding or truncating fc1 would silently
    mis-align every column after the first drop.  Retrain instead.

    `reserved_key` is the caller's adapted-provenance key, if it has
    one (see `checkpoint_schema_id`); it lets the evaluation driver
    gate an adapted per-agent checkpoint through this same function
    instead of growing a gate of its own.
    """
    got = checkpoint_schema_id(ckpt, reserved_key=reserved_key)
    if got != SCHEMA_ID:
        raise ValueError(
            '%scheckpoint observation schema %r does not match this tree\'s '
            '%r (actor %d, critic %d).  The WP6 schema change drops interior '
            'columns, so this checkpoint cannot be adapted -- it must be '
            'retrained from the current tree.'
            % ((where + ': ') if where else '', got, SCHEMA_ID,
               ACTOR_OBS_DIM, CRITIC_OBS_DIM))
    return got


def load_policy_state_dict(net: nn.Module, state_dict: dict,
                           strict: bool = True) -> bool:
    """Load a state dict into a PolicyNetwork / CriticNetwork -- STRICTLY.

    RERUN_SPEC WP6.  The silent fc1 zero-pad / truncate that used to make a
    59-dim checkpoint "load" under a 60- or 62-dim tree is REMOVED: it only
    ever happened to be safe because every added column was TRAILING.  The
    WP6 schema removes interior fields, so the same trick would shift every
    later column by one and produce a policy that looks fine and is wrong.

    A width mismatch now raises.  Always returns False (nothing is ever
    adapted); the return value is kept so the call sites are unchanged.
    """
    w_key = 'fc1.weight'
    if w_key in state_dict and hasattr(net, 'fc1'):
        have_in = state_dict[w_key].shape[1]
        want_in = net.fc1.in_features
        if have_in != want_in:
            raise ValueError(
                'checkpoint fc1 has %d input columns, this tree needs %d '
                '(observation schema %s).  Zero-padding was removed by WP6 '
                'because the schema now drops interior columns; retrain the '
                'checkpoint from this tree.' % (have_in, want_in, SCHEMA_ID))
    net.load_state_dict(state_dict, strict=strict)
    return False


def mint_schema_current_checkpoint(path, torch_seed: int = 0,
                                   net: "PolicyNetwork" = None,
                                   extra: dict = None) -> str:
    """Write a checkpoint of THIS tree's network definition, and stamp it.

    RERUN_SPEC WP6, review round 3.  The strict gate correctly REFUSES
    every checkpoint written before WP6 -- including the only tracked
    evaluation checkpoint, `ce_ck_s1234/policy_final.pt` (fc1 62 wide, no
    provenance block).  Until a retrain produces a schema-current one,
    tests that need "a checkpoint both drivers can load" must MINT one
    rather than skip, so the contracts they cover keep being exercised.
    This is that helper, and it is the one place that knows the format.

    Deterministic in `torch_seed`: the same seed gives byte-identical
    weights, so a test that mints twice compares like with like.  The
    weights are an UNTRAINED initialisation -- this mints a checkpoint
    that LOADS, it does not mint a checkpoint that PERFORMS.

    The written top-level shape is `{state_dict, version, metadata}`, the
    same three keys `ModelRegistry.save` and `make_untrained_checkpoints`
    write.  The non-dict `version` value matters: the evaluation driver
    decides "per-agent vs legacy single-policy" with
    `all(isinstance(v, dict) for v in checkpoint.values())`, and a bare
    two-key `{state_dict, metadata}` file trips that test by accident.
    """
    from . import runinfo
    if net is None:
        # Seed LOCALLY: the global torch RNG state is restored
        # afterwards, so minting a checkpoint cannot shift the draws of
        # a run that happens later in the same process (the WP2 parity
        # test mints before it starts both drivers).
        _rng = torch.get_rng_state()
        try:
            torch.manual_seed(int(torch_seed))
            net = PolicyNetwork()
        finally:
            torch.set_rng_state(_rng)
    prov_extra = {'driver': 'mint_schema_current_checkpoint',
                  'torch_seed': int(torch_seed)}
    prov_extra.update(extra or {})
    payload = {
        'state_dict': {k: v.detach().clone()
                       for k, v in net.state_dict().items()},
        'version': 0,
        'metadata': {
            'episode': 0,
            'untrained': True,
            'minted': True,
            'torch_seed': int(torch_seed),
            'provenance': runinfo.provenance(extra=prov_extra),
        },
    }
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(payload, str(path))
    assert_checkpoint_schema(payload, where='minted %s' % path)
    return str(path)


class CriticNetwork(nn.Module):
    """
    Centralised critic: takes the mean-field global observation (the MEAN of
    all agents' CRITIC feature vectors, CRITIC_OBS_DIM dims — NOT a
    concatenated joint state) and returns V(s).

    WP6: the critic features are ACTOR_OBS_FIELDS followed by
    CRITIC_GLOBAL_FIELDS (`critic_features`), and they are the ONLY input
    the critic ever sees — at collection time and in the value loss alike.
    """

    def __init__(self, global_obs_dim: int = None, hidden: int = 256):
        if global_obs_dim is None:
            global_obs_dim = CRITIC_OBS_DIM
        super().__init__()
        self.fc1 = nn.Linear(global_obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.value_head(h).squeeze(-1)


# ── Base agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    def __init__(self, node_id: str, slice_dictionary: SliceDictionary):
        self.node_id         = node_id
        self.slice_dictionary = slice_dictionary
        self.policy_version  = 0

    @abstractmethod
    def compute_action(self, observation: AgentObservation) -> PHYMACAction:
        pass

    def update_policy_version(self):
        self.policy_version += 1


# ── RL Agent ──────────────────────────────────────────────────────────────────



# ── OBSERVATION LOCALITY ─────────────────────────────────
# NONLOCAL_OBS_COLUMNS, the 23-entry table of critic-only columns that were
# zeroed in the actor tensor, is GONE (RERUN_SPEC WP6).  Locality is now a
# property of the schema itself: ACTOR_OBS_FIELDS contains only fields a
# partitioned site can produce, CRITIC_GLOBAL_FIELDS contains the rest, and
# the two vectors are built from the two lists.  Nothing is masked, so
# nothing can be masked wrongly or silently.
#
# The dropped whole-topology fields are still computed on the
# ConnectivityState dataclass, because the critic reads them by name
# (CRITIC_GLOBAL_FIELDS) and XL-DET reads its own subset by name.


# ── Capability-conditioned relay-head mask (MARL_RELAY_MASK) ─────────
# REMOVED by WP6.  The mask read the actor's `relay_capable` column, which
# the audit showed is 1.0 at every agent site (so it never masked anything
# in the evaluated configuration, where the switch is off anyway) and which
# WP6 drops from the actor vector as a constant.  The helper stays as an
# identity so that every call site (mappo_trainer, simulation) is unchanged
# and the merge stays small; setting the variable is an error rather than a
# silent no-op.
RELAY_MASK = os.environ.get('MARL_RELAY_MASK', 'off').strip().lower()
if RELAY_MASK != 'off':
    raise ValueError(
        "MARL_RELAY_MASK=%r: the capability mask was REMOVED by RERUN_SPEC "
        "WP6 together with the constant `relay_capable` observation column. "
        "Only 'off' remains." % RELAY_MASK)

RELAY_CB_IDX = list(RelayMode).index(RelayMode.CAPACITY_BOOST)   # == 3


def apply_relay_mask(logits: Dict[str, torch.Tensor],
                     obs_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Identity.  Kept only so the call sites need no edit (see above)."""
    return logits


class RLAgent(BaseAgent):
    """
    PHY/MAC MARL agent (MAPPO actor).

    Observations → PolicyNetwork → PHYMACAction.
    Training via REINFORCE with discounted returns + gradient clipping.
    """

    obs_dim = OBS_DIM

    def __init__(self, node_id: str, slice_dictionary: SliceDictionary,
                 is_training: bool = True):
        super().__init__(node_id, slice_dictionary)
        self.is_training = is_training

        self.device = torch.device('cpu')
        self.policy_net = PolicyNetwork(OBS_DIM).to(self.device)
        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)

        self.experiences: List[Experience] = []
        self.policy_losses: List[float]    = []
        self.last_action: Optional[PHYMACAction] = None
        self.last_actions: Dict = {}           # legacy; used by convergence analysis
        self.last_postcard_tick = -100

        # Restoration-bonus hysteresis state (one-shot bonus, re-armed only
        # after connectivity stays below threshold for several ticks —
        # prevents farming the bonus by oscillating across the threshold).
        self._restore_bonus_armed  = True
        self._below_restore_ticks  = 0

    # ── Observation → tensor ─────────────────────────────────────────────

    def observation_to_tensor(self, obs: AgentObservation) -> torch.Tensor:
        """The ACTOR vector: exactly ACTOR_OBS_FIELDS, in order.

        RERUN_SPEC WP6.  There is no `mask_nonlocal` parameter any more and
        no pad/truncate: the actor never builds a column it may not have,
        and a length mismatch is an assertion, not a silent reshape.  The
        CRITIC input is built by `critic_features` from a different list.
        """
        return torch.tensor(actor_feature_values(obs),
                            dtype=torch.float32).to(
            getattr(self, 'device', torch.device('cpu')))

    def critic_features(self, obs: AgentObservation) -> torch.Tensor:
        """The CRITIC vector: the 26 actor fields, then the 11 global ones.

        This is the ONLY input the centralised critic is ever evaluated on
        -- at collection time (V(s_t) for GAE) and in the value loss alike.
        Before WP6 the stored critic input was the unmasked 62-dim mean
        while V(s_t) was computed on the MASKED actor mean, so the value
        that entered the advantage came from a different input distribution
        than the one the critic was fitted on.
        """
        return torch.tensor(critic_feature_values(obs),
                            dtype=torch.float32).to(
            getattr(self, 'device', torch.device('cpu')))

    # RLAgent.evaluate_actions is DELETED (RERUN_SPEC WP8 Part A, D8.1).
    #
    # It advertised itself as "for MAPPO update: compute (new_log_probs,
    # entropy) for a batch of stored (obs, actions)" and NOTHING in the
    # production path ever called it: the live PPO update is
    # MAPPOTrainer.update (sixg_sim/mappo_trainer.py), which computes its own
    # log-probs and entropy inline.  The two had also DIVERGED — this method
    # scored the PRB head with the softmax cross-entropy surrogate, while the
    # live update uses `_prb_log_prob` / the Dirichlet density and adds
    # `_prb_dist(...).entropy().mean()` to the entropy term.  A contract test
    # had been pinned against it, so the entropy contract was being asserted
    # of a function that does not govern training.  The contract now pins
    # MAPPOTrainer.update itself (tests/test_wp7_action_space.py), and the
    # convenience copy is gone so nothing can be pinned against it again.

    # ── Action sampling ──────────────────────────────────────────────────

    def compute_action(self, observation: AgentObservation) -> PHYMACAction:
        obs_t = self.observation_to_tensor(observation).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy_net(obs_t)
        logits = apply_relay_mask(logits, obs_t)

        def pick(head_name: str) -> int:
            l = logits[head_name].squeeze(0)
            if self.is_training:
                temp = getattr(self, '_logit_temperature', 1.0)
                return Categorical(logits=l / temp).sample().item()
            return torch.argmax(l).item()

        tx_step     = pick("tx_power")
        mcs_e_idx   = pick("mcs_emrg")
        mcs_g_idx   = pick("mcs_gen")
        relay_head  = pick("relay")            # HEAD index (0..2)
        relay_idx   = relay_enum_index(relay_head)

        # PRB split — Dirichlet over the 2-simplex (see PRB_POLICY)
        prb_logits = logits["prb"].squeeze(0)
        if PRB_POLICY == 'dirichlet':
            _prb_dist = prb_distribution(prb_logits)
            _prb_raw  = _prb_dist.sample() if self.is_training else _prb_dist.mean
            prb_fracs = _prb_raw.tolist()
        else:
            prb_fracs = F.softmax(prb_logits, dim=-1).tolist()
        prb_sampled = list(prb_fracs)

        # Enforce coordinator's minimum emergency PRB guarantee
        if observation.global_policy and observation.is_island:
            min_emrg = observation.global_policy[2]   # index 2 = min_emergency_prb
            if prb_fracs[0] < min_emrg:
                deficit   = min_emrg - prb_fracs[0]
                prb_fracs[0] += deficit
                # Take deficit equally from relay and general
                prb_fracs[1] = max(0.0, prb_fracs[1] - deficit / 2)
                prb_fracs[2] = max(0.0, prb_fracs[2] - deficit / 2)
                total = sum(prb_fracs)
                prb_fracs = [f / total for f in prb_fracs]

        # Safety: enforce minimum general-traffic PRB floor to prevent
        # degenerate "zero forwarding" policy (reward-hacking prevention)
        _PRB_GENERAL_FLOOR = 0.10
        if prb_fracs[2] < _PRB_GENERAL_FLOOR:
            deficit = _PRB_GENERAL_FLOOR - prb_fracs[2]
            prb_fracs[2] = _PRB_GENERAL_FLOOR
            # Take deficit proportionally from emergency and relay
            other_sum = prb_fracs[0] + prb_fracs[1]
            if other_sum > 1e-6:
                prb_fracs[0] -= deficit * (prb_fracs[0] / other_sum)
                prb_fracs[1] -= deficit * (prb_fracs[1] / other_sum)
            prb_fracs = [max(0.0, f) for f in prb_fracs]
            total = sum(prb_fracs)
            prb_fracs = [f / total for f in prb_fracs]

        # NO POSTCARD HERE (RERUN_SPEC WP5, WP7).  The DCC postcard is a
        # control-plane HEARTBEAT -- one per agent site every
        # postcards.POSTCARD_PERIOD ticks, an environment rule; WP7 deleted
        # the postcard head -- and the simulator sends it after executing
        # this action (Simulator._send_postcard_heartbeat).  No builder of an
        # action touches a postcard or `last_postcard_tick`.
        postcard_do = False
        postcard_content = None

        action = PHYMACAction(
            tx_power_step=tx_step,
            mcs_emergency_idx=mcs_e_idx,
            mcs_general_idx=mcs_g_idx,
            prb_emergency_frac=prb_fracs[0],
            prb_relay_frac=prb_fracs[1],
            prb_general_frac=prb_fracs[2],
            prb_sampled=prb_sampled,
            relay_mode_idx=relay_idx,
            relay_head_idx=relay_head,
            send_postcard=postcard_do,
            postcard_content=postcard_content,
        )
        self.last_action = action
        return action

    def action_from_logits_row(self,
                               logits: dict,
                               row: int,
                               observation: 'AgentObservation',
                               is_training: bool) -> 'PHYMACAction':
        """
        Build a PHYMACAction from a specific row of batched logit tensors.
        Called by Simulator.batch_compute_actions() — avoids a full forward pass
        per agent.

        Args:
            logits       Dict[head_name -> Tensor(N, C)] from batched forward
            row          Index into the batch for this agent
            observation  Full AgentObservation (used for PRB coordinator fix + postcard)
            is_training  Whether to sample or argmax
        """
        def pick(head_name: str) -> int:
            l = logits[head_name][row]           # shape (C,)
            if is_training:
                return Categorical(logits=l).sample().item()
            return torch.argmax(l).item()

        tx_step     = pick("tx_power")
        mcs_e_idx   = pick("mcs_emrg")
        mcs_g_idx   = pick("mcs_gen")
        relay_head  = pick("relay")            # HEAD index (0..2)
        relay_idx   = relay_enum_index(relay_head)

        if PRB_POLICY == 'dirichlet':
            _prb_dist = prb_distribution(logits["prb"][row])
            _prb_raw  = _prb_dist.sample() if is_training else _prb_dist.mean
            prb_fracs = _prb_raw.tolist()
        else:
            prb_fracs = F.softmax(logits["prb"][row], dim=-1).tolist()
        prb_sampled = list(prb_fracs)

        # Coordinator PRB floor
        if observation.global_policy and observation.is_island:
            min_emrg = observation.global_policy[2]
            if prb_fracs[0] < min_emrg:
                deficit = min_emrg - prb_fracs[0]
                prb_fracs[0] += deficit
                prb_fracs[1] = max(0.0, prb_fracs[1] - deficit / 2)
                prb_fracs[2] = max(0.0, prb_fracs[2] - deficit / 2)
                total = sum(prb_fracs)
                prb_fracs = [f / total for f in prb_fracs]

        # NO POSTCARD HERE (RERUN_SPEC WP5, WP7).  The DCC postcard is a
        # control-plane HEARTBEAT -- one per agent site every
        # postcards.POSTCARD_PERIOD ticks, an environment rule; WP7 deleted
        # the postcard head -- and the simulator sends it after executing
        # this action (Simulator._send_postcard_heartbeat).  No builder of an
        # action touches a postcard or `last_postcard_tick`.
        postcard_do = False
        postcard_content = None

        action = PHYMACAction(
            tx_power_step=tx_step,
            mcs_emergency_idx=mcs_e_idx,
            mcs_general_idx=mcs_g_idx,
            prb_emergency_frac=prb_fracs[0],
            prb_relay_frac=prb_fracs[1],
            prb_general_frac=prb_fracs[2],
            prb_sampled=prb_sampled,
            relay_mode_idx=relay_idx,
            relay_head_idx=relay_head,
            send_postcard=postcard_do,
            postcard_content=postcard_content,
        )
        self.last_action = action
        return action

    # ── Reward ───────────────────────────────────────────────────────────
    # ── Local reward constants: NONE.  RERUN_SPEC WP8 ────────────────────
    #
    # Every rate of the agent-local reward lives in ONE object --
    # `sixg_sim/reward_spec.py::RewardCoefficients`, fields prefixed
    # `local_` -- and the frozen set is rendered verbatim into
    # `docs/REWARD_FROZEN.md`.  `calculate_reward` MEASURES and DELEGATES.
    #
    # Each local rate is the PER-NODE attribution of a term the team
    # objective already pays fleet-wide, at the SAME rate and against the
    # same denominator, so a byte is worth the same to the agent as to the
    # fleet.

    def calculate_reward(self, obs: AgentObservation,
                         action: PHYMACAction,
                         next_obs: AgentObservation) -> RewardComponents:
        """PER-AGENT DIFFERENCE REWARD (manuscript eq:local), measured here
        and priced in `sixg_sim/reward_spec.py`.

        RERUN_SPEC WP8.  This method holds NO coefficient: it reads this
        node's own quantities into a `reward_spec.LocalRewardInputs` and
        returns `reward_spec.local_reward(...)`, term by term.

        WHY A DIFFERENCE REWARD AT ALL.  The reward that preceded it,
            R = 0.40 r_conn + 0.15 r_cov + 0.10 r_qos + 0.15 r_iops
                + 0.10 r_energy,
        was measured on the Scenario-A comparison environment (seed 42, 200
        island ticks, 35 agents) to have a CROSS-AGENT standard deviation of
        0.000000 in four of its five terms: r_conn and r_cov were global
        scalars handed to every node, r_qos was pinned at 1.0 because 6125 of
        6125 sampled slices had zero offered load, and r_iops resolved to the
        no-island default on this topology.  The only per-agent signal was
        the Tx-power energy cost, so the sole credit-assigned gradient every
        agent received was "turn your transmitter down".  End to end the
        per-agent reward had a cross-agent sd of 0.0052 against a cross-tick
        sd of 23.19, 94.8 % of the advantage variance was explained by the
        TICK alone, and the joint R^2 of all of an agent's own action
        one-hots on its own advantage was 0.0052.

        Each agent is now paid for the part of the SHARED objective its OWN
        node produced, at the team objective's own per-unit rates.  A true
        COMA counterfactual was rejected on cost (~42x a tick, i.e. ~3500x a
        training episode), not on principle; these terms are the observable
        per-node decomposition of the quantities it would have priced.

        THE TERMS, and what makes each one this node's own:

          + local_delivery x           this node's carried DELIVERED UE-to-UE
            delivered_share_ue_to_ue   volume over the tick's OFFERED volume
            (+ local_general_delivery  -- the local counterpart of the
             x delivered_share_general) dominant team term, which otherwise
                                       cancels out of the actor's gradient
                                       exactly, because the counterfactual
                                       baseline subtracts the cross-agent
                                       mean and a term identical for every
                                       agent survives that as zero
          + relay throughput           traffic on THIS node's own relay
                                       links, at the team rate, capped
          + local_bridge_counterfactual fragments THIS site's own bridges
            x fragments                eliminate (TREE EDGES only, so
                                       redundant or cyclic bridging pays
                                       nothing), paid every tick held.  This
                                       is the reunification signal that
                                       survives cross-agent centring; the
                                       team's reunify_* bonuses do not.
          - local_relay_churn          THIS site's EXECUTED relay mode
                                       changed.  EXECUTED, not sampled: see
                                       `reward_spec.executed_relay_mode`.  A
                                       WP4 engine hold reverts the mode in
                                       `_step_phy_mac`, so a blocked teardown
                                       is environment dynamics and is not
                                       charged to the policy.
          - local_mcs_overreach        THIS node's applied MCS exceeded what
                                       its own SINR supports and forced an
                                       OLLA step-down this tick.
          - energy cost                THIS agent's Tx-power step.

        NO REWARD HACKING, term by term: delivery pays only for volume
        actually DELIVERED through this node, against the offered-volume
        denominator, so a node cannot raise its share by suppressing anyone
        else's demand; relay throughput is gated on delivered UE traffic
        carried on this node's own links, so it cannot be farmed by switching
        relay on and idling nor by relaying the node's own O&M telemetry; the
        bridge term uses the de-duplicated (tree-edge) set; churn, MCS
        over-reach and energy are pure COSTS; and nothing pays for relay MODE
        or relay CAPACITY per se.

        REMOVED BY WP8 (see `reward_spec.REMOVED_TERMS`): `r_qos`, constant
        at 0.0 because no LIFE_SAFETY load is ever offered in the comparison
        topology; the flat `relay_contrib` payment, zero-rated in the
        evaluated configuration; and `bridge_held`, zero-rated because the
        per-site counterfactual credit replaced it.  Deleted, not zeroed.

        UNITS.  `total_reward` is in the SAME points as the team objective
        (typical per-node range about [-3, +22]), which is what makes the
        kappa = 0.6 mix mean what it says.

        ARM-NEUTRAL.  This is the MARL TRAINING signal only.  No physics, no
        capacity, no admission rule and no arm's behaviour depends on it; the
        routing baselines and the random control never evaluate it.
        """
        from . import reward_spec as _RS

        ps = next_obs.phy_mac
        _coef = _RS.FROZEN

        # ── EXECUTED relay mode, and whether it changed ───────────────────
        # Compared against THIS agent's own stored previous EXECUTED mode,
        # not against obs.phy_mac: `obs` and `next_obs` share one live
        # PHYMACState object, so its relay_mode field is identical in both
        # and could never show a change.
        _executed = _RS.executed_relay_mode(ps)
        _prev_executed = getattr(self, '_prev_executed_relay_mode', None)
        _mode_changed = (_prev_executed is not None
                         and _prev_executed != _executed)
        self._prev_executed_relay_mode = _executed

        # ── Own MCS over-reach (OLLA step-down forced by own MCS offset) ──
        fb_now = int(getattr(ps, 'mcs_fallback_events', 0))
        fb_prev = getattr(self, '_prev_mcs_fallback_events', None)
        _stepped_down = fb_prev is not None and fb_now > fb_prev
        self._prev_mcs_fallback_events = fb_now

        terms = _RS.local_reward(_RS.LocalRewardInputs(
            delivered_share_ue_to_ue=float(
                getattr(ps, 'delivered_share_ue_to_ue', 0.0) or 0.0),
            delivered_share_general=float(
                getattr(ps, 'delivered_share_general', 0.0) or 0.0),
            in_relay_mode=_executed in (RelayMode.LOCAL_REROUTE,
                                        RelayMode.CAPACITY_BOOST),
            relay_carried_mbps=float(
                getattr(ps, 'relay_traffic_carried_mbps', 0.0) or 0.0),
            bridge_counterfactual_fragments=int(
                getattr(ps, 'bridge_counterfactual_fragments', 0) or 0),
            relay_mode_changed=_mode_changed,
            mcs_stepped_down=_stepped_down,
            tx_power_step_db=float(TX_POWER_STEPS_DB[action.tx_power_step]),
        ), _coef)

        self._reward_terms = terms.as_dict()

        # RewardComponents is the NamedTuple the drivers already consume; the
        # groupings are unchanged so no caller has to move.  `qos_reward` is
        # structurally 0.0 now that the term is deleted (WP8); the field stays
        # because the drivers and `test_mission_config.py` read the tuple by
        # name and removing a field is a change this WP does not need.
        return RewardComponents(
            qos_reward=0.0,
            energy_reward=terms.energy,
            coordination_reward=terms.relay_throughput
            + terms.bridge_counterfactual,
            stability_reward=terms.relay_churn + terms.mcs_overreach,
            total_reward=terms.total,
            connectivity_reward=terms.delivery + terms.general_delivery,
            coverage_reward=terms.relay_throughput,
            interference_penalty=0.0,   # interference modelling not implemented
        )

    # ── Policy update (REINFORCE + discounted returns) ────────────────────

    def update_policy(self, batch_size: int = 64, epochs: int = 10,
                      gamma: float = 0.99):
        """
        DEPRECATED: legacy per-agent REINFORCE path.  The maintained training
        path is MAPPOTrainer (PPO + per-agent GAE); this method is kept only
        for backward compatibility and will be removed.
        """
        import warnings
        warnings.warn(
            "RLAgent.update_policy() is deprecated — use MAPPOTrainer instead",
            DeprecationWarning, stacklevel=2,
        )
        if not self.is_training or len(self.experiences) < batch_size:
            return

        # Build tensors
        obs_list, reward_list = [], []
        action_indices: Dict[str, List[int]] = {
            h: [] for h in DISCRETE_HEADS
        }
        prb_list: List[List[float]] = []

        for exp in self.experiences:
            obs_list.append(self.observation_to_tensor(exp.observation))
            reward_list.append(exp.reward)
            act = exp.action
            action_indices["tx_power"].append(act.tx_power_step)
            action_indices["mcs_emrg"].append(act.mcs_emergency_idx)
            action_indices["mcs_gen"].append(act.mcs_general_idx)
            # the HEAD index, which is what the 3-way relay head emitted
            action_indices["relay"].append(int(act.relay_head_idx))
            prb_list.append([act.prb_emergency_frac,
                             act.prb_relay_frac,
                             act.prb_general_frac])

        obs_tensor = torch.stack(obs_list)                          # (T, 40)
        prb_target = torch.tensor(prb_list, dtype=torch.float32)   # (T, 3) — target fracs

        # Discounted returns
        returns, G = [], 0.0
        for r in reversed(reward_list):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Convert action index lists to tensors
        idx_tensors = {
            h: torch.tensor(v, dtype=torch.long)
            for h, v in action_indices.items()
        }

        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits = self.policy_net(obs_tensor)

            # ── Discrete log-probs ─────────────────────────────────────
            log_probs = torch.zeros(len(self.experiences))
            entropy = torch.zeros(len(self.experiences))
            for head in DISCRETE_HEADS:
                dist = Categorical(logits=logits[head])
                log_probs = log_probs + dist.log_prob(idx_tensors[head])
                entropy = entropy + dist.entropy()

            # ── PRB allocation: cross-entropy vs. target fractions ─────
            prb_log_soft = F.log_softmax(logits["prb"], dim=-1)   # (T, 3)
            prb_ce       = -(prb_target * prb_log_soft).sum(dim=-1)  # (T,)
            log_probs    = log_probs - prb_ce   # include PRB in total log-prob

            # Actor loss (no artificial x100 scaling — that saturated the
            # gradient-norm clip on every step) + entropy bonus
            loss = -(log_probs * returns_t).mean() - 0.5 * entropy.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_losses.append(loss.item())
        self.policy_version += 1
        self.experiences.clear()

    # ── Deployment mode ──────────────────────────────────────────────────

    def set_training_mode(self, lr: float = 3e-4, entropy: float = 0.01):
        """High-LR training (simulation phase)."""
        self.is_training = True
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def set_deployment_mode(self, lr: float = 3e-5):
        """Low-LR online fine-tuning (post-deployment)."""
        self.is_training = True   # still collect/train, but cautiously
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        print(f"[RLAgent {self.node_id}] Deployment mode: lr={lr}")

    # ── Save / Load ──────────────────────────────────────────────────────

    def save_policy(self, filepath: str):
        """Write a checkpoint in THE tree's one checkpoint format.

        RERUN_SPEC WP6: `{'state_dict': ..., 'version': ..., 'metadata':
        {'provenance': runinfo.provenance()}}`, byte-for-byte the shape
        `train_on_comparison.py`, `make_untrained_checkpoints.py` and
        `mint_schema_current_checkpoint` already write, so the provenance
        block carries this tree's `schema_id`.  This used to write a BARE
        state dict with no metadata and no id, while `load_policy` runs
        `assert_checkpoint_schema` unconditionally: the pair was
        self-inconsistent and every checkpoint this method produced was
        permanently unloadable.  The round trip is covered by
        `tests/test_wp6_checkpoint_gate.py`.

        The non-dict `version` key is load-bearing, not decoration (review
        round 3): the evaluation driver decides "per-agent vs legacy
        single-policy" with `all(isinstance(v, dict) for v in ck.values())`,
        and a bare two-key `{state_dict, metadata}` file passes that test by
        accident -- it was read through the per-agent FALLBACK branch and
        reported as "per-agent, direct match 0", loading the right weights
        only because dict insertion order happened to put `state_dict`
        first.  See `mint_schema_current_checkpoint`, which says the same.
        """
        from . import runinfo
        torch.save({'state_dict': self.policy_net.state_dict(),
                    'version': 0,
                    'metadata': {'provenance': runinfo.provenance()}},
                   filepath)

    def load_policy(self, filepath: str):
        """Load a checkpoint into this agent's actor.

        RERUN_SPEC WP6: no `except Exception: print(...)`.  A checkpoint
        that does not match this tree's observation schema used to be
        reported on stdout and then silently ignored, leaving the agent
        running its RANDOM INITIALISATION -- which looks exactly like a
        loaded policy from the outside.  It raises now.
        """
        checkpoint = torch.load(filepath, weights_only=False)
        assert_checkpoint_schema(checkpoint, where=filepath)
        state_dict = (checkpoint.get('state_dict', checkpoint)
                      if isinstance(checkpoint, dict) else checkpoint)
        load_policy_state_dict(self.policy_net, state_dict)
        self.policy_version += 1000


# ── Centralised MARL Trainer ──────────────────────────────────────────────────

class CentralizedMARLTrainer:
    """Centralised critic trainer for MAPPO.

    NOT INSTANTIATED anywhere in this tree -- `MAPPOTrainer` is the live
    trainer; this class is only imported by name in
    `sixg_sim/simulation.py`.  It is kept (rather than deleted) because
    deleting it would mean editing that engine-track-owned import line
    for no functional gain; its critic input was converted to the WP6
    mean-field CRITIC vector instead, so the file no longer ships the
    pre-WP6 contract.  See docs/DECISIONS.md D6.24.
    """

    def __init__(self, agents: Dict[str, "RLAgent"],
                 critic_net: CriticNetwork,
                 learning_rate: float = 1e-3):
        self.original_agents = {
            aid: a for aid, a in agents.items() if isinstance(a, RLAgent)
        }
        self.agents    = agents
        self.critic_net = critic_net
        self.critic_opt = torch.optim.Adam(critic_net.parameters(), lr=learning_rate)
        self.global_experiences: List[MARLExperience] = []

    def collect_experience(self,
                           global_obs:    Dict[str, AgentObservation],
                           global_action: Dict[str, PHYMACAction],
                           global_reward: Dict[str, float],
                           next_obs:      Dict[str, AgentObservation],
                           done: bool, info: Dict):
        self.global_experiences.append(MARLExperience(
            global_observation=global_obs,
            global_action=global_action,
            global_reward=global_reward,
            next_global_observation=next_obs,
            done=done,
            info=info,
        ))
        # Keep buffer bounded
        if len(self.global_experiences) > 5000:
            self.global_experiences = self.global_experiences[-2500:]

    def update_critic(self, batch_size: int = 64) -> Optional[float]:
        if len(self.global_experiences) < batch_size:
            return None

        indices = np.random.choice(len(self.global_experiences), batch_size, replace=False)
        batch   = [self.global_experiences[i] for i in indices]

        global_obs_batch, rewards_batch = [], []
        for exp in batch:
            # RERUN_SPEC WP6 (D6.24).  The critic input is the MEAN-FIELD
            # CRITIC feature vector -- CRITIC_OBS_DIM (37) wide, the 26
            # actor fields followed by the 11 global ones -- exactly what
            # `CriticNetwork` is built for and what `MAPPOTrainer` stores.
            # This used to CONCATENATE every agent's 26-wide ACTOR vector
            # into an (N_agents x 26) row and pad absent agents at the
            # ACTOR width, which is the pre-WP6 contract D6.6 removed: it
            # is a shape error against every live CriticNetwork in the
            # tree, and it fed the actor's vector to the critic.
            feats = []
            for aid in sorted(self.original_agents.keys()):
                agent = self.original_agents[aid]
                if aid in exp.global_observation:
                    feats.append(
                        agent.critic_features(exp.global_observation[aid]))
                else:
                    feats.append(torch.zeros(CRITIC_OBS_DIM))
            if not feats:
                continue
            global_obs_batch.append(
                torch.stack(feats).mean(dim=0).tolist())
            rewards_batch.append(sum(exp.global_reward.values()))

        if not global_obs_batch:
            return None
        obs_t  = torch.tensor(global_obs_batch, dtype=torch.float32)
        rew_t  = torch.tensor(rewards_batch,    dtype=torch.float32)

        v_pred = self.critic_net(obs_t)
        loss   = F.mse_loss(v_pred, rew_t)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.item()

    def train_agents(self, batch_size: int = 64, epochs: int = 10) -> Dict[str, float]:
        critic_loss = self.update_critic(batch_size) or 0.0
        policy_losses = []
        for agent in self.original_agents.values():
            agent.update_policy(batch_size, epochs)
            if agent.policy_losses:
                policy_losses.append(agent.policy_losses[-1])
        avg_pl = float(np.mean(policy_losses)) if policy_losses else 0.0
        return {"critic_loss": critic_loss, "policy_loss": avg_pl}


# ── Factory ───────────────────────────────────────────────────────────────────

def create_agent_for_node(node_id: str, node_type: str,
                          slice_dictionary: SliceDictionary) -> BaseAgent:
    """Create an RLAgent for any surviving infrastructure node."""
    return RLAgent(node_id=node_id, slice_dictionary=slice_dictionary)


# Keep old name for any lingering imports
create_marl_agent_for_node = create_agent_for_node
