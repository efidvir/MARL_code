"""
PHY/MAC MARL agents for autonomous 6G island-mode operation.

Each surviving O-RU / O-DU agent replaces the missing Near-RT RIC and
Core functions by directly controlling:
  - Tx power (coverage / interference tradeoff)
  - MCS per UE class (reliability vs. throughput)
  - PRB allocation split (emergency / relay / general)
  - Relay mode (OFF / TRANSPORT_RELAY / D2D_PEER_RELAY)
  - MAC scheduler (EMERGENCY_FIRST, PROP_FAIR, ...)
  - Handover trigger (migrate edge UEs to better-connected O-RU)

Algorithm: MAPPO (Multi-Agent PPO) with centralised training,
           decentralised execution (CTDE).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .topology import TrafficClass, NodeType
from .traffic import SliceDictionary
from .phy_mac_state import (
    PHYMACState, RelayMode, MCSLevel, MACScheduler,
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
    cooperative_relay_assigned:   float = 0.0   # 1 if neighbour designated ME as relay


@dataclass
class ConnectivityState:
    """UE and inter-island connectivity state (Block C — 22 dims)."""
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
    peer_avg_reward:              float = 0.0   # avg reward from learning postcards
    peer_best_relay_hint:         float = 0.0   # peer recommends ME as relay
    multi_island_bridge:          float = 0.0   # 1 if bridges two islands


@dataclass
class AgentObservation:
    """Complete PHY/MAC observation for one O-RU or O-DU agent (59 dims)."""
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


# ── Action dataclasses ────────────────────────────────────────────────────────

@dataclass
class ControlPostcard:
    """60-byte DCC postcard carrying PHY/MAC summary to neighbours."""
    sender_id:              str
    relay_mode_active:      bool
    best_sinr_to_neighbour: float    # normalised [0,1]
    isolated_ue_count:      int
    prb_avail_for_relay:    float    # [0,1]
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

    # MCS selection indices  (0=QPSK_1/3 … 4=QAM256)
    mcs_emergency_idx:  int = 1   # default QPSK_1/2
    mcs_general_idx:    int = 2   # default QAM16

    # PRB allocation — three-way softmax; stored as fractions summing to 1
    prb_emergency_frac: float = 0.30
    prb_relay_frac:     float = 0.00
    prb_general_frac:   float = 0.70

    # Relay mode index: 0=OFF, 1=NORMAL, 2=LOCAL_REROUTE, 3=CAPACITY_BOOST, 4=D2D_PEER_RELAY
    relay_mode_idx:     int = 0

    # Handover: 0=no, 1=trigger
    handover_idx:       int = 0

    # MAC scheduler index: 0=RR, 1=PF, 2=EMG_FIRST, 3=MAX_SINR
    scheduler_idx:      int = 1

    # Postcard
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

OBS_DIM = 59
# Layout:
#   Block A — Radio unit status       12 dims  (PHYMACState.to_block_a)
#   Block B — Neighbour radio (ICIC)  16 dims  (+4 vs. original 12)
#   Block C — Connectivity (IOPS)     22 dims  (+2+8 Multi-eNB IOPS vs. original 12)
#   Block D — Temporal + Coordinator   9 dims  (4 original + 5 global policy)

# Action head sizes
N_TX_STEPS    = len(TX_POWER_STEPS_DB)           # 5
N_MCS         = len(MCSLevel)                    # 5
N_RELAY_MODES = len(RelayMode)                   # 5 (OFF, NORMAL, LOCAL_REROUTE, CAPACITY_BOOST, D2D)
N_HANDOVER    = 2
N_SCHEDULERS  = len(MACScheduler)                # 4
N_POSTCARD    = 2


class PolicyNetwork(nn.Module):
    """
    Actor network for PHY/MAC MARL.

    Inputs:  59-dim observation vector (12 + 16 + 22 + 9)
    Outputs: one logit vector per discrete action head +
             one 3-way PRB allocation head (softmax → fractions) +
             one IOPS admit head (3 classes: DENY / ADMIT_EMERGENCY / ADMIT_ALL)
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

        # Discrete heads
        self.tx_power_head   = nn.Linear(hidden, N_TX_STEPS)
        self.mcs_emrg_head   = nn.Linear(hidden, N_MCS)
        self.mcs_gen_head    = nn.Linear(hidden, N_MCS)
        self.relay_head      = nn.Linear(hidden, N_RELAY_MODES)
        self.handover_head   = nn.Linear(hidden, N_HANDOVER)
        self.scheduler_head  = nn.Linear(hidden, N_SCHEDULERS)
        self.postcard_head   = nn.Linear(hidden, N_POSTCARD)
        # IOPS: DENY=0 / ADMIT_EMERGENCY=1 / ADMIT_ALL=2
        self.iops_head       = nn.Linear(hidden, 3)

        # PRB 3-way split (softmax logits)
        self.prb_head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor):
        """Forward pass. x: (batch, OBS_DIM)."""
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))

        return {
            "tx_power":   self.tx_power_head(h),
            "mcs_emrg":   self.mcs_emrg_head(h),
            "mcs_gen":    self.mcs_gen_head(h),
            "relay":      self.relay_head(h),
            "handover":   self.handover_head(h),
            "scheduler":  self.scheduler_head(h),
            "postcard":   self.postcard_head(h),
            "iops":       self.iops_head(h),      # [B, 3]
            "prb":        self.prb_head(h),        # [B, 3] — apply softmax outside
        }


class CriticNetwork(nn.Module):
    """Centralised critic: takes concatenated global state, returns V(s)."""

    def __init__(self, global_obs_dim: int, hidden: int = 256):
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

DISCRETE_HEADS = ["tx_power", "mcs_emrg", "mcs_gen",
                  "relay", "handover", "scheduler", "postcard"]
MCS_LEVELS     = list(MCSLevel)
RELAY_MODES    = list(RelayMode)
SCHEDULERS     = list(MACScheduler)


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

        self.policy_net = PolicyNetwork(OBS_DIM)
        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)

        self.experiences: List[Experience] = []
        self.policy_losses: List[float]    = []
        self.last_action: Optional[PHYMACAction] = None
        self.last_actions: Dict = {}           # legacy; used by convergence analysis
        self.last_postcard_tick = -100

    # ── Observation → tensor ─────────────────────────────────────────────

    def observation_to_tensor(self, obs: AgentObservation) -> torch.Tensor:
        """Convert AgentObservation to a 59-dim float tensor."""
        # Block A — Radio unit status (12 dims from PHYMACState)
        block_a = obs.phy_mac.to_block_a()

        # Block B — Neighbour radio summary (16 dims: 12 original + 4 ICIC)
        nb = obs.neighbor_radio
        block_b = [
            nb.avg_prb_util, nb.avg_sinr, nb.avg_backhaul_avail,
            nb.fraction_island, nb.best_relay_sinr, nb.best_relay_hop_norm,
            nb.neighbor_count_norm, nb.any_overloaded, nb.any_relay_active,
            nb.postcard_received, nb.policy_version_norm, nb.coordination_quality,
            # ICIC dims
            nb.neighbour_avg_tx_power_norm,
            nb.experienced_sinr_drop_norm,
            nb.interferer_count_norm,
            nb.cooperative_relay_assigned,
        ]

        # Block C — Connectivity graph state (22 dims: 12 + 2 IOPS + 8 Multi-eNB IOPS)
        cs = obs.connectivity
        block_c = [
            cs.reachable_ue_fraction, cs.isolated_ue_count_norm,
            cs.active_relay_paths_norm, cs.ue_to_ue_routed_norm,
            cs.intra_island_reachability, cs.core_distance_norm,
            cs.bridge_node_flag, cs.island_size_norm,
            cs.potential_iab_capacity_norm, cs.handover_candidate_count_norm,
            cs.ue_rsrp_min_norm, cs.ue_pair_demand_norm,
            # IOPS dims
            cs.registration_capacity_norm,
            cs.pending_iops_norm,
            # Multi-eNB IOPS dims (TS 22.346)
            cs.island_member_count_norm,
            cs.island_ue_load_balance,
            cs.xn_mesh_density,
            cs.local_epc_health,
            cs.nenb_count_norm,
            cs.peer_avg_reward,
            cs.peer_best_relay_hint,
            cs.multi_island_bridge,
        ]

        # Block D — Temporal + Coordinator (4 + 5 = 9 dims)
        gp = obs.global_policy or [0.0] * 5
        block_d = [
            min(1.0, obs.ticks_since_severance / 200.0),
            1.0 if obs.is_island else 0.0,
            1.0 if (obs.phy_mac.relay_link_active or
                    obs.phy_mac.relay_mode != RelayMode.OFF) else 0.0,
            min(1.0, obs.current_tick / 500.0),
            # Global coordinator policy vector (5 dims)
            float(gp[0]), float(gp[1]), float(gp[2]),
            float(gp[3]), float(gp[4]),
        ]

        features = block_a + block_b + block_c + block_d
        features = (features + [0.0] * OBS_DIM)[:OBS_DIM]
        return torch.tensor(features, dtype=torch.float32)

    def evaluate_actions(self, obs_batch:    torch.Tensor,
                         action_heads:      Dict[str, torch.Tensor],
                         prb_target:        torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For MAPPO update: compute (new_log_probs, entropy) for a batch
        of stored (obs, actions).

        Args:
            obs_batch    (B, OBS_DIM)
            action_heads  {head_name: LongTensor(B)}
            prb_target   (B, 3) — stored PRB fractions (softmax targets)
        Returns:
            new_log_probs  (B,)
            entropy        scalar
        """
        logits = self.policy_net(obs_batch)
        lp     = torch.zeros(obs_batch.size(0))
        H      = torch.tensor(0.0)

        for h in DISCRETE_HEADS:
            if h in logits and h in action_heads:
                dist = Categorical(logits=logits[h])
                lp   = lp + dist.log_prob(action_heads[h])
                H    = H  + dist.entropy().mean()

        if 'prb' in logits:
            log_soft = F.log_softmax(logits['prb'], dim=-1)
            prb_ce   = -(prb_target * log_soft).sum(dim=-1)
            lp       = lp - prb_ce

        return lp, H

    # ── Action sampling ──────────────────────────────────────────────────

    def compute_action(self, observation: AgentObservation) -> PHYMACAction:
        obs_t = self.observation_to_tensor(observation).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy_net(obs_t)

        def pick(head_name: str) -> int:
            l = logits[head_name].squeeze(0)
            if self.is_training:
                return Categorical(logits=l).sample().item()
            return torch.argmax(l).item()

        tx_step     = pick("tx_power")
        mcs_e_idx   = pick("mcs_emrg")
        mcs_g_idx   = pick("mcs_gen")
        relay_idx   = pick("relay")
        ho_idx      = pick("handover")
        sched_idx   = pick("scheduler")
        postcard_do = bool(pick("postcard"))
        iops_idx    = pick("iops")   # 0=DENY, 1=ADMIT_EMERGENCY, 2=ADMIT_ALL

        # PRB split — softmax over 3 logits
        prb_logits = logits["prb"].squeeze(0)
        prb_fracs  = F.softmax(prb_logits, dim=-1).tolist()

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

        # Rate-limit postcards (≥10 ticks apart)
        postcard_content = None
        ps = observation.phy_mac
        tx_norm  = max(0.0, min(1.0, (ps.tx_power_dbm - 10.0) / 33.0))
        sinr_deg = max(0.0, min(1.0, -getattr(ps, 'sinr_delta', 0.0) / 10.0))

        if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 2:
            self.last_postcard_tick = observation.current_tick
            postcard_content = ControlPostcard(
                sender_id=self.node_id,
                relay_mode_active=(relay_idx > 0),
                best_sinr_to_neighbour=min(1.0, (ps.sinr_average + 5) / 35),
                isolated_ue_count=int(observation.connectivity.isolated_ue_count_norm * 100),
                prb_avail_for_relay=prb_fracs[1],
                policy_version=self.policy_version,
                timestamp=observation.current_tick,
                most_needy_class=TrafficClass.LIFE_SAFETY,
                need_level=StrainLevel.NEAR_LIMIT if observation.is_island else StrainLevel.OKAY,
                tx_power_dbm_norm=tx_norm,
                sinr_degradation_norm=sinr_deg,
            )
        else:
            postcard_do = False

        action = PHYMACAction(
            tx_power_step=tx_step,
            mcs_emergency_idx=mcs_e_idx,
            mcs_general_idx=mcs_g_idx,
            prb_emergency_frac=prb_fracs[0],
            prb_relay_frac=prb_fracs[1],
            prb_general_frac=prb_fracs[2],
            relay_mode_idx=relay_idx,
            handover_idx=ho_idx,
            scheduler_idx=sched_idx,
            send_postcard=postcard_do,
            postcard_content=postcard_content,
        )
        # Attach IOPS decision to action as extra attribute
        action._iops_decision = iops_idx
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
        relay_idx   = pick("relay")
        ho_idx      = pick("handover")
        sched_idx   = pick("scheduler")
        postcard_do = bool(pick("postcard"))
        iops_idx    = pick("iops")

        prb_fracs = F.softmax(logits["prb"][row], dim=-1).tolist()

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

        # Postcard rate-limiting
        postcard_content = None
        ps = observation.phy_mac
        tx_norm  = max(0.0, min(1.0, (ps.tx_power_dbm - 10.0) / 33.0))
        sinr_deg = max(0.0, min(1.0, -getattr(ps, 'sinr_delta', 0.0) / 10.0))

        if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 2:
            self.last_postcard_tick = observation.current_tick
            postcard_content = ControlPostcard(
                sender_id=self.node_id,
                relay_mode_active=(relay_idx > 0),
                best_sinr_to_neighbour=min(1.0, (ps.sinr_average + 5) / 35),
                isolated_ue_count=int(observation.connectivity.isolated_ue_count_norm * 100),
                prb_avail_for_relay=prb_fracs[1],
                policy_version=self.policy_version,
                timestamp=observation.current_tick,
                most_needy_class=TrafficClass.LIFE_SAFETY,
                need_level=StrainLevel.NEAR_LIMIT if observation.is_island else StrainLevel.OKAY,
                tx_power_dbm_norm=tx_norm,
                sinr_degradation_norm=sinr_deg,
            )
        else:
            postcard_do = False

        action = PHYMACAction(
            tx_power_step=tx_step,
            mcs_emergency_idx=mcs_e_idx,
            mcs_general_idx=mcs_g_idx,
            prb_emergency_frac=prb_fracs[0],
            prb_relay_frac=prb_fracs[1],
            prb_general_frac=prb_fracs[2],
            relay_mode_idx=relay_idx,
            handover_idx=ho_idx,
            scheduler_idx=sched_idx,
            send_postcard=postcard_do,
            postcard_content=postcard_content,
        )
        action._iops_decision = iops_idx
        self.last_action = action
        return action

    # ── Reward ───────────────────────────────────────────────────────────

    def calculate_reward(self, obs: AgentObservation,
                         action: PHYMACAction,
                         next_obs: AgentObservation) -> RewardComponents:
        """
        Connectivity-centric reward with Multi-eNB IOPS components.

        R = 0.40 * R_connectivity
          + 0.15 * R_coverage
          + 0.10 * R_qos_emergency
          + 0.15 * R_iops       (inter-eNB routing within island)
          + 0.10 * R_energy
          + 0.05 * R_peer_learning
          + 0.05 * R_interference_penalty
        """
        # ── R_connectivity: fraction of UE pairs successfully routed ──────
        r_conn = next_obs.connectivity.ue_to_ue_routed_norm

        # Extra bonus for RESTORING connectivity (was zero, now non-zero)
        if obs.connectivity.ue_to_ue_routed_norm < 0.05 < next_obs.connectivity.ue_to_ue_routed_norm:
            r_conn += 0.5   # restoration bonus

        # ── R_coverage: fraction of UEs reachable ─────────────────────────
        r_cov = next_obs.connectivity.reachable_ue_fraction
        # Penalise remaining isolated UEs
        r_cov -= next_obs.connectivity.isolated_ue_count_norm * 0.5

        # ── R_qos_emergency: emergency UEs served ─────────────────────────
        emrg_frac = next_obs.phy_mac.emergency_ue_count / max(1, next_obs.phy_mac.active_ue_count)
        relay_on  = next_obs.phy_mac.relay_mode != RelayMode.OFF
        r_qos = emrg_frac * (0.5 + (0.5 if relay_on else 0.0))
        if next_obs.local_slices:
            from .topology import TrafficClass as TC
            ls = next_obs.local_slices.get(TC.LIFE_SAFETY)
            if ls and hasattr(ls, 'admission_success_rate'):
                r_qos = ls.admission_success_rate

        # ── R_iops: Multi-eNB IOPS inter-eNB routing success ──────────────
        cs = next_obs.connectivity
        r_iops = 0.0
        if next_obs.is_island:
            # Reward for active Xn mesh (coordination between eNBs)
            r_iops += cs.xn_mesh_density * 0.3
            # Reward for healthy local EPC
            r_iops += cs.local_epc_health * 0.3
            # Reward for balanced UE load across island
            # Optimal balance = 0.5 (norm centered), penalise extremes
            balance_dev = abs(cs.island_ue_load_balance - 0.5)
            r_iops += max(0.0, 0.2 - balance_dev) * 2.0  # 0..0.4
            # Bonus for NeNB deployment in island
            r_iops += cs.nenb_count_norm * 0.1

        # ── R_energy: continuous SoC ──────────────────────────────────────
        r_energy = next_obs.energy_soc * 0.5
        power_delta = TX_POWER_STEPS_DB[action.tx_power_step]
        r_energy -= max(0.0, power_delta / 6.0) * 0.3

        # ── R_peer: reward for cooperative peer learning ───────────────────
        r_peer = cs.peer_avg_reward * 0.5 + cs.peer_best_relay_hint * 0.5

        # ── R_interference: penalise over-powering ─────────────────────────
        r_interf = -max(0.0, next_obs.phy_mac.interference_caused_db / 20.0) * 0.5

        total = (
            r_conn   * 0.40 +
            r_cov    * 0.15 +
            r_qos    * 0.10 +
            r_iops   * 0.15 +
            r_energy * 0.10 +
            r_peer   * 0.05 +
            r_interf * 0.05
        )

        return RewardComponents(
            qos_reward=r_qos,
            energy_reward=r_energy,
            coordination_reward=r_conn,
            stability_reward=r_iops,
            total_reward=total,
            connectivity_reward=r_conn,
            coverage_reward=r_cov,
            interference_penalty=r_interf,
        )

    # ── Policy update (REINFORCE + discounted returns) ────────────────────

    def update_policy(self, batch_size: int = 64, epochs: int = 10,
                      gamma: float = 0.99):
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
            action_indices["relay"].append(act.relay_mode_idx)
            action_indices["handover"].append(act.handover_idx)
            action_indices["scheduler"].append(act.scheduler_idx)
            action_indices["postcard"].append(int(act.send_postcard))
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

            # Enhanced Actor Loss: Unnormalized scale + entropy for exploration
            loss = -(log_probs * returns_t).mean() * 100.0 - 0.5 * entropy.mean()
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
        torch.save(self.policy_net.state_dict(), filepath)

    def load_policy(self, filepath: str):
        try:
            state_dict = torch.load(filepath, weights_only=False)
            self.policy_net.load_state_dict(state_dict)
            self.policy_version += 1000
        except Exception as e:
            print(f"[RLAgent] Failed to load policy: {e}")


# ── Centralised MARL Trainer ──────────────────────────────────────────────────

class CentralizedMARLTrainer:
    """Centralised critic trainer for MAPPO."""

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
            features = []
            for aid in sorted(self.original_agents.keys()):
                agent = self.original_agents[aid]
                if aid in exp.global_observation:
                    features.extend(
                        agent.observation_to_tensor(exp.global_observation[aid]).tolist()
                    )
                else:
                    features.extend([0.0] * agent.obs_dim)
            global_obs_batch.append(features)
            rewards_batch.append(sum(exp.global_reward.values()))

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
