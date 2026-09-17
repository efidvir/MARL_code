"""
PHY/MAC MARL agents for autonomous 6G island-mode operation.

Each surviving O-RU / O-DU agent replaces the missing Near-RT RIC and
Core functions by directly controlling:
  - Tx power (coverage / interference tradeoff)
  - MCS per UE class (bounded OFFSET from auto CQI→MCS link adaptation;
    offset 0 = follow auto-CQI, negative = more robust, positive = more
    aggressive — see Simulator._step_phy_mac)
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

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from .topology import TrafficClass, NodeType
from .traffic import SliceDictionary
from .reward_profile import PROFILE as _RP
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
    # ── Fragment coordination cue (+2 dims, MARL_POSTCARD_FRAGMENTS) ────
    # Derived ONLY from received postcards (latched up to 30 ticks, like the
    # DCC period x3): how many DISTINCT fragments other than mine are
    # reachable by radio from here (/3), and whether one of them is heard
    # from a node that is not itself bridging.  Because labels come from the
    # relay-INCLUSIVE graph, a fragment already joined to mine carries MY
    # label -- so "foreign > 0" means exactly "a reachable fragment is still
    # disjoint from mine": the coordination variable the bridging decision
    # needs, that the earlier locality masking removed without replacement.
    foreign_fragments_norm:       float = 0.0
    foreign_unbridged:            float = 0.0


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
    peer_avg_reward:              float = 0.0   # avg reward from learning postcards
    peer_best_relay_hint:         float = 0.0   # peer recommends ME as relay
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
    # ── Fragment coordination (MARL_POSTCARD_FRAGMENTS) ──────────────────
    # The sender's connected-component label (relay-inclusive infrastructure
    # graph; stable representative = smallest node id in the component) and
    # whether the sender currently holds a distinct cross-fragment bridge.
    # LOCALITY-HONEST: a node learns its own component by intra-component
    # flooding (that is what "same component" means), and telling neighbours
    # is one label per postcard.  What was masked earlier -- the global UE
    # census, island_size over ALL nodes, the global routed fraction --
    # needed knowledge of OTHER components; this does not.
    fragment_id:            object       = -1
    holds_bridge:           bool         = False
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

# ── Fragment-coordination postcards (MARL_POSTCARD_FRAGMENTS) ─────────────
# When on, postcards carry the sender's fragment label + bridge-holder flag
# and the observation gains two dims (61: foreign_fragments_norm,
# 62: foreign_unbridged) APPENDED after Block D, so every existing column
# index -- including the locality mask (28-59) and relay_capable (50) --
# is unchanged.  Default OFF: OBS_DIM stays 60 and every shipped checkpoint,
# result and arm is bit-identical.  A 60-dim checkpoint evaluated under 62
# loads via load_policy_state_dict (fc1 zero-padded) and behaves exactly as
# trained; using the cue requires a retrain.
POSTCARD_FRAGMENTS = (os.environ.get('MARL_POSTCARD_FRAGMENTS', '0')
                      .strip().lower() in ('1', 'true', 'on', 'yes'))
OBS_DIM = 62 if POSTCARD_FRAGMENTS else 60
# Layout (60 base dims; +2 appended when POSTCARD_FRAGMENTS):
#   Block A — Radio unit status       12 dims  (PHYMACState.to_block_a)
#   Block B — Neighbour radio (ICIC)  16 dims  (+4 vs. original 12)
#   Block C — Connectivity (IOPS)     23 dims  (+2+8 Multi-eNB IOPS, +1 relay
#                                               capability, vs. original 12)
#   Block D — Temporal + Coordinator   9 dims  (4 original + 5 global policy)
#
# The 60th dim (Block C `relay_capable`) was added because the actor is
# parameter-shared: without an observable "I own a steerable MultiHaul radio"
# feature, no policy can select relay_mode=CAPACITY_BOOST at the 3 sites where
# it bridges fragments and avoid it at the ~39 where it only costs PRB.
# Checkpoints trained at OBS_DIM=59 still LOAD (load_policy_state_dict()
# zero-pads the new fc1 input column) but they cannot USE the new feature —
# a retrain is required for it to carry any weight.
#
# NOTE (MCS-offset actions): all other dims are in use — there is NO reserved/
# zero dim free to carry the auto-CQI MCS index explicitly.  That is
# acceptable because the auto-CQI index is a deterministic piecewise-constant
# function of sinr_average, which IS observable (Block A dim 3, normalised
# SINR; Block A dim 2 additionally exposes the applied spectral efficiency),
# so the policy has enough link-quality signal to choose its MCS offset.

# Action head sizes
N_TX_STEPS    = len(TX_POWER_STEPS_DB)           # 5
# MCS heads keep cardinality len(MCSLevel) (checkpoint-shape compatible) but
# encode OFFSETS from the auto-CQI MCS: idx - MCS_OFFSET_CENTER ∈ {-2..+2}.
N_MCS         = len(MCSLevel)                    # 5
N_RELAY_MODES = len(RelayMode)                   # 5 (OFF, NORMAL, LOCAL_REROUTE, CAPACITY_BOOST, D2D)
N_HANDOVER    = 2
N_SCHEDULERS  = len(MACScheduler)                # 4
N_POSTCARD    = 2

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

    Inputs:  OBS_DIM (60) observation vector (12 + 16 + 23 + 9)
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


def load_policy_state_dict(net: nn.Module, state_dict: dict,
                           strict: bool = True) -> bool:
    """Load a state dict into a PolicyNetwork/CriticNetwork, tolerating a
    change in the INPUT dimension of the first layer (`fc1`).

    OBS_DIM grew from 59 to 60 when the `relay_capable` observation dim was
    added (see the OBS_DIM layout note).  Every other parameter in the
    network is unaffected, so an older checkpoint is still a perfectly good
    warm start: the missing input column is filled with ZEROS, which makes
    the loaded policy compute EXACTLY what it computed before — the new
    feature simply contributes nothing until it is trained.

    Returns True if the checkpoint had to be adapted (i.e. its obs dim did
    not match), False if it loaded unchanged.
    """
    w_key = 'fc1.weight'
    adapted = False
    if w_key in state_dict and hasattr(net, 'fc1'):
        ckpt_w = state_dict[w_key]
        want_in = net.fc1.in_features
        have_in = ckpt_w.shape[1]
        if have_in != want_in:
            state_dict = dict(state_dict)
            if have_in < want_in:
                pad = torch.zeros(ckpt_w.shape[0], want_in - have_in,
                                  dtype=ckpt_w.dtype, device=ckpt_w.device)
                state_dict[w_key] = torch.cat([ckpt_w, pad], dim=1)
            else:
                state_dict[w_key] = ckpt_w[:, :want_in]
            adapted = True
    net.load_state_dict(state_dict, strict=strict)
    return adapted


class CriticNetwork(nn.Module):
    """
    Centralised critic: takes the mean-field global observation (the MEAN of
    all agents' per-agent observation vectors, OBS_DIM dims — NOT a
    concatenated joint state) and returns V(s).
    """

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

# NOTE: 'iops' drives real admit/deny decisions in simulation.py and is
# trained like every other head (it was previously missing here).
#
# DEAD HEADS REMOVED — 'handover' and 'scheduler'.  Both are still SAMPLED by
# the actor and written into PHYMACState (Simulator._step_phy_mac writes
# ps.mac_scheduler and ps.handover_triggered), but a repository-wide search
# shows NOTHING EVER READS THEM:
#   * ps.mac_scheduler        — written at simulation.py:4985, never read.
#     SCHEDULER_EMERGENCY_WEIGHT in phy_mac_state.py is defined and never
#     used, so the scheduler choice has no effect on any capacity, queue or
#     admission calculation.
#   * ps.handover_triggered   — written at simulation.py:4986, never read.
# (ps.d2d_active_pairs, written at simulation.py:5325, is likewise never
# read, but it is a derived OBSERVATION field rather than an action head, so
# there is no head to remove for it.)
#
# Keeping them in the trained set cost real exploration: their log-probs
# entered the PPO ratio as pure noise and the entropy bonus spent budget
# keeping two heads that cannot change the environment near-uniform.
#
# CHECKPOINT COMPATIBILITY: the PolicyNetwork still DEFINES handover_head and
# scheduler_head (see PolicyNetwork.__init__), so every existing checkpoint
# still loads with unchanged shapes and the action dataclass is unchanged.
# What changes is only which heads the PPO loss, the entropy bonus and the
# log-prob ratio cover.  A policy trained before this change carries
# handover/scheduler logits that are now simply never trained and never
# matter.  Wire them to real effects and re-add them here if the MAC
# scheduler is ever implemented.
DISCRETE_HEADS = ["tx_power", "mcs_emrg", "mcs_gen",
                  "relay", "postcard", "iops"]
# Heads the network emits but the trainer deliberately does not train.
UNTRAINED_HEADS = ["handover", "scheduler"]
MCS_LEVELS     = list(MCSLevel)
RELAY_MODES    = list(RelayMode)
SCHEDULERS     = list(MACScheduler)


# ── OBSERVATION LOCALITY ─────────────────────────────────────────────────────
# Which observation columns a partitioned site could actually produce.
#
# Layout (see observation_to_tensor): A 0-11, B 12-27, C 28-50, D 51-59.
#
# Block A is own radio state.  Block B is neighbour state carried by the
# rate-limited postcards.  Block C is where the simulator hands the actor
# whole-topology structures, and the last five of Block D are a policy vector
# from a central coordinator that by assumption may be unreachable.
#
# These are ZEROED for the actor and left intact for the critic.  Zeroed, not
# deleted: load_policy_state_dict pads/truncates only TRAILING columns, so
# dropping an interior dimension would misalign every later column of an
# existing checkpoint.
NONLOCAL_OBS_COLUMNS = {
    28: 'reachable_ue_fraction',      # network-wide UE census
    29: 'isolated_ue_count_norm',     # network-wide UE census
    30: 'active_relay_paths_norm',    # fleet-wide count of live relay links
    31: 'ue_to_ue_routed_norm',       # GLOBAL FLOW SUCCESS RATE -- the paper's
                                      # own headline KPI, fed back as an input
    32: 'intra_island_reachability',  # component size over the full infra graph
    33: 'core_distance_norm',         # multi-source BFS from all live cores
    34: 'bridge_node_flag',           # nx.bridges over the whole graph
    35: 'island_size_norm',           # surviving infra / all nodes
    40: 'registration_capacity_norm', # single simulator-wide IOPSManager
    41: 'pending_iops_norm',          # single simulator-wide IOPSManager
    42: 'island_member_count_norm',   # island membership = connected_components
    43: 'island_ue_load_balance',     #   "
    44: 'xn_mesh_density',            #   "
    45: 'local_epc_health',           #   "
    46: 'nenb_count_norm',            #   "
    47: 'peer_avg_reward',            # island-scoped learning exchanger
    48: 'peer_best_relay_hint',       #   "
    49: 'multi_island_bridge',        #   "
    55: 'coordinator_policy_0',       # the coordinator may be unreachable,
    56: 'coordinator_policy_1',       # which is the paper's own premise
    57: 'coordinator_policy_2',
    58: 'coordinator_policy_3',
    59: 'coordinator_policy_4',
}
# Kept in Block C because they are genuinely local: 36 potential_iab_capacity,
# 37 handover_candidate_count (this node's live neighbours), 38 ue_rsrp_min
# (own SINR), 39 ue_pair_demand (flows anchored on this node), 50 relay_capable
# (static site inventory).

OBS_LOCALITY = os.environ.get('MARL_OBS_LOCALITY', 'local').strip().lower()
if OBS_LOCALITY not in ('local', 'global'):
    raise ValueError(
        f"MARL_OBS_LOCALITY={OBS_LOCALITY!r} is not one of 'local', 'global'")

# ── Capability-conditioned relay-head mask (MARL_RELAY_MASK) ──────────────
# THE PROBLEM THIS SOLVES.  The actor is parameter-shared across ~42 sites but
# only ~22 own a steerable MultiHaul radio.  At a NON-steerable site the
# engine treats CAPACITY_BOOST as exactly LOCAL_REROUTE (the Priority-1
# bridging branch in _step_phy_mac requires has_multihaul; the fall-through
# "repurpose an existing hop" branch then ignores the distinction), so two of
# the five relay actions are behavioural ALIASES there.  The shared softmax
# still spreads probability over both and, worse, the gradient that should
# teach "CAPACITY_BOOST bridges islands" is averaged with experience from
# sites where the same action index does no such thing — the documented
# credit-dilution that makes the policy under-bridge relative to the
# deterministic XL-DET baseline.
#
# THE FIX.  Mask the CAPACITY_BOOST logit to -1e9 wherever the site's own
# relay_capable observation (column 50, a LOCAL, static inventory bit) is 0.
# This removes a redundant duplicate action — not a capability: nothing a
# non-steerable site could actually DO is taken away, because the engine
# already maps that action onto LOCAL_REROUTE, which remains available.
#
# CORRECTNESS.  The mask is a pure function of the stored observation, and
# every consumer of the relay logits (live sampling, collection-time old
# log-prob, update-time new log-prob, and the legacy paths) applies it through
# THIS one helper, so pi_old and pi_new always describe the same masked
# distribution and the PPO ratio stays a genuine density ratio.  A stored
# action can never be a masked action, so log-probs are always finite.
#
# Env-gated, default OFF: existing checkpoints, the shipped results and every
# other arm are bit-identical unless MARL_RELAY_MASK=capability is set.
RELAY_MASK = os.environ.get('MARL_RELAY_MASK', 'off').strip().lower()
if RELAY_MASK not in ('off', 'capability'):
    raise ValueError(
        f"MARL_RELAY_MASK={RELAY_MASK!r} is not one of 'off', 'capability'")

OBS_COL_RELAY_CAPABLE = 50            # Block C 'relay_capable' (local, static)
RELAY_CB_IDX = list(RelayMode).index(RelayMode.CAPACITY_BOOST)   # == 3
_RELAY_MASK_NEG = -1e9


def apply_relay_mask(logits: Dict[str, torch.Tensor],
                     obs_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Mask the CAPACITY_BOOST relay logit at sites that cannot steer.

    logits     dict of head -> (N, C) tensors (any head set containing 'relay')
    obs_batch  (N, OBS_DIM) — the SAME observations the logits came from

    Returns the dict with logits['relay'] replaced by a masked copy (the
    input tensors are never modified in place).  No-op unless
    MARL_RELAY_MASK=capability.
    """
    if RELAY_MASK != 'capability' or 'relay' not in logits:
        return logits
    cannot = (obs_batch[..., OBS_COL_RELAY_CAPABLE] < 0.5)      # (N,)
    if not bool(cannot.any()):
        return logits
    relay = logits['relay'].clone()
    relay[..., RELAY_CB_IDX] = (relay[..., RELAY_CB_IDX]
                                + cannot.to(relay.dtype) * _RELAY_MASK_NEG)
    out = dict(logits)
    out['relay'] = relay
    return out


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

    def observation_to_tensor(self, obs: AgentObservation,
                              mask_nonlocal: Optional[bool] = None
                              ) -> torch.Tensor:
        """Convert AgentObservation to an OBS_DIM-dim float tensor.

        mask_nonlocal
            None  (default) follow MARL_OBS_LOCALITY -- this is the ACTOR path
            False keep every column -- this is the CRITIC path, where global
                  state is legitimate under centralised training
        """
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
            # Relay capability — the dim that makes CAPACITY_BOOST learnable
            cs.relay_capable,
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
        if POSTCARD_FRAGMENTS:
            # Block E — fragment coordination cue (2 dims, appended)
            features = features + [
                float(getattr(nb, 'foreign_fragments_norm', 0.0) or 0.0),
                float(getattr(nb, 'foreign_unbridged', 0.0) or 0.0),
            ]
        features = (features + [0.0] * OBS_DIM)[:OBS_DIM]

        if mask_nonlocal is None:
            mask_nonlocal = (OBS_LOCALITY == 'local')
        if mask_nonlocal:
            # zero the columns a partitioned site could not have produced
            for _c in NONLOCAL_OBS_COLUMNS:
                features[_c] = 0.0

        return torch.tensor(features, dtype=torch.float32).to(getattr(self, 'device', torch.device('cpu')))

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
        logits = apply_relay_mask(self.policy_net(obs_batch), obs_batch)
        lp     = torch.zeros(obs_batch.size(0), device=obs_batch.device)
        H      = torch.tensor(0.0, device=obs_batch.device)

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
        relay_idx   = pick("relay")
        ho_idx      = pick("handover")
        sched_idx   = pick("scheduler")
        postcard_do = bool(pick("postcard"))
        iops_idx    = pick("iops")   # 0=DENY, 1=ADMIT_EMERGENCY, 2=ADMIT_ALL

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
            prb_sampled=prb_sampled,
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
            prb_sampled=prb_sampled,
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

    # ── Local (difference-reward) constants, in the SAME "points" units as
    #    Simulator.compute_global_connectivity_reward ────────────────────────
    #
    # Every constant below is the PER-NODE attribution of a term the global
    # reward already pays fleet-wide, at the SAME rate.  Nothing new is being
    # bought; the same reward is simply credited to the agent that caused it.
    # PRIMARY OBJECTIVE — delivered volume.  Priced at EXACTLY the global
    # rates, against the same denominator (the tick's offered volume), so a
    # byte is worth the same to the agent as it is to the fleet.  See
    # Simulator._forward_traffic's ATTRIBUTION RULE for how a node's share is
    # computed and why it is a "carried volume" credit rather than a
    # partition of the delivered volume.
    DELIVERY_POINTS        = _RP.delivery_points      # 100.0
    GEN_DELIVERY_POINTS    = _RP.gen_delivery_points  # 10.0

    # RELAY / BRIDGE — REPRICED (see the REPRICING note in calculate_reward).
    # These were paying more than delivery at the margin, which made random
    # relaying reward-positive while it was KPI-negative.
    RELAY_CONTRIB_POINTS   = _RP.relay_contrib_points  # 0.0 (was 2.0)
    RELAY_TPUT_PER_MBPS    = 1.0 / 50.0   # == global "1 pt per 50 Mbps"
    RELAY_TPUT_CAP         = _RP.relay_tput_cap        # 3.0 (was 8.0)
    BRIDGE_HELD_POINTS     = _RP.bridge_held_points    # 6.0 (was 10.0)
    BRIDGE_CF_POINTS       = _RP.bridge_cf_points      # 0.0; 25.0 in 'mission'
    RELAY_CHURN_POINTS     = 1.0    # == global RELAY_CHURN_PENALTY
    MCS_OVERREACH_POINTS   = 1.0    # own MCS above own SINR -> OLLA step-down
    ENERGY_POINTS          = 1.0    # own Tx-power consumption cost
    QOS_POINTS             = 2.0    # own life-safety delivery, when demanded

    def calculate_reward(self, obs: AgentObservation,
                         action: PHYMACAction,
                         next_obs: AgentObservation) -> RewardComponents:
        """
        PER-AGENT DIFFERENCE REWARD (credit assignment).

        WHAT THIS REPLACED, AND WHY.  The previous reward was
            R = 0.40 r_conn + 0.15 r_cov + 0.10 r_qos + 0.15 r_iops
                + 0.10 r_energy
        and it was described as the agent's "local" reward, mixed 0.4 local /
        0.6 global in the training drivers.  Instrumented on the Scenario-A
        comparison environment (seed 42, 200 island ticks, 35 agents) the
        CROSS-AGENT standard deviation of each term at a tick was:

            term        weight   cross-AGENT sd   cross-TICK sd
            r_conn        0.40         0.000000        0.093678
            r_cov         0.15         0.000000        0.000000
            r_qos         0.10         0.000000        0.000000
            r_iops        0.15         0.000000        0.000000
            r_energy      0.10         0.106296        0.020569

        i.e. FOUR OF THE FIVE TERMS WERE IDENTICAL FOR EVERY AGENT, so they
        carried exactly zero credit-assignment information no matter what
        weight they were given, and three of them (r_cov, r_qos, r_iops) were
        also constant over time — pure reward offsets:
          * r_conn = ConnectivityState.ue_to_ue_routed_norm, which
            _compute_connectivity_state fills from the GLOBAL scalar
            Simulator._ue_pair_routed_fraction (simulation.py) — the same
            number for every node;
          * r_cov  = ConnectivityState.reachable_ue_fraction, likewise the
            global Simulator._reachable_ue_fraction;
          * r_qos  = the node's LIFE_SAFETY admission_success_rate, which
            defaults to 1.0 when the slice has no offered load — and 6125 of
            6125 sampled slices had offered_load == 0, pinning the term at a
            constant 1.0;
          * r_iops = island Xn/EPC/NeNB statistics, which resolve to the
            no-island defaults on this topology (constant 0.4).
        The ONLY term with any per-agent differentiation was the Tx-power
        ENERGY COST, so the sole credit-assigned gradient every agent
        received was "turn your transmitter down".

        Consequence, measured end to end on the same run: the per-agent
        reward had a cross-agent sd of 0.0052 against a cross-tick sd of
        23.19 (ratio 2e-4), 94.8 % of the advantage variance was explained by
        the TICK alone, and the joint R^2 of ALL of an agent's own action
        one-hots on its own advantage was 0.0052.  That is the classic
        shared-reward credit-assignment failure, and it is why the learning
        curve was flat.

        WHAT THIS IS NOW.  A difference-reward surrogate: each agent is paid
        for the part of the SHARED objective that its OWN node produced, at
        the same per-unit rates the global reward already uses (see the
        constants above).  A true COMA counterfactual — re-evaluating the
        global reward with this agent's action replaced by a default — was
        rejected on cost, not on principle: a tick of this simulator costs
        ~0.19 s, so one counterfactual per agent per tick is ~42x, i.e. ~8 s
        per tick, roughly 3500x the cost of a training episode.  The terms
        below are the observable per-node decomposition of the same
        quantities the counterfactual would have priced.

        Terms (all from THIS node, all responsive to THIS node's action):
          + DELIVERY_POINTS x       THIS node's own share of delivered
            delivered_share_u2u     UE-to-UE volume — the local counterpart
            (+ GEN_DELIVERY_POINTS  of the DOMINANT global term.  See the
             x delivered_share_gen) block comment on the term itself and
                                   Simulator._forward_traffic's ATTRIBUTION
                                   RULE.  This is the fix for "the primary
                                   objective has no local counterpart, so it
                                   is cancelled out of the actor's gradient".
          + RELAY_CONTRIB_POINTS   this node is in a relay mode AND its own
                                   transport relay link carried traffic this
                                   tick.  Measured eta^2 of the relay head on
                                   relay_link_active is 0.86 — this is the
                                   strongest action-responsive local signal
                                   available.
          + relay throughput       traffic on THIS node's own relay links, at
                                   the global reward's 1 pt / 50 Mbps rate,
                                   capped at RELAY_TPUT_CAP.
          + BRIDGE_HELD_POINTS     one of THIS node's own relay links is a
                                   member of the DE-DUPLICATED bridge set
                                   (Simulator.distinct_bridge_link_ids), i.e.
                                   it performs an independent fragment merge.
          - RELAY_CHURN_POINTS     THIS agent flipped its own relay mode.
          - MCS_OVERREACH_POINTS   THIS node's applied MCS exceeded what its
                                   own SINR supports, forcing an OLLA
                                   step-down (own mcs_fallback_events delta).
          - energy cost            THIS agent's Tx-power step (unchanged in
                                   spirit from the old r_energy, now in
                                   points).
          + QOS_POINTS * rate      THIS node's life-safety delivered/offered,
                                   ONLY when this node actually has
                                   life-safety demand.

        REPRICING — why the relay/bridge rates came DOWN.
        Measured with the per-head isolation harness (free exactly one action
        head from a do-nothing policy; Simulator.pinned_action_heads,
        train_on_comparison.run_eval_episode(free_heads=...)), on Scenario A
        seed 42, 400 ticks: freeing ONLY the relay head moved the global
        reward UP while it moved the KPI DOWN — reward +6.94 for -7.0 pp of
        UE connectivity.  Random relaying being reward-positive and
        KPI-negative is a reward-hacking surface, and it existed because the
        relay block out-earned delivery at the margin.  The rates were
        therefore CUT (no new bonuses were added, and no metric changed):

            RELAY_CONTRIB_POINTS   2.0 -> 0.0   (removed, see below)
            RELAY_TPUT_CAP         8.0 -> 3.0
            BRIDGE_HELD_POINTS    10.0 -> 6.0   (tracks REUNIFY_PER_FRAGMENT)

        RELAY_CONTRIB_POINTS went all the way to ZERO.  Scoring the old and
        the new rates on an IDENTICAL trajectory (same torch seed, same
        relays formed, so the draw cannot confound it) the old pricing paid
        +19.24 for freeing the relay head and the intermediate 0.5 rate still
        paid +0.19 — both while UE connectivity fell 7.1 pp.  A FLAT per-node
        payment is the one relay term not proportional to what the relaying
        achieved: it pays a relay carrying 1 Mbps exactly what it pays one
        carrying 500.  Relaying is now paid ONLY in proportion to delivered
        UE traffic carried (RELAY_TPUT_*) or to a genuine fragment
        elimination (BRIDGE_HELD_POINTS), which is what "strictly conditional
        on carried traffic or a real fragment reduction" has to mean.

        and, on the global side, the "+0.5 per routed flow whenever ANY
        relay contributed" term was removed outright and both relay terms
        were made strictly conditional on DELIVERED UE traffic rather than on
        raw link utilisation (which also contained the node's own telemetry).
        See Simulator.compute_global_connectivity_reward for that half.

        NO REWARD HACKING — the audit, term by term:
          * the delivery term pays for volume that was actually DELIVERED
            through this node — offered-but-dropped traffic pays nothing, and
            the denominator is the offered volume, so a node cannot raise its
            own share by suppressing anyone else's demand;
          * relay contribution and throughput are gated on DELIVERED UE
            traffic actually carried on this node's own links, so they cannot
            be farmed by switching relay on and idling, nor by relaying the
            node's own O&M telemetry;
          * the bridge term uses the de-duplicated set, so piling parallel or
            cyclic relay links onto the same cut pays nothing extra;
          * churn, MCS over-reach and energy are pure COSTS;
          * the QoS term is demand-conditioned and is OMITTED (not defaulted
            to 1.0) when the node has no life-safety offered load, so it
            gives no gradient toward hoarding emergency PRB, and when demand
            exists it pays for DELIVERY, not for reservation;
          * nothing pays for relay MODE or relay CAPACITY per se.

        ARM-NEUTRAL AND PHYSICALLY HONEST.  This changes only the MARL
        TRAINING signal.  No physics, no capacity, no admission rule and no
        arm's behaviour changes; the SDN/OSPF/OLSR baselines and the
        random-action control are bit-identical, and the evaluation metric
        (achievability in run_timeline_comparison.py) is untouched.  The two
        PHYMACState fields this reads — relay_traffic_carried_mbps and
        relay_link_is_bridge — are derived by _update_phy_mac_observations
        from link utilisations and the bridge set the simulator already
        computes.

        UNITS.  total_reward is now in the SAME POINTS as
        compute_global_connectivity_reward (typical per-node range about
        [-3, +22]) instead of the old [-1.5, +2].  This is what makes the
        nominal "0.4 local / 0.6 global" mix in the training drivers mean
        what it says: on the old scale the local term was ~1.5 % of the mixed
        reward despite its nominal 40 % weight.
        NOTE: the reward changed => a RETRAIN is required.
        """
        ps = next_obs.phy_mac

        # ── OWN DELIVERED VOLUME — the primary objective, made local ───────
        # The dominant global term is +100 x volume-weighted delivered
        # fraction and it had NO local counterpart, so it was cancelled out
        # of the actor's gradient entirely: the counterfactual baseline
        # subtracts the cross-agent MEAN advantage exactly, and a term with
        # the same value for every agent survives that subtraction as zero.
        # The policy was therefore training on relay/energy/bridge terms and
        # NOT on delivered volume — the metric the evaluation scores.
        #
        # delivered_share_* is THIS node's carried delivered volume divided
        # by the tick's OFFERED volume, i.e. the same units and the same
        # denominator as the global delivered fraction, so the two views
        # price a byte identically (see Simulator._forward_traffic).  It is
        # responsive to this agent's own action through every knob it has:
        # an admission HOLD zeroes every flow crossing this node, THROTTLE
        # halves them, the MCS heads set the spectral efficiency that scales
        # this node's radio-link capacity, the PRB split sets how much of
        # that capacity serves access vs relay, and the relay head decides
        # whether a bridged path through this node exists at all.
        r_delivery = (
            self.DELIVERY_POINTS
            * float(getattr(ps, 'delivered_share_ue_to_ue', 0.0) or 0.0)
            + self.GEN_DELIVERY_POINTS
            * float(getattr(ps, 'delivered_share_general', 0.0) or 0.0))

        # ── Own relay contribution (local part of the global relay terms) ──
        carried = float(getattr(ps, 'relay_traffic_carried_mbps', 0.0) or 0.0)
        in_relay_mode = ps.relay_mode in (RelayMode.LOCAL_REROUTE,
                                          RelayMode.CAPACITY_BOOST)
        r_relay = 0.0
        if in_relay_mode and carried > 0.0:
            r_relay += self.RELAY_CONTRIB_POINTS
            r_relay += min(self.RELAY_TPUT_CAP,
                           carried * self.RELAY_TPUT_PER_MBPS)

        # ── Own bridge held (local part of the reunification term) ─────────
        r_bridge = (self.BRIDGE_HELD_POINTS
                    if getattr(ps, 'relay_link_is_bridge', False) else 0.0)
        # Per-site counterfactual (mission profile): fragments MY bridges
        # eliminate, i.e. components-without-my-links minus components-with.
        # Local to this agent, so it survives the cross-agent centring.
        r_bridge += (self.BRIDGE_CF_POINTS
                     * float(getattr(ps, 'bridge_counterfactual_fragments', 0) or 0))

        # ── Own relay churn (local part of RELAY_CHURN_PENALTY) ────────────
        # Compared against THIS agent's own previous relay action, not
        # against obs.phy_mac: `obs` and `next_obs` share one live
        # PHYMACState object, so its relay_mode field is identical in both
        # and could never show a change.
        r_churn = 0.0
        prev_relay_idx = getattr(self, '_prev_relay_action_idx', None)
        if prev_relay_idx is not None and prev_relay_idx != action.relay_mode_idx:
            r_churn = -self.RELAY_CHURN_POINTS
        self._prev_relay_action_idx = action.relay_mode_idx

        # ── Own MCS over-reach (OLLA step-down forced by own MCS offset) ───
        fb_now = int(getattr(ps, 'mcs_fallback_events', 0))
        fb_prev = getattr(self, '_prev_mcs_fallback_events', None)
        r_mcs = 0.0
        if fb_prev is not None and fb_now > fb_prev:
            r_mcs = -self.MCS_OVERREACH_POINTS
        self._prev_mcs_fallback_events = fb_now

        # ── Own energy cost (consumption-proportional; no idle bonus) ──────
        power_delta = TX_POWER_STEPS_DB[action.tx_power_step]
        r_energy = -max(0.0, power_delta / 6.0) * self.ENERGY_POINTS

        # ── Own life-safety delivery, ONLY where there is demand ───────────
        r_qos = 0.0
        if next_obs.local_slices:
            from .topology import TrafficClass as TC
            ls = next_obs.local_slices.get(TC.LIFE_SAFETY)
            if (ls is not None and getattr(ls, 'offered_load', 0.0) > 0.0
                    and hasattr(ls, 'admission_success_rate')):
                r_qos = self.QOS_POINTS * float(ls.admission_success_rate)

        total = (r_delivery + r_relay + r_bridge + r_churn + r_mcs
                 + r_energy + r_qos)

        return RewardComponents(
            qos_reward=r_qos,
            energy_reward=r_energy,
            coordination_reward=r_relay + r_bridge,
            stability_reward=r_churn + r_mcs,
            total_reward=total,
            connectivity_reward=r_delivery,
            coverage_reward=r_relay,
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
            action_indices["relay"].append(act.relay_mode_idx)
            action_indices["handover"].append(act.handover_idx)
            action_indices["scheduler"].append(act.scheduler_idx)
            action_indices["postcard"].append(int(act.send_postcard))
            action_indices["iops"].append(int(getattr(act, '_iops_decision', 0)))
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
        torch.save(self.policy_net.state_dict(), filepath)

    def load_policy(self, filepath: str):
        try:
            state_dict = torch.load(filepath, weights_only=False)
            load_policy_state_dict(self.policy_net, state_dict)
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
