"""
Non-RT RIC Global Coordinator Agent

Models the A1-policy-broadcasting function of the Non-RT RIC,
which is lost when the core is severed.

Timescale: COORD_INTERVAL ticks (≈ 2 s at 100 ms/tick)
           faster than Non-RT RIC (≥ 1 s) but identical role

Role:
    - Aggregates island-wide state from all postcards received this window
    - Outputs a GlobalPolicyVector (5 floats ∈ [0,1])
    - All local agents observe this vector in Block D of their obs
    - Local agents learn to condition their behaviour on it (e.g., if
      energy_save_mode > 0.5 → deprioritise best-effort PRBs)

Training:
    Trained with a simple REINFORCE update at the coordinator level.
    Reward = island-wide connectivity score + emergency coverage weight.

Deployment:
    Loaded from checkpoint, continues fine-tuning online.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Timescale: coordinator acts every COORD_INTERVAL ticks
COORD_INTERVAL  = 20
COORD_INPUT_DIM = 38   # aggregated island-wide feature vector (30 orig + 8 Multi-eNB IOPS)


# ─── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class GlobalPolicyVector:
    """
    A1-like policy broadcast from the coordinator to all local agents.

    Each value ∈ [0, 1]:
        energy_save_mode           0 = normal, 1 = deep sleep non-critical nodes
        relay_density_target       fraction of nodes that should be in relay mode
        min_emergency_prb_guarantee floor on emergency PRB fraction (island-wide)
        handover_aggressiveness    0 = no HO, 1 = very aggressive load-balancing HO
        power_budget_mode          0 = conservation, 1 = maximum coverage
    """
    energy_save_mode:            float = 0.0
    relay_density_target:        float = 0.30
    min_emergency_prb_guarantee: float = 0.30
    handover_aggressiveness:     float = 0.30
    power_budget_mode:           float = 0.50

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.energy_save_mode,
            self.relay_density_target,
            self.min_emergency_prb_guarantee,
            self.handover_aggressiveness,
            self.power_budget_mode,
        ], dtype=torch.float32)

    def to_list(self) -> List[float]:
        return [
            self.energy_save_mode,
            self.relay_density_target,
            self.min_emergency_prb_guarantee,
            self.handover_aggressiveness,
            self.power_budget_mode,
        ]

    @classmethod
    def default(cls) -> 'GlobalPolicyVector':
        return cls()

    @classmethod
    def neutral(cls) -> 'GlobalPolicyVector':
        """
        Fixed neutral policy that imposes NO active perturbation on agents.

        The only field with an ACTIVE effect is min_emergency_prb_guarantee:
        agents (agent.py / simulation.py) enforce it as a floor on their
        emergency PRB fraction, overriding the sampled PRB split.  Setting it
        to 0.0 disables the floor entirely.  The remaining fields are only
        observed passively in Block D of the agent observation, so any fixed
        constant is neutral there — defaults are kept for those.
        """
        return cls(min_emergency_prb_guarantee=0.0)


# ─── Policy network ────────────────────────────────────────────────────────────

class CoordinatorPolicyNet(nn.Module):
    """
    Small MLP: 38 → 64 → 64 → 5, sigmoid output.
    Output directly maps to GlobalPolicyVector fields.
    """

    def __init__(self, input_dim: int = COORD_INPUT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Coordinator Agent ─────────────────────────────────────────────────────────

class CoordinatorAgent:
    """
    Non-RT RIC Global Coordinator.

    Call `step()` every tick — it will only run inference on multiples
    of `interval` ticks, returning the same GlobalPolicyVector in between.

    Call `record_reward()` after each interval with island-wide reward.
    Call `update()` every few intervals to train.
    """

    def __init__(self, interval: int = COORD_INTERVAL, lr: float = 1e-4):
        self.interval       = interval
        self.policy_net     = CoordinatorPolicyNet()
        self.optimizer      = torch.optim.Adam(
            self.policy_net.parameters(), lr=lr
        )
        # Fixed neutral policy — no PRB floor, no perturbation of agents
        # (see update() for why the learned policy is disabled).
        self.current_policy = GlobalPolicyVector.neutral()
        self._last_state:   Optional[torch.Tensor] = None
        self._experiences:  List[Tuple[torch.Tensor, float]] = []

    # ── State aggregation ──────────────────────────────────────────────────────

    def _build_global_state(self,
                            postcards:            list,
                            phy_mac_states:       dict,
                            iops_manager,
                            ticks_since_severance: int,
                            topology,
                            iops_controller=None) -> torch.Tensor:
        """
        Build 38-dim aggregate feature vector from island-wide telemetry.
        First 30 dims: legacy features.
        Dims 30-37: Multi-eNB IOPS features.
        """
        ps_list = list(phy_mac_states.values())
        N       = max(1, len(ps_list))

        avg_prb_util   = sum(p.prb_utilization        for p in ps_list) / N
        avg_sinr_norm  = sum(                              # normalise [-5,30] → [0,1]
            max(0.0, min(1.0, (getattr(p, 'sinr_average', 15.0) + 5.0) / 35.0))
            for p in ps_list
        ) / N
        avg_prb_emrg   = sum(p.prb_emergency_fraction  for p in ps_list) / N
        avg_prb_relay  = sum(p.prb_relay_fraction       for p in ps_list) / N
        avg_tx_norm    = sum(
            max(0.0, min(1.0, (getattr(p, 'tx_power_dbm', 23.0) - 10.0) / 33.0))
            for p in ps_list
        ) / N
        relay_active_f = sum(
            1 for p in ps_list if p.relay_mode.value != 'off'
        ) / N
        relay_active_f   = sum(1 for p in ps_list if p.relay_link_active) / N

        avg_ue_count   = min(1.0, sum(p.active_ue_count   for p in ps_list) / max(1, N * 20))
        avg_emrg_cnt   = min(1.0, sum(p.emergency_ue_count for p in ps_list) / max(1, N * 5))

        # From topology
        all_nodes = list(topology.nodes.values())
        try:
            from .topology import NodeType
            ue_nodes      = [n for n in all_nodes if n.node_type == NodeType.UE]
            island_ues    = sum(1 for n in ue_nodes if n.is_island and n.is_survivor)
            total_ues     = max(1, len(ue_nodes))
            iso_ue_frac   = island_ues / total_ues
            island_infra  = sum(
                1 for n in all_nodes
                if n.node_type != NodeType.UE and n.is_island and n.is_survivor
            ) / max(1, len(all_nodes))
        except Exception:
            iso_ue_frac  = 0.5
            island_infra = 0.5

        # IOPS state
        iops_cap, iops_pend = 0.0, 0.0
        if iops_manager is not None:
            iops_cap, iops_pend = iops_manager.get_obs_features()

        # Postcard-derived
        n_pc          = max(1, len(postcards))
        relay_in_pc   = sum(
            1 for pc in postcards if getattr(pc, 'relay_mode_active', False)
        ) / n_pc
        pc_sinr_avg   = (
            sum(getattr(pc, 'best_sinr_to_neighbour', 0.5) for pc in postcards) / n_pc
        )

        tick_norm     = min(1.0, ticks_since_severance / 300.0)

        features = [
            tick_norm,                   # 0
            island_infra,                # 1
            iso_ue_frac,                 # 2
            avg_prb_util,                # 3
            avg_sinr_norm,               # 4
            avg_prb_emrg,                # 5
            avg_prb_relay,               # 6
            avg_tx_norm,                 # 7
            relay_active_f,              # 8
            relay_active_f,                # 9
            relay_in_pc,                 # 10
            min(1.0, n_pc / 10.0),       # 11
            pc_sinr_avg,                 # 12
            iops_cap,                    # 13
            iops_pend,                   # 14
            avg_ue_count,                # 15
            avg_emrg_cnt,                # 16
        ]

        # ── Multi-eNB IOPS features (dims 17-24, TS 22.346) ──────────────
        if iops_controller is not None:
            stats = iops_controller.stats()
            num_islands = stats.get('num_islands', 0)
            total_enbs  = stats.get('total_enbs', 0)
            total_nenbs = stats.get('total_nenbs', 0)
            total_ues_reg = stats.get('total_registered_ues', 0)
            # Xn mesh density (avg across islands)
            xn_densities = []
            epc_healths  = []
            mcptt_calls  = 0
            for iid, idata in stats.get('islands', {}).items():
                sz = idata.get('size', 1)
                xn = idata.get('xn_links', 0)
                max_xn = sz * (sz - 1) / 2
                xn_densities.append(xn / max(1, max_xn))
                epc = idata.get('epc', {})
                epc_healths.append(
                    1.0 if epc.get('capacity_pct', 100) < 90 else 0.5)
                mcptt_calls += epc.get('group_calls', 0)

            features.extend([
                min(1.0, num_islands / 5.0),          # 17: island_count_norm
                min(1.0, total_enbs / 80.0),          # 18: avg_island_size_norm
                min(1.0, total_nenbs / 5.0),          # 19: total_nenb_norm
                (sum(xn_densities) / max(1, len(xn_densities))
                 if xn_densities else 0.0),            # 20: xn_mesh_density
                0.0,                                   # 21: multi_island_bridge_count (placeholder)
                0.0,                                   # 22: peer_learning_convergence
                min(1.0, mcptt_calls / 10.0),          # 23: mcptt_active_calls_norm
                (sum(epc_healths) / max(1, len(epc_healths))
                 if epc_healths else 0.0),              # 24: local_epc_health_avg
            ])
        else:
            features.extend([0.0] * 8)

        # Pad to COORD_INPUT_DIM
        features += [0.0] * (COORD_INPUT_DIM - len(features))
        return torch.tensor(features[:COORD_INPUT_DIM], dtype=torch.float32)

    # ── Inference ──────────────────────────────────────────────────────────────

    def step(self,
             postcards:             list,
             phy_mac_states:        dict,
             iops_manager,
             ticks_since_severance: int,
             topology,
             current_tick:          int,
             iops_controller=None) -> GlobalPolicyVector:
        """
        Return the coordinator policy vector for this tick.

        The learned policy is currently DISABLED (see update()): running an
        untrained network here produced arbitrary outputs, and its PRB-floor
        output (min_emergency_prb_guarantee) actively perturbed the agents'
        sampled PRB splits.  Until a proper policy-gradient treatment exists,
        the coordinator emits a fixed neutral vector that imposes no floor
        and no perturbation.
        """
        return self.current_policy

    # ── Learning ───────────────────────────────────────────────────────────────

    def record_reward(self, connectivity_reward: float,
                      coverage_reward:     float,
                      emergency_served:    float = 0.0):
        """Record island-wide reward for coordinator policy update."""
        if self._last_state is not None:
            r = (0.6 * connectivity_reward +
                 0.3 * coverage_reward +
                 0.1 * emergency_served)
            self._experiences.append((self._last_state.clone(), float(r)))

    def update(self, batch_size: int = 32) -> float:
        """
        Coordinator training is DISABLED (intentional no-op).

        The previous "REINFORCE" update here was not a policy-gradient method:
        it minimised MSE(policy outputs, normalized_reward * 0.5 + 0.5),
        broadcasting ONE scalar target to all 5 policy dimensions — every knob
        moved together toward the reward z-score, which is not reinforcement
        learning and (combined with the untrained PRB-floor output) actively
        perturbed the local agents.

        TODO: implement a proper policy-gradient treatment (e.g., treat the
        5-dim vector as a Gaussian/Beta policy, collect (state, action,
        return) tuples over coordinator intervals, and apply REINFORCE/PPO
        with a baseline).  The CoordinatorPolicyNet class is kept for that
        purpose.  Until then the coordinator emits fixed neutral defaults
        (see GlobalPolicyVector.neutral()).
        """
        self._experiences.clear()
        return 0.0

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> bool:
        try:
            sd = torch.load(path, weights_only=False)
            self.policy_net.load_state_dict(sd)
            return True
        except Exception as e:
            print(f"[Coordinator] load failed: {e}")
            return False
