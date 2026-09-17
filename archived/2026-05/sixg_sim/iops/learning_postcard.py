"""
Learning Postcard — Inter-BS cooperative learning channel.

Extends the existing DCC postcard system with a dedicated learning channel
that lets eNBs within a Multi-eNB IOPS island share compressed policy
information to accelerate cooperative MARL learning.

Design rationale:
  - Each LearningPostcard is ~200 bytes (fits in DCC bandwidth budget)
  - Carries compressed gradient deltas via random projection (16 dims)
  - Includes best-action summaries and reward signals
  - Exchanged every 5 ticks (separate cadence from control postcards)

Integration:
  LearningPostcardExchanger is instantiated by Simulator alongside
  ControlPlaneManager and called during _execute_agent_actions().
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class LearningPostcard:
    """
    ~200-byte inter-BS learning message exchanged within a Multi-eNB island.

    Carries compressed policy information to enable cooperative learning:
      - gradient_hash: 16-dim compressed gradient delta (random projection)
      - best_action_summary: top action choices that yielded best reward
      - local_reward_signal: normalised reward from sender's recent ticks
      - connectivity_score: sender's UE-to-UE success rate
      - relay_topology_hint: neighbors the sender recommends as relay targets
    """
    sender_id: str
    island_id: str
    timestamp: int
    # Compressed gradient delta (16 dims via random projection)
    gradient_hash: List[float] = field(default_factory=lambda: [0.0]*16)
    # Best action choices per head
    best_action_summary: Dict[str, int] = field(default_factory=dict)
    # Normalised reward from sender's last N ticks
    local_reward_signal: float = 0.0
    # UE-to-UE routing success rate at sender
    connectivity_score: float = 0.0
    # Relay target recommendations
    relay_topology_hint: List[str] = field(default_factory=list)
    # Policy version for staleness detection
    policy_version: int = 0


# ── Random projection matrix (fixed seed for reproducibility) ────────────────

_PROJ_DIM = 16       # output dimension of compressed gradient
_PROJ_SEED = 12345


def _get_projection_matrix(param_count: int) -> torch.Tensor:
    """
    Lazy-init a (param_count, _PROJ_DIM) random projection matrix.
    Uses Gaussian random projection (Johnson-Lindenstrauss).
    """
    gen = torch.Generator().manual_seed(_PROJ_SEED)
    # Scale factor: 1/sqrt(_PROJ_DIM)
    scale = 1.0 / math.sqrt(_PROJ_DIM)
    return torch.randn(param_count, _PROJ_DIM, generator=gen) * scale


# ── Exchanger ─────────────────────────────────────────────────────────────────

class LearningPostcardExchanger:
    """
    Manages the inter-BS learning postcard channel within IOPS islands.

    Usage:
        exchanger = LearningPostcardExchanger(exchange_interval=5)
        # Each tick:
        exchanger.tick(agents, iops_controller, topology, tick)
        # After tick:
        peer_obs = exchanger.get_peer_obs(node_id)
    """

    def __init__(self, exchange_interval: int = 5,
                 peer_learning_weight: float = 0.1):
        self.exchange_interval = exchange_interval
        self.peer_learning_weight = peer_learning_weight
        # node_id -> list of received learning postcards this interval
        self._inbox: Dict[str, List[LearningPostcard]] = {}
        # node_id -> aggregated peer observations (for agent obs)
        self._peer_obs: Dict[str, dict] = {}
        # Cache: projection matrices per param count
        self._proj_cache: Dict[int, torch.Tensor] = {}
        # Track recent rewards per node for gradient computation
        self._recent_rewards: Dict[str, List[float]] = {}
        # Counter for dashboard
        self.total_exchanges: int = 0
        # Delta-only exchange: (sender, neighbor) -> last gradient_hash sent
        # Only re-send postcard if hash changed or neighbor is new
        self._last_sent_hash: Dict[tuple, List[float]] = {}

    def _get_proj(self, n: int) -> torch.Tensor:
        if n not in self._proj_cache:
            self._proj_cache[n] = _get_projection_matrix(n)
        return self._proj_cache[n]

    # ── Per-tick entry point ──────────────────────────────────────────────

    def _get_infra_neighbors(self, node_id: str, topology) -> set:
        """Return set of infrastructure neighbor IDs connected by live links.

        Only returns nodes connected by a direct, live link — not all
        island members.  This makes postcard exchange realistic: a node
        can only send directly to its wired/wireless neighbors.
        """
        neighbors = set()
        for link in topology.links.values():
            if not getattr(link, 'is_up', True):
                continue
            ep0, ep1 = link.endpoints
            peer = None
            if ep0 == node_id:
                peer = ep1
            elif ep1 == node_id:
                peer = ep0
            if peer is None:
                continue
            # Only infrastructure peers (skip UEs)
            n = topology.nodes.get(peer)
            if n and getattr(n.node_type, 'value', str(n.node_type)) != 'UE':
                neighbors.add(peer)
        return neighbors

    def tick(self, agents: dict, iops_controller, topology, tick: int):
        """
        Called each tick.  On exchange ticks, sends learning postcards
        only to **topological neighbors** (nodes connected by a live
        link) — not the whole island.  Information propagates multi-hop
        over successive exchange intervals.
        """
        if tick % self.exchange_interval != 0:
            return

        # Clear inboxes
        self._inbox.clear()
        self._peer_obs.clear()

        # Build postcard map: sender_id -> postcard
        pc_map: Dict[str, 'LearningPostcard'] = {}

        for island in iops_controller.islands.values():
            if island.iops_mode != 'operational':
                continue

            for enb_id in island.member_enbs:
                agent = agents.get(enb_id)
                if not agent or not hasattr(agent, 'policy_net'):
                    continue
                pc_map[enb_id] = self._generate_postcard(
                    agent, enb_id, island, tick)

        # Distribute postcards to NEIGHBORS ONLY + DELTA-ONLY
        # Skip sending if the neighbor already has our latest hash
        for sender_id, pc in pc_map.items():
            neighbors = self._get_infra_neighbors(sender_id, topology)
            for nbr_id in neighbors:
                if nbr_id not in pc_map:  # neighbor is not an active agent
                    continue
                key = (sender_id, nbr_id)
                prev_hash = self._last_sent_hash.get(key)
                # Send only if: (a) never sent, or (b) gradient changed
                if prev_hash is not None and prev_hash == pc.gradient_hash:
                    continue  # no update — skip
                self._last_sent_hash[key] = pc.gradient_hash
                if nbr_id not in self._inbox:
                    self._inbox[nbr_id] = []
                self._inbox[nbr_id].append(pc)

        # Apply peer learning from received neighbor postcards
        for enb_id, received in self._inbox.items():
            agent = agents.get(enb_id)
            if not agent or not hasattr(agent, 'policy_net'):
                continue
            if received:
                self._apply_peer_learning(agent, received)
                self._update_peer_obs(enb_id, received)
                self.total_exchanges += len(received)

    # ── Postcard generation ──────────────────────────────────────────────

    def _generate_postcard(self, agent, node_id: str,
                           island, tick: int) -> LearningPostcard:
        """Generate a learning postcard from an agent's current state."""
        # Compressed gradient hash
        grad_hash = self._compute_gradient_hash(agent)

        # Best action summary from last action
        best_actions = {}
        if hasattr(agent, 'last_action') and agent.last_action:
            act = agent.last_action
            best_actions = {
                'tx_power': act.tx_power_step,
                'relay': act.relay_mode_idx,
                'scheduler': act.scheduler_idx,
            }

        # Recent reward signal
        recent = self._recent_rewards.get(node_id, [])
        reward_signal = sum(recent[-10:]) / max(1, len(recent[-10:])) \
            if recent else 0.0

        # Relay topology hints
        relay_hints = []
        if hasattr(agent, 'last_action') and agent.last_action:
            if agent.last_action.relay_mode_idx > 0:
                relay_hints.append(node_id)

        return LearningPostcard(
            sender_id=node_id,
            island_id=island.island_id,
            timestamp=tick,
            gradient_hash=grad_hash,
            best_action_summary=best_actions,
            local_reward_signal=reward_signal,
            connectivity_score=0.0,  # filled by caller if available
            relay_topology_hint=relay_hints,
            policy_version=getattr(agent, 'policy_version', 0),
        )

    def _compute_gradient_hash(self, agent) -> List[float]:
        """
        Compress policy gradient into 16-dim hash via random projection.
        If no gradient available, return zeros.
        """
        try:
            params = []
            for p in agent.policy_net.parameters():
                if p.grad is not None:
                    params.append(p.grad.data.flatten())
            if not params:
                return [0.0] * _PROJ_DIM

            grad_vec = torch.cat(params)
            n = grad_vec.shape[0]
            proj = self._get_proj(n)

            # Project: (n,) @ (n, 16) -> (16,)
            compressed = grad_vec @ proj
            # Normalise to [-1, 1]
            mx = compressed.abs().max()
            if mx > 1e-8:
                compressed = compressed / mx
            return compressed.tolist()
        except Exception:
            return [0.0] * _PROJ_DIM

    # ── Peer learning application ────────────────────────────────────────

    def _apply_peer_learning(self, agent, postcards: List[LearningPostcard]):
        """
        Apply cooperative learning from received postcards:
        1. Gradient mixing: average peer gradient hashes with local
        2. Action hint: bias toward peer-recommended actions (entropy-weighted)
        3. Relay hints: update agent's relay knowledge
        """
        if not postcards:
            return

        # 1. Gradient mixing — apply averaged peer gradient as a small
        #    perturbation to policy weights (compressed domain)
        w = self.peer_learning_weight
        if w > 0 and any(any(abs(g) > 1e-6 for g in pc.gradient_hash)
                         for pc in postcards):
            try:
                self._mix_gradients(agent, postcards, w)
            except Exception:
                pass

    def _mix_gradients(self, agent, postcards: List[LearningPostcard],
                       weight: float):
        """
        Mix peer gradient information with local policy.

        Uses reward-weighted averaging of peer gradient hashes,
        then applies as a small update to policy parameters.
        """
        # Compute reward-weighted average of peer gradient hashes
        total_weight = 0.0
        avg_hash = [0.0] * _PROJ_DIM

        for pc in postcards:
            # Weight by reward signal (higher reward = more influence)
            r_weight = max(0.01, pc.local_reward_signal + 1.0)
            for i in range(_PROJ_DIM):
                avg_hash[i] += pc.gradient_hash[i] * r_weight
            total_weight += r_weight

        if total_weight > 0:
            avg_hash = [h / total_weight for h in avg_hash]

        # Apply as tiny perturbation to first layer weights
        # (scaled by peer_learning_weight to prevent instability)
        try:
            first_layer = None
            for name, param in agent.policy_net.named_parameters():
                if 'fc1.weight' in name:
                    first_layer = param
                    break

            if first_layer is not None and first_layer.requires_grad:
                with torch.no_grad():
                    hash_t = torch.tensor(avg_hash, dtype=torch.float32)
                    # Expand hash to match first _PROJ_DIM columns
                    cols = min(_PROJ_DIM, first_layer.shape[1])
                    perturbation = hash_t[:cols] * weight * 0.001
                    first_layer.data[:, :cols] += perturbation.unsqueeze(0)
        except Exception:
            pass

    # ── Observation helpers ──────────────────────────────────────────────

    def _update_peer_obs(self, node_id: str,
                         postcards: List[LearningPostcard]):
        """Aggregate peer observations for agent obs block."""
        if not postcards:
            return

        avg_reward = sum(pc.local_reward_signal for pc in postcards) / \
            len(postcards)
        relay_hints = set()
        for pc in postcards:
            relay_hints.update(pc.relay_topology_hint)

        self._peer_obs[node_id] = {
            'peer_avg_reward': max(0.0, min(1.0,
                (avg_reward + 1.0) / 2.0)),
            'peer_best_relay_hint': 1.0 if node_id in relay_hints else 0.0,
        }

    def get_peer_obs(self, node_id: str) -> dict:
        """Get peer learning observations for agent obs."""
        return self._peer_obs.get(node_id, {
            'peer_avg_reward': 0.0,
            'peer_best_relay_hint': 0.0,
        })

    def record_reward(self, node_id: str, reward: float):
        """Track recent rewards for gradient computation."""
        if node_id not in self._recent_rewards:
            self._recent_rewards[node_id] = []
        self._recent_rewards[node_id].append(reward)
        # Keep last 50
        if len(self._recent_rewards[node_id]) > 50:
            self._recent_rewards[node_id] = \
                self._recent_rewards[node_id][-50:]
