"""
MAPPO Trainer — Multi-Agent Proximal Policy Optimisation
with Generalized Advantage Estimation, EWC and Model Registry.

Algorithm: CTDE (Centralised Training, Decentralised Execution)
  - Each node runs its own Actor π_θ(a | o_i)          (deployed on-device)
  - One shared Critic V_φ(global_state)                 (training only)
  - GAE advantage estimates -> low-variance policy gradient
  - PPO ε-clip -> prevents destructive weight updates
  - Entropy bonus β -> exploration during training
  - EWC regularisation -> prevents catastrophic forgetting during online fine-tuning
"""

import copy
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ─── Hyperparameter dataclass ─────────────────────────────────────────────────

@dataclass
class MAPPOConfig:
    # Core RL
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    clip_eps:      float = 0.20     # PPO epsilon
    entropy_coef:  float = 0.10     # Start higher for early exploration
    entropy_decay: float = 0.985    # MULTIPLICATIVE decay per episode (× 0.985)
    entropy_min:   float = 0.001    # Hard floor — forces exploitation
    value_coef:    float = 1.0      # Balanced with policy loss
    max_grad_norm: float = 0.50
    n_epochs:      int   = 8        # More PPO passes per update for stability
    mini_batch:    int   = 512      # Large batch = fewer steps per epoch = fast

    # Learning rates
    lr_actor:      float = 1e-4
    lr_critic:     float = 3e-4     # Higher critic LR — critic needs to converge faster

    # Shared pool capacity (32k = ~250 ticks x 130 agents)
    pool_capacity:  int  = 32768

    # Deployment / online fine-tuning
    deploy_lr_actor:  float = 3e-5
    deploy_lr_critic: float = 1e-4
    deploy_clip_eps:  float = 0.05
    deploy_entropy:   float = 1e-3
    deploy_n_epochs:  int   = 2

    # EWC (Elastic Weight Consolidation)
    ewc_lambda:    float = 0.40

    # Multi-eNB IOPS cooperative learning (ETSI TS 22.346)
    peer_learning_weight:    float = 0.1    # Weight of peer gradient mixing
    iops_pretraining_episodes: int = 5      # Extra pretraining on IOPS scenarios

    # Checkpointing
    checkpoint_interval: int = 100


# ─── Running reward normaliser ─────────────────────────────────────────────────

class RunningMeanStd:
    """Welford online mean/variance for reward normalisation."""
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var  = 1.0
        self.count = eps

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var  += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var / max(self.count - 1, 1), 1e-8))

    def normalize(self, x: float, clip: float = 5.0) -> float:
        return float(np.clip((x - self.mean) / (self.std + 1e-8), -clip, clip))


# ─── Shared rollout pool (replaces 80 per-agent buffers) ──────────────────────

class SharedRolloutPool:
    """
    O(1) ring-buffer that ALL agents append to (deque-backed).
    Gives 80x more data per update cycle than per-agent buffers.
    Compatible with parameter-sharing MAPPO.
    """

    def __init__(self, capacity: int = 16384):
        self.capacity = capacity
        self.clear()

    def clear(self):
        from collections import deque as _deque
        self.obs       = _deque(maxlen=self.capacity)
        self.actions   = _deque(maxlen=self.capacity)
        self.prb_acts  = _deque(maxlen=self.capacity)
        self.rewards   = _deque(maxlen=self.capacity)
        self.dones     = _deque(maxlen=self.capacity)
        self.values    = _deque(maxlen=self.capacity)
        self.log_probs = _deque(maxlen=self.capacity)

    def add(self, obs: torch.Tensor,
            action_indices: Dict[str, int],
            prb_frac: List[float],
            reward: float, done: bool,
            value: float, log_prob: float):
        # deque(maxlen=capacity) evicts oldest automatically — O(1)
        self.obs.append(obs.detach())
        self.actions.append(dict(action_indices))
        self.prb_acts.append(list(prb_frac))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def __len__(self) -> int:
        return len(self.rewards)


# ─── Legacy per-agent buffer (kept for API compatibility) ─────────────────────

class AgentRolloutBuffer:
    """Thin wrapper that delegates to SharedRolloutPool."""

    def __init__(self, pool: 'SharedRolloutPool'):
        self._pool = pool

    def add(self, obs, action_indices, prb_frac, reward, done, value, log_prob):
        self._pool.add(obs, action_indices, prb_frac, reward, done, value, log_prob)

    def clear(self):
        pass  # Pool manages its own lifecycle

    def __len__(self) -> int:
        return len(self._pool)


# ─── GAE ──────────────────────────────────────────────────────────────────────

def compute_gae(rewards: List[float],
                dones:   List[bool],
                values:  List[float],
                next_value: float,
                gamma: float,
                lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (Schulman 2016).
    A_t = Σ_{k≥0} (γλ)^k δ_{t+k},  δ_t = r_t + γ V_{t+1} - V_t
    Returns:
        advantages  (T,)
        returns     (T,)   = advantages + values
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else next_value
        mask   = 0.0 if dones[t] else 1.0
        delta  = rewards[t] + gamma * v_next * mask - values[t]
        gae    = delta + gamma * lam * mask * gae
        advantages[t] = gae

    adv_t = torch.tensor(advantages, dtype=torch.float32)
    val_t = torch.tensor(values,     dtype=torch.float32)
    return adv_t, adv_t + val_t


# ─── EWC ──────────────────────────────────────────────────────────────────────

class EWCPenalty:
    """
    Elastic Weight Consolidation (Kirkpatrick 2017).

    After simulation training:
        ewc.compute_fisher(policy_net, obs_buffer)   # once
        ewc.anchor()

    During online deployment:
        loss += ewc.ewc_loss(policy_net)             # each update
    """

    def __init__(self, ewc_lambda: float = 0.4):
        self.ewc_lambda   = ewc_lambda
        self.anchor_params: Dict[str, torch.Tensor] = {}
        self.fisher:        Dict[str, torch.Tensor] = {}
        self.is_anchored    = False

    def compute_fisher(self, policy_net: nn.Module,
                       obs_buffer: List[torch.Tensor],
                       n_samples: int = 300,
                       discrete_heads: List[str] = None):
        """
        Diagonal Fisher approximation:
            F_i ≈ E[(∂ log π / ∂ θ_i)²]
        Uses sampled observations from training buffer.
        Call ONCE after simulation training completes.
        """
        policy_net.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in policy_net.named_parameters()
                  if p.requires_grad}

        if len(obs_buffer) == 0:
            return

        heads_to_use = discrete_heads or ["tx_power", "mcs_emrg", "mcs_gen",
                                           "relay", "handover", "scheduler"]
        n = min(n_samples, len(obs_buffer))
        indices = np.random.choice(len(obs_buffer), n, replace=False)

        for idx in indices:
            obs = obs_buffer[int(idx)].unsqueeze(0)
            policy_net.zero_grad()
            logits = policy_net(obs)

            log_prob = torch.tensor(0.0, requires_grad=True)
            for h in heads_to_use:
                if h in logits:
                    lp = F.log_softmax(logits[h], dim=-1).max(dim=-1).values
                    log_prob = log_prob + lp.sum()

            log_prob.backward()
            for name, p in policy_net.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.data.pow(2)

        with torch.no_grad():
            for name in fisher:
                fisher[name] /= n

        self.fisher        = fisher
        self.anchor_params = {n: p.data.clone()
                               for n, p in policy_net.named_parameters()}
        self.is_anchored   = True
        policy_net.train()

    def ewc_loss(self, policy_net: nn.Module) -> torch.Tensor:
        """(λ/2) Σ F_i (θ_i − θ*_i)²"""
        if not self.is_anchored:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for name, p in policy_net.named_parameters():
            if name in self.fisher and name in self.anchor_params:
                loss = loss + (
                    self.fisher[name] * (p - self.anchor_params[name]).pow(2)
                ).sum()
        return (self.ewc_lambda / 2.0) * loss


# ─── Model Registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Versioned checkpoint management.
    Saves policy_<agent_id>_v{N}.pt files.
    Keeps last MAX_KEEP versions to limit disk usage.
    """
    MAX_KEEP = 5

    def __init__(self, checkpoint_dir: str, agent_id: str):
        self.dir        = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.safe_id    = agent_id.replace('/', '_').replace('\\', '_').replace(' ', '_')
        self.version    = 0

    # ── Glob helpers ──────────────────────────────────────────────────────────
    def _ckpt_path(self, version: int) -> Path:
        return self.dir / f"policy_{self.safe_id}_v{version}.pt"

    def _all_versions(self) -> List[int]:
        return sorted(
            int(p.stem.split('_v')[-1])
            for p in self.dir.glob(f"policy_{self.safe_id}_v*.pt")
            if p.stem.split('_v')[-1].isdigit()
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    def save(self, policy_net: nn.Module, metadata: Optional[dict] = None) -> str:
        path = self._ckpt_path(self.version)
        torch.save({
            'state_dict': policy_net.state_dict(),
            'version':    self.version,
            'metadata':   metadata or {},
        }, path)
        self.version += 1
        # Prune old versions
        for old_v in self._all_versions()[:-self.MAX_KEEP]:
            old_p = self._ckpt_path(old_v)
            if old_p.exists():
                old_p.unlink()
        return str(path)

    # ── Load ─────────────────────────────────────────────────────────────────
    def load_latest(self, policy_net: nn.Module) -> int:
        versions = self._all_versions()
        if not versions:
            return -1
        v = versions[-1]
        ckpt = torch.load(self._ckpt_path(v), weights_only=False)
        policy_net.load_state_dict(ckpt['state_dict'])
        self.version = v + 1
        return v

    def load_version(self, policy_net: nn.Module, version: int) -> bool:
        path = self._ckpt_path(version)
        if not path.exists():
            return False
        ckpt = torch.load(path, weights_only=False)
        policy_net.load_state_dict(ckpt['state_dict'])
        return True


# ─── MAPPO Trainer ─────────────────────────────────────────────────────────────

# Which discrete heads the PPO loss covers
DISCRETE_HEADS = ["tx_power", "mcs_emrg", "mcs_gen",
                   "relay", "scheduler", "handover", "postcard"]


class MAPPOTrainer:
    """
    Multi-Agent PPO with centralised-critic, decentralised actors.

    Usage (training):
        trainer = MAPPOTrainer(agents, critic_net, config)
        # each tick:
        value    = trainer.get_value(global_obs_list)
        log_prob = trainer.compute_log_prob(agent_id, obs_t, action_indices, prb)
        trainer.collect(agent_id, obs_t, action_indices, prb, log_prob, value, reward, done)
        # every N ticks:
        metrics  = trainer.update()

    Usage (after training, before deployment):
        trainer.anchor_ewc()           # compute Fisher matrices
        trainer.set_deployment_mode()  # lower LR, activate EWC penalty

    Usage (deployment):
        trainer.collect(...)           # same as training
        every K ticks: trainer.update()
    """

    def __init__(self, agents: dict, critic_net: nn.Module,
                 config: Optional[MAPPOConfig] = None):
        self.config      = config or MAPPOConfig()
        self.critic_net  = critic_net
        self.deploy_mode = False

        # Only RL agents with policy_net
        self.agents: Dict[str, object] = {
            aid: a for aid, a in agents.items()
            if hasattr(a, 'policy_net')
        }

        # ONE shared pool for all agents (parameter-sharing MAPPO)
        self._pool = SharedRolloutPool(capacity=self.config.pool_capacity)

        # Per-agent buffer views (thin wrappers to shared pool for API compat)
        self.buffers: Dict[str, AgentRolloutBuffer] = {
            aid: AgentRolloutBuffer(self._pool) for aid in self.agents
        }

        # Running reward normaliser
        self._reward_rms = RunningMeanStd()

        # EWC penalties — use first agent's net as the shared reference
        self.ewc: Dict[str, EWCPenalty] = {
            aid: EWCPenalty(self.config.ewc_lambda) for aid in self.agents
        }

        # Observation cache (for EWC Fisher computation)
        self.obs_cache: Dict[str, List[torch.Tensor]] = {
            aid: [] for aid in self.agents
        }

        # ONE shared actor optimiser (parameter-sharing: all agents share weights)
        # Use first agent's policy_net as the reference network.
        cfg = self.config
        ref_agents = list(self.agents.values())
        if ref_agents:
            self._ref_actor = ref_agents[0].policy_net
            # Sync all agents to the same net object (true parameter sharing)
            for a in ref_agents[1:]:
                a.policy_net = self._ref_actor
            self._actor_opt = torch.optim.Adam(
                self._ref_actor.parameters(), lr=cfg.lr_actor)
        else:
            self._ref_actor = None
            self._actor_opt = None

        # Legacy per-agent opts dict (for code that references actor_opts[aid])
        self.actor_opts: Dict[str, torch.optim.Optimizer] = {
            aid: self._actor_opt for aid in self.agents
        }

        self.critic_opt = torch.optim.Adam(
            critic_net.parameters(), lr=cfg.lr_critic
        )

        self._total_updates = 0

    # ── Mode control ──────────────────────────────────────────────────────────

    def set_deployment_mode(self):
        """Lower LR, tighter clip, enable EWC.  Call once after anchor_ewc()."""
        self.deploy_mode = True
        cfg = self.config
        for opt in self.actor_opts.values():
            for pg in opt.param_groups:
                pg['lr'] = cfg.deploy_lr_actor
        for pg in self.critic_opt.param_groups:
            pg['lr'] = cfg.deploy_lr_critic
        print("[MAPPO] Deployment mode active — LR×0.1, clip=0.05, EWC enabled")

    def anchor_ewc(self):
        """
        Compute Fisher information matrices and anchor θ*.
        Call ONCE right after simulation training finishes.
        """
        for aid, agent in self.agents.items():
            cache = self.obs_cache[aid]
            if cache:
                self.ewc[aid].compute_fisher(
                    agent.policy_net, cache, n_samples=300,
                    discrete_heads=DISCRETE_HEADS
                )
                print(f"[EWC] Anchored {aid} on {len(cache)} obs samples")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_value(self, global_obs_tensors: List[torch.Tensor]) -> float:
        """Query centralised critic V(s).  Uses mean of per-agent obs tensors."""
        if not global_obs_tensors:
            return 0.0
        try:
            # Stack per-agent obs and average -> single 51-dim representation
            stacked = torch.stack(global_obs_tensors)   # (N, OBS_DIM)
            avg_obs = stacked.mean(dim=0, keepdim=True)  # (1, OBS_DIM)
            with torch.no_grad():
                return float(self.critic_net(avg_obs).item())
        except Exception:
            return 0.0

    def compute_log_prob(self, agent_id: str,
                         obs_tensor:      torch.Tensor,
                         action_indices:  Dict[str, int],
                         prb_frac:        List[float]) -> float:
        """
        Compute Σ log π(a_i | o_i) across all heads for one step.
        Used at collection time to record π_old.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return 0.0
        try:
            with torch.no_grad():
                logits = agent.policy_net(obs_tensor.unsqueeze(0))
            lp = 0.0
            for h in DISCRETE_HEADS:
                if h in logits and h in action_indices:
                    dist = Categorical(logits=logits[h])
                    idx  = torch.tensor(action_indices[h], dtype=torch.long)
                    lp  += float(dist.log_prob(idx).item())
            if 'prb' in logits:
                log_soft = F.log_softmax(logits['prb'], dim=-1)
                prb_t    = torch.tensor(prb_frac, dtype=torch.float32)
                lp      -= float((prb_t * log_soft).sum().item())
            return lp
        except Exception:
            return 0.0

    def compute_log_prob_batch(self,
                               agent_ids:     List[str],
                               obs_batch:     torch.Tensor,
                               action_list:   List[Dict[str, int]],
                               prb_list:      List[List[float]]) -> List[float]:
        """
        Vectorised log-prob computation for N agents in ONE forward pass.

        Args:
            agent_ids   List of N agent IDs (must all share the same policy net)
            obs_batch   (N, OBS_DIM) stacked observation tensor
            action_list List[N] of action_index dicts per agent
            prb_list    List[N] of [e_frac, r_frac, g_frac] per agent
        Returns:
            List[N] of float log-probs
        """
        if not agent_ids:
            return []
        ref_agent = self.agents.get(agent_ids[0])
        if ref_agent is None:
            return [0.0] * len(agent_ids)
        try:
            N = len(agent_ids)
            with torch.no_grad():
                logits = ref_agent.policy_net(obs_batch)   # Dict[head -> (N, C)]

            # Discrete heads: compute (N,) log-probs per head, sum them
            lp = torch.zeros(N)
            for h in DISCRETE_HEADS:
                if h in logits:
                    log_p = F.log_softmax(logits[h], dim=-1)    # (N, C)
                    acts  = torch.tensor(
                        [a.get(h, 0) for a in action_list], dtype=torch.long
                    )   # (N,)
                    lp = lp + log_p.gather(1, acts.unsqueeze(1)).squeeze(1)

            # PRB head: CE loss as negative log-prob
            if 'prb' in logits:
                log_soft = F.log_softmax(logits['prb'], dim=-1)  # (N, 3)
                prb_t    = torch.tensor(prb_list, dtype=torch.float32)  # (N, 3)
                lp = lp - (prb_t * log_soft).sum(dim=-1)

            return lp.tolist()
        except Exception:
            return [0.0] * len(agent_ids)

    # ── Collection ────────────────────────────────────────────────────────────

    def collect(self, agent_id: str,
                obs_tensor:     torch.Tensor,
                action_indices: Dict[str, int],
                prb_frac:       List[float],
                log_prob:       float,
                value:          float,
                reward:         float,
                done:           bool):
        """Normalise reward and store in SHARED pool (all agents together)."""
        if agent_id not in self.buffers:
            return
        # Normalise reward with running stats
        self._reward_rms.update(reward)
        r_norm = self._reward_rms.normalize(reward, clip=5.0)
        # Append directly to shared pool
        self._pool.add(obs_tensor, action_indices, prb_frac,
                       r_norm, done, value, log_prob)
        # Cache observations for EWC Fisher (capped at 2000)
        cache = self.obs_cache.get(agent_id, [])
        if len(cache) < 2000:
            cache.append(obs_tensor.detach())
            self.obs_cache[agent_id] = cache

    def buffer_size(self) -> int:
        """Return total transitions in shared pool."""
        return len(self._pool)

    # -- PPO Update (shared-parameter MAPPO) ------------------------------------

    def update(self, next_global_obs: Optional[List[torch.Tensor]] = None
               ) -> Dict[str, float]:
        """
        Shared-parameter MAPPO update.
        All agents contribute to one pool; ONE policy network is updated
        using ALL transitions — 80x more data than per-agent buffers.
        """
        cfg  = self.config
        pool = self._pool
        T    = len(pool)

        if T < cfg.mini_batch or self._ref_actor is None:
            return {'policy_loss': 0.0, 'value_loss': 0.0,
                    'entropy': 0.0, 'updates': 0, 'pool_size': T}

        eps  = cfg.deploy_clip_eps  if self.deploy_mode else cfg.clip_eps
        beta = cfg.deploy_entropy   if self.deploy_mode else cfg.entropy_coef
        K    = cfg.deploy_n_epochs  if self.deploy_mode else cfg.n_epochs

        # Convert deques to lists once (O(n) but done only at update time, not per-tick)
        _obs_list     = list(pool.obs)
        _lp_list      = list(pool.log_probs)
        _prb_list     = list(pool.prb_acts)
        _rewards_list = list(pool.rewards)
        _dones_list   = list(pool.dones)
        _values_list  = list(pool.values)
        _actions_list = list(pool.actions)

        # Re-compute GAE from lists (compute_gae expects List)
        next_val = self.get_value(next_global_obs) if next_global_obs else 0.0
        adv, ret = compute_gae(
            _rewards_list, _dones_list, _values_list,
            next_val, cfg.gamma, cfg.gae_lambda
        )
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t    = torch.stack(_obs_list)                               # (T, OBS)
        old_lp_t = torch.tensor(_lp_list,  dtype=torch.float32)        # (T,)
        prb_t    = torch.tensor(_prb_list,  dtype=torch.float32)       # (T, 3)
        ret_t    = ret                                                  # (T,) critic target

        # Build per-head integer action tensors
        head_tensors: Dict[str, torch.Tensor] = {}
        for h in DISCRETE_HEADS:
            vals = [a.get(h, 0) for a in _actions_list]
            if any(v != 0 for v in vals):
                head_tensors[h] = torch.tensor(vals, dtype=torch.long)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        # -- K epochs over the full shared pool --------------------------------
        for _ in range(K):
            perm = torch.randperm(T)
            for start in range(0, T, cfg.mini_batch):
                mb = perm[start:start + cfg.mini_batch]
                if len(mb) < 4:
                    continue

                mb_obs    = obs_t[mb]
                mb_adv    = adv[mb]
                mb_old_lp = old_lp_t[mb]
                mb_prb    = prb_t[mb]
                mb_ret    = ret_t[mb]

                logits = self._ref_actor(mb_obs)

                # New log-probs and entropy
                new_lp  = torch.zeros(len(mb))
                entropy = torch.tensor(0.0)
                for h in DISCRETE_HEADS:
                    if h in logits and h in head_tensors:
                        dist    = Categorical(logits=logits[h])
                        new_lp  = new_lp + dist.log_prob(head_tensors[h][mb])
                        entropy = entropy + dist.entropy().mean()

                # PRB cross-entropy contribution
                if 'prb' in logits:
                    log_soft = F.log_softmax(logits['prb'], dim=-1)
                    new_lp   = new_lp - (-(mb_prb * log_soft).sum(dim=-1))

                # PPO clipped actor loss
                ratio  = torch.exp((new_lp - mb_old_lp).clamp(-10, 10))
                pg1    = ratio * mb_adv
                pg2    = ratio.clamp(1.0 - eps, 1.0 + eps) * mb_adv
                p_loss = -torch.min(pg1, pg2).mean()

                # Critic MSE loss on GAE returns
                v_pred = self.critic_net(mb_obs).squeeze(-1)  # (B,)
                v_loss = F.mse_loss(v_pred, mb_ret)

                # Joint loss: actor + critic + entropy
                loss = p_loss + cfg.value_coef * v_loss - beta * entropy

                if self.deploy_mode and self.ewc:
                    ewc_pen = next(iter(self.ewc.values()))
                    loss = loss + ewc_pen.ewc_loss(self._ref_actor)

                self._actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._ref_actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), cfg.max_grad_norm)
                self._actor_opt.step()
                self.critic_opt.step()

                total_policy_loss += float(p_loss.item())
                total_value_loss  += float(v_loss.item())
                total_entropy     += float(entropy.item())
                n_updates         += 1

        # Ring-buffer (deque maxlen) handles capacity automatically.
        # No aggressive eviction — let transitions accumulate for
        # richer gradient estimates across episodes.

        self._total_updates += 1
        n = max(1, n_updates)
        return {
            'policy_loss': total_policy_loss / n,
            'value_loss':  total_value_loss / n,
            'entropy':     total_entropy / n,
            'updates':     n_updates,
            'pool_size':   T,
        }

    def _update_critic(self) -> float:
        """Deprecated: critic now updated jointly in update()."""
        return 0.0

