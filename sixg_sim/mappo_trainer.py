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
from torch.distributions import Categorical, Dirichlet

from .agent import apply_relay_mask

import os as _os
# Minimum entropy coefficient for the relay head during training (see the
# loss assembly in MAPPOTrainer.update).  0 = off (default).
_RELAY_H_FLOOR = float(_os.environ.get('MARL_RELAY_ENTROPY_FLOOR', '0') or 0)

from .agent import (PRB_POLICY as _PRB_POLICY,
                    prb_distribution as _prb_dist)


def _prb_log_prob(prb_logits: torch.Tensor,
                  prb_action: torch.Tensor) -> torch.Tensor:
    """Log-probability of a PRB allocation under the current policy.

    Dirichlet density when MARL_PRB_POLICY=dirichlet.  Under the legacy
    'softmax' setting the head is deterministic and has no density, so the
    previous cross-entropy surrogate is retained to reproduce older runs.
    The action is clamped off the simplex boundary because the Dirichlet
    density is undefined at a zero component, which the coordinator clamp and
    the general-traffic floor can produce.
    """
    if _PRB_POLICY != 'dirichlet':
        log_soft = F.log_softmax(prb_logits, dim=-1)
        return (prb_action * log_soft).sum(dim=-1)
    a = prb_action.clamp_min(1e-6)
    a = a / a.sum(dim=-1, keepdim=True)
    return _prb_dist(prb_logits).log_prob(a)



# ─── Hyperparameter dataclass ─────────────────────────────────────────────────

@dataclass
class MAPPOConfig:
    # Core RL
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    clip_eps:      float = 0.20     # PPO epsilon
    # ENTROPY.  The bonus in update() is a SUM over the discrete heads, not a
    # mean, so its effective strength scaled with the head count: with the
    # old 8-head set the term was beta * ~9.9 nats.  At beta = 0.10 that
    # gradient was comparable to (and for a shared-reward advantage with a
    # measured own-action R^2 of 0.005, larger than) the policy gradient, so
    # the actor was driven toward a uniform policy for the first third of the
    # run.  For reference the toy convergence test (test_marl_convergence.py),
    # which DOES learn, uses entropy_coef = 0.01.
    entropy_coef:       float = 0.02   # live value (decayed by the driver)
    entropy_coef_start: float = 0.02   # schedule anchor — see train_on_comparison
    entropy_decay:      float = 0.985  # legacy per-episode multiplicative decay
    entropy_min:        float = 0.001  # hard floor — forces exploitation
    value_coef:    float = 1.0      # Balanced with policy loss
    max_grad_norm: float = 0.50
    n_epochs:      int   = 8        # More PPO passes per update for stability
    mini_batch:    int   = 512      # Large batch = fewer steps per epoch = fast

    # Learning rates
    lr_actor:      float = 1e-4
    lr_critic:     float = 3e-4     # Higher critic LR — critic needs to converge faster

    # Shared pool capacity.
    #
    # SIZING BUG FIXED.  The pool is a deque(maxlen=capacity), so anything
    # beyond `capacity` is SILENTLY EVICTED — oldest first.  A default
    # train_on_comparison batch collects
    #     batch_size x episode_ticks x n_agents = 16 x 800 x 42 ~ 537,000
    # transitions, of which the old 32,768 capacity kept ~6 %.  Worse, the
    # eviction is not a uniform subsample: workers extend the pool as their
    # futures complete, so the surviving transitions come from whichever one
    # or two episodes finished LAST.  Each PPO update therefore trained on a
    # near-single-episode, non-representative slice of the batch.
    # train_on_comparison.py now sizes this from the actual batch geometry
    # (see the pool_capacity computation in main()); this default covers a
    # small run on its own.
    pool_capacity:  int  = 262144

    # Deployment / online fine-tuning
    deploy_lr_actor:  float = 5e-5    # Visible online learning (2× lower than training)
    deploy_lr_critic: float = 2e-4    # Critic tracks faster for new topology
    deploy_clip_eps:  float = 0.10    # Allow meaningful policy shifts
    deploy_entropy:   float = 5e-3    # More exploration on unseen topology
    deploy_n_epochs:  int   = 4       # More gradient steps per update

    # EWC (Elastic Weight Consolidation)
    ewc_lambda:    float = 0.40

    # Multi-eNB IOPS cooperative learning (ETSI TS 22.346)
    peer_learning_weight:    float = 0.1    # Weight of peer gradient mixing
    iops_pretraining_episodes: int = 5      # Extra pretraining on IOPS scenarios

    # Checkpointing
    checkpoint_interval: int = 100


# ─── Reward scaling ───────────────────────────────────────────────────────────

# Fixed, deterministic reward scale used instead of per-worker running
# normalisation.  Per-worker RunningMeanStd made rewards from easy and hard
# episodes incomparable (each worker had its own statistics) and per-episode
# mean-subtraction erased absolute differences between episodes.
# Typical per-tick rewards are O(100) in island mode (the global connectivity
# reward pays +100 * volume-weighted delivered fraction plus bonuses/penalties).
# With the reunification block (up to +43 after the repricing) and the capped
# stranded-user reachability term (up to +REACH_RESTORED_CAP = +5.0, cut from
# +60) the island-mode range is roughly [-60, 250], so dividing by 100 lands
# typical rewards in ~[-0.6, 2.5]
# — still the right order of magnitude for the value head and the 0.4 local /
# 0.6 global mix in worker.py.  See
# Simulator.compute_global_connectivity_reward for the term-by-term budget.
REWARD_SCALE = 100.0


# ─── Running reward normaliser (kept for reference; no longer used for rewards) ─

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
    Compatible with parameter-sharing MAPPO.

    Stores PROCESSED transitions: advantages and returns are computed
    per-agent, per-episode (correctly ordered single-agent trajectories)
    BEFORE they land here, so the pool may freely interleave agents and
    episodes — GAE is never run over this merged pool.
    """

    def __init__(self, capacity: int = 16384):
        self.capacity = capacity
        self.clear()

    def clear(self):
        from collections import deque as _deque
        self.obs        = _deque(maxlen=self.capacity)   # per-agent obs o_t
        self.global_obs = _deque(maxlen=self.capacity)   # mean-field global obs (critic input)
        self.actions    = _deque(maxlen=self.capacity)
        self.prb_acts   = _deque(maxlen=self.capacity)
        self.log_probs  = _deque(maxlen=self.capacity)   # log pi_old(a_t | o_t)
        self.advantages = _deque(maxlen=self.capacity)   # precomputed per-agent GAE
        self.returns    = _deque(maxlen=self.capacity)   # precomputed GAE returns (critic target)

    def add(self, obs: torch.Tensor,
            action_indices: Dict[str, int],
            prb_frac: List[float],
            log_prob: float,
            advantage: float,
            ret: float,
            global_obs: torch.Tensor):
        # deque(maxlen=capacity) evicts oldest automatically — O(1)
        self.obs.append(obs.detach())
        self.global_obs.append(global_obs.detach())
        self.actions.append(dict(action_indices))
        self.prb_acts.append(list(prb_frac))
        self.log_probs.append(float(log_prob))
        self.advantages.append(float(advantage))
        self.returns.append(float(ret))

    def __len__(self) -> int:
        return len(self.log_probs)


# ─── Legacy per-agent buffer (kept for API compatibility) ─────────────────────

class AgentRolloutBuffer:
    """Thin wrapper view over the SharedRolloutPool (length only)."""

    def __init__(self, pool: 'SharedRolloutPool'):
        self._pool = pool

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

        # Default mirrors the trained head set (no handover/scheduler — they
        # are never read by the simulator, so anchoring Fisher mass on them
        # would protect weights that cannot affect behaviour).
        heads_to_use = discrete_heads or ["tx_power", "mcs_emrg", "mcs_gen",
                                          "relay", "postcard", "iops"]
        n = min(n_samples, len(obs_buffer))
        indices = np.random.choice(len(obs_buffer), n, replace=False)

        for idx in indices:
            obs = obs_buffer[int(idx)].unsqueeze(0)
            policy_net.zero_grad()
            logits = policy_net(obs)

            log_prob = torch.tensor(0.0, requires_grad=True)
            for h in heads_to_use:
                if h in logits:
                    # Fisher must be estimated at SAMPLED actions a ~ pi(.|s),
                    # not at the argmax action (which biases F toward the mode).
                    dist = Categorical(logits=logits[h])
                    a    = dist.sample()
                    log_prob = log_prob + dist.log_prob(a).sum()

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
        from .agent import load_policy_state_dict
        load_policy_state_dict(policy_net, ckpt['state_dict'])
        self.version = v + 1
        return v

    def load_version(self, policy_net: nn.Module, version: int) -> bool:
        path = self._ckpt_path(version)
        if not path.exists():
            return False
        ckpt = torch.load(path, weights_only=False)
        from .agent import load_policy_state_dict
        load_policy_state_dict(policy_net, ckpt['state_dict'])
        return True


# ─── MAPPO Trainer ─────────────────────────────────────────────────────────────

# Which discrete heads the PPO loss covers
# NOTE: 'iops' drives real admit/deny decisions in simulation.py and MUST be
# trained like every other head (it was previously missing here).
#
# 'handover' and 'scheduler' are DELIBERATELY ABSENT: they are written into
# PHYMACState and never read by anything (see the DISCRETE_HEADS comment in
# agent.py for the file:line evidence and the checkpoint-compatibility note).
# This list is imported by agent.py-independent call sites, so it is kept
# byte-identical in content to agent.DISCRETE_HEADS — the assertion in
# test_marl_convergence.py guards that.
DISCRETE_HEADS = ["tx_power", "mcs_emrg", "mcs_gen",
                   "relay", "postcard", "iops"]


class MAPPOTrainer:
    """
    Multi-Agent PPO with centralised-critic, decentralised actors.

    Usage (training):
        trainer = MAPPOTrainer(agents, critic_net, config)
        # each tick, AT THE MOMENT the action a_t is sampled from o_t:
        value    = trainer.get_value(global_obs_list)          # V(s_t), mean-field obs
        log_prob = trainer.compute_log_prob(agent_id, obs_t, action_indices, prb,
                                            temperature=temp_used_at_sampling)
        # when r_t arrives (next tick), complete the transition:
        trainer.collect(agent_id, obs_t, action_indices, prb, log_prob, value,
                        reward, done, global_obs=mean_obs_t)
        # at episode end:  trainer.finish_episode()
        # every N ticks / at batch end:
        metrics  = trainer.update()     # consumes precomputed per-agent GAE

    Transitions are buffered into PER-AGENT trajectories; GAE runs on each
    agent's correctly-ordered single trajectory (never across interleaved
    agents or concatenated episodes), then processed (adv, return) tuples are
    flushed into the shared pool for minibatched PPO epochs.

    Usage (after training, before deployment):
        trainer.anchor_ewc()           # compute Fisher matrices
        trainer.set_deployment_mode()  # lower LR, activate EWC penalty

    Usage (deployment):
        trainer.collect(...)           # same as training
        every K ticks: trainer.update()
    """

    def __init__(self, agents: dict, critic_net: nn.Module,
                 config: Optional[MAPPOConfig] = None):
        self.device = torch.device('cpu')
        self.config      = config or MAPPOConfig()
        self.critic_net  = critic_net.to(self.device)
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

        # Per-agent in-flight trajectories (ordered single-agent transitions).
        # GAE is computed on these at flush time, NOT on the merged pool.
        self._traj: Dict[str, Dict[str, list]] = {}

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
            self.critic_net.parameters(), lr=cfg.lr_critic
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
            stacked = torch.stack(global_obs_tensors).to(self.device)   # (N, OBS_DIM)
            avg_obs = stacked.mean(dim=0, keepdim=True)  # (1, OBS_DIM)
            with torch.no_grad():
                return float(self.critic_net(avg_obs).item())
        except Exception:
            return 0.0

    def compute_log_prob(self, agent_id: str,
                         obs_tensor:      torch.Tensor,
                         action_indices:  Dict[str, int],
                         prb_frac:        List[float],
                         temperature:     float = 1.0) -> float:
        """
        Compute Σ log π(a_i | o_i) across all heads for one step.
        Used at collection time to record π_old.

        `temperature` MUST match the temperature used when the action was
        SAMPLED (softmax(logits / temp)), so the stored old log-prob describes
        the actual behaviour policy.  Default 1.0 = untempered.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return 0.0
        temp = max(float(temperature), 1e-6)
        try:
            with torch.no_grad():
                logits = agent.policy_net(obs_tensor.unsqueeze(0))
            logits = apply_relay_mask(logits, obs_tensor.unsqueeze(0))
            lp = 0.0
            for h in DISCRETE_HEADS:
                if h in logits and h in action_indices:
                    dist = Categorical(logits=logits[h] / temp)
                    idx  = torch.tensor(action_indices[h], dtype=torch.long, device=self.device)
                    lp  += float(dist.log_prob(idx).item())
            if 'prb' in logits:
                # PRB fractions come from an UNtempered softmax (see
                # simulation.py batch action sampling) — no temperature here.
                prb_t = torch.tensor(prb_frac, dtype=torch.float32,
                                     device=self.device)
                lp += float(_prb_log_prob(logits['prb'], prb_t).item())
            return lp
        except Exception:
            return 0.0

    def compute_log_prob_batch(self,
                               agent_ids:     List[str],
                               obs_batch:     torch.Tensor,
                               action_list:   List[Dict[str, int]],
                               prb_list:      List[List[float]],
                               temperature:   float = 1.0) -> List[float]:
        """
        Vectorised log-prob computation for N agents in ONE forward pass.

        Args:
            agent_ids   List of N agent IDs (must all share the same policy net)
            obs_batch   (N, OBS_DIM) stacked observation tensor
            action_list List[N] of action_index dicts per agent
            prb_list    List[N] of [e_frac, r_frac, g_frac] per agent
            temperature Softmax temperature used when the actions were SAMPLED
                        (so recorded log-probs describe the behaviour policy)
        Returns:
            List[N] of float log-probs
        """
        if not agent_ids:
            return []
        ref_agent = self.agents.get(agent_ids[0])
        if ref_agent is None:
            return [0.0] * len(agent_ids)
        temp = max(float(temperature), 1e-6)
        try:
            N = len(agent_ids)
            with torch.no_grad():
                logits = ref_agent.policy_net(obs_batch)   # Dict[head -> (N, C)]
            logits = apply_relay_mask(logits, obs_batch)

            # Discrete heads: compute (N,) log-probs per head, sum them
            lp = torch.zeros(N)
            for h in DISCRETE_HEADS:
                if h in logits:
                    log_p = F.log_softmax(logits[h] / temp, dim=-1)    # (N, C)
                    acts  = torch.tensor(
                        [a.get(h, 0) for a in action_list], dtype=torch.long, device=self.device
                    )   # (N,)
                    lp = lp + log_p.gather(1, acts.unsqueeze(1)).squeeze(1)

            # PRB head: CE loss as negative log-prob
            if 'prb' in logits:
                prb_t = torch.tensor(prb_list, dtype=torch.float32,
                                     device=self.device)              # (N, 3)
                lp = lp + _prb_log_prob(logits['prb'], prb_t)

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
                done:           bool,
                global_obs:     Optional[torch.Tensor] = None,
                step:            Optional[int] = None):
        """
        Store one COMPLETED transition (o_t, a_t, logπ(a_t|o_t), V(s_t), r_t, done_t)
        in this agent's private trajectory.  log_prob and value must have been
        evaluated on the SAME observation the action was sampled from.

        `global_obs` is the mean-field global observation (critic input) at
        time t; it is stored alongside so the critic trains on the same input
        distribution used to produce `value`.  Falls back to the per-agent obs
        if not provided (single-agent value estimation call sites).

        Rewards are scaled by the fixed REWARD_SCALE constant (deterministic,
        comparable across workers and episodes).

        When done=True the trajectory is flushed: GAE is computed over this
        agent's ordered transitions and the processed tuples enter the pool.
        """
        if agent_id not in self.buffers:
            return
        tr = self._traj.get(agent_id)
        if tr is None:
            tr = {'obs': [], 'global_obs': [], 'actions': [], 'prb': [],
                  'log_probs': [], 'values': [], 'rewards': [], 'dones': [],
                  'steps': []}
            self._traj[agent_id] = tr

        g_obs = global_obs if global_obs is not None else obs_tensor
        tr['obs'].append(obs_tensor.detach())
        tr['global_obs'].append(g_obs.detach())
        tr['actions'].append(dict(action_indices))
        tr['prb'].append(list(prb_frac))
        tr['log_probs'].append(float(log_prob))
        tr['values'].append(float(value))
        tr['rewards'].append(float(reward) / REWARD_SCALE)
        tr['dones'].append(bool(done))
        # Step index is used ONLY to align agents for the counterfactual
        # baseline in finish_episode().  Falls back to the trajectory
        # position when a call site does not supply one (that is exact for
        # any agent present on every tick).
        tr['steps'].append(int(step) if step is not None
                           else len(tr['rewards']) - 1)

        if done:
            self._flush_trajectory(agent_id)

        # Cache observations for EWC Fisher (capped at 2000)
        cache = self.obs_cache.get(agent_id, [])
        if len(cache) < 2000:
            cache.append(obs_tensor.detach())
            self.obs_cache[agent_id] = cache

    def _flush_trajectory(self, agent_id: str,
                          bootstrap_value: Optional[float] = None):
        """
        Compute GAE over ONE agent's ordered trajectory and move the processed
        (obs, action, log_prob, advantage, return, global_obs) tuples into the
        shared pool.  This is the ONLY place GAE runs — never over the pool.
        """
        tr = self._traj.pop(agent_id, None)
        if not tr or not tr['rewards']:
            return
        cfg = self.config
        if tr['dones'][-1]:
            next_value = 0.0     # terminal — no bootstrap
        elif bootstrap_value is not None:
            next_value = float(bootstrap_value)
        else:
            # Truncated (not terminal) with no explicit bootstrap available:
            # use V(s_T) of the last stored state as a proxy for V(s_{T+1}).
            next_value = tr['values'][-1]

        adv, ret = compute_gae(tr['rewards'], tr['dones'], tr['values'],
                               next_value, cfg.gamma, cfg.gae_lambda)
        for i in range(len(tr['rewards'])):
            self._pool.add(tr['obs'][i], tr['actions'][i], tr['prb'][i],
                           tr['log_probs'][i], float(adv[i]), float(ret[i]),
                           tr['global_obs'][i])

    def finish_episode(self, counterfactual_baseline: bool = True):
        """
        Call at episode end: marks the last stored transition of every
        in-flight trajectory as terminal (done=True), computes per-agent GAE,
        applies the COUNTERFACTUAL (difference) BASELINE across agents, and
        flushes everything into the shared pool.

        THE COUNTERFACTUAL BASELINE — why it is here and why it is valid.

        The reward every agent receives is overwhelmingly SHARED: the
        training drivers mix 0.6 x Simulator.compute_global_connectivity_reward
        (one scalar for the whole fleet at a tick) with 0.4 x the agent's own
        reward, and V(s_t) is a single mean-field scalar shared by all agents
        too (see get_value).  Instrumented on the Scenario-A comparison
        environment (seed 42, 250 ticks, 35 agents, per-agent GAE exactly as
        computed here):

            eta^2 of the TICK alone on an agent's own advantage   0.948
            joint R^2 of ALL of an agent's own action one-hots
                on its own advantage                             0.0052

        i.e. 95 % of the advantage was the shared, time-varying environment
        state and 0.5 % was anything the agent itself did.  The policy
        gradient E[sum_i grad log pi(a_i|o_i) A_i] is still UNBIASED under a
        shared A, but its variance grows with the agent count, and with ~42
        agents at that signal ratio it is dominated by noise.

        The fix is the standard MARL baseline: subtract, at each step, the
        MEAN advantage over the agents present at that step.

            A_i^diff(t) = A_i(t) - mean_j A_j(t)

        This is the cheap empirical form of COMA's counterfactual baseline
        (Foerster et al. 2018).  COMA replaces agent i's action with a default
        and re-evaluates the shared return; here the shared component of the
        return is by construction identical for every agent, so the
        cross-agent mean at a step IS an estimate of "the return that would
        have been obtained regardless of what agent i did".  Subtracting it
        cancels the shared component and leaves exactly the part of the
        advantage that differentiates agents.

        A full COMA counterfactual was rejected on COST, not principle: a tick
        of this simulator costs ~0.19 s, so re-evaluating one counterfactual
        per agent per tick is ~42x, about 8 s per tick — roughly 3500x a
        training episode.  A learned Q(s, a_-i, a_i) critic is the other COMA
        route and would require a joint-action critic over 42 agents x 6 heads.

        BIAS.  A baseline must not depend on the agent's own action to leave
        the gradient unbiased.  The cross-agent mean includes agent i's own
        advantage, so it does — but only with weight 1/N.  With N ~ 35-42 the
        residual bias is O(1/N) ~ 2-3 %, against a ~200x variance reduction on
        the measured numbers above.  This is the standard, accepted trade in
        mean-field MARL.  Set counterfactual_baseline=False to disable it.

        RETURNS ARE NOT CENTRED.  Only the advantages (the actor's signal) get
        the baseline.  pool.returns stays the true GAE return because it is the
        CRITIC's regression target: centring it would train the critic to
        predict a quantity that is zero by construction and destroy the value
        function that produced the advantages in the first place.
        """
        aids = list(self._traj.keys())
        for aid in aids:
            tr = self._traj[aid]
            if tr['dones']:
                tr['dones'][-1] = True

        if not counterfactual_baseline:
            for aid in aids:
                self._flush_trajectory(aid)
            return

        cfg = self.config
        # 1. per-agent GAE on correctly-ordered single-agent trajectories
        processed = {}
        for aid in aids:
            tr = self._traj.pop(aid, None)
            if not tr or not tr['rewards']:
                continue
            next_value = 0.0 if tr['dones'][-1] else tr['values'][-1]
            adv, ret = compute_gae(tr['rewards'], tr['dones'], tr['values'],
                                   next_value, cfg.gamma, cfg.gae_lambda)
            processed[aid] = (tr, adv, ret)

        if not processed:
            return

        # 2. cross-agent mean advantage per STEP (the counterfactual baseline)
        step_sum: Dict[int, float] = {}
        step_n:   Dict[int, int]   = {}
        for aid, (tr, adv, _ret) in processed.items():
            for i, st in enumerate(tr['steps']):
                step_sum[st] = step_sum.get(st, 0.0) + float(adv[i])
                step_n[st]   = step_n.get(st, 0) + 1
        # A step seen by only ONE agent has no cross-agent information; its
        # baseline would be the agent's own advantage, zeroing the signal.
        # Leave those steps uncentred.
        step_mean = {st: (step_sum[st] / step_n[st])
                     for st in step_sum if step_n[st] > 1}

        # 3. flush with centred advantages, uncentred returns
        for aid, (tr, adv, ret) in processed.items():
            for i in range(len(tr['rewards'])):
                a_i = float(adv[i]) - step_mean.get(tr['steps'][i], 0.0)
                self._pool.add(tr['obs'][i], tr['actions'][i], tr['prb'][i],
                               tr['log_probs'][i], a_i, float(ret[i]),
                               tr['global_obs'][i])

    def _flush_all_trajectories(self, next_global_obs: Optional[List[torch.Tensor]] = None):
        """Flush all in-flight trajectories, bootstrapping non-terminal tails."""
        boot = None
        if next_global_obs:
            boot = self.get_value(next_global_obs)
        for aid in list(self._traj.keys()):
            self._flush_trajectory(aid, bootstrap_value=boot)

    def buffer_size(self) -> int:
        """Return total transitions in shared pool + in-flight trajectories."""
        return len(self._pool) + sum(len(t['rewards']) for t in self._traj.values())

    # -- PPO Update (shared-parameter MAPPO) ------------------------------------

    def update(self, next_global_obs: Optional[List[torch.Tensor]] = None
               ) -> Dict[str, float]:
        """
        Shared-parameter MAPPO update.

        Consumes PRECOMPUTED per-agent, per-episode advantages/returns from
        the pool (GAE was already run on ordered single-agent trajectories at
        flush time) and performs minibatched PPO epochs.  GAE is NOT re-run
        over the merged pool.  Clears the pool on every successful update so
        all call sites get on-policy behaviour.
        """
        cfg  = self.config

        # Flush any in-flight trajectories first (bootstrapped if the caller
        # supplied a next global observation, else per-trajectory fallback).
        self._flush_all_trajectories(next_global_obs)

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
        _gobs_list    = list(pool.global_obs)
        _lp_list      = list(pool.log_probs)
        _prb_list     = list(pool.prb_acts)
        _actions_list = list(pool.actions)

        # Precomputed per-agent GAE — normalise advantages over the batch
        adv = torch.tensor(list(pool.advantages), dtype=torch.float32)
        ret = torch.tensor(list(pool.returns),    dtype=torch.float32)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.to(self.device)

        obs_t    = torch.stack(_obs_list).to(self.device)                               # (T, OBS)
        gobs_t   = torch.stack(_gobs_list).to(self.device)                              # (T, OBS) mean-field critic input
        old_lp_t = torch.tensor(_lp_list,  dtype=torch.float32, device=self.device)        # (T,)
        prb_t    = torch.tensor(_prb_list,  dtype=torch.float32, device=self.device)       # (T, 3)
        ret_t    = ret.to(self.device)                                                  # (T,) critic target

        # Build per-head integer action tensors.
        # ALL heads are always included: excluding a head whose stored actions
        # were all zero would drop its term from new_lp while old_lp still
        # contains it, systematically biasing the PPO ratio.
        head_tensors: Dict[str, torch.Tensor] = {}
        for h in DISCRETE_HEADS:
            vals = [a.get(h, 0) for a in _actions_list]
            head_tensors[h] = torch.tensor(vals, dtype=torch.long, device=self.device)

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
                mb_gobs   = gobs_t[mb]
                mb_adv    = adv[mb]
                mb_old_lp = old_lp_t[mb]
                mb_prb    = prb_t[mb]
                mb_ret    = ret_t[mb]

                logits = apply_relay_mask(self._ref_actor(mb_obs), mb_obs)

                # New log-probs and entropy
                new_lp  = torch.zeros(len(mb), device=self.device)
                entropy = torch.tensor(0.0, device=self.device)
                relay_H = torch.tensor(0.0, device=self.device)
                for h in DISCRETE_HEADS:
                    if h in logits and h in head_tensors:
                        dist    = Categorical(logits=logits[h])
                        new_lp  = new_lp + dist.log_prob(head_tensors[h][mb])
                        _H      = dist.entropy().mean()
                        entropy = entropy + _H
                        if h == 'relay':
                            relay_H = _H

                # PRB contribution: a real log-density over the simplex
                if 'prb' in logits:
                    new_lp = new_lp + _prb_log_prob(logits['prb'], mb_prb)
                    if _PRB_POLICY == 'dirichlet':
                        entropy = entropy + _prb_dist(logits['prb']).entropy().mean()

                # PPO clipped actor loss
                ratio  = torch.exp((new_lp - mb_old_lp).clamp(-10, 10))
                pg1    = ratio * mb_adv
                pg2    = ratio.clamp(1.0 - eps, 1.0 + eps) * mb_adv
                p_loss = -torch.min(pg1, pg2).mean()

                # Critic MSE loss on GAE returns.
                # Input is the STORED mean-field global observation — the same
                # distribution used to produce the collection-time values that
                # the returns were derived from (consistent mean-field CTDE).
                v_pred = self.critic_net(mb_gobs).squeeze(-1)  # (B,)
                v_loss = F.mse_loss(v_pred, mb_ret)

                # Joint loss: actor + critic + entropy
                loss = p_loss + cfg.value_coef * v_loss - beta * entropy
                # Relay-head entropy FLOOR (MARL_RELAY_ENTROPY_FLOOR): the
                # global entropy coefficient anneals to ~0 long before the
                # coordinated bridging configuration is discovered, and
                # post-freeze bridge churn is zero -- whatever the relay head
                # has collapsed to is locked in.  Keep the relay head's
                # coefficient from falling below the floor during TRAINING
                # (never in deployment mode, where EWC/deploy_entropy govern).
                if _RELAY_H_FLOOR > beta and not self.deploy_mode:
                    loss = loss - (_RELAY_H_FLOOR - beta) * relay_H

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

        # On-policy: clear the pool after every successful update so stale
        # transitions from the pre-update policy are never reused (this makes
        # ALL call sites — training batch loop and deployment — on-policy).
        pool.clear()

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

