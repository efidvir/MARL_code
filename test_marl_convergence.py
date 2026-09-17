#!/usr/bin/env python3
"""
Convergence smoke test for the CURRENT MAPPO stack.

Builds the real PolicyNetwork/CriticNetwork + MAPPOTrainer with 3 RLAgents
and runs a toy multi-agent bandit-style environment over real action heads:
reward genuinely depends on the SAMPLED actions (a known head combination —
relay=CAPACITY_BOOST, scheduler=EMERGENCY_FIRST, iops=ADMIT_ALL — yields high
reward).  Trajectories are collected through the fixed collection path
(log-prob and value evaluated on the SAME obs the action was sampled from,
per-agent GAE, precomputed advantages consumed by update()).

Asserts:
  (a) mean sampled reward improves significantly over the random baseline
  (b) policy entropy decreases over training
  (c) all losses stay finite

Uses the real AgentObservation / action tensor shapes so the test also guards
the observation schema (59-dim) and the full discrete-head set (incl. iops).

Runtime: well under 2 minutes on CPU.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sixg_sim.agent import (
    RLAgent, CriticNetwork, PolicyNetwork, AgentObservation,
    NeighborRadioSummary, ConnectivityState, OBS_DIM,
    DISCRETE_HEADS as AGENT_DISCRETE_HEADS,
)
from sixg_sim.mappo_trainer import (
    MAPPOTrainer, MAPPOConfig, DISCRETE_HEADS as TRAINER_DISCRETE_HEADS,
)
from sixg_sim.phy_mac_state import PHYMACState
from sixg_sim.traffic import SliceDictionary

SEED = 1234

N_AGENTS      = 3
EPISODES      = 45
STEPS_PER_EP  = 60

# Reward structure of the toy bandit: a known head combination is optimal.
# relay=3 (CAPACITY_BOOST), scheduler=2 (EMERGENCY_FIRST), iops=2 (ADMIT_ALL).
# iops is included deliberately: it verifies the IOPS head actually trains.
GOOD_RELAY, GOOD_SCHED, GOOD_IOPS = 3, 2, 2
W_RELAY, W_SCHED, W_IOPS = 0.4, 0.3, 0.3

# Analytic expected reward of a uniform-random policy:
#   relay uniform over 5, scheduler over 4, iops over 3
RANDOM_BASELINE = W_RELAY * (1 / 5) + W_SCHED * (1 / 4) + W_IOPS * (1 / 3)


def make_obs(node_id: str, tick: int, rng: np.random.RandomState) -> AgentObservation:
    """Real AgentObservation (59-dim schema) with mild observation noise."""
    ps = PHYMACState(node_id=node_id)
    ps.sinr_average = 15.0 + rng.uniform(-2.0, 2.0)
    ps.active_ue_count = 10
    ps.emergency_ue_count = 2
    return AgentObservation(
        node_id=node_id,
        is_island=True,
        energy_soc=0.8 + rng.uniform(-0.05, 0.05),
        phy_mac=ps,
        neighbor_radio=NeighborRadioSummary(),
        connectivity=ConnectivityState(),
        ticks_since_severance=tick,
        current_tick=tick,
    )


def env_reward(action) -> float:
    """Reward depends ONLY on the sampled actions (bandit)."""
    r = 0.0
    if action.relay_mode_idx == GOOD_RELAY:
        r += W_RELAY
    if action.scheduler_idx == GOOD_SCHED:
        r += W_SCHED
    if int(getattr(action, '_iops_decision', 0)) == GOOD_IOPS:
        r += W_IOPS
    return r


def action_to_heads(action) -> dict:
    return {
        'tx_power':  action.tx_power_step,
        'mcs_emrg':  action.mcs_emergency_idx,
        'mcs_gen':   action.mcs_general_idx,
        'relay':     action.relay_mode_idx,
        'handover':  action.handover_idx,
        'scheduler': action.scheduler_idx,
        'postcard':  int(action.send_postcard),
        'iops':      int(getattr(action, '_iops_decision', 0)),
    }


def run_convergence_test():
    print("=" * 60)
    print("MARL CONVERGENCE SMOKE TEST (MAPPO stack)")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)

    # ── Schema guards ──────────────────────────────────────────────────────
    assert 'iops' in AGENT_DISCRETE_HEADS,   "iops head missing from agent DISCRETE_HEADS"
    assert 'iops' in TRAINER_DISCRETE_HEADS, "iops head missing from trainer DISCRETE_HEADS"

    sd = SliceDictionary()
    agents = {f"node_{i}": RLAgent(f"node_{i}", sd, is_training=True)
              for i in range(N_AGENTS)}
    critic = CriticNetwork(OBS_DIM)

    cfg = MAPPOConfig(
        mini_batch=90,        # 3 agents x 60 steps = 180 transitions/episode
        n_epochs=4,
        entropy_coef=0.01,
        entropy_min=0.001,
        lr_actor=1e-3,        # fast toy convergence
        lr_critic=1e-3,
    )
    trainer = MAPPOTrainer(agents, critic, cfg)

    # Guard the obs tensor schema with a real forward pass
    probe = agents['node_0'].observation_to_tensor(make_obs('node_0', 0, rng))
    assert probe.shape == (OBS_DIM,), f"obs tensor is {probe.shape}, expected ({OBS_DIM},)"
    probe_logits = agents['node_0'].policy_net(probe.unsqueeze(0))
    for h in TRAINER_DISCRETE_HEADS:
        assert h in probe_logits, f"policy net missing head '{h}'"
    assert probe_logits['iops'].shape[-1] == 3, "iops head must have 3 classes"

    ep_rewards, entropies, p_losses, v_losses = [], [], [], []

    for ep in range(EPISODES):
        step_rewards = []
        for t in range(STEPS_PER_EP):
            obs_map = {aid: make_obs(aid, t, rng) for aid in agents}
            obs_t = {aid: agents[aid].observation_to_tensor(o)
                     for aid, o in obs_map.items()}

            # Mean-field global obs + value — the SAME inputs stored with the
            # transition (consistent critic training).
            all_obs = list(obs_t.values())
            global_obs = torch.stack(all_obs).mean(dim=0)
            value = trainer.get_value(all_obs)

            # Sample real actions through the actual sampling path
            actions = {aid: agents[aid].compute_action(obs_map[aid])
                       for aid in agents}
            heads_list = [action_to_heads(actions[aid]) for aid in agents]
            prb_list = [[actions[aid].prb_emergency_frac,
                         actions[aid].prb_relay_frac,
                         actions[aid].prb_general_frac] for aid in agents]

            # Old log-probs on the SAME obs the action was sampled from
            aid_list = list(agents.keys())
            log_probs = trainer.compute_log_prob_batch(
                aid_list, torch.stack([obs_t[a] for a in aid_list]),
                heads_list, prb_list, temperature=1.0,
            )

            done = (t == STEPS_PER_EP - 1)
            for j, aid in enumerate(aid_list):
                r = env_reward(actions[aid])
                step_rewards.append(r)
                trainer.collect(aid, obs_t[aid], heads_list[j], prb_list[j],
                                log_probs[j], value, r, done,
                                global_obs=global_obs)

        metrics = trainer.update()
        assert metrics['updates'] > 0, (
            f"update() performed no gradient steps (pool={metrics['pool_size']})")

        mean_r = float(np.mean(step_rewards))
        ep_rewards.append(mean_r)
        entropies.append(metrics['entropy'])
        p_losses.append(metrics['policy_loss'])
        v_losses.append(metrics['value_loss'])

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Ep {ep + 1:3d}: reward={mean_r:.3f}  "
                  f"entropy={metrics['entropy']:.3f}  "
                  f"p_loss={metrics['policy_loss']:.4f}  "
                  f"v_loss={metrics['value_loss']:.5f}")

    first5_r = float(np.mean(ep_rewards[:5]))
    last5_r = float(np.mean(ep_rewards[-5:]))
    first3_H = float(np.mean(entropies[:3]))
    last3_H = float(np.mean(entropies[-3:]))

    print()
    print(f"Random baseline (analytic) : {RANDOM_BASELINE:.3f}")
    print(f"First-5 mean reward        : {first5_r:.3f}")
    print(f"Last-5  mean reward        : {last5_r:.3f}")
    print(f"Entropy first-3 -> last-3  : {first3_H:.3f} -> {last3_H:.3f}")

    reward_improved = (last5_r > RANDOM_BASELINE + 0.15) and (last5_r > first5_r + 0.10)
    entropy_decreased = last3_H < first3_H
    losses_finite = (all(np.isfinite(p_losses)) and all(np.isfinite(v_losses))
                     and all(np.isfinite(entropies)) and all(np.isfinite(ep_rewards)))

    print()
    def chk(v): return "[PASS]" if v else "[FAIL]"
    print(f"{chk(reward_improved)}  Reward improved over random baseline : "
          f"{last5_r:.3f} vs {RANDOM_BASELINE:.3f} (+0.15 margin required)")
    print(f"{chk(entropy_decreased)}  Entropy decreased                    : "
          f"{first3_H:.3f} -> {last3_H:.3f}")
    print(f"{chk(losses_finite)}  Losses finite                        : {losses_finite}")

    if reward_improved and entropy_decreased and losses_finite:
        print("\n[ALL PASS] MAPPO training converges on the toy environment")
        sys.exit(0)
    else:
        print("\n[FAILED] Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    run_convergence_test()
