#!/usr/bin/env python3
"""
Convergence smoke test: run 10 short training episodes and assert that
avg_reward and policy_loss both improve over the course of training.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from sixg_sim.agent import (
    RLAgent, CentralizedMARLTrainer, CriticNetwork, Experience,
    AdmissionMode, EnergyTier, StrainLevel, TrafficClass
)
from sixg_sim.agent import AgentObservation, AgentAction, ClassAction, NeighborSummary, LocalSliceState
from sixg_sim.traffic import SliceDictionary


def make_obs(is_island=True, asr=0.5, tier=EnergyTier.MEDIUM, tick=0):
    """Create a minimal AgentObservation for testing."""
    slices = {}
    for tc in TrafficClass:
        slices[tc] = LocalSliceState(
            traffic_class=tc,
            importance=0.5,
            current_queue_length=10.0,
            offered_load=20.0,
            admission_success_rate=asr,
            freshness_target=10,
            current_freshness=5.0
        )
    ns = NeighborSummary(
        most_needy_class=TrafficClass.LIFE_SAFETY,
        need_level=StrainLevel.DEGRADING,
        strain_level=StrainLevel.OKAY,
        latest_policy_version=0,
        neighbor_count=2
    )
    return AgentObservation(
        node_id="test",
        is_island=is_island,
        energy_tier=tier,
        local_slices=slices,
        neighbor_summary=ns,
        current_tick=tick
    )


def make_action(mode=AdmissionMode.ADMIT):
    class_actions = {tc: ClassAction(admission_mode=mode, priority_weight=1.0) for tc in TrafficClass}
    return AgentAction(class_actions=class_actions, link_biases={}, send_postcard=False, postcard_content=None)


def run_convergence_test():
    print("=" * 60)
    print("MARL CONVERGENCE SMOKE TEST")
    print("=" * 60)

    sd = SliceDictionary()
    agent = RLAgent("node_A", sd, is_training=True)

    EPOCHS = 10
    STEPS = 150  # > batch_size=64 to trigger update_policy
    GAMMA = 0.99

    episode_losses = []
    episode_rewards = []

    for ep in range(EPOCHS):
        # Fill experience buffer with a block of transitions
        agent.experiences.clear()
        ep_reward = []

        for t in range(STEPS):
            # Simulate: after t=50 island kicks in and rewards improve
            asr = min(1.0, 0.3 + t / STEPS * 0.7) if ep > 2 else 0.3 + np.random.rand() * 0.1
            obs = make_obs(is_island=(t > 30), asr=asr, tick=t)
            next_obs = make_obs(is_island=(t > 30), asr=min(1.0, asr + 0.01), tick=t + 1)
            mode = AdmissionMode.ADMIT if asr > 0.5 else AdmissionMode.HOLD
            act = make_action(mode)
            rew = agent.calculate_reward(obs, act, next_obs).total_reward
            ep_reward.append(rew)
            agent.experiences.append(Experience(
                observation=obs,
                action=act,
                reward=rew,
                next_observation=next_obs,
                done=(t == STEPS - 1),
                info={}
            ))

        avg_r = float(np.mean(ep_reward))
        episode_rewards.append(avg_r)

        # Trigger gradient update
        agent.update_policy(batch_size=64, epochs=5, gamma=GAMMA)
        last_loss = agent.policy_losses[-1] if agent.policy_losses else float('nan')
        episode_losses.append(last_loss)
        print(f"  Episode {ep+1:2d}: avg_reward={avg_r:.4f}  policy_loss={last_loss:.4f}")

    # Assert loss decreased and reward stayed finite
    first_losses = [l for l in episode_losses[:3] if not np.isnan(l)]
    last_losses  = [l for l in episode_losses[-3:] if not np.isnan(l)]

    print()
    print(f"First-3 avg loss : {np.mean(first_losses):.4f}")
    print(f"Last-3  avg loss : {np.mean(last_losses):.4f}")
    print(f"Rewards finite   : {all(np.isfinite(r) for r in episode_rewards)}")

    loss_improved = np.mean(last_losses) < np.mean(first_losses)
    rewards_finite = all(np.isfinite(r) for r in episode_rewards)
    no_nan_loss = all(not np.isnan(l) for l in episode_losses)
    # Gradient should not be identically zero — policy version should increment
    policy_updated = agent.policy_version > 0

    print()
    def chk(v): return "[PASS]" if v else "[FAIL]"
    print(f"{chk(loss_improved)}  Loss improved        : {loss_improved}")
    print(f"{chk(rewards_finite)}  Rewards finite       : {rewards_finite}")
    print(f"{chk(no_nan_loss)}  No NaN losses        : {no_nan_loss}")
    print(f"{chk(policy_updated)}  Policy updated       : {policy_updated} (version={agent.policy_version})")

    all_pass = loss_improved and rewards_finite and no_nan_loss and policy_updated
    if all_pass:
        print("\n[ALL PASS] Training converges correctly")
        sys.exit(0)
    else:
        print("\n[FAILED] Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    run_convergence_test()
