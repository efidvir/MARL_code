"""Quick convergence checker — run after training completes.
Usage: python check_convergence.py
"""
import json, sys, os

kpi_path = os.path.join("output", "learning_kpis.json")
if not os.path.exists(kpi_path):
    print(f"ERROR: {kpi_path} not found. Training may still be running.")
    sys.exit(1)

d = json.load(open(kpi_path))
eps = d.get("episodes", [])
ticks = d.get("ticks", [])

print(f"{'='*80}")
print(f"MARL CONVERGENCE REPORT — {len(eps)} episodes, {len(ticks)} ticks")
print(f"{'='*80}")
print()
print(f"{'Ep':>3}  {'Scenario':<14}  {'Reward':>9}  {'P.Loss':>7}  {'V.Loss':>7}  "
      f"{'Entropy':>7}  {'Relay':>3}  {'UE%':>6}")
print("-" * 80)

for e in eps:
    print(f"{e['episode']:>3}  {e['scenario_type']:<14}  "
          f"{e['ep_reward_mean']:>9.1f}  "
          f"{e['final_policy_loss']:>7.4f}  "
          f"{e['final_value_loss']:>7.2f}  "
          f"{e['final_entropy']:>7.2f}  "
          f"{e['post_sev_transport_relay']:>3.0f}  "
          f"{e['post_sev_ue_conn']*100:>5.1f}%")

# Convergence analysis
print(f"\n{'='*80}")
print("CONVERGENCE ANALYSIS")
print(f"{'='*80}")

full_core_eps = [e for e in eps if e["scenario_type"] == "full_core"]
if len(full_core_eps) >= 5:
    last5 = full_core_eps[-5:]
    rewards = [e["ep_reward_mean"] for e in last5]
    ue_conns = [e["post_sev_ue_conn"] for e in last5]
    entropies = [e["final_entropy"] for e in last5]
    p_losses = [e["final_policy_loss"] for e in last5]
    v_losses = [e["final_value_loss"] for e in last5]

    import numpy as np
    r_mean, r_std = np.mean(rewards), np.std(rewards)
    ue_mean = np.mean(ue_conns) * 100
    ent_mean = np.mean(entropies)
    pl_mean = np.mean(p_losses)
    vl_mean = np.mean(v_losses)

    print(f"\nLast 5 full_core episodes:")
    print(f"  Reward:     mean={r_mean:.1f}  std={r_std:.1f}  "
          f"({'CONVERGED' if r_std < abs(r_mean)*0.05 else 'NOT converged'})")
    print(f"  UE conn:    mean={ue_mean:.1f}%  "
          f"({'GOOD' if ue_mean > 90 else 'NEEDS WORK'})")
    print(f"  Entropy:    mean={ent_mean:.2f}  "
          f"({'LOW (exploiting)' if ent_mean < 3.0 else 'HIGH (still exploring)'})")
    print(f"  Policy loss: mean={pl_mean:.4f}")
    print(f"  Value loss:  mean={vl_mean:.2f}")

    # Reward trend (linear regression)
    all_rewards = [e["ep_reward_mean"] for e in full_core_eps]
    x = np.arange(len(all_rewards))
    slope = np.polyfit(x, all_rewards, 1)[0]
    print(f"\n  Reward slope: {slope:.2f}/ep  "
          f"({'PLATEAU' if abs(slope) < 50 else 'STILL IMPROVING' if slope > 0 else 'DEGRADING'})")
else:
    print(f"Only {len(full_core_eps)} full_core episodes — need at least 5 for analysis")

print(f"\nPlots saved in: output/marl_convergence.png")
print(f"Life-safety:    output/life_safety_success.png")
