"""
Train the MARL policy on the SAME topology/scenario family that
run_timeline_comparison.py uses for evaluation.

Why: the policy used to be trained on config/topology_unified.yaml (196
nodes) but evaluated on the comparison script's own code-built topology
family (~42 infra + 150 UEs, MultiHaul sites, severance + backhaul
degradation, rescue arrivals) — a train/eval domain mismatch.  This driver
imports build_comparison_topology() / build_comparison_scenario() from
run_timeline_comparison.py so training and evaluation share one generator.

Held-out evaluation seeds 50-54 are NEVER trained on (enforced).

FRAGMENTING SEVERANCE: because this driver builds every 'recovery' episode
through run_timeline_comparison.build_comparison_scenario(), training
recovery episodes automatically include the same fragmenting-severance
class the evaluation uses — sever_core PLUS seed-varied 'fail_link' cuts
(scenario_seed=event_seed, so the fragment pattern varies per episode)
that partition the surviving infra graph into 2-4 bridgeable fragments
with ~30-60% of UE-to-UE flows spanning fragments.  The policy therefore
trains on scenarios where relay-based fragment bridging is REQUIRED to
serve cross-fragment flows, not merely helpful.  The builder prints its
"[SCENARIO] fragments=K cross_fragment_flows=N/M" verification line in
each worker.  Scenario B ('rescue_ops') keeps its original structure.

Structure mirrors sixg_sim/main.py's fixed training loop:
  * parallel batches of up to --batch-size workers (ProcessPoolExecutor)
  * per-trajectory GAE computed inside each worker (MAPPOTrainer.finish_episode)
  * central MAPPO update between batches, weight broadcast to next batch
  * checkpoint saving every batch
  * frozen-policy eval + random-action baseline phases at the end

Usage (smoke):
    python train_on_comparison.py --episodes 2 --episode-ticks 100 --batch-size 2
Full run:
    python train_on_comparison.py --episodes 400
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import argparse
import json
import math
import time
import random
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Held-out seeds used by run_timeline_comparison.py main() — never train on these.
HELD_OUT_EVAL_SEEDS = {50, 51, 52, 53, 54}

# Scenario kinds matching the two evaluation scenarios:
#   'recovery'   — Scenario A: severance mid-episode + backhaul degradation
#   'rescue_ops' — Scenario B: severed at t=0, rescue UE arrival later
SCENARIO_KINDS = ('recovery', 'rescue_ops')


# ═══════════════════════════════════════════════════════════════════════
# Worker (module-level so it pickles under Windows 'spawn')
# ═══════════════════════════════════════════════════════════════════════

def _randomized_event_ticks(kind, ticks, rng):
    """Randomised per-episode event timing (training only).

    recovery:   severance in the first ~1/8..1/4 of the episode (leaves a
                long post-recovery steady state, unlike the old 300-tick
                episodes) — backhaul degradation at the severance tick,
                mirroring run_physics_pass.  The shared scenario builder
                additionally emits the fragmenting fail_link cuts at the
                severance tick (see module docstring).
    rescue_ops: severed at t=0 (island from the start), rescue arrives
                between 1/4 and 2/3 of the episode.
    """
    if kind == 'rescue_ops':
        sev = 0
        lo = max(60, ticks // 4)
        hi = max(lo + 1, (2 * ticks) // 3)
        rescue = rng.randint(lo, hi)
        return sev, rescue, 0  # severance, rescue, bh_degrade
    lo = max(30, ticks // 8)
    hi = max(lo + 1, ticks // 4)
    sev = rng.randint(lo, hi)
    return sev, None, sev


def _apply_backhaul_degradation(sim):
    """Replicates run_physics_pass's post-disaster microwave backhaul
    degradation (capacity to 30%) — a physical event, part of the
    evaluation environment, so training must see it too."""
    if getattr(sim, '_bh_degraded', False):
        return 0
    sim._bh_degraded = True
    degraded = 0
    for lid, link in sim.topology.links.items():
        if not link.is_up:
            continue
        lt_str = str(getattr(link, 'link_type', ''))
        if any(t in lt_str for t in ('Microwave', 'microwave', 'MICROWAVE')):
            link.capacity = max(50, link.capacity * 0.30)
            ep0, ep1 = link.endpoints
            if sim.topology.graph.has_edge(ep0, ep1):
                sim.topology.graph[ep0][ep1]['capacity'] = link.capacity
            degraded += 1
    if degraded:
        sim.topology.invalidate_infrastructure_cache()
    return degraded


def run_comparison_worker_episode(args_dict):
    """Run ONE training episode on the comparison-generator environment.

    Mirrors sixg_sim/worker.py:run_worker_episode exactly (same pending-
    transition collection contract, same per-trajectory GAE at episode end,
    same pool_data IPC format) — but builds topology + scenario through
    run_timeline_comparison's builders instead of the YAML topology, so the
    obs/action pipeline is byte-for-byte the deployment one.
    """
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    import copy
    import random as _random
    import torch
    torch.set_num_threads(1)

    # DETERMINISM.  Nothing in this training path seeded torch, so every run
    # drew a different action-sampling stream and a different critic init —
    # which makes a single-run A/B between two reward configurations
    # uninterpretable, because the run-to-run spread is confounded with the
    # treatment.  Seeding here (and in main(), and in run_eval_episode) makes
    # a run a pure function of --torch-seed, so two reward profiles can be
    # compared as a PAIRED difference on an identical random stream.
    _tseed = int(args_dict.get('torch_seed', 0))
    torch.manual_seed(_tseed)
    _random.seed(_tseed)
    try:
        import numpy as _np
        _np.random.seed(_tseed % (2 ** 32))
    except Exception:
        pass

    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario)
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
    from sixg_sim.agent import CriticNetwork, RLAgent, OBS_DIM, prb_training_action
    from sixg_sim.learning_tracker import LearningTracker

    episode_idx = args_dict['episode_idx']
    topo_seed = args_dict['topology_seed']
    kind = args_dict['scenario_kind']
    episode_ticks = args_dict['episode_ticks']
    event_seed = args_dict['event_seed']
    actor_sd = args_dict['actor_state_dict']
    critic_sd = args_dict['critic_state_dict']
    g_mix = args_dict['global_reward_mix']

    # ── Environment: SAME builders the evaluation uses ──
    topology, _info = build_comparison_topology(topo_seed)
    rng = _random.Random(event_seed)
    sev_tick, rescue_tick, bh_degrade_tick = _randomized_event_ticks(
        kind, episode_ticks, rng)
    scenario = build_comparison_scenario(
        kind, topo_seed, topology=topology, ticks=episode_ticks,
        severance_tick=sev_tick, rescue_tick=rescue_tick,
        scenario_seed=event_seed)

    config = SimulationConfig(enable_island_detection=True, verbose=False)
    sim = Simulator(topology, scenario, config)
    sim._current_scenario_type = getattr(scenario, 'scenario_type', kind)

    rl_agents = {aid: a for aid, a in sim.agents.items()
                 if isinstance(a, RLAgent)}
    critic_net = CriticNetwork(OBS_DIM)

    for aid, agent in rl_agents.items():
        agent.policy_net.load_state_dict(actor_sd)
        agent.device = torch.device('cpu')
        agent.policy_net.to(agent.device)
    critic_net.load_state_dict(critic_sd)
    critic_net.to(torch.device('cpu'))

    mappo_cfg = MAPPOConfig()
    mappo_trainer = MAPPOTrainer(rl_agents, critic_net, mappo_cfg)
    mappo_trainer.device = torch.device('cpu')
    mappo_trainer.critic_net.to(torch.device('cpu'))

    episode_rewards = []
    global_rewards = {}
    pending = {}
    tracker = LearningTracker()

    for tick in range(scenario.duration_ticks):
        sim.current_tick = tick
        sim._process_events(tick)

        # Physical backhaul degradation event (matches run_physics_pass)
        if tick == bh_degrade_tick:
            _apply_backhaul_degradation(sim)

        was_island = getattr(sim, 'island_mode', False)
        sim.island_mode = sim._detect_island_mode()
        sim.control_plane.set_island_mode(sim.island_mode)
        if sim.island_mode and not was_island:
            sim.marl_ue_routing_enabled = True

        sim._forward_traffic(sim._generate_traffic())
        observations = sim._build_agent_observations()

        obs_tensors = {}
        _aids = [aid for aid in sim.agents if aid in observations]
        _obs_stack = None
        if _aids:
            _obs_stack = torch.stack([
                sim.agents[aid].observation_to_tensor(observations[aid])
                for aid in _aids
            ])
            for i, aid in enumerate(_aids):
                obs_tensors[aid] = _obs_stack[i]

        # 1. Complete pending transitions with the reward observed now
        if pending:
            try:
                global_conn_r = sim.compute_global_connectivity_reward()
            except Exception:
                global_conn_r = 0.0
            for aid, tr in list(pending.items()):
                agent = sim.agents.get(aid)
                if (agent is None or not hasattr(agent, 'calculate_reward')
                        or aid not in observations):
                    continue
                reward_comps = agent.calculate_reward(
                    tr['obs_raw'], tr['action'], observations[aid])
                r = (1.0 - g_mix) * reward_comps.total_reward + g_mix * global_conn_r
                global_rewards[aid] = r
                mappo_trainer.collect(
                    aid, tr['obs_t'], tr['heads'], tr['prb'],
                    tr['log_prob'], tr['value'], r, done=False,
                    global_obs=tr['global_obs'],
                    # `tick` aligns agents for the counterfactual baseline in
                    # finish_episode(); it is the same value for every agent
                    # completing a transition on this tick.
                    step=tick,
                )
            pending = {}
            if global_rewards:
                episode_rewards.append(sum(global_rewards.values()))

        # 2. Act: sample a_t from pi(.|o_t)
        sim._execute_agent_actions(observations)

        # 3. Buffer (o_t, a_t, logpi, V) — reward arrives next tick
        if _aids:
            # CRITIC INPUT: unmasked on purpose.  The actor's tensor has the
            # non-local columns zeroed (agent.NONLOCAL_OBS_COLUMNS); the
            # critic is centralised and is entitled to the global state, so
            # it is rebuilt with mask_nonlocal=False.
            global_obs = torch.stack([
                sim.agents[aid].observation_to_tensor(
                    observations[aid], mask_nonlocal=False)
                for aid in _aids
            ]).mean(dim=0)
            value = mappo_trainer.get_value(list(obs_tensors.values()))
            _ref_agent = next(iter(sim.agents.values()))
            _temp = getattr(_ref_agent, '_logit_temperature', 1.0)

            _batch_aids, _batch_heads, _batch_prb = [], [], []
            for aid in _aids:
                agent = sim.agents[aid]
                act = agent.last_action
                if act is None or not hasattr(agent, 'calculate_reward'):
                    continue
                act_heads = {
                    'tx_power': act.tx_power_step,
                    'mcs_emrg': act.mcs_emergency_idx,
                    'mcs_gen': act.mcs_general_idx,
                    'relay': act.relay_mode_idx,
                    'handover': act.handover_idx,
                    'scheduler': act.scheduler_idx,
                    'postcard': int(act.send_postcard),
                    'iops': int(getattr(act, '_iops_decision', 0)),
                }
                # the SAMPLED split, not the executed one: the coordinator
                # clamp and the general-traffic floor are environment,
                # not policy, so they must not be scored as the action
                prb = prb_training_action(act)
                _batch_aids.append(aid)
                _batch_heads.append(act_heads)
                _batch_prb.append(prb)

            if _batch_aids:
                _batch_obs = [obs_tensors[aid] for aid in _batch_aids]
                try:
                    _log_probs = mappo_trainer.compute_log_prob_batch(
                        _batch_aids, torch.stack(_batch_obs), _batch_heads,
                        _batch_prb, temperature=_temp
                    )
                except Exception:
                    _log_probs = [
                        mappo_trainer.compute_log_prob(aid, obs, heads, prb,
                                                       temperature=_temp)
                        for aid, obs, heads, prb in zip(
                            _batch_aids, _batch_obs, _batch_heads, _batch_prb)
                    ]
                for j, aid in enumerate(_batch_aids):
                    pending[aid] = {
                        'obs_raw':    observations[aid],
                        'action':     sim.agents[aid].last_action,
                        'obs_t':      obs_tensors[aid],
                        'heads':      _batch_heads[j],
                        'prb':        _batch_prb[j],
                        'log_prob':   _log_probs[j],
                        'value':      value,
                        'global_obs': global_obs,
                    }

        _tick_kpis = sim.get_island_kpis()
        _mean_r = (sum(global_rewards.values()) / max(1, len(global_rewards))
                   if global_rewards else 0.0)
        tracker.record_tick(
            tick=tick, episode=episode_idx, island_mode=sim.island_mode,
            ue_conn_frac=_tick_kpis['ue_conn_frac'],
            transport_relay_count=_tick_kpis['transport_relay_count'],
            transport_link_count=_tick_kpis['transport_link_count'],
            reward=_mean_r, policy_loss=0.0, value_loss=0.0, entropy=0.0
        )

    avg_ep_reward = sum(episode_rewards) / max(1, len(episode_rewards))
    opt = sim.compute_optimal_connectivity()
    ep_rec = tracker.close_episode(
        episode_idx, scenario_type=f"cmp_{kind}")

    # Per-trajectory GAE inside the worker (single place GAE ever runs)
    mappo_trainer.finish_episode()

    pool_data = {
        'obs': [t.detach().cpu().numpy() for t in mappo_trainer._pool.obs],
        'global_obs': [t.detach().cpu().numpy()
                       for t in mappo_trainer._pool.global_obs],
        'actions': [copy.deepcopy(d) for d in mappo_trainer._pool.actions],
        'prb_acts': [copy.deepcopy(p) for p in mappo_trainer._pool.prb_acts],
        'log_probs': list(mappo_trainer._pool.log_probs),
        'advantages': list(mappo_trainer._pool.advantages),
        'returns': list(mappo_trainer._pool.returns),
    }

    return {
        'episode_idx': episode_idx,
        'topology_seed': topo_seed,
        'scenario_kind': kind,
        'severance_tick': sev_tick,
        'rescue_tick': rescue_tick,
        'avg_ep_reward': avg_ep_reward,
        'optimal_connectivity': opt,
        'ep_rec': ep_rec,
        'pool_data': pool_data,
    }


# ═══════════════════════════════════════════════════════════════════════
# In-process eval / baseline episode (frozen or random policy)
# ═══════════════════════════════════════════════════════════════════════

ALL_PINNABLE_HEADS = ('tx_power', 'mcs_emrg', 'mcs_gen', 'relay',
                      'handover', 'scheduler', 'postcard', 'iops', 'prb')


def run_eval_episode(actor_sd, topo_seed, kind, episode_ticks, event_seed,
                     random_actions=False, free_heads=None, torch_seed=None):
    """Run one no-learning episode; returns summary KPIs.

    actor_sd is loaded into every agent unless random_actions=True (then
    fresh randomly-initialised agents are used — the baseline arm).

    free_heads controls the DO-NOTHING / per-head isolation control arms
    (Simulator.pinned_action_heads, a measurement scaffold that is inert
    unless set — see _execute_agent_actions):
        None        — no pinning; the normal trained / random arm.
        ()          — DO-NOTHING: every action head pinned to its
                      PHYMACAction dataclass default (tx 0 dB, MCS follows
                      auto-CQI, relay OFF, no postcard, IOPS deny, PRB
                      0.30/0.00/0.70).  Nothing the policy emits reaches the
                      environment, so this is the "the mechanism, with no
                      agent" floor.
        ('relay',)  — every head pinned EXCEPT relay, i.e. exactly one knob
                      is returned to the policy.  This is the per-head
                      isolation used to check that no head is reward-positive
                      while KPI-negative.
    """
    import random as _random
    import torch

    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario)
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.agent import RLAgent

    # See the DETERMINISM note in run_comparison_worker_episode.  The trained
    # arm samples at temperature 0.05 (near-argmax) so it barely depends on
    # this, but the random-action baseline and the pinned-head control arms
    # depend on it entirely; seeding makes them reproducible across profiles.
    if torch_seed is not None:
        torch.manual_seed(int(torch_seed))
        _random.seed(int(torch_seed))

    topology, _ = build_comparison_topology(topo_seed)
    rng = _random.Random(event_seed)
    sev_tick, rescue_tick, bh_degrade_tick = _randomized_event_ticks(
        kind, episode_ticks, rng)
    scenario = build_comparison_scenario(
        kind, topo_seed, topology=topology, ticks=episode_ticks,
        severance_tick=sev_tick, rescue_tick=rescue_tick,
        scenario_seed=event_seed)

    config = SimulationConfig(enable_island_detection=True, verbose=False)
    sim = Simulator(topology, scenario, config)
    sim._current_scenario_type = getattr(scenario, 'scenario_type', kind)

    if free_heads is not None:
        sim.pinned_action_heads = set(ALL_PINNABLE_HEADS) - set(free_heads)

    for aid, agent in sim.agents.items():
        if not isinstance(agent, RLAgent):
            continue
        agent.device = torch.device('cpu')
        if random_actions:
            agent.is_training = True     # sample from untrained policy
        elif actor_sd is None:
            # Control arms with every head pinned (do-nothing / per-head
            # isolation): no checkpoint is needed because the pinned heads
            # never read the network.  The one FREE head samples from the
            # fresh (randomly initialised) net, which is the point — the
            # question is what that KNOB does to reward and KPI, not what a
            # trained policy would set it to.
            agent.is_training = True
        else:
            agent.policy_net.load_state_dict(actor_sd)
            agent.policy_net.to(agent.device)
            agent.is_training = False    # deterministic exploitation
            agent._logit_temperature = 0.05

    rewards = []
    ue_conn, relay_counts = [], []
    peak_relays = 0
    island_ticks = 0
    for tick in range(scenario.duration_ticks):
        sim.current_tick = tick
        sim._process_events(tick)
        if tick == bh_degrade_tick:
            _apply_backhaul_degradation(sim)
        was_island = sim.island_mode
        sim.island_mode = sim._detect_island_mode()
        sim.control_plane.set_island_mode(sim.island_mode)
        if sim.island_mode and not was_island:
            sim.marl_ue_routing_enabled = True
        sim._forward_traffic(sim._generate_traffic())
        observations = sim._build_agent_observations()
        sim._execute_agent_actions(observations)
        try:
            r = sim.compute_global_connectivity_reward()
        except Exception:
            r = 0.0
        rewards.append(r)
        k = sim.get_island_kpis()
        if sim.island_mode:
            island_ticks += 1
            ue_conn.append(k['ue_conn_frac'])
            relay_counts.append(k['transport_relay_count'])
            peak_relays = max(peak_relays, k['transport_relay_count'])

    return {
        'topology_seed': topo_seed,
        'scenario_kind': kind,
        'free_heads': list(free_heads) if free_heads is not None else None,
        'avg_reward': sum(rewards) / max(1, len(rewards)),
        'post_sev_ue_conn': sum(ue_conn) / max(1, len(ue_conn)),
        'post_sev_relay_mean': sum(relay_counts) / max(1, len(relay_counts)),
        'peak_relays': peak_relays,
        'island_ticks': island_ticks,
    }


# ═══════════════════════════════════════════════════════════════════════
# Seed handling
# ═══════════════════════════════════════════════════════════════════════

def parse_seed_spec(seed_list, seed_range):
    """Build the training seed pool. Default: 42-49 and 100-139."""
    seeds = []
    if seed_list:
        seeds = [int(s) for s in seed_list.replace(' ', '').split(',') if s]
    elif seed_range:
        for part in seed_range.replace(' ', '').split(','):
            if '-' in part:
                a, b = part.split('-')
                seeds.extend(range(int(a), int(b) + 1))
            elif part:
                seeds.append(int(part))
    else:
        seeds = list(range(42, 50)) + list(range(100, 140))

    bad = sorted(set(seeds) & HELD_OUT_EVAL_SEEDS)
    if bad:
        raise SystemExit(
            f"ERROR: training seeds {bad} overlap the held-out evaluation "
            f"seeds {sorted(HELD_OUT_EVAL_SEEDS)} used by "
            f"run_timeline_comparison.py — refusing to train on them.")
    if not seeds:
        raise SystemExit("ERROR: empty training seed pool")
    return seeds


# ═══════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train the MARL policy on the comparison-evaluation "
                    "topology/scenario generator (no domain mismatch).")
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--episode-ticks', type=int, default=800,
                        help="ticks per training episode (default 800 — long "
                             "enough to include post-recovery steady state)")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed-list', type=str, default=None,
                        help='comma-separated training topology seeds')
    parser.add_argument('--seed-range', type=str, default=None,
                        help='e.g. "42-49,100-139" (default)')
    parser.add_argument('--save-policy-path', type=str,
                        default=os.path.join('checkpoints',
                                             'policy_cloud_trained.pt'))
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--global-reward-mix', type=float, default=0.6)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--baseline-episodes', type=int, default=5)
    parser.add_argument('--base-seed', type=int, default=42,
                        help='base RNG seed for per-episode event timing')
    parser.add_argument('--torch-seed', type=int, default=1234,
                        help='seeds torch/random in the parent, in every '
                             'worker (torch_seed + episode index) and in the '
                             'eval/baseline episodes.  A run is a pure '
                             'function of this, so two reward profiles can '
                             'be compared as a paired difference.')
    parser.add_argument('--reward-profile', type=str, default=None,
                        help='ablation scaffold — sets MARL_REWARD_PROFILE '
                             'before sixg_sim is imported.  One of '
                             'prev/deliv/reprice/both/full (default: full, '
                             'the production configuration).')
    args = parser.parse_args()

    # MUST precede the sixg_sim import below: sixg_sim.reward_profile reads
    # the environment once, at import time.  Exported (not just set) so the
    # spawned worker processes inherit it.
    if args.reward_profile:
        os.environ['MARL_REWARD_PROFILE'] = args.reward_profile

    import torch
    import concurrent.futures
    from run_timeline_comparison import (build_comparison_topology,
                                         build_comparison_scenario)
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.mappo_trainer import (MAPPOTrainer, MAPPOConfig, ModelRegistry,
                                        DISCRETE_HEADS as TRAINED_HEADS)
    from sixg_sim.agent import (CriticNetwork, RLAgent, OBS_DIM,
                                UNTRAINED_HEADS)
    from sixg_sim.learning_tracker import LearningTracker

    torch.manual_seed(args.torch_seed)
    random.seed(args.torch_seed)
    from sixg_sim.reward_profile import PROFILE as _RP
    print(f"[REWARD] profile={_RP.name}  delivery={_RP.delivery_points:g}/"
          f"{_RP.gen_delivery_points:g}  relay_contrib={_RP.relay_contrib_per_node:g}"
          f"/{_RP.relay_contrib_points:g}  reunify={_RP.reunify_full_bonus:g}"
          f"/{_RP.reunify_per_fragment:g}  relay_tput_cap={_RP.relay_tput_cap_global:g}"
          f"/{_RP.relay_tput_cap:g}  bridge_held={_RP.bridge_held_points:g}  "
          f"cliff={_RP.routed_flow_cliff}  ledger={_RP.relay_terms_use_delivered_ledger}  "
          f"reach={_RP.reach_per_ue:g}/{_RP.reach_cap:g} "
          f"require_traffic={_RP.reach_require_traffic}")
    print(f"[SEED]   torch_seed={args.torch_seed}")

    seed_pool = parse_seed_spec(args.seed_list, args.seed_range)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_policy_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TRAIN ON COMPARISON-EVALUATION ENVIRONMENT")
    print(f"  Episodes: {args.episodes}  ticks/ep: {args.episode_ticks}  "
          f"batch: {args.batch_size}")
    print(f"  Training seeds ({len(seed_pool)}): {seed_pool[:12]}"
          f"{' ...' if len(seed_pool) > 12 else ''}")
    print(f"  Held-out eval seeds (excluded): {sorted(HELD_OUT_EVAL_SEEDS)}")
    print(f"  Scenario kinds: {SCENARIO_KINDS} (alternating, randomized "
          f"event timing)")
    print(f"  Policy -> {save_path}")
    print("=" * 70)

    # ── Canonical sim: instantiates the shared actor + critic ──
    canonical_seed = seed_pool[0]
    topology, topo_info = build_comparison_topology(
        canonical_seed, log_prefix="[TRAIN]")
    scenario = build_comparison_scenario(
        'recovery', canonical_seed, topology=topology,
        ticks=args.episode_ticks, severance_tick=args.episode_ticks // 8)
    config = SimulationConfig(enable_island_detection=True, verbose=False)
    canonical_sim = Simulator(topology, scenario, config)
    rl_agents = {aid: a for aid, a in canonical_sim.agents.items()
                 if isinstance(a, RLAgent)}
    if not rl_agents:
        raise SystemExit("ERROR: no RL agents in comparison topology")
    critic_net = CriticNetwork(OBS_DIM)

    # ── Size the shared rollout pool to the ACTUAL batch geometry ──────────
    # SharedRolloutPool is a deque(maxlen=capacity): every transition beyond
    # the capacity is silently dropped, oldest first.  Because workers extend
    # the pool as their futures complete, an undersized pool does not give a
    # uniform subsample of the batch — it keeps whichever episodes happened
    # to finish LAST.  Size it from batch_size x ticks x agents (+10 % head
    # room) so a batch is consumed whole.
    _batch_eps = max(1, min(args.batch_size, args.episodes))
    _needed = int(_batch_eps * args.episode_ticks * len(rl_agents) * 1.1)
    _cfg = MAPPOConfig()
    _cfg.pool_capacity = max(_cfg.mini_batch * 4, _needed)
    mappo_trainer = MAPPOTrainer(rl_agents, critic_net, _cfg)
    print(f"[TRAIN] {len(rl_agents)} RL agents, OBS_DIM={OBS_DIM}")
    print(f"[TRAIN] rollout pool capacity={_cfg.pool_capacity} "
          f"(batch {_batch_eps} eps x {args.episode_ticks} ticks x "
          f"{len(rl_agents)} agents)")
    print(f"[TRAIN] trained action heads: {TRAINED_HEADS} "
          f"(untrained, no simulator effect: {UNTRAINED_HEADS})")

    kpi_tracker = LearningTracker()
    tracker_path = os.path.join(str(output_dir), 'learning_kpis_comparison.json')
    registry = ModelRegistry(args.checkpoint_dir, 'comparison_shared')

    # ── Resume ──
    resume_ep = 0
    if args.resume and save_path.exists():
        try:
            saved = torch.load(str(save_path), map_location='cpu',
                               weights_only=False)
            sd = saved.get('state_dict', saved) if isinstance(saved, dict) else saved
            # Tolerates a 59-dim (pre-`relay_capable`) checkpoint: the new
            # input column is zero-padded, so the resume is a warm start
            # that has simply not learned to use the feature yet.
            from sixg_sim.agent import load_policy_state_dict
            if load_policy_state_dict(mappo_trainer._ref_actor, sd,
                                      strict=False):
                print("[Resume] NOTE: checkpoint predates the "
                      "`relay_capable` observation dim — fc1 input padded "
                      "with zeros; the feature starts at zero weight.")
            if isinstance(saved, dict):
                resume_ep = int(saved.get('metadata', {}).get('episode', 0))
            print(f"[Resume] Loaded {save_path} (ep={resume_ep})")
            # Resume picks up the run-progress schedule at 0 % (the resumed
            # run has its own --episodes budget); see the end-of-batch block.
            cfg = mappo_trainer.config
            cfg.entropy_coef = cfg.entropy_coef_start
            lr_a = 1e-4
            lr_c = 3e-4
            if mappo_trainer._actor_opt:
                for pg in mappo_trainer._actor_opt.param_groups:
                    pg['lr'] = lr_a
                for pg in mappo_trainer.critic_opt.param_groups:
                    pg['lr'] = lr_c
            print(f"[Resume] entropy={cfg.entropy_coef:.4f}  "
                  f"lr_a={lr_a:.2e}  lr_c={lr_c:.2e}")
        except Exception as e:
            print(f"[Resume] WARNING: could not load {save_path}: {e}")
    elif args.resume:
        print(f"[Resume] No checkpoint at {save_path} — starting fresh")

    # ── Episode plan: alternate scenario kinds, cycle topology seeds ──
    plan_rng = random.Random(args.base_seed * 9176 + 3)
    shuffled_pool = list(seed_pool)
    plan_rng.shuffle(shuffled_pool)

    def episode_spec(ep_idx):
        """(topology_seed, kind, event_seed) for global episode index."""
        topo_seed = shuffled_pool[(ep_idx - 1) % len(shuffled_pool)]
        kind = SCENARIO_KINDS[(ep_idx - 1) % len(SCENARIO_KINDS)]
        event_seed = args.base_seed * 100000 + ep_idx
        return topo_seed, kind, event_seed

    # ── Batch training loop (mirrors sixg_sim/main.py) ──
    train_summary = []
    total_eps = args.episodes
    batch_size = max(1, min(args.batch_size, total_eps))
    t_start = time.time()

    for batch_start in range(resume_ep, resume_ep + total_eps, batch_size):
        batch_end = min(batch_start + batch_size, resume_ep + total_eps)

        print(f"\n{'='*50}")
        print(f"--- Launching Batch {batch_start+1} to {batch_end} ---")
        print(f"{'='*50}")

        actor_sd = {k: v.cpu() for k, v in
                    mappo_trainer._ref_actor.state_dict().items()}
        critic_sd = {k: v.cpu() for k, v in
                     mappo_trainer.critic_net.state_dict().items()}

        avg_ep_reward = 0.0
        futures = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=batch_end - batch_start) as pool:
            for ep_idx in range(batch_start + 1, batch_end + 1):
                topo_seed, kind, event_seed = episode_spec(ep_idx)
                futures.append(pool.submit(run_comparison_worker_episode, {
                    'episode_idx': ep_idx,
                    'topology_seed': topo_seed,
                    'scenario_kind': kind,
                    'episode_ticks': args.episode_ticks,
                    'event_seed': event_seed,
                    'actor_state_dict': actor_sd,
                    'critic_state_dict': critic_sd,
                    'global_reward_mix': args.global_reward_mix,
                    'torch_seed': args.torch_seed + ep_idx,
                }))

            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    ep_num = res['episode_idx']
                    ep_rec = res['ep_rec']
                    pool_data = res['pool_data']
                    _opt = res['optimal_connectivity']
                    avg_ep_reward = res['avg_ep_reward']

                    obs_tensors = [torch.from_numpy(a).float()
                                   for a in pool_data['obs']]
                    gobs_tensors = [torch.from_numpy(a).float()
                                    for a in pool_data['global_obs']]
                    mappo_trainer._pool.obs.extend(obs_tensors)
                    mappo_trainer._pool.global_obs.extend(gobs_tensors)
                    mappo_trainer._pool.actions.extend(pool_data['actions'])
                    mappo_trainer._pool.prb_acts.extend(pool_data['prb_acts'])
                    mappo_trainer._pool.log_probs.extend(pool_data['log_probs'])
                    mappo_trainer._pool.advantages.extend(
                        pool_data['advantages'])
                    mappo_trainer._pool.returns.extend(pool_data['returns'])

                    kpi_tracker.episodes.append(ep_rec)
                    train_summary.append({
                        'episode': ep_num,
                        'seed': res['topology_seed'],
                        'kind': res['scenario_kind'],
                        'avg_reward': avg_ep_reward,
                        'post_sev_ue_conn': ep_rec.post_sev_ue_conn,
                        'post_sev_relays': ep_rec.post_sev_transport_relay,
                    })

                    print(f"  Worker Episode {ep_num} "
                          f"[seed={res['topology_seed']} "
                          f"kind={res['scenario_kind']} "
                          f"sev={res['severance_tick']}"
                          f"{' rescue=' + str(res['rescue_tick']) if res['rescue_tick'] else ''}]: "
                          f"avg_reward={avg_ep_reward:.4f}  "
                          f"post_sev_UE={ep_rec.post_sev_ue_conn:.1%}  "
                          f"transport_relays={ep_rec.post_sev_transport_relay:.1f}  "
                          f"peak_relays={ep_rec.peak_transport_relay}")
                    print(f"  [OPTIMAL] theoretical_max={_opt['optimal_frac']:.1%}  "
                          f"actual={_opt['actual_frac']:.1%}  "
                          f"efficiency={_opt['efficiency']:.1%}  "
                          f"({_opt['actual_flows']}/{_opt['optimal_flows']}/"
                          f"{_opt['total_flows']} actual/optimal/total flows, "
                          f"{_opt['n_components']} components)")
                except Exception as e:
                    print(f"Worker Error: {e}")

        # ── End-of-batch central MAPPO update ──
        print(f"--- End of Batch: Performing Global MAPPO Update ---")
        try:
            final_losses = mappo_trainer.update()
        except Exception as e:
            print(f"Global MAPPO Update Error: {e}")
            final_losses = {'value_loss': 0.0, 'policy_loss': 0.0,
                            'entropy': 0.0}
        pool_sz = final_losses.get('pool_size', 0)
        print(f"  [MAPPO] critic_loss={final_losses.get('value_loss', 0.0):.4f}  "
              f"policy_loss={final_losses.get('policy_loss', 0.0):.4f}  "
              f"entropy={final_losses.get('entropy', 0.0):.4f}  "
              f"pool={pool_sz}")

        # ── Entropy + LR decay, scheduled against RUN PROGRESS ────────────
        #
        # BUG FIXED HERE.  Both schedules were keyed to the ABSOLUTE episode
        # index with half-lives (50 episodes for the actor, 100 for the
        # critic) tuned for a short run, so on the 400-episode run they
        # annihilated learning long before it ended:
        #     ep  50 -> lr_actor 5.0e-05      ep 150 -> 1.2e-05
        #     ep 272 -> 2.3e-06               ep 400 -> 1e-06 (the floor)
        # i.e. by roughly episode 150 of 400 the actor step size was already
        # two orders of magnitude down and nothing could move afterwards.
        # The entropy coefficient, decaying at 0.985^ep, meanwhile stayed
        # HIGH (0.10 -> 0.047 over the first 50 episodes) exactly while the
        # LR was still alive, and only reached its floor around episode 300
        # once the LR was gone — so the run spent its whole usable budget
        # being pushed toward a uniform policy and then had no step size left
        # to exploit what it had seen.
        #
        # Both are now expressed as a fraction of the CONFIGURED RUN LENGTH,
        # so the same schedule shape applies whether the run is 24 episodes
        # or 400: the actor LR halves every quarter of the run (ending at
        # 1/16 of its initial value), the critic every third, and the entropy
        # coefficient reaches its floor at ~70 % of the run.
        cfg = mappo_trainer.config
        _done = (batch_end - resume_ep) / max(1.0, float(total_eps))
        cfg.entropy_coef = max(
            cfg.entropy_min,
            cfg.entropy_coef_start * (cfg.entropy_min / cfg.entropy_coef_start)
            ** min(1.0, _done / 0.70))
        lr_a = max(1e-6, 1e-4 * (0.5 ** (_done / 0.25)))
        lr_c = max(5e-6, 3e-4 * (0.5 ** (_done / 0.33)))
        if mappo_trainer._actor_opt is not None:
            for pg in mappo_trainer._actor_opt.param_groups:
                pg['lr'] = lr_a
            for pg in mappo_trainer.critic_opt.param_groups:
                pg['lr'] = lr_c
        print(f"  [MAPPO] entropy={cfg.entropy_coef:.4f}  "
              f"lr_actor={lr_a:.2e}  lr_critic={lr_c:.2e}")

        # ── Checkpoints ──
        try:
            registry.save(mappo_trainer._ref_actor,
                          metadata={'episode': batch_end,
                                    'avg_reward': avg_ep_reward,
                                    'env': 'comparison'})
            # Deployable policy in run_timeline_comparison's expected format
            # (legacy: {'state_dict': ...} — the int 'version' key keeps the
            # per-agent-format autodetection from misfiring).
            torch.save({
                'state_dict': mappo_trainer._ref_actor.state_dict(),
                'version': 0,
                'metadata': {'episode': batch_end,
                             'avg_reward': avg_ep_reward,
                             'env': 'comparison',
                             'episode_ticks': args.episode_ticks},
            }, str(save_path))
        except Exception as ckpt_err:
            print(f"  [CKPT] WARNING: checkpoint save failed: {ckpt_err}")

        kpi_tracker.save(tracker_path)

    train_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE ({total_eps} episodes in {train_time:.0f}s)  "
          f"policy -> {save_path}")
    print(f"{'='*70}")

    # ═══════════════════════════════════════════════════════════════════
    # EVALUATION PHASE — frozen policy on TRAINING-distribution seeds
    # ═══════════════════════════════════════════════════════════════════
    actor_sd = {k: v.cpu() for k, v in
                mappo_trainer._ref_actor.state_dict().items()}
    eval_results, baseline_results = [], []

    if args.eval_episodes > 0:
        print(f"\n{'='*70}")
        print(f"EVALUATION PHASE ({args.eval_episodes} episodes, frozen "
              f"policy, training-distribution seeds)")
        print(f"{'='*70}")
        for i in range(args.eval_episodes):
            topo_seed = seed_pool[i % len(seed_pool)]
            kind = SCENARIO_KINDS[i % len(SCENARIO_KINDS)]
            event_seed = args.base_seed * 100000 + 50000 + i
            r = run_eval_episode(actor_sd, topo_seed, kind,
                                 args.episode_ticks, event_seed,
                                 random_actions=False,
                                 torch_seed=args.torch_seed + 900000 + i)
            eval_results.append(r)
            print(f"  Eval {i+1}/{args.eval_episodes} "
                  f"[seed={topo_seed} kind={kind}]: "
                  f"reward={r['avg_reward']:.4f}  "
                  f"UE={r['post_sev_ue_conn']:.1%}  "
                  f"relays={r['post_sev_relay_mean']:.1f}  "
                  f"peak_relays={r['peak_relays']}")

    # ═══════════════════════════════════════════════════════════════════
    # BASELINE PHASE — random-action (untrained) agents, same seeds
    # ═══════════════════════════════════════════════════════════════════
    if args.baseline_episodes > 0:
        print(f"\n{'='*70}")
        print(f"BASELINE PHASE ({args.baseline_episodes} episodes, random "
              f"actions, same seeds as eval)")
        print(f"{'='*70}")
        for i in range(args.baseline_episodes):
            topo_seed = seed_pool[i % len(seed_pool)]
            kind = SCENARIO_KINDS[i % len(SCENARIO_KINDS)]
            event_seed = args.base_seed * 100000 + 50000 + i
            r = run_eval_episode(None, topo_seed, kind,
                                 args.episode_ticks, event_seed,
                                 random_actions=True,
                                 torch_seed=args.torch_seed + 900000 + i)
            baseline_results.append(r)
            print(f"  Baseline {i+1}/{args.baseline_episodes} "
                  f"[seed={topo_seed} kind={kind}]: "
                  f"reward={r['avg_reward']:.4f}  "
                  f"UE={r['post_sev_ue_conn']:.1%}  "
                  f"relays={r['post_sev_relay_mean']:.1f}  "
                  f"peak_relays={r['peak_relays']}")

    # ── JSON summary ──
    def _mean(rows, key):
        return (sum(x[key] for x in rows) / len(rows)) if rows else None

    summary = {
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'reward_profile': _RP.name,
        'reward_profile_values': dict(vars(_RP)) if hasattr(_RP, '__dict__')
                                 else {f.name: getattr(_RP, f.name)
                                       for f in __import__('dataclasses').fields(_RP)},
        'torch_seed': args.torch_seed,
        'environment': 'run_timeline_comparison builders '
                       '(train/eval domain-matched)',
        'topology_info': topo_info,
        'episodes': total_eps,
        'episode_ticks': args.episode_ticks,
        'batch_size': batch_size,
        'training_seeds': seed_pool,
        'held_out_eval_seeds': sorted(HELD_OUT_EVAL_SEEDS),
        'resume_from_episode': resume_ep,
        'train_time_s': round(train_time, 1),
        'policy_path': str(save_path),
        'train_last10_avg_reward': _mean(train_summary[-10:], 'avg_reward'),
        'eval': {
            'episodes': eval_results,
            'mean_reward': _mean(eval_results, 'avg_reward'),
            'mean_ue_conn': _mean(eval_results, 'post_sev_ue_conn'),
            'mean_relays': _mean(eval_results, 'post_sev_relay_mean'),
        },
        'baseline': {
            'episodes': baseline_results,
            'mean_reward': _mean(baseline_results, 'avg_reward'),
            'mean_ue_conn': _mean(baseline_results, 'post_sev_ue_conn'),
            'mean_relays': _mean(baseline_results, 'post_sev_relay_mean'),
        },
    }
    summary_path = output_dir / 'train_on_comparison_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] written -> {summary_path}")
    if eval_results and baseline_results:
        print(f"[SUMMARY] eval UE conn {summary['eval']['mean_ue_conn']:.1%} "
              f"vs baseline {summary['baseline']['mean_ue_conn']:.1%}  |  "
              f"eval reward {summary['eval']['mean_reward']:.4f} "
              f"vs baseline {summary['baseline']['mean_reward']:.4f}")
    print("DONE.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
