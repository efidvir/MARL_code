"""
Command-line interface for the 6G network simulation.

Provides entry points for running simulations and analysis.
"""

import argparse
import sys
import os
import time
import math
import numpy as np
from pathlib import Path

# Force UTF-8 output on Windows so Unicode box-drawing chars / arrows don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7 — ignore


# Handle imports for both module execution and direct execution
try:
    # Try relative imports (works when run as module)
    from .simulation import run_simulation
    from .analysis import run_complete_analysis
except ImportError:
    try:
        # Try absolute imports (works when package is installed)
        from sixg_sim.simulation import run_simulation
        from sixg_sim.analysis import run_complete_analysis
    except ImportError:
        # Fallback: load modules directly (works when run as script)
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        # Import as if we're in the sixg_sim package
        import sixg_sim.simulation as simulation
        import sixg_sim.analysis as analysis
        run_simulation = simulation.run_simulation
        run_complete_analysis = analysis.run_complete_analysis


def main():
    """Main CLI entry point."""
    print("6G Network Simulation CLI starting...")

    parser = argparse.ArgumentParser(
        description="6G Network Simulation - Island Mode Proof of Concept"
    )

    parser.add_argument(
        "--topology", "-t",
        required=True,
        help="Path to topology configuration file (YAML/JSON)"
    )

    parser.add_argument(
        "--scenario", "-s",
        required=True,
        help="Path to scenario configuration file (YAML/JSON)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for results (default: output)"
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation (default: 42)"
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=0,
        help="Number of digital twin training episodes to run before live simulation"
    )

    parser.add_argument(
        "--save-policy-path",
        type=str,
        default=None,
        help="Path to save the pre-trained MARL policy after digital twin training"
    )

    parser.add_argument(
        "--load-policy-path",
        type=str,
        default=None,
        help="Path to load a pre-trained MARL policy from"
    )

    parser.add_argument(
        "--quiet-train",
        action="store_true",
        help="Disable the visual training dashboard during pre-training"
    )

    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Run live simulation in deployment mode (low-LR online fine-tuning + EWC)"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving/loading MAPPO policy checkpoints (default: checkpoints)"
    )

    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=0.4,
        help="EWC regularisation strength for online deployment (default: 0.4)"
    )

    parser.add_argument(
        "--animate",
        action="store_true",
        help="Record animated MP4 of the network + learning curves. Saved to output dir."
    )

    parser.add_argument(
        "--animate-fps",
        type=int,
        default=10,
        help="Frames per second for the animation (default: 10)"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Start the live browser dashboard at http://localhost:5050 during the run."
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="With --live: print the URL instead of auto-opening the browser."
    )

    parser.add_argument(
        "--mappo-update-interval",
        type=int,
        default=50,
        help="Run MAPPO PPO update every N ticks (default: 50). Longer = fewer updates, faster training."
    )

    parser.add_argument(
        "--global-reward-mix",
        type=float,
        default=0.6,
        help="Fraction of reward that comes from global connectivity (0-1, default: 0.6)."
    )

    parser.add_argument(
        "--reward-ue-island",
        type=float,
        default=10.0,
        help="UE-pair routing reward weight during island mode (default: 10.0)."
    )

    parser.add_argument(
        "--reward-relay-form",
        type=float,
        default=1.0,
        help="Reward per transport relay node activated post-severance (default: 1.0)."
    )

    parser.add_argument(
        "--curriculum-start",
        type=int,
        default=4,
        help="Episode number at which diverse scenario types start unlocking (default: 4)."
    )

    parser.add_argument(
        "--episode-ticks",
        type=int,
        default=800,
        help="Number of ticks per training episode (default: 800)."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest saved checkpoint (default: False)."
    )

    parser.add_argument(
        "--iops-mode",
        action="store_true",
        default=False,
        help="Enable Multi-eNB IOPS mode (ETSI TS 122 346). "
             "Adds IOPS pretraining episodes and activates peer learning."
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes with frozen trained policy (default: 5)."
    )

    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=5,
        help="Number of baseline episodes with random actions for comparison (default: 5)."
    )

    args = parser.parse_args()

    # Validate input files
    topology_path = Path(args.topology)
    scenario_path = Path(args.scenario)

    if not topology_path.exists():
        print(f"Error: Topology file not found: {topology_path}")
        sys.exit(1)

    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Arguments parsed successfully:")
        print(f"  topology: {args.topology}")
        print(f"  scenario: {args.scenario}")
        print(f"  output_dir: {args.output_dir}")

        if args.analyze_only:
            print("Analysis-only mode not yet implemented")
            # Would load existing metrics and run analysis
            sys.exit(1)
        else:
            # Handle imports
            try:
                from .topology import load_topology_from_yaml, load_topology_from_json
                from .scenario import DiverseScenarioGenerator, load_scenario_from_yaml
            except ImportError:
                import sixg_sim.topology as topology_module
                import sixg_sim.scenario as scenario_module
                load_topology_from_yaml = topology_module.load_topology_from_yaml
                load_topology_from_json = topology_module.load_topology_from_json
                DiverseScenarioGenerator = scenario_module.DiverseScenarioGenerator
                load_scenario_from_yaml  = scenario_module.load_scenario_from_yaml

            # ScenarioGenerator kept as alias (used below)
            ScenarioGenerator = DiverseScenarioGenerator

            def get_fresh_topology():
                if topology_path.suffix == '.json':
                    return load_topology_from_json(str(topology_path))
                else:
                    return load_topology_from_yaml(str(topology_path))

            topology = get_fresh_topology()

            # ── Visualizer init (outer scope — shared by train + online phases) ─────
            vis_animator = None
            vis_hook     = None
            if args.animate:
                from sixg_sim.visualizer import SimulationAnimator
                vis_animator = SimulationAnimator(
                    topology,
                    output_dir=str(output_dir),
                    fps=args.animate_fps,
                )
                vis_hook = vis_animator.make_hook()
                print(f"[Visualizer] Animation recording enabled (fps={args.animate_fps})")

            # ── Live browser dashboard ────────────────────────────────────────────────
            live_dash = None
            if args.live:
                from sixg_sim import live_dashboard as live_dash
                live_dash.start(open_browser=not args.no_browser)
                # Push topology layout so dashboard can draw the network graph
                live_dash.push_topology(topology)

            # Digital Twin Pre-Training Phase
            if args.train_episodes > 0:
                print(f"\n{'='*70}")
                print(f"STARTING DIGITAL TWIN PRE-TRAINING ({args.train_episodes} EPISODES)")
                print(f"{'='*70}")
                
                # ── Create MAPPO trainer with GAE + PPO clip ────────────────────────────
                from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig, ModelRegistry
                from sixg_sim.simulation import SimulationConfig, Simulator
                from sixg_sim.dashboard import TrainingDashboard
                from sixg_sim.agent import CriticNetwork, RLAgent, Experience
                from sixg_sim.learning_tracker import LearningTracker

                _kpi_tracker = LearningTracker()
                _tracker_path = os.path.join(str(output_dir), 'learning_kpis.json')

                _ep_ticks = getattr(args, 'episode_ticks', 800)
                generator = DiverseScenarioGenerator(
                    topology,
                    base_duration=_ep_ticks,
                    curriculum_start=getattr(args, 'curriculum_start', 4),
                )
                print(f"[Scenario] DiverseScenarioGenerator: base_duration={_ep_ticks}, "
                      f"curriculum_start={getattr(args, 'curriculum_start', 4)}")

                dashboard = None
                if not args.quiet_train:
                    dashboard = TrainingDashboard(max_points=args.train_episodes)

                canonical_topology = get_fresh_topology()
                canonical_config   = SimulationConfig(enable_island_detection=True, verbose=False)
                canonical_ep_scenario = generator.generate(episode=1, seed=args.seed)
                canonical_sim = Simulator(canonical_topology, canonical_ep_scenario, canonical_config)
                canonical_sim._current_scenario_type = canonical_ep_scenario.scenario_type

                rl_agents = {aid: a for aid, a in canonical_sim.agents.items()
                             if isinstance(a, RLAgent)}
                # Critic takes per-agent obs (59 dims) — not global concatenation.
                # Using per-agent size keeps the critic small and avoids the
                # (N_agents × 59) mismatch when agents enter/leave the island.
                from sixg_sim.agent import OBS_DIM as _OBS_DIM
                critic_net     = CriticNetwork(_OBS_DIM) if rl_agents else None

                mappo_cfg    = MAPPOConfig(ewc_lambda=args.ewc_lambda)
                mappo_trainer = MAPPOTrainer(rl_agents, critic_net, mappo_cfg) if critic_net else None

                # ── Multi-eNB IOPS mode (ETSI TS 122 346) ────────────────────
                _iops_mode = getattr(args, 'iops_mode', False)
                if _iops_mode:
                    _iops_pretrain = mappo_cfg.iops_pretraining_episodes
                    args.train_episodes += _iops_pretrain
                    print(f"[IOPS] Multi-eNB IOPS mode ACTIVE")
                    print(f"[IOPS] +{_iops_pretrain} pretraining episodes "
                          f"(total: {args.train_episodes})")
                    print(f"[IOPS] peer_learning_weight="
                          f"{mappo_cfg.peer_learning_weight}")

                registries   = {aid: ModelRegistry(args.checkpoint_dir, aid)
                                for aid in rl_agents}

                if args.load_policy_path:
                    for agent in rl_agents.values():
                        agent.load_policy(args.load_policy_path)

                # Auto-resume: load latest checkpoint for the shared policy network
                _resume_ep = 0
                if getattr(args, 'resume', False) and rl_agents and mappo_trainer:
                    import glob as _glob
                    import torch as _torch
                    _ckpts = sorted(
                        _glob.glob(f"{args.checkpoint_dir}/**/*.pt", recursive=True)
                        + _glob.glob(f"{args.checkpoint_dir}/*.pt"),
                        key=os.path.getmtime
                    )
                    if _ckpts:
                        _latest = _ckpts[-1]
                        try:
                            _saved = _torch.load(_latest, map_location='cpu')
                            # Load into the shared reference actor
                            mappo_trainer._ref_actor.load_state_dict(
                                _saved.get('state_dict', _saved), strict=False)
                            _resume_ep = int(_saved.get('metadata', {}).get('episode', 0))
                            print(f"[Resume] Loaded checkpoint: {_latest}  (ep={_resume_ep})")
                            # Restore entropy/LR schedule to where training left off
                            _cfg = mappo_trainer.config
                            _decayed_ent = max(_cfg.entropy_min,
                                0.10 * (_cfg.entropy_decay ** _resume_ep))
                            _cfg.entropy_coef = _decayed_ent
                            _decayed_lr_a = max(1e-6, 1e-4 * (0.5 ** (_resume_ep / 50.0)))
                            _decayed_lr_c = max(5e-6, 3e-4 * (0.5 ** (_resume_ep / 100.0)))
                            if mappo_trainer._actor_opt:
                                for pg in mappo_trainer._actor_opt.param_groups:
                                    pg['lr'] = _decayed_lr_a
                                for pg in mappo_trainer.critic_opt.param_groups:
                                    pg['lr'] = _decayed_lr_c
                            print(f"[Resume] entropy={_decayed_ent:.4f}  lr_a={_decayed_lr_a:.2e}  lr_c={_decayed_lr_c:.2e}")
                        except Exception as _e:
                            print(f"[Resume] WARNING: could not load {_latest}: {_e}")
                    else:
                        print("[Resume] No checkpoints found — starting fresh")

                # ── Episode training loop ──────────────────────────────────────────────
                for episode in range(args.train_episodes):
                    ep_num = episode + 1
                    ep_scenario = generator.generate(
                        episode=ep_num,
                        seed=args.seed + episode
                    )
                    print(f"\n--- Training Episode {ep_num}/{args.train_episodes} "
                          f"| scenario={ep_scenario.scenario_type} "
                          f"| {ep_scenario.description} ---")

                    
                    if dashboard:
                        scenario_info = {
                            'duration_ticks': ep_scenario.duration_ticks,
                            'events': [vars(e) for e in ep_scenario.events]
                        }
                        dashboard.update_scenario(episode + 1, args.train_episodes, scenario_info)
                        dashboard.render()
                    
                    config = SimulationConfig(enable_island_detection=True, verbose=False)
                    ep_topology = get_fresh_topology()

                    # Build sim for this episode but REPLACE its agents with our persistent ones
                    sim = Simulator(ep_topology, ep_scenario, config)
                    sim._current_scenario_type = ep_scenario.scenario_type
                    for aid, persistent_agent in rl_agents.items():
                        if aid in sim.agents:
                            sim.agents[aid] = persistent_agent
                    # Clear per-episode experience buffers so each episode trains on fresh data
                    for agent in rl_agents.values():
                        agent.experiences.clear()
                    # Per-episode state
                    prev_global_obs    = None
                    prev_global_action = {}
                    episode_rewards    = []
                    obs_tensors        = {}
                    _last_losses       = {}
                    # ── Tick loop ─────────────────────────────────────────────────────────
                    for tick in range(ep_scenario.duration_ticks):
                        sim.current_tick = tick
                        sim._process_events(tick)
                        was_island = sim.island_mode
                        sim.island_mode = sim._detect_island_mode()
                        sim.control_plane.set_island_mode(sim.island_mode)

                        if sim.island_mode and not was_island:
                            sim.marl_ue_routing_enabled = True

                        traffic_arrivals = sim._generate_traffic()
                        sim._forward_traffic(traffic_arrivals)
                        observations = sim._build_agent_observations()

                        # ── Collect obs tensors (batched) + record current actions ──
                        # _execute_agent_actions() will do the single batched forward;
                        # here we just build the obs_tensor dict (one tensor per agent)
                        # for MAPPO value/log-prob computation.
                        obs_tensors = {}
                        current_global_action = {}
                        try:
                            import torch as _torch
                            _aids = [aid for aid in sim.agents if aid in observations]
                            if _aids:
                                _obs_stack = _torch.stack([
                                    sim.agents[aid].observation_to_tensor(observations[aid])
                                    for aid in _aids
                                ])
                                for i, aid in enumerate(_aids):
                                    obs_tensors[aid] = _obs_stack[i]
                        except Exception:
                            for aid, agent in sim.agents.items():
                                if aid in observations and hasattr(agent, 'compute_action'):
                                    try:
                                        obs_tensors[aid] = agent.observation_to_tensor(observations[aid])
                                    except Exception:
                                        pass

                        # ── Store MAPPO experience + collect shaped reward ───────
                        if mappo_trainer and prev_global_obs is not None and prev_global_action:
                            global_rewards = {}
                            g_mix = getattr(args, 'global_reward_mix', 0.6)
                            try:
                                global_conn_r = sim.compute_global_connectivity_reward()
                            except Exception:
                                global_conn_r = 0.0

                            # ONE centralised critic call for all agents
                            value = mappo_trainer.get_value(list(obs_tensors.values()))

                            # Vectorised log_prob: batch all act_heads then call once
                            _batch_aids = []
                            _batch_obs  = []
                            _batch_heads = []
                            _batch_prb   = []
                            _batch_r     = []

                            # Parallel reward computation — each agent is independent
                            from concurrent.futures import ThreadPoolExecutor as _TPE

                            _eligible = [
                                (aid, agent)
                                for aid, agent in sim.agents.items()
                                if (hasattr(agent, 'calculate_reward') and
                                    aid in prev_global_obs and
                                    aid in prev_global_action and
                                    aid in observations and
                                    aid in obs_tensors)
                            ]

                            def _compute_one_reward(item):
                                aid, agent = item
                                prev_act = prev_global_action[aid]
                                reward_comps = agent.calculate_reward(
                                    prev_global_obs[aid], prev_global_action[aid], observations[aid]
                                )
                                local_r = reward_comps.total_reward
                                r = (1.0 - g_mix) * local_r + g_mix * global_conn_r
                                act_heads = {
                                    'tx_power':  prev_act.tx_power_step,
                                    'mcs_emrg':  prev_act.mcs_emergency_idx,
                                    'mcs_gen':   prev_act.mcs_general_idx,
                                    'relay':     prev_act.relay_mode_idx,
                                    'handover':  prev_act.handover_idx,
                                    'scheduler': prev_act.scheduler_idx,
                                    'postcard':  int(prev_act.send_postcard),
                                }
                                prb = [prev_act.prb_emergency_frac,
                                       prev_act.prb_relay_frac,
                                       prev_act.prb_general_frac]
                                return aid, r, act_heads, prb

                            _n_workers = min(8, max(1, len(_eligible)))
                            if _n_workers > 1:
                                with _TPE(max_workers=_n_workers) as _pool:
                                    _reward_results = list(_pool.map(_compute_one_reward, _eligible))
                            else:
                                _reward_results = [_compute_one_reward(x) for x in _eligible]

                            for aid, r, act_heads, prb in _reward_results:
                                global_rewards[aid] = r
                                _batch_aids.append(aid)
                                _batch_obs.append(obs_tensors[aid])
                                _batch_heads.append(act_heads)
                                _batch_prb.append(prb)
                                _batch_r.append(r)

                            # Batch log_prob for all agents in one forward pass
                            if _batch_aids:
                                try:
                                    import torch as _torch
                                    _log_probs = mappo_trainer.compute_log_prob_batch(
                                        _batch_aids, _torch.stack(_batch_obs), _batch_heads, _batch_prb
                                    )
                                except Exception:
                                    _log_probs = [
                                        mappo_trainer.compute_log_prob(aid, obs, heads, prb)
                                        for aid, obs, heads, prb in
                                        zip(_batch_aids, _batch_obs, _batch_heads, _batch_prb)
                                    ]
                                for j, aid in enumerate(_batch_aids):
                                    mappo_trainer.collect(
                                        aid, obs_tensors[aid],
                                        _batch_heads[j], _batch_prb[j],
                                        _log_probs[j], value, _batch_r[j], done=False
                                    )

                            if global_rewards:
                                episode_rewards.append(sum(global_rewards.values()))

                            # MAPPO update every 100 ticks (5× per episode)
                            _upd_interval = getattr(args, 'mappo_update_interval', 100)
                            if tick > 0 and tick % _upd_interval == 0 and mappo_trainer.buffer_size() >= 2048:
                                try:
                                    losses = mappo_trainer.update(
                                        next_global_obs=list(obs_tensors.values())
                                    )
                                    _last_losses = losses
                                    if live_dash:
                                        live_dash.push_state(
                                            __import__('sixg_sim.visualizer', fromlist=['build_snapshot'])
                                            .build_snapshot(
                                                sim, tick, episode+1, "train",
                                                policy_loss=float(losses.get('policy_loss', float('nan'))),
                                                value_loss =float(losses.get('value_loss',  float('nan'))),
                                                entropy    =float(losses.get('entropy',     float('nan'))),
                                                episode_reward=float(sum(global_rewards.values())/max(1,len(global_rewards))),
                                            )
                                        )
                                except Exception as ex:
                                    print(f"  [MAPPO] update error tick {tick}: {ex}")
                                else:
                                    _kpi_tracker.update_losses(
                                        float(losses.get('policy_loss', float('nan'))),
                                        float(losses.get('value_loss',  float('nan'))),
                                        float(losses.get('entropy',     float('nan'))),
                                    )

                        sim._execute_agent_actions(observations)
                        # Read actions computed by the batched forward in _execute_agent_actions
                        for aid, agent in sim.agents.items():
                            if aid in observations and agent.last_action is not None:
                                current_global_action[aid] = agent.last_action
                        prev_global_obs    = observations
                        prev_global_action = current_global_action

                        # ── LearningTracker: record per-tick KPIs ─────────────────
                        _tick_kpis = sim.get_island_kpis()
                        _mean_r = (
                            sum(global_rewards.values()) / max(1, len(global_rewards))
                            if 'global_rewards' in dir() and global_rewards else 0.0
                        )
                        _kpi_tracker.record_tick(
                            tick=tick,
                            episode=episode + 1,
                            island_mode=sim.island_mode,
                            ue_conn_frac=_tick_kpis['ue_conn_frac'],
                            transport_relay_count=_tick_kpis['transport_relay_count'],
                            transport_link_count=_tick_kpis['transport_link_count'],
                            reward=_mean_r,
                            policy_loss=float(_last_losses.get('policy_loss', float('nan'))),
                            value_loss =float(_last_losses.get('value_loss',  float('nan'))),
                            entropy    =float(_last_losses.get('entropy',     float('nan'))),
                        )

                        # ── Build snapshot for visualizer / live dashboard ────────
                        _do_snap = (vis_hook is not None or live_dash is not None) and tick % 2 == 0
                        if _do_snap:
                            from sixg_sim.visualizer import build_snapshot
                            _er = float(np.mean(episode_rewards[-5:])) if episode_rewards else 0.0
                            snap = build_snapshot(
                                sim, tick, episode+1, "train",
                                policy_loss=float(_last_losses.get('policy_loss', float('nan'))),
                                value_loss =float(_last_losses.get('value_loss',  float('nan'))),
                                entropy    =float(_last_losses.get('entropy',     float('nan'))),
                                episode_reward=_er
                            )
                            if vis_hook is not None:
                                vis_hook.on_tick(snap)
                            if live_dash is not None:
                                try:
                                    live_dash.push_state(snap)
                                except Exception:
                                    pass

                    # ── End of episode ──────────────────────────────────────────────
                    avg_ep_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
                    # Compute optimal (theoretical max) connectivity
                    _opt = sim.compute_optimal_connectivity()
                    ep_rec = _kpi_tracker.close_episode(
                        episode + 1,
                        scenario_type=getattr(ep_scenario, 'scenario_type', 'full_core')
                    )
                    print(f"  Episode {episode + 1}: avg_reward={avg_ep_reward:.4f}  "
                          f"post_sev_UE={ep_rec.post_sev_ue_conn:.1%}  "
                          f"transport_relays={ep_rec.post_sev_transport_relay:.1f}  "
                          f"peak_relays={ep_rec.peak_transport_relay}")
                    print(f"  [OPTIMAL] theoretical_max={_opt['optimal_frac']:.1%}  "
                          f"actual={_opt['actual_frac']:.1%}  "
                          f"efficiency={_opt['efficiency']:.1%}  "
                          f"({_opt['actual_flows']}/{_opt['optimal_flows']}/{_opt['total_flows']} "
                          f"actual/optimal/total flows, {_opt['n_components']} components)")

                    if mappo_trainer:
                        # Final MAPPO update at episode end
                        final_losses = mappo_trainer.update()
                        pool_sz = final_losses.get('pool_size', '?')
                        print(f"  [MAPPO] critic_loss={final_losses['value_loss']:.4f}  "
                              f"policy_loss={final_losses['policy_loss']:.4f}  "
                              f"entropy={final_losses['entropy']:.4f}  "
                              f"pool={pool_sz}")

                        # Entropy decay: MULTIPLICATIVE (×0.97/ep) with hard floor
                        _cfg = mappo_trainer.config
                        _cfg.entropy_coef = max(
                            _cfg.entropy_min,
                            _cfg.entropy_coef * _cfg.entropy_decay
                        )

                        # LR decay: actor decays faster than critic
                        _ep_num = episode + 1
                        _base_lr_actor  = 1e-4
                        _base_lr_critic = 3e-4
                        # Actor: halve every 50 episodes (gentle for 200 ep run)
                        _decayed_lr_actor  = max(1e-6, _base_lr_actor * (0.5 ** (_ep_num / 50.0)))
                        # Critic: halve every 100 episodes (very slow — let it converge)
                        _decayed_lr_critic = max(5e-6, _base_lr_critic * (0.5 ** (_ep_num / 100.0)))
                        if mappo_trainer._actor_opt is not None:
                            for pg in mappo_trainer._actor_opt.param_groups:
                                pg['lr'] = _decayed_lr_actor
                            for pg in mappo_trainer.critic_opt.param_groups:
                                pg['lr'] = _decayed_lr_critic
                        print(f"  [MAPPO] entropy={_cfg.entropy_coef:.4f}  "
                              f"lr_actor={_decayed_lr_actor:.2e}  lr_critic={_decayed_lr_critic:.2e}")

                        if dashboard:
                            dashboard.add_training_metrics(
                                final_losses['value_loss'],
                                final_losses['policy_loss'],
                                avg_ep_reward
                            )
                            dashboard.render()

                        # Save checkpoint after each episode (resilient to disk errors)
                        try:
                            for aid, agent in rl_agents.items():
                                registries[aid].save(
                                    agent.policy_net,
                                    metadata={'episode': episode, 'avg_reward': avg_ep_reward}
                                )
                        except Exception as _ckpt_err:
                            print(f"  [CKPT] WARNING: checkpoint save failed: {_ckpt_err}")

                        # Notify visualizer + live dashboard of episode end
                        conn_rate = float(getattr(sim, '_last_routed_flows', 0)) / max(1, float(getattr(sim, '_total_ue_flows', 1)))
                        if vis_hook is not None:
                            vis_hook.on_episode_end(episode + 1)
                        if live_dash is not None:
                            try:
                                live_dash.push_episode_end(episode + 1, avg_ep_reward, conn_rate)
                                # Push training progress with learning metrics
                                _reward_history = [r.ep_reward_mean for r in _kpi_tracker.episodes[-5:]]
                                _trend = ((_reward_history[-1] - _reward_history[0]) / max(1, len(_reward_history))
                                         ) if len(_reward_history) >= 2 else 0.0
                                live_dash.push_training_progress(
                                    episode=episode + 1,
                                    total_episodes=args.train_episodes,
                                    scenario_type=getattr(ep_scenario, 'scenario_type', 'full_core'),
                                    metrics={
                                        "policy_loss":    final_losses.get('policy_loss') if mappo_trainer else None,
                                        "critic_loss":    final_losses.get('value_loss') if mappo_trainer else None,
                                        "entropy":        final_losses.get('entropy') if mappo_trainer else None,
                                        "avg_reward":     avg_ep_reward,
                                        "ue_connectivity": ep_rec.post_sev_ue_conn,
                                        "relay_count":    ep_rec.post_sev_transport_relay,
                                        "peak_relays":    ep_rec.peak_transport_relay,
                                        "lr_actor":       _decayed_lr_actor if mappo_trainer else None,
                                        "lr_critic":      _decayed_lr_critic if mappo_trainer else None,
                                        "reward_trend":   _trend,
                                        "optimal_frac":   _opt['optimal_frac'],
                                        "efficiency":     _opt['efficiency'],
                                        "n_components":   _opt['n_components'],
                                    }
                                )
                            except Exception as _dash_err:
                                print(f"  [Dashboard] push error: {_dash_err}")
                            
                # ── Save LearningTracker + auto-generate convergence plots ─────────
                _kpi_tracker.save(_tracker_path)
                try:
                    from sixg_sim.plot_learning import plot_convergence
                    _plot_out = os.path.join(str(output_dir), 'marl_convergence.png')
                    plot_convergence(_tracker_path, _plot_out)
                    print(f"[KPI] Convergence plots saved -> {_plot_out}")
                except Exception as _pe:
                    print(f"[KPI] Plot generation skipped: {_pe}")

                # ── Anchor EWC after training finishes ───────────────────────────
                if mappo_trainer:
                    print("Computing EWC Fisher matrices (anchoring trained weights)...")
                    mappo_trainer.anchor_ewc()

                # Render training animation
                if vis_animator is not None:
                    print("[Visualizer] Rendering training animation...")
                    vis_animator.render_training(
                        os.path.join(str(output_dir), "training_animation.mp4"),
                        phase_filter="train"
                    )
                    vis_animator._export_convergence_summary()

                print(f"\n{'='*70}")
                print(f"DIGITAL TWIN PRE-TRAINING COMPLETE")
                print(f"{'='*70}\n")

                if dashboard:
                    time.sleep(2)
                    dashboard.close()
                
                if args.save_policy_path:
                    # Save one representative policy (all agents share weights in MAPPO)
                    for agent in rl_agents.values():
                        agent.save_policy(args.save_policy_path)
                        print(f"Saved optimized policy to {args.save_policy_path}")
                        break # Just save one copy since they share weights in MAPPO normally

                # ══════════════════════════════════════════════════════════════════════
                # EVALUATION PHASE — frozen trained policy, no exploration
                # ══════════════════════════════════════════════════════════════════════
                _eval_eps = getattr(args, 'eval_episodes', 5)
                if _eval_eps > 0 and rl_agents:
                    print(f"\n{'='*70}")
                    print(f"EVALUATION PHASE ({_eval_eps} episodes, frozen policy)")
                    print(f"{'='*70}")
                    import torch as _torch
                    # Freeze policy: disable gradients and set eval mode
                    for agent in rl_agents.values():
                        agent.policy_net.eval()
                        for p in agent.policy_net.parameters():
                            p.requires_grad_(False)

                    for eval_ep in range(_eval_eps):
                        ep_scenario = generator.generate(
                            episode=args.train_episodes + eval_ep + 1,
                            seed=args.seed + 10000 + eval_ep  # different seed space
                        )
                        print(f"\n--- Eval Episode {eval_ep+1}/{_eval_eps} "
                              f"| scenario={ep_scenario.scenario_type} ---")

                        config = SimulationConfig(enable_island_detection=True, verbose=False)
                        ep_topology = get_fresh_topology()
                        sim = Simulator(ep_topology, ep_scenario, config)
                        sim._current_scenario_type = ep_scenario.scenario_type
                        for aid, persistent_agent in rl_agents.items():
                            if aid in sim.agents:
                                persistent_agent.is_training = False  # Deterministic exploitation
                                sim.agents[aid] = persistent_agent

                        eval_rewards = []
                        _last_losses = {}
                        for tick in range(ep_scenario.duration_ticks):
                            sim.current_tick = tick
                            sim._process_events(tick)
                            was_island = sim.island_mode
                            sim.island_mode = sim._detect_island_mode()
                            sim.control_plane.set_island_mode(sim.island_mode)
                            if sim.island_mode and not was_island:
                                sim.marl_ue_routing_enabled = True
                            traffic_arrivals = sim._generate_traffic()
                            sim._forward_traffic(traffic_arrivals)
                            observations = sim._build_agent_observations()
                            sim._execute_agent_actions(observations)

                            # Compute reward (no gradient, no MAPPO update)
                            try:
                                global_conn_r = sim.compute_global_connectivity_reward()
                            except Exception:
                                global_conn_r = 0.0
                            eval_rewards.append(global_conn_r)

                            _tick_kpis = sim.get_island_kpis()
                            _kpi_tracker.record_tick(
                                tick=tick,
                                episode=args.train_episodes + eval_ep + 1,
                                island_mode=sim.island_mode,
                                ue_conn_frac=_tick_kpis['ue_conn_frac'],
                                transport_relay_count=_tick_kpis['transport_relay_count'],
                                transport_link_count=_tick_kpis['transport_link_count'],
                                reward=global_conn_r,
                            )

                        avg_eval_r = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0
                        ep_rec = _kpi_tracker.close_episode(
                            args.train_episodes + eval_ep + 1,
                            scenario_type=ep_scenario.scenario_type,
                            phase='eval'
                        )
                        print(f"  Eval {eval_ep+1}: reward={avg_eval_r:.4f}  "
                              f"UE={ep_rec.post_sev_ue_conn:.1%}  "
                              f"relays={ep_rec.post_sev_transport_relay:.1f}")

                    # Re-enable gradients for potential further use
                    for agent in rl_agents.values():
                        agent.policy_net.train()
                        for p in agent.policy_net.parameters():
                            p.requires_grad_(True)

                    print(f"\nEVALUATION PHASE COMPLETE")

                # ══════════════════════════════════════════════════════════════════════
                # BASELINE PHASE — random actions, no learning
                # ══════════════════════════════════════════════════════════════════════
                _base_eps = getattr(args, 'baseline_episodes', 5)
                if _base_eps > 0:
                    print(f"\n{'='*70}")
                    print(f"BASELINE PHASE ({_base_eps} episodes, random actions)")
                    print(f"{'='*70}")
                    import random as _rng_mod
                    import torch as _torch

                    # Create a random-action policy wrapper
                    class _RandomPolicyWrapper:
                        """Wraps an RLAgent to force random actions."""
                        def __init__(self, agent):
                            self._agent = agent
                            self._rng = _rng_mod.Random()
                        def __getattr__(self, name):
                            return getattr(self._agent, name)

                    for base_ep in range(_base_eps):
                        ep_scenario = generator.generate(
                            episode=args.train_episodes + _eval_eps + base_ep + 1,
                            seed=args.seed + 10000 + base_ep  # SAME seeds as eval for fair comparison
                        )
                        print(f"\n--- Baseline Episode {base_ep+1}/{_base_eps} "
                              f"| scenario={ep_scenario.scenario_type} ---")

                        config = SimulationConfig(enable_island_detection=True, verbose=False)
                        ep_topology = get_fresh_topology()
                        sim = Simulator(ep_topology, ep_scenario, config)
                        sim._current_scenario_type = ep_scenario.scenario_type
                        # DO NOT inject trained agents — use fresh default agents (random)
                        for agent in sim.agents.values():
                            agent.is_training = True  # Ensure they sample randomly

                        base_rewards = []
                        for tick in range(ep_scenario.duration_ticks):
                            sim.current_tick = tick
                            sim._process_events(tick)
                            was_island = sim.island_mode
                            sim.island_mode = sim._detect_island_mode()
                            sim.control_plane.set_island_mode(sim.island_mode)
                            if sim.island_mode and not was_island:
                                sim.marl_ue_routing_enabled = True
                            traffic_arrivals = sim._generate_traffic()
                            sim._forward_traffic(traffic_arrivals)
                            observations = sim._build_agent_observations()
                            sim._execute_agent_actions(observations)

                            try:
                                global_conn_r = sim.compute_global_connectivity_reward()
                            except Exception:
                                global_conn_r = 0.0
                            base_rewards.append(global_conn_r)

                            _tick_kpis = sim.get_island_kpis()
                            _kpi_tracker.record_tick(
                                tick=tick,
                                episode=args.train_episodes + _eval_eps + base_ep + 1,
                                island_mode=sim.island_mode,
                                ue_conn_frac=_tick_kpis['ue_conn_frac'],
                                transport_relay_count=_tick_kpis['transport_relay_count'],
                                transport_link_count=_tick_kpis['transport_link_count'],
                                reward=global_conn_r,
                            )

                        avg_base_r = sum(base_rewards) / len(base_rewards) if base_rewards else 0.0
                        ep_rec = _kpi_tracker.close_episode(
                            args.train_episodes + _eval_eps + base_ep + 1,
                            scenario_type=ep_scenario.scenario_type,
                            phase='baseline'
                        )
                        print(f"  Baseline {base_ep+1}: reward={avg_base_r:.4f}  "
                              f"UE={ep_rec.post_sev_ue_conn:.1%}  "
                              f"relays={ep_rec.post_sev_transport_relay:.1f}")

                    print(f"\nBASELINE PHASE COMPLETE")

                # ── Save final KPIs (includes train + eval + baseline) ─────────
                _kpi_tracker.save(_tracker_path)
                print(f"[KPI] Final KPIs saved (train + eval + baseline) -> {_tracker_path}")

                # ── Regenerate all 5 analysis plots ────────────────────────────
                try:
                    run_complete_analysis(output_dir=str(output_dir), kpi_json=_tracker_path)
                    print(f"[Analysis] All 5 plots regenerated in {output_dir}/")
                except Exception as _ae:
                    print(f"[Analysis] Plot regeneration failed: {_ae}")
            
            print("Starting 6G Network LIVE Simulation...")
            print(f"Topology: {topology_path}")
            print(f"Scenario: {scenario_path}")
            print(f"Output: {output_dir}")
            print()

            # Run simulation
            print("Running live simulation...")
            from sixg_sim.simulation import SimulationConfig, Simulator
            live_topology = get_fresh_topology()
            base_scenario = load_scenario_from_yaml(str(scenario_path), live_topology)
            
            live_config = SimulationConfig(
                enable_island_detection=True,
                verbose=True,
            )
            live_sim = Simulator(live_topology, base_scenario, live_config)
            
            # ── Load policy + set deploy mode if requested ───────────────────────
            if args.load_policy_path or args.save_policy_path:
                load_path = args.load_policy_path if args.load_policy_path else args.save_policy_path
                print(f"Loading pre-trained policy from {load_path} into Live edge agents...")
                for aid, agent in live_sim.agents.items():
                    if hasattr(agent, 'load_policy'):
                        agent.load_policy(load_path)

            if args.deploy:
                print("[DEPLOY] Setting deployment mode on all agents (low-LR + EWC).")
                for agent in live_sim.agents.values():
                    if hasattr(agent, 'set_deployment_mode'):
                        agent.set_deployment_mode(lr=3e-5)

            # ── Online: push fresh topology to dashboard ──────────────────────
            if live_dash is not None:
                try:
                    live_dash.push_topology(live_topology)
                except Exception:
                    pass

            # ── Online animation / live: custom tick loop ─────────────────────
            need_custom_loop = (args.animate and vis_hook is not None) or live_dash is not None
            if need_custom_loop:
                from sixg_sim.visualizer import build_snapshot
                if args.animate:
                    print("[Visualizer] Running instrumented live sim for online animation...")
                live_scenario = base_scenario
                for tick in range(live_scenario.duration_ticks):
                    live_sim.current_tick = tick
                    live_sim._process_events(tick)
                    was_i = live_sim.island_mode
                    live_sim.island_mode = live_sim._detect_island_mode()
                    live_sim.control_plane.set_island_mode(live_sim.island_mode)
                    if live_sim.island_mode and not was_i:
                        live_sim.marl_ue_routing_enabled = True
                    ta = live_sim._generate_traffic()
                    live_sim._forward_traffic(ta)
                    obs = live_sim._build_agent_observations()
                    live_sim._execute_agent_actions(obs)
                    if tick % 2 == 0:
                        snap = build_snapshot(live_sim, tick, 0, "online")
                        if vis_hook is not None:
                            vis_hook.on_tick(snap)
                        if live_dash is not None:
                            try:
                                live_dash.push_state(snap)
                            except Exception:
                                pass
                metrics = live_sim.metrics
            else:
                metrics = live_sim.run_simulation()

            # Save results
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # The exporter is built into the collector in the current version
            metrics.export_to_csv(str(output_dir))
            # exporter.export_to_json(metrics, f"metrics_{timestamp}") # Not implemented in Collector

            print("Simulation completed.")

            # Load topology for analysis (needed for node information)
            print("Loading topology for analysis...")
            
            # Run analysis
            print("Running analysis...")
            from sixg_sim.analysis import run_complete_analysis
            run_complete_analysis(metrics, topology.nodes, str(output_dir))
            print(f"Analysis completed. Check output in {output_dir}")

            # ── Render online animation ────────────────────────────────────
            if vis_animator is not None:
                print("[Visualizer] Rendering online animation...")
                vis_animator.render_online(
                    os.path.join(str(output_dir), "online_animation.mp4"),
                    phase_filter="online"
                )
                print(f"[Visualizer] All animations saved to {output_dir}/")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
