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
                            from sixg_sim.agent import load_policy_state_dict
                            load_policy_state_dict(
                                mappo_trainer._ref_actor,
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

                # ── Distributed Episode training loop ─────────────────────────────────
                import concurrent.futures
                from sixg_sim.worker import run_worker_episode
                
                batch_size = min(16, args.train_episodes)
                for batch_start in range(0, args.train_episodes, batch_size):
                    batch_end = min(batch_start + batch_size, args.train_episodes)
                    
                    print(f"\n{'='*50}")
                    print(f"--- Launching Batch {batch_start+1} to {batch_end} ---")
                    print(f"{'='*50}")

                    # 1. Gather state dicts
                    actor_sd = {k: v.cpu() for k, v in mappo_trainer._ref_actor.state_dict().items()}
                    critic_sd = {k: v.cpu() for k, v in mappo_trainer.critic_net.state_dict().items()}
                    
                    futures = []
                    # Note: no manual pool clear needed here — MAPPOTrainer.update()
                    # clears the pool at the end of every successful update.

                    avg_ep_reward = 0.0

                    with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as pool:
                        for ep_idx in range(batch_start + 1, batch_end + 1):
                            args_dict = {
                                'topology_path': str(topology_path),
                                'episode_idx': ep_idx,
                                'seed': args.seed + ep_idx,
                                'episode_ticks': getattr(args, 'episode_ticks', 800),
                                'curriculum_start': getattr(args, 'curriculum_start', 4),
                                'actor_state_dict': actor_sd,
                                'critic_state_dict': critic_sd,
                                'global_reward_mix': getattr(args, 'global_reward_mix', 0.6)
                            }
                            futures.append(pool.submit(run_worker_episode, args_dict))
                            
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                res = future.result()
                                ep_num = res['episode_idx']
                                ep_rec = res['ep_rec']
                                pool_data = res['pool_data']
                                _opt = res['optimal_connectivity']
                                avg_ep_reward = res['avg_ep_reward']
                                
                                # Combine PROCESSED data (advantages/returns were
                                # computed per-agent, per-episode in the worker, so
                                # merging worker pools in arbitrary order is safe —
                                # no GAE is ever run over this merged pool).
                                import torch
                                obs_tensors = [torch.from_numpy(arr).float() for arr in pool_data['obs']]
                                gobs_tensors = [torch.from_numpy(arr).float() for arr in pool_data['global_obs']]
                                mappo_trainer._pool.obs.extend(obs_tensors)
                                mappo_trainer._pool.global_obs.extend(gobs_tensors)
                                mappo_trainer._pool.actions.extend(pool_data['actions'])
                                mappo_trainer._pool.prb_acts.extend(pool_data['prb_acts'])
                                mappo_trainer._pool.log_probs.extend(pool_data['log_probs'])
                                mappo_trainer._pool.advantages.extend(pool_data['advantages'])
                                mappo_trainer._pool.returns.extend(pool_data['returns'])
                                
                                # Update KPIs natively
                                _kpi_tracker.episodes.append(ep_rec)
                                
                                print(f"  Worker Episode {ep_num}: avg_reward={avg_ep_reward:.4f}  "
                                      f"post_sev_UE={ep_rec.post_sev_ue_conn:.1%}  "
                                      f"transport_relays={ep_rec.post_sev_transport_relay:.1f}  "
                                      f"peak_relays={ep_rec.peak_transport_relay}")
                                print(f"  [OPTIMAL] theoretical_max={_opt['optimal_frac']:.1%}  "
                                      f"actual={_opt['actual_frac']:.1%}  "
                                      f"efficiency={_opt['efficiency']:.1%}  "
                                      f"({_opt['actual_flows']}/{_opt['optimal_flows']}/{_opt['total_flows']} "
                                      f"actual/optimal/total flows, {_opt['n_components']} components)")

                            except Exception as e:
                                print(f"Worker Error: {e}")

                    # 2. End of Batch MAPPO Update
                    print(f"--- End of Batch: Performing Global MAPPO Update ---")
                    try:
                        final_losses = mappo_trainer.update()
                    except Exception as e:
                        print(f"Global MAPPO Update Error: {e}")
                        final_losses = {'value_loss': 0.0, 'policy_loss': 0.0, 'entropy': 0.0}

                    # update() clears the pool afterwards — report the size it trained on
                    pool_sz = final_losses.get('pool_size', 0)
                    print(f"  [MAPPO] critic_loss={final_losses.get('value_loss', 0.0):.4f}  "
                          f"policy_loss={final_losses.get('policy_loss', 0.0):.4f}  "
                          f"entropy={final_losses.get('entropy', 0.0):.4f}  "
                          f"pool={pool_sz}")

                    # Entropy decay
                    _cfg = mappo_trainer.config
                    _cfg.entropy_coef = max(_cfg.entropy_min, _cfg.entropy_coef * _cfg.entropy_decay)

                    # LR decay
                    _base_lr_actor  = 1e-4
                    _base_lr_critic = 3e-4
                    _decayed_lr_actor  = max(1e-6, _base_lr_actor * (0.5 ** (batch_end / 50.0)))
                    _decayed_lr_critic = max(5e-6, _base_lr_critic * (0.5 ** (batch_end / 100.0)))
                    if mappo_trainer._actor_opt is not None:
                        for pg in mappo_trainer._actor_opt.param_groups:
                            pg['lr'] = _decayed_lr_actor
                        for pg in mappo_trainer.critic_opt.param_groups:
                            pg['lr'] = _decayed_lr_critic

                    print(f"  [MAPPO] entropy={_cfg.entropy_coef:.4f}  "
                          f"lr_actor={_decayed_lr_actor:.2e}  lr_critic={_decayed_lr_critic:.2e}")

                    if dashboard:
                        dashboard.add_training_metrics(
                            final_losses.get('value_loss', 0.0),
                            final_losses.get('policy_loss', 0.0),
                            avg_ep_reward
                        )
                        dashboard.render()

                    # Checkpoint at end of batch
                    try:
                        for aid, agent in rl_agents.items():
                            registries[aid].save(
                                agent.policy_net,
                                metadata={'episode': batch_end, 'avg_reward': avg_ep_reward}
                            )
                    except Exception as _ckpt_err:
                        print(f"  [CKPT] WARNING: checkpoint save failed: {_ckpt_err}")
                        
                    if live_dash is not None:
                        try:
                            conn_rate = ep_rec.post_sev_ue_conn if 'ep_rec' in locals() else 0.0
                            live_dash.push_episode_end(batch_end, avg_ep_reward, conn_rate)
                            
                            _reward_history = [r.ep_reward_mean for r in _kpi_tracker.episodes[-5:]]
                            _trend = ((_reward_history[-1] - _reward_history[0]) / max(1, len(_reward_history))
                                     ) if len(_reward_history) >= 2 else 0.0
                                     
                            live_dash.push_training_progress(
                                episode=batch_end,
                                total_episodes=args.train_episodes,
                                scenario_type=getattr(ep_scenario, 'scenario_type', 'full_core') if 'ep_scenario' in locals() else 'full_core',
                                metrics={
                                    "policy_loss":    final_losses.get('policy_loss') if mappo_trainer else None,
                                    "critic_loss":    final_losses.get('value_loss') if mappo_trainer else None,
                                    "entropy":        final_losses.get('entropy') if mappo_trainer else None,
                                    "avg_reward":     avg_ep_reward,
                                    "ue_connectivity": conn_rate,
                                    "relay_count":    ep_rec.post_sev_transport_relay if 'ep_rec' in locals() else 0,
                                    "peak_relays":    ep_rec.peak_transport_relay if 'ep_rec' in locals() else 0,
                                    "lr_actor":       _decayed_lr_actor if mappo_trainer else None,
                                    "lr_critic":      _decayed_lr_critic if mappo_trainer else None,
                                    "reward_trend":   _trend,
                                    "optimal_frac":   _opt['optimal_frac'] if '_opt' in locals() else 0.0,
                                    "efficiency":     _opt['efficiency'] if '_opt' in locals() else 0.0,
                                    "n_components":   _opt['n_components'] if '_opt' in locals() else 0,
                                }
                            )
                        except Exception as _dash_err:
                            print(f"  [Dashboard] push error: {_dash_err}")
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
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
