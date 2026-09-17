import os
import copy
import torch
import collections
from pathlib import Path
from typing import Dict, Any, List

def run_worker_episode(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Isolated worker function to run a single MARL episode.
    """
    import os
    # Strictly bind this process to 1 thread for numpy/OpenBLAS to prevent oversubscription
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    from sixg_sim.topology import load_topology_from_yaml, load_topology_from_json
    from sixg_sim.scenario import DiverseScenarioGenerator
    from sixg_sim.simulation import SimulationConfig, Simulator
    from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
    from sixg_sim.agent import CriticNetwork, RLAgent, OBS_DIM, prb_training_action
    from sixg_sim.learning_tracker import LearningTracker

    # Extract args
    topology_path = args_dict['topology_path']
    batch_size = args_dict.get('batch_size', 16)
    episode_idx = args_dict['episode_idx']
    is_leader = (episode_idx % batch_size == 1)
    
    seed = args_dict['seed']
    episode_ticks = args_dict['episode_ticks']
    curriculum_start = args_dict['curriculum_start']
    actor_sd = args_dict['actor_state_dict']
    critic_sd = args_dict['critic_state_dict']
    g_mix = args_dict['global_reward_mix']

    # Load Topology
    if topology_path.endswith('.json'):
        topology = load_topology_from_json(topology_path)
    else:
        topology = load_topology_from_yaml(topology_path)

    # Initialize Simulator
    generator = DiverseScenarioGenerator(topology, base_duration=episode_ticks, curriculum_start=curriculum_start)
    ep_scenario = generator.generate(episode=episode_idx, seed=seed)
    
    config = SimulationConfig(enable_island_detection=True, verbose=False)
    sim = Simulator(topology, ep_scenario, config)

    # ── Assign MultiHaul capability for training ──
    # Same logic as run_timeline_comparison.py — without this, agents
    # can't learn relay because CAPACITY_BOOST requires has_multihaul=True.
    relay_nodes = [nid for nid, n in topology.nodes.items()
                   if n.node_type.value in ('Relay', 'gNB-Site')]
    o_du_nodes = [nid for nid, n in topology.nodes.items()
                  if n.node_type.value == 'O-DU']
    # Top 40% relays get MultiHaul
    mh_relay_count = max(1, int(len(relay_nodes) * 0.40))
    for nid in relay_nodes[:mh_relay_count]:
        topology.nodes[nid].has_multihaul = True
    # Top 20% O-DUs get MultiHaul
    mh_du_count = max(1, int(len(o_du_nodes) * 0.20))
    for nid in o_du_nodes[:mh_du_count]:
        topology.nodes[nid].has_multihaul = True

    sim._current_scenario_type = ep_scenario.scenario_type

    # Initialize Agents and Network
    rl_agents = {aid: a for aid, a in sim.agents.items() if isinstance(a, RLAgent)}
    critic_net = CriticNetwork(OBS_DIM)

    # Load Weights
    for aid, agent in rl_agents.items():
        agent.policy_net.load_state_dict(actor_sd)
        # Ensure evaluation runs on CPU to avoid CUDA multiprocessing locks
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

    # Pending transitions: aid -> partial tuple buffered at the moment the
    # action was computed.  Completed with the reward that arrives next tick,
    # so the stored tuple is (o_t, a_t, logpi(a_t|o_t), V(s_t), r_t, done_t)
    # with log-prob and value evaluated on the SAME obs the action was
    # sampled from.
    pending = {}

    tracker = LearningTracker()

    for tick in range(ep_scenario.duration_ticks):
        sim.current_tick = tick
        sim._process_events(tick)
        was_island = getattr(sim, 'island_mode', False)
        sim.island_mode = sim._detect_island_mode()
        sim.control_plane.set_island_mode(sim.island_mode)

        if sim.island_mode and not was_island:
            sim.marl_ue_routing_enabled = True

        sim._forward_traffic(sim._generate_traffic())
        observations = sim._build_agent_observations()

        obs_tensors = {}

        # Build obs tensors
        _aids = [aid for aid in sim.agents if aid in observations]
        if _aids:
            _obs_stack = torch.stack([
                sim.agents[aid].observation_to_tensor(observations[aid])
                for aid in _aids
            ])
            for i, aid in enumerate(_aids):
                obs_tensors[aid] = _obs_stack[i]

        # ── 1. Complete pending transitions (o_{t-1}, a_{t-1}) with the
        #       reward observed now ─────────────────────────────────────────
        if pending:
            try:
                global_conn_r = sim.compute_global_connectivity_reward()
            except Exception:
                global_conn_r = 0.0

            for aid, tr in list(pending.items()):
                agent = sim.agents.get(aid)
                if (agent is None or not hasattr(agent, 'calculate_reward')
                        or aid not in observations):
                    continue   # agent left the sim — drop incomplete transition
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

        # Note: We DO NOT call mappo_trainer.update() here!
        # The update happens centrally in the main process.

        # ── 2. Act: samples a_t from pi(.|o_t) ───────────────────────────────
        sim._execute_agent_actions(observations)

        # ── 3. Buffer (o_t, a_t, logpi(a_t|o_t), V(s_t)) at the moment the
        #       action was computed; reward arrives next tick ────────────────
        if _aids:
            # Mean-field global observation (critic input) — stored per
            # transition so critic training uses the same input distribution.
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

            # Temperature actually used at sampling time (see simulation.py
            # batched sampling): behaviour policy is softmax(logits / temp).
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

        # Record KPI
        _tick_kpis = sim.get_island_kpis()
        _mean_r = (sum(global_rewards.values()) / max(1, len(global_rewards)) if global_rewards else 0.0)
        tracker.record_tick(
            tick=tick, episode=episode_idx, island_mode=sim.island_mode,
            ue_conn_frac=_tick_kpis['ue_conn_frac'],
            transport_relay_count=_tick_kpis['transport_relay_count'],
            transport_link_count=_tick_kpis['transport_link_count'],
            reward=_mean_r, policy_loss=0.0, value_loss=0.0, entropy=0.0
        )

        # Dashboard Live Streaming (Only for the first episode in the batch, 5 FPS)
        if is_leader and tick % 5 == 0:
            try:
                from sixg_sim.visualizer import build_snapshot
                import pickle
                import urllib.request
                
                snap = build_snapshot(
                    sim=sim,
                    tick=tick,
                    episode=episode_idx,
                    phase="Training",
                    policy_loss=0.0,
                    value_loss=0.0,
                    entropy=0.0,
                    episode_reward=_mean_r
                )
                
                req = urllib.request.Request(
                    "http://127.0.0.1:5050/internal_push",
                    data=pickle.dumps(snap),
                    headers={'Content-Type': 'application/octet-stream'}
                )
                with urllib.request.urlopen(req, timeout=0.1) as response:
                    pass
            except Exception:
                pass

    avg_ep_reward = sum(episode_rewards) / max(1, len(episode_rewards))
    opt = sim.compute_optimal_connectivity()
    ep_rec = tracker.close_episode(episode_idx, scenario_type=getattr(ep_scenario, 'scenario_type', 'full_core'))

    # Episode over: the very last sampled actions never receive a reward and
    # are dropped (still in `pending`).  Mark each agent's final COMPLETED
    # transition as terminal and compute GAE per-agent, per-episode inside
    # the trainer (this is the only place GAE runs — on correctly-ordered
    # single-agent trajectories).
    mappo_trainer.finish_episode()

    # Extract PROCESSED experiences (advantages/returns precomputed here in
    # the worker) from the pool.
    # We detach all tensors and convert to float32 to prevent memory leaks during IPC
    pool_data = {
        'obs': [t.detach().cpu().numpy() for t in mappo_trainer._pool.obs],
        'global_obs': [t.detach().cpu().numpy() for t in mappo_trainer._pool.global_obs],
        'actions': [copy.deepcopy(d) for d in mappo_trainer._pool.actions],
        'prb_acts': [copy.deepcopy(p) for p in mappo_trainer._pool.prb_acts],
        'log_probs': list(mappo_trainer._pool.log_probs),
        'advantages': list(mappo_trainer._pool.advantages),
        'returns': list(mappo_trainer._pool.returns),
    }

    return {
        'episode_idx': episode_idx,
        'avg_ep_reward': avg_ep_reward,
        'optimal_connectivity': opt,
        'ep_rec': ep_rec,
        'scenario_type': getattr(ep_scenario, 'scenario_type', 'full_core'),
        'pool_data': pool_data
    }
