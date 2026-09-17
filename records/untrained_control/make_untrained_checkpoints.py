# -*- coding: utf-8 -*-
"""Untrained-actor control: write the shared actor as it stands BEFORE any
training update, for each replicate torch seed, in the same checkpoint format
as policy_final.pt, so that the unchanged frozen executor (marl_static) can
execute it.

The initialisation path is the one train_on_comparison.py runs before its
first episode: seed torch and random with --torch-seed, build the canonical
simulator on the first training seed (42) with --episode-ticks 800
(severance at tick 100), then construct MAPPOTrainer, which installs the
first agent's PolicyNetwork as the one shared actor.  No gradient step is
taken.  Run it twice and compare the printed fingerprints to confirm that the
construction is deterministic.

Usage (from ~/MARL_run_7, with the training environment variables set):
  python3 make_untrained_checkpoints.py 1234 2345 3456 4567
"""
import hashlib
import os
import random
import sys

import torch

from run_timeline_comparison import (build_comparison_topology,
                                     build_comparison_scenario)
from sixg_sim.simulation import SimulationConfig, Simulator
from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
from sixg_sim.agent import CriticNetwork, RLAgent, OBS_DIM
from train_on_comparison import parse_seed_spec

EPISODE_TICKS = 800
SEED_RANGE = "42-49,100-139"


def fingerprint(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initial_actor(torch_seed):
    torch.manual_seed(torch_seed)
    random.seed(torch_seed)
    from sixg_sim.reward_profile import PROFILE  # noqa: F401  (same import as training)
    seed_pool = parse_seed_spec(None, SEED_RANGE)
    canonical_seed = seed_pool[0]
    topology, _ = build_comparison_topology(canonical_seed, log_prefix="[INIT]")
    scenario = build_comparison_scenario('recovery', canonical_seed, topology=topology,
                                         ticks=EPISODE_TICKS, severance_tick=EPISODE_TICKS // 8)
    sim = Simulator(topology, scenario, SimulationConfig(enable_island_detection=True, verbose=False))
    rl_agents = {aid: a for aid, a in sim.agents.items() if isinstance(a, RLAgent)}
    critic = CriticNetwork(OBS_DIM)
    cfg = MAPPOConfig()
    trainer = MAPPOTrainer(rl_agents, critic, cfg)
    actor = trainer._ref_actor
    assert actor is not None
    assert all(a.policy_net is actor for a in rl_agents.values()), 'actor not shared'
    return {k: v.detach().clone() for k, v in actor.state_dict().items()}, canonical_seed, len(rl_agents)


def main():
    ref = torch.load('ce_ck_s1234/policy_final.pt' if os.path.exists('ce_ck_s1234/policy_final.pt')
                     else os.path.expanduser('~/MARL_run_6/ce_ck_s1234/policy_final.pt'),
                     map_location='cpu', weights_only=False)['state_dict']
    for s in [int(x) for x in sys.argv[1:]]:
        sd, canon, n = initial_actor(s)
        assert list(sd) == list(ref), 'key mismatch with the trained checkpoint'
        assert all(tuple(sd[k].shape) == tuple(ref[k].shape) for k in sd), 'shape mismatch'
        out_dir = 'untrained_s%d' % s
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'policy_init.pt')
        torch.save({'state_dict': sd, 'version': 0,
                    'metadata': {'episode': 0, 'env': 'comparison', 'episode_ticks': EPISODE_TICKS,
                                 'untrained': True, 'torch_seed': s, 'canonical_seed': canon}}, path)
        print('torch_seed %d  canonical_seed %d  agents %d  params-fingerprint %s  -> %s'
              % (s, canon, n, fingerprint(sd), path))


if __name__ == '__main__':
    main()
