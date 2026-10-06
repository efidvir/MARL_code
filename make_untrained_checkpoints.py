# -*- coding: utf-8 -*-
"""Untrained-actor control: write the shared actor as it stands BEFORE any
training update, for each replicate torch seed, in the same checkpoint format
as policy_final.pt, so that the unchanged frozen executor (marl_static) can
execute it.

The initialisation path is the one train_on_comparison.py runs before its
first episode: seed torch and random with --torch-seed, build the canonical
simulator on the first training seed (42) at reward_spec.EPISODE_TICKS
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
import tempfile

import torch

from run_timeline_comparison import (build_comparison_topology,
                                     build_comparison_scenario)
from sixg_sim.simulation import SimulationConfig, Simulator
from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
from sixg_sim.agent import (CriticNetwork, PolicyNetwork, RLAgent,
                            CRITIC_OBS_DIM, SCHEMA_ID,
                            assert_checkpoint_schema, load_policy_state_dict,
                            mint_schema_current_checkpoint)
from train_on_comparison import parse_seed_spec
# The training-episode length is a PUBLISHED number (docs/REWARD_FROZEN.md).
# This file used to carry its own `EPISODE_TICKS = 800`, so the control
# checkpoints recorded an episode length nothing tied to the frozen one
# (D8.24).
from sixg_sim.reward_spec import EPISODE_TICKS

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
    from sixg_sim.reward_spec import FROZEN  # noqa: F401  (same import as training)
    seed_pool = parse_seed_spec(None, SEED_RANGE)
    canonical_seed = seed_pool[0]
    topology, _ = build_comparison_topology(canonical_seed, log_prefix="[INIT]")
    scenario = build_comparison_scenario('recovery', canonical_seed, topology=topology,
                                         ticks=EPISODE_TICKS, severance_tick=EPISODE_TICKS // 8)
    sim = Simulator(topology, scenario, SimulationConfig(enable_island_detection=True, verbose=False))
    rl_agents = {aid: a for aid, a in sim.agents.items() if isinstance(a, RLAgent)}
    critic = CriticNetwork(CRITIC_OBS_DIM)
    cfg = MAPPOConfig()
    trainer = MAPPOTrainer(rl_agents, critic, cfg)
    actor = trainer._ref_actor
    assert actor is not None
    assert all(a.policy_net is actor for a in rl_agents.values()), 'actor not shared'
    return {k: v.detach().clone() for k, v in actor.state_dict().items()}, canonical_seed, len(rl_agents)


def reference_state_dict(tmp_dir):
    """The INDEPENDENT side of the key/shape conformance check (D6.40).

    RERUN_SPEC WP6 removed the old reference -- the trained
    `ce_ck_s1234/policy_final.pt` actor -- because it was fitted on the
    62-dim masked schema and the strict gate refuses it (the WP6 schema
    drops INTERIOR columns, so adapting it is forbidden, not merely
    inconvenient).  Round 3 replaced it with `PolicyNetwork().state_dict()`,
    which makes the assertion compare a fresh network of this tree against
    another fresh network of this tree: near-tautological.

    The reference is now a checkpoint MINTED to disk by the shipped helper,
    stamped with this tree's schema id, read back, and put through the
    strict gate.  That is still a network of this tree -- only a retrained
    `ce_ck_*` replicate restores a genuinely independent reference -- but
    the comparison now exercises the save / stamp / gate / load round trip
    that the untrained control itself has to survive, instead of nothing.
    """
    ref_path = mint_schema_current_checkpoint(
        os.path.join(tmp_dir, 'reference_policy.pt'), torch_seed=0,
        extra={'purpose': 'make_untrained_checkpoints key/shape reference'})
    ref_ck = torch.load(ref_path, map_location='cpu', weights_only=False)
    assert_checkpoint_schema(ref_ck, where='untrained-control reference')
    load_policy_state_dict(PolicyNetwork(), ref_ck['state_dict'])
    return ref_ck['state_dict']


def main():
    from sixg_sim import runinfo
    runinfo.check_hashseed('make_untrained_checkpoints')
    with tempfile.TemporaryDirectory() as tmp_dir:
        ref = reference_state_dict(tmp_dir)
    print('schema %s (minted, gated reference checkpoint)' % SCHEMA_ID)
    for s in [int(x) for x in sys.argv[1:]]:
        sd, canon, n = initial_actor(s)
        assert list(sd) == list(ref), 'key mismatch with the minted reference'
        assert all(tuple(sd[k].shape) == tuple(ref[k].shape) for k in sd), 'shape mismatch'
        out_dir = 'untrained_s%d' % s
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'policy_init.pt')
        payload = {'state_dict': sd, 'version': 0,
                   'metadata': {'episode': 0, 'env': 'comparison', 'episode_ticks': EPISODE_TICKS,
                                'untrained': True, 'torch_seed': s, 'canonical_seed': canon,
                                'provenance': runinfo.provenance(extra={
                                    'driver': 'make_untrained_checkpoints',
                                    'torch_seed': s})}}
        # The control has to LOAD under the same strict gate the evaluation
        # driver runs, so it is checked here rather than discovered at
        # evaluation time (RERUN_SPEC WP7, untrained control).
        assert_checkpoint_schema(payload, where='untrained control %s' % path)
        load_policy_state_dict(PolicyNetwork(), sd)
        torch.save(payload, path)
        print('torch_seed %d  canonical_seed %d  agents %d  params-fingerprint %s  -> %s'
              % (s, canon, n, fingerprint(sd), path))


if __name__ == '__main__':
    main()
