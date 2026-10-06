# -*- coding: utf-8 -*-
"""WP6-FIX3 -- PPO log-prob consistency, restored.

WP6 rewrote `test_relay_mask.py` from four contract checks to three and, in
doing so, deleted its check 3:

    PPO consistency: the collection-time log-prob of a sampled batch equals
    the update-path log-prob of the same (obs, action) under the same
    temperature -- i.e. pi_old and pi_new describe the SAME distribution, so
    the PPO ratio at epoch start is exactly 1.

That invariant is not about the relay mask at all.  It is about
`MAPPOTrainer.compute_log_prob_batch` agreeing with the recomputation inside
`MAPPOTrainer.update` -- precisely the thing a 62 -> 26 observation-schema
change and an actor/critic split can break, and after the rewrite nothing in
the tree covered it (`grep -rn compute_log_prob tests/` returned nothing).
This file is its mask-free home.

If the two ever disagree, every PPO ratio starts away from 1 and the clip
does not mean what it says -- a failure that produces plausible-looking
training curves rather than an error.

The check compares `compute_log_prob_batch` against a hand-written copy of
the update path, so check 4 is a TRIPWIRE on that copy: it reads the source
of `MAPPOTrainer.update` and fails if the block the helper mirrors stops
matching it.  Without that, the helper and collection could keep agreeing
with each other while the real update path drifted away from both.

Fast: no simulator pass, no checkpoint, one forward pass per check.  Run

    PYTHONHASHSEED=0 python3 tests/run_tests.py wp6_ppo_logprob
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

N_AGENTS = 16
TORCH_SEED = 20260923


def _batch(n=N_AGENTS, seed=TORCH_SEED):
    """n agents sharing ONE actor, a random observation batch, and a batch of
    actions sampled from that actor at temperature 1.0 -- the collection path
    in miniature.  Returns (trainer, ids, obs_batch, actions, prbs)."""
    import torch
    from torch.distributions import Categorical

    from sixg_sim.agent import (ACTOR_OBS_DIM, DISCRETE_HEADS, RLAgent,
                                apply_relay_mask, prb_distribution)
    from sixg_sim.mappo_trainer import MAPPOTrainer
    from sixg_sim.traffic import SliceDictionary

    torch.manual_seed(seed)
    ids = ['site_%d' % i for i in range(n)]
    agents = {aid: RLAgent(aid, SliceDictionary(), is_training=True)
              for aid in ids}
    ref = agents[ids[0]]
    for a in agents.values():                 # ONE shared actor, as in training
        a.policy_net = ref.policy_net

    obs_batch = torch.randn(n, ACTOR_OBS_DIM)

    trainer = MAPPOTrainer.__new__(MAPPOTrainer)   # no full init needed
    trainer.agents = agents
    trainer.device = torch.device('cpu')
    trainer._ref_actor = ref.policy_net

    with torch.no_grad():
        logits = apply_relay_mask(ref.policy_net(obs_batch), obs_batch)
    actions, prbs = [], []
    for k in range(n):
        row = {h: int(Categorical(logits=logits[h][k]).sample())
               for h in DISCRETE_HEADS if h in logits}
        actions.append(row)
        prbs.append(prb_distribution(logits['prb'][k]).sample().tolist())
    return trainer, ids, obs_batch, actions, prbs


def _update_path_log_probs(trainer, obs_batch, actions, prbs):
    """Recompute the log-probs exactly as MAPPOTrainer.update does at the
    start of epoch 0: one forward through `_ref_actor`, a Categorical per
    discrete head at temperature 1, plus the PRB density."""
    import torch
    from torch.distributions import Categorical

    from sixg_sim.agent import apply_relay_mask
    from sixg_sim.mappo_trainer import DISCRETE_HEADS, _prb_log_prob

    logits = apply_relay_mask(trainer._ref_actor(obs_batch), obs_batch)
    lp = torch.zeros(len(actions))
    for h in DISCRETE_HEADS:
        if h in logits:
            idx = torch.tensor([a.get(h, 0) for a in actions],
                               dtype=torch.long)
            lp = lp + Categorical(logits=logits[h]).log_prob(idx)
    if 'prb' in logits:
        lp = lp + _prb_log_prob(logits['prb'],
                                torch.tensor(prbs, dtype=torch.float32))
    return lp


def test_collection_log_prob_equals_the_update_path_recomputation():
    """The restored check: exp(new_lp - old_lp) == 1 at epoch start."""
    import torch

    trainer, ids, obs_batch, actions, prbs = _batch()
    old_lp = trainer.compute_log_prob_batch(ids, obs_batch, actions, prbs,
                                            temperature=1.0)
    assert len(old_lp) == len(ids)
    # grad-enabled, exactly as MAPPOTrainer.update computes it; detached
    # only for the comparison
    new_lp = _update_path_log_probs(trainer, obs_batch, actions, prbs).detach()

    gap = float((new_lp - torch.tensor(old_lp)).abs().max())
    ratio = torch.exp(new_lp - torch.tensor(old_lp))
    assert gap < 1e-4, (
        'collection and update log-probs disagree by %.3e: pi_old and pi_new '
        'are not the same distribution, so every PPO ratio starts away from '
        '1 and the clip does not mean what it says' % gap)
    assert float((ratio - 1.0).abs().max()) < 1e-4, ratio.tolist()

    # ... and no stored action carries a masked / degenerate log-prob
    assert all(v > -1e6 for v in old_lp), old_lp
    # ... on the WP6 26-wide actor vector, not some other width
    from sixg_sim.agent import ACTOR_OBS_DIM
    assert obs_batch.shape[1] == ACTOR_OBS_DIM == 26, obs_batch.shape


def test_the_per_agent_fallback_agrees_with_the_batch_form():
    """The drivers fall back to `compute_log_prob` per agent when the batch
    form raises.  The fallback must score the same distribution, or a single
    failing tick would silently change the behaviour policy it records."""
    trainer, ids, obs_batch, actions, prbs = _batch(n=8)
    batch_lp = trainer.compute_log_prob_batch(ids, obs_batch, actions, prbs,
                                              temperature=1.0)
    single_lp = [trainer.compute_log_prob(aid, obs_batch[i], actions[i],
                                          prbs[i], temperature=1.0)
                 for i, aid in enumerate(ids)]
    gap = max(abs(a - b) for a, b in zip(batch_lp, single_lp))
    assert gap < 1e-4, ('the batch and per-agent log-prob forms disagree by '
                        '%.3e' % gap)
    assert not all(v == 0.0 for v in single_lp), (
        'the per-agent fallback returned all zeros -- its own except clause '
        'swallowed a real failure')


def test_compute_log_prob_batch_raises_instead_of_returning_zeros():
    """WP6-FIX3 defect 8.  The blanket `except Exception: return [0.0] * N`
    is gone: a zeroed OLD log-prob turns every PPO ratio into
    exp(new_logp - 0), which is worse than the failure it hides, and it made
    the drivers' documented per-agent fallback unreachable."""
    import torch

    trainer, ids, obs_batch, actions, prbs = _batch(n=4)
    bad = torch.randn(len(ids), obs_batch.shape[1] + 3)   # wrong actor width
    raised = None
    try:
        trainer.compute_log_prob_batch(ids, bad, actions, prbs,
                                       temperature=1.0)
    except Exception as e:                                # noqa: BLE001
        raised = e
    assert raised is not None, (
        'compute_log_prob_batch swallowed a wrong-width observation batch '
        'and returned log-probs anyway')

    src_path = os.path.join(ROOT, 'sixg_sim', 'mappo_trainer.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    body = src.split('def compute_log_prob_batch(')[1].split('\n    def ')[0]
    # CODE lines only: this function's comments describe the removed except
    # by name, and a scan that matched them would fail on its own prose.
    code = [l for l in body.splitlines()
            if l.strip() and not l.strip().startswith('#')]
    assert not any(l.strip().startswith('except ') for l in code), (
        'a blanket except is back in compute_log_prob_batch: %r'
        % [l.strip() for l in code if l.strip().startswith('except ')])
    head = '\n'.join(code).split('temp = max(')[0]
    assert 'if not agent_ids:' in head, (
        'the empty-input guard above the body was removed too')
    assert '[0.0] * len(agent_ids)' not in head, (
        'the unknown-agent guard still returns a vector of zeroed old '
        'log-probs, which is the same silent failure the except clause was')


def test_the_per_agent_form_raises_too():
    """WP7 Part A defect 1 (MEDIUM).  Removing the batch form's blanket
    except changed NOTHING at any driver call site, because the per-agent
    `compute_log_prob` -- which is what all three drivers run on that raise
    -- ended in `except Exception: return 0.0`.  The driver-visible outcome
    was byte-identical to before the round-3 fix: pi_old = 0.0 for every
    agent on the failing tick.  Both forms must now raise on the same
    input, or the collection path still has a silent zero in it."""
    import torch

    trainer, ids, obs_batch, actions, prbs = _batch(n=4)
    bad = torch.randn(len(ids), obs_batch.shape[1] + 3)   # wrong actor width

    batch_raised = None
    try:
        trainer.compute_log_prob_batch(ids, bad, actions, prbs,
                                       temperature=1.0)
    except Exception as e:                                # noqa: BLE001
        batch_raised = e
    assert batch_raised is not None, 'the batch form swallowed it'

    single_raised = None
    try:
        trainer.compute_log_prob(ids[0], bad[0], actions[0], prbs[0],
                                 temperature=1.0)
    except Exception as e:                                # noqa: BLE001
        single_raised = e
    assert single_raised is not None, (
        'compute_log_prob -- the PER-AGENT fallback every driver runs when '
        'the batch form raises -- swallowed a wrong-width observation and '
        'returned a log-prob anyway, so the batch-form fix has no effect at '
        'any call site')

    # an unknown agent id is the other zeroed-log-prob route out of this
    # function, and it must not be one either
    unknown_raised = None
    try:
        trainer.compute_log_prob('no-such-agent', obs_batch[0], actions[0],
                                 prbs[0], temperature=1.0)
    except Exception as e:                                # noqa: BLE001
        unknown_raised = e
    assert unknown_raised is not None, (
        'compute_log_prob returned 0.0 for an unknown agent id')

    # and the source carries no `except` at all in that function's body
    src_path = os.path.join(ROOT, 'sixg_sim', 'mappo_trainer.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    body = src.split('def compute_log_prob(')[1].split('\n    def ')[0]
    code = [l for l in body.splitlines()
            if l.strip() and not l.strip().startswith('#')]
    assert not any(l.strip().startswith('except ') for l in code), (
        'an except clause is back in compute_log_prob: %r'
        % [l.strip() for l in code if l.strip().startswith('except ')])
    assert not any(l.strip() == 'try:' for l in code), (
        'a try block is back in compute_log_prob')
    assert not any(l.strip() == 'return 0.0' for l in code), (
        'compute_log_prob still has a `return 0.0` escape hatch: %r'
        % [l.strip() for l in code if l.strip() == 'return 0.0'])


# ── 4. TRIPWIRE: the helper above still IS the real update path ──────────
# `_update_path_log_probs` is a hand-written copy of the new-log-prob block
# inside `MAPPOTrainer.update`, which is the weak point of this restored
# check (the deleted original had the same shape): if the real update path
# gains a temperature, swaps the PRB density or adds a head to the loss, the
# helper would keep agreeing with collection and this file would stay green
# while pi_old and pi_new diverged.  So the three ingredients are asserted
# against the SOURCE of `update`.  Driving a real minibatch through
# `update` instead would need a fully-built trainer, a populated pool and an
# optimiser step -- seconds of compute for the same statement.
UPDATE_INGREDIENTS = (
    # one masked forward through the reference actor ...
    'apply_relay_mask(self._ref_actor(mb_obs), mb_obs)',
    # ... a Categorical per discrete head, summed ...
    'Categorical(logits=logits[h])',
    'new_lp  = new_lp + dist.log_prob(head_tensors[h][mb])',
    # ... plus the PRB log-density, on the same logits
    "new_lp = new_lp + _prb_log_prob(logits['prb'], mb_prb)",
)


def _update_body():
    path = os.path.join(ROOT, 'sixg_sim', 'mappo_trainer.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    body = src.split('def update(self, next_global_obs')[1]
    return body.split('\n    def ')[0]


def test_the_update_path_helper_still_mirrors_mappo_trainer_update():
    """If this fails, `_update_path_log_probs` above no longer describes
    what `MAPPOTrainer.update` actually computes -- fix the helper, or the
    PPO consistency check is measuring itself."""
    body = _update_body()
    code = '\n'.join(l for l in body.splitlines()
                     if l.strip() and not l.strip().startswith('#'))
    missing = [i for i in UPDATE_INGREDIENTS if i not in code]
    assert not missing, (
        'MAPPOTrainer.update no longer computes its new log-probs the way '
        'tests/test_wp6_ppo_logprob._update_path_log_probs recomputes them: '
        '%r is gone.  The PPO consistency check above is only meaningful '
        'while the two agree.' % missing)

    # the discrete heads are taken from the same constant the helper imports
    assert 'for h in DISCRETE_HEADS:' in code, (
        'update no longer iterates DISCRETE_HEADS, so the helper and the '
        'update path can disagree about which heads enter the log-prob')

    # and no temperature is applied on the update side.  A temperature here
    # and not at collection is exactly the divergence this file exists to
    # catch, and it would be invisible to the helper-vs-collection check.
    # The raw body, so the block boundaries (which are comments) survive.
    lp_block = body.split('apply_relay_mask(')[1].split('ratio')[0]
    assert 'temperature' not in lp_block and '/ temp' not in lp_block, (
        'update applies a temperature to the logits it scores with; '
        'compute_log_prob_batch takes one as an argument, so the two are no '
        'longer the same distribution unless the helper follows')
