# -*- coding: utf-8 -*-
"""Contract test for the REMOVAL of the capability-conditioned relay mask.

RERUN_SPEC WP6 drops the `relay_capable` observation column: the audit shows
it is 1.0 at every agent site, so it carried no information, and the
evaluated configuration never enabled the mask (MARL_RELAY_MASK unset).  The
mask read exactly that column, so it goes with it.

This file used to prove that the mask behaved; it now proves that it is
gone, and gone LOUDLY rather than as a silent no-op:

  1. `apply_relay_mask` is the identity on any logits dict.  It survives only
     so that its call sites (mappo_trainer, simulation) need no edit.
  2. Asking for the mask is an error: importing sixg_sim.agent with
     MARL_RELAY_MASK=capability raises, rather than quietly doing nothing.
  3. `relay_capable` is no longer an actor observation field, and
     `OBS_COL_RELAY_CAPABLE` no longer exists.  The dataclass attribute
     stays, because XL-DET reads it by name.

The behaviour the mask was meant to fix -- CAPACITY_BOOST aliasing
LOCAL_REROUTE at non-steerable sites -- is handled by WP7, which reduces the
relay head to three modes.

The FOURTH check this file used to carry -- PPO consistency, that the
collection-time log-prob of a sampled batch equals the update-path log-prob
of the same (obs, action), so the ratio at epoch start is exactly 1 -- was
never about the mask.  It moved, in its mask-free form, to
tests/test_wp6_ppo_logprob.py (WP6-FIX3 defect 4); it was deleted with the
rest of this file in WP6 and went uncovered for two rounds.

Run:  python test_relay_mask.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


import torch                                                    # noqa: E402
import sixg_sim.agent as A                                       # noqa: E402

print("1. apply_relay_mask is the identity")
logits = {'relay': torch.randn(4, 5), 'tx_power': torch.randn(4, 5)}
obs = torch.zeros(4, A.ACTOR_OBS_DIM)
out = A.apply_relay_mask(logits, obs)
check(out is logits, "apply_relay_mask returns its input unchanged")
obs[:2, 0] = 1.0
check(A.apply_relay_mask(logits, obs) is logits,
      "still the identity whatever the observation carries")
check(A.RELAY_MASK == 'off', "RELAY_MASK is 'off' (%r)" % A.RELAY_MASK)

print()
print("2. MARL_RELAY_MASK=capability is an error, not a silent no-op")
code = ("import sys; sys.path.insert(0, %r); "
        "import sixg_sim.agent" % os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ)
env['MARL_RELAY_MASK'] = 'capability'
r = subprocess.run([sys.executable, '-c', code], capture_output=True,
                   text=True, env=env)
check(r.returncode != 0, "import fails with MARL_RELAY_MASK=capability")
check('REMOVED' in r.stderr and 'WP6' in r.stderr,
      "the error names WP6 and says the mask was removed"
      + ("" if 'REMOVED' in r.stderr else "\n" + r.stderr[-600:]))

print()
print("3. relay_capable is no longer an actor field")
check(not hasattr(A, 'OBS_COL_RELAY_CAPABLE'),
      "OBS_COL_RELAY_CAPABLE is gone")
check(not any('relay_capable' in n for n in A.ACTOR_OBS_NAMES),
      "no actor field named after relay_capable")
check(not any('relay_capable' in n for n in A.CRITIC_GLOBAL_NAMES),
      "no critic global field named after relay_capable either")
check(hasattr(A.ConnectivityState(), 'relay_capable'),
      "the ConnectivityState attribute stays (XL-DET reads it by name)")

print()
if FAILED:
    print("%d CHECK(S) FAILED" % len(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
