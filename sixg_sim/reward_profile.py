"""
REWARD PROFILE — an ablation scaffold for the training reward only.

WHAT THIS IS.  The reward that trains the MARL policy was changed in two
bundled steps ("per-node delivery credit" and "repricing the relay /
reunification block") and the held-out KPI fell 38.3 % -> 29.2 %.  A bundled
change cannot be attributed, so this module makes each half of the bundle
individually selectable, at the level of the CONSTANTS the two reward
functions already read:

    sixg_sim/simulation.py  Simulator.compute_global_connectivity_reward
    sixg_sim/agent.py       RLAgent.calculate_reward

WHAT THIS IS NOT.  It is not a mechanism, not a bonus, and not an arm.  It
changes NO physics, NO capacity, NO admission rule, NO metric and NO
evaluation code.  It is read exactly once, at import time, and it is
inert unless the environment variable is set: the default profile `full`
is the production configuration, so an unset environment behaves exactly
as the shipped code.  The same scaffold role as
`Simulator.pinned_action_heads`.

The routing baselines (SDN / OSPF / OLSR / BATMAN / AODV) and the
random-action control never evaluate either reward function, so no arm's
behaviour depends on this selection.  Only the MARL training signal does.

USE
    set MARL_REWARD_PROFILE=reprice   (or export, on POSIX)
    python train_on_comparison.py ...

PROFILES
    prev      the configuration that trained to 38.3 % — NO per-node delivery
              credit, relay/reunification block at its ORIGINAL rates
    deliv     prev + per-node delivery credit                        (arm a)
    reprice   prev + relay/reunification repricing                   (arm b)
    both      deliv + reprice — the configuration that trained to 29.2 % (arm c)
    full      both + the REACH_RESTORED repricing        (arm d, DEFAULT)
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardProfile:
    name: str

    # ── (a) PER-NODE DELIVERY CREDIT — sixg_sim/agent.py ──────────────────
    # The local counterpart of the global "+100 x delivered fraction".
    # Without it the actor never trains on delivered volume at all: the
    # counterfactual baseline in MAPPOTrainer.finish_episode subtracts the
    # cross-agent mean advantage exactly, so any term identical for every
    # agent — which the GLOBAL delivered fraction is, by construction —
    # cancels out of the actor's gradient.
    delivery_points: float = 100.0      # == global "+100 x delivered fraction"
    gen_delivery_points: float = 10.0   # == global "+10 x general fraction"

    # ── (b) RELAY / REUNIFICATION REPRICING ──────────────────────────────
    # Global (simulation.py)
    relay_contrib_per_node: float = 0.0    # was 2.0
    relay_tput_cap_global: float = 5.0     # was 15.0
    reunify_full_bonus: float = 25.0       # was 40.0
    reunify_per_fragment: float = 6.0      # was 10.0
    # "+0.5 per routed flow, capped at 20, whenever ANY relay contributed" —
    # removed by the repricing, restored only to reproduce the `prev` arm.
    routed_flow_cliff: bool = False
    # Relay terms read the per-link DELIVERED-UE ledger (_link_carried_ue)
    # instead of link.current_utilization, which also contained the node's
    # own O&M telemetry — traffic the delivered fractions never counted.
    relay_terms_use_delivered_ledger: bool = True
    # Local (agent.py)
    relay_contrib_points: float = 0.0      # was 2.0
    relay_tput_cap: float = 3.0            # was 8.0
    bridge_held_points: float = 6.0        # was 10.0
    # ── (d) MISSION OBJECTIVE — per-site counterfactual bridge credit ─────
    # Points per fragment THIS site's bridges eliminate
    # (PHYMACState.bridge_counterfactual_fragments), paid every tick the
    # merge is held.  This is the local, non-cancelling reunification term:
    # the global reunify_* bonuses are identical for every agent and are
    # therefore removed EXACTLY by the cross-agent advantage centring, so the
    # actor never sees them; only a per-site term survives.  0.0 in every
    # legacy profile (inert).
    bridge_cf_points: float = 0.0

    # ── (c) STRANDED-USER REACHABILITY ───────────────────────────────────
    # See the REACHABILITY block of compute_global_connectivity_reward for
    # the arithmetic behind the demotion from 1.0/60.0 to 0.10/5.0 and for
    # why the term is now conditional on the restored users carrying traffic.
    reach_per_ue: float = 0.10             # was 1.0
    reach_cap: float = 5.0                 # was 60.0
    reach_require_traffic: bool = True     # was False (paid on reachability alone)


_LEGACY_RELAY = dict(
    relay_contrib_per_node=2.0,
    relay_tput_cap_global=15.0,
    reunify_full_bonus=40.0,
    reunify_per_fragment=10.0,
    routed_flow_cliff=True,
    relay_terms_use_delivered_ledger=False,
    relay_contrib_points=2.0,
    relay_tput_cap=8.0,
    bridge_held_points=10.0,
)
_NO_DELIVERY = dict(delivery_points=0.0, gen_delivery_points=0.0)
_LEGACY_REACH = dict(reach_per_ue=1.0, reach_cap=60.0,
                     reach_require_traffic=False)

# ── (d') REUNIFICATION BOOST — full + a heavier price on GENUINE bridging ──
# WHY.  The authoritative 20-instance evaluation (auth_eval_s1234) showed the
# learned policy trails the deterministic XL-DET baseline on ONE axis only:
# residual component count on the reunifiable recovery instances (MARL 1.29
# vs XL-DET 1.14), while it leads XL-DET on delivered throughput by ~12 pp.
# The cause is the documented credit-dilution of the relay head: with a
# parameter-shared actor over ~42 sites, the CAPACITY_BOOST gradient from the
# ~22 steerable sites is averaged against the sites where boosting only burns
# PRB, so the shared policy under-bridges.  The three constants below are the
# ONLY reward terms that pay for a genuine fragment merge, and each is gated
# on a REAL reduction in component count (the de-duplicated bridge set for the
# local term; count_infra_components for the two global terms), so raising
# them concentrates the learning signal on reunification WITHOUT opening a
# reward-hacking surface — nothing here pays for relay mode or capacity per
# se.  They were cut (10->6, 40->25, 10->6) to track each other during the
# earlier anti-hacking repricing, not because bridging itself was hackable;
# this profile restores and modestly exceeds the pre-cut rates.
# Everything else is identical to `full`.  Aim: close the component gap while
# keeping the throughput lead, so MARL Pareto-dominates XL-DET.
_REUNIFY_BOOST = dict(
    bridge_held_points=12.0,      # local  (full: 6.0)  — own link is a distinct merge
    reunify_per_fragment=12.0,    # global (full: 6.0)  — per fragment eliminated
    reunify_full_bonus=40.0,      # global (full: 25.0) — island back to one component
)

# ── reunify_boost2 — a stronger step on the SAME three terms ──────────────
# The first boost (12/12/40) closed only ~half the s1234 component gap to
# XL-DET (1.29 -> 1.21 vs 1.14) while throughput rose (51.0 -> 52.2 %), so the
# direction is right and there is ample throughput headroom (+13 pp over
# XL-DET) to spend on more bridging.  The LOCAL bridge-held term is the one
# that counters the parameter-shared actor's gradient dilution, so it is
# weighted up hardest.  Still only pays for GENUINE de-duplicated fragment
# merges — no reward-hacking surface — everything else identical to `full`.
_REUNIFY_BOOST2 = dict(
    bridge_held_points=18.0,      # local  (full 6.0, boost 12.0)
    reunify_per_fragment=15.0,    # global (full 6.0, boost 12.0)
    reunify_full_bonus=45.0,      # global (full 25.0, boost 40.0)
)

# ── mission — OBJECTIVE = THE MISSION ──────────────────────────────────────
# WHY.  The mission is emergency CONNECTIVITY first, throughput second.  The
# shipped reward is the other way round: delivered volume pays up to +100 per
# tick, while full reunification pays +25 and each fragment +6 -- and those
# two are GLOBAL (identical for every agent), so the cross-agent advantage
# centring removes them exactly from the actor's gradient.  The only bridging
# signal the actor ever received was the local bridge-held term (6 points,
# paid after a bridge forms and survives) against a dense delivery term.
# Measured consequence: four replicates reach the same training reward yet
# freeze into components 1.16 .. 2.86 -- reunification was a free variable.
#
# WHAT CHANGES.  A per-site COUNTERFACTUAL credit that centring cannot
# cancel: bridge_cf_points per fragment this site's own bridges eliminate
# (tree edges only, so redundant bridging scores 0 -- not hackable), paid
# while held.  At 25/fragment/tick it exceeds a typical site's delivery share
# (~2-10 points), which is what "connectivity primary" means numerically,
# while the delivery terms are left untouched so throughput still counts.
# The global reunify_* bonuses are raised too, for the CRITIC's benefit.
# bridge_held_points goes to 0 to avoid paying the same merge twice.
_MISSION = dict(
    bridge_cf_points=25.0,
    bridge_held_points=0.0,
    reunify_full_bonus=60.0,
    reunify_per_fragment=15.0,
)

PROFILES = {
    'mission': RewardProfile('mission', **_MISSION),
    'prev':    RewardProfile('prev',    **_NO_DELIVERY, **_LEGACY_RELAY,
                             **_LEGACY_REACH),
    'deliv':   RewardProfile('deliv',   **_LEGACY_RELAY, **_LEGACY_REACH),
    'reprice': RewardProfile('reprice', **_NO_DELIVERY, **_LEGACY_REACH),
    'both':    RewardProfile('both',    **_LEGACY_REACH),
    'full':    RewardProfile('full'),
    'reunify_boost': RewardProfile('reunify_boost', **_REUNIFY_BOOST),
    'reunify_boost2': RewardProfile('reunify_boost2', **_REUNIFY_BOOST2),
}

DEFAULT_PROFILE = 'full'

_name = os.environ.get('MARL_REWARD_PROFILE', DEFAULT_PROFILE).strip().lower()
if _name not in PROFILES:
    raise ValueError(
        f"MARL_REWARD_PROFILE={_name!r} is not one of {sorted(PROFILES)}")

PROFILE = PROFILES[_name]
