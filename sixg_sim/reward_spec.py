# -*- coding: utf-8 -*-
"""THE REWARD, AS AN EXPLICIT OBJECT (RERUN_SPEC WP8).

WHY THIS MODULE EXISTS.  An audit of the manuscript found that `eq:reward`
and the `eq:local` text were WRONG -- they described a reward the code had
not computed for several revisions -- and they had to be corrected FROM a
code audit.  That could happen because the reward was arithmetic scattered
through `Simulator.compute_global_connectivity_reward` and
`RLAgent.calculate_reward`, mixed into ~250 lines of rationale, with its
coefficients reached through an environment-selected profile object.  There
was no single artefact anyone could read, print, diff or publish.

This module is that artefact:

  * `RewardCoefficients` -- one frozen dataclass, one NAMED field per
    coefficient, no environment reads.  `FROZEN` is the evaluated set.
  * `GlobalRewardInputs` / `LocalRewardInputs` -- the MEASURED quantities the
    reward is a function of, and nothing else.  Both are plain frozen
    dataclasses, so a term can be evaluated on a hand-constructed state.
  * `global_reward()` / `local_reward()` -- pure functions returning a
    per-term breakdown whose `.total` is the scalar the drivers use.
  * `mix()` -- the kappa = 0.6 team/local mixing of RERUN_SPEC WP8.  It is
    the mixing the DRIVERS call (train_on_comparison.py and
    sixg_sim/worker.py); `require_frozen_mix()` is the tripwire that keeps
    `--global-reward-mix` from becoming a second, silent kappa.
  * `coefficients_text()` -- the frozen set, rendered for VERBATIM inclusion
    in the pre-registration, so the published numbers are produced by the
    object that runs.

`Simulator.compute_global_connectivity_reward` and `RLAgent.calculate_reward`
now MEASURE and DELEGATE: they gather the inputs from the environment and
call into here.  They hold no rates.

-------------------------------------------------------------------------
TERMS REMOVED BY WP8, and why.  (RERUN_SPEC WP8: "Terms shown constant by
the constancy dump are removed and listed in the pre-registration.  Do not
keep a term multiplied by zero.")  The evidence for each is
`tools/reward_term_dump.py`; the numbers are in docs/DECISIONS.md.

  r_qos (local)              CONSTANT at 0.0.  It paid QOS_POINTS x the
                             node's LIFE_SAFETY `admission_success_rate`,
                             but only when that slice had offered load, and
                             no LIFE_SAFETY load is ever offered in the
                             comparison topology -- the WP6 constancy dump
                             records `a_emergency_fraction` as CONSTANT 0.0
                             in both scenarios for the same reason (no
                             emergency UE is ever marked).  So the condition
                             never fired and the term was 0.0 on every
                             sample.
  relay_contrib (global)     Coefficient 0.0 in the evaluated configuration
  relay_contrib (local)      (it was repriced 2.0 -> 0.5 -> 0.0 because a
                             FLAT payment for "being a relay that carried
                             something" cannot tell 1 Mbps from 500 Mbps).
                             A term multiplied by zero is not a term.
  bridge_held (local)        Coefficient 0.0 in the evaluated configuration:
                             the per-site COUNTERFACTUAL bridge credit
                             replaced it and paying both would pay the same
                             merge twice.
  recovery_ramp (global)     CONSTANT at 0.0.  Both are gated on a
  recovery_complete (global) `restore_core` event that the comparison
                             scenario builder does not emit at all, in either
                             scenario, for either driver.
  routed-flow cliff (global) "+0.5 per routed flow whenever ANY relay
                             contributed, capped at 20" -- switched OFF in
                             every shipped profile, i.e. dead code on the
                             production path.
  raw-utilisation relay      The alternative to the delivered-UE ledger.
  measurement (global)       FALSE in every shipped profile; the ledger is
                             now the only spelling.
  reachability without       `reach_require_traffic` was TRUE in every
  traffic (global)           shipped profile; the unconditional branch is
                             now the only spelling.

A removed term is DELETED, not set to zero: a zero coefficient is a term a
reader has to evaluate before knowing it does nothing, and it is how three
of the six above survived two repricings.

-------------------------------------------------------------------------
EPISODE SHAPE (RERUN_SPEC WP8, disclosed).  Training episodes are
`EPISODE_TICKS` = 800 ticks and the final tick is treated as TERMINAL, not
as a time-limit truncation: `MAPPOTrainer.finish_episode` marks each agent's
last completed transition `done=True`, so GAE bootstraps no value past it.
The returns of a truncated episode are therefore biased low by the value of
the unseen tail. This is DELIBERATE and unchanged from the evaluated tree;
it is disclosed here and in docs/REWARD_FROZEN.md because the evaluation
episodes are 3600 ticks long, so the agent is trained on a shorter horizon
than it is scored on.
"""
import hashlib
import os
from dataclasses import dataclass, fields

__all__ = [
    'RewardCoefficients', 'FROZEN', 'KAPPA', 'reward_id',
    'GlobalRewardInputs', 'GlobalRewardTerms', 'global_reward',
    'LocalRewardInputs', 'LocalRewardTerms', 'local_reward',
    'reunify_points', 'mix', 'require_frozen_mix', 'executed_relay_mode',
    'coefficients_text', 'dump_coefficients',
    'REMOVED_TERMS', 'KEPT_UNEARNED_TERMS',
    'EPISODE_TICKS', 'TRUNCATION_IS_TERMINAL', 'EPISODE_TICKS_SITES',
    'GLOBAL_TERM_NAMES', 'LOCAL_TERM_NAMES',
]

# ── Episode shape (disclosed; see the module docstring) ──────────────────
EPISODE_TICKS = 800
TRUNCATION_IS_TERMINAL = True

# Every place in the tree that needs the training-episode length, and HOW it
# gets it from `EPISODE_TICKS` above.  Each entry is checked by
# `tests/test_wp8_reward.py::test_the_episode_length_has_one_source`, which
# also fails if a file outside this table grows an independent literal 800
# bound to an episode length.  Before WP8-FIX three of the four carried their
# own `800` and only the first was bound (D8.24).
EPISODE_TICKS_SITES = (
    ('train_on_comparison.py',
     '`--episode-ticks` defaults to None; `resolve_frozen_run_shape()` '
     'fills it from `EPISODE_TICKS`'),
    ('make_untrained_checkpoints.py',
     '`EPISODE_TICKS` is imported from this module and written into the '
     'control checkpoint metadata'),
    ('sixg_sim/main.py',
     '`--episode-ticks` defaults to `reward_spec.EPISODE_TICKS`'),
    ('plot_severance_maps.py',
     "`load_training_plan()['episode_ticks']` is `reward_spec.EPISODE_TICKS`"),
)


@dataclass(frozen=True)
class RewardCoefficients:
    """Every rate in the reward, named, in one object.

    The defaults ARE the evaluated (frozen) configuration.  Nothing here
    reads the environment: RERUN_SPEC's patch discipline is that no env
    switch changes the evaluated system, and an env-selected coefficient set
    is exactly the channel by which the published equations and the running
    code drifted apart.
    """

    # ── TEAM OBJECTIVE (eq:reward), post-event / island mode ─────────────
    delivery: float = 100.0
    """Points per unit of volume-weighted delivered UE-to-UE fraction."""
    full_delivery_bonus: float = 25.0
    full_delivery_threshold: float = 0.95
    general_delivery: float = 10.0
    """Points per unit of delivered NON-UE-to-UE (general) volume fraction."""

    relay_churn: float = 1.0
    """Cost per site whose EXECUTED relay mode changed since the last tick."""

    relay_throughput_mbps_per_point: float = 50.0
    """Mb/s of relay-carried delivered UE volume per reward point.

    A DIVISOR, and deliberately so.  The pre-WP8 team objective computed
    `relay_throughput / 50.0`, and `x / 50.0` is NOT the same double as
    `x * (1.0 / 50.0)`: they differ in the last ulp for about 14% of inputs
    (measured, 200 000 uniform draws in [0, 300): 27 448 disagreements,
    first at x = 142.97908624570675).  Keeping the DIVISION reproduces the
    evaluated value bit for bit, so the only numerical change WP8 makes to
    the training path remains the executed-mode churn rule.  The agent-local
    counterpart below keeps its MULTIPLICATION for exactly the same reason:
    a multiplication by 1/50 is the arithmetic ITS pre-WP8 code wrote
    (`RLAgent.RELAY_TPUT_PER_MBPS = 1.0 / 50.0`).
    """
    relay_throughput_cap: float = 5.0

    reunify_per_fragment: float = 15.0
    reunify_residual: float = 2.0
    reunify_full_bonus: float = 60.0
    reunify_ref_fragments: int = 4
    reunify_eval_every: int = 5
    """Ticks between component recomputations; the payout is HELD between.

    A cadence, not a rate -- but a PUBLISHED one, so it is the cadence in
    force: `Simulator.compute_global_connectivity_reward` reads this field
    (`self.current_tick % _coef.reunify_eval_every`).  It used to carry an
    independent copy, `Simulator.REUNIFY_EVAL_EVERY = 5`, which was the
    number that actually ran while this one was the number that was hashed
    into `reward_id()` and printed in the pre-registration (D8.23).
    """

    bridge_loss: float = 15.0
    """Cost per DE-DUPLICATED cross-fragment bridge lost since the last tick."""

    reach_per_ue: float = 0.10
    reach_cap: float = 5.0

    # ── TEAM OBJECTIVE, pre-event / normal mode ──────────────────────────
    normal_delivery: float = 3.0
    normal_spare_capacity: float = 0.01
    normal_spare_prb_threshold: float = 0.85

    # ── AGENT-LOCAL OBJECTIVE (eq:local) ─────────────────────────────────
    local_delivery: float = 100.0
    local_general_delivery: float = 10.0
    local_relay_throughput_per_mbps: float = 1.0 / 50.0
    local_relay_throughput_cap: float = 3.0
    local_bridge_counterfactual: float = 25.0
    """Points per fragment THIS site's own bridges eliminate, per tick held."""
    local_relay_churn: float = 1.0
    local_mcs_overreach: float = 1.0
    local_energy: float = 1.0
    local_energy_ref_db: float = 6.0

    # ── MIXING (RERUN_SPEC WP8) ──────────────────────────────────────────
    kappa: float = 0.6
    """r_i = (1 - kappa) * local_i + kappa * team."""

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


FROZEN = RewardCoefficients()
KAPPA = FROZEN.kappa


def reward_id() -> str:
    """Digest of the frozen coefficient set and the removed-term list.

    Written into every checkpoint and result pickle through
    `runinfo.provenance`, so a run can be tied to the reward that produced
    it.  It is deliberately NOT folded into the WP6 observation/action
    `SCHEMA_ID`: that id answers "can this checkpoint be LOADED into this
    tree", which a reward change does not affect, while this one answers
    "was it TRAINED by this reward", which is a provenance question.
    """
    payload = chr(10).join(
        ['%s=%r' % (f.name, getattr(FROZEN, f.name)) for f in fields(FROZEN)]
        + ['removed=' + ','.join('%s@%s' % (n, w) for n, w, _ in REMOVED_TERMS)]
        + ['kept_unearned=' + ','.join('%s@%s' % (n, w)
                                       for n, w, _t, _y in KEPT_UNEARNED_TERMS)]
        + ['global_terms=' + ','.join(GLOBAL_TERM_NAMES),
           'local_terms=' + ','.join(LOCAL_TERM_NAMES),
           'episode_ticks=%d' % EPISODE_TICKS,
           'truncation_is_terminal=%r' % TRUNCATION_IS_TERMINAL])
    return 'rew1-' + hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


# ── THE ENVIRONMENT CANNOT MOVE A COEFFICIENT ANY MORE ──────────────────
# `MARL_REWARD_PROFILE` used to select one of eight coefficient sets at
# import time, and the EVALUATED configuration was `mission`, not the
# module's own default -- so the running rates depended on whether a shell
# script had exported the variable, and the published equations were written
# from the default.  That is a drift channel of exactly the kind RERUN_SPEC's
# patch discipline forbids ("no env switch changes the evaluated system").
#
# `sixg_sim/reward_profile.py` is DELETED and `RewardCoefficients` above is
# the evaluated `mission` set, hard-wired.  The variable is kept only as a
# TRIPWIRE, the same treatment `MARL_POSTCARD_ALWAYS` got at WP7: the
# evaluated scripts (confirmatory/run_amendment1.sh, smoke*.sh,
# tests/test_wp2_parity.py) still export `mission` and still work, while any
# OTHER value raises here instead of being silently ignored -- which is what
# would have happened had the module simply been removed.
_ENV_PROFILE = os.environ.get('MARL_REWARD_PROFILE', '').strip().lower()
EVALUATED_PROFILE_NAME = 'mission'
if _ENV_PROFILE and _ENV_PROFILE != EVALUATED_PROFILE_NAME:
    raise ValueError(
        'MARL_REWARD_PROFILE=%r: the reward coefficients are FROZEN by '
        'RERUN_SPEC WP8 and no environment variable selects them any more. '
        'The evaluated set is the former %r profile and it is now the only '
        'one; see sixg_sim/reward_spec.RewardCoefficients and '
        'docs/REWARD_FROZEN.md.  Unset the variable, or leave it at %r.'
        % (_ENV_PROFILE, EVALUATED_PROFILE_NAME, EVALUATED_PROFILE_NAME))

# The record that goes into the pre-registration next to the coefficients.
# (name, where it lived, why it is gone)
REMOVED_TERMS = (
    ('r_qos', 'agent-local',
     'CONSTANT 0.0: it was conditional on this node having LIFE_SAFETY '
     'offered load, and no LIFE_SAFETY load is ever offered in the '
     'comparison topology (the WP6 constancy dump records the matching '
     'observation field a_emergency_fraction as CONSTANT 0.0 in both '
     'scenarios).'),
    ('relay_contrib', 'team objective',
     'coefficient 0.0 in the evaluated configuration (repriced 2.0 -> 0.5 '
     '-> 0.0: a flat per-node payment cannot tell 1 Mbps from 500 Mbps).'),
    ('relay_contrib', 'agent-local',
     'coefficient 0.0 in the evaluated configuration, same reason.'),
    ('bridge_held', 'agent-local',
     'coefficient 0.0 in the evaluated configuration: the per-site '
     'counterfactual bridge credit replaced it, and paying both would pay '
     'the same merge twice.'),
    ('recovery_ramp / recovery_complete', 'team objective',
     'CONSTANT 0.0 in both scenarios.  Both are gated on '
     '`Simulator._recovery_started_tick`, which only a `restore_core` event '
     'sets, and `build_comparison_scenario` REPLACES the event list with one '
     'that has no such event -- Scenario A carries {sever_core, fail_link} '
     'and Scenario B {sever_core, rescue_force_arrival}.  '
     '`DiverseScenarioGenerator` removed `restore_core` deliberately ("the '
     'simulation focuses on island-internal recovery only").  Both drivers '
     'use that builder, so the two terms are identically zero in training '
     'AND in evaluation.'),
    ('routed-flow cliff', 'team objective',
     '"+0.5 per routed flow whenever ANY relay contributed, capped at 20" '
     '-- off in every shipped profile, i.e. dead on the production path.  It '
     'paid a second time for delivery the delivered-fraction term already '
     'paid for, and attributed that payment to relaying, which had not '
     'caused it.'),
    ('raw-utilisation relay measurement', 'team objective',
     'the alternative to the delivered-UE ledger; off in every shipped '
     'profile.  The ledger is now the only spelling.'),
    ('unconditional reachability', 'team objective',
     'paying for restored users that carry no traffic; off in every shipped '
     'profile.  The traffic-conditioned form is now the only spelling.'),
)


# Terms KEPT although the reference policy never earned one in any measured
# pass.  RERUN_SPEC WP8 requires the REMOVED terms to be listed in the
# pre-registration; this list exists because the converse disclosure matters
# just as much.  A reader of the pre-registration would otherwise have no way
# to learn that a published +25 bonus was never paid in either scenario, and
# would read the coefficient table as a description of what the measured
# policy was actually paid.
# (name, objective, the condition, what the constancy dump measured)
KEPT_UNEARNED_TERMS = (
    ('full_delivery', 'team objective',
     'delivered UE-to-UE fraction >= 0.95 (`full_delivery_threshold`)',
     'NEVER EARNED in either pass of `tools/reward_term_dump.py` '
     '(`output/reward_terms_wp8.json`, seed 82, 600 ticks, severance at '
     't=100): `global.full_delivery` is CONSTANT 0.0 over n=600 samples in '
     'BOTH scenarios.  It is KEPT, unlike the removed terms, because what is '
     'unreached is the THRESHOLD, not the input: `global.delivery` is live in '
     'both passes and reaches 47.1419 (Scenario A, recovery) and 52.7913 '
     '(Scenario B, rescue_ops) out of a possible 100.0 -- delivered fractions '
     'of 0.4714 and 0.5279 against the 0.95 threshold.  A term whose input '
     'moves and whose threshold this policy simply does not reach is a '
     'reachable incentive; deleting it would change the objective a BETTER '
     'policy is scored on, which is not what "do not keep a term multiplied '
     'by zero" asks for.'),
)


# ══ THE EXECUTED RELAY MODE — the WP4 adapter ═══════════════════════════
def executed_relay_mode(ps):
    """The relay mode the ENGINE applied to this site on the tick just past.

    WHAT THIS CODE DOES, EXACTLY: it reads the attribute
    `PHYMACState.relay_mode` off `ps` (falling back to `RelayMode.OFF` when
    the attribute is absent, which is the case only for the bare stand-in
    states the fast unit tests build).  It calls nothing.

    WHY THAT ATTRIBUTE IS THE EXECUTED MODE on this branch.
    `Simulator._step_phy_mac` writes the action's requested mode into
    `ps.relay_mode`, and then, when a WP4 engine hold blocks the change
    (`_relay_teardown_blocked`: beam-training lockout, bridge grace,
    make-before-break on a loaded link), it writes the INCUMBENT mode back
    into the same field -- its own comment says this "keeps the churn penalty
    consistent, since nothing physically changed this tick".  An engine hold
    therefore never shows up as a change in this field, which is precisely
    RERUN_SPEC WP8's requirement that the churn term be scored on the
    executed mode and engine holds count as environment dynamics.

    THE ENGINE'S OWN RECORD (trackAgent merge).  WP4 landed its executed
    relay-mode flip count (`Simulator._snapshot_relay_mode_flips`, the M9
    series `relay_mode_flips`), and it reads this SAME field at the end of
    the forwarding pass, after the holds.  Nothing had to be re-pointed: the
    churn the policy is charged for and the churn M9 reports are the same
    field.  `tests/test_merge_seams.py` forces a WP4 hold through the real
    `_step_phy_mac` and checks that this function, the flip counter and the
    local churn term all see the incumbent mode.  No other part of the
    reward reads a relay mode.
    """
    from .phy_mac_state import RelayMode
    mode = getattr(ps, 'relay_mode', None)
    return RelayMode.OFF if mode is None else mode


# ══ THE TEAM OBJECTIVE (eq:reward) ══════════════════════════════════════
@dataclass(frozen=True)
class GlobalRewardInputs:
    """Everything the team objective is a function of.  Measured by the
    simulator; nothing here is a coefficient and nothing is derived."""

    island_mode: bool = False

    # Delivered volume this tick, as fractions of the offered volume.
    delivered_fraction: float = 0.0
    general_delivered_fraction: float = 0.0

    # Sites whose EXECUTED relay mode differs from the previous tick's.
    churn_count: int = 0

    # Delivered UE volume carried on transport-relay links this tick, Mb/s.
    relay_throughput_mbps: float = 0.0

    # The reunification term, HELD between the environment's periodic
    # component recounts (see `reunify_points`, which computes it).
    reunify_held: float = 0.0

    # De-duplicated cross-fragment bridges that existed last tick and do not
    # exist now.
    bridges_lost: int = 0

    # Restored UEs that carried delivered traffic this tick.
    restored_ue_paid: int = 0

    # Pre-event only: sites whose PRB utilisation is below the threshold.
    spare_capacity_sites: int = 0


GLOBAL_TERM_NAMES = (
    'delivery', 'full_delivery', 'general', 'relay_churn', 'relay_throughput',
    'reunify', 'bridge_loss', 'reach',
    'normal_delivery', 'normal_spare_capacity',
)


@dataclass(frozen=True)
class GlobalRewardTerms:
    delivery: float = 0.0
    full_delivery: float = 0.0
    general: float = 0.0
    relay_churn: float = 0.0
    relay_throughput: float = 0.0
    reunify: float = 0.0
    bridge_loss: float = 0.0
    reach: float = 0.0
    normal_delivery: float = 0.0
    normal_spare_capacity: float = 0.0

    @property
    def total(self) -> float:
        return sum(getattr(self, n) for n in GLOBAL_TERM_NAMES)

    def as_dict(self):
        d = {n: getattr(self, n) for n in GLOBAL_TERM_NAMES}
        d['total'] = self.total
        return d


def reunify_points(frag_with_relays: int, frag_without_relays: int,
                   coef: RewardCoefficients = FROZEN) -> float:
    """The island-reunification term, from the two component counts.

    Pure, and the ONLY place the reunification arithmetic is written.  The
    environment recounts components every `coef.reunify_eval_every` ticks and
    HOLDS this value in between.  The cadence is applied by the environment,
    but it is READ from this object -- `compute_global_connectivity_reward`
    takes it off the same coefficient set it prices with -- so the published
    cadence is the one in force (D8.23).
    """
    eliminated = max(0, int(frag_without_relays) - int(frag_with_relays))
    r = coef.reunify_per_fragment * eliminated
    r -= coef.reunify_residual * max(0, int(frag_with_relays) - 1)
    if int(frag_with_relays) == 1 and int(frag_without_relays) > 1:
        # Scale the completion bonus by how much reunification work the
        # scenario actually demanded, so closing a single orphan does not pay
        # what closing the design partition pays.
        scale = min(1.0, (int(frag_without_relays) - 1)
                    / float(max(1, coef.reunify_ref_fragments - 1)))
        r += coef.reunify_full_bonus * scale
    return r


def global_reward(inp: GlobalRewardInputs,
                  coef: RewardCoefficients = FROZEN) -> GlobalRewardTerms:
    """The team objective, term by term.  Pure."""
    if not inp.island_mode:
        # ── Pre-event: small by design, just enough not to starve the
        #    agents before the severance.
        return GlobalRewardTerms(
            normal_delivery=coef.normal_delivery * _unit(inp.delivered_fraction),
            normal_spare_capacity=(coef.normal_spare_capacity
                                   * int(inp.spare_capacity_sites)),
        )

    d = _unit(inp.delivered_fraction)
    g = _unit(inp.general_delivered_fraction)

    tput = 0.0
    if inp.relay_throughput_mbps > 0:
        tput = min(coef.relay_throughput_cap,
                   inp.relay_throughput_mbps
                   / coef.relay_throughput_mbps_per_point)

    reach = 0.0
    if inp.restored_ue_paid > 0:
        reach = min(coef.reach_cap,
                    coef.reach_per_ue * int(inp.restored_ue_paid))

    return GlobalRewardTerms(
        delivery=coef.delivery * d,
        full_delivery=(coef.full_delivery_bonus
                       if d >= coef.full_delivery_threshold else 0.0),
        general=coef.general_delivery * g,
        relay_churn=-coef.relay_churn * int(inp.churn_count),
        relay_throughput=tput,
        reunify=float(inp.reunify_held),
        bridge_loss=-coef.bridge_loss * int(inp.bridges_lost),
        reach=reach,
    )


# ══ THE AGENT-LOCAL OBJECTIVE (eq:local) ════════════════════════════════
@dataclass(frozen=True)
class LocalRewardInputs:
    """Everything the agent-local difference reward is a function of.  Every
    field is a quantity of THIS node, responsive to THIS node's action."""

    # This node's carried DELIVERED volume over the tick's OFFERED volume --
    # same units and same denominator as the team objective's fractions.
    delivered_share_ue_to_ue: float = 0.0
    delivered_share_general: float = 0.0

    # Traffic on this node's OWN transport-relay links, Mb/s, paid only when
    # the node is itself in a relaying mode.
    in_relay_mode: bool = False
    relay_carried_mbps: float = 0.0

    # Fragments THIS site's own bridges eliminate (tree edges only, so a
    # redundant parallel bridge scores 0).
    bridge_counterfactual_fragments: int = 0

    # This site's EXECUTED relay mode changed since the previous tick
    # (see `executed_relay_mode`: engine holds are NOT changes).
    relay_mode_changed: bool = False

    # This node's applied MCS exceeded what its own SINR supports and forced
    # an OLLA step-down on this tick.
    mcs_stepped_down: bool = False

    # The Tx-power step this node's action selected, in dB above its floor.
    tx_power_step_db: float = 0.0


LOCAL_TERM_NAMES = (
    'delivery', 'general_delivery', 'relay_throughput',
    'bridge_counterfactual', 'relay_churn', 'mcs_overreach', 'energy',
)


@dataclass(frozen=True)
class LocalRewardTerms:
    delivery: float = 0.0
    general_delivery: float = 0.0
    relay_throughput: float = 0.0
    bridge_counterfactual: float = 0.0
    relay_churn: float = 0.0
    mcs_overreach: float = 0.0
    energy: float = 0.0

    @property
    def total(self) -> float:
        return sum(getattr(self, n) for n in LOCAL_TERM_NAMES)

    def as_dict(self):
        d = {n: getattr(self, n) for n in LOCAL_TERM_NAMES}
        d['total'] = self.total
        return d


def local_reward(inp: LocalRewardInputs,
                 coef: RewardCoefficients = FROZEN) -> LocalRewardTerms:
    """The agent-local difference-reward surrogate, term by term.  Pure."""
    tput = 0.0
    if inp.in_relay_mode and inp.relay_carried_mbps > 0.0:
        tput = min(coef.local_relay_throughput_cap,
                   inp.relay_carried_mbps
                   * coef.local_relay_throughput_per_mbps)

    return LocalRewardTerms(
        delivery=coef.local_delivery * float(inp.delivered_share_ue_to_ue),
        general_delivery=(coef.local_general_delivery
                          * float(inp.delivered_share_general)),
        relay_throughput=tput,
        bridge_counterfactual=(coef.local_bridge_counterfactual
                               * int(inp.bridge_counterfactual_fragments)),
        relay_churn=(-coef.local_relay_churn if inp.relay_mode_changed
                     else 0.0),
        mcs_overreach=(-coef.local_mcs_overreach if inp.mcs_stepped_down
                       else 0.0),
        energy=-max(0.0, inp.tx_power_step_db / coef.local_energy_ref_db)
        * coef.local_energy,
    )


def mix(local_total: float, team_total: float,
        coef: RewardCoefficients = FROZEN) -> float:
    """r_i = (1 - kappa) * local_i + kappa * team   (RERUN_SPEC WP8)."""
    k = coef.kappa
    return (1.0 - k) * float(local_total) + k * float(team_total)


def require_frozen_mix(value, where='--global-reward-mix'):
    """TRIPWIRE on the mixing weight: it is `kappa`, and nothing else.

    WHAT THIS FUNCTION DOES: `None` means "not specified" and resolves to
    `FROZEN.kappa`; a value equal to `FROZEN.kappa` is returned unchanged;
    ANY other value raises.

    WHY IT EXISTS.  `--global-reward-mix` carried its own default literal in
    `train_on_comparison.py` and in `sixg_sim/main.py`, and both drivers
    mixed with THAT number -- `(1 - g_mix) * local + g_mix * team` -- while
    the [REWARD] banner, the run summary and `reward_id()` all reported
    `FROZEN.kappa`.  A run at `--global-reward-mix 0.3` therefore trained at
    kappa = 0.3 while every artefact recorded 0.6, and every test stayed
    green.  That is the same drift channel as `MARL_REWARD_PROFILE` above,
    and a worse one, because the environment channel at least raised on an
    unexpected value.

    The flag is KEPT rather than deleted for the same reason the variable is:
    the scripts that pass the evaluated 0.6 keep working, and a script that
    asks for a different mixture now fails loudly instead of being obeyed
    silently.
    """
    k = FROZEN.kappa
    if value is None:
        return k
    v = float(value)
    if v != k:
        raise ValueError(
            '%s=%r: the team/local mixing weight is FROZEN by RERUN_SPEC WP8 '
            'at kappa=%r (sixg_sim.reward_spec.FROZEN.kappa, published in '
            'docs/REWARD_FROZEN.md and hashed into reward_id()).  A run at '
            'any other value would train on a reward that no artefact of the '
            'run records.  Pass %r, or leave the flag off.'
            % (where, v, k, k))
    return k


def _unit(x) -> float:
    return max(0.0, min(1.0, float(x)))


# ══ THE PRE-REGISTRATION ARTEFACT ═══════════════════════════════════════
_HEADER = """# The frozen reward (RERUN_SPEC WP8)

GENERATED FILE -- do not edit by hand.  Regenerate with:

    python3 -m sixg_sim.reward_spec > docs/REWARD_FROZEN.md

Every number below is read straight out of
`sixg_sim.reward_spec.RewardCoefficients`, the object the running code
evaluates.  `tests/test_wp8_reward.py` compares this file to that object
field by field, so the two cannot drift.

The mixed per-agent reward is

    r_i(t) = (1 - kappa) * local_i(t) + kappa * team(t),    kappa = %(kappa)s

with the advantage centring unchanged (the cross-agent counterfactual
baseline in `MAPPOTrainer.finish_episode`).

That kappa is the one the RUNNING code mixes with: `train_on_comparison.py`
and `sixg_sim/worker.py` both mix by calling `reward_spec.mix()`, and
`--global-reward-mix` accepts this value or nothing at all
(`reward_spec.require_frozen_mix`, which raises on anything else).  Until
WP8-FIX that flag carried its own default literal and was the live weight,
while the run banner, the run summary and `reward_id()` all reported the
frozen kappa -- so a run at another mixture would have recorded this one.

Training episodes are %(ticks)d ticks and truncation is treated as TERMINAL
(`finish_episode` marks the last completed transition done=True, so GAE
bootstraps no value past it; `tests/test_wp8_reward.py` checks that on the
BEHAVIOUR of `finish_episode`, by comparing the flushed return against the
bootstrapped one, not on the presence of a word in its source).

The episode length is `reward_spec.EPISODE_TICKS`.  Every site listed
below reads it from there, and `tests/test_wp8_reward.py` checks each one
on the value it RESOLVES to, not on the presence of the name:

%(ticks_sites)s

The same test scans every other production file for a stray copy (a bare
episode-length literal in a keyword, an argparse default, an assignment or
a tuple unpacking).  One production file carries the number for a
different quantity and is exempted by name, with its reason, in that test:
`sixg_sim/gen_demo_kpis.py`, the length of a synthetic demonstration series
that runs no simulator and trains nothing.

Evaluation episodes are 3600 ticks, so the training horizon is shorter than
the scoring horizon; this is unchanged from the evaluated tree and is
disclosed, not corrected, by this WP.
"""


def coefficients_text() -> str:
    """The frozen coefficient set, rendered for VERBATIM inclusion in the
    pre-registration."""
    coef = FROZEN
    sites = chr(10).join('* `%s` -- %s' % (path, how)
                         for path, how in EPISODE_TICKS_SITES)
    out = [_HEADER % {'kappa': repr(coef.kappa), 'ticks': EPISODE_TICKS,
                      'ticks_sites': sites}]

    groups = [
        ('Team objective (eq:reward), post-event / island mode', [
            'delivery', 'full_delivery_bonus', 'full_delivery_threshold',
            'general_delivery', 'relay_churn',
            'relay_throughput_mbps_per_point',
            'relay_throughput_cap', 'reunify_per_fragment',
            'reunify_residual', 'reunify_full_bonus', 'reunify_ref_fragments',
            'reunify_eval_every', 'bridge_loss', 'reach_per_ue', 'reach_cap',
        ]),
        ('Team objective, pre-event / normal mode', [
            'normal_delivery', 'normal_spare_capacity',
            'normal_spare_prb_threshold',
        ]),
        ('Agent-local objective (eq:local)', [
            'local_delivery', 'local_general_delivery',
            'local_relay_throughput_per_mbps', 'local_relay_throughput_cap',
            'local_bridge_counterfactual', 'local_relay_churn',
            'local_mcs_overreach', 'local_energy', 'local_energy_ref_db',
        ]),
        ('Mixing', ['kappa']),
    ]
    named = [n for _t, ns in groups for n in ns]
    missing = [f.name for f in fields(coef) if f.name not in named]
    assert not missing, 'coefficients_text() does not render %r' % (missing,)

    for title, names in groups:
        out.append('\n## %s\n' % title)
        out.append('| coefficient | value |')
        out.append('|---|---|')
        for n in names:
            out.append('| `%s` | %r |' % (n, getattr(coef, n)))

    out.append('\n## Terms REMOVED by WP8\n')
    out.append('| term | objective | why |')
    out.append('|---|---|---|')
    for name, where, why in REMOVED_TERMS:
        out.append('| `%s` | %s | %s |' % (name, where, why))

    out.append('\n## Terms KEPT that the reference policy never earns\n')
    out.append('These terms are LIVE -- they are not removed -- but the '
               'reference policy earned none of them in the constancy '
               'dump, so no measured reward in this study contains one.\n')
    out.append('| term | objective | condition | what was measured |')
    out.append('|---|---|---|---|')
    for name, where, cond, measured in KEPT_UNEARNED_TERMS:
        out.append('| `%s` | %s | %s | %s |' % (name, where, cond, measured))

    out.append('\n## Term order\n')
    out.append('Team objective terms, in the order `global_reward()` '
               'returns them:\n')
    out.append('`' + '`, `'.join(GLOBAL_TERM_NAMES) + '`\n')
    out.append('Agent-local terms, in the order `local_reward()` '
               'returns them:\n')
    out.append('`' + '`, `'.join(LOCAL_TERM_NAMES) + '`')
    return '\n'.join(out) + '\n'


def dump_coefficients(path: str) -> str:
    import os
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    text = coefficients_text()
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return path


if __name__ == '__main__':
    import sys
    sys.stdout.write(coefficients_text())
