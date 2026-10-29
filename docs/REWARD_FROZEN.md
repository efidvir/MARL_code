# The frozen reward (RERUN_SPEC WP8)

GENERATED FILE -- do not edit by hand.  Regenerate with:

    python3 -m sixg_sim.reward_spec > docs/REWARD_FROZEN.md

Every number below is read straight out of
`sixg_sim.reward_spec.RewardCoefficients`, the object the running code
evaluates.  `tests/test_wp8_reward.py` compares this file to that object
field by field, so the two cannot drift.

The mixed per-agent reward is

    r_i(t) = (1 - kappa) * local_i(t) + kappa * team(t),    kappa = 0.6

with the advantage centring unchanged (the cross-agent counterfactual
baseline in `MAPPOTrainer.finish_episode`).

That kappa is the one the RUNNING code mixes with: `train_on_comparison.py`
and `sixg_sim/worker.py` both mix by calling `reward_spec.mix()`, and
`--global-reward-mix` accepts this value or nothing at all
(`reward_spec.require_frozen_mix`, which raises on anything else).  Until
WP8-FIX that flag carried its own default literal and was the live weight,
while the run banner, the run summary and `reward_id()` all reported the
frozen kappa -- so a run at another mixture would have recorded this one.

Training episodes are 800 ticks and truncation is treated as TERMINAL
(`finish_episode` marks the last completed transition done=True, so GAE
bootstraps no value past it; `tests/test_wp8_reward.py` checks that on the
BEHAVIOUR of `finish_episode`, by comparing the flushed return against the
bootstrapped one, not on the presence of a word in its source).

The episode length is `reward_spec.EPISODE_TICKS`.  Every site listed
below reads it from there, and `tests/test_wp8_reward.py` checks each one
on the value it RESOLVES to, not on the presence of the name:

* `train_on_comparison.py` -- `--episode-ticks` defaults to None; `resolve_frozen_run_shape()` fills it from `EPISODE_TICKS`
* `make_untrained_checkpoints.py` -- `EPISODE_TICKS` is imported from this module and written into the control checkpoint metadata
* `sixg_sim/main.py` -- `--episode-ticks` defaults to `reward_spec.EPISODE_TICKS`
* `plot_severance_maps.py` -- `load_training_plan()['episode_ticks']` is `reward_spec.EPISODE_TICKS`

The same test scans every other production file for a stray copy (a bare
episode-length literal in a keyword, an argparse default, an assignment or
a tuple unpacking).  One production file carries the number for a
different quantity and is exempted by name, with its reason, in that test:
`sixg_sim/gen_demo_kpis.py`, the length of a synthetic demonstration series
that runs no simulator and trains nothing.

Evaluation episodes are 3600 ticks, so the training horizon is shorter than
the scoring horizon; this is unchanged from the evaluated tree and is
disclosed, not corrected, by this WP.


## Team objective (eq:reward), post-event / island mode

| coefficient | value |
|---|---|
| `delivery` | 100.0 |
| `full_delivery_bonus` | 25.0 |
| `full_delivery_threshold` | 0.95 |
| `general_delivery` | 10.0 |
| `relay_churn` | 1.0 |
| `relay_throughput_mbps_per_point` | 50.0 |
| `relay_throughput_cap` | 5.0 |
| `reunify_per_fragment` | 15.0 |
| `reunify_residual` | 2.0 |
| `reunify_full_bonus` | 60.0 |
| `reunify_ref_fragments` | 4 |
| `reunify_eval_every` | 5 |
| `bridge_loss` | 15.0 |
| `reach_per_ue` | 0.1 |
| `reach_cap` | 5.0 |

## Team objective, pre-event / normal mode

| coefficient | value |
|---|---|
| `normal_delivery` | 3.0 |
| `normal_spare_capacity` | 0.01 |
| `normal_spare_prb_threshold` | 0.85 |

## Agent-local objective (eq:local)

| coefficient | value |
|---|---|
| `local_delivery` | 100.0 |
| `local_general_delivery` | 10.0 |
| `local_relay_throughput_per_mbps` | 0.02 |
| `local_relay_throughput_cap` | 3.0 |
| `local_bridge_counterfactual` | 25.0 |
| `local_relay_churn` | 1.0 |
| `local_mcs_overreach` | 1.0 |
| `local_energy` | 1.0 |
| `local_energy_ref_db` | 6.0 |

## Mixing

| coefficient | value |
|---|---|
| `kappa` | 0.6 |

## Terms REMOVED by WP8

| term | objective | why |
|---|---|---|
| `r_qos` | agent-local | CONSTANT 0.0: it was conditional on this node having LIFE_SAFETY offered load, and no LIFE_SAFETY load is ever offered in the comparison topology (the WP6 constancy dump records the matching observation field a_emergency_fraction as CONSTANT 0.0 in both scenarios). |
| `relay_contrib` | team objective | coefficient 0.0 in the evaluated configuration (repriced 2.0 -> 0.5 -> 0.0: a flat per-node payment cannot tell 1 Mbps from 500 Mbps). |
| `relay_contrib` | agent-local | coefficient 0.0 in the evaluated configuration, same reason. |
| `bridge_held` | agent-local | coefficient 0.0 in the evaluated configuration: the per-site counterfactual bridge credit replaced it, and paying both would pay the same merge twice. |
| `recovery_ramp / recovery_complete` | team objective | CONSTANT 0.0 in both scenarios.  Both are gated on `Simulator._recovery_started_tick`, which only a `restore_core` event sets, and `build_comparison_scenario` REPLACES the event list with one that has no such event -- Scenario A carries {sever_core, fail_link} and Scenario B {sever_core, rescue_force_arrival}.  `DiverseScenarioGenerator` removed `restore_core` deliberately ("the simulation focuses on island-internal recovery only").  Both drivers use that builder, so the two terms are identically zero in training AND in evaluation. |
| `routed-flow cliff` | team objective | "+0.5 per routed flow whenever ANY relay contributed, capped at 20" -- off in every shipped profile, i.e. dead on the production path.  It paid a second time for delivery the delivered-fraction term already paid for, and attributed that payment to relaying, which had not caused it. |
| `raw-utilisation relay measurement` | team objective | the alternative to the delivered-UE ledger; off in every shipped profile.  The ledger is now the only spelling. |
| `unconditional reachability` | team objective | paying for restored users that carry no traffic; off in every shipped profile.  The traffic-conditioned form is now the only spelling. |

## Terms KEPT that the reference policy never earns

These terms are LIVE -- they are not removed -- but the reference policy earned none of them in the constancy dump, so no measured reward in this study contains one.

| term | objective | condition | what was measured |
|---|---|---|---|
| `full_delivery` | team objective | delivered UE-to-UE fraction >= 0.95 (`full_delivery_threshold`) | NEVER EARNED in either pass of `tools/reward_term_dump.py` (`output/reward_terms_wp8.json`, seed 82, 600 ticks, severance at t=100): `global.full_delivery` is CONSTANT 0.0 over n=600 samples in BOTH scenarios.  It is KEPT, unlike the removed terms, because what is unreached is the THRESHOLD, not the input: `global.delivery` is live in both passes and reaches 47.1419 (Scenario A, recovery) and 52.7913 (Scenario B, rescue_ops) out of a possible 100.0 -- delivered fractions of 0.4714 and 0.5279 against the 0.95 threshold.  A term whose input moves and whose threshold this policy simply does not reach is a reachable incentive; deleting it would change the objective a BETTER policy is scored on, which is not what "do not keep a term multiplied by zero" asks for. |

## Term order

Team objective terms, in the order `global_reward()` returns them:

`delivery`, `full_delivery`, `general`, `relay_churn`, `relay_throughput`, `reunify`, `bridge_loss`, `reach`, `normal_delivery`, `normal_spare_capacity`

Agent-local terms, in the order `local_reward()` returns them:

`delivery`, `general_delivery`, `relay_throughput`, `bridge_counterfactual`, `relay_churn`, `mcs_overreach`, `energy`
