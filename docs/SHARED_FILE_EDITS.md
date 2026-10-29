# Shared-file edits, branch `trackAgent`

GENERATED FILE -- do not edit by hand.  Regenerate with:

    python3 tools/shared_file_edits.py --write

Every entry below is derived from `git diff -U0 9dec943 bde51a9`: the ranges
are the NEW-file ranges git prints, and the region is the innermost
`def`/`class` enclosing the range in the file at `bde51a9`, the branch
head the merge into master consumed.  This is a FROZEN record of the
branch: after the merge `HEAD` also carries the work of master, so the
diff is pinned to that head and not taken against `HEAD`.
`tests/test_wp8_shared_edits.py` regenerates this file (and the hunk
table `tests/data/shared_file_edits_bde51a9.tsv`) and fails if the committed text and the diff
disagree; a tree without git checks this file against that table, so
an omitted hunk is caught there too (D8.26; trackAgent merge).

SCOPE: source paths (`*.py`).  Documentation paths are listed by name
at the end without counts -- this file is itself one of them, so a
total that included it could not be written down.

## Reconciliation

| set | paths | hunks |
|---|---|---|
| owned by this track | 12 | 99 |
| NOT owned by this track | 25 | 207 |
| **all source paths in the diff** | **37** | **306** |

`99 + 207 = 306` hunks over `12 + 25 = 37` source paths.

## Paths this track does NOT own

Each one is a merge surface a human has to adjudicate.  Provenance is
a mark on the entry, never a reason to leave one out.

### `plot_severance_maps.py` -- 2 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `load_training_plan` | 2 | 215,1 221,2 | WP8/D8.24: the figure script mirrored the training episode length with its own literal; it reads the published constant now. |

### `run_timeline_comparison.py` -- 17 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 64,3 | WP6/WP7: the evaluation driver imports the new critic network, observation widths and the schema gate. |
| `_establish_relay_links` | 1 | 544,4 | WP7: the handover and scheduler arguments are gone with their heads; the SDN arm is bit-identical without them because the scheduler write only restored the value it had just read. |
| `_load_into_agent` | 1 | 3931,5 | WP6: the observation-dimension ADAPTATION is removed -- a checkpoint from another schema is refused by the strict gate before this is reached, so the function always returns False. |
| `run_physics_pass` | 11 | 4176,1 4227,0 4230,5 4237,1 4248,0 4251,4 4265,6 4271,0 4283,1 4296,1 4305,0 | WP6: the strict checkpoint gate runs BEFORE anything is read out of a checkpoint, and the fc1 zero-padding that used to make a narrower checkpoint load is gone (the WP6 schema drops INTERIOR columns, so padding would mis-align every column after the first drop). Checkpoint-loading region; the physics pass proper is untouched. |
| `run_physics_pass._act_fn` | 3 | 4911,3 4921,7 4941,3 | WP6/WP7 PPO COLLECTION: the critic feature vector (26 actor fields followed by 15 global ones), and the relay HEAD index rather than the enum index, which is what PPO scores. Action/reward handling. |

### `sixg_sim/heuristic_controller.py` -- 5 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 3 | 11,8 110,1 112,13 | XL-DET baseline: re-expressed against the WP7 head set so the comparator emits the same actions the simulator still applies. |
| `decide` | 2 | 280,9 310,1 | XL-DET baseline: re-expressed against the WP7 head set so the comparator emits the same actions the simulator still applies. |

### `sixg_sim/iops/learning_postcard.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `LearningPostcardExchanger._generate_postcard` | 1 | 209,0 | WP7: a deleted head is no longer carried on the postcard. |

### `sixg_sim/iops/migrate_checkpoint.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `(file deleted)` | 1 | 0,0 | WP6/WP7: deleted -- it migrated checkpoints across a schema this tree no longer has. |

### `sixg_sim/main.py` -- 8 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `main` | 8 | 169,4 199,2 234,11 355,2 393,21 416,1 432,12 482,5 | WP8: kappa and the episode length are published constants, so the flags are held to them instead of carrying their own defaults (D8.24). |

### `sixg_sim/mappo_trainer.py` -- 20 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 6 | 29,2 39,4 136,2 139,4 467,1 469,6 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `EWCPenalty.compute_fisher` | 1 | 299,4 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `MAPPOTrainer.collect` | 2 | 739,8 763,16 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `MAPPOTrainer.compute_log_prob` | 2 | 631,5 637,23 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `MAPPOTrainer.compute_log_prob_batch` | 2 | 684,3 688,33 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `MAPPOTrainer.get_value` | 2 | 599,10 611,4 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `ModelRegistry` | 1 | 436,5 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `ModelRegistry.load_latest` | 1 | 446,4 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `ModelRegistry.load_version` | 1 | 459,2 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |
| `ModelRegistry.save` | 2 | 377,44 425,1 | WP6/WP7: the shared actor/critic split, the per-head distributions and the head list the update loops iterate. |

### `sixg_sim/phy_mac_state.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `PHYMACState.normalise_prb` | 1 | 362,3 | WP6 observation field the actor reads. |

### `sixg_sim/reward_profile.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `(file deleted)` | 1 | 0,0 | WP8: deleted -- an environment-selected coefficient set is the drift channel this WP exists to close. |

### `sixg_sim/runinfo.py` -- 3 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 26,7 | WP8: reward_id() joins the provenance record, so a pickle names the reward that produced it. |
| `provenance` | 1 | 225,4 | WP8: reward_id() joins the provenance record, so a pickle names the reward that produced it. |
| `reward_id` | 1 | 199,6 | WP8: reward_id() joins the provenance record, so a pickle names the reward that produced it. |

### `sixg_sim/simulation.py` -- 49 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 2 | 31,5 50,0 | WP7: the relay-head index rule and the postcard heartbeat constant are imported from `agent.py`, so the action space has ONE definition; the import of a deleted helper goes. |
| `Simulator` | 4 | 2688,1 2690,14 3183,1 3185,8 | Class body, REWARD region. The reward-rate constants were DELETED (they are fields of reward_spec.RewardCoefficients), and with D8.23 the reunification cadence went with them -- the published field is read instead of a second copy. Includes the reachability-constants block. INSIDE this track's grant (the reward function). |
| `Simulator.__init__` | 2 | 1998,14 2087,3 | WP7: the postcard heartbeat is unconditional and hard-wired because its action head is deleted; the old switch survives only as a tripwire that refuses a stale setting. |
| `Simulator._build_agent_observations` | 1 | 5767,14 | WP6 observation schema: the observation vector is built to the new field list. INSIDE this track's grant. |
| `Simulator._build_agent_observations._build_one_obs` | 2 | 5965,6 5984,2 | WP6 observation schema: the per-agent vector. INSIDE this track's grant. |
| `Simulator._build_neighbor_radio_summary` | 2 | 5643,9 5653,13 | WP6 observation schema: the neighbour radio summary the actor reads. INSIDE this track's grant. |
| `Simulator._compute_connectivity_state` | 1 | 5557,7 | WP6 actor field `c_live_infra_neighbours`, with the WP3 seam recorded in the comment (D6.4): it stays the true-graph count until the engine track lands the link-state service. INSIDE this track's grant (observation schema code). |
| `Simulator._execute_agent_actions` | 22 | 6191,5 6205,6 6213,0 6242,22 6271,3 6284,1 6290,1 6316,1 6327,1 6333,5 6352,1 6355,9 6372,1 6430,6 6439,2 6475,2 6479,15 6497,1 6517,0 6529,1 6532,0 6552,22 | WP7 ACTION APPLICATION: the deleted heads are no longer applied and the surviving ones are read at their new indices. This is the NARROW EXCEPTION region the work package grants, and the largest single merge surface on this branch. |
| `Simulator._relay_hysteresis` | 1 | 6171,6 | WP7: both sides of the hysteresis comparison are HEAD indices now; the stored enum index would have indexed a 3-wide logit row with a value up to 4. Action-application region, NARROW EXCEPTION. |
| `Simulator._step_phy_mac` | 1 | 5148,6 | WP7: the scheduler and handover WRITES are deleted with their heads; nothing ever read either field. Action-application region, NARROW EXCEPTION, kept minimal. |
| `Simulator._update_phy_mac_observations` | 2 | 4623,5 4630,2 | WP6 observation schema: the restored-user delta the reach term prices is computed against the relay-free graph here. INSIDE this track's grant (observation schema code). |
| `Simulator.compute_global_connectivity_reward` | 9 | 2784,62 2847,3 2851,2 2855,1 2857,4 2862,1 2864,7 2874,1 2881,89 | The team objective itself: it MEASURES and DELEGATES to `reward_spec.global_reward`, and reads the published cadence (D8.23). INSIDE this track's grant (the reward function). |

### `sixg_sim/worker.py` -- 6 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `run_worker_episode` | 6 | 22,2 37,8 79,1 148,1 175,10 201,3 | WP6 critic-feature switch and WP7 head-set store: the worker builds the vectors the new actor/critic split and head set require. |

### `test_ensemble_policy.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 97,5 | Root contract script: the WP6 schema gate. |

### `test_marl_convergence.py` -- 13 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 4 | 19,2 38,1 54,7 63,2 | Root contract script: the WP6 schema and the WP7 head set. |
| `action_to_heads` | 2 | 98,1 103,1 | Root contract script: the WP6 schema and the WP7 head set. |
| `env_reward` | 2 | 88,1 90,4 | Root contract script: the WP6 schema and the WP7 head set. |
| `make_obs` | 1 | 68,1 | Root contract script: the WP6 schema and the WP7 head set. |
| `run_convergence_test` | 4 | 117,8 129,6 152,4 166,11 | Root contract script: the WP6 schema and the WP7 head set. |

### `test_mission_config.py` -- 15 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 15 | 3,9 14,3 26,6 34,1 54,1 60,4 66,2 69,1 75,6 87,14 104,5 110,13 196,1 207,1 212,1 | Root contract script: the WP8 guard no longer requires a reward profile, since the coefficients are frozen. |

### `test_relay_hysteresis.py` -- 18 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 16 | 11,10 26,11 68,11 80,8 100,1 110,9 122,3 128,10 139,1 141,5 176,3 180,1 182,4 187,22 212,2 217,1 | Root contract script: re-based on the WP7 head index, and the relay-margin comparison is exercised rather than short-circuited (D8.20). |
| `probe` | 2 | 158,9 168,5 | Root contract script: re-based on the WP7 head index, and the relay-margin comparison is exercised rather than short-circuited (D8.20). |

### `test_relay_mask.py` -- 5 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 5 | 2,30 48,2 51,9 62,11 75,9 | Root contract script: the WP7 relay head replaces the deleted per-head masks. |

### `test_xldet.py` -- 5 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 5 | 37,2 114,21 174,5 182,3 186,1 | Root contract script: the XL-DET comparator under the WP7 heads. |

### `tests/data/xldet_frozen_9dec943.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 1,291 | A frozen COPY of the baseline XL-DET controller, added so the comparator can be checked against the evaluated tree. |

### `tests/test_wp1_routing_graph.py` -- 2 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `test_teardown_site_step_phy_mac_with_underscore_node_ids` | 2 | 456,1 473,0 | Shared routing test updated for the WP6 schema id. |

### `tests/test_wp2_parity.py` -- 8 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 4 | 5,2 22,6 47,0 49,11 | Shared parity test updated for the WP6/WP7 schema and head set. |
| `_checkpoint` | 1 | 64,29 | Shared parity test updated for the WP6/WP7 schema and head set. |
| `_harness` | 2 | 103,8 112,1 | Shared parity test updated for the WP6/WP7 schema and head set. |
| `test_parity_training_step_vs_harness_step` | 1 | 123,1 | Shared parity test updated for the WP6/WP7 schema and head set. |

### `tools/constancy_dump.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 1,174 | WP8 diagnostic: added by this track. |

### `tools/reward_term_dump.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 1,240 | WP8 diagnostic: added by this track. |

### `tools/shared_file_edits.py` -- 1 hunk

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `<module>` | 1 | 1,475 | This generator: added by this track. |

### `train_on_comparison.py` -- 22 hunks

| region | hunks | new-file ranges | why |
|---|---|---|---|
| `all_pinnable_heads` | 1 | 374,15 | WP7: `ALL_PINNABLE_HEADS` was a second copy of the head list; it is derived from `DISCRETE_HEADS` now. INSIDE the action-handling grant. UNDECLARED until D8.26. |
| `main` | 11 | 600,6 617,5 630,12 651,1 665,11 710,1 724,2 729,4 745,12 773,8 1025,11 | WP6/WP7/WP8: the flags, banner and run summary follow the new schema, head set and frozen reward, and `resolve_frozen_run_shape()` is called here. Partly OUTSIDE the narrow grant (argument parsing and reporting), partly inside it (the reward mixing). |
| `resolve_frozen_run_shape` | 1 | 564,27 | WP8: kappa and the episode length are filled from the frozen object instead of from flag defaults. New function, INSIDE the reward-handling grant. |
| `run_comparison_worker_episode` | 5 | 118,1 121,2 138,2 149,8 174,1 | WP6/WP7: the worker episode builds the new observation and action shapes. INSIDE the action/reward grant. |
| `run_comparison_worker_episode._act` | 2 | 260,10 283,3 | WP6 critic-feature switch (`critic_features(...)` into `get_value`) and the WP7 head-set store (one `relay` head index replacing five head entries). INSIDE the action/reward grant. UNDECLARED until D8.26 -- two of the four missing hunks. |
| `run_comparison_worker_episode._complete_pending` | 1 | 227,1 | WP6/WP7: the pending transition is completed with the new action/observation shapes. INSIDE the action/reward grant. |
| `run_eval_episode` | 1 | 466,1 | WP7: the pinned-head set comes from `all_pinnable_heads()` rather than the deleted module constant. INSIDE the action-handling grant. UNDECLARED until D8.26 -- the hunk in the function the previous declaration never named at all. |

## Paths this track owns

Listed for the arithmetic above; no merge adjudication is expected.

| path | hunks |
|---|---|
| `make_untrained_checkpoints.py` | 11 |
| `sixg_sim/agent.py` | 78 |
| `sixg_sim/reward_spec.py` | 1 |
| `tests/test_wp6_checkpoint_gate.py` | 1 |
| `tests/test_wp6_obs_schema.py` | 1 |
| `tests/test_wp6_parity.py` | 1 |
| `tests/test_wp6_ppo_logprob.py` | 1 |
| `tests/test_wp6_smoke.py` | 1 |
| `tests/test_wp7_action_space.py` | 1 |
| `tests/test_wp7_controls.py` | 1 |
| `tests/test_wp8_reward.py` | 1 |
| `tests/test_wp8_shared_edits.py` | 1 |

## Forbidden paths

| path | hunks |
|---|---|
| `sixg_sim/topology.py` | 0 |
| `sixg_sim/link_state_detector.py` | 0 |
| `sixg_sim/link_state_service.py` | 0 |
| `sixg_sim/tick_step.py` | 0 |
| `sixg_sim/transport_relay_model.py` | 0 |
| `confirmatory_analysis.py` | 0 |
| `extract_timelines.py` | 0 |
| `dump_results_json.py` | 0 |

## Non-source paths in the diff

By name, without counts (see SCOPE above).

* `docs/DECISIONS.md`
* `docs/REWARD_FROZEN.md`
