# Code, trained policies and results of the article

*Multi-Agent Reinforcement Learning for Autonomous 6G Mesh Network Recovery After Catastrophic Infrastructure Failure*

This branch (`article`, tag `article-v1`) holds the code that ran the test reported in the article,
the four trained policies and the result files behind the article's numbers, tables and
figures. The tables below say which code produced each result file and which script wrote
it. The raw outputs of the runs (one file per block of runs) and the article's own table,
figure and number-checking tools are not included, so the result files cannot be
regenerated from this branch alone.

## Commit names

The short hashes in this README are commits of the author's working repository, not of this
repository: this branch is the published copy of `c6c5de5` (with placeholders, see below);
branch `r2` (tag `confirmatory-R2`) is the published copy of `6b53f3f`, the code that trained
the policies; `25d8d2c`, a later version, is not published (only the SHA-256 list of its files,
on branch `r4-freeze0`).

## Contents

* `sixg_sim/` -- the digital twin (`simulation.py`, `tick_step.py`, `phy_mac_state.py`,
  `scenario.py`, `topology.py`, `traffic.py`, `transport_relay_model.py`, and the heartbeat
  rule shared by all controllers, `postcards.py`), the MARL controller and its training
  (`agent.py`, `mappo_trainer.py`, `reward_spec.py`, with the driver `train_on_comparison.py`
  at the top level), XL-DET (`heuristic_controller.py`), the control overlays
  (`control_plane.py`) and the link-state service of the MARL, XL-DET and random
  controllers (`link_state_service.py`).
* `run_timeline_comparison.py` -- the evaluation harness, with the routing protocols and the
  SDN controller (classes `OSPFController`, `OLSRController`, `BATMANController`,
  `AODVController`, `SDNController`).
* `confirmatory_analysis.py` -- the statistical analysis; `confirmatory/` -- the scripts
  of the run chain that ran the test and the analysis.
* `config/` -- scenario and topology configuration; `tests/` and `test_*.py` -- tests;
  `tools/` -- checking and audit tools; `docs/` -- design and run documents, including the
  decision record `docs/DECISIONS.md`.
* `policies/ck_s1234/policy_final.pt` -- the evaluated policy (recorded sha256
  `54eaa5623148cfd4...` in `confirmatory/SELECTED_r3lite.txt`; the published file differs from it
  only in masked metadata, see "Checking the policies"); `policies/ck_s2345/`, `ck_s3456/`,
  `ck_s4567/` -- the three other policies trained with the same configuration from different
  random seeds. The four were trained by the code of branch `r2` and are used here unchanged.
  The evaluated policy was selected from the four and their ensemble by its paired
  achievability against XL-DET on development and validation topologies; the rule and the
  ranking are recorded in `confirmatory/SELECTED_r3lite.txt`.
  The `ce_ck_s*/` folders of the code tree hold checkpoints of an earlier version; they are
  not the policies evaluated in the article.
* `results/` -- the result files listed below (test topologies: generator seeds 300-339).
* `records/FREEZE_c6c5de5.sha256` -- the SHA-256 of every file of the code tree.

## Results

"Made with" names the code that produced the data, as recorded in the file's own
provenance where it has one, otherwise the code that produced the data it summarises:

* *this branch* -- the test runs of the code tree of this branch;
* *r2 (training)* -- the training run of the four policies (code of branch `r2`);
* *later* -- the later version `25d8d2c` (not published). It was used for the example on
  development topology 58, and `identity_seed58.json` records that this pass reproduced,
  bit for bit, the pass of the same policy on the same topology made with the code of
  branch `r2` (*later vs r2*); for the arm `tstar_tu1` (see below); and, with an added option
  that runs the microwave and mesh hops at their own capacity, for the flag-on side of the
  backhaul test (*later + option*; the option's code is not published either).

Some files also hold controllers that the article's comparisons do not include: the
ensemble of the four policies (`ensemble`, `ens`; this branch) in `per_seed_300_339.json`,
`all_arms_300_339.json`/`.md`, `logs/all_arms.log`, `timelines_r3.json` and
`eh_summary.txt`; and a variant of XL-DET whose parameters were tuned on development
topologies after the test (`tstar_tu1`, "XL-DET tuned"; made with the later version and run
on the test topologies afterwards; see the documents on branches `r4-freeze0`,
`r4-annex-a`, `r4-annex-b` and `r4-stop-record`) in `per_seed_300_339.json`,
`all_arms_300_339.json`/`.md` and `logs/all_arms.log`. The untrained policy
(`untrained_s1234`; in `per_seed_300_339.json` and the S12 block of
`r3lite_cf_report_full.json`; its checkpoint is not published) is used in the article.

"Written by" names the published script that wrote the file; where it is empty, the script
is not published. Four of the published scripts load code from the tree named by their
`R4_TREE` setting (the later version), and two of them import helpers (`r4_*`) that exist
only there.

md5 values recorded inside the files are those of the files before the replacements
described under "Checking the code tree".

| file | content | made with | written by | bytes |
|---|---|---|---|---|
| `results/per_seed_300_339.json` | per-topology steady-state values of every controller, test topologies 300-339, both scenarios | this branch; arm tstar_tu1: later | `server_scripts/all_arms_300_339.py` | 128328 |
| `results/ref/all_arms_300_339.json` | per-controller means, t-intervals and paired differences over the 40 test topologies | this branch; arm tstar_tu1: later | `server_scripts/all_arms_300_339.py` | 132109 |
| `results/ref/all_arms_300_339.md` | the same summary as a table | this branch; arm tstar_tu1: later | `server_scripts/all_arms_300_339.py` | 7593 |
| `results/ref/r3lite_cf_report_full.json` | the statistical report: paired differences, Wilcoxon signed-rank tests, Holm correction, t-intervals | this branch | `confirmatory_analysis.py` | 444035 |
| `results/ref/settling.json` | per-topology settling times of every controller (640 runs) | this branch |  | 1357928 |
| `results/ref/comparator_timing.json` | times after the event of the first delivery, relay link and bridge of every controller, and of the Scenario B surge | this branch |  | 14414 |
| `results/ref/eh_summary.txt` | per-controller summary of further measures (islands, relay links, energy, hops, outage); only the first two blocks of this file (Scenarios A and B of the test, 480 records) are used by the article; the other blocks have no recorded provenance | this branch (first two blocks) |  | 39326 |
| `results/ref/scenario_lines_300_339.txt` | the scenario builder's line for each test topology (islands, cut links, cross-island flows) | this branch |  | 7797 |
| `results/ref/bh_results.json` | the backhaul test: MARL and XL-DET with the microwave and mesh hops at their own capacity (flag on) and as in the test (flag off), per topology | flag on: later + option; flag off: this branch |  | 285887 |
| `results/timelines_r3.json` | binned time courses of every controller in both scenarios | this branch | `server_scripts/timelines_r3.py` | 742532 |
| `results/training_curves.json` | per-episode training record of the four trained policies | r2 (training) |  | 353902 |
| `results/ref/learning_kpis_s1234.json` | per-episode training log of the evaluated policy (400 episodes) | r2 (training) | the training code of branch r2 | 201437 |
| `results/ref/train_summary_s1234.json` | training summary of the evaluated policy | r2 (training) |  | 7031 |
| `results/ref/train_ckpt_identity.json` | sha256 of the evaluated policy and of the final policy of its training run (identical) | r2 (training) |  | 734 |
| `results/endstate_ck1234_seed58.json` | end state of the evaluated policy on development topology 58, the article's example | later | `server_scripts/dump_endstate_v11.py` | 124803 |
| `results/series_ck1234_seed58.json` | per-tick series of the evaluated policy on topology 58 | later |  | 125584 |
| `results/geom_seed58_r4tree.json` | scenario geometry of topology 58 | later | `server_scripts/geom_seed58.py` | 24888 |
| `results/identity_seed58.json` | record that the pass on topology 58 reproduced, bit for bit, the pass of the same policy made with the code of branch r2 | later vs r2 | `server_scripts/identity_seed58.py` | 3117 |
| `results/logs/all_arms.log` | log of the summary extraction | this branch; arm tstar_tu1: later | `server_scripts/all_arms_300_339.py` | 8995 |
| `results/logs/timelines.log` | log of the time-course extraction | this branch | `server_scripts/timelines_r3.py` | 773 |
| `results/logs/geom_seed58.log` | log of the geometry extraction | later vs r2 | `server_scripts/geom_seed58.py` | 426 |
| `results/logs/endstate.env` | environment the end-state dump was launched with | later |  | 865 |
| `results/logs/endstate_filtered.log` | filtered log of the end-state dump | later | `server_scripts/dump_endstate_v11.py` | 31534 |
| `results/logs/identity.log` | log of the bit-identity check | later vs r2 | `server_scripts/identity_seed58.py` | 607 |
| `results/server_scripts/all_arms_300_339.py` | the extraction script named in the column "written by" | -- |  | 38249 |
| `results/server_scripts/dump_endstate_v11.py` | the extraction script named in the column "written by" | -- |  | 5338 |
| `results/server_scripts/geom_seed58.py` | the extraction script named in the column "written by" | -- |  | 4046 |
| `results/server_scripts/identity_seed58.py` | the extraction script named in the column "written by" | -- |  | 6537 |
| `results/server_scripts/timelines_r3.py` | the extraction script named in the column "written by" | -- |  | 15043 |

## Checking the code tree

The 208 files of the code tree (everything except `README.md`, `policies/`, `results/` and
`records/`) are those of `c6c5de5`. Their SHA-256 list, `records/FREEZE_c6c5de5.sha256`, is
byte-identical to `records/R3/FREEZE_R3_c6c5de5.sha256` on branch `r3lite-prereg` (tag
`prereg-R3lite`), which GitHub's activity log of this repository shows created at
2026-10-13 11:10:37 UTC; the first block of test runs started at 12:19:06 UTC the same day
(`results/ref/settling.json`, `provenance.pickles`). From a clone (`.gitattributes` is
`* -text`, so no line ending is changed):

    sha256sum -c records/FREEZE_c6c5de5.sha256 2>/dev/null | grep -c ": OK"   # 183

The other 25 files differ from their listed digest only because host names and path
strings (including user names in paths) were replaced by placeholders (`<compute-server>`, `<local-bin>`, `<local-repo>`, `<local-workdir>`, `<server-home>`, `<server-tmp>`).
Because the engine-pin list `confirmatory/ENGINE_PIN_6b53f3f.txt` records the blob ids of
the unreplaced files, the engine-pin check (`confirmatory/chain_plan.py --engine-pin-check`,
and the test `test_no_engine_file_differs_from_r2`) reports `sixg_sim/patch_agent.py`, `sixg_sim/refactor_sim.py`, `sixg_sim/refactor_sim_again.py` as different
in this published copy; it passes on the original tree.

| file | listed sha256 | published sha256 |
|---|---|---|
| `docs/COMPUTE_PLAN.md` | `51ed84d9f2aca979` | `d9828e25988da565` |
| `docs/DECISIONS.md` | `37a1c0ecadec3994` | `49ea074278276212` |
| `docs/FREEZE_CHECKLIST.md` | `56e190df011100b1` | `10fea675f32b0ec7` |
| `docs/POWER_R3LITE.md` | `ce87156138d5acf2` | `ff5505935829dea2` |
| `docs/PREREGISTRATION_R2.md` | `c72f22ea4e46a2ac` | `a33152463556e49f` |
| `docs/RERUN_SPEC.md` | `50dfa373fed733d3` | `58bb27b136dcac1b` |
| `docs/RUN_CHAIN.md` | `9d4ee8667f43c2a6` | `cea0c38af44292f3` |
| `docs/SEED_AUDIT_200_219.md` | `895fa4b3a9f0390b` | `78d29d6f590d1f6a` |
| `docs/WP9_ANALYSIS.md` | `6cf68f1ad8f8c2f8` | `196c0f704649d248` |
| `docs/wp12/wp12_sdn_probe_report.txt` | `a4d606ccc5a7f927` | `f3c3a0c66c610bad` |
| `docs/wp12/wp12_ue_transit_report.txt` | `ebb12fbdfc89a886` | `fc7a168005877bcf` |
| `sixg_sim/patch_agent.py` | `8910dc5c181111e8` | `f694c5f06c749803` |
| `sixg_sim/refactor_sim.py` | `c6b8df9ce39919e6` | `7e4d263d451479fd` |
| `sixg_sim/refactor_sim_again.py` | `663c6c1b49d5af3e` | `4441314faa978b31` |
| `tests/test_r3lite_plan.py` | `bdaff097e0e3cce0` | `6ed80df6e180568b` |
| `tests/test_run_chain.py` | `705d9b5f1e1a29ba` | `63a316f728118362` |
| `tests/test_wp2_parity.py` | `6ad0cf2bce0ffbd5` | `62b235a276f868e6` |
| `tests/test_wp6_smoke.py` | `ed52d195b3753908` | `5c949a9fc8fb7511` |
| `tests/test_wp7_controls.py` | `3f17e89ca7845412` | `279f3f4d6b2828c8` |
| `tests/test_wpdet_determinism.py` | `ea327790e7278432` | `67262f22e583632e` |
| `tools/constancy_dump.py` | `da4e64c0db748bc7` | `27c50af8cc7c1b77` |
| `tools/det_gate.sh` | `5a586ee626a05a13` | `ae5a0a468a47655f` |
| `tools/reward_term_dump.py` | `dfc718b575260eab` | `4117a2fce868ea5d` |
| `tools/run_seed_audit.sh` | `8fc4d2cd4209323b` | `86e9362ca6df313b` |
| `tools/wp9_validate.sh` | `fd555c030e083dbd` | `7ede63efe2c4992b` |

Result files with the same kind of replacements (also `<server-data>`, `LOGNAME=<user>`, `USER=<user>`): `results/per_seed_300_339.json`, `results/ref/all_arms_300_339.json`, `results/ref/all_arms_300_339.md`, `results/ref/bh_results.json`, `results/timelines_r3.json`, `results/ref/train_summary_s1234.json`, `results/ref/train_ckpt_identity.json`, `results/endstate_ck1234_seed58.json`, `results/series_ck1234_seed58.json`, `results/geom_seed58_r4tree.json`, `results/identity_seed58.json`, `results/logs/all_arms.log`, `results/logs/geom_seed58.log`, `results/logs/endstate.env`, `results/logs/endstate_filtered.log`, `results/server_scripts/all_arms_300_339.py`, `results/server_scripts/timelines_r3.py`.

No number, algorithm or result value was changed by these replacements.

## Checking the policies

Each policy file stores, next to its weights, metadata of the training run (code commit,
arguments, library versions). In that metadata the host name and the user name in a path
were masked with `x` characters, keeping their length, and the CRC-32 of the member
`data.pkl` was updated; the weights and every other byte are unchanged. The metadata's
`tree_head` `6b53f3f` is the code published on branch `r2`. The recorded sha256 of the
evaluated policy is the one in `confirmatory/SELECTED_r3lite.txt` and
`results/ref/train_ckpt_identity.json`.

| file | recorded sha256 | published sha256 |
|---|---|---|
| `policies/ck_s1234/policy_final.pt` | `54eaa5623148cfd4` | `43f3536db9985fef` |
| `policies/ck_s2345/policy_final.pt` | `c86ef14f051a9287` | `bfa0439d04230a16` |
| `policies/ck_s3456/policy_final.pt` | `53e927f87643835b` | `3579f83e405d3c33` |
| `policies/ck_s4567/policy_final.pt` | `7e640ce090b4a414` | `c4a023a54324db99` |
