# MARL-RIC: multi-agent reinforcement learning for 6G mesh recovery

This repository holds the simulator, training and evaluation code, the
evaluated checkpoints and the confirmatory-protocol records behind the
manuscript *Multi-Agent Reinforcement Learning for Autonomous 6G Mesh Network
Recovery After Catastrophic Infrastructure Failure* (UNITY-6G, Work
Package 4).

The tag **`confirmatory-A1`** marks the tree that was evaluated in the
manuscript's confirmatory test (Amendment 1, frozen executor). Everything at
the repository root is that frozen tree, and every file listed in
`records/FREEZE_A1.sha256` matches its digest.

## Layout

| Path | Contents |
|---|---|
| `sixg_sim/`, `config/` | Simulator: topology, propagation, transport, agents, MAPPO trainer, baseline protocols |
| `run_timeline_comparison.py` | Evaluation harness (all arms, both scenarios), as amended by Amendment 1 |
| `train_on_comparison.py` | Training driver used for the four replicates |
| `ce_ck_s1234/` ... `ce_ck_s4567/` | Trained shared-actor checkpoints; `ce_ck_s1234/policy_final.pt` is the evaluated one |
| `benchmark_manifest*.json`, `build_benchmark_manifest.py` | Instance manifests (validation, development, confirmatory seeds 80–99) |
| `confirmatory_analysis.py`, `A1_sensitivity.py`, `extract_timelines.py`, `dump_results_json.py` | Pre-registered analysis and result extraction |
| `marl_static.diff` | The frozen-executor code change of Amendment 1 |
| `original_harness/` | Harness and analysis script as frozen before Amendment 1 (first freeze list) |
| `confirmatory/` | Run scripts, analysis reports and logs; `confirmatory/results/*.json.gz` hold the per-tick result series (confirmatory set, development set, development stress test) |
| `records/` | Confirmatory-protocol records: pre-registration, Amendment 1, both freeze lists with SHA-256 digests and time stamps, the confirmatory results report, and the specifications and scripts of the two post hoc checks (untrained-actor control, stale relay-edge diagnostic) |
| `paper_scripts/` | Scripts that regenerate the manuscript's numbers and timeline/process figures, with their output sheets |
| `test_*.py`, `smoke*.sh` | Unit and smoke tests |
| `archived/2026-05/` | The earlier project state (May 2026), kept for reference; also tagged `archive-2026-05` |

## Verifying the frozen tree

```bash
sha256sum -c records/FREEZE_A1.sha256
```

Run it from the repository root. The first freeze list,
`records/FREEZE.sha256`, was written for the pre-amendment tree. Its
harness and analysis script are kept in `original_harness/`, and its
pre-registration and manifest entries are in `records/`.

## Reproducing an evaluation

The harness runs one or more arms over a seed list and writes a pickle to
`output/`. The engine and policy settings used for every reported run are
those in `confirmatory/run_amendment1.sh`. That script was written for the
evaluation server and changes into `~/MARL_run_6`, so adapt that line
before using it. A single run with the same settings looks like this:

```bash
mkdir -p output logs
export PYTHONHASHSEED=0 MARL_STEER_MODEL=actionable
env MARL_TRANSPORT_POWER=planning MARL_RELAY_STATE=reconcile \
    MARL_RELAY_REPOINT=fragment_aware MARL_POSTCARD_ALWAYS=1 \
    MARL_REWARD_PROFILE=mission MARL_POSTCARD_FRAGMENTS=1 MARL_RELAY_ENTROPY_FLOOR=0.01 \
    MARL_EVAL_CHECKPOINT=ce_ck_s1234/policy_final.pt \
    python3 run_timeline_comparison.py --seed-list 80-89 --arms marl_static,xldet \
        --no-figures --out example_80_89.pkl
python3 confirmatory_analysis.py --pickles output/example_80_89.pkl \
    --manifests benchmark_manifest_80_99.json --marl-arm marl_static --out example_report.json
```

The pre-registered analysis needs every arm on all twenty confirmatory
seeds. A full confirmatory run takes several hours on a multi-core server.

## Implementation audit

The manuscript's implementation audit discloses departures of this code from
the system as designed. They include a control-channel send counter that is
never reset (one postcard per site per episode), simulator-computed fragment
labels, an idealised link-state forwarding service, a modelled overhead
proxy, bridge flapping and stale relay edges. They were left in place, so
that the frozen tree is the evaluated one. `records/stale_edge_diagnostic/`
contains the flag-gated patch used to measure the effect of the stale-edge
defect.
