# R2 -- the corrected, pre-registered rerun (confirmatory seeds 200-219)

This branch holds the code that ran the R2 confirmatory test reported in
version 9 of "Multi-Agent Reinforcement Learning for Autonomous 6G Mesh
Network Recovery After Catastrophic Infrastructure Failure", and its records.
The first study (seeds 80-99, tag `confirmatory-A1`) is on `main`.

* Frozen code head: `6b53f3f` (the run-chain repository), 199 tracked files.
  `records/R2/FREEZE_R2_6b53f3f.sha256` lists the SHA-256 of every one of them
  as frozen before the run (written 2026-10-04 20:55:15 UTC).
* The chain ran 2026-10-04 22:38 UTC to 2026-10-06 13:05 UTC: 37 of 37
  invocations ended OK, no restarts, in two segments with the freeze-2 seed
  audit between them (`records/R2/freeze2_6b53f3f.txt`, `seed_audit_200_219.json`).
* Results: `records/R2/final_cf_report_full.txt` (confirmatory, primary and
  secondary analyses), `final_dev_report.txt` (development set), ledger
  `chain_real.jsonl`, selection `SELECTED_real.txt`.

## Redactions

Internal infrastructure names and local paths were replaced by placeholders
(`<compute-server>`, `<server-home>`, `<local-home>`, `<local-repo>`,
`<local-workdir>`, `<local-session>`, `<local-bin>`, `<cloud-folder>`) in the
files below. These files therefore do NOT match their FREEZE_R2 digest; every
other file matches it byte for byte (`.gitattributes` is `* -text`):

    sha256sum -c records/R2/FREEZE_R2_6b53f3f.sha256 2>/dev/null | grep -c OK   # 176

| file | frozen sha256 | published sha256 |
|---|---|---|
| `docs/COMPUTE_PLAN.md` | `51ed84d9f2aca979` | `d9828e25988da565` |
| `docs/DECISIONS.md` | `cc6a0e46cd25cdd8` | `5d8504d9274fe04e` |
| `docs/FREEZE_CHECKLIST.md` | `7a95277ce3e7d2b5` | `bdd3b30c900d3656` |
| `docs/PREREGISTRATION_R2.md` | `c72f22ea4e46a2ac` | `a33152463556e49f` |
| `docs/RERUN_SPEC.md` | `50dfa373fed733d3` | `58bb27b136dcac1b` |
| `docs/RUN_CHAIN.md` | `a3c0a8b5022c767a` | `8948272ba3532c00` |
| `docs/SEED_AUDIT_200_219.md` | `895fa4b3a9f0390b` | `78d29d6f590d1f6a` |
| `docs/WP9_ANALYSIS.md` | `6cf68f1ad8f8c2f8` | `196c0f704649d248` |
| `docs/wp12/wp12_sdn_probe_report.txt` | `a4d606ccc5a7f927` | `f3c3a0c66c610bad` |
| `docs/wp12/wp12_ue_transit_report.txt` | `ebb12fbdfc89a886` | `fc7a168005877bcf` |
| `sixg_sim/patch_agent.py` | `8910dc5c181111e8` | `f694c5f06c749803` |
| `sixg_sim/refactor_sim.py` | `c6b8df9ce39919e6` | `7e4d263d451479fd` |
| `sixg_sim/refactor_sim_again.py` | `663c6c1b49d5af3e` | `4441314faa978b31` |
| `tests/test_run_chain.py` | `88b603c41ab11f5a` | `70aefce17f94572b` |
| `tests/test_wp2_parity.py` | `6ad0cf2bce0ffbd5` | `62b235a276f868e6` |
| `tests/test_wp6_smoke.py` | `ed52d195b3753908` | `5c949a9fc8fb7511` |
| `tests/test_wp7_controls.py` | `3f17e89ca7845412` | `279f3f4d6b2828c8` |
| `tests/test_wpdet_determinism.py` | `ea327790e7278432` | `67262f22e583632e` |
| `tools/constancy_dump.py` | `da4e64c0db748bc7` | `27c50af8cc7c1b77` |
| `tools/det_gate.sh` | `5a586ee626a05a13` | `ae5a0a468a47655f` |
| `tools/reward_term_dump.py` | `dfc718b575260eab` | `4117a2fce868ea5d` |
| `tools/run_seed_audit.sh` | `1b04b0e0aa988c53` | `51c167c29a311e8d` |
| `tools/wp9_validate.sh` | `fd555c030e083dbd` | `7ede63efe2c4992b` |

Redacted record files (paths only): `FREEZE_FINAL_real.txt`, `chain_real.jsonl`, `freeze2_6b53f3f.txt`, `seed_audit_200_219.json`

No code path, constant or result changed: the replacements touch only host
names and filesystem paths in documentation, scripts and records.
