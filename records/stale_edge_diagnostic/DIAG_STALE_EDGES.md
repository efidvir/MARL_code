# Stale relay-edge defect: a post-hoc diagnostic

**Status.** This check is written, hashed and committed to the Overleaf
`records/` history before any full-length run. It is post hoc. It changes no
pre-registered hypothesis, test or decision. It is reported whatever it shows.

## The defect

The frozen simulator (`~/MARL_run_6`) tears down transport relay links in two
places: `sixg_sim/simulation.py`, around lines 5556 and 6432. Both remove the
`Link` object. Both then try to remove the routing-graph edge by splitting the
link id `TR_<a>_<b>` on the first underscore. Node ids themselves contain
underscores (`O-DU_18`), so that split never recovers the endpoints, and the
edge survives.

`consume_path` treats an edge whose link object no longer exists as a hop
with no capacity constraint (`if link is None: continue`). A torn-down relay
link can therefore still carry traffic, without limit. The fragment and
component metrics are computed from the link objects, so they do not see it.

Only arms that create relay links are affected: the frozen executor, the
adapt-then-freeze variant, XL-DET and the uniform-random control. The routing
protocols and SDN never create relay links.

## The patch (`patch_phantom.py`, tree `~/MARL_run_8`)

The patch is applied to a copy of the frozen tree, and every change is gated
by a flag:

- `MARL_DIAG_PHANTOM=1` counts, per tick:
  - the stale edges;
  - the delivered volume that crossed at least one of them.
- `MARL_FIX_PHANTOM=1` removes the edge of a torn-down relay link by the
  link's own endpoints, unless another up link still joins the pair.

## Smoke test (disclosed; run before this file was written)

Seed 82, 900 ticks, frozen executor and XL-DET, both scenarios.

**Instrumentation leaves behaviour unchanged.** With the patch present and
both flags off, and with instrumentation only, every recorded series is
bit-identical to the unpatched code.

**Stale edges exist and carry traffic.**

| Arm | Scenario | Max stale edges | Share of delivered volume crossing a stale edge |
|---|---|---|---|
| Frozen executor | A | 15 | 14% |
| Frozen executor | B | 17 | 29% |
| XL-DET | A | 2 | 18% |
| XL-DET | B | 2 | 17% |

**With the fix, results barely change.** Last-300-tick achievability over the
900-tick run:

- frozen executor, Scenario A: 47.97 → 47.96;
- every other arm and scenario: unchanged.

## Runs

- **R-fix.** `MARL_FIX_PHANTOM=1` and `MARL_DIAG_PHANTOM=1`.
  - Arms: `marl_static`, `xldet`, `random`.
  - Instances: confirmatory seeds 80–99, both scenarios, in batches of four
    seeds.
  - Same engine and policy environment as the Amendment-1 runs, with
    `PYTHONHASHSEED=0`.
  - Checkpoint: `ce_ck_s1234`.
- **R-diag.** `MARL_DIAG_PHANTOM=1` only, same arms and instances. Its
  decisions reproduce the frozen runs, and it measures stale-edge use in
  them.

## Analysis (post hoc, descriptive)

1. Recompute the pre-registered analysis (`confirmatory_analysis.py
   --marl-arm marl_static`) with the three affected arms taken from R-fix and
   every other arm from the original job. Report P1–P9 and S1–S6 under the
   pre-registered rule, labelled as a post-hoc sensitivity analysis.
2. For each affected arm, report paired differences between R-fix and the
   frozen runs: achievability, components, bridges and energy per unit.
3. From R-diag, per arm and scenario, report:
   - the mean number of stale edges;
   - the share of delivered user-to-user volume that crossed one.
4. Nothing in the pre-registered confirmatory results is replaced. The
   manuscript reports this analysis as a disclosed defect with its measured
   effect.
