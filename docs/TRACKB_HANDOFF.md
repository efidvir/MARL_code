# Track B → engine track handoff

Engine-side recording that the Track B analysis (WP9) reads.  Track B made no
engine change; every item here is a request to the engine track (WP1/WP3/WP4/
WP5 owners).  The names below are the contract the analysis, the export and
the timeline scripts already implement (`confirmatory_analysis.py` constants
`EVENT_SERIES`, `DIAG_SERIES`, `LIFETIMES_KEY`, `REMOVED_BY_REASON_KEY`,
`HOLDS_KEY`, `OVERHEAD_BY_TYPE_KEY`, `ENERGY_CTRL_KEY`, `PROVENANCE_FIELDS`).
A key that is absent is reported as "not recorded"; a key under another name
is invisible to the analysis.

## 1. Per-run keys in the result pickle (`_pass_out`)

All per-tick series are plain Python lists of length `TOTAL_TICKS`, index =
tick, value = the count **in that tick** (per-tick difference of the
cumulative simulator counter, never the cumulative value and never a counter
that needs a reset).  Numeric types: Python int/float, numpy arrays and
numpy scalars (`np.int64`, `np.float64`, …) are all accepted and counted in
full (`confirmatory_analysis.is_num`); the analysis converts them to plain
Python numbers.  Any NaN, None or non-numeric entry in any per-tick series
— the `{name: series}` members included — stops the analysis (hard assert),
except NaN in `latency`.

| key | type | meaning |
|---|---|---|
| `relay_created` | list[int] | relay links created this tick (both creation sites; Priority-2 links born up count here too) |
| `relay_removed` | list[int] | relay links torn down this tick, all reasons (M7) |
| `relay_removed_by_reason` | dict[str, list[int]] | the same split by reason, e.g. `mode_change`, `repoint`, `node_down` (sums to `relay_removed`) |
| `relay_teardown_holds` | dict[str, list[int]] | teardowns blocked this tick, by reason: `lockout`, `grace`, `not_installed`, `busy` (RERUN_SPEC WP4) |
| `bridge_teardowns` | list[int] | teardowns of links that were bridges (`rid in _relay_bridge_since`) (M8) |
| `idle_bridge_teardowns` | list[int] | bridge teardowns with usable passes == 0 (M8) |
| `relay_mode_flips` | list[int] | sites whose **executed** relay mode differs from the previous tick (M9); same definition as the reward's churn count but computed independently of the reward call |
| `relay_lifetimes` | list[dict] | one record per relay link ever created, see §2 (M11) |
| `stale_edges` | list[int] | WP1 invariant counter per tick; must be 0 (hard assert) |
| `stale_u2u_volume` | list[float] | WP1 counter per tick; must be 0 (hard assert) |
| `postcards` | list[int] | postcards **sent** this tick (already recorded); aggregate assert Σ after the event > `n_radio_sites` for MARL-path arms |
| `n_radio_sites` | int | already recorded; the aggregate postcard assert's denominator |
| `postcards_by_site` | dict[str, int] | **requested (WP5):** postcards sent after the event (t ≥ severance tick) per agent site, one entry for every agent site (0 for a site that never sent); the analysis asserts min over sites > 1 — the per-site condition of RERUN_SPEC WP5/WP9.  Alternative: the scalar `postcards_min_per_site` (int) written by the WP5 post-pass invariant.  Without either the analysis can only check the aggregate mean per site and labels the check "aggregate" |
| `decision_digest` | list[str] | **requested (S8):** per tick, the sha256 hex digest of the **executed** joint action of the MARL-path arms (marl_static, marl_freeze, xldet, random, controls), computed where `Simulator._execute_agent_actions` fills `last_action_summary` (WP2 `step_tick` step 5, after engine holds): canonical form = the tuple, sorted by site id, of `(site_id, tx_power_index, mcs_offset_emergency_index, mcs_offset_general_index, executed_relay_mode, prb_split rounded to 6 decimals)` joined with `\n` and hashed; the pre-event locked configuration digests too.  This is what makes S8 ("identical decision series under PYTHONHASHSEED 0 and 1") a check of the decisions rather than of their consequences only; `tools/hash_invariance_check.py --require-decision` fails a MARL-path run without it.  Cost: one small hash per tick; the harness may write it only for the two S8 instances if the pickle size matters, but the key must then be absent (not empty) elsewhere |
| `overhead` | list[float] | billed bytes per tick (already recorded; now the sum of the billed types) |
| `overhead_by_type` | dict[str, list[float]] | billed bytes per tick by message type: `bfd`, `hello`, `lsa`, `postcard` for the MARL-path arms; the baselines' own types (S11) |
| `energy_ctrl` | list[float] | control-plane energy term per tick (share of energy; P7–P9 insensitivity) |

Routing arms never create relay links, so their event series are all-zero
lists (not absent): absence means "not recorded" in every report.

## 2. `relay_lifetimes` record

Preferred: a dict per link with the keys
`link_id, created, up, removed, cause, was_bridge, load`
(`created`/`up`/`removed` = ticks, `removed` = None while the link is still
up at the end of the run, `cause` = the teardown reason string or None,
`was_bridge` = bool, `load` = the relay-link load fraction at teardown).
A tuple in exactly this order is accepted (`lifetime_record()`), but dicts
survive later field additions.  M11 uses `removed − up` (falls back to
`created` when `up` is None) over `was_bridge` records, right-censors links
still up at T (`removed` must then be None, never T) and reports the
Kaplan–Meier median (PREREGISTRATION_R2 §7.2); the engine only has to get
the censoring right — a link torn down on the last tick has `removed` set.

## 3. Provenance (top-level `provenance` key, D0.2)

The analysis requires `tree_head`, `tree_sha256`, `schema_id`,
`pythonhashseed` to be **present** (None or empty fails, also when it is
missing from every pickle) and identical across every pickle of one analysis
(baselines, MARL arm, controls), `pythonhashseed == '0'`,
`provenance_complete` not False, and, when present, `total_ticks == 3600`
and `severance_tick == 200`.  `schema_id` must therefore be the WP6 hash in
every pickle (the placeholder `placeholder-pre-WP6` is accepted only as long
as it is identical everywhere).  A pickle written from a copied export
without `.synced_head`/`.synced_files` (INCOMPLETE record) or by a harness
that never recorded `schema_id` is refused.  Untrained-control pickles must
come from the same tree digest: the untrained checkpoints must not be
tracked files that change the digest (D0.2 excludes only
`*_adapted_recovery*`).

`checkpoint_sha256` (already written by the harness: one sha256 per
`--checkpoint` path) must be present and identical across every pickle that
carries the tested MARL arm — the five four-seed batches of `marl_static`
must be launched with the same checkpoint path list; controls are not
compared (they run their own checkpoints).  When `marl_freeze` is present in
any pickle it must be complete (every manifest seed, both scenarios) like
every other arm.

## 4. Latency NaN

The frozen pickles hold NaN in `latency` on ticks without delivery (routing
arms, hundreds of ticks per run).  The hard assert "no NaN series" exempts
`latency` (`NAN_EXEMPT`) and reports the counts; if the engine track prefers
a defined value (e.g. None) the exemption can be dropped, but a NaN in any
other series stops the analysis.

## 5. What the asserts and S8 still need from the engine

The WP9 asserts on arms/seeds, manifest severance tick, run length and the
checkpoint identity use what the harness already writes.  Two items of §1
are new requests:

- **`postcards_by_site`** (or `postcards_min_per_site`): without it the
  postcard assert is the aggregate mean per site and says so; the per-site
  condition of RERUN_SPEC WP5 then rests on the engine-side invariant alone.
- **`decision_digest`**: hash invariance (S8) is run on two ordinary pickles
  of the selected checkpoint under PYTHONHASHSEED 0 and 1 on two instances
  (`tools/hash_invariance_check.py --require-decision`).  The harness records
  no per-tick action series, so without the digest the tool compares decision
  **consequences** only (fragments, delivered, relay series, …) and reports
  energy/latency as jitter; a hash-order-dependent tx-power or MCS decision
  whose only trace is in energy would pass.  With the digest present the
  check is of the decisions themselves.  The tool's result states which of
  the two it was (`decision_series.coverage`).
- **`pythonhashseed` in the provenance of BOTH S8 pickles** (REQUEST; no
  engine change is made here).  §3 already requires `pythonhashseed` to be
  present and `'0'` in the pickles of the *analysis*.  The S8 pair is the
  exception: its two pickles are run under 0 and 1 and each must record the
  seed it actually ran under.  Since round 3 the differing-seed precondition
  is part of the S8 verdict, so a pair whose pickles record equal seeds, or
  where either provenance record or either `pythonhashseed` is missing, is
  refused with `precondition_not_established` and exit 2 — it is a
  repeat-run pair and no number of agreeing series makes it an S8 result
  (WP9_ANALYSIS §12, D1; PREREGISTRATION_R2 §10).  The frozen A1 pickles
  carry no provenance record at all, which is why the S8 smoke step of
  `tools/wp9_validate.sh` now reports exit 2 by design: it exercises the
  plumbing, it cannot produce the study's S8 result.

## 6. Documentation track (WPDOC): items the pre-registration needs from the engine track

`docs/PREREGISTRATION_R2.md` keeps a ⟨…⟩ placeholder for everything that
depends on engine code. Each is filled from the frozen tree at freeze 1; the
owner is the WP that defines the value.

| placeholder (section) | source in the frozen tree | owner |
|---|---|---|
| `SCHEMA_ID`, tree digest, HEAD (§2) | `sixg_sim.agent.SCHEMA_ID`; `runinfo.tree_digest()`; commit | WP6 / WP13 |
| `MIN_LS_INTERVAL`, `FLOOD_HOP_TICKS`, `SPF_DELAY_TICKS` (§2.3) | `sixg_sim/link_state_service.py` constants (copied from `OSPF_MIN_LS_INTERVAL` = 5, `OSPF_FLOOD_HOP_TICKS` = 1, `OSPF_SPF_DELAY` = 5 of the baseline model) | WP3 |
| `BFD_DETECT_TICKS` = 3, `RELAY_BRIDGE_GRACE_PASSES` = 3, `RELAY_BRIDGE_BUSY_UTIL` = 0.02, `POSTCARD_PERIOD` = 10 (§2.3–2.5) | hard-wired constants; confirm names and values | WP3 / WP4 / WP5 |
| `N_ACTOR`, `N_CRITIC_GLOBAL`, the two field lists (§2.6) | `ACTOR_OBS_FIELDS`, `CRITIC_GLOBAL_FIELDS` after the constancy dump | WP6 / WP13 |
| removed reward terms (§2.8) | constancy dump | WP8 / WP13 |
| WP12(a) UE-transit decision, WP12(b) SDN decision (§2.9–2.10) | `diag/` decision notes | WP12 |
| training determinism (§3, D0.8) | decision: sort rollouts by episode index, or leave; record either way | WP2 / engine |
| sensitivity switches: installation off, re-pointing off (§6) | the two evaluation-only switches, their names, defaults (= evaluated behaviour) and their presence in provenance `marl_env` | WP3 / WP5 |
| §12 table, last column | `grep -rhoE "MARL_[A-Z0-9_]+"` over the frozen tree: which variables survive, which are removed (`MARL_OBS_LOCALITY`, `MARL_POSTCARD_FRAGMENTS`, `MARL_PRB_POLICY`, `MARL_DETECT_TICKS`, `MARL_POSTCARD_ALWAYS` are expected to go); the launcher must clear every other `MARL_*` variable (as `tools/run_invariance_pass.sh` does) | every WP that removes a switch |
| `severance_tick`, `total_ticks` in provenance (§10 asserts) | already requested in §3 above | WP0 / harness |
| `COMPUTE_PLAN.md` ⟨perf⟩ cells | per-pass and per-episode CPU, RSS, batch wall time after the performance work | performance track |

The seed audit (`docs/SEED_AUDIT_200_219.md`, `tools/seed_audit.py`) needs
nothing from the engine; re-run it from the frozen tree at freeze 1 and at
freeze 2 with the committed driver `tools/run_seed_audit.sh` (that
document's §2.1; `--false-positive "seed 42, 200 island ticks"` is the one
classified docstring; tick loops such as `for tick in range(1, 361)` and
this study's planning documents are classified by the tool itself) so that
the pre-registration's §5 evidence carries the frozen tree's date. If an engine work package adds a script or
log line that names a seed of 200–219 for any reason other than the study
itself, the audit turns TOUCHED and the line must be removed or classified
before freeze 1.

Two things the engine track should know about that audit. (1) It scans
**every** directory matching `$HOME/MARL*` and refuses (exit 4, verdict
INCOMPLETE) when one of them is not among the roots, so a new tree —
`~/MARL_run_9c` was created on 2026-09-23 — is covered automatically by the
`~/MARL_run_*` pattern of §2.1 but must not be excluded by hand. (2) It also
refuses when any file vanishes between the directory walk and the scan: the
220–239 run of 2026-09-23 08:39 UTC was refused because four checkpoint
files under `~/MARL_run_9c` disappeared while that tree was being written.
**Please do not write into any `~/MARL*` tree while a freeze audit is
running** (about 12 min per range); a run that meets a vanishing file has
to be repeated. (3) Since 2026-09-23 it also refuses a root that holds **no
file at all**, and a root whose file count is below its floor in
`tools/seed_audit_floors.txt` — the aggregate floors could not see a single
tree that lost its content. The floor for `~/MARL_run_9c` is deliberately
held at 100 while the engine track rewrites that tree (it went 239 → 138
→ 145 during 2026-09-23); every floor is re-set from the frozen counts at
freeze 1, so please say when that tree stops moving. (4) Since 2026-09-23 a
seed of the window found in a **torch checkpoint's metadata** refuses the
verdict as well; until then it was written into the report and left out of
the verdict, so such a checkpoint passed with exit 0 (record §2 step 5). If
an engine work package writes a `topology_seed` into a checkpoint, the same
rule as for scripts and logs applies. (5) Do not leave any diagnostic
output inside a `~/MARL*` tree under a name that does not contain
`seed_audit`: the audit reads it as ordinary result data. Write it under
`~/MARL_run_9b/audit_history/` (excluded from the walk) or outside the
trees.

## 7. WP12 engine requests (decisions D12.1 and D12.2, `docs/DECISIONS.md`)

Both are pre-freeze engine changes that Track B may not make. Evidence:
`docs/wp12/` (probe reports) and the probes themselves (`diag/`), which
also serve as the acceptance tests below.

### 7.1 D12.1 — data-plane paths must not transit UE nodes (all arms)

Where (baseline `d94a936` line numbers; `sixg_sim/simulation.py` unless
stated):

| site | today | requested |
|---|---|---|
| `_forward_traffic`, UE-to-UE path, 7275 | `nx.shortest_path(self.topology.graph, source_ue, target_ue)` over the UE-inclusive graph | the same call on a **path view** in which UE nodes may appear only as the flow's endpoints: `nx.subgraph_view(G, filter_node=lambda n: n in (source_ue, target_ue) or not self.topology.is_ue_node(n))`, `G` being the graph in use (true graph, or the service's installed view when it is swapped in) |
| alternates, 7290 | `nx.shortest_simple_paths(self.topology.graph, ...)` | the same path view |
| general-traffic sink path `_best_path_to_any`, 7360 (and `has_path` there) | `nx.shortest_path(self.topology.graph, src, sink)` | a view in which UE nodes other than `src` are removed |
| harness latency / hop metrics, `run_timeline_comparison.py` ~6031-6035 | `nx.shortest_path(sim.topology.graph, src, tgt)` | the same path view (one helper, shared) |
| `_can_route_ue_to_ue`, 2301 | infrastructure-only reachability gate | unchanged (it is already the leaf model) |

Rules: one helper (`topology.path_view(endpoints)` or a module function)
used by every site, so all arms select on the same view; `consume_path`
unchanged (it walks whatever path it is given); the WP3 service's installed
view is filtered the same way when it replaces `topology.graph`. A path
between two UEs that share an O-RU stays `UE > O-RU > UE` (the O-RU is
infrastructure). A UE with no infrastructure route that the gate already
rejects is unaffected.

Why all arms: the transit is a property of the shared engine path selection
(pre-event 4.2 % / 0.8 % of UE-to-UE volume on seeds 82 / 91, identical for
marl_static and ospf), and its post-event amount is arm-dependent
(marl_static 11.1 % vs ospf 1.8 % on A/82) and phase-locked to the MARL
bridge detection window (233 vs 16 units/tick). Details in D12.1.

Expected effect and matrix: every arm's delivered volume changes (routing
and SDN included) by at most its transit share minus what re-routes over
infrastructure; the §14 row of this commit therefore requires **no arm to be
bit-identical**; regenerate `output/ref/inv_ref_*` (600-tick and 1200-tick)
from the commit after it. The change precedes the retrain (training uses
the same `_forward_traffic`). Record the decision in the pre-registration
§2.9 (draft text in place).

Acceptance (server, scratch copy, monkeypatch only, no engine edit):
`diag/run_wp12.sh ue --seeds 82,91 --ticks 1200 --arms marl_static,ospf`
must report `transit` = 0.0 and `nUE` = 0 for every pass, pre-event
transit `0.0 of N`, `mismatch` 0 and `inc` 0 (the probe's attribution
self-check still holds), and the `Phase-lock check` line for marl_static
A/82 must show `transit +0.0`. Its unit tests (`tests/test_wp12_probes.py`)
need no change.

### 7.2 D12.2 — SDN LLDP discovery must terminate, and fragments and discovery must not traverse UEs (baseline model fix)

Where: `run_timeline_comparison.py`, `SDNController._compute_fragments`
(2142-2203), `SDNController.step` (the operational-phase re-queue at
2212-2219 and the LLDP phase at 2242-2284), and the LLDP billing that reads
`discovery_frontier` at 1547-1559.

Today, two things (evidence and numbers in D12.2 and `docs/wp12/`):

1. Termination: `new_frontier = N(frontier) ∩ members`; discovery is
   complete iff `new_frontier == frontier`. This is a fixed point of the
   open-neighbourhood map, which exists only on fragments with an odd
   cycle; on a bipartite fragment the frontier alternates between the two
   colour classes for ever (87A: 160/24, 98A: 159/25). A single
   `self._discovery_round_tick` is shared by all fragments, so a stuck
   fragment starves the fragments after it (81A: the controller never
   reaches `operational`).
2. UE stitching: `_compute_fragments` BFSes over every survivor node, UEs
   included, along every up link (Uu included), and the LLDP walk follows
   the same links. The recovery cuts leave 2-4 infrastructure components
   (2-3 with >= 3 switches) on every instance of 80-99, but 9-44 UEs
   anchored in two components merge them into one 31-35 switch / 150 UE
   controller fragment on every instance (`docs/wp12/sdn_static.txt`,
   "infra-only components" column). The controller therefore discovers
   switches reachable only through a UE, programs 64-105 flows per instance
   whose endpoints have no infrastructure path (never forwarded: the
   `_can_route_ue_to_ue` gate), and bills LLDP probes from UEs (the
   frontier is mostly UEs: 160 + 24 = 34 switches + 150 UEs). Under the
   leaf model with the current termination test the failure would spread
   to 16 of 20 instances (`docs/wp12/sdn_static_infra.txt`), because the
   Uu attachments supply the odd cycles today; the two fixes land together.

Requested:

(a) Cumulative BFS discovery (the iteration the class docstring describes,
"each round discovers one more hop from the controller"), per fragment:

```
# per fragment: discovered set + current layer; per-fragment round timer
if fid not in self.discovered:
    self.discovered[fid] = {leader}
    self.discovery_frontier[fid] = {leader}       # the probing layer
    self._discovery_round_tick[fid] = tick
if tick - self._discovery_round_tick[fid] >= self.TICKS_PER_LLDP_ROUND:
    self._discovery_round_tick[fid] = tick
    layer = {peer for nid in self.discovery_frontier[fid]
             for peer in up_infra_neighbours(nid)
             if peer in self.fragment_members[fid]
             and peer not in self.discovered[fid]}
    if not layer:
        self.discovery_done[fid] = True
        discovered = self.discovered[fid]         # queue its flows (see c)
    else:
        self.discovered[fid] |= layer
        self.discovery_frontier[fid] = layer
        any_discovery_pending = True
```

`up_infra_neighbours` iterates `sorted(...)` over link endpoints as the
existing code does (hash-seed independence) and yields infrastructure
peers only (see b). The billing at 1547-1559 keeps reading
`discovery_frontier[fid]` (now the newly discovered switch layer, the set
of switches that actually probe in that round) and `discovery_done`. If
the per-fragment timer is not wanted, keep the shared one but state the
serialisation in the pre-registration §2.10 (with 2-3 controllers per
instance the fragments then discover one after the other).

(b) Fragments over infrastructure survivors only. `_compute_fragments`
walks the BFS over non-UE survivors and infrastructure links (the same
node/edge definition as `_post_severance_infra_graph`); `fragment_members`
holds the switches; each UE is attached, as a leaf, to every fragment
that holds one of its live anchors (`anchors[ue]` = the infrastructure
endpoints of its up links, exactly `_infra_neighbors` in
`Simulator._can_route_ue_to_ue`; a dual-anchored UE belongs to two
fragments and to neither's walk). The election rule is unchanged: a
fragment with >= `MIN_FRAGMENT_FOR_ELECTION` switches elects the
lexicographically first switch of maximum up-link degree, with its own
`election_done_tick` (elections already run in parallel).

(c) Flow eligibility, in both places that queue flows: a flow (src, tgt)
is queued for fragment `fid` iff `anchors[src] ∩ discovered[fid]` and
`anchors[tgt] ∩ discovered[fid]` are both non-empty — at discovery
completion (today's `src in discovered and tgt in discovered`, which no
longer works when `discovered` holds switches only) AND in the
operational-phase re-queue at 2212-2219. The latter matters now: today it
queues every not-yet-programmed flow after a forced recompute without any
discovery test, which was invisible while all 222 flows were programmed at
once, but under the leaf model it would program the 64-105 cross-component
flows blindly at 3 per tick from the first operational tick. Flows whose
endpoints gain a common fragment later (SDN's relay activation bridges two
components; the forced recompute merges them) are queued by the same test
once the merged fragment's discovery is complete, or, if the engine track
prefers not to re-discover merged fragments, once the recompute finds them
in one fragment (state the choice in §2.10).  This same re-queue is the
only place a Scenario-B pre-converged controller ever queues a flow, and
that path fills no `discovered` set at all: (c2) must land together with
this item, or Scenario B stops being bit-identical.

(c2) The Scenario-B pre-converged path must seed `discovered`, or (c)
contradicts acceptance item 2.  `discovered[fid]` is new state that only
the discovery walk of (a) creates.  The rescue_ops branch at 5341-5348
sets `phase = 'operational'`, `programmed_flows = set(sim.ue_to_ue_flows)`
and `discovery_done[fid] = True` for every fragment, and never enters that
walk: it creates no `discovery_frontier[fid]` today (the committed report
`docs/wp12/wp12_sdn_probe_report.txt` shows `frontier tail []` for every
Scenario-B fragment) and would create no `discovered[fid]` either.  An
implementer following (c) literally would evaluate `anchors[src] ∩
discovered[fid]` against an absent — i.e. empty — set for every flow that
reaches the operational-phase re-queue, and drop all of them.  That is not
hypothetical: the re-queue is the only path by which the rescue-arrival
flows are ever programmed.  On the two instrumented rescue_ops passes
(`output/wp12_sdn_probe.pkl`, tree 2ba4be7, 600 ticks, rescue arrival
rescaled to tick 300) the controller programs 222 flows at creation and
then exactly 88 more (seed 87, ticks 300-329) and 70 more (seed 98, ticks
300-323) through the re-queue at 2212-2219 — 28 % and 24 % of the final
310 / 292 programmed flows, whose delivery is the 740 509 / 579 968
post-event that acceptance item 2 requires to be bit-identical.

So, in the same loop that sets `discovery_done[fid] = True`, the
pre-converged branch must also seed that fragment's discovered set with
its switches:

```
sim._sdn_ctrl._compute_fragments()
for fid, members in sim._sdn_ctrl.fragment_members.items():
    sim._sdn_ctrl.discovery_done[fid] = True
    sim._sdn_ctrl.discovered[fid] = {          # <- the added line
        n for n in members if not sim.topology.is_ue_node(n)}
```

(the fragment's walk set under (b)) — equivalently, the eligibility test of
(c) falls back to fragment membership for a fragment marked done without a
discovery walk.  Either spelling is acceptable; state which in §2.10.  Two
riders:

  * Seed it for EVERY fragment the branch marks done, including the
    sub-`MIN_FRAGMENT_FOR_ELECTION` isolated switches that have no leader,
    and let the re-queue accept a flow whose anchors lie in any fragment
    marked done, elected or not.  Consequence (d) is a statement about a
    fragment that would have to be discovered by a walk and cannot be; it
    must not be applied to a controller that is pre-converged by
    construction, which is the modelling assumption of Scenario B (the
    control plane converged before tick 0).  On 87B / 98B the isolated
    fragment carries 0 flows, so the two readings agree there — but only
    there, and §14 requires Scenario B bit-identical on every rescue_ops
    instance, not only on the two that are instrumented.
  * The re-queue calls `_compute_fragments(force=True)` immediately before
    queueing, and `_compute_fragments` rebuilds `fragment_members` from
    scratch, numbering fids by first appearance in `topology.nodes`, while
    `discovery_frontier`, `discovery_done` and `election_done_tick` keep
    their old keys and are never cleared.  A seeded `discovered` inherits
    that hazard.  REMAP the per-fragment dicts onto the new decomposition
    inside `_compute_fragments` when `force=True`, and leave no fid in
    them that the new decomposition does not define.  Do NOT clear and
    rebuild them.  An earlier revision of this rider offered "rebuild (or
    explicitly remap)" as interchangeable spellings; they are not, and the
    first one reproduces the failure (c2) exists to prevent:

      - the tick-300 forced recompute of the operational-phase re-queue
        (`run_timeline_comparison.py:2212-2219`) is the ONLY recompute a
        Scenario-B pre-converged controller ever performs;
      - the pre-converged branch that sets `discovery_done[fid] = True`
        and seeds `discovered[fid]` runs ONCE, at controller creation
        (5343-5348, inside `if not hasattr(sim, '_sdn_ctrl')`);
      - Scenario B runs no discovery walk, so nothing refills state that
        the recompute clears -- and a rebuild clears BOTH dicts, so both
        spellings of the eligibility test offered above (the seeded
        `discovered` set, and the fallback to fragment membership for a
        fragment marked done) lose their basis;
      - the gate of (c) then tests an empty set again and drops exactly
        the 88 / 70 rescue-arrival flows of 87B / 98B, on the same flows
        and the same 740 509 / 579 968 delivery totals.

    So a forced recompute must PRESERVE the pre-converged controller's
    `discovery_done[fid]` and `discovered[fid]` by remapping them, or
    re-apply the pre-converged seeding inside
    `_compute_fragments(force=True)` when `phase == 'operational'` and no
    discovery walk has run.  One of those two; a bare rebuild is not an
    admissible implementation of this rider.  Acceptance item 2's
    `programmed` 222 -> 310 (seed 87, by tick 329) and 222 -> 292 (seed
    98, by tick 323) is the check that catches a rebuild: a rebuild
    implementation ends the pass at 222.  Today the
    renumbering happens to be stable on the rescue_ops passes, because the
    rescue UEs are appended to `topology.nodes` after every infrastructure
    node and so cannot shift an infrastructure node's first appearance —
    that is what keeps the 88 / 70 flows on fid 0.  It is an accident of
    insertion order, not an invariant, and under (b) the fragment set is
    recomputed far more often than it is today.  REQUEST (engine track):
    this stale-key hazard is a pre-existing defect of `_compute_fragments`,
    independent of the change asked for here; Track B owns no engine file
    and has not touched it.

(d) Disclosed consequence to confirm or replace: a flow whose only common
component is an isolated single switch (< `MIN_FRAGMENT_FOR_ELECTION`
switches, no election) is never programmed — today the stitching folds
that switch into the controller domain and its flows are served. This
affects 6 of the 20 recovery instances of 80-99 by 2-18 flows (82A 2, 83A
7, 85A 2, 93A 18, 97A 11, 98A 5). The default is the model's own rule (a
switch without a controller programs nothing); any other rule goes into
§2.10 before freeze 1.

The Scenario-B pre-converged path (`discovery_done[fid] = True`,
`programmed_flows = all flows`, 5341-5348) therefore changes by exactly
the one added assignment of (c2) and by nothing else.  It is NOT
unchanged: the earlier revision of this section said it was, and that
sentence contradicted acceptance item 2 below, because (c) applied to a
path that never fills `discovered` drops every rescue-arrival flow.  With
the seeding in place Scenario B stays bit-identical, which is what
acceptance item 2 checks: every Scenario-B flow has an infrastructure
path (0 of 222 without one on every rescue_ops case, and 0 of the 88 / 70
rescue-arrival flows on 87B / 98B).

Expected effect and matrix: SDN changes on every Scenario-A instance
(2-3 controllers instead of one; discovery completes after `ecc(leader)`
= 1-6 rounds instead of 8-16, so first delivery moves a few ticks
earlier; the programmed set is 99-158 flows — the 117-158 with an
infrastructure path less those under (d) — instead of 222; the LLDP
billing falls, no UE probes; keepalive/churn billing scales with the
programmed set; 87A / 98A / 52A recover; 81A becomes operational, with
the churn and keepalive billing of the other seeds). Scenario B
bit-identical; routing arms bit-identical. The §14 row of this commit:
routing identical, SDN expected to change in A; regenerate the SDN part of
`inv_ref_*_t1200.pkl` afterwards. No effect on any MARL-path arm.

Static targets (already computed by the probe under the requested model,
`diag/run_wp12.sh sdn-static --fragment-model infra-only --iteration bfs`;
`docs/wp12/sdn_static_infra_bfs.txt`, `_dev.txt`):

| instance | infrastructure components | controllers (leader, ecc, flows in fragment) | programmed before any bridge | flows without a path |
|---|---|---|---|---|
| 87A | 19, 15, 1 (all bipartite) | O-RU_23 ecc 4, 51; O-RU_25 ecc 4, 72 | 122 (one flow lies in both) | 100 |
| 98A | 15, 18, 1, 1 (all bipartite) | O-RU_28 ecc 5, 33; O-RU_29 ecc 3, 89 | 121 (5 more served only by the isolated switch with 31 UEs) | 96 |
| 82A | 15, 18, 1, 1 | O-RU_27 ecc 3, 70; O-RU_28 ecc 4, 67 | 132 | 88 |
| 81A | 14, 17, 3, 1 | O-RU_25 ecc 3, 98; O-RU_29 ecc 3, 48; EdgeUPF_39 ecc 1, 0 | 144 | 78 |

Election under the leaf model takes `round(2 + 0.1 x N)` = 4 ticks for
N = 15-19 switches (5 today for N = 34), so first delivery on 87A/98A is
expected at about severance + 10 (orphan) + 4 (election) + ecc (3-5) + 1
(programming), i.e. tick 218-220 of a 1200-tick pass (severance 200).

Acceptance (server, scratch copy, monkeypatch only; run after the change
with `diag/run_wp12.sh`):
1. `sdn-static --seeds 80-99 --fragment-model infra-only --iteration bfs`
   and `... --seeds 50-59,70-79 --out-name sdn_static_infra_bfs_dev`: must
   print `predicted never-recovering: []` and `predicted partially blocked
   ...: []` (the probe is a pure replica; the engine track does not edit
   it). The engine's own fragment table, logged by `sdn-run`, must match
   the predictor's per instance: component sizes, leaders and flows per
   fragment as in the table above (the `run` report prints the
   controller's fragments and their infra-only decomposition; after the
   change the two must coincide: one component per fragment, 0 UEs
   anchored in >1 component within a fragment, 0 flows without a path
   inside a fragment).
2. `sdn-run --seeds 87,98 --ticks 600` (severance rescaled to 100): both
   Scenario-A passes must reach `operational`, with 2 eligible fragments
   each (19/15 and 15/18 switches), first delivery at about 100 + 10 + 4 +
   ecc + 1, and 122 / 121 programmed flows before any bridge forms (never
   more than the flows with an infrastructure path at that tick). The
   two rescue_ops passes must be bit-identical to the current
   `output/wp12_sdn_probe.pkl` (tree 2ba4be7: 310 / 292 flows, delivered
   740 509 / 579 968 post-event).  Check the re-queue explicitly, not only
   the totals: `programmed` must still rise from 222 at creation to 310 by
   tick 329 (seed 87) and to 292 by tick 323 (seed 98), the 88 / 70
   rescue-arrival flows of (c2).  A pass that ends at 222 programmed flows
   has the (c2) seeding missing; it is not a rounding difference, and the
   post-event delivery totals will move with it.  The same 222 reading is
   what a REBUILD-on-`force` implementation of (c2) rider 2 produces, with
   the seeding present but cleared by the tick-300 forced recompute before
   the re-queue reads it: this check does not distinguish the two causes,
   and both are failures of the same item.
3. `tests/test_wp12_probes.py`: the four `test_sdn_controller_*` tests
   encode today's behaviour (a 3-node path is never discovered, a stuck
   fragment starves the next, two components joined by a UE become one
   fragment whose cross flows are programmed). After the fix they must be
   inverted (path discovered in 2 rounds from its end node; no
   starvation; two fragments, the cross flows not programmed, the probe's
   decomposition equal to the controller's fragments); Track B will update
   them in the same commit window on request. The pure-replica tests
   (`test_sdn_predictor_*`, `test_sdn_bfs_rounds_*`,
   `test_sdn_infra_decomposition_*`) need no change.
