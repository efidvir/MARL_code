# Final corrected rerun — engineering specification (MARL_run_9)

Status: binding for every change made in this tree. Written 2026-09-17 after
reviewer round 3 (version 8). Author decisions: remove Block E; model the
cross-layer forwarding timing and bill its traffic; keep the untrained-actor
control, the adapt-then-freeze variant and a development-set re-run; stability
endpoints form their own confirmatory Holm family.

Baseline: commit `d94a936` = frozen evaluated tree `~/MARL_run_6`
(tag `confirmatory-A1` on GitHub). `~/MARL_run_5..8` are never modified.
All fixes are **hard-wired** (no env switch changes the evaluated system).
Diagnostic switches, if any, must default to the evaluated behaviour and be
recorded in checkpoint metadata and result pickles; the harness refuses a
checkpoint whose recorded schema/flags differ from the running tree.

Patch discipline: one work package (WP) per commit; after each WP run the
invariance matrix (§14) and the WP's own tests on <compute-server>
(`~/MARL_run_9`, `nice -n 19`, `PYTHONHASHSEED=0`, `MARL_POOL_WORKERS` cap).
`sixg_sim/simulation.py` mixes CRLF and LF line endings: edit with tools that
preserve bytes on untouched lines; never normalise the whole file.

---------------------------------------------------------------------------

## WP1 — Routing-graph consistency (reviewer item 1)

1. `Simulator._remove_relay_link(rid)`: pop `topology.links[rid]` (keep the
   object), drop `_relay_link_ready_tick`, `_relay_bridge_since`,
   `_relay_link_load`, `_relay_link_passes`, `_relay_route_since` entries,
   register a tombstone (§WP3) and call `topology.sync_pair(a, b)`.
   Both teardown sites (`_step_phy_mac` ~5552-5571 and
   `_maybe_repoint_to_fragment` ~6429-6441) call only this helper for every id
   returned by `clear_node_links` (including links where the node is the far
   end). No code parses link ids any more.
2. `Topology.pair_representative(u, v, up_predicate)`: among the pair's links
   satisfying the predicate, prefer TRANSPORT_RELAY with BACKHAUL interface,
   then lowest id (keeps today's "relay represents the hop while alive"
   semantics). Used by `sync_pair`, `update_link_statuses` and the link-state
   service's believed graph — one rule everywhere. `sync_pair` sets or removes
   the edge and refreshes `link_id`/`capacity`, then invalidates the
   infrastructure cache.
3. Relay link ids are unique per creation: `TR_<a>_<b>_<seq>` (global counter).
4. Relay predicates use the interface (BACKHAUL), so IAB access links are not
   counted as relays.
5. `consume_path`: a hop whose `link_id` does not resolve to a Link returns
   0.0 delivered for that flow (no `continue`), except tombstoned links handled
   by §WP3 (also 0.0 — physically down).
6. Invariant `check_graph_consistency()` on the TRUE graph at the end of every
   tick (before any believed-graph swap): every edge resolves to an up Link and
   equals `pair_representative`; every up infrastructure Link's pair has an
   edge. Violation raises (strict in all runs). Counters `stale_edges`,
   `stale_u2u_volume` are recorded per tick and must be 0.
7. UE-transit paths (possible data-plane transit through UE nodes): diagnose
   first (WP12). Change path selection only if the replay confirms it, for all
   arms, and record the decision in the pre-registration.

## WP2 — One per-tick step for training and evaluation

`Simulator.step_tick(tick, act_fn, *, forwarding)` fixes the order for both
`train_on_comparison.py` (both loops) and the harness physics pass:

1. per-tick resets (utilisation, DCC counters — self-resetting, §WP5);
2. scenario events; backhaul degradation (one shared helper: MICROWAVE,
   MICROWAVE_PTP and MULTIHAUL_MESH, 30 % of capacity, floor 50 Mb/s);
3. link-state service update of local isolation (§WP3/§WP6);
4. observations built from end-of-previous-pass snapshots
   (per-link load snapshot feeds the Block A backhaul fields);
5. actions (locked to the fixed pre-event configuration until the site's own
   local isolation flag is set — same rule in training and evaluation);
6. link-state service step;
7. forwarding on the service's installed view;
8. end-of-pass snapshots (link loads, relay pass counters §WP4).

A parity test runs the same seed/checkpoint through both drivers for 300
ticks and requires identical action, delivered and fragment series.

## WP3 — Link-state service for the cross-layer arms (reviewer item 4)

`sixg_sim/link_state_service.py` replaces `MARLLinkStateDetector`
(same API: `step`, `get_routable_flows`, `believed_graph`, `visible`, plus
`tick_messages`, `installed_since`, `isolation_state`). Used by every arm on
the MARL code path (marl_static, marl_freeze, xldet, random, untrained) in
training and evaluation.

- Pair-level adjacency: up iff some link of the pair is up.
- Detection: failure of an existing link is detected after BFD_DETECT_TICKS=3
  (BFD 1 packet/s per direction, multiplier 3). A new link's adjacency comes
  up after the BFD session, 3 ticks. A link created or torn down by one of its
  own endpoints is known to that endpoint at once (administrative change).
  After island onset a never-seen link id starts DOWN (no free adoption).
- Installation: for a confirmed pair transition at tick td each infrastructure
  endpoint originates a Router-LSA at t_o = max(td, last_orig + MIN_LS_INTERVAL)
  and the change is installed network-wide at
  t_inst = max_o(t_o + FLOOD_HOP_TICKS * ecc_o) + SPF_DELAY_TICKS, ecc_o the BFS
  eccentricity of the originator in the installed infrastructure graph.
  Constants are those of the OSPF baseline model (`_sim_ospf`); they are
  written into the pre-registration before any run.
- Tombstones: a popped Link keeps (endpoints, capacity) as physically down
  until its removal is installed; the believed graph never routes over a
  physically-down link (flows over it deliver 0).
- Forwarding and the flow gate use the installed view. `believed_graph()`
  returns None when installed == truth.
- Billing (`tick_messages`), from island onset only (pre-event billing stays
  zero for all arms):
  * BFD: 2 x 66 B per believed-up infrastructure adjacency pair per tick;
  * hellos: as `_sim_ospf` (interval, crc32 offset), 44 + 4 x infra-neighbour
    count bytes;
  * Router-LSA: one message per directed believed-up adjacency in the
    originator's component, 24 + 12 x degree bytes; no LSAck/refresh (as the
    baseline model);
  * postcards: DCC ledger (§WP5).
  The phantom floor and the change-detection proxy are removed; overhead for
  the MARL-path arms is the sum of billed bytes. Control-plane energy uses the
  OSPF per-message CPU weights. Overhead is reported **descriptively**, not as
  a comparative KPI.
- Baselines keep their own protocol models (no BFD). Remaining asymmetry:
  their gates block all flows at once — disclosed.
- Pre-registered evaluation-only sensitivity on the selected checkpoint:
  installation off (idealised timing).

## WP4 — Bridge stability (flapping) and churn instrumentation (item 7)

- Grace in usable passes: `_relay_link_passes[lid]` counts forwarding passes
  in which the relay link is installed in the arm's forwarding view (written by
  the service via `installed_since`). `_relay_teardown_blocked`: a protected
  link that is physically up but not yet installed is blocked; otherwise
  blocked while passes < RELAY_BRIDGE_GRACE_PASSES = 3; make-before-break
  busy threshold RELAY_BRIDGE_BUSY_UTIL = 0.02 and the beam-training lockout
  unchanged. Arms without the service (SDN) keep the physical stamp.
  Hold counts are recorded per reason (lockout, grace, not-installed, busy).
- Recording (cumulative simulator counters, differenced per tick in the
  harness): relay links created, torn down (by reason), bridge teardowns,
  idle-bridge teardowns (passes == 0), relay-mode flips (executed mode);
  per-link lifetime records; `fragments` and `fragments_raw` as today.
- Metrics (helpers in `confirmatory_analysis.py`, imported elsewhere):
  M1 fragment-count changes after the event; M2 policy-induced splits
  (f[t] > f[t-1] while raw components did not increase); M3 optimum losses;
  M4 fraction of post-event ticks at the attainable optimum (and M4ss over the
  steady-state window); M6 settling time (censored); M7 relay-link teardowns
  per 1000 post-event ticks; M8 bridge and idle-bridge teardowns per run;
  M9 relay-mode flips per 1000 ticks; M10 M1 over the steady-state window;
  M11 median bridge lifetime. Severance tick: A 200, B 0.

## WP5 — Control postcards (item 2)

- `DisasterControlChannel.send_postcard` keeps a counter keyed on the current
  tick (self-resetting); a post-pass invariant checks at most one send per
  site per tick and > 1 per site per episode after the event.
- Heartbeat: every agent site sends one postcard every POSTCARD_PERIOD = 10
  ticks (environment rule for all agent arms; no policy head).
- Frame model: one frame per live infrastructure link incident to the sender
  (`Link.is_up`), no fan-out cap; only agent-hosting receivers enqueue;
  declared encoded size 60 B; ledger of frames and bytes per tick.
- Census leak removed: the postcard carries the sender's own unserved-UE count
  instead of a network-wide isolated-UE count; the cooperative-relay field
  becomes "a fresh neighbour is relaying and reports unserved UEs".
- Receiver latch: per-sender table with 30-tick expiry; Block B fields are
  built from fresh entries; a freshness field (age of the freshest entry / 30)
  is added; `postcard_received` keeps meaning "a fresh postcard arrived this
  tick" (XL-DET contract).
- Block E and the fragment labels are removed from the actor and from the
  postcard. The engine's bridge-peer planner (component-aware, shared by all
  bridge-forming arms) is disclosed as an engine service; pre-registered
  evaluation-only sensitivity: re-pointing off.
- The learning-postcard exchanger is removed.

## WP6 — Observation schema and local event timer (items 3, 6; audit)

- Separate vectors: `ACTOR_OBS_FIELDS` (≈26) and `CRITIC_GLOBAL_FIELDS`
  (≈15) as ordered name/extractor lists; `observation_to_tensor` returns the
  actor vector, `critic_features` returns actor fields + global fields.
  NONLOCAL masking, `MARL_OBS_LOCALITY` and pad/truncate are removed
  (assertions instead). Dimensions are fixed in code and hashed into a
  schema id stored in every checkpoint; strict loader.
- Actor fields (final list fixed only after the constancy dump of §WP13):
  Block A (11): tx power, spectral efficiency, mean SINR, UE count,
  emergency fraction, coverage, own-link max utilisation, own-link min
  available capacity, relay mode, relay-link-active, min SINR;
  Block B (10): neighbour relay share, neighbour mean SINR, neighbour best
  SINR, fresh neighbour count, any neighbour relaying, neighbour freshness,
  relaying-neighbour fraction, neighbour mean tx power, own SINR drop,
  interferer fraction;
  Block C local (3): own relay capacity, live infrastructure neighbours (from
  the site's own believed-up links), anchored flow demand;
  Block D (2): local isolation flag, ticks since local isolation
  (min(1, Δt/200)).
  Dropped: prb utilisation (always 0), relay-capable (constant 1),
  coordinator vector, island-mode indicator (replaced), never-written and
  constant Block B fields, Block E.
- Local isolation (T1): site i is isolated iff its component in the service's
  believed-up infrastructure graph contains no core node; state updated once
  per tick inside the service step (idempotent); observations read the state
  of the previous step. Scenario B: flag set from tick 1.
- Critic value at collection is computed on the same (unmasked critic
  feature) mean the critic is fitted on; no silent exception fallbacks.

## WP7 — Action space (item 5)

- Heads: tx power (5), MCS offset emergency (5), MCS offset general (5),
  relay mode (3: OFF, LOCAL_REROUTE, CAPACITY_BOOST) + Dirichlet resource
  split over the 2-simplex. Handover, scheduler, postcard and admission heads
  are deleted from the network, the trainer (`DISCRETE_HEADS` defined once),
  the simulator and all drivers; the IOPS admission machinery is removed
  (admission follows from the split, unchanged helper).
- XL-DET returns (tx, mcs_e, mcs_g, relay, prb) with its relay rule mapped to
  the 3-mode head; its decisions must be unchanged apart from the aliasing.
- Uniform-random control: uniform over the new action space (each head
  uniform, Dirichlet(1,1,1)), redrawn every tick — pre-registered definition.
- Untrained control: regenerated from the new network definition with the
  training initialisation path.
- Entropy coefficient unchanged (sum over the four heads).

## WP8 — Reward (frozen as an explicit object)

- Structure and coefficients of the team objective and the agent-local reward
  as in the manuscript (eq:reward, eq:local), mixing κ = 0.6, advantage
  centring unchanged.
- Churn term scored on the executed relay mode (engine holds included).
- Terms shown constant by the constancy dump (e.g. r_qos) are removed and
  listed in the pre-registration.
- PPO scores the sampled action; engine holds are environment dynamics.
- 800-tick training episodes; truncation treated as terminal (disclosed).

## WP9 — Analysis and reporting

- Primary P1-P9 unchanged (MARL = marl_static), windows, comparators,
  exact Wilcoxon, Holm over 9, decision rule unchanged.
- Secondary S1-S7 as before (S7 = variant), S8 replaced by a hash-invariance
  check; S9 stability family (own Holm, 6 tests): M2, M4 (full post-event
  window) and M7, per scenario, MARL minus XL-DET, exact two-sided Wilcoxon;
  expected direction M2, M7 greater and M4 smaller for MARL; an endpoint with
  fewer than 6 non-zero differences is "not assessable". S10 descriptive:
  M1, M3, M4ss, M6, M8, M9, M10, M11 per arm. Overhead descriptive (S11).
- Hard asserts: all arms and seeds present, provenance (tree hash, schema id,
  PYTHONHASHSEED) identical across pickles, stale_edges == 0, postcards > 1
  per site per episode after the event, no NaN series.
- The validated script must reproduce `cf_report_A1.json` and
  `dev_report_A1.json` for P1-S7 from the frozen pickles.

## WP10 — Adapt-then-freeze variant

Kept as a disclosed secondary arm; its online-learning path is updated for
the new schema/heads/reward; its Phase-2 file dependency forces sequential
invocations (never concurrent).

## WP11 — Controls and scripts

Untrained-control generator and analysis updated; export, timeline and
figure scripts import the metric helpers.

## WP12 — Diagnostics before the freeze

(a) one-seed instrumented replay of the suspected UE-transit paths;
(b) SDN on confirmatory seeds 87/98 (never recovers): controller-isolation
fact or defect. Each ends with a written decision.

## WP13 — Integration gates

Constancy dump (one episode per scenario, both drivers); parity test;
invariance matrix (§14); hash-invariance smoke (PYTHONHASHSEED 0 vs 1: identical
action/delivered/fragment series); graph invariant on; postcard ledger > 1 per
site; training smoke (2 episodes); memory: peak RSS per pass measured and
`MARL_POOL_WORKERS` sized; unit tests updated and passing.

## 14 — Invariance matrix (after each WP)

Which arms must stay bit-identical to the previous commit (decisions,
delivered, fragments):
WP1: routing and SDN identical; MARL-path arms may change.
WP2: routing and SDN identical.
WP3: routing and SDN identical.
WP4: routing identical; SDN identical (physical-stamp fallback).
WP5: routing identical; SDN identical.
WP6/WP7/WP8: routing and SDN identical; XL-DET identical up to the relay-mode
aliasing (checked with the mapped modes).

## 15 — Study protocol (details in PREREGISTRATION_R2.md)

Training seeds 42-49 and 100-139; replicate torch seeds 1234, 2345, 3456,
4567; 400 episodes of 800 ticks. Validation 60-66 with the unchanged rule
(fewest components, then achievability; churn reported). Development set
50-59 and 70-79, all arms, descriptive and a technical gate only (abort on a
bug, never tune). Fresh confirmatory seeds 200-219 (audited unused), manifest
generated from the frozen final tree. Freeze 1 (code + pre-registration)
before training; freeze 2 (manifest digest) before the confirmatory run. Every
arm is run from the final tree with PYTHONHASHSEED=0; no old pickles are
reused. One unchanged re-run on technical failure.
