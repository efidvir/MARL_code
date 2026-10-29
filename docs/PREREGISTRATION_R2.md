# Pre-registration R2 — final confirmatory test of MARL-RIC on a corrected simulator

**Status: FINAL TEXT FOR FREEZE 1, written 2026-10-01 from the local
repository at code head `a33fdbc`** (branch `master`, every track merged).
Every value below was checked against the code of that head, file by file;
where this document had said something the code does not do, the code wins
and the disagreement is listed in §15. Revised the same day once <compute-server>
was reachable again; the state of the three preconditions:

1. the reduced-scale rehearsal of the whole chain on `a33fdbc`
   (`~/MARL_run_9/logs/prefreeze/rehearsal_a33fdbc.log`) **ended
   `CHAIN END profile=rehearsal rc=0`** at 2026-09-30T12:32:54Z (started
   07:43:40Z; 25 RUN lines, no stage shed or waived; every pickle it wrote
   carries `tree_head a33fdbc` and only unregistered confirmatory seeds);
2. the seed audit of the confirmatory block at the final head (§5) has been
   **MEASURED** — twice, on 2026-09-30 and 2026-10-01 — and returned **exit
   4 (INCOMPLETE)**: every hit is classified, but five launcher-test
   stand-ins in the scanned trees refuse the verdict until they are handled
   (`docs/SEED_AUDIT_200_219.md` §6; `docs/FREEZE_CHECKLIST.md` step c.0);
   the audit is re-run at the synced head before the real run starts;
3. the open decisions of §16 must be taken by the author before the launch.

**Since this text was first written, `master` has moved to `b43316a`**, a
commit made on 2026-10-01 by another session that changes
`tests/test_run_chain_launcher.py` only (§15 item 15). The statement below
that freeze 1 "differs from `a33fdbc` in `docs/` only" is therefore true
only if the author accepts a tests-only difference as covered by the
rehearsal (§16 item 8).

**Since then, XL-1 (`d7436c2`, `9ac7646`, 2026-10-04) changed code once
more** (XL-DET's interference brake, §2 item 11, §15 item 21), after the
first real-run attempt was stopped before S6 (§13); the rehearsal and the
freeze are repeated at the XL-1 head. Before it:

**D12-FIXES (`62a85ae`..`aada41e`) and D12F.8 (`c6e2aff`,
`afecb32`, `ad01ff0`) changed code again** (§16 items 1 and 9, §15 item
20): the rehearsal that covers freeze 1 is the one at the final head
(`docs/DECISIONS.md` D12F.9), and every mention of `02049e3` below as the
rehearsed head reads as that head. Before them:

**FIX-PREFREEZE changed code** (`1a486a7`..`02049e3`, §15/§16
item 9, which supersedes item 8): the rehearsed head is now **`02049e3`**.
Its reduced-scale rehearsal (`~/MARL_run_9/logs/prefreeze_fix/rehearsal_02049e3.log`)
**ended `CHAIN END profile=rehearsal rc=0`** at 2026-10-02T14:29:24Z, with
all 33 invocations of its plan OK and S6h, the S10 installation-off run and
S12's analysis S8, analysis S12 and secondary S7 computed (D-FIX.13); the
`a33fdbc` rehearsal of item 1 is the record of the head before
FIX-PREFREEZE only. Wherever the text below says freeze 1 differs from
`a33fdbc` in `docs/` only, read `02049e3`.

The commit that is synced for the real run (**freeze 1**) differs from
`a33fdbc` in `docs/` only (`docs/FREEZE_CHECKLIST.md` step a checks this);
its hash, tree digest and the records of §13 are appended to this file in the
LOCAL repository and in the manuscript repository (`records/R2/`) and are
never synced to the server while the run is in progress (a tracked-file
change on the server changes `tree_sha256` and stops the chain at its next
preflight, §12).

The launch procedure is `docs/FREEZE_CHECKLIST.md`.

## 1. Why this test exists

The confirmatory test of 16 September 2026 (seeds 80–99, `records/`) was run on
a simulator that a post-test audit found to differ materially from the
described system: torn-down relay links stayed in the routing graph without a
capacity limit; the control-postcard limiter was never reset, so each site sent
one postcard per episode and the fragment-coordination input (Block E) was
zero; the cross-layer arms forwarded over an idealised, uncharged link-state
service; four actor heads had no effect; one observation was a scripted
time-since-severance; the relay bridges of the learned policy flapped on some
instances because of an engine timing mismatch. Correcting only the stale
edges changed one of the seven confirmed hypotheses (P7). This study corrects
the simulator, retrains the policy and repeats the confirmatory test once, on
a pre-registered block of instances (§5, with its disclosure). It supersedes
the earlier test as the paper's confirmatory evidence; the earlier test is
reported as history.

## 2. System under test (the code at `a33fdbc`)

The engineering specification `docs/RERUN_SPEC.md` of the frozen tree is part
of this pre-registration; where the code departs from it the departure is
recorded in `docs/DECISIONS.md` and summarised here.

1. **Routing-graph consistency (WP1).** A torn-down relay link leaves the
   routing graph at once (`Simulator._remove_relay_link`, one
   `Topology.pair_representative` rule, `sync_pair` on every creation and
   removal path); relay link ids are unique per run; a strict per-tick
   invariant (`check_graph_consistency`) stops any run in which graph and
   links disagree; an unresolved hop delivers 0.0 and is counted
   (`stale_edges`, `stale_u2u_volume`, which the analysis requires to be 0).
2. **Shared per-tick step (WP2) and the per-site action lock (D13.2).**
   Training and evaluation step the simulator only through
   `sixg_sim/tick_step.step_tick`, in one fixed phase order (resets → events
   and backhaul degradation → global island trigger → auto link adaptation →
   observations → actions → link-state step and flow gate → graph invariant →
   forwarding on the installed view). **The action lock is per site**: an
   agent site keeps the fixed pre-event configuration (23 dBm, relay OFF,
   round-robin, 30/70 emergency/general access split, MCS from auto link
   adaptation; `lock_phy_mac_pre_event`) until ITS OWN local-isolation flag —
   the value its own Block D observes, i.e. the link-state service's state
   after the previous step — is set, and then acts for the rest of the
   episode (a latch, `Simulator.update_site_action_locks` /
   `site_action_unlocked`). A locked site takes no action, sends no postcard
   and leaves NO PPO transition; the critic input at collection is the mean
   over the acting sites. The engine's global island trigger
   (`action_lock_open`) gates only the arms without agents. Scenario B sets
   every flag from tick 1 (no site acts at tick 0). Under the
   `MARL_DETECT_TICKS=0` arm (no service) the same rule is evaluated on the
   true infrastructure graph.
3. **Link-state service** (`sixg_sim/link_state_service.py`;
   `docs/LINK_STATE_SERVICE.md`) for every arm on the MARL code path
   (`marl_static`, `marl_freeze`, `xldet`, `random`, the untrained controls),
   in training and in evaluation, with the constants of the OSPF baseline
   model (`tests/test_wp3_link_state_service.py` asserts equality with
   `run_timeline_comparison.py`): `BFD_DETECT_TICKS` = 3 (1 packet/s per
   direction, multiplier 3), `MIN_LS_INTERVAL` = 5, `FLOOD_HOP_TICKS` = 1,
   `SPF_DELAY_TICKS` = 5, `HELLO_INTERVAL` = 10, 1 tick = 1 s. No free
   adoption of a never-seen link after island onset; tombstones for removed
   links (a popped link carries exactly 0.0 until its removal installs); an
   administrative teardown is known at once at the acting endpoint only; an
   up-transition always pays the 3-tick BFD session; access (Uu) adjacencies
   follow physical truth with no delay (D3.3). **ONE Router-LSA per router,
   not per pair transition** (D13.1; RFC 2328 §12.4 and the OSPF baseline's
   "at most 1 LSA origination per node per 5 ticks"): a router originates
   one Router-LSA at max(knowledge tick of its earliest pending transition,
   last origination + `MIN_LS_INTERVAL`) carrying every transition it knows
   of by then; a transition installs network-wide at
   max over its originators (t_o + `FLOOD_HOP_TICKS` × ecc_o) +
   `SPF_DELAY_TICKS`, ecc on the installed router graph. Measured effect on
   the Block D isolation flag (D13.1, `tools/isolation_latency.py`,
   uniform-random policy, all 22 agent sites flagged): 13–14 ticks after the
   severance on seeds 82, 50 and 71, against 31–51 ticks under per-transition
   origination. Forwarding and the flow gate use the INSTALLED view; the
   flow gate is per flow.
   **Billed control traffic is reported descriptively only (S11)** and starts
   at island onset: BFD 2 × 66 B per infrastructure adjacency pair **running
   a session** — confirmed up, or with a session in progress (bring-up, or a
   failure inside its detect multiplier). This departs from RERUN_SPEC WP3's
   "per believed-up pair" (D3.24 / D3.34) and moves `overhead` and, through the
   control-plane CPU term, `energy` (≈4e-6 relative) of the MARL-path arms.
   Hellos at the steady-state `HELLO_INTERVAL` throughout (no fast-hello
   transient, unlike `_sim_ospf`: D3.25, in the cross-layer arms' favour);
   one Router-LSA flood per origination; postcards from the WP5 ledger
   (item 5). The baselines keep their own protocol models and get no BFD;
   their gates block every flow until their controller converges while this
   service gates per flow (D3.6 — favours the cross-layer arms in
   transients, outside the primary windows); their
   `_detect_topology_change` ignores link removals, so their modelled
   signalling under-counts teardown-driven re-flooding (D4.11; disclosed with
   S11).
4. **Bridge stability (WP4).** The teardown grace of a relay bridge counts
   forwarding passes in which the bridge is INSTALLED in the arm's forwarding
   view (`RELAY_BRIDGE_GRACE_PASSES` = 3, read from the service's
   `installed_since` mirror); a protected bridge that is physically up but
   not yet installed cannot be torn down (`not_installed`);
   make-before-break busy threshold `RELAY_BRIDGE_BUSY_UTIL` = 0.02 of
   capacity; the beam-training lockout is unchanged; one hold per blocked
   call under the first reason that fires (lockout, not_installed, grace,
   busy). An arm without a service (SDN, the `MARL_DETECT_TICKS=0` arm)
   keeps the tick stamp (`RELAY_BRIDGE_GRACE_TICKS` = 3). A relay that fails
   undetected accrues up to 3 passes while delivering 0.0 (D3.30/D4.6), so
   `idle_bridge_teardowns` means "never usable", not "never carried bits".
   **Disclosed, because XL-DET is the S9 comparator (D4.19):** on the WP4
   verification run (head `18606cb`, 600-tick invariance configuration,
   Scenario A, seed 82, severance at tick 100) XL-DET settled at TWO
   fragments for the rest of the run (mean post-event fragments 1.044 →
   2.018, delivered −2.95 %): its teardown requests were refused 21 times as
   `not_installed` (t = 103–123, the Router-LSAs rate-limited behind the
   severance burst), twice as `grace`, then as `busy` on every tick from 109
   to the end (491 holds); the re-pointing at t = 106 that the earlier head
   used was no longer available, and make-before-break never releases a
   loaded bridge. That is the pre-registered rule behaving as specified,
   identical for every arm, and the retrained policy meets the same rule.
   Seed 82 is a smoke seed, not a study instance, and the measurement
   predates D13.1 (per-router origination shortens the `not_installed`
   window). **At the final head it no longer settles at two fragments**
   (D-FIX.8; the post-WP13 reference `inv_ref_wp13_c7d2322.pkl` and the
   FIX-PREFREEZE invariance pass at `e785888`, identical): 4 fragments
   until t = 117, then 3, 1 from t = 118, one split to 2 at t = 134 (M2 =
   1), 1 again from t = 137 to the end; mean post-event fragments 2.018 →
   1.112; holds `not_installed` 14, `grace` 2, `lockout` 3, `busy` 0.
   Instances of this kind enter S9 (M2, M4, M7) through XL-DET and are
   reported, not removed.
5. **Control postcards (WP5; `sixg_sim/postcards.py`).** The DCC send limiter
   resets itself per tick (the one-postcard-per-episode defect is gone);
   the postcard is an environment HEARTBEAT, not an action: one per agent
   site every `POSTCARD_PERIOD` = 10 ticks, phased from the site's first
   acting tick, carrying the sender's EXECUTED state; one 60-byte frame
   (`POSTCARD_ENCODED_BYTES`, declared field layout) per live infrastructure
   link incident to the sender, delivered next tick to agent-hosting
   neighbours; a site with no live infrastructure link sends a heartbeat with
   zero frames. Each receiver keeps, per sender, the last postcard heard,
   FRESH on ages 0–29 (`POSTCARD_LATCH_TICKS` = 30, three missed heartbeats);
   Block B is built from the fresh entries. The postcard carries the
   sender's OWN unserved-UE count (`unserved_ue_count`), **no network-wide
   census**. **Block E, the fragment labels and the learning-postcard
   exchanger are removed**; `MARL_POSTCARD_FRAGMENTS` (on),
   `MARL_POSTCARD_ALWAYS` (off) and `MARL_POSTCARD_DRAIN=legacy` are refused
   at import. `PostcardSendAudit` raises on a second send by one site in one
   tick or a send by a locked site, and at the end of a pass unless every
   site present on more than 10 acting ticks sent more than one postcard.
   The engine's component-aware bridge-peer planner (candidate filter,
   most-constrained-first ranking, re-pointing) knows the true
   infrastructure components and serves every bridge-forming arm alike
   (MARL-RIC, random, XL-DET, SDN): an engine service, disclosed in
   `docs/BRIDGE_PEER_PLANNER.md`; re-pointing off is the S11 sensitivity.
6. **Observation schema (WP6, WP13; `sixg_sim/agent.py`).** Two ordered
   vectors, no masking. **Actor: 26 site-local fields** —
   Block A (11) `a_tx_power`, `a_spectral_efficiency`, `a_sinr_mean`,
   `a_ue_count`, `a_emergency_fraction`, `a_coverage`,
   `a_own_link_max_util`, `a_own_link_min_avail`, `a_relay_mode`,
   `a_relay_link_active`, `a_sinr_min`; Block B (10, from the fresh postcard
   latch) `b_neighbour_relay_share`, `b_neighbour_sinr_mean`,
   `b_neighbour_sinr_best`, `b_fresh_neighbour_count`,
   `b_any_neighbour_relaying`, `b_neighbour_freshness` (1 − age/30 of the
   freshest entry), `b_relaying_neighbour_fraction`,
   `b_neighbour_tx_power_mean`, `b_own_sinr_drop`, `b_interferer_fraction`;
   Block C (3) `c_own_relay_capacity`, `c_live_infra_neighbours` (the site's
   own BFD-confirmed adjacencies, D-MRG.4a), `c_anchored_flow_demand`;
   Block D (2) `d_local_isolation` (the service's per-site isolation state
   after the previous step: the site's component in the installed
   infrastructure graph holds no core node) and
   `d_ticks_since_local_isolation` = min(1, Δt/200). **Critic: 37 = the 26
   actor fields + 11 globals** — `g_active_relay_paths`,
   `g_ue_to_ue_routed`, `g_intra_island_reachability`, `g_core_distance`,
   `g_bridge_node_flag`, `g_island_size`, `g_pending_iops`,
   `g_island_member_count`, `g_island_ue_load_balance`,
   `g_xn_mesh_density`, `g_local_epc_health`.
   **Constant fields, decided before any result (D13.3), on production-length
   evidence**: `tools/constancy_dump.py --ticks 3600`, both scenarios, seeds
   82, 50 and 71 (six passes, 79 200 samples per field per pass, the service
   attached, D13.1 and D13.2 applied), combined by
   `tools/constancy_combine.py` into `docs/wp13/constancy_c148cc9.json`; a
   field leaves only if it is constant, at one value, in EVERY pass. Removed:
   the four critic globals `g_reachable_ue_fraction` (1.0),
   `g_isolated_ue_count` (0.0), `g_nenb_count` (0.0) and
   `g_multi_island_bridge` (0.0) (`REMOVED_CONSTANT_FIELDS`). Kept although
   the 600-tick passes had shown them constant: `a_emergency_fraction` and
   `g_pending_iops`, which MOVE in Scenario B once the rescue force arrives at
   tick 1400 (emergency UEs, pending IOPS requests). **Schema id
   `obs1-542eb7fc0d277d1d`** (`sixg_sim.agent.SCHEMA_ID`, sha256 over both
   widths, both ordered field lists, the discrete heads with their sizes, the
   relay modes and the PRB parameterisation; recomputed from the source on
   2026-10-01). Every checkpoint and pickle records it; the strict gate
   refuses any checkpoint under another id, so nothing trained before WP13 can
   be evaluated by this tree.
7. **Actions (WP7).** Four discrete heads — `tx_power` (5), `mcs_emrg` (5),
   `mcs_gen` (5), `relay` (3: OFF, LOCAL_REROUTE, CAPACITY_BOOST; the aliases
   NORMAL and D2D_PEER_RELAY fold onto OFF) — plus the Dirichlet split over
   the 2-simplex (emergency / relay / general; `MARL_PRB_POLICY=dirichlet`:
   sampled in training, its mean at evaluation). No handover, scheduler,
   postcard or admission head. Actor 26 → 128 → 128 (LayerNorm) → heads,
   23 189 parameters (D7.6; WP13 did not change the actor). XL-DET's relay
   rule is mapped onto the 3-mode head with its decisions otherwise
   unchanged (bit-identical to its pre-WP7 rule on every recorded series,
   D7.16) -- except the transmit-power climb below the half-rate cliff that
   XL-1 added (item 11). The uniform-random control draws every head uniformly and the
   split from Dirichlet(1,1,1), per site, every tick (D7.4).
8. **Reward (WP8; `sixg_sim/reward_spec.py`, rendered verbatim in
   `docs/REWARD_FROZEN.md`).** One frozen coefficient object, **reward id
   `rew1-7f89883ae19733f3`** (recomputed from the source on 2026-10-01),
   recorded in every provenance record. Mixed per-agent reward
   r_i = (1 − κ)·local_i + κ·team with **κ = 0.6**, enforced
   (`require_frozen_mix` refuses any other `--global-reward-mix`); advantage
   centring unchanged. **The churn term is scored on the EXECUTED relay
   mode** (an engine hold is environment dynamics, not a policy choice);
   **PPO scores the SAMPLED action**. 800-tick training episodes
   (`EPISODE_TICKS`), truncation treated as terminal (disclosed: the
   training horizon is shorter than the 3600-tick scoring horizon).
   Removed terms: `r_qos` (agent-local), `relay_contrib` (team and local,
   coefficient 0.0), `bridge_held` (local, 0.0), `recovery_ramp` /
   `recovery_complete` (structurally zero: no `restore_core` event exists in
   either scenario), the routed-flow cliff, the raw-utilisation relay
   measurement and the unconditional reachability term. **DISCLOSED: the
   full-delivery bonus (+25 at a delivered UE-to-UE fraction ≥ 0.95) is kept
   but was never earned in any measured pass** — `tools/reward_term_dump.py`
   (uniform-random reference policy, seed 82, 600 ticks) records it constant
   0.0 in both scenarios while the delivered fraction reached only 0.4714
   (A) and 0.5279 (B); it is kept because what is unreached is the threshold,
   not the input (D8.12, D8.19, D8.27). **DISCLOSED as well (D13.3):** the
   removal of `r_qos` rests on 600-tick evidence (no LIFE_SAFETY load
   offered; `a_emergency_fraction` constant 0.0). At production length
   `a_emergency_fraction` moves in Scenario B after the rescue arrival, and
   whether LIFE_SAFETY load is then offered — i.e. whether `r_qos` would have
   paid — is not measured in the committed record (the production-length
   reward-term dump promised as D13.6 is absent, §15 item 11). The removal
   stands as frozen.
9. **UE-transit paths — corrected (D12.1, implemented by D12-FIXES;
   §15 item 1).** Every data-plane path takes infrastructure interiors only:
   a UE is a path's ENDPOINT, never an interior hop, in every arm. The
   UE-to-UE shortest path and its alternates in
   `Simulator._forward_traffic`, the general-traffic sink path
   (`_best_path_to_any`), the harness's per-tick hop and latency metrics and
   the SDN FlowMod per-hop count all search `Topology.path_view(endpoints)`:
   the graph in use (the true graph, or the link-state service's installed
   view while it is swapped in) restricted to its infrastructure nodes plus
   the flow's own endpoints. The infrastructure node set is built once per
   graph version (once per forwarding call, once per tick in the harness);
   the view's filter is a frozenset's membership test, so neighbours are
   visited in the graph's own adjacency order and the choice among
   equal-length paths does not depend on the hash seed. The reachability
   gate (`_can_route_ue_to_ue`) is unchanged; `consume_path` walks the path
   it is given. Consequence: a flow whose endpoints are joined, in the arm's
   forwarding view, only through a dual-anchored UE is not delivered -- in
   particular while a MARL-RIC bridge is physically up but not yet
   installed (the phase-locked artefact D12.1 measured: 11.1 % of
   MARL-RIC's post-event UE-to-UE volume against 1.8 % of OSPF's on the
   first study's engine, seed 82, Scenario A), and the pre-event world of
   every arm loses the share it carried through UEs (4.2 % / 0.8 % on seeds
   82 / 91 there). The effect of the fix on every arm of the final engine,
   seeds 82 and 91, both scenarios, is §15 item 1 and `docs/DECISIONS.md`
   D12F.5. **The routing baselines' control planes follow the same leaf
   model (D12F.8, the author's decision of 2026-10-03).** OSPF, OLSR,
   B.A.T.M.A.N. and AODV share one fragment decomposition
   (`infra_fragments_with_ue_leaves`: infrastructure components over live
   infrastructure links, every surviving UE a leaf of each fragment holding
   one of its anchors), which times their convergence and gates their
   routable flows; AODV establishes routes and bills RREPs on the path view,
   and a UE does not rebroadcast a RREQ. Before D12F.8 OLSR, B.A.T.M.A.N. and
   AODV ran their fragment BFS through UEs; measured on seeds 82/91 (D12F.8):
   AODV delivers 1-2 points more in Scenario A (its discovery queue no
   longer holds unreachable flows), OLSR and B.A.T.M.A.N. are unchanged to
   within a short convergence window, Scenario B is identical, and every
   other arm is bit-identical. Not changed (descriptive overhead only):
   B.A.T.M.A.N.'s OGM overhead model lets a UE rebroadcast, and OLSR's MPR
   selection may pick a UE.
10. **SDN baseline model — corrected (D12.2, implemented by D12-FIXES;
    §15 item 2).** `SDNController` (`run_timeline_comparison.py`):
    (a) LLDP discovery is a CUMULATIVE BFS from the leader over the
    fragment's switches -- each round the newly discovered layer probes its
    infrastructure peers, and a round that discovers nothing completes the
    fragment, after ecc(leader) + 1 rounds on every fragment, bipartite or
    not -- with a round timer per fragment, so fragments discover in
    parallel once the last election has ended (the election phase gate is
    unchanged). The LLDP billing reads the probing layer as before.
    (b) Fragments are the connected components of the surviving
    infrastructure over up infrastructure links (relay links included); a
    UE is a LEAF of every fragment holding one of its live anchors (the
    infrastructure endpoints of its up links, the anchors of
    `_can_route_ue_to_ue`) and is never walked, so a dual-anchored UE no
    longer stitches two components into one domain. `fragment_members`
    holds the switches and the leaves, `fragment_switches` the walk set.
    The election rule is unchanged (at least `MIN_FRAGMENT_FOR_ELECTION` = 3
    switches; the lexicographically first switch of maximum up-link
    degree). (c) A flow is programmed only when both endpoints have a live
    anchor among the discovered switches of ONE fragment marked discovered:
    at discovery completion and in the operational-phase re-queue, which
    recomputes the fragments first. Choice stated (handoff 7.2 (c)): merged
    fragments are NOT re-discovered from scratch; a recompute that finds two
    discovered fragments joined (e.g. by SDN's relay activation) lets their
    cross flows in at once, and a fragment that gains an undiscovered switch
    continues its walk from everything already discovered. (c2) The
    Scenario-B pre-converged controller (`mark_pre_converged`) marks every
    fragment discovered WITH its switches as the discovered set -- every
    fragment, elected or not (rider 1); a forced recompute REMAPS all
    per-fragment state onto the new decomposition (unchanged fragments keep
    theirs; a merged one takes the leader of its largest part and the union
    of the parts' discovered switches) and re-applies that seeding for a
    controller that never walked (rider 2: never a bare rebuild). (d) The
    model's own rule is kept: a fragment below `MIN_FRAGMENT_FOR_ELECTION`
    switches elects nobody, so in Scenario A a flow whose only common
    component is such a switch is not programmed (until a link joins the
    switch to a led fragment and the walk reaches it); the pre-converged
    controller of Scenario B is converged by construction, so it is not
    affected. Not revoked: a programmed flow whose endpoints later lose
    their common fragment stays programmed (the data-plane gate stops it).
    Consequences (D12.2): on a Scenario-A instance the controller runs one
    controller per infrastructure component of at least three switches
    (2-3 on the first study's set), discovery completes after ecc + 1
    rounds instead of 8-16 or never (87A, 98A and the development set's 52A
    recover; 81A becomes operational), the programmed set holds the flows
    with an infrastructure path inside a discovered fragment instead of all
    222, and the LLDP billing no longer charges probes from UEs. The
    measured effect on seeds 82 and 91 is §15 item 2 and
    `docs/DECISIONS.md` D12F.5; the acceptance runs of handoff 7.2 are
    D12F.6.

Schema id of the frozen code: `obs1-542eb7fc0d277d1d`; reward id
`rew1-7f89883ae19733f3`. Tree digest (`runinfo.tree_digest()` over the
tracked files, `*_adapted_recovery*` excluded, D0.2) and HEAD: recorded by
the chain's first preflight and sealed in `confirmatory/FREEZE_FINAL_real.txt`
at S5; copied into §13 at freeze.

11. **XL-DET's interference brake — corrected (XL-1, `docs/DECISIONS.md`
    D-XL.1; the author's decision of 2026-10-04).** `tx_head`
    (`sixg_sim/heuristic_controller.py`) brakes a climb while a fresh
    neighbour postcard reports a loud neighbourhood or the site's own SINR is
    falling. Since WP5 (item 5) the postcards arrive every 10 ticks, so the
    brake never expired: after a fleet-wide step-down every site braked every
    other and XL-DET held its fleet at low power for the rest of a Scenario-B
    pass. On the development seeds (never the confirmatory block) its
    Scenario-B delivered fraction fell from 0.386 (the dry run's pre-WP5
    engine) to 0.270, below every routing baseline (0.403), as low as 0.02 on
    seed 51; the first real-run attempt found it in its S1 output (§13). The
    brake now may stop a climb but may not hold a site below its own
    half-rate cliff: below `SINR_TARGET_DB − HALF_RATE_MARGIN_DB` = 12 dB the
    site climbs, braked or not. Of three variants measured on development
    seeds 50–53 (600 ticks), this one is both the rule the author approved
    and the strongest comparator (Scenario B 0.283 → 0.524, Scenario A
    0.450 → 0.515; the give-back rule, which also steps down after a fall
    that bought nothing, is unchanged, because changing it made XL-DET
    weaker); a stronger XL-DET is the conservative direction for P1, P4 and
    S9. Every other arm is bit-identical (invariance row `XL1`).

## 3. Training and replicate selection

- **Training (chain stage S2, `confirmatory/stage_train.sh`).** Four
  replicates, torch seeds 1234, 2345, 3456, 4567, each
  `train_on_comparison.py --episodes 400 --episode-ticks 800 --seed-range
  42-49,100-139 --torch-seed <S> --base-seed 7 --batch-size 16
  --eval-episodes 0 --baseline-episodes 0` (training topologies 42–49 and
  100–139, 48 seeds; event seed of episode *k* = 700000 + *k*, so no
  (topology, damage) pair of the confirmatory block is generated in
  training). `MARL_POOL_WORKERS` = 16 per replicate, at most two replicates
  alive, the next launched as soon as one exits, under an EXCLUSIVE lock (no
  evaluation invocation of the chain runs beside training). A wave is
  DELAYED, never abandoned: it starts only with ≥ 26 GB (GiB, §12)
  MemAvailable for the wave and a 1-minute load below 20 (`train_load_ceiling`), and a dip below
  the 22 GB floor while the chain's own replicates finish is waited out on
  the same retry budget (§12).
- **Determinism (WP-DET).** The training environment is seeded with the
  harness's own rule (topology seed; D-DET.1), the batch is folded into the
  rollout pool in episode order (D-DET.3), the drivers pin their threads
  (D-DET.4) and the last OS-entropy source is gone (D-DET.5). A replicate's
  policy is therefore a function of its torch seed alone: **identical
  weights at every worker count and under concurrent replicates.** The gate
  that proves it is `tools/det_gate.sh` (2-batch smoke; weights digest
  `tools/policy_digest.py --weights-only`, not the checkpoint file, whose
  provenance differs by construction; VERDICT and exit status): on
  2026-09-23/24 it gave one policy for torch seed 1234 over ten runs at 1, 4,
  8 and 16 workers and four concurrent replicates at 2 workers, and a
  different one for seed 2345 (D-DET.10, D-DET.12, D-DET.15). Those runs
  predate WP6–WP8 and WP13; the gate has not been executed on the final
  tree in the committed record and is re-run at freeze
  (`docs/FREEZE_CHECKLIST.md` step b). `tests/test_wpdet_determinism.py`
  (the real training worker, called twice, collects identical transitions)
  runs in every suite run. Open, not closed (D-DET.13): the critic's final
  weights are not digested; `avg_reward` in the checkpoint metadata is the
  last episode's mean, not the batch's.
- **Selection (S3, S3b; `confirmatory/stage_select.sh`).** Each replicate's
  `marl_static` runs on validation seeds 60–66, both scenarios (14 passes per
  replicate, 24 workers). The **unchanged rule** of the amended first study:
  the replicate with the lowest mean steady-state component count (mean over
  the seven Scenario A instances of the mean of `fragments` over the last
  500 of 3600 ticks), ties broken by the higher mean steady-state
  achievability. The rule is applied once, to the complete set; a missing
  replicate or a replicate with no scored instance fails the stage; if all
  replicates tie on the primary criterion the script says the rule was NOT
  EXERCISED and the tie-break alone chose. Stability metrics of each
  replicate are reported next to the selection and do not enter it.
- `confirmatory/SELECTED_real.txt` (checkpoint path, torch seed, sha256 and
  the per-replicate scores) is copied into §13 after S3b.

## 4. Development set (technical gate)

Seeds 50–59 and 70–79, both scenarios: the seven policy-independent arms in
S1 (before training, 280 passes, plus 60 stress passes on 50–59 with
`MARL_STRESS_INTERFERENCE_DBM=-75` from tick 2000 for xldet, sdn, ospf) and
`marl_static` with the selected checkpoint in S4 (40 passes, plus 20 stress
passes and the end-state dump of seed 58). Results are descriptive. The run
may stop the study only on a technical failure:

- **S5 (`confirmatory/stage_gate.sh`):** every S1/S4 pickle holds exactly
  its expected (arm, scenario, seed) keys, `pythonhashseed` = '0' and a tree
  digest (`tools/stage_report.py --verify`); identical tree head, tree
  digest, schema id and hash seed across them (`--cross-provenance`); the
  selected checkpoint and the confirmatory manifest exist. It then seals
  `confirmatory/FREEZE_FINAL_real.txt` (tree digest, manifest sha256,
  selected-checkpoint sha256, every dev/stress pickle's sha256); from then on
  every preflight refuses a tree whose digest differs.
- **S5b (`confirmatory/stage_analysis.sh`, step `dev`):** the
  pre-registered `confirmatory_analysis.py` runs on the development pickles
  (`s1_dev_b*`, `s4_dev_static`) with `benchmark_manifest.json` and
  `benchmark_manifest_70_79.json`, `--marl-arm marl_static`, **without
  `--no-assert`**: its hard asserts (§10) are the remaining items of the
  technical checklist — a graph-invariant violation (the run itself raises),
  `stale_edges` or `stale_u2u_volume` above 0, one postcard or fewer per site
  after the event, a NaN or short series, a provenance mismatch. A change of
  the `marl_static` parameters within a pass raises inside the harness
  (`STATIC CHECK`), so such a pass fails before any gate.
- **Not automated by the chain:** "a baseline arm departing from its smoke
  expectations (the invariance matrix of RERUN_SPEC §14)". It is checked
  once, at the freeze head, before the launch (`docs/FREEZE_CHECKLIST.md`
  step b).

Nothing may be tuned on the development set. A stop is followed by a
documented fix, a new freeze and a new training run.

**Disclosure — development and validation seeds were exercised before the
freeze**, by pipeline tests only: the reduced rehearsals (dev 50, 51, 70, 71
at 600 ticks; 50, 51 at 3600; validation 60, 61), the full-scale dry run
(first attempt 2026-09-25 on `24e86d0`, stopped in training; completed run
2026-09-27/28 on `b728ce8`; dev 50–59 and 70–79, stress 50–59, validation
60–66, on the pre-WP6 agent), the production-length constancy dump and the isolation-latency
measurement (seeds 50 and 71, uniform-random policy; D13.1, D13.3) and the
WP12 SDN static predictor (topology only). The committed record names no
decision taken on a development-set outcome of a trained policy.

## 5. Confirmatory instance set

- **Topology seeds 200–219**, with matching scenario seeds
  (`build_comparison_scenario` uses `scenario_seed = topology_seed`), both
  scenarios: `PROFILES['real']['cf_seeds']` in `confirmatory/chain_plan.py`;
  `tests/test_run_chain.py::test_only_the_real_profile_names_the_preregistered_confirmatory_seeds`
  pins that no other profile names any of them (since `31b034c`: the
  rehearsal uses 220–223 with a full-length subset 220–221, the test profile
  220–221, the dry-run profile 220–239).
- **History of the evidence.** The audit of 2026-09-23 (11:58–12:05 UTC,
  `tools/seed_audit.py` at `d2dc46a`, driven by `tools/run_seed_audit.sh`,
  14 `$HOME/MARL*` trees, 7 206 files, verdict CLEAN, exit 0, coverage ok;
  `docs/SEED_AUDIT_200_219.md` §1–§4) found no use of the block. That
  statement is about that date. It is no longer true of the block as a
  whole. Two audits at the final head have since been MEASURED with the same
  tool and the committed driver (`MARL_TREE=$HOME/MARL_run_9 bash
  tools/run_seed_audit.sh 200 219`), both **exit 4 (INCOMPLETE)**:
  - **2026-09-30** 10:37–10:49 UTC, head `a33fdbc`, during the rehearsal,
    223-commit history dump; report sha256 `aa310f483504d225…`. Earlier
    freeze text did not cite it and predicted "exit 3".
  - **2026-10-01** 12:28–12:39 UTC, report `~/MARL_run_9/output/seed_audit_200_219.json`
    (copied to `~/MARL_run_9/audit_history/run_20261001T122828Z/`, sha256
    `3e39a99d6371868cdad9c9c62e5f56cb3c34b55a5d83d22249511b246a81c67c`). A
    sync to `b43316a` and a test suite started by another session wrote into
    the tree while it ran, so it is a measurement of the hits but not freeze
    evidence (`docs/SEED_AUDIT_200_219.md` §6.3).

  Both refused the verdict for one reason only: five synthetic stand-ins
  written into the scanned trees by the launcher tests of the `test`
  profile — three 244-byte pickles under `.json` names
  (`~/MARL_run_9/confirmatory/test_manifest_200_201.json`,
  `~/MARL_run_9/confirmatory/test_manifest_220_221.json`,
  `~/MARL_run_9d/confirmatory/test_manifest_200_201.json`), which fail the
  JSON parse (`errors`), and two 238-byte pickles under
  `output/test/untrained_s1234/policy_init.pt` in both trees, which fail to
  load as checkpoints. No simulation produced them (constant or empty
  series, provenance `faketest`). They are handled before the gate as
  `docs/FREEZE_CHECKLIST.md` step c.0 states — archived outside every
  scanned root with their path, size and sha256 recorded, after the
  author's consent, since they are untracked files inside the trees — and
  the full suite (step b.1) re-creates the `220_221` one unless the tests'
  cleanup (`b43316a`) removes it.
- **Seed disclosure — MEASURED** (`docs/SEED_AUDIT_200_219.md` §6.6–§6.7,
  verified from the ledgers, the job scripts and the pickles' provenance; no
  metric was opened):
  - **two reduced rehearsals of 2026-09-24** (not "24/25": every ledger
    event is on the 24th, UTC and local) in `~/MARL_run_9d` ran confirmatory
    stages on seeds **200–203**: attempt 1 (08:15–11:24 UTC, tree
    `0592f26`, archived in `archive_rehearsal/run1_20260924T123836Z/`) and
    attempt 3 (14:40–18:17 UTC, tree `79e15b2`, `output/rehearsal/`); S6
    ran all eight confirmatory arms, both scenarios (64 passes each time),
    S8 the untrained control (2 × 8, then 3 × 8 passes), S9
    adapt-then-freeze, S10 and S11 (8 each) — **216 passes in all, every one
    at 600 ticks** (not the nominal 3 600). S0 built a manifest for
    200–203 in attempts 1, 2 and 3 (attempt 2, tree `dd8a18e`, stopped in
    training and ran no window-seed pass). S9 left eight copies of the toy
    policy adapted online on those instances
    (`ck_s2345/policy_final_adapted_recovery_freeze_seed200…203.pt`, current
    and archived), used by nothing afterwards;
  - result data of the window: **13 pickles, 3 manifests, 8
    adapted-checkpoint file names and 1 094 log/ledger lines, all in
    `~/MARL_run_9d`, all for 200–203**. Seeds 204–219 carry no result data
    anywhere; no checkpoint among 2 793 carries a window seed in its
    metadata; no RESULT pickle or JSON in `~/MARL_run_9` carries one — the
    only file there naming the block was the synthetic launcher-test
    stand-in `confirmatory/test_manifest_200_201.json`, a 244-byte pickle
    under a `.json` name whose `seeds` label held 200 and 201 (no passes,
    provenance `faketest`), recorded by the scan as an error, not as
    result data, and removed by the suite's cleanup on 2026-10-01
    (`docs/SEED_AUDIT_200_219.md` §6.5, §6.12);
  - the policy was a **2-batch toy training** (`--episodes 32
    --episode-ticks 200 --batch-size 16` on the training topologies 42–49
    and 100–139; two replicates in attempt 1, three in attempt 3), not the
    study's training;
  - the code was **superseded**: the pre-merge run-chain trees `0592f26`
    and `79e15b2`, without `sixg_sim/link_state_service.py`,
    `sixg_sim/postcards.py` or `sixg_sim/reward_spec.py` (no frozen reward
    object; the environment-selected `sixg_sim/reward_profile.py`), with
    the placeholder observation schema `placeholder-pre-WP6` (and, read
    from the code of those trees in git, the old 60/62-wide observation and
    the eight-head action space with handover, scheduler, postcard and IOPS
    heads);
  - the primary analysis (S7) **crashed on both of its attempts in each
    rehearsal** (four failures). It did not refuse the set (the refusal of
    600-tick series came with the trackB merge): it passed its provenance
    and key-union gates, **computed the descriptive block, the primary
    family P1–P9 and the S1 secondary tests in memory**, then raised
    `AttributeError: 'float' object has no attribute 'numerator'` in
    `statistics.stdev` at `confirmatory_analysis.py:277`, the first
    secondary test that reads the pre-surge window (`S2_B_change_vs_xldet`,
    a NaN from the 900–1400 window of a 600-tick series). It **wrote no
    report and printed no statistic** — every print and the JSON dump come
    after that line, `rehearsal_cf_report.json` does not exist, and the
    captured output holds two scipy warnings and the traceback only — and
    no value was examined. S12 wrote a pass-count and provenance report;
    `untrained_analysis.py` found no input (NOT RUN, exit 90);
  - **refuted:** the "possible" rehearsal manifest of 2026-09-29 for 200–203
    named in the earlier text did not happen — the merged tree's rehearsal
    of that day started at 14:55:49 UTC, after `31b034c` (14:27:34 UTC), and
    built `rehearsal_manifest_220_223.json` (archived in
    `~/MARL_run_9/output/rehearsal_archive_a33fdbc_20260930T074340Z/`);
  - **reported for the author's classification** (unexpected under the
    freeze brief's rule until classified; `docs/SEED_AUDIT_200_219.md`
    §6.6 (e)): the launcher-test stand-ins above with their logs
    (`~/MARL_run_9/logs/chain_test.log` 38 lines, `~/MARL_run_9d/logs/chain_test.log`
    49, `~/MARL_run_9d/confirmatory/chain_test.jsonl` 2,
    `~/MARL_run_9d/logs/chain_test/s0_manifest.log` 1, and one history line
    of `b43316a`), proposed class "test fixture"; and copies of the tree
    under `~/MARL_run_9/output/wp13_ref_823620d/` and `output/wp13_stage_*`
    (28 files, 50 lines naming the real profile), proposed class
    "planning";
  - everything else is planning (the real profile, the tests that pin it,
    the planning documents, `docs/RUN_CHAIN.md`'s one line naming the
    block), the audit's own reports under `~/MARL_run_9b/output/` and
    `~/MARL_run_9/output/`, or the known `agent.py` docstring false
    positive (re-wrapped in the merged tree so that the driver's
    `--false-positive` pattern no longer matches it).
  No metric from any of these files has been opened for this record, and
  none may be: the audit records which files they are, not what they hold.
- **Decision on the block, as it actually stands (2026-10-01).** The
  analysis lead **RECOMMENDED keeping 200–219 as pre-registered and
  disclosing the rehearsal exposure.** The author was asked, did not choose
  explicitly, and instructed the work to proceed. This is the lead's
  recommendation, not the author's decision. It can still be reversed until
  the real run touches the block — which it does at its very first
  invocation (`s0_manifest` builds the block's manifest) — so a reversal
  must be made before the launch (`docs/FREEZE_CHECKLIST.md` step d).
  Reasoning of the recommendation: sixteen of the twenty seeds were never
  touched; the four that were saw only a reduced pipeline test (600-tick
  passes) with a toy policy on superseded code, and no statistic from it
  was ever written, printed or examined; switching the pre-registered block
  after the fact would raise the sharper question of what had been seen.
  Note for a reversal: 220–239 is no longer an untouched fallback (the
  full-scale dry run used 220–239, the rehearsals 220–223, the test profile
  220–221); a replacement block would need its own clean audit.
- **DISCLOSURE (text for the paper; `docs/SEED_AUDIT_200_219.md` §6.8):**
  "The confirmatory block (topology seeds 200–219, with matching scenario
  seeds) was fixed before training. Four of its seeds (200–203) were used
  on 24 September 2026, before the simulator corrections were complete, in
  two attempts of a reduced-scale rehearsal of the run pipeline: 216
  simulation passes of 600 ticks instead of 3 600, with policies trained
  for two 16-episode batches of 200 ticks, on superseded versions of the
  simulator without the corrected control plane, observation, action space
  and reward. The rehearsal's confirmatory analysis crashed in all four of
  its attempts, at a secondary test, after it had computed the primary
  tests in memory and before it wrote or printed any result; no statistic
  from those passes was reported or examined, and beyond the pipeline's
  mechanical checks (pass counts, provenance, and whether each ablation
  changed its series at all) none of their outputs was analysed. The block
  was kept as pre-registered. The other sixteen seeds carry no result data
  from any earlier run."
- **Generation.** `build_benchmark_manifest.py --seeds 200-219 --out
  confirmatory/final_manifest_200_219.json`, run by chain stage S0 **at the
  start of the run, before training** (RUN_CHAIN.md §5 item 4; the earlier
  text generated it after the development gate). Its sha256 is sealed in
  `FREEZE_FINAL_real.txt` at S5 and copied into §13 (**freeze 2**).
- **Building the manifest at S0 is PLANNING, not exposure** (decision
  D-FIX.9, FIX-PREFREEZE item 9; the same rule in
  `docs/SEED_AUDIT_200_219.md` §5 and §6.11). The manifest holds
  topology-derived planning data only — per instance the attainable
  optimum, whether the instance is reunifiable, the raw component count and
  the severance tick, computed from the topology and the scenario's events
  without running any arm — and no arm's result. Seeing it cannot inform
  any choice the pre-registration leaves open, so its generation before
  training does not expose the block. Consequences: the seed audit at
  freeze 1 runs before S0, as before; at freeze 2 the real run's own S0
  manifest and the chain's lines naming it (`logs/chain_real/s0_manifest.log`,
  the ledger) are class (b) planning references (§13), and a hit anywhere
  else — any S1–S5b artefact in particular — is class (d); the audit is not
  required "before the manifest is built".
- **Exclusions.** None. A failed invocation is repeated once, unchanged; any
  drop is reported with its cause.

## 6. Arms and runs

Chain stages of `confirmatory/chain_plan.py` (profile `real`), every pass
3 600 ticks, `PYTHONHASHSEED=0`, `nice -n 19`, `MARL_POOL_WORKERS` = 24,
one harness invocation per row (`run_timeline_comparison.py --seed-list
<seeds> --arms <arms> --no-figures --out final/<id>.pkl`):

- **S6, confirmatory, 5 invocations** `s6_cf_b0` … `s6_cf_b4` of 4 seeds × 8
  arms × 2 scenarios = 64 passes each (320): MARL-RIC (`marl_static`: frozen
  executor, selected checkpoint, argmax heads and Dirichlet mean), XL-DET
  (`xldet`), SDN/OpenFlow (`sdn`), OSPF, OLSR, B.A.T.M.A.N., AODV (`ospf`,
  `olsr`, `batman`, `aodv`) and uniform random (`random`).
- **S6h, the analysis-S8 repeat, 1 invocation** (`s6h_hash1`, 4 passes,
  after S7, never shed): `marl_static`, selected checkpoint, the two lowest
  seeds of the block, both scenarios, `PYTHONHASHSEED=1` (§10 S8).
- **S8, untrained-actor control, 4 invocations** (`s8_untrained_s<seed>`, 40
  passes each): `marl_static` executing each of the four initial actors
  produced by `make_untrained_checkpoints.py 1234 2345 3456 4567` at S0
  (minted under the current schema id).
- **S9, adapt-then-freeze variant** `marl_freeze` (secondary), one serial
  invocation of 40 passes. Its online-learning reward is the frozen
  object's mixture `reward_spec.mix(local, team)` = 0.4 · the agent's local
  reward + 0.6 · the simulator's team objective of the tick (κ = 0.6), the
  reward the replicates were trained on (FIX-PREFREEZE item 3, D-FIX.3; it
  was `0.5 × total_reward + 0.5 × (0.4·reachable_ue_fraction +
  0.3·intra_island_reachability + 0.3·ue_to_ue_routed_norm)` until then,
  D8.11, §15 item 7). Its `mappo_trainer.collect` call is unguarded, so a
  refused transition ends the pass loudly (FIX-PREFREEZE item 4, D-FIX.4;
  it sat inside `try: … except Exception: pass` until then, D6.41, §15
  item 8).
- **Evaluation-only sensitivities on the selected checkpoint, descriptive:**
  **S10 "installation off"** and **S11 "re-pointing off"**
  (`MARL_RELAY_REPOINT=off`), 40 passes each, `marl_static`. **The S10
  switch is `MARL_LS_INSTALL=0`**, the pre-registered "installation off,
  idealised timing" sensitivity of RERUN_SPEC WP3 that the engine and
  `docs/LINK_STATE_SERVICE.md` §7 name: a confirmed transition takes effect
  at once, detection and billing unchanged, so it isolates the cost of the
  flood + SPF installation model alone. (Until FIX-PREFREEZE item 1 the
  chain set `MARL_DETECT_TICKS=0` here, which removes the link-state service
  entirely — the "omniscient ablation" of `sixg_sim/link_state_service.py`;
  §15 item 3.) Each sensitivity
  must differ from its S6 baseline on its declared series
  (`tools/stage_report.py --ablation-diff`); S11 is recorded NOT EXERCISED
  if the baseline never re-pointed.
- The stage numbers S0–S12 of the chain and the analysis labels S1–S12 of
  §9–§10 are different namespaces (chain S8 = the untrained control; analysis
  S8 = hash invariance).

## 7. Metrics

All metric helpers live in `confirmatory_analysis.py` (WP9, commit
`9ad2c82`, described in `docs/WP9_ANALYSIS.md`) and are imported by every
other script (`dump_results_json.py`, `extract_timelines.py`; WP11); the
definitions below are copied from those helpers. For a run record *r* of one
arm on one instance, the per-tick series are `r['conn']` (achievability, %),
`r['delivered']`, `r['offered']`, `r['energy']`, `r['fragments']`
(relay-inclusive component count), `r['fragments_raw']` (components of the
physical infrastructure graph), and the WP4 churn series `relay_created`,
`relay_removed`, `relay_removed_by_reason`, `relay_teardown_holds`,
`bridge_teardowns`, `idle_bridge_teardowns`, `relay_mode_flips`, the per-link
records `relay_lifetimes`, the invariant counters `stale_edges`,
`stale_u2u_volume`, and the billing series `overhead`, `overhead_by_type`,
`energy_ctrl` (the contract in `docs/TRACKB_HANDOFF.md` §1–§2; every series
holds the count **in that tick**). `num(xs)` keeps the numeric, non-NaN
entries of a series; `W` = 500. A metric whose series is absent is reported
as "not recorded", never substituted.

### 7.1 Primary metrics (unchanged from the first pre-registration)

- **`steady(xs)`** = mean of `num(xs[-W:])`, the last 500 ticks.
- **Achievability** (per instance) = `steady(r['conn'])`.
- **Energy per delivered unit** (per instance) = `jpu(r)` =
  `sum(num(r['energy'])) / sum(num(r['delivered']))` over the whole run.
  Pooled J/unit over a set = ratio of the two sums over the set
  (descriptive only).
- **Components** = `steady(r['fragments'])`; an instance is **at optimum**
  iff `steady(fragments) ≤ attainable_optimum + 1e-9`, the optimum being the
  manifest's `attainable_optimum` for (scenario kind, seed).
- **`wmean(xs, lo, hi)`** = mean of `num(xs[lo:hi])`. **Pre-surge level**
  (Scenario B) = `wmean(conn, 900, 1400)`; **pre-event level** (Scenario A)
  = `wmean(conn, 0, 200)`.
- **Surge change** (Scenario B) = `steady(conn) − wmean(conn, 900, 1400)`.
- **Service-loss integral** (Scenario A) = `sli(r, 200)`: with
  `a_ref = mean(conn[:200]) × mean(offered[:200]) / mean(offered[200:])`,
  SLI = Σ_{t ≥ 200} max(0, a_ref − conn[t]) (a non-numeric `conn[t]` counts
  as 0).
- **Best routing** (per instance): the maximum achievability, or the minimum
  J/unit, over `ospf, olsr, batman, aodv`; for S2 the largest surge change
  and for S3 the highest pre-surge level over the same four.
- Windows: `T_EVENT` = {A: 200, B: 1400}; `PRE_SURGE` = (900, 1400).

### 7.2 Stability metrics (new, WP4/WP9)

Notation: f = `fragments`, r = `fragments_raw`, o = `attainable_optimum`,
t_s = severance tick (`T_SEV`: A 200, B 0; a separate constant from
`T_EVENT`), T = len(f) = 3600 (`T_TOTAL`, asserted). Post-event window =
[t_s, T); change counts use t ∈ [t_s+1, T) so that the severance transition
itself is not counted; steady-state window = [T − W, T). Comparisons with o
use a tolerance `EPS` = 1e-9. All series must be numeric on every tick
(hard assert §10); nothing is skipped.

| ID | helper | definition |
|---|---|---|
| **M1** fragment-count changes | `m1_fragment_changes(r, ts)` | \|{ t ∈ [t_s+1, T) : f[t] ≠ f[t−1] }\| |
| **M2** policy-induced splits | `m2_policy_splits(r, ts)` | \|{ t ∈ [t_s+1, T) : f[t] > f[t−1] and r[t] ≤ r[t−1] }\| |
| **M3** optimum losses | `m3_optimum_losses(r, ts, o)` | \|{ t ∈ [t_s+1, T) : f[t−1] ≤ o < f[t] }\| |
| **M4** fraction of post-event ticks at the attainable optimum | `m4_fraction_at_optimum(r, ts, o)` | mean over t ∈ [t_s, T) of 1[f[t] ≤ o] |
| **M4ss** | `m4ss_steady_at_optimum(r, o)` | the same mean over t ∈ [T−W, T) |
| **M6** settling time | `m6_settling_time(r, ts, o)` | (last t ∈ [t_s, T) with f[t] > o) + 1 − t_s; 0 if f[t] ≤ o on all of [t_s, T); **censored** at T − t_s, and flagged, if f[T−1] > o; the number of censored instances is reported with every summary |
| **M7** relay-link teardowns per 1000 post-event ticks | `m7_teardowns_per_1000(r, ts)` | 1000 × Σ_{t ∈ [t_s+1, T)} `relay_removed`[t] / (T − t_s − 1), all teardown reasons |
| **M8** bridge and idle-bridge teardowns per run | `m8_bridge_teardowns(r)` | whole-run sums of `bridge_teardowns` and of `idle_bridge_teardowns` (usable passes == 0); two numbers |
| **M9** relay-mode flips per 1000 ticks | `m9_mode_flips_per_1000(r)` | 1000 × Σ_{t ∈ [0, T)} `relay_mode_flips`[t] / T over the whole run (executed mode; agent actions are locked before each site's own flag, so their pre-event flips are 0; the OSPF/OLSR/SDN baselines switch relay modes on their own convergence rule and their M9 is not 0, D4.12) |
| **M10** | `m10_steady_fragment_changes(r)` | \|{ t ∈ [T−W, T) : f[t] ≠ f[t−1] }\| |
| **M11** median bridge lifetime | `m11_median_bridge_lifetime(r, T)` | over the `relay_lifetimes` records with `was_bridge`: lifetime = `removed` − `up` (`up` falling back to `created`); a bridge still up at T is right-censored at T − `up`. The reported value is the **Kaplan–Meier median** (`km_median`): the smallest completed lifetime t at which the product-limit survival estimate S(t) = Π_{t_i ≤ t} (1 − d_i/n_i) is ≤ 0.5 (d_i completed lifetimes equal to t_i, n_i records with lifetime ≥ t_i, a record censored at t_i still at risk at t_i); **undefined (null) when S never reaches 0.5**, which is reported as such with `n_censored`, never replaced by a substitute. Two descriptive companions are printed next to it and are not the metric: the plain median over completed lifetimes only (`median_completed`, biased upwards under censoring) and the plain median with censored records entered at T − `up` (`median_censored_at_T`, a lower bound). `n`, `n_completed`, `n_censored` accompany every summary |

`stability_metrics(r, ts, o, T)` returns all eleven for one run and is what
the export writes per run. Severance ticks: A 200, B 0, taken from the
scenario events of the run (never hard-coded, D0.5); the manifest's
`severance_tick` must equal `T_SEV` (hard assert). For Scenario B, M1–M4 are
also reported over the surge window [1400, T) (descriptive only).

Summaries of the stability metrics are **medians with inter-quartile ranges**
(`dstats`: median, q25, q75, mean, min, max; the counts are heavy-tailed: M1
ranged from 1 to 1043 on the superseded set); means are printed next to them
but are not the reported statistic.

On pickles of the frozen (uncorrected) tree, which lack the WP4 series, two
explicitly named **post hoc lower bounds** (`m7_lower_bound_relay_count`,
`m8_lower_bound_relay_bridges`, decrements of the net `relay_count` /
`relay_bridges` series) may appear under their own `lower_bound … post hoc`
keys; they are never used for a study pickle and never enter S9.

### 7.3 Overhead

Billed control-plane traffic per message type (BFD, hello, Router-LSA,
postcard; `overhead_by_type`, in Mb/s per tick like `overhead`) from
`tick_messages`, from island onset; MARL-path arms only — the baselines have
no per-type decomposition and their `overhead_by_type` is absent ("not
recorded", never zero; D3.18, D4.9). Descriptive only (S11). Not a
comparative KPI.

## 8. Primary hypotheses (Holm over 9, α = 0.05)

These are the same nine hypotheses, decision test and rule as the first
pre-registration, with MARL-RIC = `marl_static`: exact two-sided Wilcoxon
signed-rank test (`wilcoxon(d)`: zeros dropped, mid-ranks for ties, exact
enumeration of the signed-rank distribution) on the 20 paired per-instance
differences; Holm step-down over the nine p-values (`holm`). A hypothesis is
**confirmed** iff its Holm-adjusted p < 0.05 **and** the signed-rank statistic
lies in the expected direction (`W+ > W−` for "> 0", `W− > W+` for "< 0").

| ID | Scen. | Quantity | Comparison | Expected |
|---|---|---|---|---|
| P1 | A | achievability | MARL-RIC − XL-DET | > 0 |
| P2 | A | achievability | MARL-RIC − SDN | > 0 |
| P3 | A | achievability | MARL-RIC − best routing | > 0 |
| P4 | B | achievability | MARL-RIC − XL-DET | > 0 |
| P5 | B | achievability | MARL-RIC − SDN | > 0 |
| P6 | B | achievability | MARL-RIC − best routing | > 0 |
| P7 | A | energy per unit | MARL-RIC − XL-DET | < 0 |
| P8 | A | energy per unit | MARL-RIC − SDN | < 0 |
| P9 | A | energy per unit | MARL-RIC − best routing | < 0 |

The descriptive statistics are the paired mean, the 95 % t-interval
(`t_ci`), wins and the exact sign test (`sign_p`), as before. When they
disagree with the decision test the disagreement is reported; they never
override it. P2, P5 and P8 are read together with §2 item 10.

## 9. Stability family S9 (Holm over 6, α = 0.05)

The comparison is MARL-RIC − XL-DET, per scenario, on the per-instance values
of §7.2, using the exact two-sided Wilcoxon test of §8 (`paired(d, expect)`)
and Holm over the six (`S9_ENDPOINTS`).

| ID | Scen. | Quantity | Expected (from the development data) |
|---|---|---|---|
| S9a | A | M2 | > 0 |
| S9b | B | M2 | > 0 |
| S9c | A | M4 | < 0 |
| S9d | B | M4 | < 0 |
| S9e | A | M7 | > 0 |
| S9f | B | M7 | > 0 |

Rules (`stability_family`, `MIN_NONZERO` = 6):

- An endpoint with **fewer than six non-zero differences** is **not
  assessable** (the smallest exact two-sided p for n ≤ 5 is 0.0625). It is
  reported as such with its differences; it enters the Holm step-down with
  p = 1, so the family size remains 6 and it can never be confirmed. An
  endpoint whose metric is not recorded on some instance is "not recorded"
  and is treated the same way (this cannot happen on a study pickle: the
  hard asserts require the series).
- An endpoint is **confirmed in the expected direction** iff its Holm-adjusted
  p < 0.05 and the statistic has the expected sign; a Holm-adjusted p < 0.05
  with the opposite sign is reported as "MARL-RIC worse".
- **Reunification may be called monotone** for an arm only if its M2 = 0 on
  every instance of the scenario.
- **The learned policy's bridging may be called stable** only if S9 does not
  show it worse than XL-DET (Holm-adjusted p < 0.05 in the expected
  direction, i.e. MARL-RIC has more splits or more teardowns) on M2 or M7 in
  either scenario. "Not assessable" endpoints neither support nor refute the
  claim.
- No other stability comparison is inferential; S10 is descriptive.
- XL-DET instances that settle away from the optimum under the grace rule
  (§2 item 4) are part of the comparator's data and are reported, not
  removed.

## 10. Other secondary analyses (outside every Holm family)

S1–S6 are as in the first pre-registration (definitions §7.1).

- **S1** reunification: at-optimum counts per arm, and the paired
  MARL-RIC − XL-DET component difference, both scenarios.
- **S2** surge response (Scenario B): the paired difference in surge change
  against XL-DET, SDN and the best routing protocol; the manuscript may say
  MARL-RIC *responds to the surge* better only if S2 against that baseline has
  Wilcoxon p < 0.05 in the positive direction.
- **S3** pre-surge level (Scenario B) against the three comparators.
- **S4** absolute energy per tick in Scenario A, MARL-RIC − XL-DET.
- **S5** service-loss integral (Scenario A), medians per arm and MARL-RIC −
  XL-DET.
- **S6** energy per unit in Scenario B, MARL-RIC − XL-DET.

New or changed:

- **S7:** MARL-RIC versus the adapt-then-freeze variant (achievability and
  components, both scenarios, on the instances both arms completed; computed
  by `confirmatory_analysis.py` only when the `marl_freeze` pickle (chain S9)
  is among its `--pickles`).
- **S8:** hash invariance. The selected checkpoint on **seeds 200 and 201**
  (the two lowest of the block), both scenarios, under `PYTHONHASHSEED` 0
  and 1 must give identical decision, delivered and fragment series
  (`tools/hash_invariance_check.py --require-decision`, bit-level rule of
  `tools/series_diff.py`). The decision series is the per-tick
  `decision_digest` of the executed joint decision, which the frozen tree
  records for every arm (D4.7: one row per radio site with the sampled head
  indices and the executed relay mode, tx power, MCS levels and PRB split);
  every other per-tick series and per-pass scalar is compared as well, except
  energy and latency, which are reported as accounting jitter. The
  `PYTHONHASHSEED=0` side is the S6 invocation `s6_cf_b0`; the
  `PYTHONHASHSEED=1` side is one extra `marl_static` invocation on those two
  seeds.
- **S10:** M1, M3, M4ss, M6 (with censored counts), M8, M9, M10 and M11 for
  every arm, medians and IQRs, both scenarios.
- **S11:** billed control overhead by message type, per arm, descriptive,
  with the disclosures of §2 item 3.
- **S12:** the untrained-actor control (four initial actors, chain S8)
  against the trained policy, random and the best routing protocol:
  achievability, components, J/unit, descriptive.
- **Shared-world check:** the largest pre-event achievability difference
  between arms in Scenario A (expected 0).

**What the chain computes (§15 item 4, resolved by FIX-PREFREEZE item 2).**
The chain's S7 calls `confirmatory_analysis.py` on the five S6 pickles only
(P1–P9, S9, S1–S6, S10, S11 and the shared-world check). The three analyses
that need the stages after S7 are computed by the chain's S12
(`confirmatory/stage_analysis.sh`, step `secondary`):

- **S8:** chain stage **S6h** (`s6h_hash1`, after S7, never shed) runs the
  selected checkpoint, `marl_static`, on the two lowest seeds of the block,
  both scenarios, under `PYTHONHASHSEED=1` (the plan entry's `hashseed`;
  `chain_write_env` exports it and the completeness gate requires the
  pickle to record it); its pair is the S6 invocation at the same tick count
  that holds both seeds (`s6_cf_b0`). S12 checks the pair's provenance
  (same tree head, tree digest, schema id and checkpoint; the two declared
  hash seeds; `tools/stage_report.py --hash-pair`) and runs
  `tools/hash_invariance_check.py --arms marl_static --seeds <the two>
  --require-decision --out confirmatory/final_s8_hash.json`. Exit 2 (nothing
  compared, or the precondition not established) fails the stage; exit 1 (a
  difference) is the analysis's result and is reported as such.
- **S12 and S7 (secondary):** one further run of the same analysis script
  over exactly S7's primary input plus the chain-S9 pickle, with
  `--untrained untrained_s<seed>=<chain-S8 pickle>` for each of the four
  untrained actors and `--s8 confirmatory/final_s8_hash.json`, written to
  `confirmatory/final_cf_report_full.json`; S12 checks by counting that the
  four S7-secondary comparisons, one S12 block per control (both scenarios)
  and the embedded S8 block are present, and requires
  `tools/compare_reports.py` to find every leaf of the chain's primary report
  (seeds, excluded, descriptive, primary, secondary, stability, S10, S11)
  reproduced. A missing input fails the stage unless the ledger records the
  operator shedding that chain stage (S8 or S9), which is then reported as
  not computed. `untrained_analysis.py` (the first study's script, hard-wired
  to seeds 80–99) is no longer called.

None of the three is a primary analysis, and none can change P1–P9 or S9
(the comparison above enforces it).

Hard asserts of the analysis (`check_invariants`; the script refuses to
report otherwise, `--no-assert` exists only for the legacy pickles of §11):
(0) the report's `config.marl_arm` equals the pre-registered MARL-RIC arm
`marl_static` (`PREREG_MARL_ARM` in `confirmatory_analysis.py`) and
`config.arms` contains it — `--marl-arm` selects the arm reported as
MARL-RIC, and `marl_freeze` is the *secondary* adapt-then-freeze variant of
S7, so an analysis run for that arm would otherwise produce a complete and
plausible P1–P9 and S9 report under the label MARL-RIC; the flag's default is
`marl_static` and §12 pins the invocation;
(1) every arm and every control has exactly the manifest's 20 seeds in both
scenarios — the adapt-then-freeze arm included as soon as any of its runs is
present — and the manifest's `severance_tick` equals `T_SEV`; (2) every
pickle carries a provenance record; `tree_head`, `tree_sha256`, `schema_id`
and `pythonhashseed` are each present (a missing, null or empty field fails,
even when it is missing from every pickle) and identical across every pickle
of the analysis (D0.2); `pythonhashseed` = '0'; no record is INCOMPLETE;
`total_ticks` = 3600 / `severance_tick` = 200 where recorded; and
`checkpoint_sha256` is present and identical across every pickle that
carries `marl_static` (the five four-seed batches), controls excluded; (3)
`stale_edges` and `stale_u2u_volume` sum to 0 in every run; (4) for every
run of a MARL-path arm (`marl_static`, `marl_freeze`, `xldet`, `random`, the
untrained controls) **every site** sends more than one postcard after the
event: min over sites of `postcards_by_site` (or `postcards_min_per_site`)
> 1 **and**, for `postcards_by_site`, one entry per agent site
(`len(postcards_by_site)` = `n_agent_sites`, falling back to
`n_radio_sites`) — completeness is part of the assert, because a site that
never sent is absent from a dict the engine fills from senders and the
minimum over the entries present would then pass, which is exactly the
defect class the per-site check exists for; the scalar
`postcards_min_per_site` carries no site list and is accepted without a
completeness claim; the per-site recording of TRACKB_HANDOFF §1 is what the
frozen tree writes (`postcards_by_site` over a per-tick roster,
`n_agent_sites` its size; D4.13); the aggregate Σ `postcards`[t_s:] >
`n_radio_sites` is checked as well; (5) no NaN, None or non-numeric
entry in any numeric per-tick series, top-level lists, numpy arrays and the
members of `overhead_by_type` / `relay_removed_by_reason` /
`relay_teardown_holds` alike, except NaN in `latency` (undefined on a tick
without delivery; the exempt counts are reported); numpy scalars are
counted, never dropped; (6) run length: **every** numeric per-tick series of
a run — the top-level lists and numpy arrays and the members of the
`{name: series}` keys, not `conn` alone — has exactly `T_TOTAL` = 3600
entries, because a short `fragments` / `fragments_raw` / event series
shortens the M1–M7/M10 windows silently and yields a partial-window number
in place of the metric; (7) every `was_bridge` record of `relay_lifetimes`
carries a lifetime: a start tick (`up`, else `created`) that is a real
number in **[0, 3600)**, and `removed` either None (still up at T:
right-censored) or in [start, 3600).  A record without a start tick, a
record whose start tick is outside [0, T) or not a number (NaN included),
and a record with `removed` ≥ T (the engine writing T instead of None,
TRACKB_HANDOFF §2) are refused and reported, never dropped silently and
never counted as a completed lifetime by M11.  The start tick is bounded
for the same reason `removed` is and against the same engine bug: `up` = T
yields a censored lifetime of exactly 0, `up` > T a **negative** one and
`up` < 0 one longer than the run, and each of them would be counted in
n / n_censored and fed to `km_median` and `median_censored_at_T`.

Before any analysis the chain additionally requires, as counting gates that
fail on zero: identical tree head, tree digest, schema id and hash seed
across the pickles analysed (`tools/stage_report.py --cross-provenance`) and
a key union equal to the expected confirmatory key set, with no duplicate
and none missing (`--key-union`).

The S8 check itself must not be vacuous: `tools/hash_invariance_check.py`
reports IDENTICAL only when it actually compared something (`n_compared`
> 0, and a decision series under `--require-decision`); an empty selection
exits 2 with the reason in `nothing_compared`, and the embedded `S8` block
carries it.  It must also not be a repeat-run check wearing S8's name: the
two pickles must record **different** `pythonhashseed` values in their
provenance (`same_hashseed` is False).  Equal seeds, a missing provenance
record or a missing `pythonhashseed` mean the pair could not have exposed a
hash-order dependence whatever its series say, so `identical` is false, the
reason is in `precondition_not_established`, the CLI exits 2, and the S8
line of the report prints `same_hashseed` and the refusal rather than the
word "identical" alone.

## 11. Validation of the analysis plan (done before freeze 1)

`confirmatory_analysis.py`, run on the frozen first-study pickles
(`--no-assert`, because those pickles predate the provenance record and
carry the postcard defect; commands in `docs/WP9_ANALYSIS.md` §7 and
`tools/wp9_validate.sh`), was required to:

1. reproduce `~/MARL_run_6/confirmatory/cf_report_A1.json` for every
   P1–P9 and S1–S7 entry, value for value (`tools/compare_reports.py`), from
   `~/MARL_run_5/output/cf_eval_80_99.pkl` +
   `~/MARL_run_6/output/cf_static_80_89.pkl` + `cf_static_90_99.pkl` with
   `benchmark_manifest_80_99.json` and `--marl-arm marl_static`;
2. reproduce `dev_report_A1.json` the same way from
   `~/MARL_run_5/output/ce_eval_50_59.pkl`, `ce_eval_70_79.pkl`,
   `~/MARL_run_6/output/dev_static_50_59.pkl`, `dev_static_70_79.pkl` with
   `benchmark_manifest.json` and `benchmark_manifest_70_79.json`;
3. reproduce, from the same frozen pickles, the M1, M2 and M4ss values
   recorded in the mapping notes for `marl_static` and XL-DET in Scenario A
   (M1 for seeds 80–99 in order: 7, 4, 1043, 4, 3, 1, 5, 3, 2, 39, 5, 895,
   3, 2, 3, 923, 1, 4, 3, 15; XL-DET M2 ∈ {0, 1} on every seed; XL-DET M4ss
   = 1.0 on 20/20; `marl_static` M4ss = 1.0 on 11/20, ≈ 0.43 on seeds 91 and
   95, 0 on 7 seeds); the post hoc lower bounds of M7 and M8 on those
   pickles are compared against the figures in the notes (M7 lower bound
   `marl_static` A 3.5–1403, median 477; XL-DET 0–1.2; random 1023–1163;
   routing and SDN 0);
4. pass its unit tests (`tests/test_wp9_*.py`).

Items 1 and 2 were met (665 leaves each, 0 mismatches; re-confirmed at every
Track B round through 2026-09-24, `docs/DECISIONS.md` D12.4 item 4); the
validation record is `docs/WP9_ANALYSIS.md` §8 and §11–§12. The
`tests/test_wp9_*` files run in every suite run. The script's sha256 is part
of the freeze-1 record (§13). Any later change to the script requires a new
freeze.

## 12. Frozen evaluation environment and launcher settings

Every invocation of the chain runs a job script that sources a per-invocation
environment file written once by `confirmatory/chain_lib.sh`
(`chain_write_env`, `confirmatory/env_real/<inv>.sh`) and reused verbatim by
any re-run: `PYTHONHASHSEED=0`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `PYTHONUNBUFFERED=1`,
`MARL_POOL_WORKERS`, the invocation's `MARL_*` set from
`confirmatory/chain_plan.py`, `MARL_EVAL_CHECKPOINT` where a checkpoint is
used, and `MARL_CHAIN_INV` (bookkeeping, read only by the preflight's job
scan). The provenance record (`marl_env`) of every pickle and checkpoint
stores every `MARL_*` variable in force. **Each invocation starts from an
EMPTY environment** (FIX-PREFREEZE item 5, D-FIX.5): its job script
re-executes itself under `env -i` and sources only its env file, which
also records the system variables a process needs (`HOME`, `PATH`, `USER`,
`LOGNAME`, `SHELL`, `LANG`, `LC_ALL`, `TMPDIR`, `PY`, those that are set),
so no variable of the operator's shell, of `register_chain.sh` or of the
watchdog reaches it. Before the launch the preflight runs the job script
with `--print-env` from the launcher's own environment and requires its
`MARL_*` set to equal the plan's (`confirmatory/env_<profile>/<inv>.marl_env`:
the invocation's `MARL_*` set from `chain_plan.py`, `MARL_POOL_WORKERS`,
`MARL_EVAL_CHECKPOINT` where used, `MARL_CHAIN_INV`), the declared hash seed,
the thread pins, and no other variable; after the run the completeness gate
requires the pickle's (or the checkpoint's) recorded `marl_env` to equal the
same set exactly. A mismatch stops the invocation.

| variable | value in the chain (profile `real`) | read by | state in the frozen tree |
|---|---|---|---|
| `PYTHONHASHSEED` | `0` | every driver (banner if not); the preflight refuses anything else | — |
| `OMP/MKL/OPENBLAS_NUM_THREADS` | `1` | env file; drivers also pin at import; the preflight refuses anything else | — |
| `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `PYTHONUNBUFFERED` | `1` | env file | inert on <compute-server> (D-DET.11) |
| `MARL_POOL_WORKERS` | **24** for every evaluation invocation, **16** per training replicate (two alive), 1 for tool steps | `runinfo.pool_workers` | the 16/8 load rule of the shared machine is NOT applied to `real` (RUN_CHAIN.md §5 item 5); memory ceilings (16 below 12 GB, 8 below 8 GB MemAvailable) can never fire behind the 18 GB / 26 GB gates |
| `MARL_TRANSPORT_POWER` | `planning` | `Simulator.__init__` | live switch, evaluated value |
| `MARL_RELAY_STATE` | `reconcile` | `Simulator.__init__` | live switch, evaluated value |
| `MARL_RELAY_REPOINT` | `fragment_aware` (S11: `off`) | `Simulator.__init__` | default = evaluated value since WP5; `legacy` refused; training refuses `off` |
| `MARL_POSTCARD_ALWAYS` | `1` | `Simulator.__init__` | tripwire: anything but 1/true/on/yes is refused (the heartbeat is an environment rule) |
| `MARL_REWARD_PROFILE` | `mission` | `reward_spec` at import | tripwire: anything but `mission` is refused (the frozen object is hard-wired) |
| `MARL_RELAY_ENTROPY_FLOOR` | `0.01` | `mappo_trainer` (training only) | live |
| `MARL_STEER_MODEL` | `actionable` (evaluation invocations) | `simulation.STEER_MODEL` | default `actionable` |
| `MARL_EVAL_CHECKPOINT` | the selected checkpoint (S4, S6, S9–S11, S3 per replicate); an untrained actor (S8); a placeholder untrained actor on S1 that is never opened | harness, only for `marl`, `marl_freeze`, `marl_static` (`CHECKPOINT_ARMS`) | — |
| `MARL_STRESS_INTERFERENCE_DBM`, `_TICK` | `-75`, `2000` on the S1/S4 stress invocations only; unset elsewhere | `Simulator.__init__` | — |
| `MARL_DETECT_TICKS` | unset (3) | `link_state_service` | 0 removes the service (the omniscient ablation); not set by any stage |
| `MARL_LS_INSTALL` | unset (1); **`0` on S10** | `link_state_service` | 0 = installation off, idealised timing (§6) |
| `MARL_UE_RELAY` | unset (`none`) | harness (outage metric's reachability test) | — |
| `MARL_PRB_POLICY` | unset (`dirichlet`) | `agent.PRB_POLICY` | part of the schema id |
| `MARL_RELAY_MASK` | unset (`off`) | `agent` | any other value refused |
| `MARL_AGENT_FLEET` | unset (`radio`) | `simulation.AGENT_FLEET` | — |
| `MARL_RELAY_HYSTERESIS` | unset (`0`) | `simulation` | — |
| `MARL_POSTCARD_DRAIN` | unset (`shared`) | `simulation` | `legacy` refused |
| `MARL_OBS_LOCALITY` | unset | `agent` | refused if set (WP6) |
| `MARL_POSTCARD_FRAGMENTS` | unset | `agent` | refused unless unset/0 (WP5) |
| `MARL_DIAG` | unset | harness diagnostics | unset in every study run |
| `MARL_DIAG_PHANTOM`, `MARL_FIX_PHANTOM` | unset | only `confirmatory/ref_patch_phantom.py` (first study) | never read by the engine |
| `MARL_WP0_TEST_VAR` | unset | `tests/test_wp0_runinfo.py` only | — |
| `UU_CA_MAX_EXTRA_CC` | unset (`1`) | harness | — |
| `PROBE_TICK`, `PROBE_TRAIN` | unset (`dump_endstate.py` default 3400) | S4 end-state dump / one root test | — |
| `MPLBACKEND` | unset | `sixg_sim/simulation.py` | — |

`MARL_BFD_DETECT_TICKS` is a Python constant derived from `MARL_DETECT_TICKS`,
not an environment variable; `MARL_CHAIN_INV` is bookkeeping (above).

Launcher settings: `nice -n 19` throughout; the harness with `--seed-list
<seeds> --arms <arms> --no-figures --out final/<id>.pkl` and no `--ticks`
(3 600 ticks, severance at 200 in A and 0 in B, surge at 1400 in B); the
training flags of §3; the untrained generator `make_untrained_checkpoints.py
1234 2345 3456 4567`; the manifest builder `build_benchmark_manifest.py
--seeds 200-219`; all from the frozen tree on <compute-server>.

**Preflight gates (`tools/preflight.py`, before every invocation, recorded in
`confirmatory/preflight_real/`):** evaluation ≥ 18 GB MemAvailable and
≥ 100 GB free on `/home`; training ≥ 26 GB for the wave (MemAvailable plus
the PSS already held by the chain's own expected replicates) with an abort
level below 22 GB, and a wave starts only at a 1-minute load below 20; tool
steps ≥ 4 GB / 20 GB; no leftover simulator or trainer process of ours in the
tree; `PYTHONHASHSEED` = 0 and the thread pins in the launcher's own
environment; the tree digest equal to the one recorded by the run's FIRST
preflight (pinned for the whole ledger), and after S5 equal to the one
sealed in `FREEZE_FINAL_real.txt`. A refusal is retried every
`--preflight-wait` seconds up to `--preflight-retries` times and then stops
the chain; **the same budget applies to every refusal, so a permanent one
(tree digest, hash seed, thread pins, leftover job) is retried too** (known
issue; the budget is sized for the memory gate, `docs/FREEZE_CHECKLIST.md`
step d). **Unit:** every memory figure of these gates (18, 26, 22, 4) is
GiB: the preflight divides the `/proc/meminfo` kB value by 1024², the unit
the performance track measured the 19.8 GiB PSS training wave in
(`docs/COMPUTE_PLAN.md` §6, D-FIX.14); "GB" in this document's memory gates
means GiB, and the training gate is 27.9 GB decimal. It stays at 26 GiB for
every profile (§16 item 10, resolved by the author on 2026-10-02: host
memory was freed instead, and checklist step d.0 checks it before launch).

**The confirmatory analysis invocations, as the chain runs them**
(`confirmatory/stage_analysis.sh`):

```
# S5b (development set)
nice -n 19 python3 confirmatory_analysis.py \
  --pickles output/final/s1_dev_b0.pkl,...,output/final/s1_dev_b4.pkl,output/final/s4_dev_static.pkl \
  --manifests benchmark_manifest.json,benchmark_manifest_70_79.json \
  --marl-arm marl_static --out confirmatory/final_dev_report.json
# S7 (primary result), after --cross-provenance and --key-union
nice -n 19 python3 confirmatory_analysis.py \
  --pickles output/final/s6_cf_b0.pkl,output/final/s6_cf_b1.pkl,output/final/s6_cf_b2.pkl,\
output/final/s6_cf_b3.pkl,output/final/s6_cf_b4.pkl \
  --manifests confirmatory/final_manifest_200_219.json \
  --marl-arm marl_static --out confirmatory/final_cf_report.json
```

with no `--no-assert`, `--untrained-arm marl_static` and `--nan-exempt
latency` at their defaults, `PYTHONHASHSEED=0` from the launcher. The
chain's S12 adds S7 (secondary), S8 and S12 (§10):

```
# S12 (secondary analyses), after the gates of §10
python3 tools/hash_invariance_check.py output/final/s6_cf_b0.pkl output/final/s6h_hash1.pkl \
  --arms marl_static --seeds <the two lowest seeds of the block> --require-decision \
  --quiet --out confirmatory/final_s8_hash.json
nice -n 19 python3 confirmatory_analysis.py \
  --pickles output/final/s6_cf_b0.pkl,...,output/final/s6_cf_b4.pkl,output/final/s9_freeze.pkl \
  --manifests confirmatory/final_manifest_<block>.json --marl-arm marl_static \
  --s8 confirmatory/final_s8_hash.json --out confirmatory/final_cf_report_full.json \
  --untrained untrained_s1234=output/final/s8_untrained_s1234.pkl,...,untrained_s4567=output/final/s8_untrained_s4567.pkl
python3 tools/compare_reports.py confirmatory/final_cf_report.json confirmatory/final_cf_report_full.json \
  --sections seeds,excluded,descriptive,primary,secondary,stability,S10,S11
```

## 13. Freeze procedure and appended records

- **Freeze 1** (before the real chain starts): the commit synced to
  `~/MARL_run_9` by `sync_run9.sh`, which differs from the rehearsed head
  `02049e3` in `docs/` only (it was `a33fdbc` before FIX-PREFREEZE, §16
  item 9); the rehearsal on `02049e3` ended
  `CHAIN END profile=rehearsal rc=0` (D-FIX.13); the full suite, the routing/SDN
  invariance rows of RERUN_SPEC §14 and the determinism gate pass at the
  synced head; `FREEZE_R2.sha256` = the sha256 of every file in
  `.synced_files` at that head (written under `logs/freeze/` on the server
  and copied to `records/R2/` of the manuscript repository with the UTC
  time); the seed audit of §5 is re-run from that tree with the committed
  driver (`MARL_TREE=$HOME/MARL_run_9 bash tools/run_seed_audit.sh 200
  219`) over a history dump of ALL refs captured from the laptop repository
  at that head, its sha256 identical on both hosts. Exact commands:
  `docs/FREEZE_CHECKLIST.md` steps a–c.
- **The seed-audit gate at a freeze** (`docs/SEED_AUDIT_200_219.md`
  §6.10). The audit can no longer return CLEAN (§5), and the gate is no
  longer "exit 0". **As measured on 2026-09-30 and 2026-10-01 the audit
  returns exit 4 (INCOMPLETE)**, because of the five launcher-test stand-ins
  of §5; the gate cannot be met until they are handled
  (`docs/FREEZE_CHECKLIST.md` step c.0, with the author's consent). After
  that, the gate is **exit 3 (TOUCHED) with every hit classified**,
  together with every coverage condition of `docs/SEED_AUDIT_200_219.md` §5
  (items 2–5 and 7–10: `errors: []`, `coverage_ok`, the count floors and
  per-root floors, every `$HOME/MARL*` root, a scanned history dump with a
  non-zero commit count whose sha256 equals the one captured for the run,
  the tool's unit tests passing on the server, the committed driver, no
  audit output inside a scanned root), from a run during which nothing else
  writes into any scanned tree. If the stand-ins are still present, the
  expected result is exit 4 and the freeze stops. Every hit is placed in
  exactly one class (the measured groups are listed in
  `docs/SEED_AUDIT_200_219.md` §6.6):
  (a) **result data on the four rehearsal seeds** from the two reduced
  rehearsals of 2026-09-24 (§5), verified against their ledgers, logs and
  the pickles' provenance, never against their metrics;
  (b) **planning references**: the real profile in
  `confirmatory/chain_plan.py`, the tests that pin it, the planning
  documents, the superseded run-chain code of `~/MARL_run_9d`, and — at
  freeze 2 only — the real run's own S0 manifest and the chain's logs
  naming it; each classified explicitly with a reason through the tool's
  `--false-positive` patterns or its planning markers, never by weakening
  the scan; also, by the same rule, the audit's own reports and the known
  `agent.py` docstring false positive;
  (c) the groups the author classifies (the launcher-test logs, the
  `output/wp13_*` copies of the tree; §5), once and in writing;
  (d) anything else is an **UNEXPECTED exposure**: the freeze stops and it
  is reported plainly. The result (command, report path and sha256, every
  hit and its class, the disclosure paragraph) is a new dated section of
  `docs/SEED_AUDIT_200_219.md`, and its verdict block is appended here.
- **Appended after selection (S3b):** the selected replicate's torch seed and
  the sha256 of its `policy_final.pt` (`confirmatory/SELECTED_real.txt`),
  the four untrained checkpoints' sha256 (`output/final/untrained_s*/`).
- **Freeze 2** (after S5b, before any confirmatory arm runs): the chain is
  launched in two segments (S0–S5b, then `--from S6`;
  `docs/FREEZE_CHECKLIST.md` step d) so that, between them,
  `confirmatory/FREEZE_FINAL_real.txt` (tree digest, manifest sha256,
  selected-checkpoint sha256) is copied here with the UTC time and the seed
  audit is re-run under the gate above. Any new hit outside classes (a)–(c)
  stops the run before S6. **The author's go for segment 2 was given in
  advance on 2026-10-03 ("Automatic gate"), conditional on that gate:** the
  unattended pipeline (`~/MARL_watch/jobs/final_and_launch.sh`,
  `docs/DECISIONS.md` D12F.9) writes the FREEZE_FINAL / SELECTED / untrained
  records with the UTC time to `logs/freeze/freeze2_<head>.txt` (copied here
  afterwards), re-runs the seed audit, and starts S6 only if its only new
  result-data hit is the real run's own S0 manifest
  (`confirmatory/final_manifest_200_219.json`, class (b)); any other hit
  stops the run before S6 for the author.
- **The first real-run attempt (2026-10-03/04) was stopped before S6.**
  At `bbfd7b0` it ran S0 (the block's manifest and the untrained
  checkpoints), S1 (development and stress seeds only) and half of S2; the
  author held segment 2 on 2026-10-04 when its S1 output showed the XL-DET
  defect of §15 item 21, and then chose to fix it (XL-1) and rerun. No
  confirmatory-block evaluation ran (S6 never started); the S0 manifest is
  planning data (D-FIX.9). Every artefact of that attempt is archived,
  never deleted, in `~/archive_marl_realrun1_20261004.tgz` (160 files,
  sha256 `a52d09f072de92da1bd69a82d2d92f777634dbe779136f82f7fa6f099d77e96b`,
  listing beside it), outside every scanned tree. The real run is the
  second attempt, at the XL-1 head.
- **Records are appended in the local repository and the manuscript
  repository only.** Nothing tracked is synced to `~/MARL_run_9` between the
  first preflight of the real run and its last `CHAIN END`: the tree digest
  is pinned from the first invocation, and a synced documentation change
  would stop the chain.
- Nothing else in this file changes after freeze 1. Any analysis not listed
  here is labelled post hoc wherever it appears.

## 14. How the results enter the manuscript

- Every inferential statement in the abstract and conclusion comes from
  §8–§9 on seeds 200–219.
- The earlier confirmatory test (seeds 80–99) is reported as the first test on
  the uncorrected simulator, with its verdict, its post hoc stale-edge
  re-analysis and the reasons for this study.
- An unconfirmed hypothesis is reported as unconfirmed and its claim is
  removed.
- The seed disclosure of §5, the decisions not implemented (§2 items 9 and
  10) and the S10 switch (§6) are reported with the results.
- Nothing listed here changes after the confirmatory instances are generated;
  any other analysis is labelled post hoc.

## 15. Document–code disagreements found at freeze preparation (2026-10-01)

Each was resolved in favour of the code at `a33fdbc`: the sections above
state what the code does. Items 1–4 change what the study tests or reports;
the author's decisions on them are §16.

1. **UE-transit paths (D12.1) — decided, never implemented.** The draft §2.9
   described path selection restricted to infrastructure interiors for all
   arms (TRACKB_HANDOFF §7.1). `_forward_traffic` and `_best_path_to_any`
   still search the UE-inclusive graph. §2 item 9 now states the code.
   **The author decided on 2026-10-01 to implement it ("Fix both", §16
   item 1); the D12-FIXES package implements it.** **Resolved (D12F.3):**
   every data-plane path and every path metric takes
   `Topology.path_view` (§2 item 9). Measured effect on the final engine,
   seeds 82 and 91, both scenarios, 1200 ticks (D12F.5): before the fix
   the share of post-event UE-to-UE volume carried through a UE was 22.6 %
   / 4.0 % (MARL-RIC, Scenario A, seeds 82 / 91), 27.3 % / 7.5 % (XL-DET),
   24.2 % / 7.9 % (SDN), 12.2 % / 5.0 % (random), 1.8-1.9 % / 4.3-4.4 %
   (OSPF, OLSR, B.A.T.M.A.N., AODV), 6.5-10.3 % / 0.6-2.5 % in Scenario B,
   and 4.2 % / 0.8 % of the shared pre-event volume; after it, 0 on every
   pass (D12F.6). Every arm's delivered fraction ROSE, because a UE
   shortcut crossed two narrow, shared Uu links where the infrastructure
   detour runs on fibre/mesh: MARL-RIC A 0.372 -> 0.425 / 0.298 -> 0.333,
   B 0.289 -> 0.333 / 0.283 -> 0.297; OSPF A 0.373 -> 0.373 / 0.312 ->
   0.324, B 0.398 -> 0.438 / 0.363 -> 0.371; the pre-event world 0.478 ->
   0.550 / 0.442 -> 0.451; per arm and scenario in D12F.5. Connectivity
   and fragment counts did not move for any routing arm; the MARL-path arms
   moved by at most 0.01 fragments, random A/82 by 0.07.
2. **SDN model fixes (D12.2) — decided, never implemented.** The draft §2.10
   described cumulative per-fragment LLDP discovery over infrastructure
   nodes with UEs as leaves (TRACKB_HANDOFF §7.2). `SDNController` is the
   baseline model; every invariance check since then required SDN to stay
   bit-identical. §2 item 10 now states the code.
   **The author decided on 2026-10-01 to implement it ("Fix both", §16
   item 1); the D12-FIXES package implements it.** **Resolved (D12F.4):**
   §2 item 10 states the fixed model. Measured (D12F.5, D12F.6): on the
   first study's instances that never recovered, 87A and 98A now reach
   operational (two controllers each: 19 + 15 and 15 + 18 switches, first
   delivery at severance + 19 / + 18 ticks, 122 / 121 flows programmed
   before the first relay bridge, as the handoff's static predictor said);
   Scenario B is bit-identical to the code before D12.2 (310 / 292 flows,
   222 -> 310 by tick 329 and 222 -> 292 by tick 323); on seeds 82 and 91
   the routing and MARL-path arms are bit-identical to D12.1 alone, and
   SDN's Scenario-A delivered fraction moves 0.493 -> 0.505 / 0.460 ->
   0.468 (1200 ticks), connectivity 95.0 -> 96.8 % / 97.4 -> 98.1 %, mean
   fragments 1.31 -> 1.20 / 1.31 -> 1.23, with two controllers per
   instance, 132 / 154 flows programmed before the first bridge instead of
   all 222, and the controller operational 39 / 28 ticks earlier.
3. **S10 switch.** `docs/LINK_STATE_SERVICE.md` §7, D3.9 and the engine's own
   comment name `MARL_LS_INSTALL=0` as the pre-registered "installation off"
   sensitivity; `confirmatory/chain_plan.py` runs S10 with
   `MARL_DETECT_TICKS=0`, the omniscient ablation (D-MRG.4 also calls it the
   S10 sensitivity). RUN_CHAIN.md R7 still says no installation switch
   exists; `MARL_LS_INSTALL` has existed since WP3.
   **Resolved (FIX-PREFREEZE item 1):** S10 sets `MARL_LS_INSTALL=0` and no
   stage sets `MARL_DETECT_TICKS`; `tests/test_run_chain.py::test_s10_runs_installation_off_not_the_omniscient_ablation`
   checks the plan of every profile, the `--emit` block and the environment
   file the launcher writes; the S10/S11 ablation gate still reads the S6
   baseline at the ablation's own tick count
   (`test_an_ablation_is_compared_with_the_baseline_at_its_own_tick_count`).
   §6 and §12 state the switch; D-FIX.1.
4. **Analyses the chain does not produce.** The draft §12 froze one
   invocation with `--untrained` and `--s8`; the chain's S7 passes neither,
   no stage runs the `PYTHONHASHSEED=1` pass, and S7-secondary needs the S9
   pickle that only exists after S7. S12's descriptive report is a pass
   count, and `untrained_analysis.py` is recorded NOT RUN (exit 90) because
   it is hard-wired to the first study. Resolution: §10 and checklist step f.
   **Resolved in code (FIX-PREFREEZE item 2):** the chain computes all
   three — stage S6h runs the `PYTHONHASHSEED=1` repeat, and S12 runs the S8
   check and one further analysis over S7's input plus the S9 pickle and the
   four S8 controls, failing on a missing input or a missing analysis
   (§10, D-FIX.2). Checklist step f is kept only as the record of the
   commands.
5. **Worker counts.** The draft §12 and `docs/COMPUTE_PLAN.md` (still the
   2026-09-22 skeleton, an 8-worker cap for all project compute) say
   `MARL_POOL_WORKERS=8`; the chain runs 24 per evaluation invocation and
   16 per training replicate with two alive. RUN_CHAIN.md cites a
   "COMPUTE_PLAN.md (19.4 h end to end)" that is not the file in this tree.
   Determinism makes the trained weights independent of the worker count
   (§3); evaluation worker-count invariance was measured at 4 against 8
   workers on all eight arms (D-DET.10, D-DET.15), not at 24.
   **Measured at 24 (FIX-PREFREEZE item 6, D-FIX.6):** one 600-tick
   all-arm invariance pass at `MARL_POOL_WORKERS=24` against one at 8 from
   the same head (`e785888`) and the same minted checkpoint: 2 132 of 2 132
   series identical, every arm, every key (energy and latency included).
   The worker counts above 8 remain the author's approval (§16 item 6).
6. **Environment hygiene.** The draft said the launcher clears every other
   `MARL_*` variable of the calling shell and that `marl_env` is asserted
   against §12; neither is in the code (`tools/run_invariance_pass.sh` clears
   them; `confirmatory/run_final.sh` does not; nothing reads `marl_env`).
   **Resolved (FIX-PREFREEZE item 5, D-FIX.5):** every job script starts
   from `env -i` and its env file; the preflight's job-environment check
   (before the launch) and the completeness gate (after it, on the pickle or
   checkpoint) require the plan's `MARL_*` set exactly (§12). Tests in
   `tests/test_run_chain_launcher.py`; negative controls
   `job_env_not_cleared`, `marl_env_not_compared`, `job_env_check_vacuous`.
7. **`marl_freeze` online reward (D8.11)** — open since WP8, unchanged: the
   online-learning reward of the S9 arm is not the frozen object's mixture.
   **Resolved (FIX-PREFREEZE item 3, D-FIX.3):** the online loop computes
   the team objective once per learning tick and hands
   `reward_spec.mix(local, team)` to `collect`;
   `tests/test_prefreeze_online_arm.py::test_the_adapt_then_freeze_arm_trains_on_the_frozen_reward`
   checks every collected reward of a real `marl_freeze` pass against
   0.4 · local + 0.6 · team; negative control `freeze_old_reward`. Only the
   online-learning arms change (`marl_freeze`, and `marl`, which no stage
   runs).
8. **`marl_freeze` silent collect (D6.41)** — open since WP6, unchanged:
   `except Exception: pass` around `mappo_trainer.collect` in
   `run_physics_pass`. **Resolved (FIX-PREFREEZE item 4, D-FIX.4):** the
   handler is gone;
   `tests/test_prefreeze_online_arm.py::test_a_refused_transition_in_the_online_arm_is_loud`
   makes `collect` raise in a real `marl_freeze` pass and requires the pass
   to end with that error at the first call; negative control
   `freeze_silent_collect`.
9. **Manifest timing and freeze 2.** The draft generated the block's manifest
   after the development gate and re-ran the seed audit "before the manifest
   is built" (also `docs/SEED_AUDIT_200_219.md` §5). The chain builds it at
   S0, before training, and runs S5 → S5b → S6 without a pause; RUN_CHAIN.md
   §5 item 4 asked for the author's approval of the earlier generation, which
   is not recorded, and its suggested remedy (`--stages S0` late) cannot work
   as written because S1 depends on S0. Resolution: the two-segment launch of
   §13 and the checklist.
   **Recorded (FIX-PREFREEZE item 9, D-FIX.9):** building the manifest at
   S0 is planning, not exposure (§5); this text and
   `docs/SEED_AUDIT_200_219.md` §5 and §6.11 now say the same.
10. **The seed-audit gate.** The draft required exit 0 / CLEAN at both
    freezes; after the rehearsal exposure that cannot be met (§5, §13).
    `docs/SEED_AUDIT_200_219.md` §5 item 1 still states the CLEAN rule; its
    new dated section §6 (2026-10-01) records the measured audits (exit 4
    twice) and the revised gate (§6.10).
11. **The WP13 verification record is missing.** `docs/DECISIONS.md` D13.1,
    D13.2, D13.3, D13.4 and D13.5 and `docs/RUN_CHAIN.md` R11 refer to
    "D13.6" for the negative-control results (`per_transition_lsa`,
    `global_action_lock`, `short_pass_removal`, `own_drop_constant`,
    `rehearsal_s7_reduced`, `ablation_mixed_ticks`), the isolation latency
    re-run at the final head, the invariance record of WP13, the
    production-length reward-term dump and the post-WP13 references; no D13.6
    exists in the file. Those results are therefore not in the committed
    record; the suite and invariance runs of checklist step b re-establish
    the parts the freeze depends on.
    **Resolved (FIX-PREFREEZE item 7):** D13.6 is written from the WP13
    verification logs (`logs/wp13/verify_c7d2322.log` and the files it
    names): the negative controls (6/6 at `c7d2322`; the two added at
    `0367856` first ran at the FIX-PREFREEZE head, D-FIX.11), the isolation
    latency (31–51 → 13–14 ticks), the invariance record (routing + SDN
    264/264 identical), the production-length reward-term dump
    (`removed_r_qos` constant 0.0 in Scenario B at 3 600 ticks) and the
    post-WP13 references, now also under `output/ref/inv_ref_wp13_c7d2322*`
    and named by `output/ref/README.md`.
12. **XL-DET A/82 (D4.19)** was measured before D13.1; the final head's
    behaviour on that instance is not recorded (§2 item 4).
    **Recorded (FIX-PREFREEZE item 8, D-FIX.8):** at the final head the
    instance settles at ONE fragment from t = 137 (§2 item 4), the same in
    the post-WP13 reference and in the FIX-PREFREEZE invariance passes.
13. **`docs/RUN_CHAIN.md` named the rehearsal's confirmatory seeds as the
    first four of the block** after `31b034c` had moved them; corrected in
    the same commit as this text (§1 table and R11; §6 keeps the
    2026-09-24 record as it happened).
14. **Smaller items of the draft, now stated from the code:** the training
    determinism decision (resolved by WP-DET: rollouts folded in episode
    order); the removed reward terms (§2 item 8); the sensitivity switch
    names; §4's "a checkpoint fingerprint change within a pass" (enforced by
    the harness, which raises); §4's invariance item (not automated, step
    b); the S8 decision series (`decision_digest` is recorded for every arm,
    so the "should the frozen tree not record the digest" branch does not
    apply).

Found when the seed audit was measured (2026-10-01), recorded here because
they bear on the freeze; none was changed in code by this revision:

15. **The launcher tests write into the production tree.**
    `tests/test_run_chain_launcher.py` runs `confirmatory/run_final.sh
    --profile test` in the tree with `tests/fake_stage.sh` as the stage
    script, which pickles a stand-in record to every artifact path of the
    test profile, including S0's manifest (a `.json` name) and an untrained
    checkpoint (a `.pt` name); they stay in `confirmatory/` and
    `output/test/`, and the seed audit refuses its verdict on them (§5,
    §13). `b43316a`, a tests-only commit made on 2026-10-01 by another
    session after this text was written, makes the tests remove them before
    each test and at exit; it is not covered by the rehearsal on `a33fdbc`
    (§16 item 8), and whether the suite at that head leaves none behind is
    shown only by checklist step b.1. *Since resolved:* `02049e3` contains
    `b43316a`, its rehearsal ended rc=0 (§16 item 9, D-FIX.13), and the
    suite at `02049e3` left no test-profile fixture in the tree (D-FIX.12).
16. **The rehearsal's dates and its S7 failure were misdescribed.** The
    `31b034c` commit message, the docstring of
    `tests/test_run_chain.py::test_only_the_real_profile_names_the_preregistered_confirmatory_seeds`
    and the earlier text of this file date the exposure "2026-09-24/25";
    every ledger event is on 2026-09-24. The earlier text also said S7
    failed because "the analysis cannot run on 600-tick series" and that
    "no primary statistic was computed"; the logs show a crash at a
    secondary test after the primary family was computed in memory, with
    nothing written or printed (§5). `docs/RUN_CHAIN.md` R11's "as it did on
    2026-09-24" conflated that crash with the merged tree's refusal and is
    corrected. The test docstring and the commit message are code and
    history and stay as they are.
17. **The driver's `--false-positive` pattern no longer matches.** The
    merged tree re-wraps the `sixg_sim/agent.py` docstring ("seed 42, 200 /
    island ticks"), so `tools/run_seed_audit.sh`'s pattern "seed 42, 200
    island ticks" misses it and the audit counts it as a text hit (in the
    final tree, `~/MARL_run_9c`, the `output/wp13_*` copies and the
    history). It is classified by hand in `docs/SEED_AUDIT_200_219.md` §6.6
    (d); changing the pattern is a code change and was not made.
18. **Planning statements outside the planning documents.** Lines that name
    the block in files the audit does not treat as planning are counted as
    text hits: `docs/RUN_CHAIN.md` (reworded locally on 2026-10-01; the
    history keeps the old line), the `output/wp13_*` copies of the tree,
    and `tests/test_run_chain_launcher.py` at `b43316a` (a comment naming
    the old stand-in file). §5 item 10 of the seed-audit record asks record
    documents not to do this.

Found by the review of this text and corrected by FIX-PREFREEZE item 10
(documents only):

19. **Three record errors.** (a) `docs/FREEZE_CHECKLIST.md` step c.3 printed
    `r.get('git_log')`, a key the audit report does not have (it is
    `git_logs`, a list), so the history-dump check printed `None` and could
    not show which dump was scanned; it now prints each dump's path,
    sha256, commit count and `scanned`, and says the freeze stops when no
    dump was scanned. (b) `docs/SEED_AUDIT_200_219.md` §6.6 omitted the two
    history lines naming `adapted_seed209.pt` (`docs/DECISIONS.md`,
    `e4bf26f`/`f03c08f`: a file name planted in a scratch fixture of the
    audit tool's own tests); they are class (c). (c) §6.1 of that record and
    §5 above said no pickle or JSON in `~/MARL_run_9` carries a window seed;
    true of result data only — the launcher-test stand-in
    `confirmatory/test_manifest_200_201.json` was a pickle under a `.json`
    name labelled with 200 and 201. Both texts now say so (record §6.12).
20. **The other routing baselines' control planes walked UEs.** Found while
    closing D12-FIXES (D12F.3 recorded it as not changed): OLSR,
    B.A.T.M.A.N. and AODV computed their fragments through UEs, and AODV
    established routes and billed RREPs on the UE-inclusive graph, while
    OSPF and (after D12.2) SDN used the leaf model. **Resolved by D12F.8**
    (the author's decision of 2026-10-03, "Fix before freeze"): one
    decomposition for every routing baseline, AODV's discovery on the path
    view; tests, 6/6 negative controls, invariance row `D12F.8` (every arm
    but those three bit-identical at 600 and 1200 ticks), the effect in §2
    item 9, new references `output/ref/inv_ref_d12f8_ad01ff0*.pkl`.

21. **XL-DET's interference brake latched the fleet low in Scenario B.**
    Found on 2026-10-04 in the first real-run attempt's development output
    (S1, seeds 50–59 and 70–79), before any confirmatory pass: XL-DET, the
    comparator of P1, P4, P7 and S9, delivered far less than every routing
    baseline in Scenario B. Bisected to WP5's postcard heartbeat (`992c955`),
    which made the brake permanent. **Resolved by XL-1** (§2 item 11; the
    author chose to fix the comparator and rerun the whole chain rather than
    amend P4 or disclose only).

## 16. Open decisions for the author before the launch

1. **D12.1 and D12.2.** Either (a) implement both as decided (a code change:
   new invariance references, a new rehearsal, a new freeze; the training
   environment changes too), or (b) run the frozen code and disclose, as §2
   items 9 and 10 then stated. **Decided by the author on 2026-10-01: "Fix
   both"** -- option (a). Both are implemented by the D12-FIXES package
   (`docs/DECISIONS.md` D12F.0 onwards; §15 items 1 and 2).
2. **S10.** *Resolved by FIX-PREFREEZE item 1:* the chain runs the
   pre-registered `MARL_LS_INSTALL=0` (§15 item 3). The code change is part
   of the head the new rehearsal and freeze cover.
3. **S7-secondary, S8, S12.** *Resolved by FIX-PREFREEZE item 2:* the chain
   computes them (S6h and S12, §10); the author may still shed S8/S9 in the
   plan's order, which S12 then records as not computed.
4. **Manifest at S0 and freeze 2.** The generation of the block's manifest
   before training is recorded as planning, not exposure (D-FIX.9, §5).
   Still the author's: the two-segment launch, or one segment with freeze 2
   recorded as waived.
5. **The block itself (§5).** Keep 200–219 with the disclosure (the
   recommendation), or reverse before the launch.
6. **Worker counts above 8** for the real run (24 / 16 × 2), which D0.9
   reserves for the user's explicit approval.
7. **The launcher-test stand-ins (§5).** Consent to archive the five files
   outside every scanned root (checklist step c.0; their paths, sizes and
   sha256 are in `docs/SEED_AUDIT_200_219.md` §6.5), one of them inside the
   record tree `~/MARL_run_9d`; and classify the two groups the audit
   reports for the author (the launcher-test logs as test fixtures, the
   `output/wp13_*` copies as planning; §5), or name them as exposures.
8. **`b43316a`.** `master` now differs from the rehearsed `a33fdbc` in
   `tests/test_run_chain_launcher.py` as well as in `docs/`. Either accept
   that a tests-only change is covered by the rehearsal (the chain never
   executes `tests/`; checklist step a.1 then compares against `a33fdbc`
   with `tests/test_run_chain_launcher.py` allowed), or re-run the rehearsal
   at the new head. Checklist step a.1 stops until this is decided.
9. **FIX-PREFREEZE changed code** (items 1-5, `1a486a7`..`02049e3`): the
   rehearsal on `a33fdbc` no longer covers the head, and item 8 above is
   superseded. Freeze 1 needs the rehearsal at `02049e3` (or a later code
   head) to end `CHAIN END profile=rehearsal rc=0`. **Satisfied:** it did,
   at 2026-10-02T14:29:24Z, 33 of 33 invocations OK (D-FIX.13), after
   waiting 15.6 h at S2 on the host's training memory gate (item 10).
   Checklist steps 0.1 and a.1 name that rehearsal and that head.
   **Superseded by D12-FIXES**, which changed code again (D12.1 and D12.2,
   item 1; code commits `62a85ae`..`aada41e`): the `02049e3` rehearsal no
   longer covers the head, and freeze 1 needs a new rehearsal at the D12
   code head to end `CHAIN END profile=rehearsal rc=0`. The rehearsal at
   `5205863` did (2026-10-03T03:37Z, 33/33 OK), but D12F.8 changed code
   once more (`c6e2aff`), so the rehearsal that covers freeze 1 is the one
   at the final head, run unattended before the launch by
   `~/MARL_watch/jobs/final_and_launch.sh`, which launches only if it ends
   rc=0 with every invocation OK (`docs/DECISIONS.md` D12F.9).
10. **The training memory gate.** *Resolved by the author (2026-10-02):
   the 26 GiB gate stays for every profile; host memory was freed
   instead.* The gate is 26 GiB, as the performance track's plan states it
   (the "GB" of every document is GiB, §12; D-FIX.14). The host offered
   24.5-25.5 GiB from 2026-10-01 17:29 to 2026-10-02 08:57 UTC (95 refused
   preflights), so the rehearsal of item 9 waited at S2 until host memory
   was freed: at 09:00:09 UTC an orphaned MARL pool worker leaked by an
   earlier job of this campaign (pid 2675988, 0.47 GiB RSS) was killed, and
   at 09:04:26 UTC, on the author's instruction, the author's own four NVIDIA
   AODT infrastructure containers (`aodt_efid-{temporal,clickhouse,nessie,minio}-1`,
   about 14 GiB, unrelated to this study) were stopped, not removed
   (`~/MARL_watch/DISK_LOG.md`); MemAvailable rose from about 25.5 to
   39.1 GiB and the next preflight admitted the wave at 09:07:33 UTC with
   the gate unchanged. (An earlier version of this item said "a tenant
   released memory"; that was wrong and is corrected here, in D-FIX.13,
   D-FIX.14 and `docs/COMPUTE_PLAN.md` §6.) Of the options put to the
   author -- (a) keep 26 GiB everywhere, (b) a rehearsal-only threshold,
   (c) decimal GB -- the author kept the gate unchanged for every profile
   and freed the host instead. The AODT containers stay stopped until the
   real run ends, and checklist step d.0 checks before every launch,
   counting what it checked, that MemAvailable is at least 26 GiB and that
   no `aodt_efid-*` container is running. The real run has the same gate,
   once per training wave, and a two-hour retry budget (checklist d.2).
