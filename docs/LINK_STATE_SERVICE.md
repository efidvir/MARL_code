# The cross-layer link-state service — model and constants

For PREREGISTRATION_R2.md. Binding source: `docs/RERUN_SPEC.md` §WP3.
Implementation: `sixg_sim/link_state_service.py`. Written 2026-09-24 with WP3.

Every arm on the MARL code path — `marl_static`, `marl_freeze`, `xldet`,
`random`, `untrained` — runs this service, in **training and in evaluation**,
created by `make_link_state_service` and stepped by `Simulator.step_tick`
(phase 6, after the agents act and before forwarding). The routing baselines
(`ospf`, `olsr`, `batman`, `aodv`) and `sdn` keep their own protocol models
and get no BFD.

---------------------------------------------------------------------------

## 1. Constants

All timers are in ticks, and 1 tick = 1 s (`TICK_DURATION_S`,
`TICKS_PER_SECOND = 1`). Every OSPF value is the one the **OSPF baseline
model** in `run_timeline_comparison.py` uses for its own arm, so the
cross-layer arms and the OSPF baseline are timed and billed by one set of
numbers. `tests/test_wp3_link_state_service.py::test_constants_match_the_ospf_baseline`
asserts the equality, so the two copies cannot drift.

| constant | value | meaning | source |
|---|---|---|---|
| `BFD_DETECT_TICKS` | 3 | detection / bring-up delay | RFC 5880 detect multiplier 3 at 1 packet/s |
| `BFD_BYTES` | 66 | one BFD control packet on the wire | 24 B control + 8 UDP + 20 IPv4 + 14 L2 |
| `BFD_PACKETS_PER_PAIR_PER_TICK` | 2 | one per direction | RFC 5880 §6.8.7 |
| `HELLO_INTERVAL` | 10 | OSPF HelloInterval | `OSPF_HELLO_INTERVAL` |
| `FLOOD_HOP_TICKS` | 1 | LSA flood pacing per hop | `OSPF_FLOOD_HOP_TICKS` |
| `SPF_DELAY_TICKS` | 5 | SPF throttle + FIB install | `OSPF_SPF_DELAY` |
| `MIN_LS_INTERVAL` | 5 | MinLSInterval, per originator | `OSPF_MIN_LS_INTERVAL` (RFC 2328 §12.4) |
| `HELLO_BASE` | 44 | Hello packet base size | `ControlPlaneMessageSim.OSPF_HELLO_BASE` |
| `HELLO_PER_NBR` | 4 | per listed neighbour | `OSPF_HELLO_PER_NBR` (RFC 2328 §A.3.2) |
| `LSA_BASE` | 24 | Router-LSA base size | `OSPF_LSA_BASE` |
| `LSA_PER_LINK` | 12 | per link record | `OSPF_LSA_PER_LINK` |
| `CPU_MJ_BFD`, `CPU_MJ_HELLO` | 0.5 | simple-packet CPU cost | the OSPF branch's own weight |
| `CPU_MJ_LSA` | 2.0 | LSDB update + SPF | the OSPF branch's own weight |
| `CPU_MJ_POSTCARD` | 0.3 | postcard parse + NN inference | the MARL branch's former weight |

Hello staggering uses the baseline's own phase, `crc32("<node>|<interval>")
mod interval` (`ControlPlaneMessageSim._get_node_offset`), never the builtin
`hash()`, which is salted per process.

## 2. Detection

Pair-level adjacency: a pair of endpoints is up **iff some link of the pair
is up**. This is the granularity `Topology.pair_representative` and
`sync_pair` already use, so belief and truth name the same hop.

| event | when the service believes it |
|---|---|
| an existing link fails | 3 ticks after the failure is observable |
| a new link's adjacency comes up | 3 ticks after its session starts |
| a link is torn down BY ONE OF ITS OWN ENDPOINTS | at once at that endpoint (administrative change); the remote endpoint still pays the 3 ticks, which shows in its Router-LSA origination time |
| a never-seen link id, **after island onset** | starts DOWN — there is no free adoption |
| a never-seen link id, before island onset / on the first step | adopted from truth (the pre-converged initial state, the same treatment every baseline controller gets from its `mark_pre_converged` path) |
| an ACCESS (Uu) adjacency | physical truth, with no delay — see §6 |

A transition that reverts inside the window is never confirmed: that is what
a detect multiplier is for.

## 3. Installation

**One Router-LSA per router, not per pair transition** (WP13, D13.1; RFC 2328
§12.4 and §12.4.1: a Router-LSA describes *all* of the router's links, and
MinLSInterval limits re-origination of that one LSA). Each **infrastructure**
endpoint `o` of a confirmed pair transition learns of it at `t_known[o]` and
keeps it *pending* until an LSA of its own carries it. `o` originates ONE
Router-LSA at

    t_o    = max(t_known of o's EARLIEST pending transition,
                 last_orig[o] + MIN_LS_INTERVAL)

carrying **every** transition it knows of by `t_o`; a transition it learns
after `t_o` waits for its next origination. A transition is installed
**network-wide** at

    t_inst = max_o( t_o + FLOOD_HOP_TICKS * ecc_o ) + SPF_DELAY_TICKS

over its originating endpoints, with `t_o` the origination of the LSA that
carries it. So k simultaneous transitions at one router install within one
origination, as the OSPF baseline model's "at most 1 LSA origination per
NODE per 5 ticks" announces them — before WP13 each transition scheduled its
own origination per endpoint, serialised at MinLSInterval, and the last of k
was announced 5(k−1) ticks after the first (on seed 82 with the severance at
tick 10, `O-CU-UP_15` had nine transitions queued, originated them at 13,
18, …, 53, and its core withdrawal installed at 61; D-MRG.13 item 2). The
bill follows the originations: one flood per origination.

`ecc_o` is the BFS eccentricity of `o` in the **installed** infrastructure
graph at origination time — between **routers**, i.e. the nodes that run the
routing protocol on the mesh (`_is_router`: `NON_ROUTER_TYPES` =
Core/UPF/AMF/CoreUPF/UE excluded, surviving nodes only). That is the same set
that decides who originates a Router-LSA and who is billed a hello: a core
node relays no flood, so it may not lengthen anyone's eccentricity (D3.17).
The *billed* flood scope (`_component_directed_adjacencies`) deliberately
stays on the broader non-access adjacency, because it reproduces the OSPF
baseline's own counting; the asymmetry is stated in D3.17, not hidden. `t_known[o]` is `td` for both endpoints of an
ordinary transition; for an administrative teardown it is the administrative
tick at the acting endpoint and that tick + 3 at the remote one. For a
single transition at a router with no pending one this is exactly the
formula in RERUN_SPEC §WP3; the per-router coalescing is the WP13 decision
D13.1, recorded in the pre-registration before any result.

A pair with no surviving infrastructure originator, and an access adjacency,
install where they are observed (`t_inst = td`). A transition that reverts
before `t_inst` cancels the pending installation.

**Forwarding and the flow gate use the INSTALLED view.** `believed_graph()`
returns `None` when the installed view equals truth.

## 4. Tombstones

A `Link` popped from `topology.links` keeps its `(endpoints, capacity)` in
`Simulator._link_tombstones` and counts as **physically down** until its
removal is installed. While the installed view still holds the hop, the
believed graph carries the tombstone's id and capacity and `consume_path`
delivers **exactly 0.0** over it: the believed graph never routes over a
physically-down link. The tombstone is dropped once the installed view no
longer names it, the service no longer believes it up and no detection is
pending on it.

**Which `Link` carries a hop.** Belief decides whether a pair is a hop *at
all* — no connectivity is adopted without the BFD session and the install —
but once it is one, the `Link` that carries it is the one **physics** names,
i.e. the same representative the engine's true graph uses. `consume_path`
reads `link.capacity` off whichever `Link` the hop names, so naming the
believed one during a Priority-2 relay's bring-up (the relay repurposes a
pair that already has a live physical link) would forward at the *pre-relay*
capacity over a hop that physically carries only the relay share. When
nothing of the pair is physically up there is no true representative, the
believed (physically-down) one stays, and the hop blackholes (D3.15).

The substitution may not **resurrect** a hop, so it applies only while some
link of the pair is **both believed up and physically up** (D3.22, narrowed
by D3.31). When nothing the service believes up is physically up,
substituting a live `Link` of the same pair would cancel the blackhole and
adopt a brand-new link id with no completed BFD session and no installed
Router-LSA. The predicate is the **pair**, not the relay-preferred
representative: a pair carrying two believed-up links whose representative is
the failed one is still carried by the parallel link, and refusing it would
be a hop refused on evidence the service does not have (D3.31; latent on the
evaluated topology, so it moves no series). The substitution is **measured**
never to raise the capacity a hop offers in the D3.15 case (`MORE_CAP = 0`
over 712,661 hop-instances), but that is a property of these topologies, not
of the relay-first/lowest-id ordering rule, which has no capacity term
(D3.32). For the same reason the
down-representative of a hop whose removal has not installed is chosen among
**physically-down** links only: a relay re-created on that pair before the
removal installs is live but neither believed nor installed, and naming it
would let it carry traffic at full capacity (D3.16). The two are the same
rule on the two sides of one branch — no connectivity is adopted for free.

**What `visible` and `installed_since` mean.** They answer "is this `Link` in
my forwarding view?", and a `Link` is in it in either of the two ways the arm
forwards over one (D3.23): the service believes it up inside an installed
hop, **or** the view names it as the `Link` carrying an installed hop — the
substituted carrier above, i.e. a Priority-2 relay from its creation tick,
before its own session completes. WP4 counts a bridge's teardown grace with
this stamp (`installed_since`, mirrored into `Simulator._relay_route_since`),
so a carrier without one would be read as a relay that was never installed
and never used.

A hop that **blackholes** is stamped iff the first of the two ways holds
(D3.30). That is exactly the undetected-failure case: the service still
believes its carrier up, so that `Link` is still in *its* forwarding view
while the flow delivers 0.0, and the stamp clears when BFD confirms the
failure (≤ `BFD_DETECT_TICKS` ticks later). The belief-side stamp may not
consult physical truth the service has not detected — doing so would be the
omniscience the detection floor removes. Only a hop with **nothing believed
up** — a tombstone, or the physically-down `_down_representative` — is
genuinely unstamped. **What WP4 must do with it:** `_relay_link_passes`
means "installed in the arm's forwarding view", which is how RERUN_SPEC WP4
words it, and *not* "delivered bits", so a relay that fails undetected
accrues passes while delivering 0.0. It is therefore booked with
`passes > 0` and is **not** an idle-bridge teardown, and its grace is
satisfied by those ticks. If WP4 wants a "carried bits" notion it must add a
separate counter; it may not read this stamp as one.

The engine mirror `Simulator._relay_route_since` follows `installed_since` in
**both** directions and is written only for ids the engine itself treats as
relay links (D3.33). It is a relay-scoped dict whose only other writer is
`Simulator._remove_relay_link` (which only pops), so filling it with mesh and
Uu access ids, or letting an entry survive the link's exit from the view,
would show WP4 a link that has not been installed for hundreds of ticks as
installed since tick 0. The invariant is `installed_since(lid) is None` ⟹
`lid not in sim._relay_route_since`.

Tombstone hops are counted apart from orphan hops (`tombstone_edges`,
`tombstone_volume`, `tombstone_hops` against `stale_edges`,
`stale_u2u_volume`, `orphan_hops`): a tombstone is expected, an unresolved
hop on the TRUE graph is the WP1 lifecycle defect and must stay impossible.

## 5. Billing — descriptive only

Billing runs **from island onset only**. Pre-event billing is exactly zero,
for this service and for every baseline protocol model.

| type | messages per tick | bytes per message |
|---|---|---|
| BFD | 2 per **infrastructure** adjacency pair running a session: confirmed up, or with a session in progress on one of its links (bring-up, or a failure inside its detect multiplier) — D3.24 | 66 |
| hello | 1 per surviving infrastructure router per `HELLO_INTERVAL`, at its crc32 offset — the **steady-state** interval throughout (D3.25) | 44 + 4 × infra-neighbour count |
| Router-LSA | on origination, 1 per **directed** believed-up infra adjacency in the originator's believed component | 24 + 12 × originator infra degree |
| postcard | the `DisasterControlChannel` ledger of the tick (WP5): 1 **frame** per live infrastructure link (`Link.is_up`, UE access links excluded) of every agent site sending its heartbeat on the tick — one heartbeat per site every `POSTCARD_PERIOD` = 10 ticks; a site with no live infrastructure link sends no frame | 60 (`POSTCARD_ENCODED_BYTES`, field layout in `sixg_sim/postcards.py`) |

**The BFD row departs from the binding specification.** RERUN_SPEC WP3 reads
"BFD: 2 × 66 B per **believed-up** infrastructure adjacency pair per tick",
and `rerun_map.json` / `scratchpad/rerun_details.txt` item 4b — the mapping
the project rules say to follow when a design choice is forced — uses the
same wording. This service bills the **union** of the believed-up (confirmed)
pairs with the pairs running a session, because a BFD session sends 1 packet/s
per direction from the moment it starts, which is precisely what makes the
3-tick bring-up cost what it costs (D3.24, D3.34). The consequence is not
cosmetic: the pre-registration is generated from RERUN_SPEC, and this moves
`overhead` (descriptive, S11) **and** `energy` (a comparative series, through
the control-plane CPU term, ~4e-6 relative) for every MARL-path arm. If the
pre-registration quotes that sentence it must carry the D3.24 qualifier.
RERUN_SPEC.md itself is binding and is not edited.

No LSAck and no periodic LSA refresh, which are the same omissions the OSPF
baseline model makes. Control-plane energy takes the weighted mean of the
per-message CPU costs above over the types actually billed on the tick.

Overhead for the MARL-path arms is now the **sum of these billed bytes**.
The phantom floor (13 agent-less survivors billed a 200 B frame every tick, a
constant 0.0208 Mb/s) and the change-detection proxy are gone.

**Reporting rule.** Overhead is a DESCRIPTIVE secondary (RERUN_SPEC §9, S11),
never a comparative KPI, and nothing in the code or the exported keys may
imply otherwise. Cross-family ordering is not supported: the baselines'
models undercount (OLSR TC re-forwarding unbilled, OSPF has no periodic LSA
refresh) and overcount (the OSPF hello and flood scope count UE neighbours)
in ways this service does not. Exported series: `overhead` (total Mb/s) and
the breakdown `oh_bfd`, `oh_hello`, `oh_lsa`, `oh_postcard`.

## 6. What is deliberately NOT modelled

- **Per-router LSDBs and per-source forwarding views.** Installation is
  global and conservative — the farthest originator sets the time — which
  OVER-charges flows near the change. A-full was rejected as out of scope.
- **Access (Uu) adjacencies are not in the link-state domain.** A pair with
  a UE endpoint follows physical truth with no detection and no installation
  delay: the anchor observes its own radio link directly (radio-link failure
  is a physical-layer indication, not a protocol), it is the only node whose
  forwarding depends on it, and the billing model is infrastructure-only
  throughout. See D3.3.
- **The baselines keep their own protocol models and get no BFD.** Their
  gates still block every flow at once while this service gates per flow —
  an asymmetry that favours the cross-layer arms in transients. Disclosed,
  not fixed (D3.6).
- **A per-type breakdown for the baselines.** `oh_bfd`, `oh_hello`, `oh_lsa`
  and `oh_postcard` decompose the MARL-path bill only. The baseline models
  bill their own message types (OLSR TC, B.A.T.M.A.N. OGM, AODV RREQ/RREP,
  OpenFlow/LLDP), which those four names do not describe, so their series are
  all-zero beside a non-zero total. Each pass therefore exports
  `oh_breakdown_available`, and a consumer must read it before summing or
  plotting the per-type series (D3.18).
- **A ≤3-tick BFD tail after an administrative teardown.** An
  administratively removed adjacency is confirmed down at once and carries no
  session afterwards, so it stops being billed at the administrative tick
  rather than 3 ticks later at the remote end, which still sends: an
  under-count of at most 3 × 132 B per teardown. The mirror-image omission on
  the **bring-up** side — a session billed nothing for its 3 ticks, the same
  3 × 132 B per new link — was a defect, not a modelling choice, and is fixed
  (D3.24): a pair with a session in progress is billed like a confirmed one.
- **An OSPF-style convergence state machine, and with it the fast-hello
  transient.** `_sim_ospf` bills hellos at `OSPF_HELLO_FAST_INTERVAL` = 1
  while its measured control plane reports `convergence_fraction < 1`, and
  phases each node's crc32 offset on whichever interval is in force. This
  service holds no convergence state and bills `HELLO_INTERVAL` = 10 at its
  steady-state phase throughout, so during a baseline transient the two hello
  models differ by 10× in rate and by the per-node phase — in the cross-layer
  arms' favour. RERUN_SPEC WP3's "as `_sim_ospf` (same interval and crc32
  offset)" is therefore true of the steady state only (D3.25).

## 7. Switches

| variable | default | meaning |
|---|---|---|
| `MARL_DETECT_TICKS` | 3 | 0 disables the service entirely (omniscient ablation: true graph, no flow gate, no billing). Any value > 0 sets the detection interval. |
| `MARL_LS_INSTALL` | 1 | **Pre-registered evaluation-only sensitivity.** 0 turns installation off (idealised timing): a confirmed transition takes effect at once. Detection and billing are unchanged, so it isolates the cost of the flood + SPF model alone. |

Neither is a tuning knob, and the evaluated system is the default of both.
