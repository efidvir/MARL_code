# The bridge-peer planner — an engine service (RERUN_SPEC WP5)

Status: disclosure required by RERUN_SPEC WP5 ("the engine's bridge-peer
planner (component-aware, shared by all bridge-forming arms) is disclosed as
an engine service; pre-registered evaluation-only sensitivity: re-pointing
off"). Written 2026-09-28 with the WP5 change set; the decision record is
docs/DECISIONS.md D5.1–D5.12.

## 1. What it is

When a site is in `CAPACITY_BOOST` and owns a steerable MultiHaul radio, the
**engine** — not the site's policy — decides *which* peer its 60 GHz beam is
pointed at, and whether an existing beam is re-pointed. It does so with
knowledge of the **true infrastructure components** of the network, which no
site observes. That is a planning service with global knowledge, and the
paper must describe it as such.

It has three parts, all in `sixg_sim/simulation.py`:

| part | where | what it knows |
|---|---|---|
| candidate filter | `Simulator._step_phy_mac`, formation branch (`cross_fragment_candidates`) | the connected component of the site in the current infrastructure graph (`topology._build_infrastructure_graph()`): only peers **outside** it are candidates |
| ranking | `Simulator._rank_bridge_candidates` | components of the same graph, which components are already joined by an active or pending bridge, and how many other steerable sites can reach each target component (most-constrained-first, then capacity, then a per-site deterministic tie-break) |
| re-pointing | `Simulator._maybe_repoint_to_fragment`, called from `_step_phy_mac` | whether the site's current relay link is a de-duplicated **tree-edge bridge** (`distinct_bridge_link_ids`, a union-find over the relay-free partition); if not, and a feasible cross-component target exists, the non-bridge link is released (reason `repoint`) so the formation branch can bridge in the same tick; per-site cooldown `RELAY_REPOINT_COOLDOWN_TICKS` = 10, never inside the beam-training lockout |

The physical link budget (`TransportRelayModel.find_relay_candidates`: path
loss, 60 GHz interference at the receiver, the transport radio's planning
power) decides which peers are **feasible**; the planner only chooses among
them. The teardown protection of sole-path bridges (WP4,
`_relay_teardown_blocked`) reads the same component structures.

## 2. Who uses it

Every arm that forms relay bridges, identically: the learned executor
(`marl_static`, `marl_freeze`), the uniform-random control, XL-DET and SDN
(whose controller activates `CAPACITY_BOOST` through the same
`_step_phy_mac`). The routing baselines (OSPF, OLSR, B.A.T.M.A.N., AODV) form
no relay bridges. The planner is therefore **not** a policy input and **not**
a difference between arms; it is part of the environment all bridge-forming
arms act in. What each arm controls is *whether* a site relays
(`relay_mode`), its transmit power and its relay PRB share — all three move
the link budget and so the set of feasible peers — never the peer choice.

## 3. What the actor does NOT get

No component label, fragment label or bridge bit reaches the actor or the
DCC postcard. Block E (the "fragment cue") and the postcard fields
`fragment_id` / `holds_bridge` were removed by WP5: the labels were computed
by the simulator from its global link table with no flooding modelled,
delayed or billed — an oracle — and the cue could not fire in steady state
anyway, since a postcard only travels over a live link and so never comes
from a disjoint fragment. The removal is enforced by
`tests/test_wp5_postcards.py::test_no_block_e_reference_survives_anywhere`,
and `MARL_POSTCARD_FRAGMENTS=1` is refused at import.

## 4. The switch, and the pre-registered sensitivity

`MARL_RELAY_REPOINT` (read per simulator by
`simulation.relay_repoint_setting`):

| value | meaning |
|---|---|
| unset, or `fragment_aware` | re-pointing ON — the evaluated behaviour and the default since WP5 |
| `off` | **the pre-registered evaluation-only sensitivity**: no re-pointing (candidate filter and ranking unchanged) |
| anything else (including the former alias `legacy`) | refused (`ValueError`) |

Evaluation-only: the training driver refuses `off`
(`train_on_comparison._refuse_evaluation_only_switches`, in `main()` and in
every worker episode), so no policy is ever trained under the sensitivity.
The value in force is recorded in every result pickle through the provenance
`marl_env` (unset = the default). The chain runs it as stage S11
(`confirmatory/chain_plan.py` on branch trackRun: `env={'MARL_RELAY_REPOINT':
'off'}` on the selected checkpoint); the per-pass `repoints` counter shows
whether the baseline run re-pointed at all (NOT EXERCISED when it did not).

## 5. What the paper must say

* The peer a bridging site points at is chosen by an engine planning
  service with global component knowledge, shared by every bridge-forming
  arm; the policies decide whether and how strongly to relay, not where.
* The actor has no fragment-coordination input (Block E removed).
* The re-pointing-off sensitivity is reported descriptively.
* A genuinely distributed alternative (fragment labels flooded in postcards
  plus a cross-fragment 60 GHz discovery beacon, `rerun_map.json`
  postcard_label "Option A") is future work, with the structural argument
  that DCC postcards over live links cannot report a disjoint fragment.
