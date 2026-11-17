# Joint DPIR + MARL testbed: topologies

Four equipment levels for a physical testbed that demonstrates DPIR
(interference-aware multi-flow routing, Greidi et al., Allerton 2025) before
a catastrophic event and MARL recovery (beam re-steering across islands, this
repository's manuscript) after it.

| | IP-50 | TG | Sites | DPIR flows | Islands after the event | MARL bridges |
|---|---|---|---|---|---|---|
| A | 8 | 8 | 8 | 2 (+2 pocket service flows) | 4 | 3 |
| B | 6 | 6 | 7 | 2 (+1) | 3 | 2 |
| C | 4 | 6 | 7 | 2 (+1) | 3 | 2 |
| D | 4 | 4 | 5 | 2 | 2 | 1 |

**Selected: option A.** Its bench plan (site carts, the attenuator setting of
every V-band path before, at and after the event, the microwave chains) is in
`out/option_A_plan.md`, from `option_a_plan.py`. Because the TG radios point
their own beams, the bench need not emulate steering towards every bridge
candidate: `option_a_sectors.py` cables only the final plan (nine V-band
paths) and writes `out/option_A_sectors.md`: the sectors each node uses before
and after the event (DN/CN, polarity), the 13 programmable attenuators that
take links down (with their settings), and the Ethernet port map of each site
switch. `out/iso_A_final.png` draws that bench.

**On the delivered kit: A7.** The lab has one 4-sector and six 2-sector
Terragraph nodes, all node-class (DN), in four 1U Siklu "Spider" conducted
test boxes: GTS003 (two nodes with two sectors each, twice), GTST002B (one
node with four sectors) and GTST003 (taken to be two 2-sector nodes). A7
(`topologies.py`) keeps option A's sites, IP-50 hops, flows, event, islands
and bridges; the core keeps only its microwave hub, the field hospital is fed
from the hub's otherwise idle fourth sector, and the shelter and hospital are
nodes instead of terminals. `validate.py` checks each unit's sector budget,
`out/option_A7_sectors.md` gives the bench plan, `kit_change.py` draws what
changed from A and `hub_ports.py` how the hub serves six links from four
ports. Each sector is a FATJ0498 module with four SMA ports, so on the bench a
sector has no physical direction: A7 puts each node's hub link on one module
and its bridge on the other, using all 16 modules. With at most two
node-to-node links per sector, a 2-sector node holds at most four links, so 9
of the 12 bridge candidates can be offered and MARL keeps 4 of its 8 recovery
plans. SMA ports cannot carry 60 GHz, so the attenuators and splitters must
suit the ports' actual frequency (still to be confirmed with the lab), not
the V-band parts sized in `option_a_sectors.py`.

The configurations are nested on one site plan. B is A without the W1–W2
IP-50 link (a spare sector of the W1/W2 TG nodes carries it instead) and
without the field-hospital spur. C is B without the E1–E2 IP-50 link
(likewise). D is C without W2 and S.

The hardware each option needs is in `out/hardware.md` (from `bom.py`), and
`out/iso_{A,B,C,D}.png` draw each bench as an exploded isometric technical
illustration with a numbered parts list (from `iso_render.py` and the
device drawings in `hwdraw.py`).

## The bench is fully conducted (galvanic)

No radio radiates. Each microwave hop runs cart to cart through a
waveguide–coax transition, a fixed pad, a programmable attenuator and
another pad. The TG radios are RF-port builds, cabled into a V-band RF mesh.
Every possible TG path has its own programmable attenuator, set to the loss
the geometry gives: the hub links, the spare-sector links and every feasible
MARL bridge. Splitters are fitted where one sector port reaches more than one
peer. At t0 the switched PDUs cut CORE and HUB and the paths to them go to
maximum attenuation. A MARL bridge is a cabled candidate path that the
engaged site brings up, so nothing is re-cabled during a run.

Figures: `out/overview.png` (all four), `out/config_{A,B,C,D}.png` (before the
event / event and islands / MARL recovery, DPIR candidates in the bottom
row); editable `.svg` next to each. Numbers: `out/tables.md`; every check:
`out/validation_report.txt`.

## Design rules

1. **The core is severed.** As in the MARL paper (Sec. III-B, Fig. 2), the
   core sits on the microwave aggregation hub (`CORE`) inside the disaster
   zone and is lost with it; so is the TG PtMP hub (`HUB`).
2. **TG only where a beam steers.** TG units are at the TG hub and at the
   surviving sites it serves; when the hub is lost, those beams re-point
   across the gap. Links that never steer (the core spokes and the links
   inside one side) are fixed IP-50 PtP links. Where the budget has no
   IP-50 for a link inside one side (B, C, D), a spare sector of the two TG
   nodes already installed carries it, so no radio is bought for it.
3. **Every link has a job.** Each link carries a DPIR candidate path, or is
   the only link of a single-homed pocket (S, N) whose traffic is a MARL
   service flow.
4. **DPIR has a real choice and real coupling.** Each DPIR flow has two
   diverse candidates (DPIR's diverse-path heuristic, at most two hops
   longer than the shortest): one through the core over IP-50, one through
   the TG hub. The hub's west sector serves W1 and W2, its east sector E1
   and E2, so two flows on different links still share sector airtime. That
   is the interference coupling DPIR plans around; the IP-50 links are
   treated as interference-free fixed-capacity pipes.
5. **The event breaks every path.** Losing CORE and HUB removes every
   candidate of every flow, so no routing protocol can cross; only new
   physical links can.
6. **MARL bridges follow the paper's rules.** The 60 GHz budget is
   evaluated at the 23 dBm planning power (Table III model; the mesh reach
   does not follow the power the policy commands). Whether a site relays is
   the policy's decision; the peer is the bridge-peer planner's. The
   figures show one valid plan: every island joined with no redundant
   bridge, and one new link per site (an engaged head holds one steerable
   link). It is the plan whose weakest bridge keeps at least 7.5 dB of
   margin, whose steering beams all visibly turn, and whose weakest link
   is strongest. `validate.py` counts all valid plans. The node-class end
   ignites each bridge (a Terragraph CN cannot).

## Hardware notes to confirm with the vendor

- **Unit classes.** Node-class TG (N366/N367 class, 4 × 90° sectors) at the
  hub and every multi-peer or long-turn site; terminal-class (T265 class,
  ±45° scan around its mounting azimuth) only at the pockets, which link to
  node-class units only (no CN–CN links in Terragraph).
- **Polarity.** Every TG link must join an odd and an even sector, and with
  the core gone nothing can re-plan polarity after the event. The checked
  plan in `out/tables.md` sets it beforehand. A W↔E bridge closes an odd
  loop through the hub, so some site must mix polarities across its sectors
  (a Terragraph "hardware hybrid"); Terragraph's planning guide wants one
  polarity per pole and mixing only between sectors more than 90° apart.
  `validate.py` therefore picks the plan with the fewest mixed sites and
  warns if a surviving site mixes adjacent sectors. In option A the only
  mixed site is the hub, which is destroyed at t0, so nothing changes at
  recovery. Terragraph also caps a DN sector at two DN-to-DN links (15 links
  in all); option A's busiest sectors carry exactly two.
- **Ignition without the core.** The E2E controller is lost with the core,
  so each steering site needs a local controller or agent that can ignite a
  pre-declared backup link (declared in the topology, auto-ignition off)
  and run beamforming training. Confirm that the Siklu firmware exposes this
  to an external agent.
- **Mapping the policy's action heads.** The paper's heads act on the FR1
  access layer, which this testbed does not have. Proposed mapping:
  - relay mode → enable / ignite the bridge link;
  - resource split → per-link airtime and QoS queues on the TG sector;
  - MCS offset → TG MCS cap.
  The bridge peer is chosen by the planner, not the policy (Sec. IV-D).
- **Forwarding plane for DPIR.** DPIR pins per-flow paths. Use a router at
  each site with per-flow policy routing (or VLAN / SR steering), and treat
  each TG link as one L2 hop rather than letting the mesh route across
  several TG hops.
- **Distances.** Positions are schematic (scale 0.6 m per drawing unit).
  In the paper's model every pre-event TG hop has at least 15 dB of
  clear-air margin and every chosen bridge at least 8 dB (13.9 dB except
  A's W1–S). Replace these figures with the measured site
  distances and the Siklu link budget.
- **After recovery** the network is a tree, so each flow has one route.
  DPIR only gets a choice again if MARL adds a second, capacity bridge.

## Files

| File | Contents |
|---|---|
| `topologies.py` | The specification: sites, IP-50 links, TG units and links, flows, destroyed sites |
| `validate.py` | Derives DPIR candidates, islands, bridge candidates, the recovery tree, beam turns and the polarity plan, and checks every claim (`python3 validate.py`, non-zero exit on failure) |
| `render.py` | Draws `out/config_*.png|svg` and `out/overview.png|svg` |
| `make_tables.py` | Writes `out/tables.md` |
| `bom.py` | Writes `out/hardware.md`, the bill of materials per option (conducted bench, V-band RF mesh sizing) |
| `hwdraw.py` | Isometric line-art drawings of the devices (carts, IP-50, TG units, attenuators, racks, cables) |
| `iso_render.py` | Writes `out/iso_*.png`, the exploded isometric bench per option, and `out/iso_A_final.png` (final plan only) |
| `animate.py` | Writes `out/anim_*.gif`: normal → broken → reconnected, with the TG beams turning slowly to their new peers |
| `option_a_plan.py` | Writes `out/option_A_plan.md`: the option A bench plan (carts, sector ports, V-band attenuator settings per state, microwave chains) |
| `kit_change.py` | Writes `out/kit_change.png`: option A beside A7 (the delivered N367 kit), each node with its sectors, and the bridges kept, still possible or out of reach |
| `figures/bench_a7.mjs` | Writes `out/bench_a7.html` (and `.svg`; `out/bench_a7.png` is its capture): the A7 bench as one isometric figure drawn with the Anatomy kit, a 19-inch rack with the switch, agent PCs, the four Spider boxes, the hub splitters, nine IF attenuators, DC drawer and PDU, front cables per link, and the IP-50 bench. `node figures/bench_a7.mjs`; the kit and its capture and line-check scripts are vendored in `vendor/anatomy/` (MIT) |
| `hub_ports.py` | Writes `out/hub_ports.png`: how the A7 hub serves six links from four RF ports (point-to-multipoint sectors, splitters on the bench) |
| `option_a_sectors.py` | Writes `out/option_A_sectors.md`: option A with only the final plan cabled (sectors per node, attenuators that take links down, site-switch port map) |

Requires `matplotlib` and `networkx`.
