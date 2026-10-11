# Joint DPIR + MARL testbed: topologies

Four equipment levels for a physical testbed that demonstrates DPIR
(interference-aware multi-flow routing, Greidi et al., Allerton 2025) before
a catastrophic event and MARL recovery (beam re-steering across islands, this
repository's manuscript) after it.

| | IP-50 | TG | Sites | DPIR flows | Islands after the event | MARL bridges |
|---|---|---|---|---|---|---|
| A | 8 | 8 | 8 | 2 (+2 pocket service flows) | 4 | 3 |
| B | 6 | 6 | 7 | 2 (+1) | 4 | 3 |
| C | 4 | 6 | 7 | 3 (+1) | 5 | 4 |
| D | 4 | 4 | 5 | 2 | 3 | 2 |

The configurations are nested on one site plan: D is C without W2 and S, C is
B without the E1–E2 fixed link, B is A without the W1–W2 fixed link and the
field-hospital spur (N and its TG at CORE).

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
   across the gap. Every link that never steers (the core spokes and the
   links inside one side) is a fixed IP-50 PtP link.
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
6. **MARL bridges follow the paper's rules.** The 60 GHz budget is evaluated
   at the 23 dBm planning power (Table III model; the mesh reach does not
   follow the power the policy commands). Bridges form a highest-capacity
   tree over the islands, one bridge per steering head, and a
   terminal-class unit holds one link. Each island's head points back at
   the network it joins.

## Hardware notes to confirm with the vendor

- **Unit classes.** Node-class TG (N366/N367 class, 4 × 90° sectors) at the
  hub and every multi-peer or long-turn site; terminal-class (T265 class,
  ±45° scan around its mounting azimuth) only at the pockets, which link to
  node-class units only (no CN–CN links in Terragraph).
- **Polarity.** Every TG link must join an odd and an even sector, and with
  the core gone nothing can re-plan polarity after the event. The checked
  plan in `out/tables.md` sets it beforehand per sector (hybrid polarity),
  with the hub's west and east sectors on opposite polarities so that the
  W↔E bridges close. Confirm that the N366/N367 supports per-sector
  polarity.
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
  In the paper's model every pre-event TG hop has at least 16 dB of
  clear-air margin and every chosen bridge at least 13.9 dB. Replace these figures with the measured site
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

Requires `matplotlib` and `networkx`.
