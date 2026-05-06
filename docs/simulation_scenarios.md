# Simulation scenarios: real-world mapping

This document describes how **timed scenario events** in the MARL / 6G network simulation relate to **real O-RAN + transport + 5GC** behaviour, and why each mechanism is modeled the way it is.

**Scope note:** The engine defines **13** distinct **event types** (see `sixg_sim/scenario.py`), including the geographic `geo_disaster` type for radius-based destruction. The repository ships **several named scenario configurations** (YAML files, embedded script, and the `DiverseScenarioGenerator`).

**Time base:** Each simulation **tick** advances by `tick_duration_ms` (commonly **100 ms** in `run_large_network_clean.py`). Multiply tick counts by this value for wall-clock interpretation.

---

## References (standards and architecture)

- **[3GPP TS 22.179](https://www.3gpp.org/DynaReport/22179.htm)** — Mission Critical Push To Talk (MCPTT); Stage 1 service requirements (emergency priority, group/private calls). The simulation’s `mcppt_*` events are inspired by this family of requirements.
- **[O-RAN A1 overview (O-RAN SC docs)](https://docs.o-ran-sc.org/projects/o-ran-sc-nonrtric-plt-a1policymanagementservice/en/latest/overview.html)** — A1 between Non-RT-RIC / policy entities and the near-RT-RIC; a failure here corresponds to losing **policy/intent** path into the RAN automation plane.
- **[ETSI / 3GPP TS 122 179 (MCPTT Stage 1)](https://www.etsi.org/deliver/etsi_ts/122100_122199/122179/16.05.00_60/ts_122179v160500p.pdf)** — Published TS text for MCPTT requirements (example frozen release).

---

## Core network definition (`sever_core`)

`Simulator._identify_core_nodes()` treats these **node types** as **core-side** for severance: `Core`, `SMO`, `Non-RT-RIC`, `AMF`, `UPF`.

> **Note (3GPP TS 23.501 §6.3.3):** `EdgeUPF` nodes are **not** classified as core and **survive severance**. They provide **Local Data Network (DN) traffic steering**, allowing UE-to-UE traffic to be routed locally without traversing the central core.

**Real-world meaning:** Core types are **central user plane / control plane / management** functions whose loss triggers island mode. Edge UPFs remain operational as local data anchors. **Near-RT-RIC** is *not* in `_identify_core_nodes` (it is treated as part of the "survivor" island discussion in `_detect_island_mode`). Operationally, you can still **remove** a near-RT-RIC with `energy_depletion` or `node_failure` to emulate loss of the **real-time RIC** at a site or cluster.

---

## The thirteen timed event types

Each row is one **string** `event_type` accepted by `ScenarioEvent` in `sixg_sim/scenario.py`. Parameters are what the **simulation engine** reads in `Simulator._execute_event` (`sixg_sim/simulation.py`).

| # | Event type | Parameters (typical) | What happens in the model | Real-world radio / transport analogy |
|---|------------|----------------------|---------------------------|--------------------------------------|
| 1 | `sever_core` | `{}` | All links incident to **core-class** nodes are set **down**; those nodes are **non-survivor**; agents on them removed. | **Backhaul partition** or **PoP failure**: gNB sites and edge nodes can no longer reach the **5GC**, **UPF**, or **management**; the network should enter **island / local breakout** behaviour. |
| 2 | `fail_link` | `link_id` | That logical link is **down**; caches and UE routing updated. | **Fiber cut, microwave fade, routing flap, failed optical transponder** on a specific **midhaul / backhaul / N2/N3** segment. |
| 3 | `restore_link` | `link_id` | Link set **up** again. | **Repair**, **alternative path** brought in, or **protection switch** completed. |
| 4 | `energy_depletion` | `node_id` | Node **SoC → 0**, marked **non-survivor**. | **Power loss** at site (grid + battery exhausted), **genset failure**, or **equipment thermal shutdown**. |
| 5 | `node_failure` | `node_id` | Node marked **non-survivor** (no explicit energy change in handler). | **Hardware fault**, **fire/flood** at shelter, **software crash** modeled as loss of that NF. |
| 6 | `node_recovery` | `node_id` | Node marked **survivor** again. | **Power restored**, **replacement unit** online, or **controlled restart** after fault clearance. |
| 7 | `traffic_surge` | `node_id`, `duration`, `multiplier` | **Declared** in scenarios; **surge scheduling** is implemented via `TrafficProfile.surge_events` in `traffic.py`. The `_execute_event` path in `simulation.py` does **not** currently append a surge for this event type—see *Implementation note* below. | **Flash crowd**, **sensor burst**, **video uplink spike** after an incident, or **rerouted traffic** onto one DU/CU after a link failure. |
| 8 | `ue_join` | `coverage_area`, optional `ue_id`, `connect_to_rus`, `is_rescue` | New UE attached; default traffic profile added; optional rescue flows. | **Subscriber enters coverage**, **IoT device powers on**, **first responder** enters the area. |
| 9 | `ue_leave` | `ue_id` | UE removed and profile deleted. | **Evacuation**, **device off**, **death of radio link** (walk out of coverage). |
| 10 | `rescue_force_arrival` | `num_ues`, `coverage_area` | Adds many **`Rescue_UE_*`** with **MCPTT-style** profiles and **rescue** flag. | **Task force** with **many radios** entering the disaster zone; **coordination load** on local RAN. |
| 11 | `mcppt_emergency_alert` | `ue_id` (or `emergency_type` in YAML), `emergency_type` | UE put in **emergency state**; **alert propagation** counted via `ue_to_ue_flows`. | **MCPTT emergency alert** (e.g. **imminent peril** vs **emergency**) per [TS 22.179](https://www.3gpp.org/DynaReport/22179.htm) family—**priority signalling** and **notification** to nearby users / services. |
| 12 | `mcppt_emergency_call` | `caller_ue`, `target_ue`, `emergency_type` | If both are **UE** nodes, **emergency call** counted and caller marked emergency. | **MCPTT private emergency call** (floor control / priority in real systems) between **two UEs**—e.g. **victim → rescue** or **responder ↔ dispatcher** if modeled as UE. |
| 13 | `geo_disaster` | `epicenter_x`, `epicenter_y`, `radius`, `destroyed_nodes`, `mode` | All nodes within geographic `radius` of epicenter are destroyed. Modes: `center_hit`, `zone_hit`, `corridor`. Core sever auto-triggered if core nodes are in blast. | **Earthquake**, **flood zone**, **military strike** — spatially correlated infrastructure loss using `x_pos`/`y_pos` from `topology_geo.yaml`. |

**Implementation note (`traffic_surge`):** YAML configs use `traffic_surge` with `node_id`, `duration`, `multiplier`. Surge scheduling is via `TrafficProfile.surge_events` in `traffic.py`.

---

## Geographic topology model

The simulation supports a **geographically realistic** deployment via `config/topology_geo.yaml` — a **6km × 6km urban area** with metric coordinates:

| Component | Count | Location | Distance from Core |
|-----------|-------|----------|----|
| **Core** (central DC) | 3 | Center (3000, 3000)m | 0m |
| **EdgeUPF** (edge aggregation) | 4 | N/S/E/W at ~2.2km | 2,200m |
| **gNB-Site** (collapsed O-RU + O-DU) | 20 | Clustered around EdgeUPFs | 700–900m from EdgeUPF |
| **Transport Relay** (Ceragon MW) | 40 | Between cells & edges | 400–1200m from gNBs |
| **UE** | 140 | Clustered around gNBs | 50–250m from gNB |

**Links**: 370 total (37 fiber ≤2.2km, 172 transport relay ≤1.2km, 161 microwave ≤1.7km). Capacities: fiber 10 Gbps, microwave 1 Gbps, transport relay 500 Mbps.

### O-RAN functional split analysis

The gNB-Site is a **collapsed cell site** containing both O-RU and O-DU functionality. We evaluated splitting into O-RU/O-DU/O-CU components:

- In real O-RAN: O-RU at tower (high survivability), O-DU at edge (medium), O-CU at DC (low)
- **Finding**: In all 3 tested geographic disaster scenarios, **zero orphaned O-RUs** — O-RUs and their O-DUs survive or die together due to geographic co-location
- **Decision**: Retain collapsed gNB-Site. The MARL's value is in **transport-layer recovery** (relay path optimization, cross-zone routing, energy management)

---

## Packaged scenario files (YAML)

### 1. `config/scenario_severance.yaml` — “Basic Island Mode Test”

| Setting | Value |
|--------|--------|
| **Duration** | 300 ticks |
| **Sequence** | `sever_core` @ 50 → `traffic_surge` @ 100 (gNB1, 30 ticks, ×2.5) → `fail_link` @ 150 (L2) → `energy_depletion` @ 200 (relay1) |

**Narrative:** A **small topology** loses core connectivity early, then sees **load** on one gNB, **relay** overload or **backhaul** loss, and finally **relay power** collapse—typical **cascading** degradation in **disaster + island** studies.

### 2. `config/scenario_large_network.yaml` — “Large Network Island Mode Test”

| Setting | Value |
|--------|--------|
| **Duration** | 500 ticks |
| **Sequence** | Core severance @ 100 → DU surge @ 150 → backhaul `fail_link` @ 200 → CU surge @ 250 → relay `energy_depletion` @ 350 |

**Narrative:** Stress **DU** and **CU** separately with surges, add **transport** failure, and **relay** energy loss—mirrors **multi-hop** and **CU/DU split** stress in a **large RAN**.

### 3. `config/scenario_mcppt_emergency.yaml` — “MCPTT Emergency Communication Test”

| Setting | Value |
|--------|--------|
| **Duration** | 100 ticks |
| **Sequence** | `sever_core` @ 30 → `rescue_force_arrival` @ 60 → **alerts** @ 70–75 → **emergency calls** @ 85–90 |

**Narrative:** After **core loss**, **rescue UEs** appear, then **MCPTT-style** **alerts** and **calls**—**public safety** workload on top of **partitioned** network.

---

## Embedded script: `run_large_network_clean.py` (large narrative)

The script builds a **long** event list (ticks up to 1100 in the source) but **truncates** to **500 ticks** for faster runs by **dropping** events after tick 500.

**Phases (author comments in code):**

1. **Phase 1 — “Fire” in control/management cluster:** `fail_link` on **A1** (SMO–near-RT-RIC policy path) and **N2** (CU–AMF), then **energy_depletion** on **SMO**, **near-RT-RIC**, **UPF**, **AMF**. **Real-world:** **Data centre / head-end** loss and **signalling** path breaks ([A1](https://docs.o-ran-sc.org/projects/o-ran-sc-nonrtric-plt-a1policymanagementservice/en/latest/overview.html), **N2** to **AMF**).

2. **Phase 2 — `sever_core`:** Formal **island** transition after **core-facing** entities are already damaged.

3. **Phase 2.5 — Rescue force arrival:** Batches of **Rescue_UE** in **zone_0 … zone_3** with **different `num_ues`**. **Real-world:** **Staging** of responders **per geographic coverage area**.

4. **Phase 2.6 — MCPTT:** **Alerts** (`UE_5`, `UE_10`, `UE_15`) and **calls** to **`Rescue_UE_*`** targets. **Real-world:** **MCPTT** emergency and **imminent peril** flows (see [TS 22.179](https://www.3gpp.org/DynaReport/22179.htm)).

5. **Phase 3 — Traffic surge:** On **first O-DU** (×3, 100 ticks). **Real-world:** **Localised** traffic **hotspot** on **distributed** RAN.

6. **Phase 3.5 — `ue_leave`:** **UE_1**, **UE_2** leave. **Real-world:** **Evacuation** or **device loss**.

7. **Phases after tick 500** (dropped in the shortened run): **extra rescue**, **N3** fail, **O-CU-CP** depletion, **restore_link**, **node_recovery**, **second surge** on **O-CU-UP** — **recovery** and **re-homing** story.

**Node selection constraints in code:** Links are chosen from topology by **ID patterns** (`A1_*`, `N2_*`, `N3_*`). If a pattern is missing, **string fallbacks** (e.g. `"A1_1"`) are used and may **not** match the graph—**fail_link** is then ignored (see log).

---

## Randomised training: `DiverseScenarioGenerator` (`sixg_sim/scenario.py`)

| Randomized aspect | Range / rule | Real-world intent |
|-------------------|--------------|-------------------|
| **Severance tick** | 80–125 (first 25% of 500-tick episode) | **Unpredictable** time of **backhaul** or **core** loss. |
| **Scenario type** | `full_core`, `zone_loss`, `cascading`, `multi_enb_iops`, `geo_disaster` | **Diverse** failure patterns for robust policy. |
| **`geo_disaster` mode** | `center_hit` / `zone_hit` / `corridor` | **Geographically correlated** destruction. |
| **`node_failure`** | 0–8 among relay/gNB nodes (cascading) | **Spatial** damage: some **cells** lost with the **core** event. |
| **`mcppt_emergency_alert`** | 10–40% of UEs, severity `emergency` / `imminent_peril` | **Many** subscribers in **distress** at once. |
| **UE traffic profiles** | Random baselines for UEs; **zero** infra baseline | **Heterogeneous** **edge** load for **training diversity**. |

---

## Traffic profiles and “restrictions”

Scenarios **bind** `traffic_profiles` per **node_id** with four classes: **life_safety**, **operations**, **telemetry**, **best_effort** (see `README.md` QoS table). **Higher** baseline rates on **UE** and **rescue** UEs **increase** offered load where **MCPTT** and **emergency** are modeled.

---

## Quick mapping: O-RAN interfaces used in the large script

| Interface / link | Role in story |
|------------------|----------------|
| **A1** | **Non-RT-RIC ↔ SMO / policy**; loss ⇒ **no new policies / ML** to near-RT-RIC ([O-RAN A1](https://docs.o-ran-sc.org/projects/o-ran-sc-nonrtric-plt-a1policymanagementservice/en/latest/overview.html)). |
| **N2** | **O-CU-CP ↔ AMF** (control); cut ⇒ **mobility / session** signalling to **core** impaired. |
| **N3** | **User plane** toward **UPF**; cut ⇒ **no central breakout** for that path. |
| **Open-FH / F1 / …** | Not always the first failures in this script; **DU/RU** stress appears via **surges** and **node** events. |

---

## Summary

- **Thirteen** **event types** are the **building blocks**; **YAML scenarios** and **scripts** **compose** them into **stories** (**island**, **transport**, **energy**, **MCPTT**, **rescue**, **geo_disaster**).
- The **geographic topology** (`topology_geo.yaml`) enables **radius-based disaster zones** with spatially correlated node destruction.
- The **collapsed gNB-Site** model is retained after O-RAN split analysis showed no benefit for geographically realistic disaster recovery.
- **Realism** is **structural** (graph, **core** definition, **QoS** classes, **geographic coordinates**) rather than **full 3GPP stack**.

---


*Generated from repository sources: `sixg_sim/scenario.py`, `sixg_sim/simulation.py`, `sixg_sim/topology.py`, `config/scenario_*.yaml`, `config/topology_geo.yaml`.*

