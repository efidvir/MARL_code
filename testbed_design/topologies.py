"""Testbed topologies for a joint DPIR + MARL demonstration.

Hardware vocabulary
-------------------
IP-50   Ceragon IP-50 microwave PtP radio (one outdoor unit per link end,
        parabolic antenna, fixed alignment: it cannot be re-aimed after the
        event). A microwave link therefore always consumes two IP-50 units.
TG      Siklu MultiHaul TG 60 GHz (Terragraph) radio with electronic beam
        steering. Two classes are used:
          DN  node-class unit (N366/N367 class, 4 x 90 deg sectors = 360 deg),
              Terragraph distribution node, serves several peers (PtMP).
          CN  terminal-class unit (T265 class, one sector, steers within
              +/-45 deg of its mounting azimuth), Terragraph client node:
              one link at a time and it may only link to a DN.

Each configuration lists sites, the microwave links, the TG unit at each
site and its links, the DPIR flows, the sites destroyed by the event and the
recovery bridges the MARL arm forms (head -> peer).  Everything else
(unit counts, DPIR candidate paths, islands, bridge candidates, link
budgets, steering angles) is derived and checked in validate.py.
"""

# Site positions are schematic, in metres.  Distances only matter for the
# 60 GHz link budget, which uses the MARL paper's Table III model.
POS = {
    "CORE": (0, 300),
    "W2": (0, -300),
    "R1": (450, 560),
    "R2": (450, -560),
    "E1": (900, 300),
    "E2": (900, -300),
    "S": (450, -1010),
    "N": (450, 1010),
}

ROLE = {
    "CORE": "Core / gateway\nDPIR + TG controller",
    "W2": "West site",
    "R1": "MW hub",
    "R2": "TG relay",
    "E1": "Incident site E1",
    "E2": "Incident site E2",
    "S": "Shelter (responders)",
    "N": "Field hospital",
}

CONFIGS = {
    "A": dict(
        name="A - 8 IP-50 + 8 TG",
        ip50=8,
        tg=8,
        sites=["CORE", "W2", "R1", "R2", "E1", "E2", "S", "N"],
        mw=[("CORE", "R1"), ("W2", "R1"), ("R1", "E1"), ("R1", "E2")],
        tg_units={"CORE": "DN", "W2": "DN", "R1": "DN", "R2": "DN",
                  "E1": "CN", "E2": "DN", "S": "CN", "N": "CN"},
        tg_links=[("CORE", "W2"), ("W2", "R2"), ("R2", "E2"), ("R2", "S"),
                  ("E2", "E1"), ("R1", "N")],
        flows=[("F1", "CORE", "E1"), ("F2", "W2", "E2"),
               ("F3", "CORE", "S"), ("F4", "CORE", "N")],
        destroyed=["R1", "R2"],
        bridges=[("W2", "E2"), ("S", "W2"), ("N", "CORE")],
    ),
    "B": dict(
        name="B - 6 IP-50 + 6 TG",
        ip50=6,
        tg=6,
        sites=["CORE", "W2", "R1", "R2", "E1", "E2", "S"],
        mw=[("CORE", "R1"), ("R1", "E1"), ("R1", "E2")],
        tg_units={"CORE": "CN", "W2": "DN", "R2": "DN",
                  "E1": "CN", "E2": "DN", "S": "CN"},
        tg_links=[("CORE", "W2"), ("W2", "R2"), ("R2", "E2"), ("R2", "S"),
                  ("E2", "E1")],
        flows=[("F1", "CORE", "E1"), ("F2", "W2", "E2"), ("F3", "CORE", "S")],
        destroyed=["R1", "R2"],
        bridges=[("W2", "E2"), ("S", "W2")],
    ),
    "C": dict(
        name="C - 4 IP-50 + 6 TG",
        ip50=4,
        tg=6,
        sites=["CORE", "W2", "R1", "R2", "E1", "E2", "S"],
        mw=[("CORE", "R1"), ("R1", "E1")],
        tg_units={"CORE": "CN", "W2": "DN", "R2": "DN",
                  "E1": "CN", "E2": "DN", "S": "CN"},
        tg_links=[("CORE", "W2"), ("W2", "R2"), ("R2", "E2"), ("R2", "S"),
                  ("E2", "E1")],
        flows=[("F1", "CORE", "E1"), ("F2", "W2", "E2"), ("F3", "CORE", "S")],
        destroyed=["R1", "R2"],
        bridges=[("W2", "E2"), ("S", "W2")],
    ),
    "D": dict(
        name="D - 4 IP-50 + 4 TG",
        ip50=4,
        tg=4,
        sites=["CORE", "R1", "R2", "E1", "E2"],
        # D has a single west site; it sits on the west axis so both of its
        # hops into the zone are of similar length.
        pos_override={"CORE": (0, 0)},
        mw=[("CORE", "R1"), ("R1", "E1")],
        tg_units={"CORE": "DN", "R2": "DN", "E1": "DN", "E2": "DN"},
        tg_links=[("CORE", "R2"), ("R2", "E2"), ("E2", "E1")],
        flows=[("F1", "CORE", "E1"), ("F2", "CORE", "E2")],
        destroyed=["R1", "R2"],
        bridges=[("CORE", "E2")],
    ),
}

# The disaster zone is the band between the west and east sides.
ZONE_X = (255, 645)
