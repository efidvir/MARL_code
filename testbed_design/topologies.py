"""Testbed topologies for a joint DPIR + MARL demonstration.

Design rules
------------
* The core is severed by the event (MARL paper, Sec. III-B and Fig. 2): the
  core sits on the microwave aggregation hub inside the disaster zone.
* A TG radio is installed only where a beam may have to steer: the TG PtMP
  hub in the zone, and the surviving sites it serves, whose beams re-point
  across the gap after the hub is lost.  Links that never steer (core spokes,
  links inside one side of the network) are fixed IP-50 PtP links where the
  budget has them; otherwise (B, C, D) a link inside one side uses a spare
  sector of two TG nodes that are already installed for steering, so no
  radio is bought for it.
* Every link carries at least one DPIR candidate path (or is the only link
  of a single-homed pocket, S or N, whose traffic is a MARL service flow),
  and the event breaks every path, so only MARL bridges can rejoin it.
* DPIR flows are between dual-homed sites.  Their TG candidates contend for
  the airtime of the same hub sector even where they use different links,
  which is the interference coupling DPIR plans around.
* The four configurations are nested: B is A without the W1-W2 IP-50 link
  (a spare TG sector carries it) and the field-hospital spur, C is B without
  the E1-E2 IP-50 link (likewise), D is C without W2 and S.  One physical
  testbed covers all four.

Hardware vocabulary
-------------------
IP-50   Ceragon IP-50 microwave PtP radio, one outdoor unit per link end
        (two per link), parabolic dish, fixed alignment.
TG      Siklu MultiHaul TG 60 GHz (Terragraph) radio with beam steering.
          DN  node-class unit (N366/N367 class, 4 x 90 deg = 360 deg),
              Terragraph distribution node, PtMP.
          CN  terminal-class unit (T265 class, one sector, steers +/-45 deg
              around its mounting azimuth), Terragraph client node: one link
              at a time, and only to a DN.
"""

# Schematic positions in drawing units; SCALE converts them to metres.  The
# distances only matter for the 60 GHz link budget (MARL paper Table III
# model at the 23 dBm planning power): with SCALE = 0.6 every pre-event TG hop
# keeps >= 13 dB of clear-air margin above the QPSK-1/3 threshold and every
# chosen bridge >= 7.5 dB (option A's longest, W1-S, has 8 dB).  Replace with
# the measured site distances and the Siklu budget.
SCALE = 0.6
POS = {
    "W1": (0, 300),
    "W2": (0, -300),
    "CORE": (450, 560),
    "HUB": (450, 0),
    "E1": (900, 300),
    "E2": (900, -300),
    "S": (450, -860),
    "N": (450, 1010),
}

ROLE = {
    "W1": "West site",
    "W2": "West site",
    "CORE": "Core + MW hub\nsevered by the event",
    "HUB": "TG PtMP hub",
    "E1": "Incident site",
    "E2": "Incident site",
    "S": "Shelter (responders)",
    "N": "Field hospital",
}

_TG_STAR = [("HUB", "W1"), ("HUB", "W2"), ("HUB", "E1"), ("HUB", "E2"), ("HUB", "S")]

CONFIGS = {
    "A": dict(
        name="A - 8 IP-50 + 8 TG",
        ip50=8,
        tg=8,
        sites=["W1", "W2", "CORE", "HUB", "E1", "E2", "S", "N"],
        mw=[("W1", "CORE"), ("CORE", "E1"), ("W1", "W2"), ("E1", "E2")],
        tg_units={"HUB": "DN", "CORE": "DN", "W1": "DN", "W2": "DN",
                  "E1": "DN", "E2": "DN", "S": "CN", "N": "CN"},
        tg_links=_TG_STAR + [("CORE", "N")],
        flows=[("F1", "W1", "E1"), ("F2", "W2", "E2")],
        service=[("W1", "S"), ("W2", "N")],
        destroyed=["CORE", "HUB"],
    ),
    "B": dict(
        name="B - 6 IP-50 + 6 TG",
        ip50=6,
        tg=6,
        sites=["W1", "W2", "CORE", "HUB", "E1", "E2", "S"],
        mw=[("W1", "CORE"), ("CORE", "E1"), ("E1", "E2")],
        tg_units={"HUB": "DN", "W1": "DN", "W2": "DN", "E1": "DN", "E2": "DN", "S": "CN"},
        tg_links=_TG_STAR,
        tg_fixed=[("W1", "W2")],
        flows=[("F1", "W1", "E1"), ("F2", "W2", "E2")],
        service=[("W1", "S")],
        destroyed=["CORE", "HUB"],
    ),
    "C": dict(
        name="C - 4 IP-50 + 6 TG",
        ip50=4,
        tg=6,
        sites=["W1", "W2", "CORE", "HUB", "E1", "E2", "S"],
        mw=[("W1", "CORE"), ("CORE", "E1")],
        tg_units={"HUB": "DN", "W1": "DN", "W2": "DN", "E1": "DN", "E2": "DN", "S": "CN"},
        tg_links=_TG_STAR,
        tg_fixed=[("W1", "W2"), ("E1", "E2")],
        flows=[("F1", "W1", "E1"), ("F2", "W2", "E2")],
        service=[("W1", "S")],
        destroyed=["CORE", "HUB"],
    ),
    "D": dict(
        name="D - 4 IP-50 + 4 TG",
        ip50=4,
        tg=4,
        sites=["W1", "CORE", "HUB", "E1", "E2"],
        mw=[("W1", "CORE"), ("CORE", "E1")],
        tg_units={"HUB": "DN", "W1": "DN", "E1": "DN", "E2": "DN"},
        tg_links=[("HUB", "W1"), ("HUB", "E1"), ("HUB", "E2")],
        tg_fixed=[("E1", "E2")],
        flows=[("F1", "W1", "E1"), ("F2", "W1", "E2")],
        service=[],
        destroyed=["CORE", "HUB"],
    ),
}

# The disaster zone: the band between the west and east sides.
ZONE_X = (255, 645)
ZONE_Y = (-450, 800)
