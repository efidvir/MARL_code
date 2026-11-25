"""Bench descriptions for PoC 1 (A7) and PoC 2, and everything derived from them: every cable
(with an ID, part, length and VLAN), the rack layout and the exploded parts list.  blueprint.py draws them.

A design is a plain dict:
  name, poc, drawing                       titles
  boxes      [dict(id, code, kind, nodes)] Spider boxes, top to bottom in the rack
  nodes      {site: dict(box, slot, sectors, pol)}
  module     {(site, sector): (box, 'M<k>')}       the sector module each used sector sits on
  link_ends  {link: ((site, sector), (site, sector))}   Terragraph links, module to module
  splitters  {(site, sector): [link, link]}       a module that feeds two links (PtMP, PoC 1 hub)
  couplings  {name: ((site, sector), (site, sector))}   cabled interference arms (PoC 2)
  mw_links   {hop: (site, site)}                  IP-50 hops
  sites      [site]                               every site, in VLAN order
  hosts      {host: [sites]}                      traffic hosts and the site VLANs they join
  agents     [site]                               MARL agent PCs (PoC 1)
  servers    [(id, role, site or None)]           site = the site VLAN its data NIC joins
  no_data    [site]                               nodes with no wired data port (PoC 1 hub)
  att_order  [link or arm]                        attenuator order (optional)
  att_set    {link: (job, before, t0, after, margin)}   attenuator settings in dB (optional); with if_pads, the
                                                  value each link's fixed pads stand in for
  att_reuse  int                                  attenuators carried over from PoC 1 (optional)
  if_pads    bool  (PoC 1) no programmable attenuators: every IF tile path gets a fixed pad stack (BW-S<n>W2+,
             at most PAD_MAX pads) on its A-end port (the hub splitter's P1/P2, else the A tile), sized from att_set
  mw_pads    bool  (PoC 1) no J7204B (MWATT): every IP-50 hop is radio - transition - 30 dB 10 W pad - 10 dB pad -
             one SMA cable - 30 dB 10 W pad - transition - radio
  event      text; event_off [device]   (PoC 1) devices that go down at t0 lose power: servers, TG nodes and
             IP-50 radios in event_off sit on switched PDU outlets (nodes through an AC PoE injector, radios
             through their own 48 V supply); links between surviving devices are dropped with attenuators, or
             (if_pads) held down with Terragraph ignition disabled until MARL engages them: no RF change

Management is in-band: the E2E controller reaches every node through the radio network from its data
NIC, so no node ETH2 (and no IP-50 management port, until it is shown not to bridge to the data port) is
cabled to the management switch.  SW-OOB carries only lab equipment: attenuators (if any), PDUs, the analyser
and the PCs' second NICs.
"""
import csv
import math
import os
import re
import sys
from collections import Counter, OrderedDict

from validate import bearing, sector

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin"))
import catalog as CAT  # noqa: E402  (dimensions and mounting hardware, shared with the twin)

LADDER = {}            # set by twin/twin_model.py (standard cable lengths); empty when the twin is not used

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEC = {0: "E", 1: "N", 2: "W", 3: "S"}
BOX_KIND = {"GTS003": (2, 2), "GTST002B": (1, 4), "GTST003": (2, 2)}
# cable ID letter per class; PoE is 'Q' so its IDs (Q001...) cannot be read as Tripp Lite part numbers (P004-006 ...)
CLS_PREFIX = {"AC": "A", "DC": "D", "POE": "Q", "FIB": "F", "ETH": "E", "RF": "R", "GNSS": "G", "WG": "W", "AUX": "X",
              "PE": "B"}            # PE = earth bonding (B001...: 'bond')
WALL_SOCKET = "lab wall socket IEC 60309 16 A"
STRIP_OUT = 6                       # TABLE-STRIP: 6 x C13 outlets; the ones without a load stay empty
# TENTATIVE equipment choices (user, 2026-11-24): 1U switched PDUs; programmable attenuators are small bench boxes
# with a dial (4 channels each, two per 2U shelf; PoC 2 only: PoC 1 has none since 2026-11-25, fixed pads); servers and traffic hosts are 1U/2U rack servers; the bench
# controller is a PC with a monitor on the table; the IP-50 radios are Ceragon IP-50CX (band TBC).
PDU_OUT, PDU_U, PDU_PART = 12, 1, "Raritan PX3-5260R class (1U, 12 x C13, switched, 16 A)"
ATT_PER_SHELF, ATT_SHELF_U = 2, 2
ATT_BOX = (200.0, 70.0, 160.0)                       # w, h, d mm (estimate: 'small box with a dial')
SERVER_U = {"SRV-DPIR": 2, "SRV-E2E": 1}            # others 1U
TABLE_SERVERS = ("BENCH-CTL",)                       # on the bench table, with the monitor
RECT_MODULES = {"DC-RECT": 2, "DC-RECT2": 1}         # RCP-1000-48 modules per RCP-1UI shelf: one C14 inlet (and cord) each
# physical Spider boxes by their lab label: the same box keeps its label and serial in every PoC (SB1 = GTS003 #1 ...)
BOX_SERIAL = {"GTS003 #1": "SF06000007 (assumed)", "GTST002B": "SF06000001", "GTST003": "SF06000058",
              "GTS003 #2": "SF06000031 (assumed)"}
POLARITY_NOTE = ("odd / even / hybrid is the Terragraph polarity the E2E controller gives each node (a software "
                 "setting, not a label on the box): the same box can be odd in one PoC and even in the next")
# coupling-arm pad: chosen so that the arm attenuator at 0 dB gives the TOP of the INR sweep (TENTATIVE budget, see
# Bench.size_arm_pads); Mini-Circuits BW-S<n>W2+ values that exist
ARM_BUDGET = dict(span_db=46.0, snr_db=25.0, inr_max_db=24.0, coupling_db=10.0, dc_block_db=0.5, att_il_db=5.0,
                  cable_db_per_m=1.5, conn_db=0.5)
BW_S_DB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]
# fixed-pad links (design 'if_pads', PoC 1, TENTATIVE: user, 2026-11-25): at most PAD_MAX BW-S pads per tile path,
# spares per value; IP-50 hop chain (design 'mw_pads'): a 30 dB 10 W pad on each transition + one 10 dB pad on the A side
PAD_MAX = 2
PAD_SPARE = 0.10                # ~10 % spare pads per value, at least 1
MW_PAD_HI = ("SA18S10W-30", 30)                     # on each transition (10 W: the radio's full output)
MW_PAD_LO = ("BW-S10W2+", 10)                       # A side, after the 30 dB pad
MW_PAD_LO_K = "2.92 mm 10 dB fixed attenuator DC-40 GHz (part TBC)"   # replaces MW_PAD_LO if the band is 23 GHz
# port check (fixed pads, TENTATIVE): expected link figures from the A7 model (option_A7_sectors.md 'Margin up' = clear-air
# margin above the lowest data MCS threshold) and Terragraph's link-adaptation SNR thresholds (copied from poc2.MCS:
# QTI firmware 10.11 defaults, laMinMcs 2, MCS5 disabled); a link outside +/- PORT_CHECK_TOL_DB of its expected SNR gets
# a trim pad (TRIM_PADS, at the B tile) or a smaller outer pad
TG_MCS_SNR = [(4.5, 2), (5.5, 3), (6.5, 4), (7.5, 6), (8.5, 7), (9.5, 8), (10.5, 9), (13.5, 10), (14.5, 11), (16.5, 12)]
PORT_CHECK_TOL_DB = 3.0
TRIM_PADS = (3, 5)              # 4 of each (one link's four tile paths): 3, 5 or 3 + 5 = 8 dB at the B tile
BELDEN_SPOOL_M = 30.48          # Belden 8461, 100 ft spool
DC_WASTE = 0.10                 # stripping, dressing and offcuts on bulk DC cable
DC_CUT = ("Belden 8461", "bus link", "bus strap", "PE 6 mm2", "PE 2.5 mm2")   # cut from bulk to the twin route + 10 %
BREAKER_A = 5                   # DCD1M breaker rating for the 18 AWG adaptor feeds (5 A or less)
SPACER_MM = 25.0                # spacer bars between stacked IP-50 radios (convection gap)
LACE_EVERY_U = 8                # one rear lacing bar per 8 U used
STRAP_EVERY_MM, STRAP_SHARE = 300, 4     # one hook-and-loop strap per 300 mm of bundle, ~4 cables share one
LABELS_PER_CART = 250           # TENTATIVE: labels per M21-750-427 cartridge
BOM_SKIP = ("flange",)          # schedule rows that are not ordered parts
CABLE_USE = {"POE": "PoE to node ETH1", "ETH": "management", "FIB": "data"}
RACK_KINDS = {"SW-DATA": "sw_data", "SW-OOB": "sw_oob", "GNSS-SPL": "gnss", "CM": "cm", "AGENTS": "shelf",
              "SPL-TRAY": "tray", "CPL-TRAY": "tray", "PA-SHELF": "shelf", "DC-DIST": "dcdist", "DC-DIST2": "fuse",
              "DC-RECT": "rect", "DC-RECT2": "rect"}


def rack_kind(rid):
    """catalog kind of a rack row."""
    return ("spider" if rid.startswith("SB") else "attshelf" if rid.startswith("ATT-SHELF") else
            "rackserver" if rid.startswith(("SRV-", "HOST-")) else "pdu" if rid.startswith("PDU-") else
            "blank" if rid.startswith("BLANK") else RACK_KINDS[rid])

U_MM = 44.45                    # one rack unit
BEND_MM = 100                   # plug bodies and a bend at each end of an IF cable
RF_STOCK = [("CBL-1FT-SMSM+", 305), ("CBL-2FT-SMSM+", 610), ("CBL-3FT-SMSM+", 914), ("CBL-1M-SMSM+", 1000),
            ("CBL-4FT-SMSM+", 1219), ("CBL-5FT-SMSM+", 1524), ("CBL-6FT-SMSM+", 1829), ("CBL-2M-SMSM+", 2000)]
RF_LIKELY = ("CBL-4FT-SMSM+", "CBL-5FT-SMSM+", "CBL-6FT-SMSM+")   # catalogue lengths to confirm with Mini-Circuits
DATA_PORTS, DATA_1G = 28, 16    # M4350-24F4V: P1-24 SFP+ (1G only on P1-16), P25-28 SFP28 (10/25G)
OOB_RJ45, OOB_SFP = 26, 4       # GSM4230PX: 24 PoE+ + 2 RJ-45, 4 SFP+ cages (AGM734 copper SFP if needed)

AGM734 = "Netgear AGM734 1000BASE-T SFP"
OM4 = "SFP+ SR at both ends (Siklu AX-SFP+10G-MM / Netgear AXM761) + Panduit FZ2ERLNLNSNM002 OM4 LC-LC 2 m"
DAC = "Netgear AXC763-10000S SFP+ DAC 3 m"


def bus_protection(rect):
    """what protects a rectifier-to-distribution bus link (2 x 10 AWG): only the rectifier's own current limit."""
    n = RECT_MODULES.get(rect, 1)
    amps = n * CAT.RCP_MODULE_A
    return (f"bus link protected by the rectifier current limit only ({n} x RCP-1000-48, about {amps:.0f} A into 2 x 10 AWG"
            + (": confirm the RCP shuts down on sustained overload, or fit a 30 A DC fuse at each rectifier output)"
               if amps > 30 else ", within the 10 AWG rating)"))


def mod_x(m, t):
    """Assumed x (mm from the left rail) of tile port t of module M<k> on a Spider box front."""
    return 70 + (int(m[1]) - 1) * 95 + (t - 2.5) * 18


def att_x0(slot):
    """x (mm) of the left edge of an attenuator box in slot 0/1 of its shelf."""
    return 25 + slot * 235


def ch_x(c, side, slot=0):
    """Assumed x (mm) of an attenuator channel's IN / OUT port on the box front."""
    return att_x0(slot) + 28 + (c - 1) * 42 + (-9 if side == "in" else 9)


def _pad_combos(values=BW_S_DB, n=PAD_MAX):
    """{total dB: pads} for every total that 1..n available BW-S values make: the fewest pads first, then the largest
    first pad (28 -> 20 + 8, 35 -> 30 + 5, not 20 + 15)."""
    vs = sorted(values, reverse=True)
    out = OrderedDict((v, (v,)) for v in vs)
    if n >= 2:
        for a in vs:
            for b_ in vs:
                if b_ <= a:
                    out.setdefault(a + b_, (a, b_))
    return out


PAD_COMBOS = _pad_combos()


def pad_target(row):
    """the attenuation (dB) a link's fixed pads stand in for, from its A7 attenuator row (job, before, t0, after, margin):
    the 'after' setting when it has one (a bridge, as MARL engages it), else 'before' (a hub path: it only goes down with
    the hub, so its attenuator only ever set the link quality)."""
    _, bef, _, aft, _ = row
    return float(aft) if isinstance(aft, (int, float)) else float(bef)


def pad_stack(target):
    """Fixed pad stack for a target attenuation: the target rounded to the nearest whole dB (a tie rounds UP, toward more
    loss, so a path is never stronger than the model), split into at most PAD_MAX available BW-S values. A whole-dB value
    that no stack makes takes the achievable total nearest the unrounded target (a tie again toward more loss)."""
    r = int(math.floor(target + 0.5))
    if r in PAD_COMBOS:
        db, rule = r, "nearest whole dB" + (", tie rounded up" if abs(target - math.floor(target) - 0.5) < 1e-9 else "")
    else:
        db = min(PAD_COMBOS, key=lambda v: (abs(v - target), -v))
        if min(PAD_COMBOS) < r < max(PAD_COMBOS):
            rule = f"{r} dB needs more than {PAD_MAX} pads: nearest total {PAD_MAX} pads make"
        else:                       # under the smallest pad or over the largest stack: no stack of PAD_MAX pads makes it
            rule = (f"{r} dB is not achievable with at most {PAD_MAX} pads ({min(PAD_COMBOS)}-{max(PAD_COMBOS)} dB): "
                    "nearest achievable total")
    pads = list(PAD_COMBOS[db])
    return dict(target=round(target, 1), rounded=r, db=db, pads=pads, pn=[f"BW-S{v}W2+" for v in pads],
                text=" + ".join(str(v) for v in pads), rule=rule)


def link_expect(row, db):
    """the port-check figures of a fixed-pad IF link from its A7 row (job, before, t0, after, margin 'up / at max'): the
    model margin above the lowest data MCS at the A7 setting, the same at the fitted pads (db), the expected SNR (lowest
    data-MCS threshold + margin) and MCS, and the acceptance band (+/- PORT_CHECK_TOL_DB)."""
    m = re.match(r"^\s*(-?[\d.]+)", str(row[4]))
    if not m:
        return {}
    up = float(m.group(1))
    margin = up - (db - pad_target(row))
    snr = TG_MCS_SNR[0][0] + margin
    mcs = max((k for thr, k in TG_MCS_SNR if snr >= thr), default=None)
    return dict(margin_model_db=round(up, 1), margin_db=round(margin, 1), snr_exp_db=round(snr, 1), mcs_exp=mcs,
                snr_min_db=round(snr - PORT_CHECK_TOL_DB, 1), snr_max_db=round(snr + PORT_CHECK_TOL_DB, 1))


def pad_parts(pns):
    """[dict(pn, db, len_mm, dia_mm)] for a list of pad part numbers, in stack order (port end first)."""
    out = []
    for pn in pns:
        L, dia = CAT.pad_dims(pn)
        m = re.match(r"^(?:BW-S(\d+)W2\+|SA18S10W-(\d+))$", pn)
        out.append(dict(pn=pn, db=int(m.group(1) or m.group(2)), len_mm=L, dia_mm=dia))
    return out


def bridge_ops(d):
    """the MARL bridges of a fixed-pad bench (if_pads: att_set job 'bridge'): one row per bridge with its Terragraph link
    name (link-<A node>-<B node>, TBC with the lab topology), A-end node and sector, B-end sector and the A-end agent PC
    (MARL engages the bridge from it at node level: SRV-E2E is off after t0)."""
    out = OrderedDict()
    if not d.get("if_pads"):
        return out
    for name, row in d.get("att_set", {}).items():
        if row[0] != "bridge" or name not in d["link_ends"]:
            continue
        (an, asec), (bn, bsec) = d["link_ends"][name]
        out[name] = dict(link=name, tg=f"link-{an}-{bn}", a_node=an, a_sector=f"{an}.{asec}", b_node=bn,
                         b_sector=f"{bn}.{bsec}", agent=f"AGENT-{an}" if an in d.get("agents", []) else None)
    return out


TG_NAMES_TBC = "TG link names TBC with the lab topology"


def bridge_hold_text(br):
    """'disable ignition ...' for the bridges of bridge_ops (the hold before t0, on the E2E controller)."""
    tg = [v["tg"] for v in br.values()]
    names = list(br)
    return (f"on the E2E controller (SRV-E2E) disable ignition for the bridge links "
            f"{', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else names[0]} "
            f"(tg ignition link -n {tg[0]} --disable" + (f", then the same for {' and '.join(tg[1:])}" if tg[1:] else "") +
            f"; {TG_NAMES_TBC}) and read the setting back")


def bridge_engage_text(br):
    """how MARL engages a bridge after t0, with no E2E controller (SRV-E2E is switched off at t0)."""
    return ("MARL engages a bridge at node level, with no E2E controller (SRV-E2E is off after t0): the MARL agent of the "
            "bridge's A-end node sends the Terragraph minion command tg2 minion assoc to its node (initiator = the A-end "
            "sector, responder = the B-end sector) from its agent PC, in-band over the site VLAN on SW-DATA (the nodes "
            "have no SW-OOB port): " +
            "; ".join(f"{k} from {v['agent'] or 'its A-end agent'} ({v['a_sector']} to {v['b_sector']})" for k, v in br.items())
            + ". TBC: that the N367 firmware accepts a minion-level association with no controller running (open "
            "question); fallback: a standby E2E controller on BENCH-CTL (never switched), or SRV-E2E on an unswitched "
            "outlet")


# ------------------------------------------------------------------------------------------
# designs
# ------------------------------------------------------------------------------------------
def poc1_design():
    """A7 as cabled on the bench (figures/bench_a7.mjs): each node's hub link on one module and its
    bridge on the other, the hub's W and E modules split to two peers. TENTATIVE (user, 2026-11-25): links only need to
    go down, and that happens because whole devices lose power at t0, so there are no programmable attenuators and no
    J7204B: every IF tile path gets a fixed pad stack sized from the A7 settings below (if_pads), every IP-50 hop fixed
    pads (mw_pads); the bridges are held down with Terragraph ignition disabled until MARL engages them."""
    boxes = [dict(id="SB1", code="GTS003 #1", kind="GTS003", nodes=["W1", "W2"]),
             dict(id="SB2", code="GTST002B", kind="GTST002B", nodes=["HUB"]),
             dict(id="SB3", code="GTST003", kind="GTST003", nodes=["S", "N"]),
             dict(id="SB4", code="GTS003 #2", kind="GTS003", nodes=["E1", "E2"])]
    nodes = {"W1": ("SB1", "a", 2, "odd"), "W2": ("SB1", "b", 2, "odd"),
             "HUB": ("SB2", "a", 4, "hybrid (E, S odd; N, W even)"),
             "S": ("SB3", "a", 2, "even"), "N": ("SB3", "b", 2, "odd"),
             "E1": ("SB4", "a", 2, "even"), "E2": ("SB4", "b", 2, "even")}
    module = {("W1", "E"): ("SB1", "M1"), ("W1", "S"): ("SB1", "M2"), ("W2", "E"): ("SB1", "M3"),
              ("W2", "S"): ("SB1", "M4"),
              ("HUB", "W"): ("SB2", "M1"), ("HUB", "E"): ("SB2", "M2"), ("HUB", "S"): ("SB2", "M3"),
              ("HUB", "N"): ("SB2", "M4"),
              ("S", "N"): ("SB3", "M1"), ("S", "E"): ("SB3", "M2"), ("N", "S"): ("SB3", "M3"), ("N", "W"): ("SB3", "M4"),
              ("E1", "W"): ("SB4", "M1"), ("E1", "N"): ("SB4", "M2"), ("E2", "W"): ("SB4", "M3"),
              ("E2", "N"): ("SB4", "M4")}
    link_ends = OrderedDict([
        ("HUB-W1", (("HUB", "W"), ("W1", "E"))), ("HUB-W2", (("HUB", "W"), ("W2", "E"))),
        ("HUB-E1", (("HUB", "E"), ("E1", "W"))), ("HUB-E2", (("HUB", "E"), ("E2", "W"))),
        ("HUB-S", (("HUB", "S"), ("S", "N"))), ("HUB-N", (("HUB", "N"), ("N", "S"))),
        ("E2-W2", (("E2", "N"), ("W2", "S"))), ("E1-N", (("E1", "N"), ("N", "W"))), ("W1-S", (("W1", "S"), ("S", "E")))])
    # out/option_A7_sectors.md, 'V-band paths and programmable attenuators': job, before, at t0, after, margin
    # the hub's paths go down with the hub (power cut at t0): their attenuators only set the link quality before.
    # With fixed pads (if_pads) each link's pad stands in for: hub paths 'before', bridges 'after' (bench_spec.pad_target)
    down = "hub off at t0"
    att_set = OrderedDict([
        ("HUB-W1", (down, 27.6, "hold", "hold", "21.6 / -13.9")), ("HUB-W2", (down, 23.1, "hold", "hold", "21.6 / -18.4")),
        ("HUB-E1", (down, 27.6, "hold", "hold", "21.6 / -13.9")), ("HUB-E2", (down, 23.1, "hold", "hold", "21.6 / -18.4")),
        ("HUB-S", (down, 34.5, "hold", "hold", "14.6 / -13.9")), ("HUB-N", (down, 37.2, "hold", "hold", "11.9 / -13.9")),
        ("E2-W2", ("bridge", 60.7, 60.7, 30.7, "13.9 / -18.4")), ("E1-N", ("bridge", 63.0, 63.0, 34.1, "15.0 / -13.9")),
        ("W1-S", ("bridge", 63.0, 63.0, 41.2, "8.0 / -13.9"))])
    bridges = [k for k, v in att_set.items() if v[0] == "bridge"]
    return dict(
        name="PoC 1 (A7): DPIR + MARL recovery on 1 x 4-sector + 6 x 2-sector N367 and 8 IP-50",
        poc="PoC 1", drawing="UNITY6G-POC1-BP",
        boxes=boxes, nodes={k: dict(box=v[0], slot=v[1], sectors=v[2], pol=v[3]) for k, v in nodes.items()},
        module=module, link_ends=link_ends,
        splitters={("HUB", "W"): ["HUB-W1", "HUB-W2"], ("HUB", "E"): ["HUB-E1", "HUB-E2"]},
        couplings={}, mw_links=OrderedDict([("W1-CORE", ("W1", "CORE")), ("CORE-E1", ("CORE", "E1")),
                                            ("W1-W2", ("W1", "W2")), ("E1-E2", ("E1", "E2"))]),
        sites=["W1", "W2", "CORE", "E1", "E2", "S", "N"],
        hosts=OrderedDict([("W1", ["W1"]), ("W2", ["W2"]), ("E1", ["E1"]), ("E2", ["E2"]), ("S", ["S"]),
                           ("N", ["N"])]),
        agents=["W1", "W2", "E1", "E2", "S", "N"],
        servers=[("SRV-E2E", "TG E2E controller + NMS", "CORE"), ("SRV-DPIR", "DPIR planner + predictor", "CORE"),
                 ("BENCH-CTL", "bench controller: event script, PDU commands, logging (always on)", None)],
        no_data=["HUB"], att_set=att_set, if_pads=True, mw_pads=True,
        mw_set=OrderedDict([("W1-CORE", ("10 (fixed pad)", "CORE radio off")),
                            ("CORE-E1", ("10 (fixed pad)", "CORE radio off")),
                            ("W1-W2", ("10 (fixed pad)", "unchanged")), ("E1-E2", ("10 (fixed pad)", "unchanged"))]),
        event="t0: the devices that die lose power at their switched PDU outlets: the hub (its AC PoE injector), "
              "CORE's two IP-50 radios, SRV-E2E and SRV-DPIR. The hub's six paths and the W1-CORE and CORE-E1 hops go "
              "down with them; nothing RF moves at t0 (fixed pads, no programmable attenuators). W1-W2 and E1-E2 "
              f"unchanged. The bridges {', '.join(bridges)} (links between surviving nodes, on their fixed pads) are "
              "held down before t0 with Terragraph ignition disabled on the E2E controller (TG links "
              f"{', '.join(f'link-{link_ends[k][0][0]}-{link_ends[k][1][0]}' for k in bridges)}, names TBC). SRV-E2E is "
              "off after t0, so MARL engages a bridge at node level: the A-end node's agent PC sends tg2 minion assoc "
              "to its node (TBC on the N367 firmware; fallback: a standby E2E controller on BENCH-CTL). No RF change.",
        dcin_word="green 4-pole DC header",
        dcin_plug="4-pole pluggable screw-terminal plug, Phoenix MSTB 2.5/4-STF class, pitch TBC",
        event_off=["SRV-E2E", "SRV-DPIR", "HUB", "IP50-W1-CORE@CORE", "IP50-CORE-E1@CORE"],
    )


def bowtie_design():
    """Provisional PoC 2 (hub + two squares + two IP-50 rungs) until poc2.py supplies the final one."""
    pos = {"H": (0, 0), "A1": (0, 600), "A2": (600, 600), "A3": (600, 0), "B1": (-600, 0), "B2": (-600, -600),
           "B3": (0, -600)}

    def sec(x, y):
        return SEC[sector("DN", bearing(pos[x], pos[y]))]
    tg = OrderedDict([("HA1", ("H", "A1")), ("A12", ("A1", "A2")), ("A23", ("A2", "A3")), ("A3H", ("A3", "H")),
                      ("HB1", ("H", "B1")), ("B12", ("B1", "B2")), ("B23", ("B2", "B3")), ("B3H", ("B3", "H"))])
    nodes = {"A1": ("SB1", "a", 2, "odd"), "A3": ("SB1", "b", 2, "odd"), "H": ("SB2", "a", 4, "even"),
             "A2": ("SB3", "a", 2, "even"), "B2": ("SB3", "b", 2, "even"), "B1": ("SB4", "a", 2, "odd"),
             "B3": ("SB4", "b", 2, "odd")}
    boxes = [dict(id="SB1", code="GTS003 #1", kind="GTS003", nodes=["A1", "A3"]),
             dict(id="SB2", code="GTST002B", kind="GTST002B", nodes=["H"]),
             dict(id="SB3", code="GTST003", kind="GTST003", nodes=["A2", "B2"]),
             dict(id="SB4", code="GTS003 #2", kind="GTS003", nodes=["B1", "B3"])]
    link_ends = OrderedDict((k, ((a, sec(a, b)), (b, sec(b, a)))) for k, (a, b) in tg.items())
    module = {}
    for bx in boxes:
        k = 1
        for nd in bx["nodes"]:
            secs = sorted({e[1] for ends in link_ends.values() for e in ends if e[0] == nd}, key="ENWS".index)
            for sc in secs:
                module[(nd, sc)] = (bx["id"], f"M{k}")
                k += 1
    return dict(
        name="PoC 2 (provisional: hub + two squares + 2 IP-50 rungs; design pending)",
        poc="PoC 2", drawing="UNITY6G-POC2-BP",
        boxes=boxes, nodes={k: dict(box=v[0], slot=v[1], sectors=v[2], pol=v[3]) for k, v in nodes.items()},
        module=module, link_ends=link_ends, splitters={},
        couplings=OrderedDict([("K1", (("H", "N"), ("B1", "E"))), ("K2", (("H", "S"), ("A3", "W")))]),
        mw_links=OrderedDict([("MW", ("A1", "B1")), ("ME", ("A3", "B3"))]),
        sites=["A1", "A2", "A3", "H", "B1", "B2", "B3"],
        hosts=OrderedDict([("1", ["A1", "A2"]), ("2", ["B1", "B2"]), ("3", ["A3", "H"]), ("4", ["B3"])]),
        agents=[], servers=[("SRV-DPIR", "DPIR controller + predictor (GPU)", None),
                            ("SRV-E2E", "TG E2E controller + NMS (POP at H)", "H"),
                            ("BENCH-CTL", "bench controller: run script, ATT / PDU commands, logging (always on)", None)],
        no_data=[], event=None, att_reuse=9,
    )


def robust_design():
    """PoC 2, TENTATIVE: the 'robust' design picked by both reviewers of the more-paths redesign (7-node
    Terragraph bowtie, 8 fixed point-to-point links, 3 IP-50 rungs, 2 heuristic-4 coupling arms, 13 radio units).
    Box slots, module map, attenuator numbering and arms as in that design's bench notes."""
    boxes = [dict(id="SB1", code="GTS003 #1", kind="GTS003", nodes=["B", "E"]),
             dict(id="SB2", code="GTST002B", kind="GTST002B", nodes=["H"]),
             dict(id="SB3", code="GTST003", kind="GTST003", nodes=["A", "D"]),
             dict(id="SB4", code="GTS003 #2", kind="GTS003", nodes=["C", "F"])]
    nodes = {"B": ("SB1", "a", 2, "even"), "E": ("SB1", "b", 2, "even"), "H": ("SB2", "a", 4, "even"),
             "A": ("SB3", "a", 2, "odd"), "D": ("SB3", "b", 2, "odd"), "C": ("SB4", "a", 2, "odd"),
             "F": ("SB4", "b", 2, "odd")}
    module = {("B", "E"): ("SB1", "M1"), ("B", "S"): ("SB1", "M2"), ("E", "N"): ("SB1", "M3"), ("E", "W"): ("SB1", "M4"),
              ("H", "N"): ("SB2", "M1"), ("H", "E"): ("SB2", "M2"), ("H", "S"): ("SB2", "M3"), ("H", "W"): ("SB2", "M4"),
              ("A", "S"): ("SB3", "M1"), ("A", "W"): ("SB3", "M2"), ("D", "W"): ("SB3", "M3"), ("D", "S"): ("SB3", "M4"),
              ("C", "N"): ("SB4", "M1"), ("C", "E"): ("SB4", "M2"), ("F", "E"): ("SB4", "M3"), ("F", "N"): ("SB4", "M4")}
    link_ends = OrderedDict([
        ("THA", (("H", "N"), ("A", "S"))), ("TAB", (("A", "W"), ("B", "E"))), ("TBC", (("B", "S"), ("C", "N"))),
        ("TCH", (("C", "E"), ("H", "W"))), ("THD", (("H", "E"), ("D", "W"))), ("TDE", (("D", "S"), ("E", "N"))),
        ("TEF", (("E", "W"), ("F", "E"))), ("TFH", (("F", "N"), ("H", "S")))])
    snr = ("SNR 25 dB; fades", "cal.", "-", "-", "see poc2")
    att_set = OrderedDict([(k, snr) for k in link_ends] +
                          [("KBD", ("INR 13 dB", "cal.", "-", "-", "off = 63")), ("KAE", ("INR 13 dB", "cal.", "-", "-", "off = 63"))])
    return dict(
        name="PoC 2 (TENTATIVE 'robust'): 7-node TG bowtie, 3 IP-50 rungs, 2 coupling arms, DPIR only",
        poc="PoC 2", drawing="UNITY6G-POC2-BP",
        boxes=boxes, nodes={k: dict(box=v[0], slot=v[1], sectors=v[2], pol=v[3]) for k, v in nodes.items()},
        module=module, link_ends=link_ends, splitters={},
        couplings=OrderedDict([("KBD", (("B", "E"), ("D", "W"))), ("KAE", (("A", "S"), ("E", "N")))]),
        mw_links=OrderedDict([("MAF", ("A", "F")), ("MBE", ("B", "E")), ("MCD", ("C", "D"))]),
        sites=["H", "A", "B", "C", "D", "E", "F"],
        hosts=OrderedDict([("H", ["H"]), ("A", ["A"]), ("B", ["B"]), ("D", ["D"]), ("E", ["E"])]),
        agents=[], servers=[("SRV-DPIR", "DPIR controller + predictor (GPU)", None),
                            ("SRV-E2E", "TG E2E controller + NMS (POP at H)", "H"),
                            ("BENCH-CTL", "bench controller: run script, ATT / PDU commands, logging (always on)", None)],
        no_data=[], event=None, att_reuse=9, att_set=att_set, arm_dc_blocks=2,
        flows=[("F1", "H", "D", 1600), ("F2", "H", "A", 1600), ("F3", "B", "H", 1000), ("F4", "E", "H", 1000)],
    )


# ------------------------------------------------------------------------------------------
# the bench: every cable
# ------------------------------------------------------------------------------------------
class Bench:
    def __init__(self, d, single_tile=False):
        self.d = d
        self.tiles = 1 if single_tile else 4
        self.cables = []
        self.n = {}
        self.assumptions = []          # invented values, copied into the twin's assumptions list
        self._build()

    def cid(self, cls):
        self.n[cls] = self.n.get(cls, 0) + 1
        return f"{CLS_PREFIX[cls]}{self.n[cls]:03d}"

    def add(self, cls, a, b, part, length, note="", pn=None, vlan="", **kw):
        c = dict(id=self.cid(cls), cls=cls, a=a, b=b, part=part, length=length, note=note, pn=pn or part, vlan=vlan,
                 **kw)
        self.cables.append(c)
        return c

    def att_slot(self, att):
        return list(self.atts).index(att) % ATT_PER_SHELF

    def vlan(self, site):
        return 101 + self.d["sites"].index(site)

    def count(self, cls, part_sub=None):
        return sum(1 for c in self.cables if c["cls"] == cls and (part_sub is None or part_sub in c["part"]))

    def count_pn(self, pn):
        return sum(1 for c in self.cables if c["pn"] == pn)

    def outlet_of(self, ld):
        """'PDU-C.O2' for a rack load on a PDU outlet (None if it has none)."""
        o = self.outlet.get(ld)
        return f"{o[0]}.O{o[1]}" if o else None

    def bridge_after_text(self):
        """the order to restore the bench after a run (fixed-pad bench with bridges): SRV-E2E first, the bridges held down
        again and read back, then the hub and CORE, then the rest of the switched loads."""
        br = self.bridges
        sw = list(self.switched)

        def on(xs):
            return ", ".join(f"{x} ({self.outlet_of(x)})" for x in xs)
        e2e = [x for x in sw if x == "SRV-E2E"]
        hub = list(self.inj.values()) + list(self.psu.values())
        rest = [x for x in sw if x not in e2e + hub]
        return ((f"switch {on(e2e)} ON first and wait for the E2E controller to come up; " if e2e else "") +
                "disable ignition for the bridges again (" + ", ".join(v["tg"] for v in br.values()) + ") and read it "
                "back; a bridge MARL associated stays up: take it down from its A-end agent PC (tg2 minion dissoc, TBC) "
                "and check on the controller that every bridge is down; only then switch ON " + on(hub) +
                ", so the hub and CORE re-join with the bridges held down" + (f"; then {on(rest)}" if rest else "") +
                "; let every node and radio re-join before the next run")

    def load_w(self, ld):
        """typical mains draw of a rack or table load, W (TENTATIVE: catalog.LOAD_W; a rectifier module from its DC load,
        each module rated for the full load)."""
        base = ld.split(":")[0]
        if base == "DC-RECT":                                # nodes + Spider-box fans / Pi (DCIN)
            return (len(self.dc_nodes) * CAT.NODE_W + self.n_dcin * CAT.BOX_AUX_W) / CAT.RECT_EFF
        if base == "DC-RECT2":
            return sum(1 for k in self.ip50 if k not in self.psu) * CAT.IP50_W / CAT.RECT_EFF
        for pre in ("ATT", "AGENT", "HOST", "INJ", "PSU"):
            if base.startswith(pre):
                return CAT.LOAD_W[pre]
        return CAT.LOAD_W.get(base, 0)

    def _pdu_plan(self, rack_loads):
        """PDUs and {load: (PDU, outlet)}. Outlets in load order, PDU_OUT per PDU; a rectifier's 2nd / 3rd module goes on
        ANOTHER PDU (another wall circuit), on that PDU's first free outlet, so the modules do not share a supply and no
        other outlet number moves. If that plan puts a PDU over 80 % of its 16 A circuit (typical draw, load_w), the loads
        are spread evenly over one PDU more at a time until every PDU fits."""
        extra = [ld for ld in rack_loads if ":" in ld]
        base = [ld for ld in rack_loads if ld not in extra]

        def plan(n, per):
            pdus = [f"PDU-{'ABCDEF'[i]}" for i in range(n)]
            out = OrderedDict((ld, (pdus[i // per], i % per + 1)) for i, ld in enumerate(base))
            for ld in extra:
                home = out[ld.split(":")[0]][0]
                k = pdus.index(home)
                order = pdus[k + 1:] + pdus[:k] + [home]
                taken = set(out.values())
                free = [(p, o) for p in order for o in range(1, PDU_OUT + 1) if (p, o) not in taken]
                fits = [(p, o) for p, o in free if load(out, p) + self.load_w(ld) <= CAT.CIRCUIT_W]
                out[ld] = (fits or free)[0] if per < PDU_OUT else free[0]
            return pdus, out

        def load(out, p):
            return sum(self.load_w(ld) for ld, (pp, _) in out.items() if pp == p)

        n = math.ceil(len(rack_loads) / PDU_OUT)
        pdus, out = plan(n, PDU_OUT)
        while any(load(out, p) > CAT.CIRCUIT_W for p in pdus) and n < 6:
            per = math.ceil(len(base) / n)
            pdus, out = plan(n, per)
            if all(load(out, p) <= CAT.CIRCUIT_W for p in pdus):
                self.assumptions.append(
                    f"PDU outlets: {len(rack_loads)} rack loads fit {math.ceil(len(rack_loads) / PDU_OUT)} PDU(s) by outlet "
                    f"count, but the typical draw (catalog.LOAD_W estimates) would put one over 80 % of 16 A, so they are "
                    f"spread over {n} PDUs, about {per} loads each")
                break
            n += 1
        return pdus, out

    def cable(self, **match):
        cs = [c for c in self.cables if all(c[k] == v for k, v in match.items())]
        assert len(cs) == 1, (match, len(cs))
        return cs[0]

    def _build(self):
        d = self.d
        T = range(1, self.tiles + 1)
        nodes = list(d["nodes"])
        # units --------------------------------------------------------------------------------
        self.coupled = OrderedDict()
        for k, (x, y) in d["couplings"].items():
            for s in (x, y):
                self.coupled.setdefault(s, f"CPL-{s[0]}{s[1]}")
        self.split = OrderedDict((s, f"SPL-{s[0]}{s[1]}") for s in d.get("splitters", {}))
        order = d.get("att_order") or (list(d["link_ends"]) + list(d["couplings"]))
        self.atts = OrderedDict()
        # fixed pads (if_pads): no programmable attenuators; each link's pad stack is sized from its A7 setting
        self.if_pads = bool(d.get("if_pads"))
        self.pad_links = OrderedDict()
        self.pads = OrderedDict()           # pad stack id -> its data (chain, link, tile, host port, pads, dB)
        self.pad_host = {}                  # host port -> pad stack id (the stack is screwed straight onto it)
        if self.if_pads:
            assert not d["couplings"], "fixed pads replace link attenuators only"
            for name in d["link_ends"]:
                row = d["att_set"][name]
                self.pad_links[name] = dict(pad_stack(pad_target(row)), job=row[0],
                                            setting="after" if isinstance(row[3], (int, float)) else "before")
                self.pad_links[name].update(link_expect(row, self.pad_links[name]["db"]))
        else:
            for name in order:
                self.atts[f"ATT-{len(self.atts) + 1:02d}"] = ("link" if name in d["link_ends"] else "arm", name)
        self.att_of = {v[1]: k for k, v in self.atts.items()}
        self.bridges = bridge_ops(d)        # fixed pads: the MARL bridges, held down by Terragraph ignition
        self.ip50 = OrderedDict()
        for k, (a, b) in d.get("mw_links", {}).items():
            self.ip50[f"IP50-{k}@{a}"] = (k, a)
            self.ip50[f"IP50-{k}@{b}"] = (k, b)
        self.mw_pads = bool(d.get("mw_pads")) and bool(self.ip50)
        self.mwatt = bool(self.ip50) and not self.mw_pads          # Keysight J7204B on the table
        # IP-50 radios stacked on spacer bars: the 3rd / 4th hop on the 1st / 2nd (MWATT bench); with fixed pads the two
        # radios of a hop sit side by side, none stacked
        self.stacked_hops = list(d.get("mw_links", {}))[2:] if self.mwatt else []
        off = d.get("event_off", [])
        # devices that go down at t0 are switched at the PDU: nodes through an AC PoE injector, radios through
        # their own supply; every other node keeps its passive DC/PoE adaptor on the 48 V drawer
        self.inj = OrderedDict((n, f"INJ-{n}") for n in nodes if n in off)
        self.psu = OrderedDict((k, f"PSU-{k.replace('IP50-', '')}") for k in self.ip50 if k in off)
        self.dc_nodes = [n for n in nodes if n not in off]
        self.adaptor = {n: f"PA-{i + 1:02d}" for i, n in enumerate(self.dc_nodes)}
        self.n_adaptors = max(10, len(self.dc_nodes))
        self.switched = [s[0] for s in d["servers"] if s[0] in off] + list(self.inj.values()) + list(self.psu.values())
        # AC -----------------------------------------------------------------------------------
        # one load per rectifier MODULE (RCP-1UI: one C14 inlet per module slot); 'DC-RECT:2' = module 2's inlet
        self.load_port = {f"{r}:{k}": f"{r}.ac{k}" for r, n in RECT_MODULES.items() for k in range(2, n + 1)}
        rects = [f"DC-RECT:{k}" if k > 1 else "DC-RECT" for k in range(1, RECT_MODULES["DC-RECT"] + 1)]
        rack_loads = rects + ["SW-DATA", "SW-OOB", "GNSS-SPL"] + list(self.atts)
        if self.ip50:
            rack_loads[len(rects):len(rects)] = [f"DC-RECT2:{k}" if k > 1 else "DC-RECT2"
                                                 for k in range(1, RECT_MODULES["DC-RECT2"] + 1)]
        rack_loads += [f"AGENT-{s}" for s in d.get("agents", [])]
        self.rack_servers = [s[0] for s in d["servers"] if s[0] not in TABLE_SERVERS]
        rack_loads += [x for x in self.rack_servers if x not in self.switched] + [f"HOST-{h}" for h in d["hosts"]]
        rack_loads += self.switched                         # event loads go on switched rack-PDU outlets
        # the Spider-box DCIN feeds (box fans + Pi) load DC-RECT: every box is fed while a DC-DIST breaker is spare
        self.n_dcin = len(d["boxes"]) if len(self.dc_nodes) < 10 else 0
        self.pdus, out = self._pdu_plan(rack_loads)
        for i, p in enumerate(self.pdus):
            self.add("AC", f"WALL.S{i + 1}", f"{p}.inlet", "PDU cord (IEC 60309 16 A plug)", "fixed",
                     "PDU's own cord: route it, but leave the plug OUT of the wall until power-up", pn="PDU cord")
        self.outlet = OrderedDict(sorted(out.items(), key=lambda kv: (self.pdus.index(kv[1][0]), kv[1][1])))

        def outlet_lead(ld, pdu, o):
            if ld.startswith("AGENT"):
                self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", "Eaton Tripp Lite P014-006 C14-C5 'Mickey' lead", "1.8 m",
                         "agent PC power brick", pn="P014-006")
            elif ld in self.inj.values():
                self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", "Eaton Tripp Lite P014-006 C14-C5 lead (injector inlet TBC)",
                         "1.8 m", "switched off at t0 (outlet switching): the node loses power", pn="P014-006")
            elif ld in self.switched:
                srv = ld.startswith("SRV-")
                self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", "Eaton Tripp Lite P004-010 C14-C13 lead", "3.05 m",
                         "switched off at t0 (outlet switching)" +
                         ("; one PSU only: if the server has two, REMOVE PSU 2 before it goes in the rack (only this "
                          "outlet is switched at t0)" if srv else ""),
                         pn="P004-010")
            else:
                base = ld.split(":")[0]
                note = ("attenuator box power: own adapter, inlet TBC" if ld.startswith("ATT") else
                        f"rectifier module {ld.split(':')[1] if ':' in ld else 1} of {RECT_MODULES[base]} (one cord per module)"
                        if base in RECT_MODULES else "")
                self.add("AC", f"{pdu}.O{o}", self.load_port.get(ld, f"{ld}.ac"), "Eaton Tripp Lite P004-006 C14-C13 lead",
                         "1.8 m", note, pn="P004-006")
        for ld, (pdu, o) in self.outlet.items():
            if ":" not in ld:
                outlet_lead(ld, pdu, o)
        self.table_loads = [s[0] for s in d["servers"] if s[0] in TABLE_SERVERS] + ["MONITOR", "SPECAN"] + \
            (["MWATT"] if self.mwatt else [])
        for i, t in enumerate(self.table_loads):
            self.add("AC", f"TABLE-STRIP.O{i + 1}", f"{t}.ac", "Eaton Tripp Lite P004-006 C14-C13 lead", "1.8 m",
                     "inlet type TBC" if t == "MWATT" else "", pn="P004-006")
        # a rectifier's 2nd / 3rd module cord takes the next free ID after the strip loads, so every earlier AC ID keeps
        # its number
        for ld, (pdu, o) in self.outlet.items():
            if ":" in ld:
                outlet_lead(ld, pdu, o)
        # the strip's own supply: a wall socket of its own, never a switched PDU outlet (BENCH-CTL runs the event)
        self.add("AC", f"WALL.S{len(self.pdus) + 1}", "TABLE-STRIP.inlet",
                 "CORD-60309-C19-3M power cord IEC 60309 16 A (2P+E) plug to IEC C19, 3 m (part TBC)", "3 m",
                 "bench strip supply on its own wall socket (not a PDU outlet): BENCH-CTL stays powered through t0; "
                 "leave the plug OUT of the wall until power-up", pn="CORD-60309-C19-3M")
        # operator station: the bench controller drives the monitor, keyboard and mouse
        self.add("AUX", "BENCH-CTL.video", "MONITOR.in", "DisplayPort / HDMI cable 1.8 m (part TBC)", "1.8 m",
                 "operator station", pn="video cable")
        self.add("AUX", "KBD.usb", "BENCH-CTL.usb", "USB keyboard + mouse (cables attached)", "1.5 m", "operator station",
                 pn="keyboard + mouse")
        # DC -----------------------------------------------------------------------------------
        # conductor convention (every D run): the SUPPLY conductor gets a red (+48 V) or blue (-48 V) sleeve at both
        # ends before it is landed; the unsleeved conductor is the 0 V return
        self.add("DC", "DC-RECT.bus", "DC-DIST.bus", "DC bus link 2 x 10 AWG with ring lugs: RECT +V to DIST + bus, "
                 "RECT -V to DIST return", "0.3 m", "+48 V bus, output floating (not earthed); " + bus_protection("DC-RECT"),
                 pn="bus link")
        for i, nd in enumerate(self.dc_nodes):
            self.add("DC", f"DC-DIST.CB{i + 1}", f"PA-{i + 1:02d}.dc", "Belden 8461 2 x 18 AWG + Phoenix 3200519 ferrules",
                     "1 m", f"feeds node {nd}: breaker output (+48 V, red sleeve) to the adaptor + terminal, return to "
                     "its - terminal", pn="Belden 8461", metres=1.0)
        self.fuse = OrderedDict()
        if self.ip50:
            self.add("DC", "DC-RECT2.out", "DC-DIST2.bus", "DC bus link 2 x 10 AWG with ring lugs: RECT2 -V to DC-DIST2 "
                     "BAT A (-48 V), RECT2 +V to RTN", "0.3 m",
                     "-48 V bus; positive grounded by the green/yellow bond from the RTN bus to the rack earth bar; "
                     "DC-DIST2 has no main breaker: " + bus_protection("DC-RECT2"), pn="bus link")
            for k in self.ip50:
                if k in self.psu:
                    self.add("DC", f"{self.psu[k]}.out", f"{k}.dc", "Belden 8461 2 x 18 AWG + barrel-to-terminal adaptor "
                             "(IP-50 connector TBC)", "1 m",
                             "floating -48 V from its own isolated supply, switched off at t0: PSU - (blue sleeve) to the "
                             "radio -48 V pin, PSU + to the radio 0 V pin; not earthed", pn="Belden 8461", metres=1.0)
                else:
                    self.fuse[k] = len(self.fuse) + 1
                    self.add("DC", f"DC-DIST2.F{self.fuse[k]}", f"{k}.dc", "Belden 8461 2 x 18 AWG (IP-50 connector TBC)",
                             "3 m", "-48 V through a GMT fuse: fuse output (-48 V, blue sleeve) to the radio -48 V pin, "
                             "RTN to the radio 0 V pin", pn="Belden 8461", metres=3.0)
            # after the feeds, so the feed IDs stay as they were
            self.add("DC", "DC-DIST2.strapA", "DC-DIST2.busB", "DC-DIST2 bus A to bus B strap (-48 V and RTN), 2 x 10 AWG "
                     "with ring lugs", "0.3 m", "feeds bus B (fuses F6-F10) from bus A; omit if the panel ships strapped",
                     pn="bus strap")
        # Spider-box DC terminal blocks (DCIN: box fans + Raspberry Pi, 48 V assumed, TBC at the step 'Spider boxes' hold
        # point): one spare DC-DIST breaker per box; with fewer spare breakers than boxes, the last boxes share the last
        # breaker (two runs in one breaker output terminal, twin ferrule). After the strap, so earlier DC IDs keep theirs
        spare = list(range(len(self.dc_nodes) + 1, 11))
        self.dcin = OrderedDict()
        for i, x in enumerate(d["boxes"]):
            if not spare:
                break
            cb = spare[min(i, len(spare) - 1)]
            second = any(v[0] == cb for v in self.dcin.values())
            self.dcin[x["id"]] = (cb, f"DC-DIST.CB{cb}" + ("b" if second else ""))
        for bx, (cb, port) in self.dcin.items():
            mates = [k for k, v in self.dcin.items() if v[0] == cb and k != bx]
            self.add("DC", port, f"{bx}.DCIN", "Belden 8461 2 x 18 AWG + Phoenix 3200519 ferrules" +
                     (" (twin ferrule at the shared breaker output)" if mates else ""), "1 m",
                     f"{bx} box fans + Raspberry Pi: CB{cb} output (+48 V, red sleeve) to the DCIN + terminal, return to "
                     "its - terminal" + (f"; CB{cb} is shared with {', '.join(mates)} (both supply conductors in one twin "
                                         "ferrule)" if mates else "") +
                     "; CONDITIONAL: land it only once the Spider-box hold point has confirmed DCIN is a 48 V DC fan / Pi "
                     "input; any other voltage or function: STOP, design change", pn="Belden 8461", metres=1.0,
                     dcin=bx, twin=bool(mates))
        self.spare_breakers = [cb for cb in spare if cb not in {v[0] for v in self.dcin.values()}]
        # earth bonding: the rack frame to the lab earth bar, the -48 V return to the rack earth (positive ground), the
        # GNSS arrestor to the lab earth bar, and (IP-50 radios) the bench-table earth bar to the rack earth with every
        # radio chassis bonded to it
        self.add("PE", "RACK-EARTH.E1", "LAB-EARTH.bar", "green/yellow earth cable 6 mm2 (10 AWG), 2-hole lug at the "
                 "rack bar, ring lug at the lab bar", "site (allow 5 m)", "rack earth bar to the lab earth bar; fitted "
                 "first, before anything is powered", pn="PE 6 mm2", metres=5.0)
        if self.ip50:
            self.add("PE", "DC-DIST2.rtn", "RACK-EARTH.E2", "green/yellow earth cable 6 mm2 (10 AWG) with ring lugs", "1 m",
                     "-48 V return bus (= DC-RECT2 +V) to the rack earth: this bond makes the -48 V system positive "
                     "grounded", pn="PE 6 mm2", metres=1.0)
        self.add("PE", "GNSS-ARR.gnd", "LAB-EARTH.bar2", "green/yellow earth cable 6 mm2 (10 AWG) with ring lugs, short and "
                 "straight, no loops", "0.3 m", "GNSS surge arrestor ground lug to the lab earth bar; fitted BEFORE the "
                 "antenna lead is connected (if the building has its own entry arrestor, bond to that: facilities)",
                 pn="PE 6 mm2", metres=0.3)
        if self.ip50:
            self.add("PE", "TABLE-EARTH.bond", "RACK-EARTH.E3", "green/yellow earth cable 6 mm2 (10 AWG) with ring lugs",
                     "4 m", "bench-table earth bar to the rack earth bar (the radios' one earth point, through the rack); "
                     "along the floor under the cable protector with the strip cord", pn="PE 6 mm2", metres=4.0)
            for i, k in enumerate(self.ip50):
                self.add("PE", f"{k}.gnd", f"TABLE-EARTH.R{i + 1}", "green/yellow earth cable 2.5 mm2 (14 AWG) with ring "
                         "lugs", "1 m", "radio chassis ground screw to the table earth bar", pn="PE 2.5 mm2", metres=1.0)
        for n in nodes:
            if n in self.inj:
                self.add("POE", f"{self.inj[n]}.poe", f"{d['nodes'][n]['box']}.{n}.ETH1", "Panduit STP6X1MBU Cat6A S/FTP 1 m",
                         "1 m", "PoE from the AC injector (power only); switched off at t0", pn="STP6X1MBU")
            else:
                self.add("POE", f"{self.adaptor[n]}.rj45", f"{d['nodes'][n]['box']}.{n}.ETH1",
                         "Panduit STP6X1MBU Cat6A S/FTP 1 m", "1 m", "power only (passive adaptor)", pn="STP6X1MBU")
        # data: SW-DATA, 1G endpoints on P1-16 (the only 1G-capable cages), then 10G, site by site ---------
        eps = []
        for site in d["sites"]:
            v = str(self.vlan(site))
            if site in d["nodes"] and site not in d.get("no_data", []):
                eps.append((site, "10G", f"{d['nodes'][site]['box']}.{site}.SFP+", OM4, "FZ2ERLNLNSNM002", "2 m",
                            f"node {site} data, in-band management", v))
            for k, (hop, st) in self.ip50.items():
                if st == site:
                    eps.append((site, "10G", f"{k}.data", "SFP+ SR at both ends (IP-50CX SFP+ optic TBC / Netgear AXM761) + "
                                "Panduit FZ2ERLNLNSNM002 OM4 LC-LC 2 m", "FZ2ERLNLNSNM002", "2 m",
                                f"hop {hop}: IP-50CX 10GbE port (a hop can exceed 1 Gb/s)", v))
            if site in d.get("agents", []):
                eps.append((site, "1G", f"AGENT-{site}.data", f"Panduit STP6X2MBU Cat6A 2 m + {AGM734} at the switch",
                            "STP6X2MBU", "2 m", "MARL agent + site routing", v))
            for h, sts in d["hosts"].items():
                if sts == [site]:
                    eps.append((site, "10G", f"HOST-{h}.nic10g", DAC, "AXC763-10000S", "3 m", "traffic host", v))
            for sid, role, st in d["servers"]:
                if st == site:
                    eps.append((site, "10G", f"{sid}.data", DAC, "AXC763-10000S", "3 m", role, v))
        for h, sts in d["hosts"].items():
            if len(sts) > 1:
                eps.append(("trunk", "10G", f"HOST-{h}.nic10g", DAC, "AXC763-10000S", "3 m",
                            "traffic host, one sub-interface per VLAN", "trunk " + ",".join(str(self.vlan(x)) for x in sts)))
        one = [e for e in eps if e[1] == "1G"]
        ten = [e for e in eps if e[1] == "10G"]
        assert len(one) <= DATA_1G and len(one) + len(ten) <= DATA_PORTS, (len(one), len(ten))
        self.sw_port = OrderedDict()
        for p, (site, speed, src, part, pn, length, note, v) in enumerate(one + ten, start=1):
            self.sw_port[src] = (p, v, site, speed)
            self.add("FIB", src, f"SW-DATA.P{p}", part, length, note, pn=pn, vlan=v, site=site, speed=speed)
        self.n_agm_data = len(one)
        # management: SW-OOB (lab equipment and the PCs' second NICs only; no node ports) ------------
        oob = []
        for site in d["sites"]:
            if site in d.get("agents", []):
                oob.append((f"AGENT-{site}.mgmt", "agent PC logging", "STP6X2MBU", "2 m", site))
            for h, sts in d["hosts"].items():
                if sts == [site]:
                    oob.append((f"HOST-{h}.mgmt", "traffic host control", "STP6X3MBU", "3 m", site))
        for sid, role, st in d["servers"]:
            oob.append((f"{sid}.mgmt", ("bench controller: PDU and attenuator control, logging" if self.atts or self.mwatt
                                        else "bench controller: PDU control, logging") if sid in TABLE_SERVERS
                        else "server management", "STP6X3MBU", "3 m", st or "table"))
        for h, sts in d["hosts"].items():
            if len(sts) > 1:
                oob.append((f"HOST-{h}.mgmt", "traffic host control", "STP6X3MBU", "3 m", "trunk"))
        for a in self.atts:
            oob.append((f"{a}.lan", "attenuator control, SCPI over TCP", "STP6X2MBU", "2 m", "rack"))
        if self.mwatt:
            oob.append(("MWATT.lan", "J7204B control (LAN)", "STP6X3MBU", "3 m", "rack"))
        for pdu in self.pdus:
            oob.append((f"{pdu}.net", "PDU management, outlet switching", "STP6X2MBU", "2 m", "rack"))
        oob.append(("SPECAN.lan", "analyser", "STP6X3MBU", "3 m", "rack"))
        assert len(oob) <= OOB_RJ45 + OOB_SFP, len(oob)
        self.oob_port = OrderedDict()
        for q, (src, note, pn, length, grp) in enumerate(oob, start=1):
            sfp = q > OOB_RJ45
            self.oob_port[src] = (q, grp)
            self.add("ETH", src, f"SW-OOB.P{q}", f"Panduit {pn} Cat6A {length}" + (f" + {AGM734} (SFP+ cage)" if sfp else ""),
                     length, note, pn=pn, vlan="100", grp=grp)
        self.n_agm_oob = max(0, len(oob) - OOB_RJ45)
        self.oob_model = ("M4250-26G4XF-PoE+ (GSM4230PX)", OOB_RJ45, OOB_SFP)
        # GNSS -------------------------------------------------------------------------------------
        self.add("GNSS", "GNSS-ANT", "GNSS-ARR.ant", "LMR-400 antenna lead, N connectors (part TBC), from the roof "
                 "antenna to the surge arrestor where the lead enters the room", "site",
                 "connect it at the arrestor only after the arrestor's earth bond is fitted", pn="LMR-400")
        for i, nd in enumerate(nodes):
            self.add("GNSS", f"GNSS-SPL.out{i + 1}", f"{d['nodes'][nd]['box']}.{nd}.GNSS", "RG-316 SMA coax 1 m (part TBC)",
                     "1 m", "only if the node's rear SMA is a GNSS input (TBC)", pn="RG-316 SMA 1 m")
        # after the node jumpers, so G002... keep their IDs
        self.add("GNSS", "GNSS-ARR.eqp", "GNSS-SPL.in", CAT.LADDER["LMR240"][2][2], "3 m",
                 "arrestor (equipment side) to the splitter input; the splitter feeds the antenna DC through the arrestor",
                 pn=CAT.LADDER["LMR240"][2][0])
        # RF / IF: one bundle (4 tiles) at a time so that each bundle has consecutive IDs --------------
        tray = "SPL-TRAY" if self.split else "CPL-TRAY"
        for s, sid in self.split.items():
            box, mod = d["module"][s]
            for t in T:
                x = mod_x(mod, t)
                self.add("RF", f"{box}.{mod}.T{t}", f"{sid}.t{t}.S", "{pn}", "", f"{s[0]}.{s[1]} tile {t} to its splitter",
                         geo=(box, x, tray, x), bundle=("spl", s))
        if self.if_pads:
            self._pad_links(T, tray)
        for name, (sa, sb) in (() if self.if_pads else d["link_ends"].items()):
            att = self.att_of[name]
            for side, s in (("in", sa), ("out", sb)):
                box, mod = d["module"][s]
                if s in self.split:
                    k = d["splitters"][s].index(name) + 1
                    for t in T:
                        self.add("RF", f"{self.split[s]}.t{t}.P{k}", f"{att}.ch{t}.{side}", "{pn}", "", f"{name} tile {t}",
                                 geo=(tray, mod_x(mod, t), att, ch_x(t, side, self.att_slot(att))), bundle=(name, side))
                elif s in self.coupled:
                    for t in T:
                        x = mod_x(mod, t)
                        self.add("RF", f"{box}.{mod}.T{t}", f"{self.coupled[s]}.t{t}.IN", "{pn}", "",
                                 f"{name} {s[0]}.{s[1]} tile {t}: module to coupler IN", geo=(box, x, tray, x),
                                 bundle=(name, side, "cpl"))
                    for t in T:
                        self.add("RF", f"{self.coupled[s]}.t{t}.THRU", f"{att}.ch{t}.{side}", "{pn}", "", f"{name} tile {t}",
                                 geo=(tray, mod_x(mod, t), att, ch_x(t, side, self.att_slot(att))), bundle=(name, side))
                else:
                    for t in T:
                        self.add("RF", f"{box}.{mod}.T{t}", f"{att}.ch{t}.{side}", "{pn}", "", f"{name} {s[0]}.{s[1]} tile {t}",
                                 geo=(box, mod_x(mod, t), att, ch_x(t, side, self.att_slot(att))), bundle=(name, side))
        for name, (x, y) in d["couplings"].items():
            att = self.att_of[name]
            for t in T:
                self.add("RF", f"{self.coupled[x]}.t{t}.CPL", f"{att}.ch{t}.in",
                         "BLK-18-S+ DC block + {pad} on the coupler's CPL port, then {pn}", "",
                         f"arm {name} tile {t}; same stock length as the other 3 tiles (equal length, not a phase-matched set)",
                         geo=(tray, mod_x(d["module"][x][1], t), att, ch_x(t, "in", self.att_slot(att))), bundle=(name, "in"))
            both = d.get("arm_dc_blocks", 1) == 2
            for t in T:
                self.add("RF", f"{att}.ch{t}.out", f"{self.coupled[y]}.t{t}.CPL",
                         "{pn}, then BLK-18-S+ DC block on the coupler's CPL port" if both else "{pn}", "", f"arm {name} tile {t}",
                         geo=(att, ch_x(t, "out", self.att_slot(att)), tray, mod_x(d["module"][y][1], t)), bundle=(name, "out"))
        # IP-50 chains ------------------------------------------------------------------------------
        for j, (k, (a, b)) in enumerate(d.get("mw_links", {}).items()):
            ra, rb = f"IP50-{k}@{a}", f"IP50-{k}@{b}"
            if self.mw_pads:
                self._mw_pad_chain(k, ra, rb)
                continue
            self.add("WG", f"{ra}.wg", f"WGT-{k}-A.wg", "waveguide-to-coax transition PEWCA1003 (band TBC)", "flange",
                     f"hop {k}", pn="PEWCA1003")
            self.add("WG", f"WGT-{k}-A.coax", f"MWATT.ch{j + 1}.in",
                     "SA18S10W-30 pad on the transition, then Mini-Circuits CBL-1M-SMSM+", "1 m", f"hop {k}",
                     pn="CBL-1M-SMSM+")
            self.add("WG", f"MWATT.ch{j + 1}.out", f"WGT-{k}-B.coax",
                     "Mini-Circuits CBL-1M-SMSM+, then SA18S10W-30 pad on the transition", "1 m", f"hop {k}",
                     pn="CBL-1M-SMSM+")
            self.add("WG", f"WGT-{k}-B.wg", f"{rb}.wg", "waveguide-to-coax transition PEWCA1003 (band TBC)", "flange",
                     f"hop {k}", pn="PEWCA1003")
        self.size, self.rows, self.total = rack_layout(self)
        self._size_rf()

    def _pad_links(self, T, tray):
        """if_pads: one cable per tile path, from a fixed pad stack screwed straight onto the A-end port (the hub
        splitter's P1 / P2 for a split hub module, else the A tile) to the B tile; one bundle (4 tiles) per link."""
        d = self.d
        for name, (sa, sb) in d["link_ends"].items():
            pl = self.pad_links[name]
            parts_ = pad_parts(pl["pn"])
            bx_b, m_b = d["module"][sb]
            for t in T:
                if sa in self.split:
                    k = d["splitters"][sa].index(name) + 1
                    host, ua, xa = f"{self.split[sa]}.t{t}.P{k}", tray, mod_x(d["module"][sa][1], t)
                else:
                    bx_a, m_a = d["module"][sa]
                    host, ua, xa = f"{bx_a}.{m_a}.T{t}", bx_a, mod_x(m_a, t)
                pid = f"PAD-{name}.t{t}"
                self.pads[pid] = dict(chain="if", link=name, tile=t, host=host, side="a", parts=parts_, db=pl["db"],
                                      target=pl["target"], rounded=pl["rounded"], text=pl["text"],
                                      stack_mm=sum(p["len_mm"] for p in parts_), dia_mm=max(p["dia_mm"] for p in parts_))
                self.pad_host[host] = pid
                self.add("RF", f"{pid}.out", f"{bx_b}.{m_b}.T{t}", "{pn}", "",
                         f"{name} tile {t}: from the {pl['text']} dB pad stack on {host} to {sb[0]}.{sb[1]} tile {t}",
                         geo=(ua, xa, bx_b, mod_x(m_b, t)), bundle=(name, "pad"), pad=pid)

    def _mw_pad_chain(self, k, ra, rb):
        """mw_pads: radio - transition - 30 dB 10 W pad (- 10 dB pad, A side) - ONE SMA cable - 30 dB 10 W pad -
        transition - radio; the pads are stacks screwed onto the transition SMA (WGP-<hop>-A / -B)."""
        stacks = {"A": [MW_PAD_HI[0], MW_PAD_LO[0]], "B": [MW_PAD_HI[0]]}
        for side, pns in stacks.items():
            parts_ = pad_parts(pns)
            pid = f"WGP-{k}-{side}"
            host = f"WGT-{k}-{side}.coax"
            self.pads[pid] = dict(chain="ip50", link=k, tile=None, host=host, side=side.lower(), parts=parts_,
                                  db=sum(p["db"] for p in parts_), target=None, rounded=None,
                                  text=" + ".join(str(p["db"]) for p in parts_),
                                  stack_mm=sum(p["len_mm"] for p in parts_), dia_mm=max(p["dia_mm"] for p in parts_))
            self.pad_host[host] = pid
        self.add("WG", f"{ra}.wg", f"WGT-{k}-A.wg", "waveguide-to-coax transition PEWCA1003 (band TBC)", "flange",
                 f"hop {k}", pn="PEWCA1003")
        self.add("WG", f"WGP-{k}-A.out", f"WGP-{k}-B.out",
                 f"Mini-Circuits CBL-1FT-SMSM+ between the pad stacks (A: {MW_PAD_HI[0]} {MW_PAD_HI[1]} dB + {MW_PAD_LO[0]} "
                 f"{MW_PAD_LO[1]} dB on the transition; B: {MW_PAD_HI[0]} {MW_PAD_HI[1]} dB on the transition)", "0.30 m",
                 f"hop {k}: the one SMA cable of the chain, {2 * MW_PAD_HI[1] + MW_PAD_LO[1]} dB of fixed pads",
                 pn="CBL-1FT-SMSM+")
        self.add("WG", f"WGT-{k}-B.wg", f"{rb}.wg", "waveguide-to-coax transition PEWCA1003 (band TBC)", "flange",
                 f"hop {k}", pn="PEWCA1003")

    def _size_rf(self):
        """Pick each IF bundle's cable from the rack span: vertical distance between the two units' centres +
        horizontal offset between the ports (assumed panel positions) + BEND_MM; all 4 tiles get the same length."""
        centre = {r["id"]: (r["top"] - r["n"] / 2) * U_MM for r in self.rows}
        top = {r["id"]: r["top"] for r in self.rows}
        for a, (sid, slot) in self.att_shelf.items():
            centre[a], top[a] = centre[sid], top[sid]
        groups = OrderedDict()
        for c in self.cables:
            if c["cls"] == "RF":
                groups.setdefault(c["bundle"], []).append(c)
        for cs in groups.values():
            need = max(abs(centre[ua] - centre[ub]) + abs(xa - xb) + BEND_MM for ua, xa, ub, xb in (c["geo"] for c in cs))
            pn, L = next(s for s in RF_STOCK if s[1] >= need)
            ua, _, ub, _ = cs[0]["geo"]
            span = abs(top[ua] - top[ub])
            for c in cs:
                c["pn"] = pn
                c.setdefault("tpl", c["part"])
                c["length"] = f"{L / 1000:.2f} m"
                c["need_mm"] = round(need)
                c["note"] += f"; span {span}U, needs {need / 1000:.2f} m"
        self.size_arm_pads()

    def size_arm_pads(self):
        """Fixed pad on each coupling arm's input (BW-S<n>W2+), from a TENTATIVE arm budget: the pad is the largest
        existing value that still lets the arm attenuator at 0 dB reach the TOP of the INR sweep. Re-run after the
        cable lengths change (twin), then fills every IF cable's part text."""
        A = ARM_BUDGET
        self.arm_pad = OrderedDict()
        for name in self.d["couplings"]:
            cs = [c for c in self.cables if c["cls"] == "RF" and c.get("bundle", ("",))[0] == name]
            lm = sum(max(float(c["length"].split()[0]) for c in cs if c["bundle"][1] == side) for side in ("in", "out"))
            nd = self.d.get("arm_dc_blocks", 1)
            fixed = 2 * A["coupling_db"] + nd * A["dc_block_db"] + A["att_il_db"] + lm * A["cable_db_per_m"] + A["conn_db"]
            top = A["span_db"] + A["snr_db"] - A["inr_max_db"]          # arm loss at the top of the INR sweep
            need = top - fixed
            val = max([v for v in BW_S_DB if v <= need] or [BW_S_DB[0]])
            job = (self.d.get("att_set") or {}).get(name, ("",))[0]
            m = re.search(r"INR\s*([-\d.]+)", job)
            inr_nom = float(m.group(1)) if m else 13.0
            att_nom = A["span_db"] + A["snr_db"] - inr_nom - fixed - val
            self.arm_pad[name] = dict(db=val, pn=f"BW-S{val}W2+", need=round(need, 1), fixed=round(fixed, 1),
                                      att_nominal=round(att_nom, 1), inr_nominal=inr_nom, cable_m=round(lm, 2))
        if self.arm_pad:
            a = ", ".join(f"{k}: fixed {v['fixed']} dB, pad {v['db']} dB ({v['pn']}), ATT about {v['att_nominal']} dB at "
                          f"INR {v['inr_nominal']:g} dB" for k, v in self.arm_pad.items())
            self.assumptions = [x for x in self.assumptions if not x.startswith("Coupling-arm pad")]
            self.assumptions.append(
                f"Coupling-arm pad (TENTATIVE budget): tile-to-tile span loss {A['span_db']:g} dB for SNR {A['snr_db']:g} dB "
                f"(IF TX about 0 dBm, noise about -71 dBm in 2.16 GHz: measure at the port check), 2 x {A['coupling_db']:g} dB "
                f"coupling, {A['dc_block_db']:g} dB per DC block, {A['att_il_db']:g} dB attenuator insertion loss, "
                f"{A['cable_db_per_m']:g} dB/m cable; pad = largest BW-S value that keeps INR {A['inr_max_db']:g} dB reachable "
                f"at 0 dB attenuation. {a}. Swap the pad if the measured span loss differs.")
        self.format_rf_parts()

    def format_rf_parts(self):
        """IF cable part text from its template, stock part and (arm input) pad."""
        for c in self.cables:
            if c.get("tpl"):
                arm = self.arm_pad.get(c.get("bundle", ("",))[0]) if hasattr(self, "arm_pad") else None
                pad = f"Mini-Circuits {arm['pn']} {arm['db']} dB pad" if arm else "pad"
                c["part"] = c["tpl"].format(pn=f"Mini-Circuits {c['pn']}", pad=pad)

    def bundle(self, a_prefix, b_prefix):
        """The tile cables from a_prefix to b_prefix, checked to be one consecutive bundle."""
        cs = [c for c in self.cables if c["cls"] == "RF" and c["a"].startswith(a_prefix) and c["b"].startswith(b_prefix)]
        nums = [int(c["id"][1:]) for c in cs]
        assert len(cs) == self.tiles and nums == list(range(nums[0], nums[0] + len(nums))), (a_prefix, b_prefix)
        return cs


# ------------------------------------------------------------------------------------------
# rack layout: stacked from U1 up, power at the bottom as in figures/bench_a7.mjs
# ------------------------------------------------------------------------------------------
def rack_layout(b):
    d = b.d
    units = [("GNSS-SPL", 1, "GNSS splitter 1x8 (if the boxes take GNSS)", "RMALDCBS1X8"),
             ("SW-DATA", 1, "Data switch, VLAN per site", "Netgear M4350-24F4V"),
             ("SW-OOB", 1, "Management switch (lab equipment)", "Netgear " + b.oob_model[0])]
    for sid in b.rack_servers:
        role = next(s[1] for s in d["servers"] if s[0] == sid)
        units.append((sid, SERVER_U.get(sid, 1), f"{sid}  ·  {role[:40]}", f"{SERVER_U.get(sid, 1)}U rack server (lab)"))
    for h, sts in d["hosts"].items():
        units.append((f"HOST-{h}", 1, f"HOST-{h}  ·  traffic host, sites {','.join(sts)}", "1U rack server, 10G SFP+ NIC"))
    if d.get("agents"):
        units.append(("AGENTS", 2, f"Shelf: {len(d['agents'])} MARL agent PCs", "mini PCs on a 1U shelf, 2U clearance"))
    units.append(("CM", 1, "Cable manager", "Panduit NCMHF1"))
    for x in d["boxes"]:
        pols = sorted({d["nodes"][n]["pol"].split(" ")[0] for n in x["nodes"]})
        units.append((x["id"], 1, f"{x['code']}  ·  " + ", ".join(x["nodes"]),
                      f"{x['kind']} · serial {BOX_SERIAL.get(x['code'], 'TBC')} · TG polarity {'/'.join(pols)} (E2E setting)"))
    if b.split:
        units.append(("SPL-TRAY", 1, "Splitter tray (shelf)", f"{len(b.split) * b.tiles} x 2-way splitters ZN2PD-02183-S+" +
                      (", the hub links' pad stacks on P1 / P2" if b.if_pads else "")))
    if b.coupled:
        # 3U: the DC block (+ pad), the plug and the cable bend stand on each coupler's top CPL port (about 95 mm)
        units.append(("CPL-TRAY", 3, "Coupler tray (1U shelf, 2U clear above)",
                      f"{len(b.coupled) * b.tiles} couplers KRYTAR 1821; DC blocks and pads stand on the CPL ports"))
    atts = list(b.atts.items())
    b.att_shelf = {}
    for k in range(0, len(atts), ATT_PER_SHELF):
        grp = atts[k:k + ATT_PER_SHELF]
        sid = f"ATT-SHELF-{k // ATT_PER_SHELF + 1}"
        for slot, (a, v) in enumerate(grp):
            b.att_shelf[a] = (sid, slot)
        units.append((sid, ATT_SHELF_U, "Attenuators " + " + ".join(f"{a} ({v[1]})" for a, v in grp),
                      f"{ATT_SHELF_U}U space, 1U shelf, {len(grp)} programmable attenuator box{'es' if len(grp) > 1 else ''} "
                      "(4 ch, dial)"))
    units += [("PA-SHELF", 1, f"DC/PoE adaptors PA-01..PA-{b.n_adaptors:02d}" + "".join(f" + {v}" for v in b.inj.values()),
               "Siklu EH-PoE-DC-Adaptor-10" + (" + AC PoE injector" if b.inj else "") + " on a shelf"),
              ("DC-DIST", 1, "DC distribution 48 V, 10 breakers", "Eaton DCD1M"),
              ("DC-RECT", 1, "DC rectifier shelf 48 V", "Mean Well RCP-1UI + 2 x RCP-1000-48")]
    if b.ip50:
        units += [("DC-DIST2", 1, "-48 V fuse panel (IP-50)", "Westell N250110-N-0803, GMT fuses"),
                  ("DC-RECT2", 1, "-48 V rectifier (IP-50)", "RCP-1UI + RCP-1000-48, wired -48 V")]
    for p in b.pdus:
        units.append((p, PDU_U, f"{p} 16 A, {PDU_OUT} x C13, switched", PDU_PART))
    total = sum(u[1] for u in units)
    size = 25 if total <= 25 else 42
    rows, u = [], total
    for k, n, lab, sub in units:
        where, fr, rr, extra = CAT.MOUNT[rack_kind(k)]
        hw = " + ".join(x for x in ((f"{fr} x M6 cage nut + screw at the front posts" if fr else ""),
                                    (f"{rr} x M6 cage nut + screw at the rear posts" if rr else ""), extra) if x)
        rows.append(dict(id=k, top=u, n=n, label=lab, sub=sub, mount=where, hardware=hw or "none (toolless)",
                         nuts=fr + rr))
        u -= n
    return size, rows, total


# ------------------------------------------------------------------------------------------
# exploded parts list (team kit-list item in 'team')
# ------------------------------------------------------------------------------------------
def parts(b):
    d = b.d
    nb = {k: sum(1 for x in d["boxes"] if x["kind"] == k) for k in BOX_KIND}
    nn = len(d["nodes"])
    ndc = len(b.dc_nodes)
    hops = len(d.get("mw_links", {}))
    P = []

    def add(group, team, item, pn, mfr, qty, use, conf, cab=None):
        """cab = {cable part number: cables this line covers}: checked against the schedule by bom_check()."""
        if qty in (0, "0"):
            return
        P.append(dict(line=len(P) + 1, group=group, team=team, item=item, pn=pn, mfr=mfr, qty=qty, use=use, conf=conf,
                      cab=cab or {}))
    add("Radios", "11", "1U Spider box, 2 x N367 node x 2 sectors", "GTS003 (lab label)", "Siklu", nb["GTS003"],
        "4 sector modules FATJ0498, 16 SMA tile ports; " + ", ".join(f"{x['id']} = {x['code']} ({BOX_SERIAL.get(x['code'], '?')})"
                                                                    for x in d["boxes"] if x["kind"] == "GTS003"),
        "on hand; to confirm")
    add("Radios", "11", "1U Spider box, 1 x N367 node x 4 sectors (hub)", "GTST002B (lab label)", "Siklu",
        nb["GTST002B"], "16 SMA tile ports", "on hand; to confirm")
    add("Radios", "11", "1U Spider box, 2 x N367 node x 2 sectors (assumed)", "GTST003 (lab label)", "Siklu",
        nb["GTST003"], "16 SMA tile ports", "on hand; to confirm")
    add("Radios", "1", "'6*4 sector nodes - GTSTU022\\23'", "GTSTU022 / GTSTU023", "Siklu", "?",
        "not identified (TU boxes?); not used here", "to confirm with the lab")
    add("Radios", "-", "N367 node (inside the Spider boxes)", "MultiHaul TG N367 (MH-N367-CCP-PoE-MWB class)", "Siklu",
        nn, "ETH1 power in, SFP+ 10G data; managed in-band; 48 W with PoE-Out off", "datasheet")
    if hops:
        add("Radios", "-", "IP-50CX compact dual-carrier radio (1/10/25GbE ports)", "Ceragon IP-50CX (band TBC)", "Ceragon",
            2 * hops, f"{hops} conducted hops, -48 V; data on a 10GbE port", "on hand (8); band to confirm")
    add("Power", "2", "DC/PoE adaptor, passive, bag of 10 ('DC\\POE for falcon')", "EH-PoE-DC-Adaptor-10", "Siklu",
        f"1 bag ({ndc} wired, {b.n_adaptors - ndc} spare unwired)", "36-57 V DC in, RJ-45 to node ETH1 (power only)",
        "likely")
    add("Power", "5", "DC drawer: rectifier shelf 1U + 2 x 1 kW 48 V modules", "RCP-1UI + 2 x RCP-1000-48", "Mean Well", 1,
        f"48 V bus, {48 * ndc} W for {ndc} nodes; one C14 inlet and cord per module ({RECT_MODULES['DC-RECT']} cords)",
        "likely (shelf) / confirmed (modules)")
    dcin_cb = sorted({v[0] for v in b.dcin.values()})
    shared = [cb for cb in dcin_cb if sum(1 for v in b.dcin.values() if v[0] == cb) > 1]
    add("Power", "5", f"DC drawer: distribution, 10 breakers {BREAKER_A} A or less (18 AWG feeds)",
        f"DCD1M (suffix TBC; breakers {BREAKER_A} A max)", "Eaton", 1,
        f"CB1-CB{ndc} one per adaptor" +
        (f"; CB{dcin_cb[0]}-CB{dcin_cb[-1]} the Spider-box DCIN feeds (box fans / Pi)" +
         "".join(f"; {' and '.join(k for k, v in b.dcin.items() if v[0] == cb)} share CB{cb}" for cb in shared)
         if dcin_cb else "") +
        (f"; CB{b.spare_breakers[0]}-{b.spare_breakers[-1]} spare, OFF" if b.spare_breakers else "") +
        f"; {BREAKER_A} A protects each 18 AWG run (node about 1 A); the 10 AWG bus link relies on the rectifier current "
        "limit only (no main breaker counted on)", "likely")
    if b.inj:
        add("Power", "-", "AC PoE injector 60 W, 10G (powers a node that is switched off at t0)",
            "AX-IN-10G-60W-AC-PoE-xx (plug per region)", "Siklu", len(b.inj),
            ", ".join(b.inj) + " on a switched PDU outlet; 52 W to the node, AC inlet TBC", "likely")
    if hops:
        nf = len(b.fuse)
        add("Power", "5", "-48 V rectifier for the IP-50 radios", "RCP-1UI + RCP-1000-48 wired -48 V", "Mean Well", 1,
            f"{nf} radios x ~58 W (IP-50E figure; recheck for the model); {RECT_MODULES['DC-RECT2']} AC cord", "likely")
        add("Power", "5", "-48 V fuse panel 1U, 2 buses x 5 GMT positions", "N250110-N-0803", "Westell", 1,
            "one fused output per surviving IP-50 radio", "to confirm")
        add("Power", "5", "GMT fuse for the -48 V panel", "GMT 3 A (rating to confirm)", "Bussmann / Littelfuse",
            nf + 2, f"{nf} + 2 spare", "to confirm")
        add("Power", "5", "-48 V bus A to bus B strap, 2 x 10 AWG with ring lugs (cut to length)", "part TBC", "-",
            b.count_pn("bus strap"), "DC-DIST2 bus B (F6-F10) from bus A; not needed if the panel ships strapped",
            "to confirm", cab={"bus strap": b.count_pn("bus strap")})
        add("Power", "-", "IP-50 DC power cable / connector (check what ships with the radio)", "Ceragon (part TBC)",
            "Ceragon", 2 * hops, "one per radio: lands the Belden 8461 run on the radio's DC input", "to confirm")
        if b.psu:
            add("Power", "-", "Isolated 48 V AC/DC supply >= 100 W, output floating, wired -48 V (not earthed)",
                "part TBC", "-", len(b.psu), "one per IP-50 radio switched off at t0, on a switched PDU outlet",
                "to confirm")
            add("Power", "-", "Barrel (DC plug) to screw-terminal adaptor for the supply output", "part TBC", "-",
                len(b.psu), "lands the Belden 8461 run on each switched supply (PSU - to the radio -48 V pin)",
                "to confirm")
    add("Power", "7", f"Rack PDU 1U, 16 A, {PDU_OUT} x C13, switched per outlet", PDU_PART.split(" (")[0], "Raritan (or equal)",
        len(b.pdus), "IEC 60309 16 A cord fitted (one wall socket each); rear posts" +
        (f"; {', '.join(f'{b.outlet[x][0]} O{b.outlet[x][1]}' for x in b.switched)} switched off at t0" if b.switched else ""),
        "to confirm (user: 1U PDU)", cab={"PDU cord": b.count_pn("PDU cord")})
    add("Power", "-", f"Bench-table mains strip, {STRIP_OUT} x C13 outlets, C20 inlet 16 A, on/off switch",
        "lab-supplied (part TBC)", "-", 1, f"{len(b.table_loads)} table loads; fed from its own wall socket", "to confirm")
    add("Power", "-", "Floor cable protector, about 1.5 m", "part TBC", "-", 1,
        "covers the strip cord" + (" and the table earth bond" if b.ip50 else "") + " where they cross the floor from "
        "behind the rack to the table leg", "to confirm")
    add("Rack", "-", "Rack-to-table cable bridge: wire-basket cable tray section about 300 mm, 150 mm wide, with a "
        "divider and 2 brackets", "part TBC", "-", 1, "CABLE-BRIDGE between the rack's right post and the table edge, "
        "floor at table height: data in the front compartment, power in the rear one, 50 mm or more apart", "to confirm")
    # leads and patch cords are counted per part number: the twin picks each length from its routed path
    lad = {pn: (L, desc, conf) for k, v in LADDER.items() for pn, L, desc, conf in v}
    use = {"host": "traffic hosts", "agent": "agent-PC bricks", "inj": "PoE injector (inlet TBC)", "srv": "servers",
           "psu": "switched supplies", "dc": "rectifier modules"}
    for pre, team, nm in ((("P014", "C14-C5"), "8", "'Micky' lead C14 -> C5"), (("P004",), "9", "'Exten' lead C14 -> C13")):
        for pn in sorted({c["pn"] for c in b.cables if c["pn"].startswith(pre)}, key=lambda x: lad.get(x, (0,))[0]):
            L, desc, conf = lad.get(pn, (0, pn, "confirmed"))
            who = sorted({use.get(c["b"].split("-")[0].lower(), "rack and table devices") for c in b.cables if c["pn"] == pn})
            add("Power", team, f"{nm} {L / 1000:g} m", pn, "Eaton Tripp Lite", b.count_pn(pn), ", ".join(who), conf,
                cab={pn: b.count_pn(pn)})
    for pn in sorted({c["pn"] for c in b.cables if c["pn"].startswith("CORD-60309")}):
        L, desc, conf = lad.get(pn, (0, pn, "to confirm"))
        add("Power", "-", f"Power cord IEC 60309 16 A plug -> C19 {L / 1000:g} m", pn + " (part TBC)", "-", b.count_pn(pn),
            "TABLE-STRIP supply from its own wall socket", conf, cab={pn: b.count_pn(pn)})
    runs = [c for c in b.cables if c["pn"] == "Belden 8461"]
    m = sum(c.get("metres", 0) for c in runs)
    spools = math.ceil(m * (1 + DC_WASTE) / BELDEN_SPOOL_M)
    singles = 4 * len(runs) - 2 * len(shared)          # a shared breaker takes both supply conductors in one twin ferrule
    add("Power", "10", "DC cable 2 x 18 AWG (bulk, 100 ft spools) + ferrules", "Belden 8461 + Phoenix 3200519", "Belden / Phoenix",
        f"{spools} x 100 ft spool + {singles + 8} ferrules",
        f"{len(runs)} runs ({sum(1 for c in runs if c.get('dcin'))} of them to the Spider-box DCIN), cut lengths {m:g} m + "
        f"{DC_WASTE:.0%} stripping and waste = {m * (1 + DC_WASTE):.1f} m; ferrules: 4 per run" +
        (f" less 2 per shared breaker ({len(shared)})" if shared else "") + " + 8 spare", "confirmed",
        cab={"Belden 8461": len(runs)})
    nbox = len(b.d["boxes"])
    add("Power", "-", "4-pole pluggable screw-terminal plug with screw flange, mates the Spider box's green DC header",
        "Phoenix MSTB 2.5/4-STF-5.08 class (pitch 5.08 / 7.62 mm TBC)", "Phoenix Contact (or equal)", f"{nbox} + 1 spare",
        "one per Spider box (rear photo: the headers are bare); skip if the lab has the plugs", "to confirm")
    if shared:
        add("Power", "10", "Twin ferrule for two 18 AWG conductors in one breaker output terminal",
            "Phoenix AI-TWIN 2 x 1.0 mm2 class (part TBC)", "Phoenix Contact", len(shared) + 2,
            f"{len(shared)} shared DCIN breaker{'s' if len(shared) > 1 else ''} (" + ", ".join(f"CB{cb}" for cb in shared) +
            ") + 2 spare; confirm the DCD1M terminal takes it", "to confirm")
    mb = sum(c.get("metres", 0) for c in b.cables if c["pn"] == "bus link")
    add("Power", "10", "DC bus link 2 x 10 AWG with ring lugs (cut to length)", "part TBC", "-", b.count_pn("bus link"),
        f"rectifier to distribution, {mb:g} m in all", "to confirm", cab={"bus link": b.count_pn("bus link")})
    pe = [c for c in b.cables if c["pn"] == "PE 6 mm2"]
    mp = sum(c.get("metres", 0) for c in pe)
    add("Power", "-", "Green/yellow earth cable 6 mm2 (10 AWG), cut to length", "part TBC", "-",
        f"{math.ceil(mp * (1 + DC_WASTE))} m", f"{len(pe)} bonds (" + "; ".join(
            f"{c['a'].split('.')[0]} to {c['b'].split('.')[0]}" for c in pe) +
        f"): {mp:g} m (lab bar run allowed at 5 m) + 10 %", "to confirm", cab={"PE 6 mm2": len(pe)})
    pe2 = [c for c in b.cables if c["pn"] == "PE 2.5 mm2"]
    if pe2:
        m2 = sum(c.get("metres", 0) for c in pe2)
        add("Power", "-", "Green/yellow earth cable 2.5 mm2 (14 AWG), cut to length", "part TBC", "-",
            f"{math.ceil(m2 * (1 + DC_WASTE))} m", f"{len(pe2)} radio chassis bonds to TABLE-EARTH: {m2:g} m + 10 %",
            "to confirm", cab={"PE 2.5 mm2": len(pe2)})
        add("Power", "-", "Ring lug 2.5 mm2 (insulated) for the radio ground screws (screw size TBC) and the table bar",
            "part TBC", "-", math.ceil(2 * len(pe2) * 1.1), f"2 per radio bond x {len(pe2)} + 10 %", "to confirm")
        add("Power", "-", f"Bench-table earth bar, {len(pe2) + 1} M6 studs, screwed to the table's back edge",
            "part TBC", "-", 1, "TABLE-EARTH: one stud per IP-50 radio + one for the bond to the rack earth bar",
            "to confirm")
    lugs = 4 * (b.count_pn("bus link") + b.count_pn("bus strap")) + 2 * len(pe)
    add("Power", "-", "Ring lug 6 mm2 (10 AWG), insulated, M6 hole (stud size TBC)", "part TBC", "-",
        math.ceil(lugs * 1.1), f"bus links and strap 4 each (2 conductors x 2 ends), earth bonds 2 each ({len(pe)}): "
        f"{lugs} + 10 %; the rack end of the lab-bar bond is a 2-hole lug from the grounding kit", "to confirm")
    add("Power", "-", "Rack grounding kit: earth bar with studs, 2-hole and ring lugs, paint-piercing (star) washers, "
        "bonding screws", "part TBC (rack vendor kit)", "-", 1,
        "bonds the rack frame to the lab earth bar" + (", the -48 V return (positive ground) and the bench-table earth bar"
                                                       if b.ip50 else ""), "to confirm")
    add("Network", "6", "Data switch, 24 x SFP+ + 4 x SFP28, L3", "M4350-24F4V (XSM4328FV)", "Netgear", 1,
        f"{len(b.sw_port)} of 28 ports; 1G only on P1-16", "confirmed")
    add("Network", "6", "Management switch, 24 x 1G PoE+ + 2 x 1G + 4 x SFP+", b.oob_model[0], "Netgear", 1,
        f"{len(b.oob_port)} ports: {min(len(b.oob_port), OOB_RJ45)} of {OOB_RJ45} RJ-45" +
        (f" + {b.n_agm_oob} SFP+ cage" if b.n_agm_oob else "") + "; rear posts", "confirmed")
    add("Network", "-", "1000BASE-T copper SFP", "AGM734-10000S", "Netgear", b.n_agm_data + b.n_agm_oob,
        f"{b.n_agm_data} on SW-DATA P1-16" + (f", {b.n_agm_oob} on SW-OOB" if b.n_agm_oob else ""),
        "confirmed; switch compatibility to confirm")
    fib = [c for c in b.cables if c["pn"].startswith("FZ2ERLNLNSNM")]
    nd = sum(1 for c in fib if c["a"].endswith(".SFP+"))
    nh = len(fib) - nd
    add("Network", "-", "SFP+ SR optics: node side / host side / switch side", "AX-SFP+10G-MM / host NIC optic / AXM761-10000S",
        "Siklu / - / Netgear", f"{nd} / {nh} / {len(fib)}", "both ends of every OM4 run", "likely / to confirm / confirmed")
    for pre, nm in (("FZ2ERLNLNSNM", "OM4 LC-LC duplex"), ("AXC76", "SFP+ DAC"), ("STP6X", "Cat6A S/FTP patch")):
        for pn in sorted({c["pn"] for c in b.cables if c["pn"].startswith(pre)}, key=lambda x: lad.get(x, (0,))[0]):
            L, desc, conf = lad.get(pn, (0, pn, "confirmed"))
            use_ = ", ".join(f"{n} {k}" for k, n in Counter(CABLE_USE.get(c["cls"], c["cls"]) for c in b.cables
                                                            if c["pn"] == pn).items())
            add("Network", "-", f"{nm} {L / 1000:g} m", pn, desc.split()[0], b.count_pn(pn), use_, conf,
                cab={pn: b.count_pn(pn)})
    na = len(b.atts)
    reuse = d.get("att_reuse", 0)
    add("RF/IF", "-", "Programmable attenuator box, 4 ch, 0-63 dB, to 18 GHz (small box with a dial; e.g. ZTDAT-4-18G63S class)",
        "ZTDAT-4-18G63S class (model TBC)", "Mini-Circuits (or equal)", na, f"{math.ceil(na / ATT_PER_SHELF)} x {ATT_SHELF_U}U shelves",
        f"reuse {reuse} from PoC 1, buy {na - reuse}" if reuse else f"to buy: {na}")
    if b.split:
        add("RF/IF", "-", "2-way splitter 2-18 GHz", "ZN2PD-02183-S+", "Mini-Circuits", len(b.split) * b.tiles,
            "one per tile of each shared hub module", "confirmed")
    if b.if_pads:
        # fixed pads per value (pad schedule: bench_spec.pad_stack), ~10 % spare per value, at least 1
        cnt = Counter(p["pn"] for v in b.pads.values() if v["chain"] == "if" for p in v["parts"])
        # port-check trim pads: one link's four tile paths per value, fitted at the B tile if a link reads too strong
        trim = {f"BW-S{v}W2+": b.tiles for v in TRIM_PADS}
        for pn in sorted(set(cnt) | set(trim), key=lambda x: int(re.sub(r"\D", "", x.split("W2")[0]))):
            q = cnt.get(pn, 0)
            links = [k for k, v in b.pad_links.items() if pn in v["pn"]]
            per = sorted({v["pn"].count(pn) for k, v in b.pad_links.items() if k in links})
            sp = max(1, math.ceil(q * PAD_SPARE)) if q else 0
            tq = trim.get(pn, 0)
            qty = " + ".join(x for x in (f"{q}" if q else "", f"{sp} spare" if sp else "", f"{tq} trim" if tq else "") if x)
            use_ = (f"{' or '.join(map(str, per)) if per != [1] else 'one'} per tile path of {', '.join(links)} "
                    f"({len(links)} link{'s' if len(links) > 1 else ''} x {b.tiles} tiles; pad schedule: guide 'Pad schedule' "
                    f"page and blueprint sheet 6); {sp} spare" if q else "")
            if tq:
                use_ += ("; " if use_ else "") + (f"{tq} port-check trim pads (one link's {b.tiles} tile paths: fitted at the "
                                                  "B tile if the link reads too strong; guide 'Port check')")
            add("RF/IF", "-", f"Fixed attenuator (pad) DC-18 GHz 2 W, SMA M-F, {pn[4:-3]} dB", pn, "Mini-Circuits",
                qty, use_, "likely (value from the A7 model; confirm at the port check)")
        if b.split:
            # the P1 / P2 stacks of one splitter lie about 2 mm apart (estimate): P2's through an adapter if they foul
            ns = len(b.split) * b.tiles
            add("RF/IF", "-", "SMA male to SMA female adapter DC-18 GHz (short, straight)", "SM-SF50+", "Mini-Circuits",
                f"{ns} if needed + 1 spare", "P2 stacks only, if the two stacks of a splitter foul: P2's stack screws on "
                "through the adapter so its nuts stand beyond P1's (splitter-tray step); order once a pad is measured",
                "if needed")
    if b.coupled:
        add("RF/IF", "-", "10 dB directional coupler 1-18 GHz, SMA-F (one part for all 16 positions)", "KRYTAR 1821", "KRYTAR",
            len(b.coupled) * b.tiles, "IN from the module tile, THRU to the link attenuator, CPL to the arm "
            "(tray laid out for its size, about 90 mm: do not substitute a longer coupler; Mini-Circuits ZUDC10-183+, "
            "152 mm, named in older notes, does NOT fit CPL-TRAY)", "confirmed")
        add("RF/IF", "-", "DC block 18 GHz", "BLK-18-S+", "Mini-Circuits", d.get("arm_dc_blocks", 1) * len(d["couplings"]) * b.tiles,
            "both ends of each arm" if d.get("arm_dc_blocks", 1) == 2 else "arm input side, at the coupler", "confirmed")
        for pn in sorted({v["pn"] for v in b.arm_pad.values()}):
            arms = [k for k, v in b.arm_pad.items() if v["pn"] == pn]
            db = b.arm_pad[arms[0]]["db"]
            add("RF/IF", "-", f"Fixed pad DC-18 GHz 2 W, {db} dB", pn, "Mini-Circuits", len(arms) * b.tiles,
                f"arm input side, on the DC block at the coupler: arms {', '.join(arms)} (value from the arm budget; "
                f"confirm after the port check; this is the only arm pad: older notes naming BW-S10W2+ are superseded)",
                "value to confirm")
    for pn, L in RF_STOCK:
        q = b.count_pn(pn)
        tag = " ('RF cable': 10 on hand, type TBC)" if pn == "CBL-2FT-SMSM+" else ""
        wg_only = b.mw_pads and q and all(c["cls"] == "WG" for c in b.cables if c["pn"] == pn)
        add("RF/IF", "3", f"SMA test cable 18 GHz {L / 1000:.2f} m{tag}", pn, "Mini-Circuits", q,
            "IP-50 chains (the one SMA cable between a hop's pad stacks), length from the twin route" if wg_only else
            "IF chain, length from the twin route" + ("; IP-50 chains" if any(c["pn"] == pn and c["cls"] == "WG"
                                                                                for c in b.cables) else ""),
            "likely (catalogue length to confirm)" if pn in RF_LIKELY else "confirmed", cab={pn: q})
    used_tiles = {e for c in b.cables if c["cls"] == "RF" for e in (c["a"], c["b"]) if re.match(r"^SB\d+\.M\d+\.T\d+$", e)}
    used_tiles |= {h for h in b.pad_host if re.match(r"^SB\d+\.M\d+\.T\d+$", h)}      # a pad stack on the tile
    open_tiles = 16 * len(d["boxes"]) - len(used_tiles)
    open_gnss = 8 - len(d["nodes"])
    nl = open_tiles + max(0, open_gnss)
    add("RF/IF", "-", "50-ohm SMA load", "ANNE-50+", "Mini-Circuits", f"{nl} + 2 spare",
        ", ".join(x for x in ((f"{open_tiles} unused Spider tile ports" if open_tiles else ""),
                              (", ".join(f"GNSS-SPL.out{i}" for i in range(len(d["nodes"]) + 1, 9)) if open_gnss > 0 else ""))
                  if x) or "spares only", "confirmed")
    if hops:
        add("IP-50 chain", "-", "Waveguide-to-coax transition, WR-62 12.4-18 GHz (<=18 GHz set)", "PEWCA1003", "Pasternack",
            2 * hops, "band and flange mating TBC", "likely", cab={"PEWCA1003": b.count_pn("PEWCA1003")})
        add("IP-50 chain", "-", "30 dB 10 W pad DC-18 GHz (<=18 GHz set)", "SA18S10W-30", "Fairview",
            f"{2 * hops} + {max(1, math.ceil(2 * hops * PAD_SPARE))} spare" if b.mw_pads else 2 * hops,
            "one at each radio port, on the transition" +
            (f"; {max(1, math.ceil(2 * hops * PAD_SPARE))} spare (pad schedule: guide 'Pad schedule' page and blueprint "
             "sheet 6)" if b.mw_pads else ""), "confirmed")
        if b.mwatt:
            add("IP-50 chain", "-", "Programmable attenuator 4 ch DC-18 GHz (<=18 GHz set)", "J7204B opt 002", "Keysight",
                1, f"{hops} channels used; LAN to SW-OOB", "confirmed")
        if b.mw_pads:
            add("IP-50 chain", "-", f"Fixed attenuator (pad) DC-18 GHz 2 W, SMA M-F, {MW_PAD_LO[1]} dB (<=18 GHz set)",
                MW_PAD_LO[0], "Mini-Circuits", f"{hops} + {max(1, math.ceil(hops * PAD_SPARE))} spare",
                f"A side of each hop, on the {MW_PAD_HI[0]} (after the 30 dB pad, so it sees 2 W or less): "
                f"{2 * MW_PAD_HI[1] + MW_PAD_LO[1]} dB fixed per chain, no programmable attenuator", "confirmed")
        nst = len(b.stacked_hops)                          # hops whose two radios sit on spacer bars
        if nst:
            add("IP-50 chain", "-", f"Spacer bar 25 x 25 x 300 mm (aluminium) between stacked radios", "part TBC", "-",
                2 * 2 * nst, f"2 per stacked radio ({2 * nst} stacked): a {SPACER_MM:g} mm air gap over the lower "
                "radio's lid", "to confirm")
            add("IP-50 chain", "-", "Double-sided foam mounting tape, 25 mm wide, 1 roll", "part TBC", "-", 1,
                "fixes each spacer bar to the lower lid and the upper radio to the bars so it cannot slide", "to confirm")
        if b.mw_pads:
            add("IP-50 chain", "-", "Alternative set if the band is 18-26.5 GHz, e.g. 23 GHz (replaces the 3 lines above)",
                f"PEWCA1018 + FMAT4009-30 + {MW_PAD_LO_K}", "Pasternack / Fairview / -", "replaces the set",
                "2.92 mm pads and a 2.92 mm cable; the 10 dB pad becomes a 2.92 mm DC-40 GHz part", "alternative")
        else:
            add("IP-50 chain", "-", "Alternative set if the band is 18-26.5 GHz (replaces the 3 lines above)",
                "PEWCA1018 + FMAT4009-30 + 84906K/84904K/11713C", "Pasternack / Fairview / Keysight", "replaces the set",
                "2.92 mm pads and cables", "alternative")
    add("Timing", "-", "GNSS antenna + 1x8 splitter", "101898-00 + RMALDCBS1X8", "Protempis / GPS Networking", "1 + 1",
        "only if the boxes take GNSS (N367 GPS is internal)", "to confirm")
    add("Timing", "-", "GNSS antenna lead, roof to the room entry (N connectors)", "LMR-400 (part TBC)", "-",
        b.count_pn("LMR-400"), "only if the boxes take GNSS; length to the site", "to confirm",
        cab={"LMR-400": b.count_pn("LMR-400")})
    add("Timing", "-", "GNSS surge arrestor, DC pass (the splitter powers the antenna through it), N-female both ends, "
        "ground lug", "part TBC (PolyPhaser / Times Microwave GPS class)", "-", 1,
        "GNSS-ARR where the lead enters the room, its ground lug bonded to the lab earth bar; only if the boxes take "
        "GNSS", "to confirm")
    for pn in sorted({c["pn"] for c in b.cables if c["pn"].startswith("LMR-240")}):
        add("Timing", "-", f"GNSS arrestor-to-splitter jumper, {pn} (length from the twin route)", pn + " (part TBC)", "-",
            b.count_pn(pn), "GNSS-ARR to GNSS-SPL.in; only if the boxes take GNSS", "to confirm", cab={pn: b.count_pn(pn)})
    for pn in sorted({c["pn"] for c in b.cables if c["pn"].startswith("RG-316")}):
        add("Timing", "-", f"GNSS splitter jumper, {pn} (length from the twin route)", pn + " (part TBC)", "-",
            b.count_pn(pn), "splitter output to node GNSS SMA; only if the boxes take GNSS", "to confirm",
            cab={pn: b.count_pn(pn)})
    if d.get("agents"):
        add("Compute", "-", "MARL agent PC (mini PC, 2 x 1G NICs)", "lab PC", "-", len(d["agents"]), "", "on hand")
    add("Compute", "-", "Traffic host, 1U rack server (10G SFP+ NIC, iperf3/TRex)", "lab server", "-", len(d["hosts"]),
        "in the rack, with its rails", "on hand; to confirm")
    add("Compute", "-", "Rack servers: " + "; ".join(f"{x} ({SERVER_U.get(x, 1)}U)" for x in b.rack_servers), "lab servers", "-",
        len(b.rack_servers), "in the rack, with their rails", "on hand; to confirm")
    add("Compute", "-", "Bench controller PC + monitor + keyboard/mouse (operator station on the table)", "lab PC", "-",
        "1 + 1 + 1", "stays powered through the event; drives PDUs" + (" and attenuators" if b.atts or b.mwatt else "") +
        " over SW-OOB", "on hand; to confirm",
        cab={"keyboard + mouse": b.count_pn("keyboard + mouse")})
    add("Compute", "-", "Video cable (DisplayPort / HDMI) 1.8 m", "part TBC", "-", b.count_pn("video cable"), "PC to monitor",
        "to confirm", cab={"video cable": b.count_pn("video cable")})
    add("Rack", "4", f"19-inch 4-post rack ({b.total}U used of {b.size}U)",
        "StarTech 4POSTRACK25U" if b.size == 25 else "StarTech 4POSTRACK42", "StarTech", 1,
        "square-hole rails (M6 cage nuts) to confirm", "confirmed")
    shelves = [r for r in b.rows if rack_kind(r["id"]) in ("shelf", "tray", "attshelf")]
    add("Rack", "4", "1U vented shelf (2-post, 254 mm deep)", "CABSHELFV1U", "StarTech", len(shelves),
        ", ".join(r["id"] for r in shelves), "likely")
    empty = b.size - b.total
    add("Rack", "-", "Cable manager / blanking panels (pack of 10)", "NCMHF1 / AR8136BLK", "Panduit / APC",
        f"1 / {math.ceil(empty / 10)}", f"{empty} empty U", "confirmed")
    # mounting hardware, counted from the rack layout (bench_spec.rack_layout: 'mount', 'hardware', 'nuts' per row)
    nlace = math.ceil(b.total / LACE_EVERY_U)
    nuts = sum(r["nuts"] for r in b.rows) + 2 * nlace
    nuts_buy = math.ceil(nuts * 1.1)
    add("Rack", "-", "M6 cage nut + M6 x 16 screw + washer sets (pack of 50)", "CABSCREWM6 (or equal)", "StarTech",
        f"{math.ceil(nuts_buy / CAT.CAGE_PACK)} packs",
        f"{nuts} fitted (4 per item per post pair: " + ", ".join(f"{k} {v}" for k, v in Counter(
            rack_kind(r['id']) for r in b.rows if r['nuts']).items()) + f"; {nlace} lacing bars x 2) + 10 % spare",
        "likely")
    railed = [r["id"] for r in b.rows if "UNIRAILS1U" in r["hardware"]]
    add("Rack", "-", "1U adjustable rails (420-1054 mm), front and rear posts, own M5 cage nuts and screws in the kit",
        "UNIRAILS1U", "StarTech", f"{len(railed)} pairs",
        "support for the deep / heavy 1U units: " + ", ".join(railed), "likely")
    add("Rack", "-", "Cable lacing bar, 19-inch, rear posts", "part TBC", "-", nlace,
        f"one per {LACE_EVERY_U} U at the rear: power dressed on the left (seen from the front), data on the right", "to confirm")
    # consumables, counted from the cable schedule and the twin routes (bench cables carry 'route_mm' once the twin ran)
    route = sum(c.get("route_mm", 1500) for c in b.cables)
    straps = math.ceil(route / STRAP_EVERY_MM / STRAP_SHARE / 50) * 50
    add("Consumables", "-", "Hook-and-loop cable straps 20 x 200 mm (no zip ties on coax or fibre)", "VELCRO ONE-WRAP (or equal)",
        "-", straps, f"one per {STRAP_EVERY_MM} mm of bundle, about {STRAP_SHARE} cables per strap "
        f"({route / 1000:.0f} m of routed cable)", "to confirm")
    add("Consumables", "-", "Cable ID labels, self-laminating (2 per cable) + label cartridge", "Brady M21-750-427",
        "Brady", f"{2 * len(b.cables)} labels / {math.ceil(2 * len(b.cables) / LABELS_PER_CART)} cartridges",
        f"{len(b.cables)} cables x 2 ends; {LABELS_PER_CART} labels per cartridge assumed", "confirmed (cartridge) / to confirm (count)")
    add("Consumables", "-", "Fibre end-face cleaner, LC one-click", "part TBC", "-", 1, "clean every LC before mating",
        "to confirm")
    ndc = sum(1 for c in b.cables if c["cls"] == "DC")
    add("Consumables", "-", "Heat-shrink conductor markers, red (+48 V) and blue (-48 V)", "part TBC", "-",
        f"{2 * ndc + 10} pieces", "supply conductor of every D run, both ends (the unmarked conductor is the 0 V return)",
        "to confirm")
    add("Consumables", "-", "PVC insulating tape, 19 mm", "part TBC", "-", "1 roll",
        "tapes the free load ends of the DC runs until the polarity check", "to confirm")
    npoe = b.count("POE")
    add("Consumables", "-", "Yellow flag labels '48 V PoE - NOT ETHERNET'", "part TBC", "-", 2 * npoe + 4,
        f"both ends of the {npoe} Q cables (they carry 48 V on an ordinary RJ45) + 4 spare", "to confirm")
    # tools
    for item, pn, mfr, qty, use_ in (
            ("Cage-nut insertion tool", "part TBC", "-", 2, "fitting the M6 cage nuts"),
            ("ESD wrist strap", "part TBC", "-", 2, "bonded to the rack earth while handling the Spider boxes and IF parts"),
            ("SMA torque wrench 8 in-lb (0.9 N m), 5/16 in", "ST-SMA3", "Fairview", 2, "every SMA joint"),
            ("SMA 5/16 in open-end backing (hold) wrench", "part TBC", "-", 2, "holds the jack while torquing"),
            ("Torque screwdriver 0.5-5 N m", "part TBC", "-", 1,
             "DC breaker, fuse, adaptor and Spider DCIN terminals (value marked on each)"),
            ("Torque wrench 2-25 N m with 10 and 13 mm sockets", "part TBC", "-", 1,
             "ring lugs on the bus links, the strap and the earth studs (M6 / M8, value marked on each stud)"),
            ("Heat gun", "part TBC", "-", 1, "heat-shrink conductor markers on the DC runs"),
            ("GMT fuse puller", "part TBC", "-", 1, "inserting and pulling the DC-DIST2 GMT fuses"),
            ("Ferrule crimper 0.25-6 mm2 (square crimp)", "CRIMPFOX 6 (1212034)", "Phoenix Contact", 1,
             "18 AWG ferrules on the DC runs"),
            ("Insulated ring-lug ratchet crimper, 10 AWG (6 mm2)", "part TBC", "-", 1,
             "ring lugs on the bus links, the bus strap and the earth bonds (or order them pre-made)"),
            ("Wire stripper for 18 and 10 AWG", "part TBC", "-", 1, "DC runs and bus links"),
            ("Digital multimeter, DC V and ohms with lead null (REL), CAT III", "part TBC", "-", 1,
             "DC polarity and voltage at every breaker / fuse output before any load is landed; earth-bond continuity "
             "(under 0.1 ohm) and the earth-referenced bus checks at power-up"),
            ("Label printer for the ID labels", "BMP21-PLUS (or on hand)", "Brady", 1, "cable ID tags")):
        add("Tools", "-", item, pn, mfr, qty, use_, "to confirm" if pn == "part TBC" else "likely")
    add("Test", "-", "18 GHz power sensor / spectrum analyser", "NRP18S / FPL1026", "R&S", "1 / 1",
        "port checks, calibration", "confirmed")
    add("Test", "-", "N-F to SMA-F adapter for the power sensor", "NF-SF50+", "Mini-Circuits", 1, "NRP18S input is N-male",
        "confirmed")
    return P


def bom_check(b, P):
    """Mismatches between the parts list and the final cable schedule: for every cable part number, the cables the
    parts list orders must equal the cables of that part in the schedule."""
    want = Counter(c["pn"] for c in b.cables if c["pn"] not in BOM_SKIP)
    have = Counter()
    for p in P:
        have.update(p.get("cab") or {})
    return [f"{pn}: schedule {want[pn]}, parts list {have[pn]}" for pn in sorted(set(want) | set(have)) if want[pn] != have[pn]]


def end_text(ep):
    """a cable end as printed in the schedules: the port ID, with what it is for the lab's own fixtures."""
    return f"{ep} ({WALL_SOCKET})" if ep.startswith("WALL.") else ep


def write_tables(b, P, prefix):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{prefix}_cable_schedule.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "class", "from", "to", "part", "length", "vlan", "note"])
        for c in b.cables:
            w.writerow([c["id"], c["cls"], end_text(c["a"]), end_text(c["b"]), c["part"], c["length"], c["vlan"], c["note"]])
    with open(os.path.join(OUT, f"{prefix}_parts_list.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["line", "group", "team list item", "item", "part number", "manufacturer", "qty", "use", "status"])
        for p in P:
            w.writerow([p["line"], p["group"], p["team"], p["item"], p["pn"], p["mfr"], p["qty"], p["use"], p["conf"]])
    names = {"AC": "AC mains", "DC": "DC power", "POE": "Cat6A power to node ETH1", "FIB": "Data to SW-DATA (fibre / DAC / Cat6A)",
             "ETH": "1G management to SW-OOB (VLAN 100)", "RF": "IF coax (SMA, 18 GHz)", "GNSS": "GNSS",
             "WG": "IP-50 conducted chain", "AUX": "Operator station (video, USB)", "PE": "Earth bonding (green/yellow)"}
    L = [f"# {b.d['poc']} cable schedule (generated by blueprint.py)", "", f"{b.d['name']}. {len(b.cables)} cables.", "",
         "Nodes are managed in-band (E2E controller over the radio network); no node ETH2 or IP-50 management port is "
         "cabled. Lengths come from the digital twin's routes (twin/twin_model.py): IF and IP-50 SMA cables are the "
         "shortest stock length covering the bundle's longest route + 5 %, every other cable the shortest ladder "
         "length covering its route + 10 %; DC and earth cables are cut to the route + 10 %. Confirm at a dry fit.", ""]
    for cls, nm in names.items():
        cs = [c for c in b.cables if c["cls"] == cls]
        if cs:
            L += [f"## {nm} ({len(cs)})", "", "| ID | From | To | Part | Length | VLAN | Note |", "|---|---|---|---|---|---|---|"]
            L += [f"| {c['id']} | {end_text(c['a'])} | {end_text(c['b'])} | {c['part']} | {c['length']} | {c['vlan']} | {c['note']} |"
                  for c in cs] + [""]
    L += ["## SW-DATA port plan", "", "| Port | Cage | VLAN | Site | Endpoint | Cable |", "|---|---|---|---|---|---|"]
    for src, (p, v, site, speed) in b.sw_port.items():
        cage = "SFP+ (1/10G)" if p <= DATA_1G else ("SFP+ (10G)" if p <= 24 else "SFP28 (10G mode)")
        L.append(f"| P{p} | {cage} | {v} | {site} | {src} | {b.cable(a=src, cls='FIB')['id']} ({speed}) |")
    L += ["", "## SW-OOB port plan (VLAN 100)", "", "| Port | Cage | Endpoint | Cable |", "|---|---|---|---|"]
    for src, (q, grp) in b.oob_port.items():
        L.append(f"| P{q} | {'RJ-45' if q <= OOB_RJ45 else 'SFP+ + AGM734'} | {src} | {b.cable(a=src, cls='ETH')['id']} |")
    with open(os.path.join(OUT, f"{prefix}_cable_schedule.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    L = [f"# {b.d['poc']} exploded parts list (generated by blueprint.py)", "",
         "Team = item number on the lab's kit list (1 = '6*4 sector nodes - GTSTU022\\23', 2 = '10 DC\\POE for falcon', "
         "3 = '10 RF cable', 4 = RACK, 5 = DC drawer, 6 = Switch\\netgear, 7 = PDU 16\\32A, 8 = Micky cable, "
         "9 = Exten cable, 10 = DC cable, 11 = 1U Spider boxes; '-' = not on the list).", "",
         "| # | Team | Group | Item | Part number | Manufacturer | Qty | Use | Status |", "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {p['line']} | {p['team']} | {p['group']} | {p['item']} | {p['pn']} | {p['mfr']} | {p['qty']} | {p['use']} | "
          f"{p['conf']} |" for p in P]
    with open(os.path.join(OUT, f"{prefix}_parts_list.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
