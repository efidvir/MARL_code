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
  att_set    {link: (job, before, t0, after, margin)}   attenuator settings in dB (optional)
  att_reuse  int                                  attenuators carried over from PoC 1 (optional)
  event      text; event_off [load]; event_breaker [site]   (PoC 1)

Management is in-band: the E2E controller reaches every node through the radio network from its data
NIC, so no node ETH2 (and no IP-50 management port, until it is shown not to bridge to the data port) is
cabled to the management switch.  SW-OOB carries only lab equipment: attenuators, PDUs, the analyser and
the PCs' second NICs.
"""
import csv
import math
import os
from collections import OrderedDict

from validate import bearing, sector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEC = {0: "E", 1: "N", 2: "W", 3: "S"}
BOX_KIND = {"GTS003": (2, 2), "GTST002B": (1, 4), "GTST003": (2, 2)}
CLS_PREFIX = {"AC": "A", "DC": "D", "POE": "P", "FIB": "F", "ETH": "E", "RF": "R", "GNSS": "G", "WG": "W"}

U_MM = 44.45                    # one rack unit
BEND_MM = 100                   # plug bodies and a bend at each end of an IF cable
RF_STOCK = [("CBL-1FT-SMSM+", 305), ("CBL-2FT-SMSM+", 610), ("CBL-3FT-SMSM+", 914), ("CBL-1M-SMSM+", 1000),
            ("CBL-2M-SMSM+", 2000)]
DATA_PORTS, DATA_1G = 28, 16    # M4350-24F4V: P1-24 SFP+ (1G only on P1-16), P25-28 SFP28 (10/25G)
OOB_RJ45, OOB_SFP = 26, 4       # GSM4230PX: 24 PoE+ + 2 RJ-45, 4 SFP+ cages (AGM734 copper SFP if needed)

AGM734 = "Netgear AGM734 1000BASE-T SFP"
OM4 = "SFP+ SR at both ends (Siklu AX-SFP+10G-MM / Netgear AXM761) + Panduit FZ2ERLNLNSNM002 OM4 LC-LC 2 m"
DAC = "Netgear AXC763-10000S SFP+ DAC 3 m"


def mod_x(m, t):
    """Assumed x (mm from the left rail) of tile port t of module M<k> on a Spider box front."""
    return 70 + (int(m[1]) - 1) * 95 + (t - 2.5) * 18


def ch_x(c, side):
    """Assumed x (mm) of an attenuator channel's IN / OUT port."""
    return 140 + (c - 1) * 72 + (-12 if side == "in" else 12)


# ------------------------------------------------------------------------------------------
# designs
# ------------------------------------------------------------------------------------------
def poc1_design():
    """A7 as cabled on the bench (figures/bench_a7.mjs): each node's hub link on one module and its
    bridge on the other, the hub's W and E modules split to two peers, 9 attenuators."""
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
    down = "down at t0"
    att_set = OrderedDict([
        ("HUB-W1", (down, 27.6, 63.0, 63.0, "21.6 / -13.9")), ("HUB-W2", (down, 23.1, 63.0, 63.0, "21.6 / -18.4")),
        ("HUB-E1", (down, 27.6, 63.0, 63.0, "21.6 / -13.9")), ("HUB-E2", (down, 23.1, 63.0, 63.0, "21.6 / -18.4")),
        ("HUB-S", (down, 34.5, 63.0, 63.0, "14.6 / -13.9")), ("HUB-N", (down, 37.2, 63.0, 63.0, "11.9 / -13.9")),
        ("E2-W2", ("bridge", 60.7, 60.7, 30.7, "13.9 / -18.4")), ("E1-N", ("bridge", 63.0, 63.0, 34.1, "15.0 / -13.9")),
        ("W1-S", ("bridge", 63.0, 63.0, 41.2, "8.0 / -13.9"))])
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
        servers=[("SRV-E2E", "TG E2E controller + NMS", "CORE"), ("SRV-DPIR", "DPIR planner + predictor", "CORE")],
        no_data=["HUB"], att_set=att_set,
        mw_set=OrderedDict([("W1-CORE", ("10 + fades", "max")), ("CORE-E1", ("10 + fades", "max")),
                            ("W1-W2", ("10 + fades", "unchanged")), ("E1-E2", ("10 + fades", "unchanged"))]),
        event="t0: ATT-01..06 (every path to the hub) to 63 dB and MWATT ch1-2 (W1-CORE, CORE-E1) to maximum; "
              "then SRV-E2E and SRV-DPIR off at their switched PDU outlets and the hub dark (CB3 opened). "
              "W1-W2 and E1-E2 unchanged. Bridges ATT-07..09 step down when MARL engages their sites.",
        event_off=["SRV-E2E", "SRV-DPIR"], event_breaker=["HUB"],
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
                            ("SRV-E2E", "TG E2E controller + NMS (POP at H)", "H")],
        no_data=[], event=None, att_reuse=9,
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
        self._build()

    def cid(self, cls):
        self.n[cls] = self.n.get(cls, 0) + 1
        return f"{CLS_PREFIX[cls]}{self.n[cls]:03d}"

    def add(self, cls, a, b, part, length, note="", pn=None, vlan="", **kw):
        c = dict(id=self.cid(cls), cls=cls, a=a, b=b, part=part, length=length, note=note, pn=pn or part, vlan=vlan,
                 **kw)
        self.cables.append(c)
        return c

    def vlan(self, site):
        return 101 + self.d["sites"].index(site)

    def count(self, cls, part_sub=None):
        return sum(1 for c in self.cables if c["cls"] == cls and (part_sub is None or part_sub in c["part"]))

    def count_pn(self, pn):
        return sum(1 for c in self.cables if c["pn"] == pn)

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
        for name in order:
            self.atts[f"ATT-{len(self.atts) + 1:02d}"] = ("link" if name in d["link_ends"] else "arm", name)
        self.att_of = {v[1]: k for k, v in self.atts.items()}
        self.adaptor = {n: f"PA-{i + 1:02d}" for i, n in enumerate(nodes)}
        self.n_adaptors = max(10, len(nodes))
        self.ip50 = OrderedDict()
        for k, (a, b) in d.get("mw_links", {}).items():
            self.ip50[f"IP50-{k}@{a}"] = (k, a)
            self.ip50[f"IP50-{k}@{b}"] = (k, b)
        off = d.get("event_off", [])
        # AC -----------------------------------------------------------------------------------
        rack_loads = ["DC-RECT", "SW-DATA", "SW-OOB", "GNSS-SPL"] + list(self.atts)
        if self.ip50:
            rack_loads.insert(1, "DC-RECT2")
        rack_loads += [f"AGENT-{s}" for s in d.get("agents", [])]
        rack_loads += off                                   # event loads go on switched rack-PDU outlets
        self.pdus = [f"PDU-{'AB'[i]}" for i in range(math.ceil(len(rack_loads) / 16))]
        for p in self.pdus:
            self.add("AC", "Lab wall socket IEC 60309 32 A", f"{p}.inlet", "PDU cord (IEC 60309 32 A plug)", "fixed",
                     pn="AP7922B cord")
        self.outlet = OrderedDict()
        for i, ld in enumerate(rack_loads):
            pdu, o = self.pdus[i // 16], i % 16 + 1
            self.outlet[ld] = (pdu, o)
            if ld.startswith("AGENT"):
                self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", "Eaton Tripp Lite P014-006 C14-C5 'Mickey' lead", "1.8 m",
                         "agent PC power brick", pn="P014-006")
            elif ld in off:
                self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", "Eaton Tripp Lite P004-010 C14-C13 lead", "3.05 m",
                         "switched off at t0 (outlet switching); runs to the bench table", pn="P004-010")
            else:
                self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", "Eaton Tripp Lite P004-006 C14-C13 lead", "1.8 m",
                         "ZTDAT inlet type TBC" if ld.startswith("ATT") else "", pn="P004-006")
        self.table_loads = [s[0] for s in d["servers"] if s[0] not in off] + ["SPECAN"] + \
            (["MWATT"] if self.ip50 else []) + [f"HOST-{h}" for h in d["hosts"]]
        for t in self.table_loads:
            if t.startswith("HOST"):
                self.add("AC", "TABLE-STRIP", f"{t}.ac", "Eaton Tripp Lite P014-006 C14-C5 'Mickey' lead", "1.8 m",
                         pn="P014-006")
            else:
                self.add("AC", "TABLE-STRIP", f"{t}.ac", "Eaton Tripp Lite P004-006 C14-C13 lead", "1.8 m",
                         "inlet type TBC" if t == "MWATT" else "", pn="P004-006")
        # DC -----------------------------------------------------------------------------------
        self.add("DC", "DC-RECT.bus", "DC-DIST.bus", "DC bus link 2 x 10 AWG with ring lugs", "0.3 m", "48 V bus",
                 pn="bus link")
        brk = d.get("event_breaker", [])
        for i, nd in enumerate(nodes):
            self.add("DC", f"DC-DIST.CB{i + 1}", f"PA-{i + 1:02d}.dc", "Belden 8461 2 x 18 AWG + Phoenix 3200519 ferrules",
                     "1 m", f"feeds node {nd}" + ("; CB opened after t0 (node dark)" if nd in brk else ""),
                     pn="Belden 8461", metres=1.0)
        self.spare_breakers = list(range(len(nodes) + 1, 11))
        if self.ip50:
            self.add("DC", "DC-RECT2.out", "DC-DIST2.bus", "DC bus link 2 x 10 AWG with ring lugs (A and B buses)",
                     "0.3 m", "-48 V bus, positive grounded", pn="bus link")
            for i, k in enumerate(self.ip50):
                self.add("DC", f"DC-DIST2.F{i + 1}", f"{k}.dc", "Belden 8461 2 x 18 AWG (IP-50 connector TBC)", "3 m",
                         "-48 V through a GMT fuse", pn="Belden 8461", metres=3.0)
        for n in nodes:
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
                    eps.append((site, "1G", f"{k}.data", f"Panduit STP6X3MBU Cat6A 3 m + {AGM734} at the switch",
                                "STP6X3MBU", "3 m", f"hop {hop} (IP-50 data port type TBC)", v))
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
            oob.append((f"{sid}.mgmt", "server management", "STP6X3MBU", "3 m", st or "table"))
        for h, sts in d["hosts"].items():
            if len(sts) > 1:
                oob.append((f"HOST-{h}.mgmt", "traffic host control", "STP6X3MBU", "3 m", "trunk"))
        for a in self.atts:
            oob.append((f"{a}.lan", "attenuator control, SCPI over TCP", "STP6X2MBU", "2 m", "rack"))
        if self.ip50:
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
        self.add("GNSS", "GNSS-ANT", "GNSS-SPL.in", "LMR-400 coax + surge arrestor (part TBC)", "site", pn="LMR-400")
        for i, bx in enumerate(d["boxes"]):
            self.add("GNSS", f"GNSS-SPL.out{i + 1}", f"{bx['id']}.GNSS", "RG-316 SMA coax 1 m (part TBC)", "1 m",
                     "only if the box has a GNSS input (TBC)", pn="RG-316")
        # RF / IF: one bundle (4 tiles) at a time so that each bundle has consecutive IDs --------------
        tray = "SPL-TRAY" if self.split else "CPL-TRAY"
        for s, sid in self.split.items():
            box, mod = d["module"][s]
            for t in T:
                x = mod_x(mod, t)
                self.add("RF", f"{box}.{mod}.T{t}", f"{sid}.t{t}.S", "{pn}", "", f"{s[0]}.{s[1]} tile {t} to its splitter",
                         geo=(box, x, tray, x), bundle=("spl", s))
        for name, (sa, sb) in d["link_ends"].items():
            att = self.att_of[name]
            for side, s in (("in", sa), ("out", sb)):
                box, mod = d["module"][s]
                if s in self.split:
                    k = d["splitters"][s].index(name) + 1
                    for t in T:
                        self.add("RF", f"{self.split[s]}.t{t}.P{k}", f"{att}.ch{t}.{side}", "{pn}", "", f"{name} tile {t}",
                                 geo=(tray, mod_x(mod, t), att, ch_x(t, side)), bundle=(name, side))
                elif s in self.coupled:
                    for t in T:
                        x = mod_x(mod, t)
                        self.add("RF", f"{box}.{mod}.T{t}", f"{self.coupled[s]}.t{t}.IN", "{pn}", "",
                                 f"{name} {s[0]}.{s[1]} tile {t}: module to coupler IN", geo=(box, x, tray, x),
                                 bundle=(name, side, "cpl"))
                    for t in T:
                        self.add("RF", f"{self.coupled[s]}.t{t}.THRU", f"{att}.ch{t}.{side}", "{pn}", "", f"{name} tile {t}",
                                 geo=(tray, mod_x(mod, t), att, ch_x(t, side)), bundle=(name, side))
                else:
                    for t in T:
                        self.add("RF", f"{box}.{mod}.T{t}", f"{att}.ch{t}.{side}", "{pn}", "", f"{name} {s[0]}.{s[1]} tile {t}",
                                 geo=(box, mod_x(mod, t), att, ch_x(t, side)), bundle=(name, side))
        for name, (x, y) in d["couplings"].items():
            att = self.att_of[name]
            for t in T:
                self.add("RF", f"{self.coupled[x]}.t{t}.CPL", f"{att}.ch{t}.in",
                         "BLK-18-S+ DC block + BW-S<n>W2+ pad on the coupler's CPL port, then {pn}", "",
                         f"arm {name} tile {t}, phase-matched", geo=(tray, mod_x(d["module"][x][1], t), att, ch_x(t, "in")),
                         bundle=(name, "in"))
            for t in T:
                self.add("RF", f"{att}.ch{t}.out", f"{self.coupled[y]}.t{t}.CPL", "{pn}", "", f"arm {name} tile {t}",
                         geo=(att, ch_x(t, "out"), tray, mod_x(d["module"][y][1], t)), bundle=(name, "out"))
        # IP-50 chains ------------------------------------------------------------------------------
        for j, (k, (a, b)) in enumerate(d.get("mw_links", {}).items()):
            ra, rb = f"IP50-{k}@{a}", f"IP50-{k}@{b}"
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

    def _size_rf(self):
        """Pick each IF bundle's cable from the rack span: vertical distance between the two units' centres +
        horizontal offset between the ports (assumed panel positions) + BEND_MM; all 4 tiles get the same length."""
        centre = {r["id"]: (r["top"] - r["n"] / 2) * U_MM for r in self.rows}
        top = {r["id"]: r["top"] for r in self.rows}
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
                c["part"] = c["part"].format(pn=f"Mini-Circuits {pn}")
                c["length"] = f"{L / 1000:.2f} m"
                c["need_mm"] = round(need)
                c["note"] += f"; span {span}U, needs {need / 1000:.2f} m"

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
    if d.get("agents"):
        units.append(("AGENTS", 2, f"Shelf: {len(d['agents'])} MARL agent PCs", "mini PCs on a 1U shelf, 2U clearance"))
    units.append(("CM", 1, "Cable manager", "Panduit NCMHF1"))
    for x in d["boxes"]:
        pols = sorted({d["nodes"][n]["pol"].split(" ")[0] for n in x["nodes"]})
        units.append((x["id"], 1, f"{x['code']}  ·  " + ", ".join(x["nodes"]), f"{x['kind']} · {'/'.join(pols)}"))
    if b.split:
        units.append(("SPL-TRAY", 1, "Splitter tray (shelf)", f"{len(b.split) * b.tiles} x 2-way splitters ZN2PD-02183-S+"))
    if b.coupled:
        units.append(("CPL-TRAY", 1, "Coupler tray (shelf)", f"{len(b.coupled) * b.tiles} couplers, DC blocks, pads"))
    for a, v in b.atts.items():
        units.append((a, 1, f"{a}  ·  {v[0]} {v[1]}", "Mini-Circuits ZTDAT-4-18G63S"))
    units += [("PA-SHELF", 1, f"DC/PoE adaptors PA-01..PA-{b.n_adaptors:02d}", "Siklu EH-PoE-DC-Adaptor-10 on a shelf"),
              ("DC-DIST", 1, "DC distribution 48 V, 10 breakers", "Eaton DCD1M"),
              ("DC-RECT", 1, "DC rectifier shelf 48 V", "Mean Well RCP-1UI + 2 x RCP-1000-48")]
    if b.ip50:
        units += [("DC-DIST2", 1, "-48 V fuse panel (IP-50)", "Westell N250110-N-0803, GMT fuses"),
                  ("DC-RECT2", 1, "-48 V rectifier (IP-50)", "RCP-1UI + RCP-1000-48, wired -48 V")]
    for p in b.pdus:
        units.append((p, 2, f"{p} 32 A, 16 x C13, switched", "APC AP7922B"))
    total = sum(u[1] for u in units)
    size = 25 if total <= 25 else 42
    rows, u = [], total
    for k, n, lab, sub in units:
        rows.append(dict(id=k, top=u, n=n, label=lab, sub=sub))
        u -= n
    return size, rows, total


# ------------------------------------------------------------------------------------------
# exploded parts list (team kit-list item in 'team')
# ------------------------------------------------------------------------------------------
def parts(b):
    d = b.d
    nb = {k: sum(1 for x in d["boxes"] if x["kind"] == k) for k in BOX_KIND}
    nn = len(d["nodes"])
    hops = len(d.get("mw_links", {}))
    P = []

    def add(group, team, item, pn, mfr, qty, use, conf):
        if qty in (0, "0"):
            return
        P.append(dict(line=len(P) + 1, group=group, team=team, item=item, pn=pn, mfr=mfr, qty=qty, use=use, conf=conf))
    add("Radios", "11", "1U Spider box, 2 x N367 node x 2 sectors", "GTS003 (lab label)", "Siklu", nb["GTS003"],
        "4 sector modules FATJ0498, 16 SMA tile ports", "on hand; to confirm")
    add("Radios", "11", "1U Spider box, 1 x N367 node x 4 sectors (hub)", "GTST002B (lab label)", "Siklu",
        nb["GTST002B"], "16 SMA tile ports", "on hand; to confirm")
    add("Radios", "11", "1U Spider box, 2 x N367 node x 2 sectors (assumed)", "GTST003 (lab label)", "Siklu",
        nb["GTST003"], "16 SMA tile ports", "on hand; to confirm")
    add("Radios", "1", "'6*4 sector nodes - GTSTU022\\23'", "GTSTU022 / GTSTU023", "Siklu", "?",
        "not identified (TU boxes?); not used here", "to confirm with the lab")
    add("Radios", "-", "N367 node (inside the Spider boxes)", "MultiHaul TG N367 (MH-N367-CCP-PoE-MWB class)", "Siklu",
        nn, "ETH1 power in, SFP+ 10G data; managed in-band; 48 W with PoE-Out off", "datasheet")
    if hops:
        add("Radios", "-", "IP-50 radio", "Ceragon IP-50 (model and band TBC)", "Ceragon", 2 * hops,
            f"{hops} conducted hops, -48 V", "on hand (8); model to confirm")
    add("Power", "2", "DC/PoE adaptor, passive, bag of 10 ('DC\\POE for falcon')", "EH-PoE-DC-Adaptor-10", "Siklu",
        f"1 bag ({nn} wired, {b.n_adaptors - nn} spare unwired)", "36-57 V DC in, RJ-45 to node ETH1 (power only)",
        "likely")
    add("Power", "5", "DC drawer: rectifier shelf 1U + 2 x 1 kW 48 V modules", "RCP-1UI + 2 x RCP-1000-48", "Mean Well", 1,
        f"48 V bus, {48 * nn} W for {nn} nodes", "likely (shelf) / confirmed (modules)")
    add("Power", "5", "DC drawer: distribution, 10 breakers", "DCD1M (suffix TBC)", "Eaton", 1,
        f"one breaker per adaptor; CB{b.spare_breakers[0]}-{b.spare_breakers[-1]} spare, OFF" if b.spare_breakers else
        "one breaker per adaptor", "likely")
    if hops:
        add("Power", "5", "-48 V rectifier for the IP-50 radios", "RCP-1UI + RCP-1000-48 wired -48 V", "Mean Well", 1,
            f"{2 * hops} radios x ~58 W (IP-50E figure; recheck for the model)", "likely")
        add("Power", "5", "-48 V fuse panel 1U, 2 buses x 5 GMT positions", "N250110-N-0803", "Westell", 1,
            "one fused output per IP-50 radio", "to confirm")
        add("Power", "5", "GMT fuse for the -48 V panel", "GMT 3 A (rating to confirm)", "Bussmann / Littelfuse",
            2 * hops + 2, f"{2 * hops} + 2 spare", "to confirm")
    add("Power", "7", "Rack PDU 32 A, 16 x C13, switched per outlet", "AP7922B", "APC", len(b.pdus),
        "IEC 60309 32 A inlet" + (f"; {', '.join(f'{b.outlet[x][0]} O{b.outlet[x][1]}' for x in d.get('event_off', []))} "
                                  "switched at t0" if d.get("event_off") else ""), "confirmed")
    add("Power", "-", "Bench-table mains strip, C13 outlets", "lab-supplied (part TBC)", "-", 1,
        f"{len(b.table_loads)} table loads", "to confirm")
    mk = sorted({c["b"].split("-")[0].lower() for c in b.cables if c["pn"] == "P014-006"})
    add("Power", "8", "'Micky' lead C14 -> C5", "P014-006", "Eaton Tripp Lite", b.count_pn("P014-006"),
        " and ".join({"host": "traffic hosts", "agent": "agent-PC bricks"}[m] for m in mk), "confirmed")
    add("Power", "9", "'Exten' lead C14 -> C13 1.8 m", "P004-006", "Eaton Tripp Lite", b.count_pn("P004-006"),
        "PDU and table strip to devices", "meaning to confirm")
    add("Power", "9", "'Exten' lead C14 -> C13 3.05 m", "P004-010", "Eaton Tripp Lite", b.count_pn("P004-010"),
        "switched PDU outlet to the bench table", "confirmed")
    m = sum(c.get("metres", 0) for c in b.cables if c["pn"] == "Belden 8461")
    add("Power", "10", "DC cable 2 x 18 AWG (bulk) + ferrules", "Belden 8461 + Phoenix 3200519", "Belden / Phoenix",
        f"{m:.0f} m + {4 * b.count_pn('Belden 8461')} ferrules", f"{b.count_pn('Belden 8461')} runs; 100 ft spool = 30.5 m",
        "confirmed")
    add("Power", "10", "DC bus link 2 x 10 AWG with ring lugs", "part TBC", "-", b.count_pn("bus link"),
        "rectifier to distribution", "to confirm")
    add("Network", "6", "Data switch, 24 x SFP+ + 4 x SFP28, L3", "M4350-24F4V (XSM4328FV)", "Netgear", 1,
        f"{len(b.sw_port)} of 28 ports; 1G only on P1-16", "confirmed")
    add("Network", "6", "Management switch, 24 x 1G PoE+ + 2 x 1G + 4 x SFP+", b.oob_model[0], "Netgear", 1,
        f"{len(b.oob_port)} ports: {min(len(b.oob_port), OOB_RJ45)} of {OOB_RJ45} RJ-45" +
        (f" + {b.n_agm_oob} SFP+ cage" if b.n_agm_oob else ""), "confirmed")
    add("Network", "-", "1000BASE-T copper SFP", "AGM734-10000S", "Netgear", b.n_agm_data + b.n_agm_oob,
        f"{b.n_agm_data} on SW-DATA P1-16" + (f", {b.n_agm_oob} on SW-OOB" if b.n_agm_oob else ""),
        "confirmed; switch compatibility to confirm")
    nd = b.count_pn("FZ2ERLNLNSNM002")
    add("Network", "-", "SFP+ SR, node side / switch side", "AX-SFP+10G-MM / AXM761-10000S", "Siklu / Netgear",
        f"{nd} + {nd}", "node SFP+ to SW-DATA", "likely / confirmed")
    add("Network", "-", "OM4 LC-LC duplex 2 m", "FZ2ERLNLNSNM002", "Panduit", nd, "", "confirmed")
    add("Network", "-", "SFP+ DAC 3 m", "AXC763-10000S", "Netgear", b.count_pn("AXC763-10000S"),
        " and ".join(x for x in ("traffic hosts" if d["hosts"] else "", "servers" if any(s[2] for s in d["servers"]) else "")
                     if x), "confirmed")
    for pn, L in (("STP6X1MBU", "1 m"), ("STP6X2MBU", "2 m"), ("STP6X3MBU", "3 m")):
        add("Network", "-", f"Cat6A S/FTP patch {L}", pn, "Panduit", b.count_pn(pn), "", "confirmed")
    na = len(b.atts)
    reuse = d.get("att_reuse", 0)
    add("RF/IF", "-", "4-ch programmable attenuator 0-63 dB, 0.1-18 GHz", "ZTDAT-4-18G63S", "Mini-Circuits", na,
        "not on the kit list", f"reuse {reuse} from PoC 1, buy {na - reuse}" if reuse else f"to buy: {na}")
    if b.split:
        add("RF/IF", "-", "2-way splitter 2-18 GHz", "ZN2PD-02183-S+", "Mini-Circuits", len(b.split) * b.tiles,
            "one per tile of each shared hub module", "confirmed")
    if b.coupled:
        add("RF/IF", "-", "10 dB directional coupler 1-18 GHz", "KRYTAR 1821 (alt. ZUDC10-183+)", "KRYTAR",
            len(b.coupled) * b.tiles, "IN at the module", "confirmed")
        add("RF/IF", "-", "DC block 18 GHz", "BLK-18-S+", "Mini-Circuits", len(d["couplings"]) * b.tiles,
            "arm input side, at the coupler", "confirmed")
        add("RF/IF", "-", "Fixed pad DC-18 GHz 2 W (value TBC, e.g. BW-S10W2+)", "BW-S<n>W2+", "Mini-Circuits",
            len(d["couplings"]) * b.tiles, "arm input side, at the coupler", "value to confirm")
    for pn, L in RF_STOCK:
        q = b.count_pn(pn)
        tag = " ('RF cable': 10 on hand, type TBC)" if pn == "CBL-2FT-SMSM+" else ""
        add("RF/IF", "3", f"SMA test cable 18 GHz {L / 1000:.2f} m{tag}", pn, "Mini-Circuits", q,
            "IF chain, length from the rack span" + ("; IP-50 chains" if pn == "CBL-1M-SMSM+" and hops else ""),
            "confirmed")
    add("RF/IF", "-", "50-ohm SMA load", "ANNE-50+", "Mini-Circuits", "as needed", "unused tile and coupler ports",
        "confirmed")
    add("RF/IF", "-", "SMA torque wrench 8 in-lb", "ST-SMA3", "Fairview", 2, "", "confirmed")
    if hops:
        add("IP-50 chain", "-", "Waveguide-to-coax transition, WR-62 12.4-18 GHz (<=18 GHz set)", "PEWCA1003", "Pasternack",
            2 * hops, "band and flange mating TBC", "likely")
        add("IP-50 chain", "-", "30 dB 10 W pad DC-18 GHz (<=18 GHz set)", "SA18S10W-30", "Fairview", 2 * hops,
            "one at each radio port, on the transition", "confirmed")
        add("IP-50 chain", "-", "Programmable attenuator 4 ch DC-18 GHz (<=18 GHz set)", "J7204B opt 002", "Keysight", 1,
            f"{hops} channels used; LAN to SW-OOB", "confirmed")
        add("IP-50 chain", "-", "Alternative set if the band is 18-26.5 GHz (replaces the 3 lines above)",
            "PEWCA1018 + FMAT4009-30 + 84906K/84904K/11713C", "Pasternack / Fairview / Keysight", "replaces the set",
            "2.92 mm pads and cables", "alternative")
    add("Timing", "-", "GNSS antenna + 1x8 splitter", "101898-00 + RMALDCBS1X8", "Protempis / GPS Networking", "1 + 1",
        "only if the boxes take GNSS (N367 GPS is internal)", "to confirm")
    add("Timing", "-", "GNSS coax: antenna lead + surge arrestor / splitter jumpers", "LMR-400 + arrestor / RG-316 SMA 1 m",
        "part TBC", f"1 / {b.count_pn('RG-316')}", "only if the boxes take GNSS", "to confirm")
    if d.get("agents"):
        add("Compute", "-", "MARL agent PC (mini PC, 2 x 1G NICs)", "lab PC", "-", len(d["agents"]), "", "on hand")
    add("Compute", "-", "Traffic host (10G SFP+ NIC, iperf3/TRex)", "lab PC", "-", len(d["hosts"]), "", "on hand")
    add("Compute", "-", "Servers: " + "; ".join(s[1] for s in d["servers"]), "lab servers", "-", len(d["servers"]), "",
        "on hand")
    add("Rack", "4", f"19-inch 4-post rack ({b.total}U used of {b.size}U)",
        "StarTech 4POSTRACK25U" if b.size == 25 else "StarTech 4POSTRACK42", "StarTech", 1,
        "rail type (square hole?) to confirm", "confirmed")
    nshelf = 1 + bool(b.split or b.coupled) + bool(d.get("agents"))
    add("Rack", "4", "1U vented shelf", "CABSHELFV1U", "StarTech", nshelf,
        ", ".join(x for x in ("adaptors", "tray" if (b.split or b.coupled) else "", "agent PCs" if d.get("agents") else "")
                  if x), "likely")
    empty = b.size - b.total
    add("Rack", "-", "Cable manager / blanking panels (pack of 10) / labels", "NCMHF1 / AR8136BLK / Brady M21-750-427",
        "various", f"1 / {math.ceil(empty / 10)} / 3", f"{empty} empty U", "confirmed; rail type to confirm")
    add("Test", "-", "18 GHz power sensor / spectrum analyser", "NRP18S / FPL1026", "R&S", "1 / 1",
        "port checks, calibration", "confirmed")
    add("Test", "-", "N-F to SMA-F adapter for the power sensor", "NF-SF50+", "Mini-Circuits", 1, "NRP18S input is N-male",
        "confirmed")
    return P


def write_tables(b, P, prefix):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{prefix}_cable_schedule.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "class", "from", "to", "part", "length", "vlan", "note"])
        for c in b.cables:
            w.writerow([c["id"], c["cls"], c["a"], c["b"], c["part"], c["length"], c["vlan"], c["note"]])
    with open(os.path.join(OUT, f"{prefix}_parts_list.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["line", "group", "team list item", "item", "part number", "manufacturer", "qty", "use", "status"])
        for p in P:
            w.writerow([p["line"], p["group"], p["team"], p["item"], p["pn"], p["mfr"], p["qty"], p["use"], p["conf"]])
    names = {"AC": "AC mains", "DC": "DC power", "POE": "Cat6A power to node ETH1", "FIB": "Data to SW-DATA (fibre / DAC / Cat6A)",
             "ETH": "1G management to SW-OOB (VLAN 100)", "RF": "IF coax (SMA, 18 GHz)", "GNSS": "GNSS",
             "WG": "IP-50 conducted chain"}
    L = [f"# {b.d['poc']} cable schedule (generated by blueprint.py)", "", f"{b.d['name']}. {len(b.cables)} cables.", "",
         "Nodes are managed in-band (E2E controller over the radio network); no node ETH2 or IP-50 management port is "
         "cabled. IF cable lengths come from the rack span: vertical distance + horizontal port offset + "
         f"{BEND_MM} mm, rounded up to a stock length; confirm at install.", ""]
    for cls, nm in names.items():
        cs = [c for c in b.cables if c["cls"] == cls]
        if cs:
            L += [f"## {nm} ({len(cs)})", "", "| ID | From | To | Part | Length | VLAN | Note |", "|---|---|---|---|---|---|---|"]
            L += [f"| {c['id']} | {c['a']} | {c['b']} | {c['part']} | {c['length']} | {c['vlan']} | {c['note']} |"
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
