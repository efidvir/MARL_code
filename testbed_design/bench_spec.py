"""Bench descriptions for PoC 1 (A7) and PoC 2, and everything derived from them: every cable
(with an ID), the exploded parts list and the rack layout.  blueprint.py draws them.

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
  servers    [(id, role, site or None)]
  no_data    [site]                               nodes with no wired data port (PoC 1 hub)
  att_order  [link or arm]                        attenuator order (optional)
  event      text (PoC 1)
"""
import csv
import os
from collections import OrderedDict

from validate import bearing, sector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEC = {0: "E", 1: "N", 2: "W", 3: "S"}
BOX_KIND = {"GTS003": (2, 2), "GTST002B": (1, 4), "GTST003": (2, 2)}
CLS_PREFIX = {"AC": "A", "DC": "D", "POE": "P", "FIB": "F", "ETH": "E", "RF": "R", "GNSS": "G", "WG": "W"}


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
        no_data=["HUB"],
        event="t0: ATT-01..06 (every path to the hub) to 63 dB; CORE's IP-50 hops to maximum; CORE servers and "
              "the hub switched off. Bridges ATT-07..09 step down when MARL engages their sites.",
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
                            ("SRV-E2E", "TG E2E controller + NMS", None)],
        no_data=[], event=None,
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

    def add(self, cls, a, b, part, length, note=""):
        c = dict(id=self.cid(cls), cls=cls, a=a, b=b, part=part, length=length, note=note)
        self.cables.append(c)
        return c

    def vlan(self, site):
        return 101 + self.d["sites"].index(site)

    def count(self, cls, part_sub=None):
        return sum(1 for c in self.cables if c["cls"] == cls and (part_sub is None or part_sub in c["part"]))

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
        # AC -----------------------------------------------------------------------------------
        rack_loads = ["DC-RECT", "SW-DATA", "SW-OOB", "GNSS-SPL"] + list(self.atts)
        if self.ip50:
            rack_loads.insert(1, "DC-RECT2")
        rack_loads += [f"AGENT-{s}" for s in d.get("agents", [])]

        self.pdus = ["PDU-A"] if len(rack_loads) <= 16 else ["PDU-A", "PDU-B"]
        self.add("AC", "Lab wall socket IEC 60309 32 A", "PDU-A.inlet", "PDU cord (IEC 60309 32 A plug)", "fixed")
        if len(self.pdus) > 1:
            self.add("AC", "Lab wall socket IEC 60309 32 A (2nd)", "PDU-B.inlet", "PDU cord (IEC 60309 32 A plug)",
                     "fixed")
        self.outlet = OrderedDict()
        for i, ld in enumerate(rack_loads):
            pdu, o = self.pdus[i // 16], i % 16 + 1
            self.outlet[ld] = (pdu, o)
            lead = "Eaton Tripp Lite P014-006 C14-C5 'Mickey' lead" if ld.startswith("AGENT") else \
                "Eaton Tripp Lite P004-006 / StarTech PXT1001M C14-C13 lead"
            self.add("AC", f"{pdu}.O{o}", f"{ld}.ac", lead, "1-2 m",
                     "switched at t0" if (d.get("event") and ld == "x") else "")
        self.table_loads = [s[0] for s in d["servers"]] + ["SPECAN"] + [f"HOST-{h}" for h in d["hosts"]]
        for t in self.table_loads:
            lead = "P014-006 C14-C5 'Mickey' lead" if t.startswith("HOST") else "P004-006 C14-C13 lead"
            core = any(s[0] == t and s[2] == "CORE" for s in d["servers"])
            self.add("AC", "TABLE-STRIP", f"{t}.ac", lead, "1.8 m", "switched off at t0 (CORE)" if core else "")
        # DC -----------------------------------------------------------------------------------
        self.add("DC", "DC-RECT.bus", "DC-DIST.bus", "busbar link 2 x 10 AWG with lugs", "0.3 m", "48 V bus")
        for i in range(self.n_adaptors):
            nd = nodes[i] if i < len(nodes) else None
            self.add("DC", f"DC-DIST.CB{i + 1}", f"PA-{i + 1:02d}.dc", "Belden 8461 2 x 18 AWG + Phoenix AI 0,75-8",
                     "1 m", f"feeds node {nd}" if nd else "spare adaptor (not connected to a node)")
        for k, (hop, site) in self.ip50.items():
            self.add("DC", "DC-RECT2.out", f"{k}.dc", "Belden 8461 2 x 18 AWG (IP-50 -48 V; connector TBC)", "3 m",
                     "-48 V, separate rectifier, positive grounded")
        for n in nodes:
            self.add("POE", f"{self.adaptor[n]}.rj45", f"{d['nodes'][n]['box']}.{n}.ETH1",
                     "Panduit STP6X1MBU Cat6A S/FTP 1 m", "1 m", "power only (passive adaptor)")
        # data ----------------------------------------------------------------------------------
        self.sw_port = OrderedDict()
        p = 1

        def data(src, part, length, note, vlan):
            nonlocal p
            self.sw_port[src] = (p, vlan)
            self.add("FIB", src, f"SW-DATA.P{p}", part, length, note)
            p += 1
        for n in nodes:
            if n in d.get("no_data", []):
                continue
            data(f"{d['nodes'][n]['box']}.{n}.SFP+",
                 "SFP+ SR at both ends (Siklu AX-SFP+10G-MM / Netgear AXM761) + Panduit FZ2ERLNLNSNM002 OM4 2 m",
                 "2 m", f"site {n}", f"VLAN {self.vlan(n)}")
        for k, (hop, site) in self.ip50.items():
            data(f"{k}.data", "Cat6A or SFP+ per the IP-50 data port (TBC)", "3 m", f"hop {hop} at {site}",
                 f"VLAN {self.vlan(site)}")
        for s in d.get("agents", []):
            data(f"AGENT-{s}.data", "Cat6A (1G) or SFP+ DAC", "2 m", f"MARL agent + site routing at {s}",
                 f"VLAN {self.vlan(s)}")
        for h, sites in d["hosts"].items():
            v = f"VLAN {self.vlan(sites[0])}" if len(sites) == 1 else "trunk " + ", ".join(str(self.vlan(x)) for x in sites)
            data(f"HOST-{h}.nic10g", "Netgear AXC763-10000S SFP+ DAC 3 m", "3 m", "traffic host", v)
        for sid, role, site in d["servers"]:
            if site:
                data(f"{sid}.data", "Netgear AXC763-10000S SFP+ DAC 3 m", "3 m", role, f"VLAN {self.vlan(site)}")
        self.oob_port = OrderedDict()
        q = 1

        def oob(src, note="management VLAN 100"):
            nonlocal q
            self.oob_port[src] = ("SW-OOB", q)
            self.add("ETH", src, f"SW-OOB.P{q}", "Panduit STP6X1MBU / STP6X2MBU Cat6A", "1-2 m", note)
            q += 1
        for n in nodes:
            oob(f"{d['nodes'][n]['box']}.{n}.ETH2", "node management (PoE-Out on ETH2 off)")
        for a in self.atts:
            oob(f"{a}.lan", "attenuator control, SCPI over TCP")
        for k in self.ip50:
            oob(f"{k}.mgmt", "IP-50 management (port TBC)")
        for pdu in self.pdus:
            oob(f"{pdu}.net", "PDU management")
        for s in d.get("agents", []):
            oob(f"AGENT-{s}.mgmt", "agent PC logging")
        for sid, role, site in d["servers"]:
            oob(f"{sid}.mgmt")
        oob("SPECAN.lan")
        for h in d["hosts"]:
            oob(f"HOST-{h}.mgmt")
        self.oob_model = ("M4250-26G4XF-PoE+ (GSM4230PX)", 26, "confirmed") if len(self.oob_port) <= 26 else \
            ("M4250-40G8XF-PoE+ (GSM4248PX)", 48, "to confirm")
        # GNSS -------------------------------------------------------------------------------------
        self.add("GNSS", "GNSS-ANT", "GNSS-SPL.in", "LMR-400 coax + surge arrestor", "site")
        for i, bx in enumerate(d["boxes"]):
            self.add("GNSS", f"GNSS-SPL.out{i + 1}", f"{bx['id']}.GNSS", "RG-316 SMA coax 1 m", "1 m",
                     "only if the box has a GNSS input (TBC)")
        # RF / IF: one bundle (4 tiles) at a time so that each bundle has consecutive IDs --------------
        for s, sid in self.split.items():
            box, mod = d["module"][s]
            for t in T:
                self.add("RF", f"{box}.{mod}.T{t}", f"{sid}.t{t}.S", "Mini-Circuits 086-4SM+", "0.1 m",
                         f"{s[0]}.{s[1]} tile {t} to its splitter")
        for name, (sa, sb) in d["link_ends"].items():
            att = self.att_of[name]
            for side, s in (("in", sa), ("out", sb)):
                box, mod = d["module"][s]
                if s in self.split:
                    k = d["splitters"][s].index(name) + 1
                    for t in T:
                        self.add("RF", f"{self.split[s]}.t{t}.P{k}", f"{att}.ch{t}.{side}", "Mini-Circuits CBL-2FT-SMSM+",
                                 "0.6 m", f"{name} tile {t}")
                elif s in self.coupled:
                    for t in T:
                        self.add("RF", f"{box}.{mod}.T{t}", f"{self.coupled[s]}.t{t}.IN", "Mini-Circuits 086-4SM+",
                                 "0.1 m", f"{name} {s[0]}.{s[1]} tile {t}: coupler IN at the module")
                    for t in T:
                        self.add("RF", f"{self.coupled[s]}.t{t}.THRU", f"{att}.ch{t}.{side}", "Mini-Circuits CBL-2FT-SMSM+",
                                 "0.6 m", f"{name} tile {t}")
                else:
                    for t in T:
                        self.add("RF", f"{box}.{mod}.T{t}", f"{att}.ch{t}.{side}", "Mini-Circuits CBL-2FT-SMSM+",
                                 "0.6 m", f"{name} {s[0]}.{s[1]} tile {t}")
        for name, (x, y) in d["couplings"].items():
            att = self.att_of[name]
            for t in T:
                self.add("RF", f"{self.coupled[x]}.t{t}.CPL", f"{att}.ch{t}.in",
                         "BLK-18-S+ + BW-S<n>W2+ pad at the coupler, then CBL-2FT-SMSM+", "0.6 m",
                         f"arm {name} tile {t}, phase-matched")
            for t in T:
                self.add("RF", f"{att}.ch{t}.out", f"{self.coupled[y]}.t{t}.CPL", "Mini-Circuits CBL-2FT-SMSM+", "0.6 m",
                         f"arm {name} tile {t}")
        # IP-50 chains ------------------------------------------------------------------------------
        for j, (k, (a, b)) in enumerate(d.get("mw_links", {}).items()):
            ra, rb = f"IP50-{k}@{a}", f"IP50-{k}@{b}"
            self.add("WG", f"{ra}.wg", f"WGT-{k}-A", "waveguide-to-coax transition (band TBC)", "flange", f"hop {k}")
            self.add("WG", f"WGT-{k}-A", f"MWATT.ch{j + 1}.in", "30 dB 10 W pad SA18S10W-30 + coax", "1 m", f"hop {k}")
            self.add("WG", f"MWATT.ch{j + 1}.out", f"WGT-{k}-B", "30 dB 10 W pad SA18S10W-30 + coax", "1 m", f"hop {k}")
            self.add("WG", f"WGT-{k}-B", f"{rb}.wg", "waveguide-to-coax transition (band TBC)", "flange", f"hop {k}")

    def bundle(self, a_prefix, b_prefix):
        """The tile cables from a_prefix to b_prefix, checked to be one consecutive bundle."""
        cs = [c for c in self.cables if c["cls"] == "RF" and c["a"].startswith(a_prefix) and c["b"].startswith(b_prefix)]
        nums = [int(c["id"][1:]) for c in cs]
        assert len(cs) == self.tiles and nums == list(range(nums[0], nums[0] + len(nums))), (a_prefix, b_prefix)
        return cs


# ------------------------------------------------------------------------------------------
# rack layout
# ------------------------------------------------------------------------------------------
def rack_layout(b):
    d = b.d
    units = [("GNSS-SPL", 1, "GNSS splitter 1x8", "RMALDCBS1X8 (if the boxes take GNSS)"),
             ("SW-DATA", 1, "Data switch", "Netgear M4350-24F4V"),
             ("SW-OOB", 1, "Management switch", "Netgear " + b.oob_model[0])]
    if d.get("agents"):
        units.append(("AGENTS", 2, f"Shelf: {len(d['agents'])} MARL agent PCs", "mini PCs, 2U shelf"))
    units.append(("CM", 1, "Cable manager", "Panduit NCMHF1"))
    for x in d["boxes"]:
        pols = sorted({d["nodes"][n]["pol"].split(" ")[0] for n in x["nodes"]})
        units.append((x["id"], 1, f"{x['code']}  ·  " + ", ".join(x["nodes"]), f"{x['kind']} · {'/'.join(pols)}"))
    if b.split:
        units.append(("SPL-TRAY", 1, "Splitter tray (shelf)", f"{len(b.split) * b.tiles} x 2-way splitters ZN2PD-02183-S+"))
    if b.coupled:
        units.append(("CPL-TRAY", 1, "Coupler tray (shelf)", f"{len(b.coupled) * b.tiles} couplers, DC blocks, pads"))
    for a, v in b.atts.items():
        units.append((a, 1, f"{a}  ·  {v[0]} {v[1]}", "ZTDAT-4-18G63S"))
    units += [("PA-SHELF", 1, "DC/PoE adaptors PA-01..PA-10", "Siklu EH-PoE-DC-Adaptor x 10 on a shelf"),
              ("DC-DIST", 1, "DC distribution 48 V, 10 breakers", "Eaton DCD1M"),
              ("DC-RECT", 1, "DC rectifier shelf 48 V", "Mean Well RCP-1UI + 2 x RCP-1000-48")]
    if b.ip50:
        units.append(("DC-RECT2", 1, "-48 V rectifier (IP-50)", "RCP-1UI + RCP-1000-48, wired -48 V"))
    for p in b.pdus:
        units.append((p, 2, f"{p} 32 A, 16 x C13" + (", switched" if d.get("event") else ""), "APC AP7922B"))
    total = sum(u[1] for u in units)
    size = 25 if total <= 25 else 42
    rows, u = [], size
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
        P.append(dict(line=len(P) + 1, group=group, team=team, item=item, pn=pn, mfr=mfr, qty=qty, use=use, conf=conf))
    add("Radios", "11", "1U Spider box, 2 x N367 node x 2 sectors", "GTS003 (lab label)", "Siklu", nb["GTS003"],
        "4 sector modules FATJ0498, 16 SMA tile ports", "on hand")
    add("Radios", "11", "1U Spider box, 1 x N367 node x 4 sectors (hub)", "GTST002B (lab label)", "Siklu",
        nb["GTST002B"], "16 SMA tile ports", "on hand")
    add("Radios", "11", "1U Spider box, 2 x N367 node x 2 sectors (assumed)", "GTST003 (lab label)", "Siklu",
        nb["GTST003"], "16 SMA tile ports", "on hand; to confirm")
    add("Radios", "1", "'6*4 sector nodes - GTSTU022\\23'", "GTSTU022 / GTSTU023", "Siklu", "?",
        "not identified (TU boxes?); not used here", "to confirm with the lab")
    add("Radios", "-", "N367 node (inside the Spider boxes)", "MultiHaul TG N367 (MH-N367-CCP-PoE-MWB class)", "Siklu",
        nn, "ETH1 power in, ETH2 1G, SFP+ 10G; 48 W with PoE-Out off", "datasheet")
    if hops:
        add("Radios", "-", "IP-50 radio", "IP-50C class (band, licence TBC)", "Ceragon", 2 * hops,
            f"{hops} conducted hops, -48 V", "on hand (8)")
    add("Power", "2", "DC/PoE adaptor, passive, bag of 10 ('DC\\POE for falcon')", "EH-PoE-DC-Adaptor-10", "Siklu",
        f"1 bag ({nn} used)", "36-57 V DC in, RJ-45 to node ETH1 (power only)", "likely")
    add("Power", "5", "DC drawer: rectifier shelf 1U + 2 x 1 kW 48 V modules", "RCP-1UI + 2 x RCP-1000-48", "Mean Well", 1,
        f"48 V bus, {48 * nn} W for {nn} nodes", "confirmed")
    add("Power", "5", "DC drawer: distribution, 10 breakers", "DCD1M (suffix TBC)", "Eaton", 1, "one breaker per adaptor",
        "to confirm")
    if hops:
        add("Power", "5", "-48 V supply for the IP-50 radios", "RCP-1UI + RCP-1000-48 wired -48 V", "Mean Well", 1,
            f"{2 * hops} radios x ~58 W", "to confirm")
    add("Power", "7", "Rack PDU 32 A, 16 x C13" + (", switched per outlet" if d.get("event") else ""),
        "AP7922B (alt. Raritan PX3-5260R 16 A)", "APC", len(b.pdus), "IEC 60309 inlet", "confirmed")
    nm = len([c for c in b.cables if c["cls"] == "AC" and "C5" in c["part"]])
    add("Power", "8", "'Micky' lead C14 -> C5", "P014-006", "Eaton Tripp Lite", nm, "hosts and agent-PC bricks",
        "confirmed")
    add("Power", "9", "'Exten' lead C14 -> C13", "P004-006 / PXT1001M", "Eaton / StarTech",
        len([c for c in b.cables if c["cls"] == "AC" and "C13" in c["part"]]), "PDU and table strip to devices",
        "meaning to confirm")
    add("Power", "10", "DC cable 2 x 18 AWG + ferrules", "Belden 8461 + Phoenix 3200519", "Belden / Phoenix",
        b.count("DC") - 1, "breaker -> adaptor; -48 V -> IP-50", "confirmed")
    add("Network", "6", "Data switch, 24 x SFP+ + 4 x SFP28, L3", "M4350-24F4V", "Netgear", 1,
        f"{len(b.sw_port)} ports used, VLAN per site", "confirmed")
    add("Network", "6", f"Management switch, {b.oob_model[1]} x 1G", b.oob_model[0], "Netgear", 1,
        f"{len(b.oob_port)} ports used", b.oob_model[2])
    nd = sum(1 for k in b.sw_port if k.endswith("SFP+"))
    add("Network", "-", "SFP+ SR, node side / switch side", "AX-SFP+10G-MM / AXM761-10000S", "Siklu / Netgear",
        f"{nd} + {nd}", "", "likely / confirmed")
    add("Network", "-", "OM4 LC-LC duplex 2 m", "FZ2ERLNLNSNM002", "Panduit", nd, "", "confirmed")
    add("Network", "-", "SFP+ DAC 3 m", "AXC763-10000S", "Netgear",
        b.count("FIB", "DAC"), "hosts and servers", "confirmed")
    add("Network", "-", "Cat6A S/FTP patch 1-2 m", "STP6X1MBU / STP6X2MBU", "Panduit",
        b.count("POE") + b.count("ETH") + b.count("FIB", "Cat6A"), "", "confirmed")
    add("RF/IF", "-", "4-ch programmable attenuator 0-63 dB, 0.1-18 GHz", "ZTDAT-4-18G63S", "Mini-Circuits",
        len(b.atts), "9 on hand from PoC 1", "confirmed")
    if b.split:
        add("RF/IF", "-", "2-way splitter 2-18 GHz", "ZN2PD-02183-S+", "Mini-Circuits", len(b.split) * b.tiles,
            "one per tile of each shared hub module", "confirmed")
    if b.coupled:
        add("RF/IF", "-", "10 dB directional coupler 1-18 GHz", "KRYTAR 1821 (alt. ZUDC10-183+)", "KRYTAR",
            len(b.coupled) * b.tiles, "IN at the module", "confirmed")
        add("RF/IF", "-", "DC block 18 GHz", "BLK-18-S+", "Mini-Circuits", len(d["couplings"]) * b.tiles, "arms",
            "confirmed")
        add("RF/IF", "-", "Fixed pad DC-18 GHz 2 W", "BW-S<n>W2+", "Mini-Circuits", len(d["couplings"]) * b.tiles,
            "arms", "confirmed")
    add("RF/IF", "3", "SMA test cable 18 GHz 0.6 m ('RF cable': 10 on hand)", "CBL-2FT-SMSM+", "Mini-Circuits",
        b.count("RF", "CBL-2FT"), "", "confirmed")
    if b.count("RF", "086-4SM"):
        add("RF/IF", "3", "SMA jumper 4 in", "086-4SM+", "Mini-Circuits", b.count("RF", "086-4SM"), "", "confirmed")
    add("RF/IF", "-", "50-ohm SMA load", "ANNE-50+", "Mini-Circuits", "as needed", "unused tile and coupler ports",
        "confirmed")
    add("RF/IF", "-", "SMA torque wrench 8 in-lb", "ST-SMA3", "Fairview", 2, "", "confirmed")
    if hops:
        add("IP-50 chain", "-", "Waveguide-to-coax transition (band TBC)", "PEWCA1003 / PEWCA1018", "Pasternack",
            2 * hops, "", "likely")
        add("IP-50 chain", "-", "30 dB 10 W pad", "SA18S10W-30", "Fairview", 2 * hops, "", "confirmed")
        add("IP-50 chain", "-", "Programmable attenuator, 4 ch (<=18 GHz bands)", "J7204B opt 002", "Keysight", 1,
            f"{hops} channels used", "confirmed")
    add("Timing", "-", "GNSS antenna + 1x8 splitter", "101898-00 + RMALDCBS1X8", "Protempis / GPS Networking", "1 + 1",
        "only if the boxes take GNSS", "to confirm")
    if d.get("agents"):
        add("Compute", "-", "MARL agent PC (mini PC, 2 NICs)", "lab PC", "-", len(d["agents"]), "", "on hand")
    add("Compute", "-", "Traffic host (10G NIC, iperf3/TRex)", "lab PC", "-", len(d["hosts"]), "", "on hand")
    add("Compute", "-", "Servers: " + "; ".join(s[1] for s in d["servers"]), "lab servers", "-", len(d["servers"]), "",
        "on hand")
    size, rows, total = rack_layout(b)
    add("Rack", "4", f"19-inch 4-post rack ({total}U used)", f"StarTech 4POSTRACK{size}U" if size == 25 else
        "StarTech 4POSTRACK42 (to confirm)", "StarTech", 1, "", "confirmed" if size == 25 else "to confirm")
    add("Rack", "4", "1U vented shelf", "CABSHELFV1U", "StarTech", 1 + bool(b.split or b.coupled), "adaptors, trays", "likely")
    add("Rack", "-", "Cable manager, blanking panels, labels", "NCMHF1 / AR8136BLK / Brady M21-750-427", "various",
        "1 / 1 pack / 3", "", "confirmed")
    add("Test", "-", "18 GHz power sensor (gate) / spectrum analyser", "NRP18S / FPL1026", "R&S", "1 / 1",
        "port checks, calibration", "confirmed")
    return P


def write_tables(b, P, prefix):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{prefix}_cable_schedule.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "class", "from", "to", "part", "length", "note"])
        for c in b.cables:
            w.writerow([c["id"], c["cls"], c["a"], c["b"], c["part"], c["length"], c["note"]])
    with open(os.path.join(OUT, f"{prefix}_parts_list.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["line", "group", "team list item", "item", "part number", "manufacturer", "qty", "use", "status"])
        for p in P:
            w.writerow([p["line"], p["group"], p["team"], p["item"], p["pn"], p["mfr"], p["qty"], p["use"], p["conf"]])
    names = {"AC": "AC mains", "DC": "DC power", "POE": "Cat6A power to node ETH1", "FIB": "10G data (fibre / DAC)",
             "ETH": "1G management", "RF": "IF coax (SMA, 18 GHz)", "GNSS": "GNSS", "WG": "IP-50 conducted chain"}
    L = [f"# {b.d['poc']} cable schedule (generated by blueprint.py)", "", f"{b.d['name']}. {len(b.cables)} cables.", ""]
    for cls, nm in names.items():
        cs = [c for c in b.cables if c["cls"] == cls]
        if cs:
            L += [f"## {nm} ({len(cs)})", "", "| ID | From | To | Part | Length | Note |", "|---|---|---|---|---|---|"]
            L += [f"| {c['id']} | {c['a']} | {c['b']} | {c['part']} | {c['length']} | {c['note']} |" for c in cs] + [""]
    with open(os.path.join(OUT, f"{prefix}_cable_schedule.md"), "w") as fh:
        fh.write("\n".join(L))
    L = [f"# {b.d['poc']} exploded parts list (generated by blueprint.py)", "",
         "Team = item number on the lab's kit list (1 = '6*4 sector nodes - GTSTU022\\23', 2 = '10 DC\\POE for falcon', "
         "3 = '10 RF cable', 4 = RACK, 5 = DC drawer, 6 = Switch\\netgear, 7 = PDU 16\\32A, 8 = Micky cable, "
         "9 = Exten cable, 10 = DC cable, 11 = 1U Spider boxes; '-' = not on the list).", "",
         "| # | Team | Group | Item | Part number | Manufacturer | Qty | Use | Status |", "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {p['line']} | {p['team']} | {p['group']} | {p['item']} | {p['pn']} | {p['mfr']} | {p['qty']} | {p['use']} | "
          f"{p['conf']} |" for p in P]
    with open(os.path.join(OUT, f"{prefix}_parts_list.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
