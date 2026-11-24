"""Digital twin of a bench: every device placed in 3D (rack and bench table), every port positioned, every cable
routed port to port with its length measured along the route, the installation steps, and physical checks.

Input: bench_spec.py (the same data that drives the blueprints, cable schedules and parts lists).
Output: out/twin/<poc>_scene.json (read by the 3D viewer and the installation guide) and out/twin/<poc>_checks.md.

TENTATIVE geometry: dimensions and panel layouts come from twin/catalog.py; everything not confirmed by the
lab is listed in the scene's 'assumptions' and printed in the guide.

Run:  python3 twin/twin_model.py
"""
import json
import math
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bench_spec as BS  # noqa: E402
import catalog as C  # noqa: E402

BS.LADDER = C.LADDER

OUT = os.path.join(BS.OUT, "twin")


def r1(v):
    return [round(float(x), 1) for x in v]


class Scene:
    def __init__(self, b):
        self.b, self.d = b, b.d
        self.dev = OrderedDict()
        self.ports = {}
        self.assumed = list(C.RACK["assumed"]) + list(C.TABLE["assumed"])
        self.counters = {}
        self._rack()
        self._table()
        self._cables()
        self._steps()
        self._checks()

    # ---------------------------------------------------------------------------------------------
    def add_dev(self, did, kind, label, sub, pos, size, color, group, conf="estimate", note="", parent=None, **kw):
        self.dev[did] = dict(id=did, kind=kind, label=label, sub=sub, pos=r1(pos), size=r1(size), color=color,
                             group=group, conf=conf, note=note, parent=parent, ports=[], **kw)
        if conf == "estimate" and note and not parent:
            self.assumed.append(f"{did}: {note}")
        return self.dev[did]

    def add_port(self, did, name, pos, normal, ptype, label=None):
        key = f"{did}.{name}" if name else did
        self.ports[key] = dict(id=key, device=did, name=name, pos=r1(pos), normal=list(normal), type=ptype,
                               label=label or name)
        self.dev[did]["ports"].append(key)

    # rack ----------------------------------------------------------------------------------------
    def _rack(self):
        b, d = self.b, self.d
        U, y1 = C.U, C.RACK["u1_y"]
        self.rack = dict(units=b.size, used=b.total, u1_y=y1, rail_depth=C.RACK["rail_depth"], panel_w=C.PANEL_W,
                         part=C.RACK["part"], frame_w=C.RACK["frame_w"])
        kinds = {"SW-DATA": "sw_data", "SW-OOB": "sw_oob", "GNSS-SPL": "gnss", "CM": "cm", "AGENTS": "shelf",
                 "SPL-TRAY": "tray", "CPL-TRAY": "tray", "PA-SHELF": "shelf", "DC-DIST": "dcdist", "DC-DIST2": "fuse",
                 "DC-RECT": "rect", "DC-RECT2": "rect", "PDU-A": "pdu", "PDU-B": "pdu"}
        used = set()
        for r in b.rows:
            rid = r["id"]
            kind = ("spider" if rid.startswith("SB") else "attshelf" if rid.startswith("ATT-SHELF") else
                    "rackserver" if rid.startswith(("SRV-", "HOST-")) else "pdu" if rid.startswith("PDU-") else kinds[rid])
            k = C.KIND[kind]
            y0 = y1 + (r["top"] - r["n"]) * U
            h = r["n"] * U - 1.0
            for u in range(r["top"] - r["n"] + 1, r["top"] + 1):
                used.add(u)
            dv = self.add_dev(rid, kind, rid if not rid.startswith("SB") else f"{rid} {r['label'].split('·')[0].strip()}",
                              r["sub"], (0, y0, 0), (C.PANEL_W, h, k["d"]), k["color"], "rack", k["conf"], k["note"],
                              u=f"U{r['top'] - r['n'] + 1}" + (f"-U{r['top']}" if r["n"] > 1 else ""), u_top=r["top"],
                              u_n=r["n"], title=r["label"])
            getattr(self, "_ports_" + kind, lambda *a: None)(dv, rid)
        for u in range(1, b.size + 1):
            if u not in used:
                self.add_dev(f"BLANK-U{u}", "blank", f"blank U{u}", "AR8136BLK", (0, y1 + (u - 1) * U, -4),
                             (C.PANEL_W, U - 1.0, C.KIND["blank"]["d"]), C.KIND["blank"]["color"], "rack", "likely", "",
                             u=f"U{u}", u_top=u, u_n=1)

    def _face(self, dv):
        x, y, z = dv["pos"]
        w, h, dd = dv["size"]
        return x, y, z, w, h, dd

    def _ports_spider(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        box = next(bx for bx in self.d["boxes"] if bx["id"] == rid)
        used = {m: f"{s[0]}.{s[1]}" for s, (bx, m) in self.d["module"].items() if bx == rid}
        dv["modules"] = [dict(m=f"M{k}", x0=BS.mod_x(f"M{k}", 1) - 12, x1=BS.mod_x(f"M{k}", 4) + 12,
                              sector=used.get(f"M{k}", "spare")) for k in range(1, 5)]
        for k in range(1, 5):
            for t in range(1, 5):
                self.add_port(rid, f"M{k}.T{t}", (BS.mod_x(f"M{k}", t), y + h * 0.42, z), (0, 0, -1), "SMA",
                              f"M{k} T{t}")
        nn = len(box["nodes"])
        for i, nd in enumerate(box["nodes"]):
            bx0 = 120 + i * 170 if nn > 1 else 200
            for j, (nm, pt) in enumerate((("ETH1", "RJ45"), ("ETH2", "RJ45"), ("SFP+", "SFP"))):
                self.add_port(rid, f"{nd}.{nm}", (bx0 + j * 42, y + h / 2, z + dd), (0, 0, 1), pt, f"{nd} {nm}")
        self.add_port(rid, "GNSS", (45, y + h / 2, z + dd), (0, 0, 1), "SMA", "GNSS in (TBC)")
        dv["nodes"] = box["nodes"]

    def _ports_attshelf(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        bw, bh, bd = BS.ATT_BOX
        for a, (sid, slot) in self.b.att_shelf.items():
            if sid != rid:
                continue
            x0 = BS.att_x0(slot)
            kind, name = self.b.atts[a]
            k = C.KIND["ztdat"]
            self.add_dev(a, "ztdat", a, f"programmable attenuator box, 4 ch ({kind} {name})", (x0, y + 4, z + 8),
                         (bw, bh, bd), k["color"], "rack", "estimate", k["note"], parent=rid, carries=f"{kind} {name}")
            for c in range(1, 5):
                for side in ("in", "out"):
                    self.add_port(a, f"ch{c}.{side}", (BS.ch_x(c, side, slot), y + 4 + bh * 0.35, z + 8), (0, 0, -1), "SMA",
                                  f"CH{c} {side.upper()}")
            self.add_port(a, "lan", (x0 + bw * 0.3, y + 4 + bh / 2, z + 8 + bd), (0, 0, 1), "RJ45", "LAN")
            self.add_port(a, "ac", (x0 + bw * 0.75, y + 4 + bh / 2, z + 8 + bd), (0, 0, 1), "DC", "power (adapter)")

    def _ports_rackserver(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        site = next((s[2] for s in self.d["servers"] if s[0] == rid), "host")
        names = ([("nic10g", "SFP")] if rid.startswith("HOST") else ([("data", "SFP")] if site else [])) + \
            [("mgmt", "RJ45"), ("ac", "C14")]
        for i, (nm, pt) in enumerate(names):
            self.add_port(rid, nm, (120 + i * 110, y + h / 2, z + dd), (0, 0, 1), pt, nm)

    def _ports_tray(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        b = self.b
        T = range(1, b.tiles + 1)
        for sec, sid in list(b.split.items()) + list(b.coupled.items()):
            bx, m = self.d["module"][sec]
            for t in T:
                cx = BS.mod_x(m, t)
                kind = "splitter" if sec in b.split else "coupler"
                s = C.SMALL[kind]
                bid = f"{sid}.t{t}"
                self.add_dev(bid, kind, bid, s["note"], (cx - s["size"][0] / 2, y + 6, z + 4), s["size"], s["color"],
                             "rack", "estimate", s["note"], parent=rid)
                yc = y + 6 + s["size"][1] / 2
                if kind == "splitter":
                    self.add_port(bid, "S", (cx, yc + 4, z + 4), (0, 0, -1), "SMA", "S")
                    self.add_port(bid, "P1", (cx - 4, yc - 4, z + 4), (0, 0, -1), "SMA", "P1")
                    self.add_port(bid, "P2", (cx + 4, yc - 4, z + 4), (0, 0, -1), "SMA", "P2")
                else:
                    self.add_port(bid, "IN", (cx, yc + 4, z + 4), (0, 0, -1), "SMA", "IN")
                    self.add_port(bid, "THRU", (cx, yc - 4, z + 4), (0, 0, -1), "SMA", "THRU")
                    self.add_port(bid, "CPL", (cx + 6, yc, z + 4), (0, 0, -1), "SMA", "CPL")

    def _ports_shelf(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        b = self.b
        if rid == "PA-SHELF":
            spider_rear = {nd: self.ports[f"{self.d['nodes'][nd]['box']}.{nd}.ETH1"]["pos"][0] for nd in self.d["nodes"]}
            taken = []
            for nd in b.dc_nodes:
                a = b.adaptor[nd]
                self._small(a, "adaptor", spider_rear[nd], y, z + dd - 100, rid, ("rj45", "RJ45"), ("dc", "DC"))
                taken.append(spider_rear[nd])
            for nd, inj in b.inj.items():
                self._small(inj, "injector", spider_rear[nd], y, z + dd - 150, rid, ("poe", "RJ45"), ("ac", "C6?"),
                            color="#bf616a")
                taken.append(spider_rear[nd])
            xs, xx = [], 465.0
            while len(xs) < b.n_adaptors - len(b.dc_nodes):
                if all(abs(xx - t) > 36 for t in taken):
                    xs.append(xx)
                xx -= 34
            for j, xx in enumerate(sorted(xs)):
                a = f"PA-{len(b.dc_nodes) + j + 1:02d}"
                self._small(a, "adaptor", xx, y, z + dd - 100, rid, spare=True)
        elif rid == "AGENTS":
            for i, s in enumerate(self.d.get("agents", [])):
                aid = f"AGENT-{s}"
                px = 20 + i * 75
                self.add_dev(aid, "minipc", aid, "MARL agent mini PC (2 NICs)", (px, y + 4, z + 150), (70, 40, 120),
                             C.TABLE_DEV["minipc"]["color"], "rack", "estimate", "mini PC size estimate", parent=rid)
                for j, nm in enumerate(("data", "mgmt", "ac")):
                    self.add_port(aid, nm, (px + 12 + j * 22, y + 24, z + 270), (0, 0, 1),
                                  "RJ45" if nm != "ac" else "C5", nm)

    def _small(self, sid, kind, cx, y, z0, parent, *ports, color=None, spare=False):
        s = C.SMALL[kind]
        w, h, dd = s["size"]
        self.add_dev(sid, kind, sid, s["note"] + (" (spare, unwired)" if spare else ""), (cx - w / 2, y + 3, z0),
                     (w, h, dd), "#9aa5b1" if spare else (color or s["color"]), "rack", "estimate", s["note"],
                     parent=parent, spare=spare)
        for i, (nm, pt) in enumerate(ports):
            self.add_port(sid, nm, (cx, y + 3 + h * (0.72 if i == 0 else 0.3), z0 + dd), (0, 0, 1), pt, nm)

    def _ports_dcdist(self, dv, rid, n=10, pre="CB"):
        x, y, z, w, h, dd = self._face(dv)
        for i in range(1, n + 1):
            self.add_port(rid, f"{pre}{i}", (40 + (i - 1) * 36, y + h / 2, z + dd), (0, 0, 1), "DC", f"{pre}{i}")
        self.add_port(rid, "bus", (440, y + h / 2, z + dd), (0, 0, 1), "DC", "bus")

    def _ports_fuse(self, dv, rid):
        self._ports_dcdist(dv, rid, 10, "F")

    def _ports_rect(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        self.add_port(rid, "ac", (450, y + h / 2, z + dd), (0, 0, 1), "C14", "AC in")
        self.add_port(rid, "bus" if rid == "DC-RECT" else "out", (400, y + h / 2, z + dd), (0, 0, 1), "DC", "DC out")

    def _ports_pdu(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        for o in range(1, BS.PDU_OUT + 1):
            self.add_port(rid, f"O{o}", (40 + (o - 1) * 30, y + h / 2, z + dd), (0, 0, 1), "C13", f"O{o}")
        self.add_port(rid, "inlet", (462, y + h / 2, z + dd), (0, 0, 1), "IEC60309", "16 A inlet cord")
        self.add_port(rid, "net", (445, y + h * 0.7, z), (0, 0, -1), "RJ45", "net")

    def _ports_sw_data(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        for p in range(1, BS.DATA_PORTS + 1):
            col, row = (p - 1) // 2, (p - 1) % 2
            px = 45 + col * 25 + (18 if p > 24 else 0)
            self.add_port(rid, f"P{p}", (px, y + h * (0.66 - 0.32 * row), z), (0, 0, -1), "SFP", f"P{p}")
        self.add_port(rid, "ac", (450, y + h / 2, z + dd), (0, 0, 1), "C14", "AC")

    def _ports_sw_oob(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        for p in range(1, BS.OOB_RJ45 + BS.OOB_SFP + 1):
            col, row = (p - 1) // 2, (p - 1) % 2
            px = 40 + col * 24 + (16 if p > BS.OOB_RJ45 else 0)
            self.add_port(rid, f"P{p}", (px, y + h * (0.66 - 0.32 * row), z), (0, 0, -1),
                          "RJ45" if p <= BS.OOB_RJ45 else "SFP", f"P{p}")
        self.add_port(rid, "ac", (450, y + h / 2, z + dd), (0, 0, 1), "C14", "AC")

    def _ports_gnss(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        self.add_port(rid, "in", (60, y + h / 2, z + dd), (0, 0, 1), "SMA", "antenna in")
        for i in range(1, 9):
            self.add_port(rid, f"out{i}", (120 + (i - 1) * 30, y + h / 2, z + dd), (0, 0, 1), "SMA", f"out {i}")
        self.add_port(rid, "ac", (450, y + h / 2, z + dd), (0, 0, 1), "C14", "AC")

    # bench table -----------------------------------------------------------------------------------
    def _table(self):
        b, d, T = self.b, self.d, C.TABLE
        top = T["top"]
        self.table = dict(T)
        z_front = T["z0"] + 60
        x = T["x0"] + 20
        # compute zone nearest the rack: servers stacked, then PCs
        for sid, role, site in d["servers"]:
            if sid in BS.TABLE_SERVERS:
                self._tdev(sid, "pc", role, (x, top, z_front + 250), C.TABLE_DEV["pc"]["size"], ("mgmt", "RJ45"),
                           ("ac", "C14"), ("video", "DP"), ("usb", "USB"))
        self._tdev("MONITOR", "monitor", "24-inch monitor (operator station)", (x + 220, top, z_front + 330),
                   C.TABLE_DEV["monitor"]["size"], ("in", "DP"), ("ac", "C14"))
        self._tdev("KBD", "keyboard", "keyboard + mouse", (x + 200, top, z_front), C.TABLE_DEV["keyboard"]["size"],
                   ("usb", "USB"))
        hx = x + 300
        hosts = []
        # IP-50 zone
        hops = list(d.get("mw_links", {}).items())
        mx = hx + 900
        if hops:
            ms = C.TABLE_DEV["mwatt"]["size"]
            self._tdev("MWATT", "mwatt", "Keysight J7204B, 4 ch", (mx, top, z_front), ms, ("lan", "RJ45"), ("ac", "C14"))
            for j, (k, (a, bb)) in enumerate(hops):
                for side in ("in", "out"):
                    self.add_port("MWATT", f"ch{j + 1}.{side}", (mx + 50 + j * 95 + (0 if side == "in" else 40), top + 44,
                                                                 z_front), (0, 0, -1), "SMA", f"CH{j + 1} {side}")
            rs = C.TABLE_DEV["ip50"]["size"]
            for j, (k, (a, bb)) in enumerate(hops):
                zr = z_front + j * (rs[2] + 25) if j < 2 else z_front + j * (rs[2] + 25)
                zr = z_front - 40 + (j % 3) * 250
                lift = (j // 3) * (rs[1] + 5)
                for side, site, xr, face in (("A", a, mx - 520, +1), ("B", bb, mx + ms[0] + 290, -1)):
                    rid = f"IP50-{k}@{site}"
                    self._tdev(rid, "ip50", f"IP-50, hop {k} at {site}", (xr, top + lift, zr), rs, ("data", "RJ45"),
                               ("mgmt", "RJ45"), ("dc", "DC"))
                    wx = xr + rs[0] if face > 0 else xr
                    self.add_port(rid, "wg", (wx, top + lift + rs[1] / 2, zr + rs[2] / 2), (face, 0, 0), "WG", "waveguide")
                    ts = C.SMALL["transition"]["size"]
                    tid = f"WGT-{k}-{side}"
                    tx0 = wx if face > 0 else wx - ts[0]
                    self.add_dev(tid, "transition", tid, C.SMALL["transition"]["note"],
                                 (tx0, top + lift + rs[1] / 2 - ts[1] / 2, zr + rs[2] / 2 - ts[2] / 2), ts,
                                 C.SMALL["transition"]["color"], "table", "estimate", C.SMALL["transition"]["note"],
                                 parent=rid)
                    self.add_port(tid, "wg", (wx, top + lift + rs[1] / 2, zr + rs[2] / 2), (-face, 0, 0), "WG", "flange")
                    self.add_port(tid, "coax", (wx + face * ts[0], top + lift + rs[1] / 2, zr + rs[2] / 2), (face, 0, 0),
                                  "SMA", "coax + 30 dB pad")
        for i, p in enumerate(b.psu.values()):
            self._tdev(p, "psu", "48 V supply, switched at t0", (mx + i * 190, top, T["z1"] - 110),
                       C.TABLE_DEV["psu"]["size"], ("out", "DC"), ("ac", "C14"))
        # analyser and strip
        sx = mx + 1300
        self._tdev("SPECAN", "specan", "spectrum analyser / power sensor", (sx, top, z_front), C.TABLE_DEV["specan"]["size"],
                   ("lan", "RJ45"), ("ac", "C14"))
        st = C.TABLE_DEV["strip"]["size"]
        stx = (T["x0"] + sx) / 2 - 280
        self.add_dev("TABLE-STRIP", "strip", "TABLE-STRIP", "bench mains strip, C13", (stx, top, T["z1"] - 70),
                     st, C.TABLE_DEV["strip"]["color"], "table", "estimate", C.TABLE_DEV["strip"]["note"])
        n = len(b.table_loads)
        for i in range(1, max(n, 1) + 1):
            self.add_port("TABLE-STRIP", f"O{i}", (stx + 30 + (i - 1) * (st[0] - 60) / max(1, n - 1), top + st[1],
                                                   T["z1"] - 40), (0, 1, 0), "C13", f"O{i}")
        # wall socket(s) and GNSS antenna
        self.add_dev("WALL", "wall", "Wall sockets IEC 60309 16 A", "lab wall", (60, 300, C.RACK["rail_depth"] + 260),
                     (360, 120, 40), "#7f8c8d", "room", "estimate", "wall sockets behind the rack (estimate)")
        for i in range(1, len(b.pdus) + 1):
            self.add_port("WALL", f"S{i}", (120 + (i - 1) * 160, 360, C.RACK["rail_depth"] + 260), (0, 0, -1), "IEC60309",
                          f"socket {i}")
        self.add_dev("GNSS-ANT", "antenna", "GNSS-ANT", "101898-00 (roof); lead via surge arrestor",
                     (230, 2350, C.RACK["rail_depth"] - 40), (40, 90, 40), "#27ae60", "room", "estimate",
                     "GNSS antenna lead enters above the rack (estimate)")
        self.add_port("GNSS-ANT", "", (250, 2350, C.RACK["rail_depth"] - 20), (0, -1, 0), "N", "antenna")

    def _tdev(self, did, kind, sub, pos, size, *ports):
        t = C.TABLE_DEV[kind]
        self.add_dev(did, kind, did, sub, pos, size, t["color"], "table", t["conf"], t["note"])
        x, y, z = pos
        w, h, dd = size
        ps = [p for p in ports if p]
        for i, (nm, pt) in enumerate(ps):
            self.add_port(did, nm, (x + w * (i + 1) / (len(ps) + 1), y + h / 2, z + dd), (0, 0, 1), pt, nm)

    # cables ------------------------------------------------------------------------------------------
    def resolve(self, ep, cls):
        if ep == "TABLE-STRIP" or ep.startswith("Lab wall socket"):
            dev, pre = ("TABLE-STRIP", "O") if ep == "TABLE-STRIP" else ("WALL", "S")
            self.counters[dev] = self.counters.get(dev, 0) + 1
            return f"{dev}.{pre}{self.counters[dev]}"
        if ep == "GNSS-ANT":
            return "GNSS-ANT"
        if ep in self.ports:
            return ep
        raise KeyError(f"unresolved port {ep} ({cls})")

    def route(self, P0, P1, cls, lane):
        p0, n0 = P0["pos"], P0["normal"]
        p1, n1 = P1["pos"], P1["normal"]
        d0, d1 = self.dev[P0["device"]], self.dev[P1["device"]]
        g0 = "rack" if d0["group"] == "rack" else d0["group"]
        g1 = "rack" if d1["group"] == "rack" else d1["group"]
        so = 25 + 5 * (lane % 12)

        def off(p, n, s):
            return [p[0] + n[0] * s, p[1] + n[1] * s, p[2] + n[2] * s]
        a, bpt = off(p0, n0, so), off(p1, n1, so)
        rd = C.RACK["rail_depth"]
        if g0 == "rack" and g1 == "rack":
            if n0[2] < 0 and n1[2] < 0:                       # front to front: lane plane in front of the rack
                zl = -so
                pts = [p0, [p0[0], p0[1], zl], [p0[0], p1[1], zl], [p1[0], p1[1], zl], p1]
            else:                                              # rear: behind the deeper device, side lane if far apart
                zl = max(p0[2], p1[2]) + so
                side = -35 - 3 * (lane % 8) if cls in ("AC", "DC", "POE") else C.PANEL_W + 35 + 3 * (lane % 8)
                if abs(p0[0] - p1[0]) < 80 or abs(p0[1] - p1[1]) < 50:
                    pts = [p0, [p0[0], p0[1], zl], [p0[0], p1[1], zl], [p1[0], p1[1], zl], p1]
                else:
                    pts = [p0, [p0[0], p0[1], zl], [side, p0[1], zl], [side, p1[1], zl], [p1[0], p1[1], zl], p1]
            return pts
        if {g0, g1} == {"rack", "table"}:
            (pr, nr, pt, nt), rev = ((p0, n0, p1, n1), False) if g0 == "rack" else ((p1, n1, p0, n0), True)
            side = C.PANEL_W + 45 + 3 * (lane % 10)
            ytray = C.TABLE["top"] + 12 + 2 * (lane % 6)
            ztray = C.TABLE["z1"] - 25 - 4 * (lane % 8)
            q = off(pt, nt, so)
            if nr[2] < 0:                                      # front port: out, along the front to the side, to the rear
                zf = pr[2] - so
                head = [pr, [pr[0], pr[1], zf], [side, pr[1], zf], [side, pr[1], ztray]]
            else:
                zl = max(pr[2] + so, 420.0)
                head = [pr, [pr[0], pr[1], zl], [side, pr[1], zl], [side, pr[1], ztray]]
            pts = head + [[side, ytray, ztray], [q[0], ytray, ztray], [q[0], ytray, q[2]], [q[0], q[1], q[2]], pt]
            return pts[::-1] if rev else pts
        if g0 == "table" and g1 == "table":
            y = C.TABLE["top"] + 8 + 2 * (lane % 6)
            if n0[0] != 0 or n1[0] != 0:                       # waveguide chain: side ports, straight along x then z
                return [p0, a, [a[0], p0[1], a[2]], [bpt[0], p1[1], a[2]], [bpt[0], p1[1], bpt[2]], bpt, p1]
            return [p0, a, [a[0], y, a[2]], [bpt[0], y, a[2]], [bpt[0], y, bpt[2]], bpt, p1]
        # room (wall, antenna) to rack
        return [p0, a, [a[0], bpt[1], a[2]], [bpt[0], bpt[1], a[2]], bpt, p1]

    @staticmethod
    def plen(pts):
        return sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))

    def _cables(self):
        b = self.b
        self.cables = []
        self.counters = {}
        lane_of = {}
        for i, c in enumerate(b.cables):
            a = self.resolve(c["a"], c["cls"])
            z = self.resolve(c["b"], c["cls"])
            key = str(c.get("bundle", c["id"]))
            lane = lane_of.setdefault((c["cls"], key), len([k for k in lane_of if k[0] == c["cls"]]))
            pts = self.route(self.ports[a], self.ports[z], c["cls"], lane)
            L = self.plen(pts)
            m = re.match(r"^\s*([\d.]+)\s*m\b", str(c["length"]))
            spec = float(m.group(1)) * 1000 if m else None
            self.cables.append(dict(id=c["id"], cls=c["cls"], a=a, b=z, a_raw=c["a"], b_raw=c["b"], part=c["part"],
                                    pn=c["pn"], length=c["length"], spec_mm=spec, route_mm=round(L), vlan=c["vlan"],
                                    note=c["note"], route=[r1(p) for p in pts], color=C.CABLE[c["cls"]],
                                    dia=C.CABLE_DIA[c["cls"]], bundle=key))

    def apply_lengths(self):
        """Choose every non-IF cable from its standard-length ladder by its routed length (+10 %); DC is cut from bulk.
        Updates the bench's cables in place (so the schedules and parts list follow) and re-routes."""
        b = self.b
        route = {c["id"]: c["route_mm"] for c in self.cables}
        # IF bundles: one length for the 4 tiles (phase-matched), from the longest routed tile + 5 %
        groups = {}
        for c in b.cables:
            if c["cls"] == "RF" or (c["cls"] == "WG" and "CBL-" in c["pn"]):
                groups.setdefault((c["cls"], str(c.get("bundle", c["id"]))), []).append(c)
        for cs in groups.values():
            need = max(route[c["id"]] for c in cs) * 1.05
            pn, L = next((s for s in BS.RF_STOCK if s[1] >= need), BS.RF_STOCK[-1])
            for c in cs:
                if c["pn"] != pn:
                    c["part"] = re.sub(r"CBL-[\w.]+-SMSM\+", pn, c["part"])
                    c["pn"], c["length"] = pn, f"{L / 1000:.2f} m"
                c["need_mm"] = round(need)
        for c in b.cables:
            if c["cls"] == "GNSS" and c["pn"] == "RG-316":
                L = 1000 if route[c["id"]] * C.SLACK <= 1000 else 2000
                c["length"], c["part"] = f"{L / 1000:g} m", f"RG-316 SMA coax {L / 1000:g} m (part TBC)"
        for c in b.cables:
            need = route[c["id"]] * C.SLACK
            pn = c["pn"]
            lad = None
            if c["cls"] == "AC" and pn.startswith("P014"):
                lad = "AC_C5"
            elif c["cls"] == "AC" and pn.startswith("P004"):
                lad = "AC_C13"
            elif pn.startswith("STP6X"):
                lad = "CAT6A"
            elif pn.startswith("AXC76"):
                lad = "DAC" if need <= C.LADDER["DAC"][-1][1] else None
                if lad is None:                                # too far for a DAC: SR optics + OM4 fibre
                    c["pn"], c["part"] = "FZ2ERLNLNSNM002", "SFP+ SR at both ends (host NIC optic TBC / Netgear AXM761) + {om4}"
                    lad = "OM4"
            elif pn.startswith("FZ2ERLNLNSNM"):
                lad = "OM4"
            if c["cls"] == "DC" and pn == "Belden 8461":
                m = math.ceil(need / 250.0) * 0.25
                c["metres"], c["length"] = m, f"{m:g} m (cut)"
                continue
            if not lad:
                continue
            opt = next((o for o in C.LADDER[lad] if o[1] >= need), C.LADDER[lad][-1])
            old = c["pn"]
            c["pn"], c["length"] = opt[0], f"{opt[1] / 1000:g} m"
            if lad == "OM4":
                c["part"] = re.sub(r"Panduit FZ2ERLNLNSNM\d+ OM4 LC-LC [\d.]+ m", opt[2], c["part"]).replace("{om4}", opt[2])
            elif lad == "CAT6A":
                c["part"] = re.sub(r"Panduit STP6X\d+MBU( Cat6A( S/FTP)?)? [\d.]+ m", opt[2], c["part"])
            else:
                c["part"] = opt[2] + c["part"][len(c["part"].split(" lead")[0]) + 5:] if " lead" in c["part"] else opt[2]
            if old != opt[0]:
                c["note"] = (c["note"] + "; " if c["note"] else "") + f"length from the twin route ({route[c['id']]} mm)"
        self._cables()
        self._steps()
        self._checks()

    # installation steps --------------------------------------------------------------------------
    def _steps(self):
        b, d = self.b, self.d
        devs = self.dev
        rackd = [k for k, v in devs.items() if v["group"] == "rack" and not v.get("parent")]

        def by(pred):
            return [k for k in rackd if pred(k)]
        S = []

        def step(title, text, devices=(), cables=(), view="iso"):
            S.append(dict(n=len(S) + 1, title=title, text=text, devices=list(devices), cables=list(cables), view=view))
        step("Assemble the rack", f"Build the {C.RACK['part']} ({self.rack['units']}U), set the rail-to-rail depth to "
             f"{int(C.RACK['rail_depth'])} mm, level it and bond it to earth. Stack from U1 upwards: power at the bottom.",
             view="front")
        step("Mount the PDUs", "Fit the switched PDUs at the bottom (2U each), outlets facing the rear (to confirm). Do not "
             "plug them into the wall yet.", by(lambda k: k.startswith("PDU")), view="rear")
        step("Mount the DC drawers", "Rectifier shelves and distribution panels above the PDUs: -48 V for the IP-50 radios "
             "(if any), then 48 V for the node adaptors. Breakers and fuses OFF.",
             by(lambda k: k.startswith("DC-")), view="front")
        step("Adaptor shelf", "Fit the 1U shelf and place each DC/PoE adaptor under the node it feeds (spares at the right, "
             "unwired)" + (" and the AC PoE injector of each node that is switched off at t0" if b.inj else "") + ".",
             by(lambda k: k == "PA-SHELF") + [k for k, v in devs.items() if v.get("parent") == "PA-SHELF"], view="rear")
        shelves = by(lambda k: k.startswith("ATT-SHELF"))
        step("Programmable attenuators", "Fit one 1U vented shelf per 2U attenuator slot, then stand two attenuator boxes "
             "on each, ATT-01 top-left, numbers increasing left to right and downwards. SMA IN/OUT and the dial face the "
             "front; LAN and power at the back.", shelves + [k for k, v in devs.items() if v.get("parent") in shelves],
             view="front")
        trays = by(lambda k: k.endswith("TRAY"))
        if trays:
            step("Splitter / coupler tray", "Fit the 1U tray directly under the Spider boxes; place one body under each "
                 "module tile it serves, all the same way round.", trays + [k for k, v in devs.items() if v.get("parent") in trays],
                 view="front")
        step("Spider boxes", "Mount the four 1U Spider boxes in order SB1 (top) to SB4. Note the module map on each front "
             "panel (sheet 2 of the blueprint).", by(lambda k: k.startswith("SB")), view="front")
        srv = by(lambda k: k.startswith(("SRV-", "HOST-")))
        if srv:
            step("Servers and traffic hosts", "Fit the rails and slide in the rack servers (sizes in the rack schedule), "
                 "network ports at the back.", srv, view="front")
        step("Switches, cable manager, shelves", "Cable manager under the switches; SW-OOB and SW-DATA; GNSS splitter at the top"
             + ("; agent PCs on their shelf" if d.get("agents") else "") + ".",
             by(lambda k: k in ("CM", "SW-DATA", "SW-OOB", "GNSS-SPL", "AGENTS")) +
             [k for k, v in devs.items() if v.get("parent") == "AGENTS"], view="front")
        step("Blanking panels", "Blank every empty U.", [k for k in devs if k.startswith("BLANK")], view="front")
        table = [k for k, v in devs.items() if v["group"] == "table" and not v.get("parent")]
        step("Bench table", "Place the bench table to the right of the rack and set out the equipment: the operator station "
             "(bench controller PC, monitor, keyboard) nearest the rack, the IP-50 chains around MWATT, the analyser at the "
             "far end, the strip along the back edge.",
             table + [k for k, v in devs.items() if v["group"] == "table" and v.get("parent")], view="table")
        groups = [("AC", "AC mains", "rear"), ("DC", "DC power", "rear"), ("POE", "PoE to the nodes", "rear"),
                  ("FIB", "Data to SW-DATA", "rear"), ("ETH", "Management to SW-OOB", "rear"), ("GNSS", "GNSS", "rear"),
                  ("AUX", "Operator station", "table")]
        for cls, title, view in groups:
            cs = [c["id"] for c in self.cables if c["cls"] == cls]
            if cs:
                step(f"Cable: {title}", f"{len(cs)} cables; tag both ends with the ID before routing.", cables=cs, view=view)
        for name, (sa, sb) in d["link_ends"].items():
            att = b.att_of[name]
            ids = [c["id"] for c in self.cables if c["cls"] == "RF" and (f"{att}." in c["a"] + c["b"])]
            for sec in (sa, sb):
                bx, m = d["module"][sec]
                if sec in b.split or sec in b.coupled:
                    sid = b.split.get(sec) or b.coupled.get(sec)
                    ids += [c["id"] for c in self.cables if c["cls"] == "RF" and c["a"].startswith(f"{bx}.{m}.T")
                            and c["b"].startswith(sid) and c["id"] not in ids]
            step(f"IF link {name} ({att})", f"Tile i of {sa[0]}.{sa[1]} to {att} CH i IN, CH i OUT to tile i of {sb[0]}.{sb[1]}; "
                 "torque 8 in-lb.", cables=sorted(set(ids)), view="front")
        for name, (x, z) in d["couplings"].items():
            att = b.att_of[name]
            ids = [c["id"] for c in self.cables if c["cls"] == "RF" and f"{att}." in c["a"] + c["b"]]
            step(f"Coupling arm {name} ({att})", "CPL port -> DC block -> pad -> cable -> attenuator -> cable -> DC block -> "
                 "CPL port; phase-matched cables.", cables=ids, view="front")
        ws = [c["id"] for c in self.cables if c["cls"] == "WG"]
        if ws:
            step("IP-50 conducted chains", "Radio -> waveguide transition -> 30 dB pad -> 1 m SMA -> MWATT channel -> 1 m SMA "
                 "-> pad -> transition -> radio. Set every MWATT channel before powering the radios.", cables=ws, view="table")
        step("Checks and power-up", "Run the install checks (this guide's last section), then power up: PDUs, DC drawers "
             "(breakers one by one), switches, nodes, radios.", view="iso")
        self.steps = S

    # checks ------------------------------------------------------------------------------------------
    def _checks(self):
        ch = []
        used = {}
        for c in self.cables:
            for e in (c["a"], c["b"]):
                used.setdefault(e, []).append(c["id"])
        dup = {k: v for k, v in used.items() if len(v) > 1}
        ch.append(dict(name="Every port carries at most one cable", ok=not dup,
                       detail="; ".join(f"{k}: {v}" for k, v in dup.items()) or f"{len(used)} ports used"))
        short = [c for c in self.cables if c["spec_mm"] and c["route_mm"] > c["spec_mm"]]
        ch.append(dict(name="Every cable reaches along its route", ok=not short,
                       detail="; ".join(f"{c['id']} {c['a']} -> {c['b']}: route {c['route_mm']} mm > {int(c['spec_mm'])} mm"
                                        for c in short[:40]) + (f" (+{len(short) - 40} more)" if len(short) > 40 else "")
                       or f"{sum(1 for c in self.cables if c['spec_mm'])} cables with a length checked"))
        loose = [c for c in self.cables if c["spec_mm"] and c["route_mm"] < 0.35 * c["spec_mm"] and c["cls"] == "RF"]
        ch.append(dict(name="IF cables not grossly over-length (slack loops)", ok=True,
                       detail=f"{len(loose)} IF cables use under 35 % of their length (dress the slack in the cable manager)"))
        rack = [v for v in self.dev.values() if v["group"] == "rack" and not v.get("parent") and v["kind"] != "blank"]
        deep = [v["id"] for v in rack if v["size"][2] > C.RACK["rail_depth"]]
        ch.append(dict(name="Every rack device fits the rail depth", ok=not deep, detail=", ".join(deep) or
                       f"deepest {max(v['size'][2] for v in rack):.0f} mm of {C.RACK['rail_depth']:.0f} mm"))
        us = []
        for v in rack:
            us += list(range(v["u_top"] - v["u_n"] + 1, v["u_top"] + 1))
        ch.append(dict(name="No two devices share a rack unit", ok=len(us) == len(set(us)),
                       detail=f"{len(set(us))} U used of {self.rack['units']}"))
        unres = [p for p in self.ports.values() if p["type"] == "SMA" and p["device"].startswith("SB")]
        ch.append(dict(name="Spider SMA tile ports", ok=True, detail=f"{sum(1 for p in unres if p['id'] in used)} of "
                       f"{len(unres)} used; unused ones get ANNE-50+ loads"))
        self.checks = ch

    # output ----------------------------------------------------------------------------------------
    def to_json(self):
        legend = [dict(cls=k, color=C.CABLE[k], name=C.CABLE_NAME[k], n=sum(1 for c in self.cables if c["cls"] == k))
                  for k in C.CABLE]
        return dict(meta=dict(poc=self.d["poc"], name=self.d["name"], drawing=self.d["drawing"], event=self.d.get("event"),
                              switched=self.b.switched, tentative=True),
                    rack=self.rack, table=self.table, devices=list(self.dev.values()), ports=list(self.ports.values()),
                    cables=self.cables, steps=self.steps, checks=self.checks, assumptions=self.assumed, legend=legend)


def build(design, prefix, b=None):
    b = b or BS.Bench(design)
    sc = Scene(b)
    sc.apply_lengths()
    write(sc, prefix)
    return sc


def write(sc, prefix):
    design = sc.d
    os.makedirs(OUT, exist_ok=True)
    js = sc.to_json()
    with open(os.path.join(OUT, f"{prefix}_scene.json"), "w") as fh:
        json.dump(js, fh, separators=(",", ":"))
    L = [f"# {design['poc']} digital twin: install checks (generated by twin/twin_model.py)", "",
         "TENTATIVE geometry: see the assumptions list.", "", "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |" for c in sc.checks]
    L += ["", "## Assumptions to confirm", ""] + [f"- {a}" for a in sc.assumed]
    with open(os.path.join(OUT, f"{prefix}_checks.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    return sc


def main():
    for design, prefix in ((BS.poc1_design(), "poc1"), (BS.robust_design(), "poc2")):
        sc = build(design, prefix)
        fails = [c for c in sc.checks if not c["ok"]]
        print(f"{prefix}: {len(sc.dev)} devices, {len(sc.ports)} ports, {len(sc.cables)} cables, {len(sc.steps)} steps, "
              f"{len(fails)} failed checks -> out/twin/{prefix}_scene.json")
        for c in fails:
            print("  FAIL", c["name"], "::", c["detail"][:600])


if __name__ == "__main__":
    main()
