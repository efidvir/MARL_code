"""Digital twin of a bench: every device placed in 3D (rack and bench table), every port positioned, every cable
routed port to port with its length measured along the route, the installation steps, and physical checks.

Input: bench_spec.py (the same data that drives the blueprints, cable schedules and parts lists).
Output: out/twin/<poc>_scene.json (read by the 3D viewer and the installation guide) and out/twin/<poc>_checks.md.

TENTATIVE geometry: dimensions and panel layouts come from twin/catalog.py; everything not confirmed by the
lab is listed in the scene's 'assumptions' and printed in the guide.

Run:  python3 twin/twin_model.py
"""
import heapq
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
    # assumptions every bench shares; the ones that depend on what a bench has are added in _layout_assumed().
    # '{step:<title>}' and '{if_slack}' are filled in when the scene is written (to_json)
    LAYOUT_ASSUMED = (
        "PDUs and SW-OOB on the REAR posts (outlets / ports flush with the rear rail); everything else from the front "
        "rail. The Spider boxes, DC-RECT, DC-RECT2 and SW-DATA also run on StarTech UNIRAILS1U 1U rails (420-1054 mm, "
        "per the StarTech listing shipped with its own M5 cage nuts and panel screws; each rail bolts to its front and "
        "rear post with that kit hardware): confirm the rail flange shares the 1U hole pattern with the ears at a dry fit",
        "M6 cage nuts counted for the ears only: 4 per front- or rear-mounted item (2 per ear), 2 per rear lacing bar, "
        "+10 % spare; rail-kit servers, UNIRAILS1U rails (own M5 hardware) and toolless blanks take none",
        "Rack on castors: the levelling feet are lowered (or the castors locked) and the rack's stabiliser or floor "
        "anchor fitted if supplied, before the servers go in; only one server is out on its rails at a time",
        "Tray parts lie flat with the long axis along x, input end left and output end right (seen from the front)",
        "IF cables run through side lanes just outside the front ears (left or right, whichever is nearer to both "
        "ends), never across the face of another unit; that makes them longer than a straight drop",
        "Mini-Circuits CBL-4FT/5FT/6FT-SMSM+ assumed available (1.22 / 1.52 / 1.83 m) between the 1 m and 2 m stock",
        "IF and IP-50 SMA cables: the shortest stock length covering the longest routed cable of the bundle + 5 % "
        "(test cables, no termination waste; the twin's routes are square-cornered polylines, which overstate a real "
        "run with bend radii, so 5 % is kept rather than 10 %; lowest slack in this build {if_slack}); every other "
        "cable from its ladder at route + 10 %, DC and earth cut to route + 10 %",
        "Tile and arm cables need equal stock length only, not a phase-matched set: the per-tile phase spread is taken "
        "up by beam training. Confirm with the radio team; if not, order a phase-matched set of 4 per bundle (8 per arm) "
        "or add a phase / delay measurement to the port check",
        "One wall socket IEC 60309 16 A per PDU plus one for the bench strip, each on its own circuit",
        "PX3-class PDUs support a per-outlet power-on state (Off) and front-panel outlet switching: confirm on the "
        "delivered unit; if not, leave each C13 lead unplugged at its PDU outlet until its power-up stage",
        "DC conductor convention: the supply conductor of every D run gets a red (+48 V) or blue (-48 V) heat-shrink "
        "sleeve at both ends before it is landed; the unsleeved conductor is the 0 V return (Belden 8461 colours vary "
        "by reel)",
        "Electrical loads are typical-draw estimates (catalog.LOAD_W; rectifier modules from their DC load, each rated "
        "for the full load; SRV-DPIR 800 W is a GPU-server guess): replace with nameplate or measured values; each PDU "
        "and the strip is held to 80 % of 16 A at 230 V, and every PDU's input current is read at power-up (12.8 A or "
        "less) and again with SRV-DPIR's GPU under its working load",
        "Bench-table cable lanes along the back edge: data bundle 70-110 mm from the back edge, power bundle within "
        "30 mm of it (50 mm or more apart); table cables reach the lanes through the 120 mm gap between the radio "
        "rows and the aisles beside the radio columns; rack-to-table runs cross on the cable bridge (CABLE-BRIDGE: data "
        "in front, power behind); the strip cord crosses the floor under a cable protector",
        "GNSS: the roof antenna's lead enters the room through the wall behind the rack, at the surge arrestor "
        "(GNSS-ARR, DC pass) bonded to the lab earth bar; position TBC with facilities, and if the building already has "
        "an entry arrestor, the lead is bonded there instead",
    )

    def _layout_assumed(self):
        """the assumptions that only apply to what this bench has (coupler tray, agent shelf, stacked radios, switched
        servers, IP-50 earthing, Spider-box DCIN feeds)."""
        b, d = self.b, self.d
        out = []
        if b.coupled:
            out.append(f"The {len(b.coupled) * b.tiles} couplers need a 3U space (1U shelf + 2U clear): see the coupler-arm "
                       "budget entry; rear rows sit about 220 mm deep: confirm reach at a dry fit, or split them over two "
                       "trays")
        hops = list(d.get("mw_links", {}))
        if b.stacked_hops:
            up = b.stacked_hops
            out.append(f"IP-50 radios of hop{'s' if len(up) > 1 else ''} {', '.join(up)} are stacked on those of hop"
                       f"{'s' if len(up) > 1 else ''} {', '.join(hops[:len(up)])}, on two {BS.SPACER_MM:g} mm spacer bars "
                       "each (front and rear edge of the lower lid, fixed with double-sided foam tape) so the lower radio "
                       "(about 58 W) keeps a convection gap; about 6 kg each; 120 mm or more between front and back rows "
                       "for the rear connectors")
        elif hops and b.mw_pads:
            out.append("IP-50 radios in hop pairs: the two radios of a hop sit side by side on the table, flanges facing "
                       f"each other, {self.TABLE_PADS['chain_gap']:g} mm between the two pad stacks for the one SMA cable; "
                       "the four hops in a front and a back row of two columns, 120 mm or more between the rows for the "
                       f"rear connectors, {self.TABLE_PADS['aisle']:g} mm aisle between the columns; none stacked")
        elif hops:
            out.append("IP-50 radios in a front and a back row, 120 mm or more apart for the rear connectors; none stacked")
        if b.if_pads:
            P = C.PAD["BW-S"]
            rows = "; ".join(f"{k} {v['target']:g} -> {v['db']} dB = {v['text']}" + (f" ({v['rule']})" if "needs" in v["rule"]
                                                                                  or "tie" in v["rule"] else "")
                             for k, v in b.pad_links.items())
            out.append("Fixed pads instead of programmable attenuators (user, 2026-11-25: PoC 1 links only need to go down, "
                       "and whole devices lose power at t0): each IF tile path gets a stack of at most "
                       f"{BS.PAD_MAX} Mini-Circuits BW-S<n>W2+ pads (values assumed available: "
                       f"{', '.join(str(v) for v in BS.BW_S_DB)} dB), sized from the A7 attenuator setting (hub paths the "
                       "'before' value, they die with the hub; bridges the 'after' value, as MARL engages them), rounded to "
                       "the nearest whole dB (a tie rounds up, toward more loss); a value no stack makes takes the nearest "
                       f"total that one does (a tie toward more loss): {rows}. The port check confirms each link against "
                       "its expected SNR (see below)")
            out.append(f"Pad stacks: BW-S pad about {P['len']:g} mm long and {P['dia']:g} mm across (estimate; the "
                       "device research gives the FF658 case as 21.6 mm long and 9.1 mm across, which would make a stack "
                       f"about 43 mm: the twin keeps the larger estimate until a pad is measured), a two-pad stack drawn "
                       f"{2 * P['len']:g} mm (the mated overlap ignored); SMA male (coupling nut) at the port end, female "
                       "at the far end. The hub W / E links' stacks screw onto the splitter's P1 / P2: they are fitted and "
                       "torqued in the hand in the splitter-tray step (step {step:Splitter tray}), before the splitter goes "
                       "onto the tray, and lie on the tray ({tray_pad_pitch} mm between the P1 and P2 stacks, about "
                       "{tray_pad_gap} mm "
                       "between their bodies, no wrench fits there on the tray; if two stacks foul, fit P2's through a short "
                       "SMA M-F adapter so its nuts stand staggered beyond P1's, parts list 'if needed'). The tray ends of "
                       "their cables (the S cable and both stack OUT cables) are torqued in the hand in the same step, "
                       "before the splitter goes onto the tray: once the Spider boxes are in, the lowest one sits about "
                       "{tray_headroom} mm above the tray floor and no tray joint can be reached; the IF-link steps only "
                       "route those cables and land their tile ends. Every other link's stack screws onto its A-end tile (the first site of the "
                       f"link name) and stands about {2 * P['len']:g} mm proud of the Spider box: its cable is strapped to "
                       "the seam bundle at the unit's lower edge (or the cable manager) within 150 mm, then to the side "
                       "lane, so the tile carries only the stack (two pads, about 60 g, estimate), never the cable")
            out.append("Each pad stack stands in for the A7 attenuator setting alone: the programmable attenuator's own "
                       f"insertion loss (about {BS.ARM_BUDGET['att_il_db']:g} dB, bench_spec ARM_BUDGET) and the second "
                       "cable of each path are gone, so every IF path is a few dB stronger than on the attenuator bench at "
                       "the same setting. The port check (after the nodes are up, before the radios) reads every link's "
                       "SNR against the A7 model's expected value (lowest data-MCS threshold + the link's margin); a link "
                       f"more than {BS.PORT_CHECK_TOL_DB:g} dB strong gets trim pads at its B tiles "
                       f"({' / '.join(str(v) for v in BS.TRIM_PADS)} dB, {b.tiles} of each in the parts list)")
            if b.bridges:
                out.append("Bridges (links between surviving nodes) are held down before t0: " +
                           BS.bridge_hold_text(b.bridges) + ", after SRV-E2E boots and before the nodes do. " +
                           BS.bridge_engage_text(b.bridges) + ". No RF change at any time. After a run: " +
                           b.bridge_after_text() + ". Check that the Siklu E2E honours per-link ignition the same way")
        if b.mw_pads:
            Q = C.PAD[BS.MW_PAD_HI[0]]
            out.append(f"IP-50 chains without a programmable attenuator: {BS.MW_PAD_HI[0]} ({BS.MW_PAD_HI[1]} dB 10 W, about "
                       f"{Q['len']:g} x {Q['dia']:g} mm, estimate; the device research gives 47.2 mm long and 15.75 mm "
                       f"across) on each transition and one {BS.MW_PAD_LO[0]} "
                       f"({BS.MW_PAD_LO[1]} dB) on the A side after it, then one SMA cable: "
                       f"{2 * BS.MW_PAD_HI[1] + BS.MW_PAD_LO[1]} dB fixed plus the cable; the 10 dB pad stands in for the "
                       "former J7204B rest value. If the band turns out to be 23 GHz: WR-42 transitions, 2.92 mm pads and "
                       f"cable, and the 10 dB pad becomes a {BS.MW_PAD_LO_K}")
        if d.get("agents"):
            out.append("Front-row agent PCs on the AGENTS shelf are turned round (connectors to the front) so every "
                       "connector has 35 mm or more of clear space; their data and management cables go along the front "
                       "to the right side")
        srv = [x for x in b.switched if x.startswith("SRV-")]
        if srv:
            out.append(f"Servers that die at t0 ({', '.join(srv)}) run on ONE PSU: a second PSU is removed before the "
                       "server goes in the rack, so cutting its one switched outlet takes it down")
        dc = [r for r in ("DC-RECT", "DC-RECT2") if r == "DC-RECT" or b.ip50]
        out.append(f"DCD1M breakers {BS.BREAKER_A} A or less (they protect the 18 AWG adaptor and DCIN runs; a node draws "
                   "about 1 A); the 10 AWG bus links are protected by the rectifier current limit only: " +
                   "; ".join(f"{r} {BS.RECT_MODULES.get(r, 1)} x RCP-1000-48, about "
                             f"{BS.RECT_MODULES.get(r, 1) * C.RCP_MODULE_A:.0f} A" for r in dc) +
                   " (above 30 A: confirm the RCP shuts down on sustained overload, or fit a 30 A DC fuse at that "
                   "rectifier's output); a DCD1M main / input breaker, if fitted, is closed at power-up but not counted on")
        nb = len(d["boxes"])
        cbs = sorted({v[0] for v in b.dcin.values()})
        shared = [cb for cb in cbs if sum(1 for v in b.dcin.values() if v[0] == cb) > 1]
        out.append(f"Spider box {self.dcin_word} (DCIN, rear) powers only the box fans and the Raspberry Pi, assumed 48 V "
                   f"DC (TBC); the nodes are powered only through ETH1. The {nb} DCIN inputs are fed from DC-DIST "
                   f"CB{cbs[0]}-CB{cbs[-1]}" + "".join(f" ({' and '.join(k for k, v in b.dcin.items() if v[0] == cb)} "
                                                        f"share CB{cb}, twin ferrule)" for cb in shared) +
                   " and closed first at power-up so the fans run before the nodes boot. Hold point in step "
                   "{step:Spider boxes} (Spider boxes): confirm DCIN's voltage and function before any DCIN run is "
                   "landed; any other voltage or function: STOP, design change")
        out.append("Earthing: the rack frame is bonded to the lab earth bar (B001, 6 mm2 green/yellow)" +
                   ("; the -48 V system is positive grounded by bonding the DC-DIST2 RTN bus (= DC-RECT2 +V) to the rack "
                    "earth bar" if b.ip50 else "") +
                   "; the +48 V node system (DC-RECT) stays floating" +
                   (f"; the switched supplies ({', '.join(b.psu.values())}) are isolated and float" if b.psu else "") +
                   "; the GNSS arrestor is bonded to the lab earth bar" +
                   ("; every IP-50 chassis ground screw is bonded (2.5 mm2) to the bench-table earth bar TABLE-EARTH, "
                    "which is bonded (6 mm2) to the rack earth bar: the radio DC input may be isolated from the chassis "
                    "or tied to its 0 V, both fit (RTN is earthed, the switched supplies float); confirm with Ceragon's "
                    "indoor guidance" if b.ip50 else ""))
        return out

    def __init__(self, b):
        self.b, self.d = b, b.d
        self.dev = OrderedDict()
        self.ports = {}
        # what the Spider box's rear DC input is called: the lab photo (PoC 1) shows a green 4-pole pluggable header
        self.dcin_word = b.d.get("dcin_word", "DC terminal block")
        self.dcin_plug = b.d.get("dcin_plug")
        self.assumed = list(C.RACK["assumed"]) + list(C.TABLE["assumed"]) + list(self.LAYOUT_ASSUMED) + \
            self._layout_assumed()
        self.counters = {}
        self._rack()
        self._arm_budget()
        self._table()
        self._pads()
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
        used = set()
        sdepth = {"SRV-E2E": 823.0, "SRV-DPIR": 772.0}
        for r in b.rows:
            rid = r["id"]
            kind = BS.rack_kind(rid)
            k = C.KIND[kind]
            y0 = y1 + (r["top"] - r["n"]) * U
            h = r["n"] * U - 1.0
            for u in range(r["top"] - r["n"] + 1, r["top"] + 1):
                used.add(u)
            dep = sdepth.get(rid, 497.0) if kind == "rackserver" else k["d"]
            # rear-post items sit with their back face on the rear rail plane
            z0 = C.RACK["rail_depth"] - dep if r["mount"].startswith("rear posts") else 0.0
            note = k["note"]
            if kind == "attshelf":
                na = sum(1 for sid, _ in b.att_shelf.values() if sid == rid)
                note = (f"2U of rack space: a 1U vented shelf StarTech CABSHELFV1U (254 mm deep) carrying {na} attenuator "
                        f"box{'es' if na > 1 else ''}, 1U clear above")
            elif kind == "tray" and r["n"] > 1:
                note = (f"{r['n']}U of rack space: a 1U vented shelf StarTech CABSHELFV1U (254 mm deep) in the lower U; "
                        f"the upper {r['n'] - 1}U stays open for the DC blocks, pads, plugs and cable bends on the "
                        "coupler CPL ports")
            elif kind == "rect":
                nm = BS.RECT_MODULES.get(rid, 1)
                note = (f"Mean Well RCP-1UI (350.8 mm), {nm} RCP-1000-48 module{'s' if nm > 1 else ''} fitted: one C14 "
                        f"inlet per module, so {nm} AC cord{'s' if nm > 1 else ''}; DC bus at the rear")
            dv = self.add_dev(rid, kind, rid if not rid.startswith("SB") else f"{rid} {r['label'].split('·')[0].strip()}",
                              r["sub"], (0, y0, z0), (C.PANEL_W, h, dep), k["color"], "rack", k["conf"], note,
                              u=f"U{r['top'] - r['n'] + 1}" + (f"-U{r['top']}" if r["n"] > 1 else ""), u_top=r["top"],
                              u_n=r["n"], title=r["label"], mount=r["mount"], hardware=r["hardware"])
            dv["note"] = f"Mount: {r['mount']}; {r['hardware']}. {note}"
            getattr(self, "_ports_" + kind, lambda *a: None)(dv, rid)
        for u in range(1, b.size + 1):
            if u not in used:
                self.add_dev(f"BLANK-U{u}", "blank", f"blank U{u}", "AR8136BLK", (0, y1 + (u - 1) * U, -4),
                             (C.PANEL_W, U - 1.0, C.KIND["blank"]["d"]), C.KIND["blank"]["color"], "rack", "likely", "",
                             u=f"U{u}", u_top=u, u_n=1, mount="front posts, toolless", hardware="none (toolless)")

    def _arm_budget(self):
        """the coupler tray's mechanical budget above the CPL ports, as an assumption to confirm at a dry fit."""
        if not self.b.coupled or "CPL-TRAY" not in self.dev:
            return
        tr = self.dev["CPL-TRAY"]
        cpl = [p for p in self.ports.values() if p["name"] == "CPL"]
        y = max(p["pos"][1] for p in cpl)
        room = tr["pos"][1] + tr["size"][1] - y
        self.assumed.append(
            f"Coupler-arm budget (mechanical, PoC 2): {room:.0f} mm from the top of the coupler CPL ports to the unit above "
            f"the {tr['u_n']}U CPL-TRAY ({tr['u']}); on each port BLK-18-S+ about {self.BLOCK_MM:g} mm + BW-S pad about "
            f"{self.PAD_MM:g} mm (input side; estimates), the plug {self.PLUG_MM:g} mm and the bend {self.BEND_MM:g} mm: the "
            f"arm cable runs level {self.CPL_EXIT:g} mm above the port, {room - self.CPL_EXIT:.0f} mm under the unit above "
            "(a 2U tray left 71 mm, too little). Confirm the part lengths; if short of room, fit a right-angle SMA "
            "adapter on each stack")

    def _face(self, dv):
        x, y, z = dv["pos"]
        w, h, dd = dv["size"]
        return x, y, z, w, h, dd

    def _ports_spider(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        box = next(bx for bx in self.d["boxes"] if bx["id"] == rid)
        used = {m: f"{s[0]}.{s[1]}" for s, (bx, m) in self.d["module"].items() if bx == rid}
        var = C.SPIDER_VARIANT.get(rid, "C")
        ysma = y + C.SPIDER_FRONT[var][1]
        dv["modules"] = [dict(m=f"M{k}", x0=C.spider_tile_x(rid, k, 1) - 9, x1=C.spider_tile_x(rid, k, 4) + 9,
                              sector=used.get(f"M{k}", "spare")) for k in range(1, 5)]
        dv["box_label"] = box["code"]
        dv["serial"] = BS.BOX_SERIAL.get(box["code"]) or C.SPIDER_SERIAL.get(rid, "")
        dv["variant"] = var
        for k in range(1, 5):
            for t in range(1, 5):
                self.add_port(rid, f"M{k}.T{t}", (C.spider_tile_x(rid, k, t), ysma, z - 10), (0, 0, -1), "SMA",
                              f"M{k} T{t}")
        R = C.spider_rear(rid)
        yr = y + R["y"]
        for nd, grp in zip(box["nodes"], ("node_a", "node_b")):
            g = R[grp]
            for nm, key, pt in (("ETH1", "eth1", "RJ45"), ("ETH2", "eth2", "RJ45"), ("SFP+", "sfp", "SFP"),
                                ("GNSS", "gnss", "SMA"), ("USB", "usb", "USB")):
                self.add_port(rid, f"{nd}.{nm}", (g[key], yr, z + dd), (0, 0, 1), pt, f"{nd} {nm}")
        self.add_port(rid, "PI.LAN", (R["pi"], yr, z + dd), (0, 0, 1), "RJ45", "box controller LAN (Raspberry Pi; use TBC)")
        self.add_port(rid, "DCIN", (R["dcin"], yr + 6, z + dd), (0, 0, 1), "DC",
                      "box DC input: green 4-pole pluggable header with screw flange (what it feeds TBC)")
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

    # tray layouts: parts lie flat, long axis along x, input end LEFT and output end RIGHT (seen from the front), all
    # the same way round. Splitters: 4 columns (tile 1-4 left to right) x one row per split module, front to back.
    # Couplers: 2 columns (module on the left half of its box -> left column) x 4 rows per module (tile 1-4 front to
    # back), CPL on the top face. Gutters between columns take the end connectors; a body never shares x/z space.
    # Cable lanes on a tray (so a rear-row cable never runs through a front-row plug): couplers: IN / THRU leave 18 mm
    # straight out, then rear row k moves LANE_X * k further out before running forward; the arm cable rises
    # CPL_EXIT above the CPL port (DC block + pad + plug + bend), then moves 12 + LANE_X * k to the right and runs
    # forward. Splitters: P2 runs 8 mm further out than P1; the rear row rises 10 mm over the front-row plugs.
    TRAY = {"splitter": dict(x0=51.3, xpitch=105.0, z0=10.0, zpitch=55.0, cols="tile"),
            "coupler": dict(x0=(86.0, 304.0), z0=10.0, zpitch=28.0, cols="side"),
            # if_pads: a pad stack (up to 62 mm) on each P1 / P2: one column per split module (175 mm wide: S plug,
            # splitter, stack, plug and the P2 lane), rows front to back = tile 1-4
            "splitter_pads": dict(x0=51.3, xpitch=175.0, z0=10.0, zpitch=55.0, cols="module")}
    LANE_X = 6.0
    PLUG_MM, BEND_MM = 18.0, 25.0                       # mated SMA plug length; cable bend before it runs level
    BLOCK_MM, PAD_MM = 30.0, 22.0                       # BLK-18-S+ and BW-S pad lengths (estimates)

    def tray_slots(self, rid):
        """{part id: (x0, z0)} for every splitter / coupler body on a tray (relative to the tray)."""
        b = self.b
        T = list(range(1, b.tiles + 1))
        out = OrderedDict()
        if b.split and b.if_pads:
            L = self.TRAY["splitter_pads"]
            for r, (sec, sid) in enumerate(b.split.items()):
                for t in T:
                    out[f"{sid}.t{t}"] = (L["x0"] + r * L["xpitch"], L["z0"] + (t - 1) * L["zpitch"])
        elif b.split:
            L = self.TRAY["splitter"]
            for r, (sec, sid) in enumerate(b.split.items()):
                for t in T:
                    out[f"{sid}.t{t}"] = (L["x0"] + (t - 1) * L["xpitch"], L["z0"] + r * L["zpitch"])
        else:
            L = self.TRAY["coupler"]
            rows = [0, 0]
            for sec, sid in b.coupled.items():
                bx, m = self.d["module"][sec]
                col = 0 if C.spider_tile_x(bx, int(m[1]), 1) < C.PANEL_W / 2 else 1
                for t in T:
                    out[f"{sid}.t{t}"] = (L["x0"][col], L["z0"] + rows[col] * L["zpitch"])
                    rows[col] += 1
        return out

    def arm_stack_mm(self, sec):
        """height of what stands on a coupler's CPL port: DC block (+ pad on the arm input side); 0 if nothing."""
        d = self.d
        for name, (x, z) in d["couplings"].items():
            if sec == x:
                return self.BLOCK_MM + self.PAD_MM
            if sec == z:
                return self.BLOCK_MM if d.get("arm_dc_blocks", 1) == 2 else 0.0
        return 0.0

    def arm_note(self, sec, bid):
        """what stands on a coupler's CPL port, per stack (input side: DC block + the arm's pad; output side: block)."""
        d, b = self.d, self.b
        for name, (x, z) in d["couplings"].items():
            pad = getattr(b, "arm_pad", {}).get(name, {})
            if sec == x:
                return (f"BLK-18-S+ DC block + {pad.get('pn', 'BW-S pad')} {pad.get('db', '?')} dB pad on the CPL port of "
                        f"{bid}, {self.BLOCK_MM + self.PAD_MM:g} mm (arm {name} input side; lengths estimated)")
            if sec == z:
                return (f"BLK-18-S+ DC block only on the CPL port of {bid}, {self.BLOCK_MM:g} mm (arm {name} output side; "
                        "length estimated)")
        return ""

    def coupler_sec(self, did):
        """the (site, sector) of a coupler part 'CPL-BE.t1' (None for other devices)."""
        sid = did.split(".")[0]
        return next((sec for sec, v in self.b.coupled.items() if v == sid), None)

    @property
    def CPL_EXIT(self):
        """height above the CPL port where every arm cable runs level: tallest stack + plug + bend."""
        hs = [self.arm_stack_mm(sec) for sec in self.b.coupled] or [0.0]
        return max(hs) + self.PLUG_MM + self.BEND_MM

    def _ports_tray(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        b = self.b
        kind = "splitter" if b.split else "coupler"
        s = C.SMALL[kind]
        sw, sh, sd = s["size"]
        L = self.TRAY["splitter_pads" if (b.split and b.if_pads) else kind]
        sec_of = {sid: sec for sec, sid in (b.split or b.coupled).items()}
        for bid, (bx0, bz0) in self.tray_slots(rid).items():
            self.add_dev(bid, kind, bid, s["note"], (x + bx0, y + 4, z + bz0), s["size"], s["color"],
                         "rack", "estimate", s["note"], parent=rid, orientation="input end left, output end right",
                         tray_row=int(round((bz0 - L["z0"]) / L["zpitch"])))
            yc, zc = y + 4 + sh / 2, z + bz0 + sd / 2
            if kind == "splitter":
                self.add_port(bid, "S", (x + bx0, yc, zc), (-1, 0, 0), "SMA", "S (left end)")
                self.add_port(bid, "P1", (x + bx0 + sw, yc, zc - 6), (1, 0, 0), "SMA", "port 1 (right end, front)")
                self.add_port(bid, "P2", (x + bx0 + sw, yc, zc + 6), (1, 0, 0), "SMA", "port 2 (right end, rear)")
            else:
                self.add_port(bid, "IN", (x + bx0, yc, zc), (-1, 0, 0), "SMA", "IN (left end)")
                self.add_port(bid, "THRU", (x + bx0 + sw, yc, zc), (1, 0, 0), "SMA", "THRU (right end)")
                self.add_port(bid, "CPL", (x + bx0 + sw / 2, y + 4 + sh, zc), (0, 1, 0), "SMA", "CPL (top)")
                sec = sec_of[bid.split(".")[0]]
                hs = self.arm_stack_mm(sec)
                if hs:                                     # DC block (+ pad) standing on the CPL port: a body
                    a = C.SMALL["armstack"]
                    aw, _, ad = a["size"]
                    two = hs > self.BLOCK_MM
                    sid = "ARM-" + bid[len("CPL-"):]
                    self.add_dev(sid, "armstack", sid, ("BLK-18-S+ DC block + arm pad" if two else "BLK-18-S+ DC block") +
                                 f" on the CPL port of {bid} ({hs:g} mm)", (x + bx0 + sw / 2 - aw / 2, y + 4 + sh, zc - ad / 2),
                                 (aw, hs, ad), a["color"], "rack", "estimate", self.arm_note(sec, bid), parent=bid,
                                 coupler=bid, arm_side="in" if two else "out")

    def _ports_shelf(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        b = self.b
        if rid == "PA-SHELF":
            # two rows on the 254 mm shelf (x from the left ear as seen from the FRONT): REAR row = the wired adaptors
            # in node order, left to right, then the AC injector(s); FRONT row = the spares, unwired, at the far right
            aw, ad = C.SMALL["adaptor"]["size"][0], C.SMALL["adaptor"]["size"][2]
            iw, idd = C.SMALL["injector"]["size"][0], C.SMALL["injector"]["size"][2]
            pitch, xl, xr = 45.0, 30.0, (C.PANEL_W + C.BODY_W) / 2 - 5
            cx = xl + aw / 2
            for nd in b.dc_nodes:
                self._small(b.adaptor[nd], "adaptor", cx, y, z + dd - ad - 5, rid, ("rj45", "RJ45"), ("dc", "DC"),
                            row="rear", feeds=nd)
                cx += pitch
            cx += iw / 2 - aw / 2 - (pitch - aw) + 10
            for nd, inj in b.inj.items():
                self._small(inj, "injector", cx, y, z + dd - idd - 1.5, rid, ("poe", "RJ45"), ("ac", "C6?"),
                            color="#bf616a", row="rear", feeds=nd)
                cx += iw + 10
            nsp = b.n_adaptors - len(b.dc_nodes)
            for j in range(nsp):
                a = f"PA-{len(b.dc_nodes) + j + 1:02d}"
                self._small(a, "adaptor", xr - aw / 2 - (nsp - 1 - j) * pitch, y, z + 4, rid, spare=True, row="front")
        elif rid == "AGENTS":
            ms = C.TABLE_DEV["minipc"]["size"]
            # two rows of three; the FRONT row is turned round (connectors to the front) so no connector faces the
            # back row 13 mm away: every connector keeps 35 mm or more of clear space
            for i, s in enumerate(self.d.get("agents", [])):
                aid = f"AGENT-{s}"
                col, row = i % 3, i // 3
                px, pz = 30 + col * 150, z + 10 + row * 125
                front = row == 0
                self.add_dev(aid, "minipc", aid, "MARL agent mini PC (2 NICs)" + (", turned round: connectors to the "
                                                                                  "FRONT" if front else ""),
                             (px, y + 4, pz), ms, C.TABLE_DEV["minipc"]["color"], "rack", "estimate",
                             "mini PC size estimate", parent=rid, row="front" if front else "rear",
                             orientation="connectors to the front" if front else "connectors to the rear")
                for j, nm in enumerate(("data", "mgmt", "ac")):
                    self.add_port(aid, nm, (px + 20 + j * 35, y + 30, pz if front else pz + ms[2]),
                                  (0, 0, -1) if front else (0, 0, 1), "RJ45" if nm != "ac" else "C5", nm)

    def _small(self, sid, kind, cx, y, z0, parent, *ports, color=None, spare=False, **kw):
        s = C.SMALL[kind]
        w, h, dd = s["size"]
        self.add_dev(sid, kind, sid, s["note"] + (" (spare, unwired)" if spare else ""), (cx - w / 2, y + 3, z0),
                     (w, h, dd), "#9aa5b1" if spare else (color or s["color"]), "rack", "estimate", s["note"],
                     parent=parent, spare=spare, **kw)
        for i, (nm, pt) in enumerate(ports):
            self.add_port(sid, nm, (cx, y + 3 + h * (0.72 if i == 0 else 0.3), z0 + dd), (0, 0, 1), pt, nm)

    def _ports_dcdist(self, dv, rid, n=10, pre="CB"):
        x, y, z, w, h, dd = self._face(dv)
        for i in range(1, n + 1):
            self.add_port(rid, f"{pre}{i}", (40 + (i - 1) * 36, y + h / 2, z + dd), (0, 0, 1), "DC", f"{pre}{i}")
        for cb, port in getattr(self.b, "dcin", {}).values():     # a breaker shared by two DCIN runs: 2nd conductor pair
            if port.endswith("b") and port.startswith(rid + "."):
                self.add_port(rid, port.split(".", 1)[1], (40 + (cb - 1) * 36, y + h / 2 + 8, z + dd), (0, 0, 1), "DC",
                              f"{pre}{cb} output, second run (twin ferrule in the same terminal)")
        self.add_port(rid, "bus", (440, y + h / 2, z + dd), (0, 0, 1), "DC", "bus")

    def _ports_fuse(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        for i in range(1, 11):
            xx = 82 + (i - 1) * 17 if i <= 5 else 334 + (i - 6) * 16.5
            self.add_port(rid, f"F{i}", (xx, y + 13, z - 5), (0, 0, -1), "DC", f"F{i} output (bus {'A' if i <= 5 else 'B'})")
        self.add_port(rid, "bus", (52, y + 22, z - 5), (0, 0, -1), "DC", "bus A input studs (from DC-RECT2)")
        self.add_port(rid, "strapA", (52, y + 8, z - 5), (0, 0, -1), "DC", "bus A strap studs (to bus B)")
        self.add_port(rid, "busB", (300, y + 15, z - 5), (0, 0, -1), "DC", "bus B input studs (strap from bus A)")
        self.add_port(rid, "rtn", (24, y + 15, z - 5), (0, 0, -1), "PE", "RTN bus earth stud (bond to the rack earth)")

    def _ports_rect(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        for k in range(1, BS.RECT_MODULES.get(rid, 1) + 1):          # one C14 inlet per fitted module
            self.add_port(rid, "ac" if k == 1 else f"ac{k}", (450 - (k - 1) * 22, y + h / 2, z + dd), (0, 0, 1), "C14",
                          f"AC in, module {k}")
        self.add_port(rid, "bus" if rid == "DC-RECT" else "out", (400, y + h / 2, z + dd), (0, 0, 1), "DC", "DC out")

    def _ports_pdu(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        for o in range(1, BS.PDU_OUT + 1):
            self.add_port(rid, f"O{o}", (40 + (o - 1) * 30, y + h / 2, z + dd), (0, 0, 1), "C13", f"O{o}")
        self.add_port(rid, "inlet", (462, y + h / 2, z + dd), (0, 0, 1), "IEC60309", "16 A inlet cord")
        self.add_port(rid, "net", (445, y + h * 0.7, z + dd), (0, 0, 1), "RJ45", "net (outlet face, rear)")

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
            px = 90 + col * 19 + (16 if p > BS.OOB_RJ45 else 0)
            self.add_port(rid, f"P{p}", (px, y + h * (0.66 - 0.32 * row), z + dd), (0, 0, 1),
                          "RJ45" if p <= BS.OOB_RJ45 else "SFP", f"P{p}")
        self.add_port(rid, "ac", (450, y + h / 2, z + dd), (0, 0, 1), "C14", "AC")

    def _ports_gnss(self, dv, rid):
        x, y, z, w, h, dd = self._face(dv)
        self.add_port(rid, "in", (60, y + h / 2, z + dd), (0, 0, 1), "SMA", "antenna in")
        for i in range(1, 9):
            self.add_port(rid, f"out{i}", (120 + (i - 1) * 30, y + h / 2, z + dd), (0, 0, 1), "SMA", f"out {i}")
        self.add_port(rid, "ac", (450, y + h / 2, z + dd), (0, 0, 1), "C14", "AC")

    # bench table -----------------------------------------------------------------------------------
    # plan (x from the rack-side end, z from the front edge; every body >= 5 mm from the next, stacks explicit):
    #   operator station at x 705-1300: keyboard at the front, monitor behind it, BENCH-CTL behind the monitor, the
    #   switched -48 V supplies (PoC 1) beside BENCH-CTL; the strip laid lengthwise (front to back) at x 1310;
    #   an aisle at x 1360-1495 for the cables of the A-side front-row radios;
    #   IP-50 radios in two rows (front / back, 120 mm apart) left (A side) and right (B side) of MWATT, the third and
    #   fourth hop STACKED on the first two on 25 mm spacer bars; MWATT and the analyser 90 mm in from the front edge;
    #   the back 105 mm of the table is kept free for the cable lanes (data in front, power at the very back).
    TABLE_PLAN = dict(radio_rows=(-145.0, 245.0), radio_xa=1500.0, mwatt_x=2030.0, mwatt_z=-60.0, radio_gap_b=290.0,
                      specan_x=3330.0, monitor=(730.0, 40.0), ctl=(720.0, 150.0), psu=(930.0, 150.0, 180.0),
                      strip=(1310.0, 40.0), kbd=(780.0, -140.0))
    LANES = dict(data=(540.0, 580.0), power=(620.0, 650.0))       # z bands on the table (back edge at 650)
    # mw_pads (no MWATT): the two radios of a hop side by side, flanges facing; hops in a 2 x 2 grid (front / back row,
    # two columns), none stacked; chain_gap = between the two pad stacks' OUT ports (one SMA cable, about 0.3 m)
    TABLE_PADS = dict(radio_rows=(-145.0, 245.0), radio_x0=1500.0, chain_gap=180.0, aisle=150.0, specan_gap=40.0)

    def _table(self):
        b, d, T = self.b, self.d, C.TABLE
        top = T["top"]
        self.table = dict(T)
        self.table["lanes"] = dict(self.LANES)
        TP = self.TABLE_PLAN
        for sid, role, site in d["servers"]:
            if sid in BS.TABLE_SERVERS:
                self._tdev(sid, "pc", role, (TP["ctl"][0], top, TP["ctl"][1]), C.TABLE_DEV["pc"]["size"], ("usb", "USB"),
                           ("video", "DP"), ("mgmt", "RJ45"), ("ac", "C14"), fracs=(0.12, 0.30, 0.45, 0.88))
        self._tdev("MONITOR", "monitor", "24-inch monitor (operator station)", (TP["monitor"][0], top, TP["monitor"][1]),
                   C.TABLE_DEV["monitor"]["size"], ("in", "DP"), ("ac", "C14"), fracs=(0.10, 0.66))
        self._tdev("KBD", "keyboard", "keyboard + mouse", (TP["kbd"][0], top, TP["kbd"][1]), C.TABLE_DEV["keyboard"]["size"],
                   ("usb", "USB"))
        hops = list(d.get("mw_links", {}).items())
        mx, mz = TP["mwatt_x"], TP["mwatt_z"]
        specan_x, earth_cx = TP["specan_x"], mx + C.TABLE_DEV["mwatt"]["size"][0] / 2
        if hops and not b.mwatt:
            x_end = self._hop_pairs(hops, top)
            specan_x = max(specan_x, x_end + self.TABLE_PADS["specan_gap"])
            earth_cx = (self.TABLE_PADS["radio_x0"] + x_end) / 2
        if hops and b.mwatt:
            ms = C.TABLE_DEV["mwatt"]["size"]
            self._tdev("MWATT", "mwatt", "Keysight J7204B, 4 ch", (mx, top, mz), ms, ("lan", "RJ45"), ("ac", "C14"))
            for j, (k, (a, bb)) in enumerate(hops):
                for side in ("in", "out"):
                    self.add_port("MWATT", f"ch{j + 1}.{side}", (mx + 150 + j * 68 + (0 if side == "in" else 30), top + 44,
                                                                 mz), (0, 0, -1), "SMA", f"CH{j + 1} {side}")
            rs = C.TABLE_DEV["ip50"]["size"]
            rows = TP["radio_rows"]
            sp = C.SMALL["spacer"]["size"]
            for j, (k, (a, bb)) in enumerate(hops):
                zr = rows[j % len(rows)]
                lvl = j // len(rows)
                yr = top + lvl * (rs[1] + BS.SPACER_MM)        # stacked radios sit on 25 mm spacer bars
                for side, site, xr, face in (("A", a, TP["radio_xa"], +1), ("B", bb, mx + ms[0] + TP["radio_gap_b"], -1)):
                    rid = f"IP50-{k}@{site}"
                    under = None
                    if lvl:
                        k0 = hops[j - len(rows)][0]
                        under = f"IP50-{k0}@{hops[j - len(rows)][1][0 if side == 'A' else 1]}"
                    self._tdev(rid, "ip50", f"IP-50, hop {k} at {site}" + (f", stacked on {under}" if under else ""),
                               (xr, yr, zr), rs, ("data", "RJ45"), ("mgmt", "RJ45"), ("dc", "DC"))
                    self.add_port(rid, "gnd", (xr + rs[0] * 0.92, yr + 20, zr + rs[2]), (0, 0, 1), "PE",
                                  "chassis ground screw (position TBC)")
                    self.dev[rid]["stack"] = under or rid
                    if under:
                        self.dev[under]["stack"] = under
                        spc = []
                        for fr, zz in (("F", zr + 5), ("R", zr + rs[2] - 5 - sp[2])):
                            sid = f"SPC-{k}@{site}-{fr}"
                            self.add_dev(sid, "spacer", sid, f"spacer bar under {rid} ({'front' if fr == 'F' else 'rear'} "
                                         f"edge of {under}'s lid)", (xr + (rs[0] - sp[0]) / 2, yr - BS.SPACER_MM, zz), sp,
                                         C.SMALL["spacer"]["color"], "table", "estimate", C.SMALL["spacer"]["note"],
                                         parent=under, stack=under)
                            spc.append(sid)
                        self.dev[rid].update(stacked_on=under, spacer_mm=BS.SPACER_MM, spacers=spc)
                        self.dev[rid]["note"] += (f" Stacked on {under} on two {BS.SPACER_MM:g} mm spacer bars "
                                                  f"({', '.join(spc)}) for a convection gap.")
                    wx = xr + rs[0] if face > 0 else xr
                    self.add_port(rid, "wg", (wx, yr + rs[1] / 2, zr + rs[2] / 2), (face, 0, 0), "WG", "waveguide")
                    ts = C.SMALL["transition"]["size"]
                    tid = f"WGT-{k}-{side}"
                    tx0 = wx if face > 0 else wx - ts[0]
                    self.add_dev(tid, "transition", tid, C.SMALL["transition"]["note"],
                                 (tx0, yr + rs[1] / 2 - ts[1] / 2, zr + rs[2] / 2 - ts[2] / 2), ts,
                                 C.SMALL["transition"]["color"], "table", "estimate", C.SMALL["transition"]["note"],
                                 parent=rid)
                    self.add_port(tid, "wg", (wx, yr + rs[1] / 2, zr + rs[2] / 2), (-face, 0, 0), "WG", "flange")
                    self.add_port(tid, "coax", (wx + face * ts[0], yr + rs[1] / 2, zr + rs[2] / 2), (face, 0, 0),
                                  "SMA", "coax + 30 dB pad")
        px, pz, pp = TP["psu"]
        for i, p in enumerate(b.psu.values()):
            self._tdev(p, "psu", "isolated -48 V supply (floating), switched at t0 (operator end of the table)",
                       (px + i * pp, top, pz), C.TABLE_DEV["psu"]["size"], ("out", "DC"), ("ac", "C14"))
        self._tdev("SPECAN", "specan", "spectrum analyser / power sensor", (specan_x, top, mz),
                   C.TABLE_DEV["specan"]["size"], ("lan", "RJ45"), ("ac", "C14"))
        # the strip lies lengthwise (front to back): outlets on top, C20 inlet at its back end toward the power lane
        sl, sw_, sh = C.TABLE_DEV["strip"]["size"]
        stx, stz = TP["strip"]
        self.add_dev("TABLE-STRIP", "strip", "TABLE-STRIP", f"bench mains strip, {BS.STRIP_OUT} x C13 outlets, C20 inlet, "
                     "on/off switch (own wall socket)", (stx, top, stz), (sw_, sh, sl), C.TABLE_DEV["strip"]["color"], "table",
                     "estimate", C.TABLE_DEV["strip"]["note"], orientation="lengthwise, inlet at the back end")
        n = len(b.table_loads)
        for i in range(1, BS.STRIP_OUT + 1):
            self.add_port("TABLE-STRIP", f"O{i}", (stx + sw_ / 2, top + sh, stz + 50 + (i - 1) * (sl - 100) / (BS.STRIP_OUT - 1)),
                          (0, 1, 0), "C13", f"O{i}" + ("" if i <= n else " (spare)"))
        self.add_port("TABLE-STRIP", "inlet", (stx + sw_ / 2, top + sh / 2, stz + sl), (0, 0, 1), "C20", "C20 inlet")
        self._room()
        if b.ip50:                                         # bench-table earth bar on the back edge, behind MWATT
            nr = len(b.ip50)                               # (no MWATT: behind the aisle between the radio columns)
            L = 40 + 22 * nr
            cx = earth_cx
            x0, y0, z0 = cx - L / 2, top - 35, C.TABLE["z1"] + 2
            self.add_dev("TABLE-EARTH", "earthbar", "TABLE-EARTH", f"bench-table earth bar, {nr + 1} M6 studs",
                         (x0, y0, z0), (L, 30, 10), "#2e7d32", "room", "estimate",
                         "earth bar screwed to the bench table's back edge behind " +
                         ("MWATT" if b.mwatt else "the aisle between the radio columns") +
                         ", studs up just below the table top (estimate)", mount="table back edge")
            self.add_port("TABLE-EARTH", "bond", (x0 + 15, y0, z0 + 5), (0, -1, 0), "PE", "stud to the rack earth bar")
            for i in range(1, nr + 1):
                self.add_port("TABLE-EARTH", f"R{i}", (x0 + 31 + (i - 1) * 22, y0 + 30, z0 + 5), (0, 1, 0), "PE",
                              f"radio stud {i}")

    def _hop_pairs(self, hops, top):
        """mw_pads: each hop's two radios side by side, flanges facing (A on the left), the transitions on the flanges
        and the pad stacks on them (_pads); hop j in row j % 2 (front / back) and column j // 2. Returns the right end x
        of the radios."""
        TP = self.TABLE_PADS
        rs = C.TABLE_DEV["ip50"]["size"]
        ts = C.SMALL["transition"]["size"]
        b = self.b
        stack = {sd: max(v["stack_mm"] for v in b.pads.values() if v["chain"] == "ip50" and v["side"] == sd.lower())
                 for sd in "AB"}
        W = 2 * rs[0] + 2 * ts[0] + stack["A"] + stack["B"] + TP["chain_gap"]
        x_end = TP["radio_x0"]
        for j, (k, (a, bb)) in enumerate(hops):
            zr = TP["radio_rows"][j % len(TP["radio_rows"])]
            xa = TP["radio_x0"] + (j // len(TP["radio_rows"])) * (W + TP["aisle"])
            for side, site, xr, face in (("A", a, xa, +1), ("B", bb, xa + W - rs[0], -1)):
                rid = f"IP50-{k}@{site}"
                self._tdev(rid, "ip50", f"IP-50, hop {k} at {site} ({side} side, flange toward its peer)", (xr, top, zr),
                           rs, ("data", "RJ45"), ("mgmt", "RJ45"), ("dc", "DC"),
                           note=C.TABLE_DEV["ip50"]["note"].replace("toward MWATT", "toward the other radio of its hop"))
                self.add_port(rid, "gnd", (xr + rs[0] * 0.92, top + 20, zr + rs[2]), (0, 0, 1), "PE",
                              "chassis ground screw (position TBC)")
                self.dev[rid].update(stack=rid, hop=k, hop_side=side)
                wx = xr + rs[0] if face > 0 else xr
                self.add_port(rid, "wg", (wx, top + rs[1] / 2, zr + rs[2] / 2), (face, 0, 0), "WG", "waveguide")
                tid = f"WGT-{k}-{side}"
                tx0 = wx if face > 0 else wx - ts[0]
                self.add_dev(tid, "transition", tid, C.SMALL["transition"]["note"],
                             (tx0, top + rs[1] / 2 - ts[1] / 2, zr + rs[2] / 2 - ts[2] / 2), ts,
                             C.SMALL["transition"]["color"], "table", "estimate", C.SMALL["transition"]["note"], parent=rid)
                self.add_port(tid, "wg", (wx, top + rs[1] / 2, zr + rs[2] / 2), (-face, 0, 0), "WG", "flange")
                self.add_port(tid, "coax", (wx + face * ts[0], top + rs[1] / 2, zr + rs[2] / 2), (face, 0, 0), "SMA",
                              "coax (SMA-F): the pad stack screws on here")
                x_end = max(x_end, xr + rs[0])
        return x_end

    def _pads(self):
        """fixed pad stacks (bench_spec if_pads / mw_pads): a body from its host port along the port's normal, IN screwed
        onto the host (port field 'mate' on both: a joint with no cable), OUT at the far end, where the cable goes on.
        Device fields: host, chain ('if' / 'ip50'), link, tile, pads [dict(pn, db, len_mm, dia_mm)] in stack order (host
        end first), db (fitted), target_db / rounded_db (IF), stack_mm."""
        b = self.b
        s = C.SMALL["padstack"]
        for pid, v in b.pads.items():
            H = self.ports[v["host"]]
            hd = self.dev[H["device"]]
            p, n = H["pos"], H["normal"]
            L, D = v["stack_mm"], v["dia_mm"]
            lo = [p[i] - D / 2 if n[i] == 0 else (p[i] if n[i] > 0 else p[i] - L) for i in range(3)]
            size = [D if n[i] == 0 else L for i in range(3)]
            pads = " + ".join(f"{x['pn']} ({x['db']} dB)" for x in v["parts"])
            if v["chain"] == "if":
                where = (f"on {hd['id']} {H['name']}, lying on the splitter tray" if hd["kind"] == "splitter" else
                         f"on tile {H['name']} of {hd['id']}, {L:g} mm proud of the box front")
                val = f"{v['text']} = {v['db']} dB" if len(v["parts"]) > 1 else f"{v['db']} dB"
                sub = f"IF link {v['link']} tile {v['tile']}: {val} (target {v['target']:g} dB) {where}"
                support = ("fitted onto the splitter in the hand before it goes on the tray (splitter-tray step), and its "
                           "cable's tray end torqued onto its OUT in the hand in the same step; lies on the tray"
                           if hd["kind"] == "splitter" else
                           "cable strapped to the seam bundle at the unit's lower edge (or the cable manager) within "
                           "150 mm of the stack, then to the side lane, so the tile carries only the stack")
            else:
                val = f"{v['text']} = {v['db']} dB" if len(v["parts"]) > 1 else f"{v['db']} dB"
                sub = f"IP-50 hop {v['link']}, {v['side'].upper()} side: {val} on {H['device']}"
                support = "carried by the transition, which is bolted to the radio flange"
            note = (f"{pads} screwed together, IN onto {v['host']}; about {L:g} mm long, {D:g} mm across (estimate)")
            kw = {}
            if hd["kind"] == "splitter":                  # keeps its splitter port's lane on the tray
                kw = dict(tray_row=hd.get("tray_row", 0), lane_dx=8 if H["name"] == "P2" else 0)
            self.add_dev(pid, "padstack", pid, sub, lo, size, s["color"], hd["group"], "estimate", note,
                         parent=hd["id"], host=v["host"], chain=v["chain"], link=v["link"], tile=v["tile"],
                         side=v["side"], pads=[dict(x) for x in v["parts"]], db=v["db"], target_db=v["target"],
                         rounded_db=v["rounded"], text=v["text"], stack_mm=L, support=support, **kw)
            self.add_port(pid, "in", p, [-x if x else 0 for x in n], "SMA", f"IN (SMA-M, screwed onto {v['host']})")
            self.add_port(pid, "out", [p[i] + n[i] * L for i in range(3)], list(n), "SMA", "OUT (SMA-F, the cable)")
            self.ports[f"{pid}.in"]["mate"] = v["host"]
            H["mate"] = f"{pid}.in"

    def _room(self):
        b = self.b
        rd = C.RACK["rail_depth"]
        # wall sockets (one per PDU + one for the bench strip, each on its own circuit) and the lab earth bar
        nw = len(b.pdus) + 1
        wx0, wpitch = 60.0, 90.0
        self.add_dev("WALL", "wall", "Wall sockets IEC 60309 16 A", f"lab wall, {nw} sockets", (wx0, 300, rd + 260),
                     (100 + (nw - 1) * wpitch, 120, 40), "#7f8c8d", "room", "estimate",
                     f"{nw} wall sockets IEC 60309 16 A behind the rack, one circuit each: S1-S{nw - 1} for the PDUs, "
                     f"S{nw} for the bench strip (estimate)")
        for i in range(1, nw + 1):
            self.add_port("WALL", f"S{i}", (wx0 + 50 + (i - 1) * wpitch, 360, rd + 260), (0, 0, -1),
                          "IEC60309", f"socket {i}" + (" (bench strip)" if i == nw else ""))
        self.add_dev("LAB-EARTH", "earthbar", "LAB-EARTH", "lab earth bar on the wall (position TBC)", (-260, 320, rd + 260),
                     (150, 30, 20), "#2e7d32", "room", "estimate", "lab earth bar on the wall behind the rack (estimate)")
        self.add_port("LAB-EARTH", "bar", (-185, 335, rd + 260), (0, 0, -1), "PE", "earth stud")
        self.add_port("LAB-EARTH", "bar2", (-210, 350, rd + 270), (0, 1, 0), "PE", "earth stud 2 (GNSS arrestor)")
        e = C.SMALL["earthbar"]
        self.add_dev("RACK-EARTH", "earthbar", "RACK-EARTH", "rack earth bar, left rear post (grounding kit)",
                     (-30, 160, rd + 5), e["size"], e["color"], "room", "estimate", e["note"])
        for i in range(1, 5):
            self.add_port("RACK-EARTH", f"E{i}", (-15, 180 + (i - 1) * 45, rd + 15), (0, 0, 1), "PE", f"stud {i}")
        # GNSS: the roof antenna's lead comes down behind the wall and enters the room at the surge arrestor, which sits
        # just above the lab earth bar so its bond is short and straight
        self.add_dev("GNSS-ANT", "antenna", "GNSS-ANT", "101898-00 roof antenna; its LMR-400 lead comes down behind the "
                     "wall to the arrestor", (-310, 2350, rd + 300), (40, 90, 40), "#27ae60", "room", "estimate",
                     "GNSS roof antenna: the lead's way down to the room entry is TBC with facilities (estimate)")
        self.add_port("GNSS-ANT", "", (-290, 2350, rd + 320), (0, -1, 0), "N", "antenna lead")
        a = C.SMALL["arrestor"]
        self.add_dev("GNSS-ARR", "arrestor", "GNSS-ARR", a["note"], (-250, 440, rd + 235), a["size"], a["color"], "room",
                     "estimate", "GNSS surge arrestor where the antenna lead enters the room, on the wall just above the "
                     "lab earth bar (position TBC)")
        self.add_port("GNSS-ARR", "ant", (-250, 460, rd + 247), (-1, 0, 0), "N", "antenna side (N-F)")
        self.add_port("GNSS-ARR", "eqp", (-170, 460, rd + 247), (1, 0, 0), "N", "equipment side (N-F)")
        self.add_port("GNSS-ARR", "gnd", (-210, 440, rd + 247), (0, -1, 0), "PE", "ground lug")
        # rack-to-table cable bridge: its floor at table height, clear of the rack-side drops (x <= 557)
        g = C.SMALL["bridge"]
        self.add_dev("CABLE-BRIDGE", "bridge", "CABLE-BRIDGE", g["note"], (C.TABLE["x0"] - g["size"][0], C.TABLE["top"] -
                     g["size"][1], C.TABLE["z1"] - g["size"][2] + 5), g["size"], g["color"], "room", "estimate",
                     "rack-to-table cable bridge (wire-basket section, part TBC)")

    def _tdev(self, did, kind, sub, pos, size, *ports, fracs=None, note=None):
        """a table device with its ports on the rear face, evenly spread (or at the given fractions of its width:
        the bench PC keeps its mains inlet 75 mm or more from its data ports)."""
        t = C.TABLE_DEV[kind]
        self.add_dev(did, kind, did, sub, pos, size, t["color"], "table", t["conf"], note or t["note"])
        x, y, z = pos
        w, h, dd = size
        ps = [p for p in ports if p]
        for i, (nm, pt) in enumerate(ps):
            f = fracs[i] if fracs else (i + 1) / (len(ps) + 1)
            self.add_port(did, nm, (x + w * f, y + h / 2, z + dd), (0, 0, 1), pt, nm)

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

    def top_dev(self, did):
        """the rack unit (or table item) a part belongs to."""
        while self.dev[did].get("parent"):
            did = self.dev[did]["parent"]
        return self.dev[did]

    def side_x(self, side, lane):
        """x of the front IF side lane: just outside the rack ears, left or right (seen from the front)."""
        return -18 - 3 * (lane % 6) if side == "left" else C.PANEL_W + 18 + 3 * (lane % 6)

    def unit_below(self, dv):
        ub = dv.get("u_top", 0) - dv.get("u_n", 1)
        return next((v for v in self.dev.values() if v["group"] == "rack" and not v.get("parent") and
                     v.get("u_top") == ub), None)

    def tile_y(self, dv):
        """height of the SMA tile row of a Spider box (None for other units)."""
        if dv["kind"] != "spider":
            return None
        return dv["pos"][1] + C.SPIDER_FRONT[dv.get("variant", "C")][1]

    def lower_edge_y(self, dv, lane):
        """height of the IF run along the lower edge of a unit: for a Spider box, on the 1 mm seam between it and the
        unit below (as far from its own tile row and the next box's tile row as the faces allow, and across neither
        face); other units: 2-5 mm above their bottom."""
        if dv["kind"] == "spider":
            return dv["pos"][1] - 0.5
        return dv["pos"][1] + 2 + 1.5 * (lane % 3)

    def exits(self, P):
        """the first points of a cable at a port: straight out, then (on a tray) the part's own lane, so a rear-row
        cable never runs through the plug of a part in front of it. Ends where the cable can run forward freely."""
        p, n = P["pos"], P["normal"]
        if n[2] < -0.5:
            return [p]
        dv = self.dev[P["device"]]
        k = dv.get("tray_row", 0)
        if n[1] > 0.5:                                     # coupler CPL (top): over the DC block / pad, then aside
            ey = p[1] + self.CPL_EXIT
            sec = self.coupler_sec(P["device"])
            hs = self.arm_stack_mm(sec) if sec else 0.0
            # the cable's own plug sits on top of the DC block (+ pad): the route keeps its end on the port (schema) and
            # its first segment is the stack itself (cable field 'stack' names that body, so tools can skip it)
            top = [[p[0], p[1] + hs, p[2]]] if hs else []
            return [p] + top + [[p[0], ey, p[2]], [p[0] + 12 + self.LANE_X * k, ey, p[2]]]
        e = [p[0] + n[0] * self.PLUG_MM, p[1] + n[1] * self.PLUG_MM, p[2] + n[2] * self.PLUG_MM]
        pts = [p, e]
        if dv["kind"] == "coupler" and k:
            pts.append([p[0] + n[0] * (self.PLUG_MM + self.LANE_X * k), p[1], p[2]])
        elif dv["kind"] == "splitter" or (dv["kind"] == "padstack" and dv.get("tray_row") is not None):
            # a pad stack on a splitter port keeps that port's lane: P2 8 mm further out, the rear rows rise
            dx = dv.get("lane_dx", 0) if dv["kind"] == "padstack" else (8 if P["name"] == "P2" else 0)
            if dx:
                pts.append([p[0] + n[0] * (self.PLUG_MM + dx), p[1], p[2]])
            if k:
                q = pts[-1]
                pts.append([q[0], q[1] + 10 * k, q[2]])
        return pts

    def rf_front(self, P0, P1, lane, side):
        """IF cable between two front-accessible rack ports: out to the lane plane in front of the rack, down (or up) to
        the lower edge of its own unit, along that edge to the side lane, along the side lane, in along the lower edge
        of the other unit and up to the port. It never hangs across the face of another unit, so every tile stays
        reachable for the torque wrench."""
        zl = -(25 + 5 * (lane % 12))
        front = [P["pos"][2] for P in (P0, P1) if P["normal"][2] < -0.5]
        if front:                                          # a pad stack on a tile: its OUT stands proud of the tile row
            zl -= max(0.0, -10.0 - min(front))
        h0, h1 = self.exits(P0), self.exits(P1)
        e0, e1 = h0[-1], h1[-1]
        t0, t1 = self.top_dev(P0["device"]), self.top_dev(P1["device"])
        if t0["id"] == t1["id"]:
            return h0 + [[e0[0], e0[1], zl], [e1[0], e1[1], zl]] + h1[::-1]
        yb0, yb1 = self.lower_edge_y(t0, lane), self.lower_edge_y(t1, lane)
        xs = self.side_x(side, lane)
        return h0 + [[e0[0], e0[1], zl], [e0[0], yb0, zl], [xs, yb0, zl], [xs, yb1, zl], [e1[0], yb1, zl],
                     [e1[0], e1[1], zl]] + h1[::-1]

    def rack_route(self, P0, P1, cls, lane):
        """non-IF cable inside the rack: power dressed on the LEFT side (x < 0), data on the RIGHT (x > panel width),
        seen from the front; verticals never run inside the rack footprint, so they cannot pass through a unit."""
        p0, n0, p1, n1 = P0["pos"], P0["normal"], P1["pos"], P1["normal"]
        power = cls in C.POWER_CLS
        so = (25 if power else 95) + 4 * (lane % 6)        # power and data runs 50 mm or more apart behind / in front
        sx = -35 - 3 * (lane % 8) if power else C.PANEL_W + 35 + 3 * (lane % 8)
        same_u = abs(p0[1] - p1[1]) < 50 and self.top_dev(P0["device"])["id"] == self.top_dev(P1["device"])["id"]
        f0, f1 = n0[2] < -0.5, n1[2] < -0.5
        if f0 and f1:                                       # front to front: in front of the rack
            zl = min(p0[2], p1[2]) - so
            if same_u:
                return [p0, [p0[0], p0[1], zl], [p1[0], p0[1], zl], [p1[0], p1[1], zl], p1]
            return [p0, [p0[0], p0[1], zl], [sx, p0[1], zl], [sx, p1[1], zl], [p1[0], p1[1], zl], p1]
        def back(p):
            """depth of the run behind a rear port: power just behind the unit; data 50 mm further back, and behind
            the deep units (rear face beyond 700 mm) always 95 mm or more behind the rear rail, clear of the power
            runs to the rear-post PDUs and SW-OOB"""
            z = p[2] + so
            return max(z, C.RACK["rail_depth"] + so) if (not power and p[2] > 700) else z
        if f0 != f1:                                        # front <-> rear: round the side of the rack
            a = [p0[0] + n0[0] * so, p0[1] + n0[1] * so, p0[2] + n0[2] * so]
            q = [p1[0] + n1[0] * so, p1[1] + n1[1] * so, p1[2] + n1[2] * so]
            a[2] = back(p0) if n0[2] > 0 else a[2]
            q[2] = back(p1) if n1[2] > 0 else q[2]
            return [p0, a, [sx, a[1], a[2]], [sx, q[1], a[2]], [sx, q[1], q[2]], q, p1]
        if same_u:                                          # rear to rear in one unit: behind it
            zl = max(p0[2], p1[2]) + so
            return [p0, [p0[0], p0[1], zl], [p1[0], p0[1], zl], [p1[0], p1[1], zl], p1]
        # rear to rear: each end is approached just behind its own unit; the change of depth and height happens in
        # the side lane, outside the rack footprint
        z0, z1 = back(p0), back(p1)
        return [p0, [p0[0], p0[1], z0], [sx, p0[1], z0], [sx, p0[1], z1], [sx, p1[1], z1], [p1[0], p1[1], z1], p1]

    # bench-table routing: cables lie on the table top and find their way on a 10 mm grid around every body
    GRID, INFL, TURN = 10.0, 6.0, 30.0

    def lane_z(self, cls, lane):
        lo, hi = self.LANES["power" if cls in C.POWER_CLS else "data"]
        cells = [z for z in self._grid_z() if lo <= z <= hi]
        return cells[lane % len(cells)] if cls in C.POWER_CLS else cells[::-1][lane % len(cells)]

    def _grid_z(self):
        T = C.TABLE
        return [T["z0"] + self.GRID / 2 + j * self.GRID for j in range(int((T["z1"] - T["z0"]) / self.GRID))]

    def _grid(self):
        if getattr(self, "_g", None) is not None:
            return self._g
        T, G, I = C.TABLE, self.GRID, self.INFL
        nx, nz = int((T["x1"] - T["x0"]) / G), int((T["z1"] - T["z0"]) / G)
        blk = bytearray(nx * nz)
        for v in self.dev.values():
            if v["group"] != "table":
                continue
            x0, _, z0 = v["pos"]
            w, _, dd = v["size"]
            i0, i1 = int(math.floor((x0 - I - T["x0"]) / G)), int(math.ceil((x0 + w + I - T["x0"]) / G))
            j0, j1 = int(math.floor((z0 - I - T["z0"]) / G)), int(math.ceil((z0 + dd + I - T["z0"]) / G))
            for i in range(max(0, i0), min(nx, i1)):
                cx = T["x0"] + G / 2 + i * G
                if not (x0 - I < cx < x0 + w + I):
                    continue
                for j in range(max(0, j0), min(nz, j1)):
                    cz = T["z0"] + G / 2 + j * G
                    if z0 - I < cz < z0 + dd + I:
                        blk[i * nz + j] = 1
        self._g = (nx, nz, blk)
        return self._g

    def _cell(self, x, z):
        T, G = C.TABLE, self.GRID
        nx, nz, _ = self._grid()
        return (min(nx - 1, max(0, int((x - T["x0"]) // G))), min(nz - 1, max(0, int((z - T["z0"]) // G))))

    def _xz(self, c):
        T, G = C.TABLE, self.GRID
        return T["x0"] + G / 2 + c[0] * G, T["z0"] + G / 2 + c[1] * G

    def table_path(self, a, b, cls):
        """shortest grid path on the table top from (x, z) a to b, around every body, with a cost for turns and for
        running in the other bundle's lane (power and data stay 50 mm or more apart along the back edge)."""
        nx, nz, blk = self._grid()
        power = cls in C.POWER_CLS
        (dlo, dhi), (plo, phi) = self.LANES["data"], self.LANES["power"]

        def wcell(j):
            z = C.TABLE["z0"] + self.GRID / 2 + j * self.GRID
            if z < dlo:                                    # among the equipment: short stubs, not long runs
                return 1.5
            if power:
                return 4.0 if z <= dhi else (2.0 if z < plo else 1.0)
            return 1.0 if z <= dhi else (2.0 if z < plo else 4.0)
        W = [wcell(j) for j in range(nz)]
        zdev = int((dlo - C.TABLE["z0"]) // self.GRID)      # rows below the lanes
        near = self._occ["data" if power else "power"]      # cells within 50 mm of the other bundle's cables
        xop = int((self.TABLE_PLAN["strip"][0] - C.TABLE["x0"]) // self.GRID)
        zop = int((self.TABLE_PLAN["monitor"][1] - C.TABLE["z0"]) // self.GRID)
        s, g = self._cell(*a), self._cell(*b)
        for c in (s, g):
            if blk[c[0] * nz + c[1]]:
                raise ValueError(f"table route end inside a body at {self._xz(c)} ({cls})")
        D = ((1, 0), (-1, 0), (0, 1), (0, -1))
        G = self.GRID
        start = (s, -1)
        best = {start: 0.0}
        prev = {}
        hq = [(0.0, 0.0, start)]
        end = None
        while hq:
            f, gc, st = heapq.heappop(hq)
            if gc > best.get(st, 1e18):
                continue
            (ci, cj), dr = st
            if (ci, cj) == g:
                end = st
                break
            for k, (di, dj) in enumerate(D):
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < nx and 0 <= nj < nz) or blk[ni * nz + nj]:
                    continue
                m = W[nj]
                if nj < zdev and (ni, nj) in near:             # keep power and data apart among the equipment
                    m *= 3.0
                if ni < xop and nj < zop:                       # the operator's keyboard space
                    m *= 4.0
                ng = gc + G * m + (self.TURN if dr not in (-1, k) else 0.0)
                nst = ((ni, nj), k)
                if ng < best.get(nst, 1e18):
                    best[nst] = ng
                    prev[nst] = st
                    h = G * (abs(ni - g[0]) + abs(nj - g[1]))
                    heapq.heappush(hq, (ng + h, ng, nst))
        if end is None:
            raise ValueError(f"no table route from {a} to {b} ({cls})")
        cells = [end[0]]
        st = end
        while st in prev:
            st = prev[st]
            cells.append(st[0])
        cells = cells[::-1]
        mine = self._occ["power" if power else "data"]
        R = 5
        for (ci, cj) in cells:
            for di in range(-R, R + 1):
                for dj in range(-R, R + 1):
                    mine.add((ci + di, cj + dj))
        pts = [self._xz(c) for c in cells]
        out = [pts[0]]
        for i in range(1, len(pts) - 1):                   # keep the corners only
            if (pts[i - 1][0] == pts[i][0] == pts[i + 1][0]) or (pts[i - 1][1] == pts[i][1] == pts[i + 1][1]):
                continue
            out.append(pts[i])
        if len(pts) > 1:
            out.append(pts[-1])
        return out

    def table_end(self, P, lane):
        """from a table port to the table top: out 25 mm, (top-face ports: up and to the side), down to the cable
        plane. Returns the 3D points and the (x, z) where the grid path starts."""
        p, n = P["pos"], P["normal"]
        dv = self.dev[P["device"]]
        yc = C.TABLE["top"] + 3 + 1.5 * (lane % 3)
        if n[1] > 0.5:                                      # strip outlets: up, then off the strip's side
            up = [p[0], p[1] + 25, p[2]]
            x0, w = dv["pos"][0], dv["size"][0]
            sx = x0 - 15 if self._toward_left(P) else x0 + w + 15
            pts = [p, up, [sx, up[1], p[2]], [sx, yc, p[2]]]
        else:
            e = [p[0] + n[0] * 25, p[1], p[2] + n[2] * 25]
            pts = [p, e, [e[0], yc, e[2]]]
        q = pts[-1]
        c = self._xz(self._cell(q[0], q[2]))
        if abs(c[0] - q[0]) > 0.05 or abs(c[1] - q[2]) > 0.05:  # onto the grid: along the normal first
            if abs(n[0]) > 0.5:
                pts += [[c[0], yc, q[2]], [c[0], yc, c[1]]]
            else:
                pts += [[q[0], yc, c[1]], [c[0], yc, c[1]]]
        return pts, c, yc

    def _toward_left(self, P):
        """strip outlets: which side of the strip the lead leaves (toward its load)."""
        c = next((c for c in getattr(self, "_pending", []) if P["id"] in (c[0], c[1])), None)
        if not c:
            return True
        other = self.ports[c[1] if c[0] == P["id"] else c[0]]
        return other["pos"][0] < self.dev[P["device"]]["pos"][0]

    def route(self, P0, P1, cls, lane, side=None):
        p0, n0 = P0["pos"], P0["normal"]
        p1, n1 = P1["pos"], P1["normal"]
        d0, d1 = self.dev[P0["device"]], self.dev[P1["device"]]
        g0, g1 = d0["group"], d1["group"]
        so = 25 + 5 * (lane % 12)

        def off(p, n, s):
            return [p[0] + n[0] * s, p[1] + n[1] * s, p[2] + n[2] * s]
        a, bpt = off(p0, n0, so), off(p1, n1, so)
        T = C.TABLE
        if g0 == "rack" and g1 == "rack" and cls == "RF":
            return self.rf_front(P0, P1, lane, side or "left")
        if g0 == "rack" and g1 == "rack":
            return self.rack_route(P0, P1, cls, lane)
        if {g0, g1} == {"rack", "table"}:
            # rack end: out, to the RIGHT side of the rack at its lane depth, down to table height, across the
            # bridge onto the table in its lane (power at the back edge, data 50 mm or more in front of it)
            (PR, PT), rev = ((P0, P1), False) if g0 == "rack" else ((P1, P0), True)
            pr, nr = PR["pos"], PR["normal"]
            so = (25 if cls in C.POWER_CLS else 95) + 4 * (lane % 6)
            zb = self.lane_z(cls, lane)
            xs = C.PANEL_W + 45 + 3 * (lane % 10)
            tail, c, yc = self.table_end(PT, lane)
            zr = pr[2] - so if nr[2] < 0 else pr[2] + so
            head = [pr, [pr[0], pr[1], zr], [xs, pr[1], zr], [xs, pr[1], zb], [xs, yc, zb]]
            path = self.table_path((T["x0"] + self.GRID / 2, zb), c, cls)
            pts = head + [[x, yc, z] for x, z in path] + tail[::-1]
            return pts[::-1] if rev else pts
        if g0 == "table" and g1 == "table":
            h0, c0, yc0 = self.table_end(P0, lane)
            h1, c1, yc1 = self.table_end(P1, lane)
            yc = yc0
            path = self.table_path(c0, c1, cls)
            return h0 + [[x, yc, z] for x, z in path] + h1[::-1]
        if {g0, g1} == {"room", "table"} and "TABLE-EARTH" in (d0["id"], d1["id"]):
            # radio ground lead: on the table top in the power lane, then over the back edge down to its stud
            (PW, PT), rev = ((P0, P1), False) if g0 == "room" else ((P1, P0), True)
            pw = PW["pos"]
            tail, c, yc = self.table_end(PT, lane)
            zb = self.lane_z(cls, lane)
            path = self.table_path(c, (pw[0], zb), cls)
            ez = path[-1][1]
            pts = tail + [[x, yc, z] for x, z in path] + [[pw[0], yc, ez], [pw[0], yc, T["z1"]],
                                                         [pw[0], pw[1] + 3, T["z1"]], [pw[0], pw[1] + 3, pw[2]], pw]
            return pts[::-1] if g0 == "room" else pts
        if {g0, g1} == {"room", "table"}:                  # wall to table: floor, up the leg at the back edge
            (PW, PT), rev = ((P0, P1), False) if g0 == "room" else ((P1, P0), True)
            pw, nw = PW["pos"], PW["normal"]
            qw = off(pw, nw, so)
            yf = 30.0
            tail, c, yc = self.table_end(PT, lane)
            zb = self.lane_z(cls, lane)
            xl = T["x0"] + self.GRID / 2 + 2 * self.GRID
            pts = [pw, qw, [qw[0], yf, qw[2]], [xl, yf, qw[2]], [xl, yf, T["z1"]], [xl, yc, T["z1"]], [xl, yc, zb]]
            path = self.table_path((xl, zb), c, cls)
            pts += [[x, yc, z] for x, z in path] + tail[::-1]
            return pts[::-1] if rev else pts
        if {g0, g1} == {"room", "rack"} and cls == "PE":   # earth bond: left side lane, outside the rack footprint
            (PR, PM), rev = ((P0, P1), False) if g0 == "rack" else ((P1, P0), True)
            pr, nr, pm, nm = PR["pos"], PR["normal"], PM["pos"], PM["normal"]
            sx = -50 - 3 * (lane % 4)
            ar = off(pr, nr, so)
            am = off(pm, nm, so)
            pts = [pr, ar, [sx, ar[1], ar[2]], [sx, ar[1], am[2]], [sx, am[1], am[2]], am, pm]
            return pts[::-1] if rev else pts
        if {g0, g1} == {"room", "rack"}:                   # wall to rack (GNSS arrestor to the splitter): up the wall,
            (PM, PR), rev = ((P0, P1), False) if g0 == "room" else ((P1, P0), True)   # then in along the rack's left
            pm, nm, pr, nr = PM["pos"], PM["normal"], PR["pos"], PR["normal"]       # side, outside its footprint
            am = off(pm, nm, so)
            sx = min(am[0], -80.0)
            sr = 95 + 4 * (lane % 6)                        # coax: in the data depth band, 50 mm or more from power
            zr = pr[2] + sr if nr[2] > 0 else pr[2] - sr
            pts = [pm, am, [sx, am[1], am[2]], [sx, pr[1], am[2]], [sx, pr[1], zr], [pr[0], pr[1], zr], pr]
            return pts[::-1] if rev else pts
        if {d0["id"], d1["id"]} == {"TABLE-EARTH", "RACK-EARTH"}:
            # table earth bar to the rack earth bar: down to the floor, along the table's back edge to the rack-side leg,
            # then with the strip cord under the floor protector to behind the rack, and up to the rack bar's stud
            (PT, PR), rev = ((P0, P1), False) if d0["id"] == "TABLE-EARTH" else ((P1, P0), True)
            pt, pr, nr = PT["pos"], PR["pos"], PR["normal"]
            yf, xl, zf = 30.0, T["x0"] + 35.0, C.RACK["rail_depth"] + 200.0
            ar = off(pr, nr, so)
            pts = [pt, [pt[0], yf, pt[2]], [xl, yf, pt[2]], [xl, yf, zf], [ar[0], yf, zf], [ar[0], ar[1], zf], ar, pr]
            return pts[::-1] if rev else pts
        # room to room (earth bars, antenna lead to the arrestor)
        return [p0, a, [a[0], bpt[1], a[2]], [bpt[0], bpt[1], a[2]], bpt, p1]

    @staticmethod
    def plen(pts):
        return sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))

    def _cables(self):
        b = self.b
        self.cables = []
        self.counters = {}
        lane_of = {}
        self.side_of = {}
        ends = [(self.resolve(c["a"], c["cls"]), self.resolve(c["b"], c["cls"])) for c in b.cables]
        self._pending = ends
        self._occ = {"power": set(), "data": set()}          # table cells near each bundle (keeps them apart)
        lanes = []
        for c, (a, z) in zip(b.cables, ends):
            key = str(c.get("bundle", c["id"]))
            lane = lane_of.setdefault((c["cls"], key), len([k for k in lane_of if k[0] == c["cls"]]))
            # IF bundles take the side lane nearer to both ends (one side for the whole bundle of 4 tiles)
            side = self.side_of.setdefault((c["cls"], key), "left" if (self.ports[a]["pos"][0] + self.ports[z]["pos"][0]) / 2
                                           < C.PANEL_W / 2 else "right")
            lanes.append((key, lane, side))
        routes = {}
        # data and coax first, power second: on the table power then keeps clear of the data runs already laid
        order = sorted(range(len(b.cables)), key=lambda i: b.cables[i]["cls"] in C.POWER_CLS)
        for i in order:
            c, (a, z), (key, lane, side) = b.cables[i], ends[i], lanes[i]
            if c["length"] == "flange":                     # bolted joint: no cable, zero length
                routes[i] = [self.ports[a]["pos"], self.ports[z]["pos"]]
            else:
                routes[i] = self.route(self.ports[a], self.ports[z], c["cls"], lane, side)
        for i, (c, (a, z), (key, lane, side)) in enumerate(zip(b.cables, ends, lanes)):
            pts = routes[i]
            L = self.plen(pts)
            m = re.match(r"^\s*([\d.]+)\s*m\b", str(c["length"]))
            spec = float(m.group(1)) * 1000 if m else None
            stack = {}
            for e, pid in (("a", a), ("b", z)):             # DC block / pad standing between a CPL port and the plug
                sid = "ARM-" + pid[len("CPL-"):].rsplit(".", 1)[0] if pid.startswith("CPL-") and pid.endswith(".CPL") else ""
                if sid in self.dev:
                    stack = dict(stack=sid, stack_end=e, stack_mm=self.dev[sid]["size"][1])
            pe = [dict(end=e, pad=self.ports[pid]["device"], host=self.dev[self.ports[pid]["device"]]["host"])
                  for e, pid in (("a", a), ("b", z)) if self.dev[self.ports[pid]["device"]]["kind"] == "padstack"]
            if pe:                                          # the cable's end is a fixed pad stack's OUT (if_pads / mw_pads)
                stack["pad_ends"] = pe
            self.cables.append(dict(id=c["id"], cls=c["cls"], a=a, b=z, a_raw=BS.end_text(c["a"]), b_raw=BS.end_text(c["b"]),
                                    part=c["part"],
                                    pn=c["pn"], length=c["length"], spec_mm=spec, route_mm=round(L), vlan=c["vlan"],
                                    note=c["note"], route=[r1(p) for p in pts], color=C.CABLE[c["cls"]],
                                    dia=C.CABLE_DIA[c["cls"]], bundle=key,
                                    **({"lane_side": side} if c["cls"] == "RF" else {}), **stack))
            c["route_mm"] = round(L)

    def apply_lengths(self):
        """Choose every non-IF cable from its standard-length ladder by its routed length (+10 %); DC is cut from bulk.
        Updates the bench's cables in place (so the schedules and parts list follow) and re-routes."""
        b = self.b
        route = {c["id"]: c["route_mm"] for c in self.cables}
        self.length_fail = []
        # IF bundles: one stock length for the 4 tiles (equal length, not a phase-matched set), from the longest
        # routed tile + 5 %
        groups = {}
        for c in b.cables:
            if c["cls"] == "RF" or (c["cls"] == "WG" and "CBL-" in c["pn"]):
                groups.setdefault((c["cls"], str(c.get("bundle", c["id"]))), []).append(c)
        for cs in groups.values():
            need = max(route[c["id"]] for c in cs) * 1.05
            pn, L = next((s for s in BS.RF_STOCK if s[1] >= need), (None, None))
            if pn is None:
                pn, L = BS.RF_STOCK[-1]
                self.length_fail.append(f"IF bundle {', '.join(c['id'] for c in cs)}: needs {need:.0f} mm, longest stock "
                                        f"{pn} is {L} mm")
            for c in cs:
                if c["pn"] != pn:
                    if not c.get("tpl"):
                        c["part"] = re.sub(r"CBL-[\w.]+-SMSM\+", pn, c["part"])
                    c["pn"], c["length"] = pn, f"{L / 1000:.2f} m"
                c["need_mm"] = round(need)
                c["note"] = re.sub(r"needs [\d.]+ m( \(twin route\))?", f"needs {need / 1000:.2f} m (twin route)", c["note"])
        b.size_arm_pads()                                   # pads and part texts from the final IF lengths
        for k, v in self.dev.items():                      # the DC-block / pad stacks name the final pad
            if v["kind"] == "armstack":
                v["note"] = self.arm_note(self.coupler_sec(v["coupler"]), v["coupler"])
        for c in b.cables:
            need = route[c["id"]] * C.SLACK
            pn = c["pn"]
            lad = None
            if c["cls"] == "AC" and pn.startswith("CORD-60309"):
                lad = "AC_C19"
            elif c["cls"] == "AC" and (pn.startswith("P014") or pn.startswith("C14-C5")):
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
            elif pn.startswith("RG-316"):
                lad = "RG316"
            elif pn.startswith("LMR-240"):
                lad = "LMR240"
            if pn in BS.DC_CUT:
                if str(c["length"]).startswith("site"):        # run to a fixture whose position is not known
                    continue
                m = math.ceil(need / 250.0) * 0.25
                c["metres"], c["length"] = m, f"{m:g} m (cut)"
                continue
            if not lad:
                continue
            opt = next((o for o in C.LADDER[lad] if o[1] >= need), None)
            if opt is None:                                    # no stock length covers it: take the longest, flag it
                opt = C.LADDER[lad][-1]
                self.length_fail.append(f"{c['id']} {c['a']} -> {c['b']}: needs {need:.0f} mm, longest {lad} "
                                        f"{opt[0]} is {opt[1]} mm")
            old = c["pn"]
            c["pn"], c["length"] = opt[0], f"{opt[1] / 1000:g} m"
            if lad == "OM4":
                c["part"] = re.sub(r"Panduit FZ2ERLNLNSNM\d+ OM4 LC-LC [\d.]+ m", opt[2], c["part"]).replace("{om4}", opt[2])
            elif lad == "CAT6A":
                c["part"] = re.sub(r"Panduit STP6X\d+MBU( Cat6A( S/FTP)?)? [\d.]+ m", opt[2], c["part"])
            elif lad in ("RG316", "LMR240"):
                c["part"] = opt[2]
            else:
                c["part"] = opt[2] + c["part"][len(c["part"].split(" lead")[0]) + 5:] if " lead" in c["part"] else opt[2]
            if old != opt[0]:
                c["note"] = (c["note"] + "; " if c["note"] else "") + f"length from the twin route ({route[c['id']]} mm)"
        self._cables()
        self._steps()
        self._checks()

    # installation steps --------------------------------------------------------------------------
    # per cable class: which side of the rack it runs, how it is dressed, bend radius, slack, connectors
    ROUTE_TEXT = {
        "DC": "Strip each end with the wire stripper and ferrule every stranded end (ferrule crimper). Conductor "
              "convention: before landing a run, shrink a red (+48 V) or blue (-48 V) sleeve onto the SUPPLY conductor at "
              "both ends with the heat gun (breaker or fuse output, or supply terminal); the unsleeved conductor is the "
              "0 V return. Land each run at its breaker, fuse or supply end only and torque those terminals to the value "
              "marked on them (DCD1M, N250110) with the torque screwdriver; leave the LOAD end (adaptor DC terminal, "
              "{dcin_end}, radio DC plug) unlanded, tagged and taped with insulating tape: it is landed and "
              "torqued at the polarity check at power-up. {dcin}Rack runs go up the LEFT side of the rack as seen from "
              "the front (your right-hand side when you stand at the rear), on the rear lacing bars, 50 mm or more from "
              "coax and data. {table_dc}Bend radius 5 x the cable diameter or more; Velcro every 300 mm.",
        "POE": "{what} RJ45 to the node ETH1 at the rack rear, down the LEFT side with the power cables (seen from the "
               "front). WARNING: Q cables carry 48 V DC on an ordinary RJ45 (passive PoE, power only): never plug one "
               "into a switch, PC or laptop; put a yellow '48 V PoE - NOT ETHERNET' flag next to the ID label at both "
               "ends. Velcro to the lacing bars every 300 mm, no zip ties; Cat6A bend radius 4 x diameter (about 30 mm) "
               "or more; coil slack on the lacing bar, never at the connector.",
        "AC": "Fit the lead at the device inlet and at its PDU outlet (or strip outlet) only. Do NOT plug the PDU cords or "
              "the strip cord into the wall: they stay unplugged until power-up (guide section 6), and every PDU outlet "
              "stays OFF. Rack mains runs go up the LEFT side of the rack as seen from the front, 50 mm or more from coax "
              "and data, Velcroed to the rear lacing bars.{table_ac} The strip cord runs from its wall socket along the "
              "floor under the cable protector to the rack-side table leg, up the leg at the back edge and along the "
              "power lane to the strip inlet. {servers}",
        "FIB": "Data runs up the RIGHT side of the rack as seen from the front (your left-hand side at the rear). Node "
               "SFP+ (rear) to SW-DATA (front): cross from the rear to the front on the right side at SW-DATA's U. "
               "{agents}Runs to the bench table leave the rack on the right at about 560 mm behind the front rail and "
               "cross in the FRONT (data) compartment of the cable bridge (CABLE-BRIDGE) into the table's data lane, 50 mm "
               "or more in front of the power lane. Clean every LC end with the one-click cleaner before mating. Bend "
               "radius: OM4 30 mm or more, DAC 50 mm or more, Cat6A 4 x diameter. Coil slack behind the cable manager, "
               "Velcro only.",
        "ETH": "Management runs up the RIGHT side as seen from the front. SW-OOB and the PDUs sit on the rear posts, so "
               "their runs stay at the rear; {att_lan}BENCH-CTL{mwatt} and "
               "the analyser come from the table's data lane through the front (data) compartment of the cable bridge "
               "and up the right side. {agents}Cat6A bend radius 4 x diameter; Velcro every 300 mm; slack on the lacing "
               "bars.",
        "GNSS": "{arrestor}Jumpers down the RIGHT side of the rack, just outside the rails about {gz} mm behind the front "
                "rail (not the rear post), then in behind each Spider box to its node's GNSS SMA. {spare}RG-316 bend "
                "radius 25 mm or more, LMR-400 / LMR-240 per their data sheets. Torque the SMA joints to 8 in-lb "
                "(0.9 N m) with the SMA torque wrench, holding the jack with the backing wrench.",
        "AUX": "Operator station on the table: video and USB between BENCH-CTL, the monitor and the keyboard, on the table "
               "top around the monitor; keep them clear of the strip and the power lane.",
    }
    SMA_TORQUE = ("torque every SMA to 8 in-lb (0.9 N m) with the SMA torque wrench while holding the jack with the "
                  "backing wrench")

    def route_text(self, cls, step_of=None):
        b = self.b
        t = self.ROUTE_TEXT[cls]
        agents = bool(self.d.get("agents"))
        step_of = step_of or {}
        dcin = [c for c in self.cables if c["cls"] == "DC" and c["b"].endswith(".DCIN")]
        by_cb = OrderedDict()
        for c in dcin:
            by_cb.setdefault(c["a"].rstrip("b").split(".")[1], []).append(c["id"])
        twin_txt = "".join(f"{k} takes two runs ({', '.join(v)}): both supply conductors in one twin ferrule. "
                           for k, v in by_cb.items() if len(v) > 1)
        arr = self._cid(a="GNSS-ARR.gnd")
        lead = self._cid(a="GNSS-ANT")
        jmp = self._cid(a="GNSS-ARR.eqp")
        fill = dict(
            table_dc=(("The IP-50 feeds are the exception: from DC-DIST2's front fuse outputs they run along the front "
                       "of its U to the RIGHT side of the rack, back to about 640 mm behind the front rail and up to table "
                       "height in their own wrapped power bundle, then cross in the REAR (power) compartment of the cable "
                       "bridge. " if b.fuse else "") +
                      "On the table every radio feed" + (" (the switched supplies' leads too)" if b.psu else "") +
                      " runs in the power lane along the back edge, 50 mm or more behind the data lane, and reaches its "
                      "radio through the row gap and the aisles, away from the data cables. " if b.ip50 else ""),
            dcin=(f"The Spider DCIN runs ({dcin[0]['id']}-{dcin[-1]['id']}, box fans / Pi) are landed at their breakers "
                  "like the others; their box end stays unlanded until the hold point of step "
                  f"{step_of.get('Spider boxes', '?')} has confirmed DCIN is a 48 V DC input" +
                  (f" and its pin order: then each run goes into the screw terminals of its {self.dcin_plug.split(',')[0]} "
                   f"(the plug, not the {self.dcin_word}, takes the conductors)" if self.dcin_plug else "") + ". " + twin_txt
                  if dcin else ""),
            table_ac=(" The leads of the switched -48 V supplies (" + ", ".join(b.psu.values()) + ") leave the rack low "
                      "on the RIGHT in their own wrapped power bundle (about 640 mm behind the front rail), 50 mm or more "
                      "from the data bundle, and cross in the REAR (power) compartment of the cable bridge into the "
                      "table's power lane." if b.psu else ""),
            servers=("Servers switched at t0 (" + ", ".join(x for x in b.switched if x.startswith("SRV-")) + "): ONE "
                     "lead each, to PSU 1; PSU 2 was removed in the servers step (step "
                     f"{step_of.get('Servers and traffic hosts', '?')})." if any(x.startswith("SRV-") for x in b.switched)
                     else ""),
            what="Adaptor (or injector)" if b.inj else "Adaptor",
            dcin_end=(f"Spider DCIN plug ({self.dcin_plug.split(', ')[1]} 4-pole plug for the "
                      f"{self.dcin_word.replace('4-pole ', '')}, "
                      f"{self.dcin_plug.split(', ')[-1]})" if self.dcin_plug else "Spider DCIN terminal"),
            att_lan="the attenuator LAN ports are at the back of each box; " if b.atts else "",
            mwatt=", MWATT" if b.mwatt else "",
            gz=self._gnss_depth(),
            agents=("The front-row agent PCs face the front: their data and management leads run along the front of "
                    "the AGENTS shelf to the right side. " if agents else ""),
            spare=("Fit an ANNE-50+ load on every unused splitter output (" +
                   ", ".join(p for p in self.unused_ports("GNSS-SPL") if "out" in p) + "). "
                   if any("out" in p for p in self.unused_ports("GNSS-SPL")) else ""),
            arrestor=(f"First the arrestor: fit GNSS-ARR on the wall where the antenna lead enters the room, just above "
                      f"the lab earth bar, and bond its ground lug to the lab earth bar with {arr} (6 mm2 green/yellow, "
                      "short and straight, no loops; ring lugs crimped and torqued) BEFORE the antenna lead is connected "
                      "to anything; if the building already has an entry arrestor, bond to that instead (facilities). "
                      f"Then the antenna lead {lead} to the arrestor's antenna side, and the jumper {jmp} from its "
                      "equipment side up the wall and in along the rack's LEFT side (outside the rails) to GNSS-SPL's "
                      "input at the rear of the rack top. " if arr else "Antenna lead down to GNSS-SPL at the top of the "
                      "rack. "),
        )
        return t.format(**fill)

    def _cid(self, a=None, b_=None, cls=None):
        return next((c["id"] for c in getattr(self, "cables", []) if (a is None or c["a"] == a) and
                     (b_ is None or c["b"] == b_) and (cls is None or c["cls"] == cls)), None)

    def _gnss_depth(self):
        """depth (mm behind the front rail) of the GNSS jumpers' run down the right side, from the routes."""
        zs = [p[2] for c in getattr(self, "cables", []) if c["cls"] == "GNSS" and c["a"].startswith("GNSS-SPL.out")
              for p, q in zip(c["route"], c["route"][1:]) if p[0] > C.PANEL_W and abs(p[1] - q[1]) > 100]
        return int(round(sum(zs) / len(zs) / 10.0) * 10) if zs else 450

    def unused_ports(self, did):
        used = {e for c in getattr(self, "cables", []) for e in (c["a"], c["b"])}
        return [p for p in self.dev.get(did, {}).get("ports", []) if p not in used]

    def _ends(self, c):
        """a cable's two end ports plus the ports its fixed pad stacks are screwed onto (if_pads / mw_pads)."""
        return {c["a"], c["b"]} | {x["host"] for x in c.get("pad_ends", [])}

    def tile_clearance(self):
        """smallest vertical distance from a tile SMA to an IF run passing in front of it (other cables only)."""
        tiles = [p for p in self.ports.values() if re.match(r"^SB\d+\.M\d+\.T\d+$", p["id"])]
        best = 1e9
        for c in self.cables:
            if c["cls"] != "RF":
                continue
            for p, q in zip(c["route"], c["route"][1:]):
                if p[2] > -5 or q[2] > -5 or abs(p[1] - q[1]) > 0.1:
                    continue
                for t in tiles:
                    if t["id"] in self._ends(c):
                        continue
                    if min(p[0], q[0]) - 10 <= t["pos"][0] <= max(p[0], q[0]) + 10:
                        best = min(best, abs(p[1] - t["pos"][1]))
        return best

    def tray_pad_gap(self):
        """(axis pitch, body gap) in mm between the P1 and P2 pad stacks of one splitter (the tightest splitter)."""
        best = None
        for v in self.dev.values():
            if v["kind"] != "splitter":
                continue
            st = [x for x in self.dev.values() if x["kind"] == "padstack" and x.get("parent") == v["id"]]
            for i, a in enumerate(st):
                for c in st[i + 1:]:
                    pa, pc = self.ports[a["host"]]["pos"], self.ports[c["host"]]["pos"]
                    pitch = math.dist(pa, pc)
                    gap = pitch - (a["size"][1] + c["size"][1]) / 2
                    if best is None or gap < best[1]:
                        best = (pitch, gap)
        return best or (0.0, 0.0)

    def tray_headroom(self):
        """mm between the splitter-tray floor (the splitters' underside) and the underside of the rack unit just above the
        tray: what is left for hands and a wrench once that unit is in (0 when the bench has no splitter tray)."""
        spl = [v for v in self.dev.values() if v["kind"] == "splitter"]
        if not spl:
            return 0.0
        tr = self.dev[spl[0]["parent"]]
        floor = min(v["pos"][1] for v in spl)
        top = tr["pos"][1] + tr["size"][1]
        above = [v["pos"][1] for v in self.dev.values() if v["group"] == "rack" and not v.get("parent")
                 and v["kind"] != "blank" and v["pos"][1] >= top - 1]
        return (min(above) - floor) if above else 0.0

    def min_gap(self, parent):
        kids = [v for v in self.dev.values() if v.get("parent") == parent]
        g = 1e9
        for i, a in enumerate(kids):
            for c in kids[i + 1:]:
                o = [min(a["pos"][k] + a["size"][k], c["pos"][k] + c["size"][k]) - max(a["pos"][k], c["pos"][k])
                     for k in range(3)]
                g = min(g, max(-o[0], -o[2]))
        return g

    def _steps(self):
        b, d = self.b, self.d
        devs = self.dev
        rackd = [k for k, v in devs.items() if v["group"] == "rack" and not v.get("parent")]
        rows = {r["id"]: r for r in b.rows}

        def by(pred):
            return [k for k in rackd if pred(k)]

        def u(k):
            return devs[k]["u"]

        def nuts(ks):
            return sum(rows[k]["nuts"] for k in ks if k in rows)

        cid = self._cid
        S = []
        taken = set()
        step_of = {}                    # step title -> number (texts refer to other steps by number)

        def step(title, text, devices=(), cables=(), view="iso"):
            taken.update(cables)
            S.append(dict(n=len(S) + 1, title=title, text=text, devices=list(devices), cables=list(cables), view=view))
            step_of.setdefault(title, S[-1]["n"])
            return S[-1]["n"]
        # the numbers of the steps that later texts refer to, known before those steps are written
        self.step_no = getattr(self, "step_no", {})
        nlace = math.ceil(b.total / BS.LACE_EVERY_U)
        pe1 = cid(a="RACK-EARTH.E1")
        step("Assemble the rack", f"Build the {C.RACK['part']} ({self.rack['units']}U), set the rail-to-rail depth to "
             f"{int(C.RACK['rail_depth'])} mm and level it. Make it stable before anything heavy goes in: lower the "
             "levelling feet until the castors are off the floor (or lock every castor) and fit the rack's stabiliser "
             "or floor anchor if one is supplied. Fit the rack earth bar (RACK-EARTH, grounding kit) on the left rear "
             "post with paint-piercing washers; it is bonded to the lab earth bar in the next step, before anything "
             "else goes in. Stack from U1 upwards, power at the bottom. Fit the M6 cage nuts with the cage-nut tool as "
             f"each item goes in: 4 per front- or rear-mounted item, {nuts(list(rows))} for the items plus 2 per rear "
             f"lacing bar ({nuts(list(rows)) + 2 * nlace} in all; front or rear posts as each step says; rail-kit "
             "servers, the UNIRAILS1U rails (own M5 hardware) and blanking panels need none).", view="front")
        step("Cable: Earth bond", f"Bond the rack earth bar (RACK-EARTH, left rear post) to the lab earth bar with "
             f"{pe1}: 6 mm2 green/yellow, 2-hole lug with paint-piercing washers at the rack bar, ring lug at the lab "
             "bar; crimp the lugs with the ring-lug crimper and torque them with the torque wrench to the value marked "
             "on each stud. Print two ID labels and tag both ends. Check it end to end with the multimeter on ohms, test "
             "leads nulled first (REL): under 0.1 ohm. From now on wear the ESD strap bonded to this earth.", cables=[pe1], view="rear")
        pdus = by(lambda k: k.startswith("PDU"))
        self.step_no["PDUs"] = step(
            "Mount the PDUs", f"Fit the {len(pdus)} switched PDUs at {u(pdus[-1])}-{u(pdus[0])} ({BS.PDU_U}U each) on the "
            "REAR posts (4 cage nuts + screws each, at the rear), outlets and network port facing the rear. Commission "
            "each PDU bare, before any lead is fitted: plug its cord into its wall socket with nothing connected, enable "
            "front-panel outlet switching, set every outlet OFF and every outlet's power-on state to OFF (front panel or "
            "local port), check on its display that every outlet reads OFF, set its management address, then unplug "
            "the cord again. From here on the PDU cords stay out of the wall until power-up.", pdus, view="rear")
        dcs = by(lambda k: k.startswith("DC-"))
        strap = cid(a="DC-DIST2.strapA")
        pe2 = cid(a="DC-DIST2.rtn")
        tbond = cid(a="TABLE-EARTH.bond")
        rbond = [c["id"] for c in self.cables if c["cls"] == "PE" and c["b"].startswith("TABLE-EARTH.R")]
        # fitted first in the DC cable step: the -48 V system's own wiring and the radio earth bonds
        self._dc_first = [x for x in [pe2, strap, tbond] + rbond if x]
        step("Mount the DC drawers", "Rectifier shelves and distribution panels above the PDUs, on the front posts "
             "(ears on 4 cage nuts + screws each); the two rectifier shelves also run on a pair of UNIRAILS1U rails, "
             "bolted to the front and rear posts with the rail kit's own M5 cage nuts and screws. " +
             (f"DC-RECT2 ({u('DC-RECT2')}) and DC-DIST2 ({u('DC-DIST2')}) give -48 V to the IP-50 radios; the DC-DIST2 "
              f"RTN bus (= DC-RECT2 +V) is bonded to the rack earth bar ({pe2}, first item of the DC cable step), which "
              f"makes the -48 V system positive grounded, and bus B is strapped to bus A ({strap}). "
              if b.ip50 else "") +
             f"DC-RECT ({u('DC-RECT')}) and DC-DIST ({u('DC-DIST')}) give +48 V to the node adaptors and the Spider-box "
             f"DCIN inputs; DC-RECT's output stays floating (not earthed). Every breaker ({BS.BREAKER_A} A or less) OFF, "
             "the DCD1M main / input breaker too if the panel has one; every GMT fuse OUT, bagged and labelled with its "
             "position until power-up. Label the fronts with the label printer: DC-RECT and DC-DIST '+48 V FLOATING - "
             "NODES'" + (", DC-RECT2 and DC-DIST2 '-48 V POSITIVE GROUND - IP-50': the two rectifier shelves are the "
                         "same part, so the label is what tells them apart" if b.ip50 else "") + ".", dcs, view="front")
        pa = [k for k, v in devs.items() if v.get("parent") == "PA-SHELF"]
        wired = [k for k in pa if devs[k].get("row") == "rear"]
        spare = [k for k in pa if devs[k].get("spare")]
        step("Adaptor shelf", f"Fit the 1U shelf at {u('PA-SHELF')} on the front posts (4 cage nuts). Positions are "
             "measured from the left ear as seen from the FRONT (your right-hand ear when you stand at the rear). Rear "
             "row, left to right: " + ", ".join(f"{k} ({'node ' + devs[k]['feeds'] if devs[k].get('feeds') else ''})"
                                                for k in wired) +
             (f"; front row, at the far right, unwired spares {spare[0]} to {spare[-1]}" if spare else "") +
             f". RJ45 and DC terminals face the rear; every body {math.floor(self.min_gap('PA-SHELF'))} mm or more "
             "from the next.", ["PA-SHELF"] + pa, view="rear")
        shelves = by(lambda k: k.startswith("ATT-SHELF"))
        if shelves:                                    # none with fixed pads (if_pads): no attenuator step
            step("Programmable attenuators", "Fit one 1U vented shelf (front posts, 4 cage nuts) at the bottom of each 2U "
                 "attenuator space and stand its one or two attenuator boxes on it, ATT-01 top-left, numbers increasing "
                 "left to right and downwards. SMA IN/OUT and the dial face the front; LAN and power at the back. Each "
                 "box's plug-in power adapter is Velcroed to the rear lacing bar beside its shelf, never on the shelf "
                 "between the boxes.", shelves + [k for k, v in devs.items() if v.get("parent") in shelves], view="front")
        trays = by(lambda k: k.endswith("TRAY"))
        for tr in trays:
            parts = [k for k, v in devs.items() if v.get("parent") == tr]
            ends = []                   # tray cable ends fitted in this step (fixed pads: in the hand, before the tray)
            if b.split and b.if_pads:
                title = f"Splitter tray ({len(parts)} splitters)"
                # the hub W / E links' pad stacks are fitted onto the splitters here, in the hand, before the splitters
                # go onto the tray: on the tray the stacks lie flat 12 mm apart and no wrench reaches their nuts
                tp = [k for k, v in devs.items() if v["kind"] == "padstack" and v.get("parent") in parts]
                pitch, gap = self.tray_pad_gap()

                def stk(name):
                    return f"PAD-{name}.tn ({b.pad_links[name]['text']} dB)"
                per = "; ".join(f"{sid}.tn P1 takes {stk(d['splitters'][sec][0])}, P2 {stk(d['splitters'][sec][1])}"
                                for sec, sid in b.split.items())
                # every tray joint (pad onto port, pad onto pad, cable onto S and onto each stack's OUT) is made in the
                # hand here: once the Spider boxes are in, the lowest one sits {headroom} mm over the tray floor
                ends, per_spl = [], []
                for sec, sid in b.split.items():
                    l1, l2 = d["splitters"][sec]
                    rows_ = []
                    for n_ in range(1, b.tiles + 1):
                        sp = f"{sid}.t{n_}"
                        cs = next((c for c in self.cables if c["b"] == f"{sp}.S" or c["a"] == f"{sp}.S"), None)
                        c1 = next((c for c in self.cables if c["a"] == f"PAD-{l1}.t{n_}.out"), None)
                        c2 = next((c for c in self.cables if c["a"] == f"PAD-{l2}.t{n_}.out"), None)
                        for c, port, lk in ((cs, f"{sp}.S", f"{l1} + {l2}"), (c1, f"PAD-{l1}.t{n_}.out", l1),
                                            (c2, f"PAD-{l2}.t{n_}.out", l2)):
                            if c:
                                far = c["a"] if c["b"] == port else c["b"]
                                ends.append(dict(cable=c["id"], port=port, splitter=sp, link=lk, far=far,
                                                 stack=port[:-4] if port.endswith(".out") else None))
                        rows_.append((cs, c1, c2))
                    ids = lambda k: [r_[k]["id"] for r_ in rows_ if r_[k]]
                    rng_ = lambda xs: f"{xs[0]}-{xs[-1]}" if len(xs) > 1 else (xs[0] if xs else "-")
                    per_spl.append(f"{sid}.tn: S cable {rng_(ids(0))} (to {sec[0]}.{sec[1]} tile n), P1 stack's OUT "
                                   f"{rng_(ids(1))} ({l1}), P2 stack's OUT {rng_(ids(2))} ({l2})")
                headroom = self.tray_headroom()
                txt = (f"Fit the 1U tray at {u(tr)}, directly under the Spider boxes (front posts, 4 cage nuts). Then, on "
                       f"a padded work surface and BEFORE any splitter goes onto the tray, make every tray joint in the hand: "
                       f"once the Spider boxes are in (next step) the lowest one sits about {headroom:.0f} mm above the tray "
                       f"floor and no tray joint can be reached again. (1) Fit the {len(tp)} fixed pad stacks onto the "
                       f"{len(parts)} splitters (n = tile 1-{b.tiles}): {per}; the larger pad on the port, each pad's male "
                       "nut onto the female port in front of it; read the value printed on each pad first. Hold the "
                       "splitter in your hand (or flat on the padded surface), put the backing wrench on the body of the pad you are "
                       "screwing on and torque its nut to 8 in-lb (0.9 N m), the torque wrench's handle in the plane of "
                       "the two stacks and pointing away from the other stack (jaws above and below the nut); P1's stack "
                       f"first, then P2's. The stacks end up {pitch:.0f} mm apart (about {gap:.0f} mm between bodies); if "
                       "they foul, fit P2's through a short SMA M-F adapter (parts list, 'if needed') so its nuts stand "
                       "beyond P1's; never bend a stack. (2) Still in the hand, tag both ends of the splitter's three "
                       "cables and screw their tray ends on: " + "; ".join(per_spl) + ". Torque each to 8 in-lb with the "
                       "backing wrench on the cable plug's body (never on the splitter). (3) Lay the splitters flat, all "
                       "the same way round: S to the LEFT, ports 1 and 2 with their stacks to the RIGHT (port 1 toward the "
                       "front), label up; one column per split hub module, left to right "
                       f"{', '.join(f'{sid} ({sec[0]}.{sec[1]})' for sec, sid in b.split.items())}; rows front to back = "
                       f"tile 1-{b.tiles}, front row (t1) first, so that each row further back lays its cables over the "
                       f"rows in front. The stacks lie on the tray, up to "
                       f"{max(v['stack_mm'] for v in b.pads.values() if v['chain'] == 'if'):g} mm right of their "
                       "splitter. Cable lanes: every cable leaves its port or stack straight out; the port-2 stack's "
                       "cable runs 8 mm further out than port 1's; each row further back rises 10 mm more before running "
                       "forward, over the front rows' plugs; the S cables run to the tray front on the left, the link "
                       "cables on the right. (4) Coil the far (tile) ends loosely at the tray front, one coil per link, "
                       "and Velcro each coil to the tray's front lip, clear of the unit above: the IF-link steps uncoil "
                       "them, route them and land the tile ends only.")
                parts = parts + tp
            elif b.split:
                grp = [f"{sid} t1-t{b.tiles}" for sid in b.split.values()]
                title = f"Splitter tray ({len(parts)} splitters)"
                txt = (f"Fit the 1U tray at {u(tr)}, directly under the Spider boxes (front posts, 4 cage nuts). Lay the "
                       f"{len(parts)} splitters flat, all the same way round: S port to the LEFT, ports 1 and 2 to the "
                       "RIGHT (port 1 toward the front), printed label up. Columns left to right = tile 1-4; rows front to "
                       f"back: {', '.join(grp)}. The tile cables reach S from the left side lane, ports 1/2 feed the "
                       "attenuators below. Cable lanes: every cable leaves its port straight out; port 2's cable runs "
                       "8 mm further out than port 1's; the rear row's cables rise 10 mm before running forward, over "
                       "the front row's plugs.")
            else:
                cols = {0: [], 1: []}
                for sid, (bx0, _) in ((k, self.tray_slots(tr)[k]) for k in parts if k in self.tray_slots(tr)):
                    g = sid.split(".")[0]
                    c = 0 if bx0 < C.PANEL_W / 2 else 1
                    if g not in cols[c]:
                        cols[c].append(g)
                ncpl = len([k for k in parts if devs[k]["kind"] == "coupler"])
                top_ = devs[tr]["pos"][1] + devs[tr]["size"][1]
                cpl_y = max(self.ports[k + ".CPL"]["pos"][1] for k in parts if k + ".CPL" in self.ports)
                title = f"Coupler tray ({ncpl} couplers)"
                txt = (f"Fit the 1U shelf at the bottom of the {devs[tr]['u_n']}U space {u(tr)} (front posts, 4 cage "
                       f"nuts); leave the upper {devs[tr]['u_n'] - 1}U open. Lay the {ncpl} couplers flat, all the same "
                       "way round: IN on the LEFT end (toward the Spider tile), THRU on the RIGHT end (toward the link "
                       "attenuator), CPL on top (toward the arm attenuator). Two columns, rows front to back = tile 1-4 of "
                       f"each group: left column {', '.join(cols[0])}; right column {', '.join(cols[1])}. The DC blocks "
                       "and pads that stand on the CPL ports (ARM-*) go on in the coupling-arm steps: up to "
                       f"{self.BLOCK_MM + self.PAD_MM:g} mm, then the plug and the cable bend; the arm cable runs level "
                       f"{self.CPL_EXIT:g} mm above the port, {top_ - cpl_y - self.CPL_EXIT:.0f} mm under the unit above. "
                       "Cable lanes so no rear-row cable runs through a front-row plug: IN and THRU cables leave "
                       f"{self.PLUG_MM:g} mm straight out, then each row further back moves {self.LANE_X:g} mm further "
                       f"out before running forward; the arm cables move 12 mm to the right of their stack, plus "
                       f"{self.LANE_X:g} mm per row, and run forward side by side.")
            step(title, txt, [tr] + [k for k in parts if devs[k]["kind"] != "armstack"], view="front")
            if ends:                    # additive step field: cable ends fitted here, the rest of each cable later
                S[-1]["cable_ends"] = ends
            self.step_no["Splitter tray"] = S[-1]["n"]
        sbs = by(lambda k: k.startswith("SB"))
        dcin = [c for c in self.cables if c["cls"] == "DC" and c["b"].endswith(".DCIN")]
        cbs = sorted({int(re.sub(r"\D", "", c["a"].split(".")[1])) for c in dcin})
        hold = (f"Hold point before any DCIN run is landed at its box: confirm what the green {self.dcin_word.replace('green ', '')} "
                "(DCIN) on each box's rear feeds" +
                (", at what voltage, its pin order and polarity, and its pitch (the plug is in the parts list: "
                 f"{self.dcin_plug})" if self.dcin_plug else " and at what voltage") + ". The design assumes the box fans and the Raspberry "
                f"Pi at 48 V DC, fed from DC-DIST CB{cbs[0]}-CB{cbs[-1]} ({dcin[0]['id']}-{dcin[-1]['id']}, " +
                "; ".join(f"{c['b'].split('.')[0]} on {c['a'].split('.')[1].rstrip('b')}" for c in dcin) +
                "), landed at power-up only. Any other voltage or function: STOP; the DCIN runs stay unlanded and the "
                "design changes; if DCIN turns out to feed a node, that node's power path changes" +
                (" and SB2's feed must go on a switched PDU outlet (the hub dies at t0)" if b.inj else "") + "."
                if dcin else "")
        # the boxes' height off the floor: every IF joint is made at the boxes (fixed pads: 36 stacks, 2 joints each)
        y0 = min(devs[k]["pos"][1] for k in sbs) / 1000
        y1 = max(devs[k]["pos"][1] + devs[k]["size"][1] for k in sbs) / 1000
        low = (f"The boxes sit about {y0:.2f}-{y1:.2f} m off the floor: every IF joint (the pad stacks and their cables "
               "in the IF-link steps) is made down there, so work kneeling on a mat or from a low stool, with the torque "
               "wrench square to each joint. " if b.if_pads and y1 < 0.9 else "")
        step("Spider boxes", f"Mount the four 1U Spider boxes SB1 (top, {u(sbs[0])}) to SB4 ({u(sbs[-1])}). Which box goes "
             "where: " + "; ".join(f"{k} = {devs[k]['box_label']}, serial {devs[k]['serial']}" for k in sbs) +
             ". Check the serial on each box's label before it goes in. For each box fit a pair of UNIRAILS1U rails "
             "first, bolted to the front AND rear posts with the M5 cage nuts and screws in the rail kit; slide the box "
             "onto them and fix its ears on 4 M6 cage nuts + screws at the front. "
             f"{BS.POLARITY_NOTE[0].upper() + BS.POLARITY_NOTE[1:]}. The module map of each front panel is in the "
             "close-ups that follow the table of this step. " + low + hold, sbs, view="front")
        srv = by(lambda k: k.startswith(("SRV-", "HOST-")))
        sw_srv = [x for x in b.switched if x.startswith("SRV-")]
        if srv:
            step("Servers and traffic hosts", "Fit the rail kits shipped with each server (toolless in square holes) and "
                 "slide the servers in one at a time, network ports at the back (sizes in the rack schedule). Never have "
                 "more than one server pulled out on its rails at a time: the rack can tip." +
                 (f" {' and '.join(sw_srv)} die at t0 on one switched outlet each: if one has two PSUs, REMOVE PSU 2 "
                  "(and blank its bay) before it goes in." if sw_srv else ""), srv, view="front")
        cm = rows["CM"]
        above = next((r["id"] for r in b.rows if r["top"] - r["n"] + 1 == cm["top"] + 1), "the top of the rack")
        below = next((r["id"] for r in b.rows if r["top"] == cm["top"] - cm["n"]), "")
        ag = bool(d.get("agents"))
        step("Switches, cable manager" + (", agent shelf" if ag else ""), f"Cable manager at {u('CM')} (front posts), "
             f"between {above} above and {below} below; SW-DATA at {u('SW-DATA')} on the front posts, ports to the front, "
             "on a pair of UNIRAILS1U rails (front and rear posts, kit M5 hardware; Netgear's RBS1830 fits 18-30 in rails "
             f"only); SW-OOB at {u('SW-OOB')} on the REAR posts, ports to the rear; GNSS splitter at {u('GNSS-SPL')}, "
             "connectors to the rear"
             + (f"; agent PCs on their shelf at {u('AGENTS')}: back row with connectors to the rear, front row turned "
                "round with connectors to the FRONT, so no connector faces another PC; each PC's power brick is Velcroed "
                "to the rear lacing bar beside the shelf" if ag else "") +
             f". Then fit {nlace} lacing bars on the rear posts (2 cage nuts each), power on the left and data on the "
             "right as seen from the front.",
             by(lambda k: k in ("CM", "SW-DATA", "SW-OOB", "GNSS-SPL", "AGENTS")) +
             [k for k, v in devs.items() if v.get("parent") == "AGENTS"], view="front")
        step("Blanking panels", "Blank every empty U (toolless panels, no cage nuts).", [k for k in devs if k.startswith("BLANK")],
             view="front")
        table = [k for k, v in devs.items() if v["group"] == "table" and not v.get("parent")]
        stacked = [f"{k} on {v['stacked_on']}" for k, v in devs.items() if v.get("stacked_on")]
        T = C.TABLE
        dl, pl = self.LANES["data"], self.LANES["power"]
        st_ = devs["TABLE-STRIP"]
        extra = [k for k in ("CABLE-BRIDGE", "TABLE-EARTH") if k in devs]
        step("Bench table", "Place the bench table to the right of the rack and fit the cable bridge (CABLE-BRIDGE) from "
             "the rack's right post to the table's rack-side end, its floor at table height, divider in: data in the "
             "front compartment, power in the rear one. Set out the equipment at the positions in the table (every item "
             "5 mm or more from the next): the operator station nearest the rack (keyboard at the front, monitor behind "
             "it, BENCH-CTL behind the monitor"
             + (", the switched -48 V supplies beside BENCH-CTL" if b.psu else "") +
             f") and the strip lengthwise just right of the monitor ({st_['pos'][0] - T['x0']:.0f} mm from the "
             "rack-side end), inlet toward the back edge; " +
             ("MWATT in the middle, 90 mm in from the front edge, with the IP-50 radios left (A side) and right (B side) "
              "of it in a front and a back row 120 mm apart, waveguide flanges toward MWATT" if b.mwatt else
              "the IP-50 radios in hop pairs, the two radios of a hop side by side with their waveguide flanges facing "
              f"each other ({self._hop_pair_text()}), in a front and a back row 120 mm apart and two columns with a "
              f"{self.TABLE_PADS['aisle']:g} mm aisle between them; the gap between the two radios of a hop takes the "
              "transitions, the pad stacks and the hop's one SMA cable (IP-50 chain step); no radio is stacked")
             + (f"; stacked radios sit on two {BS.SPACER_MM:g} mm spacer bars on the lid of the one below (front and "
                "rear edge), each bar fixed to the lower lid with double-sided foam tape and the upper radio pressed onto "
                f"tape on the bars so it cannot slide: {', '.join(stacked)}" if stacked else "") +
             "; the analyser at the far end" +
             ("; screw the table earth bar (TABLE-EARTH) to the table's back edge behind " +
              ("MWATT" if b.mwatt else "the aisle between the radio columns") + ", studs up just below the top"
              if "TABLE-EARTH" in devs else "") +
             f". Keep the back {T['z1'] - dl[0]:.0f} mm of the table clear for the cable lanes: data bundle "
             f"{T['z1'] - dl[1]:.0f}-{T['z1'] - dl[0]:.0f} mm from the back edge, power bundle within "
             f"{T['z1'] - pl[0]:.0f} mm of it; keep the aisle between the strip and the A-side radios clear.",
             table + [k for k, v in devs.items() if v["group"] == "table" and v.get("parent") and
                      v["kind"] != "padstack"] + extra, view="table")
        # DC before AC: the DC is wired with no mains cord anywhere near a wall socket
        groups = [("DC", "DC power", "rear"), ("POE", "PoE to the nodes", "rear"), ("AC", "AC mains", "rear"),
                  ("FIB", "Data to SW-DATA", "rear"), ("ETH", "Management to SW-OOB", "rear"), ("GNSS", "GNSS", "rear"),
                  ("AUX", "Operator station", "table")]
        arr = cid(a="GNSS-ARR.gnd")
        for cls, title, view in groups:
            first = getattr(self, "_dc_first", []) if cls == "DC" else ([arr] if cls == "GNSS" and arr else [])
            cs = first + [c["id"] for c in self.cables if c["cls"] == cls and c["id"] not in taken and c["id"] not in first]
            if cs:
                lead = ""
                if cls == "DC" and first:
                    lead = ("First the earth bonds and the -48 V system's own wiring, all made before any feed is landed: "
                            + (f"bond the DC-DIST2 RTN bus earth stud to the rack earth bar with {pe2} (6 mm2 "
                               "green/yellow: positive ground); " if pe2 else "")
                            + (f"strap bus B to bus A with {strap} (skip the strap if the panel ships strapped); "
                               if strap else "")
                            + (f"bond the table earth bar to the rack earth bar with {tbond} (6 mm2, along the floor "
                               "under the cable protector with the strip cord) and each IP-50 radio's chassis ground screw "
                               f"to its TABLE-EARTH stud with {rbond[0]}-{rbond[-1]} (2.5 mm2 green/yellow; ground screw "
                               "position TBC); " if tbond and rbond else "")
                            + "crimp the lugs with the ring-lug crimper, torque them with the torque wrench to the value "
                            "marked on each stud, and check every bond end to end with the multimeter (leads nulled, under 0.1 ohm). ")
                    title = "DC power and earth bonds" if any(self.cable_cls(x) == "PE" for x in first) else title
                step(f"Cable: {title}", f"{len(cs)} cables; print two ID labels per cable on the label printer and tag "
                     f"both ends before routing. {lead}{self.route_text(cls, step_of)}", cables=cs, view=view)
        lane = {c["id"]: c.get("lane_side") for c in self.cables}
        clear = self.tile_clearance()
        route_txt = ("straight out from the port (or pad stack), down to the lower edge of its unit (for a Spider box: on "
                     "the seam with the unit below, between the two tile rows), along that edge to the side of the rack, "
                     "along the side, and in along the lower edge of the other unit; never across the face of another box, "
                     f"so every tile keeps {math.floor(clear)} mm or more clear above and below its SMA and the runs pass "
                     "15 mm or more in front of it. Connect the lowest tile first; "
                     f"{self.SMA_TORQUE}. Where a seam bundle from an earlier step lies close to a tile, ease it aside by "
                     "hand and fit the torque wrench with its handle sideways; never lever against a fitted cable. The four "
                     "tile cables are one stock length (equal length, not a phase-matched set: see the assumptions); coil "
                     "slack in the side lane.")
        n_tray = self.step_no.get("Splitter tray")
        for name, (sa, sb) in (d["link_ends"].items() if b.if_pads else ()):
            # fixed pads: tile (-> splitter S, P1 / P2) -> pad stack -> cable -> peer tile, one cable per tile path
            pl = b.pad_links[name]
            pids = [k for k, v in b.pads.items() if v["chain"] == "if" and v["link"] == name]
            ids = [c["id"] for c in self.cables if c["cls"] == "RF" and c["a"].rsplit(".", 1)[0] in pids]
            A, B = f"{sa[0]}.{sa[1]}", f"{sb[0]}.{sb[1]}"
            bxb, mb = d["module"][sb]
            pv = devs[pids[0]]
            sides = sorted({lane[i] for i in ids if lane.get(i)})
            side_txt = f"{' and '.join(sides)} side lane{'s' if len(sides) > 1 else ''}"
            value = (f"{' + '.join(f'{x} dB' for x in pl['pads'])} = {pl['db']} dB, target {pl['target']:g} dB, the A7 "
                     f"'{pl['setting']}' setting; pad schedule")
            job = (f" {name} is a bridge: held down before t0 with Terragraph ignition disabled for {b.bridges[name]['tg']} "
                   "(TG name TBC; set at power-up); after t0 (SRV-E2E off) MARL engages it at node level from "
                   f"{b.bridges[name]['agent'] or 'its A-end agent PC'} (tg2 minion assoc {A} to {B}, TBC; event "
                   "procedure). Nothing RF changes."
                   if name in b.bridges else
                   f" {name} goes down at t0 with the hub (its PoE injector's outlet is switched off); nothing RF changes.")
            if sa in b.split:
                bx, m = d["module"][sa]
                sid = b.split[sa]
                k = d["splitters"][sa].index(name) + 1
                tile = [c["id"] for c in self.cables if c["cls"] == "RF" and c["a"].startswith(f"{bx}.{m}.T") and
                        c["b"].startswith(sid + ".")]
                done = [x for x in tile if x in taken]
                if done:                       # a tile-to-splitter cable shared by two links: in the first link's step
                    n0 = min(st["n"] for st in S if set(done) & set(st["cables"]))
                    shared = f" (landed in step {n0}: {done[0]}-{done[-1]}, shared)"
                    s_route = ""
                else:
                    shared = (f" (its tray end on S since step {n_tray or '?'}; landed now, shared by " +
                              " and ".join(d["splitters"][sa]) + ")")
                    s_route = (f" The S cables ({tile[0]}-{tile[-1]}) run from the tray front to the left side lane, up "
                               f"it to the seam under {bx}, in along that seam and up into the hub tile n.")
                ids += [x for x in tile if x not in taken]
                # a stack column whose cables leave through the far side lane passes in front of the other column
                other = [o for o in b.split.values() if o != sid]
                col = ("left" if not other or devs[f"{sid}.t1"]["pos"][0] < devs[f"{other[0]}.t1"]["pos"][0]
                       else "right")
                across = (f", passing in front of {other[0]}'s column" if other and len(sides) == 1 and sides[0] != col
                          else "")
                txt = (f"For each tile n = 1-{b.tiles}: {A} tile n ({bx}.{m}.Tn) to {sid}.tn S{shared}; pad stack "
                       f"PAD-{name}.tn on {sid}.tn P{k} ({value}); cable from the stack's OUT to {B} tile n "
                       f"({bxb}.{mb}.Tn). The stacks and every tray end were fitted and torqued in the hand in step "
                       f"{n_tray or '?'}: leave the tray alone now (the Spider box above it is about "
                       f"{self.tray_headroom():.0f} mm over the tray floor). Uncoil this link's cables from the tray "
                       f"front, check each tag, and route them: along the tray front edge to the {side_txt}{across}, up "
                       f"the side to the lower edge of {bxb} (the seam with the unit below), in along that seam and up "
                       f"into the tile; never across the face of another box.{s_route} Land the tile ends in tile order, "
                       f"t1 first: screw the plug onto the tile jack, hold the jack with the backing wrench and "
                       f"{self.SMA_TORQUE.split(' while')[0]}; never pull on a cable toward the tray.{job} The four link "
                       "cables are one stock length (equal length, not phase-matched: assumptions); coil slack in the "
                       "side lane.")
                devices = []                   # the stacks went on in the splitter-tray step
            else:
                bxa, ma = d["module"][sa]
                fit = (f"Fit the pad stacks first, straight out to the front: " +
                       ", then ".join(f"{pn} ({db} dB)" + (" on the tile" if i == 0 else " on it")
                                      for i, (pn, db) in enumerate(zip(pl["pn"], pl["pads"]))) +
                       f" ({value}); read the value printed on each pad first; each pad's male nut screws onto the "
                       "female jack in front of it. Hold that jack with the backing wrench and torque every joint to 8 "
                       f"in-lb (0.9 N m). Each stack stands {pv['stack_mm']:g} mm proud of {bxa}'s front and must not "
                       "carry the cable: bend the cable down from the stack's OUT to the seam bundle at the unit's lower "
                       "edge and strap it (Velcro) to that bundle, or to the cable manager, within 150 mm of the stack, "
                       "then on to the side lane; never lever a stack sideways.")
                pad_route = ("down to the lower edge of its unit (the seam with the unit below), along it to the side of "
                             "the rack, along the side and in along the lower edge of the other unit; never across the face "
                             f"of another box (every tile keeps {math.floor(clear)} mm clear above and below its SMA; runs "
                             "pass 15 mm or more in front). Connect tile 1 first; at the B tile torque to 8 in-lb with the "
                             "backing wrench on the jack. If a seam bundle from an earlier step lies close to a tile, ease "
                             "it aside by hand and fit the wrench sideways. The four cables are one stock length (equal "
                             "length, not phase-matched: assumptions); coil slack in the side lane.")
                txt = (f"For each tile n = 1-{b.tiles}: pad stack PAD-{name}.tn screwed onto {A} tile n ({bxa}.{ma}.Tn); "
                       f"cable from the stack's OUT to {B} tile n ({bxb}.{mb}.Tn). {fit}{job} Then route every cable "
                       f"through the {side_txt}: {pad_route}")
                devices = pids
            step(f"IF link {name} ({pl['text']} dB pads)", txt, devices, cables=sorted(set(ids)), view="front")
        for name, (sa, sb) in (() if b.if_pads else d["link_ends"].items()):
            att = b.att_of[name]
            ids = [c["id"] for c in self.cables if c["cls"] == "RF" and (f"{att}." in c["a"] + c["b"])]
            shared_before = {}
            for sec in (sa, sb):
                bx, m = d["module"][sec]
                if sec in b.split or sec in b.coupled:
                    sid = b.split.get(sec) or b.coupled.get(sec)
                    tile = [c["id"] for c in self.cables if c["cls"] == "RF" and c["a"].startswith(f"{bx}.{m}.T")
                            and c["b"].startswith(sid) and c["id"] not in ids]
                    # a tile-to-splitter cable shared by two links is listed in the first link's step only
                    done = [x for x in tile if x in taken]
                    if done:
                        shared_before[sec] = (done, min(st["n"] for st in S if set(done) & set(st["cables"])))
                    ids += [x for x in tile if x not in taken]

            def path(sec, side):
                nm = f"{sec[0]}.{sec[1]}"
                if sec in b.split:
                    k = d["splitters"][sec].index(name) + 1
                    if sec in shared_before:
                        done, n0 = shared_before[sec]
                        shared = f" (already fitted in step {n0}: {done[0]}-{done[-1]}, shared)"
                    else:
                        shared = " (fitted once, shared by " + " and ".join(d["splitters"][sec]) + ")"
                    return (f"tile i of {nm} to {b.split[sec]}.ti S{shared}; P{k} to {att} CH i IN" if side == "in" else
                            f"{att} CH i OUT to {b.split[sec]}.ti P{k}; S to tile i of {nm}{shared}")
                if sec in b.coupled:
                    return (f"tile i of {nm} to {b.coupled[sec]}.ti IN; THRU to {att} CH i IN" if side == "in" else
                            f"{att} CH i OUT to {b.coupled[sec]}.ti THRU; IN to tile i of {nm}")
                return f"tile i of {nm} to {att} CH i IN" if side == "in" else f"{att} CH i OUT to tile i of {nm}"
            sides = sorted({lane[i] for i in ids if lane.get(i)})
            step(f"IF link {name} ({att})", f"{path(sa, 'in')}; {path(sb, 'out')}. Route every cable through the "
                 f"{' and '.join(sides)} side lane{'s' if len(sides) > 1 else ''}: {route_txt.replace(' (or pad stack)', '')}",
                 cables=sorted(set(ids)), view="front")
        for name, (x, z) in d["couplings"].items():
            att = b.att_of[name]
            ids = [c["id"] for c in self.cables if c["cls"] == "RF" and f"{att}." in c["a"] + c["b"]]
            pad = b.arm_pad.get(name, {})
            two = d.get("arm_dc_blocks", 1) == 2
            stacks = [k for k, v in devs.items() if v["kind"] == "armstack" and
                      v.get("coupler", "").split(".")[0] in (b.coupled[x], b.coupled[z])]
            step(f"Coupling arm {name} ({att})", f"CPL port (top) of {b.coupled[x]}.ti -> BLK-18-S+ DC block -> "
                 f"{pad.get('pn', 'pad')} {pad.get('db', '?')} dB pad -> cable -> {att} CH i IN; {att} CH i OUT -> cable -> "
                 + ("BLK-18-S+ DC block -> " if two else "") + f"CPL port (top) of {b.coupled[z]}.ti. First screw the DC "
                 f"blocks and pads onto the CPL ports ({stacks[0]} to {stacks[-1]}: "
                 f"{self.BLOCK_MM + self.PAD_MM:g} mm stacks on the input side, {self.BLOCK_MM:g} mm blocks on the output "
                 f"side) and {self.SMA_TORQUE}. Each cable's plug goes on top of its stack; the cable bends level "
                 f"{self.CPL_EXIT:g} mm above the CPL port, moves to its row's lane to the right of the stack, leaves "
                 "forward over the tray front and runs through the side lane as in the link steps. The four arm cables of "
                 "each side are one stock length (equal length, not a phase-matched set). Pad value from the arm budget "
                 f"(assumptions): attenuator about {pad.get('att_nominal', '?')} dB at INR {pad.get('inr_nominal', 13):g} "
                 "dB.", stacks, cables=ids, view="front")
        ws = [c["id"] for c in self.cables if c["cls"] == "WG"]
        if ws:
            ls = sorted({c["length"] for c in self.cables if c["cls"] == "WG" and "SMSM" in c["pn"]},
                        key=lambda v: float(v.split()[0]))
            rng = ls[0] if len(ls) == 1 else f"{ls[0].split()[0]}-{ls[-1]}"
            if b.mw_pads:
                wp = [k for k, v in devs.items() if v["kind"] == "padstack" and v["chain"] == "ip50"]
                hi, lo = BS.MW_PAD_HI, BS.MW_PAD_LO
                step("IP-50 conducted chains", f"Per hop: radio (A side) -> waveguide transition (bolted to the radio "
                     f"flange) -> {hi[0]} {hi[1]} dB 10 W pad -> {lo[0]} {lo[1]} dB pad -> ONE SMA cable ({rng}, see the "
                     f"table) -> {hi[0]} {hi[1]} dB 10 W pad -> transition -> radio (B side): {2 * hi[1] + lo[1]} dB "
                     "fixed, no programmable attenuator. First screw the pads onto each transition's SMA: the 30 dB 10 W "
                     "pad on every transition, its input (high-power) side on the transition (if it carries an arrow, the "
                     f"arrow points away from the radio), then the {lo[1]} dB pad on the A side's 30 dB pad (stacks "
                     f"{wp[0]} to {wp[-1]}); the transition carries the stack. "
                     f"{self.SMA_TORQUE[0].upper() + self.SMA_TORQUE[1:]}, every pad joint included. The SMA cable drops "
                     "from the A-side stack to the table top and rises to the B-side stack. If the band turns out to be "
                     f"23 GHz the {lo[1]} dB pad becomes a {BS.MW_PAD_LO_K} (assumptions). The radios stay unpowered "
                     "until the last stage of power-up, after every chain is complete with its pads.", wp, cables=ws,
                     view="table")
            else:
                step("IP-50 conducted chains", "Radio -> waveguide transition (bolted to the radio flange) -> SA18S10W-30 "
                     f"30 dB pad -> SMA cable ({rng}, see the table) -> MWATT CH j IN; CH j OUT -> SMA cable -> pad -> "
                     "transition -> radio. The SMA cables drop from the transition to the table top and run on it, round "
                     f"MWATT's front corner, to its front ports. {self.SMA_TORQUE[0].upper() + self.SMA_TORQUE[1:]}. The "
                     "radios stay unpowered until the last stage of power-up, after every chain is closed and every MWATT "
                     "channel is set.", cables=ws, view="table")
        self.power_up = self._power_up(step_of)
        bonds = [c["id"] for c in self.cables if c["cls"] == "PE"]
        pre = ("Before mains: every earth bond fitted and checked under 0.1 ohm (" + f"{bonds[0]}-{bonds[-1]}" + ")" +
               (f", the GNSS arrestor bonded ({arr}) before its lead is on the splitter" if arr else "") +
               "; every PDU outlet OFF with power-on state OFF (set when the PDUs were commissioned), every DC breaker "
               "OFF (the DCD1M main breaker too), every GMT fuse out, every DC load end unlanded and taped, the strip "
               "switch OFF")
        step("Checks and power-up", "Run the install checks (guide section 6). " + pre + ". Then power up: " +
             ", ".join(f"{s['title']} ({s['do']})" for s in self.power_up) + ".", view="iso")
        self.steps = S
        self.step_titles = step_of

    def _hop_pair_text(self):
        """'W1-CORE: IP50-W1-CORE@W1 | IP50-W1-CORE@CORE front row, left; ...' for the hop-pair table layout."""
        out = []
        for k, (a, bb) in self.d.get("mw_links", {}).items():
            ra, rb = self.dev[f"IP50-{k}@{a}"], self.dev[f"IP50-{k}@{bb}"]
            row = "front" if ra["pos"][2] < 0 else "back"
            col = "left" if ra["pos"][0] < self.TABLE_PADS["radio_x0"] + 10 else "right"
            out.append(f"{k}: {ra['id']} | {rb['id']}, {row} row, {col}")
        return "; ".join(out)

    def cable_cls(self, cid):
        return next((c["cls"] for c in self.cables if c["id"] == cid), None)

    def _outlet_of(self, dev_port):
        """'PDU-A.O3' for the AC lead that ends at a device port (or None)."""
        return next((c["a"] for c in self.cables if c["cls"] == "AC" and c["b"] == dev_port), None)

    def _power_up(self, step_of=None):
        """the power-up stages in the order the wiring implies (scene field 'power_up', additive): title, a short 'do'
        (used in the last step's text, kept free of other stages' keywords), the full 'detail' with port IDs, the
        devices (scene device IDs) and ports each stage energises, and (nodes stage) the Terragraph node names."""
        b, d = self.b, self.d
        step_of = step_of or {}
        out = []

        def stage(title, do, detail, devices=(), ports=(), **kw):
            out.append(dict(n=len(out) + 1, title=title, do=do, detail=detail, devices=list(devices), ports=list(ports),
                            **kw))

        def pairs(names):
            """[(outlet, load label, device)] in the given device order, one per AC lead (rectifier modules too)."""
            res = []
            for nm in names:
                for c in self.cables:
                    if c["cls"] != "AC" or not c["a"].startswith("PDU"):
                        continue
                    dv = c["b"].split(".")[0]
                    if dv == nm:
                        port = c["b"].split(".", 1)[1]
                        lab = nm + (f" module {port[2:]}" if re.match(r"^ac\d$", port) else
                                    " module 1" if nm in BS.RECT_MODULES and BS.RECT_MODULES[nm] > 1 else "")
                        res.append((c["a"], lab, nm))
            return res

        def on_list(ps):
            return ", ".join(f"{o} ({lab})" for o, lab, _ in ps)

        def rng(xs):
            return xs[0] if len(xs) == 1 else f"{xs[0]}-{xs[-1]}"
        feeds = [c for c in self.cables if c["cls"] == "AC" and c["a"].startswith("WALL.")]
        pdu_feeds = [c for c in feeds if c["b"].startswith("PDU")]
        strip = [c for c in feeds if c["b"] == "TABLE-STRIP.inlet"]
        ws = [c["a"].split(".")[1] for c in pdu_feeds]
        stage("PDUs", f"cords into WALL.{ws[0]}-{ws[-1]} and check every outlet reads OFF",
              "plug " + ", ".join(f"{c['id']} ({c['b'].split('.')[0]}) into {c['a']}" for c in pdu_feeds) +
              "; on each PDU's own display check that every outlet reads OFF (power-on state set at commissioning); if "
              "any outlet reads ON: unplug that PDU's cord, pull the C13 lead from every outlet of that PDU, recommission "
              f"it bare as in step {step_of.get('Mount the PDUs', '?')}, refit the leads, then plug the cord in again "
              "and re-check",
              [c["b"].split(".")[0] for c in pdu_feeds], [c["a"] for c in pdu_feeds])
        sw = [x for x in ("SW-DATA", "SW-OOB", "GNSS-SPL") if x in self.dev]
        swp = pairs(sw)
        home = sorted({o.split(".")[0] for o, _, _ in swp})
        stage("switches and bench controller",
              (f"strip cord into {strip[0]['a']} and strip ON; " if strip else "") + "switch the SW-DATA / SW-OOB / "
              "GNSS-SPL outlets ON at the front panel; wait for them to boot",
              (f"plug {strip[0]['id']} into {strip[0]['a']} and switch the strip ON: " + ", ".join(b.table_loads) + "; "
               if strip else "") + f"at {' and '.join(home)}'s own front panel (rack rear; there is no network yet) "
              f"switch ON {on_list(swp)}, then set the power-on state of these outlets to ON so management survives a "
              "mains dip; wait for the switches to boot; BENCH-CTL must reach every PDU and switch over SW-OOB before "
              "the next stage", ["TABLE-STRIP"] + list(b.table_loads) + sw,
              ([strip[0]["a"]] if strip else []) + [o for o, _, _ in swp])
        atts = list(b.atts)
        ap = pairs(atts)
        if atts or b.mwatt:                          # none with fixed pads (if_pads / mw_pads): no attenuator stage
            stage("attenuators", "their outlets ON; BENCH-CTL sets every attenuator" + (" and MWATT" if b.mwatt else "") +
                  " channel to its start value",
                  f"from BENCH-CTL switch ON {on_list(ap)}; set every channel to its start value (attenuator schedule)" +
                  (" and every MWATT channel to its start value" if b.mwatt else "") + "; read every value back",
                  atts + (["MWATT"] if b.mwatt else []), [o for o, _, _ in ap])
        comp = [x for x in b.rack_servers] + [f"HOST-{h}" for h in d["hosts"]] + [f"AGENT-{s_}" for s_ in d.get("agents", [])]
        comp = sorted(comp, key=lambda x: x != "SRV-E2E")              # SRV-E2E first
        cp = pairs(comp)
        e2e = "SRV-E2E" in comp
        stage("servers and hosts", "outlets ON" + ("; SRV-E2E first" if e2e else ""),
              (f"switch ON {cp[0][0]} (SRV-E2E) and wait for it to boot, so the E2E controller is up before any "
               f"Terragraph unit boots; then switch ON {on_list(cp[1:])}" if e2e else f"switch ON {on_list(cp)}"),
              comp, [o for o, _, _ in cp])
        rect = [r for r in ("DC-RECT", "DC-RECT2") if r in self.dev]
        rp = pairs(rect)
        psu = list(b.psu.values())
        pp = pairs(psu)
        bonds = [c["id"] for c in self.cables if c["cls"] == "PE"]
        cbs = [c for c in self.cables if c["cls"] == "DC" and c["a"].startswith("DC-DIST.CB")]
        fs = [c for c in self.cables if c["cls"] == "DC" and c["a"].startswith("DC-DIST2.F")]
        positions = sorted({c["a"].split(".")[1].rstrip("b") for c in cbs}, key=lambda x: int(x[2:]))
        fpos = [c["a"].split(".")[1] for c in fs]
        earth_ref = ("then check the earth reference with COM on RACK-EARTH: DC-DIST2 RTN reads 0 V and DC-DIST2 bus A "
                     "reads -43 to -56 V" if "DC-DIST2" in self.dev else "")
        insul = ("with both rectifiers off, every CB open" + (" and every GMT fuse out" if fs else "") +
                 ", meter ohms: DC-DIST + bus to DC-DIST return, + bus to RACK-EARTH and return to RACK-EARTH" +
                 ("; DC-DIST2 bus A and bus B to RTN" if "DC-DIST2" in self.dev else "") +
                 ": each must climb to OL or above 1 Mohm (the rectifier capacitors charge first); a reading under "
                 "1 kohm is a short or an earth fault (reversed strap, nicked run): find it before switching on" +
                 ("; DC-DIST2 RTN to RACK-EARTH must read under 0.1 ohm (positive-ground system)" if "DC-DIST2" in self.dev
                  else "") + "; the +48 V system stays floating: do not bond it to earth")
        shared = {}
        for c in cbs:
            shared.setdefault(c["a"].split(".")[1].rstrip("b"), []).append(c["id"])
        shared = {k: v for k, v in shared.items() if len(v) > 1}
        shared_txt = "".join(f"; {k} carries {len(v)} runs ({', '.join(v)}): meter every free end before landing any"
                             for k, v in shared.items())
        meter = ("meter COM on the return (unsleeved) conductor and the red probe on the supply (sleeved) conductor at "
                 "the free load end: a DC-DIST breaker output reads about +48 V (+43 to +56 V)" +
                 ("; a DC-DIST2 fuse output reads about -48 V (-43 to -56 V)" if b.fuse else "") +
                 ("; a switched supply output reads about -48 V (-43 to -56 V)" if psu else "") +
                 "; any other sign or value: stop, open the breaker or pull the fuse and correct the run at its supply end")
        land = ("open the breaker (pull the fuse with the fuse puller) again, then land the red / blue-sleeved conductor "
                "on the load's + terminal (PA adaptor, Spider DCIN" + (" plug" if self.dcin_plug else "") + ") or -48 V "
                "pin (radio) and the unsleeved conductor on the other, torque adaptor and DCIN " +
                ("plug terminals" if self.dcin_plug else "terminals") + " to the value marked on them" +
                (f", push each DCIN plug into its {self.dcin_word} and tighten its flange screws" if self.dcin_plug
                 else "") + ", and for a radio mate the DC connector")
        stage("DC rectifiers", "earth bonds checked; outlets ON; meter each breaker / fuse output at its free load end "
                               "before landing the load",
              f"with the rectifiers still off, check every earth bond end to end with the meter on ohms (leads nulled), under 0.1 ohm "
              f"({rng(bonds)}); {insul}; switch ON {on_list(rp)}; close the DCD1M main / input breaker if fitted, with every CB "
              "still open, and meter the DC-DIST bus: + bus to return +43 to +56 V" + (f"; {earth_ref}" if earth_ref else "")
              + "; then one position "
              f"at a time ({' and '.join(x for x in (rng(positions) if positions else '', rng(fpos) if fpos else '') if x)}"
              f"): close the breaker (or insert the fuse), {meter}; {land}" + shared_txt +
              (f"; the same for the switched supplies {on_list(pp)}: outlet ON, measure at the free radio end, outlet "
               "OFF, then land it and mate the radio's DC connector" if psu else ""), rect + psu, [o for o, _, _ in rp + pp])
        ip = pairs(list(b.inj.values()))
        boxes = [x["id"] for x in d["boxes"]]
        fans = [c for c in cbs if c["b"].endswith(".DCIN")]
        nodes = [c for c in cbs if not c["b"].endswith(".DCIN")]
        fan_cb = sorted({c["a"].split(".")[1].rstrip("b") for c in fans}, key=lambda x: int(x[2:]))
        node_cb = [c["a"].split(".")[1] for c in nodes]
        fan_txt = ", ".join(f"{cb} (" + " + ".join(x["b"].split(".")[0] for x in fans
                                                   if x["a"].split(".")[1].rstrip("b") == cb) + " DCIN)" for cb in fan_cb)

        def fan_group(bx):
            """the boxes whose fans share a DCIN breaker with bx (bx included), and those breakers."""
            dcin = sorted({c["a"].split(".")[1].rstrip("b") for c in fans if c["b"] == f"{bx}.DCIN"})
            grp = sorted({c["b"].split(".")[0] for c in fans if c["a"].split(".")[1].rstrip("b") in dcin} | {bx})
            return grp, dcin

        def box_breakers(bx):
            """what powers one Spider box down: its node breakers (and switched outlet) first, then its DCIN feed; a
            shared DCIN breaker takes the nodes of every box it feeds."""
            grp, dcin = fan_group(bx)
            cb, sw_ = [], []
            for g in grp:
                nds = d["boxes"][[x["id"] for x in d["boxes"]].index(g)]["nodes"]
                cb += [c["a"].split(".")[1] for c in nodes if self.dev[c["b"].split(".")[0]].get("feeds") in nds]
                sw_ += [o for (o, lab, _), n in zip(ip, b.inj) if n in nds]
            txt = " and ".join(x for x in (", ".join(cb), f"switch {', '.join(sw_)} OFF" if sw_ else "") if x)
            if dcin:
                txt += f", then {', '.join(dcin)}"
                if len(grp) > 1:
                    txt += f" ({', '.join(dcin)} feeds the fans of {' and '.join(grp)})"
            return txt

        down, seen = [], set()
        for bx in boxes:
            if bx in seen:
                continue
            grp, _ = fan_group(bx)
            seen.update(grp)
            down.append(f"{' or '.join(grp)}: {box_breakers(bx)}")
        cur = ("then read each PDU's input current on its own display (" + ", ".join(b.pdus) + "): each must be 12.8 A "
               "or less (80 % of 16 A); if one is above, stop and move a load")
        tight = [s_["supply"] for s_ in self.power_budget() if s_["total_w"] > 0.9 * s_["limit_w"]]
        # fixed pads (if_pads): a bridge has no RF way down, so its ignition is disabled before any node boots
        bridges = list(b.bridges)
        ign = ("before closing any breaker, " + BS.bridge_hold_text(b.bridges).replace("(SRV-E2E)", "(SRV-E2E, up since "
               "the last stage)") + ", so they stay down until MARL engages them after t0 (node level, from the A-end "
               "agent PC: event procedure); " if bridges else "")
        stage("nodes", (f"close {rng(fan_cb)} (box fans) first and check every fan runs; then " if fan_cb else "") +
              f"{rng(node_cb)} one at a time" + (f"; {' + '.join(b.inj.values())} outlet ON" if b.inj else ""),
              ign +
              (f"close DC-DIST {fan_txt} one at a time and check that each box's fans run within 1 min (if not: open "
               "that breaker, leave the nodes of every box it feeds off and find out why); then " if fan_cb else "") +
              "close DC-DIST " + ", ".join(f"{c['a'].split('.')[1]} ({c['b'].split('.')[0]}, node "
                                          f"{self.dev[c['b'].split('.')[0]].get('feeds', '?')})" for c in nodes) +
              " one at a time" + (f"; switch ON {', '.join(f'{o} ({lab}, node {n})' for (o, lab, _), n in zip(ip, b.inj))}"
                                  if b.inj else "") +
              "; if a box's fans stop while its nodes boot, power that box down: " +
              "; ".join(down) + "; wait for every node to join the E2E controller; " +
              (f"the bridges {', '.join(bridges)} stay down (ignition disabled before the first breaker): check on the "
               "E2E controller that they are not ignited and that every other link comes up on its fixed pads; "
               if bridges else "") + cur +
              (f" ({', '.join(tight)} is within 10 % of its limit on the estimates: read it first)" if tight else "") +
              ("; then the port check before the radios: every IF link's SNR against its expected value (guide section "
               "6, 'Port check'; the bridges one at a time, ignition enabled only while read)" if b.if_pads else ""),
              boxes + [c["b"].split(".")[0] for c in nodes] + list(b.inj.values()),
              [c["a"] for c in cbs] + [o for o, _, _ in ip], nodes=list(b.dc_nodes) + list(b.inj))
        if b.ip50:
            ws_ = sorted({c["id"] for c in self.cables if c["cls"] == "WG"})
            chain_ok = ("only after every chain is complete with its pads" if b.mw_pads else
                        "only after every chain is closed and every MWATT channel is set")
            chain_detail = (f"first confirm every IP-50 chain ({ws_[0]}-{ws_[-1]}) is complete and torqued (transition "
                            f"bolted, a {BS.MW_PAD_HI[0]} on each transition, the {BS.MW_PAD_LO[0]} on the A side's, the "
                            "one SMA cable on both stacks; no open flange or SMA); never power a radio with an open RF "
                            "port; then " if b.mw_pads else
                            f"first confirm every IP-50 chain ({ws_[0]}-{ws_[-1]}) is complete and torqued (transition "
                            "bolted, pad and cable on, no open flange or SMA) and MWATT reads back its start value on "
                            "every channel; never power a radio with an open RF port; then ")
            stage("radios last", (f"GMT fuses {fpos[0]}-{fpos[-1]} in one at a time" if fs else "") +
                  ("; supply outlets ON" if psu else "") + "; " + chain_ok,
                  chain_detail +
                  (f"insert the GMT fuses on DC-DIST2 one at a time: " +
                   ", ".join(f"{c['a'].split('.')[1]} ({c['b'].split('.')[0]})" for c in fs) if fs else "") +
                  (f"; switch ON {on_list(pp)}" if psu else "") +
                  "; after 30 min check every radio's temperature" +
                  (" (the stacked ones in particular)" if b.stacked_hops else "") + " in the IP-50 web UI "
                  "or CLI, reached in-band over the data network from SRV-E2E (the NMS host; BENCH-CTL has no data-"
                  "network port); " + cur,
                  list(b.ip50) + psu, [c["a"] for c in fs] + [o for o, _, _ in pp])
        for st in out:                                       # the guide splits the list on keywords: keep them apart
            assert not re.search(r"pdu", st["do"], re.I) and "," not in st["do"], st["do"]
            if st["title"] != "DC rectifiers":
                assert not re.search(r"dc", st["do"], re.I), st["do"]
            for x in st["devices"]:
                assert x in self.dev, (st["title"], x)
        return out

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
        lf = getattr(self, "length_fail", [])
        ch.append(dict(name="Every cable's route + slack is covered by a stock length", ok=not lf,
                       detail="; ".join(lf) or "every ladder / IF stock choice covers its need"))
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
        ch.append(dict(name="Spider SMA tile ports", ok=True, detail=f"{sum(1 for p in unres if p['id'] in used or p.get('mate'))} "
                       f"of {len(unres)} used" + (f" ({sum(1 for p in unres if p.get('mate'))} carry a pad stack)"
                                                  if any(p.get("mate") for p in unres) else "") +
                       "; unused ones get ANNE-50+ loads"))
        ch += self._pad_checks(used)
        ch += self._body_checks()
        off = [p["id"] for p in self.ports.values() if not all(
            self.dev[p["device"]]["pos"][k] - (1 if k < 2 else 15.5) <= p["pos"][k] <=
            self.dev[p["device"]]["pos"][k] + self.dev[p["device"]]["size"][k] + (1 if k < 2 else 15.5) for k in range(3))]
        ch.append(dict(name="Every port sits on its own device's body", ok=not off,
                       detail=", ".join(off[:20]) or f"{len(self.ports)} ports within 1 mm (face) / 15 mm (connector depth)"))
        ch += self._tile_access_check()
        dcin = [f"{x['id']}.DCIN" for x in self.d["boxes"]]
        ch.append(dict(name="Every Spider box DCIN (fans / Pi) has a DC feed from a DC-DIST breaker",
                       ok=all(p in used for p in dcin),
                       detail=", ".join(f"{p} <- {next(c['a'] for c in self.cables if c['b'] == p)}" for p in dcin if p in used)
                       or "none fed"))
        want = ["RACK-EARTH.E1", "GNSS-ARR.gnd"] + (["DC-DIST2.rtn", "TABLE-EARTH.bond"] if self.b.ip50 else []) + \
            [f"{k}.gnd" for k in self.b.ip50]
        miss = [p for p in want if not any(c["cls"] == "PE" and p in (c["a"], c["b"]) for c in self.cables)]
        ch.append(dict(name="Every earth bond is in the schedule (rack, -48 V RTN, GNSS arrestor, table bar, each radio)",
                       ok=not miss, detail=", ".join(miss) or f"{sum(1 for c in self.cables if c['cls'] == 'PE')} PE bonds"))
        ch += self._cable_checks(used)
        ch += self._arm_fit_check()
        ch += self._power_budget_check()
        if getattr(self.b, "arm_pad", None) is not None:
            bad = BS.bom_check(self.b, BS.parts(self.b))
            ch.append(dict(name="Parts list orders exactly the cables in the schedule (per part number)", ok=not bad,
                           detail="; ".join(bad) or f"{len({c['pn'] for c in self.b.cables})} cable part numbers match"))
        self.checks = ch

    # cable geometry: through bodies, through other cables' plugs, connector access, table outline ----------------
    PLUG_ZONE, PLUG_HALF, ACCESS_MM, ACCESS_HALF = 20.0, 5.0, 35.0, 4.0

    def _family(self, did):
        out = {did} | self._ancestors(did)
        stack = [did]
        while stack:
            x = stack.pop()
            for k, v in self.dev.items():
                if v.get("parent") == x and k not in out:
                    out.add(k)
                    stack.append(k)
        return out

    def _solid(self, v):
        """the solid box of a body: shelves, trays and attenuator shelves are their 4 mm plate (the parts on them are
        bodies of their own); everything else its full box."""
        lo = list(v["pos"])
        hi = [v["pos"][k] + v["size"][k] for k in range(3)]
        if v["kind"] in ("shelf", "tray", "attshelf"):
            hi[1] = lo[1] + 4
        return lo, hi

    @staticmethod
    def _seg_box(p, q, lo, hi, shrink=1.0):
        t0, t1 = 0.0, 1.0
        for i in range(3):
            a, b_ = lo[i] + shrink, hi[i] - shrink
            dd = q[i] - p[i]
            if abs(dd) < 1e-9:
                if not (a < p[i] < b_):
                    return False
            else:
                u0, u1 = (a - p[i]) / dd, (b_ - p[i]) / dd
                if u0 > u1:
                    u0, u1 = u1, u0
                t0, t1 = max(t0, u0), min(t1, u1)
                if t0 >= t1:
                    return False
        return t1 - t0 > 1e-6

    def _cable_checks(self, used):
        fam = {k: self._family(k) for k in self.dev}
        solids = {k: self._solid(v) for k, v in self.dev.items()}
        through, plugs, block, outside = [], [], [], []
        cab_box = {}
        for c in self.cables:
            xs = list(zip(*c["route"]))
            cab_box[c["id"]] = ([min(v) for v in xs], [max(v) for v in xs])
        # 1. no cable route passes through a body (other than its own end devices, their parents and children)
        for c in self.cables:
            ex = fam[self.ports[c["a"]]["device"]] | fam[self.ports[c["b"]]["device"]]
            clo, chi = cab_box[c["id"]]
            for k, (lo, hi) in solids.items():
                if k in ex or any(chi[i] < lo[i] or clo[i] > hi[i] for i in range(3)):
                    continue
                if any(self._seg_box(p, q, lo, hi) for p, q in zip(c["route"], c["route"][1:])):
                    through.append(f"{c['id']} through {k}")
        # 2. no cable runs through the plug of another cable (20 mm in front of a used port, 5 mm either side)
        zones = []
        for pid in used:
            P = self.ports[pid]
            p, n = P["pos"], P["normal"]
            lo = [p[i] - self.PLUG_HALF if n[i] == 0 else (p[i] + 0.5 if n[i] > 0 else p[i] - self.PLUG_ZONE) for i in range(3)]
            hi = [p[i] + self.PLUG_HALF if n[i] == 0 else (p[i] + self.PLUG_ZONE if n[i] > 0 else p[i] - 0.5) for i in range(3)]
            zones.append((pid, lo, hi))
        for c in self.cables:
            clo, chi = cab_box[c["id"]]
            for pid, lo, hi in zones:
                if pid in (c["a"], c["b"]) or any(chi[i] < lo[i] or clo[i] > hi[i] for i in range(3)):
                    continue
                if any(self._seg_box(p, q, lo, hi, shrink=0.0) for p, q in zip(c["route"], c["route"][1:])):
                    plugs.append(f"{c['id']} through the plug on {pid}")
        # 3. every used port has 35 mm of clear space along its normal (plug + strain relief), no other body in it
        for pid in used:
            P = self.ports[pid]
            p, n = P["pos"], P["normal"]
            lo = [p[i] - self.ACCESS_HALF if n[i] == 0 else (p[i] + 0.5 if n[i] > 0 else p[i] - self.ACCESS_MM) for i in range(3)]
            hi = [p[i] + self.ACCESS_HALF if n[i] == 0 else (p[i] + self.ACCESS_MM if n[i] > 0 else p[i] - 0.5) for i in range(3)]
            ex = fam[P["device"]]
            for k, v in self.dev.items():
                if k in ex:
                    continue
                a0, a1 = solids[k]
                if all(min(hi[i], a1[i]) - max(lo[i], a0[i]) > 0.2 for i in range(3)):
                    block.append(f"{pid} blocked by {k}")
        # 4. cables on the bench table stay inside its outline (nothing hangs over the front edge into the operator's
        #    space or off the back)
        T = C.TABLE
        for c in self.cables:
            if "table" not in {self.dev[self.ports[c["a"]]["device"]]["group"], self.dev[self.ports[c["b"]]["device"]]["group"]}:
                continue
            for q in c["route"]:
                if q[1] >= T["top"] - 1 and T["x0"] - 0.5 <= q[0] and not (T["z0"] <= q[2] <= T["z1"] and q[0] <= T["x1"]):
                    outside.append(f"{c['id']} at {q}")
                    break
        return [dict(name="No cable passes through a body (rack units, tray / shelf parts, the unit above or below, "
                          "table items)", ok=not through,
                     detail="; ".join(through[:30]) + (f" (+{len(through) - 30} more)" if len(through) > 30 else "")
                     or f"{len(self.cables)} routes clear of every body"),
                dict(name="No cable runs through another cable's plug (20 mm in front of a used port)", ok=not plugs,
                     detail="; ".join(plugs[:30]) + (f" (+{len(plugs) - 30} more)" if len(plugs) > 30 else "")
                     or f"{len(zones)} plugs clear"),
                dict(name=f"Every used connector has {self.ACCESS_MM:g} mm of clear space in front of it", ok=not block,
                     detail="; ".join(block[:30]) or f"{len(used)} connectors clear"),
                dict(name="Bench-table cables stay inside the table outline", ok=not outside,
                     detail="; ".join(outside[:20]) or "every table route inside x / z of the table")]

    def _pad_checks(self, used):
        """fixed pad stacks (if_pads / mw_pads): each IF tile path has one stack, screwed onto a port that carries no
        cable, with its OUT carrying the path's one cable; at most PAD_MAX available BW-S values, within 1 dB of the
        target; each IP-50 hop has its A and B stacks."""
        b, d = self.b, self.d
        if not b.pads:
            return []
        bad, val = [], []
        for pid, v in b.pads.items():
            host, inp, out = v["host"], f"{pid}.in", f"{pid}.out"
            if self.ports.get(host, {}).get("mate") != inp or self.ports[inp].get("mate") != host:
                bad.append(f"{pid}: not mated to {host}")
            if host in used:
                bad.append(f"{pid}: its host {host} also carries a cable")
            if len([c for c in self.cables if out in (c["a"], c["b"])]) != 1:
                bad.append(f"{pid}: OUT does not carry exactly one cable")
            if v["chain"] == "if":
                dbs = [x["db"] for x in v["parts"]]
                if len(dbs) > BS.PAD_MAX or any(x not in BS.BW_S_DB for x in dbs) or sum(dbs) != v["db"] or \
                        abs(v["db"] - v["target"]) > 1.0:
                    val.append(f"{pid}: {dbs} = {v['db']} dB for {v['target']} dB")
        paths = {(v["link"], v["tile"]) for v in b.pads.values() if v["chain"] == "if"}
        want = {(k, t) for k in (d["link_ends"] if b.if_pads else ()) for t in range(1, b.tiles + 1)}
        hops = {v["link"] + v["side"] for v in b.pads.values() if v["chain"] == "ip50"}
        want_h = {k + sd for k in (d.get("mw_links", {}) if b.mw_pads else ()) for sd in "ab"}
        miss = sorted(f"{k} tile {t}" for k, t in want - paths) + sorted(want_h - hops)
        n_if = sum(1 for v in b.pads.values() if v["chain"] == "if")
        return [dict(name="Every IF tile path and IP-50 hop end has its fixed pad stack, screwed onto a port that carries "
                          "no cable, its OUT carrying the one cable", ok=not (bad or miss),
                     detail="; ".join(bad + miss) or f"{n_if} IF stacks ({len(want)} tile paths), "
                                                    f"{len(b.pads) - n_if} IP-50 stacks ({len(want_h) // 2} hops)"),
                dict(name=f"Pad stacks: at most {BS.PAD_MAX} available BW-S values, within 1 dB of the A7 target", ok=not val,
                     detail="; ".join(val) or "; ".join(f"{k} {v['target']:g} -> {v['text']} = {v['db']} dB"
                                                         for k, v in b.pad_links.items()) or "no IF pads")]

    def _arm_fit_check(self):
        """coupler CPL stacks: DC block (+ pad) + plug + bend fit under the unit above the coupler tray."""
        if not self.b.coupled:
            return []
        tr = self.dev["CPL-TRAY"]
        top = tr["pos"][1] + tr["size"][1]
        bad, worst = [], 1e9
        for k, v in self.dev.items():
            if v["kind"] != "coupler" or k + ".CPL" not in self.ports:
                continue
            y = self.ports[k + ".CPL"]["pos"][1]
            stack = self.arm_stack_mm(next(s for s, sid in self.b.coupled.items() if k.startswith(sid + ".")))
            need = y + stack + self.PLUG_MM + self.BEND_MM + C.CABLE_DIA["RF"] / 2
            worst = min(worst, top - need)
            if need > top - 3:
                bad.append(f"{k}: needs {need - y:.0f} mm above the CPL port, {top - y:.0f} mm available")
        return [dict(name="Coupler arm stacks (DC block, pad, plug, bend) fit under the unit above the tray", ok=not bad,
                     detail="; ".join(bad) or f"{worst:.0f} mm to spare under the unit above (3 mm needed)")]

    def power_budget(self):
        """typical draw per PDU and for the strip (TENTATIVE: catalog.LOAD_W), against 80 % of a 16 A 230 V circuit."""
        b = self.b

        assert b.n_dcin == len(b.dcin) and sum(1 for k in b.ip50 if k not in b.psu) == len(b.fuse)
        watts = b.load_w                                 # the same estimate the PDU outlet plan used
        out = []
        for p in b.pdus:
            loads = [(f"{p}.O{o}", ld, round(watts(ld))) for ld, (pp, o) in b.outlet.items() if pp == p]
            feed = next(c["a"] for c in self.cables if c["b"] == f"{p}.inlet")
            tot = sum(w for _, _, w in loads)
            out.append(dict(supply=p, circuit=feed, loads=[dict(outlet=o, load=ld, w=w) for o, ld, w in loads],
                            total_w=tot, limit_w=round(C.CIRCUIT_W), ok=tot <= C.CIRCUIT_W))
        feed = next(c["a"] for c in self.cables if c["b"] == "TABLE-STRIP.inlet")
        loads = [(c["a"], c["b"].split(".")[0], round(watts(c["b"].split(".")[0]))) for c in self.cables
                 if c["cls"] == "AC" and c["a"].startswith("TABLE-STRIP.")]
        tot = sum(w for _, _, w in loads)
        out.append(dict(supply="TABLE-STRIP", circuit=feed, loads=[dict(outlet=o, load=ld, w=w) for o, ld, w in loads],
                        total_w=tot, limit_w=round(C.CIRCUIT_W), ok=tot <= C.CIRCUIT_W))
        return out

    def _power_budget_check(self):
        self.budget = self.power_budget()
        self.b.budget = self.budget                          # the blueprint prints the totals with the outlet map
        tot = {}
        for s_ in self.budget:
            for ld in s_["loads"]:
                tot[ld["load"].split(":")[0]] = tot.get(ld["load"].split(":")[0], 0) + ld["w"]
        for k, w in tot.items():
            if k in self.dev:
                self.dev[k]["watts"] = w
        bad = [s_ for s_ in self.budget if not s_["ok"]]
        return [dict(name=f"Each PDU and the strip stays at or below 80 % of 16 A at 230 V ({C.CIRCUIT_W:.0f} W, typical "
                          "draw, estimates)", ok=not bad,
                     detail="; ".join(f"{s_['supply']} ({s_['circuit']}) {s_['total_w']} W" for s_ in self.budget))]

    # body geometry: no overlaps, clearance between loose parts, every part on its support, explicit stacks
    LOOSE_PARENTS = ("shelf", "tray", "attshelf")
    CLEAR_MM = 5.0

    def _ancestors(self, did):
        out = set()
        while self.dev[did].get("parent"):
            did = self.dev[did]["parent"]
            out.add(did)
        return out

    PRE_MM = 1.0                # pad stacks pre-assembled on one splitter (fitted in the hand): bodies must not touch

    def _loose(self, v):
        """a part lying loose on a shelf / tray / the table; a pad stack screwed onto a part on a tray (the splitter-tray
        stacks) counts as loose too: it lies on the tray with its splitter."""
        par = self.dev.get(v.get("parent") or "")
        if v["kind"] == "padstack" and par and par["kind"] != "spider":
            gp = self.dev.get(par.get("parent") or "")
            if gp and gp["kind"] in self.LOOSE_PARENTS:
                return True
        return (v["group"] == "table") or bool(par and par["kind"] in self.LOOSE_PARENTS)

    def _body_checks(self):
        D = list(self.dev.values())
        anc = {v["id"]: self._ancestors(v["id"]) for v in D}
        over, tight, pre = [], [], []
        for i, a in enumerate(D):
            for c in D[i + 1:]:
                if a["id"] in anc[c["id"]] or c["id"] in anc[a["id"]]:
                    continue
                o = [min(a["pos"][k] + a["size"][k], c["pos"][k] + c["size"][k]) - max(a["pos"][k], c["pos"][k])
                     for k in range(3)]
                if all(v > 0.5 for v in o):
                    over.append(f"{a['id']} x {c['id']} ({' x '.join(str(round(v)) for v in o)} mm)")
                elif a["kind"] == c["kind"] == "padstack" and a.get("parent") == c.get("parent") and self._loose(a):
                    pre.append((max(-v for v in o), a["id"], c["id"]))  # one assembly: fitted together in the hand
                elif self._loose(a) and self._loose(c) and not (a.get("stack") and a.get("stack") == c.get("stack")):
                    gap = max(-v for v in o)
                    if gap < self.CLEAR_MM:
                        tight.append(f"{a['id']} / {c['id']} {gap:.1f} mm")
        out = []
        T = C.TABLE
        for v in D:
            par = self.dev.get(v.get("parent") or "")
            if par and par["kind"] in self.LOOSE_PARENTS:
                inside = all(par["pos"][k] - 0.5 <= v["pos"][k] and v["pos"][k] + v["size"][k] <= par["pos"][k] +
                             par["size"][k] + 0.5 for k in range(3))
                if not inside:
                    out.append(f"{v['id']} off {par['id']}")
            elif v["group"] == "table" and not par:
                x0, y0, z0 = v["pos"]
                w, h, dd = v["size"]
                if not (T["x0"] <= x0 and x0 + w <= T["x1"] and T["z0"] <= z0 and z0 + dd <= T["z1"] and y0 >= T["top"] - 0.5):
                    out.append(f"{v['id']} off the table")
        stk = []
        for v in D:
            if v.get("stacked_on"):
                lo = self.dev[v["stacked_on"]]
                lid = lo["pos"][1] + lo["size"][1]
                o = [min(v["pos"][k] + v["size"][k], lo["pos"][k] + lo["size"][k]) - max(v["pos"][k], lo["pos"][k])
                     for k in (0, 2)]
                if abs(v["pos"][1] - (lid + v.get("spacer_mm", 0))) > 0.5 or min(o) < 10:
                    stk.append(f"{v['id']} not seated on {lo['id']}")
                for sid in v.get("spacers", []):              # spacer bars: on the lower lid, under the upper radio
                    sp = self.dev[sid]
                    inside = all(lo["pos"][k] - 0.5 <= sp["pos"][k] and sp["pos"][k] + sp["size"][k] <= lo["pos"][k] +
                                 lo["size"][k] + 0.5 for k in (0, 2))
                    if abs(sp["pos"][1] - lid) > 0.5 or abs(sp["pos"][1] + sp["size"][1] - v["pos"][1]) > 0.5 or not inside:
                        stk.append(f"{sid} does not carry {v['id']} on {lo['id']}")
        nst = sum(1 for v in D if v.get("stacked_on"))
        return [dict(name="No two bodies overlap (rack units, shelf / tray parts, table items, room)", ok=not over,
                     detail="; ".join(over[:30]) + (f" (+{len(over) - 30} more)" if len(over) > 30 else "")
                     or f"{len(D)} bodies, none intersecting"),
                dict(name=f"Loose parts keep {self.CLEAR_MM:g} mm clearance (shelves, trays, table)", ok=not tight,
                     detail="; ".join(tight[:30]) or f"{sum(1 for v in D if self._loose(v))} loose parts; explicit stacks "
                                                     f"({nst}) excepted"),
                *([dict(name=f"Pad stacks on one splitter (fitted together in the hand, splitter-tray step) do not touch "
                             f"({self.PRE_MM:g} mm or more between bodies)", ok=all(g >= self.PRE_MM for g, _, _ in pre),
                        detail="; ".join(f"{x} / {y} {g:.1f} mm" for g, x, y in pre if g < self.PRE_MM) or
                        f"{len(pre)} pairs, closest {min(g for g, _, _ in pre):.1f} mm ({min(pre)[1]} / {min(pre)[2]}); "
                        "torqued off the tray, P2 on a short M-F adapter if they foul")] if pre else []),
                dict(name="Every part sits on its shelf, tray or the table", ok=not out, detail="; ".join(out) or "all inside"),
                # (only where something is stacked: a bench with every item on its own footing has nothing to check)
                *([dict(name="Stacked items are seated on the item below (upper y = lower y + height + spacer)", ok=not stk,
                        detail="; ".join(stk) or f"{nst} stacked")] if nst else []),
                dict(name="No two bodies share a position", ok=len({tuple(v["pos"]) for v in D}) == len(D),
                     detail=f"{len({tuple(v['pos']) for v in D})} distinct positions for {len(D)} bodies")]

    def _tile_access_check(self):
        """no IF cable passes in front of a Spider tile (the 20 x 24 mm the torque wrench needs around the SMA) whose
        own cable is fitted in a LATER step; crossings of tiles fitted earlier are counted as information."""
        owner = {}
        for st in getattr(self, "steps", []):
            for cid in st["cables"]:
                owner.setdefault(cid, st["n"])
        tile_cable = {}
        for c in self.cables:
            for e in self._ends(c):                         # a tile with a pad stack on it: the stack's cable
                if re.match(r"^SB\d+\.M\d+\.T\d+$", e):
                    tile_cable[e] = c["id"]
        tiles = [p for p in self.ports.values() if re.match(r"^SB\d+\.M\d+\.T\d+$", p["id"])]
        later, earlier = [], 0
        for c in self.cables:
            if c["cls"] != "RF":
                continue
            segs = [(p, q) for p, q in zip(c["route"], c["route"][1:]) if p[2] < -5 and q[2] < -5]
            for t in tiles:
                if t["id"] in self._ends(c):
                    continue
                tx, ty = t["pos"][0], t["pos"][1]
                hit = any(max(min(p[0], q[0]) - tx, tx - max(p[0], q[0]), 0) <= 10 and
                          max(min(p[1], q[1]) - ty, ty - max(p[1], q[1]), 0) <= 12 for p, q in segs)
                if not hit:
                    continue
                tc = tile_cable.get(t["id"])
                if tc and owner.get(tc, 0) > owner.get(c["id"], 0):
                    later.append(f"{c['id']} (step {owner.get(c['id'])}) in front of {t['id']} (step {owner.get(tc)})")
                else:
                    earlier += 1
        return [dict(name="No IF cable hangs in front of a tile that is connected later", ok=not later,
                     detail="; ".join(later[:20]) + (f" (+{len(later) - 20} more)" if len(later) > 20 else "") or
                     f"IF runs use the side lanes; {earlier} passes in front of tiles fitted earlier")]

    # output ----------------------------------------------------------------------------------------
    def if_slack(self):
        """lowest slack of an IF / IP-50 SMA cable (stock length over its route), as text for the assumptions."""
        stock = dict(BS.RF_STOCK)                          # exact catalogue length (the text rounds 914 mm to 0.91 m)
        sl = [(stock.get(c["pn"], c["spec_mm"]) / c["route_mm"] - 1, c["id"]) for c in self.cables
              if c["spec_mm"] and c["route_mm"] and (c["cls"] == "RF" or (c["cls"] == "WG" and "CBL-" in c["pn"]))]
        if not sl:
            return "n/a"
        v, i = min(sl)
        return f"{v * 100:.1f} % ({i})"

    def fill(self, text):
        """'{step:<title>}' -> that step's number; '{if_slack}' -> the lowest IF slack."""
        text = re.sub(r"\{step:([^}]+)\}", lambda m: str(self.step_titles.get(m.group(1)) or
                                                          getattr(self, "step_no", {}).get(m.group(1), "?")), text)
        if "{tray_pad_" in text:
            pitch, gap = self.tray_pad_gap()
            text = text.replace("{tray_pad_pitch}", f"{pitch:.0f}").replace("{tray_pad_gap}", f"{gap:.0f}")
        if "{tray_headroom}" in text:
            text = text.replace("{tray_headroom}", f"{self.tray_headroom():.0f}")
        return text.replace("{if_slack}", self.if_slack())

    def to_json(self):
        legend = [dict(cls=k, color=C.CABLE[k], name=C.CABLE_NAME[k], n=sum(1 for c in self.cables if c["cls"] == k))
                  for k in C.CABLE]
        assumed = [self.fill(a) for a in self.assumed + [a for a in self.b.assumptions if a not in self.assumed]]
        return dict(meta=dict(poc=self.d["poc"], name=self.d["name"], drawing=self.d["drawing"], event=self.d.get("event"),
                              switched=self.b.switched, tentative=True),
                    rack=self.rack, table=self.table, devices=list(self.dev.values()), ports=list(self.ports.values()),
                    cables=self.cables, steps=self.steps, checks=self.checks, assumptions=assumed, legend=legend,
                    power_up=self.power_up, power_budget=self.budget, **self.pad_schedule(), **self.bridge_procedure())

    def bridge_procedure(self):
        """scene field 'bridges' (additive, fixed-pad benches with MARL bridges only): one row per bridge (link, TG link
        name, A-end node / sector, B-end sector, agent PC) and the hold / engage / after-the-run texts."""
        b = self.b
        if not b.bridges:
            return {}
        return dict(bridges=dict(rows=[dict(v) for v in b.bridges.values()], hold=BS.bridge_hold_text(b.bridges),
                                 engage=BS.bridge_engage_text(b.bridges), after=b.bridge_after_text(),
                                 tbc="TG link names (link-<A node>-<B node>) TBC with the lab topology; minion-level "
                                     "association with no E2E controller TBC on the N367 firmware (open question)"))

    def pad_schedule(self):
        """scene field 'pad_schedule' (additive, fixed-pad benches only): one row per IF link (target, rounded, fitted
        dB, pads, where the stacks sit, A / B tiles, cables) and one per IP-50 hop (its two stacks and the cable)."""
        b, d = self.b, self.d
        if not b.pads:
            return {}
        rows = []
        for name, pl in b.pad_links.items():
            sa, sb = d["link_ends"][name]
            pids = [k for k, v in b.pads.items() if v["chain"] == "if" and v["link"] == name]
            rows.append(dict(chain="if", link=name, job=pl["job"], setting=pl["setting"], target_db=pl["target"],
                             rounded_db=pl["rounded"], db=pl["db"], pads=pl["pads"], pn=pl["pn"], rule=pl["rule"],
                             stacks=pids, hosts=[b.pads[p]["host"] for p in pids],
                             a_sector=f"{sa[0]}.{sa[1]}", b_sector=f"{sb[0]}.{sb[1]}",
                             a_tiles=["{}.{}.T{}".format(*d["module"][sa], t) for t in range(1, b.tiles + 1)],
                             b_tiles=["{}.{}.T{}".format(*d["module"][sb], t) for t in range(1, b.tiles + 1)],
                             cables=[c["id"] for c in self.cables if c["a"].rsplit(".", 1)[0] in pids],
                             # port check (A7 model): margin above the lowest data MCS, expected SNR / MCS, band
                             **{k: pl[k] for k in ("margin_model_db", "margin_db", "snr_exp_db", "mcs_exp", "snr_min_db",
                                                   "snr_max_db") if k in pl},
                             stack_step=min((st["n"] for st in self.steps if set(pids) & set(st["devices"])), default=None),
                             cable_steps=sorted({st["n"] for st in self.steps for c in st["cables"]
                                                 if c in {x["id"] for x in self.cables if x["a"].rsplit(".", 1)[0] in pids}})))
        for k in (d.get("mw_links", {}) if b.mw_pads else ()):
            st = [f"WGP-{k}-A", f"WGP-{k}-B"]
            rows.append(dict(chain="ip50", link=k, db=sum(b.pads[x]["db"] for x in st),
                             pads=[[p["db"] for p in b.pads[x]["parts"]] for x in st],
                             pn=[[p["pn"] for p in b.pads[x]["parts"]] for x in st], stacks=st,
                             hosts=[b.pads[x]["host"] for x in st],
                             cables=[c["id"] for c in self.cables if c["cls"] == "WG" and
                                     c["a"].rsplit(".", 1)[0] in st]))
        out = dict(pad_schedule=rows)
        if b.split and b.if_pads:
            out["tray_headroom_mm"] = round(self.tray_headroom(), 1)
        out.update(self.port_check())
        return out

    def node_off(self):
        """{TG node: how to switch it off for a pad change} from the wiring: its adaptor's DC-DIST breaker, or the PDU
        outlet of its PoE injector."""
        out = {}
        for c in self.cables:
            if c["cls"] == "DC" and c["a"].startswith("DC-DIST.CB") and not c["b"].endswith(".DCIN"):
                dv = self.dev.get(c["b"].split(".")[0]) or {}
                if dv.get("feeds"):
                    out[dv["feeds"]] = (f"open DC-DIST {c['a'].split('.')[1]} ({dv['id']})",
                                        f"close DC-DIST {c['a'].split('.')[1]}")
        for node, inj in self.b.inj.items():
            ac = next((c for c in self.cables if c["cls"] == "AC" and c["b"].split(".")[0] == inj), None)
            if ac:
                out[node] = (f"switch {ac['a']} ({inj}) OFF", f"switch {ac['a']} ({inj}) ON")
        return out

    def port_check(self):
        """scene field 'port_check' (additive, fixed-pad benches only): after the nodes are up (power-up stage 'nodes')
        and before the radios, every IF link's SNR is read against its expected value (A7 model margin + the lowest data
        MCS threshold); a link outside the band gets trim pads at the B tile (too strong) or a smaller outer pad (too
        weak). Rows per IF link; texts for reading, the bridges and the pad change."""
        b, d = self.b, self.d
        if not b.if_pads:
            return {}
        off = self.node_off()
        nodes_n = next((p["n"] for p in getattr(self, "power_up", []) if p["title"] == "nodes"), None)
        trim = " / ".join(f"BW-S{v}W2+" for v in BS.TRIM_PADS)
        tol = BS.PORT_CHECK_TOL_DB
        rows = []
        for name, pl in b.pad_links.items():
            if "snr_exp_db" not in pl:
                continue
            (an, _), (bn, _) = d["link_ends"][name]
            sa = d["link_ends"][name][0]
            tray = sa in b.split
            nodes = [n for n in (an, bn)]
            outer = pl["pads"][-1] if len(pl["pads"]) > 1 else None
            weak = ("a tray stack cannot be reached with the Spider boxes in: if the link is up at MCS 9 or better, "
                    "accept it and record the reading; otherwise STOP (tray rework, design change)" if tray else
                    (f"swap the outer {outer} dB pad of each of the {b.tiles} A-tile stacks for the next smaller value "
                     f"nearest the shortfall (spares, trim pads), or take it off" if outer else
                     f"swap the {pl['pads'][0]} dB pad of each A-tile stack for a smaller value (order 4)"))
            rows.append(dict(link=name, job="bridge" if name in b.bridges else pl["job"], db=pl["db"], pads=pl["pads"],
                             margin_model_db=pl["margin_model_db"], margin_db=pl["margin_db"],
                             snr_exp_db=pl["snr_exp_db"], snr_min_db=pl["snr_min_db"], snr_max_db=pl["snr_max_db"],
                             mcs_exp=pl["mcs_exp"], stacks_on="tray" if tray else "A tiles",
                             b_tiles=["{}.{}.T{}".format(*d["module"][d["link_ends"][name][1]], t)
                                      for t in range(1, b.tiles + 1)],
                             nodes=nodes, off=[off.get(n, ("?", "?"))[0] for n in nodes],
                             on=[off.get(n, ("?", "?"))[1] for n in nodes],
                             strong=(f"fit trim pads at the B tile of each of the {b.tiles} tile paths (tile jack, "
                                     f"trim pad, then the cable): {trim}, alone or both, nearest the excess"),
                             weak=weak))
        br = [f"{k}: on SRV-E2E tg ignition link -n {v['tg']} --enable, wait until it is up, read it, then "
              f"tg ignition link -n {v['tg']} --disable; take it down from {v['agent'] or 'its A-end agent PC'} "
              f"(tg2 minion dissoc, {v['a_sector']} to {v['b_sector']}, TBC) and check on the controller that it is "
              "down and its ignition reads disabled" for k, v in b.bridges.items()]
        return dict(port_check=dict(
            after_stage=nodes_n,
            when=(f"after power-up stage {nodes_n or '?'} (nodes: every node joined, the bridges held down) and before "
                  "the radios; again after any pad change"),
            read=("for every IF link, on the E2E controller / NMS (SRV-E2E), after the link has been up 2 min: SNR and "
                  "MCS in BOTH directions (the N367's counter names TBC); compare the lower of the two SNRs with the "
                  "expected value; TX power and link adaptation as the lab's node config (TPC off with fixed TX power, "
                  "if the lab runs it that way: TBC)"),
            model=(f"expected SNR = the lowest data-MCS threshold (MCS {BS.TG_MCS_SNR[0][1]}: {BS.TG_MCS_SNR[0][0]:g} dB, "
                   "Terragraph link adaptation defaults) + the A7 model margin of the link at its fitted pads (the "
                   "'Margin up' of the A7 table, less the dB the fitted stack adds over the A7 setting); the A7 model "
                   "follows the MARL paper's radio model, so the band is a first check, not a calibration"),
            accept=f"the lower SNR within +/- {tol:g} dB of the expected value, at the expected MCS or one below",
            strong=(f"SNR more than {tol:g} dB above the band centre (expected: the pads leave out the programmable "
                    f"attenuator's own loss, about {BS.ARM_BUDGET['att_il_db']:g} dB): trim pads at the B tile, the "
                    f"value nearest the excess ({' or '.join(str(v) for v in BS.TRIM_PADS)} dB, or both = "
                    f"{sum(BS.TRIM_PADS)} dB; parts list: {b.tiles} of each), the same on all {b.tiles} tile paths"),
            weak=(f"SNR more than {tol:g} dB below, or the link not up: a smaller outer pad on the A-tile stacks "
                  "(direct links); a tray link is accepted if up at MCS 9 or better, else STOP"),
            bridges=br,
            swap=[("switch both nodes of the link off (the row's 'nodes off'; INJ-HUB also takes every hub link down) "
                   "and wait 30 s: never open an SMA of a powered node"),
                  ("trim pad: unscrew the cable at the B tile (backing wrench on the tile jack), screw the trim pad onto "
                   "the tile jack, the cable onto the trim pad; stack swap: unscrew the cable, then the outer pad (backing "
                   "wrench on the inner pad's body), fit the new pad and the cable"),
                  ("torque every joint to 8 in-lb (0.9 N m), backing wrench on the body of the part you screw onto; do "
                   "the same on all four tile paths of the link; strap the cable within 150 mm as before"),
                  ("note the change on the pad schedule (link, tile, new stack) and on the cable tag"),
                  ("switch the nodes on again (the row's 'nodes on'), wait for them to join (the bridges stay held "
                   "down: ignition is a controller setting) and read the link again")],
            rows=rows))


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
    L += ["", "## Power-up order", ""] + [f"{p['n']}. **{p['title']}**: {p['detail']}" for p in sc.power_up]
    L += ["", "## Assumptions to confirm", ""] + [f"- {a}" for a in js["assumptions"]]
    with open(os.path.join(OUT, f"{prefix}_checks.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    # the cable schedule and parts list from the FINAL (twin-sized) lengths, so the BOM always matches the schedule
    BS.write_tables(sc.b, BS.parts(sc.b), prefix)
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
