#!/usr/bin/env python3
"""Installation guides for the UNITY-6G conducted test benches (PoC 1 / PoC 2).

    python3 twin/guide.py                 # twin HTML + renders + both guides (HTML and PDF)
    python3 twin/guide.py poc1            # one PoC
    python3 twin/guide.py --no-render     # reuse out/twin/<poc>_renders (must match the scene)
    python3 twin/guide.py --no-build      # do not rebuild out/twin/<poc>_twin.html
    python3 twin/guide.py --no-pdf        # HTML guide only

Inputs (not modified): out/twin/<poc>_scene.json (from twin/twin_model.py),
out/<poc>_parts_list.csv, out/<poc>_cable_schedule.md (switch port plans) and,
if present, out/twin/device_questions.md.

Steps: node twin/viewer/build.mjs -> node twin/render.mjs -> per PoC build
out/twin/<poc>_install_guide.html (images referenced from <poc>_renders/) and print
out/twin/<poc>_install_guide.pdf with headless Chromium (A4 landscape, header and
page numbers; JPEG copies of the renders keep the PDF small). The PDF is printed
twice so the table of contents carries page numbers. Every cable id of the scene
must appear in exactly one step table, and the printed layout is checked for
clipped text and over-wide tables: either failing stops the build.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # testbed_design/
OUT = ROOT / "out"
TWIN = OUT / "twin"
VIEWER = ROOT / "twin" / "viewer"
DATE = "2026-11-24"
PRINT_W_PX = 2400  # print copies of the renders (about 240 dpi across the page)

# page geometry (mm): A4 landscape minus the margins
MARGIN = {"top": 14, "bottom": 11, "left": 11, "right": 11}
PAGE_W = 297 - MARGIN["left"] - MARGIN["right"]  # 275
PAGE_H = 210 - MARGIN["top"] - MARGIN["bottom"]  # 185

ID_SCHEME = OrderedDict([
    ("A", ("AC", "AC mains")), ("D", ("DC", "DC power")), ("P", ("POE", "PoE (power only) to node ETH1")),
    ("F", ("FIB", "data to SW-DATA")), ("E", ("ETH", "management to SW-OOB")), ("R", ("RF", "IF coax (SMA)")),
    ("G", ("GNSS", "GNSS coax")), ("W", ("WG", "IP-50 conducted chain")), ("X", ("AUX", "operator station (video, USB)")),
])
KIND_NAMES = {
    "spider": "Siklu Spider box", "ztdat": "programmable attenuator", "tray": "tray", "shelf": "shelf",
    "dcdist": "DC distribution", "fuse": "fuse panel", "rect": "rectifier", "pdu": "switched PDU",
    "sw_data": "data switch", "sw_oob": "management switch", "gnss": "GNSS splitter", "cm": "cable manager",
    "blank": "blanking panel", "minipc": "mini PC", "adaptor": "DC/PoE adaptor", "injector": "PoE injector",
    "splitter": "RF splitter", "coupler": "directional coupler", "transition": "waveguide transition",
    "mwatt": "multi-channel attenuator", "ip50": "IP-50 radio", "server": "server", "rackserver": "rack server",
    "pc": "PC", "specan": "spectrum analyser", "strip": "mains strip", "psu": "power supply", "wall": "wall sockets",
    "antenna": "antenna", "attshelf": "attenuator shelf", "monitor": "monitor", "keyboard": "keyboard",
}
# display palette per cable class (same as twin/viewer/src/model.js): the scene's own
# colours put DC/POE, FIB/GNSS and AC/WG too close on paper; DC, GNSS and WG are banded
DISPLAY_COLORS = {
    "AC": "#c0392b", "DC": "#f2c200", "POE": "#e67e22", "FIB": "#16a085", "ETH": "#2980b9",
    "RF": "#8e44ad", "GNSS": "#8bc34a", "WG": "#ff4da6", "AUX": "#7f8c8d",
}
BANDED = {"DC", "GNSS", "WG"}
CONF_TEXT = {
    "estimate": "size or position estimated (photos, stand-in model)",
    "likely": "from the datasheet of the probable model",
    "confirmed": "known part and size",
}


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def natkey(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ============================================================== data model
class Bench:
    def __init__(self, poc: str):
        self.poc = poc
        self.scene = json.loads((TWIN / f"{poc}_scene.json").read_text())
        s = self.scene
        self.meta = s["meta"]
        self.R = s["rack"]
        self.T = s["table"]
        self.D = OrderedDict((d["id"], d) for d in s["devices"])
        self.P = {p["id"]: p for p in s["ports"]}
        self.C = OrderedDict((c["id"], c) for c in s["cables"])
        self.steps = s["steps"]
        self.portcable = {}
        for c in s["cables"]:
            self.portcable[c["a"]] = c
            self.portcable[c["b"]] = c
        self.children = defaultdict(list)
        for d in s["devices"]:
            if d.get("parent"):
                self.children[d["parent"]].append(d["id"])
        for c in s["cables"]:
            c["color"] = DISPLAY_COLORS.get(c["cls"], c["color"])
        for l in s["legend"]:
            l["color"] = DISPLAY_COLORS.get(l["cls"], l["color"])
        self.legend = {l["cls"]: l for l in s["legend"]}
        self.dark = {"devices": [], "ports": [], "cables": [], "links": [], "nodes": [], "why": {}}
        # each cable belongs to the FIRST step that lists it
        self.owner = {}
        for st in self.steps:
            for cid in st["cables"]:
                self.owner.setdefault(cid, st["n"])
        self.dev_step = {}
        for st in self.steps:
            for did in st["devices"]:
                self.dev_step.setdefault(did, st["n"])
        self.renders = {}
        self.manifest = None

    # ---------------------------------------------------------- helpers
    def dev_of(self, pid):
        p = self.P.get(pid)
        return self.D.get(p["device"]) if p else None

    def other_end(self, c, pid):
        return c["b"] if c["a"] == pid else c["a"]

    def sector(self, pid):
        """'SB2.M1.T1' -> 'HUB.W' (module sector of a Spider tile)."""
        m = re.match(r"^(SB\d+)\.(M\d+)\.T(\d+)$", pid)
        if not m:
            return None
        d = self.D.get(m.group(1))
        for mod in (d or {}).get("modules") or []:
            if mod["m"] == m.group(2):
                return mod.get("sector")
        return None

    def tile_no(self, pid):
        m = re.match(r"^SB\d+\.M\d+\.T(\d+)$", pid)
        return int(m.group(1)) if m else None

    def what(self, d):
        t = d.get("title")
        if t and "·" in t:
            t = t.split("·", 1)[1].strip()
        return t or KIND_NAMES.get(d["kind"], d["kind"])

    def faces(self, d):
        """connector count per face, e.g. 'front: 16 SMA; rear: 4 RJ45, 4 SFP'."""
        by = defaultdict(Counter)
        for pid in d.get("ports") or []:
            p = self.P.get(pid)
            if not p:
                continue
            n = p["normal"]
            face = "front" if n[2] < -0.5 else "rear" if n[2] > 0.5 else "top" if n[1] > 0.5 else "bottom" if n[1] < -0.5 else "side"
            by[face][ctype(p["type"])] += 1
        parts = []
        for face in ("front", "rear", "top", "side", "bottom"):
            if by[face]:
                parts.append(f"{face}: " + ", ".join(f"{n} {t}" for t, n in sorted(by[face].items(), key=lambda kv: -kv[1])))
        return "; ".join(parts)

    def below(self, d):
        """device that d sits on (bench table stacks; a gap of up to 15 mm still
        counts, e.g. rubber feet). The highest such device wins."""
        if d["group"] != "table" or d.get("parent"):
            return None
        best = None
        for o in self.scene["devices"]:
            if o is d or o["group"] != "table" or o.get("parent"):
                continue
            gap = d["pos"][1] - (o["pos"][1] + o["size"][1])
            if not -1 <= gap <= 15:
                continue
            ox = min(d["pos"][0] + d["size"][0], o["pos"][0] + o["size"][0]) - max(d["pos"][0], o["pos"][0])
            oz = min(d["pos"][2] + d["size"][2], o["pos"][2] + o["size"][2]) - max(d["pos"][2], o["pos"][2])
            if ox > 10 and oz > 10 and (best is None or o["pos"][1] + o["size"][1] > best["pos"][1] + best["size"][1]):
                best = o
        return best

    def on_top(self, d):
        """devices standing on d."""
        return [o for o in self.scene["devices"] if self.below(o) is d]

    def where(self, d):
        T = self.T
        w, h, dp = (round(v) for v in d["size"])
        if d["group"] == "rack":
            if not d.get("parent"):
                return f"{d.get('u') or '?'} · {w} × {h} mm face, {dp} mm deep"
            par = self.D[d["parent"]]
            return (f"on {par['id']} ({par.get('u')}): {round(d['pos'][0])} mm from the left ear "
                    f"(left as seen from the FRONT), {round(d['pos'][2])} mm behind the front rail")
        if d["group"] == "table":
            if d.get("parent"):
                return f"on {d['parent']} (on its waveguide flange)"
            b = self.below(d)
            s = (f"table: {round(d['pos'][0] - T['x0'])} mm from the rack-side end, "
                 f"{round(d['pos'][2] - T['z0'])} mm from the front edge")
            if b:
                s += f"; stacked on {b['id']}"
            return s
        return "room: " + (d.get("note") or "")

    # ---------------------------------------------------------- signal paths
    def walk_out(self, pid):
        """From a port, follow the cable and passive parts (splitter, coupler,
        waveguide transition) until a terminal port. Returns (tokens, end_pid)."""
        toks = []
        seen = set()
        while pid not in seen:
            seen.add(pid)
            c = self.portcable.get(pid)
            if not c:
                return toks, pid
            o = self.other_end(c, pid)
            toks.append(("cable", c["id"]))
            d = self.dev_of(o)
            name = self.P[o]["name"] if o in self.P else ""
            # ("dev", id, entry port name, exit port name): read outward from the start
            if d and d["kind"] == "splitter" and name in ("P1", "P2"):
                toks.append(("dev", d["id"], name, "S"))
                pid = f"{d['id']}.S"
                continue
            if d and d["kind"] == "coupler" and name in ("THRU", "CPL"):
                toks.append(("dev", d["id"], name, "IN"))
                pid = f"{d['id']}.IN"
                continue
            if d and d["kind"] == "transition":
                other = [q for q in d["ports"] if q != o]
                if other:
                    toks.append(("dev", d["id"], name, self.P[other[0]]["name"]))
                    pid = other[0]
                    continue
            return toks, o
        return toks, pid


def ctype(t):
    """connector type for print: 'C6?' -> 'C6 (TBC)'."""
    return re.sub(r"^(.*)\?$", r"\1 (TBC)", t or "")


def path_label(tok, rev):
    """label of a passive part in a signal path, read left to right. walk_out()
    reads outward from the attenuator, so an A-side path printed in reverse swaps
    entry and exit: R001 -> SPL.t1 (S->P1) -> R009."""
    _, _, a, b = tok
    if rev:
        a, b = b, a
    if "CPL" in (a, b):
        return f"{a}→{b} (coupled arm: samples IN)"
    return f"{a}→{b}"


# ============================================================== renders
def build_viewer(pocs):
    if not (VIEWER / "node_modules" / "three").exists():
        run(["npm", "ci", "--no-audit", "--no-fund"], cwd=VIEWER)
    run(["node", ROOT / "twin" / "viewer" / "build.mjs", *pocs], cwd=ROOT)


def render(pocs):
    run(["node", ROOT / "twin" / "render.mjs", *pocs], cwd=ROOT)


def load_manifest(b: Bench):
    mp = TWIN / f"{b.poc}_renders" / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"{mp} missing: run without --no-render")
    m = json.loads(mp.read_text())
    want = [c["id"] for c in b.scene["cables"]]
    if m.get("cable_ids") != want or m.get("n_steps") != len(b.steps) or m.get("n_devices") != len(b.D):
        raise SystemExit(f"{mp} does not match {b.poc}_scene.json (scene regenerated?): "
                         "rebuild with 'python3 twin/guide.py' (without --no-render)")
    for r in m["renders"]:
        if not (TWIN / f"{b.poc}_renders" / r["file"]).exists():
            raise SystemExit(f"render {r['file']} missing: rebuild the renders")
    b.manifest = m
    b.renders = {r["name"]: r for r in m["renders"]}
    if m.get("dark"):
        b.dark = m["dark"]


def make_print_copies(b: Bench, dest: Path) -> dict:
    """JPEG copies (<= PRINT_W_PX wide) of every render for the PDF."""
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, r in b.renders.items():
        src = TWIN / f"{b.poc}_renders" / r["file"]
        dst = dest / f"{name}.jpg"
        im = Image.open(src).convert("RGB")
        if im.width > PRINT_W_PX:
            im = im.resize((PRINT_W_PX, round(im.height * PRINT_W_PX / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=86, optimize=True, subsampling=0)
        out[name] = dst
    return out


# ============================================================== HTML pieces
CSS = r"""
@page { size: A4 landscape; }
:root { --ink:#1f2933; --ink2:#4b5563; --ink3:#6b7280; --line:#d5dae1; --line2:#e8ebef; --amber:#8a5a00; --amberbg:#fff4d6;
  --blue:#1d4ed8; --bluebg:#e0ecff; --green:#1e7b3c; --greenbg:#e3f6ea; --red:#b42318; --redbg:#fde8e7; --mono:'DejaVu Sans Mono',Menlo,monospace; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: Inter, 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: var(--ink);
  background: #fff; font-variant-ligatures: none; }
@media screen { body { background: #e9ecf0; } .sheet { width: 275mm; margin: 8mm auto; background: #fff; padding: 0; box-shadow: 0 1px 6px rgba(0,0,0,.15); }
  section.pg { padding: 6mm; border-bottom: 1px dashed #c8ced6; } }
section.pg { break-before: page; }
section.pg.first { break-before: auto; }
h1 { font-size: 19pt; line-height: 1.15; margin: 0 0 3mm; letter-spacing: -0.01em; }
h1 .num { color: var(--ink3); font-weight: 600; margin-right: 2mm; }
h2 { font-size: 14.5pt; margin: 0; line-height: 1.2; }
h2.sec { font-size: 17pt; line-height: 1.15; margin: 0 0 3mm; letter-spacing: -0.01em; }
h3 { font-size: 11pt; margin: 4mm 0 1.6mm; break-after: avoid; }
p { margin: 0 0 2mm; }
.lead { font-size: 10.5pt; color: var(--ink2); margin-bottom: 3mm; }
.mono { font-family: var(--mono); letter-spacing: -0.02em; }
.muted { color: var(--ink3); }
a { color: var(--blue); text-decoration: none; }
.tent { display: inline-block; background: var(--amberbg); color: var(--amber); border: 1px solid #f0b429; border-radius: 1mm;
  font-weight: 800; letter-spacing: .05em; padding: .2mm 1.6mm; font-size: 8.5pt; }
.chip { display: inline-block; font-size: 8pt; font-weight: 700; padding: .1mm 1.3mm; border-radius: .8mm; white-space: nowrap; }
.chip.estimate { background: var(--amberbg); color: var(--amber); }
.chip.likely { background: var(--bluebg); color: var(--blue); }
.chip.confirmed { background: var(--greenbg); color: var(--green); }
.chip.ok { background: var(--greenbg); color: var(--green); }
.chip.bad, .chip.off { background: var(--redbg); color: var(--red); }
.chip.kind { background: #eef0f3; color: var(--ink2); }
.sw { display: inline-block; width: 3.2mm; height: 3.2mm; border-radius: .6mm; vertical-align: -0.5mm; margin-right: 1.2mm; }
.cid { font-family: var(--mono); font-weight: 700; white-space: nowrap; }
td .cid { white-space: normal; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 8.5pt; line-height: 1.3; margin: 0 0 2mm; }
th { text-align: left; font-weight: 700; color: var(--ink2); background: #f2f4f7; border-bottom: 1px solid var(--line); padding: 1mm 1.4mm; vertical-align: bottom; }
td { border-bottom: 1px solid var(--line2); padding: .8mm 1.4mm; vertical-align: top; overflow-wrap: anywhere; word-break: normal; }
td.mono, td .mono { overflow-wrap: anywhere; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
tr.sub td { background: #fafbfc; color: var(--ink2); }
table.tight { line-height: 1.22; }
table.tight td { padding: .35mm 1.4mm; }
table.tight th { padding: .6mm 1.4mm; }
td.n, th.n { text-align: right; }
td.c, th.c { text-align: center; }
.tcap { font-size: 8.5pt; color: var(--ink3); margin: -1mm 0 2mm; }
figure { margin: 0 0 2mm; break-inside: avoid; }
figure .imgbox { position: relative; margin: 0 auto; border: 1px solid var(--line); border-radius: 1mm; overflow: hidden; background: #f6f7f9; }
figure .imgbox img { display: block; width: 100%; height: 100%; }
figcaption { font-size: 8.5pt; color: var(--ink3); margin-top: 1mm; text-align: center; }
figcaption b { color: var(--ink2); }
.ov { position: absolute; transform: translate(-50%, -50%); font: 800 7.5pt/1 Inter, Arial, sans-serif; color: #fff; background: #1f2933;
  border-radius: 1.2mm; padding: .5mm 1.1mm; white-space: nowrap; border: .3mm solid #fff; }
.ov.a { background: #6d28d9; }
.ov.b { background: #0f766e; }
.ov.t { background: rgba(31,41,51,.86); }
.ov.big { font-size: 9pt; padding: .7mm 1.5mm; }
.cols { display: flex; gap: 6mm; align-items: flex-start; }
.cols > * { min-width: 0; }
.box { border: 1px solid var(--line); border-radius: 1.5mm; padding: 2.5mm 3.5mm; background: #fbfcfd; }
.box.warn { border-color: #f0b429; background: #fffaf0; }
.box.note { border-color: #c7d2fe; background: #f5f7ff; }
ul { margin: 0 0 2mm; padding-left: 5mm; }
ol { margin: 0 0 2mm; padding-left: 7mm; }
li { margin: 0 0 1mm; }
/* cover */
.cover { display: flex; gap: 8mm; height: 180mm; }
.cover .l { flex: 0 0 118mm; display: flex; flex-direction: column; }
.cover .r { flex: 1; display: flex; flex-direction: column; justify-content: center; }
.cover .proj { font-size: 10.5pt; color: var(--ink3); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.cover h1 { font-size: 27pt; margin: 2mm 0 1mm; }
.cover .pocname { font-size: 11.5pt; color: var(--ink2); margin-bottom: 4mm; }
.cover .banner { background: var(--amberbg); border: 1.2px solid #f0b429; border-radius: 2mm; padding: 3mm 4mm; color: #5c3d00; margin-bottom: 4mm; }
.cover .banner b.big { display: block; font-size: 15pt; letter-spacing: .06em; color: var(--amber); margin-bottom: 1mm; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2mm; margin: 2mm 0 4mm; }
.stat { border: 1px solid var(--line); border-radius: 1.5mm; padding: 1.8mm 2.5mm; }
.stat b { display: block; font-size: 15pt; line-height: 1.1; }
.stat span { font-size: 8.5pt; color: var(--ink3); }
.meta-t td { border: 0; padding: .4mm 1.4mm .4mm 0; font-size: 9.5pt; }
/* toc */
.toc { columns: 2; column-gap: 10mm; font-size: 9.5pt; }
.toc .e { display: flex; gap: 2mm; break-inside: avoid; padding: .55mm 0; border-bottom: 1px dotted #d9dde3; }
.toc .e.top { font-weight: 700; margin-top: 1.6mm; border-bottom: 1px solid var(--line); }
.toc .e .t { flex: 1; min-width: 0; }
.toc .e .p { flex: 0 0 9mm; text-align: right; font-variant-numeric: tabular-nums; color: var(--ink2); }
.toc .e.sub .t { padding-left: 4mm; }
/* steps */
.stephead { display: flex; align-items: center; gap: 4mm; margin-bottom: 2mm; }
.stepno { flex: 0 0 auto; min-width: 13mm; height: 13mm; border-radius: 2mm; background: var(--ink); color: #fff; font-size: 17pt; font-weight: 800;
  display: flex; align-items: center; justify-content: center; padding: 0 2mm; }
.stephead .sub { font-size: 9pt; color: var(--ink3); margin-top: .5mm; }
.stephead .chips { margin-left: auto; text-align: right; }
.steptext { font-size: 10.5pt; border-left: 1.2mm solid var(--ink); background: #f6f7f9; padding: 1.8mm 3mm; margin: 0 0 2.5mm; border-radius: 0 1mm 1mm 0; }
.steptext .why { display: block; font-size: 9.5pt; color: var(--ink2); margin-top: .8mm; }
.newpage { break-before: page; }
.keep { break-inside: avoid; }
.legend-row { display: inline-flex; align-items: center; margin: 0 3mm 1mm 0; font-size: 9pt; }
.md h1 { font-size: 13pt; margin: 3mm 0 2mm; }
.md p, .md li { font-size: 10pt; }
.small { font-size: 9.5pt; }
.checkbox { display: inline-block; width: 3.4mm; height: 3.4mm; border: .35mm solid var(--ink2); border-radius: .5mm; vertical-align: -0.6mm; margin-right: 1.5mm; }
svg text { font-family: Inter, Arial, sans-serif; }
/* safety, hold, tags */
.box.safety { border: 1.2px solid #d92d20; background: #fff6f5; padding: 2mm 3.2mm; margin: 0 0 2.5mm; }
.box.safety .h { color: #b42318; font-weight: 800; letter-spacing: .02em; }
.box.safety ul { margin: .8mm 0 0; }
.box.safety li { margin: 0 0 .6mm; }
.callout { border-left: 1.4mm solid #d92d20; background: #fff6f5; padding: 1.4mm 3mm; margin: 0 0 2.2mm; border-radius: 0 1mm 1mm 0; font-size: 9.5pt; }
.callout b.h { color: #b42318; }
.callout.info { border-left-color: #1d4ed8; background: #f3f7ff; }
.callout.info b.h { color: #1d4ed8; }
.hold { border: 1.6px solid #b42318; background: #fde8e7; color: #7a1610; padding: 2mm 3.5mm; margin: 0 0 2.5mm; border-radius: 1.5mm; font-size: 10.5pt; }
.hold b.big { font-size: 13pt; letter-spacing: .06em; margin-right: 2mm; }
figure .tagfig { position: relative; margin: 0 auto; }
figure .tagfig img { position: absolute; left: 0; border: 1px solid var(--line); border-radius: .8mm; background: #f6f7f9; }
figure .tagfig svg { position: absolute; left: 0; top: 0; overflow: visible; }
.sw.band { width: 4.6mm; }
.pair { display: flex; gap: 4mm; align-items: flex-start; justify-content: center; }
.pair > figure { flex: 0 0 auto; }
"""

TAG_FS = 2.65  # mm: 7.5 pt tags on the pictures
TAG_LH = 3.2   # mm per tag line
TAG_CW = 1.66  # mm per character (DejaVu Sans Mono bold at 7.5 pt is 0.6 em, plus slack)


def chip(conf):
    c = conf or "estimate"
    return f'<span class="chip {esc(c)}">{esc(c)}</span>'


def _dark(hexc, k=0.3):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (round(r * k), round(g * k), round(b * k))


def _lum(hexc):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def swatch(color):
    banded = color in {DISPLAY_COLORS[k] for k in BANDED}
    if banded:
        bg = f"repeating-linear-gradient(90deg,{color} 0 1.15mm,{_dark(color)} 1.15mm 1.65mm)"
        return f'<span class="sw band" style="background:{bg}"></span>'
    return f'<span class="sw" style="background:{esc(color)}"></span>'


class Guide:
    def __init__(self, b: Bench, img_src):
        self.b = b
        self.img_src = img_src  # name -> URL used in <img src>
        self.toc = []  # (level, anchor, text)
        self.sections = []
        self.cable_rows = Counter()  # cable id -> times listed in a step table
        self.table_w = PAGE_W  # width (mm) of the next table, for the height estimate
        self.last_est = 0.0

    # ---------------------------------------------------------- figures
    def fig(self, name, max_w, max_h, caption="", overlays=None, cls=""):
        r = self.b.renders.get(name)
        if not r:
            raise SystemExit(f"{self.b.poc}: render '{name}' missing from the manifest")
        aspect = r["px"]["w"] / r["px"]["h"]
        w = min(max_w, max_h * aspect)
        h = w / aspect
        ov = ""
        for o in overlays or []:
            ov += (f'<span class="ov {esc(o.get("cls", ""))}" style="left:{o["x"] * 100:.2f}%;top:calc({o["y"] * 100:.2f}% + {o.get("dy", 0):.1f}mm)">'
                   f'{esc(o["text"])}</span>')
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        return (f'<figure class="{cls}"><div class="imgbox" style="width:{w:.1f}mm;height:{h:.1f}mm">'
                f'<img src="{esc(self.img_src(name))}" alt="{esc(r["title"])}">{ov}</div>{cap}</figure>')

    def fig_tags(self, name, max_w, max_h, tags, caption="", brackets=None, img_w=None):
        """A picture with ID tags around it, each joined by a leader line to the
        point it names. tags: [{x, y (fractions of the PNG), lines: [str], side:
        'top' | 'bottom' (default: nearer edge), fill, ink}]; tags outside the
        picture are dropped. brackets: [{x0, x1, text}] drawn under the picture.
        The tags are packed in rows (tiers) above / below the picture."""
        r = self.b.renders.get(name)
        if not r:
            raise SystemExit(f"{self.b.poc}: render '{name}' missing from the manifest")
        aspect = r["px"]["w"] / r["px"]["h"]
        tags = [t for t in tags if 0 <= t["x"] <= 1 and 0 <= t["y"] <= 1]
        for t in tags:
            t.setdefault("side", "top" if t["y"] < 0.5 else "bottom")
            t["tw"] = max(len(l) for l in t["lines"]) * TAG_CW + 2.2
            t["th"] = len(t["lines"]) * TAG_LH + 0.9
        lead = 3.2  # minimum leader length (mm)

        def pack(side_tags, w):
            if not side_tags:
                return 0.0, []
            ts = sorted(side_tags, key=lambda t: t["x"])
            th = max(t["th"] for t in ts)
            best = None
            for n in range(1, 8):
                rows = [ts[i::n] for i in range(n)]
                out = []
                ok = True
                shift = 0.0
                for k, row in enumerate(rows):
                    pos = [min(max(t["x"] * w, t["tw"] / 2), w - t["tw"] / 2) for t in row]
                    for i in range(1, len(row)):
                        pos[i] = max(pos[i], pos[i - 1] + (row[i - 1]["tw"] + row[i]["tw"]) / 2 + 0.8)
                    if row and pos[-1] + row[-1]["tw"] / 2 > w:
                        pos[-1] = w - row[-1]["tw"] / 2
                        for i in range(len(row) - 2, -1, -1):
                            pos[i] = min(pos[i], pos[i + 1] - (row[i]["tw"] + row[i + 1]["tw"]) / 2 - 0.8)
                    if row and pos[0] - row[0]["tw"] / 2 < -0.01:
                        ok = False
                    for t, x in zip(row, pos):
                        out.append((t, x, k))
                        shift = max(shift, abs(x - t["x"] * w))
                if not ok:
                    continue
                if best is None or shift < best[2] - 2.0:
                    best = (n, out, shift)
                if shift <= max(28.0, 0.3 * w):
                    best = (n, out, shift)
                    break
            if best is None:  # too many tags for the width: one tag per row
                best = (len(ts), [(t, w / 2, i) for i, t in enumerate(ts)], 0)
            n, out, _ = best
            return n * (th + 1.3) + lead, [(t, x, k, th) for t, x, k in out]

        w0 = w = img_w or min(max_w, max_h * aspect)
        br_h = 7.0 if brackets else 0.0
        for _ in range(3):
            h = w / aspect
            top_h, top = pack([t for t in tags if t["side"] == "top"], w)
            bot_h, bot = pack([t for t in tags if t["side"] == "bottom"], w)
            total = top_h + h + bot_h + br_h
            if total <= max_h + 1.0 or img_w or w <= 0.62 * w0 + 0.1:
                break
            # make room for the tag rows, but never shrink the picture below 62 %:
            # a smaller picture only needs more rows of tags
            w = max(0.62 * w0, min(max_w, (max_h - top_h - bot_h - br_h) * aspect))
        H = top_h + h + br_h + bot_h
        g_lines, g_tags = [], []
        for side, items, y_img in (("top", top, top_h), ("bottom", bot, top_h + h + br_h)):
            for t, x, k, th in items:
                ax, ay = t["x"] * w, top_h + t["y"] * h
                if side == "top":
                    ty = top_h - lead - (k + 1) * (th + 1.3) + 1.3
                    ly = ty + t["th"]
                else:
                    ty = y_img + lead + k * (th + 1.3)
                    ly = ty
                g_lines.append(f'<line x1="{x:.2f}" y1="{ly:.2f}" x2="{ax:.2f}" y2="{ay:.2f}" stroke="#1f2933" stroke-width="0.24"/>'
                               f'<circle cx="{ax:.2f}" cy="{ay:.2f}" r="0.8" fill="#1f2933" stroke="#fff" stroke-width="0.35"/>')
                fill = t.get("fill", "#1f2933")
                ink = t.get("ink") or ("#111827" if _lum(fill) > 0.45 else "#ffffff")
                txt = "".join(f'<text x="{x:.2f}" y="{ty + 0.45 + (i + 0.78) * TAG_LH:.2f}" text-anchor="middle" font-size="{TAG_FS}" '
                              f'font-family="DejaVu Sans Mono, monospace" font-weight="{700 if i == 0 else 400}" fill="{ink}">{esc(l)}</text>'
                              for i, l in enumerate(t["lines"]))
                g_tags.append(f'<rect x="{x - t["tw"] / 2:.2f}" y="{ty:.2f}" width="{t["tw"]:.2f}" height="{t["th"]:.2f}" rx="0.7" '
                              f'fill="{fill}" stroke="{t.get("stroke", "#ffffff")}" stroke-width="0.3"/>{txt}')
        for bk in brackets or []:
            x0, x1 = bk["x0"] * w, bk["x1"] * w
            yb = top_h + h + 1.2
            g_tags.append(f'<path d="M{x0:.2f},{yb:.2f} v1.6 H{x1:.2f} v-1.6" fill="none" stroke="{bk.get("color", "#6d28d9")}" stroke-width="0.45"/>'
                          f'<text x="{(x0 + x1) / 2:.2f}" y="{yb + 4.9:.2f}" text-anchor="middle" font-size="3.0" font-weight="800" '
                          f'fill="{bk.get("color", "#6d28d9")}">{esc(bk["text"])}</text>')
        svg = (f'<svg viewBox="0 0 {w:.2f} {H:.2f}" width="{w:.2f}mm" height="{H:.2f}mm" xmlns="http://www.w3.org/2000/svg">'
               + "".join(g_lines) + "".join(g_tags) + "</svg>")
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        return (f'<figure><div class="tagfig" style="width:{w:.1f}mm;height:{H:.1f}mm">'
                f'<img src="{esc(self.img_src(name))}" alt="{esc(r["title"])}" style="top:{top_h:.2f}mm;width:{w:.2f}mm;height:{h:.2f}mm">'
                f'{svg}</div>{cap}</figure>')

    def section(self, anchor, title, body, level=1, num=None, first=False, toc_text=None):
        heading = f"{num} {title}" if num else title
        self.toc.append((level, anchor, toc_text or heading, heading))
        tag = "h1" if level == 1 else 'h2 class="sec"'
        h = f'<{tag} id="{esc(anchor)}">{f"<span class=num>{esc(num)}</span> " if num else ""}{esc(title)}</{tag[:2]}>'
        self.sections.append(f'<section class="pg{" first" if first else ""}">{h}{body}</section>')

    # ---------------------------------------------------------- tables
    def table(self, cols, rows, widths=None, cls=""):
        """cols: list of (label, css class); rows: list of lists of HTML cells (or (html, tr-class))."""
        cg = ""
        if widths:
            tot = sum(widths)
            cg = "<colgroup>" + "".join(f'<col style="width:{w / tot * 100:.2f}%">' for w in widths) + "</colgroup>"
        th = "".join(f'<th class="{c}">{l}</th>' for l, c in cols)
        body = []
        fr = [w / sum(widths) for w in widths] if widths else [1 / len(cols)] * len(cols)
        tight = "tight" in cls
        est = 7.0
        for r in rows:
            trc = ""
            if isinstance(r, tuple):
                r, trc = r
            body.append(f'<tr class="{trc}">' + "".join(f'<td class="{cols[i][1]}">{v}</td>' for i, v in enumerate(r)) + "</tr>")
            lines = 1
            for i, v in enumerate(r):
                wmm = max(5.0, fr[i] * self.table_w - 2.8)
                n = sum(max(1, -(-len(html.unescape(re.sub(r"<[^>]+>", "", seg))) // max(1, int(wmm * 0.64)))) for seg in re.split(r"<br\s*/?>", str(v)))
                lines = max(lines, n)
            est += lines * 3.9 + (0.9 if tight else 1.9)
        self.last_est = est
        return f'<table class="{cls}">{cg}<thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'

    def cable_table(self, ids, with_class=False):
        b = self.b
        rows = []
        for cid in ids:
            c = b.C[cid]
            self.cable_rows[cid] += 1

            def end(pid, raw):
                s = f'<span class="mono">{esc(pid)}</span>'
                if raw and raw != pid:
                    s += f'<br><span class="muted">{esc(raw)}</span>'
                return s

            ln = f"<b>{esc(c.get('length') or '?')}</b>"
            if c.get("route_mm") is not None:
                ln += f' <span class="muted">· route {round(c["route_mm"])}</span>'
            note = []
            if c.get("vlan"):
                note.append(f'VLAN <b class="mono">{esc(c["vlan"])}</b>')
            if c.get("note"):
                note.append(esc(c["note"]))
            if c["a"] in b.P and b.dev_of(c["a"])["id"] in b.meta.get("switched", []) or \
                    c["b"] in b.P and b.dev_of(c["b"])["id"] in b.meta.get("switched", []):
                note.append('<span class="chip off">dead after t0</span>')
            first = f'{swatch(c["color"])}<span class="cid">{esc(cid)}</span>'
            if with_class:
                first += f'<br><span class="muted">{esc(c["cls"])}</span>'
            rows.append([first, end(c["a"], c.get("a_raw")), end(c["b"], c.get("b_raw")),
                         esc(c.get("part") or ""), ln, "<br>".join(note)])
        return self.table([("ID", ""), ("From", ""), ("To", ""), ("Part", ""), ("Length · route mm", ""), ("VLAN / note", "")],
                          rows, widths=[8, 17, 17, 21, 13, 24])

    def device_table(self, ids):
        b = self.b
        rows = []
        for i, did in enumerate(ids, 1):
            d = b.D[did]
            what = f"<b>{esc(b.what(d))}</b><br>{esc(d.get('sub') or '')}"
            notes = []
            if d.get("note"):
                notes.append(esc(d["note"]))
            f = b.faces(d)
            if f:
                notes.append(f'<span class="muted">connectors {esc(f)}</span>')
            kids = b.children.get(did) or []
            if kids:
                notes.append(f'<span class="muted">carries {len(kids)}: {esc(", ".join(kids[:12]))}{" …" if len(kids) > 12 else ""}</span>')
            if did in b.meta.get("switched", []):
                notes.append('<span class="chip off">switched OFF at t0</span>')
            rows.append([f'<span class="cid">{esc(did)}</span>', what, esc(b.where(d)), "<br>".join(notes), chip(d.get("conf"))])
        return self.table([("ID", ""), ("What it is / part", ""), ("Where <span class=\"muted\">(left = as seen from the FRONT)</span>", ""), ("Mounting notes", ""), ("Conf.", "")],
                          rows, widths=[13, 25, 24, 30, 8])

    # ============================================================== sections
    def cover(self):
        b = self.b
        m = b.meta
        nrack = sum(1 for d in b.D.values() if d["group"] == "rack" and not d.get("parent"))
        ntable = sum(1 for d in b.D.values() if d["group"] == "table")
        tw = f"{b.poc}_twin.html"
        hero = self.fig("overview_iso", 150, 120, caption=f"Whole bench (iso view): rack on the left, bench table to its right. Interactive version: <b>{esc(tw)}</b>")
        stats = "".join(f'<div class="stat"><b>{v}</b><span>{esc(k)}</span></div>' for k, v in [
            ("rack U used of " + str(b.R["units"]), b.R["used"]), ("devices", len(b.D)), ("cables", len(b.C)), ("installation steps", len(b.steps)),
        ])
        body = f"""
<div class="cover">
 <div class="l">
  <div class="proj">UNITY-6G · conducted Terragraph test bench</div>
  <h1>{esc(m['poc'])} installation guide</h1>
  <div class="pocname">{esc(m['name'])}</div>
  <div class="banner"><b class="big">TENTATIVE</b>
   The geometry is partly estimated: sizes, depths and positions come from photos, datasheets and stand-in models
   (see the assumptions, section 7). Use this guide to plan and to dry-fit; check every dimension marked
   <span class="chip estimate">estimate</span> on site before you cut, drill or order.</div>
  <div class="stats">{stats}</div>
  <p><b>What this guide covers.</b> How to build the {esc(b.R['part'])} rack U by U ({nrack} rack items), how to set
   out the bench table ({ntable} items), and how to connect every one of the {len(b.C)} cables port to port, in
   {len(b.steps)} steps. Then the switch, PDU and DC port maps, the install checks and the power-up order{', and the t0 event procedure' if m.get('event') else ''}.</p>
  <table class="meta-t"><tbody>
   <tr><td class="muted">Drawing</td><td class="mono">{esc(m.get('drawing'))}</td></tr>
   <tr><td class="muted">Date</td><td>{DATE} · status <b>TENTATIVE</b></td></tr>
   <tr><td class="muted">Interactive twin</td><td><a href="{esc(tw)}" class="mono">{esc(tw)}</a> (same folder; opens offline in a browser:
     search any cable or device ID, follow the same steps with the slider)</td></tr>
   <tr><td class="muted">Pictures</td><td class="mono">{esc(b.poc)}_renders/</td></tr>
   <tr><td class="muted">Tables</td><td class="mono">{esc(b.poc)}_cable_schedule.md, {esc(b.poc)}_parts_list.csv</td></tr>
  </tbody></table>
 </div>
 <div class="r">{hero}</div>
</div>"""
        self.sections.append(f'<section class="pg first">{body}</section>')

    def toc_page(self, pages=None):
        ents = []
        for level, anchor, text, _ in self.toc:
            p = (pages or {}).get(anchor, "")
            cls = "top" if level == 1 else "sub"
            ents.append(f'<div class="e {cls}"><span class="t"><a href="#{esc(anchor)}">{esc(text)}</a></span><span class="p">{p}</span></div>')
        return f'<section class="pg"><h1 id="toc">Contents</h1><div class="toc">{"".join(ents)}</div></section>'

    # ---------------------------------------------------------- 1 how to read
    def how_to_read(self):
        b = self.b
        used_prefix = Counter((c["id"][0], c["cls"]) for c in b.C.values())
        for (pre, cls), n in used_prefix.items():
            if pre in ID_SCHEME and ID_SCHEME[pre][0] != cls:
                raise SystemExit(f"{b.poc}: cable id prefix {pre} used for class {cls}, expected {ID_SCHEME[pre][0]}")
        rows = []
        for pre, (cls, nm) in ID_SCHEME.items():
            ids = [c for c in b.C.values() if c["cls"] == cls]
            if not ids:
                continue
            lg = b.legend.get(cls, {})
            parts = Counter(c.get("pn") or c.get("part") for c in ids).most_common(2)
            steps = sorted({b.owner[c["id"]] for c in ids})
            rows.append([f'{swatch(lg.get("color", ids[0]["color"]))}<b class="mono">{esc(cls)}</b>', f'<b class="mono">{pre}</b>001…',
                         esc(lg.get("name", nm)), str(len(ids)), esc(", ".join(str(p) for p, _ in parts)),
                         ", ".join(f'<a href="#step-{s}">{s}</a>' for s in steps)])
        legend = self.table([("Class / colour", ""), ("ID", ""), ("What it carries", ""), ("Cables", "n"), ("Typical part", ""), ("Step", "")],
                            rows, widths=[16, 10, 30, 8, 26, 10])
        loads = any("ANNE-50" in (r.get("part number") or "") for r in self.parts())
        body = f"""
<p class="lead">Each step names the parts and cables by ID. The same IDs are printed in the tables, written on the cable tags, and searchable in the interactive twin.</p>
{self.safety_box()}
<div class="cols">
 <div style="flex:1">
  <h3>Rules for every step</h3>
  <ol>
   <li><b>Tag both ends of every cable</b> with its ID before you route it. Use the label printer cartridges in the parts list.</li>
   <li><b>ESD wrist strap to the rack earth</b> whenever you touch a Spider box, attenuator, SFP or radio. <b>Torque every SMA to 8 in-lb</b> (0.9 N·m) with the SMA torque wrench. Finger-tight is not enough and more damages the connector.</li>
   <li><b>Rack units count from the bottom</b>: U1 is the lowest unit. A device marked U33-U34 takes two units. The U numbers are printed on the front-left post in the pictures.</li>
  </ol>
 </div>
 <div style="flex:1">
  <ol start="4">
   <li><b>Front</b> is the operator side (Spider-box and attenuator SMA panels). <b>Rear</b> carries the PDU outlets, DC terminals and the node ETH / SFP ports. Every picture has a badge in its top-left corner: <b>FRONT VIEW</b>, <b>REAR VIEW</b> (seen from behind: the rack's left and right swap) or <b>VIEW FROM ABOVE</b>. "Left" and "right" in the tables are always as seen from the FRONT.</li>
   <li><b>Length</b> is the stock length to fit; "route" is the length the twin measured along the cable path. Dress the slack in the cable manager.</li>
   {"<li><b>Terminate unused tile and coupler SMA ports</b> with 50-ohm loads (ANNE-50+).</li>" if loads else ""}
   <li><b>Power stays off</b> until the last step: PDUs unplugged, breakers and fuses OFF.</li>
  </ol>
 </div>
</div>
  <h3>How the pictures work</h3>
  <ul class="small" style="columns:2;column-gap:8mm">
   <li>Each step picture shows the bench as it is after that step. Items fitted in this step are bright with an orange outline and an ID tag; items from earlier steps are pale; later items are not shown.</li>
   <li>Cables are drawn in their class colour along the route the twin planned; DC, GNSS and WG cables are also banded. Small labels give the cable ID.</li>
   <li>Close-ups of a connector panel are straight on with the cables left out: every port carries a tag (port, cable ID, far end) on a leader line.</li>
   <li>Close-ups of the Spider boxes are straight on: module (M1-M4), sector (e.g. HUB.W) and tile (T1-T4) are labelled.</li>
   <li>Every picture carries the TENTATIVE mark: the scene is a planning model, not a construction drawing.</li>
  </ul>
<div class="cols newpage">
 <div style="flex:1.25">
  <h3>Cable colours and ID scheme</h3>
  {legend}
  <p class="small">Cable IDs are a class letter and a number: <b>A</b> = AC, <b>D</b> = DC, <b>P</b> = PoE, <b>F</b> = data, <b>E</b> = management,
   <b>R</b> = IF coax, <b>G</b> = GNSS, <b>W</b> = IP-50 chain, <b>X</b> = operator station. The colours are a print palette (DC, GNSS and WG
   also banded) shared with the twin; the cable schedule files keep the scene colours. Port IDs are <span class="mono">DEVICE.PORT</span>:
   <span class="mono">ATT-01.ch1.in</span> is attenuator ATT-01, channel 1 input; <span class="mono">SB2.M1.T1</span> is Spider box SB2, module M1, tile 1.</p>
  <h3>Confidence of each item</h3>
  <p class="small">{chip('estimate')} {esc(CONF_TEXT['estimate'])}. &nbsp; {chip('likely')} {esc(CONF_TEXT['likely'])}. &nbsp;
   {chip('confirmed')} {esc(CONF_TEXT['confirmed'])}.</p>
 </div>
 <div style="flex:1">

 </div>
</div>"""
        self.section("read", "How to read this guide", body, num="1")

    def rear_sides(self, sb):
        """node -> 'left' / 'right' as seen from the REAR of a Spider box."""
        b = self.b
        d = b.D[sb]
        xs = defaultdict(list)
        for pid in d.get("ports") or []:
            p = b.P.get(pid)
            if p and p["normal"][2] > 0.5:
                head = p["name"].split(".")[0]
                if head in (d.get("nodes") or []):
                    xs[head].append(p["pos"][0])
        mid = d["pos"][0] + d["size"][0] / 2
        return OrderedDict((n, "right" if sum(v) / len(v) < mid else "left") for n, v in xs.items())

    def safety_box(self):
        b = self.b
        srv = [d for d in b.D.values() if d["kind"] in ("rackserver", "server") and d["group"] == "rack"]
        radios = [d for d in b.D.values() if d["kind"] == "ip50"]
        stacked = [d for d in radios if b.below(d)]
        items = []
        if srv:
            lo = min(d["u_top"] - (d.get("u_n") or 1) + 1 for d in srv)
            hi = max(d["u_top"] for d in srv)
            deep = max(round(d["size"][2]) for d in srv)
            items.append(f"<b>Rack tipping.</b> Lock the castors or fit the stabiliser feet before you load the rack, and load from the bottom up. "
                         f"The {len(srv)} servers at U{lo}-U{hi} (up to {deep} mm deep) make it top-heavy: never pull out more than one unit at a time.")
        items.append("<b>Earth first.</b> Bond the rack earth stud to the building earth (6 mm² green/yellow lead, ring lug) before any device goes in; "
                     "clip the ESD wrist strap to the rack earth.")
        lift = []
        if srv:
            lift.append("the rack servers (two people, on their rails)")
        if radios:
            lift.append(f"the IP-50 radios (about 6 kg each, estimate{'; ' + str(len(stacked)) + ' of them stand on top of another radio' if stacked else ''})")
        if lift:
            t = "; ".join(lift)
            items.append("<b>Lifting.</b> " + t[0].upper() + t[1:] + ".")
        dc48 = [d for d in b.D.values() if d["kind"] == "dcdist"]
        dcn = [d for d in b.D.values() if d["kind"] == "fuse"]
        if dc48 or dcn:
            t = "<b>Two DC systems.</b> "
            if dc48:
                t += f"{', '.join(d['id'] for d in dc48)} is +48 V (node adaptors); "
            if dcn:
                t += f"{', '.join(d['id'] for d in dcn)} is -48 V with its positive pole earthed (radios). "
            t += "Both use the same Belden cable: tag -48 V cables blue and +48 V cables red. Breakers OFF and fuses out while you wire."
            items.append(t)
        if any(c["cls"] == "POE" for c in b.C.values()):
            items.append("<b>Passive PoE.</b> The RJ45 of every DC/PoE adaptor (the P-cables) carries 48 V: plug it only into the node's ETH1, "
                         "never into a switch, PC or laptop.")
        items.append("<b>RF.</b> Never power a node or radio while one of its SMA tiles or a waveguide flange is open: close the chain or fit a 50-ohm load first.")
        if any(c["cls"] == "FIB" for c in b.C.values()):
            items.append("<b>Lasers.</b> The SFP+ SR optics are Class 1 lasers: keep the dust caps on until you mate, never look into a fibre end, "
                         "clean every LC end (one-click cleaner) before you plug it.")
        if any(c["cls"] == "GNSS" for c in b.C.values()):
            items.append("<b>GNSS lead.</b> The roof antenna lead enters through a surge arrestor that must be earthed at the building entry (site work).")
        half = (len(items) + 1) // 2
        col = lambda xs: "<ul class=\"small\">" + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"
        return (f'<div class="box safety"><span class="h">⚠ SAFETY: read before you start.</span> The step pages repeat the line that applies there.'
                f'<div class="cols" style="gap:5mm"><div style="flex:1">{col(items[:half])}</div><div style="flex:1">{col(items[half:])}</div></div></div>')

    def step_callouts(self, st):
        """safety / hold / orientation lines for one step (matched on its title)."""
        b = self.b
        t = st["title"].lower()
        out = []

        def warn(txt):
            out.append(f'<div class="callout"><b class="h">⚠ Safety.</b> {txt}</div>')

        def info(txt):
            out.append(f'<div class="callout info"><b class="h">Note.</b> {txt}</div>')

        if t.startswith("assemble the rack"):
            warn("Bond the rack earth stud to the building earth first: 6 mm² green/yellow lead with ring lugs (stud size TBC). Lock the castors or fit "
                 "the stabiliser feet before any unit goes in, and load from the bottom up.")
        elif "pdus" in t:
            warn("Do not plug the PDU cords into the wall until the power-up step.")
        elif "dc drawers" in t:
            warn("Breakers OFF and fuses out. The two DC systems (+48 V and -48 V) stay unwired until the DC cable step.")
        elif t.startswith("programmable attenuators") or "splitter" in t or t.startswith("spider boxes"):
            warn("ESD wrist strap clipped to the rack earth; keep the SMA dust caps on until the IF-link steps.")
        elif t.startswith("servers"):
            srv = [d for d in b.D.values() if d["kind"] in ("rackserver", "server") and d["group"] == "rack"]
            warn(f"Two people per server: fit the rails first, then lift the server onto them. Fit the {len(srv)} servers one at a time; never extend more than one.")
        elif "cable: dc" in t:
            rect2 = [d["id"] for d in b.D.values() if d["kind"] == "rect" and "-48" in (d.get("title") or "") + (d.get("sub") or "")]
            warn(f"Breakers OFF and every fuse out while you wire. Earth the + pole of {', '.join(rect2) or 'the -48 V rectifier'} (positive-earth -48 V) "
                 "to the rack earth bar. Tag every -48 V cable blue and every +48 V cable red at both ends: both systems use the same Belden 8461.")
        elif "cable: poe" in t:
            warn("The P-cables carry 48 V (passive PoE) from the adaptor RJ45: they go into the node's ETH1 only, never into a switch, PC or ETH2.")
        elif "cable: data" in t:
            sw = b.D.get("SW-DATA")
            if sw:
                info(f'SW-DATA has its cages on the FRONT ({esc(sw.get("u"))}), but every data source (node and host SFP+, agent RJ45) is at the rear. '
                     f'Pass the fibres and DACs from the rear up the right-hand side of the frame (right as seen from the FRONT) to {esc(sw.get("u"))}, '
                     f'then forward along the side into the cages. Cage by cage: <a href="#sw-data-cages">picture after the cable list</a>.')
            warn("SFP+ SR optics are Class 1 lasers: dust caps on until you mate, never look into a fibre end or an open cage, clean each LC end before you plug it.")
        elif "cable: gnss" in t:
            out.insert(0, '<div class="hold"><b class="big">HOLD</b> Do this step only after the GNSS input on the Spider-box rear is confirmed '
                          '(open question 1, Spider-box rear panels). Until then leave the GNSS-SPL outputs capped.</div>')
            warn("The antenna lead (G001) enters through the surge arrestor: earth the arrestor at the building entry before you connect the lead to GNSS-SPL.")
        elif "ip-50" in t:
            warn("No radio power until its whole chain is closed: flange screwed (W-flange joints), pad and SMA on, MWATT channel set. "
                 "The radios are about 6 kg: lift, do not slide, a stacked radio.")
        if any(k in t for k in ("cable: poe", "cable: data", "cable: gnss")):
            sides = "; ".join(f"{sb}: " + ", ".join(f"{n} {sd}" for n, sd in self.rear_sides(sb).items())
                              for sb in sorted(d["id"] for d in b.D.values() if d["kind"] == "spider"))
            info(f'Spider-box rear ports by name (W1.ETH1, W2.SFP+, ...): see <a href="#sb-rear">Rear panels</a> in step 7. '
                 f'Seen from the rear the boxes are mirrored; node groups as you face the rear: {esc(sides)}.')
        return "".join(out)

    # ---------------------------------------------------------- 2 overview
    def overview(self):
        b = self.b
        body = self.fig("overview_iso", PAGE_W, 158, caption="<b>Whole bench</b>, iso view from the front right. The rack stands on the left; the bench table is to its right; the lab wall and its sockets are behind the rack.")
        self.section("overview", "Overview", body, num="2")
        # ---- rack elevation + U table
        rows = []
        u = b.R["units"]
        top = [d for d in b.D.values() if d["group"] == "rack" and not d.get("parent") and d.get("u_top")]
        occ = {}
        for d in top:
            for k in range(d["u_top"] - (d.get("u_n") or 1) + 1, d["u_top"] + 1):
                occ[k] = d
        k = u
        while k >= 1:
            d = occ.get(k)
            if d is None:
                j = k
                while j - 1 >= 1 and occ.get(j - 1) is None:
                    j -= 1
                rows.append(([f"U{k}" if j == k else f"U{j}-U{k}", '<span class="muted">empty</span>', ""], ""))
                k = j - 1
                continue
            if d["kind"] == "blank":
                j = k
                ids = [d["id"]]
                while j - 1 >= 1 and occ.get(j - 1) is not None and occ[j - 1]["kind"] == "blank":
                    j -= 1
                    ids.append(occ[j]["id"])
                rows.append(([f"U{k}" if j == k else f"U{j}-U{k}", f'<span class="muted">blanking panel{"s" if len(ids) > 1 else ""}</span>',
                              esc(d.get("sub") or "")], "sub"))
                k = j - 1
                continue
            lo = d["u_top"] - (d.get("u_n") or 1) + 1
            rows.append(([esc(d["u"]), f'<span class="cid">{esc(d["id"])}</span>', esc(d.get("sub") or "")], ""))
            k = lo - 1
        self.table_w = PAGE_W * 0.45
        utab = self.table([("U", ""), ("Device", ""), ("Part", "")], rows, widths=[15, 27, 58], cls="tight")
        r = b.renders["overview_front_rack"]
        asp = r["px"]["w"] / r["px"]["h"]
        img_h = 166
        img_w = min(img_h * asp, PAGE_W * 0.52)
        body = f"""
<div class="cols">
 <div style="flex:0 0 {img_w + 2:.0f}mm">{self.fig('overview_front_rack', img_w, img_h, caption='<b>Front elevation</b> without cables; every item labelled; U numbers on the left post')}</div>
 <div style="flex:1"><p class="small">Read from the top. Every U is listed; consecutive blanking panels share a row.
  Rack: {esc(b.R['part'])}, {b.R['units']}U, rail-to-rail depth {round(b.R['rail_depth'])} mm; U1 starts {round(b.R['u1_y'])} mm above the floor.</p>{utab}</div>
</div>"""
        self.section("rack-front", "Rack front elevation", body, level=2, toc_text="Rack front elevation (U by U)")
        self.section("rack-front-big", "Rack front elevation, enlarged", self.elev_pair("front"), level=2)
        # ---- rear elevation
        rr = []
        for d in sorted(top, key=lambda d: -d["u_top"]):
            if d["kind"] == "blank":
                continue
            rear = [self.b.P[p] for p in d.get("ports") or [] if p in b.P and b.P[p]["normal"][2] > 0.5]
            for kid in b.children.get(d["id"]) or []:
                rear += [b.P[p] for p in b.D[kid].get("ports") or [] if p in b.P and b.P[p]["normal"][2] > 0.5]
            if not rear:
                continue
            used = sum(1 for p in rear if p["id"] in b.portcable)
            types = Counter(p["type"] for p in rear)
            rr.append([esc(d["u"]), f'<span class="cid">{esc(d["id"])}</span>', esc(", ".join(f"{n} {ctype(t)}" for t, n in types.most_common())), f"{used} / {len(rear)}"])
        rtab = self.table([("U", ""), ("Device", ""), ("Rear connectors", ""), ("Used", "n")], rr, widths=[15, 27, 42, 13], cls="tight")
        r = b.renders["overview_rear_rack"]
        asp = r["px"]["w"] / r["px"]["h"]
        img_w = min(img_h * asp, PAGE_W * 0.52)
        body = f"""
<div class="cols">
 <div style="flex:0 0 {img_w + 2:.0f}mm">{self.fig('overview_rear_rack', img_w, img_h, caption='<b>Rear elevation</b> without cables, seen from behind the rack (left and right swap); the lab wall left out')}</div>
 <div style="flex:1"><p class="small">Most cabling happens at the rear: PDU outlets, DC terminals, the node power (ETH1),
  data (SFP+) and GNSS ports. Connectors that face the rear, per rack item (counting the parts on its shelf):</p>{rtab}</div>
</div>"""
        self.section("rack-rear", "Rack rear", body, level=2, toc_text="Rack rear (connectors)")
        self.section("rack-rear-big", "Rack rear elevation, enlarged", self.elev_pair("rear"), level=2)
        # ---- bench table plan
        body = self.table_tags_fig("overview_table", PAGE_W, 92, caption="<b>Bench table</b> seen from the operator side, from above (about 50°). "
                                   "Every item is tagged; a device standing on another one is tagged on its front face (dashed outline in the plan).")
        self.section("table-plan", "Bench table layout", self.table_plan() + f'<div class="keep"><h2 class="sec" id="table-view" style="margin-top:4mm">Bench table, 3D view</h2>{body}</div>',
                     level=2, toc_text="Bench table layout (from above)")
        self.toc.append((2, "table-view", "Bench table, 3D view", "Bench table, 3D view"))

    def elev_pair(self, side):
        up, lo = f"overview_{side}_rack_upper", f"overview_{side}_rack_lower"
        if up not in self.b.renders or lo not in self.b.renders:
            return ""
        w2 = PAGE_W / 2 - 2.5
        what = ("seen from the front; U numbers on the left post" if side == "front"
                else "seen from behind the rack: its left and right swap; the lab wall left out")
        return (f'<p class="small">The same elevation at twice the scale, {what}. No cables; every unit labelled beside the rack.</p>'
                f'<div class="pair">{self.fig(up, w2, 162, caption="<b>Upper half</b>")}{self.fig(lo, w2, 162, caption="<b>Lower half</b>")}</div>')

    def table_tags_fig(self, name, max_w, max_h, caption="", ids=None):
        """bench-table picture with a leader-line tag per device."""
        b = self.b
        r = b.renders[name]
        tags = []
        for key, an in r["anchors"].items():
            d = b.D.get(key)
            if not d or not an["in"] or (ids is not None and key not in ids):
                continue
            under = b.below(d)
            lines = [key] + ([f"on {under['id']}"] if under else [])
            if b.on_top(d):
                lines.append("under " + ", ".join(o["id"] for o in b.on_top(d)))
            col = "#1f2933" if d["id"] not in (b.meta.get("switched") or []) else "#b42318"
            tags.append({"x": an["x"], "y": an["y"], "lines": lines, "fill": col, "side": "top" if an["y"] < 0.62 else "bottom"})
        return self.fig_tags(name, max_w, max_h, tags, caption=caption)

    def table_plan(self):
        """to-scale SVG plan of the bench table with numbered devices + a position table."""
        b = self.b
        T = b.T
        x0, x1 = -120, T["x1"] + 160
        z0, z1 = min(T["z0"], 0) - 230, max(T["z1"], b.R["rail_depth"]) + 70
        W = x1 - x0
        H = z1 - z0
        sx = lambda x: x - x0
        sy = lambda z: z1 - z  # rear (wall) at the top, operator side at the bottom
        FS = 50  # 50 units = 3.1 mm = 8.8 pt on the page
        g = []
        fw = b.R.get("frame_w", 600)
        rx0 = (b.R["panel_w"] - fw) / 2
        g.append(f'<rect x="{sx(rx0):.0f}" y="{sy(b.R["rail_depth"] + 20):.0f}" width="{fw:.0f}" height="{b.R["rail_depth"] + 40:.0f}" fill="#e5e7eb" stroke="#4b5563" stroke-width="6"/>')
        g.append(f'<text x="{sx(rx0 + fw / 2):.0f}" y="{sy(b.R["rail_depth"] / 2) + 30:.0f}" font-size="90" font-weight="800" text-anchor="middle" fill="#374151">RACK</text>')
        g.append(f'<text x="{sx(rx0 + fw / 2):.0f}" y="{sy(-20) + 62:.0f}" font-size="{FS}" text-anchor="middle" fill="#6b7280">front ▼</text>')
        g.append(f'<rect x="{sx(T["x0"]):.0f}" y="{sy(T["z1"]):.0f}" width="{T["x1"] - T["x0"]:.0f}" height="{T["z1"] - T["z0"]:.0f}" fill="#f6ead2" stroke="#9a8462" stroke-width="5"/>')
        devs = sorted([d for d in b.D.values() if d["group"] == "table"], key=lambda d: (d.get("parent") is not None, d["pos"][0], d["pos"][2], d["pos"][1]))
        num = {d["id"]: i + 1 for i, d in enumerate(devs)}
        badges = []
        for d in sorted(devs, key=lambda d: (d["pos"][1], d.get("parent") is not None)):
            x, z = d["pos"][0], d["pos"][2]
            w, dp = d["size"][0], d["size"][2]
            up = b.below(d) is not None
            off = 28 if up else 0
            col = d.get("color") or "#cbd5e1"
            if d.get("parent"):
                g.append(f'<rect x="{sx(x):.0f}" y="{sy(z + dp):.0f}" width="{w:.0f}" height="{dp:.0f}" fill="#ffffff" stroke="{DISPLAY_COLORS['WG']}" stroke-width="5"/>')
            else:
                dash = ' stroke-dasharray="18 10"' if up else ""
                g.append(f'<rect x="{sx(x) + off:.0f}" y="{sy(z + dp) - off:.0f}" width="{w:.0f}" height="{dp:.0f}" fill="{esc(col)}" fill-opacity="{0.35 if up else 0.6}" stroke="#1f2933" stroke-width="4" rx="6"{dash}/>')
            tw = len(d["id"]) * FS * 0.7
            cy = sy(z + dp / 2) - off
            if not d.get("parent") and not up and tw + 120 <= w:
                # badge, then the ID to its right
                x_b = sx(x) + (w - tw - 100) / 2 + 42
                g.append(f'<text x="{x_b + 56:.0f}" y="{cy + 18:.0f}" font-size="{FS}" font-weight="700" fill="#111827">{esc(d["id"])}</text>')
                badges.append((x_b, cy, num[d["id"]], False))
            else:
                cx = sx(x + w / 2) + off + (w * 0.25 if up else 0)
                badges.append((cx, cy, num[d["id"]], bool(d.get("parent"))))
        for cx, cy, k, small in badges:
            r = 34 if small else 42
            g.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="{DISPLAY_COLORS['WG'] if small else "#1f2933"}" stroke="#fff" stroke-width="6"/>')
            g.append(f'<text x="{cx:.0f}" y="{cy + 17:.0f}" font-size="{46 if small else 50}" font-weight="800" text-anchor="middle" fill="#fff">{k}</text>')
        # dimensions
        yb = sy(T["z0"]) + 105
        g.append(f'<line x1="{sx(T["x0"]):.0f}" y1="{yb:.0f}" x2="{sx(T["x1"]):.0f}" y2="{yb:.0f}" stroke="#374151" stroke-width="4" marker-start="url(#ar)" marker-end="url(#ar)"/>')
        g.append(f'<text x="{sx((T["x0"] + T["x1"]) / 2):.0f}" y="{yb - 18:.0f}" font-size="{FS}" text-anchor="middle" fill="#374151">{round(T["x1"] - T["x0"])} mm (table length, assumed)</text>')
        g.append(f'<line x1="{sx(0):.0f}" y1="{yb:.0f}" x2="{sx(T["x0"]):.0f}" y2="{yb:.0f}" stroke="#6b7280" stroke-width="3" stroke-dasharray="14 10" marker-end="url(#ar)"/>')
        g.append(f'<text x="{sx(T["x0"] / 2):.0f}" y="{yb - 18:.0f}" font-size="{FS}" text-anchor="middle" fill="#6b7280">{round(T["x0"])} mm</text>')
        xr = sx(T["x1"]) + 45
        g.append(f'<line x1="{xr:.0f}" y1="{sy(T["z1"]):.0f}" x2="{xr:.0f}" y2="{sy(T["z0"]):.0f}" stroke="#374151" stroke-width="4" marker-start="url(#ar)" marker-end="url(#ar)"/>')
        g.append(f'<text transform="translate({xr + 60:.0f},{(sy(T["z1"]) + sy(T["z0"])) / 2:.0f}) rotate(-90)" font-size="{FS}" text-anchor="middle" fill="#374151">{round(T["z1"] - T["z0"])} mm</text>')
        g.append(f'<text x="{sx(T["x0"]):.0f}" y="{sy(T["z0"]) + 190:.0f}" font-size="{FS}" fill="#6b7280">▲ operator side: table front edge {round(-T["z0"])} mm in front of the rack front rail</text>')
        g.append(f'<text x="{sx(T["x0"]):.0f}" y="{sy(T["z1"]) - 22:.0f}" font-size="{FS}" fill="#6b7280">rear edge (wall side)</text>')
        svg = (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" width="{PAGE_W}mm" height="{PAGE_W * H / W:.1f}mm" xmlns="http://www.w3.org/2000/svg">'
               '<defs><marker id="ar" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
               '<path d="M0,0 L10,5 L0,10 z" fill="#374151"/></marker></defs>' + "".join(g) + "</svg>")
        rows = []
        for d in devs:
            w, h, dp = (round(v) for v in d["size"])
            rows.append([f"<b>{num[d['id']]}</b>", f'<span class="cid">{esc(d["id"])}</span>', esc(d.get("sub") or b.what(d)),
                         esc(b.where(d).replace("table: ", "")), f"{w} × {dp} × {h}", chip(d.get("conf"))])
        self.table_w = PAGE_W
        tab = self.table([("#", "n"), ("ID", ""), ("What it is", ""), ("Position (from the table's rack-side end / front edge)", ""), ("W × D × H mm", ""), ("Conf.", "")],
                         rows, widths=[3.5, 16, 31, 32, 10, 7.5], cls="tight")
        return (f'<p class="small">Plan from above, to scale. Rear (wall side) at the top, operator side at the bottom. Numbers match the table. '
                f'Dashed boxes stand on top of another device; pink squares are the waveguide transitions on the radio flanges. '
                f'Assumed: {esc("; ".join(T.get("assumed") or []))}.</p>{svg}<h3>Positions</h3>{tab}')

    # ---------------------------------------------------------- 3 BOM
    def parts(self):
        if not hasattr(self, "_parts"):
            with open(OUT / f"{self.b.poc}_parts_list.csv", newline="") as f:
                self._parts = list(csv.DictReader(f))
        return self._parts

    def bom(self):
        rows = []
        for r in self.parts():
            st = r.get("status") or ""
            cls = "ok" if st.startswith("confirmed") else "likely" if st.startswith("likely") else "estimate"
            rows.append([esc(r.get("line")), esc(r.get("group")), esc(r.get("item")), f'<span class="mono">{esc(r.get("part number"))}</span>',
                         esc(r.get("manufacturer")), esc(r.get("qty")), esc(r.get("use")), f'<span class="chip {cls}" style="white-space:normal">{esc(st)}</span>'])
        for i, (grp, item, pn, qty, use) in enumerate(self.guide_parts(), 1):
            rows.append(([f"G{i}", esc(grp), esc(item), f'<span class="mono">{esc(pn)}</span>', "-", esc(qty), esc(use),
                          '<span class="chip estimate" style="white-space:normal">added by the guide; to confirm</span>'], "sub"))
        tab = self.table([("#", "n"), ("Group", ""), ("Item", ""), ("Part number", ""), ("Maker", ""), ("Qty", ""), ("Use", ""), ("Status", "")],
                         rows, widths=[3.5, 7, 26, 17, 10, 8, 18, 10.5], cls="tight")
        body = (f'<p class="lead">From <span class="mono">{esc(self.b.poc)}_parts_list.csv</span>. Quantities include spares where noted. '
                f'"Status" is how sure the part is; check every "to confirm" line before ordering. Lines G1… (shaded) are installation '
                f'hardware this guide calls for that the parts list does not carry yet.</p>{tab}')
        self.section("bom", "Bill of materials", body, num="3")

    def guide_parts(self):
        """installation hardware the steps call for (not in the parts CSV)."""
        b = self.b
        out = [("Safety", "Rack earth bond: 6 mm² green/yellow lead with ring lugs", "TBC", "1 set", "rack earth stud to the building earth (step 1)"),
               ("Safety", "ESD wrist strap with coiled lead and clip", "TBC", "2", "clipped to the rack earth")]
        dcn = [d for d in b.D.values() if d["kind"] == "fuse"]
        bus = [c for c in b.C.values() if c["cls"] == "DC" and "bus" in (c.get("pn") or "") + (c.get("part") or "")]
        if any(c["cls"] == "DC" for c in b.C.values()):
            out.append(("Power", "Cable tags, red (+48 V) and blue (-48 V)", "TBC", "1 roll each", "tag both ends of every DC cable (step "
                        f"{self.step_of('cable: dc')})"))
        if bus:
            out.append(("Power", "Ring lugs for 2 × 10 AWG bus links, hole to match the stud", "stud size TBC (measure)",
                        f"{len(bus) * 4} + 4 spare", f"both ends of {', '.join(c['id'] for c in bus)}"))
        if dcn:
            out.append(("Power", f"A-B bus strap for {dcn[0]['id']} (both poles), or 10 AWG links with ring lugs", "TBC (vendor strap kit)",
                        f"1 per panel ({len(dcn)})", "makes bus B (F6-F10) live"))
        fl = [c for c in b.C.values() if c["cls"] == "WG" and all(b.P.get(x, {}).get("type") == "WG" for x in (c["a"], c["b"]))]
        if fl:
            out.append(("IP-50 chain", "WR-62 flange screws #4-40 with lock washers (4 per joint)", "TBC with the transition datasheet",
                        f"{len(fl) * 4} + 8 spare", f"{len(fl)} radio-to-transition flange joints; about 0.6 N·m (5 in-lb, TBC)"))
            out.append(("IP-50 chain", "WR-62 flange gasket (only for a gasketed / choke radio flange)", "TBC", f"{len(fl)} (if needed)", "open question 5"))
        if any(c["cls"] == "FIB" for c in b.C.values()):
            out.append(("Data", "LC one-click fibre cleaner and spare LC dust caps", "TBC", "1 + 20 caps", "clean before every mating (step "
                        f"{self.step_of('cable: data')})"))
        return out

    # ---------------------------------------------------------- 4 steps
    def steps(self):
        b = self.b
        intro = ["<p class=\"lead\">Follow the steps in order. Each step starts on a new page: the picture shows the bench after the step, "
                 "the tables list what to fit or connect.</p>"]
        rows = []
        for st in b.steps:
            kind = self.step_kind(st)
            own = [c for c in st["cables"] if b.owner[c] == st["n"]]
            what = f"{len(st['devices'])} items" if st["devices"] else f"{len(own)} cables" if st["cables"] else "-"
            rows.append([f'<a href="#step-{st["n"]}"><b>{st["n"]}</b></a>', esc(st["title"]), esc(kind), esc(st["view"]), what])
        intro.append(self.table([("Step", "n"), ("Title", ""), ("Kind", ""), ("View", ""), ("Items", "")], rows, widths=[6, 44, 14, 10, 12], cls="tight"))
        self.section("steps", "Installation steps", "".join(intro), num="4")
        for st in b.steps:
            self.step(st)

    def step_kind(self, st):
        if re.match(r"^(IF link|Coupling arm)\s", st["title"]):
            return "IF link" if st["title"].startswith("IF") else "coupling arm"
        if st["devices"]:
            return "devices"
        if st["cables"]:
            return "IP-50 chains" if all(self.b.C[c]["cls"] == "WG" for c in st["cables"]) else "cables"
        return "checks" if st["n"] == len(self.b.steps) else "rack"

    def step(self, st):
        b = self.b
        n = st["n"]
        kind = self.step_kind(st)
        own = [c for c in st["cables"] if b.owner[c] == n]
        shared = [c for c in st["cables"] if b.owner[c] != n]
        view_txt = {"front": "front view (rack front faces you)", "rear": "rear view (from behind the rack)", "table": "bench-table view",
                    "iso": "iso view", "top": "view from above"}.get(st["view"], st["view"] + " view")
        chips = f'<span class="chip kind">{esc(kind)}</span>'
        classes = Counter(b.C[c]["cls"] for c in st["cables"])
        for cls in classes:
            col = b.legend.get(cls, {}).get("color", "#999")
            chips += f' <span class="chip" style="background:{esc(col)};color:{"#111827" if _lum(col) > 0.45 else "#fff"}">{esc(cls)}</span>'
        head = (f'<div class="stephead"><div class="stepno">{n}</div><div><h2 id="step-{n}">{esc(st["title"])}</h2>'
                f'<div class="sub">Step {n} of {len(b.steps)} · {esc(view_txt)}</div></div><div class="chips">{chips}</div></div>')
        extra = self.step_extra_text(st, own, shared)
        text = f'<div class="steptext">{esc(st["text"])}{extra}</div>'
        call = self.step_callouts(st)
        self.toc.append((2, f"step-{n}", f"Step {n} · {st['title']}", st["title"]))
        parts = [head, text, call]
        is_link = kind in ("IF link", "coupling arm")
        # callout / hold boxes: about 4.6 mm per line of ~190 characters, plus padding
        call_h = sum(4.0 + 4.6 * (1 + len(re.sub(r"<[^>]+>", "", m)) // 190)
                     for m in re.findall(r'<div class="(?:callout[^"]*|hold)">(.*?)</div>', call))
        # page 1: heading (15 mm) + text box + callouts + picture (+ caption 7 mm) [+ the main block if it fits]
        t1 = 1 + len(st["text"]) // 140
        t2 = 1 + len(html.unescape(re.sub(r"<[^>]+>", "", extra))) // 155
        avail = PAGE_H - 15 - (6.5 + 5.2 * t1 + 4.7 * t2) - 8 - 4 - call_h
        if st["view"] == "rear" and f"step_{n:02d}" in b.renders and b.renders[f"step_{n:02d}"].get("clip"):
            self.rear_step(st, own, parts, avail)
            return
        main, est, after = "", 0.0, ""
        self.table_w = PAGE_W
        if kind == "devices":
            tab = self.device_table(st["devices"])
            main, est = f'<h3>Fit these {len(st["devices"])} items</h3>{tab}', self.last_est + 9
            sbs = [d for d in st["devices"] if b.D[d]["kind"] == "spider"]
            if sbs:
                after = self.spider_panels(sbs) + self.spider_rear_panels(sbs)
        elif kind == "rack":
            R = b.R
            rows = [["Rack", esc(R["part"])], ["Units", f'{R["units"]}U, {R["used"]}U used; U1 is the lowest unit']]
            rows += [["Rail-to-rail depth", f'{round(R["rail_depth"])} mm (deepest device {max(round(d["size"][2]) for d in b.D.values() if d["group"] == "rack" and not d.get("parent"))} mm)'],
                     ["Frame width", f'{round(R.get("frame_w", 600))} mm; 19-inch panel {R["panel_w"]} mm'],
                     ["U1 height", f'{round(R["u1_y"])} mm above the floor (bottom of U1)']]
            rows += [["Assumed", esc(a)] for a in (b.scene.get("assumptions") or [])[:2]]
            tab = self.table([("Item", ""), ("Value", "")], rows, widths=[22, 78], cls="tight")
            main, est = f'<h3>Rack facts</h3>{tab}', self.last_est + 9
        elif is_link:
            after = self.link_block(st, own, shared)
        elif kind == "IP-50 chains":
            after = self.ip50_block(st, own)
        elif st["cables"]:
            tab = self.cable_table(own)
            main, est = f'<h3>Connect these {len(own)} cables</h3>{tab}', self.last_est + 9
        elif kind == "checks":
            main, est = '<p>The install checks, the power-up order and the event procedure are in <a href="#checks">section 6</a>.</p>', 8
        # a short table shares the page with a smaller picture (no near-empty pages)
        if est and est + 55 <= avail:
            # a short table shares the page with a smaller picture (no near-empty pages)
            fig_h, same = min(145, avail - est - 4), True
        else:
            fig_h, same = min(145, avail), False
        cap = f"<b>After step {n}</b>: items of this step highlighted (orange outline); earlier items pale."
        name = f"step_{n:02d}"
        if st["view"] == "table" and not (st["cables"] and all(b.C[c]["cls"] == "AUX" for c in st["cables"])):
            ids = st["devices"] or [d for d in b.D if b.D[d]["kind"] in ("ip50", "mwatt", "psu")]
            parts.append(self.table_tags_fig(name, PAGE_W, fig_h, caption=cap + " Tags name every item of the step.", ids=set(ids)))
        else:
            parts.append(self.fig(name, PAGE_W, fig_h, caption=cap))
        if main:
            parts.append(main if same or main.startswith('<div class="newpage') else f'<div class="newpage">{main}</div>')
        parts.append(after)
        self.sections.append(f'<section class="pg step">{"".join(parts)}</section>')

    def rear_step(self, st, own, parts, avail):
        """a step seen from the rear: straight-on band(s) of the rack, then the
        whole bench (if cables run to the table), the tables and the close-ups."""
        b = self.b
        n = st["n"]
        up, low, bench = f"step_{n:02d}", f"step_{n:02d}_lower", f"step_{n:02d}_bench"
        cap_b = f"<b>After step {n}</b>, rack rear straight on: items and cables of this step highlighted; every U labelled beside the rack."
        h = min(150, avail)
        if low in b.renders:
            w2 = PAGE_W / 2 - 2.5
            parts.append(f'<div class="pair">{self.fig(up, w2, h, caption="<b>Upper part</b> of the rack")}'
                         f'{self.fig(low, w2, h, caption="<b>Lower part</b> of the rack")}</div><p class="tcap" style="text-align:center">{cap_b}</p>')
        else:
            r = b.renders[up]
            asp = r["px"]["w"] / r["px"]["h"]
            if h * asp < PAGE_W * 0.62 and not st["devices"]:
                # a tall, narrow band: where the cable ends land (and a short cable list) beside it
                self.table_w = PAGE_W - h * asp - 8
                side = self.class_notes(st)
                e1 = self.last_est + 16
                done = False
                if bench not in b.renders:
                    ct = self.cable_table(own)
                    e2 = self.last_est + 9
                    if e1 + e2 <= h + 6:
                        side += f'<h3>Connect these {len(own)} cables</h3>' + ct
                        done = True
                    elif e2 <= h + 6:
                        # all ends are in the rack: the cable list says where they land
                        side = f'<h3>Connect these {len(own)} cables</h3>' + ct
                        done = True
                    else:
                        self.cable_rows.subtract(own)  # listed again below
                parts.append(f'<div class="cols"><div style="flex:0 0 {h * asp + 2:.0f}mm">{self.fig(up, h * asp, h, caption=cap_b)}</div>'
                             f'<div style="flex:1">{side}</div></div>')
                self.table_w = PAGE_W
                if bench in b.renders:
                    parts.append(f'<div class="newpage">{self.fig(bench, PAGE_W, 105, caption="<b>Whole bench</b> from behind, table side: where the cables of this step run")}'
                                 f'<h3>Connect these {len(own)} cables</h3>{self.cable_table(own)}</div>')
                elif not done:
                    parts.append(f'<div class="newpage"><h3>Connect these {len(own)} cables</h3>{self.cable_table(own)}</div>')
                parts.append(self.rear_step_extras(st))
                self.sections.append(f'<section class="pg step">{"".join(parts)}</section>')
                return
            if h * asp < PAGE_W * 0.62 and st["devices"]:
                # a tall, narrow band: the device table goes beside it
                self.table_w = PAGE_W - h * asp - 8
                tab = self.device_table(st["devices"])
                parts.append(f'<div class="cols"><div style="flex:0 0 {h * asp + 2:.0f}mm">{self.fig(up, h * asp, h, caption=cap_b)}</div>'
                             f'<div style="flex:1"><h3>Fit these {len(st["devices"])} items</h3>{tab}</div></div>')
                self.sections.append(f'<section class="pg step">{"".join(parts)}</section>')
                return
            if st["devices"]:
                self.table_w = PAGE_W
                self.device_table(st["devices"])
                if self.last_est + 9 + 55 <= avail:
                    h = min(h, avail - self.last_est - 13)
            parts.append(self.fig(up, PAGE_W, h, caption=cap_b))
        self.table_w = PAGE_W
        if st["devices"]:
            parts.append(f'<div class="keep"><h3>Fit these {len(st["devices"])} items</h3>{self.device_table(st["devices"])}</div>')
        if own:
            if bench in b.renders:
                self.table_w = PAGE_W - 150
                parts.append(f'<div class="newpage cols"><div style="flex:0 0 148mm">{self.fig(bench, 146, 120, caption="<b>Whole bench</b> from behind, table side: where the cables of this step run")}</div>'
                             f'<div style="flex:1">{self.class_notes(st)}</div></div>')
            self.table_w = PAGE_W
            parts.append(f'<div class="newpage"><h3>Connect these {len(own)} cables</h3>{self.cable_table(own)}</div>')
        parts.append(self.rear_step_extras(st))
        self.sections.append(f'<section class="pg step">{"".join(parts)}</section>')

    def step_extra_text(self, st, own, shared):
        b = self.b
        out = []
        if st["cables"]:
            out.append(f"{len(own)} cables in this step. Tag both ends first.")
            if any(b.C[c]["cls"] in ("RF", "GNSS", "WG") for c in own):
                out.append("Torque the SMA connectors to 8 in-lb.")
        if shared:
            steps = sorted({b.owner[c] for c in shared})
            out.append(f"Also part of this link, already fitted in step {', '.join(map(str, steps))}: {', '.join(shared)}.")
        if st["devices"]:
            rack = [d for d in st["devices"] if b.D[d]["group"] == "rack" and not b.D[d].get("parent")]
            if rack:
                out.append("Rack positions: " + ", ".join(f"{d} {b.D[d].get('u')}" for d in rack) + ".")
        return f'<span class="why">{esc(" ".join(out))}</span>' if out else ""

    def class_notes(self, st):
        b = self.b
        own = [c for c in st["cables"] if b.owner[c] == st["n"]]
        by_dev = Counter()
        for c in own:
            for pid in (b.C[c]["a"], b.C[c]["b"]):
                d = b.dev_of(pid)
                if d and d["group"] == "rack":
                    top = b.D[d["parent"]] if d.get("parent") else d
                    by_dev[top["id"]] += 1
        rows = [[f'<span class="cid">{esc(k)}</span>', esc(b.D[k].get("u") or ""), str(v)] for k, v in sorted(by_dev.items(), key=lambda kv: -(b.D[kv[0]].get("u_top") or 0))]
        tab = self.table([("Rack item", ""), ("U", ""), ("Cable ends", "n")], rows, widths=[45, 25, 20])
        return (f'<h3>Where the cable ends land in the rack</h3><p class="small">Cable ends of this step per rack item, top to bottom. '
                f'Ends on the bench table are in the table that follows.</p>{tab}')

    def spider_panels(self, sbs):
        """straight-on close-ups of Spider-box fronts with T1-T4 under every tile."""
        b = self.b
        figs = []
        for did in sbs:
            name = f"sb_{did}"
            if name not in b.renders:
                continue
            r = b.renders[name]
            ov = []
            for key, an in r["anchors"].items():
                m = re.match(r"^SB\d+\.M\d+\.T(\d+)#below$", key)
                if m and an["in"]:
                    ov.append({"x": an["x"], "y": an["y"], "text": f"T{m.group(1)}", "cls": "t"})
            d = b.D[did]
            mods = ", ".join(f'{m["m"]} = {m.get("sector")}' for m in d.get("modules") or [])
            figs.append(self.fig(name, PAGE_W, 60, overlays=ov, caption=f"<b>{esc(did)}</b> ({esc(d.get('u'))}, {esc(d.get('sub'))}): {esc(mods)}"))
        return (f'<div class="newpage"><h3>Front panels: module, sector and tile</h3><p class="small">Each module (M1-M4) carries one sector; '
                f'its four tiles T1-T4 run left to right. IF cables land on these tiles in the IF-link steps.</p>{"".join(figs)}</div>')

    # ---------------------------------------------------------- port tags
    def port_info(self, pid):
        b = self.b
        c = b.portcable.get(pid)
        if not c:
            return None, None, None
        o = b.other_end(c, pid)
        return c, o, b.dev_of(o)

    def std_lines(self, pid, short=None, far="port"):
        """tag text for a port: 'NAME CABLE' / far end; grey 'spare' if unused."""
        b = self.b
        c, o, od = self.port_info(pid)
        nm = short or b.P[pid]["name"]
        if not c:
            return [nm, "spare"], "#e5e7eb"
        far_txt = (od["id"] if od else o) if far == "device" else o
        return [f"{nm} {c['id']}", far_txt], c["color"]

    def panel_tags(self, name, lines_fn, side_fn=None):
        r = self.b.renders[name]
        tags = []
        for pid in r.get("ports") or []:
            an = r["anchors"].get(pid)
            if not an or not an["in"]:
                continue
            lines, fill = lines_fn(pid)
            tags.append({"pid": pid, "x": an["x"], "y": an["y"], "lines": lines, "fill": fill})
        if side_fn:
            for t in tags:
                t["side"] = side_fn(t["pid"])
        else:
            # two rows of ports: upper row tagged above, lower row below;
            # one row: alternate above / below along the panel
            ys = sorted({round(t["y"], 2) for t in tags})
            if len(ys) >= 2 and ys[-1] - ys[0] > 0.08:
                mid = (ys[0] + ys[-1]) / 2
                for t in tags:
                    t["side"] = "top" if t["y"] < mid else "bottom"
            else:
                for i, t in enumerate(sorted(tags, key=lambda t: t["x"])):
                    t["side"] = "top" if i % 2 == 0 else "bottom"
        return tags

    def spider_rear_panels(self, sbs):
        """straight-on close-ups of the Spider-box REAR panels, every port tagged
        with its full name and cable, node groups bracketed."""
        b = self.b
        figs = []
        for did in sbs:
            name = f"sb_rear_{did}"
            if name not in b.renders:
                continue
            r = b.renders[name]
            d = b.D[did]
            nodes = d.get("nodes") or []

            def lines(pid):
                nm = b.P[pid]["name"]
                c, o, od = self.port_info(pid)
                head = nm.split(".")[0]
                if head not in nodes:
                    return [nm, "box (TBC)"], "#6b7280"
                return [nm, c["id"] if c else "unused"], (c["color"] if c else "#e5e7eb")
            tags = self.panel_tags(name, lines, side_fn=lambda pid: "top")
            br = []
            for node in nodes:
                xs = [t["x"] for t in tags if b.P[t["pid"]]["name"].split(".")[0] == node]
                if xs:
                    br.append({"x0": min(xs) - 0.012, "x1": max(xs) + 0.012, "text": f"node {node}"})
            sides = self.rear_sides(did)
            note = ", ".join(f"{n} on the {sd.upper()}" for n, sd in sides.items())
            figs.append(self.fig_tags(name, PAGE_W, 52, tags, brackets=br,
                                      caption=f"<b>{esc(did)}</b> ({esc(d.get('u'))}) seen from the REAR: {esc(note)} (mirrored against the front). "
                                              f"Grey tags: box-level ports (controller LAN, DC terminal block), use TBC."))
        if not figs:
            return ""
        return (f'<div class="newpage"><h3 id="sb-rear">Rear panels: node ports by name</h3><p class="small">The node power (ETH1, from the PoE '
                f'step), data (SFP+) and GNSS cables land on the rear of the boxes. Straight-on pictures from the rear, cables left out; every port '
                f'carries its full name and its cable. Seen from the rear, left and right swap: the node that is on the left at the front is on '
                f'the right here. The rear layout comes from one box seen from inside (open question 1).</p>{"".join(figs)}</div>')

    def rear_step_extras(self, st):
        """close-ups that belong to one rear cable step (matched on its title)."""
        b = self.b
        t = st["title"].lower()
        out = []
        if "cable: ac" in t:
            out.append('<div class="newpage"><h3 id="pdu-outlets">PDU outlets, one by one</h3><p class="small">Each PDU straight on from the rear, '
                       'cables left out. Tag = outlet and cable ID, then the device it feeds. '
                       + ('Outlets marked <span class="chip off">OFF at t0</span> are switched off by the bench controller at the event. ' if b.meta.get("switched") else '')
                       + 'The same map is kept in section 5.</p>')
            for d in [d for d in b.D.values() if d["kind"] in ("pdu", "strip")]:
                out.append(self.pdu_block(d, with_fig=True))
            out.append("</div>")
        elif "cable: dc" in t:
            out.append(self.dc_closeups())
        elif "cable: data" in t:
            out.append(self.switch_closeup("SW-DATA", in_step=True))
        elif "cable: management" in t:
            out.append(self.switch_closeup("SW-OOB", in_step=True))
        elif "cable: gnss" in t:
            g = [d for d in b.D.values() if d["kind"] == "gnss"]
            if g and f"panel_{g[0]['id']}" in b.renders:
                name = f"panel_{g[0]['id']}"
                tags = self.panel_tags(name, lambda pid: self.std_lines(pid, short=b.P[pid]["name"].upper()))
                ant = [d for d in b.D.values() if d["kind"] == "antenna"]
                out.append(f'<div class="keep"><h3>{esc(g[0]["id"])} rear, port by port</h3>'
                           + self.fig_tags(name, PAGE_W, 62, tags, caption=f"<b>{esc(g[0]['id'])}</b> ({esc(g[0].get('u'))}) seen from the rear, cables left out")
                           + (f'<p class="small"><b>Antenna.</b> {esc(ant[0]["id"])} ({esc(ant[0].get("sub") or "")}): roof or window mount with a clear view of the sky '
                              f'(site work); the lead comes down to the rack through the surge arrestor, whose earth lug goes to the building earth bar with a '
                              f'short, straight lead (size per the arrestor datasheet, TBC).</p>' if ant else "")
                           + "</div>")
        return "".join(out)

    def switch_closeup(self, dev, in_step=False):
        """straight-on picture of a switch's port face, every cage tagged with its cable."""
        b = self.b
        d = b.D.get(dev)
        name = f"panel_{dev}"
        if not d or name not in b.renders:
            return ""
        face = b.renders[name].get("face") or "front"
        tags = self.panel_tags(name, lambda pid: ([b.P[pid]["name"], b.portcable[pid]["id"]], b.portcable[pid]["color"]) if pid in b.portcable
                               else ([b.P[pid]["name"], "spare"], "#e5e7eb"))
        rd = round(b.R["rail_depth"])
        zf = round(d["pos"][2] + (d["size"][2] if face == "rear" else 0))
        if face == "front":
            ears = (f"{dev}: ears on the FRONT posts (the usual way), port cages at the front rail plane, {d.get('u')}. "
                    f"Odd ports in the top row, even ports below; cage numbers as printed on the switch (TBC: open question 7).")
        else:
            ears = (f"{dev}: ears on the FRONT posts, body reaching {zf} mm back; its ports face the REAR, {zf} mm behind the front rail "
                    f"({rd - zf} mm in front of the rear posts): reach in from the rear. Alternative if the lab prefers: ears on the REAR posts, ports "
                    f"flush with the rear (TBC: open question 7). Odd ports top row, even below; seen from the rear port 1 is on the RIGHT.")
        cap = f"<b>{esc(dev)}</b> {esc(face)} panel straight on ({'seen from the rear' if face == 'rear' else 'seen from the front'}), cables left out; tag = port / cable"
        route = ""
        hid = f' id="{dev.lower()}-cages"' if in_step else ""
        return (f'<div class="keep">{"<h3" + hid + ">" + esc(dev) + " ports as they sit on the switch</h3>" if in_step else ""}{route}'
                f'{self.fig_tags(name, PAGE_W, 70, tags, caption=cap)}<p class="small"><b>Mounting.</b> {esc(ears)}</p></div>')

    def dc_closeups(self):
        b = self.b
        out = []
        figs = []
        for d in sorted([d for d in b.D.values() if d["kind"] in ("fuse", "dcdist")], key=lambda d: d["kind"] != "fuse"):
            name = f"panel_{d['id']}"
            if name not in b.renders:
                continue
            face = b.renders[name].get("face")
            tags = self.panel_tags(name, lambda pid: self.std_lines(pid, far="device"))
            lbl = {p: b.P[p].get("label") for p in b.renders[name].get("ports") or []}
            busnote = ""
            if d["kind"] == "fuse":
                busnote = (" Bus A feeds F1-F5, bus B F6-F10; the input studs are on bus A: fit the A-B strap (both poles) so bus B is live. "
                           f"Fuses: GMT, {self.part_text('GMT')}.")
            figs.append(self.fig_tags(name, PAGE_W, 36, tags,
                                     caption=f"<b>{esc(d['id'])}</b> ({esc(d.get('u'))}, {esc(d.get('title') or '')}) seen from the {esc(face).upper()}: "
                                             f"{esc(d.get('note') or '')}.{esc(busnote)}"))
        # wiring table
        rows = []
        dc48 = [d["id"] for d in b.D.values() if d["kind"] == "dcdist"]
        dcn = [d["id"] for d in b.D.values() if d["kind"] == "fuse"]
        if dc48:
            rows.append([f'<span class="cid">{esc(", ".join(dc48))}</span> CBn', "+48 V to the node adaptors",
                         "Belden 8461 2 × 18 AWG: black core → RTN (0 V); other core (white / red) → +48 V at CBn",
                         "ferrule 0.75 mm² Phoenix 3200519 on both cores", '<b style="color:#b42318">red</b> tag both ends'])
        if dcn:
            rows.append([f'<span class="cid">{esc(", ".join(dcn))}</span> Fn', "-48 V to the IP-50 radios (positive earth)",
                         "Belden 8461 2 × 18 AWG: black core → RTN (0 V, earthed); other core → -48 V at Fn (through the GMT fuse)",
                         "ferrule 0.75 mm² Phoenix 3200519; radio end: IP-50 DC plug TBC (open question 5)", '<b style="color:#1d4ed8">blue</b> tag both ends'])
        bus = [c for c in b.C.values() if c["cls"] == "DC" and "bus" in (c.get("pn") or "") + (c.get("part") or "")]
        if bus:
            rows.append([", ".join(f'<span class="cid">{esc(c["id"])}</span>' for c in bus), "rectifier → distribution bus",
                         "2 × 10 AWG: red / black per the rectifier marking", "ring lugs, hole to match the stud: stud size TBC (measure)",
                         "same colour rule as its system"])
        if dcn:
            rows.append(["A-B strap", f"{esc(', '.join(dcn))} bus A → bus B", "short 10 AWG links (or the vendor strap), both poles",
                         "ring lugs to match the input studs (TBC)", "-"])
        tab = self.table([("Where", ""), ("System", ""), ("Conductor → terminal", ""), ("Termination", ""), ("Tag", "")], rows,
                         widths=[16, 16, 34, 22, 12], cls="tight")
        out.append(f'<div class="newpage"><h3>DC wiring rules</h3><p class="small">One convention for both systems (the conductor colours of the '
                   f'Belden 8461 on the drum decide which core is the "other" one: write it on the drum). Check polarity with a meter at every '
                   f'output before a load is connected.</p>{tab}<h3>DC terminals, one by one</h3><p class="small">Straight-on pictures with the '
                   f'cables left out. Tag = terminal and cable ID, then what it feeds.</p>{"".join(figs)}</div>')
        return "".join(out)

    def step_of(self, key):
        for st in self.b.steps:
            if key in st["title"].lower():
                return st["n"]
        return "?"

    def part_text(self, key):
        for r in self.parts():
            if key in (r.get("part number") or "") or key in (r.get("item") or ""):
                return f'{r.get("part number")} ({r.get("status")})'
        return "TBC"

    # ---------------------------------------------------------- IF link
    def link_block(self, st, own, shared):
        b = self.b
        n = st["n"]
        name = re.match(r"^(?:IF link|Coupling arm)\s+(\S+)", st["title"]).group(1)
        att_ports = sorted({p for c in st["cables"] for p in (b.C[c]["a"], b.C[c]["b"]) if re.match(r"^ATT-\d+\.ch\d+\.(in|out)$", p)}, key=natkey)
        chans = sorted({int(re.match(r"^ATT-\d+\.ch(\d+)", p).group(1)) for p in att_ports})
        att = att_ports[0].split(".")[0] if att_ports else "?"
        rows = []
        bad = []
        side_tiles = {"a": {}, "b": {}}
        for k in chans:
            ain, aout = f"{att}.ch{k}.in", f"{att}.ch{k}.out"
            ta, ea = b.walk_out(ain)
            tb, eb = b.walk_out(aout)

            def path_txt(toks, rev):
                s = []
                for t in (list(reversed(toks)) if rev else toks):
                    if t[0] == "cable":
                        s.append(f'<span class="cid">{esc(t[1])}</span>')
                    else:
                        s.append(f'<span class="mono">{esc(t[1])}</span> <span class="muted">({esc(path_label(t, rev))})</span>')
                return " → ".join(s)

            def end_txt(pid):
                sec = b.sector(pid)
                return f'<span class="mono"><b>{esc(pid)}</b></span>' + (f' <span class="muted">{esc(sec)}</span>' if sec else "")

            for side, e in (("a", ea), ("b", eb)):
                if b.tile_no(e) is not None:
                    side_tiles[side][e] = k
            tn = {b.tile_no(ea), b.tile_no(eb)} - {None}
            if tn and tn != {k}:
                bad.append(k)
            rows.append([f"<b>{k}</b>", end_txt(ea), path_txt(ta, True), f'<b class="mono">{esc(att)}</b> CH{k} <span class="muted">in→out</span>',
                         path_txt(tb, False), end_txt(eb)])
        tab = self.table([("Tile", "c"), ("A side: tile, sector", ""), ("A path", ""), ("Attenuator", ""), ("B path", ""), ("B side: tile, sector", "")],
                         rows, widths=[4, 17, 25, 14, 23, 17], cls="tight")
        warn = (f'<div class="box warn small">Tile and channel numbers differ for channel(s) {", ".join(map(str, bad))}: check the cable schedule.</div>' if bad else "")
        # tiles close-up with A/B numbers
        tiles_name = f"link_{name}_tiles"
        fig = ""
        if tiles_name in b.renders:
            r = b.renders[tiles_name]
            ov = []
            for side, cls in (("a", "a"), ("b", "b")):
                for pid, k in side_tiles[side].items():
                    an = r["anchors"].get(pid + "#below") or r["anchors"].get(pid)
                    if an and an["in"]:
                        ov.append({"x": an["x"], "y": an["y"], "text": f"{side.upper()}{k}", "cls": cls})
            fig = self.fig(tiles_name, PAGE_W, 58, overlays=ov,
                           caption=f"<b>Spider-box tiles of {esc(name)}</b>, straight on: <b style='color:#6d28d9'>A1-A4</b> = A-side tiles, "
                                   f"<b style='color:#0f766e'>B1-B4</b> = B-side tiles; number = attenuator channel.")
        sh = ""
        if shared:
            sh = (f'<p class="small muted">Cables {", ".join(shared)} belong to this link too but are listed in step '
                  f'{", ".join(str(s) for s in sorted({b.owner[c] for c in shared}))} (fitted once, shared through the splitter).</p>')
        return (f'<div class="newpage"><h3>Tile mapping: tile i → {esc(att)} channel i</h3>'
                f'<p class="small">From the A-side tile through the attenuator to the B-side tile; passive parts are named with the ports used. '
                f'3D close-up: <span class="mono">{esc(b.poc)}_renders/link_{esc(name)}.png</span> (or search {esc(own[0] if own else "")} in the twin and press "Trace signal").</p>'
                f'{tab}{warn}{fig}<h3>Connect these {len(own)} cables</h3>{self.cable_table(own)}{sh}</div>')

    # ---------------------------------------------------------- IP-50
    def ip50_block(self, st, own):
        b = self.b
        mw = [d for d in b.D.values() if d["kind"] == "mwatt"]
        rows = []
        if mw:
            m = mw[0]
            chans = sorted({int(re.match(r".*\.ch(\d+)\.", p).group(1)) for c in own for p in (b.C[c]["a"], b.C[c]["b"]) if re.match(rf"^{re.escape(m['id'])}\.ch\d+\.(in|out)$", p)})
            for k in chans:
                ta, ea = b.walk_out(f"{m['id']}.ch{k}.in")
                tb, eb = b.walk_out(f"{m['id']}.ch{k}.out")

                def ptxt(toks, rev):
                    seq = list(reversed(toks)) if rev else toks
                    return " → ".join(f'<span class="cid">{esc(t[1])}</span>' if t[0] == "cable" else f'<span class="mono">{esc(t[1])}</span>' for t in seq)

                hop = re.sub(r"^IP50-|@.*$", "", b.dev_of(ea)["id"]) if b.dev_of(ea) else "?"
                rows.append([f"<b>{esc(hop)}</b>", f'<span class="mono"><b>{esc(ea)}</b></span>', ptxt(ta, True), f'<b class="mono">{esc(m["id"])}</b> CH{k}',
                             ptxt(tb, False), f'<span class="mono"><b>{esc(eb)}</b></span>'])
        tab = self.table([("Hop", ""), ("Radio A", ""), ("A path", ""), ("Attenuator", ""), ("B path", ""), ("Radio B", "")],
                         rows, widths=[9, 17, 24, 11, 24, 17], cls="tight")
        # radio columns, tagged
        cols = []
        for name in ("ip50_col_1", "ip50_col_2"):
            r = b.renders.get(name)
            if not r:
                continue
            tags = []
            for key, an in r["anchors"].items():
                if not an["in"]:
                    continue
                if key in b.D and b.D[key]["kind"] == "ip50":
                    d = b.D[key]
                    under = b.below(d)
                    tags.append({"x": an["x"], "y": an["y"], "side": "top", "fill": "#1f2933",
                                 "lines": [key] + ([f"stacked on {re.sub(r'^IP50-', '', under['id'])}"] if under else
                                                   [f"under {re.sub(r'^IP50-', '', b.on_top(d)[0]['id'])}"] if b.on_top(d) else [])})
                elif key in b.D and b.D[key]["kind"] == "transition":
                    tr = b.D[key]
                    fl = [b.portcable[q]["id"] for q in tr["ports"] if q in b.portcable and b.P[q]["type"] == "WG"]
                    cx = [b.portcable[q]["id"] for q in tr["ports"] if q in b.portcable and b.P[q]["type"] != "WG"]
                    tags.append({"x": an["x"], "y": an["y"], "side": "bottom", "fill": DISPLAY_COLORS["WG"],
                                 "lines": [key, f"{'/'.join(fl)} flange, pad, {'/'.join(cx)}"]})
            cols.append(f'<div class="keep">{self.fig_tags(name, PAGE_W, 150, tags, caption=f"<b>{esc(r[chr(116) + "itle"])}</b>")}</div>')
        colhtml = '<div class="newpage"></div>'.join(cols)
        # MWATT front panel
        mwfig = ""
        if mw and f"panel_{mw[0]['id']}" in b.renders:
            name = f"panel_{mw[0]['id']}"

            def lines(pid):
                c, o, od = self.port_info(pid)
                nm = b.P[pid]["name"].replace("ch", "CH").replace(".in", " IN").replace(".out", " OUT")
                return ([nm, f"{c['id']} {od['id'] if od else o}"] if c else [nm, "spare"]), (c["color"] if c else "#e5e7eb")
            tags = self.panel_tags(name, lines, side_fn=lambda pid: "top" if pid.endswith(".in") else "bottom")
            mwfig = self.fig_tags(name, PAGE_W, 62, tags, caption=f"<b>{esc(mw[0]['id'])}</b> front panel straight on (operator side), cables left out: "
                                                                   f"CHn IN above, CHn OUT below")
        hw = ('<div class="box note small"><b>Hardware per waveguide joint</b> (the W-cables drawn as a ring are flange joints, not cables): '
              'radio flange to transition flange, WR-62, with 4 × #4-40 screws and lock washers (TBC with the transition datasheet), tightened '
              'crosswise to about 0.6 N·m (5 in-lb, TBC); fit the flange gasket only if the radio flange is the gasketed (choke) type (TBC, open question 5). '
              'Then screw the 30 dB pad (SA18S10W-30) directly onto the transition SMA, its input (high-power) side on the transition: if the pad '
              'carries an arrow it points away from the radio, toward the MWATT (TBC with the pad datasheet). Then the SMA cable to the MWATT, '
              'all SMA at 8 in-lb.</div>')
        return (f'<div class="newpage"><h3>Chains: radio → transition → pad → MWATT channel → pad → transition → radio</h3>{tab}{hw}{mwfig}</div>'
                f'<div class="newpage"><h3>Radio columns, tagged</h3>{colhtml}</div>'
                f'<div class="newpage"><h3>Connect these {len(own)} cables</h3>{self.cable_table(own)}</div>')

    # ---------------------------------------------------------- 5 port maps
    def port_maps(self):
        b = self.b
        body = [self.fig("power", PAGE_W, 150, caption="<b>Power only</b> (AC red, DC yellow, PoE orange), from the rear left")]
        self.section("ports", "Port maps", "".join(body), num="5")
        # switch port plans
        md = (OUT / f"{b.poc}_cable_schedule.md").read_text()
        for key, title, anchor in (("SW-DATA port plan", "SW-DATA port plan", "sw-data"), ("SW-OOB port plan", "SW-OOB port plan", "sw-oob")):
            hdr, rows = parse_md_table(md, key)
            if not hdr:
                continue
            ci = hdr.index("Cable") if "Cable" in hdr else None
            out = []
            for r in rows:
                cells = []
                for i, v in enumerate(r):
                    if i == ci:
                        m = re.match(r"^([A-Z]\d{3})(.*)$", v)
                        if m and m.group(1) in b.C:
                            cells.append(f'{swatch(b.C[m.group(1)]["color"])}<span class="cid">{esc(m.group(1))}</span>{esc(m.group(2))}')
                            continue
                    cells.append(f'<span class="mono">{esc(v)}</span>' if i in (0,) or "." in v else esc(v))
                out.append(cells)
            dev = "SW-DATA" if "DATA" in key else "SW-OOB"
            d = b.D.get(dev, {})
            lead = f'<p class="small">{esc(dev)} · {esc(d.get("sub", ""))} · {esc(d.get("u", ""))}. {esc(d.get("note", ""))}</p>'
            tab = self.table([(esc(h), "") for h in hdr], out, cls="tight")
            html_ = lead + self.switch_closeup(dev) + tab
            self.section(anchor, title, html_, level=2)
        self.section("pdu-map", "PDU outlet map", self.pdu_map(), level=2)
        self.section("dc-map", "DC breaker and fuse map", self.dc_map(), level=2)

    def pdu_block(self, d, with_fig=False):
        b = self.b
        sw = set(b.meta.get("switched") or [])
        outs = sorted([p for p in d["ports"] if re.match(r"^O\d+$", b.P[p]["name"])], key=natkey)
        rows = []
        for p in outs:
            c = b.portcable.get(p)
            if not c:
                rows.append(([esc(b.P[p]["name"]), '<span class="muted">spare</span>', "", ""], "sub"))
                continue
            o = b.other_end(c, p)
            od = b.dev_of(o)
            flag = '<span class="chip off">OFF at t0</span>' if od and od["id"] in sw else ""
            rows.append([f"<b>{esc(b.P[p]['name'])}</b>", f'<span class="cid">{esc(od["id"] if od else o)}</span> {flag}',
                         f'{swatch(c["color"])}<span class="cid">{esc(c["id"])}</span>', esc(c.get("length") or "")])
        inlet = [c for c in b.C.values() if c["b"] == f"{d['id']}.inlet" or c["a"] == f"{d['id']}.inlet"]
        feed = ""
        if inlet:
            c = inlet[0]
            feed = f'fed by <span class="cid">{esc(c["id"])}</span> from <span class="mono">{esc(b.other_end(c, d["id"] + ".inlet"))}</span>'
        else:
            ac = [c for c in b.C.values() if c["cls"] == "AC" and (b.dev_of(c["a"]) or {}).get("id") == d["id"] and not re.match(r"^O\d+$", b.P[c["a"]]["name"])]
            feed = f'fed by <span class="cid">{esc(ac[0]["id"])}</span>' if ac else ""
        hd = (f'<h3>{esc(d["id"])} <span class="muted" style="font-weight:400">{esc(d.get("u") or "table")}</span></h3>'
              f'<p class="small muted">{esc(d.get("sub") or "")}; {feed}</p>')
        cols = [("Outlet", ""), ("Feeds", ""), ("Cable", ""), ("Lead", "")]
        if not with_fig:
            tab = self.table(cols, rows, widths=[10, 44, 17, 12], cls="tight")
            return f'<div class="keep" style="flex:1">{hd}{tab}</div>'
        third = max(1, -(-len(rows) // 3))
        self.table_w = PAGE_W / 3 - 4
        tabs = [self.table(cols, rows[i:i + third], widths=[12, 46, 22, 15], cls="tight") for i in range(0, len(rows), third)]
        self.table_w = PAGE_W
        name = f"panel_{d['id']}"
        fig = ""
        if name in b.renders:
            def lines(pid):
                c, o, od = self.port_info(pid)
                nm = b.P[pid]["name"]
                if not c:
                    return [f"{nm} spare"], "#e5e7eb"
                if od and od["id"] in sw:
                    return [f"{nm} {c['id']} OFF@t0"], "#7a1610"
                return [f"{nm} {c['id']}"], c["color"]
            tags = self.panel_tags(name, lines)
            xs = {b.P[t["pid"]]["name"]: t["x"] for t in tags}
            o1 = "RIGHT" if xs.get("O1", 0) > xs.get(f"O{len(outs)}", 1) else "LEFT"
            fig = self.fig_tags(name, PAGE_W, 44, tags,
                                caption=f"<b>{esc(d['id'])}</b> rear, straight on (seen from behind the rack: outlet O1 is on the {o1})")
        return f'<div class="keep">{hd}{fig}<div class="cols">{"".join(f"<div style=flex:1>{t}</div>" for t in tabs)}</div></div>'

    def pdu_map(self):
        b = self.b
        pdus = [d for d in b.D.values() if d["kind"] in ("pdu", "strip")]
        sw = set(b.meta.get("switched") or [])
        blocks = [self.pdu_block(d) for d in pdus]
        # 2 per row
        rows_html = []
        for i in range(0, len(blocks), 2):
            rows_html.append(f'<div class="cols">{"".join(blocks[i:i + 2])}{"<div style=flex:1></div>" if len(blocks[i:i + 2]) == 1 else ""}</div>')
        ac = self.step_of("cable: ac")
        lead = (f'<p class="small">Outlet → device → cable (pictures of each PDU with its outlets tagged: <a href="#pdu-outlets">step {ac}</a>). '
                'Outlets marked <span class="chip off">OFF at t0</span> are switched off by the bench '
                'controller at the event; keep them on the switched PDU.</p>' if sw else
                f'<p class="small">Outlet → device → cable (pictures of each PDU with its outlets tagged: <a href="#pdu-outlets">step {ac}</a>).</p>')
        return lead + "".join(rows_html)

    def dc_map(self):
        b = self.b
        out = []
        for d in [d for d in b.D.values() if d["kind"] in ("dcdist", "fuse", "rect", "psu")]:
            rows = []
            for p in sorted(d.get("ports") or [], key=natkey):
                pp = b.P[p]
                if pp["type"] != "DC":
                    continue
                c = b.portcable.get(p)
                if not c:
                    rows.append(([f"<b>{esc(pp['name'])}</b>", '<span class="muted">spare (OFF)</span>', "", "", esc(pp.get("label") if pp.get("label") != pp["name"] else "")], "sub"))
                    continue
                o = b.other_end(c, p)
                od = b.dev_of(o)
                then = ""
                if od and od["kind"] in ("adaptor", "injector", "psu"):
                    for q in od.get("ports") or []:
                        cc = b.portcable.get(q)
                        if cc and cc["id"] != c["id"] and cc["cls"] in ("POE", "DC"):
                            then = f'<span class="cid">{esc(cc["id"])}</span> → <span class="mono">{esc(b.other_end(cc, q))}</span>'
                            break
                lbl = pp.get("label") if pp.get("label") and pp.get("label") != pp["name"] else ""
                note = "; ".join(x for x in (lbl, c.get("note")) if x)
                rows.append([f"<b>{esc(pp['name'])}</b>", f'<span class="mono">{esc(o)}</span>', f'{swatch(c["color"])}<span class="cid">{esc(c["id"])}</span> {esc(c.get("length") or "")}', then, esc(note)])
            if not rows:
                continue
            tab = self.table([("Position", ""), ("Feeds", ""), ("Cable", ""), ("Then", ""), ("Note", "")], rows, widths=[10, 22, 18, 26, 24], cls="tight")
            out.append(f'<div class="keep"><h3>{esc(d["id"])} <span class="muted" style="font-weight:400">{esc(d.get("u") or d["group"])} · {esc(d.get("sub") or "")}</span></h3>{tab}</div>')
        return ('<p class="small">Each breaker / fuse position, the cable on it, and what that feeds next (adaptor → node ETH1, or the radio). '
                'Spare positions stay OFF. Switch breakers on one by one at power-up.</p>' + "".join(out))

    # ---------------------------------------------------------- 6 checks
    def checks(self):
        b = self.b
        rows = [[esc(c["name"]), f'<span class="chip {"ok" if c["ok"] else "bad"}">{"PASS" if c["ok"] else "FAIL"}</span>', esc(c.get("detail") or "")] for c in b.scene["checks"]]
        tab = self.table([("Model check", ""), ("Result", ""), ("Detail", "")], rows, widths=[38, 9, 53])
        ov = overlaps(b)
        manual = [
            "Every cable carries its ID tag at both ends.",
            "Every SMA torqued to 8 in-lb; no cable bent tighter than its minimum radius.",
            "Unused Spider tiles and coupler ports terminated (50-ohm loads).",
            "Every rack device on its U as in the front elevation; blanking panels in every empty U.",
            "DC polarity checked on every breaker and fuse output before any node or radio is connected (meter).",
            "PDUs, breakers and fuses OFF; PDU cords not yet in the wall.",
            "Every attenuator set to its start value (bench controller) before any radio is powered.",
        ]
        man = "".join(f'<li style="list-style:none;margin-left:-5mm"><span class="checkbox"></span>{esc(x)}</li>' for x in manual)
        last = b.steps[-1]
        po = power_up_list(last["text"])
        detail = []
        for item in po:
            k = item.lower()
            devs = []
            if "pdu" in k:
                devs = [d for d in b.D.values() if d["kind"] == "pdu"]
                feeds = [c for c in b.C.values() if c["cls"] == "AC" and any(c[e].startswith("WALL.") for e in ("a", "b"))]
                extra = "plug the PDU cords into the wall: " + ", ".join(
                    f'{c["id"]} ({(c["b"] if c["a"].startswith("WALL.") else c["a"]).split(".")[0]} → {c["a"] if c["a"].startswith("WALL.") else c["b"]})' for c in feeds)
            elif "dc" in k:
                devs = [d for d in b.D.values() if d["kind"] in ("rect", "dcdist", "fuse")]
                extra = "rectifiers on; then breakers / fuses one at a time, checking each load"
            elif "switch" in k:
                devs = [d for d in b.D.values() if d["kind"] in ("sw_data", "sw_oob")]
                extra = "wait for the switches to boot before the nodes"
            elif "node" in k:
                devs = [d for d in b.D.values() if d["kind"] == "spider"]
                extra = "nodes come up as their breakers / injector feed them"
            elif "radio" in k:
                devs = [d for d in b.D.values() if d["kind"] == "ip50"]
                extra = "only after every MWATT channel is set"
            else:
                extra = ""
            ids = ", ".join(d["id"] for d in devs[:8]) + (" …" if len(devs) > 8 else "")
            detail.append(f"<li><b>{esc(item)}</b>{': ' + esc(extra) if extra else ''}{f' <span class=muted>({esc(ids)})</span>' if ids else ''}</li>")
        ovhtml = ""
        if ov:
            half = (len(ov) + 1) // 2
            mk = lambda part: self.table([("Item", ""), ("Intersects", ""), ("Overlap x × y × z mm", "")],
                                         [[f'<span class="cid">{esc(a)}</span>', f'<span class="cid">{esc(c)}</span>', esc(v)] for a, c, v in part], widths=[34, 34, 32], cls="tight")
            ovhtml = ('<div class="keep"><h3>Model warnings: parts whose boxes intersect in the twin</h3><p class="small">The twin places these parts '
                      'so that they overlap: they will not fit exactly as drawn. Set them out on site (stagger, space or restack them) and '
                      'report the real positions so the twin can be corrected.</p>'
                      f'<div class="cols"><div style="flex:1">{mk(ov[:half])}</div><div style="flex:1">{mk(ov[half:])}</div></div></div>')
        body = f"""
<div class="cols">
 <div style="flex:1"><h3>Model checks (from the twin)</h3>{tab}
  <h3>Before power-up: installer checklist</h3><ul>{man}</ul></div>
 <div style="flex:1"><h3>Power-up order</h3><p class="small">From the last step: “{esc(last['text'])}”</p><ol>{''.join(detail)}</ol></div>
</div>{ovhtml}"""
        self.section("checks", "Install checks and power-up", body, num="6")
        if b.meta.get("event"):
            self.event()

    def event(self):
        b = self.b
        rows = []
        for did in b.meta.get("switched") or []:
            d = b.D.get(did)
            if not d:
                continue
            feeds = []
            for p in d.get("ports") or []:
                c = b.portcable.get(p)
                if c and c["cls"] == "AC":
                    o = b.other_end(c, p)
                    feeds.append(f'<span class="mono"><b>{esc(o)}</b></span> via <span class="cid">{esc(c["id"])}</span>')
            dark = []
            dk = b.dark
            for key, why in (dk.get("why") or {}).items():
                if why.get("from") != did:
                    continue
                node = next((n for n in dk.get("nodes") or [] if f"{n['box']}.{n['node']}" == key), None)
                via = ", ".join(why.get("via") or [])
                if node:
                    dark.append(f'node <b class="mono">{esc(node["node"])}</b> in <span class="cid">{esc(node["box"])}</span> and its tiles (via <span class="cid">{esc(via)}</span>)')
                else:
                    dark.append(f'<span class="cid">{esc(key)}</span> (via <span class="cid">{esc(via)}</span>)')
            rows.append([f'<span class="cid">{esc(did)}</span>', esc(b.what(d)), "<br>".join(feeds) or "-", "<br>".join(dark) or "itself only"])
        tab = self.table([("Switched device", ""), ("What it is", ""), ("PDU outlet", ""), ("Also goes dark (no power)", "")], rows, widths=[20, 22, 24, 34])
        dk = b.dark
        links = sorted(dk.get("links") or [])
        link_rows = []
        for l in links:
            cids = [c for c in dk.get("cables") or [] if c in b.C and (l in (b.C[c].get("bundle") or "") or
                    (l.startswith("IP-50 ") and l[6:] in (b.dev_of(b.C[c]["a"]) or {}).get("id", "") + (b.dev_of(b.C[c]["b"]) or {}).get("id", "")))]
            steps = sorted({b.owner[c] for c in cids if c in b.owner})
            link_rows.append([f'<b class="mono">{esc(l)}</b>', str(len(cids)), ", ".join(f'<a href="#step-{x}">{x}</a>' for x in steps)])
        dtab = self.table([("Link down after t0", ""), ("Dead cables", "n"), ("Steps", "")], link_rows, widths=[40, 20, 40], cls="tight") if link_rows else ""
        fig = self.fig("event_t0", 160, 84, caption="<b>After t0</b>: switched devices dark with a red outline and an OFF tag; what they fed "
                       "dark with a brown outline; every dead cable grey and dashed")
        body = f"""
<div class="box warn"><b>Event at t0.</b> {esc(b.meta['event'])}</div>
<div class="cols" style="margin-top:3mm">
 <div style="flex:1"><h3>Outlets to switch OFF at t0</h3>{tab}
  <ol class="small"><li>Before t0: every link up, attenuators at their start values, logging running on BENCH-CTL.</li>
  <li>At t0: switch the outlets above OFF together from the bench controller (PDUs on SW-OOB). Do not touch any attenuator at t0.</li>
  <li>After the run: switch the same outlets ON again and let the nodes and radios re-join before the next run.</li></ol></div>
 <div style="flex:0 0 162mm">{fig}{f'<h3>What goes down with them ({len(dk.get("cables") or [])} cables)</h3>' + dtab if dtab else ''}</div>
</div>"""
        self.section("event", "Event procedure (t0)", body, level=2)

    # ---------------------------------------------------------- 7 assumptions
    def assumptions(self):
        b = self.b
        grouped = OrderedDict()
        loose = []
        for a in b.scene.get("assumptions") or []:
            m = re.match(r"^([A-Za-z0-9@._-]+): (.*)$", a)
            if m and m.group(1) in b.D:
                grouped.setdefault(m.group(2), []).append(m.group(1))
            else:
                loose.append(a)
        items = [f"<li>{esc(a)}</li>" for a in loose]
        for text, ids in grouped.items():
            items.append(f'<li><span class="mono"><b>{esc(", ".join(ids))}</b></span>: {esc(text)}</li>')
        body = f'<h3>Assumptions in the twin ({len(b.scene.get("assumptions") or [])})</h3><ul class="small">{"".join(items)}</ul>'
        q = TWIN / "device_questions.md"
        if q.exists():
            body += f'<div class="md newpage">{md_to_html(q.read_text())}</div>'
        self.section("assumptions", "Assumptions and open questions", body, num="7")

    # ---------------------------------------------------------- page
    def build(self, pages=None):
        self.toc = []
        self.sections = []
        self.cable_rows = Counter()
        self.cover()
        self.how_to_read()
        self.overview()
        self.bom()
        self.steps()
        self.port_maps()
        self.checks()
        self.assumptions()
        # every cable of the scene in exactly one step table
        want = set(self.b.C)
        listed = {c for c, k in self.cable_rows.items() if k > 0}
        missing = sorted(want - listed, key=natkey)
        twice = sorted([c for c, n in self.cable_rows.items() if n > 1], key=natkey)
        extra = sorted(listed - want, key=natkey)
        if missing or twice or extra:
            raise SystemExit(f"{self.b.poc}: cable/step table mismatch: missing {missing}, listed twice {twice}, unknown {extra}")
        toc = self.toc_page(pages)
        secs = self.sections[:1] + [toc] + self.sections[1:]
        m = self.b.meta
        title = f"UNITY-6G {m['poc']} installation guide (TENTATIVE, {DATE})"
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{esc(title)}</title><meta name="generator" content="testbed_design/twin/guide.py"><style>{CSS}</style></head>'
                f'<body><div class="sheet">{"".join(secs)}</div></body></html>')


# ============================================================== helpers
def parse_md_table(md: str, heading_key: str):
    lines = md.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("## ") and heading_key in l:
            j = i + 1
            while j < len(lines) and not lines[j].startswith("|"):
                j += 1
            rows = []
            while j < len(lines) and lines[j].startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                rows.append(cells)
                j += 1
            if len(rows) >= 2:
                return rows[0], [r for r in rows[2:]]
    return None, []


def power_up_list(text: str):
    m = re.search(r"power up:\s*(.*?)\.?$", text)
    if not m:
        return [text]
    items, depth, cur = [], 0, ""
    for ch in m.group(1):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    return items


def overlaps(b: Bench):
    """pairs of device boxes that intersect (not parent / child)."""
    out = []
    devs = [d for d in b.D.values() if d["group"] in ("rack", "table")]
    for i, a in enumerate(devs):
        for c in devs[i + 1:]:
            if a.get("parent") == c["id"] or c.get("parent") == a["id"]:
                continue
            ov = [min(a["pos"][k] + a["size"][k], c["pos"][k] + c["size"][k]) - max(a["pos"][k], c["pos"][k]) for k in range(3)]
            if all(v >= 2.0 for v in ov):
                out.append((a["id"], c["id"], " × ".join(str(round(v)) for v in ov)))
    return out


def md_to_html(md: str) -> str:
    def inline(s):
        s = esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`(.+?)`", r'<span class="mono">\1</span>', s)
        return s

    out, para, items = [], [], []

    def flush_p():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def flush_l():
        if items:
            out.append("<ol>" + "".join(f"<li>{inline(' '.join(it))}</li>" for it in items) + "</ol>")
            items.clear()

    for line in md.splitlines():
        if line.startswith("# "):
            flush_p(); flush_l()
            out.append(f'<h3 style="font-size:12pt">{inline(line[2:])}</h3>')
        elif re.match(r"^\d+\.\s", line):
            flush_p()
            items.append([re.sub(r"^\d+\.\s+", "", line)])
        elif line.startswith("   ") and items:
            items[-1].append(line.strip())
        elif not line.strip():
            flush_p()
        else:
            if items:
                flush_l()
            para.append(line.strip())
    flush_p(); flush_l()
    return "".join(out)


# ============================================================== PDF
PRINT_JS = r"""
const { chromium } = require('playwright');
// node -e <code> a b c  ->  process.argv = [node, a, b, c]
const [htmlPath, pdfPath, header, footerLeft, checkOnly] = process.argv.slice(1);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1040, height: 700 } });
  await page.goto('file://' + htmlPath, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => [...document.images].every((i) => i.complete && i.naturalWidth > 0), null, { timeout: 120000 });
  await page.emulateMedia({ media: 'print' });
  // layout checks at the printable width (275 mm = 1039 CSS px)
  const issues = await page.evaluate(() => {
    const out = [];
    const px = (el) => parseFloat(getComputedStyle(el).fontSize) * 0.75;
    const name = (el) => (el.closest('section') && el.closest('section').querySelector('h1,h2') ? el.closest('section').querySelector('h1,h2').textContent.trim().slice(0, 50) : '?');
    for (const t of document.querySelectorAll('table')) {
      const w = t.getBoundingClientRect().width;
      const pw = t.parentElement.getBoundingClientRect().width;
      if (w > pw + 1 || w > 1040) out.push(`table wider than its column (${w.toFixed(0)} > ${pw.toFixed(0)} px) in "${name(t)}"`);
    }
    for (const el of document.querySelectorAll('td, th, p, li, h1, h2, h3, figcaption, .steptext, .stat, .ov')) {
      if (el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflow !== 'visible') out.push(`clipped text in <${el.tagName.toLowerCase()}> "${el.textContent.trim().slice(0, 40)}" in "${name(el)}"`);
      if (el.scrollWidth > el.clientWidth + 2 && el.tagName === 'TD') out.push(`cell content wider than the cell: "${el.textContent.trim().slice(0, 40)}" in "${name(el)}"`);
    }
    // font sizes: body >= 9.5 pt, tables >= 8 pt (overlay tags on pictures and chips >= 7.5 pt)
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const seen = new Set();
    while (walker.nextNode()) {
      const n = walker.currentNode;
      if (!n.textContent.trim()) continue;
      const el = n.parentElement;
      if (seen.has(el)) continue;
      seen.add(el);
      if (el.closest('svg')) continue;
      const pt = px(el);
      const inTable = !!el.closest('table');
      const small = el.closest('.ov, .chip, figcaption, .tcap, .stat span, .sub, .muted, .tent, .toc');
      const min = small ? 7.5 : inTable ? 8 : 9.5;
      if (pt < min - 0.01) out.push(`font ${pt.toFixed(1)} pt < ${min} pt: "${n.textContent.trim().slice(0, 40)}" in "${name(el)}"`);
    }
    const imgs = [...document.images].filter((i) => i.getBoundingClientRect().width > 1040);
    imgs.forEach((i) => out.push(`image wider than the page: ${i.src.slice(-40)}`));
    return out;
  });
  console.log(JSON.stringify({ issues }));
  if (checkOnly === '1') { await browser.close(); return; }
  const tpl = (inner) => `<div style="width:100%;font-family:Inter,Arial,sans-serif;font-size:10.5px;color:#4b5563;padding:0 11mm;display:flex;justify-content:space-between;align-items:center;-webkit-print-color-adjust:exact">${inner}</div>`;
  await page.pdf({
    path: pdfPath, format: 'A4', landscape: true, printBackground: true, outline: true, tagged: true,
    displayHeaderFooter: true,
    headerTemplate: tpl(`<span>${header}</span><span style="color:#8a5a00;font-weight:700">TENTATIVE · geometry partly estimated</span>`),
    footerTemplate: tpl(`<span>${footerLeft}</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>`),
    margin: { top: '14mm', bottom: '11mm', left: '11mm', right: '11mm' },
  });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def print_pdf(html_path: Path, pdf_path: Path, header: str, footer_left: str):
    r = subprocess.run(["node", "-e", PRINT_JS, str(html_path), str(pdf_path), header, footer_left, "0"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"PDF printing failed:\n{r.stdout}\n{r.stderr}")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return json.loads(line[-1])["issues"] if line else []


def pdf_pages(pdf_path: Path, anchors_titles):
    """page number (1-based) of each heading, from the PDF outline."""
    from pypdf import PdfReader

    rd = PdfReader(str(pdf_path))
    flat = []  # outline entries in document order
    norm = lambda t: re.sub(r"\s+", "", t or "").lower()

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
                continue
            try:
                flat.append((norm(it.title), rd.get_destination_page_number(it) + 1))
            except Exception:
                pass

    walk(rd.outline)
    pages = {}
    i = 0
    for anchor, title in anchors_titles:
        key = norm(title)
        j = i
        while j < len(flat) and key not in flat[j][0]:
            j += 1
        if j < len(flat):
            pages[anchor] = flat[j][1]
            i = j + 1
    return pages, len(rd.pages)


def fix_outline(pdf_path: Path, entries):
    """Replace Chromium's tagged-PDF outline (it doubles section titles, e.g.
    'ContentsContents') with one bookmark per heading: entries = [(level, title, page)]."""
    from pypdf import PdfWriter

    w = PdfWriter(clone_from=str(pdf_path))
    if "/Outlines" in w._root_object:
        del w._root_object["/Outlines"]
    parent = None
    for level, title, page in entries:
        if page is None:
            continue
        it = w.add_outline_item(title, page - 1, parent=parent if level > 1 else None)
        if level == 1:
            parent = it
    w.page_mode = "/UseOutlines"
    tmp = pdf_path.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as f:
        w.write(f)
    tmp.replace(pdf_path)


# ============================================================== main
def build_guide(poc: str, do_pdf: bool):
    b = Bench(poc)
    load_manifest(b)
    out_html = TWIN / f"{poc}_install_guide.html"
    rel = lambda name: f"{poc}_renders/{b.renders[name]['file']}"
    g = Guide(b, rel)
    out_html.write_text(g.build())
    print(f"{poc}: {out_html} ({out_html.stat().st_size / 1e6:.2f} MB)")
    if not do_pdf:
        return None
    out_pdf = TWIN / f"{poc}_install_guide.pdf"
    header = f"UNITY-6G {b.meta['poc']} installation guide - TENTATIVE - {DATE}"
    footer = f"{b.meta.get('drawing') or ''} · interactive twin: {poc}_twin.html"
    with tempfile.TemporaryDirectory(prefix=f"{poc}_guide_") as td:
        td = Path(td)
        jpg = make_print_copies(b, td / "img")
        gp = Guide(b, lambda name: jpg[name].as_uri())
        html_p = td / f"{poc}_install_guide.html"
        heads = None
        pages = None
        for it in range(3):
            html_p.write_text(gp.build(pages))
            issues = print_pdf(html_p, out_pdf, header, footer)
            if issues:
                for i in issues[:60]:
                    print("  LAYOUT:", i)
                raise SystemExit(f"{poc}: {len(issues)} layout issue(s); fix them before shipping the PDF")
            heads = [(a, hd) for lvl, a, t, hd in gp.toc]
            new_pages, npages = pdf_pages(out_pdf, heads)
            if new_pages == pages:
                break
            pages = new_pages
        miss = [a for a, _ in heads if a not in (pages or {})]
        if miss:
            print(f"  note: no page number found for {len(miss)} TOC entries: {miss[:6]}")
        # clean bookmarks: one per contents entry (the cover is page 1, the contents page 2)
        fix_outline(out_pdf, [(1, "Cover", 1), (1, "Contents", 2)] + [(lvl, txt, (pages or {}).get(a)) for lvl, a, txt, _ in gp.toc])
    # the screen guide gets the same page numbers in its contents
    out_html.write_text(Guide(b, rel).build(pages))
    print(f"{poc}: {out_pdf} ({npages} pages, {out_pdf.stat().st_size / 1e6:.1f} MB)")
    return npages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pocs", nargs="*", default=["poc1", "poc2"])
    ap.add_argument("--no-build", action="store_true", help="do not rebuild the twin HTML")
    ap.add_argument("--no-render", action="store_true", help="reuse the existing renders")
    ap.add_argument("--no-pdf", action="store_true", help="HTML guide only")
    a = ap.parse_args()
    for p in a.pocs:
        if not (TWIN / f"{p}_scene.json").exists():
            raise SystemExit(f"missing out/twin/{p}_scene.json: run 'python3 -B twin/twin_model.py' first")
    if not a.no_build:
        build_viewer(a.pocs)
    if not a.no_render:
        render(a.pocs)
    res = {}
    for p in a.pocs:
        res[p] = build_guide(p, not a.no_pdf)
    for p, n in res.items():
        if n:
            print(f"{p}: {n} pages")


if __name__ == "__main__":
    main()
