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
        self.legend = {l["cls"]: l for l in s["legend"]}
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
            by[face][p["type"]] += 1
        parts = []
        for face in ("front", "rear", "top", "side", "bottom"):
            if by[face]:
                parts.append(f"{face}: " + ", ".join(f"{n} {t}" for t, n in sorted(by[face].items(), key=lambda kv: -kv[1])))
        return "; ".join(parts)

    def below(self, d):
        """device that d sits on (bench table stacks)."""
        if d["group"] != "table" or d.get("parent"):
            return None
        for o in self.scene["devices"]:
            if o is d or o["group"] != "table" or o.get("parent"):
                continue
            if abs(o["pos"][1] + o["size"][1] - d["pos"][1]) > 2:
                continue
            ox = min(d["pos"][0] + d["size"][0], o["pos"][0] + o["size"][0]) - max(d["pos"][0], o["pos"][0])
            oz = min(d["pos"][2] + d["size"][2], o["pos"][2] + o["size"][2]) - max(d["pos"][2], o["pos"][2])
            if ox > 10 and oz > 10:
                return o
        return None

    def where(self, d):
        T = self.T
        w, h, dp = (round(v) for v in d["size"])
        if d["group"] == "rack":
            if not d.get("parent"):
                return f"{d.get('u') or '?'} · {w} × {h} mm face, {dp} mm deep"
            par = self.D[d["parent"]]
            return (f"on {par['id']} ({par.get('u')}): {round(d['pos'][0])} mm from the left ear, "
                    f"{round(d['pos'][2])} mm behind the front rail")
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
            if d and d["kind"] == "splitter" and name in ("P1", "P2"):
                toks.append(("dev", d["id"], f"{name}→S"))
                pid = f"{d['id']}.S"
                continue
            if d and d["kind"] == "coupler" and name in ("THRU", "CPL"):
                toks.append(("dev", d["id"], "THRU→IN" if name == "THRU" else "CPL (samples IN)"))
                pid = f"{d['id']}.IN"
                continue
            if d and d["kind"] == "transition":
                other = [q for q in d["ports"] if q != o]
                if other:
                    toks.append(("dev", d["id"], f"{name}→{self.P[other[0]]['name']}"))
                    pid = other[0]
                    continue
            return toks, o
        return toks, pid


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
"""


def chip(conf):
    c = conf or "estimate"
    return f'<span class="chip {esc(c)}">{esc(c)}</span>'


def swatch(color):
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
        return self.table([("ID", ""), ("What it is / part", ""), ("Where", ""), ("Mounting notes", ""), ("Conf.", "")],
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
<div class="cols">
 <div style="flex:1.25">
  <h3>Cable colours and ID scheme</h3>
  {legend}
  <p class="small">Cable IDs are a class letter and a number: <b>A</b> = AC, <b>D</b> = DC, <b>P</b> = PoE, <b>F</b> = data, <b>E</b> = management,
   <b>R</b> = IF coax, <b>G</b> = GNSS, <b>W</b> = IP-50 chain, <b>X</b> = operator station. Port IDs are <span class="mono">DEVICE.PORT</span>:
   <span class="mono">ATT-01.ch1.in</span> is attenuator ATT-01, channel 1 input; <span class="mono">SB2.M1.T1</span> is Spider box SB2, module M1, tile 1.</p>
  <h3>Confidence of each item</h3>
  <p class="small">{chip('estimate')} {esc(CONF_TEXT['estimate'])}. &nbsp; {chip('likely')} {esc(CONF_TEXT['likely'])}. &nbsp;
   {chip('confirmed')} {esc(CONF_TEXT['confirmed'])}.</p>
 </div>
 <div style="flex:1">
  <h3>Rules for every step</h3>
  <ol>
   <li><b>Tag both ends of every cable</b> with its ID before you route it. Use the label printer cartridges in the parts list.</li>
   <li><b>Torque every SMA to 8 in-lb</b> (0.9 N·m) with the SMA torque wrench. Finger-tight is not enough and more damages the connector.</li>
   <li><b>Rack units count from the bottom</b>: U1 is the lowest unit. A device marked U33-U34 takes two units. The U numbers are printed on the front-left post in the pictures.</li>
   <li><b>Front</b> is the operator side (Spider-box and attenuator SMA panels). <b>Rear</b> carries the PDU outlets, DC terminals and the node ETH / SFP ports. The words FRONT and REAR are painted on the floor in the pictures.</li>
   <li><b>Length</b> is the stock length to fit; "route" is the length the twin measured along the cable path. Dress the slack in the cable manager.</li>
   {"<li><b>Terminate unused tile and coupler SMA ports</b> with 50-ohm loads (ANNE-50+).</li>" if loads else ""}
   <li><b>Power stays off</b> until the last step: PDUs unplugged, breakers and fuses OFF.</li>
  </ol>
  <h3>How the pictures work</h3>
  <ul class="small">
   <li>Each step picture shows the bench as it is after that step. Items fitted in this step are bright with an orange outline and an ID tag; items from earlier steps are pale; later items are not shown.</li>
   <li>Cables are drawn in their class colour along the route the twin planned. Small labels give the cable ID.</li>
   <li>Close-ups of the Spider boxes are straight on: module (M1-M4), sector (e.g. HUB.W) and tile (T1-T4) are labelled.</li>
   <li>Every picture carries the TENTATIVE mark: the scene is a planning model, not a construction drawing.</li>
  </ul>
 </div>
</div>"""
        self.section("read", "How to read this guide", body, num="1")

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
        utab = self.table([("U", ""), ("Device", ""), ("Part", "")], rows, widths=[15, 27, 58], cls="tight")
        r = b.renders["overview_front_rack"]
        asp = r["px"]["w"] / r["px"]["h"]
        img_h = 160
        img_w = min(img_h * asp, 150)
        body = f"""
<div class="cols">
 <div style="flex:0 0 {img_w + 2:.0f}mm">{self.fig('overview_front_rack', img_w, img_h, caption='<b>Front elevation</b>, U numbers on the left post')}</div>
 <div style="flex:1"><p class="small">Read from the top. Every U is listed; consecutive blanking panels share a row.
  Rack: {esc(b.R['part'])}, {b.R['units']}U, rail-to-rail depth {round(b.R['rail_depth'])} mm; U1 starts {round(b.R['u1_y'])} mm above the floor.</p>{utab}</div>
</div>"""
        self.section("rack-front", "Rack front elevation", body, level=2, toc_text="Rack front elevation (U by U)")
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
            rr.append([esc(d["u"]), f'<span class="cid">{esc(d["id"])}</span>', esc(", ".join(f"{n} {t}" for t, n in types.most_common())), f"{used} / {len(rear)}"])
        rtab = self.table([("U", ""), ("Device", ""), ("Rear connectors", ""), ("Used", "n")], rr, widths=[15, 27, 42, 13], cls="tight")
        r = b.renders["overview_rear_rack"]
        asp = r["px"]["w"] / r["px"]["h"]
        img_w = min(img_h * asp, 150)
        body = f"""
<div class="cols">
 <div style="flex:0 0 {img_w + 2:.0f}mm">{self.fig('overview_rear_rack', img_w, img_h, caption='<b>Rear</b>, seen from behind the rack (left and right swap)')}</div>
 <div style="flex:1"><p class="small">Most cabling happens at the rear: PDU outlets, DC terminals, the node power (ETH1),
  data (SFP+) and GNSS ports. Connectors that face the rear, per rack item (counting the parts on its shelf):</p>{rtab}</div>
</div>"""
        self.section("rack-rear", "Rack rear", body, level=2, toc_text="Rack rear (connectors)")
        # ---- bench table plan
        self.section("table-plan", "Bench table layout", self.table_plan(), level=2, toc_text="Bench table layout (from above)")
        body = self.fig("overview_table", PAGE_W, 160, caption="<b>Bench table</b> seen from the operator side, above. Labels give the device IDs.")
        self.section("table-view", "Bench table, 3D view", body, level=2)

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
                g.append(f'<rect x="{sx(x):.0f}" y="{sy(z + dp):.0f}" width="{w:.0f}" height="{dp:.0f}" fill="#ffffff" stroke="#c2185b" stroke-width="5"/>')
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
            g.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="{"#c2185b" if small else "#1f2933"}" stroke="#fff" stroke-width="6"/>')
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
        tab = self.table([("#", "n"), ("Group", ""), ("Item", ""), ("Part number", ""), ("Maker", ""), ("Qty", ""), ("Use", ""), ("Status", "")],
                         rows, widths=[3.5, 7, 26, 17, 10, 8, 18, 10.5], cls="tight")
        body = (f'<p class="lead">From <span class="mono">{esc(self.b.poc)}_parts_list.csv</span>. Quantities include spares where noted. '
                f'"Status" is how sure the part is; check every "to confirm" line before ordering.</p>{tab}')
        self.section("bom", "Bill of materials", body, num="3")

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
            chips += f' <span class="chip" style="background:{esc(b.legend.get(cls, {}).get("color", "#999"))};color:#fff">{esc(cls)}</span>'
        head = (f'<div class="stephead"><div class="stepno">{n}</div><div><h2 id="step-{n}">{esc(st["title"])}</h2>'
                f'<div class="sub">Step {n} of {len(b.steps)} · {esc(view_txt)}</div></div><div class="chips">{chips}</div></div>')
        extra = self.step_extra_text(st, own, shared)
        text = f'<div class="steptext">{esc(st["text"])}{extra}</div>'
        self.toc.append((2, f"step-{n}", f"Step {n} · {st['title']}", st["title"]))
        parts = [head, text]
        is_link = kind in ("IF link", "coupling arm")
        # the main block first: its estimated height decides the picture size
        main, est, after = "", 0.0, ""
        self.table_w = PAGE_W
        rack = f"step_{n:02d}_rack"
        if kind == "devices":
            tab = self.device_table(st["devices"])
            main, est = f'<h3>Fit these {len(st["devices"])} items</h3>{tab}', self.last_est + 9
            sbs = [d for d in st["devices"] if b.D[d]["kind"] == "spider"]
            if sbs:
                after = self.spider_panels(sbs)
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
            if rack in b.renders:
                r = b.renders[rack]
                asp = r["px"]["w"] / r["px"]["h"]
                ih = 168
                iw = min(125, ih * asp)
                after = (f'<div class="newpage cols"><div style="flex:0 0 {iw + 2:.0f}mm">{self.fig(rack, iw, ih, caption="<b>Rack part</b> of this step, " + esc(st["view"]) + " view")}</div>'
                         f'<div style="flex:1">{self.class_notes(st)}</div></div>')
                main, after = after, f'<div class="newpage">{main}</div>'
                est = 999
        elif kind == "checks":
            main, est = '<p>The install checks, the power-up order and the event procedure are in <a href="#checks">section 6</a>.</p>', 8
        # page 1: heading (15 mm) + text box + picture (+ caption 7 mm) [+ the main block if it fits]
        t1 = 1 + len(st["text"]) // 140
        t2 = 1 + len(html.unescape(re.sub(r"<[^>]+>", "", self.step_extra_text(st, own, shared)))) // 155
        avail = PAGE_H - 15 - (6.5 + 5.2 * t1 + 4.7 * t2) - 8 - 4
        if est and est + 95 <= avail:
            fig_h, same = min(145, avail - est), True
        else:
            fig_h, same = min(145, avail), False
        cap = f"<b>After step {n}</b>: items of this step highlighted (orange outline); earlier items pale."
        parts.append(self.fig(f"step_{n:02d}", PAGE_W, fig_h, caption=cap))
        if main:
            parts.append(main if same or main.startswith('<div class="newpage') else f'<div class="newpage">{main}</div>')
        parts.append(after)
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
                        s.append(f'<span class="mono">{esc(t[1])}</span> <span class="muted">({esc(t[2])})</span>')
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
                         rows, widths=[9, 17, 24, 11, 24, 17])
        fig = self.fig("ip50_chains", PAGE_W, 112, caption="<b>IP-50 chains</b> on the bench table: radios, waveguide transitions (on the radio flanges) and the MWATT channels")
        return (f'<div class="newpage"><h3>Chains: radio → transition → MWATT channel → transition → radio</h3>{tab}{fig}</div>'
                f'<div><h3>Connect these {len(own)} cables</h3>{self.cable_table(own)}</div>')

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
            pre = self.fig("data_mgmt_rack", 72, 160, caption="<b>Data and management</b> at the rack rear") if dev == "SW-DATA" else ""
            tab = self.table([(esc(h), "") for h in hdr], out, cls="tight")
            if pre:
                html_ = f'<div class="cols"><div style="flex:0 0 74mm">{pre}</div><div style="flex:1">{lead}{tab}</div></div>'
            else:
                html_ = lead + tab
            self.section(anchor, title, html_, level=2)
        self.section("pdu-map", "PDU outlet map", self.pdu_map(), level=2)
        self.section("dc-map", "DC breaker and fuse map", self.dc_map(), level=2)

    def pdu_map(self):
        b = self.b
        pdus = [d for d in b.D.values() if d["kind"] in ("pdu", "strip")]
        sw = set(b.meta.get("switched") or [])
        blocks = []
        for d in pdus:
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
            tab = self.table([("Outlet", ""), ("Feeds", ""), ("Cable", ""), ("Lead", "")], rows, widths=[10, 44, 17, 12], cls="tight")
            blocks.append(f'<div class="keep" style="flex:1"><h3>{esc(d["id"])} <span class="muted" style="font-weight:400">{esc(d.get("u") or "table")}</span></h3>'
                          f'<p class="small muted">{esc(d.get("sub") or "")}; {feed}</p>{tab}</div>')
        # 2 per row
        rows_html = []
        for i in range(0, len(blocks), 2):
            rows_html.append(f'<div class="cols">{"".join(blocks[i:i + 2])}{"<div style=flex:1></div>" if len(blocks[i:i + 2]) == 1 else ""}</div>')
        lead = ('<p class="small">Outlet → device → cable. Outlets marked <span class="chip off">OFF at t0</span> are switched off by the bench '
                'controller at the event; keep them on the switched PDU.</p>' if sw else '<p class="small">Outlet → device → cable.</p>')
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
            for p in d.get("ports") or []:
                c = b.portcable.get(p)
                if c and c["cls"] in ("POE", "DC"):
                    dark.append(f'<span class="mono">{esc(b.other_end(c, p))}</span> (<span class="cid">{esc(c["id"])}</span>)')
            rows.append([f'<span class="cid">{esc(did)}</span>', esc(b.what(d)), "<br>".join(feeds) or "-", "<br>".join(dark) or "itself"])
        tab = self.table([("Switched device", ""), ("What it is", ""), ("PDU outlet", ""), ("Also goes dark", "")], rows, widths=[22, 26, 26, 26])
        fig = self.fig("event_t0", 160, 112, caption="<b>After t0</b>: switched devices dark with a red outline and an OFF tag; their cables grey")
        body = f"""
<div class="box warn"><b>Event at t0.</b> {esc(b.meta['event'])}</div>
<div class="cols" style="margin-top:3mm">
 <div style="flex:1"><h3>Outlets to switch OFF at t0</h3>{tab}
  <ol class="small"><li>Before t0: every link up, attenuators at their start values, logging running on BENCH-CTL.</li>
  <li>At t0: switch the outlets above OFF together from the bench controller (PDUs on SW-OOB). Do not touch any attenuator at t0.</li>
  <li>After the run: switch the same outlets ON again and let the nodes and radios re-join before the next run.</li></ol></div>
 <div style="flex:0 0 162mm">{fig}</div>
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
        missing = sorted(want - set(self.cable_rows), key=natkey)
        twice = sorted([c for c, n in self.cable_rows.items() if n > 1], key=natkey)
        extra = sorted(set(self.cable_rows) - want, key=natkey)
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
