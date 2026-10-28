"""Option A without steering emulation: sectors, RF paths, attenuators, ports.

The TG radios point their beams themselves (beamforming at link ignition), so
the bench does not have to reproduce steering towards every bridge candidate.
It only cables the final plan: the pre-event TG links plus the three bridges
MARL ends up using (E2-W2, E1-N, W1-S).  What remains for the bench to do is
take links down: programmable attenuators cut every path to CORE and HUB at t0
(the islands) and hold each bridge off-beam until its sites engage.

Per node this script lists DN/CN, the sectors in use before the event and in
the final state, and the polarity of each sector; then the RF paths with their
attenuator settings, the RF junctions and cables, and the Ethernet port map of
each site switch (which radio, and so which link, sits on which port).

Run:  python3 option_a_sectors.py     -> out/option_A_sectors.md
"""
import os
from collections import Counter

from option_a_plan import (ATT_MAX, JUMPER_DB, MW_ATT_NOMINAL, MW_PAD_DB, SIDELOBE_PER_END,
                           VPAD_DB, air_equivalent_loss, splitter_loss)
from topologies import ROLE
from validate import analyze, bearing, dist

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SEC = ["E", "N", "W", "S"]
SECTORS_PER_NODE = 4


def sec_name(r, s, k):
    return "single" if r["cfg"]["tg_units"][s] == "CN" else SEC[k]


def plan():
    r = analyze("A")
    cfg, pos = r["cfg"], r["pos"]
    units = cfg["tg_units"]
    dead = set(cfg["destroyed"])
    pre = [tuple(e) for e in r["tg_all"]]
    bridges = [tuple(b) for b in cfg["bridges"]]
    links = pre + bridges

    # sector use per node, before and after
    use = {s: {"before": {}, "after": {}} for s in units}
    for a, b in links:
        for x, y in ((a, b), (b, a)):
            k = r["end_sector"](x, y)[1]
            if (a, b) in pre:
                use[x]["before"].setdefault(k, []).append(y)
            if (a, b) in bridges or not ({a, b} & dead):
                use[x]["after"].setdefault(k, []).append(y)
    nodes = []
    for s, kind in units.items():
        ks = sorted(set(use[s]["before"]) | set(use[s]["after"]))
        nodes.append(dict(site=s, kind=kind, role=ROLE[s].split("\n")[0], sectors=ks,
                          before=use[s]["before"], after={} if s in dead else use[s]["after"],
                          polarity={k: r["polarity"].get((s, k), "–") for k in ks},
                          bearings={y: bearing(pos[s], pos[y]) for k in ks
                                    for y in use[s]["before"].get(k, []) + use[s]["after"].get(k, [])}))

    # RF: one path per link of the final plan; a sector port reaching two peers gets a 2-way splitter
    fan = Counter()
    for a, b in links:
        fan[r["end_sector"](a, b)] += 1
        fan[r["end_sector"](b, a)] += 1
    paths = []
    for a, b in links:
        pa, pb = r["end_sector"](a, b), r["end_sector"](b, a)
        d = dist(pos[a], pos[b])
        target = air_equivalent_loss(d)
        fixed = 2 * VPAD_DB + splitter_loss(fan[pa]) + splitter_loss(fan[pb]) + JUMPER_DB
        on = target - fixed
        bridge = (a, b) in bridges
        before = on + 2 * SIDELOBE_PER_END if bridge else on
        at_t0 = ATT_MAX if {a, b} & dead else before
        after = on if bridge else at_t0
        paths.append(dict(a=a, b=b, pa=pa, pb=pb, d=d, target=target, on=on, before=before,
                          at_t0=at_t0, after=after,
                          job="bridge: off-beam until engaged" if bridge else
                          "taken down at t0 (island)" if {a, b} & dead else "steady"))
    junctions = sorted(p for p, k in fan.items() if k > 1)
    ports = sorted(fan)
    rf_cables = 2 * len(paths) + len(junctions)

    # microwave hops
    mw = [dict(a=a, b=b, d=dist(pos[a], pos[b]), cut=bool({a, b} & dead)) for a, b in cfg["mw"]]

    # Ethernet: one switch per site with wired kit; each radio port named by the link it serves
    endpoints = {x for _, s, t in cfg["flows"] for x in (s, t)} | \
        {x for s, t in cfg.get("service", []) for x in (s, t)}
    sw = {}
    for s in cfg["sites"]:
        p = []
        for a, b in cfg["mw"]:
            if s in (a, b):
                p.append(("IP-50", f"microwave link {s}–{b if a == s else a}"))
        if s in units:
            served = []
            for k in sorted(set(use[s]["before"]) | set(use[s]["after"])):
                peers = []
                for y in use[s]["before"].get(k, []):
                    peers.append(y)
                for y in use[s]["after"].get(k, []):
                    if y not in peers:
                        peers.append(y + " (bridge)" if (s, y) in bridges or (y, s) in bridges else y)
                served.append(f"{sec_name(r, s, k)}: " + ", ".join(peers))
            p.append((f"TG {units[s]}", "TG links " + "; ".join(served)))
        if s == "CORE":
            p += [("TG controller", "Terragraph controller (lost at t0)"),
                  ("DPIR server", "DPIR planner (lost at t0)")]
        if s in units and s not in dead:
            p.append(("Agent PC", "MARL agent + site routing"))
        if s in endpoints:
            p.append(("Traffic host", "flow / service endpoint"))
        sw[s] = p
    return r, nodes, paths, junctions, ports, rf_cables, mw, sw


def main():
    r, nodes, paths, junctions, ports, rf_cables, mw, sw = plan()
    cfg = r["cfg"]
    dead = set(cfg["destroyed"])
    n_dn = sum(1 for n in nodes if n["kind"] == "DN")
    n_cn = sum(1 for n in nodes if n["kind"] == "CN")
    used_dn = sum(len(n["sectors"]) for n in nodes if n["kind"] == "DN")
    used_cn = sum(len(n["sectors"]) for n in nodes if n["kind"] == "CN")

    def fmt(d, s):
        if not d:
            return "–"
        return "; ".join(f"{sec_name(r, s, k)} → {', '.join(v)}" for k, v in sorted(d.items()))

    L = ["# Option A: sectors, attenuators and ports (generated by option_a_sectors.py)", "",
         "No steering emulation: the TG radios point their own beams, so only the final plan is "
         "cabled (the pre-event TG links plus the bridges E2–W2, E1–N, W1–S). Programmable "
         "attenuators take links down: every path to CORE and HUB at t0, and each bridge stays "
         "off-beam until its sites engage.", "",
         "## Sectors per node", "",
         "| Node | Type | Sectors used | Before the event | Final (after recovery) | Polarity |",
         "|---|---|---|---|---|---|"]
    for n in nodes:
        s = n["site"]
        used = ", ".join(sec_name(r, s, k) for k in n["sectors"])
        after = "off (destroyed)" if s in dead else fmt(n["after"], s)
        pol = ", ".join(f"{sec_name(r, s, k)} {v}" for k, v in n["polarity"].items())
        L.append(f"| {s} | {n['kind']} | {len(n['sectors'])} ({used}) | {fmt(n['before'], s)} | "
                 f"{after} | {pol} |")
    L += ["", f"Sectors in use: {used_dn} on the {n_dn} nodes (of {SECTORS_PER_NODE * n_dn} available) "
          f"and {used_cn} on the {n_cn} terminals: {used_dn + used_cn} in all. "
          f"{SECTORS_PER_NODE * n_dn - used_dn} node sectors stay unused (disable them; terminate their "
          "ports if they are RF-port builds).", "",
          "## V-band paths and programmable attenuators (dB)", "",
          "| # | Path | Sector ports | Distance | Job | Before | At t0 | After |",
          "|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(paths, 1):
        L.append(f"| {i} | {p['a']}–{p['b']} | {p['a']}-{sec_name(r, *p['pa'])} ↔ "
                 f"{p['b']}-{sec_name(r, *p['pb'])} | {p['d']:.0f} m | {p['job']} | {p['before']:.1f} | "
                 f"{p['at_t0']:.1f} | {p['after']:.1f} |")
    L += ["", f"Chain per path: {VPAD_DB:.0f} dB pad on each sector port, a 2-way splitter where a port "
          f"reaches two peers, {JUMPER_DB:.0f} dB of jumpers (assumed), one programmable attenuator. "
          f"Cut = {ATT_MAX:.0f} dB.", "",
          f"Sector ports reaching two peers (2-way splitter each): "
          + ", ".join(f"{s}-{sec_name(r, s, k)}" for s, k in junctions) + ".", "",
          f"V-band parts: {len(paths)} programmable attenuators, {len(junctions)} splitters, "
          f"{len(ports)} pads (one per used sector port), {rf_cables} cables "
          f"(two per path, one per splitter).", "",
          "## Microwave hops", "",
          "| Hop | Chain | At rest | At t0 |", "|---|---|---|---|"]
    for m in mw:
        L.append(f"| {m['a']}–{m['b']} | transition → {MW_PAD_DB:.0f} dB pad → programmable attenuator → "
                 f"{MW_PAD_DB:.0f} dB pad → transition | {MW_ATT_NOMINAL:.0f} dB (+ fades) | "
                 f"{'95 dB: taken down' if m['cut'] else 'unchanged (can be taken down for other islands)'} |")
    n_ports = {s: len(p) for s, p in sw.items()}
    wired = [s for s in cfg["sites"] if s != "HUB"]
    L += ["", "## Ethernet: site switches and what each port carries", ""]
    for s in cfg["sites"]:
        if s == "HUB":
            L += [f"**HUB**: no switch. The TG node's port only takes PoE power; it carries the hub links "
                  f"over the air interface and must not be wired to anything else.", ""]
            continue
        L += [f"**{s}** ({n_ports[s]} ports)", ""]
        for i, (dev, what) in enumerate(sw[s], 1):
            L.append(f"{i}. {dev}: {what}")
        L.append("")
    L += [f"Site switches: {len(wired)} ({', '.join(f'{s} {n_ports[s]}' for s in wired)} ports; "
          f"{sum(n_ports[s] for s in wired)} in all), plus one out-of-band management switch for the "
          "agent PCs' second NICs, the attenuator controllers, the PDUs and logging. One managed switch "
          "with a port-based VLAN per site also works, provided inter-VLAN routing stays off: the "
          "sites may only meet through the radio links."]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "option_A_sectors.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
