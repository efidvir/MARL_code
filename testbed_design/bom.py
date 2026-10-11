"""Bill of materials for the physical testbed, per configuration.

The testbed is fully conducted (galvanic): no radio radiates.  Microwave
(IP-50) hops run through waveguide-coax transitions, fixed pads and one
programmable attenuator per link.  The 60 GHz (TG) radios are cabled
through their RF ports into a V-band RF mesh: each used sector port feeds a
splitter/combiner, and every possible TG link (the pre-event hub links, the
links on spare sectors and every feasible MARL bridge) has its own
programmable attenuator.  The attenuators set each link's loss from the
geometry, cut the paths to the destroyed sites at the event, and keep every
bridge candidate cabled, so a MARL bridge is a path brought up, not a cable
moved.

Quantities are derived from the topology spec and the validator's bridge
candidates, so they stay in step with the figures.
Run:  python3 bom.py   -> out/hardware.md
"""
import os
from collections import Counter

from topologies import CONFIGS
from validate import analyze

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rf_mesh(r):
    """Every TG path the mesh must carry, the sector ports it uses and their fan-out."""
    links = {frozenset(e) for e in r["tg_all"]}
    links |= {frozenset((c["a"], c["b"])) for c in r["cands"] if c["feasible"]}
    fan = Counter()
    for e in links:
        a, b = tuple(e)
        fan[r["end_sector"](a, b)] += 1
        fan[r["end_sector"](b, a)] += 1
    return dict(links=sorted(tuple(sorted(e)) for e in links), ports=fan,
                splitters=sum(1 for v in fan.values() if v >= 2))


def counts(cfg, r=None):
    r = r or analyze([k for k, v in CONFIGS.items() if v is cfg or v["name"] == cfg["name"]][0])
    n_sites = len(cfg["sites"])
    n_mw = len(cfg["mw"])
    n_ip = 2 * n_mw
    n_dn = sum(1 for v in cfg["tg_units"].values() if v == "DN")
    n_cn = sum(1 for v in cfg["tg_units"].values() if v == "CN")
    n_tg = n_dn + n_cn
    agents = sum(1 for s in cfg["tg_units"] if s not in cfg["destroyed"])
    endpoints = len({x for _, a, b in cfg["flows"] for x in (a, b)}
                    | {x for a, b in cfg.get("service", []) for x in (a, b)})
    m = rf_mesh(r)
    return dict(n_sites=n_sites, n_mw=n_mw, n_ip=n_ip, n_dn=n_dn, n_cn=n_cn, n_tg=n_tg,
                agents=agents, endpoints=endpoints, tg_paths=len(m["links"]),
                tg_ports=len(m["ports"]), splitters=m["splitters"],
                tg_jumpers=2 * len(m["links"]) + len(m["ports"]))


# (number, group, item, quantity rule, note)
ITEMS = [
    (1, "Microwave", "Ceragon IP-50 radio unit (e.g. IP-50C)", lambda c: c["n_ip"],
     "one per microwave link end"),
    (2, "Microwave", "Waveguide-to-coax transition, IP-50 band", lambda c: c["n_ip"],
     "conducted: replaces the dish"),
    (3, "Microwave", "Fixed attenuator pad, ~30 dB, IP-50 band", lambda c: 2 * c["n_mw"],
     "two per link: receiver protection, nominal hop loss"),
    (4, "Microwave", "Programmable attenuator, 0-95 dB, IP-50 band", lambda c: c["n_mw"],
     "one per link: fades for DPIR, cut at the event"),
    (5, "Microwave", "Low-loss coax jumper, IP-50 band", lambda c: 2 * c["n_mw"], ""),
    (6, "60 GHz", "Siklu TG node, 4 x 90 deg (N366/N367 class), RF-port build", lambda c: c["n_dn"],
     "hub, core spur and every steerable west/east site"),
    (7, "60 GHz", "Siklu TG terminal (T265 class), RF-port build", lambda c: c["n_cn"],
     "shelter / field-hospital pockets"),
    (8, "60 GHz", "V-band programmable attenuator (one per possible TG link)", lambda c: c["tg_paths"],
     "hub links, spare-sector links and every feasible bridge"),
    (9, "60 GHz", "V-band splitter / combiner", lambda c: c["splitters"],
     "one per sector port that reaches more than one peer"),
    (10, "60 GHz", "V-band fixed pad (one per used sector port)", lambda c: c["tg_ports"], ""),
    (11, "60 GHz", "V-band waveguide / mm-wave cable jumpers", lambda c: c["tg_jumpers"],
     "two per path + one per port"),
    (12, "Power", "PoE injector for TG unit (rating per datasheet)", lambda c: c["n_tg"], ""),
    (13, "Power", "Switched, metered PDU (per-outlet, network-controlled)", lambda c: c["n_sites"],
     "switches CORE and HUB off at t0; energy per site"),
    (14, "Network", "Site router / L3 switch, per-flow policy routing, SFP+", lambda c: c["n_sites"],
     "pins DPIR paths (VLAN or SR steering)"),
    (15, "Compute", "Edge agent PC per surviving TG site", lambda c: c["agents"],
     "frozen MARL actor, local TG control, mesh-attenuator control, logs"),
    (16, "Compute", "Traffic generator / sink host (iperf3, TRex)", lambda c: c["endpoints"],
     "at every flow and service endpoint"),
    (17, "Compute", "Servers: DPIR controller + predictor (GPU); TG E2E controller + NMS", lambda c: 2,
     "the TG controller sits at CORE and is lost with it"),
    (18, "Network", "Out-of-band switch (logging only)", lambda c: 1, "never a control path during the event"),
    (19, "Network", "PTP / GNSS time server", lambda c: 1, ""),
    (20, "Power", "-48 VDC supply + distribution for IP-50", lambda c: 1, "lab bench supply"),
    (21, "Cabling", "SFP+ 10G modules / LC fibre patch cords", lambda c: f"{2 * c['n_tg']} / {c['n_tg']}",
     "TG data uplinks"),
    (22, "Cabling", "Cat6A cables (PoE + data), approx.",
     lambda c: c["n_ip"] + c["n_tg"] + 2 * c["agents"] + c["endpoints"] + c["n_sites"], ""),
    (23, "Test", "Microwave + V-band power meters (calibration)", lambda c: 1, "one set"),
]


def table():
    keys = list(CONFIGS)
    cs = {k: counts(CONFIGS[k], analyze(k)) for k in keys}
    rows = [(n, group, item, [rule(cs[k]) for k in keys], note) for n, group, item, rule, note in ITEMS]
    return keys, cs, rows


def main():
    keys, cs, rows = table()
    head = ("| # | Group | Item | " + " | ".join(f"{k} ({CONFIGS[k]['ip50']}+{CONFIGS[k]['tg']})" for k in keys)
            + " | Note |")
    lines = ["# Testbed hardware (generated by bom.py)", "",
             "Fully conducted (galvanic) bench: no radio radiates. Quantities follow the topology spec and",
             "the validator's bridge candidates.", "", head, "|" + "---|" * (len(keys) + 4)]
    for n, group, item, q, note in rows:
        lines.append(f"| {n} | {group} | {item} | " + " | ".join(str(x) for x in q) + f" | {note} |")
    lines += ["", "## V-band RF mesh per option", ""]
    for k in keys:
        m = rf_mesh(analyze(k))
        lines.append(f"- **{k}**: {len(m['links'])} paths ({', '.join('–'.join(e) for e in m['links'])}); "
                     f"{len(m['ports'])} sector ports, {m['splitters']} with a splitter.")
    lines += ["", "How the bench emulates the radio layer: each programmable attenuator holds its path at the",
              "loss the geometry gives (the figures' link budget); at t0 the PDUs switch CORE and HUB off and",
              "every path to them goes to maximum attenuation; a MARL bridge is a cabled candidate path that",
              "the engaged site brings up. Confirm with Siklu that the TG units are available with RF test",
              "ports, and the port type (waveguide or connector) the V-band parts must match."]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "hardware.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
