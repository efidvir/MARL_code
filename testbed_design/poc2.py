"""PoC 2: DPIR on an all-fixed, all-point-to-point Terragraph ring (no MARL, no event).

PoC 1 (A7) shows DPIR + MARL recovery on 15 radio units.  PoC 2 keeps only what DPIR needs:
a ring of four fixed DN-DN point-to-point Terragraph links between the four 2-sector Siklu
N367 nodes of the two GTS003 Spider boxes (4 radio units, no IP-50, no hub, no PtMP).  Every
sector has exactly one peer.  Two cabled co-channel couplings (Terragraph's own "one link in
between" interference heuristic, emulated on the bench) join the west ends and the east ends of
the two parallel spans, so the spans interfere only while both carry traffic: routing can
switch the interference off, which is what DPIR exploits.

This script is the single source of the PoC 2 numbers.  It
  1. checks the design: sectors, one peer per sector, polarity, Spider-box polarity classes,
     Golay pattern, coupling polarity and leak paths;
  2. builds DPIR's candidates with the repo's diverse_paths() and checks they are every simple
     path (DPIR's candidate set is complete on the ring);
  3. evaluates every joint routing under a Terragraph TDD capacity model and compares DPIR's
     target with hop-count routing and with Terragraph's own MCS-based Open/R routing;
  4. runs the robustness cases: INR sweep, time-share bound, one coupling only, the side-hop
     fade at which DPIR goes back to the direct span, couplings off, an IP-50 side hop;
  5. models the worst case where idle links still transmit (QoS-Null), and DPIR's fallback of
     taking unused coupled links down (with a re-ignition delay);
  6. runs dynamic traces (bursty traffic) for predictive and reactive routers;
  7. writes out/poc2_results.json, out/poc2_plan.md and out/poc2_hardware.md.

Run:  python3 poc2.py        (exits non-zero if a check fails)
"""
import itertools
import json
import math
import os
import sys

import networkx as nx

from topologies import CONFIGS
from validate import MAX_STRETCH, angdiff, bearing, diverse_paths, fits_budget, sector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SEC = {0: "E", 1: "N", 2: "W", 3: "S"}

# --------------------------------------------------------------------------------------------
# 1. The design
# --------------------------------------------------------------------------------------------
# Schematic positions (drawing units).  The spans W1-E1 and E2-W2 are the short, high-grade
# links; the side hops W1-W2 and E1-E2 close the ring at a lower grade.
POS = {"W1": (0, 450), "E1": (600, 450), "E2": (600, -450), "W2": (0, -450)}

# site: (Spider box, rack slot, node in the box, sectors on the unit, polarity of both sectors)
# The ring is even, so one polarity per node works.  Each GTS003 holds two nodes of one
# polarity class (diagonal corners of the ring), so leakage inside a box joins sectors that
# transmit together and receive together: harmless in data slots.
NODES = {
    "W1": ("GTS003 #1", "U16", "a", 2, "even"),
    "E2": ("GTS003 #1", "U16", "b", 2, "even"),
    "E1": ("GTS003 #2", "U13", "a", 2, "odd"),
    "W2": ("GTS003 #2", "U13", "b", 2, "odd"),
}
# Spider-box module per (site, sector): node a = modules M1, M2; node b = M3, M4 (assumed order)
MODULE = {("W1", "E"): "M1", ("W1", "S"): "M2", ("E2", "W"): "M3", ("E2", "N"): "M4",
          ("E1", "W"): "M1", ("E1", "S"): "M2", ("W2", "E"): "M3", ("W2", "N"): "M4"}

# link: (end a, end b, channel, Golay index, SNR set by its attenuator [dB], attenuator unit, role)
LINKS = {
    "LN": ("W1", "E1", 2, 1, 25.0, "ATT-1", "north span"),
    "LE": ("E1", "E2", 2, 1, 12.0, "ATT-2", "east side hop"),
    "LS": ("E2", "W2", 2, 2, 25.0, "ATT-3", "south span"),
    "LW": ("W2", "W1", 2, 2, 12.0, "ATT-4", "west side hop"),
}
# F2 runs east to west on purpose: both flows then transmit from even sectors (subframe A), so
# each span's data lands in the other span's receive window.  Two same-direction flows on
# parallel spans of an even ring transmit in opposite subframes and cannot interfere.
FLOWS = [("F1", "W1", "E1", 1600.0), ("F2", "E2", "W2", 900.0)]

# cabled couplings: (sector X, sector Y, INR at the victim [dB] at the nominal setting, unit)
INR_NOM = 18.0
COUPLINGS = {
    "KW": (("W1", "E"), ("W2", "E"), INR_NOM, "ATT-5"),   # west ends of the two spans
    "KE": (("E1", "W"), ("E2", "W"), INR_NOM, "ATT-6"),   # east ends of the two spans
}

# Terragraph DMG data MCS: QTI firmware 10.11 default LA thresholds (decoded mcsLqmQ3 from
# meta-terragraph fw_versions/10.11.0.json) and per-direction effective TDD rates (terragraph-ctf
# Link_Adaptation PUMA_RF_LA-2.1).  laMinMcs 2, MCS5 disabled; below MCS2 no data.
MCS = [(4.5, 2, 323), (5.5, 3, 404), (6.5, 4, 485), (7.5, 6, 646), (8.5, 7, 808), (9.5, 8, 969),
       (10.5, 9, 1050), (13.5, 10, 1292), (14.5, 11, 1616), (16.5, 12, 1939)]
RATE_MCS = {r: m for _, m, r in MCS}
NO_TRAFFIC_MAX_MCS = 9       # LA no-traffic mode caps MCS at 9 (noTrafficMaxMcsFallback)
PER_TARGET = 0.005           # LA's 0.5 % PER target: an aggressor hitting more of the victim's
#                              slots than this pushes the victim's LA down to its SINR's MCS
# Idle-link transmit floor (fraction of a direction's data airtime), estimated from the MAC
# frame timing; to be measured on the bench (gated power on a coupled port, idle vs loaded).
IDLE_FLOORS = {"silent": 0.0,
               "management only": (192 - 185) / 200 * (2 / 16),   # 7 us control slot, 2 of 16 SFs
               "QoS-Null every slot": 3 * 2.9 / 200}             # ~2.9 us QoS-Null in 3 TX slots
T_IGN = 5                    # assumed re-ignition time of a link taken down [s] (to be measured)

# Terragraph Open/R MCS-based link metric (meta-terragraph base config RELEASE_M81,
# openrParams.linkMetricConfig, enabled): MCS 2-3 -> 15, 4-8 -> 3, 9-12 -> 1.  Metric changes are
# damped (10 successive reports at ~100 ms, i.e. ~1 s) and rate-limited by a token bucket
# (0.1 token/s, burst 3) per node.  Equal-cost paths share a flow by ECMP.
OPENR_METRIC = [(9, 1), (4, 3), (2, 15)]
OPENR_TOKEN_RATE, OPENR_TOKEN_BURST = 0.1, 3

# Optional variant: the east side hop as an IP-50 link (PoC 1's E1-E2 microwave chain).  ACM
# capacity at 4096QAM, low end of the datasheet range (mw research notes, medium confidence).
IP50_CAP = {"56 MHz": 538.0, "112 MHz": 1034.0}

CHECKS = []


def check(ok, claim, how):
    CHECKS.append(dict(claim=claim, how=how, ok=bool(ok)))
    print(("  PASS  " if ok else "  FAIL  ") + claim)


def mcs_of(snr):
    best = (0, 0.0)
    for thr, m, r in MCS:
        if snr >= thr:
            best = (m, r)
    return best


def rate(snr):
    return mcs_of(snr)[1]


def sinr(snr, inrs):
    return snr - 10 * math.log10(1 + sum(10 ** (i / 10) for i in inrs))


# --------------------------------------------------------------------------------------------
# 2. Graph, sectors, polarity
# --------------------------------------------------------------------------------------------
G = nx.Graph()
for _name, (_a, _b, _ch, _gol, _snr, _att, _role) in LINKS.items():
    G.add_edge(_a, _b, name=_name)
DIRS = [(a, b) for a, b in G.edges] + [(b, a) for a, b in G.edges]


def link_of(a, b=None):
    if b is None:
        a, b = a
    return G.edges[a, b]["name"]


def end_sector(x, y):
    """Sector of unit x used toward y (90-degree sectors from the drawing bearings)."""
    return SEC[sector("DN", bearing(POS[x], POS[y]))]


def tx_sec(d):
    return (d[0], end_sector(d[0], d[1]))


def rx_sec(d):
    return (d[1], end_sector(d[1], d[0]))


POL = {(n, end_sector(n, nb)): NODES[n][4] for n in NODES for nb in G.neighbors(n)}


def subframe(d):
    return "A" if POL[tx_sec(d)] == "even" else "B"


def design_checks():
    print("== sectors and polarity ==")
    peers = {}
    for a, b in G.edges:
        for x, y in ((a, b), (b, a)):
            peers.setdefault((x, end_sector(x, y)), []).append(y)
    check(all(len(v) == 1 for v in peers.values()),
          "every sector in use has exactly one peer (point-to-point): " + ", ".join(
              f"{s[0]}.{s[1]}->{v[0]}" for s, v in sorted(peers.items())),
          "count peers per (site, sector) from the drawing bearings via validate.sector()")
    for n, (box, slot, node, ns, pol) in NODES.items():
        secs = {k for (s, k) in peers if s == n}
        idx = {[i for i, c in SEC.items() if c == k][0] for k in secs}
        check(len(secs) <= ns and fits_budget(idx, ns),
              f"{n}: links on sectors {'+'.join(sorted(secs))} fit its {ns}-sector N367, side by side",
              "validate.fits_budget(sectors, 2)")
    slack = [45 - angdiff(bearing(POS[x], POS[y]), sector("DN", bearing(POS[x], POS[y])) * 90)
             for a, b in G.edges for x, y in ((a, b), (b, a))]
    check(min(slack) >= 5, f"every link end sits {min(slack):.0f} deg inside its sector (>= 5 deg)",
          "45 - |bearing - sector boresight| for every link end")
    sg = nx.Graph()
    for a, b in G.edges:
        sg.add_edge((a, end_sector(a, b)), (b, end_sector(b, a)))
    check(nx.is_bipartite(sg) and all(POL[(a, end_sector(a, b))] != POL[(b, end_sector(b, a))]
                                       for a, b in G.edges),
          "every link joins an odd and an even sector; the sector graph is bipartite",
          "networkx.is_bipartite on (site, sector) vertices + polarity at both ends")
    check(nx.is_bipartite(G) and len(nx.cycle_basis(G)) == 1 and len(G) == 4,
          "the site graph is an even ring (4 nodes), so one polarity per node works: no hybrid node",
          "networkx.is_bipartite on the site graph")
    for box in sorted({v[0] for v in NODES.values()}):
        pols = {NODES[n][4] for n in NODES if NODES[n][0] == box}
        members = [n for n in NODES if NODES[n][0] == box]
        check(len(pols) == 1, f"{box} ({'+'.join(members)}) holds one polarity class ({pols.pop()}): "
              "leakage inside the box is same-polarity, harmless in data slots", "set of polarities per box")
    ring = ["LN", "LE", "LS", "LW"]
    gol = [LINKS[k][3] for k in ring]
    check(all(not (gol[i] == gol[(i + 2) % 4]) for i in range(4)) and len({LINKS[k][2] for k in ring}) == 1,
          f"one channel ({LINKS['LN'][2]}) on all links; Golay {dict(zip(ring, gol))} runs a-a-b around the "
          "ring, so the coupled spans LN and LS have different codes (pure SINR loss, no foreign-preamble "
          "capture)", "Terragraph Topology_Management heuristic 4: no a-b-a")
    return peers


def aggressors(d, couplings):
    """(aggressor direction, INR) pairs: directions on another link whose transmitter is cabled
    to d's receiver at opposite polarity (it transmits while d's receiver listens)."""
    out = []
    y = rx_sec(d)
    for x1, x2, inr, _ in couplings.values():
        for xa, yb in ((x1, x2), (x2, x1)):
            if yb != y or POL[xa] == POL[yb]:
                continue
            out += [(a, inr) for a in DIRS if tx_sec(a) == xa and link_of(a) != link_of(d)]
    return out


def coupling_checks():
    print("== couplings ==")
    for k, (x, y, inr, att) in COUPLINGS.items():
        lx = [link_of(a, b) for a, b in G.edges if x in ((a, end_sector(a, b)), (b, end_sector(b, a)))][0]
        ly = [link_of(a, b) for a, b in G.edges if y in ((a, end_sector(a, b)), (b, end_sector(b, a)))][0]
        hits = [(a, d) for d in DIRS for a, _ in aggressors(d, {k: COUPLINGS[k]})]
        desc = "; ".join(f"{a[0]}->{a[1]} load hits {d[0]}->{d[1]} (subframe {subframe(d)})" for a, d in hits)
        check(POL[x] != POL[y] and lx != ly,
              f"{k} ({att}): {x[0]}.{x[1]} ({POL[x]}) <-> {y[0]}.{y[1]} ({POL[y]}), links {lx}/{ly}: {desc}",
              "polarity + TDD subframe rule")
        px = [n for n in LINKS[lx][:2] if n != x[0]][0]
        py = [n for n in LINKS[ly][:2] if n != y[0]][0]
        spx, spy = (px, end_sector(px, x[0])), (py, end_sector(py, y[0]))
        check(POL[spx] == POL[y] and POL[x] == POL[spy] and POL[spx] != POL[spy],
              f"{k} leak paths through coupler directivity: {spx[0]}.{spx[1]}<->{y[0]}.{y[1]} and "
              f"{x[0]}.{x[1]}<->{spy[0]}.{spy[1]} are same-polarity (harmless); "
              f"{spx[0]}.{spx[1]}<->{spy[0]}.{spy[1]} needs two leaks and sits >= 2 x span loss below",
              "the peer of an opposite-polarity sector has the victim's polarity")


# --------------------------------------------------------------------------------------------
# 3. DPIR candidates
# --------------------------------------------------------------------------------------------
def candidates(flows):
    print("== DPIR candidates (repo diverse_paths, k=2, MAX_STRETCH=%d) ==" % MAX_STRETCH)
    cands, covered = {}, set()
    for fid, s, t, _ in flows:
        ps = diverse_paths(G, s, t, k=2)
        cands[fid] = ps
        e = [{frozenset(x) for x in zip(p, p[1:])} for p in ps]
        check(len(ps) == 2 and not (e[0] & e[1]),
              f"{fid} {s}->{t}: 2 link-disjoint candidates " + " | ".join("-".join(p) for p in ps),
              "validate.diverse_paths(G, s, t, k=2)")
        simple = sorted(nx.all_simple_paths(G, s, t), key=len)
        check(sorted(map(tuple, simple)) == sorted(map(tuple, ps)),
              f"{fid}: the candidates are every simple path {s}->{t} ({len(simple)}), so DPIR's choice is "
              "the optimum over all loop-free routings, not just over its candidate set",
              "networkx.all_simple_paths")
        covered |= set().union(*e)
    check(covered == {frozenset(e) for e in G.edges}, "every link is on at least one DPIR candidate",
          "union of candidate links == link set")
    return cands


# --------------------------------------------------------------------------------------------
# 4. Capacity model
# --------------------------------------------------------------------------------------------
def maxmin(sub, caps):
    """sub: list of (demand, [directions]); max-min fair water-filling on per-direction capacity."""
    r = [0.0] * len(sub)
    rem = dict(caps)
    active = {i for i, (dem, ds) in enumerate(sub) if dem > 0 and ds and all(caps.get(d, 0) > 0 for d in ds)}
    while active:
        inc = min(min(sub[i][0] - r[i] for i in active),
                  min(rem[d] / sum(1 for j in active if d in sub[j][1]) for i in active for d in sub[i][1]))
        for i in active:
            r[i] += inc
            for d in sub[i][1]:
                rem[d] -= inc
        active -= {i for i in active if sub[i][0] - r[i] < 1e-6 or any(rem[d] < 1e-6 for d in sub[i][1])}
    return r


def evaluate(sub, couplings=COUPLINGS, model="LA", inr=None, d0=0.0, down=(), nocarry=(), snr=None,
             fixed=None):
    """Throughput of each sub-flow (demand, node path) for one routing.

    Fixed point between rates, per-direction TX duty (load / capacity, 1 when saturated) and the
    victims' capacities.  An aggressor counts when its duty exceeds the LA PER target.
      model LA: the victim's LA runs at the MCS of its SINR whenever an aggressor counts;
      model TS: ideal per-slot adaptation, capacity = E[rate] over aggressor on/off (an upper
                bound for victims, optimistic for every interference-unaware baseline).
    d0: idle transmit floor of every link that is up (or igniting) but carries no traffic.
    down: links taken down (no traffic, no emission).  nocarry: links igniting (emit d0, carry
    nothing).  snr: per-link SNR overrides.  fixed: per-link fixed capacity (an IP-50 hop: FDD,
    not coupled)."""
    snr = {k: v[4] for k, v in LINKS.items()} | (snr or {})
    fixed = fixed or {}
    cpl = {k: (x, y, inr if inr is not None else i, u) for k, (x, y, i, u) in couplings.items()}
    emit = {d for d in DIRS if link_of(d) not in down and link_of(d) not in fixed}
    usable = [d for d in DIRS if link_of(d) not in down and link_of(d) not in nocarry]
    agg = {d: [(a, i) for a, i in aggressors(d, cpl) if a in emit] if link_of(d) not in fixed else []
           for d in usable}
    base = {d: fixed.get(link_of(d), rate(snr[link_of(d)])) for d in usable}
    hops = [list(zip(p, p[1:])) for _, p in sub]
    dem = [x if all(h in base for h in hp) else 0.0 for (x, _), hp in zip(sub, hops)]
    caps = dict(base)
    seen = []
    duty = {}
    for _ in range(200):
        thr = maxmin(list(zip(dem, hops)), caps)
        load = {d: 0.0 for d in DIRS}
        for t, hp in zip(thr, hops):
            for d in hp:
                load[d] += t
        used = {d for hp, x in zip(hops, dem) if x > 0 for d in hp}
        duty = {}
        for d in emit:
            if d in used:
                duty[d] = 1.0 if caps.get(d, 0) <= 0 else max(d0, min(1.0, load[d] / caps[d]))
            else:
                duty[d] = d0
        new = {}
        for d in usable:
            if model == "LA":
                act = [i for a, i in agg[d] if duty.get(a, 0) > PER_TARGET]
                new[d] = fixed.get(link_of(d), rate(sinr(snr[link_of(d)], act)))
            else:
                tot = 0.0
                for on in itertools.product((0, 1), repeat=len(agg[d])):
                    p = 1.0
                    for (a, _), o in zip(agg[d], on):
                        p *= duty.get(a, 0) if o else 1 - duty.get(a, 0)
                    tot += p * rate(sinr(snr[link_of(d)], [i for (a, i), o in zip(agg[d], on) if o]))
                new[d] = fixed.get(link_of(d), tot)
        key = tuple(round(new[d], 3) for d in usable)
        if key in seen:
            break
        seen.append(key)
        caps = new
    thr = maxmin(list(zip(dem, hops)), caps)
    return thr, caps, duty


def per_flow(sub_fids, thr, flows=FLOWS):
    out = {f[0]: 0.0 for f in flows}
    for fid, t in zip(sub_fids, thr):
        out[fid] += t
    return out


def joint_options(flows, cands):
    return list(itertools.product(*[range(len(cands[f[0]])) for f in flows]))


def paths_of(opt, flows, cands):
    return [cands[f[0]][i] for f, i in zip(flows, opt)]


def rank(flows, cands, demands=None, **kw):
    """Every joint routing, best first (ties: fewer hops, then path names)."""
    demands = demands if demands is not None else [f[3] for f in flows]
    rows = []
    for opt in joint_options(flows, cands):
        ps = paths_of(opt, flows, cands)
        thr, _, _ = evaluate(list(zip(demands, ps)), **kw)
        rows.append(dict(opt=opt, paths=["-".join(p) for p in ps], thr=[round(t) for t in thr],
                         total=round(sum(thr), 1), hops=sum(len(p) - 1 for p in ps)))
    rows.sort(key=lambda r: (-r["total"], r["hops"], r["paths"]))
    return rows


def hopcount_option(flows, cands, report=True):
    """Hop-count routing (OSPF, ECMP, Open/R with every metric 1): unique shortest path here."""
    opt = []
    for fid, s, t, _ in flows:
        ps = cands[fid]
        i = min(range(len(ps)), key=lambda j: (len(ps[j]) - 1, ps[j]))
        n_short = len(list(nx.all_shortest_paths(G, s, t)))
        if report:
            check(n_short == 1, f"hop-count route of {fid} is unique ({'-'.join(ps[i])}, {len(ps[i]) - 1} hop): "
                  "OSPF, ECMP and Open/R with flat metrics all pick it", "networkx.all_shortest_paths")
        opt.append(i)
    return tuple(opt)


def row_of(rows, opt):
    return [r for r in rows if r["opt"] == opt][0]


# --------------------------------------------------------------------------------------------
# 5. Terragraph Open/R MCS-based routing (capacity-aware, interference-unaware)
# --------------------------------------------------------------------------------------------
def openr_metric(mcs):
    for m, metric in OPENR_METRIC:
        if mcs >= m:
            return metric
    return OPENR_METRIC[-1][1]


def openr_run(dem_sched, flows, cands, **kw):
    """Open/R with Terragraph's MCS link metric, 1 s steps.  Each node sets the metric of its own
    TX direction from that direction's MCS (LA no-traffic mode caps an idle direction at MCS9;
    an idle direction still sees the interference on it).  A changed metric is advertised after
    one step of damping if the node has a token.  Each flow takes its least-cost candidates;
    equal-cost candidates share it evenly (ECMP over the flow's 5-tuples)."""
    adv = {d: 1 for d in DIRS}
    tokens = {n: float(OPENR_TOKEN_BURST) for n in NODES}
    tot, per, changes, prev = [], [], 0, None
    for dem in dem_sched:
        sub, fids = [], []
        for (fid, _, _, _), x in zip(flows, dem):
            cost = [sum(adv[d] for d in zip(p, p[1:])) for p in cands[fid]]
            best = [p for p, c in zip(cands[fid], cost) if c == min(cost)]
            sub += [(x / len(best), p) for p in best]
            fids += [fid] * len(best)
        sig = tuple((f, tuple(p)) for f, (_, p) in zip(fids, sub))
        changes += prev is not None and sig != prev
        prev = sig
        thr, caps, _ = evaluate(sub, **kw)
        pf = per_flow(fids, thr, flows)
        tot.append(sum(thr))
        per.append(pf)
        used = {d for (x, p) in sub if x > 0 for d in zip(p, p[1:])}
        for n in tokens:
            tokens[n] = min(OPENR_TOKEN_BURST, tokens[n] + OPENR_TOKEN_RATE)
        for d in DIRS:
            m = RATE_MCS.get(caps.get(d, 0), 0)
            if d not in used:
                m = min(m, NO_TRAFFIC_MAX_MCS)
            target = openr_metric(m)
            if target != adv[d] and tokens[d[0]] >= 1:
                adv[d] = target
                tokens[d[0]] -= 1
    return dict(total=tot, per=per, changes=changes)


# --------------------------------------------------------------------------------------------
# 6. Static results
# --------------------------------------------------------------------------------------------
def static_results(cands):
    print("== joint routings at the nominal setting (INR %.0f dB at both couplings, LA model) ==" % INR_NOM)
    res = {}
    B = hopcount_option(FLOWS, cands)
    rows = rank(FLOWS, cands)
    for r in rows:
        print(f"  {' + '.join(r['paths']):28s} F1 {r['thr'][0]:5d}  F2 {r['thr'][1]:5d}  total {r['total']:6.0f}"
              f"  ({r['hops']} hops)")
    base, dpir = row_of(rows, B), rows[0]
    gain = dpir["total"] / base["total"] - 1
    check(dpir["opt"] != B and gain > 0.5,
          f"DPIR's target {' + '.join(dpir['paths'])} delivers {dpir['total']:.0f} Mb/s against hop-count "
          f"{base['total']:.0f} Mb/s (+{100 * gain:.0f} %)", "exhaustive evaluation of the 4 joint routings")
    check(sum(1 for r in rows if r["total"] == dpir["total"]) == 1,
          "DPIR's target is the unique optimum (the MCS9 side hops rule out detouring the 1600 Mb/s flow: "
          f"{row_of(rows, (1, 0))['total']:.0f} Mb/s)", "ranked joint routings")
    res.update(hopcount_opt=B, rows=rows, hopcount=base, dpir=dpir, gain=gain)

    # Open/R MCS-based routing, constant demand, 120 s
    o = openr_run([[f[3] for f in FLOWS]] * 120, FLOWS, cands)
    avg = sum(o["total"]) / len(o["total"])
    res["openr"] = dict(avg=round(avg), min=round(min(o["total"])), max=round(max(o["total"])),
                        changes=o["changes"], trace=[round(x) for x in o["total"]])
    check(base["total"] <= avg < dpir["total"] * 0.9,
          f"Terragraph's own MCS-based routing (Open/R metric map 15/3/1, damping, token bucket, ECMP) "
          f"averages {avg:.0f} Mb/s over 120 s (range {min(o['total']):.0f}-{max(o['total']):.0f}, "
          f"{o['changes']} route changes): it sees the coupled spans fall to MCS4 but not why, and "
          "swings between them and the detours", "openr_run, 120 steps of 1 s")

    # load dependence
    thr0, _, _ = evaluate([(0.0, cands["F1"][0]), (900.0, cands["F2"][0])])
    thr1, _, _ = evaluate([(1600.0, cands["F1"][0]), (900.0, cands["F2"][0])])
    check(thr0[1] == 900 and thr1[1] < 900,
          f"the coupling depends on load: F2 on its direct span gets {thr0[1]:.0f} Mb/s with F1 idle and "
          f"{thr1[1]:.0f} Mb/s with F1 at 1600 Mb/s", "evaluate with F1 demand 0 and 1600")
    res["load_dependence"] = dict(f1_idle=thr0[1], f1_on=thr1[1])

    # time-share bound
    rows_ts = rank(FLOWS, cands, model="TS")
    bts = row_of(rows_ts, B)
    check(rows_ts[0]["opt"] == dpir["opt"] and rows_ts[0]["total"] > 1.5 * bts["total"],
          f"under the ideal time-share bound (optimistic for the baselines) DPIR still wins: "
          f"{rows_ts[0]['total']:.0f} vs {bts['total']:.0f} Mb/s", "evaluate(model='TS')")
    res["ts"] = dict(hopcount=bts["total"], dpir=rows_ts[0]["total"])

    # one coupling only (smoke-test build)
    kw_only = {"KW": COUPLINGS["KW"]}
    r1, r1t = rank(FLOWS, cands, couplings=kw_only), rank(FLOWS, cands, couplings=kw_only, model="TS")
    g_la = r1[0]["total"] / row_of(r1, B)["total"] - 1
    g_ts = r1t[0]["total"] / row_of(r1t, B)["total"] - 1
    check(g_la > 0 and g_ts > 0,
          f"smoke-test build with only KW cabled: DPIR {r1[0]['total']:.0f} vs hop-count "
          f"{row_of(r1, B)['total']:.0f} Mb/s (+{100 * g_la:.0f} % LA, +{100 * g_ts:.0f} % TS): enough to check "
          "the set-up, too small to show DPIR", "rank with KW only")
    res["kw_only"] = dict(dpir=r1[0]["total"], hopcount=row_of(r1, B)["total"], gain_la=g_la, gain_ts=g_ts)

    # couplings off = control run (also LS on another channel)
    r0 = rank(FLOWS, cands, couplings={})
    check(row_of(r0, B)["total"] == r0[0]["total"] == sum(f[3] for f in FLOWS) and r0[0]["opt"] == B,
          f"control run, couplings off (or LS moved to channel 1): hop-count already delivers "
          f"{row_of(r0, B)['total']:.0f} Mb/s and DPIR keeps the direct spans, so the deficit is the coupling",
          "rank with no couplings")
    res["control"] = dict(hopcount=row_of(r0, B)["total"], dpir=r0[0]["total"], dpir_paths=r0[0]["paths"])

    # INR sweep (live ramp of ATT-5/ATT-6)
    print("== INR sweep: ATT-5/ATT-6 stepped from off towards the nominal setting ==")
    sweep = []
    for inr in range(-4, 25, 2):
        rl = rank(FLOWS, cands, inr=float(inr))
        rt = rank(FLOWS, cands, inr=float(inr), model="TS")
        oo = openr_run([[f[3] for f in FLOWS]] * 120, FLOWS, cands, inr=float(inr))
        s = dict(inr=inr, sinr=round(sinr(25.0, [inr]), 1), hopcount=row_of(rl, B)["total"],
                 openr=round(sum(oo["total"]) / 120), dpir=rl[0]["total"], dpir_paths=rl[0]["paths"],
                 hopcount_ts=row_of(rt, B)["total"], dpir_ts=rt[0]["total"])
        sweep.append(s)
        print(f"  INR {inr:>3} dB (SINR {s['sinr']:5.1f}): hop-count {s['hopcount']:5.0f}  Open/R-MCS "
              f"{s['openr']:5.0f}  DPIR {s['dpir']:5.0f}  [{' + '.join(s['dpir_paths'])}]")
    first = [s for s in sweep if s["dpir"] > s["hopcount"]][0]
    check(all(s["dpir"] >= max(s["hopcount"], s["openr"]) for s in sweep)
          and all(s["dpir_paths"] == base["paths"] for s in sweep if s["inr"] < first["inr"]),
          f"DPIR is never below either baseline; it keeps the direct spans up to INR {first['inr'] - 2} dB and "
          f"moves F2 from INR {first['inr']} dB (victim SINR {first['sinr']} dB)", "INR sweep -4..24 dB")
    res["sweep"] = sweep
    res["switch_inr"] = first["inr"]

    # encore: fade the east side hop until the detour is no longer worth it
    print("== encore: fade LE (ATT-2) at INR 14 dB ==")
    enc = []
    for s in [12, 11, 10, 9, 8, 7, 6, 5, 4]:
        rl = rank(FLOWS, cands, inr=14.0, snr={"LE": float(s)})
        enc.append(dict(le_snr=s, le_mcs=mcs_of(s)[0], hopcount=row_of(rl, B)["total"], dpir=rl[0]["total"],
                        dpir_paths=rl[0]["paths"]))
        print(f"  LE SNR {s:>2} dB (MCS{mcs_of(s)[0]}): hop-count {row_of(rl, B)['total']:5.0f}  DPIR "
              f"{rl[0]['total']:5.0f}  [{' + '.join(rl[0]['paths'])}]")
    back = [e for e in enc if e["dpir_paths"] == base["paths"]]
    check(back and enc[0]["dpir_paths"] != base["paths"],
          f"encore at INR 14 dB: with LE faded to SNR {back[0]['le_snr']} dB (MCS{back[0]['le_mcs']}) the detour "
          f"is worth less than the interference it avoids, and DPIR puts F2 back on the direct span "
          f"({back[0]['dpir']:.0f} Mb/s)", "rank with LE SNR stepped 12 -> 4 dB")
    res["encore"] = dict(inr=14, steps=enc, back_at=back[0]["le_snr"])

    # optional variant: the east side hop as an IP-50 link
    ipv = {}
    for w, c in IP50_CAP.items():
        rl = rank(FLOWS, cands, fixed={"LE": c})
        ipv[w] = dict(cap=c, hopcount=row_of(rl, B)["total"], dpir=rl[0]["total"], dpir_paths=rl[0]["paths"])
    check(all(v["dpir"] > 2 * v["hopcount"] for v in ipv.values()),
          "optional IP-50 side hop (PoC 1's E1-E2 chain, +2 radio units): DPIR "
          + ", ".join(f"{v['dpir']:.0f} vs {v['hopcount']:.0f} Mb/s at {w}" for w, v in ipv.items()),
          "rank with LE as a fixed-capacity, uncoupled FDD link")
    res["ip50_variant"] = ipv

    # QoS-Null worst case and the link-activation fallback
    print("== idle links: does routing away really silence the aggressor? ==")
    idle = {}
    for name, d0 in IDLE_FLOORS.items():
        out = {}
        for model in ("LA", "TS"):
            rl = rank(FLOWS, cands, model=model, d0=d0)
            best_off = None
            for opt in joint_options(FLOWS, cands):
                ps = paths_of(opt, FLOWS, cands)
                used = {link_of(d) for p in ps for d in zip(p, p[1:])}
                spare = [k for k in LINKS if k not in used]
                for n in range(len(spare) + 1):
                    for dn in itertools.combinations(spare, n):
                        thr, _, _ = evaluate(list(zip([f[3] for f in FLOWS], ps)), model=model, d0=d0, down=dn)
                        key = (round(sum(thr), 1), -len(dn))
                        if best_off is None or key > best_off[0]:
                            best_off = (key, ["-".join(p) for p in ps], list(dn))
            out[model] = dict(hopcount=row_of(rl, B)["total"], dpir_route_only=rl[0]["total"],
                              dpir_link_off=best_off[0][0], paths=best_off[1], down=best_off[2])
        idle[name] = dict(d0=round(d0, 5), **out)
        print(f"  {name:20s} floor {100 * d0:4.2f} %: hop-count {out['LA']['hopcount']:5.0f}  DPIR route only "
              f"{out['LA']['dpir_route_only']:5.0f}  DPIR + unused coupled link down {out['LA']['dpir_link_off']:5.0f}"
              f" (down: {', '.join(out['LA']['down']) or '-'})")
    q = idle["QoS-Null every slot"]["LA"]
    check(IDLE_FLOORS["management only"] < PER_TARGET < IDLE_FLOORS["QoS-Null every slot"],
          f"go/no-go line for the idle-emission test: a management-only floor ({100 * IDLE_FLOORS['management only']:.2f} %) "
          f"sits below LA's {100 * PER_TARGET:.1f} % PER target, QoS-Null in every slot "
          f"({100 * IDLE_FLOORS['QoS-Null every slot']:.1f} %) above it", "MAC frame timing estimate")
    check(idle["management only"]["LA"]["dpir_route_only"] == dpir["total"]
          and q["dpir_route_only"] <= q["hopcount"] + 1 and q["dpir_link_off"] == dpir["total"],
          f"if idle links send QoS-Null in every slot, re-routing alone gains nothing ({q['dpir_route_only']:.0f} vs "
          f"{q['hopcount']:.0f} Mb/s); taking the unused coupled span down ({', '.join(q['down'])}) restores "
          f"{q['dpir_link_off']:.0f} Mb/s, so link activation is DPIR's fallback action",
          "evaluate with d0 per floor, with and without unused links down")
    res["idle"] = idle
    return res


# --------------------------------------------------------------------------------------------
# 7. Dynamic traces
# --------------------------------------------------------------------------------------------
def best_opt(dem, cands, flows=FLOWS, **kw):
    rl = []
    for opt in joint_options(flows, cands):
        thr, _, _ = evaluate(list(zip(dem, paths_of(opt, flows, cands))), **kw)
        rl.append((-round(sum(thr), 3), sum(len(p) - 1 for p in paths_of(opt, flows, cands)), opt))
    return min(rl)[2]


def dynamic_silent(cands, B):
    """Run A: idle links silent.  F1 bursts (on 10-39 s and 70-99 s), F2 constant; 1 s steps."""
    print("== dynamic run A (idle links silent): F1 1600 Mb/s on at 10-39 s and 70-99 s, F2 900 constant ==")
    T = 120
    f1 = [1600.0 if (10 <= t < 40 or 70 <= t < 100) else 0.0 for t in range(T)]
    f2 = [900.0] * T
    sched = [[a, b] for a, b in zip(f1, f2)]
    opts = joint_options(FLOWS, cands)
    out = {}
    for policy in ["hopcount", "openr_mcs", "reactive", "reactive_clean", "dpir_lagged", "dpir_predictive"]:
        if policy == "openr_mcs":
            o = openr_run(sched, FLOWS, cands)
            tot, f2got, chg = sum(o["total"]), sum(p["F2"] for p in o["per"]), o["changes"]
            f2bad = sum(p["F2"] < 0.95 * x for p, x in zip(o["per"], f2))
            trace = [round(x) for x in o["total"]]
        else:
            prev, meas = B, None
            tot = f2got = 0.0
            chg = f2bad = 0
            trace = []
            for t in range(T):
                dem = sched[t]
                if policy == "hopcount":
                    opt = B
                elif policy == "dpir_predictive":
                    opt = best_opt(dem, cands)               # the predictor knows this step's demand
                elif policy == "dpir_lagged":
                    opt = best_opt(sched[t - 1] if t else dem, cands)   # interference-aware, one step late
                elif meas is None:
                    opt = B
                else:                                        # capacity-aware, interference-unaware
                    est = []
                    for o in opts:
                        hp = [list(zip(p, p[1:])) for p in paths_of(o, FLOWS, cands)]
                        thr = maxmin(list(zip(dem, hp)), meas)
                        est.append((-round(sum(thr), 3), sum(len(h) for h in hp), o != prev, o))
                    opt = min(est)[3]
                chg += opt != prev
                ps = paths_of(opt, FLOWS, cands)
                thr, caps, _ = evaluate(list(zip(dem, ps)))
                used = {d for p in ps for d in zip(p, p[1:])}
                if policy == "reactive":          # idle directions report what they see (LA no-traffic cap)
                    meas = {d: (caps[d] if d in used else min(caps[d], 1050.0)) for d in DIRS}
                elif policy == "reactive_clean":  # idle directions report their interference-free rate
                    meas = {d: (caps[d] if d in used else rate(LINKS[link_of(d)][4])) for d in DIRS}
                tot += sum(thr)
                f2got += thr[1]
                f2bad += thr[1] < 0.95 * f2[t]
                trace.append(round(sum(thr)))
                prev = opt
        offered = sum(f1) + sum(f2)
        out[policy] = dict(delivered_pct=round(100 * tot / offered, 1), f2_pct=round(100 * f2got / sum(f2), 1),
                           f2_starved_s=int(f2bad), route_changes=int(chg))
        print(f"  {policy:16s} {out[policy]}")
        out[policy]["trace"] = trace
    out["offered"] = [round(a + b) for a, b in sched]
    p = out["dpir_predictive"]
    check(p["delivered_pct"] >= max(v["delivered_pct"] for k, v in out.items() if k != "offered")
          and p["f2_starved_s"] == 0,
          f"run A: predictive DPIR delivers {p['delivered_pct']} % of the offered traffic, F2 never starved; "
          f"hop-count {out['hopcount']['delivered_pct']} %, Open/R MCS routing {out['openr_mcs']['delivered_pct']} %, "
          f"reactive capacity-aware {out['reactive']['delivered_pct']} / {out['reactive_clean']['delivered_pct']} %, "
          f"interference-aware but one step late {out['dpir_lagged']['delivered_pct']} %",
          "120-step trace with the same capacity model")
    return out


def dynamic_qosnull(cands, B):
    """Run B: idle links send QoS-Null (worst case); DPIR also takes unused coupled links down.
    F1 1600 on at 10-39 s and 70-99 s; F2 900 while F1 is on, 1600 while it is off (anti-phase).
    A link brought back up ignites for T_IGN s: it emits the idle floor but carries nothing."""
    d0 = IDLE_FLOORS["QoS-Null every slot"]
    print(f"== dynamic run B (QoS-Null floor {100 * d0:.1f} %, link activation, {T_IGN} s ignition) ==")
    T = 120
    f1 = [1600.0 if (10 <= t < 40 or 70 <= t < 100) else 0.0 for t in range(T)]
    f2 = [900.0 if x > 0 else 1600.0 for x in f1]
    sched = [[a, b] for a, b in zip(f1, f2)]
    opts = joint_options(FLOWS, cands)

    def plan(dem):
        """Best (routing, links down) for a demand; ties -> fewer links down, fewer hops."""
        best = None
        for opt in opts:
            ps = paths_of(opt, FLOWS, cands)
            used = {link_of(d) for p in ps for d in zip(p, p[1:])}
            spare = [k for k in LINKS if k not in used]
            for n in range(len(spare) + 1):
                for dn in itertools.combinations(spare, n):
                    thr, _, _ = evaluate(list(zip(dem, ps)), d0=d0, down=dn)
                    key = (round(sum(thr), 1), -len(dn), -sum(len(p) for p in ps))
                    if best is None or key > best[0]:
                        best = (key, opt, set(dn))
        return best[1], best[2]

    plans = {}

    def plan_c(dem):
        k = tuple(dem)
        if k not in plans:
            plans[k] = plan(dem)
        return plans[k]

    out = {}
    for policy in ["hopcount", "dpir_route_only", "dpir_links_lagged", "dpir_links_predictive"]:
        down, igniting = set(), {}
        tot = 0.0
        ign_events = 0
        trace = []
        for t in range(T):
            dem = sched[t]
            if policy == "hopcount":
                opt, want_down = B, set()
            elif policy == "dpir_route_only":
                opt, want_down = best_opt(dem, cands, d0=d0), set()
            elif policy == "dpir_links_lagged":
                opt, want_down = plan_c(sched[t - 1] if t else dem)
            else:
                opt, want_down = plan_c(dem)
                if t + T_IGN < T:   # pre-ignite what the plan T_IGN s ahead needs
                    _, ahead = plan_c(sched[t + T_IGN])
                    want_down = want_down & ahead
            for k in list(down):
                if k not in want_down:          # bring it back: ignition starts now
                    down.discard(k)
                    igniting[k] = T_IGN
                    ign_events += 1
            if not igniting:                    # make-before-break: take links down only once
                for k in want_down:             # nothing is still igniting
                    if k not in down:
                        down.add(k)
            avail_opts = [o for o in opts
                          if not ({link_of(d) for p in paths_of(o, FLOWS, cands) for d in zip(p, p[1:])}
                                  & (down | set(igniting)))]
            if opt not in avail_opts:            # target route not up yet: best available one
                rl = []
                for o in avail_opts:
                    thr, _, _ = evaluate(list(zip(dem, paths_of(o, FLOWS, cands))), d0=d0, down=down,
                                         nocarry=set(igniting))
                    rl.append((-round(sum(thr), 3), o))
                opt = min(rl)[1] if rl else None
            thr = [0.0]
            if opt is not None:
                thr, _, _ = evaluate(list(zip(dem, paths_of(opt, FLOWS, cands))), d0=d0, down=down,
                                     nocarry=set(igniting))
                tot += sum(thr)
            trace.append(round(sum(thr)))
            for k in list(igniting):
                igniting[k] -= 1
                if igniting[k] <= 0:
                    del igniting[k]
        offered = sum(f1) + sum(f2)
        out[policy] = dict(delivered_pct=round(100 * tot / offered, 1), ignitions=ign_events)
        print(f"  {policy:22s} {out[policy]}")
        out[policy]["trace"] = trace
    out["offered"] = [round(a + b) for a, b in sched]
    check(out["dpir_links_predictive"]["delivered_pct"] >= out["dpir_links_lagged"]["delivered_pct"]
          > out["dpir_route_only"]["delivered_pct"] and out["dpir_links_lagged"]["delivered_pct"]
          > out["hopcount"]["delivered_pct"],
          f"run B (worst case): with link activation DPIR delivers {out['dpir_links_predictive']['delivered_pct']} % "
          f"(predictive) / {out['dpir_links_lagged']['delivered_pct']} % (one step late) against route-only "
          f"{out['dpir_route_only']['delivered_pct']} % and hop-count {out['hopcount']['delivered_pct']} %",
          f"120-step trace, QoS-Null floor, {T_IGN} s ignition")
    return out


def four_flows(cands):
    print("== 4-flow extension (reverse flows added, same bench) ==")
    flows4 = FLOWS + [("F3", "E1", "W1", 1600.0), ("F4", "W2", "E2", 900.0)]
    c4 = dict(cands)
    for fid, s, t, _ in flows4[2:]:
        c4[fid] = diverse_paths(G, s, t, k=2)
    rows = rank(flows4, c4)
    b4 = row_of(rows, hopcount_option(flows4, c4, report=False))
    check(rows[0]["total"] > 1.5 * b4["total"],
          f"4 flows, 16 joint routings: DPIR {rows[0]['total']:.0f} vs hop-count {b4['total']:.0f} Mb/s "
          f"({' + '.join(rows[0]['paths'])})", "exhaustive evaluation of 16 joint routings")
    return dict(dpir=rows[0], hopcount=b4)


# --------------------------------------------------------------------------------------------
# 8. Why not smaller; device count; coupling-arm budget
# --------------------------------------------------------------------------------------------
def size_checks():
    print("== why not smaller ==")
    g2 = nx.Graph([("A", "B")])
    p2 = diverse_paths(g2, "A", "B", k=2)
    check(len(p2) < 2, f"one GTS003 box (2 nodes, links only between them): diverse_paths finds {len(p2)} "
          "route, so DPIR has no choice", "diverse_paths on the 2-node graph")
    g3 = nx.Graph([("A", "B"), ("B", "C"), ("C", "A")])
    check(len(diverse_paths(g3, "A", "B", k=2)) == 2 and not nx.is_bipartite(g3),
          "a 3-node triangle gives 2 routes but is an odd cycle: it needs a hybrid-polarity node "
          "(software hybrid halves capacity; hardware hybrid on adjacent sectors is discouraged)",
          "diverse_paths + is_bipartite on the triangle")
    poc1 = CONFIGS["A7"]
    n_tg, n_ip = len(NODES), 0
    boxes = sorted({v[0] for v in NODES.values()})
    check(n_tg + n_ip < poc1["ip50"] + poc1["tg"] and len(boxes) == 2,
          f"PoC 2 uses {n_tg} radio units (4 x 2-sector N367, 0 IP-50) in {len(boxes)} Spider boxes, both nodes "
          f"of each box used; PoC 1 (A7) uses {poc1['ip50'] + poc1['tg']} ({poc1['tg']} TG + {poc1['ip50']} IP-50) "
          "in 4 boxes", "count of NODES / topologies.CONFIGS['A7']")
    print("== coupling-arm budget (relative: IF levels still to be measured) ==")
    C_DB, IL_ATT0, SIR_STRONG, SIR_OFF = 10.0, 10.0, 3.0, 40.0
    pads = {}
    for L in (40.0, 60.0, 80.0):
        lo = max(0.0, SIR_OFF - 63.0 - 2 * C_DB - IL_ATT0 + L)
        hi = SIR_STRONG - 2 * C_DB - IL_ATT0 + L
        pads[int(L)] = (round(lo), round(hi))
        print(f"  span path loss {L:.0f} dB: arm pad {lo:.0f}-{hi:.0f} dB covers SIR {SIR_STRONG:.0f}..{SIR_OFF:.0f} dB")
    check(all(hi - lo >= 10 for lo, hi in pads.values()),
          "for span losses of 40-80 dB a fixed arm pad exists (" + ", ".join(
              f"{lo}-{hi} dB at {L}" for L, (lo, hi) in pads.items()) + ") that lets the 0-63 dB coupling "
          "attenuator reach both SIR 3 dB (INR 22, still below SNR - 2 dB) and 'off' (SIR 40 dB)",
          "arm loss = 2 x 10 dB coupler + attenuator IL + pad + setting, against span loss")
    return dict(pads=pads, units=n_tg + n_ip, boxes=boxes, poc1_units=poc1["ip50"] + poc1["tg"])


# --------------------------------------------------------------------------------------------
# 9. Hardware and bench plan
# --------------------------------------------------------------------------------------------
# (group, item, PoC 1 A7, PoC 2, note)
HARDWARE = [
    ("Radios", "Siklu N367 2-sector node (GTS003 Spider box, 2 nodes per box)", "6 (3 boxes)", "4 (2 boxes)",
     "both GTS003 boxes, both nodes of each; one polarity per box"),
    ("Radios", "Siklu N367 4-sector node (GTST002B)", "1", "0", "the hub: not needed, no PtMP"),
    ("Radios", "Ceragon IP-50 radio unit", "8", "0 (2 in the IP-50 side-hop option)", ""),
    ("IP-50 chain", "Waveguide transitions, fixed pads, 0-95 dB attenuators, -48 VDC", "8 / 8 / 4 / 1", "0",
     "stays on the PoC 1 bench"),
    ("IF (13-17.4 GHz)", "4-channel programmable attenuator, 0-63 dB (ZTDAT-4-18G63S class)", "9", "6",
     "ATT-1..4 one per link (tile i to tile i), ATT-5/6 the coupling arms; 3 spare"),
    ("IF", "Splitter / combiner (hub PtMP fan-out)", "6", "0", "all links are point-to-point"),
    ("IF", "Directional coupler 10 dB, 0.5-18 GHz (Mini-Circuits ZUDC10-183+)", "0", "16 (4 if single-tile)",
     "new: in line on every tile port of W1.E, W2.E, E1.W, E2.W"),
    ("IF", "Fixed SMA pad, DC-18 GHz", "-", "8", "coupling arms; value from the measured span loss"),
    ("IF", "DC block, SMA, 18 GHz", "-", "8 (+32 if ports carry DC)", "coupling arms; all paths if DC found"),
    ("IF", "High-pass filter, pass 13-18 GHz, reject <= 8.2 GHz", "-", "0-8", "only if LO/control tones on the ports"),
    ("IF", "SMA cables and jumpers, 18 GHz", "~24", "~64", "spans 16 each, side hops 8 each, arms 8 each"),
    ("Timing", "GNSS/PPS feed to the Spider boxes (time server + splitter or simulator)", "1", "1",
     "every DN needs PPS sync to ignite DN-DN links"),
    ("Network", "Site router / L3 switch with per-flow policy routing", "7", "4", "pins DPIR's paths if soft-drain is slow"),
    ("Compute", "Traffic generator / sink host (iperf3, TRex)", "6", "4", "F1 at W1/E1, F2 at E2/W2"),
    ("Compute", "Edge agent PC (MARL)", "6", "0", "no MARL in PoC 2"),
    ("Compute", "Servers: DPIR controller + predictor (GPU); TG E2E controller + NMS", "2", "2",
     "the controller stays up (no event)"),
    ("Power", "DC drawer and PDU", "DC drawer + 8 switched, metered PDU outlets", "DC drawer + plain PDU",
     "no site is switched off"),
    ("Network", "Out-of-band management switch", "1", "1", "attenuator control, node management, telemetry"),
    ("Test", "Spectrum analyser / power sensor 13-18 GHz with gated or zero-span mode", "1 (meters)", "1",
     "port checks, calibration, the idle-emission test"),
    ("Test", "VNA to 18 GHz (borrowed)", "-", "1", "isolation acceptance: uncoupled paths >= span loss + SNR + 10 dB"),
]

RUN_STEPS = [
    ("Rack and sync", "Rack GTS003 #1 (U16: W1, E2, all sectors even) and #2 (U13: E1, W2, all odd). Get GNSS/PPS "
     "lock on both. Bring up the E2E controller (it stays up). Per radio: channel 2, polarity, Golay LN 1 / LE 1 / "
     "LS 2 / LW 2, TPC off with fixed TX power, LA on (MCS 2-12), PBF/IM/RTCAL scans off."),
    ("Port check", "Before any coupling: on one SMA port per module check DC, LO (7-8.2 GHz) and control tones; "
     "measure IF TX level and RX range. Decide pads, DC blocks and filters from this."),
    ("Idle-emission test (go/no-go)", "Gated / zero-span power on a coupler's coupled port with the aggressor at "
     "0, 10, 50, 100 % iperf duty. Idle floor below ~0.5 % of the data airtime: routing alone switches the "
     "coupling (plan A). Floor near 4 % (QoS-Null every slot): DPIR must also take the unused span down (plan B); "
     "time a re-ignition."),
    ("Links", "Cable LN, LE, LS, LW through ATT-1..4 (tile i to tile i), coupling arms through ATT-5/6 at 63 dB "
     "(off). Find each link's drop point, then set LN and LS to SNR ~25 dB (MCS12) and LE, LW to ~12 dB (MCS9). "
     "Confirm 4-tile operation (txRficBitmap). VNA: every uncoupled path >= span loss + SNR + 10 dB."),
    ("Calibrate the couplings", "As terragraph-ctf PUMA_RF_LA-5.0: load the aggressor span (1600 Mb/s UDP), put "
     "900 Mb/s on the victim, step ATT-5 then ATT-6 in 2 dB steps from off to SIR 3 dB; log victim MCS, PER, "
     "throughput and aggressor duty; repeat at aggressor duty 0-100 %. Set the nominal SIR 7 dB (INR ~18 dB)."),
    ("Static A/B", "5 min per arm, UDP F1 W1->E1 1600 Mb/s and F2 E2->W2 900 Mb/s: (1) hop-count (Open/R metrics "
     "fixed at 1), (2) Open/R MCS-based metrics (Terragraph default), (3) DPIR (soft-drain LS, or site-router "
     "policy route E2->E1->W1->W2). Log per-flow throughput, loss, delay; per-link MCS, SNR, PER, TX duty."),
    ("Controls", "Couplings off, then LS moved to channel 1: hop-count must recover to 2500 Mb/s."),
    ("Live ramp", "Step ATT-5/ATT-6 by 2 dB per routing period from off to the nominal setting: DPIR keeps the "
     "direct spans to INR ~10 dB and moves F2 at ~12 dB; a router one period late loses that period."),
    ("Encore", "At INR ~14 dB fade LE (ATT-2) from SNR 12 to 5 dB: DPIR brings F2 back to the direct span once the "
     "detour is worth less than the interference it avoids."),
    ("Dynamic run", "F1 on/off (30 s on / 30 s off, then irregular), F2 constant, 1 s routing step: predictive DPIR, "
     "DPIR without its predictor, Open/R MCS routing, hop-count. Report delivered traffic, F2 dips at burst onset, "
     "route changes, delay. Plan B adds link activation with the measured ignition time."),
    ("Optional", "4-flow extension (F3 E1->W1, F4 W2->E2: 16 joint routings) and scheduled fades on ATT-2/ATT-4; "
     "the IP-50 side-hop variant if microwave continuity with PoC 1 is wanted."),
]

OPEN_QUESTIONS = [
    ("FATJ0498 SMA ports", "IF only, or also DC, LO or control? IF TX level and RX range? Can a link run single-tile "
     "on the reference tile (4 couplers instead of 16)?"),
    ("Idle emission", "Do idle sectors on the Siklu firmware leave their TX slots empty or send QoS-Null? "
     "This decides plan A (routing) or plan B (routing + link activation)."),
    ("Control surface", "Does Siklu expose per-radio polarity, channel, Golay, TPC/TX power, scan disable and Open/R "
     "softDisable / fixedMetric / linkMetricConfig to an external controller? How fast does a route change apply?"),
    ("Timing", "How is GNSS/PPS fed to the GTS003 boxes (internal antenna, external connector, common PPS)? "
     "How long does a link take to re-ignite?"),
    ("Spider box layout", "Which modules belong to node a and node b (assumed M1-M2 = a, M3-M4 = b)? Isolation "
     "inside a box and between boxes at 13-18 GHz?"),
    ("Attenuators", "Do PoC 1's 4-channel IF attenuators match ZTDAT-4-18G63S (insertion loss, channel isolation at "
     "13-17.4 GHz)? ZUDC10-183+ directivity there?"),
    ("Scenario", "Is a single-channel plan with deliberately cabled heuristic-4 couplings acceptable as PoC 2's "
     "interference scenario? (In this layout the coupled sectors do not face each other: the coupling is "
     "emulated and its strength is whatever the bench sets.)"),
    ("DPIR paper", "Full text not read (ideals.illinois.edu and arxiv.org blocked here). Confirm its interference "
     "model (SINR between active links?), action (a path per flow per step?), inputs, baselines and sizes; and "
     "whether 2-4 flows on 4 nodes are enough to train and evaluate it."),
]


def write_outputs(res, cands, peers):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "poc2_results.json"), "w") as fh:
        json.dump(dict(design=dict(pos=POS, nodes=NODES, links=LINKS, flows=FLOWS,
                                   couplings={k: [list(v[0]), list(v[1]), v[2], v[3]] for k, v in COUPLINGS.items()},
                                   module={f"{k[0]}.{k[1]}": v for k, v in MODULE.items()},
                                   polarity={f"{k[0]}.{k[1]}": v for k, v in POL.items()},
                                   candidates={k: ["-".join(p) for p in v] for k, v in cands.items()},
                                   idle_floors=IDLE_FLOORS, t_ign=T_IGN, per_target=PER_TARGET),
                       results=res, checks=CHECKS), fh, indent=1, default=str)

    s, h, d = res["static"], res["static"]["hopcount"], res["static"]["dpir"]
    L = ["# PoC 2: DPIR on a fixed point-to-point Terragraph ring (generated by poc2.py)", "",
         "No MARL, no event, no PtMP. Four 2-sector Siklu N367 nodes (the two GTS003 Spider boxes) form a ring of "
         "four fixed DN-DN point-to-point links. Two cabled co-channel couplings make the two spans interfere "
         "while both carry traffic; DPIR routes one flow the long way round and the interference disappears.", "",
         f"**Result at the nominal setting (INR {INR_NOM:.0f} dB):** hop-count {h['total']:.0f} Mb/s, Terragraph "
         f"MCS-based Open/R {s['openr']['avg']} Mb/s (average, {s['openr']['changes']} route changes in 120 s), "
         f"DPIR {d['total']:.0f} Mb/s (+{100 * s['gain']:.0f} % over hop-count).", "",
         "## Sites, sectors and Spider-box modules", "",
         "| Site | Box (slot) | Node | Polarity | Sector -> peer (module, link) |", "|---|---|---|---|---|"]
    for n, (box, slot, node, ns, pol) in NODES.items():
        secs = [f"{k} -> {v[0]} ({MODULE[(n, k)]}, {link_of(n, v[0])})" for (x, k), v in sorted(peers.items()) if x == n]
        L.append(f"| {n} | {box} ({slot}) | {node} | {pol} | {'; '.join(secs)} |")
    L += ["", "## Links", "", "| Link | Ends | Role | Channel / Golay | Set SNR | MCS, Mb/s per direction | Attenuator |",
          "|---|---|---|---|---|---|---|"]
    for k, (a, b, ch, gol, snr, att, role) in LINKS.items():
        m, r = mcs_of(snr)
        L.append(f"| {k} | {a}.{end_sector(a, b)} - {b}.{end_sector(b, a)} | {role} | {ch} / {gol} | {snr:.0f} dB | "
                 f"MCS{m}, {r:.0f} | {att} |")
    L += ["", "## Couplings (emulated co-channel interference)", "",
          "| Coupling | Sectors | Aggressor -> victim | Nominal | Attenuator |", "|---|---|---|---|---|"]
    for k, (x, y, inr, att) in COUPLINGS.items():
        hits = [(a, dd) for dd in DIRS for a, _ in aggressors(dd, {k: COUPLINGS[k]})]
        L.append(f"| {k} | {x[0]}.{x[1]} ({POL[x]}) - {y[0]}.{y[1]} ({POL[y]}) | "
                 + "; ".join(f"{a[0]}->{a[1]} hits {b[0]}->{b[1]} (subframe {subframe(b)})" for a, b in hits)
                 + f" | INR {inr:.0f} dB (SIR {25 - inr:.0f} dB) | {att} |")
    L += ["", "## Flows and DPIR candidates (repo diverse_paths, k = 2)", "",
          "| Flow | Demand | Candidates (= every simple path) | Hop-count | DPIR |", "|---|---|---|---|---|"]
    for i, (fid, a, b, dem) in enumerate(FLOWS):
        L.append(f"| {fid} {a}->{b} | {dem:.0f} Mb/s | {' / '.join('-'.join(p) for p in cands[fid])} | "
                 f"{h['paths'][i]} | {d['paths'][i]} |")
    L += ["", "## Joint routings at the nominal setting (LA model)", "", "| Routing | F1 | F2 | Total |", "|---|---|---|---|"]
    for r in s["rows"]:
        L.append(f"| {' + '.join(r['paths'])} | {r['thr'][0]} | {r['thr'][1]} | {r['total']:.0f} |")
    L += ["", "## Robustness", "",
          f"- Time-share bound (ideal per-slot adaptation, optimistic for the baselines): hop-count {s['ts']['hopcount']:.0f}, "
          f"DPIR {s['ts']['dpir']:.0f} Mb/s.",
          f"- Smoke test with only KW cabled: +{100 * s['kw_only']['gain_la']:.0f} % (LA) / "
          f"+{100 * s['kw_only']['gain_ts']:.0f} % (TS).",
          f"- Control (couplings off or LS on channel 1): hop-count {s['control']['hopcount']:.0f} Mb/s; DPIR keeps the "
          "direct spans.",
          f"- Live ramp: DPIR keeps the direct spans up to INR {s['switch_inr'] - 2} dB and moves F2 from "
          f"{s['switch_inr']} dB.",
          f"- Encore (INR {s['encore']['inr']} dB): fading LE to SNR {s['encore']['back_at']} dB makes DPIR put F2 back "
          "on the direct span.",
          "- IP-50 side-hop option: " + ", ".join(f"{w}: DPIR {v['dpir']:.0f} vs hop-count {v['hopcount']:.0f} Mb/s"
                                                  for w, v in s["ip50_variant"].items()) + ".", "",
          "| INR dB | victim SINR | hop-count | Open/R MCS | DPIR | DPIR routing |", "|---|---|---|---|---|---|"]
    for w in s["sweep"]:
        L.append(f"| {w['inr']} | {w['sinr']} | {w['hopcount']:.0f} | {w['openr']} | {w['dpir']:.0f} | "
                 f"{' + '.join(w['dpir_paths'])} |")
    L += ["", "## If idle links still transmit (QoS-Null)", "",
          "| Idle floor | Hop-count | DPIR route only | DPIR + unused coupled link down |", "|---|---|---|---|"]
    for name, v in s["idle"].items():
        L.append(f"| {name} ({100 * v['d0']:.2f} %) | {v['LA']['hopcount']:.0f} | {v['LA']['dpir_route_only']:.0f} | "
                 f"{v['LA']['dpir_link_off']:.0f} (down: {', '.join(v['LA']['down']) or '-'}) |")
    A, Bq = res["dynamic_silent"], res["dynamic_qosnull"]
    L += ["", "## Dynamic runs (120 s, 1 s routing step)", "",
          "Run A, idle links silent (F1 bursts, F2 constant):", "",
          "| Router | Delivered | F2 starved | Route changes |", "|---|---|---|---|"]
    names = dict(hopcount="Hop-count", openr_mcs="Open/R MCS metric (Terragraph default)",
                 reactive="Reactive capacity-aware (idle links report what they see)",
                 reactive_clean="Reactive capacity-aware (idle links report clean rate)",
                 dpir_lagged="DPIR without predictor (one step late)", dpir_predictive="DPIR with predictor")
    for k, v in A.items():
        if k == "offered":
            continue
        L.append(f"| {names[k]} | {v['delivered_pct']} % | {v['f2_starved_s']} s | {v['route_changes']} |")
    L += ["", f"Run B, worst case: QoS-Null floor, F1/F2 anti-phase, {T_IGN} s re-ignition:", "",
          "| Router | Delivered | Ignitions |", "|---|---|---|"]
    nb = dict(hopcount="Hop-count", dpir_route_only="DPIR, routing only",
              dpir_links_lagged="DPIR + link activation, one step late",
              dpir_links_predictive="DPIR + link activation, predictive (pre-ignites)")
    for k, v in Bq.items():
        if k == "offered":
            continue
        L.append(f"| {nb[k]} | {v['delivered_pct']} % | {v['ignitions']} |")
    f4 = res["four_flows"]
    L += ["", f"4-flow extension (16 joint routings): DPIR {f4['dpir']['total']:.0f} vs hop-count "
          f"{f4['hopcount']['total']:.0f} Mb/s.", "", "## Bench run", ""]
    L += [f"{i}. **{t}.** {x}" for i, (t, x) in enumerate(RUN_STEPS, 1)]
    L += ["", "## Open questions for the lab", ""] + [f"- **{t}.** {x}" for t, x in OPEN_QUESTIONS]
    L += ["", "## Model notes", "",
          "- Rates: Terragraph QTI 10.11 LA thresholds and CTF per-direction rates (Meta open source). Siklu's "
          "values may differ: calibrate on the bench before quoting gains.",
          "- Each sector is its own radio and each direction has its own subframe, so a relay costs no airtime and "
          "opposite directions share no capacity. Flows sharing a direction split it max-min fair.",
          "- The coupling is emulated: in this layout the coupled sectors do not face each other, so it stands for "
          "Terragraph's heuristic-4 interference or a reflection; its strength is set by ATT-5/6.",
          "- 'DPIR' here is the best joint routing under this model, i.e. what DPIR should find. Whether the "
          "GNN/DRL agent finds it from telemetry, and how early its predictor sees a burst, is what the bench "
          "measures. With 2 flows x 2 routes an exhaustive search also finds it; the bench shows the value of "
          "interference-awareness and prediction, not DPIR's edge over an exhaustive optimiser."]
    with open(os.path.join(OUT, "poc2_plan.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    H = ["# PoC 2 hardware against PoC 1 (generated by poc2.py)", "",
         "Fully conducted bench, IF cabling at the Spider boxes' SMA ports (about 13-17.4 GHz).", "",
         "| Group | Item | PoC 1 (A7) | PoC 2 | Note |", "|---|---|---|---|---|"]
    H += [f"| {g} | {i} | {a} | {b} | {n} |" for g, i, a, b, n in HARDWARE]
    with open(os.path.join(OUT, "poc2_hardware.md"), "w") as fh:
        fh.write("\n".join(H) + "\n")


def main():
    peers = design_checks()
    coupling_checks()
    cands = candidates(FLOWS)
    res = {"static": static_results(cands)}
    B = res["static"]["hopcount_opt"]
    res["dynamic_silent"] = dynamic_silent(cands, B)
    res["dynamic_qosnull"] = dynamic_qosnull(cands, B)
    res["four_flows"] = four_flows(cands)
    res["size"] = size_checks()
    write_outputs(res, cands, peers)
    bad = sum(not c["ok"] for c in CHECKS)
    print(f"{len(CHECKS)} checks, {bad} failed; wrote out/poc2_results.json, out/poc2_plan.md, out/poc2_hardware.md")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
