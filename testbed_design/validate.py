"""Derive and check every property the testbed drawings claim.

Run:  python3 validate.py      (exits non-zero if any check fails)
"""
import itertools
import math
import sys

import networkx as nx

from topologies import CONFIGS, POS, SCALE, ZONE_X, ZONE_Y

# ----------------------------------------------------------------------------
# 60 GHz link budget: the MARL paper's Table III model (60 GHz, 2160 MHz,
# 24 dBi per end, NF 8 dB, O2 15 dB/km, 2 dB implementation loss).  With the
# QPSK-1/3 threshold of -3 dB it reproduces the paper's derived range (1068 m
# at 23 dBm) to within 2 m.  The paper evaluates the mesh at the 23 dBm
# planning power only: the mesh radio's reach does not follow the power the
# policy commands, so no "boost" is assumed here.
# ----------------------------------------------------------------------------
MCS = [(-3, "QPSK 1/3", 0.667), (0, "QPSK 1/2", 1.0), (5, "16QAM", 2.0),
       (12, "64QAM", 3.0), (20, "256QAM", 4.0)]
TG_BW_HZ = 2160e6
P_PLAN = 23
OUTAGE_DB = -3.0
CN_HALF_SCAN = 45.0  # terminal-class unit: +/-45 deg around its mounting azimuth
MAX_STRETCH = 2      # DPIR candidates: at most 2 hops longer than the shortest


def tg_snr(d, p_dbm=P_PLAN):
    fspl = 20 * math.log10(d) + 20 * math.log10(60000) - 27.55 + 0.015 * d
    noise = -174 + 10 * math.log10(TG_BW_HZ) + 8
    return p_dbm + 48 - fspl - noise - 2


def tg_budget(d):
    s = tg_snr(d)
    best = None
    for thr, name, se in MCS:
        if s >= thr:
            best = (name, se)
    return dict(snr=s, margin=s - OUTAGE_DB, mcs=best[0] if best else None,
                gbps=best[1] * TG_BW_HZ / 1e9 if best else 0.0)


def bearing(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def dist(a, b):
    """Metres between two drawing positions."""
    return SCALE * math.hypot(a[0] - b[0], a[1] - b[1])


def in_zone(xy):
    return ZONE_X[0] <= xy[0] <= ZONE_X[1] and ZONE_Y[0] <= xy[1] <= ZONE_Y[1]


def sector(kind, b):
    """Sector of a TG unit used toward bearing b: node-class units have four
    90-degree sectors (0 = east, 1 = north, 2 = west, 3 = south); a
    terminal-class unit has one."""
    return 0 if kind == "CN" else int(((b + 45) % 360) // 90)


def build_graph(cfg):
    g = nx.Graph()
    g.add_nodes_from(cfg["sites"])
    for a, b in cfg["mw"]:
        g.add_edge(a, b, tech="MW")
    for a, b in cfg["tg_links"]:
        g.add_edge(a, b, tech="TG")
    for a, b in cfg.get("tg_fixed", []):
        g.add_edge(a, b, tech="TG", fixed=True)
    return g


def diverse_paths(g, s, t, k=2):
    """DPIR-style diverse candidates (Greidi et al., Sec. III-B): repeated
    Dijkstra in which a link's weight grows with its overlap with, and
    proximity to, the paths already chosen; candidates longer than the
    shortest path by more than MAX_STRETCH hops are not used."""
    if not nx.has_path(g, s, t):
        return []
    hmin = nx.shortest_path_length(g, s, t)
    used_e, used_n, paths = set(), set(), []
    for _ in range(4 * k):
        def w(u, v, d):
            c = 1.0
            if frozenset((u, v)) in used_e:
                c += 4.0
            c += 1.5 * sum(1 for x in (u, v) if x in used_n and x not in (s, t))
            return c
        p = nx.dijkstra_path(g, s, t, weight=w)
        if p in paths:
            break
        used_e |= {frozenset(e) for e in zip(p, p[1:])}
        used_n |= set(p)
        if len(p) - 1 <= hmin + MAX_STRETCH:
            paths.append(p)
        if len(paths) == k:
            break
    return paths


def path_tech(g, p):
    return [g.edges[a, b]["tech"] for a, b in zip(p, p[1:])]


def analyze(key):
    cfg = dict(CONFIGS[key])
    pos = {s: POS[s] for s in cfg["sites"]}
    checks = []

    def check(ok, msg):
        checks.append((bool(ok), msg))

    # ---- equipment --------------------------------------------------------
    ip_units = {s: 0 for s in cfg["sites"]}
    for a, b in cfg["mw"]:
        ip_units[a] += 1
        ip_units[b] += 1
    check(sum(ip_units.values()) == cfg["ip50"],
          f"IP-50 units = {sum(ip_units.values())} (budget {cfg['ip50']}, 2 per microwave link)")
    check(len(cfg["tg_units"]) == cfg["tg"], f"TG units = {len(cfg['tg_units'])} (budget {cfg['tg']})")
    tg_all = list(cfg["tg_links"]) + list(cfg.get("tg_fixed", []))
    for a, b in cfg.get("tg_fixed", []):
        check(all(x in {y for e in cfg["tg_links"] for y in e} for x in (a, b)),
              f"fixed link {a}-{b} uses a spare sector of TG nodes installed for steering (no extra radio)")
    for a, b in tg_all:
        check(a in cfg["tg_units"] and b in cfg["tg_units"], f"TG link {a}-{b} has a TG unit at both ends")
        check("DN" in (cfg["tg_units"][a], cfg["tg_units"][b]), f"TG link {a}-{b} has a node-class end (no CN-CN)")
        bud = tg_budget(dist(pos[a], pos[b]))
        check(bud["margin"] >= 10, f"TG link {a}-{b} ({dist(pos[a], pos[b]):.0f} m): {bud['mcs']}, "
              f"{bud['margin']:.0f} dB above outage at {P_PLAN} dBm")
    cn_mount = {}
    for s, kind in cfg["tg_units"].items():
        peers = [b if a == s else a for a, b in tg_all if s in (a, b)]
        if kind == "CN":
            check(len(peers) == 1, f"terminal-class unit at {s} holds a single link")
            cn_mount[s] = bearing(pos[s], pos[peers[0]])

    # TG sector of every link end (used for airtime sharing and polarity)
    def end_sector(s, peer):
        return (s, sector(cfg["tg_units"][s], bearing(pos[s], pos[peer])))
    shared = {}
    for a, b in tg_all:
        for x, y in ((a, b), (b, a)):
            shared.setdefault(end_sector(x, y), []).append((x, y))
    shared = {k: v for k, v in shared.items() if len(v) >= 2 and cfg["tg_units"][k[0]] == "DN"}

    # ---- before the event: DPIR ------------------------------------------
    g0 = build_graph(cfg)
    check(nx.is_connected(g0), "pre-event network is connected")
    dpir, covered = {}, set()
    for fid, s, t in cfg["flows"]:
        ps = diverse_paths(g0, s, t)
        dpir[fid] = ps
        check(len(ps) == 2, f"DPIR {fid} {s}->{t} has 2 candidate paths")
        if len(ps) == 2:
            e0 = {frozenset(e) for e in zip(ps[0], ps[0][1:])}
            e1 = {frozenset(e) for e in zip(ps[1], ps[1][1:])}
            check(not (e0 & e1), f"DPIR {fid}: the two candidates share no link")
        for p in ps:
            covered |= {frozenset(e) for e in zip(p, p[1:])}
    service = {}
    for s, t in cfg.get("service", []):
        service[(s, t)] = nx.shortest_path(g0, s, t)
        p = service[(s, t)]
        covered |= {frozenset(e) for e in zip(p, p[1:])}
    links = {frozenset(e) for e in cfg["mw"] + tg_all}
    check(not (links - covered), "every link is on a DPIR candidate or a pocket's service path"
          + (f" (unused: {[tuple(e) for e in links - covered]})" if links - covered else ""))

    def contention_keys(paths):
        keys = set()
        for p in paths:
            for a, b in zip(p, p[1:]):
                keys.add(("link", frozenset((a, b))))
                if g0.edges[a, b]["tech"] == "TG":
                    for x, y in ((a, b), (b, a)):
                        if end_sector(x, y) in shared:
                            keys.add(("sector",) + end_sector(x, y))
        return keys
    for fid in dpir:
        mine = contention_keys(dpir[fid])
        others = set()
        for f, ps in dpir.items():
            if f != fid:
                others |= contention_keys(ps)
        check({k for k in mine & others if k[0] == "sector"},
              f"DPIR {fid} contends with another flow for TG sector airtime")

    # ---- the event -----------------------------------------------------------
    zone_sites = [s for s in cfg["sites"] if in_zone(pos[s])]
    check(sorted(zone_sites) == sorted(cfg["destroyed"]),
          f"destroyed sites {cfg['destroyed']} are exactly the sites in the zone")
    check("CORE" in cfg["destroyed"], "the core is severed by the event")
    g1 = g0.copy()
    g1.remove_nodes_from(cfg["destroyed"])
    islands = [sorted(c) for c in nx.connected_components(g1)]
    islands.sort(key=lambda c: (0 if "W1" in c else 1, -len(c), -pos[c[0]][1]))
    comp = {s: i for i, c in enumerate(islands) for s in c}
    check(len(islands) >= 2, f"event leaves {len(islands)} islands")
    for fid, s, t in cfg["flows"]:
        check(comp[s] != comp[t], f"DPIR {fid} {s}->{t} is cut")
        for p in dpir[fid]:
            check(any(n in cfg["destroyed"] for n in p[1:-1]), f"{fid} candidate {'-'.join(p)} is broken")
    lost_ip = sum(ip_units[s] for s in cfg["destroyed"])
    lost_tg = sum(1 for s in cfg["destroyed"] if s in cfg["tg_units"])

    # ---- MARL bridge candidates --------------------------------------------
    survivors = [s for s in cfg["tg_units"] if s not in cfg["destroyed"]]
    cands = []
    for a, b in itertools.combinations(survivors, 2):
        if comp[a] == comp[b]:
            continue
        d = dist(pos[a], pos[b])
        bud = tg_budget(d)
        why = []
        if cfg["tg_units"][a] == "CN" and cfg["tg_units"][b] == "CN":
            why.append("CN-CN not allowed")
        for x, y in ((a, b), (b, a)):
            if cfg["tg_units"][x] == "CN" and angdiff(bearing(pos[x], pos[y]), cn_mount[x]) > CN_HALF_SCAN:
                why.append(f"{x}: outside the terminal's +/-45 deg scan")
        if bud["mcs"] is None:
            why.append(f"out of range at {P_PLAN} dBm")
        cands.append(dict(a=a, b=b, d=d, budget=bud, feasible=not why, why=why))
    feas = [c for c in cands if c["feasible"]]
    check(feas, f"{len(feas)} feasible bridge candidates")
    for i, isl in enumerate(islands):
        check(any(s in cfg["tg_units"] for s in isl), f"island {i + 1} ({', '.join(isl)}) has a TG unit")
    can_steer = {x for c in feas for x in (c["a"], c["b"])}
    for s in cfg["tg_units"]:
        if s in cfg["destroyed"]:
            served = [b if a == s else a for a, b in cfg["tg_links"] if s in (a, b)]
            check(all(x in can_steer for x in served),
                  f"TG at {s} (lost) serves only sites that can re-steer once it is gone: {', '.join(served)}")
        else:
            check(s in can_steer, f"TG at {s} can steer to a feasible bridge peer")

    # ---- recovery plan --------------------------------------------------------
    # Which sites engage is the policy's decision; the peer of an engaged head
    # is the bridge-peer planner's (an engine service, paper Sec. IV-D).  The
    # figures show one plan that obeys the paper's rules: every island joined
    # with no redundant bridge, and one new link per site (an engaged head
    # holds one steerable link).  The ranking used to pick it is given below.
    # The initiator is a node-class end (Terragraph: a CN cannot initiate).
    need = len(islands) - 1
    plans = []
    for combo in itertools.combinations(feas, need):
        ends = [x for c in combo for x in (c["a"], c["b"])]
        if len(set(ends)) != len(ends):
            continue
        h = nx.Graph()
        h.add_nodes_from(range(len(islands)))
        h.add_edges_from((comp[c["a"]], comp[c["b"]]) for c in combo)
        if nx.is_connected(h):
            plans.append(combo)
    check(plans, f"{len(plans)} valid recovery plans (every island joined, one new link per site)")

    def turn(x, y):
        """Turn of x's beam toward y if the same sector served a destroyed peer."""
        b_new = bearing(pos[x], pos[y])
        sec = end_sector(x, y)
        dead = [b if a == x else a for a, b in tg_all if x in (a, b)]
        dead = [o for o in dead if o in cfg["destroyed"] and end_sector(x, o) == sec]
        return min((angdiff(bearing(pos[x], pos[o]), b_new) for o in dead), default=None)

    def still(p):
        """Bridge ends whose beam would barely move (bridge lies along a dead link)."""
        return sum(1 for c in p for x, y in ((c["a"], c["b"]), (c["b"], c["a"]))
                   if turn(x, y) is not None and turn(x, y) < 10)
    # Prefer plans whose weakest bridge keeps >= 7.5 dB margin, then plans in
    # which every steering beam visibly turns, then the strongest weakest link.
    plans.sort(key=lambda p: (min(c["budget"]["margin"] for c in p) < 7.5, still(p),
                              -round(min(c["budget"]["snr"] for c in p), 3),
                              -round(sum(c["budget"]["snr"] for c in p), 3),
                              sorted(tuple(sorted((c["a"], c["b"]))) for c in p)))
    plan = list(plans[0]) if plans else []

    def initiator(c):
        dns = [x for x in (c["a"], c["b"]) if cfg["tg_units"][x] == "DN"]
        return min(dns, key=lambda x: (-len(islands[comp[x]]), x))
    bridges = []
    for c in sorted(plan, key=lambda c: (comp[c["a"]] + comp[c["b"]], c["d"])):
        i = initiator(c)
        bridges.append((i, c["b"] if i == c["a"] else c["a"]))
    cfg["bridges"] = bridges

    g2 = g1.copy()
    g2.add_edges_from(((a, b, {"tech": "BR"}) for a, b in bridges))
    check(nx.is_connected(g2), "after the bridges every surviving site is in one network")

    # beam bookkeeping per (site, sector)
    steer = []
    for ini, rsp in bridges:
        for x, y in ((ini, rsp), (rsp, ini)):
            kind_u = cfg["tg_units"][x]
            b_new = bearing(pos[x], pos[y])
            sec = sector(kind_u, b_new)
            pre = [b if a == x else a for a, b in tg_all if x in (a, b)]
            pre = [o for o in pre if end_sector(x, o) == (x, sec)]
            dead = [o for o in pre if o in cfg["destroyed"]]
            live = [o for o in pre if o not in cfg["destroyed"]]
            if dead:
                ref = min(dead, key=lambda o: angdiff(bearing(pos[x], pos[o]), b_new))
                b_old = bearing(pos[x], pos[ref])
                change = "re-steer"
            else:
                ref, b_old = None, None
                change = "shares a sector with " + ", ".join(live) if live else "idle sector"
            steer.append(dict(unit=x, old=ref, new=y, b_old=b_old, b_new=b_new, sector=sec,
                              delta=angdiff(b_old, b_new) if ref else None, change=change,
                              role="initiator" if x == ini else "responder"))

    # geometric slack: no link end within 5 deg of a sector edge / scan limit
    for a, b in tg_all + bridges:
        for x, y in ((a, b), (b, a)):
            bb = bearing(pos[x], pos[y])
            if cfg["tg_units"][x] == "CN":
                slack = CN_HALF_SCAN - angdiff(bb, cn_mount[x])
            else:
                slack = 45 - angdiff(bb, sector(cfg["tg_units"][x], bb) * 90)
            check(slack >= 5, f"{x} toward {y}: {slack:.1f} deg from the sector edge / scan limit")

    # ---- Terragraph polarity --------------------------------------------------
    # Every TG link must join an odd and an even sector.  The core (and the
    # controller) is gone after the event, so the polarity plan has to be set
    # beforehand and must already suit the bridges.  Multi-sector units may
    # mix polarities across sectors (hybrid polarity).
    pg = nx.Graph()
    for a, b in tg_all + bridges:
        pg.add_edge(end_sector(a, b), end_sector(b, a))
    bip = nx.is_bipartite(pg)
    check(bip, "a per-sector odd/even polarity plan supports every pre-event link and every bridge")
    polarity = {}
    if bip:
        for cc in nx.connected_components(pg):
            col = nx.bipartite.color(pg.subgraph(cc))
            flip = col[sorted(cc)[0]]
            for v, c in col.items():
                polarity[v] = "odd" if c == flip else "even"

    post = {}
    for fid, s, t in cfg["flows"]:
        post[fid] = nx.shortest_path(g2, s, t)
    for (s, t) in service:
        post[(s, t)] = nx.shortest_path(g2, s, t)

    return dict(key=key, cfg=cfg, pos=pos, ip_units=ip_units, checks=checks, dpir=dpir,
                service=service, g0=g0, islands=islands, comp=comp, lost_ip=lost_ip,
                lost_tg=lost_tg, cands=cands, steer=steer, g_post=g2, post=post,
                shared=shared, polarity=polarity, cn_mount=cn_mount, end_sector=end_sector,
                n_plans=len(plans), tg_all=tg_all,
                path_tech=lambda p, g=g0: path_tech(g, p))


def main():
    bad = 0
    for key in CONFIGS:
        r = analyze(key)
        print(f"=== {r['cfg']['name']} ===")
        for ok, msg in r["checks"]:
            print(("  PASS  " if ok else "  FAIL  ") + msg)
            bad += not ok
        print("  islands:", r["islands"])
        print("  bridges (initiator -> responder):", r["cfg"]["bridges"], f"[{r['n_plans']} valid plans]")
        for fid, ps in r["dpir"].items():
            for p in ps:
                print(f"  {fid}: {'-'.join(p)}  {r['path_tech'](p)}")
        for k, v in r["shared"].items():
            print(f"  shared sector {k[0]}.s{k[1]}: {v}")
        for c in r["cands"]:
            b = c["budget"]
            print(f"  cand {c['a']}-{c['b']} {c['d']:.0f} m  {b['mcs']}  margin {b['margin']:.1f} dB"
                  f"  {'OK' if c['feasible'] else 'X ' + '; '.join(c['why'])}")
        for s in r["steer"]:
            d = f"{s['delta']:.0f} deg" if s["delta"] is not None else ""
            print(f"  beam {s['unit']} ({s['role']}, sector {s['sector']}): {s['change']} -> {s['new']} {d}")
        print("  polarity:", {f"{k[0]}.s{k[1]}": v for k, v in sorted(r["polarity"].items())})
    print("FAILURES:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
