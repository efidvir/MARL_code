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
    for a, b in cfg["tg_links"]:
        check(a in cfg["tg_units"] and b in cfg["tg_units"], f"TG link {a}-{b} has a TG unit at both ends")
        check("DN" in (cfg["tg_units"][a], cfg["tg_units"][b]), f"TG link {a}-{b} has a node-class end (no CN-CN)")
        bud = tg_budget(dist(pos[a], pos[b]))
        check(bud["margin"] >= 10, f"TG link {a}-{b} ({dist(pos[a], pos[b]):.0f} m): {bud['mcs']}, "
              f"{bud['margin']:.0f} dB above outage at {P_PLAN} dBm")
    cn_mount = {}
    for s, kind in cfg["tg_units"].items():
        peers = [b if a == s else a for a, b in cfg["tg_links"] if s in (a, b)]
        if kind == "CN":
            check(len(peers) == 1, f"terminal-class unit at {s} holds a single link")
            cn_mount[s] = bearing(pos[s], pos[peers[0]])

    # TG sector of every link end (used for airtime sharing and polarity)
    def end_sector(s, peer):
        return (s, sector(cfg["tg_units"][s], bearing(pos[s], pos[peer])))
    shared = {}
    for a, b in cfg["tg_links"]:
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
        check(len(ps) == 2, f"DPIR {fid} {s}->{t} has 2 diverse candidate paths")
        for p in ps:
            covered |= {frozenset(e) for e in zip(p, p[1:])}
    service = {}
    for s, t in cfg.get("service", []):
        service[(s, t)] = nx.shortest_path(g0, s, t)
        p = service[(s, t)]
        covered |= {frozenset(e) for e in zip(p, p[1:])}
    links = {frozenset(e) for e in cfg["mw"] + cfg["tg_links"]}
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
        check(mine & others, f"DPIR {fid} contends with another flow (shared link or shared TG sector airtime)")

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
                  f"TG at {s} (lost) serves only sites that re-steer once it is gone: {', '.join(served)}")
        else:
            check(s in can_steer, f"TG at {s} can steer to a bridge peer")

    # ---- recovery plan --------------------------------------------------------
    # Greedy highest-capacity tree over the islands (SNR at the planning
    # power, then prefer joining W1's network, then the shorter hop); a
    # terminal holds one link.  Oriented from W1's island, so each cut-off
    # island's head points back at the network it joins: one bridge per head.
    root = comp["W1"]
    uf = {i: i for i in range(len(islands))}

    def find(i):
        while uf[i] != i:
            i = uf[i]
        return i
    tree = []
    while len({find(i) for i in uf}) > 1:
        used_cn = {x for c in tree for x in (c["a"], c["b"]) if cfg["tg_units"][x] == "CN"}
        opts = [c for c in feas if find(comp[c["a"]]) != find(comp[c["b"]])
                and not ({c["a"], c["b"]} & used_cn)]
        check(opts, "a bridge exists for every remaining island")
        if not opts:
            break
        best = min(opts, key=lambda c: (
            -round(c["budget"]["snr"], 3),
            0 if find(root) in (find(comp[c["a"]]), find(comp[c["b"]])) else 1,
            round(c["d"]), tuple(sorted((c["a"], c["b"])))))
        tree.append(best)
        uf[find(comp[best["a"]])] = find(comp[best["b"]])
    adj = {}
    for c in tree:
        adj.setdefault(comp[c["a"]], []).append(c)
        adj.setdefault(comp[c["b"]], []).append(c)
    bridges, seen, queue = [], {root}, [root]
    while queue:
        i = queue.pop(0)
        for c in sorted(adj.get(i, []), key=lambda c: c["d"]):
            j = comp[c["b"]] if comp[c["a"]] == i else comp[c["a"]]
            if j in seen:
                continue
            seen.add(j)
            queue.append(j)
            head = c["a"] if comp[c["a"]] == j else c["b"]
            bridges.append((head, c["b"] if head == c["a"] else c["a"]))
    cfg["bridges"] = bridges

    g2 = g1.copy()
    steer, heads = [], set()
    for head, peer in bridges:
        check(head not in heads, f"head {head} forms one bridge (paper Sec. IV-D)")
        heads.add(head)
        check(not g0.has_edge(head, peer), f"bridge {head}-{peer} is a new link")
        check(not nx.has_path(g2, head, peer), f"bridge {head}-{peer} joins two islands")
        g2.add_edge(head, peer, tech="BR")
        for x, y in ((head, peer), (peer, head)):
            b_new = bearing(pos[x], pos[y])
            dead = [b if a == x else a for a, b in cfg["tg_links"] if x in (a, b)]
            dead = [o for o in dead if o in cfg["destroyed"]]
            ref = min(dead, key=lambda o: angdiff(bearing(pos[x], pos[o]), b_new)) if dead else None
            b_old = bearing(pos[x], pos[ref]) if ref else None
            steer.append(dict(unit=x, old=ref, new=y, b_old=b_old, b_new=b_new,
                              delta=angdiff(b_old, b_new) if ref else None,
                              role="head" if x == head else "peer",
                              same_sector=bool(ref) and sector(cfg["tg_units"][x], b_old)
                              == sector(cfg["tg_units"][x], b_new)))
    for s, kind in cfg["tg_units"].items():
        if kind == "CN":
            check(sum(s in b for b in bridges) <= 1, f"terminal-class unit at {s} is in at most one bridge")
    check(nx.is_connected(g2), "after the bridges every surviving site is in one network")
    check(len(bridges) == len(islands) - 1, f"{len(bridges)} bridges for {len(islands)} islands")

    # ---- Terragraph polarity --------------------------------------------------
    # Every TG link must join an odd and an even sector.  The core (and the
    # controller) is gone after the event, so the polarity plan has to be set
    # beforehand and must already suit the bridges.  Multi-sector units may
    # mix polarities across sectors (hybrid polarity).
    pg = nx.Graph()
    for a, b in list(cfg["tg_links"]) + bridges:
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
        print("  bridges (head -> peer):", r["cfg"]["bridges"])
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
            d = f"{s['delta']:.0f} deg" if s["delta"] is not None else "-"
            print(f"  beam {s['unit']} ({s['role']}): {s['old']} -> {s['new']}  {d}")
        print("  polarity:", {f"{k[0]}.s{k[1]}": v for k, v in sorted(r["polarity"].items())})
    print("FAILURES:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
