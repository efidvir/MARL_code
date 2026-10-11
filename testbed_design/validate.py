"""Derive and check every property the testbed drawings claim.

Run:  python3 validate.py      (exits non-zero if any check fails)
"""
import itertools
import math
import sys

import networkx as nx

from topologies import CONFIGS, POS, ZONE_X

# ----------------------------------------------------------------------------
# 60 GHz link budget: the MARL paper's Table III model (60 GHz, 2160 MHz,
# 24 dBi per end, NF 8 dB, O2 15 dB/km, 2 dB implementation loss).  With the
# QPSK-1/3 threshold of -3 dB it reproduces the paper's derived ranges
# (1068 m at 23 dBm, 1528 m at 33 dBm) to within 2 m.
# ----------------------------------------------------------------------------
MCS = [(-3, "QPSK 1/3", 0.667), (0, "QPSK 1/2", 1.0), (5, "16QAM", 2.0),
       (12, "64QAM", 3.0), (20, "256QAM", 4.0)]
TG_BW_HZ = 2160e6
P_PLAN, P_BOOST = 23, 33
CN_HALF_SCAN = 45.0  # terminal-class unit: +/-45 deg around its mounting azimuth
ZONE_Y = (-800, 800)


def tg_snr(d, p_dbm):
    fspl = 20 * math.log10(d) + 20 * math.log10(60000) - 27.55 + 0.015 * d
    noise = -174 + 10 * math.log10(TG_BW_HZ) + 8
    return p_dbm + 48 - fspl - noise - 2


def mcs(snr):
    best = None
    for thr, name, se in MCS:
        if snr >= thr:
            best = (name, se)
    return best  # None = outage


def tg_budget(d):
    out = {}
    for p in (P_PLAN, P_BOOST):
        s = tg_snr(d, p)
        m = mcs(s)
        out[p] = dict(snr=s, mcs=m[0] if m else None,
                      gbps=(m[1] * TG_BW_HZ / 1e9) if m else 0.0)
    return out


def bearing(a, b):
    (x1, y1), (x2, y2) = a, b
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def in_zone(xy):
    return ZONE_X[0] <= xy[0] <= ZONE_X[1] and ZONE_Y[0] <= xy[1] <= ZONE_Y[1]


def positions(cfg):
    p = {s: POS[s] for s in cfg["sites"]}
    p.update(cfg.get("pos_override", {}))
    return p


def build_graph(cfg, extra_tg=()):
    g = nx.MultiGraph()
    g.add_nodes_from(cfg["sites"])
    for a, b in cfg["mw"]:
        g.add_edge(a, b, tech="MW")
    for a, b in cfg["tg_links"]:
        g.add_edge(a, b, tech="TG")
    for a, b in extra_tg:
        g.add_edge(a, b, tech="BR")
    return g


def simple(g):
    """Collapse a MultiGraph to a Graph, keeping the tech of each edge."""
    h = nx.Graph()
    h.add_nodes_from(g.nodes)
    for a, b, d in g.edges(data=True):
        h.add_edge(a, b, tech=d["tech"])
    return h


def diverse_paths(g, s, t, k=3):
    """DPIR-style diverse candidate paths (Greidi et al., Sec. III-B):
    repeated Dijkstra in which each link's weight grows with its overlap and
    proximity to the paths already chosen."""
    h = simple(g)
    if not nx.has_path(h, s, t):
        return []
    used_e, used_n, paths = set(), set(), []
    for _ in range(k):
        def w(u, v, d):
            e = frozenset((u, v))
            c = 1.0
            if e in used_e:
                c += 4.0
            if u in used_n and u not in (s, t):
                c += 1.5
            if v in used_n and v not in (s, t):
                c += 1.5
            return c
        p = nx.dijkstra_path(h, s, t, weight=w)
        if p in paths:
            break
        paths.append(p)
        used_e |= {frozenset(e) for e in zip(p, p[1:])}
        used_n |= set(p)
    return paths


def path_tech(g, p):
    h = simple(g)
    return [h.edges[a, b]["tech"] for a, b in zip(p, p[1:])]


def analyze(key):
    cfg = CONFIGS[key]
    pos = positions(cfg)
    checks = []

    def check(ok, msg):
        checks.append((bool(ok), msg))

    # ---- equipment counts ------------------------------------------------
    ip_units = {s: 0 for s in cfg["sites"]}
    for a, b in cfg["mw"]:
        ip_units[a] += 1
        ip_units[b] += 1
    n_ip = sum(ip_units.values())
    n_tg = len(cfg["tg_units"])
    check(n_ip == cfg["ip50"], f"IP-50 units = {n_ip} (budget {cfg['ip50']}, 2 per microwave link)")
    check(n_tg == cfg["tg"], f"TG units = {n_tg} (budget {cfg['tg']})")
    for a, b in cfg["tg_links"]:
        check(a in cfg["tg_units"] and b in cfg["tg_units"], f"TG link {a}-{b} has a TG unit at both ends")
        check("DN" in (cfg["tg_units"][a], cfg["tg_units"][b]),
              f"TG link {a}-{b} has a DN end (Terragraph: no CN-CN links)")
    deg_tg = {s: 0 for s in cfg["tg_units"]}
    for a, b in cfg["tg_links"]:
        deg_tg[a] += 1
        deg_tg[b] += 1
    for s, kind in cfg["tg_units"].items():
        check(deg_tg[s] >= 1, f"TG unit at {s} carries a pre-event link")
        if kind == "CN":
            check(deg_tg[s] == 1, f"CN at {s} holds a single link")
    for a, b in cfg["tg_links"]:
        d = dist(pos[a], pos[b])
        check(tg_snr(d, P_PLAN) >= -3, f"TG link {a}-{b} ({d:.0f} m) closes at {P_PLAN} dBm")

    # CN mounting azimuth = bearing to its pre-event peer
    cn_mount = {}
    for s, kind in cfg["tg_units"].items():
        if kind == "CN":
            peer = [b if a == s else a for a, b in cfg["tg_links"] if s in (a, b)][0]
            cn_mount[s] = bearing(pos[s], pos[peer])

    # ---- pre-event graph and DPIR ---------------------------------------
    g0 = build_graph(cfg)
    check(nx.is_connected(simple(g0)), "pre-event network is connected")
    dpir = {}
    covered = set()
    for fid, s, t in cfg["flows"]:
        ps = diverse_paths(g0, s, t, k=3)
        dpir[fid] = ps
        check(len(ps) >= 2, f"{fid} {s}->{t} has >= 2 DPIR candidate paths ({len(ps)})")
        for p in ps:
            covered |= {frozenset(e) for e in zip(p, p[1:])}
    all_links = {frozenset(e) for e in cfg["mw"]} | {frozenset(e) for e in cfg["tg_links"]}
    unused = all_links - covered
    check(not unused, "every pre-event link lies on a DPIR candidate path"
          + (f" (unused: {sorted(tuple(sorted(e)) for e in unused)})" if unused else ""))

    # coupling: links sharing a DN (airtime of a PtMP node) or a microwave hub
    coupling = []
    for s, kind in cfg["tg_units"].items():
        ls = [tuple(e) for e in cfg["tg_links"] if s in e]
        if kind == "DN" and len(ls) >= 2:
            coupling.append((s, "TG node shares airtime across", ls))
    for s, n in ip_units.items():
        if n >= 2:
            ls = [tuple(e) for e in cfg["mw"] if s in e]
            coupling.append((s, "co-sited IP-50 links (co-channel planning)", ls))
    for fid, s, t in cfg["flows"]:
        others = [p for f2, ps in dpir.items() if f2 != fid for p in ps]
        mine = {frozenset(e) for p in dpir[fid] for e in zip(p, p[1:])}
        shared = any(mine & {frozenset(e) for e in zip(p, p[1:])} for p in others)
        check(shared, f"{fid} shares at least one candidate link with another flow (multi-flow coupling)")

    # ---- the event ------------------------------------------------------
    zone_sites = [s for s in cfg["sites"] if in_zone(pos[s])]
    check(sorted(zone_sites) == sorted(cfg["destroyed"]),
          f"destroyed sites {cfg['destroyed']} are exactly the sites in the zone {zone_sites}")
    g1 = simple(g0).copy()
    g1.remove_nodes_from(cfg["destroyed"])
    islands = [sorted(c) for c in nx.connected_components(g1)]
    islands.sort(key=lambda c: (0 if "CORE" in c else 1, -len(c), -pos[c[0]][1]))
    comp = {s: i for i, c in enumerate(islands) for s in c}
    check(len(islands) >= 2, f"event leaves {len(islands)} islands")
    for fid, s, t in cfg["flows"]:
        check(comp[s] != comp[t], f"{fid} {s}->{t} is cut by the event")
    lost_links = [e for e in list(cfg["mw"]) + list(cfg["tg_links"])
                  if e[0] in cfg["destroyed"] or e[1] in cfg["destroyed"]]
    lost_ip = sum(ip_units[s] for s in cfg["destroyed"])
    lost_tg = sum(1 for s in cfg["destroyed"] if s in cfg["tg_units"])
    for fid, ps in dpir.items():
        for p in ps:
            check(any(n in cfg["destroyed"] for n in p[1:-1]),
                  f"{fid} candidate {'-'.join(p)} is broken by the event")

    # ---- MARL recovery candidates --------------------------------------
    survivors = [s for s in cfg["tg_units"] if s not in cfg["destroyed"]]

    def cn_can_engage(s, peer_xy, graph):
        if cfg["tg_units"][s] != "CN":
            return True, ""
        if angdiff(bearing(pos[s], peer_xy), cn_mount[s]) > CN_HALF_SCAN:
            return False, "outside the terminal's +/-45 deg scan"
        own = [(a, b) for a, b in cfg["tg_links"] if s in (a, b)
               and a not in cfg["destroyed"] and b not in cfg["destroyed"]]
        for a, b in own:  # the terminal must give up its current link
            h = graph.copy()
            h.remove_edge(a, b)
            if not nx.has_path(h, a, b):
                return False, f"would drop {a}-{b}, the sole path between them"
        return True, ""

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
            ok, r = cn_can_engage(x, pos[y], g1)
            if not ok:
                why.append(f"{x}: {r}")
        if bud[P_BOOST]["mcs"] is None:
            why.append("out of range even at 33 dBm")
        cands.append(dict(a=a, b=b, d=d, budget=bud, feasible=not why, why=why,
                          needs_boost=bud[P_PLAN]["mcs"] is None))
    feas = [c for c in cands if c["feasible"]]
    check(feas, f"{len(feas)} feasible bridge candidates")

    # ---- chosen recovery -------------------------------------------------
    heads_used = set()
    g2 = g1.copy()
    steer = []
    for head, peer in cfg["bridges"]:
        c = [c for c in feas if {c["a"], c["b"]} == {head, peer}]
        check(c, f"bridge {head}->{peer} is a feasible candidate")
        check(head not in heads_used, f"head {head} forms at most one bridge (paper Sec. IV-D)")
        heads_used.add(head)
        check(not g0.has_edge(head, peer), f"bridge {head}-{peer} did not exist before the event")
        check(not nx.has_path(g2, head, peer), f"bridge {head}-{peer} joins two different components")
        # Terragraph ignition: a DN end must already be reachable from CORE
        initiators = [x for x in (head, peer) if cfg["tg_units"][x] == "DN"
                      and nx.has_path(g2, x, "CORE")]
        check(initiators, f"bridge {head}-{peer} has a DN end reachable from the controller at CORE")
        # CN drops its old link
        if cfg["tg_units"][head] == "CN":
            for a, b in [(a, b) for a, b in cfg["tg_links"] if head in (a, b)]:
                if g2.has_edge(a, b):
                    g2.remove_edge(a, b)
        g2.add_edge(head, peer, tech="BR")
        for x, y in ((head, peer), (peer, head)):
            old = [b if a == x else a for a, b in cfg["tg_links"] if x in (a, b)]
            old_dead = [o for o in old if o in cfg["destroyed"]]
            b_new = bearing(pos[x], pos[y])
            if old_dead:
                # the beam that served a destroyed peer swings to the new peer
                ref = min(old_dead, key=lambda o: angdiff(bearing(pos[x], pos[o]), b_new))
                b_old = bearing(pos[x], pos[ref])
                steer.append(dict(unit=x, old=ref, new=y, b_old=b_old, b_new=b_new,
                                  delta=angdiff(b_old, b_new), kind="re-steer", role=(
                                      "head" if x == head else "peer")))
            else:
                # node-class unit whose links all survive: a free sector takes the new peer
                steer.append(dict(unit=x, old=None, new=y, b_old=None, b_new=b_new,
                                  delta=None, kind="new sector", role=(
                                      "head" if x == head else "peer")))
    check(nx.is_connected(g2), "after the chosen bridges every surviving site is in one network")
    check(len(cfg["bridges"]) == len(islands) - 1,
          f"{len(cfg['bridges'])} bridges for {len(islands)} islands (no redundant bridge)")

    g2m = nx.MultiGraph(g2)
    post = {}
    for fid, s, t in cfg["flows"]:
        post[fid] = diverse_paths(g2m, s, t, k=2)
        check(post[fid], f"{fid} restored after recovery: {'-'.join(post[fid][0]) if post[fid] else '-'}")

    return dict(key=key, cfg=cfg, pos=pos, ip_units=ip_units, checks=checks,
                dpir=dpir, g0=g0, islands=islands, comp=comp, lost_links=lost_links,
                lost_ip=lost_ip, lost_tg=lost_tg, cands=cands, steer=steer,
                g_post=g2, post=post, coupling=coupling, cn_mount=cn_mount,
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
        for fid, ps in r["dpir"].items():
            for p in ps:
                print(f"  {fid}: {'-'.join(p)}  {r['path_tech'](p)}")
        for c in r["cands"]:
            b = c["budget"]
            print(f"  cand {c['a']}-{c['b']} {c['d']:.0f} m  23dBm {b[23]['mcs']}  33dBm {b[33]['mcs']}"
                  f"  {'OK' if c['feasible'] else 'X ' + '; '.join(c['why'])}")
        for s in r["steer"]:
            d = f"delta {s['delta']:.0f} deg" if s["delta"] is not None else "new sector"
            print(f"  steer {s['unit']} ({s['role']}): {s['old']}->{s['new']}  {d}")
    print("FAILURES:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
