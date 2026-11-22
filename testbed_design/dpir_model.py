"""Capacity and routing model for a fixed point-to-point Terragraph / IP-50 bench (PoC 2).

A network is a set of sites; Terragraph (TG) links join sectors of N367 nodes (TDD, odd/even
polarity, one peer per sector); IP-50 links (MW) are FDD microwave hops with a fixed capacity per
direction and a continuous carrier, so they neither suffer nor cause load-dependent coupling.
Cabled couplings join opposite-polarity TG sectors of distinct links: the aggressor's data lands
in the victim's receive window only while the aggressor transmits.

Model (per link direction, 1 s steps):
- A TG direction runs at the rate of the highest MCS whose link-adaptation (LA) threshold its SINR
  meets.  Two Terragraph tables: LQM (the QTI firmware 10.11 configuration, primary) and the CTF
  ramp table (sensitivity).  Rates are the CTF per-direction effective TDD rates.
- An aggressor counts once the share of the victim's data airtime it hits, times LA's
  BLER-to-PER factor (about 30), exceeds LA's 0.5 % PER target: about 0.017 % of the airtime.
- An idle TG direction that is up emits `d0` of its data airtime (0: idle slots empty;
  about 4.35 %: a QoS-Null frame in every transmit slot, as the Terragraph MAC spec says).
- LA no-traffic mode: a direction idle for a step restarts capped at MCS9, then MCS10.
- A link brought back up ignites for `t_ign` steps (emits d0, carries nothing), then runs MTPO for
  `t_mtpo` steps capped at MCS9.
- Flows sharing a direction split it max-min fair; each sector is its own radio and each direction
  its own subframe, so relaying costs no airtime.
"""
import itertools
import math

import networkx as nx

from validate import bearing, sector

SEC = {0: "E", 1: "N", 2: "W", 3: "S"}

# (SNR threshold dB, MCS, Mb/s per direction).  laMinMcs 2.
MCS_LQM = [(4.5, 2, 323), (5.5, 3, 404), (6.5, 4, 485), (7.5, 6, 646), (8.5, 7, 808), (9.5, 8, 969),
           (10.5, 9, 1050), (13.5, 10, 1292), (14.5, 11, 1616), (16.5, 12, 1939)]
MCS_CTF = [(2.5, 2, 323), (3.0, 3, 404), (4.5, 4, 485), (5.0, 5, 525), (5.5, 6, 646), (7.5, 7, 808),
           (9.0, 8, 969), (11.5, 9, 1050), (13.0, 10, 1292), (15.0, 11, 1616), (17.0, 12, 1939)]
TABLES = {"LQM": MCS_LQM, "CTF": MCS_CTF}
RATE_MCS = {r: m for t in TABLES.values() for _, m, r in t}
MCS_RATE = {m: r for t in TABLES.values() for _, m, r in t}

PER_TARGET = 0.005
BLER2PER = 30.0
QOS_NULL = 3 * 2.9 / 200          # ~2.9 us QoS-Null at the start of each of 3 TX slots per subframe
OPENR_METRIC = [(9, 1), (4, 3), (2, 15)]   # Terragraph default MCS -> Open/R metric
TOKEN_RATE_TENTHS, TOKEN_BURST_TENTHS = 1, 30   # 0.1 token/s, burst 3 (integer tenths)


def mcs_of(snr, table="LQM"):
    best = (0, 0.0)
    for thr, m, r in TABLES[table]:
        if snr >= thr:
            best = (m, r)
    return best


def rate(snr, table="LQM"):
    return mcs_of(snr, table)[1]


def sinr(snr, inrs):
    return snr - 10 * math.log10(1 + sum(10 ** (i / 10) for i in inrs))


def openr_metric(mcs):
    for m, metric in OPENR_METRIC:
        if mcs >= m:
            return metric
    return OPENR_METRIC[-1][1]


def maxmin(sub, caps):
    """sub: list of (demand, [directions]); max-min fair water-filling on direction capacity."""
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


class Net:
    """spec: pos {site: (x, y)}; tg_nodes {site: (box, slot, n_sectors, polarity)};
    tg_links {name: (a, b, snr_dB, attenuator)}; mw_links {name: (a, b, capacity_Mbps)};
    couplings {name: ((site, sector), (site, sector), inr_dB, attenuator)}."""

    def __init__(self, spec):
        self.spec = spec
        self.pos = spec["pos"]
        self.G = nx.Graph()
        for k, (a, b, snr, att) in spec["tg_links"].items():
            self.G.add_edge(a, b, name=k, tech="TG")
        for k, (a, b, cap) in spec.get("mw_links", {}).items():
            assert not self.G.has_edge(a, b), (a, b)
            self.G.add_edge(a, b, name=k, tech="MW")
        self.tech = {d["name"]: d["tech"] for _, _, d in self.G.edges(data=True)}
        self.ends = {d["name"]: (a, b) for a, b, d in self.G.edges(data=True)}
        self.DIRS = [(a, b) for a, b in self.G.edges] + [(b, a) for a, b in self.G.edges]
        self.pol = {}
        for k, (a, b, snr, att) in spec["tg_links"].items():
            for x, y in ((a, b), (b, a)):
                self.pol[(x, self.end_sector(x, y))] = spec["tg_nodes"][x][3]
        self._agg = {}

    def link(self, a, b=None):
        if b is None:
            a, b = a
        return self.G.edges[a, b]["name"]

    def end_sector(self, x, y):
        return SEC[sector("DN", bearing(self.pos[x], self.pos[y]))]

    def tx_sec(self, d):
        return (d[0], self.end_sector(d[0], d[1]))

    def rx_sec(self, d):
        return (d[1], self.end_sector(d[1], d[0]))

    def is_tg(self, d):
        return self.tech[self.link(d)] == "TG"

    def subframe(self, d):
        return "A" if self.pol[self.tx_sec(d)] == "even" else "B"

    def aggressors(self, d):
        """[(aggressor direction, coupling name)] for TG direction d."""
        if d in self._agg:
            return self._agg[d]
        out = []
        if self.is_tg(d):
            y = self.rx_sec(d)
            for name, (x1, x2, inr, att) in self.spec["couplings"].items():
                for xa, yb in ((x1, x2), (x2, x1)):
                    if yb != y or self.pol[xa] == self.pol[yb]:
                        continue
                    out += [(a, name) for a in self.DIRS
                            if self.is_tg(a) and self.tx_sec(a) == xa and self.link(a) != self.link(d)]
        self._agg[d] = out
        return out

    def evaluate(self, sub, model="LA", inr=None, d0=0.0, down=(), nocarry=(), snr=None, cap=None,
                 table="LQM", bler2per=BLER2PER):
        """sub: [(demand, node path)].  Returns (throughput per sub-flow, capacity, duty).
        inr: None, a number (all couplings) or {coupling: INR}.  cap: {direction: max Mb/s}."""
        snr = {k: v[2] for k, v in self.spec["tg_links"].items()} | (snr or {})
        cap = cap or {}
        cpl_inr = {k: v[2] for k, v in self.spec["couplings"].items()}
        if isinstance(inr, (int, float)):
            cpl_inr = {k: float(inr) for k in cpl_inr}
        elif isinstance(inr, dict):
            cpl_inr |= inr
        mw_cap = {k: v[2] for k, v in self.spec.get("mw_links", {}).items()}
        down, nocarry = set(down), set(nocarry)
        usable = [d for d in self.DIRS if self.link(d) not in down and self.link(d) not in nocarry]
        emit = {d for d in self.DIRS if self.is_tg(d) and self.link(d) not in down}
        th = PER_TARGET / bler2per

        def base_rate(d, inrs):
            k = self.link(d)
            r = mw_cap[k] if k in mw_cap else rate(sinr(snr[k], inrs), table)
            return min(r, cap.get(d, r))

        hops = [list(zip(p, p[1:])) for _, p in sub]
        dem = [x if all(h in usable for h in hp) else 0.0 for (x, _), hp in zip(sub, hops)]
        caps = {d: base_rate(d, []) for d in usable}
        seen, duty = [], {}
        for _ in range(200):
            thr = maxmin(list(zip(dem, hops)), caps)
            load = {d: 0.0 for d in self.DIRS}
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
                ags = [(a, cpl_inr[c]) for a, c in self.aggressors(d) if a in emit]
                if model == "LA":
                    new[d] = base_rate(d, [i for a, i in ags if duty.get(a, 0) > th])
                else:
                    tot = 0.0
                    for on in itertools.product((0, 1), repeat=len(ags)):
                        p = 1.0
                        for (a, _), o in zip(ags, on):
                            p *= duty.get(a, 0) if o else 1 - duty.get(a, 0)
                        tot += p * base_rate(d, [i for (a, i), o in zip(ags, on) if o])
                    new[d] = tot
            key = tuple(round(new[d], 3) for d in usable)
            if key in seen:
                break
            seen.append(key)
            caps = new
        thr = maxmin(list(zip(dem, hops)), caps)
        return thr, caps, duty


# ------------------------------------------------------------------------------------------
# routing helpers
# ------------------------------------------------------------------------------------------
def joint_options(flows, cands):
    return list(itertools.product(*[range(len(cands[f[0]])) for f in flows]))


def sub_of(opt, flows, cands, dem):
    return [(x, cands[f[0]][i]) for f, i, x in zip(flows, opt, dem)]


def per_flow(fids, thr, flows):
    out = {f[0]: 0.0 for f in flows}
    for fid, t in zip(fids, thr):
        out[fid] += t
    return out


def ecmp_sub(net, flows, dem, weight=None):
    """Static shortest-path routing (hop count, or OSPF cost) with ECMP over equal-cost paths."""
    sub, fids = [], []
    for (fid, s, t, _), x in zip(flows, dem):
        ps = list(nx.all_shortest_paths(net.G, s, t, weight=weight))
        sub += [(x / len(ps), p) for p in ps]
        fids += [fid] * len(ps)
    return sub, fids


def ospf_weight(net, ref=10000.0):
    caps = {k: rate(v[2]) for k, v in net.spec["tg_links"].items()} | \
           {k: v[2] for k, v in net.spec.get("mw_links", {}).items()}
    return lambda a, b, d: max(1, round(ref / caps[d["name"]]))


def best_opt(net, flows, cands, dem, **kw):
    """Best joint routing over the candidates (ties: fewer hops, then lower option index)."""
    best = None
    for opt in joint_options(flows, cands):
        thr, _, _ = net.evaluate(sub_of(opt, flows, cands, dem), **kw)
        key = (-round(sum(thr), 3), sum(len(cands[f[0]][i]) for f, i in zip(flows, opt)), opt)
        if best is None or key < best[0]:
            best = (key, opt, thr)
    return best[1], best[2]


def rank(net, flows, cands, dem=None, **kw):
    dem = dem if dem is not None else [f[3] for f in flows]
    rows = []
    for opt in joint_options(flows, cands):
        thr, _, _ = net.evaluate(sub_of(opt, flows, cands, dem), **kw)
        rows.append(dict(opt=opt, paths=["-".join(cands[f[0]][i]) for f, i in zip(flows, opt)],
                         thr=[round(t) for t in thr], total=round(sum(thr), 1),
                         hops=sum(len(cands[f[0]][i]) - 1 for f, i in zip(flows, opt))))
    rows.sort(key=lambda r: (-r["total"], r["hops"], r["paths"]))
    return rows


# ------------------------------------------------------------------------------------------
# dynamic simulation
# ------------------------------------------------------------------------------------------
class Ramp:
    """LA no-traffic mode: a TG direction idle for >= 1 step restarts at MCS9, then MCS10."""

    def __init__(self, net, warm):
        self.net = net
        self.idle = {d: 0 if d in warm else 1 for d in net.DIRS}
        self.streak = {d: 1 if d in warm else 0 for d in net.DIRS}
        self.was_idle = {d: d not in warm for d in net.DIRS}

    def caps(self, active):
        out = {}
        for d in self.net.DIRS:
            if not self.net.is_tg(d) or d not in active:
                continue
            first = self.streak[d] == 0 and self.idle[d] >= 1
            second = self.streak[d] == 1 and self.was_idle[d]
            if first:
                out[d] = MCS_RATE[9]
            elif second:
                out[d] = MCS_RATE[10]
        return out

    def step(self, active):
        for d in self.net.DIRS:
            if d in active:
                if self.streak[d] == 0:
                    self.was_idle[d] = self.idle[d] >= 1
                self.streak[d] += 1
                self.idle[d] = 0
            else:
                self.idle[d] += 1
                self.streak[d] = 0


def openr_run(net, flows, sched, inr=None, d0=0.0, snr_sched=None, ramp=True, warm=None):
    """Terragraph Open/R with MCS link metrics (15/3/1), one step of damping, a per-node token
    bucket (0.1/s, burst 3) and ECMP.  An IP-50 hop has a fixed metric from its capacity."""
    mw = net.spec.get("mw_links", {})
    adv = {}
    for d in net.DIRS:
        k = net.link(d)
        adv[d] = (1 if mw[k][2] >= 1000 else 3 if mw[k][2] >= 450 else 15) if k in mw else 1
    tokens = {n: TOKEN_BURST_TENTHS for n in net.G}
    dg = nx.DiGraph()
    dg.add_edges_from(net.DIRS)
    out = dict(total=[], per=[], changes=0)
    prev = None
    rp = None
    for t, dem in enumerate(sched):
        for d in net.DIRS:
            dg.edges[d]["w"] = adv[d]
        sub, fids = [], []
        for (fid, s, tt, _), x in zip(flows, dem):
            ps = list(nx.all_shortest_paths(dg, s, tt, weight="w"))
            sub += [(x / len(ps), p) for p in ps]
            fids += [fid] * len(ps)
        sig = tuple(sorted((f, tuple(p)) for f, (_, p) in zip(fids, sub)))
        out["changes"] += prev is not None and sig != prev
        prev = sig
        active = {d for x, p in sub if x > 0 for d in zip(p, p[1:])}
        if rp is None:
            rp = Ramp(net, active if warm is None else warm)
        cap = rp.caps(active) if ramp else {}
        thr, caps, _ = net.evaluate(sub, inr=inr, d0=d0, cap=cap, snr=(snr_sched[t] if snr_sched else None))
        rp.step(active)
        out["total"].append(sum(thr))
        out["per"].append(per_flow(fids, thr, flows))
        for n in tokens:
            tokens[n] = min(TOKEN_BURST_TENTHS, tokens[n] + TOKEN_RATE_TENTHS)
        for d in net.DIRS:
            if not net.is_tg(d):
                continue
            m = RATE_MCS.get(round(caps.get(d, 0)), 0)
            if d not in active:
                m = min(m, 9)
            target = openr_metric(m) if m >= 2 else OPENR_METRIC[-1][1]
            if target != adv[d] and tokens[d[0]] >= 10:
                adv[d] = target
                tokens[d[0]] -= 10
    return out


def simulate(net, flows, cands, sched, policy, inr=None, d0=0.0, snr_sched=None, ramp=True,
             t_ign=6, t_mtpo=12, lead=0, start_opt=None):
    """Run a routing policy over a demand schedule (one demand vector per 1 s step).
    policy: 'hopcount' | 'dpir_lagged' | 'dpir_oracle' | 'reactive' (route only, all links up),
            'links_lagged' | 'links_oracle' (DPIR target plus taking unused coupled links down).
    lead (links_oracle): steps ahead whose plans keep a link up (pre-ignition)."""
    T = len(sched)
    cache = {}

    def ev(opt, dem, down=frozenset(), nocarry=frozenset(), cap=None, t=0):
        key = (opt, tuple(dem), down, nocarry, tuple(sorted((cap or {}).items())), t if snr_sched else 0)
        if key not in cache:
            cache[key] = net.evaluate(sub_of(opt, flows, cands, dem), inr=inr, d0=d0, down=down, nocarry=nocarry,
                                      cap=cap, snr=(snr_sched[t] if snr_sched else None))
        return cache[key]

    coupled = {net.link(a) for d in net.DIRS for a, _ in net.aggressors(d)} | \
              {net.link(d) for d in net.DIRS if net.aggressors(d)}
    plans = {}

    def plan(dem, t):
        k = (tuple(dem), t if snr_sched else 0)
        if k not in plans:
            best = None
            for opt in joint_options(flows, cands):
                used = {net.link(d) for f, i in zip(flows, opt) for d in zip(cands[f[0]][i], cands[f[0]][i][1:])}
                spare = sorted(coupled - used)
                for n in range(len(spare) + 1):
                    for dn in itertools.combinations(spare, n):
                        thr, _, _ = ev(opt, dem, frozenset(dn), t=t)
                        key = (round(sum(thr), 1), -len(dn), -sum(len(cands[f[0]][i]) for f, i in zip(flows, opt)))
                        if best is None or key > best[0]:
                            best = (key, opt, frozenset(dn))
            plans[k] = (best[1], best[2])
        return plans[k]

    hop_opt = start_opt
    if hop_opt is None:
        hop_opt = tuple(min(range(len(cands[f[0]])), key=lambda j: (len(cands[f[0]][j]), j)) for f in flows)
    down, ign, mtpo = set(), {}, {}
    prev_opt, meas = hop_opt, None
    delivered, changes, ignitions, trace = 0.0, 0, 0, []
    rp = None
    for t in range(T):
        dem = sched[t]
        want_down = set()
        if policy == "hopcount":
            opt = hop_opt
        elif policy == "dpir_oracle":
            opt, _ = best_opt(net, flows, cands, dem, inr=inr, d0=d0, snr=(snr_sched[t] if snr_sched else None))
        elif policy == "dpir_lagged":
            pt = max(t - 1, 0)
            opt, _ = best_opt(net, flows, cands, sched[pt], inr=inr, d0=d0, snr=(snr_sched[pt] if snr_sched else None))
        elif policy == "reactive":
            if meas is None:
                opt = hop_opt
            else:
                est = []
                for o in joint_options(flows, cands):
                    hp = [list(zip(p, p[1:])) for _, p in sub_of(o, flows, cands, dem)]
                    thr = maxmin(list(zip(dem, hp)), meas)
                    est.append((-round(sum(thr), 3), sum(len(h) for h in hp), o != prev_opt, o))
                opt = min(est)[3]
        elif policy == "links_lagged":
            opt, want_down = plan(sched[max(t - 1, 0)], max(t - 1, 0))
            want_down = set(want_down)
        elif policy == "links_oracle":
            opt, want_down = plan(dem, t)
            want_down = set(want_down)
            for k in range(1, lead + 1):
                if t + k < T:
                    want_down &= plan(sched[t + k], t + k)[1]
        else:
            raise ValueError(policy)
        # link state: bring up (ignite), then take down only when nothing is still igniting
        for k in list(down):
            if k not in want_down:
                down.discard(k)
                ign[k] = t_ign
                ignitions += 1
        if not ign:
            for k in want_down:
                if k not in down:
                    down.add(k)
                    mtpo.pop(k, None)
        unavailable = down | set(ign)
        opts_ok = [o for o in joint_options(flows, cands)
                   if not ({net.link(d) for f, i in zip(flows, o) for d in zip(cands[f[0]][i], cands[f[0]][i][1:])}
                           & unavailable)]
        cap_mtpo = {d: MCS_RATE[9] for d in net.DIRS if net.link(d) in mtpo}
        if opt not in opts_ok:
            best = None
            for o in opts_ok:
                thr, _, _ = ev(o, dem, frozenset(down), frozenset(ign), cap_mtpo or None, t)
                key = (-round(sum(thr), 3), o)
                best = key if best is None or key < best else best
            opt = best[1] if best else None
        thr = [0.0]
        if opt is not None:
            active = {d for (x, p) in sub_of(opt, flows, cands, dem) if x > 0 for d in zip(p, p[1:])}
            if rp is None:
                rp = Ramp(net, active)
            cap = dict(cap_mtpo)
            if ramp:
                for d, c in rp.caps(active).items():
                    cap[d] = min(c, cap.get(d, c))
            thr, caps, _ = ev(opt, dem, frozenset(down), frozenset(ign), cap or None, t)
            rp.step(active)
            if policy == "reactive":
                meas = {d: (caps.get(d, 0) if d in active else min(caps.get(d, 0), MCS_RATE[9])) for d in net.DIRS}
            changes += opt != prev_opt
            prev_opt = opt
        delivered += sum(thr)
        trace.append(round(sum(thr)))
        for k in list(mtpo):
            mtpo[k] -= 1
            if mtpo[k] <= 0:
                del mtpo[k]
        for k in list(ign):
            ign[k] -= 1
            if ign[k] <= 0:
                del ign[k]
                if t_mtpo:
                    mtpo[k] = t_mtpo
    offered = sum(sum(x) for x in sched)
    return dict(delivered_pct=round(100 * delivered / offered, 1), route_changes=changes, ignitions=ignitions,
                trace=trace)
