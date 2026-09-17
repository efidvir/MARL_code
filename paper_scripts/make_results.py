# -*- coding: utf-8 -*-
"""Every number in the Results section of marl_6g_recovery.tex, from one file.

Reads auth_results/ce_results.json (per-run KPI dump of the corrected-engine
sweep on the two frozen test sets, the stress sweep, and the legacy-engine
sweeps kept for the sensitivity paragraph) plus the two benchmark manifests,
and prints a self-contained report: table rows, pgfplots coordinate lists,
and the scalars quoted in prose.  Nothing here is tuned; steady state is the
mean over the final 500 ticks, energy per delivered unit is whole-run energy
over whole-run delivered volume, and reunification is scored against the
per-instance attainable optimum recorded in the manifests.

Usage:  python paper/make_results.py > paper/results_numbers.txt
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Configuration (environment; the defaults reproduce the pre-Amendment-1 report):
#   MR_JSON        per-run KPI dump to read       (default auth_results/ce_results.json)
#   MR_TAGS        tags merged into the instance set (default ce_50_59,ce_70_79)
#   MR_MARL        arm reported as MARL-RIC        (default marl_freeze)
#   MR_MANIFESTS   benchmark manifests (repo-relative)
#   MR_STRESS_TAG  tag holding the interferer stress runs (default ce_stress)
# The legacy-engine sensitivity always reads auth_results/ce_results.json.
J = json.load(open(os.environ.get('MR_JSON', os.path.join(ROOT, 'auth_results', 'ce_results.json'))))
JL = json.load(open(os.path.join(ROOT, 'auth_results', 'ce_results.json')))
TAGS = tuple(t for t in os.environ.get('MR_TAGS', 'ce_50_59,ce_70_79').split(',') if t)
STRESS_TAG = os.environ.get('MR_STRESS_TAG', 'ce_stress')
MARL = os.environ.get('MR_MARL', 'marl_freeze')
MAN = {}
for mf in os.environ.get('MR_MANIFESTS', 'benchmark_manifest.json,benchmark_manifest_70_79.json').split(','):
    for inst in json.load(open(os.path.join(ROOT, mf)))['instances']:
        MAN[(inst['scenario_kind'], inst['topology_seed'])] = inst

ARMS = [MARL, 'xldet', 'sdn', 'ospf', 'olsr', 'batman', 'aodv', 'random']
LABEL = {MARL: r'\marl (proposed)', 'xldet': 'XL-DET', 'sdn': 'SDN/OpenFlow',
         'ospf': 'OSPF', 'olsr': 'OLSR', 'batman': 'B.A.T.M.A.N.', 'aodv': 'AODV',
         'random': 'Uniform-random control'}
SHORT = {MARL: 'MARL-RIC', 'xldet': 'XL-DET', 'sdn': 'SDN', 'ospf': 'OSPF',
         'olsr': 'OLSR', 'batman': 'BATMAN', 'aodv': 'AODV', 'random': 'Random'}
VARIANT = 'marl_freeze' if MARL != 'marl_freeze' else None
if VARIANT:
    LABEL[VARIANT] = 'MARL-RIC adapt-then-freeze (variant)'
    SHORT[VARIANT] = 'MARL-AF'
SCEN = {'recovery_multi': 'recovery', 'rescue_multi': 'rescue_ops'}
SEV_TICK = {'recovery_multi': 200, 'rescue_multi': 1400}   # event tick per scenario
FLOOR = 10.0


def runs(tag, scen, arm):
    return J[tag]['scenarios'][scen][arm]


def runs_l(tag, scen, arm):
    return JL[tag]['scenarios'][scen][arm]


def has(scen, arm):
    return all(arm in J[t]['scenarios'][scen] for t in TAGS)


def merged(scen, arm, tags=TAGS):
    out = {}
    for t in tags:
        for s, r in runs(t, scen, arm).items():
            out[int(s)] = r
    return out


def msd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float('nan'), float('nan'), 0
    return (st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v))


def sli(r, scen):
    """Service-loss integral, eq. (sli): shortfall vs. the demand-corrected
    pre-event level, summed over post-event ticks (1 s each)."""
    conn = r['conn']['series']; off = r['offered']['series']
    t0 = SEV_TICK[scen]
    pre = [c for c in conn[:t0] if c is not None]
    pre_off = [o for o in off[:t0] if o is not None]
    post_off = [o for o in off[t0:] if o is not None]
    if not pre or not pre_off or not post_off:
        return float('nan'), float('nan')
    a_ref = st.mean(pre) * st.mean(pre_off) / st.mean(post_off)
    loss = sum(max(0.0, a_ref - (c if c is not None else 0.0)) for c in conn[t0:])
    return loss, a_ref


def fmt(m, sd=None, p=1):
    if sd is None:
        return '%.*f' % (p, m)
    return '%.*f $\\pm$ %.*f' % (p, m, p, sd)


def section(title):
    print('\n' + '=' * 78 + '\n' + title + '\n' + '=' * 78)


# ---------------------------------------------------------------- headline
for scen in ('recovery_multi', 'rescue_multi'):
    kind = SCEN[scen]
    seeds_all = sorted(merged(scen, 'ospf').keys())
    reun = [s for s in seeds_all if MAN[(kind, s)]['fully_reunifiable']]
    section('%s  --  %d test instances, %d with attainable optimum = 1 component'
            % (kind.upper(), len(seeds_all), len(reun)))
    print('reunifiable seeds:', reun)
    print('optima:', {s: MAN[(kind, s)]['attainable_optimum'] for s in seeds_all})
    print('raw components after cut (manifest):', {s: MAN[(kind, s)]['raw_components'] for s in seeds_all})
    raw = [MAN[(kind, s)]['raw_components'] for s in seeds_all]
    print('raw components mean %.2f range %d-%d ; mean optimum %.2f' % (
        st.mean(raw), min(raw), max(raw), st.mean([MAN[(kind, s)]['attainable_optimum'] for s in seeds_all])))

    # ---- main table: achievability, throughput, energy, J/unit, SLI, outage
    print('\n-- MAIN TABLE (all %d instances; mean +- sd across instances) --' % len(seeds_all))
    print('%-26s | %-14s | %-12s | %-14s | %-8s | %-16s | %-6s' % (
        'arm', 'achiev %', 'thr units', 'energy J/tick', 'J/unit', 'SLI pp.s', 'out %'))
    per_arm = {}
    for arm in ARMS:
        R = merged(scen, arm)
        ach = {s: R[s]['conn']['steady'] for s in seeds_all}
        thr = {s: R[s]['delivered']['steady'] for s in seeds_all}
        en = {s: R[s]['energy']['steady'] for s in seeds_all}
        jpu = sum(R[s]['energy']['total'] for s in seeds_all) / sum(R[s]['delivered']['total'] for s in seeds_all)
        jpu_seed = {s: R[s]['energy']['total'] / R[s]['delivered']['total'] for s in seeds_all}
        slis = {s: sli(R[s], scen)[0] for s in seeds_all}
        a_ref = st.mean([sli(R[s], scen)[1] for s in seeds_all])
        out = {s: R[s]['outage']['steady'] for s in seeds_all}
        comps = {s: R[s]['fragments']['steady'] for s in seeds_all}
        rawc = {s: R[s]['fragments_raw']['steady'] for s in seeds_all}
        br = {s: R[s]['relay_bridges']['steady'] for s in seeds_all}
        lat = {s: R[s]['latency']['steady'] for s in seeds_all}
        oh = {s: R[s]['overhead']['steady'] for s in seeds_all}
        rr = {s: R[s]['reach_restored']['steady'] for s in seeds_all}
        strd = {s: R[s]['stranded_ues']['steady'] for s in seeds_all}
        pr = {s: R[s]['peer_reach']['steady'] for s in seeds_all}
        per_arm[arm] = dict(ach=ach, thr=thr, en=en, jpu=jpu, jpu_seed=jpu_seed, sli=slis, a_ref=a_ref,
                            out=out, comps=comps, rawc=rawc, br=br, lat=lat, oh=oh, rr=rr, strd=strd, pr=pr)
        print('%-26s | %-14s | %-12s | %-14s | %-8s | %-16s | %-6s' % (
            LABEL[arm], fmt(*msd(ach.values())[:2]), fmt(*msd(thr.values())[:2], p=0),
            fmt(*msd(en.values())[:2]), '%.4f' % jpu,
            fmt(*msd(slis.values())[:2], p=0), '%.1f' % msd(out.values())[0]))
    print('A_ref (demand-corrected pre-event level, mean over arms/instances): %.2f %%' % st.mean(
        [per_arm[a]['a_ref'] for a in ARMS]))
    pre_off = st.mean([st.mean([o for o in merged(scen, 'ospf')[s]['offered']['series'][:SEV_TICK[scen]] if o is not None]) for s in seeds_all])
    post_off = st.mean([st.mean([o for o in merged(scen, 'ospf')[s]['offered']['series'][SEV_TICK[scen]:] if o is not None]) for s in seeds_all])
    print('offered volume per tick: pre-event %.1f  post-event %.1f' % (pre_off, post_off))
    pre_ach = {a: st.mean([st.mean([c for c in merged(scen, a)[s]['conn']['series'][:SEV_TICK[scen]] if c is not None]) for s in seeds_all]) for a in ARMS}
    print('pre-event achievability per arm (shared-world check):', {SHORT[a]: round(pre_ach[a], 2) for a in ARMS})

    # ---- margins
    m = per_arm[MARL]; x = per_arm['xldet']; d = per_arm['sdn']; o = per_arm['ospf']
    print('\n-- MARGINS (all instances) --')
    for nm, b in (('XL-DET', x), ('SDN', d), ('OSPF', o)):
        print('  achiev MARL - %s = %+.1f pp ; relative delivered volume %+.1f %% ; J/unit %.4f vs %.4f (%+.1f %%) ; energy/tick %.1f vs %.1f (%+.1f %%)' % (
            nm, msd(m['ach'].values())[0] - msd(b['ach'].values())[0],
            100 * (msd(m['thr'].values())[0] / msd(b['thr'].values())[0] - 1),
            m['jpu'], b['jpu'], 100 * (m['jpu'] / b['jpu'] - 1),
            msd(m['en'].values())[0], msd(b['en'].values())[0], 100 * (msd(m['en'].values())[0] / msd(b['en'].values())[0] - 1)))
    wins_x = sum(1 for s in seeds_all if m['ach'][s] > x['ach'][s])
    wins_d = sum(1 for s in seeds_all if m['ach'][s] > d['ach'][s])
    wins_o = sum(1 for s in seeds_all if m['ach'][s] > o['ach'][s])
    print('  paired wins on achievability: vs XL-DET %d/%d, vs SDN %d/%d, vs OSPF %d/%d' % (
        wins_x, len(seeds_all), wins_d, len(seeds_all), wins_o, len(seeds_all)))
    print('  paired J/unit wins vs XL-DET: %d/%d ; vs SDN: %d/%d' % (
        sum(1 for s in seeds_all if m['jpu_seed'][s] < x['jpu_seed'][s]), len(seeds_all),
        sum(1 for s in seeds_all if m['jpu_seed'][s] < d['jpu_seed'][s]), len(seeds_all)))
    print('  losses vs XL-DET on:', [(s, round(m['ach'][s], 1), round(x['ach'][s], 1)) for s in seeds_all if m['ach'][s] <= x['ach'][s]])

    # ---- reliability
    print('\n-- RELIABILITY (all instances): median / min / max / IQR / below %.0f%% floor --' % FLOOR)
    for arm in ARMS:
        v = sorted(per_arm[arm]['ach'].values())
        q = st.quantiles(v, n=4)
        print('  %-24s %5.1f  %5.1f  %5.1f  %4.1f  %d' % (LABEL[arm], st.median(v), min(v), max(v), q[2] - q[0],
                                                          sum(1 for a in v if a < FLOOR)))
    print('  MARL worst %.1f vs XL-DET median %.1f, SDN median %.1f, OSPF median %.1f' % (
        min(m['ach'].values()), st.median(x['ach'].values()), st.median(d['ach'].values()), st.median(o['ach'].values())))

    # ---- strip-plot coordinates (per instance, seed order)
    print('\n-- STRIP COORDINATES (seed order %s) --' % seeds_all)
    for arm in (MARL, 'xldet', 'sdn', 'ospf'):
        print('  %-12s %s' % (SHORT[arm], ' '.join('(%.1f,%s)' % (per_arm[arm]['ach'][s], '%d') for s in seeds_all)))

    # ---- reunification
    print('\n-- REUNIFICATION --')
    print('  %-24s %-22s %-22s %-10s %-8s %-8s %-8s' % ('arm', 'comps(reunif.)', 'comps(all)', 'dist2opt', 'at-opt', 'raw', 'bridges'))
    for arm in ARMS:
        c = per_arm[arm]['comps']
        cr = [c[s] for s in reun]
        dist = [c[s] - MAN[(kind, s)]['attainable_optimum'] for s in seeds_all]
        at_opt = sum(1 for s in seeds_all if c[s] <= MAN[(kind, s)]['attainable_optimum'] + 1e-9)
        print('  %-24s %-22s %-22s %-10s %-8s %-8s %-8s' % (
            LABEL[arm], '%.2f (%.2f-%.2f)' % (st.mean(cr), min(cr), max(cr)),
            '%.2f (%.2f-%.2f)' % (st.mean(c.values()), min(c.values()), max(c.values())),
            '%.2f' % st.mean(dist), '%d/%d' % (at_opt, len(seeds_all)),
            '%.2f' % st.mean(per_arm[arm]['rawc'].values()), '%.2f' % st.mean(per_arm[arm]['br'].values())))
    print('  MARL per-instance components (reunifiable):', {s: round(m['comps'][s], 2) for s in reun})
    print('  XL-DET per-instance components (reunifiable):', {s: round(x['comps'][s], 2) for s in reun})
    print('  MARL per-instance components (non-reunifiable, optimum):', {s: (round(m['comps'][s], 2), MAN[(kind, s)]['attainable_optimum']) for s in seeds_all if s not in reun})
    print('  XL-DET per-instance components (non-reunifiable, optimum):', {s: (round(x['comps'][s], 2), MAN[(kind, s)]['attainable_optimum']) for s in seeds_all if s not in reun})
    print('  achievability on reunifiable only: MARL %.1f XL-DET %.1f SDN %.1f OSPF %.1f' % tuple(
        st.mean([per_arm[a]['ach'][s] for s in reun]) for a in (MARL, 'xldet', 'sdn', 'ospf')))
    print('  J/unit on reunifiable only: MARL %.4f XL-DET %.4f SDN %.4f OSPF %.4f' % tuple(
        sum(merged(scen, a)[s]['energy']['total'] for s in reun) / sum(merged(scen, a)[s]['delivered']['total'] for s in reun)
        for a in (MARL, 'xldet', 'sdn', 'ospf')))
    print('  stranded flows after cut (100 - OSPF peer_reach steady): mean %.1f %% range %.1f-%.1f' % (
        100 - st.mean(o['pr'].values()), 100 - max(o['pr'].values()), 100 - min(o['pr'].values())))

    # ---- attribution / restoration
    print('\n-- ATTRIBUTION --')
    for arm in (MARL, 'xldet', 'random'):
        print('  %-24s bridges %.2f comps %.2f achiev %.1f UEs restored %.1f stranded %.1f' % (
            LABEL[arm], st.mean(per_arm[arm]['br'].values()), st.mean(per_arm[arm]['comps'].values()),
            st.mean(per_arm[arm]['ach'].values()), st.mean(per_arm[arm]['rr'].values()), st.mean(per_arm[arm]['strd'].values())))
    print('  stranded UEs under OSPF (no bridging): %.1f' % st.mean(o['strd'].values()))

    # ---- secondary
    print('\n-- SECONDARY: latency ms (mean+-sd), control overhead Mbps --')
    for arm in ARMS:
        print('  %-24s %s  %.3f' % (LABEL[arm], fmt(*msd(per_arm[arm]['lat'].values())[:2]), st.mean(per_arm[arm]['oh'].values())))

    if VARIANT and has(scen, VARIANT):
        V = merged(scen, VARIANT)
        va = [V[s]['conn']['steady'] for s in seeds_all]
        vc = [V[s]['fragments']['steady'] for s in seeds_all]
        vj = sum(V[s]['energy']['total'] for s in seeds_all) / sum(V[s]['delivered']['total'] for s in seeds_all)
        vat = sum(1 for s in seeds_all if V[s]['fragments']['steady'] <= MAN[(kind, s)]['attainable_optimum'] + 1e-9)
        dd = [m['ach'][s] - V[s]['conn']['steady'] for s in seeds_all]
        print('\n-- VARIANT (%s) --' % LABEL[VARIANT])
        print('  achiev %s | comps %.2f | at optimum %d/%d | J/unit pooled %.4f | energy/tick %.1f' % (
            fmt(*msd(va)[:2]), st.mean(vc), vat, len(seeds_all), vj,
            st.mean(V[s]['energy']['steady'] for s in seeds_all)))
        print('  MARL-RIC minus variant: achiev %+.2f pp (ahead on %d/%d)' % (
            st.mean(dd), sum(1 for x_ in dd if x_ > 0), len(dd)))

# ---------------------------------------------------------------- stress
if os.environ.get('MR_SKIP_STRESS'):
    print('\n[stress section skipped: MR_SKIP_STRESS set]')
    raise SystemExit(0)
section('STRESS: external -75 dBm interferer at t=2000 (Scenario A, seeds 50-59)')
kind = 'recovery'
seeds = sorted(int(s) for s in runs(STRESS_TAG, 'recovery_multi', 'xldet'))
print('%-12s %-22s %-22s %-10s %-14s' % ('arm', 'pre-stress [1500,2000)', 'post-stress [2000,end]', 'drop pp', 'comps pre/post'))
for arm in [MARL, 'xldet', 'sdn', 'ospf'] + ([VARIANT] if VARIANT and VARIANT in J[STRESS_TAG]['scenarios']['recovery_multi'] else []):
    pre = []; post = []; cpre = []; cpost = []
    for s in seeds:
        r = runs(STRESS_TAG, 'recovery_multi', arm)[str(s)]
        c = r['conn']['series']; f = r['fragments']['series']
        pre.append(st.mean([v for v in c[1500:2000] if v is not None]))
        post.append(st.mean([v for v in c[2000:] if v is not None]))
        cpre.append(st.mean([v for v in f[1500:2000] if v is not None]))
        cpost.append(st.mean([v for v in f[2000:] if v is not None]))
    print('%-12s %-22s %-22s %-10s %-14s' % (SHORT[arm], fmt(*msd(pre)[:2]), fmt(*msd(post)[:2]),
                                             '%+.1f' % (st.mean(post) - st.mean(pre)), '%.2f / %.2f' % (st.mean(cpre), st.mean(cpost))))

# ---------------------------------------------------------------- legacy sensitivity
section('ENGINE SENSITIVITY: legacy engine, same recipe retrained + validation-selected (mission s1234)')
for scen in ('recovery_multi',):
    kind = SCEN[scen]
    seeds_all = sorted(int(s) for s in runs_l('legacy_50_59', scen, 'xldet')) + sorted(int(s) for s in runs_l('legacy_70_79', scen, 'xldet'))
    reun = [s for s in seeds_all if MAN[(kind, s)]['fully_reunifiable']]
    def leg(arm):
        R = {}
        src50 = 'legacy_mission_50_59' if arm == 'marl_freeze' else 'legacy_50_59'
        for s, r in runs_l(src50, scen, arm).items(): R[int(s)] = r
        for s, r in runs_l('legacy_70_79', scen, arm).items(): R[int(s)] = r
        return R
    for arm in ('marl_freeze', 'xldet', 'sdn', 'ospf', 'random'):
        R = leg(arm)
        ach = [R[s]['conn']['steady'] for s in seeds_all]
        cr = [R[s]['fragments']['steady'] for s in reun]
        call = [R[s]['fragments']['steady'] for s in seeds_all]
        jpu = sum(R[s]['energy']['total'] for s in seeds_all) / sum(R[s]['delivered']['total'] for s in seeds_all)
        print('  %-12s achiev %s  comps(reunif) %.2f  comps(all) %.2f  J/unit %.4f' % (SHORT[arm], fmt(*msd(ach)[:2]), st.mean(cr), st.mean(call), jpu))
    # online vs frozen under the legacy engine (same mission policy, 50-59)
    on = runs_l('legacy_mission_50_59', scen, 'marl'); fr = runs_l('legacy_mission_50_59', scen, 'marl_freeze')
    s10 = sorted(int(s) for s in fr)
    print('  legacy 50-59 online-adaptation arm: achiev %.1f comps %.2f ; adapt-then-freeze: achiev %.1f comps %.2f' % (
        st.mean([on[str(s)]['conn']['steady'] for s in s10]), st.mean([on[str(s)]['fragments']['steady'] for s in s10]),
        st.mean([fr[str(s)]['conn']['steady'] for s in s10]), st.mean([fr[str(s)]['fragments']['steady'] for s in s10])))
    ls = runs_l('legacy_stress', scen, 'marl_freeze')
    print('  legacy stress arms:', sorted(JL['legacy_stress']['scenarios'][scen].keys()))
