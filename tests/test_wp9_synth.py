"""WP9 test support: tiny synthetic harness pickles and manifests.

Not a test module (no test_* functions).  Every file goes to
output/tmp_tests inside the tree (the server rules forbid /tmp).
"""
import json
import os
import pickle
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

MARL_PATH = ('marl_static', 'marl_freeze', 'xldet', 'random')
RANK = {'marl_static': 6.0, 'marl_freeze': 5.5, 'xldet': 5.0, 'sdn': 4.0, 'ospf': 3.0,
        'olsr': 2.9, 'batman': 2.8, 'aodv': 2.7, 'random': 0.0}
T_SEV_TEST = {'A': 5, 'B': 0}
T_TEST = 40
N_SITES = 3


try:
    import numpy as np          # the server has it; the numpy cases are skipped without it
except Exception:               # pragma: no cover
    np = None


def mkdtemp(prefix):
    base = os.path.join(ROOT, 'output', 'tmp_tests')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def provenance(T=T_TEST, **over):
    p = {'tree_head': 'abc1234', 'tree_sha256': 'f' * 64, 'schema_id': 'schema-test',
         'pythonhashseed': '0', 'provenance_complete': True, 'total_ticks': T,
         'severance_tick': T_SEV_TEST['A'], 'driver': 'synthetic',
         'checkpoint_sha256': ['c' * 64]}          # the harness: one sha256 per --checkpoint path
    p.update(over)
    return p


def make_run(T, seed, arm, k, n_sites=N_SITES, events=True):
    """One synthetic run.  Fragments: 1 before the event, 3 after it; the MARL
    arm then flaps 3/2 (policy splits), XL-DET settles to 2 once, routing
    arms stay at 3, random flaps 3/4/3."""
    ts = T_SEV_TEST[k]
    rank = RANK.get(arm, 1.0)
    conn = [40.0 + rank + 0.01 * seed + 0.001 * i for i in range(T)]
    offered = [1700.0 + i for i in range(T)]
    delivered = [900.0 + 10 * rank + seed + 0.5 * i for i in range(T)]
    energy = [200.0 - rank + 0.1 * seed + 0.01 * i for i in range(T)]
    frag, raw = [], []
    for t in range(T):
        if t < ts:
            f = r = 1
        else:
            r = 3
            if arm in ('marl_static', 'marl_freeze'):
                f = 3 if (t - ts) % 4 < 2 else 2
            elif arm == 'xldet':
                f = 3 if t < ts + 3 else 2
            elif arm == 'random':
                f = 4 if (t - ts) % 3 == 1 else 3
            elif arm == 'sdn':
                f = 3 if t < ts + 6 else 2
            else:
                f = 3
        frag.append(f)
        raw.append(r)
    relay_count = [0 if t < ts else (1 if frag[t] == 2 else 0) for t in range(T)]
    relay_bridges = [max(0, raw[t] - frag[t]) for t in range(T)]
    postcards = ([n_sites if (t >= ts and t % 10 == 0) else 0 for t in range(T)]
                 if arm in MARL_PATH else [0] * T)
    latency = [4.0 + 0.1 * (i % 5) for i in range(T)]
    if arm in ('ospf', 'olsr', 'batman', 'aodv'):
        latency[ts] = float('nan')          # nothing delivered on the severance tick
    run = {'conn': conn, 'offered': offered, 'delivered': delivered, 'energy': energy,
           'fragments': frag, 'fragments_raw': raw, 'relay_count': relay_count,
           'relay_bridges': relay_bridges, 'postcards': postcards, 'latency': latency,
           'overhead': [0.0 if t < ts else 100.0 + rank for t in range(T)],
           'n_radio_sites': n_sites, 'repoints': 0,
           'static_fp_start': 'fp-%s' % arm, 'static_fp_end': 'fp-%s' % arm}
    if events:
        flap = arm in ('marl_static', 'marl_freeze', 'random')
        run['relay_created'] = [1 if (t >= ts and (t - ts) % 4 == 0 and flap) else 0 for t in range(T)]
        run['relay_removed'] = [1 if (t > ts and (t - ts) % 4 == 2 and flap) else 0 for t in range(T)]
        run['bridge_teardowns'] = list(run['relay_removed'])
        run['idle_bridge_teardowns'] = [x if t % 8 == 2 else 0 for t, x in enumerate(run['relay_removed'])]
        run['relay_mode_flips'] = [2 * x for x in run['relay_removed']]
        run['relay_removed_by_reason'] = {'mode_change': list(run['relay_removed']),
                                          'repoint': [0] * T}
        run['relay_teardown_holds'] = {'grace': [1 if t > ts and flap else 0 for t in range(T)],
                                       'busy': [0] * T}
        run['stale_edges'] = [0] * T
        run['stale_u2u_volume'] = [0.0] * T
        run['energy_ctrl'] = [0.0 if t < ts else 0.01 for t in range(T)]
        run['overhead_by_type'] = {'bfd': [0.0 if t < ts else 60.0 for t in range(T)],
                                   'hello': [0.0 if t < ts else 20.0 for t in range(T)],
                                   'lsa': [0.0 if t < ts else 15.0 for t in range(T)],
                                   'postcard': [60.0 * x for x in postcards]}  # wp7-scan-ok: WP5 DCC message type, not an action head
        run['relay_lifetimes'] = ([{'link_id': 'TR_1_2_1', 'created': ts, 'up': ts + 3, 'removed': ts + 6,
                                    'cause': 'mode_change', 'was_bridge': True, 'load': 0.0},
                                   ('TR_2_3_2', ts + 8, ts + 11, None, None, True, 0.1),
                                   {'link_id': 'TR_3_4_3', 'created': ts + 1, 'up': ts + 4, 'removed': ts + 5,
                                    'cause': 'repoint', 'was_bridge': False, 'load': 0.0}]
                                  if flap else [])
    return run


def make_pickle(path, arms, seeds, T=T_TEST, prov='default', events=True, mutate=None):
    """Write a harness-style pickle; mutate(sk, arm, seed, run) may edit or
    return None to drop a run."""
    P = {'recovery_multi': {}, 'rescue_multi': {}, 'seeds': list(seeds)}
    for k, sk in (('A', 'recovery_multi'), ('B', 'rescue_multi')):
        for arm in arms:
            P[sk][arm] = {}
            for s in seeds:
                run = make_run(T, s, arm, k, events=events)
                if mutate is not None:
                    run = mutate(sk, arm, s, run)
                    if run is None:
                        continue
                P[sk][arm][s] = run
    if prov == 'default':
        prov = provenance(T)
    if prov is not None:
        P['provenance'] = prov
    with open(path, 'wb') as f:
        pickle.dump(P, f)
    return path


def make_manifest(path, seeds, ts=None):
    ts = ts or T_SEV_TEST
    inst = []
    for kind, k in (('recovery', 'A'), ('rescue_ops', 'B')):
        for s in seeds:
            inst.append({'scenario_kind': kind, 'topology_seed': s, 'severance_tick': ts[k],
                         'attainable_optimum': 2 if s % 2 else 1, 'raw_components': 3,
                         'fully_reunifiable': s % 2 == 0})
    json.dump({'instances': inst, 'note': 'synthetic'}, open(path, 'w'))
    return path


class patched:
    """Temporarily set module attributes of confirmatory_analysis for short
    synthetic runs (event ticks, surge window)."""

    def __init__(self, **kw):
        import confirmatory_analysis as CA
        self.CA, self.kw, self.saved = CA, kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = getattr(self.CA, k)
            setattr(self.CA, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.CA, k, v)
        return False


def short_run_patch():
    return patched(T_EVENT={'A': T_SEV_TEST['A'], 'B': 20}, PRE_SURGE=(10, 20), SURGE_B=20)
