# -*- coding: utf-8 -*-
"""Post-hoc diagnostic patch for the stale relay-edge defect (MARL_run_8).

Applied to a byte-identical copy of the frozen MARL_run_6 tree.  Every change
is gated by an environment flag, so with both flags unset the patched code
executes exactly the frozen code path:

  MARL_DIAG_PHANTOM=1  count, per tick, the routing-graph edges whose link
                       object no longer exists ("stale edges"), and the
                       delivered volume that crossed at least one of them
                       (instrumentation only: no RNG draw, no state change);
  MARL_FIX_PHANTOM=1   when a relay link is torn down, remove its graph edge
                       using the link's own endpoints (the frozen code splits
                       the link id on '_', which fails for node ids such as
                       'O-DU_18', so the edge survives with no link behind it).

Usage (in ~/MARL_run_8):  python3 patch_phantom.py
"""
import re

SIM = 'sixg_sim/simulation.py'
HAR = 'run_timeline_comparison.py'


def sub_once(text, old, new, label):
    n = text.count(old)
    assert n == 1, '%s: expected 1 match, found %d' % (label, n)
    return text.replace(old, new)


def sub_count(text, old, new, count, label):
    n = text.count(old)
    assert n == count, '%s: expected %d matches, found %d' % (label, count, n)
    return text.replace(old, new)


s = open(SIM, encoding='utf-8').read()

# 1. flags (module level, right after the imports block: anchor on the class)
s = sub_once(
    s, '\nclass Simulator',
    '\n_DIAG_PHANTOM = os.environ.get("MARL_DIAG_PHANTOM", "") == "1"\n'
    '_FIX_PHANTOM = os.environ.get("MARL_FIX_PHANTOM", "") == "1"\n'
    '\n\nclass Simulator', 'flags')

# 2. teardown: the frozen code pops the link, then tries a string split.
#    With the fix flag, drop the edge by the popped link's endpoints unless
#    another up link still joins the pair.
old_td_1 = (
    '                for rid in removed_ids:\n'
    '                    # Remove corresponding topology link\n'
    '                    self.topology.links.pop(rid, None)\n')
new_td_1 = (
    '                for rid in removed_ids:\n'
    '                    # Remove corresponding topology link\n'
    '                    _lk = self.topology.links.pop(rid, None)\n'
    '                    if _FIX_PHANTOM:\n'
    '                        self._diag_drop_relay_edge(rid, _lk)\n')
s = sub_once(s, old_td_1, new_td_1, 'teardown-1')

old_td_2 = (
    '        for rid in removed:\n'
    '            self.topology.links.pop(rid, None)\n')
new_td_2 = (
    '        for rid in removed:\n'
    '            _lk = self.topology.links.pop(rid, None)\n'
    '            if _FIX_PHANTOM:\n'
    '                self._diag_drop_relay_edge(rid, _lk)\n')
s = sub_once(s, old_td_2, new_td_2, 'teardown-2')

# helper method, inserted before _forward_traffic
helper = '''    def _diag_drop_relay_edge(self, rid, lk):
        """MARL_FIX_PHANTOM: remove the routing edge of a torn-down relay link
        by its endpoints, unless another up link still joins the pair."""
        if lk is None:
            return
        a, b = lk.endpoints[0], lk.endpoints[1]
        g = self.topology.graph
        if not g.has_edge(a, b):
            return
        other = [l for l in self.topology.links.values()
                 if tuple(sorted(l.endpoints)) == tuple(sorted((a, b)))
                 and getattr(l, 'is_up', True)]
        if other:
            g[a][b]['link_id'] = other[0].id
            g[a][b]['capacity'] = other[0].capacity
        else:
            g.remove_edge(a, b)
        self._diag_phantom_fixed = getattr(self, '_diag_phantom_fixed', 0) + 1

    def _diag_count_stale_edges(self):
        links = self.topology.links
        n = 0
        for u, v, d in self.topology.graph.edges(data=True):
            lid = d.get('link_id')
            if lid is not None and lid not in links:
                n += 1
        return n

    def _forward_traffic(self, traffic_arrivals):'''
s = sub_once(s, '    def _forward_traffic(self, traffic_arrivals):', helper, 'helper')

# 3. per-tick reset and stale-edge count at the start of _forward_traffic:
#    anchor on the first statement of the body (the docstring end is not
#    stable), so insert right after the def line's docstring by matching the
#    nested consume_path definition instead.
s = sub_once(
    s, '        def consume_path(path_nodes, amount, credit=None):\n',
    '        if _DIAG_PHANTOM:\n'
    '            self._diag_stale_edges_tick = self._diag_count_stale_edges()\n'
    '            self._diag_stale_u2u_tick = 0.0\n'
    '            self._diag_stale_gen_tick = 0.0\n'
    '        def consume_path(path_nodes, amount, credit=None):\n'
    '            _stale_hop = False\n', 'forward-reset')

# 4. mark a stale hop inside the bottleneck loop
s = sub_once(
    s,
    '                    link = self.topology.links.get(link_id)\n'
    '                    if link is None:\n'
    '                        continue\n',
    '                    link = self.topology.links.get(link_id)\n'
    '                    if link is None:\n'
    '                        _stale_hop = True\n'
    '                        continue\n', 'stale-mark')

# 5. account delivered volume that crossed a stale hop (after the bottleneck
#    is final and positive; anchor on the per-node credit comment)
s = sub_once(
    s,
    '            # Per-node delivered-volume credit.  Restricted to nodes that own\n',
    '            if _DIAG_PHANTOM and _stale_hop and final_bottleneck > 0:\n'
    '                if credit == \'u2u\':\n'
    '                    self._diag_stale_u2u_tick += final_bottleneck\n'
    '                else:\n'
    '                    self._diag_stale_gen_tick += final_bottleneck\n'
    '            # Per-node delivered-volume credit.  Restricted to nodes that own\n',
    'stale-volume')

open(SIM, 'w', encoding='utf-8').write(s)

# ---------------------------------------------------------------- harness
h = open(HAR, encoding='utf-8').read()
h = sub_once(
    h, '        postcards_log.append(postcards_this_tick)\n',
    '        postcards_log.append(postcards_this_tick)\n'
    '        _diag_stale_edges_log.append(getattr(sim, "_diag_stale_edges_tick", None))\n'
    '        _diag_stale_u2u_log.append(getattr(sim, "_diag_stale_u2u_tick", None))\n'
    '        _diag_stale_gen_log.append(getattr(sim, "_diag_stale_gen_tick", None))\n',
    'har-log')
h = sub_once(
    h, '    offered_log = []\n',
    '    offered_log = []\n'
    '    _diag_stale_edges_log, _diag_stale_u2u_log, _diag_stale_gen_log = [], [], []\n',
    'har-init')
h = sub_once(
    h, "        'repoints':             int(getattr(sim, '_diag_repoints', 0)),\n",
    "        'repoints':             int(getattr(sim, '_diag_repoints', 0)),\n"
    "        'stale_edges':          _diag_stale_edges_log,\n"
    "        'stale_u2u_volume':     _diag_stale_u2u_log,\n"
    "        'stale_gen_volume':     _diag_stale_gen_log,\n"
    "        'stale_fixed':          int(getattr(sim, '_diag_phantom_fixed', 0)),\n",
    'har-out')
open(HAR, 'w', encoding='utf-8').write(h)
print('patched', SIM, HAR)
