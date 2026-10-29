"""DCC control postcards: the heartbeat, the frame model, the receiver latch
and the send invariant (RERUN_SPEC WP5, reviewer item 2).

WHAT WAS WRONG.  `DisasterControlChannel.send_postcard` admitted a send only
while its per-site counter was below one, and the counter was cleared only by
`Simulator.run_simulation`, which neither the training driver nor the
evaluation harness calls.  Each site therefore delivered ONE postcard per
episode, in training and evaluation alike, and Block B (and Block E) were
inert.  The postcard also carried a network-wide isolated-UE census that no
site could have observed, and Block E compared fragment labels the simulator
computed from its global link table (no flooding modelled, delayed or billed).

WHAT THIS MODULE FIXES (the engine hooks are in control_plane.py,
simulation.py and tick_step.py; each names this module):

  * HEARTBEAT.  Every agent site sends one postcard every POSTCARD_PERIOD
    ticks.  It is an ENVIRONMENT rule, identical for every agent arm (MARL,
    the random control, XL-DET), with no policy head: the simulator decides
    it in `Simulator._send_postcard_heartbeat`, after the site's action has
    been executed, and no actor output can suppress or add a postcard.
  * LIMITER.  The DCC's per-site send counter is keyed on the CURRENT TICK
    and resets itself on the first send of a new tick, so every driver
    (training, evaluation, the tests, the legacy run_simulation) behaves the
    same whether or not it ever calls a reset.
  * FRAME MODEL.  One frame per LIVE infrastructure link incident to the
    sender (`Link.is_up`, UE access links excluded), no fan-out cap; a frame
    is enqueued at its far end only when that node hosts an agent; each frame
    has the declared encoded size POSTCARD_ENCODED_BYTES.  The DCC keeps a
    per-tick ledger (sends, frames, bytes, deliveries), which the link-state
    service bills as the 'postcard' message type.
  * CENSUS LEAK REMOVED.  The postcard carries the sender's OWN unserved-UE
    count (`PHYMACState.unserved_ue_count`, written by
    `Simulator._update_phy_mac_observations`): UEs attached to the site over a
    live Uu link that have traffic demand and whose flows delivered nothing
    on the last forwarding pass.  The cooperative-relay field becomes "a
    fresh neighbour is relaying and reports unserved UEs".
  * RECEIVER LATCH.  Each receiver keeps the last postcard heard from each
    sender for POSTCARD_LATCH_TICKS; Block B is built from the fresh entries,
    plus a freshness field.  `postcard_received` keeps its XL-DET meaning: a
    fresh postcard arrived THIS tick.
  * INVARIANT.  `PostcardSendAudit` checks, on every tick of every driver,
    that no site sent more than one postcard in the tick, and at the end of a
    pass that every agent site sent more than one postcard after the event.
    Both drivers call `check()` and raise on a violation.

Block E (the fragment cue) and the fragment labels are REMOVED from the actor
and from the postcard; nothing in this module reads a component label.
"""

# ── Constants (pre-registered; RERUN_SPEC WP5) ─────────────────────────────

# One heartbeat per agent site every POSTCARD_PERIOD ticks.
POSTCARD_PERIOD = 10

# Receiver latch horizon: an entry heard at tick a is FRESH on ticks
# a, a+1, ..., a+POSTCARD_LATCH_TICKS-1 and expired from a+POSTCARD_LATCH_TICKS
# on -- three missed heartbeats.  The same 30-tick memory XL-DET's own
# neighbour cache has (heuristic_controller.NB_STALE_MAX).
POSTCARD_LATCH_TICKS = 30

# The declared on-the-wire encoding of one postcard frame, field by field.
# Framing (46 B): Ethernet II header + FCS 18, IPv4 20, UDP 8.
# Payload (14 B): every ControlPostcard field, quantised; see
# POSTCARD_FIELD_SLOTS for which field travels in which slot.
POSTCARD_WIRE_LAYOUT = (
    ('eth_header_fcs',        18),
    ('ipv4_header',           20),
    ('udp_header',             8),
    ('msg_type_version',       1),
    ('sender_index',           2),
    ('timestamp',              2),
    ('flags',                  1),   # relay-active bit, need level, needy class
    ('best_sinr_q',            1),
    ('relay_prb_share_q',      1),
    ('tx_power_q',             1),
    ('sinr_degradation_q',     1),
    ('unserved_ue_count_u8',   1),   # saturates at UNSERVED_UE_COUNT_MAX
    ('policy_version_u8',      1),
    ('crc16',                  2),
)
POSTCARD_ENCODED_BYTES = sum(n for _, n in POSTCARD_WIRE_LAYOUT)
assert POSTCARD_ENCODED_BYTES == 60, POSTCARD_ENCODED_BYTES

# Which wire slot carries each ControlPostcard field.  A field without a slot
# would travel for free; tests/test_wp5_postcards.py checks the two agree.
POSTCARD_FIELD_SLOTS = {
    'sender_id':              'sender_index',
    'relay_mode_active':      'flags',
    'best_sinr_to_neighbour': 'best_sinr_q',
    'unserved_ue_count':      'unserved_ue_count_u8',
    'prb_avail_for_relay':    'relay_prb_share_q',
    'policy_version':         'policy_version_u8',
    'timestamp':              'timestamp',
    'tx_power_dbm_norm':      'tx_power_q',
    'sinr_degradation_norm':  'sinr_degradation_q',
    'most_needy_class':       'flags',
    'need_level':             'flags',
}

UNSERVED_UE_COUNT_MAX = 255      # one unsigned byte

# Block B normalisations (unchanged from the evaluated tree).
NEIGHBOUR_COUNT_SCALE = 10.0     # neighbor_count_norm = min(1, n / 10)
INTERFERER_TX_NORM = 0.5         # a neighbour above mid-scale power "boosted"


class PostcardInvariantError(RuntimeError):
    """The WP5 send invariant does not hold (see PostcardSendAudit)."""


# ── The sender's own unserved-UE count (the census replacement) ────────────

def site_unserved_ue_counts(ue_serving, flows, delivered, sites):
    """{site: its OWN unserved-UE count} for every site in `sites`.

    ue_serving  {ue: [serving site, ...]} over live Uu links (the UEs
                attached to each site; a dual-connected UE lists both)
    flows       the UE-to-UE flow list [(src_ue, dst_ue), ...]: a UE with
                no flow has no demand and is never "unserved"
    delivered   the UEs that were an endpoint of a flow which delivered
                volume on the LAST forwarding pass

    A UE counts at each site it is attached to when it has demand and was
    not delivered to/from.  Every input is the serving site's knowledge of
    its OWN attached users; nothing counts a UE that is attached to no site,
    which is exactly the part of the old network-wide census (surviving UEs
    minus attached UEs) that no site could observe."""
    demand = set()
    for s_ue, t_ue in (flows or ()):
        demand.add(s_ue)
        demand.add(t_ue)
    served = delivered or ()
    out = {site: 0 for site in sites}
    for ue, serving in ue_serving.items():
        if ue in demand and ue not in served:
            for site in set(serving):
                if site in out:
                    out[site] += 1
    return out


# ── Receiver latch ─────────────────────────────────────────────────────────

class NeighbourLatch:
    """Per-receiver, per-sender table of the last postcard heard.

    `ingest` is called once per tick per receiver with the postcards the DCC
    delivered to it on that tick (single-threaded, in agent order, right
    after the drain).  `fresh` and `arrived` only READ, so the per-agent
    observation builders may call them concurrently.  Re-ingesting an empty
    list is a no-op, which is what makes a second observation build in the
    same tick (AgentMonitorPlotter) return the same Block B.
    """

    def __init__(self, horizon=POSTCARD_LATCH_TICKS):
        self.horizon = int(horizon)
        self._table = {}      # receiver -> {sender: (arrival_tick, postcard)}
        self._arrival = {}    # receiver -> last tick a postcard arrived

    def ingest(self, receiver, postcards, tick):
        if not postcards:
            return
        row = self._table.setdefault(receiver, {})
        for pc in postcards:
            # Several frames of one postcard (parallel links) collapse here:
            # the entry is keyed on the SENDER.
            row[pc.sender_id] = (int(tick), pc)
        self._arrival[receiver] = int(tick)

    def arrived(self, receiver, tick):
        """True iff a postcard arrived at `receiver` on `tick` (the
        `postcard_received` contract XL-DET reads)."""
        return self._arrival.get(receiver) == int(tick)

    def fresh(self, receiver, tick):
        """[(sender, age, postcard)] for every entry heard less than
        `horizon` ticks ago, sorted by sender id (a fixed order, so the
        Block B means are summed in the same order on every run)."""
        tick = int(tick)
        out = []
        for sender, (t_arr, pc) in (self._table.get(receiver) or {}).items():
            age = tick - t_arr
            if 0 <= age < self.horizon:
                out.append((sender, age, pc))
        out.sort(key=lambda e: e[0])
        return out

    def arrivals(self, receiver, tick):
        """Number of distinct senders whose postcard arrived on `tick`."""
        return sum(1 for _, age, _ in self.fresh(receiver, tick) if age == 0)


def fill_block_b(nb, entries, arrived_now, horizon=POSTCARD_LATCH_TICKS):
    """Write the neighbour fields of `nb` (a NeighborRadioSummary) from the
    FRESH latch entries [(sender, age, postcard)].

    Written here, with the name each actor field reads (RERUN_SPEC WP6 Block
    B, branch trackAgent `_BLOCK_B_FIELDS`):

      avg_prb_util                 b_neighbour_relay_share: mean relay PRB
                                   share the fresh neighbours report (the
                                   dataclass name is historical)
      avg_sinr                     b_neighbour_sinr_mean
      best_relay_sinr              b_neighbour_sinr_best
      neighbor_count_norm          b_fresh_neighbour_count: fresh DISTINCT
                                   senders / 10, capped at 1
      any_relay_active             b_any_neighbour_relaying
      postcard_freshness_norm      b_neighbour_freshness: 1 - (age of the
                                   freshest entry) / horizon; 0 with no fresh
                                   entry
      coordination_quality         b_relaying_neighbour_fraction
      neighbour_avg_tx_power_norm  b_neighbour_tx_power_mean
      interferer_count_norm        b_interferer_fraction: fraction of fresh
                                   neighbours above mid-scale power
      cooperative_relay_assigned   a fresh neighbour is relaying AND reports
                                   unserved UEs of its own (not an actor
                                   field after WP6)
      postcard_received            1.0 iff a postcard arrived THIS tick

    `experienced_sinr_drop_norm` (b_own_sinr_drop) is the site's OWN SINR
    drop and is written by the simulator, not from postcards.  With no fresh
    entry every field keeps its dataclass default ("nothing heard").
    """
    nb.postcard_received = 1.0 if arrived_now else 0.0
    if not entries:
        return nb
    n = len(entries)
    pcs = [pc for _, _, pc in entries]
    relaying = [pc for pc in pcs if pc.relay_mode_active]
    sinr = [float(pc.best_sinr_to_neighbour) for pc in pcs]
    tx = [float(pc.tx_power_dbm_norm) for pc in pcs]
    nb.avg_prb_util = sum(float(pc.prb_avail_for_relay) for pc in pcs) / n
    nb.avg_sinr = sum(sinr) / n
    nb.best_relay_sinr = max(sinr)
    nb.neighbor_count_norm = min(1.0, n / NEIGHBOUR_COUNT_SCALE)
    nb.any_relay_active = 1.0 if relaying else 0.0
    nb.coordination_quality = len(relaying) / n
    nb.neighbour_avg_tx_power_norm = sum(tx) / n
    nb.interferer_count_norm = sum(1 for x in tx if x > INTERFERER_TX_NORM) / n
    nb.cooperative_relay_assigned = 1.0 if any(
        int(pc.unserved_ue_count) > 0 for pc in relaying) else 0.0
    youngest = min(age for _, age, _ in entries)
    nb.postcard_freshness_norm = max(0.0, 1.0 - youngest / float(horizon))
    return nb


# ── Send invariant ─────────────────────────────────────────────────────────

class PostcardSendAudit:
    """The engine-side WP5 invariant, fed by `tick_step.step_tick` on every
    tick of every driver from the simulator's cumulative per-site send
    counter (`Simulator._postcards_by_site`), NOT from the DCC's own limiter
    counter -- so a limiter that let a second send through would be caught
    here rather than agreeing with itself.

      * per tick: no site sends more than one postcard (raises at once);
      * per pass (`check`): every site that hosted an agent on more than
        POSTCARD_PERIOD acting ticks sent more than one postcard over them.

    "Acting ticks" are the ticks on which the driver's action step ran
    (the action lock was open), i.e. the post-event window; a site present
    on POSTCARD_PERIOD acting ticks or fewer is listed, not failed, because
    one heartbeat is all it can owe.  A pass with no acting tick (a routing
    or SDN arm, or a pre-event-only run) reports `applicable: False`.

    PER SITE since D13.2.  The action lock opens for each site on its OWN
    local isolation flag, so `record` is given the sites whose action step
    RAN on the tick: a site counts as present only on its own acting ticks,
    and a send by a site whose action step did not run -- a locked site --
    raises, exactly as a send on a tick with no action step always did.
    """

    def __init__(self, period=POSTCARD_PERIOD):
        self.period = int(period)
        self.onset = None          # first acting tick
        self.acting_ticks = 0
        self.present = {}          # site -> acting ticks it hosted an agent
        self.sends = {}            # site -> sends on acting ticks
        self.max_per_tick = 0
        self.ticks_seen = 0

    def record(self, tick, acted, agent_sites, before, after):
        """One tick: `before`/`after` are the cumulative per-site send
        counters around the tick's action step; `agent_sites` are the sites
        whose action step ran on it (D13.2: the per-site lock), and `acted`
        whether any did."""
        self.ticks_seen += 1
        before = before or {}
        after = after or {}
        acting = set(agent_sites or ()) if acted else set()
        for site in sorted(set(before) | set(after)):
            d = int(after.get(site, 0)) - int(before.get(site, 0))
            if d < 0:
                raise PostcardInvariantError(
                    'postcard audit: the cumulative send counter of %r fell '
                    'by %d at tick %d; it must never be reset' % (site, -d, tick))
            if d > 1:
                raise PostcardInvariantError(
                    'postcard audit: site %r sent %d postcards at tick %d '
                    '(at most one per site per tick)' % (site, d, tick))
            if d and not acted:
                raise PostcardInvariantError(
                    'postcard audit: site %r sent a postcard at tick %d, on '
                    'which no action step ran' % (site, tick))
            if d and site not in acting:
                raise PostcardInvariantError(
                    'postcard audit: site %r sent a postcard at tick %d, on '
                    'which its own action step did not run (its action lock '
                    'was closed)' % (site, tick))
            if d > self.max_per_tick:
                self.max_per_tick = d
            if d:
                self.sends[site] = self.sends.get(site, 0) + d
        if not acted:
            return
        if self.onset is None:
            self.onset = int(tick)
        self.acting_ticks += 1
        for site in agent_sites:
            self.present[site] = self.present.get(site, 0) + 1

    def report(self):
        """A plain dict (goes into the result pickle as `postcard_audit`)."""
        due = sorted(s for s, k in self.present.items() if k > self.period)
        short = sorted(s for s, k in self.present.items() if k <= self.period)
        per_site = {s: int(self.sends.get(s, 0)) for s in due}
        failing = sorted(s for s in due if per_site[s] <= 1)
        return {
            'applicable': bool(self.acting_ticks > self.period),
            'period': self.period,
            'onset_tick': self.onset,
            'acting_ticks': int(self.acting_ticks),
            'ticks_seen': int(self.ticks_seen),
            'sites_checked': len(due),
            'sites_short_presence': short,
            'min_per_site': (min(per_site.values()) if per_site else None),
            'max_per_tick': int(self.max_per_tick),
            'failing_sites': failing,
        }

    def check(self):
        """Raise PostcardInvariantError unless the pass satisfies the
        invariant; return the report.  Not applicable (no acting window
        longer than one period): returns the report without raising."""
        rep = self.report()
        if not rep['applicable']:
            return rep
        if rep['sites_checked'] == 0:
            raise PostcardInvariantError(
                'postcard audit: %d acting ticks but no agent site was '
                'present on more than %d of them, so nothing was checked'
                % (rep['acting_ticks'], self.period))
        if rep['failing_sites']:
            raise PostcardInvariantError(
                'postcard audit: %d site(s) sent at most one postcard after '
                'the event (need > 1 per site): %s'
                % (len(rep['failing_sites']),
                   ', '.join('%s=%d' % (s, self.sends.get(s, 0))
                             for s in rep['failing_sites'][:10])))
        return rep
