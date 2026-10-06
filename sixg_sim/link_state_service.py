"""Link-state service for the cross-layer arms (RERUN_SPEC WP3, reviewer item 4).

Replaces `MARLLinkStateDetector` (sixg_sim/link_state_detector.py, deleted by
this work package) behind the same public API -- `step`,
`get_routable_flows`, `believed_graph`, `visible` -- and adds the three the
spec names: `tick_messages`, `installed_since` and `isolation_state`.  Every
arm on the MARL code path (marl_static, marl_freeze, xldet, random,
untrained) uses it, in TRAINING and in EVALUATION, through
`make_link_state_service`.

WHAT THE OLD DETECTOR WAS, AND WHY IT HAD TO GO
-----------------------------------------------
It delayed each link's state by three ticks and then ran an instantaneous,
global shortest-path computation over the result.  Nothing was charged for
the keepalives that would supply that belief, nothing was charged or delayed
for distributing it, and three bypasses gave parts of the topology away for
free:

  * any link id created after tick 0 was adopted with NO delay, so only the
    Priority-1 bridges (created physically down) ever paid the bring-up cost;
  * a link REMOVED from `topology.links` was dropped from the belief at once,
    so a teardown was omniscient;
  * the training driver ran with no detector at all (WP2 fixed that half).

WHAT THIS IS
------------
An OSPF-with-BFD-class link-state service, built from the constants of the
OSPF BASELINE model in this repository (`ControlPlaneMessageSim._sim_ospf`),
so the cross-layer arms and the OSPF baseline are timed and billed by one set
of numbers.  `docs/LINK_STATE_SERVICE.md` writes those constants out for the
pre-registration, and `tests/test_wp3_link_state_service.py` asserts they
still equal the harness's.

Three layers, in this order:

  1. BFD liveness, per link.  A link's failure is confirmed after
     BFD_DETECT_TICKS = 3 (1 packet/s per direction, detect multiplier 3);
     a link's adjacency comes up 3 ticks after the session starts.  After
     island onset a NEVER-SEEN link id starts DOWN -- there is no free
     adoption.  A link torn down by one of its OWN endpoints is known to that
     endpoint at once (an administrative change needs no timeout).
  2. Adjacency, per PAIR of endpoints: the pair is up iff some link of the
     pair is believed up.  This is the granularity `Topology.sync_pair` and
     `pair_representative` already use, so belief and truth name the same hop.
  3. Installation.  ONE Router-LSA PER ROUTER, NOT PER PAIR TRANSITION
     (RFC 2328 S12.4 / S12.4.1; docs/DECISIONS.md D13.1).  Every
     infrastructure endpoint o of a confirmed pair transition learns of it at
     t_known[o] (the confirmation tick; the administrative rule below dates
     the acting endpoint's knowledge earlier than the remote one's) and keeps
     it as PENDING until an LSA of its own carries it.  o originates ONE
     Router-LSA -- which describes ALL of its links -- at
         t_o    = max(t_known of o's EARLIEST pending transition,
                      last_orig[o] + MIN_LS_INTERVAL)
     carrying EVERY transition o knows of by t_o; one it learns after t_o
     waits for o's next origination.  A transition is installed NETWORK-WIDE
     at
         t_inst = max_o(t_o + FLOOD_HOP_TICKS * ecc_o) + SPF_DELAY_TICKS,
     over its originating endpoints o, with t_o the origination of the LSA
     that CARRIES it and ecc_o the BFS eccentricity of o in the INSTALLED
     infrastructure graph at t_o -- between ROUTERS, the nodes a flood can
     actually traverse (D3.17).  So k simultaneous transitions at one router
     are announced together, as the OSPF baseline model (`_sim_ospf`: "at
     most 1 LSA origination per NODE per 5 ticks") announces them, instead
     of one per MIN_LS_INTERVAL.  Forwarding and the flow gate use the
     INSTALLED view; `believed_graph()` returns None when the installed view
     equals truth.

Tombstones.  A Link popped from `topology.links` keeps its (endpoints,
capacity) in `Simulator._link_tombstones` and counts as PHYSICALLY DOWN until
its removal is installed.  While the installed view still holds the hop, the
believed graph carries the tombstone's id and capacity, and `consume_path`
delivers exactly 0.0 over it: the believed graph never routes over a
physically-down link.  This is the hook WP1 left marked in `consume_path`.

Which Link carries a hop.  Belief decides whether a pair is a hop AT ALL --
no connectivity is adopted without the BFD session and the install -- but
once it is one, the Link that carries it is the one PHYSICS names, i.e. the
same representative the engine's own true graph uses.  Otherwise a relay in
its bring-up window (it repurposes a pair that already has a live link) would
be forwarded over at the underlying link's capacity, which the hop no longer
physically has.  A pair with nothing physically up keeps its believed
(physically-down) representative, which is the blackhole above.  D3.15.
  The substitution may not RESURRECT a hop, so it applies only while some
link of the pair is BOTH believed up and physically up (D3.22, narrowed by
D3.31): when nothing the service believes up is physically up the hop
blackholes, and substituting a live Link of the same pair there would adopt it
with no BFD session and no install -- the free adoption D3.16 closed on the
other side of the same rule.  It is measured never to raise the capacity a
hop offers in the D3.15 case, but that is a property of these topologies and
not of the relay-first/lowest-id ordering rule (D3.32).

`visible` and `installed_since` stamp a Link in EITHER of the two ways it can
be in the forwarding view (D3.23): the service believes it up inside an
installed hop, or the view NAMES it as the carrier.  So the service's own "is
this link in my forwarding view?" API cannot contradict the view it publishes,
and a hop that blackholes under an UNDETECTED failure keeps the first stamp --
belief may not consult truth the service has not detected (D3.30).  The
engine's `_relay_route_since` mirror follows `installed_since` in BOTH
directions and only for ids the engine treats as relay links, because that
dict is relay-scoped and WP4 reads it (D3.33).

Billing (`tick_messages`), FROM ISLAND ONSET ONLY -- pre-event billing is
exactly zero, for this service as for every baseline protocol model:

  * BFD        2 x 66 B per INFRASTRUCTURE adjacency pair RUNNING A SESSION,
               per tick: confirmed up, or with a session in progress on one of
               its links (a 3-tick bring-up, or a failure inside its detect
               multiplier).  D3.24.  This DEPARTS from RERUN_SPEC WP3's "per
               believed-up infrastructure adjacency pair per tick" and from
               the identical wording of rerun_map.json / rerun_details.txt
               item 4b; the divergence is recorded in D3.24/D3.34 and must
               carry that qualifier wherever the pre-registration quotes the
               spec's sentence.
  * hellos     one broadcast per surviving infrastructure router per
               HELLO_INTERVAL at its crc32 offset, 44 + 4 x infra-neighbour
               count bytes -- `_sim_ospf`'s rule, size and offset, at the
               STEADY-STATE HelloInterval throughout (D3.25: the baseline
               drops to OSPF_HELLO_FAST_INTERVAL = 1 and re-phases its offset
               on that interval while its measured control plane converges;
               this service models no convergence state machine)
  * Router-LSA one message per DIRECTED believed-up infrastructure adjacency
               in the originator's believed component, each
               24 + 12 x (originator infra degree) bytes, PER ORIGINATION --
               one origination per router per MIN_LS_INTERVAL at most, however
               many transitions it carries (D13.1).  No LSAck and no
               periodic refresh -- the same omissions the OSPF baseline model
               makes, stated rather than hidden.
  * postcards  the DisasterControlChannel ledger (RERUN_SPEC WP5): one
               60-B frame per live infrastructure link of each site sending
               its heartbeat on the tick (postcard_ledger).

The phantom overhead floor and the change-detection proxy are GONE: overhead
for the MARL-path arms is now the sum of these billed bytes and nothing else.

OVERHEAD IS DESCRIPTIVE, NEVER A COMPARATIVE KPI.  The numbers this produces
describe what this modelled service would send.  They may not be used to rank
arms across protocol families: the baselines' models undercount (OLSR TC
re-forwarding is not billed, OSPF has no periodic LSA refresh) and overcount
(the OSPF hello and flood scope count UE neighbours) in ways this one does
not.  Only MARL against XL-DET is within one service and one billing model,
and even that is reported descriptively.  Nothing in this module, in the
exported series names or in the harness's use of them may be read as a
comparative endpoint.

WHAT THIS SERVICE DELIBERATELY DOES NOT MODEL
---------------------------------------------
  * Per-router LSDBs and per-source forwarding views.  Installation is
    global and conservative (the farthest originator sets the time), which
    OVER-charges flows near the change.  A-full was rejected as out of scope.
  * The baselines keep their own protocol models and get NO BFD.  Their gates
    still block every flow at once while this service gates per flow -- an
    asymmetry that favours the cross-layer arms in transients.  It is
    disclosed in docs/DECISIONS.md, not fixed here.
  * An OSPF-style convergence state machine, and with it the FAST-HELLO
    transient.  `_sim_ospf` bills hellos at OSPF_HELLO_FAST_INTERVAL = 1 while
    its measured control plane reports convergence_fraction < 1, and phases
    each node's crc32 offset on whichever interval is in force; this service
    has no such state and bills HELLO_INTERVAL = 10 at its steady-state phase
    throughout.  The two hello models therefore differ by 10x in rate and by
    the per-node phase during a baseline transient -- a difference in the
    cross-layer arms' FAVOUR.  D3.25.
"""
import os
import zlib
from collections import deque

from .topology import is_transport_relay

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS -- all of them from the OSPF baseline model of
# run_timeline_comparison.py (`_sim_ospf` and the timer block above it).
# They are duplicated here, not imported, because run_timeline_comparison
# imports sixg_sim; the equality is asserted by
# tests/test_wp3_link_state_service.py::test_constants_match_the_ospf_baseline
# so the two copies cannot drift.  docs/LINK_STATE_SERVICE.md writes them out
# for the pre-registration.
# ══════════════════════════════════════════════════════════════════════

# ── BFD (RFC 5880) ───────────────────────────────────────────────────
# 1 control packet per second per direction, detect multiplier 3.  The
# 3-tick figure is NOT the theoretical floor (a 300 ms interval would make
# it sub-second); it is the same "3 missed keepalives" rule the baselines
# are held to, and it keeps the cross-layer arms ~13x faster than OSPF's
# 40 s dead interval instead of infinitely faster.
BFD_DETECT_TICKS = 3
# 24 B RFC 5880 control packet + 8 UDP + 20 IPv4 + 14 L2
BFD_BYTES = 66
BFD_PACKETS_PER_PAIR_PER_TICK = 2      # one per direction

# ── OSPF timers, in TICKS (1 tick = 1 s) ─────────────────────────────
HELLO_INTERVAL = 10          # OSPF_HELLO_INTERVAL (RFC 2328 broadcast default).
                             # Steady state throughout: the baseline's
                             # fast-hello transient (interval 1, re-phased
                             # offset) is not modelled here.  D3.25.
FLOOD_HOP_TICKS = 1          # OSPF_FLOOD_HOP_TICKS (flood pacing per hop)
SPF_DELAY_TICKS = 5          # OSPF_SPF_DELAY (SPF throttle + FIB install)
MIN_LS_INTERVAL = 5          # OSPF_MIN_LS_INTERVAL (RFC 2328 S12.4)

# ── OSPF message sizes ───────────────────────────────────────────────
HELLO_BASE = 44              # ControlPlaneMessageSim.OSPF_HELLO_BASE
HELLO_PER_NBR = 4            # ControlPlaneMessageSim.OSPF_HELLO_PER_NBR
LSA_BASE = 24                # ControlPlaneMessageSim.OSPF_LSA_BASE
LSA_PER_LINK = 12            # ControlPlaneMessageSim.OSPF_LSA_PER_LINK

# ── Per-message CPU cost (mJ), the OSPF branch's own weights ─────────
# Simple packet (hello, BFD control) 0.5; complex computation (LSA receipt
# -> LSDB update -> SPF) 2.0; postcard parse + NN inference 0.3.  The
# harness's control-plane energy term takes the weighted mean over the
# types actually billed this tick.
CPU_MJ_BFD = 0.5
CPU_MJ_HELLO = 0.5
CPU_MJ_LSA = 2.0
CPU_MJ_POSTCARD = 0.3

# Node types that run no routing protocol on the mesh: the same exclusion
# ControlPlaneMessageSim._get_survivor_nodes makes, so hellos are billed for
# exactly the routers the OSPF baseline bills them for.
NON_ROUTER_TYPES = frozenset({"Core", "UPF", "AMF", "CoreUPF", "UE"})

# ── Ablation and sensitivity hooks, NOT tuning knobs ─────────────────
# MARL_DETECT_TICKS=0 disables the service completely
# (make_link_state_service returns None): the omniscient ablation, true graph
# and no flow gate.  Any value > 0 sets the detection interval.
MARL_BFD_DETECT_TICKS = int(os.environ.get('MARL_DETECT_TICKS',
                                           BFD_DETECT_TICKS))
# MARL_LS_INSTALL=0 is the PRE-REGISTERED EVALUATION-ONLY SENSITIVITY of
# RERUN_SPEC WP3: installation off, i.e. idealised timing (a confirmed
# transition takes effect at once).  Detection and billing are unchanged, so
# it isolates the cost of the flood + SPF model alone.  It defaults to ON:
# the evaluated system is the one WITH installation.
MARL_LS_INSTALL = os.environ.get('MARL_LS_INSTALL', '1').strip().lower() \
    not in ('0', 'false', 'no', 'off')

# The arms that run the MARL code path.  'marl_freeze', 'random' and 'xldet'
# all rebind mode = 'marl' inside run_physics_pass, so the single
# mode == 'marl' test that creates the service covers all of them.  Equal
# detection and installation latency is what makes the comparison about the
# policy rather than about who noticed the severance first.
MARL_DETECTION_ARMS = ('marl', 'marl_freeze', 'marl_static', 'random', 'xldet')

# The message types tick_messages() reports.  'postcard' is the DCC ledger
# (RERUN_SPEC WP5; postcard_ledger below).
MESSAGE_TYPES = ('bfd', 'hello', 'lsa', 'postcard')  # wp7-scan-ok: WP5 DCC message type, not an action head
_CPU_MJ = {'bfd': CPU_MJ_BFD, 'hello': CPU_MJ_HELLO,
           'lsa': CPU_MJ_LSA, 'postcard': CPU_MJ_POSTCARD}  # wp7-scan-ok: WP5 DCC message type, not an action head


def zero_messages():
    """An all-zero `tick_messages` breakdown (the pre-event value, and the
    value for any arm that does not run this service)."""
    out = {'msgs': 0, 'bytes': 0, 'cpu_mj_per_msg': 0.0}
    for t in MESSAGE_TYPES:
        out[t + '_msgs'] = 0
        out[t + '_bytes'] = 0
    return out


def postcard_ledger(sim, tick):
    """The DCC postcard ledger of `tick` ({'sends', 'frames', 'bytes',
    'deliveries', 'refused'}; RERUN_SPEC WP5), all zero when the simulator
    has no DCC or sent nothing on that tick."""
    dcc = getattr(getattr(sim, 'control_plane', None), 'dcc', None)
    if dcc is None or not hasattr(dcc, 'ledger'):
        return {'sends': 0, 'frames': 0, 'bytes': 0, 'deliveries': 0,
                'refused': 0}
    return dcc.ledger(tick)


def postcard_only_messages(sim, tick):
    """A `tick_messages` breakdown holding the DCC postcards of `tick` and
    nothing else: the bill of a MARL-path arm that runs no link-state service
    (the MARL_DETECT_TICKS=0 ablation), whose sites still send the heartbeat."""
    out = zero_messages()
    pc = postcard_ledger(sim, tick)
    out['postcard_msgs'] = pc['frames']
    out['postcard_bytes'] = pc['bytes']
    out['msgs'] = out['postcard_msgs']
    out['bytes'] = out['postcard_bytes']
    if out['msgs'] > 0:
        out['cpu_mj_per_msg'] = float(CPU_MJ_POSTCARD)
    return out


def node_offset(nid, interval):
    """`ControlPlaneMessageSim._get_node_offset`, verbatim: a deterministic
    per-node timer phase.  crc32, NOT the builtin hash(), which is salted per
    process (PYTHONHASHSEED) and so gave every arm and every re-run a
    different hello phasing."""
    return zlib.crc32(('%s|%s' % (nid, interval)).encode('utf-8')) % max(1, interval)


class LinkStateService:
    """The forwarding view, flow gate and control-plane bill of one arm.

    See the module docstring for the model.  The three-layer state is:

      _link_belief[lid]     BFD layer: what the service believes about one link
      _pair_confirmed[pair] adjacency layer: up iff some link of the pair is
                            believed up
      _pair_installed[pair] installed layer: what forwarding and the flow gate
                            actually use, after the LSA flood and SPF delay

    ACCESS (Uu) ADJACENCIES ARE NOT IN THE LINK-STATE DOMAIN.  A pair with a
    UE endpoint follows physical truth with no detection and no installation
    delay: the anchor observes its own radio link directly (radio-link failure
    is a physical-layer indication, not a protocol), it is the only node whose
    forwarding depends on it, and the spec's billing model is infrastructure-
    only throughout ("infrastructure adjacency pair", "infra-neighbour count",
    "each infrastructure endpoint originates").  Modelling a UE attachment as
    a flooded Router-LSA would delay every attach and detach by the full
    flood + SPF time for no modelled message.  Recorded as D3.3.
    """

    def __init__(self, sim, detect_ticks=None, install=None):
        self.sim = sim
        self.detect_ticks = max(1, int(MARL_BFD_DETECT_TICKS
                                       if detect_ticks is None
                                       else detect_ticks))
        self.install_enabled = (MARL_LS_INSTALL if install is None
                                else bool(install))

        # ── layer 1: BFD ───────────────────────────────────────────
        self._link_belief = {}      # lid -> believed up
        self._pending_link = {}     # lid -> (new_state, first observed tick)
        self._admin_down = {}       # lid -> (acting endpoint, tick)
        # ── layer 2: adjacency ─────────────────────────────────────
        self._pair_confirmed = {}   # pair -> bool
        self._pair_admin = {}       # pair -> (acting endpoint, tick)
        # ── layer 3: installation ──────────────────────────────────
        self._pair_installed = {}   # pair -> bool
        self._pending_install = {}  # pair -> (state, t_inst): every LSA
                                    # that carries it has been originated
        self._awaiting = {}         # pair -> the transition still waiting
                                    # for an LSA of one of its originators:
                                    # {'id', 'state', 'waiting', 'arrival'}
        self._unannounced = {}      # router -> [(t_known, pair, id), ...]:
                                    # transitions it knows of (or will, at
                                    # t_known) that no LSA of its own has
                                    # carried yet
        self._trans_seq = 0         # transition ids (a revert supersedes)
        self._last_orig = {}        # node -> tick of its last LSA origination
        self._lsa_schedule = {}     # tick -> [originator, ...] (billing)

        # ── derived, refreshed every step ──────────────────────────
        self.visible = {}           # lid -> is it in the INSTALLED view
                                    # (believed up inside an installed hop,
                                    #  or named as that hop's carrier; D3.23)
        self._installed_since = {}  # lid -> tick it entered the installed view
        self._view_changes = []     # [(u, v, link_id or None, capacity)]
        self._stale_ids = frozenset()   # in the view, physically down Link
        self._tomb_ids = frozenset()    # in the view, Link already popped
        self.routable_flows = set()
        self.isolated = {}          # site -> bool
        self.isolated_since = {}    # site -> tick or None

        # ── bookkeeping ────────────────────────────────────────────
        self.detections = 0         # confirmed link transitions (diagnostics)
        self.installs = 0           # installed pair transitions (diagnostics)
        self.originations = 0       # Router-LSAs originated (diagnostics)
        self.announced = 0          # transition announcements they carried
        self.island_onset_tick = None
        self._bootstrapped = False
        self._last_tick = None
        self.last_messages = zero_messages()
        # Per-tick caches.  The two adjacency maps and the eccentricities are
        # walked once per originator by the installation formula and once per
        # originator by the LSA billing, so they are built at most once per
        # tick and dropped whenever the state behind them moves.
        self._ecc_cache = {}
        self._adj_confirmed = None
        self._adj_installed = None

    # ══════════════════════════════════════════════════════════════
    # the per-tick step
    # ══════════════════════════════════════════════════════════════

    def step(self, tick):
        """Advance the service by one tick.  Called by `Simulator.step_tick`
        at phase 6, after the agents have acted and before forwarding, in
        BOTH drivers."""
        tick = int(tick)
        links = self.sim.topology.links
        tombs = getattr(self.sim, '_link_tombstones', None)
        if tombs is None:
            tombs = {}

        # Island onset: the first tick on which the simulator reports island
        # mode.  Billing starts here and not before, so pre-event billing is
        # exactly zero -- the same rule ControlPlaneMessageSim applies to
        # every baseline model.
        if self.island_onset_tick is None and bool(
                getattr(self.sim, 'island_mode', False)):
            self.island_onset_tick = tick

        self._invalidate_caches()
        self._adj_confirmed = None

        self._drain_admin_events(tick)
        self._apply_matured_installs(tick)
        self._step_bfd(tick, links, tombs)
        self._step_adjacency(tick, links, tombs)
        self._originate(tick)                   # one Router-LSA per router
        self._apply_matured_installs(tick)      # immediate installs
        self._refresh_view(tick, links, tombs)
        self._update_isolation(tick)
        self._recompute_routable_flows()
        self.last_messages = self._bill(tick)
        self._last_tick = tick
        self._bootstrapped = True

    # ── layer 1: BFD ──────────────────────────────────────────────

    def _drain_admin_events(self, tick):
        """Read this tick's administrative link events from the engine.

        `Simulator.record_link_admin_event` records a teardown together with
        the endpoint that performed it.  A link torn down by one of its OWN
        endpoints is known to that endpoint at once -- it shut the interface,
        it does not wait for a keepalive to time out.

        THERE IS NO CORRESPONDING RULE FOR CREATION, and the omission is
        deliberate.  RERUN_SPEC WP3 also says "a new link's adjacency up after
        the BFD session (3 ticks)", and every relay link in this simulator is
        created BY one of its endpoints, so reading "known at once" as
        "believed up at once" would make that sentence dead for exactly the
        links it is about.  rerun_details.txt resolves it explicitly -- "An
        up-transition still needs the BFD session (3)" -- so the
        administrative shortcut applies to the DOWN direction only.  What the
        creating endpoint does know at once is that the link EXISTS: its
        session timer starts at the creation tick, not at some later
        discovery, which is what this service does by construction (it is
        stepped every tick, so first sight IS the creation tick).  D3.2.
        """
        ev = getattr(self.sim, '_link_admin_events', None)
        if not ev:
            return
        ev_tick, events = ev
        if int(ev_tick) != tick:
            return
        for lid, node, up in events:
            if not up:
                self._admin_down[lid] = (node, tick)

    def _step_bfd(self, tick, links, tombs):
        """Update the per-link belief from physical truth."""
        for lid, link in links.items():
            self._observe_link(tick, lid, bool(getattr(link, 'is_up', True)),
                               getattr(link, 'endpoints', ()))
        # Links that are no longer in topology.links.  A tombstoned link is
        # PHYSICALLY DOWN and is observed as such (the old detector dropped it
        # from the belief at once, which made every teardown omniscient);
        # anything else has simply ceased to exist and is forgotten.
        for lid in list(self._link_belief):
            if lid in links:
                continue
            tb = tombs.get(lid)
            if tb is None:
                self._forget_link(lid)
                continue
            self._observe_link(tick, lid, False, tb.get('endpoints', ()))

    def _observe_link(self, tick, lid, true_up, endpoints):
        access = self._is_access(endpoints)
        prev = self._link_belief.get(lid)

        if prev is None:
            # NEVER SEEN.  Before island onset -- and on the first step ever,
            # which is the pre-converged initial state every baseline
            # controller also gets from its own mark_pre_converged path -- the
            # service adopts truth.  AFTER island onset a never-seen link id
            # starts DOWN: there is no free adoption, which is the bypass that
            # let every Priority-2 relay link appear with zero delay.
            if access or not self._bootstrapped or self.island_onset_tick is None:
                self._link_belief[lid] = true_up
            else:
                self._link_belief[lid] = False
                if true_up:
                    self._pending_link[lid] = (True, tick)
            return

        if access:
            # Not in the link-state domain: the anchor observes its own radio
            # link directly (see the class docstring).
            self._link_belief[lid] = true_up
            self._pending_link.pop(lid, None)
            return

        if true_up == prev:
            # A flap that reverted before the detector confirmed it -- which
            # is exactly what a detect multiplier is for.
            self._pending_link.pop(lid, None)
            return

        if not true_up and lid in self._admin_down:
            # Administrative teardown by one of its own endpoints: confirmed
            # at once, with the acting endpoint recorded so that the REMOTE
            # endpoint's Router-LSA still pays the BFD timeout.
            self._link_belief[lid] = False
            self._pending_link.pop(lid, None)
            self.detections += 1
            pair = self._pair_of(endpoints)
            if pair is not None:
                self._pair_admin[pair] = self._admin_down[lid]
            return

        pend = self._pending_link.get(lid)
        if pend is None or pend[0] != true_up:
            self._pending_link[lid] = (true_up, tick)
        elif tick - pend[1] >= self.detect_ticks:
            self._link_belief[lid] = true_up
            self._pending_link.pop(lid, None)
            self.detections += 1

    def _forget_link(self, lid):
        self._link_belief.pop(lid, None)
        self._pending_link.pop(lid, None)
        self._admin_down.pop(lid, None)
        self._installed_since.pop(lid, None)
        self._unstamp_engine_mirror(lid)        # both copies, always (D3.33)
        self.visible.pop(lid, None)

    # ── layer 2 + 3: adjacency and installation ───────────────────

    def _step_adjacency(self, tick, links, tombs):
        """Recompute the confirmed pair state and schedule installations."""
        pair_lids = self._pairs_and_links(links, tombs)
        for pair in sorted(set(pair_lids) | set(self._pair_confirmed)):
            lids = pair_lids.get(pair, ())
            up = any(self._link_belief.get(l, False) for l in lids)
            prev = self._pair_confirmed.get(pair)
            self._pair_confirmed[pair] = up
            if prev is None:
                # First sight of this adjacency.  On the FIRST step that is
                # the pre-converged initial topology, which installs with it;
                # later (a relay bridge, a rescue UE's anchor) the state that
                # arrives here has already paid its BFD session, so the
                # transition below is what schedules the flood.
                if not self._bootstrapped:
                    self._set_installed(pair, up)
                    continue
                prev = False
                if not up:
                    self._pair_installed.setdefault(pair, False)
                    continue
            if up == prev:
                continue
            self._schedule_install(tick, pair, up)
        # forget pairs that have neither links nor a belief left
        for pair in [p for p, up in self._pair_confirmed.items()
                     if not up and p not in pair_lids]:
            self._pair_confirmed.pop(pair, None)
            self._pair_installed.pop(pair, None)
            self._pending_install.pop(pair, None)
            # Its originators still announce the transition (and are billed
            # for it); `_originate` finds no record to install.
            self._awaiting.pop(pair, None)
            self._pair_admin.pop(pair, None)
            self._invalidate_caches()
        # An administrative record that this tick's transitions did not
        # consume is about a link, not about a pair transition; keeping it
        # would let a LATER transition claim a knowledge time that has
        # already gone by.
        for p in [p for p, (_n, t) in self._pair_admin.items() if t != tick]:
            self._pair_admin.pop(p, None)
        self._adj_confirmed = None

    def _schedule_install(self, tick, pair, new_state):
        """Hand one confirmed pair transition (at tick td = `tick`) to its
        originating endpoints; `_originate` announces it and dates the
        installation (D13.1, RFC 2328 S12.4):

            t_o    = max(t_known of o's earliest pending transition,
                         last_orig[o] + MIN_LS_INTERVAL)
            t_inst = max_o(t_o + FLOOD_HOP_TICKS * ecc_o) + SPF_DELAY_TICKS

        with t_o the origination of the LSA of o that CARRIES this
        transition.  `t_known[o]` is td for both endpoints of an ordinary
        transition, and for an administrative teardown it is the
        administrative tick at the acting endpoint and that tick +
        BFD_DETECT_TICKS at the remote one -- which is the only place where
        "known to that endpoint at once" is observable.

        What changed at WP13 (D13.1): until then EACH transition scheduled
        its OWN origination per endpoint, serialised at MIN_LS_INTERVAL, so
        an endpoint with k simultaneous transitions announced the last one
        MIN_LS_INTERVAL * (k - 1) ticks after the first -- a slower and less
        faithful control plane than the OSPF baseline the cross-layer arms
        are compared with.  Now the transition joins the endpoint's pending
        set and rides on the endpoint's next Router-LSA.
        """
        u, v = pair
        adm = self._pair_admin.pop(pair, None)
        t_known = {u: tick, v: tick}
        if adm is not None and not new_state:
            adm_node, adm_tick = adm
            # ONLY a record from THIS tick.  An administrative teardown makes
            # the pair transition in the same step, unless another link of the
            # pair keeps it up -- and then the record is about a link, not
            # about this transition, and using it would date the origination
            # to a tick that has already gone by.
            if adm_node in (u, v) and adm_tick == tick:
                other = v if adm_node == u else u
                t_known[adm_node] = adm_tick
                t_known[other] = adm_tick + self.detect_ticks

        origs = sorted(o for o in (u, v) if self._is_router(o))
        # No modelled originator (or an access adjacency, which is not in the
        # link-state domain): the change takes effect where it is observed.
        routed = bool(origs) and not self._is_access(pair)
        tid = None
        if routed:
            # Every originating endpoint learns of the transition and will
            # announce it -- even if it reverts before it is installed: the
            # router saw its link change, and the LSA it floods is billed,
            # as it was when each transition scheduled its own.
            self._trans_seq += 1
            tid = self._trans_seq
            for o in origs:
                # `tick` in the max: knowledge is never dated into the past,
                # or its origination would fall on a tick already billed.
                self._unannounced.setdefault(o, []).append(
                    (max(tick, t_known.get(o, tick)), pair, tid))
        if self._pair_installed.get(pair, new_state) == new_state:
            # Reverted before it was installed: cancel any pending change
            # rather than installing a state the network already has.
            self._pending_install.pop(pair, None)
            self._awaiting.pop(pair, None)
            self._set_installed(pair, new_state)
            return
        if not routed or not self.install_enabled:
            # Not routed: effective at once.  MARL_LS_INSTALL=0 -- the
            # pre-registered evaluation-only sensitivity, idealised timing:
            # the LSAs are still originated and still billed (queued above);
            # only their effect is immediate.
            self._awaiting.pop(pair, None)
            self._pending_install[pair] = (new_state, tick)
            return
        self._pending_install.pop(pair, None)
        self._awaiting[pair] = {'id': tid, 'state': new_state,
                                'waiting': set(origs), 'arrival': None}

    def _originate(self, tick):
        """Originate this tick's Router-LSAs: ONE per router whose earliest
        pending transition is known and whose MinLSInterval has elapsed,

            t_o = max(t_known of the earliest pending transition,
                      last_orig + MIN_LS_INTERVAL),

        carrying EVERY pending transition the router knows of by now (RFC 2328
        S12.4: a Router-LSA describes all of the router's links, and
        MinLSInterval limits re-origination of that one LSA).  A transition
        the router learns later waits for its next origination.

        For each carried transition the flood arrives network-wide at
        t_o + FLOOD_HOP_TICKS * ecc_o, ecc_o measured on the INSTALLED router
        graph now, at the origination; once the LSA of its LAST originator
        has gone out, the transition installs at the latest arrival +
        SPF_DELAY_TICKS.  Called every step, after the adjacency layer, so a
        transition confirmed on this tick rides on an LSA originated on this
        tick.  Returns the number of LSAs originated."""
        n = 0
        for o in sorted(self._unannounced):
            queue = self._unannounced[o]
            if not queue:
                del self._unannounced[o]
                continue
            earliest = min(tk for tk, _p, _i in queue)
            last = self._last_orig.get(o)
            t_o = earliest if last is None else max(earliest,
                                                    last + MIN_LS_INTERVAL)
            if tick < t_o:
                continue
            carried = [(p, i) for tk, p, i in queue if tk <= tick]
            rest = [e for e in queue if e[0] > tick]
            self._last_orig[o] = tick
            self._lsa_schedule.setdefault(tick, []).append(o)
            self.originations += 1
            self.announced += len(carried)
            n += 1
            arrival = tick + FLOOD_HOP_TICKS * self._eccentricity(o)
            for p, i in carried:
                rec = self._awaiting.get(p)
                if rec is None or rec['id'] != i:
                    continue        # reverted or forgotten: announced only
                rec['waiting'].discard(o)
                rec['arrival'] = (arrival if rec['arrival'] is None
                                  else max(rec['arrival'], arrival))
                if not rec['waiting']:
                    del self._awaiting[p]
                    self._pending_install[p] = (
                        rec['state'], rec['arrival'] + SPF_DELAY_TICKS)
            if rest:
                self._unannounced[o] = rest
            else:
                del self._unannounced[o]
        return n

    def install_tick(self, pair):
        """The tick at which the pending transition of `pair` installs, or
        None -- no transition pending, or one whose carrying LSAs have not
        all been originated yet (`awaiting_announcement`)."""
        rec = self._pending_install.get(tuple(sorted(pair)))
        return None if rec is None else rec[1]

    def awaiting_announcement(self, pair):
        """True while a confirmed transition of `pair` waits for the Router-LSA
        of one of its originating endpoints."""
        return tuple(sorted(pair)) in self._awaiting

    def _invalidate_caches(self):
        """Drop what the INSTALLED infrastructure graph is derived from."""
        self._adj_installed = None
        self._ecc_cache = {}

    def _set_installed(self, pair, state):
        self._pair_installed[pair] = state
        self._invalidate_caches()

    def _apply_matured_installs(self, tick):
        for pair, (state, t_inst) in sorted(self._pending_install.items()):
            if tick >= t_inst:
                self._set_installed(pair, state)
                self._pending_install.pop(pair, None)
                self.installs += 1

    # ── derived view ──────────────────────────────────────────────

    def _refresh_view(self, tick, links, tombs):
        """Recompute `visible`, the believed-graph changes, the stale and
        tombstoned id sets and `installed_since`; purge installed tombstones."""
        topo = self.sim.topology
        pair_lids = self._pairs_and_links(links, tombs)
        # ONE O(L) INDEX PER STEP, not one O(L) scan per pair.
        # `Topology.pair_representative` rescans every Link in the topology on
        # every call, which is fine for the handful of calls a teardown makes
        # but not for the two calls per pair per tick this needs: at ~150
        # links and ~150 pairs that is ~45,000 comparisons a tick, on the hot
        # path of every MARL-arm pass.  `_links_by_pair` is the same index in
        # one pass and `_representative_of` is the SAME RULE (WP1: relay
        # first, then lowest id) applied to it, so belief and truth still name
        # the hop identically -- the WP1 test asserts this file uses the rule.
        pair_objs = topo._links_by_pair()
        rep_of = topo._representative_of
        visible, changes, stale, tomb_used = {}, [], set(), set()

        for pair, lids in sorted(pair_lids.items()):
            u, v = pair
            objs = pair_objs.get(pair, ())
            inst_up = self._pair_installed.get(pair)
            if inst_up is None:
                inst_up = any(bool(getattr(l, 'is_up', True)) for l in objs)
            true_rep = rep_of(objs, topo._link_is_up)
            rep_id, rep_cap = None, 0.0
            if inst_up:
                # Belief substitutes its own notion of "up" in the shared rule.
                rep = rep_of(objs, self._believed_up)
                if rep is not None:
                    # Is any link the service BELIEVES up also physically up?
                    # That, and not the state of the believed representative
                    # alone, is what decides whether the hop can carry -- see
                    # THE GUARD below (D3.22, narrowed by D3.31).
                    live = rep_of(objs, self._believed_and_physically_up)
                    if live is not None and true_rep is not None:
                        # BELIEF DECIDES WHETHER THE HOP EXISTS; PHYSICS
                        # DECIDES WHICH LINK CARRIES IT -- BUT ONLY WHILE SOME
                        # BELIEVED-UP LINK OF THE PAIR IS PHYSICALLY UP.
                        # D3.15, guarded by D3.22/D3.31.
                        #
                        # Without the substitution the believed representative
                        # could be a DIFFERENT Link from the true one: a
                        # Priority-2 relay repurposes a pair that already has
                        # a live physical link, so for its whole BFD bring-up
                        # the relay is the true representative while belief
                        # still names the underlying link -- and `consume_path`
                        # reads `link.capacity` off the Link the hop names, so
                        # the arm forwarded at the PRE-RELAY capacity over a
                        # hop that physically carries only the relay share.
                        # That is capacity the hop does not have, handed to the
                        # arms under study, and it is the same class of error
                        # as routing over a physically-down link.
                        #
                        # Belief still decides EXISTENCE -- the pair is a hop
                        # only because the service believes and has installed
                        # it, so no connectivity is adopted for free -- but
                        # once it is a hop, the Link that carries it is the
                        # one physics names, which is also the Link the
                        # engine's own true graph names.
                        #
                        # THE GUARD (`live is not None`, D3.22 as narrowed by
                        # D3.31).  The substitution may not RESURRECT a hop
                        # the service's own belief cannot carry.  When NOTHING
                        # the service believes up is physically up, the hop
                        # must blackhole (the branch below) -- and substituting
                        # an unrelated live Link of the same pair cancelled
                        # exactly that: a relay created on the pair during the
                        # detection window was adopted with `_link_belief`
                        # False, `visible` False and `installed_since` None,
                        # delivering at full capacity over a link no BFD
                        # session had confirmed and no Router-LSA had
                        # installed.  That is the free adoption D3.16 closed on
                        # the down-representative side and RERUN_SPEC WP3
                        # forbids twice ("a new link's adjacency comes up after
                        # the BFD session, 3 ticks"; "after island onset a
                        # never-seen link id starts DOWN").
                        #
                        # THE PREDICATE IS THE PAIR, NOT THE REPRESENTATIVE
                        # (D3.31).  D3.22 first tested `topo._link_is_up(rep)`,
                        # i.e. the relay-preferred BELIEVED representative
                        # alone.  A pair carrying two links the service
                        # believes up, whose representative is the one that has
                        # physically failed, then blackholed for the whole
                        # detection window although the service believed the
                        # PARALLEL link up and that link was physically
                        # carrying the pair in the true graph -- a hop refused
                        # on evidence the service does not have.  Substituting
                        # whenever SOME believed-up link is physically up
                        # closes D3.22's hole identically (no believed-up link
                        # physically up => blackhole) while keeping the correct
                        # carrier.  Latent on the evaluated topology (relay
                        # links go down administratively, where belief drops at
                        # once, or with their node, which downs both links of
                        # the pair; the static topology builds one link per
                        # infrastructure pair), so it moves no series -- pinned
                        # by `test_a_parallel_believed_up_link_still_carries`.
                        #
                        # CAPACITY.  In the D3.15 case the substitution is
                        # MEASURED never to raise what the hop offers
                        # (`MORE_CAP = 0` over 712,661 hop-instances on these
                        # seeds, D3.28) -- but that is a property of these
                        # topologies, NOT of the ordering rule, which is
                        # relay-first then LOWEST ID with no capacity term
                        # (`Topology._representative_of`).  Two relays on one
                        # pair, or a relay whose share exceeds a degraded
                        # microwave partner, would let the true representative
                        # carry more.  D3.32.  The substitution is justified by
                        # naming the Link that PHYSICALLY carries the hop, not
                        # by a capacity inequality; the D3.31 case above can
                        # legitimately raise the offered capacity above the
                        # failed believed representative's, because the
                        # alternative there is a blackhole.
                        rep = true_rep
                    rep_id, rep_cap = rep.id, rep.capacity
                    # STAMP 1 OF 2: every link of an installed hop that the
                    # service BELIEVES up.  This is the belief-side half of
                    # `visible`, and it may NOT consult physical truth: a link
                    # the service believes up inside an installed hop is in its
                    # forwarding view whether or not it has secretly failed,
                    # because not knowing that is the whole point of the
                    # detection floor.  So the believed carrier of a hop that
                    # blackholes during an UNDETECTED failure keeps its stamp
                    # for the detection window (D3.30; the `stale` branch below
                    # is what makes it deliver 0.0).  Only a hop with NOTHING
                    # believed up -- the tombstone / `_down_representative`
                    # case -- is genuinely unstamped.
                    for lid in lids:
                        if lid in links and self._link_belief.get(lid, False):
                            visible[lid] = True
                    if not bool(getattr(rep, 'is_up', True)):
                        # THE UNDETECTED FAILURE.  The service still believes
                        # this link up, so it is still the hop -- and it has
                        # physically failed, so it delivers 0.0.  Naming it
                        # here is what stops the alternate-path retry rerouting
                        # around a failure the control plane has not seen:
                        # that retry exists for SATURATION, and using it here
                        # would hand back exactly the omniscience the
                        # detection floor removes.
                        stale.add(rep.id)
                    elif rep_id in links:
                        # STAMP 2 OF 2: THE STAMP FOLLOWS THE CARRIER.  D3.23.
                        #
                        # `visible` and `installed_since` answer "is this link
                        # in my forwarding view?", and the view NAMES this Link
                        # as the one carrying the hop: `consume_path` resolves
                        # `rep_id`, reads its capacity and accumulates its
                        # utilisation.  Stamping only the links in
                        # `_link_belief` left the substituted carrier -- a
                        # Priority-2 relay for its whole bring-up -- carrying
                        # the hop with `visible` False and `installed_since`
                        # None, so the service's own API contradicted the view
                        # it publishes.  WP4 counts a bridge's teardown grace
                        # in "forwarding passes in which the relay link is
                        # installed in the arm's forwarding view (written by
                        # the service via `installed_since`)" (RERUN_SPEC WP4),
                        # and would have read 0 usable passes for a relay that
                        # had carried the hop since its creation tick, blocked
                        # its teardown as "not-installed" and booked it as an
                        # idle-bridge teardown.  This second stamp is reached
                        # only when the named carrier is physically up, i.e.
                        # when the hop really carries; a hop that BLACKHOLES
                        # adds no carrier stamp here, but keeps whatever stamp
                        # the belief-side half above gave its believed-up links
                        # (D3.30).
                        visible[rep_id] = True
                else:
                    # INSTALLED UP, BUT NOTHING OF THE PAIR IS BELIEVED UP.
                    # The hop must stay in the graph (its removal is not
                    # installed yet) but it must not carry a bit: name the
                    # physically-down Link, or the TOMBSTONE of a Link that has
                    # already been popped, so consume_path resolves the hop and
                    # returns exactly 0.0.
                    rep_id, rep_cap, is_tomb = self._down_representative(
                        objs, lids, links, tombs)
                    if rep_id is not None:
                        (tomb_used if is_tomb else stale).add(rep_id)
            true_id = None if true_rep is None else true_rep.id
            if rep_id != true_id:
                changes.append((u, v, rep_id, rep_cap))

        for lid in self._link_belief:
            visible.setdefault(lid, False)
        # installed_since: the tick a link entered the installed view.  The
        # engine mirror `Simulator._relay_route_since` is written and cleared
        # ALONGSIDE it, in both directions (D3.33): the two copies of one stamp
        # may not disagree, or WP4 -- which reads the engine copy -- sees a
        # link that left the forwarding view long ago as installed since the
        # tick it first entered, forever.
        for lid, up in visible.items():
            if up:
                if lid not in self._installed_since:
                    self._installed_since[lid] = tick
                    self._stamp_engine_mirror(lid, tick)
            else:
                if self._installed_since.pop(lid, None) is not None:
                    self._unstamp_engine_mirror(lid)
        for lid in list(self._installed_since):
            if lid not in visible:
                self._installed_since.pop(lid, None)
                self._unstamp_engine_mirror(lid)

        self.visible = visible
        self._view_changes = changes
        self._stale_ids = frozenset(stale)
        self._tomb_ids = frozenset(tomb_used)
        self._purge_tombstones(tombs, tomb_used)

    def _purge_tombstones(self, tombs, still_used):
        """A tombstone lives until its removal is INSTALLED, and until the
        removal has been DETECTED -- dropping it earlier would cancel the BFD
        timeout that a non-administrative teardown still has to pay, which is
        one of the two bypasses this work package exists to close.

        So it is dropped only when all three hold: the installed view does not
        name it as a hop, the service no longer believes it up, and no
        detection is pending on it.  At that point it contributes to nothing,
        and the engine table (and the service's belief about a link that no
        longer exists) can forget it."""
        for rid in list(tombs):
            if (rid in still_used
                    or self._link_belief.get(rid, False)
                    or rid in self._pending_link):
                continue
            tombs.pop(rid, None)
            self._forget_link(rid)

    def _down_representative(self, objs, lids, links, tombs):
        """(link_id, capacity, is_tombstone) for a hop that is installed up
        but has nothing believed up behind it: the physically-down Link of the
        pair if one is still there, else the TOMBSTONE of one that has been
        popped.  Either way `consume_path` resolves the hop and returns 0.0.

        THE PREDICATE MUST EXCLUDE PHYSICALLY-UP LINKS.  It used to accept any
        Link of the pair, and a pair can hold a physically-up one here: a
        relay re-created on a pair whose previous removal has not been
        un-installed yet is physically up but not believed up and not
        installed.  Naming it made `consume_path` resolve it, see
        `is_up == True` and forward at its full capacity -- free adoption of a
        brand-new link id with belief False, `visible` False and
        `installed_since` None, which is exactly the bypass this work package
        exists to close.  It also put physically-UP ids into
        `stale_link_ids()`, whose contract (tick_step and consume_path) is
        "in the view although the Link is physically DOWN".  D3.16."""
        rep = self.sim.topology._representative_of(
            objs, lambda lk: not self.sim.topology._link_is_up(lk))
        if rep is not None:
            return rep.id, rep.capacity, False
        cands = sorted(rid for rid in lids if rid in tombs and rid not in links)
        if cands:
            rid = cands[0]
            return rid, float(tombs[rid].get('capacity', 0.0) or 0.0), True
        return None, 0.0, False

    def _update_isolation(self, tick):
        """RERUN_SPEC WP6 / T1: site i is isolated iff its component in the
        service's infrastructure graph contains no core node.  Computed on the
        INSTALLED view -- the router's own forwarding view after SPF, which is
        what "the site knows" means operationally and what everything else
        this service feeds already uses (D3.4).  It runs here, inside step, so
        it is idempotent within a tick and observations read the state of the
        PREVIOUS step."""
        parent = {}
        find = _make_find(parent)
        for pair, up in self._pair_installed.items():
            if not up or self._is_access(pair):
                continue
            ra, rb = find(pair[0]), find(pair[1])
            if ra != rb:
                parent[ra] = rb
        core_roots = {find(c) for c in getattr(self.sim, 'core_nodes', ())}
        for site in self.sim.agents:
            iso = find(site) not in core_roots
            was = self.isolated.get(site)
            if was is None or was != iso:
                self.isolated[site] = iso
                self.isolated_since[site] = tick if iso else None

    def _recompute_routable_flows(self):
        """Connectivity in the INSTALLED view.  UEs are included because the
        flows are UE-to-UE, and access adjacencies follow truth."""
        parent = {}
        find = _make_find(parent)
        for pair, up in self._pair_installed.items():
            if not up:
                continue
            ra, rb = find(pair[0]), find(pair[1])
            if ra != rb:
                parent[ra] = rb
        self.routable_flows = {(s, t) for s, t in self.sim.ue_to_ue_flows
                               if find(s) == find(t)}

    # ══════════════════════════════════════════════════════════════
    # public API
    # ══════════════════════════════════════════════════════════════

    def get_routable_flows(self):
        """The flows this arm believes are routable, in its installed view.

        sorted(), NOT list(): the engine allocates capacity GREEDILY in the
        order this list is walked, and set iteration order over strings is
        salted per process (PYTHONHASHSEED)."""
        return sorted(self.routable_flows)

    def believed_graph(self):
        """The routing graph as the INSTALLED view holds it, or None when the
        installed view equals truth (the overwhelming majority of ticks, so
        the graph copy is paid for only while something is actually pending).

        Safety: a re-installed edge CARRIES ITS link_id, so `consume_path`
        resolves the Link, sees `is_up == False` -- or finds no Link at all,
        for a tombstone -- and returns 0.0.  An undetected-but-down link is
        therefore selectable but NOT deliverable, a blackhole exactly as in
        the field.  Dropping the link_id would instead make the edge look like
        infinite capacity and let dead links carry traffic, which is the one
        outcome this must never produce."""
        if not self._view_changes:
            return None
        G = self.sim.topology.graph.copy()
        for u, v, lid, cap in self._view_changes:
            if lid is None:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
            else:
                G.add_edge(u, v, link_id=lid, capacity=cap)
        return G

    def stale_link_ids(self):
        """Ids the installed view holds as a hop although the Link object is
        physically DOWN.  `consume_path` flags a path over one of them as
        blackholed, so the alternate-path retry -- which exists to work around
        SATURATION -- cannot silently reroute around a failure the control
        plane has not installed yet."""
        return self._stale_ids

    def tombstoned_link_ids(self):
        """Ids the installed view holds as a hop although the Link object has
        already been POPPED.  Physically down, delivers exactly 0.0, and
        EXPECTED -- as opposed to an unresolved hop on the TRUE graph, which
        is the WP1 defect and must stay impossible."""
        return self._tomb_ids

    def installed_since(self, link_id):
        """The tick at which `link_id` entered the installed forwarding view,
        or None when it is not in it.  WP4 counts a bridge's teardown grace in
        ticks the link was usable, not ticks since its beam came up.

        "In the forwarding view" means either of the two ways a Link can be in
        it, because both are ways the arm forwards over it (D3.23): the service
        believes it up inside an installed hop, or the view NAMES it as the
        Link carrying an installed hop (the substituted carrier of D3.15 --
        a Priority-2 relay carries its pair from its creation tick, before its
        own BFD session completes).

        A hop that BLACKHOLES is stamped iff the first of those holds, which
        is exactly the undetected-failure case (D3.30): the service still
        believes its carrier up, so that link is still in ITS forwarding view
        while the flow delivers 0.0, and the stamp clears when BFD confirms
        the failure.  The stamp may not consult physical truth the service has
        not detected -- doing so would be the omniscience the detection floor
        removes.  A hop with NOTHING believed up (a tombstone or the
        physically-down `_down_representative`) is genuinely unstamped.
        WP4 therefore counts a usable pass for those <= BFD_DETECT_TICKS
        ticks: `_relay_link_passes` means "installed in the arm's forwarding
        view", as RERUN_SPEC WP4 words it, NOT "delivered bits".  A bridge
        that fails undetected is booked with `passes > 0` and so is not an
        idle-bridge teardown; a "carried bits" count, if WP4 wants one, is a
        separate counter and not this stamp."""
        return self._installed_since.get(link_id)

    def believed_neighbours(self, site_id):
        """The infrastructure neighbours `site_id` believes it has a live
        adjacency with, as of the last completed step (read from the
        observation builder: the PREVIOUS step): the ADJACENCY layer
        (`_pair_confirmed`, up iff some link of the pair is BFD-believed up)
        for the pairs incident to the site, access (UE) adjacencies
        excluded.  This is the site's own knowledge of its links -- it pays
        the BFD detection delay but not the flood/SPF installation delay.
        Sorted, so a caller that sums over it does so in a fixed order.
        WP6's `c_live_infra_neighbours` reads this (trackAgent merge)."""
        out = []
        for pair, up in self._pair_confirmed.items():
            if not up or site_id not in pair or self._is_access(pair):
                continue
            out.append(pair[1] if pair[0] == site_id else pair[0])
        return sorted(out)

    def isolation_state(self, site_id):
        """(is_isolated, isolated_since_tick) for one site, as of the last
        completed step -- which, read from the observation builder, is the
        PREVIOUS step.  `isolated_since_tick` is None when the site is not
        isolated.  This is the name WP6's adapter binds to."""
        return (self.isolated.get(site_id, False),
                self.isolated_since.get(site_id))

    def tick_messages(self, tick):
        """The control-plane messages this service generates on `tick`.

        A breakdown dict: msgs, bytes, <type>_msgs and <type>_bytes for each
        of MESSAGE_TYPES, and cpu_mj_per_msg (the weighted mean the harness's
        control-plane energy term needs).  It is computed inside step() and
        cached, so calling this is free and cannot double-count; a tick that
        was never stepped bills nothing.

        DESCRIPTIVE ONLY.  See the module docstring: these bytes describe what
        this modelled service would send.  They are not a comparative KPI and
        must not be used to rank arms across protocol families."""
        if self._last_tick is None or int(tick) != self._last_tick:
            return zero_messages()
        return dict(self.last_messages)

    # ══════════════════════════════════════════════════════════════
    # billing
    # ══════════════════════════════════════════════════════════════

    def _bill(self, tick):
        out = zero_messages()
        # FROM ISLAND ONSET ONLY.  Before the event the core handles routing
        # and no distributed protocol runs on the mesh -- the identical rule
        # ControlPlaneMessageSim.compute_tick_messages applies to every
        # baseline, so pre-event billing is exactly zero for EVERY arm.
        if self.island_onset_tick is None or tick < self.island_onset_tick:
            self._lsa_schedule.pop(tick, None)
            return out

        infra_nbrs = self._infra_neighbours()

        # ── BFD: 1 packet/s per direction on every INFRASTRUCTURE adjacency
        #    pair that is RUNNING A SESSION -- one confirmed up (periodic
        #    keepalives), or one with a session in progress on one of its
        #    links (`_pending_link`: a bring-up, or a failure still inside its
        #    detect multiplier).  Counting PAIRS, not Link objects, means a
        #    Priority-2 relay that repurposes an existing hop adds no session
        #    while a sole-pair relay bridge does.
        #
        #    The in-progress half is D3.24.  Billing only CONFIRMED-up pairs
        #    charged a session nothing for its 3-tick bring-up although BFD
        #    sends 1 packet/s per direction from the moment the session starts
        #    -- which is precisely what makes the bring-up cost what it costs.
        #    Every new relay bridge escaped 3 x 2 x 66 B = 396 B of the bill it
        #    is being TIMED by.  (A failure inside its detect window needs no
        #    special case: its pair is still confirmed up, so it was, and is,
        #    billed once.)
        #
        #    THIS DEPARTS FROM THE BINDING SPECIFICATION.  RERUN_SPEC WP3 and
        #    rerun_map.json / rerun_details.txt item 4b both say "per
        #    BELIEVED-UP infrastructure adjacency pair per tick"; this bills
        #    the union of those pairs with the pairs running a session.  The
        #    divergence, its reason and its consequence for the
        #    pre-registration are recorded in docs/DECISIONS.md D3.24 (and
        #    D3.34) and in docs/LINK_STATE_SERVICE.md section 5.  It moves
        #    `overhead` (descriptive) AND `energy` (comparative) for every
        #    MARL-path arm.
        billed_pairs = {pair for pair, up in self._pair_confirmed.items()
                        if up and not self._is_access(pair)}
        for lid in self._pending_link:
            pair = self._link_pair(lid)
            if pair is not None and not self._is_access(pair):
                billed_pairs.add(pair)
        n_pairs = len(billed_pairs)
        out['bfd_msgs'] = n_pairs * BFD_PACKETS_PER_PAIR_PER_TICK
        out['bfd_bytes'] = out['bfd_msgs'] * BFD_BYTES

        # ── Hellos: exactly _sim_ospf's rule.  ONE L2 broadcast to
        #    AllSPFRouters per router per HelloInterval at its crc32 offset,
        #    44 B + 4 B per listed neighbour.  Infra-only neighbours, which is
        #    the one place this service does NOT reproduce the OSPF baseline:
        #    that branch counts UE peers too and so overstates its own hello
        #    size.  It is a known baseline defect, reported descriptively and
        #    left alone -- WP3 does not touch the baselines.
        for nid in sorted(self._routers()):
            if (tick + node_offset(nid, HELLO_INTERVAL)) % HELLO_INTERVAL == 0:
                out['hello_msgs'] += 1
                out['hello_bytes'] += (HELLO_BASE
                                       + HELLO_PER_NBR * infra_nbrs.get(nid, 0))

        # ── Router-LSAs originated on this tick.  They were scheduled at t_o
        #    by the installation model, so the MinLSInterval throttle is
        #    already in the schedule.  One message per DIRECTED believed-up
        #    infra adjacency in the originator's believed component; each
        #    message 24 + 12 x (originator infra degree) bytes.  No LSAck and
        #    no periodic refresh -- as the baseline model, stated not hidden.
        for o in sorted(self._lsa_schedule.pop(tick, [])):
            flood = self._component_directed_adjacencies(o)
            if flood <= 0:
                continue
            size = LSA_BASE + LSA_PER_LINK * infra_nbrs.get(o, 0)
            out['lsa_msgs'] += flood
            out['lsa_bytes'] += flood * size

        # ── Postcards (RERUN_SPEC WP5): the DisasterControlChannel ledger of
        #    THIS tick -- one FRAME per live infrastructure link of each
        #    sending site, each frame postcards.POSTCARD_ENCODED_BYTES (60 B)
        #    on the wire.  A message here is a frame, as for the other types.
        #    The heartbeat is sent in the action step, which step_tick runs
        #    before this service step, so the ledger is complete.
        pc = postcard_ledger(self.sim, tick)
        out['postcard_msgs'] = pc['frames']
        out['postcard_bytes'] = pc['bytes']

        out['msgs'] = sum(out[t + '_msgs'] for t in MESSAGE_TYPES)
        out['bytes'] = sum(out[t + '_bytes'] for t in MESSAGE_TYPES)
        if out['msgs'] > 0:
            out['cpu_mj_per_msg'] = sum(
                _CPU_MJ[t] * out[t + '_msgs']
                for t in MESSAGE_TYPES) / out['msgs']
        return out

    # ══════════════════════════════════════════════════════════════
    # helpers
    # ══════════════════════════════════════════════════════════════

    def _believed_up(self, link):
        return bool(self._link_belief.get(
            link.id, bool(getattr(link, 'is_up', True))))

    def _believed_and_physically_up(self, link):
        """Believed up AND physically up (D3.31).

        The predicate that decides whether an installed hop can CARRY: the
        service may refuse a hop only when nothing it believes up is
        physically up, never because the relay-preferred representative of a
        pair happens to be the failed one.  It is deliberately NOT used to
        pick the carrier -- once the hop carries, physics names the Link
        (D3.15), which in a Priority-2 relay's bring-up is a link this
        predicate rejects."""
        return (self._believed_up(link)
                and bool(getattr(link, 'is_up', True)))

    # ── the engine's copy of the installation stamp ───────────────

    def _stamp_engine_mirror(self, lid, tick):
        """Mirror `installed_since` into `Simulator._relay_route_since`.

        WP4 reads this stamp so `_relay_teardown_blocked` counts a bridge's
        grace in ticks the link was actually usable instead of ticks since the
        beam came up.  It is a RELAY-SCOPED dict -- `_remove_relay_link` is
        the only other writer and it only ever pops -- so only ids the engine
        itself treats as relay links are written (D3.33).  Filling it with
        mesh and Uu access ids left entries nothing would ever clear, and WP4
        blocks teardown of "a protected link that is physically up but not yet
        installed" by reading exactly this dict."""
        _rs = getattr(self.sim, '_relay_route_since', None)
        if isinstance(_rs, dict) and is_transport_relay(
                self.sim.topology.links.get(lid)):
            _rs[lid] = tick

    def _unstamp_engine_mirror(self, lid):
        """Drop the engine copy when the link leaves the installed view.

        Unconditional, unlike the write: a link can leave `topology.links`
        (and so stop answering `is_transport_relay`) in the same tick as it
        leaves the view, and this dict has no other writer, so an id in it
        that the service is not stamping is stale by construction."""
        _rs = getattr(self.sim, '_relay_route_since', None)
        if isinstance(_rs, dict):
            _rs.pop(lid, None)

    def _is_access(self, endpoints):
        """True for an adjacency with a UE endpoint (see the class docstring:
        access adjacencies are not in the link-state domain)."""
        is_ue = self.sim.topology.is_ue_node
        return any(is_ue(e) for e in endpoints)

    def _is_router(self, nid):
        """A surviving node that runs the routing protocol on the mesh: the
        same set ControlPlaneMessageSim._get_survivor_nodes bills hellos
        for."""
        node = self.sim.topology.nodes.get(nid)
        return bool(node is not None and node.is_survivor
                    and node.node_type.value not in NON_ROUTER_TYPES)

    def _routers(self):
        return [nid for nid, n in self.sim.topology.nodes.items()
                if n.is_survivor and n.node_type.value not in NON_ROUTER_TYPES]

    @staticmethod
    def _pair_of(endpoints):
        eps = tuple(endpoints or ())
        if len(eps) != 2:
            return None
        return tuple(sorted(eps))

    def _link_pair(self, lid):
        """The endpoint pair of `lid`, whether the Link is still in
        `topology.links` or only a tombstone, else None.  The BFD bill needs
        it for the links carrying a session that no pair state names yet."""
        link = self.sim.topology.links.get(lid)
        if link is not None:
            return self._pair_of(getattr(link, 'endpoints', ()))
        tombs = getattr(self.sim, '_link_tombstones', None) or {}
        tb = tombs.get(lid)
        if tb is None:
            return None
        return self._pair_of(tb.get('endpoints', ()))

    def _pairs_and_links(self, links, tombs):
        """{pair: [link_id, ...]} over live links AND the tombstones the
        service still believes in (one O(L) pass, insertion-order stable)."""
        out = {}
        for lid, link in links.items():
            pair = self._pair_of(getattr(link, 'endpoints', ()))
            if pair is not None:
                out.setdefault(pair, []).append(lid)
        for rid, tb in tombs.items():
            if rid not in self._link_belief:
                continue
            pair = self._pair_of(tb.get('endpoints', ()))
            if pair is None:
                continue
            bucket = out.setdefault(pair, [])
            if rid not in bucket:
                bucket.append(rid)
        return out

    @staticmethod
    def _true_pair_up(lids, links):
        return any(bool(getattr(links[l], 'is_up', True))
                   for l in lids if l in links)

    @staticmethod
    def _adjacency(pair_state, skip):
        adj = {}
        for pair, up in pair_state.items():
            if not up or skip(pair):
                continue
            adj.setdefault(pair[0], set()).add(pair[1])
            adj.setdefault(pair[1], set()).add(pair[0])
        return adj

    def _infra_adjacency(self):
        """{node: {neighbour}} over BELIEVED-UP infrastructure adjacencies.
        Cached for the tick: the billing walks it once per originator."""
        if self._adj_confirmed is None:
            self._adj_confirmed = self._adjacency(self._pair_confirmed,
                                                  self._is_access)
        return self._adj_confirmed

    def _not_a_router_pair(self, pair):
        """Skip rule for the FLOOD graph: an adjacency relays a Router-LSA
        only when BOTH ends run the routing protocol on the mesh."""
        return not all(self._is_router(e) for e in pair)

    def _installed_adjacency(self):
        """{node: {neighbour}} over INSTALLED adjacencies BETWEEN ROUTERS --
        the graph a Router-LSA is flooded across, and so the graph the
        installation formula's `ecc_o` is measured on.  Cached for the tick:
        the installation formula walks it once per originator.

        ROUTERS, not merely non-access.  `_is_router` (the same set
        ControlPlaneMessageSim._get_survivor_nodes bills hellos for, i.e.
        NON_ROUTER_TYPES excluded) decides who originates a Router-LSA and who
        is billed a hello; a Core/UPF/AMF node runs no routing protocol on the
        mesh, so it cannot relay a flood either.  Measuring `ecc_o` on a graph
        that still routed through such a node made the spec's "BFS
        eccentricity of the originator in the installed infrastructure graph"
        count hops no LSA can take -- e.g. an extra hop through the AMF
        hanging off one end of a path -- and so dated `t_inst` too late
        whenever a core node was still installed.  D3.17.

        The BILLED flood scope (`_component_directed_adjacencies`, over
        `_infra_adjacency`) is deliberately left on the broader non-access
        graph: it reproduces the OSPF baseline's own counting, which this
        service copies rather than corrects, and D3.17 records the
        asymmetry."""
        if self._adj_installed is None:
            self._adj_installed = self._adjacency(self._pair_installed,
                                                  self._not_a_router_pair)
        return self._adj_installed

    def _infra_neighbours(self):
        return {n: len(s) for n, s in self._infra_adjacency().items()}

    def _component_directed_adjacencies(self, origin):
        """Directed believed-up infra adjacencies in `origin`'s believed
        component: the flood scope of one Router-LSA (RFC 2328 S13.3 sends
        each LSA once per link per direction)."""
        adj = self._infra_adjacency()
        if origin not in adj:
            return 0
        seen, q, directed = {origin}, deque([origin]), 0
        while q:
            n = q.popleft()
            for m in adj.get(n, ()):
                directed += 1        # each pair is walked once from each end
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        return directed

    def _eccentricity(self, origin):
        """BFS eccentricity of `origin` in the INSTALLED infrastructure
        ROUTER graph (the installation formula's ecc_o -- see
        `_installed_adjacency`); 0 when it has no installed router neighbour.
        Cached for the tick, because several transitions in one tick share the
        graph."""
        if origin in self._ecc_cache:
            return self._ecc_cache[origin]
        adj = self._installed_adjacency()
        dist, q, far = {origin: 0}, deque([origin]), 0
        while q:
            n = q.popleft()
            for m in adj.get(n, ()):
                if m not in dist:
                    dist[m] = dist[n] + 1
                    far = max(far, dist[m])
                    q.append(m)
        self._ecc_cache[origin] = far
        return far


def _make_find(parent):
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    return find


def make_link_state_service(sim):
    """The forwarding view for an arm on the MARL code path: a
    LinkStateService, or None when MARL_DETECT_TICKS=0 (the omniscient
    ablation: true graph, no flow gate, no billing).  Both drivers create it
    through this function before the first tick; `Simulator.step_tick` steps
    it.

    It is attached to the simulator as `link_state_service` -- the name WP6's
    adapter binds to -- and, for the harness's own latency/outage block, as
    `_marl_detect`."""
    if MARL_BFD_DETECT_TICKS <= 0:
        return None
    svc = LinkStateService(sim)
    sim.link_state_service = svc
    sim._marl_detect = svc
    return svc
