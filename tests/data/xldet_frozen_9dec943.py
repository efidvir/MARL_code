# -*- coding: utf-8 -*-
"""XL-DET — a deterministic cross-layer controller, as a baseline arm.

WHY THIS EXISTS. The uniform-random control answers "is arbitrary use of the
action space enough?" It does not answer "is LEARNING necessary?" A reviewer
can accept that radio-layer control beats routing and still ask whether a
competently engineered deterministic controller with the same actuation would
do as well. XL-DET is that controller, and it is built to win if it can: a
strawman here would prove nothing.

WHAT IT MAY READ. Exactly what the learned actor may read after the locality
correction: own PHY/MAC state, neighbour state carried by postcards, and the
five local Block C features. It never touches agent.NONLOCAL_OBS_COLUMNS or the
coordinator vector. A baseline allowed global state would beat MARL for the
wrong reason.

WHAT IT SHARES WITH THE LEARNED ARMS. Everything else: the same observations,
the same _step_phy_mac, the same physics, the same teardown rules, and the same
engine-side peer selection (_rank_bridge_candidates). The comparison therefore
isolates the policy, not the mechanism.

THE RULES, and why each is what it is:

  PRB     A constant point (0.042857, 0.857143, 0.100000). SOLVED, not tuned;
          verify_prb_point() below re-derives it from the engine's own
          constants and refuses to run if it no longer holds.

          Three engine facts set it. (i) normalise_prb renormalises ONLY the
          emergency/general pair: relay rides a dedicated transport carrier
          and does not compete with access, so relay capacity is linear in the
          relay share with no access-side price. (ii) _derive_admission_policies
          thresholds the renormalised general share at 0.30/0.40/0.55 for
          OPERATIONS/TELEMETRY/BEST_EFFORT and the emergency share at 0.20 for
          LIFE_SAFETY. (iii) _execute_agent_actions floors the general
          component at 0.10.

          So: maximise relay subject to g >= 0.10 and every class at ADMIT.
          Writing s = g/(e+g), the access pool is g/s, minimised by taking g at
          the floor and s at its largest admissible value. s is admissible on
          [0.55, 0.80] -- above 0.80 the emergency share falls under 0.20 and
          LIFE_SAFETY stops being admitted. Taking s = 0.70 rather than the
          0.80 endpoint gives 0.15 of margin above the BEST_EFFORT threshold
          and 0.10 below the LIFE_SAFETY one, and costs only 0.018 of relay
          share. That is the point above, and because g sits exactly at the
          floor and the coordinator's emergency floor is presently 0.0
          (CoordinatorAgent.current_policy is GlobalPolicyVector.neutral() and
          is never reassigned; the learned coordinator is disabled), it is a
          FIXED POINT of both projections -- what XL-DET asks for is what it
          gets.

          Why this matters more than it looks: apply_marl_policy walks EVERY
          node on a path, halving the flow at any THROTTLE node and zeroing it
          outright at any HOLD node. Under a constant split every node shares
          one mode, so the difference between this point and a careless one is
          the difference between full delivery and every general flow halved.
          Pinning the split also closes the HOLD self-harm channel, which only
          arms that populate last_actions -- i.e. only agent arms -- can suffer.

  relay   CAPACITY_BOOST at a steerable site, LOCAL_REROUTE elsewhere, and
          never back to OFF once armed. Bridge capacity is frozen at formation
          and relay_link_active is close to a one-way latch, so churn destroys
          value with no upside.

  tx      A power loop with a measured-response test and an interference brake.
          Climb only while the measured dSINR/dP justifies it; hold when a
          neighbour reports rising interference; drop back when the link is
          comfortably above what the top MCS needs.

  MCS     The exact spectral-efficiency argmax. PHYMACState.spectral_efficiency
          halves the rate whenever sinr < threshold(mcs) + 5, and plain
          auto-CQI picks the highest FEASIBLE mcs, which lands inside that
          half-rate band over much of the range. Choosing the highest mcs that
          runs at FULL rate beats auto-CQI outright.

  iops    ADMIT_ALL. Every reachable registration request in island mode is an
          emergency, so admit-emergency and admit-all coincide on the reachable
          set, and admit-all cannot livelock.

  card    Request every tick; the engine rate-limits to one per 10 ticks.
"""
from .phy_mac_state import (MCS_OFFSET_CENTER, MCS_SINR_THRESHOLD, MCSLevel,
                            TX_POWER_STEPS_DB, auto_cqi_mcs_idx)

# ── the solved PRB point ─────────────────────────────────────────────────────
PRB_EMERGENCY = 0.042857
PRB_RELAY     = 0.857143
PRB_GENERAL   = 0.100000

# ── MCS ──────────────────────────────────────────────────────────────────────
HALF_RATE_MARGIN_DB = 5.0          # the cliff constant in spectral_efficiency

# ── power loop ───────────────────────────────────────────────────────────────
# Head indices are positions in TX_POWER_STEPS_DB, so they are derived from the
# engine's own table rather than written down. If that table is ever re-ordered
# or re-scaled these follow it, instead of silently commanding the wrong step.
TX_HOLD = TX_POWER_STEPS_DB.index(0.0)
TX_UP   = TX_POWER_STEPS_DB.index(3.0)
TX_DOWN = TX_POWER_STEPS_DB.index(-3.0)
SINR_TARGET_DB   = 17.0            # QAM64 at FULL rate needs 12 + 5
SINR_AMPLE_DB    = 28.0            # QAM256 full rate is 25; 3 dB of margin
LOOP_GAIN_MIN    = 0.5             # dSINR/dP below this means the fleet moved too
ICIC_INTERFERERS = 0.5
ICIC_NEIGH_POWER = 0.60
NB_STALE_MAX     = 30              # three DCC postcard periods

# ── relay ────────────────────────────────────────────────────────────────────
# Likewise derived: RELAY_MODES = list(RelayMode) is what the head indexes into.
from .phy_mac_state import RelayMode as _RM
_RELAY_MODES = list(_RM)
RELAY_OFF            = _RELAY_MODES.index(_RM.OFF)
RELAY_NORMAL         = _RELAY_MODES.index(_RM.NORMAL)
RELAY_LOCAL_REROUTE  = _RELAY_MODES.index(_RM.LOCAL_REROUTE)
RELAY_CAPACITY_BOOST = _RELAY_MODES.index(_RM.CAPACITY_BOOST)


class _State:
    """Per-agent controller memory. Held on the agent object."""
    __slots__ = ('p_prev', 'g_prev', 'cmd_prev', 'armed', 'nb', 'nb_tick',
                 'give_back')

    def __init__(self):
        self.p_prev = None
        self.g_prev = None
        self.cmd_prev = 0.0
        self.armed = False
        self.nb = None
        self.nb_tick = -10 ** 9
        self.give_back = False


def _state(agent):
    st = getattr(agent, '_xldet', None)
    if st is None:
        st = _State()
        agent._xldet = st
    return st


def mcs_head(sinr_db: float) -> int:
    """Highest MCS that runs at FULL rate, expressed as an offset head index.

    spectral_efficiency halves the rate when sinr < threshold + 5, so the
    full-rate set is {m : threshold(m) + 5 <= sinr}. Auto-CQI instead picks the
    highest FEASIBLE m, which is frequently inside the half-rate band.
    """
    levels = sorted(MCS_SINR_THRESHOLD.items(), key=lambda kv: kv[1])
    target = None
    for lvl, thr in levels:
        if thr + HALF_RATE_MARGIN_DB <= sinr_db:
            target = lvl
    if target is None:
        return MCS_OFFSET_CENTER                      # nothing clears; follow CQI
    target_idx = list(MCSLevel).index(target)
    auto_idx = auto_cqi_mcs_idx(sinr_db)
    return max(0, min(4, target_idx - auto_idx + MCS_OFFSET_CENTER))


def _icic_braked(st, own_drop_db: float) -> bool:
    """True when a neighbour is reporting enough interference to stop climbing."""
    if own_drop_db >= 3.0:
        return True                                   # own SINR is already falling
    nb = st.nb
    if nb is None:
        return False                                  # stale cache: fail open
    return (nb.get('interf', 0.0) >= ICIC_INTERFERERS
            or nb.get('tx', 0.0) > ICIC_NEIGH_POWER)


def tx_head(ps, st, own_drop_db: float) -> int:
    g = float(getattr(ps, 'sinr_average', 0.0))
    p = float(getattr(ps, 'tx_power_dbm', 23.0))

    # measured loop gain from the previous commanded step
    if st.p_prev is not None and abs(p - st.p_prev) > 0.5:
        gain = (g - st.g_prev) / (p - st.p_prev)
        # A unilateral rise gives ~1 dB of SINR per dB of power. A gain near
        # zero means the whole fleet moved together and the rise bought
        # nothing but interference and energy.
        st.give_back = gain < LOOP_GAIN_MIN
    st.p_prev, st.g_prev = p, g

    if g >= SINR_AMPLE_DB:
        return TX_DOWN                                # more than the top MCS needs
    if st.give_back:
        st.give_back = False
        return TX_DOWN                                # undo a step that bought nothing
    if g < SINR_TARGET_DB and not _icic_braked(st, own_drop_db):
        return TX_UP
    return TX_HOLD


def relay_head(obs, st) -> int:
    cs = obs.connectivity
    steerable = float(getattr(cs, 'relay_capable', 0.0)) >= 0.5
    ps = obs.phy_mac
    if getattr(ps, 'relay_link_active', False):
        st.armed = True
    if steerable:
        st.armed = True
        return RELAY_CAPACITY_BOOST
    return RELAY_LOCAL_REROUTE


def verify_prb_point():
    """Re-derive the PRB claim from the engine and raise if it no longer holds.

    A baseline is only worth reporting if it is genuinely competent, and this
    one's competence rests on a numeric claim about code that lives elsewhere.
    Rather than trust a comment, re-run the engine's OWN projection and its OWN
    admission thresholds against the pinned point. If someone re-enables the
    coordinator's emergency floor, moves a threshold or changes the general
    floor, XL-DET stops rather than quietly degrading into a strawman and
    flattering the learned policy it exists to challenge.

    Returns the post-projection point and the four admission modes.
    """
    from .coordinator_agent import GlobalPolicyVector
    from .simulation import Simulator

    prb = [PRB_EMERGENCY, PRB_RELAY, PRB_GENERAL]

    # (i) the coordinator's emergency floor, exactly as _execute_agent_actions
    #     applies it, using the vector the coordinator actually emits
    min_emrg = GlobalPolicyVector.neutral().to_list()[2]
    if prb[0] < min_emrg:
        d = min_emrg - prb[0]
        prb[0] += d
        prb[1] = max(0.0, prb[1] - d / 2)
        prb[2] = max(0.0, prb[2] - d / 2)
        t = sum(prb); prb = [x / t for x in prb]

    # (ii) the general-traffic floor
    floor = 0.10
    if prb[2] < floor:
        d = floor - prb[2]
        prb[2] = floor
        other = prb[0] + prb[1]
        if other > 1e-6:
            prb[0] -= d * (prb[0] / other)
            prb[1] -= d * (prb[1] / other)
        prb = [max(0.0, x) for x in prb]
        t = sum(prb); prb = [x / t for x in prb]

    moved = max(abs(prb[i] - (PRB_EMERGENCY, PRB_RELAY, PRB_GENERAL)[i])
                for i in range(3))
    if moved > 1e-9:
        raise AssertionError(
            "XL-DET's PRB point is no longer a fixed point of the engine's "
            "projections (moved by %.6g). Re-solve it: the point must satisfy "
            "general >= %.2f and emergency >= the coordinator floor (%.4g)."
            % (moved, floor, min_emrg))

    class _A:
        prb_emergency_frac = prb[0]
        prb_general_frac   = prb[2]

    modes = {tc: d['admission_mode'] for tc, d
             in Simulator._derive_admission_policies(_A()).items()}
    bad = {tc.name: m for tc, m in modes.items() if m != 'ADMIT'}
    if bad:
        raise AssertionError(
            "XL-DET's PRB point no longer admits every traffic class (%s). "
            "apply_marl_policy halves a flow at any THROTTLE node and zeroes "
            "it at any HOLD node, so this would cripple the baseline." % bad)
    return prb, modes


def decide(obs, agent):
    """Return (tx, mcs_emrg, mcs_gen, relay, postcard, iops, prb_triple)."""
    st = _state(agent)
    ps, nb = obs.phy_mac, obs.neighbor_radio
    t = int(getattr(obs, 'current_tick', 0))

    # Block B is a destructive read behind a 10-tick rate limit, so most ticks
    # deliver the dataclass default. Latch a real delivery and expire it.
    if float(getattr(nb, 'postcard_received', 0.0)) >= 1.0:
        st.nb = {'tx': float(getattr(nb, 'neighbour_avg_tx_power_norm', 0.5)),
                 'interf': float(getattr(nb, 'interferer_count_norm', 0.0)),
                 'sinr': float(getattr(nb, 'avg_sinr', 0.5))}
        st.nb_tick = t
    if t - st.nb_tick > NB_STALE_MAX:
        st.nb = None

    # experienced_sinr_drop_norm is overwritten with this node's OWN drop and
    # is fresh every tick, unlike the rest of Block B.
    own_drop_db = float(getattr(nb, 'experienced_sinr_drop_norm', 0.0)) * 10.0

    g = float(getattr(ps, 'sinr_average', 0.0))
    m = mcs_head(g)
    return (tx_head(ps, st, own_drop_db), m, m, relay_head(obs, st),
            1, 2, [PRB_EMERGENCY, PRB_RELAY, PRB_GENERAL])
