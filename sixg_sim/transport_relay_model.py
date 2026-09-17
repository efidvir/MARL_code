"""
Transport radio + relay model for the 6G disaster-recovery simulator.

HYBRID TRANSPORT (design decision 1).  This module no longer models a single
"relay radio".  Post-disaster transport in this simulator is HYBRID, and the
two media have completely different physics, so they get two separate link
budgets and two separate sets of operational rules:

  ┌──────────────────────┬────────────────────────┬────────────────────────┐
  │                      │ TG mesh (60 GHz)       │ MW PtP                 │
  ├──────────────────────┼────────────────────────┼────────────────────────┤
  │ role                 │ SHORT, FAST, STEERABLE │ LONG, STATIC           │
  │ carrier              │ 60 GHz (V-band)        │ 18 GHz                 │
  │ channel BW           │ 2160 MHz               │ 112 MHz                │
  │ antenna              │ phased array, 24 dBi   │ 0.3 m parabola, 34 dBi │
  │ oxygen absorption    │ ~15 dB/km              │ negligible             │
  │ rain (25 mm/h)       │ ~10.1 dB/km (P.838-3)  │ ~2.3 dB/km (P.838-3)   │
  │ reach (clear air)    │ ~1.07 km @23 dBm       │ ~5 km (budget-rich)    │
  │                      │ ~1.53 km @33 dBm       │                        │
  │ re-pointable?        │ YES, electronically    │ NO — mechanically      │
  │                      │ (RELAY_REPOINT_TICKS)  │ aligned, static within │
  │                      │                        │ an episode             │
  │ interference-limited?│ no (noise-limited)     │ marginal (coordinated) │
  └──────────────────────┴────────────────────────┴────────────────────────┘

ONLY the TG class may form a NEW steered bridge.  MW PtP links are
pre-existing topology (LinkType.MICROWAVE_PTP): agents may bring them up or
down, but they cannot be aimed at a new peer inside an episode, because a
mechanically aligned parabola requires a truck roll and an antenna alignment
crew.  That is the single most consequential physical constraint in this
model: it replaces a 5 km steerable reach with a ~1 km steerable reach and
makes island reunification GEOGRAPHY-LIMITED rather than policy-limited.

WHAT THIS REPLACES.  The previous version of this file was internally
inconsistent: it advertised "60 GHz Siklu MultiHaul TG" in its docstring
while computing 3.5 GHz FR1 free-space path loss over a 20 MHz channel and
hardcoding MAX_RELAY_RANGE_M = 5000.0 — a range that no 60 GHz budget can
support and that the 3.5 GHz budget was never asked to justify.  Nothing here
hardcodes a range any more: `max_range_m()` SOLVES the link budget, so a
change to power, bandwidth, noise figure or rain rate moves the reach.

NOT 3GPP IAB AT THE TRANSPORT LAYER.  The transport hop is a dedicated
transport radio (TG mesh / MW PtP / fibre), not an in-band IAB backhaul, so
it does not share PRBs with the FR1 access carrier (see
PHYMACState.normalise_prb).  Design decision 2 does add an IAB-STYLE
INTEGRATED ACCESS function — a relay site hosts its own FR1 small cell and
can serve a UE directly — and that access link IS modelled on the FR1 access
budget here (see `integrated_access_link_capacity`).

PHYSICS AND SOURCES
-------------------
Free-space path loss (Friis, dB form):
    FSPL = 20·log10(d_m) + 20·log10(f_MHz) − 27.55

Thermal noise floor:
    N = −174 dBm/Hz + 10·log10(B_Hz) + NF
  (−174 dBm/Hz is kT at 290 K.)

Oxygen (dry-air) absorption at 60 GHz: ITU-R P.676 puts the 60 GHz O2
absorption complex at ~15 dB/km at sea level; this is the term that makes
60 GHz a fundamentally SHORT-range medium and it is not optional.

Rain specific attenuation: ITU-R P.838-3,  γ_R = k · R^α  [dB/km], with the
horizontal-polarisation coefficients tabulated per frequency:
    60 GHz:  k = 0.8606,  α = 0.7656  →  25 mm/h ⇒ 10.1 dB/km
    18 GHz:  k = 0.07078, α = 1.0818  →  25 mm/h ⇒  2.3 dB/km
Path attenuation for the short hops modelled here is γ_R · d (i.e. the
ITU-R P.530 path-length reduction factor is ~1 below ~2 km and is omitted;
for the MW class, where hops reach 5 km, P.530's reduction factor would only
REDUCE the attenuation, so omitting it is the conservative choice).

Rain rate is a PARAMETER (`rain_mm_h`), default 0.0 = clear air.  Every
range/capacity helper accepts it, so a rain-fade sensitivity sweep needs no
code change.

Spectral efficiency comes from the SHARED MCS tables in phy_mac_state.py
(MCS_SINR_THRESHOLD / MCS_SPECTRAL_EFFICIENCY), so transport and access use
one link-adaptation model.  The TG class additionally caps SE at the IEEE
802.11ad/TG single-carrier peak (MCS12: 4620 Mbps in a 2160 MHz channel ⇒
2.14 bit/s/Hz); without that cap the QAM256 row (4.0 bit/s/Hz) would imply
8.6 Gbps from a radio whose modem cannot exceed 4.6 Gbps.

INTERFERENCE.  Every SINR helper here takes an optional `interference_dbm`
and combines it with thermal noise IN THE LINEAR DOMAIN — SINR = S/(N+I).
The aggregate-interference term itself is computed by the simulator
(Simulator._compute_aggregate_interference), because only the simulator knows
which transmitters are ACTIVE this tick.  Passing None means "thermal-noise
only", which is the correct default for build-time feasibility screening
(e.g. the scenario builder, which runs before any simulator exists).
"""

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from .phy_mac_state import (
    MCSLevel, MCS_SINR_THRESHOLD, MCS_SPECTRAL_EFFICIENCY,
)

# ── Universal constants ───────────────────────────────────────────────────────

KT_DBM_PER_HZ = -174.0     # kT at 290 K, dBm/Hz
FSPL_CONST_DB = -27.55     # Friis constant for (metres, MHz)


# ── ITU-R P.838-3 rain coefficients (horizontal polarisation) ────────────────
#
# γ_R = k · R^α  [dB/km].  Only the two carriers this simulator uses are
# tabulated; anything else raises, deliberately, rather than silently
# extrapolating a rain model outside its tabulated support.
P838_RAIN_COEFFS: Dict[float, Tuple[float, float]] = {
    3.5:  (0.0001071, 0.9247),   # FR1 access — rain is negligible here
    18.0: (0.07078,   1.0818),
    60.0: (0.8606,    0.7656),
}


def rain_attenuation_db_per_km(freq_ghz: float, rain_mm_h: float) -> float:
    """ITU-R P.838-3 specific rain attenuation γ_R = k·R^α  [dB/km].

    freq_ghz must be one of the tabulated carriers (3.5, 18, 60 GHz).
    rain_mm_h = 0 returns 0 exactly (clear air).
    """
    if rain_mm_h <= 0.0:
        return 0.0
    try:
        k, alpha = P838_RAIN_COEFFS[float(freq_ghz)]
    except KeyError:
        raise KeyError(
            f"No ITU-R P.838-3 rain coefficients tabulated for {freq_ghz} GHz; "
            f"available: {sorted(P838_RAIN_COEFFS)}")
    return k * (rain_mm_h ** alpha)


# ── Radio class (one per band) ────────────────────────────────────────────────

@dataclass(frozen=True)
class TransportRadioClass:
    """A complete link budget for one transport (or access) radio class.

    Everything downstream — feasible range, capacity, interference coupling —
    is DERIVED from these fields.  No range or capacity constant is hardcoded
    anywhere else in this module.
    """
    name:                    str
    freq_ghz:                float
    bandwidth_mhz:           float
    noise_figure_db:         float
    ant_gain_dbi:            float   # per end, boresight
    impl_loss_db:            float   # modem + feeder + pointing loss, total
    oxygen_db_per_km:        float
    peak_se_bps_hz:          float   # modem ceiling on spectral efficiency
    steerable:               bool    # may form a NEW link inside an episode?
    sidelobe_discrim_db:     float   # per end, off-boresight discrimination
    channel_reuse:           int     # co-channel fraction = 1/channel_reuse
    link_margin_db:          float = 0.0
    # ^ Shadow-fading / body-loss allowance, charged to the WANTED signal
    #   only.  Interference is computed WITHOUT the margin (the conservative
    #   choice: a margin subtracted from the interferer would flatter SINR).
    #   Zero for the transport classes, where the fade mechanisms (oxygen
    #   absorption and ITU-R P.838-3 rain) are modelled explicitly instead of
    #   being wrapped in a blanket margin.  See ACCESS_FR1 for the FR1 value.

    # ── Path loss and atmosphere ──────────────────────────────────────────

    def fspl_db(self, dist_m: float) -> float:
        """Friis free-space path loss.  Clamped to 0 below 1 m."""
        if dist_m <= 1.0:
            return 0.0
        return (20.0 * math.log10(dist_m)
                + 20.0 * math.log10(self.freq_ghz * 1000.0)
                + FSPL_CONST_DB)

    def atmospheric_loss_db(self, dist_m: float, rain_mm_h: float = 0.0) -> float:
        """Oxygen absorption + ITU-R P.838-3 rain, over the path length."""
        km = max(0.0, dist_m) / 1000.0
        return km * (self.oxygen_db_per_km
                     + rain_attenuation_db_per_km(self.freq_ghz, rain_mm_h))

    def total_path_loss_db(self, dist_m: float, rain_mm_h: float = 0.0) -> float:
        return self.fspl_db(dist_m) + self.atmospheric_loss_db(dist_m, rain_mm_h)

    # ── Noise ─────────────────────────────────────────────────────────────

    def noise_floor_dbm(self) -> float:
        """Thermal noise in the channel + receiver noise figure."""
        return (KT_DBM_PER_HZ
                + 10.0 * math.log10(self.bandwidth_mhz * 1e6)
                + self.noise_figure_db)

    def effective_noise_floor_dbm(self) -> float:
        """Noise floor referred to the TRANSMITTER, i.e. with the two-end
        antenna gain and implementation loss folded in, so that

            SINR = P_tx − PL − effective_noise_floor_dbm()

        holds with no further terms (see ACCESS_FR1 for the decomposition).
        """
        return (self.noise_floor_dbm()
                - 2.0 * self.ant_gain_dbi
                + self.impl_loss_db
                + self.link_margin_db)

    # ── SINR / capacity ───────────────────────────────────────────────────

    def rx_power_dbm(self, tx_power_dbm: float, dist_m: float,
                     rain_mm_h: float = 0.0) -> float:
        """Received power on the INTENDED (boresight-to-boresight) path,
        net of implementation loss and the shadow-fading link margin."""
        return (tx_power_dbm + 2.0 * self.ant_gain_dbi
                - self.total_path_loss_db(dist_m, rain_mm_h)
                - self.impl_loss_db
                - self.link_margin_db)

    def interference_power_dbm(self, tx_power_dbm: float, dist_m: float,
                               rain_mm_h: float = 0.0) -> float:
        """Received power from an UNINTENDED co-channel transmitter.

        Both ends are pointed somewhere else, so each contributes
        `sidelobe_discrim_db` of off-boresight discrimination (ITU-R F.699
        style: a 24 dBi phased array or a 34 dBi parabola radiates 15-20 dB
        below boresight into a random direction).  For a quasi-omni /
        3-sector access antenna this is small; for a 60 GHz pencil beam it is
        the dominant reason 60 GHz mesh is noise-limited rather than
        interference-limited.
        """
        return (tx_power_dbm + 2.0 * (self.ant_gain_dbi - self.sidelobe_discrim_db)
                - self.total_path_loss_db(dist_m, rain_mm_h))

    def sinr_db(self, tx_power_dbm: float, dist_m: float,
                rain_mm_h: float = 0.0,
                interference_dbm: Optional[float] = None) -> float:
        """SINR = S / (N + I), combined in the LINEAR domain.

        interference_dbm=None  →  thermal-noise-limited (I = 0).
        """
        s_dbm = self.rx_power_dbm(tx_power_dbm, dist_m, rain_mm_h)
        n_lin = 10.0 ** (self.noise_floor_dbm() / 10.0)
        i_lin = (0.0 if interference_dbm is None
                 else 10.0 ** (float(interference_dbm) / 10.0))
        return s_dbm - 10.0 * math.log10(n_lin + i_lin)

    def best_mcs_for_sinr(self, sinr_db: float) -> Tuple[float, float]:
        """(spectral_efficiency, threshold) for the best MCS that closes at
        this SINR, capped at the modem's `peak_se_bps_hz`.  (0, 0) if none."""
        best_se, best_thr = 0.0, 0.0
        for mcs in reversed(list(MCSLevel)):
            if sinr_db >= MCS_SINR_THRESHOLD[mcs]:
                best_se = min(MCS_SPECTRAL_EFFICIENCY[mcs], self.peak_se_bps_hz)
                best_thr = MCS_SINR_THRESHOLD[mcs]
                break
        return best_se, best_thr

    def capacity_mbps(self, sinr_db: float, bw_fraction: float = 1.0) -> float:
        """Shannon-free capacity: BW × SE(MCS) × bw_fraction."""
        se, _ = self.best_mcs_for_sinr(sinr_db)
        if se <= 0.0:
            return 0.0
        return self.bandwidth_mhz * se * max(0.0, min(1.0, bw_fraction))

    # ── Derived reach (NOT a hardcoded constant) ──────────────────────────

    def min_usable_sinr_db(self) -> float:
        """SINR at which the most robust MCS still closes the link."""
        return min(MCS_SINR_THRESHOLD.values())

    def max_range_m(self, tx_power_dbm: float, rain_mm_h: float = 0.0,
                    interference_dbm: Optional[float] = None,
                    resolution_m: float = 5.0,
                    hard_ceiling_m: float = 50000.0) -> float:
        """Solve the link budget for the longest hop that still closes.

        Monotone in distance (FSPL and atmospheric loss both increase), so a
        bisection is exact to `resolution_m`.  Returns 0.0 if even a 1 m hop
        cannot close (only possible with absurd interference).

        This is THE range gate.  There is no MAX_RANGE constant to contradict
        it: change the power, the bandwidth, the noise figure or the rain
        rate and the reach moves.
        """
        thr = self.min_usable_sinr_db()

        def closes(d: float) -> bool:
            return self.sinr_db(tx_power_dbm, d, rain_mm_h,
                                interference_dbm) >= thr

        if not closes(1.0):
            return 0.0
        lo, hi = 1.0, min(hard_ceiling_m, 2.0)
        while hi < hard_ceiling_m and closes(hi):
            lo, hi = hi, min(hard_ceiling_m, hi * 2.0)
        if closes(hi):
            return hi
        while hi - lo > resolution_m:
            mid = 0.5 * (lo + hi)
            if closes(mid):
                lo = mid
            else:
                hi = mid
        return lo


# ── The three radio classes used by this simulator ───────────────────────────

# 60 GHz TG mesh — Siklu MultiHaul TG class (MH-N366 / MH-B100 family).
#   BW 2160 MHz         IEEE 802.11ad channel (single 2.16 GHz channel).
#   NF 8 dB             typical 60 GHz CMOS/SiGe front end.
#   24 dBi per end      MultiHaul TG steerable phased-array node (spec sheets
#                       quote ~24-25 dBi with electronic beam steering over a
#                       wide sector; a fixed 60 GHz PtP parabola would be
#                       35-38 dBi but cannot be steered, which is exactly the
#                       trade this class represents).
#   impl 2 dB           modem implementation + residual pointing loss.
#   O2 15 dB/km         ITU-R P.676 sea-level 60 GHz oxygen absorption.
#   peak SE 2.14        802.11ad SC MCS12: 4620 Mbps / 2160 MHz.
#   sidelobe 15 dB/end  conservative for a 24 dBi array whose far sidelobes
#                       sit ~24 dB below boresight.
#   channel_reuse 3     TG mesh nodes negotiate distinct 60 GHz channels
#                       (three non-overlapping 2.16 GHz channels are
#                       available in the 57-66 GHz unlicensed band).
# DERIVED clear-air reach: ~1.07 km @23 dBm, ~1.53 km @33 dBm.
# DERIVED 25 mm/h reach:   ~0.75 km @23 dBm, ~1.04 km @33 dBm.
TG_MESH_60GHZ = TransportRadioClass(
    name="TG_MESH_60GHZ",
    freq_ghz=60.0,
    bandwidth_mhz=2160.0,
    noise_figure_db=8.0,
    ant_gain_dbi=24.0,
    impl_loss_db=2.0,
    oxygen_db_per_km=15.0,
    peak_se_bps_hz=2.14,
    steerable=True,
    sidelobe_discrim_db=15.0,
    channel_reuse=3,
)

# 18 GHz microwave PtP — Ceragon IP-20 class, the pre-existing static
# topology links (LinkType.MICROWAVE_PTP).
#   BW 112 MHz          widest common licensed channel in the 18 GHz band
#                       (56 MHz is the other standard width; 112 MHz × the
#                       QAM256 row gives 448 Mbps, consistent with the
#                       500 Mbps MICROWAVE_PTP link capacity in the
#                       comparison topology).
#   NF 6 dB             licensed MW modem front end.
#   34 dBi per end      0.3 m parabola at 18 GHz.
#   O2 0                dry-air absorption is negligible below ~22 GHz.
#   steerable = False   MECHANICALLY ALIGNED.  Cannot be re-pointed inside an
#                       episode; realigning a parabola is a truck roll.
#   sidelobe 20 dB/end  ITU-R F.699 gives >30 dB off-axis beyond ~10° for a
#                       34 dBi parabola; 20 dB is the conservative choice.
#   channel_reuse 4     licensed PtP links are frequency-coordinated by the
#                       regulator, so co-channel collisions are rare by
#                       construction.
# DERIVED clear-air reach: >50 km @23 dBm — i.e. the 18 GHz budget is not
# the binding constraint at all.  MW reach is bounded by MW_MAX_HOP_M below
# (line-of-sight / Fresnel clearance and licensed hop planning), which is the
# physically correct reason a 5 km cap exists.
MW_PTP = TransportRadioClass(
    name="MW_PTP_18GHZ",
    freq_ghz=18.0,
    bandwidth_mhz=112.0,
    noise_figure_db=6.0,
    ant_gain_dbi=34.0,
    impl_loss_db=2.0,
    oxygen_db_per_km=0.0,
    peak_se_bps_hz=4.0,
    steerable=False,
    sidelobe_discrim_db=20.0,
    channel_reuse=4,
)

# MW PtP planning reach.  The 18 GHz link budget closes far beyond this; what
# actually bounds a real MW hop at this frequency is line-of-sight, Fresnel
# clearance over terrain/clutter and the licensed hop plan.  5 km matches the
# reach quoted for the pre-existing MW topology in this scenario family.
MW_MAX_HOP_M = 5000.0

# FR1 access carrier (3.5 GHz, 20 MHz) — the Uu carrier UEs camp on and the
# carrier a relay-hosted integrated-access small cell uses (design decision 2).
#
#   link_margin_db = 9.99   (a real margin, not a fudge factor)
#     WHY THIS EXISTS.  The pre-existing access model in simulation.py used
#     SINR = P_tx − FSPL(3.5 GHz) − (−100 dBm), i.e. an EFFECTIVE noise floor
#     of −100 dBm with no explicit antenna gain, no implementation loss and no
#     fading margin.  This class decomposes that single number into its real
#     physical terms and lands on exactly the same total:
#         kTB(20 MHz)                 = −100.99 dBm
#         receiver noise figure       =   +5.00 dB   →  N = −95.99 dBm
#         two-end antenna gain 2×8 dBi=  −16.00 dB
#         implementation/feeder loss  =   +2.00 dB
#         link margin (shadow fading +
#         body loss, log-normal σ≈8 dB
#         in 3GPP TR 38.901 UMa ⇒ ~10 dB
#         for ~90 % area coverage)    =   +9.99 dB
#         ────────────────────────────────────────────
#         effective floor             = −100.00 dBm
#     Reproducing the previous number EXACTLY is deliberate: it makes the
#     addition of the aggregate-interference term the ONLY change to the
#     access budget in this revision, so every measured SINR shift is
#     attributable to interference rather than to a silent re-tuning of the
#     link budget.
#   sidelobe_discrim_db = 5 dB/end
#     A 3-sector macro/small cell radiates into a random azimuth roughly
#     10 dB below boresight (65° HPBW, 30 dB front-to-back); 5 dB per end is
#     that, split across the two ends of the interference path.
#   channel_reuse = 3
#     Hard reuse-3 / fractional frequency reuse.  A 42-site FR1 deployment is
#     NOT single-frequency: without a reuse pattern every cell in an 8×8 km
#     area is co-channel and the network is in universal outage, which is a
#     modelling artefact rather than a physical result.  Co-channel membership
#     is assigned deterministically per node (crc32, see
#     `access_channel_group`) so it is identical for every arm and every run.
ACCESS_FR1 = TransportRadioClass(
    name="ACCESS_FR1_3G5",
    freq_ghz=3.5,
    bandwidth_mhz=20.0,
    noise_figure_db=5.0,
    ant_gain_dbi=8.0,
    impl_loss_db=2.0,
    oxygen_db_per_km=0.0,
    peak_se_bps_hz=4.0,
    steerable=False,
    sidelobe_discrim_db=5.0,
    channel_reuse=3,
    link_margin_db=9.99,
)

# Band identifiers used by the simulator's aggregate-interference pass.
BAND_ACCESS    = "access_fr1"
BAND_TG        = "transport_tg60"
BAND_MW        = "transport_mw"

RADIO_CLASS_BY_BAND: Dict[str, TransportRadioClass] = {
    BAND_ACCESS: ACCESS_FR1,
    BAND_TG:     TG_MESH_60GHZ,
    BAND_MW:     MW_PTP,
}


def access_channel_group(node_id: str, reuse: int = None) -> int:
    """Deterministic co-channel group for a node's FR1 access carrier.

    crc32 (NOT the builtin `hash`, which is salted per process) so the
    channel plan replays identically across runs, processes and arms.
    """
    r = ACCESS_FR1.channel_reuse if reuse is None else int(reuse)
    if r <= 1:
        return 0
    return zlib.crc32(node_id.encode('utf-8')) % r


def transport_channel_group(node_id: str, band: str) -> int:
    """Deterministic co-channel group for a transport radio (TG or MW)."""
    radio = RADIO_CLASS_BY_BAND[band]
    if radio.channel_reuse <= 1:
        return 0
    return zlib.crc32(f"{band}|{node_id}".encode('utf-8')) % radio.channel_reuse


# ── Integrated access node (design decision 2) ───────────────────────────────
#
# A relay site is an INTEGRATED ACCESS NODE: it hosts its own FR1 small cell,
# so serving a UE directly over the air is physically sound rather than a
# bookkeeping trick.  The small cell shares the site's ACCESS spectrum, not
# the transport radio, and its coverage is bounded by
#   (a) the FR1 link budget above (the same one every macro cell uses), and
#   (b) a coverage radius, because a small cell that cannot hear the UE's
#       uplink cannot serve it however good the downlink budget looks.
IAB_ACCESS_COVERAGE_RADIUS_M = 1000.0   # PHYMACState default for RELAY sites
IAB_ACCESS_MIN_USABLE_MBPS   = 0.5      # below this the link is not worth setting up


def integrated_access_link_capacity(
        dist_m: float,
        tx_power_dbm: float,
        prb_fraction: float,
        *,
        bandwidth_mhz: float = None,
        interference_dbm: Optional[float] = None,
        coverage_radius_m: float = IAB_ACCESS_COVERAGE_RADIUS_M,
        rain_mm_h: float = 0.0,
        phy_mac_state=None,
        for_emergency: bool = False,
) -> Tuple[float, float]:
    """Capacity of an ACCESS link from a relay-hosted FR1 small cell to a UE.

    DESIGN DECISION 2 — INTEGRATED ACCESS NODE (IAB-style).  The relay is not
    a bare transport box: it hosts an FR1 small cell, so a UE orphaned by the
    disaster can camp on the relay directly.  This function is the physical
    model of that access hop, and it REPLACES the linear-in-distance formula
    that used to live in the evaluation harness:

        dist_factor = max(0.3, 1.0 - dist / 1500.0)
        ue_cap      = 25.0 * prb_relay_fraction * dist_factor

    which had no link budget behind it, could not go into outage, and paid no
    attention to interference or to the site's own Tx power.

    THE MODEL
    ---------
      1. Coverage gate.  dist_m > coverage_radius_m  ⇒  (0.0, -inf).
      2. FR1 link budget (ACCESS_FR1): 3.5 GHz FSPL, the small cell's own Tx
         power, thermal noise + NF in the small cell's access bandwidth,
         combined with `interference_dbm` in the linear domain (S/(N+I)).
      3. SINR → MCS → spectral efficiency through the SHARED OLLA path.
         When a PHYMACState is supplied, its sinr_average is temporarily set
         to this link's SINR and PHYMACState.spectral_efficiency() is used,
         so the site's own MCS choice, its OLLA step-down and its half-rate
         near-threshold rule all apply exactly as they do on any other access
         link.  Without a PHYMACState, the plain auto-CQI MCS is used.
      4. Capacity = bandwidth × SE × prb_fraction.

    ARM-AGNOSTIC.  Nothing here is MARL-specific: it takes a distance, a Tx
    power and a PRB share.  Any arm (SDN, OSPF, OLSR, BATMAN, AODV, random,
    MARL) that decides to light up an integrated-access link gets the same
    physics.  `Simulator.integrated_access_capacity_mbps` is the convenience
    wrapper that resolves node positions, PRB share and the live
    aggregate-interference term from the simulator state.

    Args:
        dist_m:            small cell ↔ UE separation, metres.
        tx_power_dbm:      the small cell's current Tx power (agent- or
                           operator-controlled; PHYMACState.tx_power_dbm).
        prb_fraction:      share of the ACCESS carrier given to this link
                           (e.g. PHYMACState.prb_relay_fraction, or an
                           explicit small-cell allocation).
        bandwidth_mhz:     access channel width; defaults to ACCESS_FR1's
                           20 MHz.
        interference_dbm:  aggregate co-channel interference at the receiver,
                           dBm.  None = thermal-noise-only.
        coverage_radius_m: hard coverage gate (metres).  Pass
                           PHYMACState.effective_coverage_radius_m to make
                           coverage scale with Tx power.
        rain_mm_h:         ITU-R P.838-3 rain rate; negligible at 3.5 GHz but
                           accepted for interface symmetry.
        phy_mac_state:     optional PHYMACState of the serving site, used for
                           the OLLA / MCS-offset path.
        for_emergency:     select the emergency MCS row when a PHYMACState is
                           supplied.

    Returns:
        (capacity_mbps, sinr_db).  capacity_mbps == 0.0 means the link is not
        usable — out of coverage, or SINR below the most robust MCS.
        sinr_db is -inf when the coverage gate rejects the link.
    """
    bw = ACCESS_FR1.bandwidth_mhz if bandwidth_mhz is None else float(bandwidth_mhz)
    if dist_m > max(1.0, coverage_radius_m):
        return 0.0, float('-inf')

    d = max(1.0, float(dist_m))
    sinr = ACCESS_FR1.sinr_db(tx_power_dbm, d, rain_mm_h, interference_dbm)

    if phy_mac_state is not None:
        # Route through the site's own OLLA path so the MCS offset, the
        # graceful step-down and the near-threshold half-rate rule apply.
        _saved = phy_mac_state.sinr_average
        try:
            phy_mac_state.sinr_average = sinr
            se = phy_mac_state.spectral_efficiency(for_emergency=for_emergency)
        finally:
            phy_mac_state.sinr_average = _saved
        se = min(se, ACCESS_FR1.peak_se_bps_hz)
    else:
        se, _thr = ACCESS_FR1.best_mcs_for_sinr(sinr)

    if se <= 0.0:
        return 0.0, sinr
    cap = bw * se * max(0.0, min(1.0, prb_fraction))
    return cap, sinr


# ── Relay link bookkeeping ───────────────────────────────────────────────────

@dataclass
class TransportRelayLink:
    """A wireless transport backhaul link created by a steered TG bridge."""
    link_id:       str
    source_node:   str
    target_node:   str
    capacity_mbps: float
    sinr_db:       float
    active:        bool = True
    radio_class:   str = TG_MESH_60GHZ.name


# Backward-compatible alias
IABLink = TransportRelayLink


class TransportRelayModel:
    """Transport radio physics + the set of active steered relay links.

    Integration with topology:
      When create_link() is called, the caller adds a corresponding Link
      object to Topology so the NetworkX graph reflects the new wireless hop;
      remove_link() signals the caller to drop it again.

    STEERING RULE (design decision 1).  can_form_relay_link() evaluates the
    TG MESH class only, because that is the only class that can be
    electronically re-pointed at a new peer inside an episode.  MW PtP hops
    are static pre-existing topology; `mw_link_feasible()` exists to sanity
    check an EXISTING MW hop's budget, and deliberately does NOT offer a
    "find me a new MW peer" entry point.
    """

    # ── Radio classes ────────────────────────────────────────────────────
    TG    = TG_MESH_60GHZ
    MW    = MW_PTP
    ACCESS = ACCESS_FR1

    # ── Legacy attribute surface (kept so existing callers keep working) ──
    #
    # NOISE_FLOOR_DBM / BANDWIDTH_MHZ / CARRIER_FREQ_GHZ describe the FR1
    # ACCESS carrier, which is what every existing consumer
    # (Simulator._update_dynamic_sinr) actually meant by them.  They are now
    # DERIVED from ACCESS_FR1 rather than being independent magic numbers.
    NOISE_FLOOR_DBM  = ACCESS_FR1.effective_noise_floor_dbm()   # ≈ −100.0 dBm
    BANDWIDTH_MHZ    = ACCESS_FR1.bandwidth_mhz                 # 20.0
    CARRIER_FREQ_GHZ = ACCESS_FR1.freq_ghz                      # 3.5
    D2D_CAPACITY_MBPS = 2.0      # Fixed capacity per active D2D sidelink pair

    # Nominal Tx power used when a caller does not supply one.
    NOMINAL_TX_DBM = 23.0

    def __init__(self, rain_mm_h: float = 0.0):
        """rain_mm_h: ITU-R P.838-3 rain rate for every transport budget.
        Default 0.0 = clear air.  Set it to 25.0 for the standard heavy-rain
        sensitivity case; it shortens the TG reach from ~1.07 km to ~0.75 km
        at 23 dBm."""
        self.active_links: Dict[str, TransportRelayLink] = {}
        self.node_positions: Dict[str, Tuple[float, float]] = {}
        self.rain_mm_h: float = float(rain_mm_h)

    # ── Position registration ────────────────────────────────────────────

    def register_position(self, node_id: str, x_m: float, y_m: float):
        self.node_positions[node_id] = (x_m, y_m)

    def distance_m(self, node_a: str, node_b: str) -> Optional[float]:
        """Euclidean distance between two nodes (None if unknown)."""
        if node_a not in self.node_positions or node_b not in self.node_positions:
            return None
        xa, ya = self.node_positions[node_a]
        xb, yb = self.node_positions[node_b]
        return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

    # ── Derived reach (replaces the old MAX_RELAY_RANGE_M constant) ───────

    def tg_max_range_m(self, tx_power_dbm: float = None,
                       interference_dbm: Optional[float] = None) -> float:
        """Longest TG hop that closes at this power / rain / interference.

        This is the reach that bounds NEW steered bridges.  There is no
        constant to contradict it — see TransportRadioClass.max_range_m.
        """
        tx = self.NOMINAL_TX_DBM if tx_power_dbm is None else tx_power_dbm
        return self.TG.max_range_m(tx, self.rain_mm_h, interference_dbm)

    def mw_max_range_m(self, tx_power_dbm: float = None) -> float:
        """Longest MW PtP hop: the smaller of the 18 GHz budget reach and the
        line-of-sight / hop-planning cap MW_MAX_HOP_M."""
        tx = self.NOMINAL_TX_DBM if tx_power_dbm is None else tx_power_dbm
        return min(MW_MAX_HOP_M, self.MW.max_range_m(tx, self.rain_mm_h))

    @property
    def MAX_RELAY_RANGE_M(self) -> float:
        """DEPRECATED name, retained for compatibility.  Now DERIVED from the
        TG link budget at nominal power instead of the old hardcoded 5000 m
        (which the 60 GHz budget cannot support by a factor of ~5)."""
        return self.tg_max_range_m()

    # ── Path loss / SINR (FR1 access, legacy signatures) ─────────────────

    def free_space_path_loss_db(self, dist_m: float) -> float:
        """FR1 (3.5 GHz) free-space path loss — the ACCESS budget.

        Kept at 3.5 GHz because every existing caller
        (Simulator._update_dynamic_sinr, the cell-edge coverage proxy) is
        computing an ACCESS link budget.  Transport path loss must go through
        TG_MESH_60GHZ / MW_PTP, which have their own carriers and their own
        atmospheric terms.
        """
        return self.ACCESS.fspl_db(dist_m)

    def access_sinr_db(self, tx_power_dbm: float, dist_m: float,
                       interference_dbm: Optional[float] = None) -> float:
        """FR1 access SINR = S/(N+I) with the interference term in linear."""
        return self.ACCESS.sinr_db(tx_power_dbm, dist_m, 0.0, interference_dbm)

    def estimate_sinr_db(self, tx_power_dbm: float, dist_m: float,
                         interference_dbm: float = -120.0) -> float:
        """Legacy signature (FR1 access).  interference_dbm is an absolute
        dBm level, combined with thermal noise in the linear domain."""
        return self.access_sinr_db(tx_power_dbm, dist_m, interference_dbm)

    def band_interference_power_dbm(self, band: str, tx_power_dbm: float,
                                    dist_m: float) -> float:
        """Received interference power from one co-channel transmitter.

        Used by Simulator._compute_aggregate_interference to build the
        aggregate I term for each band; exposed here so the antenna
        discrimination and atmospheric terms stay with the radio class.
        """
        radio = RADIO_CLASS_BY_BAND[band]
        return radio.interference_power_dbm(tx_power_dbm, max(1.0, dist_m),
                                            self.rain_mm_h)

    # ── MCS helper (legacy signature; TG modem cap applied) ──────────────

    def best_mcs_for_sinr(self, sinr_db: float) -> Tuple[float, float]:
        """(spectral_efficiency, sinr_threshold) for the best TG MCS."""
        return self.TG.best_mcs_for_sinr(sinr_db)

    # ── TG link feasibility (the ONLY steerable class) ───────────────────

    def can_form_relay_link(self, source: str, target: str,
                            tx_power_dbm: float,
                            relay_bw_fraction: float = 0.20,
                            interference_dbm: Optional[float] = None
                            ) -> Tuple[bool, float, float]:
        """Can a NEW steered 60 GHz TG bridge be formed source→target?

        TG MESH ONLY.  MW PtP is mechanically aligned and cannot be pointed
        at a new peer inside an episode, so it is never a candidate here.

        Budget: 60 GHz FSPL + 15 dB/km oxygen + ITU-R P.838-3 rain, 2×24 dBi,
        2160 MHz, NF 8 dB, SINR = S/(N+I), SE from the shared MCS table
        capped at the 802.11ad SC MCS12 modem ceiling (2.14 bit/s/Hz).

        Range is NOT a separate gate: the SINR check IS the range check, so a
        power change moves the reach instead of hitting a hardcoded wall.
        (`tg_max_range_m` reports the resulting reach for diagnostics.)

        Args:
            source, target:    node ids with registered positions.
            tx_power_dbm:      source radio Tx power.
            relay_bw_fraction: share of the TG channel given to this hop
                               (PHYMACState.prb_relay_fraction).
            interference_dbm:  aggregate co-channel 60 GHz interference at
                               the receiver; None = thermal-noise-only.

        Returns:
            (feasible, capacity_mbps, sinr_db)
        """
        dist = self.distance_m(source, target)
        if dist is None:
            # Unknown geometry: refuse rather than invent a capacity.  The old
            # model returned "feasible, 8 Mbps, 12 dB" here, which let a
            # bridge form between two nodes whose distance was unknown — i.e.
            # an unbounded-range bridge.  Every caller in this codebase
            # registers positions first, so this branch is a bug guard.
            return False, 0.0, float('-inf')

        sinr_db = self.TG.sinr_db(tx_power_dbm, max(1.0, dist),
                                  self.rain_mm_h, interference_dbm)
        cap = self.TG.capacity_mbps(sinr_db, relay_bw_fraction)
        if cap <= 0.0:
            return False, 0.0, sinr_db
        return True, cap, sinr_db

    # Backward-compatible alias
    can_form_iab_link = can_form_relay_link

    def mw_link_feasible(self, source: str, target: str,
                         tx_power_dbm: float = None,
                         bw_fraction: float = 1.0,
                         interference_dbm: Optional[float] = None
                         ) -> Tuple[bool, float, float]:
        """Budget check for an EXISTING MW PtP hop (never for a new one).

        MW_PTP.steerable is False, so this method is only meaningful for a
        link that already exists in the topology: it answers "does this
        pre-aimed hop still close, and at what capacity", e.g. after rain
        fade or a power change.  It intentionally has no candidate-search
        counterpart — a mechanically aligned parabola has no new peers.
        """
        dist = self.distance_m(source, target)
        if dist is None:
            return False, 0.0, float('-inf')
        if dist > MW_MAX_HOP_M:
            return False, 0.0, float('-inf')
        tx = self.NOMINAL_TX_DBM if tx_power_dbm is None else tx_power_dbm
        sinr = self.MW.sinr_db(tx, max(1.0, dist), self.rain_mm_h,
                               interference_dbm)
        cap = self.MW.capacity_mbps(sinr, bw_fraction)
        return (cap > 0.0), cap, sinr

    # ── Link lifecycle ───────────────────────────────────────────────────

    def create_link(self, source: str, target: str,
                    capacity_mbps: float, sinr_db: float = 15.0,
                    radio_class: str = None) -> str:
        """Create a transport relay link and return its ID."""
        link_id = f"TR_{source}_{target}"
        self.active_links[link_id] = TransportRelayLink(
            link_id=link_id,
            source_node=source,
            target_node=target,
            capacity_mbps=capacity_mbps,
            sinr_db=sinr_db,
            active=True,
            radio_class=(self.TG.name if radio_class is None else radio_class),
        )
        return link_id

    def remove_link(self, link_id: str):
        """Remove a transport relay link (returns its data or None)."""
        return self.active_links.pop(link_id, None)

    def update_link_capacity(self, link_id: str, new_cap: float):
        """Update capacity of an existing relay link (e.g. after power change)."""
        if link_id in self.active_links:
            self.active_links[link_id].capacity_mbps = new_cap

    def get_node_links(self, node_id: str) -> List[TransportRelayLink]:
        """Return all active transport relay links touching a node."""
        return [l for l in self.active_links.values()
                if l.active and node_id in (l.source_node, l.target_node)]

    def clear_node_links(self, node_id: str) -> List[str]:
        """Remove all relay links for a node; return list of removed link IDs."""
        to_remove = [lid for lid, l in self.active_links.items()
                     if node_id in (l.source_node, l.target_node)]
        for lid in to_remove:
            self.active_links.pop(lid, None)
        return to_remove

    # ── D2D helpers ──────────────────────────────────────────────────────

    def estimate_d2d_pair_capacity(self, ue_count: int) -> float:
        """
        Total sidelink capacity added by D2D relay mode.
        Simplified: each UE pair gets D2D_CAPACITY_MBPS Mbps of sidelink.
        """
        pairs = max(0, ue_count // 2)
        return pairs * self.D2D_CAPACITY_MBPS

    # ── Candidate discovery (TG only) ────────────────────────────────────

    def find_relay_candidates(self, source: str,
                              candidate_nodes: List[str],
                              tx_power_dbm: float,
                              relay_bw_fraction: float = 0.20,
                              max_candidates: int = 3,
                              interference_dbm=None
                              ) -> List[Tuple[str, float, float]]:
        """Best NEW steered TG bridge candidates for a source node.

        Returns [(node_id, capacity_mbps, sinr_db)] sorted by capacity desc.

        interference_dbm may be a scalar (same I at every candidate) or a
        callable node_id -> dBm (per-receiver I, which is what the simulator
        passes).  None = thermal-noise-only.
        """
        def _i_for(target: str):
            if interference_dbm is None:
                return None
            if callable(interference_dbm):
                return interference_dbm(target)
            return interference_dbm

        results = []
        for target in candidate_nodes:
            if target == source:
                continue
            feasible, cap, sinr = self.can_form_relay_link(
                source, target, tx_power_dbm, relay_bw_fraction,
                _i_for(target)
            )
            if feasible and cap > 0:
                results.append((target, cap, sinr))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_candidates]

    # Backward-compatible alias
    find_iab_candidates = find_relay_candidates


# Backward-compatible aliases for imports
IABRelayModel = TransportRelayModel
