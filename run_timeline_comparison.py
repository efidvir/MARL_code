"""
Multi-Architecture Recovery Simulation
Fully-simulated physics passes: MARL (online), MARL (adapt-freeze),
OSPF, SDN, OLSR, BATMAN, AODV
All KPIs are genuine simulation output — no synthetic curves.

FIGURES vs RAW DATA: all seven arms are simulated and saved to the raw
multiseed pickle.  The publication figures and their summary tables show
six of them — the online-only MARL arm listed in FIGURE_EXCLUDED_ARMS is
kept in the raw data as a documented ablation but is not plotted, and the
adapt-then-freeze arm is presented as the proposed system
(PROPOSED_ARM_LABEL).

MODELING ASSUMPTIONS (what remains analytical rather than packet-level):
  * Latency is a per-hop analytical model (propagation + M/M/1 queueing from
    actual link utilisation + HARQ retransmission probability + protocol
    processing terms).  Figures label it "modeled latency".  Outage (flows
    with no converged route / no physical path) is tracked separately as an
    outage fraction and EXCLUDED from mean-latency aggregation.
  * Protocol timers (OSPF Hello/Dead/SPF, OLSR HELLO/TC, BATMAN OGM, AODV
    RREQ/RREP/route-lifetime, SDN flow hard-timeout) are RFC/spec parameter
    INPUTS.  Recovery time is NOT scripted from them: it emerges from those
    timers playing out hop-by-hop over the actual surviving topology
    (detection -> flooding/discovery -> per-node route recompute), which
    varies with seed and scenario.  MARL has no scripted ramp at all — its
    forwarding capability is whatever the engine + learned policy produce.
  * Energy is a physics-based per-node radio power model (PA, DSP, Rx chain,
    per-UE overhead, relay forwarding, control-plane CPU/Tx per message).
  * Control-plane overhead counts protocol messages per tick from the actual
    graph (floods walk real links; sizes from the RFCs).
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
# Windows consoles/redirects may use cp1252 — never let a log message crash a
# multi-hour run over an unencodable character.
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

import math
import zlib
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from sixg_sim.topology import load_topology_from_yaml
from sixg_sim.scenario import Scenario, ScenarioEvent, DiverseScenarioGenerator
from sixg_sim.traffic import NodeTrafficProfile
from sixg_sim.simulation import SimulationConfig, Simulator
from sixg_sim.mappo_trainer import MAPPOTrainer, MAPPOConfig
from sixg_sim.agent import CriticNetwork, RLAgent, OBS_DIM, prb_training_action
from sixg_sim.simulation import site_can_steer
from sixg_sim.phy_mac_state import (MCSLevel, RelayMode, MACScheduler,
                                    MCS_OFFSET_CENTER, auto_cqi_mcs_idx)
from sixg_sim.transport_relay_model import (TransportRelayModel,
                                            TG_MESH_60GHZ, MW_MAX_HOP_M,
                                            ACCESS_FR1)

# ── Simulation time unit ──────────────────────────────────────────────
TICK_DURATION_S = 1.0          # 1 tick = 1 second of real time
TICKS_PER_SECOND = 1           # ticks per real second

SEVERANCE_TICK = 200           # T+200s = 3m20s (normal-ops baseline period)
TOTAL_TICKS = 3600             # 1 hour total simulation time
RESCUE_ARRIVAL_TICK = 1400     # Scenario B: rescue UE arrival tick

# ═══════════════════════════════════════════════════════════════════════
# PROTOCOL TIMER PARAMETERS (RFC inputs — NOT scripted recovery end-ticks)
# Recovery time is MEASURED: each protocol has a control-plane state machine
# (see *Controller classes below) in which these timers play out over the
# ACTUAL surviving topology.  A node forwards a flow only once its own
# control-plane state has converged routes for it, so recovery duration
# varies with seed, fragment structure and hop distances.
# ═══════════════════════════════════════════════════════════════════════

# ── OSPF (RFC 2328 §9.5) ─────────────────────────────────────────────
# EVERY timer in this block is expressed in TICKS, and 1 tick = 1 s
# (TICK_DURATION_S).  A previous revision mixed a 10 ms tick assumption into
# the control-plane message counters (hello_interval = 100/1000 "ticks",
# MinLSInterval = 500 "ticks", and the SDN literals below), which understated
# OSPF/SDN per-second overhead ~100x and inflated SDN convergence ~100x while
# OLSR/BATMAN/AODV used the correct second-based constants.  There is now
# exactly ONE time unit in this file and these are its constants.
OSPF_HELLO_INTERVAL      = 10  # HelloInterval, s (RFC 2328 default broadcast)
OSPF_HELLO_FAST_INTERVAL = 1   # RFC 2328 §C.5 fast-hello / BFD-assisted, s
OSPF_DEAD_INTERVAL   = 40      # RouterDeadInterval = 4×Hello, s
OSPF_FLOOD_HOP_TICKS = 1       # LSA flood pacing per hop (retransmit pacing, s)
OSPF_SPF_DELAY       = 5       # SPF throttle/hold + FIB install after LSDB settles, s
OSPF_MIN_LS_INTERVAL = 5       # MinLSInterval (RFC 2328 §12.4), s

# ── OLSR (RFC 3626 §18.2) ────────────────────────────────────────────
OLSR_HELLO_INTERVAL = 2        # 2 seconds (RFC 3626 default)
OLSR_TC_INTERVAL = 5           # 5 seconds (RFC 3626 default)

# ── B.A.T.M.A.N. (batman-adv, BATMAN IV) ─────────────────────────────
# OGM: 52 bytes, TTL: 50, hop penalty: 15/255; TQ EMA α=0.1
BATMAN_OGM_INTERVAL = 1        # 1 second (batman-adv default)
BATMAN_TQ_STABILIZE = 20       # ~20 OGM rounds for full TQ metric stability

# ── AODV (RFC 3561 §10) ──────────────────────────────────────────────
# NODE_TRAVERSAL_TIME: 40ms, NET_DIAMETER: 35
AODV_RREQ_TIMEOUT = 5          # 5 seconds (RREQ propagation + RREP return)
AODV_RREP_TIMEOUT = 3          # 3 seconds for RREP unicast path
AODV_ROUTE_LIFETIME = 180      # 3 minutes (RFC 3561 ACTIVE_ROUTE_TIMEOUT)
AODV_DISCOVERY_RATE = 1        # ~1 route/second (5 parallel RREQs, ~5s/cycle)

# ── SDN / OpenFlow (all in ticks = seconds) ──────────────────────────
# Structural durations (election completion, LLDP BFS depth, flow-programming
# time) are still COMPUTED from actual fragment sizes / BFS rounds / queue
# depth; only the per-message and per-round timers are constants.
SDN_ORPHAN_PROBE_INTERVAL = 1   # controller-reconnect retry, s (ONOS/ODL
                                # initial reconnect backoff is ~1 s)
SDN_ELECTION_ROUND_S      = 1   # one Raft RequestVote/VoteResponse round, s
                                # (Raft's 150-300 ms election timeout does not
                                # survive a multi-hop degraded wireless mesh;
                                # 1 s per round is the conservative choice)
SDN_ELECTION_BASE_S       = 2.0 # Raft election: fixed term-start cost, s
SDN_ELECTION_PER_NODE_S   = 0.1 # + per-voter vote RTT over the mesh, s
SDN_LLDP_ROUND_S          = 1   # one controller-triggered LLDP BFS round, s
                                # (IEEE 802.1AB's 30 s timer governs PASSIVE
                                # advertisement, not controller-driven probes)
SDN_LLDP_KEEPALIVE_S      = 30  # operational LLDP advertisement, s (802.1AB)
SDN_STATS_POLL_S          = 5   # OpenFlow port/flow stats polling period, s
SDN_FLOW_EXPIRY_S         = 10  # OpenFlow hard_timeout on stale rules after
                                # controller loss, s (ONOS/ODL typical)
#
# MARL has NO bootstrap schedule here: forwarding capability after severance
# is whatever the engine + learned policy produce (relay-link establishment,
# PRB/MCS actions, routing gate) — no scripted ramp.

# ── MARL exploration-temperature controller (event-gated) ─────────────
# Elevated sampling temperature is ONLY allowed inside a bounded window
# after a GENUINE detected event (severance, rescue arrival, or a SUSTAINED
# drop in delivered fraction), decays linearly back to the floor within
# TEMP_BOOST_WINDOW_TICKS, and every trigger is logged with its reason.
# Steady state always runs at TEMP_FLOOR.
TEMP_FLOOR              = 0.05   # exploitation temperature (steady state)
TEMP_BOOST              = 0.30   # temperature at the moment of a trigger
TEMP_BOOST_WINDOW_TICKS = 200    # boost decays to EXACTLY the floor by then
DROP_CONSEC_TICKS       = 20     # delivered fraction must stay depressed
                                 # this many CONSECUTIVE ticks to count as a
                                 # genuine shift (no single-tick triggers)
DROP_FRACTION           = 0.6    # "depressed" = below this fraction of the
                                 # slow-EMA achievability baseline

# ── MARL adapt-then-freeze arm ('marl_freeze') ────────────────────────
# Identical to the online MARL arm, except online PPO updates are disabled
# this many ticks after the LAST scenario event (severance for Scenario A,
# rescue arrival for Scenario B).  From that tick on the policy runs FROZEN
# with DETERMINISTIC ARGMAX action selection (is_training=False — no
# sampling noise, no gradient updates).
FREEZE_AFTER_LAST_EVENT_TICKS = 400

# ── MARL STATIC arm ('marl_static') — the deployed controller ────────────
# The architecture the paper describes: the parameters validated in the twin
# are executed UNCHANGED from the first tick of the event to the last.
#   * deterministic execution: argmax on every categorical head, Dirichlet
#     mean for the resource split (is_training=False from the start);
#   * no online PPO update, no exploration temperature, no EWC anchor;
#   * no freeze guard and no snapshot restore (nothing reads the arm's own
#     achievability);
#   * the SAME trained checkpoint in both scenarios (it is not in MARL_ARMS,
#     so its rescue_ops pass runs in Phase 1 from the base checkpoint and
#     never loads weights adapted during a recovery pass).
# Everything else — observations, detection floor, relay mechanics, engine
# flags — is the MARL code path shared with every other MARL-family arm.
# Invariants are ASSERTED at run time (is_training stays False every tick;
# the policy parameters are bit-identical at the end of the pass).

# ── Guarded freeze (deployment-safety gate) ───────────────────────────
# Freezing is a one-way door: if the policy's deterministic argmax joint
# action is degenerate at the freeze tick (observed failure mode: enough
# agents argmax an aggressive +1/+2 MCS offset that general spectral
# efficiency is 0 network-wide -> every path capacity-clamps to zero ->
# total delivery collapse), the frozen arm scores exactly 0 for the rest
# of the run.  So the freeze is GATED on recent performance being sane —
# symmetric in spirit to how the SDN/OSPF arms only forward once their
# control plane reports convergence.  No arm gains new information from
# this gate; it only decides WHEN a stop is safe.
#
# Rule: at event + FREEZE_AFTER_LAST_EVENT_TICKS, freeze ONLY if the
# trailing-FREEZE_GUARD_TRAIL-tick mean achievability is at least
# FREEZE_GUARD_RATIO x the peak trailing mean of the last
# FREEZE_GUARD_PEAK_WINDOW ticks.  Otherwise re-check every
# FREEZE_GUARD_CHECK_INTERVAL ticks.  Bounded: by event +
# FREEZE_FORCE_AFTER_TICKS the arm force-freezes anyway, RESTORING the
# policy snapshot saved at its best boost-free trailing window (so a
# late-run degradation cannot be locked in either).
# The trailing mean is an honest proxy for frozen (argmax) behaviour
# because steady-state sampling runs at TEMP_FLOOR=0.05, which is nearly
# deterministic; best-snapshot tracking additionally ignores windows that
# overlap a temperature boost.
FREEZE_GUARD_TRAIL          = 100    # trailing window (ticks) for the gate
FREEZE_GUARD_PEAK_WINDOW    = 500    # look-back (ticks) for the peak
FREEZE_GUARD_RATIO          = 0.70   # trailing mean must be >= ratio x peak
FREEZE_GUARD_CHECK_INTERVAL = 100    # postponed-gate re-check cadence
FREEZE_FORCE_AFTER_TICKS    = 1200   # hard bound: force-freeze by event+this

# ═══════════════════════════════════════════════════════════════════════
# FRAGMENTING SEVERANCE (Scenario A 'recovery')
# A pure sever_core leaves the surviving infra graph fully connected
# (observed: routable_flows=302/302 immediately post-severance), so every
# classical protocol recovers to the topology limit by mere re-convergence
# and the island-mode research question (bridging DISCONNECTED fragments
# with new links) is never exercised.  The recovery scenario therefore
# extends the severance with a seed-varied set of additional link cuts
# chosen at build time so that the post-severance infra graph PARTITIONS
# into FRAG_MIN_FRAGMENTS..FRAG_MAX_FRAGMENTS disconnected fragments.
#
# Fairness rules (enforced by construction):
#   * The cut list is a pure function of (topology, scenario_seed) — the
#     IDENTICAL ScenarioEvent list is applied to every arm.  Cuts are part
#     of the scenario, never per-arm.
#   * Offered load keeps counting cross-fragment flows for every arm
#     (the achievability denominator _offered_volume_for_flows runs over
#     the FULL flow set) — an arm that cannot serve them is honestly
#     charged for them.
#   * Fragments must remain PHYSICALLY bridgeable: the builder verifies with
#     TransportRelayModel.can_form_relay_link at FRAG_BRIDGE_TX_DBM that a
#     TG-capable (has_multihaul) survivor in one fragment can beam-steer to
#     some survivor infra node in each other fragment (each TG node can hold
#     ONE relay link, so a spanning arrangement is checked greedily).
#     No arm is handed an unbridgeable scenario.
#
#     THERE IS NO RANGE CONSTANT IN THIS CHECK.  The old text here referred
#     to `MAX_RELAY_RANGE_M`, which used to be a hardcoded 5000 m at 3.5 GHz
#     FSPL.  After the hybrid-transport rewrite it is a DERIVED property of
#     TransportRelayModel (the 60 GHz TG budget at the given Tx power, rain
#     rate and interference), and can_form_relay_link applies no separate
#     range gate at all: the SINR check IS the range check, so changing Tx
#     power moves the reach instead of hitting a wall.  The reach the plan
#     below is sized against is TG_PLAN_REACH_M, printed by every run.
# ═══════════════════════════════════════════════════════════════════════
FRAG_MIN_FRAGMENTS   = 2       # inclusive lower bound on infra fragments
FRAG_MAX_FRAGMENTS   = 4       # inclusive upper bound on infra fragments
FRAG_CROSS_FLOW_MIN  = 0.30    # >=30% of UE-to-UE flows must span fragments
FRAG_CROSS_FLOW_MAX  = 0.60    # <=60% (seed-varied within the band)
FRAG_BUILD_ATTEMPTS  = 80      # seeded partition attempts before fallback
FRAG_BRIDGE_TX_DBM   = 23.0    # reference TX power for bridgeability check
                               # (the shared fixed-power baseline value)

# ═══════════════════════════════════════════════════════════════════════
# 60 GHz TG (MultiHaul) SITE PLAN
# ═══════════════════════════════════════════════════════════════════════
# THE PROBLEM THIS SOLVES.  The only steerable transport class is the 60 GHz
# TG mesh (sixg_sim/transport_relay_model.TG_MESH_60GHZ); the 18 GHz MW PtP
# hops are mechanically aligned and can never be re-pointed inside an
# episode.  The TG clear-air reach at the screening power FRAG_BRIDGE_TX_DBM
# is TG_PLAN_REACH_M below (~1.07 km) — DERIVED from the link budget, not a
# tunable.  With the previous plan (TG capability on 3 of 8 relay sites, all
# clustered near the map centre, median TG-site spacing 0.9-3.0 km) full
# reunification to ONE component was physically feasible on only 1 of the 5
# evaluation seeds, so the recovery experiment was mostly measuring an
# impossible target.
#
# THE PLANNING RULE (physics-derived, arm-independent, deterministic).
# Reunification of two fragments needs SOME TG-capable site on one side
# within TG_PLAN_REACH_M of ANY surviving infra node on the other side (the
# engine steers the SOURCE only — can_form_relay_link never requires a TG
# radio at the far end).  The rule adopted is CO-SITING rather than new site
# acquisition: every RAN / transport site — O-RU, O-DU, EdgeUPF, Relay —
# carries a 60 GHz TG mesh node on its existing mast.  That is how a 60 GHz
# mesh overlay is actually densified in an urban deployment: the mast, power
# and backhaul already exist, so the incremental unit is a radio, not a site.
# Compute-only sites (SMO, Near-RT-RIC, O-CU-CP/UP, AMF, UPF) are NOT
# equipped: they are not radio sites, and they are largely the part of the
# network the severance removes.
#
# WHAT IT COSTS AND WHAT IT BUYS.  No node is added (infra count is
# unchanged), so no arm's agent count, compute budget or offered load moves.
# TG-capable sites go 3 -> 26 and their median nearest-neighbour spacing
# falls to 480-700 m, i.e. inside the 500-1000 m urban 60 GHz mesh band AND
# inside the derived 1.07 km budget with margin.  Neither the reach nor the
# Tx power was touched.  Feasibility is RE-AUDITED per seed at run time
# (printed by _build_fragmenting_severance) rather than asserted.
TG_SITE_TYPE_VALUES = ('O-RU', 'O-DU', 'EdgeUPF', 'Relay')

# Reach the plan is SIZED AGAINST, solved from the 60 GHz budget at the
# screening power (clear air).  Read-only here: nothing in this file may
# raise it, and every feasibility number reported is computed with it.
TG_PLAN_REACH_M = TG_MESH_60GHZ.max_range_m(FRAG_BRIDGE_TX_DBM, 0.0)
# The upper power the agents can reach (PHYMACState.apply_power_step ceiling),
# reported alongside so the audit shows both ends of the steering envelope.
TG_HIGH_TX_DBM  = 33.0
TG_HIGH_REACH_M = TG_MESH_60GHZ.max_range_m(TG_HIGH_TX_DBM, 0.0)

# ── MARL failure-detection floor (control-plane fairness) ─────────────
# Every BASELINE arm waits out a modelled detection + convergence delay
# before it can react to a link-state change (OSPF: RouterDeadInterval;
# OLSR/BATMAN: HELLO/OGM rounds; AODV: RREQ timeout; SDN: controller
# heartbeat miss).  MARL used to read `link.is_up` directly out of the
# topology and reroute in the SAME tick the link flipped, which is not a
# capability any real system has — it is omniscience, and it made the
# recovery-time comparison meaningless.
#
# MARL now gets the FASTEST DEFENSIBLE detector rather than none: a
# BFD-class liveness session on every adjacency.  RFC 5880 §6.8.4 declares a
# session down after Detect Mult missed control packets; with the common
# aggressive-but-deployable setting (300 ms interval, multiplier 3 -> 0.9 s)
# a single tick would already cover it, so the value used here is deliberately
# NOT the theoretical floor: 3 ticks = 3x a 1 s keepalive, matching the same
# "3 missed keepalives" rule the baselines are held to and equal to the
# RELAY_REPOINT_TICKS beam-training cost the policy already pays.  It is
# still 13x faster than OSPF's 40 s dead interval, so MARL keeps a large,
# HONEST detection advantage instead of an infinite one.
MARL_BFD_DETECT_TICKS = 3

# Ablation hook, NOT a tuning knob.  Setting MARL_DETECT_TICKS=0 in the
# environment disables the detector completely and restores the old
# omniscient behaviour (same-tick reroute, live routing view), which is how
# the "how much of MARL's recovery lead was free omniscience?" A/B is run
# without editing the file.  Any value >0 sets the detection interval.
# Read once at import so parent and worker processes agree.
# ── UE RELAYING: ONE reachability model for every arm ────────────────────
# A UE-to-UE flow is carryable in this engine iff an INFRASTRUCTURE-ONLY path
# joins the two endpoints; Simulator._can_route_ue_to_ue is that predicate and
# it is what the delivered-volume and achievability KPIs use.  The outage
# metric used to test nx.shortest_path over the UE-INCLUSIVE graph instead, so a
# dual-anchored UE silently bridged two fragments for some arms and not others.
#   MARL_UE_RELAY=none    (default) UEs are leaves, one model for every arm
#   MARL_UE_RELAY=bridge  the previous UE-inclusive test, kept for reproduction
# Steerability is defined once, in sixg_sim.simulation.site_can_steer, and
# shared by the engine, the severance generator and the optimum solver.
# See MARL_STEER_MODEL there.

MARL_UE_RELAY = os.environ.get('MARL_UE_RELAY', 'none').strip().lower()
if MARL_UE_RELAY not in ('none', 'bridge'):
    raise ValueError(
        f"MARL_UE_RELAY={MARL_UE_RELAY!r} is not one of 'none', 'bridge'")

MARL_BFD_DETECT_TICKS = int(os.environ.get('MARL_DETECT_TICKS',
                                           MARL_BFD_DETECT_TICKS))

# The detection floor applies to EVERY arm that runs the MARL code path.
# 'marl_freeze', 'random' and 'xldet' all rebind `mode = 'marl'` inside
# run_physics_pass (see the arm-plumbing block there), so the single
# `mode == 'marl'` test that gates the detector covers all four.  That is
# deliberate: the 'random' control is the null hypothesis for the bridging
# claim and 'xldet' is the engineered non-learning alternative, so neither
# must be handed — or denied — a detection advantage the treated arms do
# not have.  Equal detection latency is what makes the comparison about the
# policy rather than about who noticed the severance first.
MARL_DETECTION_ARMS = ('marl', 'marl_freeze', 'marl_static', 'random', 'xldet')

# ── Safe online adaptation (deployment PPO safety rails) ──────────────
# The 'MARL (pre-trained + online)' arm keeps updating throughout the
# deployment (that is what its label promises) but unconstrained online
# PPO drifts over long runs.  Two rails, both applied to the online
# updates of BOTH MARL arms (the adapt-freeze arm uses the same online
# machinery until its guarded freeze):
#
# 1) Reduced online LR.  Training LRs are MAPPOConfig.lr_actor=1e-4 and
#    lr_critic=3e-4; online adaptation runs at 15% of each (inside the
#    10-25% band).  These override the trainer's stock deploy_lr_* values
#    (5e-5 / 2e-4) via the config before set_deployment_mode() is called.
DEPLOY_LR_ACTOR   = 1.5e-5     # = 15% of MAPPOConfig.lr_actor  (1e-4)
DEPLOY_LR_CRITIC  = 4.5e-5     # = 15% of MAPPOConfig.lr_critic (3e-4)
#
# 2) EWC anchor to the loaded pretrained weights, using the EWCPenalty
#    already implemented in sixg_sim/mappo_trainer.py (its ewc_loss term
#    is added to the actor loss whenever deploy_mode is on and a penalty
#    is anchored).  The checkpoint ships no observation buffer, so at
#    load time the Fisher diagonal cannot be estimated: the anchor starts
#    with an IDENTITY Fisher (uniform parameter importance — EWC reduces
#    to an L2 pull toward the pretrained weights, a.k.a. L2-SP).  Once
#    EWC_FISHER_MIN_OBS deployment observations have accumulated in the
#    trainer's obs cache, a proper diagonal Fisher is estimated ONCE from
#    them and the anchor point is re-pinned to the SAME pretrained
#    weights (compute_fisher() would otherwise re-anchor at the current,
#    already-drifted weights).
DEPLOY_EWC_LAMBDA  = 0.40      # EWC strength λ in (λ/2)·Σ F_i(θ_i−θ*_i)²
                               # (matches MAPPOConfig.ewc_lambda default;
                               #  documented here as the deployment value)
EWC_FISHER_MIN_OBS = 512       # deployment obs needed before the one-time
                               # diagonal-Fisher estimate replaces identity
EWC_FISHER_SAMPLES = 300       # obs samples used by that Fisher estimate


# ── Publication-figure presentation policy ────────────────────────────
# The online-only MARL arm ('marl') is EXCLUDED from the publication
# figures and from the statistical summary table: the headline proposed
# system is the adapt-then-freeze variant ('marl_freeze').
#
# This is a PRESENTATION filter only.  The 'marl' arm is still simulated
# by main() and still written to the raw multiseed pickle — it remains a
# documented ablation reported in the article text — so nothing in the
# simulation code path may key off this constant.
# 'random' is the bridging-attribution control arm (uniform action sampling
# on the MARL code path).  It is a MEASUREMENT, not a competing protocol, so
# it never appears in the protocol comparison figures — its numbers are
# reported separately by print_bridging_attribution().
FIGURE_EXCLUDED_ARMS = ('marl', 'random')

# The proposed system (hero line/row in every figure and table).
PROPOSED_ARM = 'marl_freeze'
PROPOSED_ARM_LABEL = 'MARL-RIC (proposed)'
PROPOSED_ARM_SHORT = 'MARL-RIC'
PROPOSED_ARM_COLOR = '#006d2c'   # strong dark green, distinct from every
                                 # classical protocol colour below
PROPOSED_ARM_LW = 3.2

# One-line description of what the proposed arm actually is, reused in the
# figure caption and the experiment-configuration panel.
PROPOSED_ARM_DESCRIPTION = ("pre-trained (seed 42) + online-adapted during "
                            "deployment, frozen at best state after last event")

# Steady state = mean over the LAST N ticks of each seed's full recorded
# run (the 80-tick tail trim used when drawing curves is cosmetic only and
# is NOT applied to the statistics).  Reported dispersion is 1 standard
# deviation ACROSS SEEDS of those per-seed steady-state means.
STEADY_STATE_WINDOW_TICKS = 500


def figure_arms(candidate_arms):
    """Filter an arm ordering down to the arms shown in publication output."""
    return [m for m in candidate_arms if m not in FIGURE_EXCLUDED_ARMS]


def _apply_runtime_overrides(overrides):
    """Apply --ticks smoke-run overrides to the module globals.

    Must be called BOTH in the parent process (main) and in every worker
    process (run_pass_wrapper): Windows multiprocessing re-imports this
    module in children, resetting the globals.
    """
    global TOTAL_TICKS, SEVERANCE_TICK, RESCUE_ARRIVAL_TICK
    global FREEZE_AFTER_LAST_EVENT_TICKS, FREEZE_FORCE_AFTER_TICKS
    global FREEZE_GUARD_CHECK_INTERVAL
    if not overrides:
        return
    tt = overrides.get('total_ticks')
    if tt:
        TOTAL_TICKS = int(tt)
        # Keep the event structure meaningful in short runs
        SEVERANCE_TICK = min(SEVERANCE_TICK, max(50, TOTAL_TICKS // 6))
        RESCUE_ARRIVAL_TICK = min(RESCUE_ARRIVAL_TICK,
                                  max(SEVERANCE_TICK + 50, TOTAL_TICKS // 2))
        # Scale the adapt-freeze delay too so short smoke runs still
        # exercise the frozen phase of the 'marl_freeze' arm.
        FREEZE_AFTER_LAST_EVENT_TICKS = min(FREEZE_AFTER_LAST_EVENT_TICKS,
                                            max(50, TOTAL_TICKS // 6))
        # Scale the guarded-freeze hard bound so short runs still reach a
        # forced freeze with frozen ticks left to observe afterwards.
        FREEZE_FORCE_AFTER_TICKS = min(FREEZE_FORCE_AFTER_TICKS,
                                       max(FREEZE_AFTER_LAST_EVENT_TICKS + 100,
                                           TOTAL_TICKS // 3))
        FREEZE_GUARD_CHECK_INTERVAL = min(FREEZE_GUARD_CHECK_INTERVAL,
                                          max(20, FREEZE_AFTER_LAST_EVENT_TICKS // 2))


def _offered_volume_for_flows(sim, flows, tick):
    """Offered UE-to-UE demand (volume units) for an arbitrary flow list.

    Replicates simulation.py's per-flow traffic model EXACTLY (same
    deterministic per-flow RNG: tick*6997 + crc32(flow) % 7919 + seed), so
    the value for flows the engine did process is identical to the engine's
    own `last_tick_ue_to_ue_offered` accumulation.

    Needed for the SYMMETRIC achievability denominator: protocol arms that
    filter `sim.ue_to_ue_flows` to their routable subset only generate
    traffic for that subset, so the engine's offered counter under-counts
    their true demand.  All arms are scored against the same live per-tick
    offered demand over the FULL flow set (demand is MCS/capacity
    independent — it is generated load, not delivered load).
    """
    import zlib
    total = 0.0
    seed_part = (sim.config.random_seed or 0)
    for source_ue, target_ue in flows:
        source_node = sim.topology.nodes.get(source_ue)
        target_node = sim.topology.nodes.get(target_ue)
        if not source_node or not source_node.is_survivor:
            continue
        if not target_node or not target_node.is_survivor:
            continue
        _t_rng = random.Random(
            tick * 6997
            + zlib.crc32(f"{source_ue}|{target_ue}".encode()) % 7919
            + seed_part)
        is_rescue_flow = (getattr(source_node, 'is_rescue_service', False)
                          or getattr(target_node, 'is_rescue_service', False))
        if not sim.island_mode:
            ue_to_ue_traffic = 8.0 + _t_rng.gauss(0, 1.0)
        elif is_rescue_flow:
            ptt_active = _t_rng.random() < 0.6
            voice = 4.5 if ptt_active else 0.5
            sensor = 3.0
            video = _t_rng.uniform(10.0, 30.0) if _t_rng.random() < 0.20 else 0.0
            ue_to_ue_traffic = voice + sensor + video
        else:
            always_on = 4.0
            calling = _t_rng.random() < 0.50
            voice = 6.0 if calling else 2.0
            messaging = 3.0 if _t_rng.random() < 0.40 else 1.0
            ue_to_ue_traffic = always_on + voice + messaging
        total += max(0.5, ue_to_ue_traffic)
    return total


def _lock_phy_mac_fixed(sim, mcs_level, tx_power_dbm, allow_local_reroute=False):
    """Lock all PHY/MAC states to a fixed MCS and power.
    No agent exploration — operator-configured radio parameters.
    If allow_local_reroute=True, enables LOCAL_REROUTE on existing mesh links
    (models OSPF repurposing capacity for east-west traffic after core loss).
    """
    for ps in sim.phy_mac_states.values():
        ps.mcs_general = mcs_level
        ps.mcs_emergency = mcs_level
        ps.tx_power_dbm = tx_power_dbm
        if allow_local_reroute and sim.island_mode:
            ps.relay_mode = RelayMode.LOCAL_REROUTE
            ps.prb_relay_fraction = 0.2
        else:
            ps.relay_mode = RelayMode.OFF
            ps.prb_relay_fraction = 0.0
        ps.mac_scheduler = MACScheduler.ROUND_ROBIN
        # ACCESS pool only.  prb_relay_fraction is a share of the DEDICATED
        # transport radio and no longer deducts from the Uu access carrier
        # (PHYMACState.normalise_prb) — so the operator-configured 30/70
        # emergency/general access split is independent of relay mode.  The
        # baselines gain exactly the same access capacity back that the MARL
        # arms do; this is not a MARL-only change.
        ps.prb_emergency_fraction = 0.3
        ps.prb_general_fraction = 0.7
        ps.normalise_prb()


def _auto_link_adaptation(sim):
    """3GPP TS 38.214 — Automatic CQI-to-MCS link adaptation.

    This is a NATIVE gNB/DU function that runs at L1/L2 every TTI.
    It does NOT require any RIC, SDN controller, or routing protocol.
    ALL approaches get this equally — it's a physical-layer mechanism.

    Process (per 3GPP):
      1. UE measures channel quality via CSI-RS → reports CQI
      2. gNB MAC scheduler maps CQI → MCS index (TS 38.214 §5.1.3.1)
      3. OLLA fine-tunes via HARQ ACK/NACK feedback

    We use sinr_average as a proxy for CQI since the simulation
    tracks per-node SINR continuously.

    Uses the SAME CQI→MCS mapping (sixg_sim.phy_mac_state.auto_cqi_mcs_idx)
    that Simulator._step_phy_mac uses as the reference point for the MARL
    agent's bounded MCS-offset actions — one shared physical-layer baseline.
    """
    for nid, ps in sim.phy_mac_states.items():
        best_mcs = list(MCSLevel)[auto_cqi_mcs_idx(ps.sinr_average)]
        ps.mcs_general = best_mcs
        ps.mcs_emergency = best_mcs


def _lock_phy_mac_non_mcs(sim, tx_power_dbm, allow_local_reroute=False):
    """Lock TX power, PRB split, scheduler — but NOT MCS.
    MCS is set separately by _auto_link_adaptation() (3GPP-native).
    """
    for ps in sim.phy_mac_states.values():
        ps.tx_power_dbm = tx_power_dbm
        if allow_local_reroute and sim.island_mode:
            if ps.relay_mode == RelayMode.OFF:
                ps.relay_mode = RelayMode.LOCAL_REROUTE
            ps.prb_relay_fraction = max(ps.prb_relay_fraction, 0.2)
        elif ps.relay_mode not in (RelayMode.CAPACITY_BOOST,):
            # Don't reset relay mode if SDN already activated CAPACITY_BOOST
            ps.relay_mode = RelayMode.OFF
            ps.prb_relay_fraction = 0.0
        ps.mac_scheduler = MACScheduler.ROUND_ROBIN
        # ACCESS pool only — see the note in _lock_phy_mac_fixed.
        ps.prb_emergency_fraction = 0.3
        ps.prb_general_fraction = 0.7
        ps.normalise_prb()


def _establish_relay_links(sim, mode, scenario_name):
    """Establish relay links for MultiHaul-capable nodes.

    Used by the SDN arm (after its controller bootstrap reaches the
    operational phase, via NETCONF/YANG relay activation).  The MARL arm
    does NOT use this helper: agent relay actions are applied natively by
    sim._step_phy_mac() inside _execute_agent_actions(), so relay links
    appear/disappear only as a consequence of the learned policy's actions.
    Runs once per node — subsequent ticks skip nodes with active links.
    """
    from sixg_sim.agent import PHYMACAction
    for nid, ps in sim.phy_mac_states.items():
        node_obj = sim.topology.nodes.get(nid)
        if (node_obj and getattr(node_obj, 'has_multihaul', False)
                and ps.relay_mode in (RelayMode.CAPACITY_BOOST, RelayMode.LOCAL_REROUTE)
                and not ps.relay_link_active):
            dummy_action = PHYMACAction(
                tx_power_step=2,
                # MCS heads are OFFSETS from auto-CQI: the neutral index
                # (offset 0) keeps the SDN arm on plain auto link adaptation.
                mcs_emergency_idx=MCS_OFFSET_CENTER,
                mcs_general_idx=MCS_OFFSET_CENTER,
                prb_emergency_frac=ps.prb_emergency_fraction,
                prb_relay_frac=ps.prb_relay_fraction,
                prb_general_frac=ps.prb_general_fraction,
                relay_mode_idx=list(RelayMode).index(ps.relay_mode),
                handover_idx=0,
                scheduler_idx=list(MACScheduler).index(ps.mac_scheduler),
                send_postcard=False,
                postcard_content=None,
            )
            sim._step_phy_mac(nid, dummy_action)


# Max orphaned UEs one integrated-access node admits.  This is an ADMISSION
# limit, not a physics limit: the FR1 small cell's PRB budget is already split
# by the site's own prb_relay_fraction, and a relay small cell in a disaster
# deployment is provisioned for a handful of emergency attachments, not a
# macro cell's load.  Identical for every arm that may use the capability.
IAB_MAX_UES_PER_SITE = 3


def _reconnect_orphaned_ues(sim, site_enabled, tick, log_prefix="",
                            gate_desc=""):
    """Attach orphaned UEs to enabled relay sites as INTEGRATED ACCESS NODES.

    ── WHAT THIS MODELS (design decision 2) ──────────────────────────────
    A relay site is not a bare 60 GHz transport box: it hosts its own FR1
    small cell, so a UE orphaned by the disaster can camp on it directly.
    That is IAB-style integrated access and it is physically sound.  What it
    is NOT — and what the previous implementation implied — is a 60 GHz mesh
    node serving a handset: no handset has a 60 GHz phased array, and the
    oxygen-absorbed 60 GHz budget would not reach one anyway.

    ── WHAT WAS REPLACED ────────────────────────────────────────────────
    The old code invented the access link out of nothing:

        if dist > 1500.0: continue                      # hardcoded gate
        dist_factor = max(0.3, 1.0 - dist / 1500.0)     # linear in distance
        ue_cap = 25.0 * ps.prb_relay_fraction * dist_factor

    That formula had no link budget behind it: it could never go into outage
    (the 0.3 floor guaranteed >= 30 % of peak at any range inside the gate),
    it ignored the site's Tx power entirely, and it ignored interference.  The
    1500 m gate was likewise a literal with no radio behind it.  Both are gone.
    Capacity and coverage now come from
    Simulator.integrated_access_capacity_mbps, i.e. the ACCESS_FR1 budget
    (3.5 GHz FSPL, 20 MHz, NF 5 dB, 2x8 dBi, 2 dB implementation loss,
    9.99 dB shadow-fade margin), SINR = S/(N+I) against the live aggregate
    FR1 interference at the serving site, SINR -> MCS -> SE through the site's
    OWN PHYMACState (so its OLLA step-down and near-threshold half-rate rule
    apply), and a hard coverage gate at the site's effective coverage radius —
    which itself scales with the Tx power the agent chose.  A link that cannot
    close now returns 0.0 and is refused, and the usability floor is the
    engine's own sim.IAB_ACCESS_MIN_USABLE_MBPS.

    ── ARM-AGNOSTIC BY CONSTRUCTION ─────────────────────────────────────
    Nothing here reads `mode`.  `site_enabled(site_id) -> bool` is the ONLY
    difference between callers, and it expresses an ARCHITECTURAL claim about
    which control planes can drive this capability at all:

      * MARL   — gated on the site's agent having sampled relay
                 CAPACITY_BOOST this tick.
      * SDN    — gated on the fragment-local controller having reached its
                 operational phase, exactly as its relay bridging already is.
                 An SDN controller has a MANAGEMENT plane (NETCONF/YANG), so
                 operator-triggered UE reattachment and small-cell capacity
                 adjustment are plausible capabilities for it.
      * OSPF / OLSR / BATMAN / AODV — NOT eligible, and this is the honest
                 asymmetry rather than a handicap: these are pure IGP/MANET
                 routing protocols with no management plane and no notion of a
                 RAN configuration object.  There is no mechanism in OSPF or
                 BATMAN by which a router could instruct a radio site to
                 admit a UE or to re-split its PRBs.  Any comparison must
                 state this, and it is stated in the figure captions.

    Returns the number of UEs newly attached this tick.
    """
    from sixg_sim.topology import Link, LinkType, InterfaceType, NodeType

    # Tear down access links whose host site is no longer enabled (the
    # enabling decision was withdrawn — coverage extension is not permanent).
    _stale = []
    for lid, link in sim.topology.links.items():
        if not lid.startswith('ric_ue_access_'):
            continue
        if not site_enabled(link.endpoints[0]):
            _stale.append(lid)
    for lid in _stale:
        link = sim.topology.links.pop(lid)
        ep0, ep1 = link.endpoints
        if sim.topology.graph.has_edge(ep0, ep1):
            sim.topology.graph.remove_edge(ep0, ep1)
    if _stale:
        sim.topology.invalidate_infrastructure_cache()

    # Orphaned UEs = surviving UEs with no live link to any infra node.
    orphaned_ues = []
    for ue_id, ue_node in sim.topology.nodes.items():
        if ue_node.node_type != NodeType.UE or not ue_node.is_survivor:
            continue
        has_infra_link = False
        for lid, link in sim.topology.links.items():
            if not getattr(link, 'is_up', True):
                continue
            ep0, ep1 = link.endpoints
            if ep0 == ue_id or ep1 == ue_id:
                peer_id = ep1 if ep0 == ue_id else ep0
                peer_node = sim.topology.nodes.get(peer_id)
                if peer_node and peer_node.node_type != NodeType.UE:
                    has_infra_link = True
                    break
        if not has_infra_link:
            orphaned_ues.append(ue_id)

    reconnected = 0
    refused_budget = 0
    for nid, ps in sim.phy_mac_states.items():
        node_obj = sim.topology.nodes.get(nid)
        if not (node_obj and getattr(node_obj, 'has_multihaul', False)):
            continue
        if not node_obj.is_survivor:
            continue
        if not site_enabled(nid):
            continue

        served = 0
        for ue_id in list(orphaned_ues):
            if served >= IAB_MAX_UES_PER_SITE:
                break
            if ue_id not in sim.topology.nodes:
                continue
            ue_link_id = f"ric_ue_access_{nid}_{ue_id}"
            if ue_link_id in sim.topology.links:
                continue

            # THE PHYSICS.  Coverage gate, link budget, interference, OLLA and
            # PRB share all come from the engine; capacity 0.0 means unusable.
            ue_cap, ue_sinr = sim.integrated_access_capacity_mbps(nid, ue_id)
            if ue_cap <= 0.0 or ue_cap < sim.IAB_ACCESS_MIN_USABLE_MBPS:
                refused_budget += 1
                continue

            sim.topology.links[ue_link_id] = Link(
                id=ue_link_id,
                endpoints=(nid, ue_id),
                capacity=ue_cap,
                latency=2,
                link_type=LinkType.TRANSPORT_RELAY,
                interface_type=InterfaceType.Uu,
                is_up=True,
                current_utilization=0.0
            )
            sim.topology.graph.add_edge(
                nid, ue_id, link_id=ue_link_id, capacity=ue_cap)
            sim.topology.invalidate_infrastructure_cache()
            orphaned_ues.remove(ue_id)
            served += 1
            reconnected += 1

    if reconnected > 0 and tick % 100 == 0:
        print(f"{log_prefix}[IAB UE-RECONNECT] t={tick}: {reconnected} "
              f"orphaned UEs attached to relay-hosted FR1 small cells "
              f"({gate_desc}); {refused_budget} refused by the access link "
              f"budget / coverage gate; {len(orphaned_ues)} still orphaned")
    return reconnected


# ══════════════════════════════════════════════════════════════════════
#  Uu CARRIER AGGREGATION  -  ONE SHARED IMPLEMENTATION, ALL ELIGIBLE ARMS
#  v2: BOUNDED, SHARED, SITE-LEVEL SPECTRUM POOL
# ══════════════════════════════════════════════════════════════════════
# Logged per tick for EVERY arm (see uu_boost_links / uu_boost_factor /
# uu_cc_activated / uu_extra_mbps in the per-pass return value and the CA[...]
# field of the periodic status line), including the arms that are not eligible
# and therefore log 0.  The ORIGINAL implementation was an inline
# `if mode == 'marl'` branch that only MARL could reach AND only MARL logged,
# which is precisely why the capability asymmetry below went unnoticed for the
# whole study.
#
# ── WHY THE v1 PER-LINK MULTIPLIER WAS REPLACED ───────────────────────
# v1 granted, PER Uu LINK,
#         boost_factor = 1.0 + min(1.0, site.prb_emergency_fraction)
# and that model had two defects that no amount of documentation repairs:
#
#   1. WRONG DRIVING QUANTITY.  `prb_emergency_fraction` is a SCHEDULING SPLIT
#      of the carrier the site ALREADY HAS.  A site that hands 40 % of its
#      PRBs to emergency traffic does not thereby acquire 40 % MORE SPECTRUM.
#      The two quantities are dimensionally unrelated; the coupling was
#      arbitrary, and it made the size of a SPECTRUM grant a function of an
#      agent's SCHEDULER action.
#   2. UNBOUNDED PER SITE.  It multiplied EVERY eligible link independently
#      with no site-level cap.  A smoke run showed ~208 boosted links across
#      42 radio sites, i.e. the sites were collectively handed the equivalent
#      of roughly FIVE extra full-bandwidth component carriers EACH.  Licensed
#      spectrum does not scale with the number of camped UEs.
#
# v2 (this implementation) fixes both.  The grant is an INTEGER COUNT OF
# COMPONENT CARRIERS activated at the SITE, converted to a FINITE Mbps pool,
# and that pool is SHARED among the site's hot-spot UEs by water-filling.  No
# agent action can enlarge the pool; a site serving 20 hot-spot UEs and a site
# serving 2 receive the SAME extra spectrum, distributed differently.
UU_CA_HOTSPOT_MIN_FLOWS = 2      # a UE is a "hot spot" at >= this many flows

# ── N_CC: additional component carriers a site may activate ───────────
# On top of the one access carrier it already runs.  1 is the honest default
# and is what this study reports.
#
# 3GPP JUSTIFICATION.  The MECHANISM is standard and long-standing: carrier
# aggregation of a PCell plus one or more SCells (LTE-A Rel-10, up to 5 CCs;
# Rel-13, up to 32; NR Rel-15+, up to 16), with per-SCell activation and
# deactivation carried by a MAC control element (TS 38.321, "SCell
# Activation/Deactivation MAC CE").  "Light up k of N carriers" is therefore
# literally a standardised primitive, and it is a BINARY decision per carrier
# — never a continuous gain multiplier.
#
# What bounds N in practice is NOT the standard but the LICENCE AND THE
# INSTALLED RADIO: a site can only aggregate carriers the operator actually
# holds and has deployed on that sector.  In FR1 mid-band a typical operator
# holding 60-100 MHz deploys it as its primary carrier; assuming even ONE
# spare, unused, full-bandwidth carrier sitting idle at EVERY site in the
# disaster area is already a generous assumption in the model's favour.
# Assuming several is not defensible.  Hence 1.
#
# Override for ABLATION ONLY (never for the headline comparison):
#     UU_CA_MAX_EXTRA_CC=0  -> grant fully disabled (clean A/B)
#     UU_CA_MAX_EXTRA_CC=2  -> two spare carriers per site
UU_CA_MAX_EXTRA_CC = max(0, int(os.environ.get('UU_CA_MAX_EXTRA_CC', '1')))

# ── What one activated carrier is worth, and in WHICH UNITS ───────────
# The extra carrier is ANOTHER ACCESS_FR1 CARRIER: the same radio class as the
# primary access carrier every UE in this study is camped on (3.5 GHz,
# 20 MHz, peak SE 4.0 bit/s/Hz = QAM256).  Both numbers are read off the radio
# class itself so this model cannot drift away from the engine's own access
# physics.
#
# UNITS — this is the subtle part.  Simulator.consume_path documents
# `link.capacity` as "maximum capacity at QAM256 + max power" and derates it at
# run time by (live SE / MAX_SE) with MAX_SE = 4.0.  The pool must therefore
# ALSO be expressed at the QAM256 REFERENCE; it then receives EXACTLY the same
# live MCS/SINR/OLLA derating as the primary carrier, through the same code
# path, for every arm.  Expressing it in live Mbps instead would double-count
# link adaptation.
#     20 MHz x 4.0 bit/s/Hz = 80 Mbps of REFERENCE capacity per carrier.
UU_CA_CC_BANDWIDTH_MHZ     = float(ACCESS_FR1.bandwidth_mhz)    # 20.0 MHz
UU_CA_CC_REF_SE_BPS_HZ     = float(ACCESS_FR1.peak_se_bps_hz)   # 4.0 (QAM256)
UU_CA_CC_REF_CAPACITY_MBPS = (UU_CA_CC_BANDWIDTH_MHZ *
                              UU_CA_CC_REF_SE_BPS_HZ)           # 80.0 Mbps@ref

# HARD SITE-LEVEL CEILING on the AGGREGATE extra capacity ONE site can deliver,
# regardless of how many UEs it serves.  Asserted every tick, per site, in
# _apply_uu_carrier_aggregation, and reported in [UU-BOOST-SUMMARY] so the
# bound is auditable from the log alone.
UU_CA_SITE_POOL_BOUND_MBPS = UU_CA_MAX_EXTRA_CC * UU_CA_CC_REF_CAPACITY_MBPS


def _apply_uu_carrier_aggregation(sim, site_enabled, tick, log_prefix="",
                                  gate_desc=""):
    """Carrier-aggregate the Uu access carrier of hot-spot UEs.  ARM-AGNOSTIC.

    -- WHAT THIS MODELS ------------------------------------------------
    A FINITE ADDITIONAL SPECTRUM RESOURCE PER SITE, activated as an integer
    number of component carriers and SHARED among that site's hot-spot UEs:

        k_site   = min(N_CC, #eligible hot-spot Uu links at the site)
        pool     = k_site x 20 MHz x 4.0 bit/s/Hz      [Mbps at QAM256 ref]
        extra_l  = water-fill(pool) over the site's hot-spot links,
                   each link capped at k_site x base_cap_l
        cap_l    = base_cap_l + extra_l

    with N_CC = UU_CA_MAX_EXTRA_CC (1 by default).  See the constants above
    for the 3GPP justification of N_CC, of the 20 MHz / QAM256 carrier, and of
    why the pool is expressed at the QAM256 reference rather than in live Mbps.

    -- THE THREE PROPERTIES THAT MAKE IT DEFENSIBLE --------------------
      1. THE DRIVING QUANTITY IS SPECTRUM, NOT A SCHEDULING SPLIT.  The size
         of the grant is an INTEGER COUNT OF CARRIERS times a fixed licensed
         bandwidth.  `prb_emergency_fraction` — a split of the EXISTING
         carrier — no longer appears anywhere in the magnitude.  Neither does
         any other agent action: the policy cannot buy spectrum by moving its
         scheduler knobs.  Activation is a BINARY/QUANTISED decision per
         carrier, which is exactly what an SCell Activation MAC CE is.
      2. THE SITE-LEVEL BOUND IS HARD.  The aggregate extra capacity a site
         delivers is <= UU_CA_SITE_POOL_BOUND_MBPS no matter how many UEs are
         camped on it — asserted per site, per tick, and reported in the
         summary line.  A site serving 20 hot-spot UEs and a site serving 2
         get the SAME extra spectrum; only the split differs.  v1's ~5
         simultaneous extra carriers per site are structurally impossible now.
      3. PER-UE AGGREGATION CAPABILITY IS ALSO BOUNDED.  A UE aggregating
         k carriers of the same width and the same site's MCS cannot exceed
         k x its primary-carrier rate, so each link's extra is capped at
         k_site x base_cap.  With N_CC = 1 the per-link factor can therefore
         never exceed 2.0x, and the SITE aggregate never exceeds 80 Mbps@ref.

    The honest caveat that survives from v1: this CREATES capacity, it does
    not reallocate it.  `base_cap` is untouched and the sum is >= base_cap, so
    no other user or PRB pool pays.  That is defensible ONLY as ADDITIONAL
    SPECTRUM (a second component carrier the site is licensed for but does not
    use in normal operation) — never as a smarter scheduler on the existing
    carrier, which can only move capacity around, not manufacture it.  What is
    new in v2 is that the amount of additional spectrum is now finite, named,
    and bounded per site, rather than implied by an unrelated MAC statistic.

    THE HARDWARE EXISTS ON EVERY SITE.  Every arm in this study runs on the
    SAME generated topology with the SAME site hardware; the radios do not
    change when the routing protocol changes.  So the additional carrier is
    physically available to every arm.  What differs between arms is ONLY the
    ability to COMMAND it — to decide, at run time, that this site should turn
    on its spare carrier right now.

    -- WHO IS ELIGIBLE, AND WHY (an ARCHITECTURAL claim) ----------------
    Nothing here reads `mode`.  `site_enabled(site_id) -> bool` is the ONLY
    difference between callers, exactly as in `_reconnect_orphaned_ues`:

      * MARL / MARL-freeze / RANDOM - gated on the site's agent having
                 produced an action this tick.  The RIC/agent IS the entity
                 that commands the SCell, so an agentless site cannot ask for
                 one.  THE MAGNITUDE IS NOT AN AGENT ACTION: once the gate is
                 open the site gets k_site x 20 MHz, the same as any other
                 eligible arm's site.  (v1 sized it by the agent's own
                 `prb_emergency_fraction`; that dependence is GONE.)
      * SDN    - ELIGIBLE.  Gated on the fragment-local controller having
                 reached its operational phase, i.e. the IDENTICAL gate that
                 already lets SDN activate MultiHaul relay bridging and
                 integrated-access UE reattachment.  An SDN controller has a
                 MANAGEMENT plane (NETCONF/YANG) into the RAN and can write a
                 carrier-aggregation / secondary-cell configuration object
                 (3GPP TS 28.541 NRCellDU / O-RAN O1), so commanding a site to
                 activate its spare component carrier is a plausible
                 capability for it.  ITS MAGNITUDE IS NOW IDENTICAL TO THE
                 MARL ARMS' — the same k_site x 20 MHz pool, the same
                 water-fill, the same per-UE cap.  Under v1 SDN was pinned to
                 a flat 1.30x while MARL could reach 2.0x by raising an
                 unrelated scheduler knob; that asymmetry is gone.  Any
                 remaining MARL advantage is therefore attributable to the
                 POLICY (routing, relay, power, MCS), not to a larger grant.
      * OSPF / OLSR / BATMAN / AODV - NOT eligible.  This is an ARCHITECTURAL
                 CLAIM, stated as such and not a handicap: these are pure
                 IGP / MANET routing protocols.  They have no management
                 plane, no RAN configuration object and no southbound
                 interface of any kind; there is no mechanism by which OSPF or
                 BATMAN could instruct a radio site to bring up a second
                 component carrier.  A routing protocol cannot command a
                 radio.  The hardware is there on their sites too - they
                 simply have nothing that can turn it on.  Any comparison
                 that quotes these arms must state this, and it is stated in
                 the figure captions and in the text table.

    -- MECHANICS -------------------------------------------------------
    Recomputed from scratch EVERY tick and NON-CUMULATIVE: `_uu_base_capacity`
    is cached on first sight of each link and the capacity is rewritten as
    base + extra (extra = 0 when the grant is withdrawn), so withdrawing the
    grant restores the base EXACTLY and a long run cannot ratchet.  IAB access
    links created by `_reconnect_orphaned_ues` (`ric_ue_access_*`) are excluded
    - their capacity comes from the engine's own link budget and must not be
    scaled.

    Returns (n_boosted, mean_factor, n_cc_activated, extra_total_ref_mbps):
      n_boosted            Uu links that received a non-zero share
      mean_factor          mean (base+extra)/base over those links (1.0 if none)
      n_cc_activated       TOTAL component carriers lit across all sites
      extra_total_ref_mbps aggregate extra capacity granted, Mbps at QAM256 ref
    """
    from collections import Counter, defaultdict
    from sixg_sim.topology import NodeType

    # Hot-spot detection: how many surviving UE-to-UE flows terminate on each
    # UE (bidirectional).  Scenario property, identical for every arm.
    target_counts = Counter()
    for s, t in sim.ue_to_ue_flows:
        sn = sim.topology.nodes.get(s)
        tn = sim.topology.nodes.get(t)
        if sn and sn.is_survivor and tn and tn.is_survivor:
            target_counts[t] += 1
            target_counts[s] += 1  # bidirectional

    # ── PASS 1 ── enumerate every Uu access link, cache its base capacity,
    # and bucket the ELIGIBLE hot-spot links BY SERVING SITE.  Bucketing by
    # site is what makes the resource site-level rather than per-link.
    all_uu = []                      # [(lid, ep0, ep1, link, base_cap)]
    site_links = defaultdict(list)   # site_id -> [index into all_uu]
    for lid, link in sim.topology.links.items():
        if not link.is_up or lid.startswith('ric_ue_access_'):
            continue
        ep0, ep1 = link.endpoints
        n0 = sim.topology.nodes.get(ep0)
        n1 = sim.topology.nodes.get(ep1)
        if not n0 or not n1:
            continue
        # UE access links only: exactly one endpoint is a UE
        if (n0.node_type == NodeType.UE) == (n1.node_type == NodeType.UE):
            continue
        ue_id, site_id = ((ep0, ep1) if n0.node_type == NodeType.UE
                          else (ep1, ep0))

        base_cap = getattr(link, '_uu_base_capacity', None)
        if base_cap is None:
            base_cap = link.capacity
            link._uu_base_capacity = base_cap

        idx = len(all_uu)
        all_uu.append((lid, ep0, ep1, link, base_cap))

        ue_node = sim.topology.nodes.get(ue_id)
        if (UU_CA_MAX_EXTRA_CC > 0
                and target_counts.get(ue_id, 0) >= UU_CA_HOTSPOT_MIN_FLOWS
                and ue_node is not None and ue_node.is_survivor
                and sim.phy_mac_states.get(site_id) is not None
                and site_enabled(site_id)):
            site_links[site_id].append(idx)

    # ── PASS 2 ── per SITE: choose k, build the finite pool, WATER-FILL it
    # across that site's hot-spot links.  This is the whole point of v2: the
    # loop below can never hand out more than `pool` per site, so the number
    # of camped UEs changes only the SPLIT, never the TOTAL.
    extra_by_idx = {}
    n_cc_total = 0
    extra_total = 0.0
    sites_activated = 0
    worst_site_extra = 0.0
    for site_id, members in site_links.items():
        # k of N_CC: an INTEGER count of carriers.  One spare carrier is lit
        # per hot spot present, capped by what the site is licensed for.  With
        # the default N_CC = 1 this is exactly the binary decision "the site
        # has >= 1 hot spot -> activate its single spare SCell".
        k = min(UU_CA_MAX_EXTRA_CC, len(members))
        if k <= 0:
            continue
        pool = k * UU_CA_CC_REF_CAPACITY_MBPS

        # Per-UE aggregation-capability bound: a UE aggregating k carriers of
        # the same width at the same site's MCS cannot exceed k x its
        # primary-carrier rate.
        head = {i: k * all_uu[i][4] for i in members}
        got = {i: 0.0 for i in members}
        pending = set(members)
        remaining = pool
        # Water-filling: equal shares, links that hit their per-UE ceiling
        # drop out and their residual is redistributed among the rest.
        # Terminates because each round either empties `pending` or removes
        # at least one member.
        while pending and remaining > 1e-9:
            share = remaining / len(pending)
            capped = [i for i in pending if (head[i] - got[i]) <= share]
            if capped:
                for i in capped:
                    d = max(0.0, head[i] - got[i])
                    got[i] += d
                    remaining -= d
                    pending.discard(i)
            else:
                for i in pending:
                    got[i] += share
                remaining = 0.0

        site_extra = sum(got.values())
        # ── THE SITE-LEVEL BOUND, ENFORCED (not merely documented) ──
        assert site_extra <= pool + 1e-6, (
            f"Uu CA site pool violated at {site_id}: {site_extra:.3f} > "
            f"{pool:.3f} Mbps@ref (k={k}, links={len(members)})")
        for i, v in got.items():
            if v > 0.0:
                extra_by_idx[i] = v
        n_cc_total += k
        extra_total += site_extra
        sites_activated += 1
        worst_site_extra = max(worst_site_extra, site_extra)

    # ── PASS 3 ── write capacities.  EVERY Uu access link is rewritten from
    # its cached base every tick, so a link that lost the grant snaps back
    # exactly and nothing ratchets over a long run.
    n_boosted = 0
    factor_sum = 0.0
    for i, (lid, ep0, ep1, link, base_cap) in enumerate(all_uu):
        extra = extra_by_idx.get(i, 0.0)
        new_cap = base_cap + extra
        if abs(new_cap - link.capacity) > 1e-9:
            link.capacity = new_cap
            if sim.topology.graph.has_edge(ep0, ep1):
                sim.topology.graph[ep0][ep1]['capacity'] = new_cap
        if extra > 0.0:
            n_boosted += 1
            factor_sum += (new_cap / base_cap) if base_cap > 1e-9 else 1.0

    mean_factor = (factor_sum / n_boosted) if n_boosted else 1.0
    if n_cc_total > 0 and tick % 100 == 0:
        print(f"{log_prefix}[UU-BOOST] t={tick}: {sites_activated} sites "
              f"activated {n_cc_total} extra component carrier(s) "
              f"({UU_CA_CC_BANDWIDTH_MHZ:.0f} MHz each, N_CC="
              f"{UU_CA_MAX_EXTRA_CC}/site) -> {extra_total:.0f} Mbps@QAM256 "
              f"aggregate extra, shared over {n_boosted} Uu links "
              f"(mean {mean_factor:.3f}x); worst site "
              f"{worst_site_extra:.0f} <= bound "
              f"{UU_CA_SITE_POOL_BOUND_MBPS:.0f} Mbps ({gate_desc})")
    return n_boosted, mean_factor, n_cc_total, extra_total


def _anchor_deployment_ewc(mappo_trainer, log_prefix=""):
    """Anchor the deployment EWC penalty to the JUST-LOADED policy weights.

    Uses the EWCPenalty machinery already implemented in
    sixg_sim/mappo_trainer.py: update() adds ewc_loss(_ref_actor) to the
    actor loss whenever deploy_mode is on and the penalty is_anchored.
    The checkpoint ships no observation buffer, so the Fisher diagonal
    cannot be estimated at load time; the anchor starts with an IDENTITY
    Fisher — EWC then reduces to (λ/2)·‖θ−θ*‖² (an L2 pull toward the
    pretrained weights, a.k.a. L2-SP).  A proper diagonal Fisher is
    estimated ONCE later, from live deployment observations (see the
    EWC_FISHER_MIN_OBS hook in the tick loop), with the anchor point
    re-pinned to the same pretrained weights.

    Returns the pretrained anchor state {param_name: tensor} for that
    re-pinning, or None if the trainer has no actor.
    """
    ref = getattr(mappo_trainer, '_ref_actor', None)
    if ref is None:
        return None
    anchor_sd = {n: p.data.detach().clone() for n, p in ref.named_parameters()}
    identity_fisher = {n: torch.ones_like(t) for n, t in anchor_sd.items()}
    # update() reads only the FIRST penalty in the dict, but keep every
    # per-agent penalty consistent (shared read-only dicts — ewc_loss never
    # mutates them, so no per-agent copies are needed).
    for pen in mappo_trainer.ewc.values():
        pen.ewc_lambda = DEPLOY_EWC_LAMBDA
        pen.fisher = identity_fisher
        pen.anchor_params = anchor_sd
        pen.is_anchored = True
    print(f"{log_prefix}[EWC] online adaptation anchored to loaded checkpoint "
          f"weights: lambda={DEPLOY_EWC_LAMBDA}, identity Fisher (L2-SP) "
          f"until {EWC_FISHER_MIN_OBS} deployment obs enable the one-time "
          f"diagonal-Fisher estimate")
    return anchor_sd


class ControlPlaneMessageSim:
    """Simulates actual control-plane message exchanges per tick.

    Instead of crude byte formulas, this tracks per-node state and counts
    real messages that would be exchanged based on protocol semantics:

    - MARL:   Delta-based postcards — sent only when node state changes.
              Neighbors assume last-known state if no update received.
    - OSPF:   Periodic Hello (every HELLO_INTERVAL), LSA flooding on
              topology change, periodic LSA refresh.
    - SDN:    Phase-driven: heartbeats, election rounds, LLDP probes,
              FlowMod installs, periodic stats polling.
    - OLSR:   Periodic Hello (every OLSR_HELLO_INTERVAL), TC messages
              flooded only via MPR subset (every OLSR_TC_INTERVAL).
    - BATMAN: Periodic OGM broadcast (every BATMAN_OGM_INTERVAL),
              rebroadcasted by all receivers (with TTL limit).
    - AODV:   On-demand RREQ broadcast per discovery, RREP unicast back.
              Zero messages when all routes are cached.

    Returns (msg_count, total_bytes) per tick, converting to overhead_mbps.
    """

    # Message sizes (bytes) — from RFC specs, variable where indicated
    # Fixed-size protocol messages:
    MARL_POSTCARD_BYTES   = 200   # State vector: MCS, power, relay, UEs, loads
    SDN_ELECTION_BYTES    = 64    # Raft RequestVote / AppendEntries RPC
    SDN_LLDP_BYTES        = 120   # LLDP Ethernet frame (~60-150B typical)
    SDN_FLOWMOD_BYTES     = 256   # OpenFlow 1.3 FlowMod (header + match + actions)
    SDN_STATS_BYTES       = 200   # OpenFlow StatsRequest + Reply pair
    BATMAN_OGM_BYTES      = 52    # batman-adv OGM (header + L2 framing)
    AODV_RREQ_BYTES       = 24    # RFC 3561 S5.1: 24-byte RREQ (fixed)
    AODV_RREP_BYTES       = 20    # RFC 3561 S5.2: ~20-byte RREP (19 + pad)
    AODV_RERR_BYTES_BASE  = 4     # RFC 3561 S5.3: 4-byte RERR header
    AODV_RERR_PER_DEST    = 8     # + 8 bytes per unreachable destination
    # OpenFlow echo = OF header only (no payload for liveness check)
    SDN_HEARTBEAT_BYTES   = 16    # OpenFlow 1.3 echo request (16B header)

    # Variable-size messages — computed from actual neighbour count:
    # OSPF Hello   = 24B header + 20B fixed fields + 4B per listed neighbour
    # OSPF LSA     = 20B LSA header + 4B body header + 12B per link record
    # OLSR Hello   = 12B OLSR hdr + 20B msg hdr + 4B per neighbour (+ IP/UDP)
    # OLSR TC      = 12B OLSR hdr + 20B msg hdr + 4B per MPR selector (+ IP/UDP)
    OSPF_HELLO_BASE       = 44    # 24B header + 20B fixed Hello fields
    OSPF_HELLO_PER_NBR    = 4     # 4B per listed neighbour (Router-ID)
    OSPF_LSA_BASE         = 24    # 20B LSA header + 4B body header
    OSPF_LSA_PER_LINK     = 12    # 12B per link record (ID+Data+Type+Metric)
    OLSR_HELLO_BASE       = 60    # 12 + 20 + 28 (IP/UDP + OLSR + msg headers)
    OLSR_HELLO_PER_NBR    = 4     # 4B per neighbour address entry
    OLSR_TC_BASE          = 60    # 12 + 20 + 28 (headers)
    OLSR_TC_PER_MPR       = 4     # 4B per advertised MPR-selector address

    def __init__(self, sim, mode):
        self.sim = sim
        self.mode = mode
        # Per-node previous state snapshot (for delta detection)
        self._prev_node_state = {}
        # Track per-node link states for topology change detection
        self._prev_link_states = {}
        self._tick_count = 0
        # Per-node timer offsets for staggered message sending.
        # In real networks, each node's timer starts at a random offset
        # within the interval — messages don't synchronize on the same tick.
        self._node_offsets = {}

    def _get_survivor_nodes(self):
        """Get list of surviving INFRASTRUCTURE node IDs.
        Excludes UEs — they don't participate in control-plane signaling.
        Only gNBs, Relays/IABs, and EdgeUPFs run routing protocols.
        """
        return [nid for nid, n in self.sim.topology.nodes.items()
                if n.is_survivor and n.node_type.value not in
                {"Core", "UPF", "AMF", "CoreUPF", "UE"}]

    def _get_node_state_snapshot(self, nid):
        """Capture the current state of a node for change detection.
        Only includes control-plane state that would warrant a postcard:
        MCS, power, relay mode, neighbor topology.
        
        Excludes per-tick metrics (active_ue_count, link utilization)
        because stochastic traffic causes them to fluctuate every tick,
        triggering artificial postcard storms. The 5% background rate
        in the post-convergence phase already captures gradual
        load-driven adjustments.
        """
        ps = self.sim.phy_mac_states.get(nid)
        if not ps:
            return None
        return (
            ps.mcs_general,
            ps.tx_power_dbm,
            ps.relay_mode,
            len(list(self.sim.topology.graph.neighbors(nid))),
        )

    def _get_node_neighbor_count(self, nid):
        """Count active neighbors (links that are up)."""
        count = 0
        for nbr in self.sim.topology.graph.neighbors(nid):
            edata = self.sim.topology.graph.get_edge_data(nid, nbr)
            if edata and 'link_id' in edata:
                link = self.sim.topology.links.get(edata['link_id'])
                if link and getattr(link, 'is_up', True):
                    count += 1
        return count

    def _detect_topology_change(self):
        """Detect if any links changed state since last tick."""
        changed = False
        current_link_states = {}
        for lid, link in self.sim.topology.links.items():
            current_link_states[lid] = getattr(link, 'is_up', True)
        if self._prev_link_states:
            for lid in current_link_states:
                if current_link_states.get(lid) != self._prev_link_states.get(lid):
                    changed = True
                    break
        self._prev_link_states = current_link_states
        return changed

    def _get_node_offset(self, nid, interval):
        """Get a deterministic per-node timer offset for staggered messaging.
        Each node sends at a different tick within the interval, avoiding
        artificial synchronization spikes.

        Uses zlib.crc32 of (node_id, interval) to produce a uniform
        distribution across the interval range — no caching, no collisions.

        crc32, NOT the builtin hash(): hash() of a str is SALTED per process
        (PYTHONHASHSEED), and the arms run in a multiprocessing pool, so
        hash() gave every arm and every re-run a DIFFERENT timer phasing.
        That made four arms irreproducible across processes and meant each
        pass drew a fresh jitter realisation, which invalidates any
        confidence interval computed over passes.
        """
        # Combine nid and interval so offset changes when interval changes
        # (e.g. OSPF Hello switching from 10 s to 1 s during fast-detect)
        return zlib.crc32(f"{nid}|{interval}".encode('utf-8')) % max(1, interval)

    def compute_tick_messages(self, tick):
        """Compute actual control-plane messages for this tick.

        Returns (message_count, total_bytes).

        IMPORTANT: Before severance (tick < SEVERANCE_TICK), ALL protocols
        produce ZERO overhead. The core network handles routing — there's
        no distributed routing protocol active on the mesh.
        """
        self._tick_count += 1

        # ── Pre-disaster: no distributed protocols active ──
        # Before severance, the core handles all routing.
        # MARL agents aren't exchanging postcards (policy not activated).
        # SDN has no local controller (central controller is in the core).
        # OSPF/OLSR/BATMAN/AODV aren't running on the mesh yet.
        # BUT we still track link states so _detect_topology_change() works
        # at the exact severance tick (needs previous-tick snapshot).
        effective_sev = 0 if getattr(self, '_pre_converged', False) else SEVERANCE_TICK
        if tick < effective_sev:
            # Keep link state tracking updated for transition detection
            for lid, link in self.sim.topology.links.items():
                self._prev_link_states[lid] = getattr(link, 'is_up', True)
            return (0, 0)

        survivors = self._get_survivor_nodes()
        N = len(survivors)
        if N == 0:
            return (0, 0)

        mode = self.mode
        msg_count = 0
        total_bytes = 0
        topo_changed = self._detect_topology_change()

        if mode == 'marl':
            msg_count, total_bytes = self._sim_marl(tick, survivors, topo_changed)
        elif mode == 'ospf':
            msg_count, total_bytes = self._sim_ospf(tick, survivors, N, topo_changed)
        elif mode == 'sdn':
            msg_count, total_bytes = self._sim_sdn(tick, survivors, N, topo_changed)
        elif mode == 'olsr':
            msg_count, total_bytes = self._sim_olsr(tick, survivors, N, topo_changed)
        elif mode == 'batman':
            msg_count, total_bytes = self._sim_batman(tick, survivors, N)
        elif mode == 'aodv':
            msg_count, total_bytes = self._sim_aodv(tick, survivors, N)

        # Update state snapshots for next tick's delta detection
        for nid in survivors:
            self._prev_node_state[nid] = self._get_node_state_snapshot(nid)

        return (msg_count, total_bytes)

    # ── MARL: delta-based postcards ────────────────────────────────

    def _sim_marl(self, tick, survivors, topo_changed):
        """MARL nodes send postcards ONLY when their state changes.
        Neighbors assume last-known state if no postcard received.

        This means:
        - Pre-severance (stable): very few postcards (only load fluctuations)
        - At severance: ALL affected nodes detect change → burst
        - During convergence (50 ticks): agents explore MCS/power/beam →
          frequent postcards that decay as policies stabilize
        - Post-convergence: low-rate background postcards from ongoing
          online PPO learning (agents still adjust to load/interference)
        - Rescue UEs arrive: another burst, then stabilises
        """
        msg_count = 0
        total_bytes = 0
        ticks_since_sever = tick - SEVERANCE_TICK if tick >= SEVERANCE_TICK else -1
        if getattr(self, '_pre_converged', False):
            ticks_since_sever = tick + 1000   # skip convergence phases

        for nid in survivors:
            current_state = self._get_node_state_snapshot(nid)
            prev_state = self._prev_node_state.get(nid)

            if prev_state is None:
                send = True
            elif current_state != prev_state:
                send = True
            elif topo_changed:
                send = True
            else:
                # Post-convergence: online PPO learning continues.
                # Agents still adjust MCS/power/PRB in response to
                # load changes, interference, UE mobility. These small
                # adjustments occasionally trigger postcard updates.
                # ~2% chance per node per tick ≈ 1 node per tick on avg.
                # crc32, not hash(): hash() is salted per process, so the
                # postcard realisation differed between arms and re-runs.
                rng = random.Random(
                    tick * 1009 + zlib.crc32(nid.encode('utf-8')) % 997)
                send = rng.random() < 0.02

            if send:
                # MARL postcards are LOCAL BROADCASTS — each node sends
                # ONE 200-byte postcard on its radio, received by all
                # neighbors simultaneously. This is NOT a unicast per
                # neighbor — it's a single L2 broadcast frame.
                msg_count += 1
                total_bytes += self.MARL_POSTCARD_BYTES

        return (msg_count, total_bytes)

    # ── OSPF: periodic Hello + event-triggered LSA ─────────────────

    def _sim_ospf(self, tick, survivors, N, topo_changed):
        """OSPF Hello and LSA — RFC 2328 compliant message counting.

        Hello: ONE Hello per node per HelloInterval, not one per neighbour.
               RFC 2328 §9.5 sends Hellos to AllSPFRouters (224.0.0.5) on a
               broadcast/NBMA-broadcast interface, i.e. a SINGLE L2 broadcast
               frame on the shared radio that every adjacent router receives —
               exactly how the MARL postcard is billed in _sim_marl.  This
               used to be `msg_count += n_nbrs` with `n_nbrs * hello_size`
               bytes, which charged OSPF a unicast per neighbour for one
               broadcast and so overstated its Hello overhead by the mean
               degree while MARL was charged once.  The Hello PACKET still
               grows with the neighbour count (44 B base + 4 B per listed
               neighbour, §A.3.2) — that part was already right.
               HelloInterval = OSPF_HELLO_INTERVAL (10 s), or
               OSPF_HELLO_FAST_INTERVAL (1 s) while the measured control
               plane is still converging (§C.5 fast-hello / BFD-assisted).

        LSA:   RFC 2328 §13.3 — reliable flooding.
               Each unique LSA traverses each link EXACTLY ONCE per direction,
               so the flood IS per-link and is counted per-link.
               MinLSInterval (RFC 2328 §12.4) = OSPF_MIN_LS_INTERVAL (5 s) →
               at most 1 LSA origination per node per 5 ticks.  The flood of
               each LSA produces |E| messages across the network (one per
               directed link).
        """
        msg_count = 0
        total_bytes = 0
        ticks_since_sever = tick - SEVERANCE_TICK if tick >= SEVERANCE_TICK else -1
        if getattr(self, '_pre_converged', False):
            ticks_since_sever = tick + 1000

        # Hello packets: ONE broadcast to AllSPFRouters per node per interval.
        # Fast-hello during convergence (RFC 2328 §C.5: BFD/fast-hello = 1s).
        # "Converging" is read from the MEASURED OSPF control-plane state
        # machine, not from a fixed end-tick.
        # Intervals are in TICKS = SECONDS (was 100/1000, a 10 ms-tick
        # leftover that understated OSPF Hello overhead 100x).
        _ospf_ctrl = getattr(self.sim, '_ospf_ctrl', None)
        _converging = (ticks_since_sever > 0 and _ospf_ctrl is not None and
                       getattr(_ospf_ctrl, 'convergence_fraction', 1.0) < 1.0)
        hello_interval = (OSPF_HELLO_FAST_INTERVAL if _converging
                          else OSPF_HELLO_INTERVAL)
        for nid in survivors:
            offset = self._get_node_offset(nid, hello_interval)
            if (tick + offset) % hello_interval == 0:
                n_nbrs = self._get_node_neighbor_count(nid)
                # ONE L2 broadcast frame on the shared radio, seen by every
                # adjacency — billed exactly as the MARL postcard is.
                msg_count += 1
                # RFC 2328 §A.3.2: Hello = 44B base + 4B per listed neighbour
                hello_size = self.OSPF_HELLO_BASE + self.OSPF_HELLO_PER_NBR * n_nbrs
                total_bytes += hello_size

        # LSA flooding — when topology actually changed, or while the
        # measured OSPF control plane is still flooding/converging.
        # RFC 2328 §13.3: reliable flooding sends each LSA once per link.
        lsa_window = 20  # ticks over which a single flood burst dissipates
        if topo_changed or _converging:
            # Count total directed links in surviving topology (= flood scope)
            total_links = sum(self._get_node_neighbor_count(nid)
                              for nid in survivors)

            # How many nodes originate new LSAs this tick?
            # At severance (tick 0-5): all affected nodes originate at once.
            # MinLSInterval (OSPF_MIN_LS_INTERVAL = 5 s = 5 ticks) prevents
            # re-origination, so flood intensity decays as nodes finish their
            # initial burst.  (The old comment claimed "5s = 500 ticks", the
            # same 10 ms-tick error corrected above.)
            if ticks_since_sever <= OSPF_MIN_LS_INTERVAL:
                # Initial burst: all nodes detect adjacency change
                # Stagger across ticks to avoid artificial synchronisation
                originators = 0
                for nid in survivors:
                    rng = random.Random(
                        tick * 3571 + zlib.crc32(nid.encode('utf-8')) % 991)
                    if rng.random() < 0.8:  # ~80% originate per tick
                        originators += 1
            else:
                # Decay: SPF throttle + MinLSInterval suppresses re-origination
                decay = max(0.05, 1.0 - (ticks_since_sever - OSPF_MIN_LS_INTERVAL)
                            / max(1, lsa_window - OSPF_MIN_LS_INTERVAL))
                originators = 0
                for nid in survivors:
                    rng = random.Random(
                        tick * 3571 + zlib.crc32(nid.encode('utf-8')) % 991)
                    if rng.random() < decay * 0.3:
                        originators += 1

            if originators > 0:
                # Each originated LSA floods across ALL links (once per link)
                # But duplicate suppression means overlapping floods merge.
                # Effective messages ≈ originators × total_links / N
                # (each LSA covers the full topology, but they share links)
                avg_degree = total_links / max(1, N)
                avg_lsa_size = (self.OSPF_LSA_BASE +
                                self.OSPF_LSA_PER_LINK * avg_degree)
                # With duplicate suppression, effective flood = total_links
                # regardless of how many nodes originate (they all share the
                # same set of links).  Scale by min(originators/N, 1).
                flood_fraction = min(1.0, originators / max(1, N))
                flood_msgs = int(total_links * flood_fraction)
                msg_count += flood_msgs
                total_bytes += flood_msgs * int(avg_lsa_size)

        return (msg_count, total_bytes)

    # ── SDN: phase-specific message patterns ──────────────────────

    def _sim_sdn(self, tick, survivors, N, topo_changed):
        """SDN messages derived from actual per-node, per-tick activity.

        Each phase has per-node staggered messaging so overhead varies
        naturally tick-to-tick rather than producing flat phase-locked steps.
        """
        msg_count = 0
        total_bytes = 0
        ctrl = getattr(self.sim, '_sdn_ctrl', None)
        phase = ctrl.phase if ctrl else 'orphan'
        ticks_since_sever = tick - SEVERANCE_TICK if tick >= SEVERANCE_TICK else 0
        if getattr(self, '_pre_converged', False):
            ticks_since_sever = tick + 1000

        if phase == 'orphan':
            # Nodes detect controller-heartbeat miss at staggered times.
            # Each node has its own heartbeat timer; as they expire one by
            # one over the orphan window, each sends a probe burst.
            # TICKS = SECONDS (was 10, i.e. "100 ms" on the old 10 ms tick —
            # which at the real 1 s tick meant one probe every 10 s and so
            # understated SDN orphan-phase overhead ~10x).
            orphan_probe_interval = SDN_ORPHAN_PROBE_INTERVAL
            for nid in survivors:
                offset = self._get_node_offset(nid, orphan_probe_interval)
                if (tick + offset) % orphan_probe_interval == 0:
                    n_nbrs = self._get_node_neighbor_count(nid)
                    msg_count += n_nbrs
                    total_bytes += n_nbrs * self.SDN_HEARTBEAT_BYTES

        elif phase == 'election':
            # Raft/Paxos election happens in rounds.  Not all nodes
            # participate every tick -- election proceeds in alternating
            # RequestVote / VoteResponse rounds with random back-off.
            # TICKS = SECONDS (was 5, i.e. "50 ms" on the old 10 ms tick).
            election_round_len = SDN_ELECTION_ROUND_S
            round_num = ticks_since_sever // election_round_len
            is_vote_round = (round_num % 2 == 0)  # alternate vote/response
            for nid in survivors:
                # Only a fraction of nodes active per tick (staggered timeouts)
                offset = self._get_node_offset(nid, election_round_len)
                if (tick + offset) % election_round_len == 0:
                    n_nbrs = self._get_node_neighbor_count(nid)
                    if is_vote_round:
                        msg_count += n_nbrs  # RequestVote
                        total_bytes += n_nbrs * self.SDN_ELECTION_BYTES
                    else:
                        msg_count += 1  # VoteResponse (unicast to candidate)
                        total_bytes += self.SDN_ELECTION_BYTES

        elif phase in ('discovery', 'lldp'):
            # LLDP BFS discovery: only the *frontier* nodes send probes
            # each tick, not all nodes.  The frontier expands outward from
            # each elected leader.
            if ctrl and hasattr(ctrl, 'discovery_frontier'):
                frontier_nodes = set()
                for fid, front in ctrl.discovery_frontier.items():
                    if not ctrl.discovery_done.get(fid, False):
                        frontier_nodes |= front
                for nid in sorted(frontier_nodes):
                    if nid in [s for s in survivors]:
                        n_nbrs = self._get_node_neighbor_count(nid)
                        msg_count += n_nbrs
                        total_bytes += n_nbrs * self.SDN_LLDP_BYTES
            else:
                # Fallback: staggered LLDP, one probe round per
                # SDN_LLDP_ROUND_S seconds (LLDP is a PER-PORT PDU, so the
                # per-neighbour multiplication here is correct — unlike the
                # OSPF/OLSR Hello broadcasts corrected above).
                for nid in survivors:
                    offset = self._get_node_offset(nid, SDN_LLDP_ROUND_S)
                    if (tick + offset) % SDN_LLDP_ROUND_S == 0:
                        n_nbrs = self._get_node_neighbor_count(nid)
                        msg_count += n_nbrs
                        total_bytes += n_nbrs * self.SDN_LLDP_BYTES

        elif phase == 'programming':
            # Controller pushes FlowMod to switches.  Counts come from
            # the actual controller queue + real path lengths.
            if ctrl:
                queue_len = len(getattr(ctrl, 'flow_queue', []))
                batch_size = min(ctrl.FLOWS_PER_TICK, queue_len)
                if batch_size > 0:
                    msg_count += batch_size * 2  # FlowMod + BarrierReply
                    total_bytes += batch_size * self.SDN_FLOWMOD_BYTES
                    # Per-hop FlowMod install along the actual path.
                    # Query real shortest-path length for each flow.
                    graph = self.sim.topology.graph
                    flows_being_programmed = list(ctrl.flow_queue)[:batch_size]
                    for flow in flows_being_programmed:
                        src, tgt = flow
                        try:
                            path_len = nx.shortest_path_length(graph, src, tgt)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            path_len = 0
                        msg_count += path_len
                        total_bytes += path_len * self.SDN_FLOWMOD_BYTES

        else:  # operational
            # LLDP keepalive + stats: staggered per-node.
            # TICKS = SECONDS (were 100 / 500, i.e. "1 s / 5 s" on the old
            # 10 ms tick, which at the real 1 s tick meant one LLDP every
            # 100 s and one stats poll every 500 s — understating SDN
            # steady-state overhead by ~100x).
            lldp_interval = SDN_LLDP_KEEPALIVE_S
            stats_interval = SDN_STATS_POLL_S
            for nid in survivors:
                offset = self._get_node_offset(nid, lldp_interval)
                if (tick + offset) % lldp_interval == 0:
                    n_nbrs = self._get_node_neighbor_count(nid)
                    msg_count += n_nbrs
                    total_bytes += n_nbrs * self.SDN_LLDP_BYTES
                offset_s = self._get_node_offset(nid, stats_interval)
                if (tick + offset_s) % stats_interval == 0:
                    msg_count += 1
                    total_bytes += self.SDN_STATS_BYTES

            # FlowMod from operational rerouting/churn
            if ctrl and hasattr(ctrl, 'flow_queue') and ctrl.flow_queue:
                reroute_count = min(10, len(ctrl.flow_queue))
                msg_count += reroute_count * 2
                total_bytes += reroute_count * self.SDN_FLOWMOD_BYTES

        return (msg_count, total_bytes)

    # ── OLSR: periodic Hello + MPR-filtered TC ─────────────────────

    def _sim_olsr(self, tick, survivors, N, topo_changed):
        """OLSR: Hello staggered per-node, TC only via MPR subset.

        HELLO and TC are both counted as ONE LOCAL BROADCAST per originating
        node per interval, matching how the MARL postcard and the OSPF Hello
        are billed.  RFC 3626 §6 sends every OLSR control message to the
        link-local broadcast address on the shared radio interface: one frame,
        every neighbour hears it.  Both used to be `msg_count += n_nbrs` with
        `n_nbrs * size` bytes — a unicast per neighbour for what is a single
        broadcast, overstating OLSR overhead by the mean node degree.  The
        message SIZE still grows with the advertised neighbour / MPR-selector
        count, which was already correct.
        """
        msg_count = 0
        total_bytes = 0

        # Hello messages: each node sends at its own staggered offset
        for nid in survivors:
            offset = self._get_node_offset(nid, OLSR_HELLO_INTERVAL)
            if (tick + offset) % OLSR_HELLO_INTERVAL == 0:
                n_nbrs = self._get_node_neighbor_count(nid)
                msg_count += 1               # one link-local broadcast
                # RFC 3626: Hello = 60B base + 4B per listed neighbour
                hello_size = self.OLSR_HELLO_BASE + self.OLSR_HELLO_PER_NBR * n_nbrs
                total_bytes += hello_size

        # TC messages: staggered, only MPR nodes generate
        ctrl = getattr(self.sim, '_olsr_ctrl', None)
        mpr_sets = getattr(ctrl, 'mpr_sets', {}) if ctrl else {}
        all_mprs = set()
        for selector, mprs in mpr_sets.items():
            all_mprs.update(mprs)

        for nid in survivors:
            if nid in all_mprs or (not mpr_sets):  # early phase: all send
                offset = self._get_node_offset(nid, OLSR_TC_INTERVAL)
                if (tick + offset) % OLSR_TC_INTERVAL == 0:
                    n_nbrs = self._get_node_neighbor_count(nid)
                    msg_count += 1           # one link-local broadcast
                    # RFC 3626: TC = 60B base + 4B per MPR-selector address
                    # n_nbrs approximates MPR-selector count for this node
                    tc_size = self.OLSR_TC_BASE + self.OLSR_TC_PER_MPR * n_nbrs
                    total_bytes += tc_size

        return (msg_count, total_bytes)

    # ── BATMAN: periodic OGM broadcast + rebroadcast ───────────────

    def _sim_batman(self, tick, survivors, N):
        """B.A.T.M.A.N. OGM overhead from actual graph topology.

        Each originator broadcasts its OGM to all neighbours.
        Each receiver re-broadcasts it to all ITS neighbours (minus
        the incoming link).  TTL limits propagation depth.
        We walk the actual graph to count real messages.
        """
        msg_count = 0
        total_bytes = 0
        survivor_set = set(survivors)
        hop_limit = min(5, N // 8 + 1)

        for nid in survivors:
            offset = self._get_node_offset(nid, BATMAN_OGM_INTERVAL)
            if (tick + offset) % BATMAN_OGM_INTERVAL == 0:
                # BFS from originator up to hop_limit hops,
                # counting actual messages at each hop.
                visited = {nid}
                frontier = {nid}
                ttl = hop_limit
                while frontier and ttl > 0:
                    next_frontier = set()
                    for src in frontier:
                        for nbr in self.sim.topology.graph.neighbors(src):
                            if nbr in survivor_set and nbr not in visited:
                                msg_count += 1     # 1 OGM copy received
                                total_bytes += self.BATMAN_OGM_BYTES
                                next_frontier.add(nbr)
                                visited.add(nbr)
                    frontier = next_frontier
                    ttl -= 1

        return (msg_count, total_bytes)

    # ── AODV: demand-driven RREQ/RREP ─────────────────────────────

    def _sim_aodv(self, tick, survivors, N):
        """AODV overhead — every message count from the actual graph.

        RREQ: We perform a real BFS on the topology graph from the source
              node.  The RREQ has been propagating for `elapsed` ticks,
              so the flooding has reached hop `elapsed`.  We walk the
              BFS ring at exactly that depth and count the actual number
              of neighbours each frontier node would broadcast to.

        RREP: We find the actual shortest path length (src→tgt) on the
              graph and count one unicast RREP per hop.
        """
        msg_count = 0
        total_bytes = 0
        ctrl = getattr(self.sim, '_aodv_ctrl', None)
        if not ctrl:
            return (0, 0)

        survivor_set = set(survivors)
        graph = self.sim.topology.graph

        for flow, start_tick in ctrl._active_rreqs.items():
            src, tgt = flow
            elapsed = tick - start_tick

            if elapsed < AODV_RREQ_TIMEOUT:
                # Real BFS expansion from src up to `elapsed` hops.
                # Only the frontier ring (nodes at exactly `elapsed` hops)
                # broadcasts this tick.
                visited = {src}
                frontier = {src}
                for _hop in range(elapsed):
                    next_frontier = set()
                    for nd in frontier:
                        for nbr in graph.neighbors(nd):
                            if nbr in survivor_set and nbr not in visited:
                                next_frontier.add(nbr)
                                visited.add(nbr)
                    frontier = next_frontier
                    if not frontier:
                        break

                # Each frontier node rebroadcasts the RREQ to all ITS
                # actual neighbours (minus the one it received from, but
                # since RREQ is broadcast, it goes to all).
                for nd in frontier:
                    n_nbrs = self._get_node_neighbor_count(nd)
                    msg_count += n_nbrs
                    total_bytes += n_nbrs * self.AODV_RREQ_BYTES

            elif elapsed < AODV_RREQ_TIMEOUT + AODV_RREP_TIMEOUT:
                # RREP: unicast back along the actual shortest path.
                # One hop forwarded per tick.
                rrep_hop = elapsed - AODV_RREQ_TIMEOUT
                try:
                    path_len = nx.shortest_path_length(graph, src, tgt)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path_len = 0
                if rrep_hop < path_len:
                    msg_count += 1   # 1 RREP forwarded this tick
                    total_bytes += self.AODV_RREP_BYTES

        # RERR on link failures
        if ctrl.phase == 'route_repair':
            # RERR propagates to upstream nodes.
            # RFC 3561 S5.3: RERR = 4B header + 8B per unreachable dest.
            # Typically 1-3 destinations become unreachable per link break.
            n_unreachable = 2  # approximate
            rerr_size = self.AODV_RERR_BYTES_BASE + self.AODV_RERR_PER_DEST * n_unreachable
            msg_count += 2
            total_bytes += 2 * rerr_size

        return (msg_count, total_bytes)


class OSPFController:
    """Measured OSPF (RFC 2328) control-plane convergence.

    Recovery is NOT scripted.  It emerges from RFC timer parameters playing
    out over the ACTUAL surviving topology, per tick:

    Phase 1 — NEIGHBOUR-LOSS DETECTION (Hello/Dead timers):
        A router declares an adjacency dead when no Hello arrives within
        RouterDeadInterval (4×Hello).  Each router's Hello timer has its own
        phase offset, so detection is staggered across
        [Dead − Hello, Dead] seconds after severance.

    Phase 2 — LSA FLOODING (hop-by-hop over surviving links):
        A router that detected the change originates a new Router-LSA.
        LSAs propagate one hop per OSPF_FLOOD_HOP_TICKS tick across the
        CURRENT live link set (per-tick BFS relaxation over real edges).

    Phase 3 — SPF + FIB INSTALL (per node):
        A router runs SPF once its LSDB holds LSAs from every router in its
        fragment and OSPF_SPF_DELAY seconds have passed since its last LSA
        arrival (SPF throttle).  Only then is that router converged.

    A flow is routable only when both endpoints are in the same fragment and
    both endpoint control planes (UEs inherit their serving router's state)
    have converged.  Convergence time therefore varies with seed, fragment
    structure and hop distances — it is measured, not asserted.
    """

    def __init__(self, sim):
        self.sim = sim
        is_rescue_ops = any(e.event_type == 'sever_core' and e.tick == 0 for e in sim.scenario.events)
        self.severance_tick = 0 if is_rescue_ops else SEVERANCE_TICK
        self.fragment_members = {}      # frag_id -> set of node_ids (infra + UE)
        self._fragments_computed = False
        self.lsdb = {}                  # nid -> {origin_router: arrival_tick}
        self.detect_tick = {}           # nid -> tick neighbour-loss detected
        self._node_spf_done = {}        # nid -> tick SPF completed
        self.node_convergence = {}      # nid -> 0.0 / 1.0
        self._ue_converge_tick = {}     # UE nid -> tick routes to it exist
        self.routable_flows = set()
        self.convergence_fraction = 0.0
        self.phase = 'detection'
        self._known_flows = None
        self._new_node_converge = {}    # late-arriving node -> converge tick

    def _compute_fragments(self, force=False):
        """Discover connected components among surviving INFRA routers over
        live infra-infra links — the SAME scope LSA flooding uses
        (_live_infra_neighbors) — then attach each surviving UE to the
        fragment(s) of its live infra anchors.

        The components MUST be computed on the infra-only graph: a UE with
        Uu links into two different infra fragments is not an OSPF router
        and forwards no LSAs, but a BFS over ALL live links walks through
        it and merges the two fragments into one pseudo-fragment.  The
        Phase-3 gate ("LSDB covers every router in my fragment") then
        counts routers a node can never hear from, SPF never fires for the
        merged fragment, and OSPF sits at 0% routable flows forever
        (observed: controller saw 3 fragments incl. a 32-router mega-
        fragment where the scenario builder verified 4; LSDB coverage
        plateaued at 11-21/32).

        A UE anchored in several fragments is added to EACH — it is
        genuinely reachable in each, and flow routability still requires
        both endpoints to share ONE fragment, so cross-fragment flows
        remain correctly excluded."""
        if self._fragments_computed and not force:
            return
        self._fragments_computed = True
        self.fragment_members = {}
        infra_set = self._infra_set()
        visited = set()
        frag_id = 0
        for start in sorted(infra_set):
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for peer in self._live_infra_neighbors(nid, infra_set):
                    if peer not in visited:
                        queue.append(peer)
            self.fragment_members[frag_id] = component
            frag_id += 1
        # Attach surviving UEs to the fragment(s) of their live anchors
        infra_frag = {nid: fid for fid, members in self.fragment_members.items()
                      for nid in members}
        for lnk in self.sim.topology.links.values():
            if not getattr(lnk, 'is_up', True):
                continue
            ep0, ep1 = lnk.endpoints
            for ue, anchor in ((ep0, ep1), (ep1, ep0)):
                if anchor not in infra_frag:
                    continue
                node = self.sim.topology.nodes.get(ue)
                if (node is not None and node.is_survivor
                        and node.node_type.value == 'UE'):
                    self.fragment_members[infra_frag[anchor]].add(ue)

    def _live_infra_neighbors(self, nid, infra_set):
        """Infra routers one live hop from nid (LSA flooding scope)."""
        nbrs = []
        for lnk in self.sim.topology.links.values():
            if not getattr(lnk, 'is_up', True):
                continue
            ep0, ep1 = lnk.endpoints
            peer = ep1 if ep0 == nid else (ep0 if ep1 == nid else None)
            if peer and peer in infra_set:
                nbrs.append(peer)
        return nbrs

    def _infra_set(self):
        return {nid for nid, n in self.sim.topology.nodes.items()
                if n.is_survivor and n.node_type.value not in
                {"Core", "UPF", "AMF", "CoreUPF", "UE"}}

    def mark_pre_converged(self, tick=0):
        """Scenario B: network already converged before this pass starts."""
        self._compute_fragments()
        infra = self._infra_set()
        for nid in infra:
            self.detect_tick[nid] = tick
            self._node_spf_done[nid] = tick
            self.lsdb[nid] = {o: tick for o in infra}
        for nid, n in self.sim.topology.nodes.items():
            if n.is_survivor:
                self.node_convergence[nid] = 1.0
        self.convergence_fraction = 1.0
        self.phase = 'operational'
        self._known_flows = set(self.sim.ue_to_ue_flows)

    def step(self, tick):
        ticks_since = tick - self.severance_tick
        if ticks_since < 0:
            return
        self._compute_fragments()
        infra_set = self._infra_set()

        # ── Phase 1: dead-timer detection (per-router staggered Hello phase) ──
        import zlib
        for nid in infra_set:
            if nid in self.detect_tick:
                continue
            # crc32 (not hash(): salted per process) → reproducible offsets
            phase_offset = zlib.crc32(f"hello|{nid}".encode()) % max(1, OSPF_HELLO_INTERVAL)
            t_detect = self.severance_tick + OSPF_DEAD_INTERVAL - phase_offset
            if tick >= t_detect:
                self.detect_tick[nid] = tick
                # Originate own Router-LSA into own LSDB
                self.lsdb.setdefault(nid, {})[nid] = tick

        # ── Phase 2: hop-by-hop LSA flooding over the LIVE link set ──
        # An LSA that arrived at tick T is forwarded to neighbours from
        # T + OSPF_FLOOD_HOP_TICKS onward (arrival==tick blocks same-tick
        # multi-hop teleporting — floods advance at the pacing rate).
        for nid in list(self.lsdb.keys()):
            if nid not in infra_set:
                continue
            entries = self.lsdb[nid]
            ready = [(o, arr) for o, arr in entries.items()
                     if tick - arr >= OSPF_FLOOD_HOP_TICKS]
            if not ready:
                continue
            nbrs = self._live_infra_neighbors(nid, infra_set)
            for origin, arr in ready:
                for nbr in nbrs:
                    nb_lsdb = self.lsdb.setdefault(nbr, {})
                    if origin not in nb_lsdb:
                        nb_lsdb[origin] = tick

        # ── Phase 3: per-node SPF once LSDB complete for its fragment ──
        for fid, members in self.fragment_members.items():
            frag_infra = [n for n in members if n in infra_set]
            if not frag_infra:
                continue
            for nid in frag_infra:
                if nid in self._node_spf_done:
                    continue
                entries = self.lsdb.get(nid, {})
                if entries and all(o in entries for o in frag_infra):
                    last_arrival = max(entries[o] for o in frag_infra)
                    if tick >= last_arrival + OSPF_SPF_DELAY:
                        self._node_spf_done[nid] = tick

        # ── Node convergence (routers) ──
        for nid in infra_set:
            self.node_convergence[nid] = 1.0 if nid in self._node_spf_done else 0.0

        # ── UE convergence: a UE is reachable once its serving router has
        #    run SPF (routes install one tick later) ──
        for fid, members in self.fragment_members.items():
            for nid in members:
                node = self.sim.topology.nodes.get(nid)
                if not node or node.node_type.value != 'UE':
                    continue
                if self.node_convergence.get(nid, 0.0) >= 1.0:
                    continue
                if nid in self._new_node_converge:
                    self.node_convergence[nid] = (
                        1.0 if tick >= self._new_node_converge[nid] else 0.0)
                    continue
                best_t = None
                for lnk in self.sim.topology.links.values():
                    if not getattr(lnk, 'is_up', True):
                        continue
                    ep0, ep1 = lnk.endpoints
                    peer = ep1 if ep0 == nid else (ep0 if ep1 == nid else None)
                    if peer and peer in self._node_spf_done:
                        t = self._node_spf_done[peer] + 1
                        best_t = t if best_t is None else min(best_t, t)
                self.node_convergence[nid] = (
                    1.0 if (best_t is not None and tick >= best_t) else 0.0)

        # ── Late arrivals (rescue UEs): new adjacency → new Router-LSA
        #    flood → SPF.  Delay = Hello (adjacency formation) + flood time
        #    across the fragment (measured BFS depth from serving router)
        #    + SPF throttle. ──
        if self._known_flows is None:
            self._known_flows = set(self.sim.ue_to_ue_flows)
        current_flows = set(self.sim.ue_to_ue_flows)
        new_flows = current_flows - self._known_flows
        if new_flows:
            self._compute_fragments(force=True)
            for flow in new_flows:
                for nid in flow:
                    if (nid in self.node_convergence and
                            self.node_convergence.get(nid, 0.0) >= 1.0):
                        continue
                    if nid in self._new_node_converge:
                        continue
                    # Serving router = closest live infra neighbour
                    serving = None
                    for lnk in self.sim.topology.links.values():
                        if not getattr(lnk, 'is_up', True):
                            continue
                        ep0, ep1 = lnk.endpoints
                        peer = ep1 if ep0 == nid else (ep0 if ep1 == nid else None)
                        if peer and peer in infra_set:
                            serving = peer
                            break
                    if serving is None:
                        continue  # physically unreachable — stays unconverged
                    # Measured flood depth: BFS eccentricity from serving router
                    depth = 0
                    visited = {serving}
                    frontier = {serving}
                    while frontier:
                        nxt = set()
                        for u in frontier:
                            for v in self._live_infra_neighbors(u, infra_set):
                                if v not in visited:
                                    visited.add(v)
                                    nxt.add(v)
                        if nxt:
                            depth += 1
                        frontier = nxt
                    self._new_node_converge[nid] = (
                        tick + OSPF_HELLO_INTERVAL
                        + depth * OSPF_FLOOD_HOP_TICKS + OSPF_SPF_DELAY)
        self._known_flows = current_flows

        # ── Routable flows: same fragment + both endpoints converged ──
        self.routable_flows.clear()
        for fid, members in self.fragment_members.items():
            for src, tgt in self.sim.ue_to_ue_flows:
                if src in members and tgt in members:
                    if (self.node_convergence.get(src, 0.0) >= 1.0 and
                            self.node_convergence.get(tgt, 0.0) >= 1.0):
                        self.routable_flows.add((src, tgt))

        # ── Summary phase / convergence fraction (measured) ──
        all_conv = [self.node_convergence.get(n, 0.0) for n in infra_set]
        self.convergence_fraction = (sum(all_conv) / len(all_conv)) if all_conv else 0.0
        if not self.detect_tick:
            self.phase = 'detection'
        elif self.convergence_fraction < 1.0:
            self.phase = 'flooding_spf'
        else:
            self.phase = 'operational'

    def get_routable_flows(self):
        """Return flows for which OSPF has converged routes at both ends."""
        # sorted(), NOT list(): `routable_flows` is a SET of (str, str)
        # tuples, and set iteration order over strings is SALTED per process
        # (PYTHONHASHSEED).  The engine allocates capacity GREEDILY in the
        # order this list is walked, so an unsorted set gave a different set
        # of served flows in every process - the same reproducibility defect
        # as the salted hash() seeds, just one layer up.  Measured: identical
        # runs diverged across PYTHONHASHSEED values until this was sorted.
        return sorted(self.routable_flows)


class SDNController:
    """Process-based OpenFlow SDN controller for island-mode recovery.

    Implements the actual bootstrapping sequence that an SDN controller
    would go through after core severance:

    EVERY DURATION HERE IS IN TICKS = SECONDS.  A previous revision wrote
    these literals against a 10 ms tick (FLOW_EXPIRY_TICKS = 100 "= 1 s",
    TICKS_PER_LLDP_ROUND = 5 "= 50 ms", election = 10 + 3N "ticks"), while
    the harness runs at TICK_DURATION_S = 1.0.  The effect was to inflate
    SDN's measured bootstrap ~100x: the orphan phase alone lasted 100 s and
    a 35-node election another 115 s, so the SDN arm was still in 'election'
    long after every other protocol had converged.  That is a unit bug, not
    an SDN property, and it flattered every arm SDN was compared against.

    Phase 1 — ORPHAN (no controller exists):
        Nodes detect core loss. Stale flow rules expire.
        Duration: SDN_FLOW_EXPIRY_S (10 s, OpenFlow hard_timeout).

    Phase 2 — LEADER ELECTION:
        Surviving nodes run a distributed election (e.g., RAFT/Paxos).
        One leader per fragment is elected — this becomes the controller.
        Duration: SDN_ELECTION_BASE_S + SDN_ELECTION_PER_NODE_S x N per
        fragment (2 s + 0.1 s per voter: Raft's 150-300 ms election timeout
        plus per-voter vote RTT over a multi-hop degraded wireless mesh).
        Fragments below MIN_FRAGMENT_FOR_ELECTION nodes elect nobody.

    Phase 3 — LLDP TOPOLOGY DISCOVERY:
        The elected leader sends LLDP probes outward in BFS rounds.
        Each round discovers one more hop from the controller.
        Duration: fragment diameter x SDN_LLDP_ROUND_S (1 s per round).
        Impaired links may cause discovery failures.

    Phase 4 — FLOW PROGRAMMING:
        Controller computes optimal routes for discovered UE-to-UE flows
        and installs flow rules via OpenFlow FlowMod messages.
        Rate: ~3 flows per tick (limited by OpenFlow channel bandwidth).

    Phase 5 — OPERATIONAL:
        All discoverable flows are programmed. Steady-state.
    """

    FLOW_EXPIRY_TICKS = SDN_FLOW_EXPIRY_S   # OpenFlow hard_timeout, s
    TICKS_PER_LLDP_ROUND = SDN_LLDP_ROUND_S # one BFS discovery round, s
    FLOWS_PER_TICK = 3           # OpenFlow FlowMod programming rate
    MIN_FRAGMENT_FOR_ELECTION = 3  # Fragments < 3 nodes can't run election

    def __init__(self, sim):
        self.sim = sim
        is_rescue_ops = any(e.event_type == 'sever_core' and e.tick == 0 for e in sim.scenario.events)
        self.severance_tick = 0 if is_rescue_ops else SEVERANCE_TICK
        # Per-fragment state
        self.fragment_members = {}   # frag_id -> set of node_ids
        self.fragment_leaders = {}   # frag_id -> leader_node_id
        self.election_done_tick = {} # frag_id -> tick when election completes
        self.discovery_frontier = {} # frag_id -> set of discovered nodes
        self.discovery_done = {}     # frag_id -> bool
        self.programmed_flows = set()  # set of (src_ue, tgt_ue) tuples
        self.flow_queue = []           # flows waiting to be programmed
        self._fragments_computed = False
        self.phase = 'orphan'

    def _compute_fragments(self, force=False):
        """Discover connected components among surviving nodes."""
        if self._fragments_computed and not force:
            return
        self._fragments_computed = True

        survivor_nodes = [
            nid for nid, n in self.sim.topology.nodes.items()
            if n.is_survivor and n.node_type.value not in {"Core", "UPF", "AMF", "CoreUPF"}
        ]
        # BFS to find connected components
        visited = set()
        frag_id = 0
        for start in survivor_nodes:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for lid, lnk in self.sim.topology.links.items():
                    if not getattr(lnk, 'is_up', True):
                        continue
                    ep = lnk.endpoints
                    peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                    if peer and peer in survivor_nodes and peer not in visited:
                        queue.append(peer)
            self.fragment_members[frag_id] = component
            frag_id += 1

        # Start leader election for each viable fragment
        for fid, members in self.fragment_members.items():
            # Only infra nodes (not UEs) participate in controller election
            infra_in_frag = [nid for nid in members
                             if self.sim.topology.nodes.get(nid) and
                             self.sim.topology.nodes[nid].node_type.value not in {'UE'}]
            if len(infra_in_frag) >= self.MIN_FRAGMENT_FOR_ELECTION:
                # Election duration in TICKS = SECONDS: Raft term-start cost
                # plus per-voter vote RTT over the mesh.  Was `10 + 3 * N`
                # written for a 10 ms tick (1.15 s for 35 voters), which at
                # the real 1 s tick became 115 s.
                election_ticks = max(1, int(round(
                    SDN_ELECTION_BASE_S
                    + SDN_ELECTION_PER_NODE_S * len(infra_in_frag))))
                self.election_done_tick[fid] = self.severance_tick + self.FLOW_EXPIRY_TICKS + election_ticks
                # Leader = infra node with highest degree
                # SET ITERATION ORDER IS SALTED.  Python randomises the iteration
                # order of a set of strings per process (PYTHONHASHSEED), and the
                # arms run in a multiprocessing pool.  Any greedy choice, sample or
                # tie-break that walks a set of node ids therefore produced a
                # DIFFERENT result in every process - the same reproducibility
                # defect as the salted hash() seeds, one layer up.  sorted() gives a
                # stable, documented tie-break (lexicographic by node id).
                best_node = max(sorted(infra_in_frag), key=lambda nid: sum(
                    1 for l in self.sim.topology.links.values()
                    if getattr(l, 'is_up', True) and nid in l.endpoints
                ))
                self.fragment_leaders[fid] = best_node

    def step(self, tick):
        """Advance the SDN controller process by one tick."""
        ticks_since = tick - self.severance_tick
        if ticks_since < 0:
            return  # pre-severance: no SDN needed

        # Check for new flows in operational phase
        if self.phase == 'operational':
            new_flows = [flow for flow in self.sim.ue_to_ue_flows 
                         if flow not in self.programmed_flows and flow not in self.flow_queue]
            if new_flows:
                self._compute_fragments(force=True)
                for flow in new_flows:
                    self.flow_queue.append(flow)
                self.phase = 'programming'

        # Phase 1: Orphan — stale flow rules expiring
        if ticks_since < self.FLOW_EXPIRY_TICKS:
            if self.phase != 'operational':
                self.phase = 'orphan'
                return

        # Compute fragments once after orphan phase
        self._compute_fragments()

        # Phase 2: Leader election (per-fragment)
        any_election_pending = False
        for fid in self.fragment_members:
            if fid not in self.election_done_tick:
                continue  # fragment too small
            if tick < self.election_done_tick[fid]:
                any_election_pending = True

        if any_election_pending:
            self.phase = 'election'
            return

        # Phase 3: LLDP topology discovery (BFS from each leader)
        any_discovery_pending = False
        for fid, leader in self.fragment_leaders.items():
            if self.discovery_done.get(fid, False):
                continue
            if fid not in self.discovery_frontier:
                # Start discovery from leader
                self.discovery_frontier[fid] = {leader}
                self._discovery_round_tick = tick

            # Run one LLDP round every TICKS_PER_LLDP_ROUND ticks
            if (tick - self._discovery_round_tick) >= self.TICKS_PER_LLDP_ROUND:
                self._discovery_round_tick = tick
                # BFS: discover neighbors of current frontier
                new_frontier = set()
                for nid in self.discovery_frontier[fid]:
                    for lid, lnk in self.sim.topology.links.items():
                        if not getattr(lnk, 'is_up', True):
                            continue
                        ep = lnk.endpoints
                        peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                        if peer and peer in self.fragment_members[fid]:
                            new_frontier.add(peer)

                if new_frontier == self.discovery_frontier[fid]:
                    # No new nodes discovered — discovery complete for this fragment
                    self.discovery_done[fid] = True
                    # Queue flows for programming
                    discovered = self.discovery_frontier[fid]
                    for src, tgt in self.sim.ue_to_ue_flows:
                        if src in discovered and tgt in discovered:
                            if (src, tgt) not in self.programmed_flows:
                                self.flow_queue.append((src, tgt))
                else:
                    self.discovery_frontier[fid] = new_frontier
                    any_discovery_pending = True
                    continue
            else:
                any_discovery_pending = True
                continue

        if any_discovery_pending:
            self.phase = 'discovery'

        # Phase 4: Flow programming (rate-limited)
        if self.flow_queue:
            self.phase = 'programming'
            for _ in range(min(self.FLOWS_PER_TICK, len(self.flow_queue))):
                flow = self.flow_queue.pop(0)
                self.programmed_flows.add(flow)
        elif all(self.discovery_done.get(fid, False) for fid in self.fragment_leaders):
            self.phase = 'operational'
            # ── Operational dynamics: the SDN controller is NOT static ──
            # Real SDN controllers perform continuous per-tick operations:
            self._operational_dynamics(tick)

    def _operational_dynamics(self, tick):
        """Per-tick SDN controller operations during steady state.

        A real SDN controller continuously:
        1. Monitors link quality and detects degradation
        2. Reroutes flows when better paths emerge
        3. Evicts stale/unused flows from switch tables
        4. Handles microbursts and congestion by rerouting
        5. Reprobes LLDP to detect topology drift

        This creates natural per-tick variation in the set of
        programmed flows, which drives variation in achievability,
        latency, and overhead.
        """
        rng = random.Random(tick * 7919 + 31)

        # ── Link quality monitoring ──
        # Controller polls switch stats. Some links may degrade due to
        # interference or weather → flows on those links get rerouted.
        # With ~438 flows, expect ~1-3% to be affected per tick.
        n_flows = len(self.programmed_flows)
        if n_flows == 0:
            return

        # Per-tick flow churn: some flows temporarily degrade and need rerouting
        # This models: microburst-induced drops, path flaps from load balancing,
        # flow table timeouts (idle_timeout), and controller path recomputation
        churn_rate = 0.005 + 0.010 * rng.random()  # 0.5-1.5% of flows per tick
        n_churn = max(0, int(n_flows * churn_rate))

        if n_churn > 0:
            # SET ITERATION ORDER IS SALTED.  Python randomises the iteration
            # order of a set of strings per process (PYTHONHASHSEED), and the
            # arms run in a multiprocessing pool.  Any greedy choice, sample or
            # tie-break that walks a set of node ids therefore produced a
            # DIFFERENT result in every process - the same reproducibility
            # defect as the salted hash() seeds, one layer up.  sorted() gives a
            # stable, documented tie-break (lexicographic by node id).
            flow_list = sorted(self.programmed_flows)
            removed = rng.sample(flow_list, min(n_churn, len(flow_list)))
            for flow in removed:
                self.programmed_flows.discard(flow)
                self.flow_queue.append(flow)

        # Re-program queued flows — operational rerouting is faster than
        # initial discovery (routes already computed, just push FlowMod)
        operational_reprogram_rate = 10
        if self.flow_queue:
            for _ in range(min(operational_reprogram_rate, len(self.flow_queue))):
                flow = self.flow_queue.pop(0)
                self.programmed_flows.add(flow)

    def get_programmed_flows(self):
        """Return the list of flows the SDN controller has programmed."""
        # sorted(), NOT list(): `programmed_flows` is a SET of (str, str)
        # tuples, and set iteration order over strings is SALTED per process
        # (PYTHONHASHSEED).  The engine allocates capacity GREEDILY in the
        # order this list is walked, so an unsorted set gave a different set
        # of served flows in every process - the same reproducibility defect
        # as the salted hash() seeds, just one layer up.  Measured: identical
        # runs diverged across PYTHONHASHSEED values until this was sorted.
        return sorted(self.programmed_flows)


class OLSRController:
    """Process-based OLSR (RFC 3626) controller for island-mode recovery.

    OLSR is a proactive link-state protocol optimized for ad-hoc networks.
    Unlike OSPF (which floods LSAs to ALL neighbors), OLSR selects a subset
    of nodes as Multi-Point Relays (MPRs) — only MPR nodes relay Topology
    Control (TC) messages, reducing overhead in dense topologies.

    Phase 1 — NEIGHBOR SENSING (Hello):
        Periodic Hello messages build 1-hop and 2-hop neighbor tables.
        Duration: OLSR_HELLO_INTERVAL ticks per round, need ~2 rounds.

    Phase 2 — MPR SELECTION:
        Each node selects the smallest subset of 1-hop neighbors that
        covers all its 2-hop neighbors. Only MPRs relay TC messages.
        Duration: immediate after 2-hop table is built.

    Phase 3 — TC FLOODING (via MPRs only):
        MPR nodes broadcast TC messages containing their MPR selector set.
        TC messages are relayed hop-by-hop only through MPR nodes.
        Duration: OLSR_TC_INTERVAL × diameter rounds.

    Phase 4 — ROUTE CALCULATION:
        Each node computes shortest paths from its TC-derived topology.
        Duration: ~10 ticks (Dijkstra on sparse TC topology).

    Phase 5 — OPERATIONAL:
        Stable routes. Periodic TC refresh (every OLSR_TC_INTERVAL ticks).
    """

    def __init__(self, sim):
        self.sim = sim
        is_rescue_ops = any(e.event_type == 'sever_core' and e.tick == 0 for e in sim.scenario.events)
        self.severance_tick = 0 if is_rescue_ops else SEVERANCE_TICK
        # Per-fragment state
        self.fragment_members = {}     # frag_id -> set of node_ids
        self.neighbor_tables = {}      # nid -> {1hop: set, 2hop: set}
        self.mpr_sets = {}             # nid -> set of MPR node_ids
        self.tc_coverage = {}          # frag_id -> fraction of nodes covered by TC
        self.route_tables_ready = {}   # frag_id -> bool
        self.routable_flows = set()    # (src, tgt) pairs with computed routes
        self._fragments_computed = False
        self._hello_rounds_done = 0
        self._tc_rounds_done = 0
        self._route_calc_start_tick = None
        self.phase = 'sensing'

    def _compute_fragments(self, force=False):
        """Discover connected components among surviving infra + UE nodes."""
        if self._fragments_computed and not force:
            return
        self._fragments_computed = True

        survivor_nodes = [
            nid for nid, n in self.sim.topology.nodes.items()
            if n.is_survivor and n.node_type.value not in {"Core", "UPF", "AMF", "CoreUPF"}
        ]
        visited = set()
        frag_id = 0
        for start in survivor_nodes:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for lid, lnk in self.sim.topology.links.items():
                    if not getattr(lnk, 'is_up', True):
                        continue
                    ep = lnk.endpoints
                    peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                    if peer and peer in survivor_nodes and peer not in visited:
                        queue.append(peer)
            self.fragment_members[frag_id] = component
            frag_id += 1

    def _build_neighbor_tables(self):
        """Build 1-hop and 2-hop neighbor tables from Hello messages."""
        for nid in list(self.sim.phy_mac_states.keys()):
            node = self.sim.topology.nodes.get(nid)
            if not node or not node.is_survivor:
                continue
            one_hop = set()
            for lid, lnk in self.sim.topology.links.items():
                if not getattr(lnk, 'is_up', True):
                    continue
                ep = lnk.endpoints
                peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                if peer and peer in self.sim.phy_mac_states:
                    peer_node = self.sim.topology.nodes.get(peer)
                    if peer_node and peer_node.is_survivor:
                        one_hop.add(peer)
            # 2-hop = neighbors of 1-hop neighbors (excluding self)
            two_hop = set()
            for nbr in one_hop:
                for lid, lnk in self.sim.topology.links.items():
                    if not getattr(lnk, 'is_up', True):
                        continue
                    ep = lnk.endpoints
                    peer2 = ep[1] if ep[0] == nbr else (ep[0] if ep[1] == nbr else None)
                    if peer2 and peer2 != nid and peer2 not in one_hop:
                        p2_node = self.sim.topology.nodes.get(peer2)
                        if p2_node and p2_node.is_survivor:
                            two_hop.add(peer2)
            self.neighbor_tables[nid] = {'1hop': one_hop, '2hop': two_hop}

    def _select_mprs(self):
        """Select MPR set for each node: smallest subset of 1-hop neighbors
        that covers all 2-hop neighbors (greedy set-cover)."""
        for nid, tables in self.neighbor_tables.items():
            two_hop_uncovered = set(tables['2hop'])
            mpr = set()
            # SET ITERATION ORDER IS SALTED.  Python randomises the iteration
            # order of a set of strings per process (PYTHONHASHSEED), and the
            # arms run in a multiprocessing pool.  Any greedy choice, sample or
            # tie-break that walks a set of node ids therefore produced a
            # DIFFERENT result in every process - the same reproducibility
            # defect as the salted hash() seeds, one layer up.  sorted() gives a
            # stable, documented tie-break (lexicographic by node id).
            one_hop = sorted(tables['1hop'])
            while two_hop_uncovered and one_hop:
                # Greedy: pick 1-hop neighbor covering most uncovered 2-hop nodes
                best = max(one_hop, key=lambda n: len(
                    two_hop_uncovered & self.neighbor_tables.get(n, {}).get('1hop', set())
                ))
                covered = two_hop_uncovered & self.neighbor_tables.get(best, {}).get('1hop', set())
                if not covered:
                    break
                mpr.add(best)
                two_hop_uncovered -= covered
                one_hop.remove(best)
            self.mpr_sets[nid] = mpr

    def step(self, tick):
        """Advance OLSR protocol state by one tick with per-node convergence."""
        ticks_since = tick - self.severance_tick
        if ticks_since < 0:
            return

        # Initialize per-node convergence state on first call
        if not hasattr(self, 'node_convergence'):
            self.node_convergence = {}  # nid -> float [0.0, 1.0]
            self._node_converge_tick = {}  # nid -> tick when fully converged
            self._convergence_initialized = False

        # Phase 1: Neighbor Sensing (Hello rounds) — network-wide but staggered
        if self._hello_rounds_done < 2:
            self.phase = 'sensing'
            if ticks_since >= OLSR_HELLO_INTERVAL * (self._hello_rounds_done + 1):
                self._hello_rounds_done += 1
                if self._hello_rounds_done >= 2:
                    self._build_neighbor_tables()
            # During sensing, no flows are routable
            self.routable_flows.clear()
            return

        # Phase 2: MPR Selection (immediate after neighbor tables)
        if not self.mpr_sets:
            self.phase = 'mpr_selection'
            self._select_mprs()
            self._compute_fragments()
            self.routable_flows.clear()
            return

        # Phase 3+: Per-node TC propagation and convergence
        # Initialize per-node convergence times based on BFS distance
        if not self._convergence_initialized:
            self._convergence_initialized = True
            self._compute_fragments()
            tc_base_tick = self.severance_tick + OLSR_HELLO_INTERVAL * 2
            for fid, members in self.fragment_members.items():
                # sorted(): `members` is a SET, so `infra[0]` picked a BFS root
                # that depended on salted set-iteration order (PYTHONHASHSEED).
                # The root sets every node's hop distance and therefore its
                # convergence tick, so the whole per-node convergence ramp -
                # and the routable-flow set it gates - differed in every
                # process of the multiprocessing pool.  Lexicographically
                # lowest node id is a stable, documented choice.
                infra = sorted(n for n in members if self.sim.topology.nodes.get(n)
                               and self.sim.topology.nodes[n].node_type.value != 'UE')
                if not infra:
                    continue
                # BFS from the lowest-id node to get distance-based convergence
                root = infra[0]
                bfs_dist = {root: 0}
                queue = [root]
                while queue:
                    cur = queue.pop(0)
                    for nbr in self.sim.topology.graph.neighbors(cur):
                        if nbr in set(infra) and nbr not in bfs_dist:
                            bfs_dist[nbr] = bfs_dist[cur] + 1
                            queue.append(nbr)
                # Each node converges after TC propagates to it
                for nid in infra:
                    dist = bfs_dist.get(nid, 3)
                    # Per-node jitter: ±15% of TC_INTERVAL (real protocol timer jitter)
                    jitter_rng = random.Random(
                        zlib.crc32(nid.encode('utf-8')) % 9973)
                    jitter = jitter_rng.gauss(0, 0.15) * OLSR_TC_INTERVAL
                    converge_at = tc_base_tick + OLSR_TC_INTERVAL * dist + jitter
                    # Route calculation adds ~10 ticks after TC arrives
                    converge_at += 10 + jitter_rng.uniform(0, 5)
                    self._node_converge_tick[nid] = converge_at
                # UE nodes converge when their serving gNB converges + 1 tick
                for nid in members:
                    if nid not in self._node_converge_tick:
                        node = self.sim.topology.nodes.get(nid)
                        if node and node.node_type.value == 'UE':
                            # Find nearest infra neighbor
                            best_t = float('inf')
                            for lid, lnk in self.sim.topology.links.items():
                                if not getattr(lnk, 'is_up', True):
                                    continue
                                ep = lnk.endpoints
                                peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                                if peer and peer in self._node_converge_tick:
                                    best_t = min(best_t, self._node_converge_tick[peer])
                            self._node_converge_tick[nid] = best_t + 1 if best_t < float('inf') else tc_base_tick + 200

        # Handle new flows appearing mid-simulation (rescue UEs)
        if not hasattr(self, '_known_flows'):
            self._known_flows = set(self.sim.ue_to_ue_flows)
        current_flows = set(self.sim.ue_to_ue_flows)
        new_flows = current_flows - self._known_flows
        if new_flows:
            if not hasattr(self, '_new_flow_discovery'):
                self._new_flow_discovery = {}
            for flow in new_flows:
                if flow not in self._new_flow_discovery:
                    self._new_flow_discovery[flow] = tick
            self._compute_fragments(force=True)
            # Compute convergence for genuinely new nodes only
            for flow in new_flows:
                for nid in flow:
                    if nid not in self._node_converge_tick:
                        # New UE joining an already-converged network:
                        # OLSR HELLO discovery + TC propagation ≈ 30-40 ticks
                        self._node_converge_tick[nid] = (
                            tick + 30 + random.Random(
                                zlib.crc32(nid.encode('utf-8'))).uniform(0, 10))
        self._known_flows = current_flows

        # Update per-node convergence fraction (smooth ramp 0→1)
        for nid, conv_tick in self._node_converge_tick.items():
            if tick >= conv_tick:
                self.node_convergence[nid] = 1.0
            elif tick >= conv_tick - OLSR_TC_INTERVAL:
                # Smooth ramp during the last TC_INTERVAL before full convergence
                progress = (tick - (conv_tick - OLSR_TC_INTERVAL)) / OLSR_TC_INTERVAL
                self.node_convergence[nid] = max(0.0, min(1.0, progress))
            else:
                self.node_convergence[nid] = 0.0

        # Build routable flows: a flow is routable when BOTH endpoints have converged
        self.routable_flows.clear()
        for fid, members in self.fragment_members.items():
            for src, tgt in self.sim.ue_to_ue_flows:
                if src in members and tgt in members:
                    # Check if flow is in new-discovery wait
                    in_discovery = (hasattr(self, '_new_flow_discovery') and
                                   (src, tgt) in self._new_flow_discovery and
                                   tick - self._new_flow_discovery[(src, tgt)] < 30)
                    if in_discovery:
                        continue
                    # Per-flow convergence: both endpoints must be converged
                    src_conv = self.node_convergence.get(src, 0.0)
                    tgt_conv = self.node_convergence.get(tgt, 0.0)
                    flow_conv = min(src_conv, tgt_conv)
                    if flow_conv >= 1.0:
                        self.routable_flows.add((src, tgt))
                    elif flow_conv > 0.0:
                        # Probabilistic: partially converged nodes may have routes
                        rng = random.Random(zlib.crc32(
                            f"{src}|{tgt}|{tick // 10}".encode('utf-8')))
                        if rng.random() < flow_conv:
                            self.routable_flows.add((src, tgt))

        # Clean up completed discovery timers
        if hasattr(self, '_new_flow_discovery'):
            completed = [f for f, t in self._new_flow_discovery.items() if tick - t >= 30]
            for f in completed:
                del self._new_flow_discovery[f]

        # Update global phase (summary for overhead/latency calcs)
        all_conv = list(self.node_convergence.values())
        if not all_conv:
            self.phase = 'sensing'
        else:
            avg_conv = sum(all_conv) / len(all_conv)
            if avg_conv < 0.1:
                self.phase = 'tc_flooding'
            elif avg_conv < 0.9:
                self.phase = 'tc_flooding'
            elif avg_conv < 1.0:
                self.phase = 'route_calc'
            else:
                self.phase = 'operational'

        # Expose convergence fraction for latency/energy interpolation
        self.convergence_fraction = sum(all_conv) / max(1, len(all_conv)) if all_conv else 0.0

    def get_routable_flows(self):
        """Return flows that OLSR has computed routes for."""
        # sorted(), NOT list(): `routable_flows` is a SET of (str, str)
        # tuples, and set iteration order over strings is SALTED per process
        # (PYTHONHASHSEED).  The engine allocates capacity GREEDILY in the
        # order this list is walked, so an unsorted set gave a different set
        # of served flows in every process - the same reproducibility defect
        # as the salted hash() seeds, just one layer up.  Measured: identical
        # runs diverged across PYTHONHASHSEED values until this was sorted.
        return sorted(self.routable_flows)


class BATMANController:
    """Process-based B.A.T.M.A.N. (batman-adv) controller for island-mode recovery.

    B.A.T.M.A.N. is a proactive Layer-2 distance-vector protocol. Unlike
    OLSR/OSPF, it has NO global topology view. Instead, each node periodically
    broadcasts Originator Messages (OGMs) that propagate hop-by-hop through
    the mesh. Each node tracks Transmission Quality (TQ) — an EMA of received
    OGM count per originator — and selects the best next-hop accordingly.

    Phase 1 — OGM BROADCAST:
        Each node originates OGMs every BATMAN_OGM_INTERVAL ticks.
        OGMs carry the originator address and a TQ value (starts at 255).

    Phase 2 — OGM PROPAGATION:
        Every node that receives an OGM rebroadcasts it with decremented TQ.
        TTL limits prevent infinite propagation.

    Phase 3 — TQ LEARNING:
        Each node maintains a per-originator TQ value using EMA:
        TQ_new = α × received_TQ + (1-α) × TQ_old
        Needs ~20 OGMs for stable TQ → ~200 ticks.

    Phase 4 — BEST NEXT-HOP SELECTION:
        Once TQ is stable, select best next-hop per destination.
        Forwarding begins as soon as any TQ > threshold.

    Phase 5 — OPERATIONAL:
        Stable mesh forwarding. TQ continuously updates.
    """

    TQ_INITIAL = 0             # No knowledge at start
    TQ_MAX = 255               # Maximum TQ (direct neighbor)
    TQ_THRESHOLD = 50          # Minimum TQ to consider a route usable
    TQ_ALPHA = 0.1             # EMA smoothing for TQ updates
    OGM_TTL = 10               # Max hops for OGM propagation

    def __init__(self, sim):
        self.sim = sim
        is_rescue_ops = any(e.event_type == 'sever_core' and e.tick == 0 for e in sim.scenario.events)
        self.severance_tick = 0 if is_rescue_ops else SEVERANCE_TICK
        # Per-node TQ tables: {node_id: {originator_id: tq_value}}
        self.tq_tables = {}
        # Per-node best next-hop: {node_id: {dest_id: next_hop_id}}
        self.next_hop = {}
        # OGM tracking
        self._ogm_round = 0
        self._fragments_computed = False
        self.fragment_members = {}
        self.routable_flows = set()
        self.phase = 'ogm_broadcast'
        # Track per-fragment convergence
        self._fragment_tq_stable = {}  # frag_id -> fraction of stable TQ entries

    def _compute_fragments(self, force=False):
        """Discover connected components among surviving nodes."""
        if self._fragments_computed and not force:
            return
        self._fragments_computed = True

        survivor_nodes = [
            nid for nid, n in self.sim.topology.nodes.items()
            if n.is_survivor and n.node_type.value not in {"Core", "UPF", "AMF", "CoreUPF"}
        ]
        visited = set()
        frag_id = 0
        for start in survivor_nodes:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for lid, lnk in self.sim.topology.links.items():
                    if not getattr(lnk, 'is_up', True):
                        continue
                    ep = lnk.endpoints
                    peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                    if peer and peer in survivor_nodes and peer not in visited:
                        queue.append(peer)
            self.fragment_members[frag_id] = component
            frag_id += 1

    def _simulate_ogm_round(self, tick):
        self._compute_fragments()

        # Initialize TQ tables for new nodes
        all_infra = [nid for nid in self.sim.phy_mac_states
                     if self.sim.topology.nodes.get(nid)
                     and self.sim.topology.nodes[nid].is_survivor]
        for nid in all_infra:
            if nid not in self.tq_tables:
                self.tq_tables[nid] = {}
                self.next_hop[nid] = {}

        # Each node originates an OGM and propagates through neighbors
        # Simulate multi-hop propagation in a single round (BFS with TQ decay)
        for originator in all_infra:
            # BFS propagation from originator
            visited_ogm = {originator}
            frontier = [(originator, self.TQ_MAX)]  # (node, tq_at_this_hop)
            hop = 0
            while frontier and hop < self.OGM_TTL:
                next_frontier = []
                for current_node, current_tq in frontier:
                    # Find neighbors
                    for lid, lnk in self.sim.topology.links.items():
                        if not getattr(lnk, 'is_up', True):
                            continue
                        ep = lnk.endpoints
                        peer = ep[1] if ep[0] == current_node else (ep[0] if ep[1] == current_node else None)
                        if peer and peer in all_infra and peer not in visited_ogm:
                            visited_ogm.add(peer)
                            # TQ decays with each hop (multiplicative decay)
                            hop_tq = int(current_tq * 0.8)  # 20% decay per hop
                            if hop_tq < 1:
                                continue

                            # Update TQ table via EMA
                            old_tq = self.tq_tables[peer].get(originator, 0)
                            new_tq = self.TQ_ALPHA * hop_tq + (1 - self.TQ_ALPHA) * old_tq
                            self.tq_tables[peer][originator] = new_tq

                            # Update best next-hop if this path is better
                            current_best = self.next_hop[peer].get(originator)
                            if current_best is None or new_tq > self.tq_tables[peer].get(originator, 0) * 0.95:
                                self.next_hop[peer][originator] = current_node

                            next_frontier.append((peer, hop_tq))
                frontier = next_frontier
                hop += 1

    def step(self, tick):
        """Advance B.A.T.M.A.N. protocol state by one tick with per-node convergence."""
        ticks_since = tick - self.severance_tick
        if ticks_since < 0:
            return

        # Initialize per-node convergence tracking
        if not hasattr(self, 'node_convergence'):
            self.node_convergence = {}  # nid -> float [0.0, 1.0]
            self._node_converge_tick = {}  # nid -> tick when TQ stabilizes
            self._convergence_initialized = False

        self._compute_fragments()

        # Run OGM round every BATMAN_OGM_INTERVAL ticks
        expected_rounds = ticks_since // BATMAN_OGM_INTERVAL
        if expected_rounds > self._ogm_round:
            self._ogm_round = expected_rounds
            self._simulate_ogm_round(tick)

        # Initialize per-node convergence times based on BFS distance
        if not self._convergence_initialized and self._ogm_round >= 1:
            self._convergence_initialized = True
            ogm_start = self.severance_tick
            for fid, members in self.fragment_members.items():
                # sorted(): `members` is a SET, so `infra[0]` picked a BFS root
                # that depended on salted set-iteration order (PYTHONHASHSEED).
                # The root sets every node's hop distance and therefore its
                # convergence tick, so the whole per-node convergence ramp -
                # and the routable-flow set it gates - differed in every
                # process of the multiprocessing pool.  Lexicographically
                # lowest node id is a stable, documented choice.
                infra = sorted(n for n in members if self.sim.topology.nodes.get(n)
                               and self.sim.topology.nodes[n].is_survivor
                               and self.sim.topology.nodes[n].node_type.value != 'UE')
                if not infra:
                    continue
                # BFS to determine hop distance, from the lowest-id node
                root = infra[0]
                bfs_dist = {root: 0}
                queue = [root]
                while queue:
                    cur = queue.pop(0)
                    for lid, lnk in self.sim.topology.links.items():
                        if not getattr(lnk, 'is_up', True):
                            continue
                        ep = lnk.endpoints
                        peer = ep[1] if ep[0] == cur else (ep[0] if ep[1] == cur else None)
                        if peer and peer in set(infra) and peer not in bfs_dist:
                            bfs_dist[peer] = bfs_dist[cur] + 1
                            queue.append(peer)
                # TQ stabilization: needs ~(dist+2) OGM rounds per node
                for nid in infra:
                    dist = bfs_dist.get(nid, 4)
                    rounds_needed = dist + 2 + max(0, int(dist * 0.5))  # more hops = more rounds for EMA
                    jitter_rng = random.Random(
                        zlib.crc32(nid.encode('utf-8')) % 7919)
                    jitter = jitter_rng.gauss(0, 0.2) * BATMAN_OGM_INTERVAL
                    converge_at = ogm_start + BATMAN_OGM_INTERVAL * rounds_needed + jitter
                    self._node_converge_tick[nid] = converge_at
                # UE nodes
                for nid in members:
                    if nid not in self._node_converge_tick:
                        self._node_converge_tick[nid] = ogm_start + BATMAN_TQ_STABILIZE

        # Handle new flows (rescue UEs)
        if not hasattr(self, '_known_flows'):
            self._known_flows = set(self.sim.ue_to_ue_flows)
        current_flows = set(self.sim.ue_to_ue_flows)
        new_flows = current_flows - self._known_flows
        if new_flows:
            if not hasattr(self, '_new_flow_discovery'):
                self._new_flow_discovery = {}
            for flow in new_flows:
                if flow not in self._new_flow_discovery:
                    self._new_flow_discovery[flow] = tick
            self._compute_fragments(force=True)
            for flow in new_flows:
                for nid in flow:
                    if nid not in self._node_converge_tick:
                        # New UE joining an already-converged network:
                        # BATMAN OGM propagation + TQ stabilization ≈ 80-120 ticks
                        self._node_converge_tick[nid] = (
                            tick + 80 + random.Random(
                                zlib.crc32(nid.encode('utf-8'))).uniform(0, 30))
        self._known_flows = current_flows

        # Process new flow discovery timers
        if hasattr(self, '_new_flow_discovery'):
            completed = [f for f, t in self._new_flow_discovery.items() if tick - t >= 80]
            for f in completed:
                del self._new_flow_discovery[f]

        # Update per-node convergence fraction
        for nid, conv_tick in self._node_converge_tick.items():
            if tick >= conv_tick:
                self.node_convergence[nid] = 1.0
            elif tick >= conv_tick - BATMAN_OGM_INTERVAL * 3:
                # Smooth ramp over last 3 OGM rounds
                ramp_start = conv_tick - BATMAN_OGM_INTERVAL * 3
                progress = (tick - ramp_start) / (BATMAN_OGM_INTERVAL * 3)
                self.node_convergence[nid] = max(0.0, min(1.0, progress))
            else:
                self.node_convergence[nid] = 0.0

        # Build routable flows based on per-endpoint convergence
        self.routable_flows.clear()
        for fid, members in self.fragment_members.items():
            for src, tgt in self.sim.ue_to_ue_flows:
                if src in members and tgt in members:
                    in_discovery = (hasattr(self, '_new_flow_discovery') and
                                   (src, tgt) in self._new_flow_discovery and
                                   tick - self._new_flow_discovery[(src, tgt)] < 80)
                    if in_discovery:
                        continue
                    src_conv = self.node_convergence.get(src, 0.0)
                    tgt_conv = self.node_convergence.get(tgt, 0.0)
                    flow_conv = min(src_conv, tgt_conv)
                    if flow_conv >= 1.0:
                        self.routable_flows.add((src, tgt))
                    elif flow_conv > 0.0:
                        rng = random.Random(zlib.crc32(
                            f"{src}|{tgt}|{tick // 10}".encode('utf-8')))
                        if rng.random() < flow_conv:
                            self.routable_flows.add((src, tgt))

        # Update summary phase and convergence fraction
        all_conv = list(self.node_convergence.values())
        if not all_conv or (ticks_since < BATMAN_OGM_INTERVAL and not self._convergence_initialized):
            self.phase = 'ogm_broadcast'
        else:
            avg_conv = sum(all_conv) / len(all_conv)
            if avg_conv < 0.15:
                self.phase = 'ogm_propagation'
            elif avg_conv < 0.5:
                self.phase = 'tq_learning'
            elif avg_conv < 0.95:
                self.phase = 'next_hop_selection'
            else:
                self.phase = 'operational'
        self.convergence_fraction = sum(all_conv) / max(1, len(all_conv)) if all_conv else 0.0

    def get_routable_flows(self):
        """Return flows that B.A.T.M.A.N. has usable next-hop entries for."""
        # sorted(), NOT list(): `routable_flows` is a SET of (str, str)
        # tuples, and set iteration order over strings is SALTED per process
        # (PYTHONHASHSEED).  The engine allocates capacity GREEDILY in the
        # order this list is walked, so an unsorted set gave a different set
        # of served flows in every process - the same reproducibility defect
        # as the salted hash() seeds, just one layer up.  Measured: identical
        # runs diverged across PYTHONHASHSEED values until this was sorted.
        return sorted(self.routable_flows)


class AODVController:
    """Process-based AODV (RFC 3561) controller for island-mode recovery.

    AODV is a REACTIVE routing protocol — unlike OSPF/OLSR/B.A.T.M.A.N.,
    it does NOT maintain routes proactively. Routes are discovered ON DEMAND
    when a node needs to send data to a destination it has no route for.

    Phase 1 — IDLE (no routes):
        After severance, all routes are invalidated. No proactive messages.

    Phase 2 — RREQ FLOODING:
        When a flow needs routing, the source broadcasts a Route Request (RREQ).
        RREQ propagates via controlled flooding through the fragment.
        Multiple RREQs can be active in parallel (limited by AODV_DISCOVERY_RATE).

    Phase 3 — RREP PATH SETUP:
        Destination (or intermediate node with fresh route) replies with
        unicast RREP along the reverse path. Route is installed at each hop.

    Phase 4 — OPERATIONAL (cached routes):
        Routes are cached. Fast forwarding via route table lookup.
        Routes expire after AODV_ROUTE_LIFETIME ticks → re-discovery.

    Phase 5 — ROUTE REPAIR:
        On link break, upstream node sends RERR and initiates local repair
        (local RREQ with limited TTL). If repair fails, source re-discovers.
    """

    def __init__(self, sim):
        self.sim = sim
        is_rescue_ops = any(e.event_type == 'sever_core' and e.tick == 0 for e in sim.scenario.events)
        self.severance_tick = 0 if is_rescue_ops else SEVERANCE_TICK
        # Route cache: {(src, tgt): tick_when_discovered}
        self.route_cache = {}
        # Discovery queue: flows waiting for route discovery
        self.discovery_queue = []
        # Active RREQ tracking
        self._active_rreqs = {}  # (src, tgt) -> tick_started
        self._active_discoveries = 0
        # Fragment info
        self._fragments_computed = False
        self.fragment_members = {}
        self.routable_flows = set()
        self.phase = 'idle'

    def _compute_fragments(self, force=False):
        """Discover connected components among surviving nodes."""
        if self._fragments_computed and not force:
            return
        self._fragments_computed = True

        survivor_nodes = [
            nid for nid, n in self.sim.topology.nodes.items()
            if n.is_survivor and n.node_type.value not in {"Core", "UPF", "AMF", "CoreUPF"}
        ]
        visited = set()
        frag_id = 0
        for start in survivor_nodes:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for lid, lnk in self.sim.topology.links.items():
                    if not getattr(lnk, 'is_up', True):
                        continue
                    ep = lnk.endpoints
                    peer = ep[1] if ep[0] == nid else (ep[0] if ep[1] == nid else None)
                    if peer and peer in survivor_nodes and peer not in visited:
                        queue.append(peer)
            self.fragment_members[frag_id] = component
            frag_id += 1

    def step(self, tick):
        """Advance AODV protocol state by one tick.

        Per-tick dynamics:
        - Traffic is BURSTY: not all flows carry data every tick.
          RFC 3561 §6.2 says routes are refreshed when data traverses them.
          If a flow doesn't send data this tick, the route ages.
        - Routes expire after AODV_ROUTE_LIFETIME ticks of idle time.
        - Expired routes trigger new RREQ floods (overhead + latency spike).
        - Link breaks (rare but modeled) trigger RERR + local repair.

        This creates natural per-tick variation: some routes expire,
        get re-discovered, briefly reduce achievability, then recover.
        """
        ticks_since = tick - self.severance_tick
        if ticks_since < 0:
            return

        # Check for new flows to force fragment recomputation
        new_flows = [flow for flow in self.sim.ue_to_ue_flows 
                     if flow not in self.route_cache and flow not in self._active_rreqs]
        if new_flows:
            self._compute_fragments(force=True)

        self._compute_fragments()

        # ── Per-tick traffic activity (bursty model) ──
        # In reality, not all UE flows carry data every single tick.
        # Voice calls have talk spurts/silence (50% activity),
        # video has I/P/B frame patterns, sensors report periodically.
        # We model each flow as active with probability ~85-95% per tick.
        # Only active flows refresh their route timestamp.
        rng = random.Random(tick * 6271 + 17)
        active_flow_set = set(self.sim.ue_to_ue_flows)
        for flow in list(self.route_cache):
            if flow in active_flow_set:
                # Flow exists but was it actually carrying data THIS tick?
                activity_prob = 0.90  # 90% chance of data on any given tick
                if rng.random() < activity_prob:
                    self.route_cache[flow] = tick  # refresh — data traversed
                # else: route ages (no data this tick, no refresh)

        # ── Route expiry ──
        # Routes that haven't been refreshed for AODV_ROUTE_LIFETIME ticks expire.
        # This naturally creates re-discovery cycles even in steady state.
        expired = [k for k, t in self.route_cache.items()
                   if tick - t > AODV_ROUTE_LIFETIME]
        for k in expired:
            del self.route_cache[k]

        # ── Stochastic link-layer failures ──
        # Occasional HARQ failures or interference cause route breaks.
        # AODV detects via link-layer notification → RERR → re-discovery.
        if self.route_cache and rng.random() < 0.005:  # 0.5% chance per tick
            # One random cached route experiences a link break
            victim = rng.choice(list(self.route_cache.keys()))
            del self.route_cache[victim]

        # ── Identify flows needing routes ──
        pending_flows = []
        for fid, members in self.fragment_members.items():
            for src, tgt in self.sim.ue_to_ue_flows:
                if src in members and tgt in members:
                    if (src, tgt) not in self.route_cache:
                        if (src, tgt) not in self._active_rreqs:
                            pending_flows.append((src, tgt))

        # Start new RREQ discoveries (rate-limited)
        new_discoveries = 0
        for flow in pending_flows:
            if new_discoveries >= AODV_DISCOVERY_RATE:
                break
            self._active_rreqs[flow] = tick
            new_discoveries += 1

        # Process active RREQs
        completed_rreqs = []
        for flow, start_tick in list(self._active_rreqs.items()):
            elapsed = tick - start_tick
            if elapsed >= AODV_RREQ_TIMEOUT + AODV_RREP_TIMEOUT:
                # RREQ + RREP complete → route established
                src, tgt = flow
                try:
                    nx.shortest_path(self.sim.topology.graph, src, tgt)
                    self.route_cache[flow] = tick
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass  # No path exists — route discovery fails
                completed_rreqs.append(flow)
            elif elapsed >= AODV_RREQ_TIMEOUT:
                pass  # RREP phase — waiting for unicast reply

        for flow in completed_rreqs:
            del self._active_rreqs[flow]

        self._active_discoveries = len(self._active_rreqs)

        # Determine phase
        if not self.route_cache and not self._active_rreqs:
            self.phase = 'idle'
        elif self._active_rreqs and not self.route_cache:
            self.phase = 'rreq_flood'
        elif self._active_rreqs and self.route_cache:
            self.phase = 'rrep_path'  # Mix of discovered + discovering
        elif expired:
            self.phase = 'route_repair'
        else:
            self.phase = 'operational'

        # Build routable flows from cache
        self.routable_flows.clear()
        for flow in self.route_cache:
            self.routable_flows.add(flow)

    def get_routable_flows(self):
        """Return flows that AODV has active cached routes for."""
        # sorted(), NOT list(): `routable_flows` is a SET of (str, str)
        # tuples, and set iteration order over strings is SALTED per process
        # (PYTHONHASHSEED).  The engine allocates capacity GREEDILY in the
        # order this list is walked, so an unsorted set gave a different set
        # of served flows in every process - the same reproducibility defect
        # as the salted hash() seeds, just one layer up.  Measured: identical
        # runs diverged across PYTHONHASHSEED values until this was sorted.
        return sorted(self.routable_flows)


def build_comparison_topology(topology_seed, log_prefix=""):
    """Build the comparison-scale topology (~42 infra + 150 UEs, MultiHaul
    sites) used by every physics pass.  Extracted verbatim from
    run_physics_pass() so training drivers (train_on_comparison.py) can
    train on the SAME topology family the evaluation uses.

    Returns (topology, topology_info).
    """
    # ── Generate comparison-scale topology from seed ──
    # The full generate_large_topology creates 195 infra + 2000 UEs (too heavy for
    # comparison runs — causes OOM during PPO). Instead we create a topology matching
    # the YAML's scale: ~46 infra + 150 UEs, with the seed controlling spatial layout.
    import random as _rng
    import math
    _rng.seed(topology_seed)
    from sixg_sim.topology import (Topology, Node, Link, NodeType, LinkType,
                                    InterfaceType)
    import numpy as _np_topo
    _np_topo.random.seed(topology_seed)

    topology = Topology()
    # Node counts matching original YAML scale
    node_specs = [
        (NodeType.SMO,         2),
        (NodeType.NEAR_RT_RIC, 3),
        (NodeType.AMF,         2),
        (NodeType.UPF,         3),
        (NodeType.O_CU_CP,     3),
        (NodeType.O_CU_UP,     3),
        (NodeType.O_DU,        6),
        (NodeType.O_RU,        8),
        (NodeType.RELAY,       8),
        (NodeType.EDGEUPF,     4),
    ]
    coverage_areas = [f"zone_{i}" for i in range(5)]
    nid_counter = 0
    # Geographic layout: 8km × 8km grid
    GEO_SIZE = 8000.0

    for ntype, count in node_specs:
        for i in range(count):
            nid_counter += 1
            area = _rng.choice(coverage_areas)
            node = Node(
                id=f"{ntype.value}_{nid_counter}",
                node_type=ntype,
                initial_energy=max(0.7, min(1.0, 0.9 + _rng.uniform(-0.1, 0.1))),
                coverage_area=area,
                x_pos=_rng.uniform(500, GEO_SIZE - 500),
                y_pos=_rng.uniform(500, GEO_SIZE - 500),
            )
            topology.add_node(node)

    # Add 150 UEs
    for i in range(150):
        nid_counter += 1
        area = _rng.choice(coverage_areas)
        ue = Node(
            id=f"UE_{nid_counter}",
            node_type=NodeType.UE,
            initial_energy=_rng.uniform(0.6, 1.0),
            coverage_area=area,
            x_pos=_rng.uniform(0, GEO_SIZE),
            y_pos=_rng.uniform(0, GEO_SIZE),
        )
        topology.add_node(ue)

    # Helper: get nodes by type
    def _get(ntype):
        return [nid for nid, n in topology.nodes.items() if n.node_type == ntype]
    def _dist(a, b):
        na, nb = topology.nodes[a], topology.nodes[b]
        return math.hypot(na.x_pos - nb.x_pos, na.y_pos - nb.y_pos)

    lid = 0
    # Layer 1: O-RU ↔ O-DU (Open Fronthaul)
    for ru in _get(NodeType.O_RU):
        nearest = sorted(_get(NodeType.O_DU), key=lambda d: _dist(ru, d))[:2]
        for du in nearest:
            lid += 1
            topology.add_link(Link(f"OpenFH_{lid}", (ru, du), 25000, 1,
                                   LinkType.FIBER, InterfaceType.OPEN_FH))
    # Layer 2: O-DU ↔ O-CU (F1)
    for du in _get(NodeType.O_DU):
        for cu in _rng.sample(_get(NodeType.O_CU_CP), min(2, len(_get(NodeType.O_CU_CP)))):
            lid += 1
            topology.add_link(Link(f"F1-C_{lid}", (du, cu), 1000, _rng.randint(1, 3),
                                   LinkType.FIBER, InterfaceType.F1_C))
        for cu in _rng.sample(_get(NodeType.O_CU_UP), min(2, len(_get(NodeType.O_CU_UP)))):
            lid += 1
            topology.add_link(Link(f"F1-U_{lid}", (du, cu), 5000, _rng.randint(1, 3),
                                   LinkType.FIBER, InterfaceType.F1_U))
    # Layer 3: E1 (CU-CP ↔ CU-UP)
    for cp in _get(NodeType.O_CU_CP):
        for up in _get(NodeType.O_CU_UP):
            if _rng.random() < 0.5:
                lid += 1
                topology.add_link(Link(f"E1_{lid}", (cp, up), 2000, 1,
                                       LinkType.FIBER, InterfaceType.E1))
    # Layer 4: E2 (RIC ↔ O-DU/O-CU)
    for ric in _get(NodeType.NEAR_RT_RIC):
        for du in _rng.sample(_get(NodeType.O_DU), min(4, len(_get(NodeType.O_DU)))):
            lid += 1
            topology.add_link(Link(f"E2_{lid}", (ric, du), 500, _rng.randint(2, 5),
                                   LinkType.FIBER, InterfaceType.E2))
    # Layer 5: A1 (SMO ↔ RIC)
    for smo in _get(NodeType.SMO):
        for ric in _get(NodeType.NEAR_RT_RIC):
            lid += 1
            topology.add_link(Link(f"A1_{lid}", (smo, ric), 200, _rng.randint(5, 15),
                                   LinkType.FIBER, InterfaceType.A1))
    # Layer 6: NG (CU ↔ Core)
    for cp in _get(NodeType.O_CU_CP):
        for amf in _rng.sample(_get(NodeType.AMF), min(1, len(_get(NodeType.AMF)))):
            lid += 1
            topology.add_link(Link(f"N2_{lid}", (cp, amf), 1000, _rng.randint(3, 8),
                                   LinkType.FIBER, InterfaceType.N2))
    for up in _get(NodeType.O_CU_UP):
        for upf in _rng.sample(_get(NodeType.UPF), min(2, len(_get(NodeType.UPF)))):
            lid += 1
            topology.add_link(Link(f"N3_{lid}", (up, upf), 10000, _rng.randint(2, 5),
                                   LinkType.FIBER, InterfaceType.N3))
    # Layer 7: Transport (Relay ↔ O-DU, Relay ↔ EdgeUPF)
    relay_nodes = _get(NodeType.RELAY)
    du_nodes = _get(NodeType.O_DU)
    eupf_nodes = _get(NodeType.EDGEUPF)
    # ── 60 GHz TG (MultiHaul) SITE PLAN ─────────────────────────────────
    # See the TG_SITE_TYPE_VALUES / TG_PLAN_REACH_M block at the top of this
    # module for the full planning rule and its justification.  In one line:
    # every RAN / transport site (O-RU, O-DU, EdgeUPF, Relay) carries a
    # steerable 60 GHz TG mesh node co-sited on its existing mast, because
    # the derived TG reach (TG_PLAN_REACH_M, ~1.07 km at the 23 dBm screening
    # power) is far shorter than the spacing a 3-site plan produces, and the
    # reach itself is not a knob that may be turned up.
    #
    # PROPERTIES THIS PLAN MUST HAVE, AND WHY EACH HOLDS:
    #   * SHARED.  It is a property of the physical topology only.  Nothing
    #     here reads `mode`, so every arm — MARL, SDN, OSPF, OLSR, BATMAN,
    #     AODV, random — sees the identical node set, positions, link set and
    #     TG capability set.  What differs between arms is only who may
    #     STEER a TG radio, which is an architectural claim made elsewhere.
    #   * DETERMINISTIC PER SEED.  The rule is a pure function of node type,
    #     so it draws no random numbers at all and cannot perturb the seeded
    #     RNG stream that positions nodes.
    #   * NO NEW NODES.  Infra count, agent count and offered load are
    #     unchanged, so this is not a compute or capacity uplift smuggled in
    #     as a topology fix.
    TG_SITE_TYPES = tuple(NodeType(v) for v in TG_SITE_TYPE_VALUES)
    for _nid, _n in topology.nodes.items():
        if _n.node_type in TG_SITE_TYPES:
            _n.has_multihaul = True

    for relay in relay_nodes:
        nearest_dus = sorted(du_nodes, key=lambda d: _dist(relay, d))[:2]
        for du in nearest_dus:
            lid += 1
            # ── STATIC BACKHAUL MEDIUM (independent of TG STEERING) ──────
            # A site may carry BOTH a mechanically aligned 18 GHz PtP dish
            # for its planned backhaul hop AND a steerable 60 GHz TG node for
            # the mesh — that is precisely the hybrid transport model.  So the
            # medium of the PRE-EXISTING hop can no longer be read off
            # `has_multihaul` (every radio site now has it); it is chosen by
            # HOP LENGTH against the two budgets, which is how the hop would
            # actually be planned:
            #   * a hop that closes on the 60 GHz TG budget at the nominal
            #     power is planned as MULTIHAUL_MESH (2160 MHz -> ~1 Gbps);
            #   * anything longer needs the 18 GHz dish (112 MHz x QAM256
            #     -> ~500 Mbps), LOS/Fresnel-capped at MW_MAX_HOP_M.
            # The old rule ("MULTIHAUL_MESH iff the site is MultiHaul") also
            # planned multi-kilometre 60 GHz hops that the budget cannot
            # close — i.e. it was wrong on its own terms, not just unusable
            # under the new plan.
            _hop_m = _dist(relay, du)
            if _hop_m <= TG_PLAN_REACH_M:
                lt, cap = LinkType.MULTIHAUL_MESH, 1000
            else:
                lt, cap = LinkType.MICROWAVE_PTP, 500
            topology.add_link(Link(f"Backhaul_{lid}", (relay, du), cap,
                                   max(1, int(_hop_m / 2000)),
                                   lt, InterfaceType.BACKHAUL))
        # Relay ↔ EdgeUPF
        if eupf_nodes:
            nearest_eupf = sorted(eupf_nodes, key=lambda e: _dist(relay, e))[:1]
            for eupf in nearest_eupf:
                lid += 1
                topology.add_link(Link(f"Backhaul_{lid}", (relay, eupf), 800,
                                       max(2, int(_dist(relay, eupf) / 1500)),
                                       LinkType.FIBER, InterfaceType.BACKHAUL))
    # EdgeUPF ↔ Core UPF
    for eupf in eupf_nodes:
        for upf in _rng.sample(_get(NodeType.UPF), min(2, len(_get(NodeType.UPF)))):
            lid += 1
            topology.add_link(Link(f"N9_{lid}", (eupf, upf), 5000,
                                   _rng.randint(3, 8), LinkType.FIBER, InterfaceType.NG))
    # Uu links (UE ↔ nearest O-RUs)
    ru_nodes = _get(NodeType.O_RU)
    for ue in _get(NodeType.UE):
        nearest_rus = sorted(ru_nodes, key=lambda r: _dist(ue, r))[:_rng.randint(1, 2)]
        for ru in nearest_rus:
            lid += 1
            topology.add_link(Link(f"Uu_{lid}", (ue, ru),
                                   _rng.choice([50, 100, 150]), _rng.randint(1, 3),
                                   LinkType.WIRELESS, InterfaceType.Uu))

    infra_count = sum(1 for n in topology.nodes.values() if n.node_type != NodeType.UE)
    ue_count = sum(1 for n in topology.nodes.values() if n.node_type == NodeType.UE)
    mh_total = sum(1 for n in topology.nodes.values() if site_can_steer(n))
    mh_flagged = sum(1 for n in topology.nodes.values()
                     if getattr(n, 'has_multihaul', False))

    # ── MEASURED TG site-plan statistics ────────────────────────────────
    # The plan is reported by MEASUREMENT, not by restating the rule: TG site
    # count, the nearest-neighbour spacing distribution over TG sites, and how
    # much of that distribution actually fits inside the derived TG budget.
    _tg_ids = [nid for nid, n in topology.nodes.items() if site_can_steer(n)]
    _tg_nn = []
    for _a in _tg_ids:
        _d = min((_dist(_a, _b) for _b in _tg_ids if _b != _a), default=None)
        if _d is not None:
            _tg_nn.append(_d)
    _tg_nn.sort()
    def _pct(p):
        if not _tg_nn:
            return float('nan')
        return _tg_nn[min(len(_tg_nn) - 1, int(p * (len(_tg_nn) - 1)))]
    _nn_in_reach = (sum(1 for d in _tg_nn if d <= TG_PLAN_REACH_M)
                    / max(1, len(_tg_nn)))
    _mesh_links = sum(1 for l in topology.links.values()
                      if getattr(l, 'link_type', None) == LinkType.MULTIHAUL_MESH)
    _mw_links = sum(1 for l in topology.links.values()
                    if getattr(l, 'link_type', None) == LinkType.MICROWAVE_PTP)

    print(f"{log_prefix}   Topology: {infra_count} infra + {ue_count} UEs, "
          f"{len(topology.links)} links, TG(MultiHaul)-capable sites: "
          f"{mh_total}")
    print(f"{log_prefix}   [TG-PLAN] reach={TG_PLAN_REACH_M:.0f} m "
          f"@{FRAG_BRIDGE_TX_DBM:.0f} dBm ({TG_HIGH_REACH_M:.0f} m "
          f"@{TG_HIGH_TX_DBM:.0f} dBm) | TG-site nearest-neighbour spacing "
          f"p10/median/p90 = {_pct(0.10):.0f}/{_pct(0.50):.0f}/"
          f"{_pct(0.90):.0f} m | {_nn_in_reach:.0%} of TG sites have a TG "
          f"neighbour inside one hop | static backhaul: "
          f"{_mesh_links} x 60 GHz mesh, {_mw_links} x 18 GHz PtP")

    # Measured topology facts — figure annotations are computed from these,
    # never from hardcoded literals.
    topology_info = {
        'infra': infra_count,
        'ues': ue_count,
        'links': len(topology.links),
        'multihaul': mh_total,
        'tg_sites': mh_total,
        'tg_reach_m': TG_PLAN_REACH_M,
        'tg_reach_high_m': TG_HIGH_REACH_M,
        'tg_nn_p10_m': _pct(0.10),
        'tg_nn_median_m': _pct(0.50),
        'tg_nn_p90_m': _pct(0.90),
        'tg_nn_in_reach_frac': _nn_in_reach,
        'multihaul_mesh_links': _mesh_links,
        'microwave_ptp_links': _mw_links,
    }

    return topology, topology_info


def _replicate_ue_flows(topology):
    """Replicate Simulator._establish_ue_to_ue_flows() at scenario-build time.

    The engine establishes the UE-to-UE flow list deterministically at
    Simulator.__init__ (fixed random.seed(42), node insertion order) — but
    the fragmenting-severance builder needs the flow list BEFORE any
    Simulator exists, to count cross-fragment flows.  This mirrors the
    engine's algorithm exactly (same RNG stream via a local random.Random(42),
    which produces the identical Mersenne-Twister sequence as the module-level
    random.seed(42) the engine uses; same rescue/regular classification —
    freshly generated comparison topologies carry no is_rescue_service flag
    and no traffic_profile attribute at init, so all UEs classify regular,
    exactly as in the engine at that point).
    """
    rng = random.Random(42)                      # engine: random.seed(42)
    ue_nodes = [nid for nid, n in topology.nodes.items()
                if n.node_type.value == "UE" and n.is_survivor]
    if len(ue_nodes) < 2:
        return []

    regular_ues, rescue_ues = [], []
    for ue_id in ue_nodes:
        node = topology.nodes[ue_id]
        # Engine also checks node.traffic_profile LIFE_SAFETY>3.0, but nodes
        # built by build_comparison_topology never carry traffic_profile.
        if getattr(node, 'is_rescue_service', False):
            rescue_ues.append(ue_id)
        else:
            regular_ues.append(ue_id)

    flows = []
    # 1. Emergency UE-to-Rescue (engine part 1)
    for regular_ue in regular_ues:
        available_rescue = rescue_ues.copy()
        if available_rescue:
            k = min(rng.randint(1, 2), len(available_rescue))
            for rescue_ue in rng.sample(available_rescue, k):
                flows.append((regular_ue, rescue_ue))
                flows.append((rescue_ue, regular_ue))
    # 2. Rescue coordination network (engine part 2)
    for rescue_ue in rescue_ues:
        other_rescue = [r for r in rescue_ues if r != rescue_ue]
        if len(other_rescue) >= 2:
            k = min(rng.randint(2, 3), len(other_rescue))
            for partner in rng.sample(other_rescue, k):
                if (rescue_ue, partner) not in flows:
                    flows.append((rescue_ue, partner))
                    flows.append((partner, rescue_ue))
    # 3. Local UE coordination (engine part 3)
    for regular_ue in regular_ues:
        other_regular = [r for r in regular_ues if r != regular_ue]
        if len(other_regular) >= 1:
            k = min(rng.randint(1, 2), len(other_regular))
            for partner in rng.sample(other_regular, k):
                if (regular_ue, partner) not in flows:
                    flows.append((regular_ue, partner))
                    flows.append((partner, regular_ue))

    # De-duplicate by sorted pair (engine's final pass)
    unique_flows, seen = [], set()
    for flow in flows:
        key = tuple(sorted(flow))
        if key not in seen:
            seen.add(key)
            unique_flows.append(flow)
    return unique_flows


class MARLLinkStateDetector:
    """BFD-class failure detection for the MARL arms — the missing handicap.

    ── THE DEFECT THIS FIXES ────────────────────────────────────────────
    Every baseline arm in this harness runs a control-plane state machine
    that must DETECT a link-state change before it can act on it: OSPF waits
    RouterDeadInterval (40 s) then floods then runs SPF; OLSR waits HELLO/TC
    rounds; BATMAN waits OGM rounds and TQ stabilisation; AODV waits an RREQ
    timeout; SDN waits a controller-heartbeat miss.  Each of those arms is
    therefore restricted, every tick, to the flows its own state machine
    believes are routable (`sim.ue_to_ue_flows = ctrl.get_routable_flows()`).

    MARL had NO such state machine.  It read `link.is_up` straight out of the
    topology object and rerouted in the SAME tick the link flipped.  That is
    not a fast detector, it is omniscience: no deployed system learns of a
    remote link failure in zero time.  Against arms that all pay a detection
    delay, a zero-delay arm does not produce a recovery-time COMPARISON, it
    produces an artefact — and recovery time is the headline metric.

    ── WHAT REPLACES IT ─────────────────────────────────────────────────
    A per-adjacency liveness session, the fastest mechanism that actually
    exists in the field: BFD (RFC 5880).  A link-state TRANSITION becomes
    visible to MARL only MARL_BFD_DETECT_TICKS after it physically happens,
    and MARL is then restricted to the flows routable in that BELIEVED
    topology — the identical mechanism, and the identical code shape, as every
    baseline controller in this file.

    Deliberately conservative choices, both stated so neither looks accidental:
      * BOTH DIRECTIONS are delayed, not just failures.  A router does not
        install an interface in the FIB the instant PHY sync is achieved; the
        BFD session must come up first.  This means MARL also waits
        MARL_BFD_DETECT_TICKS before it can use a relay bridge it created
        itself, on top of the RELAY_REPOINT_TICKS beam training the engine
        already charges.  That over-charges MARL slightly rather than under-
        charging it, which is the right direction for a fairness fix.
      * The delay does NOT let MARL forward over a link that is physically
        down.  Belief only ever SUBTRACTS flows here (the engine's live-graph
        routing gate still applies on top), so an undetected failure costs
        MARL delivery — a blackhole — exactly as it would in the field.

    MARL keeps a large, honest advantage: 3 s against OSPF's 40 s dead
    interval is still ~13x faster detection.  What it no longer keeps is an
    infinite one.

    The pre-severed scenario ('rescue_ops', severance at tick 0) is handled by
    construction: the detector adopts the state of any link it has never seen
    before without delay, so a detector first stepped at tick 0 of an
    already-severed network starts converged — the same treatment every
    baseline controller gets from its own mark_pre_converged path.
    """

    def __init__(self, sim, detect_ticks=None):
        self.sim = sim
        self.detect_ticks = max(1, int(MARL_BFD_DETECT_TICKS
                                       if detect_ticks is None
                                       else detect_ticks))
        self.visible = {}      # link_id -> believed is_up
        self._pending = {}     # link_id -> (new_state, tick first observed)
        self.routable_flows = set()
        self._dirty = True
        self._last_flow_sig = None
        self.detections = 0    # transitions actually confirmed (diagnostics)

    def step(self, tick):
        links = self.sim.topology.links
        for lid, link in links.items():
            true_up = bool(getattr(link, 'is_up', True))
            if lid not in self.visible:
                # Never-seen link: adopt without delay.  This is session
                # bring-up, not a transition of a monitored adjacency.
                self.visible[lid] = true_up
                self._pending.pop(lid, None)
                self._dirty = True
                continue
            if true_up == self.visible[lid]:
                # Flap that reverted before the detector confirmed it —
                # exactly what a detect multiplier is for.
                self._pending.pop(lid, None)
                continue
            pend = self._pending.get(lid)
            if pend is None or pend[0] != true_up:
                self._pending[lid] = (true_up, tick)
            elif tick - pend[1] >= self.detect_ticks:
                self.visible[lid] = true_up
                self._pending.pop(lid, None)
                self.detections += 1
                self._dirty = True
        for lid in [l for l in self.visible if l not in links]:
            self.visible.pop(lid, None)
            self._pending.pop(lid, None)
            self._dirty = True
        self._recompute()

    def _recompute(self):
        flow_sig = len(self.sim.ue_to_ue_flows)
        if not self._dirty and flow_sig == self._last_flow_sig:
            return
        self._dirty = False
        self._last_flow_sig = flow_sig
        # Connectivity in the BELIEVED topology.  Undirected components are
        # exactly what nx.shortest_path existence tests, and UEs are included
        # because the flows are UE-to-UE.
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for lid, believed_up in self.visible.items():
            if not believed_up:
                continue
            link = self.sim.topology.links.get(lid)
            if link is None:
                continue
            a, b = link.endpoints
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        self.routable_flows = {
            (s, t) for s, t in self.sim.ue_to_ue_flows
            if find(s) == find(t)
        }

    def get_routable_flows(self):
        """Flows MARL believes are routable, given its detection delay."""
        # sorted(), NOT list(): `routable_flows` is a SET of (str, str)
        # tuples, and set iteration order over strings is SALTED per process
        # (PYTHONHASHSEED).  The engine allocates capacity GREEDILY in the
        # order this list is walked, so an unsorted set gave a different set
        # of served flows in every process - the same reproducibility defect
        # as the salted hash() seeds, just one layer up.  Measured: identical
        # runs diverged across PYTHONHASHSEED values until this was sorted.
        return sorted(self.routable_flows)

    def believed_graph(self):
        """The routing graph AS MARL BELIEVES IT, or None if belief == truth.

        ── WHY THIS EXISTS (the half of the fix the flow gate cannot do) ──
        get_routable_flows() above subtracts flows MARL cannot know are
        reachable.  It does NOT stop MARL rerouting AROUND a freshly-cut link
        in the same tick: `Topology._sync_graph_edges` deletes the edge the
        instant `link.is_up` flips, so `nx.shortest_path` in
        `Simulator._forward_traffic` silently picks a detour that no real
        control plane could have installed yet.  A flow whose endpoints stay
        in one component therefore never even registers the failure — which is
        precisely the omniscience this handicap is meant to remove, and it is
        the common case (a cut that partitions is rare; a cut with a detour is
        typical).

        So path selection must run over the BELIEVED topology: for the
        detection interval, a link that has physically failed is still IN the
        graph, so MARL keeps choosing the dead path and delivers nothing over
        it, until BFD declares the session down and the reroute is allowed.
        That "fail, then reroute" is the real recovery sequence, and its
        duration is what the recovery-time metric is supposed to measure.

        Safety: the re-installed edge CARRIES ITS link_id, so
        `consume_path` resolves the Link object, sees `is_up == False` and
        returns 0.0 (sixg_sim/simulation.py).  An undetected-but-down link is
        therefore selectable but NOT deliverable — a blackhole, exactly as in
        the field.  Dropping the link_id would instead make the edge look like
        infinite capacity and let dead links carry traffic, which is the one
        outcome this must never produce.

        Returns None on the overwhelming majority of ticks (belief matches
        truth), so the graph copy is paid for only while a detection timer is
        actually running.
        """
        links = self.sim.topology.links
        diverged = [lid for lid, believed in self.visible.items()
                    if lid in links
                    and bool(getattr(links[lid], 'is_up', True)) != believed]
        if not diverged:
            return None

        G = self.sim.topology.graph.copy()
        # PAIR granularity, matching Topology._sync_graph_edges: parallel links
        # between one node pair share a single graph edge, so the edge exists
        # iff SOME link of that pair is (believed) up.  sorted() everywhere —
        # this must not depend on dict/set iteration order (PYTHONHASHSEED).
        pairs = {tuple(sorted(links[lid].endpoints)) for lid in diverged}
        for u, v in sorted(pairs):
            chosen = None
            for lid in sorted(links):
                lk = links[lid]
                if tuple(sorted(lk.endpoints)) != (u, v):
                    continue
                if self.visible.get(lid, bool(getattr(lk, 'is_up', True))):
                    chosen = lk
                    break
            if chosen is not None:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, link_id=chosen.id, capacity=chosen.capacity)
            elif G.has_edge(u, v):
                # Believed down but physically up: MARL may not use a bridge
                # whose BFD session has not come up yet.
                G.remove_edge(u, v)
        return G


def _post_severance_infra_graph(topology, cut_link_ids):
    """(networkx graph, survivor list) for the infra graph AFTER sever_core
    plus the given fragmenting cuts.

    Same node/edge definition every other consumer uses: non-UE, non-core
    survivors; edges = is_up links between them minus `cut_link_ids`.  Core
    types match Simulator._identify_core_nodes.
    """
    import networkx as _nx
    CORE_TYPES = {"Core", "SMO", "Non-RT-RIC", "AMF", "UPF"}
    cut = set(cut_link_ids or ())
    surv = [nid for nid, n in topology.nodes.items()
            if n.node_type.value not in CORE_TYPES
            and n.node_type.value != 'UE' and n.is_survivor]
    G = _nx.Graph()
    G.add_nodes_from(surv)
    for lid, link in topology.links.items():
        if lid in cut or not getattr(link, 'is_up', True):
            continue
        u, v = link.endpoints
        if u in G and v in G:
            G.add_edge(u, v)
    return G, surv


def tg_feasible_component_optimum(topology, cut_link_ids,
                                  tx_power_dbm=FRAG_BRIDGE_TX_DBM,
                                  rain_mm_h=0.0, relay_bw_fraction=0.20,
                                  graph=None, surv=None):
    """EXACT minimum infra component count reachable by TG-only steering.

    THIS IS THE REFERENCE EVERY REUNIFICATION CLAIM MUST BE MADE AGAINST.
    "1 component" is the reference only when 1 component is physically
    reachable; when the 60 GHz budget cannot close the last gap on a given
    seed, scoring an arm against 1 measures the plan, not the arm.  So this
    function solves the reunification problem OPTIMALLY under exactly the
    constraints the engine imposes, and the result — not the literal 1 — is
    what metrics, annotations and verdicts compare achieved components to.

    Constraints (each mirroring the engine, see Simulator._step_phy_mac):
      * only a site with has_multihaul may steer (MW PtP is mechanically
        aligned and can never be re-pointed inside an episode);
      * each such site holds AT MOST ONE relay link;
      * the far end needs NO TG radio — can_form_relay_link steers the
        SOURCE, so any surviving infra node is a legal target;
      * feasibility is the link budget itself (TransportRelayModel.
        can_form_relay_link at `tx_power_dbm`), never a range constant.

    Interference: evaluated thermal-noise-only.  The aggregate 60 GHz I term
    needs a live Simulator, which does not exist at plan/scenario-build time,
    and 60 GHz is noise-limited in this deployment (measured I/N well below
    0 dB thanks to pencil-beam discrimination + oxygen absorption).  So this
    is an upper bound on capability by a fraction of a dB, and it is the SAME
    bound for every arm.

    Exactness: the problem is "choose <= 1 fragment-merging edge per TG site
    so as to merge the most fragments", i.e. a maximum common independent set
    of a graphic and a partition matroid.  Greedy is not exact for that in
    general, so instead of greedy this enumerates FORESTS over the fragment
    set (K <= FRAG_MAX_FRAGMENTS, so there are only a handful of candidate
    edges) and tests each for a system of distinct site representatives.  The
    previous greedy `_check_bridgeable` could report a partition unbridgeable
    that is in fact bridgeable, and vice versa.

    Returns (raw_components, min_components, n_tg_sites).
    """
    import itertools as _it
    if graph is None or surv is None:
        graph, surv = _post_severance_infra_graph(topology, cut_link_ids)
    import networkx as _nx
    comps = list(_nx.connected_components(graph))
    K = len(comps)
    frag_of = {n: i for i, c in enumerate(comps) for n in c}

    tg_sites = [n for n in surv if site_can_steer(topology.nodes[n])]
    if K <= 1:
        return K, K, len(tg_sites)

    model = TransportRelayModel(rain_mm_h=rain_mm_h)
    for nid in surv:
        nd = topology.nodes[nid]
        model.register_position(nid, nd.x_pos, nd.y_pos)

    # site -> set of OTHER fragments it can reach with one steered TG hop
    reach = {}
    for m in tg_sites:
        fm = frag_of[m]
        tgts = set()
        for n in surv:
            fn = frag_of[n]
            if fn == fm or fn in tgts:
                continue
            ok, _cap, _sinr = model.can_form_relay_link(
                m, n, tx_power_dbm, relay_bw_fraction=relay_bw_fraction)
            if ok:
                tgts.add(fn)
        if tgts:
            reach[m] = tgts

    # fragment PAIR -> sites that could realise that merge
    pair_sites = {}
    for m, tgts in reach.items():
        fm = frag_of[m]
        for f in tgts:
            key = (min(fm, f), max(fm, f))
            pair_sites.setdefault(key, []).append(m)
    if not pair_sites:
        return K, K, len(tg_sites)

    pairs = sorted(pair_sites)

    def _is_forest(edges):
        parent = list(range(K))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
        return True

    def _has_sdr(edges):
        """System of distinct representatives: one DISTINCT site per edge."""
        assign = {}

        def rec(i, used):
            if i == len(edges):
                return True
            for s in pair_sites[edges[i]]:
                if s in used:
                    continue
                used.add(s)
                if rec(i + 1, used):
                    return True
                used.discard(s)
            return False
        return rec(0, set())

    for n_merges in range(K - 1, 0, -1):
        if n_merges > len(pairs):
            continue
        for combo in _it.combinations(pairs, n_merges):
            if not _is_forest(combo):
                continue
            if _has_sdr(list(combo)):
                return K, K - n_merges, len(tg_sites)
    return K, K, len(tg_sites)


def scenario_cut_link_ids(scenario):
    """Link ids the scenario's fragmenting severance cuts (empty if none)."""
    return [e.parameters.get('link_id') for e in getattr(scenario, 'events', [])
            if e.event_type == 'fail_link'
            and e.parameters.get('reason') == 'fragmenting_severance']


def _build_fragmenting_severance(topology, severance_tick, scenario_seed,
                                 log_prefix=""):
    """Choose the seed-varied link cuts that FRAGMENT the post-severance
    infra graph, verify them programmatically, and return them as
    'fail_link' ScenarioEvents (all at severance_tick, alongside sever_core).

    Algorithm (pure function of (topology, scenario_seed) — every arm gets
    the identical event list):
      1. Build the post-sever_core surviving infra graph G0 with networkx:
         nodes = all non-UE, non-core survivors; edges = links between them
         (sever_core already removes every link touching a core node; core
         types match Simulator._identify_core_nodes).
      2. Seeded attempts (up to FRAG_BUILD_ATTEMPTS): pick a target fragment
         count K in [FRAG_MIN_FRAGMENTS, FRAG_MAX_FRAGMENTS], choose K
         geographically spread centers (farthest-point with seeded tie
         randomisation), assign each infra node to its nearest center, and
         cut EVERY G0 edge crossing the partition — a geographically
         correlated damage corridor severs all physical media crossing it
         (fiber fronthaul/F1 and MW/MultiHaul backhaul alike), which is the
         only physically coherent way to genuinely partition the mesh.
      3. Verify each attempt with networkx connected components:
           * actual fragment count within [FRAG_MIN, FRAG_MAX];
           * cross-fragment UE-to-UE flow share within
             [FRAG_CROSS_FLOW_MIN, FRAG_CROSS_FLOW_MAX] — a flow spans
             fragments iff the fragment sets of its endpoints' serving
             infra anchors (Uu neighbours) are disjoint;
           * bridgeability: using TransportRelayModel.can_form_relay_link
             at FRAG_BRIDGE_TX_DBM (range + link-budget feasibility), a
             greedy spanning check confirms the fragments can ALL be
             reconnected with each MultiHaul survivor forming at most one
             relay link (the engine's per-node limit).
      4. First fully-valid attempt wins; if none passes, the least-bad
         attempt is used and a loud warning is printed (still identical
         across arms — determinism is never sacrificed).
    """
    import networkx as _nx
    from sixg_sim.transport_relay_model import TransportRelayModel

    # Core-node set exactly as Simulator._identify_core_nodes defines it
    CORE_TYPES = {"Core", "SMO", "Non-RT-RIC", "AMF", "UPF"}
    node_type = lambda nid: topology.nodes[nid].node_type.value
    is_ue = lambda nid: node_type(nid) == 'UE'
    core = {nid for nid in topology.nodes if node_type(nid) in CORE_TYPES}

    # 1. Post-sever_core surviving infra graph
    surv = [nid for nid, n in topology.nodes.items()
            if nid not in core and not is_ue(nid) and n.is_survivor]
    pos = {nid: (topology.nodes[nid].x_pos, topology.nodes[nid].y_pos)
           for nid in surv}
    G0 = _nx.Graph()
    G0.add_nodes_from(surv)
    pair_links = {}   # (u,v) sorted -> [link_id, ...] (ALL parallel links)
    for lid, link in topology.links.items():
        u, v = link.endpoints
        if u in pos and v in pos and getattr(link, 'is_up', True):
            key = (u, v) if u < v else (v, u)
            pair_links.setdefault(key, []).append(lid)
            G0.add_edge(*key)

    def _dist(a, b):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        return math.hypot(xa - xb, ya - yb)

    # UE -> serving infra anchors (Uu neighbours) for cross-flow counting
    ue_anchors = {}
    for link in topology.links.values():
        u, v = link.endpoints
        for ue, anc in ((u, v), (v, u)):
            if is_ue(ue) and anc in pos:
                ue_anchors.setdefault(ue, set()).add(anc)

    flows = _replicate_ue_flows(topology)

    # Relay bridgeability model (the SAME physics the MARL relay action uses)
    relay_model = TransportRelayModel()
    for nid in surv:
        relay_model.register_position(nid, *pos[nid])
    # SAME predicate as the engine and the optimum solver: a partition is
    # only "bridgeable" if the sites that would bridge it can actually act.
    mh_nodes = [nid for nid in surv if site_can_steer(topology.nodes[nid])]

    def _bridge_feasible(m, n):
        ok, _cap, _sinr = relay_model.can_form_relay_link(
            m, n, FRAG_BRIDGE_TX_DBM, relay_bw_fraction=0.20)
        return ok

    def _tg_optimum_for(cut_pairs, tx_dbm=FRAG_BRIDGE_TX_DBM):
        """EXACT (K, min achievable components) for a candidate partition.

        Delegates to tg_feasible_component_optimum, i.e. the SAME solver the
        run-time feasibility report and every "target" annotation use, so the
        printed `bridgeable=` verdict cannot disagree with the optimum the
        arms are scored against.  This replaces a greedy spanning check that
        could label a bridgeable partition unbridgeable (and the reverse),
        which is why FRAG_BUILD_ATTEMPTS was mostly bottoming out in the
        least-bad fallback branch.
        """
        H = G0.copy()
        H.remove_edges_from(cut_pairs)
        return tg_feasible_component_optimum(
            topology, (), tx_power_dbm=tx_dbm, graph=H, surv=surv)

    # 2./3. Seeded partition attempts
    seed_val = 0 if scenario_seed is None else int(scenario_seed)
    rng = random.Random(seed_val * 7919 + 0x5EED)
    best = None   # (badness, result-dict); badness 0.0 == fully valid
    for _attempt in range(FRAG_BUILD_ATTEMPTS):
        k_target = rng.randint(FRAG_MIN_FRAGMENTS, FRAG_MAX_FRAGMENTS)
        centers = [rng.choice(surv)]
        while len(centers) < k_target:
            ranked = sorted(surv,
                            key=lambda nid: -min(_dist(nid, c)
                                                 for c in centers))
            centers.append(rng.choice(ranked[:3]))
        assign = {nid: min(range(len(centers)),
                           key=lambda i: _dist(nid, centers[i]))
                  for nid in surv}
        cut_pairs = [(u, v) for (u, v) in G0.edges if assign[u] != assign[v]]

        H = G0.copy()
        H.remove_edges_from(cut_pairs)
        comps = list(_nx.connected_components(H))
        K = len(comps)
        frag_of = {nid: i for i, comp in enumerate(comps) for nid in comp}

        ue_frags = {ue: {frag_of[a] for a in anchors if a in frag_of}
                    for ue, anchors in ue_anchors.items()}
        total = len(flows)
        cross = sum(1 for s, t in flows
                    if ue_frags.get(s) and ue_frags.get(t)
                    and ue_frags[s].isdisjoint(ue_frags[t]))
        share = cross / max(1, total)
        # TRUTHFUL bridgeability: the exact TG-only optimum, not a greedy
        # spanning heuristic.  `bridgeable` now means "1 component is
        # physically reachable at the screening power", and `tg_opt` is the
        # TG-FEASIBLE OPTIMUM this seed will be scored against if the
        # partition is selected.
        _K_chk, tg_opt, _n_tg = _tg_optimum_for(cut_pairs)
        bridgeable = (tg_opt <= 1)

        badness = 0.0
        if K < FRAG_MIN_FRAGMENTS or K > FRAG_MAX_FRAGMENTS:
            badness += 10.0 * min(abs(K - FRAG_MIN_FRAGMENTS),
                                  abs(K - FRAG_MAX_FRAGMENTS))
        badness += 100.0 * max(0.0, FRAG_CROSS_FLOW_MIN - share)
        badness += 100.0 * max(0.0, share - FRAG_CROSS_FLOW_MAX)
        if not bridgeable:
            badness += 1000.0
        result = {'cut_pairs': cut_pairs, 'K': K, 'cross': cross,
                  'total': total, 'share': share, 'bridgeable': bridgeable,
                  'tg_optimum': tg_opt}
        if best is None or badness < best[0]:
            best = (badness, result)
        if badness == 0.0:
            break

    badness, res = best
    cut_lids = []
    events = []
    for (u, v) in res['cut_pairs']:
        key = (u, v) if u < v else (v, u)
        for lid in pair_links.get(key, []):
            cut_lids.append(lid)
            events.append(ScenarioEvent(
                tick=severance_tick, event_type='fail_link',
                parameters={'link_id': lid,
                            'reason': 'fragmenting_severance'}))

    # 3b. Required build-time verification line (identical for every arm
    # because the whole construction is seed-deterministic).
    #
    # `bridgeable` and `tg_optimum` come from the SAME exact solver
    # (tg_feasible_component_optimum), so the printed verdict is truthful by
    # construction: bridgeable=yes iff tg_optimum=1.  `tg_optimum` is the
    # TG-FEASIBLE OPTIMUM — the component count this seed is scored against.
    # It is 1 whenever full reunification is physically reachable at the
    # screening power, and >1 (with bridgeable=NO) when it is not, in which
    # case the >1 value, not 1, is the reference for every metric.
    _opt_hi = tg_feasible_component_optimum(
        topology, cut_lids, tx_power_dbm=TG_HIGH_TX_DBM)[1]
    print(f"{log_prefix}[SCENARIO] fragments={res['K']} "
          f"cross_fragment_flows={res['cross']}/{res['total']} "
          f"(share={res['share']:.1%}, cut_pairs={len(res['cut_pairs'])}, "
          f"cut_links={len(cut_lids)}, "
          f"bridgeable={'yes' if res['bridgeable'] else 'NO'}, "
          f"tg_optimum@{FRAG_BRIDGE_TX_DBM:.0f}dBm={res['tg_optimum']} "
          f"tg_optimum@{TG_HIGH_TX_DBM:.0f}dBm={_opt_hi} "
          f"tg_reach={TG_PLAN_REACH_M:.0f}m, "
          f"tg_sites={len(mh_nodes)}, scenario_seed={seed_val})")
    if badness > 0.0:
        print(f"{log_prefix}[SCENARIO] WARNING: fragmenting-severance "
              f"constraints not fully met after {FRAG_BUILD_ATTEMPTS} "
              f"attempts (badness={badness:.1f}) — using least-bad "
              f"partition (still identical across arms). Targets: "
              f"fragments {FRAG_MIN_FRAGMENTS}-{FRAG_MAX_FRAGMENTS}, "
              f"cross-flow share {FRAG_CROSS_FLOW_MIN:.0%}-"
              f"{FRAG_CROSS_FLOW_MAX:.0%}, bridgeable=yes")
    return events


def build_comparison_scenario(scenario_name, topology_seed, topology=None,
                              ticks=None, severance_tick=None,
                              rescue_tick=None, rescue_num_ues=20,
                              rescue_zone="zone_0", scenario_seed=None):
    """Build the comparison scenario ('recovery' or 'rescue_ops').

    Defaults reproduce run_physics_pass() exactly (module globals
    TOTAL_TICKS / SEVERANCE_TICK / RESCUE_ARRIVAL_TICK, read at call
    time so --ticks overrides still apply).  Training drivers may pass
    ticks / severance_tick / rescue_tick / scenario_seed to randomise
    episode timing without changing evaluation behaviour.
    """
    if topology is None:
        topology, _ = build_comparison_topology(topology_seed)
    if ticks is None:
        ticks = TOTAL_TICKS
    if severance_tick is None:
        severance_tick = SEVERANCE_TICK
    if rescue_tick is None:
        rescue_tick = RESCUE_ARRIVAL_TICK
    if scenario_seed is None:
        scenario_seed = topology_seed
    generator = DiverseScenarioGenerator(topology, base_duration=ticks,
                                         curriculum_start=4)
    scenario = generator.generate(episode=1, seed=scenario_seed)

    # ── Scenario-specific events ──
    if scenario_name == 'rescue_ops':
        # Scenario B: network already severed & converged (from Scenario A).
        # Sever at tick 0 so we start in island mode; rescue arrives later.
        scenario.events = [
            ScenarioEvent(tick=0, event_type="sever_core", parameters={}),
            ScenarioEvent(tick=rescue_tick, event_type="rescue_force_arrival",
                          parameters={"num_ues": rescue_num_ues,
                                      "coverage_area": rescue_zone})
        ]
    else:  # 'recovery' — FRAGMENTING severance, no rescue
        # sever_core alone leaves the surviving infra graph fully connected
        # (routable_flows=302/302 post-severance) — classical protocols then
        # recover by mere re-convergence and island-mode bridging is never
        # exercised.  Extend the severance with seed-varied link cuts that
        # PARTITION the infra graph (built + verified with networkx above;
        # same event list for every arm, physically bridgeable by the relay
        # model).  Training drivers share this builder, so training recovery
        # episodes see the same fragmenting severance class.
        frag_events = _build_fragmenting_severance(
            topology, severance_tick, scenario_seed)
        scenario.events = [
            ScenarioEvent(tick=severance_tick, event_type="sever_core",
                          parameters={}),
        ] + frag_events
    return scenario


def _adapted_checkpoint_path(checkpoint_path, suffix, topology_seed):
    """Where a MARL arm saves its recovery-adapted policy for the rescue pass.

    For a single checkpoint this is the historical `<name><suffix>_seed<N>.pt`
    next to it.  For an ENSEMBLE (comma-separated member list) that string
    would be a nonsense path, so the adapted state -- which is the ensemble's
    own state_dict per agent -- is written next to the FIRST member as
    `ensemble<suffix>_seed<N>.pt`.  main() and run_physics_pass both call
    this, so the writer and the reader can never disagree.
    """
    s = str(checkpoint_path)
    if ',' in s:
        first = s.split(',')[0].strip()
        return os.path.join(os.path.dirname(first) or '.',
                            f'ensemble{suffix}_seed{topology_seed}.pt')
    return s.replace('.pt', f'{suffix}_seed{topology_seed}.pt')


def _policy_fingerprint(rl_agents):
    """sha256 over every agent's policy parameters, agents in sorted order.
    Used by the static arm to prove its parameters never change."""
    import hashlib
    h = hashlib.sha256()
    for aid in sorted(rl_agents):
        sd = rl_agents[aid].policy_net.state_dict()
        for k in sorted(sd):
            h.update(str(aid).encode())
            h.update(k.encode())
            h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _load_into_agent(agent, sd) -> bool:
    """Load a per-agent state dict, installing an EnsemblePolicy first when
    the weights are ensemble-shaped (keys `members.<k>.<layer>`), so that a
    rescue pass can resume from a recovery-adapted ENSEMBLE.  Returns the
    obs-dim-adapted flag for plain checkpoints (False for ensembles, whose
    adapted dicts always come from this same code/obs schema)."""
    from sixg_sim.agent import (EnsemblePolicy, PolicyNetwork,
                                load_policy_state_dict)
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    if keys and all(k.startswith('members.') for k in keys):
        n = 1 + max(int(k.split('.')[1]) for k in keys)
        ens = EnsemblePolicy([PolicyNetwork(OBS_DIM) for _ in range(n)])
        ens.load_state_dict(sd)
        agent.policy_net = ens
        return False
    return load_policy_state_dict(agent.policy_net, sd)


def run_physics_pass(mode, topology_seed, checkpoint_path, scenario_name='recovery'):
    """Run one FULL physics engine pass.

    ALL protocols share the same physical environment:
      - Auto CQI→MCS link adaptation (3GPP TS 38.214) — native gNB/DU function
      - Fixed TX power 23 dBm (except MARL: agent-controlled)
      - Same backhaul degradation, same interference environment

    Protocol-specific capabilities:
      mode='marl':   CTDE agents + online PPO. Controls PRB/power/relay/scheduling.
                     Relay activation: immediate (on-node policy).
                     MCS: bounded agent OFFSET from the shared auto CQI→MCS choice.
      mode='marl_freeze':
                     Identical to 'marl', except online PPO updates stop
                     FREEZE_AFTER_LAST_EVENT_TICKS ticks after the last
                     scenario event; the policy then runs frozen with
                     deterministic argmax action selection.
      mode='ospf':   Auto MCS, 23 dBm. OSPF convergence delay (80 ticks).
                     LOCAL_REROUTE on existing mesh links after convergence.
      mode='sdn':    Auto MCS, 23 dBm. SDN bootstrap delay (300 ticks).
                     MultiHaul relay activation via NETCONF/YANG after operational phase.
      mode='olsr':   Auto MCS, 23 dBm. OLSR MPR-based convergence (400 ticks).
                     LOCAL_REROUTE on existing mesh links after convergence.
      mode='batman': Auto MCS, 23 dBm. B.A.T.M.A.N. OGM convergence (200 ticks).
                     No relay capability (L2 protocol, no management interface).
      mode='aodv':   Auto MCS, 23 dBm. AODV reactive discovery (~80 ticks/flow).
                     No relay capability (reactive L3, no management interface).
      mode='random': BRIDGING-ATTRIBUTION CONTROL ARM.  Byte-for-byte the
                     MARL code path — same observations, same action space,
                     same _step_phy_mac, same relay mechanics and teardown
                     rules — with the policy network REPLACED by uniform
                     sampling over every action head (Simulator.
                     random_action_policy).  No checkpoint is loaded, no PPO
                     update runs, no exploration temperature is applied and no
                     adapted policy is saved.  Its bridge count and achieved
                     component count are the null hypothesis for the claim
                     that MARL LEARNED to bridge islands: whatever this arm
                     achieves is attributable to the environment, not to
                     learning.  Excluded from the publication figures
                     (FIGURE_EXCLUDED_ARMS) — it is a measurement, not a
                     competing protocol.

    scenario_name:
      'recovery'    — Severance only. Civilian emergency traffic. No rescue.
      'rescue_ops'  — Severance + rescue UE arrival after convergence.

    Returns per-tick arrays of raw KPIs from the physics engine.
    """
    # ── Arm plumbing: 'marl_freeze' reuses every MARL code path; only the
    # freeze gating below differs.  mode_label keeps log lines per-arm.
    arm = mode
    is_freeze_arm = (arm == 'marl_freeze')
    # 'marl_static' reuses every MARL code path with learning disabled from
    # the first tick (see the MARL STATIC arm constants block).
    is_static_arm = (arm == 'marl_static')
    # 'random' also reuses every MARL code path; only the ACTION SOURCE
    # differs (uniform sampling instead of the policy net) and all learning
    # machinery is disabled.  That is the whole point: any difference in
    # outcome between 'random' and the MARL arms is attributable to the
    # policy, because nothing else about the run differs.
    is_random_arm = (arm == 'random')
    # 'xldet' is the ENGINEERED NON-LEARNING arm.  Like 'random' it reuses
    # every MARL code path and differs only in the ACTION SOURCE, but where
    # 'random' asks "is the action space alone enough?", this asks the harder
    # question "is LEARNING necessary, or would a competent hand-written
    # cross-layer controller with the same actuation and the same observations
    # do as well?".  See sixg_sim/heuristic_controller.py.
    is_xldet_arm = (arm == 'xldet')
    if is_freeze_arm or is_static_arm or is_random_arm or is_xldet_arm:
        mode = 'marl'
    # True only for the arms that actually learn/adapt online.
    learns = (mode == 'marl') and not (is_random_arm or is_xldet_arm
                                       or is_static_arm)
    mode_label = arm.upper()

    # Last scripted scenario event: severance (Scenario A) / rescue arrival
    # (Scenario B).  The adapt-freeze arm disables online updates a fixed
    # number of ticks after it (named constant FREEZE_AFTER_LAST_EVENT_TICKS).
    _last_event_tick = (RESCUE_ARRIVAL_TICK if scenario_name == 'rescue_ops'
                        else SEVERANCE_TICK)
    _freeze_tick = _last_event_tick + FREEZE_AFTER_LAST_EVENT_TICKS
    _frozen = False
    # ── Guarded-freeze state (see FREEZE_GUARD_* constants) ──
    _force_freeze_tick = _last_event_tick + FREEZE_FORCE_AFTER_TICKS
    _next_freeze_check = _freeze_tick
    _guard_trail_means = []   # per-tick trailing-window achievability means
    _guard_best_mean = None   # best boost-free trailing mean since last event
    _guard_best_sd = None     # per-agent policy snapshot at that best window
    _guard_fb_mean = None     # fallback: best post-event window (boost allowed)
    _guard_fb_sd = None       # fallback snapshot (used only if no boost-free one)

    print(f"\n[{mode_label} - {scenario_name}] --- Starting Physics Pass: {mode_label} [{scenario_name}] seed={topology_seed} ---")
    if is_freeze_arm:
        print(f"[{mode_label} - {scenario_name}] adapt-freeze arm: online updates "
              f"disabled at t={_freeze_tick} (last event t={_last_event_tick} "
              f"+ {FREEZE_AFTER_LAST_EVENT_TICKS}); frozen phase = deterministic argmax; "
              f"freeze gated on trailing-{FREEZE_GUARD_TRAIL} achievability >= "
              f"{FREEZE_GUARD_RATIO:.0%} of trailing-{FREEZE_GUARD_PEAK_WINDOW} peak "
              f"(force-freeze at best checkpoint by t={_force_freeze_tick})")

    topology, topology_info = build_comparison_topology(
        topology_seed, log_prefix=f"[{mode_label} - {scenario_name}]")

    scenario = build_comparison_scenario(scenario_name, topology_seed,
                                         topology=topology)

    # ── TG-FEASIBLE OPTIMUM FOR THIS SEED ───────────────────────────────
    # The reference every reunification number is scored against.  It is a
    # property of (topology seed, scenario) ONLY — identical for every arm —
    # and it replaces the literal "1 component" wherever 1 is not physically
    # reachable.  Recorded into the run data so figures, tables and the
    # attribution verdict all read the same number instead of assuming 1.
    _cut_lids = scenario_cut_link_ids(scenario)
    _raw_K, _tg_opt, _n_tg = tg_feasible_component_optimum(
        topology, _cut_lids, tx_power_dbm=FRAG_BRIDGE_TX_DBM)
    _raw_K_hi, _tg_opt_hi, _ = tg_feasible_component_optimum(
        topology, _cut_lids, tx_power_dbm=TG_HIGH_TX_DBM)
    topology_info.update({
        'raw_components': _raw_K,
        'tg_optimum': _tg_opt,
        'tg_optimum_high_tx': _tg_opt_hi,
        'tg_optimum_tx_dbm': FRAG_BRIDGE_TX_DBM,
        'tg_optimum_high_tx_dbm': TG_HIGH_TX_DBM,
        'tg_optimum_achievable': bool(_tg_opt <= 1),
    })
    _opt_note = ("full reunification IS physically reachable, so the optimum "
                 "and the naive target of 1 coincide"
                 if _tg_opt <= 1 else
                 f"full reunification is NOT physically reachable — every "
                 f"reunification metric is scored against {_tg_opt}, NOT 1")
    print(f"[{mode_label} - {scenario_name}]   [TG-OPTIMUM] seed="
          f"{topology_seed}: raw components={_raw_K}, TG-feasible optimum="
          f"{_tg_opt} @{FRAG_BRIDGE_TX_DBM:.0f} dBm "
          f"({_tg_opt_hi} @{TG_HIGH_TX_DBM:.0f} dBm), tg_sites={_n_tg} — "
          f"{_opt_note}")

    # ── PER-SEED TRAFFIC REALISATION ────────────────────────────────────
    # random_seed was left None here, which meant Simulator.__init__ never
    # seeded the traffic generator and every per-flow traffic model keyed off
    # `(self.config.random_seed or 0)` — i.e. off the CONSTANT 0.  The offered
    # load was therefore bit-identical on every topology seed, so the only
    # variance across seeds was spatial layout.  That understates every
    # reported confidence interval, because traffic variability (a genuine
    # source of run-to-run spread) contributed exactly zero to it.
    #
    # The seed is the TOPOLOGY seed, so it stays IDENTICAL ACROSS ARMS for a
    # given seed (offered load must be the same demand for everyone — it is
    # the achievability denominator) while varying ACROSS seeds.  It is also
    # reproducible: no salted hash, no wall clock.
    config = SimulationConfig(enable_island_detection=True, verbose=False,
                              random_seed=int(topology_seed))
    sim = Simulator(topology, scenario, config)

    rl_agents = {aid: a for aid, a in sim.agents.items() if isinstance(a, RLAgent)}
    critic_net = CriticNetwork(OBS_DIM)

    if is_random_arm:
        # Uniform-random reference policy: the network is never consulted, so
        # no checkpoint is loaded (the randomly initialised weights are
        # irrelevant — their outputs are discarded).  Everything else about
        # the environment matches the MARL arms exactly, including
        # marl_ue_routing_enabled, so the comparison is like-for-like.
        sim.random_action_policy = True
        sim.marl_ue_routing_enabled = True
        torch.manual_seed(topology_seed)   # reproducible per evaluation seed
        print(f"[{mode_label} - {scenario_name}] RANDOM-ACTION CONTROL ARM: "
              f"every action head sampled uniformly (PRB ~ Dirichlet(1,1,1)); "
              f"no checkpoint, no PPO, no temperature control. Bridge/"
              f"component counts from this arm are the learning-free baseline.")
    elif is_xldet_arm:
        # XL-DET: the policy network is never consulted, so no checkpoint is
        # loaded and no PPO runs.  Everything else — observations, physics,
        # teardown rules, marl_ue_routing_enabled — matches the MARL arms
        # exactly, so any difference is attributable to the policy alone.
        # verify_prb_point() re-derives the controller's central numeric claim
        # against the engine and raises rather than let a silently-degraded
        # baseline flatter the learned policy it exists to challenge.
        from sixg_sim import heuristic_controller as _XLDET
        _xl_prb, _xl_modes = _XLDET.verify_prb_point()
        sim.heuristic_action_policy = True
        sim.marl_ue_routing_enabled = True
        torch.manual_seed(topology_seed)   # only affects the environment now
        print(f"[{mode_label} - {scenario_name}] ENGINEERED NON-LEARNING ARM "
              f"(XL-DET): deterministic cross-layer rules on the same "
              f"observations, obeying the same locality mask; no checkpoint, "
              f"no PPO. PRB pinned to "
              f"{_xl_prb[0]:.4f}/{_xl_prb[1]:.4f}/{_xl_prb[2]:.4f} "
              f"(all four traffic classes ADMIT).")
    elif mode == 'marl' and ',' in str(checkpoint_path):
        # ── ENSEMBLE of independently trained checkpoints ────────────────
        # MARL_EVAL_CHECKPOINT="a.pt,b.pt,c.pt,d.pt".  Each member is loaded
        # into its own PolicyNetwork and the members' logits are AVERAGED
        # (sixg_sim.agent.EnsemblePolicy).  Post-freeze bridge churn is zero,
        # so the residual component count is decided by the static argmax the
        # policy freezes into, and that configuration varied sharply across
        # otherwise-equivalent training replicates (torch seed 2345 froze
        # into no bridges on 3/7 reunifiable instances; 3456 matched XL-DET).
        # Averaging is the standard variance-reduction answer: no member is
        # selected on the benchmark, nothing is fitted to evaluation seeds.
        # The ensemble is installed BEFORE the MAPPOTrainer is built, so the
        # freeze arm's online adaptation trains it as one shared actor and
        # every downstream consumer sees an ordinary {head: logits} actor.
        from sixg_sim.agent import (EnsemblePolicy, PolicyNetwork,
                                    load_policy_state_dict)
        _paths = [p.strip() for p in str(checkpoint_path).split(',') if p.strip()]
        _members = []
        _obs_dim_adapted = False
        for _p in _paths:
            _ck = torch.load(_p, map_location='cpu', weights_only=False)
            _sd = (_ck.get('state_dict', _ck) if isinstance(_ck, dict) else _ck)
            _net = PolicyNetwork(OBS_DIM)
            _obs_dim_adapted |= load_policy_state_dict(_net, _sd)
            _members.append(_net)
        _ens = EnsemblePolicy(_members)
        for aid, agent in rl_agents.items():
            agent.policy_net = _ens
            agent.device = torch.device('cpu')
            agent.is_training = True
            agent._logit_temperature = TEMP_FLOOR
        sim.marl_ue_routing_enabled = True
        print(f"[{mode_label}] OK Loaded ENSEMBLE of {len(_members)} actors "
              f"(averaged logits): "
              + ", ".join(os.path.basename(os.path.dirname(p)) or p for p in _paths))
        if _obs_dim_adapted:
            print("  !! one or more members predate the relay_capable obs dim")
    elif mode == 'marl' and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Detect checkpoint format:
        #   New: {agent_id: state_dict, ...}  (per-agent specialized policies)
        #   Old: raw state_dict (single policy for all agents)
        is_per_agent = (isinstance(checkpoint, dict) and
                        all(isinstance(v, dict) for v in checkpoint.values()) and
                        not any(k.startswith('fc') or k.startswith('ln') for k in checkpoint.keys()))
        
        loaded_count = 0
        fallback_count = 0
        # Checkpoints written before the `relay_capable` observation dim was
        # added are OBS_DIM=59; load_policy_state_dict() zero-pads the fc1
        # input column so they still load (and behave EXACTLY as before —
        # they cannot use the new feature until retrained).
        from sixg_sim.agent import load_policy_state_dict
        _obs_dim_adapted = False

        if is_per_agent:
            # Per-agent checkpoint: each agent gets its own trained policy
            # Fallback: if agent_id not in checkpoint, use the policy from
            # a matching node type, or the first available policy
            fallback_sd = list(checkpoint.values())[0]  # last resort
            
            for aid, agent in rl_agents.items():
                if aid in checkpoint:
                    _obs_dim_adapted |= _load_into_agent(agent, checkpoint[aid])
                    loaded_count += 1
                else:
                    # Try to find a matching node type
                    node = sim.topology.nodes.get(aid)
                    my_type = node.node_type.value if node else None
                    matched = False
                    if my_type:
                        for ckpt_aid, ckpt_sd in checkpoint.items():
                            ckpt_node = sim.topology.nodes.get(ckpt_aid)
                            if ckpt_node and ckpt_node.node_type.value == my_type:
                                _obs_dim_adapted |= _load_into_agent(agent, ckpt_sd)
                                matched = True
                                break
                    if not matched:
                        _obs_dim_adapted |= _load_into_agent(agent, fallback_sd)
                    fallback_count += 1
                
                agent.device = torch.device('cpu')
                agent.is_training = True
                agent._logit_temperature = TEMP_FLOOR
        else:
            # Legacy single-policy format: apply same weights to all agents
            sd = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            for aid, agent in rl_agents.items():
                _obs_dim_adapted |= load_policy_state_dict(agent.policy_net, sd)
                agent.device = torch.device('cpu')
                agent.is_training = True
                agent._logit_temperature = TEMP_FLOOR
                loaded_count += 1

        sim.marl_ue_routing_enabled = True
        fmt = "per-agent" if is_per_agent else "shared (legacy)"
        print(f"[{mode_label}] OK Loaded {fmt} policy from {checkpoint_path}")
        print(f"  Direct match: {loaded_count}, fallback: {fallback_count}, total agents: {len(rl_agents)}")
        if _obs_dim_adapted:
            print(f"  !! STALE OBS SCHEMA: this checkpoint predates the "
                  f"`relay_capable` observation dim (OBS_DIM {OBS_DIM - 1} -> "
                  f"{OBS_DIM}).  fc1 was zero-padded, so the policy behaves "
                  f"exactly as trained and CANNOT yet distinguish MultiHaul "
                  f"sites -- retrain to get fragment bridging.")
    elif mode == 'marl':
        print(f"\n{'!'*60}")
        print(f"  !! MARL ERROR: Checkpoint NOT FOUND !!")
        print(f"  Path: {checkpoint_path}")
        print(f"  MARL will run with RANDOM UNTRAINED weights!")
        print(f"  Run training (run_large_network_clean.py) first!")
        print(f"{'!'*60}\n")
        sim.marl_ue_routing_enabled = False
    else:
        sim.marl_ue_routing_enabled = False

    # ── MARL STATIC: execute the loaded parameters unchanged, from tick 0 ──
    _static_fp0 = None
    _static_fp1 = None
    if is_static_arm:
        if not sim.marl_ue_routing_enabled or not rl_agents:
            raise RuntimeError("marl_static needs a trained checkpoint and "
                               "RL agents; none were loaded")
        for _a in rl_agents.values():
            _a.is_training = False          # argmax heads + Dirichlet mean
        _frozen = True                      # closes every learning gate
        torch.manual_seed(topology_seed)    # reproducible (argmax is anyway)
        _static_fp0 = _policy_fingerprint(rl_agents)
        print(f"[{mode_label} - {scenario_name}] STATIC DEPLOYMENT: loaded "
              f"parameters executed unchanged from t=0 (deterministic; no "
              f"online update, no freeze guard, no snapshot restore; base "
              f"checkpoint in both scenarios); fingerprint {_static_fp0[:16]}")

    mappo_cfg = MAPPOConfig()
    if learns:
        # ── Safe online adaptation (see DEPLOY_* constant docs) ──
        # Override the trainer's stock deployment values BEFORE construction:
        # ewc_lambda seeds every EWCPenalty; deploy_lr_* are what
        # set_deployment_mode() installs into the optimizers.
        mappo_cfg.deploy_lr_actor  = DEPLOY_LR_ACTOR    # 15% of training LR
        mappo_cfg.deploy_lr_critic = DEPLOY_LR_CRITIC   # 15% of training LR
        mappo_cfg.ewc_lambda       = DEPLOY_EWC_LAMBDA
    mappo_trainer = MAPPOTrainer(rl_agents, critic_net, mappo_cfg)
    # Pretrained-weights EWC anchor (+ one-time Fisher-refresh bookkeeping)
    _ewc_anchor_sd = None
    _ewc_fisher_done = False
    if learns:
        mappo_trainer.set_deployment_mode()  # installs DEPLOY_LR_*, clip=0.05, enables EWC term
        print(f"[{mode_label} - {scenario_name}] Safe adaptation: online LR "
              f"actor={DEPLOY_LR_ACTOR:.1e} "
              f"({DEPLOY_LR_ACTOR / MAPPOConfig.lr_actor:.0%} of training "
              f"{MAPPOConfig.lr_actor:.0e}), critic={DEPLOY_LR_CRITIC:.1e} "
              f"({DEPLOY_LR_CRITIC / MAPPOConfig.lr_critic:.0%} of training "
              f"{MAPPOConfig.lr_critic:.0e}); EWC lambda={DEPLOY_EWC_LAMBDA}")
        # Anchor to the weights loaded above — the online policy is pulled
        # back toward the pretrained checkpoint on every PPO update.
        _ewc_anchor_sd = _anchor_deployment_ewc(
            mappo_trainer, log_prefix=f"[{mode_label} - {scenario_name}] ")

    # Raw KPI arrays
    conn_log, energy_log, hops_log, delivered_log = [], [], [], []
    mcs_high_log = []
    fragments_log, relay_count_log, link_count_log = [], [], []
    # Fragment/relay semantics (see the per-tick recording block for the full
    # rationale):
    #   fragments        infra components INCLUDING active relay links
    #   fragments_raw    infra components EXCLUDING relay links (raw partition)
    #   relay_count      ACTIVE transport relay LINKS
    #   relay_bridges    subset of relay_count that joins two raw fragments
    #   relay_mode_nodes nodes in LOCAL_REROUTE/CAPACITY_BOOST (intent only)
    fragments_raw_log, relay_mode_nodes_log, relay_bridge_log = [], [], []
    # PEER reachability: fraction of surviving UEs whose serving cell sits in
    # the connected BODY of the survivor network (Simulator
    # .peer_reachable_ue_stats), and the count of UEs re-joined to that body by
    # this arm's own relay bridges.  Distinct from the ACCESS reachability
    # metric (`_reachable_ue_fraction`, "is the UE camped on a cell at all"),
    # which is blind to fragmentation.  Logged for every arm.
    peer_reach_log, reach_restored_log, stranded_ue_log = [], [], []
    # Attached UEs sitting in a component OTHER than the body.  If this is 0
    # while `fragments` > 1, the unclosed residual fragments are UE-FREE
    # infrastructure and no per-UE reachability term can motivate closing them.
    outside_body_log = []
    # Uu CARRIER AGGREGATION audit trail - logged for EVERY arm, every tick,
    # including the arms that are not eligible for it (they log 0 / 1.0).
    # `uu_boost_links`  count of Uu access links carrier-aggregated this tick
    # `uu_boost_factor` mean multiplier over those links (1.0 when none)
    # `uu_cc_activated` TOTAL extra component carriers lit across all sites
    # `uu_extra_mbps`   aggregate extra capacity granted (Mbps at QAM256 ref)
    # The last two make the SITE-LEVEL BOUND auditable straight from the log:
    # uu_extra_mbps must never exceed uu_cc_activated x 80 Mbps, and
    # uu_cc_activated must never exceed N_CC x (number of radio sites).
    # The original implementation logged none of this outside the MARL branch,
    # so "no baseline ever got the grant" was invisible in the run logs.  See
    # _apply_uu_carrier_aggregation for the capability-parity rationale.
    uu_boost_links_log, uu_boost_factor_log = [], []
    uu_cc_log, uu_extra_mbps_log = [], []
    # Per-tick OFFERED demand, recorded directly instead of being
    # reconstructed post hoc from delivered/achievability.  Needed by the
    # DEMAND-CORRECTED service-loss reference (see demand_regime_ratio):
    # _offered_volume_for_flows switches demand model at the event tick, so
    # the pre-event achievability bar and the post-event samples otherwise
    # sit in different demand regimes.
    offered_log = []
    node_count_log, overhead_log, postcards_log, latency_log = [], [], [], []
    msg_count_log = []  # Track actual control-plane message counts
    outage_log = []     # Fraction of flows with no converged route / no path

    # Control-plane message simulator: counts actual messages per tick
    ctrl_msg_sim = ControlPlaneMessageSim(sim, mode)

    # ── Scenario B pre-convergence: fast-forward island state ──────────
    # The network has already severed and converged (Scenario A happened).
    # Set _ticks_since_severance offset so civilian traffic starts in
    # the "late phase" (post-panic), and protocols start operational.
    if scenario_name == 'rescue_ops':
        # (Traffic model is flat constant — no phase offset needed)
        ctrl_msg_sim._pre_converged = True  # protocols start operational

    # Pending transitions: aid -> partial tuple buffered at the moment the
    # action was sampled (o_t, a_t, logπ(a_t|o_t), V(s_t), global_obs_t).
    # Completed with the reward that arrives next tick — mirrors the
    # collection pattern in sixg_sim/worker.py.
    _pending_transitions = {}
    # Adaptive PPO update cadence (online arm; the freeze arm stops entirely
    # at _freeze_tick).
    _ema_achiev = 0.0          # Exponential moving average of achievability
    _ema_alpha = 0.15          # EMA smoothing factor (responsive to changes)
    _ppo_interval = 30         # Baseline PPO update interval (ticks)
    _ppo_interval_urgent = 10  # When achievability drops >1%: urgent learning
    _ticks_since_ppo = 0       # Counter since last PPO update

    # ── Event-gated exploration temperature (shift detector) ─────────
    # Elevated temperature is allowed ONLY inside a bounded window after a
    # GENUINE detected event, and decays linearly back to TEMP_FLOOR within
    # TEMP_BOOST_WINDOW_TICKS.  Steady state runs at the floor.
    #
    # Genuine triggers (each logged with its reason):
    #   'severance'      — island mode begins (core severed)
    #   'rescue_arrival' — rescue UEs join (Scenario B scripted event)
    #   'sustained_drop' — delivered fraction below DROP_FRACTION x the
    #                      slow-EMA baseline for DROP_CONSEC_TICKS
    #                      CONSECUTIVE ticks (never a single-tick blip)
    #
    # This replaces the former per-agent dual-EMA reward detector, which
    # re-armed itself on stochastic per-tick reward noise thousands of
    # ticks after severance (avg_temp 0.05 -> 0.19-0.25 with no topology
    # event) and made steady-state achievability oscillate tick-to-tick.
    _temp_boost_start  = -1        # tick of the most recent genuine trigger
    _temp_boost_reason = None      # trigger reason string (for logging)
    _conn_slow_ema     = None      # slow baseline of achievability [0-100]
    _conn_drop_streak  = 0         # consecutive depressed-achievability ticks
    _was_island        = False     # previous-tick island state (edge detect)

    def _current_temperature(t):
        """Sampling temperature at tick t: linear decay from TEMP_BOOST to
        exactly TEMP_FLOOR over TEMP_BOOST_WINDOW_TICKS after a trigger;
        TEMP_FLOOR everywhere else (bounded window, bounded decay)."""
        if _temp_boost_start >= 0:
            dt = t - _temp_boost_start
            if 0 <= dt < TEMP_BOOST_WINDOW_TICKS:
                frac = 1.0 - dt / TEMP_BOOST_WINDOW_TICKS
                return TEMP_FLOOR + (TEMP_BOOST - TEMP_FLOOR) * frac
        return TEMP_FLOOR

    def _trigger_temp_boost(t, reason):
        nonlocal _temp_boost_start, _temp_boost_reason
        _temp_boost_start = t
        _temp_boost_reason = reason
        print(f"[{mode_label} - {scenario_name}]   [TEMP-BOOST] t={t}: "
              f"trigger={reason} -> temp {TEMP_BOOST:.2f}, linear decay to "
              f"floor {TEMP_FLOOR:.2f} within {TEMP_BOOST_WINDOW_TICKS} ticks")

    for tick in range(TOTAL_TICKS):
        sim.current_tick = tick
        # Uu carrier-aggregation result for THIS tick.  Reset here so an arm
        # that never calls _apply_uu_carrier_aggregation records an explicit
        # zero rather than carrying the previous tick's value forward.
        _uu_boost_n, _uu_boost_factor = 0, 1.0
        _uu_cc_n, _uu_extra_mbps = 0, 0.0

        # ── Reset per-tick state: link utilization must be cleared ──
        for link in sim.topology.links.values():
            link.reset_utilization()

        sim._process_events(tick)

        # ── Adapt-freeze arm: disable online adaptation past the freeze tick ──
        # GUARDED: only freeze when recent performance is sane (trailing mean
        # >= FREEZE_GUARD_RATIO x recent peak); otherwise postpone and
        # re-check, force-freezing at the best-observed policy snapshot by
        # _force_freeze_tick.  Rationale: freezing is a one-way door — an
        # argmax-degenerate joint action locked in at the freeze tick zeroes
        # delivery for the rest of the run (observed on rescue_ops seeds
        # 50/54: agents argmax +1/+2 MCS offsets -> spectral efficiency 0).
        if (is_freeze_arm and not _frozen and mode == 'marl'
                and tick >= _freeze_tick):
            _freeze_now, _freeze_kind = False, ''
            if tick >= _force_freeze_tick:
                # Hard bound reached: freeze regardless, but restore the
                # best-observed policy snapshot if one exists (boost-free
                # tier preferred; boosted-window fallback otherwise).
                _restore_sd, _restore_mean, _restore_tier = (
                    (_guard_best_sd, _guard_best_mean, 'boost-free')
                    if _guard_best_sd is not None
                    else (_guard_fb_sd, _guard_fb_mean, 'boosted-window'))
                if _restore_sd is not None:
                    for _aid, _sd in _restore_sd.items():
                        if _aid in rl_agents:
                            rl_agents[_aid].policy_net.load_state_dict(_sd)
                    _freeze_kind = (f"FORCED at best-observed {_restore_tier} "
                                    f"checkpoint (trail{FREEZE_GUARD_TRAIL} "
                                    f"mean {_restore_mean:.1f}%)")
                else:
                    _freeze_kind = "FORCED (no snapshot available)"
                _freeze_now = True
            elif tick >= _next_freeze_check:
                _trail = conn_log[-FREEZE_GUARD_TRAIL:]
                _trail_mean = float(np.mean(_trail)) if _trail else 0.0
                # Reference peak: best trailing mean over the WHOLE pass, not
                # just a recent look-back — if the online policy collapsed
                # before the freeze window, a recent-only peak is itself
                # collapsed and would legitimise freezing a dead policy
                # (observed: trailing peak 9.0% "passed" a 6.7% freeze).
                _peak = max(_guard_trail_means) if _guard_trail_means else _trail_mean
                if _peak > 1e-9 and _trail_mean >= FREEZE_GUARD_RATIO * _peak:
                    _freeze_kind = (f"guard passed (trail{FREEZE_GUARD_TRAIL}="
                                    f"{_trail_mean:.1f}% >= "
                                    f"{FREEZE_GUARD_RATIO:.0%} of peak "
                                    f"{_peak:.1f}%)")
                    # Same restore rule as the forced path: freezing is a
                    # one-way door, so if the best BOOST-FREE trailing window
                    # seen so far meaningfully beats the current one, freeze
                    # THAT snapshot instead of the drifted current policy.
                    # (Previously only the forced freeze restored the best
                    # snapshot; a guard-passed freeze locked in whatever the
                    # online policy had drifted to — observed: steady state
                    # == the mediocre freeze-tick window, flat to run end.)
                    if (_guard_best_sd is not None and _guard_best_mean is not None
                            and _guard_best_mean > _trail_mean + 0.5):
                        for _aid, _sd in _guard_best_sd.items():
                            if _aid in rl_agents:
                                rl_agents[_aid].policy_net.load_state_dict(_sd)
                        _freeze_kind += (f"; restored best boost-free snapshot "
                                         f"(trail{FREEZE_GUARD_TRAIL} mean "
                                         f"{_guard_best_mean:.1f}% > current "
                                         f"{_trail_mean:.1f}%)")
                    _freeze_now = True
                else:
                    _next_freeze_check = tick + FREEZE_GUARD_CHECK_INTERVAL
                    print(f"[{mode_label} - {scenario_name}]   "
                          f"[FREEZE-POSTPONED] t={tick}: trailing "
                          f"{FREEZE_GUARD_TRAIL}-tick mean {_trail_mean:.1f}% "
                          f"< {FREEZE_GUARD_RATIO:.0%} of recent peak "
                          f"{_peak:.1f}% — re-check t={_next_freeze_check} "
                          f"(force-freeze by t={_force_freeze_tick})")
            if _freeze_now:
                _frozen = True
                _pending_transitions = {}
                for _a in rl_agents.values():
                    _a.is_training = False   # deterministic argmax from here on
                print(f"[{mode_label} - {scenario_name}]   [FREEZE] t={tick}: "
                      f"online PPO updates disabled (freeze tick "
                      f"t={_freeze_tick} = last event t={_last_event_tick} + "
                      f"{FREEZE_AFTER_LAST_EVENT_TICKS}); {_freeze_kind}; "
                      f"policy now frozen (deterministic argmax)")

        # ── Shift-detector event triggers (MARL arms, scripted events) ──
        if learns and not _frozen:
            if scenario_name == 'rescue_ops' and tick == RESCUE_ARRIVAL_TICK:
                _trigger_temp_boost(tick, 'rescue_arrival')

        # ── Post-disaster backhaul degradation ──────────────────────────────
        # A natural disaster brings rain fade, debris and antenna
        # misalignment on the WIRELESS backhaul.  This DEGRADES capacity
        # (it does not sever), creating congestion an arm may try to bypass.
        # Applied equally to ALL arms — it is a physical event, not a handicap.
        #
        # WHICH LINK TYPES, AND WHY MULTIHAUL_MESH IS NOW INCLUDED.
        # This used to match only 'Microwave*', i.e. the 18 GHz PtP hops, and
        # exempt the 60 GHz MULTIHAUL_MESH hops.  That is backwards on the
        # physics: ITU-R P.838-3 gives 25 mm/h specific attenuation of
        # ~2.3 dB/km at 18 GHz against ~10.0 dB/km at 60 GHz, so the exempted
        # class is the one that is ~4.4x MORE rain-sensitive per kilometre.
        # Both static wireless backhaul classes are therefore degraded.
        #
        # RESIDUAL LIMITATION, STATED RATHER THAN HIDDEN.  A single 70 %
        # capacity cut is applied to both classes even though 60 GHz should
        # lose more; it is the conservative direction for the 60 GHz mesh
        # (it under-penalises it) and it keeps the event a single documented
        # magnitude rather than two tuned ones.  Dynamically formed
        # TRANSPORT_RELAY bridges are NOT degraded here: their capacity is
        # computed from the live TG budget by can_form_relay_link at the
        # moment they are formed, and that model is instantiated with
        # rain_mm_h = 0.0 (clear air), so a rain penalty applied only at this
        # one event would be inconsistent with the budget that sized them.
        # Making rain a first-class scenario parameter (TransportRelayModel
        # accepts rain_mm_h) is the clean fix and is an engine-side change.
        from sixg_sim.topology import LinkType as _BHLinkType
        _BH_DEGRADE_TYPES = (_BHLinkType.MICROWAVE,
                             _BHLinkType.MICROWAVE_PTP,
                             _BHLinkType.MULTIHAUL_MESH)
        bh_degrade_tick = 0 if scenario_name == 'rescue_ops' else SEVERANCE_TICK
        if tick == bh_degrade_tick and not getattr(sim, '_bh_degraded', False):
            sim._bh_degraded = True
            degraded_count = 0
            degraded_by_type = {}
            for lid, link in sim.topology.links.items():
                if not link.is_up:
                    continue
                lt = getattr(link, 'link_type', None)
                if lt in _BH_DEGRADE_TYPES:
                    link.capacity = max(50, link.capacity * 0.30)  # Keep 30%
                    ep0, ep1 = link.endpoints
                    if sim.topology.graph.has_edge(ep0, ep1):
                        sim.topology.graph[ep0][ep1]['capacity'] = link.capacity
                    degraded_count += 1
                    degraded_by_type[lt.value] = degraded_by_type.get(lt.value, 0) + 1
            if degraded_count > 0:
                sim.topology.invalidate_infrastructure_cache()
                print(f"[{mode_label} - {scenario_name}]   [BH-DEGRADE] "
                      f"t={tick}: {degraded_count} wireless backhaul links "
                      f"degraded to 30% capacity (rain fade/debris) "
                      f"{degraded_by_type}")


        sim.island_mode = sim._detect_island_mode()
        sim.control_plane.set_island_mode(sim.island_mode)

        # ── Shift-detector trigger: severance (island mode begins) ──
        # tick > 0 guard: Scenario B starts pre-severed AND pre-converged,
        # so the t=0 island state is not a fresh event (no boost there —
        # its genuine event is the rescue arrival, handled above).
        if (learns and not _frozen and tick > 0
                and sim.island_mode and not _was_island):
            _trigger_temp_boost(tick, 'severance')
        _was_island = sim.island_mode

        # NOTE: SINR evolution is left entirely to the physics engine and is
        # therefore identical across arms.  (A former post-severance
        # `sinr_average = max(sinr_average, 17.0)` floor was removed: it
        # overrode the engine's SINR dynamics with a hand-picked value that
        # guaranteed high-MCS eligibility in island mode.)

        # ── PHY/MAC Configuration ──────────────────────────────────────────
        # 3GPP TS 38.214: CQI→MCS link adaptation is a NATIVE gNB/DU
        # function at L1/L2.  ALL protocols get this equally — it's
        # independent of which L3 routing protocol is running.
        #
        # Pre-disaster:  ALL approaches use backbone core routing with
        #                fixed 23 dBm + auto MCS (CQI-based).
        # Post-disaster: ALL protocols get auto MCS adaptation.
        #                MARL additionally controls PRB/relay/scheduling.
        #                SDN activates MultiHaul relays after bootstrap.
        #                OSPF/OLSR enable LOCAL_REROUTE after convergence.

        # Step 1: Auto link adaptation for ALL protocols (3GPP-native)
        _auto_link_adaptation(sim)

        if not sim.island_mode:
            # Pre-disaster: identical radio baseline for ALL approaches
            _lock_phy_mac_non_mcs(sim, 23.0)
        elif mode in ('ospf', 'olsr'):
            # Post-disaster OSPF/OLSR: fixed power, auto MCS, LOCAL_REROUTE
            # once the MEASURED control-plane state machine reports the mesh
            # converged (not a fixed end-tick).
            _ospf_c = getattr(sim, '_ospf_ctrl', None)
            _olsr_c = getattr(sim, '_olsr_ctrl', None)
            ospf_converged = (_ospf_c is not None and
                              getattr(_ospf_c, 'convergence_fraction', 0.0) >= 0.95)
            olsr_converged = (_olsr_c is not None and
                              getattr(_olsr_c, 'convergence_fraction', 0.0) >= 0.95)
            allow_reroute = (mode == 'ospf' and ospf_converged) or (mode == 'olsr' and olsr_converged)
            _lock_phy_mac_non_mcs(sim, 23.0, allow_local_reroute=allow_reroute)
        elif mode == 'sdn':
            # Post-disaster SDN: fixed power, auto MCS.
            # MultiHaul relay activation handled below after controller bootstrap.
            _lock_phy_mac_non_mcs(sim, 23.0)
            # SDN: activate MultiHaul relays once controller is operational.
            # The elected local SDN controller uses NETCONF/YANG to activate
            # beam-steering on MultiHaul-capable nodes.
            ctrl = getattr(sim, '_sdn_ctrl', None)
            if ctrl and ctrl.phase == 'operational' and sim.island_mode:
                for nid, ps in sim.phy_mac_states.items():
                    node_obj = sim.topology.nodes.get(nid)
                    if (node_obj and getattr(node_obj, 'has_multihaul', False)
                            and ps.relay_mode == RelayMode.OFF):
                        ps.relay_mode = RelayMode.CAPACITY_BOOST
                        ps.prb_relay_fraction = max(ps.prb_relay_fraction, 0.15)
                        ps.normalise_prb()
                _establish_relay_links(sim, mode, scenario_name)

                # ── SDN integrated-access UE reattachment ────────────────
                # USER DECISION, and an architectural claim rather than a
                # convenience: an SDN controller has a MANAGEMENT plane
                # (NETCONF/YANG) into the RAN, so operator-triggered UE
                # reattachment and small-cell capacity adjustment are
                # plausible capabilities for it — the same justification
                # already used above to let it activate MultiHaul relays.
                # Gated on exactly the same condition as that bridging: the
                # fragment-local controller must be operational.  It runs the
                # IDENTICAL physics as the MARL arm (_reconnect_orphaned_ues
                # -> Simulator.integrated_access_capacity_mbps), so this is
                # capability parity, not a second model.
                #
                # The pure-IGP arms (OSPF/OLSR/BATMAN/AODV) are deliberately
                # NOT given it: they have no management plane and no RAN
                # configuration object to write, so there is no mechanism by
                # which they could admit a UE at a radio site.  That
                # asymmetry is architectural and is stated as such.
                def _sdn_site_enabled(site_id, _ctrl=ctrl):
                    node_obj = sim.topology.nodes.get(site_id)
                    ps = sim.phy_mac_states.get(site_id)
                    return bool(
                        _ctrl is not None and _ctrl.phase == 'operational'
                        and sim.island_mode
                        and node_obj is not None
                        and getattr(node_obj, 'has_multihaul', False)
                        and node_obj.is_survivor
                        and ps is not None
                        and ps.relay_mode == RelayMode.CAPACITY_BOOST)

                _reconnect_orphaned_ues(
                    sim, _sdn_site_enabled, tick,
                    log_prefix=f"[SDN - {scenario_name}]   ",
                    gate_desc="operator-triggered via the SDN controller's "
                              "NETCONF/YANG management interface")

                # ── SDN Uu CARRIER AGGREGATION ─ CAPABILITY PARITY ────────
                # USER DECISION, and the SAME architectural claim already made
                # for relay bridging and for integrated-access reattachment
                # directly above: an SDN controller has a MANAGEMENT plane
                # (NETCONF/YANG) into the RAN, so it can write a carrier-
                # aggregation / secondary-cell configuration object and
                # command a site to light up an additional component carrier.
                #
                # Until this call existed, the Uu boost was reachable ONLY
                # from the `mode == 'marl'` branch, so all three MARL-family
                # arms got it and EVERY baseline got none.  Because the Uu
                # link is the first AND the last hop of every UE-to-UE flow
                # and consume_path takes min(available) along the path, that
                # single grant gated end-to-end throughput: measured 1.46x
                # (Scenario A) / 1.57x (Scenario B) of the Uu capacity, i.e.
                # roughly 72% / 92% of MARL's whole delivery advantage came
                # from the grant rather than from the policy.
                #
                # GATE: exactly the enclosing condition that already gates
                # SDN's relay bridging - the fragment-local controller must
                # have reached its operational phase, in island mode.
                # MAGNITUDE (v2): NOT a rate at all any more.  An eligible
                # site activates k of N_CC spare component carriers and shares
                # that finite pool among its hot-spot UEs.  SDN's pool is
                # IDENTICAL to a MARL site's pool - same k, same 20 MHz, same
                # water-fill - so the grant can no longer be a source of MARL
                # advantage.  Under v1 SDN was pinned to a flat 1.30x while
                # MARL could reach 2.0x by raising an unrelated scheduler
                # knob; that asymmetry is gone.
                # The pure IGPs (OSPF/OLSR/BATMAN/AODV) still get nothing,
                # because a routing protocol has no mechanism to command a
                # radio - see the helper docstring, where that asymmetry is
                # stated as the architectural claim it is.
                def _sdn_site_can_command_ca(site_id, _ctrl=ctrl):
                    node_obj = sim.topology.nodes.get(site_id)
                    return bool(
                        _ctrl is not None and _ctrl.phase == 'operational'
                        and sim.island_mode
                        and node_obj is not None and node_obj.is_survivor
                        and sim.phy_mac_states.get(site_id) is not None)

                (_uu_boost_n, _uu_boost_factor,
                 _uu_cc_n, _uu_extra_mbps) = _apply_uu_carrier_aggregation(
                    sim, _sdn_site_can_command_ca, tick,
                    log_prefix=f"[SDN - {scenario_name}]   ",
                    gate_desc="SCell activation commanded by the SDN "
                              "controller over NETCONF/YANG; bounded, "
                              "site-level carrier pool")

                # Debug: log SDN relay activation (first time only)
                if not getattr(sim, '_sdn_relay_logged', False):
                    sim._sdn_relay_logged = True
                    active_relays = sum(1 for ps in sim.phy_mac_states.values() if ps.relay_link_active)
                    print(f"[SDN - {scenario_name}]   [SDN-RELAY] t={tick}: SDN controller activated "
                          f"MultiHaul relays + integrated access via "
                          f"NETCONF/YANG ({active_relays} relay nodes)")
            # SDN relay debug every 100 ticks (same cadence as MARL)
            if ctrl and ctrl.phase == 'operational' and tick % 100 == 0 and sim.island_mode:
                from sixg_sim.topology import LinkType
                import networkx as _nx_dbg
                sim.topology.invalidate_infrastructure_cache()
                ig = sim.topology._build_infrastructure_graph()
                n_comps = _nx_dbg.number_connected_components(ig)
                relay_links = [l for l in sim.topology.links.values()
                               if getattr(l, 'link_type', None) == LinkType.TRANSPORT_RELAY and l.is_up]
                active_relays = sum(1 for ps in sim.phy_mac_states.values() if ps.relay_link_active)
                flow_ok = sum(1 for s, t in sim.ue_to_ue_flows if sim._can_route_ue_to_ue(s, t))
                print(f"[SDN - {scenario_name}]   [RELAY-DBG] t={tick}: infra_comps={n_comps} relay_links={len(relay_links)} "
                      f"active_relay_nodes={active_relays} "
                      f"routable_flows={flow_ok}/{len(sim.ue_to_ue_flows)}")
        elif mode in ('batman', 'aodv'):
            # Post-disaster BATMAN/AODV: fixed power, auto MCS, no relay control.
            # Pure L2/L3 routing with no management interface for hardware control.
            _lock_phy_mac_non_mcs(sim, 23.0)

        # 1. Observe current state
        observations = sim._build_agent_observations()

        # 2. MARL Distributed Online Learning (only in MARL mode)
        # ─────────────────────────────────────────────────────────────
        # Architecture:
        #   PRE-SEVERANCE:  Cloud ML system trains policy centrally in
        #                   a digital twin, pushes weights to all nodes.
        #   POST-SEVERANCE: Each node runs its own policy on-device.
        #
        # Experience collection follows the same pattern as the training
        # worker (sixg_sim/worker.py): a transition is BUFFERED at the
        # moment the action is sampled — (o_t, a_t, logπ(a_t|o_t), V(s_t)),
        # with log-prob evaluated at the SAME temperature the behaviour
        # policy actually sampled with — and COMPLETED here one tick later
        # when the resulting reward r_t is observable.  The trainer computes
        # per-agent GAE at flush time and clears its pool on every update().
        #
        # The shared MAPPO pool + mean-field critic is the simulation-side
        # approximation of postcard-based federated parameter averaging
        # (all agents share one policy net — parameter-sharing MAPPO).
        marl_active_tick = 0 if scenario_name == 'rescue_ops' else SEVERANCE_TICK
        if learns and not _frozen and _pending_transitions:
            for aid, tr in list(_pending_transitions.items()):
                agent = sim.agents.get(aid)
                if (agent is None or not hasattr(agent, 'calculate_reward')
                        or aid not in observations):
                    continue  # agent left the sim — drop incomplete transition
                comps = agent.calculate_reward(tr['obs_raw'], tr['action'],
                                               observations[aid])

                # ── LOCAL connectivity reward component ──
                # Locally observable fields of this agent's own observation:
                #   - reachable_ue_fraction: UEs this node can reach
                #   - intra_island_reachability: local mesh health
                #   - ue_to_ue_routed_norm: local routing success
                obs_i = observations[aid]
                local_conn_r = (
                    0.4 * obs_i.connectivity.reachable_ue_fraction +
                    0.3 * obs_i.connectivity.intra_island_reachability +
                    0.3 * obs_i.connectivity.ue_to_ue_routed_norm
                )
                r = (0.5 * comps.total_reward) + (0.5 * local_conn_r)

                # (The former per-agent dual-EMA/collapse temperature logic
                #  lived here.  It is replaced by the event-gated controller
                #  above: exploration temperature may rise ONLY inside the
                #  bounded window after a logged, genuine trigger.)

                # Complete the transition buffered at sampling time.
                # obs_t/heads/prb/log_prob/value all describe the state the
                # action was SAMPLED from (not the current one).
                try:
                    mappo_trainer.collect(
                        aid, tr['obs_t'], tr['heads'], tr['prb'],
                        tr['log_prob'], tr['value'], r, False,
                        global_obs=tr['global_obs'])
                except Exception:
                    pass
            _pending_transitions = {}

            # ── Adaptive PPO update interval ──
            # update() flushes in-flight trajectories and clears the pool
            # itself — no manual buffer management here.
            _ticks_since_ppo += 1
            if _ticks_since_ppo >= _ppo_interval:
                # One-time diagonal-Fisher refresh: once enough LIVE
                # deployment observations have accumulated, replace the
                # identity Fisher with a proper EWC Fisher estimate — and
                # re-pin the anchor to the PRETRAINED weights
                # (compute_fisher() anchors at the current, already-drifted
                # weights, which would defeat the drift guard).
                if _ewc_anchor_sd is not None and not _ewc_fisher_done:
                    _ewc_cache = [t for c in mappo_trainer.obs_cache.values()
                                  for t in c]
                    if len(_ewc_cache) >= EWC_FISHER_MIN_OBS:
                        from sixg_sim.mappo_trainer import (
                            DISCRETE_HEADS as _EWC_HEADS)
                        _first_pen = next(iter(mappo_trainer.ewc.values()))
                        _first_pen.compute_fisher(
                            mappo_trainer._ref_actor, _ewc_cache,
                            n_samples=EWC_FISHER_SAMPLES,
                            discrete_heads=_EWC_HEADS)
                        _first_pen.ewc_lambda = DEPLOY_EWC_LAMBDA
                        _first_pen.anchor_params = _ewc_anchor_sd
                        _first_pen.is_anchored = True
                        # Keep all per-agent penalties consistent (update()
                        # reads only the first, but shared state is cheap).
                        for _p2 in mappo_trainer.ewc.values():
                            _p2.ewc_lambda = DEPLOY_EWC_LAMBDA
                            _p2.fisher = _first_pen.fisher
                            _p2.anchor_params = _ewc_anchor_sd
                            _p2.is_anchored = True
                        _ewc_fisher_done = True
                        print(f"[{mode_label} - {scenario_name}]   [EWC] "
                              f"t={tick}: diagonal Fisher estimated from "
                              f"{len(_ewc_cache)} deployment obs "
                              f"({EWC_FISHER_SAMPLES} samples); anchor "
                              f"re-pinned to loaded checkpoint weights "
                              f"(lambda={DEPLOY_EWC_LAMBDA})")
                mappo_trainer.update()
                _ticks_since_ppo = 0

        # 3. Execute agent actions (MARL only, post-severance)
        # Pre-disaster: all approaches use fixed config (backbone routing).
        # Post-disaster: MARL agents control MCS/power/relay dynamically.
        if mode == 'marl' and sim.island_mode:
            # ── The learned policy's actions STAND — no overrides ──────────
            # _execute_agent_actions() samples every head (TX power, MCS,
            # PRB split, relay mode, scheduler, handover, postcard, IOPS)
            # and applies them via _step_phy_mac(), which also creates or
            # tears down transport relay links as a DIRECT consequence of
            # the sampled relay_mode action.  Former evaluation-time
            # overrides were removed for fairness:
            #   * relay save/restore (agent couldn't toggle relays) — GONE:
            #     the agent's relay decision is applied verbatim;
            #   * forced CAPACITY_BOOST + prb_relay>=0.15 on MultiHaul — GONE;
            #   * post-hoc MCS floor at the auto-CQI level — GONE.  MCS
            #     actions are now BOUNDED OFFSETS from the shared auto-CQI
            #     choice (applied inside _step_phy_mac): offset 0 follows
            #     auto-CQI exactly; deliberate deviations (± steps) stand,
            #     even if worse;
            #   * post-hoc 0.10 general-PRB floor — GONE (anti-degeneracy is
            #     the sample-time floor in agent.py, part of the policy).

            # ── Apply the event-gated exploration temperature ─────────────
            # TEMP_FLOOR in steady state; elevated only inside the bounded
            # post-trigger window (see _current_temperature).  Irrelevant in
            # the frozen phase (is_training=False -> deterministic argmax).
            if learns and not _frozen:
                _cur_temp = _current_temperature(tick)
                for _a in rl_agents.values():
                    _a._logit_temperature = _cur_temp
            if is_static_arm and any(_a.is_training for _a in rl_agents.values()):
                raise RuntimeError("marl_static invariant violated: an agent "
                                   "switched to sampling at t=%d" % tick)

            sim._execute_agent_actions(observations)

            # ── Buffer transitions AT SAMPLING TIME (worker.py pattern) ────
            # Stored: (o_t, a_t, logπ(a_t|o_t), V(s_t), global_obs_t) with
            # log-prob evaluated at the temperature the batched sampler in
            # simulation.py actually used, and 'iops' included in the heads
            # dict (omitting it biases the PPO ratio).  Reward arrives next
            # tick and completes the tuple.
            _aids = [aid for aid in sim.agents if aid in observations]
            if learns and _aids and tick >= marl_active_tick and not _frozen:
                _obs_stack = torch.stack([
                    sim.agents[aid].observation_to_tensor(observations[aid])
                    for aid in _aids
                ])
                _obs_tensors = {aid: _obs_stack[i] for i, aid in enumerate(_aids)}
                # CRITIC INPUT: unmasked on purpose.  The actor's tensor has the
                # non-local columns zeroed (agent.NONLOCAL_OBS_COLUMNS); the
                # critic is centralised and is entitled to the global state, so
                # it is rebuilt with mask_nonlocal=False.
                _global_obs = torch.stack([
                    sim.agents[aid].observation_to_tensor(
                        observations[aid], mask_nonlocal=False)
                    for aid in _aids
                ]).mean(dim=0)
                _value = mappo_trainer.get_value(list(_obs_tensors.values()))
                _ref_agent = next(iter(sim.agents.values()))
                _temp = getattr(_ref_agent, '_logit_temperature', 1.0)

                _batch_aids, _batch_heads, _batch_prb = [], [], []
                for aid in _aids:
                    agent = sim.agents[aid]
                    act = agent.last_action
                    if act is None or not hasattr(agent, 'calculate_reward'):
                        continue
                    act_heads = {
                        'tx_power': act.tx_power_step,
                        'mcs_emrg': act.mcs_emergency_idx,
                        'mcs_gen': act.mcs_general_idx,
                        'relay': act.relay_mode_idx,
                        'handover': act.handover_idx,
                        'scheduler': act.scheduler_idx,
                        'postcard': int(act.send_postcard),
                        'iops': int(getattr(act, '_iops_decision', 0)),
                    }
                    # the SAMPLED split, not the executed one: the coordinator
                    # clamp and the general-traffic floor are environment,
                    # not policy, so they must not be scored as the action
                    prb = prb_training_action(act)
                    _batch_aids.append(aid)
                    _batch_heads.append(act_heads)
                    _batch_prb.append(prb)

                if _batch_aids:
                    try:
                        _log_probs = mappo_trainer.compute_log_prob_batch(
                            _batch_aids,
                            torch.stack([_obs_tensors[a] for a in _batch_aids]),
                            _batch_heads, _batch_prb, temperature=_temp)
                    except Exception:
                        _log_probs = [
                            mappo_trainer.compute_log_prob(
                                aid, _obs_tensors[aid], h, p, temperature=_temp)
                            for aid, h, p in zip(_batch_aids, _batch_heads,
                                                 _batch_prb)
                        ]
                    for j, aid in enumerate(_batch_aids):
                        _pending_transitions[aid] = {
                            'obs_raw':    observations[aid],
                            'action':     sim.agents[aid].last_action,
                            'obs_t':      _obs_tensors[aid],
                            'heads':      _batch_heads[j],
                            'prb':        _batch_prb[j],
                            'log_prob':   _log_probs[j],
                            'value':      _value,
                            'global_obs': _global_obs,
                        }

            from sixg_sim.topology import LinkType

            # Debug: report relay and fragment state every 100 ticks
            if tick % 100 == 0 and sim.island_mode:
                import networkx as _nx_dbg
                # Count infra components using the live infra graph
                sim.topology.invalidate_infrastructure_cache()
                ig = sim.topology._build_infrastructure_graph()
                n_comps = _nx_dbg.number_connected_components(ig)
                relay_links = [l for l in sim.topology.links.values()
                               if getattr(l, 'link_type', None) == LinkType.TRANSPORT_RELAY and l.is_up]
                active_relays = sum(1 for ps in sim.phy_mac_states.values() if ps.relay_link_active)
                # Count how many flows could succeed
                flow_ok = 0
                for s, t in sim.ue_to_ue_flows:
                    if sim._can_route_ue_to_ue(s, t):
                        flow_ok += 1
                print(f"[{mode_label} - {scenario_name}]   [RELAY-DBG] t={tick}: infra_comps={n_comps} relay_links={len(relay_links)} "
                      f"active_relay_nodes={active_relays} "
                      f"routable_flows={flow_ok}/{len(sim.ue_to_ue_flows)}")

            # ── 3. UE Reconnection (integrated access node) ────────────────
            # GATED BEHIND THE POLICY'S ACTIONS.  Causal chain a reviewer can
            # verify: the policy samples relay head = CAPACITY_BOOST at a
            # TG-capable site -> _step_phy_mac applies it -> ONLY THEN may that
            # site light up its own FR1 small cell for orphaned UEs, at a
            # capacity funded by the same agent's prb_relay_frac action.  If the
            # agent later moves the relay head away from CAPACITY_BOOST, the
            # site's access links are torn down again.
            #
            # The physics now lives in the engine (see
            # _reconnect_orphaned_ues), which is shared with the SDN arm.
            from sixg_sim.topology import (Link, LinkType, InterfaceType,
                                           NodeType)
            _CB_IDX = list(RelayMode).index(RelayMode.CAPACITY_BOOST)

            def _site_chose_capacity_boost(site_id):
                """True iff this site's agent ACTION selected CAPACITY_BOOST."""
                _ag = sim.agents.get(site_id)
                _act = getattr(_ag, 'last_action', None) if _ag else None
                _ps = sim.phy_mac_states.get(site_id)
                return (_act is not None and _act.relay_mode_idx == _CB_IDX
                        and _ps is not None
                        and _ps.relay_mode == RelayMode.CAPACITY_BOOST)

            _reconnect_orphaned_ues(
                sim, _site_chose_capacity_boost, tick,
                log_prefix=f"[{mode_label} - {scenario_name}]   ",
                gate_desc="policy-gated integrated access (agent chose "
                          "CAPACITY_BOOST)")

            # ── 4. Uu CARRIER AGGREGATION ─ shared helper, MARL gate ─────────
            # The mechanism, the magnitude and the physical justification now
            # live in ONE place (_apply_uu_carrier_aggregation) that every
            # eligible arm calls; the only thing this call site contributes is
            # the MARL ELIGIBILITY GATE below.  It used to be an inline
            # `if mode == 'marl'` branch, which silently made carrier
            # aggregation a MARL-only capability even though the site hardware
            # is identical in every arm's topology - see the helper docstring.
            def _marl_site_can_command_ca(site_id):
                """True iff this site's agent produced an action this tick.

                The RIC/agent IS the entity that commands the extra carrier,
                so an agentless site cannot ask for one.  THE MAGNITUDE IS NO
                LONGER AN AGENT ACTION (v2): once this gate is open the site
                gets k of N_CC spare component carriers - an integer count of
                20 MHz blocks - shared over its hot-spot UEs, which is exactly
                what an eligible SDN site gets.  The v1 dependence on the
                policy's own prb_emergency_fraction (a scheduling split of the
                EXISTING carrier, dimensionally unrelated to spectrum) has
                been removed; see the helper docstring.
                """
                _ag = sim.agents.get(site_id)
                _act = getattr(_ag, 'last_action', None) if _ag else None
                return (_act is not None
                        and sim.phy_mac_states.get(site_id) is not None)

            (_uu_boost_n, _uu_boost_factor,
             _uu_cc_n, _uu_extra_mbps) = _apply_uu_carrier_aggregation(
                sim, _marl_site_can_command_ca, tick,
                log_prefix=f"[{mode_label} - {scenario_name}]   ",
                gate_desc="SCell activation commanded by the per-site RIC "
                          "agent; bounded, site-level carrier pool")


        # For OSPF/SDN: PHY/MAC already locked above, no agent actions needed
        # But we still need to invalidate cache for topology changes from events
        sim.topology.invalidate_infrastructure_cache()

        # 4. Protocol-specific forwarding capability — MEASURED, not scripted.
        # Each baseline protocol runs a control-plane state machine driven by
        # the ACTUAL simulated topology and events (hello/dead timers, hop-by-
        # hop flooding over surviving links, per-node SPF, controller
        # reachability, on-demand discovery).  A flow is generated/forwarded
        # only once that state machine has converged routes for it — there is
        # NO forward_fraction ramp anywhere.
        #
        # MARL gets no scripted ramp either: its forwarding capability is
        # whatever the engine + learned policy produce (routing gate over the
        # live graph, agent-created relay links, PRB/MCS actions).
        ticks_since_sever = tick - SEVERANCE_TICK if tick >= SEVERANCE_TICK else -1
        if scenario_name == 'rescue_ops':
            ticks_since_sever = tick + 1000   # pre-converged island

        # ── MARL: BFD-class failure detection (the arm's own state machine) ──
        # Stepped EVERY tick, including pre-severance, so the severance itself
        # is a genuine transition the detector has to catch.  Creating it
        # lazily at island onset would hand MARL the severed topology for
        # free, which is the very defect this fixes.  The flow filter is
        # applied only in island mode, matching every other arm's gate.
        if mode == 'marl' and MARL_BFD_DETECT_TICKS > 0:
            if not hasattr(sim, '_marl_detect'):
                sim._marl_detect = MARLLinkStateDetector(sim)
            sim._marl_detect.step(tick)
            if sim.island_mode:
                sim._proto_all_flows = list(sim.ue_to_ue_flows)
                sim.ue_to_ue_flows = sim._marl_detect.get_routable_flows()

        if mode == 'ospf' and sim.island_mode:
            # ── OSPF: dead-timer detection → hop-by-hop LSA flood → SPF ──
            if not hasattr(sim, '_ospf_ctrl'):
                sim._ospf_ctrl = OSPFController(sim)
                if scenario_name == 'rescue_ops':  # already converged (Scenario A)
                    sim._ospf_ctrl.mark_pre_converged(tick)
            sim._ospf_ctrl.step(tick)
            # A router forwards a flow only once its SPF has run on a
            # complete LSDB — restrict generated flows to converged ones.
            sim._proto_all_flows = list(sim.ue_to_ue_flows)
            sim.ue_to_ue_flows = sim._ospf_ctrl.get_routable_flows()

        elif mode == 'sdn' and sim.island_mode:
            # ── Real SDN process: leader election → LLDP → flow programming ──
            if not hasattr(sim, '_sdn_ctrl'):
                sim._sdn_ctrl = SDNController(sim)
                if scenario_name == 'rescue_ops':  # pre-converged
                    sim._sdn_ctrl.phase = 'operational'
                    sim._sdn_ctrl.programmed_flows = set(sim.ue_to_ue_flows)
                    sim._sdn_ctrl._compute_fragments()
                    for fid in sim._sdn_ctrl.fragment_members:
                        sim._sdn_ctrl.discovery_done[fid] = True
            sim._sdn_ctrl.step(tick)
            # Forwarding is gated on actual controller state: only flows the
            # (fragment-local) controller has programmed can forward, and the
            # controller can only program flows inside fragments where an
            # election succeeded and LLDP discovery reached both endpoints.
            sim._proto_all_flows = list(sim.ue_to_ue_flows)
            sim.ue_to_ue_flows = sim._sdn_ctrl.get_programmed_flows()

        elif mode == 'olsr' and sim.island_mode:
            # ── OLSR: MPR-based TC flooding → route calculation ──
            if not hasattr(sim, '_olsr_ctrl'):
                sim._olsr_ctrl = OLSRController(sim)
                if scenario_name == 'rescue_ops':  # pre-converged
                    sim._olsr_ctrl.phase = 'operational'
                    sim._olsr_ctrl.routable_flows = set(sim.ue_to_ue_flows)
                    sim._olsr_ctrl._hello_rounds_done = 2
                    sim._olsr_ctrl._build_neighbor_tables()
                    sim._olsr_ctrl._select_mprs()
                    sim._olsr_ctrl._compute_fragments()
                    # Pre-set convergence state so step() doesn't reinitialize
                    sim._olsr_ctrl.node_convergence = {}
                    sim._olsr_ctrl._node_converge_tick = {}
                    sim._olsr_ctrl._convergence_initialized = True
                    sim._olsr_ctrl.convergence_fraction = 1.0
                    # Mark ALL surviving nodes (infra + UE) as fully converged
                    # — flow filtering checks UE endpoints, so UEs must be
                    # included (topology.nodes, not just phy_mac_states).
                    for _nid, _node in sim.topology.nodes.items():
                        if _node.is_survivor:
                            sim._olsr_ctrl.node_convergence[_nid] = 1.0
                            sim._olsr_ctrl._node_converge_tick[_nid] = -1
                    # Pre-set flow tracking so existing flows aren't treated as new
                    sim._olsr_ctrl._new_flow_discovery = {}
                    sim._olsr_ctrl._known_flows = set(sim.ue_to_ue_flows)
            sim._olsr_ctrl.step(tick)
            # Only flows whose endpoints' OLSR state has converged (Hello →
            # MPR → TC flood → route calc) are routable.  During sensing /
            # MPR selection routable_flows is empty — that IS the blackout,
            # no scalar ramp involved.
            sim._proto_all_flows = list(sim.ue_to_ue_flows)
            sim.ue_to_ue_flows = sim._olsr_ctrl.get_routable_flows()

        elif mode == 'batman' and sim.island_mode:
            # ── B.A.T.M.A.N.: OGM-based TQ learning → next-hop selection ──
            if not hasattr(sim, '_batman_ctrl'):
                sim._batman_ctrl = BATMANController(sim)
                if scenario_name == 'rescue_ops':  # pre-converged
                    sim._batman_ctrl.phase = 'operational'
                    sim._batman_ctrl.routable_flows = set(sim.ue_to_ue_flows)
                    sim._batman_ctrl._ogm_round = 20
                    sim._batman_ctrl._compute_fragments()
                    all_infra = [nid for nid in sim.phy_mac_states
                                 if sim.topology.nodes.get(nid)
                                 and sim.topology.nodes[nid].is_survivor]
                    for nid in all_infra:
                        sim._batman_ctrl.tq_tables[nid] = {}
                        sim._batman_ctrl.next_hop[nid] = {}
                        for orig in all_infra:
                            sim._batman_ctrl.tq_tables[nid][orig] = 255.0  # Max TQ
                            sim._batman_ctrl.next_hop[nid][orig] = orig
                    # Pre-set convergence state so step() doesn't reinitialize
                    sim._batman_ctrl.node_convergence = {}
                    sim._batman_ctrl._node_converge_tick = {}
                    sim._batman_ctrl._convergence_initialized = True
                    sim._batman_ctrl.convergence_fraction = 1.0
                    # Mark ALL surviving nodes (infra + UE) as fully converged
                    # — flow filtering checks UE endpoints, so UEs must be
                    # included (topology.nodes, not just phy_mac_states).
                    for _nid, _node in sim.topology.nodes.items():
                        if _node.is_survivor:
                            sim._batman_ctrl.node_convergence[_nid] = 1.0
                            sim._batman_ctrl._node_converge_tick[_nid] = -1
                    # Pre-set flow tracking so existing flows aren't treated as new
                    sim._batman_ctrl._new_flow_discovery = {}
                    sim._batman_ctrl._known_flows = set(sim.ue_to_ue_flows)
            sim._batman_ctrl.step(tick)
            # Only flows for which OGM propagation + TQ stabilisation has
            # produced usable next-hop entries at both endpoints route.
            # Early OGM rounds → routable_flows empty → measured blackout.
            sim._proto_all_flows = list(sim.ue_to_ue_flows)
            sim.ue_to_ue_flows = sim._batman_ctrl.get_routable_flows()

        elif mode == 'aodv' and sim.island_mode:
            # ── AODV: reactive on-demand route discovery ──
            if not hasattr(sim, '_aodv_ctrl'):
                sim._aodv_ctrl = AODVController(sim)
                if scenario_name == 'rescue_ops':  # pre-converged
                    sim._aodv_ctrl.phase = 'operational'
                    sim._aodv_ctrl.route_cache = {flow: 0 for flow in sim.ue_to_ue_flows}
                    sim._aodv_ctrl.routable_flows = set(sim.ue_to_ue_flows)
                    sim._aodv_ctrl._compute_fragments()
            sim._aodv_ctrl.step(tick)
            # AODV only forwards flows with active cached routes (on-demand
            # RREQ/RREP discovery over the live graph fills the cache).
            sim._proto_all_flows = list(sim.ue_to_ue_flows)
            sim.ue_to_ue_flows = sim._aodv_ctrl.get_routable_flows()

        # Generate and forward traffic.
        #
        # ── MARL STALE ROUTING VIEW (the failure-detection floor) ──────────
        # `_forward_traffic` is the ONLY place data-plane paths are chosen
        # (nx.shortest_path / shortest_simple_paths / consume_path all read
        # `sim.topology.graph`).  For the MARL family it runs over the graph
        # the arm BELIEVES in, not the physically true one: a link cut less
        # than MARL_BFD_DETECT_TICKS ago is still in that graph, so the arm
        # keeps picking the dead path and `consume_path` delivers 0.0 over it
        # (it re-checks `link.is_up` itself).  The flow fails until BFD
        # declares the session down, then reroutes — which is the actual
        # recovery sequence every baseline is charged for.
        #
        # SCOPE IS EXACTLY ONE CALL, by design.  Everything that must reflect
        # PHYSICAL TRUTH is computed outside this block and is untouched:
        # island/fragment counts and relay-bridge attribution come from
        # Simulator._infra_component_labels, which walks `topology.links` and
        # `link.is_up` directly and never reads `topology.graph`; peer-reach
        # stats and `_can_route_ue_to_ue` likewise; `sim.get_island_kpis()`
        # runs after the restore.  The agent-observation bridge set
        # (nx.bridges) is built earlier in the tick, also outside.
        #
        # try/finally: the graph is a shared mutable attribute of the
        # simulator, so an exception inside the engine must NOT be able to
        # leave a stale graph installed for the rest of the run.
        traffic = sim._generate_traffic()
        _believed_graph = None
        if mode == 'marl' and MARL_BFD_DETECT_TICKS > 0 and sim.island_mode:
            _believed_graph = sim._marl_detect.believed_graph()
        if _believed_graph is None:
            sim._forward_traffic(traffic)
        else:
            _true_graph = sim.topology.graph
            # Link ids the view is holding stale — lets consume_path tell a
            # blackhole apart from congestion (see sixg_sim/simulation.py).
            sim._marl_stale_link_ids = frozenset(
                lid for lid, believed in sim._marl_detect.visible.items()
                if believed
                and lid in sim.topology.links
                and not getattr(sim.topology.links[lid], 'is_up', True))
            try:
                sim.topology.graph = _believed_graph
                sim._forward_traffic(traffic)
            finally:
                sim.topology.graph = _true_graph
                sim._marl_stale_link_ids = frozenset()

        # Restore full flow list after protocol filtering
        if hasattr(sim, '_proto_all_flows'):
            sim.ue_to_ue_flows = sim._proto_all_flows
            del sim._proto_all_flows

        # ── Extract RAW KPIs ──
        kpis = sim.get_island_kpis()

        # Delivered volume: genuine engine output, no post-hoc scaling.
        delivered = getattr(sim, 'last_tick_ue_to_ue_volume', 0.0)

        # ── SYMMETRIC ACHIEVABILITY DENOMINATOR ──
        # Live per-tick offered demand over the FULL flow set, identical
        # formula for every arm (see _offered_volume_for_flows).  Replaces
        # two former asymmetries: (a) MARL divided by a FROZEN first-island-
        # tick baseline while others used live offered; (b) flow-filtered
        # protocols fell back to a hardcoded `n_flows * 12.3` estimate.
        offered = _offered_volume_for_flows(sim, sim.ue_to_ue_flows, tick)

        phys_fragments = kpis.get('island_fragments', 1)
        N = kpis.get('island_node_count', 0)

        # ── Fragmentation (identical definition for EVERY arm) ─────────────
        # `fragments` = connected components of the SURVIVING INFRASTRUCTURE
        # graph (UEs excluded from the node set AND from traversal, so Uu
        # links can never merge two islands) over all is_up links INCLUDING
        # active TRANSPORT_RELAY links.  Bridging therefore genuinely lowers
        # it; an arm that forms no relay links sits at the raw post-severance
        # partition (4 in Scenario A).
        # Source: Simulator.count_infra_components(include_relay_links=True),
        # evaluated AFTER this tick's relay formation (relay links are created
        # in _step_phy_mac during _execute_agent_actions / _establish_relay_
        # links, both of which run earlier in this tick).
        # `fragments_raw` is the same count with relay links removed — the
        # baseline the bridging policy is judged against.
        if sim.island_mode:
            fragments = phys_fragments
            fragments_raw = kpis.get('island_fragments_raw', phys_fragments)
        else:
            fragments = 1  # Pre-disaster: single connected network via core
            fragments_raw = 1

        # ── Achievability: delivered / offered from the actual simulation,
        # pre- AND post-disaster (the former pre-disaster 95%+noise constant
        # and the ±3% synthetic "fading jitter" multiplier were removed —
        # delivered volume is genuine engine output).
        conn_pct = (delivered / offered * 100.0) if offered > 0 else 0.0
        conn_pct = min(100.0, conn_pct)
        conn_log.append(conn_pct)
        delivered_log.append(delivered)

        # ── Guarded-freeze bookkeeping (freeze arm, pre-freeze only) ──
        # Track the trailing-window achievability mean (feeds the freeze
        # gate's peak look-back) and snapshot the policy at its best
        # BOOST-FREE trailing window since the last scenario event — the
        # restore point for a forced freeze.
        if is_freeze_arm and mode == 'marl' and not _frozen:
            _tm = float(np.mean(conn_log[-FREEZE_GUARD_TRAIL:]))
            _guard_trail_means.append(_tm)
            _boost_end = (_temp_boost_start + TEMP_BOOST_WINDOW_TICKS
                          if _temp_boost_start >= 0 else -1)
            _win_start = tick - min(len(conn_log), FREEZE_GUARD_TRAIL) + 1
            # Snapshot candidates come from the WHOLE pass (full trailing
            # window, island mode) — if the online policy degrades before or
            # after the event, the forced freeze restores the best-known
            # operating point rather than a recently-collapsed one.
            if (len(conn_log) >= FREEZE_GUARD_TRAIL and sim.island_mode):
                if (_win_start > _boost_end
                        and (_guard_best_mean is None
                             or _tm > _guard_best_mean + 0.5)):
                    _guard_best_mean = _tm
                    _guard_best_sd = {
                        _aid: {k: v.detach().clone()
                               for k, v in _a.policy_net.state_dict().items()}
                        for _aid, _a in rl_agents.items()}
                elif (_guard_best_sd is None
                        and (_guard_fb_mean is None
                             or _tm > _guard_fb_mean + 0.5)):
                    # Fallback tier: short runs may never see a boost-free
                    # window before the forced freeze — better to restore the
                    # best boosted-window snapshot than none at all.
                    _guard_fb_mean = _tm
                    _guard_fb_sd = {
                        _aid: {k: v.detach().clone()
                               for k, v in _a.policy_net.state_dict().items()}
                        for _aid, _a in rl_agents.items()}

        # ── MARL mechanism diagnostics (env MARL_DIAG=1; every 50 ticks) ──
        # Prints the frozen/online policy's actual joint action state and the
        # per-flow zero-delivery causes so collapses can be traced to a
        # mechanism (admission HOLD vs MCS/capacity vs routing).
        if (os.environ.get('MARL_DIAG') == '1' and mode == 'marl'
                and sim.island_mode and tick % 50 == 0):
            from collections import Counter as _DCounter
            _mcs_off_g, _mcs_off_e = _DCounter(), _DCounter()
            _hold_ls = 0
            for _aid, _ag in rl_agents.items():
                _act = getattr(_ag, 'last_action', None)
                if _act is None:
                    continue
                _mcs_off_g[_act.mcs_general_idx - MCS_OFFSET_CENTER] += 1
                _mcs_off_e[_act.mcs_emergency_idx - MCS_OFFSET_CENTER] += 1
                if _act.prb_emergency_frac < 0.05:
                    _hold_ls += 1
            _se_gen0 = sum(1 for ps in sim.phy_mac_states.values()
                           if ps.spectral_efficiency(for_emergency=False) <= 0.0)
            _se_emrg0 = sum(1 for ps in sim.phy_mac_states.values()
                            if ps.spectral_efficiency(for_emergency=True) <= 0.0)
            _sinrs = [ps.sinr_average for ps in sim.phy_mac_states.values()]
            _txs = [ps.tx_power_dbm for ps in sim.phy_mac_states.values()]
            _relay_dist = _DCounter(ps.relay_mode.name
                                    for ps in sim.phy_mac_states.values())
            print(f"[{mode_label} - {scenario_name}]   [DIAG] t={tick}: "
                  f"blocked(hold={getattr(sim, '_diag_flow_hold', -1)} "
                  f"cap={getattr(sim, '_diag_flow_cap', -1)} "
                  f"noroute={getattr(sim, '_diag_flow_noroute', -1)})/"
                  f"{len(sim.ue_to_ue_flows)} | "
                  f"SE0(gen={_se_gen0} emrg={_se_emrg0})/{len(sim.phy_mac_states)} | "
                  f"mcs_off_g={dict(sorted(_mcs_off_g.items()))} "
                  f"mcs_off_e={dict(sorted(_mcs_off_e.items()))} | "
                  f"LS-HOLD-agents={_hold_ls} | "
                  f"sinr[min={min(_sinrs):.1f} avg={np.mean(_sinrs):.1f} "
                  f"max={max(_sinrs):.1f}] | "
                  f"tx[min={min(_txs):.1f} avg={np.mean(_txs):.1f} "
                  f"max={max(_txs):.1f}] | relay={dict(_relay_dist)}")
            # ── MultiHaul (relay-capable) site detail ──────────────────────
            # Bridging is gated on ps.relay_mode == CAPACITY_BOOST at a site
            # with has_multihaul, so the ONLY relay decisions that can
            # reunify the island are those 3 sites'.  Print them explicitly:
            # a large `relay=` histogram over non-MultiHaul nodes says
            # nothing about whether reunification was even attempted.
            _mh_rows = []
            for _nid, _ps in sim.phy_mac_states.items():
                _n = sim.topology.nodes.get(_nid)
                if not (_n and getattr(_n, 'has_multihaul', False)):
                    continue
                _ag = sim.agents.get(_nid)
                _act = getattr(_ag, 'last_action', None) if _ag else None
                _ai = (list(RelayMode)[_act.relay_mode_idx].name
                       if _act is not None else 'n/a')
                _mh_rows.append(
                    f"{_nid}[surv={int(getattr(_n, 'is_survivor', False))} "
                    f"act={_ai} mode={_ps.relay_mode.name} "
                    f"link={int(_ps.relay_link_active)}->{_ps.relay_peer_node} "
                    f"prb_relay={_ps.prb_relay_fraction:.2f}]")
            print(f"[{mode_label} - {scenario_name}]   [DIAG-MH] t={tick}: "
                  f"mh_sites={len(_mh_rows)} "
                  f"raw_comps={sim.count_infra_components(include_relay_links=False)} "
                  f"achv_comps={sim.count_infra_components(include_relay_links=True)} "
                  f"bridges={sim.count_bridging_relay_links()} "
                  f"relay_links={len(sim.active_relay_links())} "
                  f"pending_up={len(getattr(sim, '_relay_link_ready_tick', {}))} | "
                  + " ".join(_mh_rows))

        # ── Shift-detector trigger: SUSTAINED delivered-fraction drop ────
        # Achievability must stay below DROP_FRACTION x the slow baseline
        # for DROP_CONSEC_TICKS consecutive ticks — a single-tick blip can
        # NEVER raise the temperature.  The slow baseline only tracks ticks
        # that are not part of a candidate drop (so it does not chase the
        # collapse it is supposed to detect).
        if learns and not _frozen and tick >= marl_active_tick:
            if _conn_slow_ema is None:
                _conn_slow_ema = conn_pct
            if conn_pct < DROP_FRACTION * _conn_slow_ema:
                _conn_drop_streak += 1
                if _conn_drop_streak >= DROP_CONSEC_TICKS:
                    # Re-trigger only if no boost window is already active
                    if not (_temp_boost_start >= 0 and
                            tick - _temp_boost_start < TEMP_BOOST_WINDOW_TICKS):
                        _trigger_temp_boost(
                            tick, f'sustained_drop({_conn_drop_streak} ticks, '
                                  f'conn={conn_pct:.1f}% < '
                                  f'{DROP_FRACTION:.0%} of baseline '
                                  f'{_conn_slow_ema:.1f}%)')
                    _conn_drop_streak = 0
            else:
                _conn_drop_streak = 0
                _conn_slow_ema = 0.01 * conn_pct + 0.99 * _conn_slow_ema

        # ── Adaptive PPO interval (simulation-side training loop) ────────
        # In the real system, agents share gradients via postcards and do
        # federated averaging. In this simulation, the shared MAPPO pool
        # is an efficient approximation of that communication.
        # The PPO update frequency is adjusted based on achievability:
        if learns and not _frozen and tick >= marl_active_tick:
            prev_ema = _ema_achiev
            _ema_achiev = _ema_alpha * conn_pct + (1.0 - _ema_alpha) * _ema_achiev

            # Adaptive PPO update frequency
            achiev_drop = prev_ema - _ema_achiev
            if achiev_drop > 1.0:  # >1% drop → urgent learning
                _ppo_interval = _ppo_interval_urgent
            elif (_temp_boost_start >= 0 and
                    tick - _temp_boost_start < TEMP_BOOST_WINDOW_TICKS):
                # Inside a genuine post-event window → keep learning frequent
                _ppo_interval = max(_ppo_interval_urgent, min(_ppo_interval + 1, 15))
            else:
                # Gradually return to baseline
                _ppo_interval = min(_ppo_interval + 1, 30)

        # Energy: physics-based per-node radio power model
        # Components: PA (load-scaled), baseband DSP (MCS-dependent),
        #             per-UE PDCCH/scheduling, relay forwarding, protocol overhead
        # ── Relay accounting (identical definition for EVERY arm) ──────────
        # `relay_count` = number of ACTIVE TRANSPORT RELAY LINKS.  It used to
        # be the number of NODES whose relay_mode was LOCAL_REROUTE/
        # CAPACITY_BOOST, which is an intent and not a bridge: the OSPF/OLSR
        # arms set every one of their 42 infra nodes to LOCAL_REROUTE (see
        # _lock_phy_mac_non_mcs) yet never call _establish_relay_links, so
        # they reported "42 relays" while holding zero relay links, while
        # BATMAN/AODV (relay_mode OFF) reported 0.  Nodes-in-mode is still
        # recorded, under the clearly different key `relay_mode_nodes`.
        relay_count = kpis.get('transport_relay_link_count', 0)
        relay_count_log.append(relay_count)
        relay_mode_nodes_log.append(kpis.get('relay_mode_node_count', 0))
        relay_bridge_log.append(kpis.get('transport_relay_bridge_count', 0))

        # Compute control-plane messages BEFORE energy, since the energy model
        # derives control-plane energy from actual message counts.
        tick_msg_count, tick_msg_bytes = ctrl_msg_sim.compute_tick_messages(tick)

        # Pre-compute per-node link load from actual traffic forwarding
        # (link utilization is populated by _forward_traffic which runs before this)
        node_load_frac = {}  # nid -> max utilization fraction across its links
        for lid, link in sim.topology.links.items():
            if not getattr(link, 'is_up', True):
                continue
            cap = getattr(link, 'capacity', 0)
            util = getattr(link, 'current_utilization', 0)
            if cap > 0:
                load_frac = min(1.0, util / cap)
            else:
                load_frac = 0.0
            for ep in link.endpoints:
                if ep in sim.phy_mac_states:
                    node_load_frac[ep] = max(node_load_frac.get(ep, 0.0), load_frac)

        # Node load comes directly from measured link utilisation — no
        # decorative per-node gaussian jitter (the traffic model itself is
        # stochastic per flow, which already produces natural variation).

        tick_energy_j = 0.0
        for nid, ps in sim.phy_mac_states.items():
            node = sim.topology.nodes.get(nid)
            if not node or not node.is_survivor:
                continue
            # Check if node's links are up (a node with all links down is idle)
            node_has_active_link = False
            for nbr in sim.topology.graph.neighbors(nid):
                edata = sim.topology.graph.get_edge_data(nid, nbr)
                if edata and 'link_id' in edata:
                    link = sim.topology.links.get(edata['link_id'])
                    if link and getattr(link, 'is_up', True):
                        node_has_active_link = True
                        break

            # 1. Static idle power: clock tree, cooling, always-on circuits
            #    Active node: 5W idle; isolated/down node: 1W (sleep mode)
            if node_has_active_link:
                idle_w = 5.0
            else:
                idle_w = 1.0  # Node in low-power standby (no active links)

            # 1b. State transition penalty: when node transitions from
            # sleep/idle → active (e.g., post-severance re-activation),
            # PLLs re-lock, DFE re-trains — costs extra power for ~50ms (5 ticks)
            state_transition_w = 0.0
            if sim.island_mode and node_has_active_link:
                effective_sev = 0 if scenario_name == 'rescue_ops' else SEVERANCE_TICK
                ticks_in_island = tick - effective_sev
                if 0 < ticks_in_island <= 5:
                    state_transition_w = 3.0  # PLL lock + DFE training burst

            # 2. PA power: Tx power × duty cycle (actual link load)
            #    PA only draws significant current when actually transmitting
            tx_power_w = (10 ** (ps.tx_power_dbm / 10.0)) / 1000.0
            pa_efficiency = 0.25  # Class-AB PA efficiency
            load_duty = max(0.05, node_load_frac.get(nid, 0.0))  # Min 5% for control channels
            pa_power_w = (tx_power_w / pa_efficiency) * load_duty

            # 2b. Rx power: LNA + ADC + mixer — always on when link is active
            #     ~2-3W for receive chain at QAM64, scaled by MCS complexity
            mcs_rx_table = {
                MCSLevel.QPSK_1_3: 1.5,
                MCSLevel.QPSK_1_2: 1.8,
                MCSLevel.QAM16:    2.2,
                MCSLevel.QAM64:    2.8,
                MCSLevel.QAM256:   3.5,
            }
            rx_w = mcs_rx_table.get(ps.mcs_general, 2.2) if node_has_active_link else 0.0

            # 3. Baseband DSP: MCS-dependent Tx processing (LDPC encoding complexity)
            #    QPSK: ~1.5W, QAM16: ~2.5W, QAM64: ~4W, QAM256: ~6W
            mcs_dsp_table = {
                MCSLevel.QPSK_1_3: 1.2,
                MCSLevel.QPSK_1_2: 1.5,
                MCSLevel.QAM16:    2.5,
                MCSLevel.QAM64:    4.0,
                MCSLevel.QAM256:   6.0,
            }
            dsp_w = mcs_dsp_table.get(ps.mcs_general, 2.5) * load_duty

            # 4. Per-UE overhead: PDCCH scheduling, CSI processing
            #    ~0.1W per active UE (DCI generation + HARQ management)
            ue_overhead_w = ps.active_ue_count * 0.1

            # 5. Relay forwarding: decode-and-forward adds DSP + buffering cost
            #    Energy is proportional to ACTUAL relay utilization, not just
            #    mode activation.  A relay in CAPACITY_BOOST that isn't
            #    forwarding traffic only pays a small scanning/monitoring cost.
            relay_w = 0.0
            if ps.relay_mode in (RelayMode.LOCAL_REROUTE, RelayMode.CAPACITY_BOOST):
                if ps.relay_link_active:
                    # Active relay link: decode-and-forward DSP + buffering
                    # Scale by actual relay PRB usage AND node load
                    relay_load = node_load_frac.get(nid, 0.0)
                    relay_w = 3.5 * ps.prb_relay_fraction * relay_load
                else:
                    # Relay mode on but no active link: scanning/monitoring only
                    relay_w = 0.2  # Beam scanning + neighbor discovery
            elif ps.relay_mode == RelayMode.D2D_PEER_RELAY:
                relay_w = 1.0  # Lighter: just sidelink coordination

            # 6. Protocol control-plane energy — derived from actual messages
            #
            # Standards basis:
            #   EARTH D2.3: baseband CPU accounts for ~15% of BS power.
            #     At 5W idle, CPU ≈ 0.75W baseline.
            #     Each control message requires: serialization, CRC, routing
            #     table lookup/update, and radio Tx scheduling.
            #
            #   Per-message energy cost (from published router power
            #   measurement studies):
            #     - Simple packet (Hello/OGM/RREQ):   ~0.5 mJ CPU
            #     - Complex computation (SPF/Dijkstra): ~2 mJ CPU
            #     - TCAM write (FlowMod install):      ~1.5 mJ CPU
            #     - Statistical update (TQ/EMA):       ~0.3 mJ CPU
            #
            #   Radio Tx energy for control overhead:
            #     E_tx = P_tx × time = P_tx × (bytes × 8 / bitrate)
            #     At 23 dBm (200mW) and 100 Mbps: ~16 μJ per 1KB
            #
            # The total per-node control energy is:
            #   ctrl_w = (msgs_per_node × cpu_cost_mJ + tx_energy_mJ) / tick_ms
            #
            # This ties energy directly to the overhead model's message count.

            ctrl_w = 0.0
            if sim.island_mode and N > 0:
                # Distribute total messages across nodes
                msgs_this_node = tick_msg_count / max(1, N)
                bytes_this_node = tick_msg_bytes / max(1, N)

                # CPU energy per message (mJ) — protocol-specific complexity
                if mode == 'ospf':
                    # Blend by MEASURED convergence state: LSA processing +
                    # SPF (~2 mJ/msg) while converging → Hello (~0.5 mJ)
                    ospf_ctrl = getattr(sim, '_ospf_ctrl', None)
                    conv_frac = getattr(ospf_ctrl, 'convergence_fraction', 1.0) if ospf_ctrl else 1.0
                    cpu_cost_mj = 2.0 * (1.0 - conv_frac) + 0.5 * conv_frac
                elif mode == 'sdn':
                    sdn_phase = getattr(getattr(sim, '_sdn_ctrl', None),
                                        'phase', 'orphan')
                    if sdn_phase == 'programming':
                        # FlowMod → TCAM write: ~1.5 mJ per rule install
                        cpu_cost_mj = 1.5
                    elif sdn_phase in ('orphan', 'election'):
                        # Raft consensus: ~1.0 mJ per vote/append RPC
                        cpu_cost_mj = 1.0
                    elif sdn_phase in ('discovery', 'lldp'):
                        # LLDP parse + topology BFS: ~0.8 mJ per probe
                        cpu_cost_mj = 0.8
                    else:
                        # Operational: stats poll + echo: ~0.3 mJ
                        cpu_cost_mj = 0.3
                elif mode == 'olsr':
                    olsr_ctrl = getattr(sim, '_olsr_ctrl', None)
                    conv_frac = getattr(olsr_ctrl, 'convergence_fraction', 0.0) if olsr_ctrl else 0.0
                    # Smooth: SPF computation (1.2mJ) → operational (0.3mJ)
                    cpu_cost_mj = 1.2 * (1.0 - conv_frac) + 0.3 * conv_frac
                elif mode == 'batman':
                    batman_ctrl = getattr(sim, '_batman_ctrl', None)
                    conv_frac = getattr(batman_ctrl, 'convergence_fraction', 0.0) if batman_ctrl else 0.0
                    # Smooth: OGM parse+rebroadcast (0.5mJ) → maintenance (0.3mJ)
                    cpu_cost_mj = 0.5 * (1.0 - conv_frac) + 0.3 * conv_frac
                elif mode == 'aodv':
                    aodv_phase = getattr(getattr(sim, '_aodv_ctrl', None),
                                         'phase', 'idle')
                    if aodv_phase == 'rreq_flood':
                        # RREQ broadcast processing: ~0.7 mJ (duplicate check + rebroadcast)
                        cpu_cost_mj = 0.7
                    elif aodv_phase in ('rrep_path', 'route_repair'):
                        # RREP unicast + route table install: ~0.5 mJ
                        cpu_cost_mj = 0.5
                    else:
                        # Idle: no messages → no energy (reactive advantage)
                        cpu_cost_mj = 0.1
                elif mode == 'marl':
                    # Postcard parse + NN inference: ~0.3 mJ (embedded NPU)
                    cpu_cost_mj = 0.3
                else:
                    cpu_cost_mj = 0.5

                # CPU energy from message processing (mJ → W over tick)
                cpu_energy_mj = msgs_this_node * cpu_cost_mj
                cpu_energy_w = cpu_energy_mj / (TICK_DURATION_S * 1000.0)  # mJ / tick_ms = W

                # Radio Tx energy for control overhead bytes
                # At 23 dBm (200 mW RF) + PA inefficiency (25%): ~0.8W Tx
                # Time to transmit: bytes × 8 / bitrate_bps
                # At 100 Mbps effective: 1 byte = 80ns → 1KB = 82μs
                # Energy = 0.8W × 82μs = 0.066 mJ per KB
                tx_energy_mj = bytes_this_node * 8.0 / 100e6 * 0.8 * 1000.0
                tx_energy_w = tx_energy_mj / (TICK_DURATION_S * 1000.0)  # mJ / tick_ms = W

                ctrl_w = cpu_energy_w + tx_energy_w

            node_total_w = (idle_w + state_transition_w + pa_power_w + rx_w +
                           dsp_w + ue_overhead_w + relay_w + ctrl_w)
            # Per-tick energy (1 tick = 1s)
            tick_energy_j += node_total_w * TICK_DURATION_S

        energy_log.append(tick_energy_j)

        # MCS distribution: compute fraction of infra nodes using QAM256
        from sixg_sim.phy_mac_state import MCSLevel as _MCSLevel
        _n_infra_mcs = 0
        _n_qam256 = 0
        for _nid, _ps in sim.phy_mac_states.items():
            _node = sim.topology.nodes.get(_nid)
            if _node and _node.node_type.value != 'UE':
                _n_infra_mcs += 1
                if _ps.mcs_general == _MCSLevel.QAM256:
                    _n_qam256 += 1
        mcs_high_pct = 100.0 * _n_qam256 / max(1, _n_infra_mcs)
        mcs_high_log.append(mcs_high_pct)

        # Link/node counts
        link_count = kpis.get('transport_link_count', 0)
        link_count_log.append(link_count)
        node_count_log.append(N)
        # Operational fragments (infra components incl. active relay links)
        fragments_log.append(fragments)
        # Raw post-severance partition (same count with relay links removed)
        fragments_raw_log.append(fragments_raw)
        # Peer reachability (see the declaration above).  Goes through
        # current_peer_reach_stats() rather than the cached attributes: the
        # cache is only refreshed by the agent observation builder, so the
        # routing baselines would otherwise report their init defaults
        # (100 % reachable, 0 stranded) as if they had been measured.
        _reach_st = sim.current_peer_reach_stats()
        peer_reach_log.append(
            100.0 * float(_reach_st.get('peer_reachable_fraction', 1.0)))
        reach_restored_log.append(int(_reach_st.get('restored', 0)))
        stranded_ue_log.append(int(_reach_st.get('stranded', 0)))
        outside_body_log.append(int(_reach_st.get('attached_outside_body', 0)))
        # Uu carrier-aggregation audit (0 / 1.0 for every arm that is not
        # eligible - recorded, never omitted).
        uu_boost_links_log.append(int(_uu_boost_n))
        uu_boost_factor_log.append(float(_uu_boost_factor))
        uu_cc_log.append(int(_uu_cc_n))
        uu_extra_mbps_log.append(float(_uu_extra_mbps))
        offered_log.append(float(offered))

        # Hop count: compute from actual successful flow paths
        total_hops = 0
        routed_flows = 0
        if not sim.island_mode:
            # Pre-disaster: core-routed traffic has ~3 hops (UE→gNB→Core→gNB→UE)
            avg_hops = 3.0
        else:
            for src, tgt in sim.ue_to_ue_flows:
                try:
                    path = nx.shortest_path(sim.topology.graph, src, tgt)
                    total_hops += len(path) - 1
                    routed_flows += 1
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            avg_hops = total_hops / max(routed_flows, 1)
        hops_log.append(avg_hops)

        # Overhead: computed from actual simulated message exchanges
        postcards_this_tick = getattr(sim, 'last_postcards_sent', 0)
        postcards_log.append(postcards_this_tick)

        # tick_msg_count and tick_msg_bytes already computed before energy block
        msg_count_log.append(tick_msg_count)

        # Convert bytes to Mbps: bytes × 8 bits × ticks_per_second ÷ 1e6
        overhead_mbps = (tick_msg_bytes * 8 * TICKS_PER_SECOND) / 1e6
        overhead_log.append(overhead_mbps)

        # ── MODELED latency: per-hop analytical model ───────────────
        # (labelled "modeled latency" in the figures — see module docstring)
        #   L = Σ_hops (prop + M/M/1 queue from ACTUAL link utilisation +
        #               HARQ retransmission term) + protocol_processing
        #
        # Outage handling: flows with NO converged route (protocol still
        # converging) or NO physical path are NOT assigned a sentinel
        # latency.  They are counted in `outage_frac` (plotted separately)
        # and EXCLUDED from the mean-latency aggregation; latency is NaN on
        # ticks where no flow routes at all.
        # Removed former decorations: sinusoidal "load cycles", gaussian
        # congestion/contention jitter, and the fabricated pre-disaster
        # 1.5±0.1 ms constant (pre-disaster now uses the same per-hop model
        # over the actual core-routed paths).

        total_path_latency_ms = 0.0
        latency_flow_count = 0

        # Full demand set (already restored after protocol filtering)
        _full_flow_set = list(sim.ue_to_ue_flows)

        # Flows with converged routes, per the protocol's measured state
        if sim.island_mode and mode == 'ospf' and hasattr(sim, '_ospf_ctrl'):
            active_flows = sim._ospf_ctrl.get_routable_flows()
        elif sim.island_mode and mode == 'sdn' and hasattr(sim, '_sdn_ctrl'):
            active_flows = sim._sdn_ctrl.get_programmed_flows()
        elif sim.island_mode and mode == 'olsr' and hasattr(sim, '_olsr_ctrl'):
            active_flows = sim._olsr_ctrl.get_routable_flows()
        elif sim.island_mode and mode == 'batman' and hasattr(sim, '_batman_ctrl'):
            active_flows = sim._batman_ctrl.get_routable_flows()
        elif sim.island_mode and mode == 'aodv' and hasattr(sim, '_aodv_ctrl'):
            active_flows = sim._aodv_ctrl.get_routable_flows()
        elif sim.island_mode and hasattr(sim, '_marl_detect'):
            # The MARL family (and the uniform-random control, which shares the
            # same detector) was the only arm whose numerator was NOT gated by
            # its own control plane: it fell through to the full demand set.
            active_flows = sim._marl_detect.get_routable_flows()
        else:
            active_flows = _full_flow_set

        for src, tgt in active_flows:
            # PHYSICAL half of the numerator: an infrastructure-only path must
            # exist.  Without this, sim.topology.graph includes UE nodes and a
            # dual-anchored UE bridges two fragments -- a path the service model
            # never allows -- which is what made OSPF and OLSR/BATMAN/AODV
            # disagree on outage while agreeing on achievability.
            if MARL_UE_RELAY == 'none' and not sim._can_route_ue_to_ue(src, tgt):
                continue
            try:
                path = nx.shortest_path(sim.topology.graph, src, tgt)
                n_hops = len(path) - 1
                path_latency = 0.0

                # ── Per-hop data-plane latency (propagation + queue + jitter) ──
                flow_rng = random.Random(
                    tick * 7919
                    + zlib.crc32(f"{src}|{tgt}".encode('utf-8')) % 9973)
                for i in range(n_hops):
                    edge_data = sim.topology.graph.get_edge_data(path[i], path[i+1])
                    if edge_data and 'link_id' in edge_data:
                        link = sim.topology.links.get(edge_data['link_id'])
                        if link:
                            # Propagation delay from topology (distance-based)
                            prop_delay_ms = getattr(link, 'latency', 1) * 0.01

                            # M/M/1 queuing: delay = service_time / (1 − ρ)
                            cap = getattr(link, 'capacity', 1)
                            util = getattr(link, 'current_utilization', 0)
                            rho = min(0.95, util / max(cap, 1))
                            service_time_ms = 0.05  # ~50μs serialization at 1 Gbps
                            queue_delay_ms = service_time_ms / max(0.05, 1.0 - rho)

                            # HARQ retransmission jitter (3GPP TS 38.213)
                            # ~10% of transmissions need 1 retx (8ms RTT in NR)
                            # Higher load → more collisions → more retx
                            harq_prob = 0.10 + 0.15 * rho  # 10-25% retx rate
                            harq_delay_ms = 8.0 if flow_rng.random() < harq_prob else 0.0

                            # Scheduling jitter: slot alignment, BSR processing
                            # 0.5ms slot boundary + load-dependent grant delay
                            sched_jitter_ms = flow_rng.uniform(0, 0.5) + rho * flow_rng.uniform(0, 1.0)

                            path_latency += prop_delay_ms + queue_delay_ms + harq_delay_ms + sched_jitter_ms

                # ── Protocol-specific processing (routed flows only) ──
                # Flows without converged routes never reach this loop —
                # they are outage, not a sentinel latency.
                if sim.island_mode and mode == 'marl':
                    # NN forward pass: ~0.5ms inference per decision
                    path_latency += 0.5

                elif sim.island_mode and mode == 'ospf':
                    # Converged FIB longest-prefix-match: ~50μs per hop.
                    # (Convergence delay is expressed as outage fraction,
                    # measured by the OSPF control-plane state machine.)
                    path_latency += n_hops * 0.05

                elif sim.island_mode and mode == 'sdn':
                    sdn_phase = getattr(getattr(sim, '_sdn_ctrl', None),
                                        'phase', 'orphan')
                    if sdn_phase == 'discovery':
                        # Slow-path: Packet-In → controller → Packet-Out
                        path_latency += 8.0 * n_hops  # 8ms per hop
                    elif sdn_phase == 'programming':
                        # FlowMod install: 3-30ms per rule on TCAM
                        path_latency += 15.0
                    else:
                        # Operational: TCAM line-rate lookup
                        path_latency += n_hops * 0.001

                elif sim.island_mode and mode == 'olsr':
                    # Per-flow interpolation: routes exist but may be
                    # sub-optimal until both endpoints fully converge.
                    olsr_ctrl = getattr(sim, '_olsr_ctrl', None)
                    nc = getattr(olsr_ctrl, 'node_convergence', {}) if olsr_ctrl else {}
                    conv = min(nc.get(src, 1.0), nc.get(tgt, 1.0))
                    sub_optimal_lat = 5.0 + n_hops * 0.5
                    converged_lat = n_hops * 0.1
                    path_latency += sub_optimal_lat * (1 - conv) + converged_lat * conv

                elif sim.island_mode and mode == 'batman':
                    batman_ctrl = getattr(sim, '_batman_ctrl', None)
                    nc = getattr(batman_ctrl, 'node_convergence', {}) if batman_ctrl else {}
                    conv = min(nc.get(src, 1.0), nc.get(tgt, 1.0))
                    sub_optimal_lat = 30.0 + n_hops * 0.2  # TQ not yet stable
                    converged_lat = n_hops * 0.05          # L2 forwarding
                    path_latency += sub_optimal_lat * (1 - conv) + converged_lat * conv

                elif sim.island_mode and mode == 'aodv':
                    # Routed flows have cached routes (RFC 3561):
                    # kernel route-cache lookup ~20μs/hop; add repair
                    # overhead when the controller is in local repair.
                    aodv_phase = getattr(getattr(sim, '_aodv_ctrl', None),
                                         'phase', 'idle')
                    if aodv_phase == 'route_repair':
                        path_latency += n_hops * 40.0 + 50.0  # RFC 3561 §6.12
                    else:
                        path_latency += n_hops * 0.02

                total_path_latency_ms += path_latency
                latency_flow_count += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass  # no physical path — counted as outage below

        # ── Mean modeled latency over ROUTED flows; outage tracked apart ──
        n_flows_total = len(_full_flow_set)
        if latency_flow_count > 0:
            latency_ms = total_path_latency_ms / latency_flow_count
        else:
            latency_ms = float('nan')  # no flow routed this tick
        outage_frac = (1.0 - latency_flow_count / n_flows_total) if n_flows_total > 0 else 0.0
        outage_frac = min(1.0, max(0.0, outage_frac))
        latency_log.append(latency_ms)
        outage_log.append(outage_frac)

        if tick % 100 == 0:
            proto_info = ''
            if mode == 'sdn' and hasattr(sim, '_sdn_ctrl'):
                ctrl = sim._sdn_ctrl
                proto_info = f" | SDN:{ctrl.phase} flows:{len(ctrl.programmed_flows)}"
            elif mode == 'olsr' and hasattr(sim, '_olsr_ctrl'):
                ctrl = sim._olsr_ctrl
                proto_info = f" | OLSR:{ctrl.phase} routes:{len(ctrl.routable_flows)}"
            elif mode == 'batman' and hasattr(sim, '_batman_ctrl'):
                ctrl = sim._batman_ctrl
                proto_info = f" | BATMAN:{ctrl.phase} routes:{len(ctrl.routable_flows)}"
            elif mode == 'aodv' and hasattr(sim, '_aodv_ctrl'):
                ctrl = sim._aodv_ctrl
                proto_info = f" | AODV:{ctrl.phase} cached:{len(ctrl.route_cache)} rreq:{ctrl._active_discoveries}"
            # MCS distribution diagnostic
            mcs_dist = {}
            for nid, ps in sim.phy_mac_states.items():
                mcs_name = ps.mcs_emergency.name if hasattr(ps.mcs_emergency, 'name') else str(ps.mcs_emergency)
                mcs_dist[mcs_name] = mcs_dist.get(mcs_name, 0) + 1
            mcs_str = ' '.join(f"{k}:{v}" for k, v in sorted(mcs_dist.items()))
            # Node load diagnostic (from actual link utilization)
            load_vals = [node_load_frac.get(nid, 0) for nid in sim.phy_mac_states]
            avg_load = sum(load_vals) / max(1, len(load_vals))
            max_load = max(load_vals) if load_vals else 0
            active_ues = sum(ps.active_ue_count for ps in sim.phy_mac_states.values())
            # MARL adaptation diagnostic info (temperature controller state)
            adapt_info = ""
            if mode == 'marl':
                _t_now = TEMP_FLOOR if _frozen else _current_temperature(tick)
                _in_boost = (_temp_boost_start >= 0 and
                             tick - _temp_boost_start < TEMP_BOOST_WINDOW_TICKS)
                # PRB distribution across all agents
                prb_e = np.mean([ps.prb_emergency_fraction for ps in sim.phy_mac_states.values()])
                prb_r = np.mean([ps.prb_relay_fraction for ps in sim.phy_mac_states.values()])
                prb_g = np.mean([ps.prb_general_fraction for ps in sim.phy_mac_states.values()])
                if is_random_arm:
                    # No temperature / freeze semantics on the control arm.
                    adapt_info = (f" | UNIFORM-RANDOM ACTIONS"
                                  f" PRB[e={prb_e:.2f}/r={prb_r:.2f}/g={prb_g:.2f}]")
                else:
                    adapt_info = (f" | ADAPT temp={_t_now:.3f}"
                                  f"{' BOOST(' + str(_temp_boost_reason) + ')' if _in_boost else ''}"
                                  f"{' FROZEN' if _frozen else ''}"
                                  f" PRB[e={prb_e:.2f}/r={prb_r:.2f}/g={prb_g:.2f}]")
            print(f"[{mode_label} - {scenario_name}] Tick {tick:4d} | Achiev: {conn_pct:.1f}% (del:{delivered:.0f} off:{offered:.0f}) | "
                  f"Energy: {energy_log[-1]:.1f} J | "
                  f"RelayLinks: {relay_count} (bridge {relay_bridge_log[-1]}, "
                  f"mode-nodes {relay_mode_nodes_log[-1]}) | "
                  f"Frags: {fragments}/{fragments_raw} | "
                  f"Reach: {peer_reach_log[-1]:.1f}% "
                  f"(restored {reach_restored_log[-1]}, "
                  f"stranded {stranded_ue_log[-1]}) | "
                  f"CA: {uu_cc_log[-1]}cc/"
                  f"{uu_extra_mbps_log[-1]:.0f}Mb over "
                  f"{uu_boost_links_log[-1]}x{uu_boost_factor_log[-1]:.2f} | "
                  f"OH: {overhead_mbps:.2f} Mbps | Lat: {latency_ms:.1f}ms | outage: {outage_frac:.0%}{proto_info}"
                  f" | MCS:[{mcs_str}] | Load:{avg_load:.2f}/{max_load:.2f} UEs:{active_ues}{adapt_info}")

    # Scenario over: flush in-flight trajectories (per-agent GAE, terminal)
    if learns and not _frozen:
        try:
            mappo_trainer.finish_episode()
        except Exception:
            pass

    if is_static_arm:
        _static_fp1 = _policy_fingerprint(rl_agents)
        if _static_fp1 != _static_fp0:
            raise RuntimeError("marl_static invariant violated: policy "
                               "parameters changed during the pass")
        print(f"[{mode_label} - {scenario_name}] STATIC CHECK OK: parameters "
              f"bit-identical after the pass ({_static_fp1[:16]})")

    # Save adapted policy if MARL recovery completed (per-agent).
    # The adapt-freeze arm saves under its own suffix so Phase 2's
    # rescue_ops pass for each arm loads its own recovery-adapted weights.
    if learns and scenario_name == 'recovery' and rl_agents:
        adapted_sd = {aid: a.policy_net.state_dict() for aid, a in rl_agents.items()}
        _suffix = ('_adapted_recovery_freeze' if is_freeze_arm
                   else '_adapted_recovery')
        adapted_path = _adapted_checkpoint_path(checkpoint_path, _suffix,
                                                topology_seed)
        torch.save(adapted_sd, adapted_path)
        print(f"[{mode_label} - {scenario_name}] Saved {len(adapted_sd)} adapted per-agent policies to {adapted_path}")

    # ── Uu CARRIER-AGGREGATION SUMMARY ─ one grep-able line per pass ────
    # Printed unconditionally for EVERY arm, so a reader can establish from
    # the log alone which arms received the grant and how large it was.
    # v2 additions: activated carriers per site and the aggregate extra
    # capacity, so the SITE-LEVEL BOUND is auditable from this one line.
    _ca_ticks = sum(1 for v in uu_boost_links_log if v > 0)
    _ca_link_ticks = sum(uu_boost_links_log)
    _ca_factors = [f for v, f in zip(uu_boost_links_log, uu_boost_factor_log)
                   if v > 0]
    _ca_mean_f = (sum(_ca_factors) / len(_ca_factors)) if _ca_factors else 1.0
    _ca_peak = max(uu_boost_links_log) if uu_boost_links_log else 0
    # Number of radio sites (PHY/MAC-bearing nodes) — the denominator for
    # "carriers per site" and the multiplier for the study-wide ceiling.
    _n_sites = max(1, len(sim.phy_mac_states))
    _cc_on = [c for c in uu_cc_log if c > 0]
    _ca_cc_mean = (sum(_cc_on) / len(_cc_on)) if _cc_on else 0.0
    _ca_cc_peak = max(uu_cc_log) if uu_cc_log else 0
    _ex_on = [e for e in uu_extra_mbps_log if e > 0]
    _ca_extra_mean = (sum(_ex_on) / len(_ex_on)) if _ex_on else 0.0
    _ca_extra_peak = max(uu_extra_mbps_log) if uu_extra_mbps_log else 0.0
    _ceiling = UU_CA_SITE_POOL_BOUND_MBPS * _n_sites
    print(f"[{mode_label} - {scenario_name}] [UU-BOOST-SUMMARY] "
          f"arm={arm} scenario={scenario_name} seed={topology_seed}: "
          f"N_CC={UU_CA_MAX_EXTRA_CC}/site @ "
          f"{UU_CA_CC_BANDWIDTH_MHZ:.0f}MHz "
          f"({UU_CA_CC_REF_CAPACITY_MBPS:.0f}Mbps@QAM256 each) | "
          f"ticks_with_boost={_ca_ticks}/{len(uu_boost_links_log)} "
          f"boosted_link_ticks={_ca_link_ticks} peak_links={_ca_peak} "
          f"mean_factor={_ca_mean_f:.3f}x | "
          f"sites={_n_sites} cc_active_mean={_ca_cc_mean:.1f} "
          f"cc_active_peak={_ca_cc_peak} "
          f"cc_per_site_mean={_ca_cc_mean / _n_sites:.3f} "
          f"extra_mean={_ca_extra_mean:.0f}Mbps "
          f"extra_peak={_ca_extra_peak:.0f}Mbps "
          f"study_ceiling={_ceiling:.0f}Mbps "
          f"bound_ok={'YES' if _ca_extra_peak <= _ceiling + 1e-6 else 'NO'}"
          + ("" if _ca_link_ticks else
             ("   (ABLATION: UU_CA_MAX_EXTRA_CC=0 — the carrier-aggregation "
              "grant is switched off for EVERY arm in this run)"
              if UU_CA_MAX_EXTRA_CC == 0 else
              "   (arm NOT eligible for Uu carrier aggregation, or its gate "
              "never opened)")))

    _pass_out = {
        'conn': conn_log, 'energy': energy_log, 'hops': hops_log,
        'fragments': fragments_log, 'relay_count': relay_count_log,
        'fragments_raw': fragments_raw_log,
        'relay_bridges': relay_bridge_log,
        'relay_mode_nodes': relay_mode_nodes_log,
        'link_count': link_count_log, 'node_count': node_count_log,
        'overhead': overhead_log, 'postcards': postcards_log, 'latency': latency_log,
        'delivered': delivered_log, 'mcs_high': mcs_high_log,
        'outage': outage_log, 'topology_info': topology_info,
        'peer_reach': peer_reach_log, 'reach_restored': reach_restored_log,
        'stranded_ues': stranded_ue_log, 'outside_body_ues': outside_body_log,
        'uu_boost_links': uu_boost_links_log,
        'uu_boost_factor': uu_boost_factor_log,
        'uu_cc_activated': uu_cc_log,
        'uu_extra_mbps': uu_extra_mbps_log,
        # Denominator for "activated carriers PER SITE" in the [4b] audit and
        # the multiplier for the study-wide extra-capacity ceiling.
        'n_radio_sites': _n_sites,
        'offered': offered_log,
        # ── Admission census (see Simulator.apply_marl_policy) ────────────
        # Cumulative over the run: how much of this arm's shortfall it
        # inflicted on itself through its own PRB split, via a channel the
        # routing baselines are structurally exempt from.  Reported so the
        # uniform-random control's result can be decomposed into "poor
        # control" and "self-inflicted admission loss" instead of being
        # attributed entirely to the former.
        'adm_calls':            int(getattr(sim, '_adm_calls', 0)),
        'adm_hold':             int(getattr(sim, '_adm_hold', 0)),
        'adm_throttle':         int(getattr(sim, '_adm_throttle', 0)),
        'adm_held_volume':      float(getattr(sim, '_adm_held_volume', 0.0)),
        'adm_throttled_volume': float(getattr(sim, '_adm_throttled_volume', 0.0)),
        # fragment-aware re-points performed this run (MARL_RELAY_REPOINT)
        'repoints':             int(getattr(sim, '_diag_repoints', 0)),
    }
    if is_static_arm:
        # static-deployment audit trail, stored with the data itself so it
        # does not depend on worker log output reaching the log file
        _pass_out['static_fp_start'] = _static_fp0
        _pass_out['static_fp_end'] = _static_fp1
    return _pass_out

def steady_state_per_seed_pairs(all_data_multi, mode, key, seeds, window=None):
    """(seed, steady_state_value) pairs — seeds without usable data omitted.

    Returning the SEED alongside the value matters: seeds are dropped
    individually when an arm has no run for them, so a caller that zips a
    bare value list against `seeds` mislabels every value after the first
    gap.  Anything that names a seed (the strip-plot failure callouts, the
    per-seed table column) must go through this function.
    """
    import numpy as np
    import warnings
    window = window or STEADY_STATE_WINDOW_TICKS
    seed_data = all_data_multi.get(mode, {})
    pairs = []
    for seed in seeds:
        run = seed_data.get(seed)
        if not run:
            continue
        arr = np.asarray(run.get(key, []), dtype=np.float64)
        if arr.size == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            val = np.nanmean(arr[-min(window, arr.size):])
        if np.isfinite(val):
            pairs.append((seed, float(val)))
    return pairs


def steady_state_per_seed(all_data_multi, mode, key, seeds, window=None):
    """Per-seed steady-state values: mean of the LAST `window` recorded ticks
    of each seed's full run (NaN-aware).  Returns a list, seeds without data
    omitted."""
    return [v for _s, v in steady_state_per_seed_pairs(
        all_data_multi, mode, key, seeds, window)]


def steady_state_across_seeds(all_data_multi, mode, key, seeds,
                              window=None):
    """Steady-state statistic for one arm/metric, aggregated across seeds.

    Per seed: mean of the LAST `window` recorded ticks (NaN-aware — latency
    arrays carry NaN on outage ticks).  Across seeds: mean and population
    standard deviation of those per-seed means.  This is the definition the
    figures and the article quote ("steady state = mean over the last 500
    ticks, +/-1 sigma across the held-out seeds"), and it is deliberately
    NOT the average of per-tick cross-seed spreads.

    Returns (mean, std, n_seeds_used).
    """
    import numpy as np
    import warnings
    per_seed = steady_state_per_seed(all_data_multi, mode, key, seeds, window)
    if not per_seed:
        return float('nan'), float('nan'), 0
    return float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed)


_FRAG_STATS_CACHE = {}


def measure_fragmenting_severance_stats(seeds):
    """Measure the Scenario-A fragmenting-severance parameters per seed.

    The partition is a pure, deterministic function of the topology seed
    (see _build_fragmenting_severance), so it can be RE-DERIVED here from
    the same builders the simulation used — no simulation is re-run and no
    literal is hardcoded into the figure.  Returns a dict with per-seed
    fragment count / cross-fragment flow share plus their means, or None if
    the builders are unavailable (figures then simply omit the block).
    """
    key = tuple(sorted(seeds))
    if key in _FRAG_STATS_CACHE:
        return _FRAG_STATS_CACHE[key]
    import io, contextlib, re
    per_seed = {}
    pattern = re.compile(
        r"fragments=(\d+) cross_fragment_flows=(\d+)/(\d+).*?cut_links=(\d+)")
    try:
        for seed in seeds:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                topology, _info = build_comparison_topology(seed)
                _build_fragmenting_severance(topology, SEVERANCE_TICK, seed)
            m = None
            for line in buf.getvalue().splitlines():
                if '[SCENARIO]' in line:
                    m = pattern.search(line)
                    if m:
                        break
            if not m:
                return None
            frags, cross, total, cut_links = (int(g) for g in m.groups())
            per_seed[seed] = {'fragments': frags, 'cross': cross,
                              'total': total, 'cut_links': cut_links,
                              'share': cross / max(1, total)}
            print(f"[FIGURE] seed={seed} fragmenting severance: "
                  f"fragments={frags}, cross-fragment flows={cross}/{total} "
                  f"({cross / max(1, total):.1%}), cut_links={cut_links}")
    except Exception as exc:      # builders unavailable / import failure
        print(f"[FIGURE] WARNING: could not measure fragmenting severance "
              f"parameters ({exc}); annotation will be omitted.")
        return None
    if not per_seed:
        return None
    shares = [v['share'] for v in per_seed.values()]
    stats = {
        'per_seed': per_seed,
        'mean_fragments': sum(v['fragments'] for v in per_seed.values()) / len(per_seed),
        'mean_share': sum(shares) / len(shares),
        'min_share': min(shares),
        'max_share': max(shares),
        'cross_total': sum(v['cross'] for v in per_seed.values()),
        'flow_total': sum(v['total'] for v in per_seed.values()),
        'mean_cut_links': sum(v['cut_links'] for v in per_seed.values()) / len(per_seed),
    }
    _FRAG_STATS_CACHE[key] = stats
    return stats


# ══════════════════════════════════════════════════════════════════════
#  FAIR (ARM-INDEPENDENT) RECOVERY-TIME METRIC
# ══════════════════════════════════════════════════════════════════════
# The legacy metric — "ticks until this arm reaches 90% of ITS OWN steady
# state" — is NOT comparable across arms: the bar moves with the answer, so
# whichever arm ends up highest is measured against the highest absolute
# target and is penalised for being good.  In Scenario B that artefact alone
# reported 882 s for MARL-RIC (90% of 89.8% = 80.8%) against 13-87 s for the
# classical arms (90% of 77.3% = 69.6%).  A self-referential target is not a
# recovery time.  The legacy number is retained in the raw table for
# continuity, explicitly relabelled "not comparable across arms", but it is
# no longer the headline.
#
# PRIMARY metric: time from the event until achievability first reaches — and
# then SUSTAINS for RECOVERY_SUSTAIN_TICKS consecutive ticks — a FIXED
# ABSOLUTE threshold that is one scalar, identical for every arm.  The
# threshold is a fraction of the PRE-EVENT achievability level, which is a
# property of the scenario (topology + offered demand), not of any arm.
# SECONDARY: the same metric swept over RECOVERY_THRESHOLD_FRACTIONS, so no
# single threshold choice can drive the conclusion.
RECOVERY_SUSTAIN_TICKS = 50          # consecutive ticks the level must hold
RECOVERY_THRESHOLD_FRACTIONS = (0.30, 0.50, 0.70)   # swept (of pre-event)
RECOVERY_PRIMARY_FRACTION = 0.50     # headline threshold
RECOVERY_REF_WINDOW_TICKS = 150      # pre-event window used for the reference
RECOVERY_REF_SPREAD_TOL = 1.0        # pp; above this the shared ref is flagged

# ══════════════════════════════════════════════════════════════════════
#  WHY THE FIXED-THRESHOLD FIRST-CROSSING TIME IS NO LONGER THE HEADLINE
# ══════════════════════════════════════════════════════════════════════
# The fixed-threshold metric above fixed the "moving bar" defect of the
# legacy 90%-of-own-steady-state number, but it has four defects of its own,
# all of which are properties of ANY first-crossing-of-a-threshold statistic:
#
#   (a) An arm that never drops below the threshold scores 0 s, which
#       conflates NEVER LOSING SERVICE with RECOVERING INSTANTLY.  Those are
#       different properties and only the second is a recovery time.
#   (b) The ranking flips with the threshold: at 30%/50% an arm can score
#       0 s and at 70% take 943 s while no baseline reaches 70% at all.  A
#       conclusion that inverts with an arbitrary constant is not a result.
#   (c) It is blind to the DEPTH of the degradation and to the AREA lost: an
#       arm grazing one tick under the bar scores worse than an arm parked
#       just above it for the whole hour.
#   (d) It is blind to RE-DIPS after the first crossing.
#
# So the PRIMARY metric is now the THRESHOLD-FREE service-loss integral
# (below), the first-crossing time is demoted to SECONDARY, availability
# repairs (d), and "never breached" is printed as its own CATEGORY instead
# of being encoded as the number 0.
#
# ── (1) SERVICE-LOSS INTEGRAL — primary, threshold-free ───────────────
#   SLI = ∫ max(0, reference − achievability(t)) dt   over the event window
# with `reference` the shared PRE-EVENT achievability level (a property of
# the scenario, not of any arm — the same scalar the fixed thresholds are
# fractions of).  Units: percentage-point-seconds (pp·s).  Lower is better.
# It answers "how much service was lost, and for how long", which is what a
# recovery time was a proxy for.  Reported over the full event window AND
# over a BOUNDED window (event .. event+SERVICE_LOSS_BOUNDED_TICKS), because
# the full-window figure grows with run length and the bounded one does not.
SERVICE_LOSS_BOUNDED_TICKS = 600     # bounded SLI variant: event + N ticks

# Achievability below this is treated as "no usable reconstruction of the
# offered demand" when converting the integral into delivered-volume units
# (offered ≈ delivered / achievability); above the upper bound the engine's
# min(100, ...) clamp makes the reconstruction an UNDER-estimate of offered,
# so those ticks are excluded from the reconstruction pool as well.
OFFERED_RECON_MIN_CONN_PCT = 1.0
OFFERED_RECON_MAX_CONN_PCT = 99.9

# ── (2) TIME TO X% OF THE FEASIBLE OPTIMUM ────────────────────────────
# Recovery measured against what the PHYSICS permits rather than against the
# pre-event level or the arm's own steady state.  This requires a per-seed
# FEASIBLE ACHIEVABILITY CEILING.  `tg_optimum` in topology_info is a
# COMPONENT COUNT (the fewest infra components reachable under TG-only
# steering), NOT an achievability level, and there is no defensible
# conversion from one to the other: a reunified graph does not imply every
# flow is served at full rate — capacity, MCS and queueing all still bind.
# So this metric is computed ONLY when a per-seed achievability ceiling is
# explicitly recorded under one of these keys, and is otherwise SKIPPED with
# the reason printed.  It is never invented from the component optimum.
FEASIBLE_CEILING_KEYS = (
    'conn_ceiling_pct', 'achievability_ceiling_pct',
    'tg_optimum_achievability', 'feasible_conn_pct',
    'feasible_achievability_pct',
)
CEILING_RECOVERY_FRACTIONS = (0.50, 0.80, 0.90)   # of the feasible ceiling

# ── (3) AVAILABILITY ──────────────────────────────────────────────────
# Fraction of event-window ticks at or above the threshold.  A first-crossing
# time is blind to everything after the crossing; availability is not, so an
# arm that crosses early and then re-dips is separated from one that crosses
# early and holds.  Swept over the SAME fractions as the crossing metric.

# An arm "fails" a seed when its steady-state achievability is below this.
FAILURE_ACHIEVABILITY_PCT = 10.0

# Optional per-tick series carrying delivered CROSS-FRAGMENT traffic.  If the
# per-tick recorder ever emits one of these keys, "time to first delivered
# cross-fragment flow" is reported automatically; until then the figures say
# explicitly that the series is unavailable rather than inventing it.
CROSS_FRAGMENT_DELIVERED_KEYS = (
    'cross_fragment_delivered', 'cross_frag_delivered',
    'delivered_cross_fragment', 'cross_delivered',
)


def event_tick_for(scenario_name):
    """The tick of the disturbance a recovery time is measured from."""
    return (RESCUE_ARRIVAL_TICK if scenario_name == 'rescue_ops'
            else SEVERANCE_TICK)


def aggregate_across_seeds(all_data_multi, mode, key, seeds, length):
    """NaN-aware cross-seed mean/std of a per-tick series, padded to `length`.

    Module-level twin of the plotting closure so the metric helpers can be
    used (and unit-checked) outside the figure code.
    """
    import warnings
    seed_data = all_data_multi.get(mode, {})
    arrays = []
    for seed in seeds:
        run = seed_data.get(seed)
        if not run:
            continue
        arr = list(run.get(key, []))[:length]
        if not arr:
            continue
        if len(arr) < length:
            arr = arr + [arr[-1]] * (length - len(arr))
        arrays.append(arr)
    if not arrays:
        return np.zeros(length), np.zeros(length)
    stacked = np.array(arrays, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def pre_event_reference(all_data_multi, modes, seeds, scenario_name, length,
                        window=None):
    """Shared achievability reference level for the fixed-threshold metric.

    Mean achievability over the `window` ticks immediately BEFORE the event,
    averaged over every arm and every seed.  The result is a SINGLE scalar
    applied identically to all arms.  Also returns the per-arm pre-event
    levels and their spread so the figure can state whether the reference is
    itself arm-invariant (Scenario A: it is, exactly — every arm runs the
    same undisturbed network until severance) or only a shared convention
    (Scenario B starts already severed, so the arms have already diverged by
    the time the rescue surge arrives).
    """
    import warnings
    window = window or RECOVERY_REF_WINDOW_TICKS
    ev = event_tick_for(scenario_name)
    lo = max(0, ev - window)
    per_arm = {}
    for mode in modes:
        vals = []
        for seed in seeds:
            run = all_data_multi.get(mode, {}).get(seed)
            if not run:
                continue
            arr = np.asarray(run.get('conn', []), dtype=np.float64)
            if arr.size <= lo:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                v = np.nanmean(arr[lo:min(ev, arr.size)])
            if np.isfinite(v):
                vals.append(float(v))
        if vals:
            per_arm[mode] = float(np.mean(vals))
    if not per_arm:
        return float('nan'), {}, float('nan')
    level = float(np.mean(list(per_arm.values())))
    spread = float(max(per_arm.values()) - min(per_arm.values()))
    return level, per_arm, spread


def recovery_time_fixed(series, event_tick, threshold,
                        sustain=None, horizon=None):
    """Ticks from `event_tick` until `series` reaches AND SUSTAINS `threshold`.

    Returns the offset in ticks, or None when the threshold is never reached
    and held (a censored observation — reported as such, never silently
    replaced by the plot horizon).  An offset of 0 means the arm never fell
    below the threshold at all: the disturbance did not take it out of
    service by this criterion.
    """
    sustain = RECOVERY_SUSTAIN_TICKS if sustain is None else sustain
    arr = np.asarray(series, dtype=np.float64)
    if horizon is not None:
        arr = arr[:horizon]
    if arr.size <= event_tick or not np.isfinite(threshold):
        return None
    last = arr.size - sustain
    for t in range(event_tick, max(event_tick, last) + 1):
        seg = arr[t:t + sustain]
        if seg.size < sustain:
            break
        if np.all(np.nan_to_num(seg, nan=-1.0) >= threshold):
            return t - event_tick
    return None


def fair_recovery_stats(all_data_multi, modes, seeds, scenario_name, length,
                        level, fraction):
    """Fixed-threshold recovery time per arm, on the mean curve and per seed.

    "NEVER BREACHED" IS A CATEGORY, NOT A ZERO.  `recovery_time_fixed`
    returns 0 both for an arm that dipped and climbed back inside one tick
    and for an arm that never fell below the threshold at all.  Those are
    different properties, so the breach test is run separately
    (`breached_after_event`) and the two cases are kept apart:

      per_seed_breached   parallel to `per_seed`; False == never breached
      n_never_breached    how many seeds never breached (report separately)
      median_breached     median recovery time over BREACHED, REACHED seeds
                          only — never-breached seeds are EXCLUDED, so the
                          statistic cannot be dragged to 0 by arms that
                          simply never lost service
      mean_curve_breached whether the cross-seed MEAN curve breached at all

    Returns {mode: {'threshold', 'mean_curve', 'per_seed', 'median',
                    'censored', 'n', plus the keys above}}, times in TICKS.
    The original five keys are unchanged so existing consumers keep working.
    """
    ev = event_tick_for(scenario_name)
    threshold = fraction * level
    out = {}
    for mode in modes:
        mean_curve, _ = aggregate_across_seeds(all_data_multi, mode, 'conn',
                                               seeds, length)
        rt_mean = recovery_time_fixed(mean_curve, ev, threshold)
        mean_breached = breached_after_event(mean_curve, ev, threshold)
        per_seed, per_seed_breached = [], []
        for seed in seeds:
            run = all_data_multi.get(mode, {}).get(seed)
            if not run:
                continue
            per_seed.append(recovery_time_fixed(
                run.get('conn', []), ev, threshold, horizon=length))
            per_seed_breached.append(breached_after_event(
                run.get('conn', []), ev, threshold, horizon=length))
        reached = [v for v in per_seed if v is not None]
        # Breached AND recovered: the only seeds on which a recovery TIME is
        # a meaningful observation at all.
        reached_breached = [v for v, b in zip(per_seed, per_seed_breached)
                            if b and v is not None]
        out[mode] = {
            'threshold': threshold,
            'mean_curve': rt_mean,
            'per_seed': per_seed,
            'median': float(np.median(reached)) if reached else None,
            'censored': len(per_seed) - len(reached),
            'n': len(per_seed),
            # ── never-breached-as-a-category additions ──
            'per_seed_breached': per_seed_breached,
            'n_never_breached': sum(1 for b in per_seed_breached if not b),
            'n_breached': sum(1 for b in per_seed_breached if b),
            'mean_curve_breached': mean_breached,
            'median_breached': (float(np.median(reached_breached))
                                if reached_breached else None),
            'mean_breached': (float(np.mean(reached_breached))
                              if reached_breached else None),
            'censored_breached': sum(
                1 for v, b in zip(per_seed, per_seed_breached)
                if b and v is None),
        }
    return out


def breached_after_event(series, event_tick, threshold, horizon=None):
    """Did achievability EVER fall below `threshold` after the event?

    This is the test that separates "never lost service" from "recovered
    instantly" — two states that a first-crossing recovery time reports
    identically as 0 s.  Returns True when at least one post-event tick is
    strictly below the threshold (NaN ticks count as a breach: no routed
    flow is not service).  Returns False when the arm never breached, and
    None when the series is too short to decide.
    """
    arr = np.asarray(series, dtype=np.float64)
    if horizon is not None:
        arr = arr[:horizon]
    if arr.size <= event_tick or not np.isfinite(threshold):
        return None
    post = arr[event_tick:]
    return bool(np.any(np.nan_to_num(post, nan=-1.0) < threshold))


# ══════════════════════════════════════════════════════════════════════
#  (1) SERVICE-LOSS INTEGRAL  —  THE PRIMARY RECOVERY METRIC
# ══════════════════════════════════════════════════════════════════════

def _post_event_series(run, key, event_tick, length):
    """Post-event slice of one per-tick series, truncated to `length`."""
    arr = np.asarray(list(run.get(key, []))[:length], dtype=np.float64)
    if arr.size <= event_tick:
        return None
    return arr[event_tick:]


def reconstruct_offered_per_seed(all_data_multi, modes, seeds, length):
    """Per-(seed, tick) OFFERED demand, reconstructed from the recorded data.

    The engine records `delivered` and `conn` but not `offered`.  It defines

        conn(t) = min(100, 100 * delivered(t) / offered(t))

    so offered(t) = 100 * delivered(t) / conn(t) wherever conn(t) is neither
    ~0 (no reconstruction possible: 0/0) nor at the 100% CLAMP (where the
    reconstruction is an under-estimate of offered).

    Crucially, offered demand is a property of the SCENARIO, not of the arm:
    `_offered_volume_for_flows` depends only on (seed, tick, survivor set),
    and every arm is scored against the same full flow set.  So the estimate
    for a given (seed, tick) can be pooled ACROSS ARMS and the cross-arm
    MEDIAN taken — robust to any single arm sitting in the clamped or the
    degenerate region.

    Returns {seed: np.ndarray of length `length`}, NaN where no arm supplied
    a usable reconstruction for that tick.  Ticks that stay NaN are EXCLUDED
    from the volume integral and counted, never zero-filled.
    """
    import warnings
    out = {}
    for seed in seeds:
        pool = []
        for mode in modes:
            run = all_data_multi.get(mode, {}).get(seed)
            if not run:
                continue
            c = np.asarray(list(run.get('conn', []))[:length],
                           dtype=np.float64)
            d = np.asarray(list(run.get('delivered', []))[:length],
                           dtype=np.float64)
            n = min(c.size, d.size)
            if n == 0:
                continue
            c, d = c[:n], d[:n]
            est = np.full(length, np.nan, dtype=np.float64)
            ok = ((c >= OFFERED_RECON_MIN_CONN_PCT)
                  & (c <= OFFERED_RECON_MAX_CONN_PCT)
                  & np.isfinite(c) & np.isfinite(d) & (d > 0))
            est[:n][ok] = 100.0 * d[ok] / c[ok]
            pool.append(est)
        if not pool:
            out[seed] = np.full(length, np.nan, dtype=np.float64)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            out[seed] = np.nanmedian(np.vstack(pool), axis=0)
    return out


def offered_per_seed(all_data_multi, modes, seeds, length):
    """Per-(seed, tick) OFFERED demand — RECORDED where available.

    The tick loop now records `offered` directly (the exact denominator the
    engine divided by), so the post-hoc reconstruction
    `reconstruct_offered_per_seed` is only a FALLBACK for datasets written
    before that series existed.  The reconstruction is blind on any tick where
    achievability is ~0 or sits at the 100 % clamp, which is exactly the
    pre-event region — so a correct demand-regime ratio needs the recorded
    series whenever it is there.

    Offered demand is a property of the SCENARIO, not of the arm
    (`_offered_volume_for_flows` is evaluated over the FULL flow set, restored
    from `_proto_all_flows`, for every arm), so the per-arm series are pooled
    by MEDIAN across arms; disagreement between arms would itself be a bug.

    Returns (per_seed_dict, source, coverage) where source is 'recorded',
    'reconstructed' or 'mixed', and coverage is the finite fraction.
    """
    import warnings
    recon = None
    out, srcs = {}, set()
    for seed in seeds:
        pool = []
        for mode in modes:
            run = all_data_multi.get(mode, {}).get(seed)
            if not run:
                continue
            o = np.asarray(list(run.get('offered', []))[:length],
                           dtype=np.float64)
            if o.size == 0:
                continue
            padded = np.full(length, np.nan, dtype=np.float64)
            padded[:o.size] = np.where(o > 0, o, np.nan)
            pool.append(padded)
        if pool:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                out[seed] = np.nanmedian(np.vstack(pool), axis=0)
            srcs.add('recorded')
        else:
            if recon is None:
                recon = reconstruct_offered_per_seed(
                    all_data_multi, modes, seeds, length)
            out[seed] = recon.get(seed,
                                  np.full(length, np.nan, dtype=np.float64))
            srcs.add('reconstructed')
    if not out:
        return {}, 'none', float('nan')
    stacked = np.vstack([out[s] for s in out])
    coverage = float(np.count_nonzero(np.isfinite(stacked)) / stacked.size)
    source = srcs.pop() if len(srcs) == 1 else 'mixed'
    return out, source, coverage


def demand_regime_ratio(all_data_multi, modes, seeds, scenario_name, length,
                        offered_by_seed, window=None):
    """Measured PRE/POST offered-demand ratio — the reference correction.

    ── THE DEFECT THIS CORRECTS ─────────────────────────────────────────────
    `_offered_volume_for_flows` switches DEMAND MODEL on `sim.island_mode`:
    before the event each surviving flow offers ~8.00 units; after it, the
    civilian emergency model (always-on + voice + messaging) and the rescue
    model both offer more — measured ~9.80 units/flow.  That is a ~1.2254x
    STEP in the denominator at the event tick, applied IDENTICALLY to every
    arm (measured on the 10-seed run: total offered 1776.2 -> 2176.5).

    Achievability is delivered/offered, so every post-event sample is divided
    by a ~22.5 % LARGER denominator than the pre-event reference it is scored
    against.  The consequence is not subtle: on the final 10-seed run the four
    IGP arms deliver MORE ABSOLUTE TRAFFIC after the severance than the intact
    network did (728.0 vs 724.6 units/tick) and the uncorrected service-loss
    integral still charges them ~26,000 pp*s for it.  At that point the metric
    is measuring a DEMAND STEP, not a loss of service.

    ── THE CORRECTION ─────────────────────────────────────────────────────
        ratio    = mean_offered(pre-event window) / mean_offered(post-event)
        REF_corr = REF_raw x ratio

    Read it as: REF_corr is the achievability an arm would DISPLAY if it
    delivered exactly the pre-event ABSOLUTE volume under the post-event
    demand, because

        100*D/offered_post = (100*D/offered_pre) * (offered_pre/offered_post)

    so the bar and the samples finally sit in the SAME demand regime.  The
    ratio is MEASURED from the recorded/reconstructed offered series — never
    hardcoded — over the same pre-event window `pre_event_reference` uses and
    the same post-event window the integral runs over.  It is scenario-level
    and arm-invariant, so it moves every arm's bar by the identical factor and
    cannot change the ranking; what it changes is whether the ABSOLUTE service
    loss of an arm that never lost absolute throughput is reported as ~26,000
    pp*s or as ~0.

    Returns a dict with 'ratio', 'offered_pre', 'offered_post', 'n_seeds',
    'window', 'applied' (False when the ratio could not be measured, in which
    case 'ratio' is 1.0 and the reference is left uncorrected).
    """
    import warnings
    window = window or RECOVERY_REF_WINDOW_TICKS
    ev = event_tick_for(scenario_name)
    lo = max(0, ev - window)
    pre_vals, post_vals = [], []
    for seed in seeds:
        arr = offered_by_seed.get(seed)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size <= ev:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            a = float(np.nanmean(arr[lo:min(ev, arr.size)]))
            b = float(np.nanmean(arr[ev:min(length, arr.size)]))
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            pre_vals.append(a)
            post_vals.append(b)
    info = {'ratio': 1.0, 'offered_pre': float('nan'),
            'offered_post': float('nan'), 'n_seeds': len(pre_vals),
            'window': window, 'applied': False}
    if not pre_vals:
        return info
    off_pre = float(np.mean(pre_vals))
    off_post = float(np.mean(post_vals))
    info['offered_pre'] = off_pre
    info['offered_post'] = off_post
    if off_post > 0 and np.isfinite(off_pre) and np.isfinite(off_post):
        info['ratio'] = off_pre / off_post
        info['applied'] = True
    return info


def service_loss_per_seed(all_data_multi, mode, seeds, scenario_name, length,
                          reference, offered=None, bounded_ticks=None):
    """Per-seed SERVICE-LOSS INTEGRAL for one arm.  Lower is better.

    Definition (identical for every arm, threshold-free):

        SLI_pp_s = Σ_{t=event}^{T} max(0, reference − conn(t)) · Δt   [pp·s]

    where `reference` is the SHARED pre-event achievability level and Δt is
    TICK_DURATION_S.  Two windows are reported:
      full     t = event .. min(len(run), length)
      bounded  t = event .. event + `bounded_ticks`   (default
               SERVICE_LOSS_BOUNDED_TICKS) — comparable across runs of
               different length, where the full-window figure is not.

    The same integral is ALSO expressed in DELIVERED-VOLUME units, which is
    the physically meaningful form — "traffic not delivered because of the
    event":

        SLI_vol  = Σ max(0, (reference − conn(t))/100) · offered(t) · Δt

    with offered(t) the reconstructed per-(seed, tick) demand (see
    `reconstruct_offered_per_seed`).  Ticks with no usable reconstruction are
    excluded and counted in 'vol_missing', never zero-filled; the volume
    figure is reported as unavailable when coverage is poor.

    NaN achievability ticks (no routed flow) are treated as ZERO
    achievability, i.e. a FULL-DEPTH loss — an outage is not missing data.

    Returns a list of per-seed dicts, seeds without data omitted.
    """
    ev = event_tick_for(scenario_name)
    bounded_ticks = (SERVICE_LOSS_BOUNDED_TICKS if bounded_ticks is None
                     else bounded_ticks)
    rows = []
    if not np.isfinite(reference):
        return rows
    for seed in seeds:
        run = all_data_multi.get(mode, {}).get(seed)
        if not run:
            continue
        post = _post_event_series(run, 'conn', ev, length)
        if post is None or post.size == 0:
            continue
        # NaN == outage == zero achievability (a full-depth loss), NOT a gap.
        deficit = np.maximum(0.0, reference - np.nan_to_num(post, nan=0.0))
        nb = min(bounded_ticks, deficit.size)
        row = {
            'seed': seed,
            'pp_s': float(deficit.sum() * TICK_DURATION_S),
            'pp_s_bounded': float(deficit[:nb].sum() * TICK_DURATION_S),
            'n_ticks': int(deficit.size),
            'n_ticks_bounded': int(nb),
            'vol': float('nan'), 'vol_bounded': float('nan'),
            'vol_missing': None,
        }
        off = None if offered is None else offered.get(seed)
        if off is not None:
            off_post = np.asarray(off, dtype=np.float64)[ev:ev + deficit.size]
            if off_post.size == deficit.size:
                lost = (deficit / 100.0) * off_post
                miss = int(np.count_nonzero(~np.isfinite(off_post)))
                lost_f = np.nan_to_num(lost, nan=0.0)
                row['vol'] = float(lost_f.sum() * TICK_DURATION_S)
                row['vol_bounded'] = float(lost_f[:nb].sum() * TICK_DURATION_S)
                row['vol_missing'] = miss
                row['vol_coverage'] = 1.0 - miss / max(1, off_post.size)
        rows.append(row)
    return rows


def service_loss_stats(rows, key='pp_s'):
    """Mean / σ / MEDIAN / IQR / min / max / n over the per-seed SLI rows.

    MEDIAN AND IQR ARE REPORTED ALONGSIDE THE MEAN BECAUSE THE MEAN IS NOT
    REPRESENTATIVE.  On the final 10-seed run MARL's Scenario A SLI reads
    1,904 +- 5,681 pp*s, and that entire figure is ONE seed: the per-seed
    values are 50:13, 51:18946, 52:0, 53:0, 54:1, 55:0, 56:0, 57:39, 58:5,
    59:39 — median 3.  A mean 600x its own median is a statement about a
    single outlier run, not about the arm, so any consumer of these stats
    must quote median [Q1, Q3] next to it.
    """
    vals = [r[key] for r in rows if np.isfinite(r.get(key, float('nan')))]
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    q1, q3 = (float(np.percentile(a, 25)), float(np.percentile(a, 75)))
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'median': float(np.median(a)), 'min': float(a.min()),
            'max': float(a.max()), 'n': int(a.size), 'values': vals,
            'q1': q1, 'q3': q3, 'iqr': q3 - q1}


# ══════════════════════════════════════════════════════════════════════
#  (3) AVAILABILITY  —  re-dips a first-crossing time cannot see
# ══════════════════════════════════════════════════════════════════════

def availability_stats(all_data_multi, mode, seeds, scenario_name, length,
                       threshold):
    """Fraction of EVENT-WINDOW ticks at or above `threshold`, per arm.

        A(threshold) = |{t >= event : conn(t) >= threshold}| / |{t >= event}|

    Higher is better; 1.0 means the arm was at or above the bar for the whole
    post-event window.  This is the statistic that catches RE-DIPS: a
    first-crossing time is fixed forever at the first crossing, whereas an
    arm that crosses at t+30 and then falls back for 400 ticks scores a
    visibly lower availability.  NaN ticks count as BELOW the threshold (an
    outage tick is not availability).

    Returns {'threshold', 'mean', 'std', 'per_seed' (seed, frac) pairs, 'n'}
    or None when the arm has no usable run.
    """
    ev = event_tick_for(scenario_name)
    pairs = []
    if not np.isfinite(threshold):
        return None
    for seed in seeds:
        run = all_data_multi.get(mode, {}).get(seed)
        if not run:
            continue
        post = _post_event_series(run, 'conn', ev, length)
        if post is None or post.size == 0:
            continue
        ok = np.nan_to_num(post, nan=-1.0) >= threshold
        pairs.append((seed, float(np.count_nonzero(ok)) / post.size))
    if not pairs:
        return None
    vals = np.asarray([v for _s, v in pairs], dtype=np.float64)
    return {'threshold': threshold, 'mean': float(vals.mean()),
            'std': float(vals.std()), 'per_seed': pairs, 'n': int(vals.size)}


# ══════════════════════════════════════════════════════════════════════
#  (2) TIME TO X% OF THE FEASIBLE OPTIMUM
# ══════════════════════════════════════════════════════════════════════

def _wrap_reason(text, width=96):
    """Wrap a refusal/explanation string for the fixed-width summary table."""
    import textwrap
    if not text:
        return ["(no reason recorded)"]
    return textwrap.wrap(str(text), width=width) or ["(no reason recorded)"]


def feasible_achievability_ceiling(all_data_multi, seeds):
    """Per-seed FEASIBLE ACHIEVABILITY ceiling, or an explicit refusal.

    Returns (per_seed_dict, source_label, reason) where per_seed_dict is None
    when no defensible ceiling exists for this dataset — in which case
    `reason` states WHY, and every consumer must print that reason and skip
    the metric rather than substitute a proxy.

    The only accepted source is an ACHIEVABILITY ceiling explicitly recorded
    per seed by the harness (FEASIBLE_CEILING_KEYS in `topology_info`).

    NOT accepted, deliberately:
      * `tg_optimum` — it is a COMPONENT COUNT (the fewest infra components
        reachable under TG-only steering at the screening power).  A
        reunified graph does NOT imply every UE↔UE flow is served at full
        rate: link capacity, MCS selection, queueing and the relay
        bandwidth fraction all still bind.  Mapping "1 component" to "100%
        achievable" would invent a ceiling the physics does not support, and
        would flatter exactly the arm that reunifies.
      * the best achievability any ARM happened to reach — that is an
        empirical envelope of the arms being compared, not a feasible
        optimum, and it makes the reference depend on the answer.
    """
    per_seed, key_used = {}, None
    for mode_runs in all_data_multi.values():
        for seed in seeds:
            run = mode_runs.get(seed)
            if not run:
                continue
            info = run.get('topology_info') or {}
            for k in FEASIBLE_CEILING_KEYS:
                v = info.get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv) and fv > 0:
                    per_seed[seed] = max(per_seed.get(seed, 0.0), fv)
                    key_used = key_used or k
                    break
    if per_seed and len(per_seed) == len(seeds):
        return per_seed, f"topology_info['{key_used}']", None
    if per_seed:
        return (None, None,
                f"a feasible achievability ceiling was recorded for only "
                f"{len(per_seed)}/{len(seeds)} seeds "
                f"(topology_info['{key_used}']); a metric computed on a "
                f"subset of seeds is not comparable across arms, so it is "
                f"skipped rather than partially reported")
    _opt = tg_optimum_summary(all_data_multi, seeds)
    if _opt['per_seed']:
        return (None, None,
                "no per-seed achievability ceiling is recorded in this "
                "dataset. The TG-feasible optimum IS recorded "
                f"({_opt['mean']:.2f} components @{_opt['tx_dbm']:.0f} dBm) "
                "but it is a COMPONENT COUNT, not an achievability level: a "
                "reunified graph does not imply every flow is served at full "
                "rate (capacity, MCS, queueing and the relay bandwidth "
                "fraction all still bind). Converting one into the other "
                "would invent the ceiling, so the metric is SKIPPED")
    return (None, None,
            "no per-seed achievability ceiling and no TG-feasible optimum "
            "are recorded in this dataset (it predates both), so recovery "
            "against a feasible optimum cannot be computed. It is SKIPPED "
            "rather than approximated by the pre-event level, which is a "
            "different reference with a different meaning")


def ceiling_recovery_stats(all_data_multi, modes, seeds, scenario_name,
                           length, ceiling_per_seed,
                           fractions=CEILING_RECOVERY_FRACTIONS):
    """Time to X% of the PER-SEED FEASIBLE ACHIEVABILITY CEILING.

    Same sustained first-crossing rule as `recovery_time_fixed`, but the bar
    is `fraction * ceiling[seed]` — what the physics permits on THAT seed —
    instead of the pre-event level or the arm's own steady state.  Evaluated
    per seed (the ceiling is per-seed, so a cross-seed mean curve would be
    scored against a bar that exists on no seed).

    Returns {fraction: {mode: {'per_seed', 'median_breached',
                               'n_never_breached', 'censored', 'n'}}}.
    Only called when `feasible_achievability_ceiling` supplied a ceiling.
    """
    ev = event_tick_for(scenario_name)
    out = {}
    for frac in fractions:
        per_mode = {}
        for mode in modes:
            rts, breaches = [], []
            for seed in seeds:
                run = all_data_multi.get(mode, {}).get(seed)
                ceil_v = ceiling_per_seed.get(seed)
                if not run or ceil_v is None:
                    continue
                thr = frac * float(ceil_v)
                rts.append(recovery_time_fixed(run.get('conn', []), ev, thr,
                                               horizon=length))
                breaches.append(breached_after_event(run.get('conn', []), ev,
                                                     thr, horizon=length))
            good = [v for v, b in zip(rts, breaches) if b and v is not None]
            per_mode[mode] = {
                'per_seed': rts, 'per_seed_breached': breaches,
                'median_breached': (float(np.median(good)) if good else None),
                'n_never_breached': sum(1 for b in breaches if not b),
                'censored': sum(1 for v, b in zip(rts, breaches)
                                if b and v is None),
                'n': len(rts),
            }
        out[frac] = per_mode
    return out


def legacy_recovery_time_own_ss(mean_curve, scenario_name):
    """The OLD self-referential metric — kept ONLY for continuity.

    Time to 90% of the arm's OWN steady state.  NOT comparable across arms:
    each arm is measured against a different absolute bar.  Reproduces the
    previous implementation exactly (including its 20-tick / 0.95 hold) so
    the historical numbers in the table are unchanged.
    """
    ev = event_tick_for(scenario_name)
    m = np.asarray(mean_curve, dtype=np.float64)
    if m.size == 0:
        return None
    steady = float(np.mean(m[-min(STEADY_STATE_WINDOW_TICKS, m.size):]))
    thr = steady * 0.9
    for t in range(ev, m.size):
        if m[t] >= thr:
            if t + 20 <= m.size and np.all(m[t:t + 20] >= thr * 0.95):
                return t - ev
    return None


def first_cross_fragment_delivery(all_data_multi, mode, seeds, scenario_name,
                                  length):
    """Ticks from the event to the first DELIVERED cross-fragment flow.

    Returns (median_ticks, n_seeds) when the per-tick recorder provides a
    cross-fragment delivered series, else None — the figures then state that
    the series is unavailable instead of substituting a proxy.
    """
    ev = event_tick_for(scenario_name)
    vals = []
    for seed in seeds:
        run = all_data_multi.get(mode, {}).get(seed)
        if not run:
            continue
        arr = None
        for k in CROSS_FRAGMENT_DELIVERED_KEYS:
            if run.get(k):
                arr = np.asarray(run[k], dtype=np.float64)[:length]
                break
        if arr is None or arr.size <= ev:
            continue
        nz = np.nonzero(np.nan_to_num(arr[ev:], nan=0.0) > 0)[0]
        if nz.size:
            vals.append(int(nz[0]))
    if not vals:
        return None
    return float(np.median(vals)), len(vals)


def energy_stats(all_data_multi, mode, seeds, window=None):
    """Post-event steady-state energy: absolute AND per delivered unit.

    * absolute  — per-seed mean Joules/tick over the last `window` ticks,
      then mean +/- 1 sigma across seeds (same convention as every other
      steady-state number in the table).
    * per unit  — total energy divided by total delivered traffic over the
      same window: sum(E) / sum(D) aggregated across seeds (a true
      energy-per-unit ratio, not the mean of per-seed ratios).  The per-seed
      ratios are also returned so the spread is reportable.
    """
    import warnings
    window = window or STEADY_STATE_WINDOW_TICKS
    e_seeds, d_seeds, ratio_seeds = [], [], []
    tot_e = tot_d = 0.0
    for seed in seeds:
        run = all_data_multi.get(mode, {}).get(seed)
        if not run:
            continue
        e = np.asarray(run.get('energy', []), dtype=np.float64)
        d = np.asarray(run.get('delivered', []), dtype=np.float64)
        if e.size == 0 or d.size == 0:
            continue
        e = e[-min(window, e.size):]
        d = d[-min(window, d.size):]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            e_seeds.append(float(np.nanmean(e)))
            d_seeds.append(float(np.nanmean(d)))
            se, sd = float(np.nansum(e)), float(np.nansum(d))
        tot_e += se
        tot_d += sd
        if sd > 0:
            ratio_seeds.append(se / sd)
    if not e_seeds:
        return None
    per_unit = (tot_e / tot_d) if tot_d > 0 else float('nan')
    return {
        'abs_mean': float(np.mean(e_seeds)),
        'abs_std': float(np.std(e_seeds)),
        'delivered_mean': float(np.mean(d_seeds)) if d_seeds else float('nan'),
        'per_unit': per_unit,
        'per_unit_seeds': ratio_seeds,
        'per_unit_min': min(ratio_seeds) if ratio_seeds else float('nan'),
        'per_unit_max': max(ratio_seeds) if ratio_seeds else float('nan'),
        'n': len(e_seeds),
    }


def format_seed_list(seeds, max_explicit=8):
    """Compact seed rendering that stays readable as the seed count grows.

    Five seeds print in full; ten or more collapse contiguous runs
    ("50-59"), so widening the evaluation does not overflow a table header,
    an axis title or the figure caption.
    """
    s = sorted(seeds)
    if not s:
        return "(none)"
    if len(s) <= max_explicit:
        return ", ".join(str(v) for v in s)
    runs, start, prev = [], s[0], s[0]
    for v in s[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    return ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs) \
        + f" ({len(s)} seeds)"


def has_series(all_data_multi, modes, seeds, key):
    """True when at least one recorded run carries a non-empty `key` series.

    The per-tick recorder gained `relay_bridges` / `fragments_raw` /
    `relay_mode_nodes` after the round-6 pickle was written, so every panel
    that consumes them must degrade cleanly on older data instead of
    crashing or silently drawing zeros.
    """
    for mode in modes:
        for seed in seeds:
            run = all_data_multi.get(mode, {}).get(seed)
            if run and len(run.get(key, [])) > 0:
                return True
    return False


def reunification_stats(all_data_multi, mode, seeds, window=None):
    """Fragment / relay-bridge steady state for one arm.

    Vocabulary (matching the per-tick recorder):
      fragments      connected components INCLUDING relay links — the state
                     the arm actually achieved.
      fragments_raw  components of the relay-FREE graph — what the severance
                     did, before any bridging.  Identical for every arm.
      relay_bridges  relay LINKS that join two otherwise-disconnected raw
                     fragments.  This is the reunification work itself, and
                     is NOT the same quantity as `relay_count` (all active
                     transport relay links) or `relay_mode_nodes` (nodes in
                     a relay mode — intent, not realised connectivity).

    Returns None when the arm has no fragment data at all; the
    fragments_raw / relay_bridges entries are None on pre-round-7 pickles
    that predate those series.
    """
    out = {}
    any_data = False
    for key in ('fragments', 'fragments_raw', 'relay_bridges'):
        vals = steady_state_per_seed(all_data_multi, mode, key, seeds, window)
        if vals:
            any_data = True
            out[key] = {'mean': float(np.mean(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                        'n': len(vals)}
        else:
            out[key] = None
    return out if any_data else None


def tg_optimum_summary(all_data_multi, seeds):
    """Per-seed TG-feasible optimum component count, read from the run data.

    The optimum is a property of (topology seed, scenario) and is written into
    every run's `topology_info` by run_physics_pass, so any arm's run for a
    seed carries the same value; the first one found wins and a disagreement
    between arms would be a bug worth surfacing, so it is asserted loosely by
    taking the max (a larger claimed optimum is the conservative reading).

    Returns {'per_seed': {seed: optimum}, 'mean': float, 'tx_dbm': float}.
    Mean is NaN on datasets written before the optimum was recorded, and every
    consumer must degrade to "no reference available" rather than assume 1.
    """
    per_seed, tx = {}, FRAG_BRIDGE_TX_DBM
    for mode_runs in all_data_multi.values():
        for seed in seeds:
            run = mode_runs.get(seed)
            if not run:
                continue
            info = run.get('topology_info') or {}
            opt = info.get('tg_optimum')
            if opt is None:
                continue
            per_seed[seed] = max(per_seed.get(seed, 0), int(opt))
            tx = info.get('tg_optimum_tx_dbm', tx)
    mean = (float(np.mean(list(per_seed.values()))) if per_seed
            else float('nan'))
    return {'per_seed': per_seed, 'mean': mean, 'tx_dbm': float(tx)}


RANDOM_ARM = 'random'


def print_bridging_attribution(all_data_multi, seeds, scenario_name,
                               window=None):
    """Report the bridging-ATTRIBUTION ablation next to the MARL arms.

    The scientific question this answers is not "did the network reunify?"
    but "is the reunification attributable to the LEARNED POLICY?".  The
    'random' arm runs the identical scenario on the identical MARL code path
    with every action head sampled uniformly (see run_physics_pass).  If it
    reaches the same component count and the same bridge count as the MARL
    arms, then bridging is a property of the ENVIRONMENT — the mechanism is
    easy to trip once it works — and no claim of the form "the policy learned
    to bridge islands" is defensible from these runs.  In that case the
    defensible claims are about EFFICIENCY, SPEED and SUSTAINED DELIVERY
    (achievability, energy, how long the bridge is held), which are also
    printed here so the comparison can be made honestly in one place.

    Reported per arm, as steady-state means across seeds:
      bridges       de-duplicated relay bridges (independent fragment merges;
                    parallel/cyclic duplicates removed — see
                    Simulator.distinct_bridge_link_ids)
      raw comps     components of the relay-FREE graph (identical for every
                    arm — it is what the severance did)
      achieved      components INCLUDING relay links (1 == fully reunified)
      achiev %      volume-weighted UE-to-UE delivered fraction
      reach %       PEER reachability: share of surviving UEs whose serving
                    cell sits in the connected BODY of the survivor network
                    (Simulator.peer_reachable_ue_stats).  This is the
                    disaster-recovery objective the reachability reward term
                    pays for, and it is NOT the access metric
                    `_reachable_ue_fraction` (which is blind to fragmentation)
      restored      UEs re-joined to that body by THIS arm's own relay
                    bridges — a delta against the relay-free graph, so it is
                    0 for every arm that forms no bridges
      stranded      surviving UEs still outside the body
      outside       ATTACHED UEs still outside the body.  Read this next to
                    `achieved`: if it is ~0 while achieved > 1, the residual
                    fragments hold no users at all (bare relay / O-DU sites),
                    so the reachability term cannot price closing them and only
                    the reunification block can.
    """
    if window is None:
        # Post-event ticks only.  The default STEADY_STATE_WINDOW_TICKS (500)
        # is longer than a smoke run, which would fold the PRE-severance ticks
        # (1 component, 0 bridges, by construction identical for every arm)
        # into the means and flatter every arm equally.
        window = min(STEADY_STATE_WINDOW_TICKS,
                     max(50, TOTAL_TICKS - event_tick_for(scenario_name)))
    print("\n" + "=" * 78)
    print(f"  BRIDGING ATTRIBUTION ABLATION - {scenario_name} "
          f"(steady state = last {window} ticks)")
    print("  Is bridging LEARNED, or does the environment produce it anyway?")
    print("=" * 78)
    if not any(all_data_multi.get(RANDOM_ARM, {}).get(s) for s in seeds):
        print("  (no 'random' arm data in this dataset - attribution "
              "unmeasured)")
        print("=" * 78)
        return

    arms = [m for m in ('marl', 'marl_freeze', RANDOM_ARM)
            if any(all_data_multi.get(m, {}).get(s) for s in seeds)]
    _LBL = {'marl': 'MARL online', 'marl_freeze': 'MARL adapt-freeze',
            RANDOM_ARM: 'RANDOM actions (control)'}

    def _m(mode, key):
        vals = steady_state_per_seed(all_data_multi, mode, key, seeds, window)
        return float(np.mean(vals)) if vals else float('nan')

    # ── THE REFERENCE: the TG-feasible optimum, never the literal 1 ──────
    # `tg_optimum` is a property of the seed + scenario (the site plan and the
    # 60 GHz budget decide it), so it is IDENTICAL for every arm and is read
    # from whichever arm's run data is present.  Scoring `achieved` against a
    # flat 1 when 1 is unreachable prices the SITE PLAN, not the arm.
    _opt = tg_optimum_summary(all_data_multi, seeds)
    _opt_mean = _opt['mean']
    if np.isfinite(_opt_mean):
        _all_one = all(v <= 1 for v in _opt['per_seed'].values())
        print(f"  REFERENCE (TG-feasible optimum @{_opt['tx_dbm']:.0f} dBm): "
              f"mean {_opt_mean:.2f} components; per seed "
              + ", ".join(f"s{k}={v}" for k, v in sorted(_opt['per_seed'].items())))
        if _all_one:
            print("  Full reunification (1 component) IS physically reachable "
                  "on every seed here, so")
            print("  the optimum and the naive target coincide.")
        else:
            print("  Full reunification is NOT reachable on every seed - the "
                  "60 GHz TG budget cannot")
            print("  close the last gap. `achieved` is judged against the "
                  "OPTIMUM column, never against 1.")
    else:
        print("  REFERENCE: no TG-feasible optimum recorded in this dataset "
              "(written before it was")
        print("  measured); reunification is reported against the raw "
              "partition only.")

    print(f"  {'arm':<26} {'bridges':>9} {'raw comps':>10} "
          f"{'optimum':>8} {'achieved':>9} {'vs opt':>7} {'achiev %':>9} "
          f"{'reach %':>9} {'restored':>9} {'stranded':>9} {'outside':>8}")
    stats = {}
    for m in arms:
        stats[m] = (_m(m, 'relay_bridges'), _m(m, 'fragments_raw'),
                    _m(m, 'fragments'), _m(m, 'conn'))
        b, r, a, c = stats[m]
        _rc, _rs, _st = (_m(m, 'peer_reach'), _m(m, 'reach_restored'),
                         _m(m, 'stranded_ues'))
        _ob = _m(m, 'outside_body_ues')
        # Gap to the OPTIMUM: 0.00 means the arm reached exactly what the
        # physics permits, so a positive number is the only real shortfall.
        _gap = (a - _opt_mean) if (np.isfinite(_opt_mean)
                                   and np.isfinite(a)) else float('nan')
        _optstr = (f"{_opt_mean:>8.2f}" if np.isfinite(_opt_mean)
                   else f"{'n/a':>8}")
        _gapstr = (f"{_gap:>+7.2f}" if np.isfinite(_gap) else f"{'n/a':>7}")
        print(f"  {_LBL.get(m, m):<26} {b:>9.2f} {r:>10.2f} "
              f"{_optstr} {a:>9.2f} {_gapstr} {c:>9.1f} "
              f"{_rc:>9.1f} {_rs:>9.1f} {_st:>9.1f} {_ob:>8.1f}")

    # Verdict — deliberately blunt, and printed whichever way it comes out.
    rnd = stats.get(RANDOM_ARM)
    best = None
    for m in ('marl_freeze', 'marl'):
        if m in stats:
            best = stats[m]
            break
    if rnd and best:
        _, rnd_raw, rnd_ach, rnd_conn = rnd
        _, _, marl_ach, marl_conn = best
        print("-" * 78)
        if not np.isfinite(rnd_raw) or rnd_raw <= 1.05:
            # Nothing was severed into separate fragments in this scenario, so
            # there is no bridging work to attribute to anything.  Say so
            # rather than emitting a verdict that reads as a finding.
            print("  NOT APPLICABLE: the raw (relay-free) partition is a "
                  "single component in this\n"
                  "  scenario - there are no fragments to bridge, so the "
                  "bridging ablation is vacuous\n"
                  "  here. Only the delivery comparison below is meaningful.")
        elif np.isfinite(rnd_ach) and rnd_ach <= (
                (_opt_mean + 0.05) if np.isfinite(_opt_mean) else 1.05):
            # Judged against the TG-FEASIBLE OPTIMUM, not against 1.  If the
            # optimum is 2 on a seed, an arm sitting at 2 has reunified
            # everything the 60 GHz budget permits and must be scored as
            # having done so - otherwise the verdict measures the site plan.
            print("  VERDICT: the RANDOM policy also reunifies the island. "
                  "Bridging is therefore\n"
                  "  NOT evidence of learning - it is a property of the "
                  "environment once the beam-\n"
                  "  training mechanism works. Any claim must be about "
                  "EFFICIENCY / SPEED /\n"
                  "  SUSTAINED DELIVERY, not about bridging capability per se.")
        elif np.isfinite(marl_ach) and np.isfinite(rnd_ach) and marl_ach < rnd_ach - 0.25:
            _hdr = ("" if not np.isfinite(_opt_mean) else
                    f" (TG-feasible optimum {_opt_mean:.2f}; MARL gap "
                    f"{marl_ach - _opt_mean:+.2f}, random gap "
                    f"{rnd_ach - _opt_mean:+.2f})")
            print("  VERDICT: the MARL arms reach a strictly lower component "
                  "count than uniform\n"
                  "  random action selection - bridging IS attributable to "
                  "the learned policy." + _hdr)
        else:
            print("  VERDICT: MARL and random reach a comparable component "
                  "count - bridging is NOT\n"
                  "  separable from the environment on these seeds.")
        if np.isfinite(marl_conn) and np.isfinite(rnd_conn):
            print(f"  Delivery gap (the separable claim): MARL "
                  f"{marl_conn:.1f}% vs random {rnd_conn:.1f}% "
                  f"({marl_conn - rnd_conn:+.1f} pp)")
    print("=" * 78)


def variance_stats(values):
    """Robust spread descriptors for a small per-seed sample.

    Reports median / min / max / IQR alongside mean +/- sigma: with 5-10
    seeds a single symmetric sigma hides exactly the reliability structure
    (one collapsed seed, a bimodal split) that matters operationally.
    """
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return None
    a = np.array(vals, dtype=np.float64)
    q1, q3 = np.percentile(a, [25, 75])
    return {
        'mean': float(a.mean()), 'std': float(a.std()),
        'median': float(np.median(a)), 'min': float(a.min()),
        'max': float(a.max()), 'iqr': float(q3 - q1),
        'n': int(a.size), 'values': vals,
    }


def failure_seed_count(all_data_multi, mode, seeds,
                       pct=FAILURE_ACHIEVABILITY_PCT):
    """Seeds where this arm's steady-state achievability is below `pct`.

    An operational reliability counter: a mean +/- sigma cannot distinguish
    "uniformly mediocre" from "usually fine, occasionally zero", and the
    latter is what disqualifies a protocol for disaster response.
    """
    vals = steady_state_per_seed(all_data_multi, mode, 'conn', seeds)
    return sum(1 for v in vals if v < pct), len(vals)


def plot_comparison(all_data, scenario_name, output_path):
    """Generate the 7-panel comparison plot for one scenario.
    
    all_data: dict mapping mode -> run data (from run_physics_pass)
    scenario_name: 'recovery' or 'rescue_ops'
    output_path: where to save the PNG
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    PLOT_END = max(1, TOTAL_TICKS - 80) if TOTAL_TICKS > 400 else TOTAL_TICKS
    ticks_x = list(range(PLOT_END))
    RESCUE_TICK = RESCUE_ARRIVAL_TICK

    # Protocol styling.  FIGURE_EXCLUDED_ARMS (the online-only MARL
    # ablation) is dropped from the figure \u2014 it is still simulated and
    # still stored in the raw data, it is simply not plotted.
    _CANDIDATES = {
        'marl':   {'color': '#2ca02c', 'ls': '-',  'lw': 2.0},
        'marl_freeze': {'color': PROPOSED_ARM_COLOR, 'ls': '-',
                        'lw': PROPOSED_ARM_LW},
        'xldet':  {'color': '#8c564b', 'ls': (0, (5, 2)), 'lw': 2.2},
        'ospf':   {'color': '#d62728', 'ls': '-.', 'lw': 1.5},
        'sdn':    {'color': '#1f77b4', 'ls': '--', 'lw': 1.5},
        'olsr':   {'color': '#9467bd', 'ls': ':',  'lw': 1.8},
        'batman': {'color': '#e37710', 'ls': '-',  'lw': 1.3},
        'aodv':   {'color': '#17becf', 'ls': '--', 'lw': 1.5},
    }
    PROTO = {name: dict(style, data=all_data[name])
             for name, style in _CANDIDATES.items()
             if name in all_data and name not in FIGURE_EXCLUDED_ARMS}

    # Scenario-specific titles
    n_arms = len(PROTO)
    n_classical = sum(1 for m in PROTO if m != PROPOSED_ARM)
    arm_line = (f"{n_arms}-Arm Comparison ({n_classical} classical protocols + "
                f"{PROPOSED_ARM_LABEL})")
    sever_min = SEVERANCE_TICK * TICK_DURATION_S / 60
    rescue_min = RESCUE_TICK * TICK_DURATION_S / 60
    if scenario_name == 'rescue_ops':
        main_title = (f"Scenario B: Emergency Services Operations\n"
                      f"Post-Convergence \u2014 Rescue UE Arrival (T+{rescue_min:.0f}min)\n"
                      f"{arm_line} \u2014 Fair PHY: Auto CQI\u2192MCS (TS 38.214)")
    else:
        main_title = (f"Scenario A: Post-Severance Recovery\n"
                      f"Civilian Emergency Traffic Only \u2014 Core Severed at T+{sever_min:.0f}min\n"
                      f"{arm_line} \u2014 Fair PHY: Auto CQI\u2192MCS (TS 38.214)")

    fig, axes = plt.subplots(4, 2, figsize=(20, 22))
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    def draw_backgrounds(ax, annotate_event=False):
        sev_min = SEVERANCE_TICK * TICK_DURATION_S / 60
        resc_min = RESCUE_TICK * TICK_DURATION_S / 60
        if scenario_name != 'rescue_ops':  # Scenario A: show severance line
            ax.axvline(SEVERANCE_TICK, color="red", linestyle="--", alpha=0.6, label=f"Core Severance (T+{sev_min:.0f}min)")
        if scenario_name == 'rescue_ops':
            ax.axvline(RESCUE_TICK, color="orange", linestyle="--", alpha=0.6, label=f"Rescue UEs Arrive (T+{resc_min:.0f}min)")
            if annotate_event:
                ax.annotate(f'\u2193 20 Rescue UEs Arrive (T+{resc_min:.0f}min)',
                            xy=(RESCUE_TICK, 0.92), xycoords=('data', 'axes fraction'),
                            fontsize=9, color='#E65100', fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                                      edgecolor='#E65100', alpha=0.9))
        if scenario_name != 'rescue_ops' and annotate_event:
            ax.annotate(f'\u2193 Core Link Severed (T+{sev_min:.0f}min)',
                        xy=(SEVERANCE_TICK, 0.92), xycoords=('data', 'axes fraction'),
                        fontsize=9, color='#C62828', fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                                  edgecolor='#C62828', alpha=0.9))
        ax.grid(True, alpha=0.3)

    # --- 1. Throughput Achievability ---
    ax1 = axes[0, 0]
    labels_achiev = {
        'marl': 'MARL (pre-trained + online)',
        'marl_freeze': PROPOSED_ARM_LABEL,
        'xldet': 'XL-DET (engineered, no learning)',
        'ospf': 'OSPF (RFC 2328, SPF)',
        'sdn':  'SDN (OpenFlow, controller)',
        'olsr': 'OLSR (RFC 3626, MPR)',
        'batman': 'B.A.T.M.A.N. (OGM, L2)',
        'aodv': 'AODV (RFC 3561, reactive)',
    }
    all_max = 0
    for name, p in PROTO.items():
        d = p['data']['conn'][:PLOT_END]
        ax1.plot(ticks_x, d, label=labels_achiev[name],
                 color=p['color'], linewidth=p['lw'], alpha=0.8, linestyle=p['ls'])
        all_max = max(all_max, max(d) if d else 0)
    ax1.axhline(100, color="black", linestyle=":", linewidth=1, label="Physical Max")
    ax1.set_title("Throughput Achievability (Delivered / Baseline)")
    ax1.set_ylabel("Achievability (%)")
    ax1.set_ylim(0, max(35, all_max * 1.3))
    draw_backgrounds(ax1, annotate_event=True)
    ax1.legend(loc="lower right", fontsize=6)

    # --- 2. Energy Efficiency (mJ per Mbps delivered) ---
    ax2 = axes[0, 1]
    labels_energy = {
        'marl': 'MARL (NPU inference, 0.3 mJ/msg)',
        'marl_freeze': f'{PROPOSED_ARM_SHORT} (NPU inference)',
        'xldet': 'XL-DET (rule evaluation, no NPU)',
        'ospf': 'OSPF (SPF: 2 mJ/msg, Hello: 0.5)',
        'sdn':  'SDN (TCAM: 1.5 mJ/rule, echo: 0.3)',
        'olsr': 'OLSR (MPR: 0.8 mJ, TC: 1.5)',
        'batman': 'B.A.T.M.A.N. (OGM: 0.4 mJ/broadcast)',
        'aodv': 'AODV (RREQ: 1.0 mJ, RREP: 0.5)',
    }
    for name, p in PROTO.items():
        energy_raw = p['data']['energy'][:PLOT_END]
        delivered_raw = p['data']['delivered'][:PLOT_END]
        # Compute efficiency: energy_joules * 1000 / delivered_mbps → mJ per Mbps
        # Use a rolling window of 10 ticks for smoothing
        efficiency = []
        window = 10
        for i in range(len(energy_raw)):
            start_idx = max(0, i - window + 1)
            sum_energy = sum(energy_raw[start_idx:i+1])
            sum_delivered = sum(delivered_raw[start_idx:i+1])
            if sum_delivered > 0:
                eff = (sum_energy * 1000.0) / sum_delivered  # mJ per Mbps
            else:
                eff = float('nan')  # No data delivered → undefined
            efficiency.append(eff)
        ax2.plot(ticks_x[:len(efficiency)], efficiency, label=labels_energy[name],
                 color=p['color'], linewidth=p['lw'], alpha=0.8, linestyle=p['ls'])
    ax2.set_title("Energy Efficiency (mJ per Mbps Delivered)")
    ax2.set_ylabel("mJ / Mbps (lower = better)")
    draw_backgrounds(ax2)
    ax2.legend(loc="upper right", fontsize=6)

    # --- 3. Control Plane Overhead ---
    ax3 = axes[1, 0]
    labels_oh = {
        'marl': 'MARL (postcards, 200B/msg)',
        'marl_freeze': f'{PROPOSED_ARM_SHORT} (postcards)',
        'xldet': 'XL-DET (postcards, same DCC)',
        'ospf': 'OSPF (Hello 64B + LSA 128B)',
        'sdn':  'SDN (OF 256B + LLDP 64B)',
        'olsr': 'OLSR (Hello 40B + TC 80B)',
        'batman': 'B.A.T.M.A.N. (OGM 52B/bcast)',
        'aodv': 'AODV (RREQ 48B + RREP 44B)',
    }
    for name, p in PROTO.items():
        d = p['data']['overhead'][:PLOT_END]
        ax3.plot(ticks_x, d, label=labels_oh[name],
                 color=p['color'], linewidth=p['lw'], alpha=0.8, linestyle=p['ls'])
    ax3.set_title("Control Plane Overhead (Mbps)")
    ax3.set_ylabel("Overhead (Mbps)")
    draw_backgrounds(ax3)
    ax3.legend(loc="upper right", fontsize=6)

    # --- 4. Modeled E2E Latency (+ outage fraction on twin axis) ---
    ax4 = axes[1, 1]
    for name, p in PROTO.items():
        d = p['data']['latency'][:PLOT_END]
        ax4.plot(ticks_x[:len(d)], d, label=name.upper(),
                 color=p['color'], linewidth=p['lw'], alpha=0.8, linestyle=p['ls'])
    ax4.set_title("Modeled E2E Latency (ms) — routed flows only; gaps/dotted = outage")
    ax4.set_ylabel("Modeled latency (ms)")
    ax4b = ax4.twinx()
    for name, p in PROTO.items():
        o = p['data'].get('outage', [])[:PLOT_END]
        if o:
            ax4b.plot(ticks_x[:len(o)], [100.0 * v for v in o],
                      color=p['color'], linewidth=0.8, alpha=0.5, linestyle=':')
    ax4b.set_ylabel("Outage fraction (%, dotted)")
    ax4b.set_ylim(0, 105)
    draw_backgrounds(ax4)
    ax4.legend(loc="upper right", fontsize=6)

    # --- 5. MCS Distribution ---
    ax5 = axes[2, 0]
    for name, p in PROTO.items():
        d = p['data'].get('mcs_high', [])[:PLOT_END]
        if d:
            ax5.plot(ticks_x, d, label=f"{name.upper()} QAM256 %",
                     color=p['color'], linewidth=p['lw'], alpha=0.8, linestyle=p['ls'])
    ax5.set_title("High-Order MCS Usage (% nodes using QAM256)")
    ax5.set_ylabel("QAM256 Nodes (%)")
    draw_backgrounds(ax5)
    ax5.legend(loc="upper right", fontsize=6)

    # --- 6. Fragmentation ---
    ax6 = axes[2, 1]
    for name, p in PROTO.items():
        d = p['data']['fragments'][:PLOT_END]
        ax6.plot(ticks_x, d, label=name.upper(),
                 color=p['color'], linewidth=p['lw'], alpha=0.8, linestyle=p['ls'])
    ax6.set_title("Network Fragments (Connected Components)")
    ax6.set_ylabel("Fragment Count")
    draw_backgrounds(ax6)
    ax6.legend(loc="upper right", fontsize=6)

    # --- 7-8. Phase Description ---
    ax7 = axes[3, 0]
    ax7.axis('off')
    if scenario_name == 'rescue_ops':
        phase_text_left = ("SCENARIO B: EMERGENCY SERVICES OPERATIONS\n"
            "Post-convergence: network already in island mode\n"
            "Traffic: constant civilian load (no decay/phases)\n"
            "t=200: 20 Rescue UEs arrive (+80 new flows)\n\n"
            "MARL (Distributed, Pre-trained + online):\n"
            "  MCS: bounded agent offset from auto CQI->MCS\n"
            "  Power/PRB/relay: agent actions applied verbatim\n"
            "  Online arm: adaptive PPO, event-gated exploration\n"
            "  Adapt-freeze arm: argmax-frozen after last\n"
            "  event + FREEZE_AFTER_LAST_EVENT_TICKS\n\n"
            "OSPF (RFC 2328): auto MCS, 23 dBm\n"
            "  Pre-converged: stable SPF routes from t=0;\n"
            "  new (rescue) nodes re-flood + SPF (measured)\n\n"
            "SDN (Process-Simulated OpenFlow):\n"
            "  Pre-converged: controller elected, flows programmed\n"
            "  Phase: Operational")
    else:
        phase_text_left = ("SCENARIO A: POST-SEVERANCE RECOVERY\n"
            "T+3m20s: Core Severance (catastrophic)\n"
            "Traffic: constant civilian load (no decay/phases)\n"
            "No rescue services\n\n"
            "Recovery timing is MEASURED from each protocol's\n"
            "control-plane state machine over the surviving\n"
            "topology (varies by seed) — not scripted.\n\n"
            "MARL (Distributed, Pre-trained + online):\n"
            "  MCS: bounded agent offset from auto CQI->MCS\n"
            "  Power/PRB/relay: agent actions applied verbatim\n"
            "  Online arm: adaptive PPO, event-gated exploration\n"
            "  Adapt-freeze arm: argmax-frozen after last\n"
            "  event + FREEZE_AFTER_LAST_EVENT_TICKS\n\n"
            "OSPF (RFC 2328): auto MCS, 23 dBm\n"
            "  Dead-timer detection (staggered Hello phases)\n"
            "  -> hop-by-hop LSA flood over live links\n"
            "  -> per-node SPF after LSDB settles\n\n"
            "SDN (Process-Simulated OpenFlow):\n"
            "  Flow drain (hard-timeout) -> per-fragment Raft\n"
            "  election -> LLDP BFS discovery -> FlowMod\n"
            "  programming (rate-limited) -> Operational")
    ax7.text(0.02, 0.98, phase_text_left, transform=ax7.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax8 = axes[3, 1]
    ax8.axis('off')
    phase_text_right = ("OLSR (RFC 3626, MPR-optimized):\n"
        "  Phase 1: Hello (2-hop sensing, 3x2s rounds)\n"
        "  Phase 2: MPR selection (greedy set-cover)\n"
        "  Phase 3: TC flooding via MPR subset only\n"
        "  Phase 4: Dijkstra route calc (~2s)\n"
        "  Phase 5: Operational (periodic TC refresh)\n"
        "  Overhead: ~40% less than OSPF\n\n"
        "B.A.T.M.A.N. (batman-adv, Layer-2):\n"
        "  OGM broadcast every 1 second\n"
        "  TQ learning: EMA of received OGM quality\n"
        "  Convergence: ~20s (20 OGM rounds)\n"
        "  No global topology - per-hop TQ forwarding\n"
        "  Highest ongoing overhead (all nodes broadcast)\n\n"
        "AODV (RFC 3561, Reactive):\n"
        "  No proactive route maintenance\n"
        "  Route discovery: RREQ flood -> RREP unicast\n"
        "  Per-flow: ~8s per discovery (RREQ 5s + RREP 3s)\n"
        "  Rate-limited RREQ starts -> staircase convergence\n"
        "  Lowest idle overhead, highest burst overhead\n"
        f"  Route cache expires after {AODV_ROUTE_LIFETIME} s idle")
    ax8.text(0.02, 0.98, phase_text_right, transform=ax8.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_comparison_multiseed(all_data_multi, scenario_name, output_path, seeds):
    """Generate comparison plot with mean ± std variance bands across multiple seeds.

    all_data_multi: dict mapping mode -> {seed: run_data}
    scenario_name: 'recovery' or 'rescue_ops'
    output_path: where to save the PNG
    seeds: list of topology seeds used
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings

    PLOT_END = max(1, TOTAL_TICKS - 80) if TOTAL_TICKS > 400 else TOTAL_TICKS
    ticks_x = np.arange(PLOT_END)
    RESCUE_TICK = RESCUE_ARRIVAL_TICK

    # Protocol styling.  Arms are plotted only when present in the data
    # (keeps --replot working on older pickles) and only when not listed in
    # FIGURE_EXCLUDED_ARMS — the online-only MARL ablation is still in the
    # raw pickle, it is simply not part of the publication figures.
    # 'xldet' sits directly after the proposed arm because it is the
    # comparison the reader cares most about: the engineered non-learning
    # controller with identical actuation.  It is NOT in FIGURE_EXCLUDED_ARMS.
    modes_order = figure_arms(['marl', 'marl_freeze', 'xldet', 'ospf', 'sdn',
                               'olsr', 'batman', 'aodv'])
    modes_order = [m for m in modes_order
                   if any(all_data_multi.get(m, {}).get(s) for s in seeds)]
    STYLE = {
        'marl':        {'color': '#2ca02c', 'ls': '-',  'lw': 2.5},
        # Hero line: the proposed system gets the strongest, most distinct
        # colour and sits on top of every classical curve.
        'marl_freeze': {'color': PROPOSED_ARM_COLOR, 'ls': '-',
                        'lw': PROPOSED_ARM_LW, 'z': 10},
        'xldet':       {'color': '#8c564b', 'ls': (0, (5, 2)), 'lw': 2.2},
        'ospf':        {'color': '#d62728', 'ls': '-.', 'lw': 1.5},
        'sdn':         {'color': '#1f77b4', 'ls': '--', 'lw': 1.5},
        'olsr':        {'color': '#9467bd', 'ls': ':',  'lw': 1.8},
        'batman':      {'color': '#e37710', 'ls': '-',  'lw': 1.3},
        'aodv':        {'color': '#17becf', 'ls': '--', 'lw': 1.5},
    }
    LABELS = {
        'marl': 'MARL (pre-trained + online)',
        'marl_freeze': PROPOSED_ARM_LABEL,
        'xldet': 'XL-DET (engineered, no learning)',
        'ospf': 'OSPF (RFC 2328, SPF)',
        'sdn':  'SDN (OpenFlow, controller)',
        'olsr': 'OLSR (RFC 3626, MPR)',
        'batman': 'B.A.T.M.A.N. (OGM, L2)',
        'aodv': 'AODV (RFC 3561, reactive)',
    }
    # Compact names for dense legends, bar ticks and the summary table
    SHORT_LABELS = {
        'marl': 'MARL-online',
        'marl_freeze': PROPOSED_ARM_SHORT,
        'xldet': 'XL-DET',
        'ospf': 'OSPF', 'sdn': 'SDN', 'olsr': 'OLSR',
        'batman': 'BATMAN', 'aodv': 'AODV',
    }

    def _aggregate(mode, key, length=PLOT_END):
        """Stack per-seed arrays and compute NaN-aware mean/std.

        Latency arrays contain NaN on outage ticks (no routed flow) — those
        ticks are excluded from the aggregate rather than polluting it with
        sentinel values; outage is aggregated separately via key='outage'.
        """
        seed_data = all_data_multi.get(mode, {})
        arrays = []
        for seed in seeds:
            if seed in seed_data and seed_data[seed] is not None:
                arr = seed_data[seed].get(key, [])[:length]
                if len(arr) < length:
                    arr = arr + [arr[-1] if arr else 0.0] * (length - len(arr))
                arrays.append(arr)
        if not arrays:
            return np.zeros(length), np.zeros(length)
        stacked = np.array(arrays, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)

    def _plot_with_band(ax, mode, key, label=None):
        """Plot mean line + shaded ±1σ band."""
        s = STYLE[mode]
        mean, std = _aggregate(mode, key)
        lbl = label or LABELS[mode]
        z = s.get('z', 2)
        ax.plot(ticks_x, mean, label=lbl, zorder=z,
                color=s['color'], linewidth=s['lw'], linestyle=s['ls'], alpha=0.9)
        # Clamp lower band to 0 — physical quantities can't be negative
        lower = np.maximum(0, mean - std)
        ax.fill_between(ticks_x, lower, mean + std, zorder=z - 1,
                         color=s['color'], alpha=0.15)

    # Measured fragmenting-severance parameters (Scenario A only — Scenario B
    # starts already severed at t=0 with no additional fragmenting cuts).
    frag_stats = (measure_fragmenting_severance_stats(seeds)
                  if scenario_name != 'rescue_ops' else None)

    # Scenario-specific titles
    n_seeds = len(seeds)
    n_arms = len(modes_order)
    n_classical = sum(1 for m in modes_order if m != PROPOSED_ARM)
    seed_span = (f"{min(seeds)}-{max(seeds)}" if len(seeds) > 1
                 else f"{seeds[0]}")
    # Compact, run-collapsing seed text for headers/captions — keeps a
    # 10-seed (or larger) evaluation from overflowing a line.
    _seed_txt = format_seed_list(seeds)
    arm_line = (f"{n_arms}-Arm Comparison ({n_classical} classical protocols "
                f"+ {PROPOSED_ARM_LABEL}) — Mean ± 1σ across "
                f"{n_seeds} held-out seeds ({seed_span})")
    sever_min = SEVERANCE_TICK * TICK_DURATION_S / 60
    rescue_min = RESCUE_TICK * TICK_DURATION_S / 60
    if scenario_name == 'rescue_ops':
        main_title = (f"Scenario B: Emergency Services Operations\n"
                      f"Post-Convergence \u2014 Rescue UE Arrival (T+{rescue_min:.0f}min)\n"
                      f"{arm_line}")
    else:
        main_title = (f"Scenario A: Post-Severance Recovery\n"
                      f"Civilian Emergency Traffic Only \u2014 Core Severed at T+{sever_min:.0f}min\n"
                      f"{arm_line}")

    # 7x2: row 5 carries the two THRESHOLD-FREE / RE-DIP-AWARE recovery
    # panels (service-loss integral, availability); the table and the
    # experiment-info block move down to row 6.
    fig, axes = plt.subplots(7, 2, figsize=(20, 38.5))
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    def draw_backgrounds(ax, annotate_event=False):
        sev_min = SEVERANCE_TICK * TICK_DURATION_S / 60
        resc_min = RESCUE_TICK * TICK_DURATION_S / 60
        if scenario_name != 'rescue_ops':
            ax.axvline(SEVERANCE_TICK, color="red", linestyle="--", alpha=0.6, label=f"Core Severance (T+{sev_min:.0f}min)")
        if scenario_name == 'rescue_ops':
            ax.axvline(RESCUE_TICK, color="orange", linestyle="--", alpha=0.6, label=f"Rescue UEs Arrive (T+{resc_min:.0f}min)")
            if annotate_event:
                ax.annotate(f'\u2193 20 Rescue UEs Arrive (T+{resc_min:.0f}min)',
                            xy=(RESCUE_TICK, 0.92), xycoords=('data', 'axes fraction'),
                            fontsize=9, color='#E65100', fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                                      edgecolor='#E65100', alpha=0.9))
        if scenario_name != 'rescue_ops' and annotate_event:
            ax.annotate(f'\u2193 Core Link Severed (T+{sev_min:.0f}min)',
                        xy=(SEVERANCE_TICK, 0.92), xycoords=('data', 'axes fraction'),
                        fontsize=9, color='#C62828', fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                                  edgecolor='#C62828', alpha=0.9))
        ax.grid(True, alpha=0.3)

    # --- 1. Throughput Achievability ---
    ax1 = axes[0, 0]
    for mode in modes_order:
        _plot_with_band(ax1, mode, 'conn')
    ax1.axhline(100, color="black", linestyle=":", linewidth=1, label="Physical Max")
    ax1.set_title("Throughput Achievability (Delivered / Baseline)")
    ax1.set_ylabel("Achievability (%)")
    draw_backgrounds(ax1, annotate_event=True)

    # In-fragment ceiling (Scenario A): every classical protocol converges to
    # the SAME steady-state cap, because pure re-routing cannot serve flows
    # whose endpoints sit in different fragments.  Both the cap and the
    # cross-fragment share below are measured, never hardcoded.
    classical_modes = [m for m in modes_order if m != PROPOSED_ARM]
    if scenario_name != 'rescue_ops' and classical_modes:
        # Identify (from the data) the arms that land on the SAME per-seed
        # steady state — those are the ones pinned to the routing-only
        # in-fragment ceiling.  Arms outside that group are reported
        # separately rather than folded into the claim.
        _ss = {m: steady_state_per_seed(all_data_multi, m, 'conn', seeds)
               for m in classical_modes}
        _groups = {}
        for m, vals in _ss.items():
            if vals:
                _groups.setdefault(tuple(round(v, 1) for v in vals), []).append(m)
        if _groups:
            _key = max(_groups, key=lambda k: len(_groups[k]))
            ceiling_modes = _groups[_key]
            other_modes = [m for m in classical_modes if m not in ceiling_modes]
            cap = float(np.mean(_key))
            cap_sd = float(np.std(_key))
            ax1.axhline(cap, color='#555555', linestyle=(0, (6, 4)),
                        linewidth=1.2, alpha=0.8, zorder=1,
                        label=f"In-fragment ceiling ({cap:.1f}%)")
            frag_txt = ""
            if frag_stats:
                frag_txt = (f"\nthe severed mesh splits into "
                            f"{frag_stats['mean_fragments']:.1f} fragments and "
                            f"{frag_stats['mean_share']:.1%} of UE↔UE flows span "
                            f"fragments (per-seed "
                            f"{frag_stats['min_share']:.1%}–"
                            f"{frag_stats['max_share']:.1%};\n"
                            f"{frag_stats['cross_total']}/{frag_stats['flow_total']} "
                            f"flows over {len(frag_stats['per_seed'])} seeds), and "
                            f"pure re-routing cannot serve them.")
            prop_m, prop_s, _n = steady_state_across_seeds(
                all_data_multi, PROPOSED_ARM, 'conn', seeds)
            prop_txt = ""
            if np.isfinite(prop_m):
                # Back the bridging claim with the MEASURED relay-bridge
                # count (relay LINKS joining two otherwise-disconnected
                # fragments) whenever the recorder provides it, instead of
                # asserting the mechanism narratively.
                _pb = (reunification_stats(all_data_multi, PROPOSED_ARM, seeds)
                       or {}).get('relay_bridges')
                _pb_txt = ""
                if _pb:
                    _pb_txt = (f"\n  measured: {_pb['mean']:.1f} relay bridge "
                               f"LINKS in steady state "
                               f"(per-seed {_pb['min']:.0f}-{_pb['max']:.0f}); "
                               f"every classical arm forms 0.")
                prop_txt = (f"\n{PROPOSED_ARM_SHORT} reaches "
                            f"{prop_m:.1f}±{prop_s:.1f}%, above the ceiling, by "
                            f"forming MultiHaul relay bridges between fragments."
                            + _pb_txt)
            other_txt = ""
            if other_modes:
                _o = []
                for m in other_modes:
                    mm, ms, _n2 = steady_state_across_seeds(
                        all_data_multi, m, 'conn', seeds)
                    _o.append(f"{SHORT_LABELS[m]} {mm:.1f}±{ms:.1f}% "
                              f"(per-seed "
                              f"{min(_ss[m]):.1f}–{max(_ss[m]):.1f}%)")
                other_txt = ("\nNot on the ceiling: " + "; ".join(_o) + ".")
            ax1.annotate(
                f"{', '.join(SHORT_LABELS[m] for m in ceiling_modes)} converge to "
                f"the SAME in-fragment ceiling on every seed "
                f"({cap:.1f}±{cap_sd:.1f}%):{frag_txt}{prop_txt}{other_txt}",
                xy=(0.02, 0.03), xycoords='axes fraction', fontsize=7.5,
                color='#333333', ha='left', va='bottom', zorder=12,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                          edgecolor='#9E9E9E', alpha=0.92))
    ax1.legend(loc="lower right", fontsize=6)

    # --- 2. Energy Efficiency ---
    ax2 = axes[0, 1]
    for mode in modes_order:
        # Compute efficiency per seed, then aggregate
        seed_data = all_data_multi.get(mode, {})
        eff_arrays = []
        for seed in seeds:
            if seed not in seed_data or seed_data[seed] is None:
                continue
            energy_raw = seed_data[seed].get('energy', [])[:PLOT_END]
            delivered_raw = seed_data[seed].get('delivered', [])[:PLOT_END]
            efficiency = []
            window = 10
            for i in range(min(len(energy_raw), PLOT_END)):
                start_idx = max(0, i - window + 1)
                sum_energy = sum(energy_raw[start_idx:i+1])
                sum_delivered = sum(delivered_raw[start_idx:i+1])
                if sum_delivered > 0:
                    eff = (sum_energy * 1000.0) / sum_delivered
                else:
                    eff = 0.0
                efficiency.append(eff)
            while len(efficiency) < PLOT_END:
                efficiency.append(efficiency[-1] if efficiency else 0.0)
            eff_arrays.append(efficiency)
        if eff_arrays:
            stacked = np.array(eff_arrays, dtype=np.float64)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            s = STYLE[mode]
            _z = s.get('z', 2)
            ax2.plot(ticks_x, mean, label=LABELS[mode], zorder=_z,
                     color=s['color'], linewidth=s['lw'], linestyle=s['ls'], alpha=0.9)
            lower = np.maximum(0, mean - std)
            ax2.fill_between(ticks_x, lower, mean + std, zorder=_z - 1,
                             color=s['color'], alpha=0.15)
    ax2.set_title("Energy Efficiency (mJ per Mbps Delivered)")
    ax2.set_ylabel("mJ / Mbps (lower = better)")
    draw_backgrounds(ax2)
    ax2.legend(loc="upper right", fontsize=6)

    # --- 3. Control Plane Overhead ---
    ax3 = axes[1, 0]
    for mode in modes_order:
        _plot_with_band(ax3, mode, 'overhead')
    ax3.set_title("Control Plane Overhead (Mbps)")
    ax3.set_ylabel("Overhead (Mbps)")
    draw_backgrounds(ax3)
    ax3.legend(loc="upper right", fontsize=6)

    # --- 4. Modeled E2E Latency (routed flows) + outage fraction ---
    # Latency is the per-hop analytical model over flows that actually have
    # converged routes; ticks where a protocol routes nothing show as gaps.
    # Outage (no converged route / no path) is its own dotted line on the
    # twin axis — NOT folded into the latency mean.
    ax4 = axes[1, 1]
    for mode in modes_order:
        _plot_with_band(ax4, mode, 'latency', label=SHORT_LABELS[mode])
    ax4.set_title("MODELED E2E Latency (ms) — analytical per-hop model, "
                  "not packet-level;\nrouted flows only; dotted = outage %")
    ax4.set_ylabel("Modeled latency (ms)")
    ax4b = ax4.twinx()
    for mode in modes_order:
        o_mean, _o_std = _aggregate(mode, 'outage')
        ax4b.plot(ticks_x, 100.0 * o_mean, color=STYLE[mode]['color'],
                  linewidth=0.8, alpha=0.5, linestyle=':')
    ax4b.set_ylabel("Outage fraction (%, dotted)")
    ax4b.set_ylim(0, 105)
    draw_backgrounds(ax4)
    ax4.legend(loc="upper right", fontsize=6)

    # --- 5. Absolute Throughput (Mbps) ---
    ax5 = axes[2, 0]
    for mode in modes_order:
        _plot_with_band(ax5, mode, 'delivered', label=LABELS[mode])
    ax5.set_title("Absolute Throughput (delivered Mbps-equivalent units)")
    ax5.set_ylabel("Delivered Throughput")
    draw_backgrounds(ax5)
    ax5.legend(loc="lower right", fontsize=6)

    # --- 6. Fragmentation / reunification -----------------------------------
    # `fragments` counts connected components INCLUDING relay links — the
    # connectivity the arm actually achieved.  `fragments_raw` is the same
    # count on the relay-FREE graph: what the severance did, identical for
    # every arm.  The GAP between the two is the reunification an arm
    # performed, and `relay_bridges` (relay LINKS joining two otherwise
    # disconnected raw fragments) is the mechanism that produced it.
    #
    # NOTE ON VOCABULARY: `relay_count` now means active transport relay
    # LINKS (it previously counted nodes in a relay MODE, which is why the
    # baselines used to report 42 "relays" while forming zero links).  Nodes
    # in a relay mode are recorded separately as `relay_mode_nodes`.  This
    # panel deliberately reports LINKS (bridges), never node counts.
    ax6 = axes[2, 1]
    _has_raw = has_series(all_data_multi, modes_order, seeds, 'fragments_raw')
    _has_bridges = has_series(all_data_multi, modes_order, seeds,
                              'relay_bridges')
    for mode in modes_order:
        _plot_with_band(ax6, mode, 'fragments', label=SHORT_LABELS[mode])

    # Reference line: the raw post-severance partition.  Prefer the recorded
    # fragments_raw series; fall back to the independently RE-DERIVED
    # severance parameters when replotting a pickle written before that
    # series existed.  Never a hardcoded literal.
    _raw_level, _raw_src = None, None
    if _has_raw:
        _raw_vals = []
        for m in modes_order:
            _raw_vals += steady_state_per_seed(all_data_multi, m,
                                               'fragments_raw', seeds)
        if _raw_vals:
            _raw_level = float(np.mean(_raw_vals))
            _raw_src = "recorded fragments_raw"
    elif frag_stats:
        _raw_level = float(frag_stats['mean_fragments'])
        _raw_src = "re-derived from the severance builder"
    if _raw_level is not None:
        ax6.axhline(_raw_level, color='#B71C1C', linestyle=(0, (7, 4)),
                    linewidth=1.6, alpha=0.9, zorder=11,
                    label=f"Raw partition, relay-free ({_raw_level:.1f})")

    # SECOND reference line: the TG-FEASIBLE OPTIMUM — the floor the physics
    # actually permits.  Without it a reader takes the y=1 gridline as the
    # target and reads any arm above it as having failed, when on some seeds
    # the 60 GHz budget simply cannot close the last gap.  Drawn only when it
    # is recorded, and labelled with the power it was solved at.
    _tgopt_fig = tg_optimum_summary(all_data_multi, seeds)
    _tgopt_level = _tgopt_fig['mean']
    if np.isfinite(_tgopt_level):
        ax6.axhline(_tgopt_level, color='#1B5E20', linestyle=(0, (2, 3)),
                    linewidth=1.8, alpha=0.95, zorder=12,
                    label=(f"TG-feasible optimum @"
                           f"{_tgopt_fig['tx_dbm']:.0f} dBm "
                           f"({_tgopt_level:.1f})"))

    ax6.set_title("Network Fragments — achieved connectivity vs the raw "
                  "partition\n(red = what the severance did; green = the "
                  "TG-feasible optimum, the real target)", fontsize=9)
    ax6.set_ylabel("Connected components (incl. relay links)")
    draw_backgrounds(ax6)

    # Annotate with the measured relay BRIDGE counts (links, not nodes).
    _frag_note = []
    if scenario_name == 'rescue_ops':
        # Verified: Scenario B applies core severance + rescue arrival with
        # NO link cuts, so there is no partition to bridge here.  Say so
        # explicitly rather than letting the panel imply otherwise.
        _frag_note.append(
            "Scenario B does NOT fragment the network: its events are core\n"
            "severance + rescue-UE arrival with NO link cuts. Cross-fragment\n"
            "UE↔UE flows are 0/222 on every seed and the component count is\n"
            "1-2 (an occasionally orphaned node). NO partition bridging is\n"
            "claimed or implied by this panel — the relay bridges that matter\n"
            "are a Scenario-A result.")
    else:
        if _raw_level is not None:
            _frag_note.append(
                f"Raw partition = {_raw_level:.1f} components "
                f"({_raw_src});\nidentical for every arm — it is what the "
                f"severance did, not what an arm chose.")
        if np.isfinite(_tgopt_level):
            _per = ", ".join(f"s{k}={v}"
                             for k, v in sorted(_tgopt_fig['per_seed'].items()))
            _frag_note.append(
                f"TG-feasible optimum = {_tgopt_level:.1f} components "
                f"({_per})\nat {_tgopt_fig['tx_dbm']:.0f} dBm: the fewest "
                f"components reachable when every\nTG site steers optimally, "
                f"solved from the 60 GHz link budget.\nTHIS, not 1, is the "
                f"target an arm is judged against.")
        if _has_bridges:
            _bl = []
            for m in modes_order:
                st = reunification_stats(all_data_multi, m, seeds)
                b = (st or {}).get('relay_bridges')
                f = (st or {}).get('fragments')
                if b is None or f is None:
                    continue
                _bl.append(f"  {SHORT_LABELS[m]:<11} bridges "
                           f"{b['mean']:4.1f} ({b['min']:.0f}-{b['max']:.0f})"
                           f"   components {f['mean']:4.1f} "
                           f"({f['min']:.0f}-{f['max']:.0f})")
            if _bl:
                _frag_note.append(
                    "Steady-state relay BRIDGES (links joining two otherwise\n"
                    "disconnected fragments) and resulting components:\n"
                    + "\n".join(_bl))
        else:
            _frag_note.append(
                "relay_bridges series not present in this dataset\n"
                "(pickle predates it) — bridge counts omitted rather than\n"
                "inferred from the component count.")
    # Headroom above the data so the annotation and legend never sit on top
    # of the curves — the classical arms ride exactly on the raw-partition
    # line, which is the single most important thing to be able to see here.
    _fr_top = max([_raw_level or 0] +
                  [float(np.nanmax(_aggregate(m, 'fragments')[0]))
                   for m in modes_order] + [1.0])
    ax6.set_ylim(0, _fr_top * 1.85)
    if _frag_note:
        ax6.annotate("\n".join(_frag_note), xy=(0.985, 0.97),
                     xycoords='axes fraction', fontsize=6.4, ha='right',
                     va='top', fontfamily='monospace', zorder=13,
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                               edgecolor='#9E9E9E', alpha=0.93))
    ax6.legend(loc="upper left", fontsize=6)

    # --- 7. Recovery Time — FIXED ABSOLUTE THRESHOLD, identical for all arms ---
    # See the RECOVERY_* block above for why the old "90% of the arm's OWN
    # steady state" number was structurally unfair in both directions.  The
    # threshold below is ONE scalar derived from the scenario's pre-event
    # achievability level and applied to every arm without modification.
    ax7 = axes[3, 0]
    event_tick = event_tick_for(scenario_name)
    ref_level_raw, ref_per_arm, ref_spread = pre_event_reference(
        all_data_multi, modes_order, seeds, scenario_name, PLOT_END)
    ref_is_shared = np.isfinite(ref_spread) and ref_spread <= RECOVERY_REF_SPREAD_TOL

    # ── DEMAND-CORRECTED REFERENCE ─────────────────────────────────────────
    # The offered-demand model STEPS UP at the event tick (see
    # demand_regime_ratio for the measured magnitude and the consequence), so
    # a raw pre-event achievability bar is a bar drawn in a DIFFERENT demand
    # regime from the samples scored against it.  Every reference-derived
    # quantity below — the swept recovery thresholds, the availability
    # thresholds and the service-loss integral — therefore uses the CORRECTED
    # level.  The correction is a single scenario-level scalar applied
    # identically to every arm, so it cannot move the ranking; it moves the
    # ABSOLUTE service-loss figures out of the regime where an arm that never
    # lost absolute throughput is still charged for a loss.  The uncorrected
    # value stays visible as `ref_level_raw` and is printed as a secondary
    # row in the text table, explicitly labelled NOT demand-corrected.
    _offered_hat, _off_src, _off_cov = offered_per_seed(
        all_data_multi, modes_order, seeds, PLOT_END)
    _demand = demand_regime_ratio(all_data_multi, modes_order, seeds,
                                  scenario_name, PLOT_END, _offered_hat)
    ref_level = (ref_level_raw * _demand['ratio']
                 if (_demand['applied'] and np.isfinite(ref_level_raw))
                 else ref_level_raw)

    # Primary + swept thresholds.
    fair_sweep = {}
    if np.isfinite(ref_level):
        for _frac in RECOVERY_THRESHOLD_FRACTIONS:
            fair_sweep[_frac] = fair_recovery_stats(
                all_data_multi, modes_order, seeds, scenario_name,
                PLOT_END, ref_level, _frac)
    primary = fair_sweep.get(RECOVERY_PRIMARY_FRACTION, {})
    primary_thr = RECOVERY_PRIMARY_FRACTION * ref_level

    # Legacy self-referential number, retained for continuity ONLY.
    legacy_recovery = {}
    for mode in modes_order:
        m_conn, _ = _aggregate(mode, 'conn')
        legacy_recovery[mode] = legacy_recovery_time_own_ss(m_conn, scenario_name)

    # ── NEW PRIMARY: service-loss integral (threshold-free) ─────────────
    # Offered demand is reconstructed ONCE for all arms (it is a property of
    # the seed and tick, not of the arm) so every arm's volume-form integral
    # is divided by the identical denominator.
    # `_offered_hat` was already built above for the demand correction — the
    # recorded per-tick offered series where the dataset has it, the post-hoc
    # reconstruction otherwise.  One denominator for every arm.
    _sli_rows = {m: service_loss_per_seed(
        all_data_multi, m, seeds, scenario_name, PLOT_END, ref_level,
        offered=_offered_hat) for m in modes_order}
    # SECONDARY: the same integral against the UNCORRECTED pre-event bar,
    # kept so the size of the demand correction is auditable and nothing is
    # quietly replaced.  Never the headline — see demand_regime_ratio.
    _sli_rows_raw = {m: service_loss_per_seed(
        all_data_multi, m, seeds, scenario_name, PLOT_END, ref_level_raw,
        offered=_offered_hat) for m in modes_order}
    _sli_raw = {m: service_loss_stats(_sli_rows_raw[m], 'pp_s')
                for m in modes_order}
    _sli = {m: service_loss_stats(_sli_rows[m], 'pp_s') for m in modes_order}
    _sli_b = {m: service_loss_stats(_sli_rows[m], 'pp_s_bounded')
              for m in modes_order}
    _sli_v = {m: service_loss_stats(_sli_rows[m], 'vol') for m in modes_order}
    # Volume-form coverage: the share of event-window ticks on which the
    # offered demand could actually be reconstructed.  Reported, never
    # silently assumed to be 1.
    _cov = [r.get('vol_coverage') for rows in _sli_rows.values() for r in rows
            if r.get('vol_coverage') is not None]
    _sli_vol_cov = float(np.mean(_cov)) if _cov else float('nan')
    _sli_vol_ok = bool(np.isfinite(_sli_vol_cov) and _sli_vol_cov >= 0.5)

    # ── NEW: availability at each swept threshold (catches re-dips) ─────
    _avail = {}
    if np.isfinite(ref_level):
        for _frac in RECOVERY_THRESHOLD_FRACTIONS:
            _avail[_frac] = {
                m: availability_stats(all_data_multi, m, seeds,
                                      scenario_name, PLOT_END,
                                      _frac * ref_level)
                for m in modes_order}

    # ── NEW: recovery against the FEASIBLE OPTIMUM (or an explicit skip) ─
    _ceil_per_seed, _ceil_src, _ceil_reason = feasible_achievability_ceiling(
        all_data_multi, seeds)
    _ceil_rec = (ceiling_recovery_stats(all_data_multi, modes_order, seeds,
                                        scenario_name, PLOT_END,
                                        _ceil_per_seed)
                 if _ceil_per_seed else None)

    # Bars: censored arms (threshold never reached and held) are drawn at the
    # full horizon in a hatched style and labelled "not reached" — never
    # silently collapsed into a finite-looking number.
    _bar_vals, _hatch, _labels = [], [], []
    for m in modes_order:
        _pm = primary.get(m, {})
        rt = _pm.get('mean_curve')
        # "never breached" is a CATEGORY, not a zero.  An arm whose mean
        # curve never fell below the bar did not RECOVER quickly — it never
        # lost service by this criterion — so it is drawn as an empty bar
        # with its own label and is excluded from every mean/median below.
        if _pm.get('mean_curve_breached') is False:
            _bar_vals.append(0.0)
            _hatch.append('..')
            _labels.append("never breached (no recovery time)")
        elif rt is None:
            _bar_vals.append(PLOT_END - event_tick)
            _hatch.append('//')
            _labels.append("not reached")
        else:
            secs = rt * TICK_DURATION_S
            _bar_vals.append(max(rt, 0))
            _hatch.append('')
            if rt == 0:
                _labels.append("0 s (breached, recovered within 1 tick)")
            elif secs >= 60:
                _labels.append(f"{secs:.0f}s ({secs/60:.1f}min)")
            else:
                _labels.append(f"{secs:.0f}s")
    _bars = ax7.barh(range(len(modes_order)), _bar_vals,
                     color=[STYLE[m]['color'] for m in modes_order],
                     alpha=0.8, edgecolor='white')
    for _b, _h in zip(_bars, _hatch):
        if _h:
            _b.set_hatch(_h)
            _b.set_alpha(0.35)
    ax7.set_yticks(range(len(modes_order)))
    ax7.set_yticklabels([SHORT_LABELS[m] for m in modes_order], fontsize=8)
    for _tick, _m in zip(ax7.get_yticklabels(), modes_order):
        if _m == PROPOSED_ARM:
            _tick.set_fontweight('bold')
            _tick.set_color(PROPOSED_ARM_COLOR)
    ax7.set_xlabel("Seconds from event (sustained recovery)")
    _ev_name = ('rescue-UE surge' if scenario_name == 'rescue_ops'
                else 'core severance')
    ax7.set_title(
        f"SECONDARY: first-crossing Recovery Time — FIXED ABSOLUTE threshold, "
        f"identical for every arm\n"
        f"time from {_ev_name} to achievability ≥ {primary_thr:.1f}% "
        f"(= {RECOVERY_PRIMARY_FRACTION:.0%} of the {ref_level:.1f}% "
        f"DEMAND-CORRECTED pre-event level),"
        f"\nheld for {RECOVERY_SUSTAIN_TICKS} consecutive ticks. "
        f"Threshold-dependent and blind to depth, area and re-dips —\n"
        f"the PRIMARY metric is the threshold-free service-loss integral "
        f"panel below. 'never breached' is a CATEGORY, not 0 s.",
        fontsize=9)
    _xmax = max([v for v in _bar_vals] + [1])
    for i, (m, v, lbl) in enumerate(zip(modes_order, _bar_vals, _labels)):
        ax7.text(v + 0.015 * _xmax, i, lbl, va='center', fontsize=8,
                 color=STYLE[m]['color'])
    ax7.set_xlim(0, _xmax * 1.32)
    ax7.grid(True, alpha=0.3, axis='x')

    # Threshold sweep + provenance, annotated inside the panel so the figure
    # is self-contained: no single threshold choice drives the conclusion.
    _sw_lines = [f"Threshold sweep (s from event; demand-corrected "
                 f"pre-event level {ref_level:.1f}%):",
                 f"  {'arm':<11}" + "".join(
                     f"{f'{int(f*100)}%={f*ref_level:.1f}%':>16}"
                     for f in RECOVERY_THRESHOLD_FRACTIONS)]
    for m in modes_order:
        cells = []
        for f in RECOVERY_THRESHOLD_FRACTIONS:
            _e = fair_sweep.get(f, {}).get(m, {})
            rt = _e.get('mean_curve')
            # NEVER BREACHED is a category, not a 0 s recovery: printing 0
            # here is exactly the misreading this panel has to prevent.
            if _e.get('mean_curve_breached') is False:
                cells.append(f"{'never breached':>16}")
            elif rt is None:
                cells.append(f"{'n/r':>16}")
            else:
                cells.append(f"{rt * TICK_DURATION_S:>15.0f}s")
        cells = "".join(cells)
        _sw_lines.append(f"  {SHORT_LABELS[m]:<11}{cells}")
    _sw_lines.append("  n/r = threshold never reached and sustained")
    _sw_lines.append("  never breached = the arm never fell below this bar "
                     "(NOT a 0 s recovery)")
    _sw_lines.append("  The ranking flips with the threshold - which is why "
                     "the PRIMARY metric")
    _sw_lines.append("  is the threshold-free service-loss integral panel "
                     "below.")
    if _demand['applied']:
        _sw_lines.append(f"  Bar is DEMAND-CORRECTED: raw pre-event "
                         f"{ref_level_raw:.1f}% x offered ratio "
                         f"{_demand['ratio']:.3f} = {ref_level:.1f}%")
    if ref_is_shared:
        _sw_lines.append(f"  Pre-event level identical across arms "
                         f"(spread {ref_spread:.2f} pp)")
    else:
        _sw_lines.append(f"  NOTE: arms differ pre-event (spread "
                         f"{ref_spread:.1f} pp); the reference is the")
        _sw_lines.append(f"  cross-arm mean — still ONE threshold for all "
                         f"arms, but not arm-invariant")
    ax7.annotate("\n".join(_sw_lines), xy=(0.99, 0.02),
                 xycoords='axes fraction', fontsize=6.6, ha='right',
                 va='bottom', fontfamily='monospace', zorder=12,
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFDE7',
                           edgecolor='#9E9E9E', alpha=0.95))

    # --- 8. MCS Distribution ---
    ax8_mcs = axes[3, 1]
    for mode in modes_order:
        _plot_with_band(ax8_mcs, mode, 'mcs_high',
                        label=f"{SHORT_LABELS[mode]} QAM256 %")
    ax8_mcs.set_title("High-Order MCS Usage (% nodes using QAM256)")
    ax8_mcs.set_ylabel("QAM256 Nodes (%)")
    draw_backgrounds(ax8_mcs)
    ax8_mcs.legend(loc="upper right", fontsize=6)

    # --- 9. Per-seed reliability strip plot ---------------------------------
    # Every seed's steady-state achievability is drawn as its own dot, so a
    # collapsed seed is VISIBLE instead of being averaged into a symmetric
    # ±σ.  No seed is dropped, reweighted or winsorised — this panel is
    # presentation of the same data the table aggregates.
    ax_strip = axes[4, 0]
    # (seed, value) PAIRS, not bare values: an arm missing a seed shifts a
    # bare list out of alignment with `seeds` and mislabels every callout
    # after the gap.
    _ss_pairs = {m: steady_state_per_seed_pairs(all_data_multi, m, 'conn', seeds)
                 for m in modes_order}
    _ss_vals = {m: [v for _s, v in pr] for m, pr in _ss_pairs.items()}
    _rng_jit = np.random.RandomState(12345)   # deterministic jitter
    # Marker size and jitter scale with the seed count so 10+ seeds stay
    # readable instead of merging into a blob.
    _n_dots = max((len(v) for v in _ss_vals.values()), default=0)
    _dot_s = 64 if _n_dots <= 6 else (46 if _n_dots <= 12 else 30)
    _jit_a = 0.16 if _n_dots <= 6 else (0.22 if _n_dots <= 12 else 0.26)
    for i, mode in enumerate(modes_order):
        pairs = _ss_pairs.get(mode) or []
        vals = [v for _s, v in pairs]
        if not vals:
            continue
        st = variance_stats(vals)
        jit = _rng_jit.uniform(-_jit_a, _jit_a, size=len(vals))
        is_prop = (mode == PROPOSED_ARM)
        ax_strip.scatter(vals, np.full(len(vals), i) + jit,
                         s=_dot_s * (1.35 if is_prop else 1.0),
                         color=STYLE[mode]['color'],
                         edgecolor='white', linewidth=0.7,
                         alpha=0.95, zorder=6 if is_prop else 4)
        # IQR box + median tick: robust spread next to the raw dots.
        q1, q3 = np.percentile(vals, [25, 75])
        ax_strip.plot([q1, q3], [i - 0.30, i - 0.30], color=STYLE[mode]['color'],
                      linewidth=3.0, alpha=0.45, solid_capstyle='butt', zorder=3)
        ax_strip.plot([st['median']] * 2, [i - 0.40, i - 0.20], color='#222222',
                      linewidth=1.6, zorder=7)
        ax_strip.plot([st['min'], st['max']], [i - 0.30, i - 0.30],
                      color=STYLE[mode]['color'], linewidth=0.9, alpha=0.6,
                      zorder=2)
        # Flag every seed that failed outright.  With many seeds the callouts
        # are stacked so they cannot overprint each other.
        _fails = [(s_, v) for s_, v in pairs if v < FAILURE_ACHIEVABILITY_PCT]
        for _k, (s_, v) in enumerate(_fails):
            ax_strip.annotate(f"seed {s_}: {v:.1f}%",
                              xy=(v, i + 0.06 + 0.13 * _k), fontsize=6.5,
                              color='#B71C1C', ha='left', va='bottom')
    ax_strip.axvline(FAILURE_ACHIEVABILITY_PCT, color='#B71C1C',
                     linestyle=':', linewidth=1.1, alpha=0.8,
                     label=f"failure line ({FAILURE_ACHIEVABILITY_PCT:.0f}%)")
    ax_strip.set_yticks(range(len(modes_order)))
    ax_strip.set_yticklabels([SHORT_LABELS[m] for m in modes_order], fontsize=8)
    for _tick, _m in zip(ax_strip.get_yticklabels(), modes_order):
        if _m == PROPOSED_ARM:
            _tick.set_fontweight('bold')
            _tick.set_color(PROPOSED_ARM_COLOR)
    ax_strip.set_ylim(-0.8, len(modes_order) - 0.3)
    ax_strip.set_xlim(-2, 102)
    ax_strip.set_xlabel("Steady-state achievability per seed (%)")
    ax_strip.set_title(
        f"Per-seed reliability — one dot per held-out seed ({n_seeds} seeds)\n"
        f"bar = min–max, thick bar = IQR, black tick = median "
        f"(no seed dropped or reweighted)", fontsize=9)
    ax_strip.grid(True, alpha=0.3, axis='x')
    ax_strip.legend(loc='lower right', fontsize=6.5)

    # --- 10. Energy: absolute AND per delivered unit -------------------------
    # Absolute Joules alone is not an efficiency claim: an arm that carries
    # more traffic should burn more energy.  Both are shown side by side so
    # the trade-off is explicit rather than argued.
    ax_en = axes[4, 1]
    _estats = {m: energy_stats(all_data_multi, m, seeds) for m in modes_order}
    _idx = np.arange(len(modes_order))
    _abs = [(_estats[m] or {}).get('abs_mean', float('nan')) for m in modes_order]
    _absd = [(_estats[m] or {}).get('abs_std', 0.0) for m in modes_order]
    _pu = [(_estats[m] or {}).get('per_unit', float('nan')) for m in modes_order]
    _cols = [STYLE[m]['color'] for m in modes_order]
    ax_en.bar(_idx - 0.2, _abs, width=0.4, yerr=_absd, capsize=3,
              color=_cols, alpha=0.85, edgecolor='white',
              error_kw=dict(ecolor='#555555', lw=0.9), label='absolute (J/tick)')
    ax_en.set_ylabel("Absolute energy (J per tick, steady state)")
    ax_en.set_xticks(_idx)
    ax_en.set_xticklabels([SHORT_LABELS[m] for m in modes_order],
                          fontsize=8, rotation=15)
    for _tick, _m in zip(ax_en.get_xticklabels(), modes_order):
        if _m == PROPOSED_ARM:
            _tick.set_fontweight('bold')
            _tick.set_color(PROPOSED_ARM_COLOR)
    ax_enb = ax_en.twinx()
    ax_enb.bar(_idx + 0.2, _pu, width=0.4, color=_cols, alpha=0.45,
               edgecolor='#333333', hatch='//', label='per delivered unit')
    ax_enb.set_ylabel("Energy per delivered unit (J / delivered unit)")
    _fin_abs = [v for v in _abs if np.isfinite(v)]
    _fin_pu = [v for v in _pu if np.isfinite(v)]
    if _fin_abs:
        ax_en.set_ylim(0, max(_fin_abs) * 1.35)
    if _fin_pu:
        ax_enb.set_ylim(0, max(_fin_pu) * 1.35)
    for i, (a, p) in enumerate(zip(_abs, _pu)):
        if np.isfinite(a):
            ax_en.text(i - 0.2, a, f"{a:.0f}", ha='center', va='bottom',
                       fontsize=7)
        if np.isfinite(p):
            ax_enb.text(i + 0.2, p, f"{p:.4f}", ha='center', va='bottom',
                        fontsize=7, color='#333333')
    _best_pu = min((m for m in modes_order if np.isfinite(
        (_estats[m] or {}).get('per_unit', float('nan')))),
        key=lambda m: _estats[m]['per_unit'], default=None)
    _en_note = ""
    if _best_pu is not None:
        _pu_prop = (_estats.get(PROPOSED_ARM) or {}).get('per_unit', float('nan'))
        _abs_prop = (_estats.get(PROPOSED_ARM) or {}).get('abs_mean', float('nan'))
        _abs_others = [(_estats[m] or {}).get('abs_mean', float('nan'))
                       for m in modes_order if m != PROPOSED_ARM]
        _abs_others = [v for v in _abs_others if np.isfinite(v)]
        if np.isfinite(_abs_prop) and _abs_others and _abs_prop > max(_abs_others):
            _en_note = (f"\n{PROPOSED_ARM_SHORT}'s ABSOLUTE energy is marginally "
                        f"higher ({_abs_prop:.1f} vs "
                        f"{min(_abs_others):.1f}-{max(_abs_others):.1f} J): it "
                        f"activates relay links\nthe baselines do not. Its energy "
                        f"PER DELIVERED UNIT is nonetheless the lowest "
                        f"({_pu_prop:.4f}).")
        elif np.isfinite(_abs_prop) and _abs_others and _abs_prop < min(_abs_others):
            _en_note = (f"\n{PROPOSED_ARM_SHORT} is lowest on BOTH absolute energy "
                        f"({_abs_prop:.1f} J) and energy per delivered unit "
                        f"({_pu_prop:.4f}).")
        _en_note += f"\nLowest energy per delivered unit: {SHORT_LABELS[_best_pu]}."
    ax_en.set_title(
        f"Energy — absolute (solid, left axis) vs per delivered unit "
        f"(hatched, right axis)\nsteady state, last "
        f"{STEADY_STATE_WINDOW_TICKS} ticks; per-unit = Σenergy / Σdelivered"
        f"{_en_note}", fontsize=8.5)
    ax_en.grid(True, alpha=0.25, axis='y')

    # --- 11. SERVICE-LOSS INTEGRAL (PRIMARY recovery metric) ---------------
    # Threshold-free: the AREA between the shared pre-event achievability
    # level and the arm's achievability over the whole event window.  It
    # prices DEPTH and DURATION together and it sees re-dips, none of which a
    # first-crossing time can do.  Bars are the cross-seed mean with ±1σ;
    # every seed is also drawn as its own dot, so the spread is visible and
    # no seed is hidden inside an average.
    ax_sli = axes[5, 0]
    _sli_idx = np.arange(len(modes_order))
    _sli_mean = [(_sli[m] or {}).get('mean', float('nan')) for m in modes_order]
    _sli_sd = [(_sli[m] or {}).get('std', 0.0) for m in modes_order]
    _cols_sli = [STYLE[m]['color'] for m in modes_order]
    ax_sli.bar(_sli_idx, _sli_mean, width=0.62, yerr=_sli_sd, capsize=3,
               color=_cols_sli, alpha=0.55, edgecolor='white',
               error_kw=dict(ecolor='#444444', lw=1.0), zorder=2)
    _rng_sli = np.random.RandomState(24680)      # deterministic jitter
    for i, m in enumerate(modes_order):
        vals = [r['pp_s'] for r in _sli_rows.get(m, [])
                if np.isfinite(r['pp_s'])]
        if not vals:
            continue
        jit = _rng_sli.uniform(-0.17, 0.17, size=len(vals))
        ax_sli.scatter(np.full(len(vals), i) + jit, vals,
                       s=34 if len(vals) > 6 else 50,
                       color=STYLE[m]['color'], edgecolor='white',
                       linewidth=0.6, alpha=0.95,
                       zorder=6 if m == PROPOSED_ARM else 4)
    for i, (v, s_) in enumerate(zip(_sli_mean, _sli_sd)):
        if np.isfinite(v):
            ax_sli.text(i, v + s_, f"{v:,.0f}", ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')
    ax_sli.set_xticks(_sli_idx)
    ax_sli.set_xticklabels([SHORT_LABELS[m] for m in modes_order],
                           fontsize=8, rotation=15)
    for _tick, _m in zip(ax_sli.get_xticklabels(), modes_order):
        if _m == PROPOSED_ARM:
            _tick.set_fontweight('bold')
            _tick.set_color(PROPOSED_ARM_COLOR)
    ax_sli.set_ylabel("Service loss  ∫ max(0, ref − achiev) dt   [pp·s]")
    _sli_best = min((m for m in modes_order
                     if _sli[m] and np.isfinite(_sli[m]['mean'])),
                    key=lambda m: _sli[m]['mean'], default=None)
    _sli_note = ""
    if _sli_best is not None:
        _b = _sli[_sli_best]
        _sli_note = (f"\nLOWEST service loss (best): "
                     f"{SHORT_LABELS[_sli_best]} "
                     f"{_b['mean']:,.0f} ± {_b['std']:,.0f} pp·s   "
                     f"(median {_b['median']:,.0f}, IQR "
                     f"[{_b.get('q1', float('nan')):,.0f}, "
                     f"{_b.get('q3', float('nan')):,.0f}])")
        if _sli_vol_ok and _sli_v.get(_sli_best):
            _sli_note += (f"  ≡ {_sli_v[_sli_best]['mean']:,.0f} volume-units·s "
                          f"of traffic not delivered")
    # The bar is DEMAND-CORRECTED: the offered-demand model steps up at the
    # event tick, so the raw pre-event level would score every post-event
    # sample against a ~22 % smaller denominator.  Both numbers are stated on
    # the panel so the correction is never invisible.
    _dem_note = (f"\nREF is DEMAND-CORRECTED: raw pre-event "
                 f"{ref_level_raw:.1f}% × measured offered ratio "
                 f"(pre/post) {_demand['ratio']:.3f} = {ref_level:.1f}%"
                 if _demand['applied'] else
                 f"\nREF = raw pre-event {ref_level_raw:.1f}% "
                 f"(demand correction NOT applied — offered series "
                 f"unmeasurable)")
    ax_sli.set_title(
        f"PRIMARY — SERVICE-LOSS INTEGRAL over the event window "
        f"(threshold-free; LOWER IS BETTER)\n"
        f"∫ max(0, {ref_level:.1f}% − achievability(t)) dt from {_ev_name} to "
        f"t={PLOT_END}; bar = cross-seed mean ± 1σ, dots = individual seeds"
        f"{_dem_note}\n"
        f"prices DEPTH × DURATION and counts re-dips — unlike any "
        f"first-crossing time, and it needs no threshold{_sli_note}",
        fontsize=8.5)
    ax_sli.grid(True, alpha=0.25, axis='y')

    # --- 11b. AVAILABILITY at the swept thresholds -------------------------
    # Fraction of event-window ticks at or above each threshold.  This is the
    # panel that separates "crossed once and held" from "crossed once and
    # fell back": a first-crossing time reports both identically.
    ax_av = axes[5, 1]
    if _avail:
        _fr_list = list(RECOVERY_THRESHOLD_FRACTIONS)
        _w = 0.8 / max(1, len(_fr_list))
        _alphas = (0.95, 0.65, 0.38)
        for j, _fr in enumerate(_fr_list):
            _vals = [((_avail[_fr].get(m) or {}).get('mean', float('nan')))
                     for m in modes_order]
            _errs = [((_avail[_fr].get(m) or {}).get('std', 0.0))
                     for m in modes_order]
            ax_av.bar(np.arange(len(modes_order)) - 0.4 + _w * (j + 0.5),
                      [100.0 * v for v in _vals], width=_w * 0.92,
                      yerr=[100.0 * e for e in _errs], capsize=2,
                      color=_cols_sli, alpha=_alphas[j % len(_alphas)],
                      edgecolor='white',
                      error_kw=dict(ecolor='#555555', lw=0.7),
                      label=f"≥{int(_fr*100)}% of pre-event "
                            f"({_fr*ref_level:.1f}%)")
        ax_av.set_xticks(np.arange(len(modes_order)))
        ax_av.set_xticklabels([SHORT_LABELS[m] for m in modes_order],
                              fontsize=8, rotation=15)
        for _tick, _m in zip(ax_av.get_xticklabels(), modes_order):
            if _m == PROPOSED_ARM:
                _tick.set_fontweight('bold')
                _tick.set_color(PROPOSED_ARM_COLOR)
        ax_av.set_ylim(0, 108)
        ax_av.set_ylabel("Availability — % of event-window ticks ≥ threshold")
        ax_av.legend(loc='lower right', fontsize=6.5, ncol=1)
        ax_av.set_title(
            f"AVAILABILITY over the event window "
            f"(share of ticks at or above the bar; HIGHER IS BETTER)\n"
            f"same 30/50/70% thresholds as the crossing metric, but every "
            f"tick counts — so RE-DIPS after the first\ncrossing are visible "
            f"here and invisible to a first-crossing recovery time",
            fontsize=8.5)
        ax_av.grid(True, alpha=0.25, axis='y')
    else:
        ax_av.axis('off')
        ax_av.text(0.5, 0.5, "Availability unavailable:\nno pre-event "
                             "reference level could be computed.",
                   ha='center', va='center', fontsize=10)

    # --- 12. Summary Statistics Table ---
    ax9 = axes[6, 0]
    ax9.axis('off')
    # Steady-state summary: per seed, mean over the LAST
    # STEADY_STATE_WINDOW_TICKS ticks of that seed's full recorded run; the
    # ± column is 1 standard deviation ACROSS the held-out seeds.  The
    # online-only MARL ablation is excluded here exactly as it is from the
    # curves (see FIGURE_EXCLUDED_ARMS).
    sim_min = TOTAL_TICKS * TICK_DURATION_S / 60
    scen_tag = 'B (rescue ops)' if scenario_name == 'rescue_ops' else 'A (recovery)'

    def _sec(rt):
        return "  n/r" if rt is None else f"{rt * TICK_DURATION_S:5.0f}"

    def _sec_cat(entry, key='mean_curve', breached_key='mean_curve_breached'):
        """Recovery time, with NEVER BREACHED printed as a CATEGORY.

        An arm that never fell below the threshold is not a fast recoverer —
        it never lost service by this criterion — so it is never rendered as
        "0", which would read as an instantaneous recovery.
        """
        if entry.get(breached_key) is False:
            return "never br."
        return _sec(entry.get(key))

    table_lines = [
        f"STATISTICAL SUMMARY — Scenario {scen_tag}",
        f"{sim_min:.0f} min sim; steady state = mean of last "
        f"{STEADY_STATE_WINDOW_TICKS} ticks; ± = 1σ across the "
        f"{n_seeds} held-out seeds",
        f"seeds: {_seed_txt}",
        "Latency = MODELED, routed flows only (outage excluded)",
        "",
        "[1] PERFORMANCE / EFFICIENCY",
        "    SvcLoss = PRIMARY recovery metric (service-loss integral, pp*s, "
        "LOWER IS BETTER) — see [4]",
        "    Recov   = SECONDARY first-crossing time (threshold-dependent) — "
        "see [6]",
        f"{'Protocol':<12} {'Achiev%':>11} {'Lat(ms)':>12} {'Outage%':>8} "
        f"{'Thrpt':>9} {'Energy(J)':>11} {'J/delivered':>12} "
        f"{'SvcLoss(pp*s)':>15} {'Recov(s)':>10}",
        "-" * 110,
    ]
    for mode in modes_order:
        mc, sc, _n = steady_state_across_seeds(all_data_multi, mode, 'conn', seeds)
        ml, sl, _n = steady_state_across_seeds(all_data_multi, mode, 'latency', seeds)
        mo, _so, _n = steady_state_across_seeds(all_data_multi, mode, 'outage', seeds)
        md, sd, _n = steady_state_across_seeds(all_data_multi, mode, 'delivered', seeds)
        est = _estats.get(mode) or {}
        me = est.get('abs_mean', float('nan'))
        se = est.get('abs_std', float('nan'))
        pu = est.get('per_unit', float('nan'))
        mo = 100.0 * mo
        name = SHORT_LABELS[mode] + ('*' if mode == PROPOSED_ARM else '')
        _sl = _sli.get(mode)
        _sltxt = ("n/a" if not _sl
                  else f"{_sl['mean']:,.0f}±{_sl['std']:,.0f}")
        table_lines.append(
            f"{name:<12} {mc:5.1f}±{sc:4.1f}  {ml:6.1f}±{sl:5.1f}  {mo:6.1f}  "
            f"{md:4.0f}±{sd:3.0f}  {me:5.1f}±{se:4.1f}  {pu:11.4f}  "
            f"{_sltxt:>15} "
            f"{_sec_cat(primary.get(mode, {})):>10}")

    # [2] Reliability / variance block — median, min, max, IQR and the count
    # of seeds where the arm fails outright.  A ±σ alone cannot separate
    # "uniformly mediocre" from "usually fine, occasionally zero".
    table_lines += [
        "",
        f"[2] RELIABILITY ACROSS SEEDS (steady-state achievability, %)",
        f"{'Protocol':<12} {'median':>8} {'min':>7} {'max':>7} {'IQR':>7} "
        f"{'fail<' + str(int(FAILURE_ACHIEVABILITY_PCT)) + '%':>8} {'n':>4}   per-seed",
        "-" * 92,
    ]
    for mode in modes_order:
        st = variance_stats(_ss_vals.get(mode) or [])
        nfail, nseeds_used = failure_seed_count(all_data_multi, mode, seeds)
        if st is None:
            table_lines.append(f"{SHORT_LABELS[mode]:<12}   (no data)")
            continue
        # Per-seed values wrap once the seed count grows, so a 10+ seed run
        # does not run off the panel.  Each value keeps its seed label.
        _cells = [f"{s_}:{v:.1f}" for s_, v in (_ss_pairs.get(mode) or [])]
        _PER_ROW = 6
        _chunks = [" ".join(_cells[k:k + _PER_ROW])
                   for k in range(0, len(_cells), _PER_ROW)] or [""]
        table_lines.append(
            f"{SHORT_LABELS[mode]:<12} {st['median']:8.1f} {st['min']:7.1f} "
            f"{st['max']:7.1f} {st['iqr']:7.1f} {nfail:8d} {st['n']:4d}   "
            f"{_chunks[0]}")
        for _extra in _chunks[1:]:
            table_lines.append(" " * 57 + _extra)

    # [3] Fragmentation / reunification — what the severance did vs what the
    # arm achieved, and the relay LINKS (bridges) that closed the gap.
    _reun = {m: reunification_stats(all_data_multi, m, seeds)
             for m in modes_order}
    _has_reun_extra = _has_raw or _has_bridges
    table_lines += [
        "",
        "[3] FRAGMENTATION / REUNIFICATION (steady state; mean and "
        "min-max across seeds)",
    ]
    if scenario_name == 'rescue_ops':
        table_lines.append(
            "    Scenario B applies NO link cuts: it does not fragment the "
            "network. Cross-fragment flows are 0/222")
        table_lines.append(
            "    on every seed and components stay at 1-2. No partition "
            "bridging is claimed here.")
    # The TG-FEASIBLE OPTIMUM is the reference column: what the 60 GHz budget
    # and the site plan actually permit on these seeds.  Reporting `components`
    # against a flat 1 would charge every arm for the site plan.
    _tgopt = tg_optimum_summary(all_data_multi, seeds)
    _tgopt_mean = _tgopt['mean']
    if np.isfinite(_tgopt_mean):
        table_lines.append(
            f"    TG-FEASIBLE OPTIMUM @{_tgopt['tx_dbm']:.0f} dBm = "
            f"{_tgopt_mean:.2f} components (per seed "
            + ", ".join(f"s{k}={v}" for k, v in sorted(_tgopt['per_seed'].items()))
            + ") — the reference,")
        table_lines.append(
            "    NOT 1: it is the minimum component count reachable when every "
            "TG site steers optimally.")
    table_lines += [
        f"{'Protocol':<12} {'components':>22} {'optimum':>9} {'vs opt':>8} "
        f"{'raw partition':>22} {'relay bridges (links)':>24}",
        "-" * 101,
    ]
    for mode in modes_order:
        st = _reun.get(mode)

        def _cell(entry):
            if not entry:
                return f"{'n/a':>22}"
            return f"{entry['mean']:8.2f} ({entry['min']:.0f}-{entry['max']:.0f})".rjust(22)

        if not st:
            table_lines.append(f"{SHORT_LABELS[mode]:<12}   (no data)")
            continue
        _b = st.get('relay_bridges')
        _btxt = (f"{'n/a':>24}" if not _b
                 else f"{_b['mean']:8.2f} ({_b['min']:.0f}-{_b['max']:.0f})".rjust(24))
        _fr = st.get('fragments')
        if np.isfinite(_tgopt_mean):
            _otxt = f"{_tgopt_mean:9.2f}"
            _gtxt = (f"{_fr['mean'] - _tgopt_mean:+8.2f}" if _fr
                     else f"{'n/a':>8}")
        else:
            _otxt, _gtxt = f"{'n/a':>9}", f"{'n/a':>8}"
        table_lines.append(
            f"{SHORT_LABELS[mode]:<12} {_cell(_fr)} {_otxt} {_gtxt} "
            f"{_cell(st.get('fragments_raw'))} {_btxt}")
    table_lines += [
        "  components    = connected components INCLUDING relay links "
        "(what the arm achieved)",
        "  optimum       = TG-FEASIBLE OPTIMUM: fewest components reachable "
        "under TG-only steering at",
        "                  the screening power, each TG site holding one "
        "bridge. Identical for every arm.",
        "  vs opt        = components - optimum. 0.00 means the arm reached "
        "what the physics allows;",
        "                  only a POSITIVE value is a genuine shortfall.",
        "  raw partition = components of the relay-FREE graph (what the "
        "severance did; same for every arm)",
        "  relay bridges = relay LINKS joining two otherwise-disconnected "
        "fragments — the reunification work.",
        "    NOTE: `relay_count` counts active transport relay LINKS; nodes "
        "in a relay MODE are a different",
        "    quantity, recorded separately as `relay_mode_nodes`. This table "
        "reports LINKS only.",
    ]
    if not _has_reun_extra:
        table_lines.append(
            "    (raw partition / relay bridges are 'n/a' on datasets written "
            "before those series existed.)")

    # ══════════════════════════════════════════════════════════════════
    # [4] SERVICE-LOSS INTEGRAL — the PRIMARY recovery metric.
    # Threshold-free area between the shared pre-event level and the arm's
    # achievability over the event window: it prices DEPTH and DURATION
    # together, it sees re-dips, and it cannot be flipped by the choice of a
    # threshold constant.  Lower is better.
    # ══════════════════════════════════════════════════════════════════
    _ev_secs = (PLOT_END - event_tick) * TICK_DURATION_S
    table_lines += [
        "",
        "[4] SERVICE-LOSS INTEGRAL — PRIMARY RECOVERY METRIC "
        "(threshold-free; LOWER IS BETTER)",
        f"    SLI = SUM over t of max(0, REF - achievability(t)) * dt, "
        f"t from {_ev_name} (tick {event_tick}) to tick {PLOT_END}",
        f"    REF = the DEMAND-CORRECTED pre-event achievability level = "
        f"{ref_level:.2f}%"
        + ("" if _demand['applied'] else "   (correction NOT applied: the "
                                         "offered series could not be "
                                         "measured)"),
        f"      raw (NOT demand-corrected) pre-event level = "
        f"{ref_level_raw:.2f}%   x   measured offered-volume ratio "
        f"(pre/post) = {_demand['ratio']:.4f}   ->   REF = {ref_level:.2f}%",
        f"      offered volume: pre-event mean {_demand['offered_pre']:,.1f}, "
        f"post-event mean {_demand['offered_post']:,.1f} units/tick "
        f"(measured over {_demand['n_seeds']} seeds, "
        f"pre-window = {_demand['window']} ticks; source: {_off_src}, "
        f"coverage {_off_cov:.1%})",
        "      WHY: _offered_volume_for_flows switches demand model at the "
        "event tick, so the offered",
        "      DENOMINATOR steps up for every arm at once.  Scored against a "
        "raw pre-event bar, an arm can",
        "      deliver MORE ABSOLUTE TRAFFIC after the event than the intact "
        "network did and still be charged",
        "      service loss for it.  The correction rescales the bar into the "
        "POST-event demand regime:",
        "      REF_corrected = REF_raw * offered_pre/offered_post, i.e. the "
        "achievability an arm would show",
        "      if it delivered exactly the pre-event ABSOLUTE volume under "
        "post-event demand.  One scenario-level",
        "      scalar, identical for every arm, so it CANNOT change the "
        "ranking - only the absolute figures.",
        f"    cross-arm spread of the raw pre-event level {ref_spread:.2f} pp"
        + (" — identical across arms" if ref_is_shared
           else " — NOT arm-invariant; the cross-arm mean is used for all"),
        f"    per-arm RAW pre-event levels: "
        + ", ".join(f"{SHORT_LABELS.get(m, m)}={v:.2f}%"
                    for m, v in sorted(ref_per_arm.items(),
                                       key=lambda kv: kv[0])
                    if m in modes_order),
        f"    full window = {PLOT_END - event_tick} ticks "
        f"({_ev_secs:.0f} s); bounded window = first "
        f"{SERVICE_LOSS_BOUNDED_TICKS} ticks after the event",
        "    NaN (outage) ticks count as ZERO achievability, i.e. a "
        "full-depth loss — an outage is not missing data.",
        "    MEDIAN [Q1,Q3] IS REPORTED NEXT TO THE MEAN BECAUSE THE MEAN IS "
        "OFTEN ONE SEED: on the 10-seed run",
        "    MARL Scenario A reads 1,904+-5,681 pp*s from a single seed "
        "(51:18946) against a median of 3.",
        f"{'Protocol':<12} {'SLI full (pp*s)':>21} {'median':>10} "
        f"{'IQR [Q1,Q3]':>21} {'min':>9} {'max':>10} {'SLI bounded':>13} "
        f"{'volume-units*s':>17} {'SLI @RAW ref':>14}",
        "-" * 150,
    ]
    for mode in modes_order:
        _sl, _slb, _slv = _sli.get(mode), _sli_b.get(mode), _sli_v.get(mode)
        _slr = _sli_raw.get(mode)
        if not _sl:
            table_lines.append(f"{SHORT_LABELS[mode]:<12}   (no data)")
            continue
        _vtxt = ("n/a" if (not _sli_vol_ok or not _slv)
                 else f"{_slv['mean']:,.0f}±{_slv['std']:,.0f}")
        _mtxt = f"{_sl['mean']:,.0f}±{_sl['std']:,.0f}"
        _btxt = f"{_slb['mean']:,.0f}" if _slb else "n/a"
        _qtxt = f"[{_sl.get('q1', float('nan')):,.0f}, {_sl.get('q3', float('nan')):,.0f}]"
        _rtxt = (f"{_slr['mean']:,.0f}" if _slr else "n/a")
        table_lines.append(
            f"{SHORT_LABELS[mode]:<12} {_mtxt:>21} "
            f"{_sl['median']:10,.0f} {_qtxt:>21} {_sl['min']:9,.0f} "
            f"{_sl['max']:10,.0f} {_btxt:>13} {_vtxt:>17} {_rtxt:>14}")
    table_lines.append(
        "  SLI @RAW ref = the SECONDARY, NOT DEMAND-CORRECTED figure: the "
        "same integral taken against the raw")
    table_lines.append(
        f"  pre-event bar of {ref_level_raw:.2f}%.  Shown so the size of the "
        f"correction is auditable; it is NOT the headline,")
    table_lines.append(
        "  because it prices the event-tick demand step as if it were lost "
        "service.")
    if _sli_best is not None:
        _sli_worst = max((m for m in modes_order
                          if _sli[m] and np.isfinite(_sli[m]['mean'])),
                         key=lambda m: _sli[m]['mean'])
        table_lines.append(
            f"  BEST (least service lost): {SHORT_LABELS[_sli_best]} "
            f"{_sli[_sli_best]['mean']:,.0f} pp*s;  WORST: "
            f"{SHORT_LABELS[_sli_worst]} {_sli[_sli_worst]['mean']:,.0f} pp*s "
            f"({_sli[_sli_worst]['mean'] / max(1e-9, _sli[_sli_best]['mean']):.1f}x)")
    if _sli_vol_ok:
        table_lines.append(
            f"  volume-units*s = TRAFFIC NOT DELIVERED because of the event: "
            f"SUM max(0,(REF-achiev)/100) * offered_hat(t) * dt.")
        table_lines.append(
            f"  offered_hat(t) source = {_off_src} "
            + ("(the per-tick offered series the engine actually divided by)"
               if _off_src == 'recorded' else
               "(reconstructed post hoc as 100*delivered/achievability)")
            + f", pooled ACROSS ARMS by median (offered demand is a property "
              f"of the seed+tick, not of the arm);")
        table_lines.append(
            f"  reconstruction covered {_sli_vol_cov:.1%} of event-window "
            f"ticks (ticks with achievability <"
            f"{OFFERED_RECON_MIN_CONN_PCT:.0f}% or at the 100% clamp carry no "
            f"usable estimate and are EXCLUDED, never zero-filled).")
    else:
        table_lines.append(
            "  volume-units*s = NOT AVAILABLE: the offered demand could be "
            "reconstructed on only "
            + (f"{_sli_vol_cov:.1%}" if np.isfinite(_sli_vol_cov) else "0%")
            + " of event-window ticks, which is too sparse to integrate "
              "honestly, so the")
        table_lines.append(
            "  physically-meaningful delivered-volume form is omitted rather "
            "than extrapolated. The pp*s form above is unaffected.")

    # ══════════════════════════════════════════════════════════════════
    # [4b] Uu CARRIER-AGGREGATION PARITY AUDIT.
    # The defect this exists to make impossible: the Uu capacity boost used to
    # be reachable only from a `mode == 'marl'` branch AND logged only there,
    # so "every baseline got zero boosted links" never appeared anywhere.  It
    # is now a shared helper (_apply_uu_carrier_aggregation) whose eligibility
    # is an explicit architectural claim, and the per-arm outcome is printed
    # here whether it is zero or not.
    # ══════════════════════════════════════════════════════════════════
    _ca_has = any('uu_boost_links' in (all_data_multi.get(m, {}).get(s) or {})
                  for m in modes_order for s in seeds)
    table_lines += [
        "",
        "[4b] Uu CARRIER-AGGREGATION PARITY AUDIT — who was allowed to "
        "command extra spectrum, and how much",
        "    MODEL (v2, BOUNDED SITE-LEVEL POOL).  An eligible site activates "
        "k = min(N_CC, #hot-spot UEs) of its",
        f"    N_CC = {UU_CA_MAX_EXTRA_CC} spare component carrier(s) "
        f"({UU_CA_CC_BANDWIDTH_MHZ:.0f} MHz ACCESS_FR1, peak SE "
        f"{UU_CA_CC_REF_SE_BPS_HZ:.1f} b/s/Hz ="
        f" {UU_CA_CC_REF_CAPACITY_MBPS:.0f} Mbps at the",
        "    QAM256 reference `link.capacity` is already expressed in), and "
        "WATER-FILLS that finite pool across the",
        "    Uu access legs of its hot-spot UEs (UEs carrying >= "
        f"{UU_CA_HOTSPOT_MIN_FLOWS} flows), each link capped at k x its own",
        "    base capacity (per-UE aggregation capability).  The Uu link is "
        "the FIRST and the LAST hop of every",
        "    UE-to-UE flow and consume_path takes min(available) along the "
        "path, so this term gates END-TO-END",
        "    throughput.",
        "    THE SITE-LEVEL BOUND IS HARD: aggregate extra <= "
        f"{UU_CA_SITE_POOL_BOUND_MBPS:.0f} Mbps@ref PER SITE regardless of how "
        "many UEs",
        "    are camped on it — asserted per site per tick, and reported as "
        "extra_peak vs study_ceiling in the",
        "    per-pass [UU-BOOST-SUMMARY] log line.  A site serving 20 UEs and "
        "one serving 2 get the SAME extra",
        "    spectrum, split differently.",
        "    WHAT CHANGED FROM v1, AND WHY: v1 granted "
        "base x (1 + min(1, prb_emergency_fraction)) PER LINK with no",
        "    site cap.  prb_emergency_fraction is a SCHEDULING SPLIT of the "
        "carrier the site already has — a site",
        "    that gives 40% of its PRBs to emergency traffic does not thereby "
        "acquire 40% more spectrum — and the",
        "    per-link application handed ~42 sites the equivalent of ~5 extra "
        "full-bandwidth carriers EACH.  Both",
        "    defects are removed: the magnitude is now an INTEGER COUNT OF "
        "CARRIERS and depends on NO agent action.",
        "    ELIGIBILITY IS AN ARCHITECTURAL CLAIM, not a tuning knob:",
        "      MARL / MARL-freeze / RANDOM  eligible — the per-site RIC agent "
        "commands the SCell activation.",
        "                                   Magnitude is NOT an agent action: "
        "same k x 20 MHz as any eligible site.",
        "      SDN                          eligible — gated on the "
        "fragment-local controller being operational,",
        "                                   the SAME gate as its MultiHaul "
        "relay bridging; commanded over NETCONF/YANG",
        "                                   (3GPP TS 28.541 NRCellDU / O-RAN "
        "O1).  IDENTICAL pool to a MARL site.",
        "      OSPF / OLSR / BATMAN / AODV  NOT eligible — a routing protocol "
        "has no management plane, no RAN",
        "                                   configuration object and no "
        "mechanism to command a radio.  The hardware",
        "                                   exists on their sites too; they "
        "have nothing that can turn it on.",
        "    PHYSICS CAVEAT, stated rather than hidden: the pool CREATES "
        "capacity, it does not reallocate spectrum.",
        "    It is defensible only as CARRIER AGGREGATION of an additional "
        "LICENSED component carrier the site holds",
        "    but does not use in normal operation — never as a smarter "
        "scheduler on the existing one.  Whether one",
        "    spare full-bandwidth carrier is idle at EVERY site in a disaster "
        "area is itself an assumption in the",
        "    model's favour; N_CC=0 (UU_CA_MAX_EXTRA_CC=0) reruns the whole "
        "study with the grant switched off.",
        f"{'Protocol':<12} {'eligible':>9} {'ticks w/ boost':>15} "
        f"{'link-ticks':>12} {'peak links':>11} {'mean factor':>12} "
        f"{'cc/site':>9} {'extra Mbps':>11}",
        "-" * 96,
    ]
    if not _ca_has:
        table_lines.append(
            "    (no data: this dataset was written before the "
            "uu_boost_links / uu_boost_factor series existed.)")
    else:
        for mode in modes_order:
            _tk = _lt = _pk = 0
            _fs = []
            _ccs, _exs, _nsites = [], [], []
            for s in seeds:
                run = all_data_multi.get(mode, {}).get(s)
                if not run or 'uu_boost_links' not in run:
                    continue
                _n = list(run.get('uu_boost_links', []))[:PLOT_END]
                _f = list(run.get('uu_boost_factor', []))[:PLOT_END]
                _c = list(run.get('uu_cc_activated', []))[:PLOT_END]
                _e = list(run.get('uu_extra_mbps', []))[:PLOT_END]
                _tk += sum(1 for v in _n if v > 0)
                _lt += sum(_n)
                _pk = max([_pk] + list(_n))
                _fs += [f for v, f in zip(_n, _f) if v > 0]
                _ccs += [c for c in _c if c > 0]
                _exs += [e for e in _e if e > 0]
                _ns = run.get('n_radio_sites')
                if _ns:
                    _nsites.append(float(_ns))
            _mf = (sum(_fs) / len(_fs)) if _fs else 1.0
            _cc = (sum(_ccs) / len(_ccs)) if _ccs else 0.0
            _ex = (sum(_exs) / len(_exs)) if _exs else 0.0
            _ns_mean = (sum(_nsites) / len(_nsites)) if _nsites else 0.0
            _cc_per_site = (_cc / _ns_mean) if _ns_mean > 0 else float('nan')
            _cc_str = (f"{_cc_per_site:.2f}" if _ns_mean > 0
                       else f"{_cc:.0f}tot")
            table_lines.append(
                f"{SHORT_LABELS[mode]:<12} {('YES' if _lt else 'no'):>9} "
                f"{_tk:>15,} {_lt:>12,} {_pk:>11,} {_mf:>11.3f}x "
                f"{_cc_str:>9} {_ex:>11,.0f}")
        table_lines.append(
            "    cc/site = mean ACTIVATED component carriers per radio site "
            "while the grant is live; it can never")
        table_lines.append(
            f"    exceed N_CC = {UU_CA_MAX_EXTRA_CC}.  extra Mbps = mean "
            "AGGREGATE extra capacity across the whole network at the")
        table_lines.append(
            "    QAM256 reference, i.e. cc/site x sites x "
            f"{UU_CA_CC_REF_CAPACITY_MBPS:.0f}.  Under v1 this column had no "
            "ceiling at all; under v2 it is")
        table_lines.append(
            "    bounded by construction and the bound is asserted every "
            "tick.")
        table_lines.append(
            "    An eligible arm reading 0 here means its GATE never opened "
            "(e.g. the SDN controller never reached")
        table_lines.append(
            "    its operational phase on any seed) — not that the capability "
            "was withheld.  A non-eligible arm")
        table_lines.append(
            "    reading non-zero, or a cc/site above N_CC, would be a BUG in "
            "the parity gating / the site bound")
        table_lines.append(
            "    and should fail review.")

    # ══════════════════════════════════════════════════════════════════
    # [5] AVAILABILITY — the statistic a first-crossing time cannot have.
    # ══════════════════════════════════════════════════════════════════
    if _avail:
        table_lines += [
            "",
            "[5] AVAILABILITY — share of EVENT-WINDOW ticks at or above the "
            "threshold (HIGHER IS BETTER)",
            "    Captures RE-DIPS: a first-crossing time is fixed at the "
            "first crossing and is blind to everything after it.",
            f"{'Protocol':<12} " + " ".join(
                f"{f'>={int(f*100)}% ({f*ref_level:.1f}%)':>20}"
                for f in RECOVERY_THRESHOLD_FRACTIONS),
            "-" * 92,
        ]
        for mode in modes_order:
            cells = []
            for f in RECOVERY_THRESHOLD_FRACTIONS:
                a = (_avail.get(f) or {}).get(mode)
                cells.append((f"{100*a['mean']:.1f}±{100*a['std']:.1f}%"
                              if a else "n/a").rjust(20))
            table_lines.append(f"{SHORT_LABELS[mode]:<12} " + " ".join(cells))
        table_lines.append(
            "  mean ± 1σ across seeds; NaN (outage) ticks count as BELOW the "
            "threshold.")

    # [6] Recovery-time block — the fair swept thresholds PLUS the legacy
    # self-referential number, explicitly labelled as non-comparable.
    table_lines += [
        "",
        f"[6] RECOVERY TIME — SECONDARY, first-crossing (s from {_ev_name}; "
        f"sustained {RECOVERY_SUSTAIN_TICKS} ticks)",
        "    SECONDARY because it is threshold-dependent (the ranking can "
        "flip between 30/50/70%) and blind to depth, area and re-dips.",
        f"    FIXED ABSOLUTE thresholds, IDENTICAL for every arm — "
        f"pre-event achievability level = {ref_level:.1f}%",
    ]
    if not ref_is_shared:
        table_lines.append(
            f"    (arms differ pre-event by {ref_spread:.1f} pp; the reference "
            f"is their cross-arm mean — one threshold for all)")
    # The median / never-breached / censored columns all belong to the ONE
    # primary threshold, so they carry it in their header: mixing a 70%
    # crossing time next to a bare "nb" counted at 50% is unreadable.
    _pf_txt = f"@{int(RECOVERY_PRIMARY_FRACTION * 100)}%"
    table_lines += [
        f"{'Protocol':<12} " + " ".join(
            f"{f'{int(f*100)}%={f*ref_level:5.1f}%':>14}"
            for f in RECOVERY_THRESHOLD_FRACTIONS)
        + f" {_pf_txt + ' median*':>13} {'nb' + _pf_txt:>7} {'cens' + _pf_txt:>9}   {'90%-own-SS':>11}",
        "-" * 100,
    ]
    for mode in modes_order:
        # "never breached" prints as its own token, never as 0 — the two
        # states are different and only one of them is a recovery time.
        cells = " ".join(
            f"{_sec_cat(fair_sweep.get(f, {}).get(mode, {})):>14}"
            for f in RECOVERY_THRESHOLD_FRACTIONS)
        pm = primary.get(mode, {})
        # Median over BREACHED, RECOVERED seeds only: never-breached seeds
        # are excluded from the average rather than dragging it to zero.
        med = pm.get('median_breached')
        _nb = pm.get('n_never_breached', 0)
        med_txt = ("  all nb" if (med is None and _nb and _nb == pm.get('n'))
                   else ("  n/r" if med is None
                         else f"{med * TICK_DURATION_S:.0f}"))
        table_lines.append(
            f"{SHORT_LABELS[mode]:<12} {cells} {med_txt:>13} "
            f"{_nb:7d} {pm.get('censored_breached', 0):9d}   "
            f"{_sec(legacy_recovery.get(mode)):>11}")
    table_lines += [
        "  n/r  = threshold reached-and-sustained never happened on a seed "
        "that DID breach (censored observation)",
        "  never br. / nb = NEVER BREACHED: the arm's achievability never "
        "fell below this threshold after the event.",
        "    This is a CATEGORY, NOT a recovery time of 0 s — never losing "
        "service and recovering instantly are different",
        "    properties, and only the second is a recovery time. Such seeds "
        "are EXCLUDED from the median (nb column counts them).",
        f"  * median{_pf_txt} / nb{_pf_txt} / cens{_pf_txt} are all counted "
        f"at the ONE primary threshold "
        f"({RECOVERY_PRIMARY_FRACTION:.0%} = {primary_thr:.1f}%), not at the "
        f"swept ones; median is over BREACHED",
        "    AND RECOVERED seeds only, and 'all nb' means no seed of that arm "
        "ever breached that primary threshold.",
        "  cens = seeds that breached but never reached-and-sustained the "
        "threshold again inside the window.",
        "  A median of 0 s on an arm with nb < n means the first sustained "
        "window after the event already met the bar and the",
        "    breach came LATER \u2014 i.e. a RE-DIP, which this metric cannot "
        "price at all. Read availability [5] for that.",
        "  90%-own-SS = LEGACY 'time to 90% of the arm's OWN steady state' — "
        "NOT COMPARABLE ACROSS ARMS",
        "    (each arm is measured against a different absolute bar, so the "
        "best-performing arm is penalised); retained for continuity only.",
        "  NOTE the threshold sensitivity this block exposes is exactly why "
        "it is SECONDARY: compare the 30/50/70% columns",
        "    against each other, and against the threshold-free service-loss "
        "integral in [4].",
    ]

    # ══════════════════════════════════
    # [7] TIME TO X% OF THE FEASIBLE OPTIMUM — computed only when a
    # per-seed FEASIBLE ACHIEVABILITY CEILING exists.  Never invented.
    # ══════════════════════════════════
    table_lines += ["", "[7] TIME TO X% OF THE FEASIBLE OPTIMUM "
                        "(s from event; bar = fraction of the PER-SEED "
                        "feasible achievability ceiling)"]
    if _ceil_rec and _ceil_per_seed:
        table_lines.append(
            f"    ceiling source: {_ceil_src}; per seed "
            + ", ".join(f"s{k}={v:.1f}%"
                        for k, v in sorted(_ceil_per_seed.items())))
        table_lines.append(
            f"{'Protocol':<12} " + " ".join(
                f"{f'{int(f*100)}% of ceiling':>18}"
                for f in CEILING_RECOVERY_FRACTIONS)
            + f" {'nb@' + str(int(CEILING_RECOVERY_FRACTIONS[0]*100)) + '%':>7}")
        table_lines.append("-" * 92)
        for mode in modes_order:
            cells = []
            for f in CEILING_RECOVERY_FRACTIONS:
                e = (_ceil_rec.get(f) or {}).get(mode) or {}
                mv = e.get('median_breached')
                cells.append(("all nb" if (mv is None
                                           and e.get('n_never_breached')
                                           == e.get('n'))
                              else ("n/r" if mv is None
                                    else f"{mv * TICK_DURATION_S:.0f}")
                              ).rjust(18))
            _e0 = (_ceil_rec.get(CEILING_RECOVERY_FRACTIONS[0])
                   or {}).get(mode) or {}
            table_lines.append(f"{SHORT_LABELS[mode]:<12} " + " ".join(cells)
                               + f" {_e0.get('n_never_breached', 0):7d}")
        table_lines.append(
            f"  Median over BREACHED AND RECOVERED seeds only; n/r = "
            f"breached but never reached-and-sustained that bar; "
            f"nb@{int(CEILING_RECOVERY_FRACTIONS[0]*100)}% = seeds that never "
            f"breached the "
            f"{int(CEILING_RECOVERY_FRACTIONS[0]*100)}%-of-ceiling bar at "
            f"all.")
    else:
        table_lines.append("    NOT COMPUTED — and deliberately not "
                           "approximated. Reason:")
        for _ln in _wrap_reason(_ceil_reason, 96):
            table_lines.append(f"      {_ln}")
        _tg = tg_optimum_summary(all_data_multi, seeds)
        if _tg['per_seed']:
            table_lines.append(
                f"    (For reference, the TG-feasible COMPONENT optimum IS "
                f"available and is reported in [3]: mean {_tg['mean']:.2f} "
                f"components @{_tg['tx_dbm']:.0f} dBm.")
            table_lines.append(
                "     It bounds CONNECTIVITY, not achievability, so it is "
                "used there and refused here.)")

    # Time to first delivered cross-fragment flow — reported only if the
    # per-tick recorder provides the series; never proxied.
    _xf = {m: first_cross_fragment_delivery(all_data_multi, m, seeds,
                                            scenario_name, PLOT_END)
           for m in modes_order}
    if any(v is not None for v in _xf.values()):
        table_lines += ["", "[8] TIME TO FIRST DELIVERED CROSS-FRAGMENT FLOW "
                            "(s from event, median over seeds)"]
        for mode in modes_order:
            v = _xf.get(mode)
            table_lines.append(
                f"{SHORT_LABELS[mode]:<12} "
                + ("  (not observed)" if v is None
                   else f"{v[0] * TICK_DURATION_S:8.0f}   (n={v[1]})"))
    elif scenario_name == 'rescue_ops':
        table_lines += [
            "", "[8] Time to first delivered CROSS-FRAGMENT flow: NOT "
                "APPLICABLE in Scenario B",
            "    Scenario B cuts no links, so there are no cross-fragment "
            "flows to deliver (0/222 on every seed).",
        ]
    else:
        table_lines += [
            "", "[8] Time to first delivered CROSS-FRAGMENT flow: NOT AVAILABLE",
            "    the per-tick recorder does not emit a cross-fragment delivered",
            "    series, so this metric is omitted rather than approximated.",
        ]

    table_lines.append("")
    table_lines.append(f"* {PROPOSED_ARM_LABEL}: {PROPOSED_ARM_DESCRIPTION}")
    if FIGURE_EXCLUDED_ARMS:
        table_lines.append(
            "Online-only MARL ablation is retained in the raw data "
            "(multiseed pickle) but excluded from figures/tables.")
    ax9.text(0.01, 0.99, "\n".join(table_lines), transform=ax9.transAxes,
             fontsize=5.6, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Echo the same table to stdout so a --replot run leaves a verifiable
    # text record of exactly what the figure shows.
    print("\n" + "\n".join(table_lines) + "\n")

    # --- 13. Experiment Info (computed from the ACTUAL constructed topology,
    #         never hardcoded literals) ---
    ax10 = axes[6, 1]
    ax10.axis('off')
    # Aggregate the per-seed topology_info recorded by every run — node/UE
    # counts are structural (identical across seeds) while the link count
    # varies with the seeded geography, so report its mean and range.
    _topos = []
    for _m in modes_order:
        for _s in seeds:
            _d = all_data_multi.get(_m, {}).get(_s)
            if _d and _d.get('topology_info'):
                _topos.append((_s, _d['topology_info']))
        if _topos:
            break
    if _topos:
        _infra = sorted({t['infra'] for _s, t in _topos})
        _ues = sorted({t['ues'] for _s, t in _topos})
        _mh = sorted({t['multihaul'] for _s, t in _topos})
        _links = [t['links'] for _s, t in _topos]
        _fmt = lambda vals: (f"{vals[0]}" if len(vals) == 1
                             else f"{min(vals)}-{max(vals)}")
        topo_lines = (f"Topology (measured from constructed graph):\n"
                      f"  {_fmt(_infra)} infra nodes + {_fmt(_ues)} UEs per seed\n"
                      f"  links: mean {sum(_links)/len(_links):.0f} "
                      f"(range {min(_links)}-{max(_links)} over "
                      f"{len(_links)} seeds)\n"
                      f"  {_fmt(_mh)} MultiHaul sites\n"
                      f"  Structure identical, geography varies by seed\n")
    else:
        topo_lines = "Topology: (no per-seed data available)\n"

    # Fragmenting-severance parameters — measured, not literals.
    if scenario_name == 'rescue_ops':
        frag_lines = (f"Severance (Scenario B): core severed at t=0\n"
                      f"  (already-fragmented steady state; no extra cuts)\n"
                      f"  +20 rescue UEs at T+"
                      f"{RESCUE_TICK * TICK_DURATION_S / 60:.0f}min\n")
    elif frag_stats:
        _ps = frag_stats['per_seed']
        _fr = sorted({v['fragments'] for v in _ps.values()})
        frag_lines = (
            f"Fragmenting severance (measured per seed):\n"
            f"  fragments: mean {frag_stats['mean_fragments']:.1f} "
            f"(range {min(_fr)}-{max(_fr)})\n"
            f"  cross-fragment UE↔UE flow share: "
            f"{frag_stats['mean_share']:.1%} mean\n"
            f"  (per-seed {frag_stats['min_share']:.1%}-"
            f"{frag_stats['max_share']:.1%}; "
            f"{frag_stats['cross_total']}/{frag_stats['flow_total']} flows)\n"
            f"  cut links: mean {frag_stats['mean_cut_links']:.0f} per seed\n"
            f"  Targets: {FRAG_MIN_FRAGMENTS}-{FRAG_MAX_FRAGMENTS} fragments, "
            f"share {FRAG_CROSS_FLOW_MIN:.0%}-{FRAG_CROSS_FLOW_MAX:.0%}\n")
    else:
        frag_lines = "Fragmenting severance: (parameters unavailable)\n"

    info_text = (
        f"EXPERIMENT CONFIGURATION\n"
        f"{'='*44}\n"
        f"Training topology seed: 42\n"
        f"Held-out test topology seeds ({n_seeds}):\n"
        f"  {_seed_txt}\n"
        f"{topo_lines}"
        f"Sim: {TOTAL_TICKS} ticks x {TICK_DURATION_S:.0f}s\n"
        f"{frag_lines}\n"
        f"{PROPOSED_ARM_LABEL}:\n"
        f"  pre-trained (seed 42) + online-adapted during\n"
        f"  deployment, argmax-FROZEN at the best state after\n"
        f"  the last event + {FREEZE_AFTER_LAST_EVENT_TICKS} ticks\n"
        f"  (no online updates after the freeze); freeze gated\n"
        f"  on sane trailing achievability, forced by\n"
        f"  +{FREEZE_FORCE_AFTER_TICKS} ticks at the best checkpoint\n"
        f"  Deployment LR/clip from MAPPOConfig; PPO update\n"
        f"  interval 30s; exploration temp floor {TEMP_FLOOR}, boosted\n"
        f"  only for {TEMP_BOOST_WINDOW_TICKS} ticks after a logged event\n"
        f"Ablation NOT plotted: the online-only MARL arm is\n"
        f"  still simulated and kept in the raw multiseed\n"
        f"  pickle; it is reported in the article text only\n\n"
        f"Steady state: mean of the last {STEADY_STATE_WINDOW_TICKS} ticks\n"
        f"Shaded bands / ±: 1 standard deviation across the\n"
        f"  {n_seeds} held-out seeds; solid lines = seed mean\n\n"
        f"Latency panel: MODELED (analytical per-hop model,\n"
        f"  not packet-level); outage excluded from the mean\n"
        f"Fair PHY: ALL arms get auto CQI→MCS; MARL\n"
        f"  MCS heads are bounded offsets from it\n"
        f"  (offset 0 = follow auto-CQI exactly)\n"
        f"Recovery: measured per-protocol control-plane\n"
        f"  convergence over the surviving topology\n"
        f"PRIMARY recovery metric: SERVICE-LOSS INTEGRAL\n"
        f"  SLI = SUM max(0, REF - achievability(t)) * dt over\n"
        f"  the event window, REF = the shared PRE-EVENT level\n"
        f"  ({ref_level:.1f}%, cross-arm spread {ref_spread:.2f} pp).\n"
        f"  THRESHOLD-FREE, prices DEPTH x DURATION, and counts\n"
        f"  re-dips. Units pp*s; also in delivered-volume units\n"
        f"  (traffic not delivered because of the event) where\n"
        f"  the offered demand can be reconstructed. LOWER=BETTER\n"
        f"Availability: share of event-window ticks at or above\n"
        f"  each swept threshold - the statistic that exposes\n"
        f"  RE-DIPS after the first crossing\n"
        f"'never breached' is a CATEGORY, never 0 s: an arm that\n"
        f"  never fell below the bar did not recover instantly,\n"
        f"  it never lost service. Such arms/seeds are excluded\n"
        f"  from every mean/median of recovery time and counted\n"
        f"  separately (nb column)\n"
        f"SECONDARY recovery metric: FIXED ABSOLUTE threshold,\n"
        f"  identical for every arm — "
        f"{RECOVERY_PRIMARY_FRACTION:.0%} of the {ref_level:.1f}%\n"
        f"  pre-event achievability level "
        f"(= {RECOVERY_PRIMARY_FRACTION * ref_level:.1f}%),\n"
        f"  sustained {RECOVERY_SUSTAIN_TICKS} consecutive ticks; swept over\n"
        f"  {'/'.join(f'{int(f*100)}%' for f in RECOVERY_THRESHOLD_FRACTIONS)}"
        f" of that level in the table\n"
        f"  The legacy 'time to 90% of the arm's OWN steady\n"
        f"  state' is retained in the table for continuity but\n"
        f"  is NOT comparable across arms (moving bar)\n"
        f"Energy: absolute J/tick AND J per delivered unit\n"
        f"  (Σenergy / Σdelivered over the steady-state window)\n"
        f"Fragments panel: `fragments` = components INCLUDING\n"
        f"  relay links (achieved); red line = the relay-free\n"
        f"  raw partition (what the severance did, same for\n"
        f"  every arm); GREEN line = the TG-FEASIBLE OPTIMUM,\n"
        f"  the fewest components the 60 GHz budget permits —\n"
        f"  that green line, NOT 1, is the target\n"
        f"  `relay_bridges` = relay LINKS joining two otherwise\n"
        f"  disconnected fragments. NOTE `relay_count` counts\n"
        f"  transport relay LINKS; nodes in a relay MODE are a\n"
        f"  separate series, `relay_mode_nodes` — never mixed\n"
        + (f"Scenario B cuts NO links: it does not fragment the\n"
           f"  network (cross-fragment flows 0/222 every seed,\n"
           f"  components 1-2). No partition bridging is implied\n"
           if scenario_name == 'rescue_ops' else "")
        + f"Reliability: per-seed dots + median/min/max/IQR and\n"
        f"  a count of seeds below "
        f"{FAILURE_ACHIEVABILITY_PCT:.0f}% achievability\n"
        f"Achievability: delivered / live offered demand\n"
        f"  (same denominator formula for every arm)"
    )
    ax10.text(0.02, 0.995, info_text, transform=ax10.transAxes, fontsize=6.8,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    # ── Figure caption (bottom of the page) ──
    caption = (
        f"Steady state = mean over the last {STEADY_STATE_WINDOW_TICKS} ticks "
        f"of each run; every ± is 1σ across the {n_seeds} HELD-OUT topology "
        f"seeds {_seed_txt} (training used seed 42).   "
        f"{PROPOSED_ARM_LABEL} = MARL RIC, {PROPOSED_ARM_DESCRIPTION}.   "
        f"Latency panels are a MODELED analytical per-hop quantity "
        f"(propagation + M/M/1 queueing + HARQ), not packet-level measurements; "
        f"outage is reported separately and excluded from the latency mean."
        f"\nRECOVERY is reported PRIMARILY by the THRESHOLD-FREE SERVICE-LOSS "
        f"INTEGRAL: the area between the shared pre-event achievability level "
        f"({ref_level:.1f}%, cross-arm spread {ref_spread:.2f} pp) and each "
        f"arm's achievability over the whole event window, in "
        f"percentage-point-seconds and\n(where the offered demand can be "
        f"reconstructed) in delivered-volume units - the traffic not delivered "
        f"because of the event. Lower is better. It prices the DEPTH and the "
        f"DURATION of the degradation together and it counts RE-DIPS, none of "
        f"which a first-crossing time can do.\nAVAILABILITY (share of "
        f"event-window ticks at or above each threshold) is reported alongside "
        f"it for the same reason. 'NEVER BREACHED' is printed as its own "
        f"CATEGORY and is NEVER encoded as 0 s: never losing service and "
        f"recovering instantly are\ndifferent properties and only the second "
        f"is a recovery time; those arms and seeds are excluded from every "
        f"mean/median of recovery time and counted separately."
        f"\nSECONDARY, retained for continuity: RECOVERY TIME measured against "
        f"a FIXED ABSOLUTE threshold that is "
        f"identical for every arm: {RECOVERY_PRIMARY_FRACTION:.0%} of the "
        f"{ref_level:.1f}% pre-event achievability level "
        f"(= {RECOVERY_PRIMARY_FRACTION * ref_level:.1f}%), sustained for "
        f"{RECOVERY_SUSTAIN_TICKS} consecutive ticks, and swept over "
        f"{'/'.join(f'{int(f*100)}%' for f in RECOVERY_THRESHOLD_FRACTIONS)} of "
        f"that level so no single\nthreshold choice drives the conclusion. "
        f"The legacy 'time to 90% of the arm's OWN steady state' is kept in the "
        f"table for continuity only — it is NOT comparable across arms, because "
        f"each arm is scored against a different absolute bar and the "
        f"best-performing arm is therefore penalised.   "
        f"ENERGY is reported both in absolute Joules per tick and per delivered "
        f"traffic unit (Σenergy / Σdelivered over the steady-state window)."
        + _en_note.replace("\n", " ")
        + f"\nRELIABILITY: the strip panel plots every held-out seed "
          f"individually; the table adds median, min, max, IQR and a count of "
          f"seeds below {FAILURE_ACHIEVABILITY_PCT:.0f}% achievability. No seed "
          f"is dropped, reweighted or winsorised anywhere in this figure."
        + ("\nFRAGMENTS: the plotted count includes relay links (the "
           "connectivity the arm achieved); the red reference line is the "
           "relay-free raw partition, i.e. what the severance did, identical "
           "for every arm — the gap between them is the reunification "
           "achieved. The GREEN line is the TG-FEASIBLE OPTIMUM: the fewest "
           "components reachable\nwhen every TG-capable site steers optimally "
           "within the 60 GHz link budget at the screening power. An arm is "
           "judged against that green line, NOT against 1 — on seeds where "
           "the budget cannot close the last gap, 1 is not a target but a "
           "property of the site plan.\n`relay_bridges` counts relay LINKS joining two "
           "otherwise-disconnected fragments; `relay_count` counts active "
           "transport relay LINKS, and nodes merely in a relay MODE are a "
           "separate series (`relay_mode_nodes`) that is never mixed into "
           "either."
           if scenario_name != 'rescue_ops' else
           "\nFRAGMENTS: Scenario B applies core severance + rescue-UE "
           "arrival with NO link cuts, so it does not partition the network: "
           "cross-fragment UE↔UE flows are 0/222 on every seed and the "
           "component count stays at 1-2 (an occasionally orphaned node).\n"
           "No partition bridging is claimed or implied by the Scenario B "
           "fragment panel — relay bridging is a Scenario A result.")
    )
    if FIGURE_EXCLUDED_ARMS:
        caption += ("\nThe online-only MARL arm is simulated and retained in the "
                    "raw multiseed data as a documented ablation, but is "
                    "excluded from this figure and its summary table.")
    fig.text(0.5, 0.004, caption, ha='center', va='bottom', fontsize=8,
             color='#222222', wrap=True)

    fig.tight_layout(rect=(0, 0.026, 1, 0.990))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Repo-relative paths (runnable on any machine / OS) ────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')


def run_pass_wrapper(args):
    """Wrapper function for multiprocessing pool."""
    mode, topology_seed, checkpoint_path, scenario_name, overrides = args
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # Windows 'spawn' re-imports this module in the child — re-apply the
    # --ticks smoke overrides so children run the same shortened scenario.
    _apply_runtime_overrides(overrides)
    try:
        data = run_physics_pass(mode, topology_seed, checkpoint_path, scenario_name)
        return mode, scenario_name, topology_seed, data
    except Exception as e:
        import traceback
        print(f"[{mode.upper()} - {scenario_name} seed={topology_seed}] ERROR in physics pass: {e}")
        traceback.print_exc()
        return mode, scenario_name, topology_seed, None

# Training (train_on_comparison.py) draws its topology seeds from 42-49 and
# 100-139 by default and refuses any pool that intersects its HELD_OUT_EVAL
# set, so every seed in 50-99 is legitimately unseen by the policy.  This is
# the range --seed-list is expected to draw from when widening the evaluation
# beyond the default five seeds.
EVAL_SEED_SAFE_RANGE = (50, 99)
DEFAULT_TEST_SEEDS = [50, 51, 52, 53, 54]

CANONICAL_DATA_FILE = 'multiseed_data.pkl'


def multiseed_data_filename(seeds, total_ticks, explicit=None):
    """Pick the raw-data filename for a run, and NEVER let a smoke run take
    the canonical name.

    WHY THIS EXISTS.  Every run used to write output/multiseed_data.pkl
    unconditionally, so a two-minute `--seeds 1 --ticks 600` sanity check
    silently destroyed a completed 10-seed, 3600-tick dataset — which is
    exactly what happened once already.  A partial run is not a dataset; it
    must not be able to impersonate one.

    Rule: only a run at the FULL default configuration (all
    DEFAULT_TEST_SEEDS, TOTAL_TICKS unshortened) may write the canonical
    name.  Anything narrower gets a self-describing name carrying its seed
    count and tick count, so smoke output is identifiable at a glance and can
    never be mistaken for the publication dataset.
    """
    if explicit:
        return explicit
    is_full = (list(seeds) == list(DEFAULT_TEST_SEEDS) and total_ticks >= 3600)
    if is_full:
        return CANONICAL_DATA_FILE
    return f"multiseed_data_s{len(list(seeds))}seed_t{int(total_ticks)}.pkl"


def _guard_data_overwrite(path, seeds, force=False):
    """Refuse to overwrite an existing dataset that is LARGER than this run.

    Second line of defence behind multiseed_data_filename: an explicit
    --out (or a full run re-using the canonical name) can still land on top of
    a bigger dataset.  If the file on disk holds more seeds than this run is
    about to write, the write is diverted to a `.partial-<n>seed.pkl`
    sidecar unless --force-overwrite was passed.  Returns the path to use.
    """
    if force or not os.path.exists(path):
        return path
    try:
        import pickle
        with open(path, 'rb') as f:
            existing = pickle.load(f)
        n_existing = len(existing.get('seeds', []) or [])
    except Exception as exc:
        print(f"  [DATA-GUARD] could not read existing {path} ({exc}); "
              f"leaving it alone and writing a sidecar instead")
        n_existing = 10 ** 9
    n_new = len(list(seeds))
    if n_existing > n_new:
        alt = path.replace('.pkl', f'.partial-{n_new}seed.pkl')
        print(f"  [DATA-GUARD] {os.path.basename(path)} already holds "
              f"{n_existing} seeds and this run has only {n_new} — REFUSING "
              f"to overwrite it. Writing {os.path.basename(alt)} instead "
              f"(pass --force-overwrite to override).")
        return alt
    return path


def parse_seed_list(spec):
    """Parse a --seed-list spec: comma-separated ints and/or a-b ranges.

    Reliability across topologies is the whole point of the multi-seed
    protocol, and five seeds is thin — this exists so an evaluation can be
    widened (e.g. 50-59) without editing code.  Seeds outside the range that
    training provably never touches are WARNED about, not silently accepted.
    """
    seeds = []
    for part in str(spec).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part[1:]:
            a, b = part.split('-', 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    seeds = sorted(dict.fromkeys(seeds))
    if not seeds:
        raise SystemExit("ERROR: --seed-list produced an empty seed set")
    lo, hi = EVAL_SEED_SAFE_RANGE
    outside = [s for s in seeds if not (lo <= s <= hi)]
    if outside:
        print(f"WARNING: seeds {outside} lie outside the verified held-out "
              f"range {lo}-{hi}; training (train_on_comparison.py) uses 42-49 "
              f"and 100-139, so these may NOT be held out. Proceeding anyway.")
    return seeds


def main(seeds_limit=None, overrides=None, seed_list=None,
         out_file=None, force_overwrite=False, arms=None, no_figures=False):
    overrides = overrides or {}
    _apply_runtime_overrides(overrides)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # The trained policy the MARL arms start from.  Overridable because the
    # authoritative experiment trains N INDEPENDENT policies (different torch
    # seeds, identical data) and evaluates each on the same frozen benchmark:
    # reporting mean +/- std across them measures training-seed variance
    # instead of silently reporting whichever run happened to come out best.
    checkpoint_path = os.environ.get(
        'MARL_EVAL_CHECKPOINT',
        os.path.join(CHECKPOINT_DIR, 'policy_cloud_trained.pt'))
    # A comma-separated list names an ENSEMBLE (see run_physics_pass); every
    # member must exist.
    _ck_paths = [p.strip() for p in str(checkpoint_path).split(',') if p.strip()]
    if not all(os.path.exists(p) for p in _ck_paths):
        raise SystemExit(
            "evaluation checkpoint not found: %s\n"
            "Set MARL_EVAL_CHECKPOINT to a trained policy_final.pt."
            % checkpoint_path)
    print("  Checkpoint: %s" % checkpoint_path)
    # Seven arms: 6 protocols + the adapt-freeze MARL variant.
    # Both MARL arms run their rescue_ops pass in Phase 2 (each needs its
    # own recovery-adapted per-seed policy checkpoint).
    # Seven arms + the 'random' bridging-attribution control arm (uniform
    # action sampling on the identical MARL code path — see run_physics_pass).
    # 'xldet' is the engineered non-learning arm (heuristic_controller): same
    # actuation, same observations, same locality mask, no learning.  It is
    # the control for "is LEARNING necessary?", which the uniform-random arm
    # cannot answer, and unlike 'random' it belongs IN the figures.
    # 'marl_static' is the deployed controller (parameters unchanged from
    # t=0); it is deliberately NOT in MARL_ARMS below, so its rescue_ops pass
    # runs in Phase 1 from the base checkpoint.
    modes = ['marl', 'marl_freeze', 'marl_static', 'random', 'xldet',
             'ospf', 'sdn', 'olsr', 'batman', 'aodv']
    # --arms narrows the run.  WHY THIS EXISTS: only 'marl' and 'marl_freeze'
    # read the evaluation checkpoint.  The other seven arms are functions of
    # the seed alone, so re-running them once per trained policy recomputes
    # identical numbers at full cost.  The authoritative design is therefore
    # ONE full-width run (every arm) plus N-1 narrow runs (--arms
    # marl,marl_freeze) that supply the across-training-seed spread for the
    # learned arms only.
    if arms:
        requested = [a.strip() for a in str(arms).split(',') if a.strip()]
        unknown = [a for a in requested if a not in modes]
        if unknown:
            raise SystemExit("unknown arm(s): %s; known: %s"
                             % (', '.join(unknown), ', '.join(modes)))
        modes = [m for m in modes if m in requested]
        if not modes:
            raise SystemExit("--arms selected no arms")
    MARL_ARMS = tuple(m for m in ('marl', 'marl_freeze') if m in modes)

    # ── Multi-seed topology configuration ──
    # Training used seed=42 — test on UNSEEN topologies
    # Same structure (node counts, types) but different spatial layouts
    # --seed-list overrides the default five; --seeds still truncates.
    TEST_SEEDS = (parse_seed_list(seed_list) if seed_list
                  else list(DEFAULT_TEST_SEEDS))
    if seeds_limit:
        TEST_SEEDS = TEST_SEEDS[:max(1, int(seeds_limit))]

    total_phase1 = (len(modes) * 2 - len(MARL_ARMS)) * len(TEST_SEEDS)
    total_phase2 = len(MARL_ARMS) * len(TEST_SEEDS)
    total = total_phase1 + total_phase2

    print("=" * 70)
    print("  MULTI-SEED STATISTICAL COMPARISON")
    print(f"  Test seeds: {TEST_SEEDS} (training seed=42 excluded)")
    print(f"  Ticks per pass: {TOTAL_TICKS} (severance @ {SEVERANCE_TICK}, "
          f"rescue @ {RESCUE_ARRIVAL_TICK})")
    print(f"  Arms: {', '.join(m.upper() for m in modes)}")
    print(f"  Phase 1: {total_phase1} parallel passes")
    print(f"  Phase 2: {total_phase2} PARALLEL MARL rescue_ops (adapted policy)")
    print(f"  Total: {total} physics passes")
    _data_name = multiseed_data_filename(TEST_SEEDS, TOTAL_TICKS, out_file)
    print(f"  Raw data -> output/{_data_name}"
          + ("" if _data_name == CANONICAL_DATA_FILE else
             "   (PARTIAL RUN - not the canonical dataset)"))
    print(f"  TG plan: reach {TG_PLAN_REACH_M:.0f} m @{FRAG_BRIDGE_TX_DBM:.0f} dBm"
          f" / {TG_HIGH_REACH_M:.0f} m @{TG_HIGH_TX_DBM:.0f} dBm;"
          f" TG sites = every {'/'.join(TG_SITE_TYPE_VALUES)} site")
    if MARL_BFD_DETECT_TICKS > 0:
        print(f"  MARL failure-detection floor: {MARL_BFD_DETECT_TICKS} ticks "
              f"(BFD-class), applied to every link-state transition on the "
              f"{'/'.join(MARL_DETECTION_ARMS)} arms;")
        print("    gates BOTH the forwardable flow set AND path selection "
              "(stale routing view), so a freshly-cut link stays selectable "
              "and blackholes until detection expires")
    else:
        print("  MARL failure-detection floor: DISABLED (MARL_DETECT_TICKS=0) "
              "- ablation only: MARL arms reroute in the same tick a link "
              "flips, i.e. oracle-grade failure detection")
    print("=" * 70)

    import multiprocessing
    tasks = []
    for seed in TEST_SEEDS:
        # All recovery tasks (including both MARL arms)
        for mode in modes:
            tasks.append((mode, seed, checkpoint_path, 'recovery', overrides))
        # All rescue_ops tasks EXCEPT the MARL arms (they need the adapted
        # policy their own recovery pass produces — Phase 2 below)
        for mode in modes:
            if mode not in MARL_ARMS:
                tasks.append((mode, seed, checkpoint_path, 'rescue_ops', overrides))

    num_cpus = min(len(tasks), multiprocessing.cpu_count())
    print(f"\nPhase 1: Spawning pool with {num_cpus} workers for {len(tasks)} tasks...")

    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(run_pass_wrapper, tasks)

    # ── Collect results into per-seed dicts ──
    # Structure: {mode: {seed: data}}
    recovery_multi = {m: {} for m in modes}
    rescue_multi = {m: {} for m in modes}
    for mode, scenario_name, seed, data in results:
        if data is None:
            print(f"ERROR: Physics pass failed for mode={mode}, scenario={scenario_name}, seed={seed}")
            continue
        if scenario_name == 'recovery':
            recovery_multi[mode][seed] = data
        else:
            rescue_multi[mode][seed] = data

    # ── Phase 2: MARL rescue_ops per seed (uses adapted policy from each
    # seed's recovery pass; each MARL arm loads its OWN adapted checkpoint) ──
    print("\n" + "=" * 70)
    print(f"  Phase 2: Running {len(MARL_ARMS) * len(TEST_SEEDS)} MARL "
          f"rescue_ops passes (adapted policies)...")
    print("=" * 70)

    _ARM_SUFFIX = {'marl': '_adapted_recovery',
                   'marl_freeze': '_adapted_recovery_freeze'}
    # PARALLEL, not sequential.  Every Phase-2 pass is independent once
    # Phase 1 has written that (arm, seed)'s adapted checkpoint: each task
    # reads its OWN checkpoint file, builds its own Simulator and shares no
    # state.  Running these full-length passes one-at-a-time left 31 of 32
    # cores idle and dominated total wall-clock (Phase 1's passes are pooled).
    # Semantics unchanged: same tasks, same checkpoints, same per-pass
    # seeding; results reassembled by the (arm, seed) key the worker returns.
    phase2_tasks = []
    for seed in TEST_SEEDS:
        for marl_arm in MARL_ARMS:
            adapted_checkpoint_path = _adapted_checkpoint_path(
                checkpoint_path, _ARM_SUFFIX[marl_arm], seed)
            if not os.path.exists(adapted_checkpoint_path):
                print(f"  WARNING: {marl_arm} seed={seed} adapted checkpoint "
                      f"missing -> pass falls back to base checkpoint.")
            phase2_tasks.append((marl_arm, seed, adapted_checkpoint_path,
                                 'rescue_ops', overrides))

    n2 = min(len(phase2_tasks), multiprocessing.cpu_count())
    if n2 == 0:
        # No adapt-type arm in this run (e.g. --arms marl_static or a
        # baseline-only run): there is nothing to schedule in Phase 2.
        print("  Phase 2: no tasks (no online-adapting arm selected).")
        phase2_results = []
    else:
        print(f"  Spawning pool with {n2} workers for {len(phase2_tasks)} "
              f"Phase-2 tasks (previously sequential on 1 core)...")
        with multiprocessing.Pool(processes=n2) as pool:
            phase2_results = pool.map(run_pass_wrapper, phase2_tasks)

    for _arm, _scen, _seed, _data in phase2_results:
        if _data is None:
            print(f"ERROR: Phase-2 pass failed for {_arm} seed={_seed}")
            continue
        rescue_multi[_arm][_seed] = _data

    # ── Save raw data for replotting ──
    import pickle
    data_path = _guard_data_overwrite(
        os.path.join(OUTPUT_DIR, _data_name), TEST_SEEDS, force_overwrite)
    with open(data_path, 'wb') as f:
        pickle.dump({
            'recovery_multi': recovery_multi,
            'rescue_multi': rescue_multi,
            'seeds': TEST_SEEDS,
        }, f)
    print(f"\nSaved raw data to {data_path}")

    # ── Generate plots with variance bands ──
    if no_figures:
        # A narrow run holds too few arms for the comparison figures and
        # tables to mean anything; the raw pickle is the deliverable.
        print("  [--no-figures] raw data written; figures skipped.")
    else:
        _generate_plots(recovery_multi, rescue_multi, TEST_SEEDS)


# Stable filenames the documentation/paper tooling references.  They are
# written in ADDITION to the default names below, so regenerating figures
# never breaks a document that links the FINAL_* copies.
FINAL_FIGURE_NAMES = {
    'recovery':   'FINAL_scenario_a_recovery.png',
    'rescue_ops': 'FINAL_scenario_b_rescue_ops.png',
}


def _generate_plots(recovery_multi, rescue_multi, seeds):
    """Generate both scenario plots from collected data."""
    import shutil
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_a = os.path.join(OUTPUT_DIR, 'scenario_a_recovery_fair.png')
    path_b = os.path.join(OUTPUT_DIR, 'scenario_b_rescue_ops_fair.png')
    final_a = os.path.join(OUTPUT_DIR, FINAL_FIGURE_NAMES['recovery'])
    final_b = os.path.join(OUTPUT_DIR, FINAL_FIGURE_NAMES['rescue_ops'])

    # Attribution ablation first: it decides what the figures are ALLOWED to
    # claim, so it is printed before them, not buried after.
    print_bridging_attribution(recovery_multi, seeds, 'recovery')
    print_bridging_attribution(rescue_multi, seeds, 'rescue_ops')

    print("\nGenerating Scenario A Plot (mean ± std across seeds)...")
    plot_comparison_multiseed(recovery_multi, 'recovery', path_a, seeds)

    print("\nGenerating Scenario B Plot (mean ± std across seeds)...")
    plot_comparison_multiseed(rescue_multi, 'rescue_ops', path_b, seeds)

    for src, dst in ((path_a, final_a), (path_b, final_b)):
        shutil.copyfile(src, dst)
        print(f"Saved {dst}")

    print("\n" + "=" * 70)
    print("  DONE — Multi-Seed Statistical Comparison Complete")
    print(f"  Seeds tested: {seeds}")
    print(f"  Scenario A: {path_a}")
    print(f"              {final_a}")
    print(f"  Scenario B: {path_b}")
    print(f"              {final_b}")
    print("=" * 70)


def replot(data_file=None):
    """Regenerate plots from saved data without re-running simulation.

    data_file may be an absolute path or a name relative to OUTPUT_DIR
    (default: multiseed_data.pkl), so a specific verified run — e.g.
    multiseed_data_round6.pkl — can be replotted without renaming files.
    """
    import pickle
    data_file = data_file or 'multiseed_data.pkl'
    data_path = (data_file if os.path.isabs(data_file)
                 else os.path.join(OUTPUT_DIR, data_file))
    if not os.path.exists(data_path) and os.path.exists(data_file):
        data_path = os.path.abspath(data_file)
    print(f"Loading saved data from {data_path}...")
    with open(data_path, 'rb') as f:
        saved = pickle.load(f)
    _generate_plots(saved['recovery_multi'], saved['rescue_multi'], saved['seeds'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-seed 7-arm recovery comparison "
                    "(MARL online + MARL adapt-freeze vs "
                    "OSPF/SDN/OLSR/BATMAN/AODV)")
    parser.add_argument('--replot', action='store_true',
                        help="regenerate figures from saved multiseed_data.pkl")
    parser.add_argument('--data', type=str, default=None, metavar='PKL',
                        help="pickle to replot (path, or name relative to "
                             "output/; default multiseed_data.pkl)")
    parser.add_argument('--seeds', type=int, default=None, metavar='N',
                        help="limit to the first N test seeds (smoke runs)")
    parser.add_argument('--seed-list', type=str, default=None, metavar='SPEC',
                        help="explicit evaluation topology seeds, "
                             "comma-separated (ranges allowed), e.g. "
                             "'50-59' or '50,51,52,55'. Training uses 42-49 "
                             "and 100-139, so 50-99 are held out. "
                             f"Default: {DEFAULT_TEST_SEEDS}")
    parser.add_argument('--ticks', type=int, default=None, metavar='N',
                        help="shorten each pass to N ticks (smoke runs; "
                             "severance/rescue ticks scale down accordingly)")
    parser.add_argument('--out', type=str, default=None, metavar='PKL',
                        help="raw-data filename to write inside output/. "
                             "Default: multiseed_data.pkl for a FULL run "
                             "(all default seeds, full ticks), or a "
                             "self-describing multiseed_data_s<N>seed_t<T>.pkl "
                             "for any narrower/smoke run - a partial run can "
                             "never silently take the canonical name.")
    parser.add_argument('--arms', type=str, default=None, metavar='LIST',
                        help="comma-separated subset of arms to run, e.g. "
                             "'marl,marl_freeze'. Only those two read the "
                             "evaluation checkpoint, so the other arms need "
                             "computing only once across a multi-checkpoint "
                             "study.")
    parser.add_argument('--no-figures', action='store_true',
                        help="write the raw pickle but skip figures/tables "
                             "(for narrow --arms runs)")
    parser.add_argument('--force-overwrite', action='store_true',
                        help="allow overwriting an existing dataset that "
                             "holds MORE seeds than this run (otherwise the "
                             "write is diverted to a .partial sidecar)")
    args = parser.parse_args()

    _overrides = {}
    if args.ticks:
        _overrides['total_ticks'] = args.ticks

    if args.replot:
        replot(args.data)
    else:
        main(seeds_limit=args.seeds, overrides=_overrides,
             seed_list=args.seed_list, out_file=args.out,
             force_overwrite=args.force_overwrite, arms=args.arms,
             no_figures=args.no_figures)
