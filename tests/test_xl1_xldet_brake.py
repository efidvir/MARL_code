# -*- coding: utf-8 -*-
"""XL-1 -- XL-DET's interference brake may stop a climb but may not hold a
site below its own half-rate cliff (docs/DECISIONS.md D-XL.1).

Since WP5 the neighbour cache XL-DET reads is refreshed every postcard
period, so the brake (`_icic_braked`: a loud neighbour, or the site's own
SINR already falling) no longer expires.  After a fleet-wide step-down every
site braked every other one and the fleet stayed at low power for the rest
of a Scenario-B pass.  `tx_head` now climbs whenever the site's own SINR is
below SINR_TARGET_DB - HALF_RATE_MARGIN_DB (12 dB), braked or not, and keeps
the brake between that floor and the target.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _PS:
    def __init__(self, sinr, p=23.0):
        self.sinr_average = sinr
        self.tx_power_dbm = p


LOUD = {'tx': 0.9, 'interf': 1.0, 'sinr': 0.2}     # brakes on both counts


def _tx(sinr, nb=LOUD, own_drop_db=0.0):
    import sixg_sim.heuristic_controller as H
    st = H._State()                                  # no previous step: no give-back
    st.nb = dict(nb) if nb is not None else None
    return H.tx_head(_PS(sinr), st, own_drop_db)


def test_the_brake_cannot_hold_a_site_below_its_half_rate_cliff():
    import sixg_sim.heuristic_controller as H
    assert H.ICIC_FLOOR_DB == H.SINR_TARGET_DB - H.HALF_RATE_MARGIN_DB == 12.0
    cases = [
        # (own SINR dB, neighbour cache, own drop dB) -> expected head
        ((3.0, LOUD, 0.0), H.TX_UP),      # far below the cliff, loud neighbours: climbs
        ((11.9, LOUD, 0.0), H.TX_UP),     # just below the cliff: climbs
        ((8.0, LOUD, 6.0), H.TX_UP),      # own SINR falling too: still climbs below the cliff
        ((12.0, LOUD, 0.0), H.TX_HOLD),   # at the cliff the brake holds again
        ((15.0, LOUD, 0.0), H.TX_HOLD),   # between cliff and target: the brake holds
        ((15.0, None, 0.0), H.TX_UP),     # no neighbour cache: fails open, as before
        ((15.0, LOUD, 4.0), H.TX_HOLD),   # own drop alone brakes above the cliff, as before
        ((20.0, LOUD, 0.0), H.TX_HOLD),   # above the target: nothing to do
        ((30.0, LOUD, 0.0), H.TX_DOWN),   # ample: steps down, as before
    ]
    for (sinr, nb, drop), want in cases:
        got = _tx(sinr, nb, drop)
        assert got == want, ('sinr %.1f nb %s drop %.1f: head %d, expected %d'
                             % (sinr, 'loud' if nb else None, drop, got, want))
    assert len(cases) == 9


def test_a_braked_fleet_climbs_back_from_a_step_down():
    """A closed loop of the real `tx_head` on a toy fleet: four sites that
    hear each other loudly (the brake is engaged on every tick) start at
    the power floor, where each one's SINR is noise-limited at 6 dB, and
    gain 1 dB of SINR per dB of power up to an interference ceiling of
    16 dB.  The pre-XL-1 rule held all four at the floor forever; the XL-1
    rule climbs them past the cliff, where the brake takes over again."""
    import sixg_sim.heuristic_controller as H
    from sixg_sim.phy_mac_state import TX_POWER_STEPS_DB
    sites = [{'p': 5.0, 'st': H._State()} for _ in range(4)]
    for s in sites:
        s['st'].nb = dict(LOUD)
    for _tick in range(40):
        for s in sites:
            g = min(16.0, 6.0 + (s['p'] - 5.0))
            head = H.tx_head(_PS(g, s['p']), s['st'], 0.0)
            s['p'] = max(5.0, min(33.0, s['p'] + TX_POWER_STEPS_DB[head]))
            s['st'].nb = dict(LOUD)               # a postcard every tick: never stale
    final = [min(16.0, 6.0 + (s['p'] - 5.0)) for s in sites]
    assert all(g >= H.ICIC_FLOOR_DB for g in final), final
    assert all(s['p'] > 5.0 for s in sites), [s['p'] for s in sites]
