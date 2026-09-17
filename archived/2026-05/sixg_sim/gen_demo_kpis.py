"""
Generate a synthetic learning_kpis.json that demonstrates the expected
convergence shape of the 6G MARL disaster-recovery scenario, then
render the convergence plot from it.

Run: python sixg_sim/gen_demo_kpis.py
"""
import json, math, random, os
random.seed(42)

N_EP    = 10
TICKS   = 800
SEV_T   = 500    # disaster severs Core/CU at tick 500
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

ticks    = []
episodes = []

# Per-episode "learned" UE connectivity post severance rises from ~8% to ~62%
def ep_ue_conn_curve(ep, tick):
    """Models: random policy early, improving relay adoption later."""
    learn_frac = (ep - 1) / max(1, N_EP - 1)     # 0 → 1 across episodes
    if tick < SEV_T:
        # Before severance: full connectivity via Core
        return random.gauss(0.97, 0.01)
    else:
        # After severance: random policy ~8%, trained policy ~65%
        post_base   = 0.08 + 0.57 * learn_frac
        ramp        = min(1.0, (tick - SEV_T) / 200.0)  # ramp up over 200 ticks
        return min(1.0, max(0, random.gauss(post_base * ramp, 0.04)))

def ep_transport_relays(ep, tick):
    learn_frac = (ep - 1) / max(1, N_EP - 1)
    if tick < SEV_T:
        return 0
    ramp = min(1.0, (tick - SEV_T) / 150.0)
    base = 2 + int(28 * learn_frac * ramp)
    return max(0, base + random.randint(-2, 2))

def ep_transport_links(ep, tick):
    return max(0, ep_transport_relays(ep, tick) - random.randint(0, 3))

def ep_reward(ep, tick):
    learn_frac = (ep - 1) / max(1, N_EP - 1)
    if tick < SEV_T:
        return random.gauss(4.8 + 0.3 * learn_frac, 0.3)
    else:
        ramp = min(1.0, (tick - SEV_T) / 200.0)
        return random.gauss(1.5 + 4.0 * learn_frac * ramp, 0.5)

# MAPPO losses: decrease over episodes
def pol_loss(ep):
    return max(0.001, 0.35 * math.exp(-0.25 * (ep - 1)) + random.gauss(0, 0.01))
def val_loss(ep):
    return max(0.002, 0.80 * math.exp(-0.22 * (ep - 1)) + random.gauss(0, 0.02))
def entropy(ep):
    return max(0.01,  0.85 * math.exp(-0.12 * (ep - 1)) + random.gauss(0, 0.03))

# Build tick records
for ep in range(1, N_EP + 1):
    ep_ticks = []
    for t in range(TICKS):
        island = t >= SEV_T
        ue     = ep_ue_conn_curve(ep, t)
        iab_r  = ep_transport_relays(ep, t)
        iab_l  = ep_transport_links(ep, t)
        rw     = ep_reward(ep, t)
        pl     = pol_loss(ep) if t % 50 == 0 else float('nan')
        vl     = val_loss(ep) if t % 50 == 0 else float('nan')
        en     = entropy(ep)  if t % 50 == 0 else float('nan')
        rec = {
            'tick': t, 'episode': ep, 'island_mode': island,
            'ue_conn_frac': round(ue, 4),
            'transport_relay_count': iab_r, 'transport_link_count': iab_l,
            'reward': round(rw, 4),
            'policy_loss': None if math.isnan(pl) else round(pl, 5),
            'value_loss':  None if math.isnan(vl) else round(vl, 5),
            'entropy':     None if math.isnan(en) else round(en, 5),
        }
        ticks.append(rec)
        ep_ticks.append(rec)

    pre_ticks  = [r for r in ep_ticks if not r['island_mode']]
    post_ticks = [r for r in ep_ticks if r['island_mode']]
    def _mean(lst): return sum(lst)/len(lst) if lst else 0.0

    pl_ep = pol_loss(ep)
    vl_ep = val_loss(ep)
    en_ep = entropy(ep)
    episodes.append({
        'episode':             ep,
        'duration_ticks':      TICKS,
        'severance_tick':      SEV_T,
        'pre_sev_ue_conn':     _mean([r['ue_conn_frac'] for r in pre_ticks]),
        'post_sev_ue_conn':    _mean([r['ue_conn_frac'] for r in post_ticks]),
        'post_sev_transport_relay':  _mean([r['transport_relay_count'] for r in post_ticks]),
        'post_sev_transport_links':  _mean([r['transport_link_count'] for r in post_ticks]),
        'peak_transport_relay':      max((r['transport_relay_count'] for r in post_ticks), default=0),
        'ep_reward_mean':      _mean([r['reward'] for r in ep_ticks]),
        'final_policy_loss':   round(pl_ep, 5),
        'final_value_loss':    round(vl_ep, 5),
        'final_entropy':       round(en_ep, 5),
    })

out_json = os.path.join(OUT_DIR, "learning_kpis_demo.json")
with open(out_json, 'w') as f:
    json.dump({'ticks': ticks, 'episodes': episodes}, f, indent=2)
print(f"Wrote {len(ticks)} tick records, {len(episodes)} episodes -> {out_json}")

# Render the convergence plot
from sixg_sim.plot_learning import plot_convergence
out_png = os.path.join(OUT_DIR, "marl_convergence_demo.png")
plot_convergence(out_json, out_png)
print(f"Plot saved → {out_png}")
