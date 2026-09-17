import json, math, statistics
from collections import Counter

d = json.load(open('learning_kpis.json'))
eps = sorted(d.get('episodes', []), key=lambda e: e['episode'])
tks = d.get('ticks', [])

rews  = [float(e.get('ep_reward_mean', 'nan')) for e in eps]
pls   = [float(e.get('final_policy_loss', 'nan')) for e in eps]
vls   = [float(e.get('final_value_loss', 'nan')) for e in eps]
ents  = [float(e.get('final_entropy', 'nan')) for e in eps]
ues   = [float(e.get('post_sev_ue_conn', 0)) for e in eps]
iabs  = [float(e.get('post_sev_iab_relay', 0)) for e in eps]

ue_nz = [v for v in ues if v > 0]
isl   = [t for t in tks if t.get('island_mode', False)]
sc    = Counter(e.get('scenario_type', '?') for e in eps)

print(f"Total episodes : {len(eps)}")
print(f"Total ticks    : {len(tks)}")
print(f"Island ticks   : {len(isl)} ({100*len(isl)//max(1,len(tks))}%)")
print(f"Scenario mix   : {dict(sc)}")
print()
print(f"Episode reward : first-5={statistics.mean(rews[:5]):.2f}  last-5={statistics.mean(rews[-5:]):.2f}  delta={statistics.mean(rews[-5:])-statistics.mean(rews[:5]):.2f}")
print(f"Policy loss    : {pls[0]:.4f} -> {pls[-1]:.4f}  ({100*(pls[0]-pls[-1])/max(abs(pls[0]),1e-9):.0f}% reduction)")
print(f"Critic loss    : first-3 avg={statistics.mean(vls[:3]):.2f}  last-3 avg={statistics.mean(vls[-3:]):.4f}")
print(f"Entropy        : {ents[0]:.3f} -> {ents[-1]:.3f}")
print(f"UE connectivity: n={len(ue_nz)} episodes non-zero, mean={statistics.mean(ue_nz):.1%}, max={max(ue_nz):.1%}")
print(f"IAB relays     : mean={statistics.mean(iabs):.1f}, max={max(iabs):.1f}")
print()
top5 = sorted(eps, key=lambda e: float(e.get('ep_reward_mean', -999)), reverse=True)[:5]
print("Top-5 episodes by reward:")
for e in top5:
    print(f"  Ep{e['episode']:3d} [{e.get('scenario_type'):12s}]  reward={float(e.get('ep_reward_mean',0)):6.1f}  UE={float(e.get('post_sev_ue_conn',0))*100:5.1f}%  IAB={round(float(e.get('post_sev_iab_relay',0)),1)}")
