import json, math

with open('output/learning_kpis.json') as f:
    d = json.load(f)

eps = d.get('episodes', [])
tks = d.get('ticks', [])

print(f'Episodes: {len(eps)}, Ticks: {len(tks)}')
print()
print('Per-episode summary:')
for e in eps:
    ep  = e['episode']
    sc  = e.get('scenario_type','?')[:8]
    rew = e.get('ep_reward_mean', float('nan'))
    pl  = e.get('final_policy_loss', float('nan'))
    ent = e.get('final_entropy', float('nan'))
    iab = e.get('post_sev_transport_relay', 0)
    ue  = e.get('post_sev_ue_conn', 0.0)
    print(f'  Ep{ep:2d} [{sc:8s}]  reward={rew:8.2f}  policy_loss={pl:.4f}  entropy={ent:.3f}  transport_relay={iab:3d}  ue_conn={ue:.3f}')

print()
tick_keys = set()
for t in tks[:20]:
    tick_keys.update(t.keys())
print('Tick keys:', sorted(tick_keys))

print()
print('First 3 ticks:')
for t in tks[:3]:
    print(' ', t)
print('Last 3 ticks:')
for t in tks[-3:]:
    print(' ', t)

# Check island-mode distribution
island_ticks = sum(1 for t in tks if t.get('island_mode', False))
print(f'\nIsland-mode ticks: {island_ticks}/{len(tks)} ({100*island_ticks/max(1,len(tks)):.1f}%)')

# UE connectivity across ticks
ue_fracs = [t.get('ue_conn_frac', t.get('reachable_frac', -1)) for t in tks]
valid = [v for v in ue_fracs if v >= 0]
if valid:
    print(f'UE conn frac: min={min(valid):.3f}  max={max(valid):.3f}  mean={sum(valid)/len(valid):.3f}')
else:
    print('UE conn frac: KEY MISSING from tick records')
    # show all available keys
    print('  Available tick keys:', sorted(tick_keys))

# Reward range
rewards = [t.get('reward', float('nan')) for t in tks]
valid_r = [r for r in rewards if not math.isnan(r)]
if valid_r:
    print(f'Tick rewards: min={min(valid_r):.2f}  max={max(valid_r):.2f}  last 50 mean={sum(valid_r[-50:])/50:.2f}')
