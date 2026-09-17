import re

with open("c:/Users/efid/OneDrive - Ceragon/UNITY-6G/WP4/MARL_code/sixg_sim/simulation.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Fix apply_marl_policy integers
# Replace: mode = agent.last_actions[traffic_class].get('admission_mode', 'ADMIT')
#        : if hasattr(mode, 'upper'): mode = mode.upper()
#        : if mode == 'THROTTLE': bottleneck = min(bottleneck, amount * 0.5)
#        : elif mode == 'HOLD': return 0.0
# With correct enum value checks (0=ADMIT, 1=THROTTLE, 2=HOLD)
def replacer1(match):
    return """                    mode_val = agent.last_actions[traffic_class].get('admission_mode', 0)
                    if mode_val == 1 or mode_val == 'THROTTLE': bottleneck = min(bottleneck, amount * 0.5)
                    elif mode_val == 2 or mode_val == 'HOLD': return 0.0"""

pattern1 = re.compile(r"                    mode = agent\.last_actions\[traffic_class\]\.get\('admission_mode', 'ADMIT'\)\n                    if hasattr\(mode, 'upper'\): mode = mode\.upper\(\)\n                    if mode == 'THROTTLE': bottleneck = min\(bottleneck, amount \* 0\.5\)\n                    elif mode == 'HOLD': return 0\.0")
content, count = pattern1.subn(replacer1, content)
print(f"Replaced apply_marl_policy: {count} times")

# 2. Fix base_ue_to_ue_traffic
# Replace: base_ue_to_ue_traffic = 0.0 if self.island_mode else 8.0
# With: base_ue_to_ue_traffic = 5.0 if self.island_mode else 8.0
pattern2 = re.compile(r"base_ue_to_ue_traffic = 0\.0 if self\.island_mode else 8\.0")
content, count = pattern2.subn(r"base_ue_to_ue_traffic = 5.0 if self.island_mode else 8.0", content)
print(f"Replaced base_ue_to_ue_traffic: {count} times")

# 3. Clean up the DEBUG prints
content = re.sub(r' +print\(f"DEBUG:.*?"\)\n', '', content)

with open("c:/Users/efid/OneDrive - Ceragon/UNITY-6G/WP4/MARL_code/sixg_sim/simulation.py", "w", encoding="utf-8") as f:
    f.write(content)
