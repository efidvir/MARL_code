import re

with open("c:/Users/efid/OneDrive - Ceragon/UNITY-6G/WP4/MARL_code/sixg_sim/agent.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update loss calculation in evaluate_actions and update_policy
# Replace:
# loss = -(log_probs * returns_t).mean()
# With:
# loss = -(log_probs * returns_t).mean() * 100.0
# Actually, the user wants entropy bonus and scaling.
replacer1 = """            # ── Discrete log-probs ─────────────────────────────────────
            log_probs = torch.zeros(len(self.experiences))
            entropy = torch.zeros(len(self.experiences))
            for head in DISCRETE_HEADS:
                dist = Categorical(logits=logits[head])
                log_probs = log_probs + dist.log_prob(idx_tensors[head])
                entropy = entropy + dist.entropy()

            # ── PRB allocation: cross-entropy vs. target fractions ─────
            prb_log_soft = F.log_softmax(logits["prb"], dim=-1)   # (T, 3)
            prb_ce       = -(prb_target * prb_log_soft).sum(dim=-1)  # (T,)
            log_probs    = log_probs - prb_ce   # include PRB in total log-prob

            # Enhanced Actor Loss: Unnormalized scale + entropy for exploration
            loss = -(log_probs * returns_t).mean() * 100.0 - 0.5 * entropy.mean()"""

pattern1 = re.compile(r"            # ── Discrete log-probs ─────────────────────────────────────\n            log_probs = torch\.zeros\(len\(self\.experiences\)\)\n            for head in DISCRETE_HEADS:\n                log_probs = log_probs \+ Categorical\(\n                    logits=logits\[head\]\n                \)\.log_prob\(idx_tensors\[head\]\)\n\n            # ── PRB allocation: cross-entropy vs\. target fractions ─────\n            prb_log_soft = F\.log_softmax\(logits\[\"prb\"\], dim=-1\)   # \(T, 3\)\n            prb_ce       = -\(prb_target \* prb_log_soft\)\.sum\(dim=-1\)  # \(T,\)\n            log_probs    = log_probs - prb_ce   # include PRB in total log-prob\n\n            loss = -\(log_probs \* returns_t\)\.mean\(\)")

content, count1 = pattern1.subn(replacer1, content)
print(f"Replaced loss calculation: {count1} times")

# 2. Update rate limit in compute_action
# Replace:
# if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 10:
# With:
# if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 2:
pattern2 = re.compile(r"if postcard_do and \(observation\.current_tick - self\.last_postcard_tick\) >= 10:")
content, count2 = pattern2.subn(r"if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 2:", content)
print(f"Replaced compute_action rate limit: {count2} times")

# 3. Update rate limit in action_from_logits_row
# Replace:
# if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 10:
# With:
# if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 2:
pattern3 = re.compile(r"if postcard_do and \(observation\.current_tick - self\.last_postcard_tick\) >= 10:")
content, count3 = pattern3.subn(r"if postcard_do and (observation.current_tick - self.last_postcard_tick) >= 2:", content)
print(f"Replaced action_from_logits_row rate limit: {count3} times")

with open("c:/Users/efid/OneDrive - Ceragon/UNITY-6G/WP4/MARL_code/sixg_sim/agent.py", "w", encoding="utf-8") as f:
    f.write(content)
