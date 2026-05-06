"""
LearningTracker — per-tick and per-episode KPI recorder for MARL convergence analysis.

Tracked metrics (all computed from sim state every tick):
  - island_mode          : bool  — True once core is severed
  - ue_conn_frac         : [0,1] — fraction of UE pairs successfully routed THIS tick
  - transport_relay_count      : int   — infra nodes currently in TRANSPORT_RELAY relay mode
  - transport_link_count       : int   — active transport wireless backhaul links
  - island_node_count    : int   — live infra nodes inside the island
  - reward               : float — mean team reward this tick
  - policy_loss          : float — latest PPO policy loss (NaN between updates)
  - value_loss           : float — latest PPO value/critic loss
  - entropy              : float — latest PPO entropy

Per-episode aggregates (saved at episode end):
  - ep_reward_mean       : mean reward over full episode
  - post_sev_ue_conn     : mean UE connectivity post-severance (0 if never severed)
  - post_sev_transport_relay   : mean transport relay count post-severance
  - post_sev_transport_links   : mean transport link count post-severance
  - peak_transport_relay       : max transport relay count in post-severance phase
  - severance_tick       : tick at which island mode first triggered (-1 if never)
  - duration_ticks       : episode length
  - final_policy_loss    : last policy loss value
  - final_value_loss     : last value loss value
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TickRecord:
    tick:            int
    episode:         int
    island_mode:     bool
    ue_conn_frac:    float
    transport_relay_count: int
    transport_link_count:  int
    reward:          float
    policy_loss:     float   # NaN between updates
    value_loss:      float
    entropy:         float


@dataclass
class EpisodeRecord:
    episode:             int
    duration_ticks:      int
    severance_tick:      int       # -1 = never severed
    scenario_type:       str       # full_core | partial_core | zone_loss | cascading
    # pre-severance avg (ticks before island)
    pre_sev_ue_conn:     float
    # post-severance avg (ticks in island mode)
    post_sev_ue_conn:    float
    post_sev_transport_relay:  float
    post_sev_transport_links:  float
    peak_transport_relay:      int
    # overall
    ep_reward_mean:      float
    final_policy_loss:   float
    final_value_loss:    float
    final_entropy:       float


class LearningTracker:
    """Accumulates KPIs tick-by-tick and produces per-episode summaries."""

    def __init__(self):
        self.ticks:    List[TickRecord]   = []
        self.episodes: List[EpisodeRecord] = []

        # mutable state for the current episode
        self._ep_ticks:        List[TickRecord] = []
        self._last_policy_loss = float('nan')
        self._last_value_loss  = float('nan')
        self._last_entropy     = float('nan')

    # ── Per-tick recording ─────────────────────────────────────────────────────

    def record_tick(self,
                    tick:            int,
                    episode:         int,
                    island_mode:     bool,
                    ue_conn_frac:    float,
                    transport_relay_count: int,
                    transport_link_count:  int,
                    reward:          float,
                    policy_loss:     float = float('nan'),
                    value_loss:      float = float('nan'),
                    entropy:         float = float('nan')) -> None:
        # Keep latest non-NaN losses for reuse
        if not math.isnan(policy_loss):
            self._last_policy_loss = policy_loss
        if not math.isnan(value_loss):
            self._last_value_loss  = value_loss
        if not math.isnan(entropy):
            self._last_entropy     = entropy

        rec = TickRecord(
            tick=tick, episode=episode, island_mode=island_mode,
            ue_conn_frac=ue_conn_frac, transport_relay_count=transport_relay_count,
            transport_link_count=transport_link_count, reward=reward,
            policy_loss=self._last_policy_loss,
            value_loss=self._last_value_loss,
            entropy=self._last_entropy,
        )
        self.ticks.append(rec)
        self._ep_ticks.append(rec)

    def update_losses(self,
                      policy_loss: float,
                      value_loss:  float,
                      entropy:     float) -> None:
        """Call whenever MAPPO update runs (does NOT create a tick record)."""
        self._last_policy_loss = policy_loss
        self._last_value_loss  = value_loss
        self._last_entropy     = entropy

    # ── Episode boundary ──────────────────────────────────────────────────────

    def close_episode(self, episode: int,
                      scenario_type: str = 'full_core') -> EpisodeRecord:
        """Compute episode aggregate and reset per-episode state."""
        ticks = self._ep_ticks
        if not ticks:
            self._ep_ticks = []
            dummy = EpisodeRecord(episode=episode, duration_ticks=0,
                severance_tick=-1, scenario_type=scenario_type,
                pre_sev_ue_conn=0.0, post_sev_ue_conn=0.0,
                post_sev_transport_relay=0.0, post_sev_transport_links=0.0, peak_transport_relay=0,
                ep_reward_mean=0.0, final_policy_loss=float('nan'),
                final_value_loss=float('nan'), final_entropy=float('nan'))
            self.episodes.append(dummy)
            return dummy

        # Find severance tick
        sev_tick = -1
        for r in ticks:
            if r.island_mode:
                sev_tick = r.tick
                break

        pre_ticks  = [r for r in ticks if not r.island_mode]
        post_ticks = [r for r in ticks if r.island_mode]

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        rec = EpisodeRecord(
            episode=episode,
            duration_ticks=len(ticks),
            severance_tick=sev_tick,
            scenario_type=scenario_type,
            pre_sev_ue_conn=_mean([r.ue_conn_frac for r in pre_ticks]),
            post_sev_ue_conn=_mean([r.ue_conn_frac for r in post_ticks]),
            post_sev_transport_relay=_mean([r.transport_relay_count for r in post_ticks]),
            post_sev_transport_links=_mean([r.transport_link_count for r in post_ticks]),
            peak_transport_relay=max((r.transport_relay_count for r in post_ticks), default=0),
            ep_reward_mean=_mean([r.reward for r in ticks]),
            final_policy_loss=self._last_policy_loss,
            final_value_loss=self._last_value_loss,
            final_entropy=self._last_entropy,
        )
        self.episodes.append(rec)
        self._ep_ticks = []
        return rec

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save all records to JSON (ticks + episodes)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'ticks':    [asdict(r) for r in self.ticks],
            'episodes': [asdict(r) for r in self.episodes],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=lambda x: None if math.isnan(x) else x)
        print(f"[LearningTracker] Saved {len(self.ticks)} tick records, "
              f"{len(self.episodes)} episodes → {path}")

    @classmethod
    def load(cls, path: str | Path) -> 'LearningTracker':
        with open(path) as f:
            data = json.load(f)
        tracker = cls()
        tracker.ticks    = [TickRecord(**r)   for r in data.get('ticks', [])]
        tracker.episodes = [EpisodeRecord(**r) for r in data.get('episodes', [])]
        return tracker
