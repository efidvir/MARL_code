"""
Real-time dashboard for Digital Twin Multi-Agent Reinforcement Learning (MARL) pre-training.

Visualizes loss metrics, rewards, episode trajectories, and action distributions.
"""

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend to prevent headless deadlocks
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque

class TrainingDashboard:
    """Real-time Matplotlib dashboard to monitor MARL agent training."""
    
    def __init__(self, max_points: int = 100):
        """
        Initialize the dashboard.
        
        Args:
            max_points: Maximum number of history points to display on the fast-moving x-axes.
        """
        self.max_points = max_points
        
        # History tracking
        self.episodes = []
        self.critic_losses = []
        self.policy_losses = []
        self.avg_rewards = []
        self.episode_lengths = []
        
        # Action distribution tracking (Admit, Throttle, Hold)
        self.action_history = {
            'Admit': deque(maxlen=max_points),
            'Throttle': deque(maxlen=max_points),
            'Hold': deque(maxlen=max_points)
        }
        
        # Performance/Optimization tracking
        self.fps_history = deque(maxlen=max_points)
        
        # Current Scenario Stats text block
        self.scenario_text = "Waiting for Scenario..."
        
        # Initialize Matplotlib interactive mode
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("6G Network Digital Twin - MARL Pre-Training", fontsize=16, fontweight='bold')
        self.grid = GridSpec(3, 2, figure=self.fig, height_ratios=[1, 2, 2])
        
        # --- Top Row: Text KPIs ---
        self.ax_kpi = self.fig.add_subplot(self.grid[0, :])
        self.ax_kpi.axis('off')
        self.kpi_text_obj = self.ax_kpi.text(0.5, 0.5, "Initializing...", 
                                             horizontalalignment='center', 
                                             verticalalignment='center',
                                             fontsize=12,
                                             bbox=dict(facecolor='lightgrey', alpha=0.5, boxstyle='round,pad=1'))
                                             
        # --- Middle Row left: Losses ---
        self.ax_loss = self.fig.add_subplot(self.grid[1, 0])
        self.ax_loss.set_title("Network Losses")
        self.ax_loss.set_xlabel("Episode")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.line_critic_loss, = self.ax_loss.plot([], [], label='Critic (Value) Loss', color='red', alpha=0.7, marker='o', markersize=4)
        self.line_policy_loss, = self.ax_loss.plot([], [], label='Policy Loss', color='blue', alpha=0.7, marker='o', markersize=4)
        self.ax_loss.legend()
        
        # --- Middle Row right: Rewards ---
        self.ax_reward = self.fig.add_subplot(self.grid[1, 1])
        self.ax_reward.set_title("Average Global Reward")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Reward")
        self.ax_reward.grid(True, alpha=0.3)
        self.line_reward, = self.ax_reward.plot([], [], label='Avg Reward', color='green', linewidth=2, marker='o', markersize=4)
        
        # --- Bottom Row: Action Distribution (Stacked plot) ---
        self.ax_actions = self.fig.add_subplot(self.grid[2, :])
        self.ax_actions.set_title("Agent Action Distribution (Rolling Window)")
        self.ax_actions.set_xlabel("Training Steps")
        self.ax_actions.set_ylabel("Percentage (%)")
        self.ax_actions.set_ylim(0, 100)
        self.ax_actions.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw()
        try:
            self.fig.canvas.draw()
        except Exception:
            pass
            
    def update_scenario(self, episode: int, total_episodes: int, scenario_info: Dict[str, Any]):
        """Update the top KPI banner with current scenario details."""
        evts = scenario_info.get('events', [])
        sev_tick = next((e.get('tick') for e in evts if e.get('event_type') == 'sever_core'), 'N/A')
        total_failures = len([e for e in evts if e.get('event_type') == 'node_failure'])
        total_surges = len([e for e in evts if e.get('event_type') == 'traffic_surge'])
        
        text = (f"Episode: {episode}/{total_episodes} | "
                f"Tick Duration: {scenario_info.get('duration_ticks', 0)} | "
                f"Severance Tick: {sev_tick} | "
                f"Node Failures: {total_failures} | "
                f"Traffic Surges: {total_surges}")
                
        self.kpi_text_obj.set_text(text)
        
    def add_training_metrics(self, critic_loss: float, policy_loss: float, avg_reward: float):
        """Append end-of-episode/training metrics."""
        self.episodes.append(len(self.episodes) + 1)
        self.critic_losses.append(critic_loss)
        self.policy_losses.append(policy_loss)
        self.avg_rewards.append(avg_reward)
        
    def add_action_distribution(self, distribution: Dict[str, float]):
        """
        Append latest action distribution.
        distribution: dict with keys 'Admit', 'Throttle', 'Hold' summing to 1.0 or 100.0
        """
        # Normalize to percentage
        total = sum(distribution.values())
        if total > 0:
            self.action_history['Admit'].append((distribution.get('Admit', 0) / total) * 100)
            self.action_history['Throttle'].append((distribution.get('Throttle', 0) / total) * 100)
            self.action_history['Hold'].append((distribution.get('Hold', 0) / total) * 100)
        else:
            self.action_history['Admit'].append(33.3)
            self.action_history['Throttle'].append(33.3)
            self.action_history['Hold'].append(33.3)
            
    def render(self):
        """Refresh the matplotlib canvas with the latest data."""
        if not plt.fignum_exists(self.fig.number):
            return # Window was closed
            
        # Update Loss plot
        if self.episodes:
            self.line_critic_loss.set_data(self.episodes, self.critic_losses)
            self.line_policy_loss.set_data(self.episodes, self.policy_losses)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view(scalex=True, scaley=True)
            
            # Bound Y to avoid massive initial spikes destroying scale
            if len(self.critic_losses) > 1:
                max_loss = max(max(self.critic_losses[1:]), max(self.policy_losses[1:])) + 0.1
                self.ax_loss.set_ylim(bottom=0, top=max_loss)
            
        # Update Reward plot
        if self.episodes:
            self.line_reward.set_data(self.episodes, self.avg_rewards)
            self.ax_reward.relim()
            self.ax_reward.autoscale_view(scalex=True, scaley=True)
            
        # Update Stacked Actions
        self.ax_actions.clear()
        self.ax_actions.set_title("Agent Action Distribution (Rolling Window)")
        self.ax_actions.set_xlabel("Training Steps")
        self.ax_actions.set_ylabel("Percentage (%)")
        self.ax_actions.set_ylim(0, 100)
        
        if len(self.action_history['Admit']) > 0:
            x_range = np.arange(len(self.action_history['Admit']))
            self.ax_actions.stackplot(
                x_range,
                self.action_history['Admit'],
                self.action_history['Throttle'],
                self.action_history['Hold'],
                labels=['Admit', 'Throttle', 'Hold'],
                colors=['#2ca02c', '#ff7f0e', '#d62728'],
                alpha=0.8
            )
            self.ax_actions.legend(loc='upper right')
            
        try:
            self.fig.canvas.draw()
        except Exception:
            pass # Handle window closed during draw
            
    def close(self):
        """Close the dashboard."""
        plt.ioff()
        plt.close(self.fig)
