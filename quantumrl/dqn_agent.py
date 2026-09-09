"""
dqn_agent.py
------------
Dueling Deep Q-Network with Prioritized Experience Replay (PER) for QuantumRL.

Classes:
  DuelingQNetwork          – Dual-stream Q-network splitting State Value V(s) and Advantage A(s,a).
  PrioritizedReplayBuffer  – Proportional priority experience replay with importance sampling weights.
  DQNAgent                 – Double-DQN agent with PER and Dueling architecture.
"""

import collections
import copy
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Dueling Q-Network
# ─────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture splitting Q(s,a) into State-Value V(s)
    and Action-Advantage A(s,a) streams.

    Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
    """

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 768):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size

        # Shared representation trunk
        self.feature_trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01),
        )

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_trunk(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# Backward-compatible alias for QNetwork
QNetwork = DuelingQNetwork


# ─────────────────────────────────────────────────────────
# Prioritized Experience Replay (PER) Buffer
# ─────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay Buffer.
    Transitions are sampled proportional to their TD-error priority: P(i) = p_i^α / ∑_k p_k^α.
    Importance sampling weights w_i correct for distribution bias.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 50000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a prioritized mini-batch of transitions.

        Returns
        -------
        states, actions, rewards, next_states, dones, weights, indices
        """
        self.frame_count += 1
        beta = min(
            1.0,
            self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames,
        )

        N = len(self.buffer)
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights,
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update sample priorities with absolute TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = float(abs(td_error) + 1e-5)

    def __len__(self) -> int:
        return len(self.buffer)


# Backward-compatible alias
ReplayBuffer = PrioritizedReplayBuffer


# ─────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────

class DQNAgent:
    """
    Dueling Double-DQN Agent with Prioritized Experience Replay (PER).
    """

    def __init__(self, obs_size: int, action_size: int, config):
        self.config = config
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        hidden_size = getattr(config, 'DQN_HIDDEN_SIZE', 768)

        # Online and target Q-networks
        self.q_net = DuelingQNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.DQN_LR)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

        alpha = getattr(config, 'PER_ALPHA', 0.6)
        beta_start = getattr(config, 'PER_BETA_START', 0.4)
        beta_frames = getattr(config, 'PER_BETA_FRAMES', 50000)

        self.buffer = PrioritizedReplayBuffer(
            capacity=config.DQN_BUFFER_SIZE,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
        )

        self.epsilon: float = config.DQN_EPSILON_START
        self.steps_done: int = 0

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def select_actions_batch(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Batch ε-greedy action selection for parallel environments on CUDA VRAM."""
        num_envs = state_tensor.shape[0]
        rand_mask = torch.rand(num_envs, device=self.device) < self.epsilon
        random_actions = torch.randint(0, self.action_size, (num_envs,), device=self.device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            greedy_actions = q_values.argmax(dim=1)

        actions = torch.where(rand_mask, random_actions, greedy_actions)
        return actions

    def update(self) -> Optional[float]:
        """Sample a prioritized mini-batch and perform Double-DQN step on GPU with AMP."""
        if len(self.buffer) < self.config.DQN_BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
            self.config.DQN_BATCH_SIZE
        )

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            # Current Q-values: Q(s, a)
            current_q = self.q_net(states_t).gather(1, actions_t)

            # Double DQN target evaluation
            with torch.no_grad():
                best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(next_states_t).gather(1, best_actions)
                target_q = rewards_t + self.config.DQN_GAMMA * max_next_q * (1.0 - dones_t)

            # Weighted MSE loss
            loss = (weights_t * F.mse_loss(current_q, target_q, reduction='none')).mean()

        # TD errors for PER update
        td_errors = (current_q - target_q).detach().cpu().numpy().squeeze()
        self.buffer.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.steps_done += 1
        return loss.item()

    def update_target(self) -> None:
        """Hard update target network weights."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay."""
        self.epsilon = max(
            self.config.DQN_EPSILON_END,
            self.epsilon * self.config.DQN_EPSILON_DECAY,
        )

    def save(self, path: str) -> None:
        """Save network weights to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        print(f"[DQNAgent] Model saved -> {path}")

    def load(self, path: str) -> None:
        """Load network weights from disk."""
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'q_net' in ckpt:
            self.q_net.load_state_dict(ckpt['q_net'])
            if hasattr(self, 'target_net'):
                self.target_net.load_state_dict(ckpt.get('target_net', ckpt['q_net']))
        else:
            self.q_net.load_state_dict(ckpt)
        self.q_net.eval()
        print(f"[DQNAgent] Model loaded <- {path}")
