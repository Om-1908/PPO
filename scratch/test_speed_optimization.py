import time
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quantumrl'))
from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent, PrioritizedReplayBuffer

class FastPrioritizedReplayBuffer:
    """Optimized PER buffer using pre-exponentiated priority cache & contiguous numpy arrays."""
    def __init__(self, capacity: int, obs_size: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 50000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.priorities_alpha = np.zeros(capacity, dtype=np.float32)
        self.max_prio = 1.0

    def push(self, state, action, reward, next_state, done):
        prio = self.max_prio
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.priorities[self.pos] = prio
        self.priorities_alpha[self.pos] = prio ** self.alpha

        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int):
        self.frame_count += 1
        beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        N = self.size

        prios_alpha = self.priorities_alpha[:N]
        total_prio = prios_alpha.sum()
        probs = prios_alpha / total_prio

        indices = np.random.choice(N, batch_size, p=probs)

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights,
            indices
        )

    def update_priorities(self, indices, td_errors):
        abs_td = np.abs(td_errors) + 1e-5
        for idx, td in zip(indices, abs_td):
            self.priorities[idx] = td
            self.priorities_alpha[idx] = td ** self.alpha
        self.max_prio = max(self.max_prio, float(abs_td.max()))

    def __len__(self):
        return self.size

def benchmark_buffer():
    print("Testing Buffer Speeds...")
    capacity = 200000
    obs_size = 18
    batch_size = 1024
    
    # Standard PER
    std_buf = PrioritizedReplayBuffer(capacity)
    dummy_obs = np.random.randn(obs_size).astype(np.float32)
    for _ in range(50000):
        std_buf.push(dummy_obs, 0, 1.0, dummy_obs, False)
        
    t0 = time.time()
    for _ in range(1000):
        _ = std_buf.sample(batch_size)
    t_std = time.time() - t0
    print(f"Standard PER 1000 samples took: {t_std:.4f} seconds ({1000/t_std:.1f} samples/sec)")
    
    # Fast PER
    fast_buf = FastPrioritizedReplayBuffer(capacity, obs_size)
    for _ in range(50000):
        fast_buf.push(dummy_obs, 0, 1.0, dummy_obs, False)
        
    t0 = time.time()
    for _ in range(1000):
        _ = fast_buf.sample(batch_size)
    t_fast = time.time() - t0
    print(f"Fast PER 1000 samples took: {t_fast:.4f} seconds ({1000/t_fast:.1f} samples/sec)")
    print(f"Speedup: {t_std / t_fast:.2f}x")

if __name__ == '__main__':
    benchmark_buffer()
