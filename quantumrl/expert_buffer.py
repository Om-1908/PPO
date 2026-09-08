"""
expert_buffer.py
----------------
Loads expert Haar synthesis trajectories into the DQN PER replay buffer
and into the PPO rollout buffer, giving the agents pre-filled experience
from perfect decompositions before any RL training begins.

For DQN (PrioritizedReplayBuffer):
  Each expert trajectory is stepped through the environment to collect
  (obs_t, action, reward, obs_{t+1}, done) tuples which are pushed into
  the buffer with maximum priority so they get sampled first.

For PPO (RolloutBuffer):
  Expert transitions are injected at the start of a rollout buffer fill,
  supplementing random environment interactions with structured expert data.

Functions
---------
load_expert_transitions_dqn(agent, env, config, dataset)
load_expert_transitions_ppo(buffer, env, config, dataset)
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch


def load_expert_transitions_dqn(
    agent,
    env,
    config,
    dataset: List[Tuple[np.ndarray, List[Tuple]]],
    max_states: Optional[int] = None,
) -> int:
    """
    Step through expert trajectories and push (s, a, r, s', done) into the DQN PER buffer.

    Transitions are pushed with maximum priority so they are sampled preferentially.

    Parameters
    ----------
    agent     : DQNAgent
    env       : QuantumCircuitEnv
    config    : Config dataclass
    dataset   : list of (target_sv, action_sequence)
    max_states: maximum number of dataset states to process

    Returns
    -------
    int : total number of transitions pushed
    """
    if max_states is None:
        max_states = getattr(config, 'EXPERT_BUFFER_STATES', 2000)

    total_pushed = 0
    subset = dataset[:max_states]

    print(f'[expert_buffer] Loading {len(subset)} expert trajectories into DQN PER buffer ...')

    for target_sv, gate_sequence in subset:
        obs, _ = env.reset(target_sv=target_sv)

        for gate_tuple in gate_sequence:
            idx = env.gate_tuple_to_action_idx(gate_tuple)
            if idx is None:
                break  # Gate not in action space

            next_obs, reward, terminated, truncated, _ = env.step(idx)
            done = terminated or truncated

            agent.buffer.push(obs, idx, reward, next_obs, float(done))
            total_pushed += 1

            obs = next_obs
            if done:
                break

    print(f'[expert_buffer] Pushed {total_pushed} expert transitions into DQN PER buffer. '
          f'Buffer size: {len(agent.buffer)}')
    return total_pushed


def load_expert_transitions_ppo(
    buffer,
    env,
    agent,
    config,
    dataset: List[Tuple[np.ndarray, List[Tuple]]],
    device: torch.device,
    max_states: Optional[int] = None,
) -> int:
    """
    Step through expert trajectories and write transitions into a PPO RolloutBuffer.

    The buffer is filled from ptr=0. Call buffer.reset() before this function.
    Stops when the buffer is full (ptr reaches rollout_steps).

    Parameters
    ----------
    buffer    : RolloutBuffer (pre-reset)
    env       : QuantumCircuitEnv
    agent     : PPOAgent (used to compute value estimates for GAE)
    config    : Config dataclass
    dataset   : list of (target_sv, action_sequence)
    device    : torch.device
    max_states: maximum number of dataset states to process

    Returns
    -------
    int : total transitions written
    """
    if max_states is None:
        max_states = getattr(config, 'EXPERT_BUFFER_STATES', 2000)

    rollout_steps = buffer.rollout_steps
    total_written = 0
    subset        = dataset[:max_states]

    print(f'[expert_buffer] Filling PPO rollout buffer with expert trajectories '
          f'(up to {rollout_steps} steps) ...')

    agent.ac.eval()
    with torch.no_grad():
        for target_sv, gate_sequence in subset:
            if buffer.ptr >= rollout_steps:
                break

            obs, _ = env.reset(target_sv=target_sv)
            obs_t  = torch.FloatTensor(obs).to(device)

            for gate_tuple in gate_sequence:
                if buffer.ptr >= rollout_steps:
                    break

                idx = env.gate_tuple_to_action_idx(gate_tuple)
                if idx is None:
                    break

                # Get log_prob and value estimate from current policy
                logits, value = agent.ac(obs_t.unsqueeze(0))
                from torch.distributions import Categorical
                dist     = Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(idx, device=device))

                next_obs, reward, terminated, truncated, _ = env.step(idx)
                done = terminated or truncated

                buffer.add(
                    obs_t.cpu(), idx, log_prob.cpu(),
                    reward, done, value.squeeze(0).cpu()
                )
                total_written += 1

                obs   = next_obs
                obs_t = torch.FloatTensor(obs).to(device)

                if done:
                    break

    agent.ac.train()
    print(f'[expert_buffer] Wrote {total_written} expert transitions into PPO rollout buffer.')
    return total_written
