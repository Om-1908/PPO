"""
pretrain.py
-----------
Supervised pretraining for QuantumRL agents using the Haar synthesis datasets.

Strategy
--------
For each (target_sv, gate_sequence) in the dataset:
  1. Roll the environment forward step-by-step following the expert sequence.
  2. At each step t, collect observation obs_t and expert action index a_t.
  3. Accumulate (obs_t, a_t) pairs into a supervised dataset.

Then train the network with cross-entropy loss:
  - DQN  : CrossEntropyLoss(q_net(obs), a)   — treats Q-values as logits
  - PPO  : CrossEntropyLoss(actor(obs), a)   — direct actor policy training

This forces the network weights to prefer the expert decomposition before
any RL exploration begins, giving the agent a warm start that is consistent
with RY→RZ (1-qubit) and RY→CNOT→KAK (2-qubit) structure.

Functions
---------
pretrain_dqn_agent(agent, env, config, dataset)
pretrain_ppo_agent(agent, env, config, dataset)
"""

import os
import math
import sys
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Build supervised dataset
# ---------------------------------------------------------------------------

def _build_supervised_dataset(
    env,
    dataset: List[Tuple[np.ndarray, List[Tuple]]],
    max_states: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll through expert trajectories in the environment and collect (obs, action) pairs.

    Parameters
    ----------
    env      : QuantumCircuitEnv instance (already constructed)
    dataset  : list of (target_sv, action_tuples) from parse_dataset.py
    max_states: if given, only process this many states

    Returns
    -------
    obs_arr    : float32 array of shape (N, obs_size)
    action_arr : int64 array of shape (N,)
    """
    obs_list    = []
    action_list = []

    subset = dataset[:max_states] if max_states is not None else dataset

    for target_sv, gate_sequence in subset:
        # Reset env with this expert target
        obs, _ = env.reset(target_sv=target_sv)

        for gate_tuple in gate_sequence:
            # Map gate tuple → action index
            idx = env.gate_tuple_to_action_idx(gate_tuple)
            if idx is None:
                # Gate not in action space — skip this state entirely
                break

            obs_list.append(obs.copy())
            action_list.append(idx)

            # Step the environment with the expert action
            obs, _, terminated, truncated, _ = env.step(idx)
            if terminated or truncated:
                break

    if len(obs_list) == 0:
        return np.zeros((0, env.observation_space.shape[0]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    obs_arr    = np.array(obs_list,    dtype=np.float32)
    action_arr = np.array(action_list, dtype=np.int64)
    return obs_arr, action_arr


def _supervised_train_loop(
    net,
    obs_arr: np.ndarray,
    action_arr: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    net_type: str = 'dqn',
    label: str = 'pretrain',
    val_ratio: float = 0.1,
    save_path: Optional[str] = None,
) -> None:
    """
    Run cross-entropy supervised training on collected (obs, action) pairs with train/val split.
    """
    if len(obs_arr) == 0:
        print(f'[{label}] No supervised data collected — skipping pretrain.')
        return

    N = len(obs_arr)
    val_size = int(N * val_ratio)
    train_size = N - val_size

    # Permute indices for train/val split
    perm = np.random.permutation(N)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    obs_train = torch.tensor(obs_arr[train_idx], dtype=torch.float32)
    acts_train = torch.tensor(action_arr[train_idx], dtype=torch.long)

    obs_val = torch.tensor(obs_arr[val_idx], dtype=torch.float32)
    acts_val = torch.tensor(action_arr[val_idx], dtype=torch.long)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float('inf')

    print(f'[{label}] Starting supervised pretraining: {train_size:,} train / {val_size:,} val samples x {epochs} epochs '
          f'(batch={batch_size}) on {device}')

    for epoch in range(epochs):
        net.train()
        net = net.to(device)
        
        indices = torch.randperm(train_size)
        epoch_loss = 0.0
        epoch_correct = 0
        batches = 0

        for start in range(0, train_size, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_b = obs_train[batch_idx].to(device)
            acts_b = acts_train[batch_idx].to(device)

            if net_type == 'dqn':
                logits = net(obs_b)
            else:
                logits, _ = net(obs_b)

            loss = criterion(logits, acts_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(acts_b)
            epoch_correct += (logits.argmax(dim=1) == acts_b).sum().item()
            batches += 1

        scheduler.step()
        train_loss = epoch_loss / train_size
        train_acc = epoch_correct / train_size * 100.0

        # Validation evaluation
        net.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for start in range(0, val_size, batch_size):
                obs_b = obs_val[start : start + batch_size].to(device)
                acts_b = acts_val[start : start + batch_size].to(device)

                if net_type == 'dqn':
                    logits = net(obs_b)
                else:
                    logits, _ = net(obs_b)

                loss = criterion(logits, acts_b)
                val_loss += loss.item() * len(acts_b)
                val_correct += (logits.argmax(dim=1) == acts_b).sum().item()

        val_loss /= max(val_size, 1)
        val_acc = val_correct / max(val_size, 1) * 100.0
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(net.state_dict(), save_path)

        marker = ' [BEST]' if is_best else ''
        print(f'[{label}] Epoch {epoch + 1:3d}/{epochs}  '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.1f}%  '
              f'lr={current_lr:.2e}{marker}', flush=True)

    print(f'[{label}] Supervised pretraining complete.  '
          f'Best Val Accuracy: {best_val_acc:.1f}% (Val Loss: {best_val_loss:.4f})')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pretrain_dqn_agent(agent, env, config, dataset, device=None) -> None:
    """
    Pretrain DQN agent's q_net using supervised cross-entropy on expert data.
    """
    if device is None:
        device = agent.device

    print(f'[pretrain_dqn] Building supervised dataset from {len(dataset)} expert states ...')
    obs_arr, action_arr = _build_supervised_dataset(
        env, dataset, max_states=getattr(config, 'PRETRAIN_MAX_STATES', None)
    )
    print(f'[pretrain_dqn] Collected {len(obs_arr)} (obs, action) pairs.')

    save_path = os.path.join(os.path.dirname(config.DQN_MODEL_PATH), 'dqn_pretrained_best.pth')
    _supervised_train_loop(
        net        = agent.q_net,
        obs_arr    = obs_arr,
        action_arr = action_arr,
        lr         = getattr(config, 'PRETRAIN_LR', 3e-4),
        epochs     = getattr(config, 'PRETRAIN_EPOCHS', 15),
        batch_size = getattr(config, 'PRETRAIN_BATCH_SIZE', 512),
        device     = device,
        net_type   = 'dqn',
        label      = 'pretrain_dqn',
        save_path  = save_path,
    )

    # Sync target network to pretrained weights
    agent.update_target()
    print('[pretrain_dqn] Target network synced to pretrained weights.')


def pretrain_ppo_agent(agent, env, config, dataset, device=None) -> None:
    """
    Pretrain PPO agent's ActorCritic network using supervised cross-entropy on expert data.
    """
    if device is None:
        device = agent.device

    print(f'[pretrain_ppo] Building supervised dataset from {len(dataset)} expert states ...')
    obs_arr, action_arr = _build_supervised_dataset(
        env, dataset, max_states=getattr(config, 'PRETRAIN_MAX_STATES', None)
    )
    print(f'[pretrain_ppo] Collected {len(obs_arr)} (obs, action) pairs.')

    save_path = os.path.join(os.path.dirname(config.PPO_MODEL_PATH), 'ppo_pretrained_best.pth')
    _supervised_train_loop(
        net        = agent.ac,
        obs_arr    = obs_arr,
        action_arr = action_arr,
        lr         = getattr(config, 'PRETRAIN_LR', 3e-4),
        epochs     = getattr(config, 'PRETRAIN_EPOCHS', 15),
        batch_size = getattr(config, 'PRETRAIN_BATCH_SIZE', 512),
        device     = device,
        net_type   = 'ppo',
        label      = 'pretrain_ppo',
        save_path  = save_path,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from configs.config_1qubit import Config as Config1Q
    from quantum_env import QuantumCircuitEnv
    from dqn_agent import DQNAgent
    from parse_dataset import load_1qubit_dataset

    cfg = Config1Q()
    cfg.PRETRAIN_MAX_STATES = 50
    cfg.PRETRAIN_EPOCHS     = 2

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'haar_1qubit_synthesis.md')
    print(f'Loading 1-qubit dataset from {path} ...')
    dataset = load_1qubit_dataset(path, max_states=100)
    print(f'Loaded {len(dataset)} states.')

    env   = QuantumCircuitEnv(cfg)
    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n
    device      = torch.device('cpu')

    agent = DQNAgent(obs_size, action_size, cfg)
    pretrain_dqn_agent(agent, env, cfg, dataset, device=device)
    print('[smoke-test] pretrain_dqn PASSED')
