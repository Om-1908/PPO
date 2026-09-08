"""
train_ppo.py
------------
PPO training entry point for QuantumRL.

Enhancements vs. original:
  1. Supervised pretraining on Haar dataset (Euler/KAK warm-start).
  2. Expert transitions injected into first rollout buffer fill.
  3. CNOT usage logging per-episode (2-qubit only).
  4. env.current_episode set at the start of each episode for CNOT forcing.

Run with:
    python train_ppo.py
"""

import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from ppo_agent import PPOAgent, RolloutBuffer
from quantum_env import QuantumCircuitEnv
from utils import (
    best_checkpoint_path,
    generate_random_statevector,
    load_logs,
    plot_training_curves,
    plot_simultaneous_curves,
    save_logs,
)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _greedy_eval_ppo(
    agent: PPOAgent,
    env: QuantumCircuitEnv,
    config: Config,
    episode: int,
) -> float:
    """
    Lightweight greedy evaluation of the PPO agent on freshly generated states.

    Runs BEST_CHECKPOINT_EVAL_STATES deterministic (argmax) episodes.
    State seeds are derived from the current episode.

    Returns
    -------
    float
        Mean fidelity across the evaluation states.
    """
    n_states  = getattr(config, 'BEST_CHECKPOINT_EVAL_STATES', 50)
    eval_seed = config.SEED + 20000 + episode

    agent.ac.eval()   # evaluation mode

    fidelities = []
    with torch.no_grad():
        for i in range(n_states):
            target_sv = generate_random_statevector(config.NUM_QUBITS, seed=eval_seed + i)
            obs, _    = env.reset(target_sv=target_sv)
            done      = False
            final_fid = 0.0
            while not done:
                action = agent.select_action_greedy(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done      = terminated or truncated
                final_fid = info['fidelity']
            fidelities.append(final_fid)

    agent.ac.train()  # restore training mode

    return float(np.mean(fidelities))


def _load_pretrain_data(config: Config):
    """Load the Haar synthesis dataset for pretraining (returns [] if disabled)."""
    if not getattr(config, 'PRETRAIN_ENABLED', False):
        return []

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_path_cfg  = getattr(config, 'PRETRAIN_DATASET_PATH', '')
    ds_path      = ds_path_cfg if os.path.isabs(ds_path_cfg) else os.path.join(project_root, ds_path_cfg)

    if not os.path.exists(ds_path):
        print(f'[train_ppo] WARNING: Pretrain dataset not found at {ds_path}. Skipping pretrain.')
        return []

    from parse_dataset import load_1qubit_dataset, load_2qubit_dataset
    max_states = getattr(config, 'PRETRAIN_MAX_STATES', None)
    print(f'[train_ppo] Loading pretrain dataset from {ds_path} (max_states={max_states}) ...')

    if config.NUM_QUBITS == 1:
        dataset = load_1qubit_dataset(ds_path, max_states=max_states)
    else:
        dataset = load_2qubit_dataset(ds_path, max_states=max_states)

    print(f'[train_ppo] Loaded {len(dataset)} expert states for pretraining.')
    return dataset


def train_ppo(config: Config) -> None:
    """Main training loop for PPO agent in QuantumRL."""
    set_seeds(config.SEED)
    num_cpus = os.cpu_count() or 8
    torch.set_num_threads(num_cpus)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f'[train_ppo] Using device: cuda ({gpu_name}) (PyTorch threads: {torch.get_num_threads()})')
    else:
        print(f'[train_ppo] WARNING: Using device: cpu — GPU not detected (PyTorch threads: {torch.get_num_threads()})')

    env         = QuantumCircuitEnv(config)
    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f'[PPO] Training configuration: {config.NUM_QUBITS} Qubit(s)')
    print(f'[PPO] obs_size={obs_size}  action_size={action_size}')

    # ── Checkpoint save guard ──────────────────────────────────────────────
    if os.path.exists(config.PPO_MODEL_PATH):
        print(
            f'[PPO][SAVE-GUARD] Warning: Existing model found at '
            f"'{config.PPO_MODEL_PATH}'. New checkpoints will overwrite this file."
        )

    best_path     = best_checkpoint_path(config.PPO_MODEL_PATH)
    eval_interval = getattr(config, 'BEST_CHECKPOINT_EVAL_INTERVAL', 1000)
    print(
        f'[PPO] Best-checkpoint eval every {eval_interval} episodes '
        f'({getattr(config, "BEST_CHECKPOINT_EVAL_STATES", 50)} states each).'
    )

    agent  = PPOAgent(obs_size, action_size, config, device)
    buffer = RolloutBuffer(config.PPO_ROLLOUT_STEPS, obs_size, device)

    # ── Load pretrain dataset ─────────────────────────────────────────────
    pretrain_dataset = _load_pretrain_data(config)

    # ── Supervised pretraining ────────────────────────────────────────────
    if pretrain_dataset and getattr(config, 'PRETRAIN_ENABLED', False):
        from pretrain import pretrain_ppo_agent
        pretrain_ppo_agent(agent, env, config, pretrain_dataset, device=device)
    else:
        print('[train_ppo] Supervised pretraining skipped.')

    # ── Training state ────────────────────────────────────────────────────
    episode_rewards     = []
    episode_fidelities  = []
    episode_steps_list  = []
    action_counts       = Counter()
    cnot_episodes       = 0    # 2-qubit: episodes where CNOT/CZ was used

    best_eval_fidelity: float = float('-inf')
    best_eval_episode:  int   = -1
    current_episode:    int   = 0
    start_episode:      int   = 0

    if os.path.exists(config.PPO_MODEL_PATH) and os.path.exists(config.PPO_LOG_PATH):
        try:
            agent.load(config.PPO_MODEL_PATH)
            agent.ac.train()
            logs = load_logs(config.PPO_LOG_PATH)
            if logs and 'rewards' in logs and len(logs['rewards']) > 0:
                episode_rewards    = list(logs['rewards'])
                episode_fidelities = list(logs['fidelities'])
                episode_steps_list = list(logs['steps'])
                current_episode    = len(episode_rewards)
                start_episode      = current_episode
                if agent.scheduler is not None:
                    for _ in range(current_episode):
                        agent.scheduler.step()
                print(f'[PPO] Resuming training from Episode {current_episode} '
                      f'(loaded {len(episode_rewards)} past episodes)')
        except Exception as e:
            print(f'[PPO] Could not load resume checkpoint: {e}')

    print(f'[PPO] Starting training for {config.PPO_EPISODES} episodes '
          f'(from {current_episode}) ...\n')

    # ── Expert buffer injection (first rollout only) ───────────────────────
    expert_injected = False

    obs, _ = env.reset()
    env.current_episode = current_episode
    obs_t = torch.FloatTensor(obs).to(device)

    episode_reward = 0.0
    episode_steps  = 0

    while current_episode < config.PPO_EPISODES:

        # ── Rollout collection phase ───────────────────────────────────────
        buffer.reset()

        # First rollout: inject expert transitions if enabled
        if (not expert_injected
                and pretrain_dataset
                and getattr(config, 'EXPERT_BUFFER_ENABLED', False)):
            from expert_buffer import load_expert_transitions_ppo
            expert_n = getattr(config, 'EXPERT_BUFFER_STATES', 2000)
            n_written = load_expert_transitions_ppo(
                buffer, env, agent, config, pretrain_dataset, device,
                max_states=expert_n,
            )
            expert_injected = True
            # Resume env state after expert injection
            obs, _  = env.reset()
            obs_t   = torch.FloatTensor(obs).to(device)
            episode_reward = 0.0
            episode_steps  = 0
            # If buffer is not full, continue filling with RL rollout
            if buffer.ptr >= config.PPO_ROLLOUT_STEPS:
                # Buffer full from expert data — skip RL rollout collection this update
                pass

        for step in range(buffer.ptr, config.PPO_ROLLOUT_STEPS):
            env.current_episode = current_episode
            with torch.no_grad():
                action, log_prob, entropy, value = agent.select_action(obs_t)

            action_counts[action] += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs_t.cpu(), action, log_prob.cpu(), reward, done, value.cpu())

            obs_t          = torch.FloatTensor(next_obs).to(device)
            episode_reward += reward
            episode_steps  += 1

            if done:
                if info.get('cnot_used', False):
                    cnot_episodes += 1

                episode_rewards.append(episode_reward)
                episode_fidelities.append(info['fidelity'])
                episode_steps_list.append(episode_steps)

                current_episode += 1
                episode_reward  = 0.0
                episode_steps   = 0

                obs, _ = env.reset()
                env.current_episode = current_episode
                obs_t  = torch.FloatTensor(obs).to(device)

                # ── Per-500-episode progress print & live plot ──────────
                if current_episode % 500 == 0 or current_episode == config.PPO_EPISODES:
                    recent_fids = episode_fidelities[-500:]
                    recent_rewards = episode_rewards[-500:]
                    recent_steps = episode_steps_list[-500:]

                    mean_fid = float(np.mean(recent_fids))
                    median_fid = float(np.median(recent_fids))
                    best_fid = float(np.max(episode_fidelities))
                    ge_0999 = float(np.mean([f >= 0.999 for f in recent_fids]))
                    ge_0999999 = float(np.mean([f >= 0.999999 for f in recent_fids]))
                    mean_gates = float(np.mean(recent_steps))
                    median_gates = float(np.median(recent_steps))

                    current_lr = agent.optimizer.param_groups[0]['lr']
                    cnot_rate = cnot_episodes / max(current_episode - start_episode, 1)

                    print(
                        f'Episode {current_episode:6d}/{config.PPO_EPISODES} | '
                        f'Reward: {episode_rewards[-1]:7.3f} | '
                        f'Fid: {info["fidelity"]:.6f} | '
                        f'Mean Fid(500): {mean_fid:.6f} | '
                        f'Median Fid: {median_fid:.6f} | '
                        f'% F>=0.999999: {ge_0999999 * 100:.1f}% | '
                        f'Gates: {mean_gates:.1f} | '
                        f'LR: {current_lr:.2e}',
                        flush=True,
                    )

                    os.makedirs(config.LOG_DIR, exist_ok=True)
                    metrics_dict = {
                        'rewards': episode_rewards,
                        'fidelities': episode_fidelities,
                        'steps': episode_steps_list,
                        'episode': current_episode,
                        'mean_reward': float(np.mean(recent_rewards)),
                        'mean_fidelity': mean_fid,
                        'median_fidelity': median_fid,
                        'best_fidelity': best_fid,
                        'success_rate': ge_0999999,
                        'fraction_fidelity_ge_0.999': ge_0999,
                        'fraction_fidelity_ge_0.999999': ge_0999999,
                        'mean_gate_count': mean_gates,
                        'median_gate_count': median_gates,
                        'learning_rate': current_lr,
                    }
                    save_logs(metrics_dict, config.PPO_LOG_PATH)

                    os.makedirs(config.PLOT_DIR, exist_ok=True)
                    plot_training_curves(
                        episode_rewards, episode_fidelities, episode_steps_list,
                        config.PPO_PLOT_PATH,
                    )
                    plot_simultaneous_curves(
                        os.path.join(config.LOG_DIR, 'dqn_logs.json'),
                        config.PPO_LOG_PATH,
                        os.path.join(config.PLOT_DIR, 'simultaneous_dqn_ppo.png'),
                    )

                # ── Periodic greedy evaluation ──────────────────────────
                if current_episode % eval_interval == 0:
                    eval_fid = _greedy_eval_ppo(agent, env, config, current_episode)
                    marker = ''
                    if eval_fid > best_eval_fidelity:
                        best_eval_fidelity = eval_fid
                        best_eval_episode = current_episode
                        agent.save(best_path)
                        marker = '  *** new best checkpoint saved ***'
                    print(
                        f'  [checkpoint-eval ep {current_episode:6d}] '
                        f'mean fidelity = {eval_fid:.6f}{marker}',
                        flush=True,
                    )

                if current_episode >= config.PPO_EPISODES:
                    break

        # ── PPO update phase ───────────────────────────────────────────────
        with torch.no_grad():
            _, _, _, last_value = agent.select_action(obs_t)

        buffer.compute_returns_and_advantages(
            last_value.cpu(), config.PPO_GAMMA, config.PPO_GAE_LAMBDA
        )
        agent.update(buffer)

        if agent.scheduler is not None:
            agent.scheduler.step()

    # ── Save final model ───────────────────────────────────────────────────
    agent.save(config.PPO_MODEL_PATH)

    # ── Final lightweight eval ─────────────────────────────────────────────
    final_eval_fid = _greedy_eval_ppo(agent, env, config, episode=config.PPO_EPISODES)

    if final_eval_fid > best_eval_fidelity:
        best_eval_fidelity = final_eval_fid
        best_eval_episode  = config.PPO_EPISODES
        agent.save(best_path)

    # ── Post-training logging & plots ──────────────────────────────────────
    os.makedirs(config.LOG_DIR, exist_ok=True)
    save_logs(
        {'rewards': episode_rewards, 'fidelities': episode_fidelities,
         'steps': episode_steps_list},
        config.PPO_LOG_PATH,
    )
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plot_training_curves(
        episode_rewards, episode_fidelities, episode_steps_list,
        config.PPO_PLOT_PATH,
    )

    # ── End-of-training summary ────────────────────────────────────────────
    total_eps = current_episode - start_episode
    cnot_rate = cnot_episodes / max(total_eps, 1)
    print(f'\n[PPO] Training complete.')
    if config.NUM_QUBITS >= 2:
        print(f'[PPO] CNOT/CZ usage rate: {cnot_rate:.2%} ({cnot_episodes}/{total_eps} episodes)')
    print(f'[PPO] {"-" * 52}')
    print(f'[PPO]  End-of-training checkpoint summary')
    print(f'[PPO] {"-" * 52}')
    print(f'[PPO]  Final model eval fidelity  : {final_eval_fid:.4f}')
    print(f'[PPO]  Best checkpoint fidelity   : {best_eval_fidelity:.4f}  '
          f'(episode {best_eval_episode})')
    delta = best_eval_fidelity - final_eval_fid
    if best_eval_episode == config.PPO_EPISODES or abs(delta) < 1e-6:
        print(f'[PPO]  The final model IS the best checkpoint.')
    else:
        print(
            f'[PPO]  An earlier checkpoint outperformed the final model '
            f'by {delta:+.4f}. Best checkpoint saved at: {best_path}'
        )
    print(f'[PPO] {"-" * 52}')

    if config.LOG_ACTION_HISTOGRAM:
        used_count = sum(1 for i in range(action_size) if action_counts.get(i, 0) > 0)
        least_used = sorted(
            ((i, action_counts.get(i, 0)) for i in range(action_size)),
            key=lambda x: (x[1], x[0]),
        )[:5]
        print(f'[PPO] Action usage: {used_count}/{action_size} actions used at least once')
        print('[PPO] 5 least-used action indices:')
        for idx, count in least_used:
            g, q, a = env.action_list[idx]
            angle_str = f' angle={a:.3f}' if a is not None else ''
            print(f'  action {idx}: {count}  ({g} q={q}{angle_str})')


if __name__ == '__main__':
    cfg = Config()
    train_ppo(cfg)
