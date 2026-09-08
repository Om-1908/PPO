"""
train_dqn.py
------------
DQN training entry point for QuantumRL.

Enhancements vs. original:
  1. Supervised pretraining on Haar dataset (Euler/KAK warm-start).
  2. Expert transitions loaded into PER buffer before RL training starts.
  3. CNOT usage logging per-episode (2-qubit only).
  4. env.current_episode set at the start of each episode for CNOT forcing.

Run with:
    python train_dqn.py
"""

import os
import random
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dqn_agent import DQNAgent
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


def _greedy_eval_dqn(
    agent: DQNAgent,
    env: QuantumCircuitEnv,
    config: Config,
    episode: int,
) -> float:
    """
    Lightweight greedy evaluation of the DQN agent on freshly generated states.

    Runs BEST_CHECKPOINT_EVAL_STATES episodes with epsilon=0 (no exploration).
    Uses torch.no_grad() throughout and restores training mode afterwards.
    State seeds are derived from the current episode so the set is fresh at
    every call but still reproducible.

    Returns
    -------
    float
        Mean fidelity across the evaluation states.
    """
    n_states  = getattr(config, 'BEST_CHECKPOINT_EVAL_STATES', 50)
    eval_seed = config.SEED + 20000 + episode

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0           # fully greedy
    agent.q_net.eval()            # evaluation mode

    fidelities = []
    with torch.no_grad():
        for i in range(n_states):
            target_sv = generate_random_statevector(config.NUM_QUBITS, seed=eval_seed + i)
            obs, _    = env.reset(target_sv=target_sv)
            done      = False
            final_fid = 0.0
            while not done:
                action = agent.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done      = terminated or truncated
                final_fid = info['fidelity']
            fidelities.append(final_fid)

    agent.q_net.train()           # restore training mode
    agent.epsilon = saved_epsilon

    return float(np.mean(fidelities))


def _load_pretrain_data(config: Config):
    """Load the Haar synthesis dataset for pretraining (returns [] if disabled)."""
    if not getattr(config, 'PRETRAIN_ENABLED', False):
        return []

    # Resolve dataset path relative to project root (parent of this file's dir)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_path_cfg  = getattr(config, 'PRETRAIN_DATASET_PATH', '')
    ds_path      = ds_path_cfg if os.path.isabs(ds_path_cfg) else os.path.join(project_root, ds_path_cfg)

    if not os.path.exists(ds_path):
        print(f'[train_dqn] WARNING: Pretrain dataset not found at {ds_path}. Skipping pretrain.')
        return []

    from parse_dataset import load_1qubit_dataset, load_2qubit_dataset
    max_states = getattr(config, 'PRETRAIN_MAX_STATES', None)
    print(f'[train_dqn] Loading pretrain dataset from {ds_path} (max_states={max_states}) ...')

    if config.NUM_QUBITS == 1:
        dataset = load_1qubit_dataset(ds_path, max_states=max_states)
    else:
        dataset = load_2qubit_dataset(ds_path, max_states=max_states)

    print(f'[train_dqn] Loaded {len(dataset)} expert states for pretraining.')
    return dataset


def train_dqn(config: Config) -> None:
    """Full DQN training loop for QuantumRL."""
    set_seeds(config.SEED)
    num_cpus = os.cpu_count() or 8
    torch.set_num_threads(num_cpus)

    env         = QuantumCircuitEnv(config)
    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f'[DQN] Training configuration: {config.NUM_QUBITS} Qubit(s)')
    print(f'[DQN] obs_size={obs_size}  action_size={action_size}')
    agent = DQNAgent(obs_size, action_size, config)
    device = agent.device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f'[train_dqn] Using device: cuda ({gpu_name}) (PyTorch threads: {torch.get_num_threads()})')
    else:
        print(f'[train_dqn] WARNING: Using device: cpu — GPU not detected (PyTorch threads: {torch.get_num_threads()})')

    # ── Checkpoint save guard ──────────────────────────────────────────────
    if os.path.exists(config.DQN_MODEL_PATH):
        print(
            f'[DQN][SAVE-GUARD] Warning: Existing model found at '
            f"'{config.DQN_MODEL_PATH}'. New checkpoints will overwrite this file."
        )

    # Derived best-checkpoint path
    best_path    = best_checkpoint_path(config.DQN_MODEL_PATH)
    eval_interval = getattr(config, 'BEST_CHECKPOINT_EVAL_INTERVAL', 1000)
    update_freq   = getattr(config, 'DQN_UPDATE_FREQ', 4)
    print(
        f'[DQN] Best-checkpoint eval every {eval_interval} episodes '
        f'({getattr(config, "BEST_CHECKPOINT_EVAL_STATES", 50)} states each). '
        f'Update freq: every {update_freq} steps.'
    )

    # ── Load pretrain dataset ─────────────────────────────────────────────
    pretrain_dataset = _load_pretrain_data(config)

    # ── Supervised pretraining ────────────────────────────────────────────
    if pretrain_dataset and getattr(config, 'PRETRAIN_ENABLED', False):
        from pretrain import pretrain_dqn_agent
        pretrain_dqn_agent(agent, env, config, pretrain_dataset, device=device)
    else:
        print('[train_dqn] Supervised pretraining skipped.')

    # ── Expert buffer loading ─────────────────────────────────────────────
    if pretrain_dataset and getattr(config, 'EXPERT_BUFFER_ENABLED', False):
        from expert_buffer import load_expert_transitions_dqn
        expert_n = getattr(config, 'EXPERT_BUFFER_STATES', 2000)
        load_expert_transitions_dqn(agent, env, config, pretrain_dataset, max_states=expert_n)
    else:
        print('[train_dqn] Expert buffer loading skipped.')

    # ── Replay buffer warm-up (random exploration) ────────────────────────
    warmup_steps = getattr(config, 'DQN_WARMUP_STEPS', 10000)
    print(f'[DQN] Warming up PER replay buffer with {warmup_steps} random transitions...')
    warmup_obs, _ = env.reset()
    for _wu in range(warmup_steps):
        warmup_action = env.action_space.sample()
        warmup_next_obs, warmup_reward, warmup_term, warmup_trunc, _ = env.step(warmup_action)
        agent.buffer.push(
            warmup_obs, warmup_action, warmup_reward,
            warmup_next_obs, float(warmup_term or warmup_trunc)
        )
        warmup_obs = warmup_next_obs
        if warmup_term or warmup_trunc:
            warmup_obs, _ = env.reset()
    print(f'[DQN] Warm-up complete. Buffer size: {len(agent.buffer)}')

    # ── Training state ────────────────────────────────────────────────────
    episode_rewards    = []
    episode_fidelities = []
    episode_steps      = []
    action_counts      = Counter()
    cnot_episodes      = 0    # 2-qubit: count episodes where CNOT was used

    best_eval_fidelity: float = float('-inf')
    best_eval_episode:  int   = -1
    total_steps:        int   = 0
    start_episode:      int   = 0

    log_path = os.path.join(config.LOG_DIR, 'dqn_logs.json')
    if os.path.exists(config.DQN_MODEL_PATH) and os.path.exists(log_path):
        try:
            agent.load(config.DQN_MODEL_PATH)
            logs = load_logs(log_path)
            if logs and 'rewards' in logs and len(logs['rewards']) > 0:
                episode_rewards    = list(logs['rewards'])
                episode_fidelities = list(logs['fidelities'])
                episode_steps      = list(logs['steps'])
                start_episode      = len(episode_rewards)
                agent.epsilon      = max(
                    config.DQN_EPSILON_END,
                    config.DQN_EPSILON_START * (config.DQN_EPSILON_DECAY ** start_episode)
                )
                print(f'[DQN] Resuming training from Episode {start_episode} '
                      f'(Epsilon={agent.epsilon:.3f})')
        except Exception as e:
            print(f'[DQN] Could not load resume checkpoint: {e}')

    print(f'[DQN] Starting training for {config.DQN_EPISODES} episodes '
          f'(from {start_episode}) ...\n')

    for episode in range(start_episode, config.DQN_EPISODES):
        # Inform env of current episode (for CNOT forcing)
        env.current_episode = episode

        obs, _ = env.reset()
        episode_reward = 0.0
        done   = False
        info   = {'fidelity': 0.0, 'steps': 0, 'cnot_used': False}

        while not done:
            total_steps += 1
            action       = agent.select_action(obs)
            action_counts[action] += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, float(done))
            if total_steps % update_freq == 0:
                agent.update()

            obs             = next_obs
            episode_reward += reward

        agent.decay_epsilon()

        if episode % config.DQN_TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if info.get('cnot_used', False):
            cnot_episodes += 1

        episode_rewards.append(episode_reward)
        episode_fidelities.append(info['fidelity'])
        episode_steps.append(info['steps'])

        # ── Per-500-episode progress print, metrics recording & live plot update ──
        if (episode + 1) % 500 == 0 or (episode + 1) == config.DQN_EPISODES:
            recent_fids = episode_fidelities[-500:]
            recent_rewards = episode_rewards[-500:]
            recent_steps = episode_steps[-500:]

            mean_fid = float(np.mean(recent_fids))
            median_fid = float(np.median(recent_fids))
            best_fid = float(np.max(episode_fidelities))
            ge_0999 = float(np.mean([f >= 0.999 for f in recent_fids]))
            ge_0999999 = float(np.mean([f >= 0.999999 for f in recent_fids]))
            mean_gates = float(np.mean(recent_steps))
            median_gates = float(np.median(recent_steps))

            current_lr = agent.optimizer.param_groups[0]['lr']
            cnot_rate = cnot_episodes / max(episode + 1 - start_episode, 1)

            print(
                f'Episode {episode + 1:6d}/{config.DQN_EPISODES} | '
                f'Reward: {episode_reward:7.3f} | '
                f'Fid: {info["fidelity"]:.6f} | '
                f'Mean Fid(500): {mean_fid:.6f} | '
                f'Median Fid: {median_fid:.6f} | '
                f'% F>=0.999999: {ge_0999999 * 100:.1f}% | '
                f'Gates: {mean_gates:.1f} | '
                f'Eps: {agent.epsilon:.4f}',
                flush=True,
            )

            # Save detailed metrics log
            os.makedirs(config.LOG_DIR, exist_ok=True)
            metrics_dict = {
                'rewards': episode_rewards,
                'fidelities': episode_fidelities,
                'steps': episode_steps,
                'episode': episode + 1,
                'mean_reward': float(np.mean(recent_rewards)),
                'mean_fidelity': mean_fid,
                'median_fidelity': median_fid,
                'best_fidelity': best_fid,
                'success_rate': ge_0999999,
                'fraction_fidelity_ge_0.999': ge_0999,
                'fraction_fidelity_ge_0.999999': ge_0999999,
                'mean_gate_count': mean_gates,
                'median_gate_count': median_gates,
                'epsilon': agent.epsilon,
                'learning_rate': current_lr,
            }
            save_logs(metrics_dict, os.path.join(config.LOG_DIR, 'dqn_logs.json'))

            os.makedirs(config.PLOT_DIR, exist_ok=True)
            plot_training_curves(
                episode_rewards, episode_fidelities, episode_steps,
                os.path.join(config.PLOT_DIR, 'dqn_training.png'),
            )
            plot_simultaneous_curves(
                os.path.join(config.LOG_DIR, 'dqn_logs.json'),
                getattr(config, 'PPO_LOG_PATH', os.path.join(config.LOG_DIR, 'ppo_logs.json')),
                os.path.join(config.PLOT_DIR, 'simultaneous_dqn_ppo.png'),
            )

        # ── Periodic greedy evaluation for best-checkpoint tracking ────────
        if (episode + 1) % eval_interval == 0:
            eval_fid = _greedy_eval_dqn(agent, env, config, episode)
            marker = ''
            if eval_fid > best_eval_fidelity:
                best_eval_fidelity = eval_fid
                best_eval_episode = episode + 1
                agent.save(best_path)
                marker = '  *** new best checkpoint saved ***'
            print(
                f'  [checkpoint-eval ep {episode + 1:6d}] '
                f'mean fidelity = {eval_fid:.6f}{marker}',
                flush=True,
            )

    # ── Save final model ───────────────────────────────────────────────────
    agent.save(config.DQN_MODEL_PATH)

    # ── Final lightweight eval ─────────────────────────────────────────────
    final_eval_fid = _greedy_eval_dqn(agent, env, config, episode=config.DQN_EPISODES)

    if final_eval_fid > best_eval_fidelity:
        best_eval_fidelity = final_eval_fid
        best_eval_episode  = config.DQN_EPISODES
        agent.save(best_path)

    # ── Post-training logging & plots ──────────────────────────────────────
    os.makedirs(config.LOG_DIR, exist_ok=True)
    save_logs(
        {'rewards': episode_rewards, 'fidelities': episode_fidelities,
         'steps': episode_steps},
        os.path.join(config.LOG_DIR, 'dqn_logs.json'),
    )
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plot_training_curves(
        episode_rewards, episode_fidelities, episode_steps,
        os.path.join(config.PLOT_DIR, 'dqn_training.png'),
    )

    # ── End-of-training summary ────────────────────────────────────────────
    total_eps = config.DQN_EPISODES - start_episode
    cnot_rate = cnot_episodes / max(total_eps, 1)
    print(f'\n[DQN] Training complete.')
    print(f'[DQN] Final Epsilon value: {agent.epsilon:.6f}')
    if config.NUM_QUBITS >= 2:
        print(f'[DQN] CNOT/CZ usage rate: {cnot_rate:.2%} ({cnot_episodes}/{total_eps} episodes)')
    print(f'[DQN] {"-" * 52}')
    print(f'[DQN]  End-of-training checkpoint summary')
    print(f'[DQN] {"-" * 52}')
    print(f'[DQN]  Final model eval fidelity  : {final_eval_fid:.4f}')
    print(f'[DQN]  Best checkpoint fidelity   : {best_eval_fidelity:.4f}  '
          f'(episode {best_eval_episode})')
    delta = best_eval_fidelity - final_eval_fid
    if best_eval_episode == config.DQN_EPISODES or abs(delta) < 1e-6:
        print(f'[DQN]  The final model IS the best checkpoint.')
    else:
        print(
            f'[DQN]  An earlier checkpoint outperformed the final model '
            f'by {delta:+.4f}. Best checkpoint saved at: {best_path}'
        )
    print(f'[DQN] {"-" * 52}')

    if config.LOG_ACTION_HISTOGRAM:
        used_count = sum(1 for i in range(action_size) if action_counts.get(i, 0) > 0)
        least_used = sorted(
            ((i, action_counts.get(i, 0)) for i in range(action_size)),
            key=lambda x: (x[1], x[0]),
        )[:5]
        print(f'[DQN] Action usage: {used_count}/{action_size} actions used at least once')
        print('[DQN] 5 least-used action indices:')
        for idx, count in least_used:
            g, q, a = env.action_list[idx]
            angle_str = f' angle={a:.3f}' if a is not None else ''
            print(f'  action {idx}: {count}  ({g} q={q}{angle_str})')


if __name__ == '__main__':
    cfg = Config()
    train_dqn(cfg)
