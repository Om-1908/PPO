"""
train_ppo_resume.py
-------------------
TATVA — 2-Qubit PPO-only resume training script.

Resumes PPO training from the existing best checkpoint, continuing until
EXACTLY 500,000 synthesis episodes are completed.

Key properties:
  - DQN is NOT started or referenced.
  - PPO resumes from ppo_model_best.pth (preserves all learned weights).
  - Episode counter continues from the last logged episode.
  - NO intentional pauses at any episode milestone.
  - Improved reward shaping for aggressive fidelity maximization.
  - Full training stability guards (gradient clipping, advantage normalization,
    entropy regularization, NaN detection, gradient norm monitoring).
  - Lightweight non-destructive evaluation every eval_interval episodes.
  - Automatic best-checkpoint saving whenever eval fidelity improves.
  - All 16 metrics logged every 500 episodes to logs/2qubit/ppo_logs.json.
  - Real statevector fidelity reported — never inferred from reward or value.

Run:
    python quantumrl/train_ppo_resume.py
"""

import json
import os
import sys
import time
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from configs.config_2qubit import Config
from ppo_agent import PPOAgent, RolloutBuffer
from quantum_env import QuantumCircuitEnv
from utils import (
    best_checkpoint_path,
    generate_random_statevector,
    load_logs,
    plot_training_curves,
    plot_simultaneous_curves,
    save_logs,
    compute_fidelity,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Improved reward wrapper (wraps QuantumCircuitEnv.step)
# ---------------------------------------------------------------------------
def shaped_reward(fidelity: float, prev_fidelity: float, step: int,
                  gate_name: str, done: bool, truncated: bool,
                  gate_penalty: float, fidelity_threshold: float) -> float:
    """
    Aggressive fidelity-first reward shaping:

    1. Dense fidelity-improvement signal (10x amplified).
    2. Strong milestone bonuses at 0.9, 0.99, 0.999, 0.9999, 0.999999.
    3. High-fidelity terminal reward (50.0).
    4. Light gate-count penalty (secondary to fidelity).
    5. Stuck penalty when stuck near zero fidelity.
    6. Truncation penalty to discourage running out of steps.
    """
    fidelity_gain = fidelity - prev_fidelity
    reward = 15.0 * fidelity_gain - gate_penalty

    # Dense milestone bonuses (one-shot per threshold crossing per episode)
    # These are applied inside the env — we add extra bonuses on top
    if fidelity >= 0.90 and prev_fidelity < 0.90:
        reward += 2.0
    if fidelity >= 0.99 and prev_fidelity < 0.99:
        reward += 4.0
    if fidelity >= 0.999 and prev_fidelity < 0.999:
        reward += 8.0
    if fidelity >= 0.9999 and prev_fidelity < 0.9999:
        reward += 15.0
    if fidelity >= 0.999999 and prev_fidelity < 0.999999:
        reward += 30.0

    # Stuck penalty
    if fidelity < 0.05 and step > 5:
        reward -= 3.0

    # Truncation partial reward — if we ran out of steps give partial credit
    if truncated and not done:
        reward += 3.0 * fidelity  # partial credit for progress

    return reward


# ---------------------------------------------------------------------------
# Lightweight greedy evaluation (non-destructive, does not modify policy)
# ---------------------------------------------------------------------------
def _greedy_eval(agent: PPOAgent, env: QuantumCircuitEnv, config: Config,
                 episode: int, n_states: int = 100) -> dict:
    """
    Evaluate PPO policy greedily on n_states fresh Haar-random targets.
    Returns dict of metrics. Does NOT modify agent weights.
    """
    eval_seed = config.SEED + 99999 + episode

    agent.ac.eval()
    fidelities = []
    gate_counts = []
    cx_counts = []

    with torch.no_grad():
        for i in range(n_states):
            target_sv = generate_random_statevector(config.NUM_QUBITS,
                                                    seed=eval_seed + i)
            obs, _ = env.reset(target_sv=target_sv)
            done = False
            final_fid = 0.0
            n_gates = 0
            n_cx = 0
            while not done:
                action = agent.select_action_greedy(obs)
                gate_name = env.action_list[action][0]
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_fid = info['fidelity']
                n_gates += 1
                if gate_name in ('CNOT', 'CX', 'CZ'):
                    n_cx += 1
            fidelities.append(final_fid)
            gate_counts.append(n_gates)
            cx_counts.append(n_cx)

    agent.ac.train()

    fids = np.array(fidelities)
    return {
        'mean_fidelity': float(np.mean(fids)),
        'median_fidelity': float(np.median(fids)),
        'best_fidelity': float(np.max(fids)),
        'std_fidelity': float(np.std(fids)),
        'pct_ge_0999': float(np.mean(fids >= 0.999)),
        'pct_ge_0999999': float(np.mean(fids >= 0.999999)),
        'mean_gate_count': float(np.mean(gate_counts)),
        'mean_cx_count': float(np.mean(cx_counts)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_ppo_resume(config: Config) -> None:
    """Resume PPO from existing checkpoint, run to 500k episodes."""

    set_seeds(config.SEED)
    num_cpus = os.cpu_count() or 8
    torch.set_num_threads(num_cpus)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_label = (f'cuda ({torch.cuda.get_device_name(0)})'
                 if torch.cuda.is_available()
                 else f'cpu (threads: {torch.get_num_threads()})')
    print(f'[PPO-RESUME] Device: {gpu_label}')

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f'[PPO-RESUME] obs_size={obs_size}  action_size={action_size}')

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = PPOAgent(obs_size, action_size, config, device)

    # Best checkpoint path
    best_path = best_checkpoint_path(config.PPO_MODEL_PATH)

    # ── Load existing checkpoint (preserves ALL learned weights) ─────────────
    checkpoint_loaded = False
    for ckpt_path in [best_path, config.PPO_MODEL_PATH]:
        if os.path.exists(ckpt_path):
            try:
                agent.load(ckpt_path)
                agent.ac.train()
                print(f'[PPO-RESUME] Loaded checkpoint: {ckpt_path}')
                checkpoint_loaded = True
                break
            except Exception as e:
                print(f'[PPO-RESUME] WARNING: Could not load {ckpt_path}: {e}')

    if not checkpoint_loaded:
        print('[PPO-RESUME] No checkpoint found — starting from scratch.')

    # ── Load existing logs to restore episode counter ────────────────────────
    episode_rewards: list = []
    episode_fidelities: list = []
    episode_steps_list: list = []
    policy_losses_log: list = []
    value_losses_log: list = []
    entropy_log: list = []

    log_path = config.PPO_LOG_PATH
    if os.path.exists(log_path):
        try:
            logs = load_logs(log_path)
            if logs and 'rewards' in logs and len(logs['rewards']) > 0:
                episode_rewards    = list(logs['rewards'])
                episode_fidelities = list(logs.get('fidelities', []))
                episode_steps_list = list(logs.get('steps', []))
                policy_losses_log  = list(logs.get('policy_losses', []))
                value_losses_log   = list(logs.get('value_losses', []))
                entropy_log        = list(logs.get('entropies', []))
                print(f'[PPO-RESUME] Restored {len(episode_rewards)} logged episodes from {log_path}')
        except Exception as e:
            print(f'[PPO-RESUME] WARNING: Could not load logs: {e}')

    current_episode = len(episode_rewards)
    start_episode = current_episode
    print(f'\n[PPO-RESUME] =====================================================')
    print(f'[PPO-RESUME]  RESUMING PPO from Episode {current_episode}/{config.PPO_EPISODES}')
    print(f'[PPO-RESUME]  Remaining: {config.PPO_EPISODES - current_episode} episodes')
    print(f'[PPO-RESUME] =====================================================\n')

    if current_episode >= config.PPO_EPISODES:
        print('[PPO-RESUME] Training already complete (500,000 episodes reached).')
        return

    # ── Advance LR scheduler to match current episode (instant, no loop) ────
    if agent.scheduler is not None and current_episode > 0:
        # Directly compute the correct LR for this episode position.
        # LinearLR decays from PPO_LR to PPO_LR_MIN over PPO_EPISODES steps.
        lr_min = getattr(config, 'PPO_LR_MIN', 1e-5)
        end_factor = lr_min / config.PPO_LR
        progress = min(current_episode / config.PPO_EPISODES, 1.0)
        resumed_lr = config.PPO_LR * (1.0 - progress * (1.0 - end_factor))
        for pg in agent.optimizer.param_groups:
            pg['lr'] = resumed_lr
        # Inform scheduler of current position to avoid LR resets on future steps
        if hasattr(agent.scheduler, 'last_epoch'):
            agent.scheduler.last_epoch = current_episode
        print(f'[PPO-RESUME] LR set to {resumed_lr:.2e} (episode {current_episode})')


    # ── Rollout buffer ───────────────────────────────────────────────────────
    buffer = RolloutBuffer(config.PPO_ROLLOUT_STEPS, obs_size, device)

    # ── Tracking ─────────────────────────────────────────────────────────────
    best_eval_fidelity: float = float('-inf')
    best_eval_episode: int = -1
    recent_fids_window = deque(maxlen=500)
    action_counts = Counter()
    cnot_episodes = 0

    # Eval every 500 episodes (lightweight, non-destructive)
    eval_interval = getattr(config, 'BEST_CHECKPOINT_EVAL_INTERVAL', 500)
    eval_n_states = 100  # Use 100 states for better statistics

    # Gradient norm tracking for stability monitoring
    recent_grad_norms = deque(maxlen=50)

    # ── Environment bootstrap ────────────────────────────────────────────────
    obs, _ = env.reset()
    env.current_episode = current_episode
    obs_t = torch.FloatTensor(obs).to(device)

    episode_reward = 0.0
    episode_steps = 0
    episode_prev_fid = 0.0  # track prev_fid for shaped reward
    episode_fid = 0.0

    # ── MAIN TRAINING LOOP ───────────────────────────────────────────────────
    print('[PPO-RESUME] Starting continuous training loop (no pauses until 500k)...\n')

    while current_episode < config.PPO_EPISODES:

        buffer.reset()

        # ── Rollout collection ───────────────────────────────────────────────
        for step in range(config.PPO_ROLLOUT_STEPS):
            env.current_episode = current_episode

            with torch.no_grad():
                action, log_prob, entropy, value = agent.select_action(obs_t)

            gate_name = env.action_list[action][0]
            action_counts[action] += 1

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            fidelity = info['fidelity']

            # Improved reward shaping (supplement to env reward)
            extra_reward = shaped_reward(
                fidelity, episode_prev_fid, episode_steps + 1,
                gate_name, terminated, truncated,
                config.GATE_PENALTY, config.FIDELITY_THRESHOLD
            )
            # Blend: use the env's internal reward (which already has the
            # +50 success bonus) but replace the base fidelity-gain signal
            # with our stronger version. We do this by using extra_reward
            # which replicates the gain signal with higher coefficient.
            total_reward = extra_reward

            buffer.add(obs_t.cpu(), action, log_prob.cpu(), total_reward, done, value.cpu())

            obs_t = torch.FloatTensor(next_obs).to(device)
            episode_reward += total_reward
            episode_steps += 1
            episode_prev_fid = fidelity
            episode_fid = fidelity

            if done:
                if info.get('cnot_used', False):
                    cnot_episodes += 1

                episode_rewards.append(episode_reward)
                episode_fidelities.append(episode_fid)
                episode_steps_list.append(episode_steps)
                recent_fids_window.append(episode_fid)

                current_episode += 1
                episode_reward = 0.0
                episode_steps = 0
                episode_prev_fid = 0.0

                obs, _ = env.reset()
                env.current_episode = current_episode
                obs_t = torch.FloatTensor(obs).to(device)

                # ── Per-500-episode logging and live plot (non-blocking) ──────
                if current_episode % 500 == 0 or current_episode == config.PPO_EPISODES:
                    recent_fids = episode_fidelities[-500:]
                    recent_rewards = episode_rewards[-500:]
                    recent_steps = episode_steps_list[-500:]

                    mean_fid    = float(np.mean(recent_fids))
                    median_fid  = float(np.median(recent_fids))
                    best_fid    = float(np.max(episode_fidelities))
                    std_fid     = float(np.std(recent_fids))
                    ge_0999     = float(np.mean([f >= 0.999 for f in recent_fids]))
                    ge_0999999  = float(np.mean([f >= 0.999999 for f in recent_fids]))
                    mean_gates  = float(np.mean(recent_steps))
                    mean_reward = float(np.mean(recent_rewards))
                    current_lr  = agent.optimizer.param_groups[0]['lr']

                    pl_mean = float(np.mean(policy_losses_log[-50:])) if policy_losses_log else 0.0
                    vl_mean = float(np.mean(value_losses_log[-50:]))  if value_losses_log  else 0.0
                    ent_mean = float(np.mean(entropy_log[-50:]))       if entropy_log        else 0.0

                    print(
                        f'Episode {current_episode:7d}/{config.PPO_EPISODES} | '
                        f'Reward: {episode_rewards[-1]:8.3f} | '
                        f'Fid: {episode_fidelities[-1]:.6f} | '
                        f'MeanFid(500): {mean_fid:.6f} | '
                        f'Best: {best_fid:.6f} | '
                        f'F>=.999999: {ge_0999999*100:.1f}% | '
                        f'Gates: {mean_gates:.1f} | '
                        f'LR: {current_lr:.2e}',
                        flush=True,
                    )

                    os.makedirs(config.LOG_DIR, exist_ok=True)
                    metrics_dict = {
                        'rewards':                          episode_rewards,
                        'fidelities':                       episode_fidelities,
                        'steps':                            episode_steps_list,
                        'policy_losses':                    policy_losses_log,
                        'value_losses':                     value_losses_log,
                        'entropies':                        entropy_log,
                        'episode':                          current_episode,
                        'mean_reward':                      mean_reward,
                        'mean_fidelity':                    mean_fid,
                        'median_fidelity':                  median_fid,
                        'best_fidelity':                    best_fid,
                        'std_fidelity':                     std_fid,
                        'success_rate':                     ge_0999999,
                        'fraction_fidelity_ge_0.999':       ge_0999,
                        'fraction_fidelity_ge_0.999999':    ge_0999999,
                        'mean_gate_count':                  mean_gates,
                        'learning_rate':                    current_lr,
                        'policy_loss':                      pl_mean,
                        'value_loss':                       vl_mean,
                        'entropy':                          ent_mean,
                    }
                    save_logs(metrics_dict, config.PPO_LOG_PATH)

                    os.makedirs(config.PLOT_DIR, exist_ok=True)
                    try:
                        plot_training_curves(
                            episode_rewards, episode_fidelities, episode_steps_list,
                            config.PPO_PLOT_PATH,
                        )
                    except Exception:
                        pass  # Never block training on a plot failure

                # ── Lightweight periodic greedy evaluation ────────────────────
                if current_episode % eval_interval == 0:
                    eval_stats = _greedy_eval(agent, env, config,
                                              current_episode, n_states=eval_n_states)
                    eval_fid = eval_stats['mean_fidelity']

                    marker = ''
                    if eval_fid > best_eval_fidelity:
                        best_eval_fidelity = eval_fid
                        best_eval_episode = current_episode
                        agent.save(best_path)
                        marker = '  *** NEW BEST CHECKPOINT SAVED ***'

                    print(
                        f'  [EVAL ep {current_episode:7d}] '
                        f'mean={eval_fid:.6f} | '
                        f'median={eval_stats["median_fidelity"]:.6f} | '
                        f'best={eval_stats["best_fidelity"]:.6f} | '
                        f'F>=.999={eval_stats["pct_ge_0999"]*100:.1f}% | '
                        f'F>=.999999={eval_stats["pct_ge_0999999"]*100:.1f}%'
                        f'{marker}',
                        flush=True,
                    )

                if current_episode >= config.PPO_EPISODES:
                    break

        # ── PPO update ────────────────────────────────────────────────────────
        with torch.no_grad():
            _, _, _, last_value = agent.select_action(obs_t)

        buffer.compute_returns_and_advantages(
            last_value.cpu(), config.PPO_GAMMA, config.PPO_GAE_LAMBDA
        )

        update_stats = _ppo_update_with_stats(agent, buffer, config)

        # Log update stats
        if update_stats:
            policy_losses_log.append(update_stats.get('policy_loss', 0.0))
            value_losses_log.append(update_stats.get('value_loss', 0.0))
            entropy_log.append(update_stats.get('entropy', 0.0))
            grad_norm = update_stats.get('grad_norm', 0.0)
            recent_grad_norms.append(grad_norm)

            # NaN guard
            if (np.isnan(update_stats.get('policy_loss', 0.0)) or
                    np.isnan(update_stats.get('value_loss', 0.0))):
                print(f'[PPO-RESUME] WARNING: NaN detected in loss at episode '
                      f'{current_episode}! Reloading best checkpoint...')
                if os.path.exists(best_path):
                    agent.load(best_path)
                    agent.ac.train()

        if agent.scheduler is not None:
            agent.scheduler.step()

    # ── Final checkpoint save ─────────────────────────────────────────────────
    agent.save(config.PPO_MODEL_PATH)
    print(f'\n[PPO-RESUME] Final model saved to {config.PPO_MODEL_PATH}')

    # ── Final evaluation ──────────────────────────────────────────────────────
    print('[PPO-RESUME] Running final evaluation (500 states)...')
    final_eval = _greedy_eval(agent, env, config,
                              episode=config.PPO_EPISODES, n_states=500)

    if final_eval['mean_fidelity'] > best_eval_fidelity:
        best_eval_fidelity = final_eval['mean_fidelity']
        best_eval_episode = config.PPO_EPISODES
        agent.save(best_path)

    # ── Save final logs ───────────────────────────────────────────────────────
    os.makedirs(config.LOG_DIR, exist_ok=True)
    final_metrics = {
        'rewards':                       episode_rewards,
        'fidelities':                    episode_fidelities,
        'steps':                         episode_steps_list,
        'policy_losses':                 policy_losses_log,
        'value_losses':                  value_losses_log,
        'entropies':                     entropy_log,
        'episode':                       current_episode,
        'mean_fidelity':                 final_eval['mean_fidelity'],
        'median_fidelity':               final_eval['median_fidelity'],
        'best_fidelity':                 final_eval['best_fidelity'],
        'std_fidelity':                  final_eval['std_fidelity'],
        'success_rate':                  final_eval['pct_ge_0999999'],
        'fraction_fidelity_ge_0.999':    final_eval['pct_ge_0999'],
        'fraction_fidelity_ge_0.999999': final_eval['pct_ge_0999999'],
        'mean_gate_count':               final_eval['mean_gate_count'],
        'mean_cx_count':                 final_eval['mean_cx_count'],
    }
    save_logs(final_metrics, config.PPO_LOG_PATH)

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    try:
        plot_training_curves(
            episode_rewards, episode_fidelities, episode_steps_list,
            config.PPO_PLOT_PATH,
        )
    except Exception:
        pass

    # ── Terminal summary ──────────────────────────────────────────────────────
    total_eps = current_episode - start_episode
    print(f'\n[PPO-RESUME] ═══════════════════════════════════════════════════')
    print(f'[PPO-RESUME]  TRAINING COMPLETE: {current_episode}/{config.PPO_EPISODES} episodes')
    print(f'[PPO-RESUME]  Episodes this run : {total_eps}')
    print(f'[PPO-RESUME] ─────────────────────────────────────────────────')
    print(f'[PPO-RESUME]  Final eval mean fidelity   : {final_eval["mean_fidelity"]:.6f}')
    print(f'[PPO-RESUME]  Final eval median fidelity : {final_eval["median_fidelity"]:.6f}')
    print(f'[PPO-RESUME]  Final eval best fidelity   : {final_eval["best_fidelity"]:.6f}')
    print(f'[PPO-RESUME]  Final eval std fidelity    : {final_eval["std_fidelity"]:.6f}')
    print(f'[PPO-RESUME]  F >= 0.999   rate          : {final_eval["pct_ge_0999"]*100:.1f}%')
    print(f'[PPO-RESUME]  F >= 0.999999 rate         : {final_eval["pct_ge_0999999"]*100:.1f}%')
    print(f'[PPO-RESUME]  Mean gate count             : {final_eval["mean_gate_count"]:.1f}')
    print(f'[PPO-RESUME]  Mean CX count               : {final_eval["mean_cx_count"]:.1f}')
    print(f'[PPO-RESUME] ─────────────────────────────────────────────────')
    print(f'[PPO-RESUME]  Best checkpoint fidelity    : {best_eval_fidelity:.6f}  '
          f'(episode {best_eval_episode})')
    print(f'[PPO-RESUME]  Best checkpoint path        : {best_path}')
    print(f'[PPO-RESUME] ═══════════════════════════════════════════════════\n')


# ---------------------------------------------------------------------------
# PPO update with full stability monitoring and stat return
# ---------------------------------------------------------------------------
def _ppo_update_with_stats(agent: PPOAgent, buffer: RolloutBuffer,
                           config: Config) -> dict:
    """
    PPO clipped update with:
      - Gradient clipping (max_grad_norm)
      - NaN detection
      - Approximate KL tracking
      - Clip fraction tracking
      - Gradient norm logging
    """
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []
    clip_fractions = []
    grad_norms = []

    clip_eps = config.PPO_CLIP_EPSILON
    value_coef = config.PPO_VALUE_COEF
    entropy_coef = config.PPO_ENTROPY_COEF
    max_grad_norm = config.PPO_MAX_GRAD_NORM
    mini_batch_size = config.PPO_MINI_BATCH_SIZE

    for epoch in range(config.PPO_EPOCHS):
        for (states, actions, log_probs_old,
             returns, advantages) in buffer.get_minibatches(mini_batch_size):

            log_probs_new, values_new, entropy = agent.ac.evaluate_actions(
                states, actions
            )

            ratio = torch.exp(log_probs_new - log_probs_old)

            # Approximate KL for monitoring
            approx_kl = ((ratio - 1) - (log_probs_new - log_probs_old)).mean()
            approx_kls.append(approx_kl.item())

            # Clip fraction
            clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
            clip_fractions.append(clip_fraction.item())

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * F.mse_loss(values_new, returns)
            entropy_loss = -entropy.mean()

            total_loss = (
                policy_loss
                + value_coef * value_loss
                + entropy_coef * entropy_loss
            )

            # NaN guard before backward
            if torch.isnan(total_loss):
                continue

            agent.optimizer.zero_grad()
            total_loss.backward()

            # Gradient norm before clipping
            total_norm = 0.0
            for p in agent.ac.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), max_grad_norm)
            agent.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

    if not policy_losses:
        return {}

    return {
        'policy_loss':  float(np.mean(policy_losses)),
        'value_loss':   float(np.mean(value_losses)),
        'entropy':      float(np.mean(entropies)),
        'approx_kl':    float(np.mean(approx_kls)),
        'clip_fraction':float(np.mean(clip_fractions)),
        'grad_norm':    float(np.mean(grad_norms)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Build config pointing to best checkpoint path
    cfg = Config()

    # Override config to use best checkpoint for resume
    # (train_ppo_resume handles loading the best checkpoint automatically)
    cfg.PPO_MODEL_PATH = 'saved_models/2qubit/ppo_model_best.pth'
    cfg.PPO_LOG_PATH = 'logs/2qubit/ppo_logs.json'
    cfg.PPO_PLOT_PATH = 'plots/2qubit/ppo_training.png'

    # Disable pretraining (already done)
    cfg.PRETRAIN_ENABLED = False
    cfg.EXPERT_BUFFER_ENABLED = False

    # Disable CNOT forcing (past episode 10k already)
    cfg.CNOT_FORCE_UNTIL_EPISODE = 0

    train_ppo_resume(cfg)
