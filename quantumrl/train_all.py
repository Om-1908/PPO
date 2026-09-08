"""
train_all.py
------------
Master training script: trains all 4 model combinations in sequence.

Order:
  1. 1-Qubit DQN   (pretrain 20k states x 30 epochs -> RL 25k episodes)
  2. 1-Qubit PPO   (pretrain 20k states x 30 epochs -> RL 25k episodes)
  3. 2-Qubit DQN   (pretrain 100k states x 15 epochs -> RL 60k episodes)
  4. 2-Qubit PPO   (pretrain 100k states x 15 epochs -> RL 60k episodes)

Run with:
    cd d:\Tatva-main\quantumrl
    python train_all.py

Mid-training fidelity checks are logged automatically every 100 episodes.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _sep(msg: str, width: int = 70) -> None:
    line = '=' * width
    print(f'\n{line}')
    print(f'  {msg}')
    print(f'{line}\n', flush=True)


def run_1qubit_training():
    """Train both DQN and PPO agents on 1-qubit synthesis."""
    from configs.config_1qubit import Config
    cfg = Config()

    _sep('PHASE 1 / 4 — 1-Qubit DQN Training')
    print(f'  Action space  : {cfg.NUM_QUBITS} qubit, 292 actions (4 fixed + 3x96 rotation)')
    print(f'  Pretrain      : {cfg.PRETRAIN_MAX_STATES:,} states x {cfg.PRETRAIN_EPOCHS} epochs')
    print(f'  RL episodes   : {cfg.DQN_EPISODES:,}')
    print(f'  Target        : fidelity >= {cfg.FIDELITY_THRESHOLD}', flush=True)

    from train_dqn import train_dqn
    t0 = time.time()
    train_dqn(cfg)
    print(f'  [1Q DQN] Finished in {(time.time()-t0)/60:.1f} min', flush=True)

    _sep('PHASE 2 / 4 — 1-Qubit PPO Training')
    print(f'  Action space  : {cfg.NUM_QUBITS} qubit, 292 actions')
    print(f'  Pretrain      : {cfg.PRETRAIN_MAX_STATES:,} states x {cfg.PRETRAIN_EPOCHS} epochs')
    print(f'  RL episodes   : {cfg.PPO_EPISODES:,}', flush=True)

    # Reload fresh config to avoid cross-contamination between DQN / PPO runs
    cfg2 = Config()
    from train_ppo import train_ppo
    t0 = time.time()
    train_ppo(cfg2)
    print(f'  [1Q PPO] Finished in {(time.time()-t0)/60:.1f} min', flush=True)


def run_2qubit_training():
    """Train both DQN and PPO agents on 2-qubit synthesis."""
    from configs.config_2qubit import Config
    cfg = Config()

    _sep('PHASE 3 / 4 — 2-Qubit DQN Training')
    print(f'  Action space  : {cfg.NUM_QUBITS} qubit, 596 actions (16 fixed + 3x96x2 rotation + 4 two-qubit)')
    print(f'  Pretrain      : {cfg.PRETRAIN_MAX_STATES:,} states x {cfg.PRETRAIN_EPOCHS} epochs')
    print(f'  RL episodes   : {cfg.DQN_EPISODES:,}')
    print(f'  CNOT forcing  : first {cfg.CNOT_FORCE_UNTIL_EPISODE:,} episodes', flush=True)

    from train_dqn import train_dqn
    t0 = time.time()
    train_dqn(cfg)
    print(f'  [2Q DQN] Finished in {(time.time()-t0)/60:.1f} min', flush=True)

    _sep('PHASE 4 / 4 — 2-Qubit PPO Training')
    cfg2 = Config()
    from train_ppo import train_ppo
    t0 = time.time()
    train_ppo(cfg2)
    print(f'  [2Q PPO] Finished in {(time.time()-t0)/60:.1f} min', flush=True)


def run_mid_eval(n_qubits: int, n_states: int = 20) -> None:
    """Quick mid-training fidelity snapshot after each phase."""
    import numpy as np
    import torch
    from quantum_env import QuantumCircuitEnv
    from utils import generate_random_statevector
    from dqn_agent import DQNAgent
    from ppo_agent import PPOAgent

    if n_qubits == 1:
        from configs.config_1qubit import Config
    else:
        from configs.config_2qubit import Config

    cfg = Config()
    env = QuantumCircuitEnv(cfg)
    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n
    device      = torch.device('cpu')

    results = {}

    for agent_type, model_path in [('DQN', cfg.DQN_MODEL_PATH), ('PPO', cfg.PPO_MODEL_PATH)]:
        if not os.path.exists(model_path):
            results[agent_type] = 'model not found'
            continue

        fids = []
        success = 0

        if agent_type == 'DQN':
            agent = DQNAgent(obs_size, action_size, cfg)
            agent.load(model_path)
            agent.epsilon = 0.0
            eval_fn = agent.select_action
        else:
            agent = PPOAgent(obs_size, action_size, cfg, device)
            agent.load(model_path)
            eval_fn = agent.select_action_greedy

        for i in range(n_states):
            sv = generate_random_statevector(n_qubits, seed=99000 + i)
            obs, _ = env.reset(target_sv=sv)
            done = False
            fid  = 0.0
            with torch.no_grad():
                while not done:
                    a = eval_fn(obs)
                    obs, _, terminated, truncated, info = env.step(a)
                    done = terminated or truncated
                    fid  = info['fidelity']
            fids.append(fid)
            if fid >= cfg.FIDELITY_THRESHOLD:
                success += 1

        results[agent_type] = {
            'mean_fid'   : float(np.mean(fids)),
            'success_rate': success / n_states,
            'min_fid'    : float(np.min(fids)),
        }

    print(f'\n  [{n_qubits}Q Mid-Eval on {n_states} Haar states]')
    for agent_type, r in results.items():
        if isinstance(r, str):
            print(f'    {agent_type}: {r}')
        else:
            print(f'    {agent_type}: mean={r["mean_fid"]:.4f}  '
                  f'success={r["success_rate"]*100:.1f}%  '
                  f'min={r["min_fid"]:.4f}')
    print(flush=True)


if __name__ == '__main__':
    total_start = time.time()
    print('\n' + '=' * 70)
    print('  QuantumRL — Full Training Pipeline')
    print('  120,000 Haar states | DQN + PPO | 1-Qubit + 2-Qubit')
    print('  Target: fidelity >= 0.99 for any Haar-random state')
    print('=' * 70 + '\n', flush=True)

    # ── 1-Qubit ────────────────────────────────────────────────────────────
    run_1qubit_training()
    run_mid_eval(n_qubits=1, n_states=50)

    # ── 2-Qubit ────────────────────────────────────────────────────────────
    run_2qubit_training()
    run_mid_eval(n_qubits=2, n_states=50)

    total_mins = (time.time() - total_start) / 60
    print(f'\n{"=" * 70}')
    print(f'  TRAINING COMPLETE in {total_mins:.1f} minutes')
    print(f'{"=" * 70}\n', flush=True)
