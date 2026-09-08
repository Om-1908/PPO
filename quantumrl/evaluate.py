import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from quantum_env import QuantumCircuitEnv
from simplify import verify_synthesis_candidate, compute_depth
from utils import best_checkpoint_path, generate_target_states


def evaluate_agent(agent, env: QuantumCircuitEnv, test_states, config: Config, model_name: str = "DQN"):
    """
    Run greedy deterministic evaluation of an agent on held-out test set with full verification.
    """
    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

    if hasattr(agent, 'ac'):
        agent.ac.eval()

    fidelities = []
    gate_counts = []
    depths = []
    cx_counts = []
    successes_0999 = []
    successes_0999999 = []
    synth_times = []
    failed_episodes = 0
    verified_archive = []

    for idx, target_sv in enumerate(test_states):
        t_start = time.time()
        obs, _ = env.reset(target_sv=target_sv)
        done = False
        final_fidelity = 0.0

        while not done:
            if model_name == "DQN":
                action = agent.select_action(obs)
            else:
                action = agent.select_action_greedy(obs)

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_fidelity = info['fidelity']

        synth_time = time.time() - t_start
        synth_times.append(synth_time)

        # Run 12-step verification pipeline
        ver_res = verify_synthesis_candidate(
            target_sv=target_sv,
            applied_actions=env.applied_actions,
            n_qubits=config.NUM_QUBITS,
            fidelity_threshold=config.FIDELITY_THRESHOLD,
        )

        is_verified = ver_res['verified']
        if is_verified:
            fid = ver_res['final_fidelity']
            g_count = ver_res['gate_count']
            d_count = ver_res['circuit_depth']
            c_count = ver_res['entangling_gate_count']
            simp_seq = ver_res['simplified_actions']
        else:
            fid = final_fidelity
            g_count = len(env.applied_actions)
            d_count = compute_depth(env.applied_actions, config.NUM_QUBITS)
            c_count = sum(1 for g, _, _ in env.applied_actions if g in ('CNOT', 'CZ'))
            simp_seq = env.applied_actions
            failed_episodes += 1

        fidelities.append(fid)
        gate_counts.append(g_count)
        depths.append(d_count)
        cx_counts.append(c_count)

        succ_0999 = float(fid >= 0.999)
        succ_0999999 = float(fid >= 0.999999 and is_verified)

        successes_0999.append(succ_0999)
        successes_0999999.append(succ_0999999)

        if is_verified:
            verified_archive.append({
                "num_qubits": config.NUM_QUBITS,
                "target_statevector": [
                    [float(amp.real), float(amp.imag)] for amp in target_sv
                ],
                "gate_sequence": [
                    {"name": g, "qubits": list(q) if isinstance(q, (tuple, list)) else [q], "theta": a}
                    for g, q, a in simp_seq
                ],
                "gate_count": g_count,
                "circuit_depth": d_count,
                "entangling_gate_count": c_count,
                "fidelity": float(fid),
                "model": model_name,
                "test_index": idx,
                "verified": True
            })

    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    return {
        'fidelities': fidelities,
        'gate_counts': gate_counts,
        'depths': depths,
        'cx_counts': cx_counts,
        'successes_0999': successes_0999,
        'successes_0999999': successes_0999999,
        'synth_times': synth_times,
        'failed_episodes': failed_episodes,
        'verified_archive': verified_archive,
    }


def plot_comparison(
    dqn_results: dict,
    ppo_results: dict,
    save_path: str,
    num_qubits: int = 2,
) -> None:
    """Save a four-metric grouped bar chart comparing DQN vs. PPO."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    metrics = ['Mean Fidelity', 'Median Fidelity', 'Success Rate (F>=0.999999)', 'Avg Gate Count']

    dqn_vals = [
        float(np.mean(dqn_results['fidelities'])),
        float(np.median(dqn_results['fidelities'])),
        float(np.mean(dqn_results['successes_0999999'])) * 100.0,
        float(np.mean(dqn_results['gate_counts'])),
    ]
    ppo_vals = [
        float(np.mean(ppo_results['fidelities'])),
        float(np.median(ppo_results['fidelities'])),
        float(np.mean(ppo_results['successes_0999999'])) * 100.0,
        float(np.mean(ppo_results['gate_counts'])),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f'DQN vs PPO — {num_qubits}-Qubit Evaluation Comparison ({len(dqn_results["fidelities"])} Test States)', color='white', fontsize=14, fontweight='bold')

    colors_dqn = '#e94560'
    colors_ppo = '#0f9b8e'

    for ax, metric, dv, pv in zip(axes, metrics, dqn_vals, ppo_vals):
        ax.set_facecolor('#16213e')
        bars = ax.bar(['DQN', 'PPO'], [dv, pv], color=[colors_dqn, colors_ppo],
                      width=0.5, edgecolor='white', linewidth=0.6)

        for bar, val in zip(bars, [dv, pv]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01 * max(dv, pv, 1),
                f'{val:.4f}' if 'Fidelity' in metric else f'{val:.1f}' + ('%' if '%' in metric else ''),
                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold',
            )

        ax.set_title(metric, color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#aaaaaa')
        ax.spines['bottom'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.set_ylim(0, max(dv, pv, 0.1) * 1.25)
        ax.grid(True, axis='y', alpha=0.2, color='#444')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[evaluate] Comparison chart saved -> {save_path}")


def run_evaluation(config: Config, use_best: bool = False) -> None:
    test_seed = config.SEED + 999
    num_test_states = config.NUM_TEST_STATES
    print(f"[evaluate] Generating {num_test_states} {config.NUM_QUBITS}-qubit unseen test states (seed={test_seed}) ...")
    test_states = generate_target_states(
        config.NUM_QUBITS, num_test_states, seed=test_seed
    )

    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dqn_path = best_checkpoint_path(config.DQN_MODEL_PATH) if use_best else config.DQN_MODEL_PATH
    ppo_path = best_checkpoint_path(config.PPO_MODEL_PATH) if use_best else config.PPO_MODEL_PATH

    if not os.path.exists(dqn_path):
        dqn_path = config.DQN_MODEL_PATH
    if not os.path.exists(ppo_path):
        ppo_path = config.PPO_MODEL_PATH

    dqn_agent = DQNAgent(obs_size, action_size, config)
    if os.path.exists(dqn_path):
        dqn_agent.load(dqn_path)
    else:
        print(f"[evaluate] WARNING: DQN checkpoint not found at {dqn_path}. Using initial weights.")

    ppo_agent = PPOAgent(obs_size, action_size, config, device)
    if os.path.exists(ppo_path):
        ppo_agent.load(ppo_path)
    else:
        print(f"[evaluate] WARNING: PPO checkpoint not found at {ppo_path}. Using initial weights.")

    print(f"\n[evaluate] Evaluating DQN on {config.NUM_QUBITS}-qubit test set ...")
    dqn_res = evaluate_agent(dqn_agent, env, test_states, config, model_name="DQN")

    print(f"[evaluate] Evaluating PPO on {config.NUM_QUBITS}-qubit test set ...")
    ppo_res = evaluate_agent(ppo_agent, env, test_states, config, model_name="PPO")

    # Compute comprehensive 16 metrics
    def calc_metrics(res):
        fids = res['fidelities']
        gates = res['gate_counts']
        depths = res['depths']
        cxs = res['cx_counts']
        times = res['synth_times']
        return {
            'mean_fid': float(np.mean(fids)),
            'median_fid': float(np.median(fids)),
            'std_fid': float(np.std(fids)),
            'best_fid': float(np.max(fids)),
            'pct_0999': float(np.mean(res['successes_0999'])) * 100.0,
            'pct_0999999': float(np.mean(res['successes_0999999'])) * 100.0,
            'mean_gates': float(np.mean(gates)),
            'median_gates': float(np.median(gates)),
            'mean_depth': float(np.mean(depths)),
            'median_depth': float(np.median(depths)),
            'mean_cx': float(np.mean(cxs)),
            'median_cx': float(np.median(cxs)),
            'success_rate': float(np.mean(res['successes_0999999'])) * 100.0,
            'avg_time': float(np.mean(times)),
            'max_time': float(np.max(times)),
            'failed_episodes': res['failed_episodes'],
        }

    m_dqn = calc_metrics(dqn_res)
    m_ppo = calc_metrics(ppo_res)

    print("\n" + "=" * 70)
    print(f"  TATVA FINAL 2-QUBIT EVALUATION REPORT ({num_test_states} Unseen Haar Targets)")
    print("=" * 70)
    print(f"{'Metric':<32} | {'DQN':<16} | {'PPO':<16}")
    print("-" * 70)
    print(f"{'1. Mean Fidelity':<32} | {m_dqn['mean_fid']:<16.6f} | {m_ppo['mean_fid']:<16.6f}")
    print(f"{'2. Median Fidelity':<32} | {m_dqn['median_fid']:<16.6f} | {m_ppo['median_fid']:<16.6f}")
    print(f"{'3. Std Fidelity':<32} | {m_dqn['std_fid']:<16.6e} | {m_ppo['std_fid']:<16.6e}")
    print(f"{'4. Best Fidelity':<32} | {m_dqn['best_fid']:<16.6f} | {m_ppo['best_fid']:<16.6f}")
    print(f"{'5. % F >= 0.999':<32} | {m_dqn['pct_0999']:<15.1f}% | {m_ppo['pct_0999']:<15.1f}%")
    print(f"{'6. % F >= 0.999999':<32} | {m_dqn['pct_0999999']:<15.1f}% | {m_ppo['pct_0999999']:<15.1f}%")
    print(f"{'7. Mean Gate Count':<32} | {m_dqn['mean_gates']:<16.2f} | {m_ppo['mean_gates']:<16.2f}")
    print(f"{'8. Median Gate Count':<32} | {m_dqn['median_gates']:<16.1f} | {m_ppo['median_gates']:<16.1f}")
    print(f"{'9. Mean Depth':<32} | {m_dqn['mean_depth']:<16.2f} | {m_ppo['mean_depth']:<16.2f}")
    print(f"{'10. Median Depth':<32} | {m_dqn['median_depth']:<16.1f} | {m_ppo['median_depth']:<16.1f}")
    print(f"{'11. Mean CX Count':<32} | {m_dqn['mean_cx']:<16.2f} | {m_ppo['mean_cx']:<16.2f}")
    print(f"{'12. Median CX Count':<32} | {m_dqn['median_cx']:<16.1f} | {m_ppo['median_cx']:<16.1f}")
    print(f"{'13. Success Rate (F>=0.999999)':<32} | {m_dqn['success_rate']:<15.1f}% | {m_ppo['success_rate']:<15.1f}%")
    print(f"{'14. Avg Synthesis Time (s)':<32} | {m_dqn['avg_time']:<16.4f} | {m_ppo['avg_time']:<16.4f}")
    print(f"{'15. Max Synthesis Time (s)':<32} | {m_dqn['max_time']:<16.4f} | {m_ppo['max_time']:<16.4f}")
    print(f"{'16. Failed Episodes':<32} | {m_dqn['failed_episodes']:<16d} | {m_ppo['failed_episodes']:<16d}")
    print("=" * 70)

    # Save best verified circuits archive
    archive = dqn_res['verified_archive'] + ppo_res['verified_archive']
    archive_path = os.path.join(r"d:\Tatva-main", "outputs", "best_verified_circuits.json")
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    with open(archive_path, 'w') as f:
        json.dump(archive, f, indent=2)

    print(f"[evaluate] Saved {len(archive)} best verified circuits -> {archive_path}")

    save_plot_path = os.path.join(config.PLOT_DIR, 'dqn_vs_ppo_comparison.png')
    plot_comparison(dqn_res, ppo_res, save_plot_path, num_qubits=config.NUM_QUBITS)
    print("\n[evaluate] Evaluation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained DQN and PPO agents on a held-out test set.',
    )
    parser.add_argument(
        '--use-best',
        action='store_true',
        default=False,
        help='Load best checkpoints instead of final models.',
    )
    args = parser.parse_args()
    cfg = Config()
    run_evaluation(cfg, use_best=args.use_best)

