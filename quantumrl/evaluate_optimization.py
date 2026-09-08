"""
evaluate_optimization.py
------------------------
Generalization Benchmark Evaluation Script for TATVA PPO Structural Search Engine.

Evaluates the multi-strategy structural optimization engine across N unseen random
2-qubit Haar targets under a hard fidelity constraint (F >= 0.999999).

Tracks:
  - High-precision success rate (F >= 0.999999)
  - Mean & median final verified fidelity
  - Average raw PPO gate count vs final optimized gate count
  - Average gate reduction count & percentage
  - Average circuit depth reduction
  - Average entangling gate count
  - Search & optimization runtime
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from ppo_agent import PPOAgent
from utils import generate_random_statevector
from synthesis import synthesize_circuit


def run_benchmark_evaluation(num_targets: int = 15, max_candidates: int = 25, seed: int = 42):
    """Run benchmark evaluation over random 2-qubit Haar targets."""
    print("=" * 76)
    print(f"       TATVA PPO STRUCTURAL OPTIMIZATION BENCHMARK (N={num_targets} Targets)")
    print("=" * 76)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Running on device : {device}")

    config = Config()
    env = QuantumCircuitEnv(config)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(obs_size=obs_size, action_size=action_size, config=config, device=device)
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "ppo_paused_ep174500.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models", "2qubit", "ppo_model_best_best.pth")

    print(f" Loaded Model Path : {os.path.relpath(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["actor_critic_state_dict"] if "actor_critic_state_dict" in ckpt else ckpt
    agent.ac.load_state_dict(sd)
    agent.ac.eval()

    records = []
    np.random.seed(seed)

    print("-" * 76)
    print(f" {'#':2s} | {'Raw Gates':9s} | {'Opt Gates':9s} | {'Reduction':10s} | {'Depth':5s} | {'Ent':3s} | {'Qiskit Fresh F':14s} | {'Time':5s}")
    print("-" * 76)

    for i in range(num_targets):
        target_sv = generate_random_statevector(2, seed=seed + i * 100)
        t0 = time.time()
        res = synthesize_circuit(
            target_sv=target_sv,
            agent=agent,
            env=env,
            config=config,
            target_fidelity=0.999999,
            max_candidates=max_candidates,
            verbose=False
        )
        elapsed = time.time() - t0

        raw_g = res['raw_gate_count']
        opt_g = res['final_gate_count']
        red_g = raw_g - opt_g
        red_pct = (red_g / raw_g * 100.0) if raw_g > 0 else 0.0
        depth = res['circuit_depth']
        ent = res['entangling_gate_count']
        fid = res['final_fidelity']

        rec = {
            'target_idx': i + 1,
            'raw_gates': raw_g,
            'opt_gates': opt_g,
            'gate_reduction': red_g,
            'reduction_pct': red_pct,
            'depth': depth,
            'entangling': ent,
            'fidelity': fid,
            'is_hp': (fid >= 0.999999),
            'time': elapsed
        }
        records.append(rec)

        print(f" {i+1:02d} | {raw_g:9d} | {opt_g:9d} | {red_g:2d} ({red_pct:5.1f}%) | {depth:5d} | {ent:3d} | {fid:14.10f} | {elapsed:4.1f}s")

    print("-" * 76)

    # Statistical Aggregation
    hp_count = sum(1 for r in records if r['is_hp'])
    hp_rate = (hp_count / num_targets) * 100.0
    fidelities = [r['fidelity'] for r in records]
    raw_gates_list = [r['raw_gates'] for r in records]
    opt_gates_list = [r['opt_gates'] for r in records]
    reductions_list = [r['gate_reduction'] for r in records]
    red_pct_list = [r['reduction_pct'] for r in records]
    depths_list = [r['depth'] for r in records]
    ent_list = [r['entangling'] for r in records]
    times_list = [r['time'] for r in records]

    print("\n==========================================================================")
    print("                 BENCHMARK SUMMARY & GENERALIZATION METRICS")
    print("==========================================================================")
    print(f" Total Haar Targets Tested       : {num_targets}")
    print(f" High-Precision Success Rate     : {hp_count}/{num_targets} ({hp_rate:.2f}%) [F >= 0.999999]")
    print(f" Mean Final Verified Fidelity   : {np.mean(fidelities):.12f}")
    print(f" Median Final Verified Fidelity : {np.median(fidelities):.12f}")
    print(f" Average Raw PPO Gate Count     : {np.mean(raw_gates_list):.2f} gates")
    print(f" Average Final Optimized Gates  : {np.mean(opt_gates_list):.2f} gates")
    print(f" Average Gate Reduction         : {np.mean(reductions_list):.2f} gates ({np.mean(red_pct_list):.2f}%)")
    print(f" Median Gate Reduction          : {np.median(reductions_list):.1f} gates ({np.median(red_pct_list):.2f}%)")
    print(f" Average Circuit Depth          : {np.mean(depths_list):.2f}")
    print(f" Average Entangling Gates       : {np.mean(ent_list):.2f}")
    print(f" Average Search & Opt Runtime   : {np.mean(times_list):.2f} seconds")
    print("==========================================================================\n")


if __name__ == '__main__':
    run_benchmark_evaluation(num_targets=10, max_candidates=20, seed=100)
