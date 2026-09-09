import os
import sys
import time
import math
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector

# Add quantumrl and scratch to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('quantumrl'))
sys.path.insert(0, os.path.abspath('scratch'))

from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent
from scratch.validate_dqn_20_targets import synthesize_target_dqn

def generate_random_haar_targets(num_targets: int = 25, seed: int = 42) -> list:
    """Generate num_targets independent Haar-random 2-qubit statevectors."""
    np.random.seed(seed)
    targets = []
    for i in range(num_targets):
        sv = random_statevector(4, seed=seed + i).data
        sv = sv / np.linalg.norm(sv)
        targets.append(sv)
    return targets

def main():
    print("==========================================================================================================", flush=True)
    print("TATVA DQN GENERALIZATION TEST ON 25 UNSEEN RANDOM HAAR 2-QUBIT TARGETS", flush=True)
    print("==========================================================================================================", flush=True)

    config = Config()
    env = QuantumCircuitEnv(config)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    
    ckpt_path = os.path.join('saved_models', '2qubit', 'dqn_model_best.pth')
    print(f"[DQNAgent] Loading checkpoint: {ckpt_path}", flush=True)
    agent.load(ckpt_path)
    agent.q_net.eval()

    unseen_targets = generate_random_haar_targets(num_targets=25, seed=5000)

    basic_success_count = 0      # F >= 0.99
    hp_success_count = 0         # F >= 0.999999
    raw_gate_counts = []
    final_gate_counts = []
    depths = []
    entangling_counts = []
    fidelities = []
    gate_reductions = []
    search_times = []

    print(f"{'Target':<8}{'Raw Gates':<11}{'Final Gates':<13}{'Depth':<8}{'Ent':<6}{'Qiskit Fid':<16}{'HP Success':<12}{'Time (s)':<10}", flush=True)
    print("----------------------------------------------------------------------------------------------------------", flush=True)

    for idx, target_sv in enumerate(unseen_targets, 1):
        best_cand = synthesize_target_dqn(target_sv, agent, env, config, target_fidelity=0.999999, max_candidates=15, verbose=False)
        fid = best_cand['final_fidelity']
        raw_g = best_cand['raw_gate_count']
        final_g = best_cand['final_gate_count']
        d = best_cand['circuit_depth']
        ent = best_cand['entangling_gate_count']
        t = best_cand['search_time_seconds']

        if fid >= 0.99:
            basic_success_count += 1
        if fid >= 0.999999:
            hp_success_count += 1

        raw_gate_counts.append(raw_g)
        final_gate_counts.append(final_g)
        depths.append(d)
        entangling_counts.append(ent)
        fidelities.append(fid)
        gate_reductions.append(raw_g - final_g)
        search_times.append(t)

        print(
            f"U-{idx:02d}    "
            f"{raw_g:<11}"
            f"{final_g:<13}"
            f"{d:<8}"
            f"{ent:<6}"
            f"{fid:<16.10f}"
            f"{str(fid >= 0.999999):<12}"
            f"{t:<10.2f}",
            flush=True
        )

    print("----------------------------------------------------------------------------------------------------------", flush=True)
    print("GENERALIZATION TEST RESULTS SUMMARY (25 UNSEEN HAAR TARGETS):", flush=True)
    print(f" Basic Success Rate (F >= 0.99): {basic_success_count}/25 ({basic_success_count/25*100:.1f}%)", flush=True)
    print(f" High-Precision Success Rate (F >= 0.999999): {hp_success_count}/25 ({hp_success_count/25*100:.1f}%)", flush=True)
    print(f" Mean Qiskit Fidelity: {np.mean(fidelities):.10f}", flush=True)
    print(f" Median Qiskit Fidelity: {np.median(fidelities):.10f}", flush=True)
    print(f" Best Qiskit Fidelity: {np.max(fidelities):.10f}", flush=True)
    print(f" Mean Final Gate Count: {np.mean(final_gate_counts):.2f}", flush=True)
    print(f" Median Final Gate Count: {np.median(final_gate_counts):.1f}", flush=True)
    print(f" Mean Circuit Depth: {np.mean(depths):.2f}", flush=True)
    print(f" Median Circuit Depth: {np.median(depths):.1f}", flush=True)
    print(f" Mean Entangling Gate Count: {np.mean(entangling_counts):.2f}", flush=True)
    print(f" Mean Gate Reduction (Raw -> Final): {np.mean(gate_reductions):.2f} gates ({(1 - np.mean(final_gate_counts)/np.mean(raw_gate_counts))*100:.1f}%)", flush=True)
    print(f" Mean Synthesis Time: {np.mean(search_times):.2f} seconds", flush=True)
    print("==========================================================================================================", flush=True)

if __name__ == '__main__':
    main()
