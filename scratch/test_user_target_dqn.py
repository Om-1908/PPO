import os
import sys
import time
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add quantumrl and scratch to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('quantumrl'))
sys.path.insert(0, os.path.abspath('scratch'))

from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent
from scratch.validate_dqn_20_targets import synthesize_target_dqn
from synthesis import format_circuit_display_lines, build_qiskit_circuit, verify_4way_consistency

def main():
    target_sv = np.array([
        +0.38283143 + 0.64381903j,
        -0.14570048 - 0.16135750j,
        -0.30778429 - 0.21702287j,
        +0.30709539 + 0.39437877j
    ], dtype=np.complex128)

    norm = np.linalg.norm(target_sv)
    target_sv = target_sv / norm

    print("==================================================================================", flush=True)
    print("TATVA 2-QUBIT DQN SYNTHESIS TEST FOR USER TARGET STATEVECTOR", flush=True)
    print("==================================================================================", flush=True)
    print(f"Target Statevector:\n  |00>: {target_sv[0]:.8f}\n  |01>: {target_sv[1]:.8f}\n  |10>: {target_sv[2]:.8f}\n  |11>: {target_sv[3]:.8f}\n", flush=True)

    config = Config()
    env = QuantumCircuitEnv(config)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    
    ckpt_path = os.path.join('saved_models', '2qubit', 'dqn_model_best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('saved_models', '2qubit', 'dqn_model.pth')
        
    print(f"[DQNAgent] Loading checkpoint: {ckpt_path}", flush=True)
    agent.load(ckpt_path)
    agent.q_net.eval()

    best_cand = synthesize_target_dqn(
        target_sv=target_sv,
        agent=agent,
        env=env,
        config=config,
        target_fidelity=0.999999,
        max_candidates=30,
        verbose=False
    )

    actions = best_cand['simplified_actions']
    fresh_qc = build_qiskit_circuit(actions, n_qubits=2)
    output_sv = Statevector.from_instruction(fresh_qc).data
    fresh_fid = float(np.abs(np.vdot(target_sv, output_sv)) ** 2)

    opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
        actions,
        target_sv=target_sv,
        n_qubits=2,
        internal_fid=best_cand['internal_fidelity']
    )

    print("\n----------------------------------------------------------------------------------", flush=True)
    print("SYNTHESIS RESULTS & CIRCUIT SPECIFICATION:", flush=True)
    print("----------------------------------------------------------------------------------", flush=True)
    print(f" Status:                       {'HIGH_PRECISION_SUCCESS' if fresh_fid >= 0.999999 else 'APPROXIMATE_SUCCESS'}", flush=True)
    print(f" Fresh Verified Qiskit Fid:     {fresh_fid:.12f}", flush=True)
    print(f" Raw DQN Gate Count:           {best_cand['raw_gate_count']}", flush=True)
    print(f" Final Optimized Gate Count:   {best_cand['final_gate_count']}", flush=True)
    print(f" Circuit Depth:                {best_cand['circuit_depth']}", flush=True)
    print(f" Entangling Gates (CNOT):      {best_cand['entangling_gate_count']}", flush=True)
    print(f" 4-Way Verification Max Dev:  {max_dev:.1e}", flush=True)
    print(f" Total Search Time:            {best_cand['search_time_seconds']:.2f} seconds", flush=True)
    print("----------------------------------------------------------------------------------", flush=True)
    print("OPTIMIZED QUANTUM CIRCUIT GATES:", flush=True)
    for line in format_circuit_display_lines(actions):
        print(line.replace('\u03b8', 'theta'), flush=True)
    print("==================================================================================", flush=True)

if __name__ == '__main__':
    main()
