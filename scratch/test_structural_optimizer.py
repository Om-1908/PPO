"""
test_structural_optimizer.py
-----------------------------
Testing upgraded multi-strategy structural search engine for TATVA PPO.
Goal: Reduce 17-gate candidate to shorter circuit while preserving F >= 0.999999.
"""

import math
import time
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath("quantumrl"))

from configs.config_2qubit import Config as Config2Q
from quantum_env import QuantumCircuitEnv
from ppo_agent import PPOAgent
from simplify import (
    simulate_actions,
    _fidelity,
    optimize_circuit_parameters,
    simplify_gate_sequence,
)
from synthesis import (
    build_qiskit_circuit,
    format_circuit_display_lines,
    verify_4way_consistency,
    synthesize_circuit,
)
from qiskit.quantum_info import Statevector


def multi_start_optimize_parameters(
    actions: list,
    target_sv: np.ndarray,
    n_qubits: int = 2,
    target_fidelity: float = 0.999999,
    restarts: int = 4
) -> tuple:
    """
    Continuous parameter optimization with multi-start restarts if initial optimization
    does not hit target fidelity threshold.
    """
    opt_actions, opt_fid = optimize_circuit_parameters(actions, target_sv, n_qubits=n_qubits, max_iter=200)

    if opt_fid >= target_fidelity or restarts <= 0:
        return opt_actions, opt_fid

    param_indices = [idx for idx, (g, _, a) in enumerate(actions) if g in ('RX', 'RY', 'RZ') and a is not None]
    if not param_indices:
        return opt_actions, opt_fid

    best_actions = opt_actions
    best_fid = opt_fid

    base_angles = [opt_actions[idx][2] for idx in param_indices]

    for trial in range(restarts):
        pert_actions = list(actions)
        for i, idx in enumerate(param_indices):
            g, q, _ = pert_actions[idx]
            noise = np.random.uniform(-0.15, 0.15)
            pert_actions[idx] = (g, q, base_angles[i] + noise)

        test_actions, test_fid = optimize_circuit_parameters(pert_actions, target_sv, n_qubits=n_qubits, max_iter=200)
        if test_fid > best_fid:
            best_fid = test_fid
            best_actions = test_actions
            if best_fid >= target_fidelity:
                break

    return best_actions, best_fid


def commute_independent_gates(actions: list) -> list:
    """
    Commute independent single-qubit gates on different qubits to bring same-qubit
    rotations adjacent for algebraic merging.
    """
    if len(actions) <= 1:
        return list(actions)

    current = list(actions)
    changed = True
    passes = 0

    while changed and passes < 5:
        changed = False
        passes += 1
        i = 0
        new_actions = []
        while i < len(current):
            if i + 1 < len(current):
                g1, q1, a1 = current[i]
                g2, q2, a2 = current[i+1]
                # If single qubit gates on disjoint qubits, commute them if it helps bring same qubit gates together
                if (isinstance(q1, int) and isinstance(q2, int) and q1 != q2 and
                    g1 in ('H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg', 'RX', 'RY', 'RZ') and
                    g2 in ('H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg', 'RX', 'RY', 'RZ')):
                    # Check if swapping i and i+1 allows current[i] to merge with current[i+2]
                    if i + 2 < len(current):
                        g3, q3, a3 = current[i+2]
                        if q1 == q3 and g1 == g3 and g1 in ('RX', 'RY', 'RZ'):
                            # Commute g1 and g2
                            new_actions.append((g2, q2, a2))
                            new_actions.append((g1, q1, a1))
                            i += 2
                            changed = True
                            continue
            new_actions.append(current[i])
            i += 1
        current = new_actions

    return current


def advanced_structural_optimization(
    actions: list,
    target_sv: np.ndarray,
    n_qubits: int = 2,
    target_fidelity: float = 0.999999,
    max_budget: int = 150
) -> tuple:
    """
    Multi-Strategy Structural Optimization Engine:
    Progressively reduces total gate count while guaranteeing fresh Qiskit F >= target_fidelity.
    """
    curr_actions = simplify_gate_sequence(actions)
    curr_actions, curr_fid = multi_start_optimize_parameters(curr_actions, target_sv, n_qubits=n_qubits, target_fidelity=target_fidelity)
    
    curr_qc = build_qiskit_circuit(curr_actions, n_qubits=n_qubits)
    curr_fid = _fidelity(target_sv, Statevector.from_instruction(curr_qc).data)

    if curr_fid < target_fidelity:
        return curr_actions, curr_fid

    best_actions = list(curr_actions)
    best_fid = curr_fid

    attempts = 0
    improved = True

    while improved and attempts < max_budget:
        improved = False

        # Strategy 1: Algebraic Commutation & Merging
        comm_actions = commute_independent_gates(best_actions)
        simp_comm = simplify_gate_sequence(comm_actions)
        if len(simp_comm) < len(best_actions):
            opt_comm, fid_comm = multi_start_optimize_parameters(simp_comm, target_sv, n_qubits=n_qubits, target_fidelity=target_fidelity)
            qc_comm = build_qiskit_circuit(opt_comm, n_qubits=n_qubits)
            fresh_fid_comm = _fidelity(target_sv, Statevector.from_instruction(qc_comm).data)

            if fresh_fid_comm >= target_fidelity and len(opt_comm) < len(best_actions):
                best_actions = opt_comm
                best_fid = fresh_fid_comm
                improved = True
                continue

        # Strategy 2: Single Gate Deletion (Forward & Backward)
        n_gates = len(best_actions)
        deletion_indices = list(range(n_gates))
        
        # Sort deletion indices: try deleting smallest rotation angles first
        rot_indices = [idx for idx in deletion_indices if best_actions[idx][0] in ('RX', 'RY', 'RZ') and best_actions[idx][2] is not None]
        non_rot_indices = [idx for idx in deletion_indices if idx not in rot_indices]
        rot_indices.sort(key=lambda idx: abs(math.fmod(best_actions[idx][2], math.pi)))
        
        ordered_indices = rot_indices + non_rot_indices

        for idx in ordered_indices:
            attempts += 1
            if attempts >= max_budget:
                break

            cand_actions = best_actions[:idx] + best_actions[idx+1:]
            if not cand_actions:
                continue

            # Continuous parameter re-optimization
            opt_cand, opt_fid = multi_start_optimize_parameters(cand_actions, target_sv, n_qubits=n_qubits, target_fidelity=target_fidelity)
            simp_cand = simplify_gate_sequence(opt_cand)
            opt_cand, opt_fid = multi_start_optimize_parameters(simp_cand, target_sv, n_qubits=n_qubits, target_fidelity=target_fidelity)

            # Fresh Qiskit verification
            test_qc = build_qiskit_circuit(opt_cand, n_qubits=n_qubits)
            fresh_fid = _fidelity(target_sv, Statevector.from_instruction(test_qc).data)

            if fresh_fid >= target_fidelity and len(opt_cand) < len(best_actions):
                print(f"   [Structural Reduction] {len(best_actions)} gates -> {len(opt_cand)} gates | Fresh Qiskit F = {fresh_fid:.12f}")
                best_actions = opt_cand
                best_fid = fresh_fid
                improved = True
                break

        # Strategy 3: Consecutive Pair Deletion
        if not improved and len(best_actions) > 2:
            for idx in range(len(best_actions) - 1):
                attempts += 1
                if attempts >= max_budget:
                    break

                cand_actions = best_actions[:idx] + best_actions[idx+2:]
                if not cand_actions:
                    continue

                opt_cand, opt_fid = multi_start_optimize_parameters(cand_actions, target_sv, n_qubits=n_qubits, target_fidelity=target_fidelity)
                simp_cand = simplify_gate_sequence(opt_cand)
                opt_cand, opt_fid = multi_start_optimize_parameters(simp_cand, target_sv, n_qubits=n_qubits, target_fidelity=target_fidelity)

                test_qc = build_qiskit_circuit(opt_cand, n_qubits=n_qubits)
                fresh_fid = _fidelity(target_sv, Statevector.from_instruction(test_qc).data)

                if fresh_fid >= target_fidelity and len(opt_cand) < len(best_actions):
                    print(f"   [Pair Reduction] {len(best_actions)} gates -> {len(opt_cand)} gates | Fresh Qiskit F = {fresh_fid:.12f}")
                    best_actions = opt_cand
                    best_fid = fresh_fid
                    improved = True
                    break

    return best_actions, best_fid


def main():
    target_sv = np.array([
        -0.013701720632-0.112386178478j,
        +0.246599787811+0.752500198277j,
        -0.003465918744+0.566219293377j,
        +0.198358388740+0.012298569641j
    ], dtype=np.complex128)
    target_sv /= np.linalg.norm(target_sv)

    cfg = Config2Q()
    env = QuantumCircuitEnv(cfg)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    ckpt_path = 'checkpoints/ppo_paused_ep174500.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['actor_critic_state_dict'] if 'actor_critic_state_dict' in ckpt else ckpt

    agent = PPOAgent(obs_size, action_size, cfg, torch.device('cpu'))
    agent.ac.load_state_dict(sd)
    agent.ac.eval()

    print("Synthesizing baseline PPO candidates (budget=30)...")
    res = synthesize_circuit(target_sv, agent=agent, env=env, config=cfg, target_fidelity=0.999999, max_candidates=30, verbose=False)

    print(f"\nInitial Best Candidate from synthesis engine:")
    print(f"  Gate Count : {res['final_gate_count']}")
    print(f"  Fidelity   : {res['final_fidelity']:.12f}")

    print("\nRunning Advanced Multi-Strategy Structural Optimization...")
    opt_actions, opt_fid = advanced_structural_optimization(res['simplified_actions'], target_sv, n_qubits=2, target_fidelity=0.999999)

    print(f"\nOptimization Complete!")
    print(f"  Final Gate Count : {len(opt_actions)}")
    print(f"  Final Fidelity   : {opt_fid:.12f}")

    lines = format_circuit_display_lines(opt_actions)
    print("\nOptimized Gate Sequence:")
    for line in lines:
        print(line)

if __name__ == '__main__':
    main()
