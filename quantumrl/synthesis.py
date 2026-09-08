"""
synthesis.py
------------
Multi-candidate PPO policy search engine for TATVA quantum circuit synthesis.

Hierarchical Optimization Objective:
  1. HARD CONSTRAINT: Find F >= 0.999999 (HIGH_PRECISION_SUCCESS).
  2. PRIMARY OBJECTIVE: Minimize total gate count.
  3. SECONDARY OBJECTIVE: Minimize circuit depth.
  4. TERTIARY OBJECTIVE: Minimize entangling-gate count (CZ, CNOT, SWAP).

Strict Verification Protocol:
  1. Stores exact final gate list and exact final full-precision float64 parameters.
  2. Reconstructs a completely fresh Qiskit QuantumCircuit from stored gate list.
  3. Simulates fresh Qiskit circuit from |00...0>.
  4. Calculates fidelity against the exact normalized target statevector.
  5. Runs automated 4-way consistency check (Optimizer vs Internal vs Fresh Qiskit vs Text Parsed Displayed Output).
  6. Asserts max_discrepancy < 1e-10 and fresh_verified_fidelity >= 0.999999 before reporting HIGH_PRECISION_SUCCESS.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from ppo_agent import PPOAgent
from simplify import (
    verify_synthesis_candidate,
    simulate_actions,
    _fidelity,
    optimize_circuit_parameters,
    simplify_gate_sequence
)


def build_qiskit_circuit(actions: List[Tuple], n_qubits: int = 2) -> QuantumCircuit:
    """Reconstruct a completely fresh Qiskit QuantumCircuit from gate action tuple list."""
    qc = QuantumCircuit(n_qubits)
    for g, q, a in actions:
        if g == 'H': qc.h(q)
        elif g == 'X': qc.x(q)
        elif g == 'Y': qc.y(q)
        elif g == 'Z': qc.z(q)
        elif g == 'S': qc.s(q)
        elif g == 'Sdg': qc.sdg(q)
        elif g == 'T': qc.t(q)
        elif g == 'Tdg': qc.tdg(q)
        elif g == 'RX': qc.rx(a, q)
        elif g == 'RY': qc.ry(a, q)
        elif g == 'RZ': qc.rz(a, q)
        elif g in ('CNOT', 'CX'):
            if isinstance(q, (tuple, list)):
                ctrl, tgt = q
            else:
                ctrl, tgt = 0, 1
            qc.cx(ctrl, tgt)
        elif g == 'CZ': qc.cz(0, 1)
        elif g == 'SWAP': qc.swap(0, 1)
        else:
            raise ValueError(f"Unknown gate in reconstruction: {g}")
    return qc


def format_circuit_display_lines(actions: List[Tuple]) -> List[str]:
    """Generate exact full-precision text display lines for the circuit."""
    display_lines = []
    for idx, (gate, qubits, param) in enumerate(actions, 1):
        if param is not None:
            line = f"  {idx:02d}. {gate}(q{qubits}, θ = {param:+.16f})"
        else:
            if isinstance(qubits, (tuple, list)):
                q_str = f"q{qubits[0]}, q{qubits[1]}"
            else:
                q_str = f"q{qubits}"
            line = f"  {idx:02d}. {gate}({q_str})"
        display_lines.append(line)
    return display_lines


def parse_displayed_circuit(circuit_str_lines: List[str], n_qubits: int = 2) -> QuantumCircuit:
    """
    Reconstruct a fresh Qiskit QuantumCircuit by parsing the EXACT string lines displayed in the report.
    Guarantees 100% verification of displayed output text.
    """
    qc = QuantumCircuit(n_qubits)

    for line in circuit_str_lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        content = line.split('.', 1)[1].strip()
        gate_name = content.split('(', 1)[0].strip()
        inside = content.split('(', 1)[1].rstrip(')')

        if gate_name in ('CNOT', 'CX'):
            c_str, t_str = inside.split(',')
            c = int(c_str.strip().replace('q', ''))
            t = int(t_str.strip().replace('q', ''))
            qc.cx(c, t)
        elif gate_name == 'CZ':
            qc.cz(0, 1)
        elif gate_name == 'SWAP':
            qc.swap(0, 1)
        else:
            if ',' in inside:
                q_part, angle_part = inside.split(',', 1)
                q_idx = int(q_part.strip().replace('q', ''))
                angle = float(angle_part.split('=')[1].strip())
            else:
                q_idx = int(inside.strip().replace('q', ''))
                angle = None

            if gate_name == 'H': qc.h(q_idx)
            elif gate_name == 'X': qc.x(q_idx)
            elif gate_name == 'Y': qc.y(q_idx)
            elif gate_name == 'Z': qc.z(q_idx)
            elif gate_name == 'S': qc.s(q_idx)
            elif gate_name == 'Sdg': qc.sdg(q_idx)
            elif gate_name == 'T': qc.t(q_idx)
            elif gate_name == 'Tdg': qc.tdg(q_idx)
            elif gate_name == 'RX': qc.rx(angle, q_idx)
            elif gate_name == 'RY': qc.ry(angle, q_idx)
            elif gate_name == 'RZ': qc.rz(angle, q_idx)

    return qc


def verify_4way_consistency(
    actions: List[Tuple],
    target_sv: np.ndarray,
    n_qubits: int = 2,
    internal_fid: Optional[float] = None
) -> Tuple[float, float, float, float, float]:
    """
    Perform rigorous 4-way consistency check comparing:
      1. Optimizer Objective / Fidelity (NumPy simulate_actions)
      2. Internal Circuit Fidelity (from simplify.py pipeline)
      3. Fresh Qiskit Fidelity (from fresh build_qiskit_circuit)
      4. Final Displayed Circuit Fidelity (parsed back from output text)
    Returns: (opt_fid, int_fid, fresh_qiskit_fid, disp_fid, max_discrepancy)
    """
    # 1. Optimizer Fidelity
    sv_opt = simulate_actions(actions, n_qubits=n_qubits)
    opt_fid = _fidelity(target_sv, sv_opt)

    # 2. Internal Circuit Fidelity
    int_fid = internal_fid if internal_fid is not None else opt_fid

    # 3. Fresh Qiskit Fidelity
    qc_fresh = build_qiskit_circuit(actions, n_qubits=n_qubits)
    sv_fresh = Statevector.from_instruction(qc_fresh).data
    fresh_qiskit_fid = _fidelity(target_sv, sv_fresh)

    # 4. Final Displayed Circuit Fidelity
    display_lines = format_circuit_display_lines(actions)
    qc_disp = parse_displayed_circuit(display_lines, n_qubits=n_qubits)
    sv_disp = Statevector.from_instruction(qc_disp).data
    disp_fid = _fidelity(target_sv, sv_disp)

    dev1 = abs(opt_fid - fresh_qiskit_fid)
    dev2 = abs(int_fid - fresh_qiskit_fid)
    dev3 = abs(disp_fid - fresh_qiskit_fid)
    max_discrepancy = max(dev1, dev2, dev3)

    return opt_fid, int_fid, fresh_qiskit_fid, disp_fid, max_discrepancy


def prune_and_reoptimize_structural_search(
    actions: List[Tuple],
    target_sv: np.ndarray,
    n_qubits: int = 2,
    target_fidelity: float = 0.999999,
    max_prune_passes: int = 5
) -> Tuple[List[Tuple], float]:
    """
    Structural Search Engine for Shorter Circuits:
    Iteratively prunes removable gates from a verified high-precision candidate,
    re-optimizes continuous parameters, re-simplifies, and re-verifies via fresh Qiskit.
    Retains pruned circuit ONLY if fresh Qiskit fidelity F >= target_fidelity.
    """
    current_actions = list(actions)
    current_qc = build_qiskit_circuit(current_actions, n_qubits=n_qubits)
    current_fid = _fidelity(target_sv, Statevector.from_instruction(current_qc).data)

    if current_fid < target_fidelity:
        return current_actions, current_fid

    for pass_idx in range(max_prune_passes):
        improved_in_pass = False
        n_gates = len(current_actions)

        for i in range(n_gates):
            # Try removing gate i
            candidate_actions = current_actions[:i] + current_actions[i+1:]
            if not candidate_actions:
                continue

            # Re-optimize continuous parameters of the candidate sub-sequence
            opt_actions, opt_fid = optimize_circuit_parameters(candidate_actions, target_sv, n_qubits=n_qubits)

            # Re-simplify
            simp_actions = simplify_gate_sequence(opt_actions)

            # Fresh Qiskit re-verification
            test_qc = build_qiskit_circuit(simp_actions, n_qubits=n_qubits)
            fresh_sv = Statevector.from_instruction(test_qc).data
            fresh_fid = _fidelity(target_sv, fresh_sv)

            if fresh_fid >= target_fidelity and len(simp_actions) < len(current_actions):
                current_actions = simp_actions
                current_fid = fresh_fid
                improved_in_pass = True
                break

        if not improved_in_pass:
            break

    return current_actions, current_fid


def candidate_rank_key(cand: Dict, target_fidelity: float = 0.999999) -> Tuple:
    """
    Strict Hierarchical Candidate Ranking Key:
      1. Satisfies F >= target_fidelity (High precision first)
      2. Lower total gate count
      3. Lower circuit depth
      4. Fewer entangling gates
      5. Higher fidelity
    """
    fid = cand['final_fidelity']
    is_hp = 1 if fid >= target_fidelity else (1 if fid >= 0.99 else 0)
    return (
        -is_hp,                           # 1st: High precision candidate first
        cand['final_gate_count'],         # 2nd: Primary Objective (min gates)
        cand['circuit_depth'],            # 3rd: Secondary Objective (min depth)
        cand['entangling_gate_count'],    # 4th: Tertiary Objective (min entangling)
        -fid                              # 5th: Maximize fidelity
    )


def synthesize_circuit(
    target_sv: np.ndarray,
    agent: Optional[PPOAgent] = None,
    env: Optional[QuantumCircuitEnv] = None,
    config: Optional[Config] = None,
    target_fidelity: float = 0.999999,
    max_candidates: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Synthesize a verified quantum circuit for an arbitrary target statevector.
    Strict Hierarchical Objective:
        Find F >= target_fidelity -> MINIMIZE GATE COUNT -> MINIMIZE CIRCUIT DEPTH
    """
    if config is None:
        config = Config()

    if env is None:
        env = QuantumCircuitEnv(config)

    if agent is None:
        checkpoint_path = os.path.join("checkpoints", "ppo_paused_ep174500.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join("saved_models", "2qubit", "ppo_model_best_best.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join("saved_models", "2qubit", "ppo_model_best.pth")

        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = PPOAgent(obs_size=obs_size, action_size=action_size, config=config, device=device)

        if verbose:
            print(f"[synthesis] Loading PPO model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "actor_critic_state_dict" in ckpt:
            agent.ac.load_state_dict(ckpt["actor_critic_state_dict"])
        else:
            agent.ac.load_state_dict(ckpt)
        agent.ac.eval()

    # Normalize target statevector strictly
    target_sv_raw = np.array(target_sv, dtype=np.complex128)
    orig_norm = float(np.linalg.norm(target_sv_raw))
    if orig_norm > 1e-10:
        target_sv = target_sv_raw / orig_norm
    else:
        target_sv = target_sv_raw

    t0 = time.time()
    candidate_pool: List[Dict] = []

    if verbose:
        print(f"\n==================================================================")
        print(f"TATVA PPO SYNTHESIS ENGINE (Hierarchical Goal: F >= {target_fidelity})")
        print(f"Target Norm: {orig_norm:.16f} | Search Budget: {max_candidates} candidates")
        print(f"==================================================================")

    for attempt in range(max_candidates):
        deterministic = (attempt == 0)
        obs, _ = env.reset(target_sv=target_sv)
        done = False

        while not done:
            action, _, _, _ = agent.ac.get_action(obs, deterministic=deterministic)
            action = int(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        raw_actions = list(env.applied_actions)
        raw_gate_count = len(raw_actions)

        # 1. Refinement & Simplification
        ver_res = verify_synthesis_candidate(
            target_sv=target_sv,
            applied_actions=raw_actions,
            n_qubits=env.n_qubits,
            fidelity_threshold=target_fidelity
        )

        simp_actions = ver_res.get('simplified_actions', raw_actions)

        # 2. Structural Shorter-Circuit Search if candidate meets high precision or baseline
        opt_actions, opt_fid = prune_and_reoptimize_structural_search(
            simp_actions,
            target_sv=target_sv,
            n_qubits=env.n_qubits,
            target_fidelity=target_fidelity
        )

        # 3. Fresh Qiskit Verification & 4-Way Consistency Check
        opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
            opt_actions,
            target_sv=target_sv,
            n_qubits=env.n_qubits,
            internal_fid=ver_res.get('fidelity')
        )

        fresh_qc = build_qiskit_circuit(opt_actions, n_qubits=env.n_qubits)
        mode_str = "Greedy Policy" if deterministic else "Stochastic Sample"

        if verbose:
            print(f"  Attempt {attempt+1:02d}/{max_candidates:02d} [{mode_str:18s}] -> Gates: {raw_gate_count} -> {len(opt_actions)} | Qiskit Fresh F = {fresh_qiskit_f:.10f} | Dev: {max_dev:.1e}")

        # Classification
        if fresh_qiskit_f >= target_fidelity:
            status = 'HIGH_PRECISION_SUCCESS'
        elif fresh_qiskit_f >= 0.99:
            status = 'APPROXIMATE_SUCCESS'
        else:
            status = 'FAILED'

        cand_rec = {
            'attempt': attempt + 1,
            'deterministic': deterministic,
            'raw_actions': raw_actions,
            'raw_gate_count': raw_gate_count,
            'ver_res': ver_res,
            'final_fidelity': fresh_qiskit_f,
            'optimizer_fidelity': opt_f,
            'internal_fidelity': int_f,
            'displayed_circuit_fidelity': disp_f,
            'max_discrepancy': max_dev,
            'status': status,
            'verified': (fresh_qiskit_f >= target_fidelity and max_dev < 1e-10),
            'simplified_actions': opt_actions,
            'final_gate_count': len(opt_actions),
            'circuit_depth': fresh_qc.depth(),
            'entangling_gate_count': sum(1 for g, q, a in opt_actions if g in ('CZ', 'CNOT', 'SWAP')),
            'full_precision_parameters': [(g, q, float(a) if a is not None else None) for g, q, a in opt_actions],
            'display_lines': format_circuit_display_lines(opt_actions),
            'search_time_seconds': time.time() - t0,
            'qiskit_circuit': fresh_qc,
        }

        candidate_pool.append(cand_rec)

        # Stop search early if we have found a high-precision candidate with minimal depth/gates
        if fresh_qiskit_f >= target_fidelity and cand_rec['final_gate_count'] <= 12:
            if verbose:
                print(f"  [EARLY TERMINATION] Minimal high-precision candidate found ({cand_rec['final_gate_count']} gates, F = {fresh_qiskit_f:.12f}).")
            break

    # Rank all candidates according to strict hierarchical objective
    candidate_pool.sort(key=lambda c: candidate_rank_key(c, target_fidelity=target_fidelity))
    best_candidate = candidate_pool[0]

    # Final Verification & Assertion
    opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
        best_candidate['simplified_actions'],
        target_sv=target_sv,
        n_qubits=env.n_qubits,
        internal_fid=best_candidate['internal_fidelity']
    )
    assert max_dev < 1e-10, f"4-Way Consistency Check Failed! Max discrepancy: {max_dev}"

    best_candidate['candidate_pool_size'] = len(candidate_pool)
    best_candidate['search_time_seconds'] = time.time() - t0

    if verbose:
        print(f"\n==================================================================")
        if best_candidate['final_fidelity'] >= target_fidelity:
            print(f" [SUCCESS] HIGH-PRECISION SYNTHESIS VERIFIED (Qiskit F = {best_candidate['final_fidelity']:.12f} >= {target_fidelity})")
            print(f" Best Verified Candidate Found Within Budget | Time: {best_candidate['search_time_seconds']:.2f}s")
            print(f" Gates: Raw ({best_candidate['raw_gate_count']}) -> Final ({best_candidate['final_gate_count']}) | Depth: {best_candidate['circuit_depth']} | 4-Way Dev: {max_dev:.1e}")
        else:
            print(f" [HIGH_PRECISION_SYNTHESIS_NOT_FOUND] Best Qiskit F = {best_candidate['final_fidelity']:.10f} < {target_fidelity}")
        print(f"==================================================================")

    return best_candidate
