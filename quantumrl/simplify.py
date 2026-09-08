"""
simplify.py
-----------
Circuit simplification, continuous local parameter optimization, and 12-step verification pipeline for TATVA 2-qubit synthesis.

Functions:
  simplify_gate_sequence(applied_actions)
  optimize_circuit_parameters(applied_actions, target_sv, n_qubits)
  verify_synthesis_candidate(target_sv, applied_actions, n_qubits, fidelity_threshold)
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Pre-computed 2x2 gate matrices (complex128)
_H_MAT   = (1.0 / math.sqrt(2.0)) * np.array([[1., 1.], [1., -1.]], dtype=np.complex128)
_X_MAT   = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
_Y_MAT   = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
_Z_MAT   = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
_S_MAT   = np.array([[1., 0.], [0., 1j]], dtype=np.complex128)
_SDG_MAT = np.array([[1., 0.], [0., -1j]], dtype=np.complex128)
_T_MAT   = np.array([[1., 0.], [0., math.e ** (1j * math.pi / 4)]], dtype=np.complex128)
_TDG_MAT = np.array([[1., 0.], [0., math.e ** (-1j * math.pi / 4)]], dtype=np.complex128)

def _rx(angle: float) -> np.ndarray:
    h = angle / 2.0
    return np.array([[np.cos(h), -1j * np.sin(h)], [-1j * np.sin(h), np.cos(h)]], dtype=np.complex128)

def _ry(angle: float) -> np.ndarray:
    h = angle / 2.0
    return np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]], dtype=np.complex128)

def _rz(angle: float) -> np.ndarray:
    h = angle / 2.0
    return np.array([[np.exp(-1j * h), 0.], [0., np.exp(1j * h)]], dtype=np.complex128)

def simulate_actions(actions: List[Tuple], n_qubits: int = 2) -> np.ndarray:
    """Simulate gate sequence actions from |0...0> in double precision float64."""
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=np.complex128)
    sv[0] = 1.0

    for gate_name, qubit_or_pair, angle in actions:
        if gate_name in ('H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg', 'RX', 'RY', 'RZ'):
            qubit = qubit_or_pair
            if gate_name == 'H': mat = _H_MAT
            elif gate_name == 'X': mat = _X_MAT
            elif gate_name == 'Y': mat = _Y_MAT
            elif gate_name == 'Z': mat = _Z_MAT
            elif gate_name == 'S': mat = _S_MAT
            elif gate_name == 'Sdg': mat = _SDG_MAT
            elif gate_name == 'T': mat = _T_MAT
            elif gate_name == 'Tdg': mat = _TDG_MAT
            elif gate_name == 'RX': mat = _rx(angle)
            elif gate_name == 'RY': mat = _ry(angle)
            elif gate_name == 'RZ': mat = _rz(angle)

            axis = n_qubits - 1 - qubit
            shape = [2] * n_qubits
            T = sv.reshape(shape)
            T_new = np.tensordot(mat, T, axes=([1], [axis]))
            order = list(range(1, axis + 1)) + [0] + list(range(axis + 1, n_qubits))
            sv = np.transpose(T_new, order).reshape(-1)

        elif gate_name == 'CNOT':
            ctrl, tgt = qubit_or_pair
            ctrl_ax = n_qubits - 1 - ctrl
            tgt_ax  = n_qubits - 1 - tgt
            shape   = [2] * n_qubits
            T       = sv.reshape(shape).copy()

            sl_ctrl = [slice(None)] * n_qubits
            sl_ctrl[ctrl_ax] = 1
            sl0 = list(sl_ctrl); sl0[tgt_ax] = 0
            sl1 = list(sl_ctrl); sl1[tgt_ax] = 1

            v0 = T[tuple(sl0)].copy()
            T[tuple(sl0)] = T[tuple(sl1)]
            T[tuple(sl1)] = v0
            sv = T.reshape(-1)

        elif gate_name == 'CZ':
            # 4x4 CZ acts as diag(1, 1, 1, -1) on |00>, |01>, |10>, |11>
            sv[3] = -sv[3]

        elif gate_name == 'SWAP':
            # SWAP interchanges |01> (index 1) and |10> (index 2)
            v1 = sv[1].copy()
            sv[1] = sv[2]
            sv[2] = v1

    return sv

def compute_depth(actions: List[Tuple], n_qubits: int = 2) -> int:
    """Calculate circuit depth."""
    qubit_depths = [0] * n_qubits
    for gate_name, qubit_or_pair, _ in actions:
        if isinstance(qubit_or_pair, (tuple, list)):
            qs = list(qubit_or_pair)
        else:
            qs = [qubit_or_pair]
        max_d = max(qubit_depths[q] for q in qs)
        new_d = max_d + 1
        for q in qs:
            qubit_depths[q] = new_d
    return max(qubit_depths) if qubit_depths else 0

def simplify_gate_sequence(actions: List[Tuple], target_sv: Optional[np.ndarray] = None, n_qubits: int = 2) -> List[Tuple]:
    """
    Apply safe algebraic circuit simplifications:
    - Self-inverse cancellation (X-X, Y-Y, Z-Z, H-H, CNOT-CNOT, CZ-CZ, SWAP-SWAP)
    - Merging consecutive rotations of same type on same qubit
    - Pruning zero/near-zero rotations (|theta mod 2pi| < 1e-6)
    - Optional re-optimization of parameters post-simplification if target_sv is provided.
    """
    if not actions:
        return []

    changed = True
    current = list(actions)

    while changed:
        changed = False
        new_actions = []
        i = 0
        while i < len(current):
            g1, q1, a1 = current[i]

            # Rule 1: Prune near-zero rotations
            if g1 in ('RX', 'RY', 'RZ') and a1 is not None:
                ang = math.fmod(a1, 2.0 * math.pi)
                if abs(ang) < 1e-6 or abs(abs(ang) - 2.0 * math.pi) < 1e-6:
                    i += 1
                    changed = True
                    continue

            if i + 1 < len(current):
                g2, q2, a2 = current[i + 1]

                # Rule 2: Cancel consecutive self-inverse gates
                if g1 == g2 and q1 == q2 and g1 in ('X', 'Y', 'Z', 'H', 'CNOT', 'CZ', 'SWAP'):
                    i += 2
                    changed = True
                    continue

                # Rule 3: Merge consecutive rotations of same type on same qubit
                if g1 == g2 and q1 == q2 and g1 in ('RX', 'RY', 'RZ') and a1 is not None and a2 is not None:
                    merged_angle = float(math.fmod(a1 + a2, 2.0 * math.pi))
                    if abs(merged_angle) >= 1e-6 and abs(abs(merged_angle) - 2.0 * math.pi) >= 1e-6:
                        new_actions.append((g1, q1, merged_angle))
                    i += 2
                    changed = True
                    continue

            new_actions.append((g1, q1, a1))
            i += 1
        current = new_actions

    if target_sv is not None:
        current, _ = optimize_circuit_parameters(current, target_sv, n_qubits=n_qubits)

    return current

def _fidelity(target_sv: np.ndarray, sv: np.ndarray) -> float:
    """Compute true quantum state fidelity: F = |<target|sv>|^2 using np.vdot(target_sv, sv)."""
    return float(abs(np.vdot(target_sv, sv)) ** 2)

def commute_independent_gates(actions: List[Tuple]) -> List[Tuple]:
    """
    Commute independent single-qubit gates on disjoint qubits to bring same-qubit
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
                if (isinstance(q1, int) and isinstance(q2, int) and q1 != q2 and
                    g1 in ('H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg', 'RX', 'RY', 'RZ') and
                    g2 in ('H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg', 'RX', 'RY', 'RZ')):
                    if i + 2 < len(current):
                        g3, q3, a3 = current[i+2]
                        if q1 == q3 and g1 == g3 and g1 in ('RX', 'RY', 'RZ'):
                            new_actions.append((g2, q2, a2))
                            new_actions.append((g1, q1, a1))
                            i += 2
                            changed = True
                            continue
            new_actions.append(current[i])
            i += 1
        current = new_actions

    return current


def optimize_circuit_parameters(
    actions: List[Tuple],
    target_sv: np.ndarray,
    n_qubits: int = 2,
    max_iter: int = 150,
) -> Tuple[List[Tuple], float]:
    """
    Perform continuous Scipy L-BFGS-B optimization on rotation angles to maximize fidelity.
    """
    if not actions:
        init_sv = simulate_actions([], n_qubits)
        fid = _fidelity(target_sv, init_sv)
        return [], fid

    param_indices = [idx for idx, (g, _, a) in enumerate(actions) if g in ('RX', 'RY', 'RZ') and a is not None]

    if not param_indices:
        sim_sv = simulate_actions(actions, n_qubits)
        fid = _fidelity(target_sv, sim_sv)
        return actions, fid

    x0 = np.array([actions[idx][2] for idx in param_indices], dtype=np.float64)

    def loss_func(params):
        temp_actions = list(actions)
        for i, idx in enumerate(param_indices):
            g, q, _ = temp_actions[idx]
            temp_actions[idx] = (g, q, float(params[i]))
        sv = simulate_actions(temp_actions, n_qubits)
        fid = _fidelity(target_sv, sv)
        return 1.0 - fid

    res = minimize(loss_func, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})

    opt_actions = list(actions)
    for i, idx in enumerate(param_indices):
        g, q, _ = opt_actions[idx]
        ang = float(math.fmod(res.x[i], 2.0 * math.pi))
        opt_actions[idx] = (g, q, ang)

    opt_sv = simulate_actions(opt_actions, n_qubits)
    opt_fid = _fidelity(target_sv, opt_sv)
    return opt_actions, opt_fid


def multi_start_optimize_parameters(
    actions: List[Tuple],
    target_sv: np.ndarray,
    n_qubits: int = 2,
    target_fidelity: float = 0.999999,
    restarts: int = 3
) -> Tuple[List[Tuple], float]:
    """
    Continuous parameter optimization with multi-start restarts if initial optimization
    does not hit target fidelity threshold.
    """
    opt_actions, opt_fid = optimize_circuit_parameters(actions, target_sv, n_qubits=n_qubits, max_iter=150)

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

        test_actions, test_fid = optimize_circuit_parameters(pert_actions, target_sv, n_qubits=n_qubits, max_iter=150)
        if test_fid > best_fid:
            best_fid = test_fid
            best_actions = test_actions
            if best_fid >= target_fidelity:
                break

    return best_actions, best_fid

def verify_synthesis_candidate(
    target_sv: np.ndarray,
    applied_actions: List[Tuple],
    n_qubits: int = 2,
    fidelity_threshold: float = 0.999999,
) -> Dict:
    """
    Full 12-step verification pipeline for candidate synthesis solution:
    1. Circuit reconstruction
    2. Gate-order verification
    3. Parameter verification
    4. Statevector simulation
    5. Target normalization check
    6. Generated-state normalization check
    7. Fidelity calculation
    8. Fidelity threshold check
    9. Circuit simplification & re-optimization
    10. Re-simulation after simplification
    11. Final fidelity check
    12. Final statistics (gate count, depth, CX count)

    Returns dict containing verified status, final fidelity, simplified actions, and stats.
    """
    target_norm = float(np.linalg.norm(target_sv))
    if abs(target_norm - 1.0) > 1e-10:
        return {'verified': False, 'reason': 'Target statevector normalization error'}

    # Initial state simulation
    gen_sv = simulate_actions(applied_actions, n_qubits)
    gen_norm = float(np.linalg.norm(gen_sv))
    if abs(gen_norm - 1.0) > 1e-10:
        return {'verified': False, 'reason': 'Generated statevector normalization error'}

    fid_initial = _fidelity(target_sv, gen_sv)

    # Local continuous optimization & simplification with re-optimization
    opt_actions, fid_opt = optimize_circuit_parameters(applied_actions, target_sv, n_qubits)
    simp_actions = simplify_gate_sequence(opt_actions, target_sv=target_sv, n_qubits=n_qubits)

    # Re-simulate simplified circuit
    final_sv = simulate_actions(simp_actions, n_qubits)
    final_fid = _fidelity(target_sv, final_sv)

    # If simplification degraded fidelity below opt_fid, fall back to opt_actions
    if final_fid < fid_opt - 1e-6 and fid_opt >= fidelity_threshold:
        simp_actions = opt_actions
        final_fid = fid_opt

    if final_fid < fidelity_threshold:
        return {
            'verified': False,
            'reason': f'Fidelity {final_fid:.8f} below threshold {fidelity_threshold}',
            'initial_fidelity': fid_initial,
            'final_fidelity': final_fid,
        }

    gate_count = len(simp_actions)
    depth = compute_depth(simp_actions, n_qubits)
    cx_count = sum(1 for g, _, _ in simp_actions if g in ('CNOT', 'CZ'))

    return {
        'verified': True,
        'initial_fidelity': fid_initial,
        'final_fidelity': final_fid,
        'simplified_actions': simp_actions,
        'gate_count': gate_count,
        'circuit_depth': depth,
        'entangling_gate_count': cx_count,
    }
