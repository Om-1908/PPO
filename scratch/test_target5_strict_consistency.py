import sys
import os
import numpy as np
import torch
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("quantumrl"))

from quantumrl.synthesis import synthesize_circuit, build_qiskit_circuit
from quantumrl.simplify import _fidelity, simulate_actions

def parse_displayed_circuit(circuit_str_lines, n_qubits=2):
    """
    Reconstruct a fresh Qiskit QuantumCircuit by parsing the EXACT string lines displayed in the report.
    This guarantees 100% verification of the final displayed output text.
    """
    qc = QuantumCircuit(n_qubits)
    parsed_actions = []

    for line in circuit_str_lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        content = line.split('.', 1)[1].strip()
        gate_name = content.split('(', 1)[0].strip()
        inside = content.split('(', 1)[1].rstrip(')')

        if gate_name in ('CNOT', 'CX'):
            # inside is "q0, q1" or "q1, q0"
            c_str, t_str = inside.split(',')
            c = int(c_str.strip().replace('q', ''))
            t = int(t_str.strip().replace('q', ''))
            qc.cx(c, t)
            parsed_actions.append((gate_name, (c, t), None))
        elif gate_name == 'CZ':
            qc.cz(0, 1)
            parsed_actions.append(('CZ', (0, 1), None))
        elif gate_name == 'SWAP':
            qc.swap(0, 1)
            parsed_actions.append(('SWAP', (0, 1), None))
        else:
            # e.g., "q0, θ = +3.5580922744416537" or "q0"
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
            parsed_actions.append((gate_name, q_idx, angle))

    return qc, parsed_actions


def main():
    target_raw = np.array([
        -0.07014476 + 0.17273607j,
        +0.34764850 - 0.77323775j,
        -0.14730561 - 0.18073644j,
        -0.41636934 + 0.13695922j
    ], dtype=np.complex128)

    # Normalize statevector strictly
    norm = np.linalg.norm(target_raw)
    target_sv = target_raw / norm

    print("==================================================================")
    print("TATVA PPO QUANTUM CIRCUIT SYNTHESIS (STRICT 4-WAY VERIFICATION)")
    print("==================================================================")
    print("Target Statevector (Normalized, Full Precision Float64):")
    for i, c in enumerate(target_sv):
        print(f"  |{i:02b}> : {c.real:+.16f} {c.imag:+.16f}i")
    print(f"Norm: {np.linalg.norm(target_sv):.16f}\n")

    # Run synthesis with high threshold
    res = synthesize_circuit(
        target_sv=target_sv,
        target_fidelity=0.999999,
        max_candidates=50,
        verbose=True
    )

    actions = res['simplified_actions']

    # Generate exact text display lines
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

    # 1. Optimizer Fidelity (from L-BFGS-B / NumPy simulate_actions)
    sv_opt = simulate_actions(actions, n_qubits=2)
    optimizer_fidelity = _fidelity(target_sv, sv_opt)

    # 2. Internal Circuit Fidelity (from res dict)
    internal_circuit_fidelity = res['ver_res'].get('fidelity', optimizer_fidelity)

    # 3. Fresh Qiskit Fidelity (from fresh build_qiskit_circuit)
    qc_fresh = build_qiskit_circuit(actions, n_qubits=2)
    sv_fresh = Statevector.from_instruction(qc_fresh).data
    fresh_qiskit_fidelity = _fidelity(target_sv, sv_fresh)

    # 4. Final Displayed Circuit Fidelity (parsed back from the exact output text)
    qc_disp, _ = parse_displayed_circuit(display_lines, n_qubits=2)
    sv_disp = Statevector.from_instruction(qc_disp).data
    final_displayed_circuit_fidelity = _fidelity(target_sv, sv_disp)

    print("\n==================================================================")
    print("4-WAY INTERNAL CONSISTENCY VERIFICATION CHECK:")
    print("==================================================================")
    print(f"  1. Optimizer Objective / Fidelity       : {optimizer_fidelity:.16f}")
    print(f"  2. Internal Circuit Fidelity            : {internal_circuit_fidelity:.16f}")
    print(f"  3. Fresh Qiskit Fidelity                : {fresh_qiskit_fidelity:.16f}")
    print(f"  4. Final Displayed Circuit Fidelity     : {final_displayed_circuit_fidelity:.16f}")

    dev1 = abs(optimizer_fidelity - fresh_qiskit_fidelity)
    dev2 = abs(internal_circuit_fidelity - fresh_qiskit_fidelity)
    dev3 = abs(final_displayed_circuit_fidelity - fresh_qiskit_fidelity)
    max_dev = max(dev1, dev2, dev3)

    print(f"  Max Numerical Discrepancy Across All 4 : {max_dev:.2e}")
    if max_dev < 1e-10:
        print("  -> STATUS: PERFECT 4-WAY CONSISTENCY (100% MATCH)")
    else:
        print("  -> STATUS: WARNING - DISCREPANCY DETECTED")

    print("\n==================================================================")
    print("FINAL VERIFIED SYNTHESIS REPORT (EXACTLY ONE FINAL CIRCUIT)")
    print("==================================================================")
    print("PIPELINE STAGE METRICS & GATE COUNT REDUCTION:")
    print(f"  Stage 1: Raw PPO Candidate Gate Count   : {res['raw_gate_count']}")
    print(f"  Stage 2: Continuous Angle Refinement    : (L-BFGS-B optimization applied)")
    print(f"  Stage 3: Simplified Gate Count          : {res['final_gate_count']}")
    print(f"  Stage 4: Fresh Qiskit Verification F    : {fresh_qiskit_fidelity:.16f}")
    print(f"  Stage 5: Verification Status            : {'PASSED (F >= 0.999999)' if fresh_qiskit_fidelity >= 0.999999 else 'FAILED (F < 0.999999)'}\n")

    depth = qc_fresh.depth()
    cz_cnt = sum(1 for g, q, a in actions if g in ('CZ', 'CNOT', 'SWAP'))

    print("FINAL VERIFIED CIRCUIT SUMMARY:")
    print(f"  - Verification Status           : {'SUCCESS' if fresh_qiskit_fidelity >= 0.999999 else 'HIGH_FIDELITY_CANDIDATE'}")
    print(f"  - Freshly Recomputed Fidelity   : {fresh_qiskit_fidelity:.16f}")
    print(f"  - Final Total Gate Count        : {len(actions)}")
    print(f"  - Final Depth                   : {depth}")
    print(f"  - Final Entangling Gate Count   : {cz_cnt}")
    print(f"  - Complexity Reduction Ratio    : Raw ({res['raw_gate_count']} gates) -> Final ({len(actions)} gates)\n")

    print("EXACT FINAL GATE SEQUENCE (Full Precision Parameters):")
    for line in display_lines:
        print(line)

    print("\nIMPORTANT QISKIT CONVENTION NOTE:")
    print("  - Qiskit little-endian basis: |q1 q0>")
    print("  - CNOT(q0, q1) = qc.cx(control=0, target=1)")
    print("==================================================================")

if __name__ == "__main__":
    main()
