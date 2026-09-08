import sys
import os
import numpy as np
import torch
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("quantumrl"))

from quantumrl.synthesis import synthesize_circuit, build_qiskit_circuit
from qiskit.quantum_info import Statevector
from quantumrl.simplify import _fidelity

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
    print("TATVA PPO QUANTUM CIRCUIT SYNTHESIS RUN")
    print("==================================================================")
    print(f"Target Statevector (Normalized, Full Precision Float64):")
    for i, c in enumerate(target_sv):
        print(f"  |{i:02b}> : {c.real:+.16f} {c.imag:+.16f}i")
    print(f"Norm: {np.linalg.norm(target_sv):.16f}\n")

    res = synthesize_circuit(
        target_sv=target_sv,
        target_fidelity=0.99,
        max_candidates=50,
        verbose=True
    )

    print("\n==================================================================")
    print("FINAL VERIFIED SYNTHESIS REPORT (EXACTLY ONE FINAL CIRCUIT)")
    print("==================================================================")

    # Pipeline Progression
    print("PIPELINE STAGE METRICS & GATE COUNT REDUCTION:")
    print(f"  Stage 1: Raw PPO Candidate Gate Count   : {res['raw_gate_count']}")
    print(f"  Stage 2: Continuous Angle Refinement    : (L-BFGS-B optimization applied)")
    print(f"  Stage 3: Simplified Gate Count          : {res['final_gate_count']}")
    print(f"  Stage 4: Fresh Qiskit Verification F    : {res['final_fidelity']:.12f}")
    print(f"  Stage 5: Verification Status            : {'PASSED (F >= 0.99)' if res['verified'] else 'FAILED'}\n")

    # Strict single circuit metrics
    actions = res['simplified_actions']
    qc = build_qiskit_circuit(actions, n_qubits=2)
    reconstructed_sv = Statevector.from_instruction(qc).data
    recomputed_fid = _fidelity(target_sv, reconstructed_sv)

    depth = qc.depth()
    cz_cnt = sum(1 for g, q, a in actions if g in ('CZ', 'CNOT', 'SWAP'))

    print("FINAL VERIFIED CIRCUIT SUMMARY:")
    print(f"  - Verification Status           : {'SUCCESS' if recomputed_fid >= 0.99 else 'FAILED'}")
    print(f"  - Freshly Recomputed Fidelity   : {recomputed_fid:.16f}")
    print(f"  - Final Total Gate Count        : {len(actions)}")
    print(f"  - Final Depth                   : {depth}")
    print(f"  - Final Entangling Gate Count   : {cz_cnt}")
    print(f"  - Complexity Reduction Ratio    : Raw ({res['raw_gate_count']} gates) -> Final ({len(actions)} gates)\n")

    print("EXACT FINAL GATE SEQUENCE (Full Precision Parameters):")
    for idx, (gate, qubits, param) in enumerate(actions, 1):
        if param is not None:
            print(f"  {idx:02d}. {gate}(q{qubits}, θ = {param:+.16f})")
        else:
            if isinstance(qubits, tuple) or isinstance(qubits, list):
                q_str = f"q{qubits[0]}, q{qubits[1]}"
            else:
                q_str = f"q{qubits}"
            print(f"  {idx:02d}. {gate}({q_str})")

    print("\nQISKIT VERIFICATION ASSERTION:")
    assert abs(recomputed_fid - res['final_fidelity']) < 1e-12, "Mismatch in Qiskit fidelity verification!"
    assert recomputed_fid >= 0.99, f"Target fidelity threshold not met! F = {recomputed_fid}"
    print(f"  [CONFIRMED] Fresh Qiskit simulation matches output circuit exactly (F = {recomputed_fid:.16f}).")
    print("==================================================================")

if __name__ == "__main__":
    main()
