import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("quantumrl"))

from quantumrl.synthesis import (
    synthesize_circuit,
    build_qiskit_circuit,
    verify_4way_consistency
)
from qiskit.quantum_info import Statevector

def main():
    target_raw = np.array([
        0.60623367 + 0.03647334j,
        0.40646626 - 0.07116254j,
       -0.47169680 - 0.41325188j,
       -0.07050210 + 0.25025180j
    ], dtype=np.complex128)

    # Normalize statevector strictly
    orig_norm = float(np.linalg.norm(target_raw))
    target_sv = target_raw / orig_norm

    print("==================================================================")
    print("TATVA FINAL PPO QUANTUM CIRCUIT SYNTHESIS ENGINE")
    print("==================================================================")
    print("Target Statevector (Normalized, Full Precision Float64):")
    for i, c in enumerate(target_sv):
        print(f"  |{i:02b}> : {c.real:+.16f} {c.imag:+.16f}i")
    print(f"Original Norm: {orig_norm:.16f} | Normalized Norm: {np.linalg.norm(target_sv):.16f}\n")

    # Run synthesis with strict hierarchical objective (Goal F >= 0.999999)
    res = synthesize_circuit(
        target_sv=target_sv,
        target_fidelity=0.999999,
        max_candidates=50,
        verbose=True
    )

    actions = res['simplified_actions']

    # 4-Way Consistency Check
    opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
        actions,
        target_sv=target_sv,
        n_qubits=2,
        internal_fid=res['internal_fidelity']
    )

    print("\n==================================================================")
    print("4-WAY INTERNAL CONSISTENCY VERIFICATION CHECK:")
    print("==================================================================")
    print(f"  1. Optimizer Objective / Fidelity       : {opt_f:.16f}")
    print(f"  2. Internal Circuit Fidelity            : {int_f:.16f}")
    print(f"  3. Fresh Qiskit Fidelity                : {fresh_qiskit_f:.16f}")
    print(f"  4. Final Displayed Circuit Fidelity     : {disp_f:.16f}")
    print(f"  Max Numerical Discrepancy Across All 4 : {max_dev:.2e}")
    if max_dev < 1e-10:
        print("  -> STATUS: PERFECT 4-WAY CONSISTENCY (100% MATCH)")

    print("\n==================================================================")
    print("FINAL VERIFIED SYNTHESIS REPORT (EXACTLY ONE FINAL CIRCUIT)")
    print("==================================================================")
    print("PIPELINE STAGE METRICS & GATE COUNT REDUCTION:")
    print(f"  Stage 1: Raw PPO Candidate Gate Count   : {res['raw_gate_count']}")
    print(f"  Stage 2: Continuous Angle Refinement    : (L-BFGS-B optimization applied)")
    print(f"  Stage 3: Simplification                : (Rotation merging & identity removal)")
    print(f"  Stage 4: Structural Shorter Search      : Final gate count = {res['final_gate_count']}")
    print(f"  Stage 5: Fresh Qiskit Verification F    : {fresh_qiskit_f:.16f}")
    print(f"  Stage 6: Final Verification Status      : {res['status']}\n")

    print("FINAL VERIFIED CIRCUIT SUMMARY:")
    print(f"  - Status                        : {res['status']}")
    print(f"  - Freshly Recomputed Fidelity   : {fresh_qiskit_f:.16f}")
    print(f"  - Final Total Gate Count        : {res['final_gate_count']}")
    print(f"  - Final Depth                   : {res['circuit_depth']}")
    print(f"  - Final Entangling Gate Count   : {res['entangling_gate_count']}")
    print(f"  - Complexity Reduction Ratio    : Raw ({res['raw_gate_count']} gates) -> Final ({res['final_gate_count']} gates)")
    print(f"  - Search Candidates Evaluated   : {res['candidate_pool_size']}\n")

    print("EXACT FINAL GATE SEQUENCE (Full Precision Parameters):")
    for line in res['display_lines']:
        print(line)

    print("\nQISKIT VERIFICATION ASSERTION:")
    assert max_dev < 1e-10, f"Consistency failure! Max Dev: {max_dev}"
    assert fresh_qiskit_f >= 0.999999, f"Target fidelity threshold not met! F = {fresh_qiskit_f}"
    print(f"  [CONFIRMED] Fresh Qiskit simulation matches output circuit exactly (F = {fresh_qiskit_f:.16f}).")
    print("==================================================================")

if __name__ == "__main__":
    main()
