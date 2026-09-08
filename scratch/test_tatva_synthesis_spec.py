import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("quantumrl"))

from quantumrl.synthesis import (
    synthesize_circuit,
    build_qiskit_circuit,
    verify_4way_consistency,
    format_circuit_display_lines
)
from qiskit.quantum_info import Statevector

def test_target(target_raw, target_idx, target_fidelity=0.999999):
    norm = np.linalg.norm(target_raw)
    target_sv = target_raw / norm

    print(f"\n==================================================================")
    print(f"BENCHMARK TEST FOR TARGET STATEVECTOR #{target_idx}")
    print(f"Goal: Find F >= {target_fidelity:.6f} -> MINIMIZE GATE COUNT -> MINIMIZE DEPTH")
    print(f"==================================================================")

    res = synthesize_circuit(
        target_sv=target_sv,
        target_fidelity=target_fidelity,
        max_candidates=25,
        verbose=True
    )

    actions = res['simplified_actions']

    # Strict 4-way verification check
    opt_f, int_f, fresh_qiskit_f, disp_f, max_dev = verify_4way_consistency(
        actions,
        target_sv=target_sv,
        n_qubits=2,
        internal_fid=res['internal_fidelity']
    )

    print("\n------------------------------------------------------------------")
    print("4-WAY CONSISTENCY CHECK RESULTS:")
    print(f"  1. Optimizer Fidelity           : {opt_f:.16f}")
    print(f"  2. Internal Circuit Fidelity    : {int_f:.16f}")
    print(f"  3. Fresh Qiskit Fidelity        : {fresh_qiskit_f:.16f}")
    print(f"  4. Final Displayed Output F     : {disp_f:.16f}")
    print(f"  Max Numerical Discrepancy       : {max_dev:.2e}")

    assert max_dev < 1e-10, f"4-Way Consistency Failure! Max Dev: {max_dev}"
    assert res['final_fidelity'] >= target_fidelity, f"Fidelity threshold not met! F = {res['final_fidelity']}"

    print("\nFINAL VERIFIED CIRCUIT SUMMARY:")
    print(f"  - Status                        : {res['status']}")
    print(f"  - Freshly Recomputed Fidelity   : {fresh_qiskit_f:.16f}")
    print(f"  - Final Total Gate Count        : {res['final_gate_count']}")
    print(f"  - Final Circuit Depth           : {res['circuit_depth']}")
    print(f"  - Final Entangling Gate Count   : {res['entangling_gate_count']}")
    print(f"  - Complexity Reduction          : Raw ({res['raw_gate_count']}) -> Final ({res['final_gate_count']})")
    print(f"  - Search Candidates Evaluated   : {res['candidate_pool_size']}")

    print("\nEXACT FINAL GATE SEQUENCE (Full Precision Parameters):")
    for line in res['display_lines']:
        print(line)
    print("==================================================================\n")


def main():
    targets = [
        # Target 1
        np.array([0.314126+0.185742j, -0.421537+0.092861j, 0.337914-0.538276j, 0.484193+0.246815j], dtype=np.complex128),
        # Target 2
        np.array([0.60623367+0.40646626j, -0.47169680-0.07050210j, 0.03647334-0.07116254j, -0.41325188+0.25025180j], dtype=np.complex128),
        # Target 3
        np.array([-0.07014476+0.17273607j, +0.34764850-0.77323775j, -0.14730561-0.18073644j, -0.41636934+0.13695922j], dtype=np.complex128),
    ]

    for idx, target_raw in enumerate(targets, 1):
        test_target(target_raw, idx)

if __name__ == "__main__":
    main()
