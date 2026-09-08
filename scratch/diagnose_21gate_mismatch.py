import sys
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("quantumrl"))

from quantumrl.simplify import simulate_actions, _fidelity
from quantumrl.synthesis import build_qiskit_circuit

# User's target statevector
target_raw = np.array([
    -0.07014476 + 0.17273607j,
    +0.34764850 - 0.77323775j,
    -0.14730561 - 0.18073644j,
    -0.41636934 + 0.13695922j
], dtype=np.complex128)
target_sv = target_raw / np.linalg.norm(target_raw)

# Exact 21-gate actions from report
actions_21 = [
    ('RX', 0, +3.5580922744416537),
    ('RY', 0, +6.0817014898641082),
    ('RX', 1, -0.3824460129132545),
    ('RZ', 0, +5.9864555292526136),
    ('RY', 1, -1.3614586133299744),
    ('RZ', 1, +0.9268265506197396),
    ('RX', 0, +0.3598243788312271),
    ('RX', 1, +2.3858683603737618),
    ('RZ', 1, +2.7364053580096530),
    ('RX', 0, +2.2251454949745266),
    ('RZ', 1, +3.5218037584151118),
    ('RZ', 0, -1.8139468556933003),
    ('RX', 1, +2.8216389513318187),
    ('RZ', 0, +1.6221696298304720),
    ('RY', 0, -2.8904765757689579),
    ('RZ', 1, +3.1777779836201607),
    ('RX', 1, -2.8223528270408131),
    ('RZ', 1, +5.2654837657896456),
    ('RZ', 0, +1.2218599902238811),
    ('CNOT', (0, 1), None),
    ('RZ', 1, +3.0220654568758616)
]

print("--- DIAGNOSING 21-GATE RECONSTRUCTION ---")
print("Target SV:", target_sv)

# 1. NumPy simulation in simplify.py
sv_np = simulate_actions(actions_21, n_qubits=2)
fid_np = _fidelity(target_sv, sv_np)
print(f"NumPy simulate_actions Fidelity: {fid_np:.12f}")

# 2. Qiskit reconstruction via build_qiskit_circuit
qc_qiskit = build_qiskit_circuit(actions_21, n_qubits=2)
sv_qiskit = Statevector.from_instruction(qc_qiskit).data
fid_qiskit = _fidelity(target_sv, sv_qiskit)
print(f"Qiskit build_qiskit_circuit Fidelity: {fid_qiskit:.12f}")

# 3. Standard Qiskit construction manually:
qc_manual = QuantumCircuit(2)
qc_manual.rx(+3.5580922744416537, 0)
qc_manual.ry(+6.0817014898641082, 0)
qc_manual.rx(-0.3824460129132545, 1)
qc_manual.rz(+5.9864555292526136, 0)
qc_manual.ry(-1.3614586133299744, 1)
qc_manual.rz(+0.9268265506197396, 1)
qc_manual.rx(+0.3598243788312271, 0)
qc_manual.rx(+2.3858683603737618, 1)
qc_manual.rz(+2.7364053580096530, 1)
qc_manual.rx(+2.2251454949745266, 0)
qc_manual.rz(+3.5218037584151118, 1)
qc_manual.rz(-1.8139468556933003, 0)
qc_manual.rx(+2.8216389513318187, 1)
qc_manual.rz(+1.6221696298304720, 0)
qc_manual.ry(-2.8904765757689579, 0)
qc_manual.rz(+3.1777779836201607, 1)
qc_manual.rx(-2.8223528270408131, 1)
qc_manual.rz(+5.2654837657896456, 1)
qc_manual.rz(+1.2218599902238811, 0)
qc_manual.cx(0, 1)
qc_manual.rz(+3.0220654568758616, 1)

sv_manual = Statevector.from_instruction(qc_manual).data
fid_manual = _fidelity(target_sv, sv_manual)
print(f"Manual Qiskit (cx 0,1) Fidelity: {fid_manual:.12f}")

# 4. Try CNOT(1, 0)
qc_manual_rev = QuantumCircuit(2)
qc_manual_rev.rx(+3.5580922744416537, 0)
qc_manual_rev.ry(+6.0817014898641082, 0)
qc_manual_rev.rx(-0.3824460129132545, 1)
qc_manual_rev.rz(+5.9864555292526136, 0)
qc_manual_rev.ry(-1.3614586133299744, 1)
qc_manual_rev.rz(+0.9268265506197396, 1)
qc_manual_rev.rx(+0.3598243788312271, 0)
qc_manual_rev.rx(+2.3858683603737618, 1)
qc_manual_rev.rz(+2.7364053580096530, 1)
qc_manual_rev.rx(+2.2251454949745266, 0)
qc_manual_rev.rz(+3.5218037584151118, 1)
qc_manual_rev.rz(-1.8139468556933003, 0)
qc_manual_rev.rx(+2.8216389513318187, 1)
qc_manual_rev.rz(+1.6221696298304720, 0)
qc_manual_rev.ry(-2.8904765757689579, 0)
qc_manual_rev.rz(+3.1777779836201607, 1)
qc_manual_rev.rx(-2.8223528270408131, 1)
qc_manual_rev.rz(+5.2654837657896456, 1)
qc_manual_rev.rz(+1.2218599902238811, 0)
qc_manual_rev.cx(1, 0)
qc_manual_rev.rz(+3.0220654568758616, 1)

sv_manual_rev = Statevector.from_instruction(qc_manual_rev).data
fid_manual_rev = _fidelity(target_sv, sv_manual_rev)
print(f"Manual Qiskit (cx 1,0) Fidelity: {fid_manual_rev:.12f}")
