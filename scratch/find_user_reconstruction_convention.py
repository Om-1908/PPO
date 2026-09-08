import sys
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("quantumrl"))

from quantumrl.simplify import _fidelity

target_raw = np.array([
    -0.07014476 + 0.17273607j,
    +0.34764850 - 0.77323775j,
    -0.14730561 - 0.18073644j,
    -0.41636934 + 0.13695922j
], dtype=np.complex128)
target_sv = target_raw / np.linalg.norm(target_raw)

# 21 gates exact parameters
gates = [
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

def eval_variant(q_swap=False, rev_gates=False, cx_swap=False, sign_flip=False, endian_swap=False):
    qc = QuantumCircuit(2)
    seq = list(reversed(gates)) if rev_gates else list(gates)
    for g, q, a in seq:
        if a is not None and sign_flip:
            a = -a
        if isinstance(q, int):
            q_idx = 1 - q if q_swap else q
        else:
            q_idx = q

        if g == 'RX': qc.rx(a, q_idx)
        elif g == 'RY': qc.ry(a, q_idx)
        elif g == 'RZ': qc.rz(a, q_idx)
        elif g == 'CNOT':
            c, t = (1, 0) if (c_swap := (q_swap ^ cx_swap)) else (0, 1)
            qc.cx(c, t)
    
    sv = Statevector.from_instruction(qc).data
    if endian_swap:
        # Swap |01> and |10> (index 1 and 2)
        sv = np.array([sv[0], sv[2], sv[1], sv[3]])
    fid = _fidelity(target_sv, sv)
    return fid

print("Testing variants to locate F = 0.5582688...")
for q_swap in [False, True]:
    for rev_gates in [False, True]:
        for cx_swap in [False, True]:
            for sign_flip in [False, True]:
                for endian_swap in [False, True]:
                    fid = eval_variant(q_swap, rev_gates, cx_swap, sign_flip, endian_swap)
                    if abs(fid - 0.5582688) < 1e-4:
                        print(f"MATCH FOUND! F={fid:.7f} -> q_swap={q_swap}, rev={rev_gates}, cx_swap={cx_swap}, sign_flip={sign_flip}, endian={endian_swap}")
                    elif fid > 0.99:
                        print(f"High F={fid:.7f} -> q_swap={q_swap}, rev={rev_gates}, cx_swap={cx_swap}, sign_flip={sign_flip}, endian={endian_swap}")
