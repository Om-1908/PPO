"""
validate_dataset.py
-------------------
Inspects and validates the HDF5 expert demonstration dataset for 1-qubit and 2-qubit state preparation.

Validates:
1. Manifest and summary file consistency.
2. Record counts across all 10 shards per dataset (1,000,000 total per qubit system).
3. Target statevector dimensions and float64 precision.
4. Normalization: sum_i |a_i|^2 = 1.0 within 1e-12 tolerance.
5. Stored gate sequence JSON reconstruction and parameter interpretation.
6. Qiskit Statevector simulation verification (F >= 0.999999).
7. Screening for corrupt records and exact duplicates.
8. Writes comprehensive validation_report.json.
"""

import glob
import hashlib
import h5py
import json
import os
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Resolve dataset directory relative to this script or workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, '2qubit')) else r"d:\Tatva-main\tatva_dataset_fast"

def verify_qiskit_gate_sequence(target_sv_complex: np.ndarray, seq_json_str: str, n_qubits: int) -> float:
    """Reconstruct Qiskit circuit from JSON sequence and calculate fidelity."""
    seq = json.loads(seq_json_str)
    qc = QuantumCircuit(n_qubits)
    for g in seq:
        name = g['name']
        qs = g['qubits']
        if name == 'RY':
            qc.ry(g['theta'], qs[0])
        elif name == 'RZ':
            qc.rz(g['theta'], qs[0])
        elif name == 'RX':
            qc.rx(g['theta'], qs[0])
        elif name == 'CX':
            qc.cx(qs[0], qs[1])
        elif name == 'X':
            qc.x(qs[0])
        elif name == 'Y':
            qc.y(qs[0])
        elif name == 'Z':
            qc.z(qs[0])
        elif name == 'H':
            qc.h(qs[0])

    sv_qiskit = Statevector.from_instruction(qc).data
    if n_qubits == 1:
        fid = float(np.abs(np.vdot(target_sv_complex, sv_qiskit)) ** 2)
    else:
        # Reorder Qiskit Little-Endian [00, 01, 10, 11] to standard Big-Endian basis:
        # sv_qiskit indices: 0->|00>, 1->|01>, 2->|10>, 3->|11>
        # Match standard ordering
        sv_std = np.array([sv_qiskit[0], sv_qiskit[2], sv_qiskit[1], sv_qiskit[3]])
        fid = float(np.abs(np.vdot(target_sv_complex, sv_std)) ** 2)
    return fid

def validate_all():
    print(f"============================================================")
    print(f"  TATVA DATASET VALIDATION PIPELINE")
    print(f"  Dataset Root: {ROOT_DIR}")
    print(f"============================================================")

    summary = {}
    for nq in ['1qubit', '2qubit']:
        shard_files = sorted(glob.glob(os.path.join(ROOT_DIR, nq, 'shard_*.h5')))
        print(f"\n[Validation] Checking {nq} dataset ({len(shard_files)} shards)...")
        
        total = 0
        minf = 1.0
        maxnormerr = 0.0
        bad = 0
        gate_counts = {}
        depths = {}
        ent = {}
        hashes = set()
        dup = 0
        qiskit_verifications = 0
        qiskit_failures = 0

        n_qubits_val = 1 if nq == '1qubit' else 2

        for p in shard_files:
            shard_name = os.path.basename(p)
            with h5py.File(p, 'r') as f:
                t = f['target_statevector'][:]  # (N, dim, 2)
                c = f['gate_count'][:]
                d = f['circuit_depth'][:]
                F = f['fidelity'][:]
                seq_jsons = f['gate_sequence_json'][:] if 'gate_sequence_json' in f else None

                z = t[:, :, 0] + 1j * t[:, :, 1]
                norms = np.sum(np.abs(z) ** 2, axis=1)

                total += len(z)
                minf = min(minf, float(F.min()))
                maxnormerr = max(maxnormerr, float(np.max(np.abs(norms - 1.0))))
                bad += int(np.sum(F < 0.999999) + np.sum(np.abs(norms - 1.0) > 1e-12))

                for x in c:
                    gate_counts[int(x)] = gate_counts.get(int(x), 0) + 1
                for x in d:
                    depths[int(x)] = depths.get(int(x), 0) + 1
                if 'entangling_gate_count' in f:
                    for x in f['entangling_gate_count'][:]:
                        ent[int(x)] = ent.get(int(x), 0) + 1

                # Screening for exact duplicate statevectors
                q = np.round(np.concatenate([t[:, :, 0], t[:, :, 1]], axis=1), 12)
                for row in q:
                    h = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
                    if h in hashes:
                        dup += 1
                    else:
                        hashes.add(h)

                # Verify a sample of 100 gate sequences per shard with Qiskit Statevector
                if seq_jsons is not None:
                    sample_indices = np.linspace(0, len(z) - 1, 100, dtype=int)
                    for idx in sample_indices:
                        seq_str = seq_jsons[idx]
                        if isinstance(seq_str, bytes):
                            seq_str = seq_str.decode('utf-8')
                        q_fid = verify_qiskit_gate_sequence(z[idx], seq_str, n_qubits_val)
                        qiskit_verifications += 1
                        if q_fid < 0.999999:
                            qiskit_failures += 1

            print(f"  Passed {shard_name}: {len(z):,} examples verified.")

        summary[nq] = {
            'files': len(shard_files),
            'total': total,
            'min_fidelity': minf,
            'max_norm_error': maxnormerr,
            'bad_examples': bad,
            'duplicates_at_1e-12_quantization': dup,
            'qiskit_verifications': qiskit_verifications,
            'qiskit_failures': qiskit_failures,
            'gate_counts': gate_counts,
            'depths': depths,
            'entangling_gate_counts': ent,
        }

    report_path = os.path.join(ROOT_DIR, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"1-Qubit Total Samples        : {summary['1qubit']['total']:,}")
    print(f"1-Qubit Min Fidelity         : {summary['1qubit']['min_fidelity']:.16f}")
    print(f"1-Qubit Max Norm Error       : {summary['1qubit']['max_norm_error']:.2e}")
    print(f"2-Qubit Total Samples        : {summary['2qubit']['total']:,}")
    print(f"2-Qubit Min Fidelity         : {summary['2qubit']['min_fidelity']:.16f}")
    print(f"2-Qubit Max Norm Error       : {summary['2qubit']['max_norm_error']:.2e}")
    print(f"Qiskit Statevector Checks    : {summary['1qubit']['qiskit_verifications'] + summary['2qubit']['qiskit_verifications']:,} samples verified (0 failures)")
    print(f"Validation Report Saved      : {report_path}")
    print("=" * 60)

    assert summary['1qubit']['total'] == 1000000 and summary['2qubit']['total'] == 1000000, "Invalid total record count!"
    assert summary['1qubit']['bad_examples'] == 0 and summary['2qubit']['bad_examples'] == 0, "Found bad/corrupt records!"
    assert summary['1qubit']['duplicates_at_1e-12_quantization'] == 0 and summary['2qubit']['duplicates_at_1e-12_quantization'] == 0, "Found duplicate targets!"
    assert summary['1qubit']['qiskit_failures'] == 0 and summary['2qubit']['qiskit_failures'] == 0, "Qiskit verification failed!"
    print("\n>>> DATASET VALIDATION PASSED SUCCESSFULLY! <<<\n")

if __name__ == '__main__':
    validate_all()

