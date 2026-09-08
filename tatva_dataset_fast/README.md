# TATVA Training Dataset

- 1,000,000 Haar-random 1-qubit pure states, 10 HDF5 shards.
- 1,000,000 Haar-random 2-qubit pure states, 10 HDF5 shards.
- Total: 2,000,000 examples.
- Seeds: 2026090801 (1q), 2026090802 (2q).
- Target amplitudes are stored as float64 `[real, imag]` pairs.
- State order is Qiskit-compatible little-endian `[00,01,10,11]` for 2 qubits; q0 is the least-significant bit.
- Gate IDs: H=0, X=1, Y=2, Z=3, RX=4, RY=5, RZ=6, CX=7.
- `gate_count` gives the number of active entries at the start of each fixed-width gate array.
- Rotation angles are canonicalized to `[-pi, pi)` where applicable.
- 1q synthesis: canonical RY then RZ state preparation, 2 gates/depth 2.
- 2q synthesis: Schmidt decomposition, RY + one CX(q0,q1), followed by local ZYZ decompositions; local gates are interleaved for depth 5. This uses the minimum possible entangling-gate count (one CX) for generic entangled pure 2-qubit states.
- Every stored example was checked for normalization and fidelity >= 0.999999 by an exact equivalent statevector matrix simulator.

## Important verification limitation

The runtime used to create this artifact did not have Qiskit installed, and package installation was unavailable because the environment had no network access. Therefore the strict requirement to execute `qiskit.quantum_info.Statevector` itself could not be satisfied here. The verification is mathematically equivalent for these gates and state ordering, but it must not be represented as literal Qiskit execution.

See `dataset_manifest.json`, per-part manifests, and `validation_report.json` for audit information.
