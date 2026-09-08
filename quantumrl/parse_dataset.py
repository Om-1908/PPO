"""
parse_dataset.py
----------------
Parses both Haar markdown synthesis dataset files into lists of
  (target_statevector, action_sequence)
tuples that are directly compatible with the QuantumRL action space.

Datasets
--------
haar_1qubit_synthesis.md  : 20,000 states solved with RY → RZ (2 gates each).
haar_2qubit_synthesis.md  : 100,000 states solved with
    RY_A → CNOT(A,B) → RZ_A → RY_A → RZ_A → RZ_B → RY_B → RZ_B  (8 gates each).

Action format returned
----------------------
Each element of action_sequence is a tuple (gate_name, qubit_or_pair, angle_radians).
  gate_name      : str, e.g. 'RY', 'RZ', 'CNOT'
  qubit_or_pair  : int for 1Q gates, (ctrl, tgt) tuple for CNOT
  angle_radians  : float for rotation gates, None for fixed gates

Angle snapping
--------------
The dataset angles are continuous.  We snap each angle to the nearest value in
the 24-angle grid used by the environment:
  [k * π/8 for k in 1..16]  (positive, π/8 … 2π)
  [-k * π/8 for k in 1..8]  (negative, -π/8 … -π)
This is necessary because the environment action space is discrete.
Snapping gives exact matches for the pretraining supervised signal.
"""

import math
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 96-angle grid (same as configs — MUST stay in sync with ROTATION_ANGLES)
# Resolution: π/32 ≈ 5.625°, max single-gate fidelity quantization loss < 0.001
# ---------------------------------------------------------------------------
_ROTATION_ANGLES: List[float] = (
    [k * math.pi / 32 for k in range(1, 65)]   # π/32 … 2π  (64 positive)
    + [-k * math.pi / 32 for k in range(1, 33)]  # -π/32 … -π  (32 negative)
)
_ANGLE_ARRAY = np.array(_ROTATION_ANGLES, dtype=np.float64)


def _snap_angle(angle_rad: float) -> float:
    """Return the grid angle (radians) closest to angle_rad."""
    diffs = np.abs(_ANGLE_ARRAY - angle_rad)
    return float(_ANGLE_ARRAY[int(np.argmin(diffs))])


# ---------------------------------------------------------------------------
# Complex statevector parser  e.g. "(+0.1882+0.4634j)"
# ---------------------------------------------------------------------------
def _parse_complex(s: str) -> complex:
    """Parse a complex string like '+0.1882+0.4634j' into a Python complex."""
    s = s.strip().lstrip('(').rstrip(')')
    # Python's complex() handles standard notation
    try:
        return complex(s)
    except ValueError:
        # Handle edge-cases like '1.0' with no imaginary part
        return complex(float(s), 0.0)


def _parse_sv_1q(sv_str: str) -> np.ndarray:
    """
    Parse 1-qubit statevector string '[alpha, beta]' from the markdown table.
    Returns complex128 array of length 2.
    """
    # Remove brackets, then split by comma with lookahead to not break complex parts
    inner = sv_str.strip().lstrip('[').rstrip(']')
    # Split at ', ' between amplitudes (not inside an amplitude)
    # Pattern: split at '),  (' or at boundary between amplitude strings
    # The format is: '(+0.1882+0.4634j), (-0.6422+0.5808j)'
    parts = re.split(r'\),\s*\(', inner)
    amps = []
    for p in parts:
        p = p.strip().lstrip('(').rstrip(')')
        amps.append(_parse_complex(p))
    sv = np.array(amps, dtype=np.complex128)
    norm = np.linalg.norm(sv)
    if norm > 1e-10:
        sv /= norm
    return sv


def _parse_sv_2q(sv_str: str) -> np.ndarray:
    """
    Parse 2-qubit statevector string '[a00, a01, a10, a11]'.
    Format: '[-0.4079+0.2197j, +0.0163+0.7408j, +0.1884-0.3761j, +0.0388+0.2404j]'
    Returns complex128 array of length 4.
    """
    inner = sv_str.strip().lstrip('[').rstrip(']')
    # Split on ', ' — the amplitudes don't have parentheses in the 2Q file
    parts = inner.split(', ')
    amps = []
    for p in parts:
        amps.append(_parse_complex(p.strip()))
    sv = np.array(amps, dtype=np.complex128)
    norm = np.linalg.norm(sv)
    if norm > 1e-10:
        sv /= norm
    return sv


# ---------------------------------------------------------------------------
# 1-qubit gate sequence parser
# Format: "RY(+0.667π) → RZ(+0.389π)"
# ---------------------------------------------------------------------------
_1Q_GATE_PATTERN = re.compile(r'(R[XYZ])\(([+-]?\d+\.\d+)π\)')


def _parse_1q_sequence(seq_str: str) -> List[Tuple[str, int, float]]:
    """
    Parse a 1-qubit gate sequence string into action tuples.

    Returns list of (gate_name, qubit=0, angle_rad_snapped).
    The sequence is always on qubit 0 for single-qubit systems.
    """
    gates = _1Q_GATE_PATTERN.findall(seq_str)
    result = []
    for gate_name, angle_pi_str in gates:
        angle_rad = float(angle_pi_str) * math.pi
        snapped = _snap_angle(angle_rad)
        result.append((gate_name, 0, snapped))
    return result


# ---------------------------------------------------------------------------
# 2-qubit gate sequence parser
# Format: "RY(+0.312π)_A → CNOT(A,B) → [RZ(+0.431π) → RY(+0.104π) → RZ(+1.390π)]_A ⊗
#          [RZ(-0.408π) → RY(+0.627π) → RZ(+0.000π)]_B"
# Mapped to 8 gates:
#   [0] RY(mu)_A        qubit=0
#   [1] CNOT(A,B)       ctrl=0, tgt=1
#   [2] RZ(alpha_A)_A   qubit=0
#   [3] RY(theta_A)_A   qubit=0
#   [4] RZ(beta_A)_A    qubit=0
#   [5] RZ(alpha_B)_B   qubit=1
#   [6] RY(theta_B)_B   qubit=1
#   [7] RZ(beta_B)_B    qubit=1
# ---------------------------------------------------------------------------
_2Q_GATE_SEQ_PATTERN = re.compile(
    r'RY\(([+-]?\d+\.\d+)π\)_A'           # 0: RY_A
    r'\s*→\s*CNOT\(A,B\)'                  # 1: CNOT
    r'\s*→\s*\['
    r'RZ\(([+-]?\d+\.\d+)π\)'             # 2: RZ_A
    r'\s*→\s*RY\(([+-]?\d+\.\d+)π\)'     # 3: RY_A
    r'\s*→\s*RZ\(([+-]?\d+\.\d+)π\)'     # 4: RZ_A
    r'\]_A'
    r'\s*⊗\s*\['
    r'RZ\(([+-]?\d+\.\d+)π\)'             # 5: RZ_B
    r'\s*→\s*RY\(([+-]?\d+\.\d+)π\)'     # 6: RY_B
    r'\s*→\s*RZ\(([+-]?\d+\.\d+)π\)'     # 7: RZ_B
    r'\]_B'
)


def _parse_2q_sequence(seq_str: str) -> List[Tuple]:
    """
    Parse a 2-qubit gate sequence string into 8 action tuples.

    Returns:
        [(gate_name, qubit_or_pair, angle_rad_snapped), ...]
    """
    m = _2Q_GATE_SEQ_PATTERN.search(seq_str)
    if m is None:
        return []

    mu_a, rz1_a, ry_a, rz2_a, rz1_b, ry_b, rz2_b = m.groups()

    def snap(s: str) -> float:
        rad = float(s) * math.pi
        return _snap_angle(rad)

    result = [
        ('RY', 0, snap(mu_a)),
        ('CNOT', (0, 1), None),
        ('RZ', 0, snap(rz1_a)),
        ('RY', 0, snap(ry_a)),
        ('RZ', 0, snap(rz2_a)),
        ('RZ', 1, snap(rz1_b)),
        ('RY', 1, snap(ry_b)),
        ('RZ', 1, snap(rz2_b)),
    ]
    return result


# ---------------------------------------------------------------------------
# Public API: load datasets
# ---------------------------------------------------------------------------

def load_1qubit_dataset(
    md_path: str,
    max_states: Optional[int] = None,
) -> List[Tuple[np.ndarray, List[Tuple]]]:
    """
    Parse haar_1qubit_synthesis.md and return a list of
      (target_sv, action_sequence)
    where target_sv is a complex128 numpy array of length 2
    and action_sequence is a list of (gate, qubit, angle) tuples.

    Parameters
    ----------
    md_path   : absolute path to haar_1qubit_synthesis.md
    max_states: if given, only load this many entries (for fast testing)

    Returns
    -------
    List of (np.ndarray shape (2,), List[Tuple]) pairs
    """
    # Pattern for table rows: | #N | `[sv]` | `seq` | 1.0 |
    row_pattern = re.compile(
        r'\|\s*#\d+\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|'
    )
    results = []
    with open(md_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = row_pattern.search(line)
            if m is None:
                continue
            sv_str, seq_str = m.group(1), m.group(2)
            try:
                sv = _parse_sv_1q(sv_str)
                seq = _parse_1q_sequence(seq_str)
            except Exception:
                continue
            if len(seq) == 0:
                continue
            results.append((sv, seq))
            if max_states is not None and len(results) >= max_states:
                break
    return results


def load_2qubit_hdf5_dataset(
    dataset_dir: str = r"d:\Tatva-main\tatva_dataset_fast\2qubit",
    max_states: Optional[int] = None,
) -> List[Tuple[np.ndarray, List[Tuple]]]:
    """
    Load expert demonstrations directly from HDF5 shards in dataset_dir.

    Returns:
        List of (target_sv, action_sequence) tuples, where:
            target_sv: complex128 array of length 4 (unit norm)
            action_sequence: list of (gate_name, qubit_or_pair, angle_rad_snapped) tuples
    """
    import glob
    import json
    import h5py

    shard_files = sorted(glob.glob(os.path.join(dataset_dir, "shard_*.h5")))
    if not shard_files:
        raise FileNotFoundError(f"No HDF5 shard files found in {dataset_dir}")

    results = []
    for shard_path in shard_files:
        with h5py.File(shard_path, 'r') as f:
            t = f['target_statevector'][:]  # shape (N, 4, 2)
            seq_jsons = f['gate_sequence_json'][:]
            
            z = t[:, :, 0] + 1j * t[:, :, 1]
            for i in range(len(z)):
                sv = z[i].astype(np.complex128)
                norm = np.linalg.norm(sv)
                if norm > 1e-10:
                    sv /= norm
                
                seq_str = seq_jsons[i]
                if isinstance(seq_str, bytes):
                    seq_str = seq_str.decode('utf-8')
                
                gate_list = json.loads(seq_str)
                action_seq = []
                for g in gate_list:
                    name = g['name']
                    qs = g['qubits']
                    theta = g.get('theta', None)
                    if name == 'CX':
                        action_seq.append(('CNOT', (qs[0], qs[1]), None))
                    elif name in ('CZ', 'SWAP'):
                        action_seq.append((name, (qs[0], qs[1]), None))
                    elif theta is not None:
                        snapped = _snap_angle(float(theta))
                        action_seq.append((name, qs[0], snapped))
                    else:
                        action_seq.append((name, qs[0], None))
                
                results.append((sv, action_seq))
                if max_states is not None and len(results) >= max_states:
                    return results

    return results


def load_2qubit_dataset(
    md_path: str,
    max_states: Optional[int] = None,
) -> List[Tuple[np.ndarray, List[Tuple]]]:
    """
    Parse 2-qubit expert dataset. First attempts loading from HDF5 shards.
    If HDF5 dataset directory exists, loads from HDF5. Otherwise falls back to markdown.
    """
    # Check if HDF5 directory exists relative to project root or dataset dir
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hdf5_dir = os.path.join(proj_root, 'tatva_dataset_fast', '2qubit')
    if os.path.exists(hdf5_dir):
        return load_2qubit_hdf5_dataset(hdf5_dir, max_states=max_states)

    if not os.path.exists(md_path):
        # Retry with absolute path relative to project root
        md_path = os.path.join(proj_root, md_path)

    row_pattern = re.compile(
        r'\|\s*#\d+\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|'
    )
    results = []
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = row_pattern.search(line)
                if m is None:
                    continue
                sv_str, seq_str = m.group(1), m.group(2)
                try:
                    sv = _parse_sv_2q(sv_str)
                    seq = _parse_2q_sequence(seq_str)
                except Exception:
                    continue
                if len(seq) == 0:
                    continue
                results.append((sv, seq))
                if max_states is not None and len(results) >= max_states:
                    break
    return results


def get_dataset_train_val_split(
    dataset: List[Tuple[np.ndarray, List[Tuple]]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, List[Tuple]]], List[Tuple[np.ndarray, List[Tuple]]]]:
    """
    Split expert dataset into isolated train and validation splits.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    val_size = int(len(dataset) * val_ratio)
    val_indices = set(indices[:val_size])
    
    train_data = [dataset[i] for i in range(len(dataset)) if i not in val_indices]
    val_data = [dataset[i] for i in range(len(dataset)) if i in val_indices]
    return train_data, val_data



# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_1q = os.path.join(_root, 'haar_1qubit_synthesis.md')
    path_2q = os.path.join(_root, 'haar_2qubit_synthesis.md')

    print('Loading 1-qubit dataset (first 5) ...')
    data1 = load_1qubit_dataset(path_1q, max_states=5)
    for sv, seq in data1:
        print(f'  sv={sv[:2]}  seq={seq}')

    print('\nLoading 2-qubit dataset (first 5) ...')
    data2 = load_2qubit_dataset(path_2q, max_states=5)
    for sv, seq in data2:
        print(f'  sv={sv}')
        for g in seq:
            print(f'    {g}')
