"""
quantum_env.py
--------------
Custom Gymnasium environment: QuantumCircuitEnv — Extended for both 1 and 2 Qubits.

The agent progressively applies quantum gates to a circuit, and receives
reward proportional to fidelity improvement between the resulting statevector
and a target statevector.

Qiskit 1.x API is used exclusively (no Aer required).

Observation (size = 4*2^n + 2):
  [Re(ψ)×2^n, Im(ψ)×2^n, Re(φ)×2^n, Im(φ)×2^n, fidelity, step/MAX_STEPS]
  = 10 floats for 1 qubit, 18 floats for 2 qubits.

Action space:
  1-qubit (76 discrete actions):
    Fixed 1Q gates: H, X, Y, Z on q0              = 4 actions   (indices 0–3)
    Rotation 1Q:   RX, RY, RZ × 24 angles × 1 q  = 72 actions  (indices 4–75)

  2-qubit (164 discrete actions):
    Fixed 1Q gates: H, X, Y, Z, S, Sdg, T, Tdg on q0, q1  = 16 actions  (indices 0–15)
    Rotation 1Q:   RX, RY, RZ × 24 angles × 2 qubits       = 144 actions (indices 16–159)
    2-qubit gates: CNOT(0→1), CNOT(1→0), CZ(0,1), SWAP(0,1)= 4 actions  (indices 160–163)

Reward (dense, per-step):
  reward = 10.0 * (current_fidelity - prev_fidelity) - GATE_PENALTY
  + CNOT_FIRST_USE_BONUS first time CNOT/CZ used (2-qubit only, episode < CNOT_FORCE_UNTIL_EPISODE)
  + 25.0 bonus when fidelity >= FIDELITY_THRESHOLD
  - 2.0 penalty when fidelity < 0.05 and step > 5

CNOT Forcing (2-qubit only, episodes < CNOT_FORCE_UNTIL_EPISODE):
  If steps >= MAX_STEPS - 3 and no CNOT/CZ used yet this episode,
  the action is overridden to a random CNOT/CZ action, guaranteeing
  the agent experiences entangling gates in its replay buffer.
  Set env.current_episode before each episode to enable this.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium
from gymnasium import spaces

from qiskit import QuantumCircuit

from utils import compute_fidelity

# ---------------------------------------------------------------------------
# Pre-computed 2×2 gate matrices (complex128)
# ---------------------------------------------------------------------------
_H_MAT   = (1.0 / math.sqrt(2.0)) * np.array([[1., 1.], [1., -1.]], dtype=np.complex128)
_X_MAT   = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
_Y_MAT   = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
_Z_MAT   = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
_S_MAT   = np.array([[1., 0.], [0., 1j]], dtype=np.complex128)        # S  = diag(1, i)
_SDG_MAT = np.array([[1., 0.], [0., -1j]], dtype=np.complex128)       # S† = diag(1, -i)
_T_MAT   = np.array([[1., 0.], [0., math.e ** (1j * math.pi / 4)]], dtype=np.complex128)   # T
_TDG_MAT = np.array([[1., 0.], [0., math.e ** (-1j * math.pi / 4)]], dtype=np.complex128)  # T†

# 4×4 CZ unitary (|00⟩,|01⟩,|10⟩,|11⟩ basis)
_CZ_FULL = np.diag([1., 1., 1., -1.]).astype(np.complex128)

# SWAP unitary
_SWAP_FULL = np.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
], dtype=np.complex128)


def _rx_mat(angle: float) -> np.ndarray:
    h = angle / 2.0
    return np.array([[np.cos(h), -1j * np.sin(h)], [-1j * np.sin(h), np.cos(h)]], dtype=np.complex128)


def _ry_mat(angle: float) -> np.ndarray:
    h = angle / 2.0
    return np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]], dtype=np.complex128)


def _rz_mat(angle: float) -> np.ndarray:
    h = angle / 2.0
    return np.array([[np.exp(-1j * h), 0.], [0., np.exp(1j * h)]], dtype=np.complex128)


# Ordered list of fixed (non-rotation) single-qubit gate names for 1-qubit config
_FIXED_GATES_1Q = ['H', 'X', 'Y', 'Z']
# Ordered list of fixed single-qubit gate names for 2-qubit config
_FIXED_GATES_2Q = ['H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg']
# Names of two-qubit non-rotation gates (for 2-qubit config)
_TWO_QUBIT_GATES = ['CNOT', 'CZ', 'SWAP']
# Rotation gate names
_ROTATION_GATES = {'RX', 'RY', 'RZ'}


class QuantumCircuitEnv(gymnasium.Env):
    """
    Gymnasium environment for n-qubit quantum circuit synthesis via RL.

    Supports 1-qubit (76 actions) and 2-qubit (164 actions) configurations.
    """

    metadata = {'render_modes': ['text']}

    def __init__(self, config, target_sv: Optional[np.ndarray] = None):
        super().__init__()

        self.n_qubits           = config.NUM_QUBITS
        self.max_steps          = config.MAX_STEPS
        self.fidelity_threshold = config.FIDELITY_THRESHOLD
        self.gate_penalty       = config.GATE_PENALTY
        self.gates: List[str]   = config.GATES
        self.rotation_angles: List[float] = config.ROTATION_ANGLES

        # CNOT forcing / bonus (2-qubit training support)
        self.cnot_force_until   = getattr(config, 'CNOT_FORCE_UNTIL_EPISODE', 0)
        self.cnot_first_bonus   = getattr(config, 'CNOT_FIRST_USE_BONUS', 0.0)

        # Current episode counter — set externally by training loop
        self.current_episode: int = 0

        # Build discrete action list
        self.action_list: List[Tuple] = self._build_action_list()

        # Index-map from (gate, qubit_or_pair, snapped_angle) → action index (fast lookup)
        self._action_index_map: Dict[Tuple, int] = {
            a: i for i, a in enumerate(self.action_list)
        }

        # Indices of CNOT/CZ actions (for forcing)
        self._entangling_indices: List[int] = [
            i for i, (g, _, _) in enumerate(self.action_list) if g in ('CNOT', 'CZ', 'SWAP')
        ]

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(self.action_list))
        sv_floats = 4 * (2 ** self.n_qubits)
        obs_size  = sv_floats + 2
        self.observation_space = spaces.Box(low=-1., high=1., shape=(obs_size,), dtype=np.float32)

        # Episode state
        self._fixed_target_sv: Optional[np.ndarray] = target_sv
        self.target_sv: Optional[np.ndarray] = None
        self.current_sv: Optional[np.ndarray] = None
        self.applied_actions: List[Tuple] = []
        self.steps: int = 0
        self.prev_fidelity: float = 0.0
        self._cnot_used_this_episode: bool = False

    # -----------------------------------------------------------------------
    # Action list construction
    # -----------------------------------------------------------------------
    def _build_action_list(self) -> List[Tuple]:
        """
        Build ordered discrete action list.

        1-qubit (76 actions):
          Fixed 1Q: H, X, Y, Z                    →  4 actions
          Rotation: RX, RY, RZ × 24 angles × 1 q  → 72 actions

        2-qubit (164 actions):
          Fixed 1Q: H, X, Y, Z, S, Sdg, T, Tdg × 2 q  → 16 actions
          Rotation: RX, RY, RZ × 24 angles × 2 qubits  → 144 actions
          2Q gates: CNOT(0→1), CNOT(1→0), CZ(0,1), SWAP(0,1) → 4 actions
        """
        actions: List[Tuple] = []

        # ── Single-qubit fixed gates ──────────────────────────────────────
        fixed_gate_names = [g for g in self.gates if g not in _ROTATION_GATES
                            and g not in _TWO_QUBIT_GATES]
        for gate in fixed_gate_names:
            for q in range(self.n_qubits):
                actions.append((gate, q, None))

        # ── Single-qubit rotation gates ───────────────────────────────────
        for gate in self.gates:
            if gate not in _ROTATION_GATES:
                continue
            for q in range(self.n_qubits):
                for angle in self.rotation_angles:
                    actions.append((gate, q, angle))

        # ── Two-qubit gates (n_qubits >= 2 only) ─────────────────────────
        if self.n_qubits >= 2:
            if 'CNOT' in self.gates:
                actions.append(('CNOT', (0, 1), None))
                actions.append(('CNOT', (1, 0), None))
            if 'CZ' in self.gates:
                actions.append(('CZ', (0, 1), None))
            if 'SWAP' in self.gates:
                actions.append(('SWAP', (0, 1), None))

        return actions

    # -----------------------------------------------------------------------
    # Gate application
    # -----------------------------------------------------------------------
    def _apply_1q_gate_numpy(self, gate_mat: np.ndarray, qubit: int) -> None:
        """Apply 2×2 gate matrix to statevector via tensor contraction."""
        axis  = self.n_qubits - 1 - qubit
        shape = [2] * self.n_qubits
        T     = self.current_sv.reshape(shape)
        T_new = np.tensordot(gate_mat, T, axes=([1], [axis]))
        order = list(range(1, axis + 1)) + [0] + list(range(axis + 1, self.n_qubits))
        self.current_sv = np.transpose(T_new, order).reshape(-1)

    def _apply_cnot_numpy(self, ctrl: int, tgt: int) -> None:
        """Apply CNOT by swapping target amplitudes where control is |1⟩."""
        ctrl_ax = self.n_qubits - 1 - ctrl
        tgt_ax  = self.n_qubits - 1 - tgt
        shape   = [2] * self.n_qubits
        T       = self.current_sv.reshape(shape).copy()

        sl_ctrl = [slice(None)] * self.n_qubits
        sl_ctrl[ctrl_ax] = 1
        sl0 = list(sl_ctrl); sl0[tgt_ax] = 0
        sl1 = list(sl_ctrl); sl1[tgt_ax] = 1

        v0 = T[tuple(sl0)].copy()
        T[tuple(sl0)] = T[tuple(sl1)]
        T[tuple(sl1)] = v0
        self.current_sv = T.reshape(-1)

    def _apply_2q_unitary(self, U: np.ndarray) -> None:
        """Apply a 4×4 unitary to the 2-qubit statevector (dim=4)."""
        # Statevector in |00⟩, |01⟩, |10⟩, |11⟩ order (same as np reshape convention
        # for 2 qubits with axis ordering we use: index = q1*2 + q0 for Qiskit convention,
        # but our env uses index = q0*2 + q1 naturally from reshape([2,2]))
        # U is given in standard |00⟩,|01⟩,|10⟩,|11⟩ basis — apply directly.
        self.current_sv = U @ self.current_sv

    def _apply_gate(self, gate_name: str, qubit_or_pair, angle: Optional[float]) -> None:
        """Dispatch gate application to the correct numpy kernel."""
        if gate_name == 'H':
            self._apply_1q_gate_numpy(_H_MAT, qubit_or_pair)
        elif gate_name == 'X':
            self._apply_1q_gate_numpy(_X_MAT, qubit_or_pair)
        elif gate_name == 'Y':
            self._apply_1q_gate_numpy(_Y_MAT, qubit_or_pair)
        elif gate_name == 'Z':
            self._apply_1q_gate_numpy(_Z_MAT, qubit_or_pair)
        elif gate_name == 'S':
            self._apply_1q_gate_numpy(_S_MAT, qubit_or_pair)
        elif gate_name == 'Sdg':
            self._apply_1q_gate_numpy(_SDG_MAT, qubit_or_pair)
        elif gate_name == 'T':
            self._apply_1q_gate_numpy(_T_MAT, qubit_or_pair)
        elif gate_name == 'Tdg':
            self._apply_1q_gate_numpy(_TDG_MAT, qubit_or_pair)
        elif gate_name == 'RX':
            self._apply_1q_gate_numpy(_rx_mat(angle), qubit_or_pair)
        elif gate_name == 'RY':
            self._apply_1q_gate_numpy(_ry_mat(angle), qubit_or_pair)
        elif gate_name == 'RZ':
            self._apply_1q_gate_numpy(_rz_mat(angle), qubit_or_pair)
        elif gate_name == 'CNOT':
            ctrl, tgt = qubit_or_pair
            self._apply_cnot_numpy(ctrl, tgt)
        elif gate_name == 'CZ':
            self._apply_2q_unitary(_CZ_FULL)
        elif gate_name == 'SWAP':
            self._apply_2q_unitary(_SWAP_FULL)
        else:
            raise ValueError(f'Unknown gate: {gate_name}')

    # -----------------------------------------------------------------------
    # Observation encoding
    # -----------------------------------------------------------------------
    def _encode_obs(self, current_sv: np.ndarray, target_sv: np.ndarray,
                    fidelity: float, step: int) -> np.ndarray:
        return np.concatenate([
            current_sv.real.astype(np.float32),
            current_sv.imag.astype(np.float32),
            target_sv.real.astype(np.float32),
            target_sv.imag.astype(np.float32),
            np.array([fidelity, step / self.max_steps], dtype=np.float32),
        ])

    def _random_target(self) -> np.ndarray:
        """Generate a Haar-random n-qubit statevector."""
        dim = 2 ** self.n_qubits
        sv  = np.random.randn(dim) + 1j * np.random.randn(dim)
        return (sv / np.linalg.norm(sv)).astype(np.complex128)

    # -----------------------------------------------------------------------
    # Reset / Step
    # -----------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        target_sv: Optional[np.ndarray] = None,
        target_override: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to ground state |0...0⟩.

        Parameters
        ----------
        target_sv       : explicit target (preferred call-site parameter name)
        target_override : alias for target_sv (used by expert_buffer.py)
        """
        super().reset(seed=seed)

        # Resolve target statevector
        resolved = target_sv if target_sv is not None else target_override
        if resolved is not None:
            sv = np.array(resolved, dtype=np.complex128)
            norm = np.linalg.norm(sv)
            if norm > 1e-10:
                sv = sv / norm
            self.target_sv = sv
        elif self._fixed_target_sv is not None:
            self.target_sv = self._fixed_target_sv.astype(np.complex128)
        else:
            self.target_sv = self._random_target()

        self.applied_actions = []
        dim = 2 ** self.n_qubits
        self.current_sv = np.zeros(dim, dtype=np.complex128)
        self.current_sv[0] = 1.0

        self.steps = 0
        self.prev_fidelity = 0.0
        self._cnot_used_this_episode = False

        init_fidelity = compute_fidelity(self.target_sv, self.current_sv)
        obs = self._encode_obs(self.current_sv, self.target_sv, init_fidelity, 0)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply gate action and return (obs, reward, terminated, truncated, info).

        For 2-qubit episodes < CNOT_FORCE_UNTIL_EPISODE:
          If steps >= MAX_STEPS - 3 and no CNOT/CZ used yet,
          override action to a random entangling gate.
        """
        # ── CNOT forcing ──────────────────────────────────────────────────
        if (
            self.n_qubits >= 2
            and self.cnot_force_until > 0
            and self.current_episode < self.cnot_force_until
            and self.steps >= self.max_steps - 3
            and not self._cnot_used_this_episode
            and len(self._entangling_indices) > 0
        ):
            action = int(np.random.choice(self._entangling_indices))

        gate_name, qubit_or_pair, angle = self.action_list[action]

        self.applied_actions.append((gate_name, qubit_or_pair, angle))
        self._apply_gate(gate_name, qubit_or_pair, angle)

        fidelity = compute_fidelity(self.target_sv, self.current_sv)
        self.steps += 1

        # Hybrid continuous parameter refinement when enabled (disabled during RL step loop for maximum speed)
        if getattr(self, 'enable_inloop_refinement', False) and fidelity >= 0.80 and self.steps >= 3:
            try:
                from simplify import optimize_circuit_parameters, simulate_actions
                opt_actions, opt_fid = optimize_circuit_parameters(
                    self.applied_actions, self.target_sv, self.n_qubits, max_iter=40
                )
                if opt_fid > fidelity:  # Only accept if improvement
                    self.applied_actions = opt_actions
                    self.current_sv = simulate_actions(opt_actions, self.n_qubits)
                    fidelity = opt_fid
            except Exception:
                pass  # Never let optimization crash an episode

        fidelity_gain = fidelity - self.prev_fidelity
        reward = 15.0 * fidelity_gain - self.gate_penalty

        # Milestone crossing bonuses (dense shaping)
        if fidelity >= 0.90 and self.prev_fidelity < 0.90:
            reward += 2.0
        if fidelity >= 0.99 and self.prev_fidelity < 0.99:
            reward += 5.0
        if fidelity >= 0.999 and self.prev_fidelity < 0.999:
            reward += 10.0
        if fidelity >= 0.9999 and self.prev_fidelity < 0.9999:
            reward += 20.0

        # Stuck penalty
        if fidelity < 0.05 and self.steps > 5:
            reward -= 3.0

        # CNOT / CZ first-use bonus (2-qubit only, early training)
        if gate_name in ('CNOT', 'CZ') and not self._cnot_used_this_episode:
            self._cnot_used_this_episode = True
            if (self.n_qubits >= 2
                    and self.current_episode < self.cnot_force_until
                    and self.cnot_first_bonus > 0.0):
                reward += self.cnot_first_bonus

        terminated = False
        if fidelity >= self.fidelity_threshold:
            reward += 60.0  # Strong terminal success reward
            terminated = True

        self.prev_fidelity = fidelity
        truncated = self.steps >= self.max_steps

        obs  = self._encode_obs(self.current_sv, self.target_sv, fidelity, self.steps)
        info = {'fidelity': fidelity, 'steps': self.steps,
                'cnot_used': self._cnot_used_this_episode}

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # gate_tuple_to_action_idx  (used by expert_buffer.py and pretrain.py)
    # -----------------------------------------------------------------------
    def gate_tuple_to_action_idx(self, gate_tuple: Tuple) -> Optional[int]:
        """
        Convert a (gate_name, qubit_or_pair, angle_rad) tuple to an integer
        action index, snapping rotation angles to the nearest grid value.

        Parameters
        ----------
        gate_tuple : (gate_name, qubit_or_pair, angle_rad_or_None)

        Returns
        -------
        int action index, or None if gate not found in action space.
        """
        gate_name, qubit_or_pair, angle = gate_tuple

        if angle is not None:
            # Snap to nearest grid angle
            diffs = np.abs(np.array(self.rotation_angles) - angle)
            snapped = self.rotation_angles[int(np.argmin(diffs))]
            key = (gate_name, qubit_or_pair, snapped)
        else:
            key = (gate_name, qubit_or_pair, None)

        return self._action_index_map.get(key, None)

    # -----------------------------------------------------------------------
    # Circuit reconstruction (for rendering / export)
    # -----------------------------------------------------------------------
    @property
    def current_circuit(self) -> QuantumCircuit:
        """Reconstruct Qiskit QuantumCircuit on demand."""
        circ = QuantumCircuit(self.n_qubits)
        for gate_name, qubit_or_pair, angle in self.applied_actions:
            if gate_name == 'H':
                circ.h(qubit_or_pair)
            elif gate_name == 'X':
                circ.x(qubit_or_pair)
            elif gate_name == 'Y':
                circ.y(qubit_or_pair)
            elif gate_name == 'Z':
                circ.z(qubit_or_pair)
            elif gate_name == 'S':
                circ.s(qubit_or_pair)
            elif gate_name == 'Sdg':
                circ.sdg(qubit_or_pair)
            elif gate_name == 'T':
                circ.t(qubit_or_pair)
            elif gate_name == 'Tdg':
                circ.tdg(qubit_or_pair)
            elif gate_name == 'RX':
                circ.rx(angle, qubit_or_pair)
            elif gate_name == 'RY':
                circ.ry(angle, qubit_or_pair)
            elif gate_name == 'RZ':
                circ.rz(angle, qubit_or_pair)
            elif gate_name == 'CNOT':
                ctrl, tgt = qubit_or_pair
                circ.cx(ctrl, tgt)
            elif gate_name == 'CZ':
                ctrl, tgt = qubit_or_pair
                circ.cz(ctrl, tgt)
            elif gate_name == 'SWAP':
                q0, q1 = qubit_or_pair
                circ.swap(q0, q1)
        return circ

    def render(self) -> None:
        circ = self.current_circuit
        try:
            print(circ.draw('text'))
        except Exception:
            print('[QuantumCircuitEnv] Circuit render fallback.')

    def close(self) -> None:
        pass
