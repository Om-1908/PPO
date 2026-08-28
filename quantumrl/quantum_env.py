"""
quantum_env.py
--------------
Custom Gymnasium environment: QuantumCircuitEnv — Scaled to 2 Qubits.

The agent progressively applies 1-qubit and 2-qubit quantum gates to a circuit,
and receives reward proportional to the fidelity improvement between the
resulting statevector and a target statevector. A large bonus reward is given
upon exceeding the fidelity threshold.

Qiskit 1.x API is used exclusively:
  - QuantumCircuit for circuit construction
  - Statevector for noiseless simulation (no Aer required)

Observation (size = 4*2^n + 2 = 18 floats for 2 qubits):
  [Re(ψ)×4, Im(ψ)×4, Re(φ)×4, Im(φ)×4, fidelity, step/MAX_STEPS]

Action space (154 discrete actions for 2 qubits):
  - Single-qubit fixed gates (H, X, Y, Z) on 2 qubits = 8 actions
  - Single-qubit rotation gates (RX, RY, RZ) over 24 angles on 2 qubits = 144 actions
  - CNOT gates (control=0 target=1, control=1 target=0) = 2 actions
  Total = 8 + 144 + 2 = 154 actions.

Reward (dense, per-step):
  reward = 10.0 * (current_fidelity - prev_fidelity) - GATE_PENALTY
  + 25.0 bonus when fidelity >= FIDELITY_THRESHOLD
  - 2.0 penalty when fidelity < 0.05 and step > 5
"""

from typing import List, Optional, Tuple
import numpy as np
import gymnasium
from gymnasium import spaces

from qiskit import QuantumCircuit

from utils import compute_fidelity

# Precomputed fixed 2x2 gate matrices
_H_MAT = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
_X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y_MAT = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
_Z_MAT = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _rx_mat(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array(
        [[np.cos(half), -1j * np.sin(half)], [-1j * np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def _ry_mat(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def _rz_mat(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array(
        [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]],
        dtype=np.complex128,
    )


class QuantumCircuitEnv(gymnasium.Env):
    """
    Gymnasium environment for 2-qubit quantum circuit synthesis via RL.

    Observation : flat float32 vector of length 18
                  = [Re(ψ)×4, Im(ψ)×4, Re(φ)×4, Im(φ)×4, fidelity, step/MAX_STEPS]

    Action      : Discrete index into self.action_list (154 actions)

    Reward      : Dense per-step fidelity-improvement signal:
                  10.0 * (fidelity_gain) - GATE_PENALTY
                  + 25.0 bonus on success, - 2.0 stuck penalty.

    Episode ends: fidelity >= threshold (terminated)
                  OR steps >= max_steps (truncated)
    """

    metadata = {'render_modes': ['text']}

    def __init__(self, config, target_sv: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        config    : Config dataclass instance
        target_sv : optional fixed target; overridden in reset() if None
        """
        super().__init__()

        self.n_qubits = config.NUM_QUBITS
        self.max_steps = config.MAX_STEPS
        self.fidelity_threshold = config.FIDELITY_THRESHOLD
        self.gate_penalty = config.GATE_PENALTY
        self.gates: List[str] = config.GATES
        self.rotation_angles: List[float] = config.ROTATION_ANGLES

        # Build discrete action list (154 actions for 2 qubits)
        self.action_list = self._build_action_list()

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(self.action_list))

        # obs_size = 4 * (2 ^ n_qubits) + 2 = 18
        sv_floats = 4 * (2 ** self.n_qubits)
        obs_size = sv_floats + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self._fixed_target_sv: Optional[np.ndarray] = target_sv
        self.target_sv: Optional[np.ndarray] = None
        self.current_sv: Optional[np.ndarray] = None
        self.applied_actions: List[Tuple] = []
        self.steps: int = 0
        self.prev_fidelity: float = 0.0

    @property
    def current_circuit(self) -> QuantumCircuit:
        """Reconstruct Qiskit QuantumCircuit on demand for rendering/export."""
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
            elif gate_name == 'RX':
                circ.rx(angle, qubit_or_pair)
            elif gate_name == 'RY':
                circ.ry(angle, qubit_or_pair)
            elif gate_name == 'RZ':
                circ.rz(angle, qubit_or_pair)
            elif gate_name == 'CNOT':
                ctrl, tgt = qubit_or_pair
                circ.cx(ctrl, tgt)
        return circ

    def _build_action_list(self) -> List[Tuple]:
        """
        Construct enumerated action list for 2-qubit system (154 actions).

        1. Single-qubit non-rotation gates (H, X, Y, Z):
           one action per (gate, qubit_index, None) -> 4 * 2 = 8 actions
        2. Single-qubit rotation gates (RX, RY, RZ):
           one action per (gate, qubit_index, angle) -> 3 * 2 * 24 = 144 actions
        3. CNOT:
           one action per (gate, (control, target), None) -> 2 actions

        Total = 8 + 144 + 2 = 154 actions.
        """
        rotation_gates = {'RX', 'RY', 'RZ'}
        single_qubit_gates = [g for g in self.gates if g != 'CNOT']
        actions = []

        for gate in single_qubit_gates:
            for q in range(self.n_qubits):
                if gate in rotation_gates:
                    for angle in self.rotation_angles:
                        actions.append((gate, q, angle))
                else:
                    actions.append((gate, q, None))

        if 'CNOT' in self.gates and self.n_qubits >= 2:
            for ctrl in range(self.n_qubits):
                for tgt in range(self.n_qubits):
                    if ctrl != tgt:
                        actions.append(('CNOT', (ctrl, tgt), None))

        return actions

    def _apply_1q_gate_numpy(self, gate_mat: np.ndarray, qubit: int) -> None:
        """Apply 2x2 gate matrix to in-memory statevector via tensor contraction."""
        axis = self.n_qubits - 1 - qubit
        shape = [2] * self.n_qubits
        T = self.current_sv.reshape(shape)
        T_new = np.tensordot(gate_mat, T, axes=([1], [axis]))
        axes_order = list(range(1, axis + 1)) + [0] + list(range(axis + 1, self.n_qubits))
        T_new = np.transpose(T_new, axes_order)
        self.current_sv = T_new.reshape(-1)

    def _apply_cnot_numpy(self, ctrl: int, tgt: int) -> None:
        """Apply CNOT by swapping target qubit amplitudes where control qubit is |1⟩."""
        ctrl_axis = self.n_qubits - 1 - ctrl
        tgt_axis = self.n_qubits - 1 - tgt
        shape = [2] * self.n_qubits
        T = self.current_sv.reshape(shape).copy()

        ctrl_slice = [slice(None)] * self.n_qubits
        ctrl_slice[ctrl_axis] = 1

        tgt_slice_0 = list(ctrl_slice)
        tgt_slice_0[tgt_axis] = 0
        tgt_slice_1 = list(ctrl_slice)
        tgt_slice_1[tgt_axis] = 1

        val0 = T[tuple(tgt_slice_0)].copy()
        val1 = T[tuple(tgt_slice_1)].copy()
        T[tuple(tgt_slice_0)] = val1
        T[tuple(tgt_slice_1)] = val0
        self.current_sv = T.reshape(-1)

    def _apply_gate(self, gate_name: str, qubit_or_pair, angle: Optional[float]) -> None:
        """Apply gate incrementally to in-memory statevector."""
        if gate_name == 'H':
            self._apply_1q_gate_numpy(_H_MAT, qubit_or_pair)
        elif gate_name == 'X':
            self._apply_1q_gate_numpy(_X_MAT, qubit_or_pair)
        elif gate_name == 'Y':
            self._apply_1q_gate_numpy(_Y_MAT, qubit_or_pair)
        elif gate_name == 'Z':
            self._apply_1q_gate_numpy(_Z_MAT, qubit_or_pair)
        elif gate_name == 'RX':
            self._apply_1q_gate_numpy(_rx_mat(angle), qubit_or_pair)
        elif gate_name == 'RY':
            self._apply_1q_gate_numpy(_ry_mat(angle), qubit_or_pair)
        elif gate_name == 'RZ':
            self._apply_1q_gate_numpy(_rz_mat(angle), qubit_or_pair)
        elif gate_name == 'CNOT':
            ctrl, tgt = qubit_or_pair
            self._apply_cnot_numpy(ctrl, tgt)
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    def _encode_obs(
        self, current_sv: np.ndarray, target_sv: np.ndarray, fidelity: float, step: int
    ) -> np.ndarray:
        """Build flat 18-float observation vector."""
        obs = np.concatenate([
            current_sv.real.astype(np.float32),
            current_sv.imag.astype(np.float32),
            target_sv.real.astype(np.float32),
            target_sv.imag.astype(np.float32),
            np.array([fidelity, step / self.max_steps], dtype=np.float32),
        ])
        return obs

    def _random_target(self) -> np.ndarray:
        """Generate a Haar-random n-qubit statevector (normalized complex128, dim=2^n)."""
        dim = 2 ** self.n_qubits
        sv = np.random.randn(dim) + 1j * np.random.randn(dim)
        return (sv / np.linalg.norm(sv)).astype(np.complex128)

    def reset(
        self,
        seed: Optional[int] = None,
        target_sv: Optional[np.ndarray] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to ground state |0...0⟩."""
        super().reset(seed=seed)

        if target_sv is not None:
            self.target_sv = target_sv.astype(np.complex128)
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
        init_fidelity = compute_fidelity(self.target_sv, self.current_sv)
        obs = self._encode_obs(self.current_sv, self.target_sv, init_fidelity, 0)
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply gate action and return (obs, reward, terminated, truncated, info)."""
        gate_name, qubit_or_pair, angle = self.action_list[action]

        self.applied_actions.append((gate_name, qubit_or_pair, angle))
        self._apply_gate(gate_name, qubit_or_pair, angle)

        fidelity = compute_fidelity(self.target_sv, self.current_sv)
        self.steps += 1

        fidelity_gain = fidelity - self.prev_fidelity
        reward = 10.0 * fidelity_gain - self.gate_penalty

        if fidelity < 0.05 and self.steps > 5:
            reward -= 2.0

        terminated = False
        if fidelity >= self.fidelity_threshold:
            reward += 25.0   # Enhanced 2-qubit success bonus
            terminated = True

        self.prev_fidelity = fidelity
        truncated = self.steps >= self.max_steps

        obs = self._encode_obs(self.current_sv, self.target_sv, fidelity, self.steps)
        info = {'fidelity': fidelity, 'steps': self.steps}

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Print text representation of the quantum circuit."""
        circ = self.current_circuit
        if circ is not None:
            try:
                print(circ.draw('text'))
            except (UnicodeEncodeError, Exception):
                try:
                    import sys
                    text_draw = str(circ.draw('text'))
                    sys.stdout.buffer.write(text_draw.encode('utf-8'))
                    sys.stdout.buffer.write(b'\n')
                    sys.stdout.flush()
                except Exception:
                    print("[QuantumCircuitEnv] (Circuit diagram rendered with UTF-8 fallback)")
        else:
            print("[QuantumCircuitEnv] Circuit not initialized.")

    def close(self) -> None:
        pass

