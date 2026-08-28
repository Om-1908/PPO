"""
test_quantumrl.py
-----------------
Unit tests for QuantumRL using pytest.
Tests environment, fidelity calculations, DQN agent, PPO agent, and end-to-end pipeline execution.
"""

import os
import pytest
import numpy as np
import torch
from qiskit.quantum_info import Statevector

from config import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent, DuelingQNetwork as QNetwork, PrioritizedReplayBuffer as ReplayBuffer
from ppo_agent import PPOAgent, ActorCritic, RolloutBuffer
from utils import compute_fidelity, generate_random_statevector, generate_target_states, encode_state


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.NUM_QUBITS in (1, 2)
        assert cfg.MAX_STEPS in (15, 20)
        assert cfg.FIDELITY_THRESHOLD == 0.99
        assert 'H' in cfg.GATES
        assert 'X' in cfg.GATES


class TestUtils:
    def test_fidelity_identical(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        fid = compute_fidelity(sv, sv)
        assert pytest.approx(fid, abs=1e-6) == 1.0

    def test_fidelity_orthogonal(self):
        sv1 = np.array([1.0, 0.0], dtype=np.complex128)
        sv2 = np.array([0.0, 1.0], dtype=np.complex128)
        fid = compute_fidelity(sv1, sv2)
        assert pytest.approx(fid, abs=1e-6) == 0.0

    def test_generate_random_statevector(self):
        sv = generate_random_statevector(n_qubits=2, seed=42)
        assert len(sv) == 4
        norm = np.linalg.norm(sv)
        assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_encode_state(self):
        sv1 = np.array([1.0, 0.0], dtype=np.complex128)
        sv2 = np.array([0.0, 1.0], dtype=np.complex128)
        encoded = encode_state(sv1, sv2, fidelity=0.5, step=1, max_steps=15)
        assert encoded.shape == (4 * (2 ** 1) + 2,)
        assert encoded.dtype == np.float32


class TestQuantumEnv:
    def test_env_init_and_reset(self):
        cfg = Config()
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=10)
        env = QuantumCircuitEnv(cfg, target_sv=target_sv)
        
        obs, info = env.reset()
        assert obs.shape == (4 * (2 ** cfg.NUM_QUBITS) + 2,)
        assert env.steps == 0
        assert env.current_circuit is not None

    def test_env_step(self):
        cfg = Config()
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=10)
        env = QuantumCircuitEnv(cfg, target_sv=target_sv)
        env.reset()

        action = 0  # e.g., H gate on qubit 0
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (4 * (2 ** cfg.NUM_QUBITS) + 2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'fidelity' in info
        assert 'steps' in info
        assert info['steps'] == 1


class TestDQNAgent:
    def test_qnetwork(self):
        net = QNetwork(obs_size=10, action_size=7, hidden_size=64)
        x = torch.randn(4, 10)
        q_vals = net(x)
        assert q_vals.shape == (4, 7)

    def test_replay_buffer(self):
        buf = ReplayBuffer(capacity=10)
        s = np.zeros(10, dtype=np.float32)
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

        states, actions, rewards, next_states, dones, weights, indices = buf.sample(1)
        assert states.shape == (1, 10)
        assert actions.shape == (1,)
        assert rewards.shape == (1,)

    def test_agent_action_selection(self):
        cfg = Config()
        agent = DQNAgent(obs_size=10, action_size=7, config=cfg)
        state = np.zeros(10, dtype=np.float32)
        action = agent.select_action(state)
        assert 0 <= action < 7


class TestPPOAgent:
    def test_actor_critic(self):
        ac = ActorCritic(obs_size=10, action_size=7, hidden_size=64)
        x = torch.randn(4, 10)
        logits, val = ac(x)
        probs = torch.softmax(logits, dim=-1)
        assert probs.shape == (4, 7)
        assert val.shape == (4, 1)
        assert pytest.approx(probs.sum(dim=-1).detach().numpy(), abs=1e-5) == np.ones(4)

    def test_agent_update(self):
        cfg = Config(PPO_EPOCHS=2, PPO_MINI_BATCH_SIZE=2)
        device = torch.device('cpu')
        agent = PPOAgent(obs_size=10, action_size=7, config=cfg, device=device)
        buf = RolloutBuffer(rollout_steps=4, obs_size=10, device=device)
        
        obs_t = torch.zeros(10)
        buf.add(obs_t, 0, torch.tensor(-1.0), 1.0, False, torch.tensor(0.5))
        buf.add(obs_t, 1, torch.tensor(-1.0), 0.5, True, torch.tensor(0.2))
        buf.add(obs_t, 0, torch.tensor(-1.0), 0.8, False, torch.tensor(0.4))
        buf.add(obs_t, 1, torch.tensor(-1.0), 0.3, True, torch.tensor(0.1))
        buf.compute_returns_and_advantages(torch.tensor(0.0), gamma=0.99, gae_lambda=0.95)
        
        losses = agent.update(buf)
        assert 'policy_loss' in losses
        assert 'value_loss' in losses


class TestIntegration:
    def test_mini_training_dqn(self):
        cfg = Config(DQN_EPISODES=2, DQN_BATCH_SIZE=2, DQN_BUFFER_SIZE=100)
        env = QuantumCircuitEnv(cfg)
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=1)
        obs, _ = env.reset(target_sv=target_sv)
        
        agent = DQNAgent(obs_size=len(obs), action_size=env.action_space.n, config=cfg)
        
        for _ in range(5):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, float(terminated or truncated))
            agent.update()
            obs = next_obs
            if terminated or truncated:
                break

    def test_mini_training_ppo(self):
        cfg = Config(PPO_EPISODES=2, PPO_ROLLOUT_STEPS=4, PPO_EPOCHS=1)
        device = torch.device('cpu')
        env = QuantumCircuitEnv(cfg)
        target_sv = generate_random_statevector(cfg.NUM_QUBITS, seed=1)
        obs, _ = env.reset(target_sv=target_sv)

        agent = PPOAgent(obs_size=len(obs), action_size=env.action_space.n, config=cfg, device=device)
        
        action, log_prob, entropy, value = agent.select_action(torch.FloatTensor(obs).to(device))
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(action, int)


class TestQuantumRegression:
    """
    Mandatory regression test verifying that the NumPy incremental simulation
    matches Qiskit's reference Statevector to 1e-10 precision across all gate types
    and both CNOT control/target directions.
    """

    def test_1qubit_gate_sequence_regression(self):
        from qiskit import QuantumCircuit

        cfg = Config()
        cfg.NUM_QUBITS = 1
        cfg.GATES = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        env = QuantumCircuitEnv(cfg)
        env.reset()

        a1 = env.rotation_angles[1]
        a2 = env.rotation_angles[3]
        a3 = env.rotation_angles[5]

        # Hand-picked sequence covering every 1-qubit gate type
        actions_to_apply = [
            ('H', 0, None),
            ('X', 0, None),
            ('Y', 0, None),
            ('Z', 0, None),
            ('RX', 0, a1),
            ('RY', 0, a2),
            ('RZ', 0, a3),
        ]

        # 1. Reference output via Qiskit
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.rx(a1, 0)
        qc.ry(a2, 0)
        qc.rz(a3, 0)
        reference_sv = Statevector(qc).data.astype(np.complex128)

        # 2. NumPy incremental output via env
        for gate_name, q, angle in actions_to_apply:
            action_idx = env.action_list.index((gate_name, q, angle))
            env.step(action_idx)

        numpy_sv = env.current_sv

        # 3. Precision assertion
        assert np.allclose(reference_sv, numpy_sv, atol=1e-10), (
            f"1-qubit simulation mismatch:\n"
            f"Qiskit: {reference_sv}\n"
            f"NumPy : {numpy_sv}"
        )

    def test_2qubit_gate_sequence_regression(self):
        from qiskit import QuantumCircuit

        cfg = Config()
        cfg.NUM_QUBITS = 2
        cfg.GATES = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CNOT']
        env = QuantumCircuitEnv(cfg)
        env.reset()

        a1 = env.rotation_angles[1]
        a2 = env.rotation_angles[2]
        a3 = env.rotation_angles[4]
        a4 = env.rotation_angles[6]

        # Hand-picked sequence covering all gate types and BOTH CNOT directions
        actions_to_apply = [
            ('H', 0, None),
            ('X', 1, None),
            ('Y', 0, None),
            ('Z', 1, None),
            ('RX', 0, a1),
            ('RY', 1, a2),
            ('RZ', 0, a3),
            ('CNOT', (0, 1), None),  # CNOT ctrl=0, tgt=1
            ('H', 1, None),
            ('RY', 0, a4),
            ('CNOT', (1, 0), None),  # CNOT ctrl=1, tgt=0
            ('RZ', 1, a1),
        ]

        # 1. Reference output via Qiskit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.y(0)
        qc.z(1)
        qc.rx(a1, 0)
        qc.ry(a2, 1)
        qc.rz(a3, 0)
        qc.cx(0, 1)
        qc.h(1)
        qc.ry(a4, 0)
        qc.cx(1, 0)
        qc.rz(a1, 1)
        reference_sv = Statevector(qc).data.astype(np.complex128)

        # 2. NumPy incremental output via env
        for gate_name, qubit_or_pair, angle in actions_to_apply:
            action_idx = env.action_list.index((gate_name, qubit_or_pair, angle))
            env.step(action_idx)

        numpy_sv = env.current_sv

        # 3. Precision assertion
        assert np.allclose(reference_sv, numpy_sv, atol=1e-10), (
            f"2-qubit simulation mismatch:\n"
            f"Qiskit: {reference_sv}\n"
            f"NumPy : {numpy_sv}"
        )


