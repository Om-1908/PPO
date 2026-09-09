import time
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quantumrl'))
from configs.config_2qubit import Config
from quantum_env import QuantumCircuitEnv
from dqn_agent import DQNAgent

def get_action_matrix_4x4(env, action_idx):
    """Compute the exact 4x4 matrix for action_idx by acting on basis vectors."""
    mat = np.zeros((4, 4), dtype=np.complex128)
    gate_name, qubit_or_pair, angle = env.action_list[action_idx]
    for b in range(4):
        sv = np.zeros(4, dtype=np.complex128)
        sv[b] = 1.0
        env.current_sv = sv
        env._apply_gate(gate_name, qubit_or_pair, angle)
        mat[:, b] = env.current_sv
    return mat

def build_all_action_matrices(env):
    n_actions = len(env.action_list)
    mats = np.zeros((n_actions, 4, 4), dtype=np.complex128)
    for i in range(n_actions):
        mats[i] = get_action_matrix_4x4(env, i)
    return torch.tensor(mats, dtype=torch.complex64, device='cuda')

def benchmark():
    cfg = Config()
    env = QuantumCircuitEnv(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    action_mats_gpu = build_all_action_matrices(env)
    print(f"Pre-computed {len(env.action_list)} 4x4 action matrices on CUDA GPU: shape {action_mats_gpu.shape}")
    
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)
    
    # Test CUDA matrix vector state update speed
    t0 = time.time()
    psi = torch.tensor([1.0, 0, 0, 0], dtype=torch.complex64, device='cuda')
    for _ in range(100000):
        a = np.random.randint(0, 164)
        psi = torch.mv(action_mats_gpu[a], psi)
    torch.cuda.synchronize()
    t_gpu_env = time.time() - t0
    print(f"100,000 CUDA GPU statevector updates took: {t_gpu_env:.4f} seconds ({100000/t_gpu_env:.1f} steps/sec)")
    
    # Test NumPy state update speed
    t0 = time.time()
    env.reset()
    for _ in range(100000):
        a = np.random.randint(0, 164)
        env.step(a)
    t_cpu_env = time.time() - t0
    print(f"100,000 NumPy CPU statevector updates took: {t_cpu_env:.4f} seconds ({100000/t_cpu_env:.1f} steps/sec)")

if __name__ == '__main__':
    benchmark()
