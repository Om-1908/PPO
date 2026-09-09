"""
cuda_env.py
-----------
Vectorized Parallel CUDA Quantum Circuit Environment for QuantumRL.
Stepping N=256 parallel environments simultaneously directly inside GPU VRAM via PyTorch.
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from quantum_env import QuantumCircuitEnv

def build_action_matrices_gpu(env) -> torch.Tensor:
    """Pre-compute all discrete 4x4 action unitaries as a CUDA PyTorch tensor of shape (N_actions, 4, 4)."""
    n_actions = len(env.action_list)
    mats = np.zeros((n_actions, 4, 4), dtype=np.complex128)
    for i in range(n_actions):
        gate_name, qubit_or_pair, angle = env.action_list[i]
        for b in range(4):
            sv = np.zeros(4, dtype=np.complex128)
            sv[b] = 1.0
            env.current_sv = sv
            env._apply_gate(gate_name, qubit_or_pair, angle)
            mats[i, :, b] = env.current_sv
    return torch.tensor(mats, dtype=torch.complex64, device='cuda')


class VectorQuantumEnvCUDA:
    """
    Parallel CUDA Vectorized Quantum Environment.
    Simultaneously steps N parallel environments (e.g. N=256) directly in GPU VRAM.
    """

    def __init__(self, config, num_envs: int = 256, device: str = 'cuda'):
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_steps = config.MAX_STEPS
        self.fidelity_threshold = config.FIDELITY_THRESHOLD
        self.gate_penalty = config.GATE_PENALTY

        # Build single CPU env to extract discrete action set
        cpu_env = QuantumCircuitEnv(config)
        self.action_list = cpu_env.action_list
        self.num_actions = len(self.action_list)
        self.action_mats = build_action_matrices_gpu(cpu_env) # shape (164, 4, 4)

        # Environment state tensors on CUDA VRAM
        self.target_sv = torch.zeros((num_envs, 4, 1), dtype=torch.complex64, device=self.device)
        self.current_sv = torch.zeros((num_envs, 4, 1), dtype=torch.complex64, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.prev_fidelity = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    def _sample_random_targets(self, mask=None) -> None:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        count = int(mask.sum().item())
        if count == 0:
            return
        real_part = torch.randn((count, 4, 1), dtype=torch.float32, device=self.device)
        imag_part = torch.randn((count, 4, 1), dtype=torch.float32, device=self.device)
        sv = torch.complex(real_part, imag_part)
        norm = torch.linalg.vector_norm(sv, dim=1, keepdim=True)
        sv = sv / (norm + 1e-10)
        self.target_sv[mask] = sv

    def reset(self, target_svs=None) -> torch.Tensor:
        self.current_sv.zero_()
        self.current_sv[:, 0, 0] = 1.0 + 0j # |00> ground state
        self.steps.zero_()

        if target_svs is not None:
            self.target_sv.copy_(torch.as_tensor(target_svs, dtype=torch.complex64, device=self.device).view(-1, 4, 1))
        else:
            self._sample_random_targets()

        self.prev_fidelity = self._compute_fidelity()
        return self._encode_obs()

    def _compute_fidelity(self) -> torch.Tensor:
        # F = | <target | current> |^2
        target_h = self.target_sv.transpose(1, 2).conj()
        dot = torch.bmm(target_h, self.current_sv).view(-1)
        return torch.abs(dot) ** 2

    def _encode_obs(self) -> torch.Tensor:
        curr_r = self.current_sv.real.squeeze(2)
        curr_i = self.current_sv.imag.squeeze(2)
        tgt_r  = self.target_sv.real.squeeze(2)
        tgt_i  = self.target_sv.imag.squeeze(2)
        fid    = self.prev_fidelity.unsqueeze(1)
        st_ratio = (self.steps.float() / self.max_steps).unsqueeze(1)

        obs = torch.cat([curr_r, curr_i, tgt_r, tgt_i, fid, st_ratio], dim=1)
        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all N parallel environments simultaneously on CUDA GPU.

        Parameters
        ----------
        actions : CUDA LongTensor of shape (num_envs,)

        Returns
        -------
        obs, rewards, dones, info
        """
        self.steps += 1

        # Gather discrete action matrices for all N environments: (N, 4, 4)
        U_batch = self.action_mats[actions]

        # Batch matrix product on CUDA VRAM: (N, 4, 4) @ (N, 4, 1) -> (N, 4, 1)
        self.current_sv = torch.bmm(U_batch, self.current_sv)

        # Compute new fidelities
        curr_fidelity = self._compute_fidelity()

        # Rewards and termination
        rewards = 10.0 * (curr_fidelity - self.prev_fidelity) - self.gate_penalty
        terminated = curr_fidelity >= self.fidelity_threshold
        truncated = self.steps >= self.max_steps
        dones = terminated | truncated

        # Vectorized auto-reset for completed environments
        self.prev_fidelity = curr_fidelity
        obs = self._encode_obs()

        if dones.any():
            reset_mask = dones
            self.current_sv[reset_mask] = 0.0
            self.current_sv[reset_mask, 0, 0] = 1.0 + 0j
            self.steps[reset_mask] = 0
            self._sample_random_targets(reset_mask)
            self.prev_fidelity[reset_mask] = self._compute_fidelity()[reset_mask]

        return obs, rewards, dones, {'fidelity': curr_fidelity}
