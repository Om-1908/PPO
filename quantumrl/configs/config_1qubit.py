"""
config_1qubit.py
----------------
Central configuration dataclass for QuantumRL — Scoped to 1 Qubit.
All hyperparameters, paths, and environment settings live here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Single source of truth for 1-Qubit QuantumRL (Dueling DQN + PER / High-Capacity PPO)."""

    # ──────────────────────────────────────────────
    # Environment (1-Qubit Configuration)
    # ──────────────────────────────────────────────
    NUM_QUBITS: int = 1           # Scoped to 1 Qubit
    MAX_STEPS: int = 15           # Maximum gates applied per episode
    FIDELITY_THRESHOLD: float = 0.99   # Target fidelity to declare success
    GATE_PENALTY: float = 0.005  # Per-step penalty

    # ──────────────────────────────────────────────
    # Gate set available to the agent (1-qubit: no 2Q gates)
    # ──────────────────────────────────────────────
    GATES: List[str] = field(
        default_factory=lambda: ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    )

    # 96-angle grid: 64 positive (π/32 … 2π) + 32 negative (-π/32 … -π)
    # Resolution π/32 ≈ 5.625° gives max single-gate fidelity loss < 0.001
    ROTATION_ANGLES: List[float] = field(
        default_factory=lambda: (
            [k * np.pi / 32 for k in range(1, 65)]   # π/32 … 2π  (64 angles)
            + [-k * np.pi / 32 for k in range(1, 33)]  # -π/32 … -π  (32 angles)
        )
    )

    # ──────────────────────────────────────────────
    # DQN Hyperparameters (1-Qubit Scoped + Dueling + PER)
    # ──────────────────────────────────────────────
    DQN_EPISODES: int = 25000
    DQN_BATCH_SIZE: int = 512
    DQN_BUFFER_SIZE: int = 200000
    DQN_LR: float = 0.0003
    DQN_GAMMA: float = 0.995
    DQN_EPSILON_START: float = 1.0
    DQN_EPSILON_END: float = 0.02
    DQN_EPSILON_DECAY: float = 0.99985
    DQN_TARGET_UPDATE_FREQ: int = 5
    DQN_HIDDEN_SIZE: int = 768
    DQN_WARMUP_STEPS: int = 10000
    DQN_UPDATE_FREQ: int = 4

    # Prioritized Experience Replay (PER)
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: float = 50000

    # ──────────────────────────────────────────────
    # PPO Hyperparameters (1-Qubit Scoped)
    # ──────────────────────────────────────────────
    PPO_EPISODES: int = 25000
    PPO_ROLLOUT_STEPS: int = 8192
    PPO_EPOCHS: int = 15
    PPO_MINI_BATCH_SIZE: int = 512
    PPO_LR: float = 3e-4
    PPO_GAMMA: float = 0.995
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_EPSILON: float = 0.2
    PPO_ENTROPY_COEF: float = 0.01
    PPO_VALUE_COEF: float = 1.0
    PPO_MAX_GRAD_NORM: float = 0.5
    PPO_HIDDEN_SIZE: int = 768
    PPO_MODEL_PATH: str = 'saved_models/1qubit/ppo_model.pth'
    PPO_LOG_PATH: str = 'logs/1qubit/ppo_logs.json'
    PPO_PLOT_PATH: str = 'plots/1qubit/ppo_training.png'

    # LR scheduler for PPO
    PPO_LR_DECAY: bool = True
    PPO_LR_MIN: float = 1e-5

    # ──────────────────────────────────────────────
    # Pretraining (supervised warm-start on FULL Haar dataset — 20,000 states)
    # ──────────────────────────────────────────────
    PRETRAIN_ENABLED: bool = True
    PRETRAIN_DATASET_PATH: str = 'haar_1qubit_synthesis.md'   # relative to project root
    PRETRAIN_MAX_STATES: int = 20000      # Use the ENTIRE 20k dataset
    PRETRAIN_EPOCHS: int = 30             # 30 epochs × 40k pairs = solid supervised learning
    PRETRAIN_LR: float = 5e-4            # Slightly lower LR for stable convergence
    PRETRAIN_BATCH_SIZE: int = 512        # Large batch for fast GPU/CPU training
    # Load expert transitions into the DQN PER buffer at episode start
    EXPERT_BUFFER_ENABLED: bool = True
    EXPERT_BUFFER_STATES: int = 5000      # 5k states × 2 gates = 10k transitions in buffer

    # ──────────────────────────────────────────────
    # CNOT forcing (disabled for 1-qubit — no 2Q gates)
    # ──────────────────────────────────────────────
    CNOT_FORCE_UNTIL_EPISODE: int = 0   # Disabled for 1-qubit
    CNOT_FIRST_USE_BONUS: float = 0.0

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    NUM_TEST_STATES: int = 500
    SEED: int = 42

    # ──────────────────────────────────────────────
    # Best-checkpoint tracking
    # ──────────────────────────────────────────────
    BEST_CHECKPOINT_EVAL_INTERVAL: int = 1000
    BEST_CHECKPOINT_EVAL_STATES: int = 50

    LOG_ACTION_HISTOGRAM: bool = True

    # Curriculum learning (disabled)
    CURRICULUM_ENABLED: bool = False
    CURRICULUM_POOL_SIZE: int = 30

    # File paths (qubit-scoped)
    DQN_MODEL_PATH: str = 'saved_models/1qubit/dqn_model.pth'
    LOG_DIR: str = 'logs/1qubit/'
    PLOT_DIR: str = 'plots/1qubit/'

    def __post_init__(self):
        """Ensure qubit-scoped paths if {n} formatting is present."""
        for attr in ('DQN_MODEL_PATH', 'PPO_MODEL_PATH', 'PPO_LOG_PATH',
                     'PPO_PLOT_PATH', 'LOG_DIR', 'PLOT_DIR'):
            val = getattr(self, attr)
            if '{n}' in val:
                setattr(self, attr, val.format(n=self.NUM_QUBITS))
